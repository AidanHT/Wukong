//! The recognition instrument: `WUKONG_NO_RECOGNIZE` must actually remove the kernel dispatch, and
//! the program must still compute the same answer without it.
//!
//! This is the one gate in the repo that can see a recognizer at all. Every other gate is
//! **gate-blind** to them: a recognizer fires identically on the interpreter and on native at every
//! `-O` level (both call the same `wukong_runtime` symbol), so `native_matches_interpreter` and the
//! opt-invariance suites pass whether the kernel fired or not. The proof has to be the emitted MIR,
//! which is why these tests grep `--emit=mir` output rather than observing behavior.
//!
//! The pair of properties is what makes the instrument trustworthy in both directions: with the
//! variable unset a known kernel *is* dispatched (otherwise "recognition off" would be vacuous),
//! and with it set the same program still runs and prints the same bytes on both backends (otherwise
//! the instrument would be measuring a broken program rather than the generic lowering).

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/wukongc; the corpus lives at <repo>/tests/run.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("run")
        .join(name)
}

/// Run `wukongc` with `args` on `path`, optionally with the instrument set.
/// Returns (exit code, stdout, stderr).
fn wukongc(args: &[&str], path: &Path, no_recognize: bool) -> (Option<i32>, String, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    cmd.args(args).arg(path);
    if no_recognize {
        cmd.env(wukong_driver::NO_RECOGNIZE_ENV, "1");
    } else {
        // The suite must not depend on the ambient environment: a developer with the instrument
        // exported would otherwise see the "recognition on" test fail for the wrong reason.
        cmd.env_remove(wukong_driver::NO_RECOGNIZE_ENV);
    }
    let out = cmd.output().expect("spawn wukongc");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The MIR of a matmul nest contains the tuned GEMM dispatch — the baseline the instrument is
/// measured against. If this ever stops holding, the test below is vacuous.
#[test]
fn a_matmul_nest_dispatches_to_wukong_sgemm() {
    let p = fixture("tensor_matmul.wk");
    let (code, mir, stderr) = wukongc(&["--emit=mir", "-O2"], &p, false);
    assert_eq!(code, Some(0), "compile failed:\n{stderr}");
    assert!(
        mir.contains("call wukong_sgemm"),
        "tensor_matmul.wk no longer dispatches to wukong_sgemm at -O2:\n{mir}"
    );
}

/// With the instrument set, no runtime kernel is dispatched at all — the nest is lowered as written.
#[test]
fn no_recognize_removes_every_kernel_call() {
    let p = fixture("tensor_matmul.wk");
    let (code, mir, stderr) = wukongc(&["--emit=mir", "-O2"], &p, true);
    assert_eq!(code, Some(0), "compile failed:\n{stderr}");
    let calls: Vec<&str> = mir
        .lines()
        .filter(|l| l.contains("call wukong_"))
        .map(str::trim)
        .collect();
    assert!(
        calls.is_empty(),
        "WUKONG_NO_RECOGNIZE left {} kernel call(s) in the MIR: {calls:?}",
        calls.len()
    );
}

/// One fixture per *cascade head*, so a head that stops answering to the switch is caught by name
/// rather than by a vague corpus statistic: `lower_program`'s whole-function interception, the
/// per-loop `lower_for` cascade, the statement-window matchers in `lower_block`, and the elementwise
/// kernel probes inside the vectorizer.
const KERNEL_WITNESSES: &[(&str, &str)] = &[
    ("tensor_matmul.wk", "wukong_sgemm"), // whole-function + lower_for
    ("softmax.wk", "wukong_norm_f32"),    // statement window (the multi-pass norm idiom)
    ("scan_1d.wk", "wukong_cumsum_f32"),  // statement window (flat 1-D scan)
    ("linear_bias_relu.wk", "wukong_sgemm_nt_epi"), // statement window (GEMM + epilogue loop)
    ("transcendental_kernel.wk", "wukong_vmath_f32"), // elementwise probe inside the vectorizer
];

/// Each witness kernel is dispatched by default and gone with the instrument set. Without the first
/// half this test could pass on a fixture that never had a kernel; without the second it would not
/// test the instrument.
#[test]
fn every_cascade_head_answers_to_the_switch() {
    for (name, kernel) in KERNEL_WITNESSES {
        let p = fixture(name);
        let (code, on, stderr) = wukongc(&["--emit=mir", "-O2"], &p, false);
        assert_eq!(code, Some(0), "{name}: compile failed:\n{stderr}");
        assert!(
            on.contains(&format!("call {kernel}")),
            "{name} no longer dispatches {kernel} at -O2, so it cannot witness the switch"
        );
        let (code, off, stderr) = wukongc(&["--emit=mir", "-O2"], &p, true);
        assert_eq!(code, Some(0), "{name}: compile failed:\n{stderr}");
        assert!(
            !off.contains(&format!("call {kernel}")),
            "{name} still dispatches {kernel} with the instrument set"
        );
    }
}

/// Phase 2 `@generic` tiled transpose must lower as written — no transpose recognizer, shared alloca
/// and cooperative barriers in the MIR.
#[test]
fn lang_tiled_transpose_has_no_transpose_kernel() {
    let p = fixture("lang_tiled_transpose.wk");
    let (code, mir, stderr) = wukongc(&["--emit=mir", "-O2"], &p, false);
    assert_eq!(code, Some(0), "compile failed:\n{stderr}");
    assert!(
        !mir.contains("call wukong_transpose"),
        "lang_tiled_transpose.wk should not dispatch to wukong_transpose:\n{mir}"
    );
    assert!(
        mir.contains("alloca.shared"),
        "expected shared alloca in MIR:\n{mir}"
    );
    assert!(mir.contains("barrier"), "expected barrier in MIR:\n{mir}");
}

/// Recognition off must change only *how* the program is compiled: the answer is identical on the
/// interpreter and on native, at -O0 and -O2. A kernel that disagreed with the loop it replaced
/// would show up here as a difference between the two columns.
#[test]
fn no_recognize_keeps_the_program_correct_on_both_backends() {
    for name in [
        "tensor_matmul.wk",
        "saxpy.wk",
        "transcendental_kernel.wk",
        "softmax.wk",
        "scan_1d.wk",
        "linear_bias_relu.wk",
    ] {
        let p = fixture(name);
        for opt in ["-O0", "-O2"] {
            for backend in ["--backend=interp", "--backend=native"] {
                let on = wukongc(&["--run", opt, backend], &p, false);
                let off = wukongc(&["--run", opt, backend], &p, true);
                assert_eq!(
                    (on.0, &on.1),
                    (off.0, &off.1),
                    "{name} {opt} {backend}: recognition off changed the observable behavior\n\
                     stderr(off): {}",
                    off.2
                );
            }
        }
    }
}

/// The `@generic` fixtures each hold the SAME body twice — once with the attribute, once without —
/// so the module's MIR must contain exactly one dispatch of the kernel that body would be recognized
/// as. Zero would mean the attribute leaked into the whole module (or the recognizer stopped firing,
/// making the fixture worthless); two would mean the attribute did nothing. This is the only way to
/// see a per-function opt-out: `run.rs` cannot, since both spellings print the same lanes.
#[test]
fn generic_opts_out_one_function_not_the_module() {
    for (name, kernel) in [
        ("lang_generic_saxpy.wk", "call wukong_velem_f32"),
        ("lang_generic_naive_gemm.wk", "call wukong_sgemm"),
    ] {
        let p = fixture(name);
        for opt in ["-O0", "-O2"] {
            let (code, mir, stderr) = wukongc(&["--emit=mir", opt], &p, false);
            assert_eq!(code, Some(0), "{name}: compile failed:\n{stderr}");
            let n = mir.matches(kernel).count();
            assert_eq!(
                n, 1,
                "{name} at {opt}: expected exactly one `{kernel}` (the un-attributed twin), got {n}"
            );
        }
    }
}

/// Autodiff and the instrument are mutually exclusive, by a named refusal rather than a zero
/// gradient: the VJP rules are keyed on the kernel calls recognition removes.
#[test]
fn autodiff_is_refused_under_no_recognize() {
    let p = fixture("tensor_matmul.wk");
    for args in [vec!["--emit=grad"], vec!["--train", "--train-steps=1"]] {
        let (code, _out, stderr) = wukongc(&args, &p, true);
        assert_eq!(code, Some(1), "{args:?} should have been refused");
        assert!(
            stderr.contains(wukong_driver::NO_RECOGNIZE_ENV),
            "{args:?}: the refusal must name the variable:\n{stderr}"
        );
    }
}
