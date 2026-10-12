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
//!
//! The named tests below pin one fixture per *cascade head*. The corpus sweep at the bottom of the
//! file pins the rest: it walks all of `tests/run`, works out which programs recognition is actually
//! responsible for, and holds each of them to its own generic lowering. Before it existed the ratio
//! was 6 checked out of 184 that depend on a recognizer.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

#[path = "common/mod.rs"]
mod common;

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

// -------------------------------------------------------------------------------------------------
// The recognition on/off output comparator.
//
// The two differentials below (the named cascade-head test and the corpus sweep) compare a
// program's stdout with recognition on against the same program with it off. Historically that
// comparison was byte-exact. That makes the instrument *un-pointable* at the one class of program
// it most needs to cover: with recognition on, a reducing or accumulating kernel (a reassociated
// 8-lane AVX2 fold, a GEMM accumulated per KC block) computes a legitimately different bit pattern
// from the sequential loop the generic lowering emits — the documented `c*sqrt(K)*eps` reduction
// exception. A byte-exact comparator reads that as a miscompile, so it can only be pointed at data
// where every value is exactly representable (which is why the corpus fixtures are built that way,
// tests/CLAUDE.md), and its sensitivity is capped by each fixture's arithmetic rather than by the
// kernel.
//
// `compare_outputs` gives the comparison a numeric mode: bit-exact by default (so sign of zero,
// NaN spelling and every integer stay fully observable, and every entry-free fixture is unchanged),
// and a per-fixture relative tolerance for the fixtures whose kernel reassociates. The tolerance is
// not free-form: it is declared with its derivation in `REASSOCIATION_TOLERANCES`.
// -------------------------------------------------------------------------------------------------

/// Per-fixture relative tolerance for the recognition on/off differential, each with the derivation
/// of its bound. A fixture with no entry is held to **bit-exact** equality — the default, and what
/// every fixture whose kernel can be pointed at exactly-representable data gets.
///
/// This is the extension point the instrument lacked. A fixture that prints raw reduction or GEMM
/// lanes at a measurement size (above the kernel's vector step, where the reassociated and
/// sequential orders genuinely diverge) is added here with its `c*sqrt(K)*eps` bound, and the gate
/// then inspects its mantissas instead of being restricted to data where neutrality is free. Keep
/// each bound cited the way `corpus_lint.rs::DECLARED_TOLERANCES` requires of the run corpus.
///
/// Two derivations are in use, and [`every_reassociation_tolerance_is_derived_from_its_fixtures_own_bound`]
/// re-derives each entry from its fixture.
///
/// APPROX-derived (the default): the fixture pins each printed lane to an exact closed form or an
/// independent f64 reference `e` within its own `EXPECT-OUT-APPROX` relative bound `t` (each `t`
/// cited in `corpus_lint.rs::DECLARED_TOLERANCES`). With recognition on and off both inside that
/// bound — `run.rs` holds the recognized column to it and [`judge_on_off`] holds the recognition-off
/// column to it on every axis of the corpus sweep — `|on - off| <= 2t|e|` and `|off| >= (1-t)|e|`,
/// so relative to the generic lowering (the comparator's reference) they differ by at most
/// `2t/(1-t)`, rounded UP to three significant figures here. That flat bound is the fixture's
/// LOOSEST line's; a lane whose own line is tighter is held to its own line by the per-line check.
/// It is the same bound `wukong_xbench compare`'s premise check applies to the premise fixtures.
///
/// ORDER-derived ([`ORDER_DERIVED_TOLERANCES`]): a fixture whose only on/off difference is the
/// summation order of a reduction may declare the tighter worst-case bound for that order instead.
/// It must still be no looser than its APPROX-derived bound.
const REASSOCIATION_TOLERANCES: &[(&str, f64)] = &[
    // The cov4 fixtures print raw exp/log/sin/cos kernel lanes, where no f32 bit pattern is "the"
    // answer: recognition swaps the kernel's polynomial (and its 8-wide reassociated row folds) for
    // the generic lowering's own transcendentals. t = 1e-3 (the loosest line of each): 2t/(1-t) =
    // 2.002e-3. Measured on/off differences at landing: 2.1e-7 (loss), 1.4e-7 (rope), 1.8e-7
    // (rowkernels_wide) — one to two f32 ULP.
    ("cov4_par_loss.wk", 2.01e-3),
    ("cov4_par_rope.wk", 2.01e-3),
    ("cov4_rowkernels_wide.wk", 2.01e-3),
    // ORDER-derived. The loss `Σ o_i²` over 32 positive f32 terms: recognized, it is one
    // `wukong_sreduce_f32` dot (8-lane FMA partials, then a fixed tree); unrecognized, the
    // sequential source loop. Any summation order of n positive terms is within γ_{n-1}·Σx of the
    // exact sum (Higham, Accuracy and Stability of Numerical Algorithms, §4.2), and each side rounds
    // each product at most once more, so the two orders differ by at most (2·31 + 2)·2^-24 ≈
    // 3.81e-6 of the sum — 4e-6 rounded up. Every `o` lane the fixture prints is the same attention
    // call on both sides and stays byte-exact.
    ("sdpa_loss_train.wk", 4e-6),
    // P12.24. t = 6.11e-5 (c*sqrt(K)*eps at K = 4096): 2t/(1-t) = 1.2221e-4. The @parallel
    // sum-of-squares lane is a recognized 8x8x64 fold with recognition on and the generic
    // vectorizer's order with it off; at landing they printed 4112.017 vs 4112.0166 (9.7e-8).
    ("premise_reduce_4096.wk", 1.23e-4),
    // P12.24. t = 2.65e-5 (c*sqrt(K)*eps at K = 768): 2t/(1-t) = 5.3001e-5. wukong_norm_f32's
    // 8-accumulator sum of squares and `ss * (1/768)` mean vs the loop's order and single divide;
    // bit-identical at landing, which nothing guarantees.
    ("premise_norm_wide.wk", 5.31e-5),
    // P12.24. t = 7.63e-6 (c*sqrt(K)*eps at K = 64): 2t/(1-t) = 1.5260e-5. wukong_sgemm's KC = 384
    // block holds the whole K = 64 contraction today, so both lowerings agree to the bit; a kernel
    // that split K would move last bits within this bound.
    ("premise_gemm_64.wk", 1.53e-5),
];

/// Entries of [`REASSOCIATION_TOLERANCES`] derived from a summation-order bound rather than from
/// their fixture's `EXPECT-OUT-APPROX` lines (see the table's doc).
const ORDER_DERIVED_TOLERANCES: &[&str] = &["sdpa_loss_train.wk"];

/// Every on/off tolerance is DERIVED from its fixture's own `EXPECT-OUT-APPROX` bound, never
/// chosen: the entry must be at least `2t/(1-t)` for the largest `t` the fixture declares (or two
/// lanes each legitimately inside that bound could fail the differential), and at most 1% above it
/// (a looser number would be a tolerance invented to make a fixture pass). A fixture that loses its
/// APPROX lines, or an entry whose fixture is gone, fails here by name.
#[test]
fn every_reassociation_tolerance_is_derived_from_its_fixtures_own_bound() {
    assert!(
        !REASSOCIATION_TOLERANCES.is_empty(),
        "the premise fixtures' entries are missing - the corpus sweep would hold them bit-exact"
    );
    for (name, declared) in REASSOCIATION_TOLERANCES {
        let path = fixture(name);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{name}: declared a tolerance but cannot be read: {e}"));
        let t = src
            .lines()
            .filter_map(|l| l.trim().strip_prefix("// EXPECT-OUT-APPROX:"))
            .map(|rest| {
                let rel = rest.split_whitespace().nth(1).unwrap_or_else(|| {
                    panic!("{name}: EXPECT-OUT-APPROX line without a tolerance: {rest:?}")
                });
                rel.parse::<f64>()
                    .unwrap_or_else(|e| panic!("{name}: unparsable tolerance {rel:?}: {e}"))
            })
            .fold(0.0f64, f64::max);
        assert!(
            t > 0.0,
            "{name} has an on/off tolerance but declares no EXPECT-OUT-APPROX bound to derive it from"
        );
        let derived = 2.0 * t / (1.0 - t);
        assert!(
            *declared >= derived || ORDER_DERIVED_TOLERANCES.contains(name),
            "{name}: declared {declared:e} is below 2t/(1-t) = {derived:e} (t = {t:e})"
        );
        assert!(
            *declared <= derived * 1.01,
            "{name}: declared {declared:e} is more than 1% above 2t/(1-t) = {derived:e} (t = {t:e})"
        );
    }
}

/// The declared tolerance for a fixture, or `None` (bit-exact) when it has no entry.
fn tolerance_for(name: &str) -> Option<f64> {
    REASSOCIATION_TOLERANCES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, t)| *t)
}

/// A fixture's declared on/off tolerance together with its own parsed directives, or `None` when it
/// has no entry (bit-exact). The directives are what [`judge_on_off`] holds the generic column to.
fn declared_for(path: &Path, name: &str) -> Option<(f64, common::Directives)> {
    let tol = tolerance_for(name)?;
    let src = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let d = common::parse_directives(&src, name).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        !d.out.is_empty(),
        "{name} declares a REASSOCIATION_TOLERANCES entry but no EXPECT-OUT* line, so the \
         recognition-off column would be held to nothing but the flat on/off bound"
    );
    Some((tol, d))
}

/// The whole recognition on/off verdict for one program/config pair.
///
/// Without a declared tolerance this is [`compare_outputs`], bit-exact. With one, the recognized
/// (`on`) and generic (`off`) stdout must agree within it, AND the generic stdout must satisfy the
/// fixture's own `EXPECT-OUT*` lines (`common::check_stdout`, the comparator `run.rs` holds the
/// recognized column to).
///
/// The second half is the premise [`REASSOCIATION_TOLERANCES`] is derived from — "both columns
/// satisfy those same lines" — made into an assertion. Nothing else executes the recognition-off
/// column of a corpus program, and the flat on/off bound is sized for the fixture's loosest line
/// (`t = 1e-3`), so without it the generic lowering of a lane whose own line demands `1e-5` could
/// drift ~200x past that contract and stay green.
fn judge_on_off(
    on: &str,
    off: &str,
    declared: Option<(f64, &common::Directives)>,
) -> Result<(), String> {
    compare_outputs(on, off, declared.map(|(t, _)| t))?;
    if let Some((_, d)) = declared {
        common::check_stdout(off, d).map_err(|e| {
            format!(
                "the recognition-off column misses the fixture's own EXPECT-OUT* lines (the \
                 premise its REASSOCIATION_TOLERANCES entry is derived from): {e}"
            )
        })?;
    }
    Ok(())
}

/// `|a - e| <= rel * |e|`, with the three cases the plain formula gets wrong — a NaN never matches,
/// and a zero or infinite reference demands exact equality. Mirrors `common/mod.rs::approx_eq`, the
/// comparator the run corpus already uses for its `EXPECT-OUT-APPROX` lines.
fn approx_eq(a: f64, e: f64, rel: f64) -> bool {
    if a.is_nan() || e.is_nan() {
        return false;
    }
    if e == 0.0 || e.is_infinite() {
        return a == e;
    }
    (a - e).abs() <= rel * e.abs()
}

/// Compare two stdout captures that must agree up to recognition.
///
/// With `tol == None` the bytes must match exactly: this is the default and preserves every
/// observable distinction (sign of zero, NaN spelling, integers). With `tol == Some(rel)`, lines
/// that are not textually equal are compared numerically — both sides must parse as a single `f64`
/// and agree within `rel` relative error (the generic lowering's value is the reference); a
/// non-numeric mismatch, a line-count change, or a NaN is still a failure. Returns the first
/// disagreement.
fn compare_outputs(on: &str, off: &str, tol: Option<f64>) -> Result<(), String> {
    if on == off {
        return Ok(());
    }
    let Some(rel) = tol else {
        return Err(format!(
            "recognition off changed the observable output and no tolerance is declared for this \
             fixture\n    on  = {on:?}\n    off = {off:?}"
        ));
    };
    let on_lines: Vec<&str> = on.lines().collect();
    let off_lines: Vec<&str> = off.lines().collect();
    if on_lines.len() != off_lines.len() {
        return Err(format!(
            "stdout line count differs with recognition off: on={} off={}",
            on_lines.len(),
            off_lines.len()
        ));
    }
    for (i, (a, b)) in on_lines.iter().zip(&off_lines).enumerate() {
        if a == b {
            continue;
        }
        let (Ok(av), Ok(bv)) = (a.trim().parse::<f64>(), b.trim().parse::<f64>()) else {
            return Err(format!(
                "line {}: {a:?} (on) vs {b:?} (off) differ and are not both numeric",
                i + 1
            ));
        };
        // The generic lowering (recognition off) is the language's definition of the answer, so it
        // is the reference the tolerance is relative to.
        if !approx_eq(av, bv, rel) {
            return Err(format!(
                "line {}: {a:?} (on) vs {b:?} (off) exceed the declared relative tolerance {rel:e}",
                i + 1
            ));
        }
    }
    Ok(())
}

/// The comparator preserves bit-exact strictness by default and only relaxes where a tolerance is
/// declared — proven directly so the instrument's own second side cannot go vacuous.
#[test]
fn the_output_comparator_is_strict_by_default_and_bounded_under_tolerance() {
    // Exact match: OK with or without a tolerance.
    assert!(compare_outputs("1\n2\n3\n", "1\n2\n3\n", None).is_ok());
    assert!(compare_outputs("1\n2\n3\n", "1\n2\n3\n", Some(1e-3)).is_ok());
    // Any difference with no declared tolerance is a hard failure (the default every fixture gets).
    assert!(compare_outputs("1\n2\n", "1\n3\n", None).is_err());
    // Sign of zero is observable and the default path preserves it.
    assert!(compare_outputs("-0\n", "0\n", None).is_err());
    // Under a tolerance: within the bound is accepted, outside it is rejected.
    assert!(compare_outputs("1000.001\n", "1000.0\n", Some(1e-5)).is_ok());
    assert!(compare_outputs("1000.5\n", "1000.0\n", Some(1e-5)).is_err());
    // A NaN never matches a finite value, even under a tolerance.
    assert!(compare_outputs("NaN\n", "0\n", Some(1e-3)).is_err());
    // A non-numeric mismatch and a line-count change are failures regardless of tolerance.
    assert!(compare_outputs("hello\n", "world\n", Some(1e-3)).is_err());
    assert!(compare_outputs("1\n2\n", "1\n", Some(1e-3)).is_err());
}

/// A declared on/off tolerance must not become the only thing holding the generic lowering. The
/// flat bound is sized for a fixture's LOOSEST line, so on its own it lets a tightly-specified lane
/// drift far past that lane's own `EXPECT-OUT-APPROX` contract with recognition off; the verdict
/// therefore also holds the generic column to the fixture's own lines.
#[test]
fn a_declared_tolerance_holds_the_generic_column_to_the_fixtures_own_lines() {
    // Two lanes mixing a 1e-5 and a 1e-3 contract, the way cov4_par_loss.wk does.
    let src = "// EXPECT-EXIT: 0\n// EXPECT-OUT-APPROX: -14.348 1e-5\n// EXPECT-OUT-APPROX: 0.4380 1e-3\n";
    let d = common::parse_directives(src, "in-memory probe").expect("probe directives parse");
    let on = "-14.348\n0.438\n";
    // A generic column within both lines (and within the on/off bound): accepted.
    assert!(judge_on_off(on, "-14.34801\n0.4381\n", Some((2.1e-3, &d))).is_ok());
    // A generic column 4.9e-4 relative off on the 1e-5 lane: inside the flat 2.1e-3 on/off bound...
    let drifted = "-14.355\n0.438\n";
    assert!(compare_outputs(on, drifted, Some(2.1e-3)).is_ok());
    // ...but ~50x outside that lane's own contract, which is what the verdict refuses.
    assert!(judge_on_off(on, drifted, Some((2.1e-3, &d))).is_err());
    // A generic column that meets its lines is still held to the on/off bound against the
    // recognized one (here the recognized lane is 9% off).
    assert!(judge_on_off("-14.348\n0.4\n", on, Some((2.1e-3, &d))).is_err());
    // No declared tolerance: bit-exact, sign of zero included, and no directives are consulted.
    assert!(judge_on_off("1\n2\n", "1\n2\n", None).is_ok());
    assert!(judge_on_off("-0\n", "0\n", None).is_err());
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
        "alias_matmul_inplace.wk",
        "alias_gemv_inplace.wk",
        "alias_transpose_inplace.wk",
        "alias_transpose_inplace8.wk",
        "embedding_alias_parallel.wk",
        "parallel_dependence.wk",
    ] {
        let p = fixture(name);
        let declared = declared_for(&p, name);
        for opt in ["-O0", "-O2"] {
            for backend in ["--backend=interp", "--backend=native"] {
                let on = wukongc(&["--run", opt, backend], &p, false);
                let off = wukongc(&["--run", opt, backend], &p, true);
                assert_eq!(on.0, Some(0), "{name} {opt} {backend}: {}", on.2);
                assert_eq!(off.0, Some(0), "{name} {opt} {backend}: {}", off.2);
                // Exit codes agree (both Some(0) above); the stdout comparison goes through the
                // tolerance-aware comparator so a fixture that prints a reassociating kernel's raw
                // lanes can declare its `c*sqrt(K)*eps` bound in REASSOCIATION_TOLERANCES instead of
                // being restricted to data where neutrality is free. With no entry it is bit-exact.
                let declared = declared.as_ref().map(|(t, d)| (*t, d));
                if let Err(why) = judge_on_off(&on.1, &off.1, declared) {
                    panic!(
                        "{name} {opt} {backend}: recognition off changed the observable output\n{why}\n\
                         stderr(off): {}",
                        off.2
                    );
                }
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

// -------------------------------------------------------------------------------------------------
// The corpus sweep.
//
// `run.rs` owns a family of corpus-wide differentials, one per axis along which the compiler may
// vary without changing the answer: `optimization_is_observationally_invariant` walks the
// *optimization* axis, `native_matches_interpreter` the *backend* axis. Neither can see the third
// axis, recognition, and by construction: a recognized kernel fires on the interpreter and on native
// alike, at every `-O` level, because both call the same `wukong_runtime` symbol. That is what
// "recognized kernels are gate-blind" means, and it leaves the substitution itself — a hand-written
// AVX2 kernel standing in for the loop the user actually wrote — as the one rewrite in the pipeline
// with no differential behind it.
//
// `WUKONG_NO_RECOGNIZE` closes that, because it gives the substitution a second side to be compared
// against. These two tests are the axis's entry in the family.
// -------------------------------------------------------------------------------------------------

/// The corpus directory. Non-recursive scans are the house rule: `tests/run/lib` holds imported
/// support modules rather than programs, which is exactly why they live in a subdirectory.
fn run_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("run")
}

/// Every `.wk` program in `tests/run`, sorted, absolute.
fn collect_programs() -> Vec<PathBuf> {
    let dir = run_dir();
    let mut programs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "wk"))
        .collect();
    programs.sort();
    assert!(
        !programs.is_empty(),
        "no .wk programs found in {}",
        dir.display()
    );
    programs
}

/// A scratch working directory, unique to this process and created once.
///
/// The file-I/O fixtures address their blobs by *relative* path, so the runs below need a working
/// directory that is neither the repository (which they would litter) nor one shared with another
/// test binary (where the concurrent sweep in `run.rs` would clobber the same `io_test_*.bin`).
fn scratch_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir()
            .join("wukongc_recognize")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    })
}

/// The `wukong_*` runtime symbols a MIR dump calls.
///
/// Textual, for the same reason the diagnostic-catalogue scan is: the emitted MIR is the only
/// artifact that records a recognizer firing, and `--emit=mir` is the only way to obtain it.
fn kernel_symbols(mir: &str) -> BTreeSet<String> {
    const CALL: &str = "call wukong_";
    let mut out = BTreeSet::new();
    for line in mir.lines() {
        let mut rest = line;
        while let Some(i) = rest.find(CALL) {
            let sym = &rest[i + "call ".len()..];
            let end = sym
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .unwrap_or(sym.len());
            // The bounds- and check-failure edges are safety checks, not recognized kernels.
            if !matches!(
                &sym[..end],
                "wukong_rt_bounds_fail" | "wukong_rt_check_fail"
            ) {
                out.insert(sym[..end].to_string());
            }
            rest = &sym[end..];
        }
    }
    out
}

/// One corpus program's kernel symbols, with recognition on and with it off.
struct Census {
    name: String,
    path: PathBuf,
    on: BTreeSet<String>,
    off: BTreeSet<String>,
}

impl Census {
    /// The symbols recognition is responsible for: dispatched with the instrument unset, gone with
    /// it set.
    ///
    /// Derived rather than listed, so a new recognizer arm joins this gate by existing. Membership
    /// is also the honest test of what "recognition off" means — it removes pattern *recognition*,
    /// not intrinsics, so a program that calls a runtime routine outright keeps it on both sides and
    /// correctly reports nothing here. Three kinds do: `sdpa.wk` names the flash-attention kernel,
    /// every heap program calls `wukong_rt_alloc`, and a `@parallel` region whose `_parallel`
    /// recognizer arm is gone falls through to the generic `wukong_parallel_for` outliner.
    fn recognizer_dependent(&self) -> Vec<&str> {
        self.on.difference(&self.off).map(String::as_str).collect()
    }
}

/// The corpus census, computed once and shared by both tests below.
fn corpus_census() -> &'static Vec<Census> {
    static CENSUS: OnceLock<Vec<Census>> = OnceLock::new();
    CENSUS.get_or_init(|| {
        collect_programs()
            .into_iter()
            .map(|path| {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                let (code, on, err) = wukongc(&["--emit=mir", "-O2"], &path, false);
                assert_eq!(code, Some(0), "{name}: --emit=mir -O2 failed:\n{err}");
                let (code, off, err) = wukongc(&["--emit=mir", "-O2"], &path, true);
                assert_eq!(
                    code,
                    Some(0),
                    "{name}: --emit=mir -O2 failed with the instrument set:\n{err}"
                );
                Census {
                    name,
                    path,
                    on: kernel_symbols(&on),
                    off: kernel_symbols(&off),
                }
            })
            .collect()
    })
}

/// Run a corpus program, returning (exit code, stdout) — the same observable pair `run.rs` compares.
fn run_program(path: &Path, opt: &str, backend: &str, no_recognize: bool) -> (Option<i32>, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    cmd.arg("--run")
        .arg(opt)
        .arg(format!("--backend={backend}"))
        .arg(path)
        .current_dir(scratch_dir());
    if no_recognize {
        cmd.env(wukong_driver::NO_RECOGNIZE_ENV, "1");
    } else {
        cmd.env_remove(wukong_driver::NO_RECOGNIZE_ENV);
    }
    let out = cmd.output().expect("spawn wukongc");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

/// Programs whose *native* half of this differential is withheld, and the open defect that is why.
///
/// This is not a convenience list. Each entry keeps its interpreter comparisons — the
/// recognizer-versus-generic question is still asked, only one axis of it is not — and each names a
/// defect with a reproducer, so an exclusion cannot quietly become permanent. Deleting an entry is
/// how the fix gets its gate.
///
/// Empty. Its one entry, `embedding_alias_parallel.wk`, was withheld for D27: the whole-function
/// `@parallel` vehicle (`parallel_spec`) checked shape only and raced on native when one array was
/// bound to two parameters (11 of 40 recognition-off runs wrong) or when the body carried a
/// dependence. `89ba3103` ("prove whole-function iteration independence before outlining") made
/// that vehicle reuse the region independence proof and fall back to serial otherwise, so the entry
/// was deleted and the program now runs on every axis here.
const NATIVE_AXIS_WITHHELD: &[(&str, &str)] = &[];

/// Every program that depends on a recognizer must compute exactly what its own source loop
/// computes.
///
/// This is the gate the repository did not have. A wrong kernel — one whose AVX2 body disagrees with
/// the loop it replaced, or a recognizer arm that matched a loop it should not have — produces the
/// same wrong answer on both backends at every `-O` level, so it passes `native_matches_interpreter`
/// and both invariance suites without a murmur. Recognition off is the only second opinion, and the
/// interpreter's generic lowering is the language's own definition of the answer.
///
/// Three comparisons per program:
///   * `-O2` on the interpreter — the oracle, and therefore the spec;
///   * `-O2` on native — the path `wukong_xbench` times for its recognition-off column, so a broken
///     generic lowering there would quietly turn every such measurement into the timing of a broken
///     program;
///   * `-O0` on the interpreter — the `mir_build` arms fire at every level, and this is the one
///     where the optimizer has not rewritten around them.
#[test]
fn every_recognizer_dependent_program_agrees_with_its_generic_lowering() {
    let mut mismatches = Vec::new();
    let mut checked = 0usize;
    // The tolerance fixtures this sweep actually reached. An entry it never reaches (the program
    // left the corpus, or stopped depending on a recognizer) is a stale exception, not a check.
    let mut tolerance_checked: BTreeSet<&str> = BTreeSet::new();
    for (name, _) in NATIVE_AXIS_WITHHELD {
        assert!(
            corpus_census().iter().any(|c| &c.name == name),
            "{name} is withheld from the native axis but is no longer in the corpus — \
             re-justify or delete the entry"
        );
    }
    for c in corpus_census() {
        let dependent = c.recognizer_dependent();
        if dependent.is_empty() {
            continue;
        }
        checked += 1;
        let declared = declared_for(&c.path, &c.name);
        if declared.is_some() {
            tolerance_checked.insert(c.name.as_str());
        }
        for (opt, backend) in [("-O2", "interp"), ("-O2", "native"), ("-O0", "interp")] {
            if backend == "native" && NATIVE_AXIS_WITHHELD.iter().any(|(n, _)| *n == c.name) {
                continue;
            }
            let on = run_program(&c.path, opt, backend, false);
            let off = run_program(&c.path, opt, backend, true);
            assert_eq!(
                on.0,
                Some(0),
                "{} @ {opt} {backend}: the corpus program itself failed",
                c.name
            );
            if on.0 != off.0 {
                mismatches.push(format!(
                    "{} @ {opt} {backend} (recognizes {dependent:?}): exit {:?} (on) vs {:?} (off)",
                    c.name, on.0, off.0
                ));
            } else if let Err(why) =
                judge_on_off(&on.1, &off.1, declared.as_ref().map(|(t, d)| (*t, d)))
            {
                mismatches.push(format!(
                    "{} @ {opt} {backend} (recognizes {dependent:?}):\n    {why}",
                    c.name
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "recognition changed the answer for {} program/config pair(s):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    let stale: Vec<&str> = REASSOCIATION_TOLERANCES
        .iter()
        .map(|(n, _)| *n)
        .filter(|n| !tolerance_checked.contains(n))
        .collect();
    assert!(
        stale.is_empty(),
        "REASSOCIATION_TOLERANCES names {stale:?}, which this sweep never compared (not in the \
         corpus, or no longer recognizer-dependent) — re-justify or delete the entry"
    );
    eprintln!(
        "checked {checked} recognizer-dependent program(s); {} under a declared tolerance, each \
         recognition-off column also held to the fixture's own EXPECT-OUT* lines",
        tolerance_checked.len()
    );
    for (name, why) in NATIVE_AXIS_WITHHELD {
        eprintln!("  native axis withheld for {name}: {why}");
    }
}

/// How much of the corpus depends on a recognizer, held as a floor.
///
/// Coverage ratchets only ever move up. The failure this catches is a recognizer that quietly stops
/// firing: the corpus keeps passing, because a program lowered as written still prints the right
/// lanes, and the only visible symptom is that the test above silently has less to check. Pinning
/// both counts turns that from invisible into a failure with a name.
#[test]
fn the_recognizer_dependent_corpus_holds_its_floor() {
    /// Corpus programs whose MIR loses a `wukong_*` call when the instrument is set. Measured: 184
    /// when this floor was introduced, 241 (of 517) when the `cov4_*` coverage fixtures landed.
    const PROGRAM_FLOOR: usize = 241;
    /// Distinct kernels those programs reach between them. Measured: 91, then 141 with `cov4_*`.
    const KERNEL_FLOOR: usize = 141;

    let census = corpus_census();
    let programs = census
        .iter()
        .filter(|c| !c.recognizer_dependent().is_empty())
        .count();
    let kernels: BTreeSet<&str> = census
        .iter()
        .flat_map(Census::recognizer_dependent)
        .collect();

    assert!(
        programs >= PROGRAM_FLOOR,
        "only {programs} of {} corpus programs still depend on a recognizer (floor {PROGRAM_FLOOR}) \
         — a recognizer stopped firing",
        census.len()
    );
    assert!(
        kernels.len() >= KERNEL_FLOOR,
        "only {} distinct kernels are still reached by the corpus (floor {KERNEL_FLOOR})",
        kernels.len()
    );
    eprintln!(
        "{programs}/{} programs depend on a recognizer; {} distinct kernels",
        census.len(),
        kernels.len()
    );
}

// -------------------------------------------------------------------------------------------------
// The emittable-vs-emitted census (P12.4).
//
// `wukong_mir_build` interns a large set of `wukong_*` runtime symbols it CAN lower a nest to. Only
// some of them fired anywhere in `tests/run`; the rest were live emit sites with no corpus witness —
// and the campaign's audit found the uncovered set was dominated by the `_parallel` twins, whose only
// difference from their covered serial form is chunking and pool routing (so a `_parallel` kernel can
// be wrong in a way its serial twin is not, with nothing to catch it). A `_parallel` kernel is also
// the arm the autodiff mirror rule is weakest on.
//
// Until wave 4 this census only RECORDED the never-emitted set (46 `_parallel` twins when it was
// last printed, every one of them uncovered). The `tests/run/cov4_par_*.wk` fixtures now give each
// of those symbols — and the five decode-builtin twins the old single-file scan could not see — a
// `@parallel` program that emits it and prints every lane against hand-derived values, so the census
// is now a GATE: an emittable symbol that no `tests/run` program emits fails here by name, unless it
// is listed in `NEVER_EMITTED_ALLOWLIST` with the reason it cannot be witnessed. Scope: `tests/run`
// only, deliberately. A kernel exercised only by `examples/` or `bench/kernels` has no hand-derived
// every-lane expectation behind it, which is exactly the coverage this gate exists to demand.
// -------------------------------------------------------------------------------------------------

/// Runtime symbols that are not recognized kernels: the timer and the generic `@parallel` outliner,
/// plus the whole `wukong_rt_*` family (allocator, free, and the file-I/O read/write and trig
/// intrinsics). Excluded from both sides so the census counts recognized kernels, not plumbing.
fn is_support(sym: &str) -> bool {
    sym.starts_with("wukong_rt_") || sym == "wukong_now_ns" || sym == "wukong_parallel_for"
}

/// Emittable kernels the `tests/run` corpus is allowed NOT to emit, each with the reason it cannot be
/// witnessed by a fixture. Empty: every emittable symbol has a witness. An entry is a written-down
/// exception, not a to-do list — it must name why no program can reach the symbol (a target the
/// test host lacks, a builtin with no surface spelling yet), and the gate below fails on an entry
/// that has gone stale (the symbol is no longer emittable, or a fixture now emits it).
const NEVER_EMITTED_ALLOWLIST: &[(&str, &str)] = &[];

/// Every `wukong_*` symbol `wukong_mir_build` interns, extracted textually from every source file of
/// the crate (the same discipline as `runtime_surface_tests.rs` and the diagnostic-catalogue scan: the
/// intern sites are string literals, so a textual pass is faithful and needs no compile). It used to
/// read `lib.rs` alone, which silently dropped the decode builtins' intern table
/// (`decode_builtins.rs`, five `_parallel` twins among them) from the emittable surface.
fn emittable_runtime_symbols() -> BTreeSet<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("wukong_mir_build")
        .join("src");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    files.sort();
    assert!(
        files.iter().any(|p| p.ends_with("lib.rs")),
        "the emittable scan no longer finds wukong_mir_build/src/lib.rs in {}",
        dir.display()
    );
    const NEEDLE: &str = ".intern(\"wukong_";
    let mut out = BTreeSet::new();
    for path in &files {
        let src = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let mut rest = src.as_str();
        while let Some(i) = rest.find(NEEDLE) {
            // Position at the start of the symbol name (`wukong_...`), inside the string literal.
            let sym_start = i + ".intern(\"".len();
            let tail = &rest[sym_start..];
            // The literal ends at the closing quote.
            let end = tail.find('"').unwrap_or(tail.len());
            let sym = &tail[..end];
            if !is_support(sym) {
                out.insert(sym.to_string());
            }
            rest = &tail[end..];
        }
    }
    out
}

/// The emittable / emitted split: floor both surfaces, and FAIL when an emittable kernel has no
/// `tests/run` witness and no allowlisted reason. The never-emitted set is printed either way.
#[test]
fn the_emittable_runtime_surface_census_is_recorded_and_floored() {
    /// `wukong_mir_build` still interns a large kernel surface. Measured: 158 when the whole-crate
    /// scan landed. A drop below this means a family of emit sites was deleted — a reviewed change.
    const EMITTABLE_FLOOR: usize = 158;
    /// Distinct kernels the `tests/run` corpus actually emits. Measured: 158 (every emittable
    /// symbol) when the `cov4_*` fixtures landed.
    const EMITTED_FLOOR: usize = 158;

    let emittable = emittable_runtime_symbols();
    let emitted: BTreeSet<String> = corpus_census()
        .iter()
        .flat_map(|c| c.on.iter().cloned())
        .filter(|s| !is_support(s))
        .collect();

    // The emittable floor goes first: it is the vacuity guard for everything below (a scan that
    // stopped matching would make the never-emitted set empty and the gate pass on nothing).
    assert!(
        emittable.len() >= EMITTABLE_FLOOR,
        "wukong_mir_build now interns only {} runtime kernel symbols (floor {EMITTABLE_FLOOR}) — a \
         family of emit sites was removed",
        emittable.len()
    );

    let never: Vec<&String> = emittable.difference(&emitted).collect();
    let (parallel, serial): (Vec<&&String>, Vec<&&String>) =
        never.iter().partition(|s| s.ends_with("_parallel"));
    eprintln!(
        "emittable-vs-emitted census: {} emittable kernels, {} emitted by tests/run, {} never \
         emitted here ({} _parallel twins, {} serial), {} allowlisted",
        emittable.len(),
        emitted.len(),
        never.len(),
        parallel.len(),
        serial.len(),
        NEVER_EMITTED_ALLOWLIST.len(),
    );
    eprintln!("  never-emitted _parallel twins: {parallel:?}");
    eprintln!("  never-emitted serial: {serial:?}");

    let mut problems = Vec::new();
    for (sym, why) in NEVER_EMITTED_ALLOWLIST {
        if why.trim().len() < 20 {
            problems.push(format!(
                "allowlist entry {sym} has no real reason ({why:?}); say why no fixture can emit it"
            ));
        }
        if !emittable.contains(*sym) {
            problems.push(format!(
                "allowlist entry {sym} is no longer interned by wukong_mir_build — delete the entry"
            ));
        } else if emitted.contains(*sym) {
            let by: Vec<&str> = corpus_census()
                .iter()
                .filter(|c| c.on.contains(*sym))
                .map(|c| c.name.as_str())
                .collect();
            problems.push(format!(
                "allowlist entry {sym} is now emitted by {by:?} — delete the entry so the witness \
                 is held to it"
            ));
        }
    }
    for sym in &never {
        if !NEVER_EMITTED_ALLOWLIST.iter().any(|(s, _)| s == sym) {
            problems.push(format!(
                "{sym} is emittable (interned by wukong_mir_build) but no tests/run program emits \
                 it at --emit=mir -O2"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "the emittable runtime surface has {} uncovered or stale entr(ies):\n  {}\n\nEvery kernel the \
         compiler can emit needs a tests/run witness that prints every lane against hand-derived \
         values (a `_parallel` twin: a `@parallel` function whose recognized arm emits it, as in \
         tests/run/cov4_par_*.wk). Add one, or, if no program can reach the symbol, list it in \
         NEVER_EMITTED_ALLOWLIST with the reason.",
        problems.len(),
        problems.join("\n  ")
    );
    // After the named gate, so a lost witness is reported by symbol rather than as a bare count.
    assert!(
        emitted.len() >= EMITTED_FLOOR,
        "the tests/run corpus now emits only {} distinct runtime kernels (floor {EMITTED_FLOOR})",
        emitted.len()
    );
}
