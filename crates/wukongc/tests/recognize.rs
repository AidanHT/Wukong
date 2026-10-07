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
        "alias_matmul_inplace.wk",
        "alias_gemv_inplace.wk",
        "alias_transpose_inplace.wk",
        "alias_transpose_inplace8.wk",
        "embedding_alias_parallel.wk",
        "parallel_dependence.wk",
    ] {
        let p = fixture(name);
        for opt in ["-O0", "-O2"] {
            for backend in ["--backend=interp", "--backend=native"] {
                let on = wukongc(&["--run", opt, backend], &p, false);
                let off = wukongc(&["--run", opt, backend], &p, true);
                assert_eq!(on.0, Some(0), "{name} {opt} {backend}: {}", on.2);
                assert_eq!(off.0, Some(0), "{name} {opt} {backend}: {}", off.2);
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
            // The slice bounds-failure edge is a safety check, not a recognized kernel.
            if &sym[..end] != "wukong_rt_bounds_fail" {
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
            if on != off {
                mismatches.push(format!(
                    "{} @ {opt} {backend} (recognizes {dependent:?}):\n    on  = {on:?}\n    off = {off:?}",
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
    eprintln!("checked {checked} recognizer-dependent program(s)");
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
    /// Corpus programs whose MIR loses a `wukong_*` call when the instrument is set. Measured.
    const PROGRAM_FLOOR: usize = 184;
    /// Distinct kernels those programs reach between them.
    const KERNEL_FLOOR: usize = 91;

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
