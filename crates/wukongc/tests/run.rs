//! End-to-end CLI tests: compile and run each `tests/run/*.wk` program through the real
//! `wukongc` binary and check its stdout and exit code against directives embedded in the file.
//!
//! The directive grammar and the stdout comparator live in `common/mod.rs`, shared with
//! `corpus_lint.rs`, so there is exactly one reader of a fixture's expectations. See that file
//! for the grammar table.
//!
//! This exercises the entire pipeline (lex -> parse -> sema -> MIR -> opt -> interpret) exactly as
//! a user would, with zero external dependencies.
//!
//! ## The floor, and why it exists
//!
//! `check_program` **requires** every fixture to declare `EXPECT-EXIT` and to say something about
//! stdout. Both checks used to be conditional on their directive being present, so a fixture that
//! declared neither was executed and asserted against nothing:
//! `tests/run/gemm_fused_bias_slice.wk` printed 36 unchecked numbers for its whole life and was
//! counted as coverage. `tests/CLAUDE.md` rule 6 already called that vacuous; nothing enforced it.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "common/mod.rs"]
mod common;

/// Floor on the number of collected fixtures.
///
/// `collect_programs` used to assert only `!programs.is_empty()`, so renaming a fixture to
/// `foo.wk.bak` — or moving it into `lib/`, which is deliberately never scanned — removed it from
/// this suite, both invariance gates and the native differential with no failure anywhere. That is
/// the cheapest way in the repo to turn a red gate green. Adding fixtures never trips this; only
/// losing one does. **Raise it in the same commit that adds fixtures.**
///
/// 517: the measured top-level count once the eighteen `cov4_*` coverage fixtures joined the 499
/// on main (the floor had been left at 390 while the tree grew past it, 109 fixtures of slack).
/// 571: the measured count once `autodiff_train_parallel_region.wk` joined (gap6/autodiff-parallel;
/// the floor had again been left behind, at 517 under 570).
/// 614: the measured count once `assert_message.wk` and `match_no_arm_trap.wk` joined
/// (gap7/diag-locations).
/// 632: the measured count once the six `spelling_act_*.wk` natural-activation fixtures joined
/// (gap7/recognizer-spellings; the floor had been left at 614 under 626).
const MIN_FIXTURES: usize = 632;

/// Monotonic counter handing every spawned `wukongc` a distinct scratch directory.
static WORK_SEQ: AtomicU64 = AtomicU64::new(0);

/// A unique scratch working directory for a single `wukongc` execution. Fixtures that touch the
/// filesystem through *relative* paths — the file-I/O round-trip tests write `io_test_*.bin` in the
/// process CWD — otherwise collide when the four test functions below run concurrently (Cargo runs
/// each `#[test]` on its own thread), producing nondeterministic clobbered blobs. Anchoring each run
/// in its own directory removes the shared name entirely. The fixture `.wk` paths are absolute, so
/// changing CWD never affects locating the program itself.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> ScratchDir {
        let seq = WORK_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir()
            .join("wukongc_run_tests")
            .join(format!("{}_{seq}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        ScratchDir(dir)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        // Best-effort cleanup; a lingering temp dir must never fail a test.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/wukongc; the suite lives at <repo>/tests/run.
    common::repo_root().join("tests").join("run")
}

fn check_program(path: &Path) {
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    let src = std::fs::read_to_string(path).unwrap();
    let expect = common::parse_directives(&src, &name).unwrap_or_else(|e| panic!("{e}"));

    // The floor. A fixture that declares neither directive is executed and asserted against
    // nothing, which reads as coverage and is not.
    assert!(
        expect.exit.is_some(),
        "{name}: no `// EXPECT-EXIT:` directive. Every fixture must pin its exit code; without \
         one the exit check is skipped entirely and a compile failure looks like a pass."
    );
    assert!(
        expect.constrains_stdout(),
        "{name}: declares no `// EXPECT-OUT:` / `// EXPECT-OUT-APPROX:` and no \
         `// EXPECT-OUT-NONE`. Without one of those the stdout check is skipped entirely and the \
         fixture asserts nothing about what it printed."
    );

    let scratch = ScratchDir::new();
    let output = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .args(&expect.run_args)
        .arg(path)
        .current_dir(&scratch.0)
        .output()
        .expect("failed to spawn wukongc");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if let Some(code) = expect.exit {
        let actual = output.status.code();
        assert_eq!(
            actual,
            Some(code),
            "{name}: exit code mismatch (stderr: {stderr})"
        );
    }

    if let Err(report) = common::check_stdout(&stdout, &expect) {
        panic!("{name}: {report}\n--- stderr ---\n{stderr}");
    }

    for want in &expect.stderr_contains {
        assert!(
            stderr.contains(want.as_str()),
            "{name}: stderr does not contain {want:?}\n--- stderr ---\n{stderr}"
        );
    }
}

fn collect_programs() -> Vec<PathBuf> {
    let dir = run_dir();
    let programs = common::collect_wk(&dir);
    assert!(
        programs.len() >= MIN_FIXTURES,
        "{} holds {} .wk fixture(s), floor is {MIN_FIXTURES}. A fixture leaves every gate in this \
         file by being renamed or moved, with no other signal; if the removal was deliberate, \
         lower MIN_FIXTURES in the same commit and say why.",
        dir.display(),
        programs.len()
    );
    programs
}

#[test]
fn run_suite() {
    let programs = collect_programs();
    for p in &programs {
        check_program(p);
    }
    eprintln!("ran {} e2e program(s)", programs.len());
}

/// Differential hardening: optimization must never change observable behavior. For every program,
/// running at -O0 and at -O1/-O2/-O3 must produce identical stdout and exit code.
///
/// `-O3` is currently byte-identical to `-O2` (`wukong_opt` gates passes on `>= 1` and `>= 2`
/// only), so that arm cannot fail today. It is kept as forward insurance for the first pass that
/// tests `>= 3`: the arm that cannot fail now is the arm that catches that pass.
#[test]
fn optimization_is_observationally_invariant() {
    for p in &collect_programs() {
        let base = run_at(p, "-O0");
        for level in ["-O1", "-O2", "-O3"] {
            let other = run_at(p, level);
            let name = p.file_name().unwrap().to_string_lossy();
            assert_eq!(
                base, other,
                "{name}: {level} output differs from -O0\n-O0: {base:?}\n{level}: {other:?}"
            );
        }
    }
}

/// Run a program at a given optimization level on the interpreter, returning (exit_code, stdout).
fn run_at(path: &Path, opt: &str) -> (Option<i32>, String) {
    run_backend(path, opt, None)
}

/// Run a program at a given optimization level on a chosen execution backend (`None` = the default
/// interpreter, `Some("native")` = the Cranelift JIT), returning (exit_code, stdout).
fn run_backend(path: &Path, opt: &str, backend: Option<&str>) -> (Option<i32>, String) {
    let scratch = ScratchDir::new();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    cmd.arg("--run").arg(opt);
    if let Some(b) = backend {
        cmd.arg(format!("--backend={b}"));
    }
    let output = cmd
        .arg(path)
        .current_dir(&scratch.0)
        .output()
        .expect("failed to spawn wukongc");
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
    )
}

/// The hard invariant, extended to native: the Cranelift backend must agree with the interpreter
/// oracle (stdout + exit code) for every program at every optimization level. The interpreter is
/// the differential oracle, and `optimization_is_observationally_invariant` above only re-runs the
/// *interpreter* across opt levels — so this is the guard that the *native* path never silently
/// diverges from the oracle. (All mismatches are collected so one failure reports the full set.)
///
/// It is blind to a recognized kernel by construction: both backends call the same
/// `wukong_runtime` symbol, so a wrong kernel is wrong identically on both sides. What can see a
/// recognizer is `recognize.rs` (did it fire) and `recognizer_census.rs` (did the set of fired
/// kernels change); what can see a wrong *value* is only a fixture's own hand-derived
/// expectations.
#[test]
fn native_matches_interpreter() {
    let mut mismatches = Vec::new();
    for p in &collect_programs() {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        for opt in ["-O0", "-O2"] {
            let interp = run_backend(p, opt, None);
            let native = run_backend(p, opt, Some("native"));
            if interp != native {
                mismatches.push(format!(
                    "{name} @ {opt}:\n    interp = {interp:?}\n    native = {native:?}"
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "interpreter/native divergence ({} case(s)):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Native optimization must be observationally invariant too: the Cranelift backend at -O0 must
/// produce identical stdout and exit code to -O1/-O2/-O3 (the interpreter analogue is
/// `optimization_is_observationally_invariant`).
#[test]
fn native_optimization_is_observationally_invariant() {
    let mut mismatches = Vec::new();
    for p in &collect_programs() {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let base = run_backend(p, "-O0", Some("native"));
        for level in ["-O1", "-O2", "-O3"] {
            let other = run_backend(p, level, Some("native"));
            if base != other {
                mismatches.push(format!(
                    "{name}: native {level} differs from -O0:\n    -O0   = {base:?}\n    {level} = {other:?}"
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "native optimization divergence ({} case(s)):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Pin the `-O2` == `-O3` equivalence explicitly.
///
/// `wukong_opt` gates its passes on `>= 1` and `>= 2` only, so `-O3` is byte-identical to `-O2`
/// today and the `-O3` arm of the two invariance suites above cannot fail. Those arms are kept as
/// forward insurance — but a silent arm cannot *announce* the day the equivalence breaks; it just
/// quietly starts doing real work. This test is the announcement: it compares the emitted MIR of
/// `-O2` and `-O3` directly, so the first pass that gates on `>= 3` fails here **by name** with a
/// message telling the author the `-O3` invariance arms now carry real coverage and their
/// expectations must be re-derived — instead of that change slipping in unremarked. It is cheap:
/// a handful of programs, two `--emit=mir` spawns each, no execution.
#[test]
fn o3_mir_is_byte_identical_to_o2() {
    fn emit_mir(path: &Path, level: &str) -> (Option<i32>, Vec<u8>) {
        let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
            .arg("--emit=mir")
            .arg(level)
            .arg(path)
            .output()
            .expect("failed to spawn wukongc");
        (out.status.code(), out.stdout)
    }
    // Programs that put real work through the optimizer: a GEMM nest, a multi-pass norm, scans, an
    // elementwise transcendental loop, a GEMM+epilogue, and a parallel reduction.
    for name in [
        "tensor_matmul.wk",
        "softmax.wk",
        "scan_1d.wk",
        "transcendental_kernel.wk",
        "linear_bias_relu.wk",
        "parallel_reduce.wk",
    ] {
        let p = run_dir().join(name);
        let (c2, m2) = emit_mir(&p, "-O2");
        let (c3, m3) = emit_mir(&p, "-O3");
        assert_eq!(c2, Some(0), "{name}: --emit=mir -O2 failed to compile");
        assert_eq!(c3, Some(0), "{name}: --emit=mir -O3 failed to compile");
        assert!(
            m2 == m3,
            "{name}: `--emit=mir -O3` is no longer byte-identical to `-O2`. A wukong_opt pass now \
             gates on `>= 3`, so the -O3 arm of optimization_is_observationally_invariant and its \
             native twin now carry real coverage — re-derive their expectations and update this \
             pin's documentation in the same change."
        );
    }
}
