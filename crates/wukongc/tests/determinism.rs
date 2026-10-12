//! Determinism guard: the compiler must be a *pure function* of its input.
//!
//! Rust's `HashMap`/`HashSet` use a per-process random hash seed, so iterating one and letting that
//! order reach the output makes compilation nondeterministic across runs. This bug class is
//! GATE-BLIND: such reorderings (e.g. hoisting independent loop-invariant instructions into a
//! preheader, or numbering callee `FuncRef`s) preserve computed values, so `native_matches_interpreter`
//! and `optimization_is_observationally_invariant` — which compare only stdout/exit, and within data
//! that may itself be one process — still agree. Yet the emitted MIR differs run to run, breaking
//! reproducible builds and any compile-output cache.
//!
//! Every test here spawns a *fresh process per compile* (each gets a distinct hash seed) and
//! asserts the bytes are identical. Three seeds per run; CI re-runs accumulate many more over time.
//! The corpus axes are covered separately because a leak can enter at any of them:
//!
//! * [`mir_emission_is_deterministic_across_processes`] — `--emit=mir -O2`, the whole optimizer
//!   including LICM, the pass whose loop-body iteration order was the one place this actually bit.
//!   The three processes also run the optimizer on different thread counts
//!   (`WUKONG_OPT_THREADS=1`, the automatic count, and `4`), so the same sweep is the corpus-wide
//!   proof that the parallel per-function schedule (`wukong_opt::parallel`) emits the MIR one
//!   thread does — an explicit count above one sends every batch of two or more functions to the
//!   pool, however small. The object and `-g` arms vary the count the same way, and
//!   [`o1_mir_does_not_depend_on_the_optimizer_thread_count`] repeats the comparison at `-O1`,
//!   where nothing is inlined and the call-graph levels are at their widest.
//! * [`object_emission_is_deterministic_across_processes`] — `--emit=obj -O2`, i.e. *codegen and
//!   symbol emission*, which the MIR gate never reaches. The in-process object-byte tests in
//!   `wukong_codegen_cranelift` compile several times inside ONE process, so all their emissions
//!   share a single `RandomState` seed and cannot see this axis at all; they cover a different one
//!   (parallel-vs-serial, verify-vs-noverify) and are still worth keeping.
//! * [`debug_object_emission_is_deterministic_across_processes`] — `--emit=obj -O2 -g`, the DWARF
//!   a `-g` object adds, on a representative slice (the largest programs and a multi-file import).
//! * [`diagnostic_emission_is_deterministic_across_processes`] — the JSON diagnostic stream of the
//!   compile-fail corpus. A `HashMap<Symbol, _>` in name resolution or the renderer would reorder
//!   the errors a multi-error program reports; `fail.rs` only asserts that ONE expected code is
//!   present somewhere, so it cannot see the ordering.
//!
//! Representative AST, textual LLVM IR, and gradient cases additionally cover those artifact
//! paths, requiring successful, nonempty output from every process. They are not corpus sweeps.
//! Source, flags, compiler/target, and inherited ambient configuration are inputs held fixed for
//! each comparison; these checks do not establish independence from environment configuration.
//!
//! The environment axis has its own arm, [`mir_emission_does_not_depend_on_ambient_kill_switches`]:
//! every other test here inherits the ambient environment into all three child processes, so a
//! `WUKONG_*` kill-switch exported in the developer's shell is held equal across the comparison and
//! is invisible to it. That arm pins the emitted MIR against a run with the known compilation-
//! affecting variables explicitly removed, turning "a stray exported kill-switch silently changed my
//! build" into a named failure.
//!
//! CAVEAT worth stating so a pass is not over-read, and a KNOWN GAP recorded here rather than left
//! implicit: the "three independent hash seeds" mechanism only bites std `HashMap`/`HashSet`. The
//! optimizer's `FxHash*` maps have a FIXED seed and are cross-process reproducible by construction,
//! so these gates cannot detect an insertion-order dependency in those. Closing it needs an
//! instrument these process-level tests cannot provide — a debug-only seed override or a
//! shuffled-insertion-order build inside `wukong_opt` — so it is tracked as an open gap, not a
//! property any test in this file establishes.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_dir(sub: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join(sub)
}

fn collect_wk(dir: &Path) -> Vec<PathBuf> {
    let mut programs: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "wk").unwrap_or(false))
        .collect();
    programs.sort();
    assert!(!programs.is_empty(), "no .wk programs in {}", dir.display());
    programs
}

fn collect_programs() -> Vec<PathBuf> {
    collect_wk(&repo_dir("run"))
}

/// A scratch directory unique to this process, removed on drop. `--emit=obj` writes a file, and two
/// concurrent `cargo test -p wukongc` invocations must not share the name.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(tag: &str) -> ScratchDir {
        let dir = std::env::temp_dir()
            .join("wukongc_determinism")
            .join(format!("{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        ScratchDir(dir)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `--emit=mir -O2` for one program in a fresh process; returns (succeeded, stdout-bytes).
fn emit_mir_o2(path: &Path) -> (bool, Vec<u8>) {
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .arg("--emit=mir")
        .arg("-O2")
        .arg(path)
        .output()
        .expect("failed to spawn wukongc");
    (out.status.success(), out.stdout)
}

/// The optimizer thread counts the corpus sweeps rotate through, one per fresh process: the
/// sequential schedule, the automatic count (`None`: the variable removed), and four threads,
/// which sends every batch of two or more functions to the pool.
const OPT_THREADS: [Option<&str>; 3] = [Some("1"), None, Some("4")];

/// Pin `WUKONG_OPT_THREADS` for one child process (`None`: removed, the automatic count).
fn with_opt_threads(cmd: &mut Command, threads: Option<&str>) {
    match threads {
        Some(n) => cmd.env("WUKONG_OPT_THREADS", n),
        None => cmd.env_remove("WUKONG_OPT_THREADS"),
    };
}

/// [`emit_mir_o2`] with the optimizer on `threads` threads (see [`OPT_THREADS`]).
fn emit_mir_o2_threads(path: &Path, threads: Option<&str>) -> (bool, Vec<u8>) {
    emit_mir_threads(path, "-O2", threads)
}

/// `--emit=mir <level>` for one program in a fresh process, the optimizer on `threads` threads.
fn emit_mir_threads(path: &Path, level: &str, threads: Option<&str>) -> (bool, Vec<u8>) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    with_opt_threads(&mut cmd, threads);
    let out = cmd
        .arg("--emit=mir")
        .arg(level)
        .arg(path)
        .output()
        .expect("failed to spawn wukongc");
    (out.status.success(), out.stdout)
}

/// `--emit=obj -O2 -o <dst>` for one program in a fresh process; returns the object bytes, or
/// `None` when the program does not lower (there is then nothing to compare). `threads` is the
/// optimizer's thread count (see [`OPT_THREADS`]).
fn emit_obj_o2(
    path: &Path,
    dst: &Path,
    serial_codegen: bool,
    threads: Option<&str>,
) -> Option<Vec<u8>> {
    let _ = std::fs::remove_file(dst);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    with_opt_threads(&mut cmd, threads);
    if serial_codegen {
        // The `WUKONG_PAR_CODEGEN=0` kill-switch forces the serial object path, so both codegen
        // paths are covered rather than whichever one the corpus happens to trigger.
        cmd.env("WUKONG_PAR_CODEGEN", "0");
    }
    let out = cmd
        .arg("--emit=obj")
        .arg("-O2")
        .arg("-o")
        .arg(dst)
        .arg(path)
        .output()
        .expect("failed to spawn wukongc");
    if !out.status.success() {
        return None;
    }
    std::fs::read(dst).ok()
}

/// The `WUKONG_*` environment variables that change the MIR `--emit=mir` prints. Enumerated so the
/// hermeticity arm below is grounded rather than guessing: `WUKONG_NO_RECOGNIZE` removes recognizer
/// dispatch (`wukong_driver`), `WUKONG_P4_NO_256` disables the 256-bit veckernel recipe
/// (`wukong_mir_build` and `wukong_opt`), `WUKONG_VEC256_MIN_TRIP` moves the optimizer's 256-bit
/// trip gate, and the five optimizer kill-switches gate vectorization, unrolling, the
/// induction-variable ramp, straight-line SLP and the vectorizer's lane-serial regions
/// (`wukong_opt`). Pure tracing switches
/// (`WUKONG_VEC_TRACE`, `WUKONG_SLP_TRACE`), the linker
/// (`WUKONG_BUILD_RUSTC`), the optimizer's thread count (`WUKONG_OPT_THREADS`, which the corpus
/// sweeps vary on purpose) and the GPU-only variables are deliberately excluded: they do not alter
/// the emitted MIR. A new kill-switch that does must be added here in the same change.
const MIR_AFFECTING_ENV: &[&str] = &[
    "WUKONG_NO_RECOGNIZE",
    "WUKONG_P4_NO_256",
    "WUKONG_NO_VECTORIZE",
    "WUKONG_NO_UNROLL",
    "WUKONG_NO_IV_RAMP",
    "WUKONG_NO_SLP",
    "WUKONG_NO_LANE_SERIAL",
    "WUKONG_VEC256_MIN_TRIP",
];

/// `--emit=mir -O2` for one program in a fresh process with `removed` cleared from the child's
/// environment; returns (succeeded, stdout-bytes).
fn emit_mir_o2_without(path: &Path, removed: &[&str]) -> (bool, Vec<u8>) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    for v in removed {
        cmd.env_remove(v);
    }
    let out = cmd
        .arg("--emit=mir")
        .arg("-O2")
        .arg(path)
        .output()
        .expect("failed to spawn wukongc");
    (out.status.success(), out.stdout)
}

/// The JSON diagnostic stream for one compile-fail fixture in a fresh process; returns stderr bytes.
fn emit_diagnostics(path: &Path) -> Vec<u8> {
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .arg("--error-format=json")
        .arg("--emit=mir")
        .arg(path)
        .output()
        .expect("failed to spawn wukongc");
    out.stderr
}

/// Compare successful textual artifacts, never treating repeated failure or silence as agreement.
fn assert_representative_text_determinism(path: &Path, args: &[&str]) -> Vec<u8> {
    let mut baseline = Vec::new();
    for run in 0..3 {
        let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
            .args(args)
            .arg("--color=never")
            .arg(path)
            .output()
            .expect("failed to spawn wukongc");
        assert!(
            out.status.success(),
            "{} {args:?}, process {run}: status {}; stderr: {}",
            path.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !out.stdout.is_empty(),
            "{} {args:?}, process {run}: empty artifact; stderr: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        );
        if run == 0 {
            baseline = out.stdout;
        } else {
            assert!(
                baseline == out.stdout,
                "{} {args:?}: artifact bytes differ in fresh process {run}",
                path.display()
            );
        }
    }
    eprintln!(
        "representative determinism: {} {args:?}, {} artifact bytes x3 fresh processes",
        path.display(),
        baseline.len()
    );
    baseline
}

#[test]
fn representative_ast_emission_is_deterministic_across_processes() {
    // Declaration/field ordering, match patterns, and generic syntax in the root-file AST.
    for name in ["enum_payload_struct.wk", "generics_monomorphization.wk"] {
        assert_representative_text_determinism(&repo_dir("run").join(name), &["--emit=ast"]);
    }
}

#[test]
fn representative_llvm_ir_emission_is_deterministic_across_processes() {
    // O0 preserves specialized functions and aggregate accesses; O2 also covers their optimized
    // emission, even when constant folding and inlining remove the original structure.
    for name in ["generics_monomorphization.wk", "struct_fn.wk"] {
        assert_representative_text_determinism(
            &repo_dir("run").join(name),
            &["--emit=llvm-ir", "-O2"],
        );
        let artifact = assert_representative_text_determinism(
            &repo_dir("run").join(name),
            &["--emit=llvm-ir", "-O0"],
        );
        let text = std::str::from_utf8(&artifact).expect("LLVM IR must be UTF-8");
        let markers: &[&str] = if name == "generics_monomorphization.wk" {
            &["define float @id$f32(", "define i32 @id$i32("]
        } else {
            &["define i32 @byval(ptr ", "load i32, ptr ", "getelementptr "]
        };
        for marker in markers {
            assert!(
                text.contains(marker),
                "{name} O0 LLVM IR must retain {marker:?}: {text}"
            );
        }
    }
}

/// A multi-artifact `--emit=a,b,..` stream (every text stage of one compile, under its headers) is
/// as reproducible as each stage alone.
#[test]
fn representative_multi_artifact_emission_is_deterministic_across_processes() {
    for name in ["generics_monomorphization.wk", "import_multi.wk"] {
        let artifact = assert_representative_text_determinism(
            &repo_dir("run").join(name),
            &["--emit=tokens,ast,mir-high,mir,mir-text,llvm-ir", "-O2"],
        );
        let text = std::str::from_utf8(&artifact).expect("text artifacts must be UTF-8");
        let headers: Vec<&str> = text.lines().filter(|l| l.starts_with("==> ")).collect();
        assert_eq!(
            headers,
            [
                "==> tokens <==",
                "==> ast <==",
                "==> mir-high <==",
                "==> mir <==",
                "==> mir-text <==",
                "==> llvm-ir <==",
            ],
            "{name}"
        );
    }
}

#[test]
fn representative_grad_emission_is_deterministic_across_processes() {
    let scratch = ScratchDir::new("grad");
    let source = scratch.0.join("cubic.wk");
    // Same cubic buffer loss as the driver's finite-difference correctness gate. Here the
    // property under test is emitted forward/backward byte order, not gradient numerics.
    std::fs::write(
        &source,
        "module m\nfn loss(x: [f32; 1], mut out: [f32; 1]) -> f32 {\n\
         let a: f32 = x[0]; let l: f32 = a*a*a; out[0] = l; return l; }\n",
    )
    .expect("write gradient determinism fixture");
    let artifact = assert_representative_text_determinism(
        &source,
        &["--emit=grad", "-O2", "--grad-of=loss", "--grad-wrt=0"],
    );
    let text = std::str::from_utf8(&artifact).expect("gradient MIR must be UTF-8");
    let (_, backward) = text
        .split_once("fn loss_grad(")
        .expect("artifact must contain the generated backward function");
    for op in ["fmul ", "fadd ", "store "] {
        assert!(
            backward.contains(op),
            "cubic backward must contain {op:?}: {backward}"
        );
    }
}

/// The optimizer's parallel schedule at its widest. `-O1` inlines nothing, so every function the
/// front end lowered — every helper, monomorph and outlined region — is a task of its call-graph
/// level, where the `-O2` sweep above mostly sees what inlining left. One thread against four, over
/// the whole corpus; the programs are spread over a few test threads to keep the arm cheap.
#[test]
fn o1_mir_does_not_depend_on_the_optimizer_thread_count() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let programs = collect_programs();
    let next = AtomicUsize::new(0);
    let compared = AtomicUsize::new(0);
    let mismatched = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                while let Some(p) = programs.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let (ok, one) = emit_mir_threads(p, "-O1", Some("1"));
                    if !ok && one.is_empty() {
                        continue; // does not lower: nothing to compare
                    }
                    let (_, four) = emit_mir_threads(p, "-O1", Some("4"));
                    compared.fetch_add(1, Ordering::Relaxed);
                    if one != four {
                        let name = p.file_name().unwrap().to_string_lossy().into_owned();
                        mismatched.lock().unwrap().push(name);
                    }
                }
            });
        }
    });
    let mut mismatched = mismatched.into_inner().unwrap();
    mismatched.sort();
    assert!(
        mismatched.is_empty(),
        "`--emit=mir -O1` differs between WUKONG_OPT_THREADS=1 and =4 for {} program(s) — a pass \
         reads state another function's optimization changes, or the parallel schedule hands a \
         function a different call-effect table than the sequential one:\n{}",
        mismatched.len(),
        mismatched.join("\n")
    );
    let compared = compared.into_inner();
    assert!(
        compared * 10 >= programs.len() * 9,
        "the -O1 thread-count arm compared only {compared} of {} programs",
        programs.len()
    );
    eprintln!(
        "-O1 thread-count determinism: compared {compared} of {} programs, 1 vs 4 threads",
        programs.len()
    );
}

#[test]
fn mir_emission_is_deterministic_across_processes() {
    let programs = collect_programs();
    let mut nondeterministic = Vec::new();
    let mut compared = 0usize;
    for p in &programs {
        // Three separate processes → three independent hash seeds, and three optimizer thread
        // counts.
        let (ok0, r0) = emit_mir_o2_threads(p, OPT_THREADS[0]);
        if !ok0 {
            // A program that does not lower (e.g. an intentionally not-yet-supported construct) emits
            // nothing on stdout; skip it — there is no MIR to compare.
            if r0.is_empty() {
                continue;
            }
        }
        let (_, r1) = emit_mir_o2_threads(p, OPT_THREADS[1]);
        let (_, r2) = emit_mir_o2_threads(p, OPT_THREADS[2]);
        compared += 1;
        if r0 != r1 || r1 != r2 {
            nondeterministic.push(p.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    assert!(
        nondeterministic.is_empty(),
        "`--emit=mir -O2` is nondeterministic across processes for {} program(s) — a HashMap/HashSet \
         iteration order is leaking into the emitted MIR, or the optimizer's output depends on its \
         thread count (the processes ran WUKONG_OPT_THREADS={OPT_THREADS:?}):\n{}",
        nondeterministic.len(),
        nondeterministic.join("\n")
    );
    // A gate that skipped every program would report `ok` having compared nothing: a regression in
    // argument parsing that makes `--emit=mir` exit non-zero with empty stdout fires the `continue`
    // above for all of them. Floor the count so silence is not mistaken for agreement.
    assert!(
        compared * 10 >= programs.len() * 9,
        "determinism gate compared only {compared} of {} programs — it is not covering the corpus",
        programs.len()
    );
    eprintln!(
        "mir determinism: compared {compared} of {} programs x3 processes \
         (WUKONG_OPT_THREADS={OPT_THREADS:?})",
        programs.len()
    );
}

/// Codegen and symbol emission must be a pure function of the input too. This is the axis the MIR
/// gate cannot see (it stops before the backend) and the in-process object-byte tests cannot see
/// (one process, one hash seed).
#[test]
fn object_emission_is_deterministic_across_processes() {
    let programs = collect_programs();
    let scratch = ScratchDir::new("obj");
    let mut nondeterministic = Vec::new();
    let mut serial_mismatch = Vec::new();
    let mut compared = 0usize;

    // The serial codegen path is exercised on the largest programs, which are the ones with enough
    // functions for the parallel path to engage in the first place. Ordering by size then name is
    // deterministic, so the subset does not drift between runs.
    let mut by_size: Vec<(u64, PathBuf)> = programs
        .iter()
        .map(|p| {
            (
                std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
                p.clone(),
            )
        })
        .collect();
    by_size.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let serial_subset: Vec<PathBuf> = by_size.into_iter().take(24).map(|(_, p)| p).collect();

    for p in &programs {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let a = scratch.0.join("a.o");
        let b = scratch.0.join("b.o");
        let c = scratch.0.join("c.o");
        let Some(o0) = emit_obj_o2(p, &a, false, OPT_THREADS[0]) else {
            continue; // does not lower to an object — nothing to compare
        };
        let (Some(o1), Some(o2)) = (
            emit_obj_o2(p, &b, false, OPT_THREADS[1]),
            emit_obj_o2(p, &c, false, OPT_THREADS[2]),
        ) else {
            nondeterministic.push(format!("{name}: emitted an object once but not again"));
            continue;
        };
        compared += 1;
        if o0 != o1 || o1 != o2 {
            nondeterministic.push(format!(
                "{name}: object bytes differ between fresh processes"
            ));
            continue;
        }
        if serial_subset.contains(p) {
            let s = scratch.0.join("s.o");
            let t = scratch.0.join("t.o");
            match (
                emit_obj_o2(p, &s, true, OPT_THREADS[0]),
                emit_obj_o2(p, &t, true, OPT_THREADS[2]),
            ) {
                (Some(s0), Some(s1)) => {
                    if s0 != s1 {
                        nondeterministic.push(format!(
                            "{name}: serial-codegen object bytes differ per process"
                        ));
                    } else if s0 != o0 {
                        serial_mismatch.push(name.clone());
                    }
                }
                _ => nondeterministic.push(format!(
                    "{name}: serial codegen failed where parallel succeeded"
                )),
            }
        }
    }

    assert!(
        nondeterministic.is_empty(),
        "`--emit=obj -O2` is nondeterministic across processes for {} program(s) — a HashMap/HashSet \
         iteration order is leaking into codegen or symbol emission, or the optimizer's output \
         depends on its thread count:\n{}",
        nondeterministic.len(),
        nondeterministic.join("\n")
    );
    assert!(
        serial_mismatch.is_empty(),
        "serial (`WUKONG_PAR_CODEGEN=0`) and parallel object bytes differ for {} program(s) — the \
         parallel codegen path is not byte-equivalent to the serial one:\n{}",
        serial_mismatch.len(),
        serial_mismatch.join("\n")
    );
    assert!(
        compared * 10 >= programs.len() * 9,
        "object determinism gate compared only {compared} of {} programs — it is not covering the \
         corpus",
        programs.len()
    );
    eprintln!(
        "object determinism: compared {compared} of {} programs x3 processes ({} also on the \
         serial codegen path)",
        programs.len(),
        serial_subset.len()
    );
}

/// `--emit=obj -O2 -g -o <dst>` in a fresh process (see [`emit_obj_o2`]).
fn emit_obj_o2_debug(
    path: &Path,
    dst: &Path,
    serial_codegen: bool,
    threads: Option<&str>,
) -> Option<Vec<u8>> {
    let _ = std::fs::remove_file(dst);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    with_opt_threads(&mut cmd, threads);
    if serial_codegen {
        cmd.env("WUKONG_PAR_CODEGEN", "0");
    }
    let out = cmd
        .arg("--emit=obj")
        .arg("-O2")
        .arg("-g")
        .arg("-o")
        .arg(dst)
        .arg(path)
        .output()
        .expect("failed to spawn wukongc");
    if !out.status.success() {
        return None;
    }
    std::fs::read(dst).ok()
}

/// `-g` objects are a pure function of the input too: the DWARF (string table, file table, line
/// program, relocations) is built in program order with insertion-ordered tables, so three fresh
/// processes and the serial codegen path must agree byte for byte. A representative slice — the 24
/// largest programs (the most functions and srcloc ranges) and the multi-file import fixture, whose
/// line table names every file it reads — keeps the cost of a fourth corpus sweep off the gate.
#[test]
fn debug_object_emission_is_deterministic_across_processes() {
    let programs = collect_programs();
    let mut by_size: Vec<(u64, PathBuf)> = programs
        .iter()
        .map(|p| {
            (
                std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
                p.clone(),
            )
        })
        .collect();
    by_size.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut slice: Vec<PathBuf> = by_size.into_iter().take(24).map(|(_, p)| p).collect();
    let multi = repo_dir("run").join("import_multi.wk");
    slice.push(multi.clone());

    let scratch = ScratchDir::new("obj_g");
    let mut failures = Vec::new();
    let mut compared = 0usize;
    for p in &slice {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let plain = emit_obj_o2(p, &scratch.0.join("plain.o"), false, OPT_THREADS[1]);
        let g = [
            emit_obj_o2_debug(p, &scratch.0.join("a.o"), false, OPT_THREADS[0]),
            emit_obj_o2_debug(p, &scratch.0.join("b.o"), false, OPT_THREADS[1]),
            emit_obj_o2_debug(p, &scratch.0.join("c.o"), false, OPT_THREADS[2]),
            emit_obj_o2_debug(p, &scratch.0.join("s.o"), true, OPT_THREADS[1]),
        ];
        match (plain, &g) {
            (None, [None, None, None, None]) => continue, // does not lower to an object
            (Some(plain), [Some(a), Some(b), Some(c), Some(s)]) => {
                compared += 1;
                if a != b || b != c {
                    failures.push(format!(
                        "{name}: `-g` object bytes differ between processes"
                    ));
                } else if a != s {
                    failures.push(format!("{name}: `-g` serial and parallel objects differ"));
                } else if a.len() <= plain.len() {
                    failures.push(format!(
                        "{name}: the `-g` object is no larger than the plain one"
                    ));
                } else if !a.windows(11).any(|w| w == b".debug_line") {
                    failures.push(format!("{name}: the `-g` object has no .debug_line"));
                }
            }
            _ => failures.push(format!("{name}: `-g` changed whether the program compiles")),
        }
    }
    // The import fixture's line table names all three files it reads (DWARF 4 file names are
    // inline strings of `.debug_line`).
    let multi_g = emit_obj_o2_debug(&multi, &scratch.0.join("m.o"), false, OPT_THREADS[1])
        .expect("import_multi");
    for file in ["import_multi.wk", "mathlib.wk", "deep.wk"] {
        assert!(
            multi_g.windows(file.len()).any(|w| w == file.as_bytes()),
            "the `-g` object of import_multi.wk does not name `{file}`"
        );
    }
    assert!(
        failures.is_empty(),
        "`--emit=obj -O2 -g` is not deterministic:
{}",
        failures.join(
            "
"
        )
    );
    assert!(
        compared >= 20,
        "the `-g` determinism slice compared only {compared} programs"
    );
    eprintln!("debug object determinism: compared {compared} programs x3 processes + serial");
}

/// Diagnostic ORDER must be deterministic too: a multi-error program has to report its errors in
/// the same sequence every run, or a build log diff is noise and an error-count-based tool is
/// unreliable. `fail.rs` asserts only that one expected code appears somewhere in the stream, so
/// nothing else in the suite constrains the ordering.
#[test]
fn diagnostic_emission_is_deterministic_across_processes() {
    let fixtures = collect_wk(&repo_dir("fail"));
    let mut nondeterministic = Vec::new();
    let mut compared = 0usize;
    for p in &fixtures {
        let e0 = emit_diagnostics(p);
        if e0.is_empty() {
            continue;
        }
        let e1 = emit_diagnostics(p);
        let e2 = emit_diagnostics(p);
        compared += 1;
        if e0 != e1 || e1 != e2 {
            nondeterministic.push(p.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    assert!(
        nondeterministic.is_empty(),
        "the JSON diagnostic stream is nondeterministic across processes for {} fixture(s) — a \
         HashMap/HashSet iteration order is leaking into diagnostic ordering:\n{}",
        nondeterministic.len(),
        nondeterministic.join("\n")
    );
    assert!(
        compared * 10 >= fixtures.len() * 9,
        "diagnostic determinism gate compared only {compared} of {} fixtures — it is not covering \
         the corpus",
        fixtures.len()
    );
    eprintln!(
        "diagnostic determinism: compared {compared} of {} fixtures x3 processes",
        fixtures.len()
    );
}

/// The emitted MIR must not depend on any ambient `WUKONG_*` kill-switch.
///
/// Every other gate in this file inherits the ambient environment into all of its child processes,
/// so a developer who has `WUKONG_NO_VECTORIZE` (or any of [`MIR_AFFECTING_ENV`]) exported from an
/// earlier A/B gets a different `--emit=mir` on every one of those runs and none of them notices —
/// the setting is held equal across the comparison. This arm is the one that can see it: it compiles
/// each program once under the ambient environment and once with every known compilation-affecting
/// variable explicitly removed, and asserts the bytes match. In a clean environment (CI) both runs
/// are identical and this is a cheap confirmation; in a shell with a kill-switch exported it fails
/// by name, which is exactly the "pure function of its input" claim the module header makes, checked
/// against the one input the other arms silently share.
///
/// Representative rather than a corpus sweep: programs chosen so every switch in the list is live —
/// a recognized GEMM and a recognized transcendental loop (recognition + the 256-bit recipe), an
/// elementwise and a scan loop (vectorization, the IV ramp, unrolling).
#[test]
fn mir_emission_does_not_depend_on_ambient_kill_switches() {
    for name in [
        "tensor_matmul.wk",
        "transcendental_kernel.wk",
        "softmax.wk",
        "saxpy.wk",
        "scan_1d.wk",
    ] {
        let p = repo_dir("run").join(name);
        let (ok_ambient, ambient) = emit_mir_o2(&p);
        let (ok_hermetic, hermetic) = emit_mir_o2_without(&p, MIR_AFFECTING_ENV);
        assert!(
            ok_ambient && ok_hermetic,
            "{name}: `--emit=mir -O2` failed (ambient ok={ok_ambient}, hermetic ok={ok_hermetic})"
        );
        assert!(
            !ambient.is_empty(),
            "{name}: `--emit=mir -O2` produced no MIR to compare"
        );
        assert!(
            ambient == hermetic,
            "{name}: the emitted MIR depends on an ambient WUKONG_* kill-switch — one of \
             {MIR_AFFECTING_ENV:?} is exported in this environment and is silently changing the \
             build. Compilation must be a pure function of its input; unset it before building, or \
             if you added a new kill-switch, add it to MIR_AFFECTING_ENV in the same change."
        );
    }
    eprintln!("environment hermeticity: 5 programs, ambient vs {MIR_AFFECTING_ENV:?}-cleared MIR");
}
