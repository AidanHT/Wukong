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
//! The three axes are covered separately because a leak can enter at any of them:
//!
//! * [`mir_emission_is_deterministic_across_processes`] — `--emit=mir -O2`, the whole optimizer
//!   including LICM, the pass whose loop-body iteration order was the one place this actually bit.
//! * [`object_emission_is_deterministic_across_processes`] — `--emit=obj -O2`, i.e. *codegen and
//!   symbol emission*, which the MIR gate never reaches. The in-process object-byte tests in
//!   `wukong_codegen_cranelift` compile several times inside ONE process, so all their emissions
//!   share a single `RandomState` seed and cannot see this axis at all; they cover a different one
//!   (parallel-vs-serial, verify-vs-noverify) and are still worth keeping.
//! * [`diagnostic_emission_is_deterministic_across_processes`] — the JSON diagnostic stream of the
//!   compile-fail corpus. A `HashMap<Symbol, _>` in name resolution or the renderer would reorder
//!   the errors a multi-error program reports; `fail.rs` only asserts that ONE expected code is
//!   present somewhere, so it cannot see the ordering.
//!
//! CAVEAT worth stating so a pass is not over-read: the "three independent hash seeds" mechanism
//! only bites std `HashMap`/`HashSet`. The optimizer's `FxHash*` maps have a FIXED seed and are
//! cross-process reproducible by construction, so these gates cannot detect an insertion-order
//! dependency in those. That axis needs a different instrument.

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

/// `--emit=obj -O2 -o <dst>` for one program in a fresh process; returns the object bytes, or
/// `None` when the program does not lower (there is then nothing to compare).
fn emit_obj_o2(path: &Path, dst: &Path, serial_codegen: bool) -> Option<Vec<u8>> {
    let _ = std::fs::remove_file(dst);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
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

#[test]
fn mir_emission_is_deterministic_across_processes() {
    let programs = collect_programs();
    let mut nondeterministic = Vec::new();
    let mut compared = 0usize;
    for p in &programs {
        // Three separate processes → three independent hash seeds.
        let (ok0, r0) = emit_mir_o2(p);
        if !ok0 {
            // A program that does not lower (e.g. an intentionally not-yet-supported construct) emits
            // nothing on stdout; skip it — there is no MIR to compare.
            if r0.is_empty() {
                continue;
            }
        }
        let (_, r1) = emit_mir_o2(p);
        let (_, r2) = emit_mir_o2(p);
        compared += 1;
        if r0 != r1 || r1 != r2 {
            nondeterministic.push(p.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    assert!(
        nondeterministic.is_empty(),
        "`--emit=mir -O2` is nondeterministic across processes for {} program(s) — a HashMap/HashSet \
         iteration order is leaking into the emitted MIR:\n{}",
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
        "mir determinism: compared {compared} of {} programs x3 processes",
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
        .map(|p| (std::fs::metadata(p).map(|m| m.len()).unwrap_or(0), p.clone()))
        .collect();
    by_size.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let serial_subset: Vec<PathBuf> = by_size.into_iter().take(24).map(|(_, p)| p).collect();

    for p in &programs {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let a = scratch.0.join("a.o");
        let b = scratch.0.join("b.o");
        let c = scratch.0.join("c.o");
        let Some(o0) = emit_obj_o2(p, &a, false) else {
            continue; // does not lower to an object — nothing to compare
        };
        let (Some(o1), Some(o2)) = (emit_obj_o2(p, &b, false), emit_obj_o2(p, &c, false)) else {
            nondeterministic.push(format!("{name}: emitted an object once but not again"));
            continue;
        };
        compared += 1;
        if o0 != o1 || o1 != o2 {
            nondeterministic.push(format!("{name}: object bytes differ between fresh processes"));
            continue;
        }
        if serial_subset.contains(p) {
            let s = scratch.0.join("s.o");
            let t = scratch.0.join("t.o");
            match (emit_obj_o2(p, &s, true), emit_obj_o2(p, &t, true)) {
                (Some(s0), Some(s1)) => {
                    if s0 != s1 {
                        nondeterministic
                            .push(format!("{name}: serial-codegen object bytes differ per process"));
                    } else if s0 != o0 {
                        serial_mismatch.push(name.clone());
                    }
                }
                _ => nondeterministic
                    .push(format!("{name}: serial codegen failed where parallel succeeded")),
            }
        }
    }

    assert!(
        nondeterministic.is_empty(),
        "`--emit=obj -O2` is nondeterministic across processes for {} program(s) — a HashMap/HashSet \
         iteration order is leaking into codegen or symbol emission:\n{}",
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
