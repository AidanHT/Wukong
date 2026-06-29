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
//! This test spawns a *fresh process per compile* (each gets a distinct hash seed) and asserts the
//! `--emit=mir -O2` output is byte-identical. `-O2` is the strong case: it runs the whole pipeline,
//! including LICM — the pass whose loop-body iteration order was the one place this actually bit. A
//! single run uses three seeds; CI re-runs accumulate many more over time.

use std::path::{Path, PathBuf};
use std::process::Command;

fn run_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("run")
}

fn collect_programs() -> Vec<PathBuf> {
    let dir = run_dir();
    let mut programs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "mer").unwrap_or(false))
        .collect();
    programs.sort();
    assert!(!programs.is_empty(), "no .mer programs in {}", dir.display());
    programs
}

/// `--emit=mir -O2` for one program in a fresh process; returns (succeeded, stdout-bytes).
fn emit_mir_o2(path: &Path) -> (bool, Vec<u8>) {
    let out = Command::new(env!("CARGO_BIN_EXE_mercuryc"))
        .arg("--emit=mir")
        .arg("-O2")
        .arg(path)
        .output()
        .expect("failed to spawn mercuryc");
    (out.status.success(), out.stdout)
}

#[test]
fn mir_emission_is_deterministic_across_processes() {
    let mut nondeterministic = Vec::new();
    for p in &collect_programs() {
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
}
