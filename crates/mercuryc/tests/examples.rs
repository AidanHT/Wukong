//! Execution guard for the `examples/*.mer` showcase programs.
//!
//! The `emit` suite only smoke-tests the `tokens`/`ast` stages for examples, and the `run` suite
//! covers `tests/run/*.mer` — so the example programs themselves were never actually *executed*
//! under test. This suite runs every example through the real binary and enforces the project's
//! hard invariant on them: the interpreter oracle and the Cranelift native backend must agree on
//! stdout and exit code, and native optimization must be observationally invariant. Examples carry
//! no `// EXPECT-*` directives, so we cannot assert a specific output — but interp/native agreement
//! and opt-invariance are checkable without one, and catch crashes and silent divergences.

use std::path::{Path, PathBuf};
use std::process::Command;

fn examples_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/mercuryc; the examples live at <repo>/examples.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("examples")
}

fn collect_examples() -> Vec<PathBuf> {
    let dir = examples_dir();
    let mut programs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "mer").unwrap_or(false))
        .collect();
    programs.sort();
    assert!(!programs.is_empty(), "no .mer examples in {}", dir.display());
    programs
}

/// Run an example at a given optimization level on a chosen backend (`None` = the default
/// interpreter, `Some("native")` = the Cranelift JIT), returning (exit_code, stdout).
fn run_backend(path: &Path, opt: &str, backend: Option<&str>) -> (Option<i32>, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_mercuryc"));
    cmd.arg("--run").arg(opt);
    if let Some(b) = backend {
        cmd.arg(format!("--backend={b}"));
    }
    let output = cmd.arg(path).output().expect("failed to spawn mercuryc");
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
    )
}

/// Every example must run identically on the interpreter oracle and the native backend, at both
/// -O0 and -O2 (the hard invariant). All mismatches are collected before failing.
#[test]
fn examples_native_matches_interpreter() {
    let mut mismatches = Vec::new();
    for p in &collect_examples() {
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
        "example interpreter/native divergence ({} case(s)):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Every example must be observationally invariant under optimization on the interpreter: -O0 must
/// match -O1/-O2/-O3 on stdout and exit code.
#[test]
fn examples_optimization_is_observationally_invariant() {
    let mut mismatches = Vec::new();
    for p in &collect_examples() {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let base = run_backend(p, "-O0", None);
        for level in ["-O1", "-O2", "-O3"] {
            let other = run_backend(p, level, None);
            if base != other {
                mismatches.push(format!(
                    "{name}: {level} differs from -O0:\n    -O0   = {base:?}\n    {level} = {other:?}"
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "example optimization divergence ({} case(s)):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
