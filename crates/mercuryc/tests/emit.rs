//! Smoke tests for every `--emit` stage.
//!
//! Front-end stages (`tokens`, `ast`) must succeed for every example and run-suite program. The
//! lowering/codegen stages (`mir-high`, `mir`, `llvm-ir`) must succeed for the run-suite programs,
//! which are all known to lower end to end. This guards the pretty-printers, the MIR verifier
//! (which runs under `--emit=mir`), and the textual LLVM-IR emitter against regressions.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn mer_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "mer").unwrap_or(false))
        .collect();
    v.sort();
    v
}

fn emit(path: &Path, stage: &str) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_mercuryc"))
        .arg(format!("--emit={stage}"))
        .arg(path)
        .output()
        .expect("spawn mercuryc");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn frontend_stages_succeed_for_all_sources() {
    let root = repo_root();
    let mut files = mer_files(&root.join("examples"));
    files.extend(mer_files(&root.join("tests").join("run")));
    assert!(!files.is_empty());

    for f in &files {
        for stage in ["tokens", "ast"] {
            let (ok, stdout, stderr) = emit(f, stage);
            let name = f.file_name().unwrap().to_string_lossy();
            assert!(ok, "{name}: --emit={stage} failed:\n{stderr}");
            assert!(
                !stdout.trim().is_empty(),
                "{name}: --emit={stage} produced no output"
            );
        }
    }
}

#[test]
fn lowering_stages_succeed_for_run_suite() {
    let dir = repo_root().join("tests").join("run");
    for f in &mer_files(&dir) {
        for stage in ["mir-high", "mir", "llvm-ir"] {
            let (ok, stdout, stderr) = emit(f, stage);
            let name = f.file_name().unwrap().to_string_lossy();
            assert!(ok, "{name}: --emit={stage} failed:\n{stderr}");
            assert!(
                !stdout.trim().is_empty(),
                "{name}: --emit={stage} produced no output"
            );
            // The MIR verifier prints internal-compiler-error lines to stderr; there must be none.
            assert!(
                !stderr.contains("internal compiler error"),
                "{name}: --emit={stage} reported a verifier ICE:\n{stderr}"
            );
        }
    }
}
