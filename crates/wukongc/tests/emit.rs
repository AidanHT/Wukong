//! Smoke tests for every `--emit` stage.
//!
//! Front-end stages (`tokens`, `ast`) must succeed for every example and run-suite program. The
//! lowering stages (`mir-high`, `mir`) must succeed for the run-suite programs, which are all known
//! to lower end to end. This guards the pretty-printers and the MIR verifier (which runs under
//! `--emit=mir`). The textual LLVM-IR emitter may refuse a construct it cannot express faithfully
//! (`C0001`, exit 2, one of `wukong_codegen_llvm::DECLINED`), and how many corpus programs it
//! refuses is a ratchet: [`LLVM_DECLINE_RATCHET`].

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn wk_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "wk").unwrap_or(false))
        .collect();
    v.sort();
    v
}

fn emit(path: &Path, stage: &str) -> (bool, String, String) {
    emit_opt(path, stage, 0)
}

fn emit_opt(path: &Path, stage: &str, opt: u8) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .arg(format!("--emit={stage}"))
        .arg(format!("-O{opt}"))
        .arg(path)
        .output()
        .expect("spawn wukongc");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn frontend_stages_succeed_for_all_sources() {
    let root = repo_root();
    let mut files = wk_files(&root.join("examples"));
    files.extend(wk_files(&root.join("tests").join("run")));
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

/// The most `(program, -O level)` pairs of the run corpus and the examples that `--emit=llvm-ir`
/// may refuse. It is a ratchet: lowering a refused construct lowers it, and raising it needs a
/// `DECLINED` row that says why the construct has no faithful LLVM spelling. At 0, every program
/// that compiles at all also emits a module.
const LLVM_DECLINE_RATCHET: usize = 0;

/// Did `--emit=llvm-ir` refuse (the backend's checked decline), as opposed to fail?
fn llvm_declined(stderr: &str) -> bool {
    stderr.contains("C0001") && stderr.contains("LLVM emission [")
}

#[test]
// The ratchet compare stays a `<=` even while its bound is 0: raising the bound must not require
// rewriting the check.
#[allow(clippy::absurd_extreme_comparisons)]
fn lowering_stages_succeed_for_run_suite() {
    let dir = repo_root().join("tests").join("run");
    let mut declined: Vec<String> = Vec::new();
    // Exercise both unoptimized and fully optimized output: -O2 turns on mem2reg/LICM, so the MIR
    // verifier (run under --emit=mir) and the phi-emitting LLVM-IR backend are checked on the
    // optimized, block-parameter form too — not just the alloca-based -O0 form.
    for f in &wk_files(&dir) {
        for opt in [0u8, 2] {
            for stage in ["mir-high", "mir", "llvm-ir"] {
                let (ok, stdout, stderr) = emit_opt(f, stage, opt);
                let name = f.file_name().unwrap().to_string_lossy();
                if stage == "llvm-ir" && !ok && llvm_declined(&stderr) {
                    declined.push(format!("{name} -O{opt}: {}", stderr.trim()));
                    continue;
                }
                assert!(ok, "{name}: --emit={stage} -O{opt} failed:\n{stderr}");
                assert!(
                    !stdout.trim().is_empty(),
                    "{name}: --emit={stage} -O{opt} produced no output"
                );
                // The MIR verifier prints internal-compiler-error lines to stderr; there must be none.
                assert!(
                    !stderr.contains("internal compiler error"),
                    "{name}: --emit={stage} -O{opt} reported a verifier ICE:\n{stderr}"
                );
            }
        }
    }
    // The examples too, wherever the front end lowers them (a few use surface the native lowering
    // does not support yet; those fail with exit 1 before the LLVM stage and are not declines).
    for f in &wk_files(&repo_root().join("examples")) {
        let (ok, _, stderr) = emit_opt(f, "llvm-ir", 2);
        if !ok && llvm_declined(&stderr) {
            let name = f.file_name().unwrap().to_string_lossy();
            declined.push(format!("{name} -O2: {}", stderr.trim()));
        }
    }
    assert!(
        declined.len() <= LLVM_DECLINE_RATCHET,
        "{} --emit=llvm-ir refusals exceed the ratchet of {LLVM_DECLINE_RATCHET}:\n{}",
        declined.len(),
        declined.join("\n")
    );
}

/// `docs/llvm-setup.md` shows the module `--emit=llvm-ir -O2 examples/fib.wk` prints, for readers
/// to compile by hand; it must be exactly what the compiler emits. The `target` lines name the
/// host, so they are compared only for shape.
#[test]
fn llvm_setup_doc_shows_the_emitted_fib_module() {
    let doc = std::fs::read_to_string(repo_root().join("docs").join("llvm-setup.md")).unwrap();
    let marker = "<!-- emitted: wukongc --emit=llvm-ir -O2 examples/fib.wk -->";
    let start = doc
        .find(marker)
        .expect("the fib module block in docs/llvm-setup.md");
    let block = doc[start + marker.len()..].trim_start();
    let block = block.strip_prefix("```llvm").expect("an llvm fence");
    let block = &block[..block.find("```").expect("a closing fence")];
    let fib = repo_root().join("examples").join("fib.wk");
    let (ok, emitted, stderr) = emit_opt(&fib, "llvm-ir", 2);
    assert!(ok, "{stderr}");
    let body = |text: &str| -> Vec<String> {
        text.lines()
            .filter(|l| !l.starts_with("target ") && !l.trim().is_empty())
            .map(|l| l.trim_end().to_string())
            .collect()
    };
    assert_eq!(
        body(block),
        body(&emitted),
        "docs/llvm-setup.md's fib module is stale; paste the current output"
    );
    for text in [block, emitted.as_str()] {
        assert!(text.contains("\ntarget datalayout = \"e-m:"), "{text}");
        assert!(text.contains("\ntarget triple = \"x86_64-"), "{text}");
    }
}
