//! `--explain` surface tests.
//!
//! Three contracts that nothing else enforced:
//!  1. a bare `--explain` lists *every* catalogued code (the no-argument listing),
//!  2. `--explain <CODE>` prints that code's title and body for every code, and
//!  3. each catalogue worked example still behaves as documented: the `bad` program produces
//!     exactly its own code as the first diagnostic, and the `good` program compiles clean.
//!
//! The examples are checked with `--emit=check` (lex + parse + imports + sema, no lowering), so a
//! sema-level example is validated without dragging in codegen limits. This is the doc-test the
//! catalogue never had: an explanation whose example drifts away from the rule now fails the gate.

use std::path::PathBuf;
use std::process::Command;

use wukong_driver::all_explanations;

fn wukongc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_wukongc"))
}

/// Write `src` to a uniquely named file in cargo's per-test temp dir and return its path.
fn tmp(name: &str, src: &str) -> PathBuf {
    let p = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&p, src).expect("write temp fixture");
    p
}

/// The first error's code, read with the renderer's pinned JSON-lines prefix (as in `fail.rs`).
fn first_error_code(stderr: &str) -> Option<String> {
    let line = stderr.lines().next()?;
    let fields = line.strip_prefix("{\"severity\":\"error\",\"code\":\"")?;
    let (code, rest) = fields.split_once('"')?;
    if rest.starts_with(",\"message\":") {
        Some(code.to_string())
    } else {
        None
    }
}

#[test]
fn bare_explain_lists_every_code() {
    let out = wukongc().arg("--explain").output().expect("spawn wukongc");
    assert!(
        out.status.success(),
        "`--explain` with no code should exit 0"
    );
    let text = String::from_utf8_lossy(&out.stdout);
    for e in all_explanations() {
        assert!(
            text.contains(e.code),
            "`--explain` listing is missing {}",
            e.code
        );
    }
}

#[test]
fn explain_each_code_prints_title_and_body() {
    for e in all_explanations() {
        let out = wukongc()
            .arg("--explain")
            .arg(e.code)
            .output()
            .expect("spawn wukongc");
        assert!(out.status.success(), "--explain {} failed", e.code);
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains(e.code), "{}: code not echoed", e.code);
        assert!(text.contains(e.title), "{}: title not echoed", e.code);
    }
}

#[test]
fn catalogue_examples_match_their_code() {
    let mut checked = 0usize;
    for e in all_explanations() {
        if let Some(bad) = e.bad {
            let p = tmp(&format!("explain_bad_{}.wk", e.code), bad);
            let out = wukongc()
                .arg("--error-format=json")
                .arg("--emit=check")
                .arg(&p)
                .output()
                .expect("spawn wukongc");
            assert!(
                !out.status.success(),
                "{}: the `bad` example compiled clean",
                e.code
            );
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert_eq!(
                first_error_code(&stderr).as_deref(),
                Some(e.code),
                "{}: `bad` example produced the wrong first code:\n{stderr}",
                e.code
            );
            checked += 1;
        }
        if let Some(good) = e.good {
            let p = tmp(&format!("explain_good_{}.wk", e.code), good);
            let out = wukongc()
                .arg("--emit=check")
                .arg(&p)
                .output()
                .expect("spawn wukongc");
            assert!(
                out.status.success(),
                "{}: the `good` example failed to compile:\n{}",
                e.code,
                String::from_utf8_lossy(&out.stderr)
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no catalogue examples were checked");
    eprintln!("checked {checked} catalogue example(s)");
}
