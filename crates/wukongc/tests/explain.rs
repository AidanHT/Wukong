//! `--explain` surface tests.
//!
//! Three contracts that nothing else enforced:
//!  1. a bare `--explain` lists *every* catalogued code (the no-argument listing),
//!  2. `--explain <CODE>` prints that code's title and body for every code, and
//!  3. each catalogue worked example still behaves as documented: the `bad` program produces
//!     exactly its own code as the first diagnostic, and the `good` program compiles clean. For
//!     an error code the `bad` build fails; for a warning (`W`) code it succeeds, its first
//!     diagnostic is that warning, and the `good` build reports nothing at all. A line
//!     `// file: <path>` in an example starts another file of the program (an import needs one).
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

/// Write the example `src` into a fresh directory `name` of cargo's per-test temp dir and return
/// the root file's path. Each `// file: <path>` line starts another file, written at `<path>`
/// relative to the root.
fn tmp(name: &str, src: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the example directory");
    let mut files: Vec<(String, String)> = vec![("main.wk".to_string(), String::new())];
    for line in src.split_inclusive('\n') {
        match line.trim_end().strip_prefix("// file: ") {
            Some(path) => files.push((path.trim().to_string(), String::new())),
            None => files.last_mut().expect("the root file").1.push_str(line),
        }
    }
    for (path, text) in &files {
        let p = dir.join(path);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("create the example's directories");
        }
        std::fs::write(&p, text).expect("write an example file");
    }
    dir.join("main.wk")
}

/// `(severity, code)` of the first diagnostic, read with the renderer's pinned JSON-lines prefix.
fn first_diagnostic(stderr: &str) -> Option<(String, String)> {
    let line = stderr.lines().next()?;
    let rest = line.strip_prefix("{\"severity\":\"")?;
    let (severity, rest) = rest.split_once('"')?;
    let rest = rest.strip_prefix(",\"code\":\"")?;
    let (code, _) = rest.split_once('"')?;
    Some((severity.to_string(), code.to_string()))
}

#[test]
fn a_multi_file_example_is_split_into_its_files() {
    let root = tmp(
        "explain_split_probe",
        "import geom;\nfn main() -> i32 { return 0; }\n// file: geom.wk\npub fn f() {}\n",
    );
    assert_eq!(
        std::fs::read_to_string(&root).unwrap(),
        "import geom;\nfn main() -> i32 { return 0; }\n"
    );
    let lib = root.with_file_name("geom.wk");
    assert_eq!(std::fs::read_to_string(lib).unwrap(), "pub fn f() {}\n");
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
        // A worked example is only useful if `--explain` shows it.
        for example in [e.bad, e.good].into_iter().flatten() {
            assert!(
                text.contains(example),
                "{}: worked example not printed",
                e.code
            );
        }
    }
}

#[test]
fn catalogue_examples_match_their_code() {
    let mut checked = 0usize;
    for e in all_explanations() {
        let warning = e.code.starts_with('W');
        if let Some(bad) = e.bad {
            let p = tmp(&format!("explain_bad_{}", e.code), bad);
            let out = wukongc()
                .arg("--error-format=json")
                .arg("--emit=check")
                .arg(&p)
                .output()
                .expect("spawn wukongc");
            let stderr = String::from_utf8_lossy(&out.stderr);
            if warning {
                assert!(
                    out.status.success(),
                    "{}: a warning's `bad` example must still compile:\n{stderr}",
                    e.code
                );
                assert_eq!(
                    first_diagnostic(&stderr),
                    Some(("warning".to_string(), e.code.to_string())),
                    "{}: `bad` example produced the wrong first diagnostic:\n{stderr}",
                    e.code
                );
            } else {
                assert!(
                    !out.status.success(),
                    "{}: the `bad` example compiled clean",
                    e.code
                );
                assert_eq!(
                    first_error_code(&stderr).as_deref(),
                    Some(e.code),
                    "{}: `bad` example produced the wrong first code:\n{stderr}",
                    e.code
                );
            }
            checked += 1;
        }
        if let Some(good) = e.good {
            let p = tmp(&format!("explain_good_{}", e.code), good);
            let out = wukongc()
                .arg("--error-format=json")
                .arg("--emit=check")
                .arg(&p)
                .output()
                .expect("spawn wukongc");
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(
                out.status.success(),
                "{}: the `good` example failed to compile:\n{stderr}",
                e.code,
            );
            if warning {
                assert!(
                    stderr.trim().is_empty(),
                    "{}: a warning's `good` example must compile with no diagnostic:\n{stderr}",
                    e.code
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "no catalogue examples were checked");
    eprintln!("checked {checked} catalogue example(s)");
}
