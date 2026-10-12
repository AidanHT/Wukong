//! `--emit=doc` and `--doctest` through the real binary, on the fixtures in `tests/doc/`.
//!
//! Every expected text here — `tests/doc/geometry.md`, the report lines, the line and column a
//! failing example is reported at — was written by hand from the fixture's source and the rules
//! in `wukong_fmt::doc` and `wukong_driver::doctest`, never pasted from the compiler's output.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Run `wukongc` from the repository root (so the paths it prints are the relative ones given).
fn wukongc(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .current_dir(repo_root())
        .args(args)
        .arg("--color=never")
        .output()
        .expect("failed to spawn wukongc");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8(out.stdout).expect("stdout is UTF-8"),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Line endings as checked out: compare text with `\n` only.
fn lf(s: &str) -> String {
    s.replace("\r\n", "\n")
}

// ---- --emit=doc ----

/// The page for `tests/doc/geometry.wk` is `tests/doc/geometry.md`, written by hand.
#[test]
fn emit_doc_prints_the_hand_written_page() {
    let want = std::fs::read_to_string(repo_root().join("tests/doc/geometry.md")).unwrap();
    let (code, out, err) = wukongc(&["--emit=doc", "tests/doc/geometry.wk"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(lf(&out), lf(&want), "\n--- got ---\n{out}");
}

/// `doc` is one stage of a list like any other — its artifact under its header is the
/// single-stage page — and with `-o <stem>` it is written to `<stem>.md`.
#[test]
fn emit_doc_composes_with_a_stage_list() {
    let (_, alone, _) = wukongc(&["--emit=doc", "tests/doc/geometry.wk"]);
    let (code, out, err) = wukongc(&["--emit=mir,doc,tokens", "tests/doc/geometry.wk"]);
    assert_eq!(code, 0, "{err}");
    let headers: Vec<&str> = out.lines().filter(|l| l.starts_with("==> ")).collect();
    assert_eq!(headers, ["==> tokens <==", "==> doc <==", "==> mir <=="]);
    let doc = &out[out.find("==> doc <==\n").unwrap() + "==> doc <==\n".len()..];
    let doc = &doc[..doc.find("==> mir <==").unwrap()];
    assert_eq!(doc, alone);

    let dir = std::env::temp_dir().join(format!("wukong-doc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let stem = dir.join("geo");
    let stem = stem.to_str().unwrap();
    let (code, out, err) = wukongc(&["--emit=ast,doc", "-o", stem, "tests/doc/geometry.wk"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.is_empty(), "{out}");
    let written = std::fs::read_to_string(dir.join("geo.md")).unwrap();
    assert_eq!(written, alone);
    assert!(err.contains("geo.md"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A file whose doc comment documents nothing has no page: E0210, exit 1, nothing on stdout. So
/// has any file that does not parse.
#[test]
fn emit_doc_needs_a_clean_parse() {
    let (code, out, err) = wukongc(&["--emit=doc", "tests/fail/doc_comment_documents_nothing.wk"]);
    assert_eq!(code, 1);
    assert!(out.is_empty(), "{out}");
    assert!(
        err.contains("error[E0210]") && err.contains("doc_comment_documents_nothing.wk:9:5"),
        "{err}"
    );
}

/// `doc` reaches every example and every `tests/run` program (each with its module title).
#[test]
fn emit_doc_succeeds_on_the_corpus() {
    let root = repo_root();
    let mut files = Vec::new();
    for dir in ["examples", "tests/run"] {
        for e in std::fs::read_dir(root.join(dir)).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_some_and(|x| x == "wk") {
                files.push(p);
            }
        }
    }
    files.sort();
    assert!(files.len() > 300, "corpus not found");
    for f in &files {
        let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
            .arg("--emit=doc")
            .arg(f)
            .output()
            .expect("failed to spawn wukongc");
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success() && text.starts_with("# "),
            "{}: {}\n{}",
            f.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

// ---- --doctest ----

const GEOMETRY_REPORT: &str = "\
running 7 doctests in tests/doc/geometry.wk
doctest tests/doc/geometry.wk:3 (the file) ... ok
doctest tests/doc/geometry.wk:18 (fn area_of) ... ok
doctest tests/doc/geometry.wk:29 (fn double) ... ok
doctest tests/doc/geometry.wk:35 (fn double) ... ok
doctest tests/doc/geometry.wk:56 (fn spin) ... ok
doctest tests/doc/geometry.wk:62 (fn spin) ... ignored
doctest tests/doc/geometry.wk:89 (fn larger) ... ok

doctest result: ok. 6 passed; 0 failed; 1 ignored
";

/// Every kind of passing example — the file's, one that runs, `compile_fail` with a code, one with
/// its own `main`, `no_run`, `ignore`, one whose own item sits between its statements — on the
/// interpreter and the native JIT, at -O0 and -O2. The `text` fence is not an example.
#[test]
fn doctests_pass_run_no_run_compile_fail_and_ignore() {
    for args in [
        &["--doctest"][..],
        &["--doctest", "-O2"][..],
        &["--doctest", "--backend=native"][..],
        &["--doctest", "--backend=native", "-O2"][..],
    ] {
        let mut a = args.to_vec();
        a.push("tests/doc/geometry.wk");
        let (code, out, err) = wukongc(&a);
        assert_eq!(code, 0, "{args:?}: {err}");
        assert_eq!(lf(&out), GEOMETRY_REPORT, "{args:?}");
        assert!(err.is_empty(), "{args:?}: {err}");
    }
}

const FAILURES_REPORT: &str = "\
running 7 doctests in tests/doc/doctest_failures.wk
doctest tests/doc/doctest_failures.wk:7 (fn one) ... FAILED
doctest tests/doc/doctest_failures.wk:13 (fn one) ... FAILED
doctest tests/doc/doctest_failures.wk:19 (fn one) ... FAILED
doctest tests/doc/doctest_failures.wk:25 (fn one) ... FAILED
doctest tests/doc/doctest_failures.wk:31 (fn one) ... FAILED
doctest tests/doc/doctest_failures.wk:37 (fn one) ... FAILED
doctest tests/doc/doctest_failures.wk:43 (fn one) ... ok

failures:
    tests/doc/doctest_failures.wk:7 (fn one)
    tests/doc/doctest_failures.wk:13 (fn one)
    tests/doc/doctest_failures.wk:19 (fn one)
    tests/doc/doctest_failures.wk:25 (fn one)
    tests/doc/doctest_failures.wk:31 (fn one)
    tests/doc/doctest_failures.wk:37 (fn one)

doctest result: FAILED. 1 passed; 6 failed; 0 ignored
";

/// Each way an example fails is reported as FAILED, with its reason on stderr, and the run exits
/// 1: a failed assertion, a non-zero exit, a compile error in an example that must run (at the
/// line and column of the doc comment), a `compile_fail` example that compiles, one that fails
/// with another code, and an unknown tag.
#[test]
fn every_failing_doctest_is_reported() {
    for backend in ["--backend=interp", "--backend=native"] {
        let (code, out, err) = wukongc(&["--doctest", backend, "tests/doc/doctest_failures.wk"]);
        assert_eq!(code, 1, "{backend}: {err}");
        assert_eq!(lf(&out), FAILURES_REPORT, "{backend}");
        for want in [
            "doctest_failures.wk:7 (fn one): assertion failed",
            "doctest_failures.wk:13 (fn one): it exited with status 7",
            "doctest_failures.wk:19 (fn one): failed to compile",
            // `undefined_name` on the code line under the fence at line 19: `/// let x: i32 = `
            // puts it at column 18.
            "error[E0301]",
            "--> tests/doc/doctest_failures.wk:20:18",
            "doctest_failures.wk:25 (fn one): compiled, but it is marked `compile_fail`",
            "doctest_failures.wk:31 (fn one): failed to compile, but not with E0503",
            "--> tests/doc/doctest_failures.wk:32:18",
            "doctest_failures.wk:37 (fn one): unknown tag `compile-fail`",
        ] {
            assert!(
                err.contains(want),
                "{backend}: stderr lacks {want:?}:\n{err}"
            );
        }
    }
}

/// The doc examples added to `examples/` pass; a file with none reports none.
#[test]
fn the_examples_doctests_pass() {
    let (code, out, err) = wukongc(&["--doctest", "examples/relu.wk"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(
        lf(&out),
        "running 2 doctests in examples/relu.wk\n\
         doctest examples/relu.wk:11 (the file) ... ok\n\
         doctest examples/relu.wk:23 (fn relu) ... ok\n\
         \n\
         doctest result: ok. 2 passed; 0 failed; 0 ignored\n"
    );
    let (code, out, err) = wukongc(&["--doctest", "--backend=native", "examples/saxpy_array.wk"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(
        lf(&out),
        "running 1 doctest in examples/saxpy_array.wk\n\
         doctest examples/saxpy_array.wk:11 (fn main) ... ok\n\
         \n\
         doctest result: ok. 1 passed; 0 failed; 0 ignored\n"
    );
    let (code, out, _) = wukongc(&["--doctest", "examples/hello.wk"]);
    assert_eq!(code, 0);
    assert_eq!(
        lf(&out),
        "running 0 doctests in examples/hello.wk\n\ndoctest result: ok. 0 passed; 0 failed; 0 ignored\n"
    );
}

/// A file that does not parse runs no example: its errors, exit 1.
#[test]
fn doctest_needs_a_clean_parse() {
    let (code, out, err) = wukongc(&["--doctest", "tests/fail/doc_comment_documents_nothing.wk"]);
    assert_eq!(code, 1);
    assert!(out.is_empty(), "{out}");
    assert!(err.contains("error[E0210]"), "{err}");
}

/// What `--doctest` cannot honour is refused by name (exit 2), not ignored.
#[test]
fn doctest_refuses_flags_it_cannot_honour() {
    for (flag, what) in [
        ("--run", "`--run`"),
        ("--emit=mir", "`--emit`"),
        ("--backend=gpu", "GPU"),
        ("-Dwarnings", "`-A`/`-W`/`-D`"),
        ("--train", "`--train`"),
    ] {
        let (code, _, err) = wukongc(&["--doctest", flag, "tests/doc/geometry.wk"]);
        assert_eq!(code, 2, "{flag}");
        assert!(
            err.contains("`--doctest` does not take") && err.contains(what),
            "{flag}: {err}"
        );
    }
}
