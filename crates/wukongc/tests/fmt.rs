//! `wukongc fmt` through the real binary: in-place rewriting, `--check`, `--stdout`, standard
//! input, the exit codes, and the MIR-identity gate.
//!
//! The gate is the end-to-end form of the formatter's guarantee. `wukong_fmt`'s own self-check
//! proves every output re-parses to the identical AST; this proves the compiler agrees: the whole
//! `tests/run` corpus (with its `lib/` imports) and `examples/` are formatted in a scratch copy,
//! and every root program's pre-optimization MIR (`--emit=mir-high -O0`) — and, for the examples,
//! the optimized MIR (`--emit=mir -O2`) — must be byte-identical before and after, exit code
//! included. Every mismatch is reported at once.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh, empty scratch directory for one test.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("wukong-fmt-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Run `wukongc args…` with `stdin`; (exit code, stdout, stderr).
fn wukongc(args: &[&str], stdin: &str) -> (Option<i32>, String, String) {
    use std::io::Write;
    let mut child = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .args(args)
        .env("NO_COLOR", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn wukongc");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(stdin.as_bytes())
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait for wukongc");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn s(p: &Path) -> String {
    p.display().to_string()
}

/// Deliberately unformatted, but well-formed and runnable: crowded statements, one-line blocks,
/// redundant parentheses and a missing `;` (W0030).
const UNFORMATTED: &str = "// EXPECT-OUT: 30
module scratch.fmt

fn sum(a: [i32; 4]) -> i32 { let mut s = 0; for i in 0..4 { s += (a[i]); } return s }


fn main() -> i32 {
  let a = [3, 6, 9, 12];   // the values
  print((sum(a)))
  return 0;
}
";

const FORMATTED: &str = "// EXPECT-OUT: 30
module scratch.fmt

fn sum(a: [i32; 4]) -> i32 {
    let mut s = 0;
    for i in 0..4 {
        s += a[i];
    }
    return s;
}

fn main() -> i32 {
    let a = [3, 6, 9, 12]; // the values
    print(sum(a));
    return 0;
}
";

#[test]
fn in_place_rewrite_is_check_clean_and_runs_the_same() {
    let dir = scratch("inplace");
    let f = dir.join("prog.wk");
    std::fs::write(&f, UNFORMATTED).unwrap();
    let before = wukongc(&["--run", &s(&f)], "");
    assert_eq!(before.0, Some(0), "{before:?}");

    let (code, out, err) = wukongc(&["fmt", &s(&f)], "");
    assert_eq!(code, Some(0), "stdout: {out}\nstderr: {err}");
    // A run of blank lines collapses to one; one is kept wherever the source had any.
    assert_eq!(std::fs::read_to_string(&f).unwrap(), FORMATTED);
    assert!(
        !dir.join("prog.wk.fmt.tmp").exists(),
        "temporary file left behind"
    );

    let (code, out, _) = wukongc(&["fmt", "--check", &s(&f)], "");
    assert_eq!(
        code,
        Some(0),
        "--check after a rewrite must be clean: {out}"
    );
    assert!(out.is_empty(), "{out}");
    let after = wukongc(&["--run", &s(&f)], "");
    assert_eq!(after.0, before.0);
    assert_eq!(after.1, before.1);
    assert_eq!(after.1.trim(), "30");
}

#[test]
fn check_lists_unformatted_files_and_writes_nothing() {
    let dir = scratch("check");
    let bad = dir.join("bad.wk");
    let good = dir.join("good.wk");
    std::fs::write(&bad, UNFORMATTED).unwrap();
    std::fs::write(&good, FORMATTED).unwrap();
    let (code, out, err) = wukongc(&["fmt", "--check", &s(&dir)], "");
    assert_eq!(code, Some(1), "stdout: {out}\nstderr: {err}");
    assert!(out.contains("bad.wk"), "{out}");
    assert!(!out.contains("good.wk"), "{out}");
    assert_eq!(std::fs::read_to_string(&bad).unwrap(), UNFORMATTED);
    assert_eq!(std::fs::read_to_string(&good).unwrap(), FORMATTED);
}

#[test]
fn a_syntax_error_exits_1_and_leaves_the_file() {
    let dir = scratch("syntax");
    let f = dir.join("broken.wk");
    let src = "fn main() -> i32 {\n    let x = ;\n    return 0;\n}\n";
    std::fs::write(&f, src).unwrap();
    let (code, _, err) = wukongc(&["fmt", &s(&f)], "");
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains("E02"),
        "the parser's code must be shown: {err}"
    );
    assert!(err.contains("broken.wk"), "{err}");
    assert_eq!(std::fs::read_to_string(&f).unwrap(), src);
}

#[test]
fn a_file_it_cannot_reprint_is_refused_with_exit_2() {
    // The AST does not keep a struct field's attributes, so reprinting would drop `@align`.
    let dir = scratch("refuse");
    let f = dir.join("attr.wk");
    let src = "struct S {\n    @align(16) x: f32,\n}\n";
    std::fs::write(&f, src).unwrap();
    let (code, _, err) = wukongc(&["fmt", &s(&f)], "");
    assert_eq!(code, Some(2), "{err}");
    assert!(
        err.contains("cannot format") && err.contains("without changing its meaning"),
        "{err}"
    );
    assert!(err.contains("file left unchanged"), "{err}");
    assert_eq!(std::fs::read_to_string(&f).unwrap(), src);
}

#[test]
fn stdin_and_stdout() {
    let (code, out, err) = wukongc(&["fmt", "-"], UNFORMATTED);
    assert_eq!(code, Some(0), "{err}");
    assert_eq!(out, FORMATTED);

    let dir = scratch("stdout");
    let f = dir.join("p.wk");
    std::fs::write(&f, UNFORMATTED).unwrap();
    let (code, out, _) = wukongc(&["fmt", "--stdout", &s(&f)], "");
    assert_eq!(code, Some(0));
    assert_eq!(out, FORMATTED);
    assert_eq!(
        std::fs::read_to_string(&f).unwrap(),
        UNFORMATTED,
        "--stdout rewrote the file"
    );
}

#[test]
fn usage_errors_exit_2_and_io_errors_exit_3() {
    for args in [
        vec!["fmt"],
        vec!["fmt", "--bogus", "x.wk"],
        vec!["fmt", "--check", "--stdout", "x.wk"],
        vec!["fmt", "--stdout", "a.wk", "b.wk"],
        vec!["fmt", "-", "a.wk"],
    ] {
        let (code, _, err) = wukongc(&args, "");
        assert_eq!(code, Some(2), "{args:?}: {err}");
        assert!(
            err.contains("wukongc fmt"),
            "{args:?} must print the fmt usage: {err}"
        );
    }
    let (code, out, _) = wukongc(&["fmt", "--help"], "");
    assert_eq!(code, Some(0));
    assert!(out.contains("--check") && out.contains("--stdout"), "{out}");

    let missing = scratch("missing").join("no_such_file.wk");
    let (code, _, err) = wukongc(&["fmt", &s(&missing)], "");
    assert_eq!(code, Some(3), "{err}");
}

/// Copy `from` (recursively) into `to`.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let p = e.path();
        let dst = to.join(e.file_name());
        if p.is_dir() {
            copy_tree(&p, &dst);
        } else {
            std::fs::copy(&p, &dst).unwrap();
        }
    }
}

/// The root `.wk` programs of a directory (not its `lib/`), sorted.
fn roots(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "wk"))
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

#[test]
fn formatting_the_corpus_leaves_every_programs_mir_unchanged() {
    let tmp = scratch("mir-gate");
    let repo = root();
    copy_tree(&repo.join("tests/run"), &tmp.join("run"));
    copy_tree(&repo.join("examples"), &tmp.join("examples"));

    let (code, out, err) = wukongc(&["fmt", &s(&tmp)], "");
    assert_eq!(code, Some(0), "formatting the corpus failed:\n{out}\n{err}");
    let (code, out, err) = wukongc(&["fmt", "--check", &s(&tmp)], "");
    assert_eq!(
        code,
        Some(0),
        "the formatted corpus is not a fixed point:\n{out}\n{err}"
    );

    // (original, formatted, flags) for every comparison.
    let mut jobs: Vec<(PathBuf, PathBuf, &'static [&'static str])> = Vec::new();
    for (sub, copy) in [("tests/run", "run"), ("examples", "examples")] {
        for name in roots(&repo.join(sub)) {
            let a = repo.join(sub).join(&name);
            let b = tmp.join(copy).join(&name);
            jobs.push((a.clone(), b.clone(), &["--emit=mir-high", "-O0"]));
            if sub == "examples" {
                jobs.push((a, b, &["--emit=mir", "-O2"]));
            }
        }
    }
    let next = AtomicUsize::new(0);
    let compiled = AtomicUsize::new(0);
    let mismatches: Mutex<Vec<String>> = Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..4 {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((a, b, flags)) = jobs.get(i) else {
                    break;
                };
                let run = |p: &Path| {
                    let mut args: Vec<String> = flags.iter().map(|f| f.to_string()).collect();
                    args.push(s(p));
                    let refs: Vec<&str> = args.iter().map(|x| x.as_str()).collect();
                    let (c, o, _) = wukongc(&refs, "");
                    (c, o)
                };
                let (ra, rb) = (run(a), run(b));
                if ra.0 == Some(0) && !ra.1.is_empty() {
                    compiled.fetch_add(1, Ordering::Relaxed);
                }
                if ra != rb {
                    let line =
                        ra.1.lines()
                            .zip(rb.1.lines())
                            .position(|(x, y)| x != y)
                            .unwrap_or(0);
                    mismatches.lock().unwrap().push(format!(
                        "{} {:?}: exit {:?} vs {:?}; first differing MIR line {}: `{}` vs `{}`",
                        a.display(),
                        flags,
                        ra.0,
                        rb.0,
                        line + 1,
                        ra.1.lines().nth(line).unwrap_or(""),
                        rb.1.lines().nth(line).unwrap_or("")
                    ));
                }
            });
        }
    });
    let mismatches = mismatches.into_inner().unwrap();
    let compiled = compiled.into_inner();
    eprintln!(
        "MIR gate: {} comparisons ({} produced MIR), {} mismatches",
        jobs.len(),
        compiled,
        mismatches.len()
    );
    // Not vacuous: two runs that both fail the same way would compare equal, so nearly every
    // original must actually have compiled to MIR.
    assert!(
        compiled * 100 >= jobs.len() * 95,
        "only {compiled} of {} originals produced MIR; the gate compares nothing",
        jobs.len()
    );
    assert!(
        mismatches.is_empty(),
        "{} program(s) changed MIR when formatted:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    let _ = std::fs::remove_dir_all(&tmp);
}
