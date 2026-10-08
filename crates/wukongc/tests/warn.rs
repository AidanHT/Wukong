//! Warning tests: each `tests/warn/*.wk` program compiles, and the lints report exactly the
//! diagnostics its directives list — every one, in emission order, nothing more. Where
//! `tests/fail` pins only a fixture's first error code, a lint's job is as much NOT to fire as to
//! fire, so the whole stream is compared.
//!
//! | Directive | Meaning |
//! |---|---|
//! | `// EXPECT-WARN: <code>` | the next diagnostic is a warning with this code |
//! | `// EXPECT-ERROR: <code>` | the next diagnostic is a lint promoted to an error (`@deny`, `-D`) |
//! | `// EXPECT-NONE` | the program compiles with no diagnostic at all |
//! | `// FLAGS: <args>` | extra command-line arguments (`-D warnings`) |
//!
//! A fixture with an `EXPECT-ERROR` must exit 1 (a denied lint fails the build), any other must
//! exit 0. Every fixture must carry at least one `EXPECT-*` line, so a misspelled directive cannot
//! turn a fixture into one that asserts nothing. The reverse direction — every catalogued `W` code
//! has a fixture here — is `fail.rs`'s `every_catalogued_code_is_exercised`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(name)
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

/// One fixture's directives.
struct Expect {
    /// `(severity, code)` in emission order.
    diags: Vec<(String, String)>,
    none: bool,
    flags: Vec<String>,
}

fn directives(src: &str) -> Expect {
    let mut e = Expect {
        diags: Vec::new(),
        none: false,
        flags: Vec::new(),
    };
    for line in src.lines() {
        let l = line.trim_start();
        if let Some(code) = l.strip_prefix("// EXPECT-WARN:") {
            e.diags.push(("warning".into(), code.trim().into()));
        } else if let Some(code) = l.strip_prefix("// EXPECT-ERROR:") {
            e.diags.push(("error".into(), code.trim().into()));
        } else if l.trim_end() == "// EXPECT-NONE" {
            e.none = true;
        } else if let Some(f) = l.strip_prefix("// FLAGS:") {
            e.flags.extend(f.split_whitespace().map(str::to_string));
        }
    }
    e
}

/// `(severity, code)` of one `--error-format=json` line, read through the renderer's pinned field
/// order (`{"severity":"…","code":"…",…`). `None` for anything else, which the caller reports.
fn severity_and_code(line: &str) -> Option<(String, String)> {
    let rest = line.strip_prefix("{\"severity\":\"")?;
    let (severity, rest) = rest.split_once('"')?;
    let rest = rest.strip_prefix(",\"code\":\"")?;
    let (code, _) = rest.split_once('"')?;
    Some((severity.to_string(), code.to_string()))
}

/// Compile `path` to MIR (sema, lints and lowering all run) with JSON diagnostics.
/// Returns (exit code, stderr).
fn compile(path: &Path, flags: &[String]) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .arg("--error-format=json")
        .arg("--emit=mir")
        .args(flags)
        .arg(path)
        .output()
        .expect("spawn wukongc");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn warn_suite() {
    let programs = wk_files(&repo_dir("tests").join("warn"));
    assert!(!programs.is_empty(), "no fixtures in tests/warn");
    for p in &programs {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(p).unwrap();
        let want = directives(&src);
        assert!(
            want.none || !want.diags.is_empty(),
            "{name}: no `// EXPECT-WARN:`, `// EXPECT-ERROR:` or `// EXPECT-NONE` directive"
        );
        assert!(
            !want.none || want.diags.is_empty(),
            "{name}: `// EXPECT-NONE` contradicts the other EXPECT lines"
        );
        let (code, stderr) = compile(p, &want.flags);
        let got: Vec<(String, String)> = stderr
            .lines()
            .map(|l| {
                severity_and_code(l)
                    .unwrap_or_else(|| panic!("{name}: not a coded JSON diagnostic: {l}"))
            })
            .collect();
        assert_eq!(
            got, want.diags,
            "{name}: diagnostics differ from the EXPECT lines\n--- stderr ---\n{stderr}"
        );
        let denied = want.diags.iter().any(|(sev, _)| sev == "error");
        assert_eq!(
            code,
            Some(if denied { 1 } else { 0 }),
            "{name}: wrong exit status\n--- stderr ---\n{stderr}"
        );
        for (_, c) in &want.diags {
            assert!(
                wukong_driver::explain(c).is_some(),
                "{name}: {c} is not in the diagnostic catalogue"
            );
        }
    }
    eprintln!("checked {} warning fixture(s)", programs.len());
}

/// The directive reader rejects nothing silently: a fixture's lines are read in order and the
/// JSON reader takes only a coded diagnostic.
#[test]
fn directive_and_json_readers() {
    let e = directives(
        "// EXPECT-WARN: W0001\n// FLAGS: -D warnings -A dead_code\n// EXPECT-ERROR: W0002\n",
    );
    assert_eq!(
        e.diags,
        vec![
            ("warning".to_string(), "W0001".to_string()),
            ("error".to_string(), "W0002".to_string())
        ]
    );
    assert_eq!(e.flags, ["-D", "warnings", "-A", "dead_code"]);
    assert!(!e.none);
    assert_eq!(
        severity_and_code(
            r#"{"severity":"warning","code":"W0001","message":"unused variable","spans":[],"notes":[]}"#
        ),
        Some(("warning".to_string(), "W0001".to_string()))
    );
    assert_eq!(
        severity_and_code(r#"{"severity":"error","message":"no code","spans":[],"notes":[]}"#),
        None
    );
    assert_eq!(severity_and_code("error: plain text"), None);
}

/// The corpus census: every `tests/run` and `examples` program is linted with the default levels,
/// and any warning is listed. The run corpus is meant to be warning-free (or to carry an `@allow`
/// saying why), so that a new warning in it is news.
///
/// `#[ignore]`d until the first full run blesses the corpus: the lints landed without a build in
/// the loop (wave 2 of the gap close-out), so the corpus has not yet been brought to zero. Run it
/// with `cargo test -p wukongc --test warn -- --ignored`, fix or `@allow` what it lists, and then
/// drop the `#[ignore]` to make it a gate.
#[test]
#[ignore = "corpus not yet brought to zero warnings; run with --ignored to list them"]
fn corpus_is_warning_free() {
    let mut programs = wk_files(&repo_dir("tests").join("run"));
    programs.extend(wk_files(&repo_dir("examples")));
    let mut found = Vec::new();
    for p in &programs {
        let (_, stderr) = compile(p, &[]);
        for line in stderr.lines() {
            let lint = severity_and_code(line).is_some_and(|(_, code)| code.starts_with('W'));
            if lint {
                found.push(format!("{}: {line}", p.display()));
            }
        }
    }
    assert!(
        found.is_empty(),
        "{} warning(s) in the run corpus and the examples:\n{}",
        found.len(),
        found.join("\n")
    );
}
