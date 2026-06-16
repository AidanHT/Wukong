//! End-to-end CLI tests: compile and run each `tests/run/*.mer` program through the real
//! `mercuryc` binary and check its stdout and exit code against directives embedded in the file.
//!
//! Directive syntax (in leading `//` comments):
//!   `// RUN: <extra cli args>`     extra flags passed to mercuryc (default: `--run`)
//!   `// EXPECT-EXIT: <n>`          required process exit code
//!   `// EXPECT-OUT: <line>`        one expected stdout line, in order
//!
//! This exercises the entire pipeline (lex → parse → sema → MIR → opt → interpret) exactly as a
//! user would, with zero external dependencies.

use std::path::{Path, PathBuf};
use std::process::Command;

struct Expect {
    run_args: Vec<String>,
    exit: Option<i32>,
    out_lines: Vec<String>,
}

fn parse_directives(src: &str) -> Expect {
    let mut run_args = vec!["--run".to_string()];
    let mut exit = None;
    let mut out_lines = Vec::new();
    for line in src.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("// RUN:") {
            run_args = rest.split_whitespace().map(|s| s.to_string()).collect();
        } else if let Some(rest) = line.strip_prefix("// EXPECT-EXIT:") {
            exit = Some(rest.trim().parse().expect("EXPECT-EXIT must be an integer"));
        } else if let Some(rest) = line.strip_prefix("// EXPECT-OUT:") {
            out_lines.push(rest.trim().to_string());
        }
    }
    Expect { run_args, exit, out_lines }
}

fn run_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/mercuryc; the suite lives at <repo>/tests/run.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("run")
}

fn check_program(path: &Path) {
    let src = std::fs::read_to_string(path).unwrap();
    let expect = parse_directives(&src);

    let output = Command::new(env!("CARGO_BIN_EXE_mercuryc"))
        .args(&expect.run_args)
        .arg(path)
        .output()
        .expect("failed to spawn mercuryc");

    let name = path.file_name().unwrap().to_string_lossy();
    let stdout = String::from_utf8_lossy(&output.stdout);

    if let Some(code) = expect.exit {
        let actual = output.status.code();
        assert_eq!(
            actual,
            Some(code),
            "{name}: exit code mismatch (stderr: {})",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    if !expect.out_lines.is_empty() {
        let actual: Vec<&str> = stdout.lines().collect();
        assert_eq!(
            actual, expect.out_lines,
            "{name}: stdout mismatch\n--- stderr ---\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn run_suite() {
    let dir = run_dir();
    let mut programs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "mer").unwrap_or(false))
        .collect();
    programs.sort();
    assert!(!programs.is_empty(), "no .mer programs found in {}", dir.display());

    for p in &programs {
        check_program(p);
    }
    eprintln!("ran {} e2e program(s)", programs.len());
}
