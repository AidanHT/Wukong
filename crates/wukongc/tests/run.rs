//! End-to-end CLI tests: compile and run each `tests/run/*.wk` program through the real
//! `wukongc` binary and check its stdout and exit code against directives embedded in the file.
//!
//! Directive syntax (in leading `//` comments):
//!   `// RUN: <extra cli args>`     extra flags passed to wukongc (default: `--run`)
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
    Expect {
        run_args,
        exit,
        out_lines,
    }
}

fn run_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/wukongc; the suite lives at <repo>/tests/run.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("run")
}

fn check_program(path: &Path) {
    let src = std::fs::read_to_string(path).unwrap();
    let expect = parse_directives(&src);

    let output = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .args(&expect.run_args)
        .arg(path)
        .output()
        .expect("failed to spawn wukongc");

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
            actual,
            expect.out_lines,
            "{name}: stdout mismatch\n--- stderr ---\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn collect_programs() -> Vec<PathBuf> {
    let dir = run_dir();
    let mut programs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "wk").unwrap_or(false))
        .collect();
    programs.sort();
    assert!(
        !programs.is_empty(),
        "no .wk programs found in {}",
        dir.display()
    );
    programs
}

#[test]
fn run_suite() {
    let programs = collect_programs();
    for p in &programs {
        check_program(p);
    }
    eprintln!("ran {} e2e program(s)", programs.len());
}

/// Differential hardening: optimization must never change observable behavior. For every program,
/// running at -O0 and at -O1/-O2/-O3 must produce identical stdout and exit code.
#[test]
fn optimization_is_observationally_invariant() {
    for p in &collect_programs() {
        let base = run_at(p, "-O0");
        for level in ["-O1", "-O2", "-O3"] {
            let other = run_at(p, level);
            let name = p.file_name().unwrap().to_string_lossy();
            assert_eq!(
                base, other,
                "{name}: {level} output differs from -O0\n-O0: {base:?}\n{level}: {other:?}"
            );
        }
    }
}

/// Run a program at a given optimization level on the interpreter, returning (exit_code, stdout).
fn run_at(path: &Path, opt: &str) -> (Option<i32>, String) {
    run_backend(path, opt, None)
}

/// Run a program at a given optimization level on a chosen execution backend (`None` = the default
/// interpreter, `Some("native")` = the Cranelift JIT), returning (exit_code, stdout).
fn run_backend(path: &Path, opt: &str, backend: Option<&str>) -> (Option<i32>, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    cmd.arg("--run").arg(opt);
    if let Some(b) = backend {
        cmd.arg(format!("--backend={b}"));
    }
    let output = cmd.arg(path).output().expect("failed to spawn wukongc");
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
    )
}

/// The hard invariant, extended to native: the Cranelift backend must agree with the interpreter
/// oracle (stdout + exit code) for every program at every optimization level. The interpreter is
/// the differential oracle, and `optimization_is_observationally_invariant` above only re-runs the
/// *interpreter* across opt levels — so this is the guard that the *native* path never silently
/// diverges from the oracle. (All mismatches are collected so one failure reports the full set.)
#[test]
fn native_matches_interpreter() {
    let mut mismatches = Vec::new();
    for p in &collect_programs() {
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
        "interpreter/native divergence ({} case(s)):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Native optimization must be observationally invariant too: the Cranelift backend at -O0 must
/// produce identical stdout and exit code to -O1/-O2/-O3 (the interpreter analogue is
/// `optimization_is_observationally_invariant`).
#[test]
fn native_optimization_is_observationally_invariant() {
    let mut mismatches = Vec::new();
    for p in &collect_programs() {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let base = run_backend(p, "-O0", Some("native"));
        for level in ["-O1", "-O2", "-O3"] {
            let other = run_backend(p, level, Some("native"));
            if base != other {
                mismatches.push(format!(
                    "{name}: native {level} differs from -O0:\n    -O0   = {base:?}\n    {level} = {other:?}"
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "native optimization divergence ({} case(s)):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
