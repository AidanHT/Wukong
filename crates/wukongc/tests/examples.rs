//! Execution guard for the `examples/*.wk` showcase programs.
//!
//! The `emit` suite only smoke-tests the `tokens`/`ast` stages for examples, and the `run` suite
//! covers `tests/run/*.wk` — so the example programs themselves were never actually *executed*
//! under test. This suite runs every example through the real binary and enforces the project's
//! hard invariant on them: the interpreter oracle and the Cranelift native backend must agree on
//! stdout and exit code, and native optimization must be observationally invariant.
//!
//! Agreement alone is not enough, though, and that was the hole: **an example that fails to
//! compile satisfies both agreement checks**, because both backends fail identically. Seven of the
//! sixteen examples currently do not execute (three hit `C0001`, one has no `main`, three take a
//! data-absent early return), so nearly half the corpus was contributing only "both backends broke
//! the same way" — a change that stopped `examples/gpt2.wk` lowering would have shipped green.
//!
//! Every example therefore carries a declared [`Class`] saying what running it must produce, and
//! anything not listed defaults to [`Class::Runs`]: a new example must execute, or be declared.

use std::path::{Path, PathBuf};
use std::process::Command;

fn examples_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/wukongc; the examples live at <repo>/examples.
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
        .filter(|p| p.extension().map(|x| x == "wk").unwrap_or(false))
        .collect();
    programs.sort();
    assert!(!programs.is_empty(), "no .wk examples in {}", dir.display());
    programs
}

/// What running an example is expected to produce. The point of declaring this is that a silent
/// slide from `Runs` into any of the others is a regression, and the agreement checks cannot see it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// Executes: exit 0 and non-empty stdout.
    Runs,
    /// A construct codegen does not lower yet: exit 1 with `error[C0001]`.
    NotLowered,
    /// A constants-only module with no `main`: exit 1 with `no entry function`.
    NoEntry,
    /// Reads the 500 MB GPT-2 weight blob and takes an early `print(-1); return 1` when it is not
    /// present. Documented native-only (the interpreter cannot hold 124M params), and two of them
    /// also print a wall-clock microsecond count, so their stdout is only a valid differential
    /// fixture while the blob is unreachable — a precondition this suite now checks rather than
    /// assumes.
    DataGuarded,
}

fn class_of(name: &str) -> Class {
    match name {
        "matmul.wk" | "softmax.wk" | "vadd.wk" => Class::NotLowered,
        "gpt2_config.wk" => Class::NoEntry,
        "gpt2_infer.wk" | "gpt2_forward_bench.wk" | "gpt2_forward_bench_par.wk" => {
            Class::DataGuarded
        }
        _ => Class::Runs,
    }
}

/// Is the GPT-2 weight blob reachable from the test process's CWD (which is `crates/wukongc`)? The
/// `DataGuarded` examples open it through a RELATIVE path, so this decides whether they take their
/// early return or run the real model.
fn data_blob_reachable() -> bool {
    Path::new("data/gpt2/gpt2_124m_weights.bin").exists()
}

/// Run an example at a given optimization level on a chosen backend (`None` = the default
/// interpreter, `Some("native")` = the Cranelift JIT), returning (exit_code, stdout, stderr).
fn run_backend(path: &Path, opt: &str, backend: Option<&str>) -> (Option<i32>, String, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    cmd.arg("--run").arg(opt);
    if let Some(b) = backend {
        cmd.arg(format!("--backend={b}"));
    }
    let output = cmd.arg(path).output().expect("failed to spawn wukongc");
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// The observable pair the agreement checks compare.
fn observable(r: &(Option<i32>, String, String)) -> (Option<i32>, String) {
    (r.0, r.1.clone())
}

/// Every example must produce what its class says it produces. Without this, the two agreement
/// tests below pass for an example that stopped executing entirely.
#[test]
fn examples_do_what_their_class_says() {
    let blob = data_blob_reachable();
    let mut failures = Vec::new();
    let mut counts = [0usize; 4];
    for p in &collect_examples() {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let class = class_of(&name);
        let (exit, stdout, stderr) = run_backend(p, "-O2", Some("native"));
        match class {
            Class::Runs => {
                counts[0] += 1;
                if exit != Some(0) || stdout.trim().is_empty() {
                    failures.push(format!(
                        "{name}: declared Runs but exited {exit:?} with stdout {:?}\n--- stderr ---\n{stderr}",
                        stdout.trim()
                    ));
                }
            }
            Class::NotLowered => {
                counts[1] += 1;
                if exit != Some(1) || !stderr.contains("error[C0001]") {
                    failures.push(format!(
                        "{name}: declared NotLowered (C0001) but exited {exit:?}; if it lowers now, \
                         move it to Class::Runs\n--- stderr ---\n{stderr}"
                    ));
                }
            }
            Class::NoEntry => {
                counts[2] += 1;
                if exit != Some(1) || !stderr.contains("no entry function") {
                    failures.push(format!(
                        "{name}: declared NoEntry but exited {exit:?}\n--- stderr ---\n{stderr}"
                    ));
                }
            }
            Class::DataGuarded => {
                counts[3] += 1;
                if blob {
                    // The blob is present, so the program ran the real model instead of taking the
                    // early return; its stdout then carries a wall-clock timing for two of the
                    // three. Nothing to assert beyond "it did not crash".
                    continue;
                }
                if exit != Some(1) || stdout.trim() != "-1" {
                    failures.push(format!(
                        "{name}: declared DataGuarded, so with data/gpt2/ absent it must print -1 \
                         and exit 1; got {exit:?} with stdout {:?}\n--- stderr ---\n{stderr}",
                        stdout.trim()
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} example(s) did not match their declared class:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!(
        "examples: {} run, {} not-lowered (C0001), {} no-entry, {} data-guarded (blob {})",
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        if blob { "PRESENT" } else { "absent" }
    );
}

/// Every example must run identically on the interpreter oracle and the native backend, at both
/// -O0 and -O2 (the hard invariant). All mismatches are collected before failing.
///
/// The `DataGuarded` examples are excluded when the weight blob is reachable: they are documented
/// native-only (the interpreter cannot hold 124M params and hits its step limit), and two of them
/// print a wall-clock microsecond count, which is not a comparable observable. While the blob is
/// absent — the normal state, and the one `examples_do_what_their_class_says` pins — they take the
/// same early return on both backends and are compared like everything else.
#[test]
fn examples_native_matches_interpreter() {
    let blob = data_blob_reachable();
    let mut mismatches = Vec::new();
    let mut skipped = Vec::new();
    for p in &collect_examples() {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if blob && class_of(&name) == Class::DataGuarded {
            skipped.push(name);
            continue;
        }
        for opt in ["-O0", "-O2"] {
            let interp = observable(&run_backend(p, opt, None));
            let native = observable(&run_backend(p, opt, Some("native")));
            if interp != native {
                mismatches.push(format!(
                    "{name} @ {opt}:\n    interp = {interp:?}\n    native = {native:?}"
                ));
            }
        }
    }
    if !skipped.is_empty() {
        eprintln!(
            "examples_native_matches_interpreter: data/gpt2/ is reachable from the test CWD, so \
             the native-only, timing-printing examples are excluded: {}",
            skipped.join(", ")
        );
    }
    assert!(
        mismatches.is_empty(),
        "example interpreter/native divergence ({} case(s)):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Every example must be observationally invariant under optimization on the interpreter: -O0 must
/// match -O1/-O2/-O3 on stdout and exit code. Same `DataGuarded` exclusion as above, for the same
/// reason: a wall-clock token differs between two runs of the *same* binary, let alone two opt
/// levels.
#[test]
fn examples_optimization_is_observationally_invariant() {
    let blob = data_blob_reachable();
    let mut mismatches = Vec::new();
    for p in &collect_examples() {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if blob && class_of(&name) == Class::DataGuarded {
            continue;
        }
        let base = observable(&run_backend(p, "-O0", None));
        for level in ["-O1", "-O2", "-O3"] {
            let other = observable(&run_backend(p, level, None));
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
