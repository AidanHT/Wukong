//! The corpus-wide recognizer dispatch census: which runtime kernels each `tests/run` fixture
//! actually dispatches at `-O2`, pinned in `tests/recognizer-census.txt`.
//!
//! ## The hole this closes
//!
//! Recognized kernels are **gate-blind**. A recognizer fires identically on the interpreter and on
//! native at every `-O` level — both call the same `wukong_runtime` symbol — so
//! `native_matches_interpreter`, both opt-invariance gates and `determinism.rs` are structurally
//! incapable of seeing whether a recognizer fired, stopped firing, or was replaced by a different
//! kernel. `recognize.rs` is the only gate that can see a recognizer at all, and by its own design
//! note it pins **one fixture per cascade head** — five sentinels. The corpus dispatches over a
//! hundred distinct symbols across more than half its fixtures, and most of those symbols appear
//! in exactly one fixture, so for them there is not even a second program to notice.
//!
//! That is the mechanism by which a whole performance workstream evaporates between campaigns: a
//! recognizer narrows, a tuned kernel silently becomes a scalar loop, every gate stays green, and
//! the only symptom is a benchmark number nobody re-ran.
//!
//! ## What this test does and does not prove
//!
//! It proves **change detection over the whole dispatch surface**: if any fixture gains or loses a
//! kernel call at `-O2`, this fails and names the fixture and the symbol.
//!
//! It does **not** prove any kernel is correct. The census is a snapshot of compiler behaviour, so
//! it can only ever say "this changed", never "this is right". Value correctness comes from the
//! fixtures' own hand-derived expectations and from each kernel's in-crate scalar-twin and
//! f64-reference tests. Reading a green census as evidence of correctness is the mistake this
//! comment exists to prevent.
//!
//! ## Updating it
//!
//! A recognizer change is *supposed* to move this file. Regenerate with
//!
//! ```text
//! WUKONG_UPDATE_CENSUS=1 cargo test -p wukongc --test recognizer_census
//! ```
//!
//! which rewrites `tests/recognizer-census.txt` and then **fails on purpose**, so the diff cannot
//! be blessed without somebody looking at it. Re-run without the variable once the diff has been
//! read, and state in the commit body which symbols moved and why.
//!
//! The technique is the `--bless` / `update_test_checks.py` snapshot workflow used by rustc's
//! ui-test suite and LLVM's FileCheck tests; the deliberate failure on rewrite is the local
//! addition, because a snapshot tool that silently succeeds is a way to make a red gate green.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/mod.rs"]
mod common;

/// Environment variables that change what the compiler emits. The census child must not inherit
/// any of them, or a developer with one exported produces a census that disagrees with CI for a
/// reason nothing in the diff explains.
const COMPILER_ENV: &[&str] = &[
    "WUKONG_NO_RECOGNIZE",
    "WUKONG_NO_VECTORIZE",
    "WUKONG_NO_UNROLL",
    "WUKONG_NO_IV_RAMP",
    "WUKONG_NO_SLP",
    "WUKONG_NO_LANE_SERIAL",
    "WUKONG_NO_OUTER_VEC",
    "WUKONG_P4_NO_256",
    "WUKONG_VEC256_MIN_TRIP",
    "WUKONG_VEC_TRACE",
    "WUKONG_SLP_TRACE",
    // Below `avx2` it turns off the raw-AVX2 loop kernels (`veckernel*N` on Windows).
    "WUKONG_MAX_ISA",
];

/// Written into the census line of a fixture whose `--emit=mir -O2` did not exit 0. Distinct from
/// "no kernels", which is also an empty symbol list: without the marker, a fixture that stopped
/// compiling would render identically to a fixture that never dispatched anything.
const COMPILE_FAILED: &str = "<COMPILE-FAILED>";

/// Vacuity floors. The census file is the tight check; these guard the case where the file has
/// been regenerated from a broken tree and blessed, which would otherwise leave a green gate over
/// a corpus that dispatches nothing.
const MIN_DISPATCHING_FIXTURES: usize = 200;
const MIN_DISTINCT_SYMBOLS: usize = 100;

fn run_dir() -> PathBuf {
    common::repo_root().join("tests").join("run")
}

fn census_path() -> PathBuf {
    common::repo_root()
        .join("tests")
        .join("recognizer-census.txt")
}

/// The slice bounds-failure intrinsic. A safety edge, not a dispatched kernel: how many survive
/// `-O2` is a property of bounds-check elimination, so the census does not record it.
const BOUNDS_FAIL: &str = "wukong_rt_bounds_fail";
/// The failure edge of every other run-time check (a range index, `str_from_utf8`): a safety edge
/// like [`BOUNDS_FAIL`], which range indexes called before it existed, so not recorded either.
const CHECK_FAIL: &str = "wukong_rt_check_fail";

/// Runtime symbols plus multiplicity witnesses and the number of inline vector-recipe calls.
/// Keep bare symbols so the original symbol/fixture floors retain their meaning.
fn kernel_calls(mir: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut counts = BTreeMap::<String, usize>::new();
    for (idx, _) in mir.match_indices("call wukong_") {
        let rest = &mir[idx + "call ".len()..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        if &rest[..end] == BOUNDS_FAIL || &rest[..end] == CHECK_FAIL {
            continue;
        }
        *counts.entry(rest[..end].to_string()).or_default() += 1;
    }
    for (symbol, count) in counts {
        out.insert(symbol.clone());
        if count > 1 {
            out.insert(format!("{symbol}*{count}"));
        }
    }
    let recipes = mir.matches("veckernel #").count();
    if recipes != 0 {
        out.insert(format!("veckernel*{recipes}"));
    }
    out
}

#[test]
fn census_detects_removed_duplicate_calls_and_vector_recipes() {
    let one = "call wukong_norm_f32(%1)\n";
    let two = format!("{one}{one}");
    assert_ne!(kernel_calls(one), kernel_calls(&two));
    let recipe = "veckernel #0(ptrs=%1, scalars=%2, n=%3)\n";
    assert_ne!(kernel_calls(""), kernel_calls(recipe));
    assert_ne!(
        kernel_calls(recipe),
        kernel_calls(&format!("{recipe}{recipe}"))
    );
    assert!(kernel_calls(&two).contains("wukong_norm_f32"));
    assert!(kernel_calls(&two).contains("wukong_norm_f32*2"));
}

/// Compile one fixture to MIR at -O2 and return its dispatched symbols.
fn census_one(path: &Path) -> BTreeSet<String> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    for k in COMPILER_ENV {
        cmd.env_remove(k);
    }
    let out = cmd
        .args(["--emit=mir", "-O2"])
        .arg(path)
        .output()
        .expect("spawn wukongc");
    if out.status.code() != Some(0) {
        return BTreeSet::from([COMPILE_FAILED.to_string()]);
    }
    kernel_calls(&String::from_utf8_lossy(&out.stdout))
}

/// Sweep the corpus with a small fixed worker pool. One `--emit=mir` per fixture is the whole cost;
/// four workers keeps it inside a couple of minutes on a shared laptop without monopolising it.
fn sweep() -> Vec<(String, BTreeSet<String>)> {
    let fixtures = common::collect_wk(&run_dir());
    let n = fixtures.len();
    let mut results: Vec<(String, BTreeSet<String>)> = Vec::with_capacity(n);
    const WORKERS: usize = 4;
    let chunks: Vec<Vec<PathBuf>> = (0..WORKERS)
        .map(|w| {
            fixtures
                .iter()
                .skip(w)
                .step_by(WORKERS)
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect();
    let mut parts: Vec<Vec<(String, BTreeSet<String>)>> = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|chunk| {
                s.spawn(move || {
                    chunk
                        .iter()
                        .map(|p| {
                            (
                                p.file_name().unwrap().to_string_lossy().into_owned(),
                                census_one(p),
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            parts.push(h.join().expect("census worker panicked"));
        }
    });
    for part in parts {
        results.extend(part);
    }
    results.sort_by(|a, b| a.0.cmp(&b.0));
    results
}

fn render(rows: &[(String, BTreeSet<String>)]) -> String {
    let mut s = String::new();
    s.push_str(
        "# Recognizer dispatch census: every `call wukong_*` symbol each tests/run fixture emits\n\
         # at `--emit=mir -O2`, one line per fixture, symbols sorted.\n\
         # `symbol*N` additionally pins repeated calls; `veckernel*N` pins inline AVX2 recipe calls.\n\
         #\n\
         # GENERATED. Regenerate with:\n\
         #   WUKONG_UPDATE_CENSUS=1 cargo test -p wukongc --test recognizer_census\n\
         # which rewrites this file and then fails on purpose, so the diff gets read.\n\
         #\n\
         # This file records what the compiler DOES, not what is correct. A line moving is a\n\
         # recognizer change and needs a justification in the commit body; a line NOT moving is\n\
         # no evidence at all that the kernel computes the right answer.\n",
    );
    for (name, syms) in rows {
        s.push_str(name);
        s.push(':');
        for k in syms {
            s.push(' ');
            s.push_str(k);
        }
        s.push('\n');
    }
    s
}

fn parse_census(text: &str) -> Vec<(String, BTreeSet<String>)> {
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let Some((name, rest)) = line.split_once(':') else {
            panic!("malformed census line: {line:?}");
        };
        rows.push((
            name.to_string(),
            rest.split_whitespace().map(|s| s.to_string()).collect(),
        ));
    }
    rows
}

#[test]
fn recognizer_dispatch_matches_the_census() {
    let rows = sweep();

    let dispatching = rows
        .iter()
        .filter(|(_, s)| s.iter().any(|name| name.starts_with("wukong_")))
        .count();
    let distinct: BTreeSet<&String> = rows
        .iter()
        .flat_map(|(_, s)| s.iter())
        .filter(|s| s.starts_with("wukong_") && !s.contains('*'))
        .collect();
    let failed: Vec<&str> = rows
        .iter()
        .filter(|(_, s)| s.contains(COMPILE_FAILED))
        .map(|(n, _)| n.as_str())
        .collect();
    assert!(
        failed.is_empty(),
        "{} fixture(s) failed to compile at --emit=mir -O2: {failed:?}",
        failed.len()
    );
    assert!(
        dispatching >= MIN_DISPATCHING_FIXTURES && distinct.len() >= MIN_DISTINCT_SYMBOLS,
        "the corpus now dispatches {} symbol(s) across {dispatching} fixture(s); floors are \
         {MIN_DISTINCT_SYMBOLS} and {MIN_DISPATCHING_FIXTURES}. Either recognition has collapsed \
         wholesale, or this census was regenerated from a broken tree.",
        distinct.len()
    );

    let rendered = render(&rows);
    let path = census_path();

    if std::env::var_os("WUKONG_UPDATE_CENSUS").is_some() {
        std::fs::write(&path, &rendered).expect("write census");
        panic!(
            "census rewritten to {} ({} fixtures, {} distinct symbols). This is a deliberate \
             failure: read `git diff` on that file, confirm every gained or lost symbol is a \
             change you meant to make, then re-run WITHOUT WUKONG_UPDATE_CENSUS=1.",
            path.display(),
            rows.len(),
            distinct.len()
        );
    }

    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}. Generate it with WUKONG_UPDATE_CENSUS=1.",
            path.display()
        )
    });
    if rendered == expected {
        eprintln!(
            "census: {} fixtures, {dispatching} dispatching, {} distinct kernel symbols",
            rows.len(),
            distinct.len()
        );
        return;
    }

    // Report per fixture, because "the file differs" is not actionable and a 388-line diff is not
    // either. Name the fixture and the symbols that appeared or vanished.
    let old = parse_census(&expected);
    let mut report = Vec::new();
    let old_names: BTreeSet<&String> = old.iter().map(|(n, _)| n).collect();
    let new_names: BTreeSet<&String> = rows.iter().map(|(n, _)| n).collect();
    for n in new_names.difference(&old_names) {
        report.push(format!("+ fixture {n} is new to the corpus"));
    }
    for n in old_names.difference(&new_names) {
        report.push(format!(
            "- fixture {n} has LEFT the corpus (deleted, renamed, or moved out of tests/run)"
        ));
    }
    for (name, now) in &rows {
        if let Some((_, before)) = old.iter().find(|(n, _)| n == name) {
            let gained: Vec<&String> = now.difference(before).collect();
            let lost: Vec<&String> = before.difference(now).collect();
            if !gained.is_empty() || !lost.is_empty() {
                report.push(format!("  {name}: gained {gained:?}, lost {lost:?}"));
            }
        }
    }
    panic!(
        "recognizer dispatch has changed in {} place(s):\n{}\n\nA kernel that stops being \
         dispatched is invisible to every other gate in this repo: the program keeps printing the \
         same bytes on both backends at every -O level. If these changes are intended, regenerate \
         with WUKONG_UPDATE_CENSUS=1 and justify them in the commit body.",
        report.len(),
        report.join("\n")
    );
}
