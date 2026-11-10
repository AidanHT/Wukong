//! Static integrity checks on the fixture corpus, and the harness's own mutation control.
//!
//! Nothing here spawns `wukongc`; every check reads files. That is what makes it affordable to run
//! in `cargo test --workspace` (CI part 3) alongside the four process-spawning gates.
//!
//! ## What this file is for
//!
//! `tests/run/README.md` states four authoring rules that are, per the root `CLAUDE.md`, the
//! *only* defence for a recognizer fixture — because a recognizer fires identically on the
//! interpreter and on native at every `-O` level, so the differential, invariance and determinism
//! gates are structurally incapable of seeing whether a kernel fired or whether it computed the
//! right answer. Until this file existed, nothing checked any of those rules: not CI, not
//! `tools/`, not any `#[test]`. An unenforced rule in the one place automation cannot reach is the
//! weakest control available, and the pasted-expectation rule was already being broken in the
//! shipped tree.
//!
//! Three of the four rules are judgement calls a lint cannot make ("print every lane", "derive the
//! values, never paste them", "verify dispatch with `--emit=mir`"). What *is* mechanical is
//! checked here:
//!
//! * every fixture's directives parse, so a typo can no longer mean zero assertions;
//! * every fixture pins an exit code and says something about stdout;
//! * every declared `EXPECT-OUT-APPROX` tolerance appears in `DECLARED_TOLERANCES` with a cited
//!   source, so a tolerance nobody can justify cannot enter the corpus;
//! * no `.wk`-adjacent stray file (`foo.wk.bak`) sits in a corpus directory, since a fixture
//!   leaves every gate by being renamed;
//! * every `.wk` name a sibling gate hardcodes still exists;
//! * `tests/fail` fixtures carry exactly one `EXPECT-CODE`;
//! * and the stdout comparator the gate runs actually rejects a wrong answer.

#[path = "common/mod.rs"]
mod common;

use common::{check_stdout, collect_wk, parse_directives, repo_root, Expectation};
use std::path::{Path, PathBuf};

/// Every relative tolerance an `EXPECT-OUT-APPROX` line in `tests/run` is allowed to declare,
/// paired with where the number comes from.
///
/// The campaign's hard rule is "do not invent a tolerance". A free-form numeric field in a
/// directive is an invitation to do exactly that, and a tolerance is only reviewable if the
/// reviewer can find out where it came from. So the field is closed: a fixture may spell any
/// tolerance it likes, and the corpus will refuse it until its provenance is written down here.
/// The token is matched **as text**, so `2e-7` and `0.0000002` are different entries and a silent
/// re-spelling cannot slip a different number past review.
const DECLARED_TOLERANCES: &[(&str, &str)] = &[
    (
        "1.625e-7",
        "wukong_runtime/src/vmath.rs:124 and :261 — the exp kernel's maximum relative error against \
         f64 `exp`, measured EXHAUSTIVELY over every f32 in [-87, 88.37626] by \
         `vmath_exp_log_exhaustive_ulp` (1.625e-7, about 1.4 ULP; max 2 ULP). It is the kernel's own \
         published contract, not a number chosen to make a fixture pass.",
    ),
    // The three row-kernel decades below share one derivation. Each transcendental row kernel's
    // `matches_f64_reference` test publishes `|got - ref| <= B * max(|ref|, 1)` against an f64
    // recompute (or an absolute `< B`, which implies it). A fixture prints raw lanes and compares
    // them RELATIVELY, so a lane's tolerance is that bound divided by `min(|ref|, 1)`, rounded up to
    // the next decade: `B` itself where |ref| >= 1, one decade looser where 0.1 <= |ref| < 1, two
    // where 0.01 <= |ref| < 0.1. The cov4 generators assign each APPROX line exactly that decade,
    // and keep every lane at |ref| >= 0.01 (no cancellation lanes), so no line is looser than its
    // kernel's own published contract implies at that magnitude.
    (
        "5e-6",
        "wukong_runtime/src/vspell.rs `vspell_relative_error_in_each_spellings_well_conditioned_range` \
         — the natural-activation spelling kernels' relative error against the exact function each \
         spells (f64, the spelling's own f32 constants), asserted over a 2^-6 grid of each \
         spelling's well-conditioned range (the tanh/erf GELUs from -1.5, every log(1 + exp(..)) \
         from -4, the rest from -40, all to 40). Measured worst: 3.8e-6 (QuickGELU near -39), \
         3.1e-6 (naive softplus near -3.9), 1.1e-6 (erf GELUs), at most 6e-7 elsewhere. Used by the \
         spelling_act_*.wk fixtures, on lanes inside those ranges whose bits the scalar semantics \
         do not pin.",
    ),
    (
        "1e-5",
        "B = 1e-5, at |ref| >= 1: the row-loss kernels' f64-reference bound — \
         wukong_runtime/src/entropy.rs:262, kldiv.rs:280, kd_loss.rs:390, xent.rs:342 and \
         xent_bwd.rs:380 (`|got - ref| / max(|ref|, 1) <= 1e-5`) and logsoftmax.rs:466 (log-sum-exp, \
         absolute `< 1e-5`). Used by cov4_par_loss.wk and cov4_rowkernels_wide.wk on lanes whose \
         independent f64 reference has |ref| >= 1.",
    ),
    (
        "1e-4",
        "B = 1e-4 at |ref| >= 1: softmax_bwd.rs:302 (`|got - ref| / max(|ref|, 1) <= 1e-4`), \
         rope.rs:354 and rope_bwd.rs:367 (absolute `max_err < 1e-4`); or one of the B = 1e-5 kernels \
         above (entropy.rs:262 ... xent_bwd.rs:380, logsoftmax.rs:466) at 0.1 <= |ref| < 1, where \
         1e-5 / 0.1 = 1e-4. Used by cov4_par_loss.wk, cov4_par_rope.wk and cov4_rowkernels_wide.wk.",
    ),
    (
        "1e-3",
        "A B = 1e-4 kernel (softmax_bwd.rs:302, rope.rs:354, rope_bwd.rs:367) at 0.1 <= |ref| < 1, \
         or a B = 1e-5 kernel (entropy.rs:262 ... xent_bwd.rs:380) at 0.01 <= |ref| < 0.1: in the \
         corpus, RoPE outputs between 0.38 and 1 and cross-entropy-backward probabilities down to \
         0.01 (cov4_par_loss.wk, cov4_par_rope.wk, cov4_rowkernels_wide.wk).",
    ),
    (
        "5e-4",
        "wukong_runtime/src/attention.rs `# Numeric contract` and every correctness test there — \
         the fused attention kernel is held to `1e-4 + 1e-4·|want|` against the materialized \
         reference. `sdpa_loss_train.wk` makes every output lane a convex combination of V rows in \
         [1, 1.75], so |o| >= 1 and that band is at most 2e-4 relative (plus ~2e-6 for the f32 \
         reference's own rounding against f64); its printed loss is a sum of 32 positive squares, \
         which at most doubles a relative error (4.1e-4 with the 32-term f32 sum). 5e-4 is that \
         bound rounded up, derived from the kernel's contract rather than its output.",
    ),
    // The three premise_* entries below (P12.24) share one derivation: c*sqrt(K)*eps with c = 8 and
    // eps = 2^-23, the in-tree bound for a reassociated f32 reduction
    // (wukong_codegen_cranelift/src/tests.rs `p4_reduction_f64_reference`), at the fixture's own
    // contraction length K, rounded UP to three significant figures. Each fixture's prose states
    // whether that is also a worst-case bound for its data and what the simulated orders reach.
    (
        "6.11e-5",
        "c*sqrt(K)*eps at K = 4096: 8 * 64 * 2^-23 = 2^-14 = 6.1035e-5, rounded up. \
         premise_reduce_4096.wk (sum / dot / sum-of-squares over 4096 positive terms; the \
         random-walk bound, with the sequential order and reduce.rs's 8x8x64 partition simulated \
         at <= 3.6e-7).",
    ),
    (
        "7.63e-6",
        "c*sqrt(K)*eps at K = 64: 8 * 8 * 2^-23 = 2^-17 = 7.629e-6, rounded up. \
         premise_gemm_64.wk (64x64x64 GEMM over positive inputs); at this K it also exceeds the \
         worst case gamma_64 = 3.81e-6 for a positive dot product in any order.",
    ),
    (
        "2.65e-5",
        "c*sqrt(K)*eps at K = 768: 8 * sqrt(768) * 2^-23 = 2.643e-5, rounded up. \
         premise_norm_wide.wk (one 768-wide RMSNorm row); it also exceeds that data's worst case, \
         gamma_768 / 2 + 5u = 2.32e-5 (the square root halves the sum-of-squares error).",
    ),
];

/// `.wk`-adjacent extensions that must never appear in a corpus directory. Each gate collects
/// fixtures with `extension == \"wk\"`, so `foo.wk.bak` is silently not a fixture: renaming is the
/// cheapest way in the repo to remove a program from `run_suite`, both invariance gates and the
/// native differential at once, with no failure anywhere.
const STRAY_SUFFIXES: &[&str] = &[
    ".wk.bak", ".wk.orig", ".wk.txt", ".wk.old", ".wk.rej", ".wk~",
];

fn run_dir() -> PathBuf {
    repo_root().join("tests").join("run")
}

fn fail_dir() -> PathBuf {
    repo_root().join("tests").join("fail")
}

fn name_of(p: &Path) -> String {
    p.file_name().unwrap().to_string_lossy().into_owned()
}

/// Every fixture's directives parse, pin an exit code, and say something about stdout.
///
/// All three were previously unenforced in different ways: an unknown directive fell off the end
/// of a `strip_prefix` chain with no diagnostic, and `run.rs`'s exit and stdout checks were each
/// guarded on their directive being present. This runs without spawning anything, so a corpus that
/// has gone vacuous fails in seconds rather than after a 388-program sweep.
#[test]
fn every_run_fixture_declares_what_it_asserts() {
    let mut problems = Vec::new();
    let fixtures = collect_wk(&run_dir());
    for p in &fixtures {
        let name = name_of(p);
        let src = std::fs::read_to_string(p).unwrap();
        let d = match parse_directives(&src, &name) {
            Ok(d) => d,
            Err(e) => {
                problems.push(e);
                continue;
            }
        };
        if d.exit.is_none() {
            problems.push(format!("{name}: no `// EXPECT-EXIT:`"));
        }
        if !d.constrains_stdout() {
            problems.push(format!(
                "{name}: no `// EXPECT-OUT:` / `// EXPECT-OUT-APPROX:` / `// EXPECT-OUT-NONE` — \
                 the fixture runs and asserts nothing about what it printed"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "{} corpus fixture problem(s):\n{}",
        problems.len(),
        problems.join("\n")
    );
    eprintln!("linted {} run fixture(s)", fixtures.len());
}

/// Every `EXPECT-OUT-APPROX` tolerance in the corpus is one of the tolerances whose provenance is
/// written down in `DECLARED_TOLERANCES`.
#[test]
fn every_declared_tolerance_has_a_cited_source() {
    let mut problems = Vec::new();
    let mut used = 0usize;
    for p in collect_wk(&run_dir()) {
        let name = name_of(&p);
        let src = std::fs::read_to_string(&p).unwrap();
        let Ok(d) = parse_directives(&src, &name) else {
            continue; // reported by the test above; do not double-report
        };
        for e in &d.out {
            if let Expectation::Approx { rel_text, .. } = e {
                used += 1;
                if !DECLARED_TOLERANCES.iter().any(|(t, _)| t == rel_text) {
                    problems.push(format!(
                        "{name}: EXPECT-OUT-APPROX tolerance `{rel_text}` has no entry in \
                         DECLARED_TOLERANCES. Add one naming the source the number comes from — a \
                         kernel's published error bound, an exhaustive measurement, a standard. \
                         If there is no such source, the tolerance was invented and the fixture \
                         needs a different assertion, not a looser one."
                    ));
                }
            }
        }
    }
    assert!(
        problems.is_empty(),
        "{} undeclared tolerance(s):\n{}",
        problems.len(),
        problems.join("\n")
    );
    eprintln!("{used} EXPECT-OUT-APPROX line(s) across the corpus");
}

/// Every `EXPECT-OUT-APPROX` line in the shipped corpus actually rejects a wrong answer.
///
/// This is the mutation control for the tolerance mechanism itself, run against the real
/// expectations rather than a synthetic one: for each APPROX line, the declared value must pass and
/// a value displaced by ten times the declared tolerance must fail. It catches a tolerance so wide
/// that the line asserts nothing, a value the comparator cannot parse, and a comparator regression
/// that made `approx_eq` always true — three ways an APPROX fixture reads as coverage and is not.
#[test]
fn no_approx_line_in_the_corpus_is_vacuous() {
    let mut checked = 0usize;
    let mut problems = Vec::new();
    for p in collect_wk(&run_dir()) {
        let name = name_of(&p);
        let src = std::fs::read_to_string(&p).unwrap();
        let Ok(d) = parse_directives(&src, &name) else {
            continue; // reported by every_run_fixture_declares_what_it_asserts
        };
        for (i, e) in d.out.iter().enumerate() {
            let Expectation::Approx {
                text, value, rel, ..
            } = e
            else {
                continue;
            };
            checked += 1;
            // A single-line Directives carrying just this expectation, so the check is per line.
            let one = common::Directives {
                run_args: Vec::new(),
                exit: None,
                out: vec![e.clone()],
                expect_no_stdout: false,
                stderr_contains: Vec::new(),
            };
            let truth = format!("{value}\n");
            if check_stdout(&truth, &one).is_err() {
                problems.push(format!(
                    "{name} line {}: the declared value {text} does not satisfy its own directive",
                    i + 1
                ));
            }
            let scale = if *value == 0.0 { 1.0 } else { value.abs() };
            let far = format!("{}\n", value + 10.0 * rel * scale);
            if check_stdout(&far, &one).is_ok() {
                problems.push(format!(
                    "{name} line {}: a value 10x the declared tolerance away still passes — that \
                     line asserts nothing",
                    i + 1
                ));
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    assert!(
        checked > 0,
        "no EXPECT-OUT-APPROX lines found in the corpus, so this mutation control is vacuous; \
         either the directive is unused (delete it from the grammar) or the scanner broke"
    );
    eprintln!("mutation-checked {checked} EXPECT-OUT-APPROX line(s)");
}

/// No `.wk`-adjacent stray file in either corpus directory.
#[test]
fn no_fixture_has_been_renamed_out_of_the_gates() {
    let mut problems = Vec::new();
    for dir in [run_dir(), fail_dir()] {
        for e in std::fs::read_dir(&dir).unwrap().flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if STRAY_SUFFIXES.iter().any(|s| n.ends_with(s)) {
                problems.push(format!("{}/{n}", dir.display()));
            }
        }
    }
    assert!(
        problems.is_empty(),
        "file(s) that look like a fixture renamed out of every gate:\n{}",
        problems.join("\n")
    );
}

/// Every `*.wk` name hardcoded by a sibling gate still exists in `tests/run`.
///
/// `recognize.rs::KERNEL_WITNESSES` and `exe.rs`'s curated AOT list name fixtures by string. A
/// rename would turn those into a spawn of a missing file, which surfaces as a confusing compile
/// error rather than "the witness is gone".
#[test]
fn gates_that_hardcode_a_fixture_name_still_find_it() {
    let tests_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut problems = Vec::new();
    let mut checked = 0usize;
    for gate in ["recognize.rs", "exe.rs"] {
        let src = std::fs::read_to_string(tests_dir.join(gate)).unwrap();
        for lit in string_literals_ending_in_wk(&src) {
            checked += 1;
            if !run_dir().join(&lit).exists() {
                problems.push(format!(
                    "{gate} names tests/run/{lit}, which does not exist"
                ));
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    assert!(
        checked >= 10,
        "found only {checked} hardcoded fixture name(s) across the sibling gates; the scanner has \
         probably stopped matching and this check has gone vacuous"
    );
}

/// Pull `"something.wk"` string literals out of Rust source, textually.
fn string_literals_ending_in_wk(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in src.split('"').skip(1).step_by(2) {
        if chunk.ends_with(".wk") && !chunk.contains('/') && !chunk.contains(' ') {
            out.push(chunk.to_string());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Every `tests/fail` fixture carries exactly one `// EXPECT-CODE:`.
///
/// `fail.rs::expected_code` takes the *first* match and panics only when there is none, so a
/// second one is silently ignored — a fixture edited to pin a different code while the old
/// directive stayed above it would keep asserting the old code.
#[test]
fn every_fail_fixture_pins_exactly_one_code() {
    let mut problems = Vec::new();
    let fixtures = collect_wk(&fail_dir());
    for p in &fixtures {
        let name = name_of(p);
        let src = std::fs::read_to_string(p).unwrap();
        let n = src
            .lines()
            .filter(|l| l.trim_start().starts_with("// EXPECT-CODE:"))
            .count();
        if n != 1 {
            problems.push(format!(
                "{name}: {n} `// EXPECT-CODE:` directive(s), expected 1"
            ));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    // Same floor logic as run.rs's MIN_FIXTURES: a compile-fail fixture also leaves every gate by
    // being renamed, and `fail.rs` asserts only that the directory is non-empty.
    assert!(
        fixtures.len() >= 140,
        "tests/fail holds {} fixture(s), floor is 140. Raise the floor in the same commit that \
         adds fixtures; if one was deliberately removed, lower it and say why.",
        fixtures.len()
    );
}

// ---------------------------------------------------------------------------------------------
// The harness's own tests. Everything above asserts things about fixtures; this asserts things
// about the code that reads them.
//
// The campaign's bar for a kernel test is "mutation-verified in both directions". The corpus
// harness had never been held to it: `parse_directives` had no self-test, the comparison path had
// no deliberately-wrong case, and every finding about this corpus is a way for an assertion to be
// silently vacuous. A parser that drops a directive and a comparator that accepts anything both
// look exactly like a green suite.
// ---------------------------------------------------------------------------------------------

fn parse(src: &str) -> common::Directives {
    parse_directives(src, "<inline>").expect("should parse")
}

#[test]
fn multi_value_expands_exact_tokens_in_mixed_directive_order() {
    let d = parse(
        "// EXPECT-EXIT: 0\n\
         // EXPECT-OUT: leading words\n\
         //EXPECT-OUT-VALUES: 1\t-2  0.25\n\
         // EXPECT-OUT-APPROX: 1.5 1e-7\n\
         // EXPECT-OUT-VALUES: true tail\n\
         // EXPECT-OUT:\n",
    );
    let expanded = parse(
        "// EXPECT-EXIT: 0\n\
         // EXPECT-OUT: leading words\n\
         // EXPECT-OUT: 1\n\
         // EXPECT-OUT: -2\n\
         // EXPECT-OUT: 0.25\n\
         // EXPECT-OUT-APPROX: 1.5 1e-7\n\
         // EXPECT-OUT: true\n\
         // EXPECT-OUT: tail\n\
         // EXPECT-OUT:\n",
    );
    assert_eq!(d.out, expanded.out);
    assert_eq!(d.exit, Some(0));
    assert!(d.constrains_stdout());
    check_stdout("leading words\n1\n-2\n0.25\n1.5\ntrue\ntail\n\n", &d).unwrap();
}

#[test]
fn multi_value_rejects_every_mutated_missing_extra_or_reordered_lane() {
    let values = ["1", "-2", "0.25", "false", "last"];
    let d = parse("// EXPECT-OUT-VALUES: 1 -2 0.25 false last\n");
    let stdout = |lines: &[&str]| format!("{}\n", lines.join("\n"));
    check_stdout(&stdout(&values), &d).unwrap();
    for lane in 0..values.len() {
        let mut mutated = values.to_vec();
        mutated[lane] = "WRONG";
        assert!(check_stdout(&stdout(&mutated), &d).is_err(), "lane {lane}");
        let mut missing = values.to_vec();
        missing.remove(lane);
        assert!(check_stdout(&stdout(&missing), &d).is_err(), "lane {lane}");
    }
    for at in 0..=values.len() {
        let mut extra = values.to_vec();
        extra.insert(at, "EXTRA");
        assert!(check_stdout(&stdout(&extra), &d).is_err(), "position {at}");
    }
    for lane in 1..values.len() {
        let mut reordered = values.to_vec();
        reordered.swap(0, lane);
        assert!(
            check_stdout(&stdout(&reordered), &d).is_err(),
            "lane {lane}"
        );
    }
}

#[test]
fn multi_value_keeps_exact_physical_line_semantics() {
    let d = parse("// EXPECT-OUT-VALUES: 1.0 -0 001\n");
    check_stdout("1.0\n-0\n001\n", &d).unwrap();
    for actual in [
        "1\n-0\n001\n",
        "1.0\n0\n001\n",
        "1.0\n-0\n1\n",
        "1.0 -0 001\n",
        " 1.0\n-0\n001\n",
        "1.0\n-0\n001 \n",
    ] {
        assert!(check_stdout(actual, &d).is_err(), "{actual:?}");
    }
}

#[test]
fn multi_value_rejects_empty_misspelled_and_conflicting_declarations() {
    for source in [
        "// EXPECT-OUT-VALUES:",
        "// EXPECT-OUT-VALUES: \t ",
        "// EXPECT-OUT-VALUE: 1 2",
        "// EXPECT-OUT-VALUES : 1 2",
        "// EXPECT-OUT-VALUES-APPROX: 1 2",
        "// EXPECT-OUT-NONE\n// EXPECT-OUT-VALUES: 1 2",
        "// EXPECT-OUT-VALUES: 1 2\n// EXPECT-OUT-NONE",
        "// EXPECT-EXIT: 0\n// EXPECT-OUT-VALUES: 1\n// EXPECT-EXIT: 0",
    ] {
        assert!(parse_directives(source, "<inline>").is_err(), "{source}");
    }
    let error = parse_directives("// prose\n// EXPECT-OUT-VALUES:", "values.wk").unwrap_err();
    assert!(error.contains("values.wk:2:"), "{error}");
    assert!(error.contains("at least one exact token"), "{error}");
}

#[test]
fn parse_directives_reads_each_form() {
    let d = parse(
        "// RUN: --run --backend=native\n\
         // EXPECT-EXIT: 7\n\
         // EXPECT-OUT: hello\n\
         // EXPECT-OUT-APPROX: 1.5 1e-7\n\
         // EXPECT-STDERR: assertion failed\n\
         fn main() -> i32 { return 7; }\n",
    );
    assert_eq!(d.run_args, vec!["--run", "--backend=native"]);
    assert_eq!(d.exit, Some(7));
    assert_eq!(d.out.len(), 2);
    assert_eq!(d.out[0], Expectation::Exact("hello".to_string()));
    match &d.out[1] {
        Expectation::Approx {
            value,
            rel,
            rel_text,
            ..
        } => {
            assert_eq!(*value, 1.5);
            assert_eq!(*rel, 1e-7);
            assert_eq!(rel_text, "1e-7");
        }
        other => panic!("expected an Approx, got {other:?}"),
    }
    assert_eq!(d.stderr_contains, vec!["assertion failed".to_string()]);
    assert!(!d.expect_no_stdout);
    assert!(d.constrains_stdout());
}

#[test]
fn parse_directives_defaults_to_run_and_no_expectations() {
    let d = parse("module x\nfn main() -> i32 { return 0; }\n");
    assert_eq!(d.run_args, vec!["--run"]);
    assert_eq!(d.exit, None);
    assert!(d.out.is_empty());
    assert!(
        !d.constrains_stdout(),
        "a fixture with no stdout directive must not claim to constrain stdout — that predicate \
         is what run.rs's floor is written against"
    );
}

#[test]
fn expect_out_none_is_a_positive_assertion_not_an_absence() {
    let d = parse("// EXPECT-EXIT: 1\n// EXPECT-OUT-NONE\n");
    assert!(d.expect_no_stdout);
    assert!(d.constrains_stdout());
    assert!(check_stdout("", &d).is_ok());
    assert!(
        check_stdout("surprise\n", &d).is_err(),
        "EXPECT-OUT-NONE must reject output, or it is indistinguishable from declaring nothing"
    );
}

#[test]
fn an_unrecognized_directive_is_a_hard_error() {
    for bad in [
        "// EXPECT-OUT-APROX: 1.0 1e-7\n",
        "//EXPECT-OUT : 5\n",
        "// EXPECTOUT: 5\n",
        "//   EXPECT-OUTT: 5\n",
    ] {
        let e = parse_directives(bad, "<inline>")
            .expect_err("a misspelled directive must not be silently dropped");
        assert!(e.contains("unrecognized directive"), "wrong error: {e}");
    }
}

#[test]
fn a_spacing_variant_of_a_real_directive_is_honoured_not_dropped() {
    // The old chain matched `"// EXPECT-OUT:"` byte-for-byte, so `//EXPECT-OUT:` asserted
    // nothing. Whitespace after `//` is now insignificant, which is why the only way to get a
    // directive wrong is to misspell its name — and that is now a hard error above.
    let d = parse("//EXPECT-EXIT: 3\n//   EXPECT-OUT:  4\n");
    assert_eq!(d.exit, Some(3));
    assert_eq!(d.out, vec![Expectation::Exact("4".to_string())]);
}

#[test]
fn prose_that_merely_starts_with_the_word_expect_is_not_a_directive() {
    // Two shipped fixtures open a sentence with `EXPECT-OUT lines below are ...`. Treating that
    // as a typo would make the hard error above unusable, and an unusable check gets deleted.
    let d = parse("// EXPECT-OUT lines below are derived by hand.\n// EXPECT-EXIT: 0\n");
    assert_eq!(d.exit, Some(0));
    assert!(d.out.is_empty());
}

#[test]
fn an_approx_tolerance_that_asserts_nothing_is_rejected() {
    for bad in [
        "// EXPECT-OUT-APPROX: 1.0 1.0\n", // rel >= 1 accepts any same-sign value
        "// EXPECT-OUT-APPROX: 1.0 0\n",   // not a tolerance
        "// EXPECT-OUT-APPROX: 1.0 -1e-7\n", // sign typo
        "// EXPECT-OUT-APPROX: 1.0\n",     // missing the tolerance entirely
    ] {
        assert!(
            parse_directives(bad, "<inline>").is_err(),
            "should have been rejected: {bad:?}"
        );
    }
}

/// The mutation control: the comparator the gate runs must actually reject a wrong answer.
///
/// A test that passes both with and without the defect is worse than no test, because it reads as
/// coverage. Each case below perturbs exactly one thing and asserts the comparison fails.
#[test]
fn a_mutated_expectation_is_rejected() {
    let d = parse("// EXPECT-EXIT: 0\n// EXPECT-OUT: 42\n// EXPECT-OUT: -7\n");
    assert!(check_stdout("42\n-7\n", &d).is_ok(), "the truth must pass");

    let must_fail: &[(&str, &str)] = &[
        ("43\n-7\n", "one digit wrong in the first line"),
        ("42\n-8\n", "one digit wrong in the last line"),
        ("-7\n42\n", "the two lines permuted"),
        ("42\n", "a line missing"),
        ("42\n-7\n0\n", "an extra line"),
        (
            "",
            "no output at all — the shape a crashed or non-compiling program has",
        ),
    ];
    for (actual, what) in must_fail {
        assert!(
            check_stdout(actual, &d).is_err(),
            "the comparator accepted {actual:?} ({what})"
        );
    }
    // Not a mutation: `str::lines()` does not require a trailing newline, and the shipped corpus
    // relies on that. Recorded here so a future tightening of the comparator cannot break 388
    // fixtures without first breaking this line.
    assert!(check_stdout("42\n-7", &d).is_ok());
}

#[test]
fn contradictory_and_overwritten_expectations_are_rejected() {
    for line in ["// EXPECT-OUT: 42", "// EXPECT-OUT-APPROX: 1.0 1.625e-7"] {
        for source in [
            format!("// EXPECT-OUT-NONE\n{line}\n"),
            format!("{line}\n// EXPECT-OUT-NONE\n"),
        ] {
            assert!(parse_directives(&source, "<inline>").is_err());
        }
    }
    for exit in [0, 1] {
        let source = format!("// EXPECT-EXIT: 0\n// EXPECT-EXIT: {exit}\n");
        assert!(parse_directives(&source, "<inline>").is_err());
    }
    let valid = parse("// EXPECT-EXIT: 0\n// EXPECT-OUT-NONE\n");
    assert!(check_stdout("", &valid).is_ok());
    assert!(check_stdout("42\n", &valid).is_err());
}

/// The APPROX comparator's three sharp edges, each asserted in both directions.
#[test]
fn approx_comparison_is_two_sided_and_rejects_nan() {
    let d = parse("// EXPECT-EXIT: 0\n// EXPECT-OUT-APPROX: 1.6487212707001282 1.625e-7\n");
    // Inside the band on both sides.
    assert!(check_stdout("1.6487213373184204\n", &d).is_ok()); // +0.56 ULP, the shipped kernel
    assert!(check_stdout("1.6487212181091309\n", &d).is_ok()); // -0.44 ULP, correctly rounded
                                                               // Outside it on both sides: 1e-6 relative is ~6x the declared band.
    assert!(check_stdout("1.6487229194214800\n", &d).is_err());
    assert!(check_stdout("1.6487196219788000\n", &d).is_err());
    // NaN is not "close to" anything, and must never pass.
    assert!(check_stdout("NaN\n", &d).is_err());
    assert!(check_stdout("inf\n", &d).is_err());
    assert!(check_stdout("banana\n", &d).is_err());

    // A zero expectation admits only zero: a relative band around 0 has width 0, and widening it
    // with an invented absolute floor is exactly what the campaign forbids.
    let z = parse("// EXPECT-EXIT: 0\n// EXPECT-OUT-APPROX: 0.0 1.625e-7\n");
    assert!(check_stdout("0\n", &z).is_ok());
    assert!(check_stdout("-0\n", &z).is_ok()); // -0.0 == 0.0 in IEEE comparison
    assert!(check_stdout("0.0000001\n", &z).is_err());
}
