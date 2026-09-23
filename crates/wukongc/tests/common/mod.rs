//! The `tests/run` directive grammar and the stdout comparator, in **one** place.
//!
//! Every consumer of a `tests/run/*.wk` fixture's directives routes through this module: the
//! executing gate (`run.rs`) and the static lint (`corpus_lint.rs`). That is deliberate. The
//! grammar used to live inline in `run.rs` as a `strip_prefix` else-if chain with no terminal
//! `else`, so a directive it did not understand — `//EXPECT-OUT:` with no space,
//! `// EXPECT-OUT-APROX:`, any future name — was discarded silently and the fixture asserted
//! nothing while still reading as rigorous. Two readers of one grammar cannot drift; two
//! independent copies can, and a second copy is how that hole comes back.
//!
//! ## Grammar
//!
//! A directive is a leading `//` comment, anywhere in the file. Leading whitespace and the
//! spacing after `//` are both insignificant.
//!
//! | directive | meaning |
//! |---|---|
//! | `// RUN: <args>` | **replaces** the whole argv (default `--run`). Honoured by `run_suite` only |
//! | `// EXPECT-EXIT: <n>` | required process exit code. Mandatory for every fixture |
//! | `// EXPECT-OUT: <line>` | one expected stdout line, compared as an exact string |
//! | `// EXPECT-OUT-APPROX: <value> <rel>` | one expected stdout line, compared as a number within a **declared** relative tolerance |
//! | `// EXPECT-OUT-NONE` | stdout must be empty. Mandatory for a fixture that prints nothing |
//! | `// EXPECT-STDERR: <substring>` | stderr must contain this substring (`run_suite` only) |
//!
//! Every fixture must carry `EXPECT-EXIT` and at least one of `EXPECT-OUT` /
//! `EXPECT-OUT-APPROX` / `EXPECT-OUT-NONE`; `run.rs::check_program` panics otherwise. Before that
//! floor existed, `run.rs`'s two checks were each guarded on the directive being present, so
//! `gemm_fused_bias_slice.wk` printed 36 unchecked numbers for its whole life.
//!
//! ## Why `EXPECT-OUT-APPROX` exists, and what it is not
//!
//! It is not a licence to loosen an assertion. It exists because without it a *derived* float
//! expectation is unwritable: `exp` is a 2-ULP kernel, so the scalar semantics of `y[i] =
//! exp(x[i])` do not name an f32 bit pattern, and the only spellings available were (a) paste the
//! kernel's own output as an exact 17-digit string — which pins the kernel's error as the spec,
//! and is how `transcendental_kernel.wk`'s expectation came to sit one ULP above the true value —
//! or (b) `print((y[i] * 1000.0) as i32)`, whose truncating cast grants an **undeclared,
//! one-sided** tolerance of up to one whole integer step. `EXPECT-OUT-APPROX` replaces a hidden
//! tolerance with a written, two-sided, reviewable one.
//!
//! The tolerance is not free-form: `corpus_lint.rs` refuses any value that is not in its
//! `DECLARED_TOLERANCES` table with a cited source, so a number nobody can justify cannot reach
//! the corpus.

#![allow(dead_code)] // each consumer uses a subset; the grammar is shared, the callers are not.

use std::path::Path;

/// One expected stdout line.
#[derive(Debug, Clone, PartialEq)]
pub enum Expectation {
    /// Compared as an exact string. The default, and what integer and bit-exactness fixtures use.
    Exact(String),
    /// Compared as a number: `|actual - value| <= rel * |value|`, two-sided.
    Approx {
        /// The directive's value field, verbatim, for error messages.
        text: String,
        value: f64,
        /// Relative tolerance as spelled in the fixture, kept as text so the lint can match the
        /// literal token against its provenance table rather than a re-formatted float.
        rel_text: String,
        rel: f64,
    },
}

/// Everything a `tests/run/<fixture>.wk` declares about its own execution.
#[derive(Debug, Clone)]
pub struct Directives {
    pub run_args: Vec<String>,
    pub exit: Option<i32>,
    pub out: Vec<Expectation>,
    /// `EXPECT-OUT-NONE` was declared: stdout must be empty.
    pub expect_no_stdout: bool,
    pub stderr_contains: Vec<String>,
}

impl Directives {
    /// True when the fixture says *something* about stdout. A fixture that says nothing is
    /// rejected by `run.rs`; this is the predicate that rejection is written against.
    pub fn constrains_stdout(&self) -> bool {
        self.expect_no_stdout || !self.out.is_empty()
    }
}

/// Parse a fixture's directives, or return a human-readable error naming the offending line.
///
/// `name` is used only in error messages.
pub fn parse_directives(src: &str, name: &str) -> Result<Directives, String> {
    let mut run_args = vec!["--run".to_string()];
    let mut exit = None;
    let mut out = Vec::new();
    let mut expect_no_stdout = false;
    let mut stderr_contains = Vec::new();

    for (idx, raw) in src.lines().enumerate() {
        let lineno = idx + 1;
        let Some(body) = raw.trim_start().strip_prefix("//") else {
            continue;
        };
        let body = body.trim_start();

        if let Some(rest) = body.strip_prefix("RUN:") {
            run_args = rest.split_whitespace().map(|s| s.to_string()).collect();
        } else if let Some(rest) = body.strip_prefix("EXPECT-EXIT:") {
            let t = rest.trim();
            exit = Some(t.parse::<i32>().map_err(|e| {
                format!("{name}:{lineno}: EXPECT-EXIT must be an integer, got {t:?} ({e})")
            })?);
        } else if let Some(rest) = body.strip_prefix("EXPECT-OUT-APPROX:") {
            out.push(parse_approx(rest, name, lineno)?);
        } else if let Some(rest) = body.strip_prefix("EXPECT-OUT-NONE") {
            if !rest.trim().is_empty() {
                return Err(format!(
                    "{name}:{lineno}: EXPECT-OUT-NONE takes no argument, got {:?}",
                    rest.trim()
                ));
            }
            expect_no_stdout = true;
        } else if let Some(rest) = body.strip_prefix("EXPECT-STDERR:") {
            stderr_contains.push(rest.trim().to_string());
        } else if let Some(rest) = body.strip_prefix("EXPECT-OUT:") {
            out.push(Expectation::Exact(rest.trim().to_string()));
        } else if looks_like_a_directive(body) {
            return Err(format!(
                "{name}:{lineno}: unrecognized directive {:?}. A directive this harness does not \
                 understand used to be discarded silently, which is zero assertions wearing the \
                 costume of a check. Known: RUN:, EXPECT-EXIT:, EXPECT-OUT:, EXPECT-OUT-APPROX:, \
                 EXPECT-OUT-NONE, EXPECT-STDERR:",
                body.trim_end()
            ));
        }
    }

    Ok(Directives {
        run_args,
        exit,
        out,
        expect_no_stdout,
        stderr_contains,
    })
}

/// A comment body that *meant* to be a directive: it starts with `EXPECT` and the first token is,
/// or is immediately followed by, a colon. Prose that merely opens with the word — two fixtures
/// begin a sentence `EXPECT-OUT lines below are ...` — has no colon there and is left alone.
fn looks_like_a_directive(body: &str) -> bool {
    if !body.starts_with("EXPECT") {
        return false;
    }
    let first = body.split_whitespace().next().unwrap_or("");
    first.ends_with(':') || body[first.len()..].trim_start().starts_with(':')
}

fn parse_approx(rest: &str, name: &str, lineno: usize) -> Result<Expectation, String> {
    let mut it = rest.split_whitespace();
    let (Some(v), Some(t)) = (it.next(), it.next()) else {
        return Err(format!(
            "{name}:{lineno}: EXPECT-OUT-APPROX needs two fields, `<value> <rel-tol>`, got {:?}",
            rest.trim()
        ));
    };
    if let Some(extra) = it.next() {
        return Err(format!(
            "{name}:{lineno}: EXPECT-OUT-APPROX takes exactly two fields; trailing {extra:?}"
        ));
    }
    let value: f64 = v
        .parse()
        .map_err(|e| format!("{name}:{lineno}: EXPECT-OUT-APPROX value {v:?}: {e}"))?;
    let rel: f64 = t
        .parse()
        .map_err(|e| format!("{name}:{lineno}: EXPECT-OUT-APPROX tolerance {t:?}: {e}"))?;
    if !(rel > 0.0 && rel < 1.0) {
        // rel >= 1 accepts any value of the same sign and so is not an assertion at all;
        // rel <= 0 (or NaN) is a typo. Neither is a tolerance anyone meant to write.
        return Err(format!(
            "{name}:{lineno}: EXPECT-OUT-APPROX tolerance must satisfy 0 < rel < 1, got {t:?}"
        ));
    }
    Ok(Expectation::Approx {
        text: v.to_string(),
        value,
        rel_text: t.to_string(),
        rel,
    })
}

/// Compare a program's stdout against its declared expectations.
///
/// Returns `Err(report)` on any mismatch. This is the single comparison path the gate uses, so
/// the harness's own mutation control (`corpus_lint.rs::a_mutated_expectation_is_rejected`)
/// exercises exactly the code the gate runs rather than a re-implementation of it.
pub fn check_stdout(actual: &str, d: &Directives) -> Result<(), String> {
    if d.expect_no_stdout {
        if !actual.is_empty() {
            return Err(format!(
                "EXPECT-OUT-NONE, but the program wrote {} byte(s) to stdout: {actual:?}",
                actual.len()
            ));
        }
        return Ok(());
    }
    let lines: Vec<&str> = actual.lines().collect();
    if lines.len() != d.out.len() {
        return Err(format!(
            "stdout has {} line(s), {} expected\n--- actual ---\n{actual}",
            lines.len(),
            d.out.len()
        ));
    }
    for (i, (got, want)) in lines.iter().zip(&d.out).enumerate() {
        match want {
            Expectation::Exact(s) => {
                if got != s {
                    return Err(format!("line {}: expected {s:?}, got {got:?}", i + 1));
                }
            }
            Expectation::Approx {
                text,
                value,
                rel_text,
                rel,
            } => {
                let Ok(a) = got.trim().parse::<f64>() else {
                    return Err(format!(
                        "line {}: EXPECT-OUT-APPROX {text} {rel_text}, but {got:?} is not a number",
                        i + 1
                    ));
                };
                if !approx_eq(a, *value, *rel) {
                    let err = a - *value;
                    let rel_err = if *value == 0.0 {
                        f64::INFINITY
                    } else {
                        (err / *value).abs()
                    };
                    return Err(format!(
                        "line {}: expected {text} +/- {rel_text} relative, got {got:?} \
                         (absolute error {err:e}, relative {rel_err:e})",
                        i + 1
                    ));
                }
            }
        }
    }
    Ok(())
}

/// `|a - e| <= rel * |e|`, with the three cases the plain formula gets wrong.
///
/// * **NaN never matches.** A kernel that returned NaN cannot pass an APPROX line. (Through the
///   corpus's dominant `as i32` print a NaN renders as `0`, and `0` is the most common expected
///   line in the corpus, so on those lanes a NaN is textually indistinguishable from the right
///   answer. On a raw print it is not, and this is the half of that hole a comparator can close.)
/// * **`e == 0` demands exact equality.** A relative tolerance around zero admits nothing, and
///   inventing an absolute floor to paper over that would be inventing a tolerance.
/// * **An infinite expectation demands the same infinity**, since `rel * inf` is meaningless.
fn approx_eq(a: f64, e: f64, rel: f64) -> bool {
    if a.is_nan() || e.is_nan() {
        return false;
    }
    if e == 0.0 || e.is_infinite() {
        return a == e;
    }
    (a - e).abs() <= rel * e.abs()
}

/// Collect the fixtures of a corpus directory the way every gate does: a **non-recursive**
/// `read_dir` filtered on `extension == "wk"`, sorted. Support modules live in `lib/`, which is
/// never scanned, and that is load-bearing.
pub fn collect_wk(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut programs: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "wk").unwrap_or(false))
        .collect();
    programs.sort();
    programs
}

/// The repo root, reached from `crates/wukongc`.
pub fn repo_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}
