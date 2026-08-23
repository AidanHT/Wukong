//! `sota-check` — part 11 of the language close-out campaign's verification gate (§3B).
//!
//! Re-runs the in-scope cross-language sweep (the general suite plus the timed `tests/compare/`
//! corpus) and evaluates it against `bench/sota/FLOOR.md`, the repository's recorded floor per
//! in-scope program. The floor is a **ratchet in both directions**: a ratio may not fall below its
//! recorded value, and a program may not silently leave the sweep — the second is the failure the
//! coverage ratchets exist for, and a cross-language sweep is strictly more vulnerable to it,
//! because a peer that fails to compile drops its column silently rather than failing the run.
//!
//! Exit codes: `0` every floor row evaluated and held; `1` a floor was violated, a program left
//! the sweep, a bar peer failed to build, an oracle failed, or a row was not evaluable (wrong or
//! unstable power state, no control); `2` the floor file itself is missing or malformed. A pass
//! REQUIRES an evaluable run: this gate refuses to go green on battery or AC+charging, because a
//! green that means "could not check" is how a sweep rots.
//!
//! Tier shortfalls (a row below its PARITY/TARGET/STRETCH bar) are REPORTED with the cause/owner/
//! wave recorded in the floor file, but do not fail this gate — §3B5's exit conditions are the
//! ratchet, presence, and peers. The tier gate is §3B7's merge rule, which is an owner decision,
//! not an exit code.

use std::path::{Path, PathBuf};

use crate::SotaRow;

/// One parsed row of `bench/sota/FLOOR.md`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FloorRow {
    pub(crate) program: String,
    pub(crate) peer: String,
    pub(crate) tier: String,
    /// The recorded floor: `peer_time / wukong_time` median, `> 1` = Wukong faster.
    pub(crate) floor: f64,
    pub(crate) power: String,
    pub(crate) cause: String,
    pub(crate) owner: String,
    pub(crate) wave: String,
}

/// The bar a tier names. `waived` rows carry an owner line naming the master plan §3.4 non-goal
/// that covers them and have no numeric bar.
fn tier_bar(tier: &str) -> Result<Option<f64>, String> {
    match tier {
        "PARITY" => Ok(Some(1.00)),
        "TARGET" => Ok(Some(1.25)),
        "STRETCH" => Ok(Some(2.00)),
        "waived" => Ok(None),
        other => Err(format!(
            "unknown tier {other:?} (PARITY | TARGET | STRETCH | waived)"
        )),
    }
}

/// Parse the floor file's table. Any line whose first non-space byte is `|` is a candidate row;
/// the header row (first cell `program`) and separator rows (cells of `-`/`:`) are skipped. A
/// malformed candidate is an error, not a skip — a floor file that silently loses rows IS the
/// failure this gate exists to catch.
pub(crate) fn parse_floor(text: &str) -> Result<Vec<FloorRow>, String> {
    let mut rows = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let t = line.trim();
        if !t.starts_with('|') {
            continue;
        }
        let cells: Vec<&str> = t.trim_matches('|').split('|').map(str::trim).collect();
        if cells.first().copied() == Some("program") {
            continue; // header
        }
        if cells
            .iter()
            .all(|c| c.chars().all(|ch| ch == '-' || ch == ':'))
        {
            continue; // separator
        }
        if cells.len() != 10 {
            return Err(format!(
                "FLOOR.md line {}: expected 10 cells (program | peer | tier | floor | date | power \
                 | twin | cause | owner | wave), got {}",
                lineno + 1,
                cells.len()
            ));
        }
        let floor: f64 = cells[3]
            .strip_suffix('x')
            .unwrap_or(cells[3])
            .parse()
            .map_err(|_| {
                format!(
                    "FLOOR.md line {}: floor {:?} is not a ratio like `0.98x`",
                    lineno + 1,
                    cells[3]
                )
            })?;
        if !(floor.is_finite() && floor > 0.0) {
            return Err(format!(
                "FLOOR.md line {}: floor must be finite and positive",
                lineno + 1
            ));
        }
        tier_bar(cells[2]).map_err(|e| format!("FLOOR.md line {}: {e}", lineno + 1))?;
        rows.push(FloorRow {
            program: cells[0].to_string(),
            peer: cells[1].to_string(),
            tier: cells[2].to_string(),
            floor,
            power: cells[5].to_string(),
            cause: cells[7].to_string(),
            owner: cells[8].to_string(),
            wave: cells[9].to_string(),
        });
    }
    Ok(rows)
}

/// Is this power-state key one the gate accepts? Part 11 runs on AC+full and idle; `AC` covers a
/// machine with no battery to report. Battery and AC+charging are different machines (the latter
/// caps all-core at ~25% on this laptop), and a gate that went green there would certify nothing.
fn power_ok(key: &str) -> bool {
    key == "AC+full" || key == "AC"
}

/// Evaluate one sweep against the floors. Returns the printable verdict lines and whether the gate
/// FAILED. Pure, so the rules are unit-testable without a machine that can be unplugged mid-test.
pub(crate) fn evaluate(floors: &[FloorRow], sweep: &[SotaRow]) -> (Vec<String>, bool) {
    let mut lines = Vec::new();
    let mut failed = false;
    let mut fail = |lines: &mut Vec<String>, msg: String| {
        lines.push(format!("  FAIL {msg}"));
        failed = true;
    };

    if floors.is_empty() {
        return (
            vec!["  FAIL FLOOR.md contains no rows — nothing is being checked".into()],
            true,
        );
    }

    for f in floors {
        let Some(s) = sweep.iter().find(|s| s.program == f.program) else {
            fail(
                &mut lines,
                format!(
                    "{}: LEFT THE SWEEP — the program produced no row at all. A change that stops \
                     a program from being measured turns nothing red on its own; this is the red.",
                    f.program
                ),
            );
            continue;
        };
        if s.peer_missing {
            fail(
                &mut lines,
                format!(
                    "{}: a bar peer failed to compile or load (peer column dropped). A missing \
                     peer is a failure, never a skip.",
                    f.program
                ),
            );
            continue;
        }
        if !s.oracle_ok {
            fail(
                &mut lines,
                format!(
                    "{}: a column failed the f64 oracle — ratios are void",
                    f.program
                ),
            );
            continue;
        }
        if !s.reportable {
            fail(
                &mut lines,
                format!(
                    "{}: the power state CHANGED during the section — not evaluable",
                    f.program
                ),
            );
            continue;
        }
        if !power_ok(&s.power) {
            fail(
                &mut lines,
                format!(
                    "{}: ran on {} — part 11 requires AC+full and an idle box; a green that means \
                     \"could not check\" is not a green",
                    f.program, s.power
                ),
            );
            continue;
        }
        let Some(r) = s.ratio else {
            fail(
                &mut lines,
                format!(
                    "{}: no admissible ratio (census: {}) — the recognition-off witness or a \
                     column is missing",
                    f.program, s.census
                ),
            );
            continue;
        };
        let Some(t) = s.twin else {
            fail(
                &mut lines,
                format!(
                    "{}: the C(twin) control did not build — the run has no floor",
                    f.program
                ),
            );
            continue;
        };
        // The ratchet, with the run's own measured resolution as the tolerance: the recorded floor
        // is violated only when today's median is below it by MORE than what a zero-difference
        // control moved this run. Anything inside that band is the instrument, not a regression.
        let tol = crate::floor_factor(t);
        if r.med * tol < f.floor {
            fail(
                &mut lines,
                format!(
                    "{}: ratio {:.2}x fell below its recorded floor {:.2}x (beyond this run's own \
                     {:.2}x control floor). Raising a floor is routine; falling below one is a \
                     regression in the language, the lowering, or a peer upgrade — name which \
                     before touching the floor file.",
                    f.program, r.med, f.floor, tol
                ),
            );
            continue;
        }
        // Held. Report the tier standing beside it (informational — §3B7's merge gate, owner-run).
        let bar = tier_bar(&f.tier).expect("validated at parse");
        let tier_note = match bar {
            None => format!("tier waived ({})", f.cause),
            Some(b) if r.med >= b => format!("at or above {} ({:.2}x >= {:.2}x)", f.tier, r.med, b),
            Some(b) => format!(
                "BELOW {} ({:.2}x < {:.2}x) — cause {}, owner {}, wave {}",
                f.tier, r.med, b, f.cause, f.owner, f.wave
            ),
        };
        lines.push(format!(
            "  ok   {}: {:.2}x [{:.2}-{:.2}] vs {} (floor {:.2}x, twin {:.2}x) — {}",
            f.program, r.med, r.lo, r.hi, s.peer, f.floor, tol, tier_note
        ));
    }

    // The other direction of the ratchet's presence rule: a sweep row with no floor is not a
    // failure (it is how a new program enters the file), but it must be impossible to miss.
    for s in sweep {
        if !floors.iter().any(|f| f.program == s.program) {
            if s.census.starts_with("NOTHING") {
                lines.push(format!(
                    "  NEW  {}: no floor recorded — measure under §4 and add its row to \
                     bench/sota/FLOOR.md",
                    s.program
                ));
            } else {
                // The bar is recognition-off by definition (§3B1); a row whose census dispatches
                // kernels measures the dispatcher and is context, not a floorable language row.
                lines.push(format!(
                    "  off-bar {}: recognizer-dialect census — context, not floorable under §3B1 \
                     ({})",
                    s.program, s.census
                ));
            }
        }
    }

    (lines, failed)
}

/// Where the floor file lives. `WUKONG_SOTA_FLOOR` overrides (tests, worktrees); the default is
/// resolved from this crate's manifest so the gate works from any working directory.
fn floor_path() -> PathBuf {
    match std::env::var_os("WUKONG_SOTA_FLOOR") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/sota/FLOOR.md"),
    }
}

/// The gate. Runs the full in-scope sweep (general + compare), prints everything the sections
/// normally print plus the scorecard, and returns the process exit code.
pub(crate) fn sota_check(cc: &str, cxx: &str, dir: &Path) -> i32 {
    let path = floor_path();
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("sota-check: cannot read {}: {e}", path.display());
            eprintln!(
                "sota-check: the floor file is the gate's entire surface — without it nothing is \
                 being checked. Create it from a measured round (see bench/sota/FLOOR.md's header \
                 once it exists, or the campaign file §3B5)."
            );
            return 2;
        }
    };
    let floors = match parse_floor(&text) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("sota-check: {e}");
            return 2;
        }
    };

    println!("================================================================================");
    println!("sota-check — the §3B bar: recognition-off Wukong vs the strongest honest peer");
    println!("================================================================================");
    println!(
        "floor file: {} ({} row(s)). The floor is a ratchet in both directions: a ratio may\n\
         not fall below its recorded value, and a program may not leave the sweep.",
        path.display(),
        floors.len()
    );
    println!();

    let mut sweep = crate::general::bench_general(cc, cxx, dir);
    println!();
    sweep.extend(crate::compare::bench_compare(cc, dir));
    println!();
    crate::print_sota_scorecard(&sweep);

    println!();
    println!("sota-check verdicts:");
    let (lines, failed) = evaluate(&floors, &sweep);
    for l in &lines {
        println!("{l}");
    }
    println!();
    if failed {
        println!("sota-check: FAIL (see the FAIL lines above)");
        1
    } else {
        println!(
            "sota-check: PASS — {} floor row(s) evaluated and held",
            floors.len()
        );
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RatioStat;

    fn stat(med: f64, lo: f64, hi: f64) -> RatioStat {
        RatioStat {
            med,
            lo,
            hi,
            rounds: 4,
        }
    }

    fn row(program: &str) -> SotaRow {
        SotaRow {
            program: program.into(),
            peer: "C".into(),
            ratio: Some(stat(1.05, 1.00, 1.10)),
            twin: Some(stat(1.00, 0.98, 1.02)),
            reportable: true,
            power: "AC+full".into(),
            census: "NOTHING — every loop stays generic Cranelift code".into(),
            peer_missing: false,
            oracle_ok: true,
        }
    }

    fn floor(program: &str, floor: f64) -> FloorRow {
        FloorRow {
            program: program.into(),
            peer: "C".into(),
            tier: "PARITY".into(),
            floor,
            power: "AC+full".into(),
            cause: "—".into(),
            owner: "—".into(),
            wave: "—".into(),
        }
    }

    /// The happy path passes and reports the tier standing.
    #[test]
    fn a_held_floor_passes() {
        let (lines, failed) = evaluate(&[floor("p", 1.00)], &[row("p")]);
        assert!(!failed, "{lines:?}");
        assert!(lines[0].contains("ok"), "{lines:?}");
    }

    /// A program that vanished from the sweep is a FAILURE, not a skip — the coverage-ratchet law.
    #[test]
    fn a_program_leaving_the_sweep_fails() {
        let (lines, failed) = evaluate(&[floor("p", 1.00)], &[]);
        assert!(failed);
        assert!(lines[0].contains("LEFT THE SWEEP"), "{lines:?}");
    }

    /// A bar peer that failed to compile is a FAILURE — a cross-language sweep drops a column
    /// silently otherwise.
    #[test]
    fn a_missing_peer_fails() {
        let mut s = row("p");
        s.peer_missing = true;
        let (lines, failed) = evaluate(&[floor("p", 1.00)], &[s]);
        assert!(failed, "{lines:?}");
    }

    /// The ratchet uses the run's own control as tolerance: a median inside the twin band does not
    /// fail, one beyond it does.
    #[test]
    fn the_ratchet_tolerates_exactly_the_twin_floor() {
        let mut s = row("p");
        s.ratio = Some(stat(0.99, 0.97, 1.01)); // below the 1.00 floor...
        s.twin = Some(stat(1.00, 0.98, 1.02)); // ...but the control moved 1.02x itself
        let (lines, failed) = evaluate(&[floor("p", 1.00)], &[s]);
        assert!(!failed, "inside the twin band must not fail: {lines:?}");

        let mut s = row("p");
        s.ratio = Some(stat(0.80, 0.78, 0.82)); // far outside any control band
        let (lines, failed) = evaluate(&[floor("p", 1.00)], &[s]);
        assert!(failed, "{lines:?}");
        assert!(lines[0].contains("fell below"), "{lines:?}");
    }

    /// The gate refuses to certify a run on the wrong power state — a green that means "could not
    /// check" is how a sweep rots.
    #[test]
    fn battery_and_charging_runs_fail_rather_than_pass() {
        for state in ["BATTERY", "AC+CHARGING"] {
            let mut s = row("p");
            s.power = state.into();
            let (lines, failed) = evaluate(&[floor("p", 1.00)], &[s]);
            assert!(failed, "{state}: {lines:?}");
        }
    }

    /// A below-tier row REPORTS (with its recorded cause) but does not fail the exit code — the
    /// tier gate is §3B7's merge rule, an owner decision, not part 11's exit conditions.
    #[test]
    fn below_tier_reports_but_does_not_fail() {
        let mut f = floor("p", 0.30);
        f.cause = "L5".into();
        let mut s = row("p");
        s.ratio = Some(stat(0.32, 0.30, 0.34));
        let (lines, failed) = evaluate(&[f], &[s]);
        assert!(!failed, "{lines:?}");
        assert!(lines[0].contains("BELOW PARITY"), "{lines:?}");
        assert!(lines[0].contains("L5"), "{lines:?}");
    }

    /// A sweep row with no floor is loudly NEW, never silently absorbed — unless its census shows
    /// dispatched kernels, in which case it is off the §3B1 bar and says so instead of nagging
    /// forever for a floor it must never get.
    #[test]
    fn an_unfloored_program_is_flagged_new_or_off_bar() {
        let mut dispatched = row("r");
        dispatched.census = "2 call site(s): wukong_sgemm_nt, wukong_vmath_f32".into();
        let (lines, failed) = evaluate(&[floor("p", 1.00)], &[row("p"), row("q"), dispatched]);
        assert!(!failed, "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("NEW  q")), "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("off-bar r")), "{lines:?}");
        assert!(
            !lines.iter().any(|l| l.contains("NEW  r")),
            "a dispatching row must not be asked for a floor: {lines:?}"
        );
    }

    /// An empty floor file is a FAILURE: a gate with no rows checks nothing and must say so.
    #[test]
    fn an_empty_floor_file_fails() {
        let (_, failed) = evaluate(&[], &[row("p")]);
        assert!(failed);
    }

    /// The parser round-trips the documented format and rejects malformed rows instead of
    /// skipping them.
    #[test]
    fn parse_floor_reads_the_documented_format() {
        let text = "\
# header prose\n\
| program | peer | tier | floor | date | power | twin | cause | owner | wave |\n\
|---|---|---|---|---|---|---|---|---|---|\n\
| compare/saxpy | C | PARITY | 0.98x | 2026-08-23 | AC+full | 1.04x | — | — | — |\n\
| general/focal_loss | C | PARITY | 0.27x | 2026-08-23 | AC+full | 1.05x | L5 | crates/wukong_opt/src/vectorize.rs | W2 |\n";
        let rows = parse_floor(text).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].program, "compare/saxpy");
        assert!((rows[0].floor - 0.98).abs() < 1e-9);
        assert_eq!(rows[1].cause, "L5");

        assert!(parse_floor("| too | few | cells |\n").is_err());
        assert!(parse_floor("| p | C | GOLD | 1.00x | d | AC+full | t | — | — | — |\n").is_err());
        assert!(parse_floor("| p | C | PARITY | fast | d | AC+full | t | — | — | — |\n").is_err());
    }
}
