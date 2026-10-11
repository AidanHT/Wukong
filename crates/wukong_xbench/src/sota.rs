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
//! the sweep, a bar peer failed to build, an oracle failed, the recognition on/off premise check
//! (`compare::premise_check`, P12.24) failed, or a row was not evaluable (wrong or unstable power
//! state, no control); `2` the floor file itself is missing or malformed. A pass
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

/// The census a sweep reports when its Wukong column never built.
///
/// Shared rather than spelled twice, because the gate branches on it: a census is a description of a
/// lowering that happened, and this is the one string that says none did. A sweep that invents its
/// own wording here would land in the off-bar arm and pass.
pub(crate) const CENSUS_DID_NOT_COMPILE: &str = "did not compile";

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
    // A program name is the key `evaluate` looks a sweep row up by, so a repeated one is not a
    // duplicate record — it is a floor that will never be read. The lookup takes the first match and
    // every later row with that name is silently skipped, which is the one failure a ratchet file
    // must not have: the row is present, the reviewer sees it, and nothing grades it.
    for (i, r) in rows.iter().enumerate() {
        if let Some(j) = rows[..i].iter().position(|p| p.program == r.program) {
            return Err(format!(
                "FLOOR.md: program {:?} appears twice (rows {} and {}) — the second would never be \
                 graded, because the sweep is looked up by name",
                r.program,
                j + 1,
                i + 1
            ));
        }
    }
    Ok(rows)
}

/// The power-state key [`crate::model::power_status_line`] reports when the OS cannot be asked.
/// Only Windows has an implementation, so every row of an off-Windows sweep carries this.
pub(crate) const POWER_UNAVAILABLE: &str = "n/a";

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

    // Part 11 is Windows-only today, and it has to say that ONCE rather than once per floored row.
    // `power_status_line` has no non-Windows implementation, so off Windows every row arrives with
    // `power: n/a`, `power_ok` rejects it, and the gate used to emit N copies of a message phrased
    // as a power-STATE problem -- which reads like "unplug the laptop", not "this gate cannot run
    // here". The refusal is the right outcome either way: a gate that cannot tell AC+full from
    // battery certifies nothing. What was missing is that it never said why.
    if !sweep.is_empty() && sweep.iter().all(|s| s.power == POWER_UNAVAILABLE) {
        fail(
            &mut lines,
            "no power reading is available on this platform, so no row can be evaluated: \
             `power_status_line` is implemented for Windows only and reports `n/a` everywhere else. \
             Part 11 is a LOCAL, Windows-only gate today -- it is deliberately not in CI (the ubuntu \
             runner has gcc, g++ and rustc but no stable clock), and a green that means \
             \"could not check the machine\" is not a green."
                .to_string(),
        );
        return (lines, true);
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
        // The control is this run's own resolution, so it has to be sound before it can be spent as
        // one. Two ways it is not: rounds that disagree by more than the harness's own publishing
        // limit, and a departure from the control's true value of 1.00 beyond that same limit. Both
        // matter in the SAME direction, which is why neither can be left to the ratchet below:
        // `floor_factor` GROWS when the control degrades, and a wider tolerance makes the ratchet
        // more forgiving — so without these two gates a broken instrument buys silence rather than a
        // failure, and the worse the run, the less it can be caught saying.
        let limit = crate::spread_limit();
        if !t.conclusive() {
            fail(
                &mut lines,
                format!(
                    "{}: the C(twin) control did not settle ({:.2}x spread over {} round(s), limit \
                     {:.2}x) — a run with no resolution cannot certify a floor",
                    f.program,
                    t.spread(),
                    t.rounds,
                    limit
                ),
            );
            continue;
        }
        let tol = crate::floor_factor(t);
        if tol > limit {
            fail(
                &mut lines,
                format!(
                    "{}: the C(twin) control moved {:.2}x on a ratio whose true value is 1.00 \
                     (limit {:.2}x) — this run's noise floor is wider than the claims it would be \
                     certifying",
                    f.program, tol, limit
                ),
            );
            continue;
        }
        // A ratio the run could not pin is not a number to ratchet against. `evaluate` reads the
        // median, and a median is only a measurement when the rounds behind it agree.
        if !r.conclusive() {
            fail(
                &mut lines,
                format!(
                    "{}: rounds disagree {:.2}x over {} round(s) (limit {:.2}x) — INCONCLUSIVE, not \
                     a ratio the floor file can be graded against",
                    f.program,
                    r.spread(),
                    r.rounds,
                    limit
                ),
            );
            continue;
        }
        // The ratchet, with the run's own measured resolution as the tolerance: the recorded floor
        // is violated only when today's median is below it by MORE than what a zero-difference
        // control moved this run. Anything inside that band is the instrument, not a regression.
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
        //
        // A median that clears the bar is not the same as a run that RESOLVED it: when the round
        // range overlaps what a byte-identical control column did, the machine in this state cannot
        // tell the two columns apart at all, and announcing "at or above PARITY" from inside the
        // noise floor is the instrument reporting its own jitter as a result.
        let bar = tier_bar(&f.tier).expect("validated at parse");
        let tier_note = match bar {
            None => format!("tier waived ({})", f.cause),
            Some(_) if crate::verdict(r, Some(tol)) == crate::Verdict::BelowFloor => format!(
                "tier NOT RESOLVED: {:.2}x [{:.2}-{:.2}] overlaps this run's own {:.2}x control \
                 floor — no tier can be read off a difference this run could not measure",
                r.med, r.lo, r.hi, tol
            ),
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
            if s.census == CENSUS_DID_NOT_COMPILE {
                // Not "context". The census is only a *description* of a lowering that happened,
                // and this one did not: the Wukong column never built. Falling through to the
                // off-bar arm below would announce a broken benchmark as a recognizer-dialect row
                // and leave the gate green, which is the same shape of silence as a program leaving
                // the sweep — the failure this gate exists for.
                fail(
                    &mut lines,
                    format!(
                        "{}: the Wukong column did not compile, so there is no lowering to describe \
                         — a benchmark that stopped building is a failure, not context",
                        s.program
                    ),
                );
            } else if s.census.starts_with("NOTHING") {
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

    // The recognition on/off premise at measurement size (P12.24): the recognition-off rows below
    // are read beside recognizer-on columns elsewhere, and that reading rests on this premise.
    let premise_ok = crate::compare::premise_check();
    let mut sweep = crate::general::bench_general(cc, cxx, dir);
    println!();
    sweep.extend(crate::compare::bench_compare(cc, dir));
    println!();
    crate::print_sota_scorecard(&sweep);

    println!();
    println!("sota-check verdicts:");
    let (lines, mut failed) = evaluate(&floors, &sweep);
    for l in &lines {
        println!("{l}");
    }
    if !premise_ok {
        println!(
            "  FAIL  recognition on/off premise (tests/run/premise_*.wk): see the `! premise` lines \
             above"
        );
        failed = true;
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
        // Tight on purpose: rounds spanning 0.30-0.34 are a 1.13x spread, past the harness's own
        // 1.10x publishing limit, and `evaluate` now refuses to grade a median its rounds do not
        // support. This test is about a tier reported BELOW bar without failing the gate, so it has
        // to hand `evaluate` a ratio the run actually resolved.
        s.ratio = Some(stat(0.32, 0.31, 0.33));
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

    /// A control whose own rounds disagree is not a resolution, and must not be spent as one.
    ///
    /// This is the direction that matters: `floor_factor` grows with a degraded control, and a wider
    /// tolerance makes the ratchet *more* permissive — so before this gate the worst runs were the
    /// hardest to catch saying anything wrong.
    #[test]
    fn a_control_that_did_not_settle_cannot_certify() {
        let mut s = row("p");
        s.twin = Some(stat(1.00, 0.80, 1.40)); // spread 1.75x, far outside the publishing limit
        let (lines, failed) = evaluate(&[floor("p", 1.00)], &[s]);
        assert!(failed, "an unsettled control certified a floor: {lines:?}");
        assert!(
            lines.iter().any(|l| l.contains("did not settle")),
            "the failure must name the control: {lines:?}"
        );
    }

    /// A control that settled but landed far from its own true value of 1.00 is a wide noise floor,
    /// and a noise floor wider than the harness's publishing limit cannot certify claims made to it.
    #[test]
    fn a_control_wider_than_the_publishing_limit_cannot_certify() {
        let mut s = row("p");
        s.twin = Some(stat(1.15, 1.11, 1.20)); // spread 1.08x (settled) but 1.20x off 1.00
        let (lines, failed) = evaluate(&[floor("p", 1.00)], &[s]);
        assert!(failed, "a 1.20x noise floor certified a floor: {lines:?}");
        assert!(
            lines.iter().any(|l| l.contains("noise floor is wider")),
            "the failure must name the floor width: {lines:?}"
        );
    }

    /// `evaluate` grades a median, and a median is a measurement only when its rounds agree.
    #[test]
    fn an_inconclusive_ratio_is_not_graded_against_a_floor() {
        let mut s = row("p");
        s.ratio = Some(stat(1.05, 0.90, 1.50)); // spread 1.67x
        let (lines, failed) = evaluate(&[floor("p", 1.00)], &[s]);
        assert!(failed, "an inconclusive ratio was graded: {lines:?}");
        assert!(
            lines.iter().any(|l| l.contains("INCONCLUSIVE")),
            "the failure must say so: {lines:?}"
        );
    }

    /// A median above the bar is not a tier when the round range overlaps the control's own band —
    /// that is the instrument reporting its jitter as a result.
    #[test]
    fn a_tier_inside_the_noise_floor_is_not_announced() {
        let mut s = row("p");
        s.ratio = Some(stat(1.01, 0.99, 1.03)); // settled, but straddles the 1.02x control band
        s.twin = Some(stat(1.00, 0.98, 1.02));
        let (lines, failed) = evaluate(&[floor("p", 1.00)], &[s]);
        assert!(!failed, "holding the floor is not a failure: {lines:?}");
        assert!(
            lines.iter().any(|l| l.contains("tier NOT RESOLVED")),
            "an unresolved tier must say so: {lines:?}"
        );
        assert!(
            !lines.iter().any(|l| l.contains("at or above")),
            "a tier was announced from inside the noise floor: {lines:?}"
        );
    }

    /// Off Windows there is no power reading at all, so the gate cannot tell AC+full from battery
    /// and must refuse. It has always refused; what it did was emit one power-STATE complaint per
    /// floored row, which reads like "unplug the laptop" rather than "this gate does not run here".
    /// One refusal, naming the platform, and no per-row noise.
    #[test]
    fn a_sweep_with_no_power_reading_refuses_once_and_says_why() {
        let mut a = row("p");
        a.ratio = Some(stat(1.40, 1.36, 1.44));
        a.twin = Some(stat(1.00, 0.98, 1.02));
        a.power = POWER_UNAVAILABLE.into();
        let mut b = row("q");
        b.ratio = Some(stat(1.40, 1.36, 1.44));
        b.twin = Some(stat(1.00, 0.98, 1.02));
        b.power = POWER_UNAVAILABLE.into();
        let (lines, failed) = evaluate(&[floor("p", 1.20), floor("q", 1.20)], &[a, b]);
        assert!(failed, "a sweep with no power reading must not pass");
        let fails: Vec<&String> = lines.iter().filter(|l| l.contains("FAIL")).collect();
        assert_eq!(fails.len(), 1, "one refusal, not one per row: {lines:?}");
        assert!(
            fails[0].contains("Windows only"),
            "the refusal must name the platform limitation: {:?}",
            fails[0]
        );
    }

    /// The whole set above must not have made the ordinary passing row impossible.
    #[test]
    fn a_clean_row_still_passes_every_new_gate() {
        let mut s = row("p");
        s.ratio = Some(stat(1.40, 1.36, 1.44)); // settled, and clear of the control band
        s.twin = Some(stat(1.00, 0.98, 1.02));
        let (lines, failed) = evaluate(&[floor("p", 1.20)], &[s]);
        assert!(!failed, "a clean row failed: {lines:?}");
        assert!(
            lines.iter().any(|l| l.contains("at or above PARITY")),
            "a resolved row must still report its tier: {lines:?}"
        );
    }

    /// A repeated program name is not a duplicate record — it is a floor that will never be graded,
    /// because `evaluate` looks the sweep up by name and takes the first match.
    #[test]
    fn a_floor_file_with_a_repeated_program_is_rejected() {
        let text =
            "| program | peer | tier | floor | date | power | twin | cause | owner | wave |\n\
|---|---|---|---|---|---|---|---|---|---|\n\
| general/s6_scan:Wukong(gen) | C | PARITY | 1.00x | d | AC+full | t | — | — | — |\n\
| general/s6_scan:Wukong(gen) | C | PARITY | 1.20x | d | AC+full | t | — | — | — |\n";
        let err = parse_floor(text).unwrap_err();
        assert!(err.contains("appears twice"), "{err}");
    }

    /// A Wukong column that never built is a failure, not "context".
    ///
    /// The sweep row still exists and still carries a census string, so it reached the unfloored-row
    /// arm and was announced as a recognizer-dialect row with the gate green. The other row here is
    /// healthy and floored, so the gate would otherwise pass — the failure has to come from `p`.
    #[test]
    fn a_column_that_did_not_compile_fails_rather_than_reading_as_context() {
        let mut bad = row("p");
        bad.census = CENSUS_DID_NOT_COMPILE.to_string();
        let (lines, failed) = evaluate(&[floor("q", 1.00)], &[row("q"), bad]);
        assert!(
            failed,
            "a column that never built passed the gate: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("did not compile")),
            "the failure must say what happened: {lines:?}"
        );
        assert!(
            !lines.iter().any(|l| l.contains("off-bar")),
            "a column that never built was described as context: {lines:?}"
        );
    }
}
