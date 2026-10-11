//! The numerics gate: a cross-language mismatch WITHHOLDS the section's standings.
//!
//! THE DEFECT (P8.3, 2026-09-13 sweep). BENCHMARKS.md opens with the guarantee that the harness
//! "cross-checks a result checksum across every language, so a miscompiled kernel is caught, not
//! silently mis-measured". Every one of the ~45 mismatch sites in `main.rs` was a bare `println!`:
//! the `!` line was printed, and two lines later the SAME section printed
//! `-> Wukong single-core is 1.31x faster than idiomatic C` from the very buffers that had just
//! failed the check. A divergent row rendered as a clean ratio - the harness could not tell "we are
//! faster" from "we are computing something else faster", which is exactly what the bf16 W9n
//! investigation then had to untangle by hand.
//!
//! THE FIX is structural rather than advisory, and it is three macros, so that the decision lives
//! in one place instead of at ~60 print sites:
//!
//! * [`section!`] prints a `=== ... ===` heading and opens a new numerics scope.
//! * [`mismatch!`] prints a `!` numerical-failure line and marks the open scope SUSPECT.
//! * [`standing!`] prints a `-> Wukong ... is N.NNx faster/slower ...` standing - unless the scope
//!   is SUSPECT, in which case the ratio is WITHHELD: the reader sees `-> SUSPECT: ratio withheld`,
//!   and the original sentence survives only as a `. withheld:` note, which neither
//!   `extract-kernels.py` nor a human skimming for `->` lines reads as a result.
//!
//! A mismatch found AFTER a standing was already printed cannot un-print it, so the gate says so
//! in a `!` line of its own (`SUSPECT: N standing line(s) above ...`), and the section is listed in
//! the end-of-run summary either way. That summary is also the exit code: a run in which any
//! section reported a numerical mismatch exits [`NUMERICS_EXIT`], so the campaign runner records
//! the process as not clean instead of publishing it. `XBENCH_NUMERICS=report` keeps the exit code
//! at zero for exploratory runs and says so in the log.
//!
//! What is deliberately NOT a mismatch: a peer column DROPPED because it disagreed (C++ vs C, a
//! relaxed C(fast)/C(omp) peer, an MKL ABI disagreement). A dropped column prints no ratio at all,
//! so there is nothing to withhold, and those lines keep their plain `println!`.
//!
//! UNCHECKED (P8.3, second half). Withholding a ratio after a FAILED check is only half of the
//! guarantee. The other half is that a check RAN: a section whose only comparison was skipped - the
//! C peer did not build, the check sat after the standing, or the section never compared Wukong's
//! buffer against anything - used to print `-> 1.31x faster` with no `!` line anywhere, and the
//! clean end-of-run summary then certified it. So the ledger also counts the checks that ran in the
//! open section:
//!
//! * [`checked!`] records that a numerics check of Wukong's output against an independent reference
//!   ran. It sits where the comparison is made, BEFORE the branch that may `mismatch!`, so a check
//!   that passes counts as well. A [`mismatch!`] counts as a check by itself.
//! * A [`standing!`] in a section with ZERO checks is WITHHELD the same way a SUSPECT one is - the
//!   reader sees `-> UNCHECKED: ratio withheld` and a `. withheld:` note - and the section is listed
//!   in the end-of-run summary, which exits [`NUMERICS_EXIT`] (unless `XBENCH_NUMERICS=report`).
//!
//! What counts as a check is deliberately narrow: a comparison whose failure is a `mismatch!`. The
//! peer filters (`relaxed_peer_ok`, the VML columns, `cpp_checked`, `mkl_transpose_exact`) do not
//! count, because their failure DROPS a peer column rather than flagging Wukong - a section whose
//! only comparison was one of those could print a ratio against a peer nothing compared Wukong with.

use std::sync::Mutex;

/// The process exit code of a run whose numerics gate failed. Distinct from Rust's panic exit
/// (101) and from the `sota-check` mode's codes, so a runner log can name the cause.
pub(crate) const NUMERICS_EXIT: i32 = 4;

/// Per-run, per-section numerics state. Pure: every decision the macros make is a method here, so
/// it is unit-testable without capturing stdout or touching the process-global instance.
pub(crate) struct Ledger {
    /// The heading of the open scope (empty before the first [`section!`]).
    section: String,
    /// The open scope reported a numerical mismatch.
    suspect: bool,
    /// Numerics checks that ran in the open scope ([`checked!`]; a [`mismatch!`] counts too).
    checks: usize,
    /// Standings printed in the open scope while it was still clean.
    clean_standings: usize,
    /// Every heading that went SUSPECT, in order, without duplicates.
    suspect_sections: Vec<String>,
    /// Every heading that printed a standing while no numerics check had run in it, in order,
    /// without duplicates.
    unchecked_sections: Vec<String>,
    /// The open scope's `C(twin)` single-thread floor (see `dispersion::twin_control`), if one ran.
    twin_floor: Option<f64>,
    /// The open scope's all-core twin floor (see `dispersion::all_core_twin_control`), if one ran.
    all_core_floor: Option<f64>,
}

impl Ledger {
    pub(crate) const fn new() -> Self {
        Ledger {
            section: String::new(),
            suspect: false,
            checks: 0,
            clean_standings: 0,
            suspect_sections: Vec::new(),
            unchecked_sections: Vec::new(),
            twin_floor: None,
            all_core_floor: None,
        }
    }

    /// Open a new scope. Returns the line to print (the heading itself).
    pub(crate) fn begin(&mut self, heading: &str) -> Vec<String> {
        self.section = heading.trim().to_string();
        self.suspect = false;
        self.checks = 0;
        self.clean_standings = 0;
        self.twin_floor = None;
        self.all_core_floor = None;
        vec![heading.to_string()]
    }

    /// Record an all-core twin floor for the open scope (widest wins, like the single-thread one).
    pub(crate) fn record_all_core_floor(&mut self, floor: f64) {
        if floor.is_finite() && floor >= 1.0 {
            self.all_core_floor = Some(self.all_core_floor.map_or(floor, |f| f.max(floor)));
        }
    }

    /// Record a `C(twin)` floor for the open scope. Several controls in one scope keep the widest:
    /// a floor is a statement of what the run could NOT resolve, so the conservative merge is max.
    pub(crate) fn record_twin_floor(&mut self, floor: f64) {
        if floor.is_finite() && floor >= 1.0 {
            self.twin_floor = Some(self.twin_floor.map_or(floor, |f| f.max(floor)));
        }
    }

    /// Record a numerical mismatch. Returns the lines to print: the mismatch itself, plus a
    /// retroactive `!` line when standings were already printed in this scope.
    pub(crate) fn mismatch(&mut self, line: &str) -> Vec<String> {
        let mut out = vec![line.to_string()];
        // A mismatch is the outcome of a check, so it is one: a section whose only check failed is
        // SUSPECT, not UNCHECKED.
        self.checks += 1;
        if !self.suspect {
            self.suspect = true;
            let name = self.section_name();
            if !self.suspect_sections.contains(&name) {
                self.suspect_sections.push(name);
            }
            if self.clean_standings > 0 {
                out.push(format!(
                    "  ! SUSPECT: {} standing line(s) printed above in this section preceded this \
                     numerical mismatch - treat them as withheld",
                    self.clean_standings
                ));
            }
        }
        out
    }

    /// Record that a numerics check of Wukong's output ran in the open scope (see [`checked!`]).
    pub(crate) fn check(&mut self) {
        self.checks += 1;
    }

    /// The open scope's name for a summary list.
    fn section_name(&self) -> String {
        if self.section.is_empty() {
            "(before the first section heading)".to_string()
        } else {
            self.section.clone()
        }
    }

    /// Route one standing line. Returns the lines to print.
    pub(crate) fn standing(&mut self, line: &str) -> Vec<String> {
        if self.suspect {
            return vec![
                "  -> SUSPECT: ratio withheld - this section reported a numerical mismatch (see \
                 the `!` line above)"
                    .to_string(),
                format!("  . withheld: {}", line.trim_start()),
            ];
        }
        // No check has run in this section yet: nothing compared the buffer this ratio was timed on
        // against anything, so the ratio is not yet a measurement of the right computation. It is
        // withheld exactly like a SUSPECT one, and the section is named in the summary.
        if self.checks == 0 {
            let name = self.section_name();
            if !self.unchecked_sections.contains(&name) {
                self.unchecked_sections.push(name);
            }
            return vec![
                "  -> UNCHECKED: ratio withheld - no numerics check of Wukong's output ran in this \
                 section before this standing"
                    .to_string(),
                format!("  . withheld: {}", line.trim_start()),
            ];
        }
        self.clean_standings += 1;
        let mut out = line.to_string();
        // A single-threaded standing no larger than the section's own zero-difference control is
        // not a measured difference. It is still printed - the reader is owed the number - but it
        // carries the qualification on its own line, where neither a skimming human nor the
        // extractor can separate the two. The C(twin) control is a single-threaded column and says
        // nothing about where the scheduler put 16 workers, so it never qualifies an all-core
        // standing on its own.
        //
        // An all-core standing is qualified by the all-core twin instead, and - when it also has a
        // single-threaded side the C(twin) measured - by the wider of the two, because its ratio
        // moves when either side moves. With no all-core control it stays untagged rather than be
        // judged by a single-threaded column's noise.
        //
        // The open section's own heading counts too: the elementwise `@parallel` rows print
        // `Wukong runtime is ...` with no marker on the line, because the whole row is all-core.
        let single =
            is_single_threaded_standing(line) && is_single_threaded_standing(&self.section);
        let (floor, tag) = if single {
            (self.twin_floor, "C(twin)")
        } else {
            (
                self.all_core_floor
                    .map(|a| self.twin_floor.map_or(a, |s| s.max(a))),
                "ALL-CORE TWIN",
            )
        };
        if let (Some(floor), Some(ratio)) = (floor, standing_ratio(line)) {
            if ratio <= floor {
                out.push_str(&format!(
                    "  [BELOW {tag} FLOOR {floor:.3}x: not established]"
                ));
            }
        }
        vec![out]
    }

    /// A table row whose ratios live in its own CELLS (the norm and streaming tables) and for which
    /// no numerics check ran: the row printed `UNCHECKED` in those cells instead of a ratio, and the
    /// section is named in the summary exactly as a withheld [`standing!`] would name it. Returns
    /// the line to print.
    pub(crate) fn unchecked_cells(&mut self, row: &str) -> Vec<String> {
        let name = self.section_name();
        if !self.unchecked_sections.contains(&name) {
            self.unchecked_sections.push(name);
        }
        vec![format!(
            "  -> UNCHECKED: ratio withheld - no numerics check of Wukong's output ran for row \
             `{row}`, so its ratio cells read UNCHECKED"
        )]
    }

    /// Whether the open scope has reported a numerical mismatch. The cell tables consult this so a
    /// row after an earlier row's mismatch prints SUSPECT in its cells too.
    pub(crate) fn is_suspect(&self) -> bool {
        self.suspect
    }

    /// Whether no numerics check has run in the open scope yet - the state in which a
    /// [`standing!`] is withheld as UNCHECKED.
    pub(crate) fn is_unchecked(&self) -> bool {
        self.checks == 0
    }

    /// Whether a ratio computed in the open scope may be pooled into a geomean: exactly when a
    /// [`standing!`] printed now would print its ratio, i.e. the scope is neither SUSPECT nor
    /// UNCHECKED. A summary geomean is a standing too, so pooling a ratio whose own `->` line was
    /// withheld would publish in the headline what the gate refused on the row.
    pub(crate) fn may_pool(&self) -> bool {
        !self.suspect && !self.is_unchecked()
    }

    #[cfg(test)]
    pub(crate) fn suspect_sections(&self) -> &[String] {
        &self.suspect_sections
    }

    #[cfg(test)]
    pub(crate) fn unchecked_sections(&self) -> &[String] {
        &self.unchecked_sections
    }

    /// The end-of-run summary lines and the exit code it implies. `report_only` is
    /// `XBENCH_NUMERICS=report`.
    pub(crate) fn finish(&self, report_only: bool) -> (Vec<String>, i32) {
        let mut out = vec!["=== numerics gate ===".to_string()];
        if self.suspect_sections.is_empty() && self.unchecked_sections.is_empty() {
            out.push(
                "  numerics gate: every section that printed a standing ran at least one numerics \
                 check before it, and every check agreed; no standing was withheld."
                    .to_string(),
            );
            return (out, 0);
        }
        if !self.suspect_sections.is_empty() {
            out.push(format!(
                "  ! NUMERICS GATE FAILED: {} section(s) reported a numerical mismatch, and every \
                 standing in them was withheld:",
                self.suspect_sections.len()
            ));
            for s in &self.suspect_sections {
                out.push(format!("  !   {s}"));
            }
        }
        if !self.unchecked_sections.is_empty() {
            out.push(format!(
                "  ! NUMERICS GATE FAILED: {} section(s) printed a standing before any numerics \
                 check ran in them - UNCHECKED, ratio withheld:",
                self.unchecked_sections.len()
            ));
            for s in &self.unchecked_sections {
                out.push(format!("  !   UNCHECKED {s}"));
            }
        }
        if report_only {
            out.push(
                "  (XBENCH_NUMERICS=report: exit code left at 0 for an exploratory run - this log \
                 is not publishable)"
                    .to_string(),
            );
            (out, 0)
        } else {
            (out, NUMERICS_EXIT)
        }
    }
}

/// The factor a standing states - `N.NN` in `N.NNx faster` / `N.NNx slower` - or `None` when the line
/// states no such factor (a %-of-roofline line, for one).
///
/// The FIRST such factor on the line is the standing: `report` appends a compile-time clause
/// (`... than C; compiles 3.0x faster`) after the runtime one, and reading that one instead would
/// judge a runtime ratio by a compile ratio.
pub(crate) fn standing_ratio(line: &str) -> Option<f64> {
    let at = ["x faster", "x slower"]
        .into_iter()
        .filter_map(|word| line.find(word))
        .min()?;
    let head = &line[..at];
    let start = head
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_ascii_digit() || *c == '.')
        .last()
        .map(|(i, _)| i)?;
    head[start..].parse::<f64>().ok().filter(|r| r.is_finite())
}

/// Markers of a standing that involves an all-core column on either side.
const MULTICORE_MARKERS: [&str; 9] = [
    "@parallel",
    "all cores",
    "all-core",
    "all threads",
    "MKL(all)",
    "C(omp)",
    "openmp",
    "multicore",
    "Tn(",
];

/// Whether both sides of a standing are single-threaded, read from the words the harness prints.
/// Conservative in the direction that matters: an unrecognised all-core spelling is merely left
/// untagged by a single-thread floor that never applied to it anyway.
pub(crate) fn is_single_threaded_standing(line: &str) -> bool {
    !MULTICORE_MARKERS.iter().any(|m| line.contains(m))
}

static LEDGER: Mutex<Ledger> = Mutex::new(Ledger::new());

fn with_ledger<T>(f: impl FnOnce(&mut Ledger) -> T) -> T {
    // A poisoned lock means a bench panicked mid-print; the ledger itself is still consistent.
    let mut guard = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

fn emit(lines: Vec<String>) {
    for l in lines {
        println!("{l}");
    }
}

pub(crate) fn section_begin(heading: &str) {
    emit(with_ledger(|l| l.begin(heading)));
}

pub(crate) fn report_mismatch(line: &str) {
    emit(with_ledger(|l| l.mismatch(line)));
}

pub(crate) fn report_standing(line: &str) {
    emit(with_ledger(|l| l.standing(line)));
}

/// Record that a numerics check ran in the open section (see [`Ledger::check`]).
pub(crate) fn record_check() {
    with_ledger(|l| l.check());
}

/// A cell-ratio table row printed `UNCHECKED` cells (see [`Ledger::unchecked_cells`]).
pub(crate) fn report_unchecked_cells(row: &str) {
    emit(with_ledger(|l| l.unchecked_cells(row)));
}

/// Record the open section's all-core twin floor (see [`Ledger::record_all_core_floor`]).
pub(crate) fn record_all_core_floor(floor: f64) {
    with_ledger(|l| l.record_all_core_floor(floor));
}

/// Record the open section's `C(twin)` floor (see [`Ledger::record_twin_floor`]).
pub(crate) fn record_twin_floor(floor: f64) {
    with_ledger(|l| l.record_twin_floor(floor));
}

/// Whether the open section has reported a numerical mismatch (see [`Ledger::is_suspect`]).
pub(crate) fn section_suspect() -> bool {
    with_ledger(|l| l.is_suspect())
}

/// Whether a ratio computed in the open section may enter a geomean (see [`Ledger::may_pool`]).
/// Every summary pool is gated on this, never on [`section_suspect`] alone.
pub(crate) fn section_may_pool() -> bool {
    with_ledger(|l| l.may_pool())
}

/// Print the end-of-run summary and return the exit code the run should end with.
pub(crate) fn finish() -> i32 {
    let report_only = std::env::var("XBENCH_NUMERICS")
        .map(|v| v.trim().eq_ignore_ascii_case("report"))
        .unwrap_or(false);
    let (lines, code) = with_ledger(|l| l.finish(report_only));
    println!();
    emit(lines);
    code
}

/// Print a `=== ... ===` section heading and open a new numerics scope.
macro_rules! section {
    ($($arg:tt)*) => {
        $crate::numerics::section_begin(&format!($($arg)*))
    };
}

/// Print a `!` numerical-mismatch line and mark the open section SUSPECT.
macro_rules! mismatch {
    ($($arg:tt)*) => {
        $crate::numerics::report_mismatch(&format!($($arg)*))
    };
}

/// Record that a numerics check of Wukong's output against an independent reference ran in the
/// open section. Place it where the comparison is made, before the branch that may `mismatch!`, so
/// a check that PASSES counts too. Prints nothing.
macro_rules! checked {
    () => {
        $crate::numerics::record_check()
    };
}

/// Print a `->` standing, or withhold it when the open section is SUSPECT or UNCHECKED.
macro_rules! standing {
    ($($arg:tt)*) => {
        $crate::numerics::report_standing(&format!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_standing_after_a_mismatch_is_withheld_not_printed() {
        let mut l = Ledger::new();
        l.begin("=== colsum 1024x1024 ===");
        l.mismatch("  ! colsum output mismatch vs C");
        let out = l.standing("  -> Wukong single-core is 1.31x faster than idiomatic C");
        assert_eq!(out.len(), 2, "{out:?}");
        assert!(out[0].contains("SUSPECT: ratio withheld"), "{out:?}");
        assert!(
            !out[0].contains("1.31x"),
            "the withheld ratio must not be on the -> line"
        );
        assert!(out[1].starts_with("  . withheld: -> Wukong"), "{out:?}");
    }

    #[test]
    fn a_clean_section_prints_its_standing_verbatim() {
        let mut l = Ledger::new();
        l.begin("=== colsum 1024x1024 ===");
        l.check();
        let line = "  -> Wukong single-core is 1.31x faster than idiomatic C";
        assert_eq!(l.standing(line), vec![line.to_string()]);
        assert!(l.suspect_sections().is_empty());
        assert!(l.unchecked_sections().is_empty());
    }

    #[test]
    fn the_suspect_flag_is_scoped_to_the_open_section() {
        let mut l = Ledger::new();
        l.begin("=== a ===");
        assert!(!l.is_suspect());
        l.mismatch("  ! a mismatch vs C");
        assert!(l.is_suspect());
        l.begin("=== b ===");
        assert!(!l.is_suspect(), "a new heading must open a clean scope");
    }

    #[test]
    fn standing_ratio_reads_the_printed_factor_and_nothing_else() {
        assert_eq!(
            standing_ratio("  -> Wukong single-core is 1.31x faster than idiomatic C"),
            Some(1.31)
        );
        assert_eq!(
            standing_ratio("  -> Wukong @parallel is 12.40x slower than MKL(all)"),
            Some(12.4)
        );
        assert_eq!(
            standing_ratio("  -> Wukong single-core = 85% of measured roofline"),
            None
        );
        assert_eq!(standing_ratio("x faster"), None);
        // The runtime factor comes first; the compile-time clause after it is not the standing.
        assert_eq!(
            standing_ratio("  -> Wukong runtime is 1.02x slower than C; compiles 3.0x faster"),
            Some(1.02)
        );
    }

    #[test]
    fn a_single_threaded_standing_inside_the_twin_floor_is_tagged_not_hidden() {
        let mut l = Ledger::new();
        l.begin("=== colsum 1024x1024 ===");
        l.check();
        l.record_twin_floor(1.03);
        let inside = l.standing("  -> Wukong single-core is 1.02x slower than idiomatic C");
        assert_eq!(inside.len(), 1);
        assert!(
            inside[0].starts_with("  -> Wukong single-core is 1.02x slower"),
            "{inside:?}"
        );
        assert!(
            inside[0].contains("[BELOW C(twin) FLOOR 1.030x"),
            "{inside:?}"
        );
        let outside = l.standing("  -> Wukong single-core is 1.31x faster than idiomatic C");
        assert!(!outside[0].contains("FLOOR"), "{outside:?}");
        // The control is single-threaded; it never qualifies an all-core standing.
        let par = l.standing("  -> Wukong @parallel is 1.01x faster than idiomatic C");
        assert!(!par[0].contains("FLOOR"), "{par:?}");
    }

    #[test]
    fn an_all_core_standing_is_judged_by_the_all_core_twin_not_the_single_one() {
        let mut l = Ledger::new();
        l.begin("=== matmul 512x512 ===");
        l.check();
        l.record_twin_floor(1.02);
        // Only a single-threaded control: an all-core standing stays untagged.
        let par = l.standing("  -> Wukong @parallel is 1.05x slower than MKL(all)");
        assert!(!par[0].contains("FLOOR"), "{par:?}");
        l.record_all_core_floor(1.30);
        let par = l.standing("  -> Wukong @parallel is 1.25x slower than MKL(all)");
        assert!(
            par[0].contains("[BELOW ALL-CORE TWIN FLOOR 1.300x"),
            "{par:?}"
        );
        // The single-threaded standing still uses the single-threaded control only.
        let one = l.standing("  -> Wukong single-core is 1.25x slower than oneMKL(1 thread)");
        assert!(!one[0].contains("FLOOR"), "{one:?}");
        // A cross-threading standing takes the wider of the two.
        let mut l = Ledger::new();
        l.begin("=== colsum ===");
        l.check();
        l.record_twin_floor(1.08);
        l.record_all_core_floor(1.03);
        let cross =
            l.standing("  -> Wukong @parallel is 1.07x faster than idiomatic single-threaded C");
        assert!(cross[0].contains("ALL-CORE TWIN FLOOR 1.080x"), "{cross:?}");
        l.begin("=== next ===");
        l.check();
        let cross =
            l.standing("  -> Wukong @parallel is 1.07x faster than idiomatic single-threaded C");
        assert!(
            !cross[0].contains("FLOOR"),
            "a new heading drops both floors: {cross:?}"
        );
    }

    #[test]
    fn an_all_core_section_heading_makes_its_unmarked_standings_all_core() {
        let mut l = Ledger::new();
        l.begin("=== saxpy@parallel (y = a*x + y, memory-bound) ===");
        l.check();
        l.record_twin_floor(1.05);
        let s = l.standing("  -> Wukong runtime is 1.01x slower than C; compiles 3.0x faster");
        assert!(
            !s[0].contains("FLOOR"),
            "an all-core row judged by C(twin): {s:?}"
        );
        l.record_all_core_floor(1.04);
        let s = l.standing("  -> Wukong runtime is 1.01x slower than C; compiles 3.0x faster");
        assert!(s[0].contains("ALL-CORE TWIN FLOOR 1.050x"), "{s:?}");
    }

    #[test]
    fn the_twin_floor_is_scoped_and_merges_conservatively() {
        let mut l = Ledger::new();
        l.begin("=== a ===");
        l.check();
        l.record_twin_floor(1.02);
        l.record_twin_floor(1.05);
        l.record_twin_floor(1.01);
        l.record_twin_floor(f64::NAN);
        let s = l.standing("  -> Wukong single-core is 1.04x faster than C");
        assert!(s[0].contains("FLOOR 1.050x"), "{s:?}");
        l.begin("=== b ===");
        l.check();
        let s = l.standing("  -> Wukong single-core is 1.04x faster than C");
        assert!(
            !s[0].contains("FLOOR"),
            "a new heading must drop the previous floor: {s:?}"
        );
    }

    #[test]
    fn multicore_markers_cover_every_all_core_spelling_the_harness_prints() {
        for line in [
            "  -> Wukong @parallel is 1.10x faster than idiomatic single-threaded C",
            "  -> Wukong single-core is 1.10x slower than C(omp) [-fopenmp -ffast-math, all cores]",
            "  -> Wukong 1-core is 1.10x slower than MKL(all)",
            "  -> Wukong (1 core) is 1.10x faster than PyTorch eager SDPA (all threads)",
        ] {
            assert!(!is_single_threaded_standing(line), "{line}");
        }
        assert!(is_single_threaded_standing(
            "  -> Wukong single-core is 1.10x faster than oneMKL(1 thread)"
        ));
    }

    #[test]
    fn a_new_heading_clears_the_suspect_scope() {
        let mut l = Ledger::new();
        l.begin("=== a ===");
        l.mismatch("  ! a mismatch vs C");
        l.begin("=== b ===");
        l.check();
        let line = "  -> Wukong single-core is 1.10x faster than idiomatic C";
        assert_eq!(l.standing(line), vec![line.to_string()]);
        assert_eq!(l.suspect_sections(), &["=== a ===".to_string()]);
    }

    #[test]
    fn a_late_mismatch_flags_the_standings_already_printed() {
        let mut l = Ledger::new();
        l.begin("=== matmul 256x256 ===");
        // A check ran and passed (the 1-thread MKL buffer agreed), the standing printed clean, and a
        // LATER check failed.
        l.check();
        l.standing("  -> Wukong 1-core is 1.05x faster than oneMKL(1 thread)");
        let out = l.mismatch("  ! Wukong vs MKL full-buffer mismatch at [3] (rel 1.0e-2)");
        assert_eq!(out.len(), 2, "{out:?}");
        assert!(out[1].contains("1 standing line(s)"), "{out:?}");
        // Only the first mismatch of a scope adds the retro line or a summary entry.
        assert_eq!(l.mismatch("  ! again").len(), 1);
        assert_eq!(l.suspect_sections().len(), 1);
    }

    #[test]
    fn the_gate_sets_the_exit_code_unless_report_only() {
        let mut l = Ledger::new();
        assert_eq!(l.finish(false).1, 0);
        l.begin("=== x ===");
        l.mismatch("  ! x mismatch vs C");
        let (lines, code) = l.finish(false);
        assert_eq!(code, NUMERICS_EXIT);
        assert!(lines.iter().any(|s| s.contains("=== x ===")), "{lines:?}");
        assert_eq!(l.finish(true).1, 0);
    }

    /// The clean summary is read by `analyze-sota-measure.py`, whose numerical-warning scan keys on
    /// a leading `!`. A clean run must not print one.
    #[test]
    fn the_clean_summary_carries_no_warning_marker() {
        let (lines, _) = Ledger::new().finish(false);
        assert!(
            lines.iter().all(|s| !s.trim_start().starts_with('!')),
            "{lines:?}"
        );
    }

    /// The defect this half of P8.3 closes: a section that printed a ratio while nothing had
    /// compared Wukong's buffer against anything printed it CLEAN, and the end-of-run summary then
    /// certified the run. Zero checks is now its own withheld state.
    #[test]
    fn a_standing_in_a_section_with_no_check_is_withheld_as_unchecked() {
        let mut l = Ledger::new();
        l.begin("=== linear 512x512 ===");
        let out = l.standing("  -> Wukong single-core is 1.31x faster than idiomatic C");
        assert_eq!(out.len(), 2, "{out:?}");
        assert!(
            out[0].starts_with("  -> UNCHECKED: ratio withheld"),
            "{out:?}"
        );
        assert!(
            !out[0].contains("1.31x"),
            "the withheld ratio must not be on the -> line: {out:?}"
        );
        assert_eq!(
            out[1],
            "  . withheld: -> Wukong single-core is 1.31x faster than idiomatic C"
        );
        assert!(!l.is_suspect(), "UNCHECKED is not a numerical mismatch");
        assert_eq!(
            l.unchecked_sections(),
            &["=== linear 512x512 ===".to_string()]
        );
        assert!(l.suspect_sections().is_empty());
        // A second standing in the same section is withheld too, and listed once.
        let again = l.standing("  -> Wukong @parallel is 2.00x faster than idiomatic C");
        assert!(again[0].contains("UNCHECKED"), "{again:?}");
        assert_eq!(l.unchecked_sections().len(), 1);
    }

    /// Order matters, and it is the order the harness prints in: a standing printed BEFORE the
    /// section's check is withheld even though a check runs later - which is what makes a
    /// check-after-standing section show up instead of reading clean.
    #[test]
    fn a_check_counts_only_for_standings_printed_after_it() {
        let mut l = Ledger::new();
        l.begin("=== matmul_tn 256x256 ===");
        let early = l.standing("  -> Wukong single-core = 85% of measured roofline");
        assert!(early[0].contains("UNCHECKED"), "{early:?}");
        l.check();
        let line = "  -> Wukong single-core (SIMD) is 3.20x faster than C single-threaded";
        assert_eq!(l.standing(line), vec![line.to_string()]);
        assert_eq!(
            l.unchecked_sections(),
            &["=== matmul_tn 256x256 ===".to_string()],
            "the early standing was withheld, so the section is still named"
        );
    }

    /// The norm and streaming tables carry their ratios in cells, not in `standing!` lines; a row
    /// with no check reports itself and lands in the same summary list.
    #[test]
    fn an_unchecked_cell_row_is_named_in_the_summary() {
        let mut l = Ledger::new();
        l.begin("=== norm (single row) ===");
        let out = l.unchecked_cells("rmsnorm");
        assert_eq!(out.len(), 1);
        assert!(
            out[0].starts_with("  -> UNCHECKED: ratio withheld"),
            "{out:?}"
        );
        assert!(out[0].contains("`rmsnorm`"), "{out:?}");
        assert_eq!(
            l.unchecked_sections(),
            &["=== norm (single row) ===".to_string()]
        );
        l.unchecked_cells("layernorm");
        assert_eq!(l.unchecked_sections().len(), 1, "one section, listed once");
        assert_eq!(l.finish(false).1, NUMERICS_EXIT);
    }

    /// A failed check is still a check: the section is SUSPECT, never UNCHECKED, and the failure
    /// withholds with the SUSPECT wording.
    #[test]
    fn a_mismatch_counts_as_the_check_that_ran() {
        let mut l = Ledger::new();
        l.begin("=== gemv 4096x4096 ===");
        l.mismatch("  ! gemv mismatch vs C: max|d|/max|C| = 1.00e-2");
        let out = l.standing("  -> Wukong single-core is 1.10x faster than idiomatic C");
        assert!(out[0].contains("SUSPECT"), "{out:?}");
        assert!(l.unchecked_sections().is_empty());
        assert_eq!(l.suspect_sections().len(), 1);
    }

    /// A new heading resets the count: a check in one section certifies nothing in the next.
    #[test]
    fn the_check_count_is_scoped_to_the_open_section() {
        let mut l = Ledger::new();
        l.begin("=== a ===");
        l.check();
        let line = "  -> Wukong single-core is 1.10x faster than idiomatic C";
        assert_eq!(l.standing(line), vec![line.to_string()]);
        l.begin("=== b ===");
        assert!(l.standing(line)[0].contains("UNCHECKED"));
        assert_eq!(l.unchecked_sections(), &["=== b ===".to_string()]);
    }

    /// A standing withheld as UNCHECKED was never printed as a ratio, so a later mismatch in the
    /// same section owes no retroactive `!` line for it.
    #[test]
    fn a_late_mismatch_does_not_double_count_an_unchecked_standing() {
        let mut l = Ledger::new();
        l.begin("=== ffn 512 ===");
        l.standing("  -> Wukong single-core = 70% of measured roofline");
        let out = l.mismatch("  ! full-buffer mismatch vs C at [0]");
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(l.unchecked_sections().len(), 1);
        assert_eq!(l.suspect_sections().len(), 1);
    }

    /// The end-of-run summary names every UNCHECKED section, sets the exit code like a mismatch
    /// does, and carries a `!` line the campaign analyser's numerical-warning scan recognises
    /// (`analyze-sota-measure.py` keys on a leading `!` plus the word UNCHECKED).
    #[test]
    fn an_unchecked_section_fails_the_gate_unless_report_only() {
        let mut l = Ledger::new();
        l.begin("=== y ===");
        l.standing("  -> Wukong single-core is 1.10x faster than idiomatic C");
        let (lines, code) = l.finish(false);
        assert_eq!(code, NUMERICS_EXIT);
        assert!(
            lines
                .iter()
                .any(|s| s.trim_start().starts_with('!') && s.contains("UNCHECKED")),
            "{lines:?}"
        );
        assert!(
            lines.iter().any(|s| s.contains("UNCHECKED === y ===")),
            "{lines:?}"
        );
        assert_eq!(l.finish(true).1, 0);
        // Both lists print when both happened.
        l.begin("=== z ===");
        l.mismatch("  ! z mismatch vs C");
        let (lines, code) = l.finish(false);
        assert_eq!(code, NUMERICS_EXIT);
        assert!(lines.iter().any(|s| s.contains("=== z ===")), "{lines:?}");
        assert!(
            lines.iter().any(|s| s.contains("UNCHECKED === y ===")),
            "{lines:?}"
        );
    }

    /// A summary geomean is a standing too, so the pool gate must be EXACTLY the standing gate.
    /// The defect: the pools consulted `is_suspect()` alone, so a ratio withheld as UNCHECKED on its
    /// own `->` line (the recognizer-on column missing, so nothing compared the wk(gen) or
    /// `@parallel` buffer with anything) was still pushed into the wk(gen) and transpose-regime
    /// geomeans and printed there as a clean number.
    #[test]
    fn a_ratio_may_be_pooled_exactly_when_its_standing_would_print() {
        let line = "  -> Wukong single-core is 1.20x faster than idiomatic C";
        // (checks before the standing, a mismatch before it)
        for (checks, mismatch) in [(0, false), (1, false), (3, false), (0, true), (2, true)] {
            let mut l = Ledger::new();
            l.begin("=== transpose 1000x1000 ===");
            for _ in 0..checks {
                l.check();
            }
            if mismatch {
                l.mismatch("  ! transpose output mismatch vs C");
            }
            let pool = l.may_pool();
            let printed = l.standing(line);
            let withheld = printed[0].contains("ratio withheld");
            assert_eq!(
                pool, !withheld,
                "checks={checks} mismatch={mismatch}: may_pool={pool} but the standing printed \
                 {printed:?}"
            );
        }
        // The old gate admitted the UNCHECKED state; the new one does not.
        let mut l = Ledger::new();
        l.begin("=== wk(gen) row ===");
        assert!(l.is_unchecked() && !l.is_suspect());
        assert!(
            !l.may_pool(),
            "an UNCHECKED section is not SUSPECT, yet it must not pool"
        );
        l.check();
        assert!(!l.is_unchecked() && l.may_pool());
        // A new heading reopens the UNCHECKED state.
        l.begin("=== next ===");
        assert!(!l.may_pool());
    }
}
