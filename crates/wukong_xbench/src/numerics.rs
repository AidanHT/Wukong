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
    /// Standings printed in the open scope while it was still clean.
    clean_standings: usize,
    /// Every heading that went SUSPECT, in order, without duplicates.
    suspect_sections: Vec<String>,
    /// The open scope's `C(twin)` single-thread floor (see `dispersion::twin_control`), if one ran.
    twin_floor: Option<f64>,
}

impl Ledger {
    pub(crate) const fn new() -> Self {
        Ledger {
            section: String::new(),
            suspect: false,
            clean_standings: 0,
            suspect_sections: Vec::new(),
            twin_floor: None,
        }
    }

    /// Open a new scope. Returns the line to print (the heading itself).
    pub(crate) fn begin(&mut self, heading: &str) -> Vec<String> {
        self.section = heading.trim().to_string();
        self.suspect = false;
        self.clean_standings = 0;
        self.twin_floor = None;
        vec![heading.to_string()]
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
        if !self.suspect {
            self.suspect = true;
            let name = if self.section.is_empty() {
                "(before the first section heading)".to_string()
            } else {
                self.section.clone()
            };
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
        self.clean_standings += 1;
        let mut out = line.to_string();
        // A single-threaded standing no larger than the section's own zero-difference control is
        // not a measured difference. It is still printed - the reader is owed the number - but it
        // carries the qualification on its own line, where neither a skimming human nor the
        // extractor can separate the two. All-core standings are never tagged: the control is a
        // single-threaded column and says nothing about where the scheduler put 16 workers.
        if let (Some(floor), Some(ratio)) = (self.twin_floor, standing_ratio(line)) {
            if ratio <= floor && is_single_threaded_standing(line) {
                out.push_str(&format!(
                    "  [BELOW C(twin) FLOOR {floor:.3}x: not established]"
                ));
            }
        }
        vec![out]
    }

    /// Whether the open scope has reported a numerical mismatch. Summaries consult this so a
    /// SUSPECT row is never pooled into a geomean whose own line would carry no warning.
    pub(crate) fn is_suspect(&self) -> bool {
        self.suspect
    }

    #[cfg(test)]
    pub(crate) fn suspect_sections(&self) -> &[String] {
        &self.suspect_sections
    }

    /// The end-of-run summary lines and the exit code it implies. `report_only` is
    /// `XBENCH_NUMERICS=report`.
    pub(crate) fn finish(&self, report_only: bool) -> (Vec<String>, i32) {
        let mut out = vec!["=== numerics gate ===".to_string()];
        if self.suspect_sections.is_empty() {
            out.push(
                "  numerics gate: every cross-language check that ran in this process agreed; no \
                 standing was withheld. (A check that never ran certifies nothing.)"
                    .to_string(),
            );
            return (out, 0);
        }
        out.push(format!(
            "  ! NUMERICS GATE FAILED: {} section(s) reported a numerical mismatch, and every \
             standing in them was withheld:",
            self.suspect_sections.len()
        ));
        for s in &self.suspect_sections {
            out.push(format!("  !   {s}"));
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
pub(crate) fn standing_ratio(line: &str) -> Option<f64> {
    ["x faster", "x slower"].iter().find_map(|word| {
        let head = &line[..line.find(word)?];
        let start = head
            .char_indices()
            .rev()
            .take_while(|(_, c)| c.is_ascii_digit() || *c == '.')
            .last()
            .map(|(i, _)| i)?;
        head[start..].parse::<f64>().ok().filter(|r| r.is_finite())
    })
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

/// Record the open section's `C(twin)` floor (see [`Ledger::record_twin_floor`]).
pub(crate) fn record_twin_floor(floor: f64) {
    with_ledger(|l| l.record_twin_floor(floor));
}

/// Whether the open section has reported a numerical mismatch (see [`Ledger::is_suspect`]).
pub(crate) fn section_suspect() -> bool {
    with_ledger(|l| l.is_suspect())
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

/// Print a `->` standing, or withhold it when the open section is SUSPECT.
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
        let line = "  -> Wukong single-core is 1.31x faster than idiomatic C";
        assert_eq!(l.standing(line), vec![line.to_string()]);
        assert!(l.suspect_sections().is_empty());
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
    }

    #[test]
    fn a_single_threaded_standing_inside_the_twin_floor_is_tagged_not_hidden() {
        let mut l = Ledger::new();
        l.begin("=== colsum 1024x1024 ===");
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
    fn the_twin_floor_is_scoped_and_merges_conservatively() {
        let mut l = Ledger::new();
        l.begin("=== a ===");
        l.record_twin_floor(1.02);
        l.record_twin_floor(1.05);
        l.record_twin_floor(1.01);
        l.record_twin_floor(f64::NAN);
        let s = l.standing("  -> Wukong single-core is 1.04x faster than C");
        assert!(s[0].contains("FLOOR 1.050x"), "{s:?}");
        l.begin("=== b ===");
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
        let line = "  -> Wukong single-core is 1.10x faster than idiomatic C";
        assert_eq!(l.standing(line), vec![line.to_string()]);
        assert_eq!(l.suspect_sections(), &["=== a ===".to_string()]);
    }

    #[test]
    fn a_late_mismatch_flags_the_standings_already_printed() {
        let mut l = Ledger::new();
        l.begin("=== matmul 256x256 ===");
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
}
