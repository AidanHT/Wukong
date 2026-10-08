//! Within-run dispersion, the cold first call, the sustained estimator, and the main suite's
//! `C(twin)` control.
//!
//! THE DEFECTS (P6.10, P6.3 - 2026-09-13 sweep).
//!
//! * **Only a minimum ever left `time_ns`.** It took 7 single calls or 15 rep-blocks, kept the
//!   fastest and discarded the rest, so no within-run dispersion ever reached the log, the analyser
//!   was left with two cross-arm observations as its only variance estimate, and a change that
//!   doubled jitter, or regressed sustained all-core throughput while leaving the turbo-window
//!   minimum intact, was invisible to every instrument in the campaign. The first call - the cold
//!   start: cold caches, cold predictors, lazily bound DLL thunks - was warmed away unmeasured.
//! * **The zero-difference control never ran where the campaign measures.** `general.rs` carries a
//!   `C(twin)` column (byte-identical C source, compiler and flags, a second DLL, true ratio exactly
//!   1.00) and classifies every cell against it. Not one main-suite family did, so every ratio the
//!   campaign states a target in was printed with no statement of the run's own resolution.
//!
//! THE FIX, in three parts that change no existing ratio by default:
//!
//! 1. `time_ns` keeps every sample it already took, as a [`TimingStats`] (cold first call, min,
//!    median, max, last, count), and hands it to the rotation through a thread-local. Which
//!    statistic becomes `ns_per_call` is [`estimator`]: `min` by default (unchanged), or
//!    `XBENCH_ESTIMATOR=median|last` - `last` is the sustained-throughput reading, the final block
//!    after the turbo window has had every earlier block to expire in.
//! 2. `abba_min_columns` records both passes of every column and prints one `. dispersion` line per
//!    rotation: each slot's pass-to-pass spread (the same column, the same binary, two positions -
//!    a twin control for that column), its median/min and max/min (the within-run jitter and the
//!    worst block, the tail proxy), and its cold/min (the first-call cost), plus the rotation's
//!    within-run resolution, the widest pass-to-pass spread.
//! 3. [`twin_control`] reads the rotation's pass-paired times for a family's `C` and `C(twin)`
//!    columns, prints the control, and records its floor ([`crate::floor_factor`], the same rule
//!    `general.rs` applies) in the numerics ledger, which then tags every single-threaded standing
//!    inside that floor `[BELOW C(twin) FLOOR ...]` (see `numerics.rs`).
//!
//! Nothing here can make a ratio look better: the published estimate is unchanged unless the
//! operator asks for another one, the dispersion line is printed beside the table rather than into
//! it, and the floor can only qualify a standing, never upgrade one.

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

/// Every sample one `time_ns` call took, per call (ns). Produced only by [`TimingStats::from_samples`],
/// which refuses an empty or non-finite sample set rather than inventing a statistic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TimingStats {
    /// The very first call of the measurement, before any warm-up.
    pub(crate) cold_ns: f64,
    /// Fastest sample - the harness's historical (and default) estimate.
    pub(crate) min_ns: f64,
    pub(crate) median_ns: f64,
    /// Slowest sample: the worst block, the only tail this harness can see.
    pub(crate) max_ns: f64,
    /// The final sample: under sustained load, the steady-state figure rather than the turbo window.
    pub(crate) last_ns: f64,
    pub(crate) samples: usize,
}

impl TimingStats {
    /// `samples` are per-call times in the order they were taken; `cold_ns` is the first call.
    pub(crate) fn from_samples(cold_ns: f64, samples: &[f64]) -> Option<TimingStats> {
        let finite = |v: f64| v.is_finite() && v >= 0.0;
        if samples.is_empty() || !finite(cold_ns) || !samples.iter().all(|&s| finite(s)) {
            return None;
        }
        Some(TimingStats {
            cold_ns,
            min_ns: samples.iter().copied().fold(f64::INFINITY, f64::min),
            median_ns: crate::median(samples),
            max_ns: samples.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            last_ns: samples[samples.len() - 1],
            samples: samples.len(),
        })
    }
}

/// Which statistic of a [`TimingStats`] becomes a column's `ns_per_call`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Estimator {
    /// Best of N: the least-interfered sample. The default, and every historical number.
    Min,
    /// The typical sample.
    Median,
    /// The final sample: sustained throughput, after the turbo window.
    Last,
}

impl Estimator {
    /// Anything unrecognised is the default, so a typo can never silently change what is published;
    /// the banner prints the estimator in force either way.
    pub(crate) fn parse(v: Option<&str>) -> Estimator {
        match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("median") => Estimator::Median,
            Some("last") | Some("sustained") => Estimator::Last,
            _ => Estimator::Min,
        }
    }

    pub(crate) fn pick(self, s: &TimingStats) -> f64 {
        match self {
            Estimator::Min => s.min_ns,
            Estimator::Median => s.median_ns,
            Estimator::Last => s.last_ns,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Estimator::Min => "min (best of N, the default)",
            Estimator::Median => "median (XBENCH_ESTIMATOR=median)",
            Estimator::Last => "last block, sustained (XBENCH_ESTIMATOR=last)",
        }
    }
}

/// The estimator in force for this process (`XBENCH_ESTIMATOR`, read once).
pub(crate) fn estimator() -> Estimator {
    static E: OnceLock<Estimator> = OnceLock::new();
    *E.get_or_init(|| Estimator::parse(std::env::var("XBENCH_ESTIMATOR").ok().as_deref()))
}

thread_local! {
    /// The statistics of the most recent `time_ns` call on this thread.
    static LAST_TIMING: Cell<Option<TimingStats>> = const { Cell::new(None) };
    /// Every column's per-pass estimate from the most recent rotation on this thread.
    static LAST_ROTATION: RefCell<Vec<[Option<f64>; 2]>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn record_timing(s: Option<TimingStats>) {
    LAST_TIMING.with(|c| c.set(s));
}

pub(crate) fn take_timing() -> Option<TimingStats> {
    LAST_TIMING.with(|c| c.take())
}

/// One column of one rotation: the estimate each pass produced (0 = forward, 1 = reverse) and the
/// statistics behind them.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ColumnVisits {
    pub(crate) by_pass: [Option<f64>; 2],
    pub(crate) stats: Vec<TimingStats>,
}

/// What one column's visits say about the instrument, every field a factor relative to the
/// column's own fastest sample.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct SlotSummary {
    /// `max/min` of the two passes' estimates - the same binary in two positions.
    pub(crate) pass_spread: Option<f64>,
    /// The worse visit's median over the fastest sample: within-run jitter.
    pub(crate) median_over_min: Option<f64>,
    /// The slowest sample over the fastest: the worst block, the tail proxy.
    pub(crate) max_over_min: Option<f64>,
    /// The column's first-ever call over its fastest sample: the cold start.
    pub(crate) cold_over_min: Option<f64>,
    /// How many samples, over both visits, the statistics above were taken from.
    pub(crate) samples: usize,
}

pub(crate) fn summarize(v: &ColumnVisits) -> SlotSummary {
    let passes: Vec<f64> = v
        .by_pass
        .iter()
        .flatten()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    let pass_spread = (passes.len() == 2).then(|| {
        let (a, b) = (passes[0], passes[1]);
        a.max(b) / a.min(b)
    });
    let min = v
        .stats
        .iter()
        .map(|s| s.min_ns)
        .fold(f64::INFINITY, f64::min);
    if v.stats.is_empty() || !(min.is_finite() && min > 0.0) {
        return SlotSummary {
            pass_spread,
            ..SlotSummary::default()
        };
    }
    let worst =
        |f: fn(&TimingStats) -> f64| v.stats.iter().map(f).fold(f64::NEG_INFINITY, f64::max) / min;
    SlotSummary {
        pass_spread,
        median_over_min: Some(worst(|s: &TimingStats| s.median_ns)),
        max_over_min: Some(worst(|s: &TimingStats| s.max_ns)),
        cold_over_min: Some(v.stats[0].cold_ns / min),
        samples: v.stats.iter().map(|s| s.samples).sum(),
    }
}

/// The rotation's within-run resolution: the widest pass-to-pass spread any column showed. It mixes
/// single-threaded and all-core columns, so it is a conservative statement about the rotation,
/// never a floor for one particular ratio (that is what [`twin_control`] is for).
pub(crate) fn resolution(slots: &[SlotSummary]) -> Option<f64> {
    slots.iter().filter_map(|s| s.pass_spread).reduce(f64::max)
}

/// The `. dispersion` line, or `None` when no column produced a sample. Deliberately free of every
/// peer label and of `n/a`, so the extractor can neither mistake it for a table header nor log it
/// as a missing column; slots are numbered in the rotation's column order.
pub(crate) fn dispersion_line(slots: &[SlotSummary], est: Estimator) -> Option<String> {
    if slots.iter().all(|s| *s == SlotSummary::default()) {
        return None;
    }
    let f = |v: Option<f64>, prec: usize| v.map_or("-".to_string(), |x| format!("{x:.prec$}"));
    let cells: Vec<String> = slots
        .iter()
        .enumerate()
        .map(|(i, s)| {
            format!(
                "{i}: {} {} {} {} n{}",
                f(s.pass_spread, 3),
                f(s.median_over_min, 3),
                f(s.max_over_min, 3),
                f(s.cold_over_min, 2),
                s.samples
            )
        })
        .collect();
    let res = resolution(slots).map_or("unmeasured".to_string(), |r| format!("{r:.3}x"));
    Some(format!(
        "  . dispersion [slot: pass-spread median/min max/min cold/min samples]: {} ; within-run resolution {res} ; estimator {}",
        cells.join(" | "),
        est.label()
    ))
}

/// Close a rotation: print its dispersion line and keep its per-pass estimates for
/// [`twin_control`].
pub(crate) fn finish_rotation(visits: &[ColumnVisits]) {
    let slots: Vec<SlotSummary> = visits.iter().map(summarize).collect();
    if let Some(line) = dispersion_line(&slots, estimator()) {
        println!("{line}");
    }
    LAST_ROTATION.with(|r| *r.borrow_mut() = visits.iter().map(|v| v.by_pass).collect());
}

/// The C(twin) control from two columns' pass-paired estimates: `(per-pass C/C(twin) ratios,
/// floor)`. Pass-paired because the two columns are adjacent in each pass and swap order between
/// passes, so the per-pass ratio is exactly the quantity a slot can bias. `None` unless both
/// columns produced both passes.
pub(crate) fn twin_floor(c: [Option<f64>; 2], twin: [Option<f64>; 2]) -> Option<([f64; 2], f64)> {
    let (c0, c1, t0, t1) = (c[0]?, c[1]?, twin[0]?, twin[1]?);
    let stat = crate::RatioStat::over_rounds(&[c0, c1], &[t0, t1])?;
    Some(([c0 / t0, c1 / t1], crate::floor_factor(stat)))
}

/// The control line for the most recent rotation's `C` (slot `c_slot`) and `C(twin)` (slot
/// `twin_slot`) columns. Records the floor in the numerics ledger, which tags every single-threaded
/// standing inside it for the rest of the section. Call it after the rotation and before the
/// section's standings. Returns the floor, for a table that prints its ratios in its own cells.
pub(crate) fn twin_control(c_slot: usize, twin_slot: usize) -> Option<f64> {
    let passes = LAST_ROTATION.with(|r| {
        let r = r.borrow();
        Some((*r.get(c_slot)?, *r.get(twin_slot)?))
    });
    match passes.and_then(|(c, t)| twin_floor(c, t)) {
        Some(([r0, r1], floor)) => {
            println!(
                "  . control C(twin) [identical C source, compiler and flags; second DLL; true \
                 ratio 1.000]: C/C(twin) per pass {r0:.3}, {r1:.3} -> single-thread floor {floor:.3}x"
            );
            crate::numerics::record_twin_floor(floor);
            Some(floor)
        }
        None => {
            println!(
                "  . control C(twin): no pass-paired sample this run, so this section has no \
                 single-thread floor and its standings are unqualified"
            );
            None
        }
    }
}

/// The ALL-CORE twin control: the same all-core peer (`C(omp)` or `MKL(all)`) timed a second time
/// at another slot of the same rotation. It is the identical-arm control Finding 01 named as its
/// decisive test - `MKL(late)/MKL(early)` - now inside every section that carries it: an all-core
/// row's ratio is read against what the IDENTICAL all-core column did between two positions, which
/// is exactly the slot/turbo/placement noise the single-threaded `C(twin)` cannot see. Records the
/// floor in the numerics ledger, which qualifies the section's all-core standings with it. Silent
/// when neither column produced a sample (a row with no all-core peer at all, which the elementwise
/// table has many of), so it never prints a control that was never in the rotation.
pub(crate) fn all_core_twin_control(label: &str, slot: usize, twin_slot: usize) -> Option<f64> {
    let passes = LAST_ROTATION.with(|r| {
        let r = r.borrow();
        Some((*r.get(slot)?, *r.get(twin_slot)?))
    });
    let any = passes.is_some_and(|(a, b)| a.iter().chain(b.iter()).any(Option::is_some));
    if !any {
        return None;
    }
    match passes.and_then(|(a, b)| twin_floor(a, b)) {
        Some(([r0, r1], floor)) => {
            println!(
                "  . control twin of {label} [the identical all-core column at another slot; true \
                 ratio 1.000]: per pass {r0:.3}, {r1:.3} -> all-core floor {floor:.3}x"
            );
            crate::numerics::record_all_core_floor(floor);
            Some(floor)
        }
        None => {
            println!(
                "  . control twin of {label}: no pass-paired sample this run, so this section has no \
                 all-core floor and its all-core standings are unqualified"
            );
            None
        }
    }
}

/// A ratio cell for the tables that print their standings in their own cells (the norm tables):
/// `BELOW FLOOR` when the factor is no larger than the section's `C(twin)` floor (the cell
/// vocabulary `general.rs` already uses), otherwise `N.NNx faster|slower`. `floor` must be `None`
/// for an all-core row - the control is single-threaded.
pub(crate) fn ratio_cell(peer_ns: f64, wukong_ns: f64, floor: Option<f64>) -> String {
    let r = peer_ns / wukong_ns;
    let v = if r >= 1.0 { r } else { 1.0 / r };
    if floor.is_some_and(|f| v <= f) {
        return "BELOW FLOOR".to_string();
    }
    format!("{v:.2}x {}", if r >= 1.0 { "faster" } else { "slower" })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(cold: f64, xs: &[f64]) -> TimingStats {
        TimingStats::from_samples(cold, xs).unwrap()
    }

    #[test]
    fn stats_keep_every_order_statistic_the_minimum_used_to_discard() {
        let s = stats(900.0, &[110.0, 100.0, 130.0, 120.0]);
        assert_eq!(s.cold_ns, 900.0);
        assert_eq!(s.min_ns, 100.0);
        assert_eq!(s.max_ns, 130.0);
        assert_eq!(s.median_ns, 115.0);
        assert_eq!(s.last_ns, 120.0);
        assert_eq!(s.samples, 4);
    }

    #[test]
    fn stats_refuse_what_they_cannot_summarize() {
        assert_eq!(TimingStats::from_samples(1.0, &[]), None);
        assert_eq!(TimingStats::from_samples(f64::NAN, &[1.0]), None);
        assert_eq!(TimingStats::from_samples(1.0, &[1.0, f64::INFINITY]), None);
    }

    #[test]
    fn the_estimator_defaults_to_the_historical_minimum() {
        assert_eq!(Estimator::parse(None), Estimator::Min);
        assert_eq!(Estimator::parse(Some("typo")), Estimator::Min);
        assert_eq!(Estimator::parse(Some(" Median ")), Estimator::Median);
        assert_eq!(Estimator::parse(Some("sustained")), Estimator::Last);
        let s = stats(5.0, &[3.0, 1.0, 2.0]);
        assert_eq!(Estimator::Min.pick(&s), 1.0);
        assert_eq!(Estimator::Median.pick(&s), 2.0);
        assert_eq!(Estimator::Last.pick(&s), 2.0);
    }

    #[test]
    fn a_slot_reports_spread_jitter_tail_and_cold_start_against_its_own_minimum() {
        let v = ColumnVisits {
            by_pass: [Some(100.0), Some(104.0)],
            stats: vec![
                stats(400.0, &[100.0, 102.0, 120.0]),
                stats(150.0, &[104.0, 110.0]),
            ],
        };
        let s = summarize(&v);
        assert!((s.pass_spread.unwrap() - 1.04).abs() < 1e-12);
        // worse visit median: max(102, 107) / 100
        assert!((s.median_over_min.unwrap() - 1.07).abs() < 1e-12);
        assert!((s.max_over_min.unwrap() - 1.20).abs() < 1e-12);
        // the FIRST visit's cold call is the column's cold start
        assert!((s.cold_over_min.unwrap() - 4.0).abs() < 1e-12);
        assert_eq!(s.samples, 5);
    }

    #[test]
    fn a_single_pass_has_no_spread_and_a_missing_column_has_nothing() {
        let one = ColumnVisits {
            by_pass: [Some(100.0), None],
            stats: vec![stats(100.0, &[100.0])],
        };
        assert_eq!(summarize(&one).pass_spread, None);
        assert_eq!(summarize(&ColumnVisits::default()), SlotSummary::default());
    }

    #[test]
    fn resolution_is_the_widest_pass_spread_and_the_line_names_no_peer() {
        let slots = [
            SlotSummary {
                pass_spread: Some(1.01),
                ..SlotSummary::default()
            },
            SlotSummary::default(),
            SlotSummary {
                pass_spread: Some(1.05),
                median_over_min: Some(1.002),
                max_over_min: Some(1.3),
                cold_over_min: Some(2.5),
                samples: 30,
            },
        ];
        assert_eq!(resolution(&slots), Some(1.05));
        let line = dispersion_line(&slots, Estimator::Min).unwrap();
        assert!(line.starts_with("  . dispersion"), "{line}");
        assert!(line.contains("within-run resolution 1.050x"), "{line}");
        assert!(line.contains("1: - - - - n0"), "{line}");
        assert!(line.contains("2: 1.050 1.002 1.300 2.50 n30"), "{line}");
        // The extractor treats `n/a` as a missing-column note and peer names as a table header.
        for forbidden in ["n/a", "Wuk", "Rust", "C (gcc)", "C++", "MKL"] {
            assert!(!line.contains(forbidden), "{forbidden} in {line}");
        }
        assert_eq!(
            dispersion_line(&[SlotSummary::default()], Estimator::Min),
            None
        );
    }

    #[test]
    fn a_ratio_cell_inside_the_floor_says_so_and_outside_it_keeps_its_direction() {
        assert_eq!(ratio_cell(131.0, 100.0, Some(1.05)), "1.31x faster");
        assert_eq!(ratio_cell(100.0, 131.0, Some(1.05)), "1.31x slower");
        assert_eq!(ratio_cell(102.0, 100.0, Some(1.05)), "BELOW FLOOR");
        assert_eq!(ratio_cell(100.0, 103.0, Some(1.05)), "BELOW FLOOR");
        assert_eq!(ratio_cell(102.0, 100.0, None), "1.02x faster");
    }

    #[test]
    fn the_twin_floor_is_the_symmetric_departure_of_the_pass_paired_control() {
        // C/C(twin) reads 1.02 in one pass and 0.97 in the other: the floor is max(1.02, 1/0.97).
        let (ratios, floor) =
            twin_floor([Some(102.0), Some(97.0)], [Some(100.0), Some(100.0)]).unwrap();
        assert!((ratios[0] - 1.02).abs() < 1e-12 && (ratios[1] - 0.97).abs() < 1e-12);
        assert!((floor - 1.0 / 0.97).abs() < 1e-12, "{floor}");
        // A perfectly agreeing control still has a floor of exactly 1.00, never below.
        let (_, one) = twin_floor([Some(5.0), Some(5.0)], [Some(5.0), Some(5.0)]).unwrap();
        assert_eq!(one, 1.0);
        // Half a control is no control.
        assert_eq!(twin_floor([Some(1.0), None], [Some(1.0), Some(1.0)]), None);
    }
}
