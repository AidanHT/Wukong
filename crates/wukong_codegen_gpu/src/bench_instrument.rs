//! `bench_instrument` — the GPU measurement instrument: a **twin control**, a **publish gate**, and
//! a **harness-emitted provenance header**. Required by `GPU_RETARGET_PLAN.md` §6.1–§6.2, which did
//! not exist in code before this module.
//!
//! # Why a control column is not optional here
//!
//! This project has retracted four published claims and **every one traced to an instrument that
//! could not resolve what it claimed**. Two precedents govern the design:
//!
//! - `wukong_xbench` gained a `C(twin)` column — the identical C source, through the identical
//!   compiler, at identical flags, in a second DLL. Its ratio against `C` has expected value exactly
//!   1.000 and carries no language content. The first round that carried one moved **1.05x–1.69x**
//!   across 15 sections and in one case read a **byte-identical binary as 1.31x faster than
//!   itself**. That single control retracted the general-suite headline.
//! - The GPU perf-identity leg (`docs/gpu/derive/p2-expected-4050-identity.md`) ran an `A / C / B`
//!   protocol where `C` is the *same binary as* `A`. Its measured `C/A` spread ranged from ±0.1% to
//!   ±13.0% *across benches in one round*, and on battery a byte-identical arm read **1.52x**.
//!
//! So: **the noise floor is a property of the run, never a constant.** It is measured, every round,
//! by arms that are known to be identical, and nothing publishes below it.
//!
//! # The contract, in five rules
//!
//! 1. **Three arms.** `A` = [`Arm::Baseline`], `C` = [`Arm::Control`] (the twin — *known* identical
//!    to `A`, carries no content), `B` = [`Arm::Contender`]. The arms alternate inside one round and
//!    the leading arm **rotates** per round ([`rotation`]), so the first-launch clock ramp is not
//!    permanently taxed to one arm. At least one **discarded warm-up round** precedes recording.
//! 2. **Ratios over MEDIANS across rounds.** Not minima: a minimum is the wrong summary for a
//!    GFLOP/s field (it picks the *worst* sample) and is meaningless for a printed ratio field. What
//!    a bench does *within* one round is its own business — a `best_of(5)` inside a round is fine;
//!    this module summarizes the per-round readings.
//! 3. **`floor = max |C/A - 1|` over the round's live fields.** A field is live only if it actually
//!    varies and both arms produced a usable positive median ([`Cell::Constant`] and
//!    [`Cell::Degenerate`] are excluded — a constant field would drive the floor to zero and make
//!    the gate infinitely sensitive).
//! 4. **A bench whose floor exceeds the pre-registered bar publishes NO verdict** — not even a tie.
//!    "The floor was ±13% and B was within it" is not a tie, it is an unresolved measurement. This
//!    is P2 of the pre-registered 4050 analysis, encoded.
//! 5. **A round whose device does not match the spec table publishes NOTHING.** Not a warning: the
//!    object that mints numbers ([`Round`]) cannot be constructed at all ([`Round::open`] returns
//!    `Err`). Marketplaces have sold MIG slices and vGPU profiles as whole A100s.
//!
//! # The GPU twin, and the cache subtlety that would fake it
//!
//! §6.2's twin is *the byte-identical PTX loaded as two separate module handles*, timed as two
//! columns. Two things would silently destroy that symmetry, and both are handled:
//!
//! - **The in-process module map keys on `&'static str` alone** (`Gpu::function`'s documented
//!   landmine). Hand both arms the same key and they are literally one `CUmodule` and one
//!   `CUfunction`: every row ties perfectly having compared nothing. [`PtxTwin::new`] **refuses**
//!   equal keys — the analogue of `tools/perf/gpu_ab.ps1`'s sha256 check on its two arm binaries.
//! - **The persistent cubin cache keys on the PTX text** (`cubin::cache_path`). Byte-identical PTX
//!   means arm `A` pays a cold `cuLink` compile and writes the cubin, and arm `C` then loads that
//!   very file warm. A cold-vs-warm pair is **not** a twin. A cold device-keyed cubin cache once
//!   read a 27x false regression in this repo's own A/B history
//!   (`p2-expected-4050-identity.md` §5.1). Two answers, both provided:
//!   * [`PtxTwin::prime`] loads **both** handles and runs warm-up launches on **both** *before* any
//!     timing, so the asymmetric compile happens entirely outside the timed region and the two
//!     resident modules hold identical SASS. This is the right protocol for steady-state timing and
//!     is what [`PtxTwin::time_arm`] measures.
//!   * If the *load* itself is the thing being timed, **there is no in-process twin at all** —
//!     and that is a measured result, not a caution. This module's first draft claimed that driving
//!     the driver's PTX JIT directly (bypassing both Wukong caches) was symmetric by construction;
//!     its own device gate disproved that on the 4050, arm `C` reading a reproducible **-20% to
//!     -32%** on byte-identical work after a discarded warm-up — three consecutive runs on each of
//!     two days and two drivers — because the *driver* keeps a JIT cache of its own.
//!     [`PtxTwin::time_module_load`] therefore refuses a second call in one process and documents
//!     the one-measurement-per-process protocol; the `#[ignore]`d
//!     `the_in_process_load_twin_is_asymmetric_and_this_is_the_measurement` gate re-runs the
//!     experiment on demand so the refusal is checkable rather than merely asserted.
//!
//! # The peer twin
//!
//! §6.2 also asks for "a cuBLAS-called-twice control on peer rounds". That is the same three-arm
//! shape with no new machinery: [`run_twin`] calls its `reference` closure for **both** `A` and `C`,
//! so passing the cuBLAS peer as `reference` and our kernel as `contender` gives a peer ratio
//! (`B/A`) alongside the peer's own noise floor (`C/A`) in the same round.
//!
//! # What is testable without a device
//!
//! Everything except the launches: the statistics, the rotation/warm-up driver (generic over a
//! measurement closure, so a synthetic closure exercises it), the spec table, the device check, the
//! `nvidia-smi` parser, the provenance formatting, the publish gate and the PTX text of the built-in
//! probe. This module is therefore declared **un-gated** in `lib.rs` — like `ptx_target` and
//! `paged_attention` — so its gates run in a plain, toolchain-free `cargo test`, and only the launch
//! layer is `#[cfg(feature = "gpu")]`.
//!
//! # Adopting it from a bench (the wiring a later, separate commit does in `gpu.rs`)
//!
//! ```ignore
//! use crate::bench_instrument as bi;
//!
//! let mut round = match bi::open_round("gemm_throughput", g) {
//!     Ok(r) => r,
//!     // A MIG slice / an unknown part / an SM count off spec lands here and publishes nothing.
//!     Err(why) => { eprintln!("[skip:provenance] {why}"); return; }
//! };
//! eprintln!("{}", round.provenance().header());
//!
//! let plan = bi::TwinPlan::default();                    // 5 rounds, 1 discarded warm-up, ±5% bar
//! let s = bi::run_twin("gemm_throughput", &plan,
//!     || vec![bi::read("gemm_ms", time_gemm(g, m, n, k) * 1e3)],   // A and C: the same work twice
//!     || vec![bi::read("gemm_ms", time_gemm_new(g, m, n, k) * 1e3)]);
//!
//! round.close(bi::query_smi());
//! let v = bi::analyze(&s, plan.bar);
//! eprintln!("{}", v.table());
//! eprintln!("{}", round.header());
//! eprintln!("gemm_ms -> {}", round.cell(&v, "gemm_ms"));  // the number, or WHY there is no number
//! ```

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Pre-registered constants
// ---------------------------------------------------------------------------------------------

/// **The pre-registered control bar**: a bench whose measured control floor exceeds ±5% cannot
/// resolve a retarget-sized effect and publishes no verdict. Taken from P2 of
/// `docs/gpu/derive/p2-expected-4050-identity.md`, where it was written down *before* the round and
/// then correctly refused to certify five of eleven benches.
pub const DEFAULT_CONTROL_BAR: f64 = 0.05;

/// **The clock-drift bar**: if the SM clock read before the round and the one read after it differ
/// by more than this, the machine was not the same machine throughout and the round publishes
/// nothing. The laptop precedent is power state (three different machines); the datacenter
/// equivalents are thermal throttle, a power cap and a noisy multi-tenant host.
pub const DEFAULT_CLOCK_DRIFT_BAR: f64 = 0.05;

// ---------------------------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------------------------

/// One column of a round. `A`/`C` are *known identical* by construction; `B` is the thing being
/// measured. `C/A` is therefore pure instrument noise and `B/A` is the claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Arm {
    /// `A` — the reference: the peer, or the baseline kernel.
    Baseline,
    /// `C` — the twin of `A`: the *same* work through a second handle. Carries no content.
    Control,
    /// `B` — the contender.
    Contender,
}

impl Arm {
    /// Canonical order, and the order [`rotation`] rotates.
    pub const ALL: [Arm; 3] = [Arm::Baseline, Arm::Control, Arm::Contender];

    /// The one-letter tag used in logs: `A` / `C` / `B`.
    pub fn tag(self) -> &'static str {
        match self {
            Arm::Baseline => "A",
            Arm::Control => "C",
            Arm::Contender => "B",
        }
    }

    fn idx(self) -> usize {
        match self {
            Arm::Baseline => 0,
            Arm::Control => 1,
            Arm::Contender => 2,
        }
    }
}

/// The arm order for round `r`: [`Arm::ALL`] rotated left by `r % 3`.
///
/// GPU clocks ramp on the first launch of a process/round, so a fixed order pays that tax to
/// whichever arm goes first, every time — a systematic bias that no amount of averaging removes.
pub fn rotation(round: usize) -> [Arm; 3] {
    let s = round % 3;
    [Arm::ALL[s], Arm::ALL[(s + 1) % 3], Arm::ALL[(s + 2) % 3]]
}

// ---------------------------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------------------------

/// Median of `xs`, or `None` if empty **or if any sample is non-finite**.
///
/// The strictness is deliberate and is the whole reason this is not `xs.iter().sum()/n`: one `NaN`
/// from a failed timing must poison the field into [`Cell::Degenerate`] rather than quietly sort to
/// one end and hand back a plausible number.
pub fn median(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() || xs.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite by the check above"));
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    })
}

/// `x / base - 1`, or `None` when the ratio is not defined (zero or non-finite denominator).
pub fn rel_gap(x: f64, base: f64) -> Option<f64> {
    if !x.is_finite() || !base.is_finite() || base == 0.0 {
        None
    } else {
        Some(x / base - 1.0)
    }
}

// ---------------------------------------------------------------------------------------------
// Samples
// ---------------------------------------------------------------------------------------------

/// One named measurement produced by an arm in one round, e.g. `("gemm_ms", 12.4)`.
pub type Reading = (String, f64);

/// Build a [`Reading`]. Sugar so a bench body reads `vec![read("gemm_ms", t * 1e3)]`.
pub fn read(name: &str, value: f64) -> Reading {
    (name.to_string(), value)
}

/// The raw per-(field, arm) samples of one bench, one entry per recorded round.
#[derive(Debug, Clone, Default)]
pub struct BenchSamples {
    /// Bench name, as it will appear in the round log.
    pub bench: String,
    /// How many recorded rounds each `(field, arm)` is expected to carry. A field with fewer is
    /// [`Cell::Malformed`] and blocks the whole bench — an arm that failed to report is a broken
    /// instrument, not a missing data point.
    pub rounds: usize,
    fields: BTreeMap<String, [Vec<f64>; 3]>,
}

impl BenchSamples {
    pub fn new(bench: &str) -> Self {
        Self {
            bench: bench.to_string(),
            rounds: 0,
            fields: BTreeMap::new(),
        }
    }

    /// Record one sample.
    pub fn push(&mut self, field: &str, arm: Arm, value: f64) {
        self.fields
            .entry(field.to_string())
            .or_default()
            .get_mut(arm.idx())
            .expect("Arm::idx is 0..3")
            .push(value);
    }

    /// Field names, alphabetical (so a round log is byte-stable).
    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }

    /// Samples for one `(field, arm)`, in round order; empty if the arm never reported it.
    pub fn samples(&self, field: &str, arm: Arm) -> &[f64] {
        self.fields
            .get(field)
            .map(|a| a[arm.idx()].as_slice())
            .unwrap_or(&[])
    }

    /// Did any field get a contender sample at all? A round with no `B` is a pure control round: it
    /// measures the floor and publishes nothing (there is nothing to publish).
    pub fn has_contender(&self) -> bool {
        self.fields
            .values()
            .any(|a| !a[Arm::Contender.idx()].is_empty())
    }
}

// ---------------------------------------------------------------------------------------------
// The rotation / warm-up driver
// ---------------------------------------------------------------------------------------------

/// How a round is sampled. Both non-defaults are load-bearing, so [`TwinPlan::new`] validates them.
#[derive(Debug, Clone, PartialEq)]
pub struct TwinPlan {
    /// Recorded rounds. The median is taken across these.
    pub rounds: usize,
    /// Discarded warm-up rounds, run first, with every arm exercised.
    ///
    /// **Not hygiene.** With `warmups = 0` the first arm touched pays every cold JIT and cache miss
    /// *inside* the timed region: the 4050 smoke run read `saxpy 5.9 GB/s` against the other arm's
    /// `168 GB/s` — a 27x false regression on the canary bench — purely because the device-keyed
    /// cubin cache was cold for one arm.
    pub warmups: usize,
    /// The pre-registered control bar for this round (see [`DEFAULT_CONTROL_BAR`]). Pre-registered
    /// means: chosen before the numbers are seen, and not adjusted afterwards.
    pub bar: f64,
}

impl Default for TwinPlan {
    fn default() -> Self {
        Self {
            rounds: 5,
            warmups: 1,
            bar: DEFAULT_CONTROL_BAR,
        }
    }
}

impl TwinPlan {
    /// A validated plan. Rejects fewer than 3 recorded rounds (a median of one or two samples is not
    /// a median), zero warm-ups, and a non-positive or absurd bar.
    pub fn new(rounds: usize, warmups: usize, bar: f64) -> Result<Self, String> {
        if rounds < 3 {
            return Err(format!(
                "TwinPlan: {rounds} recorded rounds is not enough to take a median over; use >= 3"
            ));
        }
        if warmups == 0 {
            return Err(
                "TwinPlan: warmups must be >= 1 — a cold cache inside the timed region once read a \
                 27x false regression on this repo's own canary bench"
                    .to_string(),
            );
        }
        if !(bar > 0.0 && bar < 1.0) {
            return Err(format!(
                "TwinPlan: the control bar must be a fraction in (0,1); got {bar}"
            ));
        }
        Ok(Self {
            rounds,
            warmups,
            bar,
        })
    }
}

/// Drive `measure` over the warm-up and recorded rounds, rotating the arm order every round.
///
/// `measure(arm)` returns this arm's readings for one round. It is called for the warm-up rounds
/// too — their readings are dropped, which is the point.
pub fn run_rotated<F>(bench: &str, plan: &TwinPlan, mut measure: F) -> BenchSamples
where
    F: FnMut(Arm) -> Vec<Reading>,
{
    let mut s = BenchSamples::new(bench);
    s.rounds = plan.rounds;
    for r in 0..plan.warmups + plan.rounds {
        for arm in rotation(r) {
            let readings = measure(arm);
            if r >= plan.warmups {
                for (name, v) in readings {
                    s.push(&name, arm, v);
                }
            }
        }
    }
    s
}

/// The twin protocol: `reference` is called for **both** `A` and `C`, `contender` for `B`.
///
/// This one function covers both of §6.2's controls, because they are the same experiment:
/// - **peer twin** — `reference` calls cuBLAS, so `C/A` is cuBLAS-against-itself (the peer's own
///   noise floor) and `B/A` is our kernel against cuBLAS in the same round;
/// - **self twin** — `reference` runs our baseline kernel, so `C/A` is that bench's own floor.
///
/// For the *module-handle* twin §6.2 names literally — one PTX text, two `CUmodule`s — build the two
/// arms from a [`PtxTwin`] and call [`run_rotated`] directly, so that `A` and `C` use *different*
/// handles rather than the same closure twice.
///
/// **If an arm needs `&mut Gpu` inside the timed region, use [`run_rotated`], not this.** Two
/// `FnMut` closures cannot both hold a mutable borrow of one `Gpu` (E0499), and the fix is not to
/// clone or to re-open the device — it is to pass a *single* closure that matches on the arm, so the
/// borrow happens once. That is how [`machine_floor`] drives the built-in twin. `run_twin` fits the
/// common case, which is also the shape every existing bench in `gpu.rs` already has: resolve the
/// `CudaFunction` handles first, then time with a shared borrow.
pub fn run_twin<R, C>(
    bench: &str,
    plan: &TwinPlan,
    mut reference: R,
    mut contender: C,
) -> BenchSamples
where
    R: FnMut() -> Vec<Reading>,
    C: FnMut() -> Vec<Reading>,
{
    run_rotated(bench, plan, |arm| match arm {
        Arm::Baseline | Arm::Control => reference(),
        Arm::Contender => contender(),
    })
}

// ---------------------------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------------------------

/// What one field of one bench is allowed to say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// `|B/A - 1| <= floor` at a floor that cleared the bar: an honest, publishable tie.
    Tie,
    /// The effect cleared this round's own floor: publishable as a direction.
    Moved,
    /// The floor did not clear the pre-registered bar. Publishes nothing — reporting a wide tie as
    /// a tie is exactly the failure this module exists to prevent.
    Unresolved,
    /// A median was unusable (empty, non-finite, or a non-positive baseline).
    Degenerate,
    /// The field never varied between `A` and `C`, so it measures nothing. Excluded from the floor:
    /// a constant field would make the floor `0.0` and every noise blip would "clear" it.
    Constant,
    /// An arm reported a different number of rounds than the plan recorded.
    Malformed,
    /// No contender arm at all — a pure control round.
    NoContender,
}

impl Cell {
    /// May a number be published for this cell?
    pub fn publishable(self) -> bool {
        matches!(self, Cell::Tie | Cell::Moved)
    }

    pub fn tag(self) -> &'static str {
        match self {
            Cell::Tie => "tie",
            Cell::Moved => "moved",
            Cell::Unresolved => "unresolved",
            Cell::Degenerate => "degenerate",
            Cell::Constant => "constant",
            Cell::Malformed => "malformed",
            Cell::NoContender => "no-contender",
        }
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.tag())
    }
}

/// One field's medians, ratios and verdict.
#[derive(Debug, Clone)]
pub struct FieldVerdict {
    pub field: String,
    /// Median of arm `A` across the recorded rounds.
    pub a: Option<f64>,
    /// Median of arm `C` (the twin).
    pub c: Option<f64>,
    /// Median of arm `B`.
    pub b: Option<f64>,
    /// `C/A - 1` — this field's contribution to the floor. Expected value: `0`.
    pub control_gap: Option<f64>,
    /// `B/A - 1` — the effect.
    pub effect: Option<f64>,
    /// Did this field contribute to the floor?
    pub live: bool,
    pub cell: Cell,
}

/// A whole bench's verdict: the measured floor, every field, and whether the gate is open.
///
/// The numeric fields are public because the round log has to print them — including the ones that
/// publish nothing, which are findings in their own right. They are **not** the sanctioned path to a
/// number. The three outputs that carry their own verdict are [`BenchVerdict::table`] (tags every
/// row), [`Round::cell`] (the number, or `--` and the reason) and [`Round::publish`] (a
/// [`Published`], which has no other constructor). Reading `field(..).effect` and printing it is how
/// a gate becomes a convention again.
#[derive(Debug, Clone)]
pub struct BenchVerdict {
    pub bench: String,
    /// The pre-registered bar this round was judged against.
    pub bar: f64,
    /// Recorded rounds (the median's sample count).
    pub rounds: usize,
    /// `max |C/A - 1|` over the live fields — **this round's own noise floor**. `None` when no field
    /// was live, which is itself a refusal: a round with no working control publishes nothing.
    pub floor: Option<f64>,
    pub fields: Vec<FieldVerdict>,
    /// Arms `A` and `C` produced element-wise identical sample vectors on every live field. Two real
    /// timings are never bit-identical, so this means the arms are the same object — the GPU
    /// analogue of handing `gpu_ab.ps1` the same binary twice.
    pub aliased: bool,
    /// The bench-level reason the gate is shut, if it is.
    pub blocked: Option<Refusal>,
}

/// Compute the floor, the per-field verdicts and the bench-level gate state. Pure.
pub fn analyze(samples: &BenchSamples, bar: f64) -> BenchVerdict {
    let rounds = samples.rounds;
    let has_contender = samples.has_contender();
    let mut fields = Vec::new();
    /// Why a field is not live, carried from the first pass so the cell decision never has to
    /// re-derive it from the medians (which cannot distinguish "never varied" from "unusable").
    #[derive(Clone, Copy)]
    struct Flags {
        bad_len: bool,
        constant: bool,
    }
    let mut flags: Vec<Flags> = Vec::new();
    let mut malformed: Option<String> = None;

    for name in samples.field_names() {
        let av = samples.samples(name, Arm::Baseline);
        let cv = samples.samples(name, Arm::Control);
        let bv = samples.samples(name, Arm::Contender);

        let bad_len = rounds == 0
            || av.len() != rounds
            || cv.len() != rounds
            || (!bv.is_empty() && bv.len() != rounds);

        let (a, c, b) = (median(av), median(cv), median(bv));
        // "Constant" means the control pair never moved: it cannot measure noise, so it must not be
        // allowed to drive the floor to zero.
        let constant =
            !av.is_empty() && av.iter().chain(cv.iter()).all(|x| *x == av[0]) && !bad_len;
        let usable = matches!((a, c), (Some(a), Some(c)) if a > 0.0 && c > 0.0);
        let live = !bad_len && !constant && usable;
        let control_gap = match (a, c) {
            (Some(a), Some(c)) => rel_gap(c, a),
            _ => None,
        };
        let effect = match (a, b) {
            (Some(a), Some(b)) => rel_gap(b, a),
            _ => None,
        };

        if bad_len && malformed.is_none() {
            malformed = Some(name.to_string());
        }

        flags.push(Flags { bad_len, constant });
        fields.push(FieldVerdict {
            field: name.to_string(),
            a,
            c,
            b,
            control_gap,
            effect,
            live,
            cell: Cell::Degenerate, // provisional; decided below, once the floor is known
        });
    }

    let floor = fields
        .iter()
        .filter(|f| f.live)
        .filter_map(|f| f.control_gap.map(f64::abs))
        .fold(None::<f64>, |acc, g| Some(acc.map_or(g, |m: f64| m.max(g))));

    let live_names: Vec<&str> = fields
        .iter()
        .filter(|f| f.live)
        .map(|f| f.field.as_str())
        .collect();
    let aliased = !live_names.is_empty()
        && live_names
            .iter()
            .all(|n| samples.samples(n, Arm::Baseline) == samples.samples(n, Arm::Control));

    let resolved = matches!(floor, Some(fl) if fl <= bar);
    for (fv, why) in fields.iter_mut().zip(flags) {
        let bv_len = samples.samples(&fv.field, Arm::Contender).len();
        fv.cell = if why.bad_len {
            Cell::Malformed
        } else if why.constant {
            // Usable medians but no variation: a label, not a measurement.
            Cell::Constant
        } else if !fv.live {
            // Not constant and not malformed, so the medians themselves are unusable.
            Cell::Degenerate
        } else if bv_len == 0 || fv.effect.is_none() {
            Cell::NoContender
        } else if !resolved {
            Cell::Unresolved
        } else {
            let fl = floor.expect("resolved implies Some");
            let e = fv.effect.expect("checked above");
            if e.abs() <= fl {
                Cell::Tie
            } else {
                Cell::Moved
            }
        };
    }

    // Refusal precedence: instrument-is-broken first, then STRUCTURAL, then QUALITY.
    //
    // `aliased` / `malformed` / `NoControl` say the round itself is invalid, so they outrank
    // everything. After those, `!has_contender` must be tested BEFORE `!resolved`, and the order is
    // load-bearing rather than stylistic.
    //
    // A control-only round (`machine_floor`) has no contender by construction: the floor IS its
    // result, not an obstacle to one. "The floor is too wide to resolve the effect" is a category
    // error when there is no effect under test, and `FloorAboveBar`'s message goes on to advise
    // "re-run rather than reporting a wide tie as a tie" — actively wrong for a round that was never
    // going to publish a tie. Worse, it made the verdict self-contradictory: the per-field cell
    // already reported `Cell::NoContender` while the round-level refusal said `FloorAboveBar`.
    //
    // The reversed order was invisible to the synthetic unit tests because they use clean numbers —
    // `a_control_only_round_measures_a_floor_and_publishes_nothing` feeds 10.0 vs 10.05, a ~0.5%
    // floor, so `resolved` is true there and the two conditions never overlap. It surfaced only on a
    // real box noisy enough for both to hold at once (this 4050 read a +10.21% floor against the
    // +/-5% bar). `both_refusals_true_reports_the_structural_one` now pins the interaction with no
    // device at all.
    let blocked = if aliased {
        Some(Refusal::AliasedArms)
    } else if let Some(f) = malformed {
        Some(Refusal::MalformedField(f))
    } else if floor.is_none() {
        Some(Refusal::NoControl)
    } else if !has_contender {
        Some(Refusal::NoContenderArm)
    } else if !resolved {
        Some(Refusal::FloorAboveBar {
            floor: floor.unwrap_or(f64::NAN),
            bar,
        })
    } else {
        None
    };

    BenchVerdict {
        bench: samples.bench.clone(),
        bar,
        rounds,
        floor,
        fields,
        aliased,
        blocked,
    }
}

impl BenchVerdict {
    /// The field verdict by name.
    pub fn field(&self, name: &str) -> Option<&FieldVerdict> {
        self.fields.iter().find(|f| f.field == name)
    }

    /// A fixed-width ASCII table for the round log. Reports the floor and every field's `C/A` and
    /// `B/A`, including the ones that publish nothing — an unresolved cell is a finding, not a gap.
    pub fn table(&self) -> String {
        let mut s = String::new();
        let floor = match self.floor {
            Some(f) => format!("{:+.2}%", f * 100.0),
            None => "n/a".to_string(),
        };
        let _ = writeln!(
            s,
            "bench {}: floor {} (bar +/-{:.2}%), median of {} rounds{}",
            self.bench,
            floor,
            self.bar * 100.0,
            self.rounds,
            match &self.blocked {
                Some(r) => format!("  [GATE SHUT: {r}]"),
                None => String::new(),
            }
        );
        let _ = writeln!(
            s,
            "  {:<24} {:>14} {:>14} {:>14} {:>10} {:>10}  verdict",
            "field", "A", "C", "B", "C/A-1", "B/A-1"
        );
        for f in &self.fields {
            let num = |v: Option<f64>| match v {
                Some(v) => format!("{v:.6}"),
                None => "n/a".to_string(),
            };
            let pct = |v: Option<f64>| match v {
                Some(v) => format!("{:+.2}%", v * 100.0),
                None => "n/a".to_string(),
            };
            let _ = writeln!(
                s,
                "  {:<24} {:>14} {:>14} {:>14} {:>10} {:>10}  {}{}",
                f.field,
                num(f.a),
                num(f.c),
                num(f.b),
                pct(f.control_gap),
                pct(f.effect),
                f.cell,
                if f.live { "" } else { " (not in floor)" }
            );
        }
        s
    }
}

// ---------------------------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------------------------

/// Why nothing may be published. Every variant names the fix, because a refusal a reader cannot act
/// on becomes a refusal someone routes around.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// The device did not verify against the spec table (§6.1's hard rule).
    Device(String),
    /// [`Round::close`] was never called, so there is no after-the-round evidence.
    RoundNotClosed,
    /// A clock reading is missing at one or both ends of the round.
    NoClockEvidence(&'static str),
    /// The SM clock moved between the two readings: not the same machine throughout.
    ClockDrift { drift: f64, bar: f64 },
    /// Arms `A` and `C` are the same object; the control compared nothing.
    AliasedArms,
    /// An arm did not report every round for this field.
    MalformedField(String),
    /// No field could serve as a control.
    NoControl,
    /// The measured floor exceeded the pre-registered bar.
    FloorAboveBar { floor: f64, bar: f64 },
    /// A control-only round: there is no contender to publish.
    NoContenderArm,
    /// Asked to publish a field the bench never measured.
    UnknownField(String),
    /// The cell itself is not publishable.
    Cell { field: String, cell: Cell },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::Device(why) => write!(f, "device did not verify: {why}"),
            Refusal::RoundNotClosed => write!(
                f,
                "the round was never closed — call Round::close(query_smi()) so the after-the-round \
                 clocks exist before anything is published"
            ),
            Refusal::NoClockEvidence(which) => write!(
                f,
                "no {which} clock/temp/power reading — a published round must carry nvidia-smi \
                 readings at BOTH ends (plan 6.1)"
            ),
            Refusal::ClockDrift { drift, bar } => write!(
                f,
                "SM clock drifted {:+.2}% across the round (bar +/-{:.2}%) — this was not one \
                 machine throughout",
                drift * 100.0,
                bar * 100.0
            ),
            Refusal::AliasedArms => write!(
                f,
                "arms A and C produced identical sample vectors — they are the same handle/closure, \
                 so the control compared nothing and every row would tie"
            ),
            Refusal::MalformedField(name) => write!(
                f,
                "field '{name}': an arm did not report every recorded round — the instrument \
                 failed, so no field of this bench publishes"
            ),
            Refusal::NoControl => write!(
                f,
                "no live field: the round has no working control and therefore no measured floor"
            ),
            Refusal::FloorAboveBar { floor, bar } => write!(
                f,
                "control floor +/-{:.2}% exceeds the pre-registered bar +/-{:.2}% — this round \
                 cannot resolve the effect; re-run rather than reporting a wide tie as a tie",
                floor * 100.0,
                bar * 100.0
            ),
            Refusal::NoContenderArm => write!(
                f,
                "control-only round: the floor is measured, but there is no contender arm to publish"
            ),
            Refusal::UnknownField(name) => write!(f, "field '{name}' was never measured"),
            Refusal::Cell { field, cell } => {
                write!(f, "field '{field}' is {cell}, which publishes nothing")
            }
        }
    }
}

impl std::error::Error for Refusal {}

// ---------------------------------------------------------------------------------------------
// The device spec table (§6.1: "SM count verified against a spec table")
// ---------------------------------------------------------------------------------------------

/// One row of the spec table: what a part with this name **must** report.
///
/// `sm_count` is the published SM count, and `note` records the arithmetic it came from so a
/// reviewer can check a row without trusting it. A wrong row fails *closed* (it refuses a legitimate
/// device, loudly and fixably); a missing row also fails closed ([`DeviceCheck::Unknown`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceSpec {
    /// Whitespace-separated token run matched against the normalized device name.
    pub pattern: &'static str,
    pub sm_count: i32,
    pub cc: (i32, i32),
    pub note: &'static str,
}

/// Parts this instrument can verify. **Longest token match wins**, which is why `H100 PCIE` can sit
/// beside `H100` and `L40S` beside `L4`.
///
/// Adding a part is a code change on purpose: an env var that let an operator *declare* the expected
/// SM count would hand the check back to the human it exists to replace.
pub const DEVICE_SPECS: &[DeviceSpec] = &[
    // --- the dev laptop -------------------------------------------------------------------------
    DeviceSpec {
        pattern: "GEFORCE RTX 4050 LAPTOP",
        sm_count: 20,
        cc: (8, 9),
        note: "AD107 laptop, 2560 CUDA cores / 128 per SM",
    },
    // --- Ada (sm_89): the stepping stones -------------------------------------------------------
    DeviceSpec {
        pattern: "L4",
        sm_count: 58,
        cc: (8, 9),
        note: "AD104, 7424 cores / 128",
    },
    DeviceSpec {
        pattern: "L40S",
        sm_count: 142,
        cc: (8, 9),
        note: "AD102, 18176 cores / 128",
    },
    DeviceSpec {
        pattern: "L40",
        sm_count: 142,
        cc: (8, 9),
        note: "AD102, 18176 cores / 128",
    },
    DeviceSpec {
        pattern: "RTX 6000 ADA",
        sm_count: 142,
        cc: (8, 9),
        note: "AD102 workstation, 18176 cores / 128",
    },
    DeviceSpec {
        pattern: "GEFORCE RTX 4090",
        sm_count: 128,
        cc: (8, 9),
        note: "AD102, 16384 cores / 128",
    },
    // --- Ampere (sm_80 / sm_86) ------------------------------------------------------------------
    DeviceSpec {
        pattern: "A100",
        sm_count: 108,
        cc: (8, 0),
        note: "GA100, 6912 FP32 cores / 64 — identical across SXM4/PCIe and 40/80 GB",
    },
    DeviceSpec {
        pattern: "A40",
        sm_count: 84,
        cc: (8, 6),
        note: "GA102, 10752 cores / 128",
    },
    DeviceSpec {
        pattern: "RTX A6000",
        sm_count: 84,
        cc: (8, 6),
        note: "GA102, 10752 cores / 128",
    },
    DeviceSpec {
        pattern: "A10",
        sm_count: 72,
        cc: (8, 6),
        note: "GA102, 9216 cores / 128",
    },
    DeviceSpec {
        pattern: "GEFORCE RTX 3090",
        sm_count: 82,
        cc: (8, 6),
        note: "GA102, 10496 cores / 128",
    },
    // --- Hopper (sm_90): the competitive target --------------------------------------------------
    DeviceSpec {
        pattern: "H100 PCIE",
        sm_count: 114,
        cc: (9, 0),
        note: "GH100 PCIe is a CUT part — 14592 cores / 128, NOT the SXM's 132",
    },
    DeviceSpec {
        pattern: "H100",
        sm_count: 132,
        cc: (9, 0),
        note: "GH100 SXM5 / NVL, 16896 cores / 128",
    },
    DeviceSpec {
        pattern: "H200",
        sm_count: 132,
        cc: (9, 0),
        note: "same GH100 die as H100 SXM, 141 GB HBM3e",
    },
    // --- Blackwell -------------------------------------------------------------------------------
    DeviceSpec {
        pattern: "B200",
        sm_count: 148,
        cc: (10, 0),
        note: "datacenter Blackwell (tcgen05), per GPU_RETARGET_PLAN.md 3",
    },
    DeviceSpec {
        pattern: "RTX PRO 6000",
        sm_count: 188,
        cc: (12, 0),
        note: "consumer-class Blackwell (sm_120), 24064 cores / 128",
    },
    DeviceSpec {
        pattern: "GEFORCE RTX 5090",
        sm_count: 170,
        cc: (12, 0),
        note: "GB202, 21760 cores / 128 — same ISA as RTX PRO 6000",
    },
    // --- older parts, present only so an accidental fallback host is NAMED rather than unknown ----
    DeviceSpec {
        pattern: "T4",
        sm_count: 40,
        cc: (7, 5),
        note: "TU104, 2560 cores / 64",
    },
    DeviceSpec {
        pattern: "V100",
        sm_count: 80,
        cc: (7, 0),
        note: "GV100, 5120 cores / 64",
    },
];

/// Normalize a `cuDeviceGetName` string to uppercase alphanumeric tokens: every non-alphanumeric
/// byte becomes a separator, so `"NVIDIA A100-SXM4-40GB"` and `"NVIDIA A100 80GB PCIe"` both tokenize
/// cleanly and `"1g.5gb"` splits.
pub fn name_tokens(name: &str) -> Vec<String> {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn tokens_contain(hay: &[String], needle: &[&str]) -> bool {
    if needle.is_empty() || needle.len() > hay.len() {
        return false;
    }
    hay.windows(needle.len())
        .any(|w| w.iter().zip(needle).all(|(h, n)| h == n))
}

/// The spec row for `name`, longest token match first. Token-based, so `L4` does **not** match an
/// `L40S` and `H100` does not shadow `H100 PCIe`.
pub fn lookup_spec(name: &str) -> Option<&'static DeviceSpec> {
    let toks = name_tokens(name);
    let mut best: Option<(usize, &'static DeviceSpec)> = None;
    for spec in DEVICE_SPECS {
        let pat: Vec<&str> = spec.pattern.split_whitespace().collect();
        if tokens_contain(&toks, &pat) && best.is_none_or(|(len, _)| pat.len() > len) {
            best = Some((pat.len(), spec));
        }
    }
    best.map(|(_, s)| s)
}

/// A marker that this "GPU" is a slice or a virtualization profile rather than a whole device.
///
/// Two independent detectors guard this, because either alone is defeatable: the name marker here,
/// and the SM-count comparison in [`check_device`] (a MIG `1g.10gb` slice of an A100 reports 16 SMs,
/// not 108). The `<N>C` token is the vGPU profile suffix — `GRID A100D-40C` is a *virtual* A100 and
/// has been sold as a real one.
pub fn virtualization_marker(name: &str) -> Option<&'static str> {
    for t in name_tokens(name) {
        match t.as_str() {
            "MIG" => return Some("MIG (a partition of a GPU, not a GPU)"),
            "GRID" => return Some("GRID (a vGPU profile)"),
            "VGPU" => return Some("vGPU"),
            _ => {}
        }
        let (digits, tail) = t.split_at(t.len().saturating_sub(1));
        if tail == "C" && !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
            return Some("a <N>C vGPU profile suffix");
        }
    }
    None
}

/// The probed device identity this instrument needs. Mirrors the fields of `gpu::GpuTarget`, but as
/// plain data with no `cudarc` dependency, so every check over it is unit-testable with no device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFacts {
    pub name: String,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub sm_count: i32,
    pub smem_per_block_optin: usize,
    pub l2_bytes: usize,
    pub total_mem: usize,
    /// `cuDriverGetVersion`, e.g. `12090`.
    pub driver_version: i32,
}

/// The result of checking a probed device against [`DEVICE_SPECS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceCheck {
    Verified(&'static DeviceSpec),
    Virtualized(&'static str),
    SmMismatch {
        spec: &'static DeviceSpec,
        probed: i32,
    },
    CcMismatch {
        spec: &'static DeviceSpec,
        probed: (i32, i32),
    },
    Unknown,
}

impl DeviceCheck {
    pub fn verified(&self) -> bool {
        matches!(self, DeviceCheck::Verified(_))
    }
}

impl fmt::Display for DeviceCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceCheck::Verified(s) => write!(
                f,
                "VERIFIED against spec ({} SMs, cc {}.{}; {})",
                s.sm_count, s.cc.0, s.cc.1, s.note
            ),
            DeviceCheck::Virtualized(m) => write!(
                f,
                "REFUSED: the device name carries {m} — a published round needs a whole physical GPU"
            ),
            DeviceCheck::SmMismatch { spec, probed } => write!(
                f,
                "REFUSED: {} SMs probed, spec says {} ({}) — a partitioned, cut or virtualized part",
                probed, spec.sm_count, spec.note
            ),
            DeviceCheck::CcMismatch { spec, probed } => write!(
                f,
                "REFUSED: compute capability {}.{} probed, spec says {}.{}",
                probed.0, probed.1, spec.cc.0, spec.cc.1
            ),
            DeviceCheck::Unknown => write!(
                f,
                "REFUSED: not in DEVICE_SPECS, so its SM count cannot be verified — add a row to \
                 bench_instrument::DEVICE_SPECS (a code change, reviewable) rather than trusting the \
                 probe"
            ),
        }
    }
}

/// Verify a probed device against the spec table. **This is the §6.1 hard refusal**, and it is
/// fail-closed in all three directions: a virtualization marker, a count/capability that disagrees
/// with the spec, and a part the table does not know all refuse.
pub fn check_device(facts: &DeviceFacts) -> DeviceCheck {
    if let Some(marker) = virtualization_marker(&facts.name) {
        return DeviceCheck::Virtualized(marker);
    }
    let Some(spec) = lookup_spec(&facts.name) else {
        return DeviceCheck::Unknown;
    };
    if (facts.cc_major, facts.cc_minor) != spec.cc {
        return DeviceCheck::CcMismatch {
            spec,
            probed: (facts.cc_major, facts.cc_minor),
        };
    }
    if facts.sm_count != spec.sm_count {
        return DeviceCheck::SmMismatch {
            spec,
            probed: facts.sm_count,
        };
    }
    DeviceCheck::Verified(spec)
}

// ---------------------------------------------------------------------------------------------
// nvidia-smi evidence
// ---------------------------------------------------------------------------------------------

/// One `nvidia-smi` reading. Taken before AND after a round; the pair is the evidence that the
/// machine did not change under the measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct SmiSample {
    /// e.g. `"580.95.05"` — the kernel-mode driver, which `cuDriverGetVersion` does not give.
    pub driver: String,
    /// Required: the drift check is built on it.
    pub sm_clock_mhz: f64,
    pub mem_clock_mhz: Option<f64>,
    pub temp_c: Option<f64>,
    pub power_w: Option<f64>,
}

/// The exact `--query-gpu` field list [`parse_smi_csv`] expects, in order.
pub const SMI_QUERY: &str = "driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw";

/// Parse one `nvidia-smi --format=csv,noheader,nounits` line. Tolerates the unit suffixes the
/// non-`nounits` form emits and the `[N/A]` / `[Not Supported]` cells some hosts return for power.
pub fn parse_smi_csv(line: &str) -> Result<SmiSample, String> {
    let cols: Vec<&str> = line.split(',').map(str::trim).collect();
    if cols.len() < 5 {
        return Err(format!(
            "nvidia-smi line has {} columns, expected 5 ({SMI_QUERY}): {line:?}",
            cols.len()
        ));
    }
    let num = |s: &str| -> Option<f64> {
        let head = s.split_whitespace().next()?;
        head.parse::<f64>().ok()
    };
    let sm_clock_mhz = num(cols[1]).ok_or_else(|| {
        format!(
            "nvidia-smi gave no usable clocks.sm ({:?}) — without it a round has no drift evidence",
            cols[1]
        )
    })?;
    Ok(SmiSample {
        driver: cols[0].to_string(),
        sm_clock_mhz,
        mem_clock_mhz: num(cols[2]),
        temp_c: num(cols[3]),
        power_w: num(cols[4]),
    })
}

/// Read the current clocks from `nvidia-smi`, or `None` if the tool is absent or answers nothing
/// usable. Deliberately silent on failure: it is called at both ends of a round, and the *absence*
/// of a reading is enforced later, by [`Round::publish`], rather than by a panic here.
pub fn query_smi() -> Option<SmiSample> {
    let out = std::process::Command::new("nvidia-smi")
        .arg(format!("--query-gpu={SMI_QUERY}"))
        .arg("--format=csv,noheader,nounits")
        .arg("-i")
        .arg("0")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    parse_smi_csv(text.lines().find(|l| !l.trim().is_empty())?).ok()
}

/// Whether the round's clocks were pinned. **Declared metadata, not evidence** — the evidence is the
/// before/after [`SmiSample`] pair, and a declared lock that the readings contradict is reported as
/// a contradiction rather than believed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ClockLock {
    /// `nvidia-smi -lgc` applied (root VM). Canonical publication rounds live here.
    Locked(Option<i32>),
    /// A container (Modal / RunPod / Vast): clocks cannot be locked, so the round is iteration data.
    Unlocked,
    #[default]
    Unknown,
}

impl fmt::Display for ClockLock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClockLock::Locked(Some(mhz)) => write!(f, "locked at {mhz} MHz (declared)"),
            ClockLock::Locked(None) => write!(f, "locked (declared, no frequency given)"),
            ClockLock::Unlocked => {
                write!(f, "UNLOCKED (declared) - container round, iteration data")
            }
            ClockLock::Unknown => write!(f, "UNKNOWN (undeclared)"),
        }
    }
}

/// Parse `WUKONG_GPU_CLOCK_LOCK`: `locked:1410`, `locked`, `unlocked`, or anything else -> unknown.
pub fn parse_clock_lock(v: &str) -> ClockLock {
    let v = v.trim().to_ascii_lowercase();
    if let Some(rest) = v.strip_prefix("locked") {
        let mhz = rest
            .trim_start_matches([':', '='])
            .trim()
            .parse::<i32>()
            .ok();
        return ClockLock::Locked(mhz);
    }
    if v == "unlocked" {
        return ClockLock::Unlocked;
    }
    ClockLock::Unknown
}

/// Operator-supplied round metadata: who is renting what, at what price, with what clock policy.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RoundMeta {
    pub provider: Option<String>,
    pub sku: Option<String>,
    pub usd_per_hour: Option<f64>,
    pub clock_lock: ClockLock,
}

impl RoundMeta {
    /// From `WUKONG_GPU_PROVIDER`, `WUKONG_GPU_SKU`, `WUKONG_GPU_USD_PER_HR`,
    /// `WUKONG_GPU_CLOCK_LOCK`. All optional; an absent one is printed as `n/a` rather than guessed.
    pub fn from_env() -> Self {
        let s = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        Self {
            provider: s("WUKONG_GPU_PROVIDER"),
            sku: s("WUKONG_GPU_SKU"),
            usd_per_hour: s("WUKONG_GPU_USD_PER_HR").and_then(|v| v.trim().parse().ok()),
            clock_lock: s("WUKONG_GPU_CLOCK_LOCK")
                .map(|v| parse_clock_lock(&v))
                .unwrap_or_default(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------------------------

/// The §6.1 provenance block, **emitted by the harness**.
///
/// It used to live in a PowerShell runbook, which is the same as saying it could be forgotten: a
/// round run by hand, or by a different script, or from a test binary directly, simply had no
/// provenance. Here it is produced by the code that takes the measurements, from the device the
/// measurements ran on.
#[derive(Debug, Clone)]
pub struct Provenance {
    pub round: String,
    pub facts: DeviceFacts,
    pub check: DeviceCheck,
    pub meta: RoundMeta,
    pub before: Option<SmiSample>,
    pub drift_bar: f64,
}

impl Provenance {
    /// Build the block and run the device check.
    pub fn new(
        round: &str,
        facts: DeviceFacts,
        meta: RoundMeta,
        before: Option<SmiSample>,
    ) -> Self {
        let check = check_device(&facts);
        Self {
            round: round.to_string(),
            facts,
            check,
            meta,
            before,
            drift_bar: DEFAULT_CLOCK_DRIFT_BAR,
        }
    }

    /// The pre-round half of the header — available even when the device refuses, because
    /// `bench/gpu/README.md` rule 3 requires the refusing round to be logged as evidence too.
    pub fn header(&self) -> String {
        let f = &self.facts;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "# ---- wukong GPU round provenance (GPU_RETARGET_PLAN.md 6.1, emitted by the harness) ----"
        );
        let _ = writeln!(s, "round          : {}", self.round);
        let _ = writeln!(s, "gpu            : {}", f.name);
        let _ = writeln!(s, "compute cap    : {}.{}", f.cc_major, f.cc_minor);
        let _ = writeln!(s, "SMs            : {}", f.sm_count);
        let _ = writeln!(s, "spec check     : {}", self.check);
        let _ = writeln!(s, "opt-in SMEM    : {} B", f.smem_per_block_optin);
        let _ = writeln!(s, "L2             : {} B", f.l2_bytes);
        let _ = writeln!(s, "VRAM           : {} B", f.total_mem);
        // `cuDriverGetVersion` encodes 1000*major + 10*minor, so 12090 is CUDA 12.9. Decoded here
        // because a round log is read by a human comparing it against the provider's image tag.
        let _ = writeln!(
            s,
            "driver CUDA API: {}.{} ({})",
            f.driver_version / 1000,
            (f.driver_version % 1000) / 10,
            f.driver_version
        );
        // §6.1 asks for the CUDA runtime too. There isn't one: this backend is driver-API only
        // (`cuModuleLoadData` JITs the PTX), which is exactly why it needs no toolkit. Saying so is
        // the honest answer; leaving the line out would read as an omission.
        let _ = writeln!(
            s,
            "CUDA runtime   : none linked (driver API only: PTX is JIT-loaded by the driver)"
        );
        let _ = writeln!(
            s,
            "kmd driver     : {}",
            self.before
                .as_ref()
                .map(|b| b.driver.as_str())
                .unwrap_or("n/a")
        );
        let _ = writeln!(
            s,
            "provider / SKU : {} / {}",
            self.meta.provider.as_deref().unwrap_or("n/a"),
            self.meta.sku.as_deref().unwrap_or("n/a")
        );
        let _ = writeln!(s, "clock lock     : {}", self.meta.clock_lock);
        let _ = writeln!(s, "clocks before  : {}", fmt_smi(self.before.as_ref()));
        s
    }
}

fn fmt_smi(s: Option<&SmiSample>) -> String {
    match s {
        None => "n/a (no nvidia-smi reading)".to_string(),
        Some(s) => {
            let o = |v: Option<f64>, u: &str| match v {
                Some(v) => format!("{v:.1} {u}"),
                None => format!("n/a {u}"),
            };
            format!(
                "sm {:.0} MHz  mem {}  temp {}  power {}",
                s.sm_clock_mhz,
                o(s.mem_clock_mhz, "MHz"),
                o(s.temp_c, "C"),
                o(s.power_w, "W")
            )
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The round, and the only way to mint a number
// ---------------------------------------------------------------------------------------------

/// A published number. **There is no public constructor**: the only way to obtain one is
/// [`Round::publish`], which means every number that reaches a document has passed the device
/// check, the clock evidence, the drift bar, the alias check and the floor.
#[derive(Debug, Clone)]
pub struct Published {
    bench: String,
    field: String,
    baseline: f64,
    contender: f64,
    ratio: f64,
    floor: f64,
    rounds: usize,
    cell: Cell,
}

impl Published {
    pub fn bench(&self) -> &str {
        &self.bench
    }
    pub fn field(&self) -> &str {
        &self.field
    }
    pub fn baseline(&self) -> f64 {
        self.baseline
    }
    pub fn contender(&self) -> f64 {
        self.contender
    }
    /// `B / A`, raw. **This module is unit-agnostic and does not know which direction is good** —
    /// `0.80` is a win for a millisecond field and a loss for a GB/s one. Whoever names the field
    /// owns its polarity; nothing here will catch a ratio quoted the wrong way round.
    pub fn ratio(&self) -> f64 {
        self.ratio
    }
    /// The round's measured floor, which this number cleared.
    pub fn floor(&self) -> f64 {
        self.floor
    }
    pub fn is_tie(&self) -> bool {
        self.cell == Cell::Tie
    }
}

impl fmt::Display for Published {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {:.6} -> {:.6}  ({:.4}x, {}, floor +/-{:.2}%, median of {})",
            self.field,
            self.baseline,
            self.contender,
            self.ratio,
            self.cell,
            self.floor * 100.0,
            self.rounds
        )
    }
}

#[derive(Debug, Clone)]
struct Closed {
    wall: Duration,
    after: Option<SmiSample>,
}

/// An open measurement round.
///
/// **Construction is the device gate.** [`Round::open`] returns `Err` for a MIG slice, a vGPU
/// profile, an SM count off spec or a part the table does not know, so a round that fails §6.1
/// cannot produce a [`Published`] at all — there is no flag to pass, no warning to ignore.
#[derive(Debug)]
pub struct Round {
    prov: Provenance,
    started: Instant,
    closed: Option<Closed>,
}

impl Round {
    /// Open a round, refusing outright unless the device verifies against [`DEVICE_SPECS`].
    pub fn open(prov: Provenance) -> Result<Self, Refusal> {
        if !prov.check.verified() {
            return Err(Refusal::Device(format!(
                "{} -> {}",
                prov.facts.name, prov.check
            )));
        }
        Ok(Self {
            prov,
            started: Instant::now(),
            closed: None,
        })
    }

    pub fn provenance(&self) -> &Provenance {
        &self.prov
    }

    /// Stop the clock and record the after-the-round reading. Pass [`query_smi`].
    pub fn close(&mut self, after: Option<SmiSample>) {
        self.closed = Some(Closed {
            wall: self.started.elapsed(),
            after,
        });
    }

    /// Wall time so far (or the closed total).
    pub fn wall(&self) -> Duration {
        self.closed
            .as_ref()
            .map(|c| c.wall)
            .unwrap_or_else(|| self.started.elapsed())
    }

    /// `|after/before - 1|` on the SM clock, when both readings exist.
    pub fn clock_drift(&self) -> Option<f64> {
        let before = self.prov.before.as_ref()?;
        let after = self.closed.as_ref()?.after.as_ref()?;
        rel_gap(after.sm_clock_mhz, before.sm_clock_mhz).map(f64::abs)
    }

    /// The round-level half of the gate: device (already enforced at `open`), closure, clock
    /// evidence at both ends, and drift. Bench-level reasons are checked by [`Round::publish`].
    pub fn gate(&self) -> Result<(), Refusal> {
        let Some(closed) = self.closed.as_ref() else {
            return Err(Refusal::RoundNotClosed);
        };
        if self.prov.before.is_none() {
            return Err(Refusal::NoClockEvidence("before-the-round"));
        }
        if closed.after.is_none() {
            return Err(Refusal::NoClockEvidence("after-the-round"));
        }
        match self.clock_drift() {
            Some(d) if d > self.prov.drift_bar => Err(Refusal::ClockDrift {
                drift: d,
                bar: self.prov.drift_bar,
            }),
            Some(_) => Ok(()),
            None => Err(Refusal::NoClockEvidence("comparable SM")),
        }
    }

    /// **The publish gate.** Returns a number only when the round *and* the bench *and* the cell all
    /// clear; otherwise the [`Refusal`] says which one did not.
    pub fn publish(&self, v: &BenchVerdict, field: &str) -> Result<Published, Refusal> {
        self.gate()?;
        if let Some(r) = &v.blocked {
            return Err(r.clone());
        }
        let fv = v
            .field(field)
            .ok_or_else(|| Refusal::UnknownField(field.to_string()))?;
        if !fv.cell.publishable() {
            return Err(Refusal::Cell {
                field: field.to_string(),
                cell: fv.cell,
            });
        }
        let (Some(a), Some(b), Some(floor)) = (fv.a, fv.b, v.floor) else {
            return Err(Refusal::Cell {
                field: field.to_string(),
                cell: Cell::Degenerate,
            });
        };
        Ok(Published {
            bench: v.bench.clone(),
            field: field.to_string(),
            baseline: a,
            contender: b,
            ratio: b / a,
            floor,
            rounds: v.rounds,
            cell: fv.cell,
        })
    }

    /// A table cell that is always safe to print: the number, or `--` **and the reason there is no
    /// number**. A silent blank is how a refusal turns back into a convention.
    pub fn cell(&self, v: &BenchVerdict, field: &str) -> String {
        match self.publish(v, field) {
            Ok(p) => p.to_string(),
            Err(r) => format!("-- ({r})"),
        }
    }

    /// The full round header: the pre-round block, the closing reading, drift, wall time, cost, and
    /// the gate verdict — the block a `bench/gpu/<device>/` log opens with.
    pub fn header(&self) -> String {
        let mut s = self.prov.header();
        let after = self.closed.as_ref().and_then(|c| c.after.as_ref());
        let _ = writeln!(s, "clocks after   : {}", fmt_smi(after));
        let _ = writeln!(
            s,
            "sm clock drift : {} (bar +/-{:.2}%)",
            match self.clock_drift() {
                Some(d) => format!("{:+.2}%", d * 100.0),
                None => "n/a".to_string(),
            },
            self.prov.drift_bar * 100.0
        );
        if matches!(self.prov.meta.clock_lock, ClockLock::Locked(_)) {
            if let Some(d) = self.clock_drift() {
                if d > self.prov.drift_bar {
                    let _ = writeln!(
                        s,
                        "*** the declared clock LOCK is contradicted by the readings ***"
                    );
                }
            }
        }
        let wall = self.wall().as_secs_f64();
        let _ = writeln!(s, "wall           : {wall:.1} s");
        let _ = writeln!(
            s,
            "cost estimate  : {}",
            match self.prov.meta.usd_per_hour {
                Some(rate) => format!("{:.4} USD at {:.2} USD/hr", rate * wall / 3600.0, rate),
                None => "n/a (set WUKONG_GPU_USD_PER_HR)".to_string(),
            }
        );
        match self.gate() {
            Ok(()) => {
                let _ = writeln!(s, "publish gate   : OPEN (round level)");
            }
            Err(r) => {
                let _ = writeln!(s, "publish gate   : CLOSED - {r}");
                let _ = writeln!(s, "*** THIS ROUND PUBLISHES NOTHING ***");
            }
        }
        s
    }
}

// ---------------------------------------------------------------------------------------------
// The PTX twin (text half: device-free, so its gates run in a plain `cargo test`)
// ---------------------------------------------------------------------------------------------

/// The built-in twin probe's entry name. One entry, one text — both arms load the *same* PTX.
pub const TWIN_PROBE_ENTRY: &str = "wukong_bench_twin_probe";

/// PTX for the built-in twin probe: a grid-stride `y[i] = a*x[i] + y[i]`.
///
/// Deliberately the dullest memory-bound kernel there is. Its job is not to be representative; it is
/// to be *identical between two module handles* and to move enough bytes that the reading is device
/// work rather than launch overhead. With `a = 1` and `x = 1`, every launch adds exactly `1.0` to
/// every lane, so `y[0]` counts the launches **exactly** in f32 (integers below 2^24) — which is how
/// [`TwinBuffers::verify`] proves the control actually ran instead of silently measuring an empty
/// stream. A control that did not run is worse than no control.
///
/// Tagged at the [`crate::ptx_target::HDR_SM80`] floor like every other Ampere-legal family.
pub fn twin_probe_ptx() -> String {
    format!(
        "{hdr}
.visible .entry {entry}(
    .param .u32 n,
    .param .f32 a,
    .param .u64 x,
    .param .u64 y
)
{{
    .reg .pred  %p<2>;
    .reg .f32   %f<4>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<6>;

    ld.param.u32    %r1, [n];
    ld.param.f32    %f1, [a];
    ld.param.u64    %rd1, [x];
    ld.param.u64    %rd2, [y];

    cvta.to.global.u64  %rd1, %rd1;
    cvta.to.global.u64  %rd2, %rd2;

    mov.u32     %r2, %ntid.x;
    mov.u32     %r3, %ctaid.x;
    mov.u32     %r4, %tid.x;
    mad.lo.s32  %r5, %r3, %r2, %r4;
    mov.u32     %r6, %nctaid.x;
    mul.lo.s32  %r7, %r6, %r2;

LOOP:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra    DONE;

    mul.wide.u32 %rd3, %r5, 4;
    add.s64     %rd4, %rd1, %rd3;
    add.s64     %rd5, %rd2, %rd3;
    ld.global.f32   %f2, [%rd4];
    ld.global.f32   %f3, [%rd5];
    fma.rn.f32      %f2, %f1, %f2, %f3;
    st.global.f32   [%rd5], %f2;

    add.s32     %r5, %r5, %r7;
    bra         LOOP;

DONE:
    ret;
}}
",
        hdr = crate::ptx_target::HDR_SM80,
        entry = TWIN_PROBE_ENTRY
    )
}

/// **The GPU twin control**: one PTX text, two module handles, timed as two columns.
///
/// The two arms differ in exactly one respect — which `&'static str` key their module is cached
/// under — and in nothing else. `C/A` therefore has expected value 1.00 and carries no content, so
/// its spread *is* the round's noise floor.
#[derive(Debug, Clone)]
pub struct PtxTwin {
    key_a: &'static str,
    key_c: &'static str,
    entry: String,
    ptx: String,
}

impl PtxTwin {
    /// Build a twin, refusing the one mistake that would make it vacuous.
    ///
    /// `Gpu::function` caches on `key` alone and never re-examines the PTX, so two arms sharing a key
    /// are one `CUmodule` and one `CUfunction`: every row would tie perfectly having compared
    /// nothing. That is the same failure `tools/perf/gpu_ab.ps1` catches by sha256-ing its two arm
    /// binaries — the two arms there legitimately share a cargo hash in their *filename*, so the
    /// mistake looks correct right up until it proves nothing.
    pub fn new(
        key_a: &'static str,
        key_c: &'static str,
        entry: &str,
        ptx: String,
    ) -> Result<Self, String> {
        if key_a == key_c {
            return Err(format!(
                "PtxTwin: both arms were given the module key {key_a:?}. Gpu::function caches on the \
                 key alone, so the two arms would be ONE module and ONE function: the control would \
                 compare nothing and every row would tie."
            ));
        }
        if !ptx.is_ascii() {
            return Err(
                "PtxTwin: PTX must be pure ASCII (one non-ASCII byte is a ptxas fatal at \
                        cuModuleLoadData)"
                    .to_string(),
            );
        }
        if !ptx.contains(&format!(".visible .entry {entry}(")) {
            return Err(format!(
                "PtxTwin: the PTX declares no `.visible .entry {entry}(`"
            ));
        }
        Ok(Self {
            key_a,
            key_c,
            entry: entry.to_string(),
            ptx,
        })
    }

    /// The built-in probe twin — a floor with no setup at all.
    pub fn probe() -> Self {
        Self::new(
            "bench_instrument_twin_a",
            "bench_instrument_twin_c",
            TWIN_PROBE_ENTRY,
            twin_probe_ptx(),
        )
        .expect("the built-in probe twin is well-formed")
    }

    /// The module key for an arm. `B` has no key of its own: the contender is whatever the bench is
    /// actually testing, not a third copy of the twin.
    pub fn key(&self, arm: Arm) -> Option<&'static str> {
        match arm {
            Arm::Baseline => Some(self.key_a),
            Arm::Control => Some(self.key_c),
            Arm::Contender => None,
        }
    }

    pub fn entry(&self) -> &str {
        &self.entry
    }

    pub fn ptx(&self) -> &str {
        &self.ptx
    }
}

// ---------------------------------------------------------------------------------------------
// Device layer — everything below needs a real device (`--features gpu`)
// ---------------------------------------------------------------------------------------------

/// The probed [`DeviceFacts`] of a live `Gpu`, straight from its `GpuTarget` (probed once at
/// construction — this adds no driver calls).
#[cfg(feature = "gpu")]
pub fn facts_of(g: &crate::gpu::Gpu) -> DeviceFacts {
    let t = g.target();
    DeviceFacts {
        name: t.name.clone(),
        cc_major: t.cc_major,
        cc_minor: t.cc_minor,
        sm_count: t.sm_count,
        smem_per_block_optin: t.smem_per_block_optin,
        l2_bytes: t.l2_bytes,
        total_mem: t.total_mem,
        driver_version: t.driver_version,
    }
}

/// The three-line adoption: probe the device, read the clocks, check against spec, open a round.
///
/// `Err` here means the round publishes nothing — print it and return. Keep the log: a refusal is a
/// finding (`bench/gpu/README.md` rule 3), and [`Provenance::header`] still formats for it.
#[cfg(feature = "gpu")]
pub fn open_round(name: &str, g: &crate::gpu::Gpu) -> Result<Round, Refusal> {
    Round::open(Provenance::new(
        name,
        facts_of(g),
        RoundMeta::from_env(),
        query_smi(),
    ))
}

/// Resident buffers for the twin probe, plus the launch counter that proves it ran.
#[cfg(feature = "gpu")]
pub struct TwinBuffers {
    n: u32,
    a: f32,
    x: cudarc::driver::CudaSlice<f32>,
    y: cudarc::driver::CudaSlice<f32>,
    grid: u32,
    /// Every launch adds exactly `1.0` to every lane, so this is the value `y` must hold.
    launches: u64,
}

#[cfg(feature = "gpu")]
impl TwinBuffers {
    /// Allocate `n` f32 of `x = 1.0` and `y = 0.0` on the device. `n` is rounded to nothing — the
    /// kernel is grid-strided, so any `n` is legal.
    ///
    /// The grid follows the one pattern in this crate that scales correctly across parts
    /// (`ptx_optim::grid_stride_cfg`): 256-thread blocks, `32 * sm_count` of them, capped at the work
    /// available so a small `n` does not launch mostly-idle blocks.
    pub fn new(g: &crate::gpu::Gpu, n: usize) -> Result<Self, cudarc::driver::DriverError> {
        assert!(n > 0, "TwinBuffers: n must be positive");
        let x = g.stream.memcpy_stod(&vec![1.0f32; n])?;
        let y = g.stream.memcpy_stod(&vec![0.0f32; n])?;
        let want = (n as u32).div_ceil(256);
        let grid = want.min(32 * g.sm_count().max(1) as u32).max(1);
        Ok(Self {
            n: n as u32,
            a: 1.0,
            x,
            y,
            grid,
            launches: 0,
        })
    }

    /// **The control ran.** `y[0]` must equal the launch count exactly (integers below 2^24 are exact
    /// in f32). A control column that silently measured an empty stream would report a beautiful,
    /// meaningless floor.
    pub fn verify(&self, g: &crate::gpu::Gpu) -> Result<(), String> {
        let got = g
            .stream
            .memcpy_dtov(&self.y)
            .map_err(|e| format!("twin probe readback failed: {e:?}"))?;
        let want = self.launches as f32;
        if got.first().copied() != Some(want) {
            return Err(format!(
                "twin probe did not run as counted: y[0] = {:?}, expected {want} after {} launches",
                got.first(),
                self.launches
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "gpu")]
impl PtxTwin {
    /// Load **both** module handles and warm **both** of them.
    ///
    /// This is where the cubin-cache asymmetry is neutralized. The two arms share one PTX text, so
    /// they share one on-disk cubin entry: arm `A`'s load compiles and persists it and arm `C`'s
    /// load reads it back warm. That difference is real, and it is confined here — outside every
    /// timed region — after which both handles hold identical SASS and are equally resident. Timing
    /// module *loads* under this scheme would compare a compile against a file read; use
    /// [`PtxTwin::time_module_load`] for that question instead — and read what it says about why
    /// there is no in-process twin for a module load.
    pub fn prime(
        &self,
        g: &mut crate::gpu::Gpu,
        buf: &mut TwinBuffers,
        warmups: usize,
    ) -> Result<(), String> {
        let fa = g
            .function(self.key_a, &self.ptx, &self.entry)
            .map_err(|e| format!("twin arm A load failed: {e:?}"))?;
        let fc = g
            .function(self.key_c, &self.ptx, &self.entry)
            .map_err(|e| format!("twin arm C load failed: {e:?}"))?;
        for _ in 0..warmups.max(1) {
            self.launch(g, &fa, buf, 1)
                .map_err(|e| format!("twin arm A warm-up failed: {e:?}"))?;
            self.launch(g, &fc, buf, 1)
                .map_err(|e| format!("twin arm C warm-up failed: {e:?}"))?;
        }
        g.stream
            .synchronize()
            .map_err(|e| format!("twin warm-up sync failed: {e:?}"))?;
        Ok(())
    }

    /// The loaded entry for one arm. Call [`PtxTwin::prime`] first.
    pub fn arm(
        &self,
        g: &mut crate::gpu::Gpu,
        arm: Arm,
    ) -> Result<cudarc::driver::CudaFunction, String> {
        let key = self
            .key(arm)
            .ok_or_else(|| format!("PtxTwin has no handle for arm {}", arm.tag()))?;
        g.function(key, &self.ptx, &self.entry)
            .map_err(|e| format!("twin arm {} load failed: {e:?}", arm.tag()))
    }

    fn launch(
        &self,
        g: &crate::gpu::Gpu,
        f: &cudarc::driver::CudaFunction,
        buf: &mut TwinBuffers,
        iters: usize,
    ) -> Result<(), cudarc::driver::DriverError> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let cfg = LaunchConfig {
            grid_dim: (buf.grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        for _ in 0..iters {
            let mut b = g.stream.launch_builder(f);
            b.arg(&buf.n).arg(&buf.a).arg(&buf.x).arg(&mut buf.y);
            // Four `.param`s declared, four arguments pushed — derived from the same generator as the
            // entry name (`twin_probe_ptx`), per the crate's launch-seam rule.
            unsafe { b.launch(cfg)? };
            buf.launches += 1;
        }
        Ok(())
    }

    /// Time one arm's **steady state**: `iters` launches on resident buffers, one synchronize at each
    /// end, returning seconds per launch. Module loading and the first-touch page faults are already
    /// paid by [`PtxTwin::prime`].
    ///
    /// **Why host wall-clock and not CUDA events**, given that §6.3 names event timing: a control's
    /// job is to measure the floor of *the instrument the bench actually uses*, and every one of this
    /// crate's ~80 existing bench timing sites is `Instant` around a synchronized batch. An event-
    /// timed control over a wall-clock-timed bench would certify a floor the bench never runs at.
    /// Whatever bias the batched host clock carries is common-mode across `A` and `C` and cancels in
    /// `C/A` — which is the point. A bench that *does* time with events is already served: it passes
    /// its own closures to [`run_twin`] and this method is not involved.
    pub fn time_arm(
        &self,
        g: &mut crate::gpu::Gpu,
        arm: Arm,
        buf: &mut TwinBuffers,
        iters: usize,
    ) -> Result<f64, String> {
        assert!(iters > 0, "time_arm: iters must be positive");
        let f = self.arm(g, arm)?;
        g.stream
            .synchronize()
            .map_err(|e| format!("pre-timing sync failed: {e:?}"))?;
        let t0 = Instant::now();
        self.launch(g, &f, buf, iters)
            .map_err(|e| format!("twin arm {} launch failed: {e:?}", arm.tag()))?;
        g.stream
            .synchronize()
            .map_err(|e| format!("post-timing sync failed: {e:?}"))?;
        Ok(t0.elapsed().as_secs_f64() / iters as f64)
    }

    /// Time **one** `cuModuleLoadData` of this twin's PTX, bypassing both of Wukong's caches — and
    /// refuse to do it twice in one process.
    ///
    /// **Module-load latency is not twinnable inside a process, and this was measured, not assumed.**
    /// The first draft of this method claimed the direct PTX JIT was "identical work for both arms by
    /// construction", because it bypasses the in-process module map *and* the on-disk cubin cache.
    /// The `#[ignore]`d diagnostic gate below disproved that, and re-running it on a second day and a
    /// second driver reproduced it — byte-identical PTX, a discarded warm-up load, three consecutive
    /// runs each time:
    ///
    /// ```text
    /// driver 5xx, first sitting     A 0.211 ms  C 0.160 ms  -24.2%     A 0.196/C 0.149  -24.0%     A 0.185/C 0.134  -27.3%
    /// driver 592.82, re-run         A 0.409 ms  C 0.294 ms  -28.3%     A 0.328/C 0.224  -31.6%     A 0.247/C 0.197  -20.5%
    /// ```
    ///
    /// Arm `C` reads a reproducible **20-32% "improvement"** on work that is byte-identical to arm
    /// `A`'s, and the effect does not saturate after one warm-up: every load of the text in a process
    /// is faster than the one before it. The driver keeps its own JIT cache (on disk under
    /// `~/.nv/ComputeCache`, disableable with `CUDA_CACHE_DISABLE=1`, and warm across *processes* —
    /// the first load of a never-before-seen text measured 8.8 ms here against ~0.3 ms once warm),
    /// plus per-context state that the second load of the same text does not re-pay. Removing our two
    /// caches removes neither.
    ///
    /// So the honest protocol for a load-latency claim is **one measurement per process**, alternated
    /// across processes exactly as `tools/perf/gpu_ab.ps1` alternates its two arm binaries. This
    /// method enforces the first half of that: a second call in the same process returns `Err`
    /// naming the reason, so a two-arm in-process load twin cannot be written by accident.
    ///
    /// (To time the *production* cached path instead, call [`PtxTwin::evict_cubin`] before the
    /// measurement so the load does not inherit an earlier process's cubin.)
    pub fn time_module_load(&self, g: &crate::gpu::Gpu) -> Result<f64, String> {
        use std::sync::atomic::{AtomicBool, Ordering};
        static TIMED: AtomicBool = AtomicBool::new(false);
        if TIMED.swap(true, Ordering::SeqCst) {
            return Err(
                "time_module_load: a module load was already timed in this process. The second load \
                 of a PTX text is served by the driver's own JIT cache and per-context state — \
                 measured 24-27% faster than the first on the 4050, byte-identical input, warm-up \
                 discarded — so a two-arm load twin inside one process compares cold against warm. \
                 Time ONE load per process and alternate processes."
                    .to_string(),
            );
        }
        let t0 = Instant::now();
        let _m = g
            .ctx
            .load_module(self.ptx.as_str().into())
            .map_err(|e| format!("PTX JIT failed: {e:?}"))?;
        Ok(t0.elapsed().as_secs_f64())
    }

    /// Remove this twin's shared on-disk cubin entry, if any. Returns whether a file was removed.
    ///
    /// Only useful for a load-latency measurement, and only alongside the process rule in
    /// [`PtxTwin::time_module_load`]: evicting our cubin still leaves the driver's own JIT cache warm.
    pub fn evict_cubin(&self) -> bool {
        let path = crate::cubin::cache_path(&self.ptx, crate::cubin::driver_version());
        std::fs::remove_file(path).is_ok()
    }
}

/// Is the NVIDIA driver's own PTX->SASS JIT cache disabled for this process (`CUDA_CACHE_DISABLE=1`)?
///
/// Read it when reporting a load-latency number: with the cache enabled — the default — the *first*
/// load of a given PTX in a *fresh* process is still served from `~/.nv/ComputeCache` if any earlier
/// process compiled that exact text, which on the 4050 is the difference between 8.0 ms and 0.25 ms.
/// Un-gated: it is an environment fact, not a device call.
pub fn driver_jit_cache_disabled() -> bool {
    std::env::var("CUDA_CACHE_DISABLE")
        .map(|v| v.trim() == "1")
        .unwrap_or(false)
}

/// **The machine floor**: run the built-in probe twin and report `max |C/A - 1|`.
///
/// This is the floor of the *machine's* launch and clock behaviour — a lower bound on any bench's
/// own floor, obtainable in a couple of seconds. Its use is triage: if the machine floor already
/// exceeds the bar, no bench in this round can publish, and it is far cheaper to learn that now than
/// after forty minutes of sweeps. It is **not** a substitute for a bench's own control: the honest
/// floor for a claim about GEMM is GEMM measured twice, not saxpy measured twice.
#[cfg(feature = "gpu")]
pub fn machine_floor(g: &mut crate::gpu::Gpu, plan: &TwinPlan) -> Result<BenchVerdict, String> {
    const N: usize = 1 << 22; // 16 MiB per buffer: past every cache on every target part
    const ITERS: usize = 20;
    let twin = PtxTwin::probe();
    let mut buf = TwinBuffers::new(g, N).map_err(|e| format!("twin buffers: {e:?}"))?;
    twin.prime(g, &mut buf, plan.warmups)?;

    let mut err: Option<String> = None;
    let samples = run_rotated("machine_floor", plan, |arm| {
        if err.is_some() || arm == Arm::Contender {
            return Vec::new();
        }
        match twin.time_arm(g, arm, &mut buf, ITERS) {
            Ok(s) => vec![read("probe_us", s * 1e6)],
            Err(e) => {
                err = Some(e);
                Vec::new()
            }
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    buf.verify(g)?;
    Ok(analyze(&samples, plan.bar))
}

// ---------------------------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(name: &str, cc: (i32, i32), sms: i32) -> DeviceFacts {
        DeviceFacts {
            name: name.to_string(),
            cc_major: cc.0,
            cc_minor: cc.1,
            sm_count: sms,
            smem_per_block_optin: 101_376,
            l2_bytes: 25_165_824,
            total_mem: 6_442_450_944,
            driver_version: 12_090,
        }
    }

    fn smi(sm_mhz: f64) -> SmiSample {
        SmiSample {
            driver: "580.95.05".to_string(),
            sm_clock_mhz: sm_mhz,
            mem_clock_mhz: Some(6250.0),
            temp_c: Some(45.0),
            power_w: Some(31.5),
        }
    }

    /// A verified 4050 round, closed with no drift — the fixture every gate-shape test starts from.
    fn open_ok(name: &str) -> Round {
        let prov = Provenance::new(
            name,
            facts("NVIDIA GeForce RTX 4050 Laptop GPU", (8, 9), 20),
            RoundMeta::default(),
            Some(smi(2100.0)),
        );
        let mut r = Round::open(prov).expect("the 4050 verifies against the spec table");
        r.close(Some(smi(2100.0)));
        r
    }

    /// One hand-written field: its name and the `A`, `C`, `B` per-round vectors.
    type FieldFixture<'a> = (&'a str, Vec<f64>, Vec<f64>, Vec<f64>);

    /// Build samples by hand, bypassing the driver, so the analysis can be fed exact shapes.
    fn samples_of(bench: &str, rounds: usize, fields: &[FieldFixture<'_>]) -> BenchSamples {
        let mut s = BenchSamples::new(bench);
        s.rounds = rounds;
        for (name, a, c, b) in fields {
            for v in a {
                s.push(name, Arm::Baseline, *v);
            }
            for v in c {
                s.push(name, Arm::Control, *v);
            }
            for v in b {
                s.push(name, Arm::Contender, *v);
            }
        }
        s
    }

    // --- statistics ---------------------------------------------------------------------------

    #[test]
    fn median_is_the_middle_and_refuses_non_finite_samples() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&[]), None);
        // One bad timing must poison the field rather than sort to an end and yield a plausible
        // number: NaN's comparisons are all false, so a naive sort would silently reorder nothing.
        assert_eq!(median(&[1.0, f64::NAN, 3.0]), None);
        assert_eq!(median(&[1.0, f64::INFINITY]), None);
    }

    /// The summary is a MEDIAN, not a minimum. For a rate field a minimum picks the *worst* sample,
    /// and for a printed ratio field it is meaningless — which is why the pre-registered analysis
    /// says medians and this pins it.
    #[test]
    fn the_summary_is_not_a_minimum() {
        let xs = [100.0, 101.0, 99.0, 250.0, 100.5];
        let m = median(&xs).unwrap();
        assert_eq!(m, 100.5);
        assert_ne!(m, 99.0, "a minimum would have been taken");
    }

    #[test]
    fn rel_gap_is_defined_only_against_a_usable_baseline() {
        assert!((rel_gap(11.0, 10.0).unwrap() - 0.1).abs() < 1e-12);
        assert_eq!(rel_gap(10.0, 10.0), Some(0.0));
        assert_eq!(rel_gap(1.0, 0.0), None);
        assert_eq!(rel_gap(f64::NAN, 1.0), None);
        assert_eq!(rel_gap(1.0, f64::INFINITY), None);
    }

    // --- the rotation driver ------------------------------------------------------------------

    #[test]
    fn rotation_moves_the_leading_arm_every_round() {
        assert_eq!(rotation(0), [Arm::Baseline, Arm::Control, Arm::Contender]);
        assert_eq!(rotation(1), [Arm::Control, Arm::Contender, Arm::Baseline]);
        assert_eq!(rotation(2), [Arm::Contender, Arm::Baseline, Arm::Control]);
        assert_eq!(rotation(3), rotation(0));
        for r in 0..12 {
            let mut seen = rotation(r).to_vec();
            seen.sort();
            seen.dedup();
            assert_eq!(seen.len(), 3, "round {r} dropped or duplicated an arm");
        }
    }

    /// The driver must (a) discard exactly `warmups` rounds, (b) call every arm in every round, and
    /// (c) rotate. Exercised with a synthetic closure — no device, no timing.
    #[test]
    fn the_driver_discards_warmups_and_records_every_arm() {
        let plan = TwinPlan::new(4, 2, 0.05).unwrap();
        let mut call_log: Vec<&'static str> = Vec::new();
        let mut tick = 0.0;
        let s = run_rotated("synthetic", &plan, |arm| {
            call_log.push(arm.tag());
            tick += 1.0;
            vec![read("t", tick)]
        });
        assert_eq!(call_log.len(), (4 + 2) * 3, "every arm runs in every round");
        assert_eq!(&call_log[0..3], &["A", "C", "B"]);
        assert_eq!(&call_log[3..6], &["C", "B", "A"], "round 1 must rotate");
        for arm in Arm::ALL {
            assert_eq!(
                s.samples("t", arm).len(),
                4,
                "{} recorded the wrong number of rounds",
                arm.tag()
            );
        }
        // The first 2*3 = 6 readings were discarded, so the smallest recorded value is 7.
        let lowest = s
            .samples("t", Arm::Baseline)
            .iter()
            .chain(s.samples("t", Arm::Control))
            .chain(s.samples("t", Arm::Contender))
            .fold(f64::INFINITY, |m, x| m.min(*x));
        assert_eq!(lowest, 7.0, "a warm-up reading leaked into the record");
    }

    /// The warm-up round is not hygiene. `TwinPlan::new` refuses to build a plan without one,
    /// because a cold cache inside the timed region once read a 27x false regression here.
    #[test]
    fn a_plan_without_a_warmup_or_with_too_few_rounds_is_refused() {
        assert!(TwinPlan::new(5, 0, 0.05).is_err());
        assert!(TwinPlan::new(2, 1, 0.05).is_err());
        assert!(TwinPlan::new(5, 1, 0.0).is_err());
        assert!(TwinPlan::new(5, 1, 1.5).is_err());
        assert!(TwinPlan::new(3, 1, 0.05).is_ok());
        let d = TwinPlan::default();
        assert!(
            TwinPlan::new(d.rounds, d.warmups, d.bar).is_ok(),
            "the default plan must be valid"
        );
    }

    #[test]
    fn run_twin_gives_the_reference_to_both_a_and_c() {
        let plan = TwinPlan::new(3, 1, 0.05).unwrap();
        let mut ref_calls = 0;
        let mut con_calls = 0;
        let s = run_twin(
            "peer",
            &plan,
            || {
                ref_calls += 1;
                vec![read("ms", 10.0)]
            },
            || {
                con_calls += 1;
                vec![read("ms", 9.0)]
            },
        );
        assert_eq!(
            ref_calls, 8,
            "the reference runs for BOTH A and C every round"
        );
        assert_eq!(con_calls, 4);
        assert_eq!(s.samples("ms", Arm::Baseline), &[10.0, 10.0, 10.0]);
        assert_eq!(s.samples("ms", Arm::Contender), &[9.0, 9.0, 9.0]);
    }

    // --- the publish gate ---------------------------------------------------------------------

    #[test]
    fn the_floor_is_the_worst_control_gap_over_the_live_fields() {
        let s = samples_of(
            "b",
            3,
            &[
                // C/A medians: 10.1/10.0 = +1%, 20.6/20.0 = +3%
                ("f1", vec![10.0; 3], vec![10.1; 3], vec![10.0; 3]),
                ("f2", vec![20.0; 3], vec![20.6; 3], vec![20.0; 3]),
            ],
        );
        let v = analyze(&s, 0.05);
        let floor = v.floor.unwrap();
        assert!(
            (floor - 0.03).abs() < 1e-9,
            "floor should be the WORST live field's gap, got {floor}"
        );
    }

    /// A tie is only a tie when the floor cleared the bar. This is P2 of the pre-registered analysis:
    /// five of the eleven 4050 benches sat far inside their own floors and still could not certify.
    #[test]
    fn a_wide_floor_publishes_nothing_even_though_b_sits_inside_it() {
        let round = open_ok("wide");
        let s = samples_of(
            "wide",
            5,
            &[(
                "ms",
                vec![100.0, 100.0, 100.0, 100.0, 100.0],
                vec![113.0, 113.0, 113.0, 113.0, 113.0], // a 13% control gap: the tensorcore case
                vec![100.5, 100.5, 100.5, 100.5, 100.5], // B is 0.5% off: "consistent with a tie"
            )],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);
        assert_eq!(v.field("ms").unwrap().cell, Cell::Unresolved);
        let err = round.publish(&v, "ms").unwrap_err();
        assert!(
            matches!(err, Refusal::FloorAboveBar { .. }),
            "expected a floor refusal, got {err}"
        );
        assert!(round.cell(&v, "ms").starts_with("-- "));
    }

    #[test]
    fn a_tight_floor_certifies_a_tie_and_a_real_move() {
        let round = open_ok("tight");
        let s = samples_of(
            "tight",
            5,
            &[
                (
                    "tie_ms",
                    vec![100.0; 5],
                    vec![100.5; 5], // +0.5% floor
                    vec![100.2; 5], // inside it
                ),
                (
                    "moved_ms",
                    vec![100.0; 5],
                    vec![100.4; 5],
                    vec![80.0; 5], // -20%: clears any plausible floor
                ),
            ],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);
        assert_eq!(v.field("tie_ms").unwrap().cell, Cell::Tie);
        assert_eq!(v.field("moved_ms").unwrap().cell, Cell::Moved);
        let p = round
            .publish(&v, "moved_ms")
            .expect("a cleared cell publishes");
        assert!((p.ratio() - 0.8).abs() < 1e-9);
        assert!(!p.is_tie());
        assert!(round.publish(&v, "tie_ms").unwrap().is_tie());
    }

    /// The GPU analogue of `gpu_ab.ps1`'s sha256 check. Two real timings are never bit-identical, so
    /// element-wise equality between `A` and `C` means the arms are the same object — the run looks
    /// immaculate (floor 0.0%) and proves nothing.
    #[test]
    fn identical_control_samples_are_refused_as_aliased_arms() {
        let round = open_ok("aliased");
        let s = samples_of(
            "aliased",
            3,
            &[(
                "ms",
                vec![10.0, 11.0, 12.0],
                vec![10.0, 11.0, 12.0], // literally the same readings
                vec![9.0, 9.0, 9.0],
            )],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);
        assert!(v.aliased);
        assert_eq!(
            v.floor,
            Some(0.0),
            "an aliased control reads a perfect floor"
        );
        assert_eq!(round.publish(&v, "ms").unwrap_err(), Refusal::AliasedArms);
    }

    /// A field that never varies is a label (a shape, a checksum), not a measurement. Left in the
    /// floor it would pin the floor at 0.0 and let any blip "clear" it.
    #[test]
    fn a_constant_field_neither_sets_the_floor_nor_publishes() {
        let round = open_ok("labels");
        let s = samples_of(
            "labels",
            3,
            &[
                ("n", vec![4096.0; 3], vec![4096.0; 3], vec![4096.0; 3]),
                (
                    "ms",
                    vec![10.0, 10.1, 9.9],
                    vec![10.2, 10.0, 10.1],
                    vec![9.0, 9.1, 8.9],
                ),
            ],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);
        assert_eq!(v.field("n").unwrap().cell, Cell::Constant);
        assert!(!v.field("n").unwrap().live);
        assert!(
            v.floor.unwrap() > 0.0,
            "the constant must not zero the floor"
        );
        assert!(matches!(
            round.publish(&v, "n").unwrap_err(),
            Refusal::Cell { .. }
        ));
        round
            .publish(&v, "ms")
            .expect("the real field still publishes");
    }

    #[test]
    fn a_missing_round_in_one_arm_blocks_the_whole_bench() {
        let round = open_ok("short");
        let s = samples_of(
            "short",
            3,
            &[
                (
                    "ms",
                    vec![10.0, 10.1, 9.9],
                    vec![10.0, 10.1],
                    vec![9.0, 9.0, 9.0],
                ),
                (
                    "gbs",
                    vec![100.0, 101.0, 99.0],
                    vec![100.0, 101.0, 99.5],
                    vec![110.0, 111.0, 109.0],
                ),
            ],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);
        assert_eq!(v.field("ms").unwrap().cell, Cell::Malformed);
        // The healthy sibling is blocked too: an arm that failed to report is a broken instrument.
        assert!(matches!(
            round.publish(&v, "gbs").unwrap_err(),
            Refusal::MalformedField(_)
        ));
    }

    #[test]
    fn a_control_only_round_measures_a_floor_and_publishes_nothing() {
        let round = open_ok("control-only");
        let s = samples_of(
            "control-only",
            3,
            &[("us", vec![10.0, 10.1, 9.9], vec![10.05, 10.0, 10.1], vec![])],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);
        assert!(
            v.floor.is_some(),
            "the floor is the point of a control-only round"
        );
        assert_eq!(v.field("us").unwrap().cell, Cell::NoContender);
        assert!(matches!(
            round.publish(&v, "us").unwrap_err(),
            Refusal::NoContenderArm
        ));
    }

    /// **When "no contender" and "floor above bar" are BOTH true, the structural one is reported.**
    ///
    /// The sibling above cannot see this: it feeds 10.0 vs 10.05, a ~0.5% floor, so `resolved` holds
    /// and only one refusal is ever live. Clean synthetic numbers hid a real precedence bug —
    /// `analyze` tested `!resolved` first and so reported `FloorAboveBar` for a control-only round,
    /// contradicting the per-field `Cell::NoContender` it had already assigned and printing
    /// "re-run rather than reporting a wide tie as a tie" at a round with no tie to report.
    ///
    /// It surfaced only on a box noisy enough for both conditions to hold at once: the dev 4050 read
    /// a **+10.21%** floor against the +/-5% bar and
    /// `the_machine_floor_measures_this_box_and_publishes_nothing` failed with
    /// `FloorAboveBar { floor: 0.1020..., bar: 0.05 }` where it expected `NoContenderArm`. This gate
    /// reproduces that state with no device and no noise: a **13% control gap and an empty B arm**.
    #[test]
    fn both_refusals_true_reports_the_structural_one() {
        let round = open_ok("wide-control-only");
        let s = samples_of(
            "wide-control-only",
            5,
            &[(
                "us",
                vec![100.0, 100.0, 100.0, 100.0, 100.0],
                vec![113.0, 113.0, 113.0, 113.0, 113.0], // 13% floor: well past the +/-5% bar
                vec![],                                  // ...and no contender arm at all
            )],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);

        // Both conditions genuinely hold, or this gate proves nothing.
        assert!(
            v.floor.expect("a control-only round still measures a floor") > DEFAULT_CONTROL_BAR,
            "the floor must exceed the bar for this test to exercise the interaction"
        );
        assert_eq!(v.field("us").unwrap().cell, Cell::NoContender);

        // The round-level refusal must agree with the per-field cell, not contradict it.
        assert!(
            matches!(v.blocked, Some(Refusal::NoContenderArm)),
            "a control-only round reports the structural refusal even when its floor is wide, \
             got {:?}",
            v.blocked
        );
        assert!(matches!(
            round.publish(&v, "us").unwrap_err(),
            Refusal::NoContenderArm
        ));
    }

    #[test]
    fn a_degenerate_baseline_never_becomes_a_ratio() {
        let round = open_ok("degenerate");
        let s = samples_of(
            "degenerate",
            3,
            &[
                (
                    "zero",
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0],
                    vec![1.0, 1.0, 1.0],
                ),
                (
                    "nan",
                    vec![1.0, f64::NAN, 1.0],
                    vec![1.0, 1.0, 1.0],
                    vec![1.0, 1.0, 1.0],
                ),
                // A signed field (a delta, a drift, an error term). `-6/-5 = 1.2` would read as a
                // clean 20% regression while meaning nothing, so it must never become a ratio.
                (
                    "signed",
                    vec![-5.0, -5.1, -4.9],
                    vec![-5.0, -5.05, -5.1],
                    vec![-6.0, -6.0, -6.0],
                ),
            ],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);
        for f in ["zero", "nan", "signed"] {
            assert!(!v.field(f).unwrap().live, "{f} must not be live");
            assert!(round.publish(&v, f).is_err());
        }
        // The label must say *which* way it failed. `zero` never varied; `nan` and `signed` did vary
        // but produced medians no ratio can be taken against. Deciding this from the medians alone
        // cannot tell those apart, which is why `analyze` carries the reason forward from the pass
        // that actually knows it.
        assert_eq!(v.field("zero").unwrap().cell, Cell::Constant);
        assert_eq!(v.field("nan").unwrap().cell, Cell::Degenerate);
        assert_eq!(v.field("signed").unwrap().cell, Cell::Degenerate);
    }

    #[test]
    fn publishing_an_unmeasured_field_is_an_error_not_a_zero() {
        let round = open_ok("unknown-field");
        let s = samples_of(
            "unknown-field",
            3,
            &[("ms", vec![10.0; 3], vec![10.1; 3], vec![9.0; 3])],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);
        assert_eq!(
            round.publish(&v, "gbs").unwrap_err(),
            Refusal::UnknownField("gbs".to_string())
        );
    }

    // --- the round-level gate -------------------------------------------------------------------

    #[test]
    fn an_unclosed_round_publishes_nothing() {
        let prov = Provenance::new(
            "unclosed",
            facts("NVIDIA GeForce RTX 4050 Laptop GPU", (8, 9), 20),
            RoundMeta::default(),
            Some(smi(2100.0)),
        );
        let round = Round::open(prov).unwrap();
        let s = samples_of(
            "unclosed",
            3,
            &[("ms", vec![10.0; 3], vec![10.1; 3], vec![9.0; 3])],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);
        assert_eq!(
            round.publish(&v, "ms").unwrap_err(),
            Refusal::RoundNotClosed
        );
    }

    #[test]
    fn a_round_without_clock_evidence_at_either_end_publishes_nothing() {
        let s = samples_of(
            "clocks",
            3,
            &[("ms", vec![10.0; 3], vec![10.1; 3], vec![9.0; 3])],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);

        let mut no_before = Round::open(Provenance::new(
            "no-before",
            facts("NVIDIA L4", (8, 9), 58),
            RoundMeta::default(),
            None,
        ))
        .unwrap();
        no_before.close(Some(smi(2040.0)));
        assert!(matches!(
            no_before.publish(&v, "ms").unwrap_err(),
            Refusal::NoClockEvidence(_)
        ));

        let mut no_after = Round::open(Provenance::new(
            "no-after",
            facts("NVIDIA L4", (8, 9), 58),
            RoundMeta::default(),
            Some(smi(2040.0)),
        ))
        .unwrap();
        no_after.close(None);
        assert!(matches!(
            no_after.publish(&v, "ms").unwrap_err(),
            Refusal::NoClockEvidence(_)
        ));
    }

    /// The laptop lesson ported: a machine whose clocks moved during the round was not one machine.
    #[test]
    fn a_clock_that_drifted_across_the_round_publishes_nothing() {
        let mut round = Round::open(Provenance::new(
            "drift",
            facts("NVIDIA H100 80GB HBM3", (9, 0), 132),
            RoundMeta {
                clock_lock: ClockLock::Locked(Some(1410)),
                ..RoundMeta::default()
            },
            Some(smi(1410.0)),
        ))
        .unwrap();
        round.close(Some(smi(1100.0))); // -22%: a thermal or power-cap throttle mid-round
        let s = samples_of(
            "drift",
            3,
            &[("ms", vec![10.0; 3], vec![10.1; 3], vec![9.0; 3])],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);
        assert!(matches!(
            round.publish(&v, "ms").unwrap_err(),
            Refusal::ClockDrift { .. }
        ));
        // The declared lock is metadata; the readings are evidence, and the header says so.
        let h = round.header();
        assert!(h.contains("declared clock LOCK is contradicted"), "{h}");
        assert!(h.contains("THIS ROUND PUBLISHES NOTHING"), "{h}");
    }

    // --- the device spec table --------------------------------------------------------------------

    #[test]
    fn the_dev_laptop_and_the_campaign_targets_verify() {
        for (name, cc, sms) in [
            ("NVIDIA GeForce RTX 4050 Laptop GPU", (8, 9), 20),
            ("NVIDIA L4", (8, 9), 58),
            ("NVIDIA L40S", (8, 9), 142),
            ("NVIDIA A100-SXM4-40GB", (8, 0), 108),
            ("NVIDIA A100 80GB PCIe", (8, 0), 108),
            ("NVIDIA H100 80GB HBM3", (9, 0), 132),
            ("NVIDIA H100 NVL", (9, 0), 132),
            ("NVIDIA H100 PCIe", (9, 0), 114),
            ("NVIDIA H200", (9, 0), 132),
            ("NVIDIA GeForce RTX 4090", (8, 9), 128),
            ("NVIDIA GeForce RTX 5090", (12, 0), 170),
            ("NVIDIA B200", (10, 0), 148),
        ] {
            let c = check_device(&facts(name, cc, sms));
            assert!(c.verified(), "{name} did not verify: {c}");
        }
    }

    /// `L4` must not match an `L40S`, and `H100` must not shadow the CUT `H100 PCIe`. Substring
    /// matching gets both of these wrong; token matching with longest-match-wins gets them right.
    #[test]
    fn the_spec_lookup_is_token_based_and_longest_match_wins() {
        assert_eq!(lookup_spec("NVIDIA L4").unwrap().sm_count, 58);
        assert_eq!(lookup_spec("NVIDIA L40S").unwrap().sm_count, 142);
        assert_eq!(lookup_spec("NVIDIA L40").unwrap().sm_count, 142);
        assert_eq!(lookup_spec("NVIDIA H100 PCIe").unwrap().sm_count, 114);
        assert_eq!(lookup_spec("NVIDIA H100 80GB HBM3").unwrap().sm_count, 132);
        // The PCIe H100 really is a different part; conflating them would silently accept a 114-SM
        // device as the 132-SM one the tuning constants were derived for.
        assert_ne!(
            lookup_spec("NVIDIA H100 PCIe").unwrap().sm_count,
            lookup_spec("NVIDIA H100 NVL").unwrap().sm_count
        );
        assert!(lookup_spec("AMD Instinct MI300X").is_none());
    }

    /// §6.1's hard rule, in all four ways it can fire.
    #[test]
    fn a_partitioned_cut_or_unknown_device_can_never_open_a_round() {
        let cases: Vec<(DeviceFacts, &str)> = vec![
            (
                facts("NVIDIA A100-SXM4-40GB MIG 1g.5gb", (8, 0), 14),
                "a MIG slice",
            ),
            (facts("GRID A100D-40C", (8, 0), 108), "a vGPU profile"),
            (
                facts("NVIDIA A100-SXM4-80GB", (8, 0), 56),
                "an A100 reporting half its SMs",
            ),
            (
                facts("NVIDIA H100 80GB HBM3", (8, 9), 132),
                "an H100 reporting Ada's capability",
            ),
            (
                facts("NVIDIA Marvellous 9000", (9, 0), 132),
                "a part not in the table",
            ),
        ];
        for (f, why) in cases {
            let check = check_device(&f);
            assert!(!check.verified(), "{why} was accepted: {check}");
            let prov = Provenance::new("x", f, RoundMeta::default(), Some(smi(1400.0)));
            assert!(
                Round::open(prov).is_err(),
                "{why} opened a round, so it could have published"
            );
        }
    }

    /// A refused round must still format a provenance block: `bench/gpu/README.md` rule 3 says the
    /// mismatch is itself a finding and the log is kept.
    #[test]
    fn a_refused_device_still_produces_a_loggable_provenance_block() {
        let prov = Provenance::new(
            "mig",
            facts("NVIDIA A100-SXM4-40GB MIG 1g.5gb", (8, 0), 14),
            RoundMeta::default(),
            Some(smi(1410.0)),
        );
        let h = prov.header();
        assert!(h.contains("REFUSED"), "{h}");
        assert!(h.contains("MIG"), "{h}");
        assert!(Round::open(prov).is_err());
    }

    /// Every row must be self-consistent, uniquely spelled, and match a plausible full device name —
    /// a typo'd pattern silently degrades to `Unknown`, which refuses every real device of that part.
    #[test]
    fn every_spec_row_is_well_formed_and_unique() {
        let mut seen: Vec<&str> = Vec::new();
        for s in DEVICE_SPECS {
            assert!(s.pattern.is_ascii() && s.pattern == s.pattern.to_uppercase());
            assert!(!s.pattern.trim().is_empty());
            assert!(s.sm_count > 0, "{}: SM count must be positive", s.pattern);
            assert!(
                s.cc.0 >= 7,
                "{}: pre-Volta parts are not targets",
                s.pattern
            );
            assert!(
                s.note.len() > 10,
                "{}: a row must record where its count came from",
                s.pattern
            );
            assert!(
                !seen.contains(&s.pattern),
                "duplicate pattern {}",
                s.pattern
            );
            seen.push(s.pattern);
            // The pattern must survive its own normalizer, or it can never match anything.
            let toks: Vec<String> = s.pattern.split_whitespace().map(str::to_string).collect();
            assert_eq!(
                name_tokens(s.pattern),
                toks,
                "{} does not normalize to itself",
                s.pattern
            );
        }
    }

    #[test]
    fn virtualization_markers_are_detected_by_name_as_well_as_by_sm_count() {
        assert!(virtualization_marker("NVIDIA A100-SXM4-40GB MIG 1g.10gb").is_some());
        assert!(virtualization_marker("GRID A100D-40C").is_some());
        assert!(virtualization_marker("NVIDIA A100D-20C").is_some());
        assert!(virtualization_marker("NVIDIA A100-SXM4-40GB").is_none());
        assert!(virtualization_marker("NVIDIA H100 PCIe").is_none());
        assert!(virtualization_marker("NVIDIA GeForce RTX 4050 Laptop GPU").is_none());
    }

    // --- provenance ------------------------------------------------------------------------------

    #[test]
    fn parse_smi_csv_reads_the_documented_query_and_tolerates_missing_cells() {
        let s = parse_smi_csv("580.95.05, 2100, 6250, 41, 28.53").unwrap();
        assert_eq!(s.driver, "580.95.05");
        assert_eq!(s.sm_clock_mhz, 2100.0);
        assert_eq!(s.power_w, Some(28.53));
        // The non-`nounits` form, and a host that will not report power.
        let s = parse_smi_csv("535.104.05, 1410 MHz, 1215 MHz, 33, [N/A]").unwrap();
        assert_eq!(s.sm_clock_mhz, 1410.0);
        assert_eq!(s.power_w, None);
        // No usable SM clock means no drift evidence, so it must be an error and not a zero.
        assert!(parse_smi_csv("535.104.05, [N/A], 1215, 33, 40").is_err());
        assert!(parse_smi_csv("535.104.05, 1410").is_err());
        assert_eq!(SMI_QUERY.split(',').count(), 5);
    }

    #[test]
    fn query_smi_is_a_graceful_probe_not_a_requirement() {
        // Absent on a CI box, present on the dev laptop; either way it must not panic, and any
        // reading it does return must be usable as drift evidence.
        if let Some(s) = query_smi() {
            assert!(s.sm_clock_mhz > 0.0, "a returned reading must be usable");
            assert!(!s.driver.trim().is_empty());
        }
    }

    #[test]
    fn clock_lock_is_parsed_as_declared_metadata() {
        assert_eq!(
            parse_clock_lock("locked:1410"),
            ClockLock::Locked(Some(1410))
        );
        assert_eq!(parse_clock_lock("LOCKED"), ClockLock::Locked(None));
        assert_eq!(parse_clock_lock("unlocked"), ClockLock::Unlocked);
        assert_eq!(parse_clock_lock("who knows"), ClockLock::Unknown);
        assert_eq!(ClockLock::default(), ClockLock::Unknown);
    }

    /// The header carries every §6.1 item, is emitted by the harness rather than a shell wrapper,
    /// and is pure ASCII (this file's prose is full of `+/-`-shaped temptations, and a round log is
    /// ANSI-stripped and diffed).
    #[test]
    fn the_provenance_header_carries_every_required_item_and_is_ascii() {
        let mut round = Round::open(Provenance::new(
            "gemm_throughput",
            facts("NVIDIA H100 80GB HBM3", (9, 0), 132),
            RoundMeta {
                provider: Some("modal".into()),
                sku: Some("H100".into()),
                usd_per_hour: Some(3.95),
                clock_lock: ClockLock::Unlocked,
            },
            Some(smi(1755.0)),
        ))
        .unwrap();
        round.close(Some(smi(1740.0)));
        let h = round.header();
        for needle in [
            "gemm_throughput",
            "NVIDIA H100 80GB HBM3",
            "compute cap    : 9.0",
            "SMs            : 132",
            "VERIFIED",
            "driver CUDA API: 12.9 (12090)",
            "CUDA runtime   : none linked",
            "kmd driver     : 580.95.05",
            "modal / H100",
            "clock lock",
            "clocks before",
            "clocks after",
            "sm clock drift",
            "wall",
            "cost estimate",
            "publish gate   : OPEN",
        ] {
            assert!(h.contains(needle), "the header is missing {needle:?}:\n{h}");
        }
        assert!(h.is_ascii(), "a round log must be pure ASCII:\n{h}");
    }

    #[test]
    fn the_verdict_table_is_ascii_and_shows_unpublishable_cells_too() {
        let s = samples_of(
            "t",
            3,
            &[
                (
                    "ms",
                    vec![10.0, 10.1, 9.9],
                    vec![10.2, 10.0, 10.0],
                    vec![9.0, 9.1, 8.9],
                ),
                ("n", vec![512.0; 3], vec![512.0; 3], vec![512.0; 3]),
            ],
        );
        let v = analyze(&s, DEFAULT_CONTROL_BAR);
        let t = v.table();
        assert!(t.is_ascii(), "{t}");
        assert!(t.contains("floor"));
        assert!(
            t.contains("constant"),
            "an excluded field must still be shown:\n{t}"
        );
        assert!(t.contains("(not in floor)"), "{t}");
    }

    // --- the PTX twin (text half) ------------------------------------------------------------------

    /// The crate's hard rule 1. One `x`/`->`/`>=` copied into the `format!` above is a `ptxas fatal`
    /// at `cuModuleLoadData`, and this file's prose is full of them.
    #[test]
    fn twin_probe_ptx_is_pure_ascii() {
        let ptx = twin_probe_ptx();
        assert!(ptx.is_ascii(), "twin probe PTX must be pure ASCII");
        assert!(TWIN_PROBE_ENTRY.is_ascii());
    }

    /// The floor rule: a module is tagged with the lowest target its instruction mix is legal on. The
    /// probe is plain f32 `fma`/`ld`/`st`, so it is `sm_80`, and it must come from `ptx_target`
    /// rather than a fresh literal (the 67-hardcoded-headers era is over).
    #[test]
    fn twin_probe_ptx_is_tagged_at_the_ampere_floor() {
        let ptx = twin_probe_ptx();
        assert!(ptx.starts_with(crate::ptx_target::HDR_SM80), "{ptx}");
        assert!(
            !ptx.contains("sm_89"),
            "the probe has no Ada-only instruction"
        );
    }

    /// The kernel's shape is load-bearing twice: four `.param`s must match the four arguments the
    /// launcher pushes (pushing short makes the driver read adjacent host stack as a pointer), and
    /// the grid-stride loop is what makes any `n` and any grid legal.
    #[test]
    fn twin_probe_ptx_has_the_shape_its_launcher_assumes() {
        let ptx = twin_probe_ptx();
        assert_eq!(
            ptx.matches(".param ").count(),
            4,
            "four params, four pushed args"
        );
        assert_eq!(
            ptx.matches(".visible .entry ").count(),
            1,
            "one entry, so both module handles resolve the same symbol"
        );
        assert!(ptx.contains(&format!(".visible .entry {TWIN_PROBE_ENTRY}(")));
        assert!(ptx.contains("%nctaid.x"), "the loop must be grid-strided");
        assert!(
            ptx.contains("fma.rn.f32"),
            "y[i] += a*x[i], exactly countable in f32"
        );
        assert!(ptx.ends_with("}\n"));
    }

    /// The whole point of two handles: two keys. One key would make the arms one `CUmodule` and one
    /// `CUfunction` — a control that compares nothing while reading a perfect 1.00.
    #[test]
    fn a_twin_that_shares_one_module_key_is_refused() {
        let err = PtxTwin::new("k", "k", TWIN_PROBE_ENTRY, twin_probe_ptx()).unwrap_err();
        assert!(err.contains("key"), "{err}");
        let t = PtxTwin::probe();
        assert_ne!(t.key(Arm::Baseline), t.key(Arm::Control));
        assert!(t.key(Arm::Baseline).is_some() && t.key(Arm::Control).is_some());
        assert_eq!(
            t.key(Arm::Contender),
            None,
            "B is the bench's own work, not a third twin"
        );
        // Both arms must carry the byte-identical text: that is what makes C/A contentless.
        assert_eq!(t.ptx(), twin_probe_ptx());
        assert_eq!(t.entry(), TWIN_PROBE_ENTRY);
    }

    /// A caller-supplied twin is validated at construction, not at `cuModuleLoadData`. The
    /// non-ASCII case is the crate's live failure mode: the surrounding Rust prose is full of
    /// `x`-shaped multiplication signs and arrows, and one of them inside a `format!` is a `ptxas
    /// fatal` on the metered box rather than a compile error at home.
    #[test]
    fn a_twin_over_malformed_ptx_is_refused_at_construction() {
        assert!(
            PtxTwin::new("a", "b", "no_such_entry", twin_probe_ptx()).is_err(),
            "an entry the PTX does not declare must be caught here"
        );
        let smuggled = twin_probe_ptx().replace("LOOP:", "LOOP: // 128\u{00d7}128 tile");
        assert!(
            !smuggled.is_ascii(),
            "the fixture must actually be non-ASCII"
        );
        let err = PtxTwin::new("a", "b", TWIN_PROBE_ENTRY, smuggled).unwrap_err();
        assert!(err.contains("ASCII"), "{err}");
        // An ASCII comment is legal PTX and must not be refused.
        assert!(PtxTwin::new(
            "a",
            "b",
            TWIN_PROBE_ENTRY,
            twin_probe_ptx().replace("LOOP:", "LOOP: // the grid-stride head")
        )
        .is_ok());
    }

    // --- device gates (skip loudly without a device; fail under WUKONG_GPU_REQUIRED=1) ------------

    /// The crate's third-skip-category doctrine: a gate that cannot reach a device must say so on
    /// stderr and must FAIL under `WUKONG_GPU_REQUIRED=1`, never report a green pass having executed
    /// nothing. Mirrors `gpu::tests::with_gpu`, which is private to that module.
    #[cfg(feature = "gpu")]
    fn with_gpu(name: &str, body: impl FnOnce(&mut crate::gpu::Gpu)) {
        let mut guard = crate::gpu::gpu();
        match guard.as_mut() {
            Some(g) => body(g),
            None => {
                let why = crate::gpu::init_error().unwrap_or("no CUDA device reachable");
                assert!(
                    !crate::gpu::gpu_required(),
                    "{name}: WUKONG_GPU_REQUIRED is set but the GPU is unusable: {why}"
                );
                eprintln!("[skip] {name}: GPU unavailable: {why}");
            }
        }
    }

    /// **Any box this suite runs on must be in the spec table.** An unlisted part is not a reason to
    /// relax the check — it is a one-row code change. Escalates under `WUKONG_GPU_REQUIRED=1` like
    /// every other skip in this crate.
    #[cfg(feature = "gpu")]
    #[test]
    fn this_box_verifies_against_the_spec_table() {
        with_gpu("this_box_verifies_against_the_spec_table", |g| {
            let f = facts_of(g);
            let check = check_device(&f);
            if !check.verified() {
                crate::diff::skip_or_fail(
                    "this_box_verifies_against_the_spec_table",
                    &format!(
                        "{} (cc {}.{}, {} SMs) -> {check}; add a DEVICE_SPECS row instead of \
                         loosening the check",
                        f.name, f.cc_major, f.cc_minor, f.sm_count
                    ),
                );
                return;
            }
            eprintln!("[gate] {} {} ok", f.name, check);
        });
    }

    /// **The twin actually runs, on two handles, and we can prove the launches happened.**
    ///
    /// The probe adds exactly `1.0` per launch to every lane, so `y[0]` is the launch count; a
    /// control column that silently measured an empty stream would otherwise report a beautiful
    /// meaningless floor. Prints both arms and their gap — that gap is what a bench's own floor is
    /// compared against.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_twin_runs_on_two_module_handles_and_counts_its_launches() {
        with_gpu(
            "the_twin_runs_on_two_module_handles_and_counts_its_launches",
            |g| {
                let twin = PtxTwin::probe();
                let mut buf = TwinBuffers::new(g, 1 << 20).expect("twin buffers");
                twin.prime(g, &mut buf, 2).expect("prime both arms");
                let a = twin
                    .time_arm(g, Arm::Baseline, &mut buf, 10)
                    .expect("arm A");
                let c = twin.time_arm(g, Arm::Control, &mut buf, 10).expect("arm C");
                buf.verify(g).expect("the probe must have run every launch");
                let gap = rel_gap(c, a).unwrap_or(f64::NAN);
                eprintln!(
                    "[gate] twin probe: A {:.3} us  C {:.3} us  C/A-1 {:+.2}%",
                    a * 1e6,
                    c * 1e6,
                    gap * 100.0
                );
                assert!(a > 0.0 && c > 0.0, "a timed arm must take time");
            },
        );
    }

    /// The machine floor end to end: a control-only round that measures what this box can resolve
    /// and publishes nothing (there is no contender to publish).
    #[cfg(feature = "gpu")]
    #[test]
    fn the_machine_floor_measures_this_box_and_publishes_nothing() {
        with_gpu(
            "the_machine_floor_measures_this_box_and_publishes_nothing",
            |g| {
                let plan = TwinPlan::new(5, 1, DEFAULT_CONTROL_BAR).unwrap();
                let v = match machine_floor(g, &plan) {
                    Ok(v) => v,
                    Err(e) => {
                        crate::diff::skip_or_fail(
                            "the_machine_floor_measures_this_box_and_publishes_nothing",
                            &e,
                        );
                        return;
                    }
                };
                eprintln!("{}", v.table());
                let floor = v
                    .floor
                    .expect("a control-only round still measures a floor");
                assert!(floor.is_finite() && floor >= 0.0);
                assert!(
                    !v.aliased,
                    "the two handles produced identical sample vectors"
                );
                assert_eq!(v.field("probe_us").unwrap().cell, Cell::NoContender);
                assert_eq!(v.blocked, Some(Refusal::NoContenderArm));
                eprintln!(
                    "[gate] machine floor +/-{:.2}% over {} rounds (bar +/-{:.2}%) -> {}",
                    floor * 100.0,
                    plan.rounds,
                    plan.bar * 100.0,
                    if floor <= plan.bar {
                        "this box can resolve a bar-sized effect"
                    } else {
                        "NOTHING measured on this box today is publishable"
                    }
                );
            },
        );
    }

    /// **A module load can be timed once per process, and the refusal of the second is the finding.**
    ///
    /// This gate started life asserting that a direct PTX JIT is symmetric between two arms because
    /// it bypasses both of Wukong's caches. It is not: on this 4050, with a discarded warm-up load
    /// and byte-identical text, the second load read `-24.2% / -24.0% / -27.3%` on three consecutive
    /// runs (and `8.0 ms -> 0.25 ms` the first time the text was ever compiled on the box). The
    /// driver's own JIT cache and per-context state survive everything we can evict, so the API now
    /// refuses the second in-process call and this gate pins that refusal.
    #[cfg(feature = "gpu")]
    #[test]
    fn a_module_load_can_be_timed_once_per_process_and_no_more() {
        with_gpu(
            "a_module_load_can_be_timed_once_per_process_and_no_more",
            |g| {
                let twin = PtxTwin::probe();
                let first = twin.time_module_load(g).expect("the first load times");
                assert!(first > 0.0, "a timed load must take time");
                let second = twin.time_module_load(g);
                let err = second.expect_err("a second in-process load must be refused");
                assert!(err.contains("one process"), "{err}");
                eprintln!(
                "[gate] one PTX JIT timed at {:.3} ms (driver JIT cache {}); the second call is \
                 refused: it would compare warm against cold",
                first * 1e3,
                if driver_jit_cache_disabled() {
                    "DISABLED"
                } else {
                    "enabled - a fresh process may still hit ~/.nv/ComputeCache"
                }
            );
            },
        );
    }

    /// **The experiment behind that refusal, reproducible on demand.**
    ///
    /// `#[ignore]`d because it deliberately builds the invalid instrument
    /// [`PtxTwin::time_module_load`] exists to refuse: a discarded warm-up load, then two timed loads
    /// of byte-identical PTX in one process, reported as `A` and `C`. A symmetric load twin would
    /// read `C/A - 1 ~ 0%`; it does not, and a reviewer who doubts the refusal can re-run this in one
    /// command instead of taking the doc comment's word for it.
    ///
    /// It asserts only what is invariant (three loads succeed and take positive time). The *number*
    /// is the output, not an assertion — its magnitude is a property of the driver's JIT cache and
    /// per-context state, so pinning a threshold here would pin a driver version.
    ///
    /// ```text
    /// cargo test -p wukong_codegen_gpu --features gpu --lib
    ///     the_in_process_load_twin_is_asymmetric -- --ignored --nocapture
    /// ```
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "diagnostic: deliberately builds the in-process load twin that time_module_load refuses"]
    fn the_in_process_load_twin_is_asymmetric_and_this_is_the_measurement() {
        with_gpu(
            "the_in_process_load_twin_is_asymmetric_and_this_is_the_measurement",
            |g| {
                let ptx = twin_probe_ptx();
                let load = |g: &crate::gpu::Gpu| -> f64 {
                    let t0 = Instant::now();
                    let m = g
                        .ctx
                        .load_module(ptx.as_str().into())
                        .expect("the probe PTX must JIT");
                    let dt = t0.elapsed().as_secs_f64();
                    drop(m);
                    dt
                };
                let discarded = load(g);
                let a = load(g);
                let c = load(g);
                for (tag, t) in [("warm-up", discarded), ("A", a), ("C", c)] {
                    assert!(t > 0.0, "load {tag} took no measurable time");
                }
                eprintln!(
                    "[diag] in-process load twin on {}: discarded {:.3} ms | A {:.3} ms  C {:.3} ms \
                     -> C/A-1 {:+.1}%  (driver JIT cache {})",
                    facts_of(g).name,
                    discarded * 1e3,
                    a * 1e3,
                    c * 1e3,
                    rel_gap(c, a).unwrap_or(f64::NAN) * 100.0,
                    if driver_jit_cache_disabled() {
                        "DISABLED"
                    } else {
                        "enabled"
                    }
                );
            },
        );
    }

    /// An end-to-end walk of the intended call shape, with synthetic timings standing in for the
    /// device: open, sample, close, analyze, publish. If this shape is awkward, benches will not
    /// adopt it, and an instrument nobody adopts protects nothing.
    #[test]
    fn the_whole_intended_flow_composes_in_a_dozen_lines() {
        let mut round = open_ok("flow");
        let plan = TwinPlan::default();
        let mut t = 0u32;
        let s = run_twin(
            "flow",
            &plan,
            || {
                t += 1;
                vec![read("ms", 10.0 + (t % 3) as f64 * 0.01)]
            },
            || vec![read("ms", 8.0)],
        );
        round.close(Some(smi(2100.0)));
        let v = analyze(&s, plan.bar);
        let p = round.publish(&v, "ms").expect("this round resolves");
        assert!(p.ratio() < 0.81 && p.ratio() > 0.79);
        assert!(p.to_string().contains("floor"));
        assert!(round.header().contains("publish gate   : OPEN"));
    }
}
