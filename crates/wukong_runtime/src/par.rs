//! The parallel dispatch policy every `_parallel` runtime entry shares: **whether** to fork (a
//! work gate) and **how** to schedule the fork (the atomic claim queue).
//!
//! # Whether: gate on work, not on one dimension
//!
//! A fork-join on this crate's pools costs a roughly fixed amount of wall time — waking parked
//! workers (worst on a P+E hybrid, where a parked E-core is slow to arrive), one task hand-off per
//! split, and the join. A fork pays off only when the work it divides is several times that cost,
//! so the decision has to look at the *work*: rows × columns × a per-element cost weight. The old
//! per-family gates looked at one extent each (`rows < 8`, `cols < 256`, `m < 64`, …), which is
//! wrong in both directions at once: an 8 × 2 cross-entropy forked eight tasks over sixteen
//! elements while a 7 × 50257 one (a GPT-2 decode step's vocabulary) refused fifteen idle cores,
//! and a 63 × 1 000 000 GEMV ran serial while a 64 × 1 one forked. No value of a single-dimension
//! constant can be right for both corners, so no such constant could be calibrated either.
//!
//! [`fork_rows`] / [`fork_units`] replace them with one rule: fork iff there are at least two
//! independent units to split (rows, column stripes, channels), the weighted work reaches
//! [`par_min_work`], and the unified pool has a second worker (the width-1 serial fast path).
//! Work is counted in **stream element-ops** — one element read and combined with an add, a
//! compare or a multiply-add — with three weights:
//!
//! * [`W_STREAM`] = 1 for single-pass maps, copies, scans and memory-bound MACs (GEMV, gathers,
//!   column reductions, argmax);
//! * [`W_ROWPASS`] = 4 for row normalizations and their backward passes, which read a row two or
//!   three times and reduce it in between;
//! * [`W_TRANSCENDENTAL`] = 16 for rows that evaluate an `exp`/`log`/`sin` per element (softmax,
//!   log-softmax, cross-entropy, KL, entropy, RoPE).
//!
//! The default threshold (2^18 stream element-ops) and the transcendental weight are anchored to the
//! crate's two existing 1-D crossovers rather than invented: the streaming-map gate
//! (`WUKONG_VELEM_PAR_MIN`, ~256 Ki elements) and the activation gate (`VMATH_PAR_MIN`, 16 Ki
//! elements of transcendental work — the same 2^18 at weight 16). Like those, it is a starting point
//! for a measurement, not a measurement: `WUKONG_PAR_MIN_WORK` (a positive decimal, read once)
//! overrides it so the crossover is sweepable adjacent-run without a rebuild.
//!
//! The gate is **scheduling-only**. Every entry that consults it tail-calls a serial sibling that is
//! bit-identical to its parallel form by the crate's standing law (rows/columns/channels are
//! independent; no cross-unit combine), so no setting can change a result. Under `cfg(test)` the
//! work threshold is zero, so the crate's own serial == parallel tests keep forking at every shape
//! with two or more units — including the small shapes the production gate now keeps serial.
//!
//! # How: the claim queue
//!
//! [`claim_chunks`] is the self-scheduling loop that A/B-won for both `wukong_parallel_for` and
//! the GEMM's 2D block grid on this asymmetric 6P+8E+2LPE part: workers claim granules from one
//! cache-line-isolated atomic counter, so a P-core that finishes early pulls more work instead of
//! idling while an E-core straggles. It is shared here instead of re-implemented per kernel.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Cost weight of a single-pass element (map, copy, scan, compare, memory-bound MAC).
pub(crate) const W_STREAM: usize = 1;
/// Cost weight of a row-normalization element (two or three passes with a reduction between).
pub(crate) const W_ROWPASS: usize = 4;
/// Cost weight of an element that evaluates a transcendental (`exp`/`log`/`sin`/`cos`).
pub(crate) const W_TRANSCENDENTAL: usize = 16;

/// Default minimum weighted work for a fork: 2^18 stream element-ops (see the module docs).
const DEFAULT_PAR_MIN_WORK: usize = 1 << 18;

/// int8 multiply-accumulates per stream element-op: a VNNI `vpdpbusd` retires 32 u8×i8 MACs, and
/// the packing-free int8 kernel does no other per-MAC work, so one MAC is far cheaper than a
/// streamed f32 element. 16 puts the int8 GEMM crossover at 4 Mi MACs — half the f32 GEMM's
/// measured 2^23-MAC gate (`gemm.rs`'s `PAR_MIN_MACS`), whose parallel path also pays packing.
pub(crate) const I8_MACS_PER_WORK: usize = 16;

/// `WUKONG_PAR_MIN_WORK`: a positive ASCII decimal, or `None`.
fn parse_min_work(raw: Option<&str>) -> Option<usize> {
    let s = raw?;
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<usize>().ok().filter(|&v| v > 0)
}

/// The minimum weighted work for a `_parallel` entry to fork. Zero under `cfg(test)` (see the
/// module docs), else `WUKONG_PAR_MIN_WORK` or the default, read once.
pub(crate) fn par_min_work() -> usize {
    if cfg!(test) {
        return 0;
    }
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        parse_min_work(std::env::var("WUKONG_PAR_MIN_WORK").ok().as_deref())
            .unwrap_or(DEFAULT_PAR_MIN_WORK)
    })
}

/// The pure decision. `width` is evaluated last and only when the cheap tests pass, because
/// reading the pool width provisions the pools.
fn decide(units: usize, work: usize, min_work: usize, width: impl FnOnce() -> usize) -> bool {
    units >= 2 && work >= min_work && width() > 1
}

/// Should a `_parallel` entry split `units` independent units carrying `work` weighted element-ops
/// across cores? `false` means: tail-call the serial sibling (bit-identical).
pub(crate) fn fork_units(units: usize, work: usize) -> bool {
    decide(units, work, par_min_work(), crate::wuk_pool_width)
}

/// [`fork_units`] for a row-batched kernel: `rows` independent rows of `cols` elements, each
/// element costing `weight` stream element-ops.
pub(crate) fn fork_rows(rows: usize, cols: usize, weight: usize) -> bool {
    fork_units(rows, rows.saturating_mul(cols).saturating_mul(weight))
}

/// Run `body(start, end)` over disjoint spans that exactly cover `[0, n)`, on the unified kernel
/// pool ([`crate::run_on_wuk_pool`], so the spans run on 16 MiB worker stacks), returning once every
/// span has completed. Each index is handed to exactly one span.
///
/// * `dynamic = true`: spans are claimed from a shared atomic counter in granules of
///   `max(ceil(n / (workers·8)), min_granule)` — the P/E-straggle-absorbing schedule.
/// * `dynamic = false`: one contiguous span of `max(ceil(n / workers), min_granule)` per worker.
///
/// With one worker, or when the whole range fits one granule, `body(0, n)` runs once with no fork:
/// the serial fast path. Span boundaries depend on the worker count and claim order, so `body` must
/// be index-independent — the contract every caller already states (bits never depend on which span
/// ran an index).
pub(crate) fn claim_chunks<F>(n: usize, min_granule: usize, dynamic: bool, body: F)
where
    F: Fn(usize, usize) + Sync + Send,
{
    if n == 0 {
        return;
    }
    crate::run_on_wuk_pool(move || claim_on_current_pool(n, min_granule, dynamic, &body));
}

/// [`claim_chunks`]'s schedule on the pool the CALLER is already running on (its worker count, its
/// registry) instead of hopping to the unified kernel pool first. This is the claim loop itself;
/// [`claim_chunks`] is this plus [`crate::run_on_wuk_pool`]. [`for_each_unit`] and [`map_units`]
/// use it so the `WUKONG_KERNEL_DYN` A/B swaps the *schedule* only, never the pool.
fn claim_on_current_pool<F>(n: usize, min_granule: usize, dynamic: bool, body: &F)
where
    F: Fn(usize, usize) + Sync + Send,
{
    use rayon::prelude::*;
    if n == 0 {
        return;
    }
    // Read the worker count INSIDE the running context so it reflects the pool that actually runs
    // the spans (the private physical-core pool under unification).
    let workers = rayon::current_num_threads().max(1).min(n);
    let min_granule = min_granule.max(1);
    let g = if dynamic {
        n.div_ceil(workers.saturating_mul(8)).max(min_granule)
    } else {
        n.div_ceil(workers).max(min_granule)
    };
    let ntasks = n.div_ceil(g);
    if workers <= 1 || ntasks <= 1 {
        body(0, n);
        return;
    }
    if dynamic {
        // Cache-line-isolated (128 B: line + adjacent-line-prefetch pair) so the one shared
        // write bounces only for the claims themselves.
        #[repr(align(128))]
        struct Claims(AtomicUsize);
        let claims = Claims(AtomicUsize::new(0));
        let claims = &claims;
        (0..workers.min(ntasks)).into_par_iter().for_each(|_| loop {
            // Relaxed suffices: the counter only hands out unique granules; the fork-join's
            // join publishes every write the body made.
            let c = claims.0.fetch_add(1, Ordering::Relaxed);
            if c >= ntasks {
                break;
            }
            let start = c * g;
            body(start, (start + g).min(n));
        });
    } else {
        (0..ntasks).into_par_iter().for_each(|t| {
            let start = t * g;
            body(start, (start + g).min(n));
        });
    }
}

/// Kernel claim-queue instrument (`WUKONG_KERNEL_DYN=1`, read once, **default OFF**): route every
/// `_parallel` kernel split -- the row/column/channel/chunk maps that call [`for_each_row`] /
/// [`for_each_unit`] and the fixed-chunk reductions that call [`map_units`] -- through the atomic
/// claim queue instead of rayon's adaptive per-index split. The claim queue A/B-won for the GEMM's
/// coarse, heterogeneous 2D blocks and for `@parallel` regions; whether it also wins on uniform
/// rows, where rayon's splitter already balances, is unmeasured -- so it ships as the adjacent-run
/// A/B knob, not as the default. Scheduling-only: every unit is independent (rows, stripes,
/// channels, fixed `VCHUNK`/`VMATH_CHUNK` spans) or, for a reduction, a fixed-`RCHUNK` partial
/// stored by its chunk index and folded in index order afterwards, so which worker ran a unit never
/// changes a bit (P9.15). Default OFF runs exactly the pre-P9.15 rayon spelling, so default bits and
/// default timing are unchanged.
fn kernel_dyn() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("WUKONG_KERNEL_DYN").is_ok_and(|v| v == "1"))
}

/// Map `row(i)` over `i in 0..rows` on the unified kernel pool, each index exactly once, returning
/// when all have run: rayon's per-index split by default, the claim queue under
/// `WUKONG_KERNEL_DYN=1`. The caller has already decided to fork ([`fork_rows`]).
pub(crate) fn for_each_row<F>(rows: usize, row: F)
where
    F: Fn(usize) + Sync + Send,
{
    if kernel_dyn() {
        claim_chunks(rows, 1, true, |lo, hi| (lo..hi).for_each(&row));
    } else {
        crate::run_on_wuk_pool(|| {
            use rayon::prelude::*;
            (0..rows).into_par_iter().for_each(&row);
        });
    }
}

/// Map `unit(i)` over `i in 0..n`, each index exactly once, returning when all have run -- on the
/// pool the CALLER is running on. That is the global pool for a top-level call (after
/// [`crate::ensure_global_pool`], which this provisions first: a `_parallel` kernel can be the
/// process's first rayon touch), the enclosing pool for a call from inside an outlined `@parallel`
/// region or a GEMM pack phase, and the unified pool inside an explicit
/// [`crate::run_on_wuk_pool`]. Default: rayon's adaptive split, i.e. exactly the bare
/// `(0..n).into_par_iter().for_each(unit)` every call site spelled before P9.15. Under
/// `WUKONG_KERNEL_DYN=1`: the atomic claim queue on that same pool, so the A/B compares schedules
/// on one pool. The caller has already decided to fork (a [`fork_units`]/[`fork_rows`] gate or its
/// own crossover).
pub(crate) fn for_each_unit<F>(n: usize, unit: F)
where
    F: Fn(usize) + Sync + Send,
{
    for_each_unit_sched(n, kernel_dyn(), unit);
}

/// [`for_each_unit`] with the scheduler chosen explicitly (the env read is once per process, so the
/// tests drive both arms through here).
fn for_each_unit_sched<F>(n: usize, dynamic: bool, unit: F)
where
    F: Fn(usize) + Sync + Send,
{
    crate::ensure_global_pool();
    if dynamic {
        claim_on_current_pool(n, 1, true, &|lo: usize, hi: usize| (lo..hi).for_each(&unit));
    } else {
        use rayon::prelude::*;
        (0..n).into_par_iter().for_each(unit);
    }
}

/// `(0..n).map(unit).collect()` with the `unit` calls run in parallel on the caller's pool -- the
/// fixed-chunk reductions' partial map. Element `i` of the result is `unit(i)` whatever scheduled
/// it: rayon's indexed `collect` by default (the pre-P9.15 spelling), and under
/// `WUKONG_KERNEL_DYN=1` one slot per chunk index filled by the claim queue. The caller folds the
/// partials in index order, so serial == parallel stays bit-exact under either scheduler.
pub(crate) fn map_units<T, F>(n: usize, unit: F) -> Vec<T>
where
    T: Send + Sync,
    F: Fn(usize) -> T + Sync + Send,
{
    map_units_sched(n, kernel_dyn(), unit)
}

/// [`map_units`] with the scheduler chosen explicitly (see [`for_each_unit_sched`]).
fn map_units_sched<T, F>(n: usize, dynamic: bool, unit: F) -> Vec<T>
where
    T: Send + Sync,
    F: Fn(usize) -> T + Sync + Send,
{
    if dynamic {
        let slots: Vec<OnceLock<T>> = (0..n).map(|_| OnceLock::new()).collect();
        claim_on_current_pool(n, 1, true, &|lo: usize, hi: usize| {
            for (i, slot) in slots.iter().enumerate().take(hi).skip(lo) {
                // Each index is claimed exactly once, so every `set` lands on an empty slot.
                let _ = slot.set(unit(i));
            }
        });
        slots
            .into_iter()
            .map(|s| {
                s.into_inner()
                    .expect("the claim queue runs every chunk index exactly once")
            })
            .collect()
    } else {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(unit).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The production rule at the default threshold, pinned at both corners of every family the
    /// old single-dimension gates got wrong (P12.12, P12.20, P13.8).
    #[test]
    fn par_gate_keys_on_work_not_rows() {
        let min = DEFAULT_PAR_MIN_WORK;
        let wide = || 16usize;
        let rows = |r: usize, c: usize, w: usize| decide(r, r * c * w, min, wide);
        // Tall-thin: 8 rows x 2 columns of cross-entropy used to fork eight tasks over 16 elements.
        assert!(!rows(8, 2, W_TRANSCENDENTAL));
        // Short-wide: a 7-row x 50257-vocabulary loss used to run single-core.
        assert!(rows(7, 50_257, W_TRANSCENDENTAL));
        // GEMV: 63 x 1e6 MACs used to stay serial while 64 x 1 forked.
        assert!(rows(63, 1_000_000, W_STREAM));
        assert!(!rows(64, 1, W_STREAM));
        // A single-token LayerNorm in a `@parallel` function: nothing to split.
        assert!(!rows(1, 4096, W_ROWPASS));
        // Two rows of a small norm used to fork rayon per row; a real batch still forks.
        assert!(!rows(2, 768, W_ROWPASS));
        assert!(rows(512, 768, W_ROWPASS));
        // The int8 GEMM: a 4-row decode GEMM no longer forks per row; a real one does.
        let int8 =
            |m: usize, k: usize, n: usize| decide(m, m * k * n / I8_MACS_PER_WORK, min, wide);
        assert!(!int8(4, 64, 64));
        assert!(int8(64, 768, 768));
        // Work alone never forks a single unit, and a width-1 pool never forks at all.
        assert!(!decide(1, usize::MAX, min, wide));
        assert!(!decide(1024, usize::MAX, min, || 1));
        // The width probe is not even consulted when the cheap tests fail.
        assert!(!decide(1, usize::MAX, min, || panic!(
            "width read for one unit"
        )));
        assert!(!decide(8, 1, min, || panic!(
            "width read below the work threshold"
        )));
    }

    #[test]
    fn min_work_override_parser() {
        assert_eq!(parse_min_work(Some("4096")), Some(4096));
        assert_eq!(parse_min_work(Some("0004096")), Some(4096));
        for bad in [
            None,
            Some(""),
            Some("0"),
            Some("-1"),
            Some("+5"),
            Some(" 5"),
            Some("1k"),
        ] {
            assert_eq!(parse_min_work(bad), None, "{bad:?}");
        }
        // The unit-test build forks every splittable shape so serial == parallel tests stay real.
        assert_eq!(par_min_work(), 0);
    }

    #[test]
    fn claim_chunks_covers_every_index_exactly_once() {
        for &dynamic in &[true, false] {
            for &(n, granule) in &[
                (1usize, 1usize),
                (2, 1),
                (7, 1),
                (8, 3),
                (1000, 1),
                (1000, 64),
                (1000, 5000),
                (100_003, 1),
            ] {
                let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                let spans = AtomicUsize::new(0);
                claim_chunks(n, granule, dynamic, |s, e| {
                    assert!(s < e && e <= n, "bad span {s}..{e} of {n}");
                    if e < n {
                        // Every span but the last is at least the requested granule.
                        assert!(
                            e - s >= granule.min(n),
                            "span {s}..{e} under granule {granule}"
                        );
                    }
                    spans.fetch_add(1, Ordering::Relaxed);
                    for h in &hits[s..e] {
                        h.fetch_add(1, Ordering::Relaxed);
                    }
                });
                assert!(
                    hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
                    "n={n} granule={granule} dynamic={dynamic}: an index ran zero or two times"
                );
                if granule >= n {
                    assert_eq!(
                        spans.load(Ordering::Relaxed),
                        1,
                        "one granule must not fork"
                    );
                }
            }
        }
        // n == 0 never calls the body.
        claim_chunks(0, 1, true, |_, _| panic!("empty range ran a span"));
    }

    /// P9.15: both schedulers of the shared kernel maps visit every unit exactly once, and the
    /// partial map returns `unit(i)` at index `i` -- the property the fixed-`RCHUNK` reductions'
    /// index-order fold rests on. Driven through the `_sched` forms so one process covers both arms
    /// regardless of `WUKONG_KERNEL_DYN`.
    #[test]
    fn for_each_unit_and_map_units_cover_every_index_under_both_schedulers() {
        for dynamic in [false, true] {
            for n in [0usize, 1, 2, 7, 64, 1000, 4099] {
                let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                for_each_unit_sched(n, dynamic, |i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
                assert!(
                    hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
                    "for_each_unit n={n} dynamic={dynamic}"
                );
                let got = map_units_sched(n, dynamic, |i| (i as u64) * 3 + 1);
                let want: Vec<u64> = (0..n as u64).map(|i| i * 3 + 1).collect();
                assert_eq!(got, want, "map_units n={n} dynamic={dynamic}");
            }
        }
        // A float fold over map_units partials is bit-identical under both schedulers: the slot
        // order, not the completion order, decides the fold.
        let partial = |c: usize| ((c as f32) * 0.37).sin() * 1e3 + 1e-3;
        let fold = |v: Vec<f32>| v.into_iter().fold(0.0f32, |a, b| a + b).to_bits();
        let a = fold(map_units_sched(513, false, partial));
        let b = fold(map_units_sched(513, true, partial));
        let serial = (0..513).map(partial).fold(0.0f32, |a, b| a + b).to_bits();
        assert_eq!(a, serial);
        assert_eq!(b, serial);
    }

    #[test]
    fn for_each_row_visits_every_row_once() {
        for rows in [0usize, 1, 2, 7, 1000] {
            let hits: Vec<AtomicUsize> = (0..rows).map(|_| AtomicUsize::new(0)).collect();
            for_each_row(rows, |r| {
                hits[r].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
                "rows={rows}"
            );
        }
    }
}
