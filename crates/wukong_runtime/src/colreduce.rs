//! Column reductions along the **outer** (batch / row) axis of a `[rows, cols]` row-major matrix,
//! producing a per-column `[cols]` result. Eight kinds ([`ColKind`]), one `wukong_col*_f32[_parallel]`
//! entry pair each: the **sum** `out[j] = Σ_i x[i,j]` (bias gradient `db = Σ_batch dY`, batch sum,
//! reduce-along-axis-0), **max** / **min** / **maxabs** (per-channel statistics for quantization,
//! axis-0 max/min pooling), **sumsq**, and the three that add a scalar finalize over the folded column
//! — **mean** (`/rows`), **l2** (`sqrt`), **rms** (`sqrt(_/rows)`), applied per column by
//! [`colreduce_finalize_avx2`] (8 columns per `vdivps`/`vsqrtps`) or its scalar reference
//! [`colreduce_finalize`], and so independent of the parallel column split.
//!
//! The naive spelling `for j { for i { s ⊕= x[i*N+j] } }` reads `x` with stride `N` — a *strided
//! reduction* gcc/rustc do **not** vectorize (verified: scalar `vaddss`/`vmaxss`, no packed
//! `vaddps`/`vmaxps`, at `-O3 -march=native`). These kernels instead stream `x` **row-major**, so they
//! are SIMD *and* cache-friendly.
//!
//! Row-major streaming alone is not enough, though: it is also what a *competent* C peer writes
//! (`for i { for j { out[j] ⊕= x[i*N+j] } }` with `__restrict__`), and gcc vectorizes that. Against
//! that peer the earlier form of these kernels — 8 columns per step, accumulator re-read and
//! re-written from `out[]` on **every** row — lost. The traversal is now **[`COL_RB`] rows x 32
//! columns per tile with the four accumulators resident in YMM registers**, which divides the `out[]`
//! load/store traffic by `COL_RB` and keeps `COL_RB` independent `x` row streams in flight (the
//! memory-level parallelism a single sequential stream cannot reach). The first block seeds its
//! registers directly, and the final partial block keeps its accumulators resident too. See
//! [`colreduce_avx2`].
//!
//! The fold order is **i-ascending per column** — identical to the scalar nest — in the AVX2 and
//! scalar paths, across the row blocking, and across serial/parallel (disjoint column stripes, no
//! cross-stripe combine), so each kernel is its own bit-exact oracle (the interpreter marshals the
//! serial form; the differential gate compares interp vs native, both folding the same
//! `_mm256_add_ps`/`_mm256_max_ps`/`_mm256_min_ps` lane tree).
//!
//! ONE CAVEAT, AND ONLY ONE: **the NaN payload of the additive kinds is not part of the contract.**
//! Every value that is not a NaN — finite, ±0.0, ±inf — is bit-exact across the AVX2 path, the
//! scalar twin and the column stripes, and for `Max`/`Min`/`MaxAbs` so is the NaN payload. But
//! `Sum`/`Mean`/`SumSq`/`L2`/`Rms` fold with `+`, and which of two NaN payloads survives `addps`
//! depends on which operand the compiler made SRC1. `sfold_add` writes `a + v`, yet `fadd` is
//! commutative in LLVM IR, so the optimizer may emit `v + a`: at `-O3` it auto-vectorizes the twin's
//! inner column loop and the vectorized `addps` there *is* commuted (the scalar epilogue is not,
//! which is why a 37-column case diverged from column 32 on and no lower). IEEE-754 leaves payload
//! propagation to the implementation, so both answers are correct and neither is worth pinning at
//! the cost of blocking vectorization of the no-AVX2 fallback. Tests must therefore compare the
//! additive kinds with `tests::bits_nan_collapsed`, never by payload — see its comment.

/// Which column reduction to fold: `Sum` seeds `0` and folds rows `[0, rows)`; `Max`/`Min`/`MaxAbs`
/// seed the **first row** (`x[0,j]`, or `|x[0,j]|` for `MaxAbs`) and fold rows `[1, rows)` (idempotent,
/// so equivalent to folding from 0). `MaxAbs` is the per-channel symmetric-quantization scale
/// `max_i |x[i,j]|`.
///
/// `Mean`/`SumSq`/`L2`/`Rms` are the per-channel **statistics** family — the BatchNorm running mean,
/// the per-channel energy and L2/RMS column norms. They fold like `Sum` (over `x` for `Mean`, over
/// `x²` for `SumSq`/`L2`/`Rms`, both additive from `0`), then apply a per-column **finalize** after all
/// rows are folded: `Mean` divides by `rows`, `L2` takes `sqrt`, `Rms` takes `sqrt(_/rows)`. The
/// finalize touches each output column exactly once (after its full i-ascending fold), and the parallel
/// stripes are column-disjoint, so serial == parallel bit-for-bit just like the base folds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColKind {
    Sum,
    Max,
    Min,
    MaxAbs,
    Mean,
    SumSq,
    L2,
    Rms,
}

/// Rows folded per pass with the column accumulators held in **registers** (see [`colreduce_avx2`]).
/// Four is what measured best on this box across both the L2-resident (4 MiB) and the L3-resident
/// (16 MiB) shapes: it cuts the `out[]` load/store traffic 4x *and* keeps four independent `x` row
/// streams in flight (the memory-level parallelism that lifts a single core off its one-stream
/// bandwidth ceiling). Two streams left bandwidth on the table; eight or more started to thrash the
/// L1 set that every `cols*4`-strided row lands in.
const COL_RB: usize = 4;

/// Apply the per-column finalize for the statistics kinds over `out[j0..j1]` (after all rows folded):
/// `Mean` → `/rows`, `L2` → `sqrt`, `Rms` → `sqrt(_/rows)`. `Sum`/`SumSq`/`Max`/`Min`/`MaxAbs` are
/// no-ops. **Scalar**, one column per iteration: this is the reference form, the finalize of the
/// no-AVX2 twin [`colreduce_scalar`], and the tail of the vector form [`colreduce_finalize_avx2`] —
/// which calls literally this function for its `(j1-j0) % 8` remainder rather than restating the
/// arithmetic. Independent of the column split, so it does not perturb serial == parallel.
///
/// # Safety
/// `out` valid for `[j0, j1)` f32; `rows >= 1`.
#[inline]
unsafe fn colreduce_finalize(out: *mut f32, j0: usize, j1: usize, rows: usize, kind: ColKind) {
    match kind {
        ColKind::Mean => {
            let inv = rows as f32;
            for j in j0..j1 {
                *out.add(j) /= inv;
            }
        }
        ColKind::L2 => {
            for j in j0..j1 {
                *out.add(j) = (*out.add(j)).sqrt();
            }
        }
        ColKind::Rms => {
            let inv = rows as f32;
            for j in j0..j1 {
                *out.add(j) = (*out.add(j) / inv).sqrt();
            }
        }
        _ => {}
    }
}

/// AVX2 form of [`colreduce_finalize`]: eight columns per `vdivps`/`vsqrtps` instead of one per
/// `vdivss`/`vsqrtss`, with the `(j1-j0) % 8` remainder handed straight to the scalar function.
///
/// WHY THIS EXISTS. LLVM does **not** vectorize the scalar finalize even when it is inlined into a
/// `#[target_feature(enable = "avx2")]` caller: in the release assembly of [`colreduce_avx2`] the
/// three statistics arms compile to `vmovss / vdivss / vsqrtss / vmovss / incq / cmpq / jne`, one
/// column per iteration, and the whole function contains **zero** `vsqrtps` and **zero** `vdivps`.
/// (Verified by `cargo rustc --release -p wukong_runtime --lib -- --emit asm`; the fold loops above
/// are packed, only the finalize was left scalar.) On this part the ymm divider is pipelined at
/// about 5 (`vdivps`) and 6 (`vsqrtps`) cycles per instruction against about 3 each for the scalar
/// forms, so `Rms` costs roughly 6 divider-cycles per column scalar versus 11 per **eight** columns
/// packed — about 4x on the finalize. It is a small share of a column reduction (about 3% of
/// `colrms` at 1024x1024, less at more rows, since the finalize is O(cols) against an O(rows*cols)
/// fold), which is precisely why nothing timed it and the scalar loop survived.
///
/// BIT-EXACTNESS. IEEE-754 requires `division` and `squareRoot` to be **correctly rounded**, and
/// `DIVPS`/`SQRTPS` apply exactly the same correctly-rounded operation lane-wise, under the same
/// MXCSR rounding mode and DAZ/FTZ state, that `DIVSS`/`SQRTSS` apply to one element — no cross-lane
/// interaction and no reassociation is possible in a unary/one-divisor map. The NaN payload is
/// determined here too (unlike the additive folds, whose SRC1 the optimizer may commute): the
/// operand order `v / rows` is fixed in both forms, and both instructions return the quieted source
/// NaN. So this is a strict bit-for-bit substitution and is tested as one, on NaN (quiet and
/// signalling), +/-0, +/-inf, denormals and negative inputs to `sqrt`, at every tail offset —
/// see `finalize_avx2_matches_scalar_bit_for_bit`.
///
/// # Safety
/// `out` valid for `[j0, j1)` f32; `rows >= 1`; AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn colreduce_finalize_avx2(out: *mut f32, j0: usize, j1: usize, rows: usize, kind: ColKind) {
    use std::arch::x86_64::*;
    let mut j = j0;
    // One 8-wide map per statistics kind; the non-statistics kinds leave `j == j0` and fall through
    // to the scalar no-op below, so there is no separate early return to keep in sync.
    match kind {
        ColKind::Mean => {
            let d = _mm256_set1_ps(rows as f32);
            while j + 8 <= j1 {
                let p = out.add(j);
                _mm256_storeu_ps(p, _mm256_div_ps(_mm256_loadu_ps(p), d));
                j += 8;
            }
        }
        ColKind::L2 => {
            while j + 8 <= j1 {
                let p = out.add(j);
                _mm256_storeu_ps(p, _mm256_sqrt_ps(_mm256_loadu_ps(p)));
                j += 8;
            }
        }
        ColKind::Rms => {
            let d = _mm256_set1_ps(rows as f32);
            while j + 8 <= j1 {
                let p = out.add(j);
                _mm256_storeu_ps(p, _mm256_sqrt_ps(_mm256_div_ps(_mm256_loadu_ps(p), d)));
                j += 8;
            }
        }
        _ => {}
    }
    // The `< 8` remainder, and every non-statistics kind, through the scalar reference itself.
    colreduce_finalize(out, j, j1, rows, kind);
}

/// The five per-element folds, as ordinary (non-`target_feature`) `#[inline(always)]` functions so
/// that the AVX2 kernel's scalar tails, the scalar twin [`colreduce_scalar`] and the unit tests all
/// use **literally the same code** — the fold order is the correctness contract of this file, and a
/// second hand-written copy is exactly how a twin drifts. `max`/`min` are spelled `(a > v) ? a : v` /
/// `(a < v) ? a : v` deliberately, NOT `f32::max`/`f32::min`: that is what `_mm256_max_ps` /
/// `_mm256_min_ps` compute (the second operand wins a tie and any NaN comparison), so the vector and
/// scalar paths agree on NaN, ±0 and ties. Those two are also the folds whose operand order the
/// optimizer **cannot** disturb: a select over `a > v` is not commutative, so LLVM must keep `a` as
/// the `maxps`/`minps` destination (verified in the `-O3` asm of the auto-vectorized twin), and the
/// payload is fully determined.
///
/// `sfold_add`/`sfold_sq` get no such guarantee: `fadd` *is* commutative in LLVM IR, so the order
/// written here fixes the value but not which operand becomes SRC1 — and therefore not which NaN
/// payload survives. See the module header; nothing may depend on it.
#[inline(always)]
fn sfold_add(a: f32, v: f32) -> f32 {
    a + v
}
/// `a + v*v` as a separate multiply then add (never an FMA) — see [`colreduce_avx2`].
#[inline(always)]
fn sfold_sq(a: f32, v: f32) -> f32 {
    a + v * v
}
#[inline(always)]
fn sfold_max(a: f32, v: f32) -> f32 {
    if a > v {
        a
    } else {
        v
    }
}
#[inline(always)]
fn sfold_min(a: f32, v: f32) -> f32 {
    if a < v {
        a
    } else {
        v
    }
}
#[inline(always)]
fn sfold_maxabs(a: f32, v: f32) -> f32 {
    let v = v.abs();
    if a > v {
        a
    } else {
        v
    }
}

/// Fold rows `[0, rows)` of `x` into the column range `[j0, j1)` of `out` (`out[j] = ⊕_i x[i*cols+j]`),
/// AVX2. Seeds the range (0 for `Sum`, the first row for `Max`/`Min`), then streams `x` row-major
/// folding into `out`. The per-`kind` inner loop is selected **once per call** (no per-element
/// branch); each fold is the exact `_mm256_{add,max,min}_ps` whose scalar twin is `sfold_*` above.
///
/// TRAVERSAL. The tile is **[`COL_RB`] rows x 32 columns**: four `__m256` accumulators are loaded
/// from `out[j..j+32]`, folded against `COL_RB` consecutive rows *while resident in registers*, and
/// stored back once. The earlier form re-read and re-wrote `out[j]` for **every** row, so a
/// `rows x cols` fold cost `rows*cols` accumulator loads plus `rows*cols` accumulator stores on top
/// of the unavoidable `x` read; blocking divides both by `COL_RB` and, just as importantly, keeps
/// `COL_RB` independent `x` row streams in flight, which is what lifts a single core off its
/// one-stream memory-bandwidth ceiling. A competent row-outer C peer (`for i { for j { out[j] +=
/// x[i*N+j] } }`, `__restrict__`, gcc `-O3 -march=native`) is exactly the un-blocked form, and it is
/// what this kernel used to lose to.
///
/// BIT-EXACTNESS IS UNAFFECTED, which is the only reason the reshape is admissible: for a fixed
/// column `j` the sequence is still `out[j] <- ((out[j] (+) x[i,j]) (+) x[i+1,j]) ...` with `i`
/// strictly ascending and no reassociation — the tile only changes *where the running accumulator
/// lives*, never the order it is folded in. The first block seeds registers directly, eliminating
/// the separate output initialization and its first reload. The final partial block also folds
/// while register-resident, continuing the same ascending sequence. The `(j1-j0) % 32` trailing
/// columns fold 8-wide and then scalar inside each row block.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32; `j0 <= j1 <= cols`; AVX2 available;
/// `rows >= 1`; `x` and `out` must not overlap.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn colreduce_avx2(
    x: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
    j0: usize,
    j1: usize,
    kind: ColKind,
) {
    use std::arch::x86_64::*;
    // Seed: the additive folds (Sum/Mean and the square folds SumSq/L2/Rms) -> 0; Max/Min -> first row
    // x[0, j]; MaxAbs -> |x[0, j]| (so the max/min fold starts at row 1).
    let i0 = match kind {
        ColKind::Sum | ColKind::Mean | ColKind::SumSq | ColKind::L2 | ColKind::Rms => 0,
        ColKind::Max | ColKind::Min | ColKind::MaxAbs => 1,
    };
    let sign = _mm256_set1_ps(-0.0); // for MaxAbs: clear the sign bit with andnot

    // The tiled fold, monomorphic per kind: the `match` below is hoisted out of every loop and each
    // arm pastes its own vector fold (`|a, v| …`) and scalar fold (`sfold_…`) into ONE traversal.
    // Writing the traversal once is deliberate — five hand-copied nests are five chances for one of
    // them to lose the ascending-`i` order that the whole file's bit-exactness rests on.
    macro_rules! fold_all {
        (|$a:ident, $v:ident| $vfold:expr, $sfold:ident, $vseed:expr, $sseed:expr) => {{
            let mut i = i0;
            // The first block seeds registers directly, avoiding a separate initialization pass
            // and the subsequent accumulator reload. `$first` is a macro-time constant, so the
            // steady-state tile never branches on whether its output has been initialized.
            macro_rules! fold_block {
                ($first:expr, $end:expr) => {{
                    let mut j = j0;
                    while j + 32 <= j1 {
                        let mut a0 = if $first {
                            ($vseed)(x.add(j))
                        } else {
                            _mm256_loadu_ps(out.add(j))
                        };
                        let mut a1 = if $first {
                            ($vseed)(x.add(j + 8))
                        } else {
                            _mm256_loadu_ps(out.add(j + 8))
                        };
                        let mut a2 = if $first {
                            ($vseed)(x.add(j + 16))
                        } else {
                            _mm256_loadu_ps(out.add(j + 16))
                        };
                        let mut a3 = if $first {
                            ($vseed)(x.add(j + 24))
                        } else {
                            _mm256_loadu_ps(out.add(j + 24))
                        };
                        let mut k = i;
                        while k < $end {
                            let xr = x.add(k * cols + j);
                            a0 = {
                                let ($a, $v) = (a0, _mm256_loadu_ps(xr));
                                $vfold
                            };
                            a1 = {
                                let ($a, $v) = (a1, _mm256_loadu_ps(xr.add(8)));
                                $vfold
                            };
                            a2 = {
                                let ($a, $v) = (a2, _mm256_loadu_ps(xr.add(16)));
                                $vfold
                            };
                            a3 = {
                                let ($a, $v) = (a3, _mm256_loadu_ps(xr.add(24)));
                                $vfold
                            };
                            k += 1;
                        }
                        _mm256_storeu_ps(out.add(j), a0);
                        _mm256_storeu_ps(out.add(j + 8), a1);
                        _mm256_storeu_ps(out.add(j + 16), a2);
                        _mm256_storeu_ps(out.add(j + 24), a3);
                        j += 32;
                    }
                    // (j1-j0) % 32 trailing columns of this row block: 8-wide, then scalar.
                    while j + 8 <= j1 {
                        let mut acc = if $first {
                            ($vseed)(x.add(j))
                        } else {
                            _mm256_loadu_ps(out.add(j))
                        };
                        let mut k = i;
                        while k < $end {
                            acc = {
                                let ($a, $v) = (acc, _mm256_loadu_ps(x.add(k * cols + j)));
                                $vfold
                            };
                            k += 1;
                        }
                        _mm256_storeu_ps(out.add(j), acc);
                        j += 8;
                    }
                    while j < j1 {
                        let mut s = if $first {
                            ($sseed)(x.add(j))
                        } else {
                            *out.add(j)
                        };
                        let mut k = i;
                        while k < $end {
                            s = $sfold(s, *x.add(k * cols + j));
                            k += 1;
                        }
                        *out.add(j) = s;
                        j += 1;
                    }
                }};
            }
            // Even a seed-only Max/Min call (one row) needs the first block's output stores.
            let first_end = (i + COL_RB).min(rows);
            fold_block!(true, first_end);
            i = first_end;
            while i + COL_RB <= rows {
                let end = i + COL_RB;
                fold_block!(false, end);
                i = end;
            }
            // Keep the last one to three rows in registers too, in the same ascending order.
            if i < rows {
                fold_block!(false, rows);
            }
        }};
    }
    match kind {
        // Σ x[i,j] (Mean reuses the Sum fold, then divides in the finalize).
        ColKind::Sum | ColKind::Mean => fold_all!(
            |a, v| _mm256_add_ps(a, v),
            sfold_add,
            |_: *const f32| _mm256_setzero_ps(),
            |_: *const f32| 0.0
        ),
        // Σ x[i,j]² (L2/Rms reuse this, then sqrt[/rows] in the finalize). The square is a separate
        // `mul` then `add` (NOT an FMA) so the scalar twin `a + v*v` matches it bit-for-bit (no `fma`
        // target feature assumed here, and one rounding model shared by both paths).
        ColKind::SumSq | ColKind::L2 | ColKind::Rms => {
            fold_all!(
                |a, v| _mm256_add_ps(a, _mm256_mul_ps(v, v)),
                sfold_sq,
                |_: *const f32| _mm256_setzero_ps(),
                |_: *const f32| 0.0
            )
        }
        ColKind::Max => fold_all!(
            |a, v| _mm256_max_ps(a, v),
            sfold_max,
            |p: *const f32| _mm256_loadu_ps(p),
            |p: *const f32| *p
        ),
        ColKind::Min => fold_all!(
            |a, v| _mm256_min_ps(a, v),
            sfold_min,
            |p: *const f32| _mm256_loadu_ps(p),
            |p: *const f32| *p
        ),
        // |v| = andnot(-0.0, v) (clear the sign bit), then fold by max — the same sign-mask abs the
        // RED_MAXABS reduction uses, so it agrees lane-for-lane with `sfold_maxabs`'s `v.abs()`.
        ColKind::MaxAbs => {
            fold_all!(
                |a, v| _mm256_max_ps(a, _mm256_andnot_ps(sign, v)),
                sfold_maxabs,
                |p: *const f32| _mm256_andnot_ps(sign, _mm256_loadu_ps(p)),
                |p: *const f32| (*p).abs()
            )
        }
    }
    // Per-column finalize for the statistics kinds (Mean -> /rows, L2 -> sqrt, Rms -> sqrt(/rows));
    // a no-op for Sum/SumSq/Max/Min/MaxAbs. Packed here and scalar in the twin, which is a bit-exact
    // substitution because divide and sqrt are correctly rounded and lane-independent — see
    // [`colreduce_finalize_avx2`], which LLVM would not produce from the scalar loop on its own.
    colreduce_finalize_avx2(out, j0, j1, rows, kind);
}

/// Scalar twin of [`colreduce_avx2`] (the no-AVX2 fallback and the bit-exact reference). Same
/// i-ascending per-column order and — literally — the same `sfold_*` fold functions the vector path's
/// tails call (`a > v ? a : v` == `_mm256_max_ps(a, v)`, `a < v ? a : v` == `_mm256_min_ps(a, v)`),
/// so it agrees with the AVX2 path lane-for-lane. This twin is deliberately left **un-blocked**: it
/// is the reference, and `out[j] <- fold(out[j], x[i,j])` for ascending `i` is the definition the
/// blocked vector path must reproduce.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32; `j0 <= j1 <= cols`.
unsafe fn colreduce_scalar(
    x: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
    j0: usize,
    j1: usize,
    kind: ColKind,
) {
    let i0 = match kind {
        ColKind::Sum | ColKind::Mean | ColKind::SumSq | ColKind::L2 | ColKind::Rms => {
            for j in j0..j1 {
                *out.add(j) = 0.0;
            }
            0
        }
        ColKind::Max | ColKind::Min => {
            for j in j0..j1 {
                *out.add(j) = *x.add(j);
            }
            1
        }
        ColKind::MaxAbs => {
            for j in j0..j1 {
                *out.add(j) = (*x.add(j)).abs();
            }
            1
        }
    };
    // The `match` is hoisted out of both loops (one dispatch per call, not one per element); each arm
    // pastes the same `sfold_*` the AVX2 path's scalar tails use.
    macro_rules! run {
        ($sfold:ident) => {{
            for i in i0..rows {
                let xr = x.add(i * cols);
                for j in j0..j1 {
                    *out.add(j) = $sfold(*out.add(j), *xr.add(j));
                }
            }
        }};
    }
    match kind {
        // `a + v*v` (a separate mul then add) matches the AVX2 `add(acc, mul(v,v))` exactly.
        ColKind::SumSq | ColKind::L2 | ColKind::Rms => run!(sfold_sq),
        ColKind::Sum | ColKind::Mean => run!(sfold_add),
        ColKind::Max => run!(sfold_max),
        ColKind::Min => run!(sfold_min),
        ColKind::MaxAbs => run!(sfold_maxabs),
    }
    colreduce_finalize(out, j0, j1, rows, kind);
}

/// Dispatch AVX2 vs scalar for the column range `[j0, j1)`.
///
/// # Safety
/// Operand-size contract of [`colreduce_avx2`].
#[inline]
unsafe fn colreduce_range(
    x: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
    j0: usize,
    j1: usize,
    kind: ColKind,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::isa::avx2() {
            colreduce_avx2(x, out, rows, cols, j0, j1, kind);
            return;
        }
    }
    colreduce_scalar(x, out, rows, cols, j0, j1, kind);
}

/// Multi-threaded `out[j] = ⊕_i x[i, j]`: the **columns** are split into disjoint stripes across cores
/// (each core folds its column range over all rows). Each stripe writes a disjoint slice of `out`, and
/// every column is folded in the same i-ascending order regardless of the split — so the result is
/// **bit-identical to the serial kernel** the interpreter marshals (no cross-stripe combine). Below
/// the shared work gate (`par::fork_units`, rows x cols) it runs serial.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32, non-overlapping.
#[inline]
unsafe fn colreduce_parallel(x: *const f32, out: *mut f32, rows: i64, cols: i64, kind: ColKind) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    // Work gate on rows x cols (units = 32-column stripes) + width-1 fast path; see `par`.
    if !crate::par::fork_units(c.div_ceil(32), r.saturating_mul(c)) {
        colreduce_range(x, out, r, c, 0, c, kind);
        return;
    }
    use rayon::prelude::*;
    // This entry can be the process's FIRST rayon touch, so it must configure the global pool before
    // forking (crate::ensure_global_pool's stated contract): otherwise rayon lazily builds its default
    // 2 MiB-stack registry here and the runtime's 16 MiB build_global silently loses the race for the
    // whole process, leaving later outlined @parallel region bodies (~1.5 MiB of privatized scratch at
    // S=512) on 2 MiB stacks. Configuration only -- the stripe split is unchanged, so the bits are too.
    crate::ensure_global_pool();
    // One stripe per core, each a multiple of 32 columns — the width of one [`colreduce_avx2`] tile,
    // so a stripe runs the blocked main loop rather than falling into its 8-wide tail; the last
    // stripe absorbs the remainder. The stripe width is a *partition* of the columns, and every
    // column is folded i-ascending over all rows whichever stripe owns it, so widening the stripe
    // cannot move a bit.
    let nthreads = rayon::current_num_threads().max(1);
    if nthreads == 1 {
        colreduce_range(x, out, r, c, 0, c, kind);
        return;
    }
    let per = (c.div_ceil(nthreads)).next_multiple_of(32).max(32);
    let nstripes = c.div_ceil(per);
    let (x_addr, out_addr) = (x as usize, out as usize);
    (0..nstripes).into_par_iter().for_each(|s| {
        let j0 = s * per;
        let j1 = (j0 + per).min(c);
        // SAFETY: disjoint out[] stripe per task; pointers re-derived from the captured addresses.
        unsafe {
            colreduce_range(
                x_addr as *const f32,
                out_addr as *mut f32,
                r,
                c,
                j0,
                j1,
                kind,
            );
        }
    });
}

/// `out[j] = Σ_i x[i, j]` over a `[rows, cols]` row-major matrix, single-threaded.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32, non-overlapping.
/// **Aliasing: all operands disjoint** — `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colsum_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    // P14.5: x valid for rows*cols, out for cols, out disjoint from x (documented all-disjoint).
    crate::safety::debug_extent_2d(x, rows, cols);
    crate::safety::debug_extent(out, cols);
    crate::safety::debug_disjoint(out, cols, x, rows.saturating_mul(cols));
    if rows <= 0 || cols <= 0 {
        return;
    }
    colreduce_range(
        x,
        out,
        rows as usize,
        cols as usize,
        0,
        cols as usize,
        ColKind::Sum,
    );
}

/// Multi-threaded `out[j] = Σ_i x[i, j]` (bit-identical to [`wukong_colsum_f32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_colsum_f32`].
/// **Aliasing:** that of [`wukong_colsum_f32`] — all operands disjoint: `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colsum_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    colreduce_parallel(x, out, rows, cols, ColKind::Sum);
}

/// `out[j] = max_i x[i, j]` over a `[rows, cols]` row-major matrix, single-threaded (per-channel max /
/// axis-0 max-pool). Seeds from the first row, so an empty `rows == 0` writes nothing.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32, non-overlapping.
/// **Aliasing: all operands disjoint** — `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colmax_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    colreduce_range(
        x,
        out,
        rows as usize,
        cols as usize,
        0,
        cols as usize,
        ColKind::Max,
    );
}

/// Multi-threaded `out[j] = max_i x[i, j]` (bit-identical to [`wukong_colmax_f32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_colmax_f32`].
/// **Aliasing:** that of [`wukong_colmax_f32`] — all operands disjoint: `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colmax_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    colreduce_parallel(x, out, rows, cols, ColKind::Max);
}

/// `out[j] = min_i x[i, j]` over a `[rows, cols]` row-major matrix, single-threaded (per-channel min /
/// axis-0 min-pool).
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32, non-overlapping.
/// **Aliasing: all operands disjoint** — `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colmin_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    colreduce_range(
        x,
        out,
        rows as usize,
        cols as usize,
        0,
        cols as usize,
        ColKind::Min,
    );
}

/// Multi-threaded `out[j] = min_i x[i, j]` (bit-identical to [`wukong_colmin_f32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_colmin_f32`].
/// **Aliasing:** that of [`wukong_colmin_f32`] — all operands disjoint: `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colmin_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    colreduce_parallel(x, out, rows, cols, ColKind::Min);
}

/// `out[j] = max_i |x[i, j]|` over a `[rows, cols]` row-major matrix, single-threaded — the per-channel
/// **symmetric-quantization scale** (the int8 weight/activation scale `s_j = amax_j / 127`).
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32, non-overlapping.
/// **Aliasing: all operands disjoint** — `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colmaxabs_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    colreduce_range(
        x,
        out,
        rows as usize,
        cols as usize,
        0,
        cols as usize,
        ColKind::MaxAbs,
    );
}

/// Multi-threaded `out[j] = max_i |x[i, j]|` (bit-identical to [`wukong_colmaxabs_f32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_colmaxabs_f32`].
/// **Aliasing:** that of [`wukong_colmaxabs_f32`] — all operands disjoint: `out` does not overlap
/// `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colmaxabs_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    colreduce_parallel(x, out, rows, cols, ColKind::MaxAbs);
}

/// `out[j] = (Σ_i x[i, j]) / rows` over a `[rows, cols]` row-major matrix, single-threaded — the
/// per-channel **mean** (the BatchNorm running mean / per-feature batch mean). Same strided-`Σ` gap as
/// `colsum`, with a `/rows` per-column finalize.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32, non-overlapping.
/// **Aliasing: all operands disjoint** — `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colmean_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    colreduce_range(
        x,
        out,
        rows as usize,
        cols as usize,
        0,
        cols as usize,
        ColKind::Mean,
    );
}

/// Multi-threaded `out[j] = (Σ_i x[i, j]) / rows` (bit-identical to [`wukong_colmean_f32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_colmean_f32`].
/// **Aliasing:** that of [`wukong_colmean_f32`] — all operands disjoint: `out` does not overlap
/// `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colmean_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    colreduce_parallel(x, out, rows, cols, ColKind::Mean);
}

/// `out[j] = sqrt(Σ_i x[i, j]²)` over a `[rows, cols]` row-major matrix, single-threaded — the
/// per-channel **L2 norm** (the column vector norm: weight-column norms, per-feature energy). The
/// strided `Σ x²` gcc/rustc leave scalar, with a `sqrt` per-column finalize.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32, non-overlapping.
/// **Aliasing: all operands disjoint** — `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_coll2_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    colreduce_range(
        x,
        out,
        rows as usize,
        cols as usize,
        0,
        cols as usize,
        ColKind::L2,
    );
}

/// Multi-threaded `out[j] = sqrt(Σ_i x[i, j]²)` (bit-identical to [`wukong_coll2_f32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_coll2_f32`].
/// **Aliasing:** that of [`wukong_coll2_f32`] — all operands disjoint: `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_coll2_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    colreduce_parallel(x, out, rows, cols, ColKind::L2);
}

/// `out[j] = sqrt((Σ_i x[i, j]²) / rows)` over a `[rows, cols]` row-major matrix, single-threaded —
/// the per-channel **RMS** (root-mean-square: per-feature magnitude, the RMSNorm-style scale). The
/// strided `Σ x²` gcc/rustc leave scalar, with a `sqrt(_/rows)` per-column finalize.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32, non-overlapping.
/// **Aliasing: all operands disjoint** — `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colrms_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    colreduce_range(
        x,
        out,
        rows as usize,
        cols as usize,
        0,
        cols as usize,
        ColKind::Rms,
    );
}

/// Multi-threaded `out[j] = sqrt((Σ_i x[i, j]²) / rows)` (bit-identical to [`wukong_colrms_f32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_colrms_f32`].
/// **Aliasing:** that of [`wukong_colrms_f32`] — all operands disjoint: `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colrms_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    colreduce_parallel(x, out, rows, cols, ColKind::Rms);
}

/// `out[j] = Σ_i x[i, j]²` over a `[rows, cols]` row-major matrix, single-threaded — the per-channel
/// **sum of squares** (the second moment / per-channel energy; the un-rooted, un-divided `coll2`/
/// `colrms`). Same strided `Σ x²` gcc/rustc leave scalar, no finalize.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32, non-overlapping.
/// **Aliasing: all operands disjoint** — `out` does not overlap `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colsumsq_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    colreduce_range(
        x,
        out,
        rows as usize,
        cols as usize,
        0,
        cols as usize,
        ColKind::SumSq,
    );
}

/// Multi-threaded `out[j] = Σ_i x[i, j]²` (bit-identical to [`wukong_colsumsq_f32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_colsumsq_f32`].
/// **Aliasing:** that of [`wukong_colsumsq_f32`] — all operands disjoint: `out` does not overlap
/// `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_colsumsq_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    colreduce_parallel(x, out, rows, cols, ColKind::SumSq);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_extents_are_safe() {
        // P14.6: column sum reduces down rows; the output has one entry per column.
        let (rows, cols) = (2usize, 3usize);
        let x = [1.0f32; 6];
        let mut out = vec![f32::from_bits(0x7fc0_0001); cols];
        // SAFETY: `x` sized rows*cols, `out` sized cols; a nonpositive rows is a no-op.
        unsafe {
            wukong_colsum_f32(x.as_ptr(), out.as_mut_ptr(), 0, cols as i64);
        }
        assert!(
            out.iter().all(|v| v.to_bits() == 0x7fc0_0001),
            "nonpositive rows must not write out"
        );
        // SAFETY: `x` sized rows*cols, `out` sized cols; the positive call writes cols.
        unsafe {
            wukong_colsum_f32(x.as_ptr(), out.as_mut_ptr(), rows as i64, cols as i64);
        }
        assert_eq!(out, vec![2.0, 2.0, 2.0]);
    }

    fn naive(x: &[f32], rows: usize, cols: usize, kind: ColKind) -> Vec<f32> {
        let mut out = vec![0.0f32; cols];
        for (j, o) in out.iter_mut().enumerate() {
            let mut s = match kind {
                ColKind::Sum | ColKind::Mean | ColKind::SumSq | ColKind::L2 | ColKind::Rms => {
                    0.0f32
                }
                ColKind::MaxAbs => x[j].abs(), // |first row|
                _ => x[j],                     // first row
            };
            let i0 = match kind {
                ColKind::Sum | ColKind::Mean | ColKind::SumSq | ColKind::L2 | ColKind::Rms => 0,
                _ => 1,
            };
            for i in i0..rows {
                let v = x[i * cols + j];
                s = match kind {
                    ColKind::SumSq | ColKind::L2 | ColKind::Rms => s + v * v,
                    ColKind::Sum | ColKind::Mean => s + v,
                    ColKind::Max => {
                        if s > v {
                            s
                        } else {
                            v
                        }
                    }
                    ColKind::Min => {
                        if s < v {
                            s
                        } else {
                            v
                        }
                    }
                    ColKind::MaxAbs => {
                        let v = v.abs();
                        if s > v {
                            s
                        } else {
                            v
                        }
                    }
                };
            }
            // Per-column finalize, mirroring `colreduce_finalize`.
            *o = match kind {
                ColKind::Mean => s / rows as f32,
                ColKind::L2 => s.sqrt(),
                ColKind::Rms => (s / rows as f32).sqrt(),
                _ => s,
            };
        }
        out
    }

    /// The column reductions (AVX2 serial and the multicore stripe split) must equal the naive strided
    /// fold **exactly** — same i-ascending per-column order, same `+`/`max`/`min` fold — so it is literal
    /// bit equality (no reassociation). Shapes straddle the 8-lane edge and the parallel column threshold.
    #[test]
    fn colreduce_matches_naive_and_parallel() {
        for (rows, cols) in [
            (1, 1),
            (2, 3), // rows < COL_RB: the tiled loop never runs, only the trailing-row path
            (3, 39),
            (4, 32), // exactly one COL_RB x 32 tile
            (5, 7),
            (5, 33), // rows % COL_RB == 1 and cols just past the 32-column tile
            (7, 31), // cols one short of a tile: the 8-wide + scalar tails inside a row block
            (8, 8),
            (33, 17),
            (64, 100),
            (128, 257),
            (50, 1000), // many 32-column stripes, so the multicore stripes run
        ] {
            // Distinct values per cell so max/min have a unique answer; an exact integer range so a
            // reordering (if any path had one) would show as a bit mismatch.
            let x: Vec<f32> = (0..rows * cols)
                .map(|t| ((t * 7 + 3) % 101) as f32 - 50.0)
                .collect();
            for kind in [
                ColKind::Sum,
                ColKind::Max,
                ColKind::Min,
                ColKind::MaxAbs,
                ColKind::Mean,
                ColKind::L2,
                ColKind::Rms,
                ColKind::SumSq,
            ] {
                let want = naive(&x, rows, cols, kind);
                let mut got = vec![0.0f32; cols];
                let mut got_par = vec![0.0f32; cols];
                let (f, fp): (
                    unsafe extern "C" fn(*const f32, *mut f32, i64, i64),
                    unsafe extern "C" fn(*const f32, *mut f32, i64, i64),
                ) = match kind {
                    ColKind::Sum => (wukong_colsum_f32, wukong_colsum_f32_parallel),
                    ColKind::Max => (wukong_colmax_f32, wukong_colmax_f32_parallel),
                    ColKind::Min => (wukong_colmin_f32, wukong_colmin_f32_parallel),
                    ColKind::MaxAbs => (wukong_colmaxabs_f32, wukong_colmaxabs_f32_parallel),
                    ColKind::Mean => (wukong_colmean_f32, wukong_colmean_f32_parallel),
                    ColKind::L2 => (wukong_coll2_f32, wukong_coll2_f32_parallel),
                    ColKind::Rms => (wukong_colrms_f32, wukong_colrms_f32_parallel),
                    ColKind::SumSq => (wukong_colsumsq_f32, wukong_colsumsq_f32_parallel),
                };
                // SAFETY: `x` sized rows*cols; the dispatched col-reduction outputs are sized cols.
                unsafe {
                    f(x.as_ptr(), got.as_mut_ptr(), rows as i64, cols as i64);
                    fp(x.as_ptr(), got_par.as_mut_ptr(), rows as i64, cols as i64);
                }
                assert_eq!(got, want, "{:?} {rows}x{cols} vs naive", kind as u8);
                assert_eq!(
                    got, got_par,
                    "{:?} serial vs parallel {rows}x{cols}",
                    kind as u8
                );
            }
        }
    }

    /// Every `ColKind`, both paths, compared by **raw bits**. `assert_eq!` on `f32` is the wrong
    /// oracle for this file: `NaN != NaN` makes a NaN column vacuously "equal" whatever the other
    /// path produced, and `-0.0 == 0.0` hides a sign flip — precisely the two cases where
    /// `_mm256_max_ps` and `f32::max` disagree, and the reason [`sfold_max`] is spelled
    /// `(a > v) ? a : v`.
    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    /// A NaN bit pattern, so it can never collide with the bits of a value that is *not* a NaN —
    /// which is the whole safety argument for [`bits_nan_collapsed`]. Prints as `4294967295` in a
    /// failure diff, which is deliberately unmistakable.
    const NAN_TOKEN: u32 = 0xFFFF_FFFF;

    /// Raw bits, except that **every NaN collapses to one token** — the comparison for the
    /// ADDITIVE kinds (`Sum`/`Mean`/`SumSq`/`L2`/`Rms`) and for nothing else.
    ///
    /// WHY THIS IS NOT A WEAKENING, AND WHY IT MUST NOT BE WIDENED. IEEE-754 specifies the *value*
    /// of an add exactly, and says nothing about which input NaN's payload comes out of one (§6.2
    /// makes payload propagation a recommendation, not a requirement). x86 `ADDPS`/`ADDSS` return
    /// SRC1 when SRC1 is a NaN, so the surviving payload is decided by which operand the compiler
    /// put in SRC1 — and `fadd` is commutative in LLVM IR, so that is the compiler's choice, not
    /// ours. At `-O3` LLVM auto-vectorizes [`colreduce_scalar`]'s inner column loop and emits the
    /// commuted `addps` in the vector body while leaving the scalar epilogue as written; the twin
    /// then reports a *different* NaN payload from [`colreduce_avx2`] (whose order the intrinsic
    /// pins) for exactly the columns the vector body covered. Both results are correct. Pinning the
    /// payload would mean stopping the optimizer from vectorizing the no-AVX2 fallback — real cost,
    /// for a property no caller can rely on and no other implementation would reproduce.
    ///
    /// Everything else stays strict, because for everything else the two paths genuinely must
    /// agree: a NaN here still has to be a NaN *there* (a NaN against a finite value is still a
    /// mismatch, so `NaN-in ⇒ NaN-out` and its converse are both pinned), `+0.0` and `-0.0` are
    /// distinct non-NaN patterns and are compared by bits, and so are ±inf and every finite value.
    ///
    /// **Never call this for `Max`/`Min`/`MaxAbs`.** Their fold is `(a > v) ? a : v`, a select the
    /// optimizer cannot commute, so their payload IS determined and IS the contract — that is the
    /// repo-wide `_mm256_max_ps`-vs-`f32::max` landmine, and collapsing NaN there would blind the
    /// only test that catches it.
    fn bits_nan_collapsed(v: &[f32]) -> Vec<u32> {
        v.iter()
            .map(|f| if f.is_nan() { NAN_TOKEN } else { f.to_bits() })
            .collect()
    }

    /// The kinds folded with `+` (over `x` or over `x*x`), i.e. the ones whose NaN payload the
    /// optimizer is free to change. The complement — `Max`/`Min`/`MaxAbs` — is payload-exact.
    fn is_additive(kind: ColKind) -> bool {
        matches!(
            kind,
            ColKind::Sum | ColKind::Mean | ColKind::SumSq | ColKind::L2 | ColKind::Rms
        )
    }

    /// The AVX2 tile must equal the un-blocked scalar twin **bit for bit** on ordinary data, over
    /// shapes that straddle every edge the [`COL_RB`] x 32 tile introduced: `rows` below / at / just
    /// past a row block, and `cols` below / at / just past the 32-column tile plus its 8-wide and
    /// scalar tails. Without this the only cross-check was against `naive`, which on a non-AVX2 host
    /// would compare the scalar twin with itself.
    ///
    /// The data is deliberately **order-sensitive**: every fifth cell is `1e7` while the rest are
    /// small non-dyadic values, so a running sum that has swallowed a `1e7` rounds a subsequent small
    /// addend away entirely. Summing the same column in a different order therefore lands on a
    /// different f32. An earlier version of this test used a small exact-integer range, whose sum is
    /// order-*independent* — it passed unchanged when the tile was mutated to fold its `COL_RB` rows
    /// descending, which is exactly the regression this test exists to catch.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !crate::isa::avx2() {
            return;
        }
        for &cols in &[
            1usize, 7, 8, 9, 15, 16, 31, 32, 33, 39, 40, 41, 64, 65, 100, 257,
        ] {
            for &rows in &[1usize, 2, 3, 4, 5, 8, 9, 33, 100] {
                let x: Vec<f32> = (0..rows * cols)
                    .map(|t| {
                        if t % 5 == 0 {
                            1.0e7
                        } else {
                            ((t * 13 + 5) % 97) as f32 * 0.1 + 0.333_333_34
                        }
                    })
                    .collect();
                for kind in [
                    ColKind::Sum,
                    ColKind::Max,
                    ColKind::Min,
                    ColKind::MaxAbs,
                    ColKind::Mean,
                    ColKind::L2,
                    ColKind::Rms,
                    ColKind::SumSq,
                ] {
                    let mut s = vec![f32::from_bits(0x7f80_0001); cols]; // sentinel: a signalling NaN
                    let mut v = vec![f32::from_bits(0x7f80_0001); cols];
                    // SAFETY: `x` sized rows*cols; `s`/`v` sized cols (serial vs parallel).
                    unsafe {
                        colreduce_scalar(x.as_ptr(), s.as_mut_ptr(), rows, cols, 0, cols, kind);
                        colreduce_avx2(x.as_ptr(), v.as_mut_ptr(), rows, cols, 0, cols, kind);
                    }
                    assert_eq!(
                        bits(&s),
                        bits(&v),
                        "scalar != avx2 at rows={rows} cols={cols} kind={}",
                        kind as u8
                    );
                }
            }
        }
    }

    /// The packed finalize [`colreduce_finalize_avx2`] must equal the scalar reference
    /// [`colreduce_finalize`] **by raw bits**, on the values where a divide or a square root is
    /// allowed to be interesting, at every vector/tail offset, and it must not write one byte
    /// outside `[j0, j1)`.
    ///
    /// This is a direct differential on the substitution itself rather than on a whole reduction,
    /// because a whole reduction cannot *reach* most of these inputs: the fold that feeds the
    /// finalize is a sum of squares for `L2`/`Rms`, so it can never hand them a negative operand,
    /// a denormal at the edge of underflow, or a signalling NaN. The claim being tested is the one
    /// the substitution rests on — `DIVPS`/`SQRTPS` are the same correctly-rounded IEEE operations
    /// as `DIVSS`/`SQRTSS`, lane by lane, including NaN quieting and payload, `±0`, `±inf`,
    /// underflow to a denormal and the real-indefinite `sqrt` of a negative — so it is asserted,
    /// not assumed. The **whole buffer** is compared, so a vector store running past `j1` or before
    /// `j0` (the obvious way to write this wrong) fails here rather than corrupting a neighbouring
    /// column stripe in the parallel entry.
    ///
    /// `rows` includes `16_777_217`, which is not representable in f32 and rounds to `16_777_216`:
    /// both paths must round the divisor the same way, which they do because both take it from the
    /// same `rows as f32`.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn finalize_avx2_matches_scalar_bit_for_bit() {
        if !crate::isa::avx2() {
            return;
        }
        // Every case the finalize can meet and several it cannot, so the IEEE claim is tested at
        // full strength: both zeros, both infinities, a quiet NaN, a second quiet payload, a
        // SIGNALLING NaN (which both forms must quiet identically), a negative NaN, the smallest
        // normal and the smallest denormal (whose `/rows` underflows further), a mid denormal,
        // `f32::MAX`, exact and inexact square roots, and negatives (whose `sqrt` is the real
        // indefinite).
        let pool: [f32; 20] = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            4.0,
            3.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0x7f80_0001), // signalling NaN
            f32::from_bits(0xffc0_5678), // negative quiet NaN
            f32::MIN_POSITIVE,
            f32::from_bits(1), // smallest denormal
            f32::from_bits(0x0000_1234),
            f32::MAX,
            -f32::MAX,
            -3.5,
            -f32::MIN_POSITIVE,
            1.0e-30,
        ];
        // A sentinel outside `[j0, j1)`: a signalling NaN whose payload no operation here produces.
        let guard = f32::from_bits(0x7f81_dead);
        for &j0 in &[0usize, 1, 3, 5, 8, 9] {
            for len in [
                0usize, 1, 2, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 64, 71,
            ] {
                for &rows in &[1usize, 2, 3, 7, 1024, 12345, 16_777_217] {
                    for kind in [
                        ColKind::Mean,
                        ColKind::L2,
                        ColKind::Rms,
                        ColKind::Sum,
                        ColKind::SumSq,
                        ColKind::Max,
                    ] {
                        // Two guard columns past the end as well as the `j0` prefix.
                        let mut base = vec![guard; j0 + len + 2];
                        for (k, b) in base.iter_mut().enumerate().skip(j0).take(len) {
                            // 7 is coprime with the pool length, so each offset sees a rotation of
                            // the whole pool and every value lands at every lane position.
                            *b = pool[(k * 7 + j0 * 3 + rows) % pool.len()];
                        }
                        let (mut s, mut v) = (base.clone(), base.clone());
                        // SAFETY: `s`/`v` are clones sized for the output (cols).
                        unsafe {
                            colreduce_finalize(s.as_mut_ptr(), j0, j0 + len, rows, kind);
                            colreduce_finalize_avx2(v.as_mut_ptr(), j0, j0 + len, rows, kind);
                        }
                        assert_eq!(
                            bits(&s),
                            bits(&v),
                            "finalize scalar != avx2 at j0={j0} len={len} rows={rows} kind={}",
                            kind as u8
                        );
                        // …and neither form touched anything outside `[j0, j0+len)`.
                        assert!(
                            s[..j0].iter().chain(&s[j0 + len..]).all(|f| f.to_bits()
                                == guard.to_bits())
                                && v[..j0].iter().chain(&v[j0 + len..]).all(|f| f.to_bits()
                                    == guard.to_bits()),
                            "finalize wrote outside [j0, j1) at j0={j0} len={len} rows={rows} kind={}",
                            kind as u8
                        );
                    }
                }
            }
        }
    }

    /// The fold order and comparison shape are the contract, so pin them on the values where a
    /// "reasonable-looking" rewrite breaks: **NaN, ±0.0 and exact ties**.
    ///
    /// `_mm256_max_ps(a, v)` returns `v` whenever `a > v` is false — including when *either* operand
    /// is NaN and on a tie. `f32::max` does the opposite (it is NaN-suppressing and returns the other
    /// operand), and `f32::max(-0.0, 0.0)` is unspecified between the two zeros. So a twin written
    /// with `f32::max` diverges from the vector path on exactly this input. The assertion here is
    /// only scalar-twin == AVX2 **by bits**; it deliberately does NOT hard-code which zero or which
    /// NaN payload wins, because that is the hardware's definition, not ours — what must hold is that
    /// both paths agree, at every row-block and column-tile offset.
    ///
    /// TWO MATRICES, because the two kind families are held to different strengths (see
    /// [`bits_nan_collapsed`] for the IEEE argument, and note that this test is *release-only*
    /// interesting — the divergence it guards needs the optimizer to have vectorized the twin):
    ///
    /// * `awkward` carries NaN — two distinct payloads, plus the real-indefinite `0xFFC00000` that
    ///   `+inf + -inf` manufactures, so three payloads compete — and both infinities.
    ///   `Max`/`Min`/`MaxAbs` are compared by raw bits here; the additive kinds by collapsed bits.
    /// * `ordinary` holds the same awkward *non*-NaN values (`±0.0`, exact ties, and a `1e7`/`-1e7`
    ///   pair against small non-dyadic addends so the additive fold is order-sensitive and a
    ///   reassociation would still show as a bit mismatch). It contains no NaN and no infinity, so
    ///   no operation over it can produce a NaN and **every** kind stays a strict bit gate — which
    ///   is what keeps the additive arm of this test from going vacuous under the collapse.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn nan_signed_zero_and_ties_agree_between_paths() {
        if !crate::isa::avx2() {
            return;
        }
        let nan = f32::NAN;
        let nan2 = f32::from_bits(0x7fc0_1234); // a different quiet-NaN payload
                                                // 12 rows (3 full COL_RB blocks) x 37 columns (one 32-tile + a 5-column tail) so the awkward
                                                // values land in the tiled body, the 8-wide tail and the scalar tail alike.
        let (rows, cols) = (12usize, 37usize);
        let awkward = [
            0.0f32,
            -0.0,
            nan,
            nan2,
            1.0,
            -1.0,
            3.5,
            -3.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            2.0,
            2.0, // an exact tie with the entry before it
        ];
        // No NaN and no infinity: the pool rotation below puts every entry in every column, so a
        // single `f32::INFINITY` here would make every additive column `+inf` (or, with `-inf` too,
        // the indefinite NaN) and cost this matrix the order-sensitivity it exists for.
        let ordinary = [
            0.0f32, -0.0, 1.0e7, 0.1, 1.0, -1.0, 3.5, -3.5, -1.0e7, -0.375, 2.0,
            2.0, // an exact tie with the entry before it
        ];
        let build = |pool: &[f32; 12]| {
            let mut x = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    // A shifting permutation so every column sees the awkward values at a different
                    // row, and every (row block, column tile) position sees one somewhere. `5` is
                    // coprime with the pool length, so each column is a rotation of the whole pool.
                    x[i * cols + j] = pool[(i * 5 + j * 7) % pool.len()];
                }
            }
            x
        };
        // `ordinary` runs first on purpose: it is the fully strict matrix, so when a change breaks
        // both it is the more diagnostic failure to see.
        for (name, x, nan_free) in [
            ("ordinary", build(&ordinary), true),
            ("awkward", build(&awkward), false),
        ] {
            for kind in [
                ColKind::Sum,
                ColKind::Max,
                ColKind::Min,
                ColKind::MaxAbs,
                ColKind::Mean,
                ColKind::L2,
                ColKind::Rms,
                ColKind::SumSq,
            ] {
                let mut s = vec![0.0f32; cols];
                let mut v = vec![0.0f32; cols];
                let mut p = vec![0.0f32; cols];
                // SAFETY: `x` sized rows*cols; `v`/`p` sized cols.
                unsafe {
                    colreduce_scalar(x.as_ptr(), s.as_mut_ptr(), rows, cols, 0, cols, kind);
                    colreduce_avx2(x.as_ptr(), v.as_mut_ptr(), rows, cols, 0, cols, kind);
                    // …and the two-stripe split, so the parallel entry is held to the same bits.
                    colreduce_range(x.as_ptr(), p.as_mut_ptr(), rows, cols, 0, 16, kind);
                    colreduce_range(x.as_ptr(), p.as_mut_ptr(), rows, cols, 16, cols, kind);
                }
                // Raw bits everywhere except the one place IEEE-754 leaves undefined: the NaN
                // payload of an additive fold, on the matrix that can actually produce two of them.
                let cmp: fn(&[f32]) -> Vec<u32> = if nan_free || !is_additive(kind) {
                    bits
                } else {
                    bits_nan_collapsed
                };
                assert_eq!(
                    cmp(&s),
                    cmp(&v),
                    "{name} NaN/±0/tie: scalar != avx2, kind={}",
                    kind as u8
                );
                assert_eq!(
                    cmp(&s),
                    cmp(&p),
                    "{name} NaN/±0/tie: scalar != striped, kind={}",
                    kind as u8
                );
            }
        }
    }

    /// A tie must resolve the way `_mm256_max_ps`/`_mm256_min_ps` resolve it — **the second operand
    /// (the newer row) wins** — and `sfold_max`/`sfold_min` must say the same thing. Checked on ±0.0,
    /// where the two candidate answers are distinguishable by bits: folding `max` over a column of
    /// `[+0.0, -0.0]` yields `-0.0` (because `+0.0 > -0.0` is false, so the fold takes the new row),
    /// which `f32::max` would NOT give.
    #[test]
    fn signed_zero_tie_takes_the_newer_row() {
        assert_eq!(sfold_max(0.0, -0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(sfold_max(-0.0, 0.0).to_bits(), (0.0f32).to_bits());
        assert_eq!(sfold_min(0.0, -0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(sfold_min(-0.0, 0.0).to_bits(), (0.0f32).to_bits());
        // …and through the public entry point, over a 2-row column.
        let x = [0.0f32, -0.0];
        let mut got = [1.0f32];
        // SAFETY: `x` holds 2 f32 (one column of two rows); `got` one f32 (colmax, signed-zero).
        unsafe { wukong_colmax_f32(x.as_ptr(), got.as_mut_ptr(), 2, 1) };
        assert_eq!(
            got[0].to_bits(),
            (-0.0f32).to_bits(),
            "colmax lost the ±0 tie rule"
        );
        let mut got = [1.0f32];
        // SAFETY: `x` holds 2 f32; `got` one f32 (colmin, signed-zero).
        unsafe { wukong_colmin_f32(x.as_ptr(), got.as_mut_ptr(), 2, 1) };
        assert_eq!(
            got[0].to_bits(),
            (-0.0f32).to_bits(),
            "colmin lost the ±0 tie rule"
        );
    }
}
