//! Column reductions along the **outer** (batch / row) axis of a `[rows, cols]` row-major matrix,
//! producing a per-column `[cols]` result. Eight kinds ([`ColKind`]), one `wukong_col*_f32[_parallel]`
//! entry pair each: the **sum** `out[j] = Σ_i x[i,j]` (bias gradient `db = Σ_batch dY`, batch sum,
//! reduce-along-axis-0), **max** / **min** / **maxabs** (per-channel statistics for quantization,
//! axis-0 max/min pooling), **sumsq**, and the three that add a scalar finalize over the folded column
//! — **mean** (`/rows`), **l2** (`sqrt`), **rms** (`sqrt(_/rows)`), applied by [`colreduce_finalize`]
//! per column and so independent of the parallel column split.
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
//! memory-level parallelism a single sequential stream cannot reach). See [`colreduce_avx2`].
//!
//! The fold order is **i-ascending per column** — identical to the scalar nest — in the AVX2 and
//! scalar paths, across the row blocking, and across serial/parallel (disjoint column stripes, no
//! cross-stripe combine), so each kernel is its own bit-exact oracle (the interpreter marshals the
//! serial form; the differential gate compares interp vs native, both folding the same
//! `_mm256_add_ps`/`_mm256_max_ps`/`_mm256_min_ps` lane tree).

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
/// no-ops. Scalar arithmetic only (a divide and/or a `sqrt`), identical in the AVX2 and scalar paths
/// and independent of the column split, so it does not perturb the serial == parallel bit-equality.
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

/// The five per-element folds, as ordinary (non-`target_feature`) `#[inline(always)]` functions so
/// that the AVX2 kernel's scalar tails, the scalar twin [`colreduce_scalar`] and the unit tests all
/// use **literally the same code** — the fold order is the correctness contract of this file, and a
/// second hand-written copy is exactly how a twin drifts. `max`/`min` are spelled `(a > v) ? a : v` /
/// `(a < v) ? a : v` deliberately, NOT `f32::max`/`f32::min`: that is what `_mm256_max_ps` /
/// `_mm256_min_ps` compute (the second operand wins a tie and any NaN comparison), so the vector and
/// scalar paths agree on NaN, ±0 and ties.
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
/// lives*, never the order it is folded in. The `rows % COL_RB` trailing rows fold one row at a time
/// afterwards, continuing the same ascending sequence, and the `(j1-j0) % 32` trailing columns fold
/// 8-wide and then scalar inside each row block.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32; `j0 <= j1 <= cols`; AVX2 available.
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
    let sign = _mm256_set1_ps(-0.0); // for MaxAbs: clear the sign bit with andnot

    // The tiled fold, monomorphic per kind: the `match` below is hoisted out of every loop and each
    // arm pastes its own vector fold (`|a, v| …`) and scalar fold (`sfold_…`) into ONE traversal.
    // Writing the traversal once is deliberate — five hand-copied nests are five chances for one of
    // them to lose the ascending-`i` order that the whole file's bit-exactness rests on.
    macro_rules! fold_all {
        (|$a:ident, $v:ident| $vfold:expr, $sfold:ident) => {{
            let mut i = i0;
            // Main tile: COL_RB rows x 32 columns, accumulators resident in 4 YMM registers.
            while i + COL_RB <= rows {
                let mut j = j0;
                while j + 32 <= j1 {
                    let mut a0 = _mm256_loadu_ps(out.add(j));
                    let mut a1 = _mm256_loadu_ps(out.add(j + 8));
                    let mut a2 = _mm256_loadu_ps(out.add(j + 16));
                    let mut a3 = _mm256_loadu_ps(out.add(j + 24));
                    let mut k = 0usize;
                    while k < COL_RB {
                        let xr = x.add((i + k) * cols + j);
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
                    let mut acc = _mm256_loadu_ps(out.add(j));
                    let mut k = 0usize;
                    while k < COL_RB {
                        acc = {
                            let ($a, $v) = (acc, _mm256_loadu_ps(x.add((i + k) * cols + j)));
                            $vfold
                        };
                        k += 1;
                    }
                    _mm256_storeu_ps(out.add(j), acc);
                    j += 8;
                }
                while j < j1 {
                    let mut s = *out.add(j);
                    let mut k = 0usize;
                    while k < COL_RB {
                        s = $sfold(s, *x.add((i + k) * cols + j));
                        k += 1;
                    }
                    *out.add(j) = s;
                    j += 1;
                }
                i += COL_RB;
            }
            // rows % COL_RB trailing rows, one at a time — the same ascending-`i` continuation.
            while i < rows {
                let xr = x.add(i * cols);
                let mut j = j0;
                while j + 8 <= j1 {
                    let acc = {
                        let ($a, $v) = (_mm256_loadu_ps(out.add(j)), _mm256_loadu_ps(xr.add(j)));
                        $vfold
                    };
                    _mm256_storeu_ps(out.add(j), acc);
                    j += 8;
                }
                while j < j1 {
                    *out.add(j) = $sfold(*out.add(j), *xr.add(j));
                    j += 1;
                }
                i += 1;
            }
        }};
    }
    match kind {
        // Σ x[i,j] (Mean reuses the Sum fold, then divides in the finalize).
        ColKind::Sum | ColKind::Mean => fold_all!(|a, v| _mm256_add_ps(a, v), sfold_add),
        // Σ x[i,j]² (L2/Rms reuse this, then sqrt[/rows] in the finalize). The square is a separate
        // `mul` then `add` (NOT an FMA) so the scalar twin `a + v*v` matches it bit-for-bit (no `fma`
        // target feature assumed here, and one rounding model shared by both paths).
        ColKind::SumSq | ColKind::L2 | ColKind::Rms => {
            fold_all!(|a, v| _mm256_add_ps(a, _mm256_mul_ps(v, v)), sfold_sq)
        }
        ColKind::Max => fold_all!(|a, v| _mm256_max_ps(a, v), sfold_max),
        ColKind::Min => fold_all!(|a, v| _mm256_min_ps(a, v), sfold_min),
        // |v| = andnot(-0.0, v) (clear the sign bit), then fold by max — the same sign-mask abs the
        // RED_MAXABS reduction uses, so it agrees lane-for-lane with `sfold_maxabs`'s `v.abs()`.
        ColKind::MaxAbs => {
            fold_all!(
                |a, v| _mm256_max_ps(a, _mm256_andnot_ps(sign, v)),
                sfold_maxabs
            )
        }
    }
    // Per-column finalize for the statistics kinds (Mean -> /rows, L2 -> sqrt, Rms -> sqrt(/rows));
    // a no-op for Sum/SumSq/Max/Min/MaxAbs. Identical scalar arithmetic in both paths.
    colreduce_finalize(out, j0, j1, rows, kind);
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
        if is_x86_feature_detected!("avx2") {
            colreduce_avx2(x, out, rows, cols, j0, j1, kind);
            return;
        }
    }
    colreduce_scalar(x, out, rows, cols, j0, j1, kind);
}

/// Column count below which the parallel reduction just runs serially.
const COLREDUCE_PAR_MIN: usize = 256;

/// Multi-threaded `out[j] = ⊕_i x[i, j]`: the **columns** are split into disjoint stripes across cores
/// (each core folds its column range over all rows). Each stripe writes a disjoint slice of `out`, and
/// every column is folded in the same i-ascending order regardless of the split — so the result is
/// **bit-identical to the serial kernel** the interpreter marshals (no cross-stripe combine). Below
/// `COLREDUCE_PAR_MIN` columns it runs serial.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32, non-overlapping.
#[inline]
unsafe fn colreduce_parallel(x: *const f32, out: *mut f32, rows: i64, cols: i64, kind: ColKind) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if c < COLREDUCE_PAR_MIN {
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
#[no_mangle]
pub unsafe extern "C" fn wukong_colsum_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
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
            (50, 1000), // > COLREDUCE_PAR_MIN so the multicore stripes run
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
        if !is_x86_feature_detected!("avx2") {
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
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn nan_signed_zero_and_ties_agree_between_paths() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let nan = f32::NAN;
        let nan2 = f32::from_bits(0x7fc0_1234); // a different quiet-NaN payload
                                                // 12 rows (3 full COL_RB blocks) x 37 columns (one 32-tile + a 5-column tail) so the awkward
                                                // values land in the tiled body, the 8-wide tail and the scalar tail alike.
        let (rows, cols) = (12usize, 37usize);
        let pool = [
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
        let mut x = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                // A shifting permutation so every column sees the awkward values at a different row,
                // and every (row block, column tile) position sees a NaN / ±0 / tie somewhere.
                x[i * cols + j] = pool[(i * 5 + j * 7) % pool.len()];
            }
        }
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
            unsafe {
                colreduce_scalar(x.as_ptr(), s.as_mut_ptr(), rows, cols, 0, cols, kind);
                colreduce_avx2(x.as_ptr(), v.as_mut_ptr(), rows, cols, 0, cols, kind);
                // …and the two-stripe split, so the parallel entry is held to the same bits.
                colreduce_range(x.as_ptr(), p.as_mut_ptr(), rows, cols, 0, 16, kind);
                colreduce_range(x.as_ptr(), p.as_mut_ptr(), rows, cols, 16, cols, kind);
            }
            assert_eq!(
                bits(&s),
                bits(&v),
                "NaN/±0/tie: scalar != avx2, kind={}",
                kind as u8
            );
            assert_eq!(
                bits(&s),
                bits(&p),
                "NaN/±0/tie: scalar != striped, kind={}",
                kind as u8
            );
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
        unsafe { wukong_colmax_f32(x.as_ptr(), got.as_mut_ptr(), 2, 1) };
        assert_eq!(
            got[0].to_bits(),
            (-0.0f32).to_bits(),
            "colmax lost the ±0 tie rule"
        );
        let mut got = [1.0f32];
        unsafe { wukong_colmin_f32(x.as_ptr(), got.as_mut_ptr(), 2, 1) };
        assert_eq!(
            got[0].to_bits(),
            (-0.0f32).to_bits(),
            "colmin lost the ±0 tie rule"
        );
    }
}
