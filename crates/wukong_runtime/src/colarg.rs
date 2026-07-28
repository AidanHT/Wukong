//! Per-**column** arg-reductions along the **outer** (batch / row) axis of a `[rows, cols]` row-major
//! matrix, producing a per-column `[cols]` **index**: `out[j] = argmax_i x[i, j]` (and the argmin
//! twin) — the row index of the largest (smallest) value in each column. This is the axis-0 arg-reduce
//! (`np.argmax(x, axis=0)` / `torch.argmax(x, dim=0)`): per-column best-row selection — picking, for
//! every feature/channel, which batch row attains its extreme (the per-column winner of a tournament,
//! e.g. nearest-prototype / per-class top example). It is the column-axis sibling of the per-row
//! [`crate::wukong_rowargmax_i32`] and the value-only column reductions in [`crate::wukong_colmax_f32`].
//!
//! **Tie-break is the whole correctness contract.** On an exact value tie the **LOWEST row index**
//! wins (standard argmax / PyTorch / NumPy axis-0 semantics). That rule must be identical in the scalar
//! twin, the AVX2 path, and across serial/parallel — it is what makes the kernel its own bit-exact
//! oracle. We pin it with a **STRICT** value compare: argmax updates the running best only where the
//! new value is `>` the current best (argmin: `<`), so a later equal value never displaces the earlier
//! (lower) row index.
//!
//! Fast *and* bit-exact via the same row-major, cache-friendly traversal as [`crate::wukong_colsum_f32`]
//! and friends: process **8 adjacent columns per AVX2 step**. For each row `i`, load 8 contiguous f32
//! (`x[i*cols + j .. +8]` = 8 columns of row `i`), compare against a per-lane running best-value vector,
//! and **blend-update** both that best-value vector *and* a best-index vector — the row `i` broadcast to
//! all 8 lanes (`_mm256_set1_ps(i as f32)`) is written only into the strictly-winning lanes (masked by
//! `_CMP_GT_OQ` for max, `_CMP_LT_OQ` for min), exactly the lane-update idiom of [`crate::wukong_rowargmax_i32`].
//! Each lane is an **independent column**, so — unlike the per-row kernel — there is *no* horizontal
//! lane collapse (the eight results are eight separate columns), which makes this simpler. Row indices
//! ride an f32 lane, which is exact only below the f32 mantissa limit `2^24`, so [`colarg_range`]
//! dispatches to the AVX2 path only when `rows < 2^24` and hands taller matrices to the exact-integer
//! scalar twin (the same guard shape `argreduce_chunk` uses for its i32 index lanes). The
//! best-value vector is seeded from **row 0** and the best-index vector from 0, then rows `1..rows` are
//! scanned. The `cols % 8` remainder columns are a scalar tail (the same strict-compare scan).
//!
//! Columns are independent, so `_parallel` splits the **output columns** into disjoint contiguous
//! ranges across cores (each core scans all rows for its columns) — every `out[j]` is written by
//! exactly one thread, scanning rows in the same i-ascending order regardless of the split, so the
//! result is **bit-identical to the serial kernel** the interpreter would marshal (no cross-thread
//! combine). The naive column-outer spelling `for j { for i { ... x[i*cols+j] } }` reads `x` with
//! stride `cols` — the strided arg-scan gcc/rustc leave **scalar** — the same gap the column
//! value-reductions exploit.

/// Column count below which the parallel entry just runs serially (the rayon fan-out only pays off
/// once there are enough columns to spread across cores). Matches the small-input serial thresholds
/// the other parallel column kernels use.
const COLARG_PAR_MIN: usize = 256;

/// Scalar per-column argmax/argmin reference (the no-AVX2 fallback **and** the bit-exact oracle the
/// AVX2 path must match lane-for-lane). For each column `j` in `[j0, j1)`: seed `best_row = 0`,
/// `best_val = x[0*cols + j]`, then scan rows `1..rows` with a **strict** compare
/// (`v > best_val` for argmax, `v < best_val` for argmin) so the lowest row index wins on a tie. Writes
/// the winning row index as `i32` into `out[j]`.
///
/// # Safety
/// `x` valid for `rows*cols` `f32`; `out` valid for `cols` `i32`; `j0 <= j1 <= cols`; `rows >= 1`.
unsafe fn colarg_scalar(
    x: *const f32,
    out: *mut i32,
    rows: usize,
    cols: usize,
    j0: usize,
    j1: usize,
    is_max: bool,
) {
    // One column at a time: the running best value and best row live in registers for the whole
    // i-scan, so `out[j]` is written exactly once, at the end. Nothing reads `out` before that write
    // — unlike the AVX2 twin, which needs a real seeded scratch buffer because its accumulators stay
    // resident across all rows.
    for j in j0..j1 {
        let mut best_val = *x.add(j); // x[0*cols + j]
        let mut best_row = 0usize;
        let mut i = 1usize;
        while i < rows {
            let v = *x.add(i * cols + j);
            // STRICT compare: a tie keeps the earlier (lower) row index.
            let better = if is_max { v > best_val } else { v < best_val };
            if better {
                best_val = v;
                best_row = i;
            }
            i += 1;
        }
        *out.add(j) = best_row as i32;
    }
}

/// AVX2 per-column argmax/argmin over the column range `[j0, j1)`, streaming `x` **row-major in a
/// SINGLE pass** with the full column range's running best held in L1 scratch (8 columns/AVX2 step).
/// Bit-identical to [`colarg_scalar`]: each column updates only where the new value **strictly** beats
/// its running best (`_mm256_cmp_ps(v, best, GT/LT)` → `_mm256_blendv_ps`), so a tie keeps the earlier
/// (lower) row index; rows are scanned ascending (seed row 0, then `1..rows`), the identical order.
///
/// The previous form scanned all rows **per 8-column band**, re-reading the whole matrix `cols/8`
/// times (L3-rebound, latency-bound ~ tied gcc). This form keeps `best_val`/`best_idx` for the whole
/// range L1-resident and reads each `x` element **once** from DRAM — the same single-pass, cache-
/// resident-accumulator structure as [`crate::wukong_colsum_f32`]. The `width % 8` trailing columns
/// fold in a per-row scalar tail (same strict compare), so the range is covered in one pass.
///
/// # Safety
/// `x` valid for `rows*cols` `f32`; `out` valid for `cols` `i32`; `j0 <= j1 <= cols`; `rows >= 1`;
/// AVX2 must be available. `rows < 2^24` — the running row index lives in an f32 lane, so above the
/// f32 mantissa limit it would round and stop matching [`colarg_scalar`]; [`colarg_range`] is the
/// only caller and is what guarantees the bound.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn colarg_avx2(
    x: *const f32,
    out: *mut i32,
    rows: usize,
    cols: usize,
    j0: usize,
    j1: usize,
    is_max: bool,
) {
    use std::arch::x86_64::*;
    let width = j1 - j0;
    // L1-resident running best for every column in the range (row index tracked as f32 — exact for the
    // `< rows` integers, matching the scalar twin's `cvtt`). Seeded from row 0.
    let mut best_val = vec![0f32; width];
    let mut best_idx = vec![0f32; width];
    let bvp = best_val.as_mut_ptr();
    let bip = best_idx.as_mut_ptr();
    for jj in 0..width {
        *bvp.add(jj) = *x.add(j0 + jj); // x[0, j0+jj]
        // best_idx already 0.0 (row 0)
    }
    // Stream rows 1..rows ONCE, updating the resident accumulators 8 columns at a time.
    let n8 = width & !7; // floor to multiple of 8
    let mut i = 1usize;
    while i < rows {
        let row_i = _mm256_set1_ps(i as f32);
        let xrow = x.add(i * cols + j0);
        let mut jj = 0usize;
        while jj < n8 {
            let bv = _mm256_loadu_ps(bvp.add(jj));
            let bi = _mm256_loadu_ps(bip.add(jj));
            let v = _mm256_loadu_ps(xrow.add(jj)); // 8 columns of row i
            // STRICT compare: argmax updates where v > best; argmin where v < best (tie keeps lower row).
            let mask = if is_max {
                _mm256_cmp_ps(v, bv, _CMP_GT_OQ)
            } else {
                _mm256_cmp_ps(v, bv, _CMP_LT_OQ)
            };
            _mm256_storeu_ps(bvp.add(jj), _mm256_blendv_ps(bv, v, mask));
            _mm256_storeu_ps(bip.add(jj), _mm256_blendv_ps(bi, row_i, mask));
            jj += 8;
        }
        // Per-row scalar tail for the `width % 8` trailing columns (same strict-compare update).
        while jj < width {
            let v = *xrow.add(jj);
            let better = if is_max {
                v > *bvp.add(jj)
            } else {
                v < *bvp.add(jj)
            };
            if better {
                *bvp.add(jj) = v;
                *bip.add(jj) = i as f32;
            }
            jj += 1;
        }
        i += 1;
    }
    // Write the winning row indices as i32 (round-toward-zero recovers the exact integers).
    let mut jj = 0usize;
    while jj < n8 {
        let idx_i = _mm256_cvttps_epi32(_mm256_loadu_ps(bip.add(jj)));
        _mm256_storeu_si256(out.add(j0 + jj) as *mut __m256i, idx_i);
        jj += 8;
    }
    while jj < width {
        *out.add(j0 + jj) = *bip.add(jj) as i32;
        jj += 1;
    }
}

/// Dispatch AVX2 vs scalar for the column range `[j0, j1)`.
///
/// # Safety
/// Operand-size contract of [`colarg_avx2`].
#[inline]
unsafe fn colarg_range(
    x: *const f32,
    out: *mut i32,
    rows: usize,
    cols: usize,
    j0: usize,
    j1: usize,
    is_max: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // The AVX2 twin carries the running row index in an f32 lane, exact only below the f32
        // mantissa limit; past it `i as f32` rounds and the recovered index is off by one, so the
        // exact-integer scalar twin takes over. Same guard shape `argreduce_chunk` (reduce.rs) uses
        // for its i32 index lanes; one comparison per call.
        if is_x86_feature_detected!("avx2") && rows < (1usize << 24) {
            colarg_avx2(x, out, rows, cols, j0, j1, is_max);
            return;
        }
    }
    colarg_scalar(x, out, rows, cols, j0, j1, is_max);
}

/// Serial per-column arg-reduction: `out[j] = {argmax,argmin}_i x[i, j]` for every column. Handles the
/// edge cases: `rows <= 0 || cols <= 0` writes nothing; `rows == 1` makes every `out[j] = 0` (the
/// row-0 seed with no rows to scan).
///
/// # Safety
/// `x` valid for `rows*cols` `f32`; `out` valid for `cols` `i32`; non-overlapping.
#[inline]
unsafe fn colarg_serial(x: *const f32, out: *mut i32, rows: i64, cols: i64, is_max: bool) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    colarg_range(x, out, r, c, 0, c, is_max);
}

/// Parallel per-column arg-reduction — **bit-identical** to [`colarg_serial`]. The output **columns**
/// are split into disjoint contiguous stripes across cores (each core scans all rows for its column
/// range). Every `out[j]` is written by exactly one thread, in the same i-ascending row order
/// regardless of the split, so the result (including the lowest-index tie-break) is the same as serial
/// — no cross-thread combine. Below [`COLARG_PAR_MIN`] columns it runs serial.
///
/// # Safety
/// `x` valid for `rows*cols` `f32`; `out` valid for `cols` `i32`; non-overlapping.
#[inline]
unsafe fn colarg_par(x: *const f32, out: *mut i32, rows: i64, cols: i64, is_max: bool) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if c < COLARG_PAR_MIN {
        colarg_range(x, out, r, c, 0, c, is_max);
        return;
    }
    use rayon::prelude::*;
    // This entry can be the process's FIRST rayon touch, so it must configure the global pool before
    // forking (crate::ensure_global_pool's stated contract): otherwise rayon lazily builds its default
    // 2 MiB-stack registry here and the runtime's 16 MiB build_global silently loses the race for the
    // whole process, leaving later outlined @parallel region bodies (~1.5 MiB of privatized scratch at
    // S=512) on 2 MiB stacks. Configuration only -- the stripe split is unchanged, so the bits are too.
    crate::ensure_global_pool();
    // One stripe per core, each a multiple of 8 columns (keep the AVX2 8-wide main loop aligned to the
    // stripe boundary so every stripe's tail is only its own `cols % 8`); the last stripe absorbs the
    // remainder. Raw pointers cross the rayon closure boundary as integers (the same pattern as the
    // parallel column reductions / GEMM); each task reads all rows and writes a disjoint `out[]` stripe.
    let nthreads = rayon::current_num_threads().max(1);
    let per = (c.div_ceil(nthreads)).next_multiple_of(8).max(8);
    let nstripes = c.div_ceil(per);
    let (x_addr, out_addr) = (x as usize, out as usize);
    (0..nstripes).into_par_iter().for_each(|s| {
        let j0 = s * per;
        let j1 = (j0 + per).min(c);
        // SAFETY: disjoint out[] stripe per task; pointers re-derived from the captured addresses.
        unsafe {
            colarg_range(x_addr as *const f32, out_addr as *mut i32, r, c, j0, j1, is_max);
        }
    });
}

/// `out[j] = argmax_i x[i, j]` over a `[rows, cols]` row-major matrix, single-threaded — the per-column
/// best-row index (axis-0 argmax). Lowest row index wins on a value tie. `rows <= 0 || cols <= 0`
/// writes nothing; `rows == 1` → every `out[j] = 0`.
///
/// # Safety
/// `x` valid for `rows*cols` `f32`; `out` valid for `cols` `i32`; non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn wukong_colargmax_i32(x: *const f32, out: *mut i32, rows: i64, cols: i64) {
    colarg_serial(x, out, rows, cols, true);
}

/// Multi-threaded `out[j] = argmax_i x[i, j]` (bit-identical to [`wukong_colargmax_i32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_colargmax_i32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_colargmax_i32_parallel(
    x: *const f32,
    out: *mut i32,
    rows: i64,
    cols: i64,
) {
    colarg_par(x, out, rows, cols, true);
}

/// `out[j] = argmin_i x[i, j]` over a `[rows, cols]` row-major matrix, single-threaded — the per-column
/// best-row index (axis-0 argmin). Lowest row index wins on a value tie. `rows <= 0 || cols <= 0`
/// writes nothing; `rows == 1` → every `out[j] = 0`.
///
/// # Safety
/// `x` valid for `rows*cols` `f32`; `out` valid for `cols` `i32`; non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn wukong_colargmin_i32(x: *const f32, out: *mut i32, rows: i64, cols: i64) {
    colarg_serial(x, out, rows, cols, false);
}

/// Multi-threaded `out[j] = argmin_i x[i, j]` (bit-identical to [`wukong_colargmin_i32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_colargmin_i32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_colargmin_i32_parallel(
    x: *const f32,
    out: *mut i32,
    rows: i64,
    cols: i64,
) {
    colarg_par(x, out, rows, cols, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent naive per-column reference: for each column `j`, `best=0; for i in 1..rows { if
    /// x[i,j] {>,<} x[best,j] { best=i } }` — STRICT compare, so the lowest row index wins on a tie
    /// (argmax uses `>`, argmin `<`). Computed without touching the kernel so it is a true oracle.
    fn naive(x: &[f32], rows: usize, cols: usize, is_max: bool) -> Vec<i32> {
        let mut out = vec![0i32; cols];
        for (j, o) in out.iter_mut().enumerate() {
            let mut best = 0usize;
            for i in 1..rows {
                let better = if is_max {
                    x[i * cols + j] > x[best * cols + j]
                } else {
                    x[i * cols + j] < x[best * cols + j]
                };
                if better {
                    best = i;
                }
            }
            *o = best as i32;
        }
        out
    }

    /// Deterministic, varied integer-valued input (no RNG) so ties are EXACT. Integer-valued f32s mean
    /// equal extrema compare bit-equal, which is what pins the lowest-index tie-break. A small range
    /// (`-50..=50`) → many exact duplicates within a column, so ties happen naturally and the
    /// lowest-row-index rule is exercised everywhere, not only in the planted columns.
    fn fill(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|t| (((t * 31 + 7) % 101) as i32 - 50) as f32)
            .collect()
    }

    /// The column arg-reductions (AVX2 serial and the multicore stripe split) must equal the naive
    /// strided scan **exactly** — same i-ascending per-column order, same strict compare, same
    /// lowest-row-index tie-break — so it is literal bit equality (an index, no arithmetic). Shapes
    /// straddle the 8-column edge and the parallel column threshold (`COLARG_PAR_MIN = 256`).
    #[test]
    fn colarg_matches_naive_and_parallel() {
        for (rows, cols) in [
            (1usize, 1usize),
            (7, 3),
            (8, 8),
            (16, 5),
            (33, 9),
            (100, 16),
            (500, 300), // > COLARG_PAR_MIN so the multicore stripes really run
        ] {
            let x = fill(rows, cols);
            for is_max in [true, false] {
                let want = naive(&x, rows, cols, is_max);

                let mut got = vec![0i32; cols];
                let mut got_par = vec![0i32; cols];
                let (f, fp): (
                    unsafe extern "C" fn(*const f32, *mut i32, i64, i64),
                    unsafe extern "C" fn(*const f32, *mut i32, i64, i64),
                ) = if is_max {
                    (wukong_colargmax_i32, wukong_colargmax_i32_parallel)
                } else {
                    (wukong_colargmin_i32, wukong_colargmin_i32_parallel)
                };
                unsafe {
                    f(x.as_ptr(), got.as_mut_ptr(), rows as i64, cols as i64);
                    fp(x.as_ptr(), got_par.as_mut_ptr(), rows as i64, cols as i64);
                }
                assert_eq!(got, want, "is_max={is_max} {rows}x{cols} vs naive");
                assert_eq!(got, got_par, "is_max={is_max} serial vs parallel {rows}x{cols}");
            }
        }
    }

    /// Pin the lowest-row-index tie-break directly in EVERY path (serial scalar/AVX2 + parallel):
    ///  - a column whose max appears at rows 2 AND 5 must return 2 (not 5),
    ///  - the same for argmin with the min at rows 2 AND 5,
    ///  - an all-equal column must return 0.
    /// `cols = 300 > COLARG_PAR_MIN` so the parallel path's stripe split actually runs, and the planted
    /// columns sit at different lanes/stripes; `rows = 8` so the planted low row (2) must survive all the
    /// blend updates across rows 3..7.
    #[test]
    fn duplicate_extrema_return_lowest_row_index() {
        let rows = 8usize;
        let cols = 300usize; // > COLARG_PAR_MIN, spans many 8-column blocks + a tail (300 = 37*8 + 4)
        let mut x = vec![0.0f32; rows * cols];

        // Column 0: all-equal → argmax and argmin both row 0.
        for i in 0..rows {
            x[i * cols + 0] = 3.0;
        }
        // Column 1: baseline 0, the MAX 9.0 planted at rows 2 and 5 → argmax must be 2; argmin (all the
        // 0.0 baseline, first at row 0, and 9.0 is not a min) → row 0.
        for i in 0..rows {
            x[i * cols + 1] = 0.0;
        }
        x[2 * cols + 1] = 9.0;
        x[5 * cols + 1] = 9.0;
        // Column 2: baseline 0, the MIN -9.0 planted at rows 2 and 5 → argmin must be 2; argmax → row 0.
        for i in 0..rows {
            x[i * cols + 2] = 0.0;
        }
        x[2 * cols + 2] = -9.0;
        x[5 * cols + 2] = -9.0;
        // Column 297 (a high column, lands in the last stripe / a late 8-block): same dup-max-at-2-and-5
        // so the tie-break is exercised away from column 0 too.
        let jc = 297usize;
        for i in 0..rows {
            x[i * cols + jc] = 1.0;
        }
        x[2 * cols + jc] = 7.0;
        x[5 * cols + jc] = 7.0;

        let mut amax = vec![0i32; cols];
        let mut amax_p = vec![0i32; cols];
        let mut amin = vec![0i32; cols];
        let mut amin_p = vec![0i32; cols];
        unsafe {
            wukong_colargmax_i32(x.as_ptr(), amax.as_mut_ptr(), rows as i64, cols as i64);
            wukong_colargmax_i32_parallel(x.as_ptr(), amax_p.as_mut_ptr(), rows as i64, cols as i64);
            wukong_colargmin_i32(x.as_ptr(), amin.as_mut_ptr(), rows as i64, cols as i64);
            wukong_colargmin_i32_parallel(x.as_ptr(), amin_p.as_mut_ptr(), rows as i64, cols as i64);
        }

        // argmax: col0 all-equal → 0; col1 dup max at 2 & 5 → 2; col2 (baseline 0, min planted) → max
        // 0.0 first at row 0 → 0; col297 dup max at 2 & 5 → 2.
        assert_eq!(amax[0], 0, "argmax all-equal column → row 0 (serial)");
        assert_eq!(amax[1], 2, "argmax dup max at rows 2 & 5 → 2 (serial)");
        assert_eq!(amax[2], 0, "argmax of min-planted column → row 0 (serial)");
        assert_eq!(amax[jc], 2, "argmax dup max at rows 2 & 5, high column → 2 (serial)");
        assert_eq!(amax_p, amax, "argmax lowest-row-index (parallel == serial)");

        // argmin: col0 all-equal → 0; col1 (baseline 0, max planted) → min 0.0 first at row 0 → 0;
        // col2 dup min at 2 & 5 → 2; col297 (baseline 1.0, max planted) → min 1.0 first at row 0 → 0.
        assert_eq!(amin[0], 0, "argmin all-equal column → row 0 (serial)");
        assert_eq!(amin[1], 0, "argmin of max-planted column → row 0 (serial)");
        assert_eq!(amin[2], 2, "argmin dup min at rows 2 & 5 → 2 (serial)");
        assert_eq!(amin[jc], 0, "argmin of max-planted high column → row 0 (serial)");
        assert_eq!(amin_p, amin, "argmin lowest-row-index (parallel == serial)");
    }

    /// The AVX2 twin holds the winning row index in an f32 lane, so it is exact only while
    /// `rows < 2^24`. Pin the dispatch guard that keeps that precondition true: with `rows = 2^24 + 2`
    /// the unguarded vector path rounded row 16777217 down to 16777216 and returned an index one short
    /// of the true answer, while `colarg_scalar` (exact integers) returned 16777217 — a CPU-dependent
    /// wrong answer. One column keeps the buffer at ~67 MB and still crosses the bound, because the
    /// f32 index is used by the AVX2 per-row tail as well as by its 8-lane body.
    #[test]
    fn rows_past_the_f32_mantissa_limit_keep_the_exact_row_index() {
        let rows = (1usize << 24) + 2;
        let cols = 1usize;
        let mut x = vec![0.0f32; rows * cols];
        x[(rows - 1) * cols] = 1.0; // unique max at row 16777217 (> 2^24)
        let mut got = vec![0i32; cols];
        let mut want = vec![0i32; cols];
        unsafe {
            wukong_colargmax_i32(x.as_ptr(), got.as_mut_ptr(), rows as i64, cols as i64);
            colarg_scalar(x.as_ptr(), want.as_mut_ptr(), rows, cols, 0, cols, true);
        }
        assert_eq!(want, vec![(rows - 1) as i32], "scalar twin lost the exact row index");
        assert_eq!(got, want, "colargmax dispatch != scalar twin past 2^24 rows");
    }

    /// Where AVX2 is available, the vector path must equal the scalar twin EXACTLY for every column,
    /// across widths spanning the 8-lane edge and tails (and a few row counts) — the column analogue of
    /// the rowarg scalar==avx2 pin. (i32 indices → exact equality, no tolerance.)
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for &cols in &[1usize, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 257] {
            for &rows in &[1usize, 2, 7, 33, 100] {
                let x = fill(rows, cols);
                for is_max in [true, false] {
                    let mut s = vec![0i32; cols];
                    let mut v = vec![0i32; cols];
                    unsafe {
                        colarg_scalar(x.as_ptr(), s.as_mut_ptr(), rows, cols, 0, cols, is_max);
                        colarg_avx2(x.as_ptr(), v.as_mut_ptr(), rows, cols, 0, cols, is_max);
                    }
                    assert_eq!(s, v, "scalar != avx2 at rows={rows} cols={cols} is_max={is_max}");
                }
            }
        }
    }
}
