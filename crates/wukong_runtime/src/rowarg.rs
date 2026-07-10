//! Per-row arg-reductions along the **inner** (last) axis of a `[rows, cols]` row-major matrix,
//! producing a per-row `[rows]` **index**: `out[r] = argmax_j x[r, j]` (and the argmin twin). This is
//! the classification-head / greedy-decode **top-1** — reduce-along-the-last-axis returning the index
//! of the largest (smallest) element of each row (the logits→token-id step, the per-example predicted
//! class). It is the per-row sibling of the whole-array [`crate::wukong_argreduce_f32`]: the same
//! `(value, index)` candidate with **lowest index winning on a value tie**, applied independently to
//! each row's `cols` elements.
//!
//! **Tie-break is the whole game.** Among equal extrema the SMALLEST `j` wins (standard
//! argmax/PyTorch semantics). That rule must be identical in the scalar twin, the AVX2 path, and
//! across serial/parallel — it is what makes the kernel its own bit-exact oracle. We pin it the same
//! way `argreduce` does: a **strict** value compare (`>` for argmax, `<` for argmin) so a later equal
//! value never displaces the earlier index, and an explicit lower-index rule on a true tie. The AVX2
//! path tracks a running max-vector + an index-vector (lane base `+ 0..7`): a lane updates only where
//! the new value **strictly** beats its running max (so on a tie the lane keeps its earlier, lower
//! index), then the eight lane candidates collapse left-to-right with the same lowest-index
//! [`arg_fold`] — so lane 0 (the lowest indices) is preferred through the horizontal combine, exactly
//! like the scalar ascending scan. A scalar tail finishes the `cols % 8` remainder.
//!
//! Rows are independent, so `_parallel` just maps the per-row routine across rows (each writes a
//! disjoint `out[r]`): serial == parallel bit-for-bit, no cross-row combine, the same play as the
//! per-row norms and the column reductions.

use crate::reduce::arg_fold;

/// Rows below which the parallel entry just runs serially (per-row work is cheap; the rayon
/// fan-out only pays off once there are enough rows to spread). Matches the small-input serial
/// thresholds the other parallel kernels use.
const ROWARG_PAR_MIN: usize = 64;

/// Scalar per-row argmax/argmin reference: a plain ascending scan with a **strict** value compare so
/// the lowest index wins on a tie (`best=0; for j in 1..cols { if x[j] {>,<} x[best] { best=j } }`).
/// This is the no-AVX2 fallback *and* the bit-exact oracle the AVX2 path must match lane-for-lane.
///
/// # Safety
/// `row` valid for `cols` `f32` reads; `cols >= 1`.
#[inline]
unsafe fn rowarg_scalar(row: *const f32, cols: usize, is_max: bool) -> i32 {
    // Seed with element 0 (index 0). Folding `j in 1..cols` with `arg_fold` gives exactly the strict
    // ">"/"<" tie-break: a later equal value reports `b_better == false` and `b.1 > a.1`, so `a`
    // (the lower index) is kept.
    let mut best = (*row, 0usize);
    let mut j = 1usize;
    while j < cols {
        best = arg_fold(best, (*row.add(j), j), is_max);
        j += 1;
    }
    best.1 as i32
}

/// AVX2 per-row argmax/argmin: eight running `(value, index)` lanes (lane `l` scans positions
/// `≡ l (mod 8)`), then a fixed left-to-right collapse + scalar tail. Bit-identical to
/// [`rowarg_scalar`]: within a lane a strict `_mm256_cmp_ps(value, running, GT/LT)` mask updates the
/// value and index only on a STRICT win, so a tie keeps the lane's earlier (lower) index; the
/// horizontal collapse folds lanes ascending with the lowest-index [`arg_fold`]; the tail scans the
/// final `cols % 8` in ascending order.
///
/// # Safety
/// `row` valid for `cols` `f32` reads; `cols >= 1`; AVX2 must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rowarg_avx2(row: *const f32, cols: usize, is_max: bool) -> i32 {
    use std::arch::x86_64::*;

    // Below one full vector there is nothing for the lanes to do — go straight to the scalar scan
    // (also covers the lowest-index seed cleanly).
    if cols < 8 {
        return rowarg_scalar(row, cols, is_max);
    }

    let nsteps = cols / 8; // full 8-wide steps; the [nsteps*8, cols) remainder is the scalar tail.

    // Seed the lanes from the first step (positions 0..7), indices 0..7. Seeding from real data (not
    // ±∞) means every lane already holds a valid candidate, so the strict-compare updates below need
    // no "is this lane still empty?" special-casing — exactly the ascending-scan seed of the scalar.
    let mut best_val = _mm256_loadu_ps(row);
    // Lane indices as f32: 0,1,..,7. cols fits in i32 for any realistic logits row, and these indices
    // are small integers (< cols) exactly representable in f32, so the index bookkeeping is exact.
    let mut best_idx = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    let step8 = _mm256_set1_ps(8.0);
    let mut cur_idx = best_idx; // running per-lane index for the current step

    for s in 1..nsteps {
        cur_idx = _mm256_add_ps(cur_idx, step8); // advance lane indices by 8
        let v = _mm256_loadu_ps(row.add(s * 8));
        // STRICT compare so a tie does NOT update (keeps the earlier, lower index in each lane):
        // argmax updates where v > best_val; argmin where v < best_val.
        let mask = if is_max {
            _mm256_cmp_ps(v, best_val, _CMP_GT_OQ)
        } else {
            _mm256_cmp_ps(v, best_val, _CMP_LT_OQ)
        };
        // blendv(a, b, mask) = mask ? b : a — take the new value/index only in the strictly-winning
        // lanes, keep the running best (and its lower index) elsewhere.
        best_val = _mm256_blendv_ps(best_val, v, mask);
        best_idx = _mm256_blendv_ps(best_idx, cur_idx, mask);
    }

    // Collapse the 8 lane candidates left-to-right (lane 0 → 7) with the SAME lowest-index tie-break
    // as the scalar horizontal combine: on an all-equal row, lane 0 holds the lowest index and the
    // ascending `arg_fold` keeps it.
    let mut vbuf = [0.0f32; 8];
    let mut ibuf = [0.0f32; 8];
    _mm256_storeu_ps(vbuf.as_mut_ptr(), best_val);
    _mm256_storeu_ps(ibuf.as_mut_ptr(), best_idx);
    let mut best = (vbuf[0], ibuf[0] as usize);
    for l in 1..8 {
        best = arg_fold(best, (vbuf[l], ibuf[l] as usize), is_max);
    }

    // Scalar tail for the final `cols % 8` elements, scanned ascending so a tie keeps the earlier
    // index — identical to the scalar twin's tail.
    let mut j = nsteps * 8;
    while j < cols {
        best = arg_fold(best, (*row.add(j), j), is_max);
        j += 1;
    }
    best.1 as i32
}

/// Dispatch AVX2 vs scalar for one row.
///
/// # Safety
/// `row` valid for `cols` `f32`; `cols >= 1`.
#[inline]
unsafe fn rowarg_one(row: *const f32, cols: usize, is_max: bool) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return rowarg_avx2(row, cols, is_max);
        }
    }
    rowarg_scalar(row, cols, is_max)
}

/// Serial per-row arg-reduction: `out[r] = {argmax,argmin}_j x[r, j]` for every row.
///
/// # Safety
/// `x` valid for `rows*cols` `f32`; `out` valid for `rows` `i32`; non-overlapping.
#[inline]
unsafe fn rowarg_serial(x: *const f32, out: *mut i32, rows: i64, cols: i64, is_max: bool) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    for i in 0..r {
        // SAFETY: row i occupies x[i*c .. i*c + c); out[i] in bounds.
        *out.add(i) = rowarg_one(x.add(i * c), c, is_max);
    }
}

/// Parallel per-row arg-reduction — **bit-identical** to [`rowarg_serial`]. Rows are independent and
/// each writes a disjoint `out[r]`, so mapping the per-row routine across cores reorders nothing: the
/// per-row result (including the lowest-index tie-break) is the same regardless of the split. Below
/// [`ROWARG_PAR_MIN`] rows it runs serial.
///
/// # Safety
/// `x` valid for `rows*cols` `f32`; `out` valid for `rows` `i32`; non-overlapping.
#[inline]
unsafe fn rowarg_par(x: *const f32, out: *mut i32, rows: i64, cols: i64, is_max: bool) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r < ROWARG_PAR_MIN {
        rowarg_serial(x, out, rows, cols, is_max);
        return;
    }
    use rayon::prelude::*;
    // Raw pointers cross the rayon closure boundary as integers (the same pattern as the parallel
    // GEMM / reductions); every task reads a disjoint row of `x` and writes a disjoint `out[i]`.
    let (x_addr, out_addr) = (x as usize, out as usize);
    (0..r).into_par_iter().for_each(|i| {
        // SAFETY: disjoint row read + disjoint out[i] write; pointers re-derived from the addresses.
        unsafe {
            let xi = (x_addr as *const f32).add(i * c);
            *(out_addr as *mut i32).add(i) = rowarg_one(xi, c, is_max);
        }
    });
}

/// `out[r] = argmax_j x[r, j]` over a `[rows, cols]` row-major matrix, single-threaded — the
/// classification-head / greedy-decode **top-1** (largest-logit index per row). Lowest index wins on
/// a value tie.
///
/// # Safety
/// `x` valid for `rows*cols` `f32`; `out` valid for `rows` `i32`; non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn wukong_rowargmax_i32(x: *const f32, out: *mut i32, rows: i64, cols: i64) {
    rowarg_serial(x, out, rows, cols, true);
}

/// Multi-threaded `out[r] = argmax_j x[r, j]` (bit-identical to [`wukong_rowargmax_i32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_rowargmax_i32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_rowargmax_i32_parallel(
    x: *const f32,
    out: *mut i32,
    rows: i64,
    cols: i64,
) {
    rowarg_par(x, out, rows, cols, true);
}

/// `out[r] = argmin_j x[r, j]` over a `[rows, cols]` row-major matrix, single-threaded (per-row
/// minimum index). Lowest index wins on a value tie.
///
/// # Safety
/// `x` valid for `rows*cols` `f32`; `out` valid for `rows` `i32`; non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn wukong_rowargmin_i32(x: *const f32, out: *mut i32, rows: i64, cols: i64) {
    rowarg_serial(x, out, rows, cols, false);
}

/// Multi-threaded `out[r] = argmin_j x[r, j]` (bit-identical to [`wukong_rowargmin_i32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_rowargmin_i32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_rowargmin_i32_parallel(
    x: *const f32,
    out: *mut i32,
    rows: i64,
    cols: i64,
) {
    rowarg_par(x, out, rows, cols, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent naive per-row reference: `best=0; for j in 1..cols { if x[r,j] {>,<} x[r,best] {
    /// best=j } }` — STRICT compare, so the lowest index wins on a tie (argmax uses `>`, argmin `<`).
    /// Computed without touching the kernel so it is a true oracle.
    fn naive(x: &[f32], rows: usize, cols: usize, is_max: bool) -> Vec<i32> {
        let mut out = vec![0i32; rows];
        for (r, o) in out.iter_mut().enumerate() {
            let base = r * cols;
            let mut best = 0usize;
            for j in 1..cols {
                let better = if is_max {
                    x[base + j] > x[base + best]
                } else {
                    x[base + j] < x[base + best]
                };
                if better {
                    best = j;
                }
            }
            *o = best as i32;
        }
        out
    }

    /// Deterministic, varied integer-valued input (no RNG) so ties between rows are EXACT. Integer
    /// f32s mean equal maxima compare bit-equal, which is what pins the tie-break.
    fn fill(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            // A small integer range (-50..=50) → many exact duplicates within a row, so ties happen
            // naturally and the lowest-index rule is exercised everywhere, not just in the planted rows.
            .map(|t| (((t * 31 + 7) % 101) as i32 - 50) as f32)
            .collect()
    }

    #[test]
    fn rowarg_matches_naive_and_parallel() {
        // Shapes straddle the 8-lane edge (1,7,8,16,33,100,1000 cols) and the parallel row threshold
        // (1..40 rows; 40 < ROWARG_PAR_MIN so the small ones exercise the serial fallback, but we also
        // assert serial == parallel for ALL of them).
        for (rows, cols) in [
            (1usize, 1usize),
            (3, 7),
            (4, 8),
            (5, 16),
            (9, 33),
            (16, 100),
            (40, 1000),
        ] {
            let x = fill(rows, cols);
            for is_max in [true, false] {
                let want = naive(&x, rows, cols, is_max);

                let mut got = vec![0i32; rows];
                let mut got_par = vec![0i32; rows];
                let (f, fp): (
                    unsafe extern "C" fn(*const f32, *mut i32, i64, i64),
                    unsafe extern "C" fn(*const f32, *mut i32, i64, i64),
                ) = if is_max {
                    (wukong_rowargmax_i32, wukong_rowargmax_i32_parallel)
                } else {
                    (wukong_rowargmin_i32, wukong_rowargmin_i32_parallel)
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

    #[test]
    fn duplicate_extrema_return_lowest_index() {
        // Pin the lowest-index tie-break directly in EVERY path (serial scalar/AVX2 + parallel):
        //  - an all-equal row must return index 0,
        //  - a row whose max appears at indices 2 AND 5 must return 2 (not 5),
        //  - the same for argmin with the min at 2 AND 5.
        // Wide enough (cols = 20) to span a full 8-lane step + a partial step + a tail, so the planted
        // duplicate at a low index must survive the lane updates, the horizontal collapse, and the tail.
        let cols = 20usize;
        let rows = 3usize;
        let mut x = vec![0.0f32; rows * cols];

        // Row 0: all equal → argmax and argmin both 0.
        for j in 0..cols {
            x[j] = 3.0;
        }
        // Row 1: baseline 0, the MAX 9.0 planted at indices 2 and 5 → argmax must be 2.
        let r1 = cols;
        x[r1 + 2] = 9.0;
        x[r1 + 5] = 9.0;
        // Row 2: baseline 0, the MIN -9.0 planted at indices 2 and 5 → argmin must be 2.
        let r2 = 2 * cols;
        x[r2 + 2] = -9.0;
        x[r2 + 5] = -9.0;

        let mut amax = vec![0i32; rows];
        let mut amax_p = vec![0i32; rows];
        let mut amin = vec![0i32; rows];
        let mut amin_p = vec![0i32; rows];
        unsafe {
            wukong_rowargmax_i32(x.as_ptr(), amax.as_mut_ptr(), rows as i64, cols as i64);
            wukong_rowargmax_i32_parallel(x.as_ptr(), amax_p.as_mut_ptr(), rows as i64, cols as i64);
            wukong_rowargmin_i32(x.as_ptr(), amin.as_mut_ptr(), rows as i64, cols as i64);
            wukong_rowargmin_i32_parallel(x.as_ptr(), amin_p.as_mut_ptr(), rows as i64, cols as i64);
        }

        // argmax: row0 all-equal → 0; row1 dup max at 2 & 5 → 2; row2 (baseline 0, min planted) → max
        // is 0.0 first seen at index 0 → 0.
        assert_eq!(amax, vec![0, 2, 0], "argmax lowest-index (serial)");
        assert_eq!(amax_p, amax, "argmax lowest-index (parallel == serial)");
        // argmin: row0 all-equal → 0; row1 (baseline 0, max planted) → min is 0.0 first at 0 → 0;
        // row2 dup min at 2 & 5 → 2.
        assert_eq!(amin, vec![0, 0, 2], "argmin lowest-index (serial)");
        assert_eq!(amin_p, amin, "argmin lowest-index (parallel == serial)");
    }

    #[test]
    fn duplicate_extrema_across_lanes_and_parallel_threshold() {
        // The same lowest-index guarantee, but with MANY rows (> ROWARG_PAR_MIN) so the rayon fan-out
        // really runs, and a wide row so the duplicate max straddles distinct 8-lane *lanes* (index 3
        // is lane 3 step 0; index 11 is lane 3 step 1 — same lane, so the lane keeps the lower; index
        // 4 is lane 4 — a different lane, exercising the horizontal collapse). The answer must still be
        // the single lowest index, identical serial and parallel.
        let rows = 200usize; // > ROWARG_PAR_MIN
        let cols = 64usize;
        let mut x = vec![1.0f32; rows * cols];
        // Plant the max 7.0 at indices 3, 4, 11, 40 in every row → lowest is 3.
        for r in 0..rows {
            let b = r * cols;
            for &j in &[3usize, 4, 11, 40] {
                x[b + j] = 7.0;
            }
        }
        let mut got = vec![0i32; rows];
        let mut got_p = vec![0i32; rows];
        unsafe {
            wukong_rowargmax_i32(x.as_ptr(), got.as_mut_ptr(), rows as i64, cols as i64);
            wukong_rowargmax_i32_parallel(x.as_ptr(), got_p.as_mut_ptr(), rows as i64, cols as i64);
        }
        assert!(got.iter().all(|&v| v == 3), "every row argmax == lowest dup index 3 (serial)");
        assert_eq!(got, got_p, "serial == parallel across the rayon threshold");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        // Where AVX2 is available, the vector path must equal the scalar twin EXACTLY for every row,
        // across widths spanning the 8-lane edge and tails — the per-row analogue of the argreduce
        // scalar==avx2 pin. (i32 indices → exact equality, no tolerance.)
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for &cols in &[1usize, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 1000] {
            let x = fill(1, cols);
            for is_max in [true, false] {
                let s = unsafe { rowarg_scalar(x.as_ptr(), cols, is_max) };
                let v = unsafe { rowarg_avx2(x.as_ptr(), cols, is_max) };
                assert_eq!(s, v, "scalar != avx2 at cols={cols} is_max={is_max}");
            }
        }
    }
}
