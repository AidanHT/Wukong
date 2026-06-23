//! Column reduction `out[j] = Σ_i x[i, j]` — the sum over the **outer** (batch / row) axis of a
//! `[rows, cols]` row-major matrix, producing a per-column `[cols]` result. This is the bias gradient
//! `db = Σ_batch dY`, the batch sum/mean, and the general reduce-along-axis-0.
//!
//! The naive spelling `for j { for i { s += x[i*N+j] } }` reads `x` with stride `N` — a *strided
//! reduction* gcc/rustc do **not** vectorize (verified: scalar `vaddss`, no packed `vaddps`, at
//! `-O3 -march=native`). This kernel instead streams `x` **row-major** and accumulates 8 columns at a
//! time into a cache-resident `out[]`, so it is SIMD *and* cache-friendly. The accumulation order is
//! **i-ascending per column** — identical to the scalar nest — so the kernel is **bit-exact** with it
//! (no reassociation: each `out[j]` sums `x[0,j], x[1,j], …` in order, in both the AVX2 and scalar
//! paths and across serial/parallel), a stronger gate than the float-reduction kernels' tolerance.

/// Sum rows `[0, rows)` into the column range `[j0, j1)` of `out` (`out[j] = Σ_i x[i*cols + j]`),
/// AVX2, 8 columns/step. Zeros the range first, then streams `x` row-major accumulating into `out`.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32; `j0 <= j1 <= cols`; AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn colsum_avx2(x: *const f32, out: *mut f32, rows: usize, cols: usize, j0: usize, j1: usize) {
    use std::arch::x86_64::*;
    for j in j0..j1 {
        *out.add(j) = 0.0;
    }
    for i in 0..rows {
        let xr = x.add(i * cols);
        let mut j = j0;
        while j + 8 <= j1 {
            // out[j..j+8] += x[i, j..j+8] — 8 independent column accumulators, each in i-order.
            let acc = _mm256_loadu_ps(out.add(j));
            let v = _mm256_loadu_ps(xr.add(j));
            _mm256_storeu_ps(out.add(j), _mm256_add_ps(acc, v));
            j += 8;
        }
        while j < j1 {
            *out.add(j) += *xr.add(j);
            j += 1;
        }
    }
}

/// Scalar twin of [`colsum_avx2`] (the no-AVX2 fallback and the bit-exact reference). Same i-ascending
/// per-column order, so it agrees with the AVX2 path lane-for-lane.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32; `j0 <= j1 <= cols`.
unsafe fn colsum_scalar(x: *const f32, out: *mut f32, rows: usize, cols: usize, j0: usize, j1: usize) {
    for j in j0..j1 {
        *out.add(j) = 0.0;
    }
    for i in 0..rows {
        let xr = x.add(i * cols);
        for j in j0..j1 {
            *out.add(j) += *xr.add(j);
        }
    }
}

/// Dispatch AVX2 vs scalar for the column range `[j0, j1)`.
///
/// # Safety
/// Operand-size contract of [`colsum_avx2`].
#[inline]
unsafe fn colsum_range(x: *const f32, out: *mut f32, rows: usize, cols: usize, j0: usize, j1: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            colsum_avx2(x, out, rows, cols, j0, j1);
            return;
        }
    }
    colsum_scalar(x, out, rows, cols, j0, j1);
}

/// `out[j] = Σ_i x[i, j]` over a `[rows, cols]` row-major matrix, single-threaded.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32, non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn mercury_colsum_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    colsum_range(x, out, rows as usize, cols as usize, 0, cols as usize);
}

/// Column count below which the parallel column-sum just runs serially.
const COLSUM_PAR_MIN: usize = 256;

/// Multi-threaded `out[j] = Σ_i x[i, j]`: the **columns** are split into disjoint stripes across cores
/// (each core sums its column range over all rows). Each stripe writes a disjoint slice of `out`, and
/// every column is summed in the same i-ascending order regardless of the split — so the result is
/// **bit-identical to the serial kernel** the interpreter marshals (no cross-stripe combine). Below
/// `COLSUM_PAR_MIN` columns it runs serial.
///
/// # Safety
/// Operand-size contract of [`mercury_colsum_f32`].
#[no_mangle]
pub unsafe extern "C" fn mercury_colsum_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if c < COLSUM_PAR_MIN {
        colsum_range(x, out, r, c, 0, c);
        return;
    }
    use rayon::prelude::*;
    // One stripe per core, each a multiple of 8 columns (keep the AVX2 main loop aligned to the
    // stripe boundary); the last stripe absorbs the remainder.
    let nthreads = rayon::current_num_threads().max(1);
    let per = (c.div_ceil(nthreads)).next_multiple_of(8).max(8);
    let nstripes = c.div_ceil(per);
    let (x_addr, out_addr) = (x as usize, out as usize);
    (0..nstripes).into_par_iter().for_each(|s| {
        let j0 = s * per;
        let j1 = (j0 + per).min(c);
        // SAFETY: disjoint out[] stripe per task; pointers re-derived from the captured addresses.
        unsafe {
            colsum_range(x_addr as *const f32, out_addr as *mut f32, r, c, j0, j1);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_colsum(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; cols];
        for (j, o) in out.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for i in 0..rows {
                s += x[i * cols + j];
            }
            *o = s;
        }
        out
    }

    /// The column-sum (AVX2 serial and the multicore stripe split) must equal the naive strided
    /// column-sum **exactly** — both accumulate each column in i-ascending order, so it is literal bit
    /// equality (no reassociation). Shapes straddle the 8-lane edge and the parallel column threshold.
    #[test]
    fn colsum_matches_naive_and_parallel() {
        for (rows, cols) in [
            (1, 1),
            (5, 7),
            (8, 8),
            (33, 17),
            (64, 100),
            (128, 257),
            (50, 1000), // > COLSUM_PAR_MIN so the multicore stripes run
        ] {
            // Values with an exact integer sum so "bit equal" is unambiguous even if a path reordered.
            let x: Vec<f32> = (0..rows * cols).map(|t| (t % 13) as f32 - 6.0).collect();
            let want = naive_colsum(&x, rows, cols);
            let mut got = vec![0.0f32; cols];
            let mut got_par = vec![0.0f32; cols];
            unsafe {
                mercury_colsum_f32(x.as_ptr(), got.as_mut_ptr(), rows as i64, cols as i64);
                mercury_colsum_f32_parallel(x.as_ptr(), got_par.as_mut_ptr(), rows as i64, cols as i64);
            }
            assert_eq!(got, want, "colsum {rows}x{cols} vs naive");
            assert_eq!(got, got_par, "colsum serial vs parallel {rows}x{cols}");
        }
    }
}
