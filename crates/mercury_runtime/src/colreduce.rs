//! Column reductions along the **outer** (batch / row) axis of a `[rows, cols]` row-major matrix,
//! producing a per-column `[cols]` result: the **sum** `out[j] = Σ_i x[i,j]` (bias gradient
//! `db = Σ_batch dY`, batch sum, reduce-along-axis-0), the **max** `out[j] = max_i x[i,j]`, and the
//! **min** `out[j] = min_i x[i,j]` (per-channel statistics for quantization, axis-0 max/min pooling).
//!
//! The naive spelling `for j { for i { s ⊕= x[i*N+j] } }` reads `x` with stride `N` — a *strided
//! reduction* gcc/rustc do **not** vectorize (verified: scalar `vaddss`/`vmaxss`, no packed
//! `vaddps`/`vmaxps`, at `-O3 -march=native`). These kernels instead stream `x` **row-major** and fold
//! 8 columns at a time into a cache-resident `out[]`, so they are SIMD *and* cache-friendly. The fold
//! order is **i-ascending per column** — identical to the scalar nest — in the AVX2 and scalar paths
//! and across serial/parallel (disjoint column stripes, no cross-stripe combine), so each kernel is
//! its own bit-exact oracle (the interpreter marshals the serial form; the differential gate compares
//! interp vs native, both folding the same `_mm256_add_ps`/`_mm256_max_ps`/`_mm256_min_ps` lane tree).

/// Which column reduction to fold: `Sum` seeds `0` and folds rows `[0, rows)`; `Max`/`Min`/`MaxAbs`
/// seed the **first row** (`x[0,j]`, or `|x[0,j]|` for `MaxAbs`) and fold rows `[1, rows)` (idempotent,
/// so equivalent to folding from 0). `MaxAbs` is the per-channel symmetric-quantization scale
/// `max_i |x[i,j]|`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColKind {
    Sum,
    Max,
    Min,
    MaxAbs,
}

/// Fold rows `[0, rows)` of `x` into the column range `[j0, j1)` of `out` (`out[j] = ⊕_i x[i*cols+j]`),
/// AVX2, 8 columns/step. Seeds the range (0 for `Sum`, the first row for `Max`/`Min`), then streams `x`
/// row-major folding into `out`. The per-`kind` inner loop is selected **once per call** (no per-element
/// branch); each fold is the exact `_mm256_{add,max,min}_ps` whose scalar twin is in [`colreduce_scalar`].
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
    // Seed: Sum -> 0; Max/Min -> first row x[0, j]; MaxAbs -> |x[0, j]| (so the fold starts at row 1).
    let i0 = match kind {
        ColKind::Sum => {
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
    // The fold loop, monomorphic per kind (the match is hoisted out of the row loop).
    match kind {
        ColKind::Sum => {
            for i in i0..rows {
                let xr = x.add(i * cols);
                let mut j = j0;
                while j + 8 <= j1 {
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
        ColKind::Max => {
            for i in i0..rows {
                let xr = x.add(i * cols);
                let mut j = j0;
                while j + 8 <= j1 {
                    let acc = _mm256_loadu_ps(out.add(j));
                    let v = _mm256_loadu_ps(xr.add(j));
                    _mm256_storeu_ps(out.add(j), _mm256_max_ps(acc, v));
                    j += 8;
                }
                while j < j1 {
                    let a = *out.add(j);
                    let v = *xr.add(j);
                    *out.add(j) = if a > v { a } else { v };
                    j += 1;
                }
            }
        }
        ColKind::Min => {
            for i in i0..rows {
                let xr = x.add(i * cols);
                let mut j = j0;
                while j + 8 <= j1 {
                    let acc = _mm256_loadu_ps(out.add(j));
                    let v = _mm256_loadu_ps(xr.add(j));
                    _mm256_storeu_ps(out.add(j), _mm256_min_ps(acc, v));
                    j += 8;
                }
                while j < j1 {
                    let a = *out.add(j);
                    let v = *xr.add(j);
                    *out.add(j) = if a < v { a } else { v };
                    j += 1;
                }
            }
        }
        ColKind::MaxAbs => {
            for i in i0..rows {
                let xr = x.add(i * cols);
                let mut j = j0;
                while j + 8 <= j1 {
                    let acc = _mm256_loadu_ps(out.add(j));
                    // |v| = andnot(-0.0, v) (clear the sign bit), then fold by max — the same
                    // sign-mask abs the RED_MAXABS reduction uses, so it agrees lane-for-lane.
                    let v = _mm256_andnot_ps(sign, _mm256_loadu_ps(xr.add(j)));
                    _mm256_storeu_ps(out.add(j), _mm256_max_ps(acc, v));
                    j += 8;
                }
                while j < j1 {
                    let a = *out.add(j);
                    let v = (*xr.add(j)).abs();
                    *out.add(j) = if a > v { a } else { v };
                    j += 1;
                }
            }
        }
    }
}

/// Scalar twin of [`colreduce_avx2`] (the no-AVX2 fallback and the bit-exact reference). Same
/// i-ascending per-column order and the same fold (`a > v ? a : v` == `_mm256_max_ps(a, v)`,
/// `a < v ? a : v` == `_mm256_min_ps(a, v)` on the non-NaN data the recognizer targets), so it agrees
/// with the AVX2 path lane-for-lane.
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
        ColKind::Sum => {
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
    for i in i0..rows {
        let xr = x.add(i * cols);
        for j in j0..j1 {
            let a = *out.add(j);
            let v = *xr.add(j);
            *out.add(j) = match kind {
                ColKind::Sum => a + v,
                ColKind::Max => {
                    if a > v {
                        a
                    } else {
                        v
                    }
                }
                ColKind::Min => {
                    if a < v {
                        a
                    } else {
                        v
                    }
                }
                ColKind::MaxAbs => {
                    let v = v.abs();
                    if a > v {
                        a
                    } else {
                        v
                    }
                }
            };
        }
    }
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
            colreduce_range(x_addr as *const f32, out_addr as *mut f32, r, c, j0, j1, kind);
        }
    });
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
    colreduce_range(x, out, rows as usize, cols as usize, 0, cols as usize, ColKind::Sum);
}

/// Multi-threaded `out[j] = Σ_i x[i, j]` (bit-identical to [`mercury_colsum_f32`]).
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
    colreduce_parallel(x, out, rows, cols, ColKind::Sum);
}

/// `out[j] = max_i x[i, j]` over a `[rows, cols]` row-major matrix, single-threaded (per-channel max /
/// axis-0 max-pool). Seeds from the first row, so an empty `rows == 0` writes nothing.
///
/// # Safety
/// `x` valid for `rows*cols`, `out` for `cols` f32, non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn mercury_colmax_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    colreduce_range(x, out, rows as usize, cols as usize, 0, cols as usize, ColKind::Max);
}

/// Multi-threaded `out[j] = max_i x[i, j]` (bit-identical to [`mercury_colmax_f32`]).
///
/// # Safety
/// Operand-size contract of [`mercury_colmax_f32`].
#[no_mangle]
pub unsafe extern "C" fn mercury_colmax_f32_parallel(
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
pub unsafe extern "C" fn mercury_colmin_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    colreduce_range(x, out, rows as usize, cols as usize, 0, cols as usize, ColKind::Min);
}

/// Multi-threaded `out[j] = min_i x[i, j]` (bit-identical to [`mercury_colmin_f32`]).
///
/// # Safety
/// Operand-size contract of [`mercury_colmin_f32`].
#[no_mangle]
pub unsafe extern "C" fn mercury_colmin_f32_parallel(
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
pub unsafe extern "C" fn mercury_colmaxabs_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    colreduce_range(x, out, rows as usize, cols as usize, 0, cols as usize, ColKind::MaxAbs);
}

/// Multi-threaded `out[j] = max_i |x[i, j]|` (bit-identical to [`mercury_colmaxabs_f32`]).
///
/// # Safety
/// Operand-size contract of [`mercury_colmaxabs_f32`].
#[no_mangle]
pub unsafe extern "C" fn mercury_colmaxabs_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    colreduce_parallel(x, out, rows, cols, ColKind::MaxAbs);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(x: &[f32], rows: usize, cols: usize, kind: ColKind) -> Vec<f32> {
        let mut out = vec![0.0f32; cols];
        for (j, o) in out.iter_mut().enumerate() {
            let mut s = match kind {
                ColKind::Sum => 0.0f32,
                ColKind::MaxAbs => x[j].abs(), // |first row|
                _ => x[j],                     // first row
            };
            let i0 = if kind == ColKind::Sum { 0 } else { 1 };
            for i in i0..rows {
                let v = x[i * cols + j];
                s = match kind {
                    ColKind::Sum => s + v,
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
            *o = s;
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
            (5, 7),
            (8, 8),
            (33, 17),
            (64, 100),
            (128, 257),
            (50, 1000), // > COLREDUCE_PAR_MIN so the multicore stripes run
        ] {
            // Distinct values per cell so max/min have a unique answer; an exact integer range so a
            // reordering (if any path had one) would show as a bit mismatch.
            let x: Vec<f32> = (0..rows * cols).map(|t| ((t * 7 + 3) % 101) as f32 - 50.0).collect();
            for kind in [ColKind::Sum, ColKind::Max, ColKind::Min, ColKind::MaxAbs] {
                let want = naive(&x, rows, cols, kind);
                let mut got = vec![0.0f32; cols];
                let mut got_par = vec![0.0f32; cols];
                let (f, fp): (
                    unsafe extern "C" fn(*const f32, *mut f32, i64, i64),
                    unsafe extern "C" fn(*const f32, *mut f32, i64, i64),
                ) = match kind {
                    ColKind::Sum => (mercury_colsum_f32, mercury_colsum_f32_parallel),
                    ColKind::Max => (mercury_colmax_f32, mercury_colmax_f32_parallel),
                    ColKind::Min => (mercury_colmin_f32, mercury_colmin_f32_parallel),
                    ColKind::MaxAbs => (mercury_colmaxabs_f32, mercury_colmaxabs_f32_parallel),
                };
                unsafe {
                    f(x.as_ptr(), got.as_mut_ptr(), rows as i64, cols as i64);
                    fp(x.as_ptr(), got_par.as_mut_ptr(), rows as i64, cols as i64);
                }
                assert_eq!(got, want, "{:?} {rows}x{cols} vs naive", kind as u8);
                assert_eq!(got, got_par, "{:?} serial vs parallel {rows}x{cols}", kind as u8);
            }
        }
    }
}
