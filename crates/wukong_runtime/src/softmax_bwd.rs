//! Softmax backward — the Jacobian-vector product of the row-wise softmax, the gradient that flows
//! through every attention block and every classification head in training:
//!
//! ```text
//! dx[r, i] = y[r, i] * (dy[r, i] - Σ_j y[r, j] * dy[r, j])
//! ```
//!
//! where `y` is the softmax *output* `[rows, cols]`, `dy` the upstream gradient, and `dx` the gradient
//! w.r.t. the softmax input. Each row needs one **dot** `s = Σ_j y[r,j]·dy[r,j]` (gcc/rustc keep this
//! reduction *scalar* — verified: a serial `vaddss` chain, no `vaddps` accumulator, at `-O3
//! -march=native`, because they will not reassociate the float sum) followed by an elementwise
//! `y·(dy − s)` over the row.
//!
//! The dot is delegated to the **identical bit-exact `wukong_sreduce_f32(RED_DOT)`** the `@parallel`
//! dot dispatch already uses (8-lane accumulators + a fixed-order horizontal combine, serial ==
//! parallel == its scalar twin), so this kernel introduces **no new reduction-accumulation order** to
//! reason about — it is the proven dot plus a pure elementwise map. Rows are independent, so the
//! `_parallel` form maps the per-row routine across cores and is **bit-identical** to the serial one
//! (no cross-row combine); the interpreter marshals the serial form, so the differential oracle stays
//! exact. (The dot's lane reassociation is the documented reassociated-reduction exception: every
//! backend runs this same kernel, so they agree.)

use crate::reduce::{wukong_sreduce_f32, RED_DOT};

/// Apply one row's elementwise tail `dx[i] = y[i] * (dy[i] - s)` (`s` the precomputed dot), AVX2,
/// 8 elements/step + a scalar tail. Pure per-lane arithmetic (a sub then a mul, no reduction, no FMA),
/// so the AVX2 path and the scalar tail are bit-identical lane-for-lane.
///
/// # Safety
/// `y`, `dy`, `dx` valid for `cols` f32; AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_row_avx2(y: *const f32, dy: *const f32, dx: *mut f32, cols: usize, s: f32) {
    use std::arch::x86_64::*;
    let sv = _mm256_set1_ps(s);
    let mut i = 0;
    while i + 8 <= cols {
        let yv = _mm256_loadu_ps(y.add(i));
        let dyv = _mm256_loadu_ps(dy.add(i));
        // y * (dy - s) — sub then mul, matching the scalar twin exactly.
        _mm256_storeu_ps(dx.add(i), _mm256_mul_ps(yv, _mm256_sub_ps(dyv, sv)));
        i += 8;
    }
    while i < cols {
        *dx.add(i) = *y.add(i) * (*dy.add(i) - s);
        i += 1;
    }
}

/// Scalar twin / no-AVX2 fallback of [`apply_row_avx2`] — same `y * (dy - s)` per element.
///
/// # Safety
/// `y`, `dy`, `dx` valid for `cols` f32.
unsafe fn apply_row_scalar(y: *const f32, dy: *const f32, dx: *mut f32, cols: usize, s: f32) {
    for i in 0..cols {
        *dx.add(i) = *y.add(i) * (*dy.add(i) - s);
    }
}

/// One row of softmax backward: the bit-exact dot `s = Σ_j y[j]·dy[j]` (delegated to
/// `wukong_sreduce_f32`) then `dx[i] = y[i]·(dy[i] − s)`.
///
/// # Safety
/// `y`, `dy`, `dx` valid for `cols` f32.
#[inline]
unsafe fn softmax_bwd_row(y: *const f32, dy: *const f32, dx: *mut f32, cols: usize) {
    let s = wukong_sreduce_f32(y, dy, cols as i64, RED_DOT);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            apply_row_avx2(y, dy, dx, cols, s);
            return;
        }
    }
    apply_row_scalar(y, dy, dx, cols, s);
}

/// `dx[r,i] = y[r,i] · (dy[r,i] − Σ_j y[r,j]·dy[r,j])` over a `[rows, cols]` row-major batch,
/// single-threaded. `dx` may alias `dy` (the dot over a row is read before that row's `dx` is written).
///
/// # Safety
/// `y`, `dy`, `dx` valid for `rows*cols` f32; `y`/`dy` not aliasing `dx` *across* rows is not required
/// (each row is self-contained), but `dx` must not overlap `y` (its row is read during the apply).
#[no_mangle]
pub unsafe extern "C" fn wukong_softmax_bwd_f32(
    y: *const f32,
    dy: *const f32,
    dx: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    for row in 0..r {
        let off = row * c;
        softmax_bwd_row(y.add(off), dy.add(off), dx.add(off), c);
    }
}

/// Row count below which the parallel softmax-backward just runs serially.
const SOFTMAX_BWD_PAR_MIN: usize = 8;

/// Multi-threaded softmax backward: rows are mapped across cores, each computed by the identical
/// per-row routine — so the result is **bit-identical to [`wukong_softmax_bwd_f32`]** (rows are
/// independent, no cross-row combine).
///
/// # Safety
/// Operand-size contract of [`wukong_softmax_bwd_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_softmax_bwd_f32_parallel(
    y: *const f32,
    dy: *const f32,
    dx: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r < SOFTMAX_BWD_PAR_MIN {
        wukong_softmax_bwd_f32(y, dy, dx, rows, cols);
        return;
    }
    use rayon::prelude::*;
    let (y_addr, dy_addr, dx_addr) = (y as usize, dy as usize, dx as usize);
    (0..r).into_par_iter().for_each(|row| {
        let off = row * c;
        // SAFETY: disjoint row slices; pointers re-derived from the captured addresses.
        unsafe {
            softmax_bwd_row(
                (y_addr as *const f32).add(off),
                (dy_addr as *const f32).add(off),
                (dx_addr as *mut f32).add(off),
                c,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference using the **same** bit-exact dot the kernel delegates to, then the naive apply — so
    /// equality is literal (the kernel is its own oracle; this pins AVX2-apply == scalar-apply and
    /// serial == parallel across the 8-lane edge and the parallel row threshold).
    #[test]
    fn softmax_bwd_matches_reference_and_parallel() {
        for (rows, cols) in [(1, 1), (3, 7), (4, 8), (5, 16), (9, 33), (16, 100)] {
            let y: Vec<f32> = (0..rows * cols).map(|t| ((t * 5 + 1) % 19) as f32 * 0.05).collect();
            let dy: Vec<f32> = (0..rows * cols).map(|t| ((t * 3 + 2) % 23) as f32 * 0.1 - 1.0).collect();
            // Reference: per-row s via the proven kernel, then dx = y*(dy-s).
            let mut want = vec![0.0f32; rows * cols];
            for row in 0..rows {
                let off = row * cols;
                let s = unsafe {
                    wukong_sreduce_f32(y[off..].as_ptr(), dy[off..].as_ptr(), cols as i64, RED_DOT)
                };
                for i in 0..cols {
                    want[off + i] = y[off + i] * (dy[off + i] - s);
                }
            }
            let mut got = vec![0.0f32; rows * cols];
            let mut got_par = vec![0.0f32; rows * cols];
            unsafe {
                wukong_softmax_bwd_f32(
                    y.as_ptr(),
                    dy.as_ptr(),
                    got.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
                wukong_softmax_bwd_f32_parallel(
                    y.as_ptr(),
                    dy.as_ptr(),
                    got_par.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            assert_eq!(got, want, "softmax_bwd {rows}x{cols} vs reference");
            assert_eq!(got, got_par, "softmax_bwd serial vs parallel {rows}x{cols}");
        }
    }

    /// Independent f64 reference for softmax backward. The per-row dot `s = Σ_j y_j·dy_j` and the
    /// apply `dx_i = y_i·(dy_i − s)` are both recomputed directly in **f64**, with no call into any
    /// `wukong_*` kernel — unlike the self-oracle above, which borrows the kernel's own `sreduce`
    /// dot. This makes it a genuine external oracle (the `wukong_softmax_bwd` family is gate-blind:
    /// the interpreter marshals the identical symbol the native backend calls, so interp==native
    /// proves nothing about the kernel's own arithmetic). `y` is a real per-row softmax distribution
    /// (positive, row-sums to 1) so the dot stays O(1) and well-conditioned. Shapes straddle the
    /// 8-lane edge and the parallel row threshold (`SOFTMAX_BWD_PAR_MIN = 8`); both the serial and
    /// the `_parallel` entry points are checked against the same f64 reference.
    #[test]
    fn matches_f64_reference() {
        for &(rows, cols) in &[
            (1usize, 1usize),
            (3, 7),
            (8, 8),
            (8, 9),
            (9, 33),
            (16, 100),
            (8, 257),
            (40, 320),
        ] {
            // Per-row-normalized y (a genuine softmax output) + sign-mixed upstream dy; deterministic.
            let mut y: Vec<f32> =
                (0..rows * cols).map(|i| ((i as f32) * 0.017).sin() * 0.4 + 0.6).collect();
            for row in 0..rows {
                let off = row * cols;
                let sum: f32 = (0..cols).map(|j| y[off + j]).sum();
                for j in 0..cols {
                    y[off + j] /= sum;
                }
            }
            let dy: Vec<f32> =
                (0..rows * cols).map(|i| ((i as f32) * 0.013 + 0.7).cos() * 1.1 - 0.2).collect();
            let mut got = vec![0.0f32; rows * cols];
            let mut got_par = vec![0.0f32; rows * cols];
            unsafe {
                wukong_softmax_bwd_f32(
                    y.as_ptr(),
                    dy.as_ptr(),
                    got.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
                wukong_softmax_bwd_f32_parallel(
                    y.as_ptr(),
                    dy.as_ptr(),
                    got_par.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for row in 0..rows {
                let off = row * cols;
                // s = Σ_j y_j·dy_j accumulated in f64 — independent of the kernel's reduction.
                let s: f64 = (0..cols).map(|j| y[off + j] as f64 * dy[off + j] as f64).sum();
                for j in 0..cols {
                    let want = y[off + j] as f64 * (dy[off + j] as f64 - s);
                    let denom = want.abs().max(1.0);
                    let val = got[off + j] as f64;
                    assert!(
                        (val - want).abs() / denom <= 1e-4,
                        "softmax_bwd f64 ref mismatch {rows}x{cols} row={row} j={j}: {val} vs {want}"
                    );
                    let valp = got_par[off + j] as f64;
                    assert!(
                        (valp - want).abs() / denom <= 1e-4,
                        "softmax_bwd parallel f64 ref {rows}x{cols} row={row} j={j}: {valp} vs {want}"
                    );
                }
            }
        }
    }
}
