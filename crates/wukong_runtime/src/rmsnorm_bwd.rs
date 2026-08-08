//! RMSNorm backward — the input-gradient that flows through every RMSNorm in a transformer's
//! backward pass (the LLaMA/Mistral/Qwen normalization; the modern replacement for LayerNorm).
//!
//! Forward (per row, `C = cols`, `gamma` an optional per-column scale, `gamma_j = 1` when null):
//!
//! ```text
//! ms  = (Σ_i x_i^2) / C
//! r   = 1 / sqrt(ms + eps)              // the "rms_inv" scale, shared across the row
//! y_i = x_i * r * gamma_i
//! ```
//!
//! Differentiating `y` w.r.t. `x` and contracting with the upstream gradient `dy` gives the standard
//! input gradient:
//!
//! ```text
//! g_i = dy_i * gamma_i                  // the gamma-weighted upstream gradient
//! s   = Σ_i g_i * x_i
//! dx_i = r * ( g_i - x_i * (r*r) * (s / C) )
//! ```
//!
//! Each row needs **two reductions** — the mean-of-squares `Σ x_i^2` (exactly the RMSNorm *forward*
//! reduction) and the gradient dot `s = Σ g_i x_i` — followed by a pure elementwise apply. Both
//! reductions use the **identical fixed 8-lane accumulator + balanced horizontal combine** the
//! forward `wukong_norm_f32` (RMSNorm) uses (lane `j` folds elements `≡ j (mod 8)` in the same order,
//! `mul_add` in the scalar twin == `fmadd` in the AVX2 path, both store to `[f32; 8]` and call the
//! same `hsum8`), so the AVX2 kernel and the scalar fallback agree **bit-for-bit** (pinned by a unit
//! test across partial-chunk / tail sizes). The apply is per-lane arithmetic (`g − x·coef` via one FMA,
//! then a multiply by `r`), bit-identical lane-for-lane.
//!
//! **Determinism.** Rows are independent, so the `_parallel` entry just maps the identical per-row
//! routine across rows — `serial == parallel` bit-for-bit with no cross-row combine. The interpreter
//! marshals the serial form, so the differential oracle stays exact. (Each reduction's lane
//! reassociation is the documented reassociated-reduction exception: every backend runs this same
//! kernel, so they agree.)

use rayon::prelude::*;

/// Fixed-order horizontal sum of 8 lane accumulators — a balanced tree, identical in the scalar twin
/// and the AVX2 path (which stores its `__m256` to `[f32; 8]` and calls this), so both give the same
/// bits. Same shape as `norm::hsum8` / the reduction kernel's `hcombine8`.
#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

// --- scalar twin (the AVX2 tail + the no-AVX2 fallback) -------------------------------------------

/// One row of RMSNorm backward, scalar reference / AVX2 tail / no-AVX2 fallback.
///
/// Computes the two per-row reductions with the same lane structure the AVX2 path uses, then the
/// elementwise apply. `g_i = dy_i * gamma_i` (or `dy_i` when `gamma` is null) is computed identically
/// in the reduction pass and the apply pass, so the two passes are consistent.
///
/// # Safety
/// `x`, `dy`, `dx` valid for `n` f32 (`dx` may alias `dy` — each `g_i`/the row's `s` are read before
/// that row's `dx` is written); `gamma` null or valid for `n` f32.
unsafe fn rmsnorm_bwd_row_scalar(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    dx: *mut f32,
    n: usize,
    eps: f32,
) {
    let nb = n / 8;
    let t = nb * 8;
    let invn = 1.0 / (n as f32);
    let g_null = gamma.is_null();
    // Two fixed 8-lane accumulators: ss = Σ x^2 (the forward RMSNorm reduction), gx = Σ g·x.
    let mut ss = [0.0f32; 8];
    let mut gx = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for j in 0..8 {
            let v = *x.add(b + j);
            let gj = if g_null {
                *dy.add(b + j)
            } else {
                *dy.add(b + j) * *gamma.add(b + j)
            };
            ss[j] = v.mul_add(v, ss[j]);
            gx[j] = gj.mul_add(v, gx[j]);
        }
    }
    for j in 0..(n - t) {
        let v = *x.add(t + j);
        let gj = if g_null {
            *dy.add(t + j)
        } else {
            *dy.add(t + j) * *gamma.add(t + j)
        };
        ss[j] = v.mul_add(v, ss[j]);
        gx[j] = gj.mul_add(v, gx[j]);
    }
    let ms = hsum8(ss) * invn;
    let r = 1.0 / (ms + eps).sqrt();
    let s = hsum8(gx);
    // coef = (r*r) * (s / C); dx_i = r * (g_i - x_i * coef) = r * fma(x_i, -coef, g_i).
    let coef = (r * r) * (s * invn);
    let ncoef = -coef;
    for i in 0..n {
        let v = *x.add(i);
        let gi = if g_null {
            *dy.add(i)
        } else {
            *dy.add(i) * *gamma.add(i)
        };
        *dx.add(i) = r * v.mul_add(ncoef, gi);
    }
}

// --- AVX2 kernel (mirrors the scalar twin lane-for-lane on finite inputs) -------------------------

/// One row of RMSNorm backward, AVX2/FMA. The two reductions keep one `__m256` accumulator each whose
/// lane `j` folds elements `≡ j (mod 8)` exactly as the scalar twin does (`fmadd` == `mul_add`), then
/// store to `[f32; 8]` and call the same `hsum8`; the apply is `r * fmadd(x, -coef, g)` per lane.
///
/// # Safety
/// `x`, `dy`, `dx` valid for `n` f32 (`dx` may alias `dy`); `gamma` null or valid for `n` f32; AVX2+FMA
/// available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rmsnorm_bwd_row_avx2(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    dx: *mut f32,
    n: usize,
    eps: f32,
) {
    use std::arch::x86_64::*;
    let invn = 1.0 / (n as f32);
    let g_null = gamma.is_null();
    // Σ x^2 and Σ g·x, one accumulator each (same lane structure as the scalar twin).
    let mut ssv = _mm256_setzero_ps();
    let mut gxv = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.add(i));
        let dyv = _mm256_loadu_ps(dy.add(i));
        let gv = if g_null {
            dyv
        } else {
            _mm256_mul_ps(dyv, _mm256_loadu_ps(gamma.add(i)))
        };
        ssv = _mm256_fmadd_ps(v, v, ssv);
        gxv = _mm256_fmadd_ps(gv, v, gxv);
        i += 8;
    }
    let mut ss = [0.0f32; 8];
    let mut gx = [0.0f32; 8];
    _mm256_storeu_ps(ss.as_mut_ptr(), ssv);
    _mm256_storeu_ps(gx.as_mut_ptr(), gxv);
    // Tail: fold into the SAME lanes, same ops, as the scalar twin.
    for j in 0..(n - i) {
        let v = *x.add(i + j);
        let gj = if g_null {
            *dy.add(i + j)
        } else {
            *dy.add(i + j) * *gamma.add(i + j)
        };
        ss[j] = v.mul_add(v, ss[j]);
        gx[j] = gj.mul_add(v, gx[j]);
    }
    let ms = hsum8(ss) * invn;
    let r = 1.0 / (ms + eps).sqrt();
    let s = hsum8(gx);
    let coef = (r * r) * (s * invn);
    let ncoef = -coef;
    // Apply: dx = r * fmadd(x, -coef, g) — 8 lanes + scalar tail, bit-identical.
    let rv = _mm256_set1_ps(r);
    let ncv = _mm256_set1_ps(ncoef);
    i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.add(i));
        let dyv = _mm256_loadu_ps(dy.add(i));
        let gv = if g_null {
            dyv
        } else {
            _mm256_mul_ps(dyv, _mm256_loadu_ps(gamma.add(i)))
        };
        // r * (g - x*coef) = r * fma(x, -coef, g).
        _mm256_storeu_ps(dx.add(i), _mm256_mul_ps(rv, _mm256_fmadd_ps(v, ncv, gv)));
        i += 8;
    }
    while i < n {
        let v = *x.add(i);
        let gi = if g_null {
            *dy.add(i)
        } else {
            *dy.add(i) * *gamma.add(i)
        };
        *dx.add(i) = r * v.mul_add(ncoef, gi);
        i += 1;
    }
}

/// One row through RMSNorm backward, AVX2 when available, else the scalar twin.
///
/// # Safety
/// `x`, `dy`, `dx` valid for `n` f32 (`dx` may alias `dy`); `gamma` null or valid for `n` f32.
#[inline]
unsafe fn rmsnorm_bwd_row(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    dx: *mut f32,
    n: usize,
    eps: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return rmsnorm_bwd_row_avx2(x, dy, gamma, dx, n, eps);
        }
    }
    rmsnorm_bwd_row_scalar(x, dy, gamma, dx, n, eps);
}

/// RMSNorm input-gradient `dx[r,i] = r·(g − x·(r²)·(s/C))` over a `[rows, cols]` row-major batch,
/// single-threaded. `gamma` (length `cols`, shared across rows) may be null (⇒ scale 1). `eps_bits`
/// is `eps.to_bits()` widened to i64 (the all-integer dispatch ABI, decoded like `wukong_norm_f32`).
/// `dx` may alias `dy` (each row's reductions are read before that row's `dx` is written).
///
/// # Safety
/// `x`, `dy`, `dx` valid for `rows*cols` f32; `gamma` null or valid for `cols` f32. `dx` must not
/// overlap `x` (each row's `x` is read during the apply after the reductions).
#[no_mangle]
pub unsafe extern "C" fn wukong_rmsnorm_bwd_f32(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    dx: *mut f32,
    rows: i64,
    cols: i64,
    eps_bits: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    let eps = f32::from_bits(eps_bits as u32);
    for row in 0..r {
        let off = row * c;
        // SAFETY: row `row` occupies [off, off+c) ⊆ [0, rows*cols); gamma indexed within [0, c).
        rmsnorm_bwd_row(x.add(off), dy.add(off), gamma, dx.add(off), c, eps);
    }
}

/// Row count below which the parallel RMSNorm-backward just runs serially.
const RMSNORM_BWD_PAR_MIN: usize = 8;

/// Multi-threaded RMSNorm backward: rows are mapped across cores, each computed by the identical
/// per-row routine — so the result is **bit-identical to [`wukong_rmsnorm_bwd_f32`]** (rows are
/// independent, no cross-row combine, so thread count is irrelevant and the interpreter's serial call
/// agrees with this `@parallel` path exactly).
///
/// # Safety
/// Operand-size contract of [`wukong_rmsnorm_bwd_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_rmsnorm_bwd_f32_parallel(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    dx: *mut f32,
    rows: i64,
    cols: i64,
    eps_bits: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r < RMSNORM_BWD_PAR_MIN {
        wukong_rmsnorm_bwd_f32(x, dy, gamma, dx, rows, cols, eps_bits);
        return;
    }
    let eps = f32::from_bits(eps_bits as u32);
    // Raw pointers cross the rayon boundary as integers (null gamma round-trips through 0); each row
    // is a disjoint sub-slice, gamma is shared read-only.
    let (x_addr, dy_addr, g_addr, dx_addr) = (x as usize, dy as usize, gamma as usize, dx as usize);
    // This fork can be the process's FIRST rayon touch, so it must provision the global pool first —
    // [`crate::ensure_global_pool`]'s stated precondition. Forking bare builds rayon's default
    // 2 MiB-stack registry, so the runtime's later 16 MiB `build_global` silently loses the race and
    // outlined `@parallel` region bodies are left on undersized stacks. Idempotent (`Once`) and
    // provisioning-only: the work split below is unchanged, so serial == parallel stays bit-exact.
    crate::ensure_global_pool();
    (0..r).into_par_iter().for_each(|row| {
        let off = row * c;
        // SAFETY: disjoint row slices; pointers re-derived from the captured addresses; gamma valid
        // for `c` (or null) by contract.
        unsafe {
            rmsnorm_bwd_row(
                (x_addr as *const f32).add(off),
                (dy_addr as *const f32).add(off),
                g_addr as *const f32,
                (dx_addr as *mut f32).add(off),
                c,
                eps,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic, mildly varied streams (no RNG — reproducible). Distinct multipliers/phases so x,
    // dy, gamma are not accidentally equal.
    fn fill_x(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.017).sin() * 2.3 - 0.4)
            .collect()
    }
    fn fill_dy(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.013 + 0.7).cos() * 1.1 - 0.2)
            .collect()
    }
    fn fill_gamma(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.011 + 1.3).cos() * 0.6 + 1.0)
            .collect()
    }

    const EPS: f32 = 1e-5;

    /// The exact gradient the kernel computes, in scalar — its own bit-exact oracle. Uses the SAME
    /// fixed 8-lane reduction (so `r`, `s` match the kernel bit-for-bit) and the SAME fused apply.
    fn reference(x: &[f32], dy: &[f32], gamma: Option<&[f32]>, n: usize, eps: f32) -> Vec<f32> {
        let nb = n / 8;
        let t = nb * 8;
        let invn = 1.0 / (n as f32);
        let g = |i: usize| -> f32 {
            match gamma {
                Some(ga) => dy[i] * ga[i],
                None => dy[i],
            }
        };
        let mut ss = [0.0f32; 8];
        let mut gx = [0.0f32; 8];
        for s in 0..nb {
            let b = s * 8;
            for j in 0..8 {
                let v = x[b + j];
                ss[j] = v.mul_add(v, ss[j]);
                gx[j] = g(b + j).mul_add(v, gx[j]);
            }
        }
        for j in 0..(n - t) {
            let v = x[t + j];
            ss[j] = v.mul_add(v, ss[j]);
            gx[j] = g(t + j).mul_add(v, gx[j]);
        }
        let ms = hsum8(ss) * invn;
        let r = 1.0 / (ms + eps).sqrt();
        let s = hsum8(gx);
        let coef = (r * r) * (s * invn);
        let ncoef = -coef;
        (0..n).map(|i| r * x[i].mul_add(ncoef, g(i))).collect()
    }

    /// 1) serial == parallel bit-for-bit across shapes straddling the 8-lane edge and the parallel
    ///    threshold, gamma both NULL and non-null; and == the scalar reference (the kernel's own
    ///    oracle).
    #[test]
    fn rmsnorm_bwd_matches_reference_and_parallel() {
        for (rows, cols) in [
            (1usize, 1usize),
            (3, 7),
            (4, 8),
            (9, 33),
            (16, 100),
            (64, 257),
            (40, 1000),
        ] {
            let x = fill_x(rows * cols);
            let dy = fill_dy(rows * cols);
            let gamma = fill_gamma(cols);
            for use_gamma in [false, true] {
                let gptr = if use_gamma {
                    gamma.as_ptr()
                } else {
                    std::ptr::null()
                };
                // Reference (row by row; gamma is per-column, shared across rows).
                let mut want = vec![0.0f32; rows * cols];
                for row in 0..rows {
                    let off = row * cols;
                    let g = if use_gamma { Some(&gamma[..]) } else { None };
                    let row_ref =
                        reference(&x[off..off + cols], &dy[off..off + cols], g, cols, EPS);
                    want[off..off + cols].copy_from_slice(&row_ref);
                }
                let mut got = vec![0.0f32; rows * cols];
                let mut got_par = vec![0.0f32; rows * cols];
                unsafe {
                    wukong_rmsnorm_bwd_f32(
                        x.as_ptr(),
                        dy.as_ptr(),
                        gptr,
                        got.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                        EPS.to_bits() as i64,
                    );
                    wukong_rmsnorm_bwd_f32_parallel(
                        x.as_ptr(),
                        dy.as_ptr(),
                        gptr,
                        got_par.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                        EPS.to_bits() as i64,
                    );
                }
                for i in 0..rows * cols {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "rmsnorm_bwd {rows}x{cols} gamma={use_gamma} vs reference at {i}: {} vs {}",
                        got[i],
                        want[i]
                    );
                    assert_eq!(
                        got[i].to_bits(),
                        got_par[i].to_bits(),
                        "rmsnorm_bwd serial != parallel {rows}x{cols} gamma={use_gamma} at {i}"
                    );
                }
            }
        }
    }

    /// 2a) explicit scalar-path == AVX2-path bit-for-bit at non-multiple-of-8 cols (the reduction-tail
    /// and apply-tail agreement that underwrites the differential gate).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        // Sizes exercising the 8-lane body, non-multiple-of-8 tails, and rows shorter than 8.
        for &n in &[1usize, 3, 7, 8, 9, 15, 16, 17, 31, 64, 100, 257, 1000] {
            let x = fill_x(n);
            let dy = fill_dy(n);
            let gamma = fill_gamma(n);
            for use_gamma in [false, true] {
                let gptr = if use_gamma {
                    gamma.as_ptr()
                } else {
                    std::ptr::null()
                };
                let mut a = vec![0.0f32; n];
                let mut b = vec![0.0f32; n];
                unsafe {
                    rmsnorm_bwd_row_scalar(x.as_ptr(), dy.as_ptr(), gptr, a.as_mut_ptr(), n, EPS);
                    rmsnorm_bwd_row_avx2(x.as_ptr(), dy.as_ptr(), gptr, b.as_mut_ptr(), n, EPS);
                }
                for i in 0..n {
                    assert_eq!(
                        a[i].to_bits(),
                        b[i].to_bits(),
                        "scalar != avx2 n={n} gamma={use_gamma} i={i}: {} vs {}",
                        a[i],
                        b[i]
                    );
                }
            }
        }
    }

    /// 2b) independent f64 reference with a tight relative tolerance — guards the *formula* (not just
    /// scalar==avx2): an honest double-precision recompute of `dx = r·(g − x·r²·s/C)`.
    #[test]
    fn rmsnorm_bwd_matches_f64_reference() {
        let n = 512usize;
        let x = fill_x(n);
        let dy = fill_dy(n);
        let gamma = fill_gamma(n);
        for use_gamma in [false, true] {
            let gptr = if use_gamma {
                gamma.as_ptr()
            } else {
                std::ptr::null()
            };
            let mut got = vec![0.0f32; n];
            unsafe {
                wukong_rmsnorm_bwd_f32(
                    x.as_ptr(),
                    dy.as_ptr(),
                    gptr,
                    got.as_mut_ptr(),
                    1,
                    n as i64,
                    EPS.to_bits() as i64,
                );
            }
            // f64 oracle.
            let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let gd: Vec<f64> = (0..n)
                .map(|i| {
                    let g = if use_gamma { gamma[i] as f64 } else { 1.0 };
                    dy[i] as f64 * g
                })
                .collect();
            let c = n as f64;
            let ms = xd.iter().map(|&v| v * v).sum::<f64>() / c;
            let r = 1.0 / (ms + EPS as f64).sqrt();
            let s = (0..n).map(|i| gd[i] * xd[i]).sum::<f64>();
            let coef = (r * r) * (s / c);
            for i in 0..n {
                let want = r * (gd[i] - xd[i] * coef);
                let denom = want.abs().max(1.0);
                assert!(
                    ((got[i] as f64 - want).abs() / denom) <= 1e-4,
                    "rmsnorm_bwd f64 ref n={n} gamma={use_gamma} i={i}: {} vs {want}",
                    got[i]
                );
            }
        }
    }

    /// gamma == ones must equal the gamma == null path (a degenerate-affine sanity check; the two are
    /// arithmetically identical because `dy*1.0 == dy` is exact in IEEE-754).
    #[test]
    fn gamma_ones_matches_null() {
        let (rows, cols) = (5usize, 40usize);
        let x = fill_x(rows * cols);
        let dy = fill_dy(rows * cols);
        let ones = vec![1.0f32; cols];
        let mut with_ones = vec![0.0f32; rows * cols];
        let mut with_null = vec![0.0f32; rows * cols];
        unsafe {
            wukong_rmsnorm_bwd_f32(
                x.as_ptr(),
                dy.as_ptr(),
                ones.as_ptr(),
                with_ones.as_mut_ptr(),
                rows as i64,
                cols as i64,
                EPS.to_bits() as i64,
            );
            wukong_rmsnorm_bwd_f32(
                x.as_ptr(),
                dy.as_ptr(),
                std::ptr::null(),
                with_null.as_mut_ptr(),
                rows as i64,
                cols as i64,
                EPS.to_bits() as i64,
            );
        }
        for i in 0..rows * cols {
            assert_eq!(
                with_ones[i].to_bits(),
                with_null[i].to_bits(),
                "gamma=1 != gamma=null at {i}"
            );
        }
    }

    /// Edge: zero/negative rows or cols are a no-op (don't write, don't panic).
    #[test]
    fn degenerate_shapes_are_noops() {
        let x = fill_x(8);
        let dy = fill_dy(8);
        let mut dx = vec![42.0f32; 8];
        unsafe {
            wukong_rmsnorm_bwd_f32(
                x.as_ptr(),
                dy.as_ptr(),
                std::ptr::null(),
                dx.as_mut_ptr(),
                0,
                4,
                EPS.to_bits() as i64,
            );
            wukong_rmsnorm_bwd_f32(
                x.as_ptr(),
                dy.as_ptr(),
                std::ptr::null(),
                dx.as_mut_ptr(),
                2,
                0,
                EPS.to_bits() as i64,
            );
            wukong_rmsnorm_bwd_f32_parallel(
                x.as_ptr(),
                dy.as_ptr(),
                std::ptr::null(),
                dx.as_mut_ptr(),
                -1,
                4,
                EPS.to_bits() as i64,
            );
        }
        assert!(dx.iter().all(|&v| v == 42.0), "no-op must not write dx");
    }
}
