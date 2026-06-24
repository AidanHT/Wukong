//! LayerNorm backward (input gradient) — the gradient that flows back through every LayerNorm in a
//! transformer's training step. Given the original LayerNorm **input** `x[rows, cols]`, the upstream
//! gradient `dy[rows, cols]`, and the (optional) per-column scale `gamma[cols]`, it produces the
//! gradient w.r.t. the input:
//!
//! ```text
//! mean   = (Σ_i x_i) / C
//! var    = (Σ_i (x_i - mean)^2) / C            // population variance (÷C, not ÷(C-1))
//! rstd   = 1 / sqrt(var + eps)
//! xhat_i = (x_i - mean) * rstd
//! g_i    = dy_i * gamma_i                       // gamma_i = 1 if `gamma` is null
//! s1     = Σ_i g_i
//! s2     = Σ_i g_i * xhat_i
//! dx_i   = rstd * ( g_i - s1/C - xhat_i * (s2/C) )
//! ```
//!
//! Statistics are **recomputed from `x`** (we do not assume saved mean/rstd), exactly as the forward
//! [`crate::norm`] LayerNorm does. The four per-row reductions — `Σx`, `Σ(x-mean)²`, `Σg`,
//! `Σ g·xhat` — use the *same* fixed 8-lane accumulator + fixed-order horizontal combine (`hsum8`)
//! the forward kernel uses, so the AVX2 path and the scalar twin/tail are **bit-identical**
//! lane-for-lane (the AVX2 reductions store their `__m256` accumulator to the same `[f32; 8]` the
//! scalar twin builds and call the *same* `hsum8`). `Σ(x-mean)²` and `Σ g·xhat` FMA-contract
//! (`mul_add` == `fmadd`), matching the forward variance discipline. The final per-element `dx`
//! writeback is pure elementwise (8-wide AVX2 + scalar tail), bit-identical.
//!
//! **Determinism.** Rows are independent, so the `_parallel` entry maps the identical per-row routine
//! across cores: `serial == parallel` bit-for-bit, no cross-row combine. The interpreter marshals the
//! serial form, so the differential oracle stays exact. (The lane reassociation in the four sums is
//! the documented reassociated-reduction exception: every backend runs *this* kernel, so they agree.)

/// Fixed-order horizontal sum of 8 lane accumulators — a balanced tree, identical in the scalar twin
/// and the AVX2 path (which stores its `__m256` to `[f32; 8]` and calls this), so both give the same
/// bits. Same shape as the forward norm kernel's `hsum8`.
#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

/// The four per-row reductions of LayerNorm backward, scalar reference (also the AVX2 tail's twin and
/// the no-AVX2 fallback). Returns `(mean, rstd, s1_over_c, s2_over_c)` — everything the elementwise
/// `dx` apply needs. `gamma` null ⇒ every `gamma_i = 1`.
///
/// The lane structure is identical to the forward LayerNorm's mean/variance (8 lanes; lane `j` folds
/// elements ≡ `j` (mod 8), tail into lanes `0..`), extended with two more 8-lane accumulators for
/// `Σ g` and `Σ g·xhat`. `Σ(x-mean)²` and `Σ g·xhat` use `mul_add` to FMA-contract, matching the
/// AVX2 `fmadd`.
///
/// # Safety
/// `x`, `dy` valid for `n` f32; `gamma` null or valid for `n` f32.
unsafe fn stats_row_scalar(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    n: usize,
    eps: f32,
) -> (f32, f32, f32, f32) {
    let nb = n / 8;
    let t = nb * 8;
    let invn = 1.0 / (n as f32);
    let g_null = gamma.is_null();
    // mean = (Σ x) / C
    let mut sm = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, smj) in sm.iter_mut().enumerate() {
            *smj += *x.add(b + j);
        }
    }
    for (j, smj) in sm.iter_mut().enumerate().take(n - t) {
        *smj += *x.add(t + j);
    }
    let mean = hsum8(sm) * invn;
    // var = (Σ (x - mean)^2) / C  (the addend FMA-contracts, matching the AVX2 fmadd)
    let mut vv = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, vvj) in vv.iter_mut().enumerate() {
            let d = *x.add(b + j) - mean;
            *vvj = d.mul_add(d, *vvj);
        }
    }
    for (j, vvj) in vv.iter_mut().enumerate().take(n - t) {
        let d = *x.add(t + j) - mean;
        *vvj = d.mul_add(d, *vvj);
    }
    let rstd = 1.0 / (hsum8(vv) * invn + eps).sqrt();
    // s1 = Σ g_i ; s2 = Σ g_i * xhat_i, with g_i = dy_i * gamma_i, xhat_i = (x_i - mean) * rstd.
    let mut s1 = [0.0f32; 8];
    let mut s2 = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for j in 0..8 {
            let gi = if g_null {
                *dy.add(b + j)
            } else {
                *dy.add(b + j) * *gamma.add(b + j)
            };
            let xhat = (*x.add(b + j) - mean) * rstd;
            s1[j] += gi;
            s2[j] = gi.mul_add(xhat, s2[j]);
        }
    }
    for j in 0..(n - t) {
        let gi = if g_null {
            *dy.add(t + j)
        } else {
            *dy.add(t + j) * *gamma.add(t + j)
        };
        let xhat = (*x.add(t + j) - mean) * rstd;
        s1[j] += gi;
        s2[j] = gi.mul_add(xhat, s2[j]);
    }
    (mean, rstd, hsum8(s1) * invn, hsum8(s2) * invn)
}

/// One row's elementwise `dx` apply, scalar reference / no-AVX2 fallback:
/// `dx_i = rstd * (g_i - s1c - xhat_i * s2c)`, with `g_i = dy_i * gamma_i`,
/// `xhat_i = (x_i - mean) * rstd`. Pure per-lane arithmetic, matching the AVX2 twin lane-for-lane.
///
/// # Safety
/// `x`, `dy` valid for `n` f32; `gamma` null or valid for `n` f32; `dx` valid for `n` f32.
unsafe fn apply_row_scalar(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    dx: *mut f32,
    n: usize,
    mean: f32,
    rstd: f32,
    s1c: f32,
    s2c: f32,
) {
    let g_null = gamma.is_null();
    for i in 0..n {
        let gi = if g_null {
            *dy.add(i)
        } else {
            *dy.add(i) * *gamma.add(i)
        };
        let xhat = (*x.add(i) - mean) * rstd;
        // dx = rstd * (g - s1c - xhat*s2c). Compute the inner combination then scale, matching AVX2.
        let inner = gi - s1c - xhat * s2c;
        *dx.add(i) = rstd * inner;
    }
}

// --- AVX2 (mirrors the scalar twins lane-for-lane on finite inputs) --------------------------------

/// AVX2 twin of [`stats_row_scalar`]: the four per-row reductions with 256-bit accumulators whose
/// horizontal combine is the *same* `hsum8` the scalar path uses (store the `__m256` to `[f32; 8]`,
/// fold the tail in the identical lane order, call `hsum8`).
///
/// # Safety
/// `x`, `dy` valid for `n` f32; `gamma` null or valid for `n` f32; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn stats_row_avx2(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    n: usize,
    eps: f32,
) -> (f32, f32, f32, f32) {
    use std::arch::x86_64::*;
    let invn = 1.0 / (n as f32);
    let g_null = gamma.is_null();
    // mean
    let mut sv = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        sv = _mm256_add_ps(sv, _mm256_loadu_ps(x.add(i)));
        i += 8;
    }
    let mut sm = [0.0f32; 8];
    _mm256_storeu_ps(sm.as_mut_ptr(), sv);
    for (j, smj) in sm.iter_mut().enumerate().take(n - i) {
        *smj += *x.add(i + j);
    }
    let mean = hsum8(sm) * invn;
    let mb = _mm256_set1_ps(mean);
    // variance
    let mut vv = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        let d = _mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb);
        vv = _mm256_fmadd_ps(d, d, vv);
        i += 8;
    }
    let mut va = [0.0f32; 8];
    _mm256_storeu_ps(va.as_mut_ptr(), vv);
    for (j, vaj) in va.iter_mut().enumerate().take(n - i) {
        let d = *x.add(i + j) - mean;
        *vaj = d.mul_add(d, *vaj);
    }
    let rstd = 1.0 / (hsum8(va) * invn + eps).sqrt();
    let rb = _mm256_set1_ps(rstd);
    // s1 = Σ g ; s2 = Σ g·xhat, g = dy*gamma, xhat = (x-mean)*rstd.
    let mut s1v = _mm256_setzero_ps();
    let mut s2v = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        let dyv = _mm256_loadu_ps(dy.add(i));
        let gv = if g_null {
            dyv
        } else {
            _mm256_mul_ps(dyv, _mm256_loadu_ps(gamma.add(i)))
        };
        let xhat = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb), rb);
        s1v = _mm256_add_ps(s1v, gv);
        s2v = _mm256_fmadd_ps(gv, xhat, s2v);
        i += 8;
    }
    let mut s1 = [0.0f32; 8];
    let mut s2 = [0.0f32; 8];
    _mm256_storeu_ps(s1.as_mut_ptr(), s1v);
    _mm256_storeu_ps(s2.as_mut_ptr(), s2v);
    for j in 0..(n - i) {
        let gi = if g_null {
            *dy.add(i + j)
        } else {
            *dy.add(i + j) * *gamma.add(i + j)
        };
        let xhat = (*x.add(i + j) - mean) * rstd;
        s1[j] += gi;
        s2[j] = gi.mul_add(xhat, s2[j]);
    }
    (mean, rstd, hsum8(s1) * invn, hsum8(s2) * invn)
}

/// AVX2 twin of [`apply_row_scalar`]: `dx_i = rstd * (g_i - s1c - xhat_i * s2c)`, 8 lanes/step + a
/// scalar tail. The op sequence mirrors the scalar twin exactly — `g = dy*gamma`,
/// `xhat = (x-mean)*rstd`, `inner = (g - s1c) - xhat*s2c` (a `mul` then a `sub`, not a fused negate),
/// then `dx = rstd * inner` — so scalar and AVX2 agree lane-for-lane.
///
/// # Safety
/// `x`, `dy` valid for `n` f32; `gamma` null or valid for `n` f32; `dx` valid for `n` f32;
/// AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_row_avx2(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    dx: *mut f32,
    n: usize,
    mean: f32,
    rstd: f32,
    s1c: f32,
    s2c: f32,
) {
    use std::arch::x86_64::*;
    let g_null = gamma.is_null();
    let mb = _mm256_set1_ps(mean);
    let rb = _mm256_set1_ps(rstd);
    let s1b = _mm256_set1_ps(s1c);
    let s2b = _mm256_set1_ps(s2c);
    let mut i = 0;
    while i + 8 <= n {
        let dyv = _mm256_loadu_ps(dy.add(i));
        let gv = if g_null {
            dyv
        } else {
            _mm256_mul_ps(dyv, _mm256_loadu_ps(gamma.add(i)))
        };
        let xhat = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb), rb);
        // inner = (g - s1c) - xhat*s2c   (mul then sub — mirrors the scalar `xhat * s2c` then subtract)
        let inner = _mm256_sub_ps(_mm256_sub_ps(gv, s1b), _mm256_mul_ps(xhat, s2b));
        _mm256_storeu_ps(dx.add(i), _mm256_mul_ps(rb, inner));
        i += 8;
    }
    while i < n {
        let gi = if g_null {
            *dy.add(i)
        } else {
            *dy.add(i) * *gamma.add(i)
        };
        let xhat = (*x.add(i) - mean) * rstd;
        let inner = gi - s1c - xhat * s2c;
        *dx.add(i) = rstd * inner;
        i += 1;
    }
}

/// One row of LayerNorm backward: recompute `(mean, rstd, s1/C, s2/C)` from `x`/`dy`/`gamma`, then
/// write `dx_i = rstd*(g_i - s1/C - xhat_i*(s2/C))`. AVX2 when available, else the scalar twin.
///
/// # Safety
/// `x`, `dy` valid for `n` f32; `gamma` null or valid for `n` f32; `dx` valid for `n` f32.
#[inline]
unsafe fn layernorm_bwd_row(
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
            let (mean, rstd, s1c, s2c) = stats_row_avx2(x, dy, gamma, n, eps);
            apply_row_avx2(x, dy, gamma, dx, n, mean, rstd, s1c, s2c);
            return;
        }
    }
    let (mean, rstd, s1c, s2c) = stats_row_scalar(x, dy, gamma, n, eps);
    apply_row_scalar(x, dy, gamma, dx, n, mean, rstd, s1c, s2c);
}

/// LayerNorm backward (input gradient) over a `[rows, cols]` row-major batch, single-threaded.
/// `x` = the original LayerNorm input; `dy` = upstream gradient of the output; `gamma` = per-column
/// scale of length `cols` (may be **null** ⇒ every `gamma_i = 1`); `dx` = output gradient.
/// `eps_bits` is `eps.to_bits()` widened into an i64 (the all-integer ABI the other dispatch kernels
/// use — mirror of [`crate::norm`]). Statistics are recomputed from `x` per row (saved stats not
/// assumed).
///
/// # Safety
/// `x`, `dy`, `dx` valid for `rows*cols` f32; `gamma` null or valid for `cols` f32. `dx` must not
/// overlap `x`/`dy` (each row reads `x`/`dy` during its apply).
#[no_mangle]
pub unsafe extern "C" fn mercury_layernorm_bwd_f32(
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
        layernorm_bwd_row(x.add(off), dy.add(off), gamma, dx.add(off), c, eps);
    }
}

/// Row count below which the parallel LayerNorm-backward just runs serially.
const LAYERNORM_BWD_PAR_MIN: usize = 8;

/// Multi-threaded LayerNorm backward — **bit-identical** to [`mercury_layernorm_bwd_f32`]. Rows are
/// independent, so each is computed by the identical per-row routine regardless of which thread runs
/// it; there is no cross-row combine, so the result does not depend on thread count and the
/// interpreter (serial) agrees with this `@parallel` path exactly.
///
/// # Safety
/// Operand-size contract of [`mercury_layernorm_bwd_f32`].
#[no_mangle]
pub unsafe extern "C" fn mercury_layernorm_bwd_f32_parallel(
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
    if r < LAYERNORM_BWD_PAR_MIN {
        mercury_layernorm_bwd_f32(x, dy, gamma, dx, rows, cols, eps_bits);
        return;
    }
    let eps = f32::from_bits(eps_bits as u32);
    use rayon::prelude::*;
    // Raw pointers cross the rayon boundary as integers (null `gamma` round-trips through 0); rows are
    // disjoint, gamma is shared read-only — the same pattern as the parallel norm/reduce kernels.
    let (x_addr, dy_addr, g_addr, dx_addr) =
        (x as usize, dy as usize, gamma as usize, dx as usize);
    (0..r).into_par_iter().for_each(|row| {
        let off = row * c;
        // SAFETY: disjoint row slice; pointers re-derived from the captured addresses; gamma shared.
        unsafe {
            layernorm_bwd_row(
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

    const EPS: f32 = 1e-5;

    // Deterministic, mildly varied inputs (no RNG — reproducible). Distinct streams for x / dy / gamma.
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

    /// A scalar reference computed via the kernel's own scalar path (the self-oracle): recompute the
    /// four reductions with `stats_row_scalar`, then the naive `dx` apply. Equality with the kernel
    /// output is literal bits (this pins AVX2 == scalar across the 8-lane edge, gamma null/non-null).
    /// `gamma` is the length-`cols` per-column array shared across rows.
    fn reference(x: &[f32], dy: &[f32], gamma: Option<&[f32]>, rows: usize, cols: usize) -> Vec<f32> {
        let mut want = vec![0.0f32; rows * cols];
        let gptr = match gamma {
            Some(g) => g.as_ptr(),
            None => std::ptr::null(),
        };
        for row in 0..rows {
            let off = row * cols;
            unsafe {
                let (mean, rstd, s1c, s2c) =
                    stats_row_scalar(x[off..].as_ptr(), dy[off..].as_ptr(), gptr, cols, EPS);
                apply_row_scalar(
                    x[off..].as_ptr(),
                    dy[off..].as_ptr(),
                    gptr,
                    want[off..].as_mut_ptr(),
                    cols,
                    mean,
                    rstd,
                    s1c,
                    s2c,
                );
            }
        }
        want
    }

    #[test]
    fn layernorm_bwd_matches_reference_and_parallel() {
        // Shapes straddle the 8-lane edge and the parallel row threshold (LAYERNORM_BWD_PAR_MIN=8).
        for &(rows, cols) in &[
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
                let gamma_opt = if use_gamma { Some(&gamma[..]) } else { None };
                let gptr = if use_gamma {
                    gamma.as_ptr()
                } else {
                    std::ptr::null()
                };
                let want = reference(&x, &dy, gamma_opt, rows, cols);
                let mut got = vec![0.0f32; rows * cols];
                let mut got_par = vec![0.0f32; rows * cols];
                unsafe {
                    mercury_layernorm_bwd_f32(
                        x.as_ptr(),
                        dy.as_ptr(),
                        gptr,
                        got.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                        EPS.to_bits() as i64,
                    );
                    mercury_layernorm_bwd_f32_parallel(
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
                        "kernel != reference {rows}x{cols} gamma={use_gamma} i={i}: {} vs {}",
                        got[i],
                        want[i]
                    );
                    assert_eq!(
                        got[i].to_bits(),
                        got_par[i].to_bits(),
                        "serial != parallel {rows}x{cols} gamma={use_gamma} i={i}: {} vs {}",
                        got[i],
                        got_par[i]
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        // Sizes exercising the 8-lane body, a non-multiple-of-8 tail, and rows shorter than 8.
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
                // stats: scalar vs avx2 must agree bit-for-bit.
                let (ms, rs, a1s, a2s) =
                    unsafe { stats_row_scalar(x.as_ptr(), dy.as_ptr(), gptr, n, EPS) };
                let (mv, rv, a1v, a2v) =
                    unsafe { stats_row_avx2(x.as_ptr(), dy.as_ptr(), gptr, n, EPS) };
                assert_eq!(ms.to_bits(), mv.to_bits(), "mean scalar!=avx2 n={n} g={use_gamma}");
                assert_eq!(rs.to_bits(), rv.to_bits(), "rstd scalar!=avx2 n={n} g={use_gamma}");
                assert_eq!(a1s.to_bits(), a1v.to_bits(), "s1/C scalar!=avx2 n={n} g={use_gamma}");
                assert_eq!(a2s.to_bits(), a2v.to_bits(), "s2/C scalar!=avx2 n={n} g={use_gamma}");
                // apply: scalar vs avx2 must agree bit-for-bit.
                let mut a = vec![0.0f32; n];
                let mut b = vec![0.0f32; n];
                unsafe {
                    apply_row_scalar(x.as_ptr(), dy.as_ptr(), gptr, a.as_mut_ptr(), n, ms, rs, a1s, a2s);
                    apply_row_avx2(x.as_ptr(), dy.as_ptr(), gptr, b.as_mut_ptr(), n, mv, rv, a1v, a2v);
                }
                for i in 0..n {
                    assert_eq!(
                        a[i].to_bits(),
                        b[i].to_bits(),
                        "apply scalar != avx2 n={n} g={use_gamma} i={i}: {} vs {}",
                        a[i],
                        b[i]
                    );
                }
            }
        }
    }

    #[test]
    fn matches_f64_reference() {
        // Independent f64 computation of the LayerNorm input gradient, tight relative tolerance.
        let (rows, cols) = (8usize, 320usize);
        let x = fill_x(rows * cols);
        let dy = fill_dy(rows * cols);
        let gamma = fill_gamma(cols);
        for use_gamma in [false, true] {
            let gptr = if use_gamma {
                gamma.as_ptr()
            } else {
                std::ptr::null()
            };
            let mut got = vec![0.0f32; rows * cols];
            unsafe {
                mercury_layernorm_bwd_f32(
                    x.as_ptr(),
                    dy.as_ptr(),
                    gptr,
                    got.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                    EPS.to_bits() as i64,
                );
            }
            let c = cols as f64;
            for row in 0..rows {
                let off = row * cols;
                let xd: Vec<f64> = x[off..off + cols].iter().map(|&v| v as f64).collect();
                let gd: Vec<f64> = (0..cols)
                    .map(|j| if use_gamma { gamma[j] as f64 } else { 1.0 })
                    .collect();
                let dyd: Vec<f64> = dy[off..off + cols].iter().map(|&v| v as f64).collect();
                let mean = xd.iter().sum::<f64>() / c;
                let var = xd.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / c;
                let rstd = 1.0 / (var + EPS as f64).sqrt();
                let g: Vec<f64> = (0..cols).map(|j| dyd[j] * gd[j]).collect();
                let xhat: Vec<f64> = (0..cols).map(|j| (xd[j] - mean) * rstd).collect();
                let s1 = g.iter().sum::<f64>() / c;
                let s2 = (0..cols).map(|j| g[j] * xhat[j]).sum::<f64>() / c;
                for j in 0..cols {
                    let want = rstd * (g[j] - s1 - xhat[j] * s2);
                    let val = got[off + j] as f64;
                    let denom = want.abs().max(1.0);
                    assert!(
                        (val - want).abs() / denom <= 1e-4,
                        "f64 ref mismatch row={row} j={j} g={use_gamma}: {val} vs {want}"
                    );
                }
            }
        }
    }
}
