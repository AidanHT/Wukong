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
//! [`crate::norm`] LayerNorm does. The row is walked **three** times, not four: `Σx` for the mean,
//! then `Σ(x-mean)²`, `Σg` and `Σ g·(x-mean)` **together** in one pass (they all need only `mean`,
//! because `Σ g·xhat = rstd · Σ g·(x-mean)` factors the row-constant `rstd` out of the sum), then the
//! `dx` writeback. Each reduction runs `UNROLL` independent accumulator chains rather than one, so it
//! is bound by FMA *throughput* instead of FMA *latency* — see the block above [`sum8_scalar`] for
//! both mechanisms and the register-pressure reason `UNROLL` is 4.
//!
//! Every reduction is a matched scalar/AVX2 pair with the identical accumulator layout and the same
//! fixed-order horizontal combine (`hsum8`), so the AVX2 path and the scalar twin/tail are
//! **bit-identical** lane-for-lane (the AVX2 reductions store their `__m256` accumulators to the same
//! `[f32; 8]` the scalar twin builds and call the *same* combine). `Σ(x-mean)²` and `Σ g·(x-mean)`
//! FMA-contract (`mul_add` == `fmadd`), matching the forward variance discipline. The final
//! per-element `dx` writeback is pure elementwise (8-wide AVX2 + scalar tail), bit-identical, and its
//! arithmetic is unchanged from the original kernel.
//!
//! **Determinism.** Rows are independent, so the `_parallel` entry maps the identical per-row routine
//! across cores: `serial == parallel` bit-for-bit, no cross-row combine. The interpreter marshals the
//! serial form, so the differential oracle stays exact. (The lane reassociation in the sums is the
//! documented reassociated-reduction exception: every backend runs *this* kernel, so they agree.)

/// Fixed-order horizontal sum of 8 lane accumulators — a balanced tree, identical in the scalar twin
/// and the AVX2 path (which stores its `__m256` to `[f32; 8]` and calls this), so both give the same
/// bits. Same shape as the forward norm kernel's `hsum8`.
#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

// --- row reductions: UNROLL independent accumulator chains, THREE passes not four ---------------
//
// Two defects were fixed here together, both visible only under a perf instrument (a recognized
// kernel is gate-blind — every backend calls this same function, so no differential or invariance
// gate can see either one).
//
// **(1) One accumulator.** Each reduction pass ran a single `__m256` accumulator:
//
//     while i + 8 <= n { let d = sub(load(x+i), mb); vv = fmadd(d, d, vv); i += 8; }
//
// `vv` is loop-carried, so the pass retires one 8-lane step per FMA *latency* (4 cycles on
// Golden/Redwood Cove) instead of per FMA *throughput* (two per cycle). The `UNROLL` accumulator
// chains below put that many independent FMAs in flight and make the pass throughput-bound. This is
// the same lever `crate::norm`'s forward reductions take, and the same one llama.cpp's
// `GGML_F32_ARR = 4` sum-of-partials takes in `ggml_vec_dot_f32`.
//
// **(2) Four passes over the row where three suffice.** The shipped kernel walked the row for `Σx`,
// again for `Σ(x-mean)²`, again for `Σg` and `Σ g·xhat`, and a fourth time to write `dx`. But
// `xhat = (x-mean)·rstd` is just `(x-mean)` scaled by a **row constant**, so
//
//     Σ g·xhat = rstd · Σ g·(x-mean)
//
// and `Σ g·(x-mean)` needs only `mean` — not `rstd`, and therefore not the variance. The variance,
// `Σ g` and `Σ g·(x-mean)` all sit at the same dependency level and ride ONE pass
// ([`var_s1_t2_8_scalar`] / [`var_s1_t2_8_avx2`]). That removes a whole `Σ(x-mean)²` walk of `x`:
// oneDNN's `jit_uni_layer_normalization` and PyTorch ATen's `LayerNormBackwardKernelImplInternal`
// both compute their backward statistics in a single pass for the same reason (they go further and
// take `mean`/`rstd` from the forward, which a recognizer over a from-scratch loop nest cannot).
//
// `UNROLL` is **4, not 8**: the fused pass carries three reductions at once, so eight chains would
// need 24 live `__m256` accumulators against AVX2's 16 architectural `ymm` registers and would spill
// every step. Four chains x three reductions is 12, leaving `mb` and the step's temporaries in
// registers — and 12 chains in flight already covers the 4-cycle FMA latency at two issues a cycle.
//
// **Each reduction is a matched scalar/AVX2 pair with the identical block structure**, because this
// module's contract is that the AVX2 kernel and its scalar twin agree bit-for-bit, and an accumulator
// layout is exactly what silently desyncs. Lane `j` of scalar accumulator `t` folds the same elements
// as lane `j` of vector accumulator `t`, and both end with the same explicit halving tree.
//
// Consequences worth stating: for `n < BLOCK` the blocked body never runs and everything falls
// through the single-accumulator remainder, so short rows keep byte-identical results to the
// pre-unroll kernel. For `n >= BLOCK` the summation order moves toward pairwise summation, so the
// error bound improves; and `s2` now rounds once per row instead of once per element (`rstd·Σ g·d`
// rather than `Σ g·(d·rstd)`), which is also strictly fewer roundings.

/// Accumulator chains in flight over a reduction's blocked body.
const UNROLL: usize = 4;
/// Elements consumed per blocked step (`UNROLL` 8-lane vectors).
const BLOCK: usize = UNROLL * 8;

/// Fold the `UNROLL` lane-accumulator arrays down to one with an explicit halving tree. The AVX2
/// twins run the identical tree on `__m256` values and `_mm256_add_ps` is lane-wise, so lane `j` here
/// is lane `j` there.
#[inline(always)]
fn combine(a: &mut [[f32; 8]; UNROLL]) {
    let mut w = UNROLL;
    while w > 1 {
        let h = w / 2;
        let (lo, hi) = a[..w].split_at_mut(h);
        for (l, r) in lo.iter_mut().zip(hi.iter()) {
            for (lj, rj) in l.iter_mut().zip(r.iter()) {
                *lj += *rj;
            }
        }
        w = h;
    }
}

/// `g_i = dy_i * gamma_i`, with a null `gamma` meaning `gamma_i = 1`.
///
/// # Safety
/// `dy` valid at `i`; `gamma` null or valid at `i`.
#[inline(always)]
unsafe fn gi(dy: *const f32, gamma: *const f32, i: usize) -> f32 {
    if gamma.is_null() {
        *dy.add(i)
    } else {
        *dy.add(i) * *gamma.add(i)
    }
}

/// `Σ x[i]` as 8 lane accumulators (the mean pass), scalar twin of [`sum8_avx2`].
///
/// # Safety
/// `x` valid for `n` f32.
unsafe fn sum8_scalar(x: *const f32, n: usize) -> [f32; 8] {
    let mut a = [[0.0f32; 8]; UNROLL];
    let mut i = 0;
    while i + BLOCK <= n {
        for (t, at) in a.iter_mut().enumerate() {
            for (j, atj) in at.iter_mut().enumerate() {
                *atj += *x.add(i + 8 * t + j);
            }
        }
        i += BLOCK;
    }
    while i + 8 <= n {
        for (j, a0j) in a[0].iter_mut().enumerate() {
            *a0j += *x.add(i + j);
        }
        i += 8;
    }
    combine(&mut a);
    let mut sm = a[0];
    for (j, smj) in sm.iter_mut().enumerate().take(n - i) {
        *smj += *x.add(i + j);
    }
    sm
}

/// The fused statistics pass: `(Σ (x-mean)², Σ g, Σ g·(x-mean))` as 8 lane accumulators each, scalar
/// twin of [`var_s1_t2_8_avx2`]. The two product addends `mul_add`, matching the AVX2 `fmadd`.
///
/// # Safety
/// `x`, `dy` valid for `n` f32; `gamma` null or valid for `n` f32.
unsafe fn var_s1_t2_8_scalar(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    n: usize,
    mean: f32,
) -> ([f32; 8], [f32; 8], [f32; 8]) {
    let mut va = [[0.0f32; 8]; UNROLL];
    let mut sa = [[0.0f32; 8]; UNROLL];
    let mut ta = [[0.0f32; 8]; UNROLL];
    let mut i = 0;
    while i + BLOCK <= n {
        for t in 0..UNROLL {
            for j in 0..8 {
                let k = i + 8 * t + j;
                let d = *x.add(k) - mean;
                let g = gi(dy, gamma, k);
                va[t][j] = d.mul_add(d, va[t][j]);
                sa[t][j] += g;
                ta[t][j] = g.mul_add(d, ta[t][j]);
            }
        }
        i += BLOCK;
    }
    while i + 8 <= n {
        for j in 0..8 {
            let d = *x.add(i + j) - mean;
            let g = gi(dy, gamma, i + j);
            va[0][j] = d.mul_add(d, va[0][j]);
            sa[0][j] += g;
            ta[0][j] = g.mul_add(d, ta[0][j]);
        }
        i += 8;
    }
    combine(&mut va);
    combine(&mut sa);
    combine(&mut ta);
    let (mut v, mut s1, mut t2) = (va[0], sa[0], ta[0]);
    for j in 0..(n - i) {
        let d = *x.add(i + j) - mean;
        let g = gi(dy, gamma, i + j);
        v[j] = d.mul_add(d, v[j]);
        s1[j] += g;
        t2[j] = g.mul_add(d, t2[j]);
    }
    (v, s1, t2)
}

/// `Σ x[i]` on `UNROLL` 256-bit accumulator chains, AVX2 twin of [`sum8_scalar`].
///
/// # Safety
/// `x` valid for `n` f32; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum8_avx2(x: *const f32, n: usize) -> [f32; 8] {
    use std::arch::x86_64::*;
    let mut a = [_mm256_setzero_ps(); UNROLL];
    let mut i = 0;
    while i + BLOCK <= n {
        for (t, at) in a.iter_mut().enumerate() {
            *at = _mm256_add_ps(*at, _mm256_loadu_ps(x.add(i + 8 * t)));
        }
        i += BLOCK;
    }
    while i + 8 <= n {
        a[0] = _mm256_add_ps(a[0], _mm256_loadu_ps(x.add(i)));
        i += 8;
    }
    let mut w = UNROLL;
    while w > 1 {
        let h = w / 2;
        for t in 0..h {
            a[t] = _mm256_add_ps(a[t], a[t + h]);
        }
        w = h;
    }
    let mut sm = [0.0f32; 8];
    _mm256_storeu_ps(sm.as_mut_ptr(), a[0]);
    for (j, smj) in sm.iter_mut().enumerate().take(n - i) {
        *smj += *x.add(i + j);
    }
    sm
}

/// The fused statistics pass on `UNROLL` chains per reduction, AVX2 twin of [`var_s1_t2_8_scalar`].
///
/// # Safety
/// `x`, `dy` valid for `n` f32; `gamma` null or valid for `n` f32; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn var_s1_t2_8_avx2(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    n: usize,
    mean: f32,
) -> ([f32; 8], [f32; 8], [f32; 8]) {
    use std::arch::x86_64::*;
    let g_null = gamma.is_null();
    let mb = _mm256_set1_ps(mean);
    let mut va = [_mm256_setzero_ps(); UNROLL];
    let mut sa = [_mm256_setzero_ps(); UNROLL];
    let mut ta = [_mm256_setzero_ps(); UNROLL];
    let mut i = 0;
    while i + BLOCK <= n {
        for t in 0..UNROLL {
            let o = i + 8 * t;
            let d = _mm256_sub_ps(_mm256_loadu_ps(x.add(o)), mb);
            let dyv = _mm256_loadu_ps(dy.add(o));
            let g = if g_null {
                dyv
            } else {
                _mm256_mul_ps(dyv, _mm256_loadu_ps(gamma.add(o)))
            };
            va[t] = _mm256_fmadd_ps(d, d, va[t]);
            sa[t] = _mm256_add_ps(sa[t], g);
            ta[t] = _mm256_fmadd_ps(g, d, ta[t]);
        }
        i += BLOCK;
    }
    while i + 8 <= n {
        let d = _mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb);
        let dyv = _mm256_loadu_ps(dy.add(i));
        let g = if g_null {
            dyv
        } else {
            _mm256_mul_ps(dyv, _mm256_loadu_ps(gamma.add(i)))
        };
        va[0] = _mm256_fmadd_ps(d, d, va[0]);
        sa[0] = _mm256_add_ps(sa[0], g);
        ta[0] = _mm256_fmadd_ps(g, d, ta[0]);
        i += 8;
    }
    let mut w = UNROLL;
    while w > 1 {
        let h = w / 2;
        for t in 0..h {
            va[t] = _mm256_add_ps(va[t], va[t + h]);
            sa[t] = _mm256_add_ps(sa[t], sa[t + h]);
            ta[t] = _mm256_add_ps(ta[t], ta[t + h]);
        }
        w = h;
    }
    let (mut v, mut s1, mut t2) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
    _mm256_storeu_ps(v.as_mut_ptr(), va[0]);
    _mm256_storeu_ps(s1.as_mut_ptr(), sa[0]);
    _mm256_storeu_ps(t2.as_mut_ptr(), ta[0]);
    for j in 0..(n - i) {
        let d = *x.add(i + j) - mean;
        let g = gi(dy, gamma, i + j);
        v[j] = d.mul_add(d, v[j]);
        s1[j] += g;
        t2[j] = g.mul_add(d, t2[j]);
    }
    (v, s1, t2)
}

/// The per-row reductions of LayerNorm backward, scalar reference (also the AVX2 tail's twin and the
/// no-AVX2 fallback). Returns `(mean, rstd, s1_over_c, s2_over_c)` — everything the elementwise `dx`
/// apply needs. `gamma` null ⇒ every `gamma_i = 1`.
///
/// Two passes: `mean` from [`sum8_scalar`], then variance, `Σ g` and `Σ g·(x-mean)` together from
/// [`var_s1_t2_8_scalar`]. `s2/C` is `rstd · Σ g·(x-mean) / C`, algebraically the shipped
/// `Σ g·xhat / C` with the row-constant `rstd` factored out of the sum (see the block above).
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
    let invn = 1.0 / (n as f32);
    let mean = hsum8(sum8_scalar(x, n)) * invn;
    let (v, s1, t2) = var_s1_t2_8_scalar(x, dy, gamma, n, mean);
    let rstd = 1.0 / (hsum8(v) * invn + eps).sqrt();
    (mean, rstd, hsum8(s1) * invn, (rstd * hsum8(t2)) * invn)
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

/// AVX2 twin of [`stats_row_scalar`]: the same two passes, on the same accumulator layout, ending in
/// the same `hsum8` — so the two paths agree bit-for-bit (pinned by `scalar_matches_avx2_bit_for_bit`
/// across every block / remainder / tail seam).
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
    let invn = 1.0 / (n as f32);
    let mean = hsum8(sum8_avx2(x, n)) * invn;
    let (v, s1, t2) = var_s1_t2_8_avx2(x, dy, gamma, n, mean);
    let rstd = 1.0 / (hsum8(v) * invn + eps).sqrt();
    (mean, rstd, hsum8(s1) * invn, (rstd * hsum8(t2)) * invn)
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
/// **Aliasing: all operands disjoint** — `dx` overlaps none of `x`, `dy`, `gamma`.
#[no_mangle]
pub unsafe extern "C" fn wukong_layernorm_bwd_f32(
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

/// Multi-threaded LayerNorm backward — **bit-identical** to [`wukong_layernorm_bwd_f32`]. Rows are
/// independent, so each is computed by the identical per-row routine regardless of which thread runs
/// it; there is no cross-row combine, so the result does not depend on thread count and the
/// interpreter (serial) agrees with this `@parallel` path exactly.
///
/// # Safety
/// Operand-size contract of [`wukong_layernorm_bwd_f32`].
/// **Aliasing:** that of [`wukong_layernorm_bwd_f32`] — all operands disjoint: `dx` overlaps none
/// of `x`, `dy`, `gamma`.
#[no_mangle]
pub unsafe extern "C" fn wukong_layernorm_bwd_f32_parallel(
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
    if !crate::par::fork_rows(r, c, crate::par::W_ROWPASS) {
        wukong_layernorm_bwd_f32(x, dy, gamma, dx, rows, cols, eps_bits);
        return;
    }
    let eps = f32::from_bits(eps_bits as u32);
    use rayon::prelude::*;
    // Raw pointers cross the rayon boundary as integers (null `gamma` round-trips through 0); rows are
    // disjoint, gamma is shared read-only — the same pattern as the parallel norm/reduce kernels.
    let (x_addr, dy_addr, g_addr, dx_addr) = (x as usize, dy as usize, gamma as usize, dx as usize);
    // This fork can be the process's FIRST rayon touch, so it must provision the global pool first —
    // [`crate::ensure_global_pool`]'s stated precondition. Forking bare builds rayon's default
    // 2 MiB-stack registry, so the runtime's later 16 MiB `build_global` silently loses the race and
    // outlined `@parallel` region bodies are left on undersized stacks. Idempotent (`Once`) and
    // provisioning-only: the work split below is unchanged, so serial == parallel stays bit-exact.
    crate::ensure_global_pool();
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

    #[test]
    fn boundary_extents_are_safe() {
        // P14.6: the backward writes a finite dx for every element of a well-formed row.
        let (rows, cols) = (2usize, 3usize);
        let x = vec![0.5f32, -1.0, 2.0, 0.0, 1.0, -0.5];
        let dy = vec![1.0f32; rows * cols];
        let gamma = vec![1.0f32; cols];
        let mut dx = vec![f32::from_bits(0x7fc0_0001); rows * cols];
        let eps_bits = (1e-5f32).to_bits() as i64;
        unsafe {
            wukong_layernorm_bwd_f32(
                x.as_ptr(),
                dy.as_ptr(),
                gamma.as_ptr(),
                dx.as_mut_ptr(),
                0,
                cols as i64,
                eps_bits,
            );
        }
        assert!(
            dx.iter().all(|v| v.to_bits() == 0x7fc0_0001),
            "nonpositive rows must not write dx"
        );
        unsafe {
            wukong_layernorm_bwd_f32(
                x.as_ptr(),
                dy.as_ptr(),
                gamma.as_ptr(),
                dx.as_mut_ptr(),
                rows as i64,
                cols as i64,
                eps_bits,
            );
        }
        assert!(
            dx.iter().all(|v| v.is_finite()),
            "scalar layernorm backward must write finite dx"
        );
    }

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
    fn reference(
        x: &[f32],
        dy: &[f32],
        gamma: Option<&[f32]>,
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
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
        // Shapes straddle the 8-lane edge and the old 8-row gate (the test build forks every shape
        // with >= 2 rows, see par.rs).
        for &(rows, cols) in &[
            (1usize, 1usize),
            (3, 7),
            (4, 8),
            (9, 33),
            (16, 100),
            (64, 257),
            (40, 1000),
            (12, 512),
            (10, 768),
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
                    wukong_layernorm_bwd_f32(
                        x.as_ptr(),
                        dy.as_ptr(),
                        gptr,
                        got.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                        EPS.to_bits() as i64,
                    );
                    wukong_layernorm_bwd_f32_parallel(
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
        // Sizes exercising every seam the two blockings introduce: rows shorter than 8, a ragged
        // tail, the 8-element remainder loop, and the BLOCK = 32 blocked body at its edges
        // (31/32/33, 63/64/65, 95/96/97) plus the real transformer widths.
        for &n in &[
            1usize, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 39, 40, 63, 64, 65, 71, 95, 96, 97,
            100, 128, 257, 512, 768, 1000, 1024,
        ] {
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
                assert_eq!(
                    ms.to_bits(),
                    mv.to_bits(),
                    "mean scalar!=avx2 n={n} g={use_gamma}"
                );
                assert_eq!(
                    rs.to_bits(),
                    rv.to_bits(),
                    "rstd scalar!=avx2 n={n} g={use_gamma}"
                );
                assert_eq!(
                    a1s.to_bits(),
                    a1v.to_bits(),
                    "s1/C scalar!=avx2 n={n} g={use_gamma}"
                );
                assert_eq!(
                    a2s.to_bits(),
                    a2v.to_bits(),
                    "s2/C scalar!=avx2 n={n} g={use_gamma}"
                );
                // apply: scalar vs avx2 must agree bit-for-bit.
                let mut a = vec![0.0f32; n];
                let mut b = vec![0.0f32; n];
                unsafe {
                    apply_row_scalar(
                        x.as_ptr(),
                        dy.as_ptr(),
                        gptr,
                        a.as_mut_ptr(),
                        n,
                        ms,
                        rs,
                        a1s,
                        a2s,
                    );
                    apply_row_avx2(
                        x.as_ptr(),
                        dy.as_ptr(),
                        gptr,
                        b.as_mut_ptr(),
                        n,
                        mv,
                        rv,
                        a1v,
                        a2v,
                    );
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

    /// The twin test above runs on finite, well-behaved data. This one pins the AVX2 kernel to its
    /// scalar twin on the values that usually desync an accumulator layout: **signed zeros** (which
    /// distinguish `+0 + -0` orderings) and a **NaN placed past the first `BLOCK`**, so it lands in a
    /// non-zero accumulator chain and has to survive the halving-tree combine in the same lane on
    /// both sides. There is no `max`/`min` fold anywhere in this kernel, so there is no `maxps` tie
    /// rule to mirror; every operation is add / sub / mul / fma, whose NaN result is the quieted
    /// first NaN source on both the scalar `mul_add` and the AVX2 `fmadd`.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_on_nan_and_signed_zero_rows() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for &n in &[8usize, 33, 40, 64, 96, 100, 129] {
            for case in 0..4 {
                let mut x = fill_x(n);
                let mut dy = fill_dy(n);
                let gamma = fill_gamma(n);
                match case {
                    0 => {
                        // +0 / -0 sprinkled through every stream.
                        for i in (0..n).step_by(3) {
                            x[i] = if i % 2 == 0 { 0.0 } else { -0.0 };
                        }
                        for i in (0..n).step_by(5) {
                            dy[i] = -0.0;
                        }
                    }
                    1 => {
                        // A whole row of one value: every deviation is an exact zero, var is 0 and
                        // rstd saturates at 1/sqrt(eps).
                        for v in x.iter_mut() {
                            *v = -1.5;
                        }
                    }
                    2 => x[if n > 34 { 34 } else { n - 1 }] = f32::NAN,
                    _ => dy[if n > 37 { 37 } else { n - 1 }] = f32::NAN,
                }
                for use_gamma in [false, true] {
                    let gptr = if use_gamma {
                        gamma.as_ptr()
                    } else {
                        std::ptr::null()
                    };
                    let (ms, rs, a1s, a2s) =
                        unsafe { stats_row_scalar(x.as_ptr(), dy.as_ptr(), gptr, n, EPS) };
                    let (mv, rv, a1v, a2v) =
                        unsafe { stats_row_avx2(x.as_ptr(), dy.as_ptr(), gptr, n, EPS) };
                    for (sv, vv, what) in [
                        (ms, mv, "mean"),
                        (rs, rv, "rstd"),
                        (a1s, a1v, "s1/C"),
                        (a2s, a2v, "s2/C"),
                    ] {
                        assert_eq!(
                            sv.to_bits(),
                            vv.to_bits(),
                            "{what} scalar!=avx2 n={n} case={case} g={use_gamma}: {sv} vs {vv}"
                        );
                    }
                    let mut a = vec![0.0f32; n];
                    let mut b = vec![0.0f32; n];
                    unsafe {
                        apply_row_scalar(
                            x.as_ptr(),
                            dy.as_ptr(),
                            gptr,
                            a.as_mut_ptr(),
                            n,
                            ms,
                            rs,
                            a1s,
                            a2s,
                        );
                        apply_row_avx2(
                            x.as_ptr(),
                            dy.as_ptr(),
                            gptr,
                            b.as_mut_ptr(),
                            n,
                            mv,
                            rv,
                            a1v,
                            a2v,
                        );
                    }
                    for i in 0..n {
                        assert_eq!(
                            a[i].to_bits(),
                            b[i].to_bits(),
                            "apply scalar!=avx2 n={n} case={case} g={use_gamma} i={i}"
                        );
                    }
                }
            }
        }
    }

    /// The variance/`s1`/`s2` pass fusion is only safe because it keeps `(x - mean)` explicit. The
    /// tempting further fusion — accumulate `Σx` and `Σx²` in the *same* pass and take
    /// `var = Σx²/C - mean²` — would collapse the kernel to two passes, and no fixture built from a
    /// zero-mean generator could see what it costs. This test builds a row with `mean ≈ 1e4` and
    /// `std ≈ 0.35` and compares both formulations against an f64 reference over the *same* f32
    /// inputs. It asserts a **relation**, not an absolute error budget: the shipped formulation must
    /// beat the cancelling one by at least two decades. It does so by roughly six, because
    /// `Σx²/C ≈ 1e8` has an f32 ULP of 8 while the variance it is supposed to reveal is 0.125 — the
    /// subtraction returns noise.
    ///
    /// The shipped kernel's own error here is set by the f32 mean: a 512-element row sums to 5.1e6,
    /// whose f32 ULP is 0.5, so `mean` lands a few ULP off and that error is ~1e-3 of the 0.35 std.
    /// This is a property of f32 on an ill-conditioned row, not of the change — the four-chain
    /// pairwise mean introduced here is strictly *more* accurate than the single chain it replaced.
    #[test]
    fn variance_pass_keeps_x_minus_mean_rather_than_cancelling() {
        const COLS: usize = 512;
        let big = 1.0e4f32;
        let x: Vec<f32> = (0..COLS)
            .map(|i| big + ((i as f32) * 0.017).sin() * 0.5)
            .collect();
        let dy = fill_dy(COLS);
        let gamma = fill_gamma(COLS);
        let (_, rstd, _, _) =
            unsafe { stats_row_scalar(x.as_ptr(), dy.as_ptr(), gamma.as_ptr(), COLS, EPS) };

        // f64 truth over the identical f32 inputs.
        let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let c = COLS as f64;
        let mean_t = xd.iter().sum::<f64>() / c;
        let var_t = xd.iter().map(|&v| (v - mean_t) * (v - mean_t)).sum::<f64>() / c;
        let rstd_t = 1.0 / (var_t + EPS as f64).sqrt();

        // The cancelling rewrite, in f32, exactly as a "one fewer pass" simplification would spell it.
        let mut sx = 0.0f32;
        let mut sxx = 0.0f32;
        for &v in &x {
            sx += v;
            sxx = v.mul_add(v, sxx);
        }
        let m2 = sx / COLS as f32;
        let var_cancel = sxx / COLS as f32 - m2 * m2;
        let rstd_cancel = 1.0f32 / (var_cancel.max(0.0) + EPS).sqrt();

        let err_kernel = ((rstd as f64) - rstd_t).abs() / rstd_t;
        let err_cancel = ((rstd_cancel as f64) - rstd_t).abs() / rstd_t;
        assert!(
            err_cancel > 100.0 * err_kernel.max(1e-12),
            "the cancelling form is supposed to be far worse here, but measured \
             kernel rel err {err_kernel:.3e} vs cancelling {err_cancel:.3e} \
             (rstd {rstd} / {rstd_cancel} vs f64 {rstd_t}) — either the guard row stopped being \
             ill-conditioned or the kernel started cancelling"
        );
        // And the shipped side still tracks f64 within the conditioning bound derived above.
        assert!(
            err_kernel < 1e-2,
            "kernel rstd rel err {err_kernel:.3e} on the large-mean row: {rstd} vs f64 {rstd_t}"
        );
    }

    #[test]
    fn matches_f64_reference() {
        // Independent f64 computation of the LayerNorm input gradient, tight relative tolerance.
        let (rows, cols) = (8usize, 768usize);
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
                wukong_layernorm_bwd_f32(
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
