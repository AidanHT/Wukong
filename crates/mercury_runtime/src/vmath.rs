//! Vectorized elementwise transcendentals — the **256-bit AVX2** path the Cranelift backend cannot
//! emit (`f32x8` does not legalize, so the generic vectorizer is stuck at 128-bit SSE). These are the
//! transformer activation family — `exp`, `log`, `tanh`, `sigmoid` — and they are *compute*-bound (a
//! ~20-flop minimax polynomial per element), so doubling the SIMD width nearly doubles throughput.
//!
//! The compiler recognizes an elementwise `for i { out[i] = f(x[i]) }` loop and lowers it to one
//! [`mercury_vmath_f32`] call (the same play as the matmul→GEMM dispatch). The interpreter marshals
//! its abstract memory through the **identical** kernel, so the differential oracle stays bit-for-bit
//! exact even though the kernel reassociates across lanes.
//!
//! The per-element op sequence mirrors the inlined MIR polynomials in `mercury_mir_build`
//! (`emit_exp_f32` / `emit_log_f32`) — same Cephes constants, same FMA structure — so a dispatched
//! `exp(x)` agrees with a composed/scalar `exp(x)`. The scalar tail and the no-AVX2 fallback use the
//! scalar twins (`exp1`/`log1`), which share the constants, so every lane of every path agrees.

// --- op codes (shared with the recognizer in mercury_mir_build) ------------------------------------
pub const VM_EXP: i64 = 0;
pub const VM_LOG: i64 = 1;
pub const VM_TANH: i64 = 2;
pub const VM_SIGMOID: i64 = 3;
pub const VM_RELU: i64 = 4;
pub const VM_SILU: i64 = 5;
pub const VM_GELU: i64 = 6;
pub const VM_ELU: i64 = 7;
pub const VM_LEAKYRELU: i64 = 8;
pub const VM_SOFTPLUS: i64 = 9;
pub const VM_MISH: i64 = 10;

/// Leaky-ReLU negative-slope (the conventional 0.01); fixed so `leaky_relu` stays a single-arg
/// intrinsic that fits the elementwise dispatch.
const LEAKY_ALPHA: f32 = 0.01;

// GELU (tanh approximation) constants: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
const GELU_C0: f32 = 0.797_884_6; // √(2/π)
const GELU_C1: f32 = 0.044715;

// --- Cephes single-precision constants (mirror mercury_mir_build's `exp`/`log` poly constants) ----
const LOG2EF: f32 = std::f32::consts::LOG2_E;
const EXP_MAGIC: f32 = 12582912.0; // 1.5 * 2^23 — round-to-nearest-even via add-then-subtract
const EXP_C1: f32 = 0.693_359_4; // ln2, high part
const EXP_C2: f32 = -2.1219444e-4; // ln2, low correction
const EXP_HI: f32 = 88.376_26;
const EXP_LO: f32 = -88.376_26;
const EXP_P: [f32; 6] = [
    1.987_569_1e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    1.666_666_6e-1,
    5e-1,
];
const LOG_SQRTHF: f32 = std::f32::consts::FRAC_1_SQRT_2; // √0.5
const INV_2P23: f32 = 1.0 / 8_388_608.0; // 2^-23 (exact)
const LOG_P: [f32; 9] = [
    7.037_683_6e-2,
    -1.151_461e-1,
    1.167_699_84e-1,
    -1.242_014_1e-1,
    1.424_932_3e-1,
    -1.666_805_7e-1,
    2.000_071_4e-1,
    -2.499_999_4e-1,
    3.333_333e-1,
];

// --- scalar twins (the AVX2 tail + the no-AVX2 fallback; mirror the MIR poly element-for-element) --

/// `e^x` (≈1 ULP), the Cephes single-precision algorithm: range-reduce `x = r + n·ln2`, a degree-5
/// minimax poly for `e^r`, then scale by `2^n` assembled from the IEEE-754 exponent field.
///
/// `pub(crate)` so the fused-softmax kernel in `norm.rs` reuses the *exact* same scalar exp — its
/// AVX2 path uses [`exp8`] and its tail uses this, so a fused `softmax` agrees lane-for-lane with a
/// dispatched `exp` and the differential oracle stays bit-for-bit exact.
#[inline]
pub(crate) fn exp1(x: f32) -> f32 {
    // `min` then `max`, not `clamp`: this mirrors the AVX2 `_mm256_min_ps`/`_mm256_max_ps` order
    // lane-for-lane (incl. their NaN behavior), which is what keeps the scalar tail bit-identical.
    #[allow(clippy::manual_clamp)]
    let x = x.min(EXP_HI).max(EXP_LO);
    let t = x.mul_add(LOG2EF, EXP_MAGIC);
    let n = t - EXP_MAGIC;
    let r = n.mul_add(-EXP_C1, x);
    let r = n.mul_add(-EXP_C2, r);
    let mut p = EXP_P[0];
    p = p.mul_add(r, EXP_P[1]);
    p = p.mul_add(r, EXP_P[2]);
    p = p.mul_add(r, EXP_P[3]);
    p = p.mul_add(r, EXP_P[4]);
    p = p.mul_add(r, EXP_P[5]);
    let r2 = r * r;
    let p = p.mul_add(r2, r) + 1.0;
    let pow2 = f32::from_bits((((n as i32) + 127) << 23) as u32);
    p * pow2
}

/// `ln(x)` for `x > 0` (≈1 ULP), Cephes single-precision: decompose `x = m·2^e`, a degree-8 minimax
/// poly for `log(m)`, add back `e·ln2` with the same hi/lo split `exp` uses.
#[inline]
fn log1(x: f32) -> f32 {
    let bits = x.to_bits() as i32;
    let epart = bits & 0x7F80_0000;
    let efield = (epart as f32) * INV_2P23;
    let mut e = efield - 126.0;
    let mant = bits & 0x007F_FFFF;
    let mbits = mant | 0x3F00_0000;
    let mut m = f32::from_bits(mbits as u32);
    let lt = m < LOG_SQRTHF;
    let m_lt = (m + m) - 1.0;
    let m_ge = m - 1.0;
    m = if lt { m_lt } else { m_ge };
    if lt {
        e -= 1.0;
    }
    let z = m * m;
    let mut p = LOG_P[0];
    for &c in &LOG_P[1..] {
        p = p.mul_add(m, c);
    }
    let pm = p * m;
    let mut y = pm * z;
    y = e.mul_add(EXP_C2, y);
    y = z.mul_add(-0.5, y);
    let r = m + y;
    e.mul_add(EXP_C1, r)
}

/// `tanh(x) = 1 - 2/(e^{2x}+1)` — the exp-based form (the activation benchmarks and the MIR lowering
/// both use it). The clamped `exp` keeps it finite and saturating at ±1 for large |x|.
#[inline]
fn tanh1(x: f32) -> f32 {
    1.0 - 2.0 / (exp1(2.0 * x) + 1.0)
}

/// `sigmoid(x) = 1/(1 + e^{-x})`.
#[inline]
fn sigmoid1(x: f32) -> f32 {
    1.0 / (1.0 + exp1(-x))
}

/// `silu(x) = x·sigmoid(x)` (swish) — the Llama / modern-transformer gating activation. `pub(crate)`
/// so the GEMM fused epilogue (`gemm.rs`) applies the *identical* scalar form a standalone `silu(...)`
/// would, keeping a fused `silu(x·Wᵀ+b)` bit-equal to the unfused `{ t = x·Wᵀ+b; silu(t) }`.
#[inline]
pub(crate) fn silu1(x: f32) -> f32 {
    x * sigmoid1(x)
}

/// `gelu(x)` (tanh approximation) — the BERT/GPT-2/ViT activation. `pub(crate)` for the GEMM fused
/// epilogue (see [`silu1`]).
#[inline]
pub(crate) fn gelu1(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = GELU_C0 * GELU_C1.mul_add(x3, x);
    (0.5 * x) * (1.0 + tanh1(inner))
}

/// `elu(x) = x>0 ? x : e^x − 1` (α=1) — the exponential linear unit (smooth, saturating negative
/// tail). Reuses [`exp1`], so the AVX2 [`elu8`], the tail, and the composed scalar MIR all agree.
#[inline]
fn elu1(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        exp1(x) - 1.0
    }
}

/// `leaky_relu(x) = x>0 ? x : 0.01·x` — the leaky rectifier (a small negative slope, no saturation).
#[inline]
fn leakyrelu1(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        LEAKY_ALPHA * x
    }
}

/// `softplus(x) = ln(1 + e^x)`, the smooth ReLU — evaluated in the numerically stable form
/// `max(x,0) + ln(1 + e^{−|x|})` so the `exp` never overflows for large positive `x` (it saturates to
/// `x`) and underflows cleanly to 0 for large negative `x`. Reuses [`exp1`]/[`log1`], so the AVX2
/// [`softplus8`], the tail, and the composed scalar MIR agree.
#[inline]
fn softplus1(x: f32) -> f32 {
    x.max(0.0) + log1(1.0 + exp1(-x.abs()))
}

/// `mish(x) = x · tanh(softplus(x))` — the smooth, self-gated activation (YOLOv4 / modern vision).
#[inline]
fn mish1(x: f32) -> f32 {
    x * tanh1(softplus1(x))
}

/// Scalar dispatch for one element (used by the AVX2 tail and the no-AVX2 fallback).
#[inline]
fn apply1(op: i64, x: f32) -> f32 {
    match op {
        VM_EXP => exp1(x),
        VM_LOG => log1(x),
        VM_TANH => tanh1(x),
        VM_SIGMOID => sigmoid1(x),
        VM_RELU => x.max(0.0),
        VM_SILU => silu1(x),
        VM_GELU => gelu1(x),
        VM_ELU => elu1(x),
        VM_LEAKYRELU => leakyrelu1(x),
        VM_SOFTPLUS => softplus1(x),
        VM_MISH => mish1(x),
        _ => x,
    }
}

/// `out[i] = f(x[i])` for `i in 0..n`, where `f` is selected by `op` (see the `VM_*` codes). Uses the
/// 256-bit AVX2 kernels when available (8 lanes/step + a scalar tail), else the scalar fallback. `x`
/// and `out` may alias (the recognizer allows in-place activations).
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn mercury_vmath_f32(x: *const f32, out: *mut f32, n: i64, op: i64) {
    if n <= 0 {
        return;
    }
    let n = n as usize;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; buffers valid for n by the caller contract.
            unsafe { vmath_avx2(x, out, n, op) };
            return;
        }
    }
    for i in 0..n {
        // SAFETY: i < n; buffers valid for n.
        unsafe { *out.add(i) = apply1(op, *x.add(i)) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vmath_avx2(x: *const f32, out: *mut f32, n: usize, op: i64) {
    use std::arch::x86_64::*;
    let f: unsafe fn(__m256) -> __m256 = match op {
        VM_EXP => exp8,
        VM_LOG => log8,
        VM_TANH => tanh8,
        VM_SIGMOID => sigmoid8,
        VM_RELU => relu8,
        VM_SILU => silu8,
        VM_GELU => gelu8,
        VM_ELU => elu8,
        VM_LEAKYRELU => leakyrelu8,
        VM_SOFTPLUS => softplus8,
        VM_MISH => mish8,
        _ => return,
    };
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.add(i));
        _mm256_storeu_ps(out.add(i), f(v));
        i += 8;
    }
    // Scalar tail (same poly as the lanes, via the scalar twins) for the final < 8 elements.
    while i < n {
        *out.add(i) = apply1(op, *x.add(i));
        i += 1;
    }
}

// --- AVX2 kernels (mirror the scalar twins lane-for-lane) -----------------------------------------

/// `pub(crate)` so `norm.rs`'s AVX2 softmax uses the *exact* same 8-lane exp as a dispatched
/// `exp` loop — keeping fused softmax bit-identical to the composed form.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn exp8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_min_ps(x, _mm256_set1_ps(EXP_HI));
    let x = _mm256_max_ps(x, _mm256_set1_ps(EXP_LO));
    let t = _mm256_fmadd_ps(x, _mm256_set1_ps(LOG2EF), _mm256_set1_ps(EXP_MAGIC));
    let n = _mm256_sub_ps(t, _mm256_set1_ps(EXP_MAGIC));
    let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-EXP_C1), x);
    let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-EXP_C2), r);
    let mut p = _mm256_set1_ps(EXP_P[0]);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_P[1]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_P[2]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_P[3]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_P[4]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_P[5]));
    let r2 = _mm256_mul_ps(r, r);
    let p = _mm256_fmadd_ps(p, r2, r);
    let p = _mm256_add_ps(p, _mm256_set1_ps(1.0));
    // 2^n = bitcast((n + 127) << 23). n is an exact integer in f32, so the truncating convert is exact.
    let ni = _mm256_cvttps_epi32(n);
    let biased = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    let pow2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(biased));
    _mm256_mul_ps(p, pow2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let bits = _mm256_castps_si256(x);
    let epart = _mm256_and_si256(bits, _mm256_set1_epi32(0x7F80_0000));
    let efield = _mm256_mul_ps(_mm256_cvtepi32_ps(epart), _mm256_set1_ps(INV_2P23));
    let mut e = _mm256_sub_ps(efield, _mm256_set1_ps(126.0));
    let mant = _mm256_and_si256(bits, _mm256_set1_epi32(0x007F_FFFF));
    let mbits = _mm256_or_si256(mant, _mm256_set1_epi32(0x3F00_0000));
    let mut m = _mm256_castsi256_ps(mbits);
    let lt = _mm256_cmp_ps::<_CMP_LT_OQ>(m, _mm256_set1_ps(LOG_SQRTHF));
    let one = _mm256_set1_ps(1.0);
    let m_lt = _mm256_sub_ps(_mm256_add_ps(m, m), one);
    let m_ge = _mm256_sub_ps(m, one);
    m = _mm256_blendv_ps(m_ge, m_lt, lt);
    e = _mm256_blendv_ps(e, _mm256_sub_ps(e, one), lt);
    let z = _mm256_mul_ps(m, m);
    let mut p = _mm256_set1_ps(LOG_P[0]);
    for &c in &LOG_P[1..] {
        p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(c));
    }
    let pm = _mm256_mul_ps(p, m);
    let mut y = _mm256_mul_ps(pm, z);
    y = _mm256_fmadd_ps(e, _mm256_set1_ps(EXP_C2), y);
    y = _mm256_fmadd_ps(z, _mm256_set1_ps(-0.5), y);
    let r = _mm256_add_ps(m, y);
    _mm256_fmadd_ps(e, _mm256_set1_ps(EXP_C1), r)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // 1 - 2/(exp(2x)+1)
    let e = exp8(_mm256_mul_ps(x, _mm256_set1_ps(2.0)));
    let d = _mm256_add_ps(e, _mm256_set1_ps(1.0));
    _mm256_sub_ps(_mm256_set1_ps(1.0), _mm256_div_ps(_mm256_set1_ps(2.0), d))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // 1/(1+exp(-x))
    let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
    let d = _mm256_add_ps(_mm256_set1_ps(1.0), e);
    _mm256_div_ps(_mm256_set1_ps(1.0), d)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn relu8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_max_ps(x, _mm256_setzero_ps())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // x * sigmoid(x)
    _mm256_mul_ps(x, sigmoid8(x))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // 0.5·x·(1 + tanh(C0·(x + C1·x³))) — mirrors gelu1 op-for-op.
    let x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
    let inner = _mm256_mul_ps(
        _mm256_set1_ps(GELU_C0),
        _mm256_fmadd_ps(_mm256_set1_ps(GELU_C1), x3, x),
    );
    let onep = _mm256_add_ps(_mm256_set1_ps(1.0), tanh8(inner));
    _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5), x), onep)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn elu8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // x>0 ? x : exp(x)−1 — blendv picks `x` where the (x>0) mask is set, else the saturating tail.
    // Mirrors elu1: the negative branch is exp8 (== exp1), so lanes and tail agree.
    let em1 = _mm256_sub_ps(exp8(x), _mm256_set1_ps(1.0));
    let pos = _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_setzero_ps());
    _mm256_blendv_ps(em1, x, pos)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn leakyrelu8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // x>0 ? x : 0.01·x — mirrors leakyrelu1.
    let scaled = _mm256_mul_ps(x, _mm256_set1_ps(LEAKY_ALPHA));
    let pos = _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_setzero_ps());
    _mm256_blendv_ps(scaled, x, pos)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softplus8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // max(x,0) + log(1 + exp(-|x|)) — the stable softplus, mirroring softplus1. `|x|` clears the sign
    // bit (== f32::abs), matching the scalar twin bit-for-bit.
    let absx = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF)));
    let nabs = _mm256_sub_ps(_mm256_setzero_ps(), absx);
    let e = exp8(nabs);
    let l = log8(_mm256_add_ps(_mm256_set1_ps(1.0), e));
    let relu = _mm256_max_ps(x, _mm256_setzero_ps());
    _mm256_add_ps(relu, l)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mish8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // x · tanh(softplus(x)) — mirrors mish1.
    _mm256_mul_ps(x, tanh8(softplus8(x)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dispatched kernel is ≈1 ULP of `libm` for exp/log/tanh/sigmoid over a representative
    /// range — the accuracy the cross-language checksum and the activation tests rely on.
    // (op code, libm reference, tolerance) — a small table, so allow the tuple type.
    #[test]
    #[allow(clippy::type_complexity)]
    fn vmath_matches_libm() {
        let xs: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) * 0.01).collect();
        let mut out = vec![0.0f32; xs.len()];
        let cases: &[(i64, fn(f32) -> f32, f32)] = &[
            (VM_EXP, |x| x.exp(), 2e-5),
            (VM_TANH, |x| x.tanh(), 2e-5),
            (VM_SIGMOID, |x| 1.0 / (1.0 + (-x).exp()), 2e-5),
            (VM_SILU, |x| x / (1.0 + (-x).exp()), 2e-5),
            // GELU tanh-approx reference, reusing the module's f32 constants (GELU_C0=√(2/π)).
            (
                VM_GELU,
                |x| 0.5 * x * (1.0 + (GELU_C0 * (x + GELU_C1 * x * x * x)).tanh()),
                5e-5,
            ),
            (VM_ELU, |x| if x > 0.0 { x } else { x.exp() - 1.0 }, 2e-5),
            (
                VM_LEAKYRELU,
                |x| if x > 0.0 { x } else { LEAKY_ALPHA * x },
                1e-6,
            ),
            // softplus/mish: the mixed bound's absolute floor covers the underflowing tail
            // (softplus(−20)≈2e−9 → 0) that a pure-relative check would reject.
            (VM_SOFTPLUS, |x| (1.0 + x.exp()).ln(), 1e-4),
            (VM_MISH, |x| x * (1.0 + x.exp()).ln().tanh(), 2e-4),
        ];
        for &(op, libm, tol) in cases {
            unsafe {
                mercury_vmath_f32(xs.as_ptr(), out.as_mut_ptr(), xs.len() as i64, op);
            }
            for (i, &x) in xs.iter().enumerate() {
                let want = libm(x);
                let got = out[i];
                assert!(
                    (got - want).abs() <= tol + tol * want.abs(),
                    "op {op} x={x}: got {got} want {want}"
                );
            }
        }
        // log over positive inputs only.
        let xs: Vec<f32> = (1..4096).map(|i| i as f32 * 0.05).collect();
        let mut out = vec![0.0f32; xs.len()];
        unsafe {
            mercury_vmath_f32(xs.as_ptr(), out.as_mut_ptr(), xs.len() as i64, VM_LOG);
        }
        for (i, &x) in xs.iter().enumerate() {
            let want = x.ln();
            assert!(
                (out[i] - want).abs() <= 2e-5 + 2e-5 * want.abs(),
                "log x={x}: got {} want {want}",
                out[i]
            );
        }
    }

    /// The AVX2 lanes and the scalar tail/fallback must agree element-for-element, so a length that is
    /// not a multiple of 8 produces a consistent result regardless of where the tail starts.
    #[test]
    fn vmath_tail_matches_lanes() {
        let xs: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.013).collect();
        for op in [
            VM_EXP, VM_LOG, VM_TANH, VM_SIGMOID, VM_RELU, VM_SILU, VM_GELU,
        ] {
            if op == VM_LOG {
                continue; // negative inputs are out of log's domain
            }
            let mut full = vec![0.0f32; xs.len()];
            unsafe {
                mercury_vmath_f32(xs.as_ptr(), full.as_mut_ptr(), xs.len() as i64, op);
            }
            // Recompute each element scalar and require an exact match with the kernel output.
            for (i, &x) in xs.iter().enumerate() {
                let s = apply1(op, x);
                assert_eq!(full[i].to_bits(), s.to_bits(), "op {op} i {i} x {x}");
            }
        }
    }
}
