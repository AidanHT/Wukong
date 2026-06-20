//! Vectorized elementwise transcendentals — the **256-bit AVX2** path the Cranelift backend cannot
//! emit (`f32x8` does not legalize, so the generic vectorizer is stuck at 128-bit SSE). These are the
//! transformer/vision activation family — `exp`, `log`, `tanh`, `sigmoid`, `relu`, `silu`, `gelu`,
//! `elu`, `leaky_relu`, `softplus`, `mish`, `selu`, `tanhshrink`, `hardsigmoid`, `hardswish` — and the
//! transcendental ones are *compute*-bound (a ~20-flop minimax polynomial per element, more for the
//! composed ones), so doubling the SIMD width nearly doubles throughput. The transcendental
//! activations all build on the shared `exp`/`log` polynomials (e.g. `silu = x·sigmoid`,
//! `softplus = ln(1+eˣ)`, `mish = x·tanh(softplus)`, `selu` = scaled `elu`), so one ≈1-ULP `exp` keeps
//! the family accurate; the piecewise ones (`relu`, `leaky_relu`, `hardsigmoid`, `hardswish`) use
//! min/max whose scalar twins match the AVX2 `max_ps`/`min_ps` bit-for-bit (incl. NaN/±0). All are
//! bit-identical across backends.
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
pub const VM_SELU: i64 = 11;
pub const VM_TANHSHRINK: i64 = 12;
pub const VM_HARDSIGMOID: i64 = 13;
pub const VM_HARDSWISH: i64 = 14;
pub const VM_SIN: i64 = 15;
pub const VM_COS: i64 = 16;
pub const VM_ERF: i64 = 17;
pub const VM_EXP2: i64 = 18;
pub const VM_LOG2: i64 = 19;
pub const VM_SINH: i64 = 20;
pub const VM_COSH: i64 = 21;
pub const VM_ASINH: i64 = 22;
pub const VM_ACOSH: i64 = 23;
pub const VM_ATANH: i64 = 24;
pub const VM_ATAN: i64 = 25;
pub const VM_EXPM1: i64 = 26;
pub const VM_LOG1P: i64 = 27;

/// `1/6` in f32 — the hard-sigmoid/hard-swish scale. Used as a multiply (not a divide) identically in
/// the scalar twin, the AVX2 lanes, and the composed MIR, so all three agree bit-for-bit.
const INV6: f32 = 1.0 / 6.0;

// SELU (self-normalizing networks, Klambauer 2017) constants — the fixed λ, α that make the
// activation variance-preserving.
const SELU_LAMBDA: f32 = 1.050_700_98;
const SELU_ALPHA: f32 = 1.673_263_2;

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

// --- sin/cos and erf constants (mirror mercury_mir_build's inlined `emit_trig_f32`/`emit_erf_f32`
// poly constants, so a dispatched `sin`/`cos`/`erf` loop agrees with a composed/scalar one) ---------
// The `f64 as f32` casts reproduce exactly what `splat_const_f` does to the matching f64 literals
// there, keeping the dispatched kernel bit-identical to the inlined form.
const TWO_OVER_PI: f32 = std::f64::consts::FRAC_2_PI as f32; // 2/π — quadrant count = round(x·2/π)
const PIO2_1: f32 = 1.5703125_f64 as f32; // π/2 high (2 × Cephes DP1)
const PIO2_2: f32 = 4.837_512_969_970_703e-4_f64 as f32; // π/2 mid (2 × DP2)
const PIO2_3: f32 = 7.549_789_954_891_88e-8_f64 as f32; // π/2 low (2 × DP3)
const SIN_P: [f32; 3] = [
    -1.9515295891e-4_f64 as f32,
    8.3321608736e-3_f64 as f32,
    -1.6666654611e-1_f64 as f32,
];
const COS_P: [f32; 3] = [
    2.443_315_711_809_948e-5_f64 as f32,
    -1.388_731_625_493_765e-3_f64 as f32,
    4.166_664_568_298_827e-2_f64 as f32,
];
// erf (Abramowitz–Stegun 7.1.26): erf(|x|) = 1 − (a₁t + … + a₅t⁵)·e^(−x²), t = 1/(1 + P·|x|).
const ERF_P: f32 = 0.327_591_1_f64 as f32;
const ERF_A: [f32; 5] = [
    0.254_829_592_f64 as f32,
    -0.284_496_736_f64 as f32,
    1.421_413_741_f64 as f32,
    -1.453_152_027_f64 as f32,
    1.061_405_429_f64 as f32,
];

// Base-conversion constants for exp2/log2 (defined as `f64 as f32` to match an inlined
// `emit_exp(x*ln2)` / `emit_log(x)*log2e` bit-for-bit).
const LN_2: f32 = std::f64::consts::LN_2 as f32; // exp2(x) = exp(x·ln2)
const LOG2_E: f32 = std::f64::consts::LOG2_E as f32; // log2(x) = log(x)·log2(e)

// atan (Cephes `atanf`): a 3-region reduction of |x| at the two breakpoints tan(π/8)=√2−1 and
// tan(3π/8)=1+√2, each mapping into [0, tan(π/8)] where a degree-3 odd minimax poly is ≈1 ULP. The
// `f64 as f32` casts mirror `splat_const_f` in mir_build so the dispatched kernel equals the inlined form.
const ATAN_TAN_3PI8: f32 = 2.414213562373095_f64 as f32; // tan(3π/8) = 1 + √2
const ATAN_TAN_PI8: f32 = 0.4142135623730950_f64 as f32; // tan(π/8) = √2 − 1
const ATAN_PIO2: f32 = std::f64::consts::FRAC_PI_2 as f32; // π/2 offset (big region)
const ATAN_PIO4: f32 = std::f64::consts::FRAC_PI_4 as f32; // π/4 offset (mid region)
const ATAN_P: [f32; 4] = [
    0.080_537_444_953_8_f64 as f32,
    -0.138_776_856_032_f64 as f32,
    0.199_777_106_478_f64 as f32,
    -0.333_329_491_539_f64 as f32,
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

/// `selu(x) = λ·(x>0 ? x : α·(eˣ−1))` — the scaled ELU of self-normalizing networks (fixed λ, α).
/// A scaled [`elu1`]; reuses [`exp1`], so the AVX2 [`selu8`] and tail agree.
#[inline]
fn selu1(x: f32) -> f32 {
    SELU_LAMBDA
        * if x > 0.0 {
            x
        } else {
            SELU_ALPHA * (exp1(x) - 1.0)
        }
}

/// `tanhshrink(x) = x − tanh(x)` — the high-pass-shaped activation (audio/signal models). Reuses the
/// shared `tanh` (built on `exp`), so the AVX2 [`tanhshrink8`] and tail agree.
#[inline]
fn tanhshrink1(x: f32) -> f32 {
    x - tanh1(x)
}

/// `hardsigmoid(x) = clamp(x+3, 0, 6)/6` — the cheap piecewise-linear sigmoid of MobileNetV3 /
/// EfficientNet. The clamp is written as the manual branches that match `_mm256_max_ps`/`_mm256_min_ps`
/// (and the `Cmp+Select` MIR) bit-for-bit on NaN/±0 — `v>0`/`v<6` are false for NaN, so it returns the
/// bound, exactly as the SSE max/min instructions do (the same trick `velem`'s relu6 uses).
#[inline]
fn hardsigmoid1(x: f32) -> f32 {
    let y = x + 3.0;
    let lo = if y > 0.0 { y } else { 0.0 };
    let hi = if lo < 6.0 { lo } else { 6.0 };
    hi * INV6
}

/// `hardswish(x) = x · hardsigmoid(x)` — the MobileNetV3 self-gated activation.
#[inline]
fn hardswish1(x: f32) -> f32 {
    x * hardsigmoid1(x)
}

/// `sin(x)` (`is_cos == false`) or `cos(x)` (`true`), the Cephes single-precision algorithm: reduce
/// `x` to `r ∈ [−π/4, π/4]` by `q = round(x·2/π)` quadrants (the add-magic round-to-nearest), evaluate
/// the `sinf`/`cosf` minimax polys on `r`, and pick ±sin/±cos by `q mod 4`. Mirrors `emit_trig_f32`
/// op-for-op (same constants, same FMA structure, the quadrant blended with float-eq masks), so a
/// dispatched `sin(x)` loop agrees with a composed/scalar one. Accurate to ≈1 ULP for the |x| where the
/// 3-part π/2 split holds (RoPE angles, ≲ a few thousand); large |x| loses the reduction, as with libm.
#[inline]
fn sincos1(x: f32, is_cos: bool) -> f32 {
    let tt = x.mul_add(TWO_OVER_PI, EXP_MAGIC);
    let qf = tt - EXP_MAGIC;
    let r = qf.mul_add(-PIO2_1, x);
    let r = qf.mul_add(-PIO2_2, r);
    let r = qf.mul_add(-PIO2_3, r);
    let z = r * r;
    // sin_p(r) = r + (poly·z)·r,  poly = ((s₀z + s₁)z + s₂)
    let mut s = SIN_P[0];
    s = s.mul_add(z, SIN_P[1]);
    s = s.mul_add(z, SIN_P[2]);
    let sz = s * z;
    let sin_p = sz.mul_add(r, r);
    // cos_p(r) = (1 − 0.5z) + (poly·z²),  poly = ((c₀z + c₁)z + c₂)
    let mut c = COS_P[0];
    c = c.mul_add(z, COS_P[1]);
    c = c.mul_add(z, COS_P[2]);
    let z2 = z * z;
    let cz2 = c * z2;
    let hz = (-0.5f32).mul_add(z, 1.0);
    let cos_p = hz + cz2;
    // quadrant: (int)qf & 3, back to an exact float for the float-eq selects.
    let quad = ((qf as i32) & 3) as f32;
    // `* -1.0` (not unary `-`) to match the AVX2 `mulps` and `emit_trig`'s `FMul(_, -1)` on signed zero.
    let neg_sin = sin_p * -1.0;
    let neg_cos = cos_p * -1.0;
    let (a0, a1, a2, a3) = if is_cos {
        (cos_p, neg_sin, neg_cos, sin_p)
    } else {
        (sin_p, cos_p, neg_sin, neg_cos)
    };
    let sel23 = if quad == 2.0 { a2 } else { a3 };
    let sel123 = if quad == 1.0 { a1 } else { sel23 };
    if quad == 0.0 {
        a0
    } else {
        sel123
    }
}

/// `erf(x)` (Abramowitz–Stegun 7.1.26, ~1.5e-7 max error — f32-grade), reusing the shared [`exp1`].
/// Mirrors `emit_erf_f32` op-for-op (incl. `|x| = max(x, −x)` via compare+select and the odd-function
/// sign fixup), so a dispatched `erf(x)` loop agrees with a composed one — and gives the exact
/// (erf-based) GELU the original BERT/GPT-2 use.
#[inline]
fn erf1(x: f32) -> f32 {
    let negx = x * -1.0;
    let ax = if x > negx { x } else { negx };
    let t = 1.0 / ERF_P.mul_add(ax, 1.0);
    let mut h = ERF_A[4];
    h = h.mul_add(t, ERF_A[3]);
    h = h.mul_add(t, ERF_A[2]);
    h = h.mul_add(t, ERF_A[1]);
    h = h.mul_add(t, ERF_A[0]);
    let poly = h * t;
    let ax2 = ax * ax;
    let e = exp1(ax2 * -1.0);
    let mag = 1.0 - poly * e;
    if x >= 0.0 {
        mag
    } else {
        mag * -1.0
    }
}

/// `exp2(x) = 2^x = e^{x·ln2}` (FlashAttention-2 base-2 softmax, entropy in bits). Reuses [`exp1`].
#[inline]
fn exp2_1(x: f32) -> f32 {
    exp1(x * LN_2)
}

/// `log2(x) = ln(x)·log2(e)` (quantization bit-width, entropy/mutual-information in bits). Reuses [`log1`].
#[inline]
fn log2_1(x: f32) -> f32 {
    log1(x) * LOG2_E
}

/// `sinh(x) = (e^x − e^{−x})/2`. Reuses [`exp1`]; overflows like `libm` for large |x|.
#[inline]
fn sinh1(x: f32) -> f32 {
    (exp1(x) - exp1(x * -1.0)) * 0.5
}

/// `cosh(x) = (e^x + e^{−x})/2`. Reuses [`exp1`]; overflows like `libm` for large |x|.
#[inline]
fn cosh1(x: f32) -> f32 {
    (exp1(x) + exp1(x * -1.0)) * 0.5
}

/// `asinh(x) = sign(x)·ln(|x| + √(x²+1))` — the all-real inverse hyperbolic sine (Poincaré/hyperbolic
/// embeddings, the `symlog` robust activation). Reusing `|x|` (and restoring the sign by `copysign`)
/// evaluates the `log` on `|x| + √(…) ≥ 1`, dodging the catastrophic `x + √(x²+1)` cancellation that
/// the naive form suffers for large negative `x`. Reuses [`log1`]; `√`, the abs/sign bit-masks, are all
/// exact, so the scalar twin and the AVX2 [`asinh8`] agree bit-for-bit (the tail-match test pins it).
#[inline]
fn asinh1(x: f32) -> f32 {
    let ax = f32::from_bits(x.to_bits() & 0x7FFF_FFFF); // |x|
    let t = log1(ax + (ax * ax + 1.0).sqrt()); // ≥ 0
    f32::from_bits(t.to_bits() | (x.to_bits() & 0x8000_0000)) // copysign(t, x)
}

/// `acosh(x) = ln(x + √(x²−1))` for `x ≥ 1` (`NaN` below, matching `libm`'s domain). Reuses [`log1`].
#[inline]
fn acosh1(x: f32) -> f32 {
    let x2 = x * x;
    log1(x + (x2 - 1.0).sqrt())
}

/// `atanh(x) = ½·ln((1+x)/(1−x))` for `|x| < 1` (the Fisher z-transform; ±∞ at ±1). Reuses [`log1`].
#[inline]
fn atanh1(x: f32) -> f32 {
    let r = (1.0 + x) / (1.0 - x);
    log1(r) * 0.5
}

/// `atan(x)` (≈1 ULP), the Cephes single-precision algorithm: fold to `|x|`, reduce into
/// `[0, tan(π/8)]` by the two breakpoints (a degree-3 odd minimax poly there), then restore the
/// `π/4`/`π/2` offset and the sign. Branchless (all three region candidates computed, then selected)
/// so the scalar twin equals the AVX2 [`atan8`] lane-for-lane — the tail-match test pins it. The
/// headline ML use is angle/geometry ops and `atan2`-style positional schemes.
#[inline]
fn atan1(x: f32) -> f32 {
    let ax = f32::from_bits(x.to_bits() & 0x7FFF_FFFF); // |x|
    let big = ax > ATAN_TAN_3PI8;
    let mid = ax > ATAN_TAN_PI8; // includes `big`; `big` overrides below (mirrors if/else-if/else)
    let xr_mid = (ax - 1.0) / (ax + 1.0);
    let xr_big = -1.0 / ax;
    let mut xr = ax;
    xr = if mid { xr_mid } else { xr };
    xr = if big { xr_big } else { xr };
    let mut y = 0.0f32;
    y = if mid { ATAN_PIO4 } else { y };
    y = if big { ATAN_PIO2 } else { y };
    let z = xr * xr;
    let mut p = ATAN_P[0];
    p = p.mul_add(z, ATAN_P[1]);
    p = p.mul_add(z, ATAN_P[2]);
    p = p.mul_add(z, ATAN_P[3]);
    let pz = p * z;
    let pzx = pz * xr;
    let res = pzx + xr;
    let yf = y + res; // ≥ 0 for ax ≥ 0 (atan is odd)
    f32::from_bits(yf.to_bits() | (x.to_bits() & 0x8000_0000)) // copysign(yf, x)
}

/// `expm1(x) = eˣ − 1`, the numerically-stable form (the exact ELU/`SELU` negative tail, stable
/// losses). Kahan's correction `(u−1)·x/ln(u)` with `u = eˣ` cancels the catastrophic `eˣ − 1` loss
/// for small `x` (the ratio `(u−1)/ln(u) → 1` as `u → 1`); the guard returns `x` when `u` rounds to 1
/// (else the `0·∞` would be `NaN`). Reuses [`exp1`]/[`log1`], so the scalar twin, AVX2 [`expm1_8`], and
/// inlined MIR agree. Computed branchlessly (mirrors the AVX2 blend) for the tail-match.
#[inline]
fn expm1_1(x: f32) -> f32 {
    let u = exp1(x);
    let um1 = u - 1.0;
    let val = um1 * (x / log1(u));
    if u == 1.0 {
        x
    } else {
        val
    }
}

/// `log1p(x) = ln(1+x)`, the numerically-stable form (stable BCE/log-sum-exp). Kahan's correction
/// `ln(u)·x/(u−1)` with `u = 1+x` undoes the rounding of `1+x` for small `x`; the guard returns `x`
/// when `u` rounds to 1. Reuses [`log1`]; agrees bit-for-bit across the scalar twin, AVX2 [`log1p_8`],
/// and inlined MIR.
#[inline]
fn log1p_1(x: f32) -> f32 {
    let u = 1.0 + x;
    let d = u - 1.0;
    let val = log1(u) * (x / d);
    if u == 1.0 {
        x
    } else {
        val
    }
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
        VM_SELU => selu1(x),
        VM_TANHSHRINK => tanhshrink1(x),
        VM_HARDSIGMOID => hardsigmoid1(x),
        VM_HARDSWISH => hardswish1(x),
        VM_SIN => sincos1(x, false),
        VM_COS => sincos1(x, true),
        VM_ERF => erf1(x),
        VM_EXP2 => exp2_1(x),
        VM_LOG2 => log2_1(x),
        VM_SINH => sinh1(x),
        VM_COSH => cosh1(x),
        VM_ASINH => asinh1(x),
        VM_ACOSH => acosh1(x),
        VM_ATANH => atanh1(x),
        VM_ATAN => atan1(x),
        VM_EXPM1 => expm1_1(x),
        VM_LOG1P => log1p_1(x),
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
        VM_SELU => selu8,
        VM_TANHSHRINK => tanhshrink8,
        VM_HARDSIGMOID => hardsigmoid8,
        VM_HARDSWISH => hardswish8,
        VM_SIN => sin8,
        VM_COS => cos8,
        VM_ERF => erf8,
        VM_EXP2 => exp2_8,
        VM_LOG2 => log2_8,
        VM_SINH => sinh8,
        VM_COSH => cosh8,
        VM_ASINH => asinh8,
        VM_ACOSH => acosh8,
        VM_ATANH => atanh8,
        VM_ATAN => atan8,
        VM_EXPM1 => expm1_8,
        VM_LOG1P => log1p_8,
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn selu8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // λ·(x>0 ? x : α·(eˣ−1)) — mirrors selu1 op-for-op, incl. the multiply *order* `λ·(α·em1)` (float
    // mul isn't associative, and the scalar tail must match these lanes bit-for-bit).
    let lambda = _mm256_set1_ps(SELU_LAMBDA);
    let posval = _mm256_mul_ps(lambda, x);
    let em1 = _mm256_sub_ps(exp8(x), _mm256_set1_ps(1.0));
    let negval = _mm256_mul_ps(lambda, _mm256_mul_ps(_mm256_set1_ps(SELU_ALPHA), em1));
    let pos = _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_setzero_ps());
    _mm256_blendv_ps(negval, posval, pos)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanhshrink8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // x − tanh(x) — mirrors tanhshrink1.
    _mm256_sub_ps(x, tanh8(x))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hardsigmoid8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // clamp(x+3, 0, 6)·(1/6) — max_ps/min_ps match hardsigmoid1's `v>0`/`v<6` branches bit-for-bit.
    let y = _mm256_add_ps(x, _mm256_set1_ps(3.0));
    let clamped = _mm256_min_ps(_mm256_max_ps(y, _mm256_setzero_ps()), _mm256_set1_ps(6.0));
    _mm256_mul_ps(clamped, _mm256_set1_ps(INV6))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hardswish8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // x · hardsigmoid(x) — mirrors hardswish1.
    _mm256_mul_ps(x, hardsigmoid8(x))
}

/// 8-lane `sin`/`cos`, mirroring [`sincos1`] op-for-op (so the scalar tail agrees with the lanes and a
/// dispatched trig loop matches the inlined `emit_trig_f32`). The quadrant blend uses float-eq masks +
/// `blendv`, matching the scalar `if quad == k` chain.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sincos8(x: std::arch::x86_64::__m256, is_cos: bool) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let magic = _mm256_set1_ps(EXP_MAGIC);
    let tt = _mm256_fmadd_ps(x, _mm256_set1_ps(TWO_OVER_PI), magic);
    let qf = _mm256_sub_ps(tt, magic);
    let r = _mm256_fmadd_ps(qf, _mm256_set1_ps(-PIO2_1), x);
    let r = _mm256_fmadd_ps(qf, _mm256_set1_ps(-PIO2_2), r);
    let r = _mm256_fmadd_ps(qf, _mm256_set1_ps(-PIO2_3), r);
    let z = _mm256_mul_ps(r, r);
    // sin_p(r) = r + (poly·z)·r
    let mut s = _mm256_set1_ps(SIN_P[0]);
    s = _mm256_fmadd_ps(s, z, _mm256_set1_ps(SIN_P[1]));
    s = _mm256_fmadd_ps(s, z, _mm256_set1_ps(SIN_P[2]));
    let sz = _mm256_mul_ps(s, z);
    let sin_p = _mm256_fmadd_ps(sz, r, r);
    // cos_p(r) = (1 − 0.5z) + poly·z²
    let mut c = _mm256_set1_ps(COS_P[0]);
    c = _mm256_fmadd_ps(c, z, _mm256_set1_ps(COS_P[1]));
    c = _mm256_fmadd_ps(c, z, _mm256_set1_ps(COS_P[2]));
    let z2 = _mm256_mul_ps(z, z);
    let cz2 = _mm256_mul_ps(c, z2);
    let hz = _mm256_fmadd_ps(_mm256_set1_ps(-0.5), z, _mm256_set1_ps(1.0));
    let cos_p = _mm256_add_ps(hz, cz2);
    // quad = (int)qf & 3, back to float for the float-eq masks (mirrors the scalar truncating cast).
    let quad_i = _mm256_and_si256(_mm256_cvttps_epi32(qf), _mm256_set1_epi32(3));
    let quad = _mm256_cvtepi32_ps(quad_i);
    let neg1 = _mm256_set1_ps(-1.0);
    let neg_sin = _mm256_mul_ps(sin_p, neg1);
    let neg_cos = _mm256_mul_ps(cos_p, neg1);
    let (a0, a1, a2, a3) = if is_cos {
        (cos_p, neg_sin, neg_cos, sin_p)
    } else {
        (sin_p, cos_p, neg_sin, neg_cos)
    };
    // blendv(a, b, mask) = mask ? b : a, so this is the `if quad==k {hit} else {miss}` chain.
    let eq0 = _mm256_cmp_ps::<_CMP_EQ_OQ>(quad, _mm256_setzero_ps());
    let eq1 = _mm256_cmp_ps::<_CMP_EQ_OQ>(quad, _mm256_set1_ps(1.0));
    let eq2 = _mm256_cmp_ps::<_CMP_EQ_OQ>(quad, _mm256_set1_ps(2.0));
    let sel23 = _mm256_blendv_ps(a3, a2, eq2);
    let sel123 = _mm256_blendv_ps(sel23, a1, eq1);
    _mm256_blendv_ps(sel123, a0, eq0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sin8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    sincos8(x, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cos8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    sincos8(x, true)
}

/// 8-lane `erf`, mirroring [`erf1`] op-for-op (reuses [`exp8`] for `e^(−x²)`, so a dispatched `erf`
/// agrees with the composed exp). `|x|` is the compare+select form (not `andps`) to match the scalar
/// twin and `emit_erf_f32` bit-for-bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn erf8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let neg1 = _mm256_set1_ps(-1.0);
    let negx = _mm256_mul_ps(x, neg1);
    let gt = _mm256_cmp_ps::<_CMP_GT_OQ>(x, negx);
    let ax = _mm256_blendv_ps(negx, x, gt); // gt ? x : negx  == |x|
    let denom = _mm256_fmadd_ps(_mm256_set1_ps(ERF_P), ax, one);
    let t = _mm256_div_ps(one, denom);
    let mut h = _mm256_set1_ps(ERF_A[4]);
    h = _mm256_fmadd_ps(h, t, _mm256_set1_ps(ERF_A[3]));
    h = _mm256_fmadd_ps(h, t, _mm256_set1_ps(ERF_A[2]));
    h = _mm256_fmadd_ps(h, t, _mm256_set1_ps(ERF_A[1]));
    h = _mm256_fmadd_ps(h, t, _mm256_set1_ps(ERF_A[0]));
    let poly = _mm256_mul_ps(h, t);
    let ax2 = _mm256_mul_ps(ax, ax);
    let e = exp8(_mm256_mul_ps(ax2, neg1));
    let mag = _mm256_sub_ps(one, _mm256_mul_ps(poly, e));
    let neg_mag = _mm256_mul_ps(mag, neg1);
    let ge = _mm256_cmp_ps::<_CMP_GE_OQ>(x, _mm256_setzero_ps());
    _mm256_blendv_ps(neg_mag, mag, ge) // ge ? mag : -mag
}

/// 8-lane `exp2(x) = exp(x·ln2)` — mirrors [`exp2_1`] (pre-scale then [`exp8`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp2_8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    exp8(_mm256_mul_ps(x, _mm256_set1_ps(LN_2)))
}

/// 8-lane `log2(x) = log(x)·log2(e)` — mirrors [`log2_1`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log2_8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_mul_ps(log8(x), _mm256_set1_ps(LOG2_E))
}

/// 8-lane `sinh(x) = (e^x − e^{−x})·0.5` — mirrors [`sinh1`] (`-x` via `*-1` to match the scalar twin).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sinh8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let ex = exp8(x);
    let enx = exp8(_mm256_mul_ps(x, _mm256_set1_ps(-1.0)));
    _mm256_mul_ps(_mm256_sub_ps(ex, enx), _mm256_set1_ps(0.5))
}

/// 8-lane `cosh(x) = (e^x + e^{−x})·0.5` — mirrors [`cosh1`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cosh8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let ex = exp8(x);
    let enx = exp8(_mm256_mul_ps(x, _mm256_set1_ps(-1.0)));
    _mm256_mul_ps(_mm256_add_ps(ex, enx), _mm256_set1_ps(0.5))
}

/// 8-lane `asinh(x) = copysign(log(|x| + √(x²+1)), x)` — mirrors [`asinh1`] (bit-mask abs / sqrt /
/// bit-or sign restore, so the lanes equal the scalar twin exactly). `mul`+`add` (not `fmadd`) for
/// `|x|²+1` matches the scalar twin's separate ops.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn asinh8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x8000_0000_u32 as i32));
    let ax = _mm256_and_ps(x, absmask);
    let x2 = _mm256_mul_ps(ax, ax);
    let s = _mm256_sqrt_ps(_mm256_add_ps(x2, _mm256_set1_ps(1.0)));
    let t = log8(_mm256_add_ps(ax, s));
    _mm256_or_ps(t, _mm256_and_ps(x, signmask))
}

/// 8-lane `acosh(x) = log(x + √(x²−1))` (x ≥ 1) — mirrors [`acosh1`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn acosh8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x2 = _mm256_mul_ps(x, x);
    let s = _mm256_sqrt_ps(_mm256_sub_ps(x2, _mm256_set1_ps(1.0)));
    log8(_mm256_add_ps(x, s))
}

/// 8-lane `atanh(x) = ½·log((1+x)/(1−x))` (|x| < 1) — mirrors [`atanh1`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn atanh8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let num = _mm256_add_ps(one, x);
    let den = _mm256_sub_ps(one, x);
    _mm256_mul_ps(log8(_mm256_div_ps(num, den)), _mm256_set1_ps(0.5))
}

/// 8-lane `atan(x)` (Cephes) — mirrors [`atan1`] op-for-op: bit-mask `|x|`, the two `_CMP_GT_OQ`
/// region masks, both reduced candidates blended in, the degree-3 FMA poly, then the offset and a
/// bit-or sign restore. The `mid` mask includes `big`; blending `big` last overrides it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn atan8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x8000_0000_u32 as i32));
    let ax = _mm256_and_ps(x, absmask);
    let big = _mm256_cmp_ps::<_CMP_GT_OQ>(ax, _mm256_set1_ps(ATAN_TAN_3PI8));
    let mid = _mm256_cmp_ps::<_CMP_GT_OQ>(ax, _mm256_set1_ps(ATAN_TAN_PI8));
    let one = _mm256_set1_ps(1.0);
    let xr_mid = _mm256_div_ps(_mm256_sub_ps(ax, one), _mm256_add_ps(ax, one));
    let xr_big = _mm256_div_ps(_mm256_set1_ps(-1.0), ax);
    let mut xr = ax;
    xr = _mm256_blendv_ps(xr, xr_mid, mid);
    xr = _mm256_blendv_ps(xr, xr_big, big);
    let mut y = _mm256_setzero_ps();
    y = _mm256_blendv_ps(y, _mm256_set1_ps(ATAN_PIO4), mid);
    y = _mm256_blendv_ps(y, _mm256_set1_ps(ATAN_PIO2), big);
    let z = _mm256_mul_ps(xr, xr);
    let mut p = _mm256_set1_ps(ATAN_P[0]);
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(ATAN_P[1]));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(ATAN_P[2]));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(ATAN_P[3]));
    let pz = _mm256_mul_ps(p, z);
    let pzx = _mm256_mul_ps(pz, xr);
    let res = _mm256_add_ps(pzx, xr);
    let yf = _mm256_add_ps(y, res);
    _mm256_or_ps(yf, _mm256_and_ps(x, signmask))
}

/// 8-lane `expm1(x) = (u−1)·x/log(u)`, `u = eˣ`, guard `u==1 → x` — mirrors [`expm1_1`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn expm1_8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let u = exp8(x);
    let um1 = _mm256_sub_ps(u, one);
    let val = _mm256_mul_ps(um1, _mm256_div_ps(x, log8(u)));
    let is1 = _mm256_cmp_ps::<_CMP_EQ_OQ>(u, one);
    _mm256_blendv_ps(val, x, is1)
}

/// 8-lane `log1p(x) = log(u)·x/(u−1)`, `u = 1+x`, guard `u==1 → x` — mirrors [`log1p_1`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log1p_8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let u = _mm256_add_ps(one, x);
    let d = _mm256_sub_ps(u, one);
    let val = _mm256_mul_ps(log8(u), _mm256_div_ps(x, d));
    let is1 = _mm256_cmp_ps::<_CMP_EQ_OQ>(u, one);
    _mm256_blendv_ps(val, x, is1)
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
            (
                VM_SELU,
                |x| {
                    SELU_LAMBDA
                        * if x > 0.0 {
                            x
                        } else {
                            SELU_ALPHA * (x.exp() - 1.0)
                        }
                },
                2e-5,
            ),
            (VM_TANHSHRINK, |x| x - x.tanh(), 2e-5),
            (VM_HARDSIGMOID, |x| (x + 3.0).max(0.0).min(6.0) * INV6, 1e-6),
            (
                VM_HARDSWISH,
                |x| x * ((x + 3.0).max(0.0).min(6.0) * INV6),
                1e-6,
            ),
            // sin/cos vs Rust's libm over [-20.48, 20.47] — the 3-part π/2 reduction holds well past
            // RoPE's angle range. The 1e-4 absolute floor covers the near-zero crossings a pure
            // relative check would blow up.
            (VM_SIN, |x| x.sin(), 1e-4),
            (VM_COS, |x| x.cos(), 1e-4),
            // atan: bounded & ≈1-ULP over the whole real line (Cephes 3-region reduction).
            (VM_ATAN, |x| x.atan(), 5e-5),
            // exp2/sinh/cosh reuse exp, so ≈exp's accuracy; relative over the full range.
            (VM_EXP2, |x| x.exp2(), 5e-5),
            (VM_SINH, |x| x.sinh(), 5e-5),
            (VM_COSH, |x| x.cosh(), 5e-5),
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
        // log / log2 over positive inputs only.
        let xs: Vec<f32> = (1..4096).map(|i| i as f32 * 0.05).collect();
        let mut out = vec![0.0f32; xs.len()];
        for (op, libm) in [
            (VM_LOG, f32::ln as fn(f32) -> f32),
            (VM_LOG2, f32::log2 as fn(f32) -> f32),
        ] {
            unsafe {
                mercury_vmath_f32(xs.as_ptr(), out.as_mut_ptr(), xs.len() as i64, op);
            }
            for (i, &x) in xs.iter().enumerate() {
                let want = libm(x);
                assert!(
                    (out[i] - want).abs() <= 5e-5 + 5e-5 * want.abs(),
                    "op {op} x={x}: got {} want {want}",
                    out[i]
                );
            }
        }
    }

    /// `erf` has no `std` reference, so check known values (odd-symmetric, → ±1 at the tails) to a
    /// tolerance comfortably inside the A&S formula's ~1.5e-7 error — enough to catch a transcription
    /// slip in either the scalar twin or the AVX2 lanes (the tail-match test pins the two equal).
    #[test]
    fn vmath_erf_known_values() {
        // (x, erf(x)) reference values.
        let refs: &[(f32, f32)] = &[
            (0.0, 0.0),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (2.0, 0.995_322_3),
            (-1.0, -0.842_700_8),
            (-0.25, -0.276_326_4),
            (3.0, 0.999_977_9),
        ];
        let xs: Vec<f32> = refs.iter().map(|&(x, _)| x).collect();
        let mut out = vec![0.0f32; xs.len()];
        unsafe {
            mercury_vmath_f32(xs.as_ptr(), out.as_mut_ptr(), xs.len() as i64, VM_ERF);
        }
        for (i, &(x, want)) in refs.iter().enumerate() {
            assert!(
                (out[i] - want).abs() <= 3e-5,
                "erf({x}): got {} want {want}",
                out[i]
            );
        }
    }

    /// asinh/acosh/atanh vs `std`, each over its domain. The composed `log(x+√(x²±1))` /
    /// `½·log((1+x)/(1−x))` is ≈exp/log-grade. `acosh` loses precision as `x → 1⁺` (the `x²−1`
    /// cancellation), so it is checked from 1.2 up; `atanh` to |x| ≤ 0.9 (it diverges at ±1). The
    /// domain-restricted tail-match (scalar twin == AVX2 lanes, bit-for-bit) is folded in here since
    /// `acosh`/`atanh` can't ride the all-real `vmath_tail_matches_lanes` loop.
    #[test]
    fn vmath_inverse_hyperbolic() {
        let check = |op: i64, xs: &[f32], reference: fn(f32) -> f32, tol: f32| {
            let mut out = vec![0.0f32; xs.len()];
            unsafe { mercury_vmath_f32(xs.as_ptr(), out.as_mut_ptr(), xs.len() as i64, op) };
            for (i, &x) in xs.iter().enumerate() {
                let want = reference(x);
                assert!(
                    (out[i] - want).abs() <= tol + tol * want.abs(),
                    "op {op} x={x}: got {} want {want}",
                    out[i]
                );
                // scalar twin (the AVX2 tail) must equal the kernel lane bit-for-bit.
                assert_eq!(out[i].to_bits(), apply1(op, x).to_bits(), "op {op} tail x={x}");
            }
        };
        // asinh: all-real (the sign-stable form holds for large negative x too). 1001 ≠ 8k → tail.
        let xs: Vec<f32> = (0..1001).map(|i| (i as f32 - 500.0) * 0.05).collect();
        check(VM_ASINH, &xs, |x| x.asinh(), 5e-5);
        // acosh: x ≥ 1.2.
        let xs: Vec<f32> = (0..1001).map(|i| 1.2 + i as f32 * 0.05).collect();
        check(VM_ACOSH, &xs, |x| x.acosh(), 5e-5);
        // atanh: |x| ≤ 0.9.
        let xs: Vec<f32> = (0..1001).map(|i| (i as f32 - 500.0) * 0.0018).collect();
        check(VM_ATANH, &xs, |x| x.atanh(), 5e-5);
    }

    /// expm1/log1p vs `std`, including the **small-x relative** check the Kahan correction exists for:
    /// at `x = 1e-3 … 1e-6`, `expm1(x) ≈ x` and `log1p(x) ≈ x` to ≈1 ULP, where naive `eˣ−1` / `log(1+x)`
    /// lose ~1e-4 relative. The wide-range pass uses the combined abs+rel floor; the small-x pass is
    /// relative-only (so it would fail for the naive forms). Also pins scalar twin == AVX2 lanes.
    #[test]
    fn vmath_expm1_log1p() {
        let kernel = |op: i64, xs: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; xs.len()];
            unsafe { mercury_vmath_f32(xs.as_ptr(), out.as_mut_ptr(), xs.len() as i64, op) };
            out
        };
        // Wide range. expm1 all-real (to +10 before exp gets large); log1p needs x > −1.
        let xe: Vec<f32> = (0..2001).map(|i| (i as f32 - 1000.0) * 0.01).collect();
        for (i, &x) in xe.iter().enumerate() {
            let (got, want) = (kernel(VM_EXPM1, &xe)[i], x.exp_m1());
            assert!(
                (got - want).abs() <= 5e-5 + 5e-5 * want.abs(),
                "expm1({x}): got {got} want {want}"
            );
            assert_eq!(got.to_bits(), apply1(VM_EXPM1, x).to_bits(), "expm1 tail {x}");
        }
        let xl: Vec<f32> = (0..2001).map(|i| -0.9 + i as f32 * 0.01).collect();
        for (i, &x) in xl.iter().enumerate() {
            let (got, want) = (kernel(VM_LOG1P, &xl)[i], x.ln_1p());
            assert!(
                (got - want).abs() <= 5e-5 + 5e-5 * want.abs(),
                "log1p({x}): got {got} want {want}"
            );
            assert_eq!(got.to_bits(), apply1(VM_LOG1P, x).to_bits(), "log1p tail {x}");
        }
        // Small-x relative accuracy — the whole point of the stable forms.
        let small: Vec<f32> = vec![1e-3, 1e-4, 1e-5, 1e-6, -1e-3, -1e-4, -1e-5];
        let (em, lm) = (kernel(VM_EXPM1, &small), kernel(VM_LOG1P, &small));
        for (i, &x) in small.iter().enumerate() {
            let (we, wl) = (x.exp_m1(), x.ln_1p());
            assert!((em[i] - we).abs() <= 5e-5 * we.abs(), "expm1 rel {x}: {} vs {we}", em[i]);
            assert!((lm[i] - wl).abs() <= 5e-5 * wl.abs(), "log1p rel {x}: {} vs {wl}", lm[i]);
        }
    }

    /// The AVX2 lanes and the scalar tail/fallback must agree element-for-element, so a length that is
    /// not a multiple of 8 produces a consistent result regardless of where the tail starts.
    #[test]
    fn vmath_tail_matches_lanes() {
        let xs: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.013).collect();
        for op in [
            VM_EXP, VM_LOG, VM_TANH, VM_SIGMOID, VM_RELU, VM_SILU, VM_GELU, VM_SIN, VM_COS, VM_ERF,
            VM_EXP2, VM_LOG2, VM_SINH, VM_COSH, VM_ASINH, VM_ATAN, VM_EXPM1,
        ] {
            if op == VM_LOG || op == VM_LOG2 {
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
