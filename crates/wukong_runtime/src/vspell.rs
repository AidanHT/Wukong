//! **Natural spellings** of the elementwise activations, as `wukong_vmath_f32` op codes
//! [`VM_SPELL_FIRST`]`..=`[`VM_SPELL_LAST`] (P12.10).
//!
//! The compiler dispatches `for i { out[i] = gelu(x[i]) }` to the 256-bit `VM_GELU` kernel, but
//! real code writes the activation out: HuggingFace's `0.5*x*(1 + tanh(0.7978845608*(x +
//! 0.044715*x*x*x)))`, PyTorch's `x*0.5*(1 + erf(x*M_SQRT1_2))`, the textbook `x/(1 + exp(-x))`.
//! Those spellings are **different functions** from the intrinsics in the last bits: the language's
//! `tanh` is the <=1-ULP piecewise kernel while `gelu` keeps the exp-composed block inside, the HF
//! cubic `((C1*x)*x)*x` rounds differently from `C1*((x*x)*x)`, a division is not a multiplication
//! by a reciprocal, `min(max(x + 3, 0), 6)` sends a NaN to `0` where `hardsigmoid` keeps it. So
//! substituting the intrinsic's kernel would change the answer, and the loops ran through the
//! generic vectorizer at 128-bit (or scalar) instead.
//!
//! Each code here is the kernel of **one spelling**: its scalar twin is that spelling's expression,
//! operation for operation, in the order and with the FMA contractions the language lowers it to.
//! `wukong_mir_build::spell` holds the same expression as a tree and dispatches a loop only when the
//! loop's own expression lowers to exactly that tree (up to swapping the operands of a `*` or `+`
//! whose other operand is a constant, which is exact). The eight-lane body is the same expression
//! over `__m256`; both are generated from ONE definition by `spell!`, so the twin and the lanes
//! cannot drift: every `o::mul` is `*` in one and `_mm256_mul_ps` in the other, every `o::fma` is
//! `mul_add` / `vfmadd`, every transcendental is the shared `vmath` kernel (`tanh1` / `tanh8`,
//! `exp1` / `exp8`, ...), which the inlined MIR already mirrors op for op.
//!
//! **NaN.** An IEEE operation with one NaN operand returns that NaN quieted, and with two returns
//! the *first* source operand's on x86 -- the rule the interpreter and native code follow, but one
//! an optimizing compiler may break by commuting a multiply (the runtime is built at `-O3` for the
//! release gate). So the NaN each spelling produces for a NaN input is derived from its expression
//! and **pinned** in both bodies (`Quiet` -> the input quieted, `QuietNeg` -> its sign flipped too,
//! `Zero` -> `+0`, `Raw` -> the input itself), rather than left to whichever operand order the
//! compiler chose. For a non-NaN input every NaN an expression can produce is the one default NaN an
//! invalid operation makes (`inf * 0`, `inf - inf`), whatever the operand order.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__m256;

// --- op codes (mirrored as `VMATH_*` constants in `wukong_mir_build` and `VM_*` in the autodiff
// tape) ---------------------------------------------------------------------------------------------

/// `0.5*x*(1 + tanh(C0*(x + C1*(x*x*x))))` -- PyTorch's C++ GELU (`x_cube = x*x*x`), the language's
/// `tanh`. The cubic contracts as `fma(C1, x^3, x)`.
pub const VM_GELU_TANH: i64 = 36;
/// `0.5*x*(1 + tanh(C0*(x + C1*x*x*x)))` -- HuggingFace `NewGELUActivation` / GPT-2 with the cube
/// written left to right: `fma(round(round(C1*x)*x), x, x)`.
pub const VM_GELU_TANH_HF: i64 = 37;
/// llm.c `gelu_forward`: `let cube = C1*x*x*x; 0.5*x*(1 + tanh(C0*(x + cube)))` -- a product bound
/// to a name does not contract, so `x + cube` is a plain add.
pub const VM_GELU_TANH_NOFMA: i64 = 38;
/// HuggingFace `FastGELUActivation`: `0.5*x*(1 + tanh(x*C0*(1 + C1*x*x)))`.
pub const VM_GELU_TANH_FAST: i64 = 39;
/// JAX `gelu(approximate=True)`: `x * (0.5*(1 + tanh(C0*(x + C1*(x*x*x)))))` -- the CDF first.
pub const VM_GELU_TANH_CDF: i64 = 40;
/// HuggingFace `QuickGELUActivation` (CLIP): `x * sigmoid(1.702*x)`.
pub const VM_QUICK_GELU: i64 = 41;
/// BERT / HuggingFace `_gelu_python`: `0.5*x*(1 + erf(x / sqrt(2)))`.
pub const VM_GELU_ERF_DIV: i64 = 42;
/// PyTorch's C++ GELU: `x*0.5*(1 + erf(x*M_SQRT1_2))`.
pub const VM_GELU_ERF_MUL: i64 = 43;
/// JAX `gelu(approximate=False)`: `x*(erf(x/sqrt(2)) + 1)/2`.
pub const VM_GELU_ERF_CDF: i64 = 44;
/// The textbook SiLU: `x / (1 + exp(-x))`.
pub const VM_SILU_DIV: i64 = 45;
/// `sigmoid(x) * x` -- `x * sigmoid(x)` (which IS `VM_SILU`) with the operands swapped, which moves
/// the NaN a NaN input produces: the sigmoid's negated exponential comes first.
pub const VM_SILU_SX: i64 = 46;
/// `exp(x) / (1 + exp(x))`.
pub const VM_SIGMOID_RATIO: i64 = 47;
/// `log(1 + exp(x))` -- the naive softplus (`+inf` once `exp` saturates).
pub const VM_SOFTPLUS_NAIVE: i64 = 48;
/// `log1p(exp(x))`.
pub const VM_SOFTPLUS_LOG1P: i64 = 49;
/// `x * tanh(log(1 + exp(x)))` -- mish over the naive softplus and the language's `tanh`.
pub const VM_MISH_NAIVE: i64 = 50;
/// `x * tanh(softplus(x))` -- the `mish` definition with the language's `tanh` (`mish()` keeps the
/// exp-composed block inside).
pub const VM_MISH_SOFTPLUS: i64 = 51;
/// `x * tanh(log1p(exp(x)))`.
pub const VM_MISH_LOG1P: i64 = 52;
/// `min(max(x + 3, 0), 6) / 6` -- PyTorch's `relu6(x + 3) / 6`. A NaN input gives `+0` (both
/// ordered compares are false), exactly as the source spelling does.
pub const VM_HARDSIGMOID_MINMAX: i64 = 53;
/// `clamp(x + 3, 0, 6) / 6` -- `clamp` passes a NaN through.
pub const VM_HARDSIGMOID_CLAMP: i64 = 54;
/// `x * min(max(x + 3, 0), 6) / 6` -- PyTorch's `x * relu6(x + 3) / 6`.
pub const VM_HARDSWISH_MINMAX: i64 = 55;
/// `x * clamp(x + 3, 0, 6) / 6`.
pub const VM_HARDSWISH_CLAMP: i64 = 56;
/// `x > 0 ? x : expm1(x)` -- the ELU over the Kahan-stable `expm1` (`elu()` subtracts one from
/// `exp`).
pub const VM_ELU_EXPM1: i64 = 57;
/// `x < 0 ? 0.01*x : x` -- the leaky ReLU tested from the negative side. Equal to `VM_LEAKYRELU` on
/// every input but a signalling NaN, which the select returns untouched where the leaky kernel's
/// multiply quiets it.
pub const VM_LEAKY_LT: i64 = 58;
/// `x >= 0 ? x : 0.01*x` -- the leaky ReLU with the zero on the identity side. Equal to
/// `VM_LEAKYRELU` on EVERY input (at `+-0` it returns `x` where the leaky kernel returns `0.01*x`,
/// the same zero; a NaN fails both compares and is multiplied in both), so its forward could have
/// been that kernel. It is its own code for the backward: the derivative at `x = +-0` is the
/// branch the select takes (`1`), not the leaky kernel's slope, and the code is the only thing the
/// autodiff tape can tell the two spellings apart by.
pub const VM_LEAKY_GE: i64 = 59;
/// `max(x, 0.01*x)`, i.e. `x > 0.01*x ? x : 0.01*x`. Equal to `VM_LEAKYRELU` on EVERY input (for a
/// finite `x > 0` the product rounds below `x`; for `x < 0` and `-inf` it is `>= x`; at `+-0` and
/// `+inf` the compare is false and the product is `x`'s own zero and `+inf`; a NaN is multiplied in
/// both), and like [`VM_LEAKY_GE`] its own code for the backward: at `+inf` its select hands the
/// adjoint to the product (`0.01`) where the leaky kernel's `x > 0` takes the identity.
pub const VM_LEAKY_MAX: i64 = 60;

/// The first spelling code.
pub const VM_SPELL_FIRST: i64 = VM_GELU_TANH;
/// The last spelling code -- and the last `wukong_vmath_f32` code of all.
pub const VM_SPELL_LAST: i64 = VM_LEAKY_MAX;

// --- constants: the f32 values the source literals round to ---------------------------------------

/// `0.7978845608` (sqrt(2/pi)) -- every common spelling (`0.7978845608`, `0.7978845608028654`,
/// `0.79788456`) rounds to these bits; `vspell_constants_are_the_literals_they_stand_for` pins it.
const C0: f32 = f32::from_bits(0x3f4c_422a);
/// `0.044715`.
const C1: f32 = f32::from_bits(0x3d37_2713);
/// `1.702`, QuickGELU's slope.
const QUICK: f32 = f32::from_bits(0x3fd9_db23);
/// sqrt(2) rounded to f32 -- `sqrt(2.0)` and the literal `1.4142135623730951` alike.
const SQRT2: f32 = f32::from_bits(0x3fb5_04f3);
/// 1/sqrt(2) rounded to f32 -- `0.7071067811865476`, `0.70710678` and `1.0 / sqrt(2.0)` alike.
const RSQRT2: f32 = f32::from_bits(0x3f35_04f3);
/// `0.01`, the leaky ReLU slope (`vmath::LEAKY_ALPHA`).
const LEAKY: f32 = 0.01;

// --- the two operation sets one definition is expanded over ---------------------------------------

/// Scalar operations: the twin, the AVX2 tail and the no-AVX2 fallback.
mod s {
    #[inline(always)]
    pub fn k(c: f32) -> f32 {
        c
    }
    #[inline(always)]
    pub fn add(a: f32, b: f32) -> f32 {
        a + b
    }
    #[inline(always)]
    pub fn mul(a: f32, b: f32) -> f32 {
        a * b
    }
    #[inline(always)]
    pub fn div(a: f32, b: f32) -> f32 {
        a / b
    }
    #[inline(always)]
    pub fn fma(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
    /// `a > b ? t : e` -- the MIR `Cmp(Fogt)` + `Select` (false on a NaN).
    #[inline(always)]
    pub fn sel_gt(a: f32, b: f32, t: f32, e: f32) -> f32 {
        if a > b {
            t
        } else {
            e
        }
    }
    /// `a >= b ? t : e` -- the MIR `Cmp(Foge)` + `Select` (false on a NaN).
    #[inline(always)]
    pub fn sel_ge(a: f32, b: f32, t: f32, e: f32) -> f32 {
        if a >= b {
            t
        } else {
            e
        }
    }
    #[inline(always)]
    pub fn exp(a: f32) -> f32 {
        crate::vmath::exp1(a)
    }
    #[inline(always)]
    pub fn log(a: f32) -> f32 {
        crate::vmath::log1(a)
    }
    #[inline(always)]
    pub fn tanh(a: f32) -> f32 {
        crate::vmath::tanh1(a)
    }
    #[inline(always)]
    pub fn erf(a: f32) -> f32 {
        crate::vmath::erf1(a)
    }
    #[inline(always)]
    pub fn sigmoid(a: f32) -> f32 {
        crate::vmath::sigmoid1(a)
    }
    #[inline(always)]
    pub fn softplus(a: f32) -> f32 {
        crate::vmath::softplus1(a)
    }
    #[inline(always)]
    pub fn log1p(a: f32) -> f32 {
        crate::vmath::log1p_1(a)
    }
    #[inline(always)]
    pub fn expm1(a: f32) -> f32 {
        crate::vmath::expm1_1(a)
    }
    /// `-a` -- flips the sign bit, NaNs included (the MIR `neg`).
    #[inline(always)]
    pub fn neg(a: f32) -> f32 {
        -a
    }
}

/// Eight-lane operations: each is the AVX2 instruction for the scalar operation of the same name.
#[cfg(target_arch = "x86_64")]
mod v {
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn k(c: f32) -> __m256 {
        _mm256_set1_ps(c)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn add(a: __m256, b: __m256) -> __m256 {
        _mm256_add_ps(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn mul(a: __m256, b: __m256) -> __m256 {
        _mm256_mul_ps(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn div(a: __m256, b: __m256) -> __m256 {
        _mm256_div_ps(a, b)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fma(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fmadd_ps(a, b, c)
    }
    /// `a > b ? t : e` per lane (ordered: false on a NaN).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn sel_gt(a: __m256, b: __m256, t: __m256, e: __m256) -> __m256 {
        _mm256_blendv_ps(e, t, _mm256_cmp_ps::<_CMP_GT_OQ>(a, b))
    }
    /// `a >= b ? t : e` per lane (ordered: false on a NaN).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn sel_ge(a: __m256, b: __m256, t: __m256, e: __m256) -> __m256 {
        _mm256_blendv_ps(e, t, _mm256_cmp_ps::<_CMP_GE_OQ>(a, b))
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn exp(a: __m256) -> __m256 {
        crate::vmath::exp8(a)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn log(a: __m256) -> __m256 {
        crate::vmath::log8(a)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn tanh(a: __m256) -> __m256 {
        crate::vmath::tanh8(a)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn erf(a: __m256) -> __m256 {
        crate::vmath::erf8(a)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn sigmoid(a: __m256) -> __m256 {
        crate::vmath::sigmoid8(a)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn softplus(a: __m256) -> __m256 {
        crate::vmath::softplus8(a)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn log1p(a: __m256) -> __m256 {
        crate::vmath::log1p_8(a)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn expm1(a: __m256) -> __m256 {
        crate::vmath::expm1_8(a)
    }
    /// Sign-bit flip (`xor -0.0`), NaNs included -- the scalar unary `-`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn neg(a: __m256) -> __m256 {
        _mm256_xor_ps(a, _mm256_set1_ps(-0.0))
    }
}

/// The NaN a spelling returns for a NaN input `x` (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NanRule {
    /// `x` quieted, sign and payload kept.
    Quiet,
    /// `-x` quieted: the sign flipped, the payload kept.
    QuietNeg,
    /// `+0.0` -- the spelling's ordered compares drop the NaN.
    Zero,
    /// `x` itself, untouched -- a select hands the input through with no arithmetic on it.
    Raw,
}

#[inline(always)]
fn nan_scalar(rule: NanRule, x: f32) -> f32 {
    match rule {
        NanRule::Quiet => f32::from_bits(x.to_bits() | 0x0040_0000),
        NanRule::QuietNeg => f32::from_bits((x.to_bits() ^ 0x8000_0000) | 0x0040_0000),
        NanRule::Zero => 0.0,
        NanRule::Raw => x,
    }
}

/// Replace the NaN lanes of `r` (where `x` is NaN) by the rule's value.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn nan_lanes(rule: NanRule, x: __m256, r: __m256) -> __m256 {
    use std::arch::x86_64::*;
    let nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x);
    let quiet = _mm256_castsi256_ps(_mm256_set1_epi32(0x0040_0000));
    let pinned = match rule {
        NanRule::Quiet => _mm256_or_ps(x, quiet),
        NanRule::QuietNeg => _mm256_or_ps(_mm256_xor_ps(x, _mm256_set1_ps(-0.0)), quiet),
        NanRule::Zero => _mm256_setzero_ps(),
        NanRule::Raw => x,
    };
    _mm256_blendv_ps(r, pinned, nan)
}

/// One spelling from one definition: `$one` is the scalar twin, `$eight` the AVX2 body, both the
/// expression `$body` with `o` bound to [`s`] or [`v`], and both pin the NaN by `$nan`.
macro_rules! spell {
    ($(#[$m:meta])* $one:ident, $eight:ident, $nan:expr, |$x:ident| $body:expr) => {
        $(#[$m])*
        #[inline]
        pub(crate) fn $one($x: f32) -> f32 {
            if $x.is_nan() {
                return nan_scalar($nan, $x);
            }
            #[allow(unused_imports)]
            use self::s as o;
            $body
        }

        $(#[$m])*
        #[cfg(target_arch = "x86_64")]
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        pub(crate) unsafe fn $eight($x: __m256) -> __m256 {
            #[allow(unused_imports)]
            use self::v as o;
            // SAFETY: every `o::` operation is an avx2+fma intrinsic, and this function enables both.
            let r = unsafe { $body };
            // SAFETY: as above.
            unsafe { nan_lanes($nan, $x, r) }
        }
    };
}

spell!(
    /// [`VM_GELU_TANH`].
    gelu_tanh1, gelu_tanh8, NanRule::Quiet, |x| {
        let x3 = o::mul(o::mul(x, x), x);
        let inner = o::mul(o::k(C0), o::fma(o::k(C1), x3, x));
        o::mul(o::mul(o::k(0.5), x), o::add(o::k(1.0), o::tanh(inner)))
    }
);

spell!(
    /// [`VM_GELU_TANH_HF`].
    gelu_tanh_hf1, gelu_tanh_hf8, NanRule::Quiet, |x| {
        let c = o::mul(o::mul(o::k(C1), x), x);
        let inner = o::mul(o::k(C0), o::fma(c, x, x));
        o::mul(o::mul(o::k(0.5), x), o::add(o::k(1.0), o::tanh(inner)))
    }
);

spell!(
    /// [`VM_GELU_TANH_NOFMA`].
    gelu_tanh_nofma1, gelu_tanh_nofma8, NanRule::Quiet, |x| {
        let cube = o::mul(o::mul(o::mul(o::k(C1), x), x), x);
        let inner = o::mul(o::k(C0), o::add(x, cube));
        o::mul(o::mul(o::k(0.5), x), o::add(o::k(1.0), o::tanh(inner)))
    }
);

spell!(
    /// [`VM_GELU_TANH_FAST`].
    gelu_tanh_fast1, gelu_tanh_fast8, NanRule::Quiet, |x| {
        let poly = o::fma(o::mul(o::k(C1), x), x, o::k(1.0));
        let inner = o::mul(o::mul(o::k(C0), x), poly);
        o::mul(o::mul(o::k(0.5), x), o::add(o::k(1.0), o::tanh(inner)))
    }
);

spell!(
    /// [`VM_GELU_TANH_CDF`].
    gelu_tanh_cdf1, gelu_tanh_cdf8, NanRule::Quiet, |x| {
        let x3 = o::mul(o::mul(x, x), x);
        let inner = o::mul(o::k(C0), o::fma(o::k(C1), x3, x));
        o::mul(x, o::mul(o::k(0.5), o::add(o::k(1.0), o::tanh(inner))))
    }
);

spell!(
    /// [`VM_QUICK_GELU`].
    quick_gelu1, quick_gelu8, NanRule::Quiet, |x| {
        o::mul(x, o::sigmoid(o::mul(o::k(QUICK), x)))
    }
);

spell!(
    /// [`VM_GELU_ERF_DIV`].
    gelu_erf_div1, gelu_erf_div8, NanRule::Quiet, |x| {
        let e = o::erf(o::div(x, o::k(SQRT2)));
        o::mul(o::mul(o::k(0.5), x), o::add(o::k(1.0), e))
    }
);

spell!(
    /// [`VM_GELU_ERF_MUL`].
    gelu_erf_mul1, gelu_erf_mul8, NanRule::Quiet, |x| {
        let e = o::erf(o::mul(o::k(RSQRT2), x));
        o::mul(o::mul(o::k(0.5), x), o::add(o::k(1.0), e))
    }
);

spell!(
    /// [`VM_GELU_ERF_CDF`].
    gelu_erf_cdf1, gelu_erf_cdf8, NanRule::Quiet, |x| {
        let e = o::erf(o::div(x, o::k(SQRT2)));
        o::div(o::mul(x, o::add(o::k(1.0), e)), o::k(2.0))
    }
);

spell!(
    /// [`VM_SILU_DIV`].
    silu_div1, silu_div8, NanRule::Quiet, |x| {
        o::div(x, o::add(o::k(1.0), o::exp(o::neg(x))))
    }
);

spell!(
    /// [`VM_SILU_SX`].
    silu_sx1, silu_sx8, NanRule::QuietNeg, |x| {
        o::mul(o::sigmoid(x), x)
    }
);

spell!(
    /// [`VM_SIGMOID_RATIO`].
    sigmoid_ratio1, sigmoid_ratio8, NanRule::Quiet, |x| {
        o::div(o::exp(x), o::add(o::k(1.0), o::exp(x)))
    }
);

spell!(
    /// [`VM_SOFTPLUS_NAIVE`].
    softplus_naive1, softplus_naive8, NanRule::Quiet, |x| {
        o::log(o::add(o::k(1.0), o::exp(x)))
    }
);

spell!(
    /// [`VM_SOFTPLUS_LOG1P`].
    softplus_log1p1, softplus_log1p8, NanRule::Quiet, |x| {
        o::log1p(o::exp(x))
    }
);

spell!(
    /// [`VM_MISH_NAIVE`].
    mish_naive1, mish_naive8, NanRule::Quiet, |x| {
        o::mul(x, o::tanh(o::log(o::add(o::k(1.0), o::exp(x)))))
    }
);

spell!(
    /// [`VM_MISH_SOFTPLUS`].
    mish_softplus1, mish_softplus8, NanRule::Quiet, |x| {
        o::mul(x, o::tanh(o::softplus(x)))
    }
);

spell!(
    /// [`VM_MISH_LOG1P`].
    mish_log1p1, mish_log1p8, NanRule::Quiet, |x| {
        o::mul(x, o::tanh(o::log1p(o::exp(x))))
    }
);

spell!(
    /// [`VM_HARDSIGMOID_MINMAX`]: `max(y, 0)` is `y > 0 ? y : 0` and `min(m, 6)` is
    /// `m < 6 ? m : 6`, i.e. `6 > m ? m : 6`.
    hardsigmoid_minmax1, hardsigmoid_minmax8, NanRule::Zero, |x| {
        let y = o::add(o::k(3.0), x);
        let m = o::sel_gt(y, o::k(0.0), y, o::k(0.0));
        let h = o::sel_gt(o::k(6.0), m, m, o::k(6.0));
        o::div(h, o::k(6.0))
    }
);

spell!(
    /// [`VM_HARDSIGMOID_CLAMP`]: `clamp(y, 0, 6)` is `y < 0 ? 0 : (y > 6 ? 6 : y)`.
    hardsigmoid_clamp1, hardsigmoid_clamp8, NanRule::Quiet, |x| {
        let y = o::add(o::k(3.0), x);
        let capped = o::sel_gt(y, o::k(6.0), o::k(6.0), y);
        let h = o::sel_gt(o::k(0.0), y, o::k(0.0), capped);
        o::div(h, o::k(6.0))
    }
);

spell!(
    /// [`VM_HARDSWISH_MINMAX`].
    hardswish_minmax1, hardswish_minmax8, NanRule::Quiet, |x| {
        let y = o::add(o::k(3.0), x);
        let m = o::sel_gt(y, o::k(0.0), y, o::k(0.0));
        let h = o::sel_gt(o::k(6.0), m, m, o::k(6.0));
        o::div(o::mul(x, h), o::k(6.0))
    }
);

spell!(
    /// [`VM_HARDSWISH_CLAMP`].
    hardswish_clamp1, hardswish_clamp8, NanRule::Quiet, |x| {
        let y = o::add(o::k(3.0), x);
        let capped = o::sel_gt(y, o::k(6.0), o::k(6.0), y);
        let h = o::sel_gt(o::k(0.0), y, o::k(0.0), capped);
        o::div(o::mul(x, h), o::k(6.0))
    }
);

spell!(
    /// [`VM_ELU_EXPM1`].
    elu_expm1_1, elu_expm1_8, NanRule::Quiet, |x| {
        o::sel_gt(x, o::k(0.0), x, o::expm1(x))
    }
);

spell!(
    /// [`VM_LEAKY_LT`]: `x < 0` is `0 > x`.
    leaky_lt1, leaky_lt8, NanRule::Raw, |x| {
        o::sel_gt(o::k(0.0), x, o::mul(o::k(LEAKY), x), x)
    }
);

spell!(
    /// [`VM_LEAKY_GE`].
    leaky_ge1, leaky_ge8, NanRule::Quiet, |x| {
        o::sel_ge(x, o::k(0.0), x, o::mul(o::k(LEAKY), x))
    }
);

spell!(
    /// [`VM_LEAKY_MAX`]: `max(a, b)` is `a > b ? a : b`.
    leaky_max1, leaky_max8, NanRule::Quiet, |x| {
        let scaled = o::mul(o::k(LEAKY), x);
        o::sel_gt(x, scaled, x, scaled)
    }
);

/// The scalar twin of spelling code `op`, or `None` when `op` is not one. The eight-lane bodies are
/// listed in `vmath`'s `with_vmath8_table`, next to every other code's;
/// `vspell_codes_are_total_in_both_tables` keeps the two lists equal.
#[inline]
pub(crate) fn apply_spell(op: i64, x: f32) -> Option<f32> {
    Some(match op {
        VM_GELU_TANH => gelu_tanh1(x),
        VM_GELU_TANH_HF => gelu_tanh_hf1(x),
        VM_GELU_TANH_NOFMA => gelu_tanh_nofma1(x),
        VM_GELU_TANH_FAST => gelu_tanh_fast1(x),
        VM_GELU_TANH_CDF => gelu_tanh_cdf1(x),
        VM_QUICK_GELU => quick_gelu1(x),
        VM_GELU_ERF_DIV => gelu_erf_div1(x),
        VM_GELU_ERF_MUL => gelu_erf_mul1(x),
        VM_GELU_ERF_CDF => gelu_erf_cdf1(x),
        VM_SILU_DIV => silu_div1(x),
        VM_SILU_SX => silu_sx1(x),
        VM_SIGMOID_RATIO => sigmoid_ratio1(x),
        VM_SOFTPLUS_NAIVE => softplus_naive1(x),
        VM_SOFTPLUS_LOG1P => softplus_log1p1(x),
        VM_MISH_NAIVE => mish_naive1(x),
        VM_MISH_SOFTPLUS => mish_softplus1(x),
        VM_MISH_LOG1P => mish_log1p1(x),
        VM_HARDSIGMOID_MINMAX => hardsigmoid_minmax1(x),
        VM_HARDSIGMOID_CLAMP => hardsigmoid_clamp1(x),
        VM_HARDSWISH_MINMAX => hardswish_minmax1(x),
        VM_HARDSWISH_CLAMP => hardswish_clamp1(x),
        VM_ELU_EXPM1 => elu_expm1_1(x),
        VM_LEAKY_LT => leaky_lt1(x),
        VM_LEAKY_GE => leaky_ge1(x),
        VM_LEAKY_MAX => leaky_max1(x),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmath::{wukong_vmath_f32, wukong_vmath_f32_parallel};

    fn codes() -> std::ops::RangeInclusive<i64> {
        VM_SPELL_FIRST..=VM_SPELL_LAST
    }

    /// The constants are the bits the source literals the recognizer accepts round to, through the
    /// compiler's own literal path (`f64` parse, then `as f32`).
    #[test]
    fn vspell_constants_are_the_literals_they_stand_for() {
        let lit = |s: &str| (s.parse::<f64>().unwrap() as f32).to_bits();
        for s in ["0.7978845608", "0.7978845608028654", "0.79788456"] {
            assert_eq!(lit(s), C0.to_bits(), "{s}");
        }
        assert_eq!(lit("0.044715"), C1.to_bits());
        assert_eq!(lit("1.702"), QUICK.to_bits());
        assert_eq!(lit("1.4142135623730951"), SQRT2.to_bits());
        assert_eq!(2.0f32.sqrt().to_bits(), SQRT2.to_bits());
        assert_eq!(lit("0.7071067811865476"), RSQRT2.to_bits());
        assert_eq!(lit("0.70710678"), RSQRT2.to_bits());
        assert_eq!((1.0f32 / 2.0f32.sqrt()).to_bits(), RSQRT2.to_bits());
        assert_eq!(lit("0.01"), LEAKY.to_bits());
    }

    /// Every code has its twin and its lanes, and nothing outside the range claims to be one -- in
    /// this module's table and in `vmath`'s eight-lane table alike (a code missing from the latter
    /// would silently run the scalar twin, which every value test below would still pass).
    #[test]
    fn vspell_codes_are_total_in_both_tables() {
        for op in codes() {
            assert!(apply_spell(op, 0.5).is_some(), "no scalar twin for {op}");
            #[cfg(target_arch = "x86_64")]
            assert!(
                crate::vmath::vmath8_for(op).is_some(),
                "no eight-lane body for {op} in vmath's table"
            );
        }
        for op in [-1, 0, VM_SPELL_FIRST - 1, VM_SPELL_LAST + 1, 9999] {
            assert!(
                apply_spell(op, 0.5).is_none(),
                "{op} claims to be a spelling"
            );
        }
    }

    /// The inputs every spelling is checked on: a dense finite sweep across the interesting range
    /// (with exact zeros), the tails where the transcendentals saturate, both infinities, both zeros,
    /// subnormals, and NaNs of both signs with distinct payloads (one signalling).
    fn probe_inputs() -> Vec<f32> {
        let mut xs: Vec<f32> = (0..2001).map(|i| (i as f32 - 1000.0) * 0.0123).collect();
        xs.extend([
            0.0,
            -0.0,
            -3.0,
            3.0,
            1e-45,
            -1e-45,
            1e-39,
            -1e-39,
            1e-20,
            -1e-20,
            40.0,
            -40.0,
            88.0,
            -88.0,
            89.0,
            -104.0,
            1e10,
            -1e10,
            3.0e38,
            -3.0e38,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_0000),
            f32::from_bits(0xffc0_0000),
            f32::from_bits(0x7fc1_2345),
            f32::from_bits(0xffc5_4321),
            f32::from_bits(0x7f80_0001),
        ]);
        xs
    }

    /// The eight-lane body equals the scalar twin bit for bit on every probe, in every lane position
    /// and in the tail, through the shipped (monomorphized) kernel, the serial and the parallel entry.
    #[test]
    fn vspell_lanes_equal_the_scalar_twin_bit_for_bit() {
        let xs = probe_inputs();
        for op in codes() {
            for shift in 0..8 {
                let src: Vec<f32> = xs[shift..].to_vec();
                let n = src.len();
                let mut out = vec![f32::from_bits(0x7f80_0123); n];
                let mut par = vec![f32::from_bits(0x7f80_0123); n];
                // SAFETY: both buffers hold `n` f32.
                unsafe {
                    wukong_vmath_f32(src.as_ptr(), out.as_mut_ptr(), n as i64, op);
                    wukong_vmath_f32_parallel(src.as_ptr(), par.as_mut_ptr(), n as i64, op);
                }
                for i in 0..n {
                    let want = apply_spell(op, src[i]).unwrap();
                    assert_eq!(
                        out[i].to_bits(),
                        want.to_bits(),
                        "op {op} shift {shift} i {i} x {:#x}: lanes {} vs twin {}",
                        src[i].to_bits(),
                        out[i],
                        want
                    );
                    assert_eq!(
                        out[i].to_bits(),
                        par[i].to_bits(),
                        "op {op}: serial vs parallel"
                    );
                }
            }
        }
    }

    /// The pinned NaN of every spelling, on both bodies.
    #[test]
    fn vspell_nan_inputs_follow_each_spellings_rule() {
        for op in codes() {
            for bits in [0x7fc0_0000u32, 0xffc0_1234, 0x7fc5_0001, 0xff80_0001] {
                let x = f32::from_bits(bits);
                let want = match op {
                    VM_SILU_SX => (bits ^ 0x8000_0000) | 0x0040_0000,
                    VM_HARDSIGMOID_MINMAX => 0,
                    VM_LEAKY_LT => bits,
                    _ => bits | 0x0040_0000,
                };
                assert_eq!(apply_spell(op, x).unwrap().to_bits(), want, "twin op {op}");
                let mut xs = [0.25f32; 19];
                xs[3] = x;
                xs[18] = x;
                let mut out = [0.0f32; 19];
                // SAFETY: both arrays hold 19 f32.
                unsafe { wukong_vmath_f32(xs.as_ptr(), out.as_mut_ptr(), 19, op) };
                assert_eq!(out[3].to_bits(), want, "lane op {op}");
                assert_eq!(out[18].to_bits(), want, "tail op {op}");
            }
        }
    }

    /// The `>=` leaky select and `max(x, 0.01*x)` are `VM_LEAKYRELU` on every input, signalling
    /// NaN included: their own codes exist for their derivatives (at zero, at `+inf`), not for
    /// different bits.
    #[test]
    fn the_ge_and_max_leaky_spellings_are_the_leaky_kernel_on_every_input() {
        for x in probe_inputs() {
            let want = crate::vmath::apply1(crate::vmath::VM_LEAKYRELU, x);
            for (op, got) in [(VM_LEAKY_GE, leaky_ge1(x)), (VM_LEAKY_MAX, leaky_max1(x))] {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "op {op} x {:#x}",
                    x.to_bits()
                );
            }
        }
    }

    /// The spellings that are the same function as an existing kernel on every non-NaN input are
    /// exactly that: the leaky select from the negative side is `VM_LEAKYRELU` (a quiet NaN too).
    #[test]
    fn the_negative_side_leaky_select_is_the_leaky_kernel_off_signalling_nans() {
        for x in probe_inputs() {
            if x.to_bits() == 0x7f80_0001 {
                continue;
            }
            let want = crate::vmath::apply1(crate::vmath::VM_LEAKYRELU, x);
            assert_eq!(
                leaky_lt1(x).to_bits(),
                want.to_bits(),
                "x {:#x}",
                x.to_bits()
            );
        }
    }

    /// Each twin is the function it claims to be: within a few ULP-scaled units of an independent
    /// `f64` evaluation of the textbook formula over the finite sweep (the spelling's own rounding
    /// order and the f32 transcendentals are the only differences).
    #[test]
    fn vspell_twins_track_an_f64_reference() {
        fn erf64(x: f64) -> f64 {
            crate::math64::wukong_rt_math_f64(x, crate::vmath::VM_ERF)
        }
        fn gelu_t(x: f64) -> f64 {
            let c0 = (2.0 / std::f64::consts::PI).sqrt();
            0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
        }
        fn gelu_e(x: f64) -> f64 {
            0.5 * x * (1.0 + erf64(x / std::f64::consts::SQRT_2))
        }
        fn softplus(z: f64) -> f64 {
            z.max(0.0) + (-z.abs()).exp().ln_1p()
        }
        fn hardsigmoid(x: f64) -> f64 {
            (x + 3.0).clamp(0.0, 6.0) / 6.0
        }
        type Ref = fn(f64) -> f64;
        let refs: [(i64, Ref); 25] = [
            (VM_GELU_TANH, gelu_t),
            (VM_GELU_TANH_HF, gelu_t),
            (VM_GELU_TANH_NOFMA, gelu_t),
            (VM_GELU_TANH_FAST, gelu_t),
            (VM_GELU_TANH_CDF, gelu_t),
            (VM_QUICK_GELU, |x| x / (1.0 + (-1.702 * x).exp())),
            (VM_GELU_ERF_DIV, gelu_e),
            (VM_GELU_ERF_MUL, gelu_e),
            (VM_GELU_ERF_CDF, gelu_e),
            (VM_SILU_DIV, |x| x / (1.0 + (-x).exp())),
            (VM_SILU_SX, |x| x / (1.0 + (-x).exp())),
            (VM_SIGMOID_RATIO, |x| 1.0 / (1.0 + (-x).exp())),
            (VM_SOFTPLUS_NAIVE, softplus),
            (VM_SOFTPLUS_LOG1P, softplus),
            (VM_MISH_NAIVE, |x| x * softplus(x).tanh()),
            (VM_MISH_SOFTPLUS, |x| x * softplus(x).tanh()),
            (VM_MISH_LOG1P, |x| x * softplus(x).tanh()),
            (VM_HARDSIGMOID_MINMAX, hardsigmoid),
            (VM_HARDSIGMOID_CLAMP, hardsigmoid),
            (VM_HARDSWISH_MINMAX, |x| x * hardsigmoid(x)),
            (VM_HARDSWISH_CLAMP, |x| x * hardsigmoid(x)),
            (VM_ELU_EXPM1, |x| if x > 0.0 { x } else { x.exp_m1() }),
            (VM_LEAKY_LT, |x| if x < 0.0 { 0.01 * x } else { x }),
            (VM_LEAKY_GE, |x| if x >= 0.0 { x } else { 0.01 * x }),
            (VM_LEAKY_MAX, |x| x.max(0.01 * x)),
        ];
        // Finite sweep only, inside [-12, 12]. The error is measured against the scale of the
        // terms the spelling combines, `max(|f(x)|, |x|, 1e-3)`: every spelling here is `O(|x|)`, and
        // the GELU forms compute `0.5x + 0.5x*tanh(..)`, whose cancellation on the negative side
        // (the result is ~1e-3 of `0.5x` at x = -3) leaves an error of a few ULP of `0.5|x|`, not of
        // the result.
        for (op, reference) in refs {
            let mut worst = 0.0f64;
            for i in 0..=4800 {
                let x = (i as f64 - 2400.0) * 0.005;
                let got = apply_spell(op, x as f32).unwrap() as f64;
                let want = reference((x as f32) as f64);
                let err = (got - want).abs() / want.abs().max(x.abs()).max(1e-3);
                worst = worst.max(err);
            }
            // The erf twin is Abramowitz-Stegun (~1.5e-7 absolute), the composed forms add a few
            // f32 roundings; 2e-5 of the scale is far above both and far below any wrong formula.
            assert!(worst < 2e-5, "op {op}: worst relative error {worst:e}");
        }
    }

    /// The relative error each spelling kernel publishes against the exact function it spells, over
    /// the inputs where that spelling is well conditioned -- the bound the `tests/run/spelling_act_*`
    /// fixtures declare (`corpus_lint.rs`'s `DECLARED_TOLERANCES`, token `5e-6`) for every lane whose
    /// value the scalar semantics do not pin to a bit pattern.
    ///
    /// The reference is the closed form in `f64` with the spelling's own `f32` constants (`C0`, `C1`,
    /// `QUICK`, `SQRT2`, `RSQRT2`, `LEAKY` as the literals round), so what is measured is the rounding
    /// of the f32 evaluation, not the choice of constant. The regions leave out exactly the inputs
    /// where the SPELLING (not the kernel, which reproduces it bit for bit) cancels: the tanh and
    /// erf GELUs below `-1.5` (`0.5x + 0.5x*tanh(..)` with `tanh -> -1`), every `log(1 + exp(..))`
    /// below `-4` -- the naive softplus and mish, and the language's own `softplus`, whose
    /// `log(1 + exp(-|x|))` is not a `log1p` -- (`1 + exp(x)` rounds `exp(x)` away), and
    /// `exp(x) - 1` in `(-0.25, 0)`. Every grid point is an exact f32 (a multiple of 2^-6), so the
    /// sweep is the fixtures' own domain. Measured worst cases: QuickGELU 3.8e-6 near `x = -39`
    /// (the rounding of `1.702*x` is an absolute error in the exponent), the naive softplus and mish
    /// 3.1e-6 near `x = -3.9`, the erf GELUs 1.1e-6, every other spelling at most 6e-7.
    #[test]
    fn vspell_relative_error_in_each_spellings_well_conditioned_range() {
        use crate::vmath::{VM_ELU, VM_LEAKYRELU, VM_SIGMOID, VM_SILU, VM_SOFTPLUS};
        let erf = |x: f64| crate::math64::wukong_rt_math_f64(x, crate::vmath::VM_ERF);
        let (c0, c1) = (C0 as f64, C1 as f64);
        let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
        let softplus = |z: f64| z.max(0.0) + (-z.abs()).exp().ln_1p();
        let hardsig = |x: f64| (x + 3.0).clamp(0.0, 6.0) / 6.0;
        let gelu_t = |x: f64| 0.5 * x * (1.0 + (c0 * (x + c1 * x * x * x)).tanh());
        // (code, lowest x, reference)
        type Ref<'a> = &'a dyn Fn(f64) -> f64;
        let rows: [(i64, f64, Ref); 32] = [
            (VM_GELU_TANH, -1.5, &gelu_t),
            (VM_GELU_TANH_HF, -1.5, &gelu_t),
            (VM_GELU_TANH_NOFMA, -1.5, &gelu_t),
            (VM_GELU_TANH_FAST, -1.5, &gelu_t),
            (VM_GELU_TANH_CDF, -1.5, &gelu_t),
            (VM_QUICK_GELU, -40.0, &|x| x * sig(QUICK as f64 * x)),
            (VM_GELU_ERF_DIV, -1.5, &|x| {
                0.5 * x * (1.0 + erf(x / SQRT2 as f64))
            }),
            (VM_GELU_ERF_MUL, -1.5, &|x| {
                0.5 * x * (1.0 + erf(x * RSQRT2 as f64))
            }),
            (VM_GELU_ERF_CDF, -1.5, &|x| {
                0.5 * x * (1.0 + erf(x / SQRT2 as f64))
            }),
            (VM_SILU, -40.0, &|x| x * sig(x)),
            (VM_SILU_DIV, -40.0, &|x| x * sig(x)),
            (VM_SILU_SX, -40.0, &|x| x * sig(x)),
            (VM_SIGMOID, -40.0, &sig),
            (VM_SIGMOID_RATIO, -40.0, &sig),
            (VM_SOFTPLUS, -4.0, &softplus),
            (VM_SOFTPLUS_NAIVE, -4.0, &softplus),
            (VM_SOFTPLUS_LOG1P, -40.0, &softplus),
            (VM_MISH_NAIVE, -4.0, &|x| x * softplus(x).tanh()),
            (VM_MISH_SOFTPLUS, -4.0, &|x| x * softplus(x).tanh()),
            (VM_MISH_LOG1P, -40.0, &|x| x * softplus(x).tanh()),
            (VM_HARDSIGMOID_MINMAX, -40.0, &hardsig),
            (VM_HARDSIGMOID_CLAMP, -40.0, &hardsig),
            (VM_HARDSWISH_MINMAX, -40.0, &|x| x * hardsig(x)),
            (VM_HARDSWISH_CLAMP, -40.0, &|x| x * hardsig(x)),
            (VM_ELU, -40.0, &|x| if x > 0.0 { x } else { x.exp_m1() }),
            (VM_ELU_EXPM1, -40.0, &|x| {
                if x > 0.0 {
                    x
                } else {
                    x.exp_m1()
                }
            }),
            (VM_LEAKYRELU, -40.0, &|x| {
                if x > 0.0 {
                    x
                } else {
                    LEAKY as f64 * x
                }
            }),
            (VM_LEAKY_LT, -40.0, &|x| {
                if x < 0.0 {
                    LEAKY as f64 * x
                } else {
                    x
                }
            }),
            (VM_LEAKY_GE, -40.0, &|x| {
                if x >= 0.0 {
                    x
                } else {
                    LEAKY as f64 * x
                }
            }),
            (VM_LEAKY_MAX, -40.0, &|x| x.max(LEAKY as f64 * x)),
            // The two compositions of an existing kernel the spellings also name.
            (VM_SIGMOID, -40.0, &sig),
            (VM_SILU, -40.0, &|x| x * sig(x)),
        ];
        const BOUND: f64 = 5e-6;
        for (op, lo, reference) in rows {
            let mut worst = (0.0f64, 0.0f64);
            let mut i = (lo * 64.0) as i64;
            while i <= 40 * 64 {
                let x = i as f64 / 64.0;
                i += 1;
                // `exp(x) - 1` cancels just below zero; `expm1` and every other spelling does not.
                if op == VM_ELU && x > -0.25 && x < 0.0 {
                    continue;
                }
                let got = crate::vmath::apply1(op, x as f32) as f64;
                let want = reference(x);
                let err = if want == 0.0 {
                    got.abs()
                } else {
                    ((got - want) / want).abs()
                };
                if err > worst.0 {
                    worst = (err, x);
                }
            }
            assert!(
                worst.0 <= BOUND,
                "op {op}: relative error {:e} at x = {} exceeds {BOUND:e}",
                worst.0,
                worst.1
            );
        }
    }
}
