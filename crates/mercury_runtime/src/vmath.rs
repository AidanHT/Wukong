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
pub const VM_EXP10: i64 = 28;
pub const VM_LOG10: i64 = 29;
pub const VM_SOFTSIGN: i64 = 30;
pub const VM_LOGSIGMOID: i64 = 31;
pub const VM_TAN: i64 = 32;
pub const VM_ASIN: i64 = 33;
pub const VM_ACOS: i64 = 34;
pub const VM_CBRT: i64 = 35;

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
const EXP_MAGIC: f32 = 12582912.0; // 1.5 * 2^23 — round-to-nearest-even via add-then-subtract
const EXP_C1: f32 = 0.693_359_4; // ln2, high part (log's k·ln2 reconstruction; exp splits it /8)
const EXP_C2: f32 = -2.1219444e-4; // ln2, low correction
const EXP_HI: f32 = 88.376_26;
const EXP_LO: f32 = -88.376_26;
const INV_2P23: f32 = 1.0 / 8_388_608.0; // 2^-23 (exact)

// --- exp: 8-bucket table reduction (mirrors mercury_mir_build's inlined `emit_exp_f32`) ------------
//
// `e^x` decomposes as `x = n·(ln2/8) + r` with `n = round(x·8/ln2)` (the same add-magic rounding
// as before, scaled ×8), so `e^x = 2^(n/8)·e^r = 2^e·T[j]·e^r` with `e = n >> 3` and `j = n & 7`:
// on two's complement the arithmetic shift floors, so `n = 8e + j` with `j ∈ [0, 7]` holds for
// negative `n` too. `T[j] = 2^(j/8)` rounded once to f32 lives whole in one ymm — the AVX2 lanes
// look it up with one in-register `vpermps` (no memory gather) and the scalar twin indexes the
// same array, so lane == scalar stays bit-for-bit. The reduction leaves `|r| ≤ ln2/16 ≈ 0.0433`,
// where a 3-FMA cubic `e^r ≈ 1 + r + P2·r² + P3·r³` is ≈1.3-ULP grade — against the old
// full-range degree-5 Estrin poly's 8 FP ops. `r` comes from a two-term Cody-Waite split of
// ln2/8: EXP_TBL_C1 = EXP_C1/8 (exact /8; 9 significant bits, 15 trailing mantissa zeros, so
// n·C1 is EXACT for |n| ≤ 2047 ≫ the clamp range's 1020) and EXP_TBL_C2 = EXP_C2/8 (also exact).
// `n` is read straight out of the magic-sum's mantissa bits — `m = bits(t) − EXP_TBL_MBIAS =
// n + 1016`, folding the +127 exponent bias (127·8 = 1016) into the one integer subtract:
// `j = m & 7` (1016 ≡ 0 mod 8) and `biased = max(m, 0) >> 3`, the max(·,0) flushing the 2^e
// underflow to +0.0 (exp(EXP_LO) = +0.0 bit-for-bit as before; the old kernel's n = −126 band
// produced denormals down to x ≈ −87.68, the table split flushes from x ≈ −87.38 — everything
// in that band is < 2^−126, out of every gate's domain). Overflow saturation at the EXP_HI
// clamp is bit-identical to the old kernel (0x7F3504A4 ≈ 2.406e38, no +∞), NaN still funnels
// through the same min-then-max clamp order, and exp(0) = 1.0 exactly (n = 0, r = 0, T[0] = 1,
// poly = 1). Measured max relative error vs f64 exp: 1.59e-7 over [−87, 88] (4M points) and
// 1.24e-7 near 0 — see `vmath_exp_dense_sweep`. Per vector this is 16 SIMD ops (11 on the FP
// ports) vs the old 18 (15 FP) — the VML-style table trade the throttled-clock A/B asked for.
const EXP_TBL_SCALE: f32 = (8.0f64 / std::f64::consts::LN_2) as f32; // 8/ln2 — n = round(x·8/ln2)
const EXP_TBL_C1: f32 = EXP_C1 / 8.0; // ln2/8 high — n·C1 exact for |n| ≤ 2047 (test-pinned)
const EXP_TBL_C2: f32 = EXP_C2 / 8.0; // ln2/8 low correction (a /8 of an f32 is exact)
const EXP_TBL_MBIAS: i32 = 0x4B40_0000 - 1016; // bits(EXP_MAGIC) − 127·8: m = bits(t)−MBIAS = n+1016
// T[j] = 2^(j/8) rounded once to f32 (T[0] pinned exactly 1.0 → exp(0) = 1.0 exactly). The
// `vmath_exp_tables_consistent` test below re-derives every entry bit-for-bit.
const EXP_TBL_T: [f32; 8] = [
    1.0,
    1.090_507_7,
    1.189_207_1,
    1.296_839_6,
    1.414_213_5,
    1.542_210_8,
    1.681_792_9,
    1.834_008_1,
];
// e^r ≈ ((P3·r + P2)·r + 1)·r + 1 on |r| ≤ ln2/16: P3 = 1/6; P2 = 1/2 + (√2−1)/12·(ln2/16)² —
// the Chebyshev-optimal shift of the r² coefficient, which absorbs the even r⁴/24 truncation
// term (plain 1/2 measured 2.77e-7 max relative; the shift more than halves it to 1.59e-7).
const EXP_TBL_P2: f32 = 0.500_064_8;
const EXP_TBL_P3: f32 = 1.0 / 6.0;

// --- log: 8-bucket table reduction (mirrors mercury_mir_build's inlined `emit_log_f32`) ------------
//
// `ln(x)` decomposes as `x = z·2^k` with `z ∈ [0.6953125, 1.390625)` by pure bit arithmetic:
// `tmp = bits(x) − LOG_OFF` splits at the bit pattern `LOG_OFF` instead of at an exponent boundary,
// so the reduced range *straddles 1.0* — `kbits = tmp & 0xFF80_0000` is `k·2^23` (signed, exact in
// f32 after an i32→f32 convert), and `z = bits(x) − kbits` reinterpreted as a float. The top 3
// mantissa bits of `tmp` (bits 22..20) index 8 uniform-in-bits buckets of `z`; per bucket a
// reciprocal-ish `R[j] ≈ 1/mid_j` and its exact log `L[j] = −ln(R[j])` give
//
//   ln(x) = k·ln2 + L[j] + ln(1 + s),   s = fma(z, R[j], −1),  |s| ≤ ~0.058,
//
// with `ln(1+s)` a short degree-5 Taylor tail (s − s²/2 + s³/3 − s⁴/4 + s⁵/5) — ~5 FMAs + 2 muls
// against the old full-range degree-8 minimax poly's 9. The AVX2 lanes do the two table lookups as
// in-register `vpermps` on the 8-entry `__m256` constants (no memory gather); the scalar twin
// indexes the same arrays, so lane == scalar stays bit-for-bit.
//
// Accuracy: bucket 4 spans z ∈ [0.9453125, 1.015625) — the bucket *containing 1.0* — and pins
// `R = 1.0, L = 0.0` exactly, so near x = 1 the whole thing collapses to `poly(s)` with `s = z − 1`
// exact (Sterbenz): no `L + k·ln2` cancellation where `ln(x)` is tiny and relative error would blow
// up. Measured max relative error vs f64 `ln` is 6.9e-7 (exhaustive over [0.25, 4), 33.5M values;
// 5.2e-7 over 1e6 log-spaced points spanning [1e-38, 1e38]) — see `vmath_log_dense_sweep`. The
// domain contract is unchanged (x > 0 normal; no guards): x = +0 (→ −127·ln2 ≈ −88.03), x = +∞ and
// NaN produce bit-identical values to the old kernel; denormals stay same-class garbage; only the
// x < 0 garbage values differ (out of every gate's domain).
const LOG_OFF: i32 = 0x3F32_0000; // z-range split point: z ∈ [0.6953125, 1.390625)
// R[j] = 1/mid_j of bucket j rounded once to f32 (bucket 4 pinned to exactly 1.0 — see above);
// L[j] = −ln(R[j]) computed in f64 *from the rounded-f32 R* and rounded once to f32. The
// `vmath_log_tables_consistent` test below re-derives both invariants bit-for-bit.
const LOG_TBL_R: [f32; 8] = [
    1.376_344_1,
    1.267_326_7,
    1.174_311_9,
    1.094_017_1,
    1.0,
    0.927_536_25,
    0.831_168_83,
    0.752_941_2,
];
const LOG_TBL_L: [f32; 8] = [
    -0.319_430_77,
    -0.236_909_73,
    -0.160_682_34,
    -0.089_856_38,
    0.0,
    0.075_223_4,
    0.184_922_34,
    0.283_768_15,
];
// ln(1+s) = s + s²·(C1 + C2·s + C3·s² + C4·s³): the degree-5 Taylor tail −1/2, 1/3, −1/4, 1/5.
// Degree 4 (dropping C4) measured 9.1e-6 max relative — outside the 4e-6 target — so degree 5 it is.
const LOG_C1: f32 = -0.5;
const LOG_C2: f32 = 0.333_333_34; // (1/3) rounded to f32
const LOG_C3: f32 = -0.25;
const LOG_C4: f32 = 0.2;

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
const LN_10: f32 = std::f64::consts::LN_10 as f32; // exp10(x) = exp(x·ln10)
const LOG10_E: f32 = std::f64::consts::LOG10_E as f32; // log10(x) = log(x)·log10(e)

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

/// `e^x` (≈1.3 ULP; measured max 1.59e-7 relative — see the table block above): the 8-bucket
/// table reduction `e^x = 2^e·T[j]·poly(r)`, `n = round(x·8/ln2)`, `j = n & 7`, `e = n >> 3`,
/// with the 3-FMA cubic tail and the /8 Cody-Waite ln2 split. Mirrors the AVX2 [`exp8`] lanes
/// op-for-op — the array index here IS the `vpermps` there (same 3 bits, same f32 constants),
/// `max(m, 0)` is its `pmaxsd`, and every `mul_add` is its `fmadd`, so lane == scalar stays
/// bit-identical.
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
    let t = x.mul_add(EXP_TBL_SCALE, EXP_MAGIC);
    let n = t - EXP_MAGIC;
    // t = EXP_MAGIC + n exactly (t sits in the [2^23, 2^24) binade where ULP = 1), so n drops out
    // of t's mantissa bits as an integer — with the +127·8 exponent bias pre-folded into the one
    // subtract: m = n + 1016. `wrapping_sub` = the AVX2 `vpsubd`.
    let m = (t.to_bits() as i32).wrapping_sub(EXP_TBL_MBIAS);
    // r = x − n·(ln2/8) in two fmas: n·C1 is exact (see the constants block), so the first fma
    // is one clean rounding and the low-part correction lands on a tiny residual.
    let r = n.mul_add(-EXP_TBL_C1, x);
    let r = n.mul_add(-EXP_TBL_C2, r);
    // Bucket lookup: the `& 7` is the scalar spelling of vpermps consuming bits 2..0 per lane
    // (two's complement keeps it right for the m < 0 fringe, where the result flushes to 0 anyway).
    let tj = EXP_TBL_T[(m & 7) as usize];
    // e^r ≈ ((P3·r + P2)·r + 1)·r + 1 — three serial FMAs (vs the old poly's 8 FP ops).
    let q = r.mul_add(EXP_TBL_P3, EXP_TBL_P2);
    let q = r.mul_add(q, 1.0);
    let p = r.mul_add(q, 1.0);
    // 2^e from the exponent field, bias already folded into m; max(m, 0) (= pmaxsd) flushes
    // e ≤ −127 to +0.0. After the max the value is nonnegative, so >> matches the vector psrad.
    let pow2 = f32::from_bits(((m.max(0) >> 3) as u32) << 23);
    // T[j]·p first — both sit near 1, so the near-exact 2^e scale multiplies last.
    (tj * p) * pow2
}

/// `ln(x)` for `x > 0` (≈2-ULP class; measured max 6.9e-7 relative — see the table block above):
/// the 8-bucket table reduction `ln(x) = k·ln2 + L[j] + poly(s)`, `s = fma(z, R[j], −1)`, with the
/// degree-5 Taylor tail and the same hi/lo `ln2` split `exp` uses. Mirrors the AVX2 [`log8`] lanes
/// op-for-op — the array index here IS the `vpermps` there (same 3 bits, same f32 constants), and
/// every `mul_add` is its `fmadd`, so lane == scalar stays bit-identical. `pub(crate)` so the fused
/// log-softmax kernel (`norm.rs`) can take the log of its row sum through the identical scalar log.
#[inline]
pub(crate) fn log1(x: f32) -> f32 {
    let bits = x.to_bits() as i32;
    // Split at the bit pattern LOG_OFF (≈ bits of 0.695): everything above the low 23 bits of `tmp`
    // is k·2^23; subtracting it back off `bits` renormalizes x to z = x·2^-k ∈ [0.6953125, 1.390625).
    // `wrapping_sub` = the AVX2 `vpsubd` (only garbage inputs — sign bit set — ever wrap).
    let tmp = bits.wrapping_sub(LOG_OFF);
    let kbits = tmp & 0xFF80_0000u32 as i32;
    let iz = bits.wrapping_sub(kbits);
    let z = f32::from_bits(iz as u32);
    // k as f32, shift-free: kbits = k·2^23 is exact in f32 (≤9 significant bits, sign included via
    // the signed convert), and the 2^-23 scale is a power of two — both steps exact.
    let e = (kbits as f32) * INV_2P23;
    // Bucket = bits 22..20 of tmp — the top 3 mantissa bits of (z's offset from LOG_OFF). The `& 7`
    // is the scalar spelling of vpermps consuming only bits 2..0 of each index lane.
    let j = (((tmp as u32) >> 20) & 7) as usize;
    let r = LOG_TBL_R[j];
    let l = LOG_TBL_L[j];
    // s = z·R[j] − 1 in ONE rounding; for bucket 4 (R = 1) this is z − 1, exact by Sterbenz.
    let s = z.mul_add(r, -1.0);
    // ln(1+s) − s = s²·(C1 + C2·s + C3·s² + C4·s³), Estrin: the two coefficient-pair FMAs run in
    // parallel, then one w-combine — a 2-FMA critical path.
    let w = s * s;
    let q0 = LOG_C2.mul_add(s, LOG_C1);
    let q1 = LOG_C4.mul_add(s, LOG_C3);
    let p = q1.mul_add(w, q0);
    let p = p * w;
    // Reconstruct smallest-first so the near-1 path stays pure poly: t = L + s²·P (both 0 at
    // bucket 4), + e·ln2_lo, + s, + e·ln2_hi — the same hi/lo ln2 split as `exp`, so large |k|
    // doesn't lose the low bits.
    let t = l + p;
    let y = e.mul_add(EXP_C2, t);
    let r2 = s + y;
    e.mul_add(EXP_C1, r2)
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

/// `softsign(x) = x / (1 + |x|)` — a bounded activation (range (−1, 1)) that saturates polynomially
/// rather than exponentially like `tanh`, so it's cheaper (no transcendental) and keeps larger
/// gradients in the tails. `|x|` clears the sign bit (== `f32::abs`), matching the AVX2 [`softsign8`]
/// bit-for-bit; the divide is the only real cost.
#[inline]
fn softsign1(x: f32) -> f32 {
    let ax = f32::from_bits(x.to_bits() & 0x7FFF_FFFF); // |x|
    x / (1.0 + ax)
}

/// `logsigmoid(x) = ln(σ(x)) = −softplus(−x)` — the numerically-stable log-sigmoid (PyTorch's
/// `F.logsigmoid`). The headline use is binary-cross-entropy-with-logits and contrastive/RL losses,
/// where `log(sigmoid(x))` computed naively overflows for large negative `x`; routing through the
/// stable [`softplus1`] (`max(t,0) + ln(1+e^{−|t|})`) is exact across the range. Reuses [`softplus1`],
/// so the AVX2 [`logsigmoid8`] and the tail agree. C/Rust have no `logsigmoidf` at all — it's two
/// scalar libm calls (`log(1/(1+exp(-x)))`), neither vectorizable.
#[inline]
fn logsigmoid1(x: f32) -> f32 {
    -softplus1(-x)
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
pub(crate) fn sincos1(x: f32, is_cos: bool) -> f32 {
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

/// `exp10(x) = 10^x = e^{x·ln10}` (decibels, log-scale features, base-10 schedules). Reuses [`exp1`],
/// so the AVX2 [`exp10_8`], the tail, and the composed scalar MIR all agree bit-for-bit.
#[inline]
fn exp10_1(x: f32) -> f32 {
    exp1(x * LN_10)
}

/// `log10(x) = ln(x)·log10(e)` (decibels, perplexity in base 10, log-scale metrics). Reuses [`log1`].
#[inline]
fn log10_1(x: f32) -> f32 {
    log1(x) * LOG10_E
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

/// `tan(x) = sin(x)/cos(x)` — reuses the shared sin/cos (one range reduction each), so the AVX2
/// [`tan8`], the tail, and the composed scalar MIR agree bit-for-bit. Accurate where `cos(x)` is not
/// near zero (away from the ±π/2 poles), exactly as libm's `tanf` is.
#[inline]
fn tan1(x: f32) -> f32 {
    sincos1(x, false) / sincos1(x, true)
}

/// `asin(x) = atan(x/√(1−x²))` over `[−1, 1]` — the angle whose sine is `x` (3D rotation, geometry,
/// graphics/vision ML). Reuses [`atan1`] and `√`; the endpoints fall out for free (`±1/√0 = ±∞`,
/// `atan(±∞) = ±π/2`) and `|x| > 1` yields `NaN` like libm. ≈atan's ≈1-ULP accuracy except very near
/// ±1 where the `1−x²` cancellation bites. Bit-identical to [`asin8`] (same ops, same order).
#[inline]
fn asin1(x: f32) -> f32 {
    atan1(x / (1.0 - x * x).sqrt())
}

/// `acos(x) = π/2 − asin(x)` over `[−1, 1]`. Reuses [`asin1`], so [`acos8`] and the tail agree.
#[inline]
fn acos1(x: f32) -> f32 {
    ATAN_PIO2 - asin1(x)
}

/// `cbrt(x) = copysign(e^{ln|x|/3}, x)` — the all-real cube root (LAB color, variance-stabilizing
/// transforms, physics). Evaluating on `|x|` and restoring the sign keeps `log` in its domain;
/// the `|x| == 0 → 0` guard avoids `log(0)`'s garbage feeding `exp`. Reuses [`exp1`]/[`log1`], with
/// the abs/sign as bit masks, so [`cbrt8`] and the tail agree bit-for-bit (the ±0 sign mirrors the
/// `asinh` precedent: copysign restores it).
#[inline]
fn cbrt1(x: f32) -> f32 {
    let ax = f32::from_bits(x.to_bits() & 0x7FFF_FFFF); // |x|
    let mag = exp1(log1(ax) * (1.0 / 3.0));
    let mag = if ax == 0.0 { 0.0 } else { mag };
    f32::from_bits(mag.to_bits() | (x.to_bits() & 0x8000_0000)) // copysign(mag, x)
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

/// Scalar dispatch for one element (used by the AVX2 tail and the no-AVX2 fallback). `pub(crate)` so
/// the broadcast-bias kernel in `bias.rs` folds the *identical* fused activation onto its `x + b[j]`
/// sum — one source of truth for the scalar activation, keeping bias-with-activation bit-for-bit
/// consistent with `mercury_vmath_f32` (an unrecognized/sentinel `op` returns `x`, i.e. identity).
#[inline]
pub(crate) fn apply1(op: i64, x: f32) -> f32 {
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
        VM_EXP10 => exp10_1(x),
        VM_LOG10 => log10_1(x),
        VM_SOFTSIGN => softsign1(x),
        VM_LOGSIGMOID => logsigmoid1(x),
        VM_TAN => tan1(x),
        VM_ASIN => asin1(x),
        VM_ACOS => acos1(x),
        VM_CBRT => cbrt1(x),
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

/// Select the 8-lane AVX2 kernel for op `op` (the `VM_*` codes), or `None` for an unrecognized op.
/// Shared by the f32 ([`vmath_avx2`]) and bf16-input ([`vmath_bf16_avx2`]) dispatchers so both apply
/// the *identical* activation — the only difference is how the 8 lanes are loaded (f32 vs widened
/// bf16), which keeps `mercury_vmath_bf16` bit-for-bit consistent with `mercury_vmath_f32`.
/// `pub(crate)` so the broadcast-bias AVX2 kernel in `bias.rs` applies the *identical* 8-lane
/// activation to its `x + b[j]` sum vector — the vector twin of [`apply1`], keeping the fused-activation
/// bias bit-for-bit consistent with `mercury_vmath_f32`. Returns `None` for an unrecognized/sentinel
/// `op` (the caller then leaves the sum unmodified — identity).
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn vmath8_for(
    op: i64,
) -> Option<unsafe fn(std::arch::x86_64::__m256) -> std::arch::x86_64::__m256> {
    Some(match op {
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
        VM_EXP10 => exp10_8,
        VM_LOG10 => log10_8,
        VM_SOFTSIGN => softsign8,
        VM_LOGSIGMOID => logsigmoid8,
        VM_TAN => tan8,
        VM_ASIN => asin8,
        VM_ACOS => acos8,
        VM_CBRT => cbrt8,
        _ => return None,
    })
}

/// Total streamed bytes (all live arrays) at/above which the output store goes non-temporal. Mirrors
/// the `velem`/`vhorner` streaming gate exactly (same ~10 MiB ≈ this machine's L3): a `vmovntps` skips
/// the **read-for-ownership** a cacheable store pays, and that only wins once the working set spills L3.
/// Below the threshold a normal store keeps the write-once output hot (a later re-read stays in cache),
/// above it streaming the array out saves the RFO traffic. The gate keys on the *total* bytes touched
/// (input + output), not the length, so the one-input activation (2 streams) crosses it at the same
/// working-set size velem's 2-stream map does — the >L3 activation-tensor sizes the exp/log dispatch hits.
const NT_MIN_BYTES: usize = 10 * 1024 * 1024;

/// Whether a kernel touching `streams` arrays of `n` f32 each should stream its stores — true once the
/// working set spills L3 (see [`NT_MIN_BYTES`]). `streams` counts every live array (the output plus
/// each input read), since they all compete for cache residency. Copied verbatim from `velem::use_nt`
/// so the two kernels make the identical crossover decision.
#[inline]
fn use_nt(n: usize, streams: usize) -> bool {
    streams.saturating_mul(n).saturating_mul(4) >= NT_MIN_BYTES
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vmath_avx2(x: *const f32, out: *mut f32, n: usize, op: i64) {
    use std::arch::x86_64::*;
    let Some(f) = vmath8_for(op) else { return };
    // Non-temporal store regime — two streams (`x` in, `out` out), so the pair spills L3 at the same
    // total-bytes threshold `velem`/`vhorner` use. Above it the streaming store skips the
    // read-for-ownership a cacheable store pays for a write-once tensor (the >L3 activation sizes the
    // exp/log benchmark hits); below it a plain store keeps the (maybe re-read) output hot — so `storeu`.
    // NT and cacheable stores write the *same bits*; only the cache path differs, so the result is
    // bit-for-bit identical either way.
    let nt = use_nt(n, 2);
    let mut i = 0usize;
    // For the streaming store, peel a scalar prologue until `out` is 32-byte aligned (`vmovntps` faults
    // on a misaligned address); after that each 8-lane step advances 32 bytes and stays aligned. The
    // scalar twin is bit-identical to the lanes, so peeling never perturbs the result — and `i < n`
    // guards a buffer too small to ever reach alignment (which can only happen below the NT threshold).
    if nt {
        while i < n && (out.add(i) as usize) & 31 != 0 {
            *out.add(i) = apply1(op, *x.add(i));
            i += 1;
        }
    }
    macro_rules! store {
        ($p:expr, $v:expr) => {
            if nt {
                _mm256_stream_ps($p, $v);
            } else {
                _mm256_storeu_ps($p, $v);
            }
        };
    }
    // Unroll ×4 (32 elements/step): run four *independent* load→poly→store chains at once. Even with
    // the polys in Estrin form (3-FMA-deep exp, 4-FMA-deep log — half Horner's serial depth), each
    // 8-lane function is still a mostly **dependent** SIMD chain, so a single-vector step is
    // latency-bound and idles the two FMA ports; four data-independent vectors overlap in the
    // out-of-order window and fill them (the same lever `velem`/`vhorner` pull), and Estrin's smaller
    // live set keeps the ×4 window inside the 16 ymm registers where four 8-deep Horner chains
    // spilled. Every lane still runs the identical `f`, and these are pure elementwise ops with no
    // cross-lane/cross-vector state, so the output is bit-for-bit identical to the one-at-a-time loop.
    while i + 32 <= n {
        let r0 = f(_mm256_loadu_ps(x.add(i)));
        let r1 = f(_mm256_loadu_ps(x.add(i + 8)));
        let r2 = f(_mm256_loadu_ps(x.add(i + 16)));
        let r3 = f(_mm256_loadu_ps(x.add(i + 24)));
        store!(out.add(i), r0);
        store!(out.add(i + 8), r1);
        store!(out.add(i + 16), r2);
        store!(out.add(i + 24), r3);
        i += 32;
    }
    // 8-wide remainder (the `n % 32` the ×4 body could not cover), same `f` and store policy.
    while i + 8 <= n {
        store!(out.add(i), f(_mm256_loadu_ps(x.add(i))));
        i += 8;
    }
    // Non-temporal stores are weakly ordered; fence before the buffer is read back by anyone.
    if nt {
        _mm_sfence();
    }
    // Scalar tail (same poly as the lanes, via the scalar twins) for the final < 8 elements.
    while i < n {
        *out.add(i) = apply1(op, *x.add(i));
        i += 1;
    }
}

// --- two-input transcendentals (pow/atan2/hypot) ---------------------------------------------------
// The 256-bit twin of the inlined two-arg poly: `out[i] = f(x[i], y[i])`. Op codes are a separate
// namespace (`VM2_*`) from the one-input `VM_*`. Each kernel mirrors `emit_pow`/`emit_atan2`/
// `emit_hypot` op-for-op (via the shared exp8/log8/atan8/sqrt that already match the inlined MIR), so
// a dispatched loop equals the composed/scalar form bit-for-bit.
pub const VM2_POW: i64 = 0;
pub const VM2_ATAN2: i64 = 1;
pub const VM2_HYPOT: i64 = 2;
// Activation *backward* (training gradient `dx = dy · act'(x)`): the two inputs are `(x, dy)` — the
// pre-activation input and the upstream gradient — and the kernel fuses the upstream multiply into the
// 256-bit derivative, so a whole `dx[i] = act_backward(x[i], dy[i])` loop is one pass. The derivative
// is a transcendental (silu' folds a sigmoid, gelu' a tanh), exactly the libm wall C/Rust can't
// vectorize — so this wins for the same reason the *forward* activation dispatch does.
pub const VM2_SILU_BWD: i64 = 3;
pub const VM2_GELU_BWD: i64 = 4;
// The foundational gate gradients (every LSTM/GRU/attention gate): σ'(x)=σ(1−σ), tanh'(x)=1−tanh².
pub const VM2_SIGMOID_BWD: i64 = 5;
pub const VM2_TANH_BWD: i64 = 6;
// ELU / softplus gradients (CNN / VAE-flow nets): elu'(x)=x>0?1:eˣ, softplus'(x)=σ(x).
pub const VM2_ELU_BWD: i64 = 7;
pub const VM2_SOFTPLUS_BWD: i64 = 8;
// Gated-FFN activation (SwiGLU / GeGLU — the Llama/PaLM/Gemma feed-forward gate): `out = act(a)·b`,
// the gate branch `a` (activated) weighting the linear branch `b`. The activation folds an `exp` C/Rust
// keep scalar, so the fused 256-bit gate wins like the forward dispatch; reuses `silu8`/`gelu8`, so it
// is bit-identical with the forward activation family. Inputs are positional `(a, b)`.
pub const VM2_SILU_GATE: i64 = 9;
pub const VM2_GELU_GATE: i64 = 10;
pub const VM2_SIGMOID_GATE: i64 = 11; // sigmoid_gate(a, b) = sigmoid(a) · b — the classic GLU gate

/// `pow(x, y) = e^{y·ln x}` (x > 0) — mirrors `emit_pow` via the shared exp/log.
#[inline]
fn pow2_1(x: f32, y: f32) -> f32 {
    exp1(y * log1(x))
}

/// `atan2(y, x)` — `atan(y/x)` + the quadrant fix (`x<0` adds `copysign(π, y)`; `x=0` falls out via
/// `atan(±∞)=±π/2`). Mirrors `emit_atan2`.
#[inline]
fn atan2_1(y: f32, x: f32) -> f32 {
    let a = atan1(y / x);
    let adj = if x < 0.0 {
        if y < 0.0 {
            -std::f32::consts::PI
        } else {
            std::f32::consts::PI
        }
    } else {
        0.0
    };
    a + adj
}

/// `hypot(a, b) = m·√((a/m)²+(b/m)²)`, `m = max(|a|,|b|)`, guarded `m==0 → 0`. Mirrors `emit_hypot`
/// (the `fma(rb,rb,ra²)` order matches; abs clears the sign bit).
#[inline]
fn hypot_1(a: f32, b: f32) -> f32 {
    let aa = f32::from_bits(a.to_bits() & 0x7FFF_FFFF);
    let bb = f32::from_bits(b.to_bits() & 0x7FFF_FFFF);
    let m = if aa > bb { aa } else { bb };
    let ra = a / m;
    let rb = b / m;
    let scaled = m * rb.mul_add(rb, ra * ra).sqrt();
    if m == 0.0 {
        0.0
    } else {
        scaled
    }
}

/// `silu_backward(x, dy) = dy · silu'(x)`, the SiLU/swish training gradient. With `s = σ(x)`,
/// `silu'(x) = s + x·s·(1−s) = fma(x·s, 1−s, s)` (a single-rounded FMA, matching the AVX2 [`silu_bwd8`]
/// and the inlined `emit_silu_backward` op-for-op). Reuses the shared [`sigmoid1`] (built on [`exp1`]),
/// so it agrees with the forward `silu`/`sigmoid` dispatch and is bit-identical across the scalar
/// twin / AVX2 lanes / inlined MIR. C/Rust compute `silu'` through a scalar `expf` (in the sigmoid),
/// which a loop with the call won't vectorize — so the 256-bit kernel wins like the forward pass.
#[inline]
fn silu_bwd_2(x: f32, dy: f32) -> f32 {
    let s = sigmoid1(x);
    let oms = 1.0 - s;
    let xs = x * s;
    let g = xs.mul_add(oms, s); // x·s·(1−s) + s = silu'(x)
    dy * g
}

/// `gelu_backward(x, dy) = dy · gelu'(x)` for the **tanh-approximation** GELU (paired with the forward
/// [`gelu1`], so the activation and its gradient use the *same* approximation). With
/// `I = c0·(x + c1·x³)`, `u = tanh(I)`:
/// `gelu'(x) = ½·(1 + u) + ½·x·(1 − u²)·c0·(1 + 3·c1·x²)`.
/// The inner `I`/`u` are computed exactly as [`gelu1`] (so `u` matches the forward dispatch), and the
/// derivative's extra terms use only the *same* constants `c0`/`c1` (the `3·c1·x²` written as
/// `fma(c1, 3·x², 1)` to avoid a new fragile constant) — all replicated op-for-op in the AVX2
/// [`gelu_bwd8`] and the inlined `emit_gelu_backward`, so every path agrees bit-for-bit. The `tanh`
/// (an `expf`) is the part C/Rust keep scalar, so the 256-bit kernel wins.
#[inline]
fn gelu_bwd_2(x: f32, dy: f32) -> f32 {
    let x2 = x * x;
    let x3 = x2 * x;
    let t = GELU_C1.mul_add(x3, x); // c1·x³ + x  (== gelu1's inner numerator)
    let inner = GELU_C0 * t;
    let u = tanh1(inner);
    let half_onep = 0.5 * (1.0 + u); // ½·(1 + u)
    let sech2 = 1.0 - u * u; // 1 − tanh²(I)
    let di = GELU_C1.mul_add(3.0 * x2, 1.0); // c1·(3x²) + 1 = 1 + 3·c1·x²
    let dinner = GELU_C0 * di; // I'(x)
    let hx = 0.5 * x;
    let term2 = hx * sech2 * dinner; // ½·x·(1−u²)·I'(x)
    dy * (half_onep + term2)
}

/// `sigmoid_backward(x, dy) = dy · σ'(x) = dy · σ(x)·(1 − σ(x))`, the logistic-gate training gradient.
/// Reuses the shared [`sigmoid1`] (an `expf`), so it agrees with the forward `sigmoid` dispatch and is
/// bit-identical across the scalar twin / AVX2 [`sigmoid_bwd8`] / inlined MIR.
#[inline]
fn sigmoid_bwd_2(x: f32, dy: f32) -> f32 {
    let s = sigmoid1(x);
    let oms = 1.0 - s;
    dy * (s * oms)
}

/// `tanh_backward(x, dy) = dy · tanh'(x) = dy · (1 − tanh²(x))`, the tanh-gate training gradient (RNN /
/// LSTM cell). Reuses the shared [`tanh1`] (an `expf`), so it agrees with the forward `tanh` dispatch
/// and is bit-identical across the scalar twin / AVX2 [`tanh_bwd8`] / inlined MIR.
#[inline]
fn tanh_bwd_2(x: f32, dy: f32) -> f32 {
    let t = tanh1(x);
    let sech2 = 1.0 - t * t;
    dy * sech2
}

/// `elu_backward(x, dy) = dy · elu'(x)`, `elu'(x) = x>0 ? 1 : eˣ` (α=1) — the ELU training gradient. The
/// branchless `if x>0 {1} else {eˣ}` mirrors the forward [`elu1`]'s blend and the AVX2 [`elu_bwd8`], so
/// the negative branch reuses the shared [`exp1`] and all paths agree bit-for-bit.
#[inline]
fn elu_bwd_2(x: f32, dy: f32) -> f32 {
    let e = exp1(x);
    let g = if x > 0.0 { 1.0 } else { e };
    dy * g
}

/// `softplus_backward(x, dy) = dy · softplus'(x) = dy · σ(x)` (since `d/dx ln(1+eˣ) = σ(x)`) — the
/// softplus training gradient (VAEs / normalizing flows / the Mish base). Reuses the shared [`sigmoid1`],
/// so it agrees with the forward `sigmoid`/`softplus` family; bit-identical across twin / AVX2 / inlined.
#[inline]
fn softplus_bwd_2(x: f32, dy: f32) -> f32 {
    dy * sigmoid1(x)
}

/// `silu_gate(a, b) = silu(a) · b` — the SwiGLU FFN gate. Reuses the shared [`silu1`] so it matches the
/// AVX2 [`silu_gate8`] and the forward activation dispatch bit-for-bit.
#[inline]
fn silu_gate_2(a: f32, b: f32) -> f32 {
    silu1(a) * b
}

/// `gelu_gate(a, b) = gelu(a) · b` — the GeGLU FFN gate (tanh-approx gelu, matching [`gelu1`]).
#[inline]
fn gelu_gate_2(a: f32, b: f32) -> f32 {
    gelu1(a) * b
}

/// `sigmoid_gate(a, b) = sigmoid(a) · b` — the classic GLU gate (Dauphin et al.). Reuses the shared
/// [`sigmoid1`] so it matches the AVX2 [`sigmoid_gate8`] and the activation family bit-for-bit.
#[inline]
fn sigmoid_gate_2(a: f32, b: f32) -> f32 {
    sigmoid1(a) * b
}

/// Scalar dispatch for one element pair (the AVX2 tail and the no-AVX2 fallback). The two inputs are
/// positional: `(base, exp)` for pow, `(y, x)` for atan2, `(a, b)` for hypot, `(x, dy)` for the
/// activation backwards.
#[inline]
fn apply2_1(op: i64, x: f32, y: f32) -> f32 {
    match op {
        VM2_POW => pow2_1(x, y),
        VM2_ATAN2 => atan2_1(x, y),
        VM2_HYPOT => hypot_1(x, y),
        VM2_SILU_BWD => silu_bwd_2(x, y),
        VM2_GELU_BWD => gelu_bwd_2(x, y),
        VM2_SIGMOID_BWD => sigmoid_bwd_2(x, y),
        VM2_TANH_BWD => tanh_bwd_2(x, y),
        VM2_ELU_BWD => elu_bwd_2(x, y),
        VM2_SOFTPLUS_BWD => softplus_bwd_2(x, y),
        VM2_SILU_GATE => silu_gate_2(x, y),
        VM2_GELU_GATE => gelu_gate_2(x, y),
        VM2_SIGMOID_GATE => sigmoid_gate_2(x, y),
        _ => x,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pow2_8(
    x: std::arch::x86_64::__m256,
    y: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    exp8(_mm256_mul_ps(y, log8(x)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn atan2_8(
    y: std::arch::x86_64::__m256,
    x: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let a = atan8(_mm256_div_ps(y, x));
    let zero = _mm256_setzero_ps();
    let yneg = _mm256_cmp_ps::<_CMP_LT_OQ>(y, zero);
    // yneg ? −π : π
    let pi_signed = _mm256_blendv_ps(
        _mm256_set1_ps(std::f32::consts::PI),
        _mm256_set1_ps(-std::f32::consts::PI),
        yneg,
    );
    let xneg = _mm256_cmp_ps::<_CMP_LT_OQ>(x, zero);
    let adj = _mm256_blendv_ps(zero, pi_signed, xneg); // xneg ? pi_signed : 0
    _mm256_add_ps(a, adj)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hypot8_2(
    a: std::arch::x86_64::__m256,
    b: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let aa = _mm256_and_ps(a, absmask);
    let bb = _mm256_and_ps(b, absmask);
    let m = _mm256_blendv_ps(bb, aa, _mm256_cmp_ps::<_CMP_GT_OQ>(aa, bb)); // max(|a|,|b|)
    let ra = _mm256_div_ps(a, m);
    let rb = _mm256_div_ps(b, m);
    let sum = _mm256_fmadd_ps(rb, rb, _mm256_mul_ps(ra, ra));
    let scaled = _mm256_mul_ps(m, _mm256_sqrt_ps(sum));
    let mzero = _mm256_cmp_ps::<_CMP_EQ_OQ>(m, _mm256_setzero_ps());
    _mm256_blendv_ps(scaled, _mm256_setzero_ps(), mzero) // mzero ? 0 : scaled
}

/// `silu(a) · b` over 8 lanes — mirrors [`silu_gate_2`] (reuses the forward [`silu8`]), so the lanes,
/// the scalar tail, and the forward activation dispatch all agree bit-for-bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_gate8(
    a: std::arch::x86_64::__m256,
    b: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    std::arch::x86_64::_mm256_mul_ps(silu8(a), b)
}

/// `gelu(a) · b` over 8 lanes — mirrors [`gelu_gate_2`] (reuses the forward [`gelu8`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_gate8(
    a: std::arch::x86_64::__m256,
    b: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    std::arch::x86_64::_mm256_mul_ps(gelu8(a), b)
}

/// `sigmoid(a) · b` over 8 lanes — mirrors [`sigmoid_gate_2`] (reuses the forward [`sigmoid8`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_gate8(
    a: std::arch::x86_64::__m256,
    b: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    std::arch::x86_64::_mm256_mul_ps(sigmoid8(a), b)
}

/// `dy · silu'(x)` over 8 lanes — mirrors [`silu_bwd_2`] op-for-op (`silu'(x) = fma(x·s, 1−s, s)`,
/// `s = sigmoid8(x)`), so the lanes, the scalar tail, and the inlined MIR all agree bit-for-bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_bwd8(
    x: std::arch::x86_64::__m256,
    dy: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let s = sigmoid8(x);
    let one = _mm256_set1_ps(1.0);
    let oms = _mm256_sub_ps(one, s); // 1 − s
    let xs = _mm256_mul_ps(x, s); // x·s
    let g = _mm256_fmadd_ps(xs, oms, s); // x·s·(1−s) + s = silu'(x)
    _mm256_mul_ps(dy, g)
}

/// `dy · gelu'(x)` (tanh approximation) over 8 lanes — mirrors [`gelu_bwd_2`] op-for-op (and reuses
/// the same inner `I`/`tanh8` as the forward [`gelu8`]), so dispatched == composed == scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_bwd8(
    x: std::arch::x86_64::__m256,
    dy: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let c0 = _mm256_set1_ps(GELU_C0);
    let c1 = _mm256_set1_ps(GELU_C1);
    let one = _mm256_set1_ps(1.0);
    let x2 = _mm256_mul_ps(x, x);
    let x3 = _mm256_mul_ps(x2, x);
    let t = _mm256_fmadd_ps(c1, x3, x); // c1·x³ + x  (== gelu8's inner numerator)
    let inner = _mm256_mul_ps(c0, t);
    let u = tanh8(inner);
    let half_onep = _mm256_mul_ps(_mm256_set1_ps(0.5), _mm256_add_ps(one, u)); // ½·(1 + u)
    let sech2 = _mm256_sub_ps(one, _mm256_mul_ps(u, u)); // 1 − u²
    let three_x2 = _mm256_mul_ps(_mm256_set1_ps(3.0), x2);
    let di = _mm256_fmadd_ps(c1, three_x2, one); // c1·(3x²) + 1
    let dinner = _mm256_mul_ps(c0, di); // I'(x)
    let hx = _mm256_mul_ps(_mm256_set1_ps(0.5), x);
    let term2 = _mm256_mul_ps(_mm256_mul_ps(hx, sech2), dinner); // ½·x·(1−u²)·I'(x)
    _mm256_mul_ps(dy, _mm256_add_ps(half_onep, term2))
}

/// `dy · σ'(x)` over 8 lanes — mirrors [`sigmoid_bwd_2`] op-for-op (`σ'(x) = s·(1−s)`, `s = sigmoid8`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_bwd8(
    x: std::arch::x86_64::__m256,
    dy: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let s = sigmoid8(x);
    let oms = _mm256_sub_ps(_mm256_set1_ps(1.0), s); // 1 − s
    _mm256_mul_ps(dy, _mm256_mul_ps(s, oms)) // dy · s·(1−s)
}

/// `dy · tanh'(x)` over 8 lanes — mirrors [`tanh_bwd_2`] op-for-op (`tanh'(x) = 1 − t²`, `t = tanh8`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_bwd8(
    x: std::arch::x86_64::__m256,
    dy: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let t = tanh8(x);
    let sech2 = _mm256_sub_ps(_mm256_set1_ps(1.0), _mm256_mul_ps(t, t)); // 1 − t²
    _mm256_mul_ps(dy, sech2)
}

/// `dy · elu'(x)` over 8 lanes — mirrors [`elu_bwd_2`] (`blendv(eˣ, 1, x>0)`, same compare as `elu8`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn elu_bwd8(
    x: std::arch::x86_64::__m256,
    dy: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let e = exp8(x);
    let pos = _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_setzero_ps());
    let g = _mm256_blendv_ps(e, _mm256_set1_ps(1.0), pos); // x>0 ? 1 : eˣ
    _mm256_mul_ps(dy, g)
}

/// `dy · σ(x)` over 8 lanes (softplus') — mirrors [`softplus_bwd_2`]; reuses `sigmoid8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softplus_bwd8(
    x: std::arch::x86_64::__m256,
    dy: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_mul_ps(dy, sigmoid8(x))
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn vmath2_8_for(
    op: i64,
) -> Option<
    unsafe fn(std::arch::x86_64::__m256, std::arch::x86_64::__m256) -> std::arch::x86_64::__m256,
> {
    Some(match op {
        VM2_POW => pow2_8,
        VM2_ATAN2 => atan2_8,
        VM2_HYPOT => hypot8_2,
        VM2_SILU_BWD => silu_bwd8,
        VM2_GELU_BWD => gelu_bwd8,
        VM2_SIGMOID_BWD => sigmoid_bwd8,
        VM2_TANH_BWD => tanh_bwd8,
        VM2_ELU_BWD => elu_bwd8,
        VM2_SOFTPLUS_BWD => softplus_bwd8,
        VM2_SILU_GATE => silu_gate8,
        VM2_GELU_GATE => gelu_gate8,
        VM2_SIGMOID_GATE => sigmoid_gate8,
        _ => return None,
    })
}

/// `out[i] = f(x[i], y[i])` for the two-input transcendentals (`VM2_*`). The 256-bit AVX2 twin of the
/// inlined two-arg poly an `out[i] = pow/atan2/hypot(x[i], y[i])` loop lowers to; mirrors the inlined
/// MIR op-for-op, so the interpreter marshalling through this kernel keeps native == interp exact.
///
/// # Safety
/// `x`, `y`, and `out` must each be valid for `n` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn mercury_vmath2_f32(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: i64,
    op: i64,
) {
    if n <= 0 {
        return;
    }
    let n = n as usize;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; buffers valid for n by the caller contract.
            unsafe { vmath2_avx2(x, y, out, n, op) };
            return;
        }
    }
    for i in 0..n {
        // SAFETY: i < n; buffers valid for n.
        unsafe { *out.add(i) = apply2_1(op, *x.add(i), *y.add(i)) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vmath2_avx2(x: *const f32, y: *const f32, out: *mut f32, n: usize, op: i64) {
    use std::arch::x86_64::*;
    let Some(f) = vmath2_8_for(op) else { return };
    // Same ×4-ILP + non-temporal-store treatment as `vmath_avx2` (this loop had the identical
    // single-vector, store-immediately structure). Three streams (`x`, `y` in, `out` out), so the
    // working set spills L3 — and wants `vmovntps` — at a *smaller* length than the one-input kernel.
    let nt = use_nt(n, 3);
    let mut i = 0usize;
    // Peel to 32-byte `out` alignment before any streaming store (`vmovntps` faults otherwise); the
    // scalar twin is bit-identical to the lanes, so the prologue never perturbs the result.
    if nt {
        while i < n && (out.add(i) as usize) & 31 != 0 {
            *out.add(i) = apply2_1(op, *x.add(i), *y.add(i));
            i += 1;
        }
    }
    macro_rules! store {
        ($p:expr, $v:expr) => {
            if nt {
                _mm256_stream_ps($p, $v);
            } else {
                _mm256_storeu_ps($p, $v);
            }
        };
    }
    // ×4 unroll (32 elements/step): four independent chains overlap the long dependent poly (pow folds
    // exp∘log, the activation backwards a sigmoid/tanh), filling the two FMA ports a single-vector step
    // leaves idle. Pure elementwise, one `f` per lane — so bit-for-bit identical to the old loop.
    while i + 32 <= n {
        let r0 = f(_mm256_loadu_ps(x.add(i)), _mm256_loadu_ps(y.add(i)));
        let r1 = f(_mm256_loadu_ps(x.add(i + 8)), _mm256_loadu_ps(y.add(i + 8)));
        let r2 = f(_mm256_loadu_ps(x.add(i + 16)), _mm256_loadu_ps(y.add(i + 16)));
        let r3 = f(_mm256_loadu_ps(x.add(i + 24)), _mm256_loadu_ps(y.add(i + 24)));
        store!(out.add(i), r0);
        store!(out.add(i + 8), r1);
        store!(out.add(i + 16), r2);
        store!(out.add(i + 24), r3);
        i += 32;
    }
    // 8-wide remainder (`n % 32`), same `f` and store policy.
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let yv = _mm256_loadu_ps(y.add(i));
        store!(out.add(i), f(xv, yv));
        i += 8;
    }
    // Weakly-ordered non-temporal stores: fence before anyone reads the buffer back.
    if nt {
        _mm_sfence();
    }
    while i < n {
        *out.add(i) = apply2_1(op, *x.add(i), *y.add(i));
        i += 1;
    }
}

/// `out[i] = f(widen(x[i]))` — the **bf16-input** twin of [`mercury_vmath_f32`]: the same 28-op
/// activation/transcendental dispatch (selected by `op`), but reading bf16 (2 bytes/elem, widened
/// *losslessly* to f32 with the shared [`crate::lowp::widen_bf16`]) and writing f32. Because the
/// widen is exact and `f` is the *same* kernel the f32 path uses, this equals
/// `mercury_vmath_f32(widen(x), …)` bit-for-bit — the interpreter marshals through this very kernel,
/// so interp == native on the differential gate.
///
/// The point is bandwidth: a *cheap* memory-bound op (relu, the elementwise the f32 kernel only ties
/// C/Rust on) moves 6 bytes/elem here (2 in + 4 out) vs the f32 kernel's 8, so it runs ~1.3× faster;
/// an *expensive* transcendental was already a big win (C/Rust can't vectorize a libm call), and bf16
/// input makes C's gap wider still since it also can't vectorize the bf16→f32 widen.
///
/// # Safety
/// `x` must be valid for `n` `u16` (bf16 bits); `out` for `n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn mercury_vmath_bf16(x: *const u16, out: *mut f32, n: i64, op: i64) {
    if n <= 0 {
        return;
    }
    let n = n as usize;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; buffers valid for n by the caller contract.
            unsafe { vmath_bf16_avx2(x, out, n, op) };
            return;
        }
    }
    for i in 0..n {
        // SAFETY: i < n; buffers valid for n.
        unsafe { *out.add(i) = apply1(op, crate::bf16_bits_to_f32(*x.add(i))) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vmath_bf16_avx2(x: *const u16, out: *mut f32, n: usize, op: i64) {
    use std::arch::x86_64::*;
    let Some(f) = vmath8_for(op) else { return };
    let mut i = 0;
    while i + 8 <= n {
        // lossless bf16→f32 widen (the same `<<16` the bf16 reductions use), then the shared kernel.
        let v = crate::lowp::widen_bf16(x.add(i));
        _mm256_storeu_ps(out.add(i), f(v));
        i += 8;
    }
    while i < n {
        *out.add(i) = apply1(op, crate::bf16_bits_to_f32(*x.add(i)));
        i += 1;
    }
}

/// `out[i] = f(widen(x[i]))` — the **IEEE-f16** twin of [`mercury_vmath_bf16`]: same 28-op dispatch,
/// reading f16 (widened with F16C `vcvtph2ps`, lossless) and writing f32. Equals
/// `mercury_vmath_f32(widen(x), …)` bit-for-bit (the widen is exact and `f` is the same kernel), so
/// the interpreter marshals through this kernel and interp == native.
///
/// # Safety
/// `x` must be valid for `n` `u16` (f16 bits); `out` for `n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn mercury_vmath_f16(x: *const u16, out: *mut f32, n: i64, op: i64) {
    if n <= 0 {
        return;
    }
    let n = n as usize;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: features detected; buffers valid for n by the caller contract.
            unsafe { vmath_f16_avx2(x, out, n, op) };
            return;
        }
    }
    for i in 0..n {
        // SAFETY: i < n; buffers valid for n.
        unsafe { *out.add(i) = apply1(op, crate::f16_bits_to_f32(*x.add(i))) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn vmath_f16_avx2(x: *const u16, out: *mut f32, n: usize, op: i64) {
    use std::arch::x86_64::*;
    let Some(f) = vmath8_for(op) else { return };
    let mut i = 0;
    while i + 8 <= n {
        // lossless f16→f32 widen (F16C `vcvtph2ps`), then the shared activation kernel.
        let v = crate::lowp::widen_f16(x.add(i));
        _mm256_storeu_ps(out.add(i), f(v));
        i += 8;
    }
    while i < n {
        *out.add(i) = apply1(op, crate::f16_bits_to_f32(*x.add(i)));
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
    // The 8-bucket table reduction, mirroring `exp1` op-for-op (see the constants block + `exp1`
    // for the algorithm and error analysis). vs the old full-range degree-5 Estrin body this
    // trades 8 poly FP ops for a 3-FMA cubic plus one in-register `vpermps` lookup (the T table
    // lives whole in one ymm — no memory gather) and swaps cvttps/paddd for a single vpsubd on
    // the magic-sum bits — 16 SIMD ops (11 FP-port) per vector vs the old 18 (15 FP-port), which
    // is what the ×4-unrolled dispatch loop turns into throughput when the clock is starved.
    let x = _mm256_min_ps(x, _mm256_set1_ps(EXP_HI));
    let x = _mm256_max_ps(x, _mm256_set1_ps(EXP_LO));
    let t = _mm256_fmadd_ps(x, _mm256_set1_ps(EXP_TBL_SCALE), _mm256_set1_ps(EXP_MAGIC));
    let n = _mm256_sub_ps(t, _mm256_set1_ps(EXP_MAGIC));
    // m = n + 1016 straight from t's mantissa bits (t = MAGIC + n exactly, ULP = 1 binade);
    // the +127·8 exponent bias is pre-folded into the subtrahend.
    let m = _mm256_sub_epi32(_mm256_castps_si256(t), _mm256_set1_epi32(EXP_TBL_MBIAS));
    let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-EXP_TBL_C1), x);
    let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-EXP_TBL_C2), r);
    // T[j]: vpermps reads only bits 2..0 of each index lane, which is the `& 7` in the scalar twin.
    let ttab = _mm256_loadu_ps(EXP_TBL_T.as_ptr());
    let tj = _mm256_permutevar8x32_ps(ttab, m);
    // e^r ≈ ((P3·r + P2)·r + 1)·r + 1 — three FMAs.
    let q = _mm256_fmadd_ps(r, _mm256_set1_ps(EXP_TBL_P3), _mm256_set1_ps(EXP_TBL_P2));
    let q = _mm256_fmadd_ps(r, q, _mm256_set1_ps(1.0));
    let p = _mm256_fmadd_ps(r, q, _mm256_set1_ps(1.0));
    // 2^e: max(m, 0) flushes e ≤ −127 to +0.0 (the underflow band), then shift the pre-biased
    // exponent into place. After the max the lanes are nonnegative, so psrad == the scalar >>.
    let mc = _mm256_max_epi32(m, _mm256_setzero_si256());
    let pow2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_srai_epi32::<3>(mc)));
    _mm256_mul_ps(_mm256_mul_ps(tj, p), pow2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn log8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // The 8-bucket table reduction, mirroring `log1` op-for-op (see the constants block + `log1` for
    // the algorithm and error analysis). vs the old full-range degree-8 poly this trades the 9-FMA
    // Estrin body + compare/blend range split for: 3 cheap integer ops, two 1-op in-register
    // `vpermps` table lookups (the tables live whole in one ymm each — no memory gather), and a
    // 5-FMA/2-mul residual — about half the FP-port pressure per vector, which is what the
    // ×4-unrolled dispatch loop turns into throughput.
    let bits = _mm256_castps_si256(x);
    let tmp = _mm256_sub_epi32(bits, _mm256_set1_epi32(LOG_OFF));
    let kbits = _mm256_and_si256(tmp, _mm256_set1_epi32(0xFF80_0000u32 as i32));
    let iz = _mm256_sub_epi32(bits, kbits);
    let z = _mm256_castsi256_ps(iz);
    // k as f32 — kbits = k·2^23 converts exactly (signed), and the 2^-23 scale is a power of two.
    let e = _mm256_mul_ps(_mm256_cvtepi32_ps(kbits), _mm256_set1_ps(INV_2P23));
    // Bucket index in bits 2..0 after the shift; vpermps reads only those 3 bits per lane, which is
    // the `& 7` in the scalar twin.
    let j = _mm256_srli_epi32::<20>(tmp);
    let rtab = _mm256_loadu_ps(LOG_TBL_R.as_ptr());
    let ltab = _mm256_loadu_ps(LOG_TBL_L.as_ptr());
    let r = _mm256_permutevar8x32_ps(rtab, j);
    let l = _mm256_permutevar8x32_ps(ltab, j);
    // s = z·R[j] − 1 in one rounding (exact z − 1 in the R = 1 bucket straddling x = 1).
    let s = _mm256_fmadd_ps(z, r, _mm256_set1_ps(-1.0));
    // Degree-5 Taylor tail in Estrin form: two parallel pair-FMAs, one w-combine (2-FMA path).
    let w = _mm256_mul_ps(s, s);
    let q0 = _mm256_fmadd_ps(_mm256_set1_ps(LOG_C2), s, _mm256_set1_ps(LOG_C1));
    let q1 = _mm256_fmadd_ps(_mm256_set1_ps(LOG_C4), s, _mm256_set1_ps(LOG_C3));
    let p = _mm256_fmadd_ps(q1, w, q0);
    let p = _mm256_mul_ps(p, w);
    // Smallest-first reconstruction with the hi/lo ln2 split (see `log1`).
    let t = _mm256_add_ps(l, p);
    let y = _mm256_fmadd_ps(e, _mm256_set1_ps(EXP_C2), t);
    let r2 = _mm256_add_ps(s, y);
    _mm256_fmadd_ps(e, _mm256_set1_ps(EXP_C1), r2)
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

/// `pub(crate)` so the GEMM fused epilogue (`gemm.rs`) applies the *identical* 8-lane SiLU as a
/// dispatched `silu` loop — `silu8` mirrors `silu1` bit-for-bit (the tail-match test pins it), so a
/// vectorized fused `silu(x·Wᵀ+b)` writeback equals the scalar epilogue the interpreter oracle uses.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn silu8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // x * sigmoid(x)
    _mm256_mul_ps(x, sigmoid8(x))
}

/// `pub(crate)` for the GEMM fused epilogue (see [`silu8`]); `gelu8` mirrors `gelu1` bit-for-bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn gelu8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
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
unsafe fn softsign8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // x / (1 + |x|) — `|x|` clears the sign bit (== softsign1's bit-clear), so lanes/tail agree.
    let absx = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF)));
    _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0), absx))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn logsigmoid8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // −softplus(−x) — mirrors logsigmoid1; reuses softplus8 so the stable form (and the tail) agree.
    let nx = _mm256_sub_ps(_mm256_setzero_ps(), x);
    _mm256_sub_ps(_mm256_setzero_ps(), softplus8(nx))
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
pub(crate) unsafe fn sincos8(x: std::arch::x86_64::__m256, is_cos: bool) -> std::arch::x86_64::__m256 {
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
pub(crate) unsafe fn sin8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    sincos8(x, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn cos8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp10_8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    exp8(_mm256_mul_ps(x, _mm256_set1_ps(LN_10)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log10_8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_mul_ps(log8(x), _mm256_set1_ps(LOG10_E))
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

/// 8-lane `tan(x) = sin(x)/cos(x)` — mirrors [`tan1`] (sin8/cos8 lanes already == sincos1).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tan8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_div_ps(sin8(x), cos8(x))
}

/// 8-lane `asin(x) = atan(x/√(1−x²))` — mirrors [`asin1`] op-for-op (`x²`, `1−x²`, `√`, divide, then
/// the shared [`atan8`]), so the lanes equal the scalar tail bit-for-bit on `[−1, 1]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn asin8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x2 = _mm256_mul_ps(x, x);
    let s = _mm256_sqrt_ps(_mm256_sub_ps(_mm256_set1_ps(1.0), x2));
    atan8(_mm256_div_ps(x, s))
}

/// 8-lane `acos(x) = π/2 − asin(x)` — mirrors [`acos1`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn acos8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_sub_ps(_mm256_set1_ps(ATAN_PIO2), asin8(x))
}

/// 8-lane `cbrt(x) = copysign(e^{ln|x|/3}, x)` with the `|x|==0 → 0` guard — mirrors [`cbrt1`]
/// (bit-mask abs, blend the zero case, bit-or the sign).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cbrt8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x8000_0000_u32 as i32));
    let ax = _mm256_and_ps(x, absmask);
    let mag = exp8(_mm256_mul_ps(log8(ax), _mm256_set1_ps(1.0 / 3.0)));
    let iszero = _mm256_cmp_ps::<_CMP_EQ_OQ>(ax, _mm256_setzero_ps());
    let mag = _mm256_blendv_ps(mag, _mm256_setzero_ps(), iszero); // iszero ? 0 : mag
    _mm256_or_ps(mag, _mm256_and_ps(x, signmask)) // copysign(mag, x)
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
            // exp2/exp10/sinh/cosh reuse exp, so ≈exp's accuracy; relative over the full range.
            (VM_EXP2, |x| x.exp2(), 5e-5),
            (VM_EXP10, |x| 10.0f32.powf(x), 5e-5),
            (VM_SINH, |x| x.sinh(), 5e-5),
            (VM_COSH, |x| x.cosh(), 5e-5),
            (VM_CBRT, |x| x.cbrt(), 5e-5), // all-real, incl. negatives and 0
            // softsign: exact (just abs + div). logsigmoid: the stable log-sigmoid; the naive
            // `ln(σ(x))` reference is overflow-safe over [-20.48, 20.47], and the mixed bound's
            // absolute floor covers the ≈x linear tail for large negative x.
            (VM_SOFTSIGN, |x| x / (1.0 + x.abs()), 1e-6),
            (VM_LOGSIGMOID, |x| (1.0 / (1.0 + (-x).exp())).ln(), 1e-4),
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
            (VM_LOG10, f32::log10 as fn(f32) -> f32),
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
                assert_eq!(
                    out[i].to_bits(),
                    apply1(op, x).to_bits(),
                    "op {op} tail x={x}"
                );
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

    /// tan/asin/acos vs `std`, each over a domain that keeps the composition accurate (tan away from
    /// the ±π/2 poles where cos → 0; asin/acos to |x| ≤ 0.95, the `1−x²` cancellation biting nearer
    /// ±1). Like the inverse-hyperbolics, the scalar twin == AVX2 lanes (bit-for-bit) tail-match is
    /// folded in — these can't ride the all-real `vmath_tail_matches_lanes` loop (poles / |x|>1 NaN).
    #[test]
    fn vmath_inverse_trig() {
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
                assert_eq!(
                    out[i].to_bits(),
                    apply1(op, x).to_bits(),
                    "op {op} tail x={x}"
                );
            }
        };
        // tan: |x| ≤ 1.4 (cos stays well off zero; tan(1.4) ≈ 5.8). 1001 ≠ 8k → exercises the tail.
        let xs: Vec<f32> = (0..1001).map(|i| (i as f32 - 500.0) * 0.0028).collect();
        check(VM_TAN, &xs, |x| x.tan(), 5e-5);
        // asin/acos: |x| ≤ 0.95.
        let xs: Vec<f32> = (0..1001).map(|i| (i as f32 - 500.0) * 0.0019).collect();
        check(VM_ASIN, &xs, |x| x.asin(), 1e-4);
        check(VM_ACOS, &xs, |x| x.acos(), 1e-4);
    }

    /// The two-input kernel (`mercury_vmath2_f32`): pow/atan2/hypot vs `std`, plus the scalar twin ==
    /// AVX2 lanes (bit-for-bit) tail match. `pow` over x∈(0,4], y∈[−2,2]; atan2/hypot over a paired
    /// sweep avoiding the (0,0) origin. 1003 pairs (≠ 8k) exercises the < 8 tail.
    #[test]
    fn vmath2_matches_libm() {
        let n = 1003;
        let check = |op: i64, xs: &[f32], ys: &[f32], reference: fn(f32, f32) -> f32, tol: f32| {
            let mut out = vec![0.0f32; xs.len()];
            unsafe {
                mercury_vmath2_f32(
                    xs.as_ptr(),
                    ys.as_ptr(),
                    out.as_mut_ptr(),
                    xs.len() as i64,
                    op,
                );
            }
            for i in 0..xs.len() {
                let want = reference(xs[i], ys[i]);
                assert!(
                    (out[i] - want).abs() <= tol + tol * want.abs(),
                    "op {op} ({},{}): got {} want {want}",
                    xs[i],
                    ys[i],
                    out[i]
                );
                assert_eq!(
                    out[i].to_bits(),
                    apply2_1(op, xs[i], ys[i]).to_bits(),
                    "op {op} tail ({},{})",
                    xs[i],
                    ys[i]
                );
            }
        };
        // pow: base in (0, 4], exponent in [-2, 2].
        let xb: Vec<f32> = (0..n).map(|i| 0.01 + (i % 400) as f32 * 0.01).collect();
        let ye: Vec<f32> = (0..n).map(|i| (i % 81) as f32 * 0.05 - 2.0).collect();
        check(VM2_POW, &xb, &ye, |x, y| x.powf(y), 5e-5);
        // atan2/hypot: paired sweep over the four quadrants, never both zero.
        let ya: Vec<f32> = (0..n).map(|i| (i as f32 - 501.0) * 0.013).collect();
        let xa: Vec<f32> = (0..n).map(|i| (i as f32 - 499.0) * 0.011).collect();
        check(VM2_ATAN2, &ya, &xa, |y, x| y.atan2(x), 5e-5);
        check(VM2_HYPOT, &ya, &xa, |a, b| a.hypot(b), 5e-5);
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
            assert_eq!(
                got.to_bits(),
                apply1(VM_EXPM1, x).to_bits(),
                "expm1 tail {x}"
            );
        }
        let xl: Vec<f32> = (0..2001).map(|i| -0.9 + i as f32 * 0.01).collect();
        for (i, &x) in xl.iter().enumerate() {
            let (got, want) = (kernel(VM_LOG1P, &xl)[i], x.ln_1p());
            assert!(
                (got - want).abs() <= 5e-5 + 5e-5 * want.abs(),
                "log1p({x}): got {got} want {want}"
            );
            assert_eq!(
                got.to_bits(),
                apply1(VM_LOG1P, x).to_bits(),
                "log1p tail {x}"
            );
        }
        // Small-x relative accuracy — the whole point of the stable forms.
        let small: Vec<f32> = vec![1e-3, 1e-4, 1e-5, 1e-6, -1e-3, -1e-4, -1e-5];
        let (em, lm) = (kernel(VM_EXPM1, &small), kernel(VM_LOG1P, &small));
        for (i, &x) in small.iter().enumerate() {
            let (we, wl) = (x.exp_m1(), x.ln_1p());
            assert!(
                (em[i] - we).abs() <= 5e-5 * we.abs(),
                "expm1 rel {x}: {} vs {we}",
                em[i]
            );
            assert!(
                (lm[i] - wl).abs() <= 5e-5 * wl.abs(),
                "log1p rel {x}: {} vs {wl}",
                lm[i]
            );
        }
    }

    /// The AVX2 lanes and the scalar tail/fallback must agree element-for-element, so a length that is
    /// not a multiple of 8 produces a consistent result regardless of where the tail starts.
    #[test]
    fn vmath_tail_matches_lanes() {
        let xs: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.013).collect();
        for op in [
            VM_EXP,
            VM_LOG,
            VM_TANH,
            VM_SIGMOID,
            VM_RELU,
            VM_SILU,
            VM_GELU,
            VM_SIN,
            VM_COS,
            VM_ERF,
            VM_EXP2,
            VM_LOG2,
            VM_SINH,
            VM_COSH,
            VM_ASINH,
            VM_ATAN,
            VM_EXPM1,
            VM_EXP10,
            VM_LOG10,
            VM_SOFTSIGN,
            VM_LOGSIGMOID,
            VM_CBRT,
        ] {
            if op == VM_LOG || op == VM_LOG2 || op == VM_LOG10 {
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

    /// The 8-bucket log tables are self-consistent: R[j] is 1/midpoint of bucket j rounded once to
    /// f32 (bucket 4 — the bucket straddling z = 1 — pinned to exactly 1.0 with L = 0.0, so ln of
    /// x near 1 reduces to the pure poly path with no cancellation), and L[j] = −ln(R[j]) computed
    /// in f64 *from the rounded-f32 R* and rounded once to f32. Re-derives both from scratch and
    /// pins every entry bit-for-bit, so a transcription slip in either table (or a drift from the
    /// mir_build mirror's f64 literals, which must round to these same bits) cannot survive.
    #[test]
    fn vmath_log_tables_consistent() {
        for j in 0..8u32 {
            let lo = f32::from_bits((LOG_OFF as u32) + j * 0x0010_0000) as f64;
            let hi = f32::from_bits((LOG_OFF as u32) + (j + 1) * 0x0010_0000) as f64;
            let (want_r, want_l) = if j == 4 {
                assert!(lo < 1.0 && 1.0 < hi, "bucket 4 must straddle 1.0");
                (1.0f32, 0.0f32)
            } else {
                let r = (1.0 / (0.5 * (lo + hi))) as f32;
                (r, (-((r as f64).ln())) as f32)
            };
            assert_eq!(LOG_TBL_R[j as usize].to_bits(), want_r.to_bits(), "R[{j}]");
            assert_eq!(LOG_TBL_L[j as usize].to_bits(), want_l.to_bits(), "L[{j}]");
        }
    }

    /// Dense accuracy sweep of the table-based log core vs f64 `ln`: (1) 300k log-spaced points
    /// across the full normal range [1.2e-38, 1e38], and (2) 200k linear points in [0.9, 1.1] —
    /// the danger zone where ln(x) → 0 and any L[j] + k·ln2 cancellation would blow relative error
    /// up (bucket 4's pinned R = 1, L = 0 is what prevents it). Measured max relative error of the
    /// shipped kernel: 5.24e-7 over 1e6 log-spaced points, 6.85e-7 over [0.9, 1.1], and 6.93e-7
    /// exhaustive over every f32 in [0.25, 4) (33.5M values, measured once offline); asserted
    /// < 1e-6 here. Also pins, per element, kernel == scalar twin bit-for-bit (the log-family
    /// lane==tail check `vmath_tail_matches_lanes` skips for domain reasons), and ln(1) == 0
    /// exactly.
    #[test]
    fn vmath_log_dense_sweep() {
        let mut xs: Vec<f32> = Vec::new();
        let (llo, lhi) = ((1.2e-38f64).ln(), (1e38f64).ln());
        for i in 0..300_000 {
            xs.push((llo + (lhi - llo) * (i as f64) / 299_999.0).exp() as f32);
        }
        for i in 0..200_000 {
            xs.push((0.9 + 0.2 * (i as f64) / 199_999.0) as f32);
        }
        xs.push(1.0);
        let mut out = vec![0.0f32; xs.len()];
        unsafe {
            mercury_vmath_f32(xs.as_ptr(), out.as_mut_ptr(), xs.len() as i64, VM_LOG);
        }
        let mut max_rel = 0.0f64;
        let mut worst = 0.0f32;
        for (i, &x) in xs.iter().enumerate() {
            let got = out[i];
            // Lane == scalar twin, bit-for-bit (the tail and the no-AVX2 fallback both take log1).
            assert_eq!(got.to_bits(), apply1(VM_LOG, x).to_bits(), "lane vs scalar at {x}");
            let want = (x as f64).ln();
            if want == 0.0 {
                assert_eq!(got.to_bits(), 0.0f32.to_bits(), "ln(1) must be exactly 0");
                continue;
            }
            let rel = ((got as f64 - want) / want).abs();
            if rel > max_rel {
                max_rel = rel;
                worst = x;
            }
        }
        assert!(
            max_rel < 1e-6,
            "log max relative error {max_rel:.3e} at x={worst} exceeds 1e-6"
        );
    }

    /// Monotonicity across the 8-bucket boundaries: scan consecutive f32s (±256 ULPs) around every
    /// bucket edge — including the two k-transition edges where z wraps — and require ln to never
    /// decrease. The shipped tables measure zero violations (a table/poly mismatch at an edge would
    /// show up as a multi-ULP step down), so this asserts strict no-decrease rather than a wobble
    /// budget.
    #[test]
    fn vmath_log_bucket_boundary_monotone() {
        for j in 0..=8u32 {
            let boundary = (LOG_OFF as u32) + j * 0x0010_0000;
            for b in (boundary - 256)..(boundary + 256) {
                let x0 = f32::from_bits(b);
                let x1 = f32::from_bits(b + 1);
                let (y0, y1) = (apply1(VM_LOG, x0), apply1(VM_LOG, x1));
                assert!(
                    y1 >= y0,
                    "log not monotone at bucket edge: x={x0:?} -> {y0:?}, next {x1:?} -> {y1:?}"
                );
            }
        }
    }

    /// The 8-bucket exp tables and reduction constants are self-consistent: T[j] is 2^(j/8)
    /// computed in f64 and rounded once to f32 (T[0] pinned exactly 1.0 so exp(0) = 1.0 exactly),
    /// SCALE is 8/ln2 rounded once, MBIAS folds the 127·8 exponent bias into the magic-bits
    /// subtract, and the Cody-Waite pair is the exact /8 of the shared EXP_C1/EXP_C2 ln2 split —
    /// with n·C1 checked EXACT over |n| ≤ 2047 (the clamp range only needs 1020), which is the
    /// whole hi/lo argument. Re-derives everything from scratch and pins the bits, so a
    /// transcription slip in either this table or the mir_build mirror's f64 literals (which must
    /// round to these same bits) cannot survive.
    #[test]
    fn vmath_exp_tables_consistent() {
        for j in 0..8usize {
            let want = ((j as f64) / 8.0).exp2() as f32;
            assert_eq!(EXP_TBL_T[j].to_bits(), want.to_bits(), "T[{j}]");
        }
        assert_eq!(EXP_TBL_T[0].to_bits(), 1.0f32.to_bits(), "T[0] must be exactly 1");
        assert_eq!(EXP_TBL_SCALE.to_bits(), ((8.0f64 / std::f64::consts::LN_2) as f32).to_bits());
        assert_eq!(EXP_TBL_MBIAS, EXP_MAGIC.to_bits() as i32 - 127 * 8);
        // Cody-Waite hi: the exact /8 of EXP_C1, with ≥10 trailing mantissa zero bits (it has 15)
        // so that n·C1 is exact for the full |n| ≤ 1020 clamp range — checked exhaustively with
        // 2× headroom against the f64 product.
        assert_eq!(EXP_TBL_C1.to_bits(), (EXP_C1 / 8.0).to_bits());
        assert_eq!(EXP_TBL_C2.to_bits(), (EXP_C2 / 8.0).to_bits());
        assert!(EXP_TBL_C1.to_bits().trailing_zeros() >= 10, "C1 lost its trailing zeros");
        for n in -2047i32..=2047 {
            let prod = (n as f32) * EXP_TBL_C1;
            assert_eq!(prod as f64, (n as f64) * (EXP_TBL_C1 as f64), "n·C1 inexact at n={n}");
        }
        // hi+lo reproduce ln2/8 to ≈2e-13 (×|n| ≤ 1020 → ≤2.2e-10 absolute in r — invisible in f32).
        let resid = ((EXP_TBL_C1 as f64 + EXP_TBL_C2 as f64) - std::f64::consts::LN_2 / 8.0).abs();
        assert!(resid < 1e-12, "Cody-Waite pair drifted off ln2/8: {resid:e}");
        // P2 is the Chebyshev-shifted r² coefficient 1/2 + (√2−1)/12·h² (h = ln2/16), rounded once;
        // P3 is 1/6 rounded once.
        let h = std::f64::consts::LN_2 / 16.0;
        let want_p2 = (0.5 + (2f64.sqrt() - 1.0) / 12.0 * h * h) as f32;
        assert_eq!(EXP_TBL_P2.to_bits(), want_p2.to_bits(), "P2");
        assert_eq!(EXP_TBL_P3.to_bits(), ((1.0f64 / 6.0) as f32).to_bits(), "P3");
    }

    /// Dense accuracy sweep of the table-based exp core vs f64 `exp`: (1) 2M linear points over
    /// [−87, 88] — the whole clamp range above the denormal-output band, where relative error is
    /// meaningful; and (2) 1M points in [−0.07, 0.07] — the near-1 region the expm1 Kahan
    /// correction and the softmax tail lean on, spanning every bucket transition around n = 0.
    /// Measured max relative error of the shipped kernel: 1.59e-7 wide and 1.24e-7 near 0
    /// (the degree-3 tail with the Chebyshev-shifted P2; plain 1/2 measured 2.77e-7); asserted
    /// < 2.4e-7 (≈2 ULP at 1.0) here. Also pins, per element, kernel == scalar twin bit-for-bit,
    /// and exp(0) == 1.0 exactly.
    #[test]
    fn vmath_exp_dense_sweep() {
        let mut xs: Vec<f32> = Vec::new();
        for i in 0..2_000_000 {
            xs.push((-87.0 + 175.0 * (i as f64) / 1_999_999.0) as f32);
        }
        for i in 0..1_000_000 {
            xs.push((-0.07 + 0.14 * (i as f64) / 999_999.0) as f32);
        }
        xs.push(0.0);
        let mut out = vec![0.0f32; xs.len()];
        unsafe {
            mercury_vmath_f32(xs.as_ptr(), out.as_mut_ptr(), xs.len() as i64, VM_EXP);
        }
        let mut max_rel = 0.0f64;
        let mut worst = 0.0f32;
        for (i, &x) in xs.iter().enumerate() {
            let got = out[i];
            // Lane == scalar twin, bit-for-bit (the tail and the no-AVX2 fallback both take exp1).
            assert_eq!(got.to_bits(), apply1(VM_EXP, x).to_bits(), "lane vs scalar at {x}");
            if x == 0.0 {
                assert_eq!(got.to_bits(), 1.0f32.to_bits(), "exp(0) must be exactly 1");
                continue;
            }
            let want = (x as f64).exp();
            let rel = ((got as f64 - want) / want).abs();
            if rel > max_rel {
                max_rel = rel;
                worst = x;
            }
        }
        assert!(
            max_rel < 2.4e-7,
            "exp max relative error {max_rel:.3e} at x={worst} exceeds 2.4e-7"
        );
    }

    /// Clamp/saturation semantics of the table exp, pinned bit-for-bit: everything at or below
    /// EXP_LO (including −∞) flushes to +0.0 — the `max(m, 0)` route; everything at or above
    /// EXP_HI (including +∞ — and NaN, which the min-then-max clamp order has always funneled to
    /// the HI path) saturates to the same finite value the old full-range-poly kernel produced
    /// (0x7F3504A4 ≈ 2.406e38 — never +∞); exp(0) == 1.0 exactly; and exp is monotone
    /// non-decreasing through the n-increment boundaries x ≈ (n+½)·ln2/8, where the bucket j and
    /// (every 8th) the exponent e both step — ±256 consecutive f32s scanned in value order around
    /// each, zero violations measured.
    #[test]
    fn vmath_exp_boundary_saturation() {
        for x in [EXP_LO, -88.4, -100.0, -1e6, f32::NEG_INFINITY] {
            assert_eq!(exp1(x).to_bits(), 0.0f32.to_bits(), "exp({x}) must be +0.0");
        }
        let sat = exp1(EXP_HI);
        assert!(sat.is_finite());
        assert_eq!(sat.to_bits(), 0x7F35_04A4, "HI saturation value drifted");
        for x in [88.4, 1e6, f32::INFINITY, f32::NAN] {
            assert_eq!(exp1(x).to_bits(), sat.to_bits(), "exp({x}) must saturate to exp(EXP_HI)");
        }
        assert_eq!(exp1(0.0).to_bits(), 1.0f32.to_bits(), "exp(0) must be exactly 1");
        // Monotonicity across n boundaries (bucket steps, and exponent steps at n ≡ 4 mod 8's
        // neighbors ±1020 covers e transitions too). Bits ascend with value for positive floats
        // and descend for negative ones — walk in value order either way.
        for nb in [-1000i32, -500, -100, -9, -1, 0, 1, 9, 100, 500, 1000] {
            let x0 = ((f64::from(nb) + 0.5) * std::f64::consts::LN_2 / 8.0) as f32;
            let b0 = x0.to_bits();
            for d in 0..512u32 {
                let (ba, bb) = if x0 >= 0.0 {
                    (b0 - 256 + d, b0 - 256 + d + 1)
                } else {
                    (b0 + 256 - d, b0 + 256 - d - 1)
                };
                let (xa, xb) = (f32::from_bits(ba), f32::from_bits(bb));
                let (ya, yb) = (exp1(xa), exp1(xb));
                assert!(
                    yb >= ya,
                    "exp not monotone at n-edge {nb}: x={xa:?} -> {ya:?}, next {xb:?} -> {yb:?}"
                );
            }
        }
    }

    /// Activation **backward** kernels `dx = dy·act'(x)` (silu/gelu): (1) the kernel output equals the
    /// scalar twin `apply2_1` bit-for-bit over a non-multiple-of-8 length — the AVX2 lanes == the scalar
    /// tail, which is what lets the interpreter marshal through this kernel and still match native; and
    /// (2) the scalar twin agrees with an independent f64 closed-form derivative to f32 grade, proving
    /// the kernels compute the real gradient (not just a self-consistent one).
    #[test]
    fn vmath2_activation_backward() {
        // Closed-form derivatives in f64 (independent of the kernel's poly path).
        fn silu_grad_f64(x: f64) -> f64 {
            let s = 1.0 / (1.0 + (-x).exp());
            s + x * s * (1.0 - s)
        }
        fn gelu_grad_f64(x: f64) -> f64 {
            let c0 = (2.0 / std::f64::consts::PI).sqrt();
            let c1 = 0.044715f64;
            let u = (c0 * (x + c1 * x * x * x)).tanh();
            0.5 * (1.0 + u) + 0.5 * x * (1.0 - u * u) * c0 * (1.0 + 3.0 * c1 * x * x)
        }
        fn sigmoid_grad_f64(x: f64) -> f64 {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 - s)
        }
        fn tanh_grad_f64(x: f64) -> f64 {
            let t = x.tanh();
            1.0 - t * t
        }
        fn elu_grad_f64(x: f64) -> f64 {
            if x > 0.0 {
                1.0
            } else {
                x.exp()
            }
        }
        fn softplus_grad_f64(x: f64) -> f64 {
            1.0 / (1.0 + (-x).exp())
        }
        let n = 1003usize; // not a multiple of 8 → exercises the lane body and the scalar tail
        let xs: Vec<f32> = (0..n).map(|i| (i as f32 - 500.0) * 0.011).collect();
        let dys: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        for (op, gref) in [
            (VM2_SILU_BWD, silu_grad_f64 as fn(f64) -> f64),
            (VM2_GELU_BWD, gelu_grad_f64 as fn(f64) -> f64),
            (VM2_SIGMOID_BWD, sigmoid_grad_f64 as fn(f64) -> f64),
            (VM2_TANH_BWD, tanh_grad_f64 as fn(f64) -> f64),
            (VM2_ELU_BWD, elu_grad_f64 as fn(f64) -> f64),
            (VM2_SOFTPLUS_BWD, softplus_grad_f64 as fn(f64) -> f64),
        ] {
            let mut got = vec![0.0f32; n];
            // SAFETY: xs/dys/got are exactly n f32 long — the kernel's contract.
            unsafe {
                mercury_vmath2_f32(xs.as_ptr(), dys.as_ptr(), got.as_mut_ptr(), n as i64, op);
            }
            for i in 0..n {
                // (1) lanes == tail == scalar twin, bit-for-bit.
                assert_eq!(
                    got[i].to_bits(),
                    apply2_1(op, xs[i], dys[i]).to_bits(),
                    "op {op} i {i} x {}",
                    xs[i]
                );
                // (2) ≈ dy · act'(x) from the f64 closed form.
                let want = dys[i] as f64 * gref(xs[i] as f64);
                assert!(
                    (got[i] as f64 - want).abs() <= 1e-4 + 1e-4 * want.abs(),
                    "op {op} backward({}, {}): got {} want {want}",
                    xs[i],
                    dys[i],
                    got[i]
                );
            }
        }
    }

    /// The gated-FFN activations `out = act(a)·b` (SwiGLU/GeGLU): (1) the kernel output equals the
    /// scalar twin `apply2_1` bit-for-bit over a non-multiple-of-8 length (AVX2 lanes == scalar tail,
    /// what lets the interpreter marshal through this kernel); and (2) the scalar twin agrees with an
    /// independent f64 forward `act(a)·b` to f32 grade, proving it's the real gate.
    #[test]
    fn vmath2_gate() {
        fn silu_f64(x: f64) -> f64 {
            x / (1.0 + (-x).exp())
        }
        fn gelu_f64(x: f64) -> f64 {
            let c0 = (2.0 / std::f64::consts::PI).sqrt();
            let c1 = 0.044715f64;
            0.5 * x * (1.0 + (c0 * (x + c1 * x * x * x)).tanh())
        }
        fn sigmoid_f64(x: f64) -> f64 {
            1.0 / (1.0 + (-x).exp())
        }
        let n = 1003usize; // not a multiple of 8 → exercises the lane body and the scalar tail
        let a: Vec<f32> = (0..n).map(|i| (i as f32 - 500.0) * 0.011).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        for (op, fref) in [
            (VM2_SILU_GATE, silu_f64 as fn(f64) -> f64),
            (VM2_GELU_GATE, gelu_f64 as fn(f64) -> f64),
        ] {
            let mut got = vec![0.0f32; n];
            // SAFETY: a/b/got are exactly n f32 long — the kernel's contract.
            unsafe {
                mercury_vmath2_f32(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), n as i64, op);
            }
            for i in 0..n {
                assert_eq!(
                    got[i].to_bits(),
                    apply2_1(op, a[i], b[i]).to_bits(),
                    "op {op} i {i} a {}",
                    a[i]
                );
                let want = b[i] as f64 * fref(a[i] as f64);
                assert!(
                    (got[i] as f64 - want).abs() <= 1e-4 + 1e-4 * want.abs(),
                    "op {op} gate({}, {}): got {} want {want}",
                    a[i],
                    b[i],
                    got[i]
                );
            }
        }
    }

    /// The bf16/f16-input activation kernels must equal the f32 kernel run on the *widened* inputs,
    /// bit-for-bit — that equality is exactly what lets the interpreter marshal `mercury_vmath_{bf16,
    /// f16}` (reconstructing the half bits, calling the very kernel) and still match the native
    /// backend. Covers a non-multiple-of-8 length to exercise both the 8-lane body and the scalar tail.
    #[test]
    fn vmath_lowp_matches_f32_on_widened() {
        let n = 1003usize;
        let xf: Vec<f32> = (0..n).map(|i| (i as f32 - 500.0) * 0.004).collect();
        let ops = [
            VM_EXP,
            VM_TANH,
            VM_SIGMOID,
            VM_RELU,
            VM_SILU,
            VM_GELU,
            VM_ELU,
            VM_SOFTPLUS,
            VM_MISH,
            VM_SIN,
            VM_COS,
            VM_ERF,
            VM_EXP2,
            VM_SINH,
            VM_COSH,
            VM_ASINH,
            VM_ATAN,
            VM_EXPM1,
        ];
        // bf16
        let bbits: Vec<u16> = xf.iter().map(|&v| crate::f32_to_bf16_bits(v)).collect();
        let bwide: Vec<f32> = bbits.iter().map(|&b| crate::bf16_bits_to_f32(b)).collect();
        // f16
        let hbits: Vec<u16> = xf.iter().map(|&v| crate::f32_to_f16_bits(v)).collect();
        let hwide: Vec<f32> = hbits.iter().map(|&b| crate::f16_bits_to_f32(b)).collect();
        for op in ops {
            let (mut gb, mut gh, mut rb, mut rh) =
                (vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]);
            unsafe {
                mercury_vmath_bf16(bbits.as_ptr(), gb.as_mut_ptr(), n as i64, op);
                mercury_vmath_f32(bwide.as_ptr(), rb.as_mut_ptr(), n as i64, op);
                mercury_vmath_f16(hbits.as_ptr(), gh.as_mut_ptr(), n as i64, op);
                mercury_vmath_f32(hwide.as_ptr(), rh.as_mut_ptr(), n as i64, op);
            }
            for i in 0..n {
                assert_eq!(gb[i].to_bits(), rb[i].to_bits(), "bf16 op {op} i {i}");
                assert_eq!(gh[i].to_bits(), rh[i].to_bits(), "f16 op {op} i {i}");
            }
        }
    }

    /// In-process throughput: the 256-bit AVX2 kernel vs a scalar `libm`-call loop computing the
    /// *same* function — exactly the code gcc/rustc emit for a `for i { out[i] = f(x[i]) }` loop, which
    /// **cannot vectorize a call** (no `libmvec` on this toolchain). So `scalar/kernel` is the real
    /// per-function speedup over scalar `libm`, measured locally (no cross-language build needed). The
    /// working set fits L2 so the comparison is compute-bound, not bandwidth-bound.
    /// Observed on a Meteor Lake laptop (ratios are clock-invariant): exp ~5.9×, gelu ~5.5×, tan ~3.6×,
    /// asin ~5.4×, acos ~5.0×, exp10 ~13× (Rust's `powf` is a heavy general path), log10 ~4.1×,
    /// logsigmoid ~4.6×.
    /// Run: `cargo test -p mercury_runtime --release vmath_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "throughput bench; run explicitly in --release"]
    fn vmath_throughput() {
        use std::time::Instant;
        let n = 1 << 16; // 64K f32 = 256 KB, fits L2 → compute-bound
        let xs: Vec<f32> = (0..n).map(|i| ((i % 1801) as f32 / 1000.0) - 0.9).collect(); // [-0.9, 0.9]
        let mut out = vec![0.0f32; n];
        let best = |iters: usize, mut f: Box<dyn FnMut()>| -> f64 {
            f(); // warmup
            let mut t = f64::INFINITY;
            for _ in 0..iters {
                let t0 = Instant::now();
                f();
                t = t.min(t0.elapsed().as_secs_f64());
            }
            t
        };
        // (op, name, scalar libm twin) — the new transcendentals plus a few references. All are true
        // libm calls C must keep scalar (softsign omitted: it's abs+div, which C *can* autovectorize).
        let cases: &[(i64, &str, fn(f32) -> f32)] = &[
            (VM_EXP, "exp", |x| x.exp()),
            (VM_GELU, "gelu", |x| {
                0.5 * x * (1.0 + (GELU_C0 * (x + GELU_C1 * x * x * x)).tanh())
            }),
            (VM_TAN, "tan", |x| x.tan()),
            (VM_ASIN, "asin", |x| x.asin()),
            (VM_ACOS, "acos", |x| x.acos()),
            (VM_EXP10, "exp10", |x| 10.0f32.powf(x)),
            (VM_LOG10, "log10", |x| (x + 1.1).log10()), // shift into the positive domain
            (VM_LOGSIGMOID, "logsigmoid", |x| {
                (1.0 / (1.0 + (-x).exp())).ln()
            }),
        ];
        let iters = 200;
        eprintln!("vmath throughput over {n} elements (best of {iters}), kernel vs scalar libm:");
        for &(op, name, scalar) in cases {
            let xp = xs.as_ptr() as usize;
            let op2 = xs.clone();
            let outp = out.as_mut_ptr() as usize;
            let t_kernel = best(
                iters,
                Box::new(move || unsafe {
                    mercury_vmath_f32(xp as *const f32, outp as *mut f32, n as i64, op);
                    std::hint::black_box(outp);
                }),
            );
            let mut so = vec![0.0f32; n];
            let t_scalar = best(
                iters,
                Box::new(move || {
                    for i in 0..n {
                        so[i] = scalar(op2[i]);
                    }
                    std::hint::black_box(so.as_ptr());
                }),
            );
            let gelems = |t: f64| n as f64 / t / 1e9;
            eprintln!(
                "  {name:<11} kernel {:.0} Melem/s | scalar libm {:.0} Melem/s | {:.1}x",
                gelems(t_kernel) * 1e3,
                gelems(t_scalar) * 1e3,
                t_scalar / t_kernel
            );
        }
    }

    /// Two-input throughput: the 256-bit `mercury_vmath2_f32` kernel vs a scalar `libm`-call loop
    /// (`powf`/`atan2f`/`hypotf`), the code gcc/rustc emit and cannot vectorize. `scalar/kernel` is the
    /// per-function speedup, clock-invariant. Working set fits L2 → compute-bound.
    /// Observed (Meteor Lake): pow ~4.6×, atan2 ~7.1×, hypot ~4.4× — the 256-bit two-input kernel over
    /// the unvectorizable scalar `powf`/`atan2f`/`hypotf` loop.
    /// Run: `cargo test -p mercury_runtime --release vmath2_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "throughput bench; run explicitly in --release"]
    fn vmath2_throughput() {
        use std::time::Instant;
        let n = 1 << 16;
        let xs: Vec<f32> = (0..n).map(|i| 0.5 + (i % 397) as f32 * 0.01).collect();
        let ys: Vec<f32> = (0..n).map(|i| 0.5 + (i % 311) as f32 * 0.01).collect();
        let mut out = vec![0.0f32; n];
        let best = |iters: usize, mut f: Box<dyn FnMut()>| -> f64 {
            f();
            let mut t = f64::INFINITY;
            for _ in 0..iters {
                let t0 = Instant::now();
                f();
                t = t.min(t0.elapsed().as_secs_f64());
            }
            t
        };
        let cases: &[(i64, &str, fn(f32, f32) -> f32)] = &[
            (VM2_POW, "pow", |x, y| x.powf(y)),
            (VM2_ATAN2, "atan2", |y, x| y.atan2(x)),
            (VM2_HYPOT, "hypot", |a, b| a.hypot(b)),
        ];
        let iters = 200;
        eprintln!("vmath2 throughput over {n} pairs (best of {iters}), kernel vs scalar libm:");
        for &(op, name, scalar) in cases {
            let (xp, yp) = (xs.as_ptr() as usize, ys.as_ptr() as usize);
            let outp = out.as_mut_ptr() as usize;
            let t_kernel = best(
                iters,
                Box::new(move || unsafe {
                    mercury_vmath2_f32(xp as *const f32, yp as *const f32, outp as *mut f32, n as i64, op);
                    std::hint::black_box(outp);
                }),
            );
            let (xc, yc) = (xs.clone(), ys.clone());
            let mut so = vec![0.0f32; n];
            let t_scalar = best(
                iters,
                Box::new(move || {
                    for i in 0..n {
                        so[i] = scalar(xc[i], yc[i]);
                    }
                    std::hint::black_box(so.as_ptr());
                }),
            );
            eprintln!(
                "  {name:<6} kernel {:.0} Melem/s | scalar libm {:.0} Melem/s | {:.1}x",
                n as f64 / t_kernel / 1e6,
                n as f64 / t_scalar / 1e6,
                t_scalar / t_kernel
            );
        }
    }

    /// The **non-temporal store regime** of the one-input kernel must stay bit-identical to the scalar
    /// twin. A >L3 length forces `use_nt`, so this exercises the whole new streaming structure — the
    /// 32-byte alignment prologue, the ×4 body, the 8-wide remainder, the `sfence`, and the scalar tail.
    /// NT and cacheable stores write the *same bits* (only the cache path differs), so every lane must
    /// still equal `apply1` — the equality the interpreter marshalling relies on. An odd length gives a
    /// non-multiple-of-8 tail, and the prologue peels to alignment regardless of the allocation's base.
    #[test]
    fn vmath_nt_tail_matches_lanes() {
        let n = 1_500_001usize; // 2-stream = 12 MiB > NT_MIN_BYTES: forces NT + prologue + tail
        // Kept > 0 so log/log-based ops stay in domain; a short period spans the poly's regions.
        let xs: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.1 + 0.05).collect();
        for op in [VM_EXP, VM_LOG, VM_TANH, VM_GELU, VM_SIN, VM_ERF, VM_RELU] {
            let mut got = vec![0.0f32; n];
            unsafe { mercury_vmath_f32(xs.as_ptr(), got.as_mut_ptr(), n as i64, op) };
            for (i, &x) in xs.iter().enumerate() {
                assert_eq!(got[i].to_bits(), apply1(op, x).to_bits(), "op {op} i {i} x {x}");
            }
        }
    }

    /// The NT regime of the two-input kernel: a >L3 (3-stream) length forces the prologue / ×4 body /
    /// 8-wide remainder / `sfence` / scalar tail, and every lane must equal `apply2_1` bit-for-bit
    /// (again NT vs cacheable stores write identical bits). `pow` keeps its base > 0.
    #[test]
    fn vmath2_nt_tail_matches_lanes() {
        let n = 1_500_001usize; // 3-stream = 18 MiB > NT_MIN_BYTES: forces NT + prologue + tail
        let xs: Vec<f32> = (0..n).map(|i| (i % 89) as f32 * 0.05 + 0.1).collect();
        let ys: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect();
        for op in [VM2_POW, VM2_ATAN2, VM2_HYPOT, VM2_SILU_BWD, VM2_TANH_BWD, VM2_SILU_GATE] {
            let mut got = vec![0.0f32; n];
            unsafe { mercury_vmath2_f32(xs.as_ptr(), ys.as_ptr(), got.as_mut_ptr(), n as i64, op) };
            for i in 0..n {
                assert_eq!(
                    got[i].to_bits(),
                    apply2_1(op, xs[i], ys[i]).to_bits(),
                    "op {op} i {i}"
                );
            }
        }
    }
}
