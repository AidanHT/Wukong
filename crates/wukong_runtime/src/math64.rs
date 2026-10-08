//! Double-precision transcendentals — the `f64` twin of the f32 `vmath` family, and the library an
//! `f64` `exp`/`log`/`sin`/`pow`/… call in a Wukong program resolves to on **every** backend.
//!
//! The compiler lowers a transcendental on an `f64` operand to one call of
//! [`wukong_rt_math_f64`] (`VM_*` code) or [`wukong_rt_math2_f64`] (`VM2_*` code). The interpreter
//! calls these Rust functions directly; the Cranelift JIT binds the same symbols; the `--emit=exe`
//! link resolves them out of this crate's rlib; the textual-LLVM emitter declares them. There is
//! exactly one implementation, so interp == native == AOT bit-for-bit — which is the contract the
//! old lowering could only keep by demoting every `f64` transcendental to the f32 polynomial
//! (≈7 significant digits on a type that promises ≈16).
//!
//! **Pure Rust, no platform libm, no FMA.** The only floating primitives used are IEEE-754
//! `+ − × ÷`, `sqrt`, `floor`/`trunc` and bit manipulation — each correctly rounded or exact on
//! every host — so the result of every function is the same bit pattern on Windows, Linux and macOS,
//! on a chip with or without FMA. (A `mul_add` would be correctly rounded too, but it falls back to
//! a software `fma` on hosts without the instruction; these algorithms were designed without it.)
//!
//! **Algorithms.** FreeBSD msun / fdlibm (Sun Microsystems, 1993–2004) as carried by musl and the
//! `libm` crate (MIT), ported function-for-function: `exp`/`log`/`pow`/`expm1`/`log1p`/`exp2`/
//! `log2`/`log10`, the `sin`/`cos`/`tan` kernels with the **Payne–Hanek** large-argument reduction
//! (`rem_pio2_large`, exact for every finite `f64`, so `sin(1e22)` is right), `asin`/`acos`/`atan`/
//! `atan2`, the `expm1`-based hyperbolics, `erf`, `cbrt` and `hypot`. The constants are the
//! intended hex values from the originals; `math64_constants_match_their_hex` pins every one.
//!
//! **Accuracy (ULP of the correctly rounded result, finite normal results).** The documented bounds
//! of the originals, which `math64_reference_values_within_ulp_bound` checks against mpmath
//! references embedded below:
//!
//! | function | bound |
//! |---|---|
//! | `exp`, `log`, `log2`, `log10`, `expm1`, `log1p`, `exp2`, `pow`, `hypot` | ≤ 1 |
//! | `sin`, `cos`, `tan` (every finite argument), `asin`, `acos`, `atan`, `atan2` | ≤ 1 |
//! | `erf`, `cbrt` | ≤ 1 |
//! | `sinh`, `cosh`, `tanh` (≤ 2 in `[0.1, 0.2554]`), `asinh`, `acosh`, `atanh` | ≤ 2 |
//! | `exp10` (`exp2(log2(10)·frac(x))·10^int(x)`) | ≤ 3 |
//!
//! **Special values** follow C23 Annex F: `exp(−∞) = +0`, `log(±0) = −∞`, `log(x<0) = NaN`,
//! `sin/cos/tan(±∞) = NaN`, `asin(±1) = ±π/2`, `acos(−1) = π`, `atan(±∞) = ±π/2`, the full
//! `atan2` quadrant/axis table, `tanh(±∞) = ±1`, `erf(±∞) = ±1`, `atanh(±1) = ±∞`, `cbrt(±0) = ±0`,
//! `cbrt(±∞) = ±∞`, `pow`'s 23 special cases (`pow(x, ±0) = 1` and `pow(1, y) = 1` even for NaN),
//! `hypot(±∞, NaN) = +∞`. Odd functions return `−0` for `−0`, and every function is exactly `x`
//! (sign included) on arguments below its linear threshold. A NaN argument yields a NaN; its
//! payload is whatever the originals propagate (the compiler never observes a payload here).
//!
//! An unknown op code returns the first argument unchanged — the same total-table convention as
//! [`wukong_rt_trig_f32`](crate::wukong_rt_trig_f32) and `wukong_vmath_f32`; the compiler only ever
//! emits the codes the dispatch tables list, and `math64_unknown_opcode_is_identity` pins it.

// The originals are written as C does them: long decimal constants carrying their hex intent,
// shifts mixed with adds in exponent arithmetic, and single-letter working variables. Rewriting
// them to satisfy style lints would hide the one-to-one correspondence with the reviewed sources,
// which is this module's correctness argument. (Their `x - x` NaN idiom is the one deliberate
// departure: it is spelled [`domain_nan`] here, see there.)
#![allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::needless_late_init,
    clippy::manual_range_contains,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::identity_op,
    clippy::precedence,
    clippy::needless_return,
    clippy::assign_op_pattern,
    clippy::if_same_then_else
)]

use crate::special::{M64_DIGAMMA, M64_ERFC, M64_ERFINV, M64_LGAMMA};
use crate::{
    VM2_ATAN2, VM2_HYPOT, VM2_POW, VM_ACOS, VM_ACOSH, VM_ASIN, VM_ASINH, VM_ATAN, VM_ATANH,
    VM_CBRT, VM_COS, VM_COSH, VM_ERF, VM_EXP, VM_EXP10, VM_EXP2, VM_EXPM1, VM_LOG, VM_LOG10,
    VM_LOG1P, VM_LOG2, VM_SIN, VM_SINH, VM_TAN, VM_TANH,
};

// --- the C ABI the compiler emits against ------------------------------------------------------

/// Single-argument double-precision transcendental: `f(x)` for the `VM_*` code `op`.
///
/// Codes implemented: `VM_EXP`, `VM_LOG`, `VM_TANH`, `VM_SIN`, `VM_COS`, `VM_ERF`, `VM_EXP2`,
/// `VM_LOG2`, `VM_SINH`, `VM_COSH`, `VM_ASINH`, `VM_ACOSH`, `VM_ATANH`, `VM_ATAN`, `VM_EXPM1`,
/// `VM_LOG1P`, `VM_EXP10`, `VM_LOG10`, `VM_TAN`, `VM_ASIN`, `VM_ACOS`, `VM_CBRT` — the transcendental
/// core — plus the library-only special functions `M64_ERFC`, `M64_ERFINV`, `M64_LGAMMA` and
/// `M64_DIGAMMA` ([`crate::special`]), which have no f32 kernel: an `f32` call of one widens to
/// `f64`, comes here, and rounds back once. The activations (`sigmoid`, `silu`, `gelu`, `softplus`, …) are composed from these in MIR
/// by `wukong_mir_build`, exactly as for f32, so they stay differentiable through primitive ops;
/// their codes (and any other) return `x` unchanged.
#[no_mangle]
pub extern "C" fn wukong_rt_math_f64(x: f64, op: i64) -> f64 {
    match op {
        VM_EXP => exp(x),
        VM_LOG => log(x),
        VM_TANH => tanh(x),
        VM_SIN => sin(x),
        VM_COS => cos(x),
        VM_ERF => erf(x),
        VM_EXP2 => exp2(x),
        VM_LOG2 => log2(x),
        VM_SINH => sinh(x),
        VM_COSH => cosh(x),
        VM_ASINH => asinh(x),
        VM_ACOSH => acosh(x),
        VM_ATANH => atanh(x),
        VM_ATAN => atan(x),
        VM_EXPM1 => expm1(x),
        VM_LOG1P => log1p(x),
        VM_EXP10 => exp10(x),
        VM_LOG10 => log10(x),
        VM_TAN => tan(x),
        VM_ASIN => asin(x),
        VM_ACOS => acos(x),
        VM_CBRT => cbrt(x),
        // The special functions with no f32 `vmath` kernel: every precision computes here.
        M64_ERFC => erfc(x),
        M64_ERFINV => crate::special::erfinv(x),
        M64_LGAMMA => crate::special::lgamma(x),
        M64_DIGAMMA => crate::special::digamma(x),
        _ => x,
    }
}

/// Two-argument double-precision transcendental: `f(a, b)` for the `VM2_*` code `op` —
/// `VM2_POW` is `pow(a, b)`, `VM2_ATAN2` is `atan2(a, b)` (`a` the ordinate, as the language's
/// `atan2(y, x)` spells it), `VM2_HYPOT` is `hypot(a, b)`. Any other code returns `a`.
#[no_mangle]
pub extern "C" fn wukong_rt_math2_f64(a: f64, b: f64, op: i64) -> f64 {
    match op {
        VM2_POW => pow(a, b),
        VM2_ATAN2 => atan2(a, b),
        VM2_HYPOT => hypot(a, b),
        _ => a,
    }
}

// --- word access helpers (the fdlibm GET/SET_HIGH/LOW_WORD idiom) ------------------------------

#[inline]
fn hi(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

#[inline]
fn lo(x: f64) -> u32 {
    x.to_bits() as u32
}

#[inline]
fn with_hi(x: f64, h: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xffff_ffff) | ((h as u64) << 32))
}

#[inline]
fn with_lo(x: f64, l: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xffff_ffff_0000_0000) | (l as u64))
}

#[inline]
fn from_words(h: u32, l: u32) -> f64 {
    f64::from_bits(((h as u64) << 32) | (l as u64))
}

/// A quiet NaN for a domain error (`log(−1)`, `sin(±∞)`, …) on a non-NaN argument; a NaN argument
/// is returned as is. The originals compute `x - x` or `(x - x) / 0` here, which propagates an
/// incoming payload but otherwise yields the hardware's *default* NaN — whose sign bit differs
/// between x86 (negative) and AArch64 (positive). `f64::NAN` is the same bits on every host, which
/// keeps the module's promise that interp, JIT and AOT agree bit-for-bit wherever they run.
#[inline]
fn domain_nan(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else {
        f64::NAN
    }
}

// --- scalbn ------------------------------------------------------------------------------------

/// `x · 2ⁿ`, exact, with the two-step prescale that keeps a representable result representable
/// when `2ⁿ` itself is not (musl `scalbn`).
pub fn scalbn(mut x: f64, mut n: i32) -> f64 {
    let x1p1023 = f64::from_bits(0x7fe0_0000_0000_0000);
    let x1p53 = f64::from_bits(0x4340_0000_0000_0000);
    let x1p_1022 = f64::from_bits(0x0010_0000_0000_0000);
    if n > 1023 {
        x *= x1p1023;
        n -= 1023;
        if n > 1023 {
            x *= x1p1023;
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        // 2^-1022 · 2^53 = 2^-969: a representable intermediate that never flushes a result the
        // final scale could still hold.
        x *= x1p_1022 * x1p53;
        n += 1022 - 53;
        if n < -1022 {
            x *= x1p_1022 * x1p53;
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    x * f64::from_bits(((0x3ff + n) as u64) << 52)
}

// --- exp ---------------------------------------------------------------------------------------

const EXP_HALF: [f64; 2] = [0.5, -0.5];
const LN2HI: f64 = 6.93147180369123816490e-01; /* 0x3fe62e42, 0xfee00000 */
const LN2LO: f64 = 1.90821492927058770002e-10; /* 0x3dea39ef, 0x35793c76 */
const INVLN2: f64 = 1.44269504088896338700e+00; /* 0x3ff71547, 0x652b82fe */
const EXP_P1: f64 = 1.66666666666666019037e-01; /* 0x3FC55555, 0x5555553E */
const EXP_P2: f64 = -2.77777777770155933842e-03; /* 0xBF66C16C, 0x16BEBD93 */
const EXP_P3: f64 = 6.61375632143793436117e-05; /* 0x3F11566A, 0xAF25DE2C */
const EXP_P4: f64 = -1.65339022054652515390e-06; /* 0xBEBBBD41, 0xC5D26BF1 */
const EXP_P5: f64 = 4.13813679705723846039e-08; /* 0x3E663769, 0x72BEA4D0 */

/// `eˣ` (fdlibm `e_exp.c`): `x = k·ln2 + r`, `|r| ≤ ½ln2`, with `r = hi − lo` kept in two parts;
/// `eʳ` from the degree-5 Remez rational `1 + r + r·c/(2 − c)`; then `2ᵏ` by `scalbn`. < 1 ULP.
/// Overflows to `+∞` above `709.782712893384`, underflows to `+0` below `−745.133219101941`.
pub fn exp(x: f64) -> f64 {
    let x1p1023 = f64::from_bits(0x7fe0_0000_0000_0000);
    let mut hx = hi(x);
    let sign = (hx >> 31) as i32;
    hx &= 0x7fff_ffff;
    if hx >= 0x4086_232b {
        // |x| >= 708.39...
        if x.is_nan() {
            return x;
        }
        if x > 709.782712893383973096 {
            // overflow (also +inf)
            return x * x1p1023;
        }
        if x < -745.13321910194110842 {
            // underflow to zero (also -inf)
            return 0.0;
        }
    }
    let (r, hi_part, lo_part, k) = if hx > 0x3fd6_2e42 {
        // |x| > 0.5 ln2
        let k = if hx >= 0x3ff0_a2b2 {
            // |x| >= 1.5 ln2
            (INVLN2 * x + EXP_HALF[sign as usize]) as i32
        } else {
            1 - sign - sign
        };
        let kf = k as f64;
        let h = x - kf * LN2HI; // k·ln2hi is exact here
        let l = kf * LN2LO;
        (h - l, h, l, k)
    } else if hx > 0x3e30_0000 {
        // |x| > 2^-28
        (x, x, 0.0, 0)
    } else {
        return 1.0 + x;
    };
    let xx = r * r;
    let c = r - xx * (EXP_P1 + xx * (EXP_P2 + xx * (EXP_P3 + xx * (EXP_P4 + xx * EXP_P5))));
    let y = 1.0 + (r * c / (2.0 - c) - lo_part + hi_part);
    if k == 0 {
        y
    } else {
        scalbn(y, k)
    }
}

// --- log ---------------------------------------------------------------------------------------

const LG1: f64 = 6.666666666666735130e-01; /* 3FE55555 55555593 */
const LG2: f64 = 3.999999999940941908e-01; /* 3FD99999 9997FA04 */
const LG3: f64 = 2.857142874366239149e-01; /* 3FD24924 94229359 */
const LG4: f64 = 2.222219843214978396e-01; /* 3FCC71C5 1D8E78AF */
const LG5: f64 = 1.818357216161805012e-01; /* 3FC74664 96CB03DE */
const LG6: f64 = 1.531383769920937332e-01; /* 3FC39A09 D078C69F */
const LG7: f64 = 1.479819860511658591e-01; /* 3FC2F112 DF3E5244 */

/// The shared log core: splits a positive finite `x` into `2ᵏ·(1+f)` with `√½ < 1+f < √2` and
/// returns `(k, f, s·(hfsq + R) − hfsq + f)` — the pieces `log`/`log2`/`log10` reassemble with their
/// own hi/lo `ln2` or `log(2)` constants. Returns `Err(result)` for the special values.
#[inline]
fn log_reduce(mut x: f64) -> Result<(i32, f64, f64, f64, f64, f64), f64> {
    let x1p54 = f64::from_bits(0x4350_0000_0000_0000);
    let mut ui = x.to_bits();
    let mut hx = (ui >> 32) as u32;
    let mut k: i32 = 0;
    if hx < 0x0010_0000 || (hx >> 31) != 0 {
        if ui << 1 == 0 {
            return Err(f64::NEG_INFINITY); // log(±0) = −∞
        }
        if (hx >> 31) != 0 {
            return Err(domain_nan(x)); // log(negative) = NaN
        }
        // subnormal: scale up
        k -= 54;
        x *= x1p54;
        ui = x.to_bits();
        hx = (ui >> 32) as u32;
    } else if hx >= 0x7ff0_0000 {
        return Err(x); // +inf or NaN
    } else if hx == 0x3ff0_0000 && ui << 32 == 0 {
        return Err(0.0); // log(1) = +0 exactly
    }
    // reduce x into [sqrt(2)/2, sqrt(2)]
    hx += 0x3ff0_0000 - 0x3fe6_a09e;
    k += ((hx >> 20) as i32) - 0x3ff;
    hx = (hx & 0x000f_ffff) + 0x3fe6_a09e;
    ui = ((hx as u64) << 32) | (ui & 0xffff_ffff);
    x = f64::from_bits(ui);
    let f = x - 1.0;
    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    Ok((k, f, hfsq, s, r, s * (hfsq + r)))
}

/// `ln x` (fdlibm `e_log.c`): `x = 2ᵏ(1+f)`, `s = f/(2+f)`, `log(1+f) = f − (hfsq − s·(hfsq + R))`,
/// `R` the degree-14 even Remez series in `s`; `k·ln2` added as hi + lo. < 1 ULP.
pub fn log(x: f64) -> f64 {
    let (k, f, hfsq, _s, _r, shr) = match log_reduce(x) {
        Ok(parts) => parts,
        Err(special) => return special,
    };
    let dk = k as f64;
    shr + dk * LN2LO - hfsq + f + dk * LN2HI
}

const IVLN2HI: f64 = 1.44269504072144627571e+00; /* 0x3ff71547, 0x65200000 */
const IVLN2LO: f64 = 1.67517131648865118353e-10; /* 0x3de705fc, 0x2eefa200 */

/// `log₂ x` (fdlibm `e_log2.c`): the log core's `log(1+f)` split as `hi + lo` with `hi` truncated to
/// 32 bits so `hi·ivln2hi` is exact, then `k` added in double-double. < 1 ULP.
pub fn log2(x: f64) -> f64 {
    let (k, f, hfsq, _s, _r, shr) = match log_reduce(x) {
        Ok(parts) => parts,
        Err(special) => return special,
    };
    // hi + lo = f − hfsq + s·(hfsq + R) ≈ log(1+f)
    let hi_ = with_lo(f - hfsq, 0);
    let lo_ = f - hi_ - hfsq + shr;
    let val_hi = hi_ * IVLN2HI;
    let mut val_lo = (lo_ + hi_) * IVLN2LO + lo_ * IVLN2HI;
    // spadd(val_hi, val_lo, y)
    let y = k as f64;
    let w = y + val_hi;
    val_lo += (y - w) + val_hi;
    val_lo + w
}

const IVLN10HI: f64 = 4.34294481878168880939e-01; /* 0x3fdbcb7b, 0x15200000 */
const IVLN10LO: f64 = 2.50829467116452752298e-11; /* 0x3dbb9438, 0xca9aadd5 */
const LOG10_2HI: f64 = 3.01029995663611771306e-01; /* 0x3FD34413, 0x509F6000 */
const LOG10_2LO: f64 = 3.69423907715893078616e-13; /* 0x3D59FEF3, 0x11F12B36 */

/// `log₁₀ x` (fdlibm `e_log10.c`): as [`log2`], with `k·log₁₀2` in hi/lo parts. < 1 ULP.
pub fn log10(x: f64) -> f64 {
    let (k, f, hfsq, _s, _r, shr) = match log_reduce(x) {
        Ok(parts) => parts,
        Err(special) => return special,
    };
    let hi_ = with_lo(f - hfsq, 0);
    let lo_ = f - hi_ - hfsq + shr;
    let val_hi = hi_ * IVLN10HI;
    let dk = k as f64;
    let y = dk * LOG10_2HI;
    let mut val_lo = dk * LOG10_2LO + (lo_ + hi_) * IVLN10LO + lo_ * IVLN10HI;
    let w = y + val_hi;
    val_lo += (y - w) + val_hi;
    val_lo + w
}

// --- argument reduction for the trig family ----------------------------------------------------

const PIO2_1: f64 = 1.57079632673412561417e+00; /* 0x3FF921FB, 0x54400000 — first 33 bits of π/2 */
const PIO2_1T: f64 = 6.07710050650619224932e-11; /* 0x3DD0B461, 0x1A626331 — π/2 − PIO2_1 */
const PIO2_2: f64 = 6.07710050630396597660e-11; /* 0x3DD0B461, 0x1A600000 — second 33 bits */
const PIO2_2T: f64 = 2.02226624879595063154e-21; /* 0x3BA3198A, 0x2E037073 */
const PIO2_3: f64 = 2.02226624871116645580e-21; /* 0x3BA3198A, 0x2E000000 — third 33 bits */
const PIO2_3T: f64 = 8.47842766036889956997e-32; /* 0x397B839A, 0x252049C1 */
const INV_PIO2: f64 = 6.36619772367581382433e-01; /* 0x3FE45F30, 0x6DC9C883 — 53 bits of 2/π */
const TO_INT: f64 = 1.5 / 2.2204460492503131e-16; // 1.5 · 2^52: the round-to-nearest magic

/// `x mod π/2` as `(n, y0, y1)` with `x = n·π/2 + y0 + y1`, `|y0 + y1| ≤ π/4` (fdlibm
/// `e_rem_pio2.c`): the Cody–Waite three-part `π/2` for `|x| < 2²⁰·π/2` (up to three rounds, each
/// exact), and [`rem_pio2_large`] — Payne–Hanek over the 2/π bit table — above, so every finite
/// argument reduces exactly. The caller handles `|x| ≤ π/4` itself.
fn rem_pio2(x: f64) -> (i32, f64, f64) {
    let x1p24 = f64::from_bits(0x4170_0000_0000_0000);
    let sign = (x.to_bits() >> 63) as i32;
    let ix = hi(x) & 0x7fff_ffff;

    fn medium(x: f64, ix: u32) -> (i32, f64, f64) {
        // rint(x / (π/2)), assuming round-to-nearest
        let f_n = (x * INV_PIO2 + TO_INT) - TO_INT;
        let n = f_n as i32;
        let mut r = x - f_n * PIO2_1;
        let mut w = f_n * PIO2_1T; // 1st round, good to 85 bits
        let mut y0 = r - w;
        let ey = (y0.to_bits() >> 52) as i32 & 0x7ff;
        let ex = (ix >> 20) as i32;
        if ex - ey > 16 {
            // 2nd round, good to 118 bits
            let t = r;
            w = f_n * PIO2_2;
            r = t - w;
            w = f_n * PIO2_2T - ((t - r) - w);
            y0 = r - w;
            let ey = (y0.to_bits() >> 52) as i32 & 0x7ff;
            if ex - ey > 49 {
                // 3rd round, good to 151 bits, covers all cases
                let t = r;
                w = f_n * PIO2_3;
                r = t - w;
                w = f_n * PIO2_3T - ((t - r) - w);
                y0 = r - w;
            }
        }
        let y1 = (r - y0) - w;
        (n, y0, y1)
    }

    if ix <= 0x400f_6a7a {
        // |x| ~<= 5π/4
        if (ix & 0xf_ffff) == 0x9_21fb {
            // |x| ~= π/2 or 2π/2: cancellation, use the medium case
            return medium(x, ix);
        }
        if ix <= 0x4002_d97c {
            // |x| ~<= 3π/4
            if sign == 0 {
                let z = x - PIO2_1; // one round good to 85 bits
                let y0 = z - PIO2_1T;
                let y1 = (z - y0) - PIO2_1T;
                return (1, y0, y1);
            } else {
                let z = x + PIO2_1;
                let y0 = z + PIO2_1T;
                let y1 = (z - y0) + PIO2_1T;
                return (-1, y0, y1);
            }
        } else if sign == 0 {
            let z = x - 2.0 * PIO2_1;
            let y0 = z - 2.0 * PIO2_1T;
            let y1 = (z - y0) - 2.0 * PIO2_1T;
            return (2, y0, y1);
        } else {
            let z = x + 2.0 * PIO2_1;
            let y0 = z + 2.0 * PIO2_1T;
            let y1 = (z - y0) + 2.0 * PIO2_1T;
            return (-2, y0, y1);
        }
    }
    if ix <= 0x401c_463b {
        // |x| ~<= 9π/4
        if ix <= 0x4015_fdbc {
            // |x| ~<= 7π/4
            if ix == 0x4012_d97c {
                // |x| ~= 3π/2
                return medium(x, ix);
            }
            if sign == 0 {
                let z = x - 3.0 * PIO2_1;
                let y0 = z - 3.0 * PIO2_1T;
                let y1 = (z - y0) - 3.0 * PIO2_1T;
                return (3, y0, y1);
            } else {
                let z = x + 3.0 * PIO2_1;
                let y0 = z + 3.0 * PIO2_1T;
                let y1 = (z - y0) + 3.0 * PIO2_1T;
                return (-3, y0, y1);
            }
        } else {
            if ix == 0x4019_21fb {
                // |x| ~= 4π/2
                return medium(x, ix);
            }
            if sign == 0 {
                let z = x - 4.0 * PIO2_1;
                let y0 = z - 4.0 * PIO2_1T;
                let y1 = (z - y0) - 4.0 * PIO2_1T;
                return (4, y0, y1);
            } else {
                let z = x + 4.0 * PIO2_1;
                let y0 = z + 4.0 * PIO2_1T;
                let y1 = (z - y0) + 4.0 * PIO2_1T;
                return (-4, y0, y1);
            }
        }
    }
    if ix < 0x4139_21fb {
        // |x| ~< 2^20·(π/2), medium size
        return medium(x, ix);
    }
    // all other (large) arguments
    if ix >= 0x7ff0_0000 {
        // inf or NaN
        let y0 = domain_nan(x);
        return (0, y0, y0);
    }
    // set z = scalbn(|x|, −ilogb(x) + 23) and split it into three 24-bit integer-valued doubles
    let mut ui = x.to_bits();
    ui &= (!1u64) >> 12;
    ui |= (0x3ff + 23) << 52;
    let mut z = f64::from_bits(ui);
    let mut tx = [0.0f64; 3];
    for t in tx.iter_mut().take(2) {
        *t = z as i32 as f64;
        z = (z - *t) * x1p24;
    }
    tx[2] = z;
    // skip zero terms; the first term is non-zero
    let mut i = 2;
    while i != 0 && tx[i] == 0.0 {
        i -= 1;
    }
    let mut ty = [0.0f64; 2];
    let n = rem_pio2_large(&tx[..=i], &mut ty, ((ix as i32) >> 20) - (0x3ff + 23));
    if sign != 0 {
        return (-n, -ty[0], -ty[1]);
    }
    (n, ty[0], ty[1])
}

/// The bits of 2/π after the binary point, 24 per entry: entry `i` holds bits `24i .. 24i+23`, so
/// the value is `Σ IPIO2[i]·2^(−24(i+1))`. 66 entries cover every `f64` exponent (the reduction
/// needs `(e0 − 3)/24 + 4 ≤ 46` of them for `e0 ≤ 1000`); quad precision would need 690.
#[rustfmt::skip]
const IPIO2: [i32; 66] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163,
    0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C,
    0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292,
    0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B,
];

/// `π/2` cut into 24-bit chunks (`Σ PIO2[k]` to 192 bits).
const PIO2_CHUNKS: [f64; 8] = [
    1.57079625129699707031e+00, /* 0x3FF921FB, 0x40000000 */
    7.54978941586159635335e-08, /* 0x3E74442D, 0x00000000 */
    5.39030252995776476554e-15, /* 0x3CF84698, 0x80000000 */
    3.28200341580791294123e-22, /* 0x3B78CC51, 0x60000000 */
    1.27065575308067607349e-29, /* 0x39F01B83, 0x80000000 */
    1.22933308981111328932e-36, /* 0x387A2520, 0x40000000 */
    2.73370053816464559624e-44, /* 0x36E38222, 0x80000000 */
    2.16741683877804819444e-51, /* 0x3569F31D, 0x00000000 */
];

/// Payne–Hanek reduction (fdlibm `k_rem_pio2.c`, specialised to 53-bit precision, `jk = 4`):
/// `x[]` is the argument as up to three 24-bit integer-valued doubles with scaled exponent `e0`;
/// returns `n mod 8` and writes `y[0] + y[1] = x − n·π/2` with `|y| ≤ π/4`. It multiplies by only
/// the bits of 2/π that matter (the part of the product that is a multiple of 8 is skipped), so
/// the cost is independent of the exponent and the result is exact to 53+ bits for every `e0`.
fn rem_pio2_large(x: &[f64], y: &mut [f64; 2], e0: i32) -> i32 {
    const JK: usize = 4; // initial number of 2/π terms for 53-bit precision
    let x1p24 = f64::from_bits(0x4170_0000_0000_0000);
    let x1p_24 = f64::from_bits(0x3e70_0000_0000_0000);
    let nx = x.len();
    let mut f = [0.0f64; 20];
    let mut fq = [0.0f64; 20];
    let mut q = [0.0f64; 20];
    let mut iq = [0i32; 20];
    let jp = JK;

    // determine jx, jv, q0; note that 3 > q0
    let jx = nx - 1;
    let mut jv = (e0 - 3) / 24;
    if jv < 0 {
        jv = 0;
    }
    let mut q0 = e0 - 24 * (jv + 1);
    let jv = jv as usize;

    // set up f[0] to f[jx+jk] where f[jx+jk] = ipio2[jv+jk]
    let m = jx + JK;
    let first = jv as i64 - jx as i64;
    for (i, fi) in f.iter_mut().take(m + 1).enumerate() {
        let j = first + i as i64;
        *fi = if j < 0 { 0.0 } else { IPIO2[j as usize] as f64 };
    }

    // compute q[0], q[1], ..., q[jk]
    for i in 0..=JK {
        let mut fw = 0.0;
        for j in 0..=jx {
            fw += x[j] * f[jx + i - j];
        }
        q[i] = fw;
    }

    let mut jz = JK;
    let mut z: f64;
    let mut n: i32;
    let mut ih: i32;
    loop {
        // distill q[] into iq[] reversingly
        z = q[jz];
        for (i, j) in (1..=jz).rev().enumerate() {
            let fw = (x1p_24 * z) as i32 as f64;
            iq[i] = (z - x1p24 * fw) as i32;
            z = q[j - 1] + fw;
        }

        // compute n
        z = scalbn(z, q0); // actual value of z
        z -= 8.0 * (z * 0.125).floor(); // trim off integer >= 8
        n = z as i32;
        z -= n as f64;
        ih = 0;
        if q0 > 0 {
            // need iq[jz-1] to determine n
            let i = iq[jz - 1] >> (24 - q0);
            n += i;
            iq[jz - 1] -= i << (24 - q0);
            ih = iq[jz - 1] >> (23 - q0);
        } else if q0 == 0 {
            ih = iq[jz - 1] >> 23;
        } else if z >= 0.5 {
            ih = 2;
        }

        if ih > 0 {
            // q > 0.5
            n += 1;
            let mut carry = 0i32;
            for item in iq.iter_mut().take(jz) {
                // compute 1 − q
                let j = *item;
                if carry == 0 {
                    if j != 0 {
                        carry = 1;
                        *item = 0x100_0000 - j;
                    }
                } else {
                    *item = 0xff_ffff - j;
                }
            }
            if q0 > 0 {
                // rare case: chance is 1 in 12
                match q0 {
                    1 => iq[jz - 1] &= 0x7f_ffff,
                    2 => iq[jz - 1] &= 0x3f_ffff,
                    _ => {}
                }
            }
            if ih == 2 {
                z = 1.0 - z;
                if carry != 0 {
                    z -= scalbn(1.0, q0);
                }
            }
        }

        // check if recomputation is needed
        if z == 0.0 {
            let mut j = 0;
            for &v in iq.iter().take(jz).skip(JK) {
                j |= v;
            }
            if j == 0 {
                // need recomputation: k = number of terms needed
                let mut k = 1;
                while k <= JK && iq[JK - k] == 0 {
                    k += 1;
                }
                for i in (jz + 1)..=(jz + k) {
                    // add q[jz+1] to q[jz+k]
                    f[jx + i] = IPIO2[jv + i] as f64;
                    let mut fw = 0.0;
                    for j in 0..=jx {
                        fw += x[j] * f[jx + i - j];
                    }
                    q[i] = fw;
                }
                jz += k;
                continue;
            }
        }
        break;
    }

    // chop off zero terms
    if z == 0.0 {
        jz -= 1;
        q0 -= 24;
        while iq[jz] == 0 {
            jz -= 1;
            q0 -= 24;
        }
    } else {
        // break z into 24-bit if necessary
        z = scalbn(z, -q0);
        if z >= x1p24 {
            let fw = (x1p_24 * z) as i32 as f64;
            iq[jz] = (z - x1p24 * fw) as i32;
            jz += 1;
            q0 += 24;
            iq[jz] = fw as i32;
        } else {
            iq[jz] = z as i32;
        }
    }

    // convert integer "bit" chunks to floating-point values
    let mut fw = scalbn(1.0, q0);
    for i in (0..=jz).rev() {
        q[i] = fw * (iq[i] as f64);
        fw *= x1p_24;
    }

    // compute PIo2[0..jp] · q[jz..0]
    for i in (0..=jz).rev() {
        let mut fw = 0.0;
        let mut k = 0;
        while k <= jp && k <= jz - i {
            fw += PIO2_CHUNKS[k] * q[i + k];
            k += 1;
        }
        fq[jz - i] = fw;
    }

    // compress fq[] into y[] (53-bit precision: two doubles)
    let mut fw = 0.0;
    for i in (0..=jz).rev() {
        fw += fq[i];
    }
    y[0] = if ih == 0 { fw } else { -fw };
    fw = fq[0] - fw;
    for &v in fq.iter().take(jz + 1).skip(1) {
        fw += v;
    }
    y[1] = if ih == 0 { fw } else { -fw };
    n & 7
}

// --- sin / cos / tan ---------------------------------------------------------------------------

const S1: f64 = -1.66666666666666324348e-01; /* 0xBFC55555, 0x55555549 */
const S2: f64 = 8.33333333332248946124e-03; /* 0x3F811111, 0x1110F8A6 */
const S3: f64 = -1.98412698298579493134e-04; /* 0xBF2A01A0, 0x19C161D5 */
const S4: f64 = 2.75573137070700676789e-06; /* 0x3EC71DE3, 0x57B1FE7D */
const S5: f64 = -2.50507602534068634195e-08; /* 0xBE5AE5E6, 0x8A2B9CEB */
const S6: f64 = 1.58969099521155010221e-10; /* 0x3DE5D93A, 0x5ACFD57C */

/// `sin(x + y)` on `[−π/4, π/4]` with `y` the tail of `x` (fdlibm `k_sin.c`): the degree-13 odd
/// polynomial, with the `(1 − x²/2)·y` correction when `iy != 0`.
fn k_sin(x: f64, y: f64, iy: i32) -> f64 {
    let z = x * x;
    let w = z * z;
    let r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
    let v = z * x;
    if iy == 0 {
        x + v * (S1 + z * r)
    } else {
        x - ((z * (0.5 * y - v * r) - y) - v * S1)
    }
}

const C1: f64 = 4.16666666666666019037e-02; /* 0x3FA55555, 0x5555554C */
const C2: f64 = -1.38888888888741095749e-03; /* 0xBF56C16C, 0x16C15177 */
const C3: f64 = 2.48015872894767294178e-05; /* 0x3EFA01A0, 0x19CB1590 */
const C4: f64 = -2.75573143513906633035e-07; /* 0xBE927E4F, 0x809C52AD */
const C5: f64 = 2.08757232129817482790e-09; /* 0x3E21EE9E, 0xBDB4B1C4 */
const C6: f64 = -1.13596475577881948265e-11; /* 0xBDA8FAE9, 0xBE8838D4 */

/// `cos(x + y)` on `[−π/4, π/4]` (fdlibm `k_cos.c`): `1 − x²/2 + r` with the `x·y` correction,
/// reassociated as `w + (tmp + (r − x·y))` so the leading `1 − x²/2` rounds once.
fn k_cos(x: f64, y: f64) -> f64 {
    let z = x * x;
    let w = z * z;
    let r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    w + (((1.0 - w) - hz) + (z * r - x * y))
}

const TAN_T: [f64; 13] = [
    3.33333333333334091986e-01,  /* 3FD55555, 55555563 */
    1.33333333333201242699e-01,  /* 3FC11111, 1110FE7A */
    5.39682539762260521377e-02,  /* 3FABA1BA, 1BB341FE */
    2.18694882948595424599e-02,  /* 3F9664F4, 8406D637 */
    8.86323982359930005737e-03,  /* 3F8226E3, E96E8493 */
    3.59207910759131235356e-03,  /* 3F6D6D22, C9560328 */
    1.45620945432529025516e-03,  /* 3F57DBC8, FEE08315 */
    5.88041240820264096874e-04,  /* 3F4344D8, F2F26501 */
    2.46463134818469906812e-04,  /* 3F3026F7, 1A8D1068 */
    7.81794442939557092300e-05,  /* 3F147E88, A03792A6 */
    7.14072491382608190305e-05,  /* 3F12B80F, 32F0A7E9 */
    -1.85586374855275456654e-05, /* BEF375CB, DB605373 */
    2.59073051863633712884e-05,  /* 3EFB2A70, 74BF7AD4 */
];
const PIO4: f64 = 7.85398163397448278999e-01; /* 3FE921FB, 54442D18 */
const PIO4_LO: f64 = 3.06161699786838301793e-17; /* 3C81A626, 33145C07 */

/// `tan(x + y)` on `[−π/4, π/4]`, or `−1/tan` when `odd != 0` (fdlibm `k_tan.c`): the degree-27 odd
/// polynomial, with the `[0.6744, π/4]` band folded through `tan(π/4 − y)`.
fn k_tan(mut x: f64, mut y: f64, odd: i32) -> f64 {
    let hx = hi(x);
    let big = (hx & 0x7fff_ffff) >= 0x3fe5_9428; // |x| >= 0.6744
    if big {
        let sign = hx >> 31;
        if sign != 0 {
            x = -x;
            y = -y;
        }
        x = (PIO4 - x) + (PIO4_LO - y);
        y = 0.0;
    }
    let z = x * x;
    let w = z * z;
    // x^5·(T[1] + x²·T[2] + …) split into even and odd parts
    let r = TAN_T[1]
        + w * (TAN_T[3] + w * (TAN_T[5] + w * (TAN_T[7] + w * (TAN_T[9] + w * TAN_T[11]))));
    let v = z
        * (TAN_T[2]
            + w * (TAN_T[4] + w * (TAN_T[6] + w * (TAN_T[8] + w * (TAN_T[10] + w * TAN_T[12])))));
    let s = z * x;
    let r = y + z * (s * (r + v) + y) + s * TAN_T[0];
    let w = x + r;
    if big {
        let sign = hx >> 31;
        let s = 1.0 - 2.0 * odd as f64;
        let v = s - 2.0 * (x + (r - w * w / (w + s)));
        return if sign != 0 { -v } else { v };
    }
    if odd == 0 {
        return w;
    }
    // −1/(x + r) has up to 2 ULP error, so compute it accurately
    let w0 = with_lo(w, 0);
    let v = r - (w0 - x); // w0 + v = r + x
    let a = -1.0 / w;
    let a0 = with_lo(a, 0);
    a0 + a * (1.0 + a0 * w0 + a0 * v)
}

/// `sin x` (fdlibm `s_sin.c`): `x` itself below `2⁻²⁶`, the kernel on `|x| ≤ π/4`, otherwise
/// [`rem_pio2`] then the quadrant table. `sin(±0) = ±0`, `sin(±∞) = NaN`. < 1 ULP.
pub fn sin(x: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        // |x| ~<= π/4
        if ix < 0x3e50_0000 {
            // |x| < 2^-26
            return x;
        }
        return k_sin(x, 0.0, 0);
    }
    if ix >= 0x7ff0_0000 {
        // sin(inf or NaN) is NaN
        return domain_nan(x);
    }
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => k_sin(y0, y1, 1),
        1 => k_cos(y0, y1),
        2 => -k_sin(y0, y1, 1),
        _ => -k_cos(y0, y1),
    }
}

/// `cos x` (fdlibm `s_cos.c`): `1` below `2⁻²⁷·√2`, the kernel on `|x| ≤ π/4`, otherwise
/// [`rem_pio2`] then the quadrant table. `cos(±∞) = NaN`. < 1 ULP.
pub fn cos(x: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        // |x| ~<= π/4
        if ix < 0x3e46_a09e {
            // |x| < 2^-27·√2
            return 1.0;
        }
        return k_cos(x, 0.0);
    }
    if ix >= 0x7ff0_0000 {
        return domain_nan(x);
    }
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => k_cos(y0, y1),
        1 => -k_sin(y0, y1, 1),
        2 => -k_cos(y0, y1),
        _ => k_sin(y0, y1, 1),
    }
}

/// `tan x` (fdlibm `s_tan.c`): `x` itself below `2⁻²⁷`, the kernel on `|x| ≤ π/4`, otherwise
/// [`rem_pio2`] with the odd quadrants as `−1/tan`. `tan(±0) = ±0`, `tan(±∞) = NaN`. < 1 ULP.
pub fn tan(x: f64) -> f64 {
    let ix = hi(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        if ix < 0x3e40_0000 {
            // |x| < 2^-27
            return x;
        }
        return k_tan(x, 0.0, 0);
    }
    if ix >= 0x7ff0_0000 {
        return domain_nan(x);
    }
    let (n, y0, y1) = rem_pio2(x);
    k_tan(y0, y1, n & 1)
}

// --- asin / acos / atan / atan2 ----------------------------------------------------------------

const PIO2_HI: f64 = 1.57079632679489655800e+00; /* 0x3FF921FB, 0x54442D18 */
const PIO2_LO: f64 = 6.12323399573676603587e-17; /* 0x3C91A626, 0x33145C07 */
const P_S0: f64 = 1.66666666666666657415e-01; /* 0x3FC55555, 0x55555555 */
const P_S1: f64 = -3.25565818622400915405e-01; /* 0xBFD4D612, 0x03EB6F7D */
const P_S2: f64 = 2.01212532134862925881e-01; /* 0x3FC9C155, 0x0E884455 */
const P_S3: f64 = -4.00555345006794114027e-02; /* 0xBFA48228, 0xB5688F3B */
const P_S4: f64 = 7.91534994289814532176e-04; /* 0x3F49EFE0, 0x7501B288 */
const P_S5: f64 = 3.47933107596021167570e-05; /* 0x3F023DE1, 0x0DFDF709 */
const Q_S1: f64 = -2.40339491173441421878e+00; /* 0xC0033A27, 0x1C8A2D4B */
const Q_S2: f64 = 2.02094576023350569471e+00; /* 0x40002AE5, 0x9C598AC8 */
const Q_S3: f64 = -6.88283971605453293030e-01; /* 0xBFE6066C, 0x1B8D0159 */
const Q_S4: f64 = 7.70381505559019352791e-02; /* 0x3FB3B8C5, 0xB12E9282 */

/// The shared `asin`/`acos` rational `R(z) ≈ (asin(√z) − √z)/√z³`-style correction.
fn asin_r(z: f64) -> f64 {
    let p = z * (P_S0 + z * (P_S1 + z * (P_S2 + z * (P_S3 + z * (P_S4 + z * P_S5)))));
    let q = 1.0 + z * (Q_S1 + z * (Q_S2 + z * (Q_S3 + z * Q_S4)));
    p / q
}

/// `asin x` (fdlibm `e_asin.c`): `x + x·R(x²)` below ½; `π/2 − 2·(s + s·R(z))` with
/// `z = (1 − |x|)/2`, `s = √z` above, the `s` split into hi/lo for `|x| ≤ 0.975`. `asin(±1) = ±π/2`,
/// `|x| > 1` is NaN. < 1 ULP.
pub fn asin(x: f64) -> f64 {
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x3ff0_0000 {
        // |x| >= 1 or NaN
        let lx = lo(x);
        if ((ix - 0x3ff0_0000) | lx) == 0 {
            // asin(±1) = ±π/2 (inexact)
            return x * PIO2_HI + f64::from_bits(0x3870_0000_0000_0000);
        }
        return domain_nan(x);
    }
    if ix < 0x3fe0_0000 {
        // |x| < 0.5
        if ix >= 0x0010_0000 && ix < 0x3e50_0000 {
            // 2^-1022 <= |x| < 2^-26: asin(x) = x to the last bit
            return x;
        }
        return x + x * asin_r(x * x);
    }
    // 1 > |x| >= 0.5
    let z = (1.0 - x.abs()) * 0.5;
    let s = z.sqrt();
    let r = asin_r(z);
    let v = if ix >= 0x3fef_3333 {
        // |x| > 0.975
        PIO2_HI - (2.0 * (s + s * r) - PIO2_LO)
    } else {
        // f + c = sqrt(z)
        let f = with_lo(s, 0);
        let c = (z - f * f) / (s + f);
        0.5 * PIO2_HI - (2.0 * s * r - (PIO2_LO - 2.0 * c) - (0.5 * PIO2_HI - 2.0 * f))
    };
    if hx >> 31 != 0 {
        -v
    } else {
        v
    }
}

/// `acos x` (fdlibm `e_acos.c`): `π/2 − (x − (π/2_lo − x·R(x²)))` below ½, the `2·(s + w)` forms
/// with `z = (1 ∓ x)/2` above, `s` split into hi/lo. `acos(1) = 0`, `acos(−1) = π`, `|x| > 1` is
/// NaN. < 1 ULP.
pub fn acos(x: f64) -> f64 {
    let x1p_120 = f64::from_bits(0x3870_0000_0000_0000);
    let hx = hi(x);
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x3ff0_0000 {
        // |x| >= 1 or NaN
        let lx = lo(x);
        if ((ix - 0x3ff0_0000) | lx) == 0 {
            // acos(1) = 0, acos(−1) = π
            if (hx >> 31) != 0 {
                return 2.0 * PIO2_HI + x1p_120;
            }
            return 0.0;
        }
        return domain_nan(x);
    }
    if ix < 0x3fe0_0000 {
        // |x| < 0.5
        if ix <= 0x3c60_0000 {
            // |x| < 2^-57
            return PIO2_HI + x1p_120;
        }
        return PIO2_HI - (x - (PIO2_LO - x * asin_r(x * x)));
    }
    if (hx >> 31) != 0 {
        // x < −0.5
        let z = (1.0 + x) * 0.5;
        let s = z.sqrt();
        let w = asin_r(z) * s - PIO2_LO;
        return 2.0 * (PIO2_HI - (s + w));
    }
    // x > 0.5
    let z = (1.0 - x) * 0.5;
    let s = z.sqrt();
    let df = with_lo(s, 0);
    let c = (z - df * df) / (s + df);
    let w = asin_r(z) * s + c;
    2.0 * (df + w)
}

const ATANHI: [f64; 4] = [
    4.63647609000806093515e-01, /* atan(0.5)hi 0x3FDDAC67, 0x0561BB4F */
    7.85398163397448278999e-01, /* atan(1.0)hi 0x3FE921FB, 0x54442D18 */
    9.82793723247329054082e-01, /* atan(1.5)hi 0x3FEF730B, 0xD281F69B */
    1.57079632679489655800e+00, /* atan(inf)hi 0x3FF921FB, 0x54442D18 */
];
const ATANLO: [f64; 4] = [
    2.26987774529616870924e-17, /* atan(0.5)lo 0x3C7A2B7F, 0x222F65E2 */
    3.06161699786838301793e-17, /* atan(1.0)lo 0x3C81A626, 0x33145C07 */
    1.39033110312309984516e-17, /* atan(1.5)lo 0x3C700788, 0x7AF0CBBD */
    6.12323399573676603587e-17, /* atan(inf)lo 0x3C91A626, 0x33145C07 */
];
const AT: [f64; 11] = [
    3.33333333333329318027e-01,  /* 0x3FD55555, 0x5555550D */
    -1.99999999998764832476e-01, /* 0xBFC99999, 0x9998EBC4 */
    1.42857142725034663711e-01,  /* 0x3FC24924, 0x920083FF */
    -1.11111104054623557880e-01, /* 0xBFBC71C6, 0xFE231671 */
    9.09088713343650656196e-02,  /* 0x3FB745CD, 0xC54C206E */
    -7.69187620504482999495e-02, /* 0xBFB3B0F2, 0xAF749A6D */
    6.66107313738753120669e-02,  /* 0x3FB10D66, 0xA0D03D51 */
    -5.83357013379057348645e-02, /* 0xBFADDE2D, 0x52DEFD9A */
    4.97687799461593236017e-02,  /* 0x3FA97B4B, 0x24760DEB */
    -3.65315727442169155270e-02, /* 0xBFA2B444, 0x2C6A6C2F */
    1.62858201153657823623e-02,  /* 0x3F90AD3A, 0xE322DA11 */
];

/// `atan x` (fdlibm `s_atan.c`): reduce `|x|` to `[0, 7/16)` through one of the four breakpoints
/// `0.5`, `1`, `1.5`, `∞` (whose arctangents are tabulated as hi + lo), then the degree-23 odd
/// polynomial split into even/odd halves. `atan(±∞) = ±π/2`. < 1 ULP.
pub fn atan(x: f64) -> f64 {
    let mut x = x;
    let mut ix = hi(x);
    let sign = ix >> 31;
    ix &= 0x7fff_ffff;
    if ix >= 0x4410_0000 {
        // |x| >= 2^66
        if x.is_nan() {
            return x;
        }
        let z = ATANHI[3] + f64::from_bits(0x3870_0000_0000_0000);
        return if sign != 0 { -z } else { z };
    }
    let id: i32 = if ix < 0x3fdc_0000 {
        // |x| < 0.4375
        if ix < 0x3e40_0000 {
            // |x| < 2^-27
            return x;
        }
        -1
    } else {
        x = x.abs();
        if ix < 0x3ff3_0000 {
            // |x| < 1.1875
            if ix < 0x3fe6_0000 {
                // 7/16 <= |x| < 11/16
                x = (2.0 * x - 1.0) / (2.0 + x);
                0
            } else {
                // 11/16 <= |x| < 19/16
                x = (x - 1.0) / (x + 1.0);
                1
            }
        } else if ix < 0x4003_8000 {
            // |x| < 2.4375
            x = (x - 1.5) / (1.0 + 1.5 * x);
            2
        } else {
            // 2.4375 <= |x| < 2^66
            x = -1.0 / x;
            3
        }
    };
    let z = x * x;
    let w = z * z;
    // break Σ AT[i]·z^(i+1) into odd and even polynomials
    let s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))));
    let s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))));
    if id < 0 {
        return x - x * (s1 + s2);
    }
    let id = id as usize;
    let z = ATANHI[id] - (x * (s1 + s2) - ATANLO[id] - x);
    if sign != 0 {
        -z
    } else {
        z
    }
}

const PI: f64 = 3.1415926535897931160E+00; /* 0x400921FB, 0x54442D18 */
const PI_LO: f64 = 1.2246467991473531772E-16; /* 0x3CA1A626, 0x33145C07 */

/// `atan2(y, x)` (fdlibm `e_atan2.c`): [`atan`] of `|y/x|` corrected into the right quadrant with
/// `π` as hi + lo, plus the full axis/infinity table (`atan2(±0, x<0) = ±π`, `atan2(±∞, ±∞) =
/// ±π/4` or `±3π/4`, …). < 1 ULP.
pub fn atan2(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    let mut ix = hi(x);
    let lx = lo(x);
    let mut iy = hi(y);
    let ly = lo(y);
    if ((ix.wrapping_sub(0x3ff0_0000)) | lx) == 0 {
        // x = 1.0
        return atan(y);
    }
    let m = ((iy >> 31) & 1) | ((ix >> 30) & 2); // 2·sign(x) + sign(y)
    ix &= 0x7fff_ffff;
    iy &= 0x7fff_ffff;
    // y = 0
    if (iy | ly) == 0 {
        return match m {
            0 | 1 => y,
            2 => PI,
            _ => -PI,
        };
    }
    // x = 0
    if (ix | lx) == 0 {
        return if m & 1 != 0 { -PI / 2.0 } else { PI / 2.0 };
    }
    // x is ±inf
    if ix == 0x7ff0_0000 {
        if iy == 0x7ff0_0000 {
            return match m {
                0 => PI / 4.0,
                1 => -PI / 4.0,
                2 => 3.0 * PI / 4.0,
                _ => -3.0 * PI / 4.0,
            };
        } else {
            return match m {
                0 => 0.0,
                1 => -0.0,
                2 => PI,
                _ => -PI,
            };
        }
    }
    // |y/x| > 2^64
    if ix.wrapping_add(64 << 20) < iy || iy == 0x7ff0_0000 {
        return if m & 1 != 0 { -PI / 2.0 } else { PI / 2.0 };
    }
    // z = atan(|y/x|) without spurious underflow
    let z = if (m & 2 != 0) && iy.wrapping_add(64 << 20) < ix {
        // |y/x| < 2^-64, x < 0
        0.0
    } else {
        atan((y / x).abs())
    };
    match m {
        0 => z,
        1 => -z,
        2 => PI - (z - PI_LO),
        _ => (z - PI_LO) - PI,
    }
}

// --- expm1 / log1p -----------------------------------------------------------------------------

const EXPM1_O_THRESHOLD: f64 = 7.09782712893383973096e+02; /* 0x40862E42, 0xFEFA39EF */
const EXPM1_Q1: f64 = -3.33333333333331316428e-02; /* BFA11111 111110F4 */
const EXPM1_Q2: f64 = 1.58730158725481460165e-03; /* 3F5A01A0 19FE5585 */
const EXPM1_Q3: f64 = -7.93650757867487942473e-05; /* BF14CE19 9EAADBB7 */
const EXPM1_Q4: f64 = 4.00821782732936239552e-06; /* 3ED0CFCA 86E65239 */
const EXPM1_Q5: f64 = -2.01099218183624371326e-07; /* BE8AFDB7 6E09C32D */

/// `eˣ − 1` (fdlibm `s_expm1.c`), accurate even for tiny `x` where `exp(x) − 1` would cancel:
/// `k·ln2 + r` reduction with the correction term `c`, the degree-5 Remez rational, then the
/// `k`-dependent reconstructions. `expm1(−∞) = −1`. < 1 ULP.
pub fn expm1(mut x: f64) -> f64 {
    let ui = x.to_bits();
    let hx = ((ui >> 32) & 0x7fff_ffff) as u32;
    let sign = (ui >> 63) as i32;
    // filter out huge and non-finite arguments
    if hx >= 0x4043_687a {
        // |x| >= 56·ln2
        if x.is_nan() {
            return x;
        }
        if sign != 0 {
            return -1.0;
        }
        if x > EXPM1_O_THRESHOLD {
            return x * f64::from_bits(0x7fe0_0000_0000_0000);
        }
    }
    // argument reduction
    let c: f64;
    let k: i32;
    if hx > 0x3fd6_2e42 {
        // |x| > 0.5 ln2
        let hi_part: f64;
        let lo_part: f64;
        if hx < 0x3ff0_a2b2 {
            // and |x| < 1.5 ln2
            if sign == 0 {
                hi_part = x - LN2HI;
                lo_part = LN2LO;
                k = 1;
            } else {
                hi_part = x + LN2HI;
                lo_part = -LN2LO;
                k = -1;
            }
        } else {
            k = (INVLN2 * x + if sign != 0 { -0.5 } else { 0.5 }) as i32;
            let t = k as f64;
            hi_part = x - t * LN2HI; // t·ln2hi is exact here
            lo_part = t * LN2LO;
        }
        x = hi_part - lo_part;
        c = (hi_part - x) - lo_part;
    } else if hx < 0x3c90_0000 {
        // |x| < 2^-54: expm1(x) = x
        return x;
    } else {
        c = 0.0;
        k = 0;
    }
    // x is now in the primary range
    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let r1 = 1.0
        + hxs
            * (EXPM1_Q1 + hxs * (EXPM1_Q2 + hxs * (EXPM1_Q3 + hxs * (EXPM1_Q4 + hxs * EXPM1_Q5))));
    let t = 3.0 - r1 * hfx;
    let mut e = hxs * ((r1 - t) / (6.0 - x * t));
    if k == 0 {
        // c is 0
        return x - (x * e - hxs);
    }
    e = x * (e - c) - c;
    e -= hxs;
    // exp(x) ≈ 2^k·(x_reduced − e + 1)
    if k == -1 {
        return 0.5 * (x - e) - 0.5;
    }
    if k == 1 {
        if x < -0.25 {
            return -2.0 * (e - (x + 0.5));
        }
        return 1.0 + 2.0 * (x - e);
    }
    let twopk = f64::from_bits(((0x3ff + k) as u64) << 52); // 2^k
    if !(0..=56).contains(&k) {
        // suffices to return exp(x) − 1
        let mut y = x - e + 1.0;
        if k == 1024 {
            y = y * 2.0 * f64::from_bits(0x7fe0_0000_0000_0000);
        } else {
            y *= twopk;
        }
        return y - 1.0;
    }
    let uf = f64::from_bits(((0x3ff - k) as u64) << 52); // 2^-k
    if k < 20 {
        (x - e + (1.0 - uf)) * twopk
    } else {
        (x - (e + uf) + 1.0) * twopk
    }
}

/// `ln(1 + x)` (fdlibm `s_log1p.c`), accurate for tiny `x`: the log core on `u = 1 + x` with the
/// correction `c ≈ log(1+x) − log(u)` folded into the `k·ln2_lo` term. `log1p(−1) = −∞`,
/// `x < −1` is NaN. < 1 ULP.
pub fn log1p(x: f64) -> f64 {
    let mut ui = x.to_bits();
    let hx = (ui >> 32) as u32;
    let mut k: i32 = 1;
    let mut f: f64 = 0.0;
    let mut c: f64 = 0.0;
    if hx < 0x3fda_827a || (hx >> 31) > 0 {
        // 1 + x < sqrt(2)+
        if hx >= 0xbff0_0000 {
            // x <= −1.0
            if x == -1.0 {
                return f64::NEG_INFINITY;
            }
            return domain_nan(x);
        }
        if hx << 1 < 0x3ca0_0000 << 1 {
            // |x| < 2^-53
            return x;
        }
        if hx <= 0xbfd2_bec4 {
            // sqrt(2)/2− <= 1 + x < sqrt(2)+
            k = 0;
            c = 0.0;
            f = x;
        }
    } else if hx >= 0x7ff0_0000 {
        return x;
    }
    if k > 0 {
        ui = (1.0 + x).to_bits();
        let mut hu = (ui >> 32) as u32;
        hu += 0x3ff0_0000 - 0x3fe6_a09e;
        k = (hu >> 20) as i32 - 0x3ff;
        // correction term ~ log(1+x) − log(u), avoiding underflow in c/u
        if k < 54 {
            c = if k >= 2 {
                1.0 - (f64::from_bits(ui) - x)
            } else {
                x - (f64::from_bits(ui) - 1.0)
            };
            c /= f64::from_bits(ui);
        } else {
            c = 0.0;
        }
        // reduce u into [sqrt(2)/2, sqrt(2)]
        hu = (hu & 0x000f_ffff) + 0x3fe6_a09e;
        ui = ((hu as u64) << 32) | (ui & 0xffff_ffff);
        f = f64::from_bits(ui) - 1.0;
    }
    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    let dk = k as f64;
    s * (hfsq + r) + (dk * LN2LO + c) - hfsq + f + dk * LN2HI
}

// --- hyperbolics -------------------------------------------------------------------------------

/// `exp(|x|)/2` for `|x| > ln(DBL_MAX)` without overflowing the intermediate: `exp(x − k·ln2)·2^(k−1)`
/// with `k = 2043` odd and the scale applied as `scale·scale` (fdlibm `__expo2`).
fn expo2(x: f64) -> f64 {
    const K: i32 = 2043;
    let kln2 = f64::from_bits(0x4096_2066_151a_dd8b);
    let scale = from_words(((0x3ff + K / 2) as u32) << 20, 0);
    exp(x - kln2) * scale * scale
}

/// `sinh x` (musl `sinh.c`): from `t = expm1(|x|)` — `½·(2t − t²/(t+1))` below 1, `½·(t + t/(t+1))`
/// up to `ln(DBL_MAX)`, [`expo2`] beyond; `x` itself below `2⁻²⁶`. `sinh(±∞) = ±∞`. ≤ 2 ULP.
pub fn sinh(x: f64) -> f64 {
    let ui = x.to_bits();
    let h = if ui >> 63 != 0 { -0.5 } else { 0.5 };
    let absx = f64::from_bits(ui & 0x7fff_ffff_ffff_ffff);
    let w = (absx.to_bits() >> 32) as u32;
    if w < 0x4086_2e42 {
        // |x| < log(DBL_MAX)
        let t = expm1(absx);
        if w < 0x3ff0_0000 {
            if w < 0x3ff0_0000 - (26 << 20) {
                return x;
            }
            return h * (2.0 * t - t * t / (t + 1.0));
        }
        return h * (t + t / (t + 1.0));
    }
    // |x| > log(DBL_MAX) or NaN
    2.0 * h * expo2(absx)
}

/// `cosh x` (musl `cosh.c`): `1 + t²/(2(1+t))` from `t = expm1(|x|)` below `ln2`, `½(eˣ + 1/eˣ)` up
/// to `ln(DBL_MAX)`, [`expo2`] beyond; exactly `1` below `2⁻²⁶`. `cosh(±∞) = +∞`. ≤ 2 ULP.
pub fn cosh(x: f64) -> f64 {
    let ix = x.to_bits() & 0x7fff_ffff_ffff_ffff;
    let x = f64::from_bits(ix);
    let w = (ix >> 32) as u32;
    if w < 0x3fe6_2e42 {
        // |x| < log(2)
        if w < 0x3ff0_0000 - (26 << 20) {
            return 1.0;
        }
        let t = expm1(x);
        return 1.0 + t * t / (2.0 * (1.0 + t));
    }
    if w < 0x4086_2e42 {
        // |x| < log(DBL_MAX)
        let t = exp(x);
        return 0.5 * (t + 1.0 / t);
    }
    expo2(x)
}

/// `tanh x` (musl `tanh.c`): `1 − 2/(expm1(2|x|) + 2)` above `ln3/2`, `t/(t+2)` on
/// `(ln(5/3)/2, ln3/2]`, `−t/(t+2)` with `t = expm1(−2|x|)` below (≤ 2 ULP in `[0.1, 0.2554]`),
/// `x` itself for subnormals; exactly `±1` above `|x| = 20`. `tanh(±0) = ±0` and the sign is always
/// restored from the argument, so a negative argument never yields `+0`.
pub fn tanh(x: f64) -> f64 {
    let ui = x.to_bits();
    let sign = ui >> 63 != 0;
    let ax = f64::from_bits(ui & 0x7fff_ffff_ffff_ffff);
    let w = (ax.to_bits() >> 32) as u32;
    let t = if w > 0x3fe1_93ea {
        // |x| > log(3)/2 ~= 0.5493 or NaN
        if w > 0x4034_0000 {
            // |x| > 20 or NaN: saturate (NaN propagates through the division)
            1.0 - 0.0 / ax
        } else {
            let t = expm1(2.0 * ax);
            1.0 - 2.0 / (t + 2.0)
        }
    } else if w > 0x3fd0_58ae {
        // |x| > log(5/3)/2 ~= 0.2554
        let t = expm1(2.0 * ax);
        t / (t + 2.0)
    } else if w >= 0x0010_0000 {
        // |x| >= 2^-1022
        let t = expm1(-2.0 * ax);
        -t / (t + 2.0)
    } else {
        // subnormal or zero: tanh(x) = x
        ax
    };
    if sign {
        -t
    } else {
        t
    }
}

const LN2: f64 = 0.693147180559945309417232121458176568; /* 0x3fe62e42, 0xfefa39ef */

/// `asinh x` (musl `asinh.c`): `log(|x|) + ln2` above `2²⁶`, `log(2|x| + 1/(√(x²+1) + |x|))` above 2,
/// `log1p(|x| + x²/(√(x²+1) + 1))` above `2⁻²⁶` (≤ 1.6 ULP in `[0.125, 0.5]`), `x` itself below.
/// `asinh(±∞) = ±∞`, `asinh(±0) = ±0`. ≤ 2 ULP.
pub fn asinh(x: f64) -> f64 {
    let u = x.to_bits();
    let e = ((u >> 52) as usize) & 0x7ff;
    let sign = (u >> 63) != 0;
    let mut ax = f64::from_bits(u & 0x7fff_ffff_ffff_ffff);
    if e >= 0x3ff + 26 {
        // |x| >= 2^26 or inf or NaN
        ax = log(ax) + LN2;
    } else if e > 0x3ff {
        // |x| >= 2
        ax = log(2.0 * ax + 1.0 / ((ax * ax + 1.0).sqrt() + ax));
    } else if e >= 0x3ff - 26 {
        // |x| >= 2^-26
        ax = log1p(ax + ax * ax / ((ax * ax + 1.0).sqrt() + 1.0));
    }
    if sign {
        -ax
    } else {
        ax
    }
}

/// `acosh x` (musl `acosh.c`): `log1p(x − 1 + √((x−1)² + 2(x−1)))` below 2 (≤ 2 ULP in
/// `[1, 1.125]`), `log(2x − 1/(x + √(x²−1)))` below `2²⁶`, `log(x) + ln2` above. `acosh(1) = 0`,
/// `x < 1` is NaN. ≤ 2 ULP.
pub fn acosh(x: f64) -> f64 {
    let u = x.to_bits();
    let e = ((u >> 52) as usize) & 0x7ff;
    if e < 0x3ff + 1 {
        // |x| < 2 (x < 1 reaches log1p's domain error)
        return log1p(x - 1.0 + ((x - 1.0) * (x - 1.0) + 2.0 * (x - 1.0)).sqrt());
    }
    if e < 0x3ff + 26 {
        // |x| < 2^26
        return log(2.0 * x - 1.0 / (x + (x * x - 1.0).sqrt()));
    }
    // |x| >= 2^26 or NaN
    log(x) + LN2
}

/// `atanh x` (musl `atanh.c`): `½·log1p(2|x| + 2x²/(1 − |x|))` below ½ (≤ 1.7 ULP),
/// `½·log1p(2|x|/(1 − |x|))` above, `x` itself below `2⁻³²`. `atanh(±1) = ±∞`, `|x| > 1` is NaN.
/// ≤ 2 ULP.
pub fn atanh(x: f64) -> f64 {
    let u = x.to_bits();
    let e = ((u >> 52) as usize) & 0x7ff;
    let sign = (u >> 63) != 0;
    let mut y = f64::from_bits(u & 0x7fff_ffff_ffff_ffff);
    if e < 0x3ff - 1 {
        if e >= 0x3ff - 32 {
            // |x| < 0.5
            y = 0.5 * log1p(2.0 * y + 2.0 * y * y / (1.0 - y));
        }
        // else |x| < 2^-32: atanh(x) = x
    } else {
        // avoid overflow
        y = 0.5 * log1p(2.0 * (y / (1.0 - y)));
    }
    if sign {
        -y
    } else {
        y
    }
}

// --- erf ---------------------------------------------------------------------------------------

const ERX: f64 = 8.45062911510467529297e-01; /* 0x3FEB0AC1, 0x60000000 */
const EFX8: f64 = 1.02703333676410069053e+00; /* 0x3FF06EBA, 0x8214DB69 */
const PP0: f64 = 1.28379167095512558561e-01; /* 0x3FC06EBA, 0x8214DB68 */
const PP1: f64 = -3.25042107247001499370e-01; /* 0xBFD4CD7D, 0x691CB913 */
const PP2: f64 = -2.84817495755985104766e-02; /* 0xBF9D2A51, 0xDBD7194F */
const PP3: f64 = -5.77027029648944159157e-03; /* 0xBF77A291, 0x236668E4 */
const PP4: f64 = -2.37630166566501626084e-05; /* 0xBEF8EAD6, 0x120016AC */
const QQ1: f64 = 3.97917223959155352819e-01; /* 0x3FD97779, 0xCDDADC09 */
const QQ2: f64 = 6.50222499887672944485e-02; /* 0x3FB0A54C, 0x5536CEBA */
const QQ3: f64 = 5.08130628187576562776e-03; /* 0x3F74D022, 0xC4D36B0F */
const QQ4: f64 = 1.32494738004321644526e-04; /* 0x3F215DC9, 0x221C1A10 */
const QQ5: f64 = -3.96022827877536812320e-06; /* 0xBED09C43, 0x42A26120 */
const PA0: f64 = -2.36211856075265944077e-03; /* 0xBF6359B8, 0xBEF77538 */
const PA1: f64 = 4.14856118683748331666e-01; /* 0x3FDA8D00, 0xAD92B34D */
const PA2: f64 = -3.72207876035701323847e-01; /* 0xBFD7D240, 0xFBB8C3F1 */
const PA3: f64 = 3.18346619901161753674e-01; /* 0x3FD45FCA, 0x805120E4 */
const PA4: f64 = -1.10894694282396677476e-01; /* 0xBFBC6398, 0x3D3E28EC */
const PA5: f64 = 3.54783043256182359371e-02; /* 0x3FA22A36, 0x599795EB */
const PA6: f64 = -2.16637559486879084300e-03; /* 0xBF61BF38, 0x0A96073F */
const QA1: f64 = 1.06420880400844228286e-01; /* 0x3FBB3E66, 0x18EEE323 */
const QA2: f64 = 5.40397917702171048937e-01; /* 0x3FE14AF0, 0x92EB6F33 */
const QA3: f64 = 7.18286544141962662868e-02; /* 0x3FB2635C, 0xD99FE9A7 */
const QA4: f64 = 1.26171219808761642112e-01; /* 0x3FC02660, 0xE763351F */
const QA5: f64 = 1.36370839120290507362e-02; /* 0x3F8BEDC2, 0x6B51DD1C */
const QA6: f64 = 1.19844998467991074170e-02; /* 0x3F888B54, 0x5735151D */
const RA0: f64 = -9.86494403484714822705e-03; /* 0xBF843412, 0x600D6435 */
const RA1: f64 = -6.93858572707181764372e-01; /* 0xBFE63416, 0xE4BA7360 */
const RA2: f64 = -1.05586262253232909814e+01; /* 0xC0251E04, 0x41B0E726 */
const RA3: f64 = -6.23753324503260060396e+01; /* 0xC04F300A, 0xE4CBA38D */
const RA4: f64 = -1.62396669462573470355e+02; /* 0xC0644CB1, 0x84282266 */
const RA5: f64 = -1.84605092906711035994e+02; /* 0xC067135C, 0xEBCCABB2 */
const RA6: f64 = -8.12874355063065934246e+01; /* 0xC0545265, 0x57E4D2F2 */
const RA7: f64 = -9.81432934416914548592e+00; /* 0xC023A0EF, 0xC69AC25C */
const SA1: f64 = 1.96512716674392571292e+01; /* 0x4033A6B9, 0xBD707687 */
const SA2: f64 = 1.37657754143519042600e+02; /* 0x4061350C, 0x526AE721 */
const SA3: f64 = 4.34565877475229228821e+02; /* 0x407B290D, 0xD58A1A71 */
const SA4: f64 = 6.45387271733267880336e+02; /* 0x40842B19, 0x21EC2868 */
const SA5: f64 = 4.29008140027567833386e+02; /* 0x407AD021, 0x57700314 */
const SA6: f64 = 1.08635005541779435134e+02; /* 0x405B28A3, 0xEE48AE2C */
const SA7: f64 = 6.57024977031928170135e+00; /* 0x401A47EF, 0x8E484A93 */
const SA8: f64 = -6.04244152148580987438e-02; /* 0xBFAEEFF2, 0xEE749A62 */
const RB0: f64 = -9.86494292470009928597e-03; /* 0xBF843412, 0x39E86F4A */
const RB1: f64 = -7.99283237680523006574e-01; /* 0xBFE993BA, 0x70C285DE */
const RB2: f64 = -1.77579549177547519889e+01; /* 0xC031C209, 0x555F995A */
const RB3: f64 = -1.60636384855821916062e+02; /* 0xC064145D, 0x43C5ED98 */
const RB4: f64 = -6.37566443368389627722e+02; /* 0xC083EC88, 0x1375F228 */
const RB5: f64 = -1.02509513161107724954e+03; /* 0xC0900461, 0x6A2E5992 */
const RB6: f64 = -4.83519191608651397019e+02; /* 0xC07E384E, 0x9BDC383F */
const SB1: f64 = 3.03380607434824582924e+01; /* 0x403E568B, 0x261D5190 */
const SB2: f64 = 3.25792512996573918826e+02; /* 0x40745CAE, 0x221B9F0A */
const SB3: f64 = 1.53672958608443695994e+03; /* 0x409802EB, 0x189D5118 */
const SB4: f64 = 3.19985821950859553908e+03; /* 0x40A8FFB7, 0x688C246A */
const SB5: f64 = 2.55305040643316442583e+03; /* 0x40A3F219, 0xCEDF3BE6 */
const SB6: f64 = 4.74528541206955367215e+02; /* 0x407DA874, 0xE79FE763 */
const SB7: f64 = -2.24409524465858183362e+01; /* 0xC03670E2, 0x42712D62 */

/// `erfc(x)` on `[0.84375, 1.25)`: `1 − erx − P(s)/Q(s)`, `s = |x| − 1`.
fn erfc1(x: f64) -> f64 {
    let s = x.abs() - 1.0;
    let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
    let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
    1.0 - ERX - p / q
}

/// `erfc(x)` on `[0.84375, 28)`: the asymptotic `exp(−x² − 0.5625)·exp((z−x)(z+x) + R/S)/x` with
/// `z` the top 32 bits of `x`, two rational fits split at `1/0.35`.
fn erfc2(ix: u32, x: f64) -> f64 {
    if ix < 0x3ff4_0000 {
        // |x| < 1.25
        return erfc1(x);
    }
    let x = x.abs();
    let s = 1.0 / (x * x);
    let (r, big_s) = if ix < 0x4006_db6d {
        // |x| < 1/0.35 ~ 2.85714
        (
            RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7)))))),
            1.0 + s
                * (SA1
                    + s * (SA2
                        + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8))))))),
        )
    } else {
        // |x| > 1/0.35
        (
            RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6))))),
            1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7)))))),
        )
    };
    let z = with_lo(x, 0);
    exp(-z * z - 0.5625) * exp((z - x) * (z + x) + r / big_s) / x
}

/// `erf x` (fdlibm `s_erf.c`): `x + x·P(x²)/Q(x²)` below `0.84375`, `1 − erfc` through the two
/// rational fits and the `exp` tail up to 6, exactly `±1` beyond. `erf(±0) = ±0`,
/// `erf(±∞) = ±1`; below `2⁻²⁸` it is `x·(1 + 2/√π − 1)`-exact. < 1 ULP.
pub fn erf(x: f64) -> f64 {
    let mut ix = hi(x);
    let sign = (ix >> 31) as usize;
    ix &= 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        // erf(NaN) = NaN, erf(±inf) = ±1
        return 1.0 - 2.0 * (sign as f64) + 1.0 / x;
    }
    if ix < 0x3feb_0000 {
        // |x| < 0.84375
        if ix < 0x3e30_0000 {
            // |x| < 2^-28: avoid underflow
            return 0.125 * (8.0 * x + EFX8 * x);
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        let y = r / s;
        return x + x * y;
    }
    let y = if ix < 0x4018_0000 {
        // 0.84375 <= |x| < 6
        1.0 - erfc2(ix, x)
    } else {
        1.0 - f64::from_bits(0x0010_0000_0000_0000)
    };
    if sign != 0 {
        -y
    } else {
        y
    }
}

/// `erfc x = 1 − erf x` (fdlibm `s_erf.c`, musl `erfc`), computed without the cancellation of
/// `1 − erf x`: below `0.84375` from the same `P/Q` fit as [`erf`], from `1/4` up rearranged as
/// `0.5 − (x − 0.5 + x·y)`; on `[0.84375, 28)` the two rational tails of `erfc2`, whose
/// `exp(−z²−0.5625)·exp((z−x)(z+x) + R/S)/x` keeps full relative accuracy deep into the tail; `+0`
/// from 28 up (`2` toward `−∞`). `erfc(NaN) = NaN`, `erfc(+∞) = +0`, `erfc(−∞) = 2`. ≤ 3 ULP
/// measured against mpmath (`special::tests::special_references_within_bound` pins sample points);
/// relative, not absolute, accuracy in the tail, which is the point of the function.
pub fn erfc(x: f64) -> f64 {
    let mut ix = hi(x);
    let sign = ix >> 31;
    ix &= 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        // erfc(NaN) = NaN, erfc(+inf) = 0, erfc(-inf) = 2
        return 2.0 * (sign as f64) + 1.0 / x;
    }
    if ix < 0x3feb_0000 {
        // |x| < 0.84375
        if ix < 0x3c70_0000 {
            // |x| < 2^-56
            return 1.0 - x;
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        let y = r / s;
        if sign != 0 || ix < 0x3fd0_0000 {
            // x < 1/4
            return 1.0 - (x + x * y);
        }
        return 0.5 - (x - 0.5 + x * y);
    }
    if ix < 0x403c_0000 {
        // 0.84375 <= |x| < 28
        if sign != 0 {
            return 2.0 - erfc2(ix, x);
        }
        return erfc2(ix, x);
    }
    if sign != 0 {
        2.0 - f64::from_bits(0x0010_0000_0000_0000)
    } else {
        0.0
    }
}

// --- cbrt --------------------------------------------------------------------------------------

const CBRT_B1: u32 = 715094163; /* B1 = (1023 − 1023/3 − 0.03306235651)·2^20 */
const CBRT_B2: u32 = 696219795; /* B2 = (1023 − 1023/3 − 54/3 − 0.03306235651)·2^20 */
/* |1/cbrt(x) − p(x)| < 2^-23.5 (~[-7.93e-8, 7.929e-8]) */
const CBRT_P0: f64 = 1.87595182427177009643; /* 0x3ffe03e6, 0x0f61e692 */
const CBRT_P1: f64 = -1.88497979543377169875; /* 0xbffe28e0, 0x92f02420 */
const CBRT_P2: f64 = 1.621429720105354466140; /* 0x3ff9f160, 0x4a49d6c2 */
const CBRT_P3: f64 = -0.758397934778766047437; /* 0xbfe844cb, 0xbee751d9 */
const CBRT_P4: f64 = 0.145996192886612446982; /* 0x3fc2b000, 0xd4e4edd7 */

/// `∛x` (fdlibm/musl `s_cbrt.c`): a 5-bit estimate from integer division of the exponent field,
/// the degree-4 polynomial in `t³/x` to 23 bits, rounding `t` away from zero to 23 bits, then one
/// Newton step with exact `t²` and `t+t`. Error < 0.667 ULP. `cbrt(±0) = ±0`, `cbrt(±∞) = ±∞`;
/// odd, so the sign of the argument is the sign of the result.
pub fn cbrt(x: f64) -> f64 {
    let x1p54 = f64::from_bits(0x4350_0000_0000_0000);
    let mut ui = x.to_bits();
    let mut hx = (ui >> 32) as u32 & 0x7fff_ffff;
    if hx >= 0x7ff0_0000 {
        // cbrt(NaN, ±inf) is itself
        return x + x;
    }
    // Rough cbrt to 5 bits: cbrt(2^e·(1+m)) ≈ 2^(e/3)·(1 + (e%3 + m)/3), computed by integer
    // division of the biased exponent field with the bias folded into B1/B2.
    if hx < 0x0010_0000 {
        // zero or subnormal
        ui = (x * x1p54).to_bits();
        hx = (ui >> 32) as u32 & 0x7fff_ffff;
        if hx == 0 {
            return x; // cbrt(±0) is itself
        }
        hx = hx / 3 + CBRT_B2;
    } else {
        hx = hx / 3 + CBRT_B1;
    }
    ui &= 1u64 << 63;
    ui |= (hx as u64) << 32;
    let mut t = f64::from_bits(ui);
    // New cbrt to 23 bits: cbrt(x) = t·cbrt(x/t³) ≈ t·P(t³/x).
    let r = (t * t) * (t / x);
    t = t * ((CBRT_P0 + r * (CBRT_P1 + r * CBRT_P2)) + ((r * r) * r) * (CBRT_P3 + r * CBRT_P4));
    // Round t away from zero to 23 bits.
    ui = t.to_bits();
    ui = (ui + 0x8000_0000) & 0xffff_ffff_c000_0000;
    t = f64::from_bits(ui);
    // One Newton step to 53 bits with error < 0.667 ULP.
    let s = t * t; // t·t is exact
    let r = x / s; // error <= 0.5 ULP; |r| < |t|
    let w = t + t; // t + t is exact
    let r = (r - t) / (w + r); // r − t is exact; w + r ≈ 3t
    t + t * r
}

// --- exp2 / exp10 ------------------------------------------------------------------------------

/// `exp2t[i] = 2^(i/256 + eps[i])` for `i − 128`, with `eps[i]` chosen so `exp2t[i]` is accurate to
/// `2⁻⁶⁴` (FreeBSD `s_exp2.c`): pairs of `(exp2t bits, eps bits)`.
#[rustfmt::skip]
const EXP2_TBL: [u64; 512] = [
    //  exp2(z + eps)          eps
    0x3fe6a09e667f3d5d, 0x3d39880000000000,
    0x3fe6b052fa751744, 0x3cd8000000000000,
    0x3fe6c012750bd9fe, 0xbd28780000000000,
    0x3fe6cfdcddd476bf, 0x3d1ec00000000000,
    0x3fe6dfb23c651a29, 0xbcd8000000000000,
    0x3fe6ef9298593ae3, 0xbcbc000000000000,
    0x3fe6ff7df9519386, 0xbd2fd80000000000,
    0x3fe70f7466f42da3, 0xbd2c880000000000,
    0x3fe71f75e8ec5fc3, 0x3d13c00000000000,
    0x3fe72f8286eacf05, 0xbd38300000000000,
    0x3fe73f9a48a58152, 0xbd00c00000000000,
    0x3fe74fbd35d7ccfc, 0x3d2f880000000000,
    0x3fe75feb564267f1, 0x3d03e00000000000,
    0x3fe77024b1ab6d48, 0xbd27d00000000000,
    0x3fe780694fde5d38, 0xbcdd000000000000,
    0x3fe790b938ac1d00, 0x3ce3000000000000,
    0x3fe7a11473eb0178, 0xbced000000000000,
    0x3fe7b17b0976d060, 0x3d20400000000000,
    0x3fe7c1ed0130c133, 0x3ca0000000000000,
    0x3fe7d26a62ff8636, 0xbd26900000000000,
    0x3fe7e2f336cf4e3b, 0xbd02e00000000000,
    0x3fe7f3878491c3e8, 0xbd24580000000000,
    0x3fe80427543e1b4e, 0x3d33000000000000,
    0x3fe814d2add1071a, 0x3d0f000000000000,
    0x3fe82589994ccd7e, 0xbd21c00000000000,
    0x3fe8364c1eb942d0, 0x3d29d00000000000,
    0x3fe8471a4623cab5, 0x3d47100000000000,
    0x3fe857f4179f5bbc, 0x3d22600000000000,
    0x3fe868d99b4491af, 0xbd32c40000000000,
    0x3fe879cad931a395, 0xbd23000000000000,
    0x3fe88ac7d98a65b8, 0xbd2a800000000000,
    0x3fe89bd0a4785800, 0xbced000000000000,
    0x3fe8ace5422aa223, 0x3d33280000000000,
    0x3fe8be05bad619fa, 0x3d42b40000000000,
    0x3fe8cf3216b54383, 0xbd2ed00000000000,
    0x3fe8e06a5e08664c, 0xbd20500000000000,
    0x3fe8f1ae99157807, 0x3d28280000000000,
    0x3fe902fed0282c0e, 0xbd1cb00000000000,
    0x3fe9145b0b91ff96, 0xbd05e00000000000,
    0x3fe925c353aa2ff9, 0x3cf5400000000000,
    0x3fe93737b0cdc64a, 0x3d17200000000000,
    0x3fe948b82b5f98ae, 0xbd09000000000000,
    0x3fe95a44cbc852cb, 0x3d25680000000000,
    0x3fe96bdd9a766f21, 0xbd36d00000000000,
    0x3fe97d829fde4e2a, 0xbd01000000000000,
    0x3fe98f33e47a23a3, 0x3d2d000000000000,
    0x3fe9a0f170ca0604, 0xbd38a40000000000,
    0x3fe9b2bb4d53ff89, 0x3d355c0000000000,
    0x3fe9c49182a3f15b, 0x3d26b80000000000,
    0x3fe9d674194bb8c5, 0xbcec000000000000,
    0x3fe9e86319e3238e, 0x3d17d00000000000,
    0x3fe9fa5e8d07f302, 0x3d16400000000000,
    0x3fea0c667b5de54d, 0xbcf5000000000000,
    0x3fea1e7aed8eb8f6, 0x3d09e00000000000,
    0x3fea309bec4a2e27, 0x3d2ad80000000000,
    0x3fea42c980460a5d, 0xbd1af00000000000,
    0x3fea5503b23e259b, 0x3d0b600000000000,
    0x3fea674a8af46213, 0x3d38880000000000,
    0x3fea799e1330b3a7, 0x3d11200000000000,
    0x3fea8bfe53c12e8d, 0x3d06c00000000000,
    0x3fea9e6b5579fcd2, 0xbd29b80000000000,
    0x3feab0e521356fb8, 0x3d2b700000000000,
    0x3feac36bbfd3f381, 0x3cd9000000000000,
    0x3fead5ff3a3c2780, 0x3ce4000000000000,
    0x3feae89f995ad2a3, 0xbd2c900000000000,
    0x3feafb4ce622f367, 0x3d16500000000000,
    0x3feb0e07298db790, 0x3d2fd40000000000,
    0x3feb20ce6c9a89a9, 0x3d12700000000000,
    0x3feb33a2b84f1a4b, 0x3d4d470000000000,
    0x3feb468415b747e7, 0xbd38380000000000,
    0x3feb59728de5593a, 0x3c98000000000000,
    0x3feb6c6e29f1c56a, 0x3d0ad00000000000,
    0x3feb7f76f2fb5e50, 0x3cde800000000000,
    0x3feb928cf22749b2, 0xbd04c00000000000,
    0x3feba5b030a10603, 0xbd0d700000000000,
    0x3febb8e0b79a6f66, 0x3d0d900000000000,
    0x3febcc1e904bc1ff, 0x3d02a00000000000,
    0x3febdf69c3f3a16f, 0xbd1f780000000000,
    0x3febf2c25bd71db8, 0xbd10a00000000000,
    0x3fec06286141b2e9, 0xbd11400000000000,
    0x3fec199bdd8552e0, 0x3d0be00000000000,
    0x3fec2d1cd9fa64ee, 0xbd09400000000000,
    0x3fec40ab5fffd02f, 0xbd0ed00000000000,
    0x3fec544778fafd15, 0x3d39660000000000,
    0x3fec67f12e57d0cb, 0xbd1a100000000000,
    0x3fec7ba88988c1b6, 0xbd58458000000000,
    0x3fec8f6d9406e733, 0xbd1a480000000000,
    0x3feca3405751c4df, 0x3ccb000000000000,
    0x3fecb720dcef9094, 0x3d01400000000000,
    0x3feccb0f2e6d1689, 0x3cf0200000000000,
    0x3fecdf0b555dc412, 0x3cf3600000000000,
    0x3fecf3155b5bab3b, 0xbd06900000000000,
    0x3fed072d4a0789bc, 0x3d09a00000000000,
    0x3fed1b532b08c8fa, 0xbd15e00000000000,
    0x3fed2f87080d8a85, 0x3d1d280000000000,
    0x3fed43c8eacaa203, 0x3d01a00000000000,
    0x3fed5818dcfba491, 0x3cdf000000000000,
    0x3fed6c76e862e6a1, 0xbd03a00000000000,
    0x3fed80e316c9834e, 0xbd0cd80000000000,
    0x3fed955d71ff6090, 0x3cf4c00000000000,
    0x3feda9e603db32ae, 0x3cff900000000000,
    0x3fedbe7cd63a8325, 0x3ce9800000000000,
    0x3fedd321f301b445, 0xbcf5200000000000,
    0x3fede7d5641c05bf, 0xbd1d700000000000,
    0x3fedfc97337b9aec, 0xbd16140000000000,
    0x3fee11676b197d5e, 0x3d0b480000000000,
    0x3fee264614f5a3e7, 0x3d40ce0000000000,
    0x3fee3b333b16ee5c, 0x3d0c680000000000,
    0x3fee502ee78b3fb4, 0xbd09300000000000,
    0x3fee653924676d68, 0xbce5000000000000,
    0x3fee7a51fbc74c44, 0xbd07f80000000000,
    0x3fee8f7977cdb726, 0xbcf3700000000000,
    0x3feea4afa2a490e8, 0x3ce5d00000000000,
    0x3feeb9f4867ccae4, 0x3d161a0000000000,
    0x3feecf482d8e680d, 0x3cf5500000000000,
    0x3feee4aaa2188514, 0x3cc6400000000000,
    0x3feefa1bee615a13, 0xbcee800000000000,
    0x3fef0f9c1cb64106, 0xbcfa880000000000,
    0x3fef252b376bb963, 0xbd2c900000000000,
    0x3fef3ac948dd7275, 0x3caa000000000000,
    0x3fef50765b6e4524, 0xbcf4f00000000000,
    0x3fef6632798844fd, 0x3cca800000000000,
    0x3fef7bfdad9cbe38, 0x3cfabc0000000000,
    0x3fef91d802243c82, 0xbcd4600000000000,
    0x3fefa7c1819e908e, 0xbd0b0c0000000000,
    0x3fefbdba3692d511, 0xbcc0e00000000000,
    0x3fefd3c22b8f7194, 0xbd10de8000000000,
    0x3fefe9d96b2a23ee, 0x3cee430000000000,
    0x3ff0000000000000, 0x0,
    0x3ff00b1afa5abcbe, 0xbcb3400000000000,
    0x3ff0163da9fb3303, 0xbd12170000000000,
    0x3ff02168143b0282, 0x3cba400000000000,
    0x3ff02c9a3e77806c, 0x3cef980000000000,
    0x3ff037d42e11bbca, 0xbcc7400000000000,
    0x3ff04315e86e7f89, 0x3cd8300000000000,
    0x3ff04e5f72f65467, 0xbd1a3f0000000000,
    0x3ff059b0d315855a, 0xbd02840000000000,
    0x3ff0650a0e3c1f95, 0x3cf1600000000000,
    0x3ff0706b29ddf71a, 0x3d15240000000000,
    0x3ff07bd42b72a82d, 0xbce9a00000000000,
    0x3ff0874518759bd0, 0x3ce6400000000000,
    0x3ff092bdf66607c8, 0xbd00780000000000,
    0x3ff09e3ecac6f383, 0xbc98000000000000,
    0x3ff0a9c79b1f3930, 0x3cffa00000000000,
    0x3ff0b5586cf988fc, 0xbcfac80000000000,
    0x3ff0c0f145e46c8a, 0x3cd9c00000000000,
    0x3ff0cc922b724816, 0x3d05200000000000,
    0x3ff0d83b23395dd8, 0xbcfad00000000000,
    0x3ff0e3ec32d3d1f3, 0x3d1bac0000000000,
    0x3ff0efa55fdfa9a6, 0xbd04e80000000000,
    0x3ff0fb66affed2f0, 0xbd0d300000000000,
    0x3ff1073028d7234b, 0x3cf1500000000000,
    0x3ff11301d0125b5b, 0x3cec000000000000,
    0x3ff11edbab5e2af9, 0x3d16bc0000000000,
    0x3ff12abdc06c31d5, 0x3ce8400000000000,
    0x3ff136a814f2047d, 0xbd0ed00000000000,
    0x3ff1429aaea92de9, 0x3ce8e00000000000,
    0x3ff14e95934f3138, 0x3ceb400000000000,
    0x3ff15a98c8a58e71, 0x3d05300000000000,
    0x3ff166a45471c3df, 0x3d03380000000000,
    0x3ff172b83c7d5211, 0x3d28d40000000000,
    0x3ff17ed48695bb9f, 0xbd05d00000000000,
    0x3ff18af9388c8d93, 0xbd1c880000000000,
    0x3ff1972658375d66, 0x3d11f00000000000,
    0x3ff1a35beb6fcba7, 0x3d10480000000000,
    0x3ff1af99f81387e3, 0xbd47390000000000,
    0x3ff1bbe084045d54, 0x3d24e40000000000,
    0x3ff1c82f95281c43, 0xbd0a200000000000,
    0x3ff1d4873168b9b2, 0x3ce3800000000000,
    0x3ff1e0e75eb44031, 0x3ceac00000000000,
    0x3ff1ed5022fcd938, 0x3d01900000000000,
    0x3ff1f9c18438cdf7, 0xbd1b780000000000,
    0x3ff2063b88628d8f, 0x3d2d940000000000,
    0x3ff212be3578a81e, 0x3cd8000000000000,
    0x3ff21f49917ddd41, 0x3d2b340000000000,
    0x3ff22bdda2791323, 0x3d19f80000000000,
    0x3ff2387a6e7561e7, 0xbd19c80000000000,
    0x3ff2451ffb821427, 0x3d02300000000000,
    0x3ff251ce4fb2a602, 0xbd13480000000000,
    0x3ff25e85711eceb0, 0x3d12700000000000,
    0x3ff26b4565e27d16, 0x3d11d00000000000,
    0x3ff2780e341de00f, 0x3d31ee0000000000,
    0x3ff284dfe1f5633e, 0xbd14c00000000000,
    0x3ff291ba7591bb30, 0xbd13d80000000000,
    0x3ff29e9df51fdf09, 0x3d08b00000000000,
    0x3ff2ab8a66d10e9b, 0xbd227c0000000000,
    0x3ff2b87fd0dada3a, 0x3d2a340000000000,
    0x3ff2c57e39771af9, 0xbd10800000000000,
    0x3ff2d285a6e402d9, 0xbd0ed00000000000,
    0x3ff2df961f641579, 0xbcf4200000000000,
    0x3ff2ecafa93e2ecf, 0xbd24980000000000,
    0x3ff2f9d24abd8822, 0xbd16300000000000,
    0x3ff306fe0a31b625, 0xbd32360000000000,
    0x3ff31432edeea50b, 0xbd70df8000000000,
    0x3ff32170fc4cd7b8, 0xbd22480000000000,
    0x3ff32eb83ba8e9a2, 0xbd25980000000000,
    0x3ff33c08b2641766, 0x3d1ed00000000000,
    0x3ff3496266e3fa27, 0xbcdc000000000000,
    0x3ff356c55f929f0f, 0xbd30d80000000000,
    0x3ff36431a2de88b9, 0x3d22c80000000000,
    0x3ff371a7373aaa39, 0x3d20600000000000,
    0x3ff37f26231e74fe, 0xbd16600000000000,
    0x3ff38cae6d05d838, 0xbd0ae00000000000,
    0x3ff39a401b713ec3, 0xbd44720000000000,
    0x3ff3a7db34e5a020, 0x3d08200000000000,
    0x3ff3b57fbfec6e95, 0x3d3e800000000000,
    0x3ff3c32dc313a8f2, 0x3cef800000000000,
    0x3ff3d0e544ede122, 0xbd17a00000000000,
    0x3ff3dea64c1234bb, 0x3d26300000000000,
    0x3ff3ec70df1c4ecc, 0xbd48a60000000000,
    0x3ff3fa4504ac7e8c, 0xbd3cdc0000000000,
    0x3ff40822c367a0bb, 0x3d25b80000000000,
    0x3ff4160a21f72e95, 0x3d1ec00000000000,
    0x3ff423fb27094646, 0xbd13600000000000,
    0x3ff431f5d950a920, 0x3d23980000000000,
    0x3ff43ffa3f84b9eb, 0x3cfa000000000000,
    0x3ff44e0860618919, 0xbcf6c00000000000,
    0x3ff45c2042a7d201, 0xbd0bc00000000000,
    0x3ff46a41ed1d0016, 0xbd12800000000000,
    0x3ff4786d668b3326, 0x3d30e00000000000,
    0x3ff486a2b5c13c00, 0xbd2d400000000000,
    0x3ff494e1e192af04, 0x3d0c200000000000,
    0x3ff4a32af0d7d372, 0xbd1e500000000000,
    0x3ff4b17dea6db801, 0x3d07800000000000,
    0x3ff4bfdad53629e1, 0xbd13800000000000,
    0x3ff4ce41b817c132, 0x3d00800000000000,
    0x3ff4dcb299fddddb, 0x3d2c700000000000,
    0x3ff4eb2d81d8ab96, 0xbd1ce00000000000,
    0x3ff4f9b2769d2d02, 0x3d19200000000000,
    0x3ff508417f4531c1, 0xbd08c00000000000,
    0x3ff516daa2cf662a, 0xbcfa000000000000,
    0x3ff5257de83f51ea, 0x3d4a080000000000,
    0x3ff5342b569d4eda, 0xbd26d80000000000,
    0x3ff542e2f4f6ac1a, 0xbd32440000000000,
    0x3ff551a4ca5d94db, 0x3d483c0000000000,
    0x3ff56070dde9116b, 0x3d24b00000000000,
    0x3ff56f4736b529de, 0x3d415a0000000000,
    0x3ff57e27dbe2c40e, 0xbd29e00000000000,
    0x3ff58d12d497c76f, 0xbd23080000000000,
    0x3ff59c0827ff0b4c, 0x3d4dec0000000000,
    0x3ff5ab07dd485427, 0xbcc4000000000000,
    0x3ff5ba11fba87af4, 0x3d30080000000000,
    0x3ff5c9268a59460b, 0xbd26c80000000000,
    0x3ff5d84590998e3f, 0x3d469a0000000000,
    0x3ff5e76f15ad20e1, 0xbd1b400000000000,
    0x3ff5f6a320dcebca, 0x3d17700000000000,
    0x3ff605e1b976dcb8, 0x3d26f80000000000,
    0x3ff6152ae6cdf715, 0x3d01000000000000,
    0x3ff6247eb03a5531, 0xbd15d00000000000,
    0x3ff633dd1d1929b5, 0xbd12d00000000000,
    0x3ff6434634ccc313, 0xbcea800000000000,
    0x3ff652b9febc8efa, 0xbd28600000000000,
    0x3ff6623882553397, 0x3d71fe0000000000,
    0x3ff671c1c708328e, 0xbd37200000000000,
    0x3ff68155d44ca97e, 0x3ce6800000000000,
    0x3ff690f4b19e9471, 0xbd29780000000000,
];

/// `2ˣ` (FreeBSD `s_exp2.c`, accurate tables): `x = k + i/256 + z − eps[i]`, the 256-entry table
/// for `2^(i/256)` and a degree-5 minimax polynomial for `2^(z − eps[i])` on `|z| ≤ 2⁻⁹`, then
/// `scalbn`. `exp2(±∞)`, overflow above 1024 and underflow below −1075 as for `exp`. < 1 ULP
/// (peak 0.503 ULP for normal results).
pub fn exp2(x: f64) -> f64 {
    const TBLSIZE: u32 = 256;
    let redux = f64::from_bits(0x4338_0000_0000_0000) / TBLSIZE as f64; // 1.5·2^52/256
    let p1 = f64::from_bits(0x3fe6_2e42_fefa_39ef);
    let p2 = f64::from_bits(0x3fce_bfbd_ff82_c575);
    let p3 = f64::from_bits(0x3fac_6b08_d704_a0a6);
    let p4 = f64::from_bits(0x3f83_b2ab_88f7_0400);
    let p5 = f64::from_bits(0x3f55_d880_0387_5c74);
    let x1p1023 = f64::from_bits(0x7fe0_0000_0000_0000);
    // Filter out exceptional cases.
    let ui = x.to_bits();
    let ix = (ui >> 32) & 0x7fff_ffff;
    if ix >= 0x408f_f000 {
        // |x| >= 1022 or NaN
        if ix >= 0x4090_0000 && ui >> 63 == 0 {
            // x >= 1024 or NaN: overflow (NaN propagates)
            return x * x1p1023;
        }
        if ix >= 0x7ff0_0000 {
            // −inf or −NaN
            return -1.0 / x;
        }
        if ui >> 63 != 0 && x <= -1075.0 {
            // underflow to +0
            return 0.0;
        }
    } else if ix < 0x3c90_0000 {
        // |x| < 2^-54
        return 1.0 + x;
    }
    // Reduce x, computing z, i0, and k.
    let ui = (x + redux).to_bits();
    let mut i0 = ui as u32;
    i0 = i0.wrapping_add(TBLSIZE / 2);
    let ku = i0 / TBLSIZE * TBLSIZE;
    let ki = (ku as i32) / (TBLSIZE as i32);
    i0 %= TBLSIZE;
    let uf = f64::from_bits(ui) - redux;
    let mut z = x - uf;
    // Compute r = exp2(y) = exp2t[i0]·p(z − eps[i]).
    let t = f64::from_bits(EXP2_TBL[2 * i0 as usize]); // exp2t[i0]
    z -= f64::from_bits(EXP2_TBL[2 * i0 as usize + 1]); // eps[i0]
    let r = t + t * z * (p1 + z * (p2 + z * (p3 + z * (p4 + z * p5))));
    scalbn(r, ki)
}

const EXP10_LOG2_10: f64 = 3.32192809488736234787031942948939;
const EXP10_P10: [f64; 31] = [
    1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
];

/// `10ˣ` (musl `exp10.c`): for `|x| < 16`, `x = n + y` with `n` integral — exactly the power-of-ten
/// table entry when `y = 0` (so `exp10(3) == 1000` exactly), else `exp2(log₂10·y)·10ⁿ`; otherwise
/// [`pow`]`(10, x)`. ≤ 3 ULP (the `log₂10·y` product rounds once before `exp2`).
pub fn exp10(x: f64) -> f64 {
    let n = x.trunc();
    let y = x - n;
    let u = n.to_bits();
    // |n| < 16 without raising invalid on NaN (a NaN's exponent field is 0x7ff)
    if ((u >> 52) & 0x7ff) < 0x3ff + 4 {
        let idx = (n as isize + 15) as usize;
        if y == 0.0 {
            return EXP10_P10[idx];
        }
        return exp2(EXP10_LOG2_10 * y) * EXP10_P10[idx];
    }
    pow(10.0, x)
}

// --- hypot -------------------------------------------------------------------------------------

const HYPOT_SPLIT: f64 = 134217728.0 + 1.0; // 2^27 + 1: the Dekker split constant

/// `x²` as an exact double-double `(hi, lo)` by Dekker splitting.
fn sq(x: f64) -> (f64, f64) {
    let xc = x * HYPOT_SPLIT;
    let xh = x - xc + xc;
    let xl = x - xh;
    let hi = x * x;
    let lo = xh * xh - hi + 2.0 * xh * xl + xl * xl;
    (hi, lo)
}

/// `√(x² + y²)` (musl `hypot.c`): both squares as exact double-doubles summed smallest-first, so
/// the `sqrt` argument is correctly rounded; scaled by `2^±700` to avoid over/underflow.
/// `hypot(±∞, anything)` (NaN included) is `+∞`; `hypot(x, ±0) = |x|`. < 1 ULP.
pub fn hypot(x: f64, y: f64) -> f64 {
    let x1p700 = f64::from_bits(0x6bb0_0000_0000_0000);
    let x1p_700 = f64::from_bits(0x1430_0000_0000_0000);
    let mut uxi = x.to_bits() & 0x7fff_ffff_ffff_ffff;
    let mut uyi = y.to_bits() & 0x7fff_ffff_ffff_ffff;
    // arrange |x| >= |y|
    if uxi < uyi {
        std::mem::swap(&mut uxi, &mut uyi);
    }
    let ex = (uxi >> 52) as i64;
    let ey = (uyi >> 52) as i64;
    let mut x = f64::from_bits(uxi);
    let mut y = f64::from_bits(uyi);
    // hypot(inf, nan) == inf
    if ey == 0x7ff {
        return y;
    }
    if ex == 0x7ff || uyi == 0 {
        return x;
    }
    // hypot(x, y) ≈ x + y²/x/2 for small y/x: 64 bits of difference is enough
    if ex - ey > 64 {
        return x + y;
    }
    // precise sqrt argument without overflow: xh·xh must not overflow, xl·xl must not underflow
    let mut z = 1.0;
    if ex > 0x3ff + 510 {
        z = x1p700;
        x *= x1p_700;
        y *= x1p_700;
    } else if ey < 0x3ff - 450 {
        z = x1p_700;
        x *= x1p700;
        y *= x1p700;
    }
    let (hx, lx) = sq(x);
    let (hy, ly) = sq(y);
    z * (ly + lx + hy + hx).sqrt()
}

// --- pow ---------------------------------------------------------------------------------------

const POW_BP: [f64; 2] = [1.0, 1.5];
const POW_DP_H: [f64; 2] = [0.0, 5.84962487220764160156e-01]; /* 0x3fe2b803, 0x40000000 */
const POW_DP_L: [f64; 2] = [0.0, 1.35003920212974897128e-08]; /* 0x3E4CFDEB, 0x43CFD006 */
const POW_TWO53: f64 = 9007199254740992.0; /* 0x43400000, 0x00000000 */
const POW_HUGE: f64 = 1.0e300;
const POW_TINY: f64 = 1.0e-300;
/* poly coefs for (3/2)·(log(x) − 2s − 2/3·s³) */
const POW_L1: f64 = 5.99999999999994648725e-01; /* 0x3fe33333, 0x33333303 */
const POW_L2: f64 = 4.28571428578550184252e-01; /* 0x3fdb6db6, 0xdb6fabff */
const POW_L3: f64 = 3.33333329818377432918e-01; /* 0x3fd55555, 0x518f264d */
const POW_L4: f64 = 2.72728123808534006489e-01; /* 0x3fd17460, 0xa91d4101 */
const POW_L5: f64 = 2.30660745775561754067e-01; /* 0x3fcd864a, 0x93c9db65 */
const POW_L6: f64 = 2.06975017800338417784e-01; /* 0x3fca7e28, 0x4a454eef */
const POW_LG2: f64 = 6.93147180559945286227e-01; /* 0x3fe62e42, 0xfefa39ef */
const POW_LG2_H: f64 = 6.93147182464599609375e-01; /* 0x3fe62e43, 0x00000000 */
const POW_LG2_L: f64 = -1.90465429995776804525e-09; /* 0xbe205c61, 0x0ca86c39 */
const POW_OVT: f64 = 8.0085662595372944372e-017; /* −(1024 − log2(ovfl + .5ulp)) */
const POW_CP: f64 = 9.61796693925975554329e-01; /* 0x3feec709, 0xdc3a03fd = 2/(3ln2) */
const POW_CP_H: f64 = 9.61796700954437255859e-01; /* 0x3feec709, 0xe0000000 = (float)cp */
const POW_CP_L: f64 = -7.02846165095275826516e-09; /* 0xbe3e2fe0, 0x145b01f5 = tail of cp_h */
const POW_IVLN2: f64 = 1.44269504088896338700e+00; /* 0x3ff71547, 0x652b82fe = 1/ln2 */
const POW_IVLN2_H: f64 = 1.44269502162933349609e+00; /* 0x3ff71547, 0x60000000 = 24b 1/ln2 */
const POW_IVLN2_L: f64 = 1.92596299112661746887e-08; /* 0x3e54ae0b, 0xf85ddf44 = 1/ln2 tail */

/// `xʸ` (fdlibm `e_pow.c`): `log₂x` in two pieces (`t1` with 29 trailing zero bits, `t2`), the
/// product `y·log₂x = n + y'` in simulated multi-precision, then `2ⁿ·exp(y'·ln2)`. The 23 special
/// cases of C99 Annex F are handled first, so `pow(x, ±0) = 1` and `pow(1, y) = 1` even for NaN,
/// `pow(−0, odd) = −0`, `pow(negative, non-integer) = NaN`, and `pow(integer, integer)` is exact
/// whenever the result is representable. < 1 ULP.
pub fn pow(x: f64, y: f64) -> f64 {
    let hx = hi(x) as i32;
    let lx = lo(x);
    let hy = hi(y) as i32;
    let ly = lo(y);
    let mut ix = hx & 0x7fff_ffff;
    let iy = hy & 0x7fff_ffff;
    // x^0 = 1, even if x is NaN
    if ((iy as u32) | ly) == 0 {
        return 1.0;
    }
    // 1^y = 1, even if y is NaN
    if hx == 0x3ff0_0000 && lx == 0 {
        return 1.0;
    }
    // NaN if either argument is NaN
    if ix > 0x7ff0_0000
        || (ix == 0x7ff0_0000 && lx != 0)
        || iy > 0x7ff0_0000
        || (iy == 0x7ff0_0000 && ly != 0)
    {
        return x + y;
    }
    // Determine whether y is an odd integer when x < 0:
    //   yisint = 0 … y is not an integer; 1 … odd integer; 2 … even integer.
    let mut yisint: i32 = 0;
    if hx < 0 {
        if iy >= 0x4340_0000 {
            yisint = 2; // even integer y (|y| >= 2^53)
        } else if iy >= 0x3ff0_0000 {
            let k = (iy >> 20) - 0x3ff; // exponent
            if k > 20 {
                let j = (ly >> (52 - k)) as i32;
                if (j << (52 - k)) == (ly as i32) {
                    yisint = 2 - (j & 1);
                }
            } else if ly == 0 {
                let j = iy >> (20 - k);
                if (j << (20 - k)) == iy {
                    yisint = 2 - (j & 1);
                }
            }
        }
    }
    // special values of y
    if ly == 0 {
        if iy == 0x7ff0_0000 {
            // y is ±inf
            return if ((ix - 0x3ff0_0000) | (lx as i32)) == 0 {
                1.0 // (−1)^±inf is 1
            } else if ix >= 0x3ff0_0000 {
                // (|x| > 1)^±inf = inf, 0
                if hy >= 0 {
                    y
                } else {
                    0.0
                }
            } else {
                // (|x| < 1)^±inf = 0, inf
                if hy >= 0 {
                    0.0
                } else {
                    -y
                }
            };
        }
        if iy == 0x3ff0_0000 {
            // y is ±1
            return if hy >= 0 { x } else { 1.0 / x };
        }
        if hy == 0x4000_0000 {
            // y is 2
            return x * x;
        }
        if hy == 0x3fe0_0000 && hx >= 0 {
            // y is 0.5, x >= +0
            return x.sqrt();
        }
    }
    let mut ax = x.abs();
    // special values of x
    if lx == 0 && (ix == 0x7ff0_0000 || ix == 0 || ix == 0x3ff0_0000) {
        // x is ±0, ±inf, ±1
        let mut z = ax;
        if hy < 0 {
            z = 1.0 / z; // z = 1/|x|
        }
        if hx < 0 {
            if ((ix - 0x3ff0_0000) | yisint) == 0 {
                z = f64::NAN; // (−1)^non-int is NaN
            } else if yisint == 1 {
                z = -z; // (x < 0)^odd = −(|x|^odd)
            }
        }
        return z;
    }
    let mut s = 1.0; // sign of the result
    if hx < 0 {
        if yisint == 0 {
            // (x < 0)^(non-int) is NaN
            return f64::NAN;
        }
        if yisint == 1 {
            s = -1.0; // (x < 0)^(odd int)
        }
    }
    let t1: f64;
    let t2: f64;
    // |y| is huge
    if iy > 0x41e0_0000 {
        // |y| > 2^31
        if iy > 0x43f0_0000 {
            // |y| > 2^64: must over/underflow
            if ix <= 0x3fef_ffff {
                return if hy < 0 {
                    POW_HUGE * POW_HUGE
                } else {
                    POW_TINY * POW_TINY
                };
            }
            if ix >= 0x3ff0_0000 {
                return if hy > 0 {
                    POW_HUGE * POW_HUGE
                } else {
                    POW_TINY * POW_TINY
                };
            }
        }
        // over/underflow if x is not close to one
        if ix < 0x3fef_ffff {
            return if hy < 0 {
                s * POW_HUGE * POW_HUGE
            } else {
                s * POW_TINY * POW_TINY
            };
        }
        if ix > 0x3ff0_0000 {
            return if hy > 0 {
                s * POW_HUGE * POW_HUGE
            } else {
                s * POW_TINY * POW_TINY
            };
        }
        // now |1 − x| is tiny <= 2^-20: log(x) by x − x²/2 + x³/3 − x⁴/4 suffices
        let t = ax - 1.0; // t has 20 trailing zeros
        let w = (t * t) * (0.5 - t * (0.3333333333333333333333 - t * 0.25));
        let u = POW_IVLN2_H * t; // ivln2_h has 21 significant bits
        let v = t * POW_IVLN2_L - w * POW_IVLN2;
        t1 = with_lo(u + v, 0);
        t2 = v - (t1 - u);
    } else {
        let mut n: i32 = 0;
        if ix < 0x0010_0000 {
            // subnormal x
            ax *= POW_TWO53;
            n -= 53;
            ix = hi(ax) as i32;
        }
        n += (ix >> 20) - 0x3ff;
        let j = ix & 0x000f_ffff;
        // determine the interval
        ix = j | 0x3ff0_0000; // normalize ix
        let k: usize;
        if j <= 0x3988e {
            k = 0; // |x| < sqrt(3/2)
        } else if j < 0xbb67a {
            k = 1; // |x| < sqrt(3)
        } else {
            k = 0;
            n += 1;
            ix -= 0x0010_0000;
        }
        ax = with_hi(ax, ix as u32);
        // compute ss = s_h + s_l = (x − 1)/(x + 1) or (x − 1.5)/(x + 1.5)
        let u = ax - POW_BP[k]; // bp[0] = 1.0, bp[1] = 1.5
        let v = 1.0 / (ax + POW_BP[k]);
        let ss = u * v;
        let s_h = with_lo(ss, 0);
        // t_h = ax + bp[k], high part
        let t_h = from_words(
            ((ix as u32 >> 1) | 0x2000_0000) + 0x0008_0000 + ((k as u32) << 18),
            0,
        );
        let t_l = ax - (t_h - POW_BP[k]);
        let s_l = v * ((u - s_h * t_h) - s_h * t_l);
        // compute log(ax)
        let s2 = ss * ss;
        let mut r = s2
            * s2
            * (POW_L1
                + s2 * (POW_L2 + s2 * (POW_L3 + s2 * (POW_L4 + s2 * (POW_L5 + s2 * POW_L6)))));
        r += s_l * (s_h + ss);
        let s2 = s_h * s_h;
        let t_h = with_lo(3.0 + s2 + r, 0);
        let t_l = r - ((t_h - 3.0) - s2);
        // u + v = ss·(1 + …)
        let u = s_h * t_h;
        let v = s_l * t_h + t_l * ss;
        // 2/(3log2)·(ss + …)
        let p_h = with_lo(u + v, 0);
        let p_l = v - (p_h - u);
        let z_h = POW_CP_H * p_h; // cp_h + cp_l = 2/(3·log2)
        let z_l = POW_CP_L * p_h + p_l * POW_CP + POW_DP_L[k];
        // log2(ax) = (ss + …)·2/(3·log2) = n + dp_h + z_h + z_l
        let t = n as f64;
        t1 = with_lo(((z_h + z_l) + POW_DP_H[k]) + t, 0);
        t2 = z_l - (((t1 - t) - POW_DP_H[k]) - z_h);
    }
    // split up y into y1 + y2 and compute (y1 + y2)·(t1 + t2)
    let y1 = with_lo(y, 0);
    let p_l = (y - y1) * t1 + y * t2;
    let mut p_h = y1 * t1;
    let z = p_l + p_h;
    let j = hi(z) as i32;
    let i = lo(z) as i32;
    if j >= 0x4090_0000 {
        // z >= 1024
        if ((j - 0x4090_0000) | i) != 0 {
            return s * POW_HUGE * POW_HUGE; // overflow
        }
        if p_l + POW_OVT > z - p_h {
            return s * POW_HUGE * POW_HUGE; // overflow
        }
    } else if (j & 0x7fff_ffff) >= 0x4090_cc00 {
        // z <= −1075
        if (((j as u32).wrapping_sub(0xc090_cc00)) | (i as u32)) != 0 {
            return s * POW_TINY * POW_TINY; // underflow
        }
        if p_l <= z - p_h {
            return s * POW_TINY * POW_TINY; // underflow
        }
    }
    // compute 2^(p_h + p_l)
    let i = j & 0x7fff_ffff;
    let mut k = (i >> 20) - 0x3ff;
    let mut n: i32 = 0;
    if i > 0x3fe0_0000 {
        // if |z| > 0.5, set n = [z + 0.5]
        n = j + (0x0010_0000 >> (k + 1));
        k = ((n & 0x7fff_ffff) >> 20) - 0x3ff; // new k for n
        let t = from_words((n & !(0x000f_ffff >> k)) as u32, 0);
        n = ((n & 0x000f_ffff) | 0x0010_0000) >> (20 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }
    let t = with_lo(p_l + p_h, 0);
    let u = t * POW_LG2_H;
    let v = (p_l - (t - p_h)) * POW_LG2 + t * POW_LG2_L;
    let mut z = u + v;
    let w = v - (z - u);
    let t = z * z;
    let t1 = z - t * (EXP_P1 + t * (EXP_P2 + t * (EXP_P3 + t * (EXP_P4 + t * EXP_P5))));
    let r = (z * t1) / (t1 - 2.0) - (w + z * w);
    z = 1.0 - (r - z);
    let mut j = hi(z) as i32;
    j += n << 20;
    if (j >> 20) <= 0 {
        z = scalbn(z, n); // subnormal output
    } else {
        z = with_hi(z, j as u32);
    }
    s * z
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI as STD_PI};

    /// The monotone integer image of a double (−0 and +0 both map to 0), so that the difference of
    /// two images is the number of representable doubles between them.
    fn ordered(x: f64) -> i64 {
        let b = x.to_bits() as i64;
        if b < 0 {
            i64::MIN - b
        } else {
            b
        }
    }

    fn ulp_diff(a: f64, b: f64) -> u64 {
        (ordered(a) - ordered(b)).unsigned_abs()
    }

    fn f1(op: i64, x: f64) -> f64 {
        wukong_rt_math_f64(x, op)
    }

    fn f2(op: i64, a: f64, b: f64) -> f64 {
        wukong_rt_math2_f64(a, b, op)
    }

    /// Correctly rounded references from mpmath at 200 bits (`float(mpf)` rounds to nearest), one
    /// row per `(op, argument bits, expected bits, ULP bound)`. The bound column is the documented
    /// bound of the algorithm, not a recording of this implementation's error.
    const REF1: &[(i64, u64, u64, u64)] = &[
        (VM_EXP, 0x3ff0000000000000, 0x4005bf0a8b145769, 1), // exp(1.0) = 2.718281828459045
        (VM_EXP, 0xbff0000000000000, 0x3fd78b56362cef38, 1), // exp(-1.0) = 0.36787944117144233
        (VM_EXP, 0x3fe0000000000000, 0x3ffa61298e1e069c, 1), // exp(0.5) = 1.6487212707001282
        (VM_EXP, 0xbfe0000000000000, 0x3fe368b2fc6f960a, 1), // exp(-0.5) = 0.6065306597126334
        (VM_EXP, 0x3ddb7cdfd9d7bdbb, 0x3ff000000006df38, 1), // exp(1e-10) = 1.0000000001
        (VM_EXP, 0xbddb7cdfd9d7bdbb, 0x3feffffffff24190, 1), // exp(-1e-10) = 0.9999999999
        (VM_EXP, 0x4024000000000000, 0x40d5829dcf950560, 1), // exp(10.0) = 22026.465794806718
        (VM_EXP, 0xc024000000000000, 0x3f07cd79b5647c9b, 1), // exp(-10.0) = 4.5399929762484854e-05
        (VM_EXP, 0x4059000000000000, 0x48f3494a9b171bf5, 1), // exp(100.0) = 2.6881171418161356e+43
        (VM_EXP, 0xc059000000000000, 0x36ea8c1f14e2af5d, 1), // exp(-100.0) = 3.720075976020836e-44
        (VM_EXP, 0x4085e00000000000, 0x7f0d945df4f8ec8e, 1), // exp(700.0) = 1.0142320547350045e+304
        (VM_EXP, 0xc085e00000000000, 0x00d14f2b0fb9307f, 1), // exp(-700.0) = 9.85967654375977e-305
        (VM_EXP, 0x40862d999999999a, 0x7fed75ae7a50ee14, 1), // exp(709.7) = 1.6549840276802644e+308
        (VM_EXP, 0xc087480000000000, 0x0000000000000001, 1), // exp(-745.0) = 5e-324
        (VM_EXP, 0x3fb999999999999a, 0x3ff1aec7b35a00d4, 1), // exp(0.1) = 1.1051709180756477
        (VM_EXP, 0x4004000000000000, 0x40285d6fd931e0bb, 1), // exp(2.5) = 12.182493960703473
        (VM_EXP, 0xc004000000000000, 0x3fb50385c094f425, 1), // exp(-2.5) = 0.0820849986238988
        (VM_EXP, 0x400921f9f01b866e, 0x4037240068789162, 1), // exp(3.14159) = 23.14063122695496
        (VM_EXP, 0x40341f9add373963, 0x41c05be66d679343, 1), // exp(20.123456789) = 548916442.8091816
        (VM_EXP, 0xc042c00000000000, 0x3c8dd5c566301ec8, 1), // exp(-37.5) = 5.175555005801869e-17
        (VM_EXP, 0x3ee4f8b588e368f1, 0x3ff0000a7c5e340e, 1), // exp(1e-05) = 1.00001000005
        (VM_EXP, 0x3fe62e42fefa39ef, 0x4000000000000000, 1), // exp(0.6931471805599453) = 2.0
        (VM_EXP, 0x4056200000000000, 0x47e99b987155fe28, 1), // exp(88.5) = 2.7230878250681117e+38
        (VM_EXP, 0xbfd62e42fefa39ee, 0x3fe6a09e667f3bcd, 1), // exp(-0.3465735902799726) = 0.7071067811865476
        (VM_EXP, 0x01a56e1fc2f8f359, 0x3ff0000000000000, 1), // exp(1e-300) = 1.0
        (VM_EXP, 0xbe45798ee2308c3a, 0x3feffffffaa19c48, 1), // exp(-1e-08) = 0.9999999900000001
        (VM_EXP, 0x3e601b2b29a4692b, 0x3ff00000080d9597, 1), // exp(3e-08) = 1.0000000300000005
        (VM_LOG, 0x3fe0000000000000, 0xbfe62e42fefa39ef, 1), // log(0.5) = -0.6931471805599453
        (VM_LOG, 0x4000000000000000, 0x3fe62e42fefa39ef, 1), // log(2.0) = 0.6931471805599453
        (VM_LOG, 0x4024000000000000, 0x40026bb1bbb55516, 1), // log(10.0) = 2.302585092994046
        (VM_LOG, 0x3ddb7cdfd9d7bdbb, 0xc037069e2aa2aa5b, 1), // log(1e-10) = -23.025850929940457
        (VM_LOG, 0x4202a05f20000000, 0x4037069e2aa2aa5b, 1), // log(10000000000.0) = 23.025850929940457
        (VM_LOG, 0x7e37e43c8800759c, 0x4085963447f87fb5, 1), // log(1e+300) = 690.7755278982137
        (VM_LOG, 0x01a56e1fc2f8f359, 0xc085963447f87fb5, 1), // log(1e-300) = -690.7755278982137
        (VM_LOG, 0x3fefff2e48e8a71e, 0xbf1a3738d2cf1cc2, 1), // log(0.9999) = -0.00010000500033334732
        (VM_LOG, 0x3ff00068db8bac71, 0x3f1a368d0657fcd4, 1), // log(1.0001) = 9.999500033329732e-05
        (VM_LOG, 0x4008000000000000, 0x3ff193ea7aad030b, 1), // log(3.0) = 1.0986122886681098
        (VM_LOG, 0x401d000000000000, 0x3fffb22e98a1c24e, 1), // log(7.25) = 1.9810014688665833
        (VM_LOG, 0x40fe240c9fbe76c9, 0x40277281cad8a844, 1), // log(123456.789) = 11.723646487185881
        (VM_LOG, 0x0010000000000000, 0xc086232bdd7abcd2, 1), // log(2.2250738585072014e-308) = -708.3964185322641
        (VM_LOG, 0x0000000000000001, 0xc0874385446d71c3, 1), // log(5e-324) = -744.4400719213812
        (VM_LOG, 0x7fefffffffffffff, 0x40862e42fefa39ef, 1), // log(1.7976931348623157e+308) = 709.782712893384
        (VM_LOG, 0x3fe6a0902de00d1b, 0xbfd62e6b3842a25e, 1), // log(0.7071) = -0.346583180371942
        (VM_LOG, 0x3ff6a0902de00d1b, 0x3fd62e1ac5b1d181, 1), // log(1.4142) = 0.3465640001880033
        (VM_LOG, 0x3fb999999999999a, 0xc0026bb1bbb55515, 1), // log(0.1) = -2.3025850929940455
        (VM_LOG, 0x4005bf0a8b145769, 0x3ff0000000000000, 1), // log(2.718281828459045) = 1.0
        (VM_SIN, 0x3fe0000000000000, 0x3fdeaee8744b05f0, 1), // sin(0.5) = 0.479425538604203
        (VM_SIN, 0xbfe0000000000000, 0xbfdeaee8744b05f0, 1), // sin(-0.5) = -0.479425538604203
        (VM_SIN, 0x3ff0000000000000, 0x3feaed548f090cee, 1), // sin(1.0) = 0.8414709848078965
        (VM_SIN, 0x4000000000000000, 0x3fed18f6ead1b446, 1), // sin(2.0) = 0.9092974268256817
        (VM_SIN, 0x4008000000000000, 0x3fc210386db6d55b, 1), // sin(3.0) = 0.1411200080598672
        (VM_SIN, 0x400921f9f01b866e, 0x3ec6428a6aa44cd0, 1), // sin(3.14159) = 2.65358979335273e-06
        (VM_SIN, 0x412e848000000000, 0xbfd6664b2568d867, 1), // sin(1000000.0) = -0.34999350217129294
        (VM_SIN, 0x4480f0cf064dd592, 0xbfeb453ab76bf397, 1), // sin(1e+22) = -0.8522008497671888
        (VM_SIN, 0x7e37e43c8800759c, 0xbfea2c16b010e385, 1), // sin(1e+300) = -0.8178819121159085
        (VM_SIN, 0x43b0000000000000, 0xbfea94adab06665c, 1), // sin(1.152921504606847e+18) = -0.8306492176372546
        (VM_SIN, 0x40c81cd6c8b43958, 0xbfe687d5890974a5, 1), // sin(12345.678) = -0.7040813137533816
        (VM_SIN, 0x3fe921fafc8b007a, 0x3fe6a09e28779cb0, 1), // sin(0.785398) = 0.7071066656470943
        (VM_SIN, 0x3ff921fb54442d18, 0x3ff0000000000000, 1), // sin(1.5707963267948966) = 1.0
        (VM_SIN, 0xc059200000000000, 0x3f9fb3f833470ff1, 1), // sin(-100.5) = 0.030959966783271346
        (VM_SIN, 0x3ee4f8b588e368f1, 0x3ee4f8b588e1e8a2, 1), // sin(1e-05) = 9.999999999833334e-06
        (VM_SIN, 0x3fb999999999999a, 0x3fb98eaecb8bcb2c, 1), // sin(0.1) = 0.09983341664682815
        (VM_SIN, 0x401921fb54442d18, 0xbcb1a62633145c07, 1), // sin(6.283185307179586) = -2.4492935982947064e-16
        (VM_SIN, 0x4202a05f20000000, 0xbfdf334c7896a4e3, 1), // sin(10000000000.0) = -0.4875060250875107
        (VM_SIN, 0x41f2a05f20080000, 0xbfcf184e95244226, 1), // sin(5000000000.5) = -0.2429292896490704
        (VM_SIN, 0x4341c37937e08000, 0x3fe8f334432ebba5, 1), // sin(1e+16) = 0.7796880066069788
        (VM_SIN, 0x400921fb54442d18, 0x3ca1a62633145c07, 1), // sin(3.141592653589793) = 1.2246467991473532e-16
        (VM_SIN, 0x4012d97c7f3321d2, 0xbff0000000000000, 1), // sin(4.71238898038469) = -1.0
        (VM_SIN, 0x3e25798ee2308c3a, 0x3e25798ee2308c3a, 1), // sin(2.5e-09) = 2.5e-09
        (VM_SIN, 0x01a56e1fc2f8f359, 0x01a56e1fc2f8f359, 1), // sin(1e-300) = 1e-300
        (VM_COS, 0x3fe0000000000000, 0x3fec1528065b7d50, 1), // cos(0.5) = 0.8775825618903728
        (VM_COS, 0xbfe0000000000000, 0x3fec1528065b7d50, 1), // cos(-0.5) = 0.8775825618903728
        (VM_COS, 0x3ff0000000000000, 0x3fe14a280fb5068c, 1), // cos(1.0) = 0.5403023058681398
        (VM_COS, 0x4000000000000000, 0xbfdaa22657537205, 1), // cos(2.0) = -0.4161468365471424
        (VM_COS, 0x4008000000000000, 0xbfefae04be85e5d2, 1), // cos(3.0) = -0.9899924966004454
        (VM_COS, 0x400921f9f01b866e, 0xbfefffffffff8420, 1), // cos(3.14159) = -0.9999999999964793
        (VM_COS, 0x412e848000000000, 0x3fedf9df9906d32c, 1), // cos(1000000.0) = 0.9367521275331447
        (VM_COS, 0x4480f0cf064dd592, 0x3fe0be2cef01c8f4, 1), // cos(1e+22) = 0.523214785395139
        (VM_COS, 0x7e37e43c8800759c, 0xbfe2699022adc4c1, 1), // cos(1e+300) = -0.5753861119575491
        (VM_COS, 0x43b0000000000000, 0xbfe1d146047d6947, 1), // cos(1.152921504606847e+18) = -0.5567960822766417
        (VM_COS, 0x40c81cd6c8b43958, 0x3fe6b94c3bbe24b8, 1), // cos(12345.678) = 0.7101193587160628
        (VM_COS, 0x3fe921fafc8b007a, 0x3fe6a09ea486da3f, 1), // cos(0.785398) = 0.7071068967259818
        (VM_COS, 0x3ff921fb54442d18, 0x3c91a62633145c07, 1), // cos(1.5707963267948966) = 6.123233995736766e-17
        (VM_COS, 0xc059200000000000, 0x3feffc12adaecec2, 1), // cos(-100.5) = 0.9995206253283515
        (VM_COS, 0x3ee4f8b588e368f1, 0x3feffffffff920c8, 1), // cos(1e-05) = 0.99999999995
        (VM_COS, 0x3fb999999999999a, 0x3fefd712f9a817c1, 1), // cos(0.1) = 0.9950041652780258
        (VM_COS, 0x401921fb54442d18, 0x3ff0000000000000, 1), // cos(6.283185307179586) = 1.0
        (VM_COS, 0x4202a05f20000000, 0x3febf098901c931a, 1), // cos(10000000000.0) = 0.873119622676856
        (VM_COS, 0x41f2a05f20080000, 0xbfef0a99b67cfab7, 1), // cos(5000000000.5) = -0.9700439991209667
        (VM_COS, 0x4341c37937e08000, 0xbfe40991e398dbfc, 1), // cos(1e+16) = -0.6261681981330862
        (VM_COS, 0x400921fb54442d18, 0xbff0000000000000, 1), // cos(3.141592653589793) = -1.0
        (VM_COS, 0x4012d97c7f3321d2, 0xbcaa79394c9e8a0a, 1), // cos(4.71238898038469) = -1.8369701987210297e-16
        (VM_COS, 0x3e25798ee2308c3a, 0x3ff0000000000000, 1), // cos(2.5e-09) = 1.0
        (VM_TAN, 0x3fe0000000000000, 0x3fe17b4f5bf3474a, 1), // tan(0.5) = 0.5463024898437905
        (VM_TAN, 0xbfe0000000000000, 0xbfe17b4f5bf3474a, 1), // tan(-0.5) = -0.5463024898437905
        (VM_TAN, 0x3ff0000000000000, 0x3ff8eb245cbee3a6, 1), // tan(1.0) = 1.5574077246549023
        (VM_TAN, 0x4000000000000000, 0xc0017af62e0950f8, 1), // tan(2.0) = -2.185039863261519
        (VM_TAN, 0x4008000000000000, 0xbfc23ef71254b86f, 1), // tan(3.0) = -0.1425465430742778
        (VM_TAN, 0x400921f9f01b866e, 0xbec6428a6aa4a2fc, 1), // tan(3.14159) = -2.6535897933620727e-06
        (VM_TAN, 0x412e848000000000, 0xbfd7e9768ab734c0, 1), // tan(1000000.0) = -0.373624453987599
        (VM_TAN, 0x4480f0cf064dd592, 0xbffa0f79c1b6b257, 1), // tan(1e+22) = -1.6287782256068988
        (VM_TAN, 0x7e37e43c8800759c, 0x3ff6be411f37ac77, 1), // tan(1e+300) = 1.4214488238747245
        (VM_TAN, 0x43b0000000000000, 0x3ff7de90e3bee983, 1), // tan(1.152921504606847e+18) = 1.4918373962705977
        (VM_TAN, 0x40c81cd6c8b43958, 0xbfefba5836323a4e, 1), // tan(12345.678) = -0.9914971407432149
        (VM_TAN, 0x3fe921fafc8b007a, 0x3fefffff508da8a4, 1), // tan(0.785398) = 0.9999996732051568
        (VM_TAN, 0x3ff921fb54442d18, 0x434d02967c31cdb5, 1), // tan(1.5707963267948966) = 1.633123935319537e+16
        (VM_TAN, 0xc059200000000000, 0x3f9fb7dcab49130d, 1), // tan(-100.5) = 0.030974815325197236
        (VM_TAN, 0x3ee4f8b588e368f1, 0x3ee4f8b588e6698e, 1), // tan(1e-05) = 1.0000000000333334e-05
        (VM_TAN, 0x3fb999999999999a, 0x3fb9af8877430b80, 1), // tan(0.1) = 0.10033467208545055
        (VM_TAN, 0x3fe6666666666666, 0x3feaf406c2fc78ae, 1), // tan(0.7) = 0.8422883804630794
        (VM_TAN, 0x4202a05f20000000, 0xbfe1de000f443f50, 1), // tan(10000000000.0) = -0.5583496378112418
        (VM_TAN, 0x400921fb54442d18, 0xbca1a62633145c07, 1), // tan(3.141592653589793) = -1.2246467991473532e-16
        (VM_TAN, 0x3fe594af4f0d844d, 0x3fe995054ea00c37, 1), // tan(0.6744) = 0.799441007199113
        (VM_TAN, 0x3fe5c28f5c28f5c3, 0x3fe9e08dd1f8d294, 1), // tan(0.68) = 0.8086613751425653
        (VM_ASIN, 0x3fb999999999999a, 0x3fb9a49276037884, 1), // asin(0.1) = 0.1001674211615598
        (VM_ASIN, 0x3fe0000000000000, 0x3fe0c152382d7366, 1), // asin(0.5) = 0.5235987755982989
        (VM_ASIN, 0x3fe8000000000000, 0x3feb235315c680dc, 1), // asin(0.75) = 0.848062078981481
        (VM_ASIN, 0x3feccccccccccccd, 0x3ff1ea93705fa172, 1), // asin(0.9) = 1.1197695149986342
        (VM_ASIN, 0x3fefae147ae147ae, 0x3ff6de3c6f33d51d, 1), // asin(0.99) = 1.4292568534704693
        (VM_ASIN, 0x3feffffde7210be9, 0x3ff91c306b2c13ad, 1), // asin(0.999999) = 1.5693821131146521
        (VM_ASIN, 0xbfd3333333333333, 0xbfd380159e14f6ff, 1), // asin(-0.3) = -0.3046926540153975
        (VM_ASIN, 0xbfee666666666666, 0xbff40d41159f3409, 1), // asin(-0.95) = -1.253235897503375
        (VM_ASIN, 0x3ee4f8b588e368f1, 0x3ee4f8b588e4e940, 1), // asin(1e-05) = 1.0000000000166668e-05
        (VM_ASIN, 0x3fd0000000000000, 0x3fd02be9ce0b87cd, 1), // asin(0.25) = 0.25268025514207865
        (VM_ASIN, 0x3fe6a09e667f3bcd, 0x3fe921fb54442d19, 1), // asin(0.7071067811865476) = 0.7853981633974484
        (VM_ASIN, 0x3fdf5c28f5c28f5c, 0x3fe0630a0caf011a, 1), // asin(0.49) = 0.5120897529341477
        (VM_ASIN, 0x3fe051eb851eb852, 0x3fe1203bda719c39, 1), // asin(0.51) = 0.5351847902755998
        (VM_ASIN, 0x3fef3b645a1cac08, 0x3ff59ec9e036730a, 1), // asin(0.976) = 1.3512667425449671
        (VM_ACOS, 0x3fb999999999999a, 0x3ff787b22ce3f590, 1), // acos(0.1) = 1.4706289056333368
        (VM_ACOS, 0x3fe0000000000000, 0x3ff0c152382d7366, 1), // acos(0.5) = 1.0471975511965979
        (VM_ACOS, 0x3fe8000000000000, 0x3fe720a392c1d955, 1), // acos(0.75) = 0.7227342478134157
        (VM_ACOS, 0x3feccccccccccccd, 0x3fdcdd9f8f922e98, 1), // acos(0.9) = 0.45102681179626236
        (VM_ACOS, 0x3fefae147ae147ae, 0x3fc21df72882bfd8, 1), // acos(0.99) = 0.1415394733244273
        (VM_ACOS, 0x3feffffde7210be9, 0x3f572ba46065af16, 1), // acos(0.999999) = 0.0014142136802445852
        (VM_ACOS, 0xbfd3333333333333, 0x3ffe0200bbc96ad8, 1), // acos(-0.3) = 1.8754889808102941
        (VM_ACOS, 0xbfee666666666666, 0x4006979e34f1b091, 1), // acos(-0.95) = 2.824032224298272
        (VM_ACOS, 0x3ee4f8b588e368f1, 0x3ff921f0d7e968a6, 1), // acos(1e-05) = 1.5707863267948965
        (VM_ACOS, 0x3fd0000000000000, 0x3ff51700e0c14b25, 1), // acos(0.25) = 1.318116071652818
        (VM_ACOS, 0x3fe6a09e667f3bcd, 0x3fe921fb54442d18, 1), // acos(0.7071067811865476) = 0.7853981633974483
        (VM_ACOS, 0xbfe051eb851eb852, 0x4000d90ca0be7d9a, 1), // acos(-0.51) = 2.1059811170704963
        (VM_ACOS, 0xbfeff7ced916872b, 0x4008c66280ae928e, 1), // acos(-0.999) = 3.09686756642106
        (VM_ACOS, 0x3bc79ca10c924223, 0x3ff921fb54442d18, 1), // acos(1e-20) = 1.5707963267948966
        (VM_ATAN, 0x3fb999999999999a, 0x3fb983e282e2cc4d, 1), // atan(0.1) = 0.09966865249116204
        (VM_ATAN, 0x3fe0000000000000, 0x3fddac670561bb4f, 1), // atan(0.5) = 0.4636476090008061
        (VM_ATAN, 0x3ff0000000000000, 0x3fe921fb54442d18, 1), // atan(1.0) = 0.7853981633974483
        (VM_ATAN, 0x4000000000000000, 0x3ff1b6e192ebbe44, 1), // atan(2.0) = 1.1071487177940904
        (VM_ATAN, 0x4024000000000000, 0x3ff789bd2c160054, 1), // atan(10.0) = 1.4711276743037347
        (VM_ATAN, 0x40f86a0000000000, 0x3ff921f0d7e968a8, 1), // atan(100000.0) = 1.570786326794897
        (VM_ATAN, 0x4415af1d78b58c40, 0x3ff921fb54442d18, 1), // atan(1e+20) = 1.5707963267948966
        (VM_ATAN, 0xbfd3333333333333, 0xbfd2a73a661eaf06, 1), // atan(-0.3) = -0.2914567944778671
        (VM_ATAN, 0xc008000000000000, 0xbff3fc176b7a8560, 1), // atan(-3.0) = -1.2490457723982544
        (VM_ATAN, 0x3fdc000000000000, 0x3fda64eec3cc23fd, 1), // atan(0.4375) = 0.4124104415973873
        (VM_ATAN, 0x3ff3000000000000, 0x3febde70ed439fe7, 1), // atan(1.1875) = 0.870903457075653
        (VM_ATAN, 0x4003800000000000, 0x3ff2e75728833a54, 1), // atan(2.4375) = 1.1814796049617557
        (VM_ATAN, 0x3ee4f8b588e368f1, 0x3ee4f8b588e06854, 1), // atan(1e-05) = 9.999999999666668e-06
        (VM_ATAN, 0x3fe6000000000000, 0x3fe345f01cce37bb, 1), // atan(0.6875) = 0.6022873461349642
        (VM_ATAN, 0x3fdb851eb851eb85, 0x3fd9fd82b5dc5e5f, 1), // atan(0.43) = 0.40609805831761564
        (VM_ATAN, 0x4003851eb851eb85, 0x3ff2e8d07ac56862, 1), // atan(2.44) = 1.1818394466158897
        (VM_ATAN, 0x39b4484bfeebc2a0, 0x39b4484bfeebc2a0, 1), // atan(1e-30) = 1e-30
        (VM_SINH, 0x3fb999999999999a, 0x3fb9a487337b59b3, 2), // sinh(0.1) = 0.10016675001984403
        (VM_SINH, 0x3fe0000000000000, 0x3fe0acd00fe63b97, 2), // sinh(0.5) = 0.5210953054937474
        (VM_SINH, 0x3ff0000000000000, 0x3ff2cd9fc44eb982, 2), // sinh(1.0) = 1.1752011936438014
        (VM_SINH, 0x4000000000000000, 0x400d03cf63b6e19f, 2), // sinh(2.0) = 3.6268604078470186
        (VM_SINH, 0x4014000000000000, 0x40528d0166f07374, 2), // sinh(5.0) = 74.20321057778875
        (VM_SINH, 0x4024000000000000, 0x40c5829dced69992, 2), // sinh(10.0) = 11013.232874703393
        (VM_SINH, 0x4034000000000000, 0x41aceb088b68e804, 2), // sinh(20.0) = 242582597.70489514
        (VM_SINH, 0x4036000000000000, 0x41dab5adb9c43600, 2), // sinh(22.0) = 1792456423.065796
        (VM_SINH, 0x4059000000000000, 0x48e3494a9b171bf5, 2), // sinh(100.0) = 1.3440585709080678e+43
        (VM_SINH, 0x4085e00000000000, 0x7efd945df4f8ec8e, 2), // sinh(700.0) = 5.0711602736750225e+303
        (VM_SINH, 0x3fd058793dd97f63, 0x3fd0861d230af843, 2), // sinh(0.2554) = 0.25818565771249186
        (VM_SINH, 0x3fd3333333333333, 0x3fd37d42af54b926, 2), // sinh(0.3) = 0.3045202934471426
        (VM_SINH, 0x3fe199999999999a, 0x3fe28037cac6a8a7, 2), // sinh(0.55) = 0.5781516037434543
        (VM_SINH, 0x3ee4f8b588e368f1, 0x3ee4f8b588e4e940, 2), // sinh(1e-05) = 1.0000000000166668e-05
        (VM_SINH, 0xbfe6666666666666, 0xbfe8465153d5bdbd, 2), // sinh(-0.7) = -0.7585837018395335
        (VM_SINH, 0xc008000000000000, 0xc0240926e70949ae, 2), // sinh(-3.0) = -10.017874927409903
        (VM_SINH, 0x3f847ae147ae147b, 0x3f847af7a654e9ef, 2), // sinh(0.01) = 0.010000166667500003
        (VM_SINH, 0x40862f3333333333, 0x7fe1fdb71f9335ba, 2), // sinh(709.9) = 1.010701028059782e+308
        (VM_SINH, 0x3e112e0be826d695, 0x3e112e0be826d695, 2), // sinh(1e-09) = 1e-09
        (VM_COSH, 0x3fb999999999999a, 0x3ff0147f40224b38, 2), // cosh(0.1) = 1.0050041680558035
        (VM_COSH, 0x3fe0000000000000, 0x3ff20ac1862ae8d0, 2), // cosh(0.5) = 1.1276259652063807
        (VM_COSH, 0x3ff0000000000000, 0x3ff8b07551d9f550, 2), // cosh(1.0) = 1.5430806348152437
        (VM_COSH, 0x4000000000000000, 0x400e18fa0df2d9bc, 2), // cosh(2.0) = 3.7621956910836314
        (VM_COSH, 0x4014000000000000, 0x40528d6fcbeff3aa, 2), // cosh(5.0) = 74.20994852478785
        (VM_COSH, 0x4024000000000000, 0x40c5829dd053712d, 2), // cosh(10.0) = 11013.232920103323
        (VM_COSH, 0x4034000000000000, 0x41aceb088b68e804, 2), // cosh(20.0) = 242582597.70489514
        (VM_COSH, 0x4036000000000000, 0x41dab5adb9c43600, 2), // cosh(22.0) = 1792456423.065796
        (VM_COSH, 0x4059000000000000, 0x48e3494a9b171bf5, 2), // cosh(100.0) = 1.3440585709080678e+43
        (VM_COSH, 0x4085e00000000000, 0x7efd945df4f8ec8e, 2), // cosh(700.0) = 5.0711602736750225e+303
        (VM_COSH, 0x3fd058793dd97f63, 0x3ff086512addccec, 2), // cosh(0.2554) = 1.0327922510594432
        (VM_COSH, 0x3fd3333333333333, 0x3ff0b9b4e0b6ec4c, 2), // cosh(0.3) = 1.0453385141288605
        (VM_COSH, 0x3fe199999999999a, 0x3ff27b4b9ed39c62, 2), // cosh(0.55) = 1.155101414123941
        (VM_COSH, 0x3ee4f8b588e368f1, 0x3ff0000000036f9c, 2), // cosh(1e-05) = 1.00000000005
        (VM_COSH, 0xbfe6666666666666, 0x3ff4152c1862342f, 2), // cosh(-0.7) = 1.255169005630943
        (VM_COSH, 0xc008000000000000, 0x402422a497d6185e, 2), // cosh(-3.0) = 10.067661995777765
        (VM_COSH, 0x3f847ae147ae147b, 0x3ff000346de27853, 2), // cosh(0.01) = 1.0000500004166681
        (VM_COSH, 0x40862f3333333333, 0x7fe1fdb71f9335ba, 2), // cosh(709.9) = 1.010701028059782e+308
        (VM_COSH, 0x3fe6147ae147ae14, 0x3ff3f65b718f35a0, 2), // cosh(0.69) = 1.2476458011545688
        (VM_TANH, 0x3fb999999999999a, 0x3fb983d7795f413a, 2), // tanh(0.1) = 0.09966799462495582
        (VM_TANH, 0x3fe0000000000000, 0x3fdd9353d7568af3, 2), // tanh(0.5) = 0.46211715726000974
        (VM_TANH, 0x3ff0000000000000, 0x3fe85efab514f394, 2), // tanh(1.0) = 0.7615941559557649
        (VM_TANH, 0x4000000000000000, 0x3feed9505e1bc3d4, 2), // tanh(2.0) = 0.9640275800758169
        (VM_TANH, 0x4014000000000000, 0x3fefff419668df11, 2), // tanh(5.0) = 0.9999092042625951
        (VM_TANH, 0x4024000000000000, 0x3feffffffdc96f35, 2), // tanh(10.0) = 0.9999999958776927
        (VM_TANH, 0x4033e66666666666, 0x3ff0000000000000, 2), // tanh(19.9) = 1.0
        (VM_TANH, 0x3fd058793dd97f63, 0x3fcfff9b3e30a278, 2), // tanh(0.2554) = 0.24998798882122109
        (VM_TANH, 0x3fd3333333333333, 0x3fd2a4dda7d914fa, 2), // tanh(0.3) = 0.2913126124515909
        (VM_TANH, 0x3fe199999999999a, 0x3fe00442f6419203, 2), // tanh(0.55) = 0.5005202111902353
        (VM_TANH, 0x3ee4f8b588e368f1, 0x3ee4f8b588e06854, 2), // tanh(1e-05) = 9.999999999666668e-06
        (VM_TANH, 0xbfe6666666666666, 0xbfe356fb17af2e91, 2), // tanh(-0.7) = -0.6043677771171635
        (VM_TANH, 0xc008000000000000, 0xbfefd77d111a0b00, 2), // tanh(-3.0) = -0.9950547536867305
        (VM_TANH, 0x3f847ae147ae147b, 0x3f847ab48ae4595e, 2), // tanh(0.01) = 0.00999966667999946
        (VM_TANH, 0x3fc3333333333333, 0x3fc30eaa2f235455, 2), // tanh(0.15) = 0.14888503362331798
        (VM_TANH, 0x3fc999999999999a, 0x3fc9439830b3a590, 2), // tanh(0.2) = 0.197375320224904
        (VM_TANH, 0x3fd0a3d70a3d70a4, 0x3fd04660c5098d54, 2), // tanh(0.26) = 0.25429553262639115
        (VM_TANH, 0x3fe147ae147ae148, 0x3fdf8d1d668c4aa5, 2), // tanh(0.54) = 0.49298796667532435
        (VM_TANH, 0x3fe1eb851eb851ec, 0x3fe04159e3ace145, 2), // tanh(0.56) = 0.5079774328978962
        (VM_TANH, 0x3e112e0be826d695, 0x3e112e0be826d695, 2), // tanh(1e-09) = 1e-09
        (VM_TANH, 0x4035000000000000, 0x3ff0000000000000, 2), // tanh(21.0) = 1.0
        (VM_ASINH, 0x3fb999999999999a, 0x3fb98eb9e7e5fc3e, 2), // asinh(0.1) = 0.09983407889920756
        (VM_ASINH, 0x3fd3333333333333, 0x3fd2ec4ea565fcc1, 2), // asinh(0.3) = 0.29567304756342244
        (VM_ASINH, 0x3fe0000000000000, 0x3fdecc2caec5160a, 2), // asinh(0.5) = 0.48121182505960347
        (VM_ASINH, 0x3ff0000000000000, 0x3fec34366179d427, 2), // asinh(1.0) = 0.881373587019543
        (VM_ASINH, 0x4000000000000000, 0x3ff719218313d087, 2), // asinh(2.0) = 1.4436354751788103
        (VM_ASINH, 0x4024000000000000, 0x4007fc5c506d2bdb, 2), // asinh(10.0) = 2.99822295029797
        (VM_ASINH, 0x4202a05f20000000, 0x4037b810429a7c2a, 2), // asinh(10000000000.0) = 23.7189981105004
        (VM_ASINH, 0x46293e5939a08cea, 0x4051715325f7f438, 2), // asinh(1e+30) = 69.77069997038132
        (VM_ASINH, 0xbfd0000000000000, 0xbfcfacfb2399e637, 2), // asinh(-0.25) = -0.24746646154726346
        (VM_ASINH, 0x3ee4f8b588e368f1, 0x3ee4f8b588e1e8a2, 2), // asinh(1e-05) = 9.999999999833334e-06
        (VM_ASINH, 0x3fc0000000000000, 0x3fbfead0b6996973, 2), // asinh(0.125) = 0.12467674692144275
        (VM_ASINH, 0x3ffe666666666666, 0x3ff65e3388a03dc2, 2), // asinh(1.9) = 1.3979983651114343
        (VM_ASINH, 0x4000cccccccccccd, 0x3ff7ccbacd8bc4e5, 2), // asinh(2.1) = 1.4874828366412711
        (VM_ASINH, 0x4190b07600000000, 0x4032c1d4c719a1e8, 2), // asinh(70000000.0) = 18.75715298057358
        (VM_ASINH, 0x7e37e43c8800759c, 0x40859bbfd8b83e44, 2), // asinh(1e+300) = 691.4686750787737
        (VM_ACOSH, 0x3ff00068db8bac71, 0x3f8cf67d7ec0d4e5, 2), // acosh(1.0001) = 0.014142017775251545
        (VM_ACOSH, 0x3ff199999999999a, 0x3fdc636c1a882f2c, 2), // acosh(1.1) = 0.4435682543851154
        (VM_ACOSH, 0x3ff8000000000000, 0x3feecc2caec5160a, 2), // acosh(1.5) = 0.9624236501192069
        (VM_ACOSH, 0x4000000000000000, 0x3ff5124271980435, 2), // acosh(2.0) = 1.3169578969248168
        (VM_ACOSH, 0x4024000000000000, 0x4007f21ed1ce05d4, 2), // acosh(10.0) = 2.993222846126381
        (VM_ACOSH, 0x4202a05f20000000, 0x4037b810429a7c2a, 2), // acosh(10000000000.0) = 23.7189981105004
        (VM_ACOSH, 0x46293e5939a08cea, 0x4051715325f7f438, 2), // acosh(1e+30) = 69.77069997038132
        (VM_ACOSH, 0x3ff0cccccccccccd, 0x3fd427ba2a941d48, 2), // acosh(1.05) = 0.314924756603848
        (VM_ACOSH, 0x3ffe666666666666, 0x3ff41d795efea940, 2), // acosh(1.9) = 1.2571958266003804
        (VM_ACOSH, 0x4000cccccccccccd, 0x3ff5f73b26689535, 2), // acosh(2.1) = 1.37285914424258
        (VM_ACOSH, 0x4190b07600000000, 0x4032c1d4c719a1e8, 2), // acosh(70000000.0) = 18.75715298057358
        (VM_ACOSH, 0x7e37e43c8800759c, 0x40859bbfd8b83e44, 2), // acosh(1e+300) = 691.4686750787737
        (VM_ACOSH, 0x3ff2000000000000, 0x3fdfacfb2399e637, 2), // acosh(1.125) = 0.4949329230945269
        (VM_ATANH, 0x3fb999999999999a, 0x3fb9af93cd234412, 2), // atanh(0.1) = 0.10033534773107558
        (VM_ATANH, 0x3fd3333333333333, 0x3fd3cf2b50617c95, 2), // atanh(0.3) = 0.3095196042031117
        (VM_ATANH, 0x3fe0000000000000, 0x3fe193ea7aad030b, 2), // atanh(0.5) = 0.5493061443340549
        (VM_ATANH, 0x3fe8000000000000, 0x3fef2272ae325a57, 2), // atanh(0.75) = 0.9729550745276566
        (VM_ATANH, 0x3feccccccccccccd, 0x3ff78e360604b32d, 2), // atanh(0.9) = 1.4722194895832204
        (VM_ATANH, 0x3feff7ced916872b, 0x400e66cfde9c7c2d, 2), // atanh(0.999) = 3.8002011672501994
        (VM_ATANH, 0xbfd0000000000000, 0xbfd058aefa811452, 2), // atanh(-0.25) = -0.25541281188299536
        (VM_ATANH, 0x3ee4f8b588e368f1, 0x3ee4f8b588e6698e, 2), // atanh(1e-05) = 1.0000000000333334e-05
        (VM_ATANH, 0x3fdf5c28f5c28f5c, 0x3fe1276801cd9ab5, 2), // atanh(0.49) = 0.5360603366105666
        (VM_ATANH, 0x3fe051eb851eb852, 0x3fe201e1dc7b4c9e, 2), // atanh(0.51) = 0.5627297693521489
        (VM_ATANH, 0x3d719799812dea11, 0x3d719799812dea11, 2), // atanh(1e-12) = 1e-12
        (VM_ATANH, 0x3fefffffca501acb, 0x4020cfad9b61ff69, 2), // atanh(0.9999999) = 8.40562139102231
        (VM_ERF, 0x3fb999999999999a, 0x3fbcca5ea24fb334, 1),   // erf(0.1) = 0.1124629160182849
        (VM_ERF, 0x3fe0000000000000, 0x3fe0a7ef5c18edd2, 1),   // erf(0.5) = 0.5204998778130465
        (VM_ERF, 0x3feae147ae147ae1, 0x3fe87c0c914f1a58, 1),   // erf(0.84) = 0.7651427114549945
        (VM_ERF, 0x3ff0000000000000, 0x3feaf767a741088b, 1),   // erf(1.0) = 0.8427007929497149
        (VM_ERF, 0x3ff4000000000000, 0x3fed8865d98abe01, 1),   // erf(1.25) = 0.9229001282564583
        (VM_ERF, 0x4000000000000000, 0x3fefd9ae142795e3, 1),   // erf(2.0) = 0.9953222650189527
        (VM_ERF, 0x4006cccccccccccd, 0x3fefff8b47c320c2, 1),   // erf(2.85) = 0.9999443437200386
        (VM_ERF, 0x4008000000000000, 0x3fefffd1ac4135f9, 1),   // erf(3.0) = 0.9999779095030014
        (VM_ERF, 0x4010000000000000, 0x3feffffff7b91176, 1),   // erf(4.0) = 0.9999999845827421
        (VM_ERF, 0x401799999999999a, 0x3fefffffffffffff, 1),   // erf(5.9) = 0.9999999999999999
        (VM_ERF, 0x401a000000000000, 0x3ff0000000000000, 1),   // erf(6.5) = 1.0
        (VM_ERF, 0xbfe6666666666666, 0xbfe5b08c21171646, 1),   // erf(-0.7) = -0.6778011938374184
        (VM_ERF, 0x3ee4f8b588e368f1, 0x3ee7a9f084b5e44c, 1),   // erf(1e-05) = 1.1283791670579e-05
        (VM_ERF, 0x3fd3333333333333, 0x3fd50838881dea0f, 1),   // erf(0.3) = 0.3286267594591274
        (VM_ERF, 0x3feb000000000000, 0x3fe88d1cd474a2e0, 1),   // erf(0.84375) = 0.7672256612323416
        (VM_ERF, 0x3ff3d70a3d70a3d7, 0x3fed74c749cb9bf0, 1),   // erf(1.24) = 0.9205051842990297
        (VM_ERF, 0x3ff428f5c28f5c29, 0x3fed9b8867e51082, 1),   // erf(1.26) = 0.9252359418101295
        (VM_ERF, 0x4006e147ae147ae1, 0x3fefff921b559db7, 1),   // erf(2.86) = 0.9999475988269556
        (VM_ERF, 0x3e112e0be826d695, 0x3e1362a9ffc610dd, 1), // erf(1e-09) = 1.1283791670955127e-09
        (VM_ERF, 0x4017f5c28f5c28f6, 0x3ff0000000000000, 1), // erf(5.99) = 1.0
        (VM_CBRT, 0x4000000000000000, 0x3ff428a2f98d728b, 1), // cbrt(2.0) = 1.2599210498948732
        (VM_CBRT, 0x4008000000000000, 0x3ff7137449123ef6, 1), // cbrt(3.0) = 1.4422495703074083
        (VM_CBRT, 0x4020000000000000, 0x4000000000000000, 1), // cbrt(8.0) = 2.0
        (VM_CBRT, 0x403b000000000000, 0x4008000000000000, 1), // cbrt(27.0) = 3.0
        (VM_CBRT, 0x4202a05f20000000, 0x40a0d4de8fb11dad, 1), // cbrt(10000000000.0) = 2154.4346900318837
        (VM_CBRT, 0x3ddb7cdfd9d7bdbb, 0x3f3e6b4b396428e8, 1), // cbrt(1e-10) = 0.0004641588833612779
        (VM_CBRT, 0x3fb999999999999a, 0x3fddb4c7760bcff3, 1), // cbrt(0.1) = 0.4641588833612779
        (VM_CBRT, 0x4014000000000000, 0x3ffb5c0fbcfec4d4, 1), // cbrt(5.0) = 1.709975946676697
        (VM_CBRT, 0x405edd2f1a9fbe77, 0x4013ead4f488ee35, 1), // cbrt(123.456) = 4.979327984674048
        (VM_CBRT, 0xbf50624dd2f1a9fc, 0xbfb999999999999a, 1), // cbrt(-0.001) = -0.1
        (VM_CBRT, 0x7e37e43c8800759c, 0x54b249ad2594c37d, 1), // cbrt(1e+300) = 1e+100
        (VM_CBRT, 0x01a56e1fc2f8f359, 0x2b2bff2ee48e0530, 1), // cbrt(1e-300) = 1e-100
        (VM_CBRT, 0x401c000000000000, 0x3ffe9b5dba58189e, 1), // cbrt(7.0) = 1.9129311827723892
        (VM_CBRT, 0x0000000000000001, 0x2990000000000000, 1), // cbrt(5e-324) = 1.7031839360032603e-108
        (VM_CBRT, 0x7fefffffffffffff, 0x554428a2f98d728b, 1), // cbrt(1.7976931348623157e+308) = 5.643803094122362e+102
        (VM_CBRT, 0xc050000000000000, 0xc010000000000000, 1), // cbrt(-64.0) = -4.0
        (VM_CBRT, 0x3fe0000000000000, 0x3fe965fea53d6e3d, 1), // cbrt(0.5) = 0.7937005259840998
        (VM_CBRT, 0x408f400000000000, 0x4024000000000000, 1), // cbrt(1000.0) = 10.0
        (VM_EXPM1, 0x3ddb7cdfd9d7bdbb, 0x3ddb7cdfd9dda4e3, 1), // expm1(1e-10) = 1.00000000005e-10
        (VM_EXPM1, 0x3fb999999999999a, 0x3fbaec7b35a00d3a, 1), // expm1(0.1) = 0.10517091807564763
        (VM_EXPM1, 0x3fe0000000000000, 0x3fe4c2531c3c0d38, 1), // expm1(0.5) = 0.6487212707001282
        (VM_EXPM1, 0x3ff0000000000000, 0x3ffb7e151628aed3, 1), // expm1(1.0) = 1.7182818284590453
        (VM_EXPM1, 0x4000000000000000, 0x40198e64b8d4ddae, 1), // expm1(2.0) = 6.38905609893065
        (VM_EXPM1, 0x4024000000000000, 0x40d5825dcf950560, 1), // expm1(10.0) = 22025.465794806718
        (VM_EXPM1, 0x4044000000000000, 0x438a220d397972eb, 1), // expm1(40.0) = 2.3538526683702e+17
        (VM_EXPM1, 0x404e000000000000, 0x45579dbc9dc53c66, 1), // expm1(60.0) = 1.1420073898156842e+26
        (VM_EXPM1, 0x4085e00000000000, 0x7f0d945df4f8ec8e, 1), // expm1(700.0) = 1.0142320547350045e+304
        (VM_EXPM1, 0xbfb999999999999a, 0xbfb85c933156a62c, 1), // expm1(-0.1) = -0.09516258196404043
        (VM_EXPM1, 0xbff0000000000000, 0xbfe43a54e4e98864, 1), // expm1(-1.0) = -0.6321205588285577
        (VM_EXPM1, 0xc024000000000000, 0xbfefffa0ca192a6e, 1), // expm1(-10.0) = -0.9999546000702375
        (VM_EXPM1, 0xc044000000000000, 0xbff0000000000000, 1), // expm1(-40.0) = -1.0
        (VM_EXPM1, 0x3f50624dd2f1a9fc, 0x3f506466dfb8cf3a, 1), // expm1(0.001) = 0.0010005001667083417
        (VM_EXPM1, 0x3fd62e33eff19503, 0x3fda82644e282273, 1), // expm1(0.34657) = 0.4142084849595797
        (VM_EXPM1, 0x3ff0a3d70a3d70a4, 0x3ffd44790f5ed3cf, 1), // expm1(1.04) = 1.8292170143515596
        (VM_EXPM1, 0x3fd6666666666666, 0x3fdad200b20177b2, 1), // expm1(0.35) = 0.41906754859325723
        (VM_EXPM1, 0xbfd6666666666666, 0xbfd2e663ed31c11e, 1), // expm1(-0.35) = -0.29531191028128656
        (VM_EXPM1, 0x3fe6147ae147ae14, 0x3fefcc848499ccb6, 1), // expm1(0.69) = 0.9937155332430823
        (VM_EXPM1, 0xbfe6147ae147ae14, 0xbfdfe62d7cb9ec68, 1), // expm1(-0.69) = -0.49842393093394444
        (VM_EXPM1, 0x3ff0cccccccccccd, 0x3ffdb8f060f75892, 1), // expm1(1.05) = 1.8576511180631639
        (VM_EXPM1, 0xbff0cccccccccccd, 0xbfe4cd4f597dd0c5, 1), // expm1(-1.05) = -0.6500622508888446
        (VM_EXPM1, 0x4039000000000000, 0x4230c3d3920862c9, 1), // expm1(25.0) = 72004899336.38588
        (VM_EXPM1, 0x3bc79ca10c924223, 0x3bc79ca10c924223, 1), // expm1(1e-20) = 1e-20
        (VM_EXPM1, 0x4086280000000000, 0x7fdd422d2be5dc9b, 1), // expm1(709.0) = 8.218407461554972e+307
        (VM_LOG1P, 0x3ddb7cdfd9d7bdbb, 0x3ddb7cdfd9d1d693, 1), // log1p(1e-10) = 9.999999999500001e-11
        (VM_LOG1P, 0x3fb999999999999a, 0x3fb8663f793c46c7, 1), // log1p(0.1) = 0.09531017980432487
        (VM_LOG1P, 0x3fe0000000000000, 0x3fd9f323ecbf984c, 1), // log1p(0.5) = 0.4054651081081644
        (VM_LOG1P, 0x3ff0000000000000, 0x3fe62e42fefa39ef, 1), // log1p(1.0) = 0.6931471805599453
        (VM_LOG1P, 0x4000000000000000, 0x3ff193ea7aad030b, 1), // log1p(2.0) = 1.0986122886681098
        (VM_LOG1P, 0x4024000000000000, 0x40032ee3b77f374c, 1), // log1p(10.0) = 2.3978952727983707
        (VM_LOG1P, 0x4202a05f20000000, 0x4037069e2aa3184e, 1), // log1p(10000000000.0) = 23.025850930040455
        (VM_LOG1P, 0xbfb999999999999a, 0xbfbaf8e8210a415e, 1), // log1p(-0.1) = -0.10536051565782631
        (VM_LOG1P, 0xbfe0000000000000, 0xbfe62e42fefa39ef, 1), // log1p(-0.5) = -0.6931471805599453
        (VM_LOG1P, 0xbfeccccccccccccd, 0xc0026bb1bbb55516, 1), // log1p(-0.9) = -2.302585092994046
        (VM_LOG1P, 0x7e37e43c8800759c, 0x4085963447f87fb5, 1), // log1p(1e+300) = 690.7755278982137
        (VM_LOG1P, 0x3fda826aa8eb4635, 0x3fd62e386e3de810, 1), // log1p(0.41421) = 0.3465710712986274
        (VM_LOG1P, 0xbfd2beb5b2d4d402, 0xbfd62e2fe73833b5, 1), // log1p(-0.29289) = -0.3465690382006941
        (VM_LOG1P, 0x3fdae147ae147ae1, 0x3fd6712984ec8f15, 1), // log1p(0.42) = 0.35065687161316933
        (VM_LOG1P, 0xbfd3333333333333, 0xbfd6d3c324e13f4e, 1), // log1p(-0.3) = -0.35667494393873234
        (VM_LOG1P, 0x3bc79ca10c924223, 0x3bc79ca10c924223, 1), // log1p(1e-20) = 1e-20
        (VM_LOG1P, 0x4341c37937e08000, 0x40426bb1bbb55516, 1), // log1p(1e+16) = 36.841361487904734
        (VM_LOG1P, 0x3ea0c6f7a0b5ed8d, 0x3ea0c6f75a578912, 1), // log1p(5e-07) = 4.999998750000417e-07
        (VM_LOG1P, 0xbfeff7ced916872b, 0xc01ba18a998fff9f, 1), // log1p(-0.999) = -6.907755278982136
        (VM_EXP2, 0x3fe0000000000000, 0x3ff6a09e667f3bcd, 1),  // exp2(0.5) = 1.4142135623730951
        (VM_EXP2, 0x3ff8000000000000, 0x4006a09e667f3bcd, 1),  // exp2(1.5) = 2.8284271247461903
        (VM_EXP2, 0x4024800000000000, 0x409306fe0a31b715, 1),  // exp2(10.25) = 1217.7480857627863
        (VM_EXP2, 0x4059066666666666, 0x463125fbee250651, 1), // exp2(100.1) = 1.3586342730928144e+30
        (VM_EXP2, 0x408f45999999999a, 0x7e79fdf8bcce5424, 1), // exp2(1000.7) = 1.7406708678068106e+301
        (VM_EXP2, 0xc08f45999999999a, 0x0163b2c47bff827a, 1), // exp2(-1000.7) = 5.744911450491315e-302
        (VM_EXP2, 0x408fff3333333333, 0x7feddb680117aa8e, 1), // exp2(1023.9) = 1.6773070034857416e+308
        (VM_EXP2, 0xc090ca0000000000, 0x0000000000000001, 1), // exp2(-1074.5) = 5e-324
        (VM_EXP2, 0x3ddb7cdfd9d7bdbb, 0x3ff000000004c366, 1), // exp2(1e-10) = 1.0000000000693148
        (VM_EXP2, 0xbfd3333333333333, 0x3fe9fdf8bcce533e, 1), // exp2(-0.3) = 0.8122523963562356
        (VM_EXP2, 0x3fbf9acffa7eb6bf, 0x3ff16df11dc91d2d, 1), // exp2(0.123456) = 1.089341274595266
        (VM_EXP2, 0x4008000000000000, 0x4020000000000000, 1), // exp2(3.0) = 8.0
        (VM_EXP2, 0xbf88000000000000, 0x3fefbdba3692d514, 1), // exp2(-0.01171875) = 0.9919100824251097
        (VM_EXP2, 0x3f60000000000000, 0x3ff0058c86da1c0a, 1), // exp2(0.001953125) = 1.0013547198921082
        (VM_EXP2, 0x407fffd70a3d70a4, 0x5fefc769e9b9c396, 1), // exp2(511.99) = 1.3315193435487256e+154
        (VM_EXP2, 0xc08ff40000000000, 0x000b504f333f9de6, 1), // exp2(-1022.5) = 1.5733648139913585e-308
        (VM_EXP2, 0x3bc79ca10c924223, 0x3ff0000000000000, 1), // exp2(1e-20) = 1.0
        (VM_EXP2, 0x3fe0000035afe535, 0x3ff6a09e80cf8439, 1), // exp2(0.5000001) = 1.4142136603989128
        (VM_LOG2, 0x4008000000000000, 0x3ff95c01a39fbd68, 1), // log2(3.0) = 1.584962500721156
        (VM_LOG2, 0x4024000000000000, 0x400a934f0979a371, 1), // log2(10.0) = 3.321928094887362
        (VM_LOG2, 0x3fb999999999999a, 0xc00a934f0979a371, 1), // log2(0.1) = -3.321928094887362
        (VM_LOG2, 0x4202a05f20000000, 0x40409c1165ec0627, 1), // log2(10000000000.0) = 33.219280948873624
        (VM_LOG2, 0x3ddb7cdfd9d7bdbb, 0xc0409c1165ec0627, 1), // log2(1e-10) = -33.219280948873624
        (VM_LOG2, 0x3ff8000000000000, 0x3fe2b803473f7ad1, 1), // log2(1.5) = 0.5849625007211562
        (VM_LOG2, 0x3fe8000000000000, 0xbfda8ff971810a5e, 1), // log2(0.75) = -0.4150374992788438
        (VM_LOG2, 0x7e37e43c8800759c, 0x408f24a09f1a8b89, 1), // log2(1e+300) = 996.5784284662087
        (VM_LOG2, 0x0000000000000001, 0xc090c80000000000, 1), // log2(5e-324) = -1074.0
        (VM_LOG2, 0x4020000000000000, 0x4008000000000000, 1), // log2(8.0) = 3.0
        (VM_LOG2, 0x4090000000000000, 0x4024000000000000, 1), // log2(1024.0) = 10.0
        (VM_LOG2, 0x3fe6a0902de00d1b, 0xbfe0001d03eb60d8, 1), // log2(0.7071) = -0.5000138355781258
        (VM_LOG2, 0x3ff6a0f9096bb98c, 0x3fe000b8eafd2ee9, 1), // log2(1.4143) = 0.5000881757382248
        (VM_LOG2, 0x0010000000000000, 0xc08ff00000000000, 1), // log2(2.2250738585072014e-308) = -1022.0
        (VM_LOG2, 0x3ff000001ad7f29b, 0x3e835d0fea5fccb7, 1), // log2(1.0000001) = 1.4426949695965583e-07
        (VM_LOG10, 0x4000000000000000, 0x3fd34413509f79ff, 1), // log10(2.0) = 0.3010299956639812
        (VM_LOG10, 0x4008000000000000, 0x3fde8927964fd5fd, 1), // log10(3.0) = 0.47712125471966244
        (VM_LOG10, 0x408f400000000000, 0x4008000000000000, 1), // log10(1000.0) = 3.0
        (VM_LOG10, 0x3ddb7cdfd9d7bdbb, 0xc024000000000000, 1), // log10(1e-10) = -10.0
        (VM_LOG10, 0x3fe0000000000000, 0xbfd34413509f79ff, 1), // log10(0.5) = -0.3010299956639812
        (VM_LOG10, 0x40fe240000000000, 0x40145db55fe4b478, 1), // log10(123456.0) = 5.091512201627772
        (VM_LOG10, 0x7e37e43c8800759c, 0x4072c00000000000, 1), // log10(1e+300) = 300.0
        (VM_LOG10, 0x0000000000000001, 0xc07434e6420f4374, 1), // log10(5e-324) = -323.3062153431158
        (VM_LOG10, 0x401c000000000000, 0x3feb0b0b0b78cc3f, 1), // log10(7.0) = 0.8450980400142568
        (VM_LOG10, 0x4059000000000000, 0x4000000000000000, 1), // log10(100.0) = 2.0
        (VM_LOG10, 0x3f50624dd2f1a9fc, 0xc008000000000000, 1), // log10(0.001) = -3.0
        (VM_LOG10, 0x3ff000001ad7f29b, 0x3e6750e5ca0b1098, 1), // log10(1.0000001) = 4.3429446044209946e-08
        (VM_LOG10, 0x40094c583ada5b53, 0x3fe0000000000000, 1), // log10(3.1622776601683795) = 0.5
        (VM_EXP10, 0x3fe0000000000000, 0x40094c583ada5b53, 3), // exp10(0.5) = 3.1622776601683795
        (VM_EXP10, 0x3ff8000000000000, 0x403f9f6e4990f227, 3), // exp10(1.5) = 31.622776601683793
        (VM_EXP10, 0x4000000000000000, 0x4059000000000000, 3), // exp10(2.0) = 100.0
        (VM_EXP10, 0xc000000000000000, 0x3f847ae147ae147b, 3), // exp10(-2.0) = 0.01
        (VM_EXP10, 0x402f000000000000, 0x43267826493b2177, 3), // exp10(15.5) = 3162277660168379.5
        (VM_EXP10, 0x4030800000000000, 0x435c162fdb89e9d4, 3), // exp10(16.5) = 3.162277660168379e+16
        (VM_EXP10, 0x4059133333333333, 0x54c23e96113fbf71, 3), // exp10(100.3) = 1.9952623149688666e+100
        (VM_EXP10, 0xc059133333333333, 0x2b1c1033900be810, 3), // exp10(-100.3) = 5.011872336272756e-101
        (VM_EXP10, 0x4072c33333333333, 0x7e42eeca6f084bf8, 3), // exp10(300.2) = 1.584893192461072e+300
        (VM_EXP10, 0xc072c33333333333, 0x019b0b0542a00de8, 3), // exp10(-300.2) = 6.309573444802098e-301
        (VM_EXP10, 0x3fbf7ced916872b0, 0x3ff53d01f88d7444, 3), // exp10(0.123) = 1.3273944577297394
        (VM_EXP10, 0x3ddb7cdfd9d7bdbb, 0x3ff00000000fd2c0, 3), // exp10(1e-10) = 1.0000000002302585
        (VM_EXP10, 0x402e000000000000, 0x430c6bf526340000, 3), // exp10(15.0) = 1000000000000000.0
        (VM_EXP10, 0xc02e000000000000, 0x3cd203af9ee75616, 3), // exp10(-15.0) = 1e-15
        (VM_EXP10, 0x3fefff2e48e8a71e, 0x4023fed23af5a1be, 3), // exp10(0.9999) = 9.997697679981567
        (VM_EXP10, 0x4073380000000000, 0x7fc68423588e3be1, 3), // exp10(307.5) = 3.1622776601683796e+307
        (VM_EXP10, 0xc073380000000000, 0x0016bd3d7ae36ea2, 3), // exp10(-307.5) = 3.162277660168379e-308
        (VM_EXP10, 0x4008000000000000, 0x408f400000000000, 3), // exp10(3.0) = 1000.0
    ];
    const REF2: &[(i64, u64, u64, u64, u64)] = &[
        (
            VM2_POW,
            0x4000000000000000,
            0x3fe0000000000000,
            0x3ff6a09e667f3bcd,
            1,
        ), // pow(2.0, 0.5) = 1.4142135623730951
        (
            VM2_POW,
            0x4000000000000000,
            0x4024000000000000,
            0x4090000000000000,
            1,
        ), // pow(2.0, 10.0) = 1024.0
        (
            VM2_POW,
            0x4024000000000000,
            0xc008000000000000,
            0x3f50624dd2f1a9fc,
            1,
        ), // pow(10.0, -3.0) = 0.001
        (
            VM2_POW,
            0x3ff8000000000000,
            0x4004000000000000,
            0x40060b9fd68a4554,
            1,
        ), // pow(1.5, 2.5) = 2.7556759606310752
        (
            VM2_POW,
            0x3fb999999999999a,
            0x3fb999999999999a,
            0x3fe96b230bcdc434,
            1,
        ), // pow(0.1, 0.1) = 0.7943282347242815
        (
            VM2_POW,
            0x405edd2f1a9fbe77,
            0x401f8f5c28f5c28f,
            0x435c37e53d5e93d2,
            1,
        ), // pow(123.456, 7.89) = 3.1771028258180936e+16
        (
            VM2_POW,
            0x4000000000000000,
            0x408f440000000000,
            0x7e76a09e667f3bcd,
            1,
        ), // pow(2.0, 1000.5) = 1.5153420044823246e+301
        (
            VM2_POW,
            0x3fe0000000000000,
            0xc08f440000000000,
            0x7e76a09e667f3bcd,
            1,
        ), // pow(0.5, -1000.5) = 1.5153420044823246e+301
        (
            VM2_POW,
            0x4008000000000000,
            0xbfe0000000000000,
            0x3fe279a74590331c,
            1,
        ), // pow(3.0, -0.5) = 0.5773502691896257
        (
            VM2_POW,
            0x4202a05f20000000,
            0x3f50624dd2f1a9fc,
            0x3ff05f687901262c,
            1,
        ), // pow(10000000000.0, 0.001) = 1.023292992280754
        (
            VM2_POW,
            0x401c000000000000,
            0x4008000000000000,
            0x4075700000000000,
            1,
        ), // pow(7.0, 3.0) = 343.0
        (
            VM2_POW,
            0x3ff000001ad7f29b,
            0x416312d000000000,
            0x4005bf0a790ce6f2,
            1,
        ), // pow(1.0000001, 10000000.0) = 2.7182816941320818
        (
            VM2_POW,
            0x3fefffffca501acb,
            0x416312d000000000,
            0x3fd78b5622a21056,
            1,
        ), // pow(0.9999999, 10000000.0) = 0.367879422971105
        (
            VM2_POW,
            0xc000000000000000,
            0x4008000000000000,
            0xc020000000000000,
            1,
        ), // pow(-2.0, 3.0) = -8.0
        (
            VM2_POW,
            0xc000000000000000,
            0x4010000000000000,
            0x4030000000000000,
            1,
        ), // pow(-2.0, 4.0) = 16.0
        (
            VM2_POW,
            0x7e37e43c8800759c,
            0x3ff004189374bc6a,
            0x7e47d5bf80286965,
            1,
        ), // pow(1e+300, 1.001) = 1.995262314968728e+300
        (
            VM2_POW,
            0x4004000000000000,
            0xbfe8000000000000,
            0x3fe0185b9cde883a,
            1,
        ), // pow(2.5, -0.75) = 0.5029733718731741
        (
            VM2_POW,
            0x3fd3333333333333,
            0x4059000000000000,
            0x3513beceda168ae7,
            1,
        ), // pow(0.3, 100.0) = 5.153775207320094e-53
        (
            VM2_POW,
            0x3ff000000006df38,
            0x41cdcd6500000000,
            0x3ff1aec7b5ce0287,
            1,
        ), // pow(1.0000000001, 1000000000.0) = 1.105170927214347
        (
            VM2_POW,
            0x4024000000000000,
            0x4073400000000000,
            0x7fe1ccf385ebc8a0,
            1,
        ), // pow(10.0, 308.0) = 1e+308
        (
            VM2_POW,
            0x4000000000000000,
            0xc090c80000000000,
            0x0000000000000001,
            1,
        ), // pow(2.0, -1074.0) = 5e-324
        (
            VM2_POW,
            0x3ff3333333333333,
            0x3ee4f8b588e368f1,
            0x3ff00001e96a88b6,
            1,
        ), // pow(1.2, 1e-05) = 1.00000182321723
        (
            VM2_ATAN2,
            0x3ff0000000000000,
            0x4000000000000000,
            0x3fddac670561bb4f,
            1,
        ), // atan2(1.0, 2.0) = 0.4636476090008061
        (
            VM2_ATAN2,
            0x4000000000000000,
            0x3ff0000000000000,
            0x3ff1b6e192ebbe44,
            1,
        ), // atan2(2.0, 1.0) = 1.1071487177940904
        (
            VM2_ATAN2,
            0xbff0000000000000,
            0x4000000000000000,
            0xbfddac670561bb4f,
            1,
        ), // atan2(-1.0, 2.0) = -0.4636476090008061
        (
            VM2_ATAN2,
            0x3ff0000000000000,
            0xc000000000000000,
            0x40056c6e7397f5ae,
            1,
        ), // atan2(1.0, -2.0) = 2.677945044588987
        (
            VM2_ATAN2,
            0xbff0000000000000,
            0xc000000000000000,
            0xc0056c6e7397f5ae,
            1,
        ), // atan2(-1.0, -2.0) = -2.677945044588987
        (
            VM2_ATAN2,
            0x4008000000000000,
            0x4010000000000000,
            0x3fe4978fa3269ee1,
            1,
        ), // atan2(3.0, 4.0) = 0.6435011087932844
        (
            VM2_ATAN2,
            0x3fe0000000000000,
            0xbfe0000000000000,
            0x4002d97c7f3321d2,
            1,
        ), // atan2(0.5, -0.5) = 2.356194490192345
        (
            VM2_ATAN2,
            0x4202a05f20000000,
            0x3ff0000000000000,
            0x3ff921fb543d4de0,
            1,
        ), // atan2(10000000000.0, 1.0) = 1.5707963266948965
        (
            VM2_ATAN2,
            0x3ff0000000000000,
            0x4202a05f20000000,
            0x3ddb7cdfd9d7bdbb,
            1,
        ), // atan2(1.0, 10000000000.0) = 1e-10
        (
            VM2_ATAN2,
            0xc008000000000000,
            0x3fb999999999999a,
            0xbff8997fbb8b19c0,
            1,
        ), // atan2(-3.0, 0.1) = -1.5374753309166493
        (
            VM2_ATAN2,
            0x01a56e1fc2f8f359,
            0x3ff0000000000000,
            0x01a56e1fc2f8f359,
            1,
        ), // atan2(1e-300, 1.0) = 1e-300
        (
            VM2_ATAN2,
            0x3ff0000000000000,
            0x01a56e1fc2f8f359,
            0x3ff921fb54442d18,
            1,
        ), // atan2(1.0, 1e-300) = 1.5707963267948966
        (
            VM2_ATAN2,
            0xbfe0000000000000,
            0xbfe0000000000000,
            0xc002d97c7f3321d2,
            1,
        ), // atan2(-0.5, -0.5) = -2.356194490192345
        (
            VM2_ATAN2,
            0x3fd3333333333333,
            0x3ff0000000000000,
            0x3fd2a73a661eaf06,
            1,
        ), // atan2(0.3, 1.0) = 0.2914567944778671
        (
            VM2_HYPOT,
            0x4008000000000000,
            0x4010000000000000,
            0x4014000000000000,
            1,
        ), // hypot(3.0, 4.0) = 5.0
        (
            VM2_HYPOT,
            0x7e37e43c8800759c,
            0x7e37e43c8800759c,
            0x7e40e4d50f99b211,
            1,
        ), // hypot(1e+300, 1e+300) = 1.4142135623730952e+300
        (
            VM2_HYPOT,
            0x01a56e1fc2f8f359,
            0x01a56e1fc2f8f359,
            0x01ae4e8d12762225,
            1,
        ), // hypot(1e-300, 1e-300) = 1.414213562373095e-300
        (
            VM2_HYPOT,
            0x3ff0000000000000,
            0x3bc79ca10c924223,
            0x3ff0000000000000,
            1,
        ), // hypot(1.0, 1e-20) = 1.0
        (
            VM2_HYPOT,
            0x3fb999999999999a,
            0x3fc999999999999a,
            0x3fcc9f25c5bfedda,
            1,
        ), // hypot(0.1, 0.2) = 0.223606797749979
        (
            VM2_HYPOT,
            0x6974e718d7d7625a,
            0x16687e92154ef7ac,
            0x6974e718d7d7625a,
            1,
        ), // hypot(1e+200, 1e-200) = 1e+200
        (
            VM2_HYPOT,
            0x0000000000000001,
            0x0000000000000001,
            0x0000000000000001,
            1,
        ), // hypot(5e-324, 5e-324) = 5e-324
        (
            VM2_HYPOT,
            0x7fee42d130773b76,
            0x3ff0000000000000,
            0x7fee42d130773b76,
            1,
        ), // hypot(1.7e+308, 1.0) = 1.7e+308
        (
            VM2_HYPOT,
            0x3fe6666666666666,
            0x3fe6666666666666,
            0x3fefadaa8f7eed51,
            1,
        ), // hypot(0.7, 0.7) = 0.9899494936611665
        (
            VM2_HYPOT,
            0x1eb67e9c127b6e74,
            0x1eb67e9c127b6e74,
            0x1ebfcfe76481c4b2,
            1,
        ), // hypot(1e-160, 1e-160) = 1.414213562373095e-160
    ];

    #[test]
    fn math64_reference_values_within_ulp_bound() {
        let mut worst = 0u64;
        for &(op, xb, eb, bound) in REF1 {
            let x = f64::from_bits(xb);
            let expected = f64::from_bits(eb);
            let got = f1(op, x);
            assert!(
                got.is_finite() && expected.is_finite(),
                "op {op} x={x:e}: got {got:e}, expected {expected:e}"
            );
            let d = ulp_diff(got, expected);
            assert!(
                d <= bound,
                "op {op} x={x:e} ({xb:#018x}): got {got:e} ({:#018x}), expected {expected:e} \
                 ({eb:#018x}), {d} ULP > {bound}",
                got.to_bits()
            );
            worst = worst.max(d);
        }
        for &(op, ab, bb, eb, bound) in REF2 {
            let a = f64::from_bits(ab);
            let b = f64::from_bits(bb);
            let expected = f64::from_bits(eb);
            let got = f2(op, a, b);
            let d = ulp_diff(got, expected);
            assert!(
                d <= bound,
                "op2 {op} a={a:e} b={b:e}: got {got:e} ({:#018x}), expected {expected:e} \
                 ({eb:#018x}), {d} ULP > {bound}",
                got.to_bits()
            );
            worst = worst.max(d);
        }
        // The tables carry 1-ULP rows for every family; the f32 demotion this module replaced was
        // off by ~2^29 ULP on them, so this gate cannot be satisfied vacuously.
        assert!(worst <= 3, "worst-case error {worst} ULP");
        assert!(REF1.len() >= 300 && REF2.len() >= 40);
    }

    #[test]
    fn math64_special_values_are_exact() {
        let inf = f64::INFINITY;
        let three_quarter_pi = 3.0 * STD_PI / 4.0;
        let root2_e308 = f64::from_bits(0x7fe9_2c80_954c_51f5); // √2·1e308, correctly rounded
        let cases: &[(i64, f64, f64)] = &[
            (VM_EXP, 0.0, 1.0),
            (VM_EXP, -0.0, 1.0),
            (VM_EXP, inf, inf),
            (VM_EXP, -inf, 0.0),
            (VM_EXP, 710.0, inf),
            (VM_EXP, -746.0, 0.0),
            (VM_LOG, 1.0, 0.0),
            (VM_LOG, 0.0, -inf),
            (VM_LOG, -0.0, -inf),
            (VM_LOG, inf, inf),
            (VM_SIN, 0.0, 0.0),
            (VM_SIN, -0.0, -0.0),
            (VM_SIN, 1e-300, 1e-300),
            (VM_SIN, -5e-324, -5e-324),
            (VM_COS, 0.0, 1.0),
            (VM_COS, -0.0, 1.0),
            (VM_COS, 1e-300, 1.0),
            (VM_TAN, 0.0, 0.0),
            (VM_TAN, -0.0, -0.0),
            (VM_TAN, -1e-300, -1e-300),
            (VM_ASIN, 1.0, FRAC_PI_2),
            (VM_ASIN, -1.0, -FRAC_PI_2),
            (VM_ASIN, 0.0, 0.0),
            (VM_ASIN, -0.0, -0.0),
            (VM_ASIN, 1e-300, 1e-300),
            (VM_ACOS, 1.0, 0.0),
            (VM_ACOS, -1.0, STD_PI),
            (VM_ACOS, 0.0, FRAC_PI_2),
            (VM_ATAN, inf, FRAC_PI_2),
            (VM_ATAN, -inf, -FRAC_PI_2),
            (VM_ATAN, 1.0, FRAC_PI_4),
            (VM_ATAN, 0.0, 0.0),
            (VM_ATAN, -0.0, -0.0),
            (VM_ATAN, -1e-300, -1e-300),
            (VM_SINH, 0.0, 0.0),
            (VM_SINH, -0.0, -0.0),
            (VM_SINH, inf, inf),
            (VM_SINH, -inf, -inf),
            (VM_SINH, 1e-300, 1e-300),
            (VM_SINH, 1000.0, inf),
            (VM_COSH, 0.0, 1.0),
            (VM_COSH, -0.0, 1.0),
            (VM_COSH, inf, inf),
            (VM_COSH, -inf, inf),
            (VM_COSH, 1e-300, 1.0),
            (VM_COSH, -1000.0, inf),
            (VM_TANH, 0.0, 0.0),
            (VM_TANH, -0.0, -0.0),
            (VM_TANH, inf, 1.0),
            (VM_TANH, -inf, -1.0),
            (VM_TANH, 25.0, 1.0),
            (VM_TANH, -25.0, -1.0),
            (VM_TANH, -1e-300, -1e-300),
            (VM_TANH, 5e-324, 5e-324),
            (VM_ASINH, 0.0, 0.0),
            (VM_ASINH, -0.0, -0.0),
            (VM_ASINH, inf, inf),
            (VM_ASINH, -inf, -inf),
            (VM_ASINH, -1e-300, -1e-300),
            (VM_ACOSH, 1.0, 0.0),
            (VM_ACOSH, inf, inf),
            (VM_ATANH, 0.0, 0.0),
            (VM_ATANH, -0.0, -0.0),
            (VM_ATANH, 1.0, inf),
            (VM_ATANH, -1.0, -inf),
            (VM_ATANH, 1e-300, 1e-300),
            (VM_ERF, 0.0, 0.0),
            (VM_ERF, -0.0, -0.0),
            (VM_ERF, inf, 1.0),
            (VM_ERF, -inf, -1.0),
            (VM_ERF, 7.0, 1.0),
            (VM_ERF, -7.0, -1.0),
            (VM_CBRT, 0.0, 0.0),
            (VM_CBRT, -0.0, -0.0),
            (VM_CBRT, inf, inf),
            (VM_CBRT, -inf, -inf),
            (VM_CBRT, 8.0, 2.0),
            (VM_CBRT, 27.0, 3.0),
            (VM_CBRT, -64.0, -4.0),
            (VM_CBRT, 1.0, 1.0),
            (VM_EXPM1, 0.0, 0.0),
            (VM_EXPM1, -0.0, -0.0),
            (VM_EXPM1, inf, inf),
            (VM_EXPM1, -inf, -1.0),
            (VM_EXPM1, 710.0, inf),
            (VM_EXPM1, -1e-300, -1e-300),
            (VM_LOG1P, 0.0, 0.0),
            (VM_LOG1P, -0.0, -0.0),
            (VM_LOG1P, -1.0, -inf),
            (VM_LOG1P, inf, inf),
            (VM_LOG1P, 1e-300, 1e-300),
            (VM_EXP2, 0.0, 1.0),
            (VM_EXP2, 10.0, 1024.0),
            (VM_EXP2, -1.0, 0.5),
            (VM_EXP2, 1023.0, f64::from_bits(0x7fe0_0000_0000_0000)),
            (VM_EXP2, 1024.0, inf),
            (VM_EXP2, -1074.0, 5e-324),
            (VM_EXP2, -1075.0, 0.0),
            (VM_EXP2, inf, inf),
            (VM_EXP2, -inf, 0.0),
            (VM_LOG2, 1.0, 0.0),
            (VM_LOG2, 8.0, 3.0),
            (VM_LOG2, 1024.0, 10.0),
            (VM_LOG2, 0.5, -1.0),
            (VM_LOG2, 5e-324, -1074.0),
            (VM_LOG2, 0.0, -inf),
            (VM_LOG2, inf, inf),
            (VM_LOG10, 1.0, 0.0),
            (VM_LOG10, 100.0, 2.0),
            (VM_LOG10, 1000.0, 3.0),
            (VM_LOG10, 1e15, 15.0),
            (VM_LOG10, 0.0, -inf),
            (VM_LOG10, inf, inf),
            (VM_EXP10, 0.0, 1.0),
            (VM_EXP10, -0.0, 1.0),
            (VM_EXP10, 3.0, 1000.0),
            (VM_EXP10, -3.0, 1e-3),
            (VM_EXP10, 15.0, 1e15),
            (VM_EXP10, -15.0, 1e-15),
            (VM_EXP10, inf, inf),
            (VM_EXP10, -inf, 0.0),
            (VM_EXP10, 400.0, inf),
            (VM_EXP10, -400.0, 0.0),
        ];
        for &(op, x, expected) in cases {
            let got = f1(op, x);
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "op {op} x={x:?}: got {got:?}, expected {expected:?}"
            );
        }
        let nan_cases: &[(i64, f64)] = &[
            (VM_EXP, f64::NAN),
            (VM_LOG, -1.0),
            (VM_LOG, -inf),
            (VM_LOG, f64::NAN),
            (VM_SIN, inf),
            (VM_SIN, -inf),
            (VM_COS, inf),
            (VM_TAN, inf),
            (VM_ASIN, 1.5),
            (VM_ASIN, -1.5),
            (VM_ACOS, 1.5),
            (VM_ATAN, f64::NAN),
            (VM_SINH, f64::NAN),
            (VM_COSH, f64::NAN),
            (VM_TANH, f64::NAN),
            (VM_ASINH, f64::NAN),
            (VM_ACOSH, 0.5),
            (VM_ACOSH, -inf),
            (VM_ATANH, 2.0),
            (VM_ATANH, -2.0),
            (VM_ERF, f64::NAN),
            (VM_CBRT, f64::NAN),
            (VM_EXPM1, f64::NAN),
            (VM_LOG1P, -2.0),
            (VM_LOG1P, -inf),
            (VM_EXP2, f64::NAN),
            (VM_LOG2, -1.0),
            (VM_LOG10, -1.0),
            (VM_EXP10, f64::NAN),
        ];
        for &(op, x) in nan_cases {
            assert!(f1(op, x).is_nan(), "op {op} x={x:?} must be NaN");
        }
        let cases2: &[(i64, f64, f64, f64)] = &[
            (VM2_POW, f64::NAN, 0.0, 1.0),
            (VM2_POW, 0.0, 0.0, 1.0),
            (VM2_POW, inf, -0.0, 1.0),
            (VM2_POW, 1.0, f64::NAN, 1.0),
            (VM2_POW, 1.0, inf, 1.0),
            (VM2_POW, 2.0, 10.0, 1024.0),
            (VM2_POW, -2.0, 3.0, -8.0),
            (VM2_POW, -2.0, 4.0, 16.0),
            (VM2_POW, 0.0, -1.0, inf),
            (VM2_POW, -0.0, -1.0, -inf),
            (VM2_POW, -0.0, 3.0, -0.0),
            (VM2_POW, -0.0, 2.0, 0.0),
            (VM2_POW, 0.0, 3.0, 0.0),
            (VM2_POW, inf, -1.0, 0.0),
            (VM2_POW, inf, 2.0, inf),
            (VM2_POW, -inf, 3.0, -inf),
            (VM2_POW, -inf, 2.0, inf),
            (VM2_POW, -inf, -3.0, -0.0),
            // C23 Annex F (pow): pow(−∞, y) = +∞ for y > 0 not an odd integer — not a NaN.
            (VM2_POW, -inf, 0.5, inf),
            (VM2_POW, 2.0, inf, inf),
            (VM2_POW, 0.5, inf, 0.0),
            (VM2_POW, -1.0, inf, 1.0),
            (VM2_POW, -1.0, -inf, 1.0),
            (VM2_POW, 2.0, -inf, 0.0),
            (VM2_POW, 0.5, -inf, inf),
            (VM2_POW, 4.0, 0.5, 2.0),
            (VM2_POW, 9.0, 0.5, 3.0),
            (VM2_POW, 2.0, -1.0, 0.5),
            (VM2_POW, 10.0, 2.0, 100.0),
            (VM2_POW, 2.0, 1023.0, f64::from_bits(0x7fe0_0000_0000_0000)),
            (VM2_POW, 2.0, 1024.0, inf),
            (VM2_POW, -2.0, 1025.0, -inf),
            (VM2_POW, 2.0, -1100.0, 0.0),
            (VM2_POW, 10.0, 22.0, 1e22),
            (VM2_POW, 3.0, 20.0, 3486784401.0),
            (VM2_ATAN2, 0.0, 1.0, 0.0),
            (VM2_ATAN2, -0.0, 1.0, -0.0),
            (VM2_ATAN2, 0.0, -1.0, STD_PI),
            (VM2_ATAN2, -0.0, -1.0, -STD_PI),
            (VM2_ATAN2, 0.0, -0.0, STD_PI),
            (VM2_ATAN2, -0.0, -0.0, -STD_PI),
            (VM2_ATAN2, 0.0, 0.0, 0.0),
            (VM2_ATAN2, 1.0, 0.0, FRAC_PI_2),
            (VM2_ATAN2, -1.0, 0.0, -FRAC_PI_2),
            (VM2_ATAN2, 1.0, -0.0, FRAC_PI_2),
            (VM2_ATAN2, inf, inf, FRAC_PI_4),
            (VM2_ATAN2, -inf, inf, -FRAC_PI_4),
            (VM2_ATAN2, inf, -inf, three_quarter_pi),
            (VM2_ATAN2, -inf, -inf, -three_quarter_pi),
            (VM2_ATAN2, 1.0, inf, 0.0),
            (VM2_ATAN2, -1.0, inf, -0.0),
            (VM2_ATAN2, 1.0, -inf, STD_PI),
            (VM2_ATAN2, -1.0, -inf, -STD_PI),
            (VM2_ATAN2, inf, 1.0, FRAC_PI_2),
            (VM2_ATAN2, -inf, -1.0, -FRAC_PI_2),
            (VM2_ATAN2, 1.0, 1.0, FRAC_PI_4),
            (VM2_ATAN2, 1e-300, -1.0, STD_PI),
            (VM2_HYPOT, 3.0, 4.0, 5.0),
            (VM2_HYPOT, -3.0, 4.0, 5.0),
            (VM2_HYPOT, 3.0, -4.0, 5.0),
            (VM2_HYPOT, 0.0, 0.0, 0.0),
            (VM2_HYPOT, -0.0, -0.0, 0.0),
            (VM2_HYPOT, -2.5, 0.0, 2.5),
            (VM2_HYPOT, 0.0, -2.5, 2.5),
            (VM2_HYPOT, inf, f64::NAN, inf),
            (VM2_HYPOT, f64::NAN, -inf, inf),
            (VM2_HYPOT, -inf, 1.0, inf),
            // √2·1e308 ≈ 1.414e308 is below DBL_MAX: the 2^±700 scaling keeps it finite (mpmath,
            // correctly rounded).
            (VM2_HYPOT, 1e308, 1e308, root2_e308),
            (VM2_HYPOT, 1.5e308, 1.5e308, inf),
            (VM2_HYPOT, 1.0, 1e-300, 1.0),
            (VM2_HYPOT, 5.0, 12.0, 13.0),
        ];
        for &(op, a, b, expected) in cases2 {
            let got = f2(op, a, b);
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "op2 {op} a={a:?} b={b:?}: got {got:?}, expected {expected:?}"
            );
        }
        let nan_cases2: &[(i64, f64, f64)] = &[
            (VM2_POW, -2.0, 0.5),
            (VM2_POW, -1.0, 1.5),
            (VM2_POW, f64::NAN, 1.0),
            (VM2_POW, 2.0, f64::NAN),
            (VM2_ATAN2, f64::NAN, 1.0),
            (VM2_ATAN2, 1.0, f64::NAN),
            (VM2_HYPOT, f64::NAN, 1.0),
            (VM2_HYPOT, 1.0, f64::NAN),
        ];
        for &(op, a, b) in nan_cases2 {
            assert!(
                f2(op, a, b).is_nan(),
                "op2 {op} a={a:?} b={b:?} must be NaN"
            );
        }
    }

    /// Every odd function is odd bit-for-bit and every even one even: the sign is handled by
    /// symmetric code paths, never by an approximation that could round differently.
    #[test]
    fn math64_symmetry_is_bit_exact() {
        let odd = [
            VM_SIN, VM_TAN, VM_ASIN, VM_ATAN, VM_SINH, VM_TANH, VM_ASINH, VM_ATANH, VM_ERF, VM_CBRT,
        ];
        let even = [VM_COS, VM_COSH];
        let mut xs = vec![
            1e-300, 1e-20, 1e-9, 1e-5, 0.01, 0.1, 0.25, 0.3, 0.5, 0.7, 0.9, 0.999,
        ];
        for k in 0..200 {
            xs.push(1.0 + 0.05 * k as f64);
        }
        xs.extend([25.0, 100.0, 1e6, 1e10, 1e22, 1e100, 1e300]);
        for &x in &xs {
            for &op in &odd {
                let p = f1(op, x);
                let n = f1(op, -x);
                if p.is_nan() {
                    // Outside the domain (asin/acos/atanh past ±1) both sides are NaN.
                    assert!(n.is_nan(), "op {op} x={x:e}");
                    continue;
                }
                assert_eq!(n.to_bits(), (-p).to_bits(), "op {op} x={x:e}");
            }
            for &op in &even {
                assert_eq!(f1(op, x).to_bits(), f1(op, -x).to_bits(), "op {op} x={x:e}");
            }
        }
    }

    /// A composition the compiler builds in MIR from these calls — `sigmoid(x) = 1/(1+exp(−x))` —
    /// agrees with the mpmath value to the error the composition can carry (one division and
    /// one addition on top of a ≤1-ULP exp), so an f64 activation really is double precision.
    #[test]
    fn math64_composed_sigmoid_is_double_precision() {
        // mpmath: 1/(1+exp(-1)) = 0.7310585786300048792511592...
        let expected = 0.7310585786300049_f64;
        let got = 1.0 / (1.0 + f1(VM_EXP, -1.0));
        assert!(ulp_diff(got, expected) <= 2, "{got:e} vs {expected:e}");
        // The f32 demotion this replaced was 0.7310586 — 3e-8 away, ~2^27 ULP.
        assert!((got - expected).abs() < 1e-15);
    }

    #[test]
    fn math64_unknown_opcode_is_identity() {
        for x in [0.0, -0.0, 0.5, -3.25, 1e300, f64::INFINITY] {
            assert_eq!(f1(-1, x).to_bits(), x.to_bits());
            assert_eq!(f1(4, x).to_bits(), x.to_bits()); // VM_RELU: an activation, composed in MIR
            assert_eq!(f1(1000, x).to_bits(), x.to_bits());
            assert_eq!(f2(-1, x, 7.0).to_bits(), x.to_bits());
            assert_eq!(f2(3, x, 7.0).to_bits(), x.to_bits()); // VM2_SILU_BWD: composed in MIR
        }
        let nan = f64::from_bits(0x7ff8_0000_dead_beef);
        assert_eq!(f1(-1, nan).to_bits(), nan.to_bits());
    }

    /// Each code the compiler emits reaches a real function (not the identity default).
    #[test]
    fn math64_dispatch_covers_every_emitted_code() {
        let core = [
            VM_EXP,
            VM_LOG,
            VM_TANH,
            VM_SIN,
            VM_COS,
            VM_ERF,
            VM_EXP2,
            VM_LOG2,
            VM_SINH,
            VM_COSH,
            VM_ASINH,
            VM_ACOSH,
            VM_ATANH,
            VM_ATAN,
            VM_EXPM1,
            VM_LOG1P,
            VM_EXP10,
            VM_LOG10,
            VM_TAN,
            VM_ASIN,
            VM_ACOS,
            VM_CBRT,
            M64_ERFC,
            M64_ERFINV,
            M64_LGAMMA,
            M64_DIGAMMA,
        ];
        for op in core {
            assert_ne!(
                f1(op, 0.5).to_bits(),
                0.5f64.to_bits(),
                "op {op} is the identity"
            );
        }
        for op in [VM2_POW, VM2_ATAN2, VM2_HYPOT] {
            assert_ne!(
                f2(op, 0.5, 0.3).to_bits(),
                0.5f64.to_bits(),
                "op2 {op} is the identity"
            );
        }
    }

    /// Every decimal constant in this file that carries its intended hex words in a trailing
    /// comment parses to exactly those bits — the originals warn that this is the one way a port
    /// goes silently wrong.
    #[test]
    fn math64_constants_match_their_hex() {
        let src = include_str!("math64.rs");
        let mut checked = 0;
        for line in src.lines() {
            let (Some(cstart), Some(cend)) = (line.find("/*"), line.find("*/")) else {
                continue;
            };
            if cend < cstart {
                continue;
            }
            let words: Vec<u32> = line[cstart + 2..cend]
                .split(|c: char| !c.is_ascii_hexdigit() && c != 'x')
                .filter_map(|w| {
                    let w = w.strip_prefix("0x").unwrap_or(w);
                    if w.len() == 8 {
                        u32::from_str_radix(w, 16).ok()
                    } else {
                        None
                    }
                })
                .collect();
            if words.len() != 2 {
                continue;
            }
            let code = line[..cstart].trim_end().trim_end_matches([';', ',']);
            let lit = code.rsplit(['=', ' ']).next().unwrap_or("");
            let Ok(dec) = lit.parse::<f64>() else {
                continue;
            };
            let expected = ((words[0] as u64) << 32) | words[1] as u64;
            assert_eq!(dec.to_bits(), expected, "{line}");
            checked += 1;
        }
        assert!(checked >= 160, "only {checked} constants carry hex words");
    }

    /// The 256-entry `exp2` table is internally consistent: entry `i` is `2^((i−128)/256 + eps[i])`
    /// to within 2 ULP of the host's own `exp2` (the table is accurate to 2⁻⁶⁴, so the host's
    /// ≤1-ULP result must land within 2 ULP of it), `eps[i]` is tiny (below 2⁻³⁹; FreeBSD's largest
    /// is 1.02e-12, just over 2⁻⁴⁰), and the entries increase strictly. A single wrong hex digit in
    /// any of the 512 words fails here, which the sampled reference values above cannot promise for
    /// every bucket.
    #[test]
    fn math64_exp2_table_is_consistent() {
        let mut previous = 0.0f64;
        for i in 0..256usize {
            let t = f64::from_bits(EXP2_TBL[2 * i]);
            let eps = f64::from_bits(EXP2_TBL[2 * i + 1]);
            assert!(eps.abs() < 2.0f64.powi(-39), "eps[{i}] = {eps:e}");
            let want = ((i as f64 - 128.0) / 256.0 + eps).exp2();
            assert!(
                ulp_diff(t, want) <= 2,
                "exp2 table entry {i}: {t:e} ({:#018x}) vs {want:e}",
                EXP2_TBL[2 * i]
            );
            assert!(t > previous, "exp2 table entry {i} does not increase");
            previous = t;
        }
        assert_eq!(
            f64::from_bits(EXP2_TBL[256]),
            1.0,
            "entry 128 is exactly 2^0"
        );
        assert_eq!(EXP2_TBL[257], 0, "entry 128 needs no correction");
    }

    #[test]
    fn math64_scalbn_edges() {
        assert_eq!(scalbn(1.0, 1023).to_bits(), 0x7fe0_0000_0000_0000);
        assert_eq!(scalbn(1.0, 1024), f64::INFINITY);
        assert_eq!(scalbn(1.0, -1074).to_bits(), 1);
        assert_eq!(scalbn(1.0, -1075), 0.0);
        assert_eq!(scalbn(1.5, -1074).to_bits(), 2); // 1.5·2^-1074 rounds to even: 2^-1073
        assert_eq!(scalbn(-0.0, 10).to_bits(), (-0.0f64).to_bits());
        assert_eq!(scalbn(3.0, 0), 3.0);
        // 2^-1022 · 2^2045 = 2^1023 goes through the 2^1023 prescale; one more doubling overflows.
        assert_eq!(
            scalbn(f64::MIN_POSITIVE, 2045),
            f64::from_bits(0x7fe0_0000_0000_0000)
        );
        assert_eq!(scalbn(f64::MIN_POSITIVE, 2046), f64::INFINITY);
    }
}
