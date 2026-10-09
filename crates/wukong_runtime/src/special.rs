//! Double-precision special functions with no f32 `vmath` kernel: `erfinv`, `lgamma`, and
//! `digamma` (the derivative of `lgamma`, which the autodiff pass emits; it is not a source
//! builtin). `erfc` lives next to `erf` in [`crate::math64`], whose `erfc2` tail it shares.
//!
//! The compiler reaches all four through [`crate::math64::wukong_rt_math_f64`] with an `M64_*`
//! code, for **every** float precision: an `f32` (or `f16`/`bf16`) argument is widened to `f64`,
//! evaluated here, and rounded back once — so the f32 result is within a hair of correctly rounded,
//! and the interpreter, the JIT and an AOT executable call this one Rust implementation and agree
//! bit-for-bit. Like `math64` the module uses only IEEE `+ − × ÷`, `sqrt`, `floor` and the
//! `math64` transcendentals — no platform libm, no FMA — so the bits are the same on every host.
//!
//! **Accuracy** (measured against mpmath at 120 bits over dense random sweeps when the algorithms
//! were designed; `special_references_within_bound` pins sample points at ≤ 4 ULP, which leaves
//! room for this crate's own ≤ 1-ULP `exp`/`log`/`sin` and still fails any wrong coefficient):
//!
//! | function | bound |
//! |---|---|
//! | `erfinv` on `(−1, 1)` | ≤ 3 ULP measured, through `1 − 2⁻⁵³` |
//! | `lgamma` on `x > 0` | ≤ 2 ULP measured |
//! | `lgamma` on `x < 0` | error ≤ 6e-16 · max(1, \|lgamma(x)\|); relative accuracy is lost only next to the zeros of `lgamma`, where the reflection formula cancels (as in every libm that reflects) |
//! | `digamma` | error ≤ 2e-15 · max(1, \|ψ(x)\|) |
//!
//! **Special values.** `erfinv(±1) = ±∞`, `erfinv(x) = NaN` for `|x| > 1`, `erfinv(±0) = ±0`.
//! `lgamma(+∞) = lgamma(−∞) = +∞`, `lgamma` of `0` or a negative integer is `+∞` (a pole of Γ),
//! and it overflows to `+∞` above `x ≈ 2.556e305`. `digamma(+∞) = +∞`; `digamma` of `−∞`, `0` or a
//! negative integer is NaN (a pole). A NaN argument returns itself.

// The constants are long decimal expansions computed with mpmath (their provenance is stated at
// each table), and the algorithms read best with single-letter working variables.
#![allow(clippy::excessive_precision, clippy::many_single_char_names)]

use crate::math64::{cos, erf, erfc, exp, log, log1p, sin, tan};
use std::f64::consts::{FRAC_2_SQRT_PI, PI};

/// `erfc(x)` — the [`wukong_rt_math_f64`](crate::math64::wukong_rt_math_f64) code. The library-only
/// codes start at 100, clear of the 36-code `VM_*` space the f32 `vmath` kernels share.
pub const M64_ERFC: i64 = 100;
/// `erfinv(x)`.
pub const M64_ERFINV: i64 = 101;
/// `lgamma(x) = ln |Γ(x)|`.
pub const M64_LGAMMA: i64 = 102;
/// `digamma(x) = ψ(x) = d/dx lgamma(x)` — emitted only by autodiff, as the derivative of `lgamma`.
pub const M64_DIGAMMA: i64 = 103;

// --- erfinv ------------------------------------------------------------------------------------

/// The single-precision `erfinv` approximation of M. Giles, "Approximating the erfinv function"
/// (GPU Computing Gems, 2011): `erfinv(a) ≈ a·P(w)` with `w = −ln((1−a)(1+a))`, a degree-8
/// polynomial in `w − 2.5` below `w = 5` and in `√w − 3` above. Relative error ~1e-7 up to
/// `|a| = 1 − 1e-6`, degrading to ~12% at `1 − 2⁻⁵³` — it is only the starting point here.
const ERFINV_P1: [f64; 9] = [
    2.81022636e-08,
    3.43273939e-07,
    -3.5233877e-06,
    -4.39150654e-06,
    0.00021858087,
    -0.00125372503,
    -0.00417768164,
    0.246640727,
    1.50140941,
];
const ERFINV_P2: [f64; 9] = [
    -0.000200214257,
    0.000100950558,
    0.00134934322,
    -0.00367342844,
    0.00573950773,
    -0.0076224613,
    0.00943887047,
    1.00167406,
    2.83297682,
];

/// `c[0]·tⁿ⁻¹ + … + c[n−1]`, highest coefficient first (Giles' order).
fn horner(c: &[f64], t: f64) -> f64 {
    let mut p = c[0];
    for &k in &c[1..] {
        p = k + p * t;
    }
    p
}

/// `√π/2`, the slope of `erfinv` at 0.
const HALF_SQRT_PI: f64 = 0.886226925452758;

/// `erfinv(y)`, the inverse of [`erf`] on `(−1, 1)`.
///
/// Giles' approximation seeds the root; it is then polished in double precision against this
/// crate's own `erf`/`erfc`. For `|y| ≤ ½` two Halley steps on `erf(x) − |y|` (with `f'' = −2x·f'`
/// the step is `x − d/(1 + x·d)`, `d` the Newton step). For `|y| > ½` four Newton steps on
/// `ln(erfc(x)/q)` with `q = 1 − |y|` exact (Sterbenz): in the log the function is nearly a
/// parabola, so the iteration converges from the seed's 12% tail error, and `erfc` keeps the
/// residual's full relative accuracy where `erf(x) − y` would have cancelled to nothing. Below
/// `2⁻²⁷` the series `y·√π/2·(1 + πy²/12 + …)` is `y·√π/2` to within half an ULP.
pub fn erfinv(y: f64) -> f64 {
    if y.is_nan() {
        return y;
    }
    let a = y.abs();
    if a > 1.0 {
        return f64::NAN;
    }
    if a == 1.0 {
        return if y > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    if a < f64::from_bits(0x3e40_0000_0000_0000) {
        // |y| < 2^-27 (this also returns ±0 for ±0)
        return y * HALF_SQRT_PI;
    }
    let w = -log((1.0 - a) * (1.0 + a));
    let p = if w < 5.0 {
        horner(&ERFINV_P1, w - 2.5)
    } else {
        horner(&ERFINV_P2, w.sqrt() - 3.0)
    };
    let mut x = p * a;
    if a <= 0.5 {
        for _ in 0..2 {
            let e = erf(x) - a;
            let d = e / (FRAC_2_SQRT_PI * exp(-x * x));
            x -= d / (1.0 + x * d);
        }
    } else {
        let q = 1.0 - a;
        for _ in 0..4 {
            let c = erfc(x);
            let g = log(c / q);
            x += g * c / (FRAC_2_SQRT_PI * exp(-x * x));
        }
    }
    if y < 0.0 {
        -x
    } else {
        x
    }
}

// --- lgamma ------------------------------------------------------------------------------------

/// `lgamma(2 + t) = Σ_{k≥1} S2[k−1]·tᵏ` for `|t| ≤ ½`: `S2[0] = 1 − γ` and
/// `S2[k−1] = (−1)ᵏ(ζ(k) − 1)/k` (mpmath, 200 bits). The terms fall like `4⁻ᵏ/k` at `|t| = ½`, so
/// thirty of them reach far below an ULP of the result.
const LGAMMA_S2: [f64; 30] = [
    0.42278433509846713,
    0.3224670334241132,
    -0.0673523010531981,
    0.020580808427784546,
    -0.007385551028673986,
    0.0028905103307415234,
    -0.001192753911703261,
    0.0005096695247430425,
    -0.00022315475845357939,
    9.945751278180853e-05,
    -4.492623673813314e-05,
    2.050721277567069e-05,
    -9.439488275268397e-06,
    4.374866789907488e-06,
    -2.039215753801366e-06,
    9.55141213040742e-07,
    -4.492469198764566e-07,
    2.1207184805554665e-07,
    -1.0043224823968099e-07,
    4.7698101693639804e-08,
    -2.2711094608943164e-08,
    1.0838659214896955e-08,
    -5.183475041970047e-09,
    2.4836745438024785e-09,
    -1.1921401405860912e-09,
    5.731367241678862e-10,
    -2.7595228851242334e-10,
    1.330476437424449e-10,
    -6.4229645638381e-11,
    3.1044247747322276e-11,
];

/// The minimum of `lgamma` on `x > 0`: `X0` is the double nearest the root of ψ, `F0 = lgamma(X0)`,
/// and `lgamma(X0 + t) = F0 + Σ_{k≥1} LGAMMA_MIN[k−1]·tᵏ` with `LGAMMA_MIN[k−1] = ψ⁽ᵏ⁻¹⁾(X0)/k!`
/// (mpmath, 200 bits — the first is not zero because `X0` is rounded). Used on `[1.25, 1.75)`,
/// where expanding around 2 or 1 would cancel against `log`.
const LGAMMA_X0: f64 = 1.4616321449683622;
const LGAMMA_F0: f64 = -0.12148629053584961;
const LGAMMA_MIN: [f64; 26] = [
    -9.241265521729427e-17,
    0.4838361227238106,
    -0.14758772299453074,
    0.06462494023891277,
    -0.03278854108848132,
    0.017970675115210402,
    -0.010314223036636393,
    0.006100536020517891,
    -0.0036845696083163755,
    0.002259764823221812,
    -0.0014022514459044518,
    0.000878232634717682,
    -0.0005541949527966826,
    0.000351912956837847,
    -0.00022465344369595542,
    0.00014407039542093313,
    -9.276098655471758e-05,
    5.9934733439794323e-05,
    -3.884583889452663e-05,
    2.5247663291730135e-05,
    -1.6450858338395626e-05,
    1.0743455256375948e-05,
    -7.0307052573803595e-06,
    4.609738653644762e-06,
    -3.027671832878031e-06,
    1.991759921239265e-06,
];

/// Stirling's series `Σ_{k≥1} B₂ₖ/(2k(2k−1))·x^{1−2k}`, `k = 1..9`, and `½·ln(2π) − ½`.
const LGAMMA_STIRLING: [f64; 9] = [
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
    -3617.0 / 122400.0,
    43867.0 / 244188.0,
];
const LGAMMA_W0: f64 = 0.4189385332046727;

/// Horner: `Σ c[k]·tᵏ⁺¹` (the series start at the linear term).
fn series(c: &[f64], t: f64) -> f64 {
    let mut acc = 0.0;
    for &k in c.iter().rev() {
        acc = acc * t + k;
    }
    acc * t
}

/// `sin(π·x)` for finite `x ≥ 0`, with the argument reduced exactly: `x = n/2 + d`, `|d| ≤ ¼`
/// (`x − n/2` is exact by Sterbenz), then `sin(nπ/2 + πd)` is `±sin(πd)` or `±cos(πd)`.
fn sin_pi(x: f64) -> f64 {
    let n = (x * 2.0 + 0.5).floor();
    let d = x - n * 0.5;
    let a = PI * d;
    match (n as i64) & 3 {
        0 => sin(a),
        1 => cos(a),
        2 => -sin(a),
        _ => -cos(a),
    }
}

/// `lgamma` on `x ≥ 2⁻⁷⁰`.
fn lgamma_pos(x: f64) -> f64 {
    if x < 0.5 {
        // Γ(x) = Γ(x + 2)/((x + 1)·x); `x` itself is the series variable, exactly.
        return series(&LGAMMA_S2, x) - log1p(x) - log(x);
    }
    if x < 1.25 {
        // Γ(x) = Γ(x + 1)/x with t = x − 1 exact (Sterbenz).
        let t = x - 1.0;
        return series(&LGAMMA_S2, t) - log1p(t);
    }
    if x < 1.75 {
        return LGAMMA_F0 + series(&LGAMMA_MIN, x - LGAMMA_X0);
    }
    if x < 2.5 {
        return series(&LGAMMA_S2, x - 2.0);
    }
    if x < 8.0 {
        // Γ(x) = (x−1)(x−2)…(x−n)·Γ(x−n) with x − n in [1.5, 2.5); every x − k is exact.
        let n = (x - 1.5).floor() as i32;
        let mut p = 1.0;
        let mut y = x;
        for _ in 0..n {
            y -= 1.0;
            p *= y;
        }
        return log(p) + series(&LGAMMA_S2, y - 2.0);
    }
    let w = 1.0 / x;
    let w2 = w * w;
    let mut acc = LGAMMA_STIRLING[8];
    for &c in LGAMMA_STIRLING[..8].iter().rev() {
        acc = acc * w2 + c;
    }
    (x - 0.5) * (log(x) - 1.0) + (LGAMMA_W0 + w * acc)
}

/// `lgamma(x) = ln |Γ(x)|`.
///
/// For `x > 0`: a series around 2 (`lgamma(2 + t)` in ζ-function coefficients) on `[1.75, 2.5)`,
/// and through `Γ(x + 1) = x·Γ(x)` on `(0, 1.25)` without ever rounding the series argument; a
/// Taylor series around the minimum `x₀ ≈ 1.4616` on `[1.25, 1.75)`; the downward recurrence
/// into `[1.5, 2.5)` on `[2.5, 8)`; Stirling's series (nine terms) from 8 up. For `x < 0` the
/// reflection `Γ(x)·Γ(−x) = −π/(x·sin πx)` with an exactly reduced `sin πx`. Below `2⁻⁷⁰` in
/// magnitude it is `−ln|x|`.
pub fn lgamma(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    let ax = x.abs();
    if ax < f64::from_bits(0x3b90_0000_0000_0000) {
        // |x| < 2^-70: lgamma(x) = -ln|x| - γx + O(x²), and γx is below half an ULP.
        if ax == 0.0 {
            return f64::INFINITY;
        }
        return -log(ax);
    }
    if x < 0.0 {
        if x == x.floor() {
            // A pole of Γ.
            return f64::INFINITY;
        }
        let s = sin_pi(ax);
        let nadj = log(PI / (s * ax).abs());
        return nadj - lgamma_pos(ax);
    }
    lgamma_pos(x)
}

// --- digamma -----------------------------------------------------------------------------------

/// `B₂ₖ/(2k)` for `k = 1..7` (Bernoulli numbers), for ψ's asymptotic series.
const DIGAMMA_C: [f64; 7] = [
    1.0 / 12.0,
    1.0 / 120.0,
    1.0 / 252.0,
    1.0 / 240.0,
    1.0 / 132.0,
    691.0 / 32760.0,
    1.0 / 12.0,
];

/// `digamma(x) = ψ(x)`, the derivative of [`lgamma`]: the upward recurrence `ψ(x) = ψ(x + 1) − 1/x`
/// to `x ≥ 12`, then `ln x − 1/(2x) − Σ B₂ₖ/(2k·x²ᵏ)` (seven terms, below 1e-17 there); for
/// `x ≤ 0` the reflection `ψ(x) = ψ(1 − x) − π/tan(πx)` with `tan(πx)` reduced exactly. Error
/// ≤ 2e-15 relative to `max(1, |ψ(x)|)` — it is the gradient of `lgamma`, where absolute accuracy
/// is what matters.
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    let mut x = x;
    let mut result = 0.0;
    if x <= 0.0 {
        if x == x.floor() {
            // A pole.
            return f64::NAN;
        }
        // tan(πx) has period 1; r = x − round(x) is exact.
        let r = x - (x + 0.5).floor();
        result = -PI / tan(PI * r);
        x = 1.0 - x;
    }
    while x < 12.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let t = 1.0 / (x * x);
    let mut s = DIGAMMA_C[6];
    for &c in DIGAMMA_C[..6].iter().rev() {
        s = c - t * s;
    }
    result + (log(x) - 0.5 / x - s * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The monotone integer image of a double (−0 and +0 both map to 0).
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

    /// Correctly rounded references from mpmath at 150 bits (`float(mpf)` rounds to nearest):
    /// `(code, argument, expected, ULP bound)`. A bound of 0 is an exact case; the others allow
    /// the algorithms' measured worst case (see the module table) with room for this crate's
    /// `exp`/`log`/`sin` (each ≤ 1 ULP) — a wrong coefficient or branch is off by thousands.
    const REFS: &[(i64, f64, f64, u64)] = &[
        (M64_ERFC, 0.0, 1.0, 0),
        (M64_ERFC, 0.1, 0.887537083981715, 4),
        (M64_ERFC, 0.3, 0.6713732405408726, 4),
        (M64_ERFC, 0.5, 0.4795001221869535, 4),
        (M64_ERFC, -0.5, 1.5204998778130465, 4),
        (M64_ERFC, 0.84375, 0.23277433876765838, 4),
        (M64_ERFC, 1.0, 0.15729920705028513, 4),
        (M64_ERFC, 1.25, 0.07709987174354177, 4),
        (M64_ERFC, 2.0, 0.004677734981047266, 4),
        (M64_ERFC, 3.0, 2.209049699858544e-05, 4),
        (M64_ERFC, -3.0, 1.9999779095030015, 4),
        (M64_ERFC, 5.0, 1.537459794428035e-12, 4),
        (M64_ERFC, 10.0, 2.088487583762545e-45, 4),
        (M64_ERFC, 20.0, 5.395865611607901e-176, 4),
        (M64_ERFC, 26.5, 2.2109076642637343e-307, 4),
        (M64_ERFINV, 1e-10, 8.862269254527581e-11, 1),
        (M64_ERFINV, 0.1, 0.08885599049425769, 4),
        (M64_ERFINV, 0.25, 0.2253120550121781, 4),
        (M64_ERFINV, 0.5, 0.4769362762044699, 4),
        (M64_ERFINV, -0.5, -0.4769362762044699, 4),
        (M64_ERFINV, -0.75, -0.8134198475976185, 4),
        (M64_ERFINV, 0.9, 1.1630871536766743, 4),
        (M64_ERFINV, 0.999, 2.3267537655135246, 4),
        (M64_ERFINV, 0.999999, 3.458910737275499, 4),
        (M64_ERFINV, 0.999999999999, 5.042031898572696, 4),
        (M64_ERFINV, 0.9999999999999999, 5.8635847487551676, 4),
        (M64_LGAMMA, 1.0, 0.0, 0),
        (M64_LGAMMA, 2.0, 0.0, 0),
        (M64_LGAMMA, 1.4616321449683622, -0.12148629053584961, 0),
        (M64_LGAMMA, 1e-05, 11.512919692895826, 4),
        (M64_LGAMMA, 0.1, 2.252712651734206, 4),
        (M64_LGAMMA, 0.5, 0.5723649429247001, 4),
        (M64_LGAMMA, 0.9, 0.06637623973474295, 4),
        (M64_LGAMMA, 1.25, -0.09827183642181316, 4),
        (M64_LGAMMA, 1.5, -0.12078223763524522, 4),
        (M64_LGAMMA, 1.75, -0.08440112102048555, 4),
        (M64_LGAMMA, 2.5, 0.2846828704729192, 4),
        (M64_LGAMMA, 3.0, std::f64::consts::LN_2, 4), // lgamma(3) = ln 2
        (M64_LGAMMA, 7.5, 7.534364236758733, 4),
        (M64_LGAMMA, 8.0, 8.525161361065415, 4),
        (M64_LGAMMA, 10.0, 12.801827480081469, 4),
        (M64_LGAMMA, 100.0, 359.1342053695754, 4),
        (M64_LGAMMA, 1e300, 6.897755278982137e302, 4),
        (M64_LGAMMA, -0.5, 1.2655121234846454, 4),
        (M64_LGAMMA, -10.5, -15.147270590717842, 4),
    ];

    #[test]
    fn special_references_within_bound() {
        for &(op, x, want, bound) in REFS {
            let got = crate::math64::wukong_rt_math_f64(x, op);
            let d = ulp_diff(got, want);
            assert!(
                d <= bound,
                "op {op} x={x:e}: got {got:e} ({:#018x}), expected {want:e} ({:#018x}), {d} ULP \
                 > {bound}",
                got.to_bits(),
                want.to_bits()
            );
        }
    }

    #[test]
    fn special_values() {
        let inf = f64::INFINITY;
        assert!(erfinv(f64::NAN).is_nan());
        assert!(erfinv(1.5).is_nan() && erfinv(-1.0000000000000002).is_nan());
        assert_eq!(erfinv(1.0), inf);
        assert_eq!(erfinv(-1.0), -inf);
        assert_eq!(erfinv(0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(erfinv(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(crate::math64::erfc(inf).to_bits(), 0.0f64.to_bits());
        assert_eq!(crate::math64::erfc(-inf), 2.0);
        assert!(crate::math64::erfc(f64::NAN).is_nan());
        assert_eq!(crate::math64::erfc(30.0), 0.0);
        assert_eq!(lgamma(inf), inf);
        assert_eq!(lgamma(-inf), inf);
        assert_eq!(lgamma(0.0), inf);
        assert_eq!(lgamma(-0.0), inf);
        assert_eq!(lgamma(-3.0), inf);
        assert_eq!(lgamma(1e306), inf);
        assert!(lgamma(f64::NAN).is_nan());
        assert_eq!(digamma(inf), inf);
        assert!(digamma(-inf).is_nan());
        assert!(digamma(0.0).is_nan() && digamma(-2.0).is_nan());
        assert!(digamma(f64::NAN).is_nan());
    }

    /// ψ against mpmath (150 bits), to the documented error `2e-15·max(1, |ψ|)`.
    #[test]
    fn digamma_references() {
        for &(x, want) in &[
            (1.0, -0.5772156649015329),
            (0.5, -1.9635100260214235),
            (1.4616321449683622, -9.241265521729427e-17),
            (2.5, 0.7031566406452432),
            (10.0, 2.251752589066721),
            (100.5, 4.605174352581845),
            (1e10, 23.025850929890456),
            (-0.5, 0.03648997397857652),
            (-2.75, -1.959055264977997),
        ] {
            let got = digamma(x);
            let tol = 2e-15 * f64::max(1.0, f64::abs(want));
            assert!(
                (got - want).abs() <= tol,
                "digamma({x}) = {got:e}, expected {want:e}"
            );
        }
    }
}
