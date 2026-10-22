//! 8-bit floating point — the OCP *8-bit Floating Point Specification* (v1.0) formats every FP8
//! checkpoint ships in (DeepSeek-V3, Llama-3 FP8, Transformer Engine), held as raw `u8` storage:
//!
//! | format | sign/exp/mantissa | bias | max finite | min normal | min subnormal | inf | NaN |
//! |---|---|---|---|---|---|---|---|
//! | `E4M3FN` | 1/4/3 | 7 | 448 | 2⁻⁶ | 2⁻⁹ | none | `S.1111.111` |
//! | `E5M2` | 1/5/2 | 15 | 57344 | 2⁻¹⁴ | 2⁻¹⁶ | `S.11111.00` | `S.11111.{01,10,11}` |
//!
//! These are the **reference** conversions. The language's `f8e4m3_to_f32` / `f32_to_f8e4m3` /
//! `f8e5m2_to_f32` / `f32_to_f8e5m2` builtins do not call them — `wukong_mir_build` lowers each to
//! plain integer MIR on the `f32` bit pattern, so the conversions agree bit-for-bit on every
//! backend by construction (a conversion loop is an ordinary candidate for the loop vectorizer,
//! whose widened ops compute each lane exactly as the scalar op does; its constant-amount shifts
//! widen as vector-by-scalar shifts) — and `wukong_codegen_cranelift`'s
//! `fp8_builtins_match_the_runtime_reference` pins that lowering, on the interpreter and on native,
//! to these functions on all 256 codes and on every rounding boundary. They are
//! written differently on purpose: decoding by the formula `±2^(e−bias)·(1 + m/2ᵖ)`, encoding by
//! exact `f64` scaling and an explicit round-half-to-even, then a table lookup of the code.
//!
//! **Semantics** (those of `ml_dtypes` / PyTorch, with Transformer Engine's saturation for E4M3):
//! * decoding is exact; an E4M3FN NaN decodes to the quiet `f32` NaN with the code's sign, an
//!   E5M2 NaN as its `f16` would (the quiet bit set, the payload kept);
//! * encoding rounds to nearest, ties to even, with subnormals;
//! * `E4M3FN` has no infinity, so it **saturates**: a finite value beyond ±448 and ±∞ encode as
//!   ±448 (`0x7E`/`0xFE`), and a NaN as `0x7F`/`0xFF`;
//! * `E5M2` follows IEEE: a value that rounds beyond 57344 is ±∞ (`0x7C`/`0xFC`), a NaN is
//!   `0x7E`/`0xFE`.
//!
//! The **first-class `f8e4m3` / `f8e5m2` types** use the same rules. A value of either type lives
//! in an `f32` register on the grid and is stored as its code; every backend rounds onto the grid
//! with the bit-level codec below ([`encode_f32`], [`encode_f64`], [`decode_f32`]) — the interpreter
//! calls it, Cranelift and the LLVM emitter emit the identical integer sequence inline — and the
//! reference functions above are its oracle (`fast_codec_matches_the_reference`). `encode_f64`
//! rounds an `f64` straight to the grid, once: going through `f32` first would round twice.

/// `2^k` for `k` in the normal `f64` exponent range.
fn pow2(k: i32) -> f64 {
    f64::from_bits(((1023 + k) as u64) << 52)
}

/// One FP8 format's parameters.
#[derive(Clone, Copy)]
struct Format {
    mbits: u32,
    bias: i32,
    /// The largest finite magnitude.
    max: f64,
    /// `E5M2` has infinities (and IEEE overflow); `E4M3FN` saturates.
    has_inf: bool,
}

const E4M3: Format = Format {
    mbits: 3,
    bias: 7,
    max: 448.0,
    has_inf: false,
};
const E5M2: Format = Format {
    mbits: 2,
    bias: 15,
    max: 57344.0,
    has_inf: true,
};

/// Decode a code: `None` for a NaN, else the exact value (`±∞` included).
fn decode(b: u8, f: Format) -> Option<f64> {
    let ebits = 7 - f.mbits;
    let emax = (1u32 << ebits) - 1;
    let e = u32::from(b >> f.mbits) & emax;
    let m = u32::from(b) & ((1 << f.mbits) - 1);
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    if e == emax {
        if f.has_inf {
            return if m == 0 {
                Some(sign * f64::INFINITY)
            } else {
                None
            };
        }
        if m == (1 << f.mbits) - 1 {
            return None;
        }
    }
    let scale = f64::from(1u32 << f.mbits);
    let mag = if e == 0 {
        f64::from(m) / scale * pow2(1 - f.bias)
    } else {
        (1.0 + f64::from(m) / scale) * pow2(e as i32 - f.bias)
    };
    Some(sign * mag)
}

/// The quiet `f32` NaN carrying the sign of code `b` and the `payload` bits below the quiet bit.
fn nan_of(b: u8, payload: u32) -> f32 {
    f32::from_bits((u32::from(b & 0x80) << 24) | 0x7fc0_0000 | payload)
}

/// E4M3FN to `f32` (exact).
pub fn f8e4m3_to_f32(b: u8) -> f32 {
    decode(b, E4M3).map_or_else(|| nan_of(b, 0), |v| v as f32)
}

/// E5M2 to `f32` (exact). A NaN keeps its two mantissa bits as payload under the quiet bit,
/// exactly as decoding the `f16` whose top byte the code is does (`f16` to `f32` widening).
pub fn f8e5m2_to_f32(b: u8) -> f32 {
    decode(b, E5M2).map_or_else(|| nan_of(b, u32::from(b & 3) << 21), |v| v as f32)
}

/// Encode `x` (any `f32`) by round-to-nearest-even of its exact value onto the format's grid.
fn encode(x: f32, f: Format) -> u8 {
    encode_exact(f64::from(x), f)
}

/// Encode `x` (any `f64`) by round-to-nearest-even of its exact value onto the format's grid.
fn encode_exact(x: f64, f: Format) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    if x.is_nan() {
        // E4M3FN's only NaN is S.1111.111; E5M2's canonical quiet NaN is S.11111.10.
        return sign | if f.has_inf { 0x7e } else { 0x7f };
    }
    let a = x.abs();
    let emin = 1 - f.bias;
    let rounded = if a.is_infinite() {
        f64::INFINITY
    } else if a == 0.0 {
        0.0
    } else {
        // For a normal `a` (an f32 always is) the exponent field is floor(log2 a); an f64
        // subnormal reads as -1023 and is clamped to `emin`, which is all the rounding needs.
        let e = (((a.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(emin);
        let ulp = pow2(e - f.mbits as i32);
        let q = a / ulp; // exact: a power-of-two scaling
        let mut r = q.floor();
        let frac = q - r;
        if frac > 0.5 || (frac == 0.5 && r % 2.0 == 1.0) {
            r += 1.0;
        }
        r * ulp
    };
    let target = if rounded > f.max {
        if f.has_inf {
            f64::INFINITY
        } else {
            f.max
        }
    } else {
        rounded
    };
    // The code whose value is `target` (every finite grid value and ±∞ has exactly one positive
    // code).
    (0u8..0x80)
        .find(|&c| decode(c, f) == Some(target))
        .map_or(sign, |c| sign | c)
}

/// `f32` to E4M3FN: round to nearest even, saturating to ±448; NaN to `0x7F`/`0xFF`.
pub fn f32_to_f8e4m3(x: f32) -> u8 {
    encode(x, E4M3)
}

/// `f32` to E5M2: round to nearest even, overflowing to ±∞; NaN to `0x7E`/`0xFE`.
pub fn f32_to_f8e5m2(x: f32) -> u8 {
    encode(x, E5M2)
}

/// The two first-class FP8 formats, as the bit-level codec sees them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fp8Format {
    /// OCP E4M3FN: bias 7, 3 mantissa bits, saturating, NaN `S.1111.111`.
    E4M3,
    /// OCP E5M2: bias 15, 2 mantissa bits, IEEE infinities, canonical NaN `S.11111.10`.
    E5M2,
}

/// The constants of the bit-level codec for one format and one source width. `shift` is the
/// source mantissa width minus the format's, `rebias` the exponent rebias in code units
/// (`(src_bias - bias) << mbits`), `top` the largest code a finite rounding may produce (448 for
/// E4M3FN, the infinity code for E5M2), `min_normal_exp` the biased source exponent of the least
/// normal, `sub_scale_exp` the power of two that scales the least subnormal to 1, `nan` the code a
/// NaN encodes to.
#[derive(Clone, Copy, Debug)]
pub struct Fp8Codec {
    pub shift: u32,
    pub rebias: u64,
    pub top: u64,
    pub min_normal_exp: u64,
    pub sub_scale_exp: i32,
    pub nan: u8,
}

impl Fp8Format {
    /// The mantissa bits of the format.
    pub fn mbits(self) -> u32 {
        match self {
            Fp8Format::E4M3 => 3,
            Fp8Format::E5M2 => 2,
        }
    }

    /// The exponent bias of the format.
    pub fn bias(self) -> u32 {
        match self {
            Fp8Format::E4M3 => 7,
            Fp8Format::E5M2 => 15,
        }
    }

    /// The largest finite value of the format (448 for E4M3FN, 57344 for E5M2).
    pub fn max_finite(self) -> f64 {
        match self {
            Fp8Format::E4M3 => 448.0,
            Fp8Format::E5M2 => 57344.0,
        }
    }

    /// The codec constants for a source float whose significand has `src_mbits` stored bits and
    /// whose exponent bias is `src_bias` (`23`/`127` for `f32`, `52`/`1023` for `f64`).
    pub fn codec(self, src_mbits: u32, src_bias: u64) -> Fp8Codec {
        let (mbits, bias) = (self.mbits(), u64::from(self.bias()));
        Fp8Codec {
            shift: src_mbits - mbits,
            rebias: (src_bias - bias) << mbits,
            top: match self {
                Fp8Format::E4M3 => 0x7e,
                Fp8Format::E5M2 => 0x7c,
            },
            min_normal_exp: src_bias + 1 - bias,
            sub_scale_exp: bias as i32 - 1 + mbits as i32,
            nan: match self {
                Fp8Format::E4M3 => 0x7f,
                Fp8Format::E5M2 => 0x7e,
            },
        }
    }
}

/// `f32` to an FP8 code by the bit-level codec every backend runs: a normal result rounds on the
/// bit pattern (add `half - 1` plus the kept lsb, so a carry bumps the exponent), rebiases and
/// clamps to the largest code; a result below the least normal scales `|x|` by `2^k` (exact) and
/// rounds that to an integer, ties to even. The sign is copied, a NaN gets the format's NaN code.
pub fn encode_f32(x: f32, fmt: Fp8Format) -> u8 {
    let c = fmt.codec(23, 127);
    let b = x.to_bits();
    let sign = ((b >> 24) & 0x80) as u8;
    let a = b & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return sign | c.nan;
    }
    let code = if u64::from(a) < c.min_normal_exp << 23 {
        let scaled = f32::from_bits(a) * pow2_f32(c.sub_scale_exp);
        scaled.round_ties_even() as u64
    } else {
        let a = u64::from(a);
        let lsb = (a >> c.shift) & 1;
        let rounded = a + ((1u64 << (c.shift - 1)) - 1) + lsb;
        ((rounded >> c.shift) - c.rebias).min(c.top)
    };
    sign | code as u8
}

/// `f64` to an FP8 code in ONE rounding: the [`encode_f32`] algorithm on the `f64` bit pattern.
/// (`x as f32` first would round twice: a value just above a midpoint can land on it in `f32` and
/// then tie to the even code below.)
pub fn encode_f64(x: f64, fmt: Fp8Format) -> u8 {
    let c = fmt.codec(52, 1023);
    let b = x.to_bits();
    let sign = ((b >> 56) & 0x80) as u8;
    let a = b & 0x7fff_ffff_ffff_ffff;
    if a > 0x7ff0_0000_0000_0000 {
        return sign | c.nan;
    }
    let code = if a < c.min_normal_exp << 52 {
        let scaled = f64::from_bits(a) * f64::from(pow2_f32(c.sub_scale_exp));
        scaled.round_ties_even() as u64
    } else {
        let lsb = (a >> c.shift) & 1;
        let rounded = a + ((1u64 << (c.shift - 1)) - 1) + lsb;
        ((rounded >> c.shift) - c.rebias).min(c.top)
    };
    sign | code as u8
}

/// An FP8 code to `f32`, exactly: the bit-level decoder every backend runs. E4M3FN's NaN decodes
/// to the quiet NaN of its sign; an E5M2 NaN keeps its payload under the quiet bit, as the `f16`
/// it is the top byte of does.
pub fn decode_f32(code: u8, fmt: Fp8Format) -> f32 {
    let b = u32::from(code);
    let sign = (b & 0x80) << 24;
    let mbits = fmt.mbits();
    let e = (b & 0x7f) >> mbits;
    let m = b & ((1 << mbits) - 1);
    let emax = (1u32 << (7 - mbits)) - 1;
    let mag = match fmt {
        Fp8Format::E4M3 if b & 0x7f == 0x7f => 0x7fc0_0000,
        Fp8Format::E5M2 if e == emax => {
            if m == 0 {
                0x7f80_0000
            } else {
                0x7fc0_0000 | (m << 21)
            }
        }
        _ if e == 0 => (m as f32 * pow2_f32(-fmt.codec(23, 127).sub_scale_exp)).to_bits(),
        _ => ((e + 127 - fmt.bias()) << 23) | (m << (23 - mbits)),
    };
    f32::from_bits(sign | mag)
}

/// `x` (an `f32` register value) rounded onto the grid of `fmt`: `decode_f32(encode_f32(x))`.
pub fn round_f32(x: f32, fmt: Fp8Format) -> f32 {
    decode_f32(encode_f32(x, fmt), fmt)
}

/// `2^k` as an `f32`, for `k` in the normal range.
fn pow2_f32(k: i32) -> f32 {
    f32::from_bits(((127 + k) as u32) << 23)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(x: f32) -> u32 {
        x.to_bits()
    }

    /// All 256 E4M3FN codes: the anchors of the OCP table, sign symmetry, strict monotonicity of
    /// the 127 finite non-negative codes, and the single NaN pair.
    #[test]
    fn e4m3_decode_exhaustive() {
        assert_eq!(f8e4m3_to_f32(0x00), 0.0);
        assert_eq!(bits(f8e4m3_to_f32(0x80)), bits(-0.0));
        assert_eq!(f8e4m3_to_f32(0x01), 0.001953125); // 2^-9, the least subnormal
        assert_eq!(f8e4m3_to_f32(0x07), 0.013671875); // 7·2^-9, the greatest subnormal
        assert_eq!(f8e4m3_to_f32(0x08), 0.015625); // 2^-6, the least normal
        assert_eq!(f8e4m3_to_f32(0x38), 1.0);
        assert_eq!(f8e4m3_to_f32(0x39), 1.125);
        assert_eq!(f8e4m3_to_f32(0x7e), 448.0);
        assert_eq!(f8e4m3_to_f32(0xfe), -448.0);
        assert_eq!(bits(f8e4m3_to_f32(0x7f)), 0x7fc0_0000);
        assert_eq!(bits(f8e4m3_to_f32(0xff)), 0xffc0_0000);
        let mut prev = -1.0f32;
        for c in 0u8..0x7f {
            let v = f8e4m3_to_f32(c);
            assert!(v.is_finite() && v > prev, "code {c:#04x}: {v} after {prev}");
            assert_eq!(bits(f8e4m3_to_f32(c | 0x80)), bits(-v), "code {c:#04x}");
            prev = v;
        }
    }

    /// All 256 E5M2 codes: E5M2 is the top byte of an IEEE `f16`, which is how `ml_dtypes` defines
    /// it, so every code must decode to exactly `f16::from_bits(code << 8)` — NaN payloads too.
    #[test]
    fn e5m2_decode_exhaustive_matches_f16_top_byte() {
        for c in 0u8..=255 {
            let v = f8e5m2_to_f32(c);
            let h = half::f16::from_bits(u16::from(c) << 8).to_f32();
            assert_eq!(bits(v), bits(h), "code {c:#04x}");
        }
        assert_eq!(bits(f8e5m2_to_f32(0x7d)), 0x7fe0_0000);
        assert_eq!(bits(f8e5m2_to_f32(0xfe)), 0xffc0_0000);
        assert_eq!(f8e5m2_to_f32(0x3c), 1.0);
        assert_eq!(f8e5m2_to_f32(0x7b), 57344.0);
        assert_eq!(f8e5m2_to_f32(0x7c), f32::INFINITY);
        assert_eq!(f8e5m2_to_f32(0x01), f32::powi(2.0, -16)); // 2^-16
    }

    /// Encoding every finite code's own value returns the code (signed zeros included).
    #[test]
    fn encode_round_trips_every_code() {
        for c in 0u8..=255 {
            let v = f8e4m3_to_f32(c);
            if !v.is_nan() {
                assert_eq!(f32_to_f8e4m3(v), c, "e4m3 {c:#04x}");
            }
            let w = f8e5m2_to_f32(c);
            if !w.is_nan() {
                assert_eq!(f32_to_f8e5m2(w), c, "e5m2 {c:#04x}");
            }
        }
    }

    /// Every midpoint between neighbouring codes rounds to the even one, and one `f32` ULP either
    /// side of it to the nearer one.
    #[test]
    fn encode_rounds_half_to_even_at_every_midpoint() {
        type Codec = (fn(u8) -> f32, fn(f32) -> u8, u8);
        let codecs: [Codec; 2] = [
            (f8e4m3_to_f32, f32_to_f8e4m3, 0x7e),
            (f8e5m2_to_f32, f32_to_f8e5m2, 0x7b),
        ];
        for (dec, enc, top) in codecs {
            for c in 0u8..top {
                let (lo, hi) = (dec(c), dec(c + 1));
                let mid = (lo + hi) / 2.0; // exact: a few significant bits
                let even = if c.is_multiple_of(2) { c } else { c + 1 };
                assert_eq!(enc(mid), even, "midpoint of {c:#04x}");
                assert_eq!(enc(mid.next_down()), c, "below the midpoint of {c:#04x}");
                assert_eq!(enc(mid.next_up()), c + 1, "above the midpoint of {c:#04x}");
                assert_eq!(enc(-mid), even | 0x80, "negative midpoint of {c:#04x}");
            }
        }
    }

    #[test]
    fn encode_saturates_e4m3_and_overflows_e5m2() {
        assert_eq!(f32_to_f8e4m3(464.0), 0x7e); // tie between 448 and (absent) 480: even
        assert_eq!(f32_to_f8e4m3(1e6), 0x7e);
        assert_eq!(f32_to_f8e4m3(f32::INFINITY), 0x7e);
        assert_eq!(f32_to_f8e4m3(f32::NEG_INFINITY), 0xfe);
        assert_eq!(f32_to_f8e4m3(f32::NAN), 0x7f);
        assert_eq!(f32_to_f8e4m3(-f32::NAN), 0xff);
        assert_eq!(f32_to_f8e4m3(0.0009765625), 0x00); // 2^-10: tie between 0 and 2^-9
        assert_eq!(f32_to_f8e4m3(1e-30), 0x00);
        assert_eq!(f32_to_f8e4m3(-1e-30), 0x80);
        assert_eq!(f32_to_f8e5m2(61439.996), 0x7b);
        assert_eq!(f32_to_f8e5m2(61440.0), 0x7c); // tie between 57344 and 2^16: even is inf
        assert_eq!(f32_to_f8e5m2(1e6), 0x7c);
        assert_eq!(f32_to_f8e5m2(f32::NEG_INFINITY), 0xfc);
        assert_eq!(f32_to_f8e5m2(f32::NAN), 0x7e);
        assert_eq!(f32_to_f8e5m2(f32::powi(2.0, -17)), 0x00); // 2^-17: tie between 0 and 2^-16
    }

    /// The bit-level codec the backends run is the reference, on every code and on a sweep of
    /// `f32` patterns: each of the 65536 top halves with the low half at 0, 1, a midpoint and all
    /// ones (every exponent, both signs, NaNs and infinities, the rounding boundaries of both
    /// formats), plus every midpoint and its two `f32` neighbours.
    #[test]
    fn fast_codec_matches_the_reference() {
        for c in 0u8..=255 {
            assert_eq!(
                bits(decode_f32(c, Fp8Format::E4M3)),
                bits(f8e4m3_to_f32(c)),
                "e4m3 decode {c:#04x}"
            );
            assert_eq!(
                bits(decode_f32(c, Fp8Format::E5M2)),
                bits(f8e5m2_to_f32(c)),
                "e5m2 decode {c:#04x}"
            );
        }
        let check = |x: f32| {
            assert_eq!(
                encode_f32(x, Fp8Format::E4M3),
                f32_to_f8e4m3(x),
                "e4m3 encode {:#010x}",
                x.to_bits()
            );
            assert_eq!(
                encode_f32(x, Fp8Format::E5M2),
                f32_to_f8e5m2(x),
                "e5m2 encode {:#010x}",
                x.to_bits()
            );
            assert_eq!(encode_f64(f64::from(x), Fp8Format::E4M3), f32_to_f8e4m3(x));
            assert_eq!(encode_f64(f64::from(x), Fp8Format::E5M2), f32_to_f8e5m2(x));
        };
        for hi in 0u32..=0xffff {
            for lo in [0u32, 1, 0x8000, 0xffff] {
                check(f32::from_bits((hi << 16) | lo));
            }
        }
        for (dec, top) in [
            (f8e4m3_to_f32 as fn(u8) -> f32, 0x7e),
            (f8e5m2_to_f32, 0x7b),
        ] {
            for c in 0u8..top {
                let mid = (dec(c) + dec(c + 1)) / 2.0;
                for x in [mid, mid.next_down(), mid.next_up(), -mid] {
                    check(x);
                }
            }
        }
    }

    /// `encode_f64` rounds an `f64` once. One `f64` ULP above a midpoint rounds up, where going
    /// through `f32` lands exactly on the midpoint and ties to the even code below; every finite
    /// value near a midpoint agrees with the exact reference.
    #[test]
    fn encode_f64_rounds_once() {
        // 1.0625 is the midpoint of 1.0 (0x38) and 1.125 (0x39) in E4M3FN.
        let above = 1.0625f64 + f64::EPSILON;
        assert_eq!(above as f32, 1.0625f32);
        assert_eq!(encode_f32(above as f32, Fp8Format::E4M3), 0x38);
        assert_eq!(encode_f64(above, Fp8Format::E4M3), 0x39);
        assert_eq!(encode_f64(-above, Fp8Format::E4M3), 0xb9);
        // 1.125 is the midpoint of 1.0 (0x3c) and 1.25 (0x3d) in E5M2.
        let above = 1.125f64 + f64::EPSILON;
        assert_eq!(encode_f64(above, Fp8Format::E5M2), 0x3d);
        let formats = [(Fp8Format::E4M3, E4M3), (Fp8Format::E5M2, E5M2)];
        for (fmt, reference) in formats {
            let dec = |c: u8| f64::from(decode_f32(c, fmt));
            for c in 0u8..0x7b {
                let mid = (dec(c) + dec(c + 1)) / 2.0;
                for x in [
                    mid,
                    mid.next_down(),
                    mid.next_up(),
                    mid * (1.0 + 1e-12),
                    mid * (1.0 - 1e-12),
                    -mid.next_up(),
                ] {
                    assert_eq!(
                        encode_f64(x, fmt),
                        encode_exact(x, reference),
                        "{fmt:?} {x:e}"
                    );
                }
            }
            for x in [
                0.0,
                -0.0,
                f64::MIN_POSITIVE,
                -5e-324,
                1e300,
                -1e300,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                -f64::NAN,
                448.0,
                464.0,
                464.0 - 1e-9,
                57344.0,
                61440.0,
                61440.0 - 1e-9,
            ] {
                assert_eq!(
                    encode_f64(x, fmt),
                    encode_exact(x, reference),
                    "{fmt:?} {x:e}"
                );
            }
        }
    }

    /// `round_f32` is the identity on every grid value and idempotent everywhere.
    #[test]
    fn round_f32_is_idempotent() {
        for fmt in [Fp8Format::E4M3, Fp8Format::E5M2] {
            for c in 0u8..=255 {
                let v = decode_f32(c, fmt);
                if !v.is_nan() {
                    assert_eq!(bits(round_f32(v, fmt)), bits(v), "{fmt:?} {c:#04x}");
                }
            }
            for x in [0.3f32, -7.77, 1e-5, 1e5, f32::INFINITY, f32::NAN] {
                let r = round_f32(x, fmt);
                assert_eq!(bits(round_f32(r, fmt)), bits(r));
            }
        }
    }
}
