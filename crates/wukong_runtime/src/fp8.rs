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
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    if x.is_nan() {
        // E4M3FN's only NaN is S.1111.111; E5M2's canonical quiet NaN is S.11111.10.
        return sign | if f.has_inf { 0x7e } else { 0x7f };
    }
    let a = f64::from(x).abs();
    let emin = 1 - f.bias;
    let rounded = if a.is_infinite() {
        f64::INFINITY
    } else if a == 0.0 {
        0.0
    } else {
        // `a` is a normal f64 (an f32 always is), so its exponent field is floor(log2 a).
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
}
