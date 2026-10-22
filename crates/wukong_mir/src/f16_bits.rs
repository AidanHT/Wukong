//! IEEE binary16 (`f16`) conversions in integer arithmetic, bit-identical to the scalar path.
//!
//! The scalar path converts every f16 element through the runtime shims `wukong_f32_to_f16_bits` /
//! `wukong_f16_bits_to_f32`, which are the `half` crate: F16C's `vcvtps2ph`/`vcvtph2ps` on a host
//! that has it, the crate's integer software routine on one that does not, the two agreeing on every
//! input. The vector-kernel reference evaluator ([`crate::VecValue::round_f16`], what the
//! interpreter runs for a recipe's `LoadF16`, `StoreF16` and `RoundF16`) must round exactly as they
//! do, but it cannot call them: this crate depends on `wukong_span` and `wukong_types` only, because
//! every consumer of the IR depends on it. So the software routine is transcribed here (half 2.7.1,
//! `binary16/arch.rs`: `f32_to_f16_fallback` and `f16_to_f32_fallback`).
//!
//! The tests check the transcription bit for bit against `half` itself (a dev-dependency, which
//! reaches no consumer), both its host path (`from_f32`/`to_f32`) and its software one
//! (`from_f32_const`/`to_f32_const`): f16 to f32 on all 65536 encodings, and f32 to f16 on every
//! class of input: every sign and exponent with every value of the eleven mantissa bits a normal
//! result keeps or rounds on, under a zero and a non-zero tail.

/// The f16 encoding nearest `value`: round to nearest, ties to even; `65520` and above (in
/// magnitude) to infinity; a NaN keeps the top ten bits of its payload and gains the quiet bit.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = x & 0x8000_0000;
    let exp = x & 0x7f80_0000;
    let man = x & 0x007f_ffff;
    // Infinity or NaN: a NaN keeps its payload's top ten bits and is quieted.
    if exp == 0x7f80_0000 {
        let nan_bit = if man == 0 { 0 } else { 0x0200 };
        return ((sign >> 16) | 0x7c00 | nan_bit | (man >> 13)) as u16;
    }
    let half_sign = sign >> 16;
    let unbiased_exp = ((exp >> 23) as i32) - 127;
    let half_exp = unbiased_exp + 15;
    // Overflow: infinity.
    if half_exp >= 0x1f {
        return (half_sign | 0x7c00) as u16;
    }
    // Underflow: a subnormal result, or a signed zero below half the smallest subnormal.
    if half_exp <= 0 {
        if 14 - half_exp > 24 {
            return half_sign as u16;
        }
        let man = man | 0x0080_0000;
        let mut half_man = man >> (14 - half_exp);
        let round_bit = 1 << (13 - half_exp);
        if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
            half_man += 1;
        }
        return (half_sign | half_man) as u16;
    }
    // A normal result, rounded at bit 13 (a carry out of the mantissa bumps the exponent, up to
    // infinity, which is the right answer).
    let half_exp = (half_exp as u32) << 10;
    let half_man = man >> 13;
    let round_bit = 0x0000_1000;
    if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
        ((half_sign | half_exp | half_man) + 1) as u16
    } else {
        (half_sign | half_exp | half_man) as u16
    }
}

/// The `f32` an f16 encoding denotes, exactly; a NaN keeps its payload and gains the quiet bit.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    // Signed zero.
    if bits & 0x7fff == 0 {
        return f32::from_bits(u32::from(bits) << 16);
    }
    let half_sign = u32::from(bits & 0x8000);
    let half_exp = u32::from(bits & 0x7c00);
    let half_man = u32::from(bits & 0x03ff);
    // Infinity, or a NaN that keeps its payload and is quieted.
    if half_exp == 0x7c00 {
        if half_man == 0 {
            return f32::from_bits((half_sign << 16) | 0x7f80_0000);
        }
        return f32::from_bits((half_sign << 16) | 0x7fc0_0000 | (half_man << 13));
    }
    let sign = half_sign << 16;
    let unbiased_exp = ((half_exp as i32) >> 10) - 15;
    // A subnormal, normalized by moving its leading one to the hidden bit.
    if half_exp == 0 {
        let e = (half_man as u16).leading_zeros() - 6;
        let exp = (127 - 15 - e) << 23;
        let man = (half_man << (14 + e)) & 0x7f_ffff;
        return f32::from_bits(sign | exp | man);
    }
    let exp = ((unbiased_exp + 127) as u32) << 23;
    let man = (half_man & 0x03ff) << 13;
    f32::from_bits(sign | exp | man)
}

/// `value` rounded to the f16 grid and widened back: what an f16 element stored from it reads back
/// as (`vcvtps2ph` with round to nearest even, then `vcvtph2ps`).
pub fn round_f16(value: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every crate that consumes the IR depends on this one, so what this crate depends on reaches
    /// nearly the whole build: the README promises `wukong_span` and `wukong_types` only, and this
    /// pins it (a `[dev-dependencies]` entry reaches no consumer and is allowed).
    #[test]
    fn the_crate_depends_on_wukong_span_and_wukong_types_only() {
        let manifest = include_str!("../Cargo.toml");
        let mut section = "";
        let mut deps = Vec::new();
        for line in manifest.lines().map(str::trim) {
            if line.starts_with('[') {
                section = line;
                continue;
            }
            let linked =
                section.ends_with("dependencies]") && !section.ends_with("dev-dependencies]");
            if linked && !line.is_empty() && !line.starts_with('#') {
                deps.push(
                    line.split(['=', '.', ' '])
                        .next()
                        .unwrap_or(line)
                        .to_string(),
                );
            }
        }
        assert_eq!(
            deps,
            ["wukong_span", "wukong_types"],
            "wukong_mir gained a dependency; every consumer of the IR now builds it"
        );
    }

    /// Every f16 encoding widens as `half` widens it, on its host path and its software one: every
    /// NaN keeps its payload and gains the quiet bit, every subnormal is exact.
    #[test]
    fn every_f16_encoding_widens_like_half() {
        for h in 0..=u16::MAX {
            let got = f16_bits_to_f32(h).to_bits();
            let host = half::f16::from_bits(h).to_f32().to_bits();
            let soft = half::f16::from_bits(h).to_f32_const().to_bits();
            assert_eq!(got, host, "f16 {h:#06x}: {got:#010x}, half {host:#010x}");
            assert_eq!(
                got, soft,
                "f16 {h:#06x}: {got:#010x}, half (soft) {soft:#010x}"
            );
        }
    }

    /// f32 to f16, and the round trip, match `half` on every class of input. The result depends on
    /// the sign, the exponent, the ten mantissa bits a normal result keeps (bits 13-22), the bit it
    /// rounds on (12) and whether anything below is set; a subnormal result keeps fewer and rounds
    /// higher, still inside bits 12-22 or on the hidden bit, and a NaN keeps bits 13-22. So every
    /// sign, every exponent and every value of bits 12-22, under the tails 0, 1, 0x800 and 0xfff,
    /// is every input class: every tie, every overflow to infinity, every subnormal boundary and
    /// every NaN payload's top ten bits, quiet or signalling.
    #[test]
    fn every_f32_class_narrows_like_half() {
        for sign in [0u32, 0x8000_0000] {
            for exp in 0u32..256 {
                for top in 0u32..2048 {
                    for tail in [0u32, 1, 0x800, 0xfff] {
                        let x = f32::from_bits(sign | (exp << 23) | (top << 12) | tail);
                        let got = f32_to_f16_bits(x);
                        let host = half::f16::from_f32(x).to_bits();
                        let soft = half::f16::from_f32_const(x).to_bits();
                        let bits = x.to_bits();
                        assert_eq!(got, host, "f32 {bits:#010x}: {got:#06x}, half {host:#06x}");
                        assert_eq!(got, soft, "f32 {bits:#010x}: {got:#06x}, soft {soft:#06x}");
                        let round = round_f16(x).to_bits();
                        let want = half::f16::from_f32(x).to_f32().to_bits();
                        assert_eq!(round, want, "round_f16({bits:#010x}): {round:#010x}");
                    }
                }
            }
        }
    }
}
