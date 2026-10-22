//! The general vectorizer's half-precision lanes: `<4 x bf16>` and `<4 x f16>`.
//!
//! Every backend computes a bf16 or f16 value in an `f32` register and rounds it to its 16-bit grid
//! at four points only: a load, a store, a constant and a cast (`wukong_opt::vectorize`'s module
//! docs). So a vector of four half lanes is an `f32x4` in a register and four 2-byte elements in
//! memory, and this module is the whole difference between it and an `f32x4`: the load that
//! widens four stored elements, the store that rounds and packs four lanes, and the lane-wise grid
//! rounding of a cast. Each is the lane-wise form of the scalar path, bit for bit:
//!
//! * **bf16** is the top half of an `f32`. Widening is a shift; rounding is the scalar
//!   `round_to_bf16`'s integer round-to-nearest-even, a NaN keeping its top half with the quiet
//!   bit set (`wukong_runtime::f32_to_bf16_bits`).
//! * **f16** has its own exponent and mantissa widths, and the scalar path converts through the
//!   runtime shims `wukong_f32_to_f16_bits` / `wukong_f16_bits_to_f32` — the `half` crate, which
//!   is F16C's `vcvtps2ph`/`vcvtph2ps` on a host that has it and an integer software routine on
//!   one that does not (the two agree on every input; the tests pin both). Cranelift 0.124 has no
//!   vector half conversion, so [`f32x4_to_f16_bits`] and [`f16_bits_to_f32x4`] build it from
//!   SSE2 integer operations, with one IEEE `f32` addition or subtraction for the subnormal range
//!   whose result is exact or correctly rounded by construction (see each function). Nothing here
//!   depends on the host's half-conversion instructions, so a baseline `--target-cpu=x86-64` build
//!   computes the same bits.
//!
//! The f16 conversions follow F. Giesen's branch-free `float_to_half_fast3_rtne` and
//! `half_to_float` (public domain), with the NaN payload rule the `half` crate and F16C share: a
//! NaN keeps the top ten bits of its payload and gains the quiet bit, both ways.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types, Endianness, InstBuilder, MemFlags, Type, Value};
use cranelift_frontend::FunctionBuilder;
use wukong_mir::MirType;

/// The lane type of the general vectorizer's `<4 x bf16>` or `<4 x f16>` — four half lanes held
/// as an `f32x4`, stored as four 2-byte elements — or `None` for any other type.
pub(crate) fn half_x4(t: &MirType) -> Option<&MirType> {
    match t {
        MirType::Vec(lane, 4) if matches!(**lane, MirType::BF16 | MirType::F16) => Some(lane),
        _ => None,
    }
}

/// Load four 2-byte `lane` elements at `addr` into the `f32x4` register form, each widened exactly
/// as the scalar load widens one.
pub(crate) fn load_x4(
    b: &mut FunctionBuilder,
    lane: &MirType,
    flags: MemFlags,
    addr: Value,
) -> Value {
    let ext = b.ins().uload16x4(flags, addr, 0);
    if *lane == MirType::BF16 {
        // The 16 bits become the top half of an f32.
        let shifted = b.ins().ishl_imm(ext, 16);
        bitcast(b, types::F32X4, shifted)
    } else {
        f16_bits_to_f32x4(b, ext)
    }
}

/// Store the four lanes of the `f32x4` `v` at `addr` as four 2-byte `lane` elements, each rounded
/// exactly as the scalar store rounds one. Every lane's bits are in `0..=0xffff` before the pack,
/// so the saturating narrow (`packusdw`, or Cranelift's SSE2 sequence for it) truncates nothing.
pub(crate) fn store_x4(
    b: &mut FunctionBuilder,
    lane: &MirType,
    flags: MemFlags,
    v: Value,
    addr: Value,
) {
    let bits = if *lane == MirType::BF16 {
        let rounded = round_to_bf16_x4(b, v);
        let wide = bitcast(b, types::I32X4, rounded);
        b.ins().ushr_imm(wide, 16)
    } else {
        f32x4_to_f16_bits(b, v)
    };
    let packed = b.ins().unarrow(bits, bits);
    let wide = bitcast(b, types::I64X2, packed);
    let low = b.ins().extractlane(wide, 0);
    b.ins().store(flags, low, addr, 0);
}

/// Each lane of the `f32x4` `x` rounded to the `lane` grid (bf16 or f16) and held as an `f32` —
/// the scalar `round_to_bf16` / `round_to_f16` (an `f32 -> f16 -> f32` round trip through the
/// shims) on every lane.
pub(crate) fn round_x4(b: &mut FunctionBuilder, lane: &MirType, x: Value) -> Value {
    if *lane == MirType::BF16 {
        round_to_bf16_x4(b, x)
    } else {
        let h = f32x4_to_f16_bits(b, x);
        f16_bits_to_f32x4(b, h)
    }
}

/// The scalar `round_to_bf16` on each lane of an `f32x4`: integer round-to-nearest-even on the
/// bits, and the quiet-NaN rule (keep the top half, set the quiet bit), with a lane mask from an
/// unordered compare where the scalar form tests `|bits| > 0x7f800000`.
pub(crate) fn round_to_bf16_x4(b: &mut FunctionBuilder, x: Value) -> Value {
    let bits = bitcast(b, types::I32X4, x);
    let shr = b.ins().ushr_imm(bits, 16);
    let one = splat_i32(b, 1);
    let lsb = b.ins().band(shr, one);
    let half = splat_i32(b, 0x7fff);
    let bias = b.ins().iadd(lsb, half);
    let summed = b.ins().iadd(bits, bias);
    let top = splat_i32(b, 0xffff_0000);
    let nonnan = b.ins().band(summed, top);
    let masked = b.ins().band(bits, top);
    let quiet = splat_i32(b, 0x0040_0000);
    let nanres = b.ins().bor(masked, quiet);
    let isnan = b
        .ins()
        .fcmp(cranelift_codegen::ir::condcodes::FloatCC::Unordered, x, x);
    let isnan = bitcast(b, types::I32X4, isnan);
    let resbits = b.ins().bitselect(isnan, nanres, nonnan);
    bitcast(b, types::F32X4, resbits)
}

/// The f16 encodings of the four lanes of the `f32x4` `x`, round to nearest even, as an `i32x4`
/// whose lanes hold the 16 bits zero-extended — `wukong_runtime::f32_to_f16_bits` on every lane.
///
/// Every lane is computed four ways and the right one selected by the lane's magnitude `a`
/// (the bits without the sign), all compares signed because `a < 2^31`:
///
/// * **normal** (`2^-14 <= |x| < 65536`, `0x38800000 <= a < 0x47800000`): rebias the exponent by
///   `-112 << 23`, then round at bit 13 to nearest even by adding `0xfff` plus the bit that becomes
///   the result's last one, and shift. A carry out of the mantissa is the correct next binade, and
///   out of the largest binade it is infinity (`65520` and up round to `0x7c00`).
/// * **subnormal or zero** (`a < 0x38800000`): `|x| + 0.5` in `f32`. The sum lies in `[0.5, 1)`,
///   whose spacing is `2^-24` — the f16 subnormal step — so the one IEEE round-to-nearest-even
///   addition rounds `|x|` to a multiple of `2^-24` exactly as the scalar routine does, ties to the
///   even multiple, and the sum's bits minus those of `0.5` are that multiple: the subnormal
///   encoding, or `0x400` (the smallest normal) when it rounds up into the next binade. An `f32`
///   subnormal input is below `2^-126`, far under half the smallest f16 subnormal, so a
///   denormals-are-zero mode could not change the result, and the sum is never subnormal.
/// * **too large or infinite** (`a >= 0x47800000`, `|x| >= 65536`): infinity, `0x7c00`.
/// * **NaN** (`a > 0x7f800000`): `0x7e00` (exponent all ones, quiet bit) with the top ten payload
///   bits, the `half` crate's and F16C's rule.
///
/// The sign is the f32 sign moved to bit 15.
pub(crate) fn f32x4_to_f16_bits(b: &mut FunctionBuilder, x: Value) -> Value {
    let bits = bitcast(b, types::I32X4, x);
    let sign_mask = splat_i32(b, 0x8000_0000);
    let sign = b.ins().band(bits, sign_mask);
    let a = b.ins().bxor(bits, sign);
    // Normal: (a - (112 << 23) + 0xfff + lsb) >> 13, the constant folded to 0xc8000fff (wrapping).
    let top = b.ins().ushr_imm(a, 13);
    let one = splat_i32(b, 1);
    let lsb = b.ins().band(top, one);
    let rebias = splat_i32(b, 0xc800_0fff);
    let biased = b.ins().iadd(a, rebias);
    let rounded = b.ins().iadd(biased, lsb);
    let normal = b.ins().ushr_imm(rounded, 13);
    // Subnormal or zero: the bits of |x| + 0.5, less those of 0.5.
    let af = bitcast(b, types::F32X4, a);
    let half = splat_f32(b, 0.5);
    let sum = b.ins().fadd(af, half);
    let sum_bits = bitcast(b, types::I32X4, sum);
    let half_bits = splat_i32(b, 0x3f00_0000);
    let sub = b.ins().isub(sum_bits, half_bits);
    // NaN: the top ten payload bits under 0x7e00.
    let payload_mask = splat_i32(b, 0x3ff);
    let payload = b.ins().band(top, payload_mask);
    let qnan = splat_i32(b, 0x7e00);
    let nan = b.ins().bor(payload, qnan);
    let min_normal = splat_i32(b, 0x3880_0000);
    let is_sub = b.ins().icmp(IntCC::SignedLessThan, a, min_normal);
    let r = b.ins().bitselect(is_sub, sub, normal);
    let max_finite = splat_i32(b, 0x477f_ffff);
    let is_inf = b.ins().icmp(IntCC::SignedGreaterThan, a, max_finite);
    let inf = splat_i32(b, 0x7c00);
    let r = b.ins().bitselect(is_inf, inf, r);
    let f32_inf = splat_i32(b, 0x7f80_0000);
    let is_nan = b.ins().icmp(IntCC::SignedGreaterThan, a, f32_inf);
    let r = b.ins().bitselect(is_nan, nan, r);
    let sign16 = b.ins().ushr_imm(sign, 16);
    b.ins().bor(r, sign16)
}

/// The `f32x4` values of four f16 encodings, the `i32x4` `h` holding each zero-extended in its
/// lane — `wukong_runtime::f16_bits_to_f32` on every lane, which is exact.
///
/// With `e` the magnitude bits (`h & 0x7fff`) moved up 13 places into the f32 positions:
///
/// * **normal** (`0x400 <= e < 0x7c00`): add `112 << 23` to rebias the exponent.
/// * **zero or subnormal** (`e < 0x400`, a value `m · 2^-24`): build `2^-14 · (1 + m/1024)` by
///   adding `113 << 23`, then subtract `2^-14` in `f32`. Both operands are normal and within a
///   factor of two of each other, so the difference is exact (Sterbenz) — `m · 2^-24`, a normal
///   `f32` or `+0.0` — and no flush-to-zero or denormals-are-zero mode can touch it.
/// * **infinity or NaN** (`e >= 0x7c00`): add `224 << 23`, carrying the exponent to all ones and
///   keeping the payload; a NaN (`e > 0x7c00`) also gains the quiet bit `0x00400000`, the `half`
///   crate's and F16C's rule.
///
/// The sign moves from bit 15 to bit 31.
pub(crate) fn f16_bits_to_f32x4(b: &mut FunctionBuilder, h: Value) -> Value {
    let mag_mask = splat_i32(b, 0x7fff);
    let e = b.ins().band(h, mag_mask);
    let sign16 = b.ins().bxor(h, e);
    let sign = b.ins().ishl_imm(sign16, 16);
    let shifted = b.ins().ishl_imm(e, 13);
    // Normal.
    let rebias = splat_i32(b, 0x3800_0000);
    let normal = b.ins().iadd(shifted, rebias);
    // Infinity or NaN, a NaN quieted.
    let special_rebias = splat_i32(b, 0x7000_0000);
    let special = b.ins().iadd(shifted, special_rebias);
    let f16_inf = splat_i32(b, 0x7c00);
    let is_nan = b.ins().icmp(IntCC::SignedGreaterThan, e, f16_inf);
    let quiet_bit = splat_i32(b, 0x0040_0000);
    let quiet = b.ins().band(is_nan, quiet_bit);
    let special = b.ins().bor(special, quiet);
    // Zero or subnormal: (2^-14 + m * 2^-24) - 2^-14.
    let magic = splat_i32(b, 0x3880_0000);
    let biased = b.ins().iadd(shifted, magic);
    let biased_f = bitcast(b, types::F32X4, biased);
    let magic_f = bitcast(b, types::F32X4, magic);
    let diff = b.ins().fsub(biased_f, magic_f);
    let sub = bitcast(b, types::I32X4, diff);
    let min_normal = splat_i32(b, 0x400);
    let is_small = b.ins().icmp(IntCC::SignedLessThan, e, min_normal);
    let r = b.ins().bitselect(is_small, sub, normal);
    let max_finite = splat_i32(b, 0x7bff);
    let is_special = b.ins().icmp(IntCC::SignedGreaterThan, e, max_finite);
    let r = b.ins().bitselect(is_special, special, r);
    let r = b.ins().bor(r, sign);
    bitcast(b, types::F32X4, r)
}

/// A lane-preserving bitcast between 128-bit vector types (little-endian lane order).
fn bitcast(b: &mut FunctionBuilder, ty: Type, x: Value) -> Value {
    if b.func.dfg.value_type(x) == ty {
        return x;
    }
    let flags = MemFlags::new().with_endianness(Endianness::Little);
    b.ins().bitcast(ty, flags, x)
}

fn splat_i32(b: &mut FunctionBuilder, k: u32) -> Value {
    let c = b.ins().iconst(types::I32, i64::from(k as i32));
    b.ins().splat(types::I32X4, c)
}

fn splat_f32(b: &mut FunctionBuilder, k: f32) -> Value {
    let c = b.ins().f32const(k);
    b.ins().splat(types::F32X4, c)
}
