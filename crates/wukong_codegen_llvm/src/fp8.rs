//! The 8-bit floats `f8e4m3` (OCP E4M3FN) and `f8e5m2` (OCP E5M2) in textual LLVM IR.
//!
//! LLVM has no FP8 type, and the language would not want one: an f8 value lives in a `float`
//! register on its grid and is stored as its `i8` code, rounded at the interpreter's observation
//! boundaries (a constant, a load, a store, an `fptrunc` to it, an `fpext` from it). The rounding is
//! `wukong_runtime::fp8`'s bit-level codec written out instruction for instruction — the sequence
//! the Cranelift backend emits — so the module computes what the oracle computes; an `f64` source
//! is encoded from its own bits, once. The encoder's subnormal branch needs `llvm.roundeven` and
//! the saturating `llvm.fptosi.sat`, which [`encoder_intrinsics`] declares.

use std::fmt::Write as _;

use wukong_mir::{CastKind, Function, MirType, Op};
use wukong_runtime::fp8::Fp8Format;

use crate::{reg_mir, Emitter, Signature};

/// The FP8 format of an f8 MIR type, or `None` for any other type.
pub(crate) fn f8_format(t: &MirType) -> Option<Fp8Format> {
    match t {
        MirType::F8E4M3 => Some(Fp8Format::E4M3),
        MirType::F8E5M2 => Some(Fp8Format::E5M2),
        _ => None,
    }
}

/// The register type an instruction encodes into an f8 code from, when it emits the encoder: the
/// stored `float` of an f8 store, the source of an `fptrunc` to an f8, the f8 `float` an `fpext`
/// rounds before widening.
pub(crate) fn encoder_source(f: &Function, op: &Op) -> Option<MirType> {
    match op {
        Op::Store { value, .. } if f.value_type(*value).is_f8() => Some(MirType::F32),
        Op::Cast(CastKind::FpTrunc, v, to) if to.is_f8() => Some(reg_mir(f.value_type(*v))),
        Op::Cast(CastKind::FpExt, v, _) if f.value_type(*v).is_f8() => Some(MirType::F32),
        _ => None,
    }
}

/// The two intrinsics the encoder calls on a `src` (`float` or `double`) value.
pub(crate) fn encoder_intrinsics(src: &MirType) -> [(String, Signature); 2] {
    let (ft, it, suffix) = if *src == MirType::F64 {
        ("double", "i64", "f64")
    } else {
        ("float", "i32", "f32")
    };
    [
        (
            format!("llvm.roundeven.{suffix}"),
            (ft.to_owned(), vec![ft.to_owned()]),
        ),
        (
            format!("llvm.fptosi.sat.{it}.{suffix}"),
            (it.to_owned(), vec![ft.to_owned()]),
        ),
    ]
}

/// A floating constant in LLVM's hexadecimal spelling (the bits of the exactly widened double).
fn hex(v: f64) -> String {
    format!("0x{:016X}", v.to_bits())
}

impl Emitter<'_> {
    /// The FP8 code of the register value `value` (register type `reg`: `float` or `double`) as
    /// an `i32` in `0..=255` — `wukong_runtime::fp8::encode_f32` / `encode_f64`.
    pub(crate) fn f8_encode(
        &self,
        out: &mut String,
        value: &str,
        reg: &MirType,
        fmt: Fp8Format,
    ) -> String {
        let wide = *reg == MirType::F64;
        let (it, ft, suffix, c, mant, inf, abs, sign_shift): (_, _, _, _, u32, i64, i64, u32) =
            if wide {
                (
                    "i64",
                    "double",
                    "f64",
                    fmt.codec(52, 1023),
                    52,
                    0x7ff0_0000_0000_0000,
                    0x7fff_ffff_ffff_ffff,
                    56,
                )
            } else {
                (
                    "i32",
                    "float",
                    "f32",
                    fmt.codec(23, 127),
                    23,
                    0x7f80_0000,
                    0x7fff_ffff,
                    24,
                )
            };
        let t = self.fresh();
        let half_m1 = (1i64 << (c.shift - 1)) - 1;
        let min_normal = (c.min_normal_exp << mant) as i64;
        let scale = hex(2f64.powi(c.sub_scale_exp));
        let lines = [
            format!("{t}.bits = bitcast {ft} {value} to {it}"),
            format!("{t}.hi = lshr {it} {t}.bits, {sign_shift}"),
            format!("{t}.sign = and {it} {t}.hi, 128"),
            format!("{t}.a = and {it} {t}.bits, {abs}"),
            format!("{t}.nan = icmp ugt {it} {t}.a, {inf}"),
            format!("{t}.lsbsh = lshr {it} {t}.a, {}", c.shift),
            format!("{t}.lsb = and {it} {t}.lsbsh, 1"),
            format!("{t}.biased = add {it} {t}.a, {half_m1}"),
            format!("{t}.rnd = add {it} {t}.biased, {t}.lsb"),
            format!("{t}.csh = lshr {it} {t}.rnd, {}", c.shift),
            format!("{t}.cn = sub {it} {t}.csh, {}", c.rebias),
            format!("{t}.over = icmp sgt {it} {t}.cn, {}", c.top),
            format!(
                "{t}.normal = select i1 {t}.over, {it} {}, {it} {t}.cn",
                c.top
            ),
            format!("{t}.ax = bitcast {it} {t}.a to {ft}"),
            format!("{t}.scaled = fmul {ft} {t}.ax, {scale}"),
            format!("{t}.r = call {ft} @llvm.roundeven.{suffix}({ft} {t}.scaled)"),
            format!("{t}.sub = call {it} @llvm.fptosi.sat.{it}.{suffix}({ft} {t}.r)"),
            format!("{t}.issub = icmp ult {it} {t}.a, {min_normal}"),
            format!("{t}.fin = select i1 {t}.issub, {it} {t}.sub, {it} {t}.normal"),
            format!("{t}.code = select i1 {t}.nan, {it} {}, {it} {t}.fin", c.nan),
        ];
        for l in lines {
            let _ = writeln!(out, "  {l}");
        }
        if wide {
            let _ = writeln!(out, "  {t}.sc = or i64 {t}.code, {t}.sign");
            let _ = writeln!(out, "  {t} = trunc i64 {t}.sc to i32");
        } else {
            let _ = writeln!(out, "  {t} = or i32 {t}.code, {t}.sign");
        }
        t
    }

    /// The `float` an FP8 code decodes to, exactly — `wukong_runtime::fp8::decode_f32`. `code` is
    /// an `i32` holding the code zero-extended.
    pub(crate) fn f8_decode(&self, out: &mut String, code: &str, fmt: Fp8Format) -> String {
        let t = self.fresh();
        let mbits = fmt.mbits();
        let c = fmt.codec(23, 127);
        let mut lines = vec![
            format!("{t}.s = and i32 {code}, 128"),
            format!("{t}.sign = shl i32 {t}.s, 24"),
            format!("{t}.low7 = and i32 {code}, 127"),
            format!("{t}.e = lshr i32 {t}.low7, {mbits}"),
            format!("{t}.m = and i32 {code}, {}", (1 << mbits) - 1),
            format!("{t}.e32 = add i32 {t}.e, {}", 127 - fmt.bias()),
            format!("{t}.e32s = shl i32 {t}.e32, 23"),
            format!("{t}.ms = shl i32 {t}.m, {}", 23 - mbits),
            format!("{t}.normal = or i32 {t}.e32s, {t}.ms"),
            format!("{t}.mf = sitofp i32 {t}.m to float"),
            format!(
                "{t}.subf = fmul float {t}.mf, {}",
                hex(2f64.powi(-c.sub_scale_exp))
            ),
            format!("{t}.sub = bitcast float {t}.subf to i32"),
            format!("{t}.issub = icmp eq i32 {t}.e, 0"),
            format!("{t}.fin = select i1 {t}.issub, i32 {t}.sub, i32 {t}.normal"),
        ];
        match fmt {
            Fp8Format::E4M3 => lines.extend([
                format!("{t}.isnan = icmp eq i32 {t}.low7, 127"),
                format!("{t}.mag = select i1 {t}.isnan, i32 2143289344, i32 {t}.fin"),
            ]),
            Fp8Format::E5M2 => lines.extend([
                format!(
                    "{t}.isspecial = icmp eq i32 {t}.e, {}",
                    (1 << (7 - mbits)) - 1
                ),
                format!("{t}.haspay = icmp ne i32 {t}.m, 0"),
                format!("{t}.head = select i1 {t}.haspay, i32 2143289344, i32 2139095040"),
                format!("{t}.pay = shl i32 {t}.m, 21"),
                format!("{t}.special = or i32 {t}.head, {t}.pay"),
                format!("{t}.mag = select i1 {t}.isspecial, i32 {t}.special, i32 {t}.fin"),
            ]),
        }
        lines.extend([
            format!("{t}.bits = or i32 {t}.sign, {t}.mag"),
            format!("{t} = bitcast i32 {t}.bits to float"),
        ]);
        for l in lines {
            let _ = writeln!(out, "  {l}");
        }
        t
    }

    /// `value` (register type `reg`) rounded once onto the grid of `fmt`, as a `float`.
    pub(crate) fn round_to_f8(
        &self,
        out: &mut String,
        value: &str,
        reg: &MirType,
        fmt: Fp8Format,
    ) -> String {
        let code = self.f8_encode(out, value, reg, fmt);
        self.f8_decode(out, &code, fmt)
    }
}
