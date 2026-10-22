//! The 8-bit floats `f8e4m3` (OCP E4M3FN) and `f8e5m2` (OCP E5M2) on Cranelift.
//!
//! An f8 value lives in an `f32` register on its grid and is stored as its 1-byte code, exactly as
//! the 16-bit floats live in `f32` registers (`cl_type`). Its grid is reached at the observation
//! boundaries the interpreter rounds at — a constant, a load, a store, an `FpTrunc` to it and an
//! `FpExt` from it — by the bit-level codec of `wukong_runtime::fp8`, emitted inline here
//! instruction for instruction ([`FnTranslator::f8_encode`], [`FnTranslator::f8_decode`]) so the
//! two agree bit for bit with no runtime call. `fp8_type_codec_matches_the_runtime` in the tests
//! pins it on every code, every midpoint and a sweep of `f32` and `f64` patterns, natively and on
//! the interpreter.
//!
//! The encoder takes an `f32` or an `f64` source and rounds it once: an `f64` is encoded from its
//! own bits (going through `f32` would round twice).

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types, InstBuilder, MemFlags, Value};
use wukong_mir::MirType;
use wukong_runtime::fp8::Fp8Format;

use crate::FnTranslator;

/// The FP8 format of an f8 MIR type, or `None` for any other type.
pub(crate) fn f8_format(t: &MirType) -> Option<Fp8Format> {
    match t {
        MirType::F8E4M3 => Some(Fp8Format::E4M3),
        MirType::F8E5M2 => Some(Fp8Format::E5M2),
        _ => None,
    }
}

impl FnTranslator<'_> {
    /// The FP8 code of `x` (an `f32` or `f64` value) as an `i32` in `0..=255`:
    /// `wukong_runtime::fp8::encode_f32` / `encode_f64`, inline. A normal result rounds on the
    /// bit pattern (add `half - 1` plus the kept lsb), rebiases and clamps to the largest code; one
    /// below the least normal scales `|x|` by `2^k` (exact) and rounds to an integer, ties to
    /// even; a NaN takes the format's NaN code; the sign is copied.
    pub(crate) fn f8_encode(&mut self, x: Value, fmt: Fp8Format) -> Value {
        let wide = self.dfg_ty(x) == types::F64;
        let (ity, fty, c, mant, inf, abs, sign_shift) = if wide {
            (
                types::I64,
                types::F64,
                fmt.codec(52, 1023),
                52,
                0x7ff0_0000_0000_0000i64,
                0x7fff_ffff_ffff_ffffi64,
                56,
            )
        } else {
            (
                types::I32,
                types::F32,
                fmt.codec(23, 127),
                23,
                0x7f80_0000,
                0x7fff_ffff,
                24,
            )
        };
        let bits = self.builder.ins().bitcast(ity, MemFlags::new(), x);
        let hi = self.builder.ins().ushr_imm(bits, sign_shift);
        let sign = self.builder.ins().band_imm(hi, 0x80);
        let a = self.builder.ins().band_imm(bits, abs);
        let is_nan = self
            .builder
            .ins()
            .icmp_imm(IntCC::UnsignedGreaterThan, a, inf);
        // The normal range: round on the bits, rebias, clamp to the largest code.
        let lsb_sh = self.builder.ins().ushr_imm(a, i64::from(c.shift));
        let lsb = self.builder.ins().band_imm(lsb_sh, 1);
        let biased = self.builder.ins().iadd_imm(a, (1i64 << (c.shift - 1)) - 1);
        let rounded = self.builder.ins().iadd(biased, lsb);
        let code_sh = self.builder.ins().ushr_imm(rounded, i64::from(c.shift));
        let code_n = self.builder.ins().iadd_imm(code_sh, -(c.rebias as i64));
        let top = self.builder.ins().iconst(ity, c.top as i64);
        let over = self
            .builder
            .ins()
            .icmp(IntCC::SignedGreaterThan, code_n, top);
        let normal = self.builder.ins().select(over, top, code_n);
        // Below the least normal: |x|·2^k is exact; round it to the nearest (even) integer. The
        // saturating conversion never traps on the lanes this select discards.
        let ax = self.builder.ins().bitcast(fty, MemFlags::new(), a);
        let k = 2f64.powi(c.sub_scale_exp);
        let scale = if wide {
            self.builder.ins().f64const(k)
        } else {
            self.builder.ins().f32const(k as f32)
        };
        let scaled = self.builder.ins().fmul(ax, scale);
        let r = self.builder.ins().nearest(scaled);
        let sub = self.builder.ins().fcvt_to_sint_sat(ity, r);
        let is_sub = self.builder.ins().icmp_imm(
            IntCC::UnsignedLessThan,
            a,
            (c.min_normal_exp << mant) as i64,
        );
        let finite = self.builder.ins().select(is_sub, sub, normal);
        let nan = self.builder.ins().iconst(ity, i64::from(c.nan));
        let code = self.builder.ins().select(is_nan, nan, finite);
        let signed = self.builder.ins().bor(code, sign);
        if wide {
            self.builder.ins().ireduce(types::I32, signed)
        } else {
            signed
        }
    }

    /// The `f32` an FP8 code decodes to, exactly — `wukong_runtime::fp8::decode_f32`, inline.
    /// `code` is an `i32` holding the code zero-extended (`0..=255`). E4M3FN's NaN decodes to the
    /// quiet NaN of its sign; an E5M2 NaN keeps its payload under the quiet bit.
    pub(crate) fn f8_decode(&mut self, code: Value, fmt: Fp8Format) -> Value {
        let t = types::I32;
        let mbits = fmt.mbits();
        let c = fmt.codec(23, 127);
        let s = self.builder.ins().band_imm(code, 0x80);
        let sign = self.builder.ins().ishl_imm(s, 24);
        let low7 = self.builder.ins().band_imm(code, 0x7f);
        let e = self.builder.ins().ushr_imm(low7, i64::from(mbits));
        let m = self.builder.ins().band_imm(code, (1i64 << mbits) - 1);
        let e32 = self.builder.ins().iadd_imm(e, 127 - i64::from(fmt.bias()));
        let e32s = self.builder.ins().ishl_imm(e32, 23);
        let ms = self.builder.ins().ishl_imm(m, i64::from(23 - mbits));
        let normal = self.builder.ins().bor(e32s, ms);
        // A subnormal: m · 2^-k, formed in f32 (both steps exact) and read back as bits.
        let mf = self.builder.ins().fcvt_from_sint(types::F32, m);
        let scale = self.builder.ins().f32const(2f32.powi(-c.sub_scale_exp));
        let subf = self.builder.ins().fmul(mf, scale);
        let sub = self.builder.ins().bitcast(t, MemFlags::new(), subf);
        let is_sub = self.builder.ins().icmp_imm(IntCC::Equal, e, 0);
        let finite = self.builder.ins().select(is_sub, sub, normal);
        let mag = match fmt {
            Fp8Format::E4M3 => {
                let is_nan = self.builder.ins().icmp_imm(IntCC::Equal, low7, 0x7f);
                let qnan = self.builder.ins().iconst(t, 0x7fc0_0000);
                self.builder.ins().select(is_nan, qnan, finite)
            }
            Fp8Format::E5M2 => {
                let emax = (1i64 << (7 - mbits)) - 1;
                let is_special = self.builder.ins().icmp_imm(IntCC::Equal, e, emax);
                let has_payload = self.builder.ins().icmp_imm(IntCC::NotEqual, m, 0);
                let qnan = self.builder.ins().iconst(t, 0x7fc0_0000);
                let inf = self.builder.ins().iconst(t, 0x7f80_0000);
                let head = self.builder.ins().select(has_payload, qnan, inf);
                let payload = self.builder.ins().ishl_imm(m, 21);
                let special = self.builder.ins().bor(head, payload);
                self.builder.ins().select(is_special, special, finite)
            }
        };
        let bits = self.builder.ins().bor(sign, mag);
        self.builder
            .ins()
            .bitcast(types::F32, MemFlags::new(), bits)
    }

    /// `x` (an `f32` or `f64` value) rounded once onto the grid of `fmt`, as the `f32` it lives in.
    pub(crate) fn round_to_f8(&mut self, x: Value, fmt: Fp8Format) -> Value {
        let code = self.f8_encode(x, fmt);
        self.f8_decode(code, fmt)
    }
}
