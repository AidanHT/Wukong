//! Lowering of the `i128` operations Cranelift's x64 backend has no instruction for.
//!
//! An MIR `i128` is a CLIF `I128` held in a register pair, and x64 lowers addition, subtraction,
//! multiplication, the bitwise operations, shifts (amount modulo 128), negation, compares, `select`,
//! loads, stores and the integer extensions and truncations to and from it directly. Three things
//! it cannot do:
//!
//! * materialize a constant wider than 64 bits: `iconst.i128` takes a 64-bit immediate, so a
//!   constant is built from its two halves with `iconcat`;
//! * divide: there is no `sdiv`/`udiv`/`srem`/`urem` on `I128`, so each is a call of
//!   `wukong_rt_i128_divrem`, which also gives a zero divisor the language's result `0` and wraps
//!   `i128::MIN / -1` — the guards the narrower widths build in CLIF;
//! * convert to or from a float: `wukong_rt_i128_to_f64` / `_to_f32` round once, straight to the
//!   target precision, and `wukong_rt_f64_to_i128` saturates (an `f32` operand widens to `f64`
//!   exactly first).
//!
//! A runtime entry takes an `i128` operand as two `i64` halves and returns a 128-bit result
//! through a 16-byte stack slot (`wukong_runtime::int128`), so no platform `i128` calling
//! convention is involved. The interpreter computes all of it on its own `i128` values, and the
//! runtime entries are unit-tested against those definitions.

use cranelift_codegen::ir::{
    types, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Type, Value,
};
use wukong_mir::BinOp;

use crate::text_rt::{RT_F64_TO_I128, RT_I128_DIVREM, RT_I128_TO_F32, RT_I128_TO_F64};
use crate::FnTranslator;

impl FnTranslator<'_> {
    /// The 128-bit constant `v`, from its two 64-bit halves.
    pub(crate) fn i128_const(&mut self, v: i128) -> Value {
        let lo = self.builder.ins().iconst(types::I64, v as i64);
        let hi = self.builder.ins().iconst(types::I64, (v >> 64) as i64);
        self.builder.ins().iconcat(lo, hi)
    }

    /// `c ? a : b` on `I128` values, one 64-bit half at a time. x64 lowers a 128-bit `select`
    /// itself, but Cranelift's mid-end turns `select(icmp(x, y), x, y)` into `smin`/`smax`/`umin`/
    /// `umax`, for which x64 has no 128-bit lowering (a `min(a, b)` would then fail to compile at
    /// `-O2`). The halves are not the compared values, so the rewrite never matches.
    pub(crate) fn i128_select(&mut self, c: Value, a: Value, b: Value) -> Value {
        let (a_lo, a_hi) = self.builder.ins().isplit(a);
        let (b_lo, b_hi) = self.builder.ins().isplit(b);
        let lo = self.builder.ins().select(c, a_lo, b_lo);
        let hi = self.builder.ins().select(c, a_hi, b_hi);
        self.builder.ins().iconcat(lo, hi)
    }

    /// The address of a fresh 16-byte, 16-aligned stack slot a runtime entry writes a 128-bit
    /// result into.
    fn i128_result_slot(&mut self) -> Value {
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            16,
            4,
        ));
        self.builder.ins().stack_addr(self.ptr_ty, slot, 0)
    }

    /// Read the 128-bit result a runtime entry stored at `addr`.
    fn i128_load_result(&mut self, addr: Value) -> Value {
        self.builder
            .ins()
            .load(types::I128, MemFlags::trusted(), addr, 0)
    }

    /// `a op b` for one of the four integer division operations on `I128` operands.
    pub(crate) fn i128_divrem(&mut self, op: BinOp, a: Value, b: Value) -> Value {
        use wukong_runtime::int128::{I128_SDIV, I128_SREM, I128_UDIV, I128_UREM};
        let code = match op {
            BinOp::SDiv => I128_SDIV,
            BinOp::UDiv => I128_UDIV,
            BinOp::SRem => I128_SREM,
            BinOp::URem => I128_UREM,
            _ => unreachable!("i128_divrem of a non-division op"),
        };
        let (a_lo, a_hi) = self.builder.ins().isplit(a);
        let (b_lo, b_hi) = self.builder.ins().isplit(b);
        let op = self.builder.ins().iconst(types::I64, code);
        let out = self.i128_result_slot();
        let fref = self.runtime_ref(RT_I128_DIVREM);
        self.builder
            .ins()
            .call(fref, &[a_lo, a_hi, b_lo, b_hi, op, out]);
        self.i128_load_result(out)
    }

    /// The `I128` value `x` (read as signed when `signed`) as a float of type `to` — `F32` (also
    /// the register type of `f16`/`bf16`) or `F64` — rounded once.
    pub(crate) fn i128_to_float(&mut self, x: Value, to: Type, signed: bool) -> Value {
        let (lo, hi) = self.builder.ins().isplit(x);
        let signed = self.builder.ins().iconst(types::I64, i64::from(signed));
        let entry = if to == types::F64 {
            RT_I128_TO_F64
        } else {
            RT_I128_TO_F32
        };
        let fref = self.runtime_ref(entry);
        let call = self.builder.ins().call(fref, &[lo, hi, signed]);
        self.builder.inst_results(call)[0]
    }

    /// The float `x` converted to `I128`, saturating (to the `i128` range when `signed`, else the
    /// `u128` range; NaN is `0`).
    pub(crate) fn float_to_i128(&mut self, x: Value, signed: bool) -> Value {
        let x = if self.dfg_ty(x) == types::F64 {
            x
        } else {
            self.builder.ins().fpromote(types::F64, x)
        };
        let signed = self.builder.ins().iconst(types::I64, i64::from(signed));
        let out = self.i128_result_slot();
        let fref = self.runtime_ref(RT_F64_TO_I128);
        self.builder.ins().call(fref, &[x, signed, out]);
        self.i128_load_result(out)
    }
}
