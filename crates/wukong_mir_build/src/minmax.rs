//! The `min(a, b)` / `max(a, b)` builtins.
//!
//! Integer `min` was not lowered at all — a call to it was the catalogued `C0001` "call is not yet
//! supported by codegen" — and it is the one function every ragged loop tiling is written with:
//! `for i in i0..min(i0 + T, M)` is how a blocked kernel covers an extent the tile does not divide
//! (`examples/matmul.wk` is written that way). Without it the only way to block a loop nest was to
//! require the tiles to divide every extent exactly.
//!
//! Semantics, chosen to be the obvious ones and to agree with the float pair that already exists:
//!
//! * on integers, `min(a, b)` is `a < b ? a : b` and `max(a, b)` is `a > b ? a : b`, compared in
//!   the result type's signedness (an unsigned `usize` compares unsigned);
//! * on floats they are exactly `fmin` / `fmax` (`a < b ? a : b`, `a > b ? a : b` — on a tie or a
//!   NaN the second operand), so `min(x, y)` and `fmin(x, y)` are the same expression;
//! * the result type is the operands' common type exactly as for `a + b` — the wider integer, or
//!   the float when either operand is one (`wukong_sema`'s `intrinsic_ret_ty`); each operand is
//!   converted to it with its own signedness before the compare, from the width its value actually
//!   has (a loop counter can be wider than its declared type, as in `lower_binary`). Any other
//!   arity, or a non-numeric operand, is not lowered here (`lower_call` then reports `C0001` at the
//!   call, as before).
//!
//! It is plain MIR (`cmp` + `select`), so both backends agree bit-for-bit and nothing downstream
//! needs to know it exists. A user `fn min` / `fn max` shadows the builtin (`lower_call` resolves
//! user functions first).

use wukong_ast::Expr;
use wukong_mir::{CmpOp, MirType, Op, ValueId};
use wukong_span::Symbol;
use wukong_types::Ty;

use crate::FnLowerer;

impl FnLowerer<'_> {
    /// Lower `min(a, b)` / `max(a, b)`; `None` for any other callee or a malformed call.
    pub(super) fn lower_min_max(
        &mut self,
        name: Symbol,
        args: &[Expr],
        e: &Expr,
    ) -> Option<ValueId> {
        let is_min = match self.interner.resolve(name) {
            "min" => true,
            "max" => false,
            _ => return None,
        };
        if args.len() != 2 {
            return None;
        }
        let Ty::Scalar(sc) = self.expr_ty(e) else {
            return None;
        };
        let pred = if sc.is_float() {
            if is_min {
                CmpOp::Folt
            } else {
                CmpOp::Fogt
            }
        } else if sc.is_int() {
            match (is_min, sc.is_signed()) {
                (true, true) => CmpOp::Slt,
                (true, false) => CmpOp::Ult,
                (false, true) => CmpOp::Sgt,
                (false, false) => CmpOp::Ugt,
            }
        } else {
            return None;
        };
        let rty = self.expr_mir(e);
        // Operands in source order, each converted to the result type with its own signedness.
        let a = self.lower_min_max_operand(&args[0], &rty);
        let b = self.lower_min_max_operand(&args[1], &rty);
        let c = self.builder.build(MirType::I1, Op::Cmp(pred, a, b));
        Some(self.builder.build(rty, Op::Select(c, a, b)))
    }

    /// One operand, converted to `rty` from the width its lowered value has.
    fn lower_min_max_operand(&mut self, arg: &Expr, rty: &MirType) -> ValueId {
        let v = self.lower_expr(arg);
        let from = self.operand_mir(v, arg);
        let signed = self.signed(arg);
        self.coerce_to(v, &from, rty, signed)
    }
}
