//! Lowering of the printing builtins by the printed value's **source** type.
//!
//! `print`/`println` lower to a void call the backends route to a per-type runtime printer. Most of
//! the routing is visible in MIR alone: an `f64` argument and an `f32` (or `f16` / `bf16`, which
//! live in `f32` registers) one differ in MIR type, and every backend picks `wukong_rt_print_f64`
//! or `wukong_rt_print_f32` from it. What MIR cannot see is a `bool` (an `i1`, the same register as
//! a comparison result) or a `char` (an `i32`, the same register as an integer): those are routed
//! here, by their sema type, to the `print_bool` / `print_char` symbols, zero-extended to `i64` —
//! so `print(true)` prints `true` and `print('A')` prints `A`, on every backend, instead of `1` and
//! `65`.

use wukong_ast::Expr;
use wukong_mir::{MirType, Op, ValueId};
use wukong_span::{Interner, Symbol};
use wukong_types::{Scalar, Ty};

use crate::FnLowerer;

/// The symbols this module lowers to, interned once at the top of `lower_program` (`FnLowerer`
/// holds only a shared `&Interner`) and carried on `GemmSyms`.
#[derive(Clone, Copy)]
pub(crate) struct TextSyms {
    print_bool: Symbol,
    println_bool: Symbol,
    print_char: Symbol,
    println_char: Symbol,
}

impl TextSyms {
    pub(crate) fn new(interner: &mut Interner) -> TextSyms {
        TextSyms {
            print_bool: interner.intern("print_bool"),
            println_bool: interner.intern("println_bool"),
            print_char: interner.intern("print_char"),
            println_char: interner.intern("println_char"),
        }
    }
}

impl FnLowerer<'_> {
    /// `print(x)` / `println(x)` of a `bool` or a `char`: its own printer, the value zero-extended
    /// to `i64`. `None` for any other argument, which keeps the generic print path.
    pub(crate) fn lower_print_scalar(&mut self, newline: bool, arg: &Expr) -> Option<ValueId> {
        let syms = self.gemm.text;
        let (plain, line) = match self.expr_ty(arg) {
            Ty::Scalar(Scalar::Bool) => (syms.print_bool, syms.println_bool),
            Ty::Scalar(Scalar::Char) => (syms.print_char, syms.println_char),
            _ => return None,
        };
        let from = self.expr_mir(arg);
        let v0 = self.lower_expr(arg);
        let v = self.coerce_to(v0, &from, &MirType::I64, false);
        let func = if newline { line } else { plain };
        self.builder.build_void(Op::Call {
            func,
            args: vec![v],
        });
        Some(self.const_zero(MirType::I32))
    }
}
