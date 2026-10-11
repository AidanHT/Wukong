//! The `atomic_add(place, v)` builtin.
//!
//! `place = place + v`, wrapping at the place's width, as one indivisible read-modify-write: a
//! single `Op::AtomicAdd` (no result). Sema (`type_atomic_add`) has already checked that `place`
//! is a place in memory other than a bare name (`b[i]`, `t[i, j]`, `s.f`, `t.0`, `*p`), that its
//! type is an integer scalar, and that `v` has that type (an unsuffixed literal adapted to it), so
//! the coercion below is the identity on every accepted program.
//!
//! Evaluation order is the compound assignment's (`place += v` in `lower_stmt`): the value first,
//! then the place's address — its GEP and, for a `[]T` slice, the bounds check — exactly as
//! `b[i] += v` would compute them.
//!
//! Parallelism: the whole-function `@parallel` outliner needs nothing (every iteration's add lands
//! exactly once, in any order, so the result is the serial one bit-for-bit). The mid-function
//! region scanner (`scan_region_expr`) declines every impure call, this one included, so a
//! `@parallel for` region inside a larger function that contains an `atomic_add` stays serial.

use wukong_ast::Expr;

use crate::FnLowerer;

impl FnLowerer<'_> {
    /// Lower `atomic_add(place, v)`; `None` for any other callee or arity, so `lower_call` falls
    /// through.
    pub(super) fn lower_atomic_add(&mut self, name: &str, args: &[Expr]) -> Option<()> {
        if !wukong_sema::is_atomic_add(name) || args.len() != 2 {
            return None;
        }
        let (place, value) = (&args[0], &args[1]);
        let rhs0 = self.lower_expr(value);
        let rhs_ty = self.expr_mir(value);
        let (ptr, elem) = self.lower_place(place);
        let rhs = self.coerce_to(rhs0, &rhs_ty, &elem, self.signed(value));
        self.builder.atomic_add(ptr, rhs);
        Some(())
    }
}
