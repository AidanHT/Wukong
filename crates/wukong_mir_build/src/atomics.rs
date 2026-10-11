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
//! Parallelism: this is the one impure call the `@parallel` region proof admits, and an atomic
//! loop does get outlined. `match_parallel_region` is shared by the whole-function outliner
//! (`parallel_spec`) and by mid-function regions (`try_emit_parallel_region`). Its
//! `scan_region_expr` hands `atomic_add(arr[i], v)` to `scan_region_atomic`. That reads `v` and the
//! index as ordinary expressions. When `arr` is a captured fixed-size array, it records `arr` in
//! `RegionScan::atomic` instead of as a write, so the add needs no per-iteration disjointness
//! proof: integer adds commute, every iteration's add lands exactly once in any order, and the
//! result is the serial one bit for bit. A body-local array is private to its iteration.
//!
//! The region still declines (and the loop lowers serially) if the atomically updated array, or
//! any capture that may share its storage, has any other read or write in the loop: such a read
//! would see a schedule-dependent count. A shared-storage capture is one with the same MIR base,
//! or a parameter over the same element scalar with no `@noalias` promise between the two. A
//! field, pointee or multi-index place declines too, and so does a user function or local named
//! `atomic_add` (an ordinary callee). `tests/run/atomic_add_histogram.wk` exercises both paths.

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
