//! Elementwise loops over an **unsigned 64-bit** counter: `for i in 0..n` with `n: usize`, a
//! `usize` const, or the hidden dimension parameter of a symbolic tensor (`fn f<N>(x:
//! Tensor[f32, N])`), which is the shape-safe spelling the language leads with.
//!
//! Every runtime kernel takes its extent as a signed `i64`, and the generic vectorizer's strips and
//! tails compare the counter with *signed* predicates, so both are only the loop the user wrote
//! when the trip count lies in the signed domain. `kernel_range_supported` proves that statically
//! for literals and literal-valued consts, and `try_vectorize_for` refuses every unsigned counter
//! outright, so `for i in 0..N { out[i] = x[i] * s }` over a `Tensor[f32, N]` dispatched no kernel
//! and was not even vectorized.
//!
//! This module closes that with the same shape `emit_nonempty_reduction` uses for a possibly-empty
//! reduction: decide at **run time**.
//!
//! ```text
//!   e = <end> as i64              // the very bits the scalar loop compares against
//!   if e >= 0 (signed) {          // every counter value in [0, e) is then < 2^63, so
//!       <kernel or vector loop>   //   `slt i, e` and `ult i, e` agree on every iteration
//!   } else {
//!       <the loop exactly as before>   // scalar, unsigned compare, no recognizer
//!   }
//! ```
//!
//! When the endpoint is statically known to fit (`end_fits_signed_i64`), the test is omitted.
//!
//! What is NOT routed here, deliberately: a reduction (`s = s + x[i]`). The vectorized reduction
//! reassociates the sum; a `usize` loop that used to fold in source order would change its result.
//! A `@parallel` function is also left alone, so the mid-function region outliner keeps first claim
//! on its loops. And the endpoint must be side-effect free (it is evaluated once for the test and
//! again by the fallback): a literal, a name, arithmetic over those, or a slice's `.len()`.

use wukong_ast::{self as ast, Block, Expr, ExprKind, ForIter, Pattern};
use wukong_mir::{CmpOp, MirType, Op};
use wukong_span::Symbol;
use wukong_types::Ty;

use crate::{as_int_lit, canon, is_pure_loop_bound, FnLowerer};

impl FnLowerer<'_> {
    /// Lower `for <pat> in start..end { body }` through the signed-domain fast path when the counter
    /// is unsigned 64-bit and something — an elementwise kernel or the generic vectorizer — wants
    /// the loop. Returns false, having emitted nothing, when the loop is not such a loop.
    pub(super) fn try_unsigned_range_fast_path(
        &mut self,
        label: Option<Symbol>,
        pat: &Pattern,
        iter: &ForIter,
        start: &Expr,
        end: &Expr,
        body: &Block,
    ) -> bool {
        if self.parallel_fn {
            return false;
        }
        let ast::PatKind::Ident(j) = pat.kind else {
            return false;
        };
        let (ity, signed) = self.range_counter_type(start, end);
        if signed
            || ity != MirType::I64
            || !self.expr_mir(start).is_int()
            || !self.expr_mir(end).is_int()
            || as_int_lit(start, self.interner) != Some(0)
            || !self.is_reevaluable_bound(end)
        {
            return false;
        }
        if !self.signed_domain_loop_wanted(j, body) {
            return false;
        }
        if self.end_fits_signed_i64(end) {
            // Statically in the signed domain: no test, and no fallback copy of the body. The
            // kernel probes already saw this loop in `try_vectorize_for` (`kernel_range_supported`
            // accepts a provably-fitting unsigned endpoint) and declined it, so `kernel_bounds` is
            // off: what is new here is only the generic vectorizer, which that function refuses
            // for every unsigned counter.
            let zero = self
                .builder
                .build(MirType::I64, Op::ConstInt(0, MirType::I64));
            let e = self.lower_range_bound(end, &ity, signed);
            return self.try_vectorize_ranged(j, zero, e, &MirType::I64, body, false);
        }
        // A run-time test needs a second, recognizer-free lowering of the loop as the fallback; that
        // lowering re-enters `lower_for` with recognition off, which is also what keeps this arm
        // from recursing. With recognition already off there is nothing to guard.
        if !self.gemm.recognize {
            return false;
        }
        let zero = self
            .builder
            .build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let e = self.lower_range_bound(end, &ity, signed);
        let fits = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Sge, e, zero));
        let fast = self.builder.new_block();
        let slow = self.builder.new_block();
        let join = self.builder.new_block();
        self.builder.cond_br(fits, fast, vec![], slow, vec![]);

        self.builder.switch_to(fast);
        self.terminated = false;
        // The two-input map (`pow`/`atan2`/`hypot`) has no `@parallel`-chunk arm in
        // `try_vectorize_ranged`, so it is probed here; every other chunk kernel is probed there.
        let done = match self.match_vmath2_body(j, body) {
            Some(calls) => {
                self.emit_vmath2_calls(zero, e, calls);
                true
            }
            None => self.try_vectorize_ranged(j, zero, e, &MirType::I64, body, true),
        };
        self.builder.br(if done { join } else { slow }, vec![]);

        self.builder.switch_to(slow);
        self.terminated = false;
        // The original `lower_for` already decided this loop is not strip-mined, so the fallback
        // must not pop a function-item `@tile` width either.
        let tiles = std::mem::take(&mut self.tile_queue);
        let recognize = self.gemm.recognize;
        self.gemm.recognize = false;
        self.lower_for(label, pat, iter, body, &[]);
        self.gemm.recognize = recognize;
        self.tile_queue = tiles;
        if !self.terminated {
            self.builder.br(join, vec![]);
        }
        self.builder.switch_to(join);
        self.terminated = false;
        true
    }

    /// Would the signed-domain lowering do anything with this body? Pure: the two-input map, one of
    /// the elementwise chunk kernels `try_vectorize_ranged` probes (directly or through the
    /// value-binding view), or the generic vectorizer. A reduction is deliberately not in the list
    /// (see the module docs).
    fn signed_domain_loop_wanted(&self, j: Symbol, body: &Block) -> bool {
        if self.reduction_of(body, j).is_some() {
            return false;
        }
        if self.vectorizable(body, j).is_some() {
            return true;
        }
        if !self.gemm.recognize {
            return false;
        }
        if self.match_vmath_body(j, body, None).is_some()
            || self.match_vmath2_body(j, body).is_some()
            || self.match_velem_body(j, body).is_some()
            || self.match_dequant_body(j, body).is_some()
        {
            return true;
        }
        canon::inline_value_lets(body, self.sema, self.interner).is_some_and(|view| {
            self.match_vmath_body(j, &view, None).is_some()
                || self.match_velem_body(j, &view).is_some()
        })
    }

    /// May `end` be evaluated twice — once for the range test, once by the fallback loop — with the
    /// same value and no effect? [`is_pure_loop_bound`]'s whitelist, plus a slice's `.len()`, the
    /// one call that is a plain field read of the fat pointer.
    fn is_reevaluable_bound(&self, end: &Expr) -> bool {
        if is_pure_loop_bound(end) {
            return true;
        }
        let ExprKind::Call { callee, args, .. } = &end.kind else {
            return false;
        };
        let ExprKind::Field { base, name } = &callee.kind else {
            return false;
        };
        args.is_empty()
            && self.interner.resolve(name.sym) == "len"
            && matches!(&base.kind, ExprKind::Path(p) if p.is_single())
            && matches!(self.expr_ty(base), Ty::Slice(_))
    }
}
