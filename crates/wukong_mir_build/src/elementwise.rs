//! Spelling coverage for the **elementwise** kernel recognizers (`wukong_vmath_f32`,
//! `wukong_velem_f32`, `wukong_vmath2_f32`, `wukong_vhorner_f32`, `wukong_dequant_f32`).
//!
//! A respelling used to cost the 256-bit kernel while computing exactly what the recognized form
//! computes:
//!
//! * **A hoisted value binding** — `let v = x[i]; o[i] = gelu(v);`. The probes are retried over the
//!   view [`crate::canon::inline_value_lets`] builds, and only when the original body declined, so a
//!   body that already dispatched keeps its kernel and a body the view cannot help is lowered from
//!   the original AST exactly as before.
//!
//! The textbook GELU expansion — `0.5*v*(1.0 + tanh(0.7978845608*(v + 0.044715*(v*v*v))))` — is
//! deliberately NOT folded into `VMATH_GELU` (a fold once did). The fold's premise was that the
//! source's `tanh` is the kernel's internal `tanh_exp1` (`1 - 2/(exp(2z)+1)`), so the composed
//! expression and the kernel agree bit-for-bit. The language's `tanh` is now the runtime's piecewise
//! kernel on every spelling (`5f423d1e`), while `gelu` keeps `tanh_exp1` inside; the expansion is
//! therefore a different function in the last bits on ordinary inputs (6 of 16 lanes of
//! `tests/run/spelling_value_bindings.wk`), and substituting the kernel would change the answer.
//! It lowers as written, through the generic vectorizer's inline `tanh`.

use wukong_ast::{Block, Expr};
use wukong_mir::{MirType, ValueId};
use wukong_span::Symbol;

use crate::{canon, FnLowerer};

impl FnLowerer<'_> {
    /// Retry the five elementwise kernel probes of `try_vectorize_for` over the body with its
    /// leading value bindings inlined. Called only after every probe declined the original body.
    pub(super) fn try_elementwise_kernels_on_value_view(
        &mut self,
        j: Symbol,
        start: &Expr,
        end: &Expr,
        body: &Block,
    ) -> bool {
        let Some(view) = canon::inline_value_lets(body, self.sema, self.interner) else {
            return false;
        };
        self.try_vmath_for(j, start, end, &view)
            || self.try_vmath2_for(j, start, end, &view)
            || self.try_velem_for(j, start, end, &view)
            || self.try_vhorner_for(j, start, end, &view)
            || self.try_dequant_for(j, start, end, &view)
    }

    /// The `@parallel`-chunk twin of [`Self::try_elementwise_kernels_on_value_view`]: the bounds are
    /// already lowered, as `try_vectorize_ranged` has them. Only the activation and streaming affine
    /// map chunk kernels are retried (either emitter may still decline, e.g. on a hoisted
    /// coefficient that reads the output); anything else falls to the generic vectorizer over the
    /// original body, exactly as before.
    pub(super) fn try_ranged_kernels_on_value_view(
        &mut self,
        j: Symbol,
        start_val: ValueId,
        end_val: ValueId,
        ity: &MirType,
        body: &Block,
    ) -> bool {
        let Some(view) = canon::inline_value_lets(body, self.sema, self.interner) else {
            return false;
        };
        if let Some(calls) = self.match_vmath_body(j, &view, None) {
            let s = self.coerce_to(start_val, ity, &MirType::I64, true);
            let e = self.coerce_to(end_val, ity, &MirType::I64, true);
            // A decline (an alias the run-time guard cannot range) emitted nothing but pure setup.
            if self.emit_vmath_calls(s, e, calls) {
                return true;
            }
        }
        if let Some(plan) = self.match_velem_body(j, &view) {
            let s = self.coerce_to(start_val, ity, &MirType::I64, true);
            let e = self.coerce_to(end_val, ity, &MirType::I64, true);
            if self.emit_velem_call(s, e, &plan) {
                return true;
            }
        }
        false
    }

    /// The batched (`[rows, cols]`) activation probe over the inner body's value view.
    pub(super) fn batched_vmath_on_value_view(
        &self,
        j: Symbol,
        inner: &Block,
        batch: (Symbol, &Expr),
    ) -> Option<Vec<(Symbol, Symbol, u32, MirType)>> {
        let view = canon::inline_value_lets(inner, self.sema, self.interner)?;
        self.match_vmath_body(j, &view, Some(batch))
    }
}
