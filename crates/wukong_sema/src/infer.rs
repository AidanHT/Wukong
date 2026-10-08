//! Literal inference: `let mut s = 0.0;` is a `{float}` until a use says which float.
//!
//! An unsuffixed literal has always adapted to the type its *position* pins — a `let` annotation,
//! a parameter, a `return`, a plain assignment's place. A `let` with no annotation pinned nothing,
//! so the binding froze at the literal's default (`i32`, or `f32`), and every later use had to live
//! with it: `s += x[i]` over `f64` data accumulated in `f32` (the compound path coerced the operand
//! down to the place), `s = s + x[i]` was an error, and `return s` from `-> f64` was an error. Rust
//! gives such a literal an inference variable that the first constraining use resolves; this is
//! that rule, for the one place it bites: a binding initialized only from unsuffixed literals.
//!
//! **Which bindings.** A `let` with a single-name pattern, no annotation, and an initializer made
//! only of unsuffixed numeric literals (`0`, `-1.5`, `0 - 16`) — [`crate::LocalFact::InferLit`].
//!
//! **Which uses decide it.** The *first* constraining use met in source order, and only it, decides
//! — as in Rust, where the first use that unifies the variable with a concrete type fixes it. The
//! kind never changes (an integer literal stays an integer). A use pairs the binding with a concrete
//! scalar type `T`:
//!
//! * an assignment `s = v` / `v = s`, a `return s`, a call argument `f(s)`, or a `let x: T = s` —
//!   the binding becomes `T` at any width (these positions demand exact agreement);
//! * a compound assignment `s op= v` / `v op= s`, an arithmetic or comparison operator `s op v` /
//!   `v op s` — the binding becomes `T` when `T` is at least as wide as its default, and otherwise
//!   keeps the default: a narrower `T` promotes into it losslessly anyway, so `let k = 2; k + x_u8`
//!   stays the `i32` sum it always was.
//!
//! A use that decides the default (the same type, a narrower operand, the other kind) still
//! *pins* the binding, so a later use cannot change it behind the first one's back: in
//! `let k = 2; f_i32(k); x_i64 * k` the call fixes `k: i32`, and the product promotes it. A use
//! that pairs two such bindings (`b = a`) decides neither — they are both still open.
//!
//! **How.** Sema checks a body in one pass, and a use met after a binding cannot retype the uses
//! before it. So a pass that learns some binding's type ends there: the body is checked again from
//! the top with every decided binding fixed as if annotated (its literal re-stamped through
//! `retype_adapted_literal` and range-checked), and the superseded pass's diagnostics are dropped —
//! until a pass learns nothing that changes a type. A pass that only pins needs no re-check: it
//! typed every binding as the next pass would. Each re-check fixes at least one more binding, so
//! the passes are bounded by the number of such `let`s, and by [`MAX_PASSES`] regardless.

use wukong_ast::{Expr, ExprKind, PatKind, Pattern, TypeExpr};
use wukong_types::{Scalar, Ty};

use crate::{LocalFact, Sema};

/// Passes over one function body at most; each pass after the first fixes at least one binding.
pub(crate) const MAX_PASSES: u32 = 8;

impl Sema<'_> {
    /// The type of a `let` with no annotation whose initializer `init` typed as `init_ty`: what an
    /// earlier pass decided for this binding (re-stamping its literal), else `init_ty`.
    pub(crate) fn literal_let_type(
        &mut self,
        pat: &Pattern,
        init: Option<&Expr>,
        init_ty: &Ty,
    ) -> Ty {
        if let (Some(t), Some(e)) = (self.infer_overrides.get(&pat.id).cloned(), init) {
            if self.literal_adapts(&t, e) {
                self.retype_adapted_literal(e, &t);
                self.range_check_int_literal(e, &t);
                return t;
            }
        }
        init_ty.clone()
    }

    /// Mark a just-bound `let` as literal-inferred ([`LocalFact::InferLit`]) when it qualifies: a
    /// single name, no annotation, an initializer of unsuffixed numeric literals only, an integer
    /// or float type, and not already decided by an earlier pass.
    pub(crate) fn note_literal_binding(
        &mut self,
        pat: &Pattern,
        ann: Option<&TypeExpr>,
        init: Option<&Expr>,
        bound: &Ty,
    ) {
        let (PatKind::Ident(name), None, Some(e)) = (&pat.kind, ann, init) else {
            return;
        };
        if self.infer_overrides.contains_key(&pat.id) {
            return;
        }
        let Ty::Scalar(s) = bound else {
            return;
        };
        if !(s.is_float() || is_plain_int(*s)) {
            return;
        }
        if !self.is_literal_operand(e) || !self.literal_adapts(bound, e) {
            return;
        }
        if let Some(facts) = self.local_facts.last_mut() {
            facts.insert(*name, LocalFact::InferLit(pat.id));
        }
    }

    /// A constraining use of `e` (see the module docs): when `e` names a literal-inferred binding
    /// and its counterpart — the expression `other_expr`, if there is one, of type `other` — is a
    /// concrete scalar, decide the binding's type unless an earlier use already did. `widen_only`
    /// marks an operator use, which changes the type only to a type at least as wide.
    pub(crate) fn constrain_literal_binding(
        &mut self,
        e: &Expr,
        other_expr: Option<&Expr>,
        other: &Ty,
        widen_only: bool,
    ) {
        let Some((id, sym)) = self.literal_binding(e) else {
            return;
        };
        // Two open bindings meeting decide nothing (both are still `{integer}`/`{float}`).
        if other_expr.is_some_and(|o| self.literal_binding(o).is_some()) {
            return;
        }
        if self.infer_found.contains_key(&id) || self.infer_pinned.contains_key(&id) {
            return;
        }
        let (Some(Ty::Scalar(cur)), Ty::Scalar(want)) = (self.lookup_local(sym), other) else {
            return;
        };
        let want = *want;
        let same_kind = if cur.is_float() {
            want.is_float()
        } else {
            is_plain_int(want)
        };
        if want != cur && same_kind && !(widen_only && want.size() < cur.size()) {
            self.infer_found.insert(id, Ty::Scalar(want));
        } else {
            self.infer_pinned.insert(id, Ty::Scalar(cur));
        }
    }

    /// The `let` id and name of the literal-inferred binding the single-name path `e` resolves to.
    fn literal_binding(&self, e: &Expr) -> Option<(wukong_ast::NodeId, wukong_span::Symbol)> {
        let ExprKind::Path(p) = &e.kind else {
            return None;
        };
        if !p.is_single() {
            return None;
        }
        let sym = p.first().sym;
        match self.local_fact(sym)? {
            LocalFact::InferLit(id) => Some((id, sym)),
            LocalFact::NonNegCounter => None,
        }
    }
}

/// An integer type a literal can become: not `bool`, not `char`.
fn is_plain_int(s: Scalar) -> bool {
    s.is_int() && s != Scalar::Char
}

#[cfg(test)]
mod tests {
    use wukong_ast::{Item, ItemKind, Module, StmtKind};
    use wukong_span::{Interner, SourceId};
    use wukong_types::{Scalar, Ty};

    /// Check `src` (which must check clean) and return the type recorded for the initializer of
    /// the first `let` of the first function body.
    fn first_let_init_ty(src: &str) -> Ty {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (res, diags) = crate::check(&module, &interner);
        assert!(
            diags.iter().all(|d| !d.is_error()),
            "sema errors for {src:?}: {diags:?}"
        );
        let init = first_let_init(&module).expect("a `let` with an initializer");
        res.types.get(&init).cloned().expect("a recorded type")
    }

    fn first_let_init(m: &Module) -> Option<wukong_ast::NodeId> {
        m.items.iter().find_map(|it: &Item| match &it.kind {
            ItemKind::Fn(f) => f.body.as_ref()?.stmts.iter().find_map(|s| match &s.kind {
                StmtKind::Let { init: Some(e), .. } => Some(e.id),
                _ => None,
            }),
            _ => None,
        })
    }

    /// The diagnostic codes `src` checks to.
    fn codes(src: &str) -> Vec<&'static str> {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        crate::check(&module, &interner)
            .1
            .into_iter()
            .filter_map(|d| d.code)
            .collect()
    }

    #[test]
    fn a_literal_binding_takes_the_type_its_first_use_needs() {
        // Compound accumulation over f64 data: the accumulator becomes f64 (it used to stay f32 and
        // narrow every partial sum).
        assert_eq!(
            first_let_init_ty(
                "fn f(x: [f64; 4]) -> f64 { let mut s = 0.0; for i in 0..4 { s += x[i]; } \
                 return s; }"
            ),
            Ty::Scalar(Scalar::F64)
        );
        // A plain assignment and a return decide it at any width.
        assert_eq!(
            first_let_init_ty("fn f(h: bf16) -> bf16 { let mut m = 0.0; m = h; return m; }"),
            Ty::Scalar(Scalar::Bf16)
        );
        assert_eq!(
            first_let_init_ty("fn f() -> i64 { let k = 7; return k; }"),
            Ty::Scalar(Scalar::I64)
        );
        // A call argument decides it (`g` has no `let`, so the first `let` is `f`'s).
        assert_eq!(
            first_let_init_ty(
                "fn g(n: i64) -> i64 { return n; } fn f() -> i64 { let k = 7; return g(k); }"
            ),
            Ty::Scalar(Scalar::I64)
        );
        // An operator with a wider operand of the same kind.
        assert_eq!(
            first_let_init_ty("fn f(x: u32) -> u32 { let k = 3; return x * k; }"),
            Ty::Scalar(Scalar::U32)
        );
        // Two open bindings meeting decide nothing; a later concrete use decides (second pass).
        assert_eq!(
            first_let_init_ty(
                "fn f(x: f64) -> f64 { let mut a = 1.0; let mut b = 2.0; b = a; a = x; \
                 return a + b; }"
            ),
            Ty::Scalar(Scalar::F64)
        );
    }

    #[test]
    fn the_first_use_pins_the_default() {
        // A narrower operand promotes into the default losslessly: the sum stays i32.
        assert_eq!(
            first_let_init_ty("fn f(x: u8) -> i32 { let k = 2; return x + k; }"),
            Ty::Scalar(Scalar::I32)
        );
        // An integer literal never becomes a float.
        assert_eq!(
            first_let_init_ty("fn f(x: f64) -> f64 { let n = 4; return x / (n as f64); }"),
            Ty::Scalar(Scalar::I32)
        );
        // The call fixes `k: i32` first, so the later product promotes it instead of retyping it.
        let src = "fn g(n: i32) -> i32 { return n; } \
                   fn f(x: i64) -> i64 { let k = 2; let a = g(k); return x * k + (a as i64); }";
        assert!(codes(src).is_empty(), "{:?}", codes(src));
        // Not an all-literal initializer: nothing to infer.
        assert_eq!(
            first_let_init_ty("fn f(x: f64) -> f64 { let y = x; return y; }"),
            Ty::Scalar(Scalar::F64)
        );
    }
}
