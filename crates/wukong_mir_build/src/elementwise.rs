//! Spelling coverage for the **elementwise** kernel recognizers (`wukong_vmath_f32`,
//! `wukong_velem_f32`, `wukong_vmath2_f32`, `wukong_vhorner_f32`, `wukong_dequant_f32`).
//!
//! Two respellings used to cost the 256-bit kernel while computing exactly what the recognized form
//! computes:
//!
//! * **A hoisted value binding** — `let v = x[i]; o[i] = gelu(v);`. The probes are retried over the
//!   view [`crate::canon::inline_value_lets`] builds, and only when the original body declined, so a
//!   body that already dispatched keeps its kernel and a body the view cannot help is lowered from
//!   the original AST exactly as before.
//! * **The textbook GELU expansion** — `0.5*v*(1.0 + tanh(0.7978845608*(v + 0.044715*(v*v*v))))`.
//!   Folded into `VMATH_GELU` only in the association that IS the kernel's arithmetic (see
//!   [`FnLowerer::match_gelu_tanh_expansion`]), so the dispatched loop and the composed expression
//!   agree bit-for-bit. The erf form (`0.5*v*(1+erf(v*0.70710678))`) is a different function from
//!   the kernel's tanh approximation and is never substituted.

use wukong_ast::{self as ast, Block, Expr, ExprKind};
use wukong_mir::{MirType, ValueId};
use wukong_span::Symbol;

use crate::{canon, parse_float, parse_int, FnLowerer, MathIntrinsic};

/// `√(2/π)` exactly as the runtime's `gelu1`/`gelu8` (`vmath.rs` `GELU_C0`) and the inline
/// `emit_gelu` hold it, as an f32. Every decimal spelling a model transcribes —
/// `0.7978845608`, `0.79788456`, Python's `math.sqrt(2/math.pi)` = `0.7978845608028654` — rounds to
/// this same f32 (`0x3f4c422a`).
const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;
/// The cubic coefficient of the tanh approximation (`vmath.rs` `GELU_C1`), `0x3d372713`.
const GELU_CUBIC: f32 = 0.044715;

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
    /// already lowered, as `try_vectorize_ranged` has them. Only the two chunk kernels with an
    /// infallible emitter (activation and streaming affine map) are retried; anything else falls to
    /// the generic vectorizer over the original body, exactly as before.
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
            self.emit_vmath_calls(s, e, calls);
            return true;
        }
        if let Some(plan) = self.match_velem_body(j, &view) {
            let s = self.coerce_to(start_val, ity, &MirType::I64, true);
            let e = self.coerce_to(end_val, ity, &MirType::I64, true);
            self.emit_velem_call(s, e, &plan);
            return true;
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

    /// Recognize the tanh-approximation GELU written out by hand, over one unit-stride f32 read
    /// `v = x[j]` (or `x[r*C + j]` when `batch` is set), and return `x`.
    ///
    /// Accepted — and ONLY this association, because it is the one whose IEEE evaluation is the
    /// runtime kernel's (`vmath.rs` `gelu1`: `(0.5*x) * (1 + tanh_exp1(C0 * fma(C1, (x*x)*x, x)))`)
    /// and the inline `emit_gelu`'s, operation for operation:
    ///
    /// ```text
    ///   (0.5 * v) * (1.0 + tanh(C0 * (v + C1 * ((v * v) * v))))
    /// ```
    ///
    /// with each single binary operation's operands in either order (`*` and `+` are commutative
    /// in IEEE arithmetic, and the source's `v + C1*x3` is contracted to the single-rounding
    /// `fma(C1, x3, v)` by `try_contract_fma`, which is the kernel's `GELU_C1.mul_add(x3, x)`), the
    /// cube as `(v*v)*v` or `v*(v*v)`, and `C0`/`C1` any literal that rounds to the kernel's f32
    /// constants. `tanh` here is the inline `tanh()` intrinsic (`emit_tanh`,
    /// `1 - 2/(exp(2z)+1)`), which is exactly the kernel's `tanh_exp1`.
    ///
    /// Declined, deliberately:
    /// * HuggingFace's literal `0.044715 * v * v * v`, which parses as `((C1*v)*v)*v` and contracts
    ///   to `fma((C1*v)*v, v, v)` — a different rounding of the cubic term, so substituting the
    ///   kernel would change results in the last bits;
    /// * `0.5 * (v * (1 + t))` and other re-associations of the outer product;
    /// * the erf form `0.5*v*(1+erf(v*K))`, which is exact GELU, a different function.
    pub(super) fn match_gelu_tanh_expansion(
        &self,
        value: &Expr,
        j: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<Symbol> {
        if self.expr_mir(value) != MirType::F32 {
            return None;
        }
        let (a, b) = mul_operands(value)?;
        let try_order = |half: &Expr, sum: &Expr| -> Option<Symbol> {
            let x = self.gelu_half_v(half, j, batch)?;
            let z = self.gelu_one_plus_tanh(sum)?;
            let x2 = self.gelu_tanh_arg(z, j, batch)?;
            (x == x2).then_some(x)
        };
        try_order(a, b).or_else(|| try_order(b, a))
    }

    /// `e` is a float literal typed `f32` whose value, rounded to f32 as the lowering rounds it, is
    /// bit-identical to `want`.
    fn is_f32_literal(&self, e: &Expr, want: f32) -> bool {
        if self.expr_mir(e) != MirType::F32 {
            return false;
        }
        let v = match &e.kind {
            ExprKind::Float(t) => parse_float(self.interner.resolve(*t)),
            ExprKind::Int(t) => parse_int(self.interner.resolve(*t)) as f64,
            _ => return false,
        };
        (v as f32).to_bits() == want.to_bits()
    }

    /// `e` is the unit-stride f32 read `x[j]` (batch-offset aware); returns `x`.
    fn gelu_v(&self, e: &Expr, j: Symbol, batch: Option<(Symbol, &Expr)>) -> Option<Symbol> {
        if self.expr_mir(e) != MirType::F32 {
            return None;
        }
        self.index_off(e, j, batch)
    }

    /// `0.5 * v` or `v * 0.5`.
    fn gelu_half_v(&self, e: &Expr, j: Symbol, batch: Option<(Symbol, &Expr)>) -> Option<Symbol> {
        let (a, b) = mul_operands(e)?;
        if self.is_f32_literal(a, 0.5) {
            return self.gelu_v(b, j, batch);
        }
        if self.is_f32_literal(b, 0.5) {
            return self.gelu_v(a, j, batch);
        }
        None
    }

    /// `1.0 + tanh(z)` or `tanh(z) + 1.0`; returns `z`. Neither operand is a product, so no FMA
    /// contraction applies and the lowering is the plain `fadd` the kernel performs.
    fn gelu_one_plus_tanh<'e>(&self, e: &'e Expr) -> Option<&'e Expr> {
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs,
            rhs,
        } = &e.kind
        else {
            return None;
        };
        let tanh_arg = |t: &'e Expr| -> Option<&'e Expr> {
            let ExprKind::Call { callee, args, .. } = &t.kind else {
                return None;
            };
            if args.len() != 1
                || !matches!(
                    self.vectorizable_intrinsic(callee),
                    Some(MathIntrinsic::Tanh)
                )
            {
                return None;
            }
            Some(&args[0])
        };
        let (lhs, rhs): (&'e Expr, &'e Expr) = (lhs, rhs);
        if self.is_f32_literal(lhs, 1.0) {
            return tanh_arg(rhs);
        }
        if self.is_f32_literal(rhs, 1.0) {
            return tanh_arg(lhs);
        }
        None
    }

    /// `C0 * (v + C1*x3)` (either factor order of the outer product); returns `x`.
    fn gelu_tanh_arg(&self, e: &Expr, j: Symbol, batch: Option<(Symbol, &Expr)>) -> Option<Symbol> {
        let (a, b) = mul_operands(e)?;
        let inner = if self.is_f32_literal(a, GELU_SQRT_2_OVER_PI) {
            b
        } else if self.is_f32_literal(b, GELU_SQRT_2_OVER_PI) {
            a
        } else {
            return None;
        };
        self.gelu_cubic_sum(inner, j, batch)
    }

    /// `v + C1*x3` or `C1*x3 + v` (and `x3*C1`). `try_contract_fma` contracts the product operand —
    /// the left one when it is a product, else the right — so both spellings lower to
    /// `fma(C1, x3, v)` (the multiplicands of a fused multiply-add commute exactly). `v` itself is an
    /// index read, never a product, so it can never be the operand that gets contracted.
    fn gelu_cubic_sum(
        &self,
        e: &Expr,
        j: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<Symbol> {
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs,
            rhs,
        } = &e.kind
        else {
            return None;
        };
        let (v, prod) = if mul_operands(lhs).is_some() {
            (rhs.as_ref(), lhs.as_ref())
        } else {
            (lhs.as_ref(), rhs.as_ref())
        };
        let x = self.gelu_v(v, j, batch)?;
        let (p, q) = mul_operands(prod)?;
        let cube = if self.is_f32_literal(p, GELU_CUBIC) {
            q
        } else if self.is_f32_literal(q, GELU_CUBIC) {
            p
        } else {
            return None;
        };
        (self.gelu_cube(cube, j, batch)? == x).then_some(x)
    }

    /// `(v*v)*v` or `v*(v*v)` — both round `v*v` first and then multiply by `v`, which is the
    /// kernel's `x * x * x`.
    fn gelu_cube(&self, e: &Expr, j: Symbol, batch: Option<(Symbol, &Expr)>) -> Option<Symbol> {
        let (a, b) = mul_operands(e)?;
        let (sq, single) = if mul_operands(a).is_some() {
            (a, b)
        } else {
            (b, a)
        };
        let x = self.gelu_v(single, j, batch)?;
        let (s1, s2) = mul_operands(sq)?;
        (self.gelu_v(s1, j, batch)? == x && self.gelu_v(s2, j, batch)? == x).then_some(x)
    }
}

/// The two operands of a binary `*`.
fn mul_operands(e: &Expr) -> Option<(&Expr, &Expr)> {
    match &e.kind {
        ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } => Some((lhs.as_ref(), rhs.as_ref())),
        _ => None,
    }
}
