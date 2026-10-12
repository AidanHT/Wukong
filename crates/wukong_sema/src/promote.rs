//! Arithmetic between two scalars of different types: the lossless common type, or `E0440`.
//!
//! An arithmetic operator used to promote any two scalars to "the wider, a float beats an integer,
//! the LEFT operand on a tie". So `i32 + u32` was `i32` while `u32 + i32` was `u32`, `bf16 + f16`
//! and `f16 + bf16` disagreed, `i64 + f32` silently became `f32` (every integer above 2^24 rounded),
//! and `/` and `%` took their signedness from the left operand's own type, which made `u32 / i64`
//! an UNSIGNED 64-bit divide (`7 / -2 == 0`). Both backends agree on all of it, so no differential
//! gate could see it.
//!
//! The rule now is Zig's peer-type resolution restricted to promotions that lose nothing and do not
//! depend on operand order (Rust rejects every mix; this keeps the ones a reader cannot get wrong):
//!
//! | operands | common type |
//! |---|---|
//! | two integers of one signedness | the wider (`i32 + i64` is `i64`) |
//! | an unsigned integer and a strictly wider signed one | the signed (`u32 + i64` is `i64`) |
//! | a signed integer known to be non-negative and an unsigned one at least as wide | the unsigned |
//! | `f16`/`bf16` with `f32`/`f64`, or `f32` with `f64` | the wider float — never `f16` with `bf16` |
//! | `f8e4m3`/`f8e5m2` with any wider float | the wider float — never `f8e4m3` with `f8e5m2` |
//! | an integer and a float that holds every value of it exactly | the float (`i16 + f32`, `i32 + f64`, `u8 + bf16`) |
//! | `usize` with `u64`, `isize` with `i64` | one 64-bit type under two names: the left operand's name |
//!
//! Every other pair of non-literal scalars is `E0440`, with an `as` hint. *Known to be non-negative*
//! is a `for` counter that starts at a non-negative literal, steps up and is never assigned in the
//! body (`for i in 0..n` is the loop of every kernel, and `i * N` with a `usize` dimension `N` stays
//! legal), or a `const` whose value folds to a non-negative integer. An unsuffixed literal operand
//! adapts as it always has, so `x * 2` and `2.0 * x` need no cast — and a tie in width now goes to
//! the typed operand, so `1 + x` and `x + 1` have one type. A shift amount is a count and takes no
//! part, and `bool` operands keep their own rules.

use wukong_ast::{
    AssignOp, BinOp, Block, Expr, ExprKind, ForIter, PatKind, Pattern, Stmt, StmtKind, UnOp,
};
use wukong_diag::Diagnostic;
use wukong_span::{Span, Symbol};
use wukong_types::{Scalar, Ty};

use crate::{eval_const_int, has_float_suffix, has_int_suffix, DefKind, LocalFact, Sema};

/// The common type of arithmetic on two scalars `a` and `b` (`a` the left operand), or `None` when
/// no lossless, order-independent one exists. `a_nonneg`/`b_nonneg` say the operand is a signed
/// value known never to be negative.
pub(crate) fn promote(a: Scalar, b: Scalar, a_nonneg: bool, b_nonneg: bool) -> Option<Scalar> {
    use Scalar::*;
    if a == b {
        return Some(a);
    }
    // One 64-bit type under two names: no value changes either way.
    if matches!(
        (a, b),
        (Usize, U64) | (U64, Usize) | (Isize, I64) | (I64, Isize)
    ) {
        return Some(a);
    }
    if matches!(a, Char | Bool) || matches!(b, Char | Bool) {
        return None;
    }
    match (a.is_float(), b.is_float()) {
        (true, true) => {
            // Neither of a pair of same-width floats holds the other: `f16`/`bf16` trade range
            // for precision, and so do the two 8-bit floats. Every f8 value is exactly an
            // `f16`, a `bf16`, an `f32` and an `f64`, so an f8 widens to any of those.
            if matches!(
                (a, b),
                (F16, Bf16) | (Bf16, F16) | (F8E4M3, F8E5M2) | (F8E5M2, F8E4M3)
            ) {
                None
            } else if a.size() >= b.size() {
                Some(a)
            } else {
                Some(b)
            }
        }
        (false, false) => {
            if a.is_signed() == b.is_signed() {
                return Some(if a.size() >= b.size() { a } else { b });
            }
            let (s, u, s_nonneg) = if a.is_signed() {
                (a, b, a_nonneg)
            } else {
                (b, a, b_nonneg)
            };
            if s.size() > u.size() {
                Some(s)
            } else if s_nonneg {
                Some(u)
            } else {
                None
            }
        }
        (true, false) => exact_in(b, a).then_some(a),
        (false, true) => exact_in(a, b).then_some(b),
    }
}

/// Whether `a` and `b` are one type, possibly under the two names of a 64-bit alias pair.
fn same_repr(a: Scalar, b: Scalar) -> bool {
    use Scalar::*;
    a == b
        || matches!(
            (a, b),
            (Usize, U64) | (U64, Usize) | (Isize, I64) | (I64, Isize)
        )
}

/// Whether float type `f` holds every value of integer type `i` exactly: its magnitude bits fit
/// the float's significand (11 bits for `f16`, 8 for `bf16`, 24 for `f32`, 53 for `f64`).
fn exact_in(i: Scalar, f: Scalar) -> bool {
    let magnitude_bits = i.size() * 8 - u64::from(i.is_signed());
    let precision: u64 = match f {
        Scalar::F8E5M2 => 3,
        Scalar::F8E4M3 => 4,
        Scalar::F16 => 11,
        Scalar::Bf16 => 8,
        Scalar::F32 => 24,
        _ => 53,
    };
    magnitude_bits <= precision
}

/// A type to suggest converting both operands of a rejected pair to: `f32` for two half floats,
/// `f64` for an integer with a float, else the signed integer twice as wide as the wider operand
/// (capped at `i64`).
fn suggested_type(a: Scalar, b: Scalar) -> Scalar {
    match (a.is_float(), b.is_float()) {
        (true, true) => Scalar::F32,
        (true, false) | (false, true) => Scalar::F64,
        (false, false) => match a.size().max(b.size()) {
            1 => Scalar::I16,
            2 => Scalar::I32,
            16 => Scalar::I128,
            _ => Scalar::I64,
        },
    }
}

/// Why `a` and `b` have no lossless common type, for the `E0440` message.
fn why_not(a: Scalar, b: Scalar) -> String {
    let (an, bn) = (a.name(), b.name());
    if a == Scalar::Char || b == Scalar::Char {
        return "`char` arithmetic combines only with another `char` or an integer literal"
            .to_string();
    }
    match (a.is_float(), b.is_float()) {
        (true, true) => format!("`{an}` and `{bn}` each hold values the other cannot"),
        (false, false) => format!(
            "`{an}` and `{bn}` differ in signedness and neither holds every value of the other, \
             so the result would depend on which operand's signedness it took"
        ),
        _ => {
            let (i, f) = if a.is_float() { (bn, an) } else { (an, bn) };
            format!("`{f}` cannot hold every `{i}` value exactly")
        }
    }
}

impl Sema<'_> {
    /// The result type of arithmetic `lhs op rhs` whose operands are two *different* scalars
    /// (`l`, `r`): the lossless common type, `Ty::Error` after reporting `E0440`, or `None` when the
    /// pair is not this rule's to judge (not two scalars, a shift, a `bool` operand, or an
    /// unsuffixed literal — which adapts, except that a tie in width goes to the typed operand).
    pub(crate) fn arith_promotion(
        &mut self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        l: &Ty,
        r: &Ty,
        span: Span,
    ) -> Option<Ty> {
        let (Ty::Scalar(a), Ty::Scalar(b)) = (l, r) else {
            return None;
        };
        let (a, b) = (*a, *b);
        if a == b || matches!(op, BinOp::Shl | BinOp::Shr) || a == Scalar::Bool || b == Scalar::Bool
        {
            return None;
        }
        let (llit, rlit) = (self.is_literal_operand(lhs), self.is_literal_operand(rhs));
        if llit || rlit {
            // The literal adapts to the typed operand. The wider type wins as before, but a tie in
            // width used to go to whichever operand was written first, so `1 + x` (with `x: u32`)
            // was an `i32` and `1 / x` a SIGNED divide; it is `u32` now, as `x + 1` always was.
            if llit && !rlit && a.size() == b.size() && a.is_float() == b.is_float() {
                return Some(r.clone());
            }
            return None;
        }
        let (an, bn) = (self.is_nonneg_operand(lhs), self.is_nonneg_operand(rhs));
        if let Some(t) = promote(a, b, an, bn) {
            return Some(Ty::Scalar(t));
        }
        self.diags.push(
            Diagnostic::error(format!(
                "mixed-type arithmetic: `{} {} {}` has no lossless common type",
                a.name(),
                op.glyph(),
                b.name()
            ))
            .with_code("E0440")
            .primary(span, why_not(a, b))
            .help(format!(
                "convert the operands with `as` to the type the result should have, for example \
                 `(x as {t}) {} (y as {t})`; implicit promotion only widens (`i32` to `i64`, `u8` \
                 to `i32`, `f32` to `f64`, `i16` to `f32`)",
                op.glyph(),
                t = suggested_type(a, b).name(),
            )),
        );
        Some(Ty::Error)
    }

    /// The plain-assignment rule applied to a compound assignment `x op= v` on scalars (`E0441`):
    /// the implied `x op v` must be of the place's own type. So `v` may be the place's type, a type
    /// that promotes into it losslessly (`i64 += i32`, `f64 += f32`, `i32 += u8`), or an
    /// unsuffixed literal (range-checked against an integer place, and never a float literal into
    /// an integer place). A shift amount is a count of any integer type. A `bool` on either side is
    /// left to the operator rules.
    pub(crate) fn check_compound_operand(
        &mut self,
        op: AssignOp,
        target: &Expr,
        value: &Expr,
        ts: Scalar,
        vs: Scalar,
    ) {
        if ts == vs || ts == Scalar::Bool || vs == Scalar::Bool {
            return;
        }
        let shift = matches!(op, AssignOp::Shl | AssignOp::Shr);
        if self.is_literal_operand(value) {
            if ts.is_int() && vs.is_float() {
                self.report_compound(op, target, value, ts, vs, Some(vs));
            } else if ts.is_int() && !shift {
                self.range_check_int_literal(value, &Ty::Scalar(ts));
            } else if ts.is_f8() && vs.is_float() {
                // The operand is rounded onto the place's 8-bit grid before the operation.
                self.range_check_f8_literal(value, ts);
            } else if ts.is_f8() && vs.is_int() && !shift {
                // So is an integer literal (`e += 1000`), converted to the place's type (`f8.rs`).
                self.range_check_f8_int_literal(value, ts);
            }
            return;
        }
        if shift {
            if !vs.is_int() {
                self.report_compound(op, target, value, ts, vs, Some(vs));
            }
            return;
        }
        let implied = promote(
            ts,
            vs,
            self.is_nonneg_operand(target),
            self.is_nonneg_operand(value),
        );
        if implied.is_some_and(|t| same_repr(t, ts)) {
            return;
        }
        self.report_compound(op, target, value, ts, vs, implied);
    }

    fn report_compound(
        &mut self,
        op: AssignOp,
        target: &Expr,
        value: &Expr,
        ts: Scalar,
        vs: Scalar,
        implied: Option<Scalar>,
    ) {
        let (tn, vn) = (ts.name(), vs.name());
        let bin = op.glyph().trim_end_matches('=');
        let why = match implied {
            Some(t) => format!(
                "`{tn} {bin} {vn}` computes in `{}`, which does not fit back into the `{tn}` place",
                t.name()
            ),
            None => format!("`{tn}` and `{vn}` have no lossless common type"),
        };
        self.diags.push(
            Diagnostic::error(format!(
                "compound assignment `{}` of a `{vn}` operand to a `{tn}` place: {why}",
                op.glyph()
            ))
            .with_code("E0441")
            .primary(value.span, format!("this is `{vn}`"))
            .secondary(target.span, format!("this place is `{tn}`"))
            .help(format!(
                "convert the operand explicitly (`x {} (v as {tn})`), or give the place the wider \
                 type; a `let` initialized from a literal (`let mut s = 0.0;`) takes it \
                 automatically from its first such use",
                op.glyph()
            )),
        );
    }

    /// Whether `e` is an unsuffixed numeric literal — possibly negated, or arithmetic on such
    /// literals — whose type adapts to the other operand.
    pub(crate) fn is_literal_operand(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Int(s) => !has_int_suffix(self.sym_str(*s)),
            ExprKind::Float(s) => !has_float_suffix(self.sym_str(*s)),
            ExprKind::Unary {
                op: UnOp::Neg,
                expr,
            } => self.is_literal_operand(expr),
            ExprKind::Binary {
                op:
                    BinOp::Add
                    | BinOp::Sub
                    | BinOp::Mul
                    | BinOp::Div
                    | BinOp::Rem
                    | BinOp::BitAnd
                    | BinOp::BitOr
                    | BinOp::BitXor,
                lhs,
                rhs,
            } => self.is_literal_operand(lhs) && self.is_literal_operand(rhs),
            _ => false,
        }
    }

    /// Whether `e` names a value that is never negative: a non-negative `for` counter
    /// ([`LocalFact::NonNegCounter`]) or a top-level `const` whose value folds to an integer >= 0.
    pub(crate) fn is_nonneg_operand(&self, e: &Expr) -> bool {
        let ExprKind::Path(p) = &e.kind else {
            return false;
        };
        if !p.is_single() {
            return false;
        }
        let sym = p.first().sym;
        if self.lookup_local(sym).is_some() {
            return self.local_fact(sym) == Some(LocalFact::NonNegCounter);
        }
        matches!(
            self.defs.lookup(sym).map(|d| &d.kind),
            Some(DefKind::Const(Ty::Scalar(s))) if s.is_int()
        ) && self
            .consts
            .get(&sym)
            .and_then(|init| eval_const_int(init, self.interner))
            .is_some_and(|v| v >= 0)
    }

    /// Record a `for` counter that can never be negative: a single-name pattern over a range that
    /// starts at a non-negative integer literal and steps up (no `step`, or a positive literal one),
    /// whose body never assigns or mutably borrows it. Called right after the pattern is bound.
    pub(crate) fn note_nonneg_counter(&mut self, pat: &Pattern, iter: &ForIter, body: &Block) {
        let PatKind::Ident(name) = &pat.kind else {
            return;
        };
        let ForIter::Range { start, step, .. } = iter else {
            return;
        };
        let starts_nonneg = eval_const_int(start, self.interner).is_some_and(|v| v >= 0);
        let steps_up = step
            .as_ref()
            .is_none_or(|s| eval_const_int(s, self.interner).is_some_and(|v| v > 0));
        if starts_nonneg && steps_up && !block_writes(body, *name) {
            if let Some(facts) = self.local_facts.last_mut() {
                facts.insert(*name, LocalFact::NonNegCounter);
            }
        }
    }
}

/// Whether `b` may change the local `name`: an assignment to it (plain or compound) or a `&mut` of
/// it, anywhere inside. Conservative: a shadowing binding's assignments count too.
pub(crate) fn block_writes(b: &Block, name: Symbol) -> bool {
    b.stmts.iter().any(|s| stmt_writes(s, name))
        || b.tail.as_ref().is_some_and(|t| expr_writes(t, name))
}

fn stmt_writes(s: &Stmt, name: Symbol) -> bool {
    match &s.kind {
        StmtKind::Let { init, .. } => init.as_ref().is_some_and(|e| expr_writes(e, name)),
        StmtKind::Assign { target, value, .. } => {
            is_name(target, name) || expr_writes(target, name) || expr_writes(value, name)
        }
        StmtKind::Expr(e) | StmtKind::Defer(e) => expr_writes(e, name),
        StmtKind::Return(e) | StmtKind::Break(_, e) => {
            e.as_ref().is_some_and(|e| expr_writes(e, name))
        }
        StmtKind::Continue(_) => false,
        StmtKind::While { cond, body, .. } => expr_writes(cond, name) || block_writes(body, name),
        StmtKind::For { iter, body, .. } => {
            let in_iter = match iter {
                ForIter::Range {
                    start, end, step, ..
                } => {
                    expr_writes(start, name)
                        || end.as_ref().is_some_and(|e| expr_writes(e, name))
                        || step.as_ref().is_some_and(|e| expr_writes(e, name))
                }
                ForIter::Expr(e) => expr_writes(e, name),
            };
            in_iter || block_writes(body, name)
        }
    }
}

fn expr_writes(e: &Expr, name: Symbol) -> bool {
    match &e.kind {
        ExprKind::Unary {
            op: UnOp::RefMut,
            expr,
        } => is_name(expr, name) || expr_writes(expr, name),
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => expr_writes(expr, name),
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => expr_writes(base, name),
        ExprKind::Binary { lhs, rhs, .. } => expr_writes(lhs, name) || expr_writes(rhs, name),
        ExprKind::Call { callee, args, .. } => {
            expr_writes(callee, name) || args.iter().any(|a| expr_writes(a, name))
        }
        ExprKind::Index { base, indices } => {
            expr_writes(base, name) || indices.iter().any(|i| expr_writes(i, name))
        }
        ExprKind::StructLit { fields, rest, .. } => {
            fields.iter().any(|f| expr_writes(&f.value, name))
                || rest.as_ref().is_some_and(|r| expr_writes(r, name))
        }
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => xs.iter().any(|x| expr_writes(x, name)),
        ExprKind::ArrayRepeat { value, count } => {
            expr_writes(value, name) || expr_writes(count, name)
        }
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => block_writes(b, name),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            expr_writes(cond, name)
                || block_writes(then_branch, name)
                || else_branch.as_ref().is_some_and(|e| expr_writes(e, name))
        }
        ExprKind::Match { scrutinee, arms } => {
            expr_writes(scrutinee, name)
                || arms.iter().any(|a| {
                    a.guard.as_ref().is_some_and(|g| expr_writes(g, name))
                        || expr_writes(&a.body, name)
                })
        }
        // A closure's capture is a read-only copy (assigning it is E0304), so only a `let` in its
        // body can write the name; counting that errs toward "written", the safe answer.
        ExprKind::Closure { body, .. } => expr_writes(body, name),
        ExprKind::Let { scrutinee, .. } => expr_writes(scrutinee, name),
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::Path(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_) => false,
    }
}

/// Whether `e` is exactly the single-segment path `name`.
fn is_name(e: &Expr, name: Symbol) -> bool {
    matches!(&e.kind, ExprKind::Path(p) if p.is_single() && p.first().sym == name)
}

#[cfg(test)]
mod tests {
    use super::promote;
    use wukong_span::{Interner, SourceId};
    use wukong_types::Scalar::*;

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
    fn lossy_or_order_dependent_mixes_are_e0440() {
        for src in [
            "fn f(a: u32, b: i32) -> i32 { return a / b; }",
            "fn f(a: i64, b: u64) -> u64 { return a + b; }",
            "fn f(a: i64, b: f32) -> f32 { return b * a; }",
            "fn f(a: i32, b: f32) -> f32 { return a * b; }",
            "fn f(a: f16, b: bf16) -> f32 { return (a + b) as f32; }",
            "fn f(c: char, k: i32) -> i32 { return (c + k) as i32; }",
            "fn f(a: i32, n: usize) -> usize { return a * n; }",
            // A counter that may go negative, or that the body reassigns, is not exempt.
            "fn f(n: usize) -> usize { let mut s: usize = 0; for i in -2..4 { s = s + i * n; } \
             return s; }",
            "fn f(n: usize) -> usize { let mut s: usize = 0; for i in 0..4 { i = 1; s = s + i * n; } \
             return s; }",
            "const K: i32 = -4; fn f(n: u32) -> u32 { return K * n; }",
        ] {
            assert!(codes(src).contains(&"E0440"), "expected E0440 for {src:?}");
        }
    }

    #[test]
    fn lossless_mixes_and_literals_are_accepted() {
        for src in [
            "fn f(a: u32, b: i64) -> i64 { return a / b; }",
            "fn f(a: i32, b: i64) -> i64 { return b - a; }",
            "fn f(a: u8, b: i32) -> i32 { return a % b; }",
            "fn f(a: i16, b: f32) -> f32 { return a * b; }",
            "fn f(a: i32, b: f64) -> f64 { return b + a; }",
            "fn f(a: bf16, b: f32) -> f32 { return a + b; }",
            "fn f(a: f32, b: f64) -> f64 { return a * b; }",
            "fn f(a: usize, b: u64) -> usize { return a + b; }",
            // The kernel loop: a non-negative counter times a `usize` dimension.
            "fn f(n: usize) -> usize { let mut s: usize = 0; for i in 0..4 { s = s + i * n; } \
             return s; }",
            "fn f<N>(a: Tensor[f32, N]) -> f32 { let mut s: f32 = 0.0; \
             for i in 0..N { s = s + a[N - 1 - i]; } return s; }",
            "const K: i32 = 4; fn f(n: u32) -> u32 { return K * n; }",
            // Literals adapt; a same-width literal on the left takes the typed operand's type.
            "fn f(x: u32) -> u32 { let y: u32 = 1 + x; return y * 2; }",
            "fn f(x: f64) -> f64 { return 2.0 * x + 1.0; }",
            "fn f(x: u8) -> i32 { return x + 1; }",
            // A shift amount is a count.
            "fn f(x: u64, k: i32) -> u64 { return x >> k; }",
        ] {
            let c = codes(src);
            assert!(c.is_empty(), "unexpected {c:?} for {src:?}");
        }
    }

    #[test]
    fn promotion_is_lossless_and_symmetric() {
        let all = [
            I8, I16, I32, I64, U8, U16, U32, U64, Usize, Isize, F16, Bf16, F32, F64, Char,
        ];
        for a in all {
            for b in all {
                let ab = promote(a, b, false, false);
                let ba = promote(b, a, false, false);
                // Order-independent, except for the name of the 64-bit alias pairs.
                let alias = matches!(
                    (a, b),
                    (Usize, U64) | (U64, Usize) | (Isize, I64) | (I64, Isize)
                );
                if !alias {
                    assert_eq!(ab, ba, "{a:?} and {b:?}");
                }
                // The common type is one of the operands, at least as wide as both.
                if let Some(t) = ab {
                    assert!(t == a || t == b, "{a:?} {b:?} -> {t:?}");
                    assert!(
                        t.size() >= a.size() && t.size() >= b.size(),
                        "{a:?} {b:?} -> {t:?}"
                    );
                }
            }
        }
        assert_eq!(promote(I32, I64, false, false), Some(I64));
        assert_eq!(promote(U32, I64, false, false), Some(I64));
        assert_eq!(promote(U8, I32, false, false), Some(I32));
        assert_eq!(promote(I32, U32, false, false), None);
        assert_eq!(promote(I64, U64, false, false), None);
        assert_eq!(promote(I32, Usize, false, false), None);
        assert_eq!(promote(I32, Usize, true, false), Some(Usize));
        assert_eq!(promote(Usize, I32, false, true), Some(Usize));
        assert_eq!(promote(F16, Bf16, false, false), None);
        assert_eq!(promote(Bf16, F32, false, false), Some(F32));
        assert_eq!(promote(F32, F64, false, false), Some(F64));
        assert_eq!(promote(I16, F32, false, false), Some(F32));
        assert_eq!(promote(I32, F32, false, false), None);
        assert_eq!(promote(I32, F64, false, false), Some(F64));
        assert_eq!(promote(I64, F64, false, false), None);
        assert_eq!(promote(U8, Bf16, false, false), Some(Bf16));
        assert_eq!(promote(U16, F16, false, false), None);
        assert_eq!(promote(Char, I32, false, false), None);
        assert_eq!(promote(Usize, U64, false, false), Some(Usize));
    }
}
