//! **Compile-time constraints on a function's dimensions**: `where` predicates, `static_assert`,
//! and the type of a `const` generic parameter.
//!
//! `where` used to be parsed and then dropped, so `fn f<K>(..) where K % 16 == 0` compiled as if
//! unconstrained while its author believed it enforced. Like Mojo's `constrained[]` and Rust's
//! `where` bounds plus `const { assert!(..) }`, a dimension predicate is now checked:
//!
//! * **`where` predicates** — `where K % 16 == 0, M <= 4096` — compare *dimension expressions*:
//!   integer literals, the function's generic dimensions and top-level `const`s, combined with
//!   `+ - * / %` (truncating, in 128-bit integers) and joined with `&&` / `||`. Anything else is
//!   `E0524`. Each is evaluated at every call whose dimensions it names are all compile-time
//!   constants, and a false one is `E0521` — "constraint `K % 16 == 0` of `tile` is not satisfied
//!   (K = 24)" — at the call, with the predicate labelled. A predicate over no dimension is decided
//!   once, at the declaration.
//! * **`static_assert(cond)`** / **`static_assert(cond, "message")`** is a statement checked at
//!   compile time and lowered to nothing: a false condition is `E0522`. A condition over the
//!   function's generic dimensions is checked at each call exactly like a `where` predicate, and
//!   must then be a top-level statement of the function body (`E0524` otherwise); one over
//!   literals and `const`s only is decided where it is written.
//! * **A `const` generic parameter** (`<const N: usize>`) is a compile-time dimension, passed as a
//!   hidden `usize` like any other: a declared type other than `usize` (`const FLAG: bool`, which
//!   silently became a `usize`) is `E0523`.
//!
//! A call made from a generic body whose dimensions are still symbolic is not checked there: the
//! constraint is checked where the dimensions become known. A predicate is not propagated into
//! the caller's own constraints.

use wukong_ast::{
    BinOp, Expr, ExprKind, FnDecl, GenericParamKind, ItemKind, Module, StmtKind, TypeKind,
};
use wukong_diag::Diagnostic;
use wukong_span::{FxHashMap as HashMap, Span, Symbol};
use wukong_types::Dim;

use crate::Sema;

/// One constraint of a function, checked at each of its calls.
#[derive(Clone, Copy)]
pub(crate) struct Pred<'a> {
    expr: &'a Expr,
    /// `false` for a `where` predicate, `true` for a `static_assert`.
    is_assert: bool,
    /// A `static_assert`'s message literal.
    msg: Option<Symbol>,
}

/// A value of a constraint expression.
#[derive(Clone, Copy, PartialEq)]
enum Val {
    Int(i128),
    Bool(bool),
}

/// Binding strength of an operator, for printing a predicate back with only the parentheses its
/// structure needs.
fn prec(op: BinOp) -> u8 {
    use BinOp::*;
    match op {
        Or => 1,
        And => 2,
        Eq | Ne | Lt | Le | Gt | Ge => 3,
        Add | Sub => 5,
        Mul | Div | Rem => 6,
        _ => 4,
    }
}

/// Whether `op` may appear in a constraint.
fn constraint_op(op: BinOp) -> bool {
    use BinOp::*;
    matches!(
        op,
        Add | Sub | Mul | Div | Rem | Eq | Ne | Lt | Le | Gt | Ge | And | Or
    )
}

/// Whether `e` names any of `generics`.
fn mentions_any(e: &Expr, generics: &[Symbol]) -> bool {
    match &e.kind {
        ExprKind::Path(p) => p.is_single() && generics.contains(&p.first().sym),
        ExprKind::Binary { lhs, rhs, .. } => {
            mentions_any(lhs, generics) || mentions_any(rhs, generics)
        }
        _ => false,
    }
}

/// The generics `e` names, in first-seen order.
fn collect_names(e: &Expr, generics: &[Symbol], out: &mut Vec<Symbol>) {
    match &e.kind {
        ExprKind::Path(p) => {
            let s = p.first().sym;
            if generics.contains(&s) && !out.contains(&s) {
                out.push(s);
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            collect_names(lhs, generics, out);
            collect_names(rhs, generics, out);
        }
        _ => {}
    }
}

/// The `(condition, message)` of a `static_assert(..)` call statement, if `e` is one.
fn static_assert_args<'e>(e: &'e Expr, s: &Sema<'_>) -> Option<(&'e Expr, Option<Symbol>)> {
    let ExprKind::Call { callee, args, .. } = &e.kind else {
        return None;
    };
    let ExprKind::Path(p) = &callee.kind else {
        return None;
    };
    if !p.is_single() || s.sym_str(p.first().sym) != "static_assert" {
        return None;
    }
    let cond = args.first()?;
    let msg = match args.get(1).map(|m| &m.kind) {
        Some(ExprKind::Str(sym)) => Some(*sym),
        _ => None,
    };
    Some((cond, msg))
}

impl<'a> Sema<'a> {
    /// Collect and validate every function's `where` predicates and top-level `static_assert`s,
    /// and check its `const` generic parameters' types. Runs once the definitions and `const`s are
    /// registered, before the body pass.
    pub(crate) fn collect_constraints(&mut self, module: &'a Module) {
        for item in &module.items {
            let ItemKind::Fn(f) = &item.kind else {
                continue;
            };
            self.check_const_param_types(f);
            let generics: Vec<Symbol> = f.generics.iter().map(crate::generic_param_sym).collect();
            let mut preds = Vec::new();
            for w in &f.where_preds {
                if self.valid_constraint(w, &generics, "a `where` predicate") {
                    preds.push(Pred {
                        expr: w,
                        is_assert: false,
                        msg: None,
                    });
                }
            }
            if let Some(body) = &f.body {
                for s in &body.stmts {
                    let StmtKind::Expr(e) = &s.kind else {
                        continue;
                    };
                    let Some((cond, msg)) = static_assert_args(e, self) else {
                        continue;
                    };
                    // A condition over no dimension is decided in the body pass, where it is typed.
                    if mentions_any(cond, &generics) {
                        self.assert_conds.insert(cond.id);
                        if self.valid_constraint(cond, &generics, "a `static_assert` condition") {
                            preds.push(Pred {
                                expr: cond,
                                is_assert: true,
                                msg,
                            });
                        }
                    }
                }
            }
            // A predicate over no dimension holds or fails for every call: decide it now.
            let mut kept = Vec::with_capacity(preds.len());
            for p in preds {
                if mentions_any(p.expr, &generics) {
                    kept.push(p);
                    continue;
                }
                let empty: HashMap<Symbol, Dim> = HashMap::default();
                if self.eval_constraint(p.expr, &generics, &empty, 0) == Some(Val::Bool(false)) {
                    let msg = format!(
                        "constraint `{}` of `{}` can never be satisfied",
                        self.constraint_text(p.expr),
                        self.sym_str(f.name.sym)
                    );
                    let d = Diagnostic::error(msg)
                        .with_code("E0521")
                        .primary(p.expr.span, "this predicate is false");
                    self.diags.push(d);
                }
            }
            if !kept.is_empty() {
                self.fn_preds.insert(f.name.sym, kept);
            }
        }
    }

    /// `const N: usize` is the only `const` generic parameter type: the value is a dimension,
    /// passed as a hidden `usize` (`E0523`).
    fn check_const_param_types(&mut self, f: &FnDecl) {
        for g in &f.generics {
            let GenericParamKind::Const { name, ty } = &g.kind else {
                continue;
            };
            let is_usize = matches!(&ty.kind, TypeKind::Path(p)
                if p.is_single() && self.sym_str(p.first().sym) == "usize");
            if !is_usize {
                let shown = match &ty.kind {
                    TypeKind::Path(p) => self.sym_str(p.segments.last().unwrap().sym).to_string(),
                    _ => "this type".to_string(),
                };
                let msg = format!(
                    "const generic parameter `{}` has type `{shown}`, but a const generic \
                     parameter is a compile-time dimension of type `usize`",
                    self.sym_str(name.sym)
                );
                let d = Diagnostic::error(msg)
                    .with_code("E0523")
                    .primary(ty.span, "declare it `usize`");
                self.diags.push(d);
            }
        }
    }

    /// Whether `e` is a constraint expression (see the module docs), reporting the first part that
    /// is not (`E0524`).
    fn valid_constraint(&mut self, e: &Expr, generics: &[Symbol], what: &str) -> bool {
        let ok = match &e.kind {
            ExprKind::Int(_) => true,
            ExprKind::Path(p) => {
                p.is_single()
                    && (generics.contains(&p.first().sym)
                        || self.consts.contains_key(&p.first().sym))
            }
            ExprKind::Binary { op, lhs, rhs } => {
                if !constraint_op(*op) {
                    false
                } else {
                    return self.valid_constraint(lhs, generics, what)
                        && self.valid_constraint(rhs, generics, what);
                }
            }
            _ => false,
        };
        if !ok {
            let d = Diagnostic::error(format!(
                "{what} must be a compile-time dimension expression: integer literals, this \
                 function's generic dimensions and `const`s, combined with `+ - * / %`, \
                 compared, and joined with `&&` / `||`"
            ))
            .with_code("E0524")
            .primary(e.span, "not a dimension expression");
            self.diags.push(d);
        }
        ok
    }

    /// Evaluate a constraint with the callee's dimensions `dims` (in the caller's terms). `None`
    /// when a dimension it names is not a compile-time constant here, or the arithmetic has no
    /// value (a zero divisor, an overflow).
    fn eval_constraint(
        &self,
        e: &Expr,
        generics: &[Symbol],
        dims: &HashMap<Symbol, Dim>,
        depth: u32,
    ) -> Option<Val> {
        if depth > 64 {
            return None;
        }
        match &e.kind {
            ExprKind::Int(s) => {
                crate::parse_u64_text(self.sym_str(*s)).map(|n| Val::Int(n as i128))
            }
            ExprKind::Path(p) => {
                let sym = p.first().sym;
                if generics.contains(&sym) {
                    match dims.get(&sym) {
                        Some(Dim::Const(n)) => Some(Val::Int(*n as i128)),
                        _ => None,
                    }
                } else if self.consts.contains_key(&sym) {
                    Some(Val::Int(self.const_length(sym, 0) as i128))
                } else {
                    None
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.eval_constraint(lhs, generics, dims, depth + 1)?;
                let r = self.eval_constraint(rhs, generics, dims, depth + 1)?;
                use BinOp::*;
                match (l, r) {
                    (Val::Int(a), Val::Int(b)) => match op {
                        Add => a.checked_add(b).map(Val::Int),
                        Sub => a.checked_sub(b).map(Val::Int),
                        Mul => a.checked_mul(b).map(Val::Int),
                        Div => a.checked_div(b).map(Val::Int),
                        Rem => a.checked_rem(b).map(Val::Int),
                        Eq => Some(Val::Bool(a == b)),
                        Ne => Some(Val::Bool(a != b)),
                        Lt => Some(Val::Bool(a < b)),
                        Le => Some(Val::Bool(a <= b)),
                        Gt => Some(Val::Bool(a > b)),
                        Ge => Some(Val::Bool(a >= b)),
                        _ => None,
                    },
                    (Val::Bool(a), Val::Bool(b)) => match op {
                        And => Some(Val::Bool(a && b)),
                        Or => Some(Val::Bool(a || b)),
                        Eq => Some(Val::Bool(a == b)),
                        Ne => Some(Val::Bool(a != b)),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// A constraint as source text, with the parentheses its structure needs.
    fn constraint_text(&self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Int(s) => self.sym_str(*s).to_string(),
            ExprKind::Path(p) => self.sym_str(p.first().sym).to_string(),
            ExprKind::Binary { op, lhs, rhs } => {
                let side = |x: &Expr, right: bool| {
                    let t = self.constraint_text(x);
                    match &x.kind {
                        ExprKind::Binary { op: inner, .. }
                            if prec(*inner) < prec(*op) || (right && prec(*inner) == prec(*op)) =>
                        {
                            format!("({t})")
                        }
                        _ => t,
                    }
                };
                format!("{} {} {}", side(lhs, false), op.glyph(), side(rhs, true))
            }
            _ => "..".to_string(),
        }
    }

    /// Check `callee`'s constraints against the dimensions one call bound (`check_fn_call`).
    pub(crate) fn check_call_constraints(
        &mut self,
        callee: Symbol,
        generics: &[Symbol],
        dims: &HashMap<Symbol, Dim>,
        span: Span,
    ) {
        let Some(preds) = self.fn_preds.get(&callee).cloned() else {
            return;
        };
        for p in preds {
            if self.eval_constraint(p.expr, generics, dims, 0) != Some(Val::Bool(false)) {
                continue;
            }
            let text = self.constraint_text(p.expr);
            let mut names: Vec<Symbol> = Vec::new();
            collect_names(p.expr, generics, &mut names);
            let binds: Vec<String> = names
                .iter()
                .filter_map(|g| match dims.get(g) {
                    Some(Dim::Const(n)) => Some(format!("{} = {n}", self.sym_str(*g))),
                    _ => None,
                })
                .collect();
            let fname = self.sym_str(callee).to_string();
            let d = if p.is_assert {
                let mut d = Diagnostic::error(format!(
                    "static assertion `{text}` in `{fname}` failed ({})",
                    binds.join(", ")
                ))
                .with_code("E0522")
                .primary(span, "the dimensions of this call fail it")
                .secondary(p.expr.span, "asserted here");
                if let Some(m) = p.msg {
                    let bytes = crate::text::decode_string_literal(self.sym_str(m));
                    d = d.note(String::from_utf8_lossy(&bytes).into_owned());
                }
                d
            } else {
                Diagnostic::error(format!(
                    "constraint `{text}` of `{fname}` is not satisfied ({})",
                    binds.join(", ")
                ))
                .with_code("E0521")
                .primary(span, "the dimensions of this call violate it")
                .secondary(p.expr.span, "constraint declared here")
            };
            self.diags.push(d);
        }
    }

    /// Type `static_assert(cond)` / `static_assert(cond, "message")`, or return `None` when `name`
    /// is not `static_assert`. A condition over no generic dimension is decided here (`E0522`); one
    /// over the function's dimensions was collected by [`Self::collect_constraints`] and is checked
    /// at each call, which needs it at the top level of the body (`E0524`).
    pub(crate) fn type_static_assert(
        &mut self,
        name: &str,
        args: &[Expr],
        span: Span,
    ) -> Option<wukong_types::Ty> {
        if name != "static_assert" {
            return None;
        }
        let unit = Some(wukong_types::Ty::Unit);
        if args.is_empty() || args.len() > 2 {
            self.error(
                span,
                "E0503",
                format!(
                    "`static_assert` takes a condition and an optional message string, but {} \
                     arguments were supplied",
                    args.len()
                ),
            );
            return unit;
        }
        if let Some(m) = args.get(1) {
            if !matches!(m.kind, ExprKind::Str(_)) {
                self.error(
                    m.span,
                    "E0437",
                    "the message of `static_assert` must be a string literal",
                );
                return unit;
            }
        }
        let cond = &args[0];
        let generics: Vec<Symbol> = self.generics.iter().copied().collect();
        if mentions_any(cond, &generics) {
            if !self.assert_conds.contains(&cond.id) {
                self.error(
                    cond.span,
                    "E0524",
                    "a `static_assert` over the function's generic dimensions is checked at each \
                     call, so it must be a top-level statement of the function body",
                );
            }
            return unit;
        }
        if !self.valid_constraint(cond, &[], "a `static_assert` condition") {
            return unit;
        }
        let empty: HashMap<Symbol, Dim> = HashMap::default();
        match self.eval_constraint(cond, &[], &empty, 0) {
            Some(Val::Bool(true)) => {}
            Some(Val::Bool(false)) => {
                let text = self.constraint_text(cond);
                let mut d = Diagnostic::error(format!("static assertion `{text}` failed"))
                    .with_code("E0522")
                    .primary(cond.span, "this condition is false");
                if let Some(ExprKind::Str(m)) = args.get(1).map(|m| &m.kind) {
                    let bytes = crate::text::decode_string_literal(self.sym_str(*m));
                    d = d.note(String::from_utf8_lossy(&bytes).into_owned());
                }
                self.diags.push(d);
            }
            _ => self.error(
                cond.span,
                "E0524",
                "a `static_assert` condition must be a comparison that evaluates at compile time",
            ),
        }
        unit
    }
}
