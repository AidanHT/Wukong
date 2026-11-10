//! **Closures** — `|x| body` literals, resolved at compile time and checked where they are written.
//!
//! A closure literal is legal as an argument of a function-typed parameter of a call — to a user
//! function, a method (`v.map(|a| a * k)`), a function-typed parameter or a closure-bound local —
//! and as the initializer of a `let` (`let f = |x: f32| x * k;`). When it captures, the name such a
//! `let` binds can only be called or passed to a function-typed parameter (`E0421`: it has no
//! run-time value), and the binding must be one immutable name (`E0422` for `let mut`, a pattern or
//! a reassignment). Anywhere else in a function body — a `let mut`, a `return`, an assignment, a
//! field, an element — a closure is a **run-time function value**, legal only when it captures
//! nothing (`E0427`), as is a closure-bound local that captures nothing used as a value
//! ([`Sema::type_closure_value`]). Outside a function body a closure is `E0411`.
//! `wukong_mir_build` then lifts each literal to a top-level
//! function whose leading parameters are its captures, and specializes the callee per closure
//! exactly as it does per named function (`apply$main$closure$0`), so a call through `f` stays a
//! direct call with no function pointer at run time. Everything that lift needs is decided here and
//! published: a [`ClosureInfo`] per literal ([`crate::SemaResult::closures`]), and the literal each
//! use of a closure-bound local denotes ([`crate::SemaResult::closure_refs`]).
//!
//! * **Typing.** Parameter `i` takes its annotation, else the `i`-th parameter of the expected
//!   function type once the call's other arguments have bound what they can of the callee's
//!   generics ([`Sema::type_args_against`]); one that still names a callee generic, and every
//!   unannotated parameter of a `let`-bound closure with no function-type annotation, is `E0419`.
//!   The return type is the annotation, else the expected return when it is generic-free, else the
//!   body's type (and a body that then uses `return` is `E0419` too: nothing fixes what it
//!   returns). The closure's type `fn(P..) -> R` must equal the expected one after the call's
//!   generic binding — the existing `E0401` of `check_fn_value_args`.
//! * **Captures.** Every enclosing local the body reads is captured. One it only reads is COPIED
//!   where the closure is created (when the call evaluates the argument, or at the `let`): a
//!   scalar, an enum, a raw pointer or reference, a fixed-size array, a tuple, a struct, a slice or
//!   `str` (its fat pointer) or a function value. One it
//!   writes — assigns, takes `&mut` of, or passes as an aggregate to a `mut` parameter or as a
//!   builtin's output operand — is captured BY REFERENCE ([`ClosureInfo::by_ref`],
//!   [`Sema::note_capture_write`]): the closure and the enclosing function share it, and its own
//!   mutability rules apply (`E0304` for an immutable binding, as outside the closure). Either way
//!   a capture has the type its variable is stored at: a `for` range counter's is its wider bound's
//!   ([`Sema::enter_range_counter`]). A local holding a run-time function value is copied like a
//!   pointer, and a slice or `str` as its fat pointer (another view of the same elements). A tensor
//!   or vector, the enclosing function's function-typed parameter, a value whose type names an
//!   enclosing generic and a range counter whose width one decides are `E0418`, as is any use of
//!   an enclosing generic parameter (the lifted function is not generic).
//!   A closure-bound local is captured through its closure ([`ClosureInfo::bound`]).
//! * **Control.** `return` leaves the closure; `break`/`continue` cannot cross it (`E0303`, the
//!   enclosing loops are hidden while its body is checked). Nested closures are allowed: a name an
//!   inner closure captures is captured by every closure it crosses.

use wukong_ast::{Expr, ExprKind, ForIter, NodeId, PatKind, Pattern, Stmt, TypeExpr};
use wukong_diag::Diagnostic;
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Span, Symbol};
use wukong_types::{Dim, Ty};

use crate::{DefKind, Sema};

/// What sema decided about one closure literal, keyed in [`crate::SemaResult::closures`] by the
/// literal's `NodeId`. `wukong_mir_build` lifts the literal to `fn L(captures.., params..) -> ret`.
#[derive(Clone, Debug)]
pub struct ClosureInfo {
    /// The closure's parameter types, in order.
    pub params: Vec<Ty>,
    /// Its return type.
    pub ret: Ty,
    /// Every enclosing local it captures, in order of first use, with its type. A capture is
    /// copied when the closure is created unless it is in [`Self::by_ref`].
    pub captures: Vec<(Symbol, Ty)>,
    /// The captures the closure writes (`c = ..`, `&mut c`, `a[i] = ..`, `relu6(x, out)` into a
    /// `mut` parameter, `softmax(x, out)`): captured BY REFERENCE, so the closure and the enclosing
    /// function share the variable.
    pub by_ref: Vec<Symbol>,
    /// The captures that are locals bound to a closure (`let f = |..| ..;` outside, `f(x)` inside),
    /// with the literal each denotes: the lifted function receives that closure's captures instead.
    pub bound: Vec<(Symbol, NodeId)>,
}

/// One closure whose body is being checked (innermost last in [`Sema::closure_ctx`]).
pub(crate) struct ClosureCtx {
    /// `scopes.len()` before the closure's parameter scope was pushed: a local found in a scope
    /// below this index belongs to the enclosing code, so reading it is a capture.
    base: usize,
    captures: Vec<(Symbol, Ty)>,
    /// The captures written in the body (published as [`ClosureInfo::by_ref`]).
    by_ref: Vec<Symbol>,
    /// The captures that are closure-bound locals (published as [`ClosureInfo::bound`]).
    bound: Vec<(Symbol, NodeId)>,
    /// Names already reported (`E0418`) for this closure: one report per closure and name.
    reported: Vec<Symbol>,
    /// An `E0418`/`E0419`/arity error was reported for this closure: its type is `Error`, so the
    /// call does not add a cascading `E0401`.
    failed: bool,
    /// The body used `return` (an inferred return type cannot account for it).
    returns: bool,
}

/// Why a local cannot be captured.
enum NoCopy {
    /// An aggregate, slice, tensor, vector, unit or undetermined value: not one scalar parameter.
    Value,
    /// A function-typed parameter: bound at compile time, it has no value to copy.
    FnParam,
    /// Its type names a generic parameter of the enclosing function.
    Generic,
    /// A `for` range counter stored at a width a generic of the enclosing function decides.
    CounterWidth,
}

/// The storage of a `for` range counter whose type is not its storage (see
/// [`Sema::enter_range_counter`]).
#[derive(Clone)]
pub(crate) enum CounterWidth {
    /// Stored at this wider integer type, which a capture takes it at.
    Wide(Ty),
    /// Stored at a width the enclosing function's generic parameters decide, which a lifted
    /// (non-generic) closure cannot take it at.
    Generic,
}

impl Sema<'_> {
    /// Type a call's arguments left to right. A closure literal passed to a function-typed
    /// parameter of a user function, of a call through a function-typed parameter or of a call of
    /// a closure-bound local is checked against that parameter's type ([`Self::type_closure`]);
    /// every other argument is typed as usual (a closure there is `E0411`, from `type_expr`).
    pub(crate) fn type_call_args(&mut self, callee: &Expr, args: &[Expr]) -> Vec<Ty> {
        let target = if self.args_need_params(args) {
            match &callee.kind {
                ExprKind::Path(p) if p.is_single() => match self.lookup_local(p.first().sym) {
                    // A call through a run-time function value: nothing is specialized for a
                    // closure given to it, so the closure is a run-time value too.
                    Some(Ty::Fn { params, .. }) if self.holds_run_time_fn(p.first().sym) => {
                        return self.type_args_of_value_call(args, &params);
                    }
                    Some(Ty::Fn { params, .. }) => Some((params, Vec::new())),
                    Some(_) => None,
                    None => match self.defs.lookup(p.first().sym).map(|d| &d.kind) {
                        Some(DefKind::Fn(sig)) => Some((sig.params.clone(), sig.generics.clone())),
                        _ => None,
                    },
                },
                _ => None,
            }
        } else {
            None
        };
        match target {
            Some((params, generics)) => self.type_args_against(args, &params, &generics),
            None => args.iter().map(|a| self.type_expr(a)).collect(),
        }
    }

    /// Whether the local `name` of function type always holds a run-time function value: a `let`,
    /// pattern or `for` binding that is not bound to a closure. A call through one is an
    /// `Op::CallIndirect` in every instance. A parameter (of the function or of a closure) may be
    /// bound at compile time instead, and a call through a closure-bound local is a direct call of
    /// the lifted closure.
    pub(crate) fn holds_run_time_fn(&self, name: Symbol) -> bool {
        !self.is_param_binding(name) && self.closure_bound(name).is_none()
    }

    /// Type the arguments `args` of a call through a run-time function value whose parameter types
    /// are `params`. A closure argument of a function-typed parameter becomes the value the callee
    /// receives — the lifted function's address — so it is typed by
    /// [`Self::type_closure_value`] (one that captures is `E0427`), and a closure-bound local
    /// passed there must capture nothing either.
    pub(crate) fn type_args_of_value_call(&mut self, args: &[Expr], params: &[Ty]) -> Vec<Ty> {
        let mut tys = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let want = params.get(i).filter(|t| matches!(t, Ty::Fn { .. }));
            let t = match (want, &a.kind) {
                (Some(want), ExprKind::Closure { .. }) => {
                    let t = self.type_closure_value(a, Some(want));
                    self.types.insert(a.id, t.clone());
                    t
                }
                (Some(_), ExprKind::Path(p)) if p.is_single() => {
                    let captures: Vec<Symbol> = self
                        .closure_bound(p.first().sym)
                        .and_then(|lit| self.closures.get(&lit))
                        .map(|info| info.captures.iter().map(|(n, _)| *n).collect())
                        .unwrap_or_default();
                    // Typed as an argument of a function-typed parameter, so the use is recorded
                    // without `E0421`'s "only called or passed" (it is passed).
                    let outer = self.fn_value_arg.replace(a.id);
                    let t = self.type_expr(a);
                    self.fn_value_arg = outer;
                    if captures.is_empty() || !matches!(t, Ty::Fn { .. }) {
                        t
                    } else {
                        if self.inst_subst.is_empty() {
                            self.report_capturing_value(a.span, &captures);
                        }
                        self.types.insert(a.id, Ty::Error);
                        Ty::Error
                    }
                }
                _ => self.type_expr(a),
            };
            tys.push(t);
        }
        tys
    }

    /// Whether typing `args` needs the callee's parameter types: an argument is a closure literal,
    /// or names a closure-bound local (which may only be passed to a function-typed parameter).
    pub(crate) fn args_need_params(&self, args: &[Expr]) -> bool {
        args.iter().any(|a| match &a.kind {
            ExprKind::Closure { .. } => true,
            ExprKind::Path(p) => p.is_single() && self.closure_bound(p.first().sym).is_some(),
            _ => false,
        })
    }

    /// Type `args` against the parameter types `params` of a callee whose generics are `generics`
    /// (see [`Self::type_call_args`]): a closure literal at a function-typed parameter is checked
    /// against it, and a single name there may denote a closure-bound local.
    ///
    /// When a closure's expected type names a generic of the callee (`fn fold3<T>(f: fn(T, T) -> T,
    /// a: T, ..)`, `fn mapn<N>(f: fn(Tensor[f32, N]) -> f32, t: Tensor[f32, N])`), the call's other
    /// arguments are typed first and bind what they can — the non-literal ones, then an unsuffixed
    /// literal at its default type, exactly the order the call's own inference uses — and the
    /// closure is checked against the expected type with those bindings applied. A generic they
    /// leave open still needs an annotation (`E0419`).
    pub(crate) fn type_args_against(
        &mut self,
        args: &[Expr],
        params: &[Ty],
        generics: &[Symbol],
    ) -> Vec<Ty> {
        let is_generic = |g: Symbol| generics.contains(&g);
        let closure_at = |i: usize, a: &Expr| {
            matches!(a.kind, ExprKind::Closure { .. })
                && matches!(params.get(i), Some(Ty::Fn { .. }))
        };
        let deferred = !generics.is_empty()
            && args.iter().enumerate().any(|(i, a)| {
                closure_at(i, a) && params.get(i).is_some_and(|p| names_generic(p, &is_generic))
            });
        let mut tys: Vec<Option<Ty>> = vec![None; args.len()];
        for (i, a) in args.iter().enumerate() {
            if deferred && closure_at(i, a) {
                continue;
            }
            tys[i] = Some(self.type_arg(a, params.get(i), generics));
        }
        if deferred {
            let mut types: HashMap<Symbol, Ty> = HashMap::default();
            let mut dims: HashMap<Symbol, Dim> = HashMap::default();
            for literal in [false, true] {
                for (i, a) in args.iter().enumerate() {
                    if let (Some(p), Some(Some(t))) = (params.get(i), tys.get(i)) {
                        if self.is_adaptable_num_literal(a) == literal {
                            crate::shape::infer_type_generics(p, t, generics, &mut types);
                            bind_dims(p, t, &is_generic, &mut dims);
                        }
                    }
                }
            }
            for (i, a) in args.iter().enumerate() {
                if tys[i].is_some() {
                    continue;
                }
                let want = params
                    .get(i)
                    .map(|p| crate::shape::apply_subst(p, &dims, &types));
                let t = self.type_arg(a, want.as_ref(), generics);
                // A closure checked here binds what its own type fixes, for the ones after it.
                if let Some(p) = params.get(i) {
                    crate::shape::infer_type_generics(p, &t, generics, &mut types);
                    bind_dims(p, &t, &is_generic, &mut dims);
                }
                tys[i] = Some(t);
            }
        }
        tys.into_iter().map(|t| t.unwrap_or(Ty::Error)).collect()
    }

    /// Type one argument `a` of a parameter of type `want` (see [`Self::type_args_against`]).
    fn type_arg(&mut self, a: &Expr, want: Option<&Ty>, generics: &[Symbol]) -> Ty {
        let want = want.filter(|t| matches!(t, Ty::Fn { .. }));
        match (want, &a.kind) {
            (Some(want), ExprKind::Closure { .. }) => {
                let t = self.type_closure(a, Some(want), generics);
                self.types.insert(a.id, t.clone());
                t
            }
            (Some(_), ExprKind::Path(p)) if p.is_single() => {
                let outer = self.fn_value_arg.replace(a.id);
                let t = self.type_expr(a);
                self.fn_value_arg = outer;
                t
            }
            _ => self.type_expr(a),
        }
    }

    /// `let f = |..| ..;` — bind the name to the closure. Calls through `f` and arguments `g(f)`
    /// are resolved at compile time, exactly like a closure literal written at each such site,
    /// with the captures copied once, here. So the binding must be immutable and a single name
    /// (`E0422`), and the name can only be called or passed (`E0421`, [`Self::closure_value_use`]).
    /// An annotation, if any, is the closure's function type (`let f: fn(f32) -> f32 = |x| ..`).
    pub(crate) fn type_closure_let(
        &mut self,
        s: &Stmt,
        pat: &Pattern,
        mutable: bool,
        ann: Option<&TypeExpr>,
        e: &Expr,
    ) {
        let report = self.inst_subst.is_empty();
        let name = match &pat.kind {
            PatKind::Ident(n) if !mutable => *n,
            // `let mut f = |..| ..;` — a function variable, when the closure captures nothing.
            PatKind::Ident(n) => return self.type_closure_mut_let(s, pat, *n, ann, e),
            _ => {
                if report {
                    let msg = if mutable {
                        "a closure cannot be bound by a `mut` binding"
                    } else {
                        "a closure must be bound to a name, not a pattern"
                    };
                    self.diags.push(
                        Diagnostic::error(msg)
                            .with_code("E0422")
                            .primary(pat.span, "the closure is bound here")
                            .note(
                                "calls through a closure-bound local are resolved at compile time, \
                                 so the local must name the one closure it is initialized with",
                            )
                            .help("bind it with a plain `let name = |..| ..;`"),
                    );
                }
                self.types.insert(e.id, Ty::Error);
                self.bind_pattern(pat, &Ty::Error);
                return;
            }
        };
        let expected = ann.map(|t| self.lower_type(t));
        let t = match &expected {
            Some(want @ Ty::Fn { .. }) => {
                let t = self.type_closure(e, Some(want), &[]);
                if matches!(t, Ty::Fn { .. }) && t != *want && report {
                    let (w, f) = (want.display(self.interner), t.display(self.interner));
                    self.error(
                        s.span,
                        "E0401",
                        format!("type mismatch: `let` is annotated `{w}` but the closure is `{f}`"),
                    );
                }
                t
            }
            Some(Ty::Error | Ty::Unknown) => Ty::Error,
            Some(other) => {
                if report {
                    let o = other.display(self.interner);
                    self.error(
                        s.span,
                        "E0401",
                        format!(
                            "type mismatch: `let` is annotated `{o}` but the value is a closure"
                        ),
                    );
                }
                Ty::Error
            }
            None => self.type_closure(e, None, &[]),
        };
        self.types.insert(e.id, t.clone());
        self.bind(name, t.clone());
        if matches!(t, Ty::Fn { .. }) {
            if let Some(binds) = self.closure_binds.last_mut() {
                binds.insert(name, e.id);
            }
        }
        // Immutable like any initialized `let`: `f = ..` is `E0304`.
        let mut_at = crate::let_mut_position(s, pat, ann, Some(e));
        if let Some(set) = self.immutable_locals.last_mut() {
            set.insert(
                name,
                crate::LetDecl {
                    name: pat.span,
                    mut_at,
                },
            );
        }
    }

    /// `for name in start..end` has just bound its counter (in the innermost scope). Its type is
    /// the start's (`type_for_iter`), but `wukong_mir_build` stores it at the wider of the two
    /// bounds' integer types (`range_counter_type`), and the loop body sees that wider value. A
    /// closure capturing it must take it the same way — a copy of the wide value, or a pointer to
    /// the wide slot — so the width is recorded here for [`Self::note_local_use`]: the wider type,
    /// or [`CounterWidth::Generic`] when a bound's type names an enclosing generic (its width is
    /// then decided per instance). Returns whether a record was pushed (popped with the scope).
    pub(crate) fn enter_range_counter(&mut self, pat: &Pattern, iter: &ForIter) -> bool {
        let (
            PatKind::Ident(name),
            ForIter::Range {
                start,
                end: Some(end),
                ..
            },
        ) = (&pat.kind, iter)
        else {
            return false;
        };
        let (start_ty, end_ty) = (self.types.get(&start.id), self.types.get(&end.id));
        let width = if [start_ty, end_ty]
            .into_iter()
            .flatten()
            .any(|t| self.mentions_enclosing_generic(t))
        {
            CounterWidth::Generic
        } else {
            match (start_ty, end_ty) {
                (Some(Ty::Scalar(s)), Some(Ty::Scalar(e)))
                    if s.is_int() && e.is_int() && e.size() > s.size() =>
                {
                    CounterWidth::Wide(Ty::Scalar(*e))
                }
                _ => return false,
            }
        };
        let depth = self.scopes.len().saturating_sub(1);
        self.range_counters.push((depth, *name, width));
        true
    }

    /// The closure literal the local `name` is bound to, if it is a closure-bound local (resolved
    /// innermost scope first, so a shadowing binding of any other kind hides it).
    pub(crate) fn closure_bound(&self, name: Symbol) -> Option<NodeId> {
        let i = (0..self.scopes.len())
            .rev()
            .find(|&i| self.scopes[i].contains_key(&name))?;
        self.closure_binds.get(i)?.get(&name).copied()
    }

    /// A closure-bound local `name` used as the value expression `e`. Legal as a single-name
    /// argument of a function-typed parameter ([`Sema::fn_value_arg`]) — a call `f(..)` never
    /// reaches here (`type_fn_param_call`) — and, when the closure captures nothing, anywhere a
    /// function value is: it is the lifted function's address. Anywhere else a closure that
    /// captures would escape: `E0421`.
    pub(crate) fn closure_value_use(&mut self, e: &Expr, name: Symbol, span: Span, t: Ty) -> Ty {
        let Some(lit) = self.closure_bound(name) else {
            return t;
        };
        let captures_nothing = self
            .closures
            .get(&lit)
            .is_some_and(|info| info.captures.is_empty());
        if self.fn_value_arg == Some(e.id) || captures_nothing {
            self.closure_refs.insert(e.id, lit);
            self.note_value_use(name, span);
            return t;
        }
        if self.inst_subst.is_empty() {
            let nm = self.sym_str(name).to_string();
            self.diags.push(
                Diagnostic::error(format!(
                    "`{nm}` is bound to a closure, which can only be called or passed to a \
                     function-typed parameter"
                ))
                .with_code("E0421")
                .primary(span, "the closure would escape here")
                .note(format!(
                    "a closure is resolved at compile time: every call through `{nm}` becomes a \
                     direct call of the closure's code, so `{nm}` has no run-time value to \
                     store, return or compute with"
                ))
                .help(format!(
                    "call it (`{nm}(..)`), or pass it to a parameter of function type \
                     (`fn apply(f: fn(f32) -> f32, ..)`)"
                )),
            );
        }
        Ty::Error
    }

    /// A call `f(..)` through the closure-bound local `f`: record which closure it calls.
    pub(crate) fn note_closure_call(&mut self, name: Symbol, callee: &Expr) {
        if let Some(lit) = self.closure_bound(name) {
            self.closure_refs.insert(callee.id, lit);
        }
    }

    /// Check a closure literal passed where a value of function type `expected` is wanted (a
    /// parameter of a callee whose generics are `callee_generics`), and return its type
    /// `fn(P..) -> R` (or `Error` once something about it was reported).
    pub(crate) fn type_closure(
        &mut self,
        e: &Expr,
        expected: Option<&Ty>,
        callee_generics: &[Symbol],
    ) -> Ty {
        let ExprKind::Closure { params, ret, body } = &e.kind else {
            return Ty::Error;
        };
        if !self.in_fn_body {
            return self.misplaced_closure(e);
        }
        // An instantiation re-check of a generic caller neither re-reports nor re-records.
        let report = self.inst_subst.is_empty();
        // With no expected type (a `let`-bound closure), every parameter is annotated.
        let (want_params, want_ret): (&[Ty], Option<&Ty>) = match expected {
            Some(Ty::Fn { params, ret }) => (params, Some(ret)),
            _ => (&[], None),
        };
        // At any depth, a tensor's element type and dimensions included: the callee's own `T` or
        // `N` in a parameter type means nothing outside the callee.
        let generic_free = |t: &Ty| !names_generic(t, &|g| callee_generics.contains(&g));

        self.closure_ctx.push(ClosureCtx {
            base: self.scopes.len(),
            captures: Vec::new(),
            by_ref: Vec::new(),
            bound: Vec::new(),
            reported: Vec::new(),
            failed: false,
            returns: false,
        });

        if let (Some(expected), true) = (expected, want_params.len() != params.len()) {
            if report {
                let w = expected.display(self.interner);
                let (n, m) = (params.len(), want_params.len());
                self.diags.push(
                    Diagnostic::error(format!(
                        "type mismatch: expected a function `{w}`, found a closure taking {n} \
                         parameter{}",
                        if n == 1 { "" } else { "s" }
                    ))
                    .with_code("E0401")
                    .primary(
                        e.span,
                        format!(
                            "this closure takes {n} parameter{}, but `{w}` takes {m}",
                            if n == 1 { "" } else { "s" }
                        ),
                    ),
                );
            }
            self.closure_failed();
        }

        // Parameter types. The context is already pushed, so an annotation naming an enclosing
        // generic is `E0418` (`lower_type`).
        let mut ptys = Vec::with_capacity(params.len());
        // The unannotated parameters whose expected type names a callee generic.
        let mut uninferred: Vec<&wukong_ast::ClosureParam> = Vec::new();
        for (i, p) in params.iter().enumerate() {
            let t = match (&p.ty, want_params.get(i)) {
                (Some(te), _) => self.lower_type(te),
                // Bound by the call's other arguments to a type of the enclosing function's own
                // generics: the lifted function is not generic (`E0418`, as for an annotation).
                (None, Some(t)) if generic_free(t) && self.mentions_enclosing_generic(t) => {
                    if let Some(g) = self.enclosing_generic_in(t) {
                        self.note_generic_use(g, p.span);
                    }
                    Ty::Error
                }
                (None, Some(t)) if generic_free(t) => t.clone(),
                (None, Some(_)) => {
                    uninferred.push(p);
                    Ty::Error
                }
                // A `let`-bound closure has no function type to take it from.
                (None, None) if expected.is_none() => {
                    uninferred.push(p);
                    Ty::Error
                }
                // Past the expected arity, already reported above.
                (None, None) => Ty::Error,
            };
            ptys.push(t);
        }
        if let Some(first) = uninferred.first() {
            if report {
                let names: Vec<String> = uninferred
                    .iter()
                    .map(|p| self.sym_str(p.name.sym).to_string())
                    .collect();
                let shown = names
                    .iter()
                    .map(|n| format!("`{n}`"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let fix = names
                    .iter()
                    .map(|n| format!("{n}: i32"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let what = if names.len() == 1 {
                    "parameter"
                } else {
                    "parameters"
                };
                let note = match expected {
                    Some(w) => format!(
                        "the closure is passed as `{}`, whose parameter types are generic \
                         parameters of the callee that the call's other arguments do not fix",
                        w.display(self.interner)
                    ),
                    None => "a closure bound by `let` has no function type to take its parameter \
                             types from"
                        .to_string(),
                };
                self.diags.push(
                    Diagnostic::error(format!("cannot infer the type of closure {what} {shown}"))
                        .with_code("E0419")
                        .primary(first.span, "type annotation needed")
                        .note(note)
                        .help(format!("annotate the {what}, for example `|{fix}| ..`")),
                );
            }
            self.closure_failed();
        }
        // The expected return type applies only to a closure whose parameters are the expected
        // ones: `|x: i32| x` passed as `fn(f32) -> f32` is one mismatch (the call's `E0401`), not a
        // second one at its body.
        let params_as_expected = want_params.len() == params.len()
            && ptys
                .iter()
                .zip(want_params)
                .all(|(p, w)| !generic_free(w) || p == w || matches!(p, Ty::Error | Ty::Unknown));
        let known_ret = match ret {
            Some(r) => Some(self.lower_type(r)),
            None if params_as_expected => want_ret.filter(|r| generic_free(r)).cloned(),
            None => None,
        };

        // The body is checked in its own world: its `return` is the closure's, the enclosing
        // loops are out of reach, and `sync_block()` is not in a `@parallel` function.
        let saved_ret =
            std::mem::replace(&mut self.ret_ty, known_ret.clone().unwrap_or(Ty::Unknown));
        let saved_loops = std::mem::take(&mut self.loop_ctx);
        let saved_parallel = std::mem::replace(&mut self.parallel_fn, false);
        let saved_param_tys = self.param_tys.clone();

        self.push_scope();
        let mut seen: HashSet<Symbol> = HashSet::default();
        for (p, t) in params.iter().zip(&ptys) {
            if !seen.insert(p.name.sym) {
                let nm = self.sym_str(p.name.sym).to_string();
                self.error(
                    p.name.span,
                    "E0300",
                    format!("duplicate closure parameter name `{nm}`"),
                );
            }
            // Immutable, like a non-`mut` function parameter.
            if let Some(set) = self.immutable_params.last_mut() {
                set.insert(p.name.sym, p.name.span);
            }
            self.param_tys.insert(p.name.sym, t.clone());
            self.bind(p.name.sym, t.clone());
        }
        let body_ty = self.type_expr(body);
        if let Some(rt) = &known_ret {
            // The implicit return is checked like a function body's tail (an explicit `return` is
            // checked at its statement, against `ret_ty`), and a closure that promises a value
            // must produce one on every path, exactly like a function (`check_fn_pass`).
            let returns_value = match &body.kind {
                ExprKind::Block(b) => {
                    if let Some(tail) = &b.tail {
                        self.check_return_shape(&body_ty, tail, tail.span);
                    }
                    !matches!(body_ty, Ty::Unit) || crate::block_diverges(b)
                }
                _ => {
                    self.check_return_shape(&body_ty, body, body.span);
                    !matches!(body_ty, Ty::Unit)
                }
            };
            if !matches!(rt, Ty::Unit | Ty::Unknown | Ty::Error) && !returns_value {
                self.error(
                    body.span,
                    "E0434",
                    "not all control-flow paths return a value (this closure can fall off its \
                     end)",
                );
            }
        }
        self.pop_scope();
        self.param_tys = saved_param_tys;
        self.parallel_fn = saved_parallel;
        self.loop_ctx = saved_loops;
        self.ret_ty = saved_ret;

        let ctx = self
            .closure_ctx
            .pop()
            .expect("closure_ctx push/pop balanced");
        if known_ret.is_none()
            && (params_as_expected || expected.is_none())
            && ctx.returns
            && !ctx.failed
        {
            if report {
                let note = match expected {
                    Some(w) => format!(
                        "the closure is passed as `{}`, whose return type is a generic \
                         parameter of the callee",
                        w.display(self.interner)
                    ),
                    None => "a closure bound by `let` has no function type to take its return \
                             type from"
                        .to_string(),
                };
                self.diags.push(
                    Diagnostic::error(
                        "cannot infer the return type of a closure that uses `return`",
                    )
                    .with_code("E0419")
                    .primary(e.span, "return type annotation needed")
                    .note(note)
                    .help("annotate it: `|..| -> i32 { .. }`"),
                );
            }
            return Ty::Error;
        }
        if ctx.failed {
            return Ty::Error;
        }
        let ret = known_ret.unwrap_or(body_ty);
        if report {
            self.closures.insert(
                e.id,
                crate::ClosureInfo {
                    params: ptys.clone(),
                    ret: ret.clone(),
                    captures: ctx.captures,
                    by_ref: ctx.by_ref,
                    bound: ctx.bound,
                },
            );
        }
        Ty::Fn {
            params: ptys,
            ret: Box::new(ret),
        }
    }

    /// `E0411`: a closure outside a function body (a `const` initializer, a `where` predicate, a
    /// `static_assert`).
    pub(crate) fn misplaced_closure(&mut self, e: &Expr) -> Ty {
        self.diags.push(
            Diagnostic::error("a closure can only be written inside a function body")
                .with_code("E0411")
                .primary(e.span, "a closure cannot be used here")
                .help(
                    "declare a named function, or write the closure in the function that uses it",
                ),
        );
        Ty::Error
    }

    /// `e` typed where a value of type `want` is expected: a closure literal is checked against
    /// `want` when that is a function type ([`Self::type_closure_value`]); anything else is typed
    /// as usual.
    pub(crate) fn type_expr_expecting(&mut self, e: &Expr, want: &Ty) -> Ty {
        match (&e.kind, want) {
            (ExprKind::Closure { .. }, Ty::Fn { .. }) => {
                let t = self.type_closure_value(e, Some(want));
                self.types.insert(e.id, t.clone());
                t
            }
            _ => self.type_expr(e),
        }
    }

    /// A closure literal used as a **run-time function value** — the initializer of a `let mut`,
    /// a `return`ed value, an assigned value, a field or an element — rather than an argument of
    /// a function-typed parameter or a `let`-bound closure. Its value is the lifted function's
    /// address, so it may capture nothing: a closure that captures is `E0427` (there is nowhere
    /// for the captured values to live). `expected`, when known, types its parameters.
    pub(crate) fn type_closure_value(&mut self, e: &Expr, expected: Option<&Ty>) -> Ty {
        if !self.in_fn_body {
            return self.misplaced_closure(e);
        }
        let t = self.type_closure(e, expected, &[]);
        let captures: Vec<Symbol> = self
            .closures
            .get(&e.id)
            .map(|info| info.captures.iter().map(|(n, _)| *n).collect())
            .unwrap_or_default();
        if captures.is_empty() || !matches!(t, Ty::Fn { .. }) {
            return t;
        }
        if self.inst_subst.is_empty() {
            self.report_capturing_value(e.span, &captures);
        }
        Ty::Error
    }

    /// `E0427`: a closure that captures `captures`, used as a run-time function value at `span`.
    fn report_capturing_value(&mut self, span: Span, captures: &[Symbol]) {
        let names: Vec<String> = captures
            .iter()
            .map(|n| format!("`{}`", self.sym_str(*n)))
            .collect();
        let (what, list) = if names.len() == 1 {
            ("variable", names[0].clone())
        } else {
            ("variables", names.join(", "))
        };
        self.diags.push(
            Diagnostic::error(
                "a closure that captures variables cannot be a run-time function value",
            )
            .with_code("E0427")
            .primary(span, format!("this closure captures the {what} {list}"))
            .note(
                "a function value is only the address of code; the values a closure captures \
                 have nowhere to live in it",
            )
            .help(
                "pass the captured values as ordinary parameters, or pass the closure directly to \
                 a function-typed parameter of the function that calls it",
            ),
        );
    }

    /// `let mut f = |..| ..;`: a function variable initialized with the closure's address, which
    /// a later `f = ..` may replace. The closure must capture nothing (`E0422` names the `mut`,
    /// as for a capturing closure bound by `let mut` before function values existed).
    fn type_closure_mut_let(
        &mut self,
        s: &Stmt,
        pat: &Pattern,
        name: Symbol,
        ann: Option<&TypeExpr>,
        e: &Expr,
    ) {
        let report = self.inst_subst.is_empty();
        let expected = ann.map(|t| self.lower_type(t));
        let want = expected.as_ref().filter(|t| matches!(t, Ty::Fn { .. }));
        let mut t = self.type_closure(e, want, &[]);
        let captures = self
            .closures
            .get(&e.id)
            .is_some_and(|info| !info.captures.is_empty());
        if captures && matches!(t, Ty::Fn { .. }) {
            if report {
                self.diags.push(
                    Diagnostic::error("a closure that captures cannot be bound by a `mut` binding")
                        .with_code("E0422")
                        .primary(pat.span, "the closure is bound here")
                        .note(
                            "a `mut` binding holds a run-time function value, which is only the \
                             address of code; the values this closure captures have nowhere to \
                             live in it",
                        )
                        .help("bind it with a plain `let name = |..| ..;`"),
                );
            }
            t = Ty::Error;
        }
        match (&expected, &t) {
            (Some(want @ Ty::Fn { .. }), Ty::Fn { .. }) if *want != t && report => {
                let (w, f) = (want.display(self.interner), t.display(self.interner));
                self.error(
                    s.span,
                    "E0401",
                    format!("type mismatch: `let` is annotated `{w}` but the closure is `{f}`"),
                );
            }
            (Some(other), Ty::Fn { .. }) if !matches!(other, Ty::Fn { .. }) && report => {
                let o = other.display(self.interner);
                self.error(
                    s.span,
                    "E0401",
                    format!("type mismatch: `let` is annotated `{o}` but the value is a closure"),
                );
                t = Ty::Error;
            }
            _ => {}
        }
        self.types.insert(e.id, t.clone());
        self.check_fn_binding_name(name, pat.span, &t);
        self.bind(name, t);
    }

    /// A use of the name `name` (of type `t`) as the value `e`, at `span`. A local holding a
    /// function value is recorded for `wukong_mir_build` ([`Self::note_fn_local_read`]); a
    /// top-level function is checked to have one address ([`Self::reject_fn_item_value`]).
    /// Returns the use's type.
    pub(crate) fn fn_value_use(&mut self, e: &Expr, name: Symbol, span: Span, t: Ty) -> Ty {
        if !matches!(t, Ty::Fn { .. }) {
            return t;
        }
        if self.lookup_local(name).is_some() {
            self.note_fn_local_read(e, name);
            return t;
        }
        if !self.in_fn_body && self.reject_fn_value_outside_body(name, span) {
            return Ty::Error;
        }
        if self.reject_fn_item_value(name, span) {
            return Ty::Error;
        }
        t
    }

    /// `E0411`: the top-level function `name` used as a value outside a function body — in a
    /// `const` initializer, which is not one value but an expression lowered again at every use.
    /// A function value made there would escape the whole-program analyses that must see every
    /// function used as a value (which buffers a call through a pointer may alias, which instance
    /// of a function with function-typed parameters has an address) and would resolve its name in
    /// the scope of each use. Returns whether it was refused.
    fn reject_fn_value_outside_body(&mut self, name: Symbol, span: Span) -> bool {
        if !matches!(
            self.defs.lookup(name).map(|d| &d.kind),
            Some(DefKind::Fn(_))
        ) {
            return false;
        }
        if self.inst_subst.is_empty() {
            let nm = self.sym_str(name).to_string();
            self.diags.push(
                Diagnostic::error(format!(
                    "the function `{nm}` cannot be used as a value outside a function body"
                ))
                .with_code("E0411")
                .primary(span, "a function value cannot be made here")
                .note(
                    "a `const` initializer is evaluated again at each use rather than stored once, \
                     so a `const` cannot hold a function value",
                )
                .help(format!(
                    "bind it in the function that uses it (`let f = {nm};`), or write a function \
                     that returns it"
                )),
            );
        }
        true
    }

    /// The single-name path `e` reads the local `name` of function type, as a value or as a
    /// callee: recorded in [`crate::SemaResult::fn_value_locals`], and in
    /// [`crate::SemaResult::fn_value_lets`] too unless the local is a parameter.
    pub(crate) fn note_fn_local_read(&mut self, e: &Expr, name: Symbol) {
        self.fn_value_locals.insert(e.id);
        if !self.is_param_binding(name) {
            self.fn_value_lets.insert(e.id);
        }
    }

    /// Whether the local `name` resolves (innermost scope first) to a parameter: of the function
    /// being checked (bound in the outermost scope, `check_fn_pass`) or of a closure being checked
    /// (bound in the scope at its context's `base`, [`Self::type_closure`]). Every other local —
    /// a `let`, a pattern or a `for` binding — lives in a scope a block or an arm pushed.
    pub(crate) fn is_param_binding(&self, name: Symbol) -> bool {
        match self.lookup_local_depth(name) {
            Some((depth, _)) => depth == 0 || self.closure_ctx.iter().any(|c| c.base == depth),
            None => false,
        }
    }

    /// A top-level function `name` used as a run-time function value at `span`: a generic one has
    /// no single address, and one without a body (`extern`) is not Wukong code (`E0429`).
    /// Returns whether it was refused.
    pub(crate) fn reject_fn_item_value(&mut self, name: Symbol, span: Span) -> bool {
        let (generic, extents) = match self.defs.lookup(name).map(|d| &d.kind) {
            Some(DefKind::Fn(sig)) => (
                !sig.generics.is_empty(),
                sig.params.iter().any(has_runtime_extent),
            ),
            _ => return false,
        };
        let bodyless = self.bodyless_fns.contains(&name);
        if !generic && !extents && !bodyless {
            return false;
        }
        if self.inst_subst.is_empty() {
            let nm = self.sym_str(name).to_string();
            let (label, note) = if generic {
                (
                    format!("`{nm}` is generic"),
                    "a generic function is compiled once per instantiation, chosen at each call \
                     from its arguments, so it has no single address",
                )
            } else if extents {
                (
                    format!("`{nm}` takes a tensor with a run-time extent (`?`)"),
                    "each run-time extent is passed as a hidden argument read off the tensor at a \
                     direct call, which a call through a function value cannot pass",
                )
            } else {
                (
                    format!("`{nm}` is declared without a body"),
                    "an `extern` function is C code, which a run-time function value cannot \
                     reach on every backend",
                )
            };
            self.diags.push(
                Diagnostic::error(format!("`{nm}` cannot be a run-time function value"))
                    .with_code("E0429")
                    .primary(span, label)
                    .note(note)
                    .help(format!(
                        "wrap it in a non-generic function of the concrete type and use that \
                         (`fn {nm}_i32(x: i32) -> i32 {{ return {nm}(x); }}`)"
                    )),
            );
        }
        true
    }

    /// `E0428`: a comparison whose operand is a function value. Returns whether it was refused.
    pub(crate) fn reject_fn_value_comparison(
        &mut self,
        op: wukong_ast::BinOp,
        l: &Ty,
        r: &Ty,
        span: Span,
    ) -> bool {
        use wukong_ast::BinOp::*;
        if !matches!(op, Eq | Ne | Lt | Le | Gt | Ge)
            || ![l, r].into_iter().any(|t| matches!(t, Ty::Fn { .. }))
        {
            return false;
        }
        let shown = [l, r]
            .into_iter()
            .find(|t| matches!(t, Ty::Fn { .. }))
            .map(|t| t.display(self.interner))
            .unwrap_or_default();
        self.diags.push(
            Diagnostic::error(format!(
                "function values cannot be compared with `{}`",
                op.glyph()
            ))
            .with_code("E0428")
            .primary(
                span,
                format!("an operand is a function value of type `{shown}`"),
            )
            .note(
                "which functions share an address is the compiler's choice: the optimizer may \
                 merge or clone them, and the interpreter numbers what native code addresses",
            )
            .help("keep a tag (an enum or an integer) beside the function and compare that"),
        );
        true
    }

    /// The closure being checked has an error of its own: its type is `Error` (and so is every
    /// enclosing closure's, whose body contains it).
    fn closure_failed(&mut self) {
        for c in &mut self.closure_ctx {
            c.failed = true;
        }
    }

    /// A `return` statement: inside a closure, it returns from the closure.
    pub(crate) fn note_closure_return(&mut self) {
        if let Some(c) = self.closure_ctx.last_mut() {
            c.returns = true;
        }
    }

    /// The single-name value `name` was used at `span`. Inside a closure, a local of the enclosing
    /// code is a capture ([`Self::note_local_use`]), and an enclosing generic is `E0418`.
    pub(crate) fn note_value_use(&mut self, name: Symbol, span: Span) {
        if self.closure_ctx.is_empty() {
            return;
        }
        if self.lookup_local(name).is_some() {
            self.note_local_use(name, span);
        } else if self.generics.contains(&name) {
            self.note_generic_use(name, span);
        }
    }

    /// The local `name` was read at `span`: for every closure it is declared outside of, record
    /// it as a capture (once, in order of first use) — or report `E0418` when it cannot be copied.
    pub(crate) fn note_local_use(&mut self, name: Symbol, span: Span) {
        if self.closure_ctx.is_empty() {
            return;
        }
        let Some((depth, ty)) = self.lookup_local_depth(name) else {
            return;
        };
        let crossed: Vec<usize> = (0..self.closure_ctx.len())
            .filter(|&i| depth < self.closure_ctx[i].base)
            .collect();
        // A name of the closure's own, or one whose type already failed to check.
        if crossed.is_empty() || matches!(ty, Ty::Error) {
            return;
        }
        // A local bound to a closure: the lifted function receives that closure's captures.
        if let Some(lit) = self
            .closure_binds
            .get(depth)
            .and_then(|b| b.get(&name))
            .copied()
        {
            for i in crossed {
                let c = &mut self.closure_ctx[i];
                if !c.captures.iter().any(|(n, _)| *n == name) {
                    c.captures.push((name, ty.clone()));
                    c.bound.push((name, lit));
                }
            }
            return;
        }
        // A range counter is captured at the width it is stored at, as the loop body sees it.
        let counter = self
            .range_counters
            .iter()
            .rev()
            .find(|(d, n, _)| *d == depth && *n == name)
            .map(|(_, _, w)| w.clone());
        let (ty, why) = match counter {
            Some(CounterWidth::Wide(wide)) => (wide, None),
            Some(CounterWidth::Generic) => (ty, Some(NoCopy::CounterWidth)),
            None => (ty, None),
        };
        // A local holding a run-time function value is a code address, copied like a pointer.
        // Only the enclosing function's own function-typed parameter (bound at compile time, at
        // the function's outermost scope) has no value to copy.
        let why = match why {
            None if matches!(ty, Ty::Fn { .. })
                && !(depth == 0 && self.param_tys.contains_key(&name))
                && !self.mentions_enclosing_generic(&ty) =>
            {
                for i in crossed {
                    let c = &mut self.closure_ctx[i];
                    if !c.captures.iter().any(|(n, _)| *n == name) {
                        c.captures.push((name, ty.clone()));
                    }
                }
                return;
            }
            why => why,
        };
        let why = match why.map_or_else(|| self.capture_kind(&ty), Err) {
            Ok(()) => {
                for i in crossed {
                    let c = &mut self.closure_ctx[i];
                    if !c.captures.iter().any(|(n, _)| *n == name) {
                        c.captures.push((name, ty.clone()));
                    }
                }
                return;
            }
            Err(why) => why,
        };
        let already = crossed
            .iter()
            .any(|&i| self.closure_ctx[i].reported.contains(&name));
        for &i in &crossed {
            self.closure_ctx[i].reported.push(name);
        }
        self.closure_failed();
        if already || !self.inst_subst.is_empty() {
            return;
        }
        let nm = self.sym_str(name).to_string();
        let tys = ty.display(self.interner);
        let d = match why {
            NoCopy::Value => Diagnostic::error(format!(
                "closure captures `{nm}`, a value of type `{tys}` it cannot copy"
            ))
            .with_code("E0418")
            .primary(span, "captured here")
            .note(
                "a closure copies what it captures: a scalar, an enum, a pointer or reference, a \
                 fixed-size array, a tuple, a struct, a slice or a `str` (its view); a tensor or \
                 SIMD vector holds a shape or lanes a copy cannot carry",
            )
            .help(format!(
                "capture a pointer instead (`let p{nm} = &{nm};` before the call, then \
                 `(*p{nm})[i]` in the closure), or pass the value to the function as a \
                 parameter"
            )),
            NoCopy::FnParam => Diagnostic::error(format!(
                "closure uses `{nm}`, a function-typed parameter it cannot capture"
            ))
            .with_code("E0418")
            .primary(span, "used here")
            .note("a function-typed parameter is bound at compile time and has no value to copy")
            .help(
                "pass the function to the callee as a function-typed parameter of its own, or \
                 call it before the closure and capture the result",
            ),
            NoCopy::CounterWidth => Diagnostic::error(format!(
                "closure captures the loop counter `{nm}`, whose width depends on a generic \
                 parameter of the enclosing function"
            ))
            .with_code("E0418")
            .primary(span, "captured here")
            .note(
                "a range counter is stored at the wider of its bounds' integer types, and a bound \
                 here is generic, while a closure is lifted to one non-generic function",
            )
            .help(format!(
                "copy it to a local first (`let k = {nm};`) and capture that, or pass it to the \
                 closure as an argument"
            )),
            NoCopy::Generic => Diagnostic::error(format!(
                "closure captures `{nm}`, whose type `{tys}` names a generic parameter of the \
                 enclosing function"
            ))
            .with_code("E0418")
            .primary(span, "captured here")
            .note(
                "a closure is lifted to a non-generic function, so it cannot depend on the \
                 enclosing function's generic parameters",
            )
            .help("pass the value to the function as a parameter instead of capturing it"),
        };
        self.diags.push(d);
    }

    /// An enclosing generic parameter named inside a closure, as a value (`N`) or in a type
    /// (`x as T`, `|x: T|`): `E0418`, once per closure and name.
    pub(crate) fn note_generic_use(&mut self, name: Symbol, span: Span) {
        if self.closure_ctx.is_empty() {
            return;
        }
        let already = self.closure_ctx.iter().any(|c| c.reported.contains(&name));
        for c in &mut self.closure_ctx {
            c.reported.push(name);
        }
        self.closure_failed();
        if already || !self.inst_subst.is_empty() {
            return;
        }
        let nm = self.sym_str(name).to_string();
        let local = nm.to_lowercase();
        self.diags.push(
            Diagnostic::error(format!(
                "closure uses `{nm}`, a generic parameter of the enclosing function"
            ))
            .with_code("E0418")
            .primary(span, "used here")
            .note(
                "a closure is lifted to a non-generic function, so it cannot depend on the \
                 enclosing function's generic parameters",
            )
            .help(format!(
                "bind it to a local before the call (`let {local} = {nm};`) and use that in the \
                 closure"
            )),
        );
    }

    /// A write to the place `target` (`c = ..`, `c op= ..`, `&mut c`, `a[i] = ..`, `s.f = ..`,
    /// `atomic_add(a[i], ..)`, a `mut self` method on `v`, an aggregate passed to a `mut` parameter
    /// ([`Self::note_mut_arg_writes`]), a builtin's output operand such as `relu(x, out)`): when its
    /// root is a variable a closure being checked captures, that closure captures it BY REFERENCE —
    /// it receives a pointer to the variable, so the write reaches the enclosing function (and every
    /// closure in between captures it the same way). A projection through a pointer or reference
    /// (`p.x = ..`, `(*p)[i] = ..`) writes a pointee, not the captured variable, which stays a copy.
    pub(crate) fn note_capture_write(&mut self, target: &Expr) {
        if self.closure_ctx.is_empty() {
            return;
        }
        let Some(name) = self.written_root(target) else {
            return;
        };
        let Some((depth, _)) = self.lookup_local_depth(name) else {
            return;
        };
        for c in &mut self.closure_ctx {
            if depth < c.base && !c.by_ref.contains(&name) {
                c.by_ref.push(name);
            }
        }
    }

    /// The arguments `args` of a call of the function `callee`, bound to its parameters from
    /// position `offset` on (a method call's receiver takes position 0). An aggregate is passed by
    /// reference and the callee's `mut` is what lets it write one, so an aggregate argument of a
    /// `mut` parameter is written in place: `relu6(x, out)` writes `out` exactly as `out[i] = ..`
    /// would, and a closure passing a variable it captures there shares it.
    pub(crate) fn note_mut_arg_writes(&mut self, callee: Symbol, args: &[Expr], offset: usize) {
        if self.closure_ctx.is_empty() {
            return;
        }
        let Some(muts) = self.methods.mut_params.get(&callee).cloned() else {
            return;
        };
        for k in muts {
            let Some(a) = k.checked_sub(offset).and_then(|i| args.get(i)) else {
                continue;
            };
            // A scalar or a pointer is copied into the callee; a slice is a view whose elements a
            // write reaches without changing the variable.
            let by_reference = self
                .types
                .get(&a.id)
                .is_some_and(|t| self.is_aggregate_ty(t) && !matches!(t, Ty::Slice(_) | Ty::Str));
            if by_reference {
                self.note_capture_write(a);
            }
        }
    }

    /// The arguments `args` of a call through a function value — a function-typed parameter or
    /// `let`, a field, an element, a call's result. A function type does not say which parameters
    /// are `mut`, and an aggregate is passed by reference, so the function the value holds may
    /// write any aggregate argument in place: a closure passing a variable it captures there
    /// shares it, as it would at a direct call of a function whose parameter is `mut`
    /// ([`Self::note_mut_arg_writes`]). Conservative: a callee that only reads sees the variable
    /// itself rather than a copy, the same value unless the enclosing code writes it between
    /// creating a `let`-bound closure and calling it.
    pub(crate) fn note_value_call_writes(&mut self, args: &[Expr]) {
        if self.closure_ctx.is_empty() {
            return;
        }
        for a in args {
            let by_reference = self
                .types
                .get(&a.id)
                .is_some_and(|t| self.is_aggregate_ty(t) && !matches!(t, Ty::Slice(_) | Ty::Str));
            if by_reference {
                self.note_capture_write(a);
            }
        }
    }

    /// The variable whose own storage a write to `e` changes, if `e` is a name or a field / element
    /// projection of one that does not pass through a pointer or reference, or a tensor view of
    /// one (`out as Tensor[f32, 2, 3]`, a zero-copy reshape of its storage).
    fn written_root(&self, e: &Expr) -> Option<Symbol> {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() => Some(p.first().sym),
            ExprKind::Field { base, .. }
            | ExprKind::TupleField { base, .. }
            | ExprKind::Index { base, .. } => match self.types.get(&base.id) {
                Some(Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Slice(_)) => None,
                _ => self.written_root(base),
            },
            ExprKind::Cast { expr, .. }
                if matches!(self.types.get(&e.id), Some(Ty::Tensor { .. })) =>
            {
                self.written_root(expr)
            }
            _ => None,
        }
    }

    /// A tensor local bound to `init` (`let t = out as Tensor[f32, 2, 2];`, or the rows of a `for`
    /// over one) is a handle on the storage `init` names, so a write through it is a write of that
    /// variable. A closure binding such a view of a variable it captures therefore shares the
    /// variable (by reference), whether or not it then writes through the view.
    pub(crate) fn note_capture_view(&mut self, ty: &Ty, init: Option<&Expr>) {
        if let (Ty::Tensor { .. }, Some(e)) = (ty, init) {
            self.note_capture_write(e);
        }
    }

    /// Whether the assignment target `target` is a bare closure-bound local (`f = ..`).
    pub(crate) fn closure_bound_place(&self, target: &Expr) -> bool {
        matches!(&target.kind, ExprKind::Path(p) if p.is_single() && self.closure_bound(p.first().sym).is_some())
    }

    /// `f = ..` for a closure-bound local `f`: one `E0422` (the value is still checked).
    pub(crate) fn reject_closure_assign(&mut self, target: &Expr, value: &Expr) {
        match (&value.kind, self.types.get(&target.id).cloned()) {
            (ExprKind::Closure { .. }, _) => {
                let want = match &target.kind {
                    ExprKind::Path(p) => self.lookup_local(p.first().sym),
                    _ => None,
                };
                let t = self.type_closure(value, want.as_ref(), &[]);
                self.types.insert(value.id, t);
            }
            _ => {
                self.type_expr(value);
            }
        }
        if !self.inst_subst.is_empty() {
            return;
        }
        let nm = match &target.kind {
            ExprKind::Path(p) => self.sym_str(p.first().sym).to_string(),
            _ => String::new(),
        };
        self.diags.push(
            Diagnostic::error(format!(
                "cannot assign to `{nm}`, which is bound to a closure"
            ))
            .with_code("E0422")
            .primary(target.span, "a closure-bound local cannot be reassigned")
            .note(
                "calls through a closure-bound local are resolved at compile time, so the \
                     local must name the one closure it is initialized with",
            )
            .help(format!(
                "bind the other closure to a new name (`let {nm}2 = |..| ..;`)"
            )),
        );
    }

    /// Whether a local of type `ty` can be captured by value.
    fn capture_kind(&self, ty: &Ty) -> Result<(), NoCopy> {
        match ty {
            Ty::Scalar(_) => Ok(()),
            Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } => {
                if self.mentions_enclosing_generic(pointee) {
                    Err(NoCopy::Generic)
                } else if has_runtime_extent(pointee) {
                    Err(NoCopy::Value)
                } else {
                    Ok(())
                }
            }
            Ty::Named(n) if self.generics.contains(n) => Err(NoCopy::Generic),
            // A C-style enum is its discriminant; a struct or a data-carrying enum is copied whole.
            Ty::Named(n) => match self.defs.lookup(*n).map(|d| &d.kind) {
                Some(DefKind::Enum(_) | DefKind::Struct(_)) => Ok(()),
                _ => Err(NoCopy::Value),
            },
            Ty::Fn { .. } => Err(NoCopy::FnParam),
            _ if self.mentions_enclosing_generic(ty) => Err(NoCopy::Generic),
            // A slice or `str` is its 16-byte fat pointer `(data, len)`: a copy is another view of
            // the same elements, exactly what passing it to a parameter makes.
            Ty::Slice(_) | Ty::Str => Ok(()),
            // A fixed-size array or a tuple is copied whole, into a buffer the lifted function
            // receives by pointer. A `shared` (GPU block-shared) array is not a value to copy, and
            // an element whose extent is only known at run time cannot be laid out.
            Ty::Array { shared: false, .. } | Ty::Tuple(_) if !has_runtime_extent(ty) => Ok(()),
            _ => Err(NoCopy::Value),
        }
    }

    /// Does `t` name a generic parameter of the enclosing function — as a type, a tensor element
    /// or a tensor dimension?
    fn mentions_enclosing_generic(&self, t: &Ty) -> bool {
        names_generic(t, &|g| self.generics.contains(&g))
    }

    /// One generic parameter of the enclosing function that `t` names, for its diagnostic.
    fn enclosing_generic_in(&self, t: &Ty) -> Option<Symbol> {
        let mut gs: Vec<Symbol> = self.generics.iter().copied().collect();
        gs.sort_by_key(|g| g.0);
        gs.into_iter().find(|g| names_generic(t, &|x| x == *g))
    }
}

/// Does `t` name a generic `is_generic` accepts — as a type, a tensor's element type or one of its
/// dimensions, at any depth? Exhaustive over `Ty`, so a new type form must decide here.
fn names_generic(t: &Ty, is_generic: &dyn Fn(Symbol) -> bool) -> bool {
    match t {
        Ty::Named(n) => is_generic(*n),
        Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } | Ty::Slice(pointee) => {
            names_generic(pointee, is_generic)
        }
        Ty::Array { elem, .. } => names_generic(elem, is_generic),
        Ty::Tuple(ts) => ts.iter().any(|x| names_generic(x, is_generic)),
        Ty::Fn { params, ret } => {
            params.iter().any(|x| names_generic(x, is_generic)) || names_generic(ret, is_generic)
        }
        Ty::Tensor {
            shape, elem_var, ..
        } => {
            elem_var.is_some_and(is_generic)
                || shape.0.iter().any(|d| match d {
                    Dim::Var(v) => is_generic(*v),
                    Dim::Expr(p) => p.vars().iter().any(|v| is_generic(*v)),
                    Dim::Const(_) | Dim::Dynamic => false,
                })
        }
        Ty::Scalar(_) | Ty::Str | Ty::Unit | Ty::Vector { .. } | Ty::Unknown | Ty::Error => false,
    }
}

/// Bind each dimension generic of a callee (`is_generic`) that `param` names to the extent `arg`
/// gives it, structurally — a tensor axis against a tensor axis, a rank-1 tensor against a
/// fixed-size array's length, through pointers, elements, tuple fields and function types. The
/// first binding wins (as in the call's own inference); a run-time `?` extent binds nothing.
pub(crate) fn bind_dims(
    param: &Ty,
    arg: &Ty,
    is_generic: &dyn Fn(Symbol) -> bool,
    dims: &mut HashMap<Symbol, Dim>,
) {
    match (param, arg) {
        (Ty::Tensor { shape: ps, .. }, Ty::Tensor { shape: as_, .. })
            if ps.0.len() == as_.0.len() =>
        {
            for (p, a) in ps.0.iter().zip(&as_.0) {
                if let Dim::Var(v) = p {
                    if is_generic(*v) && !matches!(a, Dim::Dynamic) {
                        dims.entry(*v).or_insert_with(|| a.clone());
                    }
                }
            }
        }
        (Ty::Tensor { shape: ps, .. }, Ty::Array { len, .. }) if ps.0.len() == 1 => {
            if let Dim::Var(v) = &ps.0[0] {
                if is_generic(*v) {
                    dims.entry(*v).or_insert(Dim::Const(*len));
                }
            }
        }
        (Ty::Ptr { pointee: p, .. }, Ty::Ptr { pointee: a, .. })
        | (Ty::Ref { pointee: p, .. }, Ty::Ref { pointee: a, .. })
        | (Ty::Slice(p), Ty::Slice(a)) => bind_dims(p, a, is_generic, dims),
        (Ty::Array { elem: p, .. }, Ty::Array { elem: a, .. }) => bind_dims(p, a, is_generic, dims),
        (Ty::Tuple(ps), Ty::Tuple(as_)) => {
            for (p, a) in ps.iter().zip(as_) {
                bind_dims(p, a, is_generic, dims);
            }
        }
        (
            Ty::Fn {
                params: ps,
                ret: pr,
            },
            Ty::Fn {
                params: as_,
                ret: ar,
            },
        ) => {
            for (p, a) in ps.iter().zip(as_) {
                bind_dims(p, a, is_generic, dims);
            }
            bind_dims(pr, ar, is_generic, dims);
        }
        _ => {}
    }
}

/// Does `t` contain a tensor whose extent is only known at run time (a `?` axis)? Its extents
/// travel as hidden parameters of the function that received it, which the lifted closure does not
/// have, so a pointer to one cannot be captured.
fn has_runtime_extent(t: &Ty) -> bool {
    match t {
        Ty::Tensor { shape, .. } => shape.0.iter().any(|d| !matches!(d, Dim::Const(_))),
        Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } | Ty::Slice(pointee) => {
            has_runtime_extent(pointee)
        }
        Ty::Array { elem, .. } => has_runtime_extent(elem),
        Ty::Tuple(ts) => ts.iter().any(has_runtime_extent),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::ClosureInfo;
    use wukong_ast::{ExprKind, ItemKind, Module, StmtKind};
    use wukong_diag::Diagnostic;
    use wukong_span::{Interner, SourceId};
    use wukong_types::{Scalar, Ty};

    const APPLY: &str = "fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } ";

    /// Check `src` (which must parse), returning the module, the result and the diagnostics.
    fn check(src: &str) -> (Module, crate::SemaResult, Vec<Diagnostic>, Interner) {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (res, diags) = crate::check(&module, &interner);
        (module, res, diags, interner)
    }

    /// The codes of the errors `src` checks to.
    fn error_codes(src: &str) -> Vec<&'static str> {
        check(src)
            .2
            .into_iter()
            .filter(|d| d.is_error())
            .filter_map(|d| d.code)
            .collect()
    }

    /// The closure infos of `src` (which must check clean), with capture names resolved, in the
    /// order the parser finished the literals: an inner literal before the one containing it.
    fn infos(src: &str) -> Vec<(Vec<String>, ClosureInfo)> {
        let (_m, res, diags, interner) = check(src);
        assert!(
            diags.iter().all(|d| !d.is_error()),
            "unexpected errors: {diags:?}"
        );
        let mut all: Vec<_> = res.closures.iter().collect();
        all.sort_by_key(|(id, _)| id.0);
        all.into_iter()
            .map(|(_, info)| {
                let names = info
                    .captures
                    .iter()
                    .map(|(s, _)| interner.resolve(*s).to_string())
                    .collect();
                (names, info.clone())
            })
            .collect()
    }

    #[test]
    fn captures_are_recorded_in_order_of_first_use() {
        let src = format!(
            "{APPLY} fn main() -> i32 {{ let a = 1.0; let b = 2.0; \
             return apply(|x| x * b + a, 3.0) as i32; }}"
        );
        let got = infos(&src);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, vec!["b", "a"]);
        let info = &got[0].1;
        assert_eq!(info.params, vec![Ty::Scalar(Scalar::F32)]);
        assert_eq!(info.ret, Ty::Scalar(Scalar::F32));
        assert_eq!(info.captures[0].1, Ty::Scalar(Scalar::F32));
    }

    #[test]
    fn a_local_of_the_body_shadowing_an_outer_name_is_not_a_capture() {
        let src = format!(
            "{APPLY} fn main() -> i32 {{ let a = 1.0; \
             return apply(|x| {{ let a = 5.0; x + a }}, 3.0) as i32; }}"
        );
        let got = infos(&src);
        assert_eq!(got[0].0, Vec::<String>::new());
        // ...while the read before the shadowing `let` is one.
        let src = format!(
            "{APPLY} fn main() -> i32 {{ let a = 1.0; \
             return apply(|x| {{ let y = a; let a = 5.0; x + a + y }}, 3.0) as i32; }}"
        );
        assert_eq!(infos(&src)[0].0, vec!["a"]);
    }

    #[test]
    fn a_nested_closure_capture_is_captured_by_the_outer_closure_too() {
        let src = format!(
            "{APPLY} fn main() -> i32 {{ let k = 3.0; let s = 2.5; \
             return apply(|x| apply(|y| y * k, x) + s, 2.0) as i32; }}"
        );
        let got = infos(&src);
        assert_eq!(got.len(), 2);
        // The inner literal captures `k`; the outer one captured it first, through the inner.
        assert_eq!(got[0].0, vec!["k"]);
        assert_eq!(got[1].0, vec!["k", "s"]);
        // An outer closure's parameter is a capture of the inner closure only.
        let src = format!(
            "{APPLY} fn main() -> i32 {{ \
             return apply(|x| apply(|y| y * x, 2.0), 3.0) as i32; }}"
        );
        let got = infos(&src);
        assert_eq!(got[0].0, vec!["x"]);
        assert_eq!(got[1].0, Vec::<String>::new());
    }

    #[test]
    fn uncopyable_captures_are_e0418() {
        // A slice or `str` is copied as its fat pointer: another view of the same elements.
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn main() -> i32 {{ let s = alloc_f32(2); let t: str = \"ab\"; \
                 let r = apply(|x| x + s[0] + (t.len() as f32), 1.0) as i32; free(s); \
                 return r; }}"
            )),
            Vec::<&str>::new()
        );
        // A tensor view holds a shape a copy cannot carry.
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn g(t: Tensor[f32, 2]) -> f32 {{ return apply(|x| x * t[1], 2.0); }} \
                 fn main() -> i32 {{ let a: [f32; 2] = [1.0, 2.0]; return g(a) as i32; }}"
            )),
            vec!["E0418"]
        );
        // The enclosing function's function-typed parameter.
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn g(h: fn(f32) -> f32) -> f32 {{ return apply(|x| h(x), 1.0); }} \
                 fn sq(x: f32) -> f32 {{ return x * x; }} \
                 fn main() -> i32 {{ return g(sq) as i32; }}"
            )),
            vec!["E0418"]
        );
        // An enclosing generic parameter, as a value.
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn s<N>(a: [f32; N]) -> f32 {{ return apply(|x| x * (N as f32), a[0]); }} \
                 fn main() -> i32 {{ let a: [f32; 2] = [1.0, 2.0]; return s(a) as i32; }}"
            )),
            vec!["E0418"]
        );
        // ...which a local binding of it fixes.
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn s<N>(a: [f32; N]) -> f32 {{ let n = N as f32; \
                 return apply(|x| x * n, a[0]); }} \
                 fn main() -> i32 {{ let a: [f32; 2] = [1.0, 2.0]; return s(a) as i32; }}"
            )),
            Vec::<&str>::new()
        );
        // A pointer to the slice is the way to read it.
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn main() -> i32 {{ let s = alloc_f32(2); let ps = &s; \
                 let r = apply(|x| x + (*ps)[1], 1.0) as i32; free(s); return r; }}"
            )),
            Vec::<&str>::new()
        );
    }

    /// An array, a tuple, a struct and a data-carrying enum are copied whole: each is a capture
    /// of its own type, read in the body like the original.
    #[test]
    fn aggregates_are_captured_by_value() {
        let src = format!(
            "{APPLY} struct P {{ x: f32, y: f32 }} enum E {{ A(f32), B }} \
             fn main() -> i32 {{ let w: [f32; 2] = [1.0, 2.0]; let t = (3.0, 4.0); \
             let p = P {{ x: 5.0, y: 6.0 }}; let e = E::A(7.0); \
             let k = match e {{ E::A(v) => v, E::B => 0.0 }}; \
             return apply(|x| x + w[1] + t.0 + p.y + k + (match e {{ E::A(v) => v, E::B => 0.0 }}), \
             1.0) as i32; }}"
        );
        let got = infos(&src);
        let mut names = got[0].0.clone();
        names.sort();
        assert_eq!(names, vec!["e", "k", "p", "t", "w"]);
        assert!(got[0].1.by_ref.is_empty());
    }

    /// A local named in a turbofish is read for the dimension's value, so in a closure it is a
    /// capture. Missed, the lifted function had no binding of the name and the dimension quietly
    /// came from the argument's shape instead (`sumn::<n>` over 4 elements summed all 4).
    #[test]
    fn a_turbofish_local_is_a_capture() {
        let sumn = "fn sumn<N>(a: [f32; N]) -> f32 { let mut s = 0.0; \
                    for i in 0..N { s = s + a[i]; } return s; } ";
        let mut got = infos(&format!(
            "{APPLY}{sumn} fn main() -> i32 {{ let xs: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; \
             let px = &xs; let n = 3; return apply(|x| x + sumn::<n>(*px), 0.0) as i32; }}"
        ))[0]
            .0
            .clone();
        got.sort();
        assert_eq!(got, vec!["n", "px"]);
        // A kernel builtin's dimension turbofish reads its local the same way.
        let mut got = infos(&format!(
            "{APPLY} fn main() -> i32 {{ let xs: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; \
             let px = &xs; let m = 3; return apply(|x| x + sum::<m>(*px), 0.0) as i32; }}"
        ))[0]
            .0
            .clone();
        got.sort();
        assert_eq!(got, vec!["m", "px"]);
    }

    /// An enclosing generic named in a turbofish inside a closure is `E0418`, as anywhere else in
    /// the closure (the lifted function has no generics).
    #[test]
    fn an_enclosing_generic_in_a_turbofish_is_e0418() {
        let dimv = "fn dimv<N>() -> f32 { return N as f32; } ";
        assert_eq!(
            error_codes(&format!(
                "{APPLY}{dimv} fn g<M>(a: [f32; M]) -> f32 {{ \
                 return apply(|x| x + dimv::<M>(), a[0]); }} \
                 fn main() -> i32 {{ let a: [f32; 2] = [1.0, 2.0]; return g(a) as i32; }}"
            )),
            vec!["E0418"]
        );
        // A kernel builtin's turbofish too, and binding the generic to a local is the fix.
        let g = |dim: &str| {
            format!(
                "{APPLY} fn g<M>(a: [f32; M]) -> f32 {{ let b: [f32; 2] = [1.0, 2.0]; let m = M; \
                 let pb = &b; return apply(|x| x + sum::<{dim}>(*pb), a[0] + (m as f32)); }} \
                 fn main() -> i32 {{ let a: [f32; 2] = [1.0, 2.0]; return g(a) as i32; }}"
            )
        };
        assert_eq!(error_codes(&g("M")), vec!["E0418"]);
        assert_eq!(error_codes(&g("m")), Vec::<&str>::new());
    }

    /// A callee generic hidden in a tensor parameter type — its element type or a dimension — is
    /// bound by the call's other arguments when they fix it (the closure is then checked at the
    /// bound type), and leaves the closure parameter as open as a bare `T` does when they do not:
    /// `E0419`, not the callee's own `T` or `N` leaking into the closure's type.
    #[test]
    fn a_callee_generic_inside_a_tensor_parameter_is_e0419() {
        for callee in [
            "fn first<T>(f: fn(Tensor[T, 4]) -> f32, t: Tensor[T, 4]) -> f32 { return f(t); }",
            "fn first<N>(f: fn(Tensor[f32, N]) -> f32, t: Tensor[f32, N]) -> f32 { return f(t); }",
        ] {
            let src = format!(
                "{callee} fn main() -> i32 {{ let xs: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; \
                 return first(|t| t[0], xs) as i32; }}"
            );
            assert_eq!(error_codes(&src), Vec::<&str>::new(), "{callee}");
            let got = infos(&src);
            let Ty::Tensor { elem, shape, .. } = &got[0].1.params[0] else {
                panic!("{:?}", got[0].1.params);
            };
            assert_eq!(*elem, Scalar::F32, "{callee}");
            assert_eq!(shape.0, vec![wukong_types::Dim::Const(4)], "{callee}");
        }
        // Nothing else binds them: annotations needed.
        for callee in [
            "fn first<T>(f: fn(Tensor[T, 4]) -> f32) -> f32 { return 1.0; }",
            "fn first<N>(f: fn(Tensor[f32, N]) -> f32) -> f32 { return 1.0; }",
        ] {
            assert_eq!(
                error_codes(&format!(
                    "{callee} fn main() -> i32 {{ return first(|t| t[0]) as i32; }}"
                )),
                vec!["E0419"],
                "{callee}"
            );
        }
        // Annotated, the closure is no longer open: it is checked exactly like a named function of
        // the same type passed there, and the function's extent binds `N`.
        let first = "fn first<N>(f: fn(Tensor[f32, N]) -> f32, t: Tensor[f32, N]) -> f32 { \
                     return f(t); } ";
        let annotated = error_codes(&format!(
            "{first} fn main() -> i32 {{ let xs: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; \
             return first(|t: Tensor[f32, 4]| t[0], xs) as i32; }}"
        ));
        let named = error_codes(&format!(
            "{first} fn head(t: Tensor[f32, 4]) -> f32 {{ return t[0]; }} \
             fn main() -> i32 {{ let xs: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; \
             return first(head, xs) as i32; }}"
        ));
        assert!(annotated.is_empty(), "{annotated:?}");
        assert_eq!(annotated, named);
    }

    /// A function-typed parameter whose type names a dimension generic accepts a function whose
    /// extents the call binds it to — and only that: the binding is the one every argument agrees
    /// on, checked like any other.
    #[test]
    fn a_dimension_generic_function_parameter_binds_from_the_function() {
        let mapn = "fn mapn<N>(f: fn(Tensor[f32, N]) -> f32, t: Tensor[f32, N]) -> f32 { \
                    return f(t); } fn head(t: Tensor[f32, 4]) -> f32 { return t[0]; } ";
        assert_eq!(
            error_codes(&format!(
                "{mapn} fn main() -> i32 {{ let xs: [f32; 4] = [1.0; 4]; \
                 return mapn(head, xs) as i32; }}"
            )),
            Vec::<&str>::new()
        );
        // `head` binds `N = 4` first, so an 8-element buffer is the mismatch.
        assert_eq!(
            error_codes(&format!(
                "{mapn} fn main() -> i32 {{ let xs: [f32; 8] = [1.0; 8]; \
                 return mapn(head, xs) as i32; }}"
            )),
            vec!["E0501"]
        );
        // The buffer binds `N = 8` first, so `head` is the mismatch.
        let rev = "fn mapr<N>(t: Tensor[f32, N], f: fn(Tensor[f32, N]) -> f32) -> f32 { \
                   return f(t); } fn head(t: Tensor[f32, 4]) -> f32 { return t[0]; } ";
        assert_eq!(
            error_codes(&format!(
                "{rev} fn main() -> i32 {{ let xs: [f32; 8] = [1.0; 8]; \
                 return mapr(xs, head) as i32; }}"
            )),
            vec!["E0401"]
        );
        // A dimension named only by the function parameter is bound by the function alone.
        assert_eq!(
            error_codes(
                "fn gen<N>(_f: fn(Tensor[f32, N]) -> f32) -> f32 { return N as f32; } \
                 fn head(t: Tensor[f32, 4]) -> f32 { return t[0]; } \
                 fn main() -> i32 { return gen(head) as i32; }"
            ),
            Vec::<&str>::new()
        );
    }

    /// An unannotated closure parameter typed by a callee generic takes the type the call's other
    /// arguments bind: the non-literal ones first, then a literal at its default type.
    #[test]
    fn a_parameter_typed_by_a_callee_generic_is_e0419() {
        let fold3 =
            "fn fold3<T>(f: fn(T, T) -> T, a: T, b: T, c: T) -> T { return f(f(a, b), c); } ";
        let params = |main: &str| {
            let got = infos(&format!("{fold3} {main}"));
            got[0].1.params.clone()
        };
        assert_eq!(
            params("fn main() -> i32 { return fold3(|a, b| a + b, 1, 2, 3); }"),
            vec![Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I32)]
        );
        assert_eq!(
            params(
                "fn main() -> i32 { let x: i64 = 1; return fold3(|a, b| a + b, 2, x, 3) as i32; }"
            ),
            vec![Ty::Scalar(Scalar::I64), Ty::Scalar(Scalar::I64)]
        );
        // A later closure is checked against what an earlier one bound.
        let two = "fn two<T>(f: fn(T) -> T, g: fn(T) -> T) -> i32 { return 0; } ";
        assert_eq!(
            error_codes(&format!(
                "{two} fn main() -> i32 {{ return two(|a: f64| a, |b| b * 2.0); }}"
            )),
            Vec::<&str>::new()
        );
        // Nothing binds `T`: the parameter needs an annotation.
        let mk = "fn mk<T>(f: fn(T) -> T) -> i32 { return 0; } ";
        assert_eq!(
            error_codes(&format!("{mk} fn main() -> i32 {{ return mk(|x| x); }}")),
            vec!["E0419"]
        );
        assert_eq!(
            error_codes(&format!(
                "{mk} fn main() -> i32 {{ return mk(|x: u8| x); }}"
            )),
            Vec::<&str>::new()
        );
        // Bound to the enclosing function's own generic: the lifted function is not generic.
        assert_eq!(
            error_codes(
                "fn ap<U>(f: fn(U) -> U, x: U) -> U { return f(x); } \
                 fn outer<T>(x: T) -> T { return ap(|y| y, x); } \
                 fn main() -> i32 { return outer(3); }"
            ),
            vec!["E0418"]
        );
    }

    #[test]
    fn a_closure_value_captures_nothing_and_lives_in_a_body() {
        // Anywhere else in a body a closure is a function value: one that captures nothing is
        // legal, one that captures is `E0427`.
        for body in [
            "let a = [|x: i32| x]; return a[0](1);",
            "let t = (1, |x: i32| x); return t.0;",
            "let mut f = |x: i32| x; f = |x| x + 1; return f(1);",
        ] {
            assert_eq!(
                error_codes(&format!("fn main() -> i32 {{ {body} }}")),
                Vec::<&str>::new(),
                "{body}"
            );
        }
        for body in [
            "let k = 1; let a = [|x: i32| x + k]; return 0;",
            "let k = 1; let mut f = |x: i32| x; f = |x| x + k; return f(1);",
        ] {
            assert_eq!(
                error_codes(&format!("fn main() -> i32 {{ {body} }}")),
                vec!["E0427"],
                "{body}"
            );
        }
        assert_eq!(
            error_codes(
                "fn mk() -> fn(i32) -> i32 { return |x| x + 1; } \
                 fn main() -> i32 { return mk()(1); }"
            ),
            Vec::<&str>::new()
        );
        assert_eq!(
            error_codes(
                "fn mk(k: i32) -> fn(i32) -> i32 { return |x| x + k; } \
                 fn main() -> i32 { return mk(1)(1); }"
            ),
            vec!["E0427"]
        );
        // Returned where the function's type is no function type: a mismatch.
        assert!(error_codes("fn main() -> i32 { return |x: i32| x; }").contains(&"E0401"));
        // A const initializer.
        assert!(error_codes(&format!(
            "{APPLY} const C: f32 = apply(|x| x, 1.0); fn main() -> i32 {{ return C as i32; }}"
        ))
        .contains(&"E0411"));
        // A method-call argument and a `let` are not: both are checked.
        assert_eq!(
            error_codes(&format!(
                "{APPLY} struct V {{ x: f32 }} impl V {{ fn get(self, f: fn(f32) -> f32) -> f32 {{ \
                 return f(self.x); }} }} \
                 fn main() -> i32 {{ let v = V {{ x: 1.0 }}; let k = 2.0; \
                 return v.get(|x| x * k) as i32; }}"
            )),
            Vec::<&str>::new()
        );
    }

    /// `let f = |..| ..;` binds a closure: every call `f(..)` and argument `g(f)` records which
    /// literal it denotes, and the literal is checked with no expected type.
    #[test]
    fn a_let_bound_closure_records_each_use() {
        let src = format!(
            "{APPLY} fn main() -> i32 {{ let k = 2.0; let f = |x: f32| x * k; \
             let a = f(1.0); let b = apply(f, 3.0); return (a + b) as i32; }}"
        );
        let (_m, res, diags, interner) = check(&src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        assert_eq!(res.closures.len(), 1);
        let (&lit, info) = res.closures.iter().next().unwrap();
        let names: Vec<_> = info
            .captures
            .iter()
            .map(|(s, _)| interner.resolve(*s))
            .collect();
        assert_eq!(names, vec!["k"]);
        assert_eq!(res.closure_refs.len(), 2);
        assert!(res.closure_refs.values().all(|l| *l == lit));
        // An annotated binding: the annotation is the expected type.
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn main() -> i32 {{ let f: fn(f32) -> f32 = |x| x + 1.0; \
                 return apply(f, 1.0) as i32; }}"
            )),
            Vec::<&str>::new()
        );
        assert_eq!(
            error_codes("fn main() -> i32 { let f: fn(i32) -> i32 = |x: f32| x; return 0; }"),
            vec!["E0401"]
        );
        // Unannotated, a parameter or a `return`-using body has nothing to be inferred from.
        assert_eq!(
            error_codes("fn main() -> i32 { let f = |x| x; return f(1); }"),
            vec!["E0419"]
        );
        assert_eq!(
            error_codes(
                "fn main() -> i32 { let f = |x: i32| { if x > 0 { return x; } 0 }; return f(1); }"
            ),
            vec!["E0419"]
        );
    }

    /// A closure-bound local can only be called or passed to a function-typed parameter: anything
    /// else would need a run-time function value (`E0421`). It cannot be rebound (`E0422`).
    #[test]
    fn a_let_bound_closure_cannot_escape() {
        for body in [
            "let g = f; return 0;",
            "print(f); return 0;",
            "return f(1.0) as i32 + f;",
            "let t = (f, 1); return t.1;",
            "let p = &f; return 0;",
        ] {
            assert_eq!(
                error_codes(&format!(
                    "fn main() -> i32 {{ let k = 2.0; let f = |x: f32| x * k; {body} }}"
                )),
                vec!["E0421"],
                "{body}"
            );
        }
        // Passed to a parameter that is not of function type.
        assert_eq!(
            error_codes(
                "fn takes(x: f32) -> f32 { return x; } \
                 fn main() -> i32 { let k = 2.0; let f = |x: f32| x * k; \
                 return takes(f) as i32; }"
            ),
            vec!["E0421"]
        );
        // A closure that captures nothing is a function value: copying it is fine.
        assert_eq!(
            error_codes(
                "fn main() -> i32 { let f = |x: f32| x; let g = f; return g(1.0) as i32; }"
            ),
            Vec::<&str>::new()
        );
        assert_eq!(
            error_codes(
                "fn main() -> i32 { let k = 1.0; let mut f = |x: f32| x + k; \
                 return f(1.0) as i32; }"
            ),
            vec!["E0422"]
        );
        assert_eq!(
            error_codes("fn main() -> i32 { let mut f = |x: f32| x; return f(1.0) as i32; }"),
            Vec::<&str>::new()
        );
        assert_eq!(
            error_codes(
                "fn main() -> i32 { let f = |x: f32| x; f = |x: f32| x + 1.0; \
                 return f(1.0) as i32; }"
            ),
            vec!["E0422"]
        );
        // A shadowing binding of another kind hides the closure.
        assert_eq!(
            error_codes(
                "fn main() -> i32 { let f = |x: f32| x; let a = f(1.0); let f = 2.0; \
                 return (a + f) as i32; }"
            ),
            Vec::<&str>::new()
        );
    }

    /// A closure that calls a closure-bound local of the enclosing code captures it: the lifted
    /// function receives that closure's own captures (`ClosureInfo::bound`).
    #[test]
    fn a_closure_bound_local_is_captured_through_its_closure() {
        let src = format!(
            "{APPLY} fn main() -> i32 {{ let k = 2.0; let f = |x: f32| x * k; \
             return apply(|y| f(y) + 1.0, 3.0) as i32; }}"
        );
        let (_m, res, diags, interner) = check(&src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        let mut all: Vec<_> = res.closures.iter().collect();
        all.sort_by_key(|(id, _)| id.0);
        let (&f_lit, _) = all[0];
        let outer = all[1].1;
        assert_eq!(outer.bound.len(), 1);
        assert_eq!(interner.resolve(outer.bound[0].0), "f");
        assert_eq!(outer.bound[0].1, f_lit);
    }

    /// A closure that writes a variable it captures — `c = ..`, `c += ..`, `&mut c`, an element
    /// or field of it — captures it BY REFERENCE (`ClosureInfo::by_ref`): the variable's own
    /// mutability rules apply, so an immutable binding is `E0304` as it is outside the closure.
    /// Writing through a captured pointer writes a pointee, and leaves the pointer a copy.
    #[test]
    fn writing_a_capture_is_e0304_and_writing_through_one_is_not() {
        let by_ref = |body: &str| -> Vec<String> {
            let src = format!(
                "{APPLY} fn bump(p: *mut f32) {{ *p = *p + 1.0; }} \
                 struct P {{ x: f32 }} \
                 fn main() -> i32 {{ let mut c = 1.0; let mut a: [f32; 2] = [1.0, 2.0]; \
                 let mut p = P {{ x: 1.0 }}; let mut total = 0.0; let pt = &mut total; \
                 let r = apply(|x| {{ {body} x }}, 1.0); \
                 return (r + c + a[0] + p.x + total) as i32; }}"
            );
            let (_m, res, diags, interner) = check(&src);
            assert!(diags.iter().all(|d| !d.is_error()), "{body}: {diags:?}");
            let info = res.closures.values().next().expect("a closure");
            let mut names: Vec<String> = info
                .by_ref
                .iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect();
            names.sort();
            names
        };
        assert_eq!(by_ref("c = x;"), vec!["c"]);
        assert_eq!(by_ref("c += x;"), vec!["c"]);
        assert_eq!(by_ref("bump(&mut c);"), vec!["c"]);
        assert_eq!(by_ref("a[1] = x;"), vec!["a"]);
        assert_eq!(by_ref("p.x = x;"), vec!["p"]);
        assert_eq!(by_ref("let y = c + a[0] + p.x; c = y;"), vec!["c"]);
        // Through a captured pointer: the pointer is a copy.
        assert_eq!(by_ref("*pt = *pt + x;"), Vec::<String>::new());
        // An immutable binding cannot be written, inside a closure or out.
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn main() -> i32 {{ let c = 1.0; \
                 return apply(|x| {{ c = x; x }}, 1.0) as i32; }}"
            )),
            vec!["E0304"]
        );
        // A nested closure's write is a by-reference capture of every closure it crosses.
        let src = format!(
            "{APPLY} fn main() -> i32 {{ let mut z = 0.0; \
             let r = apply(|x| apply(|y| {{ z = z + y; y }}, x) + z, 2.0); \
             return (r + z) as i32; }}"
        );
        let (_m, res, diags, _i) = check(&src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        assert!(res.closures.values().all(|c| c.by_ref.len() == 1));
    }

    /// A call writes an aggregate argument in place when it is bound to a `mut` parameter (of a
    /// function, a method or an associated function) or is a builtin's output operand: a closure
    /// passing a variable it captures there writes it, so it captures it by reference. A non-`mut`
    /// parameter, a scalar `mut` parameter (a copy) and a builtin's input only read.
    #[test]
    fn an_aggregate_a_call_writes_is_captured_by_reference() {
        let by_ref = |body: &str| -> Vec<String> {
            let src = format!(
                "fn each(f: fn(i32), n: i32) {{ for i in 0..n {{ f(i); }} }} \
                 fn fill(mut o: [f32; 4], v: f32) {{ for i in 0..4 {{ o[i] = v; }} }} \
                 fn peek(o: [f32; 4]) -> f32 {{ return o[0]; }} \
                 fn bump(mut k: i32) -> i32 {{ k += 1; return k; }} \
                 struct S {{ o: [f32; 4] }} \
                 impl S {{ fn put(self, mut o: [f32; 4]) {{ o[0] = self.o[0]; }} }} \
                 fn main() -> i32 {{ let x: [f32; 4] = [1.0, -2.0, 3.0, -4.0]; \
                 let mut out: [f32; 4] = [0.0; 4]; let mut s = S {{ o: [0.0; 4] }}; \
                 let n = 2; let mut ix: [i64; 4] = [0; 4]; let mut buf: [u8; 8] = [0; 8]; \
                 each(|i| {{ {body} print(i); }}, 1); \
                 return (out[0] + s.o[0]) as i32 + n + (ix[0] as i32) + (buf[0] as i32); }}"
            );
            let (_m, res, diags, interner) = check(&src);
            assert!(diags.iter().all(|d| !d.is_error()), "{body}: {diags:?}");
            let info = res.closures.values().next().expect("a closure");
            let mut names: Vec<String> = info
                .by_ref
                .iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect();
            names.sort();
            names
        };
        assert_eq!(by_ref("fill(out, 1.0);"), vec!["out"]);
        assert_eq!(by_ref("fill(s.o, 1.0);"), vec!["s"]);
        assert_eq!(by_ref("s.put(out);"), vec!["out"]);
        assert_eq!(by_ref("S::put(s, out);"), vec!["out"]);
        // The kernel, ML and text builtins' output operands.
        assert_eq!(by_ref("relu(x, out);"), vec!["out"]);
        assert_eq!(by_ref("softmax(x, out);"), vec!["out"]);
        assert_eq!(by_ref("argsort_f32(x, ix);"), vec!["ix"]);
        assert_eq!(by_ref("sprintf(buf, \"hi\");"), vec!["buf"]);
        // Through a tensor view of the capture, which is its storage.
        assert_eq!(by_ref("relu(x, out as Tensor[f32, 2, 2]);"), vec!["out"]);
        assert_eq!(
            by_ref("let t = out as Tensor[f32, 2, 2]; t[0, 1] = 1.0;"),
            vec!["out"]
        );
        // Read only: the copy stays a copy.
        assert_eq!(by_ref("print(peek(out));"), Vec::<String>::new());
        assert_eq!(by_ref("print(bump(n));"), Vec::<String>::new());
        assert_eq!(by_ref("print(peek(s.o) + x[0]);"), Vec::<String>::new());
        assert_eq!(
            by_ref("print(sum(out as Tensor[f32, 4]));"),
            Vec::<String>::new()
        );
    }

    /// A function type does not say which parameters are `mut`, so a call through a function value
    /// — a captured `let`, a field, an element, a call's result, the closure's own function-typed
    /// parameter — may write an aggregate argument in place: a closure passing a variable it
    /// captures there captures it by reference, as for `fill(out, ..)` above. A scalar is a copy
    /// whatever the callee, and a closure-bound local's closure cannot write its arguments (a
    /// closure parameter is never `mut`).
    #[test]
    fn an_aggregate_a_call_through_a_function_value_gets_is_captured_by_reference() {
        let by_ref = |body: &str| -> Vec<String> {
            let src = format!(
                "fn each(f: fn(i32), n: i32) {{ for i in 0..n {{ f(i); }} }} \
                 fn each2(f: fn(fn([f32; 4]), i32), n: i32) {{ f(fill, n); }} \
                 fn fill(mut o: [f32; 4]) {{ for i in 0..4 {{ o[i] = 1.0; }} }} \
                 fn pick() -> fn([f32; 4]) {{ return fill; }} \
                 fn inc(k: i32) -> i32 {{ return k + 1; }} \
                 struct S {{ op: fn([f32; 4]) }} \
                 fn main() -> i32 {{ let mut out: [f32; 4] = [0.0; 4]; let n = 2; \
                 let f: fn([f32; 4]) = fill; let g: fn(i32) -> i32 = inc; \
                 let s = S {{ op: fill }}; let ops: [fn([f32; 4]); 1] = [fill]; \
                 let peek = |o: [f32; 4]| o[0]; \
                 each(|i| {{ {body} print(i); }}, 1); \
                 each2(|h: fn([f32; 4]), i: i32| {{ h(out); print(i); }}, 1); \
                 f(out); s.op(out); ops[0](out); pick()(out); \
                 return out[0] as i32 + n + g(1) + peek(out) as i32; }}"
            );
            let (_m, res, diags, interner) = check(&src);
            assert!(diags.iter().all(|d| !d.is_error()), "{body}: {diags:?}");
            // The `each` closure: one parameter and no result (`peek` returns an `f32`, and the
            // `each2` closure takes two).
            let info = res
                .closures
                .values()
                .find(|c| c.params.len() == 1 && c.ret == Ty::Unit)
                .expect("the `each` closure");
            let mut names: Vec<String> = info
                .by_ref
                .iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect();
            names.sort();
            names
        };
        assert_eq!(by_ref("f(out);"), vec!["out"]);
        assert_eq!(by_ref("pick()(out);"), vec!["out"]);
        assert_eq!(by_ref("s.op(out);"), vec!["out"]);
        assert_eq!(by_ref("ops[0](out);"), vec!["out"]);
        // A scalar is copied into any callee; a closure-bound local's closure only reads.
        assert_eq!(by_ref("print(g(n));"), Vec::<String>::new());
        assert_eq!(by_ref("print(peek(out));"), Vec::<String>::new());
    }

    /// A `for` counter is stored at the wider of its bounds' integer types, and a closure captures
    /// it at that width, by value or by reference: `for i in 0..n` with `n: i64` is an `i64`
    /// capture though `i` is an `i32`. A bound whose type names an enclosing generic leaves the
    /// width to each instance, which the one lifted function cannot follow: `E0418`.
    #[test]
    fn a_range_counter_is_captured_at_its_storage_width() {
        let each = "fn each(f: fn(i32), n: i32) { for j in 0..n { f(j); } } ";
        let capture = |bounds: &str, body: &str| -> (Ty, bool) {
            let got = infos(&format!(
                "{each} fn main() -> i32 {{ let n: i64 = 3; let m: i8 = 2; let mut t = 0; \
                 for i in {bounds} {{ each(|j| {{ t += j; {body} }}, 2); }} \
                 return t + (m as i32) + (n as i32); }}"
            ));
            let (names, info) = &got[0];
            let at = names
                .iter()
                .position(|n| n == "i")
                .expect("`i` is captured");
            let (sym, ty) = &info.captures[at];
            (ty.clone(), info.by_ref.contains(sym))
        };
        let (i32_, i64_) = (Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I64));
        assert_eq!(capture("0..n", "print(i);"), (i64_.clone(), false));
        assert_eq!(capture("0..n", "i = -2;"), (i64_.clone(), true));
        assert_eq!(capture("0..3", "print(i);"), (i32_.clone(), false));
        // The start's type is the counter's when it is the wider one.
        assert_eq!(capture("n..10", "print(i);"), (i64_, false));
        assert_eq!(capture("m..10", "print(i);"), (i32_, false));
        // A nested closure takes the same width.
        let got = infos(&format!(
            "{each} fn main() -> i32 {{ let n: i64 = 3; for i in 0..n {{ \
             each(|_j| {{ each(|_k| {{ print(i); }}, 1); }}, 1); }} return 0; }}"
        ));
        assert!(got
            .iter()
            .all(|(_, c)| c.captures[0].1 == Ty::Scalar(Scalar::I64)));
        // An end bound of an enclosing generic type.
        let generic = |body: &str| {
            error_codes(&format!(
                "{each} fn g<T>(n: T) -> i32 {{ let mut t = 0; \
                 for i in 0..n {{ {body} }} return t; }} \
                 fn main() -> i32 {{ let n: i64 = 3; return g(n); }}"
            ))
        };
        assert_eq!(generic("each(|j| { t += i + j; }, 2);"), vec!["E0418"]);
        assert_eq!(
            generic("let k = i; each(|j| { t += k + j; }, 2);"),
            Vec::<&str>::new()
        );
    }

    #[test]
    fn an_arity_mismatch_is_e0401() {
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn main() -> i32 {{ return apply(|x, y| x + y, 1.0) as i32; }}"
            )),
            vec!["E0401"]
        );
        // A mismatched annotation is the call's ordinary function-type mismatch.
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn main() -> i32 {{ return apply(|x: i32| x, 1.0) as i32; }}"
            )),
            vec!["E0401"]
        );
    }

    #[test]
    fn break_cannot_cross_a_closure() {
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn main() -> i32 {{ let mut s = 0.0; \
                 for i in 0..3 {{ s = s + apply(|x| {{ break; x }}, 1.0); }} return s as i32; }}"
            )),
            vec!["E0303"]
        );
    }

    /// A closure's `return` leaves the closure only: it neither satisfies the enclosing function's
    /// definite-return check nor makes the code after the call unreachable.
    #[test]
    fn a_closure_return_does_not_end_the_enclosing_function() {
        let src = format!(
            "{APPLY} fn f(c: bool) -> f32 {{ if c {{ apply(|x| {{ return x; }}, 1.0); }} \
             else {{ apply(|x| -> f32 {{ return x; }}, 2.0); }} }} \
             fn main() -> i32 {{ return f(true) as i32; }}"
        );
        assert_eq!(error_codes(&src), vec!["E0434"]);
        // The block-bodied closure with an early `return`, and the code after the call, are fine.
        let src = format!(
            "{APPLY} fn main() -> i32 {{ \
             let v = apply(|x| -> f32 {{ if x < 0.0 {{ return -x; }} x }}, -4.0); \
             print(v); return 0; }}"
        );
        let (module, res, diags, interner) = check(&src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        let lints = crate::lints::run(&module, &res, &interner, &Default::default());
        assert!(lints.is_empty(), "{lints:?}");
        // The lifted body is the literal's: a block with the early return as its first statement.
        let ItemKind::Fn(main) = &module.items[1].kind else {
            panic!("main")
        };
        let StmtKind::Let {
            init: Some(init), ..
        } = &main.body.as_ref().unwrap().stmts[0].kind
        else {
            panic!("let v")
        };
        let ExprKind::Call { args, .. } = &init.kind else {
            panic!("call")
        };
        assert!(matches!(args[0].kind, ExprKind::Closure { .. }));
    }

    /// A closure that promises a value must produce one on every path, like a function.
    #[test]
    fn a_closure_that_can_fall_off_its_end_is_e0434() {
        for body in [
            "|x| { print(x); }",
            "|x| -> f32 { if x > 0.0 { return x; } }",
        ] {
            assert_eq!(
                error_codes(&format!(
                    "{APPLY} fn main() -> i32 {{ return apply({body}, 1.0) as i32; }}"
                )),
                vec!["E0434"],
                "{body}"
            );
        }
    }

    /// A capture counts as a use of the enclosing binding, and a closure parameter is a parameter
    /// of the closure: neither lints as unused when read.
    #[test]
    fn captures_and_closure_parameters_lint_as_used() {
        let src = format!(
            "{APPLY} fn main() -> i32 {{ let k = 2.0; return apply(|x| x * k, 1.0) as i32; }}"
        );
        let (module, res, diags, interner) = check(&src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        let lints = crate::lints::run(&module, &res, &interner, &Default::default());
        assert!(lints.is_empty(), "{lints:?}");
        // An unread closure parameter is an unused parameter.
        let src = format!("{APPLY} fn main() -> i32 {{ return apply(|x| 1.0, 1.0) as i32; }}");
        let (module, res, diags, interner) = check(&src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        let lints = crate::lints::run(&module, &res, &interner, &Default::default());
        let codes: Vec<_> = lints.iter().filter_map(|d| d.code).collect();
        assert_eq!(codes, vec!["W0001"]);
    }
}
