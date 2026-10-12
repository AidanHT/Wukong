//! **Closures** — non-escaping `|x| body` literals, checked where they are written.
//!
//! A closure literal is legal only as a direct argument of a plain call `g(..)` to a user function
//! whose corresponding parameter has a function type (`fn apply(f: fn(f32) -> f32, x: f32)`);
//! anywhere else it is `E0411`. `wukong_mir_build` then lifts it to a top-level function whose
//! leading parameters are its captures, and specializes the callee per closure exactly as it does
//! per named function (`apply$main$closure$0`), so a call through `f` stays a direct call with no
//! function pointer at run time. Everything that lift needs is decided here and published as a
//! [`ClosureInfo`] per literal ([`crate::SemaResult::closures`]):
//!
//! * **Typing.** Parameter `i` takes its annotation, else the `i`-th parameter of the expected
//!   function type when that names no generic of the callee (else `E0419`). The return type is the
//!   annotation, else the expected return when it is generic-free, else the body's type (and a body
//!   that then uses `return` is `E0419` too: nothing fixes what it returns). The closure's type
//!   `fn(P..) -> R` must equal the expected one after the call's generic binding — the existing
//!   `E0401` of `check_fn_value_args`.
//! * **Captures.** Every enclosing local the body reads is captured BY VALUE, copied when the call
//!   evaluates that argument. Only what can be copied into a scalar parameter may be captured: a
//!   scalar, a C-style enum, a raw pointer or a reference. Anything else — an aggregate, a slice, a
//!   tensor, a vector, the enclosing function's function-typed parameter, a value whose type names
//!   an enclosing generic — is `E0418`, as is any use of an enclosing generic parameter (the lifted
//!   function is not generic). A capture is read-only: assigning it or taking `&mut` of it is
//!   `E0304`; writing THROUGH a captured pointer is the way to update the caller's data.
//! * **Control.** `return` leaves the closure; `break`/`continue` cannot cross it (`E0303`, the
//!   enclosing loops are hidden while its body is checked). Nested closures are allowed: a name an
//!   inner closure captures is captured by every closure it crosses.

use wukong_ast::{Expr, ExprKind};
use wukong_diag::Diagnostic;
use wukong_span::{FxHashSet as HashSet, Span, Symbol};
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
    /// Every enclosing local it captures (by value), in order of first use, with its type.
    pub captures: Vec<(Symbol, Ty)>,
}

/// One closure whose body is being checked (innermost last in [`Sema::closure_ctx`]).
pub(crate) struct ClosureCtx {
    /// `scopes.len()` before the closure's parameter scope was pushed: a local found in a scope
    /// below this index belongs to the enclosing code, so reading it is a capture.
    base: usize,
    captures: Vec<(Symbol, Ty)>,
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
}

impl Sema<'_> {
    /// Type a call's arguments left to right. A closure literal passed to a function-typed
    /// parameter of a plain call to a user function is checked against that parameter's type
    /// ([`Self::type_closure`]); every other argument is typed as usual (a closure there is
    /// `E0411`, from `type_expr`).
    pub(crate) fn type_call_args(&mut self, callee: &Expr, args: &[Expr]) -> Vec<Ty> {
        let sig = if args
            .iter()
            .any(|a| matches!(a.kind, ExprKind::Closure { .. }))
        {
            match &callee.kind {
                ExprKind::Path(p)
                    if p.is_single() && self.lookup_local(p.first().sym).is_none() =>
                {
                    match self.defs.lookup(p.first().sym).map(|d| &d.kind) {
                        Some(DefKind::Fn(sig)) => Some(sig.clone()),
                        _ => None,
                    }
                }
                _ => None,
            }
        } else {
            None
        };
        let mut tys = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let want = sig.as_ref().and_then(|s| match s.params.get(i) {
                Some(t @ Ty::Fn { .. }) => Some((t, s.generics.as_slice())),
                _ => None,
            });
            let t = match (want, &a.kind) {
                (Some((want, generics)), ExprKind::Closure { .. }) => {
                    let t = self.type_closure(a, want, generics);
                    self.types.insert(a.id, t.clone());
                    t
                }
                _ => self.type_expr(a),
            };
            tys.push(t);
        }
        tys
    }

    /// Check a closure literal passed where a value of function type `expected` is wanted (a
    /// parameter of a callee whose generics are `callee_generics`), and return its type
    /// `fn(P..) -> R` (or `Error` once something about it was reported).
    pub(crate) fn type_closure(
        &mut self,
        e: &Expr,
        expected: &Ty,
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
        let (want_params, want_ret): (&[Ty], Option<&Ty>) = match expected {
            Ty::Fn { params, ret } => (params, Some(ret)),
            _ => (&[], None),
        };
        // At any depth, a tensor's element type and dimensions included: the callee's own `T` or
        // `N` in a parameter type means nothing outside the callee.
        let generic_free = |t: &Ty| !names_generic(t, &|g| callee_generics.contains(&g));

        self.closure_ctx.push(ClosureCtx {
            base: self.scopes.len(),
            captures: Vec::new(),
            reported: Vec::new(),
            failed: false,
            returns: false,
        });

        if want_params.len() != params.len() {
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
                (None, Some(t)) if generic_free(t) => t.clone(),
                (None, Some(_)) => {
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
                let w = expected.display(self.interner);
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
                self.diags.push(
                    Diagnostic::error(format!("cannot infer the type of closure {what} {shown}"))
                        .with_code("E0419")
                        .primary(first.span, "type annotation needed")
                        .note(format!(
                            "the closure is passed as `{w}`, whose parameter types are generic \
                             parameters of the callee; the closure is checked before the call's \
                             other arguments bind them"
                        ))
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
        if known_ret.is_none() && params_as_expected && ctx.returns && !ctx.failed {
            if report {
                let w = expected.display(self.interner);
                self.diags.push(
                    Diagnostic::error(
                        "cannot infer the return type of a closure that uses `return`",
                    )
                    .with_code("E0419")
                    .primary(e.span, "return type annotation needed")
                    .note(format!(
                        "the closure is passed as `{w}`, whose return type is a generic \
                         parameter of the callee"
                    ))
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
                },
            );
        }
        Ty::Fn {
            params: ptys,
            ret: Box::new(ret),
        }
    }

    /// `E0411`: a closure anywhere but a direct argument of a function-typed parameter.
    pub(crate) fn misplaced_closure(&mut self, e: &Expr) -> Ty {
        self.diags.push(
            Diagnostic::error(
                "a closure can only be passed directly to a function-typed parameter of a \
                 function call",
            )
            .with_code("E0411")
            .primary(e.span, "a closure cannot be used here")
            .help(
                "pass it as an argument of a call whose parameter has a function type \
                 (`fn apply(f: fn(f32) -> f32, x: f32)`), or declare a named function",
            ),
        );
        Ty::Error
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
        let why = match self.capture_kind(&ty) {
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
                "a closure captures by value, and only a scalar, a C-style enum, a raw pointer \
                 or a reference can be copied into it",
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

    /// `E0304` for a write to a capture: `c = ..` / `c op= ..` (`borrow == false`) or `&mut c`
    /// (`borrow == true`) where `c` is a bare name the innermost closure captured. A projection
    /// (`p.x = ..`, `(*p)[i] = ..`) writes through a captured pointer and is not checked here.
    /// Returns whether it reported.
    pub(crate) fn check_capture_write(&mut self, target: &Expr, borrow: bool) -> bool {
        let Some(base) = self.closure_ctx.last().map(|c| c.base) else {
            return false;
        };
        let ExprKind::Path(p) = &target.kind else {
            return false;
        };
        if !p.is_single() {
            return false;
        }
        let name = p.first().sym;
        match self.lookup_local_depth(name) {
            Some((depth, _)) if depth < base => {}
            _ => return false,
        }
        if !self.inst_subst.is_empty() {
            return true;
        }
        let nm = self.sym_str(name).to_string();
        let (msg, label) = if borrow {
            (
                format!(
                    "cannot borrow `{nm}` mutably: the closure captured a copy of it, which is \
                     read-only"
                ),
                "mutable borrow of a captured value",
            )
        } else {
            (
                format!(
                    "cannot assign to `{nm}`: the closure captured a copy of it, which is \
                     read-only"
                ),
                "assignment to a captured value",
            )
        };
        self.diags.push(
            Diagnostic::error(msg)
                .with_code("E0304")
                .primary(target.span, label)
                .help(format!(
                    "to update the caller's variable, capture a pointer to it: `let p{nm} = &mut \
                     {nm};` before the call, then `*p{nm} = ..` in the closure"
                )),
        );
        true
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
            Ty::Named(n) => match self.defs.lookup(*n).map(|d| &d.kind) {
                Some(DefKind::Enum(vs)) if vs.iter().all(|v| v.payload.is_unit()) => Ok(()),
                _ => Err(NoCopy::Value),
            },
            Ty::Fn { .. } => Err(NoCopy::FnParam),
            _ if self.mentions_enclosing_generic(ty) => Err(NoCopy::Generic),
            _ => Err(NoCopy::Value),
        }
    }

    /// Does `t` name a generic parameter of the enclosing function — as a type, a tensor element
    /// or a tensor dimension?
    fn mentions_enclosing_generic(&self, t: &Ty) -> bool {
        names_generic(t, &|g| self.generics.contains(&g))
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
        // An array.
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn main() -> i32 {{ let a: [f32; 2] = [1.0, 2.0]; \
                 return apply(|x| x + a[0], 1.0) as i32; }}"
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
        // A pointer to the array is the way to read it.
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn main() -> i32 {{ let a: [f32; 2] = [1.0, 2.0]; let pa = &a; \
                 return apply(|x| x + (*pa)[1], 1.0) as i32; }}"
            )),
            Vec::<&str>::new()
        );
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

    /// A callee generic hidden in a tensor parameter type — its element type or a dimension —
    /// leaves the closure parameter as open as a bare `T` does: `E0419`, not the callee's own `T`
    /// or `N` leaking into the closure's type.
    #[test]
    fn a_callee_generic_inside_a_tensor_parameter_is_e0419() {
        for callee in [
            "fn first<T>(f: fn(Tensor[T, 4]) -> f32, t: Tensor[T, 4]) -> f32 { return f(t); }",
            "fn first<N>(f: fn(Tensor[f32, N]) -> f32, t: Tensor[f32, N]) -> f32 { return f(t); }",
        ] {
            assert_eq!(
                error_codes(&format!(
                    "{callee} fn main() -> i32 {{ let xs: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; \
                     return first(|t| t[0], xs) as i32; }}"
                )),
                vec!["E0419"],
                "{callee}"
            );
        }
        // Annotated, the closure is no longer open: it is checked exactly like a named function of
        // the same type passed there (whatever that call's verdict is), never `E0419`.
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
        assert!(!annotated.contains(&"E0419"), "{annotated:?}");
        assert_eq!(annotated, named);
    }

    #[test]
    fn a_parameter_typed_by_a_callee_generic_is_e0419() {
        let fold3 =
            "fn fold3<T>(f: fn(T, T) -> T, a: T, b: T, c: T) -> T { return f(f(a, b), c); } ";
        assert_eq!(
            error_codes(&format!(
                "{fold3} fn main() -> i32 {{ return fold3(|a, b| a + b, 1, 2, 3); }}"
            )),
            vec!["E0419"]
        );
        // Annotated, it checks, and the closure binds `T`.
        assert_eq!(
            error_codes(&format!(
                "{fold3} fn main() -> i32 {{ let off = 10; \
                 return fold3(|a: i32, b: i32| a + b + off, 1, 2, 3); }}"
            )),
            Vec::<&str>::new()
        );
    }

    #[test]
    fn a_closure_outside_a_call_argument_is_e0411() {
        for body in ["let f = |x: f32| x; return 0;", "return |x: i32| x;"] {
            assert_eq!(
                error_codes(&format!("fn main() -> i32 {{ {body} }}")),
                vec!["E0411"],
                "{body}"
            );
        }
        // A method-call argument.
        assert!(error_codes(&format!(
            "{APPLY} struct V {{ x: f32 }} impl V {{ fn get(self, f: fn(f32) -> f32) -> f32 {{ \
             return f(self.x); }} }} \
             fn main() -> i32 {{ let v = V {{ x: 1.0 }}; return v.get(|x| x) as i32; }}"
        ))
        .contains(&"E0411"));
        // A const initializer.
        assert!(error_codes(&format!(
            "{APPLY} const C: f32 = apply(|x| x, 1.0); fn main() -> i32 {{ return C as i32; }}"
        ))
        .contains(&"E0411"));
    }

    #[test]
    fn writing_a_capture_is_e0304_and_writing_through_one_is_not() {
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn main() -> i32 {{ let mut c = 1.0; \
                 return apply(|x| {{ c = x; x }}, 1.0) as i32; }}"
            )),
            vec!["E0304"]
        );
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn bump(p: *mut f32) {{ *p = *p + 1.0; }} \
                 fn main() -> i32 {{ let mut c = 1.0; \
                 return apply(|x| {{ bump(&mut c); x }}, 1.0) as i32; }}"
            )),
            vec!["E0304"]
        );
        assert_eq!(
            error_codes(&format!(
                "{APPLY} fn main() -> i32 {{ let mut total = 0.0; let pt = &mut total; \
                 let r = apply(|x| {{ *pt = *pt + x; x }}, 1.0); return (r + total) as i32; }}"
            )),
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
