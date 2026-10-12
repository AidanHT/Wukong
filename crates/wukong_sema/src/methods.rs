//! **Methods, traits, and function-typed parameters** — behavioural abstraction with static
//! dispatch.
//!
//! The parser flattens `impl Type { fn m(self, ..) .. }` into an ordinary top-level function named
//! `Type.method` (receiver desugared to a `self: Type` parameter) and leaves an
//! [`ItemKind::Impl`] marker; `trait Name { fn m(self) -> R; }` is an [`ItemKind::Trait`]. This
//! module gives those meaning:
//!
//! * **Method tables.** `(type, method) -> function` for every `impl` block, so `x.m(a)` is the
//!   call `Type.m(x, a)` (the receiver auto-dereferences one `&`/`*`) and `Type::m(a)` the call
//!   `Type.m(a)`. A method call on a struct or enum that has no such method is `E0315` — it used to
//!   type as an unmodelled `Unknown` and die in codegen.
//! * **Traits.** An `impl Trait for Type` must define exactly the trait's methods with the trait's
//!   signatures (`Self` read as `Type`); anything else is `E0410`. A generic parameter may be
//!   bounded (`fn f<T: Shape>(x: T)`, or `where T: Shape`), which is what lets its body call
//!   `x.area()` / `T::make()`; every call of `f` must then bind `T` to a type implementing `Shape`
//!   (`E0410` otherwise). Dispatch is static: `mir_build` already specializes a type-generic
//!   function per instantiation, and resolves a call through a bound against the instantiated
//!   type's method table. A trait method has no default body (`E0410`).
//! * **Function-typed parameters.** `fn apply(f: fn(f32) -> f32, x: f32)` — the only place a
//!   function type may appear (`E0411` elsewhere). Every call must pass a named non-generic
//!   function, a closure literal (`closures.rs`), or forward the caller's own function parameter;
//!   `mir_build` specializes the callee per bound function (`apply$square`, `apply$main$closure$0`),
//!   so a call through `f` is a direct call with no function pointer at run time and nothing a
//!   kernel recognizer cannot see.
//!
//! What is recorded here — the method tables and every resolved method call — is published in
//! [`crate::SemaResult::methods`].

use wukong_ast::{Expr, ExprKind, FnDecl, ImplDecl, ItemKind, Module, NodeId, TypeExpr, TypeKind};
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Span, Symbol};
use wukong_types::{Scalar, Ty};

use crate::provenance::CallSite;
use crate::{DefKind, FnSig, Sema};

/// How one method call resolved, keyed in [`MethodResolution::calls`] by the call expression.
#[derive(Clone, Debug)]
pub struct MethodCall {
    /// The function the call reaches when the receiver's type is concrete. `None` for a call
    /// through a trait bound on a generic type (`x.area()` with `x: T`, `T: Shape`), which
    /// `mir_build` resolves per instantiation from [`MethodResolution::table`].
    pub target: Option<Symbol>,
    /// The method name as written.
    pub method: Symbol,
    /// `true` for `x.m(..)` (the receiver is the first argument); `false` for `T::m(..)`.
    pub receiver: bool,
    /// The receiver is a reference or pointer to the type and is dereferenced to call.
    pub deref: bool,
}

/// The published method tables.
#[derive(Clone, Debug, Default)]
pub struct MethodResolution {
    /// `(type, method)` → the function implementing it (`Type.method`).
    pub table: HashMap<(Symbol, Symbol), Symbol>,
    /// Every method call sema typed, keyed by the call expression.
    pub calls: HashMap<NodeId, MethodCall>,
}

impl MethodResolution {
    /// The function `ty.method` resolves to, if `ty` names a type with that method.
    pub fn lookup(&self, ty: &Ty, method: Symbol) -> Option<Symbol> {
        let t = match ty {
            Ty::Ref { pointee, .. } | Ty::Ptr { pointee, .. } => pointee.as_ref(),
            t => t,
        };
        match t {
            Ty::Named(n) => self.table.get(&(*n, method)).copied(),
            _ => None,
        }
    }
}

/// One method signature of a trait. `Self` is `Ty::Named(Self)`.
#[derive(Clone)]
struct TraitMethod {
    name: Symbol,
    has_self: bool,
    /// The receiver is `mut self` / `&mut self`: the call mutates the receiver in place.
    mut_self: bool,
    /// The parameters after the receiver.
    params: Vec<Ty>,
    ret: Ty,
}

/// Sema-internal method/trait state.
#[derive(Default)]
pub(crate) struct MethodState {
    pub(crate) resolution: MethodResolution,
    /// Trait name → its method signatures.
    traits: HashMap<Symbol, Vec<TraitMethod>>,
    /// The functions whose first parameter is the receiver `self`.
    receivers: HashSet<Symbol>,
    /// The subset of `receivers` whose receiver is `mut self` / `&mut self`.
    mut_receivers: HashSet<Symbol>,
    /// `(trait, type)` for every `impl Trait for Type`.
    impls: HashSet<(Symbol, Symbol)>,
    /// Each function's trait bounds: `(generic, traits)` in declaration order.
    fn_bounds: HashMap<Symbol, Vec<(Symbol, Vec<Symbol>)>>,
    /// The bounds of the function whose body is being checked.
    cur_bounds: HashMap<Symbol, Vec<Symbol>>,
    /// Set only while lowering the type of a parameter of a function with a body — the one place a
    /// function type is legal.
    pub(crate) fn_type_ok: bool,
}

impl MethodState {
    /// Whether `name` is a trait this program declares (a trait named where a type belongs is its
    /// own `E0324` message, `resolve.rs`).
    pub(crate) fn is_trait(&self, name: Symbol) -> bool {
        self.traits.contains_key(&name)
    }
}

/// `Self` (as `Ty::Named(self_sym)`) replaced by `to` throughout `t`.
fn subst_self(t: &Ty, self_sym: Symbol, to: &Ty) -> Ty {
    match t {
        Ty::Named(n) if *n == self_sym => to.clone(),
        Ty::Ptr { mutable, pointee } => Ty::Ptr {
            mutable: *mutable,
            pointee: Box::new(subst_self(pointee, self_sym, to)),
        },
        Ty::Ref { mutable, pointee } => Ty::Ref {
            mutable: *mutable,
            pointee: Box::new(subst_self(pointee, self_sym, to)),
        },
        Ty::Slice(e) => Ty::Slice(Box::new(subst_self(e, self_sym, to))),
        Ty::Array { elem, len, shared } => Ty::Array {
            elem: Box::new(subst_self(elem, self_sym, to)),
            len: *len,
            shared: *shared,
        },
        Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|x| subst_self(x, self_sym, to)).collect()),
        Ty::Fn { params, ret } => Ty::Fn {
            params: params.iter().map(|x| subst_self(x, self_sym, to)).collect(),
            ret: Box::new(subst_self(ret, self_sym, to)),
        },
        other => other.clone(),
    }
}

/// Does `t` mention the type name `g` anywhere (by value, behind a pointer, or in an element)?
pub(crate) fn mentions(t: &Ty, g: Symbol) -> bool {
    match t {
        Ty::Named(n) => *n == g,
        Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } | Ty::Slice(pointee) => {
            mentions(pointee, g)
        }
        Ty::Array { elem, .. } => mentions(elem, g),
        Ty::Tuple(ts) => ts.iter().any(|x| mentions(x, g)),
        Ty::Fn { params, ret } => params.iter().any(|x| mentions(x, g)) || mentions(ret, g),
        _ => false,
    }
}

impl Sema<'_> {
    /// Lower a parameter's type. A function type is legal here — and only here, for a function
    /// with a body (an `extern` callback has no compile-time-known callee). A shape-generic array
    /// parameter (`[f32; N]`) is the tensor it denotes (`tensor::lower_shape_param_type`).
    pub(crate) fn lower_param_type(&mut self, has_body: bool, t: &TypeExpr) -> Ty {
        self.methods.fn_type_ok = has_body;
        let ty = self.lower_shape_param_type(t);
        self.methods.fn_type_ok = false;
        ty
    }

    /// A function-typed parameter may not be `mut` (it names the function each call binds at
    /// compile time), and may not take a builtin's name: every kernel recognizer and the
    /// vectorizer read a call `gelu(x)` as the intrinsic unless a top-level function shadows it,
    /// so a parameter called `gelu` would be silently replaced by the builtin in a loop.
    pub(crate) fn check_fn_param_decl(&mut self, p: &wukong_ast::Param, ty: &Ty) {
        if !matches!(ty, Ty::Fn { .. }) {
            return;
        }
        if p.mutable {
            self.error(
                p.name.span,
                "E0411",
                "a function-typed parameter cannot be `mut`: it names the function each call \
                 binds at compile time",
            );
        }
        let nm = self.sym_str(p.name.sym);
        let builtin = crate::shape::intrinsic_ret_ty(nm, &[]).is_some()
            || crate::kernel_builtin(nm, 2).is_some()
            || crate::scalar_builtin(nm).is_some()
            || crate::heap_alloc_elem(nm).is_some()
            || crate::file_io_elem(nm).is_some()
            || crate::text::text_builtin(nm).is_some()
            || matches!(
                nm,
                "print" | "println" | "assert" | "free" | "now_ns" | "sync_block" | "atomic_add"
            );
        if builtin {
            let nm = nm.to_string();
            self.error(
                p.name.span,
                "E0411",
                format!(
                    "a function-typed parameter cannot be named `{nm}`, the name of a builtin; \
                     rename it"
                ),
            );
        }
    }

    /// The `TypeKind::Fn` arm of `lower_type`. `ok` is whether this position may hold one.
    pub(crate) fn lower_fn_type(
        &mut self,
        ok: bool,
        params: &[TypeExpr],
        ret: Option<&TypeExpr>,
        span: Span,
    ) -> Ty {
        if !ok {
            self.error(
                span,
                "E0411",
                "a function type is only allowed as the type of a function parameter (bind it to a \
                 named function at each call); it cannot be a local, field, element, return \
                 value or `extern` parameter",
            );
        }
        let params: Vec<Ty> = params.iter().map(|p| self.lower_type(p)).collect();
        let ret = ret.map(|r| self.lower_type(r)).unwrap_or(Ty::Unit);
        if ok {
            Ty::Fn {
                params,
                ret: Box::new(ret),
            }
        } else {
            Ty::Error
        }
    }

    /// A `let` may not bind a function value: it is not a compile-time-bound parameter, so nothing
    /// could resolve a call through it. Returns the type to bind (`Error` in that case, so the name
    /// cannot be called and cascade).
    pub(crate) fn checked_let_type(&mut self, bound: Ty, span: Span) -> Ty {
        if matches!(bound, Ty::Fn { .. }) {
            self.error(
                span,
                "E0411",
                "a function cannot be bound to a local; pass it directly to a parameter of \
                 function type instead",
            );
            return Ty::Error;
        }
        bound
    }

    /// Collect the traits, the `impl` blocks and every function's trait bounds, and check each
    /// trait implementation. Runs after the signature refresh, so every function signature it
    /// compares is final.
    pub(crate) fn collect_methods(&mut self, module: &Module) {
        let self_sym = self.interner.get("Self");
        let self_word = self.interner.get("self");
        let mut decls: HashMap<Symbol, &FnDecl> = HashMap::default();
        for item in &module.items {
            if let ItemKind::Fn(f) = &item.kind {
                decls.insert(f.name.sym, f);
                if f.params
                    .first()
                    .is_some_and(|p| Some(p.name.sym) == self_word)
                {
                    self.methods.receivers.insert(f.name.sym);
                    if f.params[0].mutable {
                        self.methods.mut_receivers.insert(f.name.sym);
                    }
                }
            }
        }

        // Traits first: an `impl` may precede its trait in source.
        for item in &module.items {
            let ItemKind::Trait(t) = &item.kind else {
                continue;
            };
            self.reject_str_decl(t.name.sym, t.name.span, "trait");
            if self.methods.traits.contains_key(&t.name.sym)
                || self.defs.lookup(t.name.sym).is_some()
            {
                let nm = self.sym_str(t.name.sym).to_string();
                self.error(
                    t.name.span,
                    "E0300",
                    format!("the name `{nm}` is defined more than once"),
                );
                continue;
            }
            let mut sigs: Vec<TraitMethod> = Vec::with_capacity(t.methods.len());
            for f in &t.methods {
                if sigs.iter().any(|m| m.name == f.name.sym) {
                    let nm = self.sym_str(f.name.sym).to_string();
                    self.error(
                        f.name.span,
                        "E0300",
                        format!("duplicate trait method `{nm}`"),
                    );
                    continue;
                }
                if f.body.is_some() {
                    let nm = self.sym_str(f.name.sym).to_string();
                    self.error(
                        f.name.span,
                        "E0410",
                        format!(
                            "trait method `{nm}` has a body; a trait declares signatures only — \
                             write the body in each `impl`"
                        ),
                    );
                }
                if !f.generics.is_empty() {
                    let nm = self.sym_str(f.name.sym).to_string();
                    self.error(
                        f.name.span,
                        "E0410",
                        format!("trait method `{nm}` cannot have generic parameters"),
                    );
                }
                let has_self = f
                    .params
                    .first()
                    .is_some_and(|p| Some(p.name.sym) == self_word);
                let params = f
                    .params
                    .iter()
                    .skip(usize::from(has_self))
                    .map(|p| self.lower_type(&p.ty))
                    .collect();
                let ret = f
                    .ret
                    .as_ref()
                    .map(|r| self.lower_type(r))
                    .unwrap_or(Ty::Unit);
                sigs.push(TraitMethod {
                    name: f.name.sym,
                    has_self,
                    mut_self: has_self && f.params[0].mutable,
                    params,
                    ret,
                });
            }
            self.methods.traits.insert(t.name.sym, sigs);
        }

        for item in &module.items {
            let ItemKind::Impl(imp) = &item.kind else {
                continue;
            };
            self.collect_impl(imp, &decls, self_sym, self_word);
        }

        // Every function's bounds name traits.
        for item in &module.items {
            let ItemKind::Fn(f) = &item.kind else {
                continue;
            };
            let mut bounds: Vec<(Symbol, Vec<Symbol>)> = Vec::new();
            for g in &f.generics {
                let wukong_ast::GenericParamKind::Type(id) = &g.kind else {
                    continue;
                };
                let mut traits = Vec::new();
                for b in &g.bounds {
                    let tname = b.segments.last().map(|s| s.sym);
                    match tname {
                        Some(tn) if self.methods.traits.contains_key(&tn) => traits.push(tn),
                        _ => {
                            let nm = tname.map_or("?", |tn| self.sym_str(tn)).to_string();
                            self.error(b.span, "E0315", format!("cannot find trait `{nm}`"));
                        }
                    }
                }
                if !traits.is_empty() {
                    bounds.push((id.sym, traits));
                }
            }
            if !bounds.is_empty() {
                self.methods.fn_bounds.insert(f.name.sym, bounds);
            }
        }
    }

    /// Register one `impl` block's methods and, for `impl Trait for Type`, check it against the
    /// trait.
    fn collect_impl(
        &mut self,
        imp: &ImplDecl,
        decls: &HashMap<Symbol, &FnDecl>,
        self_sym: Option<Symbol>,
        self_word: Option<Symbol>,
    ) {
        let Some(ty) = imp.self_ty.segments.last().map(|s| s.sym) else {
            return;
        };
        if !matches!(
            self.defs.lookup(ty).map(|d| &d.kind),
            Some(DefKind::Struct(_) | DefKind::Enum(_))
        ) {
            let nm = self.sym_str(ty).to_string();
            self.error(
                imp.self_ty.span,
                "E0315",
                format!(
                    "cannot find struct or enum `{nm}` to implement methods for (an `impl` block \
                     extends a struct or enum declared in this program)"
                ),
            );
            return;
        }
        for m in &imp.methods {
            // A second method of the same name is a second function `Type.m`, already E0300.
            self.methods
                .resolution
                .table
                .entry((ty, m.name.sym))
                .or_insert(m.fn_name);
        }
        let Some(tref) = &imp.trait_ref else {
            return;
        };
        let Some(tname) = tref.segments.last().map(|s| s.sym) else {
            return;
        };
        let Some(sigs) = self.methods.traits.get(&tname).cloned() else {
            let nm = self.sym_str(tname).to_string();
            self.error(tref.span, "E0315", format!("cannot find trait `{nm}`"));
            return;
        };
        let (tn, yn) = (
            self.sym_str(tname).to_string(),
            self.sym_str(ty).to_string(),
        );
        if !self.methods.impls.insert((tname, ty)) {
            self.error(
                imp.self_ty.span,
                "E0300",
                format!("trait `{tn}` is implemented for `{yn}` more than once"),
            );
            return;
        }
        let self_ty = Ty::Named(ty);
        for tm in &sigs {
            let mname = self.sym_str(tm.name).to_string();
            let Some(m) = imp.methods.iter().find(|m| m.name.sym == tm.name) else {
                self.error(
                    imp.self_ty.span,
                    "E0410",
                    format!("`impl {tn} for {yn}` is missing the trait method `{mname}`"),
                );
                continue;
            };
            let Some(DefKind::Fn(sig)) = self.defs.lookup(m.fn_name).map(|d| d.kind.clone()) else {
                continue;
            };
            let has_self = decls
                .get(&m.fn_name)
                .and_then(|f| f.params.first())
                .is_some_and(|p| Some(p.name.sym) == self_word);
            let want_params: Vec<Ty> = tm
                .params
                .iter()
                .map(|t| self_sym.map_or(t.clone(), |s| subst_self(t, s, &self_ty)))
                .collect();
            let want_ret = self_sym.map_or(tm.ret.clone(), |s| subst_self(&tm.ret, s, &self_ty));
            let skip = usize::from(has_self);
            let receiver_ok = !has_self || sig.params.first() == Some(&self_ty);
            let ok = has_self == tm.has_self
                && receiver_ok
                && sig.generics.is_empty()
                && sig.params.len() == want_params.len() + skip
                && sig.params[skip..] == want_params[..]
                && sig.ret == want_ret;
            if !ok {
                let span = decls.get(&m.fn_name).map_or(m.name.span, |f| f.name.span);
                let recv = if tm.has_self { "self" } else { "no receiver" };
                let want: Vec<String> = want_params
                    .iter()
                    .map(|t| t.display(self.interner))
                    .collect();
                self.error(
                    span,
                    "E0410",
                    format!(
                        "method `{mname}` of `impl {tn} for {yn}` does not match the trait: \
                         it must take {recv} and then ({}), and return `{}`",
                        want.join(", "),
                        want_ret.display(self.interner)
                    ),
                );
            }
        }
        for m in &imp.methods {
            if !sigs.iter().any(|tm| tm.name == m.name.sym) {
                let mname = self.sym_str(m.name.sym).to_string();
                self.error(
                    m.name.span,
                    "E0410",
                    format!("method `{mname}` is not a member of trait `{tn}`"),
                );
            }
        }
    }

    /// Establish the trait bounds of the function whose body is checked next.
    pub(crate) fn enter_fn_bounds(&mut self, f: Symbol) {
        self.methods.cur_bounds = self
            .methods
            .fn_bounds
            .get(&f)
            .map(|v| v.iter().cloned().collect())
            .unwrap_or_default();
    }

    /// The trait method `method` reachable through the bounds of generic `g` in the current body.
    fn bound_method(&self, g: Symbol, method: Symbol) -> Option<TraitMethod> {
        let traits = self.methods.cur_bounds.get(&g)?;
        traits.iter().find_map(|t| {
            self.methods
                .traits
                .get(t)?
                .iter()
                .find(|m| m.name == method)
                .cloned()
        })
    }

    /// Does `t` implement `tr`? A generic of the current function does when it is bounded by `tr`.
    fn satisfies(&self, t: &Ty, tr: Symbol) -> bool {
        match t {
            Ty::Named(n) if self.generics.contains(n) => self
                .methods
                .cur_bounds
                .get(n)
                .is_some_and(|ts| ts.contains(&tr)),
            Ty::Named(n) => self.methods.impls.contains(&(tr, *n)),
            Ty::Unknown | Ty::Error => true,
            _ => false,
        }
    }

    /// At a call of `callee`: every bounded type parameter must be bound to a type that implements
    /// its traits (`E0410`). The bindings are re-derived exactly as `check_fn_call` derives them —
    /// a type turbofish, then inference from the arguments.
    pub(crate) fn check_call_bounds(
        &mut self,
        callee: Symbol,
        sig: &FnSig,
        generic_args: &[TypeExpr],
        arg_tys: &[Ty],
        span: Span,
    ) {
        let Some(bounds) = self.methods.fn_bounds.get(&callee).cloned() else {
            return;
        };
        let mut tys: HashMap<Symbol, Ty> = HashMap::default();
        if generic_args.len() == sig.generics.len() {
            for (g, ga) in sig.generics.iter().zip(generic_args) {
                if let TypeKind::Path(p) = &ga.kind {
                    if p.is_single() {
                        let n = p.first().sym;
                        let t = match Scalar::from_name(self.sym_str(n)) {
                            Some(sc) => Ty::Scalar(sc),
                            None => Ty::Named(n),
                        };
                        tys.insert(*g, t);
                    }
                }
            }
        }
        for (p, a) in sig.params.iter().zip(arg_tys) {
            crate::shape::infer_type_generics(p, a, &sig.generics, &mut tys);
        }
        for (g, traits) in bounds {
            let Some(t) = tys.get(&g).cloned() else {
                continue;
            };
            for tr in traits {
                if !self.satisfies(&t, tr) {
                    let (ts, trs, gs) = (
                        t.display(self.interner),
                        self.sym_str(tr).to_string(),
                        self.sym_str(g).to_string(),
                    );
                    self.error(
                        span,
                        "E0410",
                        format!(
                            "the trait bound `{ts}: {trs}` is not satisfied (type parameter \
                             `{gs}` of `{}` requires it; add `impl {trs} for {ts}`)",
                            self.sym_str(callee)
                        ),
                    );
                }
            }
        }
    }

    /// Validate the arguments bound to function-typed parameters, and refuse a function passed
    /// anywhere else. A function argument must name a non-generic top-level function, or forward
    /// the caller's own function-typed parameter, whose type equals the parameter's (with the
    /// callee's type parameters substituted as the call binds them).
    pub(crate) fn check_fn_value_args(&mut self, sig: &FnSig, args: &[Expr], arg_tys: &[Ty]) {
        let mut tys: HashMap<Symbol, Ty> = HashMap::default();
        for (p, a) in sig.params.iter().zip(arg_tys) {
            crate::shape::infer_type_generics(p, a, &sig.generics, &mut tys);
        }
        for (i, a) in args.iter().enumerate() {
            let (Some(pty), Some(aty)) = (sig.params.get(i), arg_tys.get(i)) else {
                continue;
            };
            match (pty, aty) {
                (Ty::Fn { .. }, Ty::Unknown | Ty::Error) => {}
                (Ty::Fn { .. }, _) => {
                    let named = match &a.kind {
                        ExprKind::Path(p) if p.is_single() => Some(p.first().sym),
                        _ => None,
                    };
                    // A closure literal is known too: `mir_build` lifts it to a function
                    // (`closures.rs`), and its type is compared below like a named function's.
                    let closure = matches!(a.kind, ExprKind::Closure { .. });
                    let known = closure
                        || named.is_some_and(|n| match self.lookup_local(n) {
                            Some(t) => matches!(t, Ty::Fn { .. }),
                            None => matches!(
                                self.defs.lookup(n).map(|d| &d.kind),
                                Some(DefKind::Fn(s)) if s.generics.is_empty()
                            ),
                        });
                    if !known {
                        self.error(
                            a.span,
                            "E0411",
                            "a function-typed parameter must be given the name of a non-generic \
                             function, a closure literal, or the caller's own function-typed \
                             parameter, known at compile time",
                        );
                        continue;
                    }
                    let want = crate::shape::apply_subst(pty, &HashMap::default(), &tys);
                    let unresolved = sig.generics.iter().any(|g| mentions(&want, *g));
                    if !matches!(aty, Ty::Fn { .. }) || (!unresolved && *aty != want) {
                        let (w, f) = (want.display(self.interner), aty.display(self.interner));
                        self.error(
                            a.span,
                            "E0401",
                            format!("type mismatch: expected a function `{w}`, found `{f}`"),
                        );
                    }
                }
                (_, Ty::Fn { .. }) if !matches!(pty, Ty::Unknown | Ty::Error) => {
                    self.error(
                        a.span,
                        "E0411",
                        "a function can only be passed to a parameter of function type",
                    );
                }
                _ => {}
            }
        }
    }

    /// A call `f(..)` where `f` is a local of function type — a function-typed parameter. `None`
    /// when `name` is not one.
    pub(crate) fn type_fn_param_call(
        &mut self,
        name: Symbol,
        callee: &Expr,
        generic_args: &[TypeExpr],
        args: &[Expr],
        arg_tys: &mut [Ty],
        span: Span,
    ) -> Option<Ty> {
        let Some(Ty::Fn { params, ret }) = self.lookup_local(name) else {
            return None;
        };
        // Inside a closure, the enclosing function's function-typed parameter cannot be captured.
        self.note_local_use(name, callee.span);
        self.types.insert(
            callee.id,
            Ty::Fn {
                params: params.clone(),
                ret: ret.clone(),
            },
        );
        if !generic_args.is_empty() {
            self.error(
                span,
                "E0503",
                "a call through a function-typed parameter takes no generic arguments",
            );
        }
        self.adapt_scalar_literals(&params, args, arg_tys, 0);
        let sig = FnSig {
            generics: Vec::new(),
            params,
            ret: *ret,
        };
        self.check_fn_value_args(&sig, args, arg_tys);
        let site = CallSite::new(None, None, args);
        Some(self.check_fn_call(&sig, &[], arg_tys, span, callee.id, &site))
    }

    /// Adapt each unsuffixed numeric literal argument to its concrete scalar parameter, exactly as
    /// `type_call` does for a resolved function (`params[i + offset]` is argument `i`'s parameter;
    /// the receiver, when there is one, is parameter 0 and never an argument here). A
    /// literal-initialized binding passed as an argument takes the parameter's type if this is its
    /// first use (`infer.rs`), as it does at a plain call: `let d = 0.1; p.add(d)` with `d: f64`.
    fn adapt_scalar_literals(
        &mut self,
        params: &[Ty],
        args: &[Expr],
        arg_tys: &mut [Ty],
        offset: usize,
    ) {
        for (i, a) in args.iter().enumerate() {
            if let Some(param @ (Ty::Scalar(_) | Ty::Str)) = params.get(i + offset) {
                self.constrain_literal_binding(a, None, param, false);
                if self.literal_adapts(param, a) {
                    self.retype_adapted_literal(a, param);
                    self.range_check_int_literal(a, param);
                    if let Some(slot) = arg_tys.get_mut(i + offset) {
                        *slot = param.clone();
                    }
                }
            }
        }
    }

    /// Type a method call — `x.m(..)` on a struct/enum value (or a reference/pointer to one), or
    /// `Type::m(..)` / `T::m(..)` — or return `None` when the call is not one (an enum variant
    /// constructor, a slice's `.len()`, a vector builtin, …), leaving it to `type_call`.
    pub(crate) fn type_method_call(
        &mut self,
        e: &Expr,
        callee: &Expr,
        generic_args: &[TypeExpr],
        args: &[Expr],
    ) -> Option<Ty> {
        let ExprKind::Field { base, name } = &callee.kind else {
            return None;
        };
        let method = name.sym;

        // `Type::m(..)` / `T::m(..)`: a type (or type parameter) used as a namespace. A top-level
        // `const` is a value, so `ORIGIN.norm()` on a struct-typed const is a receiver call below.
        if let ExprKind::Path(p) = &base.kind {
            if p.is_single()
                && self.lookup_local(p.first().sym).is_none()
                && !matches!(
                    self.defs.lookup(p.first().sym).map(|d| &d.kind),
                    Some(DefKind::Const(_))
                )
            {
                let tsym = p.first().sym;
                if let Some(&target) = self.methods.resolution.table.get(&(tsym, method)) {
                    self.types.insert(base.id, Ty::Unknown);
                    let ret =
                        self.type_resolved_method(e, callee, target, None, generic_args, args);
                    self.methods.resolution.calls.insert(
                        e.id,
                        MethodCall {
                            target: Some(target),
                            method,
                            receiver: false,
                            deref: false,
                        },
                    );
                    return Some(ret);
                }
                if self.generics.contains(&tsym) && self.may_be_bound_method(tsym, method) {
                    // `T::m(..)` inside a closure names the enclosing generic (`closures.rs`).
                    self.note_generic_use(tsym, base.span);
                    self.types.insert(base.id, Ty::Unknown);
                    return Some(self.type_bound_call(e, callee, tsym, method, None, args));
                }
                if matches!(
                    self.defs.lookup(tsym).map(|d| &d.kind),
                    Some(DefKind::Struct(_))
                ) {
                    self.types.insert(base.id, Ty::Unknown);
                    for a in args {
                        self.type_expr(a);
                    }
                    let m = self.sym_str(method).to_string();
                    let t = self.sym_str(tsym).to_string();
                    self.error(
                        callee.span,
                        "E0315",
                        format!("no function or method named `{m}` found for struct `{t}`"),
                    );
                    return Some(Ty::Error);
                }
                return None;
            }
        }

        // `x.m(..)`: only a receiver whose type is a struct/enum (or one reference to one).
        let recv_ty = self.type_expr(base);
        let (inner, deref) = match &recv_ty {
            Ty::Ref { pointee, .. } | Ty::Ptr { pointee, .. } => (pointee.as_ref().clone(), true),
            t => (t.clone(), false),
        };
        // Anything else (a slice's `.len()`, a tensor or vector builtin, a field call) is left to
        // `type_call`, which types such calls; it types the receiver again, which is idempotent.
        let Ty::Named(tsym) = inner else {
            return None;
        };
        if let Some(&target) = self.methods.resolution.table.get(&(tsym, method)) {
            if !self.methods.receivers.contains(&target) {
                let (t, m) = (
                    self.sym_str(tsym).to_string(),
                    self.sym_str(method).to_string(),
                );
                for a in args {
                    self.type_expr(a);
                }
                self.error(
                    callee.span,
                    "E0315",
                    format!(
                        "`{t}::{m}` takes no `self`, so it is an associated function, not a \
                         method; call it as `{t}::{m}(..)`"
                    ),
                );
                return Some(Ty::Error);
            }
            if !deref && self.methods.mut_receivers.contains(&target) {
                self.check_mut_receiver(base, method);
            }
            let ret = self.type_resolved_method(
                e,
                callee,
                target,
                Some(Ty::Named(tsym)),
                generic_args,
                args,
            );
            self.methods.resolution.calls.insert(
                e.id,
                MethodCall {
                    target: Some(target),
                    method,
                    receiver: true,
                    deref,
                },
            );
            return Some(ret);
        }
        if self.generics.contains(&tsym) {
            if !self.may_be_bound_method(tsym, method) {
                return None;
            }
            if deref {
                for a in args {
                    self.type_expr(a);
                }
                self.error(
                    callee.span,
                    "E0315",
                    "a method of a type parameter is called on a value of that type, not through \
                     a reference or pointer to it; dereference it first: `(*x).m(..)`",
                );
                return Some(Ty::Error);
            }
            return Some(self.type_bound_call(e, callee, tsym, method, Some(base), args));
        }
        let is_type = matches!(
            self.defs.lookup(tsym).map(|d| &d.kind),
            Some(DefKind::Struct(_) | DefKind::Enum(_))
        );
        let is_field = matches!(
            self.defs.lookup(tsym).map(|d| &d.kind),
            Some(DefKind::Struct(fields)) if fields.iter().any(|(f, _)| *f == method)
        );
        if !is_type || is_field {
            return None;
        }
        for a in args {
            self.type_expr(a);
        }
        let (m, t) = (
            self.sym_str(method).to_string(),
            self.sym_str(tsym).to_string(),
        );
        self.error(
            callee.span,
            "E0315",
            format!("no method named `{m}` found for type `{t}`"),
        );
        Some(Ty::Error)
    }

    /// A `mut self` / `&mut self` method writes its receiver in place, and an aggregate receiver is
    /// passed by reference — so calling one on a non-`mut` aggregate parameter (or a projection of
    /// one) would mutate the caller's value through it: the same `E0304` an assignment `p.f = …`
    /// through that parameter gets. A receiver reached through a dereference is a pointee and is
    /// not this rule's concern.
    fn check_mut_receiver(&mut self, recv: &Expr, method: Symbol) {
        let Some(root) = self.assign_root_param(recv) else {
            return;
        };
        if self.is_immutable_param(root) && self.param_is_aggregate(root) {
            let (nm, m) = (
                self.sym_str(root).to_string(),
                self.sym_str(method).to_string(),
            );
            self.error(
                recv.span,
                "E0304",
                format!(
                    "cannot call the `mut self` method `{m}` on the non-`mut` parameter `{nm}`; an \
                     aggregate parameter is passed by reference, so this would mutate the caller's \
                     value — add `mut` to the parameter (`fn …(mut {nm}: …)`)"
                ),
            );
        }
    }

    /// Type a call that resolved to the concrete function `target`, with `recv` the receiver's type
    /// when the call has one (it is then argument 0).
    fn type_resolved_method(
        &mut self,
        e: &Expr,
        callee: &Expr,
        target: Symbol,
        recv: Option<Ty>,
        generic_args: &[TypeExpr],
        args: &[Expr],
    ) -> Ty {
        let Some(DefKind::Fn(sig)) = self.defs.lookup(target).map(|d| d.kind.clone()) else {
            for a in args {
                self.type_expr(a);
            }
            return Ty::Error;
        };
        self.types.insert(
            callee.id,
            Ty::Fn {
                params: sig.params.clone(),
                ret: Box::new(sig.ret.clone()),
            },
        );
        let offset = usize::from(recv.is_some());
        let mut arg_tys: Vec<Ty> = Vec::with_capacity(args.len() + offset);
        arg_tys.extend(recv);
        for a in args {
            let t = self.type_expr(a);
            arg_tys.push(t);
        }
        self.adapt_scalar_literals(&sig.params, args, &mut arg_tys, offset);
        // The receiver is no function value; check the written arguments against the parameters
        // after it.
        let rest = FnSig {
            generics: sig.generics.clone(),
            params: sig.params.get(offset..).unwrap_or(&[]).to_vec(),
            ret: sig.ret.clone(),
        };
        self.check_fn_value_args(&rest, args, arg_tys.get(offset..).unwrap_or(&[]));
        self.check_call_bounds(target, &sig, generic_args, &arg_tys, e.span);
        let receiver = match &callee.kind {
            ExprKind::Field { base, .. } if offset == 1 => Some(base.span),
            _ => None,
        };
        let site = CallSite::new(Some(target), receiver, args);
        self.check_fn_call(&sig, generic_args, &arg_tys, e.span, callee.id, &site)
    }

    /// Type a call through the trait bounds of type parameter `g`: `x.m(..)` with `x: g`
    /// (`recv = Some(x)`) or `g::m(..)`. Recorded with no target; `mir_build` resolves it against
    /// each instantiation.
    fn type_bound_call(
        &mut self,
        e: &Expr,
        callee: &Expr,
        g: Symbol,
        method: Symbol,
        recv: Option<&Expr>,
        args: &[Expr],
    ) -> Ty {
        let Some(tm) = self.bound_method(g, method) else {
            for a in args {
                self.type_expr(a);
            }
            let (m, gs) = (
                self.sym_str(method).to_string(),
                self.sym_str(g).to_string(),
            );
            self.error(
                callee.span,
                "E0315",
                format!(
                    "no method named `{m}` in the trait bounds of type parameter `{gs}` (bound it: \
                     `{gs}: SomeTrait`)"
                ),
            );
            return Ty::Error;
        };
        if tm.has_self != recv.is_some() {
            for a in args {
                self.type_expr(a);
            }
            let m = self.sym_str(method).to_string();
            let how = if tm.has_self {
                "takes `self`; call it on a value, `x.m(..)`"
            } else {
                "takes no `self`; call it on the type, `T::m(..)`"
            };
            self.error(callee.span, "E0315", format!("trait method `{m}` {how}"));
            return Ty::Error;
        }
        if let Some(r) = recv.filter(|_| tm.mut_self) {
            self.check_mut_receiver(r, method);
        }
        // `T::m(..)` with no receiver is resolved per instantiation from the argument types, so
        // `T` must be the type of something the call's caller passes in. That is a property of the
        // generic signature, decided by the generic pass: an instance re-check (`inst_subst`
        // non-empty) sees the parameters already substituted and must not re-ask it.
        if recv.is_none() && self.inst_subst.is_empty() && !self.generic_reaches_params(g) {
            for a in args {
                self.type_expr(a);
            }
            let gs = self.sym_str(g).to_string();
            self.error(
                callee.span,
                "E0315",
                format!(
                    "`{gs}::…` needs `{gs}` to be the type of a parameter of this function, which \
                     is how each call binds it"
                ),
            );
            return Ty::Error;
        }
        self.types.insert(callee.id, Ty::Unknown);
        let gty = Ty::Named(g);
        let self_sym = self.interner.get("Self");
        let sub = |t: &Ty| self_sym.map_or(t.clone(), |s| subst_self(t, s, &gty));
        let mut params: Vec<Ty> = Vec::with_capacity(tm.params.len() + 1);
        if recv.is_some() {
            params.push(gty.clone());
        }
        params.extend(tm.params.iter().map(&sub));
        let ret = sub(&tm.ret);
        let offset = usize::from(recv.is_some());
        let mut arg_tys: Vec<Ty> = Vec::with_capacity(args.len() + offset);
        if recv.is_some() {
            arg_tys.push(gty.clone());
        }
        for a in args {
            let t = self.type_expr(a);
            arg_tys.push(t);
        }
        self.adapt_scalar_literals(&params, args, &mut arg_tys, offset);
        let sig = FnSig {
            generics: Vec::new(),
            params,
            ret,
        };
        let site = CallSite::new(None, recv.map(|r| r.span), args);
        let out = self.check_fn_call(&sig, &[], &arg_tys, e.span, callee.id, &site);
        self.methods.resolution.calls.insert(
            e.id,
            MethodCall {
                target: None,
                method,
                receiver: recv.is_some(),
                deref: false,
            },
        );
        out
    }

    /// Is `g.method(..)` a call this module must resolve through `g`'s trait bounds? Yes when `g`
    /// has bounds, or when some type in the program has a method of that name (forgetting the
    /// bound is then the likely mistake, `E0315`). An unbounded type parameter calling a name no
    /// `impl` defines stays the lenient unmodelled call it always was (a `.len()` on a parameter
    /// that every instantiation binds to a slice, say).
    fn may_be_bound_method(&self, g: Symbol, method: Symbol) -> bool {
        self.methods.cur_bounds.contains_key(&g)
            || self
                .methods
                .resolution
                .table
                .keys()
                .any(|(_, m)| *m == method)
    }

    /// Does generic `g` appear in a parameter type of the function whose body is being checked?
    fn generic_reaches_params(&self, g: Symbol) -> bool {
        self.param_tys.values().any(|t| mentions(t, g))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::{Interner, SourceId};

    /// Check `src` (which must parse), returning the error codes and the published method tables.
    fn analyze(src: &str) -> (Vec<&'static str>, MethodResolution, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        let (res, diags) = crate::check(&module, &interner);
        let codes = diags
            .iter()
            .filter(|d| d.is_error())
            .filter_map(|d| d.code)
            .collect();
        (codes, res.methods, interner)
    }

    fn codes(src: &str) -> Vec<&'static str> {
        analyze(src).0
    }

    const VEC2: &str = "struct Vec2 { x: f32, y: f32 } \
        impl Vec2 { \
            fn new(x: f32, y: f32) -> Self { return Self { x: x, y: y }; } \
            fn dot(self, o: Vec2) -> f32 { return self.x * o.x + self.y * o.y; } \
            fn scale(&mut self, k: f32) { self.x = self.x * k; } \
        }";

    #[test]
    fn methods_and_associated_functions_resolve() {
        let src = format!(
            "{VEC2} fn main() -> i32 {{ let a = Vec2::new(1.0, 2.0); let mut b = a; \
             b.scale(2.0); let d: f32 = a.dot(b); return d as i32; }}"
        );
        let (errs, res, interner) = analyze(&src);
        assert!(errs.is_empty(), "{errs:?}");
        let mut targets: Vec<String> = res
            .calls
            .values()
            .map(|c| interner.resolve(c.target.expect("concrete")).to_string())
            .collect();
        targets.sort();
        assert_eq!(targets, vec!["Vec2.dot", "Vec2.new", "Vec2.scale"]);
        assert!(res.calls.values().any(|c| !c.receiver));
    }

    #[test]
    fn a_method_through_a_reference_auto_derefs() {
        let src = format!("{VEC2} fn f(p: &Vec2) -> f32 {{ return p.dot(*p); }}");
        let (errs, res, _) = analyze(&src);
        assert!(errs.is_empty(), "{errs:?}");
        assert!(res.calls.values().any(|c| c.deref));
    }

    #[test]
    fn an_unknown_method_is_a_resolution_error() {
        assert!(codes(&format!(
            "{VEC2} fn f(a: Vec2) -> f32 {{ return a.nrom(); }}"
        ))
        .contains(&"E0315"));
        assert!(
            codes(&format!("{VEC2} fn f() -> f32 {{ return Vec2::zero(); }}")).contains(&"E0315")
        );
        // `new` takes no `self`: it is not a method.
        assert!(codes(&format!(
            "{VEC2} fn f(a: Vec2) -> Vec2 {{ return a.new(1.0, 2.0); }}"
        ))
        .contains(&"E0315"));
        assert!(codes("impl Nope { fn f() {} } fn main() -> i32 { return 0; }").contains(&"E0315"));
    }

    /// `Enum::f(..)` naming a function of an `impl Enum` block (not a variant) is an associated
    /// function call, and `Enum::m(x)` a method called with its receiver spelled out; neither is the
    /// `E0301` unknown-variant error the variant-constructor check would otherwise report.
    #[test]
    fn an_enum_associated_function_is_not_a_variant() {
        let src = "enum Dir { N, S } \
                   impl Dir { fn first() -> Self { return Dir::N; } \
                              fn flip(self) -> Dir { return match self { Dir::N => Dir::S, \
                              Dir::S => Dir::N }; } } \
                   fn main() -> i32 { let d = Dir::flip(Dir::first()); \
                                      let e = Dir::first().flip(); return 0; }";
        let (errs, res, interner) = analyze(src);
        assert!(errs.is_empty(), "{errs:?}");
        let mut targets: Vec<String> = res
            .calls
            .values()
            .map(|c| interner.resolve(c.target.expect("concrete")).to_string())
            .collect();
        targets.sort();
        assert_eq!(
            targets,
            vec!["Dir.first", "Dir.first", "Dir.flip", "Dir.flip"]
        );
        assert!(
            codes("enum Dir { N, S } fn f() -> i32 { let d = Dir::W(); return 0; }")
                .contains(&"E0301")
        );
    }

    /// A method call on a struct-typed top-level `const` is a receiver call, not a namespace path.
    #[test]
    fn a_const_receiver_is_a_value() {
        let src = format!(
            "{VEC2} const O: Vec2 = Vec2 {{ x: 1.0, y: 2.0 }}; \
             fn f() -> f32 {{ return O.dot(O); }}"
        );
        let (errs, res, _) = analyze(&src);
        assert!(errs.is_empty(), "{errs:?}");
        assert!(res.calls.values().any(|c| c.receiver));
    }

    /// A `&mut self` method writes its receiver in place, so calling it on a non-`mut` aggregate
    /// parameter would mutate the caller's value — `E0304`, like `v.x = …` through that parameter.
    #[test]
    fn a_mut_self_method_needs_a_mutable_aggregate_parameter() {
        let immutable = format!("{VEC2} fn f(v: Vec2) {{ v.scale(2.0); }}");
        assert!(
            codes(&immutable).contains(&"E0304"),
            "{:?}",
            codes(&immutable)
        );
        let mutable = format!("{VEC2} fn f(mut v: Vec2) {{ v.scale(2.0); }}");
        assert!(codes(&mutable).is_empty(), "{:?}", codes(&mutable));
        // A read-only method on an immutable parameter is fine.
        let read = format!("{VEC2} fn f(v: Vec2) -> f32 {{ return v.dot(v); }}");
        assert!(codes(&read).is_empty(), "{:?}", codes(&read));
    }

    #[test]
    fn a_method_argument_is_type_checked() {
        assert!(codes(&format!(
            "{VEC2} fn f(a: Vec2) -> f32 {{ return a.dot(1.0, 2.0); }}"
        ))
        .contains(&"E0503"));
    }

    const SHAPE: &str = "trait Shape { fn area(self) -> f32; fn unit() -> Self; } \
        struct Sq { s: f32 } struct Circ { r: f32 } \
        impl Shape for Sq { \
            fn area(self) -> f32 { return self.s * self.s; } \
            fn unit() -> Self { return Sq { s: 1.0 }; } \
        }";

    #[test]
    fn a_bounded_generic_calls_trait_methods() {
        let src = format!(
            "{SHAPE} fn twice<T: Shape>(x: T) -> f32 {{ let u = T::unit(); \
             return x.area() + u.area(); }} \
             fn main() -> i32 {{ let q = Sq {{ s: 3.0 }}; return twice(q) as i32; }}"
        );
        let (errs, res, _) = analyze(&src);
        assert!(errs.is_empty(), "{errs:?}");
        // The three calls inside `twice` are trait-bound (no single target).
        assert_eq!(res.calls.values().filter(|c| c.target.is_none()).count(), 3);
    }

    #[test]
    fn a_where_clause_is_a_bound() {
        let src = format!(
            "{SHAPE} fn area_of<T>(x: T) -> f32 where T: Shape {{ return x.area(); }} \
             fn main() -> i32 {{ return area_of(Sq {{ s: 2.0 }}) as i32; }}"
        );
        assert!(codes(&src).is_empty(), "{:?}", codes(&src));
    }

    #[test]
    fn an_unsatisfied_bound_is_rejected_at_the_call() {
        let src = format!(
            "{SHAPE} fn area_of<T: Shape>(x: T) -> f32 {{ return x.area(); }} \
             fn main() -> i32 {{ return area_of(Circ {{ r: 1.0 }}) as i32; }}"
        );
        assert!(codes(&src).contains(&"E0410"), "{:?}", codes(&src));
        let scalar = format!(
            "{SHAPE} fn area_of<T: Shape>(x: T) -> f32 {{ return x.area(); }} \
             fn main() -> i32 {{ return area_of(2.0) as i32; }}"
        );
        assert!(codes(&scalar).contains(&"E0410"));
    }

    #[test]
    fn an_unbounded_generic_has_no_methods() {
        // Some type has an `area`, so the missing bound is the mistake.
        let src = "trait Shape { fn area(self) -> f32; } struct Sq { s: f32 } \
                   impl Shape for Sq { fn area(self) -> f32 { return self.s; } } \
                   fn area_of<T>(x: T) -> f32 { return x.area(); }";
        assert!(codes(src).contains(&"E0315"));
        // A name no `impl` defines stays the lenient unmodelled call it always was.
        let lenient = "fn n<T>(x: T) -> i64 { return x.len(); }";
        assert!(codes(lenient).is_empty(), "{:?}", codes(lenient));
    }

    #[test]
    fn an_impl_must_match_its_trait() {
        let missing = "trait Shape { fn area(self) -> f32; fn sides(self) -> i32; } \
                       struct Sq { s: f32 } \
                       impl Shape for Sq { fn area(self) -> f32 { return self.s; } }";
        assert!(codes(missing).contains(&"E0410"));
        let extra = "trait Shape { fn area(self) -> f32; } struct Sq { s: f32 } \
                     impl Shape for Sq { fn area(self) -> f32 { return self.s; } \
                     fn perimeter(self) -> f32 { return self.s; } }";
        assert!(codes(extra).contains(&"E0410"));
        let wrong = "trait Shape { fn area(self) -> f32; } struct Sq { s: f32 } \
                     impl Shape for Sq { fn area(self) -> i32 { return 1; } }";
        assert!(codes(wrong).contains(&"E0410"));
        let default_body = "trait Shape { fn area(self) -> f32 { return 1.0; } }";
        assert!(codes(default_body).contains(&"E0410"));
        let unknown = "struct Sq { s: f32 } impl Shape for Sq { fn area(self) -> f32 { \
                       return self.s; } }";
        assert!(codes(unknown).contains(&"E0315"));
    }

    const APPLY: &str = "fn square(x: f32) -> f32 { return x * x; } \
        fn to_i(x: f32) -> i32 { return x as i32; } \
        fn id<T>(x: T) -> T { return x; } \
        fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); }";

    #[test]
    fn a_function_parameter_takes_a_named_function() {
        let ok = format!("{APPLY} fn main() -> i32 {{ return apply(square, 3.0) as i32; }}");
        assert!(codes(&ok).is_empty(), "{:?}", codes(&ok));
        let forward = format!(
            "{APPLY} fn twice(g: fn(f32) -> f32, x: f32) -> f32 {{ return apply(g, apply(g, x)); }}"
        );
        assert!(codes(&forward).is_empty(), "{:?}", codes(&forward));
        let wrong = format!("{APPLY} fn main() -> i32 {{ return apply(to_i, 3.0) as i32; }}");
        assert!(codes(&wrong).contains(&"E0401"));
        let generic = format!("{APPLY} fn main() -> i32 {{ return apply(id, 3.0) as i32; }}");
        assert!(codes(&generic).contains(&"E0411"));
        let value = format!("{APPLY} fn main() -> i32 {{ return apply(2.0, 3.0) as i32; }}");
        assert!(codes(&value).contains(&"E0411"));
    }

    #[test]
    fn a_function_type_is_only_a_parameter_type() {
        let bound = format!("{APPLY} fn main() -> i32 {{ let g = square; return 0; }}");
        assert!(codes(&bound).contains(&"E0411"));
        let annotated =
            format!("{APPLY} fn main() -> i32 {{ let g: fn(f32) -> f32 = square; return 0; }}");
        assert!(codes(&annotated).contains(&"E0411"));
        let field = "struct S { f: fn(f32) -> f32 }";
        assert!(codes(field).contains(&"E0411"));
        let ret = format!("{APPLY} fn pick() -> fn(f32) -> f32 {{ return square; }}");
        assert!(codes(&ret).contains(&"E0411"));
        let passed = format!(
            "{APPLY} fn k(x: f32) -> f32 {{ return x; }} \
                              fn main() -> i32 {{ return k(square) as i32; }}"
        );
        assert!(codes(&passed).contains(&"E0411"));
    }

    #[test]
    fn a_function_parameter_is_neither_mut_nor_a_builtin_name() {
        let m = "fn apply(mut f: fn(f32) -> f32, x: f32) -> f32 { return f(x); }";
        assert!(codes(m).contains(&"E0411"));
        let builtin = "fn apply(gelu: fn(f32) -> f32, x: f32) -> f32 { return gelu(x); }";
        assert!(codes(builtin).contains(&"E0411"));
    }
}
