//! Lowering for **methods, trait-bound calls and function-typed parameters** — all static dispatch.
//!
//! `wukong_sema::methods` resolves every method call and publishes the method tables
//! ([`wukong_sema::MethodResolution`]). Three things happen here:
//!
//! * [`desugar_module`] rewrites every method call whose target sema already knows — `x.m(a)` on a
//!   concrete type and `Type::m(a)` — into the plain call `Type.m(x, a)` *before anything else
//!   looks at the tree*, so the whole-program alias analysis, the monomorphization collector, the
//!   recognizers and the canonicalizers all see an ordinary call. A receiver reached through a
//!   reference or pointer is wrapped in an explicit `*` (an aggregate's `*p` is its base pointer).
//! * A call through a **trait bound** (`x.area()` with `x: T`, `T: Shape`) has no single target:
//!   it is resolved per monomorphized instance, against the instantiated type's method table, by
//!   [`dispatch_in_instance`] (the instance collector) and [`FnLowerer::lower_dispatched_call`]
//!   (the lowerer) — the same rule at both sites.
//! * A **function-typed parameter** is a compile-time-only generic: the callee is specialized per
//!   bound function exactly like a type generic, keyed by the pseudo-generic `$fn$<param>` whose
//!   binding is `Ty::Named(<function>)` (see [`fn_value_keys`]). The parameter has no runtime
//!   value at all: `lower_fn` declares no MIR parameter for it, `lower_call` passes no argument,
//!   and a call through it is a direct call of the bound function.
//! * Given a **run-time function value** instead (a local, a field, an element, a call's result —
//!   anything [`fn_value_of`] cannot name), the key is bound to the parameter's own function type
//!   (`Ty::Fn`, [`is_dynamic_fn`]): that instance declares a `Ptr` parameter for it, the call
//!   passes the pointer, and a call through it is an `Op::CallIndirect`. Taking the address of a
//!   function that has function-typed parameters names its instance with every key bound that
//!   way ([`dynamic_subst`]).
//!
//! Neither kind of indirect call is visible to `alias.rs`, which reasons only about calls it can
//! see. [`dispatch_taint`] therefore marks every function such a call can reach — and everything
//! those call in turn — as not provably alias-free, the conservative answer.

use wukong_ast::{self as ast, Block, Expr, ExprKind, FnDecl, ForIter, Module, NodeId, Stmt};
use wukong_diag::Diagnostic;
use wukong_mir::{MirType, Op, ValueId};
use wukong_sema::{DefKind, FnSig, SemaResult};
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Interner, Span, Symbol};
use wukong_types::Ty;

use crate::{subst_ty, FnLowerer};

/// The ids this pass mints for the nodes it synthesizes come from a band of their own, below the
/// one `tindex` mints from (it counts down from `u32::MAX - 1` to `u32::MAX / 2`) and far above
/// anything the parser allocates (it counts up from 0).
const FRESH_ID_TOP: u32 = u32::MAX / 2 - 1;
const FRESH_ID_FLOOR: u32 = u32::MAX / 4;

/// Whether a parameter of type `t` has a runtime value. A function-typed parameter does not: it is
/// specialized away, so `lower_fn` declares no MIR parameter for it and `lower_call` passes none —
/// unless the instance binds it to a run-time function value ([`is_dynamic_fn`]).
pub(crate) fn param_has_value(t: &Ty) -> bool {
    !matches!(t, Ty::Fn { .. })
}

/// Whether the function-typed parameter `name` of the instance whose substitution is `subst` is
/// bound to a run-time function value — the instance receives a pointer for it and calls through
/// it — rather than to a function known at compile time.
pub(crate) fn is_dynamic_fn(
    name: Symbol,
    subst: &HashMap<Symbol, Ty>,
    interner: &Interner,
) -> bool {
    if subst.is_empty() {
        return false;
    }
    let Some(key) = interner.get(&fn_value_key(interner.resolve(name))) else {
        return false;
    };
    matches!(subst.get(&key), Some(Ty::Fn { .. }))
}

/// The binding of a function-typed parameter of declared type `pty` given the argument `arg`: the
/// function it names (`Ty::Named`, specialized at compile time), or — for a run-time function
/// value — the parameter's own function type with the call's type bindings `bound` applied
/// (`Ty::Fn`, [`is_dynamic_fn`]). `outer` is the caller's substitution.
pub(crate) fn fn_arg_binding(
    arg: &Expr,
    pty: &Ty,
    bound: &HashMap<Symbol, Ty>,
    outer: &HashMap<Symbol, Ty>,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<Ty> {
    if let Some(f) = fn_value_of(arg, outer, sema, interner) {
        return Some(Ty::Named(f));
    }
    match subst_ty(pty, bound) {
        t @ Ty::Fn { .. } => Some(t),
        _ => None,
    }
}

/// The substitution under which a function with function-typed parameters is compiled when its
/// ADDRESS is taken: every such parameter bound to a run-time function value (its own type). `None`
/// for a function with type generics, which has no single address (sema refuses one, `E0429`).
pub(crate) fn dynamic_subst(
    keys: &[(usize, Symbol)],
    sig: &FnSig,
    tg: &[Symbol],
) -> Option<HashMap<Symbol, Ty>> {
    let mut subst: HashMap<Symbol, Ty> = HashMap::default();
    for &(i, key) in keys {
        subst.insert(key, sig.params.get(i)?.clone());
    }
    tg.iter().all(|g| subst.contains_key(g)).then_some(subst)
}

/// The pseudo-generic a function-typed parameter named `param` is specialized under.
pub(crate) fn fn_value_key(param: &str) -> String {
    format!("$fn${param}")
}

/// For a function with function-typed parameters: each such parameter's index and its
/// pseudo-generic key (interned here). Empty for every other function.
pub(crate) fn fn_value_keys(
    f: &FnDecl,
    sig: &FnSig,
    interner: &mut Interner,
) -> Vec<(usize, Symbol)> {
    let mut keys = Vec::new();
    for (i, (p, t)) in f.params.iter().zip(&sig.params).enumerate() {
        if matches!(t, Ty::Fn { .. }) {
            let name = fn_value_key(interner.resolve(p.name.sym));
            keys.push((i, interner.intern(&name)));
        }
    }
    keys
}

/// The function a function-typed parameter `name` is bound to in the instance whose substitution
/// is `subst`, if `name` is such a parameter there.
pub(crate) fn bound_fn(
    name: Symbol,
    subst: &HashMap<Symbol, Ty>,
    interner: &Interner,
) -> Option<Symbol> {
    if subst.is_empty() {
        return None;
    }
    let key = interner.get(&fn_value_key(interner.resolve(name)))?;
    match subst.get(&key)? {
        Ty::Named(f) => Some(*f),
        _ => None,
    }
}

/// The function an argument passed to a function-typed parameter denotes: the caller's own bound
/// function-typed parameter, or a top-level function named directly (a lifted closure's value
/// `(L, captures..)` names `L`, `closures.rs`).
pub(crate) fn fn_value_of(
    arg: &Expr,
    subst: &HashMap<Symbol, Ty>,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<Symbol> {
    let code = crate::closures::closure_code(arg);
    let ExprKind::Path(p) = &code.kind else {
        return None;
    };
    if !p.is_single() {
        return None;
    }
    // A `let`, pattern or `for` binding holds a run-time function value — even where it shadows a
    // function-typed parameter this instance binds at compile time (`let f = cube;` in
    // `outer$sq`), so the parameter's binding must not be asked first.
    if sema.fn_value_lets.contains(&code.id) {
        return None;
    }
    let name = p.first().sym;
    if let Some(f) = bound_fn(name, subst, interner) {
        return Some(f);
    }
    // A local holding a run-time function value — even one named like a top-level function.
    if sema.fn_value_locals.contains(&code.id) {
        return None;
    }
    matches!(
        sema.defs.lookup(name).map(|d| &d.kind),
        Some(DefKind::Fn(_))
    )
    .then_some(name)
}

/// For a call through a trait bound, the function it reaches in the instance whose substitution is
/// `subst`, and the call's full argument list (the receiver first). `None` for every other call.
pub(crate) fn dispatch_in_instance(
    e: &Expr,
    callee: &Expr,
    args: &[Expr],
    subst: &HashMap<Symbol, Ty>,
    sema: &SemaResult,
) -> Option<(Symbol, Vec<Expr>)> {
    let mc = sema.methods.calls.get(&e.id)?;
    if mc.target.is_some() {
        return None;
    }
    let ExprKind::Field { base, .. } = &callee.kind else {
        return None;
    };
    let ty = if mc.receiver {
        subst_ty(sema.types.get(&base.id)?, subst)
    } else {
        let ExprKind::Path(p) = &base.kind else {
            return None;
        };
        subst.get(&p.first().sym)?.clone()
    };
    let target = sema.methods.lookup(&ty, mc.method)?;
    let mut full = Vec::with_capacity(args.len() + 1);
    if mc.receiver {
        full.push((**base).clone());
    }
    full.extend(args.iter().cloned());
    Some((target, full))
}

/// Rewrite every method call with a known target into the plain call of its function. Returns the
/// rewritten module and the types of the nodes it synthesized, or `None` (cloning nothing) for a
/// module with no such call.
pub(crate) fn desugar_module(
    module: &Module,
    sema: &SemaResult,
) -> Option<(Module, HashMap<NodeId, Ty>)> {
    if !sema.methods.calls.values().any(|c| c.target.is_some()) {
        return None;
    }
    let mut d = Desugar {
        sema,
        extra: HashMap::default(),
        next_id: FRESH_ID_TOP,
        failed: false,
    };
    let mut out = module.clone();
    for item in &mut out.items {
        if let ast::ItemKind::Fn(f) = &mut item.kind {
            if let Some(body) = &mut f.body {
                d.block(body);
            }
        }
    }
    if d.failed {
        return None;
    }
    Some((out, d.extra))
}

struct Desugar<'a> {
    sema: &'a SemaResult,
    extra: HashMap<NodeId, Ty>,
    next_id: u32,
    /// The id band ran out (unreachable in practice): the caller keeps the original module.
    failed: bool,
}

impl Desugar<'_> {
    fn fresh(&mut self) -> NodeId {
        if self.next_id <= FRESH_ID_FLOOR {
            self.failed = true;
            return NodeId::DUMMY;
        }
        let id = NodeId(self.next_id);
        self.next_id -= 1;
        id
    }

    fn block(&mut self, b: &mut Block) {
        for s in &mut b.stmts {
            self.stmt(s);
        }
        if let Some(t) = &mut b.tail {
            self.expr(t);
        }
    }

    fn stmt(&mut self, s: &mut Stmt) {
        match &mut s.kind {
            ast::StmtKind::Let { init, .. } => {
                if let Some(e) = init {
                    self.expr(e);
                }
            }
            ast::StmtKind::Assign { target, value, .. } => {
                self.expr(target);
                self.expr(value);
            }
            ast::StmtKind::Expr(e) | ast::StmtKind::Defer(e) => self.expr(e),
            ast::StmtKind::Return(e) | ast::StmtKind::Break(_, e) => {
                if let Some(e) = e {
                    self.expr(e);
                }
            }
            ast::StmtKind::Continue(_) => {}
            ast::StmtKind::While { cond, body, .. } => {
                self.expr(cond);
                self.block(body);
            }
            ast::StmtKind::For { iter, body, .. } => {
                match iter {
                    ForIter::Range {
                        start, end, step, ..
                    } => {
                        self.expr(start);
                        if let Some(e) = end {
                            self.expr(e);
                        }
                        if let Some(e) = step {
                            self.expr(e);
                        }
                    }
                    ForIter::Expr(e) => self.expr(e),
                }
                self.block(body);
            }
        }
    }

    fn expr(&mut self, e: &mut Expr) {
        match &mut e.kind {
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::Path(_)
            | ExprKind::SizeOf(_)
            | ExprKind::AlignOf(_)
            | ExprKind::Closure { .. } => {}
            ExprKind::Unary { expr, .. }
            | ExprKind::Field { base: expr, .. }
            | ExprKind::TupleField { base: expr, .. }
            | ExprKind::Cast { expr, .. } => self.expr(expr),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            ExprKind::Call { callee, args, .. } => {
                self.expr(callee);
                for a in args {
                    self.expr(a);
                }
            }
            ExprKind::Index { base, indices } => {
                self.expr(base);
                for ix in indices.iter_mut() {
                    self.expr(ix);
                }
            }
            ExprKind::StructLit { fields, rest, .. } => {
                for f in fields {
                    self.expr(&mut f.value);
                }
                if let Some(r) = rest {
                    self.expr(r);
                }
            }
            ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
                for x in xs {
                    self.expr(x);
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.expr(value);
                self.expr(count);
            }
            ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => self.block(b),
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.expr(cond);
                self.block(then_branch);
                if let Some(x) = else_branch {
                    self.expr(x);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee);
                for a in arms {
                    if let Some(g) = &mut a.guard {
                        self.expr(g);
                    }
                    self.expr(&mut a.body);
                }
            }
            ExprKind::Let { scrutinee, .. } => self.expr(scrutinee),
        }

        // Children first (a receiver may itself be a method call); then this call.
        let Some(mc) = self.sema.methods.calls.get(&e.id) else {
            return;
        };
        let (Some(target), receiver, deref) = (mc.target, mc.receiver, mc.deref) else {
            return;
        };
        let ExprKind::Call { callee, args, .. } = &mut e.kind else {
            return;
        };
        let span = callee.span;
        let path = Expr {
            id: callee.id,
            kind: ExprKind::Path(ast::Path {
                segments: vec![ast::Ident { sym: target, span }],
                span,
            }),
            span,
        };
        let old = std::mem::replace(&mut **callee, path);
        if !receiver {
            return;
        }
        let ExprKind::Field { base, .. } = old.kind else {
            return;
        };
        let mut recv = *base;
        if deref {
            let pointee = match self.sema.types.get(&recv.id) {
                Some(Ty::Ref { pointee, .. } | Ty::Ptr { pointee, .. }) => (**pointee).clone(),
                _ => Ty::Unknown,
            };
            let id = self.fresh();
            self.extra.insert(id, pointee);
            let rspan = recv.span;
            recv = Expr {
                id,
                kind: ExprKind::Unary {
                    op: ast::UnOp::Deref,
                    expr: Box::new(recv),
                },
                span: rspan,
            };
        }
        args.insert(0, recv);
    }
}

/// Every function a call this module cannot see may reach — the targets of calls through trait
/// bounds, the functions passed to function-typed parameters, and every function used as a
/// run-time value (an `Op::CallIndirect` may reach any of them) — closed over the calls those
/// functions make. `alias.rs` cannot attribute such a call's arguments to its callee, so none of
/// these functions may claim its buffer parameters are provably disjoint.
pub(crate) fn dispatch_taint(module: &Module, sema: &SemaResult) -> HashSet<Symbol> {
    let mut bodies: HashMap<Symbol, &Block> = HashMap::default();
    for item in &module.items {
        if let ast::ItemKind::Fn(f) = &item.kind {
            if let Some(b) = &f.body {
                bodies.insert(f.name.sym, b);
            }
        }
    }
    // Roots: every function a trait-bound call could reach, and every function passed as a value.
    let mut roots: Vec<Symbol> = Vec::new();
    let bound_methods: HashSet<Symbol> = sema
        .methods
        .calls
        .values()
        .filter(|c| c.target.is_none())
        .map(|c| c.method)
        .collect();
    if !bound_methods.is_empty() {
        let mut keys: Vec<(&(Symbol, Symbol), &Symbol)> = sema.methods.table.iter().collect();
        keys.sort_by_key(|((t, m), _)| (t.0, m.0));
        for ((_, m), f) in keys {
            if bound_methods.contains(m) {
                roots.push(*f);
            }
        }
    }
    let mut scan = CallScan {
        sema,
        calls: Vec::new(),
        values: Vec::new(),
    };
    for b in bodies.values() {
        scan.block(b);
    }
    roots.extend(scan.values.iter().copied());
    if roots.is_empty() {
        return HashSet::default();
    }
    let mut taint: HashSet<Symbol> = HashSet::default();
    while let Some(f) = roots.pop() {
        if !taint.insert(f) {
            continue;
        }
        if let Some(b) = bodies.get(&f) {
            let mut inner = CallScan {
                sema,
                calls: Vec::new(),
                values: Vec::new(),
            };
            inner.block(b);
            roots.extend(inner.calls);
            roots.extend(inner.values);
        }
    }
    taint
}

/// The functions a body calls directly (by name, or through a method call with a known target, or
/// any function a trait-bound call could reach), and the functions it passes as values.
struct CallScan<'a> {
    sema: &'a SemaResult,
    calls: Vec<Symbol>,
    values: Vec<Symbol>,
}

impl CallScan<'_> {
    fn is_fn(&self, s: Symbol) -> bool {
        matches!(
            self.sema.defs.lookup(s).map(|d| &d.kind),
            Some(DefKind::Fn(_))
        )
    }

    fn block(&mut self, b: &Block) {
        for s in &b.stmts {
            match &s.kind {
                ast::StmtKind::Let { init, .. } => {
                    if let Some(e) = init {
                        self.expr(e);
                    }
                }
                ast::StmtKind::Assign { target, value, .. } => {
                    self.expr(target);
                    self.expr(value);
                }
                ast::StmtKind::Expr(e) | ast::StmtKind::Defer(e) => self.expr(e),
                ast::StmtKind::Return(e) | ast::StmtKind::Break(_, e) => {
                    if let Some(e) = e {
                        self.expr(e);
                    }
                }
                ast::StmtKind::Continue(_) => {}
                ast::StmtKind::While { cond, body, .. } => {
                    self.expr(cond);
                    self.block(body);
                }
                ast::StmtKind::For { iter, body, .. } => {
                    match iter {
                        ForIter::Range {
                            start, end, step, ..
                        } => {
                            self.expr(start);
                            if let Some(e) = end {
                                self.expr(e);
                            }
                            if let Some(e) = step {
                                self.expr(e);
                            }
                        }
                        ForIter::Expr(e) => self.expr(e),
                    }
                    self.block(body);
                }
            }
        }
        if let Some(t) = &b.tail {
            self.expr(t);
        }
    }

    fn expr(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::SizeOf(_)
            | ExprKind::AlignOf(_)
            | ExprKind::Closure { .. } => {}
            // A function used as a value (stored, returned, passed on): a call through the
            // pointer may reach it with any arguments, so it is a root like a passed function. A
            // local that shares a function's name (`let mut loss = 0.0;` in `fn loss`) is not.
            ExprKind::Path(p) => {
                if p.is_single()
                    && self.is_fn(p.first().sym)
                    && matches!(self.sema.types.get(&e.id), Some(Ty::Fn { .. }))
                    && !self.sema.fn_value_locals.contains(&e.id)
                {
                    self.values.push(p.first().sym);
                }
            }
            ExprKind::Unary { expr, .. }
            | ExprKind::Field { base: expr, .. }
            | ExprKind::TupleField { base: expr, .. }
            | ExprKind::Cast { expr, .. } => self.expr(expr),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            ExprKind::Call { callee, args, .. } => {
                if let ExprKind::Path(p) = &callee.kind {
                    if p.is_single() && self.is_fn(p.first().sym) {
                        self.calls.push(p.first().sym);
                    }
                } else {
                    self.expr(callee);
                }
                if let Some(mc) = self.sema.methods.calls.get(&e.id) {
                    match mc.target {
                        Some(t) => self.calls.push(t),
                        None => {
                            let mut hits: Vec<Symbol> = self
                                .sema
                                .methods
                                .table
                                .iter()
                                .filter(|((_, m), _)| *m == mc.method)
                                .map(|(_, f)| *f)
                                .collect();
                            hits.sort_by_key(|s| s.0);
                            self.calls.extend(hits);
                        }
                    }
                }
                for a in args {
                    if let ExprKind::Path(p) = &crate::closures::closure_code(a).kind {
                        if p.is_single() && self.is_fn(p.first().sym) {
                            self.values.push(p.first().sym);
                        }
                    }
                    self.expr(a);
                }
            }
            ExprKind::Index { base, indices } => {
                self.expr(base);
                for ix in indices {
                    self.expr(ix);
                }
            }
            ExprKind::StructLit { fields, rest, .. } => {
                for f in fields {
                    self.expr(&f.value);
                }
                if let Some(r) = rest {
                    self.expr(r);
                }
            }
            ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
                for x in xs {
                    self.expr(x);
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.expr(value);
                self.expr(count);
            }
            ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => self.block(b),
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.expr(cond);
                self.block(then_branch);
                if let Some(x) = else_branch {
                    self.expr(x);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee);
                for a in arms {
                    if let Some(g) = &a.guard {
                        self.expr(g);
                    }
                    self.expr(&a.body);
                }
            }
            ExprKind::Let { scrutinee, .. } => self.expr(scrutinee),
        }
    }
}

impl FnLowerer<'_> {
    /// A call `lower_call` must redirect before its ordinary resolution: a call through a
    /// function-typed parameter (to the function this instance binds), or a call through a trait
    /// bound (to the instantiated type's method). Returns `None` for every other call.
    pub(crate) fn lower_dispatched_call(
        &mut self,
        callee: &Expr,
        generic_args: &[ast::TypeExpr],
        args: &[Expr],
        e: &Expr,
    ) -> Option<ValueId> {
        // A callee this function synthesized is final: re-resolving it could loop (a parameter
        // bound to a function whose name another parameter of the same instance carries).
        if callee.id == NodeId::DUMMY {
            return None;
        }
        let (target, full): (Symbol, Option<Vec<Expr>>) = match &callee.kind {
            ExprKind::Path(p) if p.is_single() => {
                let name = p.first().sym;
                if self.lookup(name).is_some() {
                    return None;
                }
                (bound_fn(name, &self.subst, self.interner)?, None)
            }
            ExprKind::Field { .. } => {
                let (t, full) = dispatch_in_instance(e, callee, args, &self.subst, self.sema)?;
                (t, Some(full))
            }
            _ => return None,
        };
        let span = callee.span;
        let path = Expr {
            id: NodeId::DUMMY,
            kind: ExprKind::Path(ast::Path {
                segments: vec![ast::Ident { sym: target, span }],
                span,
            }),
            span,
        };
        let args = full.as_deref().unwrap_or(args);
        // A call through a parameter bound to a closure with captures passes the environment this
        // instance received for that parameter first: the lifted function's leading parameters
        // (`closures.rs`).
        if let (ExprKind::Path(p), Some(lift)) = (&callee.kind, self.closure_env(target)) {
            let mut with_env = self.env_arg_exprs(p.first().sym, lift, span);
            with_env.extend(args.iter().cloned());
            return Some(self.lower_call(&path, generic_args, &with_env, e));
        }
        Some(self.lower_call(&path, generic_args, args, e))
    }

    /// A call through a run-time function value: a local holding one (a function-typed parameter
    /// this instance binds at compile time has no local, and is `lower_dispatched_call`'s), or any
    /// other callee expression of function type — a field (`s.op(x)`), an element (`ops[i](x)`),
    /// a call's result. Lowered to `Op::CallIndirect` with the ABI a direct call of the target
    /// uses: an aggregate result through a leading destination buffer (`sret`), an aggregate
    /// argument by its base pointer, a function argument as its address. `None` for every other
    /// call.
    pub(crate) fn lower_indirect_call(&mut self, callee: &Expr, args: &[Expr]) -> Option<ValueId> {
        match &callee.kind {
            ExprKind::Path(p) if p.is_single() && self.lookup(p.first().sym).is_some() => {}
            ExprKind::Path(_) => return None,
            _ => {}
        }
        let Ty::Fn { params, ret } = self.expr_ty(callee) else {
            return None;
        };
        let target = self.lower_expr(callee);
        let mut argv: Vec<ValueId> = Vec::with_capacity(args.len() + 1);
        let ret_mir = self.mir_ty_of(&ret);
        let sret =
            matches!(ret_mir, MirType::Array(..)).then(|| self.builder.alloca(ret_mir.clone()));
        argv.extend(sret);
        for (a, pty) in args.iter().zip(&params) {
            let v = self.lower_arg_coerced(a, Some(pty));
            argv.push(v);
        }
        if let Some(dst) = sret {
            self.builder.build_void(Op::CallIndirect {
                callee: target,
                args: argv,
                ret: MirType::Void,
            });
            return Some(dst);
        }
        if ret_mir == MirType::Void {
            self.builder.build_void(Op::CallIndirect {
                callee: target,
                args: argv,
                ret: MirType::Void,
            });
            return Some(self.const_zero(MirType::I32));
        }
        Some(self.builder.build(
            ret_mir.clone(),
            Op::CallIndirect {
                callee: target,
                args: argv,
                ret: ret_mir,
            },
        ))
    }

    /// A function used as a value (`let f = add1;`, a function-typed parameter of this instance
    /// passed on as a value): its address. A function with function-typed parameters is compiled,
    /// for its address, as the instance that takes each of them at run time
    /// (`methods::dynamic_subst`). A parameter bound to a closure that captures has no address that
    /// could carry the captured values (`E0427`). `None` when `name` is no function here.
    pub(crate) fn lower_fn_value(&mut self, name: Symbol, span: Span) -> Option<ValueId> {
        let target = match bound_fn(name, &self.subst, self.interner) {
            Some(f) => {
                if self.closure_env(f).is_some() {
                    let nm = self.interner.resolve(name).to_string();
                    self.diags.push(
                        Diagnostic::error(format!(
                            "`{nm}` is bound to a closure that captures, which cannot be a \
                             run-time function value"
                        ))
                        .with_code("E0427")
                        .primary(span, "used as a value here")
                        .note(
                            "a function value is only the address of code; the values the \
                             closure passed for this parameter captures have nowhere to live in it",
                        )
                        .help(
                            "pass a named function or a closure that captures nothing, or only \
                             call the parameter (and pass it on) instead of using it as a value",
                        ),
                    );
                }
                f
            }
            None => {
                if !matches!(
                    self.sema.defs.lookup(name).map(|d| &d.kind),
                    Some(DefKind::Fn(_))
                ) {
                    return None;
                }
                name
            }
        };
        let addr = self.fn_value_instance(target);
        Some(self.builder.build(MirType::Ptr, Op::FuncAddr(addr)))
    }

    /// The function whose address `f` denotes: `f` itself, or — for a function with
    /// function-typed parameters — its instance taking each of them at run time.
    fn fn_value_instance(&self, f: Symbol) -> Symbol {
        let Some(mono) = self.mono else {
            return f;
        };
        let (Some(keys), Some(DefKind::Fn(sig))) = (
            mono.fn_value_keys.get(&f),
            self.sema.defs.lookup(f).map(|d| &d.kind),
        ) else {
            return f;
        };
        let tg = mono.tg(f);
        methods_dynamic_instance(mono, f, keys, sig, tg, self.interner).unwrap_or(f)
    }
}

/// The registered instance of `f` with every function-typed parameter bound to a run-time value.
fn methods_dynamic_instance(
    mono: &crate::Mono,
    f: Symbol,
    keys: &[(usize, Symbol)],
    sig: &FnSig,
    tg: &[Symbol],
    interner: &Interner,
) -> Option<Symbol> {
    let subst = dynamic_subst(keys, sig, tg)?;
    let canon = crate::canon_type_args(tg, &subst, interner)?;
    mono.instance_of.get(&(f, canon)).copied()
}

#[cfg(test)]
mod tests {
    use crate::lower_program;
    use wukong_span::{Interner, SourceId};

    /// Lower `src` (which must type-check) to verified MIR, returning each function's text.
    fn lower(src: &str) -> Vec<(String, String)> {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (prog, diags) = lower_program(&module, &sema, &mut interner);
        assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
        let errors = wukong_mir::verify::verify_program(&prog);
        assert!(errors.is_empty(), "invalid MIR: {errors:?}");
        prog.funcs
            .iter()
            .map(|f| {
                (
                    interner.resolve(f.name).to_string(),
                    wukong_mir::print::print_function(f, &interner),
                )
            })
            .collect()
    }

    fn body<'a>(funcs: &'a [(String, String)], name: &str) -> &'a str {
        &funcs
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("no function `{name}` in {funcs:?}"))
            .1
    }

    #[test]
    fn a_method_call_is_a_direct_call_of_its_function() {
        let funcs = lower(
            "struct V { x: f32 } \
             impl V { fn new(x: f32) -> Self { return V { x: x }; } \
                      fn get(self) -> f32 { return self.x; } } \
             fn through(p: &V) -> f32 { return p.get(); } \
             fn main() -> i32 { let v = V::new(2.0); return (v.get() + through(&v)) as i32; }",
        );
        let main = body(&funcs, "main");
        assert!(main.contains("call V.new("), "{main}");
        assert!(main.contains("call V.get("), "{main}");
        assert!(body(&funcs, "through").contains("call V.get("));
        assert!(body(&funcs, "V.get").starts_with("fn V.get("));
    }

    #[test]
    fn a_function_parameter_specializes_its_callee() {
        let funcs = lower(
            "fn square(x: f32) -> f32 { return x * x; } \
             fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
             fn twice(g: fn(f32) -> f32, x: f32) -> f32 { return apply(g, apply(g, x)); } \
             fn main() -> i32 { return (apply(square, 3.0) + twice(square, 2.0)) as i32; }",
        );
        // The template is never lowered; each binding is its own instance.
        assert!(!funcs.iter().any(|(n, _)| n == "apply"), "{funcs:?}");
        let inst = body(&funcs, "apply$square");
        // One runtime parameter (`x`): the function parameter has no value.
        assert!(inst.starts_with("fn apply$square(v0 f32)"), "{inst}");
        assert!(inst.contains("call square("), "{inst}");
        let twice = body(&funcs, "twice$square");
        assert_eq!(twice.matches("call apply$square(").count(), 2, "{twice}");
        assert!(body(&funcs, "main").contains("call apply$square("));
    }

    /// A function value is its address; a call through one is a call through the pointer; a
    /// function-typed parameter given a run-time value gets an instance taking the pointer, and
    /// one given a named function keeps its specialized instance.
    #[test]
    fn a_run_time_function_value_is_called_through_its_pointer() {
        let funcs = lower(
            "fn square(x: f32) -> f32 { return x * x; } \
             fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
             fn main() -> i32 { let mut g = square; let a = g(2.0); \
                                return (a + apply(g, 3.0) + apply(square, 1.0)) as i32; }",
        );
        let main = body(&funcs, "main");
        assert!(main.contains("func_addr square"), "{main}");
        assert!(main.contains("call_indirect"), "{main}");
        assert!(main.contains("call apply$fn1$f32$to$f32("), "{main}");
        assert!(main.contains("call apply$square("), "{main}");
        let dynamic = body(&funcs, "apply$fn1$f32$to$f32");
        // The function-typed parameter is a pointer parameter of its own.
        assert!(
            dynamic.starts_with("fn apply$fn1$f32$to$f32(v0 ptr, v1 f32)"),
            "{dynamic}"
        );
        assert!(dynamic.contains("call_indirect"), "{dynamic}");
        assert!(body(&funcs, "apply$square").contains("call square("));
    }

    /// A `let` shadowing a function-typed parameter holds a run-time value in every instance,
    /// including the one specialized for the parameter's function: passed on, it is a pointer.
    #[test]
    fn a_let_shadowing_a_bound_parameter_is_a_run_time_value() {
        let funcs = lower(
            "fn sq(x: i32) -> i32 { return x * x; } \
             fn cube(x: i32) -> i32 { return x * x * x; } \
             fn apply_twice(f: fn(i32) -> i32, x: i32) -> i32 { return f(f(x)); } \
             fn outer(f: fn(i32) -> i32, x: i32) -> i32 { \
                 let f: fn(i32) -> i32 = cube; return apply_twice(f, x); } \
             fn main() -> i32 { return outer(sq, 2); }",
        );
        let outer = body(&funcs, "outer$sq");
        assert!(outer.contains("func_addr cube"), "{outer}");
        assert!(
            outer.contains("call apply_twice$fn1$i32$to$i32("),
            "{outer}"
        );
        assert!(
            !funcs.iter().any(|(n, _)| n == "apply_twice$sq"),
            "{funcs:?}"
        );
    }

    /// A closure that captures a function value is instantiated for the call through the
    /// parameter it is bound to: the call passes the environment (`k`, then the pointer `g`)
    /// first, and the instance collector lines the call's own arguments up after it — with `g`
    /// second, the call's one argument used to land on `g`'s slot and no instance was registered.
    #[test]
    fn a_closure_capturing_a_function_value_is_instantiated_through_a_parameter() {
        let funcs = lower(
            "fn twice(x: i32) -> i32 { return x * 2; } \
             fn apply(f: fn(i32) -> i32, x: i32) -> i32 { return f(x); } \
             fn with(h: fn(fn(fn(i32) -> i32, i32) -> i32, i32) -> i32, k: i32) -> i32 { \
                 return h(apply, k); } \
             fn main() -> i32 { let g: fn(i32) -> i32 = twice; let k = 100; \
                 let c = |x: i32| k + g(x); \
                 return apply(|x| k + g(x), 3) + apply(c, 4) \
                     + with(|a: fn(fn(i32) -> i32, i32) -> i32, y: i32| a(|x| x + k, y), 5); }",
        );
        // `$0` is the `let`-bound `c`, `$1` the literal, `$2` the closure inside `$3`.
        for l in ["main$closure$0", "main$closure$1"] {
            let apply = body(&funcs, &format!("apply${l}"));
            assert!(
                apply.contains(&format!("call {l}$fn1$i32$to$i32(")),
                "{apply}"
            );
            let inst = body(&funcs, &format!("{l}$fn1$i32$to$i32"));
            assert!(inst.contains("call_indirect"), "{inst}");
        }
        // `with`'s closure has its own function-typed parameter `a`, bound to `apply` at compile
        // time through `h(apply, k)` (the environment `k` comes first): a direct call, never the
        // instance that would take `a` at run time.
        let with = body(&funcs, "with$main$closure$3");
        assert!(with.contains("call main$closure$3$apply("), "{with}");
        assert!(
            !funcs
                .iter()
                .any(|(n, _)| n.starts_with("main$closure$3$fn")),
            "{funcs:?}"
        );
    }

    /// The address of a function with a function-typed parameter is its instance taking that
    /// parameter at run time.
    #[test]
    fn the_address_of_a_higher_order_function_is_its_dynamic_instance() {
        let funcs = lower(
            "fn square(x: f32) -> f32 { return x * x; } \
             fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
             fn main() -> i32 { let h: fn(fn(f32) -> f32, f32) -> f32 = apply; \
                                return h(square, 3.0) as i32; }",
        );
        let main = body(&funcs, "main");
        assert!(main.contains("func_addr apply$fn1$f32$to$f32"), "{main}");
        assert!(main.contains("func_addr square"), "{main}");
    }

    /// Every function used as a value may be called through a pointer with any arguments, so its
    /// buffer parameters are not provably disjoint; a local that only shares a function's name is
    /// no such use.
    #[test]
    fn a_function_used_as_a_value_is_dispatch_tainted() {
        let src = "fn fill(mut a: [f32; 4], b: [f32; 4]) { for i in 0..4 { a[i] = b[i]; } } \
                   fn loss(x: [f32; 4]) -> f32 { let mut loss: f32 = 0.0; \
                       for i in 0..4 { loss = loss + x[i]; } return loss; } \
                   fn main() -> i32 { let f: fn([f32; 4], [f32; 4]) = fill; \
                       let mut a: [f32; 4] = [0.0; 4]; let b: [f32; 4] = [1.0; 4]; f(a, b); \
                       return (a[0] + loss(b)) as i32; }";
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "{pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "{sdiags:?}");
        let taint = super::dispatch_taint(&module, &sema);
        assert!(taint.contains(&interner.intern("fill")), "{taint:?}");
        assert!(!taint.contains(&interner.intern("loss")), "{taint:?}");
    }

    #[test]
    fn a_trait_bound_call_reaches_the_instantiated_impl() {
        let funcs = lower(
            "trait Shape { fn area(self) -> f32; } \
             struct Sq { s: f32 } struct Rc { w: f32, h: f32 } \
             impl Shape for Sq { fn area(self) -> f32 { return self.s * self.s; } } \
             impl Shape for Rc { fn area(self) -> f32 { return self.w * self.h; } } \
             fn double<T: Shape>(x: T) -> f32 { return 2.0 * x.area(); } \
             fn main() -> i32 { let a = Sq { s: 2.0 }; let b = Rc { w: 1.0, h: 3.0 }; \
                                return (double(a) + double(b)) as i32; }",
        );
        assert!(body(&funcs, "double$Sq").contains("call Sq.area("));
        assert!(body(&funcs, "double$Rc").contains("call Rc.area("));
    }
}
