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
//!
//! Neither kind of indirect call is visible to `alias.rs`, which reasons only about calls it can
//! see. [`dispatch_taint`] therefore marks every function such a call can reach — and everything
//! those call in turn — as not provably alias-free, the conservative answer.

use wukong_ast::{self as ast, Block, Expr, ExprKind, FnDecl, ForIter, Module, NodeId, Stmt};
use wukong_mir::ValueId;
use wukong_sema::{DefKind, FnSig, SemaResult};
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Interner, Symbol};
use wukong_types::Ty;

use crate::{subst_ty, FnLowerer};

/// The ids this pass mints for the nodes it synthesizes come from a band of their own, below the
/// one `tindex` mints from (it counts down from `u32::MAX - 1` to `u32::MAX / 2`) and far above
/// anything the parser allocates (it counts up from 0).
const FRESH_ID_TOP: u32 = u32::MAX / 2 - 1;
const FRESH_ID_FLOOR: u32 = u32::MAX / 4;

/// Whether a parameter of type `t` has a runtime value. A function-typed parameter does not: it is
/// specialized away, so `lower_fn` declares no MIR parameter for it and `lower_call` passes none.
pub(crate) fn param_has_value(t: &Ty) -> bool {
    !matches!(t, Ty::Fn { .. })
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
    let ExprKind::Path(p) = &crate::closures::closure_code(arg).kind else {
        return None;
    };
    if !p.is_single() {
        return None;
    }
    let name = p.first().sym;
    if let Some(f) = bound_fn(name, subst, interner) {
        return Some(f);
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
/// bounds and the functions passed to function-typed parameters — closed over the calls those
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
