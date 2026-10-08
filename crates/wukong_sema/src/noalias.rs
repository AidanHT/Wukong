//! The `@noalias` parameter promise: what may carry it, and the two rules that keep it honest.
//!
//! `fn k(@noalias mut out: []f32, a: []f32)` promises that, for the duration of a call, the storage
//! `out` points into is reached only through `out` — nothing else the call can name overlaps it.
//! It is C's `restrict` and LLVM's `noalias` argument, and it is the one fact the optimizer cannot
//! derive for itself: two buffer parameters may otherwise be the same caller buffer, so a store to
//! `out[i]` would have to be assumed to clobber `a[i + 1]`. `wukong_mir_build` copies the promise
//! onto the MIR parameter (`wukong_mir::ParamAttrs`), and `wukong_opt::alias`, the loop dependence
//! test, the vectorizer's alias guard, the kernel recognizers and both native backends consume it.
//!
//! A promise only the programmer checks is a miscompile waiting for a typo, so this pass enforces
//! the part of it that is decidable at compile time — `E0316`, the analogue of Rust's rule that a
//! `&mut` argument cannot overlap another argument:
//!
//! 1. **Call sites.** A call to a function with a `@noalias` parameter may not pass that
//!    parameter's argument and any other buffer argument rooted in the same storage: `k(a, a)`,
//!    `k(a, s)` after `let s: []f32 = a`, `k(p.w, p)`, `k(m[0], m[1])`. Disjoint fields of one
//!    struct (`k(p.w, p.b)`) stay legal. Element indices are not compared, so two elements of one
//!    array count as one root — the same conservative line Rust's borrow checker draws.
//! 2. **Rebinding.** A `@noalias` parameter whose value is a view (`[]T`, `*T`, `&T`, a tensor)
//!    cannot be reassigned as a whole: `s = t` would point the promised name at storage the promise
//!    never covered. Writing through it (`s[i] = …`) is the point of the attribute and stays legal; a
//!    fixed array or aggregate parameter is assigned by copy, so reassigning one is legal too.
//!
//! What it does **not** decide: a call whose two arguments are not provably the same storage — two
//! views of unknown origin, a pointer read out of memory, a caller's own unpromised parameters. Such
//! a call compiles, and if the arguments overlap after all the program has broken its promise: the
//! interpreter still computes the sequential meaning, but optimized native code may not. That is
//! exactly C's contract for `restrict`, and it is why the attribute is opt-in.
//!
//! The attribute is legal only on a parameter (`crate::attrs`), takes no arguments, and only on a
//! parameter that is passed by pointer — an array, slice, tensor, pointer, reference, tuple or
//! struct. On a scalar it would promise nothing, so it is `E0306` there.

use wukong_ast::{
    BinOp as AstBinOp, Block, Expr, ExprKind, FnDecl, ForIter, ItemKind, Module, NodeId, Param,
    PatKind, Pattern, Stmt, StmtKind, TypeExpr, TypeKind, UnOp, VariantPat,
};
use wukong_diag::Diagnostic;
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Interner, Symbol};
use wukong_types::Ty;

use crate::{DefKind, DefMap};

/// The attribute's name, without the `@`.
pub const NOALIAS_ATTR: &str = "noalias";

/// Does this parameter carry `@noalias`? The one spelling test every consumer shares (sema here,
/// `wukong_mir_build` when it records the promise on the MIR parameter).
pub fn is_noalias_param(p: &Param, interner: &Interner) -> bool {
    p.attrs
        .iter()
        .any(|a| interner.resolve(a.name.sym) == NOALIAS_ATTR)
}

/// Can a parameter of this type carry the promise? Only one the ABI passes by pointer: a scalar,
/// vector or function value arrives as a copy and has no storage to promise anything about. A
/// `Ty::Named` is a struct or data-carrying enum (by pointer) unless it is a C-style enum (an
/// integer); a name `DefMap` does not know is a generic type parameter, which may be instantiated
/// at a buffer, and is accepted — `wukong_mir_build` records the promise only on a parameter whose
/// lowered ABI type really is a pointer.
fn can_carry_promise(t: &Ty, defs: &DefMap) -> bool {
    match t {
        Ty::Array { .. }
        | Ty::Tensor { .. }
        | Ty::Slice(_)
        | Ty::Ptr { .. }
        | Ty::Ref { .. }
        | Ty::Tuple(_) => true,
        Ty::Named(n) => !matches!(
            defs.lookup(*n).map(|d| &d.kind),
            Some(DefKind::Enum(vs)) if vs.iter().all(|v| v.payload.is_unit())
        ),
        // Already reported (or deliberately lenient) elsewhere; do not pile on.
        Ty::Unknown | Ty::Error => true,
        Ty::Scalar(_) | Ty::Unit | Ty::Vector { .. } | Ty::Fn { .. } => false,
    }
}

/// Is an argument (or parameter) of this type passed by pointer, so that it can share storage with
/// another argument? `Unknown`/`Error` make no claim.
fn is_buffer(t: &Ty, defs: &DefMap) -> bool {
    !matches!(t, Ty::Unknown | Ty::Error) && can_carry_promise(t, defs)
}

/// Does assigning a whole value of this type re-point the name rather than copy into its storage?
/// A slice, pointer or reference is a view, and a tensor value is handled as one; an array, tuple
/// or struct parameter is assigned element by element into the caller's storage.
fn assignment_rebinds(t: &Ty) -> bool {
    matches!(
        t,
        Ty::Slice(_) | Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Tensor { .. }
    )
}

/// Is a value of this type a view — a pointer to storage it does not own?
fn is_view_ty(t: &Ty) -> bool {
    matches!(t, Ty::Slice(_) | Ty::Ptr { .. } | Ty::Ref { .. })
}

/// One promised parameter, as a call site needs to see it.
#[derive(Clone, Copy)]
struct Promise {
    noalias: bool,
    name: Symbol,
}

/// Check every `@noalias` parameter's type, every call to a function with one, and every
/// reassignment of one. Runs after body checking, so every expression's type is in `types`.
pub(crate) fn check(
    module: &Module,
    interner: &Interner,
    defs: &DefMap,
    types: &HashMap<NodeId, Ty>,
    diags: &mut Vec<Diagnostic>,
) {
    let mut promises: HashMap<Symbol, Vec<Promise>> = HashMap::default();
    for item in &module.items {
        let decls: Vec<&FnDecl> = match &item.kind {
            ItemKind::Fn(f) => vec![f],
            ItemKind::Extern(block) => block.items.iter().collect(),
            _ => continue,
        };
        for f in decls {
            if let Some(ps) = collect_promises(f, interner, defs, diags) {
                promises.insert(f.name.sym, ps);
            }
        }
    }
    // Both rules are about a promise; a program that makes none (the common case) is not walked.
    if promises.is_empty() {
        return;
    }
    for item in &module.items {
        if let ItemKind::Fn(f) = &item.kind {
            if let Some(body) = &f.body {
                let mut w = Walker {
                    interner,
                    defs,
                    types,
                    promises: &promises,
                    diags: &mut *diags,
                    scopes: Vec::new(),
                    bindings: Vec::new(),
                    fixed: HashSet::default(),
                };
                w.function(f, body);
            }
        }
    }
}

/// The promise vector for `f` — `None` when no parameter carries one. A promise on a parameter
/// that cannot carry it is reported (`E0306`) and dropped, so no call site is checked against it.
fn collect_promises(
    f: &FnDecl,
    interner: &Interner,
    defs: &DefMap,
    diags: &mut Vec<Diagnostic>,
) -> Option<Vec<Promise>> {
    if !f.params.iter().any(|p| is_noalias_param(p, interner)) {
        return None;
    }
    let sig = match defs.lookup(f.name.sym).map(|d| &d.kind) {
        Some(DefKind::Fn(sig)) => Some(sig),
        _ => None,
    };
    let mut out = Vec::with_capacity(f.params.len());
    for (i, p) in f.params.iter().enumerate() {
        let mut noalias = is_noalias_param(p, interner);
        if noalias {
            if let Some(ty) = sig.and_then(|s| s.params.get(i)) {
                if !can_carry_promise(ty, defs) {
                    let span = p
                        .attrs
                        .iter()
                        .find(|a| interner.resolve(a.name.sym) == NOALIAS_ATTR)
                        .map_or(p.span, |a| a.span);
                    diags.push(
                        Diagnostic::error(format!(
                            "attribute `@{NOALIAS_ATTR}` is not valid on a parameter of type `{}`",
                            ty.display(interner)
                        ))
                        .with_code("E0306")
                        .primary(span, "this parameter is passed by value")
                        .help(
                            "`@noalias` promises something about the storage a parameter points \
                             into; only an array, slice, tensor, pointer, reference, tuple or \
                             struct parameter has any",
                        ),
                    );
                    noalias = false;
                }
            }
        }
        out.push(Promise {
            noalias,
            name: p.name.sym,
        });
    }
    out.iter().any(|p| p.noalias).then_some(out)
}

/// One step of a place path.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Proj {
    Field(Symbol),
    Tuple(u32),
}

/// The storage an argument designates, as far as this pass can name it.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Root {
    /// The storage of binding `.0` itself, narrowed by the field projections `.1`. An element index
    /// is not recorded, so `m[0]` and `m[1]` are both `Storage(m, [])`.
    Storage(u32, Vec<Proj>),
    /// The storage pointed at by the view value held in binding `.0` at the exact field place `.1`.
    Target(u32, Vec<Proj>),
}

/// Can storage named `a` and storage named `b` overlap, provably? Two places in one binding overlap
/// when one path is a prefix of the other (`p` and `p.w`; `p.w` and `p.w`), but not two different
/// fields. Two view targets are the same storage only when they are the same view value.
fn overlaps(a: &Root, b: &Root) -> bool {
    match (a, b) {
        (Root::Storage(x, p), Root::Storage(y, q)) => {
            x == y && (p.starts_with(q) || q.starts_with(p))
        }
        (Root::Target(x, p), Root::Target(y, q)) => x == y && p == q,
        (Root::Storage(..), Root::Target(..)) | (Root::Target(..), Root::Storage(..)) => false,
    }
}

/// What this pass knows about one binding (a parameter, a `let`, a `for` or `match` binding).
#[derive(Clone, Default)]
struct Binding {
    /// An immutable view binding initialized from storage this pass can name: what it points at.
    /// Read when the binding is used as a view.
    view_of: Option<Root>,
    /// An owned binding that is really a place inside other storage — a `for` element of an
    /// aggregate type, which `wukong_mir_build` binds by pointer into the iterated buffer.
    storage_of: Option<Root>,
}

struct Walker<'a> {
    interner: &'a Interner,
    defs: &'a DefMap,
    types: &'a HashMap<NodeId, Ty>,
    promises: &'a HashMap<Symbol, Vec<Promise>>,
    diags: &'a mut Vec<Diagnostic>,
    /// Lexical scopes, innermost last: name -> binding id (an index into `bindings`).
    scopes: Vec<HashMap<Symbol, u32>>,
    bindings: Vec<Binding>,
    /// The current function's `@noalias` view parameters, which may not be reassigned.
    fixed: HashSet<u32>,
}

impl Walker<'_> {
    fn ty(&self, e: &Expr) -> Option<&Ty> {
        self.types.get(&e.id)
    }

    fn is_view(&self, e: &Expr) -> bool {
        self.ty(e).is_some_and(is_view_ty)
    }

    fn lookup(&self, s: Symbol) -> Option<u32> {
        self.scopes.iter().rev().find_map(|sc| sc.get(&s).copied())
    }

    fn bind(&mut self, s: Symbol, b: Binding) -> u32 {
        let id = self.bindings.len() as u32;
        self.bindings.push(b);
        if let Some(sc) = self.scopes.last_mut() {
            sc.insert(s, id);
        }
        id
    }

    /// Bind every name `p` introduces. Only a plain identifier pattern can carry what its
    /// initializer is known to be; every name a destructuring pattern binds starts fresh.
    fn bind_pattern(&mut self, p: &Pattern, b: Binding) {
        match &p.kind {
            PatKind::Ident(s) => {
                self.bind(*s, b);
            }
            PatKind::Tuple(ps) | PatKind::Or(ps) => {
                for q in ps {
                    self.bind_pattern(q, Binding::default());
                }
            }
            PatKind::Variant { fields, .. } => match fields {
                VariantPat::Tuple(ps) => {
                    for q in ps {
                        self.bind_pattern(q, Binding::default());
                    }
                }
                VariantPat::Struct(fs) => {
                    for fp in fs {
                        self.bind_pattern(&fp.pat, Binding::default());
                    }
                }
            },
            PatKind::Struct { fields, .. } => {
                for fp in fields {
                    self.bind_pattern(&fp.pat, Binding::default());
                }
            }
            PatKind::Binding { name, sub } => {
                self.bind(*name, Binding::default());
                self.bind_pattern(sub, Binding::default());
            }
            PatKind::Slice { prefix, suffix, .. } => {
                for q in prefix.iter().chain(suffix) {
                    self.bind_pattern(q, Binding::default());
                }
            }
            PatKind::Wildcard
            | PatKind::Unit
            | PatKind::Int { .. }
            | PatKind::Char(_)
            | PatKind::Bool(_)
            | PatKind::Path(_)
            | PatKind::Range { .. } => {}
        }
    }

    fn function(&mut self, f: &FnDecl, body: &Block) {
        // Copied out of `self` so the signature borrow does not hold `self` across `bind`.
        let defs = self.defs;
        let sig_params: &[Ty] = match defs.lookup(f.name.sym).map(|d| &d.kind) {
            Some(DefKind::Fn(sig)) => &sig.params,
            _ => &[],
        };
        self.scopes.push(HashMap::default());
        for (i, p) in f.params.iter().enumerate() {
            let id = self.bind(p.name.sym, Binding::default());
            if is_noalias_param(p, self.interner)
                && sig_params.get(i).is_some_and(assignment_rebinds)
            {
                self.fixed.insert(id);
            }
        }
        self.block(body);
        self.scopes.pop();
    }

    fn block(&mut self, b: &Block) {
        self.scopes.push(HashMap::default());
        for s in &b.stmts {
            self.stmt(s);
        }
        if let Some(t) = &b.tail {
            self.expr(t);
        }
        self.scopes.pop();
    }

    fn stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Let {
                pat,
                mutable,
                ty,
                init,
            } => {
                // Classified BEFORE the binding exists: `let a = a;` reads the enclosing `a`.
                let view_of = match (init, &pat.kind) {
                    (Some(e), PatKind::Ident(_)) if !*mutable => self.view_init(ty.as_ref(), e),
                    _ => None,
                };
                if let Some(e) = init {
                    self.expr(e);
                }
                self.bind_pattern(
                    pat,
                    Binding {
                        view_of,
                        storage_of: None,
                    },
                );
            }
            StmtKind::Assign { target, value, .. } => {
                self.expr(target);
                self.expr(value);
                self.check_rebind(target);
            }
            StmtKind::Expr(e) | StmtKind::Defer(e) => self.expr(e),
            StmtKind::Return(e) | StmtKind::Break(_, e) => {
                if let Some(e) = e {
                    self.expr(e);
                }
            }
            StmtKind::Continue(_) => {}
            StmtKind::While { cond, body, .. } => {
                self.expr(cond);
                self.block(body);
            }
            StmtKind::For {
                pat, iter, body, ..
            } => {
                let element = match iter {
                    ForIter::Range {
                        start, end, step, ..
                    } => {
                        self.expr(start);
                        for e in end.iter().chain(step.iter()) {
                            self.expr(e);
                        }
                        Binding::default()
                    }
                    ForIter::Expr(e) => {
                        self.expr(e);
                        self.for_element(e)
                    }
                };
                self.scopes.push(HashMap::default());
                self.bind_pattern(pat, element);
                self.block(body);
                self.scopes.pop();
            }
        }
    }

    /// What a `for x in e` binding is. An element of aggregate type is bound by pointer into the
    /// iterated buffer, so it is a place inside that storage; a scalar (or view) element is a copy.
    fn for_element(&self, e: &Expr) -> Binding {
        let elem = match self.ty(e) {
            Some(Ty::Array { elem, .. }) | Some(Ty::Slice(elem)) => elem,
            _ => return Binding::default(),
        };
        if !matches!(**elem, Ty::Array { .. } | Ty::Tuple(_) | Ty::Named(_)) {
            return Binding::default();
        }
        Binding {
            view_of: None,
            storage_of: self.root_of(e),
        }
    }

    /// For an immutable `let` binding of a view: what it points at, when the initializer names it.
    fn view_init(&self, ann: Option<&TypeExpr>, init: &Expr) -> Option<Root> {
        let bound_is_view = match ann {
            Some(t) => matches!(
                t.kind,
                TypeKind::Slice(_) | TypeKind::Pointer { .. } | TypeKind::Ref { .. }
            ),
            None => self.is_view(init),
        };
        if !bound_is_view {
            return None;
        }
        // `let s: []f32 = a;` over an owned array views `a` itself (the one implicit conversion);
        // `let t = s;` over a view copies the pointer and views what `s` views.
        self.root_of(init)
    }

    /// Rule 2: a promised view parameter may not be re-pointed.
    fn check_rebind(&mut self, target: &Expr) {
        let ExprKind::Path(p) = &target.kind else {
            return;
        };
        if !p.is_single() {
            return;
        }
        let name = p.first().sym;
        if !self.lookup(name).is_some_and(|b| self.fixed.contains(&b)) {
            return;
        }
        let shown = self.interner.resolve(name).to_string();
        self.diags.push(
            Diagnostic::error(format!("cannot reassign `@{NOALIAS_ATTR}` parameter `{shown}`"))
                .with_code("E0316")
                .primary(
                    target.span,
                    "this would point the promised name at storage the promise never covered",
                )
                .help(format!(
                    "a `@{NOALIAS_ATTR}` parameter names one buffer for the whole call; write through \
                     it (`{shown}[i] = …`), or bind a local view (`let mut v = {shown};`) and move that"
                )),
        );
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
            | ExprKind::AlignOf(_) => {}
            ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => self.expr(expr),
            ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => self.expr(base),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            ExprKind::Call { callee, args, .. } => {
                self.expr(callee);
                for a in args {
                    self.expr(a);
                }
                self.check_call(callee, args);
            }
            ExprKind::Index { base, indices } => {
                self.expr(base);
                for i in indices {
                    self.expr(i);
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
            ExprKind::ArrayLit(items) | ExprKind::TupleLit(items) => {
                for i in items {
                    self.expr(i);
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
                if let Some(e) = else_branch {
                    self.expr(e);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee);
                for arm in arms {
                    self.scopes.push(HashMap::default());
                    self.bind_pattern(&arm.pat, Binding::default());
                    if let Some(g) = &arm.guard {
                        self.expr(g);
                    }
                    self.expr(&arm.body);
                    self.scopes.pop();
                }
            }
        }
    }

    /// Rule 1: no argument of a promised parameter may provably share storage with another buffer
    /// argument of the same call. Resolution follows `type_call`: a single-segment path naming a
    /// function definition.
    fn check_call(&mut self, callee: &Expr, args: &[Expr]) {
        let ExprKind::Path(p) = &callee.kind else {
            return;
        };
        if !p.is_single() {
            return;
        }
        let fname = p.first().sym;
        // Both tables are copied out of `self`, so their borrows do not hold `self` while the
        // diagnostics below are pushed.
        let (all, defs) = (self.promises, self.defs);
        let Some(promises) = all.get(&fname) else {
            return;
        };
        let Some(DefKind::Fn(sig)) = defs.lookup(fname).map(|d| &d.kind) else {
            return;
        };
        if promises.len() != args.len() || sig.params.len() != args.len() {
            return; // an arity error, reported by type checking
        }
        let roots: Vec<Option<Root>> = args
            .iter()
            .zip(&sig.params)
            .map(|(a, pty)| {
                let buffer = is_buffer(pty, self.defs)
                    && self.ty(a).is_some_and(|t| is_buffer(t, self.defs));
                if buffer {
                    self.root_of(a)
                } else {
                    None
                }
            })
            .collect();
        for (i, pi) in promises.iter().enumerate() {
            if !pi.noalias {
                continue;
            }
            let Some(ri) = &roots[i] else {
                continue;
            };
            for (j, pj) in promises.iter().enumerate() {
                // Each unordered pair once: a pair of two promised parameters is reported from
                // the lower index.
                if j == i || (pj.noalias && j < i) {
                    continue;
                }
                if !roots[j].as_ref().is_some_and(|rj| overlaps(ri, rj)) {
                    continue;
                }
                let callee_name = self.interner.resolve(fname).to_string();
                let promised = self.interner.resolve(pi.name).to_string();
                let other = self.interner.resolve(pj.name).to_string();
                self.diags.push(
                    Diagnostic::error(format!(
                        "this call passes overlapping storage to `@{NOALIAS_ATTR}` parameter \
                         `{promised}` of `{callee_name}`"
                    ))
                    .with_code("E0316")
                    .primary(
                        args[j].span,
                        format!("this argument (`{other}`) shares storage with `{promised}`"),
                    )
                    .secondary(
                        args[i].span,
                        format!("passed to `{promised}`, which promises nothing else overlaps it"),
                    )
                    .help(
                        "pass storage that does not overlap (copy it into a fresh local first), or \
                         drop the attribute if the callee must accept aliased arguments",
                    ),
                );
            }
        }
    }

    /// The storage an argument designates when it is passed as a buffer: a view argument passes
    /// what it points at, an owned one passes its own place.
    fn root_of(&self, e: &Expr) -> Option<Root> {
        if self.is_view(e) {
            self.view_target(e)
        } else {
            self.place_storage(e)
        }
    }

    /// The storage the place `e` occupies.
    fn place_storage(&self, e: &Expr) -> Option<Root> {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() => {
                let b = self.lookup(p.first().sym)?;
                Some(match &self.bindings[b as usize].storage_of {
                    Some(r) => r.clone(),
                    None => Root::Storage(b, Vec::new()),
                })
            }
            ExprKind::Field { base, name } => self.project(base, Proj::Field(name.sym)),
            ExprKind::TupleField { base, index } => self.project(base, Proj::Tuple(*index)),
            // An element index is not compared: an element stands for the whole indexed object.
            ExprKind::Index { base, .. } => {
                if self.is_view(base) {
                    self.view_target(base)
                } else {
                    self.place_storage(base)
                }
            }
            ExprKind::Unary {
                op: UnOp::Deref,
                expr,
            } => self.view_target(expr),
            _ => None,
        }
    }

    /// The place `base.<proj>`. Through a view the projection lands inside the pointee, which is
    /// then named as a whole.
    fn project(&self, base: &Expr, proj: Proj) -> Option<Root> {
        if self.is_view(base) {
            return self.view_target(base);
        }
        match self.place_storage(base)? {
            Root::Storage(b, mut path) => {
                path.push(proj);
                Some(Root::Storage(b, path))
            }
            target @ Root::Target(..) => Some(target),
        }
    }

    /// The storage the view value `e` points at.
    fn view_target(&self, e: &Expr) -> Option<Root> {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() => {
                let b = self.lookup(p.first().sym)?;
                Some(match &self.bindings[b as usize].view_of {
                    Some(r) => r.clone(),
                    None => Root::Target(b, Vec::new()),
                })
            }
            // A view held in a field of an owned aggregate, named by that exact place.
            ExprKind::Field { .. } | ExprKind::TupleField { .. } => {
                let (b, path) = self.exact_place(e)?;
                Some(Root::Target(b, path))
            }
            ExprKind::Unary {
                op: UnOp::Ref | UnOp::RefMut,
                expr,
            } => self.place_storage(expr),
            ExprKind::Cast { expr, .. } if self.is_view(expr) => self.view_target(expr),
            ExprKind::Binary {
                op: AstBinOp::Add | AstBinOp::Sub,
                lhs,
                ..
            } if self.is_view(lhs) => self.view_target(lhs),
            _ => None,
        }
    }

    /// `e` as a binding plus a chain of field projections through owned aggregates only — no
    /// element index, no dereference — so two equal results are provably one place.
    fn exact_place(&self, e: &Expr) -> Option<(u32, Vec<Proj>)> {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() => {
                let b = self.lookup(p.first().sym)?;
                if self.bindings[b as usize].storage_of.is_some() {
                    return None;
                }
                Some((b, Vec::new()))
            }
            ExprKind::Field { base, name } if !self.is_view(base) => {
                let (b, mut path) = self.exact_place(base)?;
                path.push(Proj::Field(name.sym));
                Some((b, path))
            }
            ExprKind::TupleField { base, index } if !self.is_view(base) => {
                let (b, mut path) = self.exact_place(base)?;
                path.push(Proj::Tuple(*index));
                Some((b, path))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use wukong_span::{Interner, SourceId};

    /// Codes reported for `src` by the whole checker.
    fn codes(src: &str) -> Vec<&'static str> {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        let (_res, diags) = crate::check(&module, &interner);
        diags.into_iter().filter_map(|d| d.code).collect()
    }

    const KERNELS: &str = "
fn k(@noalias mut o: [f32; 4], x: [f32; 4]) { o[0] = x[0]; }
fn k2(o: [f32; 4], @noalias x: [f32; 4]) -> f32 { return o[0] + x[0]; }
fn ks(@noalias mut o: []f32, x: []f32) { o[0] = x[0]; }
fn kv(@noalias mut o: [f32; 4], s: f32) { o[0] = s; }
fn u(mut o: [f32; 4], x: [f32; 4]) { o[0] = x[0]; }
struct P { a: [f32; 4], b: [f32; 4] }
fn kp(@noalias mut o: [f32; 4], q: P) { o[0] = q.a[0]; }
";

    fn in_main(body: &str) -> Vec<&'static str> {
        codes(&format!(
            "{KERNELS}\nfn main() -> i32 {{\n\
             let mut a: [f32; 4] = [0.0; 4];\n\
             let mut b: [f32; 4] = [1.0; 4];\n\
             let mut p: P = P {{ a: [0.0; 4], b: [0.0; 4] }};\n\
             let mut m: [[f32; 4]; 2] = [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]];\n\
             {body}\n\
             return 0;\n}}"
        ))
    }

    #[test]
    fn distinct_buffers_satisfy_the_promise() {
        assert_eq!(in_main("k(a, b);"), Vec::<&str>::new());
        assert_eq!(in_main("ks(a, b);"), Vec::<&str>::new());
        // Two different fields of one struct are two buffers.
        assert_eq!(in_main("k(p.a, p.b);"), Vec::<&str>::new());
        // A scalar argument is a copy, whatever it was read from.
        assert_eq!(in_main("kv(a, a[0]);"), Vec::<&str>::new());
        // No promise, no rule: the ordinary aliasing call stays legal.
        assert_eq!(in_main("u(a, a);"), Vec::<&str>::new());
    }

    #[test]
    fn the_same_storage_twice_is_e0316() {
        assert_eq!(in_main("k(a, a);"), vec!["E0316"], "same array");
        assert_eq!(
            in_main("ks(a, a);"),
            vec!["E0316"],
            "same array as two slices"
        );
        assert_eq!(
            in_main("let r = k2(a, a);"),
            vec!["E0316"],
            "promise on the 2nd param"
        );
        assert_eq!(in_main("k(p.a, p.a);"), vec!["E0316"], "same field");
        assert_eq!(
            in_main("kp(p.a, p);"),
            vec!["E0316"],
            "a field and its struct"
        );
        assert_eq!(
            in_main("k(m[0], m[1]);"),
            vec!["E0316"],
            "two elements of one array"
        );
    }

    /// A view does not hide its storage: an immutable `let` view, a re-view of a view, and a `for`
    /// element of an aggregate all resolve back to the buffer they point into.
    #[test]
    fn a_view_of_the_same_storage_is_e0316() {
        assert_eq!(
            in_main("let s: []f32 = a; ks(s, a);"),
            vec!["E0316"],
            "slice view"
        );
        assert_eq!(
            in_main("let s: []f32 = a; let t = s; ks(t, a);"),
            vec!["E0316"],
            "view of a view"
        );
        assert_eq!(
            in_main("for row in m { k(row, m[0]); }"),
            vec!["E0316"],
            "for element"
        );
        // A view of a different buffer is a different buffer.
        assert_eq!(in_main("let s: []f32 = b; ks(s, a);"), Vec::<&str>::new());
    }

    #[test]
    fn reassigning_a_promised_view_parameter_is_e0316() {
        assert_eq!(
            codes("fn r(@noalias mut s: []f32, t: []f32) { s = t; }"),
            vec!["E0316"]
        );
        // A fixed array is assigned by copy: the name keeps its storage.
        assert_eq!(
            codes("fn r(@noalias mut a: [f32; 4], b: [f32; 4]) { a = b; }"),
            Vec::<&str>::new()
        );
        // Writing through the promised view is the point of the attribute.
        assert_eq!(
            codes("fn r(@noalias mut s: []f32, t: []f32) { s[0] = t[0]; }"),
            Vec::<&str>::new()
        );
        // A local that shadows the parameter is a different binding.
        assert_eq!(
            codes("fn r(@noalias s: []f32, t: []f32) { let mut s: []f32 = t; s = t; }"),
            Vec::<&str>::new()
        );
    }

    #[test]
    fn a_promise_on_a_by_value_parameter_is_e0306() {
        assert_eq!(
            codes("fn f(@noalias x: i32) -> i32 { return x; }"),
            vec!["E0306"]
        );
        assert_eq!(codes("fn f(@noalias x: f32x8) {}"), vec!["E0306"]);
    }
}
