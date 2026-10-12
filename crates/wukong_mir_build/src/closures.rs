//! **Closure lifting** — the first rewrite of [`crate::lower_program_with`], before anything else
//! looks at the tree.
//!
//! Sema checked every closure literal where it was written and published its parameter, return
//! and capture types (`wukong_sema::ClosureInfo`). This pass turns each literal into an ordinary
//! top-level function and the literal into the name of that function:
//!
//! ```text
//! fn main() { let a = 2.0; apply(|x| x * a, 3.0) }
//!   =>  fn main() { let a = 2.0; apply((main$closure$0, a), 3.0) }
//!       fn main$closure$0(a: f32, x: f32) -> f32 { x * a }
//! ```
//!
//! The lifted function's leading parameters are the captures, in sema's order (first use), then the
//! closure's own parameters. `main$closure$0` is a registered function (`DefMap::insert_synthetic`),
//! so everything downstream already handles it: the function-typed parameter machinery of
//! `methods.rs` specializes `apply` per bound function (`apply$main$closure$0`) and turns the call
//! through `f` into a direct call.
//!
//! **The closure value keeps its captured reads in the tree.** A literal with captures becomes the
//! tuple `(L, c0, c1, ..)` — the textbook closure conversion, code and environment — typed as the
//! closure's function type; one without captures becomes the bare name `L`. Every pass below this
//! one decides from what the tree mentions: `canon` inlines and deletes a `let` whose only reads are
//! index positions and sinks an accumulator seed past the statements that do not mention it, the
//! fused-window recognizers prove the scalars a kernel hides are not read afterwards
//! (`block_mentions`), and `defer` checks that nothing a pending call reads is rebound. A bare `L`
//! mentions none of the captures, so each of those passes would delete, move or hide a binding the
//! closure still reads, and the capture copy would resolve to an outer binding of the same name (a
//! wrong value on every backend and at every `-O` level) or to nothing. As tuple elements the
//! captures are ordinary reads, at the point where the copy happens, for every walker at once.
//! [`closure_code`] is the function such a value denotes.
//!
//! **A `let`-bound closure** (`let f = |x: f32| x * k;`) is lifted the same way, and its `let` is
//! replaced by one hidden local per captured value, copied there (`let $cap$L$0 = k;`). Sema
//! recorded which literal each use of `f` denotes (`SemaResult::closure_refs`), and each is
//! rewritten in place: a call `f(x)` becomes the direct call `L($cap$L$0, x)`, and an argument
//! `apply(f, ..)` the closure value `(L, $cap$L$0)` read from those locals. A closure that uses `f`
//! captures those hidden locals instead (`ClosureInfo::bound`). No use of `f` survives this pass.
//!
//! **What a capture carries.** One the closure only reads is a copy: a scalar by value, an
//! aggregate (an array, tuple, struct or data-carrying enum) as a fresh buffer filled when the
//! closure value is evaluated (`capture_value`; a `let`'s hidden local already is one). One the
//! closure writes (`ClosureInfo::by_ref`) is the element `&mut c`: the lifted function's parameter
//! is a `*mut T`, and `lower_fn` binds the name to the storage it points at (`bind_by_ref`), so the
//! body reads and writes the enclosing variable itself. A capture's type is the width the variable
//! is stored at, which for a `for` range counter is its wider bound's type rather than its own
//! (`lower_for`'s `range_counter_type`; sema records it), so a copy and a pointer both match the slot.
//!
//! What closures add to lowering is the **environment ABI**:
//!
//! * an instance whose function-typed parameter `f` is bound to a closure with captures declares
//!   one hidden MIR parameter per capture at `f`'s position (`lower_fn`), and binds them as
//!   `$env$f$0`, `$env$f$1`, … (spellings no identifier can take);
//! * a call passing a function-typed argument passes, at that position, the closure's capture
//!   values: the tuple's captures, read from the enclosing locals at the literal's own site, or
//!   forwarded from the caller's own `$env$g$j` when the argument is the caller's function-typed
//!   parameter `g` (`lower_call`);
//! * a call through `f` calls the lifted function with `$env$f$..` first (`lower_dispatched_call`).
//!
//! The environment is keyed by the PARAMETER, not by the closure: two parameters of one instance
//! can be bound to the same closure literal with different captured values (through recursion), and
//! each carries its own. There is no function pointer and no new MIR construct, so the interpreter,
//! both native backends and the GPU lowering see only direct calls of ordinary functions.
//!
//! The lifted function inherits the enclosing function's `@strict_fp`, `@generic` and `@unchecked`.
//! Its body is that function's code, written inside it, and each of the three is a promise about all
//! of that code: lowering keys them by item, so without the copy a closure body in a `@strict_fp`
//! function was contracted to an `Fma`, and one in a `@generic` function was recognized.
//!
//! A module with no closure is returned as `None` and cloned not at all, so closure-free programs
//! lower byte-identically.

use wukong_ast::{
    Attr, Block, Expr, ExprKind, FnDecl, ForIter, Ident, Item, ItemKind, Module, NodeId, Param,
    PatKind, Path, Pattern, Stmt, StmtKind, TypeExpr, TypeKind, UnOp,
};
use wukong_mir::{MirType, Op, ValueId};
use wukong_sema::{ClosureInfo, Def, DefKind, FnSig, SemaResult};
use wukong_span::{FxHashMap as HashMap, Interner, Span, Symbol};
use wukong_types::Ty;

use crate::{methods, FnLowerer};

/// The ids this pass mints come from a band of their own, below the `defer` band
/// (`(u32::MAX / 8, u32::MAX / 4)`, which sits below the method desugaring's and `tindex`'s) and
/// far above anything the parser allocates. Counted downward.
const LIFT_ID_TOP: u32 = u32::MAX / 8 - 1;
const LIFT_ID_FLOOR: u32 = u32::MAX / 16;

/// What lowering needs about one lifted closure, keyed in [`ClosureTable`] by its function's name.
pub(crate) struct ClosureLift {
    /// The capture types: the lifted function's leading parameter types, and the types of the
    /// hidden parameters an instance bound to it declares.
    pub(crate) env_tys: Vec<Ty>,
    /// Per leading parameter: it is a pointer to the enclosing variable the closure writes (its
    /// type is `*mut T`), and the lifted body names the variable itself — `lower_fn` binds the
    /// name to the pointed-to storage instead of to a copy.
    pub(crate) by_ref: Vec<bool>,
}

/// The attributes a lifted closure takes over from the function it is written in: the ones that
/// govern how that function's own code is lowered. `@parallel` and `@tile` shape the enclosing
/// function's loops and are not inherited.
const INHERITED_ATTRS: &[&str] = &["strict_fp", "generic", "unchecked"];

/// The function a value passed to a function-typed parameter names: the first element of a
/// closure value `(L, captures..)`, or the expression itself (a function's name, a forwarded
/// function-typed parameter, or a capture-free closure's `L`).
pub(crate) fn closure_code(a: &Expr) -> &Expr {
    match &a.kind {
        ExprKind::TupleLit(xs) => xs.first().unwrap_or(a),
        _ => a,
    }
}

/// Every lifted closure, by the name of its function (`main$closure$0`).
pub(crate) type ClosureTable = HashMap<Symbol, ClosureLift>;

/// The name an instance binds the `j`-th captured value of its function-typed parameter `param`
/// under. A `$` cannot appear in a source identifier, so it shadows nothing.
pub(crate) fn env_name(param: &str, j: usize) -> String {
    format!("$env${param}${j}")
}

/// Lift every closure literal of `module` (see the module docs). `None` when sema recorded no
/// closure: nothing is cloned. Otherwise the rewritten module, sema's result extended with the
/// lifted functions' signatures and the new nodes' types, and the lifted closures' table.
pub(crate) fn lift_module(
    module: &Module,
    sema: &SemaResult,
    interner: &mut Interner,
) -> Option<(Module, SemaResult, ClosureTable)> {
    if sema.closures.is_empty() {
        return None;
    }
    let mut out = module.clone();
    let mut l = Lifter {
        sema,
        interner,
        next_id: LIFT_ID_TOP,
        failed: false,
        prefix: String::new(),
        counter: 0,
        attrs: Vec::new(),
        types: HashMap::default(),
        defs: Vec::new(),
        items: Vec::new(),
        table: ClosureTable::default(),
        bound: HashMap::default(),
    };
    for item in &mut out.items {
        let Item { attrs, kind, .. } = item;
        if let ItemKind::Fn(f) = kind {
            if let Some(body) = &mut f.body {
                l.prefix = l.interner.resolve(f.name.sym).to_string();
                l.counter = 0;
                l.attrs = attrs
                    .iter()
                    .filter(|a| INHERITED_ATTRS.contains(&l.interner.resolve(a.name.sym)))
                    .cloned()
                    .collect();
                l.block(body);
            }
        }
    }
    if l.failed {
        // Unreachable for a module sema accepted (a recorded closure for every literal, an id band
        // of a billion nodes); lowering then reports the literal it cannot lower.
        return None;
    }
    let Lifter {
        types,
        defs,
        items,
        table,
        ..
    } = l;
    out.items.extend(items);

    // The environment names every instance may bind: one per function-typed parameter and capture
    // index. Interned here because the lowerer holds only a shared `&Interner`.
    let widest = table.values().map(|c| c.env_tys.len()).max().unwrap_or(0);
    if widest > 0 {
        let mut params: Vec<Symbol> = Vec::new();
        for item in &out.items {
            if let ItemKind::Fn(f) = &item.kind {
                if let Some(DefKind::Fn(sig)) = sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
                    for (p, t) in f.params.iter().zip(&sig.params) {
                        if matches!(t, Ty::Fn { .. }) && !params.contains(&p.name.sym) {
                            params.push(p.name.sym);
                        }
                    }
                }
            }
        }
        for p in params {
            let name = interner.resolve(p).to_string();
            for j in 0..widest {
                interner.intern(&env_name(&name, j));
            }
        }
    }

    let mut s = sema.clone();
    s.types.extend(types);
    for def in defs {
        s.defs.insert_synthetic(def);
    }
    Some((out, s, table))
}

struct Lifter<'a> {
    sema: &'a SemaResult,
    interner: &'a mut Interner,
    next_id: u32,
    /// The id band ran out, or a literal had no sema record.
    failed: bool,
    /// The enclosing top-level function's name, which each lifted function's name extends.
    prefix: String,
    /// Closures lifted from the current top-level function so far.
    counter: usize,
    /// The current top-level function's [`INHERITED_ATTRS`], which every closure in it carries.
    attrs: Vec<Attr>,
    /// Types of the nodes this pass creates.
    types: HashMap<NodeId, Ty>,
    defs: Vec<Def>,
    items: Vec<Item>,
    table: ClosureTable,
    /// Every closure bound by a `let`, by its literal's id (see [`Bound`]).
    bound: HashMap<NodeId, Bound>,
}

/// A closure bound by `let f = |..| ..;`: its lifted function, the function's type, and the hidden
/// locals (`$cap$<L>$<j>`) its `let` became — one per captured value, copied there — with their
/// types. A use of `f` reads those locals.
struct Bound {
    code: Symbol,
    code_ty: Ty,
    env: Vec<(Symbol, Ty)>,
}

/// One value a lifted closure receives ahead of its own parameters: the parameter's name and
/// type, whether it stands for the enclosing variable itself (a pointer to it, `by_ref`), and the
/// expression, in the enclosing scope, that produces it where the closure is created.
struct Cap {
    name: Symbol,
    ty: Ty,
    by_ref: bool,
    value: Expr,
}

impl Lifter<'_> {
    fn fresh(&mut self) -> NodeId {
        if self.next_id <= LIFT_ID_FLOOR {
            self.failed = true;
            return NodeId::DUMMY;
        }
        let id = NodeId(self.next_id);
        self.next_id -= 1;
        id
    }

    /// A single-name path expression with a fresh id typed `ty`.
    fn path(&mut self, sym: Symbol, span: Span, ty: Ty) -> Expr {
        let id = self.fresh();
        self.types.insert(id, ty);
        Expr {
            id,
            kind: ExprKind::Path(Path {
                segments: vec![Ident { sym, span }],
                span,
            }),
            span,
        }
    }

    fn block(&mut self, b: &mut Block) {
        let stmts = std::mem::take(&mut b.stmts);
        let mut out = Vec::with_capacity(stmts.len());
        for mut s in stmts {
            if let Some(lets) = self.closure_let(&mut s) {
                out.extend(lets);
                continue;
            }
            self.stmt(&mut s);
            out.push(s);
        }
        b.stmts = out;
        if let Some(t) = &mut b.tail {
            self.expr(t);
        }
    }

    /// `let f = |..| ..;`: lift the literal, and replace the statement by one hidden `let` per
    /// captured value (`let $cap$L$0 = k;`), where the closure is created. Every use of `f` reads
    /// them (see [`Self::expr`]). `None` for every other statement.
    fn closure_let(&mut self, s: &mut Stmt) -> Option<Vec<Stmt>> {
        let StmtKind::Let {
            pat,
            init: Some(init),
            ..
        } = &mut s.kind
        else {
            return None;
        };
        if !matches!(init.kind, ExprKind::Closure { .. }) || !matches!(pat.kind, PatKind::Ident(_))
        {
            return None;
        }
        let lit = init.id;
        let span = init.span;
        // Lifted like a literal anywhere else; its value `(L, captures..)` is taken apart here.
        self.expr(init);
        let (code, code_ty, values) = match &init.kind {
            ExprKind::TupleLit(xs) if !xs.is_empty() => {
                let code = closure_code(init);
                let ExprKind::Path(p) = &code.kind else {
                    self.failed = true;
                    return None;
                };
                (p.first().sym, self.ty_of(init.id), xs[1..].to_vec())
            }
            ExprKind::Path(p) => (p.first().sym, self.ty_of(init.id), Vec::new()),
            _ => {
                self.failed = true;
                return None;
            }
        };
        let base = self.interner.resolve(code).to_string();
        let mut lets = Vec::with_capacity(values.len());
        let mut env = Vec::with_capacity(values.len());
        for (j, value) in values.into_iter().enumerate() {
            let name = self.interner.intern(&format!("$cap${base}${j}"));
            let ty = self.ty_of(value.id);
            env.push((name, ty));
            let pid = self.fresh();
            lets.push(Stmt {
                id: self.fresh(),
                attrs: Vec::new(),
                kind: StmtKind::Let {
                    pat: Pattern {
                        id: pid,
                        kind: PatKind::Ident(name),
                        span,
                    },
                    mutable: false,
                    ty: None,
                    init: Some(value),
                },
                span: s.span,
            });
        }
        self.bound.insert(lit, Bound { code, code_ty, env });
        Some(lets)
    }

    /// The type this pass gave a node it created (or sema gave an original one).
    fn ty_of(&self, id: NodeId) -> Ty {
        self.types
            .get(&id)
            .or_else(|| self.sema.types.get(&id))
            .cloned()
            .unwrap_or(Ty::Unknown)
    }

    /// The closure value `(L, env..)` — or the bare `L` — of the `let`-bound closure `b`, read from
    /// its hidden locals: what a use of the bound name passes to a function-typed parameter.
    fn bound_value(&mut self, lit: NodeId, span: Span) -> Option<Expr> {
        let b = self.bound.get(&lit)?;
        let (code, code_ty, env) = (b.code, b.code_ty.clone(), b.env.clone());
        let code_path = self.path(code, span, code_ty.clone());
        if env.is_empty() {
            return Some(code_path);
        }
        let mut elems = Vec::with_capacity(env.len() + 1);
        elems.push(code_path);
        for (sym, ty) in env {
            elems.push(self.path(sym, span, ty));
        }
        let id = self.fresh();
        self.types.insert(id, code_ty);
        Some(Expr {
            id,
            kind: ExprKind::TupleLit(elems),
            span,
        })
    }

    fn stmt(&mut self, s: &mut Stmt) {
        match &mut s.kind {
            StmtKind::Let { init, .. } => {
                if let Some(e) = init {
                    self.expr(e);
                }
            }
            StmtKind::Assign { target, value, .. } => {
                self.expr(target);
                self.expr(value);
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
            StmtKind::For { iter, body, .. } => {
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
            ExprKind::Path(_) => {
                // A closure-bound local passed to a function-typed parameter: the closure's value.
                if let Some(&lit) = self.sema.closure_refs.get(&e.id) {
                    match self.bound_value(lit, e.span) {
                        Some(v) => *e = v,
                        None => self.failed = true,
                    }
                }
            }
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::SizeOf(_)
            | ExprKind::AlignOf(_) => {}
            ExprKind::Unary { expr, .. }
            | ExprKind::Field { base: expr, .. }
            | ExprKind::TupleField { base: expr, .. }
            | ExprKind::Cast { expr, .. } => self.expr(expr),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            ExprKind::Call { callee, args, .. } => {
                for a in args.iter_mut() {
                    self.expr(a);
                }
                // A call through a closure-bound local `f(x)` is the direct call `L(env.., x)`.
                match self.sema.closure_refs.get(&callee.id).copied() {
                    Some(lit) => match self.bound.get(&lit) {
                        Some(b) => {
                            let (code, code_ty, env) = (b.code, b.code_ty.clone(), b.env.clone());
                            let span = callee.span;
                            **callee = self.path(code, span, code_ty);
                            let mut full = Vec::with_capacity(env.len() + args.len());
                            for (sym, ty) in env {
                                full.push(self.path(sym, span, ty));
                            }
                            full.append(args);
                            *args = full;
                        }
                        None => self.failed = true,
                    },
                    None => self.expr(callee),
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
            ExprKind::Closure { body, .. } => {
                // Inner literals first: the lifted body holds only the names of its inner closures.
                self.expr(body);
                self.lift(e);
            }
            ExprKind::Let { scrutinee, .. } => self.expr(scrutinee),
        }
    }

    /// The values the closure `info` receives ahead of its own parameters (see [`Cap`]): each
    /// capture by value, a written capture as a pointer to the variable (`&mut c`), and a captured
    /// closure-bound local as that closure's own hidden values.
    fn caps(&mut self, info: &ClosureInfo, span: Span) -> Vec<Cap> {
        let mut caps = Vec::with_capacity(info.captures.len());
        for (sym, ty) in &info.captures {
            if let Some(&(_, lit)) = info.bound.iter().find(|(n, _)| n == sym) {
                let Some(b) = self.bound.get(&lit) else {
                    self.failed = true;
                    continue;
                };
                for (h, hty) in b.env.clone() {
                    let value = self.path(h, span, hty.clone());
                    caps.push(Cap {
                        name: h,
                        ty: hty,
                        by_ref: false,
                        value,
                    });
                }
            } else if info.by_ref.contains(sym) {
                let pty = Ty::Ptr {
                    mutable: true,
                    pointee: Box::new(ty.clone()),
                };
                let inner = self.path(*sym, span, ty.clone());
                let id = self.fresh();
                self.types.insert(id, pty.clone());
                caps.push(Cap {
                    name: *sym,
                    ty: pty,
                    by_ref: true,
                    value: Expr {
                        id,
                        kind: ExprKind::Unary {
                            op: UnOp::RefMut,
                            expr: Box::new(inner),
                        },
                        span,
                    },
                });
            } else {
                let value = self.path(*sym, span, ty.clone());
                caps.push(Cap {
                    name: *sym,
                    ty: ty.clone(),
                    by_ref: false,
                    value,
                });
            }
        }
        caps
    }

    /// Replace the closure literal `e` (its body already lifted) by the value of a new function:
    /// `(L, captures..)`, or the bare `L` when it captures nothing (see the module docs).
    fn lift(&mut self, e: &mut Expr) {
        let Some(info) = self.sema.closures.get(&e.id) else {
            self.failed = true;
            return;
        };
        let info = info.clone();
        let span = e.span;
        let name = self
            .interner
            .intern(&format!("{}$closure${}", self.prefix, self.counter));
        self.counter += 1;
        let fn_ty = Ty::Fn {
            params: info.params.clone(),
            ret: Box::new(info.ret.clone()),
        };
        // Each capture is read where its copy is taken, in the enclosing scope, so every pass
        // that asks what this statement reads sees it.
        let caps = self.caps(&info, span);
        let code = self.path(name, span, fn_ty.clone());
        let value = if caps.is_empty() {
            code
        } else {
            let mut elems = Vec::with_capacity(caps.len() + 1);
            elems.push(code);
            elems.extend(caps.iter().map(|c| c.value.clone()));
            let id = self.fresh();
            self.types.insert(id, fn_ty);
            Expr {
                id,
                kind: ExprKind::TupleLit(elems),
                span,
            }
        };
        let ExprKind::Closure { params, ret, body } = std::mem::replace(e, value).kind else {
            unreachable!("lift is called on a closure literal");
        };

        // `Param::ty` is read only for an `extern` function (`externs.rs`); `lower_fn` takes every
        // parameter type from the registered signature, so an unannotated parameter's is a
        // placeholder.
        let placeholder = |span: Span| TypeExpr {
            id: NodeId::DUMMY,
            kind: TypeKind::Unit,
            span,
        };
        let mut fparams = Vec::with_capacity(caps.len() + params.len());
        for c in &caps {
            fparams.push(Param {
                id: self.fresh(),
                attrs: Vec::new(),
                mutable: false,
                name: Ident { sym: c.name, span },
                ty: placeholder(span),
                span,
            });
        }
        for p in params {
            fparams.push(Param {
                id: p.id,
                attrs: Vec::new(),
                mutable: false,
                name: p.name,
                ty: p.ty.unwrap_or_else(|| placeholder(p.span)),
                span: p.span,
            });
        }
        let body = match body.kind {
            ExprKind::Block(b) => b,
            _ => {
                let bspan = body.span;
                Block {
                    id: self.fresh(),
                    stmts: Vec::new(),
                    tail: Some(body),
                    span: bspan,
                }
            }
        };
        let mut sig_params: Vec<Ty> = caps.iter().map(|c| c.ty.clone()).collect();
        sig_params.extend(info.params.iter().cloned());
        self.defs.push(Def {
            name,
            kind: DefKind::Fn(FnSig {
                generics: Vec::new(),
                params: sig_params,
                ret: info.ret.clone(),
            }),
            span,
        });
        let item_id = self.fresh();
        self.items.push(Item {
            id: item_id,
            attrs: self.attrs.clone(),
            docs: Default::default(),
            kind: ItemKind::Fn(FnDecl {
                // The literal's span is the function's declaration span (source locations and
                // diagnostics point at the closure).
                name: Ident { sym: name, span },
                is_pub: false,
                generics: Vec::new(),
                params: fparams,
                ret: ret.map(|r| *r),
                where_preds: Vec::new(),
                body: Some(body),
            }),
            span,
        });
        self.table.insert(
            name,
            ClosureLift {
                by_ref: caps.iter().map(|c| c.by_ref).collect(),
                env_tys: caps.into_iter().map(|c| c.ty).collect(),
            },
        );
    }
}

impl<'a> FnLowerer<'a> {
    /// For the lifted closure `f` being lowered: which of its leading parameters are pointers to a
    /// variable it writes (captured by reference, [`ClosureLift::by_ref`]). Empty for any other
    /// function.
    pub(crate) fn by_ref_params(&self, f: Symbol) -> Vec<bool> {
        self.mono
            .and_then(|m| m.closures.get(&f))
            .map(|c| c.by_ref.clone())
            .unwrap_or_default()
    }

    /// Bind the by-reference capture `name`, whose parameter value `ptr` points at the enclosing
    /// function's variable of type `pointee`: the name denotes that storage itself, so reads and
    /// writes in the closure body go to the variable.
    pub(crate) fn bind_by_ref(&mut self, name: Symbol, ptr: ValueId, pointee: &Ty) {
        let mty = self.param_slot(pointee);
        let aggregate = matches!(mty, MirType::Array(..));
        self.bind_slice(name, ptr, mty, matches!(pointee, Ty::Slice(_)));
        if aggregate {
            self.bind_field_bases(name, ptr, pointee);
        }
    }

    /// The lifted closure `f` names, when it captures anything (a closure without captures needs no
    /// environment and is passed exactly like a named function).
    pub(crate) fn closure_env(&self, f: Symbol) -> Option<&'a ClosureLift> {
        let mono: &'a crate::Mono = self.mono?;
        mono.closures.get(&f).filter(|c| !c.env_tys.is_empty())
    }

    /// The hidden arguments a function-typed argument `a` passes: the captured values of the
    /// closure it denotes (empty for a named function or a capture-free closure). At the literal's
    /// own site — the closure value `(L, captures..)` — each capture is copied from the enclosing
    /// local now, so a later argument cannot change it; a forwarded function-typed parameter passes
    /// on what it received.
    pub(crate) fn closure_env_values(&mut self, a: &Expr) -> Vec<ValueId> {
        let Some(l) = methods::fn_value_of(a, &self.subst, self.sema, self.interner) else {
            return Vec::new();
        };
        let Some(lift) = self.closure_env(l) else {
            return Vec::new();
        };
        match &a.kind {
            ExprKind::TupleLit(xs) if xs.len() == lift.env_tys.len() + 1 => xs[1..]
                .iter()
                .zip(&lift.env_tys)
                .map(|(c, ty)| self.capture_value(c, ty))
                .collect(),
            ExprKind::Path(p) => {
                let name = p.first().sym;
                (0..lift.env_tys.len())
                    .map(|j| self.load_env(name, j, a.span))
                    .collect()
            }
            _ => {
                self.unsupported(a.span, "closure environment");
                Vec::new()
            }
        }
    }

    /// One captured value of a closure value `(L, captures..)`, copied now. An aggregate (an
    /// array, tuple, struct or data-carrying enum) is a base pointer, so its copy is a fresh buffer
    /// holding its contents: the closure must not see a later write to the original. A hidden
    /// `$cap$` local of a `let`-bound closure already is such a copy, made at the `let`, and nothing
    /// writes it. A scalar is passed as loaded: its capture type is the width it is stored at, which
    /// for a `for` range counter is the wider bound's type, not the counter's own (sema's
    /// `enter_range_counter`), so the value and the environment parameter agree.
    fn capture_value(&mut self, c: &Expr, ty: &Ty) -> ValueId {
        let v = self.lower_expr(c);
        let mty = self.mir_ty_of(ty);
        let hidden = match &c.kind {
            ExprKind::Path(p) => self.interner.resolve(p.first().sym).starts_with("$cap$"),
            _ => false,
        };
        if !matches!(mty, MirType::Array(..)) || hidden {
            return v;
        }
        let dst = self.builder.alloca(mty);
        self.emit_copy(dst, v, ty);
        dst
    }

    /// The `j`-th captured value the function-typed parameter `param` of this instance received.
    pub(crate) fn load_env(&mut self, param: Symbol, j: usize, span: Span) -> ValueId {
        let slot = self
            .interner
            .get(&env_name(self.interner.resolve(param), j))
            .and_then(|s| self.lookup(s));
        match slot {
            // An aggregate's binding is its buffer: its value is the base pointer.
            Some((slot, MirType::Array(..))) => slot,
            Some((slot, ty)) => self.builder.build(ty.clone(), Op::Load(slot, ty)),
            None => {
                self.unsupported(span, "closure environment");
                self.const_zero(MirType::I32)
            }
        }
    }

    /// Bind the hidden environment parameters `lower_fn` declared: each value is spilled to a slot
    /// named `$env$<param>$<j>` (like an ordinary scalar parameter; an aggregate is bound to the
    /// buffer it points at), where a call through the parameter and a forwarding call read it.
    pub(crate) fn bind_closure_env(&mut self, env: Vec<(Symbol, usize, ValueId, Ty)>) {
        for (param, j, val, ty) in env {
            let Some(sym) = self
                .interner
                .get(&env_name(self.interner.resolve(param), j))
            else {
                continue;
            };
            let mty = self.mir_ty_of(&ty);
            // An aggregate arrives as the base pointer of the caller's copy, which is its storage.
            if matches!(mty, MirType::Array(..)) {
                self.bind(sym, val, mty);
                continue;
            }
            let slot = self.builder.alloca(mty.clone());
            self.builder.build_void(Op::Store {
                ptr: slot,
                value: val,
            });
            self.bind(sym, slot, mty);
        }
    }

    /// The environment arguments of a call through the function-typed parameter `param`, bound to
    /// the lifted closure `lift`: one path per capture naming this instance's `$env$<param>$<j>`.
    pub(crate) fn env_arg_exprs(&self, param: Symbol, lift: &ClosureLift, span: Span) -> Vec<Expr> {
        (0..lift.env_tys.len())
            .filter_map(|j| {
                let sym = self
                    .interner
                    .get(&env_name(self.interner.resolve(param), j))?;
                Some(Expr {
                    // A synthesized node: its value is a local's load and needs no sema record.
                    id: NodeId::DUMMY,
                    kind: ExprKind::Path(Path {
                        segments: vec![Ident { sym, span }],
                        span,
                    }),
                    span,
                })
            })
            .collect()
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
    fn a_closure_is_lifted_and_its_callee_specialized_with_its_environment() {
        let funcs = lower(
            "fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
             fn main() -> i32 { let a = 2.0; return apply(|x| x * a, 3.0) as i32; }",
        );
        // The literal is a function of its capture, then its parameter.
        assert!(
            body(&funcs, "main$closure$0").starts_with("fn main$closure$0(v0 f32, v1 f32)"),
            "{funcs:?}"
        );
        // The callee is specialized per closure, with the capture as a hidden parameter at the
        // function-typed parameter's position, and calls the closure directly.
        let inst = body(&funcs, "apply$main$closure$0");
        assert!(
            inst.starts_with("fn apply$main$closure$0(v0 f32, v1 f32)"),
            "{inst}"
        );
        assert!(inst.contains("call main$closure$0("), "{inst}");
        assert!(body(&funcs, "main").contains("call apply$main$closure$0("));
        // The template is never lowered.
        assert!(!funcs.iter().any(|(n, _)| n == "apply"), "{funcs:?}");
    }

    #[test]
    fn a_closure_free_module_is_not_lifted() {
        let mut interner = Interner::new();
        let src = "fn square(x: f32) -> f32 { return x * x; } \
                   fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
                   fn main() -> i32 { return apply(square, 3.0) as i32; }";
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        assert!(super::lift_module(&module, &sema, &mut interner).is_none());
    }

    #[test]
    fn a_forwarded_closure_carries_its_environment() {
        let funcs = lower(
            "fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
             fn twice(g: fn(f32) -> f32, x: f32) -> f32 { return apply(g, apply(g, x)); } \
             fn main() -> i32 { let s = 1.5; return twice(|x| x + s, 1.0) as i32; }",
        );
        let twice = body(&funcs, "twice$main$closure$0");
        assert!(
            twice.starts_with("fn twice$main$closure$0(v0 f32, v1 f32)"),
            "{twice}"
        );
        assert_eq!(
            twice.matches("call apply$main$closure$0(").count(),
            2,
            "{twice}"
        );
        assert!(body(&funcs, "apply$main$closure$0").contains("call main$closure$0("));
    }

    /// A deferred call copies its closure's captures when it RUNS, so rebinding a captured name
    /// while the `defer` is pending would retarget it: `E0314`, as for any name it reads directly.
    #[test]
    fn a_pending_defer_reads_its_closure_captures() {
        let src = "fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
                   fn main() -> i32 { let c = 1.0; \
                   { defer print(apply(|x| x + c, 1.0)); let c = 5.0; print(c); } return 0; }";
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (_prog, diags) = lower_program(&module, &sema, &mut interner);
        let codes: Vec<_> = diags.iter().filter_map(|d| d.code).collect();
        assert_eq!(codes, vec!["E0314"], "{diags:?}");
    }

    /// The statements of the first nested block of `main`, after the lift and `canon` — the passes
    /// that run before any recognizer, in that order — as `let` / `expr` / `for` / `other`.
    fn inner_block_after_canon(src: &str) -> Vec<&'static str> {
        use wukong_ast::{ExprKind, ItemKind, StmtKind};
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (lifted, lsema, _) =
            super::lift_module(&module, &sema, &mut interner).expect("the module has a closure");
        let out = crate::canon::canonicalize_module(&lifted, &lsema).unwrap_or(lifted);
        let main = out
            .items
            .iter()
            .find_map(|it| match &it.kind {
                ItemKind::Fn(f) if interner.resolve(f.name.sym) == "main" => f.body.as_ref(),
                _ => None,
            })
            .expect("main");
        let inner = main
            .stmts
            .iter()
            .find_map(|s| match &s.kind {
                StmtKind::Expr(e) => match &e.kind {
                    ExprKind::Block(b) => Some(b),
                    _ => None,
                },
                _ => None,
            })
            .expect("a nested block");
        inner
            .stmts
            .iter()
            .map(|s| match &s.kind {
                StmtKind::Let { .. } => "let",
                StmtKind::Expr(_) => "expr",
                StmtKind::For { .. } => "for",
                _ => "other",
            })
            .collect()
    }

    /// `canon` inlines an integer `let` whose every read is an index, then deletes it. A closure
    /// that captures it reads it too, at its call: deleting the `let` left the capture to resolve
    /// to the OUTER `k` (`30 / 103 / 3` where `30 / 102 / 3` is right, on every backend).
    #[test]
    fn canon_keeps_a_let_a_closure_captures() {
        let src = |closure: &str| {
            format!(
                "fn applyi(f: fn(i64) -> i64, x: i64) -> i64 {{ return f(x); }} \
                 fn main() -> i32 {{ let a: [i64; 4] = [10, 20, 30, 40]; let j: i64 = 1; \
                 let k: i64 = 3; {{ let k = j + 1; print(a[k]); print(applyi({closure}, 100)); }} \
                 print(k); return 0; }}"
            )
        };
        assert_eq!(
            inner_block_after_canon(&src("|x| x + k")),
            vec!["let", "expr", "expr"]
        );
        // The control: a closure that does not read `k` leaves only the index read, which canon
        // rewrites — so the test above exercises the rewrite it guards against.
        assert_eq!(
            inner_block_after_canon(&src("|x| x + j")),
            vec!["expr", "expr"]
        );
    }

    /// `canon` sinks a literal accumulator seed to the first statement that mentions it. A closure
    /// reading the seed before its loop mentions it: sinking past the read made the capture see an
    /// outer `s` (`11` where `1` is right).
    #[test]
    fn canon_does_not_sink_a_seed_past_a_closure_that_reads_it() {
        let src = |closure: &str| {
            format!(
                "fn apply(f: fn(f32) -> f32, x: f32) -> f32 {{ return f(x); }} \
                 fn main() -> i32 {{ let xs: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; let s: f32 = 50.0; \
                 {{ let mut s = 0.0; print(apply({closure}, 1.0)); \
                 for i in 0..4 {{ s = s + xs[i]; }} print(s); }} print(s); return 0; }}"
            )
        };
        assert_eq!(
            inner_block_after_canon(&src("|x| x + s")),
            vec!["let", "expr", "for", "expr"]
        );
        // The control: with no read of `s` in the closure the seed does move down to its loop.
        assert_eq!(
            inner_block_after_canon(&src("|x| x + 1.0")),
            vec!["expr", "let", "for", "expr"]
        );
    }

    /// A fused-window recognizer replaces a running sum by `wukong_cumsum_f32` only when nothing
    /// after the window reads the accumulator the kernel hides. A closure capturing it reads it:
    /// the kernel fired anyway, the `let` was never lowered, and the capture read the outer `acc`
    /// (`100` where `10` is right).
    #[test]
    fn a_window_kernel_does_not_hide_a_scalar_a_closure_captures() {
        let src = |closure: &str| {
            format!(
                "fn apply(f: fn(f32) -> f32, x: f32) -> f32 {{ return f(x); }} \
                 fn main() -> i32 {{ let x: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; \
                 let mut out: [f32; 4] = [0.0; 4]; let acc: f32 = 100.0; \
                 {{ let mut acc: f32 = 0.0; for i in 0..4 {{ acc = acc + x[i]; out[i] = acc; }} \
                 print(apply({closure}, 0.0)); }} print(out[3]); print(acc); return 0; }}"
            )
        };
        let funcs = lower(&src("|v| v + acc"));
        assert!(
            !body(&funcs, "main").contains("wukong_cumsum_f32"),
            "{funcs:?}"
        );
        // The control: the same window followed by a closure that does not read `acc` dispatches.
        let funcs = lower(&src("|v| v + 1.0"));
        assert!(
            body(&funcs, "main").contains("wukong_cumsum_f32"),
            "{funcs:?}"
        );
    }

    /// The int8 GEMM + dequant fusion never writes the i32 accumulator `c`, so it drops `c` only
    /// when nothing after the window names `c`. A `let`-bound closure that writes `c` holds a
    /// pointer to it from its `let` on (`let $cap$L$0 = &mut c;`), and a later call names only that
    /// pointer: the fusion fired anyway, and every lane read back through the closure was 0. A raw
    /// `let pc = &mut c;` had the same hole.
    #[test]
    fn the_int8_dequant_fusion_keeps_an_accumulator_a_pointer_holds() {
        let src = |pre: &str, post: &str| {
            format!(
                "fn main() -> i32 {{ let mut a: [u8; 12] = [0 as u8; 12]; \
                 let mut b: [i8; 12] = [0 as i8; 12]; let mut c: [i32; 9] = [0; 9]; \
                 let mut out: [f32; 9] = [0.0; 9]; {pre} \
                 for i in 0..12 {{ a[i] = (i + 1) as u8; b[i] = ((i as i32) - 6) as i8; }} \
                 let scale_a: f32 = 0.5; let scale_b: [f32; 3] = [2.0, 1.0, 0.5]; \
                 for i in 0..3 {{ for j in 0..3 {{ let mut s: i32 = 0; \
                 for k in 0..4 {{ s = s + (a[i * 4 + k] as i32) * (b[j * 4 + k] as i32); }} \
                 c[i * 3 + j] = s; }} }} \
                 for i in 0..3 {{ for j in 0..3 {{ \
                 out[i * 3 + j] = (c[i * 3 + j] as f32) * scale_a * scale_b[j]; }} }} \
                 print(out[0]); {post} return 0; }}"
            )
        };
        let fused = |pre: &str, post: &str| {
            body(&lower(&src(pre, post)), "main").contains("wukong_i8gemm_nt_deq")
        };
        // The control: nothing holds `c`, so the window fuses.
        assert!(fused("", ""));
        assert!(!fused(
            "let peek = |k: i32| -> i32 { c[k] = c[k] + 0; c[k] };",
            "print(peek(0));"
        ));
        assert!(!fused("let pc = &mut c;", "print((*pc)[0]);"));
        // A closure that only reads `c` copied it at its `let`: nothing holds `c` itself.
        assert!(fused(
            "let first = |k: i32| -> i32 { c[k] };",
            "print(first(0));"
        ));
    }

    /// A closure body is code of the function it is written in, and lowering keys `@strict_fp`,
    /// `@generic` and `@unchecked` by item: the lifted item carries them, and nothing else.
    #[test]
    fn a_lifted_closure_inherits_its_functions_lowering_attributes() {
        use wukong_ast::ItemKind;
        let mut interner = Interner::new();
        let src = "fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
                   @strict_fp @generic @unchecked @parallel \
                   fn g(a: f32) -> f32 { return apply(|x| x * a, 1.0); } \
                   fn h(a: f32) -> f32 { return apply(|x| x + a, 1.0); } \
                   fn main() -> i32 { return (g(1.0) + h(1.0)) as i32; }";
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (out, _, _) = super::lift_module(&module, &sema, &mut interner).expect("closures");
        let attrs_of = |name: &str| -> Vec<String> {
            out.items
                .iter()
                .find(|it| {
                    matches!(&it.kind, ItemKind::Fn(f) if interner.resolve(f.name.sym) == name)
                })
                .unwrap_or_else(|| panic!("no item `{name}`"))
                .attrs
                .iter()
                .map(|a| interner.resolve(a.name.sym).to_string())
                .collect()
        };
        assert_eq!(
            attrs_of("g$closure$0"),
            vec!["strict_fp", "generic", "unchecked"]
        );
        assert!(attrs_of("h$closure$0").is_empty());
    }

    /// The `@strict_fp` promise for a closure body: `x * a + b` is a rounded multiply and a rounded
    /// add, never one `Fma`, exactly as it is in the function's own body.
    #[test]
    fn a_closure_in_a_strict_fp_function_is_not_contracted() {
        let src = |attr: &str| {
            format!(
                "fn apply(f: fn(f32) -> f32, x: f32) -> f32 {{ return f(x); }} \
                 {attr} fn g(a: f32, b: f32) -> f32 {{ return apply(|x| x * a + b, 3.0); }} \
                 fn main() -> i32 {{ print(g(0.1, -0.3)); return 0; }}"
            )
        };
        let funcs = lower(&src("@strict_fp"));
        assert!(!body(&funcs, "g$closure$0").contains("fma"), "{funcs:?}");
        // The control: without the attribute the same body contracts.
        let funcs = lower(&src(""));
        assert!(body(&funcs, "g$closure$0").contains("fma"), "{funcs:?}");
    }

    #[test]
    fn a_capture_free_closure_needs_no_environment() {
        let funcs = lower(
            "fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
             fn main() -> i32 { return apply(|x| x * 2.0, 3.0) as i32; }",
        );
        assert!(body(&funcs, "main$closure$0").starts_with("fn main$closure$0(v0 f32)"));
        assert!(body(&funcs, "apply$main$closure$0").starts_with("fn apply$main$closure$0(v0 f32)"));
    }

    /// `let f = |..| ..;` is lifted like any literal; its `let` becomes one hidden local per
    /// captured value, copied there, and every use reads them: a call `f(x)` is the direct call
    /// `L(env.., x)`, and an argument `apply(f, ..)` passes the closure value `(L, env..)`.
    #[test]
    fn a_let_bound_closure_is_called_directly_with_its_hidden_captures() {
        let funcs = lower(
            "fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
             fn main() -> i32 { let k = 2.0; let f = |x: f32| x * k; \
             return (f(1.0) + apply(f, 3.0)) as i32; }",
        );
        assert!(
            body(&funcs, "main$closure$0").starts_with("fn main$closure$0(v0 f32, v1 f32)"),
            "{funcs:?}"
        );
        let main = body(&funcs, "main");
        assert!(main.contains("call main$closure$0("), "{main}");
        assert!(main.contains("call apply$main$closure$0("), "{main}");
        assert!(body(&funcs, "apply$main$closure$0").contains("call main$closure$0("));
    }

    /// The lift's view of a `let`-bound closure: the statement is replaced by its hidden locals
    /// (none for a closure that captures nothing), and no use of the name survives.
    #[test]
    fn a_closure_let_becomes_its_hidden_captures() {
        use wukong_ast::{ItemKind, PatKind, StmtKind};
        let src = "fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
                   fn main() -> i32 { let k = 2.0; let f = |x: f32| x * k; \
                   let g = |x: f32| x + 1.0; return (f(1.0) + apply(g, 3.0)) as i32; }";
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (out, _, table) =
            super::lift_module(&module, &sema, &mut interner).expect("the module has closures");
        let main = out
            .items
            .iter()
            .find_map(|it| match &it.kind {
                ItemKind::Fn(f) if interner.resolve(f.name.sym) == "main" => f.body.as_ref(),
                _ => None,
            })
            .expect("main");
        let lets: Vec<String> = main
            .stmts
            .iter()
            .filter_map(|s| match &s.kind {
                StmtKind::Let { pat, .. } => match &pat.kind {
                    PatKind::Ident(n) => Some(interner.resolve(*n).to_string()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(lets, vec!["k", "$cap$main$closure$0$0"]);
        assert_eq!(table.len(), 2);
    }

    /// A capture the closure writes is passed as a pointer to the enclosing variable, and the
    /// lifted body reads and writes the variable through it — no copy.
    #[test]
    fn a_written_capture_is_passed_by_reference() {
        let funcs = lower(
            "fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
             fn main() -> i32 { let mut c = 0.0; \
             let r = apply(|x| { c = c + x; x }, 1.0); return (r + c) as i32; }",
        );
        let lifted = body(&funcs, "main$closure$0");
        assert!(
            lifted.starts_with("fn main$closure$0(v0 ptr, v1 f32)"),
            "{lifted}"
        );
        // The write goes to the incoming pointer itself.
        assert!(lifted.contains(", v0\n"), "{lifted}");
        assert!(!lifted.contains("alloca ptr"), "{lifted}");
        // The instance forwards it, and `main` passes the variable's own slot.
        assert!(body(&funcs, "apply$main$closure$0").starts_with("fn apply$main$closure$0(v0 ptr"));
    }

    /// An aggregate capture is copied where the closure is created: the call passes a fresh
    /// buffer, so a later write to the original (here by the next argument) is not seen.
    #[test]
    fn an_aggregate_capture_is_copied_at_the_call() {
        let funcs = lower(
            "fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); } \
             fn zero(p: *mut [f32; 2]) -> f32 { (*p)[0] = 0.0; return 1.0; } \
             fn main() -> i32 { let mut w: [f32; 2] = [1.0, 2.0]; \
             return apply(|x| x * w[0], zero(&mut w)) as i32; }",
        );
        let main = body(&funcs, "main");
        assert_eq!(main.matches("alloca [2 x f32]").count(), 2, "{main}");
        assert!(
            body(&funcs, "main$closure$0").starts_with("fn main$closure$0(v0 ptr, v1 f32)"),
            "{funcs:?}"
        );
    }

    /// A closure passed to a method is lifted and the method specialized for it, exactly as for a
    /// free function.
    #[test]
    fn a_closure_argument_of_a_method_specializes_the_method() {
        let funcs = lower(
            "struct V { x: f32 } \
             impl V { fn map(self, f: fn(f32) -> f32) -> f32 { return f(self.x); } } \
             fn main() -> i32 { let k = 3.0; let v = V { x: 2.0 }; \
             return v.map(|x| x * k) as i32; }",
        );
        assert!(body(&funcs, "main").contains("call V.map$main$closure$0("));
        assert!(body(&funcs, "V.map$main$closure$0").contains("call main$closure$0("));
    }

    /// A dimension generic named by a function-typed parameter is a hidden dimension parameter
    /// of the instance, bound at the call from the function passed.
    #[test]
    fn a_dimension_of_a_function_parameter_is_passed_to_the_instance() {
        let funcs = lower(
            "fn gen<N>(_f: fn(Tensor[f32, N]) -> f32) -> f32 { return N as f32; } \
             fn head(t: Tensor[f32, 4]) -> f32 { return t[0]; } \
             fn main() -> i32 { return gen(head) as i32; }",
        );
        assert!(
            body(&funcs, "gen$head").starts_with("fn gen$head(v0 i64)"),
            "{funcs:?}"
        );
        assert!(body(&funcs, "main").contains("const.i64 4"));
    }
}
