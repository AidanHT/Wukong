//! `defer` lowering: a post-sema rewrite of every function body that replaces each `defer <expr>;`
//! by copies of `<expr>` placed on every path that leaves its block.
//!
//! Semantics (the Zig/Swift model):
//!
//! - `defer e;` schedules `e` to run when control leaves the enclosing block, after everything the
//!   block does from that point on. Several defers run in reverse (LIFO) order, innermost block
//!   first.
//! - Every exit runs exactly the defers it leaves: falling off the end of the block (after the
//!   block's value has been computed), `return` (every pending defer of the function), and
//!   `break`/`continue` (the defers of the blocks inside the targeted loop, including the loop body
//!   itself on `continue`).
//! - A returned / broken / block-tail value is evaluated BEFORE the defers run, so `defer n += 1;
//!   return n;` returns the old `n`. (The value is held in a compiler temporary named `defer.N` —
//!   a spelling no source identifier can take.)
//! - `e` sees the variables in scope at the `defer` statement with their values at the time it
//!   RUNS. Rebinding one of the names `e` uses while the defer is pending would silently retarget
//!   it, so that is **E0314**, as is control flow leaving the deferred expression (`return`, or a
//!   `break`/`continue` aimed at a loop outside it).
//! - A failed `assert` (or any abort) leaves without running defers.
//!
//! Why here, as the first AST normalization of `lower_program_with` (before `tindex` and `canon`):
//! sema has already typed every deferred expression in the scope where it was written (so name
//! resolution is the source's), and the rewrite produces only ordinary statements, so every lowering
//! path after it — the plain lowerer, the `@parallel` vehicles, the kernel recognizers, autodiff and
//! all backends — sees plain control flow and needs no notion of `defer`. Like `tindex`, it clones
//! the module only when there is something to rewrite and returns the new nodes' types as an
//! overlay; each copy gets fresh `NodeId`s (see [`DEFER_ID_TOP`]) carrying the original's types.

use wukong_ast::{
    Block, Expr, ExprKind, ForIter, ItemKind, Module, NodeId, PatKind, Path, Pattern, Stmt,
    StmtKind, TypeExpr, TypeKind, UnOp, VariantPat,
};
use wukong_diag::Diagnostic;
use wukong_sema::{BuiltinCall, DimSnapshot, MethodCall, SemaResult};
use wukong_span::{FxHashMap, FxHashSet, Interner, Span, Symbol};
use wukong_types::{Dim, Ty};

/// Fresh `NodeId`s are minted DOWNWARD from here. The parser hands ids out upward from 0,
/// `tindex` mints downward from `u32::MAX - 1` but never at or below `u32::MAX / 2`, and the method
/// desugaring (`methods.rs`, which runs just before this pass) owns `(u32::MAX / 4, u32::MAX / 2)`,
/// so this band `(DEFER_ID_FLOOR, DEFER_ID_TOP]` collides with none of them for any program that
/// fits in memory.
const DEFER_ID_TOP: u32 = u32::MAX / 4 - 1;
const DEFER_ID_FLOOR: u32 = u32::MAX / 8;

/// What a rewrite adds to sema's `NodeId`-keyed tables for the nodes it created: every copy of a
/// deferred expression carries its original's entries under its fresh id.
pub(crate) struct DeferOverlay {
    /// Entries layered over `sema.types`.
    pub(crate) types: FxHashMap<NodeId, Ty>,
    /// Entries layered over `sema.call_dims` (keyed by a call's CALLEE id): a copied call to a
    /// shape-generic function keeps the dimension bindings sema solved for the original, which
    /// the hidden runtime-dim ABI reads.
    pub(crate) call_dims: FxHashMap<NodeId, Vec<(Symbol, Dim)>>,
    /// Entries layered over `sema.dim_snapshots` (keyed by a call's CALLEE id, like `call_dims`):
    /// a copied call whose turbofish names a local (`f::<n>(..)`) reads it into its dimension's
    /// slot itself. Without the record the copy wrote nothing there, while its `call_dims` and
    /// result type still named the dimension: reading the result's extent was `C0001`, or read 0
    /// when another read of the same binding had made the slot on a path that did not run.
    pub(crate) dim_snapshots: FxHashMap<NodeId, Vec<DimSnapshot>>,
    /// Entries layered over `sema.builtin_calls` (keyed by the CALL id): a copied named-kernel
    /// builtin call (`matmul(..)`) keeps the resolution its lowering reads.
    pub(crate) builtin_calls: FxHashMap<NodeId, BuiltinCall>,
    /// Entries layered over `sema.methods.calls` (keyed by the CALL id): a copied trait-bound call
    /// (`x.area()` with `x: T`) keeps the record its per-instance dispatch reads. Method calls with
    /// a known target were already rewritten into plain calls before this pass.
    pub(crate) method_calls: FxHashMap<NodeId, MethodCall>,
    /// Entries layered over `sema.str_print_args`: a copied `str` print argument (`defer
    /// println(s)`) still prints as text, not as the `[]u8` it is published as.
    pub(crate) str_print_args: FxHashSet<NodeId>,
}

impl DeferOverlay {
    /// Layer every entry over a clone of sema's result.
    pub(crate) fn apply(&self, s: &mut SemaResult) {
        s.types
            .extend(self.types.iter().map(|(id, t)| (*id, t.clone())));
        s.call_dims
            .extend(self.call_dims.iter().map(|(id, d)| (*id, d.clone())));
        s.dim_snapshots
            .extend(self.dim_snapshots.iter().map(|(id, d)| (*id, d.clone())));
        s.builtin_calls
            .extend(self.builtin_calls.iter().map(|(id, c)| (*id, c.clone())));
        s.methods
            .calls
            .extend(self.method_calls.iter().map(|(id, c)| (*id, c.clone())));
        s.str_print_args.extend(self.str_print_args.iter().copied());
    }
}

/// Rewrite every `defer` in `module` away. `Ok(None)`: the module has no `defer` and nothing was
/// cloned. `Ok(Some((module, overlay)))`: the rewritten module and the sema entries of its new
/// nodes (to be layered over `sema.types` / `sema.call_dims`). `Err`: the E0314 diagnostics — the
/// module must not be lowered.
pub(crate) fn lower_defers(
    module: &Module,
    sema: &SemaResult,
    interner: &mut Interner,
) -> Result<Option<(Module, DeferOverlay)>, Vec<Diagnostic>> {
    let has_defer = module.items.iter().any(|item| match &item.kind {
        ItemKind::Fn(f) => f.body.as_ref().is_some_and(block_has_defer),
        _ => false,
    });
    if !has_defer {
        return Ok(None);
    }
    let mut out = module.clone();
    let mut d = Desugar {
        base: &sema.types,
        extra: FxHashMap::default(),
        sema,
        extra_call_dims: FxHashMap::default(),
        extra_dim_snapshots: FxHashMap::default(),
        extra_builtin_calls: FxHashMap::default(),
        extra_method_calls: FxHashMap::default(),
        extra_str_print_args: FxHashSet::default(),
        interner,
        next_id: DEFER_ID_TOP,
        diags: Vec::new(),
        frames: Vec::new(),
        temps: 0,
    };
    for item in &mut out.items {
        if let ItemKind::Fn(f) = &mut item.kind {
            if let Some(body) = &mut f.body {
                if block_has_defer(body) {
                    d.frames.clear();
                    d.block(body);
                }
            }
        }
    }
    if d.diags.is_empty() {
        Ok(Some((
            out,
            DeferOverlay {
                types: d.extra,
                call_dims: d.extra_call_dims,
                dim_snapshots: d.extra_dim_snapshots,
                builtin_calls: d.extra_builtin_calls,
                method_calls: d.extra_method_calls,
                str_print_args: d.extra_str_print_args,
            },
        )))
    } else {
        Err(d.diags)
    }
}

/// A deferred expression waiting for its block to be left.
struct Pending {
    expr: Expr,
    /// The single-segment names `expr` mentions: none may be rebound while it is pending.
    mentions: Vec<Symbol>,
    /// The `defer` statement, for diagnostics.
    span: Span,
}

enum Frame {
    Block(Vec<Pending>),
    Loop(Option<Symbol>),
    /// Inside a deferred expression: control may not leave it.
    DeferBody,
}

struct Desugar<'a> {
    /// Sema's expression types for the original nodes.
    base: &'a FxHashMap<NodeId, Ty>,
    /// Types of the nodes this pass creates.
    extra: FxHashMap<NodeId, Ty>,
    /// Sema's id-keyed records for the original nodes (call dimensions, run-time turbofish reads,
    /// builtin calls, method calls), which every copy inherits under its fresh id.
    sema: &'a SemaResult,
    /// Call dimension bindings of the copies this pass creates.
    extra_call_dims: FxHashMap<NodeId, Vec<(Symbol, Dim)>>,
    /// Run-time turbofish reads of the copies this pass creates.
    extra_dim_snapshots: FxHashMap<NodeId, Vec<DimSnapshot>>,
    /// Builtin-call resolutions of the copies this pass creates.
    extra_builtin_calls: FxHashMap<NodeId, BuiltinCall>,
    /// Method-call records of the copies this pass creates.
    extra_method_calls: FxHashMap<NodeId, MethodCall>,
    /// The copies this pass creates of a `str` print argument.
    extra_str_print_args: FxHashSet<NodeId>,
    interner: &'a mut Interner,
    next_id: u32,
    diags: Vec<Diagnostic>,
    frames: Vec<Frame>,
    temps: u32,
}

/// What to do with a value (`return v`, `break v`, a block tail) that must be computed before the
/// defers run.
enum Hold {
    /// Unit-typed: evaluate it as a statement, then run the defers.
    Stmt,
    /// A constant: running the defers first cannot change it.
    AsIs,
    /// Bind it to a fresh temporary, run the defers, then use the temporary.
    Temp(Ty),
}

impl Desugar<'_> {
    fn fresh(&mut self) -> NodeId {
        assert!(
            self.next_id > DEFER_ID_FLOOR,
            "defer lowering exhausted its NodeId band (over a billion synthesized nodes)"
        );
        let id = NodeId(self.next_id);
        self.next_id -= 1;
        id
    }

    fn ty(&self, id: NodeId) -> Option<&Ty> {
        self.extra.get(&id).or_else(|| self.base.get(&id))
    }

    fn error(&mut self, span: Span, msg: String, label: &str, secondary: Option<(Span, &str)>) {
        let mut d = Diagnostic::error(msg)
            .with_code("E0314")
            .primary(span, label);
        if let Some((sp, l)) = secondary {
            d = d.secondary(sp, l);
        }
        self.diags.push(d);
    }

    // ---- blocks and exits -------------------------------------------------------------------

    fn block(&mut self, b: &mut Block) {
        self.frames.push(Frame::Block(Vec::new()));
        let mut out: Vec<Stmt> = Vec::with_capacity(b.stmts.len());
        for mut s in std::mem::take(&mut b.stmts) {
            let span = s.span;
            if let StmtKind::Defer(e) = &mut s.kind {
                self.frames.push(Frame::DeferBody);
                self.expr(e);
                self.frames.pop();
                let mut mentions = Vec::new();
                mentions_expr(e, &mut mentions);
                let expr = e.clone();
                if let Some(Frame::Block(pending)) = self.frames.last_mut() {
                    pending.push(Pending {
                        expr,
                        mentions,
                        span,
                    });
                }
            } else {
                self.stmt(s, &mut out);
            }
        }
        if let Some(tail) = &mut b.tail {
            self.expr(tail);
        }
        let Some(Frame::Block(pending)) = self.frames.pop() else {
            unreachable!("block frame pushed above");
        };
        if !pending.is_empty() && !ends_in_jump(&out, b.tail.is_some()) {
            let span = b.span;
            match b.tail.take() {
                None => self.emit_defers(&pending, &mut out),
                Some(tail) => {
                    let value = self.hold(*tail, &mut out, span);
                    self.emit_defers(&pending, &mut out);
                    b.tail = value.map(Box::new);
                }
            }
        }
        b.stmts = out;
    }

    /// Prepare `value` to survive the defers that follow: returns the expression that yields it
    /// afterwards (`None` for a unit value, which has been pushed as a statement).
    fn hold(&mut self, value: Expr, out: &mut Vec<Stmt>, span: Span) -> Option<Expr> {
        match self.hold_kind(&value) {
            Hold::Stmt => {
                out.push(self.stmt_node(StmtKind::Expr(value), span));
                None
            }
            Hold::AsIs => Some(value),
            Hold::Temp(ty) => {
                let name = self.interner.intern(&format!("defer.{}", self.temps));
                self.temps += 1;
                let pat_id = self.fresh();
                self.extra.insert(pat_id, ty.clone());
                let vspan = value.span;
                let pat = Pattern {
                    id: pat_id,
                    kind: PatKind::Ident(name),
                    span: vspan,
                };
                out.push(self.stmt_node(
                    StmtKind::Let {
                        pat,
                        mutable: false,
                        ty: None,
                        init: Some(value),
                    },
                    span,
                ));
                let id = self.fresh();
                self.extra.insert(id, ty);
                Some(Expr {
                    id,
                    kind: ExprKind::Path(Path {
                        segments: vec![wukong_ast::Ident {
                            sym: name,
                            span: vspan,
                        }],
                        span: vspan,
                    }),
                    span: vspan,
                })
            }
        }
    }

    fn hold_kind(&self, value: &Expr) -> Hold {
        if is_constant(value) {
            return Hold::AsIs;
        }
        match self.ty(value.id) {
            Some(Ty::Unit) => Hold::Stmt,
            Some(
                ty @ (Ty::Scalar(_)
                | Ty::Ptr { .. }
                | Ty::Ref { .. }
                | Ty::Vector { .. }
                | Ty::Named(_)
                | Ty::Tuple(_)
                | Ty::Array { .. }
                | Ty::Slice(_)),
            ) => Hold::Temp(ty.clone()),
            // A tensor view, a function value, or a value sema typed leniently (`Unknown`): there
            // is no temporary `mir_build` can hold it in, so the value is computed after the
            // defers — observable only if a defer writes what the value reads.
            _ => Hold::AsIs,
        }
    }

    fn stmt_node(&mut self, kind: StmtKind, span: Span) -> Stmt {
        Stmt {
            id: self.fresh(),
            attrs: Vec::new(),
            kind,
            span,
        }
    }

    /// Push fresh copies of `pending`, last-registered first, as statements.
    fn emit_defers(&mut self, pending: &[Pending], out: &mut Vec<Stmt>) {
        for p in pending.iter().rev() {
            let mut e = p.expr.clone();
            self.refresh_expr(&mut e);
            let span = p.span;
            out.push(self.stmt_node(StmtKind::Expr(e), span));
        }
    }

    /// The deferred expressions an exit through frames `from..` runs, innermost first, cloned out
    /// of the frame stack so they can be emitted while it is borrowed again.
    fn pending_from(&self, from: usize) -> Vec<Pending> {
        let mut all = Vec::new();
        for frame in self.frames[from..].iter().rev() {
            if let Frame::Block(pending) = frame {
                for p in pending.iter().rev() {
                    all.push(Pending {
                        expr: p.expr.clone(),
                        mentions: Vec::new(),
                        span: p.span,
                    });
                }
            }
        }
        // `all` is in run order; `emit_defers` runs a slice back to front (registration order).
        all.reverse();
        all
    }

    /// The frame index a `break`/`continue` with `label` leaves to (the loop frame itself), or
    /// `None` when it would leave a deferred expression (reported) or targets no loop (sema has
    /// already reported E0303).
    fn loop_target(&mut self, label: Option<Symbol>, span: Span) -> Option<usize> {
        let mut leaves_defer = false;
        let mut target = None;
        for (i, frame) in self.frames.iter().enumerate().rev() {
            match frame {
                Frame::Loop(l) if label.is_none() || *l == label => {
                    target = Some(i);
                    break;
                }
                Frame::DeferBody => {
                    leaves_defer = true;
                    break;
                }
                _ => {}
            }
        }
        if leaves_defer {
            self.error(
                span,
                "`break`/`continue` cannot leave a deferred expression".to_string(),
                "jumps out of the `defer`",
                None,
            );
        }
        target
    }

    fn in_defer_body(&self) -> bool {
        self.frames.iter().any(|f| matches!(f, Frame::DeferBody))
    }

    // ---- statements -------------------------------------------------------------------------

    fn stmt(&mut self, mut s: Stmt, out: &mut Vec<Stmt>) {
        let span = s.span;
        match &mut s.kind {
            StmtKind::Let { pat, init, .. } => {
                if let Some(e) = init {
                    self.expr(e);
                }
                self.check_rebind(pat);
            }
            StmtKind::Assign { target, value, .. } => {
                self.expr(target);
                self.expr(value);
            }
            StmtKind::Expr(e) => self.expr(e),
            StmtKind::Defer(_) => unreachable!("handled by `block`"),
            StmtKind::While {
                label, cond, body, ..
            } => {
                self.expr(cond);
                self.frames.push(Frame::Loop(label.map(|l| l.sym)));
                self.block(body);
                self.frames.pop();
            }
            StmtKind::For {
                label,
                pat,
                iter,
                body,
            } => {
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
                self.check_rebind(pat);
                self.frames.push(Frame::Loop(label.map(|l| l.sym)));
                self.block(body);
                self.frames.pop();
            }
            StmtKind::Return(value) => {
                if let Some(v) = value {
                    self.expr(v);
                }
                if self.in_defer_body() {
                    self.error(
                        span,
                        "`return` cannot leave a deferred expression".to_string(),
                        "returns out of the `defer`",
                        None,
                    );
                } else {
                    let pending = self.pending_from(0);
                    if !pending.is_empty() {
                        let held = value.take().and_then(|v| self.hold(v, out, span));
                        self.emit_defers(&pending, out);
                        *value = held;
                    }
                }
            }
            StmtKind::Break(label, value) => {
                if let Some(v) = value {
                    self.expr(v);
                }
                if let Some(target) = self.loop_target(label.map(|l| l.sym), span) {
                    let pending = self.pending_from(target + 1);
                    if !pending.is_empty() {
                        let held = value.take().and_then(|v| self.hold(v, out, span));
                        self.emit_defers(&pending, out);
                        *value = held;
                    }
                }
            }
            StmtKind::Continue(label) => {
                if let Some(target) = self.loop_target(label.map(|l| l.sym), span) {
                    let pending = self.pending_from(target + 1);
                    self.emit_defers(&pending, out);
                }
            }
        }
        out.push(s);
    }

    /// E0314 when `pat` binds a name some pending deferred expression uses.
    fn check_rebind(&mut self, pat: &Pattern) {
        let mut bound = Vec::new();
        pattern_bindings(pat, &mut bound);
        for name in bound {
            let hit = self.frames.iter().find_map(|f| match f {
                Frame::Block(pending) => pending
                    .iter()
                    .find(|p| p.mentions.contains(&name))
                    .map(|p| p.span),
                _ => None,
            });
            if let Some(defer_span) = hit {
                let nm = self.interner.resolve(name).to_string();
                self.error(
                    pat.span,
                    format!("`{nm}` cannot be rebound while a `defer` that uses it is pending"),
                    "rebinds a name the deferred expression reads",
                    Some((defer_span, "this `defer` uses the outer binding")),
                );
            }
        }
    }

    // ---- expressions: find nested blocks, loops and jumps -----------------------------------

    fn expr(&mut self, e: &mut Expr) {
        match &mut e.kind {
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::Path(_) => {}
            ExprKind::Unary { expr, .. } => self.expr(expr),
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
                for i in indices {
                    self.expr(i);
                }
            }
            ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => self.expr(base),
            ExprKind::Cast { expr, .. } => self.expr(expr),
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
            ExprKind::Block(b) => self.block(b),
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
                    self.check_rebind(&arm.pat);
                    if let Some(g) = &mut arm.guard {
                        self.expr(g);
                    }
                    self.expr(&mut arm.body);
                }
            }
            // An `if let` chain's `let` binds names like an arm's pattern does.
            ExprKind::Let { pat, scrutinee } => {
                self.expr(scrutinee);
                self.check_rebind(pat);
            }
            ExprKind::Loop { label, body } => {
                self.frames.push(Frame::Loop(label.map(|l| l.sym)));
                self.block(body);
                self.frames.pop();
            }
            // Types carry expressions only as constant array lengths: no blocks to rewrite.
            ExprKind::SizeOf(_) | ExprKind::AlignOf(_) | ExprKind::Closure { .. } => {}
        }
    }

    // ---- fresh copies -----------------------------------------------------------------------

    fn renumber(&mut self, id: &mut NodeId) {
        let new = self.fresh();
        if let Some(t) = self.ty(*id).cloned() {
            self.extra.insert(new, t);
        }
        let dims = self
            .extra_call_dims
            .get(id)
            .or_else(|| self.sema.call_dims.get(id))
            .cloned();
        if let Some(dims) = dims {
            self.extra_call_dims.insert(new, dims);
        }
        let snaps = self
            .extra_dim_snapshots
            .get(id)
            .or_else(|| self.sema.dim_snapshots.get(id))
            .cloned();
        if let Some(snaps) = snaps {
            self.extra_dim_snapshots.insert(new, snaps);
        }
        let builtin = self
            .extra_builtin_calls
            .get(id)
            .or_else(|| self.sema.builtin_calls.get(id))
            .cloned();
        if let Some(b) = builtin {
            self.extra_builtin_calls.insert(new, b);
        }
        let method = self
            .extra_method_calls
            .get(id)
            .or_else(|| self.sema.methods.calls.get(id))
            .cloned();
        if let Some(m) = method {
            self.extra_method_calls.insert(new, m);
        }
        if self.extra_str_print_args.contains(id) || self.sema.str_print_args.contains(id) {
            self.extra_str_print_args.insert(new);
        }
        *id = new;
    }

    fn refresh_expr(&mut self, e: &mut Expr) {
        self.renumber(&mut e.id);
        match &mut e.kind {
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::Path(_) => {}
            ExprKind::Unary { expr, .. } => self.refresh_expr(expr),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.refresh_expr(lhs);
                self.refresh_expr(rhs);
            }
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                self.refresh_expr(callee);
                for t in generic_args {
                    self.refresh_ty(t);
                }
                for a in args {
                    self.refresh_expr(a);
                }
            }
            ExprKind::Index { base, indices } => {
                self.refresh_expr(base);
                for i in indices {
                    self.refresh_expr(i);
                }
            }
            ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => {
                self.refresh_expr(base)
            }
            ExprKind::Cast { expr, ty } => {
                self.refresh_expr(expr);
                self.refresh_ty(ty);
            }
            ExprKind::StructLit { fields, rest, .. } => {
                for f in fields {
                    self.refresh_expr(&mut f.value);
                }
                if let Some(r) = rest {
                    self.refresh_expr(r);
                }
            }
            ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
                for x in xs {
                    self.refresh_expr(x);
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.refresh_expr(value);
                self.refresh_expr(count);
            }
            ExprKind::Block(b) => self.refresh_block(b),
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.refresh_expr(cond);
                self.refresh_block(then_branch);
                if let Some(e) = else_branch {
                    self.refresh_expr(e);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.refresh_expr(scrutinee);
                for arm in arms {
                    self.refresh_pattern(&mut arm.pat);
                    if let Some(g) = &mut arm.guard {
                        self.refresh_expr(g);
                    }
                    self.refresh_expr(&mut arm.body);
                }
            }
            ExprKind::Loop { body, .. } => self.refresh_block(body),
            ExprKind::Let { pat, scrutinee } => {
                self.refresh_expr(scrutinee);
                self.refresh_pattern(pat);
            }
            ExprKind::SizeOf(t) | ExprKind::AlignOf(t) => self.refresh_ty(t),
            // Lifted away before this pass runs (`closures.rs`).
            ExprKind::Closure { .. } => {}
        }
    }

    fn refresh_block(&mut self, b: &mut Block) {
        self.renumber(&mut b.id);
        for s in &mut b.stmts {
            self.refresh_stmt(s);
        }
        if let Some(t) = &mut b.tail {
            self.refresh_expr(t);
        }
    }

    fn refresh_stmt(&mut self, s: &mut Stmt) {
        self.renumber(&mut s.id);
        match &mut s.kind {
            StmtKind::Let { pat, ty, init, .. } => {
                self.refresh_pattern(pat);
                if let Some(t) = ty {
                    self.refresh_ty(t);
                }
                if let Some(e) = init {
                    self.refresh_expr(e);
                }
            }
            StmtKind::Assign { target, value, .. } => {
                self.refresh_expr(target);
                self.refresh_expr(value);
            }
            StmtKind::Expr(e) | StmtKind::Defer(e) => self.refresh_expr(e),
            StmtKind::Return(e) | StmtKind::Break(_, e) => {
                if let Some(e) = e {
                    self.refresh_expr(e);
                }
            }
            StmtKind::Continue(_) => {}
            StmtKind::While { cond, body, .. } => {
                self.refresh_expr(cond);
                self.refresh_block(body);
            }
            StmtKind::For {
                pat, iter, body, ..
            } => {
                self.refresh_pattern(pat);
                match iter {
                    ForIter::Range {
                        start, end, step, ..
                    } => {
                        self.refresh_expr(start);
                        if let Some(e) = end {
                            self.refresh_expr(e);
                        }
                        if let Some(e) = step {
                            self.refresh_expr(e);
                        }
                    }
                    ForIter::Expr(e) => self.refresh_expr(e),
                }
                self.refresh_block(body);
            }
        }
    }

    fn refresh_pattern(&mut self, p: &mut Pattern) {
        self.renumber(&mut p.id);
        match &mut p.kind {
            PatKind::Tuple(subs) | PatKind::Or(subs) => {
                for sub in subs {
                    self.refresh_pattern(sub);
                }
            }
            PatKind::Variant { fields, .. } => match fields {
                VariantPat::Tuple(subs) => {
                    for sub in subs {
                        self.refresh_pattern(sub);
                    }
                }
                VariantPat::Struct(fps) => {
                    for fp in fps {
                        self.refresh_pattern(&mut fp.pat);
                    }
                }
            },
            PatKind::Range { lo, hi, .. } => {
                self.refresh_pattern(lo);
                self.refresh_pattern(hi);
            }
            PatKind::Struct { fields, .. } => {
                for fp in fields {
                    self.refresh_pattern(&mut fp.pat);
                }
            }
            PatKind::Binding { sub, .. } => self.refresh_pattern(sub),
            PatKind::Slice {
                prefix,
                rest_binding,
                suffix,
                ..
            } => {
                for sub in prefix
                    .iter_mut()
                    .chain(rest_binding.as_deref_mut())
                    .chain(suffix.iter_mut())
                {
                    self.refresh_pattern(sub);
                }
            }
            PatKind::Wildcard
            | PatKind::Ident(_)
            | PatKind::Unit
            | PatKind::Int { .. }
            | PatKind::Char(_)
            | PatKind::Bool(_)
            | PatKind::Path(_) => {}
        }
    }

    fn refresh_ty(&mut self, t: &mut TypeExpr) {
        self.renumber(&mut t.id);
        match &mut t.kind {
            TypeKind::Path(_) | TypeKind::Int(_) | TypeKind::Unit => {}
            TypeKind::Generic { args, .. } => {
                for a in args {
                    if let wukong_ast::TypeArg::Type(t) = a {
                        self.refresh_ty(t);
                    }
                }
            }
            TypeKind::Pointer { pointee, .. } | TypeKind::Ref { pointee, .. } => {
                self.refresh_ty(pointee)
            }
            TypeKind::Slice(elem) | TypeKind::Vector { elem, .. } => self.refresh_ty(elem),
            TypeKind::Array { elem, len, .. } => {
                self.refresh_ty(elem);
                self.refresh_expr(len);
            }
            TypeKind::Tuple(elems) => {
                for e in elems {
                    self.refresh_ty(e);
                }
            }
            TypeKind::Tensor { elem, .. } => self.refresh_ty(elem),
            TypeKind::Fn { params, ret } => {
                for p in params {
                    self.refresh_ty(p);
                }
                if let Some(r) = ret {
                    self.refresh_ty(r);
                }
            }
        }
    }
}

/// A value the defers cannot change: a literal, or a negated literal.
fn is_constant(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_) => true,
        ExprKind::Unary {
            op: UnOp::Neg,
            expr,
        } => is_constant(expr),
        _ => false,
    }
}

/// Whether the block's statements end in an unconditional jump (no tail), so a fallthrough copy of
/// the defers would be dead code.
fn ends_in_jump(stmts: &[Stmt], has_tail: bool) -> bool {
    !has_tail
        && stmts.last().is_some_and(|s| {
            matches!(
                s.kind,
                StmtKind::Return(_) | StmtKind::Break(..) | StmtKind::Continue(_)
            )
        })
}

/// The names a pattern binds.
fn pattern_bindings(p: &Pattern, out: &mut Vec<Symbol>) {
    match &p.kind {
        PatKind::Ident(s) => out.push(*s),
        PatKind::Tuple(subs) | PatKind::Or(subs) => {
            for sub in subs {
                pattern_bindings(sub, out);
            }
        }
        PatKind::Variant { fields, .. } => match fields {
            VariantPat::Tuple(subs) => {
                for sub in subs {
                    pattern_bindings(sub, out);
                }
            }
            VariantPat::Struct(fps) => {
                for fp in fps {
                    pattern_bindings(&fp.pat, out);
                }
            }
        },
        PatKind::Struct { fields, .. } => {
            for fp in fields {
                pattern_bindings(&fp.pat, out);
            }
        }
        PatKind::Binding { name, sub } => {
            out.push(*name);
            pattern_bindings(sub, out);
        }
        PatKind::Slice {
            prefix,
            rest_binding,
            suffix,
            ..
        } => {
            for sub in prefix.iter().chain(rest_binding.as_deref()).chain(suffix) {
                pattern_bindings(sub, out);
            }
        }
        PatKind::Range { .. }
        | PatKind::Wildcard
        | PatKind::Unit
        | PatKind::Int { .. }
        | PatKind::Char(_)
        | PatKind::Bool(_)
        | PatKind::Path(_) => {}
    }
}

/// Every single-segment name an expression mentions (a superset of the variables it reads).
fn mentions_expr(e: &Expr, out: &mut Vec<Symbol>) {
    match &e.kind {
        ExprKind::Path(p) => {
            if p.is_single() && !out.contains(&p.segments[0].sym) {
                out.push(p.segments[0].sym);
            }
        }
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_)
        | ExprKind::Closure { .. } => {}
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => mentions_expr(expr, out),
        ExprKind::Binary { lhs, rhs, .. } => {
            mentions_expr(lhs, out);
            mentions_expr(rhs, out);
        }
        ExprKind::Call {
            callee,
            generic_args,
            args,
        } => {
            mentions_expr(callee, out);
            // A turbofish dimension may name a local the call reads (`f::<n>(..)`), at the exit
            // where the copy runs, by name.
            for g in generic_args {
                if let TypeKind::Path(p) = &g.kind {
                    if p.is_single() && !out.contains(&p.segments[0].sym) {
                        out.push(p.segments[0].sym);
                    }
                }
            }
            for a in args {
                mentions_expr(a, out);
            }
        }
        ExprKind::Index { base, indices } => {
            mentions_expr(base, out);
            for i in indices {
                mentions_expr(i, out);
            }
        }
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => {
            mentions_expr(base, out)
        }
        ExprKind::StructLit { fields, rest, .. } => {
            for f in fields {
                mentions_expr(&f.value, out);
            }
            if let Some(r) = rest {
                mentions_expr(r, out);
            }
        }
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
            for x in xs {
                mentions_expr(x, out);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            mentions_expr(value, out);
            mentions_expr(count, out);
        }
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => mentions_block(b, out),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            mentions_expr(cond, out);
            mentions_block(then_branch, out);
            if let Some(e) = else_branch {
                mentions_expr(e, out);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            mentions_expr(scrutinee, out);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    mentions_expr(g, out);
                }
                mentions_expr(&arm.body, out);
            }
        }
        ExprKind::Let { scrutinee, .. } => mentions_expr(scrutinee, out),
    }
}

fn mentions_block(b: &Block, out: &mut Vec<Symbol>) {
    for s in &b.stmts {
        match &s.kind {
            StmtKind::Let { init, .. } => {
                if let Some(e) = init {
                    mentions_expr(e, out);
                }
            }
            StmtKind::Assign { target, value, .. } => {
                mentions_expr(target, out);
                mentions_expr(value, out);
            }
            StmtKind::Expr(e) | StmtKind::Defer(e) => mentions_expr(e, out),
            StmtKind::Return(e) | StmtKind::Break(_, e) => {
                if let Some(e) = e {
                    mentions_expr(e, out);
                }
            }
            StmtKind::Continue(_) => {}
            StmtKind::While { cond, body, .. } => {
                mentions_expr(cond, out);
                mentions_block(body, out);
            }
            StmtKind::For { iter, body, .. } => {
                match iter {
                    ForIter::Range {
                        start, end, step, ..
                    } => {
                        mentions_expr(start, out);
                        if let Some(e) = end {
                            mentions_expr(e, out);
                        }
                        if let Some(e) = step {
                            mentions_expr(e, out);
                        }
                    }
                    ForIter::Expr(e) => mentions_expr(e, out),
                }
                mentions_block(body, out);
            }
        }
    }
    if let Some(t) = &b.tail {
        mentions_expr(t, out);
    }
}

fn block_has_defer(b: &Block) -> bool {
    b.stmts.iter().any(stmt_has_defer) || b.tail.as_deref().is_some_and(expr_has_defer)
}

fn stmt_has_defer(s: &Stmt) -> bool {
    match &s.kind {
        StmtKind::Defer(_) => true,
        StmtKind::Let { init, .. } => init.as_ref().is_some_and(expr_has_defer),
        StmtKind::Assign { target, value, .. } => expr_has_defer(target) || expr_has_defer(value),
        StmtKind::Expr(e) => expr_has_defer(e),
        StmtKind::Return(e) | StmtKind::Break(_, e) => e.as_ref().is_some_and(expr_has_defer),
        StmtKind::Continue(_) => false,
        StmtKind::While { cond, body, .. } => expr_has_defer(cond) || block_has_defer(body),
        StmtKind::For { iter, body, .. } => {
            let iter_has = match iter {
                ForIter::Range {
                    start, end, step, ..
                } => {
                    expr_has_defer(start)
                        || end.as_ref().is_some_and(expr_has_defer)
                        || step.as_ref().is_some_and(expr_has_defer)
                }
                ForIter::Expr(e) => expr_has_defer(e),
            };
            iter_has || block_has_defer(body)
        }
    }
}

fn expr_has_defer(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => block_has_defer(b),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            expr_has_defer(cond)
                || block_has_defer(then_branch)
                || else_branch.as_deref().is_some_and(expr_has_defer)
        }
        ExprKind::Match { scrutinee, arms } => {
            expr_has_defer(scrutinee)
                || arms.iter().any(|a| {
                    a.guard.as_ref().is_some_and(expr_has_defer) || expr_has_defer(&a.body)
                })
        }
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => expr_has_defer(expr),
        ExprKind::Binary { lhs, rhs, .. } => expr_has_defer(lhs) || expr_has_defer(rhs),
        ExprKind::Call { callee, args, .. } => {
            expr_has_defer(callee) || args.iter().any(expr_has_defer)
        }
        ExprKind::Index { base, indices } => {
            expr_has_defer(base) || indices.iter().any(expr_has_defer)
        }
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => expr_has_defer(base),
        ExprKind::StructLit { fields, rest, .. } => {
            fields.iter().any(|f| expr_has_defer(&f.value))
                || rest.as_deref().is_some_and(expr_has_defer)
        }
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => xs.iter().any(expr_has_defer),
        ExprKind::ArrayRepeat { value, count } => expr_has_defer(value) || expr_has_defer(count),
        ExprKind::Let { scrutinee, .. } => expr_has_defer(scrutinee),
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::Path(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_)
        | ExprKind::Closure { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::SourceId;

    fn analyze(src: &str) -> (Module, SemaResult, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        (module, sema, interner)
    }

    /// The rewritten module as the AST printer renders it, trimmed to the lines that matter.
    fn rewritten_lines(src: &str, keep: impl Fn(&str) -> bool) -> Vec<String> {
        let (module, sema, mut interner) = analyze(src);
        let (out, _) = lower_defers(&module, &sema, &mut interner)
            .expect("no E0314")
            .expect("the module has a defer");
        wukong_ast::print::print_module(&out, &interner)
            .lines()
            .map(str::trim)
            .filter(|l| keep(l))
            .map(str::to_string)
            .collect()
    }

    fn e0314(src: &str) -> usize {
        let (module, sema, mut interner) = analyze(src);
        match lower_defers(&module, &sema, &mut interner) {
            Ok(_) => 0,
            Err(diags) => {
                assert!(diags.iter().all(|d| d.code == Some("E0314")), "{diags:?}");
                diags.len()
            }
        }
    }

    #[test]
    fn a_module_without_defer_is_not_cloned() {
        let (module, sema, mut interner) = analyze("fn f() -> i32 { return 1; }");
        assert!(matches!(
            lower_defers(&module, &sema, &mut interner),
            Ok(None)
        ));
    }

    /// Each exit runs exactly the defers it leaves, innermost block first, LIFO within a block:
    /// the early `return 7` runs `g(2)` then `g(1)`; the final `return 8` runs only `g(1)`; the
    /// `if` block ends in a return, so it gets no dead fallthrough copy.
    #[test]
    fn defers_are_copied_onto_every_exit_in_lifo_order() {
        let lines = rewritten_lines(
            "fn g(x: i32) {}\n\
             fn f(c: bool) -> i32 { defer g(1); if c { defer g(2); return 7; } g(3); return 8; }",
            |l| l.starts_with("int ") || l == "return",
        );
        assert_eq!(
            lines,
            ["int 2", "int 1", "return", "int 7", "int 3", "int 1", "return", "int 8"],
        );
    }

    /// A returned value is computed BEFORE the defers run: `return n` is held in a temporary, so
    /// `defer n += 1` cannot change what is returned. A constant needs no temporary.
    #[test]
    fn a_returned_value_is_held_before_the_defers_run() {
        let lines = rewritten_lines(
            "fn f() -> i32 { let mut n = 5; defer n += 1; return n; }",
            |l| l.contains("defer.") || l == "return",
        );
        assert_eq!(lines, ["pat defer.0", "return", "path defer.0"]);
        let lines = rewritten_lines(
            "fn f() -> i32 { let mut n = 5; defer n += 1; return 3; }",
            |l| l.contains("defer."),
        );
        assert!(lines.is_empty(), "{lines:?}");
    }

    /// `continue` leaves the loop body (its defers run); `break` out of an inner block runs the
    /// body's and the inner block's; the loop's fallthrough runs the body's once per iteration.
    #[test]
    fn loop_exits_run_the_loop_body_defers() {
        let lines = rewritten_lines(
            "fn g(x: i32) {}\n\
             fn f() { for i in 0..4 { defer g(1); if i == 1 { continue; } \
             if i == 3 { defer g(2); break; } g(0); } }",
            |l| l.starts_with("int ") || l == "continue" || l == "break",
        );
        assert_eq!(
            lines,
            [
                "int 0", "int 4", // the range bounds
                "int 1", "int 1", "continue", // `i == 1`, then g(1); continue
                "int 3", "int 2", "int 1", "break", // `i == 3`, then g(2); g(1); break
                "int 0", "int 1", // g(0), then the fallthrough g(1)
            ],
        );
    }

    #[test]
    fn control_flow_out_of_a_defer_is_e0314() {
        assert_eq!(e0314("fn f() -> i32 { defer { return 1; }; return 0; }"), 1);
        assert_eq!(e0314("fn f() { for i in 0..3 { defer { break; }; } }"), 1);
        // A loop entirely inside the deferred expression may break out of itself.
        assert_eq!(
            e0314("fn g(x: i32) {} fn f() { defer { for i in 0..2 { g(i); break; } }; }"),
            0
        );
    }

    #[test]
    fn rebinding_a_name_a_pending_defer_uses_is_e0314() {
        assert_eq!(
            e0314("fn g(x: i32) {} fn f() { let x = 1; defer g(x); let x = 2; g(x); }"),
            1
        );
        assert_eq!(
            e0314("fn g(x: i32) {} fn f() { let x = 1; defer g(x); for x in 0..2 { g(x); } }"),
            1
        );
        // Rebinding after the defer has run (its block ended) is fine.
        assert_eq!(
            e0314("fn g(x: i32) {} fn f() { let x = 1; if true { defer g(x); } let x = 2; g(x); }"),
            0
        );
        // A turbofish reads the local it names when the defer runs (`f::<n>(..)`): the copy at
        // the exit would read the new `n` where sema typed the outer one.
        assert_eq!(
            e0314(&format!(
                "{ROW}fn f(buf: [i32; 6]) {{ let n = 3; defer print(row::<2, n>(buf, 1)[0]); \
                 let n = 2; print(n); }}"
            )),
            1
        );
    }

    const ROW: &str = "fn row<M, N>(a: Tensor[i32, M, N], i: i64) -> Tensor[i32, N] { \
                       return a[i]; }\n";

    /// Every copy of a deferred call whose turbofish reads a local carries the record of that read
    /// under its fresh id, so it writes the dimension's slot itself. Without it a copy read the
    /// slot unwritten: `C0001` when nothing else made the slot, 0 when another read of `n` had.
    #[test]
    fn a_copied_turbofish_call_reads_its_own_dimension() {
        let src = format!(
            "{ROW}fn f(buf: [i32; 6], c: bool) -> i64 {{ let n = 3; \
             defer print(row::<2, n>(buf, 1).dim(0)); if c {{ return 1; }} return 2; }}"
        );
        let (module, sema, mut interner) = analyze(&src);
        let (_, overlay) = lower_defers(&module, &sema, &mut interner)
            .expect("no E0314")
            .expect("the module has a defer");
        let original: Vec<_> = sema.dim_snapshots.values().collect();
        assert_eq!(original.len(), 1, "{original:?}");
        // One copy per exit (`return 1`, `return 2`), each with the original's reads.
        assert_eq!(
            overlay.dim_snapshots.len(),
            2,
            "{:?}",
            overlay.dim_snapshots
        );
        for snaps in overlay.dim_snapshots.values() {
            assert_eq!(snaps, original[0]);
        }
        let (program, diags) = crate::lower_program(&module, &sema, &mut interner);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        for f in &program.funcs {
            let errs = wukong_mir::verify::verify_function(f);
            assert!(errs.is_empty(), "{errs:?}");
        }
    }

    /// The whole pipeline lowers a `defer` program to verifier-clean MIR with no C0001.
    #[test]
    fn a_defer_program_lowers_to_verified_mir() {
        let (module, sema, mut interner) = analyze(
            "fn g(x: i32) {}\n\
             fn f(n: i32) -> i32 {\n\
                 let mut total = 0;\n\
                 defer g(total);\n\
                 for i in 0..n { defer total += 10; if i == 1 { continue; } total += i; }\n\
                 let r = loop { defer total += 1; if total > 30 { break total * 2; } };\n\
                 r + 1\n\
             }",
        );
        let (program, diags) = crate::lower_program(&module, &sema, &mut interner);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        for f in &program.funcs {
            let errs = wukong_mir::verify::verify_function(f);
            assert!(errs.is_empty(), "{errs:?}");
        }
    }
}
