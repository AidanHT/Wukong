//! Pre-lowering AST canonicalization — spelling normalization for the kernel recognizers.
//!
//! Every recognizer in this crate matches a *syntactic* loop-nest shape on the raw AST, before
//! `wukong_opt` has run mem2reg or CSE. That makes recognition brittle in a way that has nothing to
//! do with the arithmetic: hoisting a row base into a local (`let ib = i*256;` then `a[ib + p]`) is
//! plain common-subexpression elimination — it changes no floating-point result and every optimizer
//! performs it anyway — yet it moved the index arithmetic out of the index expression, so
//! `match_row_col` no longer saw `i*N + p` and the whole GEMM fell back to a scalar nest. Measured
//! across the recognizer family, binding *any* index subexpression to a local had a 0/17 hit rate.
//!
//! This pass runs once, on the whole module, at the very top of [`crate::lower_program`], and
//! rewrites the AST into the canonical spelling the matchers already understand. It is a *pure*
//! source-to-source rewrite: every rewrite below is value-identical at every node, so lowering the
//! normalized tree produces the same results as lowering the original.
//!
//! # Why it needs no new sema entries
//!
//! Substitution moves an already-type-checked subtree to a new position and keeps its `NodeId`s, so
//! `sema.types` answers for every node exactly as before. `SemaResult::defs` is keyed by *name*, not
//! by `NodeId`, and local resolution in `FnLowerer` is by name as well, so a duplicated `NodeId` is
//! read-only aliasing, never a collision. The soundness conditions that make this true are enforced
//! in [`Subst::collect`]:
//!
//! * the binding is declared **exactly once** in the function and is never a top-level `const`/`fn`
//!   name, so every `Path` occurrence of that name in the body denotes it (no shadowing, no capture);
//! * neither the binding nor any free variable of its initializer is ever **assigned**, so the
//!   substituted expression evaluates to the same value at the use site as it did at the `let`;
//! * the initializer is a **pure integer** expression (literals, those variables, and `+ - *`), so
//!   duplicating it cannot duplicate a side effect, a trap (no `/`, no `%`, no call, no index) or a
//!   float rounding, and re-evaluating it cannot observe a different value;
//! * every use site's recorded type **equals** the initializer's recorded type, which is what rules
//!   out a narrowing/widening `let` annotation (`let ib: i32 = i * 256;` with `i: i64`) silently
//!   turning into full-width arithmetic.
//!
//! Once every use is substituted the `let` is dead and is deleted — that deletion is the actual
//! enabler, because a leftover statement in a loop body is itself enough to make a whole-nest
//! matcher (which requires the body to be exactly the inner `for`) decline.

use wukong_ast::{
    Block, Expr, ExprKind, ForIter, Module, PatKind, Pattern, Stmt, StmtKind, UnOp, VariantPat,
};
use wukong_sema::SemaResult;
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Symbol};
use wukong_types::Ty;

/// Bound on the substitute-and-retry fixpoint. Each round deletes at least one `let` or stops, so
/// this is a belt-and-braces cap, not the normal exit; chained bases (`let ib = i*N; let ic = ib+C;`)
/// need one round per link.
const MAX_ROUNDS: usize = 8;

/// Canonicalize `module` for the recognizers. Returns `None` — having touched nothing and cloned
/// nothing — when no function in the module admits a rewrite, which keeps the output byte-identical
/// for the programs this cannot help.
pub(crate) fn canonicalize_module(module: &Module, sema: &SemaResult) -> Option<Module> {
    // Names that a single-segment `Path` could denote *without* being the local we are substituting
    // (a top-level `const`, a `fn`, a struct/enum). A local of the same name shadows them inside the
    // function, so a body-wide substitution keyed on the bare name would rewrite the wrong node.
    let mut shadowed: HashSet<Symbol> = HashSet::default();
    for d in &sema.defs.defs {
        shadowed.insert(d.name);
    }
    for name in sema.consts.keys() {
        shadowed.insert(*name);
    }

    let mut out = module.clone();
    let mut changed = false;
    for item in &mut out.items {
        if let wukong_ast::ItemKind::Fn(f) = &mut item.kind {
            let Some(body) = f.body.as_mut() else { continue };
            let params: Vec<Symbol> = f.params.iter().map(|p| p.name.sym).collect();
            changed |= canonicalize_fn(body, &params, sema, &shadowed);
        }
    }
    if changed {
        Some(out)
    } else {
        None
    }
}

/// Run the fixpoint over one function body. Returns whether anything was rewritten.
fn canonicalize_fn(
    body: &mut Block,
    params: &[Symbol],
    sema: &SemaResult,
    shadowed: &HashSet<Symbol>,
) -> bool {
    let mut any = false;
    for _ in 0..MAX_ROUNDS {
        let subst = Subst::collect(body, params, sema, shadowed);
        if subst.map.is_empty() {
            break;
        }
        subst.apply_block(body);
        any = true;
    }
    any
}

/// The substitutions chosen for one round: binding name -> the initializer to inline at every use.
struct Subst {
    map: HashMap<Symbol, Expr>,
}

/// Per-function facts gathered in one scan, used to prove a candidate binding safe.
#[derive(Default)]
struct FnFacts {
    /// How many times each name is *declared* (parameter, `let`, `for` binding, pattern binding).
    /// A name declared more than once may be shadowed, so a body-wide rewrite of it is unsound.
    decls: HashMap<Symbol, u32>,
    /// Names that are the root of an assignment target anywhere in the function (`x = …`, `x += …`).
    /// Only the *root* counts: in `a[i] = v` the buffer `a` is assigned, `i` is merely read.
    assigned: HashSet<Symbol>,
    /// Names that appear under a `&` / `&mut`, whose address therefore escapes.
    addressed: HashSet<Symbol>,
    /// The function contains a `defer`, whose deferred expression runs at a scope exit where a
    /// substituted loop variable no longer holds the value it had at the `let`. Decline wholesale.
    has_defer: bool,
}

impl Subst {
    /// Choose every binding in `body` that can be safely forward-substituted into its uses.
    fn collect(
        body: &Block,
        params: &[Symbol],
        sema: &SemaResult,
        shadowed: &HashSet<Symbol>,
    ) -> Subst {
        let mut facts = FnFacts::default();
        for p in params {
            *facts.decls.entry(*p).or_insert(0) += 1;
        }
        scan_block(body, &mut facts);

        let mut map: HashMap<Symbol, Expr> = HashMap::default();
        if facts.has_defer {
            return Subst { map };
        }
        let mut cands = Vec::new();
        collect_candidates(body, &mut cands);
        for (name, init) in cands {
            if map.contains_key(&name) {
                continue;
            }
            if !binding_is_substitutable(name, init, &facts, sema, shadowed) {
                continue;
            }
            // Every use must be in index position, and must carry the initializer's exact type.
            let init_ty = match sema.types.get(&init.id) {
                Some(t) => t,
                None => continue,
            };
            let mut uses = UseScan {
                name,
                init_ty,
                sema,
                count: 0,
                ok: true,
            };
            uses.block(body, false);
            if uses.ok && uses.count > 0 {
                map.insert(name, init.clone());
            }
        }
        Subst { map }
    }

    fn apply_block(&self, b: &mut Block) {
        b.stmts.retain(|s| !self.is_dead_let(s));
        for s in &mut b.stmts {
            self.apply_stmt(s);
        }
        if let Some(t) = &mut b.tail {
            self.apply_expr(t);
        }
    }

    /// A `let` whose binding was fully substituted has no readers left, and its initializer is pure
    /// by construction, so the statement is dead. Deleting it is the point: a leftover statement in
    /// a loop body makes every whole-nest matcher decline.
    fn is_dead_let(&self, s: &Stmt) -> bool {
        match &s.kind {
            StmtKind::Let {
                pat: Pattern {
                    kind: PatKind::Ident(sym),
                    ..
                },
                init: Some(_),
                ..
            } => self.map.contains_key(sym),
            _ => false,
        }
    }

    fn apply_stmt(&self, s: &mut Stmt) {
        match &mut s.kind {
            StmtKind::Let { init, .. } => {
                if let Some(e) = init {
                    self.apply_expr(e);
                }
            }
            StmtKind::Assign { target, value, .. } => {
                self.apply_expr(target);
                self.apply_expr(value);
            }
            StmtKind::Expr(e) | StmtKind::Defer(e) => self.apply_expr(e),
            StmtKind::Return(e) | StmtKind::Break(_, e) => {
                if let Some(e) = e {
                    self.apply_expr(e);
                }
            }
            StmtKind::Continue(_) => {}
            StmtKind::While { cond, body, .. } => {
                self.apply_expr(cond);
                self.apply_block(body);
            }
            StmtKind::For { iter, body, .. } => {
                match iter {
                    ForIter::Range {
                        start, end, step, ..
                    } => {
                        self.apply_expr(start);
                        if let Some(e) = end {
                            self.apply_expr(e);
                        }
                        if let Some(e) = step {
                            self.apply_expr(e);
                        }
                    }
                    ForIter::Expr(e) => self.apply_expr(e),
                }
                self.apply_block(body);
            }
        }
    }

    fn apply_expr(&self, e: &mut Expr) {
        if let ExprKind::Path(p) = &e.kind {
            if p.is_single() {
                if let Some(rep) = self.map.get(&p.first().sym) {
                    *e = rep.clone();
                    return;
                }
            }
        }
        match &mut e.kind {
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::Path(_)
            | ExprKind::SizeOf(_)
            | ExprKind::AlignOf(_) => {}
            ExprKind::Unary { expr, .. }
            | ExprKind::Cast { expr, .. }
            | ExprKind::Field { base: expr, .. }
            | ExprKind::TupleField { base: expr, .. } => self.apply_expr(expr),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.apply_expr(lhs);
                self.apply_expr(rhs);
            }
            ExprKind::Call { callee, args, .. } => {
                self.apply_expr(callee);
                for a in args {
                    self.apply_expr(a);
                }
            }
            ExprKind::Index { base, indices } => {
                self.apply_expr(base);
                for i in indices {
                    self.apply_expr(i);
                }
            }
            ExprKind::StructLit { fields, rest, .. } => {
                for f in fields {
                    self.apply_expr(&mut f.value);
                }
                if let Some(r) = rest {
                    self.apply_expr(r);
                }
            }
            ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
                for x in xs {
                    self.apply_expr(x);
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.apply_expr(value);
                self.apply_expr(count);
            }
            ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => self.apply_block(b),
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.apply_expr(cond);
                self.apply_block(then_branch);
                if let Some(e) = else_branch {
                    self.apply_expr(e);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.apply_expr(scrutinee);
                for a in arms {
                    if let Some(g) = &mut a.guard {
                        self.apply_expr(g);
                    }
                    self.apply_expr(&mut a.body);
                }
            }
        }
    }
}

/// Is this binding safe to inline at every one of its uses? Checks everything that does not depend
/// on the use sites themselves (those are checked by [`UseScan`]).
fn binding_is_substitutable(
    name: Symbol,
    init: &Expr,
    facts: &FnFacts,
    sema: &SemaResult,
    shadowed: &HashSet<Symbol>,
) -> bool {
    if shadowed.contains(&name) || facts.decls.get(&name) != Some(&1) {
        return false;
    }
    if facts.assigned.contains(&name) || facts.addressed.contains(&name) {
        return false;
    }
    // Integer only: a float `let` would move a rounding step, and an aggregate would move a copy.
    if !matches!(sema.types.get(&init.id), Some(Ty::Scalar(s)) if s.is_int()) {
        return false;
    }
    pure_index_expr(init, facts, sema, shadowed)
}

/// Is `e` an expression that may be duplicated and re-evaluated at an arbitrary later point in the
/// same function without changing what the program computes?
///
/// Deliberately tiny: integer literals, immutable integer variables, and `+ - *` (which wrap
/// identically however many times they are evaluated). `/` and `%` are excluded because they can
/// trap, and moving a trap under a branch that may not be taken *removes* a fault the original
/// program had. Calls, indexing and field access are excluded because their value depends on memory
/// that a later statement may have written.
fn pure_index_expr(
    e: &Expr,
    facts: &FnFacts,
    sema: &SemaResult,
    shadowed: &HashSet<Symbol>,
) -> bool {
    match &e.kind {
        ExprKind::Int(_) => true,
        ExprKind::Path(p) => {
            if !p.is_single() {
                return false;
            }
            let sym = p.first().sym;
            // A free variable of the initializer must be a local that cannot change between the
            // `let` and any use, and cannot be a differently-scoped homonym.
            if shadowed.contains(&sym) || facts.decls.get(&sym) != Some(&1) {
                return false;
            }
            if facts.assigned.contains(&sym) || facts.addressed.contains(&sym) {
                return false;
            }
            matches!(sema.types.get(&e.id), Some(Ty::Scalar(s)) if s.is_int())
        }
        ExprKind::Unary {
            op: UnOp::Neg,
            expr,
        } => pure_index_expr(expr, facts, sema, shadowed),
        ExprKind::Binary { op, lhs, rhs } => {
            use wukong_ast::BinOp::*;
            matches!(op, Add | Sub | Mul)
                && pure_index_expr(lhs, facts, sema, shadowed)
                && pure_index_expr(rhs, facts, sema, shadowed)
        }
        _ => false,
    }
}

/// Walks the body checking that every occurrence of `name` sits in **index position** and carries
/// exactly `init_ty`. Index position is the point of the whole pass: those are the occurrences the
/// recognizers' affine analysis needs to see through. Restricting to them also bounds the blast
/// radius — a binding read anywhere else is left completely alone.
struct UseScan<'a> {
    name: Symbol,
    init_ty: &'a Ty,
    sema: &'a SemaResult,
    count: u32,
    ok: bool,
}

impl UseScan<'_> {
    fn block(&mut self, b: &Block, in_index: bool) {
        for s in &b.stmts {
            self.stmt(s, in_index);
        }
        if let Some(t) = &b.tail {
            self.expr(t, in_index);
        }
    }

    fn stmt(&mut self, s: &Stmt, in_index: bool) {
        match &s.kind {
            StmtKind::Let { init, .. } => {
                if let Some(e) = init {
                    self.expr(e, in_index);
                }
            }
            StmtKind::Assign { target, value, .. } => {
                self.expr(target, in_index);
                self.expr(value, in_index);
            }
            StmtKind::Expr(e) | StmtKind::Defer(e) => self.expr(e, in_index),
            StmtKind::Return(e) | StmtKind::Break(_, e) => {
                if let Some(e) = e {
                    self.expr(e, in_index);
                }
            }
            StmtKind::Continue(_) => {}
            StmtKind::While { cond, body, .. } => {
                self.expr(cond, in_index);
                self.block(body, in_index);
            }
            StmtKind::For { iter, body, .. } => {
                match iter {
                    ForIter::Range {
                        start, end, step, ..
                    } => {
                        self.expr(start, in_index);
                        if let Some(e) = end {
                            self.expr(e, in_index);
                        }
                        if let Some(e) = step {
                            self.expr(e, in_index);
                        }
                    }
                    ForIter::Expr(e) => self.expr(e, in_index),
                }
                self.block(body, in_index);
            }
        }
    }

    fn expr(&mut self, e: &Expr, in_index: bool) {
        if !self.ok {
            return;
        }
        match &e.kind {
            ExprKind::Path(p) if p.is_single() && p.first().sym == self.name => {
                // A use outside an index, or one whose recorded type differs from the initializer's
                // (a narrowing/widening `let` annotation), disqualifies the whole binding.
                if !in_index || self.sema.types.get(&e.id) != Some(self.init_ty) {
                    self.ok = false;
                } else {
                    self.count += 1;
                }
            }
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::Path(_)
            | ExprKind::SizeOf(_)
            | ExprKind::AlignOf(_) => {}
            ExprKind::Unary { expr, .. }
            | ExprKind::Cast { expr, .. }
            | ExprKind::Field { base: expr, .. }
            | ExprKind::TupleField { base: expr, .. } => self.expr(expr, in_index),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(lhs, in_index);
                self.expr(rhs, in_index);
            }
            ExprKind::Call { callee, args, .. } => {
                self.expr(callee, in_index);
                for a in args {
                    self.expr(a, in_index);
                }
            }
            ExprKind::Index { base, indices } => {
                self.expr(base, in_index);
                for i in indices {
                    self.expr(i, true);
                }
            }
            ExprKind::StructLit { fields, rest, .. } => {
                for f in fields {
                    self.expr(&f.value, in_index);
                }
                if let Some(r) = rest {
                    self.expr(r, in_index);
                }
            }
            ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
                for x in xs {
                    self.expr(x, in_index);
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.expr(value, in_index);
                self.expr(count, in_index);
            }
            ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => self.block(b, in_index),
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.expr(cond, in_index);
                self.block(then_branch, in_index);
                if let Some(x) = else_branch {
                    self.expr(x, in_index);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee, in_index);
                for a in arms {
                    if let Some(g) = &a.guard {
                        self.expr(g, in_index);
                    }
                    self.expr(&a.body, in_index);
                }
            }
        }
    }
}

// ---- the one-shot fact scan --------------------------------------------------------------------

fn scan_block(b: &Block, f: &mut FnFacts) {
    for s in &b.stmts {
        scan_stmt(s, f);
    }
    if let Some(t) = &b.tail {
        scan_expr(t, f);
    }
}

fn scan_pattern(p: &Pattern, f: &mut FnFacts) {
    match &p.kind {
        PatKind::Ident(s) => *f.decls.entry(*s).or_insert(0) += 1,
        PatKind::Tuple(subs) | PatKind::Or(subs) => {
            for s in subs {
                scan_pattern(s, f);
            }
        }
        PatKind::Variant { fields, .. } => match fields {
            VariantPat::Tuple(subs) => {
                for s in subs {
                    scan_pattern(s, f);
                }
            }
            VariantPat::Struct(fps) => {
                for fp in fps {
                    scan_pattern(&fp.pat, f);
                }
            }
        },
        PatKind::Wildcard
        | PatKind::Unit
        | PatKind::Int { .. }
        | PatKind::Char(_)
        | PatKind::Bool(_)
        | PatKind::Path(_)
        | PatKind::Range { .. } => {}
    }
}

/// The name whose *storage* an assignment target names — the root of a `x`, `x.f`, `x[i]`, `x.0`
/// chain. `x[i] = v` writes through `x`, so `x` is recorded; `i` is only read and is not.
fn assign_root(e: &Expr) -> Option<Symbol> {
    match &e.kind {
        ExprKind::Path(p) if p.is_single() => Some(p.first().sym),
        ExprKind::Field { base, .. }
        | ExprKind::TupleField { base, .. }
        | ExprKind::Index { base, .. }
        | ExprKind::Unary { expr: base, .. } => assign_root(base),
        _ => None,
    }
}

fn scan_stmt(s: &Stmt, f: &mut FnFacts) {
    match &s.kind {
        StmtKind::Let { pat, init, .. } => {
            scan_pattern(pat, f);
            if let Some(e) = init {
                scan_expr(e, f);
            }
        }
        StmtKind::Assign { target, value, .. } => {
            if let Some(root) = assign_root(target) {
                f.assigned.insert(root);
            }
            scan_expr(target, f);
            scan_expr(value, f);
        }
        StmtKind::Expr(e) => scan_expr(e, f),
        StmtKind::Defer(e) => {
            f.has_defer = true;
            scan_expr(e, f);
        }
        StmtKind::Return(e) | StmtKind::Break(_, e) => {
            if let Some(e) = e {
                scan_expr(e, f);
            }
        }
        StmtKind::Continue(_) => {}
        StmtKind::While { cond, body, .. } => {
            scan_expr(cond, f);
            scan_block(body, f);
        }
        StmtKind::For {
            pat, iter, body, ..
        } => {
            scan_pattern(pat, f);
            match iter {
                ForIter::Range {
                    start, end, step, ..
                } => {
                    scan_expr(start, f);
                    if let Some(e) = end {
                        scan_expr(e, f);
                    }
                    if let Some(e) = step {
                        scan_expr(e, f);
                    }
                }
                ForIter::Expr(e) => scan_expr(e, f),
            }
            scan_block(body, f);
        }
    }
}

fn scan_expr(e: &Expr, f: &mut FnFacts) {
    match &e.kind {
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::Path(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_) => {}
        ExprKind::Unary { op, expr } => {
            if matches!(op, UnOp::Ref | UnOp::RefMut) {
                if let Some(root) = assign_root(expr) {
                    f.addressed.insert(root);
                }
            }
            scan_expr(expr, f);
        }
        ExprKind::Cast { expr, .. }
        | ExprKind::Field { base: expr, .. }
        | ExprKind::TupleField { base: expr, .. } => scan_expr(expr, f),
        ExprKind::Binary { lhs, rhs, .. } => {
            scan_expr(lhs, f);
            scan_expr(rhs, f);
        }
        ExprKind::Call { callee, args, .. } => {
            scan_expr(callee, f);
            for a in args {
                scan_expr(a, f);
            }
        }
        ExprKind::Index { base, indices } => {
            scan_expr(base, f);
            for i in indices {
                scan_expr(i, f);
            }
        }
        ExprKind::StructLit { fields, rest, .. } => {
            for fi in fields {
                scan_expr(&fi.value, f);
            }
            if let Some(r) = rest {
                scan_expr(r, f);
            }
        }
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
            for x in xs {
                scan_expr(x, f);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            scan_expr(value, f);
            scan_expr(count, f);
        }
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => scan_block(b, f),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            scan_expr(cond, f);
            scan_block(then_branch, f);
            if let Some(x) = else_branch {
                scan_expr(x, f);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            scan_expr(scrutinee, f);
            for a in arms {
                scan_pattern(&a.pat, f);
                if let Some(g) = &a.guard {
                    scan_expr(g, f);
                }
                scan_expr(&a.body, f);
            }
        }
    }
}

/// Every `let <ident> = <init>;` in the function, innermost blocks included, in source order.
fn collect_candidates<'a>(b: &'a Block, out: &mut Vec<(Symbol, &'a Expr)>) {
    for s in &b.stmts {
        cand_stmt(s, out);
    }
    if let Some(t) = &b.tail {
        cand_expr(t, out);
    }
}

fn cand_stmt<'a>(s: &'a Stmt, out: &mut Vec<(Symbol, &'a Expr)>) {
    match &s.kind {
        StmtKind::Let { pat, init, .. } => {
            if let (PatKind::Ident(sym), Some(e)) = (&pat.kind, init) {
                out.push((*sym, e));
                cand_expr(e, out);
            } else if let Some(e) = init {
                cand_expr(e, out);
            }
        }
        StmtKind::Assign { target, value, .. } => {
            cand_expr(target, out);
            cand_expr(value, out);
        }
        StmtKind::Expr(e) | StmtKind::Defer(e) => cand_expr(e, out),
        StmtKind::Return(e) | StmtKind::Break(_, e) => {
            if let Some(e) = e {
                cand_expr(e, out);
            }
        }
        StmtKind::Continue(_) => {}
        StmtKind::While { cond, body, .. } => {
            cand_expr(cond, out);
            collect_candidates(body, out);
        }
        StmtKind::For { iter, body, .. } => {
            match iter {
                ForIter::Range {
                    start, end, step, ..
                } => {
                    cand_expr(start, out);
                    if let Some(e) = end {
                        cand_expr(e, out);
                    }
                    if let Some(e) = step {
                        cand_expr(e, out);
                    }
                }
                ForIter::Expr(e) => cand_expr(e, out),
            }
            collect_candidates(body, out);
        }
    }
}

fn cand_expr<'a>(e: &'a Expr, out: &mut Vec<(Symbol, &'a Expr)>) {
    match &e.kind {
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => collect_candidates(b, out),
        ExprKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_candidates(then_branch, out);
            if let Some(x) = else_branch {
                cand_expr(x, out);
            }
        }
        ExprKind::Match { arms, .. } => {
            for a in arms {
                cand_expr(&a.body, out);
            }
        }
        _ => {}
    }
}
