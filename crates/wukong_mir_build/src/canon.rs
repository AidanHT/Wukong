//! Pre-lowering AST canonicalization — spelling normalization for the kernel recognizers.
//!
//! Every recognizer in this crate matches a *syntactic* loop-nest shape on the raw AST, before
//! `wukong_opt` has run mem2reg or CSE. That makes recognition brittle in a way that has nothing to
//! do with the arithmetic: hoisting a row base into a local (`let ib = i*256;` then `a[ib + p]`) is
//! plain common-subexpression elimination — it changes no floating-point result and every optimizer
//! performs it anyway — yet it moves the index arithmetic *out* of the index expression, so
//! `match_row_col` no longer sees `i*N + p` and the whole GEMM falls back to a scalar nest. Measured
//! across the recognizer family, binding an index subexpression to a local had a 0/17 hit rate.
//!
//! This pass runs once, on the whole module, at the very top of [`crate::lower_program`], and
//! rewrites the AST into the canonical spelling the matchers already understand. It is a *pure*
//! source-to-source rewrite: every rewrite is value-identical at every node, so lowering the
//! normalized tree computes exactly what lowering the original computed.
//!
//! # Why it needs no new sema entries
//!
//! Substitution moves an already-type-checked subtree to a new position and keeps its `NodeId`s, so
//! `sema.types` answers for every node exactly as before. `SemaResult::defs` is keyed by *name*, not
//! by `NodeId`, and local resolution in `FnLowerer` is by name as well, so a duplicated `NodeId` is
//! read-only aliasing, never a collision.
//!
//! # The soundness conditions
//!
//! A binding is inlined only into the **region** it dominates — the statements after its `let` in the
//! same block, plus that block's tail — and only when, over that whole region:
//!
//! * the binding is never re-declared, never assigned and never `&`-taken, so every `Path` naming it
//!   in the region denotes this binding and holds this binding's value;
//! * no free variable of the initializer is re-declared, assigned or `&`-taken either, so the
//!   initializer evaluates to the same value at a use as it did at the `let` (this is what rejects
//!   `let b = t*4;` in a loop that later does `t = t + 1`);
//! * the initializer is a **pure integer** expression — literals, those variables, and `+ - *` — so
//!   duplicating it cannot duplicate a side effect, cannot duplicate or *move* a trap (no `/`, no
//!   `%`, no call, no index), and cannot move a float rounding;
//! * every use site's recorded type **equals** the initializer's recorded type, which is what rejects
//!   a narrowing/widening annotation (`let ib: i32 = i * 256;` with `i: i64`) silently becoming
//!   full-width arithmetic;
//! * every use is in **index position**. That is where the recognizers' affine analysis needs to see
//!   the arithmetic, and restricting to it bounds the blast radius: a binding read anywhere else is
//!   left completely alone.
//!
//! Once every use is substituted the `let` is dead and is deleted — the deletion is half the point,
//! because a leftover statement in a loop body is itself enough to make a whole-nest matcher (which
//! requires the body to be exactly the inner `for`) decline.
//!
//! Substituting into a nested `let` initializer is not allowed (that is not index position), so a
//! *chain* of bases (`let d = 16; let ib = i*d; … a[ib + p]`) needs one round per link: the pass is
//! a fixpoint, and each round strictly removes at least one `let`.

use wukong_ast::{
    Block, Expr, ExprKind, ForIter, ItemKind, Module, PatKind, Pattern, Stmt, StmtKind, UnOp,
    VariantPat,
};
use wukong_sema::SemaResult;
use wukong_span::{FxHashSet as HashSet, Symbol};
use wukong_types::Ty;

/// Bound on the substitute-and-retry fixpoint. Every round that reports a change has deleted at
/// least one `let`, so the loop terminates on its own; this is only a belt-and-braces cap. Chained
/// bases need one round per link.
const MAX_ROUNDS: usize = 8;

/// Canonicalize `module` for the recognizers. Returns `None` — having cloned nothing — when no
/// function admits a rewrite, so the output stays byte-identical for the programs this cannot help.
pub(crate) fn canonicalize_module(module: &Module, sema: &SemaResult) -> Option<Module> {
    let consts = int_literal_consts(sema);
    // Read-only feasibility first. Every legality test below is a pure analysis, so it can be run on
    // the borrowed module; only the rewrite needs an owned one. A compile this pass cannot help must
    // not pay for a deep copy of the AST — compile speed is one of this compiler's few genuine
    // strengths. A later fixpoint round can only find work because an earlier one rewrote something,
    // so "round 1 finds nothing" is a sound answer for the whole fixpoint.
    if !module.items.iter().any(|item| {
        let ItemKind::Fn(f) = &item.kind else {
            return false;
        };
        let Some(body) = f.body.as_ref() else {
            return false;
        };
        if block_has_defer(body) {
            return false;
        }
        let params: Vec<Symbol> = f.params.iter().map(|p| p.name.sym).collect();
        !consts_to_fold(body, &params, &consts, sema).is_empty() || block_admits_any(body, sema)
    }) {
        return None;
    }

    let mut out = module.clone();
    let mut changed = false;
    for item in &mut out.items {
        let ItemKind::Fn(f) = &mut item.kind else {
            continue;
        };
        let params: Vec<Symbol> = f.params.iter().map(|p| p.name.sym).collect();
        let Some(body) = f.body.as_mut() else { continue };
        // A `defer` runs its expression at a *scope exit*, where a substituted loop variable no
        // longer holds the value it had at the `let`. Decline the whole function rather than reason
        // about it (lowering rejects `defer` anyway).
        if block_has_defer(body) {
            continue;
        }
        changed |= fold_consts(body, &params, &consts, sema);
        for _ in 0..MAX_ROUNDS {
            if !canon_block(body, sema) {
                break;
            }
            changed = true;
        }
    }
    if changed {
        Some(out)
    } else {
        None
    }
}

/// Module-level `const`s that may be inlined at their uses: integer scalars whose initializer is a
/// plain integer literal that sema stamped with the *declared* type.
///
/// This one is MIR-identical by construction rather than merely value-identical: `FnLowerer`'s
/// `Path` arm already lowers a const reference by inlining this very initializer expression, because
/// the def map records only a const's type, not its value. Doing it in the AST first changes nothing
/// about what is lowered — it only lets the recognizers, which run *before* lowering, see the
/// literal. Without it `as_dim` reads a bare const path as an opaque `Dim::Var`, the strides and
/// bounds still compare equal symbolically, and then `dim_value` finds no local slot of that name
/// and the whole nest declines. That is the sharp edge documented in
/// `examples/gpt2_forward_bench.wk` — "only literal or local-variable dims resolve; a bare imported
/// const used directly as a dim does not" — which is why that model re-binds every dimension to a
/// local `let` before use.
///
/// The initializer's own recorded type must equal the declared type. Sema re-stamps an *adapting*
/// literal with the annotation, but an initializer that type-checks merely via `compatible` keeps
/// its own, possibly narrower, type; inlining that where the declared type is expected would change
/// the arithmetic width. Requiring a literal (optionally negated) is what makes the two agree.
fn int_literal_consts(sema: &SemaResult) -> Vec<(Symbol, &Expr)> {
    // Substitutions of distinct names are independent — each replacement is a literal containing no
    // `Path` — so the order this is iterated in cannot affect the result.
    sema.consts
        .iter()
        .filter_map(|(name, init)| {
            let Some(wukong_sema::DefKind::Const(decl)) =
                sema.defs.lookup(*name).map(|d| &d.kind)
            else {
                return None;
            };
            if !matches!(decl, Ty::Scalar(s) if s.is_int()) {
                return None;
            }
            if !is_int_literal(init) || sema.types.get(&init.id) != Some(decl) {
                return None;
            }
            Some((*name, init))
        })
        .collect()
}

/// An integer literal, or a negated one (`const NEG: i64 = -1;` — sema re-stamps through the `-`).
fn is_int_literal(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Int(_) => true,
        ExprKind::Unary {
            op: UnOp::Neg,
            expr,
        } => matches!(expr.kind, ExprKind::Int(_)),
        _ => false,
    }
}

/// Which of `consts` this function body actually uses and may have folded in. Read-only, so it also
/// answers the feasibility question before anything is cloned.
fn consts_to_fold<'a>(
    body: &Block,
    params: &[Symbol],
    consts: &'a [(Symbol, &'a Expr)],
    sema: &SemaResult,
) -> Vec<(Symbol, &'a Expr)> {
    if consts.is_empty() {
        return Vec::new();
    }
    // `lower_expr` resolves a single-segment `Path` in the LOCALS first and only falls through to
    // `sema.consts`, so a function that declares — or takes as a parameter — a name equal to a
    // const's shadows it there and the const must be left alone in this function.
    let mut decls = RegionFacts::default();
    for p in params {
        decls.declared.insert(*p);
    }
    scan_block(body, &mut decls);

    consts
        .iter()
        .filter(|(name, init)| {
            if decls.declared.contains(name) {
                return false;
            }
            let Some(ty) = sema.types.get(&init.id) else {
                return false;
            };
            let mut uses = UseScan {
                name: *name,
                init_ty: ty,
                sema,
                require_index: false,
                count: 0,
                ok: true,
            };
            uses.block(body, false);
            // `ok` can only be false if a use site is typed differently from the initializer, which
            // sema should never produce for a const — treat it as a decline rather than assume.
            uses.ok && uses.count > 0
        })
        .copied()
        .collect()
}

/// Inline every foldable const into one function body. Returns whether anything was rewritten.
fn fold_consts(
    body: &mut Block,
    params: &[Symbol],
    consts: &[(Symbol, &Expr)],
    sema: &SemaResult,
) -> bool {
    let chosen = consts_to_fold(body, params, consts, sema);
    for (name, init) in &chosen {
        subst_block(body, *name, init);
    }
    !chosen.is_empty()
}

/// Read-only twin of [`canon_block`]: would any `let` in this block, or in a block nested inside it,
/// be forward-substituted? Used to decide whether the module is worth cloning at all.
fn block_admits_any(b: &Block, sema: &SemaResult) -> bool {
    for (i, s) in b.stmts.iter().enumerate() {
        if let StmtKind::Let {
            pat:
                Pattern {
                    kind: PatKind::Ident(x),
                    ..
                },
            init: Some(e),
            ..
        } = &s.kind
        {
            if s.attrs.is_empty() && region_admits(&b.stmts[i + 1..], b.tail.as_deref(), *x, e, sema)
            {
                return true;
            }
        }
    }
    b.stmts.iter().any(|s| stmt_admits_any(s, sema))
        || b.tail.as_deref().is_some_and(|t| expr_admits_any(t, sema))
}

fn stmt_admits_any(s: &Stmt, sema: &SemaResult) -> bool {
    match &s.kind {
        StmtKind::Let { init, .. } => init.as_ref().is_some_and(|e| expr_admits_any(e, sema)),
        StmtKind::Assign { target, value, .. } => {
            expr_admits_any(target, sema) || expr_admits_any(value, sema)
        }
        StmtKind::Expr(e) | StmtKind::Defer(e) => expr_admits_any(e, sema),
        StmtKind::Return(e) | StmtKind::Break(_, e) => {
            e.as_ref().is_some_and(|e| expr_admits_any(e, sema))
        }
        StmtKind::Continue(_) => false,
        StmtKind::While { cond, body, .. } => {
            expr_admits_any(cond, sema) || block_admits_any(body, sema)
        }
        StmtKind::For { iter, body, .. } => {
            let it = match iter {
                ForIter::Range {
                    start, end, step, ..
                } => {
                    expr_admits_any(start, sema)
                        || end.as_ref().is_some_and(|e| expr_admits_any(e, sema))
                        || step.as_ref().is_some_and(|e| expr_admits_any(e, sema))
                }
                ForIter::Expr(e) => expr_admits_any(e, sema),
            };
            it || block_admits_any(body, sema)
        }
    }
}

fn expr_admits_any(e: &Expr, sema: &SemaResult) -> bool {
    match &e.kind {
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => block_admits_any(b, sema),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            expr_admits_any(cond, sema)
                || block_admits_any(then_branch, sema)
                || else_branch.as_ref().is_some_and(|x| expr_admits_any(x, sema))
        }
        ExprKind::Match { scrutinee, arms } => {
            expr_admits_any(scrutinee, sema)
                || arms.iter().any(|a| {
                    a.guard.as_ref().is_some_and(|g| expr_admits_any(g, sema))
                        || expr_admits_any(&a.body, sema)
                })
        }
        ExprKind::Unary { expr, .. }
        | ExprKind::Cast { expr, .. }
        | ExprKind::Field { base: expr, .. }
        | ExprKind::TupleField { base: expr, .. } => expr_admits_any(expr, sema),
        ExprKind::Binary { lhs, rhs, .. } => {
            expr_admits_any(lhs, sema) || expr_admits_any(rhs, sema)
        }
        ExprKind::Call { callee, args, .. } => {
            expr_admits_any(callee, sema) || args.iter().any(|a| expr_admits_any(a, sema))
        }
        ExprKind::Index { base, indices } => {
            expr_admits_any(base, sema) || indices.iter().any(|i| expr_admits_any(i, sema))
        }
        ExprKind::StructLit { fields, rest, .. } => {
            fields.iter().any(|f| expr_admits_any(&f.value, sema))
                || rest.as_ref().is_some_and(|r| expr_admits_any(r, sema))
        }
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
            xs.iter().any(|x| expr_admits_any(x, sema))
        }
        ExprKind::ArrayRepeat { value, count } => {
            expr_admits_any(value, sema) || expr_admits_any(count, sema)
        }
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::Path(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_) => false,
    }
}

/// Rewrite one block: first every `let` it declares itself (each against the remainder of *this*
/// block, which is exactly the region that binding dominates), then, recursively, the blocks nested
/// inside the statements that survive. Returns whether anything was rewritten.
fn canon_block(b: &mut Block, sema: &SemaResult) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i < b.stmts.len() {
        // A `let <ident> = <init>;` with no statement attributes (dropping the statement would drop
        // them). Cloning the initializer up front releases the borrow on `b` for the rewrite below.
        let cand = match &b.stmts[i].kind {
            StmtKind::Let {
                pat:
                    Pattern {
                        kind: PatKind::Ident(x),
                        ..
                    },
                init: Some(e),
                ..
            } if b.stmts[i].attrs.is_empty() => Some((*x, e.clone())),
            _ => None,
        };
        if let Some((x, init)) = cand {
            if region_admits(&b.stmts[i + 1..], b.tail.as_deref(), x, &init, sema) {
                let rest = &mut b.stmts[i + 1..];
                for s in rest.iter_mut() {
                    subst_stmt(s, x, &init);
                }
                if let Some(t) = b.tail.as_mut() {
                    subst_expr(t, x, &init);
                }
                b.stmts.remove(i);
                changed = true;
                continue; // the statement that shifted into slot `i` has not been looked at yet
            }
        }
        i += 1;
    }
    for s in &mut b.stmts {
        changed |= canon_stmt(s, sema);
    }
    if let Some(t) = b.tail.as_mut() {
        changed |= canon_expr(t, sema);
    }
    changed
}

fn canon_stmt(s: &mut Stmt, sema: &SemaResult) -> bool {
    match &mut s.kind {
        StmtKind::Let { init, .. } => init.as_mut().is_some_and(|e| canon_expr(e, sema)),
        StmtKind::Assign { target, value, .. } => {
            canon_expr(target, sema) | canon_expr(value, sema)
        }
        StmtKind::Expr(e) | StmtKind::Defer(e) => canon_expr(e, sema),
        StmtKind::Return(e) | StmtKind::Break(_, e) => {
            e.as_mut().is_some_and(|e| canon_expr(e, sema))
        }
        StmtKind::Continue(_) => false,
        StmtKind::While { cond, body, .. } => canon_expr(cond, sema) | canon_block(body, sema),
        StmtKind::For { iter, body, .. } => {
            let mut c = match iter {
                ForIter::Range {
                    start, end, step, ..
                } => {
                    canon_expr(start, sema)
                        | end.as_mut().is_some_and(|e| canon_expr(e, sema))
                        | step.as_mut().is_some_and(|e| canon_expr(e, sema))
                }
                ForIter::Expr(e) => canon_expr(e, sema),
            };
            c |= canon_block(body, sema);
            c
        }
    }
}

fn canon_expr(e: &mut Expr, sema: &SemaResult) -> bool {
    match &mut e.kind {
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => canon_block(b, sema),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            canon_expr(cond, sema)
                | canon_block(then_branch, sema)
                | else_branch.as_mut().is_some_and(|x| canon_expr(x, sema))
        }
        ExprKind::Match { scrutinee, arms } => {
            let mut c = canon_expr(scrutinee, sema);
            for a in arms {
                if let Some(g) = &mut a.guard {
                    c |= canon_expr(g, sema);
                }
                c |= canon_expr(&mut a.body, sema);
            }
            c
        }
        ExprKind::Unary { expr, .. }
        | ExprKind::Cast { expr, .. }
        | ExprKind::Field { base: expr, .. }
        | ExprKind::TupleField { base: expr, .. } => canon_expr(expr, sema),
        ExprKind::Binary { lhs, rhs, .. } => canon_expr(lhs, sema) | canon_expr(rhs, sema),
        ExprKind::Call { callee, args, .. } => {
            let mut c = canon_expr(callee, sema);
            for a in args {
                c |= canon_expr(a, sema);
            }
            c
        }
        ExprKind::Index { base, indices } => {
            let mut c = canon_expr(base, sema);
            for i in indices {
                c |= canon_expr(i, sema);
            }
            c
        }
        ExprKind::StructLit { fields, rest, .. } => {
            let mut c = false;
            for f in fields {
                c |= canon_expr(&mut f.value, sema);
            }
            if let Some(r) = rest {
                c |= canon_expr(r, sema);
            }
            c
        }
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
            let mut c = false;
            for x in xs {
                c |= canon_expr(x, sema);
            }
            c
        }
        ExprKind::ArrayRepeat { value, count } => {
            canon_expr(value, sema) | canon_expr(count, sema)
        }
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::Path(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_) => false,
    }
}

// ---- legality ----------------------------------------------------------------------------------

/// What the region does to the *names* a substitution depends on. One scan answers all of it.
#[derive(Default)]
struct RegionFacts {
    /// Names the region re-declares (a `let`, a `for` binding, a `match` pattern binding). A name
    /// re-declared in the region may denote a different binding at a use, so neither the substituted
    /// binding nor any free variable of its initializer may appear here.
    declared: HashSet<Symbol>,
    /// Names that are the *root* of an assignment target in the region. Only the root counts: in
    /// `a[i] = v` the buffer `a` is written and `i` is merely read.
    assigned: HashSet<Symbol>,
    /// Names whose address is taken (`&x` / `&mut x`) in the region.
    addressed: HashSet<Symbol>,
}

/// May `x`, bound to `init`, be inlined at every one of its uses in this region?
fn region_admits(
    stmts: &[Stmt],
    tail: Option<&Expr>,
    x: Symbol,
    init: &Expr,
    sema: &SemaResult,
) -> bool {
    // Integer only: a float binding would move a rounding step and an aggregate would move a copy.
    let Some(init_ty) = sema.types.get(&init.id) else {
        return false;
    };
    if !matches!(init_ty, Ty::Scalar(s) if s.is_int()) {
        return false;
    }
    // Reject on the initializer's SHAPE before scanning anything. Everything below this line is
    // linear in the size of the region, and the overwhelming majority of `let`s in real code
    // (`let n = a.len()`, `let t = f(x)`, `let q = a / b`) die here for free.
    if !pure_shape(init, sema) {
        return false;
    }
    let mut facts = RegionFacts::default();
    for s in stmts {
        scan_stmt(s, &mut facts);
    }
    if let Some(t) = tail {
        scan_expr(t, &mut facts);
    }
    if facts.declared.contains(&x) || facts.assigned.contains(&x) || facts.addressed.contains(&x) {
        return false;
    }
    if !pure_index_expr(init, &facts, sema) {
        return false;
    }
    let mut uses = UseScan {
        name: x,
        init_ty,
        sema,
        require_index: true,
        count: 0,
        ok: true,
    };
    for s in stmts {
        uses.stmt(s, false);
    }
    if let Some(t) = tail {
        uses.expr(t, false);
    }
    uses.ok && uses.count > 0
}

/// The region-independent half of the purity test: is `e` built only from integer literals, integer
/// variables and `+ - *`?
///
/// Deliberately tiny. `+ - *` wrap identically however many times they are evaluated; `/` and `%`
/// are excluded because they can trap, and re-siting a trap under a branch that may not be taken
/// *removes* a fault the original program had. Calls, indexing and field access are excluded
/// because their value depends on memory a later statement may have written. This is a cheap,
/// purely local check, so `region_admits` runs it before any scan of the region.
fn pure_shape(e: &Expr, sema: &SemaResult) -> bool {
    match &e.kind {
        ExprKind::Int(_) => true,
        ExprKind::Path(p) => {
            p.is_single() && matches!(sema.types.get(&e.id), Some(Ty::Scalar(s)) if s.is_int())
        }
        ExprKind::Unary {
            op: UnOp::Neg,
            expr,
        } => pure_shape(expr, sema),
        ExprKind::Binary { op, lhs, rhs } => {
            use wukong_ast::BinOp::{Add, Mul, Sub};
            matches!(op, Add | Sub | Mul) && pure_shape(lhs, sema) && pure_shape(rhs, sema)
        }
        _ => false,
    }
}

/// The region-dependent half: on top of [`pure_shape`], every free variable must be one the region
/// cannot disturb — not re-declared (a homonym would be a different binding at the use site), not
/// assigned and not `&`-taken — so the expression evaluates to the same value at a use as it did at
/// the `let`.
fn pure_index_expr(e: &Expr, facts: &RegionFacts, sema: &SemaResult) -> bool {
    match &e.kind {
        ExprKind::Int(_) => true,
        ExprKind::Path(p) => {
            if !p.is_single() {
                return false;
            }
            let sym = p.first().sym;
            if facts.declared.contains(&sym)
                || facts.assigned.contains(&sym)
                || facts.addressed.contains(&sym)
            {
                return false;
            }
            matches!(sema.types.get(&e.id), Some(Ty::Scalar(s)) if s.is_int())
        }
        ExprKind::Unary {
            op: UnOp::Neg,
            expr,
        } => pure_index_expr(expr, facts, sema),
        ExprKind::Binary { op, lhs, rhs } => {
            use wukong_ast::BinOp::{Add, Mul, Sub};
            matches!(op, Add | Sub | Mul)
                && pure_index_expr(lhs, facts, sema)
                && pure_index_expr(rhs, facts, sema)
        }
        _ => false,
    }
}

/// Checks that every occurrence of `name` in the region sits in **index position** and carries
/// exactly `init_ty`. A single occurrence that fails disqualifies the whole binding, so the `let`
/// is only deleted when every reader has been rewritten.
struct UseScan<'a> {
    name: Symbol,
    init_ty: &'a Ty,
    sema: &'a SemaResult,
    /// Demand index position. True for a `let` binding, where index position is both the payoff and
    /// the blast-radius bound; false for a module `const`, whose value is a compile-time literal that
    /// `lower_expr` already inlines at *every* position, so there is nothing to restrict.
    require_index: bool,
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
                if (self.require_index && !in_index)
                    || self.sema.types.get(&e.id) != Some(self.init_ty)
                {
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
            // The one place `in_index` turns on: an index expression of `base[i0, .., in]`.
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

// ---- substitution ------------------------------------------------------------------------------

fn subst_block(b: &mut Block, x: Symbol, init: &Expr) {
    for s in &mut b.stmts {
        subst_stmt(s, x, init);
    }
    if let Some(t) = &mut b.tail {
        subst_expr(t, x, init);
    }
}

fn subst_stmt(s: &mut Stmt, x: Symbol, init: &Expr) {
    match &mut s.kind {
        StmtKind::Let { init: i, .. } => {
            if let Some(e) = i {
                subst_expr(e, x, init);
            }
        }
        StmtKind::Assign { target, value, .. } => {
            subst_expr(target, x, init);
            subst_expr(value, x, init);
        }
        StmtKind::Expr(e) | StmtKind::Defer(e) => subst_expr(e, x, init),
        StmtKind::Return(e) | StmtKind::Break(_, e) => {
            if let Some(e) = e {
                subst_expr(e, x, init);
            }
        }
        StmtKind::Continue(_) => {}
        StmtKind::While { cond, body, .. } => {
            subst_expr(cond, x, init);
            subst_block(body, x, init);
        }
        StmtKind::For { iter, body, .. } => {
            match iter {
                ForIter::Range {
                    start, end, step, ..
                } => {
                    subst_expr(start, x, init);
                    if let Some(e) = end {
                        subst_expr(e, x, init);
                    }
                    if let Some(e) = step {
                        subst_expr(e, x, init);
                    }
                }
                ForIter::Expr(e) => subst_expr(e, x, init),
            }
            subst_block(body, x, init);
        }
    }
}

fn subst_expr(e: &mut Expr, x: Symbol, init: &Expr) {
    if let ExprKind::Path(p) = &e.kind {
        if p.is_single() && p.first().sym == x {
            *e = init.clone();
            return;
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
        | ExprKind::TupleField { base: expr, .. } => subst_expr(expr, x, init),
        ExprKind::Binary { lhs, rhs, .. } => {
            subst_expr(lhs, x, init);
            subst_expr(rhs, x, init);
        }
        ExprKind::Call { callee, args, .. } => {
            subst_expr(callee, x, init);
            for a in args {
                subst_expr(a, x, init);
            }
        }
        ExprKind::Index { base, indices } => {
            subst_expr(base, x, init);
            for i in indices {
                subst_expr(i, x, init);
            }
        }
        ExprKind::StructLit { fields, rest, .. } => {
            for f in fields {
                subst_expr(&mut f.value, x, init);
            }
            if let Some(r) = rest {
                subst_expr(r, x, init);
            }
        }
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
            for v in xs {
                subst_expr(v, x, init);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            subst_expr(value, x, init);
            subst_expr(count, x, init);
        }
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => subst_block(b, x, init),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            subst_expr(cond, x, init);
            subst_block(then_branch, x, init);
            if let Some(v) = else_branch {
                subst_expr(v, x, init);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            subst_expr(scrutinee, x, init);
            for a in arms {
                if let Some(g) = &mut a.guard {
                    subst_expr(g, x, init);
                }
                subst_expr(&mut a.body, x, init);
            }
        }
    }
}

// ---- the region fact scan ----------------------------------------------------------------------

fn scan_block(b: &Block, f: &mut RegionFacts) {
    for s in &b.stmts {
        scan_stmt(s, f);
    }
    if let Some(t) = &b.tail {
        scan_expr(t, f);
    }
}

fn scan_pattern(p: &Pattern, f: &mut RegionFacts) {
    match &p.kind {
        PatKind::Ident(s) => {
            f.declared.insert(*s);
        }
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

/// The name whose *storage* an assignment target names — the root of an `x`, `x.f`, `x[i]`, `x.0`,
/// `*x` chain. `x[i] = v` writes through `x`, so `x` is recorded and `i` is not.
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

fn scan_stmt(s: &Stmt, f: &mut RegionFacts) {
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
        StmtKind::Expr(e) | StmtKind::Defer(e) => scan_expr(e, f),
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

fn scan_expr(e: &Expr, f: &mut RegionFacts) {
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

// ---- `defer` detection -------------------------------------------------------------------------

fn block_has_defer(b: &Block) -> bool {
    let mut found = false;
    scan_defer_block(b, &mut found);
    found
}

fn scan_defer_block(b: &Block, found: &mut bool) {
    for s in &b.stmts {
        scan_defer_stmt(s, found);
    }
    if let Some(t) = &b.tail {
        scan_defer_expr(t, found);
    }
}

fn scan_defer_stmt(s: &Stmt, found: &mut bool) {
    match &s.kind {
        StmtKind::Defer(_) => *found = true,
        StmtKind::While { body, .. } | StmtKind::For { body, .. } => scan_defer_block(body, found),
        StmtKind::Let {
            init: Some(e), ..
        }
        | StmtKind::Expr(e)
        | StmtKind::Return(Some(e))
        | StmtKind::Break(_, Some(e)) => scan_defer_expr(e, found),
        StmtKind::Assign { target, value, .. } => {
            scan_defer_expr(target, found);
            scan_defer_expr(value, found);
        }
        _ => {}
    }
}

fn scan_defer_expr(e: &Expr, found: &mut bool) {
    match &e.kind {
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => scan_defer_block(b, found),
        ExprKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            scan_defer_block(then_branch, found);
            if let Some(x) = else_branch {
                scan_defer_expr(x, found);
            }
        }
        ExprKind::Match { arms, .. } => {
            for a in arms {
                scan_defer_expr(&a.body, found);
            }
        }
        _ => {}
    }
}
