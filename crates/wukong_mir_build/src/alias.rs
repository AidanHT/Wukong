//! Whole-program **buffer-parameter aliasing** facts.
//!
//! # The question this answers
//!
//! `fn mm(a: [f32; 16], b: [f32; 16], mut c: [f32; 16])` holding an `i/j/k` matmul nest is
//! recognized and lowered to one `wukong_sgemm(a, b, c, …)` call. That kernel writes `c` in a
//! *tiled* order while it is still reading `a` and `b`, so it equals the nest only when `c` is
//! disjoint from `a` and from `b`. Nothing in the language forbids `mm(x, y, x)`, and until this
//! module existed nothing in the compiler looked: the call was emitted with the alias visible in
//! the MIR (`call wukong_sgemm(v0, v9, v0, …)`) and both backends printed the same wrong answer —
//! `10 10 10 10 …` where the nest says `10 19 36 69 …`. Every differential, invariance and
//! determinism gate in the tree agrees on it, because both backends call the same kernel.
//!
//! `ty_is_buffer_param` already records the underlying fact: two DISTINCT parameters of array /
//! tensor / slice / pointer / reference / tuple / named type can name the same storage, because
//! all of them are passed by base pointer. What was missing is the other half — *whether any call
//! site in this program actually does that*.
//!
//! # The decision, and why
//!
//! Recognition is an optimization. The repo's own gate
//! `no_recognize_keeps_the_program_correct_on_both_backends` states the contract: "Recognition off
//! must change only *how* the program is compiled." So the language's meaning of an aliased
//! recognized call is **the meaning of the source nest** — not the kernel's, not a copy-in/copy-out
//! semantics invented by the interpreter's marshalling, and not a compile error (nothing about
//! `mm(x, y, x)` is ill-typed, and a diagnostic would reject programs that are perfectly well
//! defined as written).
//!
//! Making the three implementations agree on that meaning therefore has exactly one shape: a
//! recognizer whose kernel needs disjointness must **decline** unless disjointness is proven, and
//! let the scalar nest lower. That is what LLVM's `LoopIdiomRecognize` does when alias analysis
//! cannot separate the store range from the load range (it forms `memmove` where it can and bails
//! otherwise), and what Polly/LoopVersioning do when they cannot: they emit a runtime alias check
//! and keep the original loop on the may-alias arm. Both arms of that precedent exist here —
//! `FnLowerer::emit_nonoverlapping_call` is the runtime-check form, used by the embedding gather
//! and scatter; this module supplies the compile-time form for everything else.
//!
//! # What the analysis is
//!
//! One conservative, whole-program, flow-insensitive pass, run once in `lower_program_with` before
//! any function is lowered. Wukong is a whole-program compiler: imports are merged before sema,
//! there are no globals (`ItemKind` has `Const` but no `static`), and a call resolves only through
//! a single-segment `Path` that names a `DefKind::Fn` — so there are no indirect calls and every
//! caller of every function is visible. That makes "no call site aliases two of `f`'s buffer
//! params" a checkable whole-program property rather than an assumption.
//!
//! For each call `f(e0, …, en)` the pass classifies each argument's [`Origin`] and asks whether two
//! of them can be the same storage. A **place** (`x`, `x[i]`, `s.w`, `&x`) carries its root symbol
//! and its projection chain, so `f(l.wq, l.wk)` — the ordinary transformer-weights spelling — stays
//! provably disjoint while `f(x, x)` and `f(m, m[0])` do not. A **fresh** value (a struct/array/tuple
//! literal, or a call returning an aggregate by the sret ABI, or `alloc_f32(n)`) is storage nobody
//! else holds. Anything else is **unknown** and aliases conservatively.
//!
//! The result propagates transitively to a fixpoint: if `g`'s params `i`/`j` may alias and `g` calls
//! `f(p_i, p_j)`, then `f`'s corresponding params may alias too. Recursion terminates because the
//! per-function fact is a single monotone bit.
//!
//! # Precision, and why the coarseness is affordable
//!
//! The fact exposed is one bit per function ([`AliasFacts::alias_free`]), not a pair set: a function
//! that some call site can hand the same array twice gets no disjointness-requiring kernel *at all*.
//! That is deliberate. A pair-precise fact would have to survive the trip into an outlined
//! `@parallel` body (whose recognizers see `env` captures, not parameters) and into the recognizer
//! probes (which build a `FnLowerer` with no parameter bindings at all), and each of those seams is
//! a place to get it silently wrong. One bit that rides on `GemmSyms` — the struct whose own doc
//! says it exists because it "is already threaded to every recognizer and every recognizer probe" —
//! cannot be lost on the way down.
//!
//! The cost is bounded to functions that a call site really does alias, which in the 371-program
//! corpus is one (`tests/run/embedding_alias_parallel.wk`). Every other function keeps every kernel
//! it had, byte-for-byte.
//!
//! # Known limits (all of them pre-existing holes this pass narrows rather than opens)
//!
//! * A buffer reaching a call through a `match`/`if`/`loop` **expression** value is `Unknown`, so it
//!   aliases conservatively — sound, occasionally pessimistic.
//! * A binding introduced by a `match` arm pattern over an aggregate is treated as its own storage.
//!   Kernel operands do not currently arrive that way.
//! * Two *different constant* indices into one array (`f(m[0], m[1])`) are treated as may-alias,
//!   because this pass does not evaluate index expressions.

use wukong_ast::{self as ast, Block, Expr, ExprKind, ForIter, Module, Stmt, StmtKind};
use wukong_sema::{DefKind, SemaResult};
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Symbol};
use wukong_types::Ty;

use crate::ty_is_buffer_param;

/// One step of a place projection. `Index` is deliberately index-value-blind: two indices into one
/// array are assumed to meet.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Proj {
    Field(Symbol),
    Tuple(u32),
    Index,
}

/// A syntactic place: a root symbol plus the projections applied to it. `l.blocks[i].wq` is
/// `{ base: l, proj: [Field(blocks), Index, Field(wq)] }`.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Place {
    base: Symbol,
    proj: Vec<Proj>,
}

/// Where a call argument's storage comes from.
#[derive(Clone, Debug)]
enum Origin {
    /// A place in the caller's frame (or reachable from one of its parameters).
    At(Place),
    /// Storage created by the expression itself and held by nobody else: an array/struct/tuple
    /// literal, an sret aggregate return, `alloc_f32(n)`. Cannot alias anything.
    Fresh,
    /// Not classified. Aliases everything, conservatively.
    Unknown,
}

/// Can these two places name overlapping storage? Different roots cannot (a local is its own
/// allocation, and two *parameters* that could be the same array are related through the call-site
/// pass below, not here). Same root: walk the projections in step and look for a pair that proves
/// separation — two different named fields, or two different tuple indices. `Index` never proves
/// anything, and a prefix relation (`l` vs `l.wq`) is containment, which is overlap.
fn places_may_overlap(a: &Place, b: &Place) -> bool {
    if a.base != b.base {
        return false;
    }
    for (pa, pb) in a.proj.iter().zip(b.proj.iter()) {
        match (pa, pb) {
            (Proj::Field(x), Proj::Field(y)) if x != y => return false,
            (Proj::Tuple(x), Proj::Tuple(y)) if x != y => return false,
            _ => {}
        }
    }
    true
}

fn origins_may_overlap(a: &Origin, b: &Origin) -> bool {
    match (a, b) {
        (Origin::Fresh, _) | (_, Origin::Fresh) => false,
        (Origin::Unknown, _) | (_, Origin::Unknown) => true,
        (Origin::At(x), Origin::At(y)) => places_may_overlap(x, y),
    }
}

/// A call whose return value can hand back storage the callee did not create — a `[]T` slice, a raw
/// pointer or a reference. An aggregate (`[T; N]`, `Tensor[..]`, a struct/tuple) returns through the
/// sret ABI into a buffer the *caller* just allocated, so that one really is fresh.
fn ret_is_borrowing(t: &Ty) -> bool {
    matches!(t, Ty::Slice(_) | Ty::Ptr { .. } | Ty::Ref { .. })
}

/// Classify an argument expression. See [`Origin`].
fn origin_of(e: &Expr, sema: &SemaResult) -> Origin {
    match &e.kind {
        ExprKind::Path(p) if p.is_single() => Origin::At(Place {
            base: p.first().sym,
            proj: vec![],
        }),
        ExprKind::Field { base, name } => project(base, Proj::Field(name.sym), sema),
        ExprKind::TupleField { base, index } => project(base, Proj::Tuple(*index), sema),
        ExprKind::Index { base, .. } => project(base, Proj::Index, sema),
        ExprKind::Unary {
            op: ast::UnOp::Ref | ast::UnOp::RefMut | ast::UnOp::Deref,
            expr,
        } => origin_of(expr, sema),
        ExprKind::Cast { expr, .. } => origin_of(expr, sema),
        // Literal aggregates build a temporary the caller owns outright.
        ExprKind::ArrayLit(_)
        | ExprKind::ArrayRepeat { .. }
        | ExprKind::StructLit { .. }
        | ExprKind::TupleLit(_) => Origin::Fresh,
        // A call's value is fresh unless the callee's return type is a *borrowing* one, in which
        // case it can be one of the callee's own arguments coming back out.
        ExprKind::Call { .. } => match sema.types.get(&e.id) {
            Some(t) if ret_is_borrowing(t) => Origin::Unknown,
            Some(_) => Origin::Fresh,
            None => Origin::Unknown,
        },
        // Scalars cannot alias; everything else is unclassified.
        _ => match sema.types.get(&e.id) {
            Some(t) if !ty_is_buffer_param(t) => Origin::Fresh,
            _ => Origin::Unknown,
        },
    }
}

fn project(base: &Expr, step: Proj, sema: &SemaResult) -> Origin {
    match origin_of(base, sema) {
        Origin::At(mut p) => {
            p.proj.push(step);
            Origin::At(p)
        }
        other => other,
    }
}

/// One resolved call site: which function, and where each argument's storage came from.
struct Site {
    callee: Symbol,
    args: Vec<Origin>,
}

/// Everything the fixpoint needs about one function.
#[derive(Default)]
struct FnInfo {
    /// Indices of the parameters that are passed by base pointer, and their AST names.
    buffer_params: Vec<(usize, Symbol)>,
    /// Calls made from this function's body.
    sites: Vec<Site>,
}

/// The whole-program answer. See the module doc.
pub(crate) struct AliasFacts {
    /// Functions for which some call site can bind two DISTINCT buffer parameters to overlapping
    /// storage. Absence is the common case, so the set is normally empty.
    aliasing: HashSet<Symbol>,
}

impl AliasFacts {
    /// `true` when no two distinct buffer parameters of `f` can name overlapping storage at any call
    /// site in the program — i.e. a recognizer inside `f` may rely on two distinct operand symbols
    /// being distinct storage. Conservative: `false` also covers "could not tell".
    pub(crate) fn alias_free(&self, f: Symbol) -> bool {
        !self.aliasing.contains(&f)
    }

    /// Run the analysis over a whole (import-merged, normalized) module.
    pub(crate) fn analyze(module: &Module, sema: &SemaResult) -> Self {
        let mut info: HashMap<Symbol, FnInfo> = HashMap::default();

        for item in &module.items {
            let ast::ItemKind::Fn(f) = &item.kind else {
                continue;
            };
            let Some(body) = &f.body else {
                continue;
            };
            let param_tys: Vec<Ty> = match sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
                Some(DefKind::Fn(sig)) => sig.params.clone(),
                // No resolved signature: fall back to "every parameter could be a buffer", which
                // only ever makes the answer more conservative.
                _ => f
                    .params
                    .iter()
                    .map(|_| Ty::Ptr {
                        mutable: true,
                        pointee: Box::new(Ty::Unknown),
                    })
                    .collect(),
            };
            let buffer_params: Vec<(usize, Symbol)> = f
                .params
                .iter()
                .enumerate()
                .filter(|(i, _)| param_tys.get(*i).is_none_or(ty_is_buffer_param))
                .map(|(i, p)| (i, p.name.sym))
                .collect();

            // A `let` that binds a slice / pointer / reference is a *view* of its initializer's
            // storage (`lower_slice_from_source` stores the source's base pointer into the fat
            // pointer — "no element copy"), so an argument spelled with the view's name has to be
            // resolved back to what it views. Every other aggregate `let` deep-copies
            // (`emit_copy`, "the local owns its storage (value semantics)") and needs no entry.
            let mut views: HashMap<Symbol, Place> = HashMap::default();
            collect_views(body, sema, &mut views);

            let mut sites = Vec::new();
            collect_sites(body, sema, &views, &mut sites);

            info.insert(
                f.name.sym,
                FnInfo {
                    buffer_params,
                    sites,
                },
            );
        }

        // Fixpoint. Seed: a call site whose two argument origins can overlap makes the callee
        // aliased. Propagate: once a caller is aliased, two of ITS parameters can be one array, so
        // a call that forwards them makes the callee aliased too.
        let mut aliasing: HashSet<Symbol> = HashSet::default();
        loop {
            let mut changed = false;
            for (caller, fi) in &info {
                let caller_aliased = aliasing.contains(caller);
                for site in &fi.sites {
                    let Some(callee) = info.get(&site.callee) else {
                        continue;
                    };
                    if aliasing.contains(&site.callee) {
                        continue;
                    }
                    if site_aliases(site, callee, fi, caller_aliased) {
                        aliasing.insert(site.callee);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        AliasFacts { aliasing }
    }
}

/// Does this call site bind two distinct buffer parameters of the callee to overlapping storage?
fn site_aliases(site: &Site, callee: &FnInfo, caller: &FnInfo, caller_aliased: bool) -> bool {
    for (n, &(i, _)) in callee.buffer_params.iter().enumerate() {
        for &(j, _) in &callee.buffer_params[n + 1..] {
            let (Some(ai), Some(aj)) = (site.args.get(i), site.args.get(j)) else {
                continue;
            };
            if origins_may_overlap(ai, aj) {
                return true;
            }
            // The caller's own parameters may already be one array (established on an earlier
            // round). Two arguments rooted at two of them therefore forward that aliasing.
            if caller_aliased && both_root_at_buffer_params(ai, aj, caller) {
                return true;
            }
        }
    }
    false
}

fn both_root_at_buffer_params(a: &Origin, b: &Origin, caller: &FnInfo) -> bool {
    let root = |o: &Origin| match o {
        Origin::At(p) => Some(p.base),
        _ => None,
    };
    let (Some(ra), Some(rb)) = (root(a), root(b)) else {
        return false;
    };
    let is_param = |s: Symbol| caller.buffer_params.iter().any(|(_, n)| *n == s);
    is_param(ra) && is_param(rb)
}

// ---- AST walk -------------------------------------------------------------------------------

fn collect_views(b: &Block, sema: &SemaResult, out: &mut HashMap<Symbol, Place>) {
    walk_block(b, &mut |s| {
        if let StmtKind::Let {
            pat,
            ty,
            init: Some(e),
            ..
        } = &s.kind
        {
            let ast::PatKind::Ident(name) = &pat.kind else {
                return;
            };
            let annotated_view = ty.as_ref().is_some_and(|t| {
                matches!(
                    &t.kind,
                    ast::TypeKind::Slice(_)
                        | ast::TypeKind::Pointer { .. }
                        | ast::TypeKind::Ref { .. }
                )
            });
            let inferred_view = sema.types.get(&e.id).is_some_and(ret_is_borrowing);
            if !(annotated_view || inferred_view) {
                return;
            }
            if let Origin::At(p) = origin_of(e, sema) {
                out.insert(*name, p);
            }
        }
    });
}

fn collect_sites(
    b: &Block,
    sema: &SemaResult,
    views: &HashMap<Symbol, Place>,
    out: &mut Vec<Site>,
) {
    walk_block_exprs(b, &mut |e| {
        let ExprKind::Call { callee, args, .. } = &e.kind else {
            return;
        };
        let ExprKind::Path(p) = &callee.kind else {
            return;
        };
        if !p.is_single() {
            return;
        }
        out.push(Site {
            callee: p.first().sym,
            args: args
                .iter()
                .map(|a| resolve_views(origin_of(a, sema), views))
                .collect(),
        });
    });
}

/// Rewrite a place rooted at a slice/pointer view back onto the storage it views, transitively
/// (bounded, so a `let a = b; let b = a;` cycle the front end would have rejected cannot spin).
fn resolve_views(o: Origin, views: &HashMap<Symbol, Place>) -> Origin {
    let Origin::At(mut p) = o else {
        return o;
    };
    for _ in 0..8 {
        let Some(src) = views.get(&p.base) else {
            return Origin::At(p);
        };
        if src.base == p.base {
            return Origin::At(p);
        }
        let mut proj = src.proj.clone();
        proj.extend(p.proj);
        p = Place {
            base: src.base,
            proj,
        };
    }
    Origin::Unknown
}

fn walk_block(b: &Block, f: &mut impl FnMut(&Stmt)) {
    for s in &b.stmts {
        f(s);
        match &s.kind {
            StmtKind::While { body, .. } | StmtKind::For { body, .. } => walk_block(body, f),
            StmtKind::Let { init: Some(e), .. }
            | StmtKind::Expr(e)
            | StmtKind::Return(Some(e))
            | StmtKind::Break(_, Some(e))
            | StmtKind::Defer(e) => walk_expr_blocks(e, f),
            StmtKind::Assign { target, value, .. } => {
                walk_expr_blocks(target, f);
                walk_expr_blocks(value, f);
            }
            _ => {}
        }
        if let StmtKind::For {
            iter: ForIter::Expr(e),
            ..
        } = &s.kind
        {
            walk_expr_blocks(e, f);
        }
    }
    if let Some(t) = &b.tail {
        walk_expr_blocks(t, f);
    }
}

/// Reach the blocks nested inside an expression (`if`, `match`, `loop`, a block expression) so a
/// `let` or a call written there is not missed.
fn walk_expr_blocks(e: &Expr, f: &mut impl FnMut(&Stmt)) {
    match &e.kind {
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => walk_block(b, f),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            walk_expr_blocks(cond, f);
            walk_block(then_branch, f);
            if let Some(e) = else_branch {
                walk_expr_blocks(e, f);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            walk_expr_blocks(scrutinee, f);
            for a in arms {
                walk_expr_blocks(&a.body, f);
            }
        }
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => walk_expr_blocks(expr, f),
        ExprKind::Binary { lhs, rhs, .. } => {
            walk_expr_blocks(lhs, f);
            walk_expr_blocks(rhs, f);
        }
        ExprKind::Call { callee, args, .. } => {
            walk_expr_blocks(callee, f);
            for a in args {
                walk_expr_blocks(a, f);
            }
        }
        ExprKind::Index { base, indices } => {
            walk_expr_blocks(base, f);
            for i in indices {
                walk_expr_blocks(i, f);
            }
        }
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => {
            walk_expr_blocks(base, f)
        }
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
            for x in xs {
                walk_expr_blocks(x, f);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            walk_expr_blocks(value, f);
            walk_expr_blocks(count, f);
        }
        ExprKind::StructLit { fields, rest, .. } => {
            for fi in fields {
                walk_expr_blocks(&fi.value, f);
            }
            if let Some(r) = rest {
                walk_expr_blocks(r, f);
            }
        }
        _ => {}
    }
}

/// Visit every expression in a block, including the ones nested in statements and in other
/// expressions.
fn walk_block_exprs(b: &Block, f: &mut impl FnMut(&Expr)) {
    walk_block(b, &mut |s| match &s.kind {
        StmtKind::Let { init: Some(e), .. }
        | StmtKind::Expr(e)
        | StmtKind::Return(Some(e))
        | StmtKind::Break(_, Some(e))
        | StmtKind::Defer(e) => walk_exprs(e, f),
        StmtKind::Assign { target, value, .. } => {
            walk_exprs(target, f);
            walk_exprs(value, f);
        }
        StmtKind::While { cond, .. } => walk_exprs(cond, f),
        StmtKind::For { iter, .. } => match iter {
            ForIter::Range {
                start, end, step, ..
            } => {
                walk_exprs(start, f);
                if let Some(e) = end {
                    walk_exprs(e, f);
                }
                if let Some(e) = step {
                    walk_exprs(e, f);
                }
            }
            ForIter::Expr(e) => walk_exprs(e, f),
        },
        _ => {}
    });
}

fn walk_exprs(e: &Expr, f: &mut impl FnMut(&Expr)) {
    f(e);
    match &e.kind {
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => walk_exprs(expr, f),
        ExprKind::Binary { lhs, rhs, .. } => {
            walk_exprs(lhs, f);
            walk_exprs(rhs, f);
        }
        ExprKind::Call { callee, args, .. } => {
            walk_exprs(callee, f);
            for a in args {
                walk_exprs(a, f);
            }
        }
        ExprKind::Index { base, indices } => {
            walk_exprs(base, f);
            for i in indices {
                walk_exprs(i, f);
            }
        }
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => walk_exprs(base, f),
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
            for x in xs {
                walk_exprs(x, f);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            walk_exprs(value, f);
            walk_exprs(count, f);
        }
        ExprKind::StructLit { fields, rest, .. } => {
            for fi in fields {
                walk_exprs(&fi.value, f);
            }
            if let Some(r) = rest {
                walk_exprs(r, f);
            }
        }
        ExprKind::If {
            cond, else_branch, ..
        } => {
            walk_exprs(cond, f);
            if let Some(e) = else_branch {
                walk_exprs(e, f);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            walk_exprs(scrutinee, f);
            for a in arms {
                walk_exprs(&a.body, f);
            }
        }
        _ => {}
    }
}
