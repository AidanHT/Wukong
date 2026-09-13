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
//! **The traversal is one exhaustive walk** ([`walk_block`] / [`walk_stmt`] / [`walk_expr`], feeding
//! a single [`Node`] callback), and both `match`es over `StmtKind` and `ExprKind` are wildcard-free
//! so a new AST node is a compile error here rather than a silent hole. That is not decoration: the
//! first cut of this module had two overlapping walkers, one of which handed a block's `tail` only
//! to the *statement* path, so `fn go(a, b, mut c) { mm(a, b, c) }` — the same call minus its
//! semicolon — was invisible to the site collector and `mm` stayed "alias free". A `match` arm
//! guard was invisible for the same reason. An analysis with a hole a formatting choice walks
//! through is worse than no analysis, because it is trusted.
//!
//! # Two names for one buffer INSIDE a function
//!
//! The call-site fact above is only half of what `FnLowerer::all_provably_disjoint` needs; the other
//! half is that two distinct operand *symbols* in one function body are distinct storage. That is
//! not free either, because a `[]T` / `*T` / `&T` binding is a **view**: `lower_slice_from_source`
//! stores the source's base pointer into the fat pointer ("no element copy"), so after
//! `let mut sa: []f32 = a;` the names `sa` and `a` are one buffer, and a recognized nest written
//! over both would dispatch a disjointness-requiring kernel over overlapping storage — with no call
//! site aliasing anything, and in `main`, which has no parameters and is therefore *trivially*
//! alias-free under the call-site fact alone.
//!
//! So a function that **publishes a second name** for storage it can already reach is not alias
//! free either. Three spellings do that, and [`FnInfo::internal_view`] records all three:
//!
//! * `let v: []T/&T/*T = <place>;` — the view binding above (also recorded in `views`, because an
//!   argument spelled with the view's name has to be resolved back to what it views before the
//!   call-site test can see the aliasing).
//! * an aggregate literal field initialised from such a place — `Layer { w: sa }` gives `l.w`, which
//!   `FnLowerer::field_base` records as its own kernel-operand symbol `"<local>.<field>"`.
//! * an assignment whose value is such a place — `p = &x;`.
//!
//! Every other aggregate `let` deep-copies (`emit_copy`, "the local owns its storage (value
//! semantics)") and publishes nothing, which is why `let b: [f32; 4] = a;` and
//! `let mut x: []f32 = alloc_f32(n);` leave the function alias free — the latter because
//! `alloc_f32(n)` is `Fresh`, not a place. The taint is one bit for the whole function rather than
//! a per-symbol set, for the same reason the call-site fact is (below).
//!
//! # Precision, and why the coarseness is affordable
//!
//! The fact exposed is one bit per function ([`AliasFacts::alias_free`]), not a pair set: a function
//! that some call site can hand the same array twice — or that names one buffer twice itself — gets
//! no disjointness-requiring kernel *at all*. That is deliberate. A pair-precise fact would have to
//! survive the trip into an outlined `@parallel` body (whose recognizers see `env` captures, not
//! parameters) and into the recognizer probes (which build a `FnLowerer` with no parameter bindings
//! at all), and each of those seams is a place to get it silently wrong. One bit that rides on
//! `GemmSyms` — the struct whose own doc says it exists because it "is already threaded to every
//! recognizer and every recognizer probe" — cannot be lost on the way down.
//!
//! # Known limits (all of them pre-existing holes this pass narrows rather than opens)
//!
//! * A buffer reaching a call through a `match`/`if`/`loop` **expression** value is `Unknown`, so it
//!   aliases conservatively — sound, occasionally pessimistic.
//! * A binding introduced by a `match` arm pattern over an aggregate is treated as its own storage.
//!   Kernel operands do not currently arrive that way.
//! * Two *different constant* indices into one array (`f(m[0], m[1])`) are treated as may-alias,
//!   because this pass does not evaluate index expressions.
//! * An aggregate that **carries** a view into a callee is not seen through: with
//!   `let l = Layer { w: sa };` (where `sa` views `a`), the call `f(l, a)` classifies `l` and `a` as
//!   two roots, so `f` stays alias free even though `l.w` and `a` are one buffer. The *caller* is
//!   tainted by the aggregate-literal rule above, so its own recognizers decline; the callee's do
//!   not. Closing it needs a per-field view map rather than one bit and is not worth that today —
//!   no corpus, example or bench program writes it — but it is a real hole, stated rather than
//!   argued away.
//! * Likewise a function returning an aggregate that contains a view: [`origin_of`] calls such a
//!   return `Fresh` (the sret buffer really is fresh), and only a *borrowing* projection out of it
//!   (`make().buf`) is demoted to `Unknown` by [`project`].

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

/// A type that hands back storage its holder did not create — a `[]T` slice, a raw pointer or a
/// reference. Used two ways: a *call* whose return type is one of these can be handing back one of
/// its own arguments, and a *binding* of one of these is a view of whatever it was initialised
/// from. An aggregate (`[T; N]`, `Tensor[..]`, a struct/tuple) is neither: it returns through the
/// sret ABI into a buffer the caller just allocated, and it deep-copies on `let`.
fn ty_is_borrowing(t: &Ty) -> bool {
    matches!(t, Ty::Slice(_) | Ty::Ptr { .. } | Ty::Ref { .. })
}

/// Classify an argument expression. See [`Origin`].
fn origin_of(e: &Expr, sema: &SemaResult) -> Origin {
    match &e.kind {
        ExprKind::Path(p) if p.is_single() => Origin::At(Place {
            base: p.first().sym,
            proj: vec![],
        }),
        ExprKind::Field { base, name } => project(e, base, Proj::Field(name.sym), sema),
        ExprKind::TupleField { base, index } => project(e, base, Proj::Tuple(*index), sema),
        ExprKind::Index { base, .. } => project(e, base, Proj::Index, sema),
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
            Some(t) if ty_is_borrowing(t) => Origin::Unknown,
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

/// Extend the base's origin by one projection step. `whole` is the projected expression itself, and
/// it is needed for exactly one case: a *borrowing* projection out of a **fresh** aggregate is NOT
/// fresh. `make().buf` allocates its sret temporary here, but the `[]f32` inside it points at
/// storage the callee already had, so propagating `Fresh` through the field would classify it as
/// "storage nobody else holds" — the one answer that can never be walked back.
fn project(whole: &Expr, base: &Expr, step: Proj, sema: &SemaResult) -> Origin {
    match origin_of(base, sema) {
        Origin::At(mut p) => {
            p.proj.push(step);
            Origin::At(p)
        }
        Origin::Fresh if sema.types.get(&whole.id).is_some_and(ty_is_borrowing) => Origin::Unknown,
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
    /// The body publishes a second name for storage it can already reach — see the module doc's
    /// "Two names for one buffer INSIDE a function". Kept out of the fixpoint deliberately: it says
    /// nothing about whether this function's own *parameters* can be one array, which is the only
    /// thing that forwards to a callee.
    internal_view: bool,
}

/// The whole-program answer. See the module doc.
pub(crate) struct AliasFacts {
    /// Functions for which some call site can bind two DISTINCT buffer parameters to overlapping
    /// storage. Absence is the common case, so the set is normally empty.
    aliasing: HashSet<Symbol>,
    /// Functions whose own body names one buffer twice. A separate concern from `aliasing`, ORed
    /// into [`AliasFacts::alias_free`].
    internal_views: HashSet<Symbol>,
}

impl AliasFacts {
    /// `true` when no two distinct buffer parameters of `f` can name overlapping storage at any call
    /// site in the program, AND `f`'s own body does not give one buffer two names — i.e. a
    /// recognizer inside `f` may rely on two distinct operand symbols being distinct storage.
    /// Conservative: `false` also covers "could not tell".
    pub(crate) fn alias_free(&self, f: Symbol) -> bool {
        !self.aliasing.contains(&f) && !self.internal_views.contains(&f)
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

            let (views, internal_view, mut sites) = scan_body(body, sema);
            // Resolve each argument through the view map AFTER the walk rather than during it. The
            // analysis is flow-insensitive either way, so a view declared textually below a call
            // resolves identically — and doing it here is what lets ONE traversal do the job the
            // first cut of this module needed two for.
            for site in &mut sites {
                for a in &mut site.args {
                    *a = resolve_views(std::mem::replace(a, Origin::Unknown), &views);
                }
            }

            info.insert(
                f.name.sym,
                FnInfo {
                    buffer_params,
                    sites,
                    internal_view,
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
        let internal_views = info
            .iter()
            .filter(|(_, fi)| fi.internal_view)
            .map(|(f, _)| *f)
            .collect();
        AliasFacts {
            aliasing,
            internal_views,
        }
    }
}

/// One walk of one function body: the view bindings it introduces, whether it publishes a second
/// name for storage it can already reach, and every call it makes (arguments unresolved).
fn scan_body(body: &Block, sema: &SemaResult) -> (HashMap<Symbol, Place>, bool, Vec<Site>) {
    let mut views: HashMap<Symbol, Place> = HashMap::default();
    let mut internal_view = false;
    let mut sites = Vec::new();
    walk_block(body, &mut |n| match n {
        Node::Stmt(s) => match &s.kind {
            StmtKind::Let {
                pat,
                ty,
                init: Some(e),
                ..
            } => {
                let ast::PatKind::Ident(name) = &pat.kind else {
                    return;
                };
                // A `let` binds a view when the BINDING borrows: written down (`let s: []f32 = a;`)
                // or inferred from the initializer.
                let annotated_view = ty.as_ref().is_some_and(|t| {
                    matches!(
                        &t.kind,
                        ast::TypeKind::Slice(_)
                            | ast::TypeKind::Pointer { .. }
                            | ast::TypeKind::Ref { .. }
                    )
                });
                let inferred_view = sema.types.get(&e.id).is_some_and(ty_is_borrowing);
                if !(annotated_view || inferred_view) {
                    return;
                }
                if let Origin::At(p) = origin_of(e, sema) {
                    // Both halves, always together: the map is what lets a CALL SITE spelled with
                    // the view's name be resolved back onto the storage it views, and the bit is
                    // what stops a recognizer in THIS function from treating the two names as two
                    // buffers.
                    views.insert(*name, p);
                    internal_view = true;
                }
            }
            // `p = &x;` / `s = a;` — a borrowing place stored into a binding that already exists.
            // Not recorded in `views` (that map is keyed by the name a `let` introduces, and
            // resolving an assignment flow-insensitively would be a guess), but it does publish a
            // second name, so it taints.
            StmtKind::Assign { value, .. } => {
                internal_view |= borrowing_place(value, sema);
            }
            _ => {}
        },
        Node::Expr(e) => match &e.kind {
            // An aggregate literal that takes a borrowing place as a field value hands that storage
            // a second name: `Layer { w: sa }` makes `l.w`, which `FnLowerer::field_base` records
            // as its own kernel-operand symbol.
            ExprKind::StructLit { fields, rest, .. } => {
                internal_view |= fields.iter().any(|fi| borrowing_place(&fi.value, sema))
                    || rest.as_deref().is_some_and(|r| borrowing_place(r, sema));
            }
            ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
                internal_view |= xs.iter().any(|x| borrowing_place(x, sema));
            }
            ExprKind::ArrayRepeat { value, .. } => {
                internal_view |= borrowing_place(value, sema);
            }
            ExprKind::Call { callee, args, .. } => {
                let ExprKind::Path(p) = &callee.kind else {
                    return;
                };
                if !p.is_single() {
                    return;
                }
                sites.push(Site {
                    callee: p.first().sym,
                    args: args.iter().map(|a| origin_of(a, sema)).collect(),
                });
            }
            _ => {}
        },
    });
    (views, internal_view, sites)
}

/// Is `e` a *place* whose type borrows — the thing that, stored anywhere, gives one buffer a second
/// name? `alloc_f32(n)` has a borrowing type but is not a place (it is `Fresh`), so it is not one,
/// which is why the whole `alloc_*` corpus keeps its kernels.
fn borrowing_place(e: &Expr, sema: &SemaResult) -> bool {
    sema.types.get(&e.id).is_some_and(ty_is_borrowing)
        && matches!(origin_of(e, sema), Origin::At(_))
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

// ---- AST walk -------------------------------------------------------------------------------
//
// ONE traversal, both node kinds, no wildcard arms over `StmtKind`/`ExprKind`. The predecessor of
// this walk had two mutually recursive halves — a statement walker and an expression walker — and a
// block's `tail` reached only the first of them, so a call written without its trailing semicolon
// was invisible to the site collector. Everything that carries a value out of a block (a `tail`, a
// non-block `match` arm body, a `match` arm GUARD) is an expression position, and every expression
// position has to reach the same callback a statement one does, or the fact this module computes is
// a fact about formatting.

/// One node of the walk. `Stmt` and `Expr` are both delivered because the two consumers want
/// different halves: view bindings are statements, call sites are expressions.
enum Node<'a> {
    Stmt(&'a Stmt),
    Expr(&'a Expr),
}

fn walk_block(b: &Block, f: &mut impl FnMut(Node<'_>)) {
    for s in &b.stmts {
        walk_stmt(s, f);
    }
    if let Some(t) = &b.tail {
        walk_expr(t, f);
    }
}

fn walk_stmt(s: &Stmt, f: &mut impl FnMut(Node<'_>)) {
    f(Node::Stmt(s));
    match &s.kind {
        StmtKind::Let { init: opt, .. } | StmtKind::Return(opt) | StmtKind::Break(_, opt) => {
            if let Some(e) = opt {
                walk_expr(e, f);
            }
        }
        StmtKind::Assign { target, value, .. } => {
            walk_expr(target, f);
            walk_expr(value, f);
        }
        StmtKind::Expr(e) | StmtKind::Defer(e) => walk_expr(e, f),
        StmtKind::Continue(_) => {}
        StmtKind::While { cond, body, .. } => {
            walk_expr(cond, f);
            walk_block(body, f);
        }
        StmtKind::For { iter, body, .. } => {
            match iter {
                ForIter::Range {
                    start, end, step, ..
                } => {
                    walk_expr(start, f);
                    if let Some(e) = end {
                        walk_expr(e, f);
                    }
                    if let Some(e) = step {
                        walk_expr(e, f);
                    }
                }
                ForIter::Expr(e) => walk_expr(e, f),
            }
            walk_block(body, f);
        }
    }
}

fn walk_expr(e: &Expr, f: &mut impl FnMut(Node<'_>)) {
    f(Node::Expr(e));
    match &e.kind {
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => walk_expr(expr, f),
        ExprKind::Binary { lhs, rhs, .. } => {
            walk_expr(lhs, f);
            walk_expr(rhs, f);
        }
        ExprKind::Call { callee, args, .. } => {
            walk_expr(callee, f);
            for a in args {
                walk_expr(a, f);
            }
        }
        ExprKind::Index { base, indices } => {
            walk_expr(base, f);
            for i in indices {
                walk_expr(i, f);
            }
        }
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => walk_expr(base, f),
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
            for x in xs {
                walk_expr(x, f);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            walk_expr(value, f);
            walk_expr(count, f);
        }
        ExprKind::StructLit { fields, rest, .. } => {
            for fi in fields {
                walk_expr(&fi.value, f);
            }
            if let Some(r) = rest {
                walk_expr(r, f);
            }
        }
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => walk_block(b, f),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            walk_expr(cond, f);
            walk_block(then_branch, f);
            if let Some(x) = else_branch {
                walk_expr(x, f);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            walk_expr(scrutinee, f);
            for a in arms {
                if let Some(g) = &a.guard {
                    walk_expr(g, f);
                }
                walk_expr(&a.body, f);
            }
        }
        // Leaves. Spelled out rather than `_` so a new `ExprKind` is a compile error here.
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::Path(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_) => {}
    }
}
