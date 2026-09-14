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
//! So a function that **publishes a second name** for storage it can already reach needs a second
//! test. Both halves of it come out of the same walk ([`BodyScan`]).
//!
//! The republishing spellings are:
//!
//! * `let v: []T/&T/*T = <place>;` — the view binding above, recorded in `views`. It is used twice
//!   over: an ARGUMENT spelled `v` must resolve back to `<place>` before the call-site test can see
//!   the aliasing, and an OPERAND spelled `v` must resolve back to it before the body test can.
//! * an aggregate literal field initialised from such a place and bound by a `let`:
//!   `let l = Layer { wq: swq };` publishes the operand `l.wq`, which `FnLowerer::field_base` names
//!   with the synthetic symbol `"<local>.<field>"`. `field_views` maps `(l, wq)` onto what `swq`
//!   views.
//! * an assignment whose value is such a place (`p = &x;`), an array or tuple literal built from
//!   one, a struct functional update (`Layer { ..other }`), or a struct literal no `let` binds to a
//!   name. None of those can be attributed to a NAME flow-insensitively, so each taints the
//!   function outright — the coarse answer, kept for the shapes that have no precise one.
//! * a name given a view meaning TWICE (`let s: []f32 = c; … let s: []f32 = d;`). Both maps are
//!   keyed by NAME and last-writer-wins, so the second binding silently re-points every spelling of
//!   `s` — including the ones textually above it — at the wrong buffer. That taints too, and it
//!   poisons the name for the call-site half as well ([`BodyScan::note_view_name`]); a name-keyed
//!   flow-insensitive map has no sound precise answer here.
//!
//! The other half is the operand vocabulary. A recognizer names a buffer operand in exactly one
//! syntactic position — the BASE OF AN INDEX — and resolves it with one of two functions:
//! `single_path`, which accepts a bare local, or `buffer_base`, which accepts that plus the
//! one-level projection `l.wq` (named with the synthetic `"<local>.<field>"` symbol
//! `intern_field_bases` mints). Every helper that hands an emitter an operand symbol bottoms out in
//! one of those two over an `ExprKind::Index` base — `as_index1`, `match_operand_row_col[_off]`,
//! `affine_index`, the assignment targets — and a nest operand is by construction a buffer the
//! nest indexes. [`Operand`] is that vocabulary and [`BodyScan::operands`] records BOTH shapes at
//! every index base, so it is a superset of what any recognizer can name; the function is tainted
//! only when two DISTINCT operands resolve onto overlapping storage.
//!
//! Resolving an operand is a BOUNDED FIXPOINT over the two maps ([`resolve_operand`]), not one pass
//! over each, because a view can sit on top of a field view: `let v: []f32 = l.wq;` needs the whole
//! chain `v -> l.wq -> swq -> wq`. Stopping short of it answers `l` — a root no other operand
//! shares — and that is exactly how a GEMM came to be dispatched over an output aliasing its own A
//! operand, on both backends, at every `-O` level, after the operand test above was already in.
//!
//! That last distinction is not refinement for its own sake. The first cut of this test tainted on
//! the mere PRESENCE of a view, and the cost landed squarely on the shape the field vocabulary was
//! added for. `crates/wukong_xbench/wk/general/block_d_struct.wk`, the benchmark peer that holds a
//! transformer block's weights in a `struct Layer` the way a real model does, packs ten parameters
//! into ten `[]f32` views and then reads them only as `l.wq[…]`; presence-tainting declined all
//! NINE of its GEMM dispatches while the bare-parameter twin beside it kept them. That is not a
//! wrong answer, but it quietly turns a benchmark peer into a measurement of this pass, and nothing
//! in the ten-part gate compiles that directory. Under the operand test each of those ten names has
//! exactly ONE operand spelling, nothing overlaps, and the nine GEMMs stand.
//!
//! Every other aggregate `let` deep-copies (`emit_copy`, "the local owns its storage (value
//! semantics)") and publishes nothing, which is why `let b: [f32; 4] = a;` leaves the function
//! alone. The taint stays one bit for the whole function rather than a per-operand set, for the
//! same reason the call-site fact is (below).
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
//! # Known limits
//!
//! Two kinds, and a reader must not take the list as uniformly safe. The ones marked CONSERVATIVE
//! are places the pass cannot tell and therefore taints: they cost a dispatch and nothing else. The
//! ones marked HOLE are spellings that DO reach one buffer twice and are still answered "disjoint".
//! Every hole listed is one the compiler already had before this module existed — it dispatched
//! every aliased spelling — so none of them is made reachable here; but this module does not close
//! them, and that is a weaker claim than "pre-existing holes this pass narrows rather than opens",
//! which is what this heading used to assert.
//!
//! * CONSERVATIVE. A buffer reaching a call through a `match`/`if`/`loop` **expression** value is
//!   `Unknown`, so it aliases everything — sound, occasionally pessimistic.
//! * HOLE. A binding introduced by a `match` arm pattern over an aggregate is treated as its own
//!   storage. Kernel operands do not currently arrive that way.
//! * CONSERVATIVE. Two *different constant* indices into one array (`f(m[0], m[1])`) are treated as
//!   may-alias, because this pass does not evaluate index expressions.
//! * HOLE. A borrowing binding initialised from a CALL is not treated as a view. `let v: []f32 =
//!   alloc_f32(n)` is fresh storage and `let v: []f32 = pass_through(a)` is `a`, and this pass
//!   cannot tell them apart; tainting on the second would taint every `alloc_*` program in the
//!   corpus, so both are left as their own storage and the slice-returning user function is a
//!   stated hole. At a CALL SITE it is still conservative — [`origin_of`] classifies such a call
//!   `Unknown`, which aliases everything — so it is only the in-body operand test that lets it
//!   through.
//! * HOLE. An aggregate that **carries** a view into a callee is not seen through: with
//!   `let l = Layer { w: sa };` (where `sa` views `a`), the call `f(l, a)` classifies `l` and `a` as
//!   two roots, so `f` stays alias free even though `l.w` and `a` are one buffer. Inside the CALLER
//!   `field_views` does see it — `l.w` and `a` resolve onto one place, and the caller's own
//!   recognizers decline — but the fact that crosses the call edge is one bit, which cannot carry
//!   a per-field map. A real hole, stated rather than argued away.
//! * HOLE. A struct local built by a CALL rather than by a literal (`let l: Layer = make(a);`) has
//!   no `field_views` entry, so `l.w` is treated as its own storage even if `make` stored one of its
//!   arguments there. Same root cause as the bullet above.
//! * HOLE. Likewise a function returning an aggregate that contains a view: [`origin_of`] calls such
//!   a return `Fresh` (the sret buffer really is fresh), and only a *borrowing* projection out of it
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
    /// literal, or a call whose return type is an AGGREGATE (it comes back through the sret ABI
    /// into a buffer the caller just allocated). Cannot alias anything.
    ///
    /// This does NOT include `alloc_f32(n)`: its return type is `[]f32`, which borrows, so
    /// [`origin_of`] classifies it `Unknown`. What keeps the whole `alloc_*` slice corpus
    /// dispatching is that `Unknown` is not a PLACE, so it publishes no second name — not that it
    /// is fresh, which this pass has no way to prove.
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
    /// Two DIFFERENT kernel-operand names in this body can reach one buffer, or the body
    /// republishes storage in a shape this pass cannot attribute to a name — see the module doc's
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

            let mut scan = scan_body(body, sema);
            // Two answers come out of that one walk, both resolved AFTER it rather than during it.
            // The analysis is flow-insensitive either way, so a view declared textually below its
            // use resolves identically — and deferring is what lets ONE traversal do the job the
            // first cut of this module needed two for.
            //
            // First: can two DIFFERENT operand names in this body reach one buffer?
            let internal_view = scan.opaque_republish || body_names_one_buffer_twice(&scan);
            // Second: an argument spelled with a view's name has to resolve back onto the storage
            // it views before the call-site test can see the aliasing.
            for site in &mut scan.sites {
                for a in &mut site.args {
                    *a = resolve_views(
                        std::mem::replace(a, Origin::Unknown),
                        &scan.views,
                        &scan.rebound,
                    );
                }
            }

            info.insert(
                f.name.sym,
                FnInfo {
                    buffer_params,
                    sites: scan.sites,
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

/// A name a recognizer can use as a kernel buffer **operand**. The lowerer turns a nest
/// subexpression into an operand symbol through two functions: `single_path`, which accepts a bare
/// local `x`, and `buffer_base`, which accepts that plus the one-level projection `l.wq`, named
/// with the synthetic `"<local>.<field>"` symbol `intern_field_bases` mints. `buffer_base` is the
/// wider of the two, so its two shapes are the whole vocabulary, and anything else both decline.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Operand {
    Name(Symbol),
    Field(Symbol, Symbol),
}

/// The [`Operand`] an index base names, in `buffer_base`'s two shapes (`single_path`, else a
/// `Field` whose own base is a `single_path`). A SUPERSET of what `buffer_base` accepts, never a
/// subset: it declines wherever `buffer_base` declines, and it additionally accepts one case
/// `buffer_base` rejects — `buffer_base` resolves the synthetic `"<local>.<field>"` name with
/// `interner.get`, so an `l.f` whose symbol was never minted comes back `None` there and `Some`
/// here. Wide is the sound direction: an operand recorded here that no recognizer can name costs at
/// most a decline, while one missed here is a kernel emitted over storage it does not own.
fn operand_of(e: &Expr) -> Option<Operand> {
    match &e.kind {
        ExprKind::Path(p) if p.is_single() => Some(Operand::Name(p.first().sym)),
        ExprKind::Field { base, name } => match &base.kind {
            ExprKind::Path(p) if p.is_single() => Some(Operand::Field(p.first().sym, name.sym)),
            _ => None,
        },
        _ => None,
    }
}

/// Everything one walk of one function body produces.
#[derive(Default)]
struct BodyScan {
    /// `let v: []T/&T/*T = <place>;` — `v` reaches whatever `<place>` reaches.
    views: HashMap<Symbol, Place>,
    /// `let l = Layer { wq: <place>, … };` — the operand `l.wq` reaches `<place>`.
    field_views: HashMap<(Symbol, Symbol), Place>,
    /// Every name that has been given a view meaning, whether through `views` or as the local of a
    /// `field_views` entry. Only used to detect the SECOND one — see [`BodyScan::note_view_name`].
    view_names: HashSet<Symbol>,
    /// Names that acquired a view meaning twice. Both maps are keyed by NAME and last-writer-wins,
    /// so nothing spelled with one of these resolves soundly: [`resolve_views`] answers `Unknown`
    /// for it (which aliases everything) rather than answering the surviving binding.
    rebound: HashSet<Symbol>,
    /// The base of every `Index` in the body, in [`Operand`]'s vocabulary: exactly the names a
    /// recognizer can hand [`crate::FnLowerer::all_provably_disjoint`], and therefore the only
    /// names two of which can be one nest's operands.
    operands: HashSet<Operand>,
    /// The body republishes storage in a shape this pass cannot attribute to a NAME. Taints the
    /// function outright — see the module doc.
    opaque_republish: bool,
    /// Calls made from this body, arguments unresolved.
    sites: Vec<Site>,
}

impl BodyScan {
    /// Record that `name` now has a view meaning, and taint the function if it already had one.
    ///
    /// `views` and `field_views` are keyed by NAME and this pass is flow-insensitive, so a second
    /// binding of a name overwrites the first in `views` (and merges into the same local's entries
    /// in `field_views`); every operand or argument spelled with that name — including the ones
    /// textually ABOVE the rebinding — then resolves onto the wrong buffer.
    /// `fn mm(b, mut c, d) { let s: []f32 = c; <nest over s and b writing c>; let s: []f32 = d; }`
    /// resolved the nest's `s` to `d`, found two distinct roots, and dispatched a GEMM writing its
    /// own A operand. There is no sound precise answer for a name-keyed map here, so the coarse one
    /// is the honest answer: the function is tainted, and `rebound` additionally poisons the name so
    /// a CALL ARGUMENT spelled with it cannot resolve into a false disjointness either
    /// (`let s: []f32 = c; mm(s, c); let s: []f32 = d;` is the same defect one frame up).
    fn note_view_name(&mut self, name: Symbol) {
        if !self.view_names.insert(name) {
            self.opaque_republish = true;
            self.rebound.insert(name);
        }
    }
}

/// One walk of one function body. See [`BodyScan`].
fn scan_body(body: &Block, sema: &SemaResult) -> BodyScan {
    let mut out = BodyScan::default();
    // Struct literals that a `let` binds to a name are recorded precisely in `field_views`; the
    // generic expression arm below must not ALSO taint them, or the precision is undone. Safe
    // because `walk_stmt` delivers the statement before it descends into the initializer, so the id
    // is recorded before the literal is reached.
    let mut attributed: HashSet<ast::NodeId> = HashSet::default();
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
                if annotated_view || inferred_view {
                    // The map serves both tests: a CALL SITE spelled with the view's name resolves
                    // back onto the storage it views, and so does an OPERAND spelled with it.
                    if let Origin::At(p) = origin_of(e, sema) {
                        out.note_view_name(*name);
                        out.views.insert(*name, p);
                    }
                    return;
                }
                // An aggregate `let` deep-copies every field EXCEPT the borrowing ones: a
                // `[]T`/`*T`/`&T` field keeps the source's base pointer, so `l.wq` becomes a second
                // name for it. A functional update (`..other`) publishes fields this pass cannot
                // enumerate, and a synthesized literal has no id to attribute, so both fall through
                // to the blanket taint in the expression arm.
                let ExprKind::StructLit { fields, rest, .. } = &e.kind else {
                    return;
                };
                if rest.is_some() || e.id == ast::NodeId::DUMMY {
                    return;
                }
                attributed.insert(e.id);
                let mut publishes = false;
                for fi in fields {
                    if sema.types.get(&fi.value.id).is_some_and(ty_is_borrowing) {
                        if let Origin::At(p) = origin_of(&fi.value, sema) {
                            out.field_views.insert((*name, fi.name.sym), p);
                            publishes = true;
                        }
                    }
                }
                // Once per `let`, not once per field: one literal names one local, and registering
                // it per field would report the second field of `Layer { wq: swq, wk: swk }` as a
                // rebinding and cost `block_d_struct.wk` every kernel it has.
                if publishes {
                    out.note_view_name(*name);
                }
            }
            // `p = &x;` / `s = a;` — a borrowing place stored into a binding that already exists.
            // WHICH name now reaches that storage is a flow-sensitive question this pass does not
            // answer, so it taints outright rather than guessing.
            StmtKind::Assign { value, .. } => {
                out.opaque_republish |= borrowing_place(value, sema);
            }
            _ => {}
        },
        Node::Expr(e) => match &e.kind {
            // The base of every index is a candidate kernel operand — see [`Operand`].
            ExprKind::Index { base, .. } => {
                if let Some(o) = operand_of(base) {
                    out.operands.insert(o);
                }
            }
            // An aggregate literal that takes a borrowing place as a field value hands that storage
            // a second name. The `let`-bound struct literal is recorded precisely above; every
            // other spelling names its fields in a way no `buffer_base` can reproduce, so it taints.
            ExprKind::StructLit { fields, rest, .. } => {
                if attributed.contains(&e.id) {
                    return;
                }
                out.opaque_republish |= fields.iter().any(|fi| borrowing_place(&fi.value, sema))
                    || rest.as_deref().is_some_and(|r| borrowing_place(r, sema));
            }
            ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
                out.opaque_republish |= xs.iter().any(|x| borrowing_place(x, sema));
            }
            ExprKind::ArrayRepeat { value, .. } => {
                out.opaque_republish |= borrowing_place(value, sema);
            }
            ExprKind::Call { callee, args, .. } => {
                let ExprKind::Path(p) = &callee.kind else {
                    return;
                };
                if !p.is_single() {
                    return;
                }
                out.sites.push(Site {
                    callee: p.first().sym,
                    args: args.iter().map(|a| origin_of(a, sema)).collect(),
                });
            }
            _ => {}
        },
    });
    out
}

/// Can two DIFFERENT kernel-operand names in this body reach the same storage?
///
/// Each operand is resolved through both view maps and then compared with the same
/// [`places_may_overlap`] the call-site test uses. Two operands that land on one place — `sa` and
/// `a` after `let mut sa: []f32 = a;`, or `l.wq` and `wq` after `Layer { wq: swq }` — are the
/// aliased call in disguise: no call site aliases anything, and `main`, having no parameters, is
/// trivially alias free under the call-site fact alone.
///
/// Only names that actually appear as an operand count. A view whose source is never spelled again
/// — the ten `let swq: []f32 = wq;` bindings a struct-held weight block opens with — leaves one
/// name reaching that buffer and nothing to confuse; tainting on the view's mere presence is what
/// silently cost `block_d_struct.wk` its nine GEMMs.
///
/// Grouping by resolved ROOT keeps this linear in practice: different roots cannot overlap, and a
/// function's operands almost always have different roots, so the quadratic term runs only inside a
/// group.
fn body_names_one_buffer_twice(scan: &BodyScan) -> bool {
    let mut by_root: HashMap<Symbol, Vec<Place>> = HashMap::default();
    for op in &scan.operands {
        let Some(p) = resolve_operand(*op, scan) else {
            return true; // an unresolvable chain is "cannot tell", which is "may overlap"
        };
        let group = by_root.entry(p.base).or_default();
        if group.iter().any(|q| places_may_overlap(q, &p)) {
            return true;
        }
        group.push(p);
    }
    false
}

/// Walk one operand spelling down to the storage it really names, alternating the two view maps to
/// a **bounded fixpoint** (capped like [`resolve_views`]; an exhausted cap is `None`, which the
/// caller reads as "may overlap").
///
/// One pass over each map is not enough, and the shortfall is not theoretical: a view OF a struct
/// field, `let v: []f32 = l.w;`, puts a `views` entry on top of a `field_views` entry. Resolving
/// `v` through `views` alone yields `l.w`, whose root `l` is a local no other operand shares, so
/// the chain `v -> l.w -> sa -> c` stopped short and a recognized GEMM was dispatched over an
/// output that aliased its own A operand — the same wrong answer on both backends at every `-O`
/// level, which is the one thing this module exists to prevent. Each round resolves views to their
/// own fixpoint and then takes at most one `field_views` step, so a chain with k `field_views`
/// links costs k + 1 rounds.
fn resolve_operand(op: Operand, scan: &BodyScan) -> Option<Place> {
    let mut p = match op {
        Operand::Name(s) => Place {
            base: s,
            proj: Vec::new(),
        },
        Operand::Field(b, f) => Place {
            base: b,
            proj: vec![Proj::Field(f)],
        },
    };
    for _ in 0..8 {
        let Origin::At(q) = resolve_views(Origin::At(p), &scan.views, &scan.rebound) else {
            return None;
        };
        p = q;
        // Is what we now hold a field this body republished? `field_views` is keyed by the
        // one-level `(local, field)` pair `buffer_base` can name, so only a LEADING `Field` step
        // can match it; anything below that step rides along unchanged.
        let Some(Proj::Field(f)) = p.proj.first() else {
            return Some(p);
        };
        let Some(src) = scan.field_views.get(&(p.base, *f)) else {
            return Some(p);
        };
        let mut proj = src.proj.clone();
        proj.extend(p.proj.iter().skip(1).cloned());
        p = Place {
            base: src.base,
            proj,
        };
    }
    None
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
///
/// A root in `rebound` has no resolution this pass can trust — the map kept one of its two bindings
/// and which one is a flow-sensitive question — so it answers `Unknown`, which aliases everything.
/// That is what keeps the CALL-SITE half honest: `let s: []f32 = c; mm(s, c); let s: []f32 = d;`
/// would otherwise resolve the argument `s` to `d`, see two distinct roots, and leave `mm` alias
/// free while both of its arguments are one array.
fn resolve_views(o: Origin, views: &HashMap<Symbol, Place>, rebound: &HashSet<Symbol>) -> Origin {
    let Origin::At(mut p) = o else {
        return o;
    };
    for _ in 0..8 {
        if rebound.contains(&p.base) {
            return Origin::Unknown;
        }
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
