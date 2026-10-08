//! **Hand-tiled GEMM.** Blocking a matmul nest is what a performance-minded author writes — and it
//! used to make the program two orders of magnitude slower, because every GEMM matcher requires its
//! spatial loops to start at a literal `0` (`match_two_spatial_fors`, `match_k_loop`), so the tiled
//! nest fell past every kernel into a fully scalar loop while the naive nest got `wukong_sgemm`.
//!
//! The fix is a *view*, in the same style as the other spelling-coverage modules: a chain of tile
//! loops and point loops is presented to `recognize_matmul` as the untiled nest it enumerates, and if
//! the recognizer and `emit_sgemm` accept that nest, the kernel replaces the whole chain. If anything
//! declines, the statements are lowered exactly as written.
//!
//! # The spellings
//!
//! A **tile loop** enumerates tile bases `v0`; its **point loop** runs `v` over one tile. Three tile
//! loops and two point ranges are understood, in any combination, at any depth, interleaved with
//! untiled point loops (`for k in 0..K`):
//!
//! ```text
//!   for i0 in 0..M step T { … }                       // stepped: bases 0, T, 2T, … below M
//!   let mut i0 = 0; while i0 < M { …; i0 = i0 + T; }  // the same, as a `while` (`+=` too)
//!   for it in 0..NT { let i0 = it * T; … }            // counted: bases it*T for it < NT
//!
//!   for i in i0..i0 + T { … }                         // a full tile
//!   for i in i0..min(i0 + T, M) { … }                 // a tile clamped to the extent (the tail)
//! ```
//!
//! and either point end may be named first (`let i1 = min(i0 + T, M);` … `for i in i0..i1`), which is
//! how `examples/matmul.wk` writes it. `T` is an integer literal, a function-local `const` (an
//! immutable `let` of a literal earlier in the window's block) or a top-level `const`; `M` and `NT`
//! are any side-effect-free integer expressions over names the window does not bind. A contraction
//! may be tiled too — around the memory accumulate `c[i,j] += a*b`, or around the scalar
//! accumulation seeded from `C` (`let mut s = c[i,j]; for k in k0..k1 { s += a*b; } c[i,j] = s;`),
//! which `match_matmul_ijk_cseed` below recognizes as `C += A·B`.
//!
//! # Why the view is the same computation
//!
//! * **Same iteration set.** Each point variable takes every value of `0..M` exactly once:
//!   - a clamped point range under a stepped tile over the same `M` always covers `0..M` exactly;
//!   - a full-tile range under a stepped tile covers `0..ceil(M/T)*T`, which is `0..M` iff
//!     `M % T == 0`;
//!   - a clamped range under a counted tile covers `0..min(NT*T, M)` (a base at or past `M` yields an
//!     empty range), which is `0..M` iff `NT*T >= M`;
//!   - a full-tile range under a counted tile names no extent at all and is declined.
//!   Each condition is checked at compile time when its operands are constants; otherwise it is
//!   tested at run time on entry to the window, and the window as written is the fallback.
//! * **Only the order differs**, and the order is what the GEMM recognizer + `emit_sgemm` already
//!   reason about: the recognized bodies write one output cell `c[i, j]` and read `a`, `b` (and that
//!   cell), and `emit_sgemm` declines unless `c` is provably disjoint from `a`, `b` and any bias — so
//!   distinct cells commute. For a single cell, the contraction index is visited in ascending order
//!   in both nests (a tile loop always encloses its own point loop), and a `C`-seeded accumulation
//!   round-trips the running sum through the cell between contraction tiles — an exact `f32`
//!   store/load — so it folds each cell in the same order before the kernel replaces it under the
//!   same contract as the untiled nest.
//! * Tile names (bases, counters, named ends) appear nowhere in the view, and nothing a guard or the
//!   view evaluates outside its loop mentions a name the window binds, so evaluating it once at the
//!   window's entry gives the value the source used.
//! * A `while` tile counter is declared in the enclosing block; it must not be mentioned after the
//!   window, since the kernel path never assigns it its final value.

use wukong_ast::{self as ast, Block, Expr, ExprKind, ForIter, Pattern, Stmt, StmtKind};
use wukong_mir::{BinOp, CmpOp, MirType, Op, ValueId};
use wukong_sema::{DefKind, SemaResult};
use wukong_span::{Interner, Symbol};
use wukong_types::Ty;

use crate::{
    as_int_lit, block_mentions, exprs_struct_eq, ijk_either_order, is_f32_expr, is_pure_loop_bound,
    match_k_loop, match_two_spatial_fors, pat_binds, recognize_matmul, single_path, stmt_mentions,
    FnLowerer, MatmulNest, EPI_ACT_IDENTITY,
};

/// How a tile loop enumerates its bases.
#[derive(Clone, Copy)]
enum TileKind<'w> {
    /// `for v0 in 0..E step T`, or `let mut v0 = 0; while v0 < E { …; v0 = v0 + T; }`: the bases
    /// `0, T, 2T, …` below `E`.
    Stepped { extent: &'w Expr },
    /// `for t in 0..NT { let v0 = t * T; … }`: the bases `t*T` for `t < NT`.
    Counted { count: &'w Expr },
}

/// The end of a tile's point range `v0..END`.
#[derive(Clone, Copy)]
enum PointEnd<'w> {
    /// `v0 + T`: a full tile.
    Exact,
    /// `min(v0 + T, E)`: a tile clamped to the extent `E`.
    Clamped(&'w Expr),
}

/// One tile loop of the chain.
struct Tile<'w> {
    /// The base every point of the tile starts from (`v0`).
    base: Symbol,
    /// The counted form's loop variable (`t` in `let v0 = t * T`).
    counter: Option<Symbol>,
    kind: TileKind<'w>,
    /// The tile loop's own literal `0` start, reused as the start of the view's point loop.
    zero: &'w Expr,
    width: i64,
    /// `let v1 = v0 + T;` / `let v1 = min(v0 + T, E);` in the tile's body: a name a point loop may
    /// end at, and the end it stands for.
    end_let: Option<(Symbol, PointEnd<'w>)>,
    used: bool,
}

/// One point loop of the chain, in nest order, with the range it has in the view.
struct Point<'w> {
    stmt: &'w Stmt,
    pat: &'w Pattern,
    body_id: ast::NodeId,
    iter: ForIter,
}

/// A run-time condition under which the view enumerates the source's iteration set (see the module
/// docs). Only conditions whose operands are not compile-time constants are kept.
enum Guard<'w> {
    /// `extent % width == 0`: full tiles over `0..extent step width` cover `0..extent` exactly.
    Divides { extent: &'w Expr, width: i64 },
    /// `count >= 0 && count * width >= extent`: `count` clamped tiles reach `extent`.
    Covers {
        count: &'w Expr,
        width: i64,
        extent: &'w Expr,
    },
}

#[derive(Default)]
struct Chain<'w> {
    tiles: Vec<Tile<'w>>,
    points: Vec<Point<'w>>,
    guards: Vec<Guard<'w>>,
    /// Every name the window's headers bind: tile bases, counters, named ends, point variables.
    names: Vec<Symbol>,
    /// Every expression evaluated outside the loop that spells it — tile extents and counts (the
    /// view's loop bounds, the guards) and the width spellings: none may mention a name in `names`.
    headers: Vec<&'w Expr>,
}

/// The untiled nest and the run-time conditions under which it is the window.
struct View<'w> {
    nest: Stmt,
    guards: Vec<Guard<'w>>,
}

/// The pure analysis, over the lowerer's scopes and the statements that precede the window.
struct Detiler<'l, 'a> {
    lw: &'l FnLowerer<'a>,
    prior: &'l [Stmt],
}

impl FnLowerer<'_> {
    /// Try the hand-tiled GEMM window at `b.stmts[at..]`: a tile `for` (one statement) or a
    /// `let mut v0 = 0; while v0 < E { …; v0 = v0 + T; }` pair (two). Returns the number of
    /// statements consumed, having lowered them, or `None` having emitted nothing.
    pub(super) fn try_detiled_gemm(&mut self, b: &Block, at: usize) -> Option<usize> {
        // A pending function-item `@tile` width belongs to the next loop the strip-miner can take
        // (`lower_for` honors it before any recognizer); replacing the window's loops would leave it
        // unapplied (`E0309`).
        if !self.tile_queue.is_empty() {
            return None;
        }
        let consumed = match &b.stmts.get(at)?.kind {
            StmtKind::For { .. } => 1,
            StmtKind::Let { .. } => 2,
            _ => return None,
        };
        let window = b.stmts.get(at..at + consumed)?;
        let view = Detiler {
            lw: &*self,
            prior: &b.stmts[..at],
        }
        .view(window)?;
        if consumed == 2 {
            // The `while` counter outlives the window; the kernel path never gives it its final
            // value, so nothing after the window may read it.
            let StmtKind::Let { pat, .. } = &window[0].kind else {
                return None;
            };
            let ast::PatKind::Ident(v0) = pat.kind else {
                return None;
            };
            if block_mentions(&b.stmts[at + 2..], b.tail.as_deref(), v0) {
                return None;
            }
        }
        let StmtKind::For {
            pat, iter, body, ..
        } = &view.nest.kind
        else {
            return None;
        };
        let nest = recognize_matmul(pat, iter, body, self.sema, self.interner)?;
        if view.guards.is_empty() {
            return self
                .emit_nonempty_reduction_stmts(nest.k, window, |me| {
                    me.emit_sgemm(&nest, me.parallel_fn)
                })
                .then_some(consumed);
        }
        // The kernel under the run-time cover test, the window as written otherwise.
        let ok = self.lower_detile_guards(&view.guards);
        let kern = self.builder.new_block();
        let src = self.builder.new_block();
        let join = self.builder.new_block();
        self.builder.cond_br(ok, kern, vec![], src, vec![]);
        self.builder.switch_to(kern);
        self.terminated = false;
        let done = self.emit_nonempty_reduction_stmts(nest.k, window, |me| {
            me.emit_sgemm(&nest, me.parallel_fn)
        });
        self.builder.br(if done { join } else { src }, vec![]);
        self.builder.switch_to(src);
        self.terminated = false;
        let recognize = self.gemm.recognize;
        self.gemm.recognize = false;
        for s in window {
            if self.terminated {
                break;
            }
            self.lower_stmt(s);
        }
        self.gemm.recognize = recognize;
        if !self.terminated {
            self.builder.br(join, vec![]);
        }
        self.builder.switch_to(join);
        self.terminated = false;
        Some(consumed)
    }

    /// The conjunction of `guards`, evaluated in the 64-bit domain of each extent's signedness.
    /// `guards` is never empty.
    fn lower_detile_guards(&mut self, guards: &[Guard<'_>]) -> ValueId {
        let mut all: Option<ValueId> = None;
        for g in guards {
            let ok = match *g {
                Guard::Divides { extent, width } => {
                    let signed = self.signed(extent);
                    let e = self.lower_i64(extent);
                    let w = self
                        .builder
                        .build(MirType::I64, Op::ConstInt(width as i128, MirType::I64));
                    let rem = if signed { BinOp::SRem } else { BinOp::URem };
                    let r = self.builder.build(MirType::I64, Op::Bin(rem, e, w));
                    let zero = self
                        .builder
                        .build(MirType::I64, Op::ConstInt(0, MirType::I64));
                    self.builder.build(MirType::I1, Op::Cmp(CmpOp::Eq, r, zero))
                }
                Guard::Covers {
                    count,
                    width,
                    extent,
                } => {
                    let signed = self.signed(extent);
                    let n = self.lower_i64(count);
                    let e = self.lower_i64(extent);
                    let w = self
                        .builder
                        .build(MirType::I64, Op::ConstInt(width as i128, MirType::I64));
                    let reach = self.builder.build(MirType::I64, Op::Bin(BinOp::Mul, n, w));
                    let pred = if signed { CmpOp::Sge } else { CmpOp::Uge };
                    let covers = self.builder.build(MirType::I1, Op::Cmp(pred, reach, e));
                    if signed {
                        // A negative count runs no tile; it must not pass by wrapping the product.
                        let zero = self
                            .builder
                            .build(MirType::I64, Op::ConstInt(0, MirType::I64));
                        let nonneg = self
                            .builder
                            .build(MirType::I1, Op::Cmp(CmpOp::Sge, n, zero));
                        self.builder
                            .build(MirType::I1, Op::Bin(BinOp::And, nonneg, covers))
                    } else {
                        covers
                    }
                }
            };
            all = Some(match all {
                None => ok,
                Some(a) => self.builder.build(MirType::I1, Op::Bin(BinOp::And, a, ok)),
            });
        }
        all.unwrap_or_else(|| {
            self.builder
                .build(MirType::I1, Op::ConstInt(1, MirType::I1))
        })
    }

    /// Lower an integer expression and widen it to `i64` with its own signedness.
    fn lower_i64(&mut self, e: &Expr) -> ValueId {
        let v = self.lower_expr(e);
        let ty = self.operand_mir(v, e);
        let signed = self.signed(e);
        self.coerce_to(v, &ty, &MirType::I64, signed)
    }
}

impl Detiler<'_, '_> {
    fn sema(&self) -> &SemaResult {
        self.lw.sema
    }

    fn interner(&self) -> &Interner {
        self.lw.interner
    }

    /// The untiled `for` nest a tiled chain enumerates, or `None` when `window` is not a tiled chain
    /// (no tile loop at all included — an untiled nest is the ordinary cascade's business).
    fn view<'w>(&self, window: &'w [Stmt]) -> Option<View<'w>> {
        let mut chain = Chain::default();
        let innermost = self.walk(window, &mut chain)?;
        let mut inner = innermost.clone();
        // A `C`-seeded accumulation may tile its own contraction loop.
        if let Some(kiter) = self.cseed_tiled_contraction(innermost, &mut chain) {
            if let StmtKind::For { iter, .. } = &mut inner.stmts[1].kind {
                *iter = kiter;
            }
        }
        if chain.tiles.is_empty() || chain.tiles.iter().any(|t| !t.used) {
            return None;
        }
        // Every bound name is distinct, and nothing evaluated outside its own loop depends on one.
        for (i, n) in chain.names.iter().enumerate() {
            if chain.names[i + 1..].contains(n) {
                return None;
            }
        }
        for h in &chain.headers {
            if !is_pure_loop_bound(h) || chain.names.iter().any(|n| crate::expr_mentions(h, *n)) {
                return None;
            }
        }
        // Rebuild the nest from the inside out.
        for p in chain.points.iter().rev() {
            let stmt = Stmt {
                id: p.stmt.id,
                attrs: Vec::new(),
                kind: StmtKind::For {
                    label: None,
                    pat: p.pat.clone(),
                    iter: p.iter.clone(),
                    body: inner,
                },
                span: p.stmt.span,
            };
            inner = Block {
                id: p.body_id,
                stmts: vec![stmt],
                tail: None,
                span: p.stmt.span,
            };
        }
        let nest = inner.stmts.pop()?;
        // The tile names are gone from the view: a base, counter or named end used anywhere but its
        // own point range would read a value the untiled nest never computes.
        for t in &chain.tiles {
            let names = [Some(t.base), t.counter, t.end_let.map(|(n, _)| n)];
            if names.iter().flatten().any(|n| stmt_mentions(&nest, *n)) {
                return None;
            }
        }
        Some(View {
            nest,
            guards: chain.guards,
        })
    }

    /// Walk one level of the chain — `stmts` is the whole statement list at that level — recording
    /// tiles and points, and return the innermost point loop's body.
    fn walk<'w>(&self, stmts: &'w [Stmt], chain: &mut Chain<'w>) -> Option<&'w Block> {
        match stmts {
            [s] => {
                let StmtKind::For {
                    label: None,
                    pat,
                    iter,
                    body,
                } = &s.kind
                else {
                    return None;
                };
                if !self.attrs_ok(&s.attrs) || body.tail.is_some() {
                    return None;
                }
                let ast::PatKind::Ident(v) = pat.kind else {
                    return None;
                };
                let ForIter::Range {
                    start,
                    end: Some(end),
                    inclusive: false,
                    step,
                } = iter
                else {
                    return None;
                };
                if let Some(step) = step {
                    // A stepped tile `for v0 in 0..E step T`: it must have a chain below it.
                    let width = self.width_of(step)?;
                    chain.headers.push(step);
                    self.push_tile(
                        chain,
                        Tile {
                            base: v,
                            counter: None,
                            kind: TileKind::Stepped { extent: end },
                            zero: start,
                            width,
                            end_let: None,
                            used: false,
                        },
                    )?;
                    return self.walk_tile_body(&body.stmts, chain);
                }
                if let Some((base, width, wexpr)) = self.counted_base(&body.stmts, v) {
                    // A counted tile `for t in 0..NT { let v0 = t * T; … }`.
                    chain.headers.push(wexpr);
                    self.push_tile(
                        chain,
                        Tile {
                            base,
                            counter: Some(v),
                            kind: TileKind::Counted { count: end },
                            zero: start,
                            width,
                            end_let: None,
                            used: false,
                        },
                    )?;
                    return self.walk_tile_body(&body.stmts[1..], chain);
                }
                // A point loop.
                let view_iter = self.point_range(start, end, chain)?;
                chain.names.push(v);
                chain.points.push(Point {
                    stmt: s,
                    pat,
                    body_id: body.id,
                    iter: view_iter,
                });
                if continues_chain(&body.stmts) {
                    self.walk(&body.stmts, chain)
                } else {
                    Some(body)
                }
            }
            [let_s, while_s] => {
                // `let mut v0 = 0; while v0 < E { <chain>; v0 = v0 + T; }`
                let StmtKind::Let {
                    pat:
                        Pattern {
                            kind: ast::PatKind::Ident(v0),
                            ..
                        },
                    mutable: true,
                    init: Some(init),
                    ..
                } = &let_s.kind
                else {
                    return None;
                };
                let StmtKind::While {
                    label: None,
                    cond,
                    body,
                } = &while_s.kind
                else {
                    return None;
                };
                if !let_s.attrs.is_empty() || !self.attrs_ok(&while_s.attrs) || body.tail.is_some()
                {
                    return None;
                }
                let ExprKind::Binary {
                    op: ast::BinOp::Lt,
                    lhs,
                    rhs,
                } = &cond.kind
                else {
                    return None;
                };
                if single_path(lhs) != Some(*v0) {
                    return None;
                }
                let (last, inner) = body.stmts.split_last()?;
                let (width, wexpr) = self.increment(last, *v0)?;
                // The counter is assigned by the increment and nowhere else in the body.
                if inner.iter().any(|s| assigns(s, *v0)) {
                    return None;
                }
                chain.headers.push(wexpr);
                self.push_tile(
                    chain,
                    Tile {
                        base: *v0,
                        counter: None,
                        kind: TileKind::Stepped {
                            extent: rhs.as_ref(),
                        },
                        zero: init,
                        width,
                        end_let: None,
                        used: false,
                    },
                )?;
                self.walk_tile_body(inner, chain)
            }
            _ => None,
        }
    }

    /// The statements of the newest tile's body: an optional named end (`let v1 = min(v0 + T, E);`)
    /// and then the next link of the chain.
    fn walk_tile_body<'w>(&self, stmts: &'w [Stmt], chain: &mut Chain<'w>) -> Option<&'w Block> {
        let t = chain.tiles.len().checked_sub(1)?;
        let (base, width) = (chain.tiles[t].base, chain.tiles[t].width);
        let mut rest = stmts;
        if let [first, tail @ ..] = stmts {
            if !tail.is_empty() {
                if let Some((v1, end, wexpr)) = self.end_let(first, base, width) {
                    chain.tiles[t].end_let = Some((v1, end));
                    chain.names.push(v1);
                    chain.headers.push(wexpr);
                    rest = tail;
                }
            }
        }
        self.walk(rest, chain)
    }

    /// Record a tile whose loop starts at a literal `0` and steps by a positive width.
    fn push_tile<'w>(&self, chain: &mut Chain<'w>, tile: Tile<'w>) -> Option<()> {
        if as_int_lit(tile.zero, self.interner()) != Some(0) || tile.width <= 0 {
            return None;
        }
        match tile.kind {
            TileKind::Stepped { extent } => chain.headers.push(extent),
            TileKind::Counted { count } => chain.headers.push(count),
        }
        chain.names.push(tile.base);
        if let Some(c) = tile.counter {
            chain.names.push(c);
        }
        chain.tiles.push(tile);
        Some(())
    }

    /// The range a point loop `v in start..end` has in the view: its tile's whole extent when it
    /// spans one tile (marking the tile used and recording what must hold for that to be exact), or
    /// its own range when it is an untiled `0..E`.
    fn point_range<'w>(
        &self,
        start: &'w Expr,
        end: &'w Expr,
        chain: &mut Chain<'w>,
    ) -> Option<ForIter> {
        let Some(v0) = single_path(start) else {
            return (as_int_lit(start, self.interner()) == Some(0)).then(|| ForIter::Range {
                start: start.clone(),
                end: Some(end.clone()),
                inclusive: false,
                step: None,
            });
        };
        let t = chain.tiles.iter().position(|t| t.base == v0 && !t.used)?;
        let (width, kind, end_let, zero) = (
            chain.tiles[t].width,
            chain.tiles[t].kind,
            chain.tiles[t].end_let,
            chain.tiles[t].zero,
        );
        let form = match end_let {
            Some((v1, form)) if single_path(end) == Some(v1) => form,
            _ => {
                let (form, wexpr) = self.end_form(end, v0, width)?;
                chain.headers.push(wexpr);
                form
            }
        };
        let extent = match (kind, form) {
            (TileKind::Stepped { extent }, PointEnd::Exact) => {
                match self.const_of(extent) {
                    Some(e) if e % width == 0 => {}
                    Some(_) => return None,
                    None if self.is_int(extent) => {
                        chain.guards.push(Guard::Divides { extent, width })
                    }
                    None => return None,
                }
                extent
            }
            (TileKind::Stepped { extent }, PointEnd::Clamped(clamp)) => {
                if !self.same_value(extent, clamp) {
                    return None;
                }
                extent
            }
            (TileKind::Counted { count }, PointEnd::Clamped(extent)) => {
                // Counted against clamped in one domain, so neither side is converted.
                if !self.same_int_type(count, extent) {
                    return None;
                }
                match (self.const_of(count), self.const_of(extent)) {
                    (Some(n), Some(e)) => {
                        // An overflowing product is past every `i64` extent: it covers.
                        let covers = n >= 0 && n.checked_mul(width).is_none_or(|r| r >= e);
                        if !covers {
                            return None;
                        }
                    }
                    _ => chain.guards.push(Guard::Covers {
                        count,
                        width,
                        extent,
                    }),
                }
                chain.headers.push(extent);
                extent
            }
            (TileKind::Counted { .. }, PointEnd::Exact) => return None,
        };
        chain.tiles[t].used = true;
        Some(ForIter::Range {
            start: zero.clone(),
            end: Some(extent.clone()),
            inclusive: false,
            step: None,
        })
    }

    /// `let v1 = <point end over v0>;` (immutable) → `(v1, end, width spelling)`.
    fn end_let<'w>(
        &self,
        s: &'w Stmt,
        v0: Symbol,
        width: i64,
    ) -> Option<(Symbol, PointEnd<'w>, &'w Expr)> {
        let StmtKind::Let {
            pat:
                Pattern {
                    kind: ast::PatKind::Ident(v1),
                    ..
                },
            mutable: false,
            init: Some(init),
            ..
        } = &s.kind
        else {
            return None;
        };
        if !s.attrs.is_empty() {
            return None;
        }
        let (end, wexpr) = self.end_form(init, v0, width)?;
        Some((*v1, end, wexpr))
    }

    /// `v0 + T` → a full tile; `min(v0 + T, E)` / `min(E, v0 + T)` → a clamped one (both operands
    /// of one type, so the builtin converts neither). Also returns the width spelling.
    fn end_form<'w>(
        &self,
        e: &'w Expr,
        v0: Symbol,
        width: i64,
    ) -> Option<(PointEnd<'w>, &'w Expr)> {
        if let Some(w) = self.plus_width(e, v0, width) {
            return Some((PointEnd::Exact, w));
        }
        let ExprKind::Call {
            callee,
            generic_args,
            args,
        } = &e.kind
        else {
            return None;
        };
        if !generic_args.is_empty()
            || args.len() != 2
            || !self.is_builtin_min(callee)
            || !self.same_int_type(&args[0], &args[1])
        {
            return None;
        }
        if let Some(w) = self.plus_width(&args[0], v0, width) {
            return Some((PointEnd::Clamped(&args[1]), w));
        }
        if let Some(w) = self.plus_width(&args[1], v0, width) {
            return Some((PointEnd::Clamped(&args[0]), w));
        }
        None
    }

    /// `v0 + T` or `T + v0` with `T` the width → the spelling of `T`.
    fn plus_width<'w>(&self, e: &'w Expr, v0: Symbol, width: i64) -> Option<&'w Expr> {
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs,
            rhs,
        } = &e.kind
        else {
            return None;
        };
        let w: &'w Expr = if single_path(lhs) == Some(v0) {
            rhs.as_ref()
        } else if single_path(rhs) == Some(v0) {
            lhs.as_ref()
        } else {
            return None;
        };
        (self.const_of(w) == Some(width)).then_some(w)
    }

    /// The counted tile's first statement `let v0 = t * T;` (or `T * t`) → `(v0, T, its spelling)`.
    /// A body that is nothing but that `let` is no tile.
    fn counted_base<'w>(&self, stmts: &'w [Stmt], t: Symbol) -> Option<(Symbol, i64, &'w Expr)> {
        let [first, _, ..] = stmts else {
            return None;
        };
        let StmtKind::Let {
            pat:
                Pattern {
                    kind: ast::PatKind::Ident(v0),
                    ..
                },
            mutable: false,
            init: Some(init),
            ..
        } = &first.kind
        else {
            return None;
        };
        if !first.attrs.is_empty() {
            return None;
        }
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &init.kind
        else {
            return None;
        };
        let w: &'w Expr = if single_path(lhs) == Some(t) {
            rhs.as_ref()
        } else if single_path(rhs) == Some(t) {
            lhs.as_ref()
        } else {
            return None;
        };
        Some((*v0, self.width_of(w)?, w))
    }

    /// `v0 = v0 + T`, `v0 = T + v0` or `v0 += T` → `(T, its spelling)`.
    fn increment<'w>(&self, s: &'w Stmt, v0: Symbol) -> Option<(i64, &'w Expr)> {
        let StmtKind::Assign { target, op, value } = &s.kind else {
            return None;
        };
        if single_path(target) != Some(v0) {
            return None;
        }
        let w: &'w Expr = match op {
            ast::AssignOp::Add => value,
            ast::AssignOp::Assign => {
                let ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } = &value.kind
                else {
                    return None;
                };
                if single_path(lhs) == Some(v0) {
                    rhs.as_ref()
                } else if single_path(rhs) == Some(v0) {
                    lhs.as_ref()
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        Some((self.width_of(w)?, w))
    }

    /// When the innermost body is a `C`-seeded accumulation — `let mut s = c[..]; for k in R { … };
    /// c[..] = s;` — whose contraction loop spans one tile, that loop's range in the view (its tile
    /// marked used). The running sum goes back through the cell after each contraction tile and is
    /// reloaded before the next, so the per-cell fold is the untiled one; the recognizer then checks
    /// the body is the accumulation it claims to be (`match_matmul_ijk_cseed`).
    fn cseed_tiled_contraction<'w>(
        &self,
        body: &'w Block,
        chain: &mut Chain<'w>,
    ) -> Option<ForIter> {
        let [seed, kloop, store] = body.stmts.as_slice() else {
            return None;
        };
        if body.tail.is_some() {
            return None;
        }
        let StmtKind::Let {
            pat:
                Pattern {
                    kind: ast::PatKind::Ident(s),
                    ..
                },
            init: Some(c0),
            ..
        } = &seed.kind
        else {
            return None;
        };
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &store.kind
        else {
            return None;
        };
        if single_path(value) != Some(*s)
            || !matches!(c0.kind, ExprKind::Index { .. })
            || !exprs_struct_eq(c0, target)
        {
            return None;
        }
        let StmtKind::For {
            label: None,
            pat,
            iter,
            body: kbody,
        } = &kloop.kind
        else {
            return None;
        };
        if !kloop.attrs.is_empty() || kbody.tail.is_some() {
            return None;
        }
        let ast::PatKind::Ident(k) = pat.kind else {
            return None;
        };
        let ForIter::Range {
            start,
            end: Some(end),
            inclusive: false,
            step: None,
        } = iter
        else {
            return None;
        };
        // An untiled contraction (`0..K`) needs no rewrite.
        single_path(start)?;
        let view_iter = self.point_range(start, end, chain)?;
        chain.names.push(k);
        Some(view_iter)
    }

    /// The statement-level attributes a chain loop may carry: only `@parallel`, which is read from a
    /// `fn` item and nowhere else, so it changes nothing about the loop.
    fn attrs_ok(&self, attrs: &[ast::Attr]) -> bool {
        attrs
            .iter()
            .all(|a| self.interner().resolve(a.name.sym) == "parallel")
    }

    /// A compile-time integer: a literal, a function-local `const` (the nearest binding of the name
    /// among the statements before the window is an immutable `let` of an integer literal), or an
    /// unshadowed top-level `const`.
    fn const_of(&self, e: &Expr) -> Option<i64> {
        if let Some(v) = as_int_lit(e, self.interner()) {
            return Some(v);
        }
        if let Some(sym) = single_path(e) {
            for s in self.prior.iter().rev() {
                if let StmtKind::Let {
                    pat, mutable, init, ..
                } = &s.kind
                {
                    if pat_binds(pat, sym) {
                        if *mutable {
                            return None;
                        }
                        return init.as_ref().and_then(|i| as_int_lit(i, self.interner()));
                    }
                }
            }
        }
        self.lw
            .const_runtime_u64_expr(e)
            .and_then(|v| i64::try_from(v).ok())
    }

    /// A positive compile-time tile width.
    fn width_of(&self, e: &Expr) -> Option<i64> {
        self.const_of(e).filter(|w| *w > 0)
    }

    /// The same value in the same place: the same spelling, or the same constant.
    fn same_value(&self, a: &Expr, b: &Expr) -> bool {
        exprs_struct_eq(a, b)
            || matches!((self.const_of(a), self.const_of(b)), (Some(x), Some(y)) if x == y)
    }

    /// Is `e` typed as an integer scalar?
    fn is_int(&self, e: &Expr) -> bool {
        matches!(self.sema().types.get(&e.id), Some(Ty::Scalar(s)) if s.is_int())
    }

    /// Are `a` and `b` integer scalars of one and the same type (so nothing converts either)?
    fn same_int_type(&self, a: &Expr, b: &Expr) -> bool {
        self.is_int(a) && self.sema().types.get(&a.id) == self.sema().types.get(&b.id)
    }

    /// Is `callee` the `min` builtin — not a user `fn min`, not a local of that name?
    fn is_builtin_min(&self, callee: &Expr) -> bool {
        let Some(sym) = single_path(callee) else {
            return false;
        };
        self.interner().resolve(sym) == "min"
            && !matches!(
                self.sema().defs.lookup(sym).map(|d| &d.kind),
                Some(DefKind::Fn(_))
            )
            && self.lw.lookup(sym).is_none()
    }
}

/// Is the statement list below a point loop the next link of a chain (a `for`, or a `while` tile
/// pair), rather than the innermost GEMM body?
fn continues_chain(stmts: &[Stmt]) -> bool {
    match stmts {
        [s] => matches!(&s.kind, StmtKind::For { .. }),
        [l, w] => {
            matches!(&l.kind, StmtKind::Let { mutable: true, .. })
                && matches!(&w.kind, StmtKind::While { .. })
        }
        _ => false,
    }
}

/// Does `s` (at any depth) assign the name `v`, or re-declare it?
fn assigns(s: &Stmt, v: Symbol) -> bool {
    match &s.kind {
        StmtKind::Assign { target, .. } => single_path(target) == Some(v),
        StmtKind::Let { pat, .. } => pat_binds(pat, v),
        StmtKind::For { pat, body, .. } => {
            pat_binds(pat, v) || body.stmts.iter().any(|s| assigns(s, v))
        }
        StmtKind::While { body, .. } => body.stmts.iter().any(|s| assigns(s, v)),
        // Any other statement form makes the window a non-chain anyway; be conservative.
        _ => true,
    }
}

/// The `C`-seeded `ijk` matmul — `C += A·B` accumulated in a scalar:
///
/// ```text
///   for i { for j {
///       let mut s = c[i*N + j];                              // or c[i, j]
///       for k { s = s + a[i*K + k] * b[k*N + j]; }           // or `s += …`
///       c[i*N + j] = s;
///   } }
/// ```
///
/// The cell is loaded once, the products folded into it in ascending `k`, and stored back: the same
/// computation as the memory accumulator `c[i*N+j] += a*b` (`match_matmul_ijk_memacc` without a
/// zero-init) with the running sum held in a register, so it is the same `beta = 1` GEMM. The seed
/// and the store must be the same element; everything else — the product, the strides, the loop
/// orders, a counting `while` k-loop — is checked by the same helpers as the zero-seeded
/// `match_matmul_ijk`. A contraction tiled around this form (`k in k0..k1`) is presented here
/// untiled by the view above.
pub(crate) fn match_matmul_ijk_cseed<'a>(
    pat: &Pattern,
    iter: &ForIter,
    body: &'a Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<MatmulNest<'a>> {
    let (v1, b1, v2, b2, jbody) = match_two_spatial_fors(pat, iter, body, interner)?;
    if jbody.tail.is_some() || jbody.stmts.len() < 3 {
        return None;
    }
    let StmtKind::Let {
        pat: sp,
        init: Some(seed),
        ..
    } = &jbody.stmts[0].kind
    else {
        return None;
    };
    let ast::PatKind::Ident(s_sym) = sp.kind else {
        return None;
    };
    if !matches!(seed.kind, ExprKind::Index { .. }) || !is_f32_expr(seed, sema) {
        return None;
    }
    let store = jbody.stmts.last()?;
    let (kvar, kdim, kbody) = match_k_loop(&jbody.stmts[1..jbody.stmts.len() - 1], interner)?;
    if kbody.len() != 1 {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &kbody[0].kind else {
        return None;
    };
    if single_path(target) != Some(s_sym) {
        return None;
    }
    let prod = match op {
        ast::AssignOp::Add => value,
        ast::AssignOp::Assign => {
            let ExprKind::Binary {
                op: ast::BinOp::Add,
                lhs,
                rhs,
            } = &value.kind
            else {
                return None;
            };
            if single_path(lhs) != Some(s_sym) {
                return None;
            }
            rhs.as_ref()
        }
        _ => return None,
    };
    if !is_f32_expr(prod, sema) {
        return None;
    }
    let StmtKind::Assign {
        target: ct,
        op: ast::AssignOp::Assign,
        value: cv,
    } = &store.kind
    else {
        return None;
    };
    if single_path(cv) != Some(s_sym) || !exprs_struct_eq(ct, seed) {
        return None;
    }
    ijk_either_order(
        v1,
        v2,
        kvar,
        b1,
        b2,
        kdim,
        prod,
        ct,
        1,
        None,
        None,
        EPI_ACT_IDENTITY,
        sema,
        interner,
    )
}
