//! **Hand-tiled GEMM.** Blocking a matmul nest is what a performance-minded author writes — and it
//! used to make the program two orders of magnitude slower, because every GEMM matcher requires its
//! spatial loops to start at a literal `0` (`match_two_spatial_fors`, `match_k_loop`), so the tiled
//! nest fell past every kernel into a fully scalar loop while the naive nest got `wukong_sgemm`.
//!
//! The fix is a *view*, in the same style as the other spelling-coverage modules: a perfect chain of
//! tile loops and point loops is presented to `recognize_matmul` as the untiled nest it enumerates,
//! and if the recognizer and `emit_sgemm` accept that nest, the kernel replaces the whole chain. If
//! anything declines, the statements are lowered exactly as written. Two tile spellings:
//!
//! ```text
//!   for i0 in 0..M step TI {                    let mut i0 = 0;
//!     for j0 in 0..N step TJ {                  while i0 < M {
//!       for i in i0..i0 + TI {                    let mut j0 = 0;
//!         for j in j0..j0 + TJ {                  while j0 < N {
//!           <ijk / ikj / memacc GEMM body>          for i in i0..(i0 + TI) { for j in j0..(j0 + TJ) { … } }
//!         } } } }                                   j0 = j0 + TJ;
//!                                                 }
//!                                                 i0 = i0 + TI;
//!                                               }
//! ```
//!
//! and any interleaving of tile loops, their point loops and untiled point loops (`for k in 0..K`),
//! including a tiled contraction (`for k0 … step TK { … for k in k0..k0 + TK { c[..] += … } }`).
//!
//! # Why the view is the same computation
//!
//! * **Same iteration set.** Each tile loop `0..E step T` (or its `while`) has literal `E` and `T`
//!   with `E % T == 0`, and its single point loop spans `v0..v0 + T`, so the point variable takes
//!   every value of `0..E` exactly once. Tile variables appear nowhere else (not in the GEMM body,
//!   not in another loop's bounds), so the body executed for a point tuple depends only on that
//!   tuple, and each tuple's body runs once in both nests.
//! * **Only the order differs**, and the order is what the GEMM recognizer + `emit_sgemm` already
//!   reason about: the recognized bodies write one output cell `c[i, j]` and read `a`, `b` (and that
//!   cell), and `emit_sgemm` declines unless `c` is provably disjoint from `a`, `b` and any bias — so
//!   distinct cells commute. For a single cell, the contraction index is visited in ascending order
//!   in both nests (a tile loop always encloses its own point loop), so even the accumulate-into-`C`
//!   form folds each cell in the same order before the kernel replaces it under the same contract as
//!   the untiled nest.
//! * A `while` tile counter is declared in the enclosing block; it must not be mentioned after the
//!   window, since the kernel path never assigns it its final value.

use wukong_ast::{self as ast, Block, Expr, ExprKind, ForIter, Pattern, Stmt, StmtKind};
use wukong_sema::SemaResult;
use wukong_span::{Interner, Symbol};

use crate::{
    as_int_lit, block_mentions, for_iter_mentions, recognize_matmul, single_path, FnLowerer,
};

/// One tile loop of the chain: `v0` over `0..E` in steps of `T`.
struct Tile<'a> {
    var: Symbol,
    start: &'a Expr,
    end: &'a Expr,
    width: i64,
    used: bool,
}

/// One point loop of the chain, in nest order.
struct Point<'a> {
    stmt: &'a Stmt,
    pat: &'a Pattern,
    body_id: ast::NodeId,
    /// `Some(tile index)` for `v in v0..v0 + T`; `None` for an untiled `v in 0..E` (kept verbatim).
    tile: Option<usize>,
    iter: &'a ForIter,
}

struct Chain<'a> {
    tiles: Vec<Tile<'a>>,
    points: Vec<Point<'a>>,
}

impl FnLowerer<'_> {
    /// Try the hand-tiled GEMM window at `b.stmts[at..]`: a tile `for` (one statement) or a
    /// `let mut v0 = 0; while v0 < E { …; v0 = v0 + T; }` pair (two). Returns the number of
    /// statements consumed, having lowered them, or `None` having emitted nothing.
    pub(super) fn try_detiled_gemm(&mut self, b: &Block, at: usize) -> Option<usize> {
        let consumed = match &b.stmts.get(at)?.kind {
            StmtKind::For { .. } => 1,
            StmtKind::Let { .. } => 2,
            _ => return None,
        };
        let window = b.stmts.get(at..at + consumed)?;
        let view = detile_view(window, self.sema, self.interner)?;
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
        } = &view.kind
        else {
            return None;
        };
        let nest = recognize_matmul(pat, iter, body, self.sema, self.interner)?;
        self.emit_nonempty_reduction_stmts(nest.k, window, |me| {
            me.emit_sgemm(&nest, me.parallel_fn)
        })
        .then_some(consumed)
    }
}

/// The untiled `for` nest a tiled chain enumerates, or `None` when `window` is not a tiled chain
/// (no tile loop at all included — an untiled nest is the ordinary cascade's business). Pure.
fn detile_view(window: &[Stmt], sema: &SemaResult, interner: &Interner) -> Option<Stmt> {
    let mut chain = Chain {
        tiles: Vec::new(),
        points: Vec::new(),
    };
    let innermost = walk(window, &mut chain, sema, interner)?;
    if chain.tiles.is_empty() || chain.tiles.iter().any(|t| !t.used) {
        return None;
    }
    // A tile variable may appear only in its own point loop's bounds.
    for t in &chain.tiles {
        if block_mentions(&innermost.stmts, innermost.tail.as_deref(), t.var)
            || chain
                .points
                .iter()
                .any(|p| p.tile.is_none() && for_iter_mentions(p.iter, t.var))
        {
            return None;
        }
    }
    // Point variables are distinct from each other and from every tile variable.
    let mut seen: Vec<Symbol> = chain.tiles.iter().map(|t| t.var).collect();
    for p in &chain.points {
        let ast::PatKind::Ident(v) = p.pat.kind else {
            return None;
        };
        if seen.contains(&v) {
            return None;
        }
        seen.push(v);
    }
    // Rebuild the nest from the inside out.
    let mut inner = innermost.clone();
    for p in chain.points.iter().rev() {
        let iter = match p.tile {
            Some(t) => ForIter::Range {
                start: chain.tiles[t].start.clone(),
                end: Some(chain.tiles[t].end.clone()),
                inclusive: false,
                step: None,
            },
            None => p.iter.clone(),
        };
        let stmt = Stmt {
            id: p.stmt.id,
            attrs: Vec::new(),
            kind: StmtKind::For {
                label: None,
                pat: p.pat.clone(),
                iter,
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
    inner.stmts.pop()
}

/// Walk one level of the chain — `stmts` is the whole statement list at that level — recording
/// tiles and points, and return the innermost point loop's body.
fn walk<'a>(
    stmts: &'a [Stmt],
    chain: &mut Chain<'a>,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<&'a Block> {
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
            if !s.attrs.is_empty() || body.tail.is_some() {
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
                // A tile loop `for v0 in 0..E step T`: it must have a chain below it.
                let width = as_int_lit(step, interner)?;
                push_tile(chain, v, start, end, width, sema, interner)?;
                return walk(&body.stmts, chain, sema, interner);
            }
            let tile = match tile_of_point(chain, start, end, sema, interner) {
                Some(t) => {
                    chain.tiles[t].used = true;
                    Some(t)
                }
                None if as_int_lit(start, interner) == Some(0) => None,
                None => return None,
            };
            chain.points.push(Point {
                stmt: s,
                pat,
                body_id: body.id,
                tile,
                iter,
            });
            if continues_chain(&body.stmts) {
                walk(&body.stmts, chain, sema, interner)
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
            if !let_s.attrs.is_empty() || !while_s.attrs.is_empty() || body.tail.is_some() {
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
            let width = unit_increment(last, *v0, interner)?;
            push_tile(chain, *v0, init, rhs, width, sema, interner)?;
            // The counter is assigned by the increment and nowhere else in the body.
            if inner.iter().any(|s| assigns(s, *v0)) {
                return None;
            }
            walk(inner, chain, sema, interner)
        }
        _ => None,
    }
}

/// Record a tile loop over `start..end` in steps of `width`, where both bounds are integer literals
/// of one type, `start` is 0 and the width divides the extent exactly.
fn push_tile<'a>(
    chain: &mut Chain<'a>,
    var: Symbol,
    start: &'a Expr,
    end: &'a Expr,
    width: i64,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<()> {
    let extent = as_int_lit(end, interner)?;
    if as_int_lit(start, interner)? != 0
        || width <= 0
        || extent <= 0
        || extent % width != 0
        || sema.types.get(&start.id).is_none()
        || sema.types.get(&start.id) != sema.types.get(&end.id)
        || chain.tiles.iter().any(|t| t.var == var)
    {
        return None;
    }
    chain.tiles.push(Tile {
        var,
        start,
        end,
        width,
        used: false,
    });
    Some(())
}

/// The index of the unused tile whose point loop `start..end` is — `v0..v0 + T` or `v0..T + v0`
/// with `T` the tile's width and `v0` typed as the tile's bounds.
fn tile_of_point(
    chain: &Chain<'_>,
    start: &Expr,
    end: &Expr,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<usize> {
    let v0 = single_path(start)?;
    let t = chain.tiles.iter().position(|t| t.var == v0 && !t.used)?;
    let ExprKind::Binary {
        op: ast::BinOp::Add,
        lhs,
        rhs,
    } = &end.kind
    else {
        return None;
    };
    let width = if single_path(lhs) == Some(v0) {
        as_int_lit(rhs, interner)?
    } else if single_path(rhs) == Some(v0) {
        as_int_lit(lhs, interner)?
    } else {
        return None;
    };
    let tile = &chain.tiles[t];
    (width == tile.width && sema.types.get(&start.id) == sema.types.get(&tile.start.id))
        .then_some(t)
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

/// `v0 = v0 + T`, `v0 = T + v0` or `v0 += T` with `T` a positive integer literal → `T`.
fn unit_increment(s: &Stmt, v0: Symbol, interner: &Interner) -> Option<i64> {
    let StmtKind::Assign { target, op, value } = &s.kind else {
        return None;
    };
    if single_path(target) != Some(v0) {
        return None;
    }
    match op {
        ast::AssignOp::Add => as_int_lit(value, interner),
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
                as_int_lit(rhs, interner)
            } else if single_path(rhs) == Some(v0) {
                as_int_lit(lhs, interner)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Does `s` (at any depth) assign the name `v`, or re-declare it?
fn assigns(s: &Stmt, v: Symbol) -> bool {
    match &s.kind {
        StmtKind::Assign { target, .. } => single_path(target) == Some(v),
        StmtKind::Let { pat, .. } => crate::pat_binds(pat, v),
        StmtKind::For { pat, body, .. } => {
            crate::pat_binds(pat, v) || body.stmts.iter().any(|s| assigns(s, v))
        }
        StmtKind::While { body, .. } => body.stmts.iter().any(|s| assigns(s, v)),
        // Any other statement form makes the window a non-chain anyway; be conservative.
        _ => true,
    }
}
