//! **Out-of-place** softmax / log-softmax: the source row `x` is kept and the result goes to `o`.
//!
//! ```text
//!   let mut m = x[0];
//!   for i in 0..N { m = fmax(m, x[i]); }
//!   let mut s = 0.0;
//!   for i in 0..N { o[i] = exp(x[i] - m); s = s + o[i]; }
//!   for i in 0..N { o[i] = o[i] / s; }                          // => wukong_norm_f32(x, o, 1, N, ..)
//! ```
//!
//! This is what must be written whenever the scores are still needed afterwards (attention with a
//! saved score matrix, a functional forward, logits kept for the backward), and the fused-norm
//! windows recognized only the in-place spelling, so it lowered to loops. The kernel already takes
//! distinct source and destination pointers (`try_fuse_copy_norm` relies on it), so the work here
//! is entirely in proving that the window IS `o = softmax(x)`.
//!
//! # The argument
//!
//! Call the user's window `W`. Build `W'` from it by re-sourcing, to `o`, exactly these reads of `x`:
//! every read in the statements BEFORE the first one that mentions `o`, and, inside that first
//! statement (a `for v in 0..N` loop), the reads `x[v]` in the body statements up to and including
//! the first store `o[v] = …`. Then:
//!
//! 1. `W` behaves as `for c in 0..N { o[c] = x[c]; } W'` (on disjoint `x`, `o`): before the first
//!    statement touching `o`, `o` still equals that copy of `x`; inside the first `o`-writing loop,
//!    iteration `v` reads `o[v]` before it is written, and no earlier iteration wrote it (each
//!    writes only its own `o[v]`), so `o[v]` still holds `x[v]`. Every read NOT re-sourced is
//!    identical in `W` and `W'`, and `W'` never writes `x`.
//! 2. `W'` is checked to be one of the five in-place fused-norm windows on `o`
//!    (`match_inplace_norm_window`) — so `copy; W'` is exactly the pair `try_fuse_copy_norm` already
//!    folds into `norm(x, o, …)`, and for the same reason.
//! 3. Any read of `x` the re-sourcing did not license is still spelled `x` in `W'`, and `W'`'s window
//!    must mention `x` nowhere, so it declines instead (`s = s + x[i]` after the exp store, an `x`
//!    read at another index, `o` read before it is written, …).
//!
//! `x`, `o` and any affine factor must be provably disjoint, and none may be re-bound in the window
//! (the matchers compare names, not bindings).

use wukong_ast::{self as ast, Block, Expr, ExprKind, ForIter, Stmt, StmtKind};
use wukong_span::Symbol;

use crate::{expr_mentions, stmt_mentions, window_rebinds, FnLowerer};

/// The longest window the in-place matchers consume (the split softmax with a reciprocal pair).
const MAX_WINDOW: usize = 7;

impl FnLowerer<'_> {
    /// Try the out-of-place softmax / log-softmax window at `b.stmts[at..]`. Returns the number of
    /// statements consumed, having emitted the kernel, or `None` having emitted nothing.
    pub(super) fn try_fuse_out_of_place_norm(&mut self, b: &Block, at: usize) -> Option<usize> {
        let stmts = &b.stmts[at..];
        if stmts.len() < 5 {
            return None;
        }
        // Cheap structural pre-filter: the window opens with the max pass over `x`.
        let (m, _) = Self::let_init(&stmts[0])?;
        let (v1, _, body1) = self.as_range0_for(&stmts[1])?;
        let x = self.match_max_reduce_body(body1, v1, m, None)?;
        // The destination: the first array a later loop of the window stores into. `x` itself is the
        // in-place window, which the cascade has already had its chance at.
        let limit = stmts.len().min(MAX_WINDOW);
        let o = stmts[2..limit].iter().find_map(|s| match &s.kind {
            StmtKind::For { body, .. } => first_store_base(body),
            _ => None,
        })?;
        if o == x {
            return None;
        }
        let view = resourced_view(b, at, x, o, limit)?;
        let (n, op, n_expr, eps, gamma, beta) = self.match_inplace_norm_window(&view, 0, o)?;
        // (3): every read of `x` in the matched window was re-sourced by (1)'s rule.
        if view.stmts[..n].iter().any(|s| stmt_mentions(s, x)) {
            return None;
        }
        let window = &b.stmts[at..at + n];
        if window_rebinds(window, x) || window_rebinds(window, o) {
            return None;
        }
        let mut buffers = vec![x, o];
        buffers.extend(gamma);
        buffers.extend(beta);
        if !self.all_provably_disjoint(&buffers) {
            return None;
        }
        self.emit_norm(x, o, None, &n_expr, eps, op, gamma, beta)
            .then_some(n)
    }
}

/// The base of the first `base[..] = …` store in `body`'s own statements, if any.
fn first_store_base(body: &Block) -> Option<Symbol> {
    body.stmts.iter().find_map(|s| match &s.kind {
        StmtKind::Assign { target, .. } => match &target.kind {
            ExprKind::Index { base, .. } => crate::single_path(base),
            _ => None,
        },
        _ => None,
    })
}

/// `W'`: `b.stmts[at..]` (plus the tail, which the matchers consult for escaping scalars) with the
/// licensed reads of `x` re-sourced to `o` — see the module docs. `None` when the window does not
/// have the required shape (the first statement mentioning `o` is not a loop whose first `o`
/// statement is a plain store `o[v] = …` that does not read `o`).
fn resourced_view(b: &Block, at: usize, x: Symbol, o: Symbol, limit: usize) -> Option<Block> {
    let mut stmts: Vec<Stmt> = b.stmts[at..].to_vec();
    let f = stmts[..limit].iter().position(|s| stmt_mentions(s, o))?;
    for s in &mut stmts[..f] {
        resource_stmt(s, x, o, None);
    }
    let StmtKind::For { pat, body, .. } = &mut stmts[f].kind else {
        return None;
    };
    let ast::PatKind::Ident(v) = pat.kind else {
        return None;
    };
    let first = body.stmts.iter().position(|s| stmt_mentions(s, o))?;
    let StmtKind::Assign {
        target,
        op: ast::AssignOp::Assign,
        value,
    } = &body.stmts[first].kind
    else {
        return None;
    };
    if !is_index_by(target, o, v) || expr_mentions(value, o) {
        return None;
    }
    for s in &mut body.stmts[..=first] {
        resource_stmt(s, x, o, Some(v));
    }
    Some(Block {
        id: b.id,
        stmts,
        tail: b.tail.clone(),
        span: b.span,
    })
}

/// `e` is exactly `arr[v]`.
fn is_index_by(e: &Expr, arr: Symbol, v: Symbol) -> bool {
    match &e.kind {
        ExprKind::Index { base, indices } => {
            crate::single_path(base) == Some(arr)
                && indices.len() == 1
                && crate::single_path(&indices[0]) == Some(v)
        }
        _ => false,
    }
}

/// Re-source the reads of `x` in one statement to `o`. With `only = Some(v)`, only the reads
/// spelled exactly `x[v]`; with `None`, every read. An assignment TARGET is never re-sourced (its
/// index expression is a read and is), so a write to `x` keeps its name and fails the caller's
/// no-`x` check.
fn resource_stmt(s: &mut Stmt, x: Symbol, o: Symbol, only: Option<Symbol>) {
    match &mut s.kind {
        StmtKind::Let { init, .. } => {
            if let Some(e) = init {
                resource_expr(e, x, o, only);
            }
        }
        StmtKind::Assign { target, value, .. } => {
            if let ExprKind::Index { indices, .. } = &mut target.kind {
                for i in indices {
                    resource_expr(i, x, o, only);
                }
            }
            resource_expr(value, x, o, only);
        }
        StmtKind::For { iter, body, .. } => {
            if let ForIter::Range {
                start, end, step, ..
            } = iter
            {
                resource_expr(start, x, o, only);
                if let Some(e) = end {
                    resource_expr(e, x, o, only);
                }
                if let Some(e) = step {
                    resource_expr(e, x, o, only);
                }
            }
            for st in &mut body.stmts {
                resource_stmt(st, x, o, only);
            }
        }
        // Anything else keeps its spelling; if it reads `x`, the window's no-`x` check declines.
        _ => {}
    }
}

/// Re-source reads of `x` inside an expression (see [`resource_stmt`]). Only the arithmetic, call,
/// cast and index forms a norm window is made of are entered; a read hidden anywhere else keeps its
/// name, which can only make the caller decline.
fn resource_expr(e: &mut Expr, x: Symbol, o: Symbol, only: Option<Symbol>) {
    match &mut e.kind {
        ExprKind::Index { base, indices } => {
            let licensed = match only {
                None => true,
                Some(v) => indices.len() == 1 && crate::single_path(&indices[0]) == Some(v),
            };
            if licensed {
                if let ExprKind::Path(p) = &mut base.kind {
                    if p.is_single() && p.segments[0].sym == x {
                        p.segments[0].sym = o;
                    }
                }
            }
            for i in indices {
                resource_expr(i, x, o, only);
            }
        }
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => {
            resource_expr(expr, x, o, only)
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            resource_expr(lhs, x, o, only);
            resource_expr(rhs, x, o, only);
        }
        ExprKind::Call { args, .. } => {
            for a in args {
                resource_expr(a, x, o, only);
            }
        }
        _ => {}
    }
}
