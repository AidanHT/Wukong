//! The **row-outer** spelling of a column reduction: the accumulator lives in `out` and `x` is
//! streamed row-major, which is what anyone who has thought about the cache writes (and what the C
//! peers of the `col*` benchmark rows are written as):
//!
//! ```text
//!   for j in 0..N { out[j] = 0.0; }                       // or: let mut out: [f32; N] = [0.0; N];
//!   for i in 0..M { for j in 0..N { out[j] += x[i*N + j]; } }
//! ```
//!
//! `match_colsum` only knows the column-outer form (`for j { let mut s = 0.0; for i { s += x[i*N+j] };
//! out[j] = s }`), so this spelling dispatched nothing. Both forms perform, for every column `j`, the
//! identical sequence of f32 operations — seed, then fold rows in ascending `i` — and the accumulator
//! round-trips through memory as an exact f32 store/load, so the kernel the column-outer form
//! dispatches is the row-outer form's value too. Only the seed differs in spelling, which is why this
//! is a two-statement window: the fold `out[j] += …` is "add into live data", and only a seed proven
//! immediately before it makes that data the kernel's seed.
//!
//! Accepted seeds and folds, each one the kernel's exact scalar twin (`colreduce.rs` `sfold_*`):
//!
//! | seed (`for j in 0..N { … }`)   | fold (`for i in i0..M { for j in 0..N { … } }`) | kernel        |
//! |---------------------------------|--------------------------------------------------|---------------|
//! | `out[j] = 0.0` / `[0.0; N]` let | `out[j] = out[j] + x[i*N+j]` or `+=` (`i0 = 0`)   | `colsum`      |
//! | `out[j] = 0.0` / `[0.0; N]` let | `out[j] += x[i*N+j] * x[i*N+j]` (`i0 = 0`)        | `colsumsq`    |
//! | `out[j] = x[j]`                 | `out[j] = fmax(out[j], x[i*N+j])` (`i0` 0 or 1)   | `colmax`      |
//! | `out[j] = x[j]`                 | `out[j] = fmin(out[j], x[i*N+j])` (`i0` 0 or 1)   | `colmin`      |
//! | `out[j] = abs(x[j])`            | `out[j] = fmax(out[j], abs(x[i*N+j]))`            | `colmaxabs`   |
//!
//! Three spellings are declined on purpose, because the kernel would change a bit:
//! * `out[j] = out[j] + x*x` — the source contracts that to ONE fused multiply-add while the kernel
//!   squares and adds with two roundings (`sfold_sq` is `a + v*v`, "never an FMA"); the compound
//!   `out[j] += x*x` lowers as a multiply then an add, which is exactly the kernel;
//! * `x + out[j]` / `fmax(x, out[j])` — the accumulator must be the left operand, as in `sfold_*`:
//!   `fmax(a, b)` is `a > b ? a : b`, which picks a different zero or NaN when the operands swap;
//! * a fold starting at `i0 = 1` under a zero seed (that skips row 0 — a different sum).
//!
//! `emit_colsum` proves `x`/`out` disjoint (`kernel_base_ptrs_disjoint`) and declines otherwise, and
//! a possibly-empty row count keeps the two source statements as the run-time fallback (the kernel
//! writes nothing at `rows <= 0`, but the seed loop would still have run).

use wukong_ast::{self as ast, Block, Expr, ExprKind, Stmt, StmtKind};
use wukong_span::Symbol;

use crate::{
    as_dim, as_index1, as_int_lit, exprs_struct_eq, fusable_for, index_by_var, intrinsic_callee,
    is_float_zero, match_row_col_off, range_bounds, scalar_of, single_stmt, ColSumNest, Dim,
    FnLowerer, MathIntrinsic, COL_MAX, COL_MAXABS, COL_MIN, COL_SUM, COL_SUMSQ,
};

/// How the window seeds `out` — which fixes the fold kernel the window can be.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColSeed {
    /// `out[j] = 0.0` — the additive folds.
    Zero,
    /// `out[j] = x[j]` — row 0 of `x`, the max/min folds.
    FirstRow(Symbol),
    /// `out[j] = abs(x[j])` — row 0 of `|x|`, the absmax fold.
    FirstRowAbs(Symbol),
}

impl FnLowerer<'_> {
    /// Try the row-outer column-reduction window at `b.stmts[at..]` (seed, then nest). Returns the
    /// number of statements consumed, having lowered them, or `None` having emitted nothing.
    pub(super) fn try_fuse_rowouter_colreduce(&mut self, b: &Block, at: usize) -> Option<usize> {
        // A pending function-item `@tile` width belongs to the next loop the strip-miner can take;
        // replacing these loops would leave it unapplied (`E0309`).
        if !self.tile_queue.is_empty() {
            return None;
        }
        let seed_stmt = b.stmts.get(at)?;
        let nest_stmt = b.stmts.get(at + 1)?;
        let (out, n_seed, seed, seed_is_let) = self.match_col_seed(seed_stmt)?;
        let nest = self.match_rowouter_fold(nest_stmt, out, n_seed, seed)?;
        if seed_is_let {
            // The seed is the destination's own zero-filled declaration: it must be lowered (it
            // introduces `out`), and the kernel then overwrites exactly the `N` elements the nest
            // would have accumulated into.
            self.lower_stmt(seed_stmt);
            if !self.emit_nonempty_reduction_stmts(
                nest.rows,
                std::slice::from_ref(nest_stmt),
                |me| me.emit_colsum(&nest, me.parallel_fn),
            ) && !self.terminated
            {
                self.lower_stmt(nest_stmt);
            }
            return Some(2);
        }
        self.emit_nonempty_reduction_stmts(nest.rows, &b.stmts[at..at + 2], |me| {
            me.emit_colsum(&nest, me.parallel_fn)
        })
        .then_some(2)
    }

    /// The seed statement: a `for j in 0..N` fill of `out[j]` (zero, row 0 of some `x`, or its
    /// `abs`), or a `let mut out = [0.0; N];` declaration. Returns `(out, N, seed, is_let)`. Pure.
    fn match_col_seed<'s>(&self, s: &'s Stmt) -> Option<(Symbol, &'s Expr, ColSeed, bool)> {
        if let StmtKind::Let {
            pat:
                ast::Pattern {
                    kind: ast::PatKind::Ident(out),
                    ..
                },
            init:
                Some(Expr {
                    kind: ExprKind::ArrayRepeat { value, count },
                    ..
                }),
            ..
        } = &s.kind
        {
            if s.attrs.is_empty()
                && is_float_zero(value, self.interner)
                && scalar_of(value, self.sema) == Some(wukong_types::Scalar::F32)
            {
                return Some((*out, count.as_ref(), ColSeed::Zero, true));
            }
            return None;
        }
        let (pat, iter, body) = fusable_for(s)?;
        let ast::PatKind::Ident(j) = pat.kind else {
            return None;
        };
        let (start, end) = range_bounds(iter)?;
        if as_int_lit(start, self.interner)? != 0 {
            return None;
        }
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &single_stmt(body)?.kind
        else {
            return None;
        };
        let out = index_by_var(target, j)?;
        if scalar_of(target, self.sema) != Some(wukong_types::Scalar::F32) {
            return None;
        }
        if matches!(value.kind, ExprKind::Float(_)) && is_float_zero(value, self.interner) {
            return Some((out, end, ColSeed::Zero, false));
        }
        if let Some(x) = index_by_var(value, j) {
            return (x != out && scalar_of(value, self.sema) == Some(wukong_types::Scalar::F32))
                .then_some((out, end, ColSeed::FirstRow(x), false));
        }
        let ExprKind::Call { callee, args, .. } = &value.kind else {
            return None;
        };
        if args.len() != 1
            || !matches!(
                intrinsic_callee(callee, self.sema, self.interner),
                Some(MathIntrinsic::Abs)
            )
            || scalar_of(&args[0], self.sema) != Some(wukong_types::Scalar::F32)
        {
            return None;
        }
        let x = index_by_var(&args[0], j)?;
        (x != out).then_some((out, end, ColSeed::FirstRowAbs(x), false))
    }

    /// The fold nest `for i in i0..M { for j in 0..N { out[j] = fold(out[j], x[i*N + j]) } }` over the
    /// seed's `out` and `N`. Pure.
    fn match_rowouter_fold(
        &self,
        s: &Stmt,
        out: Symbol,
        n_seed: &Expr,
        seed: ColSeed,
    ) -> Option<ColSumNest> {
        let (ipat, iiter, ibody) = fusable_for(s)?;
        let ast::PatKind::Ident(ivar) = ipat.kind else {
            return None;
        };
        let (is_, ie) = range_bounds(iiter)?;
        let istart = as_int_lit(is_, self.interner)?;
        let rows = as_dim(ie, self.interner)?;
        let (jpat, jiter, jbody) = fusable_for(single_stmt(ibody)?)?;
        let ast::PatKind::Ident(jvar) = jpat.kind else {
            return None;
        };
        let (js, je) = range_bounds(jiter)?;
        if as_int_lit(js, self.interner)? != 0 || !exprs_struct_eq(je, n_seed) {
            return None;
        }
        let cols = as_dim(je, self.interner)?;
        if ivar == jvar || [ivar, jvar].contains(&out) {
            return None;
        }
        let StmtKind::Assign { target, op, value } = &single_stmt(jbody)?.kind else {
            return None;
        };
        if index_by_var(target, jvar) != Some(out)
            || scalar_of(target, self.sema) != Some(wukong_types::Scalar::F32)
        {
            return None;
        }
        let is_acc = |e: &Expr| index_by_var(e, jvar) == Some(out);
        let (colop, data): (i64, &Expr) = match seed {
            ColSeed::Zero => {
                if istart != 0 {
                    return None;
                }
                let (addend, compound) = match op {
                    ast::AssignOp::Add => (value, true),
                    ast::AssignOp::Assign => match &value.kind {
                        ExprKind::Binary {
                            op: ast::BinOp::Add,
                            lhs,
                            rhs,
                        } if is_acc(lhs.as_ref()) => (rhs.as_ref(), false),
                        _ => return None,
                    },
                    _ => return None,
                };
                match &addend.kind {
                    ExprKind::Binary {
                        op: ast::BinOp::Mul,
                        lhs,
                        rhs,
                    } if exprs_struct_eq(lhs, rhs) => {
                        // `out[j] = out[j] + d*d` is contracted to one FMA; the kernel rounds twice.
                        if !compound {
                            return None;
                        }
                        (COL_SUMSQ, lhs.as_ref())
                    }
                    _ => (COL_SUM, addend),
                }
            }
            ColSeed::FirstRow(_) | ColSeed::FirstRowAbs(_) => {
                if (istart != 0 && istart != 1) || !matches!(op, ast::AssignOp::Assign) {
                    return None;
                }
                let ExprKind::Call { callee, args, .. } = &value.kind else {
                    return None;
                };
                if args.len() != 2 || !is_acc(&args[0]) {
                    return None;
                }
                let fold = intrinsic_callee(callee, self.sema, self.interner);
                match (seed, fold) {
                    (ColSeed::FirstRow(_), Some(MathIntrinsic::Fmax)) => (COL_MAX, &args[1]),
                    (ColSeed::FirstRow(_), Some(MathIntrinsic::Fmin)) => (COL_MIN, &args[1]),
                    (ColSeed::FirstRowAbs(_), Some(MathIntrinsic::Fmax)) => {
                        let ExprKind::Call {
                            callee: ac,
                            args: aargs,
                            ..
                        } = &args[1].kind
                        else {
                            return None;
                        };
                        if aargs.len() != 1
                            || !matches!(
                                intrinsic_callee(ac, self.sema, self.interner),
                                Some(MathIntrinsic::Abs)
                            )
                        {
                            return None;
                        }
                        (COL_MAXABS, &aargs[0])
                    }
                    _ => return None,
                }
            }
        };
        // The data is `x[i*N + j]`: row stride exactly the column count, no further offset, f32.
        let (x, xidx) = as_index1(data)?;
        let (stride, off) = match_row_col_off(xidx, ivar, jvar, self.interner)?;
        if !off.is_empty()
            || stride != cols
            || x == out
            || [ivar, jvar].contains(&x)
            || scalar_of(data, self.sema) != Some(wukong_types::Scalar::F32)
        {
            return None;
        }
        // A first-row seed must read row 0 of the SAME matrix the fold streams.
        match seed {
            ColSeed::FirstRow(x0) | ColSeed::FirstRowAbs(x0) if x0 != x => return None,
            _ => {}
        }
        if matches!(cols, Dim::Lit(c) if c <= 0) {
            return None;
        }
        Some(ColSumNest {
            x,
            out,
            rows,
            cols,
            op: colop,
        })
    }
}
