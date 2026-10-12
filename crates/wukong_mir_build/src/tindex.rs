//! **Tensor multi-index normalization** — the pre-lowering pass that makes the shape-typed surface
//! and the flat-array surface the *same* program.
//!
//! `a[i, j]` on a `Tensor[f32, M, N]` and `a[i*N + j]` on a `[f32; M*N]` address the identical byte
//! and always have; but only the second spelling was ever *seen* as an index by `mir_build`. Every
//! kernel recognizer and the whole autovectorizer match on `ExprKind::Index { indices }` with
//! `indices.len() == 1` — a 2-index access simply fell off the end of the match and lowered to a
//! scalar `Gep` nest. The measured consequence: the shape-typed spelling of an elementwise loop ran
//! **5.7x slower** than the flat spelling of the same arithmetic over the same memory, because the
//! flat one vectorized to a 256-bit `veckernel` and the tensor one did not vectorize at all.
//!
//! So before anything is lowered, rewrite `a[i0, …, in]` into `a[Σ ik·stridek]` — the row-major flat
//! offset the lowering was going to compute anyway ([`FnLowerer::lower_multi_index`]), only now
//! spelled as an *index expression*, which every downstream matcher already understands. The strides
//! are compile-time constants, so the optimizer folds them exactly as it folds the hand-written
//! `i*N + j`. Nothing about the addressing changes; what changes is that the recognizers can see it.
//!
//! **Scope — deliberately narrow.** A tensor is rewritten only when every one of these holds:
//!   * contiguous (row-major) layout, rank >= 1, rank == the number of indices;
//!   * every extent is a compile-time `Dim::Const` (a symbolic `Tensor[f32, M, N]` keeps the hidden
//!     runtime-dim path in [`FnLowerer::lower_multi_index_dyn`] untouched);
//!   * the element count is at most `i32::MAX`, and every index expression has the *same* integer
//!     scalar type of at least 32 bits.
//!
//! The last two together are what make the rewrite **value-identical**: the old lowering did the
//! flat-offset arithmetic in `i64` after sign-extending each index, the new one does it in the
//! index's own type — and for any in-bounds index the largest intermediate is `count - 1 <=
//! i32::MAX`, which every accepted index type represents exactly. Out of bounds the two differ, but
//! out of bounds is exactly where the flat spelling already differed from itself, and matching the
//! flat spelling bit-for-bit is the point.
//!
//! **Rank-reducing views.** The same pass gives a partial index its lowering. Sema types `a[i]` on
//! a `Tensor[T, M, N]` as the VIEW `Tensor[T, N]` — the row, a sub-buffer of `a` — and the address
//! of a row is the address of its first element, so the view is rewritten to `&a[i, 0]` (padded
//! with zero indices to full rank, then flattened like any other full index). Every downstream
//! path already lowers an element address, and no recognizer mistakes it for an element read: the
//! one-index spelling `a[flat]` on a rank-2 base means "element" after this pass, never "row". An
//! index INTO a view, `a[i][j]`, is first folded back into the full index `a[i, j]` it means, so the
//! chained spelling is flattened and recognized exactly like the direct one.
//!
//! Synthesized nodes get fresh [`NodeId`]s minted downward from just under `NodeId::DUMMY` (see
//! [`FRESH_ID_TOP`] for why "one past the largest *typed* id" is not collision-free), and their
//! types are handed back to the caller to merge into the type table, so `expr_ty` answers for them
//! like any other node. Ids are allocated in AST order, so the pass is a pure function of its input.

use wukong_ast::{
    self as ast, Block, Expr, ExprKind, ForIter, ItemKind, Module, NodeId, Stmt, StmtKind,
};
use wukong_sema::SemaResult;
use wukong_span::{FxHashMap as HashMap, Interner, Span};
use wukong_types::{Dim, Layout, Scalar, Ty};

/// Rewrite every flattenable tensor multi-index in `module`. Returns the rewritten module and the
/// types of the nodes it synthesized, or `None` when nothing was rewritten — in which case the
/// caller keeps the original `&Module`/`&SemaResult` and the whole pass costs one scan of the type
/// table (no clone, byte-identical output for every program with no statically-shaped tensor).
pub(crate) fn linearize_module(
    module: &Module,
    sema: &SemaResult,
    interner: &mut Interner,
) -> Option<(Module, HashMap<NodeId, Ty>)> {
    // Cheap pre-filter: a program with no tensor type anywhere cannot have a tensor multi-index, and
    // must not pay for a module clone. This is a flat scan of an already-built map, not a tree walk.
    if !sema.types.values().any(|t| matches!(t, Ty::Tensor { .. })) {
        return None;
    }
    let mut lz = Linearizer {
        types: &sema.types,
        interner,
        extra: HashMap::default(),
        next_id: FRESH_ID_TOP,
        changed: false,
    };
    let mut out = module.clone();
    for item in &mut out.items {
        if let ItemKind::Fn(f) = &mut item.kind {
            if let Some(body) = &mut f.body {
                lz.block(body);
            }
        }
    }
    if !lz.changed {
        return None;
    }
    Some((out, lz.extra))
}

/// Fresh [`NodeId`]s are handed out **downward** from just below [`NodeId::DUMMY`], because the
/// parser hands them out upward from 0 and never runs out. Any id in `[FRESH_ID_TOP - synthesized,
/// FRESH_ID_TOP]` is therefore unreachable by a parsed node, whatever the program's size.
///
/// The obvious alternative — `max(sema.types.keys()) + 1` — is **not** collision-free: `sema.types`
/// records expression nodes only, so a statement, block, pattern, or an expression sema never typed
/// (an array-length expression inside a type, say) can carry an id above that maximum. Reusing one
/// would make `expr_ty` answer for that real node with a synthesized node's type.
const FRESH_ID_TOP: u32 = u32::MAX - 1;

/// The smallest id this pass will mint. Reaching it needs ~2 billion synthesized nodes; the check
/// exists so the counter can never wrap into parser territory rather than because it can be hit.
const FRESH_ID_FLOOR: u32 = u32::MAX / 2;

struct Linearizer<'a> {
    types: &'a HashMap<NodeId, Ty>,
    interner: &'a mut Interner,
    extra: HashMap<NodeId, Ty>,
    next_id: u32,
    changed: bool,
}

impl Linearizer<'_> {
    // ---- synthesis ----

    /// A `NodeId` no parsed node can hold, or `None` once the (unreachable) floor is hit — which
    /// declines the whole rewrite rather than minting an id that might already be in use.
    fn fresh(&mut self) -> Option<NodeId> {
        if self.next_id <= FRESH_ID_FLOOR {
            return None;
        }
        let id = NodeId(self.next_id);
        self.next_id -= 1;
        Some(id)
    }

    /// An integer literal node of scalar type `sc`. The literal's text is what `parse_int` reads at
    /// lowering, so it must be plain decimal — `n` is a row-major stride, always non-negative.
    fn int_lit(&mut self, n: u64, sc: Scalar, span: Span) -> Option<Expr> {
        let sym = self.interner.intern(&n.to_string());
        let id = self.fresh()?;
        self.extra.insert(id, Ty::Scalar(sc));
        Some(Expr {
            id,
            kind: ExprKind::Int(sym),
            span,
        })
    }

    fn binary(
        &mut self,
        op: ast::BinOp,
        lhs: Expr,
        rhs: Expr,
        sc: Scalar,
        span: Span,
    ) -> Option<Expr> {
        let id = self.fresh()?;
        self.extra.insert(id, Ty::Scalar(sc));
        Some(Expr {
            id,
            kind: ExprKind::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            },
            span,
        })
    }

    /// The row-major flat index for `base[indices…]`, or `None` to leave the access alone (see the
    /// module docs for the full acceptance list). Pure with respect to the AST: it only *reads*
    /// `base`/`indices` and builds a fresh expression.
    fn flatten(&mut self, base: &Expr, indices: &[Expr]) -> Option<Expr> {
        let Some(Ty::Tensor { shape, layout, .. }) = self.types.get(&base.id) else {
            return None;
        };
        if !matches!(layout, Layout::Contiguous | Layout::ColMajor)
            || shape.0.len() != indices.len()
        {
            return None;
        }
        let mut extents = Vec::with_capacity(shape.0.len());
        for d in &shape.0 {
            match d {
                Dim::Const(n) => extents.push(*n),
                // A symbolic / derived / runtime extent has no compile-time stride;
                // `lower_multi_index_dyn` builds it from the hidden dim params instead, and stays the
                // only path for it.
                Dim::Var(_) | Dim::Expr(_) | Dim::Dynamic => return None,
            }
        }
        let mut count: u64 = 1;
        for e in &extents {
            count = count.checked_mul(*e)?;
        }
        // Bound the largest intermediate (`count - 1`) so the arithmetic below cannot overflow the
        // index type for any in-bounds access — this is what makes the rewrite value-identical to
        // the i64 flat-offset the old lowering computed.
        if count == 0 || count > i32::MAX as u64 {
            return None;
        }
        let sc = self.index_scalar(indices)?;

        let rank = extents.len();
        let mut strides = vec![1u64; rank];
        if matches!(layout, Layout::ColMajor) {
            // Column-major: the leading axis has unit stride (`i + j*M`).
            for k in 1..rank {
                strides[k] = strides[k - 1] * extents[k - 1];
            }
        } else {
            for k in (0..rank.saturating_sub(1)).rev() {
                strides[k] = strides[k + 1] * extents[k + 1];
            }
        }

        let mut acc: Option<Expr> = None;
        for (ix, &st) in indices.iter().zip(strides.iter()) {
            // A unit stride contributes the index unchanged — the same "no multiply emitted" rule
            // `lower_multi_index` follows, so the innermost axis stays a bare `j`.
            let term = if st == 1 {
                ix.clone()
            } else {
                let lit = self.int_lit(st, sc, ix.span)?;
                self.binary(ast::BinOp::Mul, ix.clone(), lit, sc, ix.span)?
            };
            acc = Some(match acc {
                None => term,
                Some(a) => {
                    let span = a.span;
                    self.binary(ast::BinOp::Add, a, term, sc, span)?
                }
            });
        }
        acc
    }

    /// The one integer scalar every index of an access shares, if they share one and it is wide
    /// enough to hold the flat offset. A mixed-width access (`a[i32_i, i64_j]`) is declined rather
    /// than silently coerced: the synthesized `Mul`/`Add` must be single-typed to lower cleanly.
    fn index_scalar(&self, indices: &[Expr]) -> Option<Scalar> {
        let mut found: Option<Scalar> = None;
        for ix in indices {
            // A zero this pass padded a view with is typed in `extra`, not in sema's table.
            let Some(Ty::Scalar(s)) = self.types.get(&ix.id).or_else(|| self.extra.get(&ix.id))
            else {
                return None;
            };
            // `Char` is nominally an integer in MIR but is never a sane index; `size() >= 4` keeps
            // an `i8`/`i16` index — whose product with a stride would wrap at once — off this path.
            if !s.is_int() || *s == Scalar::Char || s.size() < 4 {
                return None;
            }
            match found {
                None => found = Some(*s),
                Some(prev) if prev == *s => {}
                Some(_) => return None,
            }
        }
        found
    }

    // ---- traversal ----

    fn block(&mut self, b: &mut Block) {
        for s in &mut b.stmts {
            self.stmt(s);
        }
        if let Some(t) = &mut b.tail {
            self.expr(t);
        }
    }

    fn stmt(&mut self, s: &mut Stmt) {
        match &mut s.kind {
            StmtKind::Let { init, .. } => {
                if let Some(e) = init {
                    self.expr(e);
                }
            }
            StmtKind::Assign { target, value, .. } => {
                self.expr(target);
                self.expr(value);
            }
            StmtKind::Expr(e) | StmtKind::Defer(e) => self.expr(e),
            StmtKind::Return(e) | StmtKind::Break(_, e) => {
                if let Some(e) = e {
                    self.expr(e);
                }
            }
            StmtKind::Continue(_) => {}
            StmtKind::While { cond, body, .. } => {
                self.expr(cond);
                self.block(body);
            }
            StmtKind::For { iter, body, .. } => {
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
                self.block(body);
            }
        }
    }

    /// Is `e` a partial index — an `Index` of a TENSOR that sema typed as a tensor (the VIEW of the
    /// axes it left open)? A full index of a tensor is typed as its element. The base is checked
    /// too: indexing a pointer to (or an array of) tensor handles also yields a tensor-typed value,
    /// and that is an element, not a view.
    fn is_view(&self, e: &Expr) -> bool {
        let ExprKind::Index { base, .. } = &e.kind else {
            return false;
        };
        matches!(self.types.get(&e.id), Some(Ty::Tensor { .. }))
            && matches!(self.types.get(&base.id), Some(Ty::Tensor { .. }))
    }

    /// Fold an index INTO a view back into the full index it means: `a[i][j]` is `a[i, j]`, and
    /// `(x as Tensor[f32, R, C])[r][c]` is `(x as …)[r, c]`. Repeats for a longer chain. The node
    /// keeps its own id, so its type is still the one sema gave the whole chain.
    fn merge_view_chain(&mut self, e: &mut Expr) {
        while let ExprKind::Index { base, indices } = &mut e.kind {
            if !self.is_view(base) {
                return;
            }
            let span = base.span;
            let inner = std::mem::replace(
                base.as_mut(),
                Expr {
                    id: NodeId::DUMMY,
                    kind: ExprKind::TupleLit(Vec::new()),
                    span,
                },
            );
            let ExprKind::Index {
                base: inner_base,
                indices: mut inner_indices,
            } = inner.kind
            else {
                return; // unreachable: `is_view` checked the kind
            };
            inner_indices.append(indices);
            *base = inner_base;
            *indices = inner_indices;
            self.changed = true;
        }
    }

    /// Rewrite the partial index `e` (`a[i]`, a view) to `&a[i, 0, …]` — the address of the view's
    /// first element — with the padded full index flattened when its shape allows. The address
    /// node keeps `e`'s id and so its view type; the element access inside is a fresh node typed as
    /// the element. `None` leaves `e` alone (only when fresh ids run out, which needs billions).
    fn rewrite_view(&mut self, e: &mut Expr) -> Option<()> {
        let ExprKind::Index { base, indices } = &e.kind else {
            return None;
        };
        let Some(Ty::Tensor {
            shape,
            elem,
            elem_var,
            ..
        }) = self.types.get(&base.id)
        else {
            return None;
        };
        // A dtype-generic element is its type parameter, which each instance substitutes.
        let elem = match elem_var {
            Some(v) => Ty::Named(*v),
            None => Ty::Scalar(*elem),
        };
        let rank = shape.0.len();
        // The zero padding takes the type of the written indices (so a uniform index list stays
        // uniform and still flattens), defaulting to `i64`.
        let sc = indices
            .iter()
            .find_map(|ix| match self.types.get(&ix.id) {
                Some(Ty::Scalar(s)) if s.is_int() && *s != Scalar::Char => Some(*s),
                _ => None,
            })
            .unwrap_or(Scalar::I64);
        let base = (**base).clone();
        let mut full = indices.clone();
        while full.len() < rank {
            full.push(self.int_lit(0, sc, e.span)?);
        }
        let inner_kind = match self.flatten(&base, &full) {
            Some(flat) => ExprKind::Index {
                base: Box::new(base),
                indices: vec![flat],
            },
            None => ExprKind::Index {
                base: Box::new(base),
                indices: full,
            },
        };
        let id = self.fresh()?;
        self.extra.insert(id, elem);
        let inner = Expr {
            id,
            kind: inner_kind,
            span: e.span,
        };
        e.kind = ExprKind::Unary {
            op: ast::UnOp::Ref,
            expr: Box::new(inner),
        };
        self.changed = true;
        Some(())
    }

    fn expr(&mut self, e: &mut Expr) {
        // An index into a view is the full index it means; fold it before anything else looks.
        self.merge_view_chain(e);
        // Children first, so a nested multi-index (`w[a[i, j], k]`) is already flat when the outer
        // access is considered.
        match &mut e.kind {
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::Path(_)
            | ExprKind::SizeOf(_)
            | ExprKind::AlignOf(_)
            | ExprKind::Closure { .. } => {}
            ExprKind::Unary { expr, .. }
            | ExprKind::Field { base: expr, .. }
            | ExprKind::TupleField { base: expr, .. }
            | ExprKind::Cast { expr, .. } => self.expr(expr),
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
                for ix in indices.iter_mut() {
                    self.expr(ix);
                }
            }
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
            ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => self.block(b),
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.expr(cond);
                self.block(then_branch);
                if let Some(x) = else_branch {
                    self.expr(x);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee);
                for a in arms {
                    if let Some(g) = &mut a.guard {
                        self.expr(g);
                    }
                    self.expr(&mut a.body);
                }
            }
        }

        // A partial index is a view: its address, not an element (see the module docs).
        if self.is_view(e) {
            self.rewrite_view(e);
            return;
        }
        let flat = match &e.kind {
            ExprKind::Index { base, indices } if indices.len() >= 2 => self.flatten(base, indices),
            _ => None,
        };
        if let Some(flat) = flat {
            if let ExprKind::Index { indices, .. } = &mut e.kind {
                *indices = vec![flat];
                self.changed = true;
            }
        }
    }
}
