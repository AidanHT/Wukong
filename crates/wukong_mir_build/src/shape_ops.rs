//! Lowering for the shape-typed surface beyond plain indexing — an `impl FnLowerer` continuation:
//!
//!  * **Derived extents** (`Tensor[f32, 2*N]`, `Tensor[f32, S, H*HD]`): a dimension expression is
//!    evaluated at run time from the hidden dimension parameters, in the canonical normal form sema
//!    type-checked ([`FnLowerer::emit_dim_poly`]), so a stride is computed from exactly the extent
//!    the shape checker proved things about.
//!  * **Run-time `?` extents**: every `?` axis of a tensor PARAMETER is threaded in as one more
//!    hidden `i64` parameter ([`dynamic_extent_params`], after the symbolic dims), so `t.dim(k)`,
//!    a stride over an interior `?` axis and a `for` over a `?`-length tensor can read it.
//!  * **Extent queries** `t.dim(k)` / `t.len()` on a tensor or array, and **`for` over a tensor**
//!    (elements of a rank-1 tensor, row views of a rank-2+ one).
//!
//! A rank-reducing view (`a[i]` on a `Tensor[f32, M, N]`) needs no code here: `tindex` rewrites it
//! to the address of its first element (`&a[i, 0]`), which every existing path already lowers.

use wukong_ast::{self as ast, Block, Expr, ExprKind, ForIter, Pattern, StmtKind};
use wukong_mir::{BinOp, CmpOp, MirType, Op, ValueId};
use wukong_span::{FxHashMap as HashMap, Symbol};
use wukong_types::{Dim, DimAtom, DimPoly, Layout, Ty};

use crate::FnLowerer;

/// The hidden run-time extents a signature threads in for its `?` axes: one `i64` per `?` axis of
/// each parameter whose type IS a tensor, as `(parameter index, axis)`, in parameter order then axis
/// order. A tensor nested in a tuple, behind a pointer, or in a return type gets none (its `?`
/// extent stays unreadable). `bind_dim_params` declares them right after the symbolic dims and
/// `call_dim_args` passes them in the same order, so the ABI agrees by construction.
pub(crate) fn dynamic_extent_params(param_tys: &[Ty]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (pi, t) in param_tys.iter().enumerate() {
        if let Ty::Tensor { shape, .. } = t {
            for (axis, d) in shape.0.iter().enumerate() {
                if matches!(d, Dim::Dynamic) {
                    out.push((pi, axis));
                }
            }
        }
    }
    out
}

/// Does `b` anywhere assign a whole new value to the name `name` (`name = …`)? A binding that is
/// re-pointed may no longer have the extent its `?` axes were bound with, so its extents are only
/// recorded when it is never rebound. Syntactic and conservative: an assignment to an inner
/// shadowing binding of the same name counts too.
pub(crate) fn rebinds_name(b: &Block, name: Symbol) -> bool {
    b.stmts.iter().any(|s| stmt_rebinds(s, name))
        || b.tail.as_ref().is_some_and(|t| expr_rebinds(t, name))
}

fn stmt_rebinds(s: &ast::Stmt, name: Symbol) -> bool {
    match &s.kind {
        StmtKind::Let { init, .. } => init.as_ref().is_some_and(|e| expr_rebinds(e, name)),
        StmtKind::Assign { target, value, .. } => {
            matches!(&target.kind, ExprKind::Path(p) if p.is_single() && p.first().sym == name)
                || expr_rebinds(target, name)
                || expr_rebinds(value, name)
        }
        StmtKind::Expr(e) | StmtKind::Defer(e) => expr_rebinds(e, name),
        StmtKind::Return(e) | StmtKind::Break(_, e) => {
            e.as_ref().is_some_and(|e| expr_rebinds(e, name))
        }
        StmtKind::Continue(_) => false,
        StmtKind::While { cond, body, .. } => expr_rebinds(cond, name) || rebinds_name(body, name),
        StmtKind::For { iter, body, .. } => {
            let in_iter = match iter {
                ForIter::Range {
                    start, end, step, ..
                } => {
                    expr_rebinds(start, name)
                        || end.as_ref().is_some_and(|e| expr_rebinds(e, name))
                        || step.as_ref().is_some_and(|e| expr_rebinds(e, name))
                }
                ForIter::Expr(e) => expr_rebinds(e, name),
            };
            in_iter || rebinds_name(body, name)
        }
    }
}

fn expr_rebinds(e: &Expr, name: Symbol) -> bool {
    match &e.kind {
        ExprKind::Unary { expr, .. }
        | ExprKind::Field { base: expr, .. }
        | ExprKind::TupleField { base: expr, .. }
        | ExprKind::Cast { expr, .. } => expr_rebinds(expr, name),
        ExprKind::Binary { lhs, rhs, .. } => expr_rebinds(lhs, name) || expr_rebinds(rhs, name),
        ExprKind::Call { callee, args, .. } => {
            expr_rebinds(callee, name) || args.iter().any(|a| expr_rebinds(a, name))
        }
        ExprKind::Index { base, indices } => {
            expr_rebinds(base, name) || indices.iter().any(|i| expr_rebinds(i, name))
        }
        ExprKind::StructLit { fields, rest, .. } => {
            fields.iter().any(|f| expr_rebinds(&f.value, name))
                || rest.as_deref().is_some_and(|r| expr_rebinds(r, name))
        }
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => xs.iter().any(|x| expr_rebinds(x, name)),
        ExprKind::ArrayRepeat { value, count } => {
            expr_rebinds(value, name) || expr_rebinds(count, name)
        }
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => rebinds_name(b, name),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            expr_rebinds(cond, name)
                || rebinds_name(then_branch, name)
                || else_branch
                    .as_deref()
                    .is_some_and(|x| expr_rebinds(x, name))
        }
        ExprKind::Match { scrutinee, arms } => {
            expr_rebinds(scrutinee, name)
                || arms.iter().any(|a| {
                    a.guard.as_ref().is_some_and(|g| expr_rebinds(g, name))
                        || expr_rebinds(&a.body, name)
                })
        }
        _ => false,
    }
}

impl FnLowerer<'_> {
    /// A dimension variable's run-time value as an `i64`: its binding (a hidden dimension parameter,
    /// or a run-time turbofish value's slot, `turbofish.rs`), coerced.
    ///
    /// A dimension in a TYPE names one of this function's generics, which `bind_dim_params` bound
    /// in the outermost scope, never a local the body declares later under the same name: sema
    /// resolves `Tensor[i32, N]` to the generic whatever is in scope. Looked up innermost-first, a
    /// `let N = 2;` in `fn g<N>(a: Tensor[i32, N])` made every later `a.dim(0)` and every bound on
    /// `a` read 2.
    pub(crate) fn dim_var_value(&mut self, sym: Symbol) -> Option<ValueId> {
        let own = self
            .scopes
            .first()
            .filter(|_| self.type_params.contains(&sym))
            .and_then(|s| s.get(&sym))
            .cloned();
        let (slot, ty) = match own {
            Some(b) => b,
            None => self.lookup(sym)?,
        };
        let v = self.builder.build(ty.clone(), Op::Load(slot, ty.clone()));
        Some(self.coerce_to(v, &ty, &MirType::I64, true))
    }

    /// An `i64` constant. (Named apart from `builtins.rs`'s `const_i64(i128)`: both are methods of
    /// `FnLowerer`, so one name would be a duplicate definition.)
    fn dim_i64(&mut self, v: i64) -> ValueId {
        self.builder
            .build(MirType::I64, Op::ConstInt(v as i128, MirType::I64))
    }

    fn bin_i64(&mut self, op: BinOp, a: ValueId, b: ValueId) -> ValueId {
        self.builder.build(MirType::I64, Op::Bin(op, a, b))
    }

    fn is_negative_i64(&mut self, v: ValueId) -> ValueId {
        let zero = self.dim_i64(0);
        self.builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, v, zero))
    }

    /// Evaluate a derived extent at run time: `Σ coeff·Π atom^exp` in `i64`, with each variable read
    /// through `var` and each division/remainder atom as the FLOOR `/` `%` of its evaluated operands
    /// — the semantics the normal form (and every compile-time substitution of it) assumes. It
    /// evaluates the CANONICAL form — the one sema compared — so two spellings of one extent produce
    /// one value. A constant-divisor atom's numerator has non-negative coefficients by construction
    /// (see `wukong_types::dim`), so the machine's truncating division already is the floor there.
    /// A symbolic divisor's numerator can be negative for valid extents — `(L - K)/S + 1` at `L < K`
    /// is `0`, which truncation would make `1` — so that division is corrected to the floor.
    pub(crate) fn emit_dim_poly(
        &mut self,
        p: &DimPoly,
        var: &dyn Fn(&mut Self, Symbol) -> Option<ValueId>,
    ) -> Option<ValueId> {
        let mut acc: Option<ValueId> = None;
        for (m, c) in p.terms() {
            let mut term: Option<ValueId> = None;
            for (atom, e) in m.factors() {
                let av = match atom {
                    DimAtom::Var(s) => var(&mut *self, *s)?,
                    DimAtom::Div(n, d) => {
                        let nv = self.emit_dim_poly(n, var)?;
                        let dv = self.emit_dim_poly(d, var)?;
                        let q = self.bin_i64(BinOp::SDiv, nv, dv);
                        if d.as_constant().is_some() {
                            q
                        } else {
                            // floor = trunc - 1 exactly when the remainder is negative (for the
                            // positive divisor an extent has).
                            let r = self.bin_i64(BinOp::SRem, nv, dv);
                            let neg = self.is_negative_i64(r);
                            let one = self.dim_i64(1);
                            let q1 = self.bin_i64(BinOp::Sub, q, one);
                            self.builder.build(MirType::I64, Op::Select(neg, q1, q))
                        }
                    }
                    DimAtom::Rem(n, d) => {
                        let nv = self.emit_dim_poly(n, var)?;
                        let dv = self.emit_dim_poly(d, var)?;
                        let r = self.bin_i64(BinOp::SRem, nv, dv);
                        if d.as_constant().is_some() {
                            r
                        } else {
                            // The non-negative remainder that matches the floor quotient.
                            let neg = self.is_negative_i64(r);
                            let r1 = self.bin_i64(BinOp::Add, r, dv);
                            self.builder.build(MirType::I64, Op::Select(neg, r1, r))
                        }
                    }
                };
                for _ in 0..*e {
                    term = Some(match term {
                        None => av,
                        Some(t) => self.bin_i64(BinOp::Mul, t, av),
                    });
                }
            }
            let coeff = i64::try_from(*c).ok()?;
            let t = match term {
                None => self.dim_i64(coeff),
                Some(t) if coeff == 1 => t,
                Some(t) => {
                    let k = self.dim_i64(coeff);
                    self.bin_i64(BinOp::Mul, t, k)
                }
            };
            acc = Some(match acc {
                None => t,
                Some(a) => self.bin_i64(BinOp::Add, a, t),
            });
        }
        Some(match acc {
            Some(v) => v,
            None => self.dim_i64(0),
        })
    }

    /// Where a tensor-valued expression's `?` extents live: the binding slot of the parameter (or
    /// immutable view local) it reads, and the axis offset into that binding's shape — `0` for the
    /// binding itself, `k` for a view `&base[i0, …, ik-1, 0, …]` that fixed `k` leading axes
    /// (`tindex`'s spelling of the partial index `base[i0, …]`).
    fn view_source(&self, e: &Expr) -> Option<(ValueId, usize)> {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() => Some((self.lookup(p.first().sym)?.0, 0)),
            ExprKind::Unary {
                op: ast::UnOp::Ref,
                expr,
            } => {
                let ExprKind::Index { base, .. } = &expr.kind else {
                    return None;
                };
                let (Ty::Tensor { shape: bs, .. }, Ty::Tensor { shape: vs, .. }) =
                    (self.expr_ty(base), self.expr_ty(e))
                else {
                    return None;
                };
                let fixed = bs.rank().checked_sub(vs.rank())?;
                let (slot, off) = self.view_source(base)?;
                Some((slot, off + fixed))
            }
            _ => None,
        }
    }

    /// The run-time extent of axis `axis` of the tensor-valued expression `base` whose shape is
    /// `dims`, as an `i64`: a fixed, symbolic or derived extent from the shape itself, or — for a
    /// `?` axis — the hidden extent its parameter was called with. `None` when a `?` extent is not
    /// known here (a `?` tensor returned by a call, or a rebound binding).
    pub(crate) fn tensor_axis_extent(
        &mut self,
        base: &Expr,
        dims: &[Dim],
        axis: usize,
    ) -> Option<ValueId> {
        match dims.get(axis)? {
            Dim::Dynamic => {
                let (slot, off) = self.view_source(base)?;
                let ext = *self.dyn_extents.get(&(slot, axis + off))?;
                Some(
                    self.builder
                        .build(MirType::I64, Op::Load(ext, MirType::I64)),
                )
            }
            d => self.tensor_dim_value(d),
        }
    }

    /// Record the hidden `?` extents `bind_dim_params` declared for parameter `pi` against the slot
    /// that parameter was bound at, unless the body re-points the parameter (then they could go
    /// stale, and reading one is refused instead).
    pub(crate) fn record_param_extents(
        &mut self,
        dyn_params: &[(usize, usize, ValueId)],
        pi: usize,
        bound: ValueId,
        name: Symbol,
        body: &Block,
    ) {
        if !dyn_params.iter().any(|(p, _, _)| *p == pi) || rebinds_name(body, name) {
            return;
        }
        for &(p, axis, ext) in dyn_params {
            if p == pi {
                self.dyn_extents.insert((bound, axis), ext);
            }
        }
    }

    /// An immutable `let v = <tensor>` view inherits the `?` extents of what it views (a parameter,
    /// another such view, or a partial index of one), so `v.dim(k)` and a stride over `v`'s `?` axis
    /// are readable. A `mut` binding can be re-pointed and inherits nothing.
    pub(crate) fn inherit_dyn_extents(&mut self, init: &Expr, slot: ValueId) {
        let Ty::Tensor { shape, .. } = self.expr_ty(init) else {
            return;
        };
        let Some((src, off)) = self.view_source(init) else {
            return;
        };
        for (axis, d) in shape.0.iter().enumerate() {
            if matches!(d, Dim::Dynamic) {
                if let Some(&ext) = self.dyn_extents.get(&(src, axis + off)) {
                    self.dyn_extents.insert((slot, axis), ext);
                }
            }
        }
    }

    /// The extent of axis `axis` of argument `arg` for the `?` axis of the tensor parameter `pty`:
    /// read off a tensor argument's own shape, or — for an array decaying to the parameter — the
    /// array's length divided by the product of the parameter's OTHER extents (`dim_map` holds the
    /// callee's dimension values for this call). `None` when it cannot be determined.
    pub(crate) fn arg_axis_extent(
        &mut self,
        pty: &Ty,
        arg: &Expr,
        axis: usize,
        dim_map: &HashMap<Symbol, ValueId>,
    ) -> Option<ValueId> {
        let Ty::Tensor { shape: ps, .. } = pty else {
            return None;
        };
        match self.expr_ty(arg) {
            Ty::Tensor { shape, .. } if shape.rank() == ps.rank() => {
                self.tensor_axis_extent(arg, &shape.0, axis)
            }
            Ty::Array { len, .. } => {
                let mut others: Option<ValueId> = None;
                for (k, d) in ps.0.iter().enumerate() {
                    if k == axis {
                        continue;
                    }
                    let v = match d {
                        Dim::Const(n) => self.dim_i64(i64::try_from(*n).ok()?),
                        Dim::Var(g) => *dim_map.get(g)?,
                        Dim::Expr(p) => self.emit_dim_poly(p, &|_, s| dim_map.get(&s).copied())?,
                        // Two `?` axes share one array length: neither is determined.
                        Dim::Dynamic => return None,
                    };
                    others = Some(match others {
                        None => v,
                        Some(acc) => self.bin_i64(BinOp::Mul, acc, v),
                    });
                }
                let len_v = self.dim_i64(i64::try_from(len).ok()?);
                Some(match others {
                    None => len_v,
                    Some(o) => self.bin_i64(BinOp::SDiv, len_v, o),
                })
            }
            _ => None,
        }
    }

    /// `t.dim(k)` / `t.len()` on a tensor or a fixed-size array: the extent of axis `k` (`len` is
    /// axis 0) as an `i64` — a constant for a fixed axis, the hidden dimension parameter (or a
    /// derived extent computed from them) for a symbolic one, the hidden extent for a `?` one.
    /// `None` for any other receiver or method, which the caller lowers as before.
    pub(crate) fn lower_extent_method(
        &mut self,
        base: &Expr,
        name: Symbol,
        args: &[Expr],
    ) -> Option<ValueId> {
        let method = self.interner.resolve(name);
        let is_dim = method == "dim";
        if !is_dim && method != "len" {
            return None;
        }
        let bty = self.expr_ty(base);
        if !matches!(bty, Ty::Tensor { .. } | Ty::Array { .. }) {
            return None;
        }
        let axis = if is_dim {
            // Sema accepted only a compile-time constant axis in range; read it the same
            // non-speculative way every other lowered constant is read.
            match args
                .first()
                .and_then(|a| self.const_runtime_u64_expr(a))
                .and_then(|k| usize::try_from(k).ok())
            {
                Some(k) => k,
                None => {
                    self.unsupported(base.span, "`.dim(k)` whose axis is not an integer constant");
                    return Some(self.dim_i64(0));
                }
            }
        } else {
            0
        };
        // The receiver is evaluated for its effects unless it is a plain name, so `f().len()` still
        // calls `f` (the extent itself comes from the type).
        if !matches!(base.kind, ExprKind::Path(_)) {
            let _ = self.lower_expr(base);
        }
        match bty {
            Ty::Array { len, .. } => Some(self.dim_i64(i64::try_from(len).unwrap_or(i64::MAX))),
            Ty::Tensor { shape, .. } => match self.tensor_axis_extent(base, &shape.0, axis) {
                Some(v) => Some(v),
                None => {
                    self.unsupported(
                        base.span,
                        "reading a `?` extent that is not known here (only a `?` axis of a tensor \
                         parameter, or of an immutable view of one, carries its run-time extent)",
                    );
                    Some(self.dim_i64(0))
                }
            },
            _ => None,
        }
    }

    /// `for x in t` over a tensor: a rank-1 tensor yields its elements (exactly `for x in arr`), a
    /// contiguous rank-2+ tensor yields its rows as VIEWS of the remaining axes — `x` is the
    /// sub-tensor `t[i]`, a tensor local holding the row's base pointer, never a copy. The trip
    /// count is the extent of axis 0, and the row stride the product of the remaining extents, each
    /// computed once before the loop. Returns `false` (the caller reports the construct) for a
    /// layout or pattern this does not lower.
    pub(crate) fn lower_for_tensor(
        &mut self,
        label: Option<Symbol>,
        pat: &Pattern,
        e: &Expr,
        body: &Block,
    ) -> bool {
        let bind_name = match &pat.kind {
            ast::PatKind::Ident(name) => Some(*name),
            ast::PatKind::Wildcard => None,
            _ => return false,
        };
        let Ty::Tensor {
            elem,
            shape,
            layout,
            ..
        } = self.expr_ty(e)
        else {
            return false;
        };
        let rank = shape.rank();
        let layout_ok = match rank {
            0 => false,
            1 => matches!(layout, Layout::Contiguous | Layout::ColMajor),
            _ => layout == Layout::Contiguous,
        };
        if !layout_ok {
            return false;
        }
        let Some(count) = self.tensor_axis_extent(e, &shape.0, 0) else {
            self.unsupported(
                e.span,
                "iterating a tensor whose leading `?` extent is not known here",
            );
            return true;
        };
        let mut row: Option<ValueId> = None;
        for k in 1..rank {
            let Some(d) = self.tensor_axis_extent(e, &shape.0, k) else {
                self.unsupported(
                    e.span,
                    "iterating the rows of a tensor whose inner `?` extent is not known here",
                );
                return true;
            };
            row = Some(match row {
                None => d,
                Some(r) => self.bin_i64(BinOp::Mul, r, d),
            });
        }
        let base_ptr = self.lower_expr(e);
        let elem_mir = MirType::from_scalar(elem);
        let ity = MirType::I64;

        let slot = self.builder.alloca(ity.clone());
        let zero = self.dim_i64(0);
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: zero,
        });

        self.push_scope();

        let header = self.builder.new_block();
        let body_bb = self.builder.new_block();
        let latch = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);

        // header: i < extent(axis 0)
        self.builder.switch_to(header);
        self.terminated = false;
        let i_val = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let c = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, i_val, count));
        self.builder.cond_br(c, body_bb, vec![], exit, vec![]);

        // body: bind `x` to element / row `i`, then the user body; `continue` targets the latch.
        self.builder.switch_to(body_bb);
        self.terminated = false;
        if let Some(name) = bind_name {
            let i_cur = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
            let index = match row {
                None => i_cur,
                Some(r) => self.bin_i64(BinOp::Mul, i_cur, r),
            };
            let elem_ptr = self.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: base_ptr,
                    index,
                    elem: elem_mir.clone(),
                },
            );
            if row.is_none() {
                // A scalar element loads into a fresh slot: the loop variable is an ordinary local.
                let v = self
                    .builder
                    .build(elem_mir.clone(), Op::Load(elem_ptr, elem_mir.clone()));
                let xslot = self.builder.alloca(elem_mir.clone());
                self.builder.build_void(Op::Store {
                    ptr: xslot,
                    value: v,
                });
                self.bind(name, xslot, elem_mir.clone());
            } else {
                // A row is a view: a tensor local whose slot holds the row's base pointer.
                let vslot = self.builder.alloca(MirType::Ptr);
                self.builder.build_void(Op::Store {
                    ptr: vslot,
                    value: elem_ptr,
                });
                self.bind(name, vslot, MirType::Ptr);
                // The row's `?` axes are the iterand's axes 1.. — readable when the iterand's are
                // and the body never re-points the loop variable.
                if !rebinds_name(body, name) {
                    if let Some((src, off)) = self.view_source(e) {
                        for k in 1..rank {
                            if matches!(shape.0[k], Dim::Dynamic) {
                                if let Some(&ext) = self.dyn_extents.get(&(src, k + off)) {
                                    self.dyn_extents.insert((vslot, k - 1), ext);
                                }
                            }
                        }
                    }
                }
            }
        }
        self.loops.push((label, latch, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            self.builder.br(latch, vec![]);
        }

        // latch: i += 1, back to the header.
        self.builder.switch_to(latch);
        self.terminated = false;
        let cur = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let one = self.dim_i64(1);
        let next = self.bin_i64(BinOp::Add, cur, one);
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: next,
        });
        self.builder.br(header, vec![]);

        self.pop_scope();
        self.builder.switch_to(exit);
        self.terminated = false;
        true
    }
}

#[cfg(test)]
mod tests {
    use wukong_diag::Diagnostic;
    use wukong_mir::verify::verify_program;
    use wukong_mir::{Function, Op, Program};
    use wukong_span::{Interner, SourceId};

    fn lower(src: &str) -> (Program, Vec<Diagnostic>, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (prog, diags) = crate::lower_program(&module, &sema, &mut interner);
        (prog, diags, interner)
    }

    fn func<'p>(prog: &'p Program, interner: &Interner, name: &str) -> &'p Function {
        prog.funcs
            .iter()
            .find(|f| interner.resolve(f.name) == name)
            .unwrap_or_else(|| panic!("no function `{name}`"))
    }

    /// Lower `src` and require clean, whole-program-verified MIR — the call-vs-signature check is
    /// what proves the hidden dimension / extent ABI agrees on both sides of every call.
    #[track_caller]
    fn lowers_clean(src: &str) -> (Program, Interner) {
        let (prog, diags, interner) = lower(src);
        assert!(diags.is_empty(), "lowering diagnostics: {diags:?}");
        let errs = verify_program(&prog);
        assert!(errs.is_empty(), "MIR verifier: {errs:?}");
        (prog, interner)
    }

    #[test]
    fn a_question_mark_axis_is_a_hidden_extent_parameter() {
        let (prog, interner) = lowers_clean(
            "fn cols(a: Tensor[f32, 2, ?]) -> i64 { return a.dim(1); } \
             fn get(a: Tensor[f32, 2, ?], i: i64, j: i64) -> f32 { return a[i, j]; } \
             fn main() -> i32 { let b: [f32; 6] = [0.0; 6]; \
             return (cols(b) as f32 + get(b, 1, 2)) as i32; }",
        );
        // One hidden `i64` extent for the `?` axis, then the buffer.
        assert_eq!(func(&prog, &interner, "cols").params.len(), 2);
        // ...then the two scalar indices.
        assert_eq!(func(&prog, &interner, "get").params.len(), 4);
    }

    #[test]
    fn a_derived_extent_threads_its_variables() {
        let (prog, interner) = lowers_clean(
            "fn pairs<N>(a: [f32; 2*N]) -> f32 { let mut s: f32 = 0.0; \
             for i in 0..N { s = s + a[2*i] * a[2*i + 1]; } return s; } \
             fn split<S, D>(qkv: Tensor[f32, S, 3*D], mut q: Tensor[f32, S, D]) { \
             for s in 0..S { for d in 0..D { q[s, d] = qkv[s, 2*D + d]; } } } \
             fn main() -> i32 { let x: [f32; 8] = [1.0; 8]; let b: [f32; 12] = [1.0; 12]; \
             let mut q: [f32; 4] = [0.0; 4]; split(b as Tensor[f32, 2, 6], q); \
             return pairs(x) as i32; }",
        );
        // `N` (solved from `2*N = 8`) leads the buffer; `S` and `D` lead `split`'s two buffers.
        assert_eq!(func(&prog, &interner, "pairs").params.len(), 2);
        assert_eq!(func(&prog, &interner, "split").params.len(), 4);
    }

    #[test]
    fn views_reshapes_extents_and_tensor_loops_lower() {
        let (prog, interner) = lowers_clean(
            "fn row_sum<N>(r: Tensor[f32, N]) -> f32 { let mut s: f32 = 0.0; \
             for x in r { s = s + x; } return s; } \
             fn main() -> i32 { let mut buf: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; \
             let m = buf as Tensor[f32, 2, 3]; let r = m[1]; \
             let mut t: f32 = r[0] + m[0][2] + row_sum(m[0]); \
             for row in m { t = t + row_sum(row); } \
             return (t + (m.dim(1) + m.len() + buf.len()) as f32) as i32; }",
        );
        // A view is an address, never a copy: no call is made to build one, and `main` calls only
        // the user function.
        let main = func(&prog, &interner, "main");
        let calls: Vec<&str> = main
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter_map(|i| match &i.op {
                Op::Call { func, .. } => Some(interner.resolve(*func)),
                _ => None,
            })
            .collect();
        assert!(
            calls.iter().all(|c| *c == "row_sum"),
            "unexpected calls in main: {calls:?}"
        );
    }

    /// A symbolic divisor's quotient is the FLOOR the normal form (and sema's compile-time
    /// substitution) assumes: `(L - K)/S + 1` at `L < K` is 0, not the 1 truncation gives, so the
    /// run-time extent carries the fix-up `select`. A constant divisor's numerator is non-negative
    /// by construction (`wukong_types::dim`) and needs none.
    #[test]
    fn a_symbolic_divisor_is_floor_divided_at_run_time() {
        let (prog, interner) = lowers_clean(
            "fn outs<L, K, S>(a: Tensor[f32, (L - K)/S + 1]) -> i64 { return a.len(); }              fn half<N>(a: Tensor[f32, (N + 1)/2]) -> i64 { return a.len(); }              fn main() -> i32 { let x: [f32; 2] = [0.0; 2]; let y: [f32; 2] = [0.0; 2];              return (outs::<5, 3, 2>(x) + half::<3>(y)) as i32; }",
        );
        let selects = |name: &str| {
            func(&prog, &interner, name)
                .blocks
                .iter()
                .flat_map(|b| b.insts.iter())
                .filter(|i| matches!(i.op, Op::Select(..)))
                .count()
        };
        assert_eq!(selects("outs"), 1);
        assert_eq!(selects("half"), 0);
    }

    #[test]
    fn a_question_mark_extent_binds_a_generic_dimension_at_a_call() {
        // Sema leaves `N` unbound against a `?` argument (the lenient escape hatch); lowering
        // passes the caller's run-time extent instead of refusing the call with C0001.
        let (prog, interner) = lowers_clean(
            "fn gsum<N>(a: Tensor[f32, N]) -> f32 { let mut s: f32 = 0.0; \
             for i in 0..N { s = s + a[i]; } return s; } \
             fn fwd(a: Tensor[f32, ?]) -> f32 { return gsum(a); } \
             fn main() -> i32 { let b: [f32; 4] = [1.0; 4]; return fwd(b) as i32; }",
        );
        // `N`, then the buffer; the `?` extent, then the buffer.
        assert_eq!(func(&prog, &interner, "gsum").params.len(), 2);
        assert_eq!(func(&prog, &interner, "fwd").params.len(), 2);
    }

    #[test]
    fn an_unknown_question_mark_extent_is_refused_not_guessed() {
        // A `?` tensor RETURNED by a call carries no extent the caller can pass on.
        let (_, diags, _) = lower(
            "fn id(a: Tensor[f32, ?]) -> Tensor[f32, ?] { return a; } \
             fn len(a: Tensor[f32, ?]) -> i64 { return a.dim(0); } \
             fn main() -> i32 { let b: [f32; 4] = [0.0; 4]; return len(id(b)) as i32; }",
        );
        assert!(
            diags.iter().any(|d| d.code == Some("C0001")),
            "expected a C0001 refusal: {diags:?}"
        );
    }
}
