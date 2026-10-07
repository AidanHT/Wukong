//! Tensor shape operations beyond plain call unification — an `impl Sema<'_>` continuation like
//! `shape.rs`:
//!
//!  * **Derived extents** (`Tensor[f32, 3*D]`, `Tensor[f32, S, H, HD]` beside `Tensor[f32, S,
//!    H*HD]`): folding a dimension expression into `wukong_types`' canonical normal form, and the
//!    call-site solver that binds a callee's dimension from an argument extent through one
//!    (`2*N = 8` gives `N = 4`).
//!  * **Rank-reducing views**: indexing a contiguous tensor with fewer indices than its rank is the
//!    sub-tensor of the remaining axes, a view of the same buffer (`a[i]` on `Tensor[f32, M, N]` is a
//!    `Tensor[f32, N]`). The type side is in `type_index`; the guard rails — a view is never a
//!    scalar, and a view is not an assignable place — are here.
//!  * **Reshape** (`x as Tensor[f32, S, H, HD]`): a zero-copy reinterpretation whose element count
//!    must provably equal the source's.
//!  * **Extent queries**: `t.dim(k)` and `t.len()` on a tensor or a fixed-size array.
//!  * **Shape-generic array parameters**: `fn f<N>(a: [f32; N])` is the shape-generic buffer
//!    `Tensor[f32, N]` it denotes, instead of a length that silently evaluated to 0.
//!
//! Codes: `E0505` (a dimension expression with no extent), `E0506` (a reshape whose element count is
//! not provably preserved), `E0507` (a misused view or extent query), plus the existing `E0401` /
//! `E0501` / `E0502` / `E0503` families where the error is one of those.

use wukong_span::FxHashMap as HashMap;

use wukong_ast::{BinOp, Expr, ExprKind, PatKind, Pattern, TypeExpr, TypeKind};
use wukong_span::{Span, Symbol};
use wukong_types::{Dim, DimOp, Layout, Scalar, Shape, Ty};

use crate::shape::dims_equal;
use crate::Sema;

/// One queued derived-extent equation from a call site: the parameter's declared extent `decl` (in
/// the callee's dimension names) must equal the argument's extent `arg` (in the caller's names).
pub(crate) struct PendingDimEq {
    pub(crate) decl: Dim,
    pub(crate) arg: Dim,
    pub(crate) span: Span,
    /// The axis the extent sits on, or `None` when the equation is a whole element count (an
    /// array decaying to a rank-2+ tensor).
    pub(crate) axis: Option<usize>,
}

/// What one attempt at a [`PendingDimEq`] achieved.
enum Solve {
    /// Retired: checked, reported, or solved for a new binding.
    Progress,
    /// Needs a binding another equation may still supply.
    Open,
}

/// The `DimOp` for an AST operator, or `None` for one the dimension grammar does not contain.
fn dim_op(op: BinOp) -> Option<DimOp> {
    Some(match op {
        BinOp::Add => DimOp::Add,
        BinOp::Sub => DimOp::Sub,
        BinOp::Mul => DimOp::Mul,
        BinOp::Div => DimOp::Div,
        BinOp::Rem => DimOp::Rem,
        _ => return None,
    })
}

/// Every derived extent (`Dim::Expr`) in the tensor shapes inside `ty`.
fn collect_derived(ty: &Ty, out: &mut Vec<Dim>) {
    match ty {
        Ty::Tensor { shape, .. } => {
            out.extend(
                shape
                    .0
                    .iter()
                    .filter(|d| matches!(d, Dim::Expr(_)))
                    .cloned(),
            );
        }
        Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } | Ty::Slice(pointee) => {
            collect_derived(pointee, out)
        }
        Ty::Array { elem, .. } => collect_derived(elem, out),
        Ty::Tuple(items) => {
            for t in items {
                collect_derived(t, out);
            }
        }
        _ => {}
    }
}

/// Does the type expression name a scalar type directly (`f32`, `i64`)?
fn names_scalar(t: &TypeExpr, s: &Sema<'_>) -> Option<Scalar> {
    match &t.kind {
        TypeKind::Path(p) if p.is_single() => Scalar::from_name(s.sym_str(p.first().sym)),
        _ => None,
    }
}

impl Sema<'_> {
    /// Fold one dimension operator `l <op> r` (the operands already lowered) into normal form,
    /// reporting an expression that has no extent (`E0505`): a negative constant (`2 - 5`), a zero or
    /// negative divisor, an overflow. A `?` operand is reported by the caller (`lower_dim`), which can
    /// tell a written `?` from the `?` an inner error was poisoned to; here both are the silent `?`.
    pub(crate) fn fold_dim_binary(&mut self, op: BinOp, l: &Dim, r: &Dim, span: Span) -> Dim {
        let Some(dop) = dim_op(op) else {
            // The parser builds a dimension `Binary` only from `+ - * / %`.
            return Dim::Dynamic;
        };
        match Dim::binary(dop, l, r) {
            Ok(d) => d,
            Err(e) => {
                // Gated like E0504: during collection a `const` may not be registered yet and would
                // fold to a spurious 0 divisor. The body pass re-lowers every signature.
                if self.checking_bodies {
                    let (ls, rs) = (self.dim_str(l), self.dim_str(r));
                    self.error(
                        span,
                        "E0505",
                        format!(
                            "dimension `{ls} {} {rs}` has no extent: {}",
                            dop.glyph(),
                            e.describe()
                        ),
                    );
                }
                Dim::Dynamic
            }
        }
    }

    /// Report a `?` written as an operand of dimension arithmetic: it has no value to compute with,
    /// and silently making the whole extent `?` would throw away the relation the author wrote.
    pub(crate) fn report_dynamic_dim_operand(&mut self, span: Span, op: BinOp) {
        if self.checking_bodies {
            self.error(
                span,
                "E0505",
                format!(
                    "a `?` dimension cannot be an operand of `{}`: it has no compile-time value; \
                     name the dimension with a generic parameter to compute with it",
                    op.glyph()
                ),
            );
        }
    }

    /// Solve the derived-extent equations a call queued (`unify_dim`) once every parameter has
    /// bound what it binds directly. An equation whose dimensions are all bound is CHECKED; one
    /// with exactly one unbound dimension that appears linearly (`a*N + rest`, `a` constant) is
    /// SOLVED for it, which may enable another; anything else stays open, and the dimension stays
    /// unbound (the call's return type keeps it symbolic and lowering asks for a turbofish).
    pub(crate) fn solve_pending_dims(
        &mut self,
        generics: &[Symbol],
        dims: &mut HashMap<Symbol, Dim>,
        mut open: Vec<PendingDimEq>,
    ) {
        // Every productive round retires at least one equation, so this terminates.
        while !open.is_empty() {
            let mut progress = false;
            let mut next = Vec::with_capacity(open.len());
            for eq in open {
                match self.solve_dim_eq(generics, dims, &eq) {
                    Solve::Progress => progress = true,
                    Solve::Open => next.push(eq),
                }
            }
            open = next;
            if !progress {
                break;
            }
        }
    }

    fn solve_dim_eq(
        &mut self,
        generics: &[Symbol],
        dims: &mut HashMap<Symbol, Dim>,
        eq: &PendingDimEq,
    ) -> Solve {
        let unbound: Vec<Symbol> = eq
            .decl
            .vars()
            .into_iter()
            .filter(|s| generics.contains(s) && !dims.contains_key(s))
            .collect();
        let at = match eq.axis {
            Some(a) => format!("axis {a}"),
            None => "the element count".to_string(),
        };
        let decl_s = self.dim_str(&eq.decl);
        let arg_s = self.dim_str(&eq.arg);
        if unbound.is_empty() {
            match eq.decl.subst(&|s| dims.get(&s).cloned()) {
                Err(e) => self.error(
                    eq.span,
                    "E0505",
                    format!(
                        "dimension `{decl_s}` has no extent at this call: {}",
                        e.describe()
                    ),
                ),
                Ok(now) => {
                    if !dims_equal(&now, &eq.arg) {
                        let now_s = self.dim_str(&now);
                        self.error(
                            eq.span,
                            "E0502",
                            format!(
                                "dimension mismatch on {at}: expected `{decl_s}` = {now_s}, found \
                                 {arg_s}"
                            ),
                        );
                    }
                }
            }
            return Solve::Progress;
        }
        let [g] = unbound.as_slice() else {
            return Solve::Open;
        };
        let g = *g;
        let Some(poly) = eq.decl.to_poly() else {
            return Solve::Progress;
        };
        let Some((a, rest)) = poly.linear_in(g) else {
            return Solve::Open;
        };
        // A `?` argument extent carries no value: nothing to solve, and `?` stays lenient.
        let Some(arg) = eq.arg.to_poly() else {
            return Solve::Progress;
        };
        let target = rest
            .substitute(&|s| dims.get(&s).and_then(Dim::to_poly))
            .and_then(|rest_now| arg.sub(&rest_now));
        let target = match target {
            Ok(t) => t,
            Err(e) => {
                self.error(
                    eq.span,
                    "E0505",
                    format!(
                        "dimension `{decl_s}` has no extent at this call: {}",
                        e.describe()
                    ),
                );
                return Solve::Progress;
            }
        };
        let gname = self.sym_str(g).to_string();
        match target.exact_div(a) {
            Some(q) => match Dim::from_poly(q) {
                Ok(d) => {
                    dims.insert(g, d);
                }
                Err(_) => self.error(
                    eq.span,
                    "E0502",
                    format!(
                        "dimension mismatch on {at}: no extent `{gname}` makes `{decl_s}` equal \
                         {arg_s} (it would be negative)"
                    ),
                ),
            },
            None if target.as_constant().is_some() => self.error(
                eq.span,
                "E0502",
                format!(
                    "dimension mismatch on {at}: no whole-number extent `{gname}` makes \
                     `{decl_s}` equal {arg_s}"
                ),
            ),
            None => self.error(
                eq.span,
                "E0502",
                format!(
                    "dimension mismatch on {at}: cannot prove that {arg_s} has the form \
                     `{decl_s}` for every value of its dimensions (it is not provably a multiple of \
                     {a}); give the argument a shape that spells the factor, e.g. `{a}*K`"
                ),
            ),
        }
        Solve::Progress
    }

    /// After a call's dimensions are bound, every derived extent in `ty` (the call's result type)
    /// whose dimensions are all known must still be an extent: `-> Tensor[f32, N - 4]` called with
    /// `N = 2` has no result shape (`E0505`).
    pub(crate) fn check_derived_extents(
        &mut self,
        ty: &Ty,
        dims: &HashMap<Symbol, Dim>,
        span: Span,
    ) {
        let mut derived = Vec::new();
        collect_derived(ty, &mut derived);
        for d in derived {
            if !d.vars().iter().all(|s| dims.contains_key(s)) {
                continue;
            }
            if let Err(e) = d.subst(&|s| dims.get(&s).cloned()) {
                let ds = self.dim_str(&d);
                self.error(
                    span,
                    "E0505",
                    format!(
                        "dimension `{ds}` of this call's result has no extent: {}",
                        e.describe()
                    ),
                );
            }
        }
    }

    // ---- Views: a tensor value is not a scalar ----

    /// The diagnostic for a whole-tensor value used where a scalar is required. A tensor indexed
    /// with fewer indices than its rank is a VIEW of the remaining axes, so the commonest way to get
    /// here is a missing index — say so, with the type that was produced.
    pub(crate) fn whole_tensor_message(&self, t: &Ty, what: &str) -> String {
        format!(
            "{what} is not defined for a whole tensor (`{}`); index every axis to read an element \
             — an index list shorter than the rank selects a view of the remaining axes — and \
             operate on elements in a loop",
            t.display(self.interner)
        )
    }

    /// `E0401` when either operand of binary `op` is a whole tensor. Returns whether it reported.
    pub(crate) fn reject_tensor_operands(&mut self, l: &Ty, r: &Ty, op: BinOp, span: Span) -> bool {
        let Some(t) = [l, r].into_iter().find(|t| matches!(t, Ty::Tensor { .. })) else {
            return false;
        };
        let msg = self.whole_tensor_message(t, &format!("binary operator `{}`", op.glyph()));
        self.error(span, "E0401", msg);
        true
    }

    /// `E0401` when `t` is a whole tensor used as a scalar (a condition, a printed value, a unary
    /// operand, a math intrinsic argument). Returns whether it reported.
    pub(crate) fn reject_tensor_value(&mut self, t: &Ty, what: &str, span: Span) -> bool {
        if !matches!(t, Ty::Tensor { .. }) {
            return false;
        }
        let msg = self.whole_tensor_message(t, what);
        self.error(span, "E0401", msg);
        true
    }

    /// The assignment rules a view adds. A partial index `a[i] = …` names a sub-tensor, not an
    /// element: it is not a place a value can be stored into (a view is a handle on the caller's
    /// buffer, and replacing it would have to copy), so it is `E0507`. And a whole tensor cannot be
    /// stored into a scalar place, nor a scalar into a tensor-typed one (`E0401`) — the
    /// scalar/aggregate clash check does not look at tensors.
    pub(crate) fn check_tensor_assign(
        &mut self,
        target: &Expr,
        target_ty: &Ty,
        value_ty: &Ty,
        span: Span,
    ) {
        let indexes_a_tensor = match &target.kind {
            ExprKind::Index { base, .. } => {
                matches!(self.types.get(&base.id), Some(Ty::Tensor { .. }))
            }
            _ => false,
        };
        if indexes_a_tensor && matches!(target_ty, Ty::Tensor { .. }) {
            let t = target_ty.display(self.interner);
            self.error(
                target.span,
                "E0507",
                format!(
                    "cannot assign to a tensor view: this index list selects a `{t}` sub-tensor, \
                     not an element; assign its elements with a full index list (one index per \
                     axis) in a loop"
                ),
            );
            return;
        }
        let scalarish = |t: &Ty| matches!(t, Ty::Scalar(_) | Ty::Vector { .. });
        let tensor = |t: &Ty| matches!(t, Ty::Tensor { .. });
        if (scalarish(target_ty) && tensor(value_ty)) || (tensor(target_ty) && scalarish(value_ty))
        {
            self.error(
                span,
                "E0401",
                format!(
                    "type mismatch: cannot assign a value of type `{}` to a place of type `{}`",
                    value_ty.display(self.interner),
                    target_ty.display(self.interner)
                ),
            );
        }
    }

    /// The type of `base[indices]` for a tensor indexed with FEWER indices than its rank: the view
    /// of the remaining axes. Only a contiguous (row-major) tensor has one — fixing the leading
    /// axes of a row-major buffer leaves a contiguous sub-buffer, while the same rows of a
    /// `.col_major` / `.strided` / `.tiled` tensor are scattered (`E0507`).
    pub(crate) fn partial_index_type(
        &mut self,
        elem: Scalar,
        shape: &Shape,
        layout: &Layout,
        indices: usize,
        span: Span,
    ) -> Ty {
        if *layout != Layout::Contiguous {
            let dims = self.shape_str(shape);
            self.error(
                span,
                "E0507",
                format!(
                    "a tensor indexed with {indices} of its {} axes is a view of the remaining \
                     axes, which only a contiguous (row-major) tensor has; this `{dims}` tensor is \
                     not contiguous — index every axis",
                    shape.rank()
                ),
            );
            return Ty::Error;
        }
        Ty::Tensor {
            elem,
            shape: Shape(shape.0[indices..].to_vec()),
            layout: Layout::Contiguous,
        }
    }

    /// The non-`mut` aggregate PARAMETER the tensor-valued `e` views, if any: the parameter itself,
    /// a partial index or reshape of it, or a local recorded as a view of it.
    fn viewed_immutable_param(&self, e: &Expr) -> Option<Symbol> {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() => {
                let s = p.first().sym;
                if self.is_immutable_param(s) && self.param_is_aggregate(s) {
                    Some(s)
                } else {
                    self.tensor_view_root(s)
                }
            }
            ExprKind::Index { base, .. } | ExprKind::Cast { expr: base, .. } => {
                self.viewed_immutable_param(base)
            }
            _ => None,
        }
    }

    /// The non-`mut` parameter the local `name` is a view of, resolving `name` innermost-scope first
    /// like `is_immutable_local`, so a shadowing binding answers for itself.
    pub(crate) fn tensor_view_root(&self, name: Symbol) -> Option<Symbol> {
        for i in (0..self.scopes.len()).rev() {
            if self.scopes[i].contains_key(&name) {
                return self.tensor_views.get(i).and_then(|m| m.get(&name)).copied();
            }
        }
        None
    }

    /// Record what a new binding `pat: ty = init` views. A tensor local is a handle on someone
    /// else's buffer, so `let r = a[i]; r[j] = x;` writes into `a` — when `a` is a non-`mut`
    /// parameter that is the caller's buffer, which the `mut` rule exists to protect. Any other
    /// binding of the name clears the record (a shadowing `let` in the same scope).
    pub(crate) fn note_tensor_view(&mut self, pat: &Pattern, ty: &Ty, init: Option<&Expr>) {
        let PatKind::Ident(name) = &pat.kind else {
            return;
        };
        let root = match (ty, init) {
            (Ty::Tensor { .. }, Some(e)) => self.viewed_immutable_param(e),
            _ => None,
        };
        if let Some(scope) = self.tensor_views.last_mut() {
            match root {
                Some(r) => {
                    scope.insert(*name, r);
                }
                None => {
                    scope.remove(name);
                }
            }
        }
    }

    /// `E0304` for a write through `view`, a tensor local viewing the non-`mut` parameter `param`.
    pub(crate) fn report_write_through_view(&mut self, span: Span, view: Symbol, param: Symbol) {
        let (v, p) = (
            self.sym_str(view).to_string(),
            self.sym_str(param).to_string(),
        );
        self.error(
            span,
            "E0304",
            format!(
                "cannot mutate `{p}` through the view `{v}`: `{v}` names part of the non-`mut` \
                 parameter `{p}`, which is the caller's buffer — add `mut` to the parameter \
                 (`fn …(mut {p}: …)`) to allow in-place mutation"
            ),
        );
    }

    // ---- Reshape ----

    /// `x as Tensor[T, d0, …]`: a zero-copy reinterpretation of a contiguous buffer — a tensor, or a
    /// fixed-size array of scalars — under a new shape. The element type must be the same (`E0502`:
    /// reshape moves no data, so it cannot convert), both sides must be contiguous (`E0502`), and the
    /// ELEMENT COUNT must be provably unchanged (`E0506`): `S*H*HD` against `S*(H*HD)` is equal in
    /// normal form, `S*D` against `S*H*(D/H)` is not (floor division loses the remainder) and is
    /// rejected rather than assumed. A `?` extent has no count to prove, so it is rejected too.
    ///
    /// Returns `true` when the cast was judged here (`to` is a tensor and `from` a buffer), `false`
    /// to leave it to the ordinary cast rules.
    pub(crate) fn check_reshape(&mut self, from: &Ty, to: &Ty, span: Span) -> bool {
        let Ty::Tensor {
            elem: te,
            shape: ts,
            layout: tl,
        } = to
        else {
            return false;
        };
        if from == to {
            return true;
        }
        let (fe, fdims, src_contiguous): (Scalar, Vec<Dim>, bool) = match from {
            Ty::Tensor {
                elem,
                shape,
                layout,
            } => (*elem, shape.0.clone(), *layout == Layout::Contiguous),
            // A fixed-size array is a row-major buffer of `len` elements.
            Ty::Array { elem, len, .. } => match elem.as_ref() {
                Ty::Scalar(s) => (*s, vec![Dim::Const(*len)], true),
                _ => {
                    self.error(
                        span,
                        "E0401",
                        format!(
                            "invalid cast: only an array of scalars can be viewed as a tensor, \
                             not `{}`",
                            from.display(self.interner)
                        ),
                    );
                    return true;
                }
            },
            Ty::Unknown | Ty::Error => return true,
            _ => return false,
        };
        if !src_contiguous || *tl != Layout::Contiguous {
            self.error(
                span,
                "E0502",
                "a reshape reinterprets a contiguous (row-major) buffer under a new shape; both the \
                 source and the target must be contiguous"
                    .to_string(),
            );
            return true;
        }
        if fe != *te {
            self.error(
                span,
                "E0502",
                format!(
                    "a reshape cannot convert elements: the source holds `{}` and the target \
                     `{}`; convert elements in a loop",
                    fe.name(),
                    te.name()
                ),
            );
            return true;
        }
        let src = self.shape_str(&Shape(fdims.clone()));
        let dst = self.shape_str(ts);
        if fdims.iter().chain(&ts.0).any(|d| matches!(d, Dim::Dynamic)) {
            self.error(
                span,
                "E0506",
                format!(
                    "cannot reshape {src} to {dst}: a `?` dimension has no compile-time extent, so \
                     the element counts cannot be proved equal"
                ),
            );
            return true;
        }
        match (Dim::product(&fdims), Dim::product(&ts.0)) {
            (Ok(nf), Ok(nt)) if nf == nt => {}
            (Ok(nf), Ok(nt)) => {
                let (nf_s, nt_s) = (self.dim_str(&nf), self.dim_str(&nt));
                let why = if matches!((&nf, &nt), (Dim::Const(_), Dim::Const(_))) {
                    "a reshape must keep the element count"
                } else {
                    "the counts are not provably equal for every value of the dimensions"
                };
                self.error(
                    span,
                    "E0506",
                    format!(
                        "cannot reshape {src} ({nf_s} elements) to {dst} ({nt_s} elements): {why}"
                    ),
                );
            }
            _ => self.error(
                span,
                "E0506",
                format!("cannot reshape {src} to {dst}: an element count overflows"),
            ),
        }
        true
    }

    // ---- Extent queries ----

    /// `t.dim(k)` / `t.len()` on a value of type `bty`: the run-time extent of axis `k` (`len` is
    /// axis 0), an `i64`. Defined on a tensor (every axis — a fixed one folds to its constant, a
    /// symbolic or `?` one is read at run time), a fixed-size array (rank 1) and, for `len`, a slice.
    /// The axis must be a compile-time constant in range (`E0507` / `E0501`). `None` when `bty` has
    /// no such query, which leaves the lenient method path to the caller.
    pub(crate) fn type_extent_method(
        &mut self,
        bty: &Ty,
        is_dim: bool,
        args: &[Expr],
        span: Span,
    ) -> Option<Ty> {
        let rank = match bty {
            Ty::Tensor { shape, .. } => shape.rank(),
            Ty::Array { .. } => 1,
            Ty::Slice(_) if !is_dim && args.is_empty() => 1,
            _ => return None,
        };
        let what = match bty {
            Ty::Tensor { .. } => "tensor",
            Ty::Array { .. } => "array",
            _ => "slice",
        };
        if is_dim {
            if args.len() != 1 {
                self.error(
                    span,
                    "E0503",
                    format!(
                        "`.dim` takes exactly 1 argument (the axis), but {} were supplied",
                        args.len()
                    ),
                );
            } else {
                match self.eval_index_const(&args[0]) {
                    Some(k) if k >= 0 && (k as u64) < rank as u64 => {}
                    Some(k) => self.error(
                        args[0].span,
                        "E0501",
                        format!("axis {k} is out of range for a rank-{rank} {what}"),
                    ),
                    None => self.error(
                        args[0].span,
                        "E0507",
                        "the axis of `.dim(k)` must be a compile-time constant (an integer literal \
                         or a `const`): each axis of a shape-typed value has its own extent"
                            .to_string(),
                    ),
                }
            }
        } else if !args.is_empty() {
            self.error(
                span,
                "E0503",
                format!(
                    "`.len()` takes no arguments, but {} were supplied",
                    args.len()
                ),
            );
        } else if rank == 0 {
            self.error(
                span,
                "E0501",
                "`.len()` is the extent of axis 0, and a rank-0 tensor has no axes".to_string(),
            );
        }
        Some(Ty::Scalar(Scalar::I64))
    }

    // ---- Shape-generic array parameters ----

    /// The type of a function PARAMETER. Identical to `lower_type` except for a fixed-size array of
    /// scalars whose length is a dimension expression over the function's generics — `[f32; N]`,
    /// `[i32; 2*N]` — which is the shape-generic buffer it denotes, `Tensor[T, N]`: the same object
    /// at run time (a base pointer to `N` contiguous elements), indexed, iterated and queried
    /// (`.len()`) the same way, with `N` bound per call from the argument's length exactly like any
    /// other symbolic dimension. Previously the length was exempt from the name check and then
    /// evaluated to 0. A local array, a return type and a non-scalar element keep the fixed-length
    /// rule (`E0406` for a generic length): they would need storage whose size is a run-time value.
    pub(crate) fn lower_param_type(&mut self, t: &TypeExpr) -> Ty {
        if let TypeKind::Array {
            elem,
            len,
            shared: false,
        } = &t.kind
        {
            if let Some(sc) = names_scalar(elem, self) {
                if self.mentions_generic(len) {
                    if let Some(dim) = self.expr_dim(len) {
                        return Ty::Tensor {
                            elem: sc,
                            shape: Shape(vec![dim]),
                            layout: Layout::Contiguous,
                        };
                    }
                }
            }
        }
        self.lower_type(t)
    }

    fn mentions_generic(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() => self.generics.contains(&p.first().sym),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.mentions_generic(lhs) || self.mentions_generic(rhs)
            }
            _ => false,
        }
    }

    /// An array-length expression as a dimension: an integer literal, a generic, a foldable `const`
    /// (the same fold `array_len` uses, so the two cannot disagree), or `+ - * / %` on those. `None`
    /// for anything else, including arithmetic with no extent, which then reaches `lower_type` and
    /// its existing `E0406`.
    fn expr_dim(&mut self, e: &Expr) -> Option<Dim> {
        match &e.kind {
            ExprKind::Int(s) => crate::parse_u64_text(self.sym_str(*s)).map(Dim::Const),
            ExprKind::Path(p) if p.is_single() => {
                let sym = p.first().sym;
                if self.generics.contains(&sym) {
                    Some(Dim::Var(sym))
                } else if self.array_len_is_const(e, 0, false) {
                    Some(Dim::Const(self.eval_usize(e)))
                } else {
                    None
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let dop = dim_op(*op)?;
                let l = self.expr_dim(lhs)?;
                let r = self.expr_dim(rhs)?;
                Dim::binary(dop, &l, &r).ok()
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::check;
    use wukong_diag::Diagnostic;
    use wukong_span::{Interner, SourceId};

    fn diags(src: &str) -> Vec<Diagnostic> {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        check(&module, &interner).1
    }

    fn codes(src: &str) -> Vec<&'static str> {
        diags(src).into_iter().filter_map(|d| d.code).collect()
    }

    #[track_caller]
    fn ok(src: &str) {
        let d = diags(src);
        assert!(d.is_empty(), "unexpected diagnostics for {src:?}: {d:?}");
    }

    #[track_caller]
    fn rejects(src: &str, code: &str) {
        let c = codes(src);
        assert!(c.contains(&code), "expected {code} for {src:?}, got {c:?}");
    }

    #[test]
    fn derived_extents_compare_by_normal_form() {
        // `2*N` and `N + N` are one type, so a body may spell its return either way...
        ok("fn twice<N>(a: Tensor[f32, 2*N]) -> Tensor[f32, N + N] { return a; }");
        ok("fn qkv<D>(a: Tensor[f32, 3*D]) -> Tensor[f32, D + D + D] { return a; }");
        // ...but not as a different extent: the rigid body check still refuses a lie.
        rejects(
            "fn lie<N>(a: Tensor[f32, 2*N]) -> Tensor[f32, N + 1] { return a; }",
            "E0502",
        );
        // A `const` inside a dimension expression folds to its value.
        ok("const H: usize = 4; fn f(a: Tensor[f32, 2*H]) -> Tensor[f32, 8] { return a; }");
    }

    #[test]
    fn a_call_solves_a_derived_extent_for_its_dimension() {
        // `2*N = 8` binds `N = 4` from a decaying array; the bound `N` then checks a sibling.
        ok(
            "fn f<N>(a: Tensor[f32, 2*N], b: Tensor[f32, N]) -> f32 { return a[0] + b[0]; } \
            fn g() -> f32 { let x: [f32; 8] = [0.0; 8]; let y: [f32; 4] = [0.0; 4]; \
            return f(x, y); }",
        );
        rejects(
            "fn f<N>(a: Tensor[f32, 2*N], b: Tensor[f32, N]) -> f32 { return a[0] + b[0]; } \
             fn g() -> f32 { let x: [f32; 8] = [0.0; 8]; let y: [f32; 3] = [0.0; 3]; \
             return f(x, y); }",
            "E0502",
        );
        // No whole-number `N` makes `2*N` equal 7.
        rejects(
            "fn f<N>(a: Tensor[f32, 2*N]) -> f32 { return a[0]; } \
             fn g() -> f32 { let x: [f32; 7] = [0.0; 7]; return f(x); }",
            "E0502",
        );
        // Fused QKV: `S` from axis 0, then `3*D = 6` gives `D = 2`, which `q`'s shape must match.
        ok(
            "fn split<S, D>(qkv: Tensor[f32, S, 3*D], q: Tensor[f32, S, D]) -> f32 { \
            return qkv[0, 0] + q[0, 0]; } \
            fn g(x: Tensor[f32, 2, 6], q: Tensor[f32, 2, 2]) -> f32 { return split(x, q); }",
        );
        rejects(
            "fn split<S, D>(qkv: Tensor[f32, S, 3*D], q: Tensor[f32, S, D]) -> f32 { \
             return qkv[0, 0] + q[0, 0]; } \
             fn g(x: Tensor[f32, 2, 6], q: Tensor[f32, 2, 3]) -> f32 { return split(x, q); }",
            "E0502",
        );
        // A caller's own generic extent that is not provably a multiple of 2.
        rejects(
            "fn f<N>(a: Tensor[f32, 2*N]) -> f32 { return a[0]; } \
             fn g<M>(x: Tensor[f32, M]) -> f32 { return f(x); }",
            "E0502",
        );
        // ...and one that is.
        ok("fn f<N>(a: Tensor[f32, 2*N]) -> f32 { return a[0]; } \
            fn g<M>(x: Tensor[f32, 2*M]) -> f32 { return f(x); }");
    }

    #[test]
    fn a_dimension_expression_must_be_an_extent() {
        rejects("fn f(a: Tensor[f32, 2 - 5]) {}", "E0505");
        rejects("fn f<N>(a: Tensor[f32, N / 0]) {}", "E0505");
        rejects("fn f<N>(a: Tensor[f32, ? * N]) {}", "E0505");
        // At a call: `N - 4` with `N = 2` has no extent.
        rejects(
            "extern { fn shrink<N>(a: Tensor[f32, N]) -> Tensor[f32, N - 4]; } \
             fn g() { let x: [f32; 2] = [0.0; 2]; let y = shrink(x); }",
            "E0505",
        );
    }

    #[test]
    fn a_partial_index_is_a_view_of_the_remaining_axes() {
        ok("fn f(a: Tensor[f32, 4, 3]) -> f32 { let r = a[1]; return r[2]; }");
        ok("fn row<M, N>(a: Tensor[f32, M, N], i: i64) -> Tensor[f32, N] { return a[i]; }");
        ok("fn f(a: Tensor[f32, 4, 3]) -> f32 { return a[1][2]; }");
        ok("fn sum<N>(r: Tensor[f32, N]) -> f32 { return r[0]; } \
            fn f(a: Tensor[f32, 4, 3]) -> f32 { return sum(a[1]); }");
        // A view is never a scalar: each use as one is a type error naming the view.
        for bad in [
            "fn f(a: Tensor[f32, 4, 3]) -> f32 { return a[1] + 1.0; }",
            "fn f(a: Tensor[f32, 4, 3]) -> f32 { let x: f32 = a[1]; return x; }",
            "fn f(a: Tensor[f32, 4, 3]) { print(a[1]); }",
            "fn f(a: Tensor[f32, 4, 3]) -> f32 { return exp(a[1]); }",
            "fn f(a: Tensor[f32, 4, 3]) { if a[1] { } }",
            "fn f(a: Tensor[f32, 4, 3]) -> bool { return a[1] < a[2]; }",
            "fn f(a: Tensor[f32, 4, 3]) -> f32 { return -a[1]; }",
            "fn f(a: Tensor[f32, 4, 3]) { let mut x: f32 = 0.0; x = a[1]; }",
        ] {
            rejects(bad, "E0401");
        }
        // The leading indices are still bounds-checked.
        rejects(
            "fn f(a: Tensor[f32, 4, 3]) -> f32 { return a[4][0]; }",
            "E0501",
        );
        // Only a contiguous tensor has row views, and a view is not an assignable place.
        rejects(
            "fn f(a: Tensor[f32, 4, 3, .col_major]) -> f32 { let r = a[1]; return r[0]; }",
            "E0507",
        );
        rejects(
            "fn f(mut a: Tensor[f32, 4, 3], b: Tensor[f32, 4, 3]) { a[1] = b[1]; }",
            "E0507",
        );
    }

    #[test]
    fn a_view_of_a_non_mut_parameter_is_read_only() {
        rejects(
            "fn f(a: Tensor[f32, 4, 3]) { let r = a[1]; r[0] = 1.0; }",
            "E0304",
        );
        rejects(
            "fn f(a: Tensor[f32, 4, 3]) { for row in a { row[0] = 1.0; } }",
            "E0304",
        );
        rejects(
            "fn f(a: Tensor[f32, 6]) { let m = a as Tensor[f32, 2, 3]; m[0, 0] = 1.0; }",
            "E0304",
        );
        ok("fn f(mut a: Tensor[f32, 4, 3]) { let r = a[1]; r[0] = 1.0; }");
        ok("fn f(mut a: Tensor[f32, 4, 3]) { for row in a { row[0] = 1.0; } }");
    }

    #[test]
    fn reshape_must_provably_keep_the_element_count() {
        ok("fn f<S, H, HD>(x: Tensor[f32, S, H*HD]) -> f32 { \
            let h = x as Tensor[f32, S, H, HD]; return h[0, 0, 0]; }");
        ok("fn f(x: [f32; 6]) -> f32 { let m = x as Tensor[f32, 2, 3]; return m[1, 2]; }");
        // `D` is not provably `H*(D/H)`: floor division drops the remainder.
        rejects(
            "fn f<S, D, H>(x: Tensor[f32, S, D]) -> f32 { \
             let h = x as Tensor[f32, S, H, D/H]; return h[0, 0, 0]; }",
            "E0506",
        );
        rejects(
            "fn f(x: [f32; 6]) -> f32 { let m = x as Tensor[f32, 4, 2]; return m[0, 0]; }",
            "E0506",
        );
        rejects(
            "fn f(x: Tensor[f32, ?]) -> f32 { let m = x as Tensor[f32, 2, 3]; return m[0, 0]; }",
            "E0506",
        );
        // Reshape moves no data, so it cannot convert the element type.
        rejects(
            "fn f(x: Tensor[f32, 6]) -> i32 { let m = x as Tensor[i32, 6]; return m[0]; }",
            "E0502",
        );
    }

    #[test]
    fn extent_queries_are_typed_and_checked() {
        ok("fn f(a: Tensor[f32, ?, 4]) -> i64 { return a.dim(0) + a.dim(1) + a.len(); }");
        ok("fn f(a: [i32; 5]) -> i64 { return a.len() + a.dim(0); }");
        ok("fn f<N>(a: [i32; N]) -> i64 { return a.len(); }");
        rejects(
            "fn f(a: Tensor[f32, ?, 4]) -> i64 { return a.dim(2); }",
            "E0501",
        );
        rejects(
            "fn f(a: Tensor[f32, ?, 4], k: i64) -> i64 { return a.dim(k); }",
            "E0507",
        );
        rejects(
            "fn f(a: Tensor[f32, ?, 4]) -> i64 { return a.dim(); }",
            "E0503",
        );
        // `for` over a tensor: elements of a rank-1 tensor, row views of a rank-2 one.
        ok("fn f(a: Tensor[f32, ?]) -> f32 { let mut s: f32 = 0.0; \
            for x in a { s = s + x; } return s; }");
        ok("fn f(a: Tensor[f32, 2, 3]) -> f32 { let mut s: f32 = 0.0; \
            for row in a { s = s + row[0]; } return s; }");
    }

    #[test]
    fn a_fixed_array_parameter_rejects_a_tensor_of_another_size() {
        rejects(
            "fn take4(a: [f32; 4]) -> f32 { return a[3]; } \
             fn f(t: Tensor[f32, 8]) -> f32 { return take4(t); }",
            "E0501",
        );
        rejects(
            "fn take4(a: [f32; 4]) -> f32 { return a[3]; } \
             fn f<N>(a: [f32; N]) -> f32 { return take4(a); }",
            "E0502",
        );
        ok("fn take4(a: [f32; 4]) -> f32 { return a[3]; } \
            fn f(t: Tensor[f32, 2, 2]) -> f32 { return take4(t); }");
    }
}
