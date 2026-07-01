//! Shape checking — the part of sema that makes tensor dimensions a compile-time guarantee.
//!
//! Two checks live here:
//!  * **Call unification**: when a (user-defined) generic function is called, its declared
//!    parameter shapes are unified against the argument shapes. Dimension variables (`M`, `N`,
//!    `K`) are bound either from an explicit turbofish or inferred from the arguments; a
//!    conflicting binding is a `E0502` dimension mismatch and a differing rank is `E0501`.
//!  * **Index rank**: indexing a tensor with the wrong number of indices is a `E0501`.

use std::collections::HashMap;

use mercury_ast::{Expr, ExprKind, TypeExpr, TypeKind};
use mercury_span::{Span, Symbol};
use mercury_types::{Dim, Scalar, Shape, Ty};

use crate::{DefKind, FnSig, Sema};

/// Result types for the math intrinsics the backends lower directly (`sqrt`, `rsqrt`, `exp`, `log`,
/// `fmax`, `fmin`). The result is the float type of the first argument, defaulting to `f32` so a
/// bare `exp(x)` is still typed when the argument's type is unknown. Returns `None` for any other
/// callee — that keeps `type_call` lenient on the unmodeled-builtin path (`min`, `f32x8::load`, …).
fn intrinsic_ret_ty(name: &str, args: &[Ty]) -> Option<Ty> {
    let float_ty = match args.first() {
        Some(Ty::Scalar(s)) if s.is_float() => Ty::Scalar(*s),
        Some(Ty::Vector { elem, lanes }) if elem.is_float() => Ty::Vector {
            elem: *elem,
            lanes: *lanes,
        },
        _ => Ty::Scalar(Scalar::F32),
    };
    // `abs`/`round`/`floor`/`ceil`/`trunc` preserve the argument's type, including integers
    // (`abs(-5): i32`, an everyday operation); for an integer they lower to an integer abs / the
    // identity in mir_build. All the other intrinsics inherently produce a float.
    let preserve_ty = match args.first() {
        Some(t @ (Ty::Scalar(_) | Ty::Vector { .. })) => t.clone(),
        _ => Ty::Scalar(Scalar::F32),
    };
    match name {
        "abs" | "round" | "floor" | "ceil" | "trunc" => Some(preserve_ty),
        "sqrt" | "rsqrt" | "exp" | "log" | "pow" | "exp2" | "log2" | "exp10" | "log10" | "expm1"
        | "log1p" | "sinh" | "cosh" | "asinh" | "acosh" | "atanh" | "atan" | "tan" | "asin"
        | "acos" | "atan2" | "hypot" | "cbrt" | "erf" | "sin" | "cos" | "tanh" | "sigmoid"
        | "silu" | "gelu" | "silu_backward" | "gelu_backward" | "sigmoid_backward"
        | "tanh_backward" | "elu_backward" | "softplus_backward" | "elu" | "leaky_relu"
        | "softplus" | "mish" | "selu" | "tanhshrink" | "hardsigmoid" | "hardswish" | "softsign"
        | "logsigmoid" | "fmax" | "fmin" => Some(float_ty),
        _ => None,
    }
}

/// A short human name for a type's *kind*, for cross-kind mismatch diagnostics (`unify`).
fn kind_name(t: &Ty) -> &'static str {
    match t {
        Ty::Tensor { .. } => "a tensor",
        Ty::Vector { .. } => "a vector",
        Ty::Scalar(_) => "a scalar",
        Ty::Array { .. } => "an array",
        Ty::Ptr { .. } => "a pointer",
        Ty::Ref { .. } => "a reference",
        Ty::Tuple(_) => "a tuple",
        _ => "a different type",
    }
}

impl Sema<'_> {
    pub(crate) fn type_call(
        &mut self,
        callee: &Expr,
        generic_args: &[TypeExpr],
        args: &[Expr],
        span: Span,
    ) -> Ty {
        let mut arg_tys: Vec<Ty> = args.iter().map(|a| self.type_expr(a)).collect();

        if let ExprKind::Path(p) = &callee.kind {
            if p.is_single() {
                let name = p.first().sym;
                if let Some(def) = self.defs.lookup(name) {
                    if let DefKind::Fn(sig) = &def.kind {
                        let sig = sig.clone();
                        // Adapt an unsuffixed numeric literal argument to a concrete scalar parameter
                        // — the same `{integer}`/`{float}` inference a `let` annotation gets — so
                        // `process(10)` where the parameter is `usize`/`i64`/`f64` type-checks instead
                        // of erroring on the `i32`/`f32` literal default. Only concrete scalar params
                        // (not generics/tensors) and only adaptable literals are touched.
                        for (i, param) in sig.params.iter().enumerate() {
                            if matches!(param, Ty::Scalar(_)) {
                                if let Some(arg) = args.get(i) {
                                    if self.literal_adapts(param, arg) {
                                        self.retype_adapted_literal(arg, param);
                                        self.range_check_int_literal(arg, param);
                                        arg_tys[i] = param.clone();
                                    }
                                }
                            }
                        }
                        self.types.insert(
                            callee.id,
                            Ty::Fn {
                                params: sig.params.clone(),
                                ret: Box::new(sig.ret.clone()),
                            },
                        );
                        return self.check_fn_call(&sig, generic_args, &arg_tys, span);
                    }
                }
                // A few math builtins are lowered directly by the backends, so their *result*
                // type is real rather than `Unknown` — this lets `let e = exp(x)` infer `f32`
                // and the lowerer pick the right result type. The callee path itself stays
                // unmodeled (`Unknown`).
                if let Some(ret) = intrinsic_ret_ty(self.sym_str(name), &arg_tys) {
                    self.types.insert(callee.id, Ty::Unknown);
                    // A math intrinsic operates on a scalar (an integer is coerced to float, like
                    // `sqrt(2)`), a SIMD vector, or a tensor — never an aggregate, pointer, reference,
                    // unit, or function value. Such an argument used to slip through (the result is
                    // still typed `f32`), then mir_build emitted e.g. `sqrt` on the struct's / the
                    // pointer's value — invalid MIR the verifier and Cranelift reject, while the
                    // interpreter ran it on the raw value and silently produced garbage: a backend
                    // divergence on accepted input. Reject it here. (Tensors are not aggregates by
                    // `is_aggregate_ty`, so tensor intrinsics stay lenient, as do `Unknown` args.)
                    if let Some(bad) = arg_tys.iter().find(|t| {
                        matches!(t, Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Unit | Ty::Fn { .. })
                            || self.is_aggregate_ty(t)
                    }) {
                        self.error(
                            span,
                            "E0401",
                            format!(
                                "`{}` expects a scalar, vector, or tensor argument, not a value of \
                                 type `{}`",
                                self.sym_str(name),
                                bad.display(self.interner)
                            ),
                        );
                        return Ty::Unknown;
                    }
                    return ret;
                }
                // `assert(cond)` is a builtin taking exactly one condition argument. With the wrong
                // arity it used to fall through to a malformed void call: the interpreter read the
                // missing condition as false and trapped (exit 1) while native treated it as a no-op
                // (exit 0) — a backend divergence. Pin the arity here (E0503), like a user fn's
                // arg-count check; this fails compilation before either backend runs.
                if self.sym_str(name) == "assert" && args.len() != 1 {
                    self.error(
                        span,
                        "E0503",
                        format!(
                            "`assert` takes exactly 1 argument, but {} were supplied",
                            args.len()
                        ),
                    );
                    self.types.insert(callee.id, Ty::Unknown);
                    return Ty::Unknown;
                }
                // Unresolved or non-function callee: lenient (builtins like `min`).
                self.types.insert(callee.id, Ty::Unknown);
                return Ty::Unknown;
            }
        }

        // Field/method or complex callee: type it (so its base is recorded) and stay lenient.
        self.type_expr(callee);
        Ty::Unknown
    }

    fn check_fn_call(
        &mut self,
        sig: &FnSig,
        generic_args: &[TypeExpr],
        arg_tys: &[Ty],
        span: Span,
    ) -> Ty {
        if arg_tys.len() != sig.params.len() {
            self.error(
                span,
                "E0503",
                format!(
                    "this function takes {} argument(s) but {} were supplied",
                    sig.params.len(),
                    arg_tys.len()
                ),
            );
            return sig.ret.clone();
        }

        let mut dims: HashMap<Symbol, Dim> = HashMap::new();
        let mut tys: HashMap<Symbol, Ty> = HashMap::new();

        if !generic_args.is_empty() {
            if generic_args.len() != sig.generics.len() {
                self.error(
                    span,
                    "E0503",
                    format!(
                        "this function takes {} generic argument(s) but {} were supplied",
                        sig.generics.len(),
                        generic_args.len()
                    ),
                );
                // Don't continue into the misaligned `zip` below, which would bind dims off-by-one
                // and report a spurious secondary E0502.
                return sig.ret.clone();
            }
            for (g, ga) in sig.generics.iter().zip(generic_args) {
                match &ga.kind {
                    TypeKind::Int(s) => {
                        dims.insert(*g, Dim::Const(parse_dim_text(self.sym_str(*s))));
                    }
                    TypeKind::Path(p) if p.is_single() => {
                        let nm = p.first().sym;
                        match Scalar::from_name(self.sym_str(nm)) {
                            Some(sc) => {
                                tys.insert(*g, Ty::Scalar(sc));
                            }
                            None => {
                                dims.insert(*g, Dim::Var(nm));
                            }
                        }
                    }
                    _ => {
                        let t = self.lower_type(ga);
                        tys.insert(*g, t);
                    }
                }
            }
        }

        for (param, arg) in sig.params.iter().zip(arg_tys) {
            let p = apply_subst(param, &dims, &tys);
            // Call site: the callee's not-yet-substituted dim vars are INFERENCE variables to bind
            // from the argument shapes (`rigid == false`).
            self.unify(&p, arg, &mut dims, span, false);
        }

        apply_subst(&sig.ret, &dims, &tys)
    }

    pub(crate) fn unify(
        &mut self,
        param: &Ty,
        arg: &Ty,
        dims: &mut HashMap<Symbol, Dim>,
        span: Span,
        rigid: bool,
    ) {
        if arg.is_unknown() || arg.is_error() || param.is_unknown() || param.is_error() {
            return;
        }
        match (param, arg) {
            (
                Ty::Tensor {
                    elem: pe,
                    shape: ps,
                    ..
                },
                Ty::Tensor {
                    elem: ae,
                    shape: as_,
                    ..
                },
            ) => {
                if pe != ae {
                    self.error(
                        span,
                        "E0502",
                        format!(
                            "tensor element type mismatch: expected `{}`, found `{}`",
                            pe.name(),
                            ae.name()
                        ),
                    );
                }
                if ps.rank() != as_.rank() {
                    self.error(
                        span,
                        "E0501",
                        format!(
                            "tensor rank mismatch: expected rank {}, found rank {}",
                            ps.rank(),
                            as_.rank()
                        ),
                    );
                    return;
                }
                for (pd, ad) in ps.0.iter().zip(&as_.0) {
                    self.unify_dim(*pd, *ad, dims, span, rigid);
                }
            }
            // Array → tensor *decay*: an `Array` argument is intentionally accepted for a tensor
            // parameter (see `tests/run/tensor_add.mer`, which feeds a `[f32; 6]` to a
            // `Tensor[f32, 2, 3]`). But when the tensor's shape and the array length are *both*
            // statically known, the buffer must hold the right number of elements of the right
            // type — otherwise a too-small/wrong array satisfies any tensor and every
            // in-tensor-bounds index becomes an out-of-bounds read (the interpreter traps, native
            // codegen does not: a backend divergence). A symbolic / dynamic dim stays lenient
            // (there is no concrete element count to check against).
            (Ty::Tensor { elem: pe, shape, .. }, Ty::Array { elem: ae, len }) => {
                if let Ty::Scalar(ae) = ae.as_ref() {
                    if pe != ae {
                        self.error(
                            span,
                            "E0502",
                            format!(
                                "tensor element type mismatch: expected `{}`, found `{}`",
                                pe.name(),
                                ae.name()
                            ),
                        );
                    }
                }
                // A rank-1 tensor parameter with a SYMBOLIC dim binds that dim to the array's
                // length, so a later sibling parameter `Tensor[f32, N]` fed a different-length array
                // conflicts. The dim was previously left unbound: two `Tensor[f32, N]` params
                // silently accepted arrays of different lengths, and indexing past the shorter one
                // diverged (interpreter traps OOB, native reads past the buffer). A multi-dim tensor
                // stays on the total-element-count check below (an array length can't be split into
                // symbolic factors).
                if shape.0.len() == 1 {
                    if let Dim::Var(_) = shape.0[0] {
                        self.unify_dim(shape.0[0], Dim::Const(*len), dims, span, rigid);
                    }
                }
                // The array must supply a whole number of the tensor's *known* (const) sub-slabs.
                // With every dim const this is the exact element count (`len == const_prod`); with a
                // symbolic/dynamic dim still free, the length must at least be divisible by the
                // product of the const dims — otherwise NO integer value of the free dim(s) could
                // ever produce this buffer, yet an in-shape multi-index still flattens past the end.
                // Gating the check on *all* dims being const (the previous behavior) left that hole
                // wide open: `Tensor[f32, ?, 4]` or `Tensor[f32, N, 4]` accepted a length-6 array and
                // `a[1, 3]` read/wrote element 7 — the interpreter traps, native codegen does not (a
                // backend divergence AND a memory-safety violation). The product-of-const check is
                // also parameter-order-independent: a const inner dim rejects the bad length whether
                // or not the sibling that binds the symbolic outer dim is unified first.
                let mut const_prod: u64 = 1;
                let mut has_free_dim = false;
                for d in &shape.0 {
                    match d {
                        Dim::Const(n) => const_prod = const_prod.saturating_mul(*n),
                        _ => has_free_dim = true,
                    }
                }
                let bad = if has_free_dim {
                    // `> 1` avoids a modulo-by-zero on a degenerate zero-size const dim and is a
                    // no-op when no const dim constrains the length (product 1).
                    const_prod > 1 && *len % const_prod != 0
                } else {
                    *len != const_prod
                };
                if bad {
                    let msg = if has_free_dim {
                        format!(
                            "array of length {len} is not a multiple of the tensor's known \
                             dimensions (product {const_prod})"
                        )
                    } else {
                        format!(
                            "array of length {len} cannot satisfy a tensor of {const_prod} element{}",
                            if const_prod == 1 { "" } else { "s" }
                        )
                    };
                    self.error(span, "E0501", msg);
                }
            }
            (Ty::Ptr { pointee: pp, .. }, Ty::Ptr { pointee: ap, .. })
            | (Ty::Ref { pointee: pp, .. }, Ty::Ref { pointee: ap, .. }) => {
                self.unify(pp, ap, dims, span, rigid)
            }
            (
                Ty::Vector {
                    elem: pe,
                    lanes: pl,
                },
                Ty::Vector {
                    elem: ae,
                    lanes: al,
                },
            ) => {
                if pe != ae || pl != al {
                    self.error(
                        span,
                        "E0401",
                        format!(
                            "vector type mismatch: expected `{}x{}`, found `{}x{}`",
                            pe.name(),
                            pl,
                            ae.name(),
                            al
                        ),
                    );
                }
            }
            (Ty::Scalar(a), Ty::Scalar(b)) => {
                if a != b {
                    self.error(
                        span,
                        "E0401",
                        format!(
                            "type mismatch: expected `{}`, found `{}`",
                            a.name(),
                            b.name()
                        ),
                    );
                }
            }
            // A scalar can't satisfy a tensor/vector parameter, nor a tensor a vector (or vice
            // versa): a kind confusion the call-unification core otherwise let fall through. Array →
            // tensor *decay* stays lenient (an `Array` argument is intentionally accepted for a
            // tensor parameter — see `tests/run/tensor_add.mer`), so only Scalar/Vector/Tensor kind
            // clashes are reported here.
            (Ty::Tensor { .. }, Ty::Scalar(_) | Ty::Vector { .. })
            | (Ty::Scalar(_) | Ty::Vector { .. }, Ty::Tensor { .. }) => {
                self.error(
                    span,
                    "E0501",
                    format!(
                        "type mismatch: expected {}, found {}",
                        kind_name(param),
                        kind_name(arg)
                    ),
                );
            }
            (Ty::Vector { .. }, Ty::Scalar(_)) | (Ty::Scalar(_), Ty::Vector { .. }) => {
                self.error(
                    span,
                    "E0401",
                    format!(
                        "type mismatch: expected {}, found {}",
                        kind_name(param),
                        kind_name(arg)
                    ),
                );
            }
            // A pointer/reference, array, or tuple versus a scalar/vector are different ABI kinds.
            // Passing one where the other is expected slips past the lenient unification core and
            // then either traps the interpreter or ICEs the native backend (e.g. an i64 pointer
            // marshalled into a 32-bit scalar slot) — a backend divergence on a program that should
            // never have compiled. Pointers/arrays/tuples here carry concrete element types (a bare
            // generic parameter is `Named`, handled below), so reporting the clash cannot
            // false-positive on a generic. Array→tensor decay is matched earlier and unaffected.
            (
                Ty::Scalar(_) | Ty::Vector { .. },
                Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Array { .. } | Ty::Tuple(_),
            )
            | (
                Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Array { .. } | Ty::Tuple(_),
                Ty::Scalar(_) | Ty::Vector { .. },
            ) => {
                self.error(
                    span,
                    "E0401",
                    format!(
                        "type mismatch: expected {}, found {}",
                        kind_name(param),
                        kind_name(arg)
                    ),
                );
            }
            // Named generic type variable, or anything else: stay lenient.
            _ => {}
        }
    }

    fn unify_dim(
        &mut self,
        pd: Dim,
        ad: Dim,
        dims: &mut HashMap<Symbol, Dim>,
        span: Span,
        rigid: bool,
    ) {
        // RIGID mode — used by the body shape checks (return type, operator / branch-arm operands).
        // There both shapes are FULLY DETERMINED by the signature and the body, so there is nothing
        // to infer: a function's own generic dims are universally quantified and must match by
        // IDENTITY (`N` matches only `N`), never bind. Without this a generic function could "unify"
        // its declared return shape against a differently-shaped body value — binding its own `N := M`
        // (two distinct generics), or silently accepting `Const(5)` against an unbound `Var(N)` — i.e.
        // lie about its output shape. A turbofished caller then propagates that bogus shape into an
        // in-type-bounds-but-real-out-of-bounds index: the interpreter traps while native reads past
        // the buffer, a backend divergence on a program that should never have compiled. Call-site
        // unification stays `rigid == false` and keeps inferring (a callee's dim var binds from the
        // argument shapes). `dims_equal` already implements the rigid relation (`Var(x)==Var(y)` iff
        // same symbol; `Const` vs `Var` = false; `Dynamic` matches anything, so `?` stays lenient).
        if rigid {
            if !dims_equal(pd, ad) {
                self.error(
                    span,
                    "E0502",
                    format!(
                        "dimension mismatch: expected {}, found {}",
                        self.dim_str(pd),
                        self.dim_str(ad)
                    ),
                );
            }
            return;
        }
        match pd {
            Dim::Var(v) => match dims.get(&v) {
                Some(bound) => {
                    if !dims_equal(*bound, ad) {
                        self.error(
                            span,
                            "E0502",
                            format!(
                                "dimension `{}` was inferred as {} but is {} here",
                                self.sym_str(v),
                                self.dim_str(*bound),
                                self.dim_str(ad)
                            ),
                        );
                    }
                }
                None => {
                    // Don't bind a symbolic dim to `?` (Dynamic): a deferred dim carries no value,
                    // so binding `N := ?` would make every later concrete sibling compare equal
                    // (`dims_equal` treats `?` as matching anything), masking a real conflict. Leave
                    // `N` unbound so a concrete sibling binds it and subsequent ones are checked.
                    if !matches!(ad, Dim::Dynamic) {
                        dims.insert(v, ad);
                    }
                }
            },
            Dim::Const(pc) => {
                if let Dim::Const(ac) = ad {
                    if pc != ac {
                        self.error(
                            span,
                            "E0502",
                            format!("dimension mismatch: expected {pc}, found {ac}"),
                        );
                    }
                }
            }
            Dim::Dynamic => {}
        }
    }

    /// Evaluate a compile-time index: a literal (or negated literal) directly, or a top-level `const`
    /// name resolved to its recorded initializer (recursing for a const-references-const). Without the
    /// const resolution, `xs[I]` for `const I = 99` slipped past the bounds check — sema accepted it,
    /// then mir_build inlined the const and the index went out of bounds (interp traps, native reads
    /// past the buffer: a backend divergence on a program that should never have compiled). The depth
    /// cap guards against a recursive const (separately reported as E0403) looping here.
    fn eval_index_const(&self, e: &Expr) -> Option<i64> {
        self.eval_index_const_depth(e, 0)
    }

    fn eval_index_const_depth(&self, e: &Expr, depth: u32) -> Option<i64> {
        if depth > 32 {
            return None;
        }
        if let ExprKind::Path(p) = &e.kind {
            if p.is_single() {
                if let Some(init) = self.consts.get(&p.first().sym) {
                    return self.eval_index_const_depth(init, depth + 1);
                }
            }
        }
        crate::eval_const_int(e, self.interner)
    }

    pub(crate) fn type_index(&mut self, base: &Expr, indices: &[Expr], span: Span) -> Ty {
        let base_ty = self.type_expr(base);
        for ix in indices {
            let ix_ty = self.type_expr(ix);
            // An index is an integer offset; `bool` would silently coerce to 0/1 (`a[true]` reads
            // `a[1]`). Reject a concrete bool index — `Unknown`/`Error` stays lenient.
            if matches!(ix_ty, Ty::Scalar(Scalar::Bool)) {
                self.error(
                    ix.span,
                    "E0401",
                    "array index must be an integer, not `bool`".to_string(),
                );
            }
        }
        // Indexing `base[i]` requires an indexable base. A *definitely* non-indexable base — a scalar
        // (`x[0]` on `x: i32`), a struct (`s[1]`), or a tuple (`t[1]`) — is a type error, not an
        // unmodeled construct: it type-checked through the lenient fallthrough below, then mir_build
        // GEPed off a non-pointer, which the verifier/Cranelift reject (an ICE for a scalar) or which
        // the two backends lower divergently (a by-pointer tuple/struct base: interp 0, native the
        // real element). Reject those three; an `Unknown`/`Ref`/other base stays lenient (it may be an
        // indexable construct sema does not yet model, and `Unknown` unifies with anything).
        if matches!(base_ty, Ty::Scalar(_) | Ty::Named(_) | Ty::Tuple(_)) {
            let disp = base_ty.display(self.interner);
            self.error(
                span,
                "E0401",
                format!(
                    "cannot index a value of type `{disp}` with `[..]`; only arrays, tensors, \
                     slices, and pointers are indexable (use `.field` / `.0` for a struct / tuple)"
                ),
            );
            return Ty::Unknown;
        }
        match base_ty {
            Ty::Tensor { elem, shape, .. } => {
                if indices.len() != shape.rank() {
                    self.error(
                        span,
                        "E0501",
                        format!(
                            "this tensor has rank {} but is indexed with {} {}",
                            shape.rank(),
                            indices.len(),
                            if indices.len() == 1 {
                                "index"
                            } else {
                                "indices"
                            }
                        ),
                    );
                } else {
                    // Each compile-time-known index into a *static* dimension must be in range —
                    // the headline bounds check the fixed-size-array arm below enforces, extended
                    // to the shape-typed surface. Without it an out-of-bounds tensor index is
                    // silently accepted, then traps in the interpreter but reads out of bounds in
                    // native codegen — the backends disagree, violating the one hard invariant. A
                    // symbolic/dynamic dim or a non-constant index is left unconstrained.
                    for (ix, dim) in indices.iter().zip(shape.0.iter()) {
                        if let Dim::Const(n) = dim {
                            if let Some(v) = self.eval_index_const(ix) {
                                if v < 0 || v as u64 >= *n {
                                    self.error(
                                        ix.span,
                                        "E0501",
                                        format!(
                                            "index {v} is out of bounds for a dimension of length {n}"
                                        ),
                                    );
                                }
                            }
                        }
                    }
                }
                Ty::Scalar(elem)
            }
            Ty::Array { elem, len } => {
                // A constant index known at compile time must be in bounds — the headline
                // compile-time-safety check applied to fixed-size arrays (an out-of-bounds read is
                // otherwise undefined: the backends disagree). Only a literal index is checked; a
                // runtime `a[i]` is unconstrained.
                if indices.len() == 1 {
                    if let Some(v) = self.eval_index_const(&indices[0]) {
                        if v < 0 || v as u64 >= len {
                            self.error(
                                span,
                                "E0501",
                                format!("index {v} is out of bounds for an array of length {len}"),
                            );
                        }
                    }
                }
                *elem
            }
            Ty::Slice(e) | Ty::Ptr { pointee: e, .. } => *e,
            _ => Ty::Unknown,
        }
    }

    fn dim_str(&self, d: Dim) -> String {
        match d {
            Dim::Const(n) => n.to_string(),
            Dim::Var(s) => self.sym_str(s).to_string(),
            Dim::Dynamic => "?".to_string(),
        }
    }
}

fn dims_equal(a: Dim, b: Dim) -> bool {
    match (a, b) {
        (Dim::Dynamic, _) | (_, Dim::Dynamic) => true,
        (Dim::Const(x), Dim::Const(y)) => x == y,
        (Dim::Var(x), Dim::Var(y)) => x == y,
        _ => false,
    }
}

fn parse_dim_text(text: &str) -> u64 {
    text.chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap_or(0)
}

fn apply_subst(ty: &Ty, dims: &HashMap<Symbol, Dim>, tys: &HashMap<Symbol, Ty>) -> Ty {
    match ty {
        Ty::Named(v) => tys.get(v).cloned().unwrap_or_else(|| ty.clone()),
        Ty::Tensor {
            elem,
            shape,
            layout,
        } => {
            let new = shape
                .0
                .iter()
                .map(|d| match d {
                    Dim::Var(v) => dims.get(v).copied().unwrap_or(*d),
                    other => *other,
                })
                .collect();
            Ty::Tensor {
                elem: *elem,
                shape: Shape(new),
                layout: layout.clone(),
            }
        }
        Ty::Ptr { mutable, pointee } => Ty::Ptr {
            mutable: *mutable,
            pointee: Box::new(apply_subst(pointee, dims, tys)),
        },
        Ty::Ref { mutable, pointee } => Ty::Ref {
            mutable: *mutable,
            pointee: Box::new(apply_subst(pointee, dims, tys)),
        },
        Ty::Slice(e) => Ty::Slice(Box::new(apply_subst(e, dims, tys))),
        Ty::Array { elem, len } => Ty::Array {
            elem: Box::new(apply_subst(elem, dims, tys)),
            len: *len,
        },
        Ty::Tuple(items) => Ty::Tuple(items.iter().map(|i| apply_subst(i, dims, tys)).collect()),
        other => other.clone(),
    }
}
