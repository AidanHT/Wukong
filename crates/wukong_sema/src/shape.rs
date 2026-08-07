//! Shape checking — the part of sema that makes tensor dimensions a compile-time guarantee.
//!
//! An `impl Sema<'_>` continuation of `lib.rs` (same struct, same private helpers), holding:
//!  * **Call typing** (`type_call`): every call expression. A resolved user function goes to
//!    `check_fn_call`; otherwise this is where the nominally-typed builtins live (the math
//!    intrinsics, `alloc_*`/`free`, the `read_*`/`write_*` file-I/O family, `now_ns`, `print`/
//!    `println`, `assert`, and `.len()` on a slice), each with its arity/argument rules. Anything
//!    else — an unmodeled builtin, a method, a multi-segment path — stays `Ty::Unknown`.
//!  * **Call unification** (`check_fn_call`): the callee's declared parameter types are unified
//!    against the argument types — for *every* resolved call, generic or not, so a plain scalar
//!    mismatch is caught here too. Dimension variables (`M`, `N`, `K`) are bound either from an
//!    explicit turbofish or inferred from the arguments; a conflicting binding is a `E0502`
//!    dimension mismatch and a differing rank is `E0501`. Value-*type* generics (`fn f<T>`) are
//!    bound by `infer_type_generics` and substituted by `apply_subst`.
//!  * **Indexing** (`type_index`): a non-indexable base is `E0401`, a wrong index count is `E0501`,
//!    and a compile-time-constant index outside a static dimension / array length is `E0501` too.
//!
//! `unify` has **two modes**, and the `rigid` flag is load-bearing (see `unify_dim`):
//!  * `rigid == false` — **call sites only**. The callee's dim vars are inference variables to bind
//!    from the argument shapes.
//!  * `rigid == true` — **body checks** (`check_return_shape`, `check_binop_shapes`). Both shapes are
//!    already fully determined, so the function's own generic dims are universally quantified and
//!    match by *identity* (`dims_equal`); nothing binds. This is what stops a generic function lying
//!    about the shape it returns or merges.
//!
//! `Dim::Dynamic` (`?`) is the documented lenient escape hatch in both modes, and `Unknown`/`Error`
//! on either side short-circuits `unify` — the crate-wide leniency rule.

use wukong_span::FxHashMap as HashMap;

use wukong_ast::{Expr, ExprKind, TypeExpr, TypeKind};
use wukong_span::{Span, Symbol};
use wukong_types::{Dim, Layout, Scalar, Shape, Ty};

use crate::{DefKind, FnSig, Sema};

/// Result types for the math intrinsics the backends lower directly (`sqrt`, `rsqrt`, `exp`, `log`,
/// `fmax`, `fmin`, …). The result is the float type of the first argument (an `f32`/`f64` scalar or a
/// float SIMD vector), defaulting to `f32` so a bare `exp(x)` is still typed when the argument's type
/// is unknown — except for `abs`/`round`/`floor`/`ceil`/`trunc`, which *preserve* the argument's type
/// including an integer one (see the comment below). Returns `None` for any other callee — that keeps
/// `type_call` lenient on the unmodeled-builtin path (`min`, `f32x8::load`, …).
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
        "sqrt" | "rsqrt" | "exp" | "log" | "pow" | "exp2" | "log2" | "exp10" | "log10"
        | "expm1" | "log1p" | "sinh" | "cosh" | "asinh" | "acosh" | "atanh" | "atan" | "tan"
        | "asin" | "acos" | "atan2" | "hypot" | "cbrt" | "erf" | "sin" | "cos" | "tanh"
        | "sigmoid" | "silu" | "gelu" | "silu_backward" | "gelu_backward" | "sigmoid_backward"
        | "tanh_backward" | "elu_backward" | "softplus_backward" | "elu" | "leaky_relu"
        | "softplus" | "mish" | "selu" | "tanhshrink" | "hardsigmoid" | "hardswish"
        | "softsign" | "logsigmoid" | "fmax" | "fmin" => Some(float_ty),
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
        Ty::Slice(_) => "a slice",
        Ty::Named(_) => "a named type",
        _ => "a different type",
    }
}

/// A short human name for a tensor's declared layout, for the layout-mismatch diagnostic.
fn layout_name(l: &Layout) -> String {
    match l {
        Layout::Contiguous => "contiguous (row-major)".to_string(),
        Layout::ColMajor => "col_major".to_string(),
        Layout::Strided => "strided".to_string(),
        Layout::Tiled(v) => format!("tiled{v:?}"),
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
                        matches!(
                            t,
                            Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Unit | Ty::Fn { .. }
                        ) || self.is_aggregate_ty(t)
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
                // The heap builtins: the typed `alloc_<T>(n) -> []T` family and `free(s)`. Typed
                // here (like the math intrinsics) so `let s = alloc_f32(n)` infers `[]f32` and the
                // slice composes with the existing machinery (`s[i]`, `s.len()`, `for x in s`,
                // call-site passing). Misuse is rejected with the standard arity/type codes —
                // mir_build lowers these structurally, so a malformed call must not reach it.
                if let Some(elem) = crate::heap_alloc_elem(self.sym_str(name)) {
                    self.types.insert(callee.id, Ty::Unknown);
                    if args.len() != 1 {
                        self.error(
                            span,
                            "E0503",
                            format!(
                                "`{}` takes exactly 1 argument (the element count), but {} were \
                                 supplied",
                                self.sym_str(name),
                                args.len()
                            ),
                        );
                    } else {
                        match &arg_tys[0] {
                            // Any integer type is a valid length (a negative value clamps to an
                            // empty slice at runtime); `Unknown`/`Error` stays lenient.
                            Ty::Scalar(s) if s.is_int() => {}
                            Ty::Unknown | Ty::Error => {}
                            other => self.error(
                                span,
                                "E0401",
                                format!(
                                    "`{}` expects an integer element count, found a value of type \
                                     `{}`",
                                    self.sym_str(name),
                                    other.display(self.interner)
                                ),
                            ),
                        }
                    }
                    return Ty::Slice(Box::new(Ty::Scalar(elem)));
                }
                if self.sym_str(name) == "free" {
                    self.types.insert(callee.id, Ty::Unknown);
                    if args.len() != 1 {
                        self.error(
                            span,
                            "E0503",
                            format!(
                                "`free` takes exactly 1 argument (the slice to release), but {} \
                                 were supplied",
                                args.len()
                            ),
                        );
                    } else {
                        match &arg_tys[0] {
                            Ty::Slice(_) | Ty::Unknown | Ty::Error => {}
                            other => self.error(
                                span,
                                "E0401",
                                format!(
                                    "`free` expects a slice returned by an `alloc_*` builtin, \
                                     found a value of type `{}`",
                                    other.display(self.interner)
                                ),
                            ),
                        }
                    }
                    return Ty::Unit;
                }
                // The file-I/O builtins: the `read_<T>(path, buf) -> i64` / `write_<T>(path, buf)
                // -> i64` family over the dtypes (f32, f64, i32, i64, i8, u8). Typed here (like heap
                // builtins) so the result is a concrete `i64` — `let n = read_f32(p, s)` composes
                // with arithmetic and comparisons — and misuse is rejected with the standard
                // arity/type codes before the structural lowering in mir_build. `path` is a
                // NUL-terminated `*u8` (a string literal or a `*u8` binding); `buf` is a `[]T` slice
                // whose element type must match the name's dtype (a `[]i32` handed to `read_f32`
                // reads the wrong element stride — a silent backend divergence). A user-defined
                // function of the same name shadows the builtin (handled by the resolved-callee path
                // above). `Unknown`/`Error` arguments stay lenient (they already carry an error).
                if let Some(elem) = crate::file_io_elem(self.sym_str(name)) {
                    self.types.insert(callee.id, Ty::Unknown);
                    if args.len() != 2 {
                        self.error(
                            span,
                            "E0503",
                            format!(
                                "`{}` takes exactly 2 arguments (a `*u8` path and a `[]{}` \
                                 buffer), but {} were supplied",
                                self.sym_str(name),
                                elem.name(),
                                args.len()
                            ),
                        );
                    } else {
                        // arg0 — the path: a NUL-terminated `*u8` (a string literal or `*u8` value).
                        match &arg_tys[0] {
                            Ty::Ptr { pointee, .. }
                                if matches!(pointee.as_ref(), Ty::Scalar(Scalar::U8)) => {}
                            Ty::Unknown | Ty::Error => {}
                            other => self.error(
                                span,
                                "E0401",
                                format!(
                                    "`{}` expects a `*u8` path as its first argument, found a value \
                                     of type `{}`",
                                    self.sym_str(name),
                                    other.display(self.interner)
                                ),
                            ),
                        }
                        // arg1 — the buffer: a `[]T` slice whose element matches the name's dtype.
                        match &arg_tys[1] {
                            Ty::Slice(e) if matches!(e.as_ref(), Ty::Scalar(s) if *s == elem) => {}
                            Ty::Unknown | Ty::Error => {}
                            other => self.error(
                                span,
                                "E0401",
                                format!(
                                    "`{}` expects a `[]{}` buffer as its second argument, found a \
                                     value of type `{}`",
                                    self.sym_str(name),
                                    elem.name(),
                                    other.display(self.interner)
                                ),
                            ),
                        }
                    }
                    return Ty::Scalar(Scalar::I64);
                }
                // The `now_ns()` timing intrinsic: a zero-argument monotonic nanosecond clock, typed
                // `i64` here (like the file-I/O family) so `let t = now_ns()` composes in arithmetic —
                // `t1 - t0` is the elapsed span. `mir_build` lowers it to the `wukong_now_ns` runtime
                // call. It takes no arguments; any argument is the standard arity error (E0503),
                // caught before lowering. A user-defined `now_ns` shadows the builtin (handled by the
                // resolved-callee path above).
                if crate::is_now_ns(self.sym_str(name)) {
                    self.types.insert(callee.id, Ty::Unknown);
                    if !args.is_empty() {
                        self.error(
                            span,
                            "E0503",
                            format!(
                                "`now_ns` takes no arguments, but {} were supplied",
                                args.len()
                            ),
                        );
                    }
                    return Ty::Scalar(Scalar::I64);
                }
                // `print`/`println` render exactly one value. Extra arguments were silently dropped
                // (`print(1, 2)` printed just `1`) — both backends agree, so it is not a divergence,
                // but a quiet footgun where the programmer expects all arguments to appear. Reject a
                // too-many-argument call (E0503). A zero-argument `print()` / `println()` (a blank
                // line / newline) stays valid.
                {
                    let nm = self.sym_str(name);
                    if nm == "print" || nm == "println" {
                        if args.len() > 1 {
                            self.error(
                                span,
                                "E0503",
                                format!(
                                    "`{nm}` takes a single value to print, but {} were supplied",
                                    args.len()
                                ),
                            );
                            self.types.insert(callee.id, Ty::Unknown);
                            return Ty::Unknown;
                        }
                        // A `()`/unit argument has no printable value — e.g. `print(x)` where
                        // `x = if c { 1 }` (an else-less `if` is unit) or `x = match … { … }` with
                        // statement arms. mir_build lowered the unit `let` to an `alloca void` and the
                        // read to a `load void`: native -O0 read uninitialized stack (nondeterministic
                        // garbage), while the interpreter and -O2 (mem2reg) produced 0 — a divergence
                        // that broke BOTH invariants at once. Reject it (E0401), consistent with `()`
                        // already being non-computable in an arithmetic operand.
                        if matches!(arg_tys.first(), Some(Ty::Unit)) {
                            self.error(
                                span,
                                "E0401",
                                format!(
                                    "`{nm}` cannot print a `()` value; supply a printable scalar \
                                     (an `if` with no `else`, or a `match` with statement arms, \
                                     yields `()`)"
                                ),
                            );
                            self.types.insert(callee.id, Ty::Unknown);
                            return Ty::Unknown;
                        }
                    }
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

        // `s.len()` on a slice is a *modeled* builtin method: its result is the slice's runtime
        // length, an `i64`. Without this the call typed `Unknown` and a range bound `0..s.len()`
        // defaulted the loop counter to `i32` while mir_build lowers the length as an `i64` load —
        // a mixed-width `Cmp` the MIR verifier rejects (an ICE on `for i in 0..s.len()`).
        if let ExprKind::Field { base, name } = &callee.kind {
            if args.is_empty() && self.sym_str(name.sym) == "len" {
                let bty = self.type_expr(base);
                if matches!(bty, Ty::Slice(_)) {
                    self.types.insert(callee.id, Ty::Unknown);
                    return Ty::Scalar(Scalar::I64);
                }
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

        let mut dims: HashMap<Symbol, Dim> = HashMap::default();
        let mut tys: HashMap<Symbol, Ty> = HashMap::default();

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
                    // Decode with the same radix/`_`/suffix-aware parser mir_build's
                    // `turbofish_dim_value` uses (`parse_int`). The old digits-only scan bound
                    // `::<3_0>` to 3 here while codegen passed the hidden dim 30 — sema shape-checked
                    // one dimension and the callee addressed with another (accepted program, native
                    // read past the buffer while the interpreter trapped) — and bound `::<0x4>` to 0,
                    // rejecting valid code with a dimension the source never wrote.
                    TypeKind::Int(s) => {
                        let n = crate::parse_u64_text(self.sym_str(*s)).unwrap_or(0);
                        dims.insert(*g, Dim::Const(n));
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
            // Infer VALUE-TYPE generics (a `Ty::Named(g)` parameter position) from the concrete
            // argument type, so `id(c)` with `c: f32` binds `T := f32` and the whole call types as
            // `f32` — its result can then be cast/used at the concrete type, and `mir_build`
            // monomorphizes the callee to `id$f32`. A dimension generic never appears as `Ty::Named`
            // (it is a `Dim::Var` inside a shape), so this binds only type generics; dims are still
            // inferred by `unify` from the shapes below.
            infer_type_generics(param, arg, &sig.generics, &mut tys);
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
                    layout: pl,
                },
                Ty::Tensor {
                    elem: ae,
                    shape: as_,
                    layout: al,
                },
            ) => {
                // The declared layout is part of the type: `tensor_strides` and
                // `lower_multi_index_dyn` both DECLINE a non-`Contiguous` tensor (C0001), so a
                // `.col_major` value routed through a `Contiguous`-typed callee was silently
                // reinterpreted row-major — the same access is a hard error in place and a wrong
                // answer one call away (observed: prints 3, the row-major reading, where the
                // column-major reading is 1).
                if pl != al {
                    self.error(
                        span,
                        "E0502",
                        format!(
                            "tensor layout mismatch: expected {}, found {}",
                            layout_name(pl),
                            layout_name(al)
                        ),
                    );
                }
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
            // parameter (see `tests/run/tensor_add.wk`, which feeds a `[f32; 6]` to a
            // `Tensor[f32, 2, 3]`). But when the tensor's shape and the array length are *both*
            // statically known, the buffer must hold the right number of elements of the right
            // type — otherwise a too-small/wrong array satisfies any tensor and every
            // in-tensor-bounds index becomes an out-of-bounds read (the interpreter traps, native
            // codegen does not: a backend divergence). A symbolic / dynamic dim stays lenient
            // (there is no concrete element count to check against).
            (
                Ty::Tensor {
                    elem: pe,
                    shape,
                    layout: pl,
                },
                Ty::Array { elem: ae, len },
            ) => {
                // A fixed-size array is row-major, so it can only decay to a `Contiguous` tensor.
                if *pl != Layout::Contiguous {
                    self.error(
                        span,
                        "E0502",
                        format!(
                            "tensor layout mismatch: expected {}, found a row-major array",
                            layout_name(pl)
                        ),
                    );
                }
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
            // A slice parameter `[]T` accepts a slice `[]T` or — via unsizing — a fixed-size array
            // `[T; N]` argument; either way the element types must unify (a slice of a different
            // element type has a different stride, so `s[i]` would read the wrong bytes). The
            // array's static length is intentionally dropped to the slice's runtime length.
            (Ty::Slice(pe), Ty::Slice(ae) | Ty::Array { elem: ae, .. }) => {
                self.unify(pe, ae, dims, span, rigid)
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
            // tensor parameter — see `tests/run/tensor_add.wk`), so only Scalar/Vector/Tensor kind
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
            // A `[]T` slice is a 16-byte `(ptr, len)` fat pointer and a tuple is an aggregate —
            // neither is a tensor base pointer, and neither had an arm here, so both fell to the
            // lenient `_` below and were silently accepted. mir_build then passed the address of the
            // slice HEADER where the callee expects the data base: `get1(alloc_f32(4), 2)` printed 0
            // on every backend and `a[0]` read the low half of the heap POINTER as an `f32` — an
            // address leaked as data, with both backends agreeing so the differential gate is blind.
            (Ty::Tensor { .. }, Ty::Slice(_) | Ty::Tuple(_))
            | (Ty::Slice(_) | Ty::Tuple(_), Ty::Tensor { .. }) => {
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
            // Same hole for a struct/enum value. `Ty::Named` also spells a bare generic TYPE variable
            // (`fn f<T>(x: T)`), which must stay lenient, so only a name that resolves to a declared
            // struct/enum is reported.
            (Ty::Tensor { .. }, Ty::Named(n)) | (Ty::Named(n), Ty::Tensor { .. })
                if self.is_declared_named_ty(*n) =>
            {
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
                Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Array { .. } | Ty::Tuple(_) | Ty::Slice(_),
            )
            | (
                Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Array { .. } | Ty::Tuple(_) | Ty::Slice(_),
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
            Dim::Const(pc) => match ad {
                Dim::Const(ac) if pc != ac => self.error(
                    span,
                    "E0502",
                    format!("dimension mismatch: expected {pc}, found {ac}"),
                ),
                // A caller's own universally-quantified dim cannot be ASSUMED equal to the callee's
                // fixed size — that is a claim the checker cannot prove, and the very hole `rigid`
                // mode closes on the return path was left open here on the call path: `fn fwd<N>(a:
                // Tensor[f32, N]) { takes64(a) }` compiled clean and `fwd([1.0, 2.0])` read 248 bytes
                // past an 8-byte stack array (interp trapped, native returned 0). `Dim::Dynamic`
                // stays the documented `?` escape hatch.
                Dim::Var(v) => {
                    let name = self.sym_str(v).to_string();
                    self.error(
                        span,
                        "E0502",
                        format!(
                            "dimension mismatch: expected {pc}, found the generic dimension `{name}` \
                             (a universally-quantified dimension cannot be assumed equal to {pc}; \
                             declare the parameter `?` if the size is a runtime value)"
                        ),
                    );
                }
                _ => {}
            },
            Dim::Dynamic => {}
        }
    }

    /// Evaluate a compile-time index: a literal, a negation or fold of literals (`eval_const_int`), an
    /// enum variant's discriminant, or a top-level `const` name resolved to its recorded initializer
    /// (recursing for a const-references-const). `None` means "not compile-time known", which leaves
    /// the index unconstrained rather than erroring. Without the const resolution, `xs[I]` for
    /// `const I = 99` slipped past the bounds check — sema accepted it,
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
        // An enum variant used as an index (`xs[E::V]`, or via a const of enum type) is its
        // discriminant — resolve it so the compile-time bounds check covers it (it lowers to
        // that constant, so an out-of-range variant would otherwise trap-vs-OOB-read at runtime).
        if let Some(d) = self.enum_variant_disc(e) {
            return Some(d);
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
        // indexable construct sema does not yet model, and `Unknown` unifies with anything). NOTE: a
        // bare generic type variable is also spelled `Ty::Named`, so `x[i]` on an `x: T` parameter is
        // rejected here as well — unlike `unify`, this arm does not filter through
        // `is_declared_named_ty`. No fixture indexes a generic-typed value (they annotate the
        // container instead: `let a: [T; 2]`, whose base type is an `Array`), so the leniency
        // asymmetry with `unify` has never been exercised.
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

    /// Whether a `Ty::Named` refers to a DECLARED struct/enum rather than a generic type variable —
    /// both are spelled `Named`, and only the former can be reported as a kind clash.
    fn is_declared_named_ty(&self, n: Symbol) -> bool {
        matches!(
            self.defs.lookup(n).map(|d| &d.kind),
            Some(DefKind::Struct(_) | DefKind::Enum(_))
        )
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

/// Infer value-type generics from a concrete argument: bind each `Ty::Named(g)` parameter position
/// (with `g` in `generics`) to the corresponding argument type, recursing through pointer /
/// reference / slice / array / tuple structure exactly like [`apply_subst`]. Only *type* generics are
/// bound (a dimension generic is a `Dim::Var` inside a shape, never a `Ty::Named`); a first binding
/// wins, and an `Unknown`/`Error` argument binds nothing (it would poison every later use of `g`).
fn infer_type_generics(param: &Ty, arg: &Ty, generics: &[Symbol], tys: &mut HashMap<Symbol, Ty>) {
    match (param, arg) {
        (Ty::Named(g), a) if generics.contains(g) => {
            if !matches!(a, Ty::Unknown | Ty::Error) {
                tys.entry(*g).or_insert_with(|| a.clone());
            }
        }
        (Ty::Ptr { pointee: p, .. }, Ty::Ptr { pointee: a, .. })
        | (Ty::Ref { pointee: p, .. }, Ty::Ref { pointee: a, .. })
        | (Ty::Slice(p), Ty::Slice(a)) => infer_type_generics(p, a, generics, tys),
        (Ty::Array { elem: p, .. }, Ty::Array { elem: a, .. }) => {
            infer_type_generics(p, a, generics, tys)
        }
        (Ty::Tuple(ps), Ty::Tuple(as_)) => {
            for (p, a) in ps.iter().zip(as_) {
                infer_type_generics(p, a, generics, tys);
            }
        }
        _ => {}
    }
}

/// Substitute a call's resolved generics into a declared type: a `Ty::Named` in `tys` becomes the
/// bound type, and each `Dim::Var` in a tensor shape becomes its binding in `dims`. A var with no
/// binding is left **as-is** (not an error): the caller unifies parameter by parameter, so a dim can
/// still be bound by a later argument, and an unbound dim in the *return* type stays symbolic.
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
