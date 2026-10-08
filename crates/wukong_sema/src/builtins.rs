//! **Named kernel builtins** — the explicit, shape-checked door to the tuned runtime kernels.
//!
//! Before this module the only way to reach a `wukong_runtime` kernel from `.wk` was to write a loop
//! nest one of the recognizers in `wukong_mir_build` happens to match; the one exception, `sdpa`, had
//! no signature at all, so a malformed attention call surfaced as a codegen `C0001` at the call site
//! (or, with a dynamic dimension, as nothing). A recognizer is an *optimization*: a program that is a
//! GEMM spelled a hair differently silently gets scalar code, and nothing lets the programmer say
//! "I mean this kernel". The builtins below are that statement, and — the point of a shape-safe
//! language — every one of them has a real **shape signature** checked here, with the same `E05xx`
//! codes a user-written generic function gets:
//!
//! | builtin | shape signature (all operands `f32`, row-major) | kernel |
//! |---|---|---|
//! | `matmul::<M, K, N>(a, b, out)` | `a: [M, K]`, `b: [K, N]`, `out: [M, N]` | `wukong_sgemm` |
//! | `matmul_nt::<M, K, N>(a, b, out)` | `a: [M, K]`, `b: [N, K]`, `out: [M, N]` (`A·Bᵀ`) | `wukong_sgemm_nt` |
//! | `matmul_tn::<M, K, N>(a, b, out)` | `a: [K, M]`, `b: [K, N]`, `out: [M, N]` (`Aᵀ·B`) | `wukong_sgemm_tn` |
//! | `linear::<M, K, N>(x, w, b, out)` | `x: [M, K]`, `w: [N, K]`, `b: [N]`, `out: [M, N]` | `wukong_sgemm_nt_epi` |
//! | `transpose::<R, C>(x, out)` | `x: [R, C]`, `out: [C, R]` | `wukong_transpose_f32` |
//! | `softmax::<R, C>(x, out)`, `log_softmax` | `x, out: [R, C]`, over the last axis | `wukong_norm_f32` |
//! | `layernorm::<R, C>(x, gamma, beta, out, eps)` | `x, out: [R, C]`, `gamma, beta: [C]` | `wukong_norm_affine_f32` |
//! | `rmsnorm::<R, C>(x, gamma, out, eps)` | `x, out: [R, C]`, `gamma: [C]` | `wukong_norm_affine_f32` |
//! | `gelu`/`silu`/`sigmoid`/`tanh`/`exp`/`log`/`relu(x, out)` | `x, out: [N]` | `wukong_vmath_f32` |
//! | `sum`/`amax`/`amin(x) -> f32`, `dot(x, y) -> f32` | `x, y: [N]` | `wukong_sreduce_f32` |
//! | `sdpa(q, k, v, out, s, d, scale, causal)` | `q, k, v, out` hold `s·d` elements | `wukong_attention_f32` |
//!
//! **Where the dimensions come from**, in this order: an explicit turbofish (`matmul::<2, 3, 4>`, an
//! integer literal, a `const`, a generic dimension in scope, or an in-scope integer variable, which
//! is then a *runtime* extent); the shape of every `Tensor[f32, ..]` operand, unified axis by axis
//! exactly like a call to a user generic function (a conflict is `E0502`, a rank clash `E0501`); the
//! length of a fixed `[f32; N]` array operand, which fixes the one dimension of its shape that is
//! still free (a length that is not a whole multiple of the fixed ones is `E0501`); for the row-wise
//! operations, a row count left free by everything else defaults to **1** (a vector is one row);
//! and last the runtime length of a `[]f32` slice operand. A dimension none of those fixes is
//! `E0409`. A fully static array whose length disagrees with its shape is `E0501`; an operand whose
//! extent is only known at run time (a slice, or an array against a runtime dimension) is checked
//! at run time by the lowering, which traps exactly like a failed `assert`.
//!
//! The one-argument activations (`gelu(x)`) stay the scalar math intrinsics; only the two-argument
//! buffer form is the kernel builtin. A user function, or a local binding, of the same name shadows
//! any builtin.
//!
//! The resolution is recorded per call in [`crate::SemaResult::builtin_calls`] so `mir_build` lowers
//! the call from the dimensions decided here and the two crates cannot drift.

use wukong_ast::{Expr, ExprKind, TypeExpr, TypeKind};
use wukong_span::{Span, Symbol};
use wukong_types::{Dim, Layout, Scalar, Ty};

use crate::Sema;

/// A named kernel builtin (see the module docs for the full table).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelBuiltin {
    /// `matmul(a, b, out)` — `out = a·b`.
    Matmul,
    /// `matmul_nt(a, b, out)` — `out = a·bᵀ` (`b` stored `[N, K]`).
    MatmulNt,
    /// `matmul_tn(a, b, out)` — `out = aᵀ·b` (`a` stored `[K, M]`).
    MatmulTn,
    /// `linear(x, w, b, out)` — `out = x·wᵀ + b` (`nn.Linear`).
    Linear,
    /// `transpose(x, out)` — `out = xᵀ`.
    Transpose,
    /// `softmax(x, out)` over the last axis.
    Softmax,
    /// `log_softmax(x, out)` over the last axis.
    LogSoftmax,
    /// `layernorm(x, gamma, beta, out, eps)` over the last axis.
    LayerNorm,
    /// `rmsnorm(x, gamma, out, eps)` over the last axis.
    RmsNorm,
    /// An elementwise activation `act(x, out)`.
    Map(Activation),
    /// `sum(x) -> f32`.
    Sum,
    /// `amax(x) -> f32` (largest element).
    Amax,
    /// `amin(x) -> f32` (smallest element).
    Amin,
    /// `dot(x, y) -> f32`.
    Dot,
    /// `sdpa(q, k, v, out, s, d, scale, causal)` — fused scaled-dot-product attention.
    Sdpa,
}

/// The elementwise activations exposed as two-argument buffer builtins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Activation {
    Exp,
    Log,
    Tanh,
    Sigmoid,
    Relu,
    Silu,
    Gelu,
}

/// One dimension of a builtin call, as sema resolved it. `mir_build` materializes each as an `i64`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinDim {
    /// A compile-time extent.
    Const(u64),
    /// A runtime extent held by the in-scope integer binding `name`: a generic dimension's hidden
    /// parameter, or an integer variable named in the turbofish. `signed` is the binding's
    /// signedness, which decides how a narrower integer widens to the kernels' `i64`.
    Named { name: Symbol, signed: bool },
    /// A runtime extent: the length of slice argument `arg`, divided by `div` (the product of the
    /// operand's other, constant, dimensions — `div` is 1 for a rank-1 operand).
    SliceLen { arg: usize, div: u64 },
    /// A runtime extent derived from other dimensions — a tensor operand's `3*D` or `H*HD`
    /// (`wukong_types::Dim::Expr`, never a bare constant or variable) — which `mir_build` evaluates
    /// from the in-scope dimension bindings exactly as a tensor stride does.
    Derived(Dim),
}

/// The resolution of one builtin call, keyed by the call expression's `NodeId`.
#[derive(Clone, Debug)]
pub struct BuiltinCall {
    pub kind: KernelBuiltin,
    /// Each of the builtin's dimensions ([`builtin_dims`] order). Empty for `sdpa`, whose extents
    /// are its `s`/`d` value arguments.
    pub dims: Vec<BuiltinDim>,
}

/// Which builtin `name` names when called with `arity` arguments, or `None` when it is not one.
///
/// The activations share their names with the scalar math intrinsics: `gelu(x)` (one argument) is
/// still the scalar intrinsic, and only the two-argument buffer form `gelu(x, out)` is the kernel.
/// `relu` has no scalar intrinsic, so every arity is the builtin (and a wrong one is `E0503`).
pub fn kernel_builtin(name: &str, arity: usize) -> Option<KernelBuiltin> {
    use KernelBuiltin::*;
    Some(match name {
        "matmul" => Matmul,
        "matmul_nt" => MatmulNt,
        "matmul_tn" => MatmulTn,
        "linear" => Linear,
        "transpose" => Transpose,
        "softmax" => Softmax,
        "log_softmax" => LogSoftmax,
        "layernorm" => LayerNorm,
        "rmsnorm" => RmsNorm,
        "sum" => Sum,
        "amax" => Amax,
        "amin" => Amin,
        "dot" => Dot,
        "sdpa" => Sdpa,
        "relu" => Map(Activation::Relu),
        _ => {
            let act = match name {
                "exp" => Activation::Exp,
                "log" => Activation::Log,
                "tanh" => Activation::Tanh,
                "sigmoid" => Activation::Sigmoid,
                "silu" => Activation::Silu,
                "gelu" => Activation::Gelu,
                _ => return None,
            };
            if arity == 1 {
                return None;
            }
            Map(act)
        }
    })
}

/// One parameter of a builtin's signature.
#[derive(Clone, Copy)]
enum Param {
    /// An `f32` buffer operand whose row-major shape is the listed builtin dimensions.
    Buf {
        name: &'static str,
        dims: &'static [usize],
    },
    /// A numeric scalar (an `eps`), coerced to `f32` by the lowering.
    Float(&'static str),
}

/// A builtin's shape signature.
struct Spec {
    name: &'static str,
    /// Dimension names, in turbofish order.
    dims: &'static [&'static str],
    params: &'static [Param],
    /// The row dimension of a row-wise operation: it defaults to 1 when nothing else fixes it, and
    /// an operand of the row-wise shape `[R, C]` may also be written as a rank-1 vector `[C]`.
    rows: Option<usize>,
    /// A flat (rank-1, elementwise/reduction) operation: an operand may be a tensor of any rank
    /// with constant dimensions, standing for its element count.
    flat: bool,
    returns_f32: bool,
}

const fn buf(name: &'static str, dims: &'static [usize]) -> Param {
    Param::Buf { name, dims }
}

const GEMM_DIMS: &[&str] = &["M", "K", "N"];
const ROW_DIMS: &[&str] = &["R", "C"];
const FLAT_DIMS: &[&str] = &["N"];

const MATMUL: Spec = Spec {
    name: "matmul",
    dims: GEMM_DIMS,
    params: &[buf("a", &[0, 1]), buf("b", &[1, 2]), buf("out", &[0, 2])],
    rows: None,
    flat: false,
    returns_f32: false,
};
const MATMUL_NT: Spec = Spec {
    name: "matmul_nt",
    dims: GEMM_DIMS,
    params: &[buf("a", &[0, 1]), buf("b", &[2, 1]), buf("out", &[0, 2])],
    rows: None,
    flat: false,
    returns_f32: false,
};
const MATMUL_TN: Spec = Spec {
    name: "matmul_tn",
    dims: GEMM_DIMS,
    params: &[buf("a", &[1, 0]), buf("b", &[1, 2]), buf("out", &[0, 2])],
    rows: None,
    flat: false,
    returns_f32: false,
};
const LINEAR: Spec = Spec {
    name: "linear",
    dims: GEMM_DIMS,
    params: &[
        buf("x", &[0, 1]),
        buf("w", &[2, 1]),
        buf("b", &[2]),
        buf("out", &[0, 2]),
    ],
    rows: None,
    flat: false,
    returns_f32: false,
};
const TRANSPOSE: Spec = Spec {
    name: "transpose",
    dims: ROW_DIMS,
    params: &[buf("x", &[0, 1]), buf("out", &[1, 0])],
    rows: None,
    flat: false,
    returns_f32: false,
};
const SOFTMAX: Spec = Spec {
    name: "softmax",
    dims: ROW_DIMS,
    params: &[buf("x", &[0, 1]), buf("out", &[0, 1])],
    rows: Some(0),
    flat: false,
    returns_f32: false,
};
const LOG_SOFTMAX: Spec = Spec {
    name: "log_softmax",
    dims: ROW_DIMS,
    params: &[buf("x", &[0, 1]), buf("out", &[0, 1])],
    rows: Some(0),
    flat: false,
    returns_f32: false,
};
const LAYERNORM: Spec = Spec {
    name: "layernorm",
    dims: ROW_DIMS,
    params: &[
        buf("x", &[0, 1]),
        buf("gamma", &[1]),
        buf("beta", &[1]),
        buf("out", &[0, 1]),
        Param::Float("eps"),
    ],
    rows: Some(0),
    flat: false,
    returns_f32: false,
};
const RMSNORM: Spec = Spec {
    name: "rmsnorm",
    dims: ROW_DIMS,
    params: &[
        buf("x", &[0, 1]),
        buf("gamma", &[1]),
        buf("out", &[0, 1]),
        Param::Float("eps"),
    ],
    rows: Some(0),
    flat: false,
    returns_f32: false,
};
const MAP: Spec = Spec {
    name: "activation",
    dims: FLAT_DIMS,
    params: &[buf("x", &[0]), buf("out", &[0])],
    rows: None,
    flat: true,
    returns_f32: false,
};
const REDUCE: Spec = Spec {
    name: "reduction",
    dims: FLAT_DIMS,
    params: &[buf("x", &[0])],
    rows: None,
    flat: true,
    returns_f32: true,
};
const DOT: Spec = Spec {
    name: "dot",
    dims: FLAT_DIMS,
    params: &[buf("x", &[0]), buf("y", &[0])],
    rows: None,
    flat: true,
    returns_f32: true,
};

fn spec_of(kind: KernelBuiltin) -> &'static Spec {
    use KernelBuiltin::*;
    match kind {
        Matmul => &MATMUL,
        MatmulNt => &MATMUL_NT,
        MatmulTn => &MATMUL_TN,
        Linear => &LINEAR,
        Transpose => &TRANSPOSE,
        Softmax => &SOFTMAX,
        LogSoftmax => &LOG_SOFTMAX,
        LayerNorm => &LAYERNORM,
        RmsNorm => &RMSNORM,
        Map(_) => &MAP,
        Sum | Amax | Amin => &REDUCE,
        Dot => &DOT,
        // `sdpa` keeps its positional value-dimension ABI and is checked by `check_sdpa`.
        Sdpa => &MAP,
    }
}

/// The dimension names of `kind`, in the order [`BuiltinCall::dims`] lists them.
pub fn builtin_dims(kind: KernelBuiltin) -> &'static [&'static str] {
    if kind == KernelBuiltin::Sdpa {
        return &[];
    }
    spec_of(kind).dims
}

/// The argument index of the buffer `kind` writes, or `None` for a reduction (which returns its
/// result). Shared with `mir_build`, which guards that operand against overlapping an input.
pub fn builtin_output(kind: KernelBuiltin) -> Option<usize> {
    use KernelBuiltin::*;
    match kind {
        Matmul | MatmulNt | MatmulTn | RmsNorm => Some(2),
        Linear | LayerNorm | Sdpa => Some(3),
        Transpose | Softmax | LogSoftmax | Map(_) => Some(1),
        Sum | Amax | Amin | Dot => None,
    }
}

/// For each buffer parameter of `kind` (in argument order), the builtin dimensions its row-major
/// shape is made of. `None` for a non-buffer (scalar) argument. Shared with `mir_build`, which needs
/// the element count of every operand for its runtime extent and overlap checks.
pub fn builtin_operand_dims(kind: KernelBuiltin) -> Vec<Option<&'static [usize]>> {
    if kind == KernelBuiltin::Sdpa {
        return Vec::new();
    }
    spec_of(kind)
        .params
        .iter()
        .map(|p| match p {
            Param::Buf { dims, .. } => Some(*dims),
            Param::Float(_) => None,
        })
        .collect()
}

/// The user-facing spelling of a builtin (its source name), for diagnostics.
fn builtin_name(kind: KernelBuiltin) -> &'static str {
    use KernelBuiltin::*;
    match kind {
        Map(a) => match a {
            Activation::Exp => "exp",
            Activation::Log => "log",
            Activation::Tanh => "tanh",
            Activation::Sigmoid => "sigmoid",
            Activation::Relu => "relu",
            Activation::Silu => "silu",
            Activation::Gelu => "gelu",
        },
        Sum => "sum",
        Amax => "amax",
        Amin => "amin",
        Sdpa => "sdpa",
        other => spec_of(other).name,
    }
}

/// The inference state of one builtin dimension.
#[derive(Clone, PartialEq, Eq)]
enum Res {
    /// Fixed by a turbofish, a tensor shape or an array length: a `Dim::Const`, a `Dim::Var`, or a
    /// derived `Dim::Expr` from a tensor operand's shape.
    Static(Dim),
    /// The runtime length of slice argument `arg`, divided by `div`.
    Slice { arg: usize, div: u64 },
}

/// How a buffer argument was classified.
enum Operand {
    /// Not checkable (`Unknown`/`Error`, or already reported).
    Skip,
    Tensor(Vec<Dim>),
    /// A fixed array holding this many `f32` elements.
    Array(u64),
    Slice,
}

/// The number of `f32` scalars a (possibly nested) fixed array of `f32` holds, or `None` when an
/// element is not `f32` or the count overflows.
fn flat_f32_len(t: &Ty) -> Option<u64> {
    match t {
        Ty::Scalar(Scalar::F32) => Some(1),
        Ty::Array { elem, len, .. } => flat_f32_len(elem)?.checked_mul(*len),
        _ => None,
    }
}

/// The innermost element of a (possibly nested) array, for the element-type diagnostic.
fn array_leaf(t: &Ty) -> &Ty {
    match t {
        Ty::Array { elem, .. } => array_leaf(elem),
        other => other,
    }
}

/// Two extents provably equal: the normal form makes structural equality semantic for constants,
/// variables and derived extents alike (`2*N` and `N + N` are one `Dim`); a `?` equals nothing.
fn dims_same(a: &Dim, b: &Dim) -> bool {
    match (a, b) {
        (Dim::Const(x), Dim::Const(y)) => x == y,
        (Dim::Var(x), Dim::Var(y)) => x == y,
        (Dim::Expr(x), Dim::Expr(y)) => x == y,
        _ => false,
    }
}

impl Sema<'_> {
    /// Type a call whose callee is a named kernel builtin, or return `None` when it is not one (so
    /// `type_call` handles it). A top-level definition of the same name shadows the builtin.
    pub(crate) fn type_kernel_builtin(
        &mut self,
        call: &Expr,
        callee: &Expr,
        generic_args: &[TypeExpr],
        args: &[Expr],
    ) -> Option<Ty> {
        let ExprKind::Path(p) = &callee.kind else {
            return None;
        };
        if !p.is_single() {
            return None;
        }
        let name = p.first().sym;
        // A top-level definition or a local binding of the same name (a function-typed parameter
        // called `relu`, say) shadows the builtin.
        if self.defs.lookup(name).is_some() || self.lookup_local(name).is_some() {
            return None;
        }
        let kind = kernel_builtin(self.sym_str(name), args.len())?;
        self.types.insert(callee.id, Ty::Unknown);
        Some(if kind == KernelBuiltin::Sdpa {
            self.check_sdpa(call, generic_args, args)
        } else {
            self.check_kernel_call(call, kind, generic_args, args)
        })
    }

    /// A builtin writes its output operand in place, so an output that is (a projection of) a
    /// non-`mut` aggregate parameter would mutate the caller's value — the same `E0304` an
    /// assignment `p[i] = …` through that parameter gets.
    fn check_builtin_output_mutable(&mut self, out: &Expr, builtin: &str) {
        let Some(root) = self.assign_root_param(out) else {
            return;
        };
        if self.is_immutable_param(root) && self.param_is_aggregate(root) {
            let nm = self.sym_str(root).to_string();
            self.error(
                out.span,
                "E0304",
                format!(
                    "cannot write the output of `{builtin}` through the non-`mut` parameter \
                     `{nm}`; an aggregate parameter is passed by reference, so this would mutate \
                     the caller's value — add `mut` to the parameter (`fn …(mut {nm}: …)`)"
                ),
            );
        }
    }

    fn builtin_dim_str(&self, d: &Dim) -> String {
        d.display(self.interner)
    }

    fn builtin_shape_str(&self, dims: &[Dim]) -> String {
        let parts: Vec<String> = dims.iter().map(|d| self.builtin_dim_str(d)).collect();
        format!("[{}]", parts.join(", "))
    }

    /// A compile-time integer: a literal (or folded literal arithmetic), an enum discriminant, or a
    /// top-level `const` resolved through its initializer. `None` when not compile-time known.
    fn builtin_const_int(&self, e: &Expr, depth: u32) -> Option<i64> {
        if depth > 32 {
            return None;
        }
        if let ExprKind::Path(p) = &e.kind {
            if p.is_single() && self.lookup_local(p.first().sym).is_none() {
                if let Some(init) = self.consts.get(&p.first().sym) {
                    return self.builtin_const_int(init, depth + 1);
                }
            }
        }
        if let Some(d) = self.enum_variant_disc(e) {
            return Some(d);
        }
        crate::eval_const_int(e, self.interner)
    }

    /// Resolve one turbofish argument of a builtin to a dimension: an integer literal, a generic
    /// dimension in scope, an in-scope integer variable (a runtime extent), or a top-level `const`.
    /// The flag is `true` for an *unsigned* integer variable, which must zero-extend at run time.
    fn builtin_turbofish_dim(&mut self, ga: &TypeExpr, builtin: &str) -> Option<(Dim, bool)> {
        match &ga.kind {
            TypeKind::Int(s) => match crate::parse_u64_text(self.sym_str(*s)) {
                Some(n) => Some((Dim::Const(n), false)),
                None => {
                    self.error(
                        ga.span,
                        "E0401",
                        format!(
                            "`{builtin}` dimension `{}` is not a representable extent",
                            self.sym_str(*s)
                        ),
                    );
                    None
                }
            },
            TypeKind::Path(p) if p.is_single() => {
                let nm = p.first().sym;
                if self.generics.contains(&nm) {
                    return Some((Dim::Var(nm), false));
                }
                if let Some(t) = self.lookup_local(nm) {
                    return match t {
                        Ty::Scalar(s) if s.is_int() => Some((Dim::Var(nm), !s.is_signed())),
                        Ty::Unknown | Ty::Error => None,
                        other => {
                            self.error(
                                ga.span,
                                "E0401",
                                format!(
                                    "a `{builtin}` dimension must be an integer, but `{}` has \
                                     type `{}`",
                                    self.sym_str(nm),
                                    other.display(self.interner)
                                ),
                            );
                            None
                        }
                    };
                }
                if self.consts.contains_key(&nm) {
                    return Some((Dim::Const(self.const_length(nm, 0)), false));
                }
                self.error(
                    ga.span,
                    "E0504",
                    format!(
                        "unknown dimension `{}` in the `{builtin}` turbofish (a dimension must be an \
                         integer literal, a `const`, a generic dimension, or an integer variable in \
                         scope)",
                        self.sym_str(nm)
                    ),
                );
                None
            }
            _ => {
                self.error(
                    ga.span,
                    "E0401",
                    format!("a `{builtin}` turbofish argument must be an integer extent"),
                );
                None
            }
        }
    }

    /// Bind builtin dimension `idx` to `d`, or check `d` against its existing binding.
    #[allow(clippy::too_many_arguments)]
    fn bind_builtin_dim(
        &mut self,
        state: &mut [Option<Res>],
        spec: &Spec,
        builtin: &str,
        idx: usize,
        d: Dim,
        what: &str,
        span: Span,
    ) {
        if matches!(d, Dim::Dynamic) {
            return;
        }
        match &state[idx] {
            None => state[idx] = Some(Res::Static(d)),
            Some(Res::Static(b)) => {
                if !dims_same(b, &d) {
                    let extra = match (b, &d) {
                        (Dim::Const(_), Dim::Var(_) | Dim::Expr(_))
                        | (Dim::Var(_) | Dim::Expr(_), Dim::Const(_)) => {
                            " (a generic or runtime dimension cannot be assumed equal to a \
                             different extent)"
                        }
                        _ => "",
                    };
                    self.error(
                        span,
                        "E0502",
                        format!(
                            "dimension mismatch on `{}` of `{builtin}` ({what}): expected {}, found \
                             {}{extra}",
                            spec.dims[idx],
                            self.builtin_dim_str(b),
                            self.builtin_dim_str(&d)
                        ),
                    );
                }
            }
            // Slices bind last, after every static source; nothing static reaches here afterwards.
            Some(Res::Slice { .. }) => {}
        }
    }

    /// Classify buffer argument `arg_ty` of `builtin` (parameter `pname`), reporting a non-`f32`
    /// element (`E0502`), a non-contiguous tensor (`E0502`) or a non-buffer (`E0401`).
    fn classify_builtin_operand(
        &mut self,
        arg_ty: &Ty,
        builtin: &str,
        pname: &str,
        span: Span,
    ) -> Operand {
        match arg_ty {
            Ty::Unknown | Ty::Error => Operand::Skip,
            Ty::Tensor {
                elem,
                shape,
                layout,
            } => {
                let mut ok = true;
                if *elem != Scalar::F32 {
                    self.error(
                        span,
                        "E0502",
                        format!(
                            "tensor element type mismatch: operand `{pname}` of `{builtin}` must \
                             hold `f32`, found `{}`",
                            elem.name()
                        ),
                    );
                    ok = false;
                }
                if *layout != Layout::Contiguous {
                    self.error(
                        span,
                        "E0502",
                        format!(
                            "tensor layout mismatch: operand `{pname}` of `{builtin}` must be \
                             contiguous (row-major)"
                        ),
                    );
                    ok = false;
                }
                if ok {
                    Operand::Tensor(shape.0.clone())
                } else {
                    Operand::Skip
                }
            }
            Ty::Array { .. } => match flat_f32_len(arg_ty) {
                Some(n) => Operand::Array(n),
                None => {
                    let leaf = array_leaf(arg_ty);
                    if !matches!(leaf, Ty::Unknown | Ty::Error) {
                        self.error(
                            span,
                            "E0502",
                            format!(
                                "element type mismatch: operand `{pname}` of `{builtin}` must hold \
                                 `f32`, found an array of `{}`",
                                leaf.display(self.interner)
                            ),
                        );
                    }
                    Operand::Skip
                }
            },
            Ty::Slice(elem) => match elem.as_ref() {
                Ty::Scalar(Scalar::F32) => Operand::Slice,
                Ty::Unknown | Ty::Error => Operand::Skip,
                other => {
                    self.error(
                        span,
                        "E0502",
                        format!(
                            "element type mismatch: operand `{pname}` of `{builtin}` must hold \
                             `f32`, found a slice of `{}`",
                            other.display(self.interner)
                        ),
                    );
                    Operand::Skip
                }
            },
            other => {
                self.error(
                    span,
                    "E0401",
                    format!(
                        "operand `{pname}` of `{builtin}` must be an `f32` buffer (a \
                         `Tensor[f32, ..]`, an `[f32; N]` array or a `[]f32` slice), found a value \
                         of type `{}`",
                        other.display(self.interner)
                    ),
                );
                Operand::Skip
            }
        }
    }

    /// Shape-check a call to one of the table builtins and record its resolved dimensions.
    fn check_kernel_call(
        &mut self,
        call: &Expr,
        kind: KernelBuiltin,
        generic_args: &[TypeExpr],
        args: &[Expr],
    ) -> Ty {
        let spec = spec_of(kind);
        let builtin = builtin_name(kind);
        let span = call.span;
        let arg_tys: Vec<Ty> = args.iter().map(|a| self.type_expr(a)).collect();
        let ret = if spec.returns_f32 {
            Ty::Scalar(Scalar::F32)
        } else {
            Ty::Unit
        };
        if args.len() != spec.params.len() {
            let names: Vec<&str> = spec
                .params
                .iter()
                .map(|p| match p {
                    Param::Buf { name, .. } | Param::Float(name) => *name,
                })
                .collect();
            self.error(
                span,
                "E0503",
                format!(
                    "`{builtin}` takes {} argument(s) ({}), but {} were supplied",
                    spec.params.len(),
                    names.join(", "),
                    args.len()
                ),
            );
            return ret;
        }
        let errors_before = self.diags.len();
        let mut state: Vec<Option<Res>> = vec![None; spec.dims.len()];
        // Turbofish variables of an unsigned integer type (they zero-extend to the kernel's i64).
        let mut unsigned_names: Vec<Symbol> = Vec::new();

        // (1) An explicit turbofish, positional over the dimension list.
        if !generic_args.is_empty() {
            if generic_args.len() != spec.dims.len() {
                self.error(
                    span,
                    "E0503",
                    format!(
                        "`{builtin}` takes {} dimension argument(s) `<{}>`, but {} were supplied",
                        spec.dims.len(),
                        spec.dims.join(", "),
                        generic_args.len()
                    ),
                );
                return ret;
            }
            for (i, ga) in generic_args.iter().enumerate() {
                if let Some((d, unsigned)) = self.builtin_turbofish_dim(ga, builtin) {
                    if let (Dim::Var(name), true) = (&d, unsigned) {
                        unsigned_names.push(*name);
                    }
                    self.bind_builtin_dim(&mut state, spec, builtin, i, d, "turbofish", ga.span);
                }
            }
        }

        // Classify every buffer operand once.
        let mut operands: Vec<Option<Operand>> = Vec::with_capacity(args.len());
        for (i, p) in spec.params.iter().enumerate() {
            match p {
                Param::Buf { name, .. } => {
                    let op =
                        self.classify_builtin_operand(&arg_tys[i], builtin, name, args[i].span);
                    operands.push(Some(op));
                }
                Param::Float(name) => {
                    match &arg_tys[i] {
                        Ty::Scalar(s) if s.is_int() || s.is_float() => {}
                        Ty::Unknown | Ty::Error => {}
                        other => self.error(
                            args[i].span,
                            "E0401",
                            format!(
                                "`{name}` of `{builtin}` must be a numeric scalar, found a value \
                                 of type `{}`",
                                other.display(self.interner)
                            ),
                        ),
                    }
                    operands.push(None);
                }
            }
        }
        if let Some(o) = builtin_output(kind) {
            self.check_builtin_output_mutable(&args[o], builtin);
        }

        // (2) Tensor operands: unify their shapes axis by axis, like a generic function's call.
        for (i, p) in spec.params.iter().enumerate() {
            let (Param::Buf { name, dims }, Some(Operand::Tensor(shape))) = (p, &operands[i])
            else {
                continue;
            };
            let shape = shape.clone();
            let pr = dims.len();
            let ar = shape.len();
            if ar == pr {
                for (axis, (&di, d)) in dims.iter().zip(&shape).enumerate() {
                    let what = format!("axis {axis} of `{name}`");
                    let d = d.clone();
                    self.bind_builtin_dim(&mut state, spec, builtin, di, d, &what, args[i].span);
                }
            } else if spec.flat && pr == 1 {
                // A flat operation reads a tensor of any rank as its element count.
                let mut count: Option<u64> = Some(1);
                for d in &shape {
                    count = match (count, d) {
                        (Some(c), Dim::Const(n)) => c.checked_mul(*n),
                        _ => None,
                    };
                }
                match count {
                    Some(n) => {
                        let what = format!("element count of `{name}`");
                        self.bind_builtin_dim(
                            &mut state,
                            spec,
                            builtin,
                            dims[0],
                            Dim::Const(n),
                            &what,
                            args[i].span,
                        );
                    }
                    None => {
                        let found = self.builtin_shape_str(&shape);
                        self.error(
                            args[i].span,
                            "E0501",
                            format!(
                                "tensor rank mismatch: operand `{name}` of `{builtin}` must be \
                                 rank 1 or have constant dimensions, found rank {ar} {found}"
                            ),
                        );
                    }
                }
            } else if spec.rows.is_some_and(|r| dims[0] == r) && pr == 2 && ar == 1 {
                // A row-wise operation reads a vector as one row.
                let what = format!("axis 0 of `{name}`");
                self.bind_builtin_dim(
                    &mut state,
                    spec,
                    builtin,
                    dims[0],
                    Dim::Const(1),
                    "a vector operand is one row",
                    args[i].span,
                );
                self.bind_builtin_dim(
                    &mut state,
                    spec,
                    builtin,
                    dims[1],
                    shape[0].clone(),
                    &what,
                    args[i].span,
                );
            } else {
                let expected: Vec<&str> = dims.iter().map(|&d| spec.dims[d]).collect();
                let found = self.builtin_shape_str(&shape);
                self.error(
                    args[i].span,
                    "E0501",
                    format!(
                        "tensor rank mismatch: operand `{name}` of `{builtin}` expects rank {pr} \
                         [{}], found rank {ar} {found}",
                        expected.join(", ")
                    ),
                );
            }
        }

        // (3)/(4) Array lengths fix the one free dimension of their shape; then a row-wise
        // operation's still-free row count defaults to 1; then arrays again.
        self.resolve_builtin_arrays(&mut state, spec, builtin, &operands, args);
        if let Some(r) = spec.rows {
            if state[r].is_none() {
                state[r] = Some(Res::Static(Dim::Const(1)));
                self.resolve_builtin_arrays(&mut state, spec, builtin, &operands, args);
            }
        }

        // (5) A slice's runtime length fixes the one free dimension of its shape, when the others
        // are constants.
        loop {
            let mut progress = false;
            for (i, p) in spec.params.iter().enumerate() {
                let (Param::Buf { dims, .. }, Some(Operand::Slice)) = (p, &operands[i]) else {
                    continue;
                };
                let free: Vec<usize> = dims
                    .iter()
                    .copied()
                    .filter(|&d| state[d].is_none())
                    .collect();
                if free.len() != 1 {
                    continue;
                }
                let mut div: Option<u64> = Some(1);
                for &d in dims.iter() {
                    if d == free[0] {
                        continue;
                    }
                    div = match (div, &state[d]) {
                        (Some(acc), Some(Res::Static(Dim::Const(n)))) => acc.checked_mul(*n),
                        _ => None,
                    };
                }
                if let Some(div) = div.filter(|&v| v > 0) {
                    state[free[0]] = Some(Res::Slice { arg: i, div });
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }

        // (6) Every dimension must now be fixed.
        let mut unresolved = false;
        for (i, st) in state.iter().enumerate() {
            if st.is_none() {
                unresolved = true;
                self.error(
                    span,
                    "E0409",
                    format!(
                        "cannot infer dimension `{}` of `{builtin}`: no tensor shape, array length \
                         or slice fixes it; supply it explicitly, e.g. `{builtin}::<{}>(…)`",
                        spec.dims[i],
                        spec.dims.join(", ")
                    ),
                );
            }
        }
        if unresolved {
            return ret;
        }

        // (7) A fully static array must hold exactly the elements its shape names.
        for (i, p) in spec.params.iter().enumerate() {
            let (Param::Buf { name, dims }, Some(Operand::Array(len))) = (p, &operands[i]) else {
                continue;
            };
            let mut need: Option<u64> = Some(1);
            let mut shape = Vec::with_capacity(dims.len());
            for &d in dims.iter() {
                match &state[d] {
                    Some(Res::Static(Dim::Const(n))) => {
                        need = need.and_then(|acc| acc.checked_mul(*n));
                        shape.push(Dim::Const(*n));
                    }
                    _ => need = None,
                }
            }
            if shape.len() != dims.len() {
                // A runtime dimension: the lowering checks the length at run time.
                continue;
            }
            if need != Some(*len) {
                let shape_s = self.builtin_shape_str(&shape);
                let need_s =
                    need.map_or_else(|| "more than u64::MAX".to_string(), |n| n.to_string());
                self.error(
                    args[i].span,
                    "E0501",
                    format!(
                        "extent mismatch: operand `{name}` of `{builtin}` is an array of {len} \
                         element(s), but its shape {shape_s} holds {need_s}"
                    ),
                );
            }
        }

        if self.diags[errors_before..].iter().any(|d| d.is_error()) {
            return ret;
        }
        let dims = state
            .iter()
            .map(|st| match st {
                Some(Res::Static(Dim::Const(n))) => BuiltinDim::Const(*n),
                Some(Res::Static(Dim::Var(s))) => BuiltinDim::Named {
                    name: *s,
                    signed: !unsigned_names.contains(s),
                },
                Some(Res::Slice { arg, div }) => BuiltinDim::SliceLen {
                    arg: *arg,
                    div: *div,
                },
                Some(Res::Static(d @ Dim::Expr(_))) => BuiltinDim::Derived(d.clone()),
                // Unreachable: Dynamic is never bound and `None` returned above.
                _ => BuiltinDim::Const(0),
            })
            .collect();
        self.builtin_calls
            .insert(call.id, BuiltinCall { kind, dims });
        ret
    }

    /// Steps (3)/(4) of [`Self::check_kernel_call`]: to a fixed point, an array operand whose shape
    /// has exactly one free dimension (every other one a constant) fixes it to `len / product`.
    fn resolve_builtin_arrays(
        &mut self,
        state: &mut [Option<Res>],
        spec: &Spec,
        builtin: &str,
        operands: &[Option<Operand>],
        args: &[Expr],
    ) {
        loop {
            let mut progress = false;
            for (i, p) in spec.params.iter().enumerate() {
                let (Param::Buf { name, dims }, Some(Operand::Array(len))) = (p, &operands[i])
                else {
                    continue;
                };
                let free: Vec<usize> = dims
                    .iter()
                    .copied()
                    .filter(|&d| state[d].is_none())
                    .collect();
                if free.len() != 1 {
                    continue;
                }
                let mut prod: Option<u64> = Some(1);
                for &d in dims.iter() {
                    if d == free[0] {
                        continue;
                    }
                    prod = match (prod, &state[d]) {
                        (Some(acc), Some(Res::Static(Dim::Const(n)))) => acc.checked_mul(*n),
                        _ => None,
                    };
                }
                let Some(prod) = prod else { continue };
                if prod == 0 {
                    continue;
                }
                let len = *len;
                if !len.is_multiple_of(prod) {
                    self.error(
                        args[i].span,
                        "E0501",
                        format!(
                            "extent mismatch: operand `{name}` of `{builtin}` is an array of {len} \
                             element(s), which is not a whole number of rows of {prod} (dimension \
                             `{}` cannot be inferred from it)",
                            spec.dims[free[0]]
                        ),
                    );
                }
                // Pinned even after an error, so it is reported once and nothing re-derives it.
                state[free[0]] = Some(Res::Static(Dim::Const(len / prod)));
                progress = true;
            }
            if !progress {
                break;
            }
        }
    }

    /// `sdpa(q, k, v, out, s, d, scale, causal)` — the fused attention builtin's signature: four
    /// `f32` buffers (each a `Tensor[f32, ..]`, an `[f32; N]` array or a `[]f32` slice, the last two
    /// possibly behind one reference), integer `s`/`d`, a numeric `scale` and an integer or `bool`
    /// `causal`.
    /// When `s` and `d` are compile-time constants, every operand whose extent its type pins must
    /// hold `s·d` elements (`E0501`), and a rank-2 tensor operand must be `[s, d]` (`E0502`); rank-2
    /// tensor operands must agree with each other either way.
    fn check_sdpa(&mut self, call: &Expr, generic_args: &[TypeExpr], args: &[Expr]) -> Ty {
        let span = call.span;
        let arg_tys: Vec<Ty> = args.iter().map(|a| self.type_expr(a)).collect();
        if !generic_args.is_empty() {
            self.error(
                span,
                "E0503",
                "`sdpa` takes no generic arguments; its extents are the `s` and `d` arguments"
                    .to_string(),
            );
        }
        if args.len() != 8 {
            self.error(
                span,
                "E0503",
                format!(
                    "`sdpa` takes 8 arguments (q, k, v, out, s, d, scale, causal), but {} were \
                     supplied",
                    args.len()
                ),
            );
            return Ty::Unit;
        }
        let errors_before = self.diags.len();
        self.check_builtin_output_mutable(&args[3], "sdpa");
        const NAMES: [&str; 4] = ["q", "k", "v", "out"];
        // Each operand's static extent: `Some(Some(n))` an array of n elements, `Some(None)` a
        // runtime extent (a slice) — tensors are kept separately with their shape.
        let mut arrays: Vec<(usize, u64)> = Vec::new();
        let mut tensors: Vec<(usize, Vec<Dim>)> = Vec::new();
        for (i, name) in NAMES.iter().enumerate() {
            // One level of `&`/`*` is looked through, and only to an array or a slice: the lowering
            // reads an array's base and a slice's fat pointer through it, which is all a reference
            // to either holds. A reference to a tensor, or a pointer to a pointer, is refused here
            // rather than handed to the kernel as if it were the data.
            let (t, behind_ref) = match &arg_tys[i] {
                Ty::Ref { pointee, .. } | Ty::Ptr { pointee, .. } => (&**pointee, true),
                other => (other, false),
            };
            if behind_ref && !matches!(t, Ty::Array { .. } | Ty::Slice(_) | Ty::Unknown | Ty::Error)
            {
                self.error(
                    args[i].span,
                    "E0401",
                    format!(
                        "operand `{name}` of `sdpa` must be an `f32` buffer (a `Tensor[f32, ..]`, \
                         an `[f32; N]` array, a `[]f32` slice, or a reference to an array or slice), \
                         found a value of type `{}`",
                        arg_tys[i].display(self.interner)
                    ),
                );
                continue;
            }
            match self.classify_builtin_operand(t, "sdpa", name, args[i].span) {
                Operand::Tensor(shape) => tensors.push((i, shape)),
                Operand::Array(n) => arrays.push((i, n)),
                Operand::Slice | Operand::Skip => {}
            }
        }
        for (i, what) in [(4usize, "s"), (5, "d")] {
            match &arg_tys[i] {
                Ty::Scalar(s) if s.is_int() => {}
                Ty::Unknown | Ty::Error => {}
                other => self.error(
                    args[i].span,
                    "E0401",
                    format!(
                        "`{what}` of `sdpa` must be an integer extent, found a value of type `{}`",
                        other.display(self.interner)
                    ),
                ),
            }
        }
        match &arg_tys[6] {
            Ty::Scalar(s) if s.is_int() || s.is_float() => {}
            Ty::Unknown | Ty::Error => {}
            other => self.error(
                args[6].span,
                "E0401",
                format!(
                    "`scale` of `sdpa` must be a numeric scalar, found a value of type `{}`",
                    other.display(self.interner)
                ),
            ),
        }
        match &arg_tys[7] {
            Ty::Scalar(s) if s.is_int() || *s == Scalar::Bool => {}
            Ty::Unknown | Ty::Error => {}
            other => self.error(
                args[7].span,
                "E0401",
                format!(
                    "`causal` of `sdpa` must be an integer or `bool` flag, found a value of type \
                     `{}`",
                    other.display(self.interner)
                ),
            ),
        }
        let s = self
            .builtin_const_int(&args[4], 0)
            .and_then(|v| u64::try_from(v).ok());
        let d = self
            .builtin_const_int(&args[5], 0)
            .and_then(|v| u64::try_from(v).ok());
        // Rank-2 tensor operands are `[S, D]`: they agree with each other and with constant s/d.
        let mut sd: [Option<(Dim, usize)>; 2] = [None, None];
        for (i, shape) in &tensors {
            if shape.len() != 2 {
                continue;
            }
            for (axis, dim) in shape.iter().enumerate() {
                if matches!(dim, Dim::Dynamic) {
                    continue;
                }
                let which = if axis == 0 { "S" } else { "D" };
                match &sd[axis] {
                    None => sd[axis] = Some((dim.clone(), *i)),
                    Some((b, j)) => {
                        let j = *j;
                        if !dims_same(b, dim) {
                            self.error(
                                args[*i].span,
                                "E0502",
                                format!(
                                    "dimension mismatch on `{which}` (axis {axis}) of `sdpa`: \
                                     operand `{}` has {}, operand `{}` has {}",
                                    NAMES[j],
                                    self.builtin_dim_str(b),
                                    NAMES[*i],
                                    self.builtin_dim_str(dim)
                                ),
                            );
                        }
                    }
                }
                let value = if axis == 0 { s } else { d };
                if let (Some(v), Dim::Const(n)) = (value, dim) {
                    if v != *n {
                        let arg = if axis == 0 { "s" } else { "d" };
                        self.error(
                            args[*i].span,
                            "E0502",
                            format!(
                                "dimension mismatch on `{which}` (axis {axis}) of `sdpa`: the call \
                                 passes {arg} = {v}, but operand `{}` has extent {n}",
                                NAMES[*i]
                            ),
                        );
                    }
                }
            }
        }
        if let (Some(s), Some(d)) = (s, d) {
            let need = s.checked_mul(d);
            let mut statics: Vec<(usize, u64)> = arrays.clone();
            for (i, shape) in &tensors {
                let mut n: Option<u64> = Some(1);
                for dim in shape {
                    n = match (n, dim) {
                        (Some(acc), Dim::Const(x)) => acc.checked_mul(*x),
                        _ => None,
                    };
                }
                if let Some(n) = n {
                    statics.push((*i, n));
                }
            }
            for (i, len) in statics {
                let short = match need {
                    Some(need) => len < need,
                    None => true,
                };
                if short {
                    let need_s =
                        need.map_or_else(|| "more than u64::MAX".to_string(), |n| n.to_string());
                    self.error(
                        args[i].span,
                        "E0501",
                        format!(
                            "extent mismatch: operand `{}` of `sdpa` holds {len} element(s), but \
                             the kernel reads s·d = {s}·{d} = {need_s}",
                            NAMES[i]
                        ),
                    );
                }
            }
        }
        if !self.diags[errors_before..].iter().any(|d| d.is_error()) {
            self.builtin_calls.insert(
                call.id,
                BuiltinCall {
                    kind: KernelBuiltin::Sdpa,
                    dims: Vec::new(),
                },
            );
        }
        Ty::Unit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::{Interner, SourceId};

    /// Check `src`, returning the recorded builtin calls (in source order of their node ids) and
    /// the error codes.
    fn analyze(src: &str) -> (Vec<BuiltinCall>, Vec<&'static str>) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        let (res, diags) = crate::check(&module, &interner);
        let mut calls: Vec<(wukong_ast::NodeId, BuiltinCall)> =
            res.builtin_calls.into_iter().collect();
        calls.sort_by_key(|(id, _)| *id);
        let codes = diags
            .iter()
            .filter(|d| d.is_error())
            .filter_map(|d| d.code)
            .collect();
        (calls.into_iter().map(|(_, c)| c).collect(), codes)
    }

    fn consts(c: &BuiltinCall) -> Vec<u64> {
        c.dims
            .iter()
            .map(|d| match d {
                BuiltinDim::Const(n) => *n,
                other => panic!("expected a constant dimension, found {other:?}"),
            })
            .collect()
    }

    #[test]
    fn tensor_operands_bind_every_dimension() {
        let (calls, codes) = analyze(
            "fn f(a: Tensor[f32, 2, 3], b: Tensor[f32, 3, 4], mut c: Tensor[f32, 2, 4]) { \
             matmul(a, b, c); }",
        );
        assert!(codes.is_empty(), "{codes:?}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].kind, KernelBuiltin::Matmul);
        assert_eq!(consts(&calls[0]), vec![2, 3, 4]);
    }

    #[test]
    fn a_contraction_conflict_is_a_dimension_mismatch() {
        let (calls, codes) = analyze(
            "fn f(a: Tensor[f32, 2, 3], b: Tensor[f32, 5, 4], mut c: Tensor[f32, 2, 4]) { \
             matmul(a, b, c); }",
        );
        assert!(codes.contains(&"E0502"), "{codes:?}");
        assert!(calls.is_empty(), "a rejected call must not be recorded");
    }

    #[test]
    fn flat_arrays_need_a_turbofish_for_a_matmul() {
        let src = "fn main() -> i32 { let a: [f32; 6] = [0.0; 6]; let b: [f32; 12] = [0.0; 12]; \
                   let mut c: [f32; 8] = [0.0; 8]; matmul(a, b, c); return 0; }";
        let (_, codes) = analyze(src);
        assert!(codes.contains(&"E0409"), "{codes:?}");
        let fixed = src.replace("matmul(a, b, c)", "matmul::<2, 3, 4>(a, b, c)");
        let (calls, codes) = analyze(&fixed);
        assert!(codes.is_empty(), "{codes:?}");
        assert_eq!(consts(&calls[0]), vec![2, 3, 4]);
    }

    #[test]
    fn a_static_array_must_hold_its_shape() {
        let (_, codes) = analyze(
            "fn main() -> i32 { let a: [f32; 6] = [0.0; 6]; let b: [f32; 12] = [0.0; 12]; \
             let mut c: [f32; 9] = [0.0; 9]; matmul::<2, 3, 4>(a, b, c); return 0; }",
        );
        assert!(codes.contains(&"E0501"), "{codes:?}");
    }

    #[test]
    fn linear_infers_its_dimensions_from_the_bias() {
        // b: [N] fixes N = 4, then w: [N, K] fixes K = 3, then x: [M, K] fixes M = 2.
        let (calls, codes) = analyze(
            "fn main() -> i32 { let x: [f32; 6] = [0.0; 6]; let w: [f32; 12] = [0.0; 12]; \
             let b: [f32; 4] = [0.0; 4]; let mut o: [f32; 8] = [0.0; 8]; linear(x, w, b, o); \
             return 0; }",
        );
        assert!(codes.is_empty(), "{codes:?}");
        assert_eq!(calls[0].kind, KernelBuiltin::Linear);
        assert_eq!(consts(&calls[0]), vec![2, 3, 4]);
    }

    #[test]
    fn a_row_wise_operation_defaults_to_one_row_and_reads_rows_from_gamma() {
        let (calls, codes) = analyze(
            "fn main() -> i32 { let x: [f32; 6] = [0.0; 6]; let mut o: [f32; 6] = [0.0; 6]; \
             softmax(x, o); \
             let g: [f32; 3] = [1.0; 3]; let bt: [f32; 3] = [0.0; 3]; \
             layernorm(x, g, bt, o, 0.00001); return 0; }",
        );
        assert!(codes.is_empty(), "{codes:?}");
        assert_eq!(calls.len(), 2);
        assert_eq!(consts(&calls[0]), vec![1, 6]);
        assert_eq!(consts(&calls[1]), vec![2, 3]);
    }

    #[test]
    fn a_slice_operand_is_a_runtime_extent() {
        let (calls, codes) = analyze("fn f(x: []f32, y: []f32) -> f32 { return dot(x, y); }");
        assert!(codes.is_empty(), "{codes:?}");
        assert_eq!(calls[0].kind, KernelBuiltin::Dot);
        assert_eq!(calls[0].dims, vec![BuiltinDim::SliceLen { arg: 0, div: 1 }]);
    }

    #[test]
    fn a_wrong_element_type_is_refused() {
        let (calls, codes) = analyze(
            "fn main() -> i32 { let x: [f64; 4] = [0.0; 4]; let mut o: [f32; 4] = [0.0; 4]; \
             gelu(x, o); return 0; }",
        );
        assert!(codes.contains(&"E0502"), "{codes:?}");
        assert!(calls.is_empty());
    }

    #[test]
    fn the_one_argument_activation_stays_the_scalar_intrinsic() {
        let (calls, codes) =
            analyze("fn main() -> i32 { let y: f32 = gelu(1.0); return y as i32; }");
        assert!(codes.is_empty(), "{codes:?}");
        assert!(calls.is_empty(), "`gelu(x)` is not the buffer builtin");
    }

    #[test]
    fn a_user_function_or_local_shadows_the_builtin() {
        let (calls, codes) = analyze(
            "fn sum(a: [f32; 2]) -> f32 { return a[0] + a[1]; } \
             fn main() -> i32 { let a: [f32; 2] = [1.0, 2.0]; return sum(a) as i32; }",
        );
        assert!(codes.is_empty(), "{codes:?}");
        assert!(calls.is_empty());
    }

    #[test]
    fn sdpa_has_a_shape_signature() {
        let ok = "fn main() -> i32 { let q: [f32; 4] = [0.0; 4]; let mut o: [f32; 4] = [0.0; 4]; \
                  sdpa(q, q, q, o, 2, 2, 1.0, 0); return 0; }";
        let (calls, codes) = analyze(ok);
        assert!(codes.is_empty(), "{codes:?}");
        assert_eq!(calls[0].kind, KernelBuiltin::Sdpa);

        let short = ok.replace(
            "let mut o: [f32; 4] = [0.0; 4]",
            "let mut o: [f32; 2] = [0.0; 2]",
        );
        assert!(analyze(&short).1.contains(&"E0501"));

        let wrong = ok.replace("let q: [f32; 4] = [0.0; 4]", "let q: [f64; 4] = [0.0; 4]");
        assert!(analyze(&wrong).1.contains(&"E0502"));

        let scalar = ok.replace("sdpa(q, q, q, o", "sdpa(1.0, q, q, o");
        assert!(analyze(&scalar).1.contains(&"E0401"));

        let tensor =
            "fn f(q: Tensor[f32, 2, 2], mut o: Tensor[f32, 3, 2]) { sdpa(q, q, q, o, 2, 2, \
                      1.0, 0); }";
        assert!(analyze(tensor).1.contains(&"E0502"));
    }
}
