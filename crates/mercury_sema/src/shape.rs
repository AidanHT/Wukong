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
    match name {
        "sqrt" | "rsqrt" | "abs" | "round" | "floor" | "ceil" | "trunc" | "exp" | "log" | "pow"
        | "exp2" | "log2" | "sinh" | "cosh"
        | "erf" | "sin" | "cos" | "tanh" | "sigmoid" | "silu" | "gelu" | "elu" | "leaky_relu"
        | "softplus" | "mish" | "selu" | "tanhshrink" | "hardsigmoid" | "hardswish" | "fmax"
        | "fmin" => Some(float_ty),
        _ => None,
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
        let arg_tys: Vec<Ty> = args.iter().map(|a| self.type_expr(a)).collect();

        if let ExprKind::Path(p) = &callee.kind {
            if p.is_single() {
                let name = p.first().sym;
                if let Some(def) = self.defs.lookup(name) {
                    if let DefKind::Fn(sig) = &def.kind {
                        let sig = sig.clone();
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
                    return ret;
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
            self.unify(&p, arg, &mut dims, span);
        }

        apply_subst(&sig.ret, &dims, &tys)
    }

    fn unify(&mut self, param: &Ty, arg: &Ty, dims: &mut HashMap<Symbol, Dim>, span: Span) {
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
                    self.unify_dim(*pd, *ad, dims, span);
                }
            }
            (Ty::Ptr { pointee: pp, .. }, Ty::Ptr { pointee: ap, .. })
            | (Ty::Ref { pointee: pp, .. }, Ty::Ref { pointee: ap, .. }) => {
                self.unify(pp, ap, dims, span)
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
            // Named generic type variable, or anything else: stay lenient.
            _ => {}
        }
    }

    fn unify_dim(&mut self, pd: Dim, ad: Dim, dims: &mut HashMap<Symbol, Dim>, span: Span) {
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
                    dims.insert(v, ad);
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

    pub(crate) fn type_index(&mut self, base: &Expr, indices: &[Expr], span: Span) -> Ty {
        let base_ty = self.type_expr(base);
        for ix in indices {
            self.type_expr(ix);
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
                }
                Ty::Scalar(elem)
            }
            Ty::Slice(e) | Ty::Array { elem: e, .. } | Ty::Ptr { pointee: e, .. } => *e,
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
