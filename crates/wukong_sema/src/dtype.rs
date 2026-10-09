//! **Dtype-generic tensors** and the generic heap allocator.
//!
//! A tensor's element was always a concrete scalar, so `Tensor[T, M, N]` with `T` a type parameter
//! was `E0302`, and every kernel had to be written once per element type (the corpus carries
//! separate `matmul_bf16` / `linear_f16` copies) — while the `E0406` hint for a generic array length
//! recommended exactly that rejected spelling. Like Mojo's `DType` parameters and a Rust
//! `T: Float` generic, the element may now be a type parameter of the enclosing function:
//!
//! * `lower_type` gives `Tensor[T, ..]` a [`Ty::Tensor`] whose `elem_var` is `Some(T)` (its `elem`
//!   is then a placeholder). Indexing it yields `T` ([`tensor_elem_ty`]), a view keeps the
//!   parameter, and two generic elements agree only when they are the same parameter.
//! * At a call, the element parameter is bound from the argument's element
//!   ([`bind_tensor_elem`]) and substituted into the signature ([`subst_tensor_elem`]); the
//!   instantiation is re-checked at the concrete element (`instantiate.rs`).
//! * mir_build monomorphizes the function per element (`saxpy$bf16`), substituting the element
//!   into every tensor type, so each instance lowers — and its loops are recognized — exactly like
//!   the hand-written `bf16` function.
//!
//! A named kernel builtin (`matmul`, `softmax`, …) takes only `f32` buffers, and a file-I/O
//! builtin moves one fixed element type, so neither accepts a generic-element tensor: a generic
//! body cannot promise the element, and sema records those calls once for the generic body.
//!
//! `alloc::<T>(n)` is the generic heap allocator: `T` is a scalar or a type parameter, the result
//! is a zeroed `[]T` of `n` elements, and each instance lowers to the per-scalar `alloc_<T>` it
//! names. A missing or non-scalar element argument is `E0325`.

use wukong_ast::{Expr, TypeExpr};
use wukong_span::{FxHashMap as HashMap, Span, Symbol};
use wukong_types::{Scalar, Ty};

use crate::Sema;

/// The type of one element of a tensor: its scalar, or its type parameter.
pub(crate) fn tensor_elem_ty(elem: Scalar, elem_var: Option<Symbol>) -> Ty {
    match elem_var {
        Some(v) => Ty::Named(v),
        None => Ty::Scalar(elem),
    }
}

/// The scalar leaf a buffer argument holds: a tensor's element, or the innermost element of a
/// (possibly nested) fixed-size array of scalars. `None` for anything else.
fn buffer_leaf(arg: &Ty) -> Option<Ty> {
    match arg {
        Ty::Tensor { elem, elem_var, .. } => Some(tensor_elem_ty(*elem, *elem_var)),
        Ty::Array { elem, .. } => match elem.as_ref() {
            Ty::Scalar(s) => Some(Ty::Scalar(*s)),
            inner @ Ty::Array { .. } => buffer_leaf(inner),
            _ => None,
        },
        _ => None,
    }
}

/// Bind the element parameter `g` of a dtype-generic tensor parameter from the buffer argument
/// `arg` (a tensor, or an array decaying to one). A first binding wins, as for every type generic;
/// a generic caller's own element parameter binds as that parameter (`Ty::Named`).
pub(crate) fn bind_tensor_elem(
    g: Symbol,
    arg: &Ty,
    generics: &[Symbol],
    tys: &mut HashMap<Symbol, Ty>,
) {
    if !generics.contains(&g) {
        return;
    }
    if let Some(t) = buffer_leaf(arg) {
        tys.entry(g).or_insert(t);
    }
}

/// Substitute a call's type bindings into a tensor element: a bound scalar makes it concrete, a
/// binding to another type parameter renames it, and anything else (unbound, or bound to a type
/// that is not a scalar, which the instantiation check reports) leaves it as written.
pub(crate) fn subst_tensor_elem(
    elem: Scalar,
    elem_var: Option<Symbol>,
    tys: &HashMap<Symbol, Ty>,
) -> (Scalar, Option<Symbol>) {
    let Some(g) = elem_var else {
        return (elem, None);
    };
    match tys.get(&g) {
        Some(Ty::Scalar(s)) => (*s, None),
        Some(Ty::Named(n)) => (elem, Some(*n)),
        _ => (elem, Some(g)),
    }
}

/// Whether two tensor elements are the same type. A generic element is its parameter, so it equals
/// only the same parameter.
pub(crate) fn tensor_elems_agree(a: (Scalar, Option<Symbol>), b: (Scalar, Option<Symbol>)) -> bool {
    match (a.1, b.1) {
        (None, None) => a.0 == b.0,
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

impl Sema<'_> {
    /// How a tensor element reads in a diagnostic: `f32`, or its type parameter `T`.
    pub(crate) fn tensor_elem_name(&self, elem: Scalar, elem_var: Option<Symbol>) -> String {
        match elem_var {
            Some(v) => self.sym_str(v).to_string(),
            None => elem.name().to_string(),
        }
    }

    /// Type `alloc::<T>(n)`, or return `None` when `name` is not `alloc`. The element argument must
    /// be a scalar or a type parameter of the enclosing function (`E0325`), and the count an integer
    /// (`E0503` arity, `E0437` kind, like the `alloc_<T>` family).
    pub(crate) fn type_generic_alloc(
        &mut self,
        name: &str,
        generic_args: &[TypeExpr],
        args: &[Expr],
        arg_tys: &[Ty],
        span: Span,
    ) -> Option<Ty> {
        if name != "alloc" {
            return None;
        }
        if args.len() != 1 {
            self.error(
                span,
                "E0503",
                format!(
                    "`alloc` takes exactly 1 argument (the element count), but {} were supplied",
                    args.len()
                ),
            );
        } else {
            match &arg_tys[0] {
                Ty::Scalar(s) if s.is_int() => {}
                Ty::Unknown | Ty::Error => {}
                other => {
                    let shown = other.display(self.interner);
                    self.error(
                        span,
                        "E0437",
                        format!("`alloc` expects an integer element count, found a value of type `{shown}`"),
                    );
                }
            }
        }
        let [ga] = generic_args else {
            self.error(
                span,
                "E0325",
                format!(
                    "`alloc` needs exactly one element type, written `alloc::<f32>(n)` or \
                     `alloc::<T>(n)`, but {} were supplied",
                    generic_args.len()
                ),
            );
            return Some(Ty::Error);
        };
        let elem = self.lower_type(ga);
        match elem {
            Ty::Scalar(_) => Some(Ty::Slice(Box::new(elem))),
            Ty::Named(n) if self.generics.contains(&n) => Some(Ty::Slice(Box::new(elem))),
            Ty::Error => Some(Ty::Error),
            other => {
                let shown = other.display(self.interner);
                self.error(
                    ga.span,
                    "E0325",
                    format!(
                        "`alloc::<{shown}>`: the element type of a heap buffer must be a scalar or \
                         a type parameter, not `{shown}`"
                    ),
                );
                Some(Ty::Error)
            }
        }
    }
}
