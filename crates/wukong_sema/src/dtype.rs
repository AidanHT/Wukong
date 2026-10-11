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
//! **Builtins with a fixed element.** A named kernel builtin (`matmul`, `softmax`, `sdpa`, …) takes
//! only `f32` buffers, and a typed file-I/O builtin (`read_bf16`, `write_f32`, …) moves the one
//! element type its name spells. A generic body cannot promise either, so a generic-element tensor
//! operand is checked **per instantiation**, like every other `T`-dependent rule: the generic body
//! pass accepts `Tensor[T, ..]` there (and records the call once — its dimensions never depend on
//! `T`, so the one record serves every instance), and each concrete instantiation is re-checked
//! with `T` substituted (`instantiate.rs`). An instance that binds `T` to the builtin's element —
//! `f32` for a kernel builtin, the suffix type for a file-I/O one — reaches the builtin exactly
//! like the hand-written concrete function (`mm$f32` dispatches `wukong_sgemm`); any other binding
//! is `E0415` (kernel builtin) or `E0416` (typed I/O builtin), reported at the operand with the
//! instantiation that required it. A `[]T` slice or `[T; N]` array operand is still the plain
//! element mismatch: only the dtype-generic tensor is a deferred element.
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
    /// Is a concrete instantiation of a generic body being re-checked (`instantiate.rs`)? That pass
    /// runs only over a module whose generic body pass was clean, so a fixed-element builtin
    /// operand whose element is wrong HERE was a deferred `Tensor[T, ..]` the substitution made
    /// concrete — which is what turns the plain element mismatch into `E0415` / `E0416`.
    pub(crate) fn rechecking_instance(&self) -> bool {
        !self.inst_subst.is_empty()
    }

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

#[cfg(test)]
mod tests {
    use crate::{BuiltinDim, KernelBuiltin};
    use wukong_span::{Interner, SourceId};

    /// Check `src`: the recorded builtin calls' kinds and dimensions, and every error code.
    fn check(src: &str) -> (Vec<(KernelBuiltin, Vec<BuiltinDim>)>, Vec<&'static str>) {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (res, diags) = crate::check(&module, &interner);
        let calls = res
            .builtin_calls
            .values()
            .map(|c| (c.kind, c.dims.clone()))
            .collect();
        let codes = diags
            .iter()
            .filter(|d| d.is_error())
            .filter_map(|d| d.code)
            .collect();
        (calls, codes)
    }

    /// `mm<T>` over `Tensor[T, 4, 4]` calling `matmul`, instantiated from arrays of `elem`.
    fn generic_matmul(elem: &str) -> String {
        format!(
            "fn mm<T>(a: Tensor[T, 4, 4], b: Tensor[T, 4, 4], mut c: Tensor[T, 4, 4]) {{ \
             matmul(a, b, c); }} \
             fn main() -> i32 {{ let a: [{elem}; 16] = [1.0; 16]; let b: [{elem}; 16] = [1.0; 16]; \
             let mut c: [{elem}; 16] = [0.0; 16]; mm(a, b, c); return 0; }}"
        )
    }

    /// N18: the generic body's kernel-builtin call is shape-checked and recorded once — its
    /// dimensions come from the tensor shapes, never from `T` — and the `T = f32` instance is clean.
    #[test]
    fn a_dtype_generic_tensor_reaches_a_kernel_builtin_at_f32() {
        let (calls, codes) = check(&generic_matmul("f32"));
        assert!(codes.is_empty(), "{codes:?}");
        let want = vec![BuiltinDim::Const(4); 3];
        assert_eq!(calls, vec![(KernelBuiltin::Matmul, want)]);
        // A generic that is never instantiated has nothing to re-check.
        let (calls, codes) = check(
            "fn mm<T>(a: Tensor[T, 4, 4], b: Tensor[T, 4, 4], mut c: Tensor[T, 4, 4]) { \
             matmul(a, b, c); } fn main() -> i32 { return 0; }",
        );
        assert!(codes.is_empty(), "{codes:?}");
        assert_eq!(calls.len(), 1);
    }

    /// Any other element is `E0415`, once per offending operand and with no `E0409` cascade (the
    /// operands' shapes are still known).
    #[test]
    fn a_dtype_generic_kernel_operand_at_another_element_is_e0415() {
        for elem in ["f64", "bf16", "f16"] {
            let (_, codes) = check(&generic_matmul(elem));
            assert_eq!(codes, ["E0415"; 3], "T = {elem}");
        }
        // `sdpa` follows the same rule.
        let sdpa = |elem: &str| {
            format!(
                "fn attn<T>(q: Tensor[T, 2, 2], mut o: Tensor[T, 2, 2]) {{ \
                 sdpa(q, q, q, o, 2, 2, 0.5, 0); }} \
                 fn main() -> i32 {{ let q: [{elem}; 4] = [1.0; 4]; \
                 let mut o: [{elem}; 4] = [0.0; 4]; attn(q, o); return 0; }}"
            )
        };
        assert!(check(&sdpa("f32")).1.is_empty());
        assert_eq!(check(&sdpa("f64")).1, ["E0415"; 4]);
    }

    /// Only the dtype-generic TENSOR is a deferred element: a `[]T` slice operand is still the plain
    /// element mismatch of the generic body.
    #[test]
    fn a_dtype_generic_slice_operand_is_still_a_plain_mismatch() {
        let (_, codes) = check(
            "fn f<T>(x: []T, mut o: [f32; 4]) { relu(x, o); } \
             fn main() -> i32 { let a: [f32; 4] = [1.0; 4]; let s: []f32 = a; \
             let mut o: [f32; 4] = [0.0; 4]; f(s, o); return 0; }",
        );
        assert!(codes.contains(&"E0502"), "{codes:?}");
        assert!(!codes.contains(&"E0415"), "{codes:?}");
    }

    /// The typed I/O builtins: a bound `T` that one of them moves reaches it; any other binding is
    /// `E0416`.
    #[test]
    fn a_dtype_generic_tensor_reaches_the_typed_io_builtin_of_its_element() {
        let io = |builtin: &str, elem: &str| {
            format!(
                "fn save<T>(x: Tensor[T, 4]) -> i64 {{ return {builtin}(\"t.bin\", x); }} \
                 fn main() -> i32 {{ let x: [{elem}; 4] = [1.0; 4]; return save(x) as i32; }}"
            )
        };
        assert!(check(&io("write_f32", "f32")).1.is_empty());
        assert!(check(&io("write_bf16", "bf16")).1.is_empty());
        assert_eq!(check(&io("write_f32", "f64")).1, ["E0416"]);
        assert_eq!(check(&io("write_bf16", "f32")).1, ["E0416"]);
        // A read writes its buffer, so the generic parameter is `mut`.
        let read = "fn load<T>(mut y: Tensor[T, 4]) -> i64 { return read_f32(\"t.bin\", y); } \
                    fn main() -> i32 { let mut y: [f32; 4] = [0.0; 4]; return load(y) as i32; }";
        assert!(check(read).1.is_empty());
    }
}
