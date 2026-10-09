//! Type syntax: the surface forms a programmer writes for types.

use crate::{BinOp, Expr, NodeId, Path};
use wukong_span::{Span, Symbol};

/// A type as written in source. Resolved to a `wukong_types::Type` during semantic analysis.
#[derive(Clone, Debug)]
pub struct TypeExpr {
    pub id: NodeId,
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TypeKind {
    /// A named type or a dimension/const-generic variable: `f32`, `M`, `MyStruct`.
    Path(Path),
    /// An integer literal used as a const-generic / dimension argument: the `512` in
    /// `matmul::<512, 512, 512>`.
    Int(Symbol),
    /// The unit type `()`.
    Unit,
    /// `*T` / `*mut T`.
    Pointer {
        mutable: bool,
        pointee: Box<TypeExpr>,
    },
    /// `&T` / `&mut T`.
    Ref {
        mutable: bool,
        pointee: Box<TypeExpr>,
    },
    /// `[]T`.
    Slice(Box<TypeExpr>),
    /// `[T; N]` or `shared [T; N]`.
    Array {
        elem: Box<TypeExpr>,
        len: Box<Expr>,
        shared: bool,
    },
    /// `(A, B, ...)`.
    Tuple(Vec<TypeExpr>),
    /// A SIMD vector type: `f32x8` or `vec[T, N]`.
    Vector { elem: Box<TypeExpr>, lanes: u32 },
    /// A shape-typed tensor: `Tensor[f32, M, N]` with an optional layout.
    Tensor {
        elem: Box<TypeExpr>,
        dims: Vec<Dim>,
        layout: Option<Layout>,
    },
    /// A function type `fn(A, B) -> R` (`ret: None` for `()`). Legal only as the type of a function
    /// parameter, which every call site must bind to a named function known at compile time.
    Fn {
        params: Vec<TypeExpr>,
        ret: Option<Box<TypeExpr>>,
    },
    /// A named type applied to arguments: `Mat<4, 8>`, `Pair<f32>` (or `Mat::<4, 8>`, the only
    /// spelling accepted as a cast target, where a bare `<` is a comparison). Only a generic type
    /// alias takes arguments; the driver's alias expansion replaces every node of this kind before
    /// sema.
    /// (`path` is boxed to keep `TypeExpr` no larger than its `Tensor` variant makes it.)
    Generic { path: Box<Path>, args: Vec<TypeArg> },
}

/// One argument of a [`TypeKind::Generic`]: a type, or a dimension the parser could only read as
/// one (`4`, `?`, `N*2`). A bare name (`M`) parses as a type and serves as either.
#[derive(Clone, Debug)]
pub enum TypeArg {
    Type(TypeExpr),
    Dim(Dim),
}

/// One dimension of a tensor type.
#[derive(Clone, Debug)]
pub struct Dim {
    pub kind: DimKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum DimKind {
    /// A compile-time constant extent: `512`.
    Int(u64),
    /// A named extent: `M`. Usually a generic dimension parameter, but sema also resolves a
    /// same-named top-level `const` to its integer value (a declared generic shadows the const).
    Named(Symbol),
    /// A dynamic (runtime) extent: `?`.
    Dynamic,
    /// A derived extent: arithmetic on dimensions, `N*2`, `H*HD`, `(L - K)/S + 1`. The parser builds
    /// it only from `+ - * / %` (and parentheses, which reuse the inner node), so `op` is always one
    /// of `Add`, `Sub`, `Mul`, `Div`, `Rem`; sema folds it into `wukong_types::Dim`'s normal form.
    Binary {
        op: BinOp,
        lhs: Box<Dim>,
        rhs: Box<Dim>,
    },
}

/// Physical memory layout of a tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    Contiguous,
    ColMajor,
    Strided,
    Tiled(Vec<u64>),
}
