//! Top-level items, declarations, and attributes.

use crate::{Block, Expr, Ident, NodeId, Path, TypeExpr};
use wukong_span::{Span, Symbol};

#[derive(Clone, Debug)]
pub struct Item {
    pub id: NodeId,
    pub attrs: Vec<Attr>,
    pub kind: ItemKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ItemKind {
    Fn(FnDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Const(ConstDecl),
    Import(Import),
    Extern(ExternBlock),
}

#[derive(Clone, Debug)]
pub struct FnDecl {
    pub name: Ident,
    pub is_pub: bool,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub ret: Option<TypeExpr>,
    /// `None` for declarations without a body (extern fns, trait signatures).
    pub body: Option<Block>,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub id: NodeId,
    pub attrs: Vec<Attr>,
    /// `mut` on the parameter (`fn f(mut p: T)`). A non-`mut` parameter may not be reassigned or
    /// mutated through a projection; an aggregate parameter is passed by reference, so `mut` opts
    /// into (caller-visible) in-place mutation. `false` for the common immutable case.
    pub mutable: bool,
    pub name: Ident,
    pub ty: TypeExpr,
    pub span: Span,
}

/// A generic parameter: a type/dimension variable, or an explicit `const N: T`.
#[derive(Clone, Debug)]
pub struct GenericParam {
    pub kind: GenericParamKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum GenericParamKind {
    /// A bare `<T>` / `<M>` — may be used as a type or, when indexed into a shape, a dimension.
    Type(Ident),
    /// An explicit `const N: usize`.
    Const { name: Ident, ty: TypeExpr },
}

#[derive(Clone, Debug)]
pub struct StructDecl {
    pub name: Ident,
    pub is_pub: bool,
    pub generics: Vec<GenericParam>,
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: Ident,
    pub is_pub: bool,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct EnumDecl {
    pub name: Ident,
    pub is_pub: bool,
    pub generics: Vec<GenericParam>,
    pub variants: Vec<Variant>,
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub name: Ident,
    pub data: VariantData,
    pub discriminant: Option<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum VariantData {
    Unit,
    Tuple(Vec<TypeExpr>),
    Struct(Vec<Field>),
}

#[derive(Clone, Debug)]
pub struct ConstDecl {
    pub name: Ident,
    pub is_pub: bool,
    pub ty: TypeExpr,
    pub value: Expr,
}

#[derive(Clone, Debug)]
pub struct Import {
    pub path: Path,
    pub alias: Option<Ident>,
    /// `import a.b.{x, y}` brings a selected set of names into scope.
    pub items: Option<Vec<Ident>>,
}

#[derive(Clone, Debug)]
pub struct ExternBlock {
    pub abi: Symbol,
    pub items: Vec<FnDecl>,
}

/// An attribute such as `@simd`, `@tile(64)`, `@parallel(grain = 1)`, or `@export("name")`.
#[derive(Clone, Debug)]
pub struct Attr {
    pub name: Ident,
    pub args: Vec<AttrArg>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum AttrArg {
    /// A bareword argument: the `restrict` in `@derive(restrict)`.
    Word(Symbol, Span),
    /// An integer argument: the `64` in `@tile(64)`.
    Int(Symbol, Span),
    /// A string argument: the `"name"` in `@export("name")`.
    Str(Symbol, Span),
    /// A `key = value` argument: the `grain = 1` in `@parallel(grain = 1)`.
    KeyValue { key: Ident, value: AttrVal },
}

#[derive(Clone, Debug)]
pub enum AttrVal {
    Int(Symbol),
    Str(Symbol),
    Word(Symbol),
    Bool(bool),
}

impl Attr {
    /// Find a `key = value` argument by key name (resolved via the interner by the caller).
    pub fn args(&self) -> &[AttrArg] {
        &self.args
    }
}
