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
    /// `trait Name { fn m(self, …) -> R; … }` — an interface: a set of method signatures.
    Trait(TraitDecl),
    /// `impl Type { … }` / `impl Trait for Type { … }`. The methods are NOT stored here: the parser
    /// flattens each into an ordinary top-level [`ItemKind::Fn`] item named `Type.method` (a
    /// `.`-joined symbol no source identifier can spell), so every pass that handles functions
    /// handles methods unchanged. This item records what sema needs to resolve `x.method(..)` /
    /// `Type::method(..)` and to check a trait implementation.
    Impl(ImplDecl),
    /// `type Name = T;` / `type Mat<M, N> = Tensor[f32, M, N];` — a transparent name for a type.
    /// The driver expands every use and removes these items before sema
    /// (`wukong_driver::loader::alias`), so no later stage sees an alias.
    TypeAlias(TypeAliasDecl),
}

/// A type alias. `generics` are the alias's parameters (a type, a dimension, or a `const N: T`);
/// any trait bound written on one is ignored, since an alias is substituted, never instantiated.
#[derive(Clone, Debug)]
pub struct TypeAliasDecl {
    pub name: Ident,
    pub is_pub: bool,
    pub generics: Vec<GenericParam>,
    pub ty: TypeExpr,
}

/// A trait declaration. Each method is a signature-only [`FnDecl`]; a method that takes a receiver
/// has a first parameter named `self` typed `Self` (the parser desugars `self` / `mut self` /
/// `&self` / `&mut self`). A default body parses (and is rejected by sema).
#[derive(Clone, Debug)]
pub struct TraitDecl {
    pub name: Ident,
    pub is_pub: bool,
    pub methods: Vec<FnDecl>,
}

/// An `impl` block: the type it extends, the trait it implements (if any), and its methods.
#[derive(Clone, Debug)]
pub struct ImplDecl {
    pub self_ty: Path,
    pub trait_ref: Option<Path>,
    pub methods: Vec<ImplMethod>,
}

/// One method of an `impl` block: its name as written, and the symbol of the flattened top-level
/// function that implements it (`Type.method`).
#[derive(Clone, Debug)]
pub struct ImplMethod {
    pub name: Ident,
    pub fn_name: Symbol,
}

#[derive(Clone, Debug)]
pub struct FnDecl {
    pub name: Ident,
    pub is_pub: bool,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub ret: Option<TypeExpr>,
    /// The dimension predicates of a `where` clause, in source order: `where K % 16 == 0, M <= 4096`.
    /// Each is checked at every call that fixes the dimensions it names. (A trait bound in the same
    /// clause, `where T: Shape`, is folded into that generic parameter's `bounds` instead.)
    pub where_preds: Vec<Expr>,
    /// `None` for a declaration with no body: an `extern` fn, or any `fn` header the parser found
    /// followed by neither a `{…}` block nor `= expr;`. `Some` for both body forms — an `= expr;`
    /// body is a synthesized `Block` whose `tail` is that expression.
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
    /// Trait bounds on a type parameter: `<T: Shape + Named>`, or a `where T: Shape` clause, which
    /// the parser folds in here. Empty for an unbounded parameter.
    pub bounds: Vec<Path>,
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
    /// All arguments in source order, in whatever form each was written. This is NOT a lookup:
    /// it does not select a `key = value` pair, so `@parallel(grain = 1, restrict)` yields the
    /// `restrict` word alongside the `grain` pair and a caller that reads position 1 as "the
    /// value of grain" configures the grain from the wrong argument. Match on
    /// [`AttrArg::KeyValue`] to read a keyed argument.
    pub fn args(&self) -> &[AttrArg] {
        &self.args
    }
}
