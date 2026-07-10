//! `wukong_ast` — the Wukong abstract syntax tree.
//!
//! The AST is a data-only crate: the parser produces it, and semantic analysis annotates it via
//! side tables keyed by [`NodeId`] (it never rewrites the tree). Recursive node kinds live in
//! the [`expr`], [`stmt`], [`ty`], and [`item`] modules; a [`print`] pretty-printer renders a
//! stable textual form for `--emit=ast` and snapshot tests.

mod expr;
mod item;
mod stmt;
mod ty;

pub mod print;

pub use expr::*;
pub use item::*;
pub use stmt::*;
pub use ty::*;

use wukong_span::{Span, Symbol};

/// A dense identifier for an AST node, used to key semantic side tables. Allocated by the
/// parser; `DUMMY` marks synthesized nodes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const DUMMY: NodeId = NodeId(u32::MAX);
}

impl std::fmt::Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// An interned identifier with its source location.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ident {
    pub sym: Symbol,
    pub span: Span,
}

/// A dotted path such as `std.simd.f32x8` or `matmul`.
#[derive(Clone, Debug)]
pub struct Path {
    pub segments: Vec<Ident>,
    pub span: Span,
}

impl Path {
    /// A path of a single segment (the common case for plain names).
    pub fn is_single(&self) -> bool {
        self.segments.len() == 1
    }

    pub fn first(&self) -> Ident {
        self.segments[0]
    }
}

/// A parsed source file: an optional `module a.b;` header plus its items.
#[derive(Clone, Debug)]
pub struct Module {
    pub name: Option<Path>,
    pub items: Vec<Item>,
    pub span: Span,
}
