//! Statements, blocks, and patterns.

use crate::{AssignOp, Attr, Expr, Ident, NodeId, TypeExpr};
use mercury_span::{Span, Symbol};

/// A braced block: zero or more statements and an optional trailing expression (its value).
#[derive(Clone, Debug)]
pub struct Block {
    pub id: NodeId,
    pub stmts: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub id: NodeId,
    /// Attributes attached to a statement (e.g. `@parallel @simd for ...`).
    pub attrs: Vec<Attr>,
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Let {
        pat: Pattern,
        mutable: bool,
        ty: Option<TypeExpr>,
        init: Option<Expr>,
    },
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
    Expr(Expr),
    Return(Option<Expr>),
    Break(Option<Ident>),
    Continue(Option<Ident>),
    Defer(Expr),
    While {
        label: Option<Ident>,
        cond: Expr,
        body: Block,
    },
    For {
        label: Option<Ident>,
        pat: Pattern,
        iter: ForIter,
        body: Block,
    },
    Loop {
        label: Option<Ident>,
        body: Block,
    },
}

/// The thing a `for` loop iterates over.
#[derive(Clone, Debug)]
pub enum ForIter {
    /// `start..end`, `start..=end`, optionally `step s`. An open-ended range has `end: None`.
    Range {
        start: Expr,
        end: Option<Expr>,
        inclusive: bool,
        step: Option<Expr>,
    },
    /// An arbitrary iterable expression (slice, etc.).
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub struct Pattern {
    pub id: NodeId,
    pub kind: PatKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum PatKind {
    Wildcard,
    Ident(Symbol),
    Tuple(Vec<Pattern>),
    Unit,
}
