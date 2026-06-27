//! Statements, blocks, and patterns.

use crate::{AssignOp, Attr, Expr, Ident, NodeId, Path, TypeExpr};
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
    /// An integer (or other numeric) literal pattern, e.g. `1` / `-3` in a `match` arm. The raw
    /// source text is stored (like `ExprKind::Int`) and parsed later; a leading `-` is folded in by
    /// the parser (`neg: true`), since a literal pattern has no sub-expressions to negate.
    Int { sym: Symbol, neg: bool },
    /// A boolean literal pattern (`true` / `false`).
    Bool(bool),
    /// An or-pattern `A | B | C` — matches if any alternative matches. Alternatives are typically
    /// literals / enum variants (binding-free); a binding inside one is not recommended.
    Or(Vec<Pattern>),
    /// A path pattern — an enum variant such as `Color::Red`, matched by its integer discriminant.
    Path(Path),
    /// A range pattern `lo..hi` (half-open) or `lo..=hi` (inclusive). The bounds are integer-literal
    /// patterns (`PatKind::Int`); the scrutinee matches when it falls within the range.
    Range {
        lo: Box<Pattern>,
        hi: Box<Pattern>,
        inclusive: bool,
    },
}
