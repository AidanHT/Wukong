//! Expressions.

use crate::{Block, Ident, NodeId, Path, Pattern, TypeExpr};
use mercury_span::{Span, Symbol};

#[derive(Clone, Debug)]
pub struct Expr {
    pub id: NodeId,
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    /// Integer literal, stored as raw source text (incl. any suffix); parsed in sema.
    Int(Symbol),
    /// Float literal, raw source text.
    Float(Symbol),
    /// String literal, raw source text including quotes.
    Str(Symbol),
    /// Char literal, raw source text including quotes.
    Char(Symbol),
    Bool(bool),
    /// A variable, function, or namespaced item reference (`x`, `f32x8::splat`).
    Path(Path),
    Unary {
        op: UnOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `callee::<generic_args>(args)`.
    Call {
        callee: Box<Expr>,
        generic_args: Vec<TypeExpr>,
        args: Vec<Expr>,
    },
    /// Multi-dimensional index: `a[i, j]`.
    Index {
        base: Box<Expr>,
        indices: Vec<Expr>,
    },
    /// `base.name` field access.
    Field {
        base: Box<Expr>,
        name: Ident,
    },
    /// `base.0` tuple field access.
    TupleField {
        base: Box<Expr>,
        index: u32,
    },
    Cast {
        expr: Box<Expr>,
        ty: TypeExpr,
    },
    StructLit {
        path: Path,
        fields: Vec<FieldInit>,
        rest: Option<Box<Expr>>,
    },
    ArrayLit(Vec<Expr>),
    /// `[value; count]`.
    ArrayRepeat {
        value: Box<Expr>,
        count: Box<Expr>,
    },
    TupleLit(Vec<Expr>),
    Block(Block),
    If {
        cond: Box<Expr>,
        then_branch: Block,
        else_branch: Option<Box<Expr>>,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    SizeOf(TypeExpr),
    AlignOf(TypeExpr),
}

#[derive(Clone, Debug)]
pub struct FieldInit {
    pub name: Ident,
    pub value: Expr,
}

#[derive(Clone, Debug)]
pub struct MatchArm {
    pub pat: Pattern,
    pub body: Expr,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
    Deref,
    Ref,
    RefMut,
}

impl UnOp {
    pub fn glyph(self) -> &'static str {
        match self {
            UnOp::Neg => "-",
            UnOp::Not => "!",
            UnOp::Deref => "*",
            UnOp::Ref => "&",
            UnOp::RefMut => "&mut ",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl BinOp {
    pub fn glyph(self) -> &'static str {
        use BinOp::*;
        match self {
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Rem => "%",
            BitAnd => "&",
            BitOr => "|",
            BitXor => "^",
            Shl => "<<",
            Shr => ">>",
            Eq => "==",
            Ne => "!=",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            And => "&&",
            Or => "||",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

impl AssignOp {
    pub fn glyph(self) -> &'static str {
        use AssignOp::*;
        match self {
            Assign => "=",
            Add => "+=",
            Sub => "-=",
            Mul => "*=",
            Div => "/=",
            Rem => "%=",
            BitAnd => "&=",
            BitOr => "|=",
            BitXor => "^=",
            Shl => "<<=",
            Shr => ">>=",
        }
    }
}
