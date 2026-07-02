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
    /// `[label:] loop { body }` — a loop as a value-producing expression. Its value comes from
    /// `break <expr>` (the loop's type is the join of every break value; a break-less loop is
    /// infinite and has type unit). A statement-position `loop` is this same node wrapped in
    /// `StmtKind::Expr`, and `while`/`for` remain statement-only (`StmtKind`).
    Loop {
        label: Option<Ident>,
        body: Block,
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
    /// An optional `if <expr>` guard: the arm matches only when the pattern fits *and* the guard is
    /// true. The guard may reference an `Ident` pattern's binding.
    pub guard: Option<Expr>,
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

    /// Fold this op as a compile-time array length: `l OP r` in u64. Shared by sema's `eval_usize`
    /// and mir_build's `const_usize_expr` so the two CANNOT disagree on a length — a slot-size vs
    /// bounds-check desync reads out of bounds (native segfault / interp != native). Division/shift by
    /// zero, an over-wide shift, and a non-arithmetic op fold to 0 (an invalid length rejected
    /// downstream) rather than panicking.
    pub fn fold_const_len(self, l: u64, r: u64) -> u64 {
        use BinOp::*;
        match self {
            Add => l.wrapping_add(r),
            Sub => l.wrapping_sub(r),
            Mul => l.wrapping_mul(r),
            Div => {
                if r != 0 {
                    l / r
                } else {
                    0
                }
            }
            Rem => {
                if r != 0 {
                    l % r
                } else {
                    0
                }
            }
            Shl => {
                if r < 64 {
                    l.wrapping_shl(r as u32)
                } else {
                    0
                }
            }
            Shr => {
                if r < 64 {
                    l.wrapping_shr(r as u32)
                } else {
                    0
                }
            }
            BitAnd => l & r,
            BitOr => l | r,
            BitXor => l ^ r,
            Eq | Ne | Lt | Le | Gt | Ge | And | Or => 0,
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
