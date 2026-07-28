//! Expressions.

use crate::{Block, Ident, NodeId, Path, Pattern, TypeExpr};
use wukong_span::{Span, Symbol};

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

/// The largest const array length the compiler can carry end to end: `mir_build` stores a slot
/// length in a `u32`, so any longer length is truncated on the way to MIR while sema's bounds
/// check still believes the full value — the exact slot-size vs bounds-check desync
/// [`BinOp::fold_const_len`] exists to prevent. Lengths above this fold to the invalid-length
/// sentinel `0`, which sema already rejects (`E0501`) at the first index.
pub const MAX_CONST_ARRAY_LEN: u64 = u32::MAX as u64;

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

#[cfg(test)]
mod tests {
    use super::*;

    /// In-range lengths are unchanged — the folder is only tightened at the representable edge.
    #[test]
    fn fold_const_len_in_range_is_unchanged() {
        assert_eq!(BinOp::Add.fold_const_len(2, 2), 4);
        assert_eq!(BinOp::Sub.fold_const_len(8, 3), 5);
        assert_eq!(BinOp::Mul.fold_const_len(4, 16), 64);
        assert_eq!(BinOp::Div.fold_const_len(64, 8), 8);
        assert_eq!(BinOp::Rem.fold_const_len(65, 8), 1);
        assert_eq!(BinOp::Shl.fold_const_len(1, 10), 1024);
        assert_eq!(BinOp::Shr.fold_const_len(1024, 2), 256);
        assert_eq!(BinOp::BitOr.fold_const_len(8, 1), 9);
        assert_eq!(
            BinOp::Add.fold_const_len(MAX_CONST_ARRAY_LEN - 1, 1),
            MAX_CONST_ARRAY_LEN
        );
    }

    /// A length that wraps in u64 (`[i32; 2 - 5]`) must not be handed on as a colossal positive
    /// length: sema then believes every index is in bounds while mir_build allocas the low 32 bits.
    #[test]
    fn fold_const_len_declines_wrapping() {
        assert_eq!(BinOp::Sub.fold_const_len(2, 5), 0);
        assert_eq!(BinOp::Add.fold_const_len(u64::MAX, 2), 0);
        assert_eq!(BinOp::Mul.fold_const_len(u64::MAX, 2), 0);
    }

    /// A length above the MIR slot width (`[i32; 4294967296 + 4]`) is unrepresentable downstream,
    /// so it must fold to the invalid-length sentinel rather than to a value mir_build truncates.
    #[test]
    fn fold_const_len_declines_above_slot_width() {
        assert_eq!(BinOp::Add.fold_const_len(MAX_CONST_ARRAY_LEN, 1), 0);
        assert_eq!(BinOp::Add.fold_const_len(4294967296, 4), 0);
        assert_eq!(BinOp::Mul.fold_const_len(65536, 65536), 0);
        assert_eq!(BinOp::Shl.fold_const_len(1, 40), 0);
    }

    /// An operand that is itself unrepresentable cannot be mirrored: mir_build's leaf decoder
    /// narrows it to u32 before folding, so sema must not fold it to a length it would then trust.
    #[test]
    fn fold_const_len_declines_out_of_range_operand() {
        assert_eq!(BinOp::Shr.fold_const_len(4294967296, 1), 0);
        assert_eq!(BinOp::Div.fold_const_len(4294967296, 2), 0);
        assert_eq!(BinOp::Mul.fold_const_len(4294967296, 0), 0);
    }

    /// Pre-existing sentinels stay: division/shift by zero and a non-arithmetic op fold to 0.
    #[test]
    fn fold_const_len_keeps_zero_sentinels() {
        assert_eq!(BinOp::Div.fold_const_len(4, 0), 0);
        assert_eq!(BinOp::Rem.fold_const_len(4, 0), 0);
        assert_eq!(BinOp::Shl.fold_const_len(1, 64), 0);
        assert_eq!(BinOp::Shr.fold_const_len(1, 64), 0);
        assert_eq!(BinOp::Lt.fold_const_len(1, 2), 0);
        assert_eq!(BinOp::And.fold_const_len(1, 1), 0);
    }
}
