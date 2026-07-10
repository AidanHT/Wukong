//! MIR instructions, operations, and terminators.

use crate::{BlockId, MirType, ValueId};
use wukong_span::Symbol;

/// A binary arithmetic/bitwise operation. Signedness is explicit (`SDiv` vs `UDiv`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
}

impl BinOp {
    pub fn name(self) -> &'static str {
        use BinOp::*;
        match self {
            Add => "add",
            Sub => "sub",
            Mul => "mul",
            SDiv => "sdiv",
            UDiv => "udiv",
            SRem => "srem",
            URem => "urem",
            FAdd => "fadd",
            FSub => "fsub",
            FMul => "fmul",
            FDiv => "fdiv",
            FRem => "frem",
            And => "and",
            Or => "or",
            Xor => "xor",
            Shl => "shl",
            LShr => "lshr",
            AShr => "ashr",
        }
    }

    pub fn is_float(self) -> bool {
        use BinOp::*;
        matches!(self, FAdd | FSub | FMul | FDiv | FRem)
    }
}

/// A comparison predicate (integer signed/unsigned, or ordered float). Result is `i1`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpOp {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
    Foeq,
    Fone,
    Folt,
    Fole,
    Fogt,
    Foge,
}

impl CmpOp {
    pub fn name(self) -> &'static str {
        use CmpOp::*;
        match self {
            Eq => "eq",
            Ne => "ne",
            Slt => "slt",
            Sle => "sle",
            Sgt => "sgt",
            Sge => "sge",
            Ult => "ult",
            Ule => "ule",
            Ugt => "ugt",
            Uge => "uge",
            Foeq => "foeq",
            Fone => "fone",
            Folt => "folt",
            Fole => "fole",
            Fogt => "fogt",
            Foge => "foge",
        }
    }

    pub fn is_float(self) -> bool {
        use CmpOp::*;
        matches!(self, Foeq | Fone | Folt | Fole | Fogt | Foge)
    }
}

/// A type conversion.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CastKind {
    SExt,
    ZExt,
    Trunc,
    FpToSi,
    FpToUi,
    SiToFp,
    UiToFp,
    FpExt,
    FpTrunc,
    Bitcast,
    IntToPtr,
    PtrToInt,
}

impl CastKind {
    pub fn name(self) -> &'static str {
        use CastKind::*;
        match self {
            SExt => "sext",
            ZExt => "zext",
            Trunc => "trunc",
            FpToSi => "fptosi",
            FpToUi => "fptoui",
            SiToFp => "sitofp",
            UiToFp => "uitofp",
            FpExt => "fpext",
            FpTrunc => "fptrunc",
            Bitcast => "bitcast",
            IntToPtr => "inttoptr",
            PtrToInt => "ptrtoint",
        }
    }
}

/// An operation that may produce a value.
#[derive(Clone, Debug)]
pub enum Op {
    ConstInt(i128, MirType),
    ConstFloat(f64, MirType),
    Bin(BinOp, ValueId, ValueId),
    Cmp(CmpOp, ValueId, ValueId),
    /// Arithmetic negation (`-x`); `FNeg` for floats is folded here by result type.
    Neg(ValueId),
    /// Bitwise/boolean complement.
    Not(ValueId),
    Cast(CastKind, ValueId, MirType),
    Select(ValueId, ValueId, ValueId),
    /// A stack slot; result is a pointer.
    Alloca(MirType),
    /// Load a value of the given type from a pointer.
    Load(ValueId, MirType),
    /// Store `value` to `ptr` (no result).
    Store {
        ptr: ValueId,
        value: ValueId,
    },
    /// `ptr + index * sizeof(elem)`; result is a pointer.
    Gep {
        ptr: ValueId,
        index: ValueId,
        elem: MirType,
    },
    /// A direct call to a function by name.
    Call {
        func: Symbol,
        args: Vec<ValueId>,
    },
    /// The machine address of a function (a `Ptr`), for passing it to a runtime that calls it back
    /// — e.g. the outlined body of a `@parallel for`. Pure and side-effect-free.
    FuncAddr(Symbol),
    /// The read-only address (a `Ptr`) of a named static-data blob in `Program::statics` — how a
    /// string literal is materialized: its NUL-terminated UTF-8 bytes live once in `.rodata`, and
    /// every use (including a `return`ed or threaded `*u8`) is this address, so the pointer stays
    /// valid after the defining frame is gone (unlike a stack byte-buffer, which dangled on native).
    /// The closest analogue to `FuncAddr` — the address of a named thing. Pure and side-effect-free.
    GlobalAddr(Symbol),
    /// Broadcast a scalar across every lane of a SIMD vector. The result is a `Vec(elem, n)` whose
    /// lane type matches the operand; the vectorizer uses it to lift loop-invariant scalars into
    /// vector form. Pure and side-effect-free.
    Splat(ValueId),
    /// Fused multiply-add: `a * b + c` with a *single* rounding. The front-end contracts a float
    /// `x + y*z` into this; it is faster (one instruction) and more accurate than separate
    /// `FMul`+`FAdd`. All three operands and the result share one float type (scalar or `Vec`).
    /// The interpreter evaluates it with `mul_add` so it stays bit-identical to the native `fma`.
    Fma(ValueId, ValueId, ValueId),
    /// Square root of a float (scalar or `Vec`); operand and result share one float type. Lowers
    /// to a hardware `sqrt`; the interpreter mirrors it with `f32`/`f64::sqrt`, so the two stay
    /// bit-identical. Pure and side-effect-free. (`rsqrt` and `exp` are built from primitive ops,
    /// so they need no dedicated variant.)
    Sqrt(ValueId),
    /// Round a float to an integral value (scalar or `Vec`), by `RoundMode`; operand and result share
    /// one float type. Lowers to a hardware round (`roundss`/`roundps`); the interpreter mirrors it
    /// with the matching `f32`/`f64` method (`Nearest` ⇒ `round_ties_even`, the IEEE
    /// round-to-nearest-ties-to-even that `nearest` emits — *not* `round`, which is ties-away), so the
    /// two stay bit-identical. Pure and side-effect-free.
    Round(RoundMode, ValueId),
    /// Call a synthesized 256-bit AVX2 vector kernel (P4) by index into [`crate::Function::vec_kernels`],
    /// over the vector part `[0, n)` of a loop the general vectorizer widened past Cranelift's 128-bit
    /// CLIF-vector ceiling. `ptrs` points at a stack array of the stream base pointers (each already
    /// offset to the loop start); `scalars` at the loop-invariant f32s; `n` is a multiple of 8 (the
    /// caller runs the scalar remainder). Side-effecting — it stores through the output stream
    /// pointers — and yields no value. The interpreter marshals the recipe lane-wise (the differential
    /// oracle); the Cranelift backend assembles it to raw AVX2 (`avx2.rs`) and calls it. `kernel` is a
    /// pure index, not a `ValueId`, so it is invisible to SSA renaming.
    VecKernelCall {
        kernel: u32,
        ptrs: ValueId,
        scalars: ValueId,
        n: ValueId,
    },
}

/// Rounding direction for [`Op::Round`]. Each maps to one Cranelift instruction and one `f32`/`f64`
/// method, chosen so the native and interpreter results agree bit-for-bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundMode {
    /// To nearest, ties to even (`nearest` / `round_ties_even`).
    Nearest,
    /// Toward −∞ (`floor`).
    Floor,
    /// Toward +∞ (`ceil`).
    Ceil,
    /// Toward zero (`trunc`).
    Trunc,
}

/// One instruction: an optional result value plus its operation.
#[derive(Clone, Debug)]
pub struct Inst {
    pub result: Option<ValueId>,
    pub op: Op,
}

/// How a basic block ends. Exactly one per block.
#[derive(Clone, Debug)]
pub enum Terminator {
    Ret(Option<ValueId>),
    Br {
        target: BlockId,
        args: Vec<ValueId>,
    },
    CondBr {
        cond: ValueId,
        then_blk: BlockId,
        then_args: Vec<ValueId>,
        else_blk: BlockId,
        else_args: Vec<ValueId>,
    },
    Unreachable,
}
