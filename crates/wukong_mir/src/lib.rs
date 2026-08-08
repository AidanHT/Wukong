//! `wukong_mir` — Wukong IR.
//!
//! A typed, block-structured SSA representation. Each [`Function`] is a list of
//! [`BasicBlock`]s; every block ends in exactly one [`Terminator`]. SSA value merges use
//! **block parameters** (à la Cranelift/MLIR) rather than explicit phi nodes, which keeps
//! construction and verification simple. There is only one lowering stage today: `mir_build`
//! produces scalar SSA directly, so every [`Program`] is [`MirLevel::Low`] from birth (see that
//! type's note before building on the two-level design).

mod builder;
mod inst;
pub mod print;
pub mod vec_kernel;
pub mod verify;

pub use builder::Builder;
pub use inst::{BinOp, CastKind, CmpOp, Inst, Op, RoundMode, Terminator};
pub use vec_kernel::{
    host_supports_vec_kernels, op_operands, op_produces_value, VecBin, VecCmp, VecKernel, VecOp,
    VecPressure, VecRedOp, VecReduce, VEC_LANES, VEC_MAX_STREAMS, VEC_NREG,
};

use wukong_span::Symbol;
use wukong_types::Scalar;

/// A single-assignment value (an instruction result or block parameter).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl std::fmt::Debug for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// A basic block within a function.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl std::fmt::Debug for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// A low-level type used by MIR and the backends. Integers are signless (signedness lives on
/// the operation, as in LLVM); pointers are opaque.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum MirType {
    I1,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
    Ptr,
    Vec(Box<MirType>, u32),
    /// A fixed-size array of `count` elements, laid out contiguously. Used as the operand type of an
    /// `alloca` for any aggregate local — an array of `elem`, or a struct/tuple/slice as an
    /// `Array(I8, size_in_bytes)` byte blob — and as a `Gep`'s `elem` when the element being indexed
    /// is itself an aggregate (`mir_build`'s `gep_elem`: native scales the index by `size_of(elem)`,
    /// the interpreter by `slot_count(elem)`). It is never a *value* type: the alloca's *result* is
    /// still a `Ptr` to the first element.
    Array(Box<MirType>, u32),
    Void,
}

impl MirType {
    pub fn from_scalar(s: Scalar) -> MirType {
        use Scalar::*;
        match s {
            Bool => MirType::I1,
            I8 | U8 => MirType::I8,
            I16 | U16 => MirType::I16,
            I32 | U32 => MirType::I32,
            I64 | U64 | Usize | Isize => MirType::I64,
            F16 => MirType::F16,
            Bf16 => MirType::BF16,
            F32 => MirType::F32,
            F64 => MirType::F64,
            // A Unicode scalar value is a 32-bit integer in MIR (signless, like every int).
            Char => MirType::I32,
        }
    }

    pub fn is_int(&self) -> bool {
        matches!(
            self,
            MirType::I1 | MirType::I8 | MirType::I16 | MirType::I32 | MirType::I64
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(
            self,
            MirType::F16 | MirType::BF16 | MirType::F32 | MirType::F64
        )
    }

    /// For a SIMD `Vec`, the per-lane scalar type; for anything else, the type itself. Used to
    /// classify vector arithmetic (a `<4 x f32>` `fadd` is a float op on its `f32` lanes).
    pub fn lane_type(&self) -> &MirType {
        match self {
            MirType::Vec(elem, _) => elem,
            other => other,
        }
    }

    pub fn is_vector(&self) -> bool {
        matches!(self, MirType::Vec(..))
    }

    pub fn display(&self) -> String {
        match self {
            MirType::I1 => "i1".into(),
            MirType::I8 => "i8".into(),
            MirType::I16 => "i16".into(),
            MirType::I32 => "i32".into(),
            MirType::I64 => "i64".into(),
            MirType::F16 => "f16".into(),
            MirType::BF16 => "bf16".into(),
            MirType::F32 => "f32".into(),
            MirType::F64 => "f64".into(),
            MirType::Ptr => "ptr".into(),
            MirType::Vec(e, n) => format!("<{} x {}>", n, e.display()),
            MirType::Array(e, n) => format!("[{} x {}]", n, e.display()),
            MirType::Void => "void".into(),
        }
    }
}

/// One basic block: typed parameters, a straight-line instruction sequence, and a terminator.
#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub id: BlockId,
    pub params: Vec<ValueId>,
    pub insts: Vec<Inst>,
    pub term: Terminator,
}

/// A function: a value arena (`ValueId -> MirType`), a list of blocks, and an entry.
#[derive(Clone, Debug)]
pub struct Function {
    pub name: Symbol,
    pub params: Vec<ValueId>,
    pub ret: MirType,
    pub blocks: Vec<BasicBlock>,
    pub value_types: Vec<MirType>,
    pub entry: BlockId,
    /// Synthesized 256-bit AVX2 vector kernels (P4) this function calls, indexed by
    /// [`Op::VecKernelCall`]'s `kernel` field. Function-local so lowering can register a recipe on
    /// the same [`crate::Builder`] it is already threading — no program-wide side table. The
    /// interpreter marshals these lane-wise (the oracle); the Cranelift backend assembles each to
    /// raw AVX2 (`avx2.rs`). Empty for every function the vectorizer did not widen.
    pub vec_kernels: Vec<VecKernel>,
}

impl Function {
    pub fn value_type(&self, v: ValueId) -> &MirType {
        &self.value_types[v.0 as usize]
    }

    pub fn block(&self, b: BlockId) -> &BasicBlock {
        &self.blocks[b.0 as usize]
    }
}

/// How far MIR has been lowered.
///
/// NOT YET USED: no pass constructs `High`, `Program::new` starts at `Low`, and nothing anywhere
/// reads `Program::level` — every mention is either a copy of an existing level or the literal
/// `Low`. Do not write `if program.level == MirLevel::High { … }`: it can never fire. Note also
/// that `wukongc --emit=mir-high` means *pre-optimization* MIR, not this `High`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MirLevel {
    High,
    Low,
}

/// A read-only static-data blob living in `.rodata`, referenced by address via `Op::GlobalAddr`.
/// Currently only string literals: `bytes` holds the literal's UTF-8 plus a trailing NUL, so the
/// address is directly usable by `print_str` (which scans to the NUL). Emitted once per unique
/// content (deduped in `mir_build`), so a returned/threaded `*u8` points at stable storage instead
/// of a reclaimed stack frame.
#[derive(Clone, Debug)]
pub struct StaticData {
    pub name: Symbol,
    pub bytes: Vec<u8>,
}

/// A whole compiled program: its functions, read-only static data, and an inert `level` field
/// (always [`MirLevel::Low`] — read that type's note before relying on it).
#[derive(Clone, Debug)]
pub struct Program {
    pub funcs: Vec<Function>,
    /// Read-only data blobs (string literals) referenced by `Op::GlobalAddr`; each `name` is a
    /// symbol interned once in `mir_build`, unique per distinct content.
    pub statics: Vec<StaticData>,
    pub level: MirLevel,
}

impl Program {
    pub fn new() -> Program {
        Program {
            funcs: Vec::new(),
            statics: Vec::new(),
            level: MirLevel::Low,
        }
    }

    pub fn function(&self, name: Symbol) -> Option<&Function> {
        self.funcs.iter().find(|f| f.name == name)
    }
}

impl Default for Program {
    fn default() -> Program {
        Program::new()
    }
}
