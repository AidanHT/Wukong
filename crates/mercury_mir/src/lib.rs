//! `mercury_mir` — Mercury IR.
//!
//! A typed, block-structured SSA representation. Each [`Function`] is a list of
//! [`BasicBlock`]s; every block ends in exactly one [`Terminator`]. SSA value merges use
//! **block parameters** (à la Cranelift/MLIR) rather than explicit phi nodes, which keeps
//! construction and verification simple. The IR carries a [`MirLevel`] invariant so lowering
//! passes can advance it from `High` (structured tensor/loop ops) to `Low` (scalar SSA only).

mod builder;
mod inst;
pub mod print;
pub mod verify;

pub use builder::Builder;
pub use inst::{BinOp, CastKind, CmpOp, Inst, Op, Terminator};

use mercury_span::Symbol;
use mercury_types::Scalar;

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
    /// A fixed-size array of `count` elements, laid out contiguously. Used as the operand type of
    /// an `alloca` for an array local; the alloca's *result* is still a `Ptr` to the first element.
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
}

impl Function {
    pub fn value_type(&self, v: ValueId) -> &MirType {
        &self.value_types[v.0 as usize]
    }

    pub fn block(&self, b: BlockId) -> &BasicBlock {
        &self.blocks[b.0 as usize]
    }
}

/// How far MIR has been lowered. The backend only ever sees `Low`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MirLevel {
    High,
    Low,
}

/// A whole compiled program: its functions and the current lowering level.
#[derive(Clone, Debug)]
pub struct Program {
    pub funcs: Vec<Function>,
    pub level: MirLevel,
}

impl Program {
    pub fn new() -> Program {
        Program {
            funcs: Vec::new(),
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
