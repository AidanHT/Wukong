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
#[cfg(test)]
mod lane_op_tests;
pub mod print;
pub mod vec_kernel;
pub mod verify;

pub use builder::Builder;
pub use inst::{
    narrow_result, switch_case_value, widen_result, AddrSpace, BinOp, CastKind, CmpOp, Half, Inst,
    IntUnOp, NarrowKind, Op, RoundMode, SwitchCase, Terminator,
};
pub use vec_kernel::{
    host_supports_vec_kernels, op_operands, op_produces_value, VecBin, VecCmp, VecKernel, VecLane,
    VecOp, VecPressure, VecRedOp, VecReduce, VecValue, VEC_BYTES, VEC_LANES, VEC_MAX_STREAMS,
    VEC_NREG,
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

/// What the source program states about one parameter's storage. Index-parallel to
/// [`Function::params`]; see [`Function::param_attr`].
///
/// These are *facts the source promised*, not facts any pass proved, so the rule for every producer
/// and consumer is the same: dropping an entry (or the whole vector) is always sound — it only
/// forgets a promise — while inventing one is a miscompile. The interpreter ignores them entirely:
/// the language's meaning of a program never depends on a promise, only what an optimizer may
/// assume when the promise is kept.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ParamAttrs {
    /// `@noalias` on the parameter: for the duration of the call, the storage it points into (for a
    /// slice, both the 16-byte fat pointer and the elements it views) is accessed only through
    /// pointers derived from this parameter. No other parameter, and nothing else the caller can
    /// name, overlaps it. This is C's `restrict` / LLVM's `noalias` argument attribute.
    pub noalias: bool,
    /// The parameter is a `[]T` slice: its value points at the caller's 16-byte fat pointer (data
    /// pointer at offset 0, `i64` length at offset 8), which is a live object for the whole call.
    /// A slice's elements never overlap its own fat pointer. Unlike `noalias` this is a fact of
    /// the parameter's type, so `mir_build` records it on every slice parameter, with or without
    /// the attribute: alias analysis reads it only beside `noalias`, and the vectorizer reads it
    /// alone, as the proof that reading the fat pointer cannot fault on a path the scalar loop
    /// never takes (`wukong_opt::vectorize::never_faults`). A raw `*T` parameter never carries it —
    /// `0 as *T` is null.
    pub slice: bool,
    /// The parameter is a fixed-extent buffer — a `[T; N]` array (nested arrays included) or a
    /// statically-shaped contiguous tensor of scalar `T` — so its value points at (at least)
    /// `elems` scalars of `elem_bytes` bytes each, live for the whole call. This is the contract of
    /// the parameter's declared type, and it is checked only where the argument's extent is known:
    /// an argument of another constant length is `E0501`, but a `?` (run-time) extent binds a
    /// fixed one unchecked, so a shorter buffer bound that way is outside the defined contract —
    /// the guide's rule for every run-time size (`docs/language-guide.md`, "Extents"). Wukong has
    /// no reinterpreting view. Recorded independently of `@noalias`; `None` for every other
    /// parameter. The vectorizer reads it to run a search loop's test ahead of the scalar loop over
    /// a parameter array (`wukong_opt::vectorize`), reading up to `W - 1` elements past the
    /// scalar loop's exit but never past `elems`.
    pub extent: Option<ParamExtent>,
}

/// The fixed storage extent of a buffer parameter ([`ParamAttrs::extent`]): `elems` scalar leaves
/// of `elem_bytes` bytes each. Counted in scalar leaves because that is the one unit both execution
/// models agree on — the interpreter's memory is one slot per scalar leaf, the native backend's is
/// `elem_bytes` bytes per leaf — so a range proved inside `[0, elems)` holds in both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ParamExtent {
    pub elems: u64,
    pub elem_bytes: u32,
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
    /// Per-parameter source facts, index-parallel to `params`. Often empty: a missing entry means
    /// [`ParamAttrs::default`], so read it through [`Function::param_attr`], never by indexing.
    /// Set by `mir_build` from a `@noalias` parameter, a `[]T` slice and a fixed-extent buffer;
    /// consumed by `wukong_opt::alias`, the backends' alias annotations and the vectorizer.
    pub param_attrs: Vec<ParamAttrs>,
}

impl Function {
    pub fn value_type(&self, v: ValueId) -> &MirType {
        &self.value_types[v.0 as usize]
    }

    /// The source facts for parameter number `i` (an index into `params`). A parameter with no
    /// recorded entry — every parameter of a function nothing annotated — answers the default,
    /// which promises nothing.
    pub fn param_attr(&self, i: usize) -> ParamAttrs {
        self.param_attrs.get(i).copied().unwrap_or_default()
    }

    /// Does any parameter carry a `@noalias` promise? The cheap gate every consumer checks first,
    /// so a function without one pays nothing for the feature.
    pub fn has_noalias_params(&self) -> bool {
        self.param_attrs.iter().any(|a| a.noalias)
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

/// One parameter or the result of an [`ExternFn`]: its MIR type plus the signedness the C ABI
/// needs to extend a narrow integer (MIR integers are signless). `signed` is `false` for floats,
/// pointers and `void`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternParam {
    pub ty: MirType,
    pub signed: bool,
}

/// A C function declared in an `extern "C"` block. A call names it with an ordinary
/// [`Op::Call`] whose `func` is no MIR function's name; every backend resolves that name here:
/// Cranelift imports the symbol (the JIT binds it from the host process, an object leaves it to the
/// linker), the LLVM emitter writes a `declare`, and the interpreter calls the same host symbol for
/// a scalar signature (`wukong_runtime::cffi`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternFn {
    /// The C symbol name (never module-mangled).
    pub name: Symbol,
    pub params: Vec<ExternParam>,
    /// `MirType::Void` for a C `void` function.
    pub ret: ExternParam,
}

/// A whole compiled program: its functions, read-only static data, the C functions it imports, and
/// an inert `level` field (always [`MirLevel::Low`] — read that type's note before relying on it).
#[derive(Clone, Debug)]
pub struct Program {
    pub funcs: Vec<Function>,
    /// Read-only data blobs (string literals) referenced by `Op::GlobalAddr`; each `name` is a
    /// symbol interned once in `mir_build`, unique per distinct content.
    pub statics: Vec<StaticData>,
    /// `extern "C"` functions, in declaration order, unique by name.
    pub externs: Vec<ExternFn>,
    pub level: MirLevel,
}

impl Program {
    pub fn new() -> Program {
        Program {
            funcs: Vec::new(),
            statics: Vec::new(),
            externs: Vec::new(),
            level: MirLevel::Low,
        }
    }

    pub fn function(&self, name: Symbol) -> Option<&Function> {
        self.funcs.iter().find(|f| f.name == name)
    }

    /// The `extern "C"` declaration named `name`, if the program imports one.
    pub fn extern_fn(&self, name: Symbol) -> Option<&ExternFn> {
        self.externs.iter().find(|e| e.name == name)
    }
}

impl Default for Program {
    fn default() -> Program {
        Program::new()
    }
}
