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

/// A comparison predicate (integer signed/unsigned, or float). The result of a scalar compare is
/// `i1`; of a vector compare, a lane mask of the same width as the operands (`Vec(_, n)`).
///
/// The float predicates are **ordered** (false when either operand is NaN) except `Fone`, which is
/// the source language's `!=` and is **unordered** — true when either operand is NaN, i.e. Rust's
/// `f64::ne`, Cranelift's `FloatCC::NotEqual` and LLVM's `fcmp une`. The interpreter defines it that
/// way and every backend must match (float truthiness `x != 0.0` depends on it: NaN is truthy).
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

/// Where an [`Op::Alloca`] slot lives. CPU backends treat [`AddrSpace::Shared`] as ordinary local
/// storage; the GPU backend maps it to PTX `.shared`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum AddrSpace {
    #[default]
    Local,
    Shared,
}

/// An operation that may produce a value.
#[derive(Clone, Debug)]
pub enum Op {
    ConstInt(i128, MirType),
    ConstFloat(f64, MirType),
    /// Both operands and the result share one type — with one exception. A **vector shifted by a
    /// scalar** (`Shl`/`LShr`/`AShr` whose left operand and result are `Vec(T, n)` and whose right
    /// operand is the lane type `T`) shifts every lane by that one amount, masked to the lane width
    /// exactly as a scalar shift masks it. That is the shape every vector ISA has (Cranelift's
    /// `ishl`/`ushr`/`sshr` take a scalar amount, x86's `psllw`/`psrad` take one count), so the loop
    /// vectorizer emits it for a uniform shift amount. A lane-wise amount (`Vec` by `Vec`) stays
    /// legal MIR for the interpreter and the LLVM text path, but no vectorizer produces it.
    Bin(BinOp, ValueId, ValueId),
    Cmp(CmpOp, ValueId, ValueId),
    /// Arithmetic negation (`-x`); `FNeg` for floats is folded here by result type.
    Neg(ValueId),
    /// Bitwise/boolean complement.
    Not(ValueId),
    Cast(CastKind, ValueId, MirType),
    Select(ValueId, ValueId, ValueId),
    /// A stack slot; result is a pointer.
    Alloca {
        ty: MirType,
        space: AddrSpace,
    },
    /// A cooperative thread-block barrier (`sync_block()` in source). No result; a no-op on CPU.
    Barrier,
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
    /// Read one lane out of a SIMD vector: the operand is a `Vec(elem, n)`, the index is `< n`, and
    /// the result has the lane type. The inverse of [`Op::Splat`], and the *only* way back from
    /// vector to scalar form.
    ///
    /// It exists for one job: a float reduction may not be reassociated (IEEE addition is not
    /// associative and Wukong's interpreter is the language specification), so a vectorizer that
    /// widens the body feeding an accumulator has to fold the `n` lane values into it **one at a
    /// time, in index order**. Without a lane read that has to go through memory — store the vector
    /// to a stack slot and load the elements back — which costs a store-forwarding stall per group;
    /// `extractlane` is one shuffle, and lane 0 is free.
    ///
    /// Pure and side-effect-free. The interpreter indexes its lane arena; the Cranelift backend
    /// emits `extractlane`; the textual-LLVM emitter emits `extractelement`.
    ExtractLane(ValueId, u32),
    /// The lane-index ramp of a SIMD vector: the result is the `Vec(int_lane, n)` named by the
    /// operand type and lane `k` holds the integer `k`. No operands; pure, constant, and
    /// side-effect-free.
    ///
    /// It exists because a vector whose lanes differ cannot be *built* out of the other vector ops.
    /// [`Op::Splat`] makes every lane the same and every arithmetic op is lane-wise, so any
    /// expression over splatted scalars is uniform across lanes — a non-uniform vector needs one
    /// non-uniform source, and this is it.
    ///
    /// The job it exists for is the loop vectorizer's **lane ramp**. A loop body that reads its
    /// induction variable as a *value* rather than as an address (`if c == target`, `(i as f32)`)
    /// cannot splat it: lane `k` of the widened group is scalar iteration `i + k` and must see
    /// `i + k`, not `i`. With a unit step that vector is exactly `splat(i) + iota`, so an integer
    /// induction variable compared against a target yields a per-lane mask instead of one scalar
    /// bool.
    ///
    /// The interpreter materializes the lanes directly; the Cranelift backend emits a `vconst` from
    /// the constant pool; the textual-LLVM emitter emits the vector literal. The GPU MIR→PTX path
    /// scalarizes it like every other vectorized-MIR op, materializing lane `k` as the constant `k`
    /// (`mov.b64`) — exact for the same reason the interpreter can store lanes untruncated: the
    /// verifier admits integer lanes only, and `n` is at most 16.
    Iota(MirType),
    /// Convert one half of a SIMD vector's lanes to the type twice as wide: the operand is
    /// `Vec(T, n)`, the result `Vec(T2, n/2)`, and result lane `k` is source lane
    /// `half.first_lane(n) + k` converted by `kind` — `SExt`/`ZExt` from an integer lane to the
    /// integer type of twice its width, or `FpExt` from `f32` to `f64` ([`widen_result`] is the
    /// complete table). Every lane is exactly the scalar `Cast(kind, ..)` of its source lane.
    ///
    /// It exists because a widening cast changes the lane count. A loop that loads `i8` and
    /// computes in `f32` covers sixteen iterations per `<16 x i8>` load but only four per
    /// `<4 x f32>` value, so one source vector becomes several result vectors — `Lo` and `Hi` are
    /// the two halves that make that split expressible. Cranelift lowers it to
    /// `swiden_low/high`, `uwiden_low/high` or `fvpromote_low` (`pmovsx`/`punpck`/`cvtps2pd`), the
    /// textual-LLVM emitter to a half-width `shufflevector` plus `sext`/`zext`/`fpext`, and the GPU
    /// path scalarizes it. Pure and side-effect-free.
    Widen {
        v: ValueId,
        half: Half,
        kind: CastKind,
    },
    /// The inverse of [`Op::Widen`]: two vectors `Vec(T2, n)` become one `Vec(T, 2n)` of half the
    /// lane width, the lanes of `lo` first and then the lanes of `hi`, each reduced by `kind` (see
    /// [`NarrowKind`] and [`narrow_result`]). Every lane is a pure function of its own source lane,
    /// so interp and native agree lane by lane. Cranelift lowers it to `snarrow`/`unarrow`
    /// (`packss`/`packus`), to a masked `unarrow` for truncation, and to `fvdemote` plus a merge
    /// shuffle for `f64 -> f32`. Pure and side-effect-free.
    Narrow {
        lo: ValueId,
        hi: ValueId,
        kind: NarrowKind,
    },
    /// A two-source constant lane permutation: `a` and `b` are both `Vec(T, n)`, the result is
    /// `Vec(T, n)`, and result lane `k` is lane `mask[k]` of the concatenation `a ++ b` (so an index
    /// below `n` reads `a`, one in `n..2n` reads `b`). `mask.len()` is exactly `n` — the verifier
    /// checks both. A pure lane copy, so it is bit-exact on every backend for every lane type.
    ///
    /// The loop vectorizer uses it to split an interleaved access group (`x[2i]`, `x[2i+1]`) out of
    /// contiguous vector loads and to interleave values back before contiguous stores. Cranelift
    /// lowers it to its byte `shuffle` (`pshufb`, or `shufps`/`punpck`/`pshufd` for the masks those
    /// cover); the textual-LLVM emitter to `shufflevector`. Pure and side-effect-free.
    Shuffle {
        a: ValueId,
        b: ValueId,
        mask: Vec<u8>,
    },
    /// Write one lane of a SIMD vector: the operand is a `Vec(T, n)`, the scalar is a `T`, the index
    /// is `< n`, and the result is the same vector with lane `k` replaced. The inverse of
    /// [`Op::ExtractLane`], and the way the loop vectorizer packs per-lane scalar results (a gather,
    /// a division x86 cannot widen) back into a vector for the widened ops that consume them.
    /// Cranelift emits `insertlane`; the textual-LLVM emitter `insertelement`. Pure and
    /// side-effect-free.
    InsertLane(ValueId, ValueId, u32),
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
    /// caller runs the scalar remainder). An *elementwise* recipe (`VecKernel::reduce == None`) is
    /// side-effecting — it stores through the output stream pointers — and yields no value; a
    /// *reduction* recipe stores nothing and yields its horizontal fold as an `F32`. The verifier
    /// enforces that pairing. The interpreter marshals the recipe lane-wise (the differential
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

/// Which half of a SIMD vector's lanes an [`Op::Widen`] reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Half {
    /// Lanes `0 .. n/2`.
    Lo,
    /// Lanes `n/2 .. n`.
    Hi,
}

impl Half {
    pub fn name(self) -> &'static str {
        match self {
            Half::Lo => "lo",
            Half::Hi => "hi",
        }
    }

    /// The first source lane this half reads, for a source vector of `n` lanes.
    pub fn first_lane(self, n: u32) -> u32 {
        match self {
            Half::Lo => 0,
            Half::Hi => n / 2,
        }
    }
}

/// How an [`Op::Narrow`] reduces each lane to half its width. Every kind is a pure function of one
/// source lane, read as the *signed* value the interpreter holds it as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NarrowKind {
    /// Keep the low bits: two's-complement wraparound, exactly the scalar `CastKind::Trunc`.
    Trunc,
    /// Clamp to the narrow type's signed range (`packss`). The composition of two of these is the
    /// clamp to the narrower range, which is how a saturating `f32 -> i8` is built: `i32 -> i16 ->
    /// i8`, each step saturating.
    SatSigned,
    /// Clamp the signed source to `0 ..= 2^w - 1` and keep that bit pattern (`packus`): `-5` becomes
    /// `0`, `300` becomes `255` in an `i16 -> i8` step.
    SatUnsigned,
    /// `f64 -> f32`, rounding to nearest-even: the scalar `CastKind::FpTrunc`.
    FpTrunc,
}

impl NarrowKind {
    pub fn name(self) -> &'static str {
        match self {
            NarrowKind::Trunc => "trunc",
            NarrowKind::SatSigned => "sat.s",
            NarrowKind::SatUnsigned => "sat.u",
            NarrowKind::FpTrunc => "fptrunc",
        }
    }
}

/// The result type of [`Op::Widen`] over `src` by `kind`, or `None` when the pair is not one the op
/// admits: an integer lane of 8, 16 or 32 bits sign- or zero-extended to twice its width, or `f32`
/// extended to `f64`, over an even lane count.
pub fn widen_result(src: &MirType, kind: CastKind) -> Option<MirType> {
    let MirType::Vec(lane, n) = src else {
        return None;
    };
    if *n < 2 || !n.is_multiple_of(2) {
        return None;
    }
    let to = match (kind, &**lane) {
        (CastKind::SExt | CastKind::ZExt, MirType::I8) => MirType::I16,
        (CastKind::SExt | CastKind::ZExt, MirType::I16) => MirType::I32,
        (CastKind::SExt | CastKind::ZExt, MirType::I32) => MirType::I64,
        (CastKind::FpExt, MirType::F32) => MirType::F64,
        _ => return None,
    };
    Some(MirType::Vec(Box::new(to), n / 2))
}

/// The result type of [`Op::Narrow`] whose two operands are each `src`, or `None` when the pair is
/// not one the op admits. Truncation takes `i16`/`i32`/`i64` lanes; the two saturating kinds take
/// `i16`/`i32` (there is no packed 64-to-32 saturation to lower them to); `FpTrunc` takes `f64`.
pub fn narrow_result(src: &MirType, kind: NarrowKind) -> Option<MirType> {
    let MirType::Vec(lane, n) = src else {
        return None;
    };
    if *n == 0 {
        return None;
    }
    let to = match (kind, &**lane) {
        (NarrowKind::Trunc | NarrowKind::SatSigned | NarrowKind::SatUnsigned, MirType::I16) => {
            MirType::I8
        }
        (NarrowKind::Trunc | NarrowKind::SatSigned | NarrowKind::SatUnsigned, MirType::I32) => {
            MirType::I16
        }
        (NarrowKind::Trunc, MirType::I64) => MirType::I32,
        (NarrowKind::FpTrunc, MirType::F64) => MirType::F32,
        _ => return None,
    };
    Some(MirType::Vec(Box::new(to), n.checked_mul(2)?))
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
    /// A multi-way branch on an integer: control goes to the case whose `value` equals the
    /// scrutinee, or to `default` when none does. The scrutinee is a scalar `i8`/`i16`/`i32`/`i64`
    /// (never `i1`, never a vector), every case value is in canonical form for that type (see
    /// [`switch_case_value`]) and no two cases share a value — the verifier checks all three. Two
    /// cases, or a case and the default, may name the same target; each is its own edge with its
    /// own arguments.
    ///
    /// The semantics are exactly a chain of `cond_br (cmp.eq scrutinee, value)` tested in case
    /// order, which is what `wukong_opt`'s switch formation builds it from: the interpreter compares
    /// the scrutinee's integer value against each case, the Cranelift backend hands the cases to
    /// `cranelift_frontend::Switch` (a jump table, bit tests or a search tree), and the textual-LLVM
    /// emitter prints an LLVM `switch`. Nothing in `mir_build` emits one.
    Switch {
        scrutinee: ValueId,
        cases: Vec<SwitchCase>,
        default: BlockId,
        default_args: Vec<ValueId>,
    },
    Unreachable,
}

/// One arm of a [`Terminator::Switch`]: when the scrutinee equals `value`, branch to `target`,
/// passing `args` to its parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchCase {
    /// The case value in **canonical form**: the scrutinee type's bit pattern sign-extended from
    /// its width, so an `i8` case lies in `-128..=127` and the byte `0xFF` is spelled `-1`. That is
    /// how the interpreter holds every integer of that type, so a case matches exactly when
    /// `cmp.eq scrutinee, value` would. [`switch_case_value`] produces it.
    pub value: i128,
    pub target: BlockId,
    pub args: Vec<ValueId>,
}

/// The canonical [`SwitchCase::value`] of the integer `value` for a scrutinee of type `ty`: its
/// low `ty`-width bits, sign-extended. `None` when `ty` cannot be a switch scrutinee — anything but
/// a scalar `i8`, `i16`, `i32` or `i64`.
pub fn switch_case_value(value: i128, ty: &MirType) -> Option<i128> {
    let bits: u32 = match ty {
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => return None,
    };
    let shift = 128 - bits;
    Some((value << shift) >> shift)
}

impl Terminator {
    /// How many outgoing edges this terminator has. A target reached along two edges (both arms
    /// of a `cond_br`, or two switch cases) counts once per edge.
    pub fn edge_count(&self) -> usize {
        match self {
            Terminator::Ret(_) | Terminator::Unreachable => 0,
            Terminator::Br { .. } => 1,
            Terminator::CondBr { .. } => 2,
            Terminator::Switch { cases, .. } => cases.len() + 1,
        }
    }

    /// The target of outgoing edge `i` in **edge order** — a `cond_br`'s then-edge before its
    /// else-edge, a `switch`'s cases in order and then its default — or `None` past the last edge.
    pub fn edge_target(&self, i: usize) -> Option<BlockId> {
        match self {
            Terminator::Ret(_) | Terminator::Unreachable => None,
            Terminator::Br { target, .. } => (i == 0).then_some(*target),
            Terminator::CondBr {
                then_blk, else_blk, ..
            } => match i {
                0 => Some(*then_blk),
                1 => Some(*else_blk),
                _ => None,
            },
            Terminator::Switch { cases, default, .. } => match cases.get(i) {
                Some(c) => Some(c.target),
                None => (i == cases.len()).then_some(*default),
            },
        }
    }

    /// Visit every outgoing edge as `(target, args)`, in edge order.
    pub fn for_each_edge(&self, mut f: impl FnMut(BlockId, &[ValueId])) {
        match self {
            Terminator::Ret(_) | Terminator::Unreachable => {}
            Terminator::Br { target, args } => f(*target, args),
            Terminator::CondBr {
                then_blk,
                then_args,
                else_blk,
                else_args,
                ..
            } => {
                f(*then_blk, then_args);
                f(*else_blk, else_args);
            }
            Terminator::Switch {
                cases,
                default,
                default_args,
                ..
            } => {
                for c in cases {
                    f(c.target, &c.args);
                }
                f(*default, default_args);
            }
        }
    }

    /// Visit every outgoing edge mutably, in edge order: retarget it, or edit its arguments.
    pub fn for_each_edge_mut(&mut self, mut f: impl FnMut(&mut BlockId, &mut Vec<ValueId>)) {
        match self {
            Terminator::Ret(_) | Terminator::Unreachable => {}
            Terminator::Br { target, args } => f(target, args),
            Terminator::CondBr {
                then_blk,
                then_args,
                else_blk,
                else_args,
                ..
            } => {
                f(then_blk, then_args);
                f(else_blk, else_args);
            }
            Terminator::Switch {
                cases,
                default,
                default_args,
                ..
            } => {
                for c in cases {
                    f(&mut c.target, &mut c.args);
                }
                f(default, default_args);
            }
        }
    }
}
