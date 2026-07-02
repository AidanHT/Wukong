//! `mercury_codegen_cranelift` — native code generation via Cranelift (no LLVM toolchain required).
//!
//! Lowers fully-lowered (Low) scalar MIR to Cranelift IR and either JIT-compiles and runs it in
//! process (`jit_run`, the fast execution path and a differential oracle alongside the interpreter)
//! or emits a native object file (`emit_object`, linked into an executable by the driver).
//!
//! The MIR maps onto Cranelift almost one-to-one: it is already block-parameter SSA (exactly
//! Cranelift's model), integers are signless with signedness on the op, and memory is explicit
//! `alloca`/`load`/`store`/`gep`. The only semantic gaps we bridge to stay bit-identical to the
//! interpreter oracle are: divide-by-zero yields 0 (no trap), float→int casts saturate, and `i1`
//! results are normalised to their low bit.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    types, AbiParam, Block, BlockArg, FuncRef, InstBuilder, MemFlags, Signature, StackSlotData,
    StackSlotKind, Type, Value,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module};

use mercury_mir::{BinOp, CastKind, CmpOp, Function, MirType, Op, Program, Terminator, ValueId};
use mercury_span::{Interner, Symbol};

mod backend;
pub use backend::CraneliftBackend;

// --- The minimal runtime the generated code calls back into ---------------------------------
//
// `print`/`println`/`assert` lower to calls to these symbols. For the JIT they are real Rust
// functions registered with the module; output is captured into a process-global buffer so the
// native path produces exactly the bytes the interpreter would, which is what the differential
// gate compares. A global run lock serialises native runs sharing that buffer.

static OUTPUT: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static ASSERT_FAILED: AtomicBool = AtomicBool::new(false);
static RUN_LOCK: Mutex<()> = Mutex::new(());

extern "C" fn rt_print_i64(x: i64) {
    if let Ok(mut o) = OUTPUT.lock() {
        o.extend_from_slice(format!("{x}\n").as_bytes());
    }
}

extern "C" fn rt_print_f64(x: f64) {
    if let Ok(mut o) = OUTPUT.lock() {
        o.extend_from_slice(format!("{x}\n").as_bytes());
    }
}

extern "C" fn rt_assert(cond: i64) {
    if cond == 0 {
        ASSERT_FAILED.store(true, Ordering::SeqCst);
    }
}

// Float remainder. Cranelift has no `frem`; the interpreter uses Rust `%` (true fmod), so the native
// backend calls these to stay bit-identical instead of the lossy `a - trunc(a/b)*b` identity (which
// drifts once `a/b` exceeds the float's integer-precision range, e.g. `1e18 % 3`).
extern "C" fn rt_fmod_f64(a: f64, b: f64) -> f64 {
    a % b
}

extern "C" fn rt_fmod_f32(a: f32, b: f32) -> f32 {
    a % b
}

/// Names of the runtime symbols, shared by the JIT (which binds them to the `rt_*` functions) and
/// the object emitter (which leaves them as undefined imports resolved at link time).
const RT_PRINT_I64: &str = "mercury_rt_print_i64";
const RT_PRINT_F64: &str = "mercury_rt_print_f64";
const RT_ASSERT: &str = "mercury_rt_assert";
const RT_PARALLEL_FOR: &str = "mercury_parallel_for";
const RT_SGEMM: &str = "mercury_sgemm";
const RT_SGEMM_PARALLEL: &str = "mercury_sgemm_parallel";
const RT_SGEMM_NT: &str = "mercury_sgemm_nt";
const RT_SGEMM_NT_PARALLEL: &str = "mercury_sgemm_nt_parallel";
const RT_SGEMM_TN: &str = "mercury_sgemm_tn";
const RT_SGEMM_TN_PARALLEL: &str = "mercury_sgemm_tn_parallel";
const RT_SGEMM_BF16_NT: &str = "mercury_sgemm_bf16_nt";
const RT_SGEMM_BF16_NT_PARALLEL: &str = "mercury_sgemm_bf16_nt_parallel";
const RT_SGEMM_F16_NT: &str = "mercury_sgemm_f16_nt";
const RT_SGEMM_F16_NT_PARALLEL: &str = "mercury_sgemm_f16_nt_parallel";
const RT_SGEMM_BF16_TN: &str = "mercury_sgemm_bf16_tn";
const RT_SGEMM_BF16_TN_PARALLEL: &str = "mercury_sgemm_bf16_tn_parallel";
const RT_SGEMM_F16_TN: &str = "mercury_sgemm_f16_tn";
const RT_SGEMM_F16_TN_PARALLEL: &str = "mercury_sgemm_f16_tn_parallel";
const RT_SGEMM_NT_EPI: &str = "mercury_sgemm_nt_epi";
const RT_SGEMM_NT_EPI_PAR: &str = "mercury_sgemm_nt_epi_parallel";
const RT_SGEMV: &str = "mercury_sgemv";
const RT_SGEMV_PAR: &str = "mercury_sgemv_parallel";
const RT_SGEMM_NT_ALPHA: &str = "mercury_sgemm_nt_alpha";
const RT_SGEMM_NT_ALPHA_PAR: &str = "mercury_sgemm_nt_alpha_parallel";
const RT_SGEMM_BF16_NT_EPI: &str = "mercury_sgemm_bf16_nt_epi";
const RT_SGEMM_BF16_NT_EPI_PAR: &str = "mercury_sgemm_bf16_nt_epi_parallel";
const RT_SGEMM_F16_NT_EPI: &str = "mercury_sgemm_f16_nt_epi";
const RT_SGEMM_F16_NT_EPI_PAR: &str = "mercury_sgemm_f16_nt_epi_parallel";
const RT_VMATH: &str = "mercury_vmath_f32";
const RT_VMATH2: &str = "mercury_vmath2_f32";
const RT_SOFTMAX_BWD: &str = "mercury_softmax_bwd_f32";
const RT_SOFTMAX_BWD_PAR: &str = "mercury_softmax_bwd_f32_parallel";
const RT_LRSCAN: &str = "mercury_lrscan_f32";
const RT_LRSCAN_PAR: &str = "mercury_lrscan_f32_parallel";
const RT_RMSNORM_BWD: &str = "mercury_rmsnorm_bwd_f32";
const RT_RMSNORM_BWD_PAR: &str = "mercury_rmsnorm_bwd_f32_parallel";
const RT_LAYERNORM_BWD: &str = "mercury_layernorm_bwd_f32";
const RT_LAYERNORM_BWD_PAR: &str = "mercury_layernorm_bwd_f32_parallel";
const RT_XENT: &str = "mercury_xent_fwd_f32";
const RT_XENT_PAR: &str = "mercury_xent_fwd_f32_parallel";
const RT_XENT_BWD: &str = "mercury_xent_bwd_f32";
const RT_XENT_BWD_PAR: &str = "mercury_xent_bwd_f32_parallel";
const RT_ROPE: &str = "mercury_rope_f32";
const RT_ROPE_PAR: &str = "mercury_rope_f32_parallel";
const RT_ROPE_BWD: &str = "mercury_rope_bwd_f32";
const RT_ROPE_BWD_PAR: &str = "mercury_rope_bwd_f32_parallel";
const RT_LOGSUMEXP: &str = "mercury_logsumexp_f32";
const RT_LOGSUMEXP_PAR: &str = "mercury_logsumexp_f32_parallel";
const RT_KLDIV: &str = "mercury_kldiv_f32";
const RT_KLDIV_PAR: &str = "mercury_kldiv_f32_parallel";
const RT_ENTROPY: &str = "mercury_entropy_f32";
const RT_ENTROPY_PAR: &str = "mercury_entropy_f32_parallel";
const RT_KD_LOSS: &str = "mercury_kd_loss_f32";
const RT_KD_LOSS_PAR: &str = "mercury_kd_loss_f32_parallel";
const RT_ROWARGMAX: &str = "mercury_rowargmax_i32";
const RT_ROWARGMAX_PAR: &str = "mercury_rowargmax_i32_parallel";
const RT_ROWARGMIN: &str = "mercury_rowargmin_i32";
const RT_ROWARGMIN_PAR: &str = "mercury_rowargmin_i32_parallel";
const RT_COLARGMAX: &str = "mercury_colargmax_i32";
const RT_COLARGMAX_PAR: &str = "mercury_colargmax_i32_parallel";
const RT_COLARGMIN: &str = "mercury_colargmin_i32";
const RT_COLARGMIN_PAR: &str = "mercury_colargmin_i32_parallel";
const RT_CUMSUM: &str = "mercury_cumsum_f32";
const RT_CUMSUM_PAR: &str = "mercury_cumsum_f32_parallel";
const RT_CUMPROD: &str = "mercury_cumprod_f32";
const RT_CUMPROD_PAR: &str = "mercury_cumprod_f32_parallel";
const RT_CUMMAX: &str = "mercury_cummax_f32";
const RT_CUMMAX_PAR: &str = "mercury_cummax_f32_parallel";
const RT_CUMMIN: &str = "mercury_cummin_f32";
const RT_CUMMIN_PAR: &str = "mercury_cummin_f32_parallel";
const RT_MAXPOOL2D: &str = "mercury_maxpool2d_f32";
const RT_MAXPOOL2D_PAR: &str = "mercury_maxpool2d_f32_parallel";
const RT_AVGPOOL2D: &str = "mercury_avgpool2d_f32";
const RT_AVGPOOL2D_PAR: &str = "mercury_avgpool2d_f32_parallel";
const RT_VMATH_BF16: &str = "mercury_vmath_bf16";
const RT_VMATH_F16: &str = "mercury_vmath_f16";
const RT_TRANSPOSE: &str = "mercury_transpose_f32";
const RT_TRANSPOSE_PAR: &str = "mercury_transpose_f32_parallel";
const RT_TRANSPOSE_U16: &str = "mercury_transpose_u16";
const RT_TRANSPOSE_U16_PAR: &str = "mercury_transpose_u16_parallel";
const RT_COLSUM: &str = "mercury_colsum_f32";
const RT_COLSUM_PAR: &str = "mercury_colsum_f32_parallel";
const RT_COLMAX: &str = "mercury_colmax_f32";
const RT_COLMAX_PAR: &str = "mercury_colmax_f32_parallel";
const RT_COLMIN: &str = "mercury_colmin_f32";
const RT_COLMIN_PAR: &str = "mercury_colmin_f32_parallel";
const RT_COLMAXABS: &str = "mercury_colmaxabs_f32";
const RT_COLMAXABS_PAR: &str = "mercury_colmaxabs_f32_parallel";
const RT_COLMEAN: &str = "mercury_colmean_f32";
const RT_COLMEAN_PAR: &str = "mercury_colmean_f32_parallel";
const RT_COLSUMSQ: &str = "mercury_colsumsq_f32";
const RT_COLSUMSQ_PAR: &str = "mercury_colsumsq_f32_parallel";
const RT_COLL2: &str = "mercury_coll2_f32";
const RT_COLL2_PAR: &str = "mercury_coll2_f32_parallel";
const RT_COLRMS: &str = "mercury_colrms_f32";
const RT_COLRMS_PAR: &str = "mercury_colrms_f32_parallel";
const RT_VELEM: &str = "mercury_velem_f32";
const RT_VHORNER: &str = "mercury_vhorner_f32";
const RT_SREDUCE: &str = "mercury_sreduce_f32";
const RT_SREDUCE_PARALLEL: &str = "mercury_sreduce_f32_parallel";
const RT_ARGREDUCE: &str = "mercury_argreduce_f32";
const RT_ARGREDUCE_PARALLEL: &str = "mercury_argreduce_f32_parallel";
const RT_NORM: &str = "mercury_norm_f32";
const RT_NORM_PARALLEL: &str = "mercury_norm_f32_parallel";
const RT_NORM_AFFINE: &str = "mercury_norm_affine_f32";
const RT_NORM_AFFINE_PARALLEL: &str = "mercury_norm_affine_f32_parallel";
const RT_I8GEMM_NT: &str = "mercury_i8gemm_nt";
const RT_I8GEMM_NT_PARALLEL: &str = "mercury_i8gemm_nt_parallel";
const RT_I8GEMM_NT_DEQ: &str = "mercury_i8gemm_nt_deq";
const RT_I8GEMM_NT_DEQ_PARALLEL: &str = "mercury_i8gemm_nt_deq_parallel";
const RT_EMBEDDING: &str = "mercury_embedding_f32";
const RT_EMBEDDING_PAR: &str = "mercury_embedding_f32_parallel";
const RT_SCATTER_ADD: &str = "mercury_scatter_add_f32";
const RT_SCATTER_ADD_PAR: &str = "mercury_scatter_add_f32_parallel";
const RT_DOT_BF16: &str = "mercury_dot_bf16";
const RT_SUM_BF16: &str = "mercury_sum_bf16";
const RT_REDUCE_BF16: &str = "mercury_reduce_bf16";
const RT_DOT_F16: &str = "mercury_dot_f16";
const RT_SUM_F16: &str = "mercury_sum_f16";
const RT_REDUCE_F16: &str = "mercury_reduce_f16";
// f16 has no cheap inline round (unlike bf16's `<<16`), so f16 load/store/cast call these shims —
// the *same* `half`-crate conversion the interpreter uses, keeping native == interp bit-for-bit.
const RT_F32_TO_F16: &str = "mercury_f32_to_f16_bits";
const RT_F16_TO_F32: &str = "mercury_f16_bits_to_f32";
const RT_AXPBY_BF16: &str = "mercury_axpby_bf16";
const RT_AXPBY_F16: &str = "mercury_axpby_f16";
const RT_FMOD_F64: &str = "mercury_rt_fmod_f64";
const RT_FMOD_F32: &str = "mercury_rt_fmod_f32";

/// The runtime function an intrinsic call lowers to.
#[derive(Clone, Copy)]
enum Intrinsic {
    PrintInt,
    PrintFloat,
    Assert,
}

fn classify_intrinsic(name: &str, arg_is_float: bool) -> Option<Intrinsic> {
    match name {
        "print" | "println" => Some(if arg_is_float {
            Intrinsic::PrintFloat
        } else {
            Intrinsic::PrintInt
        }),
        "assert" => Some(Intrinsic::Assert),
        _ => None,
    }
}

// --- Type and size mapping ------------------------------------------------------------------

/// Map a MIR type to its Cranelift type, or `None` for `Void`. `i1` becomes `i8`; `f16`/`bf16` are
/// computed as `f32` (matching the interpreter's promotion of sub-`f32` floats).
fn cl_type(t: &MirType, ptr_ty: Type) -> Option<Type> {
    Some(match t {
        MirType::I1 | MirType::I8 => types::I8,
        MirType::I16 => types::I16,
        MirType::I32 => types::I32,
        MirType::I64 => types::I64,
        MirType::F16 | MirType::BF16 | MirType::F32 => types::F32,
        MirType::F64 => types::F64,
        MirType::Ptr => ptr_ty,
        MirType::Vec(elem, n) => {
            let lane = cl_type(elem, ptr_ty)?;
            lane.by(*n)?
        }
        MirType::Array(..) => ptr_ty,
        MirType::Void => return None,
    })
}

/// Size in bytes of a MIR type (for `gep` scaling and stack-slot sizing).
fn size_of(t: &MirType) -> u32 {
    match t {
        MirType::I1 | MirType::I8 => 1,
        MirType::I16 | MirType::F16 | MirType::BF16 => 2,
        MirType::I32 | MirType::F32 => 4,
        MirType::I64 | MirType::F64 | MirType::Ptr => 8,
        MirType::Vec(e, n) | MirType::Array(e, n) => size_of(e) * n,
        MirType::Void => 0,
    }
}

fn int_cc(op: CmpOp) -> IntCC {
    use CmpOp::*;
    match op {
        Eq => IntCC::Equal,
        Ne => IntCC::NotEqual,
        Slt => IntCC::SignedLessThan,
        Sle => IntCC::SignedLessThanOrEqual,
        Sgt => IntCC::SignedGreaterThan,
        Sge => IntCC::SignedGreaterThanOrEqual,
        Ult => IntCC::UnsignedLessThan,
        Ule => IntCC::UnsignedLessThanOrEqual,
        Ugt => IntCC::UnsignedGreaterThan,
        Uge => IntCC::UnsignedGreaterThanOrEqual,
        _ => unreachable!("float predicate in int_cc"),
    }
}

fn float_cc(op: CmpOp) -> FloatCC {
    use CmpOp::*;
    match op {
        Foeq => FloatCC::Equal,
        // The interpreter implements `!=` with Rust's `f64::ne` (unordered-or-not-equal), so match
        // that rather than the ordered `one`.
        Fone => FloatCC::NotEqual,
        Folt => FloatCC::LessThan,
        Fole => FloatCC::LessThanOrEqual,
        Fogt => FloatCC::GreaterThan,
        Foge => FloatCC::GreaterThanOrEqual,
        _ => unreachable!("int predicate in float_cc"),
    }
}

fn align_shift(bytes: u32) -> u8 {
    // Align stack slots to their (rounded-up power-of-two) size, capped at 16 bytes.
    let mut a = 1u32;
    let mut shift = 0u8;
    while a < bytes && a < 16 {
        a <<= 1;
        shift += 1;
    }
    shift
}

fn successors(t: &Terminator) -> Vec<u32> {
    match t {
        Terminator::Br { target, .. } => vec![target.0],
        Terminator::CondBr {
            then_blk, else_blk, ..
        } => {
            if then_blk == else_blk {
                vec![then_blk.0]
            } else {
                vec![then_blk.0, else_blk.0]
            }
        }
        Terminator::Ret(_) | Terminator::Unreachable => vec![],
    }
}

// --- Per-function lowering ------------------------------------------------------------------

/// Lowers one MIR `Function` into a Cranelift function body already attached to `builder`.
struct FnTranslator<'a> {
    builder: FunctionBuilder<'a>,
    func: &'a Function,
    interner: &'a Interner,
    ptr_ty: Type,
    /// MIR `ValueId` -> Cranelift value. Block params and instruction results fill this in.
    vmap: Vec<Option<Value>>,
    /// MIR block index -> Cranelift block.
    blocks: Vec<Block>,
    /// Pre-declared FuncRefs for callees and runtime imports in this function.
    func_refs: &'a HashMap<Symbol, FuncRef>,
    rt_refs: &'a HashMap<&'static str, FuncRef>,
}

impl<'a> FnTranslator<'a> {
    fn val(&self, v: ValueId) -> Value {
        self.vmap[v.0 as usize].expect("MIR value used before definition (non-SSA input?)")
    }

    fn set(&mut self, v: ValueId, cv: Value) {
        self.vmap[v.0 as usize] = Some(cv);
    }

    fn ty_of(&self, v: ValueId) -> &MirType {
        self.func.value_type(v)
    }

    fn dfg_ty(&self, v: Value) -> Type {
        self.builder.func.dfg.value_type(v)
    }

    /// Reverse-postorder of blocks so every definition is emitted before its uses.
    fn rpo(func: &Function) -> Vec<u32> {
        let mut visited = vec![false; func.blocks.len()];
        let mut post = Vec::new();
        let mut stack = vec![(func.entry.0, 0usize)];
        visited[func.entry.0 as usize] = true;
        while let Some(&mut (b, ref mut i)) = stack.last_mut() {
            let succs = successors(&func.blocks[b as usize].term);
            if *i < succs.len() {
                let s = succs[*i];
                *i += 1;
                if !visited[s as usize] {
                    visited[s as usize] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
                stack.pop();
            }
        }
        post.reverse();
        post
    }

    fn translate(&mut self) {
        // Give each Cranelift block its params (the entry block's are the function params).
        for (i, mb) in self.func.blocks.iter().enumerate() {
            let cb = self.blocks[i];
            if mb.id == self.func.entry {
                self.builder.append_block_params_for_function_params(cb);
                let cps: Vec<Value> = self.builder.block_params(cb).to_vec();
                for (p, cv) in mb.params.iter().zip(cps) {
                    self.set(*p, cv);
                }
            } else {
                for p in &mb.params {
                    let t = cl_type(self.ty_of(*p), self.ptr_ty).unwrap_or(self.ptr_ty);
                    let cv = self.builder.append_block_param(cb, t);
                    self.set(*p, cv);
                }
            }
        }

        let order = Self::rpo(self.func);
        for &bi in &order {
            let cb = self.blocks[bi as usize];
            self.builder.switch_to_block(cb);
            let mb = self.func.blocks[bi as usize].clone();
            for inst in &mb.insts {
                self.lower_inst(inst);
            }
            self.lower_term(&mb.term);
        }
        self.builder.seal_all_blocks();
    }

    fn lower_inst(&mut self, inst: &mercury_mir::Inst) {
        let res = inst.result;
        let cv = match &inst.op {
            Op::ConstInt(v, ty) => {
                if matches!(ty, MirType::I1) {
                    self.builder.ins().iconst(types::I8, (*v as i64) & 1)
                } else {
                    let t = cl_type(ty, self.ptr_ty).unwrap_or(types::I64);
                    self.builder.ins().iconst(t, *v as i64)
                }
            }
            Op::ConstFloat(v, ty) => {
                if matches!(ty, MirType::F64) {
                    self.builder.ins().f64const(*v)
                } else {
                    self.builder.ins().f32const(*v as f32)
                }
            }
            Op::Bin(op, l, r) => {
                let rt = self.ty_of(inst.result.unwrap()).clone();
                self.lower_bin(*op, *l, *r, &rt)
            }
            Op::Cmp(op, l, r) => {
                if op.is_float() {
                    // The front-end's float literals are loosely typed, so operands may differ in
                    // width; compare in the wider type (the interpreter compares in f64).
                    let (a0, b0) = (self.val(*l), self.val(*r));
                    let common = if self.dfg_ty(a0) == types::F64 || self.dfg_ty(b0) == types::F64 {
                        types::F64
                    } else {
                        types::F32
                    };
                    let a = self.coerce_float(a0, common);
                    let b = self.coerce_float(b0, common);
                    self.builder.ins().fcmp(float_cc(*op), a, b)
                } else {
                    let (a, b) = (self.val(*l), self.val(*r));
                    self.builder.ins().icmp(int_cc(*op), a, b)
                }
            }
            Op::Neg(v) => {
                let x = self.val(*v);
                if self.ty_of(*v).is_float() {
                    self.builder.ins().fneg(x)
                } else {
                    self.builder.ins().ineg(x)
                }
            }
            Op::Not(v) => {
                let x = self.val(*v);
                self.builder.ins().bnot(x)
            }
            Op::Cast(kind, v, to) => self.lower_cast(*kind, *v, to),
            Op::Select(c, a, b) => {
                let cc = self.val(*c);
                let av0 = self.val(*a);
                let bv0 = self.val(*b);
                let avty = self.dfg_ty(av0);
                if avty.is_vector() {
                    // Vector blend: the mask is a per-lane all-ones/all-zeros vector (from a vector
                    // compare). Reinterpret it to the value vector type and select bitwise.
                    let cty = self.dfg_ty(cc);
                    let mask = if cty == avty {
                        cc
                    } else {
                        self.builder.ins().bitcast(avty, MemFlags::new(), cc)
                    };
                    self.builder.ins().bitselect(mask, av0, bv0)
                } else {
                    // Cranelift requires both arms to share a type; unify mismatched float widths.
                    let (av, bv) = if self.dfg_ty(av0).is_float() || self.dfg_ty(bv0).is_float() {
                        let common =
                            if self.dfg_ty(av0) == types::F64 || self.dfg_ty(bv0) == types::F64 {
                                types::F64
                            } else {
                                types::F32
                            };
                        (
                            self.coerce_float(av0, common),
                            self.coerce_float(bv0, common),
                        )
                    } else {
                        (av0, bv0)
                    };
                    self.builder.ins().select(cc, av, bv)
                }
            }
            Op::Alloca(ty) => {
                let bytes = size_of(ty).max(1);
                let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    bytes,
                    align_shift(bytes),
                ));
                self.builder.ins().stack_addr(self.ptr_ty, slot, 0)
            }
            Op::Load(p, ty) => {
                let addr = self.val(*p);
                if matches!(ty, MirType::BF16) {
                    // bf16 storage is 2 bytes; load the 16 bits and widen to f32 exactly (the bits
                    // become the top half of the f32). Register type is f32 (see `cl_type`).
                    let half = self
                        .builder
                        .ins()
                        .load(types::I16, MemFlags::trusted(), addr, 0);
                    let ext = self.builder.ins().uextend(types::I32, half);
                    let shifted = self.builder.ins().ishl_imm(ext, 16);
                    self.builder
                        .ins()
                        .bitcast(types::F32, MemFlags::new(), shifted)
                } else if matches!(ty, MirType::F16) {
                    // f16 storage is 2 bytes; widen via the runtime shim (no cheap inline bit-extend
                    // like bf16). Register type is f32.
                    let half = self
                        .builder
                        .ins()
                        .load(types::I16, MemFlags::trusted(), addr, 0);
                    let ext = self.builder.ins().uextend(types::I32, half);
                    let fref = self.rt_refs[RT_F16_TO_F32];
                    let call = self.builder.ins().call(fref, &[ext]);
                    self.builder.inst_results(call)[0]
                } else {
                    let t = cl_type(ty, self.ptr_ty).unwrap_or(self.ptr_ty);
                    self.builder.ins().load(t, MemFlags::trusted(), addr, 0)
                }
            }
            Op::Store { ptr, value } => {
                let addr = self.val(*ptr);
                if matches!(self.ty_of(*value), MirType::BF16) {
                    // Round the f32 register to bf16 and store the top 16 bits (2 bytes).
                    let v = self.val(*value);
                    let rounded = self.round_to_bf16(v);
                    let bits = self
                        .builder
                        .ins()
                        .bitcast(types::I32, MemFlags::new(), rounded);
                    let hi = self.builder.ins().ushr_imm(bits, 16);
                    let half = self.builder.ins().ireduce(types::I16, hi);
                    self.builder.ins().store(MemFlags::trusted(), half, addr, 0);
                } else if matches!(self.ty_of(*value), MirType::F16) {
                    // Round the f32 register to f16 via the runtime shim, store the 16 bits (2 bytes).
                    let v = self.val(*value);
                    let fref = self.rt_refs[RT_F32_TO_F16];
                    let call = self.builder.ins().call(fref, &[v]);
                    let bits = self.builder.inst_results(call)[0]; // i32, low 16 = f16 bits
                    let half = self.builder.ins().ireduce(types::I16, bits);
                    self.builder.ins().store(MemFlags::trusted(), half, addr, 0);
                } else {
                    let v = self.val(*value);
                    self.builder.ins().store(MemFlags::trusted(), v, addr, 0);
                }
                return;
            }
            Op::Gep { ptr, index, elem } => {
                let base = self.val(*ptr);
                let idx = self.ptr_int(*index);
                let scaled = self.builder.ins().imul_imm(idx, size_of(elem) as i64);
                self.builder.ins().iadd(base, scaled)
            }
            Op::Call { func, args } => match self.lower_call(*func, args) {
                Some(v) => v,
                None => return,
            },
            Op::FuncAddr(sym) => {
                let fref = self.func_refs[sym];
                self.builder.ins().func_addr(self.ptr_ty, fref)
            }
            Op::Splat(v) => {
                let rty = self.ty_of(inst.result.unwrap()).clone();
                let vec_ty = cl_type(&rty, self.ptr_ty).unwrap_or(self.ptr_ty);
                let x = self.val(*v);
                self.builder.ins().splat(vec_ty, x)
            }
            // Fused multiply-add: Cranelift `fma(a, b, c)` is `a*b + c` with one rounding, lowering
            // to a hardware `vfmadd` (scalar or 128-bit vector) on FMA3 hosts.
            Op::Fma(a, b, c) => {
                let av = self.val(*a);
                let bv = self.val(*b);
                let cvv = self.val(*c);
                self.builder.ins().fma(av, bv, cvv)
            }
            // Hardware square root (scalar or 128-bit vector); the interpreter mirrors it with
            // `f32`/`f64::sqrt`, so the two agree bit-for-bit.
            Op::Sqrt(v) => {
                let x = self.val(*v);
                self.builder.ins().sqrt(x)
            }
            // Hardware round-to-integral (scalar or 128-bit vector); the interpreter mirrors each mode
            // with the matching `f32`/`f64` method (`Nearest` ⇒ `round_ties_even`), so the two agree.
            Op::Round(mode, v) => {
                let x = self.val(*v);
                match mode {
                    mercury_mir::RoundMode::Nearest => self.builder.ins().nearest(x),
                    mercury_mir::RoundMode::Floor => self.builder.ins().floor(x),
                    mercury_mir::RoundMode::Ceil => self.builder.ins().ceil(x),
                    mercury_mir::RoundMode::Trunc => self.builder.ins().trunc(x),
                }
            }
        };
        if let Some(r) = res {
            // Normalise the result to its declared type so `vmap[r]` always has the MIR type's
            // Cranelift type: `i1` -> low bit, and narrow floats rounded to width — both matching
            // the interpreter's per-result normalisation.
            let rty = self.ty_of(r).clone();
            let cv = if matches!(rty, MirType::I1) {
                self.builder.ins().band_imm(cv, 1)
            } else if rty.is_float() {
                let want = cl_type(&rty, self.ptr_ty).unwrap_or(types::F64);
                self.coerce_float(cv, want)
            } else {
                cv
            };
            self.set(r, cv);
        }
    }

    fn lower_bin(&mut self, op: BinOp, l: ValueId, r: ValueId, res_ty: &MirType) -> Value {
        use BinOp::*;
        // Float ops compute in their result type with operands coerced to it (the front-end's
        // float literals are loosely typed); this mirrors the interpreter, keeping the two exact.
        if op.is_float() {
            let target = cl_type(res_ty, self.ptr_ty).unwrap_or(types::F64);
            let a0 = self.val(l);
            let b0 = self.val(r);
            let a = self.coerce_float(a0, target);
            let b = self.coerce_float(b0, target);
            return match op {
                FAdd => self.builder.ins().fadd(a, b),
                FSub => self.builder.ins().fsub(a, b),
                FMul => self.builder.ins().fmul(a, b),
                FDiv => self.builder.ins().fdiv(a, b),
                // True fmod via a runtime call (Rust `%`), bit-identical to the interpreter. The old
                // `a - trunc(a/b)*b` identity drifts once `a/b` exceeds the mantissa's integer range
                // (e.g. `1e18 % 3` gave 0 instead of 1). y == 0 yields NaN in both, as before.
                FRem => {
                    let name = if target == types::F32 {
                        RT_FMOD_F32
                    } else {
                        RT_FMOD_F64
                    };
                    let fref = self.rt_refs[name];
                    let call = self.builder.ins().call(fref, &[a, b]);
                    self.builder.inst_results(call)[0]
                }
                _ => unreachable!(),
            };
        }
        let (a, b) = (self.val(l), self.val(r));
        match op {
            Add => self.builder.ins().iadd(a, b),
            Sub => self.builder.ins().isub(a, b),
            Mul => self.builder.ins().imul(a, b),
            FAdd | FSub | FMul | FDiv | FRem => unreachable!("handled above"),
            And => self.builder.ins().band(a, b),
            Or => self.builder.ins().bor(a, b),
            Xor => self.builder.ins().bxor(a, b),
            Shl => self.builder.ins().ishl(a, b),
            LShr => self.builder.ins().ushr(a, b),
            AShr => self.builder.ins().sshr(a, b),
            // Guard the divisor so a zero yields 0 instead of trapping, matching the interpreter.
            // Signed div/rem additionally trap the hardware on `INT_MIN / -1` (2's-complement
            // overflow), so fold a -1 divisor into the guard and patch the result to the
            // interpreter's wrapping value (`INT_MIN / -1 == INT_MIN`, `x % -1 == 0`). Unsigned
            // division never overflows, so -1 there is just an ordinary large divisor.
            SDiv | UDiv | SRem | URem => {
                let ty = self.dfg_ty(a);
                let zero = self.builder.ins().iconst(ty, 0);
                let one = self.builder.ins().iconst(ty, 1);
                let is_zero = self.builder.ins().icmp(IntCC::Equal, b, zero);
                let is_neg1 = if matches!(op, SDiv | SRem) {
                    let neg1 = self.builder.ins().iconst(ty, -1);
                    Some(self.builder.ins().icmp(IntCC::Equal, b, neg1))
                } else {
                    None
                };
                let danger = match is_neg1 {
                    Some(n) => self.builder.ins().bor(is_zero, n),
                    None => is_zero,
                };
                let safe = self.builder.ins().select(danger, one, b);
                let q = match op {
                    SDiv => self.builder.ins().sdiv(a, safe),
                    UDiv => self.builder.ins().udiv(a, safe),
                    SRem => self.builder.ins().srem(a, safe),
                    URem => self.builder.ins().urem(a, safe),
                    _ => unreachable!(),
                };
                // `INT_MIN / -1`: the divisor was swapped to 1, so sdiv produced `a` — negate it to
                // get `-a` (== INT_MIN for INT_MIN). `x % -1 == 0` already falls out of `srem(a, 1)`.
                let q = match (op, is_neg1) {
                    (SDiv, Some(n)) => {
                        let nega = self.builder.ins().ineg(a);
                        self.builder.ins().select(n, nega, q)
                    }
                    _ => q,
                };
                self.builder.ins().select(is_zero, zero, q)
            }
        }
    }

    fn lower_cast(&mut self, kind: CastKind, v: ValueId, to: &MirType) -> Value {
        let x = self.val(v);
        let from_ty = self.dfg_ty(x);
        let to_ty = cl_type(to, self.ptr_ty).unwrap_or(self.ptr_ty);
        use CastKind::*;
        match kind {
            SExt => self.resize_int(x, from_ty, to_ty, true),
            ZExt | Trunc => self.resize_int(x, from_ty, to_ty, false),
            FpToSi => self.builder.ins().fcvt_to_sint_sat(to_ty, x),
            FpToUi => self.builder.ins().fcvt_to_uint_sat(to_ty, x),
            SiToFp => self.builder.ins().fcvt_from_sint(to_ty, x),
            UiToFp => self.builder.ins().fcvt_from_uint(to_ty, x),
            FpExt => {
                if to_ty == from_ty {
                    x
                } else {
                    self.builder.ins().fpromote(to_ty, x)
                }
            }
            FpTrunc => {
                if matches!(to, MirType::BF16) {
                    // bf16's register type is f32; demote an f64 source first, then round to bf16.
                    let f = if from_ty == types::F64 {
                        self.builder.ins().fdemote(types::F32, x)
                    } else {
                        x
                    };
                    self.round_to_bf16(f)
                } else if matches!(to, MirType::F16) {
                    // f16's register type is f32; demote an f64 source first, then round to f16.
                    let f = if from_ty == types::F64 {
                        self.builder.ins().fdemote(types::F32, x)
                    } else {
                        x
                    };
                    self.round_to_f16(f)
                } else if to_ty == from_ty {
                    x
                } else {
                    self.builder.ins().fdemote(to_ty, x)
                }
            }
            Bitcast => {
                if to_ty == from_ty {
                    x
                } else {
                    self.builder.ins().bitcast(to_ty, MemFlags::new(), x)
                }
            }
            IntToPtr | PtrToInt => self.resize_int(x, from_ty, to_ty, false),
        }
    }

    /// Round an `f32` to bf16 precision and back to `f32`, emitting the *identical* integer
    /// arithmetic as `mercury_runtime::round_bf16` so the native backend and the interpreter agree
    /// bit-for-bit. Returns the rounded `f32` (its low 16 mantissa bits are zero).
    fn round_to_bf16(&mut self, x: Value) -> Value {
        let bits = self.builder.ins().bitcast(types::I32, MemFlags::new(), x);
        // non-NaN, round to nearest even: (bits + 0x7fff + ((bits>>16)&1)) & 0xffff0000
        let shr = self.builder.ins().ushr_imm(bits, 16);
        let lsb = self.builder.ins().band_imm(shr, 1);
        let bias = self.builder.ins().iadd_imm(lsb, 0x7fff);
        let summed = self.builder.ins().iadd(bits, bias);
        let nonnan = self
            .builder
            .ins()
            .band_imm(summed, 0xffff_0000u32 as i32 as i64);
        // NaN stays NaN (quiet): (bits & 0xffff0000) | 0x00400000
        let masked = self
            .builder
            .ins()
            .band_imm(bits, 0xffff_0000u32 as i32 as i64);
        let nanres = self.builder.ins().bor_imm(masked, 0x0040_0000);
        let absb = self.builder.ins().band_imm(bits, 0x7fff_ffff);
        let isnan = self
            .builder
            .ins()
            .icmp_imm(IntCC::UnsignedGreaterThan, absb, 0x7f80_0000);
        let resbits = self.builder.ins().select(isnan, nanres, nonnan);
        self.builder
            .ins()
            .bitcast(types::F32, MemFlags::new(), resbits)
    }

    /// Round an `f32` to f16 precision and back to `f32`, by calling the runtime shims
    /// (`mercury_f32_to_f16_bits` then `mercury_f16_bits_to_f32`). f16's exponent/mantissa layout has
    /// no cheap inline round like bf16, so a call to the *identical* `half`-crate conversion the
    /// interpreter uses keeps native == interp bit-for-bit. Returns the f16-rounded `f32`.
    fn round_to_f16(&mut self, x: Value) -> Value {
        let pack = self.rt_refs[RT_F32_TO_F16];
        let c1 = self.builder.ins().call(pack, &[x]);
        let bits = self.builder.inst_results(c1)[0]; // i32, low 16 = f16 bits
        let unpack = self.rt_refs[RT_F16_TO_F32];
        let c2 = self.builder.ins().call(unpack, &[bits]);
        self.builder.inst_results(c2)[0]
    }

    /// Sign- or zero-extend, truncate, or pass through an integer to a target width.
    fn resize_int(&mut self, x: Value, from: Type, to: Type, signed: bool) -> Value {
        match to.bits().cmp(&from.bits()) {
            std::cmp::Ordering::Equal => x,
            std::cmp::Ordering::Greater => {
                if signed {
                    self.builder.ins().sextend(to, x)
                } else {
                    self.builder.ins().uextend(to, x)
                }
            }
            std::cmp::Ordering::Less => self.builder.ins().ireduce(to, x),
        }
    }

    /// Promote/demote a float value to a target float width (no-op if already that width).
    fn coerce_float(&mut self, x: Value, to: Type) -> Value {
        let from = self.dfg_ty(x);
        if from == to || !from.is_float() || !to.is_float() {
            return x;
        }
        if to.bits() > from.bits() {
            self.builder.ins().fpromote(to, x)
        } else {
            self.builder.ins().fdemote(to, x)
        }
    }

    /// Coerce an index value to the pointer-width integer used for address arithmetic.
    fn ptr_int(&mut self, index: ValueId) -> Value {
        let x = self.val(index);
        let from = self.dfg_ty(x);
        self.resize_int(x, from, self.ptr_ty, true)
    }

    fn lower_call(&mut self, func: Symbol, args: &[ValueId]) -> Option<Value> {
        if let Some(&fref) = self.func_refs.get(&func) {
            let argv: Vec<Value> = args.iter().map(|a| self.val(*a)).collect();
            let call = self.builder.ins().call(fref, &argv);
            return self.builder.inst_results(call).first().copied();
        }
        let name = self.interner.resolve(func);
        // The parallel-for runtime entry: mercury_parallel_for(n, body_ptr, env_ptr).
        if name == RT_PARALLEL_FOR && args.len() == 3 {
            let n = self.coerce_to_i64(args[0]);
            let body = self.val(args[1]);
            let env = self.val(args[2]);
            let fref = self.rt_refs[RT_PARALLEL_FOR];
            self.builder.ins().call(fref, &[n, body, env]);
            return None;
        }
        // The GEMM microkernel: mercury_sgemm(a, b, c, m, k, n, beta) and its parallel / transposed
        // (nn.Linear, C = A·Bᵀ) variants — all share the (ptr,ptr,ptr,i64,i64,i64,i64) signature.
        if matches!(
            name,
            RT_SGEMM
                | RT_SGEMM_PARALLEL
                | RT_SGEMM_NT
                | RT_SGEMM_NT_PARALLEL
                | RT_SGEMM_TN
                | RT_SGEMM_TN_PARALLEL
                | RT_SGEMM_BF16_NT
                | RT_SGEMM_BF16_NT_PARALLEL
                | RT_SGEMM_F16_NT
                | RT_SGEMM_F16_NT_PARALLEL
                | RT_SGEMM_BF16_TN
                | RT_SGEMM_BF16_TN_PARALLEL
                | RT_SGEMM_F16_TN
                | RT_SGEMM_F16_TN_PARALLEL
        ) && args.len() == 7
        {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let c = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let k = self.coerce_to_i64(args[4]);
            let n = self.coerce_to_i64(args[5]);
            let beta = self.coerce_to_i64(args[6]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[a, b, c, m, k, n, beta]);
            return None;
        }
        // The fused-epilogue Linear: mercury_sgemm_nt_epi[_parallel](a, b, c, m, k, n, beta, bias, act)
        // — three pointers, four i64 (m,k,n,beta), a bias pointer, and an i64 activation code. The
        // `@parallel` variant and the bf16/f16 half-input twins (a/b are 2-byte-element pointers; the
        // kernel widens losslessly) all share the identical 9-arg signature, so just route by `name`.
        if matches!(
            name,
            RT_SGEMM_NT_EPI
                | RT_SGEMM_NT_EPI_PAR
                | RT_SGEMM_BF16_NT_EPI
                | RT_SGEMM_BF16_NT_EPI_PAR
                | RT_SGEMM_F16_NT_EPI
                | RT_SGEMM_F16_NT_EPI_PAR
        ) && args.len() == 9
        {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let c = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let k = self.coerce_to_i64(args[4]);
            let n = self.coerce_to_i64(args[5]);
            let beta = self.coerce_to_i64(args[6]);
            let bias = self.val(args[7]);
            let act = self.coerce_to_i64(args[8]);
            let fref = self.rt_refs[name];
            self.builder
                .ins()
                .call(fref, &[a, b, c, m, k, n, beta, bias, act]);
            return None;
        }
        // GEMV: mercury_sgemv[_parallel](a, x, y, m, n) — three pointers and two i64. Void.
        if matches!(name, RT_SGEMV | RT_SGEMV_PAR) && args.len() == 5 {
            let a = self.val(args[0]);
            let x = self.val(args[1]);
            let y = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let n = self.coerce_to_i64(args[4]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[a, x, y, m, n]);
            return None;
        }
        // α-scaled Linear: mercury_sgemm_nt_alpha[_parallel](a, b, c, m, k, n, beta, alpha) — three
        // pointers, four i64, and one f32 scalar `alpha` (passed by value, like the i8 dequant scale).
        if matches!(name, RT_SGEMM_NT_ALPHA | RT_SGEMM_NT_ALPHA_PAR) && args.len() == 8 {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let c = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let k = self.coerce_to_i64(args[4]);
            let n = self.coerce_to_i64(args[5]);
            let beta = self.coerce_to_i64(args[6]);
            let alpha = self.val(args[7]);
            let fref = self.rt_refs[name];
            self.builder
                .ins()
                .call(fref, &[a, b, c, m, k, n, beta, alpha]);
            return None;
        }
        // The vectorized elementwise transcendental: mercury_vmath_f32(x, out, n, op) — two pointers
        // and two i64 (element count, op code). The 256-bit AVX2 kernel an `out[i]=f(x[i])` loop
        // lowers to.
        // Same 4-arg shape for the bf16/f16-input twins (x is a 2-byte-element pointer; the kernel
        // widens losslessly). Identical signature, so just route by name.
        if (name == RT_VMATH || name == RT_VMATH_BF16 || name == RT_VMATH_F16) && args.len() == 4 {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let n = self.coerce_to_i64(args[2]);
            let op = self.coerce_to_i64(args[3]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[x, out, n, op]);
            return None;
        }
        // Batched log-sum-exp: mercury_logsumexp_f32[_parallel](x, out, rows, cols) — two pointers +
        // two i64, the same vmath shape; route by name. Void. Row-entropy and the per-row arg-reductions
        // (`mercury_rowarg{max,min}_i32`, whose `out` is an i32 buffer — a pointer rides the slot either
        // way) share the exact `(ptr, ptr, i64, i64)` shape.
        if matches!(
            name,
            RT_LOGSUMEXP
                | RT_LOGSUMEXP_PAR
                | RT_ENTROPY
                | RT_ENTROPY_PAR
                | RT_ROWARGMAX
                | RT_ROWARGMAX_PAR
                | RT_ROWARGMIN
                | RT_ROWARGMIN_PAR
                | RT_COLARGMAX
                | RT_COLARGMAX_PAR
                | RT_COLARGMIN
                | RT_COLARGMIN_PAR
                | RT_CUMSUM
                | RT_CUMSUM_PAR
                | RT_CUMMAX
                | RT_CUMMAX_PAR
                | RT_CUMMIN
                | RT_CUMMIN_PAR
                | RT_CUMPROD
                | RT_CUMPROD_PAR
        ) && args.len() == 4
        {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let rows = self.coerce_to_i64(args[2]);
            let cols = self.coerce_to_i64(args[3]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[x, out, rows, cols]);
            return None;
        }
        // KL divergence / soft-label cross-entropy: (p|x, q, out, rows, cols) — three pointers + two
        // i64, the same vmath2 shape; route by name. Void.
        if matches!(name, RT_KLDIV | RT_KLDIV_PAR | RT_KD_LOSS | RT_KD_LOSS_PAR) && args.len() == 5 {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let out = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let cols = self.coerce_to_i64(args[4]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[a, b, out, rows, cols]);
            return None;
        }
        // The cache-blocked transpose: mercury_transpose_f32[_parallel](src, dst, rows, cols) — two
        // pointers and two i64. Same (ptr, ptr, i64, i64) signature as vmath; route by name.
        if matches!(
            name,
            RT_TRANSPOSE | RT_TRANSPOSE_PAR | RT_TRANSPOSE_U16 | RT_TRANSPOSE_U16_PAR
        ) && args.len() == 4
        {
            let src = self.val(args[0]);
            let dst = self.val(args[1]);
            let rows = self.coerce_to_i64(args[2]);
            let cols = self.coerce_to_i64(args[3]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[src, dst, rows, cols]);
            return None;
        }
        // 2D pooling: mercury_{max,avg}pool2d_f32[_parallel](x, out, channels, h, w, kh, kw, sh, sw) —
        // two pointers and seven i64. Route by name to the declared import.
        if matches!(
            name,
            RT_MAXPOOL2D | RT_MAXPOOL2D_PAR | RT_AVGPOOL2D | RT_AVGPOOL2D_PAR
        ) && args.len() == 9
        {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let mut call_args = vec![x, out];
            for a in &args[2..] {
                call_args.push(self.coerce_to_i64(*a));
            }
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &call_args);
            return None;
        }
        // The SIMD column reductions: mercury_col{sum,max,min,maxabs,mean,sumsq,l2,rms}_f32[_parallel]
        // (x, out, rows, cols) — two pointers and two i64. Same (ptr, ptr, i64, i64) signature; by name.
        if matches!(
            name,
            RT_COLSUM
                | RT_COLSUM_PAR
                | RT_COLMAX
                | RT_COLMAX_PAR
                | RT_COLMIN
                | RT_COLMIN_PAR
                | RT_COLMAXABS
                | RT_COLMAXABS_PAR
                | RT_COLMEAN
                | RT_COLMEAN_PAR
                | RT_COLSUMSQ
                | RT_COLSUMSQ_PAR
                | RT_COLL2
                | RT_COLL2_PAR
                | RT_COLRMS
                | RT_COLRMS_PAR
        ) && args.len() == 4
        {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let rows = self.coerce_to_i64(args[2]);
            let cols = self.coerce_to_i64(args[3]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[x, out, rows, cols]);
            return None;
        }
        // The two-input transcendental: mercury_vmath2_f32(x, y, out, n, op) — three pointers, two i64
        // (element count, op code). The 256-bit AVX2 kernel an `out[i]=pow/atan2/hypot(x[i],y[i])`
        // loop lowers to.
        if name == RT_VMATH2 && args.len() == 5 {
            let x = self.val(args[0]);
            let y = self.val(args[1]);
            let out = self.val(args[2]);
            let n = self.coerce_to_i64(args[3]);
            let op = self.coerce_to_i64(args[4]);
            let fref = self.rt_refs[RT_VMATH2];
            self.builder.ins().call(fref, &[x, y, out, n, op]);
            return None;
        }
        // Fused softmax-backward: mercury_softmax_bwd_f32[_parallel](y, dy, dx, rows, cols) — three
        // pointers, two i64. Same (ptr, ptr, ptr, i64, i64) shape as vmath2; route by name. Void.
        if matches!(name, RT_SOFTMAX_BWD | RT_SOFTMAX_BWD_PAR) && args.len() == 5 {
            let y = self.val(args[0]);
            let dy = self.val(args[1]);
            let dx = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let cols = self.coerce_to_i64(args[4]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[y, dy, dx, rows, cols]);
            return None;
        }

        // Linear-recurrence scan: mercury_lrscan_f32[_parallel](a, b, out, rows, cols) — three pointers,
        // two i64 (same (ptr,ptr,ptr,i64,i64) shape as softmax_bwd); route by name. Void.
        if matches!(name, RT_LRSCAN | RT_LRSCAN_PAR) && args.len() == 5 {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let out = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let cols = self.coerce_to_i64(args[4]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[a, b, out, rows, cols]);
            return None;
        }
        // Fused cross-entropy loss: mercury_xent_fwd_f32[_parallel](x, target, loss, rows, cols) — three
        // pointers (x f32, target i32, loss f32 — a ptr is a ptr) + two i64, the same vmath2 shape. Void.
        if matches!(name, RT_XENT | RT_XENT_PAR) && args.len() == 5 {
            let x = self.val(args[0]);
            let target = self.val(args[1]);
            let loss = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let cols = self.coerce_to_i64(args[4]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[x, target, loss, rows, cols]);
            return None;
        }
        // Cross-entropy backward: mercury_xent_bwd_f32[_parallel](x, target, dx, rows, cols) — same
        // vmath2 shape (3 ptr + 2 i64). Void.
        if matches!(name, RT_XENT_BWD | RT_XENT_BWD_PAR) && args.len() == 5 {
            let x = self.val(args[0]);
            let target = self.val(args[1]);
            let dx = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let cols = self.coerce_to_i64(args[4]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[x, target, dx, rows, cols]);
            return None;
        }
        // RoPE: mercury_rope_f32[_parallel](x, inv_freq, out, rows, half) — three f32 pointers + two
        // i64, the same vmath2 shape. Void.
        if matches!(name, RT_ROPE | RT_ROPE_PAR | RT_ROPE_BWD | RT_ROPE_BWD_PAR) && args.len() == 5 {
            let x = self.val(args[0]);
            let inv_freq = self.val(args[1]);
            let out = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let half = self.coerce_to_i64(args[4]);
            let fref = self.rt_refs[name];
            self.builder
                .ins()
                .call(fref, &[x, inv_freq, out, rows, half]);
            return None;
        }
        // Fused RMSNorm-backward: mercury_rmsnorm_bwd_f32[_parallel](x, dy, gamma, dx, rows, cols,
        // eps_bits) — four pointers, three i64. The two per-row reductions fold 8-wide. Void.
        if matches!(
            name,
            RT_RMSNORM_BWD | RT_RMSNORM_BWD_PAR | RT_LAYERNORM_BWD | RT_LAYERNORM_BWD_PAR
        ) && args.len() == 7
        {
            let x = self.val(args[0]);
            let dy = self.val(args[1]);
            let gamma = self.val(args[2]);
            let dx = self.val(args[3]);
            let rows = self.coerce_to_i64(args[4]);
            let cols = self.coerce_to_i64(args[5]);
            let eps = self.coerce_to_i64(args[6]);
            let fref = self.rt_refs[name];
            self.builder
                .ins()
                .call(fref, &[x, dy, gamma, dx, rows, cols, eps]);
            return None;
        }
        // The streaming affine+activation kernel: mercury_velem_f32(x, y, out, n, a, b, c, op) — three
        // pointers, one i64 count, three f32 coefficients, one i64 op (256-bit AVX2 + non-temporal
        // stores, the saxpy/scale/add/bias/ReLU map a recognized streaming loop lowers to). Void.
        if name == RT_VELEM && args.len() == 8 {
            let x = self.val(args[0]);
            let y = self.val(args[1]);
            let out = self.val(args[2]);
            let n = self.coerce_to_i64(args[3]);
            let a = self.val(args[4]);
            let b = self.val(args[5]);
            let c = self.val(args[6]);
            let op = self.coerce_to_i64(args[7]);
            let fref = self.rt_refs[RT_VELEM];
            self.builder.ins().call(fref, &[x, y, out, n, a, b, c, op]);
            return None;
        }
        // The streaming Horner-polynomial kernel: mercury_vhorner_f32(x, out, n, coeffs, ncoeff) —
        // three pointers (x, out, coeffs) and two i64 (element count, coefficient count). Void.
        if name == RT_VHORNER && args.len() == 5 {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let n = self.coerce_to_i64(args[2]);
            let coeffs = self.val(args[3]);
            let ncoeff = self.coerce_to_i64(args[4]);
            let fref = self.rt_refs[RT_VHORNER];
            self.builder.ins().call(fref, &[x, out, n, coeffs, ncoeff]);
            return None;
        }
        // The deterministic reduction kernel: mercury_sreduce_f32[_parallel](x, y, n, op) -> f32 —
        // two pointers, two i64, and an f32 scalar result (the dot/ssd/sum a `@parallel` reduction
        // loop lowers to). Unlike the void kernels above, this returns the accumulated value.
        if matches!(name, RT_SREDUCE | RT_SREDUCE_PARALLEL) && args.len() == 4 {
            let x = self.val(args[0]);
            let y = self.val(args[1]);
            let n = self.coerce_to_i64(args[2]);
            let op = self.coerce_to_i64(args[3]);
            let fref = self.rt_refs[name];
            let call = self.builder.ins().call(fref, &[x, y, n, op]);
            return self.builder.inst_results(call).first().copied();
        }
        // The deterministic arg-reduction: mercury_argreduce_f32(x, n, op) -> i64 (the argmax/argmin a
        // recognized `if x[k] CMP bv {...}` loop reconciles against). One pointer, two i64, an i64
        // index result — bind the result like the sreduce kernel above.
        if (name == RT_ARGREDUCE || name == RT_ARGREDUCE_PARALLEL) && args.len() == 3 {
            let x = self.val(args[0]);
            let n = self.coerce_to_i64(args[1]);
            let op = self.coerce_to_i64(args[2]);
            let fref = self.rt_refs[name];
            let call = self.builder.ins().call(fref, &[x, n, op]);
            return self.builder.inst_results(call).first().copied();
        }
        // The bf16 mixed-precision reductions: mercury_dot_bf16(x, y, n) -> f32 (3 args) and
        // mercury_sum_bf16(x, n) -> f32 (2 args). bf16 storage, f32 accumulate; both return the
        // accumulated f32, so bind the call result like the sreduce kernel above.
        if (name == RT_DOT_BF16 || name == RT_DOT_F16) && args.len() == 3 {
            let x = self.val(args[0]);
            let y = self.val(args[1]);
            let n = self.coerce_to_i64(args[2]);
            let fref = self.rt_refs[name];
            let call = self.builder.ins().call(fref, &[x, y, n]);
            return self.builder.inst_results(call).first().copied();
        }
        if (name == RT_SUM_BF16 || name == RT_SUM_F16) && args.len() == 2 {
            let x = self.val(args[0]);
            let n = self.coerce_to_i64(args[1]);
            let fref = self.rt_refs[name];
            let call = self.builder.ins().call(fref, &[x, n]);
            return self.builder.inst_results(call).first().copied();
        }
        // mercury_reduce_{bf16,f16}(x, n, op) -> f32 — the max-family reduction (max/min/absmax). Same
        // f32 return as the sum/dot kernels above; the op selects the fold inside the kernel.
        if (name == RT_REDUCE_BF16 || name == RT_REDUCE_F16) && args.len() == 3 {
            let x = self.val(args[0]);
            let n = self.coerce_to_i64(args[1]);
            let op = self.coerce_to_i64(args[2]);
            let fref = self.rt_refs[name];
            let call = self.builder.ins().call(fref, &[x, n, op]);
            return self.builder.inst_results(call).first().copied();
        }
        // The bf16/f16 mixed-precision streaming axpby: mercury_axpby_{bf16,f16}(x, y, out, n, a, b) —
        // two half-precision input pointers, one f32 output pointer, an i64 count, and two f32
        // coefficients (half in, f32 out, f32 math — the saxpy/axpby a recognized loop lowers to). Void.
        if (name == RT_AXPBY_BF16 || name == RT_AXPBY_F16) && args.len() == 6 {
            let x = self.val(args[0]);
            let y = self.val(args[1]);
            let out = self.val(args[2]);
            let n = self.coerce_to_i64(args[3]);
            let a = self.val(args[4]);
            let b = self.val(args[5]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[x, y, out, n, a, b]);
            return None;
        }
        // The fused normalization kernel: mercury_norm_f32[_parallel](x, out, rows, cols, eps_bits,
        // op) — two pointers and four i64 (the softmax/LayerNorm/RMSNorm a recognized multi-pass norm
        // lowers to). Void, like the GEMM/vmath kernels.
        if matches!(name, RT_NORM | RT_NORM_PARALLEL) && args.len() == 6 {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let rows = self.coerce_to_i64(args[2]);
            let cols = self.coerce_to_i64(args[3]);
            let eps = self.coerce_to_i64(args[4]);
            let op = self.coerce_to_i64(args[5]);
            let fref = self.rt_refs[name];
            self.builder
                .ins()
                .call(fref, &[x, out, rows, cols, eps, op]);
            return None;
        }
        // The affine fused norm: mercury_norm_affine_f32[_parallel](x, out, gamma, beta, rows, cols,
        // eps_bits, op) — four pointers (gamma/beta may be null) and four i64 (LayerNorm/RMSNorm with a
        // learned per-column scale/shift). Void, like the plain norm; the _parallel one maps rows across
        // cores (bit-equal to serial — rows independent).
        if matches!(name, RT_NORM_AFFINE | RT_NORM_AFFINE_PARALLEL) && args.len() == 8 {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let gamma = self.val(args[2]);
            let beta = self.val(args[3]);
            let rows = self.coerce_to_i64(args[4]);
            let cols = self.coerce_to_i64(args[5]);
            let eps = self.coerce_to_i64(args[6]);
            let op = self.coerce_to_i64(args[7]);
            let fref = self.rt_refs[name];
            self.builder
                .ins()
                .call(fref, &[x, out, gamma, beta, rows, cols, eps, op]);
            return None;
        }
        // The int8 quantized nn.Linear: mercury_i8gemm_nt[_parallel](a, b, c, m, k, n) — three
        // pointers and three i64 (no beta; the kernel always overwrites C). Void.
        if matches!(name, RT_I8GEMM_NT | RT_I8GEMM_NT_PARALLEL) && args.len() == 6 {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let c = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let k = self.coerce_to_i64(args[4]);
            let n = self.coerce_to_i64(args[5]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[a, b, c, m, k, n]);
            return None;
        }
        // Embedding lookup: mercury_embedding_f32[_parallel](out, weight, ids, t, h, v) — three pointers
        // (out/weight f32*, ids i32*) and three i64 dims (the table height `v` rides in as a large
        // sentinel from the recognizer, so the kernel's out-of-range clamp never fires). Void.
        if matches!(name, RT_EMBEDDING | RT_EMBEDDING_PAR) && args.len() == 6 {
            let out = self.val(args[0]);
            let weight = self.val(args[1]);
            let ids = self.val(args[2]);
            let t = self.coerce_to_i64(args[3]);
            let h = self.coerce_to_i64(args[4]);
            let v = self.coerce_to_i64(args[5]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[out, weight, ids, t, h, v]);
            return None;
        }

        // Scatter-add / embedding-gradient backward: mercury_scatter_add_f32[_parallel](grad_w, grad_out,
        // ids, t, h, v) — same 3-ptr + 3-i64 ABI as the embedding gather (grad_w/grad_out f32*, ids i32*),
        // but `v` is the real table height (the parallel kernel partitions its output rows). Void.
        if matches!(name, RT_SCATTER_ADD | RT_SCATTER_ADD_PAR) && args.len() == 6 {
            let grad_w = self.val(args[0]);
            let grad_out = self.val(args[1]);
            let ids = self.val(args[2]);
            let t = self.coerce_to_i64(args[3]);
            let h = self.coerce_to_i64(args[4]);
            let v = self.coerce_to_i64(args[5]);
            let fref = self.rt_refs[name];
            self.builder.ins().call(fref, &[grad_w, grad_out, ids, t, h, v]);
            return None;
        }
        // The int8 quantized nn.Linear with fused dequant epilogue:
        // mercury_i8gemm_nt_deq[_parallel](a, b, out, m, k, n, scale_a, scale_b, bias, act) — three
        // pointers, three i64, one f32 scalar (scale_a), two pointers (scale_b, bias; bias may be
        // null), one i64 act code. Void.
        if matches!(name, RT_I8GEMM_NT_DEQ | RT_I8GEMM_NT_DEQ_PARALLEL) && args.len() == 10 {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let out = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let k = self.coerce_to_i64(args[4]);
            let n = self.coerce_to_i64(args[5]);
            let scale_a = self.val(args[6]);
            let scale_b = self.val(args[7]);
            let bias = self.val(args[8]);
            let act = self.coerce_to_i64(args[9]);
            let fref = self.rt_refs[name];
            self.builder
                .ins()
                .call(fref, &[a, b, out, m, k, n, scale_a, scale_b, bias, act]);
            return None;
        }
        let arg_is_float = args
            .first()
            .map(|a| self.ty_of(*a).is_float())
            .unwrap_or(false);
        let intr = classify_intrinsic(name, arg_is_float)?;
        match intr {
            Intrinsic::PrintInt => {
                if let Some(&a) = args.first() {
                    let v = self.coerce_to_i64(a);
                    let fref = self.rt_refs[RT_PRINT_I64];
                    self.builder.ins().call(fref, &[v]);
                }
            }
            Intrinsic::PrintFloat => {
                if let Some(&a) = args.first() {
                    let v = self.coerce_to_f64(a);
                    let fref = self.rt_refs[RT_PRINT_F64];
                    self.builder.ins().call(fref, &[v]);
                }
            }
            Intrinsic::Assert => {
                if let Some(&a) = args.first() {
                    let v = self.coerce_to_i64(a);
                    let fref = self.rt_refs[RT_ASSERT];
                    self.builder.ins().call(fref, &[v]);
                }
            }
        }
        None
    }

    fn coerce_to_i64(&mut self, v: ValueId) -> Value {
        let x = self.val(v);
        let from = self.dfg_ty(x);
        if from.is_int() {
            self.resize_int(x, from, types::I64, true)
        } else {
            self.builder.ins().fcvt_to_sint_sat(types::I64, x)
        }
    }

    fn coerce_to_f64(&mut self, v: ValueId) -> Value {
        let x = self.val(v);
        let from = self.dfg_ty(x);
        if from == types::F64 {
            x
        } else {
            self.builder.ins().fpromote(types::F64, x)
        }
    }

    fn lower_term(&mut self, term: &Terminator) {
        match term {
            Terminator::Ret(None) => {
                self.builder.ins().return_(&[]);
            }
            Terminator::Ret(Some(v)) => {
                let x = self.val(*v);
                self.builder.ins().return_(&[x]);
            }
            Terminator::Br { target, args } => {
                let argv: Vec<BlockArg> =
                    args.iter().map(|a| BlockArg::Value(self.val(*a))).collect();
                let cb = self.blocks[target.0 as usize];
                self.builder.ins().jump(cb, &argv);
            }
            Terminator::CondBr {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                let c = self.val(*cond);
                let ta: Vec<BlockArg> = then_args
                    .iter()
                    .map(|a| BlockArg::Value(self.val(*a)))
                    .collect();
                let ea: Vec<BlockArg> = else_args
                    .iter()
                    .map(|a| BlockArg::Value(self.val(*a)))
                    .collect();
                let tb = self.blocks[then_blk.0 as usize];
                let eb = self.blocks[else_blk.0 as usize];
                self.builder.ins().brif(c, tb, &ta, eb, &ea);
            }
            Terminator::Unreachable => {
                let tc = cranelift_codegen::ir::TrapCode::user(1).unwrap();
                self.builder.ins().trap(tc);
            }
        }
    }
}

// --- Module-level driving (shared by JIT and object emission) -------------------------------

struct RtFuncs {
    print_i64: FuncId,
    print_f64: FuncId,
    assert: FuncId,
    parallel_for: FuncId,
    sgemm: FuncId,
    sgemm_parallel: FuncId,
    sgemm_nt: FuncId,
    sgemm_nt_parallel: FuncId,
    sgemm_tn: FuncId,
    sgemm_tn_parallel: FuncId,
    sgemm_bf16_nt: FuncId,
    sgemm_bf16_nt_parallel: FuncId,
    sgemm_f16_nt: FuncId,
    sgemm_f16_nt_parallel: FuncId,
    sgemm_bf16_tn: FuncId,
    sgemm_bf16_tn_parallel: FuncId,
    sgemm_f16_tn: FuncId,
    sgemm_f16_tn_parallel: FuncId,
    sgemm_nt_epi: FuncId,
    sgemm_nt_epi_par: FuncId,
    sgemv: FuncId,
    sgemv_par: FuncId,
    sgemm_nt_alpha: FuncId,
    sgemm_nt_alpha_par: FuncId,
    sgemm_bf16_nt_epi: FuncId,
    sgemm_bf16_nt_epi_par: FuncId,
    sgemm_f16_nt_epi: FuncId,
    sgemm_f16_nt_epi_par: FuncId,
    vmath: FuncId,
    vmath2: FuncId,
    softmax_bwd: FuncId,
    softmax_bwd_par: FuncId,
    lrscan: FuncId,
    lrscan_par: FuncId,
    rmsnorm_bwd: FuncId,
    rmsnorm_bwd_par: FuncId,
    layernorm_bwd: FuncId,
    layernorm_bwd_par: FuncId,
    xent: FuncId,
    xent_par: FuncId,
    xent_bwd: FuncId,
    xent_bwd_par: FuncId,
    rope: FuncId,
    rope_par: FuncId,
    rope_bwd: FuncId,
    rope_bwd_par: FuncId,
    logsumexp: FuncId,
    logsumexp_par: FuncId,
    kldiv: FuncId,
    kldiv_par: FuncId,
    entropy: FuncId,
    entropy_par: FuncId,
    rowargmax: FuncId,
    rowargmax_par: FuncId,
    rowargmin: FuncId,
    rowargmin_par: FuncId,
    colargmax: FuncId,
    colargmax_par: FuncId,
    colargmin: FuncId,
    colargmin_par: FuncId,
    cumsum: FuncId,
    cumsum_par: FuncId,
    cumprod: FuncId,
    cumprod_par: FuncId,
    cummax: FuncId,
    cummax_par: FuncId,
    cummin: FuncId,
    cummin_par: FuncId,
    maxpool2d: FuncId,
    maxpool2d_par: FuncId,
    avgpool2d: FuncId,
    avgpool2d_par: FuncId,
    kd_loss: FuncId,
    kd_loss_par: FuncId,
    vmath_bf16: FuncId,
    vmath_f16: FuncId,
    transpose: FuncId,
    transpose_par: FuncId,
    transpose_u16: FuncId,
    transpose_u16_par: FuncId,
    colsum: FuncId,
    colsum_par: FuncId,
    colmax: FuncId,
    colmax_par: FuncId,
    colmin: FuncId,
    colmin_par: FuncId,
    colmaxabs: FuncId,
    colmaxabs_par: FuncId,
    colmean: FuncId,
    colmean_par: FuncId,
    colsumsq: FuncId,
    colsumsq_par: FuncId,
    coll2: FuncId,
    coll2_par: FuncId,
    colrms: FuncId,
    colrms_par: FuncId,
    velem: FuncId,
    vhorner: FuncId,
    sred: FuncId,
    sred_par: FuncId,
    argreduce: FuncId,
    argreduce_par: FuncId,
    norm: FuncId,
    norm_par: FuncId,
    norm_affine: FuncId,
    norm_affine_par: FuncId,
    i8nt: FuncId,
    i8nt_par: FuncId,
    i8nt_deq: FuncId,
    i8nt_deq_par: FuncId,
    /// Embedding lookup `mercury_embedding_f32[_parallel](out, weight, ids, t, h, v)` — token-id row
    /// gather (the first layer of every LLM). Reuses the 3-ptr + 3-i64 void `sig_i8gemm` signature.
    embedding: FuncId,
    embedding_par: FuncId,
    scatter_add: FuncId,
    scatter_add_par: FuncId,
    dot_bf16: FuncId,
    sum_bf16: FuncId,
    reduce_bf16: FuncId,
    axpby_f16: FuncId,
    dot_f16: FuncId,
    sum_f16: FuncId,
    reduce_f16: FuncId,
    f32_to_f16: FuncId,
    f16_to_f32: FuncId,
    axpby_bf16: FuncId,
    fmod_f64: FuncId,
    fmod_f32: FuncId,
}

fn signature_of(
    f: &Function,
    ptr_ty: Type,
    call_conv: cranelift_codegen::isa::CallConv,
) -> Signature {
    let mut sig = Signature::new(call_conv);
    for p in &f.params {
        if let Some(t) = cl_type(f.value_type(*p), ptr_ty) {
            sig.params.push(AbiParam::new(t));
        }
    }
    if let Some(t) = cl_type(&f.ret, ptr_ty) {
        sig.returns.push(AbiParam::new(t));
    }
    sig
}

fn make_isa(pic: bool) -> Result<std::sync::Arc<dyn cranelift_codegen::isa::TargetIsa>, String> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| e.to_string())?;
    // The JIT requires non-PIC; the object emitter wants PIC. Configure per caller.
    flag_builder
        .set("is_pic", if pic { "true" } else { "false" })
        .map_err(|e| e.to_string())?;
    // Emit stack-probe code in any prologue whose frame exceeds a page, so a function with a large
    // local array (e.g. an im2col scratch buffer) touches each guard page instead of jumping past it
    // and faulting. The *inline* strategy emits the probe loop directly — no external `__chkstk`
    // symbol to bind, which the JIT cannot provide. Probing only commits stack pages, so it changes
    // no computed value and keeps the interpreter/native differential exact.
    flag_builder
        .set("enable_probestack", "true")
        .map_err(|e| e.to_string())?;
    flag_builder
        .set("probestack_strategy", "inline")
        .map_err(|e| e.to_string())?;
    let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
    isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| e.to_string())
}

/// Declare every user function and the runtime imports into `module`, then define each body.
fn populate_module<M: Module>(
    module: &mut M,
    program: &Program,
    interner: &Interner,
) -> Result<HashMap<Symbol, FuncId>, String> {
    let ptr_ty = module.target_config().pointer_type();
    let call_conv = module.target_config().default_call_conv;

    let mut sig_i = Signature::new(call_conv);
    sig_i.params.push(AbiParam::new(types::I64));
    let mut sig_f = Signature::new(call_conv);
    sig_f.params.push(AbiParam::new(types::F64));
    // mercury_parallel_for(n: i64, body: ptr, env: ptr)
    let mut sig_par = Signature::new(call_conv);
    sig_par.params.push(AbiParam::new(types::I64));
    sig_par.params.push(AbiParam::new(ptr_ty));
    sig_par.params.push(AbiParam::new(ptr_ty));
    // mercury_sgemm(a: ptr, b: ptr, c: ptr, m: i64, k: i64, n: i64, beta: i64)
    let mut sig_gemm = Signature::new(call_conv);
    for _ in 0..3 {
        sig_gemm.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..4 {
        sig_gemm.params.push(AbiParam::new(types::I64));
    }
    // mercury_sgemm_nt_epi(a, b, c: ptr, m, k, n, beta: i64, bias: ptr, act: i64) — fused Linear.
    let mut sig_gemm_epi = Signature::new(call_conv);
    for _ in 0..3 {
        sig_gemm_epi.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..4 {
        sig_gemm_epi.params.push(AbiParam::new(types::I64));
    }
    sig_gemm_epi.params.push(AbiParam::new(ptr_ty)); // bias
    sig_gemm_epi.params.push(AbiParam::new(types::I64)); // act
    // mercury_sgemv[_parallel](a: ptr, x: ptr, y: ptr, m: i64, n: i64) — matrix-times-vector (void).
    let mut sig_gemv = Signature::new(call_conv);
    for _ in 0..3 {
        sig_gemv.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..2 {
        sig_gemv.params.push(AbiParam::new(types::I64));
    }
    // mercury_sgemm_nt_alpha[_parallel](a, b, c: ptr, m, k, n, beta: i64, alpha: f32) — α-scaled Linear.
    let mut sig_gemm_alpha = Signature::new(call_conv);
    for _ in 0..3 {
        sig_gemm_alpha.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..4 {
        sig_gemm_alpha.params.push(AbiParam::new(types::I64));
    }
    sig_gemm_alpha.params.push(AbiParam::new(types::F32)); // alpha
    // mercury_vmath_f32(x: ptr, out: ptr, n: i64, op: i64) — vectorized elementwise transcendental.
    let mut sig_vmath = Signature::new(call_conv);
    sig_vmath.params.push(AbiParam::new(ptr_ty));
    sig_vmath.params.push(AbiParam::new(ptr_ty));
    sig_vmath.params.push(AbiParam::new(types::I64));
    sig_vmath.params.push(AbiParam::new(types::I64));
    // mercury_vmath2_f32(x, y, out: ptr, n: i64, op: i64) — two-input transcendental (pow/atan2/hypot).
    let mut sig_vmath2 = Signature::new(call_conv);
    sig_vmath2.params.push(AbiParam::new(ptr_ty));
    sig_vmath2.params.push(AbiParam::new(ptr_ty));
    sig_vmath2.params.push(AbiParam::new(ptr_ty));
    sig_vmath2.params.push(AbiParam::new(types::I64));
    sig_vmath2.params.push(AbiParam::new(types::I64));
    // mercury_velem_f32(x, y, out: ptr, n: i64, a, b, c: f32, op: i64) — streaming affine+activation.
    let mut sig_velem = Signature::new(call_conv);
    sig_velem.params.push(AbiParam::new(ptr_ty));
    sig_velem.params.push(AbiParam::new(ptr_ty));
    sig_velem.params.push(AbiParam::new(ptr_ty));
    sig_velem.params.push(AbiParam::new(types::I64));
    sig_velem.params.push(AbiParam::new(types::F32));
    sig_velem.params.push(AbiParam::new(types::F32));
    sig_velem.params.push(AbiParam::new(types::F32));
    sig_velem.params.push(AbiParam::new(types::I64));
    // mercury_vhorner_f32(x, out: ptr, n: i64, coeffs: ptr, ncoeff: i64) — streaming Horner poly.
    let mut sig_vhorner = Signature::new(call_conv);
    sig_vhorner.params.push(AbiParam::new(ptr_ty));
    sig_vhorner.params.push(AbiParam::new(ptr_ty));
    sig_vhorner.params.push(AbiParam::new(types::I64));
    sig_vhorner.params.push(AbiParam::new(ptr_ty));
    sig_vhorner.params.push(AbiParam::new(types::I64));
    // mercury_sreduce_f32[_parallel](x, y: ptr, n, op: i64) -> f32 — deterministic reduction kernel.
    let mut sig_sreduce = Signature::new(call_conv);
    sig_sreduce.params.push(AbiParam::new(ptr_ty));
    sig_sreduce.params.push(AbiParam::new(ptr_ty));
    sig_sreduce.params.push(AbiParam::new(types::I64));
    sig_sreduce.params.push(AbiParam::new(types::I64));
    sig_sreduce.returns.push(AbiParam::new(types::F32));
    // mercury_argreduce_f32(x: ptr, n, op: i64) -> i64 — the deterministic argmax/argmin index.
    let mut sig_argreduce = Signature::new(call_conv);
    sig_argreduce.params.push(AbiParam::new(ptr_ty));
    sig_argreduce.params.push(AbiParam::new(types::I64));
    sig_argreduce.params.push(AbiParam::new(types::I64));
    sig_argreduce.returns.push(AbiParam::new(types::I64));
    // mercury_norm_f32[_parallel](x, out: ptr, rows, cols, eps_bits, op: i64) — fused row-wise norm.
    let mut sig_norm = Signature::new(call_conv);
    sig_norm.params.push(AbiParam::new(ptr_ty));
    sig_norm.params.push(AbiParam::new(ptr_ty));
    for _ in 0..4 {
        sig_norm.params.push(AbiParam::new(types::I64));
    }
    // mercury_norm_affine_f32(x, out, gamma, beta: ptr, rows, cols, eps_bits, op: i64) — affine norm.
    let mut sig_norm_affine = Signature::new(call_conv);
    for _ in 0..4 {
        sig_norm_affine.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..4 {
        sig_norm_affine.params.push(AbiParam::new(types::I64));
    }
    // mercury_{max,avg}pool2d_f32[_parallel](x, out: ptr, channels, h, w, kh, kw, sh, sw: i64) — 2D
    // pooling (2 ptr + 7 i64, void).
    let mut sig_pool2d = Signature::new(call_conv);
    sig_pool2d.params.push(AbiParam::new(ptr_ty));
    sig_pool2d.params.push(AbiParam::new(ptr_ty));
    for _ in 0..7 {
        sig_pool2d.params.push(AbiParam::new(types::I64));
    }
    // mercury_rmsnorm_bwd_f32[_parallel](x, dy, gamma, dx: ptr, rows, cols, eps_bits: i64) — fused
    // batched RMSNorm input-gradient (4 ptr + 3 i64, void).
    let mut sig_rmsnorm_bwd = Signature::new(call_conv);
    for _ in 0..4 {
        sig_rmsnorm_bwd.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..3 {
        sig_rmsnorm_bwd.params.push(AbiParam::new(types::I64));
    }
    // mercury_i8gemm_nt[_parallel](a, b, c: ptr, m, k, n: i64) — int8 quantized nn.Linear (void).
    let mut sig_i8gemm = Signature::new(call_conv);
    for _ in 0..3 {
        sig_i8gemm.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..3 {
        sig_i8gemm.params.push(AbiParam::new(types::I64));
    }
    // mercury_i8gemm_nt_deq[_parallel](a, b, out: ptr, m, k, n: i64, scale_a: f32, scale_b, bias: ptr,
    // act: i64) — int8 quantized nn.Linear with fused dequant epilogue (void).
    let mut sig_i8gemm_deq = Signature::new(call_conv);
    for _ in 0..3 {
        sig_i8gemm_deq.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..3 {
        sig_i8gemm_deq.params.push(AbiParam::new(types::I64));
    }
    sig_i8gemm_deq.params.push(AbiParam::new(types::F32));
    sig_i8gemm_deq.params.push(AbiParam::new(ptr_ty));
    sig_i8gemm_deq.params.push(AbiParam::new(ptr_ty));
    sig_i8gemm_deq.params.push(AbiParam::new(types::I64));
    // mercury_dot_bf16(x, y: ptr, n: i64) -> f32 — bf16 mixed-precision dot (f32 accumulate).
    let mut sig_dot_bf16 = Signature::new(call_conv);
    sig_dot_bf16.params.push(AbiParam::new(ptr_ty));
    sig_dot_bf16.params.push(AbiParam::new(ptr_ty));
    sig_dot_bf16.params.push(AbiParam::new(types::I64));
    sig_dot_bf16.returns.push(AbiParam::new(types::F32));
    // mercury_sum_bf16(x: ptr, n: i64) -> f32 — bf16 mixed-precision sum (f32 accumulate).
    let mut sig_sum_bf16 = Signature::new(call_conv);
    sig_sum_bf16.params.push(AbiParam::new(ptr_ty));
    sig_sum_bf16.params.push(AbiParam::new(types::I64));
    sig_sum_bf16.returns.push(AbiParam::new(types::F32));
    // mercury_reduce_bf16(x: ptr, n: i64, op: i64) -> f32 — bf16 max/min/absmax reduction.
    let mut sig_reduce_bf16 = Signature::new(call_conv);
    sig_reduce_bf16.params.push(AbiParam::new(ptr_ty));
    sig_reduce_bf16.params.push(AbiParam::new(types::I64));
    sig_reduce_bf16.params.push(AbiParam::new(types::I64));
    sig_reduce_bf16.returns.push(AbiParam::new(types::F32));
    // mercury_f32_to_f16_bits(f32) -> i32 (low 16 = f16 bits); mercury_f16_bits_to_f32(i32) -> f32.
    let mut sig_f32_to_f16 = Signature::new(call_conv);
    sig_f32_to_f16.params.push(AbiParam::new(types::F32));
    sig_f32_to_f16.returns.push(AbiParam::new(types::I32));
    let mut sig_f16_to_f32 = Signature::new(call_conv);
    sig_f16_to_f32.params.push(AbiParam::new(types::I32));
    sig_f16_to_f32.returns.push(AbiParam::new(types::F32));
    // mercury_axpby_bf16(x, y: ptr<bf16>, out: ptr<f32>, n: i64, a, b: f32) — bf16→f32 axpby. Void.
    let mut sig_axpby_bf16 = Signature::new(call_conv);
    sig_axpby_bf16.params.push(AbiParam::new(ptr_ty));
    sig_axpby_bf16.params.push(AbiParam::new(ptr_ty));
    sig_axpby_bf16.params.push(AbiParam::new(ptr_ty));
    sig_axpby_bf16.params.push(AbiParam::new(types::I64));
    sig_axpby_bf16.params.push(AbiParam::new(types::F32));
    sig_axpby_bf16.params.push(AbiParam::new(types::F32));
    // mercury_rt_fmod_f64(a, b) -> f64 and the f32 variant — true fmod backing float `%`.
    let mut sig_fmod_f64 = Signature::new(call_conv);
    sig_fmod_f64.params.push(AbiParam::new(types::F64));
    sig_fmod_f64.params.push(AbiParam::new(types::F64));
    sig_fmod_f64.returns.push(AbiParam::new(types::F64));
    let mut sig_fmod_f32 = Signature::new(call_conv);
    sig_fmod_f32.params.push(AbiParam::new(types::F32));
    sig_fmod_f32.params.push(AbiParam::new(types::F32));
    sig_fmod_f32.returns.push(AbiParam::new(types::F32));
    let rt = RtFuncs {
        print_i64: module
            .declare_function(RT_PRINT_I64, Linkage::Import, &sig_i)
            .map_err(|e| e.to_string())?,
        print_f64: module
            .declare_function(RT_PRINT_F64, Linkage::Import, &sig_f)
            .map_err(|e| e.to_string())?,
        assert: module
            .declare_function(RT_ASSERT, Linkage::Import, &sig_i)
            .map_err(|e| e.to_string())?,
        parallel_for: module
            .declare_function(RT_PARALLEL_FOR, Linkage::Import, &sig_par)
            .map_err(|e| e.to_string())?,
        sgemm: module
            .declare_function(RT_SGEMM, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_parallel: module
            .declare_function(RT_SGEMM_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_nt: module
            .declare_function(RT_SGEMM_NT, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_nt_parallel: module
            .declare_function(RT_SGEMM_NT_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_tn: module
            .declare_function(RT_SGEMM_TN, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_tn_parallel: module
            .declare_function(RT_SGEMM_TN_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_bf16_nt: module
            .declare_function(RT_SGEMM_BF16_NT, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_bf16_nt_parallel: module
            .declare_function(RT_SGEMM_BF16_NT_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_f16_nt: module
            .declare_function(RT_SGEMM_F16_NT, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_f16_nt_parallel: module
            .declare_function(RT_SGEMM_F16_NT_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_bf16_tn: module
            .declare_function(RT_SGEMM_BF16_TN, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_bf16_tn_parallel: module
            .declare_function(RT_SGEMM_BF16_TN_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_f16_tn: module
            .declare_function(RT_SGEMM_F16_TN, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_f16_tn_parallel: module
            .declare_function(RT_SGEMM_F16_TN_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_nt_epi: module
            .declare_function(RT_SGEMM_NT_EPI, Linkage::Import, &sig_gemm_epi)
            .map_err(|e| e.to_string())?,
        sgemm_nt_epi_par: module
            .declare_function(RT_SGEMM_NT_EPI_PAR, Linkage::Import, &sig_gemm_epi)
            .map_err(|e| e.to_string())?,
        sgemv: module
            .declare_function(RT_SGEMV, Linkage::Import, &sig_gemv)
            .map_err(|e| e.to_string())?,
        sgemv_par: module
            .declare_function(RT_SGEMV_PAR, Linkage::Import, &sig_gemv)
            .map_err(|e| e.to_string())?,
        sgemm_nt_alpha: module
            .declare_function(RT_SGEMM_NT_ALPHA, Linkage::Import, &sig_gemm_alpha)
            .map_err(|e| e.to_string())?,
        sgemm_nt_alpha_par: module
            .declare_function(RT_SGEMM_NT_ALPHA_PAR, Linkage::Import, &sig_gemm_alpha)
            .map_err(|e| e.to_string())?,
        sgemm_bf16_nt_epi: module
            .declare_function(RT_SGEMM_BF16_NT_EPI, Linkage::Import, &sig_gemm_epi)
            .map_err(|e| e.to_string())?,
        sgemm_bf16_nt_epi_par: module
            .declare_function(RT_SGEMM_BF16_NT_EPI_PAR, Linkage::Import, &sig_gemm_epi)
            .map_err(|e| e.to_string())?,
        sgemm_f16_nt_epi: module
            .declare_function(RT_SGEMM_F16_NT_EPI, Linkage::Import, &sig_gemm_epi)
            .map_err(|e| e.to_string())?,
        sgemm_f16_nt_epi_par: module
            .declare_function(RT_SGEMM_F16_NT_EPI_PAR, Linkage::Import, &sig_gemm_epi)
            .map_err(|e| e.to_string())?,
        vmath: module
            .declare_function(RT_VMATH, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        vmath2: module
            .declare_function(RT_VMATH2, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        softmax_bwd: module
            .declare_function(RT_SOFTMAX_BWD, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        softmax_bwd_par: module
            .declare_function(RT_SOFTMAX_BWD_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        lrscan: module
            .declare_function(RT_LRSCAN, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        lrscan_par: module
            .declare_function(RT_LRSCAN_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        rmsnorm_bwd: module
            .declare_function(RT_RMSNORM_BWD, Linkage::Import, &sig_rmsnorm_bwd)
            .map_err(|e| e.to_string())?,
        rmsnorm_bwd_par: module
            .declare_function(RT_RMSNORM_BWD_PAR, Linkage::Import, &sig_rmsnorm_bwd)
            .map_err(|e| e.to_string())?,
        layernorm_bwd: module
            .declare_function(RT_LAYERNORM_BWD, Linkage::Import, &sig_rmsnorm_bwd)
            .map_err(|e| e.to_string())?,
        layernorm_bwd_par: module
            .declare_function(RT_LAYERNORM_BWD_PAR, Linkage::Import, &sig_rmsnorm_bwd)
            .map_err(|e| e.to_string())?,
        xent: module
            .declare_function(RT_XENT, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        xent_par: module
            .declare_function(RT_XENT_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        xent_bwd: module
            .declare_function(RT_XENT_BWD, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        xent_bwd_par: module
            .declare_function(RT_XENT_BWD_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        rope: module
            .declare_function(RT_ROPE, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        rope_par: module
            .declare_function(RT_ROPE_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        rope_bwd: module
            .declare_function(RT_ROPE_BWD, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        rope_bwd_par: module
            .declare_function(RT_ROPE_BWD_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        logsumexp: module
            .declare_function(RT_LOGSUMEXP, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        logsumexp_par: module
            .declare_function(RT_LOGSUMEXP_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        kldiv: module
            .declare_function(RT_KLDIV, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        kldiv_par: module
            .declare_function(RT_KLDIV_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        entropy: module
            .declare_function(RT_ENTROPY, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        entropy_par: module
            .declare_function(RT_ENTROPY_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        kd_loss: module
            .declare_function(RT_KD_LOSS, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        kd_loss_par: module
            .declare_function(RT_KD_LOSS_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        rowargmax: module
            .declare_function(RT_ROWARGMAX, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        rowargmax_par: module
            .declare_function(RT_ROWARGMAX_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        rowargmin: module
            .declare_function(RT_ROWARGMIN, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        rowargmin_par: module
            .declare_function(RT_ROWARGMIN_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colargmax: module
            .declare_function(RT_COLARGMAX, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colargmax_par: module
            .declare_function(RT_COLARGMAX_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colargmin: module
            .declare_function(RT_COLARGMIN, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colargmin_par: module
            .declare_function(RT_COLARGMIN_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cumsum: module
            .declare_function(RT_CUMSUM, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cumsum_par: module
            .declare_function(RT_CUMSUM_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cumprod: module
            .declare_function(RT_CUMPROD, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cumprod_par: module
            .declare_function(RT_CUMPROD_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cummax: module
            .declare_function(RT_CUMMAX, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cummax_par: module
            .declare_function(RT_CUMMAX_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cummin: module
            .declare_function(RT_CUMMIN, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cummin_par: module
            .declare_function(RT_CUMMIN_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        maxpool2d: module
            .declare_function(RT_MAXPOOL2D, Linkage::Import, &sig_pool2d)
            .map_err(|e| e.to_string())?,
        maxpool2d_par: module
            .declare_function(RT_MAXPOOL2D_PAR, Linkage::Import, &sig_pool2d)
            .map_err(|e| e.to_string())?,
        avgpool2d: module
            .declare_function(RT_AVGPOOL2D, Linkage::Import, &sig_pool2d)
            .map_err(|e| e.to_string())?,
        avgpool2d_par: module
            .declare_function(RT_AVGPOOL2D_PAR, Linkage::Import, &sig_pool2d)
            .map_err(|e| e.to_string())?,
        // bf16/f16-input twins: identical (ptr, ptr, i64, i64) signature.
        vmath_bf16: module
            .declare_function(RT_VMATH_BF16, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        vmath_f16: module
            .declare_function(RT_VMATH_F16, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        transpose: module
            .declare_function(RT_TRANSPOSE, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        transpose_par: module
            .declare_function(RT_TRANSPOSE_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        transpose_u16: module
            .declare_function(RT_TRANSPOSE_U16, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        transpose_u16_par: module
            .declare_function(RT_TRANSPOSE_U16_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colsum: module
            .declare_function(RT_COLSUM, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colsum_par: module
            .declare_function(RT_COLSUM_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmax: module
            .declare_function(RT_COLMAX, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmax_par: module
            .declare_function(RT_COLMAX_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmin: module
            .declare_function(RT_COLMIN, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmin_par: module
            .declare_function(RT_COLMIN_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmaxabs: module
            .declare_function(RT_COLMAXABS, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmaxabs_par: module
            .declare_function(RT_COLMAXABS_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmean: module
            .declare_function(RT_COLMEAN, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmean_par: module
            .declare_function(RT_COLMEAN_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colsumsq: module
            .declare_function(RT_COLSUMSQ, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colsumsq_par: module
            .declare_function(RT_COLSUMSQ_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        coll2: module
            .declare_function(RT_COLL2, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        coll2_par: module
            .declare_function(RT_COLL2_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colrms: module
            .declare_function(RT_COLRMS, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colrms_par: module
            .declare_function(RT_COLRMS_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        velem: module
            .declare_function(RT_VELEM, Linkage::Import, &sig_velem)
            .map_err(|e| e.to_string())?,
        vhorner: module
            .declare_function(RT_VHORNER, Linkage::Import, &sig_vhorner)
            .map_err(|e| e.to_string())?,
        sred: module
            .declare_function(RT_SREDUCE, Linkage::Import, &sig_sreduce)
            .map_err(|e| e.to_string())?,
        sred_par: module
            .declare_function(RT_SREDUCE_PARALLEL, Linkage::Import, &sig_sreduce)
            .map_err(|e| e.to_string())?,
        argreduce: module
            .declare_function(RT_ARGREDUCE, Linkage::Import, &sig_argreduce)
            .map_err(|e| e.to_string())?,
        argreduce_par: module
            .declare_function(RT_ARGREDUCE_PARALLEL, Linkage::Import, &sig_argreduce)
            .map_err(|e| e.to_string())?,
        norm: module
            .declare_function(RT_NORM, Linkage::Import, &sig_norm)
            .map_err(|e| e.to_string())?,
        norm_par: module
            .declare_function(RT_NORM_PARALLEL, Linkage::Import, &sig_norm)
            .map_err(|e| e.to_string())?,
        norm_affine: module
            .declare_function(RT_NORM_AFFINE, Linkage::Import, &sig_norm_affine)
            .map_err(|e| e.to_string())?,
        norm_affine_par: module
            .declare_function(RT_NORM_AFFINE_PARALLEL, Linkage::Import, &sig_norm_affine)
            .map_err(|e| e.to_string())?,
        i8nt: module
            .declare_function(RT_I8GEMM_NT, Linkage::Import, &sig_i8gemm)
            .map_err(|e| e.to_string())?,
        i8nt_par: module
            .declare_function(RT_I8GEMM_NT_PARALLEL, Linkage::Import, &sig_i8gemm)
            .map_err(|e| e.to_string())?,
        i8nt_deq: module
            .declare_function(RT_I8GEMM_NT_DEQ, Linkage::Import, &sig_i8gemm_deq)
            .map_err(|e| e.to_string())?,
        i8nt_deq_par: module
            .declare_function(RT_I8GEMM_NT_DEQ_PARALLEL, Linkage::Import, &sig_i8gemm_deq)
            .map_err(|e| e.to_string())?,
        // Embedding lookup reuses the 3-ptr + 3-i64 void signature (the pointer element type is
        // irrelevant to the ABI — out/weight are f32*, ids is i32*).
        embedding: module
            .declare_function(RT_EMBEDDING, Linkage::Import, &sig_i8gemm)
            .map_err(|e| e.to_string())?,
        embedding_par: module
            .declare_function(RT_EMBEDDING_PAR, Linkage::Import, &sig_i8gemm)
            .map_err(|e| e.to_string())?,
        scatter_add: module
            .declare_function(RT_SCATTER_ADD, Linkage::Import, &sig_i8gemm)
            .map_err(|e| e.to_string())?,
        scatter_add_par: module
            .declare_function(RT_SCATTER_ADD_PAR, Linkage::Import, &sig_i8gemm)
            .map_err(|e| e.to_string())?,
        axpby_bf16: module
            .declare_function(RT_AXPBY_BF16, Linkage::Import, &sig_axpby_bf16)
            .unwrap(),
        dot_bf16: module
            .declare_function(RT_DOT_BF16, Linkage::Import, &sig_dot_bf16)
            .map_err(|e| e.to_string())?,
        sum_bf16: module
            .declare_function(RT_SUM_BF16, Linkage::Import, &sig_sum_bf16)
            .map_err(|e| e.to_string())?,
        reduce_bf16: module
            .declare_function(RT_REDUCE_BF16, Linkage::Import, &sig_reduce_bf16)
            .map_err(|e| e.to_string())?,
        // f16 streaming/reductions reuse the bf16 signatures (same shapes; only the widen differs).
        axpby_f16: module
            .declare_function(RT_AXPBY_F16, Linkage::Import, &sig_axpby_bf16)
            .map_err(|e| e.to_string())?,
        dot_f16: module
            .declare_function(RT_DOT_F16, Linkage::Import, &sig_dot_bf16)
            .map_err(|e| e.to_string())?,
        sum_f16: module
            .declare_function(RT_SUM_F16, Linkage::Import, &sig_sum_bf16)
            .map_err(|e| e.to_string())?,
        reduce_f16: module
            .declare_function(RT_REDUCE_F16, Linkage::Import, &sig_reduce_bf16)
            .map_err(|e| e.to_string())?,
        f32_to_f16: module
            .declare_function(RT_F32_TO_F16, Linkage::Import, &sig_f32_to_f16)
            .map_err(|e| e.to_string())?,
        f16_to_f32: module
            .declare_function(RT_F16_TO_F32, Linkage::Import, &sig_f16_to_f32)
            .map_err(|e| e.to_string())?,
        fmod_f64: module
            .declare_function(RT_FMOD_F64, Linkage::Import, &sig_fmod_f64)
            .map_err(|e| e.to_string())?,
        fmod_f32: module
            .declare_function(RT_FMOD_F32, Linkage::Import, &sig_fmod_f32)
            .map_err(|e| e.to_string())?,
    };

    // Declare all user functions first so calls resolve regardless of definition order.
    let mut ids: HashMap<Symbol, FuncId> = HashMap::new();
    for f in &program.funcs {
        let sig = signature_of(f, ptr_ty, call_conv);
        let name = interner.resolve(f.name);
        let id = module
            .declare_function(name, Linkage::Export, &sig)
            .map_err(|e| e.to_string())?;
        ids.insert(f.name, id);
    }

    let mut ctx = module.make_context();
    let mut fbctx = FunctionBuilderContext::new();
    for f in &program.funcs {
        ctx.func.signature = signature_of(f, ptr_ty, call_conv);
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fbctx);

            // Pre-declare callee and runtime FuncRefs into this function's DFG.
            let mut func_refs: HashMap<Symbol, FuncRef> = HashMap::new();
            for (&sym, &fid) in &ids {
                let r = module.declare_func_in_func(fid, builder.func);
                func_refs.insert(sym, r);
            }
            let mut rt_refs: HashMap<&'static str, FuncRef> = HashMap::new();
            rt_refs.insert(
                RT_PRINT_I64,
                module.declare_func_in_func(rt.print_i64, builder.func),
            );
            rt_refs.insert(
                RT_PRINT_F64,
                module.declare_func_in_func(rt.print_f64, builder.func),
            );
            rt_refs.insert(
                RT_ASSERT,
                module.declare_func_in_func(rt.assert, builder.func),
            );
            rt_refs.insert(
                RT_PARALLEL_FOR,
                module.declare_func_in_func(rt.parallel_for, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM,
                module.declare_func_in_func(rt.sgemm, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_PARALLEL,
                module.declare_func_in_func(rt.sgemm_parallel, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_NT,
                module.declare_func_in_func(rt.sgemm_nt, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_NT_PARALLEL,
                module.declare_func_in_func(rt.sgemm_nt_parallel, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_TN,
                module.declare_func_in_func(rt.sgemm_tn, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_TN_PARALLEL,
                module.declare_func_in_func(rt.sgemm_tn_parallel, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_BF16_NT,
                module.declare_func_in_func(rt.sgemm_bf16_nt, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_BF16_NT_PARALLEL,
                module.declare_func_in_func(rt.sgemm_bf16_nt_parallel, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_F16_NT,
                module.declare_func_in_func(rt.sgemm_f16_nt, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_F16_NT_PARALLEL,
                module.declare_func_in_func(rt.sgemm_f16_nt_parallel, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_BF16_TN,
                module.declare_func_in_func(rt.sgemm_bf16_tn, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_BF16_TN_PARALLEL,
                module.declare_func_in_func(rt.sgemm_bf16_tn_parallel, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_F16_TN,
                module.declare_func_in_func(rt.sgemm_f16_tn, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_F16_TN_PARALLEL,
                module.declare_func_in_func(rt.sgemm_f16_tn_parallel, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_NT_EPI,
                module.declare_func_in_func(rt.sgemm_nt_epi, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_NT_EPI_PAR,
                module.declare_func_in_func(rt.sgemm_nt_epi_par, builder.func),
            );
            rt_refs.insert(
                RT_SGEMV,
                module.declare_func_in_func(rt.sgemv, builder.func),
            );
            rt_refs.insert(
                RT_SGEMV_PAR,
                module.declare_func_in_func(rt.sgemv_par, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_NT_ALPHA,
                module.declare_func_in_func(rt.sgemm_nt_alpha, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_NT_ALPHA_PAR,
                module.declare_func_in_func(rt.sgemm_nt_alpha_par, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_BF16_NT_EPI,
                module.declare_func_in_func(rt.sgemm_bf16_nt_epi, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_BF16_NT_EPI_PAR,
                module.declare_func_in_func(rt.sgemm_bf16_nt_epi_par, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_F16_NT_EPI,
                module.declare_func_in_func(rt.sgemm_f16_nt_epi, builder.func),
            );
            rt_refs.insert(
                RT_SGEMM_F16_NT_EPI_PAR,
                module.declare_func_in_func(rt.sgemm_f16_nt_epi_par, builder.func),
            );
            rt_refs.insert(
                RT_VMATH,
                module.declare_func_in_func(rt.vmath, builder.func),
            );
            rt_refs.insert(
                RT_VMATH2,
                module.declare_func_in_func(rt.vmath2, builder.func),
            );
            rt_refs.insert(
                RT_SOFTMAX_BWD,
                module.declare_func_in_func(rt.softmax_bwd, builder.func),
            );
            rt_refs.insert(
                RT_SOFTMAX_BWD_PAR,
                module.declare_func_in_func(rt.softmax_bwd_par, builder.func),
            );
            rt_refs.insert(RT_LRSCAN, module.declare_func_in_func(rt.lrscan, builder.func));
            rt_refs.insert(
                RT_LRSCAN_PAR,
                module.declare_func_in_func(rt.lrscan_par, builder.func),
            );
            rt_refs.insert(
                RT_RMSNORM_BWD,
                module.declare_func_in_func(rt.rmsnorm_bwd, builder.func),
            );
            rt_refs.insert(
                RT_RMSNORM_BWD_PAR,
                module.declare_func_in_func(rt.rmsnorm_bwd_par, builder.func),
            );
            rt_refs.insert(
                RT_LAYERNORM_BWD,
                module.declare_func_in_func(rt.layernorm_bwd, builder.func),
            );
            rt_refs.insert(
                RT_LAYERNORM_BWD_PAR,
                module.declare_func_in_func(rt.layernorm_bwd_par, builder.func),
            );
            rt_refs.insert(RT_XENT, module.declare_func_in_func(rt.xent, builder.func));
            rt_refs.insert(
                RT_XENT_PAR,
                module.declare_func_in_func(rt.xent_par, builder.func),
            );
            rt_refs.insert(
                RT_XENT_BWD,
                module.declare_func_in_func(rt.xent_bwd, builder.func),
            );
            rt_refs.insert(
                RT_XENT_BWD_PAR,
                module.declare_func_in_func(rt.xent_bwd_par, builder.func),
            );
            rt_refs.insert(RT_ROPE, module.declare_func_in_func(rt.rope, builder.func));
            rt_refs.insert(
                RT_ROPE_PAR,
                module.declare_func_in_func(rt.rope_par, builder.func),
            );
            rt_refs.insert(
                RT_ROPE_BWD,
                module.declare_func_in_func(rt.rope_bwd, builder.func),
            );
            rt_refs.insert(
                RT_ROPE_BWD_PAR,
                module.declare_func_in_func(rt.rope_bwd_par, builder.func),
            );
            rt_refs.insert(
                RT_LOGSUMEXP,
                module.declare_func_in_func(rt.logsumexp, builder.func),
            );
            rt_refs.insert(
                RT_LOGSUMEXP_PAR,
                module.declare_func_in_func(rt.logsumexp_par, builder.func),
            );
            rt_refs.insert(RT_KLDIV, module.declare_func_in_func(rt.kldiv, builder.func));
            rt_refs.insert(
                RT_KLDIV_PAR,
                module.declare_func_in_func(rt.kldiv_par, builder.func),
            );
            rt_refs.insert(
                RT_ENTROPY,
                module.declare_func_in_func(rt.entropy, builder.func),
            );
            rt_refs.insert(
                RT_ENTROPY_PAR,
                module.declare_func_in_func(rt.entropy_par, builder.func),
            );
            rt_refs.insert(
                RT_KD_LOSS,
                module.declare_func_in_func(rt.kd_loss, builder.func),
            );
            rt_refs.insert(
                RT_KD_LOSS_PAR,
                module.declare_func_in_func(rt.kd_loss_par, builder.func),
            );
            rt_refs.insert(
                RT_ROWARGMAX,
                module.declare_func_in_func(rt.rowargmax, builder.func),
            );
            rt_refs.insert(
                RT_ROWARGMAX_PAR,
                module.declare_func_in_func(rt.rowargmax_par, builder.func),
            );
            rt_refs.insert(
                RT_ROWARGMIN,
                module.declare_func_in_func(rt.rowargmin, builder.func),
            );
            rt_refs.insert(
                RT_ROWARGMIN_PAR,
                module.declare_func_in_func(rt.rowargmin_par, builder.func),
            );
            rt_refs.insert(
                RT_COLARGMAX,
                module.declare_func_in_func(rt.colargmax, builder.func),
            );
            rt_refs.insert(
                RT_COLARGMAX_PAR,
                module.declare_func_in_func(rt.colargmax_par, builder.func),
            );
            rt_refs.insert(
                RT_COLARGMIN,
                module.declare_func_in_func(rt.colargmin, builder.func),
            );
            rt_refs.insert(
                RT_COLARGMIN_PAR,
                module.declare_func_in_func(rt.colargmin_par, builder.func),
            );
            rt_refs.insert(
                RT_CUMSUM,
                module.declare_func_in_func(rt.cumsum, builder.func),
            );
            rt_refs.insert(
                RT_CUMSUM_PAR,
                module.declare_func_in_func(rt.cumsum_par, builder.func),
            );
            rt_refs.insert(RT_CUMPROD, module.declare_func_in_func(rt.cumprod, builder.func));
            rt_refs.insert(
                RT_CUMPROD_PAR,
                module.declare_func_in_func(rt.cumprod_par, builder.func),
            );
            rt_refs.insert(
                RT_CUMMAX,
                module.declare_func_in_func(rt.cummax, builder.func),
            );
            rt_refs.insert(
                RT_CUMMAX_PAR,
                module.declare_func_in_func(rt.cummax_par, builder.func),
            );
            rt_refs.insert(
                RT_CUMMIN,
                module.declare_func_in_func(rt.cummin, builder.func),
            );
            rt_refs.insert(
                RT_CUMMIN_PAR,
                module.declare_func_in_func(rt.cummin_par, builder.func),
            );
            rt_refs.insert(
                RT_MAXPOOL2D,
                module.declare_func_in_func(rt.maxpool2d, builder.func),
            );
            rt_refs.insert(
                RT_MAXPOOL2D_PAR,
                module.declare_func_in_func(rt.maxpool2d_par, builder.func),
            );
            rt_refs.insert(
                RT_AVGPOOL2D,
                module.declare_func_in_func(rt.avgpool2d, builder.func),
            );
            rt_refs.insert(
                RT_AVGPOOL2D_PAR,
                module.declare_func_in_func(rt.avgpool2d_par, builder.func),
            );
            rt_refs.insert(
                RT_VMATH_BF16,
                module.declare_func_in_func(rt.vmath_bf16, builder.func),
            );
            rt_refs.insert(
                RT_VMATH_F16,
                module.declare_func_in_func(rt.vmath_f16, builder.func),
            );
            rt_refs.insert(
                RT_TRANSPOSE,
                module.declare_func_in_func(rt.transpose, builder.func),
            );
            rt_refs.insert(
                RT_TRANSPOSE_PAR,
                module.declare_func_in_func(rt.transpose_par, builder.func),
            );
            rt_refs.insert(
                RT_TRANSPOSE_U16,
                module.declare_func_in_func(rt.transpose_u16, builder.func),
            );
            rt_refs.insert(
                RT_TRANSPOSE_U16_PAR,
                module.declare_func_in_func(rt.transpose_u16_par, builder.func),
            );
            rt_refs.insert(
                RT_COLSUM,
                module.declare_func_in_func(rt.colsum, builder.func),
            );
            rt_refs.insert(
                RT_COLSUM_PAR,
                module.declare_func_in_func(rt.colsum_par, builder.func),
            );
            rt_refs.insert(
                RT_COLMAX,
                module.declare_func_in_func(rt.colmax, builder.func),
            );
            rt_refs.insert(
                RT_COLMAX_PAR,
                module.declare_func_in_func(rt.colmax_par, builder.func),
            );
            rt_refs.insert(
                RT_COLMIN,
                module.declare_func_in_func(rt.colmin, builder.func),
            );
            rt_refs.insert(
                RT_COLMIN_PAR,
                module.declare_func_in_func(rt.colmin_par, builder.func),
            );
            rt_refs.insert(
                RT_COLMAXABS,
                module.declare_func_in_func(rt.colmaxabs, builder.func),
            );
            rt_refs.insert(
                RT_COLMAXABS_PAR,
                module.declare_func_in_func(rt.colmaxabs_par, builder.func),
            );
            rt_refs.insert(
                RT_COLMEAN,
                module.declare_func_in_func(rt.colmean, builder.func),
            );
            rt_refs.insert(
                RT_COLMEAN_PAR,
                module.declare_func_in_func(rt.colmean_par, builder.func),
            );
            rt_refs.insert(
                RT_COLSUMSQ,
                module.declare_func_in_func(rt.colsumsq, builder.func),
            );
            rt_refs.insert(
                RT_COLSUMSQ_PAR,
                module.declare_func_in_func(rt.colsumsq_par, builder.func),
            );
            rt_refs.insert(
                RT_COLL2,
                module.declare_func_in_func(rt.coll2, builder.func),
            );
            rt_refs.insert(
                RT_COLL2_PAR,
                module.declare_func_in_func(rt.coll2_par, builder.func),
            );
            rt_refs.insert(
                RT_COLRMS,
                module.declare_func_in_func(rt.colrms, builder.func),
            );
            rt_refs.insert(
                RT_COLRMS_PAR,
                module.declare_func_in_func(rt.colrms_par, builder.func),
            );
            rt_refs.insert(
                RT_VELEM,
                module.declare_func_in_func(rt.velem, builder.func),
            );
            rt_refs.insert(
                RT_VHORNER,
                module.declare_func_in_func(rt.vhorner, builder.func),
            );
            rt_refs.insert(
                RT_SREDUCE,
                module.declare_func_in_func(rt.sred, builder.func),
            );
            rt_refs.insert(
                RT_SREDUCE_PARALLEL,
                module.declare_func_in_func(rt.sred_par, builder.func),
            );
            rt_refs.insert(
                RT_ARGREDUCE,
                module.declare_func_in_func(rt.argreduce, builder.func),
            );
            rt_refs.insert(
                RT_ARGREDUCE_PARALLEL,
                module.declare_func_in_func(rt.argreduce_par, builder.func),
            );
            rt_refs.insert(RT_NORM, module.declare_func_in_func(rt.norm, builder.func));
            rt_refs.insert(
                RT_NORM_PARALLEL,
                module.declare_func_in_func(rt.norm_par, builder.func),
            );
            rt_refs.insert(
                RT_NORM_AFFINE,
                module.declare_func_in_func(rt.norm_affine, builder.func),
            );
            rt_refs.insert(
                RT_NORM_AFFINE_PARALLEL,
                module.declare_func_in_func(rt.norm_affine_par, builder.func),
            );
            rt_refs.insert(
                RT_I8GEMM_NT,
                module.declare_func_in_func(rt.i8nt, builder.func),
            );
            rt_refs.insert(
                RT_I8GEMM_NT_PARALLEL,
                module.declare_func_in_func(rt.i8nt_par, builder.func),
            );
            rt_refs.insert(
                RT_I8GEMM_NT_DEQ,
                module.declare_func_in_func(rt.i8nt_deq, builder.func),
            );
            rt_refs.insert(
                RT_I8GEMM_NT_DEQ_PARALLEL,
                module.declare_func_in_func(rt.i8nt_deq_par, builder.func),
            );
            rt_refs.insert(
                RT_EMBEDDING,
                module.declare_func_in_func(rt.embedding, builder.func),
            );
            rt_refs.insert(
                RT_EMBEDDING_PAR,
                module.declare_func_in_func(rt.embedding_par, builder.func),
            );
            rt_refs.insert(
                RT_SCATTER_ADD,
                module.declare_func_in_func(rt.scatter_add, builder.func),
            );
            rt_refs.insert(
                RT_SCATTER_ADD_PAR,
                module.declare_func_in_func(rt.scatter_add_par, builder.func),
            );
            rt_refs.insert(
                RT_AXPBY_BF16,
                module.declare_func_in_func(rt.axpby_bf16, builder.func),
            );
            rt_refs.insert(
                RT_DOT_BF16,
                module.declare_func_in_func(rt.dot_bf16, builder.func),
            );
            rt_refs.insert(
                RT_SUM_BF16,
                module.declare_func_in_func(rt.sum_bf16, builder.func),
            );
            rt_refs.insert(
                RT_REDUCE_BF16,
                module.declare_func_in_func(rt.reduce_bf16, builder.func),
            );
            rt_refs.insert(
                RT_AXPBY_F16,
                module.declare_func_in_func(rt.axpby_f16, builder.func),
            );
            rt_refs.insert(
                RT_DOT_F16,
                module.declare_func_in_func(rt.dot_f16, builder.func),
            );
            rt_refs.insert(
                RT_SUM_F16,
                module.declare_func_in_func(rt.sum_f16, builder.func),
            );
            rt_refs.insert(
                RT_REDUCE_F16,
                module.declare_func_in_func(rt.reduce_f16, builder.func),
            );
            rt_refs.insert(
                RT_F32_TO_F16,
                module.declare_func_in_func(rt.f32_to_f16, builder.func),
            );
            rt_refs.insert(
                RT_F16_TO_F32,
                module.declare_func_in_func(rt.f16_to_f32, builder.func),
            );
            rt_refs.insert(
                RT_FMOD_F64,
                module.declare_func_in_func(rt.fmod_f64, builder.func),
            );
            rt_refs.insert(
                RT_FMOD_F32,
                module.declare_func_in_func(rt.fmod_f32, builder.func),
            );

            let blocks: Vec<Block> = f.blocks.iter().map(|_| builder.create_block()).collect();
            let mut t = FnTranslator {
                builder,
                func: f,
                interner,
                ptr_ty,
                vmap: vec![None; f.value_types.len()],
                blocks,
                func_refs: &func_refs,
                rt_refs: &rt_refs,
            };
            t.translate();
            t.builder.finalize();
        }
        let fid = ids[&f.name];
        module
            .define_function(fid, &mut ctx)
            .map_err(|e| format!("cranelift define `{}`: {e:?}", interner.resolve(f.name)))?;
        module.clear_context(&mut ctx);
    }
    Ok(ids)
}

/// A JIT-compiled program: holds the executable memory and a pointer to the entry point. Compile
/// once with [`jit_compile`], then [`JitProgram::run`] (capturing output, for correctness) or
/// [`JitProgram::call`] (raw, for timing) as many times as needed.
pub struct JitProgram {
    module: Option<cranelift_jit::JITModule>,
    code: *const u8,
    ret: MirType,
}

impl JitProgram {
    /// Invoke the entry point, transmuting to the right ABI for its return type. Caller must hold
    /// the run lock (so the shared capture buffer isn't raced).
    unsafe fn invoke(&self) -> i64 {
        match &self.ret {
            MirType::Void => {
                let f: extern "C" fn() = std::mem::transmute(self.code);
                f();
                0
            }
            t if t.is_float() => {
                let f: extern "C" fn() -> f64 = std::mem::transmute(self.code);
                f() as i64
            }
            MirType::I64 => {
                let f: extern "C" fn() -> i64 = std::mem::transmute(self.code);
                f()
            }
            _ => {
                let f: extern "C" fn() -> i32 = std::mem::transmute(self.code);
                f() as i64
            }
        }
    }

    /// Run once, returning the exit code and captured stdout (the native counterpart to the
    /// interpreter's `run_with_output`).
    pub fn run(&self) -> Result<(i64, Vec<u8>), String> {
        let _guard = RUN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).clear();
        ASSERT_FAILED.store(false, Ordering::SeqCst);
        let exit_code = unsafe { self.invoke() };
        let out = OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if ASSERT_FAILED.load(Ordering::SeqCst) {
            return Err("assertion failed".into());
        }
        Ok((exit_code, out))
    }

    /// Run once for timing: clears (but does not clone) the capture buffer so repeated prints don't
    /// grow memory, and returns just the exit code.
    pub fn call(&self) -> i64 {
        let _guard = RUN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).clear();
        unsafe { self.invoke() }
    }
}

impl Drop for JitProgram {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            // SAFETY: no JIT'd code from this module runs after the handle is dropped.
            unsafe { m.free_memory() };
        }
    }
}

/// JIT-compile `program`, returning a callable handle to `entry`.
pub fn jit_compile(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> Result<JitProgram, String> {
    use cranelift_jit::{JITBuilder, JITModule};

    let isa = make_isa(false)?;
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    builder.symbol(RT_PRINT_I64, rt_print_i64 as *const u8);
    builder.symbol(RT_PRINT_F64, rt_print_f64 as *const u8);
    builder.symbol(RT_ASSERT, rt_assert as *const u8);
    builder.symbol(
        RT_PARALLEL_FOR,
        mercury_runtime::mercury_parallel_for as *const u8,
    );
    builder.symbol(RT_SGEMM, mercury_runtime::mercury_sgemm as *const u8);
    builder.symbol(
        RT_SGEMM_PARALLEL,
        mercury_runtime::mercury_sgemm_parallel as *const u8,
    );
    builder.symbol(RT_SGEMM_NT, mercury_runtime::mercury_sgemm_nt as *const u8);
    builder.symbol(
        RT_SGEMM_NT_PARALLEL,
        mercury_runtime::mercury_sgemm_nt_parallel as *const u8,
    );
    builder.symbol(RT_SGEMM_TN, mercury_runtime::mercury_sgemm_tn as *const u8);
    builder.symbol(
        RT_SGEMM_TN_PARALLEL,
        mercury_runtime::mercury_sgemm_tn_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT,
        mercury_runtime::mercury_sgemm_bf16_nt as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT_PARALLEL,
        mercury_runtime::mercury_sgemm_bf16_nt_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT,
        mercury_runtime::mercury_sgemm_f16_nt as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT_PARALLEL,
        mercury_runtime::mercury_sgemm_f16_nt_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_TN,
        mercury_runtime::mercury_sgemm_bf16_tn as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_TN_PARALLEL,
        mercury_runtime::mercury_sgemm_bf16_tn_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_TN,
        mercury_runtime::mercury_sgemm_f16_tn as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_TN_PARALLEL,
        mercury_runtime::mercury_sgemm_f16_tn_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_EPI,
        mercury_runtime::mercury_sgemm_nt_epi as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_EPI_PAR,
        mercury_runtime::mercury_sgemm_nt_epi_parallel as *const u8,
    );
    builder.symbol(RT_SGEMV, mercury_runtime::mercury_sgemv as *const u8);
    builder.symbol(
        RT_SGEMV_PAR,
        mercury_runtime::mercury_sgemv_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_ALPHA,
        mercury_runtime::mercury_sgemm_nt_alpha as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_ALPHA_PAR,
        mercury_runtime::mercury_sgemm_nt_alpha_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT_EPI,
        mercury_runtime::mercury_sgemm_bf16_nt_epi as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT_EPI_PAR,
        mercury_runtime::mercury_sgemm_bf16_nt_epi_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT_EPI,
        mercury_runtime::mercury_sgemm_f16_nt_epi as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT_EPI_PAR,
        mercury_runtime::mercury_sgemm_f16_nt_epi_parallel as *const u8,
    );
    builder.symbol(RT_VMATH, mercury_runtime::mercury_vmath_f32 as *const u8);
    builder.symbol(RT_VMATH2, mercury_runtime::mercury_vmath2_f32 as *const u8);
    builder.symbol(
        RT_SOFTMAX_BWD,
        mercury_runtime::mercury_softmax_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_SOFTMAX_BWD_PAR,
        mercury_runtime::mercury_softmax_bwd_f32_parallel as *const u8,
    );
    builder.symbol(RT_LRSCAN, mercury_runtime::mercury_lrscan_f32 as *const u8);
    builder.symbol(
        RT_LRSCAN_PAR,
        mercury_runtime::mercury_lrscan_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_RMSNORM_BWD,
        mercury_runtime::mercury_rmsnorm_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_RMSNORM_BWD_PAR,
        mercury_runtime::mercury_rmsnorm_bwd_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_LAYERNORM_BWD,
        mercury_runtime::mercury_layernorm_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_LAYERNORM_BWD_PAR,
        mercury_runtime::mercury_layernorm_bwd_f32_parallel as *const u8,
    );
    builder.symbol(RT_XENT, mercury_runtime::mercury_xent_fwd_f32 as *const u8);
    builder.symbol(
        RT_XENT_PAR,
        mercury_runtime::mercury_xent_fwd_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_XENT_BWD,
        mercury_runtime::mercury_xent_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_XENT_BWD_PAR,
        mercury_runtime::mercury_xent_bwd_f32_parallel as *const u8,
    );
    builder.symbol(RT_ROPE, mercury_runtime::mercury_rope_f32 as *const u8);
    builder.symbol(
        RT_ROPE_PAR,
        mercury_runtime::mercury_rope_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ROPE_BWD,
        mercury_runtime::mercury_rope_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_ROPE_BWD_PAR,
        mercury_runtime::mercury_rope_bwd_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_LOGSUMEXP,
        mercury_runtime::mercury_logsumexp_f32 as *const u8,
    );
    builder.symbol(
        RT_LOGSUMEXP_PAR,
        mercury_runtime::mercury_logsumexp_f32_parallel as *const u8,
    );
    builder.symbol(RT_KLDIV, mercury_runtime::mercury_kldiv_f32 as *const u8);
    builder.symbol(
        RT_KLDIV_PAR,
        mercury_runtime::mercury_kldiv_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ENTROPY,
        mercury_runtime::mercury_entropy_f32 as *const u8,
    );
    builder.symbol(
        RT_ENTROPY_PAR,
        mercury_runtime::mercury_entropy_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_KD_LOSS,
        mercury_runtime::mercury_kd_loss_f32 as *const u8,
    );
    builder.symbol(
        RT_KD_LOSS_PAR,
        mercury_runtime::mercury_kd_loss_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ROWARGMAX,
        mercury_runtime::mercury_rowargmax_i32 as *const u8,
    );
    builder.symbol(
        RT_ROWARGMAX_PAR,
        mercury_runtime::mercury_rowargmax_i32_parallel as *const u8,
    );
    builder.symbol(
        RT_ROWARGMIN,
        mercury_runtime::mercury_rowargmin_i32 as *const u8,
    );
    builder.symbol(
        RT_ROWARGMIN_PAR,
        mercury_runtime::mercury_rowargmin_i32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLARGMAX,
        mercury_runtime::mercury_colargmax_i32 as *const u8,
    );
    builder.symbol(
        RT_COLARGMAX_PAR,
        mercury_runtime::mercury_colargmax_i32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLARGMIN,
        mercury_runtime::mercury_colargmin_i32 as *const u8,
    );
    builder.symbol(
        RT_COLARGMIN_PAR,
        mercury_runtime::mercury_colargmin_i32_parallel as *const u8,
    );
    builder.symbol(RT_CUMSUM, mercury_runtime::mercury_cumsum_f32 as *const u8);
    builder.symbol(
        RT_CUMSUM_PAR,
        mercury_runtime::mercury_cumsum_f32_parallel as *const u8,
    );
    builder.symbol(RT_CUMPROD, mercury_runtime::mercury_cumprod_f32 as *const u8);
    builder.symbol(
        RT_CUMPROD_PAR,
        mercury_runtime::mercury_cumprod_f32_parallel as *const u8,
    );
    builder.symbol(RT_CUMMAX, mercury_runtime::mercury_cummax_f32 as *const u8);
    builder.symbol(
        RT_CUMMAX_PAR,
        mercury_runtime::mercury_cummax_f32_parallel as *const u8,
    );
    builder.symbol(RT_CUMMIN, mercury_runtime::mercury_cummin_f32 as *const u8);
    builder.symbol(
        RT_CUMMIN_PAR,
        mercury_runtime::mercury_cummin_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_MAXPOOL2D,
        mercury_runtime::mercury_maxpool2d_f32 as *const u8,
    );
    builder.symbol(
        RT_MAXPOOL2D_PAR,
        mercury_runtime::mercury_maxpool2d_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_AVGPOOL2D,
        mercury_runtime::mercury_avgpool2d_f32 as *const u8,
    );
    builder.symbol(
        RT_AVGPOOL2D_PAR,
        mercury_runtime::mercury_avgpool2d_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_VMATH_BF16,
        mercury_runtime::mercury_vmath_bf16 as *const u8,
    );
    builder.symbol(
        RT_VMATH_F16,
        mercury_runtime::mercury_vmath_f16 as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE,
        mercury_runtime::mercury_transpose_f32 as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE_PAR,
        mercury_runtime::mercury_transpose_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE_U16,
        mercury_runtime::mercury_transpose_u16 as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE_U16_PAR,
        mercury_runtime::mercury_transpose_u16_parallel as *const u8,
    );
    builder.symbol(RT_COLSUM, mercury_runtime::mercury_colsum_f32 as *const u8);
    builder.symbol(
        RT_COLSUM_PAR,
        mercury_runtime::mercury_colsum_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLMAX, mercury_runtime::mercury_colmax_f32 as *const u8);
    builder.symbol(
        RT_COLMAX_PAR,
        mercury_runtime::mercury_colmax_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLMIN, mercury_runtime::mercury_colmin_f32 as *const u8);
    builder.symbol(
        RT_COLMIN_PAR,
        mercury_runtime::mercury_colmin_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLMAXABS,
        mercury_runtime::mercury_colmaxabs_f32 as *const u8,
    );
    builder.symbol(
        RT_COLMAXABS_PAR,
        mercury_runtime::mercury_colmaxabs_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLMEAN, mercury_runtime::mercury_colmean_f32 as *const u8);
    builder.symbol(
        RT_COLMEAN_PAR,
        mercury_runtime::mercury_colmean_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLSUMSQ,
        mercury_runtime::mercury_colsumsq_f32 as *const u8,
    );
    builder.symbol(
        RT_COLSUMSQ_PAR,
        mercury_runtime::mercury_colsumsq_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLL2, mercury_runtime::mercury_coll2_f32 as *const u8);
    builder.symbol(
        RT_COLL2_PAR,
        mercury_runtime::mercury_coll2_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLRMS, mercury_runtime::mercury_colrms_f32 as *const u8);
    builder.symbol(
        RT_COLRMS_PAR,
        mercury_runtime::mercury_colrms_f32_parallel as *const u8,
    );
    builder.symbol(RT_VELEM, mercury_runtime::mercury_velem_f32 as *const u8);
    builder.symbol(
        RT_VHORNER,
        mercury_runtime::mercury_vhorner_f32 as *const u8,
    );
    builder.symbol(
        RT_SREDUCE,
        mercury_runtime::mercury_sreduce_f32 as *const u8,
    );
    builder.symbol(
        RT_SREDUCE_PARALLEL,
        mercury_runtime::mercury_sreduce_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ARGREDUCE,
        mercury_runtime::mercury_argreduce_f32 as *const u8,
    );
    builder.symbol(
        RT_ARGREDUCE_PARALLEL,
        mercury_runtime::mercury_argreduce_f32_parallel as *const u8,
    );
    builder.symbol(RT_NORM, mercury_runtime::mercury_norm_f32 as *const u8);
    builder.symbol(
        RT_NORM_PARALLEL,
        mercury_runtime::mercury_norm_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_NORM_AFFINE,
        mercury_runtime::mercury_norm_affine_f32 as *const u8,
    );
    builder.symbol(
        RT_NORM_AFFINE_PARALLEL,
        mercury_runtime::mercury_norm_affine_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_I8GEMM_NT,
        mercury_runtime::mercury_i8gemm_nt as *const u8,
    );
    builder.symbol(
        RT_I8GEMM_NT_PARALLEL,
        mercury_runtime::mercury_i8gemm_nt_parallel as *const u8,
    );
    builder.symbol(
        RT_I8GEMM_NT_DEQ,
        mercury_runtime::mercury_i8gemm_nt_deq as *const u8,
    );
    builder.symbol(
        RT_I8GEMM_NT_DEQ_PARALLEL,
        mercury_runtime::mercury_i8gemm_nt_deq_parallel as *const u8,
    );
    builder.symbol(
        RT_EMBEDDING,
        mercury_runtime::mercury_embedding_f32 as *const u8,
    );
    builder.symbol(
        RT_EMBEDDING_PAR,
        mercury_runtime::mercury_embedding_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_SCATTER_ADD,
        mercury_runtime::mercury_scatter_add_f32 as *const u8,
    );
    builder.symbol(
        RT_SCATTER_ADD_PAR,
        mercury_runtime::mercury_scatter_add_f32_parallel as *const u8,
    );
    builder.symbol(RT_DOT_BF16, mercury_runtime::mercury_dot_bf16 as *const u8);
    builder.symbol(RT_SUM_BF16, mercury_runtime::mercury_sum_bf16 as *const u8);
    builder.symbol(
        RT_REDUCE_BF16,
        mercury_runtime::mercury_reduce_bf16 as *const u8,
    );
    builder.symbol(
        RT_AXPBY_F16,
        mercury_runtime::mercury_axpby_f16 as *const u8,
    );
    builder.symbol(RT_DOT_F16, mercury_runtime::mercury_dot_f16 as *const u8);
    builder.symbol(RT_SUM_F16, mercury_runtime::mercury_sum_f16 as *const u8);
    builder.symbol(
        RT_REDUCE_F16,
        mercury_runtime::mercury_reduce_f16 as *const u8,
    );
    builder.symbol(
        RT_F32_TO_F16,
        mercury_runtime::mercury_f32_to_f16_bits as *const u8,
    );
    builder.symbol(
        RT_F16_TO_F32,
        mercury_runtime::mercury_f16_bits_to_f32 as *const u8,
    );
    builder.symbol(
        RT_AXPBY_BF16,
        mercury_runtime::mercury_axpby_bf16 as *const u8,
    );
    builder.symbol(RT_FMOD_F64, rt_fmod_f64 as *const u8);
    builder.symbol(RT_FMOD_F32, rt_fmod_f32 as *const u8);
    let mut module = JITModule::new(builder);

    let ids = populate_module(&mut module, program, interner)?;
    module.finalize_definitions().map_err(|e| e.to_string())?;

    let entry_fn = program
        .function(entry)
        .ok_or_else(|| format!("no entry function `{}`", interner.resolve(entry)))?;
    if !entry_fn.params.is_empty() {
        unsafe { module.free_memory() };
        return Err("native entry point must take no parameters".into());
    }
    let entry_id = ids[&entry];
    let code = module.get_finalized_function(entry_id);

    Ok(JitProgram {
        module: Some(module),
        code,
        ret: entry_fn.ret.clone(),
    })
}

/// JIT-compile and execute `entry`, returning its exit code and captured stdout.
pub fn jit_run(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    jit_compile(program, entry, interner)?.run()
}

/// A JIT-compiled module exposing raw pointers to any of its functions, for callers that know a
/// function's ABI and want to invoke it directly (e.g. timing a kernel `(*const f32, …)`).
pub struct JitModuleHandle {
    module: Option<cranelift_jit::JITModule>,
    ids: HashMap<Symbol, FuncId>,
}

impl JitModuleHandle {
    /// The finalized machine-code pointer for `sym`, if the module defines it. The caller must
    /// transmute it to the correct ABI; the pointer is valid until this handle is dropped.
    pub fn func_ptr(&self, sym: Symbol) -> Option<*const u8> {
        let id = *self.ids.get(&sym)?;
        Some(self.module.as_ref().unwrap().get_finalized_function(id))
    }
}

impl Drop for JitModuleHandle {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            unsafe { m.free_memory() };
        }
    }
}

/// JIT-compile every function in `program`, returning a handle for fetching their code pointers.
pub fn jit_module(program: &Program, interner: &Interner) -> Result<JitModuleHandle, String> {
    use cranelift_jit::{JITBuilder, JITModule};

    let isa = make_isa(false)?;
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    builder.symbol(RT_PRINT_I64, rt_print_i64 as *const u8);
    builder.symbol(RT_PRINT_F64, rt_print_f64 as *const u8);
    builder.symbol(RT_ASSERT, rt_assert as *const u8);
    builder.symbol(
        RT_PARALLEL_FOR,
        mercury_runtime::mercury_parallel_for as *const u8,
    );
    builder.symbol(RT_SGEMM, mercury_runtime::mercury_sgemm as *const u8);
    builder.symbol(
        RT_SGEMM_PARALLEL,
        mercury_runtime::mercury_sgemm_parallel as *const u8,
    );
    builder.symbol(RT_SGEMM_NT, mercury_runtime::mercury_sgemm_nt as *const u8);
    builder.symbol(
        RT_SGEMM_NT_PARALLEL,
        mercury_runtime::mercury_sgemm_nt_parallel as *const u8,
    );
    builder.symbol(RT_SGEMM_TN, mercury_runtime::mercury_sgemm_tn as *const u8);
    builder.symbol(
        RT_SGEMM_TN_PARALLEL,
        mercury_runtime::mercury_sgemm_tn_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT,
        mercury_runtime::mercury_sgemm_bf16_nt as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT_PARALLEL,
        mercury_runtime::mercury_sgemm_bf16_nt_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT,
        mercury_runtime::mercury_sgemm_f16_nt as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT_PARALLEL,
        mercury_runtime::mercury_sgemm_f16_nt_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_TN,
        mercury_runtime::mercury_sgemm_bf16_tn as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_TN_PARALLEL,
        mercury_runtime::mercury_sgemm_bf16_tn_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_TN,
        mercury_runtime::mercury_sgemm_f16_tn as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_TN_PARALLEL,
        mercury_runtime::mercury_sgemm_f16_tn_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_EPI,
        mercury_runtime::mercury_sgemm_nt_epi as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_EPI_PAR,
        mercury_runtime::mercury_sgemm_nt_epi_parallel as *const u8,
    );
    builder.symbol(RT_SGEMV, mercury_runtime::mercury_sgemv as *const u8);
    builder.symbol(
        RT_SGEMV_PAR,
        mercury_runtime::mercury_sgemv_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_ALPHA,
        mercury_runtime::mercury_sgemm_nt_alpha as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_ALPHA_PAR,
        mercury_runtime::mercury_sgemm_nt_alpha_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT_EPI,
        mercury_runtime::mercury_sgemm_bf16_nt_epi as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT_EPI_PAR,
        mercury_runtime::mercury_sgemm_bf16_nt_epi_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT_EPI,
        mercury_runtime::mercury_sgemm_f16_nt_epi as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT_EPI_PAR,
        mercury_runtime::mercury_sgemm_f16_nt_epi_parallel as *const u8,
    );
    builder.symbol(RT_VMATH, mercury_runtime::mercury_vmath_f32 as *const u8);
    builder.symbol(RT_VMATH2, mercury_runtime::mercury_vmath2_f32 as *const u8);
    builder.symbol(
        RT_SOFTMAX_BWD,
        mercury_runtime::mercury_softmax_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_SOFTMAX_BWD_PAR,
        mercury_runtime::mercury_softmax_bwd_f32_parallel as *const u8,
    );
    builder.symbol(RT_LRSCAN, mercury_runtime::mercury_lrscan_f32 as *const u8);
    builder.symbol(
        RT_LRSCAN_PAR,
        mercury_runtime::mercury_lrscan_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_RMSNORM_BWD,
        mercury_runtime::mercury_rmsnorm_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_RMSNORM_BWD_PAR,
        mercury_runtime::mercury_rmsnorm_bwd_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_LAYERNORM_BWD,
        mercury_runtime::mercury_layernorm_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_LAYERNORM_BWD_PAR,
        mercury_runtime::mercury_layernorm_bwd_f32_parallel as *const u8,
    );
    builder.symbol(RT_XENT, mercury_runtime::mercury_xent_fwd_f32 as *const u8);
    builder.symbol(
        RT_XENT_PAR,
        mercury_runtime::mercury_xent_fwd_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_XENT_BWD,
        mercury_runtime::mercury_xent_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_XENT_BWD_PAR,
        mercury_runtime::mercury_xent_bwd_f32_parallel as *const u8,
    );
    builder.symbol(RT_ROPE, mercury_runtime::mercury_rope_f32 as *const u8);
    builder.symbol(
        RT_ROPE_PAR,
        mercury_runtime::mercury_rope_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ROPE_BWD,
        mercury_runtime::mercury_rope_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_ROPE_BWD_PAR,
        mercury_runtime::mercury_rope_bwd_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_LOGSUMEXP,
        mercury_runtime::mercury_logsumexp_f32 as *const u8,
    );
    builder.symbol(
        RT_LOGSUMEXP_PAR,
        mercury_runtime::mercury_logsumexp_f32_parallel as *const u8,
    );
    builder.symbol(RT_KLDIV, mercury_runtime::mercury_kldiv_f32 as *const u8);
    builder.symbol(
        RT_KLDIV_PAR,
        mercury_runtime::mercury_kldiv_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ENTROPY,
        mercury_runtime::mercury_entropy_f32 as *const u8,
    );
    builder.symbol(
        RT_ENTROPY_PAR,
        mercury_runtime::mercury_entropy_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_KD_LOSS,
        mercury_runtime::mercury_kd_loss_f32 as *const u8,
    );
    builder.symbol(
        RT_KD_LOSS_PAR,
        mercury_runtime::mercury_kd_loss_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ROWARGMAX,
        mercury_runtime::mercury_rowargmax_i32 as *const u8,
    );
    builder.symbol(
        RT_ROWARGMAX_PAR,
        mercury_runtime::mercury_rowargmax_i32_parallel as *const u8,
    );
    builder.symbol(
        RT_ROWARGMIN,
        mercury_runtime::mercury_rowargmin_i32 as *const u8,
    );
    builder.symbol(
        RT_ROWARGMIN_PAR,
        mercury_runtime::mercury_rowargmin_i32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLARGMAX,
        mercury_runtime::mercury_colargmax_i32 as *const u8,
    );
    builder.symbol(
        RT_COLARGMAX_PAR,
        mercury_runtime::mercury_colargmax_i32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLARGMIN,
        mercury_runtime::mercury_colargmin_i32 as *const u8,
    );
    builder.symbol(
        RT_COLARGMIN_PAR,
        mercury_runtime::mercury_colargmin_i32_parallel as *const u8,
    );
    builder.symbol(RT_CUMSUM, mercury_runtime::mercury_cumsum_f32 as *const u8);
    builder.symbol(
        RT_CUMSUM_PAR,
        mercury_runtime::mercury_cumsum_f32_parallel as *const u8,
    );
    builder.symbol(RT_CUMPROD, mercury_runtime::mercury_cumprod_f32 as *const u8);
    builder.symbol(
        RT_CUMPROD_PAR,
        mercury_runtime::mercury_cumprod_f32_parallel as *const u8,
    );
    builder.symbol(RT_CUMMAX, mercury_runtime::mercury_cummax_f32 as *const u8);
    builder.symbol(
        RT_CUMMAX_PAR,
        mercury_runtime::mercury_cummax_f32_parallel as *const u8,
    );
    builder.symbol(RT_CUMMIN, mercury_runtime::mercury_cummin_f32 as *const u8);
    builder.symbol(
        RT_CUMMIN_PAR,
        mercury_runtime::mercury_cummin_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_MAXPOOL2D,
        mercury_runtime::mercury_maxpool2d_f32 as *const u8,
    );
    builder.symbol(
        RT_MAXPOOL2D_PAR,
        mercury_runtime::mercury_maxpool2d_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_AVGPOOL2D,
        mercury_runtime::mercury_avgpool2d_f32 as *const u8,
    );
    builder.symbol(
        RT_AVGPOOL2D_PAR,
        mercury_runtime::mercury_avgpool2d_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_VMATH_BF16,
        mercury_runtime::mercury_vmath_bf16 as *const u8,
    );
    builder.symbol(
        RT_VMATH_F16,
        mercury_runtime::mercury_vmath_f16 as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE,
        mercury_runtime::mercury_transpose_f32 as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE_PAR,
        mercury_runtime::mercury_transpose_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE_U16,
        mercury_runtime::mercury_transpose_u16 as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE_U16_PAR,
        mercury_runtime::mercury_transpose_u16_parallel as *const u8,
    );
    builder.symbol(RT_COLSUM, mercury_runtime::mercury_colsum_f32 as *const u8);
    builder.symbol(
        RT_COLSUM_PAR,
        mercury_runtime::mercury_colsum_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLMAX, mercury_runtime::mercury_colmax_f32 as *const u8);
    builder.symbol(
        RT_COLMAX_PAR,
        mercury_runtime::mercury_colmax_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLMIN, mercury_runtime::mercury_colmin_f32 as *const u8);
    builder.symbol(
        RT_COLMIN_PAR,
        mercury_runtime::mercury_colmin_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLMAXABS,
        mercury_runtime::mercury_colmaxabs_f32 as *const u8,
    );
    builder.symbol(
        RT_COLMAXABS_PAR,
        mercury_runtime::mercury_colmaxabs_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLMEAN, mercury_runtime::mercury_colmean_f32 as *const u8);
    builder.symbol(
        RT_COLMEAN_PAR,
        mercury_runtime::mercury_colmean_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLSUMSQ,
        mercury_runtime::mercury_colsumsq_f32 as *const u8,
    );
    builder.symbol(
        RT_COLSUMSQ_PAR,
        mercury_runtime::mercury_colsumsq_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLL2, mercury_runtime::mercury_coll2_f32 as *const u8);
    builder.symbol(
        RT_COLL2_PAR,
        mercury_runtime::mercury_coll2_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLRMS, mercury_runtime::mercury_colrms_f32 as *const u8);
    builder.symbol(
        RT_COLRMS_PAR,
        mercury_runtime::mercury_colrms_f32_parallel as *const u8,
    );
    builder.symbol(RT_VELEM, mercury_runtime::mercury_velem_f32 as *const u8);
    builder.symbol(
        RT_VHORNER,
        mercury_runtime::mercury_vhorner_f32 as *const u8,
    );
    builder.symbol(
        RT_SREDUCE,
        mercury_runtime::mercury_sreduce_f32 as *const u8,
    );
    builder.symbol(
        RT_SREDUCE_PARALLEL,
        mercury_runtime::mercury_sreduce_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ARGREDUCE,
        mercury_runtime::mercury_argreduce_f32 as *const u8,
    );
    builder.symbol(
        RT_ARGREDUCE_PARALLEL,
        mercury_runtime::mercury_argreduce_f32_parallel as *const u8,
    );
    builder.symbol(RT_NORM, mercury_runtime::mercury_norm_f32 as *const u8);
    builder.symbol(
        RT_NORM_PARALLEL,
        mercury_runtime::mercury_norm_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_NORM_AFFINE,
        mercury_runtime::mercury_norm_affine_f32 as *const u8,
    );
    builder.symbol(
        RT_NORM_AFFINE_PARALLEL,
        mercury_runtime::mercury_norm_affine_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_I8GEMM_NT,
        mercury_runtime::mercury_i8gemm_nt as *const u8,
    );
    builder.symbol(
        RT_I8GEMM_NT_PARALLEL,
        mercury_runtime::mercury_i8gemm_nt_parallel as *const u8,
    );
    builder.symbol(
        RT_I8GEMM_NT_DEQ,
        mercury_runtime::mercury_i8gemm_nt_deq as *const u8,
    );
    builder.symbol(
        RT_I8GEMM_NT_DEQ_PARALLEL,
        mercury_runtime::mercury_i8gemm_nt_deq_parallel as *const u8,
    );
    builder.symbol(
        RT_EMBEDDING,
        mercury_runtime::mercury_embedding_f32 as *const u8,
    );
    builder.symbol(
        RT_EMBEDDING_PAR,
        mercury_runtime::mercury_embedding_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_SCATTER_ADD,
        mercury_runtime::mercury_scatter_add_f32 as *const u8,
    );
    builder.symbol(
        RT_SCATTER_ADD_PAR,
        mercury_runtime::mercury_scatter_add_f32_parallel as *const u8,
    );
    builder.symbol(RT_DOT_BF16, mercury_runtime::mercury_dot_bf16 as *const u8);
    builder.symbol(RT_SUM_BF16, mercury_runtime::mercury_sum_bf16 as *const u8);
    builder.symbol(
        RT_REDUCE_BF16,
        mercury_runtime::mercury_reduce_bf16 as *const u8,
    );
    builder.symbol(
        RT_AXPBY_F16,
        mercury_runtime::mercury_axpby_f16 as *const u8,
    );
    builder.symbol(RT_DOT_F16, mercury_runtime::mercury_dot_f16 as *const u8);
    builder.symbol(RT_SUM_F16, mercury_runtime::mercury_sum_f16 as *const u8);
    builder.symbol(
        RT_REDUCE_F16,
        mercury_runtime::mercury_reduce_f16 as *const u8,
    );
    builder.symbol(
        RT_F32_TO_F16,
        mercury_runtime::mercury_f32_to_f16_bits as *const u8,
    );
    builder.symbol(
        RT_F16_TO_F32,
        mercury_runtime::mercury_f16_bits_to_f32 as *const u8,
    );
    builder.symbol(
        RT_AXPBY_BF16,
        mercury_runtime::mercury_axpby_bf16 as *const u8,
    );
    builder.symbol(RT_FMOD_F64, rt_fmod_f64 as *const u8);
    builder.symbol(RT_FMOD_F32, rt_fmod_f32 as *const u8);
    let mut module = JITModule::new(builder);
    let ids = populate_module(&mut module, program, interner)?;
    module.finalize_definitions().map_err(|e| e.to_string())?;
    Ok(JitModuleHandle {
        module: Some(module),
        ids,
    })
}

/// Compile `program` to a native object file (bytes) for the host target. The runtime symbols are
/// left as undefined imports for the linker to resolve against a small C runtime.
pub fn emit_object(program: &Program, interner: &Interner) -> Result<Vec<u8>, String> {
    use cranelift_object::{ObjectBuilder, ObjectModule};

    let isa = make_isa(true)?;
    let builder = ObjectBuilder::new(isa, "mercury", cranelift_module::default_libcall_names())
        .map_err(|e| e.to_string())?;
    let mut module = ObjectModule::new(builder);
    populate_module(&mut module, program, interner)?;
    let product = module.finish();
    product.emit().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod fuzz;
