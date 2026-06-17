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
const RT_SGEMM_NT_EPI: &str = "mercury_sgemm_nt_epi";
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
            RT_SGEMM | RT_SGEMM_PARALLEL | RT_SGEMM_NT | RT_SGEMM_NT_PARALLEL
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
        // The fused-epilogue Linear: mercury_sgemm_nt_epi(a, b, c, m, k, n, beta, bias, act) —
        // three pointers, four i64 (m,k,n,beta), a bias pointer, and an i64 activation code.
        if name == RT_SGEMM_NT_EPI && args.len() == 9 {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let c = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let k = self.coerce_to_i64(args[4]);
            let n = self.coerce_to_i64(args[5]);
            let beta = self.coerce_to_i64(args[6]);
            let bias = self.val(args[7]);
            let act = self.coerce_to_i64(args[8]);
            let fref = self.rt_refs[RT_SGEMM_NT_EPI];
            self.builder
                .ins()
                .call(fref, &[a, b, c, m, k, n, beta, bias, act]);
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
    sgemm_nt_epi: FuncId,
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
        sgemm_nt_epi: module
            .declare_function(RT_SGEMM_NT_EPI, Linkage::Import, &sig_gemm_epi)
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
                RT_SGEMM_NT_EPI,
                module.declare_func_in_func(rt.sgemm_nt_epi, builder.func),
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
    builder.symbol(
        RT_SGEMM_NT_EPI,
        mercury_runtime::mercury_sgemm_nt_epi as *const u8,
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
    builder.symbol(
        RT_SGEMM_NT_EPI,
        mercury_runtime::mercury_sgemm_nt_epi as *const u8,
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
