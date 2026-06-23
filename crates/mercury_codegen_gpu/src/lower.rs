//! General **MIR -> PTX lowering** (Phase 4) — the lever that turns GPU coverage from "a fixed
//! recognizer kernel menu" into "the whole language."
//!
//! The existing `--backend=gpu` path is an *offloading interpreter*: it tree-walks the program on the
//! CPU and only the recognized GEMM/vmath/norm/reduce calls ever reach the device. This module instead
//! consumes **MIR directly** and emits a **PTX kernel** for the entry function (and a `.func` per other
//! function), then driver-JIT-loads and launches it via the shared [`crate::gpu::Gpu`] harness — so an
//! *arbitrary* Mercury program runs on the GPU. It mirrors the Cranelift backend
//! (`mercury_codegen_cranelift`): same `Backend` seam, same `jit_run(program, entry, interner) ->
//! (exit_code, stdout)` shape, PTX virtual registers instead of CLIF SSA values.
//!
//! Mapping (the seams the prompt names):
//! - **SSA values -> PTX virtual registers.** Integers/pointers live in 64-bit `%rd` registers kept
//!   *sign-extended* and masked to the MIR width after each op (mirroring the interpreter's `i128`
//!   value + `mask`); `f32` in `%f`, `f64` in `%fd`; predicates are ephemeral `%p`.
//! - **Block params -> a register-passing convention.** Each block param gets a fixed register; every
//!   branch edge copies its args into the target's param registers (through temporaries, so permuted
//!   / self-referential edges are safe) and then `bra`s — exactly the role Cranelift block params play.
//! - **`CondBr` -> predicated `bra`.** `Br`/`CondBr`/`Ret`/`Unreachable` map to `bra`/`@p bra`/`ret`.
//! - **`Alloca`/`Load`/`Store`/`Gep` -> local/global memory.** Allocas share one `.local` frame
//!   (generic-addressed via `cvta.local`); loads/stores use generic `ld`/`st`, so the same code works
//!   for stack locals and device-global tensor pointers.
//! - **`print`/`println`/`assert` -> a device record buffer.** The kernel can't format on the host, so
//!   it appends `(tag, payload)` records to a context buffer (atomic counter); the host replays them
//!   with the *identical* `format!("{}\n")` the interpreter/Cranelift runtime uses — byte-identical
//!   integer output, tolerance-comparable float output.
//!
//! Recognized ops (matmul / vmath / norm / reduce / `@parallel`) are *not yet* lowered here; they
//! return a classifiable `UNSUPPORTED:` error so the coverage gate reports them honestly as
//! not-yet-covered (the fast-path dispatch + grid-stride `@parallel` lowering land in later
//! increments). The headline for this phase is **coverage measured against the interpreter oracle**,
//! never an asserted speed.
//!
//! Correctness gate: a sweep of `tests/run/*.mer` run through this backend matches the interpreter
//! oracle (integer/control-flow programs bit-exact; float programs within the CPU<->GPU tolerance) and
//! `-O0` == `-O3` (see the `#[cfg(test)]` module). The existing offload path and the toolchain-free
//! core stay byte-for-byte unchanged — this path is purely additive.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};

use cudarc::driver::{LaunchConfig, PushKernelArg};

use mercury_backend::{Artifact, Backend};
use mercury_mir::{BinOp, CastKind, CmpOp, Function, MirType, Op, Program, RoundMode, Terminator, ValueId};
use mercury_span::{Interner, Symbol};

// --- Device context-buffer layout (u64 slots) -----------------------------------------------
// [0] = record count (atomic), [1] = assert-failed flag, [2] = exit code (i64), [3] = reserved;
// records start at slot 4, two u64 each: [tag, payload]. tag 0 = int (payload = i64 bits), 1 =
// float (payload = f64 bits). Byte offsets are what the PTX uses.
const CTX_COUNT_OFF: u64 = 0;
const CTX_ASSERT_OFF: u64 = 8;
const CTX_EXIT_OFF: u64 = 16;
const CTX_RECORDS_OFF: u64 = 32;
const RECORD_BYTES: u64 = 16;
/// Capacity of the print-record buffer. Generous for the test corpus; overflow is *detected* (the
/// atomic counter still advances) and surfaced as an error rather than silently corrupting memory.
const RECORD_CAP: u64 = 1 << 16;

/// The PTX name of the launched entry kernel (a thin `.visible .entry` wrapper).
const KERNEL_NAME: &str = "mercury_kernel";

/// Prefix marking a *lowering decline* (an op/construct not yet handled) as opposed to a genuine
/// JIT/launch failure — the coverage gate treats the former as "not yet covered" (skip) and the
/// latter as a hard failure.
pub const UNSUPPORTED: &str = "UNSUPPORTED:";

// ============================================================================================
// Public API (mirrors `mercury_codegen_cranelift`)
// ============================================================================================

/// The general MIR->PTX GPU backend: lowers an arbitrary program to PTX, JITs it on the device, and
/// runs `entry`, capturing stdout + the exit code into [`Artifact::Executed`] — the same shape the
/// interpreter and Cranelift produce, so the driver and differential gate treat all three alike.
pub struct GpuLowerBackend;

impl Backend for GpuLowerBackend {
    fn name(&self) -> &'static str {
        "gpu-lower"
    }

    fn compile(
        &self,
        program: &Program,
        entry: Symbol,
        interner: &Interner,
    ) -> Result<Artifact, String> {
        let (exit_code, stdout) = jit_run(program, entry, interner)?;
        Ok(Artifact::Executed { exit_code, stdout })
    }
}

/// Lower `program` to PTX, JIT it on the local CUDA device, run `entry`, and return its exit code and
/// captured stdout — the GPU-lowering counterpart to `mercury_codegen_cranelift::jit_run` and
/// `mercury_interp::run_with_output`.
pub fn jit_run(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    let entry_fn = program
        .function(entry)
        .ok_or_else(|| format!("no entry function `{}`", interner.resolve(entry)))?;
    if !entry_fn.params.is_empty() {
        return Err(format!(
            "{UNSUPPORTED} gpu-native entry point `{}` must take no parameters",
            interner.resolve(entry)
        ));
    }
    let ptx = emit_ptx(program, entry, interner)?;
    let ret = entry_fn.ret.clone();

    let mut guard = crate::gpu::gpu();
    let g = guard.as_mut().ok_or_else(|| {
        "`--backend=gpu-native` requires a CUDA device, but none was reachable (the driver \
         dlopens `nvcuda.dll`; check the NVIDIA driver is installed)"
            .to_string()
    })?;
    run_on_device(g, &ptx, &ret)
}

/// Emit the full PTX module for `program` with `entry` as the launched kernel. Public so tests and
/// tooling can inspect the generated PTX (and so a future `--emit=ptx` can surface it).
pub fn emit_ptx(program: &Program, entry: Symbol, interner: &Interner) -> Result<String, String> {
    let entry_idx = program
        .funcs
        .iter()
        .position(|f| f.name == entry)
        .ok_or_else(|| format!("no entry function `{}`", interner.resolve(entry)))?;

    let mut func_idx: HashMap<Symbol, usize> = HashMap::new();
    for (i, f) in program.funcs.iter().enumerate() {
        func_idx.insert(f.name, i);
    }

    let mut out = String::new();
    out.push_str(".version 7.8\n.target sm_89\n.address_size 64\n\n");

    // Emit a device `.func` helper for each recognized runtime kernel the program calls (matmul /
    // reduce / norm / int8 GEMM / elementwise). These reproduce the CPU microkernels' numeric
    // contract as naive single-thread device loops, so the whole program runs on the GPU; the
    // CPU<->GPU tolerance gate covers reduction-order / SFU differences. Emitted before the user
    // functions that call them. (Recognized-op dispatch to the *tuned* launchers is a later phase.)
    {
        let mut seen = std::collections::HashSet::new();
        for f in &program.funcs {
            for b in &f.blocks {
                for inst in &b.insts {
                    if let Op::Call { func, .. } = &inst.op {
                        if let Some(h) = rt_helper(interner.resolve(*func)) {
                            if seen.insert(h.ptx_name) {
                                out.push_str(h.def);
                                out.push('\n');
                            }
                        }
                    }
                }
            }
        }
    }

    // Forward-declare every function so calls (incl. recursion / mutual recursion) resolve
    // regardless of definition order.
    for (i, f) in program.funcs.iter().enumerate() {
        out.push_str(&fn_signature(f, i));
        out.push_str(";\n");
    }
    out.push('\n');

    // Define each function body.
    for (i, f) in program.funcs.iter().enumerate() {
        let body = FnEmit::lower(f, i, &func_idx, program, interner)?;
        out.push_str(&body);
        out.push('\n');
    }

    // The launched entry kernel: a thin wrapper that calls the entry `.func` and records the exit code.
    out.push_str(&emit_entry_kernel(entry_idx, &program.funcs[entry_idx].ret));
    Ok(out)
}

// ============================================================================================
// Register classes & type mapping
// ============================================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum RC {
    /// 64-bit integer / pointer (`%rd`). Integer values are kept sign-extended to 64 bits.
    Rd,
    /// 32-bit float (`%f`). `f16`/`bf16` compute as `f32`, matching the interpreter/Cranelift.
    F32,
    /// 64-bit float (`%fd`).
    F64,
}

fn rc_of(t: &MirType) -> RC {
    match t {
        MirType::F64 => RC::F64,
        MirType::F16 | MirType::BF16 | MirType::F32 => RC::F32,
        _ => RC::Rd,
    }
}

/// The `.param`/ABI type string for a value of MIR type `t`.
fn abi_ty(t: &MirType) -> &'static str {
    match rc_of(t) {
        RC::Rd => "b64",
        RC::F32 => "f32",
        RC::F64 => "f64",
    }
}

/// The `mov` type suffix for a register class.
fn mov_ty(rc: RC) -> &'static str {
    match rc {
        RC::Rd => "b64",
        RC::F32 => "f32",
        RC::F64 => "f64",
    }
}

/// Width in bits of an integer MIR type (mirrors `mercury_interp::int_bits`).
fn int_bits(t: &MirType) -> u32 {
    match t {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        _ => 64,
    }
}

/// Size in bytes of a MIR type (for `gep` scaling and `alloca` frame sizing); mirrors Cranelift's.
fn size_of(t: &MirType) -> u64 {
    match t {
        MirType::I1 | MirType::I8 => 1,
        MirType::I16 | MirType::F16 | MirType::BF16 => 2,
        MirType::I32 | MirType::F32 => 4,
        MirType::I64 | MirType::F64 | MirType::Ptr => 8,
        MirType::Vec(e, n) | MirType::Array(e, n) => size_of(e) * (*n as u64),
        MirType::Void => 0,
    }
}

/// Sign-extend a constant to a 64-bit value matching the interpreter's `mask(v, ty)` (then printed
/// to PTX as the unsigned 64-bit reinterpretation, which always parses).
fn const_to_u64(v: i128, ty: &MirType) -> u64 {
    let sext: i64 = match ty {
        MirType::I1 => (v & 1) as i64,
        MirType::I8 => v as i8 as i64,
        MirType::I16 => v as i16 as i64,
        MirType::I32 => v as i32 as i64,
        _ => v as i64,
    };
    sext as u64
}

// ============================================================================================
// Per-function PTX emission
// ============================================================================================

struct FnEmit<'a> {
    func: &'a Function,
    program: &'a Program,
    interner: &'a Interner,
    func_idx: &'a HashMap<Symbol, usize>,

    body: String,
    // Register counters per class (final values give the `.reg` declaration counts).
    n_rd: u32,
    n_r: u32, // 32-bit scratch (`%r`) for narrow loads/stores, shift counts, bitcasts
    n_f: u32,
    n_fd: u32,
    n_p: u32,
    // ValueId -> assigned register name (assigned lazily, once, on first use).
    vreg: HashMap<u32, String>,
    // ValueId -> its N lane registers, for SIMD `<N x T>` values (scalarized per lane).
    vlanes: HashMap<u32, Vec<String>>,
    // Unique label counter (for cond-branch edge fixups).
    n_lbl: u32,

    // Stack frame: `ValueId(alloca) -> byte offset`, plus the total size and the generic base reg.
    frame_off: HashMap<u32, u64>,
    frame_bytes: u64,
    frame_base: String,
    // The cvta'd-to-global context-buffer base register (print/assert/exit live here).
    ctx_reg: String,
}

impl<'a> FnEmit<'a> {
    fn lower(
        func: &'a Function,
        idx: usize,
        func_idx: &'a HashMap<Symbol, usize>,
        program: &'a Program,
        interner: &'a Interner,
    ) -> Result<String, String> {
        let mut e = FnEmit {
            func,
            program,
            interner,
            func_idx,
            body: String::new(),
            n_rd: 0,
            n_r: 0,
            n_f: 0,
            n_fd: 0,
            n_p: 0,
            vreg: HashMap::new(),
            vlanes: HashMap::new(),
            n_lbl: 0,
            frame_off: HashMap::new(),
            frame_bytes: 0,
            frame_base: String::new(),
            ctx_reg: String::new(),
        };
        e.assign_frame();
        e.translate()?;

        // Assemble: signature, reg/local declarations, then preamble + body.
        let mut out = String::new();
        out.push_str(&fn_signature(func, idx));
        out.push_str("\n{\n");
        if e.n_rd > 0 {
            let _ = writeln!(out, "    .reg .b64 %rd<{}>;", e.n_rd);
        }
        if e.n_r > 0 {
            let _ = writeln!(out, "    .reg .b32 %r<{}>;", e.n_r);
        }
        if e.n_f > 0 {
            let _ = writeln!(out, "    .reg .f32 %f<{}>;", e.n_f);
        }
        if e.n_fd > 0 {
            let _ = writeln!(out, "    .reg .f64 %fd<{}>;", e.n_fd);
        }
        if e.n_p > 0 {
            let _ = writeln!(out, "    .reg .pred %p<{}>;", e.n_p);
        }
        if e.frame_bytes > 0 {
            let _ = writeln!(out, "    .local .align 8 .b8 __frame[{}];", e.frame_bytes);
        }
        out.push_str(&e.body);
        out.push_str("}\n");
        Ok(out)
    }

    // --- register / temp allocation ---

    fn fresh(&mut self, rc: RC) -> String {
        match rc {
            RC::Rd => {
                let r = format!("%rd{}", self.n_rd);
                self.n_rd += 1;
                r
            }
            RC::F32 => {
                let r = format!("%f{}", self.n_f);
                self.n_f += 1;
                r
            }
            RC::F64 => {
                let r = format!("%fd{}", self.n_fd);
                self.n_fd += 1;
                r
            }
        }
    }

    fn fresh_pred(&mut self) -> String {
        let r = format!("%p{}", self.n_p);
        self.n_p += 1;
        r
    }

    /// A fresh 32-bit scratch register (`%r`), for narrow loads/stores, shift counts, and bitcasts.
    fn fresh_r32(&mut self) -> String {
        let r = format!("%r{}", self.n_r);
        self.n_r += 1;
        r
    }

    /// The register holding `v` (assigned on first reference, by `v`'s MIR type).
    fn reg(&mut self, v: ValueId) -> String {
        if let Some(r) = self.vreg.get(&v.0) {
            return r.clone();
        }
        let rc = rc_of(self.func.value_type(v));
        let r = self.fresh(rc);
        self.vreg.insert(v.0, r.clone());
        r
    }

    fn ty(&self, v: ValueId) -> MirType {
        self.func.value_type(v).clone()
    }

    fn emit(&mut self, s: &str) {
        self.body.push_str("    ");
        self.body.push_str(s);
        self.body.push('\n');
    }

    // --- stack frame ---

    /// Pre-scan all allocas and assign each a (8-aligned) byte offset within one `.local` frame.
    fn assign_frame(&mut self) {
        let mut off = 0u64;
        for b in &self.func.blocks {
            for inst in &b.insts {
                if let Op::Alloca(ty) = &inst.op {
                    if let Some(r) = inst.result {
                        let sz = size_of(ty).max(1);
                        let slot = sz.div_ceil(8) * 8; // round up to 8
                        self.frame_off.insert(r.0, off);
                        off += slot;
                    }
                }
            }
        }
        self.frame_bytes = off;
    }

    // --- translation ---

    fn translate(&mut self) -> Result<(), String> {
        // Preamble: bind ctx (first hidden param), the stack-frame base, and the real params.
        let ctx_raw = self.fresh(RC::Rd);
        self.ctx_reg = self.fresh(RC::Rd);
        self.emit(&format!("ld.param.u64 {ctx_raw}, [p_ctx];"));
        self.emit(&format!("cvta.to.global.u64 {}, {ctx_raw};", self.ctx_reg));

        if self.frame_bytes > 0 {
            let loc = self.fresh(RC::Rd);
            self.frame_base = self.fresh(RC::Rd);
            self.emit(&format!("mov.u64 {loc}, __frame;"));
            self.emit(&format!("cvta.local.u64 {}, {loc};", self.frame_base));
        }

        let params: Vec<ValueId> = self.func.params.clone();
        for (i, p) in params.iter().enumerate() {
            let r = self.reg(*p);
            let abi = abi_ty(self.func.value_type(*p));
            self.emit(&format!("ld.param.{abi} {r}, [p{i}];"));
        }

        // Branch to the entry block, then emit blocks in id order.
        self.emit(&format!("bra BB{};", self.func.entry.0));
        let blocks = self.func.blocks.clone();
        for b in &blocks {
            self.body.push_str(&format!("BB{}:\n", b.id.0));
            for inst in &b.insts {
                self.lower_inst(inst)?;
            }
            self.lower_term(&b.term)?;
        }
        Ok(())
    }

    fn lower_inst(&mut self, inst: &mercury_mir::Inst) -> Result<(), String> {
        // Result-less ops first.
        match &inst.op {
            Op::Store { ptr, value } => return self.lower_store(*ptr, *value),
            Op::Call { func, args } => return self.lower_call(*func, args, inst.result),
            _ => {}
        }
        let r = inst
            .result
            .ok_or_else(|| format!("{UNSUPPORTED} value-less op {:?}", inst.op))?;
        let rty = self.ty(r);

        // Vectorized MIR (`<N x T>`): the AST-level vectorizer fires even at -O0 on array loops.
        // Scalarize each lane through the same reg-based `*_into` emitters the scalar path uses.
        if rty.is_vector() {
            return self.lower_vec_inst(inst, r, &rty);
        }

        match &inst.op {
            Op::ConstInt(v, ty) => {
                let d = self.reg(r);
                self.emit(&format!("mov.b64 {d}, {};", const_to_u64(*v, ty)));
            }
            Op::ConstFloat(v, ty) => {
                let d = self.reg(r);
                if matches!(ty, MirType::F64) {
                    self.emit(&format!("mov.f64 {d}, 0d{:016X};", v.to_bits()));
                } else {
                    self.emit(&format!("mov.f32 {d}, 0f{:08X};", (*v as f32).to_bits()));
                }
            }
            Op::Bin(op, l, r2) => self.lower_bin(*op, *l, *r2, &rty, r)?,
            Op::Cmp(op, l, r2) => self.lower_cmp(*op, *l, *r2, r),
            Op::Neg(v) => {
                let x = self.reg(*v);
                let d = self.reg(r);
                self.neg_into(&d, &x, &rty);
            }
            Op::Not(v) => {
                let x = self.reg(*v);
                let d = self.reg(r);
                self.not_into(&d, &x, &rty);
            }
            Op::Cast(kind, v, to) => self.lower_cast(*kind, *v, to, r)?,
            Op::Select(c, a, b) => self.lower_select(*c, *a, *b, &rty, r),
            Op::Alloca(_) => {
                let off = *self.frame_off.get(&r.0).expect("alloca offset");
                let base = self.frame_base.clone();
                let d = self.reg(r);
                if off == 0 {
                    self.emit(&format!("mov.b64 {d}, {base};"));
                } else {
                    self.emit(&format!("add.s64 {d}, {base}, {off};"));
                }
            }
            Op::Load(p, ty) => self.lower_load(*p, ty, r),
            Op::Gep { ptr, index, elem } => {
                let base = self.reg(*ptr);
                let idx = self.reg(*index);
                let off = self.fresh(RC::Rd);
                let d = self.reg(r);
                self.emit(&format!("mul.lo.s64 {off}, {idx}, {};", size_of(elem)));
                self.emit(&format!("add.s64 {d}, {base}, {off};"));
            }
            Op::Fma(a, b, c) => {
                let rc = rc_of(&rty);
                let av = self.coerce_float(*a, rc);
                let bv = self.coerce_float(*b, rc);
                let cv = self.coerce_float(*c, rc);
                let d = self.reg(r);
                self.fma_into(&d, &av, &bv, &cv, &rty);
            }
            Op::Sqrt(v) => {
                let rc = rc_of(&rty);
                let x = self.coerce_float(*v, rc);
                let d = self.reg(r);
                self.sqrt_into(&d, &x, &rty);
            }
            Op::Round(mode, v) => {
                let rc = rc_of(&rty);
                let x = self.coerce_float(*v, rc);
                let d = self.reg(r);
                self.round_into(&d, *mode, &x, &rty);
            }
            Op::Splat(_) => {
                return Err(format!(
                    "{UNSUPPORTED} SIMD `Splat` (vectorized MIR) not yet lowered to PTX"
                ))
            }
            Op::FuncAddr(_) => {
                return Err(format!(
                    "{UNSUPPORTED} `FuncAddr` (@parallel outlined body) not yet lowered to PTX"
                ))
            }
            Op::Store { .. } | Op::Call { .. } => unreachable!("handled above"),
        }
        Ok(())
    }

    // --- integer / float binary ops ---

    fn lower_bin(
        &mut self,
        op: BinOp,
        l: ValueId,
        r2: ValueId,
        rty: &MirType,
        res: ValueId,
    ) -> Result<(), String> {
        let (a, b) = if op.is_float() {
            let rc = rc_of(rty);
            (self.coerce_float(l, rc), self.coerce_float(r2, rc))
        } else {
            (self.reg(l), self.reg(r2))
        };
        let d = self.reg(res);
        self.bin_into(&d, op, &a, &b, rty)
    }

    /// Emit `d = a OP b` for one (scalar or lane) value. Operands are registers already of the result
    /// type; floats round per op (no auto-FMA), ints compute in 64-bit and mask to the result width.
    fn bin_into(&mut self, d: &str, op: BinOp, a: &str, b: &str, rty: &MirType) -> Result<(), String> {
        use BinOp::*;
        if op.is_float() {
            let sfx = float_suffix(rty);
            match op {
                FAdd => self.emit(&format!("add.rn.{sfx} {d}, {a}, {b};")),
                FSub => self.emit(&format!("sub.rn.{sfx} {d}, {a}, {b};")),
                FMul => self.emit(&format!("mul.rn.{sfx} {d}, {a}, {b};")),
                FDiv => self.emit(&format!("div.rn.{sfx} {d}, {a}, {b};")),
                // No PTX frem, and the `a - trunc(a/b)*b` identity loses precision past the mantissa's
                // integer range (`1e18 % 3` -> 0, not 1), so it cannot meet the differential gate. A
                // true device fmod is a later increment; decline cleanly for now.
                FRem => {
                    return Err(format!(
                        "{UNSUPPORTED} float `%` (fmod) not yet lowered to PTX (needs a true fmod)"
                    ))
                }
                _ => unreachable!(),
            }
            return Ok(());
        }
        let bits = int_bits(rty);
        match op {
            Add => self.emit(&format!("add.s64 {d}, {a}, {b};")),
            Sub => self.emit(&format!("sub.s64 {d}, {a}, {b};")),
            Mul => self.emit(&format!("mul.lo.s64 {d}, {a}, {b};")),
            And => self.emit(&format!("and.b64 {d}, {a}, {b};")),
            Or => self.emit(&format!("or.b64 {d}, {a}, {b};")),
            Xor => self.emit(&format!("xor.b64 {d}, {a}, {b};")),
            SDiv => self.guarded_divrem("div.s64", a, b, d),
            SRem => self.guarded_divrem("rem.s64", a, b, d),
            UDiv => {
                let au = self.mask_unsigned(a, bits);
                let bu = self.mask_unsigned(b, bits);
                self.guarded_divrem("div.u64", &au, &bu, d);
            }
            URem => {
                let au = self.mask_unsigned(a, bits);
                let bu = self.mask_unsigned(b, bits);
                self.guarded_divrem("rem.u64", &au, &bu, d);
            }
            Shl => {
                let sc = self.shift_count(b, bits);
                self.emit(&format!("shl.b64 {d}, {a}, {sc};"));
            }
            LShr => {
                let au = self.mask_unsigned(a, bits);
                let sc = self.shift_count(b, bits);
                self.emit(&format!("shr.u64 {d}, {au}, {sc};"));
            }
            AShr => {
                let sc = self.shift_count(b, bits);
                self.emit(&format!("shr.s64 {d}, {a}, {sc};"));
            }
            FAdd | FSub | FMul | FDiv | FRem => unreachable!("handled above"),
        }
        self.mask_int(d, rty);
        Ok(())
    }

    fn neg_into(&mut self, d: &str, a: &str, rty: &MirType) {
        if rty.is_float() {
            self.emit(&format!("neg.{} {d}, {a};", float_suffix(rty)));
        } else {
            self.emit(&format!("neg.s64 {d}, {a};"));
            self.mask_int(d, rty);
        }
    }

    fn not_into(&mut self, d: &str, a: &str, rty: &MirType) {
        self.emit(&format!("not.b64 {d}, {a};"));
        self.mask_int(d, rty);
    }

    fn fma_into(&mut self, d: &str, a: &str, b: &str, c: &str, rty: &MirType) {
        self.emit(&format!("fma.rn.{} {d}, {a}, {b}, {c};", float_suffix(rty)));
    }

    fn sqrt_into(&mut self, d: &str, a: &str, rty: &MirType) {
        self.emit(&format!("sqrt.rn.{} {d}, {a};", float_suffix(rty)));
    }

    fn round_into(&mut self, d: &str, mode: RoundMode, a: &str, rty: &MirType) {
        let sfx = float_suffix(rty);
        let m = match mode {
            RoundMode::Nearest => "rni",
            RoundMode::Floor => "rmi",
            RoundMode::Ceil => "rpi",
            RoundMode::Trunc => "rzi",
        };
        self.emit(&format!("cvt.{m}.{sfx}.{sfx} {d}, {a};"));
    }

    /// `d = (b == 0) ? 0 : (a OP b)` with `b` swapped to 1 in the divide so the hardware never sees a
    /// zero divisor (matching the interpreter/Cranelift "div-by-zero yields 0, no trap" contract).
    fn guarded_divrem(&mut self, opc: &str, a: &str, b: &str, d: &str) {
        let pz = self.fresh_pred();
        let one = self.fresh(RC::Rd);
        let zero = self.fresh(RC::Rd);
        let safe = self.fresh(RC::Rd);
        let q = self.fresh(RC::Rd);
        self.emit(&format!("setp.eq.s64 {pz}, {b}, 0;"));
        self.emit(&format!("mov.b64 {one}, 1;"));
        self.emit(&format!("mov.b64 {zero}, 0;"));
        self.emit(&format!("selp.b64 {safe}, {one}, {b}, {pz};"));
        self.emit(&format!("{opc} {q}, {a}, {safe};"));
        self.emit(&format!("selp.b64 {d}, {zero}, {q}, {pz};"));
    }

    /// Mask the shift amount to the result width (x86/Cranelift/interp semantics) and narrow to the
    /// `.u32` PTX shift operand.
    fn shift_count(&mut self, b: &str, bits: u32) -> String {
        let masked = self.fresh(RC::Rd);
        let sc = self.fresh_r32();
        self.emit(&format!("and.b64 {masked}, {b}, {};", bits - 1));
        // PTX shift amount is a 32-bit value.
        self.emit(&format!("cvt.u32.u64 {sc}, {masked};"));
        sc
    }

    /// Zero-extend the low `bits` of a sign-extended value (the unsigned-width window the interpreter
    /// reads for unsigned ops). No-op at 64 bits.
    fn mask_unsigned(&mut self, src: &str, bits: u32) -> String {
        if bits >= 64 {
            return src.to_string();
        }
        let d = self.fresh(RC::Rd);
        let mask: u64 = (1u64 << bits) - 1;
        self.emit(&format!("and.b64 {d}, {src}, {mask};"));
        d
    }

    /// Narrow a computed 64-bit integer result to its declared MIR width, sign-extended (mirrors the
    /// interpreter's per-result `mask`). No-op for `i64`/pointer.
    fn mask_int(&mut self, reg: &str, ty: &MirType) {
        match ty {
            MirType::I1 => self.emit(&format!("and.b64 {reg}, {reg}, 1;")),
            MirType::I8 => self.emit(&format!("cvt.s64.s8 {reg}, {reg};")),
            MirType::I16 => self.emit(&format!("cvt.s64.s16 {reg}, {reg};")),
            MirType::I32 => self.emit(&format!("cvt.s64.s32 {reg}, {reg};")),
            _ => {}
        }
    }

    // --- compares / select ---

    fn lower_cmp(&mut self, op: CmpOp, l: ValueId, r2: ValueId, res: ValueId) {
        let (a, b, opnd_ty) = if op.is_float() {
            // Compare in the wider operand type (Cranelift's rule), which preserves order vs the
            // interpreter's f64 compare.
            let common = if matches!(self.ty(l), MirType::F64) || matches!(self.ty(r2), MirType::F64)
            {
                MirType::F64
            } else {
                MirType::F32
            };
            let rc = rc_of(&common);
            (self.coerce_float(l, rc), self.coerce_float(r2, rc), common)
        } else {
            (self.reg(l), self.reg(r2), self.ty(l))
        };
        let d = self.reg(res);
        self.cmp_into(&d, op, &a, &b, &opnd_ty);
    }

    /// Emit `d = (a CMP b) ? 1 : 0` into a 64-bit `i1` holder. `opnd_ty` selects the compare width.
    fn cmp_into(&mut self, d: &str, op: CmpOp, a: &str, b: &str, opnd_ty: &MirType) {
        let p = self.fresh_pred();
        if op.is_float() {
            self.emit(&format!(
                "setp.{}.{} {p}, {a}, {b};",
                float_cc(op),
                float_suffix(opnd_ty)
            ));
        } else {
            self.emit(&format!("setp.{} {p}, {a}, {b};", int_cc(op)));
        }
        let one = self.fresh(RC::Rd);
        let zero = self.fresh(RC::Rd);
        self.emit(&format!("mov.b64 {one}, 1;"));
        self.emit(&format!("mov.b64 {zero}, 0;"));
        self.emit(&format!("selp.b64 {d}, {one}, {zero}, {p};"));
    }

    fn lower_select(&mut self, c: ValueId, a: ValueId, b: ValueId, rty: &MirType, res: ValueId) {
        let rc = rc_of(rty);
        let cc = self.reg(c);
        let (av, bv) = if rty.is_float() {
            (self.coerce_float(a, rc), self.coerce_float(b, rc))
        } else {
            (self.reg(a), self.reg(b))
        };
        let d = self.reg(res);
        self.select_into(&d, &cc, &av, &bv, rty);
    }

    /// Emit `d = c ? a : b` for one (scalar or lane) value; `c` is a 64-bit `i1` (0/1).
    fn select_into(&mut self, d: &str, c: &str, a: &str, b: &str, rty: &MirType) {
        let rc = rc_of(rty);
        let p = self.fresh_pred();
        self.emit(&format!("setp.ne.s64 {p}, {c}, 0;"));
        self.emit(&format!("selp.{} {d}, {a}, {b}, {p};", mov_ty(rc)));
        if !rty.is_float() {
            self.mask_int(d, rty);
        }
    }

    // --- casts ---

    fn lower_cast(
        &mut self,
        kind: CastKind,
        v: ValueId,
        to: &MirType,
        res: ValueId,
    ) -> Result<(), String> {
        let from = self.ty(v);
        let x = self.reg(v);
        let d = self.reg(res);
        self.cast_into(&d, kind, &x, &from, to)
    }

    /// Emit a cast `d = (to)a` for one (scalar or lane) value, mirroring `mercury_interp::apply_cast`.
    fn cast_into(
        &mut self,
        d: &str,
        kind: CastKind,
        x: &str,
        from: &MirType,
        to: &MirType,
    ) -> Result<(), String> {
        use CastKind::*;
        match kind {
            SExt | Trunc => {
                self.emit(&format!("mov.b64 {d}, {x};"));
                self.mask_int(d, to);
            }
            ZExt => {
                let u = self.mask_unsigned(x, int_bits(from));
                self.emit(&format!("mov.b64 {d}, {u};"));
                self.mask_int(d, to);
            }
            SiToFp => {
                if matches!(to, MirType::F64) {
                    self.emit(&format!("cvt.rn.f64.s64 {d}, {x};"));
                } else {
                    // Match the interpreter's i->f64->f32 double rounding exactly.
                    let t = self.fresh(RC::F64);
                    self.emit(&format!("cvt.rn.f64.s64 {t}, {x};"));
                    self.emit(&format!("cvt.rn.f32.f64 {d}, {t};"));
                }
            }
            UiToFp => {
                let u = self.mask_unsigned(x, int_bits(from));
                if matches!(to, MirType::F64) {
                    self.emit(&format!("cvt.rn.f64.u64 {d}, {u};"));
                } else {
                    let t = self.fresh(RC::F64);
                    self.emit(&format!("cvt.rn.f64.u64 {t}, {u};"));
                    self.emit(&format!("cvt.rn.f32.f64 {d}, {t};"));
                }
            }
            FpToSi => self.fp_to_int_into(d, x, from, to, true),
            FpToUi => self.fp_to_int_into(d, x, from, to, false),
            FpExt => {
                // Widen to f64 (the only widening FpExt target here); narrowing is FpTrunc.
                if rc_of(from) == RC::F64 {
                    self.emit(&format!("mov.f64 {d}, {x};"));
                } else {
                    self.emit(&format!("cvt.f64.f32 {d}, {x};"));
                }
            }
            FpTrunc => {
                if matches!(to, MirType::F32) {
                    if rc_of(from) == RC::F64 {
                        self.emit(&format!("cvt.rn.f32.f64 {d}, {x};"));
                    } else {
                        self.emit(&format!("mov.f32 {d}, {x};"));
                    }
                } else {
                    return Err(format!(
                        "{UNSUPPORTED} FpTrunc to {:?} (bf16/f16 narrowing) not yet lowered",
                        to
                    ));
                }
            }
            Bitcast => self.bitcast_into(d, x, from, to),
            IntToPtr | PtrToInt => self.emit(&format!("mov.b64 {d}, {x};")),
        }
        Ok(())
    }

    fn fp_to_int_into(&mut self, d: &str, x: &str, from: &MirType, to: &MirType, signed: bool) {
        let fsfx = float_suffix(from);
        let s = if signed { 's' } else { 'u' };
        let w = match to {
            MirType::I8 => 8,
            MirType::I16 => 16,
            MirType::I64 => 64,
            _ => 32,
        };
        if w == 64 {
            self.emit(&format!("cvt.rzi.{s}64.{fsfx} {d}, {x};"));
        } else {
            // cvt saturates to the destination width; sign-extend into the 64-bit holder, then mask.
            self.emit(&format!("cvt.rzi.{s}{w}.{fsfx} {d}, {x};"));
            self.mask_int(d, to);
        }
    }

    fn bitcast_into(&mut self, d: &str, x: &str, from: &MirType, to: &MirType) {
        match (from, to) {
            (MirType::I32, MirType::F32) => {
                let lo = self.fresh_r32();
                self.emit(&format!("cvt.u32.u64 {lo}, {x};"));
                self.emit(&format!("mov.b32 {d}, {lo};"));
            }
            (MirType::F32, MirType::I32) => {
                let lo = self.fresh_r32();
                self.emit(&format!("mov.b32 {lo}, {x};"));
                self.emit(&format!("cvt.s64.s32 {d}, {lo};"));
            }
            (MirType::I64, MirType::F64) | (MirType::F64, MirType::I64) => {
                self.emit(&format!("mov.b64 {d}, {x};"));
            }
            _ => {
                // Same-class passthrough.
                self.emit(&format!("mov.{} {d}, {x};", mov_ty(rc_of(to))));
            }
        }
    }

    // --- memory ---

    fn lower_load(&mut self, p: ValueId, ty: &MirType, res: ValueId) {
        let addr = self.reg(p);
        let d = self.reg(res);
        self.load_into(&d, &addr, ty);
    }

    /// Load one (scalar or lane) value of MIR type `ty` from generic address `addr` into `d`.
    fn load_into(&mut self, d: &str, addr: &str, ty: &MirType) {
        match ty {
            MirType::F32 => self.emit(&format!("ld.f32 {d}, [{addr}];")),
            MirType::F64 => self.emit(&format!("ld.f64 {d}, [{addr}];")),
            MirType::I64 | MirType::Ptr => self.emit(&format!("ld.u64 {d}, [{addr}];")),
            MirType::I8 => {
                let w = self.fresh_r32();
                self.emit(&format!("ld.s8 {w}, [{addr}];"));
                self.emit(&format!("cvt.s64.s32 {d}, {w};"));
            }
            MirType::I16 => {
                let w = self.fresh_r32();
                self.emit(&format!("ld.s16 {w}, [{addr}];"));
                self.emit(&format!("cvt.s64.s32 {d}, {w};"));
            }
            _ => {
                // i32 / i1: load 32 bits, sign-extend to 64.
                let w = self.fresh_r32();
                self.emit(&format!("ld.u32 {w}, [{addr}];"));
                self.emit(&format!("cvt.s64.s32 {d}, {w};"));
            }
        }
    }

    fn lower_store(&mut self, ptr: ValueId, value: ValueId) -> Result<(), String> {
        let vty = self.ty(value);
        let addr = self.reg(ptr);
        // Vector store: write each lane at addr + i*sizeof(lane).
        if let MirType::Vec(elem, n) = &vty {
            let elem = (**elem).clone();
            let esz = size_of(&elem);
            let lanes = self.lanes(value)?;
            for (i, lv) in lanes.into_iter().enumerate().take(*n as usize) {
                let la = if i == 0 {
                    addr.clone()
                } else {
                    let t = self.fresh(RC::Rd);
                    self.emit(&format!("add.s64 {t}, {addr}, {};", i as u64 * esz));
                    t
                };
                self.store_into(&la, &lv, &elem)?;
            }
            return Ok(());
        }
        let v = self.reg(value);
        self.store_into(&addr, &v, &vty)
    }

    /// Store one (scalar or lane) value `v` of MIR type `ty` to generic address `addr`.
    fn store_into(&mut self, addr: &str, v: &str, ty: &MirType) -> Result<(), String> {
        match ty {
            MirType::F32 => self.emit(&format!("st.f32 [{addr}], {v};")),
            MirType::F64 => self.emit(&format!("st.f64 [{addr}], {v};")),
            MirType::I64 | MirType::Ptr => self.emit(&format!("st.u64 [{addr}], {v};")),
            MirType::F16 | MirType::BF16 => {
                return Err(format!("{UNSUPPORTED} store of {:?} (half precision)", ty))
            }
            // Narrow stores take a 32-bit source register (low bits); narrow the 64-bit value first.
            MirType::I8 => {
                let w = self.fresh_r32();
                self.emit(&format!("cvt.u32.u64 {w}, {v};"));
                self.emit(&format!("st.u8 [{addr}], {w};"));
            }
            MirType::I16 => {
                let w = self.fresh_r32();
                self.emit(&format!("cvt.u32.u64 {w}, {v};"));
                self.emit(&format!("st.u16 [{addr}], {w};"));
            }
            _ => {
                // i32 / i1
                let w = self.fresh_r32();
                self.emit(&format!("cvt.u32.u64 {w}, {v};"));
                self.emit(&format!("st.u32 [{addr}], {w};"));
            }
        }
        Ok(())
    }

    // --- SIMD (vector) lowering: scalarize each `<N x T>` op into N lane ops ---

    /// The N lane registers of a SIMD value (materialized by the op that produced it).
    fn lanes(&mut self, v: ValueId) -> Result<Vec<String>, String> {
        self.vlanes.get(&v.0).cloned().ok_or_else(|| {
            format!(
                "{UNSUPPORTED} SIMD value v{} used before definition (vector block params not lowered)",
                v.0
            )
        })
    }

    /// Address of lane `i` given a base address register and the lane byte size.
    fn lane_addr(&mut self, base: &str, i: usize, esz: u64) -> String {
        if i == 0 {
            base.to_string()
        } else {
            let t = self.fresh(RC::Rd);
            self.emit(&format!("add.s64 {t}, {base}, {};", i as u64 * esz));
            t
        }
    }

    /// Scalarize a vectorized (`<N x T>`) instruction into N lane operations, reusing the same
    /// reg-based emitters the scalar path uses. Vector loads/stores stride by the lane size; a `Splat`
    /// aliases the scalar across all lanes. (Vectorized values are produced and consumed within one
    /// block, so there are no vector block params to thread.)
    fn lower_vec_inst(
        &mut self,
        inst: &mercury_mir::Inst,
        r: ValueId,
        rty: &MirType,
    ) -> Result<(), String> {
        let n = match rty {
            MirType::Vec(_, n) => *n as usize,
            _ => unreachable!("lower_vec_inst on non-vector"),
        };
        let lane = rty.lane_type().clone();
        let lane_rc = rc_of(&lane);

        let out: Vec<String> = match &inst.op {
            Op::Splat(s) => {
                let sr = self.reg(*s);
                vec![sr; n]
            }
            Op::Load(p, _) => {
                let addr = self.reg(*p);
                let esz = size_of(&lane);
                let mut ls = Vec::with_capacity(n);
                for i in 0..n {
                    let la = self.lane_addr(&addr, i, esz);
                    let d = self.fresh(lane_rc);
                    self.load_into(&d, &la, &lane);
                    ls.push(d);
                }
                ls
            }
            Op::Bin(op, l, r2) => {
                let la = self.lanes(*l)?;
                let lb = self.lanes(*r2)?;
                let mut ls = Vec::with_capacity(n);
                for i in 0..n {
                    let d = self.fresh(lane_rc);
                    self.bin_into(&d, *op, &la[i], &lb[i], &lane)?;
                    ls.push(d);
                }
                ls
            }
            Op::Fma(a, b, c) => {
                let la = self.lanes(*a)?;
                let lb = self.lanes(*b)?;
                let lc = self.lanes(*c)?;
                let mut ls = Vec::with_capacity(n);
                for i in 0..n {
                    let d = self.fresh(lane_rc);
                    self.fma_into(&d, &la[i], &lb[i], &lc[i], &lane);
                    ls.push(d);
                }
                ls
            }
            Op::Sqrt(v) => {
                let lv = self.lanes(*v)?;
                let mut ls = Vec::with_capacity(n);
                for i in 0..n {
                    let d = self.fresh(lane_rc);
                    self.sqrt_into(&d, &lv[i], &lane);
                    ls.push(d);
                }
                ls
            }
            Op::Round(mode, v) => {
                let lv = self.lanes(*v)?;
                let mut ls = Vec::with_capacity(n);
                for i in 0..n {
                    let d = self.fresh(lane_rc);
                    self.round_into(&d, *mode, &lv[i], &lane);
                    ls.push(d);
                }
                ls
            }
            Op::Neg(v) => {
                let lv = self.lanes(*v)?;
                let mut ls = Vec::with_capacity(n);
                for i in 0..n {
                    let d = self.fresh(lane_rc);
                    self.neg_into(&d, &lv[i], &lane);
                    ls.push(d);
                }
                ls
            }
            Op::Not(v) => {
                let lv = self.lanes(*v)?;
                let mut ls = Vec::with_capacity(n);
                for i in 0..n {
                    let d = self.fresh(lane_rc);
                    self.not_into(&d, &lv[i], &lane);
                    ls.push(d);
                }
                ls
            }
            Op::Cmp(op, l, r2) => {
                let opnd_lane = self.ty(*l).lane_type().clone();
                let la = self.lanes(*l)?;
                let lb = self.lanes(*r2)?;
                let mut ls = Vec::with_capacity(n);
                for i in 0..n {
                    let d = self.fresh(RC::Rd);
                    self.cmp_into(&d, *op, &la[i], &lb[i], &opnd_lane);
                    ls.push(d);
                }
                ls
            }
            Op::Select(c, a, b) => {
                let lc = self.lanes(*c)?;
                let la = self.lanes(*a)?;
                let lb = self.lanes(*b)?;
                let mut ls = Vec::with_capacity(n);
                for i in 0..n {
                    let d = self.fresh(lane_rc);
                    self.select_into(&d, &lc[i], &la[i], &lb[i], &lane);
                    ls.push(d);
                }
                ls
            }
            Op::Cast(kind, v, to) => {
                let from_lane = self.ty(*v).lane_type().clone();
                let to_lane = to.lane_type().clone();
                let lv = self.lanes(*v)?;
                let mut ls = Vec::with_capacity(n);
                for i in 0..n {
                    let d = self.fresh(rc_of(&to_lane));
                    self.cast_into(&d, *kind, &lv[i], &from_lane, &to_lane)?;
                    ls.push(d);
                }
                ls
            }
            other => return Err(format!("{UNSUPPORTED} SIMD op {other:?} not yet lowered to PTX")),
        };
        self.vlanes.insert(r.0, out);
        Ok(())
    }

    // --- calls / intrinsics ---

    fn lower_call(
        &mut self,
        func: Symbol,
        args: &[ValueId],
        result: Option<ValueId>,
    ) -> Result<(), String> {
        if let Some(&idx) = self.func_idx.get(&func) {
            return self.lower_user_call(idx, args, result);
        }
        let name = self.interner.resolve(func).to_string();
        match name.as_str() {
            "print" | "println" => self.lower_print(args),
            "assert" => self.lower_assert(args),
            other => {
                if let Some(h) = rt_helper(other) {
                    self.emit_helper_call(h.ptx_name, h.ret, args, result);
                    Ok(())
                } else {
                    Err(format!("{UNSUPPORTED} call to `{other}` not yet lowered to PTX"))
                }
            }
        }
    }

    /// Emit a call to a device-side runtime helper (`mrt_*`): declare a param scope, pass each arg by
    /// its ABI type, call, and bind the return (if any). Same convention as a user-function call.
    fn emit_helper_call(
        &mut self,
        ptx_name: &str,
        ret: Option<RC>,
        args: &[ValueId],
        result: Option<ValueId>,
    ) {
        let abis: Vec<&'static str> = args
            .iter()
            .map(|a| abi_ty(self.func.value_type(*a)))
            .collect();
        let regs: Vec<String> = args.iter().map(|a| self.reg(*a)).collect();
        self.emit("{");
        for (i, abi) in abis.iter().enumerate() {
            self.emit(&format!(".param .{abi} _a{i};"));
        }
        if let Some(rc) = ret {
            self.emit(&format!(".param .{} _r;", mov_ty(rc)));
        }
        for (i, (r, abi)) in regs.iter().zip(&abis).enumerate() {
            self.emit(&format!("st.param.{abi} [_a{i}], {r};"));
        }
        let mut arglist = String::new();
        for i in 0..args.len() {
            if i > 0 {
                arglist.push_str(", ");
            }
            arglist.push_str(&format!("_a{i}"));
        }
        match ret {
            Some(_) => self.emit(&format!("call.uni (_r), {ptx_name}, ({arglist});")),
            None => self.emit(&format!("call.uni {ptx_name}, ({arglist});")),
        }
        if let (Some(rc), Some(res)) = (ret, result) {
            let d = self.reg(res);
            self.emit(&format!("ld.param.{} {d}, [_r];", mov_ty(rc)));
        }
        self.emit("}");
    }

    fn lower_user_call(
        &mut self,
        idx: usize,
        args: &[ValueId],
        result: Option<ValueId>,
    ) -> Result<(), String> {
        let callee = &self.program.funcs[idx];
        let ret = callee.ret.clone();
        let has_ret = !matches!(ret, MirType::Void);

        // Snapshot arg regs/types before mutating the builder.
        let arg_regs: Vec<(String, &'static str)> = args
            .iter()
            .map(|a| (self.reg(*a), abi_ty(self.func.value_type(*a))))
            .collect();
        let ctx = self.ctx_reg.clone();

        self.emit("{");
        self.emit(".param .b64 _pc;");
        for (i, (_, abi)) in arg_regs.iter().enumerate() {
            self.emit(&format!(".param .{abi} _p{i};"));
        }
        if has_ret {
            self.emit(&format!(".param .{} _pr;", abi_ty(&ret)));
        }
        self.emit(&format!("st.param.b64 [_pc], {ctx};"));
        for (i, (r, abi)) in arg_regs.iter().enumerate() {
            self.emit(&format!("st.param.{abi} [_p{i}], {r};"));
        }
        let mut arglist = String::from("_pc");
        for i in 0..arg_regs.len() {
            arglist.push_str(&format!(", _p{i}"));
        }
        if has_ret {
            self.emit(&format!("call.uni (_pr), mfn_{idx}, ({arglist});"));
        } else {
            self.emit(&format!("call.uni mfn_{idx}, ({arglist});"));
        }
        if let Some(r) = result {
            if has_ret {
                let d = self.reg(r);
                self.emit(&format!("ld.param.{} {d}, [_pr];", abi_ty(&ret)));
            }
        }
        self.emit("}");
        Ok(())
    }

    fn lower_print(&mut self, args: &[ValueId]) -> Result<(), String> {
        let arg = *args
            .first()
            .ok_or_else(|| format!("{UNSUPPORTED} print with no argument"))?;
        let aty = self.ty(arg);
        let (tag, payload) = if aty.is_float() {
            let f = self.reg(arg);
            let pl = self.fresh(RC::Rd);
            if rc_of(&aty) == RC::F32 {
                let fd = self.fresh(RC::F64);
                self.emit(&format!("cvt.f64.f32 {fd}, {f};"));
                self.emit(&format!("mov.b64 {pl}, {fd};"));
            } else {
                self.emit(&format!("mov.b64 {pl}, {f};"));
            }
            (1u64, pl)
        } else {
            (0u64, self.reg(arg))
        };
        self.emit_record(tag, &payload);
        Ok(())
    }

    /// Append one `(tag, payload)` print record to the context buffer (atomic slot, bounds-guarded).
    fn emit_record(&mut self, tag: u64, payload: &str) {
        let ctx = self.ctx_reg.clone();
        let idx = self.fresh(RC::Rd);
        let off = self.fresh(RC::Rd);
        let addr = self.fresh(RC::Rd);
        let tagr = self.fresh(RC::Rd);
        let pw = self.fresh_pred();
        self.emit(&format!("atom.global.add.u64 {idx}, [{ctx}+{CTX_COUNT_OFF}], 1;"));
        self.emit(&format!("setp.lt.u64 {pw}, {idx}, {RECORD_CAP};"));
        self.emit(&format!("mul.lo.s64 {off}, {idx}, {RECORD_BYTES};"));
        self.emit(&format!("add.s64 {addr}, {ctx}, {off};"));
        self.emit(&format!("add.s64 {addr}, {addr}, {CTX_RECORDS_OFF};"));
        self.emit(&format!("mov.b64 {tagr}, {tag};"));
        self.emit(&format!("@{pw} st.global.u64 [{addr}], {tagr};"));
        self.emit(&format!("@{pw} st.global.u64 [{addr}+8], {payload};"));
    }

    fn lower_assert(&mut self, args: &[ValueId]) -> Result<(), String> {
        let arg = *args
            .first()
            .ok_or_else(|| format!("{UNSUPPORTED} assert with no argument"))?;
        let aty = self.ty(arg);
        let ctx = self.ctx_reg.clone();
        let cond = self.reg(arg);
        let pz = self.fresh_pred();
        if aty.is_float() {
            let z = self.fresh(rc_of(&aty));
            let sfx = float_suffix(&aty);
            self.emit(&format!("mov.{sfx} {z}, 0{};", if sfx == "f64" { "d0000000000000000" } else { "f00000000" }));
            self.emit(&format!("setp.eq.{sfx} {pz}, {cond}, {z};"));
        } else {
            self.emit(&format!("setp.eq.s64 {pz}, {cond}, 0;"));
        }
        let one = self.fresh(RC::Rd);
        self.emit(&format!("mov.b64 {one}, 1;"));
        self.emit(&format!("@{pz} st.global.u64 [{ctx}+{CTX_ASSERT_OFF}], {one};"));
        Ok(())
    }

    // --- terminators ---

    fn lower_term(&mut self, term: &Terminator) -> Result<(), String> {
        match term {
            Terminator::Ret(None) => self.emit("ret;"),
            Terminator::Ret(Some(v)) => {
                let abi = abi_ty(self.func.value_type(*v));
                let r = self.reg(*v);
                self.emit(&format!("st.param.{abi} [retp], {r};"));
                self.emit("ret;");
            }
            Terminator::Br { target, args } => {
                self.emit_edge(target.0, args);
                self.emit(&format!("bra BB{};", target.0));
            }
            Terminator::CondBr {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                let c = self.reg(*cond);
                let p = self.fresh_pred();
                let lbl = self.n_lbl;
                self.n_lbl += 1;
                self.emit(&format!("setp.ne.s64 {p}, {c}, 0;"));
                self.emit(&format!("@!{p} bra ELSE{lbl};"));
                self.emit_edge(then_blk.0, then_args);
                self.emit(&format!("bra BB{};", then_blk.0));
                self.body.push_str(&format!("ELSE{lbl}:\n"));
                self.emit_edge(else_blk.0, else_args);
                self.emit(&format!("bra BB{};", else_blk.0));
            }
            Terminator::Unreachable => self.emit("trap;"),
        }
        Ok(())
    }

    /// Realize the block-param convention for one edge: copy each arg into a temp, then each temp into
    /// the target block's param register. The temp hop makes permuted / self-referential edges safe.
    fn emit_edge(&mut self, target: u32, args: &[ValueId]) {
        if args.is_empty() {
            return;
        }
        let params: Vec<ValueId> = self.func.blocks[target as usize].params.clone();
        let mut temps = Vec::with_capacity(args.len());
        for a in args {
            let rc = rc_of(self.func.value_type(*a));
            let t = self.fresh(rc);
            let src = self.reg(*a);
            self.emit(&format!("mov.{} {t}, {src};", mov_ty(rc)));
            temps.push((t, rc));
        }
        for (p, (t, rc)) in params.iter().zip(temps) {
            let pr = self.reg(*p);
            self.emit(&format!("mov.{} {pr}, {t};", mov_ty(rc)));
        }
    }

    fn coerce_float(&mut self, v: ValueId, want: RC) -> String {
        let from = rc_of(&self.ty(v));
        let r = self.reg(v);
        if from == want {
            return r;
        }
        match (from, want) {
            (RC::F32, RC::F64) => {
                let d = self.fresh(RC::F64);
                self.emit(&format!("cvt.f64.f32 {d}, {r};"));
                d
            }
            (RC::F64, RC::F32) => {
                let d = self.fresh(RC::F32);
                self.emit(&format!("cvt.rn.f32.f64 {d}, {r};"));
                d
            }
            // An int operand reaching a float op is a lowering bug upstream; pass through.
            _ => r,
        }
    }
}

fn float_suffix(t: &MirType) -> &'static str {
    if matches!(t, MirType::F64) {
        "f64"
    } else {
        "f32"
    }
}

fn int_cc(op: CmpOp) -> &'static str {
    use CmpOp::*;
    match op {
        Eq => "eq.s64",
        Ne => "ne.s64",
        Slt => "lt.s64",
        Sle => "le.s64",
        Sgt => "gt.s64",
        Sge => "ge.s64",
        Ult => "lt.u64",
        Ule => "le.u64",
        Ugt => "gt.u64",
        Uge => "ge.u64",
        _ => unreachable!("float predicate in int_cc"),
    }
}

fn float_cc(op: CmpOp) -> &'static str {
    use CmpOp::*;
    match op {
        Foeq => "eq",
        // The interpreter's `!=` is Rust's (unordered-or-not-equal), so use the unordered form.
        Fone => "neu",
        Folt => "lt",
        Fole => "le",
        Fogt => "gt",
        Foge => "ge",
        _ => unreachable!("int predicate in float_cc"),
    }
}

/// The PTX `.func` header (without the trailing body or `;`), shared by the prototype and definition.
/// Every function takes a hidden `p_ctx` first parameter (the device context buffer) so prints /
/// asserts anywhere in the call graph reach it.
fn fn_signature(f: &Function, idx: usize) -> String {
    let mut s = String::from(".func ");
    if !matches!(f.ret, MirType::Void) {
        let _ = write!(s, "(.param .{} retp) ", abi_ty(&f.ret));
    }
    let _ = write!(s, "mfn_{idx} (.param .b64 p_ctx");
    for (i, p) in f.params.iter().enumerate() {
        let _ = write!(s, ", .param .{} p{i}", abi_ty(f.value_type(*p)));
    }
    s.push(')');
    s
}

/// The launched entry kernel: load the context pointer, call the entry `.func`, and store its return
/// value into the context's exit-code slot (`as_int`-truncated for a float entry).
fn emit_entry_kernel(entry_idx: usize, ret: &MirType) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        ".visible .entry {KERNEL_NAME}(.param .u64 ctxp)\n{{\n"
    ));
    s.push_str("    .reg .b64 %rd<4>;\n");
    match rc_of(ret) {
        RC::F32 if !matches!(ret, MirType::Void) => s.push_str("    .reg .f32 %f<1>;\n"),
        RC::F64 if !matches!(ret, MirType::Void) => s.push_str("    .reg .f64 %fd<1>;\n"),
        _ => {}
    }
    s.push_str("    ld.param.u64 %rd0, [ctxp];\n");
    let has_ret = !matches!(ret, MirType::Void);
    s.push_str("    {\n");
    s.push_str("    .param .b64 _pc;\n");
    if has_ret {
        s.push_str(&format!("    .param .{} _pr;\n", abi_ty(ret)));
    }
    s.push_str("    st.param.b64 [_pc], %rd0;\n");
    if has_ret {
        s.push_str(&format!("    call.uni (_pr), mfn_{entry_idx}, (_pc);\n"));
        match rc_of(ret) {
            RC::Rd => s.push_str("    ld.param.b64 %rd1, [_pr];\n"),
            RC::F32 => {
                s.push_str("    ld.param.f32 %f0, [_pr];\n");
                s.push_str("    cvt.rzi.s64.f32 %rd1, %f0;\n");
            }
            RC::F64 => {
                s.push_str("    ld.param.f64 %fd0, [_pr];\n");
                s.push_str("    cvt.rzi.s64.f64 %rd1, %fd0;\n");
            }
        }
    } else {
        s.push_str(&format!("    call.uni mfn_{entry_idx}, (_pc);\n"));
        s.push_str("    mov.b64 %rd1, 0;\n");
    }
    s.push_str("    }\n");
    s.push_str("    cvta.to.global.u64 %rd2, %rd0;\n");
    s.push_str(&format!("    st.global.u64 [%rd2+{CTX_EXIT_OFF}], %rd1;\n"));
    s.push_str("    ret;\n}\n");
    s
}

// ============================================================================================
// Recognized-kernel device helpers (`mrt_*`)
//
// One `.func` per recognized runtime symbol, reproducing the CPU microkernel's numeric contract as
// a naive single-thread device loop (the host single-thread kernel calls them in place). Op codes
// and formulas match `mercury_runtime` (reduce/gemm/i8gemm/...); the CPU<->GPU tolerance gate covers
// reduction-order differences. Each helper is self-contained (no inter-helper calls) and uses
// function-scoped labels prefixed per kernel so they never collide.
// ============================================================================================

/// A device helper for a recognized runtime call: its PTX `.func` name, return register class, and
/// the full `.func` definition text (emitted once per program when the symbol is called).
struct RtHelper {
    ptx_name: &'static str,
    ret: Option<RC>,
    def: &'static str,
}

/// Map a recognized runtime-symbol name to its device helper, or `None` for general lowering.
fn rt_helper(name: &str) -> Option<RtHelper> {
    let (ptx_name, ret, def): (&'static str, Option<RC>, &'static str) = match name {
        "mercury_sreduce_f32" | "mercury_sreduce_f32_parallel" => {
            ("mrt_sreduce", Some(RC::F32), PTX_SREDUCE)
        }
        "mercury_sgemm_nt" | "mercury_sgemm_nt_parallel" => ("mrt_sgemm_nt", None, PTX_SGEMM_NT),
        "mercury_sgemm" | "mercury_sgemm_parallel" => ("mrt_sgemm", None, PTX_SGEMM),
        "mercury_i8gemm_nt" | "mercury_i8gemm_nt_parallel" => {
            ("mrt_i8gemm_nt", None, PTX_I8GEMM_NT)
        }
        "mercury_norm_f32" | "mercury_norm_f32_parallel" => ("mrt_norm", None, PTX_NORM),
        _ => return None,
    };
    Some(RtHelper { ptx_name, ret, def })
}

/// `mercury_sreduce_f32(x, y, n, op) -> f32`: dot(0)/ssd(1)/sum(2)/sumsq(3)/max(4)/min(5)/maxabs(6).
/// Sequential fold (not the CPU's fixed-chunk tree) — additive ops differ only by reduction order
/// (tolerance-gated); max/min/maxabs are order-independent and exact.
const PTX_SREDUCE: &str = r#".func (.param .f32 _r) mrt_sreduce (.param .b64 px, .param .b64 py, .param .b64 pn, .param .b64 pop)
{
    .reg .b64 %rd<8>;
    .reg .f32 %f<6>;
    .reg .pred %p<6>;
    ld.param.u64 %rd0, [px];
    ld.param.u64 %rd1, [py];
    ld.param.u64 %rd2, [pn];
    ld.param.u64 %rd3, [pop];
    mov.f32 %f0, 0f00000000;
    setp.eq.s64 %p0, %rd3, 4;
    setp.eq.s64 %p1, %rd3, 6;
    or.pred %p0, %p0, %p1;
    @%p0 mov.f32 %f0, 0fFF800000;
    setp.eq.s64 %p1, %rd3, 5;
    @%p1 mov.f32 %f0, 0f7F800000;
    mov.b64 %rd4, 0;
RED_LOOP:
    setp.ge.s64 %p2, %rd4, %rd2;
    @%p2 bra RED_DONE;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd0, %rd5;
    ld.f32 %f1, [%rd6];
    add.s64 %rd7, %rd1, %rd5;
    ld.f32 %f2, [%rd7];
    setp.eq.s64 %p3, %rd3, 0;
    @%p3 fma.rn.f32 %f0, %f1, %f2, %f0;
    setp.eq.s64 %p3, %rd3, 1;
    @%p3 sub.rn.f32 %f3, %f1, %f2;
    @%p3 fma.rn.f32 %f0, %f3, %f3, %f0;
    setp.eq.s64 %p3, %rd3, 2;
    @%p3 add.rn.f32 %f0, %f0, %f1;
    setp.eq.s64 %p3, %rd3, 3;
    @%p3 fma.rn.f32 %f0, %f1, %f1, %f0;
    setp.eq.s64 %p3, %rd3, 4;
    @%p3 max.f32 %f0, %f0, %f1;
    setp.eq.s64 %p3, %rd3, 5;
    @%p3 min.f32 %f0, %f0, %f1;
    setp.eq.s64 %p3, %rd3, 6;
    @%p3 abs.f32 %f4, %f1;
    @%p3 max.f32 %f0, %f0, %f4;
    add.s64 %rd4, %rd4, 1;
    bra RED_LOOP;
RED_DONE:
    st.param.f32 [_r], %f0;
    ret;
}
"#;

/// `mercury_sgemm_nt(a, b, c, m, k, n, beta)`: `C[i*n+j] = sum_p A[i*k+p]*B[j*k+p]`; `beta != 0`
/// accumulates into the existing C (B is row-major `[n,k]`, i.e. transposed — the nn.Linear form).
const PTX_SGEMM_NT: &str = r#".func mrt_sgemm_nt (.param .b64 pa, .param .b64 pb, .param .b64 pc, .param .b64 pm, .param .b64 pk, .param .b64 pn, .param .b64 pbeta)
{
    .reg .b64 %rd<24>;
    .reg .f32 %f<8>;
    .reg .pred %p<6>;
    ld.param.u64 %rd0, [pa];
    ld.param.u64 %rd1, [pb];
    ld.param.u64 %rd2, [pc];
    ld.param.u64 %rd3, [pm];
    ld.param.u64 %rd4, [pk];
    ld.param.u64 %rd5, [pn];
    ld.param.u64 %rd6, [pbeta];
    mov.b64 %rd7, 0;
SNT_LI:
    setp.ge.s64 %p0, %rd7, %rd3;
    @%p0 bra SNT_EI;
    mov.b64 %rd8, 0;
SNT_LJ:
    setp.ge.s64 %p1, %rd8, %rd5;
    @%p1 bra SNT_EJ;
    mov.f32 %f0, 0f00000000;
    mul.lo.s64 %rd9, %rd7, %rd4;
    shl.b64 %rd9, %rd9, 2;
    add.s64 %rd9, %rd0, %rd9;
    mul.lo.s64 %rd10, %rd8, %rd4;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd10, %rd1, %rd10;
    mov.b64 %rd11, 0;
SNT_LL:
    setp.ge.s64 %p2, %rd11, %rd4;
    @%p2 bra SNT_EL;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    add.s64 %rd14, %rd10, %rd12;
    ld.f32 %f2, [%rd14];
    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.s64 %rd11, %rd11, 1;
    bra SNT_LL;
SNT_EL:
    mul.lo.s64 %rd15, %rd7, %rd5;
    add.s64 %rd15, %rd15, %rd8;
    shl.b64 %rd15, %rd15, 2;
    add.s64 %rd15, %rd2, %rd15;
    setp.eq.s64 %p3, %rd6, 0;
    @%p3 bra SNT_ST;
    ld.f32 %f3, [%rd15];
    add.rn.f32 %f0, %f0, %f3;
SNT_ST:
    st.f32 [%rd15], %f0;
    add.s64 %rd8, %rd8, 1;
    bra SNT_LJ;
SNT_EJ:
    add.s64 %rd7, %rd7, 1;
    bra SNT_LI;
SNT_EI:
    ret;
}
"#;

/// `mercury_sgemm(a, b, c, m, k, n, beta)`: `C[i*n+j] = sum_p A[i*k+p]*B[p*n+j]`; `beta != 0`
/// accumulates (both A `[m,k]` and B `[k,n]` row-major).
const PTX_SGEMM: &str = r#".func mrt_sgemm (.param .b64 pa, .param .b64 pb, .param .b64 pc, .param .b64 pm, .param .b64 pk, .param .b64 pn, .param .b64 pbeta)
{
    .reg .b64 %rd<24>;
    .reg .f32 %f<8>;
    .reg .pred %p<6>;
    ld.param.u64 %rd0, [pa];
    ld.param.u64 %rd1, [pb];
    ld.param.u64 %rd2, [pc];
    ld.param.u64 %rd3, [pm];
    ld.param.u64 %rd4, [pk];
    ld.param.u64 %rd5, [pn];
    ld.param.u64 %rd6, [pbeta];
    mov.b64 %rd7, 0;
SM_LI:
    setp.ge.s64 %p0, %rd7, %rd3;
    @%p0 bra SM_EI;
    mov.b64 %rd8, 0;
SM_LJ:
    setp.ge.s64 %p1, %rd8, %rd5;
    @%p1 bra SM_EJ;
    mov.f32 %f0, 0f00000000;
    mul.lo.s64 %rd9, %rd7, %rd4;
    shl.b64 %rd9, %rd9, 2;
    add.s64 %rd9, %rd0, %rd9;
    mov.b64 %rd11, 0;
SM_LL:
    setp.ge.s64 %p2, %rd11, %rd4;
    @%p2 bra SM_EL;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    mul.lo.s64 %rd14, %rd11, %rd5;
    add.s64 %rd14, %rd14, %rd8;
    shl.b64 %rd14, %rd14, 2;
    add.s64 %rd14, %rd1, %rd14;
    ld.f32 %f2, [%rd14];
    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.s64 %rd11, %rd11, 1;
    bra SM_LL;
SM_EL:
    mul.lo.s64 %rd15, %rd7, %rd5;
    add.s64 %rd15, %rd15, %rd8;
    shl.b64 %rd15, %rd15, 2;
    add.s64 %rd15, %rd2, %rd15;
    setp.eq.s64 %p3, %rd6, 0;
    @%p3 bra SM_ST;
    ld.f32 %f3, [%rd15];
    add.rn.f32 %f0, %f0, %f3;
SM_ST:
    st.f32 [%rd15], %f0;
    add.s64 %rd8, %rd8, 1;
    bra SM_LJ;
SM_EJ:
    add.s64 %rd7, %rd7, 1;
    bra SM_LI;
SM_EI:
    ret;
}
"#;

/// `mercury_i8gemm_nt(a, b, c, m, k, n)`: `C[i*n+j] = sum_p (u8 A[i*k+p]) * (i8 B[j*k+p])`, exact
/// i32 (wrapping mod 2^32). A is `u8 [m,k]`, B is `i8 [n,k]`, C is `i32 [m,n]`.
const PTX_I8GEMM_NT: &str = r#".func mrt_i8gemm_nt (.param .b64 pa, .param .b64 pb, .param .b64 pc, .param .b64 pm, .param .b64 pk, .param .b64 pn)
{
    .reg .b64 %rd<24>;
    .reg .b32 %r<8>;
    .reg .pred %p<4>;
    ld.param.u64 %rd0, [pa];
    ld.param.u64 %rd1, [pb];
    ld.param.u64 %rd2, [pc];
    ld.param.u64 %rd3, [pm];
    ld.param.u64 %rd4, [pk];
    ld.param.u64 %rd5, [pn];
    mov.b64 %rd6, 0;
I8_LI:
    setp.ge.s64 %p0, %rd6, %rd3;
    @%p0 bra I8_EI;
    mov.b64 %rd7, 0;
I8_LJ:
    setp.ge.s64 %p1, %rd7, %rd5;
    @%p1 bra I8_EJ;
    mov.b32 %r0, 0;
    mul.lo.s64 %rd8, %rd6, %rd4;
    add.s64 %rd8, %rd0, %rd8;
    mul.lo.s64 %rd9, %rd7, %rd4;
    add.s64 %rd9, %rd1, %rd9;
    mov.b64 %rd10, 0;
I8_LL:
    setp.ge.s64 %p2, %rd10, %rd4;
    @%p2 bra I8_EL;
    add.s64 %rd11, %rd8, %rd10;
    ld.u8 %r1, [%rd11];
    add.s64 %rd12, %rd9, %rd10;
    ld.s8 %r2, [%rd12];
    mad.lo.s32 %r0, %r1, %r2, %r0;
    add.s64 %rd10, %rd10, 1;
    bra I8_LL;
I8_EL:
    mul.lo.s64 %rd13, %rd6, %rd5;
    add.s64 %rd13, %rd13, %rd7;
    shl.b64 %rd13, %rd13, 2;
    add.s64 %rd13, %rd2, %rd13;
    st.u32 [%rd13], %r0;
    add.s64 %rd7, %rd7, 1;
    bra I8_LJ;
I8_EJ:
    add.s64 %rd6, %rd6, 1;
    bra I8_LI;
I8_EI:
    ret;
}
"#;

/// `mercury_norm_f32(x, out, rows, cols, eps_bits, op)`: row-wise softmax(0)/layernorm(1)/rmsnorm(2)
/// over an `[rows, cols]` matrix. `exp` uses the SFU `ex2.approx` (matches the offload path; the
/// CPU oracle's Cephes `exp` differs by <~1e-6, inside the tolerance gate). `eps_bits` is the f32
/// bits of epsilon. Sequential per-row reductions (CPU uses an 8-lane tree) — tolerance-gated.
const PTX_NORM: &str = r#".func mrt_norm (.param .b64 px, .param .b64 pout, .param .b64 prows, .param .b64 pcols, .param .b64 peps, .param .b64 pop)
{
    .reg .b64 %rd<16>;
    .reg .f32 %f<16>;
    .reg .b32 %r<4>;
    .reg .pred %p<6>;
    ld.param.u64 %rd0, [px];
    ld.param.u64 %rd1, [pout];
    ld.param.u64 %rd2, [prows];
    ld.param.u64 %rd3, [pcols];
    ld.param.u64 %rd4, [peps];
    ld.param.u64 %rd5, [pop];
    cvt.u32.u64 %r0, %rd4;
    mov.b32 %f15, %r0;
    mov.b64 %rd6, 0;
NORM_ROW:
    setp.ge.s64 %p0, %rd6, %rd2;
    @%p0 bra NORM_DONE;
    mul.lo.s64 %rd7, %rd6, %rd3;
    shl.b64 %rd8, %rd7, 2;
    add.s64 %rd9, %rd0, %rd8;
    add.s64 %rd10, %rd1, %rd8;
    setp.eq.s64 %p1, %rd5, 0;
    @%p1 bra NORM_SM;
    setp.eq.s64 %p1, %rd5, 1;
    @%p1 bra NORM_LN;
    bra NORM_RMS;
NORM_SM:
    mov.f32 %f0, 0fFF800000;
    mov.b64 %rd11, 0;
SM_MAX:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra SM_MAXE;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    max.f32 %f0, %f0, %f1;
    add.s64 %rd11, %rd11, 1;
    bra SM_MAX;
SM_MAXE:
    mov.f32 %f2, 0f00000000;
    mov.b64 %rd11, 0;
SM_EXP:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra SM_EXPE;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    sub.f32 %f3, %f1, %f0;
    mul.f32 %f3, %f3, 0f3FB8AA3B;
    ex2.approx.f32 %f4, %f3;
    add.s64 %rd14, %rd10, %rd12;
    st.f32 [%rd14], %f4;
    add.f32 %f2, %f2, %f4;
    add.s64 %rd11, %rd11, 1;
    bra SM_EXP;
SM_EXPE:
    mov.f32 %f6, 0f3F800000;
    div.rn.f32 %f5, %f6, %f2;
    mov.b64 %rd11, 0;
SM_NORM:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra NORM_NEXT;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd14, %rd10, %rd12;
    ld.f32 %f4, [%rd14];
    mul.f32 %f4, %f4, %f5;
    st.f32 [%rd14], %f4;
    add.s64 %rd11, %rd11, 1;
    bra SM_NORM;
NORM_LN:
    mov.f32 %f2, 0f00000000;
    mov.b64 %rd11, 0;
LN_S1:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra LN_S1E;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    add.f32 %f2, %f2, %f1;
    add.s64 %rd11, %rd11, 1;
    bra LN_S1;
LN_S1E:
    cvt.rn.f32.s64 %f7, %rd3;
    div.rn.f32 %f8, %f2, %f7;
    mov.f32 %f2, 0f00000000;
    mov.b64 %rd11, 0;
LN_S2:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra LN_S2E;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    sub.f32 %f3, %f1, %f8;
    fma.rn.f32 %f2, %f3, %f3, %f2;
    add.s64 %rd11, %rd11, 1;
    bra LN_S2;
LN_S2E:
    div.rn.f32 %f9, %f2, %f7;
    add.f32 %f9, %f9, %f15;
    sqrt.rn.f32 %f9, %f9;
    mov.f32 %f6, 0f3F800000;
    div.rn.f32 %f10, %f6, %f9;
    mov.b64 %rd11, 0;
LN_S3:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra NORM_NEXT;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    sub.f32 %f3, %f1, %f8;
    mul.f32 %f3, %f3, %f10;
    add.s64 %rd14, %rd10, %rd12;
    st.f32 [%rd14], %f3;
    add.s64 %rd11, %rd11, 1;
    bra LN_S3;
NORM_RMS:
    mov.f32 %f2, 0f00000000;
    mov.b64 %rd11, 0;
RMS_S1:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra RMS_S1E;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    fma.rn.f32 %f2, %f1, %f1, %f2;
    add.s64 %rd11, %rd11, 1;
    bra RMS_S1;
RMS_S1E:
    cvt.rn.f32.s64 %f7, %rd3;
    div.rn.f32 %f9, %f2, %f7;
    add.f32 %f9, %f9, %f15;
    sqrt.rn.f32 %f9, %f9;
    mov.f32 %f6, 0f3F800000;
    div.rn.f32 %f10, %f6, %f9;
    mov.b64 %rd11, 0;
RMS_S3:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra NORM_NEXT;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    mul.f32 %f3, %f1, %f10;
    add.s64 %rd14, %rd10, %rd12;
    st.f32 [%rd14], %f3;
    add.s64 %rd11, %rd11, 1;
    bra RMS_S3;
NORM_NEXT:
    add.s64 %rd6, %rd6, 1;
    bra NORM_ROW;
NORM_DONE:
    ret;
}
"#;

// ============================================================================================
// Device execution
// ============================================================================================

fn run_on_device(
    g: &mut crate::gpu::Gpu,
    ptx: &str,
    ret: &MirType,
) -> Result<(i64, Vec<u8>), String> {
    let _ = ret;
    // A stable per-PTX module key (leaked once per distinct module — bounded by programs compiled in
    // the process) so the shared `Gpu` in-process + cubin caches apply.
    let mut h = DefaultHasher::new();
    ptx.hash(&mut h);
    let key: &'static str = Box::leak(format!("mir_lower_{:016x}", h.finish()).into_boxed_str());

    if std::env::var_os("MERCURY_GPU_DUMP_PTX").is_some() {
        eprintln!("--- gpu-native PTX ---\n{ptx}\n--- end PTX ---");
    }

    let f = g.function(key, ptx, KERNEL_NAME).map_err(|e| {
        // Persist the PTX so a JIT/ptxas error (otherwise an opaque driver code) is debuggable.
        let p = std::env::temp_dir().join(format!("{key}.ptx"));
        let _ = std::fs::write(&p, ptx);
        format!("gpu-native JIT/load failed: {e:?}\n  (PTX written to {})", p.display())
    })?;

    let ctx_len = 4 + 2 * RECORD_CAP as usize;
    let host = vec![0u64; ctx_len];
    let mut ctx_d = g
        .stream
        .memcpy_stod(&host)
        .map_err(|e| format!("gpu-native ctx alloc failed: {e:?}"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = g.stream.launch_builder(&f);
    b.arg(&mut ctx_d);
    unsafe {
        b.launch(cfg)
            .map_err(|e| format!("gpu-native launch failed: {e:?}"))?;
    }
    let out = g
        .stream
        .memcpy_dtov(&ctx_d)
        .map_err(|e| format!("gpu-native readback failed: {e:?}"))?;

    let count = out[CTX_COUNT_OFF as usize / 8];
    let assert_failed = out[CTX_ASSERT_OFF as usize / 8];
    let exit = out[CTX_EXIT_OFF as usize / 8] as i64;

    if assert_failed != 0 {
        return Err("assertion failed".into());
    }
    if count > RECORD_CAP {
        return Err(format!(
            "gpu-native print buffer overflow: {count} records exceed capacity {RECORD_CAP}"
        ));
    }

    let mut stdout = Vec::new();
    let base = CTX_RECORDS_OFF as usize / 8;
    for i in 0..count as usize {
        let tag = out[base + 2 * i];
        let payload = out[base + 2 * i + 1];
        let line = if tag == 1 {
            format!("{}\n", f64::from_bits(payload))
        } else {
            format!("{}\n", payload as i64)
        };
        stdout.extend_from_slice(line.as_bytes());
    }
    Ok((exit, stdout))
}

// ============================================================================================
// Coverage gate: tests/run corpus through the GPU-lowering backend vs the interpreter oracle.
// ============================================================================================
#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Build a program from `.mer` source at `opt` (lex -> parse -> sema -> mir_build -> opt).
    fn build(src: &str, opt: u8) -> Option<(Program, Interner)> {
        use mercury_span::SourceMap;
        let mut sm = SourceMap::new();
        let id = sm.add("gate.mer".to_string(), src.to_string());
        let (tokens, ld) = mercury_lexer::tokenize(sm.source(id), id);
        if ld.iter().any(|d| d.is_error()) {
            return None;
        }
        let mut interner = Interner::new();
        let (module, pd) = mercury_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        if pd.iter().any(|d| d.is_error()) {
            return None;
        }
        let (sema, sd) = mercury_sema::check(&module, &interner);
        if sd.iter().any(|d| d.is_error()) {
            return None;
        }
        let (mut program, md) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        if md.iter().any(|d| d.is_error()) {
            return None;
        }
        mercury_opt::optimize(&mut program, opt);
        Some((program, interner))
    }

    /// Compare one output line numerically (float tolerance) or exactly (everything else).
    fn line_matches(g: &str, c: &str) -> bool {
        if g == c {
            return true;
        }
        match (g.trim().parse::<f64>(), c.trim().parse::<f64>()) {
            (Ok(a), Ok(b)) => {
                let diff = (a - b).abs();
                diff <= 1e-3 || diff <= 1e-2 * b.abs().max(a.abs())
            }
            _ => false,
        }
    }

    fn outputs_match(gpu: &[u8], cpu: &[u8]) -> bool {
        let gs = String::from_utf8_lossy(gpu);
        let cs = String::from_utf8_lossy(cpu);
        let gl: Vec<&str> = gs.lines().collect();
        let cl: Vec<&str> = cs.lines().collect();
        gl.len() == cl.len() && gl.iter().zip(&cl).all(|(a, b)| line_matches(a, b))
    }

    #[test]
    fn run_corpus_matches_interp_oracle() {
        if crate::gpu::gpu().is_none() {
            eprintln!("skip run_corpus_matches_interp_oracle: no CUDA device");
            return;
        }
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/run");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("read tests/run")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "mer").unwrap_or(false))
            .collect();
        files.sort();

        let mut covered = 0usize;
        let mut skipped: Vec<String> = Vec::new();
        let mut mismatches: Vec<String> = Vec::new();

        for path in &files {
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            let src = std::fs::read_to_string(path).unwrap();

            // -O0 and -O3 must both match the interpreter (which gives -O0 == -O3 transitively).
            let mut covered_here = true;
            for opt in [0u8, 3u8] {
                let Some((program, mut interner)) = build(&src, opt) else {
                    covered_here = false;
                    skipped.push(format!("{name}@O{opt} (frontend rejected)"));
                    break;
                };
                let entry = interner.intern("main");
                if program.function(entry).is_none() {
                    covered_here = false;
                    skipped.push(format!("{name} (no main)"));
                    break;
                }
                let cpu = mercury_interp::run_with_output(&program, entry, &interner);
                let gpu = jit_run(&program, entry, &interner);
                match (&cpu, &gpu) {
                    (Err(_), Err(_)) => {} // both error (e.g. assertion failed) — agree.
                    (Ok((ce, co)), Ok((ge, go))) => {
                        if ce != ge || !outputs_match(go, co) {
                            mismatches.push(format!(
                                "{name}@O{opt}: cpu=({ce},{:?}) gpu=({ge},{:?})",
                                String::from_utf8_lossy(co),
                                String::from_utf8_lossy(go)
                            ));
                            covered_here = false;
                            break;
                        }
                    }
                    (Ok(_), Err(e)) => {
                        if e.starts_with(UNSUPPORTED) {
                            covered_here = false;
                            skipped.push(format!("{name}@O{opt}: {e}"));
                        } else {
                            mismatches.push(format!("{name}@O{opt}: gpu errored: {e}"));
                            covered_here = false;
                        }
                        break;
                    }
                    (Err(ce), Ok((ge, _))) => {
                        mismatches.push(format!("{name}@O{opt}: cpu err `{ce}` but gpu ok ({ge})"));
                        covered_here = false;
                        break;
                    }
                }
            }
            if covered_here {
                covered += 1;
            }
        }

        eprintln!(
            "\n=== gpu-native MIR->PTX coverage: {covered}/{} programs match the interp oracle (-O0==-O3) ===",
            files.len()
        );
        if !skipped.is_empty() {
            eprintln!("-- not yet covered ({}):", skipped.len());
            for s in &skipped {
                eprintln!("   {s}");
            }
        }
        assert!(
            mismatches.is_empty(),
            "GPU-lowered programs disagree with the interpreter oracle (miscompiles):\n{}",
            mismatches.join("\n")
        );
        assert!(covered > 0, "no programs covered — pipeline broken");
    }
}
