//! General **MIR -> PTX lowering** (Phase 4) — the lever that turns GPU coverage from "a fixed
//! recognizer kernel menu" into "the whole language."
//!
//! The existing `--backend=gpu` path is an *offloading interpreter*: it tree-walks the program on the
//! CPU and only the recognized GEMM/vmath/norm/reduce calls ever reach the device. This module instead
//! consumes **MIR directly** and emits a **PTX kernel** for the entry function (and a `.func` per other
//! function), then driver-JIT-loads and launches it via the shared [`crate::gpu::Gpu`] harness — so an
//! *arbitrary* Wukong program runs on the GPU. It mirrors the Cranelift backend
//! (`wukong_codegen_cranelift`): same `Backend` seam, same `jit_run(program, entry, interner) ->
//! (exit_code, stdout)` shape, PTX virtual registers instead of CLIF SSA values. (The `Backend` impl
//! below is the *declared* seam only — nothing in the workspace calls `Backend::compile`;
//! `wukong_driver::run_on_gpu_lower` calls [`jit_run`](crate::lower::jit_run) directly off its own
//! `BackendKind::GpuLower`.)
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
//! Recognized ops (matmul / vmath / norm / reduce / `@parallel`) *are* lowered here: `rt_helper` maps
//! each recognized `wukong_*` runtime symbol — and its `_parallel` twin — to an `mrt_*` PTX `.func`
//! reproducing the CPU microkernel's numeric contract as a naive single-thread device loop, and
//! `wukong_parallel_for(n, func_addr F, ctx)` lowers to the single sequential chunk `F(0, n, ctx)`
//! (correct because the outlined body is chunk-decomposable; grid-stride parallelism is a later
//! performance increment, and the *cooperative* forms of these ops are emitted by `lower_call_mega`
//! in mega mode, which [`crate::megakernel`] drives).
//! A symbol with no helper — and an op code the helper does not implement — returns a classifiable
//! `UNSUPPORTED:` error so the coverage gate reports it honestly as not-yet-covered. The headline for
//! this phase is **coverage measured against the interpreter oracle**, never an asserted speed.
//!
//! Correctness gate: a sweep of `tests/run/*.wk` run through this backend matches the interpreter
//! oracle (integer/control-flow programs bit-exact; float programs within the CPU<->GPU tolerance) and
//! `-O0` == `-O3` (see the `#[cfg(test)]` module). The existing offload path and the toolchain-free
//! core stay byte-for-byte unchanged — this path is purely additive.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};

use cudarc::driver::{LaunchConfig, PushKernelArg};

use wukong_backend::{Artifact, Backend};
use wukong_mir::{BinOp, CastKind, CmpOp, Function, MirType, Op, Program, RoundMode, Terminator, ValueId};
use wukong_span::{Interner, Symbol};

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
const KERNEL_NAME: &str = "wukong_kernel";

/// The PTX name of the cooperative-megakernel entry (`megakernel.rs` launches a block of threads on
/// it). Distinct from [`KERNEL_NAME`] so a program can have both lowerings cached side-by-side.
pub const MEGA_KERNEL_NAME: &str = "wukong_mega";

/// Prefix marking a *lowering decline* (an op/construct not yet handled) as opposed to a genuine
/// JIT/launch failure — the coverage gate treats the former as "not yet covered" (skip) and the
/// latter as a hard failure.
pub const UNSUPPORTED: &str = "UNSUPPORTED:";

// ============================================================================================
// Public API (mirrors `wukong_codegen_cranelift`)
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
/// captured stdout — the GPU-lowering counterpart to `wukong_codegen_cranelift::jit_run` and
/// `wukong_interp::run_with_output`.
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

    // Fast path: an eligible program runs as the cooperative megakernel (whole program, one launch,
    // recognized ops cooperative across the block). Anything it can't accelerate returns `Ok(None)`
    // and falls through to the correct single-thread lowering below — the universal reference. The
    // `WUKONG_GPU_NO_MEGA` escape hatch forces the single-thread path (for A/B timing / debugging).
    if std::env::var_os("WUKONG_GPU_NO_MEGA").is_none() {
        match crate::megakernel::try_run(program, entry, interner)? {
            Some(r) => return Ok(r),
            None => {}
        }
    }

    jit_run_single(program, entry, interner)
}

/// The single-thread lowering only — bypasses the cooperative megakernel. This is the universal
/// correctness reference (any program, any control flow) and the A/B baseline the megakernel speedup
/// is measured against same-run. Public so benches can time the two paths back-to-back.
pub fn jit_run_single(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    let entry_fn = program
        .function(entry)
        .ok_or_else(|| format!("no entry function `{}`", interner.resolve(entry)))?;
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
        let mut needs_fmod = false;
        for f in &program.funcs {
            for b in &f.blocks {
                for inst in &b.insts {
                    match &inst.op {
                        Op::Call { func, .. } => {
                            if let Some(h) = rt_helper(interner.resolve(*func)) {
                                if seen.insert(h.ptx_name) {
                                    out.push_str(&h.def);
                                    out.push('\n');
                                }
                            }
                        }
                        // Float `%` lowers to a call to the device fmod helper (below).
                        Op::Bin(BinOp::FRem, _, _) => needs_fmod = true,
                        _ => {}
                    }
                }
            }
        }
        if needs_fmod {
            out.push_str(PTX_FMOD);
            out.push('\n');
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

/// Emit the **cooperative-megakernel** PTX module for `program` with `entry` as the single
/// `.visible .entry wukong_mega` (run by a block of threads — see [`crate::megakernel`]). The entry
/// is lowered SPMD with its frame in a shared `.global` buffer; recognized ops emit a block-wide-tree
/// (`mrt_sreduce_coop`), chunked-cooperative (the serial `mrt_*` over per-thread sub-ranges) or
/// `tid==0`-serial (`mrt_*`) call bracketed by `bar.sync`. Eligibility
/// (`crate::fusion::analyze`) guarantees the entry calls no other user function, so no `mfn_*`
/// bodies are emitted — only the helpers it uses + the kernel. Public so `megakernel.rs` and tooling
/// can JIT / inspect it.
pub fn emit_mega_ptx(program: &Program, entry: Symbol, interner: &Interner) -> Result<String, String> {
    let entry_fn = program
        .function(entry)
        .ok_or_else(|| format!("no entry function `{}`", interner.resolve(entry)))?;
    if !entry_fn.params.is_empty() {
        return Err(format!(
            "{UNSUPPORTED} mega entry `{}` must take no parameters",
            interner.resolve(entry)
        ));
    }

    let mut func_idx: HashMap<Symbol, usize> = HashMap::new();
    for (i, f) in program.funcs.iter().enumerate() {
        func_idx.insert(f.name, i);
    }

    let mut out = String::new();
    out.push_str(".version 7.8\n.target sm_89\n.address_size 64\n\n");

    // Scan the entry for the helpers it needs: cooperative reduce (+ its shared scratch), the serial
    // `mrt_*` defs for not-yet-cooperative recognized ops, and the device fmod for float `%`.
    let mut needs_reduce_coop = false;
    let mut needs_fmod = false;
    let mut seen: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
    let mut serial_defs = String::new();
    for b in &entry_fn.blocks {
        for inst in &b.insts {
            match &inst.op {
                Op::Call { func, .. } => {
                    let nm = interner.resolve(*func);
                    use crate::fusion::{classify_call, CallClass, CoopKind};
                    if matches!(classify_call(nm), CallClass::Coop(CoopKind::Reduce)) {
                        needs_reduce_coop = true;
                    } else if let Some(h) = rt_helper(nm) {
                        if seen.insert(h.ptx_name) {
                            serial_defs.push_str(&h.def);
                            serial_defs.push('\n');
                        }
                    }
                }
                Op::Bin(BinOp::FRem, _, _) => needs_fmod = true,
                _ => {}
            }
        }
    }
    if needs_reduce_coop {
        out.push_str(PTX_MEGA_SMEM);
        out.push_str(PTX_SREDUCE_COOP);
        out.push('\n');
    }
    if needs_fmod {
        out.push_str(PTX_FMOD);
        out.push('\n');
    }
    out.push_str(&serial_defs);

    let body = FnEmit::lower_mega(entry_fn, &func_idx, program, interner)?;
    out.push_str(&body);
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

/// Width in bits of an integer MIR type (mirrors `wukong_interp::int_bits`).
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

/// Per-unit byte stride of a chunked pointer argument (see `FnEmit::emit_chunked_call`): a
/// compile-time constant (elementwise: 4 B f32 / 2 B bf16), or `args[d] * c` bytes — a runtime
/// dimension scaled by an element size (a GEMM C-row is `n` elements * 4 B, an int8 A-row is `k` * 1 B).
#[derive(Clone, Copy)]
enum Stride {
    Const(u64),
    Arg(usize, u64),
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
    n_rs: u32, // 16-bit scratch (`%rs`) for bf16/f16 storage conversions
    n_f: u32,
    n_fd: u32,
    n_p: u32,
    // ValueId -> assigned register name (assigned lazily, once, on first use).
    vreg: HashMap<u32, String>,
    // ValueId -> its N lane registers, for SIMD `<N x T>` values (scalarized per lane).
    vlanes: HashMap<u32, Vec<String>>,
    // ValueId -> its constant integer value (for op-code dispatch of recognized kernel calls).
    const_ints: HashMap<u32, i128>,
    // ValueId(`func_addr F`) -> F, so `wukong_parallel_for(n, func_addr F, ctx)` resolves to a
    // direct call to F at compile time (PTX has no portable function pointers).
    func_addr_of: HashMap<u32, Symbol>,
    // Unique label counter (for cond-branch edge fixups).
    n_lbl: u32,

    // Stack frame: `ValueId(alloca) -> byte offset`, plus the total size and the generic base reg.
    frame_off: HashMap<u32, u64>,
    frame_bytes: u64,
    frame_base: String,
    // The cvta'd-to-global context-buffer base register (print/assert/exit live here).
    ctx_reg: String,

    // --- cooperative-megakernel mode (see `megakernel.rs`) ---
    // When set, this function is lowered as the `.visible .entry` megakernel run by a *block* of
    // threads (SPMD): the frame lives in one shared `.global` buffer (not a per-thread `.local`
    // frame), every `Store` / side-effect is predicated to `tid==0`, and every recognized op is
    // bracketed by `bar.sync` (cooperative across the block, or `@tid0` serial). Default `false`
    // keeps the single-thread `.func` path byte-for-byte unchanged.
    mega: bool,
    // The predicate register holding `threadIdx.x == 0` (only bound in `mega` mode).
    tid0: String,
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
            n_rs: 0,
            n_f: 0,
            n_fd: 0,
            n_p: 0,
            vreg: HashMap::new(),
            vlanes: HashMap::new(),
            const_ints: HashMap::new(),
            func_addr_of: HashMap::new(),
            n_lbl: 0,
            frame_off: HashMap::new(),
            frame_bytes: 0,
            frame_base: String::new(),
            ctx_reg: String::new(),
            mega: false,
            tid0: String::new(),
        };
        e.assign_frame();
        e.translate()?;

        // Assemble: signature, reg/local declarations, then preamble + body.
        let mut out = String::new();
        out.push_str(&fn_signature(func, idx));
        out.push_str("\n{\n");
        e.write_reg_decls(&mut out);
        if e.frame_bytes > 0 {
            let _ = writeln!(out, "    .local .align 8 .b8 __frame[{}];", e.frame_bytes);
        }
        out.push_str(&e.body);
        out.push_str("}\n");
        Ok(out)
    }

    /// Lower `func` as the cooperative-megakernel `.visible .entry` (mega mode). Same per-op emitters
    /// as [`lower`](Self::lower), but the frame is a shared `.global` param (no `.local __frame`),
    /// stores/side-effects are `tid==0`-guarded, and recognized ops are `bar.sync`-bracketed. The
    /// entry takes no MIR params and returns via the context exit-code slot. Returns the `.entry` text.
    fn lower_mega(
        func: &'a Function,
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
            n_rs: 0,
            n_f: 0,
            n_fd: 0,
            n_p: 0,
            vreg: HashMap::new(),
            vlanes: HashMap::new(),
            const_ints: HashMap::new(),
            func_addr_of: HashMap::new(),
            n_lbl: 0,
            frame_off: HashMap::new(),
            frame_bytes: 0,
            frame_base: String::new(),
            ctx_reg: String::new(),
            mega: true,
            tid0: String::new(),
        };
        e.assign_frame();
        e.translate_mega()?;

        let mut out = String::new();
        out.push_str(&format!(
            ".visible .entry {MEGA_KERNEL_NAME}(.param .u64 p_ctx, .param .u64 p_frame)\n{{\n"
        ));
        e.write_reg_decls(&mut out);
        out.push_str(&e.body);
        out.push_str("}\n");
        Ok(out)
    }

    /// Emit the per-class `.reg` count declarations for the assembled function/kernel.
    fn write_reg_decls(&self, out: &mut String) {
        if self.n_rd > 0 {
            let _ = writeln!(out, "    .reg .b64 %rd<{}>;", self.n_rd);
        }
        if self.n_r > 0 {
            let _ = writeln!(out, "    .reg .b32 %r<{}>;", self.n_r);
        }
        if self.n_rs > 0 {
            let _ = writeln!(out, "    .reg .b16 %rs<{}>;", self.n_rs);
        }
        if self.n_f > 0 {
            let _ = writeln!(out, "    .reg .f32 %f<{}>;", self.n_f);
        }
        if self.n_fd > 0 {
            let _ = writeln!(out, "    .reg .f64 %fd<{}>;", self.n_fd);
        }
        if self.n_p > 0 {
            let _ = writeln!(out, "    .reg .pred %p<{}>;", self.n_p);
        }
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

    /// A fresh 16-bit scratch register (`%rs`), for bf16/f16 storage conversions.
    fn fresh_r16(&mut self) -> String {
        let r = format!("%rs{}", self.n_rs);
        self.n_rs += 1;
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

        self.emit_blocks()
    }

    /// Mega preamble (the `.visible .entry` cooperative megakernel): the frame lives in a shared
    /// `.global` buffer passed as `p_frame` (one copy the whole block sub-allocates), `p_ctx` is the
    /// same print/assert/exit record buffer, and `tid0` is the `threadIdx.x == 0` predicate that
    /// guards every store / side effect. The entry takes no MIR params (eligibility enforces this).
    fn translate_mega(&mut self) -> Result<(), String> {
        let ctx_raw = self.fresh(RC::Rd);
        self.ctx_reg = self.fresh(RC::Rd);
        self.emit(&format!("ld.param.u64 {ctx_raw}, [p_ctx];"));
        self.emit(&format!("cvta.to.global.u64 {}, {ctx_raw};", self.ctx_reg));

        if self.frame_bytes > 0 {
            let fr_raw = self.fresh(RC::Rd);
            self.frame_base = self.fresh(RC::Rd);
            self.emit(&format!("ld.param.u64 {fr_raw}, [p_frame];"));
            self.emit(&format!("cvta.to.global.u64 {}, {fr_raw};", self.frame_base));
        }

        // tid0 = (threadIdx.x == 0): the predicate guarding stores + side effects in SPMD mode.
        let tid = self.fresh_r32();
        self.tid0 = self.fresh_pred();
        self.emit(&format!("mov.u32 {tid}, %tid.x;"));
        self.emit(&format!("setp.eq.u32 {}, {tid}, 0;", self.tid0));

        self.emit_blocks()
    }

    /// The shared block-emitting loop used by both the single-thread `.func` path and the mega entry.
    fn emit_blocks(&mut self) -> Result<(), String> {
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

    fn lower_inst(&mut self, inst: &wukong_mir::Inst) -> Result<(), String> {
        // Result-less ops first.
        match &inst.op {
            Op::Store { ptr, value } => return self.lower_store(*ptr, *value),
            Op::Call { func, args } => return self.lower_call(*func, args, inst.result),
            // A raw-AVX2 CPU microkernel call (the general 256-bit loop vectorizer's `VecKernelCall`)
            // has no PTX analog; report it UNSUPPORTED so the coverage gate skips this function.
            Op::VecKernelCall { .. } => {
                return Err(format!(
                    "{UNSUPPORTED} `VecKernelCall` (CPU raw-AVX2 kernel) not lowered to PTX"
                ))
            }
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
                self.const_ints.insert(r.0, *v);
                let d = self.reg(r);
                self.emit(&format!("mov.b64 {d}, {};", const_to_u64(*v, ty)));
            }
            Op::ConstFloat(v, ty) => {
                let d = self.reg(r);
                if matches!(ty, MirType::F64) {
                    self.emit(&format!("mov.f64 {d}, 0d{:016X};", v.to_bits()));
                } else {
                    // A bf16/f16 const IS its grid-rounded value, so materialize it rounded — the
                    // same host-side `wukong_runtime::round_*` the interpreter and Cranelift use at
                    // this exact point. The per-store rounding in `store_into` alone is fragile:
                    // mem2reg/CSE forward the value past its store, so `let b: bf16 = 0.1` reached
                    // the multiply as the raw f32 0.1 at -O3 (and as the bf16 grid value at -O0).
                    let f = match ty {
                        MirType::BF16 => wukong_runtime::round_bf16(*v as f32),
                        MirType::F16 => wukong_runtime::round_f16(*v as f32),
                        _ => *v as f32,
                    };
                    self.emit(&format!("mov.f32 {d}, 0f{:08X};", f.to_bits()));
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
            Op::Load(p, ty) => self.lower_load(*p, ty, r)?,
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
            Op::ExtractLane(..) => {
                return Err(format!(
                    "{UNSUPPORTED} SIMD `ExtractLane` (vectorized MIR) not yet lowered to PTX"
                ))
            }
            Op::Iota(..) => {
                return Err(format!(
                    "{UNSUPPORTED} SIMD `Iota` (vectorized MIR) not yet lowered to PTX"
                ))
            }
            Op::FuncAddr(sym) => {
                // Record the target for the `wukong_parallel_for` special-case; the value itself is
                // never materialized as a device function pointer, so emit no instruction.
                self.func_addr_of.insert(r.0, *sym);
            }
            Op::GlobalAddr(_) => {
                // A `.rodata` string address — strings never occur in GPU-eligible numeric kernels,
                // so decline (skip to the single-thread / CPU path) rather than miscompile.
                return Err(format!(
                    "{UNSUPPORTED} `GlobalAddr` (static string data) not lowered to PTX"
                ));
            }
            Op::Store { .. } | Op::Call { .. } | Op::VecKernelCall { .. } => {
                unreachable!("handled above")
            }
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
                // No PTX frem op, and the `a - trunc(a/b)*b` identity loses precision past the
                // mantissa's integer range (`1e18 % 3` -> 0, not 1). Call the exact device fmod
                // helper instead; compute in f64 (exact for f32 inputs) and narrow the f32 result.
                FRem => {
                    let (af, bf) = if rc_of(rty) == RC::F64 {
                        (a.to_string(), b.to_string())
                    } else {
                        let ta = self.fresh(RC::F64);
                        self.emit(&format!("cvt.f64.f32 {ta}, {a};"));
                        let tb = self.fresh(RC::F64);
                        self.emit(&format!("cvt.f64.f32 {tb}, {b};"));
                        (ta, tb)
                    };
                    let rr = self.fresh(RC::F64);
                    self.emit("{");
                    self.emit(".param .f64 _fa;");
                    self.emit(".param .f64 _fb;");
                    self.emit(".param .f64 _fr;");
                    self.emit(&format!("st.param.f64 [_fa], {af};"));
                    self.emit(&format!("st.param.f64 [_fb], {bf};"));
                    self.emit("call.uni (_fr), mrt_fmod, (_fa, _fb);");
                    self.emit(&format!("ld.param.f64 {rr}, [_fr];"));
                    self.emit("}");
                    if rc_of(rty) == RC::F64 {
                        self.emit(&format!("mov.f64 {d}, {rr};"));
                    } else {
                        self.emit(&format!("cvt.rn.f32.f64 {d}, {rr};"));
                    }
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

    /// Emit a cast `d = (to)a` for one (scalar or lane) value, mirroring `wukong_interp::apply_cast`.
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
                // A bf16/f16 source must round to its grid before widening. bf16/f16 share f32's
                // register, so the per-store rounding is fragile — an optimizer can forward a bf16 op
                // result past its store, leaving an unrounded f32 that this widen would pass through
                // unchanged. Round at the observation boundary too, exactly as
                // `wukong_codegen_cranelift`'s FpExt arm and the interpreter's `apply_cast` do.
                let xr = match from {
                    MirType::BF16 | MirType::F16 => {
                        let h = self.fresh_r16();
                        let t = self.fresh(RC::F32);
                        let c = if matches!(from, MirType::BF16) { "bf16" } else { "f16" };
                        self.emit(&format!("cvt.rn.{c}.f32 {h}, {x};"));
                        self.emit(&format!("cvt.f32.{c} {t}, {h};"));
                        t
                    }
                    _ => x.to_string(),
                };
                match to {
                    // Widen to f64. Source is f64 already (mov) or f32/bf16/f16 held as f32 (cvt up).
                    MirType::F64 => {
                        if rc_of(from) == RC::F64 {
                            self.emit(&format!("mov.f64 {d}, {xr};"));
                        } else {
                            self.emit(&format!("cvt.f64.f32 {d}, {xr};"));
                        }
                    }
                    // bf16/f16 -> f32: loads/casts already widen low-precision floats into an f32
                    // register, so after the grid rounding above the widening is just a move.
                    MirType::F32 => self.emit(&format!("mov.f32 {d}, {xr};")),
                    _ => return Err(format!("{UNSUPPORTED} FpExt to {:?}", to)),
                }
            }
            FpTrunc => {
                // Demote an f64 source to f32 first; the result reg is always f32 (rc_of bf16/f16/f32).
                let src = if rc_of(from) == RC::F64 {
                    let t = self.fresh(RC::F32);
                    self.emit(&format!("cvt.rn.f32.f64 {t}, {x};"));
                    t
                } else {
                    x.to_string()
                };
                match to {
                    MirType::F32 => self.emit(&format!("mov.f32 {d}, {src};")),
                    // Round to the bf16/f16 grid (RNE) and widen back to f32 — matches the
                    // interpreter's round_bf16/round_f16 (tolerance-gated).
                    MirType::BF16 => {
                        let h = self.fresh_r16();
                        self.emit(&format!("cvt.rn.bf16.f32 {h}, {src};"));
                        self.emit(&format!("cvt.f32.bf16 {d}, {h};"));
                    }
                    MirType::F16 => {
                        let h = self.fresh_r16();
                        self.emit(&format!("cvt.rn.f16.f32 {h}, {src};"));
                        self.emit(&format!("cvt.f32.f16 {d}, {h};"));
                    }
                    _ => {
                        return Err(format!("{UNSUPPORTED} FpTrunc to {:?}", to));
                    }
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
        // PTX float->int `cvt` *saturates* to the destination integer type (out-of-range clamps to the
        // type's min/max; NaN -> 0) — exactly the Rust `as` / interpreter / Cranelift `fcvt_to_*_sat`
        // semantics. For a 32/64-bit target the single `cvt` is the whole story.
        if w == 64 {
            self.emit(&format!("cvt.rzi.{s}64.{fsfx} {d}, {x};"));
        } else if w == 32 {
            // Saturating cvt to a 32-bit temp, then sign/zero-extend into the 64-bit holder. `mask_int`
            // is a no-op for i32 (the extend already canonicalizes) but keeps the i1 fallback (`& 1`).
            let t = self.fresh_r32();
            self.emit(&format!("cvt.rzi.{s}32.{fsfx} {t}, {x};"));
            self.emit(&format!("cvt.{s}64.{s}32 {d}, {t};"));
            self.mask_int(d, to);
        } else {
            // Narrow (i8/i16/u8/u16): PTX can only cvt-saturate to i32/u32, so saturate to 32 bits then
            // CLAMP to the narrow type's range and reduce — mirroring Cranelift's `fcvt_to_int_sat`
            // (smin/smax for signed; umin for unsigned, whose lower bound is already 0 from the
            // unsigned saturation). This makes e.g. `300.0 as u8 == 255`, `-300.0 as i8 == -128`,
            // `-1.0 as u8 == 0`, not the mod-2^w truncation the old `mask_int` produced.
            let t = self.fresh_r32();
            self.emit(&format!("cvt.rzi.{s}32.{fsfx} {t}, {x};"));
            if signed {
                let (lo, hi) = if w == 8 { (-128, 127) } else { (-32768, 32767) };
                self.emit(&format!("min.s32 {t}, {t}, {hi};"));
                self.emit(&format!("max.s32 {t}, {t}, {lo};"));
                self.emit(&format!("cvt.s64.s32 {d}, {t};"));
            } else {
                let hi = if w == 8 { 255 } else { 65535 };
                self.emit(&format!("min.u32 {t}, {t}, {hi};"));
                self.emit(&format!("cvt.u64.u32 {d}, {t};"));
                // `cvt.u64.u32` leaves the clamped value ZERO-extended, but every integer register in
                // this backend is kept SIGN-extended to its MIR width (the interpreter's `mask`). Skip
                // this and `250.0 as u8` sits in the register as +250 while `const.i8 250` materializes
                // as -6, so a following `setp.eq.s64` against it is false. The signed branch above is
                // already canonical (`cvt.s64.s32` after the clamp sign-extends).
                self.mask_int(d, to);
            }
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

    fn lower_load(&mut self, p: ValueId, ty: &MirType, res: ValueId) -> Result<(), String> {
        let addr = self.reg(p);
        let d = self.reg(res);
        self.load_into(&d, &addr, ty)
    }

    /// Load one (scalar or lane) value of MIR type `ty` from generic address `addr` into `d`.
    fn load_into(&mut self, d: &str, addr: &str, ty: &MirType) -> Result<(), String> {
        match ty {
            // `i1` occupies ONE byte in memory (`size_of(I1) == 1`, and `Op::Gep` strides bools by 1),
            // exactly like Cranelift's `types::I8` mapping — so it must be accessed with `ld.u8`. A
            // 4-byte access would read three neighbouring bools (or run past the frame) and, on an
            // odd address, faults the device outright with CUDA_ERROR_MISALIGNED_ADDRESS.
            MirType::I1 => {
                let w = self.fresh_r32();
                self.emit(&format!("ld.u8 {w}, [{addr}];"));
                self.emit(&format!("cvt.u64.u32 {d}, {w};"));
                self.mask_int(d, ty);
            }
            MirType::F32 => self.emit(&format!("ld.f32 {d}, [{addr}];")),
            MirType::F64 => self.emit(&format!("ld.f64 {d}, [{addr}];")),
            MirType::I64 | MirType::Ptr => self.emit(&format!("ld.u64 {d}, [{addr}];")),
            // bf16/f16 storage is 2 bytes; load the 16 bits and widen to the f32 register.
            MirType::BF16 => {
                let h = self.fresh_r16();
                self.emit(&format!("ld.u16 {h}, [{addr}];"));
                self.emit(&format!("cvt.f32.bf16 {d}, {h};"));
            }
            MirType::F16 => {
                let h = self.fresh_r16();
                self.emit(&format!("ld.u16 {h}, [{addr}];"));
                self.emit(&format!("cvt.f32.f16 {d}, {h};"));
            }
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
            // i32: load 32 bits, sign-extend to 64.
            MirType::I32 => {
                let w = self.fresh_r32();
                self.emit(&format!("ld.u32 {w}, [{addr}];"));
                self.emit(&format!("cvt.s64.s32 {d}, {w};"));
            }
            // An aggregate/void has no single-register load. Falling through to the 4-byte `ld.u32`
            // would emit *plausible but wrong* PTX (a `load [2 x i32]` would read only its first
            // element) — the contract is to decline so the gate reports it as not-yet-covered.
            MirType::Vec(..) | MirType::Array(..) | MirType::Void => {
                return Err(format!("{UNSUPPORTED} load of aggregate type {ty:?}"))
            }
        }
        Ok(())
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

    /// A store predicate prefix: in mega mode every store targets the *shared* frame, so only
    /// `tid==0` may write it (the SPMD threads otherwise race on identical data and multiply the
    /// atomic print counter); in single-thread mode it is empty (unconditional).
    fn st_guard(&self) -> String {
        if self.mega {
            format!("@{} ", self.tid0)
        } else {
            String::new()
        }
    }

    /// Store one (scalar or lane) value `v` of MIR type `ty` to generic address `addr`.
    fn store_into(&mut self, addr: &str, v: &str, ty: &MirType) -> Result<(), String> {
        let g = self.st_guard();
        match ty {
            MirType::F32 => self.emit(&format!("{g}st.f32 [{addr}], {v};")),
            MirType::F64 => self.emit(&format!("{g}st.f64 [{addr}], {v};")),
            MirType::I64 => self.emit(&format!("{g}st.u64 [{addr}], {v};")),
            // A pointer homed in the shared frame (an inlined function's Tensor/array param slot) is a
            // UNIFORM value across the SPMD threads — it is frame-base / kernel-param derived, never a
            // loaded, per-thread-varying value. It must be stored *unconditionally* (no `@tid0` guard):
            // in a megakernel the scalar glue can `load` that slot on **every** thread and dereference
            // it (e.g. a non-recognized reduction over a Tensor param), so a `tid==0`-only store would
            // leave the zero-initialized slot on threads != 0 and their deref of that null pointer
            // faults (CUDA_ERROR_ILLEGAL_ADDRESS) — even though the result is ultimately discarded.
            // Writing the same pointer from all threads is a benign identical-data race (a pointer slot
            // is homed once, never read-modify-written), so every thread reads the correct base. In the
            // single-thread lowering `st_guard` is empty anyway, so this is a no-op there.
            MirType::Ptr => self.emit(&format!("st.u64 [{addr}], {v};")),
            // bf16/f16 storage is 2 bytes; narrow the f32 value to 16 bits, store the raw u16.
            MirType::BF16 => {
                let h = self.fresh_r16();
                self.emit(&format!("cvt.rn.bf16.f32 {h}, {v};"));
                self.emit(&format!("{g}st.u16 [{addr}], {h};"));
            }
            MirType::F16 => {
                let h = self.fresh_r16();
                self.emit(&format!("cvt.rn.f16.f32 {h}, {v};"));
                self.emit(&format!("{g}st.u16 [{addr}], {h};"));
            }
            // `i1` is a ONE-byte object (see `load_into`): store the low byte, masked to 0/1 so the
            // byte read back is canonical whatever produced the value.
            MirType::I1 => {
                let w = self.fresh_r32();
                self.emit(&format!("cvt.u32.u64 {w}, {v};"));
                self.emit(&format!("and.b32 {w}, {w}, 1;"));
                self.emit(&format!("{g}st.u8 [{addr}], {w};"));
            }
            // Narrow stores take a 32-bit source register (low bits); narrow the 64-bit value first.
            MirType::I8 => {
                let w = self.fresh_r32();
                self.emit(&format!("cvt.u32.u64 {w}, {v};"));
                self.emit(&format!("{g}st.u8 [{addr}], {w};"));
            }
            MirType::I16 => {
                let w = self.fresh_r32();
                self.emit(&format!("cvt.u32.u64 {w}, {v};"));
                self.emit(&format!("{g}st.u16 [{addr}], {w};"));
            }
            MirType::I32 => {
                let w = self.fresh_r32();
                self.emit(&format!("cvt.u32.u64 {w}, {v};"));
                self.emit(&format!("{g}st.u32 [{addr}], {w};"));
            }
            // As in `load_into`: an aggregate/void has no single-register store, and a 4-byte
            // `st.u32` would write only part of it. Decline instead of emitting a partial write.
            MirType::Vec(..) | MirType::Array(..) | MirType::Void => {
                return Err(format!("{UNSUPPORTED} store of aggregate type {ty:?}"))
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
        inst: &wukong_mir::Inst,
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
                    self.load_into(&d, &la, &lane)?;
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
        // Side effects share both paths (their record/flag writes are mega-guarded internally).
        match name.as_str() {
            "print" | "println" => return self.lower_print(args),
            "assert" => return self.lower_assert(args),
            _ => {}
        }
        // Mega mode: bracket every recognized op with `bar.sync` and run it cooperatively (or
        // `tid==0`-serial) over the shared frame. The single-thread dispatch below is left intact.
        if self.mega {
            return self.lower_call_mega(&name, args, result);
        }
        match name.as_str() {
            // The elementwise transcendental kernel dispatches on a compile-time op code; only lower
            // the ops `mrt_vmath` implements (a few inverse fns need an atan polynomial, not yet done).
            // The bf16/f16 variants share the op-switch (and the op gate) — only the input load differs.
            // NOTE: every spelling `rt_helper` maps to `mrt_vmath*` must be listed here, or it falls
            // through to the generic `other =>` arm below and reaches the helper with this op-code
            // gate BYPASSED — an unimplemented op would then silently run `mrt_vmath`'s identity
            // default instead of declining.
            "wukong_vmath_f32" | "wukong_vmath_f32_parallel" | "wukong_vmath_bf16"
            | "wukong_vmath_f16"
                if args.len() == 4 =>
            {
                let op = self
                    .const_ints
                    .get(&args[3].0)
                    .copied()
                    .ok_or_else(|| format!("{UNSUPPORTED} vmath op code is not a constant"))?;
                if vmath_supported(op) {
                    let pname = match name.as_str() {
                        "wukong_vmath_bf16" => "mrt_vmath_bf16",
                        "wukong_vmath_f16" => "mrt_vmath_f16",
                        _ => "mrt_vmath",
                    };
                    self.emit_helper_call(pname, None, args, None);
                    Ok(())
                } else {
                    Err(format!("{UNSUPPORTED} vmath op {op} not yet lowered to PTX"))
                }
            }
            // Two-arg transcendentals: pow(0)/atan2(1)/hypot(2), dispatched on a compile-time op code.
            "wukong_vmath2_f32" if args.len() == 5 => {
                let op = self
                    .const_ints
                    .get(&args[4].0)
                    .copied()
                    .ok_or_else(|| format!("{UNSUPPORTED} vmath2 op code is not a constant"))?;
                if matches!(op, 0..=2) {
                    self.emit_helper_call("mrt_vmath2", None, args, None);
                    Ok(())
                } else {
                    Err(format!("{UNSUPPORTED} vmath2 op {op} not yet lowered to PTX"))
                }
            }
            // `wukong_parallel_for(n, func_addr F, ctx)`: the @parallel outliner splits `[0,n)` into
            // per-core chunks and runs `F(lo, hi, ctx)` on each. The chunking is deterministic and the
            // outlined body is chunk-decomposable, so a single sequential chunk `F(0, n, ctx)` gives
            // the same result as the interpreter (serial == parallel). (Grid-stride parallelism is a
            // later performance increment; this is the correct, general lowering.)
            "wukong_parallel_for" if args.len() == 3 => {
                let sym = self.func_addr_of.get(&args[1].0).copied().ok_or_else(|| {
                    format!("{UNSUPPORTED} wukong_parallel_for target is not a func_addr")
                })?;
                let idx = *self.func_idx.get(&sym).ok_or_else(|| {
                    format!(
                        "{UNSUPPORTED} wukong_parallel_for target `{}` is not a known function",
                        self.interner.resolve(sym)
                    )
                })?;
                let zero = self.fresh(RC::Rd);
                self.emit(&format!("mov.b64 {zero}, 0;"));
                let n_reg = self.reg(args[0]);
                let ctx_reg = self.reg(args[2]);
                // F(lo=0, hi=n, ctx); all three args are 64-bit (i64 count / pointers).
                let arg_regs = vec![(zero, "b64"), (n_reg, "b64"), (ctx_reg, "b64")];
                self.emit_call_to(idx, &arg_regs, None);
                Ok(())
            }
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

    /// Mega-mode dispatch of one recognized op: bracket it with `bar.sync` (so the cooperating
    /// threads see the prior `tid==0` setup and the next `tid==0` read sees this op's output), and run
    /// it one of three ways — a block-wide *tree* (`Reduce`, result broadcast to every thread),
    /// **chunked-cooperative** over disjoint per-thread sub-ranges of the serial helper
    /// (`Vmath`/`Vmath2`/`Velem` by elements, `Gemm*`/`I8GemmNt` by output rows, `Norm`/`NormAffine`
    /// by rows — see [`emit_chunked_call`](Self::emit_chunked_call)), or `tid==0`-serial (the
    /// remaining ops, e.g. axpby and the bf16/f16 reductions). An unsupported op code — and
    /// `wukong_parallel_for`, which has no cooperative form — declines with `UNSUPPORTED:` so the
    /// whole program falls back to the single-thread path.
    fn lower_call_mega(
        &mut self,
        name: &str,
        args: &[ValueId],
        result: Option<ValueId>,
    ) -> Result<(), String> {
        use crate::fusion::{classify_call, CallClass, CoopKind};
        let CallClass::Coop(kind) = classify_call(name) else {
            return Err(format!("{UNSUPPORTED} mega call to `{name}` (not a cooperative op)"));
        };
        // Op-code gates, mirroring the single-thread dispatch.
        match kind {
            CoopKind::Vmath if args.len() == 4 => {
                let op = self
                    .const_ints
                    .get(&args[3].0)
                    .copied()
                    .ok_or_else(|| format!("{UNSUPPORTED} vmath op code is not a constant"))?;
                if !vmath_supported(op) {
                    return Err(format!("{UNSUPPORTED} vmath op {op} not yet lowered to PTX"));
                }
            }
            CoopKind::Vmath2 if args.len() == 5 => {
                let op = self
                    .const_ints
                    .get(&args[4].0)
                    .copied()
                    .ok_or_else(|| format!("{UNSUPPORTED} vmath2 op code is not a constant"))?;
                if !matches!(op, 0..=2) {
                    return Err(format!("{UNSUPPORTED} vmath2 op {op} not yet lowered to PTX"));
                }
            }
            CoopKind::ParallelFor => {
                return Err(format!("{UNSUPPORTED} wukong_parallel_for not cooperative in mega"));
            }
            _ => {}
        }

        self.emit("bar.sync 0;");
        match kind {
            // Cooperative reduction across the whole block; the result is broadcast to every thread.
            CoopKind::Reduce => {
                self.emit_helper_call_pred("mrt_sreduce_coop", Some(RC::F32), args, result, false);
            }
            // Elementwise: each thread runs the serial `mrt_*` over a contiguous slice of the `n`
            // elements. `mrt_vmath`(x,out,n,op) / `mrt_vmath2`(a,b,out,n,op) / `mrt_velem`(x,y,out,n,
            // a,b,c,op). The vmath INPUT is f32 (4 B) or bf16/f16 storage (2 B), but its OUTPUT is
            // *always* f32 (4 B) — the activation result is computed and stored in f32 — so the two
            // pointers can have different per-element strides (a 2/4 mismatch was a misaligned store).
            CoopKind::Vmath => {
                // Keyed on the f32 family (`wukong_vmath_f32` and its `_parallel` twin), NOT on the
                // one exact serial spelling: giving an f32 input buffer the bf16 2-byte stride would
                // walk each thread's chunk over the wrong half of the array.
                let in_esz = if name.starts_with("wukong_vmath_f32") { 4 } else { 2 };
                let h = rt_helper(name)
                    .ok_or_else(|| format!("{UNSUPPORTED} no device helper for recognized op `{name}`"))?;
                self.emit_chunked_call(
                    h.ptx_name,
                    h.ret,
                    args,
                    result,
                    2,
                    &[(0, Stride::Const(in_esz)), (1, Stride::Const(4))],
                );
            }
            CoopKind::Vmath2 => {
                let h = rt_helper(name)
                    .ok_or_else(|| format!("{UNSUPPORTED} no device helper for recognized op `{name}`"))?;
                self.emit_chunked_call(
                    h.ptx_name,
                    h.ret,
                    args,
                    result,
                    3,
                    &[(0, Stride::Const(4)), (1, Stride::Const(4)), (2, Stride::Const(4))],
                );
            }
            CoopKind::Velem => {
                let h = rt_helper(name)
                    .ok_or_else(|| format!("{UNSUPPORTED} no device helper for recognized op `{name}`"))?;
                self.emit_chunked_call(
                    h.ptx_name,
                    h.ret,
                    args,
                    result,
                    3,
                    &[(0, Stride::Const(4)), (1, Stride::Const(4)), (2, Stride::Const(4))],
                );
            }
            // GEMM: the output rows are independent, so chunk `m` (rows). C[i,:] = A[i,:]·Bᵀ (+epi):
            // each thread owns a contiguous A-row / C-row block; B / bias / beta / act are shared. A-row
            // = `k` elems, C-row = `n` elems. (a=0, b=1, c=2, m=3, k=4, n=5, ...)
            CoopKind::Gemm | CoopKind::GemmNt | CoopKind::GemmNtEpi => {
                let h = rt_helper(name)
                    .ok_or_else(|| format!("{UNSUPPORTED} no device helper for recognized op `{name}`"))?;
                self.emit_chunked_call(
                    h.ptx_name,
                    h.ret,
                    args,
                    result,
                    3,
                    &[(0, Stride::Arg(4, 4)), (2, Stride::Arg(5, 4))],
                );
            }
            // int8 GEMM: A rows are u8 (`k`*1 B), C rows are i32 (`n`*4 B).
            CoopKind::I8GemmNt => {
                let h = rt_helper(name)
                    .ok_or_else(|| format!("{UNSUPPORTED} no device helper for recognized op `{name}`"))?;
                self.emit_chunked_call(
                    h.ptx_name,
                    h.ret,
                    args,
                    result,
                    3,
                    &[(0, Stride::Arg(4, 1)), (2, Stride::Arg(5, 4))],
                );
            }
            // Row-wise norm: each row's softmax/LayerNorm/RMSNorm is independent -> chunk `rows`.
            // (x=0, out=1, rows=2, cols=3, ...); each row is `cols` f32.
            CoopKind::Norm => {
                let h = rt_helper(name)
                    .ok_or_else(|| format!("{UNSUPPORTED} no device helper for recognized op `{name}`"))?;
                self.emit_chunked_call(
                    h.ptx_name,
                    h.ret,
                    args,
                    result,
                    2,
                    &[(0, Stride::Arg(3, 4)), (1, Stride::Arg(3, 4))],
                );
            }
            // Affine norm: (x=0, out=1, gamma=2, beta=3, rows=4, cols=5, ...); gamma/beta are
            // per-column (shared across rows), so only x/out are row-offset by `cols`*4 B.
            CoopKind::NormAffine => {
                let h = rt_helper(name)
                    .ok_or_else(|| format!("{UNSUPPORTED} no device helper for recognized op `{name}`"))?;
                self.emit_chunked_call(
                    h.ptx_name,
                    h.ret,
                    args,
                    result,
                    4,
                    &[(0, Stride::Arg(5, 4)), (1, Stride::Arg(5, 4))],
                );
            }
            // Serial on `tid==0` over the shared frame (correct; cooperative bodies land later for
            // the remaining ops — e.g. axpby).
            _ => {
                let h = rt_helper(name).ok_or_else(|| {
                    format!("{UNSUPPORTED} no device helper for recognized op `{name}`")
                })?;
                self.emit_helper_call_pred(h.ptx_name, h.ret, args, result, true);
            }
        }
        self.emit("bar.sync 0;");
        Ok(())
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
        self.emit_helper_call_pred(ptx_name, ret, args, result, false);
    }

    /// As [`emit_helper_call`], but when `tid0_guard` the `call.uni` itself is predicated on
    /// `tid==0` — the mega-mode "serial recognized op" path: a single thread runs the naive `mrt_*`
    /// over the shared frame (bracketed by `bar.sync` at the call site), the rest idle. Arg marshaling
    /// and result `ld.param` stay unpredicated (per-thread param space; non-zero threads' result is
    /// discarded by the `tid==0`-guarded store that consumes it).
    fn emit_helper_call_pred(
        &mut self,
        ptx_name: &str,
        ret: Option<RC>,
        args: &[ValueId],
        result: Option<ValueId>,
        tid0_guard: bool,
    ) {
        let arg_regs: Vec<(String, &'static str)> = args
            .iter()
            .map(|a| (self.reg(*a), abi_ty(self.func.value_type(*a))))
            .collect();
        self.emit_helper_call_regs(ptx_name, ret, &arg_regs, result, tid0_guard);
    }

    /// As [`emit_helper_call_pred`] but with pre-built `(reg, abi)` argument pairs — lets the
    /// chunked-cooperative path pass *adjusted* pointer/count registers (each thread's sub-range)
    /// without re-deriving them from `ValueId`s.
    fn emit_helper_call_regs(
        &mut self,
        ptx_name: &str,
        ret: Option<RC>,
        arg_regs: &[(String, &'static str)],
        result: Option<ValueId>,
        tid0_guard: bool,
    ) {
        let g = if tid0_guard {
            format!("@{} ", self.tid0)
        } else {
            String::new()
        };
        self.emit("{");
        for (i, (_, abi)) in arg_regs.iter().enumerate() {
            self.emit(&format!(".param .{abi} _a{i};"));
        }
        if let Some(rc) = ret {
            self.emit(&format!(".param .{} _r;", mov_ty(rc)));
        }
        for (i, (r, abi)) in arg_regs.iter().enumerate() {
            self.emit(&format!("st.param.{abi} [_a{i}], {r};"));
        }
        let mut arglist = String::new();
        for i in 0..arg_regs.len() {
            if i > 0 {
                arglist.push_str(", ");
            }
            arglist.push_str(&format!("_a{i}"));
        }
        match ret {
            Some(_) => self.emit(&format!("{g}call.uni (_r), {ptx_name}, ({arglist});")),
            None => self.emit(&format!("{g}call.uni {ptx_name}, ({arglist});")),
        }
        if let (Some(rc), Some(res)) = (ret, result) {
            let d = self.reg(res);
            self.emit(&format!("ld.param.{} {d}, [_r];", mov_ty(rc)));
        }
        self.emit("}");
    }

    /// Emit a **chunked-cooperative** call of a recognized op: partition `[0, count)` (the arg at
    /// `count_idx` — elements for elementwise, rows for GEMM/norm) into contiguous per-thread chunks,
    /// offset each pointer arg in `ptr_strides` (`(arg_idx, Stride)`) by `lo*stride`, and have every
    /// thread call the *existing serial* `mrt_*` over its own disjoint sub-range. No new kernel, no
    /// internal barrier (chunks are disjoint), no races; the caller's `bar.sync` bracket orders it
    /// against neighbours. This is the GPU analogue of `wukong_runtime::parallel_for`'s fixed
    /// chunking, so it is deterministic and serial==parallel.
    fn emit_chunked_call(
        &mut self,
        ptx_name: &str,
        ret: Option<RC>,
        args: &[ValueId],
        result: Option<ValueId>,
        count_idx: usize,
        ptr_strides: &[(usize, Stride)],
    ) {
        // tid / ntid as 64-bit.
        let tid32 = self.fresh_r32();
        let ntid32 = self.fresh_r32();
        self.emit(&format!("mov.u32 {tid32}, %tid.x;"));
        self.emit(&format!("mov.u32 {ntid32}, %ntid.x;"));
        let tid = self.fresh(RC::Rd);
        let ntid = self.fresh(RC::Rd);
        self.emit(&format!("cvt.u64.u32 {tid}, {tid32};"));
        self.emit(&format!("cvt.u64.u32 {ntid}, {ntid32};"));

        let count = self.reg(args[count_idx]);
        // chunk = (count + ntid - 1) / ntid   (ceil; count>0, ntid>0)
        let chunk = self.fresh(RC::Rd);
        self.emit(&format!("add.s64 {chunk}, {count}, {ntid};"));
        self.emit(&format!("sub.s64 {chunk}, {chunk}, 1;"));
        self.emit(&format!("div.u64 {chunk}, {chunk}, {ntid};"));
        // lo = tid*chunk ; hi = min(lo+chunk, count) ; cnt = max(hi-lo, 0)
        let lo = self.fresh(RC::Rd);
        self.emit(&format!("mul.lo.s64 {lo}, {tid}, {chunk};"));
        let hi = self.fresh(RC::Rd);
        self.emit(&format!("add.s64 {hi}, {lo}, {chunk};"));
        self.emit(&format!("min.s64 {hi}, {hi}, {count};"));
        let cnt = self.fresh(RC::Rd);
        self.emit(&format!("sub.s64 {cnt}, {hi}, {lo};"));
        self.emit(&format!("max.s64 {cnt}, {cnt}, 0;"));

        // Build the adjusted argument registers: pointers offset by lo*stride, count replaced by cnt.
        let mut arg_regs: Vec<(String, &'static str)> = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            if i == count_idx {
                arg_regs.push((cnt.clone(), "b64"));
            } else if let Some((_, stride)) = ptr_strides.iter().find(|(idx, _)| *idx == i) {
                // Per-unit byte stride: a constant, or a runtime dim (`args[d]`) times a constant
                // (a GEMM row = `k` elements * 4 B, an int8-GEMM A-row = `k` * 1 B, ...).
                let stride_bytes = match *stride {
                    Stride::Const(c) => {
                        let s = self.fresh(RC::Rd);
                        self.emit(&format!("mov.b64 {s}, {c};"));
                        s
                    }
                    Stride::Arg(d, c) => {
                        let dim = self.reg(args[d]);
                        let s = self.fresh(RC::Rd);
                        self.emit(&format!("mul.lo.s64 {s}, {dim}, {c};"));
                        s
                    }
                };
                let base = self.reg(*a);
                let off = self.fresh(RC::Rd);
                let np = self.fresh(RC::Rd);
                self.emit(&format!("mul.lo.s64 {off}, {lo}, {stride_bytes};"));
                self.emit(&format!("add.s64 {np}, {base}, {off};"));
                arg_regs.push((np, "b64"));
            } else {
                arg_regs.push((self.reg(*a), abi_ty(self.func.value_type(*a))));
            }
        }
        self.emit_helper_call_regs(ptx_name, ret, &arg_regs, result, false);
    }

    fn lower_user_call(
        &mut self,
        idx: usize,
        args: &[ValueId],
        result: Option<ValueId>,
    ) -> Result<(), String> {
        // Snapshot arg regs/types before mutating the builder.
        let arg_regs: Vec<(String, &'static str)> = args
            .iter()
            .map(|a| (self.reg(*a), abi_ty(self.func.value_type(*a))))
            .collect();
        self.emit_call_to(idx, &arg_regs, result);
        Ok(())
    }

    /// Emit a call to user function `idx` with pre-built `(reg, abi)` arg pairs. Every `mfn_*` takes
    /// the device context buffer as an implicit leading param (for print/assert/exit), followed by
    /// the explicit args; the return (if any) is bound into `result`.
    fn emit_call_to(
        &mut self,
        idx: usize,
        arg_regs: &[(String, &'static str)],
        result: Option<ValueId>,
    ) {
        let callee = &self.program.funcs[idx];
        let ret = callee.ret.clone();
        let has_ret = !matches!(ret, MirType::Void);
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
    /// In mega mode the whole append is `tid==0`-only: the SPMD threads all compute the *same* record
    /// (control flow is data-independent), so emitting once preserves byte-identical stdout — and the
    /// atomic counter must advance exactly once, not once per thread.
    fn emit_record(&mut self, tag: u64, payload: &str) {
        let ctx = self.ctx_reg.clone();
        let idx = self.fresh(RC::Rd);
        let off = self.fresh(RC::Rd);
        let addr = self.fresh(RC::Rd);
        let tagr = self.fresh(RC::Rd);
        let pw = self.fresh_pred();
        let ag = self.st_guard(); // `@tid0 ` (mega) or empty
        self.emit(&format!("{ag}atom.global.add.u64 {idx}, [{ctx}+{CTX_COUNT_OFF}], 1;"));
        self.emit(&format!("setp.lt.u64 {pw}, {idx}, {RECORD_CAP};"));
        if self.mega {
            // Fold tid0 into the bounds predicate so non-zero threads (whose `idx` is undefined,
            // having skipped the guarded atomic) never store.
            self.emit(&format!("and.pred {pw}, {pw}, {};", self.tid0));
        }
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
        if self.mega {
            // Only `tid==0` records the (identical, data-independent) assert outcome.
            self.emit(&format!("and.pred {pz}, {pz}, {};", self.tid0));
        }
        let one = self.fresh(RC::Rd);
        self.emit(&format!("mov.b64 {one}, 1;"));
        self.emit(&format!("@{pz} st.global.u64 [{ctx}+{CTX_ASSERT_OFF}], {one};"));
        Ok(())
    }

    // --- terminators ---

    fn lower_term(&mut self, term: &Terminator) -> Result<(), String> {
        match term {
            // In mega mode the entry *is* the kernel: a `Ret` writes the exit code into the context
            // exit slot (tid==0) and returns, mirroring `emit_entry_kernel`'s wrapper.
            Terminator::Ret(None) if self.mega => {
                self.emit_mega_exit(None);
                self.emit("ret;");
            }
            Terminator::Ret(Some(v)) if self.mega => {
                let v = *v;
                self.emit_mega_exit(Some(v));
                self.emit("ret;");
            }
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

    /// Store the megakernel exit code (the entry's return, truncated to i64) into the context exit
    /// slot, `tid==0`-only. A void/`None` return exits 0 (matching the single-thread wrapper).
    fn emit_mega_exit(&mut self, v: Option<ValueId>) {
        let ctx = self.ctx_reg.clone();
        let tid0 = self.tid0.clone();
        let code = self.fresh(RC::Rd);
        match v {
            None => self.emit(&format!("mov.b64 {code}, 0;")),
            Some(v) => {
                let rc = rc_of(&self.ty(v));
                let r = self.reg(v);
                match rc {
                    RC::Rd => self.emit(&format!("mov.b64 {code}, {r};")),
                    RC::F32 => self.emit(&format!("cvt.rzi.s64.f32 {code}, {r};")),
                    RC::F64 => self.emit(&format!("cvt.rzi.s64.f64 {code}, {r};")),
                }
            }
        }
        self.emit(&format!("@{tid0} st.global.u64 [{ctx}+{CTX_EXIT_OFF}], {code};"));
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
// and formulas match `wukong_runtime` (reduce/gemm/i8gemm/...); the CPU<->GPU tolerance gate covers
// reduction-order differences. Each helper is self-contained (no inter-helper calls) and uses
// function-scoped labels prefixed per kernel so they never collide.
// ============================================================================================

/// A device helper for a recognized runtime call: its PTX `.func` name, return register class, and
/// the full `.func` definition text (emitted once per program when the symbol is called). `def` is
/// owned because the mixed-precision helpers are generated per (op, precision).
struct RtHelper {
    ptx_name: &'static str,
    ret: Option<RC>,
    def: String,
}

/// Map a recognized runtime-symbol name to its device helper, or `None` for general lowering.
fn rt_helper(name: &str) -> Option<RtHelper> {
    let (ptx_name, ret, def): (&'static str, Option<RC>, String) = match name {
        "wukong_sreduce_f32" | "wukong_sreduce_f32_parallel" => {
            ("mrt_sreduce", Some(RC::F32), PTX_SREDUCE.to_string())
        }
        "wukong_sgemm_nt" | "wukong_sgemm_nt_parallel" => {
            ("mrt_sgemm_nt", None, PTX_SGEMM_NT.to_string())
        }
        "wukong_sgemm" | "wukong_sgemm_parallel" => ("mrt_sgemm", None, PTX_SGEMM.to_string()),
        "wukong_i8gemm_nt" | "wukong_i8gemm_nt_parallel" => {
            ("mrt_i8gemm_nt", None, PTX_I8GEMM_NT.to_string())
        }
        "wukong_norm_f32" | "wukong_norm_f32_parallel" => ("mrt_norm", None, PTX_NORM.to_string()),
        "wukong_norm_affine_f32" | "wukong_norm_affine_f32_parallel" => {
            ("mrt_norm_affine", None, PTX_NORM_AFFINE.to_string())
        }
        "wukong_vmath_f32" | "wukong_vmath_f32_parallel" => {
            ("mrt_vmath", None, PTX_VMATH.to_string())
        }
        "wukong_sgemm_nt_epi" | "wukong_sgemm_nt_epi_parallel" => {
            ("mrt_sgemm_nt_epi", None, PTX_SGEMM_NT_EPI.to_string())
        }
        // Mixed-precision (bf16/f16 storage, f32 compute) — generated per precision.
        "wukong_vmath_bf16" => ("mrt_vmath_bf16", None, ptx_vmath_lowp("mrt_vmath_bf16", "bf16")?),
        "wukong_vmath_f16" => ("mrt_vmath_f16", None, ptx_vmath_lowp("mrt_vmath_f16", "f16")?),
        "wukong_dot_bf16" | "wukong_dot_bf16_parallel" => {
            ("mrt_dot_bf16", Some(RC::F32), ptx_dot_lowp("mrt_dot_bf16", "bf16"))
        }
        "wukong_dot_f16" | "wukong_dot_f16_parallel" => {
            ("mrt_dot_f16", Some(RC::F32), ptx_dot_lowp("mrt_dot_f16", "f16"))
        }
        "wukong_sum_bf16" | "wukong_sum_bf16_parallel" => {
            ("mrt_sum_bf16", Some(RC::F32), ptx_sum_lowp("mrt_sum_bf16", "bf16"))
        }
        "wukong_sum_f16" | "wukong_sum_f16_parallel" => {
            ("mrt_sum_f16", Some(RC::F32), ptx_sum_lowp("mrt_sum_f16", "f16"))
        }
        "wukong_reduce_bf16" | "wukong_reduce_bf16_parallel" => (
            "mrt_reduce_bf16",
            Some(RC::F32),
            ptx_reduce_lowp("mrt_reduce_bf16", "bf16"),
        ),
        "wukong_reduce_f16" | "wukong_reduce_f16_parallel" => (
            "mrt_reduce_f16",
            Some(RC::F32),
            ptx_reduce_lowp("mrt_reduce_f16", "f16"),
        ),
        "wukong_axpby_bf16" => (
            "mrt_axpby_bf16",
            None,
            ptx_axpby_lowp("mrt_axpby_bf16", "bf16"),
        ),
        "wukong_axpby_f16" => ("mrt_axpby_f16", None, ptx_axpby_lowp("mrt_axpby_f16", "f16")),
        // Two-arg transcendentals (pow/atan2/hypot), op-gated in lower_call.
        "wukong_vmath2_f32" => ("mrt_vmath2", None, PTX_VMATH2.to_string()),
        // Streaming elementwise act(a*x + b*y + c) (residual add, fused bias/act).
        "wukong_velem_f32" | "wukong_velem_f32_parallel" => {
            ("mrt_velem", None, PTX_VELEM.to_string())
        }
        _ => return None,
    };
    Some(RtHelper { ptx_name, ret, def })
}

/// Which `wukong_vmath_f32` op codes `mrt_vmath` implements: the full single-arg set 0..=35
/// (atan(25)/asin(33)/acos(34) use an inline minimax `atan` poly since PTX has no SFU for them).
fn vmath_supported(op: i128) -> bool {
    matches!(op, 0..=35)
}

/// `wukong_sreduce_f32(x, y, n, op) -> f32`: dot(0)/ssd(1)/sum(2)/sumsq(3)/max(4)/min(5)/maxabs(6)/
/// sumabs(9, Σ|x|)/absdiff(10, Σ|x−y|). (Codes 7/8 are the arg-reductions — a separate `-> i64` ABI —
/// so they never reach here.) Sequential fold (not the CPU's fixed-chunk tree) — additive ops differ
/// only by reduction order (tolerance-gated); max/min/maxabs are order-independent and exact.
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
    setp.eq.s64 %p3, %rd3, 9;
    @%p3 abs.f32 %f4, %f1;
    @%p3 add.rn.f32 %f0, %f0, %f4;
    setp.eq.s64 %p3, %rd3, 10;
    @%p3 sub.rn.f32 %f5, %f1, %f2;
    @%p3 abs.f32 %f5, %f5;
    @%p3 add.rn.f32 %f0, %f0, %f5;
    add.s64 %rd4, %rd4, 1;
    bra RED_LOOP;
RED_DONE:
    st.param.f32 [_r], %f0;
    ret;
}
"#;

/// Block-shared scratch for the cooperative reduction tree (`mrt_sreduce_coop`). One f32 per thread;
/// 1024 covers the largest block the megakernel launcher uses. Static `.shared` (4 KiB), so the
/// launch needs no dynamic shared memory.
const PTX_MEGA_SMEM: &str = ".shared .align 4 .b32 mrt_red_smem[1024];\n";

/// **Cooperative** `wukong_sreduce_f32(x, y, n, op) -> f32` for the megakernel: every thread in the
/// block folds a `tid`-strided slice of `[0,n)` into a private partial (the *same* per-element op as
/// the single-thread `mrt_sreduce`), then a fixed shared-memory **tree** combines the partials and
/// broadcasts the result to all threads. Deterministic (no atomics, M12); the tree's reassociation vs
/// the sequential fold is covered by the CPU<->GPU tolerance gate. Block size must be a power of two
/// (the launcher uses 256). Combine: add for the additive ops dot/ssd/sum/sumsq (op<4) *and*
/// sumabs/absdiff (op>=9), max for max/maxabs, min for min.
const PTX_SREDUCE_COOP: &str = r#".func (.param .f32 _r) mrt_sreduce_coop (.param .b64 px, .param .b64 py, .param .b64 pn, .param .b64 pop)
{
    .reg .b64 %rd<12>;
    .reg .b32 %r<10>;
    .reg .f32 %f<8>;
    .reg .pred %p<8>;
    ld.param.u64 %rd0, [px];
    ld.param.u64 %rd1, [py];
    ld.param.u64 %rd2, [pn];
    ld.param.u64 %rd3, [pop];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ntid.x;
    cvt.u64.u32 %rd4, %r0;
    cvt.u64.u32 %rd5, %r1;
    mov.f32 %f0, 0f00000000;
    setp.eq.s64 %p0, %rd3, 4;
    setp.eq.s64 %p1, %rd3, 6;
    or.pred %p0, %p0, %p1;
    @%p0 mov.f32 %f0, 0fFF800000;
    setp.eq.s64 %p1, %rd3, 5;
    @%p1 mov.f32 %f0, 0f7F800000;
RC_LOOP:
    setp.ge.s64 %p2, %rd4, %rd2;
    @%p2 bra RC_DONE;
    shl.b64 %rd6, %rd4, 2;
    add.s64 %rd7, %rd0, %rd6;
    ld.f32 %f1, [%rd7];
    add.s64 %rd8, %rd1, %rd6;
    ld.f32 %f2, [%rd8];
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
    setp.eq.s64 %p3, %rd3, 9;
    @%p3 abs.f32 %f4, %f1;
    @%p3 add.rn.f32 %f0, %f0, %f4;
    setp.eq.s64 %p3, %rd3, 10;
    @%p3 sub.rn.f32 %f4, %f1, %f2;
    @%p3 abs.f32 %f4, %f4;
    @%p3 add.rn.f32 %f0, %f0, %f4;
    add.s64 %rd4, %rd4, %rd5;
    bra RC_LOOP;
RC_DONE:
    shl.b32 %r2, %r0, 2;
    mov.u32 %r3, mrt_red_smem;
    add.s32 %r4, %r3, %r2;
    st.shared.f32 [%r4], %f0;
    bar.sync 0;
    shr.u32 %r5, %r1, 1;
RC_TREE:
    setp.eq.s32 %p4, %r5, 0;
    @%p4 bra RC_TREE_DONE;
    setp.lt.u32 %p5, %r0, %r5;
    @!%p5 bra RC_TREE_SYNC;
    ld.shared.f32 %f5, [%r4];
    add.u32 %r6, %r0, %r5;
    shl.b32 %r6, %r6, 2;
    add.u32 %r7, %r3, %r6;
    ld.shared.f32 %f6, [%r7];
    setp.eq.s64 %p6, %rd3, 4;
    setp.eq.s64 %p7, %rd3, 6;
    or.pred %p6, %p6, %p7;
    @%p6 max.f32 %f5, %f5, %f6;
    setp.eq.s64 %p7, %rd3, 5;
    @%p7 min.f32 %f5, %f5, %f6;
    setp.lt.s64 %p6, %rd3, 4;
    setp.ge.s64 %p7, %rd3, 9;
    or.pred %p6, %p6, %p7;
    @%p6 add.rn.f32 %f5, %f5, %f6;
    st.shared.f32 [%r4], %f5;
RC_TREE_SYNC:
    bar.sync 0;
    shr.u32 %r5, %r5, 1;
    bra RC_TREE;
RC_TREE_DONE:
    mov.u32 %r8, mrt_red_smem;
    ld.shared.f32 %f0, [%r8];
    st.param.f32 [_r], %f0;
    ret;
}
"#;

/// `wukong_sgemm_nt(a, b, c, m, k, n, beta)`: `C[i*n+j] = sum_p A[i*k+p]*B[j*k+p]`; `beta != 0`
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

/// `wukong_sgemm(a, b, c, m, k, n, beta)`: `C[i*n+j] = sum_p A[i*k+p]*B[p*n+j]`; `beta != 0`
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

/// `wukong_i8gemm_nt(a, b, c, m, k, n)`: `C[i*n+j] = sum_p (u8 A[i*k+p]) * (i8 B[j*k+p])`, exact
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

/// `wukong_norm_f32(x, out, rows, cols, eps_bits, op)`: row-wise softmax(0)/layernorm(1)/rmsnorm(2)/
/// log-softmax(3)/l2norm(4) over an `[rows, cols]` matrix. `exp` uses the SFU `ex2.approx` and `log`
/// uses `lg2.approx·ln2` (both match the offload path; the CPU oracle's Cephes `exp`/`log` differ by
/// <~1e-6, inside the tolerance gate). `eps_bits` is the f32 bits of epsilon (used by layernorm/
/// rmsnorm/l2norm; softmax/log-softmax ignore it). Sequential per-row reductions (CPU uses an 8-lane
/// tree) — tolerance-gated. Ops 0..=3 branch explicitly and 4 (l2norm) is the *default* arm, so no op
/// falls into the wrong body — but the arm is a bare `bra`, not an equality test: extending the
/// recognizer past code 4 must add a branch here (and an op gate in `lower_call`, which currently
/// gates only vmath/vmath2) or the new code would silently run l2norm.
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
    setp.eq.s64 %p1, %rd5, 2;
    @%p1 bra NORM_RMS;
    setp.eq.s64 %p1, %rd5, 3;
    @%p1 bra NORM_LS;
    bra NORM_L2;
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
NORM_LS:
    mov.f32 %f0, 0fFF800000;
    mov.b64 %rd11, 0;
LS_MAX:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra LS_MAXE;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    max.f32 %f0, %f0, %f1;
    add.s64 %rd11, %rd11, 1;
    bra LS_MAX;
LS_MAXE:
    mov.f32 %f2, 0f00000000;
    mov.b64 %rd11, 0;
LS_EXP:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra LS_EXPE;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    sub.f32 %f3, %f1, %f0;
    mul.f32 %f3, %f3, 0f3FB8AA3B;
    ex2.approx.f32 %f4, %f3;
    add.f32 %f2, %f2, %f4;
    add.s64 %rd11, %rd11, 1;
    bra LS_EXP;
LS_EXPE:
    lg2.approx.f32 %f5, %f2;
    mul.f32 %f5, %f5, 0f3F317218;
    add.f32 %f6, %f0, %f5;
    mov.b64 %rd11, 0;
LS_OUT:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra NORM_NEXT;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    sub.f32 %f3, %f1, %f6;
    add.s64 %rd14, %rd10, %rd12;
    st.f32 [%rd14], %f3;
    add.s64 %rd11, %rd11, 1;
    bra LS_OUT;
NORM_L2:
    mov.f32 %f2, 0f00000000;
    mov.b64 %rd11, 0;
L2_S1:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra L2_S1E;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    fma.rn.f32 %f2, %f1, %f1, %f2;
    add.s64 %rd11, %rd11, 1;
    bra L2_S1;
L2_S1E:
    add.f32 %f9, %f2, %f15;
    sqrt.rn.f32 %f9, %f9;
    mov.f32 %f6, 0f3F800000;
    div.rn.f32 %f10, %f6, %f9;
    mov.b64 %rd11, 0;
L2_S3:
    setp.ge.s64 %p2, %rd11, %rd3;
    @%p2 bra NORM_NEXT;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    mul.f32 %f3, %f1, %f10;
    add.s64 %rd14, %rd10, %rd12;
    st.f32 [%rd14], %f3;
    add.s64 %rd11, %rd11, 1;
    bra L2_S3;
NORM_NEXT:
    add.s64 %rd6, %rd6, 1;
    bra NORM_ROW;
NORM_DONE:
    ret;
}
"#;

/// `wukong_norm_affine_f32(x, out, gamma, beta, rows, cols, eps_bits, op)`: per-row LayerNorm(1)/
/// RMSNorm(2) with an affine writeback `out = norm·g + b`, where `g = gamma[col]` (1.0 if gamma is
/// null) and `b = beta[col]` (0.0 if beta is null) — gamma/beta arrive as integer 0 when absent.
/// (Softmax is never affine, so only op 1/2 reach here.) eps is the low 32 bits of `eps_bits`.
const PTX_NORM_AFFINE: &str = r#".func mrt_norm_affine (.param .b64 px, .param .b64 pout, .param .b64 pgamma, .param .b64 pbeta, .param .b64 prows, .param .b64 pcols, .param .b64 peps, .param .b64 pop)
{
    .reg .b64 %rd<18>;
    .reg .f32 %f<16>;
    .reg .b32 %r<3>;
    .reg .pred %p<6>;
    ld.param.u64 %rd0, [px];
    ld.param.u64 %rd1, [pout];
    ld.param.u64 %rd2, [pgamma];
    ld.param.u64 %rd3, [pbeta];
    ld.param.u64 %rd4, [prows];
    ld.param.u64 %rd5, [pcols];
    ld.param.u64 %rd6, [peps];
    ld.param.u64 %rd7, [pop];
    cvt.u32.u64 %r0, %rd6;
    mov.b32 %f0, %r0;
    cvt.rn.f32.s64 %f1, %rd5;
    mov.b64 %rd8, 0;
NA_ROW:
    setp.ge.s64 %p0, %rd8, %rd4;
    @%p0 bra NA_DONE;
    mul.lo.s64 %rd9, %rd8, %rd5;
    shl.b64 %rd10, %rd9, 2;
    add.s64 %rd11, %rd0, %rd10;
    add.s64 %rd12, %rd1, %rd10;
    setp.eq.s64 %p1, %rd7, 1;
    @%p1 bra NA_LN;
    bra NA_RMS;
NA_LN:
    mov.f32 %f2, 0f00000000; mov.b64 %rd13, 0;
LNA1:
    setp.ge.s64 %p2, %rd13, %rd5; @%p2 bra LNA1E;
    shl.b64 %rd14, %rd13, 2; add.s64 %rd15, %rd11, %rd14;
    ld.f32 %f3, [%rd15]; add.f32 %f2, %f2, %f3;
    add.s64 %rd13, %rd13, 1; bra LNA1;
LNA1E:
    div.rn.f32 %f4, %f2, %f1;
    mov.f32 %f2, 0f00000000; mov.b64 %rd13, 0;
LNA2:
    setp.ge.s64 %p2, %rd13, %rd5; @%p2 bra LNA2E;
    shl.b64 %rd14, %rd13, 2; add.s64 %rd15, %rd11, %rd14;
    ld.f32 %f3, [%rd15]; sub.f32 %f3, %f3, %f4; fma.rn.f32 %f2, %f3, %f3, %f2;
    add.s64 %rd13, %rd13, 1; bra LNA2;
LNA2E:
    div.rn.f32 %f5, %f2, %f1; add.f32 %f5, %f5, %f0; sqrt.rn.f32 %f5, %f5;
    mov.f32 %f6, 0f3F800000; div.rn.f32 %f5, %f6, %f5;
    mov.b64 %rd13, 0;
LNA3:
    setp.ge.s64 %p2, %rd13, %rd5; @%p2 bra NA_NEXT;
    shl.b64 %rd14, %rd13, 2; add.s64 %rd15, %rd11, %rd14;
    ld.f32 %f3, [%rd15]; sub.f32 %f3, %f3, %f4; mul.f32 %f3, %f3, %f5;
    mov.f32 %f7, 0f3F800000; setp.ne.s64 %p3, %rd2, 0;
    @%p3 add.s64 %rd16, %rd2, %rd14; @%p3 ld.f32 %f7, [%rd16];
    mov.f32 %f8, 0f00000000; setp.ne.s64 %p3, %rd3, 0;
    @%p3 add.s64 %rd16, %rd3, %rd14; @%p3 ld.f32 %f8, [%rd16];
    fma.rn.f32 %f3, %f3, %f7, %f8;
    add.s64 %rd16, %rd12, %rd14; st.f32 [%rd16], %f3;
    add.s64 %rd13, %rd13, 1; bra LNA3;
NA_RMS:
    mov.f32 %f2, 0f00000000; mov.b64 %rd13, 0;
RMA1:
    setp.ge.s64 %p2, %rd13, %rd5; @%p2 bra RMA1E;
    shl.b64 %rd14, %rd13, 2; add.s64 %rd15, %rd11, %rd14;
    ld.f32 %f3, [%rd15]; fma.rn.f32 %f2, %f3, %f3, %f2;
    add.s64 %rd13, %rd13, 1; bra RMA1;
RMA1E:
    div.rn.f32 %f5, %f2, %f1; add.f32 %f5, %f5, %f0; sqrt.rn.f32 %f5, %f5;
    mov.f32 %f6, 0f3F800000; div.rn.f32 %f5, %f6, %f5;
    mov.b64 %rd13, 0;
RMA3:
    setp.ge.s64 %p2, %rd13, %rd5; @%p2 bra NA_NEXT;
    shl.b64 %rd14, %rd13, 2; add.s64 %rd15, %rd11, %rd14;
    ld.f32 %f3, [%rd15]; mul.f32 %f3, %f3, %f5;
    mov.f32 %f7, 0f3F800000; setp.ne.s64 %p3, %rd2, 0;
    @%p3 add.s64 %rd16, %rd2, %rd14; @%p3 ld.f32 %f7, [%rd16];
    mov.f32 %f8, 0f00000000; setp.ne.s64 %p3, %rd3, 0;
    @%p3 add.s64 %rd16, %rd3, %rd14; @%p3 ld.f32 %f8, [%rd16];
    fma.rn.f32 %f3, %f3, %f7, %f8;
    add.s64 %rd16, %rd12, %rd14; st.f32 [%rd16], %f3;
    add.s64 %rd13, %rd13, 1; bra RMA3;
NA_NEXT:
    add.s64 %rd8, %rd8, 1; bra NA_ROW;
NA_DONE:
    ret;
}
"#;

/// `wukong_vmath_f32(x, out, n, op)`: elementwise activation, dispatched on the compile-time op
/// code (gated by `vmath_supported`). Transcendentals use the SFU (`ex2.approx`/`lg2.approx`/
/// `sin.approx`/`cos.approx`) and the same closed forms as the CPU kernel; the CPU<->GPU tolerance
/// gate covers the SFU-vs-Cephes difference (the offload path uses the identical SFU formulas).
const PTX_VMATH: &str = r#".func mrt_vmath (.param .b64 px, .param .b64 pout, .param .b64 pn, .param .b64 pop)
{
    .reg .b64 %rd<8>;
    .reg .f32 %f<12>;
    .reg .pred %p<4>;
    ld.param.u64 %rd0, [px];
    ld.param.u64 %rd1, [pout];
    ld.param.u64 %rd2, [pn];
    ld.param.u64 %rd3, [pop];
    mov.b64 %rd4, 0;
VM_LOOP:
    setp.ge.s64 %p0, %rd4, %rd2;
    @%p0 bra VM_DONE;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd0, %rd5;
    ld.f32 %f1, [%rd6];
    setp.eq.s64 %p1, %rd3, 0;  @%p1 bra VM0;
    setp.eq.s64 %p1, %rd3, 1;  @%p1 bra VM1;
    setp.eq.s64 %p1, %rd3, 2;  @%p1 bra VM2;
    setp.eq.s64 %p1, %rd3, 3;  @%p1 bra VM3;
    setp.eq.s64 %p1, %rd3, 4;  @%p1 bra VM4;
    setp.eq.s64 %p1, %rd3, 5;  @%p1 bra VM5;
    setp.eq.s64 %p1, %rd3, 6;  @%p1 bra VM6;
    setp.eq.s64 %p1, %rd3, 7;  @%p1 bra VM7;
    setp.eq.s64 %p1, %rd3, 8;  @%p1 bra VM8;
    setp.eq.s64 %p1, %rd3, 9;  @%p1 bra VM9;
    setp.eq.s64 %p1, %rd3, 10; @%p1 bra VM10;
    setp.eq.s64 %p1, %rd3, 11; @%p1 bra VM11;
    setp.eq.s64 %p1, %rd3, 12; @%p1 bra VM12;
    setp.eq.s64 %p1, %rd3, 13; @%p1 bra VM13;
    setp.eq.s64 %p1, %rd3, 14; @%p1 bra VM14;
    setp.eq.s64 %p1, %rd3, 15; @%p1 bra VM15;
    setp.eq.s64 %p1, %rd3, 16; @%p1 bra VM16;
    setp.eq.s64 %p1, %rd3, 17; @%p1 bra VM17;
    setp.eq.s64 %p1, %rd3, 18; @%p1 bra VM18;
    setp.eq.s64 %p1, %rd3, 19; @%p1 bra VM19;
    setp.eq.s64 %p1, %rd3, 20; @%p1 bra VM20;
    setp.eq.s64 %p1, %rd3, 21; @%p1 bra VM21;
    setp.eq.s64 %p1, %rd3, 22; @%p1 bra VM22;
    setp.eq.s64 %p1, %rd3, 23; @%p1 bra VM23;
    setp.eq.s64 %p1, %rd3, 24; @%p1 bra VM24;
    setp.eq.s64 %p1, %rd3, 26; @%p1 bra VM26;
    setp.eq.s64 %p1, %rd3, 27; @%p1 bra VM27;
    setp.eq.s64 %p1, %rd3, 28; @%p1 bra VM28;
    setp.eq.s64 %p1, %rd3, 29; @%p1 bra VM29;
    setp.eq.s64 %p1, %rd3, 30; @%p1 bra VM30;
    setp.eq.s64 %p1, %rd3, 31; @%p1 bra VM31;
    setp.eq.s64 %p1, %rd3, 32; @%p1 bra VM32;
    setp.eq.s64 %p1, %rd3, 25; @%p1 bra VM25;
    setp.eq.s64 %p1, %rd3, 33; @%p1 bra VM33;
    setp.eq.s64 %p1, %rd3, 34; @%p1 bra VM34;
    setp.eq.s64 %p1, %rd3, 35; @%p1 bra VM35;
    bra VM_DEF;
VM0:
    mul.f32 %f2, %f1, 0f3FB8AA3B; ex2.approx.f32 %f2, %f2; bra VM_ST;
VM1:
    lg2.approx.f32 %f2, %f1; mul.f32 %f2, %f2, 0f3F317218; bra VM_ST;
VM2:
    add.f32 %f3, %f1, %f1; mul.f32 %f3, %f3, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3;
    add.f32 %f3, %f3, 0f3F800000; mov.f32 %f4, 0f40000000; div.rn.f32 %f4, %f4, %f3;
    mov.f32 %f2, 0f3F800000; sub.f32 %f2, %f2, %f4; bra VM_ST;
VM3:
    neg.f32 %f3, %f1; mul.f32 %f3, %f3, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3;
    add.f32 %f3, %f3, 0f3F800000; mov.f32 %f2, 0f3F800000; div.rn.f32 %f2, %f2, %f3; bra VM_ST;
VM4:
    max.f32 %f2, %f1, 0f00000000; bra VM_ST;
VM5:
    neg.f32 %f3, %f1; mul.f32 %f3, %f3, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3;
    add.f32 %f3, %f3, 0f3F800000; mov.f32 %f4, 0f3F800000; div.rn.f32 %f4, %f4, %f3;
    mul.f32 %f2, %f1, %f4; bra VM_ST;
VM6:
    mul.f32 %f3, %f1, %f1; mul.f32 %f3, %f3, %f1;
    fma.rn.f32 %f3, %f3, 0f3D372713, %f1; mul.f32 %f3, %f3, 0f3F4C422A;
    add.f32 %f4, %f3, %f3; mul.f32 %f4, %f4, 0f3FB8AA3B; ex2.approx.f32 %f4, %f4;
    add.f32 %f4, %f4, 0f3F800000; mov.f32 %f5, 0f40000000; div.rn.f32 %f5, %f5, %f4;
    mov.f32 %f6, 0f3F800000; sub.f32 %f6, %f6, %f5; add.f32 %f6, %f6, 0f3F800000;
    mul.f32 %f2, %f1, %f6; mul.f32 %f2, %f2, 0f3F000000; bra VM_ST;
VM7:
    mul.f32 %f3, %f1, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3; sub.f32 %f3, %f3, 0f3F800000;
    setp.gt.f32 %p1, %f1, 0f00000000; selp.f32 %f2, %f1, %f3, %p1; bra VM_ST;
VM8:
    mul.f32 %f3, %f1, 0f3C23D70A; setp.gt.f32 %p1, %f1, 0f00000000;
    selp.f32 %f2, %f1, %f3, %p1; bra VM_ST;
VM9:
    abs.f32 %f3, %f1; neg.f32 %f3, %f3; mul.f32 %f3, %f3, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3;
    add.f32 %f3, %f3, 0f3F800000; lg2.approx.f32 %f3, %f3; mul.f32 %f3, %f3, 0f3F317218;
    max.f32 %f4, %f1, 0f00000000; add.f32 %f2, %f4, %f3; bra VM_ST;
VM10:
    abs.f32 %f3, %f1; neg.f32 %f3, %f3; mul.f32 %f3, %f3, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3;
    add.f32 %f3, %f3, 0f3F800000; lg2.approx.f32 %f3, %f3; mul.f32 %f3, %f3, 0f3F317218;
    max.f32 %f4, %f1, 0f00000000; add.f32 %f5, %f4, %f3;
    add.f32 %f6, %f5, %f5; mul.f32 %f6, %f6, 0f3FB8AA3B; ex2.approx.f32 %f6, %f6;
    add.f32 %f6, %f6, 0f3F800000; mov.f32 %f7, 0f40000000; div.rn.f32 %f7, %f7, %f6;
    mov.f32 %f8, 0f3F800000; sub.f32 %f8, %f8, %f7; mul.f32 %f2, %f1, %f8; bra VM_ST;
VM11:
    mul.f32 %f3, %f1, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3; sub.f32 %f3, %f3, 0f3F800000;
    mul.f32 %f3, %f3, 0f3FD62D7D; setp.gt.f32 %p1, %f1, 0f00000000;
    selp.f32 %f2, %f1, %f3, %p1; mul.f32 %f2, %f2, 0f3F867D5F; bra VM_ST;
VM12:
    add.f32 %f3, %f1, %f1; mul.f32 %f3, %f3, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3;
    add.f32 %f3, %f3, 0f3F800000; mov.f32 %f4, 0f40000000; div.rn.f32 %f4, %f4, %f3;
    mov.f32 %f5, 0f3F800000; sub.f32 %f5, %f5, %f4; sub.f32 %f2, %f1, %f5; bra VM_ST;
VM13:
    add.f32 %f3, %f1, 0f40400000; mul.f32 %f3, %f3, 0f3E2AAAAB;
    max.f32 %f3, %f3, 0f00000000; min.f32 %f2, %f3, 0f3F800000; bra VM_ST;
VM14:
    add.f32 %f3, %f1, 0f40400000; mul.f32 %f3, %f3, 0f3E2AAAAB;
    max.f32 %f3, %f3, 0f00000000; min.f32 %f3, %f3, 0f3F800000; mul.f32 %f2, %f1, %f3; bra VM_ST;
VM15:
    sin.approx.f32 %f2, %f1; bra VM_ST;
VM16:
    cos.approx.f32 %f2, %f1; bra VM_ST;
VM17:
    abs.f32 %f3, %f1; fma.rn.f32 %f4, %f3, 0f3EA7BA05, 0f3F800000;
    mov.f32 %f5, 0f3F800000; div.rn.f32 %f4, %f5, %f4;
    mov.f32 %f6, 0f3F87DC22;
    fma.rn.f32 %f6, %f6, %f4, 0fBFBA00E3;
    fma.rn.f32 %f6, %f6, %f4, 0f3FB5F0E3;
    fma.rn.f32 %f6, %f6, %f4, 0fBE91A98E;
    fma.rn.f32 %f6, %f6, %f4, 0f3E827906;
    mul.f32 %f6, %f6, %f4;
    mul.f32 %f7, %f3, %f3; neg.f32 %f7, %f7; mul.f32 %f7, %f7, 0f3FB8AA3B; ex2.approx.f32 %f7, %f7;
    mul.f32 %f6, %f6, %f7; mov.f32 %f2, 0f3F800000; sub.f32 %f2, %f2, %f6;
    copysign.f32 %f2, %f1, %f2; bra VM_ST;
VM18:
    ex2.approx.f32 %f2, %f1; bra VM_ST;
VM19:
    lg2.approx.f32 %f2, %f1; bra VM_ST;
VM20:
    mul.f32 %f3, %f1, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3;
    neg.f32 %f4, %f1; mul.f32 %f4, %f4, 0f3FB8AA3B; ex2.approx.f32 %f4, %f4;
    sub.f32 %f2, %f3, %f4; mul.f32 %f2, %f2, 0f3F000000; bra VM_ST;
VM21:
    mul.f32 %f3, %f1, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3;
    neg.f32 %f4, %f1; mul.f32 %f4, %f4, 0f3FB8AA3B; ex2.approx.f32 %f4, %f4;
    add.f32 %f2, %f3, %f4; mul.f32 %f2, %f2, 0f3F000000; bra VM_ST;
VM22:
    abs.f32 %f3, %f1; fma.rn.f32 %f4, %f1, %f1, 0f3F800000; sqrt.rn.f32 %f4, %f4;
    add.f32 %f4, %f3, %f4; lg2.approx.f32 %f4, %f4; mul.f32 %f4, %f4, 0f3F317218;
    copysign.f32 %f2, %f1, %f4; bra VM_ST;
VM23:
    fma.rn.f32 %f4, %f1, %f1, 0fBF800000; sqrt.rn.f32 %f4, %f4; add.f32 %f4, %f1, %f4;
    lg2.approx.f32 %f4, %f4; mul.f32 %f2, %f4, 0f3F317218; bra VM_ST;
VM24:
    add.f32 %f3, %f1, 0f3F800000; mov.f32 %f4, 0f3F800000; sub.f32 %f4, %f4, %f1;
    div.rn.f32 %f3, %f3, %f4; lg2.approx.f32 %f3, %f3; mul.f32 %f3, %f3, 0f3F317218;
    mul.f32 %f2, %f3, 0f3F000000; bra VM_ST;
VM26:
    mul.f32 %f3, %f1, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3; sub.f32 %f2, %f3, 0f3F800000; bra VM_ST;
VM27:
    add.f32 %f3, %f1, 0f3F800000; lg2.approx.f32 %f3, %f3; mul.f32 %f2, %f3, 0f3F317218; bra VM_ST;
VM28:
    mul.f32 %f3, %f1, 0f40549A78; ex2.approx.f32 %f2, %f3; bra VM_ST;
VM29:
    lg2.approx.f32 %f3, %f1; mul.f32 %f2, %f3, 0f3E9A209B; bra VM_ST;
VM30:
    abs.f32 %f3, %f1; add.f32 %f3, %f3, 0f3F800000; div.rn.f32 %f2, %f1, %f3; bra VM_ST;
VM31:
    abs.f32 %f3, %f1; neg.f32 %f3, %f3; mul.f32 %f3, %f3, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3;
    add.f32 %f3, %f3, 0f3F800000; lg2.approx.f32 %f3, %f3; mul.f32 %f3, %f3, 0f3F317218;
    neg.f32 %f4, %f1; max.f32 %f4, %f4, 0f00000000; add.f32 %f3, %f4, %f3; neg.f32 %f2, %f3; bra VM_ST;
VM32:
    sin.approx.f32 %f3, %f1; cos.approx.f32 %f4, %f1; div.rn.f32 %f2, %f3, %f4; bra VM_ST;
VM35:
    abs.f32 %f3, %f1; lg2.approx.f32 %f3, %f3; mul.f32 %f3, %f3, 0f3F317218;
    mul.f32 %f3, %f3, 0f3EAAAAAB; mul.f32 %f3, %f3, 0f3FB8AA3B; ex2.approx.f32 %f3, %f3;
    copysign.f32 %f2, %f1, %f3; bra VM_ST;
VM25:
    // atan(x): minimax poly atan(z)=z*P(z^2) on |z|<=1 (~1e-5 at z=1); reduce |x|>1 via pi/2-atan(1/|x|).
    abs.f32 %f3, %f1; setp.gt.f32 %p1, %f3, 0f3F800000;
    rcp.approx.f32 %f4, %f3; selp.f32 %f4, %f4, %f3, %p1;
    mul.f32 %f5, %f4, %f4; mov.f32 %f6, 0f3CAAAE5F;
    fma.rn.f32 %f6, %f6, %f5, 0fBDAE5A36; fma.rn.f32 %f6, %f6, %f5, 0f3E3876E2;
    fma.rn.f32 %f6, %f6, %f5, 0fBEA91D04; fma.rn.f32 %f6, %f6, %f5, 0f3F7FF738;
    mul.f32 %f6, %f6, %f4; mov.f32 %f7, 0f3FC90FDB; sub.f32 %f7, %f7, %f6;
    selp.f32 %f6, %f7, %f6, %p1; copysign.f32 %f2, %f1, %f6; bra VM_ST;
VM33:
    // asin(x) = atan(x / sqrt(1-x^2)), sign preserved.
    mul.f32 %f3, %f1, %f1; mov.f32 %f4, 0f3F800000; sub.f32 %f4, %f4, %f3; sqrt.rn.f32 %f4, %f4;
    abs.f32 %f5, %f1; div.rn.f32 %f3, %f5, %f4;
    setp.gt.f32 %p1, %f3, 0f3F800000; rcp.approx.f32 %f4, %f3; selp.f32 %f4, %f4, %f3, %p1;
    mul.f32 %f5, %f4, %f4; mov.f32 %f6, 0f3CAAAE5F;
    fma.rn.f32 %f6, %f6, %f5, 0fBDAE5A36; fma.rn.f32 %f6, %f6, %f5, 0f3E3876E2;
    fma.rn.f32 %f6, %f6, %f5, 0fBEA91D04; fma.rn.f32 %f6, %f6, %f5, 0f3F7FF738;
    mul.f32 %f6, %f6, %f4; mov.f32 %f7, 0f3FC90FDB; sub.f32 %f7, %f7, %f6;
    selp.f32 %f6, %f7, %f6, %p1; copysign.f32 %f2, %f1, %f6; bra VM_ST;
VM34:
    // acos(x) = pi/2 - asin(x).
    mul.f32 %f3, %f1, %f1; mov.f32 %f4, 0f3F800000; sub.f32 %f4, %f4, %f3; sqrt.rn.f32 %f4, %f4;
    abs.f32 %f5, %f1; div.rn.f32 %f3, %f5, %f4;
    setp.gt.f32 %p1, %f3, 0f3F800000; rcp.approx.f32 %f4, %f3; selp.f32 %f4, %f4, %f3, %p1;
    mul.f32 %f5, %f4, %f4; mov.f32 %f6, 0f3CAAAE5F;
    fma.rn.f32 %f6, %f6, %f5, 0fBDAE5A36; fma.rn.f32 %f6, %f6, %f5, 0f3E3876E2;
    fma.rn.f32 %f6, %f6, %f5, 0fBEA91D04; fma.rn.f32 %f6, %f6, %f5, 0f3F7FF738;
    mul.f32 %f6, %f6, %f4; mov.f32 %f7, 0f3FC90FDB; sub.f32 %f7, %f7, %f6;
    selp.f32 %f6, %f7, %f6, %p1; copysign.f32 %f6, %f1, %f6;
    mov.f32 %f7, 0f3FC90FDB; sub.f32 %f2, %f7, %f6; bra VM_ST;
VM_DEF:
    mov.f32 %f2, %f1;
VM_ST:
    add.s64 %rd6, %rd1, %rd5;
    st.f32 [%rd6], %f2;
    add.s64 %rd4, %rd4, 1;
    bra VM_LOOP;
VM_DONE:
    ret;
}
"#;

/// `wukong_vmath2_f32(in1, in2, out, n, op)`: the two-arg transcendentals over f32 arrays —
/// pow(0) = `in1^in2` = exp2(in2·log2(in1)); atan2(1) = angle of (in2, in1) = atan(in1/in2) +
/// quadrant fix; hypot(2) = `sqrt(in1²+in2²)`. atan uses the same inline minimax poly as VM25; the
/// CPU<->GPU tolerance gate covers the SFU/poly-vs-libm difference.
const PTX_VMATH2: &str = r#".func mrt_vmath2 (.param .b64 p1, .param .b64 p2, .param .b64 pout, .param .b64 pn, .param .b64 pop)
{
    .reg .b64 %rd<9>;
    .reg .f32 %f<12>;
    .reg .pred %p<4>;
    ld.param.u64 %rd0, [p1];
    ld.param.u64 %rd1, [p2];
    ld.param.u64 %rd2, [pout];
    ld.param.u64 %rd3, [pn];
    ld.param.u64 %rd4, [pop];
    mov.b64 %rd5, 0;
V2_LOOP:
    setp.ge.s64 %p0, %rd5, %rd3;
    @%p0 bra V2_DONE;
    shl.b64 %rd6, %rd5, 2;
    add.s64 %rd7, %rd0, %rd6;
    ld.f32 %f1, [%rd7];
    add.s64 %rd7, %rd1, %rd6;
    ld.f32 %f2, [%rd7];
    setp.eq.s64 %p1, %rd4, 0; @%p1 bra V2_POW;
    setp.eq.s64 %p1, %rd4, 1; @%p1 bra V2_ATAN2;
    setp.eq.s64 %p1, %rd4, 2; @%p1 bra V2_HYPOT;
    mov.f32 %f3, %f1; bra V2_ST;
V2_POW:
    lg2.approx.f32 %f3, %f1; mul.f32 %f3, %f3, %f2; ex2.approx.f32 %f3, %f3; bra V2_ST;
V2_HYPOT:
    mul.f32 %f3, %f1, %f1; fma.rn.f32 %f3, %f2, %f2, %f3; sqrt.rn.f32 %f3, %f3; bra V2_ST;
V2_ATAN2:
    div.rn.f32 %f4, %f1, %f2;
    abs.f32 %f5, %f4; setp.gt.f32 %p1, %f5, 0f3F800000;
    rcp.approx.f32 %f6, %f5; selp.f32 %f6, %f6, %f5, %p1;
    mul.f32 %f7, %f6, %f6; mov.f32 %f3, 0f3CAAAE5F;
    fma.rn.f32 %f3, %f3, %f7, 0fBDAE5A36; fma.rn.f32 %f3, %f3, %f7, 0f3E3876E2;
    fma.rn.f32 %f3, %f3, %f7, 0fBEA91D04; fma.rn.f32 %f3, %f3, %f7, 0f3F7FF738;
    mul.f32 %f3, %f3, %f6; mov.f32 %f8, 0f3FC90FDB; sub.f32 %f8, %f8, %f3;
    selp.f32 %f3, %f8, %f3, %p1; copysign.f32 %f3, %f4, %f3;
    setp.lt.f32 %p1, %f2, 0f00000000;
    mov.f32 %f8, 0f40490FDB; copysign.f32 %f8, %f1, %f8; add.f32 %f9, %f3, %f8;
    selp.f32 %f3, %f9, %f3, %p1; bra V2_ST;
V2_ST:
    add.s64 %rd7, %rd2, %rd6;
    st.f32 [%rd7], %f3;
    add.s64 %rd5, %rd5, 1;
    bra V2_LOOP;
V2_DONE:
    ret;
}
"#;

/// `wukong_velem_f32(x, y, out, n, a, b, c, op)`: streaming elementwise `out[i] = act(base)` where
/// `base` is the **binary** product `x·y` (`op & 512`, VE_HADAMARD) or quotient `x/y` (`op & 1024`,
/// VE_DIV) — both read `y` — else the **affine** `a·x[i] + inner` with `inner = (op & 256) ? b·y[i]+c :
/// c`; `act = op & 0xff` is identity(0)/relu(1)/relu6(2). `div.rn.f32` is correctly-rounded IEEE
/// division, matching the CPU kernel's `x / y` bit-for-bit within the gate. The two-fma affine chain
/// matches the CPU kernel (e.g. the residual add `x + y` is `op=256, a=b=1, c=0`).
const PTX_VELEM: &str = r#".func mrt_velem (.param .b64 px, .param .b64 py, .param .b64 pout, .param .b64 pn, .param .f32 pa, .param .f32 pb, .param .f32 pc, .param .b64 pop)
{
    .reg .b64 %rd<12>;
    .reg .f32 %f<8>;
    .reg .pred %p<4>;
    ld.param.u64 %rd0, [px];
    ld.param.u64 %rd1, [py];
    ld.param.u64 %rd2, [pout];
    ld.param.u64 %rd3, [pn];
    ld.param.f32 %f4, [pa];
    ld.param.f32 %f5, [pb];
    ld.param.f32 %f6, [pc];
    ld.param.u64 %rd4, [pop];
    and.b64 %rd5, %rd4, 256;
    and.b64 %rd6, %rd4, 255;
    and.b64 %rd10, %rd4, 512;
    and.b64 %rd11, %rd4, 1024;
    or.b64 %rd5, %rd5, %rd10;
    or.b64 %rd5, %rd5, %rd11;
    mov.b64 %rd7, 0;
VE_LOOP:
    setp.ge.s64 %p0, %rd7, %rd3;
    @%p0 bra VE_DONE;
    shl.b64 %rd8, %rd7, 2;
    add.s64 %rd9, %rd0, %rd8;
    ld.f32 %f1, [%rd9];
    setp.ne.s64 %p1, %rd5, 0;
    mov.f32 %f7, %f6;
    @%p1 add.s64 %rd9, %rd1, %rd8;
    @%p1 ld.f32 %f2, [%rd9];
    @%p1 fma.rn.f32 %f7, %f5, %f2, %f6;
    fma.rn.f32 %f3, %f4, %f1, %f7;
    setp.ne.s64 %p2, %rd10, 0;
    setp.ne.s64 %p3, %rd11, 0;
    @%p2 mul.f32 %f3, %f1, %f2;
    @%p3 div.rn.f32 %f3, %f1, %f2;
    setp.eq.s64 %p1, %rd6, 1;
    setp.eq.s64 %p2, %rd6, 2;
    @%p1 max.f32 %f3, %f3, 0f00000000;
    @%p2 max.f32 %f3, %f3, 0f00000000;
    @%p2 min.f32 %f3, %f3, 0f40C00000;
    add.s64 %rd9, %rd2, %rd8;
    st.f32 [%rd9], %f3;
    add.s64 %rd7, %rd7, 1;
    bra VE_LOOP;
VE_DONE:
    ret;
}
"#;

/// `mrt_fmod(a, b) -> a % b` (true IEEE fmod, f64), the helper float `%` calls. Exact via binary
/// scaling: align `rb·2^k` just below `ra=|a|`, then subtract-and-halve down to `rb` — every step
/// (`*2`, `*0.5`, `ra-=m`) is exact, so the result is the exact remainder regardless of magnitude
/// (the `a-trunc(a/b)*b` identity fails past 2^53). f32 `%` widens to f64 here and narrows the result.
const PTX_FMOD: &str = r#".func (.param .f64 _r) mrt_fmod (.param .f64 pa, .param .f64 pb)
{
    .reg .f64 %fd<6>;
    .reg .pred %p<3>;
    ld.param.f64 %fd0, [pa];
    ld.param.f64 %fd1, [pb];
    abs.f64 %fd2, %fd0;
    abs.f64 %fd3, %fd1;
    setp.eq.f64 %p0, %fd3, 0d0000000000000000;
    @%p0 bra FMOD_RET;
    setp.lt.f64 %p0, %fd2, %fd3;
    @%p0 bra FMOD_RET;
    mov.f64 %fd4, %fd3;
FMOD_UP:
    add.f64 %fd5, %fd4, %fd4;
    setp.gt.f64 %p1, %fd5, %fd2;
    @%p1 bra FMOD_LOOP;
    mov.f64 %fd4, %fd5;
    bra FMOD_UP;
FMOD_LOOP:
    setp.lt.f64 %p1, %fd2, %fd4;
    @%p1 bra FMOD_SKIP;
    sub.f64 %fd2, %fd2, %fd4;
FMOD_SKIP:
    setp.eq.f64 %p1, %fd4, %fd3;
    @%p1 bra FMOD_RET;
    mul.f64 %fd4, %fd4, 0d3FE0000000000000;
    bra FMOD_LOOP;
FMOD_RET:
    copysign.f64 %fd2, %fd0, %fd2;
    st.param.f64 [_r], %fd2;
    ret;
}
"#;

// --------------------------------------------------------------------------------------------
// Mixed-precision (bf16/f16 storage, f32 compute) device helpers.
//
// These reproduce the `wukong_{vmath,dot,sum,reduce,axpby}_{bf16,f16}` CPU kernels: the storage is
// 2-byte bf16/f16 (loaded with `ld.u16` + `cvt.f32.{bf16,f16}`, a lossless widen for the bf16-/f16-
// exact corpus values), all math is f32, and outputs/accumulators are f32 (the standard ML
// "low-precision storage, f32 accumulate" contract). PTX labels are function-scoped (verified), so
// the bf16 and f16 variants reuse the same internal label names without colliding. `cvt` is
// `"bf16"` or `"f16"`; the two variants differ only in that conversion suffix.
// --------------------------------------------------------------------------------------------

/// `wukong_vmath_{bf16,f16}(x, out, n, op)`: elementwise activation reading a bf16/f16 input array
/// (2-byte stride) and writing f32 (4-byte stride). Reuses the f32 kernel's op-switch verbatim (the
/// op codes and SFU formulas are identical; only the input load differs), so the result is bit-for-
/// bit the f32 kernel run on the widened inputs.
fn ptx_vmath_lowp(fn_name: &str, cvt: &str) -> Option<String> {
    // Splice in the shared dispatch + op bodies (`setp …; VM0: …; VM_DEF: mov %f2,%f1`) from the
    // committed f32 kernel so there is one source of truth for the ~30 activation formulas.
    // The two markers are an invariant of `PTX_VMATH` in this same file; if an edit ever renames or
    // reflows them, DECLINE (the caller turns `None` into an `UNSUPPORTED:` diagnostic) instead of
    // panicking with internal text on a path a user program reaches. `vmath_lowp_markers_present`
    // below pins the invariant so a desync is caught at edit time, not at compile time of a program.
    let s = PTX_VMATH;
    let start = s.find(VM_DISPATCH_MARKER)?;
    let end = s.find(VM_STORE_MARKER)?;
    let ops = &s[start..end];
    Some(format!(
        r#".func {fn_name} (.param .b64 px, .param .b64 pout, .param .b64 pn, .param .b64 pop)
{{
    .reg .b64 %rd<9>;
    .reg .f32 %f<12>;
    .reg .b16 %rs<1>;
    .reg .pred %p<4>;
    ld.param.u64 %rd0, [px];
    ld.param.u64 %rd1, [pout];
    ld.param.u64 %rd2, [pn];
    ld.param.u64 %rd3, [pop];
    mov.b64 %rd4, 0;
VM_LOOP:
    setp.ge.s64 %p0, %rd4, %rd2;
    @%p0 bra VM_DONE;
    shl.b64 %rd5, %rd4, 1;
    add.s64 %rd6, %rd0, %rd5;
    ld.u16 %rs0, [%rd6];
    cvt.f32.{cvt} %f1, %rs0;
{ops}VM_ST:
    shl.b64 %rd8, %rd4, 2;
    add.s64 %rd6, %rd1, %rd8;
    st.f32 [%rd6], %f2;
    add.s64 %rd4, %rd4, 1;
    bra VM_LOOP;
VM_DONE:
    ret;
}}
"#
    ))
}

/// The two `PTX_VMATH` anchors [`ptx_vmath_lowp`] splices between: the first line of the op-code
/// dispatch, and the shared store label that ends the op bodies. Named constants so the test below
/// pins exactly what the splice looks for.
const VM_DISPATCH_MARKER: &str = "    setp.eq.s64 %p1, %rd3, 0;";
const VM_STORE_MARKER: &str = "VM_ST:";

/// `wukong_dot_{bf16,f16}(x, y, n) -> f32`: `Σ widen(x[k])·widen(y[k])`, f32 accumulate.
fn ptx_dot_lowp(fn_name: &str, cvt: &str) -> String {
    format!(
        r#".func (.param .f32 _r) {fn_name} (.param .b64 px, .param .b64 py, .param .b64 pn)
{{
    .reg .b64 %rd<7>;
    .reg .f32 %f<4>;
    .reg .b16 %rs<2>;
    .reg .pred %p<2>;
    ld.param.u64 %rd0, [px];
    ld.param.u64 %rd1, [py];
    ld.param.u64 %rd2, [pn];
    mov.b64 %rd3, 0;
    mov.f32 %f0, 0f00000000;
DOT_LOOP:
    setp.ge.s64 %p0, %rd3, %rd2;
    @%p0 bra DOT_DONE;
    shl.b64 %rd4, %rd3, 1;
    add.s64 %rd5, %rd0, %rd4;
    ld.u16 %rs0, [%rd5];
    cvt.f32.{cvt} %f1, %rs0;
    add.s64 %rd6, %rd1, %rd4;
    ld.u16 %rs1, [%rd6];
    cvt.f32.{cvt} %f2, %rs1;
    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.s64 %rd3, %rd3, 1;
    bra DOT_LOOP;
DOT_DONE:
    st.param.f32 [_r], %f0;
    ret;
}}
"#
    )
}

/// `wukong_sum_{bf16,f16}(x, n) -> f32`: `Σ widen(x[k])`, f32 accumulate.
fn ptx_sum_lowp(fn_name: &str, cvt: &str) -> String {
    format!(
        r#".func (.param .f32 _r) {fn_name} (.param .b64 px, .param .b64 pn)
{{
    .reg .b64 %rd<6>;
    .reg .f32 %f<2>;
    .reg .b16 %rs<1>;
    .reg .pred %p<2>;
    ld.param.u64 %rd0, [px];
    ld.param.u64 %rd1, [pn];
    mov.b64 %rd2, 0;
    mov.f32 %f0, 0f00000000;
SUM_LOOP:
    setp.ge.s64 %p0, %rd2, %rd1;
    @%p0 bra SUM_DONE;
    shl.b64 %rd3, %rd2, 1;
    add.s64 %rd4, %rd0, %rd3;
    ld.u16 %rs0, [%rd4];
    cvt.f32.{cvt} %f1, %rs0;
    add.f32 %f0, %f0, %f1;
    add.s64 %rd2, %rd2, 1;
    bra SUM_LOOP;
SUM_DONE:
    st.param.f32 [_r], %f0;
    ret;
}}
"#
    )
}

/// `wukong_reduce_{bf16,f16}(x, n, op) -> f32`: max(4)/min(5)/maxabs(6) fold over `widen(x[k])`.
/// Order-independent and exact (no rounding), so it matches the CPU kernel's result for any order.
fn ptx_reduce_lowp(fn_name: &str, cvt: &str) -> String {
    format!(
        r#".func (.param .f32 _r) {fn_name} (.param .b64 px, .param .b64 pn, .param .b64 pop)
{{
    .reg .b64 %rd<6>;
    .reg .f32 %f<2>;
    .reg .b16 %rs<1>;
    .reg .pred %p<3>;
    ld.param.u64 %rd0, [px];
    ld.param.u64 %rd1, [pn];
    ld.param.u64 %rd2, [pop];
    mov.b64 %rd3, 0;
    // init: -inf for max/maxabs, +inf for min.
    mov.f32 %f0, 0fFF800000;
    setp.eq.s64 %p1, %rd2, 5;
    @%p1 mov.f32 %f0, 0f7F800000;
RED_LOOP:
    setp.ge.s64 %p0, %rd3, %rd1;
    @%p0 bra RED_DONE;
    shl.b64 %rd4, %rd3, 1;
    add.s64 %rd5, %rd0, %rd4;
    ld.u16 %rs0, [%rd5];
    cvt.f32.{cvt} %f1, %rs0;
    setp.eq.s64 %p2, %rd2, 6;
    @%p2 abs.f32 %f1, %f1;
    setp.eq.s64 %p1, %rd2, 5;
    @%p1 min.f32 %f0, %f0, %f1;
    @!%p1 max.f32 %f0, %f0, %f1;
    add.s64 %rd3, %rd3, 1;
    bra RED_LOOP;
RED_DONE:
    st.param.f32 [_r], %f0;
    ret;
}}
"#
    )
}

/// `wukong_axpby_{bf16,f16}(x, y, out, n, a, b)`: `out[i] = a·widen(x[i]) + b·widen(y[i])` (bf16/f16
/// in, f32 out, f32 math).
fn ptx_axpby_lowp(fn_name: &str, cvt: &str) -> String {
    format!(
        r#".func {fn_name} (.param .b64 px, .param .b64 py, .param .b64 pout, .param .b64 pn, .param .f32 pa, .param .f32 pb)
{{
    .reg .b64 %rd<8>;
    .reg .f32 %f<6>;
    .reg .b16 %rs<2>;
    .reg .pred %p<2>;
    ld.param.u64 %rd0, [px];
    ld.param.u64 %rd1, [py];
    ld.param.u64 %rd2, [pout];
    ld.param.u64 %rd3, [pn];
    ld.param.f32 %f4, [pa];
    ld.param.f32 %f5, [pb];
    mov.b64 %rd4, 0;
AXPBY_LOOP:
    setp.ge.s64 %p0, %rd4, %rd3;
    @%p0 bra AXPBY_DONE;
    shl.b64 %rd5, %rd4, 1;
    add.s64 %rd6, %rd0, %rd5;
    ld.u16 %rs0, [%rd6];
    cvt.f32.{cvt} %f1, %rs0;
    add.s64 %rd6, %rd1, %rd5;
    ld.u16 %rs1, [%rd6];
    cvt.f32.{cvt} %f2, %rs1;
    mul.f32 %f3, %f4, %f1;
    fma.rn.f32 %f3, %f5, %f2, %f3;
    shl.b64 %rd7, %rd4, 2;
    add.s64 %rd6, %rd2, %rd7;
    st.f32 [%rd6], %f3;
    add.s64 %rd4, %rd4, 1;
    bra AXPBY_LOOP;
AXPBY_DONE:
    ret;
}}
"#
    )
}

/// `wukong_sgemm_nt_epi(a, b, c, m, k, n, beta, bias, act)`: fused `C = act(A.Bt [+ bias])` (the
/// nn.Linear/FFN epilogue). `bias` is null (passed as integer 0) for no bias; `act` is
/// identity(0)/relu(1)/gelu(2)/silu(3). silu/gelu reuse the SFU forms from mrt_vmath; tolerance-gated.
const PTX_SGEMM_NT_EPI: &str = r#".func mrt_sgemm_nt_epi (.param .b64 pa, .param .b64 pb, .param .b64 pc, .param .b64 pm, .param .b64 pk, .param .b64 pn, .param .b64 pbeta, .param .b64 pbias, .param .b64 pact)
{
    .reg .b64 %rd<24>;
    .reg .f32 %f<12>;
    .reg .pred %p<6>;
    ld.param.u64 %rd0, [pa];
    ld.param.u64 %rd1, [pb];
    ld.param.u64 %rd2, [pc];
    ld.param.u64 %rd3, [pm];
    ld.param.u64 %rd4, [pk];
    ld.param.u64 %rd5, [pn];
    ld.param.u64 %rd6, [pbeta];
    ld.param.u64 %rd16, [pbias];
    ld.param.u64 %rd17, [pact];
    mov.b64 %rd7, 0;
EPI_LI:
    setp.ge.s64 %p0, %rd7, %rd3;
    @%p0 bra EPI_EI;
    mov.b64 %rd8, 0;
EPI_LJ:
    setp.ge.s64 %p1, %rd8, %rd5;
    @%p1 bra EPI_EJ;
    mov.f32 %f0, 0f00000000;
    mul.lo.s64 %rd9, %rd7, %rd4;
    shl.b64 %rd9, %rd9, 2;
    add.s64 %rd9, %rd0, %rd9;
    mul.lo.s64 %rd10, %rd8, %rd4;
    shl.b64 %rd10, %rd10, 2;
    add.s64 %rd10, %rd1, %rd10;
    mov.b64 %rd11, 0;
EPI_LL:
    setp.ge.s64 %p2, %rd11, %rd4;
    @%p2 bra EPI_EL;
    shl.b64 %rd12, %rd11, 2;
    add.s64 %rd13, %rd9, %rd12;
    ld.f32 %f1, [%rd13];
    add.s64 %rd14, %rd10, %rd12;
    ld.f32 %f2, [%rd14];
    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.s64 %rd11, %rd11, 1;
    bra EPI_LL;
EPI_EL:
    mul.lo.s64 %rd15, %rd7, %rd5;
    add.s64 %rd15, %rd15, %rd8;
    shl.b64 %rd15, %rd15, 2;
    add.s64 %rd15, %rd2, %rd15;
    setp.eq.s64 %p3, %rd6, 0;
    @%p3 bra EPI_NOACC;
    ld.f32 %f3, [%rd15];
    add.rn.f32 %f0, %f0, %f3;
EPI_NOACC:
    setp.eq.s64 %p3, %rd16, 0;
    @%p3 bra EPI_NOBIAS;
    shl.b64 %rd18, %rd8, 2;
    add.s64 %rd18, %rd16, %rd18;
    ld.f32 %f3, [%rd18];
    add.rn.f32 %f0, %f0, %f3;
EPI_NOBIAS:
    setp.eq.s64 %p3, %rd17, 1;
    @%p3 bra EPI_RELU;
    setp.eq.s64 %p3, %rd17, 2;
    @%p3 bra EPI_GELU;
    setp.eq.s64 %p3, %rd17, 3;
    @%p3 bra EPI_SILU;
    bra EPI_ST;
EPI_RELU:
    max.f32 %f0, %f0, 0f00000000;
    bra EPI_ST;
EPI_SILU:
    neg.f32 %f4, %f0; mul.f32 %f4, %f4, 0f3FB8AA3B; ex2.approx.f32 %f4, %f4;
    add.f32 %f4, %f4, 0f3F800000; mov.f32 %f5, 0f3F800000; div.rn.f32 %f5, %f5, %f4;
    mul.f32 %f0, %f0, %f5;
    bra EPI_ST;
EPI_GELU:
    mul.f32 %f4, %f0, %f0; mul.f32 %f4, %f4, %f0;
    fma.rn.f32 %f4, %f4, 0f3D372713, %f0; mul.f32 %f4, %f4, 0f3F4C422A;
    add.f32 %f5, %f4, %f4; mul.f32 %f5, %f5, 0f3FB8AA3B; ex2.approx.f32 %f5, %f5;
    add.f32 %f5, %f5, 0f3F800000; mov.f32 %f6, 0f40000000; div.rn.f32 %f6, %f6, %f5;
    mov.f32 %f7, 0f3F800000; sub.f32 %f7, %f7, %f6; add.f32 %f7, %f7, 0f3F800000;
    mul.f32 %f0, %f0, %f7; mul.f32 %f0, %f0, 0f3F000000;
    bra EPI_ST;
EPI_ST:
    st.f32 [%rd15], %f0;
    add.s64 %rd8, %rd8, 1;
    bra EPI_LJ;
EPI_EJ:
    add.s64 %rd7, %rd7, 1;
    bra EPI_LI;
EPI_EI:
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

    if std::env::var_os("WUKONG_GPU_DUMP_PTX").is_some() {
        eprintln!("--- gpu-native PTX ---\n{ptx}\n--- end PTX ---");
    }

    let f = g.function(key, ptx, KERNEL_NAME).map_err(|e| {
        // Persist the PTX so a JIT/ptxas error (otherwise an opaque driver code) is debuggable.
        let p = std::env::temp_dir().join(format!("{key}.ptx"));
        let _ = std::fs::write(&p, ptx);
        format!("gpu-native JIT/load failed: {e:?}\n  (PTX written to {})", p.display())
    })?;

    let host = new_ctx_host();
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

    decode_ctx(&out)
}

/// The zeroed host mirror of the device context buffer (record count + assert flag + exit code +
/// reserved, then `RECORD_CAP` two-u64 print records). Shared by the single-thread and mega launchers.
pub(crate) fn new_ctx_host() -> Vec<u64> {
    vec![0u64; 4 + 2 * RECORD_CAP as usize]
}

/// Decode a context buffer read back from the device into `(exit_code, stdout)` — the print records
/// replayed with the *identical* `format!("{}\n")` the interpreter/Cranelift runtime uses. Shared by
/// `run_on_device` and the megakernel launcher so both produce byte-identical output.
pub(crate) fn decode_ctx(out: &[u64]) -> Result<(i64, Vec<u8>), String> {
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

/// Total bytes of `func`'s alloca frame (mirrors `FnEmit::assign_frame`'s slot sizing) — the size of
/// the shared `.global` frame buffer the megakernel launcher allocates + zeroes for the whole block.
pub fn mega_frame_bytes(func: &Function) -> u64 {
    let mut off = 0u64;
    for b in &func.blocks {
        for inst in &b.insts {
            if let Op::Alloca(ty) = &inst.op {
                if inst.result.is_some() {
                    let sz = size_of(ty).max(1);
                    off += sz.div_ceil(8) * 8;
                }
            }
        }
    }
    off
}

// ============================================================================================
// Coverage gate: tests/run corpus through the GPU-lowering backend vs the interpreter oracle.
// ============================================================================================
#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// `ptx_vmath_lowp` builds the bf16/f16 activation kernels by slicing `PTX_VMATH` between two
    /// literal markers. That is a self-referential invariant of this file: renaming `VM_ST:` or
    /// reflowing the first dispatch line silently turns every bf16/f16 activation program into an
    /// `UNSUPPORTED:` decline. Pin it here — this test needs no CUDA device, so it fires on any
    /// `--features gpu` build the moment the PTX literal and the splice desync.
    #[test]
    fn vmath_lowp_markers_present() {
        assert!(
            PTX_VMATH.contains(VM_DISPATCH_MARKER),
            "PTX_VMATH no longer contains the op-dispatch marker {VM_DISPATCH_MARKER:?} that \
             ptx_vmath_lowp splices from"
        );
        assert!(
            PTX_VMATH.contains(VM_STORE_MARKER),
            "PTX_VMATH no longer contains the store marker {VM_STORE_MARKER:?} that \
             ptx_vmath_lowp splices to"
        );
        // And the splice really produces a kernel (markers in the right order, non-empty body).
        let k = ptx_vmath_lowp("mrt_vmath_bf16", "bf16").expect("bf16 vmath kernel");
        assert!(k.contains("cvt.f32.bf16"), "spliced kernel lost its bf16 input widen");
        assert!(k.is_ascii(), "PTX must stay ASCII");
    }

    /// The comparator the corpus gate is built on, pinned against the real divergence it used to
    /// wave through. The four "before" lines are the actual gpu-native -O3 output of
    /// `tests/run/bf16_literal_grid.wk` when bf16/f16 constants were not grid-rounded; the oracle
    /// column is the interpreter's. Every one of them passed the old 1e-2 relative tolerance.
    /// Needs no CUDA device.
    #[test]
    fn integer_lines_compare_exactly() {
        for (gpu, cpu) in [("10000", "10009"), ("10000", "9997"), ("32968", "33125"), ("3141", "3140")]
        {
            assert!(
                !line_matches(gpu, cpu),
                "integer line {gpu} must not match the oracle's {cpu}"
            );
        }
        assert!(!outputs_match(b"10000\n10000\n32968\n3141\n", b"10009\n9997\n33125\n3140\n"));
        // Equal integers still match, in any spelling that parses to the same value.
        assert!(line_matches("12345678901", "12345678901"));
        assert!(line_matches("-0", "0"));
        // Floats keep the CPU<->GPU tolerance (reduction order / SFU differences).
        assert!(line_matches("1.0000001", "1.0"));
        assert!(line_matches("0.30000000000000004", "0.3"));
        assert!(!line_matches("1.5", "2.5"));
    }

    /// Build a program from `.wk` source at `opt` (lex -> parse -> sema -> mir_build -> opt).
    fn build(src: &str, opt: u8) -> Option<(Program, Interner)> {
        use wukong_span::SourceMap;
        let mut sm = SourceMap::new();
        let id = sm.add("gate.wk".to_string(), src.to_string());
        let (tokens, ld) = wukong_lexer::tokenize(sm.source(id), id);
        if ld.iter().any(|d| d.is_error()) {
            return None;
        }
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        if pd.iter().any(|d| d.is_error()) {
            return None;
        }
        let (sema, sd) = wukong_sema::check(&module, &interner);
        if sd.iter().any(|d| d.is_error()) {
            return None;
        }
        let (mut program, md) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        if md.iter().any(|d| d.is_error()) {
            return None;
        }
        wukong_opt::optimize(&mut program, opt);
        Some((program, interner))
    }

    /// Compare one output line. **Integer output is compared exactly**: `decode_ctx` replays an
    /// integer print record with the identical `format!("{}\n", payload as i64)` the interpreter
    /// uses, so any difference on such a line is a miscompile, never a rounding artifact. Only a
    /// line carrying a float form gets the CPU<->GPU tolerance, which exists for reduction-order and
    /// SFU differences.
    ///
    /// The tolerance used to apply to *every* numeric line, which made a large class of miscompiles
    /// report green: on `bf16_literal_grid.wk` the GPU printed 10000 where the oracle printed 10009
    /// and `9 <= 1e-2 * 10009` accepted it. bf16/f16 error is structurally bounded by 2^-9 relative,
    /// so a 1e-2 relative slack could never catch that bug class.
    fn line_matches(g: &str, c: &str) -> bool {
        if g == c {
            return true;
        }
        let (gt, ct) = (g.trim(), c.trim());
        if let (Ok(a), Ok(b)) = (gt.parse::<i128>(), ct.parse::<i128>()) {
            return a == b;
        }
        match (gt.parse::<f64>(), ct.parse::<f64>()) {
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
        // The mission corpus: the e2e `tests/run` fixtures. (The example/bench kernels are
        // compute-heavy perf benchmarks — e.g. fib(30) is ~2.7M calls — which the single-thread
        // execution model can't run within the OS GPU watchdog; their *lowering* is correct, but
        // running them needs the grid-parallel perf phase. They are not a lowering-correctness gate.)
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/run");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("read tests/run")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "wk").unwrap_or(false))
            .collect();
        files.sort();

        let mut covered = 0usize;
        let mut skipped: Vec<String> = Vec::new();
        let mut mismatches: Vec<String> = Vec::new();
        // Genuine *driver* faults (a kernel that dereferenced out of bounds -> ILLEGAL_ADDRESS), kept
        // distinct from miscompiles (wrong output on a healthy context). A fault poisons the shared
        // primary context; `crate::gpu::reset_gpu` attempts recovery, but on this driver error 700 is
        // process-fatal (the re-retain still returns it), so recovery degrades to record-and-skip:
        // the remaining programs land on `lost` with a loud count instead of cascading as false
        // failures. Only the *root* fault is ever reported as a fault.
        let mut faults: Vec<String> = Vec::new();
        let mut lost: Vec<String> = Vec::new();

        for path in &files {
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            // After an unrecoverable fault, stop attempting GPU work: every further run would fail
            // with the same sticky error and misreport healthy programs as broken.
            if crate::gpu::device_lost() {
                lost.push(name);
                continue;
            }
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
                let cpu = wukong_interp::run_with_output(&program, entry, &interner);
                let gpu = jit_run(&program, entry, &interner);
                match (&cpu, &gpu) {
                    // Both errored. That is only *agreement* when the GPU's error is the one
                    // deterministic program-level error both executors produce identically. An
                    // unchecked "both Err" arm counted an `UNSUPPORTED:` decline as coverage and,
                    // worse, let a genuine driver fault through without landing on `faults` or
                    // resetting the context — so the next healthy program inherited the sticky
                    // CUDA error and took the blame, breaking this gate's root-cause-only rule.
                    (Err(ce), Err(ge)) => {
                        if ge.starts_with(UNSUPPORTED)
                            || ge.starts_with("gpu-native print buffer overflow")
                        {
                            // A decline or a device-capacity limit: no GPU output existed to compare.
                            covered_here = false;
                            skipped.push(format!("{name}@O{opt}: {ge}"));
                            break;
                        }
                        if ge != "assertion failed" {
                            covered_here = false;
                            faults.push(format!("{name}@O{opt}: gpu errored: {ge}"));
                            crate::gpu::reset_gpu();
                            break;
                        }
                        if ce != ge {
                            covered_here = false;
                            mismatches
                                .push(format!("{name}@O{opt}: cpu err `{ce}` gpu err `{ge}`"));
                            break;
                        }
                        // Both `assertion failed` — agree.
                    }
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
                        covered_here = false;
                        if e.starts_with(UNSUPPORTED) {
                            skipped.push(format!("{name}@O{opt}: {e}"));
                        } else {
                            // A real driver fault poisons the shared context: record it on the fault
                            // ledger and attempt a context reset. If the reset fails (sticky
                            // process-level error), `device_lost()` flips and the loop above skips
                            // the remaining programs loudly instead of cascading.
                            faults.push(format!("{name}@O{opt}: gpu errored: {e}"));
                            crate::gpu::reset_gpu();
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
        if !faults.is_empty() {
            eprintln!("-- driver FAULTS ({}, root causes only — no cascade):", faults.len());
            for f in &faults {
                eprintln!("   {f}");
            }
        }
        if !lost.is_empty() {
            eprintln!(
                "-- NOT RUN ({}): device lost after the fault above (sticky CUDA error; restart to \
                 re-test): {}",
                lost.len(),
                lost.join(", ")
            );
        }
        assert!(
            mismatches.is_empty(),
            "GPU-lowered programs disagree with the interpreter oracle (miscompiles):\n{}",
            mismatches.join("\n")
        );
        assert!(
            faults.is_empty(),
            "GPU-lowered programs faulted on the device ({} root fault(s); {} later programs \
             were skipped after device loss, NOT failed — these are genuine kernel faults, not \
             miscompiles):\n{}",
            faults.len(),
            lost.len(),
            faults.join("\n")
        );
        assert!(covered > 0, "no programs covered — pipeline broken");
    }

    /// Diagnostic: JIT the PTX file named by `WUKONG_PTX_FILE` with the driver error-log buffer
    /// attached, printing ptxas's actual line/error. Run:
    /// `WUKONG_PTX_FILE=... cargo test -p wukong_codegen_gpu --features gpu jit_log_file -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic; prints the driver JIT log for a PTX file"]
    fn jit_log_file() {
        let path = std::env::var("WUKONG_PTX_FILE").expect("set WUKONG_PTX_FILE");
        let ptx = std::fs::read_to_string(&path).unwrap();
        let mut guard = crate::gpu::gpu();
        let Some(g) = guard.as_mut() else {
            eprintln!("skip jit_log_file: no CUDA device");
            return;
        };
        {
            use cudarc::driver::sys;
            g.ctx.bind_to_thread().unwrap();
            let ptx_c = std::ffi::CString::new(ptx.as_str()).unwrap();
            let mut log = vec![0u8; 32768];
            let mut opts = [
                sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
                sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            ];
            let mut vals: [*mut std::ffi::c_void; 2] =
                [log.as_mut_ptr() as *mut _, log.len() as *mut _];
            let mut module: sys::CUmodule = std::ptr::null_mut();
            let res = unsafe {
                sys::cuModuleLoadDataEx(
                    &mut module,
                    ptx_c.as_ptr() as *const _,
                    2,
                    opts.as_mut_ptr(),
                    vals.as_mut_ptr(),
                )
            };
            eprintln!(
                "=== JIT result {:?} ===\n{}",
                res,
                String::from_utf8_lossy(&log).trim_end_matches('\0')
            );
        }
    }
}
