//! P4 raw-AVX2 emission path for the general (non-recognized) loop vectorizer.
//!
//! Cranelift's x64 vector ISA is capped at 128-bit: a 256-bit `f32x8` SSA value is rejected at
//! `define_function` ("Unexpected SSA-value type" — see the `p4_probe_vec256_ops` tripwire). So the
//! general vectorizer's widest CLIF-legal lane is `f32x4`, and it recovers AVX-class throughput only
//! by 4×-unrolling 128-bit chains — half the FMA width the core actually has.
//!
//! This module closes that gap the same way the runtime GEMM/vmath microkernels do: by emitting true
//! 256-bit AVX2 machine code directly (here via the `iced-x86` `code_asm` assembler, so VEX/ModRM and
//! branch fixups are correct by construction). A vectorizable straight-line loop body is captured as a
//! flat [`VecKernel`] recipe (in `wukong_mir`, backend-agnostic) and this module assembles it into a
//! self-contained function that both the Cranelift JIT/object backend installs (via
//! `define_function_bytes`) and — for the differential oracle — the interpreter marshals lane-wise
//! from the *same* recipe. Both agree bit-for-bit — lane-wise for an elementwise body, and for a
//! reduction because `VecKernel::eval_reduction` and `plan_registers` take the accumulator count from
//! the same `wukong_mir` source, so the two reassociate identically. The gate polices it.
//!
//! Four lane types are covered, each with its own instruction family: `f32` (`*ps`, 8 lanes), `f64`
//! (`*pd`, 4 lanes), and the wrapping integer lanes `i32` (`vpaddd`/`vpsubd`/`vpmulld`, 8 lanes) and
//! `i64` (`vpaddq`/`vpsubq`, 4 lanes — AVX2 has no 64-bit lane multiply, so `pressure()` refuses one).
//! Integer compares are the signed `vpcmpgt*`/`vpcmpeq*` pair; `!=`, `<=` and `>=` are computed as
//! the complement of `==`, `>` and `<` and the consuming blend swaps its operands instead of spending a
//! register on a `not`.
//!
//! ABI of an assembled kernel: `fn(ptrs: *const *mut u8, scalars: *const T, n: u64)`, returning
//! nothing for an elementwise body and the horizontal fold of lane type `T` (XMM0 for a float lane,
//! RAX/EAX for an integer one) for a reduction — the Cranelift side declares the signatures
//! accordingly. `ptrs[k]` is the base of stream `k`, `scalars[k]` the k-th loop-invariant value, `n`
//! the (multiple-of-lanes) element count the caller assigns to the vector part; the caller runs the
//! scalar remainder itself. The kernel makes no calls, so it needs no shadow space and no frame — the
//! only prologue/epilogue is pushing the callee-saved GPRs that carry stream bases five and up
//! (rsi, rdi, rbx, r12-r15), saving the low xmm halves of any of ymm6..ymm15 it allocates (also
//! callee-saved under Win64), and a final `vzeroupper`.
//!
//! An elementwise kernel whose output-only streams are all 32-byte aligned and whose streamed traffic
//! reaches [`NT_MIN_BYTES`] stores those streams with `vmovnt*` (a non-temporal store skips the
//! read-for-ownership a cacheable store pays on a stream larger than the caches), then `sfence`s
//! before returning. The choice is made at run time inside the kernel — the runtime-sized loops that
//! need it most have no compile-time extent — and only selects which store instruction writes the
//! identical values. `WUKONG_P4_NO_NT=1` assembles only the cacheable form (a same-run A/B knob).
//!
//! The body reads its three arguments out of **rcx/rdx/r8** — the Win64 integer argument registers —
//! and executes VEX.256 AVX2 + FMA3 encodings. Neither is negotiable here: the register mapping is
//! literal in the emitted bytes, and there is no runtime dispatch inside a kernel. So
//! [`assemble_kernel`] refuses on any host that does not supply both (see [`host_supports_kernels`]),
//! which is the only CPU-feature gate on this path — the Cranelift side declares the kernel with the
//! module's `default_call_conv` and installs the bytes verbatim, so a mismatch is silent corruption.
//!
//! That refusal is a **backstop, not the policy**: `wukong_mir_build` consults the same predicate
//! before it attaches a recipe at all, so on a host outside the gate the loop simply stays on the
//! 128-bit CLIF strips (the result-identical form `WUKONG_P4_NO_256=1` forces). Reaching
//! [`assemble_kernel`]'s `Err` now means a recipe was built where the vectorizer said it would not
//! be — a bug in that gating, not an expected platform outcome.

// Covers the test-only `assemble_saxpy_probe` reference emitter plus `Plan::hoist_scalars`, which
// the recipe path computes but no longer reads.
#![allow(dead_code)]

use iced_x86::code_asm::*;
use wukong_mir::{VecBin, VecCmp, VecKernel, VecLane, VecOp, VecPressure, VecRedOp};

/// f32 lanes per YMM register (256-bit / 32-bit).
pub const LANES: u32 = wukong_mir::VEC_LANES;
/// Bytes one unrolled copy advances: one YMM register of any lane type.
const GROUP_BYTES: i32 = wukong_mir::VEC_BYTES as i32;
/// Total architectural YMM registers.
const NREG: u32 = wukong_mir::VEC_NREG;

/// Streamed bytes (every stream, loads and stores, over the vector part) from which an elementwise
/// kernel switches its output-only streams to non-temporal stores. The same 10 MiB default the
/// runtime's streaming kernels use (`wukong_runtime::nt_policy`), for the same reason: below the
/// last-level cache a non-temporal store evicts lines a later loop would have hit.
pub const NT_MIN_BYTES: u64 = 10 * 1024 * 1024;

/// Whether this host can run the machine code this module emits: the AVX2 + FMA3 instructions the
/// body encodes, delivered under the Win64 argument-register mapping (rcx/rdx/r8) the body hardcodes.
/// Every other AVX2 consumer in the tree (`wukong_runtime`'s gemm/gemv/attention/… kernels) gates the
/// same way and keeps a scalar twin; this path has no in-kernel fallback, so the gate is the refusal
/// in [`assemble_kernel`].
///
/// The predicate itself is **single-sourced in `wukong_mir`** ([`wukong_mir::host_supports_vec_kernels`]),
/// because the general vectorizer in `wukong_mir_build` has to reach the same answer: it must decline
/// to *build* a recipe this module would then refuse to assemble, or `populate_module` hard-fails the
/// whole compile. Two mirrored `cfg` tests is exactly the shape that has desynced here before, so this
/// stays a delegating wrapper — keep it, the call sites and the docs name it.
pub fn host_supports_kernels() -> bool {
    wukong_mir::host_supports_vec_kernels()
}

/// The Cranelift type a reduction kernel of `lane` returns.
pub fn lane_clif_type(lane: VecLane) -> cranelift_codegen::ir::Type {
    use cranelift_codegen::ir::types;
    match lane {
        VecLane::F32 => types::F32,
        VecLane::F64 => types::F64,
        VecLane::I32 => types::I32,
        VecLane::I64 => types::I64,
    }
}

fn ymm(n: u8) -> AsmRegisterYmm {
    [
        ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13,
        ymm14, ymm15,
    ][n as usize]
}
fn xmm(n: u8) -> AsmRegisterXmm {
    [
        xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13,
        xmm14, xmm15,
    ][n as usize]
}
/// Stream index → its base-pointer GPR, `wukong_mir::VEC_MAX_STREAMS` of them. Loaded once before
/// the loop; `rdx` is reused for stream 3 only after the invariant scalars have been broadcast out of
/// `[rdx]`, and streams [`FIRST_SAVED_STREAM`] and up live in callee-saved registers the prologue
/// pushes. rcx (the element counter), r8 (`n`), rax (scratch), rsp and rbp are never bases.
fn base_gpr(stream: u32) -> AsmRegister64 {
    [r9, r10, r11, rdx, rsi, rdi, rbx, r12, r13, r14, r15][stream as usize]
}
/// The first stream whose base register is callee-saved under Win64 (`rsi`).
const FIRST_SAVED_STREAM: u32 = 4;

/// `vcmpps`/`vcmppd` imm8 predicate for an ordered compare that is false on NaN (except `Ne`,
/// unordered/true on NaN) — matching Rust's `<,<=,>,>=,==,!=` on the interpreter oracle side.
fn cmp_imm(p: VecCmp) -> u32 {
    match p {
        VecCmp::Eq => 0x00, // EQ_OQ
        VecCmp::Lt => 0x01, // LT_OS
        VecCmp::Le => 0x02, // LE_OS
        VecCmp::Ne => 0x04, // NEQ_UQ (unordered ⇒ true on NaN, matching Rust `!=`)
        VecCmp::Ge => 0x0d, // GE_OS
        VecCmp::Gt => 0x0e, // GT_OS
    }
}

/// Does an integer-lane compare compute the *complement* of its predicate (and leave the swap to
/// the consuming blend)? AVX2 has only `==` and signed `>` on integer lanes.
fn int_cmp_inverted(p: VecCmp) -> bool {
    matches!(p, VecCmp::Ne | VecCmp::Le | VecCmp::Ge)
}

/// A register-allocation plan for a kernel body: how many interleaved groups run per main-loop
/// iteration and in how many register banks, the per-group live-register count, and which invariant
/// broadcasts / sign-mask occupy fixed registers.
struct Plan {
    unroll: u32,
    /// Register banks the `unroll` copies cycle through (copy `u` uses bank `u % banks`). Equal to
    /// `unroll` for an elementwise kernel; for a reduction it may be smaller, because the canonical
    /// tree fixes the accumulator count independently of register pressure.
    banks: u32,
    per_group: u8,              // P: registers a single group needs
    hoist_scalars: Vec<u32>,    // distinct invariant scalars, each in a fixed reg
    signmask_reg: Option<u8>,   // fixed reg holding -0.0 lanes, if a float body negates
    scalar_reg: Vec<(u32, u8)>, // scalar index → fixed reg
    acc_regs: Vec<u8>, // reduction accumulators, one per unroll copy (empty if elementwise)
    saved: Vec<u8>,    // xmm6..xmm15 we must preserve (callee-saved on Win64)
    last_use: Vec<usize>, // last_use[i] = last op index referencing value i
    /// `inverted[i]`: op `i` is an integer compare that computed its predicate's complement.
    inverted: Vec<bool>,
    /// Output-only streams (stored, never loaded) an aligned, large call stores non-temporally.
    nt_streams: Vec<u32>,
}

/// Plan the register allocation, or return `Err` (→ the caller keeps the 128-bit fallback) when the
/// body cannot be expressed in this emitter (too many streams, a `Const` the vectorizer should have
/// folded into `scalars`, an op outside its lane's instruction family, or a single group that will
/// not fit in 16 YMM registers).
fn plan_registers(k: &VecKernel) -> Result<Plan, String> {
    // P (peak live), H's parts (hoisted scalars + sign-mask) and last_use come from the
    // backend-agnostic `pressure()` in `wukong_mir`, single-sourced with the vectorizer's gate so a
    // body the vectorizer accepted never fails to assemble here.
    let pressure = k.pressure().ok_or_else(|| {
        format!(
            "avx2: body outside emitter coverage (lane={:?}, streams={}, or Const present)",
            k.lane, k.streams
        )
    })?;
    let VecPressure {
        hoist_scalars,
        needs_signmask,
        last_use,
        per_group,
    } = pressure;
    // Prove the bound in u32 and narrow only after: a `per_group` of 256 truncates to 0 (integer
    // divide by zero below), 257 truncates to 1 (`free.pop()` panics), so a body that overflows must
    // become the promised `Err` and not a panic reaching the user through the driver.
    let hoist = hoist_scalars.len() as u32 + needs_signmask as u32;
    if per_group == 0 || per_group + hoist > NREG {
        return Err(format!(
            "avx2: body needs {per_group}+{hoist} > {NREG} registers"
        ));
    }
    // A reduction additionally needs one accumulator register per unrolled copy; an elementwise
    // kernel needs none. The accumulator count comes from `wukong_mir` (the canonical tree, which the
    // interpreter's `eval_reduction` reproduces), and so does the bank count, so the two agree.
    let is_reduction = k.reduce.is_some();
    let (unroll, banks) = if is_reduction {
        if !k.reduction_fits() {
            return Err(format!(
                "avx2: reduction body needs {per_group}+{}+{hoist} > {NREG} registers",
                k.reduction_unroll()
            ));
        }
        (k.reduction_unroll(), k.reduction_banks())
    } else {
        // Interleave as many groups as fit (dense from ymm0), capped by the requested unroll.
        let u = k.unroll.max(1).min((NREG - hoist) / per_group).max(1);
        (u, u)
    };
    let (per_group, hoist) = (per_group as u8, hoist as u8);
    debug_assert!(banks as u8 * per_group + hoist + is_reduction as u8 * unroll as u8 <= 16);

    // Register assignment (dense, low first, to minimize callee-saved spills): body banks occupy
    // ymm[0 .. banks*P); a reduction's accumulators occupy the next `unroll`; the hoisted broadcasts
    // occupy the `hoist` after that.
    let mut next = banks as u8 * per_group;
    let acc_regs: Vec<u8> = if is_reduction {
        (0..unroll as u8).map(|c| next + c).collect()
    } else {
        Vec::new()
    };
    next += acc_regs.len() as u8;
    let mut scalar_reg = Vec::new();
    for &s in &hoist_scalars {
        scalar_reg.push((s, next));
        next += 1;
    }
    let signmask_reg = if needs_signmask {
        let r = next;
        next += 1;
        Some(r)
    } else {
        None
    };
    // Any register in [6,15] we touch is callee-saved on Win64 → preserve its low xmm half.
    let top = next; // one past the highest register used
    let saved: Vec<u8> = (6..top).collect();

    let inverted: Vec<bool> = k
        .ops
        .iter()
        .map(|o| match *o {
            VecOp::Cmp { pred, .. } => !k.lane.is_float() && int_cmp_inverted(pred),
            _ => false,
        })
        .collect();
    // A stream qualifies for a non-temporal store only if the body never reads it: a load of a
    // just-stored stream (the recipe's store-to-load forwarding) or an in-place update would read a
    // line the non-temporal store is evicting. Reductions store nothing.
    let nt_streams: Vec<u32> = if is_reduction || std::env::var_os("WUKONG_P4_NO_NT").is_some() {
        Vec::new()
    } else {
        (0..k.streams)
            .filter(|&s| {
                k.ops
                    .iter()
                    .any(|o| matches!(*o, VecOp::Store { stream, .. } if stream == s))
                    && !k
                        .ops
                        .iter()
                        .any(|o| matches!(*o, VecOp::Load { stream } if stream == s))
            })
            .collect()
    };

    Ok(Plan {
        unroll,
        banks,
        per_group,
        hoist_scalars,
        signmask_reg,
        scalar_reg,
        acc_regs,
        saved,
        last_use,
        inverted,
        nt_streams,
    })
}

/// Assemble `k` into a self-contained AVX2 function `fn(ptrs, scalars, n)` (Win64 ABI: rcx, rdx,
/// r8). Returns the raw machine code, or `Err` if the host cannot run these bytes at all, or if the
/// body is outside this emitter's coverage.
pub fn assemble_kernel(k: &VecKernel) -> Result<Vec<u8>, String> {
    assemble_kernel_with_nt_threshold(k, NT_MIN_BYTES)
}

/// [`assemble_kernel`] with an explicit non-temporal threshold in streamed bytes — how the tests put
/// a small, aligned buffer on the non-temporal path.
pub(crate) fn assemble_kernel_with_nt_threshold(
    k: &VecKernel,
    nt_min_bytes: u64,
) -> Result<Vec<u8>, String> {
    // The ISA/ABI gate comes first: the bytes below are AVX2+FMA3 under the Win64 argument
    // registers, with no in-kernel dispatch. Emitting them for a host that lacks either is a #UD at
    // the first `vmovups ymm` or a wild dereference of whatever rcx held — neither diagnosable.
    if !host_supports_kernels() {
        return Err(
            "avx2: host lacks AVX2+FMA3 under the Win64 ABI these kernels encode".to_string(),
        );
    }
    let plan = plan_registers(k)?;
    emit(k, &plan, nt_min_bytes).map_err(|e| format!("avx2 encode: {e}"))
}

fn emit(k: &VecKernel, plan: &Plan, nt_min_bytes: u64) -> Result<Vec<u8>, IcedError> {
    let mut a = CodeAssembler::new(64)?;
    let lane = k.lane;

    // --- Prologue: push the callee-saved GPRs that hold stream bases 4.., then preserve the
    // callee-saved xmm6..xmm15 we use (low 128 bits). No calls are made, so no shadow space;
    // vmovups avoids any stack-alignment requirement. ---
    let saved_gprs: Vec<AsmRegister64> = (FIRST_SAVED_STREAM..k.streams).map(base_gpr).collect();
    for &r in &saved_gprs {
        a.push(r)?;
    }
    let save_bytes = plan.saved.len() as i32 * 16;
    if save_bytes > 0 {
        a.sub(rsp, save_bytes)?;
        for (i, &r) in plan.saved.iter().enumerate() {
            a.vmovups(xmmword_ptr(rsp + i as i32 * 16), xmm(r))?;
        }
    }

    // rcx = ptrs, rdx = scalars, r8 = n. Load every stream base out of `ptrs` except stream 3,
    // whose register (rdx) still addresses the scalars.
    for s in 0..k.streams {
        if s != 3 {
            a.mov(base_gpr(s), qword_ptr(rcx + (8 * s) as i32))?;
        }
    }
    // Broadcast the invariant scalars (still readable via rdx) into their fixed registers.
    let scalar_bytes = lane.bytes() as i32;
    for &(sidx, reg) in &plan.scalar_reg {
        emit_broadcast(&mut a, lane, reg, rdx + sidx as i32 * scalar_bytes)?;
    }
    // Build the -0.0 sign mask in-register (only the sign bit set per lane) for a float negation.
    if let Some(r) = plan.signmask_reg {
        a.vpcmpeqd(ymm(r), ymm(r), ymm(r))?; // all ones
        if lane == VecLane::F64 {
            a.vpsllq(ymm(r), ymm(r), 63u32)?; // → 0x8000000000000000 per lane
        } else {
            a.vpslld(ymm(r), ymm(r), 31u32)?; // → 0x80000000 per lane
        }
    }
    // Initialise the reduction accumulators to the fold identity (all lanes): 0 for a sum, −∞/+∞
    // for max/min so empty tail lanes never win. The ±∞ patterns are built in-register from
    // all-ones (no data section): f32 −∞ = 0xFF800000 = all-ones << 23, +∞ = (all-ones >> 24) << 23;
    // f64 −∞ = 0xFFF0… = all-ones << 52, +∞ = (all-ones >> 53) << 52.
    if let Some(red) = k.reduce {
        for &r in &plan.acc_regs {
            let y = ymm(r);
            match (red.op, lane) {
                (VecRedOp::Add, VecLane::F32) => a.vxorps(y, y, y)?,
                (VecRedOp::Add, VecLane::F64) => a.vxorpd(y, y, y)?,
                (VecRedOp::Add, VecLane::I32 | VecLane::I64) => a.vpxor(y, y, y)?,
                (VecRedOp::Fmax, VecLane::F32) => {
                    a.vpcmpeqd(y, y, y)?;
                    a.vpslld(y, y, 23u32)?; // → 0xFF800000 = −∞
                }
                (VecRedOp::Fmin, VecLane::F32) => {
                    a.vpcmpeqd(y, y, y)?;
                    a.vpsrld(y, y, 24u32)?; // → 0x000000FF
                    a.vpslld(y, y, 23u32)?; // → 0x7F800000 = +∞
                }
                (VecRedOp::Fmax, VecLane::F64) => {
                    a.vpcmpeqd(y, y, y)?;
                    a.vpsllq(y, y, 52u32)?; // → 0xFFF0000000000000 = −∞
                }
                (VecRedOp::Fmin, VecLane::F64) => {
                    a.vpcmpeqd(y, y, y)?;
                    a.vpsrlq(y, y, 53u32)?; // → 0x00000000000007FF
                    a.vpsllq(y, y, 52u32)?; // → 0x7FF0000000000000 = +∞
                }
                (VecRedOp::Fmax | VecRedOp::Fmin, VecLane::I32 | VecLane::I64) => {
                    unreachable!("validate() refuses a max/min fold on an integer lane")
                }
            }
        }
    }
    // The 4th stream base reuses rdx now that the scalars have been read.
    if k.streams > 3 {
        a.mov(rdx, qword_ptr(rcx + 24))?;
    }

    // --- The loops. An elementwise kernel with output-only streams carries a second copy whose
    // stores to those streams are non-temporal, entered when the call streams at least
    // `nt_min_bytes` and every such stream's base is 32-byte aligned (`vmovnt*` faults otherwise;
    // every in-loop offset is a multiple of 32, so the base decides). ---
    let mut finish = a.create_label();
    if !plan.nt_streams.is_empty() {
        let mut cached = a.create_label();
        let traffic_per_elem = k.streams as u64 * lane.bytes() as u64;
        let nt_min_elems = nt_min_bytes
            .div_ceil(traffic_per_elem)
            .clamp(1, i32::MAX as u64) as i32;
        a.cmp(r8, nt_min_elems)?;
        a.jb(cached)?;
        a.mov(rax, base_gpr(plan.nt_streams[0]))?;
        for &s in &plan.nt_streams[1..] {
            a.or(rax, base_gpr(s))?;
        }
        a.test(eax, 31i32)?;
        a.jne(cached)?;
        let mut nt_done = a.create_label();
        emit_loops(&mut a, k, plan, true, &mut nt_done)?;
        // Order the weakly-ordered non-temporal stores before anything the caller stores next
        // (a `@parallel` chunk's completion flag, say).
        a.sfence()?;
        a.jmp(finish)?;
        a.set_label(&mut cached)?;
    }
    emit_loops(&mut a, k, plan, false, &mut finish)?;

    // --- Reduction finish: combine the accumulators into acc[0], then a sequential lane-0..L
    // horizontal fold into the return register. Done before the epilogue restores the callee-saved
    // xmm halves, since an accumulator may live in one of them. ---
    if let Some(red) = k.reduce {
        let acc0 = plan.acc_regs[0];
        for &r in &plan.acc_regs[1..] {
            fold_ymm(&mut a, lane, red.op, acc0, r)?;
        }
        // Spill the lanes and fold them in order; a local 32-byte scratch keeps rsp balanced.
        a.sub(rsp, GROUP_BYTES)?;
        emit_store(&mut a, lane, ymmword_ptr(rsp), acc0, false)?;
        match lane {
            VecLane::F32 => {
                a.vmovss(xmm(0), dword_ptr(rsp))?;
                for l in 1..lane.lanes() as i32 {
                    fold_ss(&mut a, red.op, dword_ptr(rsp + l * 4))?;
                }
            }
            VecLane::F64 => {
                a.vmovsd(xmm(0), qword_ptr(rsp))?;
                for l in 1..lane.lanes() as i32 {
                    fold_sd(&mut a, red.op, qword_ptr(rsp + l * 8))?;
                }
            }
            // An integer reduction is a wrapping sum (`validate()`), returned in eax/rax.
            VecLane::I32 => {
                a.mov(eax, dword_ptr(rsp))?;
                for l in 1..lane.lanes() as i32 {
                    a.add(eax, dword_ptr(rsp + l * 4))?;
                }
            }
            VecLane::I64 => {
                a.mov(rax, qword_ptr(rsp))?;
                for l in 1..lane.lanes() as i32 {
                    a.add(rax, qword_ptr(rsp + l * 8))?;
                }
            }
        }
        a.add(rsp, GROUP_BYTES)?;
    }

    // --- Epilogue. ---
    if save_bytes > 0 {
        for (i, &r) in plan.saved.iter().enumerate() {
            a.vmovups(xmm(r), xmmword_ptr(rsp + i as i32 * 16))?;
        }
        a.add(rsp, save_bytes)?;
    }
    for &r in saved_gprs.iter().rev() {
        a.pop(r)?;
    }
    a.vzeroupper()?;
    a.ret()?;
    a.assemble(0x0)
}

/// The main loop (`plan.unroll` copies per iteration, when that is more than one) and the
/// single-group cleanup loop, over `[0, n)` with `rcx` counting elements. `done` is set on the
/// first instruction after the cleanup loop. `nt` selects non-temporal stores for
/// `plan.nt_streams`.
fn emit_loops(
    a: &mut CodeAssembler,
    k: &VecKernel,
    plan: &Plan,
    nt: bool,
    done: &mut CodeLabel,
) -> Result<(), IcedError> {
    let lanes = k.lane.lanes();
    a.xor(rcx, rcx)?; // i = 0 (rcx is free now that ptrs is consumed)

    // The single-vector loop head is the main loop's exit target too, so it carries exactly one
    // label (iced forbids two labels on one instruction).
    let mut single_head = a.create_label();

    // --- Unrolled main loop (only when it does more than the single-vector loop). ---
    if plan.unroll > 1 {
        let step = (plan.unroll * lanes) as i32;
        let mut head = a.create_label();
        a.set_label(&mut head)?;
        a.lea(rax, qword_ptr(rcx + step))?;
        a.cmp(rax, r8)?;
        a.ja(single_head)?;
        for u in 0..plan.unroll {
            emit_group(
                a,
                k,
                plan,
                u % plan.banks,
                u as i32 * GROUP_BYTES,
                plan.acc_regs.get(u as usize).copied(),
                nt,
            )?;
        }
        a.add(rcx, step)?;
        a.jmp(head)?;
    }

    // --- Single-vector cleanup loop (step = lanes); a reduction folds into accumulator 0. ---
    a.set_label(&mut single_head)?;
    a.lea(rax, qword_ptr(rcx + lanes as i32))?;
    a.cmp(rax, r8)?;
    a.ja(*done)?;
    emit_group(a, k, plan, 0, 0, plan.acc_regs.first().copied(), nt)?;
    a.add(rcx, lanes as i32)?;
    a.jmp(single_head)?;
    a.set_label(done)?;
    Ok(())
}

/// Emit one lane-group of the body: unroll copy at byte displacement `disp` (a multiple of 32) from
/// `base + i*lane_bytes`, in register bank `bank` (its own `per_group` registers). Loads/stores are
/// unit-stride.
///
/// Allocation rule (uniform, hazard-free): resolve operand registers, allocate the result register
/// from a free list that still excludes every live operand, emit, *then* free operands whose last
/// use is this op. So the result never aliases a still-needed operand — even across the two-op FMA
/// (`vmovaps` + `vfmadd231ps`) or an integer negation (`vpxor` + `vpsub`). `per_group` was sized to
/// admit result-plus-live-operands, so the free list never underflows.
fn emit_group(
    a: &mut CodeAssembler,
    k: &VecKernel,
    plan: &Plan,
    bank: u32,
    disp: i32,
    acc_reg: Option<u8>,
    nt: bool,
) -> Result<(), IcedError> {
    let lane = k.lane;
    let scale = lane.bytes();
    let bank_base = bank as u8 * plan.per_group;
    let mut free: Vec<u8> = (0..plan.per_group).map(|r| bank_base + r).collect();
    let mut vreg: Vec<Option<u8>> = vec![None; k.ops.len()];
    // The element of stream `s` this group touches.
    let at = |s: u32| ymmword_ptr(base_gpr(s) + rcx * scale + disp);

    // The register currently holding value `v` (a hoisted scalar resolves to its fixed register).
    let reg_of = |vreg: &[Option<u8>], v: u32| -> u8 {
        match k.ops[v as usize] {
            VecOp::Splat { scalar } => plan
                .scalar_reg
                .iter()
                .find(|(s, _)| *s == scalar)
                .map(|(_, r)| *r)
                .expect("hoisted scalar reg"),
            _ => vreg[v as usize].expect("value has a register"),
        }
    };
    // Free value `v`'s body register after op `i`, if this is its last use. An op may name the same
    // value twice (`a*a`, or an `Fma` sharing an operand), so this runs more than once for one value
    // — `take()` makes the release idempotent. Without it the register is pushed twice and two later
    // allocations receive the same physical register, the second silently clobbering the first.
    // Taking is sound: `free_after` only fires when `last_use[v] == i`, so no later op reads
    // `vreg[v]`; the reduction fold's addend / fused operands have `last_use == ops.len()` (set by
    // `pressure()`), which equals no op index, so their registers survive to the post-loop fold.
    let free_after = |free: &mut Vec<u8>, vreg: &mut [Option<u8>], i: usize, v: u32| {
        if plan.last_use[v as usize] == i
            && !matches!(k.ops[v as usize], VecOp::Splat { .. } | VecOp::Const { .. })
        {
            if let Some(r) = vreg[v as usize].take() {
                debug_assert!(!free.contains(&r), "avx2: ymm{r} released twice at op {i}");
                free.push(r);
            }
        }
    };

    for (i, op) in k.ops.iter().enumerate() {
        // Stores and hoisted values need no result register.
        if let VecOp::Store { stream, val } = *op {
            let rv = reg_of(&vreg, val);
            let streaming = nt && plan.nt_streams.contains(&stream);
            emit_store(a, lane, at(stream), rv, streaming)?;
            free_after(&mut free, &mut vreg, i, val);
            continue;
        }
        if matches!(op, VecOp::Splat { .. } | VecOp::Const { .. }) {
            continue; // hoisted before the loop
        }
        // Resolve operands, allocate a fresh result register (excludes all live operands), emit.
        match *op {
            VecOp::Load { stream } => {
                let d = free.pop().expect("free reg");
                emit_load(a, lane, d, at(stream))?;
                vreg[i] = Some(d);
            }
            VecOp::Bin { op, a: x, b: y } => {
                let (rx, ry) = (reg_of(&vreg, x), reg_of(&vreg, y));
                let d = free.pop().expect("free reg");
                emit_bin(a, lane, op, d, rx, ry)?;
                vreg[i] = Some(d);
                free_after(&mut free, &mut vreg, i, x);
                free_after(&mut free, &mut vreg, i, y);
            }
            VecOp::Fma { a: x, b: y, c } => {
                let (rx, ry, rc) = (reg_of(&vreg, x), reg_of(&vreg, y), reg_of(&vreg, c));
                let d = free.pop().expect("free reg");
                match lane {
                    VecLane::F32 => {
                        a.vmovaps(ymm(d), ymm(rc))?;
                        a.vfmadd231ps(ymm(d), ymm(rx), ymm(ry))?; // d = x*y + c
                    }
                    VecLane::F64 => {
                        a.vmovapd(ymm(d), ymm(rc))?;
                        a.vfmadd231pd(ymm(d), ymm(rx), ymm(ry))?; // d = x*y + c
                    }
                    VecLane::I32 | VecLane::I64 => {
                        unreachable!("validate() refuses Fma on an integer lane")
                    }
                }
                vreg[i] = Some(d);
                free_after(&mut free, &mut vreg, i, x);
                free_after(&mut free, &mut vreg, i, y);
                free_after(&mut free, &mut vreg, i, c);
            }
            VecOp::Sqrt { a: x } => {
                let rx = reg_of(&vreg, x);
                let d = free.pop().expect("free reg");
                match lane {
                    VecLane::F32 => a.vsqrtps(ymm(d), ymm(rx))?,
                    VecLane::F64 => a.vsqrtpd(ymm(d), ymm(rx))?,
                    VecLane::I32 | VecLane::I64 => {
                        unreachable!("validate() refuses Sqrt on an integer lane")
                    }
                }
                vreg[i] = Some(d);
                free_after(&mut free, &mut vreg, i, x);
            }
            VecOp::Neg { a: x } => {
                let rx = reg_of(&vreg, x);
                let d = free.pop().expect("free reg");
                match lane {
                    VecLane::F32 => {
                        let sm = plan.signmask_reg.expect("signmask reg");
                        a.vxorps(ymm(d), ymm(rx), ymm(sm))?;
                    }
                    VecLane::F64 => {
                        let sm = plan.signmask_reg.expect("signmask reg");
                        a.vxorpd(ymm(d), ymm(rx), ymm(sm))?;
                    }
                    // `0 - x`, wrapping (so `-MIN == MIN`, as `wrapping_neg`). `d` excludes the
                    // live `x`, so zeroing it first cannot destroy the operand.
                    VecLane::I32 => {
                        a.vpxor(ymm(d), ymm(d), ymm(d))?;
                        a.vpsubd(ymm(d), ymm(d), ymm(rx))?;
                    }
                    VecLane::I64 => {
                        a.vpxor(ymm(d), ymm(d), ymm(d))?;
                        a.vpsubq(ymm(d), ymm(d), ymm(rx))?;
                    }
                }
                vreg[i] = Some(d);
                free_after(&mut free, &mut vreg, i, x);
            }
            VecOp::Cmp { pred, a: x, b: y } => {
                let (rx, ry) = (reg_of(&vreg, x), reg_of(&vreg, y));
                let d = free.pop().expect("free reg");
                emit_cmp(a, lane, pred, d, rx, ry)?;
                vreg[i] = Some(d);
                free_after(&mut free, &mut vreg, i, x);
                free_after(&mut free, &mut vreg, i, y);
            }
            VecOp::Select { mask, a: x, b: y } => {
                let (rm, rx, ry) = (reg_of(&vreg, mask), reg_of(&vreg, x), reg_of(&vreg, y));
                let d = free.pop().expect("free reg");
                // An inverted integer compare holds the complement mask, so the blend picks the
                // other operand: `!m ? x : y` is `m ? y : x`.
                let (rx, ry) = if plan.inverted[mask as usize] {
                    (ry, rx)
                } else {
                    (rx, ry)
                };
                // vblendv* dst, src1, src2, mask ⇒ dst = mask ? src2 : src1. We want mask ? x : y.
                match lane {
                    VecLane::F32 => a.vblendvps(ymm(d), ymm(ry), ymm(rx), ymm(rm))?,
                    VecLane::F64 => a.vblendvpd(ymm(d), ymm(ry), ymm(rx), ymm(rm))?,
                    VecLane::I32 | VecLane::I64 => {
                        a.vpblendvb(ymm(d), ymm(ry), ymm(rx), ymm(rm))?
                    }
                }
                vreg[i] = Some(d);
                free_after(&mut free, &mut vreg, i, mask);
                free_after(&mut free, &mut vreg, i, x);
                free_after(&mut free, &mut vreg, i, y);
            }
            VecOp::Store { .. } | VecOp::Splat { .. } | VecOp::Const { .. } => unreachable!(),
        }
    }
    // A reduction folds this group's addend into this copy's accumulator. A fused product folds in one
    // rounding via `vfmadd231ps`/`pd` (acc += X*Y), matching `eval_reduction`'s `mul_add`; otherwise
    // the plain `fold` of the addend value.
    if let (Some(acc), Some(red)) = (acc_reg, k.reduce) {
        match red.fma {
            Some((x, y)) => {
                let (rx, ry) = (reg_of(&vreg, x), reg_of(&vreg, y));
                match lane {
                    VecLane::F32 => a.vfmadd231ps(ymm(acc), ymm(rx), ymm(ry))?, // acc = X*Y + acc
                    VecLane::F64 => a.vfmadd231pd(ymm(acc), ymm(rx), ymm(ry))?,
                    VecLane::I32 | VecLane::I64 => {
                        unreachable!("validate() refuses a fused integer reduction")
                    }
                }
            }
            None => {
                let rv = vreg[red.value as usize].expect("reduce value has a register");
                fold_ymm(a, lane, red.op, acc, rv)?;
            }
        }
    }
    Ok(())
}

/// A unit-stride 256-bit load of lane type `lane` into `ymm(d)`.
fn emit_load(
    a: &mut CodeAssembler,
    lane: VecLane,
    d: u8,
    at: AsmMemoryOperand,
) -> Result<(), IcedError> {
    match lane {
        VecLane::F32 => a.vmovups(ymm(d), at),
        VecLane::F64 => a.vmovupd(ymm(d), at),
        VecLane::I32 | VecLane::I64 => a.vmovdqu(ymm(d), at),
    }
}

/// A unit-stride 256-bit store of `ymm(v)`; `streaming` selects the non-temporal form (whose address
/// must be 32-byte aligned).
fn emit_store(
    a: &mut CodeAssembler,
    lane: VecLane,
    at: AsmMemoryOperand,
    v: u8,
    streaming: bool,
) -> Result<(), IcedError> {
    match (lane, streaming) {
        (VecLane::F32, false) => a.vmovups(at, ymm(v)),
        (VecLane::F32, true) => a.vmovntps(at, ymm(v)),
        (VecLane::F64, false) => a.vmovupd(at, ymm(v)),
        (VecLane::F64, true) => a.vmovntpd(at, ymm(v)),
        (VecLane::I32 | VecLane::I64, false) => a.vmovdqu(at, ymm(v)),
        (VecLane::I32 | VecLane::I64, true) => a.vmovntdq(at, ymm(v)),
    }
}

/// Broadcast one invariant scalar of lane type `lane` at `at` into every lane of `ymm(d)`.
fn emit_broadcast(
    a: &mut CodeAssembler,
    lane: VecLane,
    d: u8,
    at: AsmMemoryOperand,
) -> Result<(), IcedError> {
    match lane {
        VecLane::F32 => a.vbroadcastss(ymm(d), dword_ptr(at)),
        VecLane::F64 => a.vbroadcastsd(ymm(d), qword_ptr(at)),
        VecLane::I32 => a.vpbroadcastd(ymm(d), dword_ptr(at)),
        VecLane::I64 => a.vpbroadcastq(ymm(d), qword_ptr(at)),
    }
}

/// `ymm(d) = ymm(x) ⊕ ymm(y)` in the lane type's instruction family.
fn emit_bin(
    a: &mut CodeAssembler,
    lane: VecLane,
    op: VecBin,
    d: u8,
    x: u8,
    y: u8,
) -> Result<(), IcedError> {
    let (d, x, y) = (ymm(d), ymm(x), ymm(y));
    match (lane, op) {
        (VecLane::F32, VecBin::Add) => a.vaddps(d, x, y),
        (VecLane::F32, VecBin::Sub) => a.vsubps(d, x, y),
        (VecLane::F32, VecBin::Mul) => a.vmulps(d, x, y),
        (VecLane::F32, VecBin::Div) => a.vdivps(d, x, y),
        (VecLane::F64, VecBin::Add) => a.vaddpd(d, x, y),
        (VecLane::F64, VecBin::Sub) => a.vsubpd(d, x, y),
        (VecLane::F64, VecBin::Mul) => a.vmulpd(d, x, y),
        (VecLane::F64, VecBin::Div) => a.vdivpd(d, x, y),
        (VecLane::I32, VecBin::Add) => a.vpaddd(d, x, y),
        (VecLane::I32, VecBin::Sub) => a.vpsubd(d, x, y),
        (VecLane::I32, VecBin::Mul) => a.vpmulld(d, x, y),
        (VecLane::I64, VecBin::Add) => a.vpaddq(d, x, y),
        (VecLane::I64, VecBin::Sub) => a.vpsubq(d, x, y),
        (VecLane::I64, VecBin::Mul) => unreachable!("pressure() refuses an i64 multiply"),
        (VecLane::I32 | VecLane::I64, VecBin::Div) => {
            unreachable!("validate() refuses an integer division")
        }
    }
}

/// `ymm(d) = mask(x pred y)`. A float compare is one `vcmpps`/`vcmppd`; an integer one is the
/// signed `>` or `==` of the right operand order, computing the *complement* for `!=`, `<=` and `>=`
/// (see [`int_cmp_inverted`] — the consuming `Select` swaps its operands).
fn emit_cmp(
    a: &mut CodeAssembler,
    lane: VecLane,
    pred: VecCmp,
    d: u8,
    x: u8,
    y: u8,
) -> Result<(), IcedError> {
    let (d, x, y) = (ymm(d), ymm(x), ymm(y));
    match lane {
        VecLane::F32 => a.vcmpps(d, x, y, cmp_imm(pred)),
        VecLane::F64 => a.vcmppd(d, x, y, cmp_imm(pred)),
        VecLane::I32 => match pred {
            VecCmp::Gt | VecCmp::Le => a.vpcmpgtd(d, x, y), // x > y  (Le: ¬(x > y))
            VecCmp::Lt | VecCmp::Ge => a.vpcmpgtd(d, y, x), // y > x  (Ge: ¬(y > x))
            VecCmp::Eq | VecCmp::Ne => a.vpcmpeqd(d, x, y), // x == y (Ne: ¬(x == y))
        },
        VecLane::I64 => match pred {
            VecCmp::Gt | VecCmp::Le => a.vpcmpgtq(d, x, y),
            VecCmp::Lt | VecCmp::Ge => a.vpcmpgtq(d, y, x),
            VecCmp::Eq | VecCmp::Ne => a.vpcmpeqq(d, x, y),
        },
    }
}

/// `acc = fold(acc, src)` on 256-bit lanes (`acc` = src1, so `vmaxps`/`vminps` keep src2 on
/// ties/NaN, matching [`VecRedOp::fold`]).
fn fold_ymm(
    a: &mut CodeAssembler,
    lane: VecLane,
    op: VecRedOp,
    acc: u8,
    src: u8,
) -> Result<(), IcedError> {
    let (d, s) = (ymm(acc), ymm(src));
    match (lane, op) {
        (VecLane::F32, VecRedOp::Add) => a.vaddps(d, d, s),
        (VecLane::F32, VecRedOp::Fmax) => a.vmaxps(d, d, s),
        (VecLane::F32, VecRedOp::Fmin) => a.vminps(d, d, s),
        (VecLane::F64, VecRedOp::Add) => a.vaddpd(d, d, s),
        (VecLane::F64, VecRedOp::Fmax) => a.vmaxpd(d, d, s),
        (VecLane::F64, VecRedOp::Fmin) => a.vminpd(d, d, s),
        (VecLane::I32, VecRedOp::Add) => a.vpaddd(d, d, s),
        (VecLane::I64, VecRedOp::Add) => a.vpaddq(d, d, s),
        (VecLane::I32 | VecLane::I64, VecRedOp::Fmax | VecRedOp::Fmin) => {
            unreachable!("validate() refuses a max/min fold on an integer lane")
        }
    }
}

/// `xmm0 = fold(xmm0, [mem])` on one f32 scalar (the horizontal reduction step).
fn fold_ss(a: &mut CodeAssembler, op: VecRedOp, src: AsmMemoryOperand) -> Result<(), IcedError> {
    match op {
        VecRedOp::Add => a.addss(xmm(0), src),
        VecRedOp::Fmax => a.maxss(xmm(0), src),
        VecRedOp::Fmin => a.minss(xmm(0), src),
    }
}

/// `xmm0 = fold(xmm0, [mem])` on one f64 scalar (the horizontal reduction step).
fn fold_sd(a: &mut CodeAssembler, op: VecRedOp, src: AsmMemoryOperand) -> Result<(), IcedError> {
    match op {
        VecRedOp::Add => a.addsd(xmm(0), src),
        VecRedOp::Fmax => a.maxsd(xmm(0), src),
        VecRedOp::Fmin => a.minsd(xmm(0), src),
    }
}

/// Bring-up kernel (Phase A): hand-written saxpy `out[i] = a*x[i] + y[i]` over `n` (multiple of 8)
/// elements, `ptrs = [x, y, out]`, `scalars = [a]`. Proves the encode → `define_function_bytes` →
/// finalize → call → result seam end-to-end with a real 256-bit `vfmadd` before the recipe compiler
/// exists. Returns the raw machine code.
pub fn assemble_saxpy_probe() -> Result<Vec<u8>, IcedError> {
    let mut a = CodeAssembler::new(64)?;

    // Win64 argument registers: rcx = ptrs, rdx = scalars, r8 = n — the same hardcoded mapping the
    // recipe emitter uses, and the reason `host_supports_kernels` refuses non-Windows hosts. The
    // caller drives this through the host C ABI directly, so it is only valid where that gate passes.
    a.mov(r9, qword_ptr(rcx))?; // x   = ptrs[0]
    a.mov(r10, qword_ptr(rcx + 8))?; // y   = ptrs[1]
    a.mov(r11, qword_ptr(rcx + 16))?; // out = ptrs[2]
    a.vbroadcastss(ymm5, dword_ptr(rdx))?; // ymm5 = splat(a)
    a.xor(rcx, rcx)?; // i = 0 (reuse rcx as the counter now that ptrs is consumed)

    let mut head = a.create_label();
    let mut done = a.create_label();
    a.set_label(&mut head)?;
    a.lea(rax, qword_ptr(rcx + 8))?; // rax = i + 8
    a.cmp(rax, r8)?;
    a.ja(done)?; // if i+8 > n, exit (unsigned)
    a.vmovups(ymm0, ymmword_ptr(r9 + rcx * 4))?; // x[i..i+8]
    a.vmovups(ymm1, ymmword_ptr(r10 + rcx * 4))?; // y[i..i+8]
    a.vfmadd213ps(ymm0, ymm5, ymm1)?; // ymm0 = ymm0*ymm5 + ymm1 = a*x + y
    a.vmovups(ymmword_ptr(r11 + rcx * 4), ymm0)?; // out[i..i+8]
    a.add(rcx, 8)?;
    a.jmp(head)?;
    a.set_label(&mut done)?;
    a.vzeroupper()?;
    a.ret()?;

    a.assemble(0x0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::ir::{AbiParam, Signature};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};

    // Include both sides of the eight-lane and four-vector unroll boundaries.
    const BOUNDARY_LENGTHS: [usize; 17] = [
        0, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64, 65,
    ];

    /// Writable redzones detect stray stores, not overreads. Keep a full four-vector group on
    /// either side, including for empty streams. Compare raw bits so NaN payloads and signed zero
    /// cannot hide corruption; the payload itself also has to match through the untouched tail.
    struct GuardedStream {
        storage: Vec<f32>,
        len: usize,
    }

    impl GuardedStream {
        const GUARD: usize = 32;

        fn canary(index: usize) -> u32 {
            0x7fc0_5a00 | index as u32
        }

        fn new(values: &[f32]) -> Self {
            let mut storage: Vec<f32> = (0..values.len() + 2 * Self::GUARD)
                .map(|i| f32::from_bits(Self::canary(i)))
                .collect();
            storage[Self::GUARD..Self::GUARD + values.len()].copy_from_slice(values);
            Self {
                storage,
                len: values.len(),
            }
        }

        fn as_mut_ptr(&mut self) -> *mut f32 {
            self.storage.as_mut_ptr().wrapping_add(Self::GUARD)
        }

        fn payload(&self) -> &[f32] {
            &self.storage[Self::GUARD..Self::GUARD + self.len]
        }

        fn assert_matches(&self, expected: &[f32], label: &str) {
            assert_eq!(self.len, expected.len());
            for (i, value) in self.storage.iter().enumerate() {
                if i < Self::GUARD || i >= Self::GUARD + self.len {
                    assert_eq!(value.to_bits(), Self::canary(i), "{label}: guard word {i}");
                }
            }
            for (i, (got, want)) in self.payload().iter().zip(expected).enumerate() {
                assert_eq!(got.to_bits(), want.to_bits(), "{label}: payload lane {i}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "guard word")]
    fn guarded_stream_detects_understore() {
        let mut stream = GuardedStream::new(&[1.0; 8]);
        stream.storage[GuardedStream::GUARD - 1] =
            f32::from_bits(GuardedStream::canary(GuardedStream::GUARD - 1) ^ 1);
        stream.assert_matches(&[1.0; 8], "injected understore");
    }

    #[test]
    #[should_panic(expected = "guard word")]
    fn guarded_stream_detects_overstore() {
        let mut stream = GuardedStream::new(&[1.0; 8]);
        let end = GuardedStream::GUARD + stream.len;
        stream.storage[end] = f32::from_bits(GuardedStream::canary(end) ^ 1);
        stream.assert_matches(&[1.0; 8], "injected overstore");
    }

    #[test]
    #[should_panic(expected = "payload lane 8")]
    fn guarded_stream_detects_tail_write() {
        let mut expected = vec![1.0; 9];
        expected[8] = -0.0;
        let mut stream = GuardedStream::new(&expected);
        stream.storage[GuardedStream::GUARD + 8] = 0.0;
        stream.assert_matches(&expected, "injected tail write");
    }

    /// Phase A: assemble the saxpy kernel, install its raw bytes via `define_function_bytes`, call
    /// it, and check it computed `a*x + y` in true 256-bit strides. This is the whole raw-AVX2 seam.
    #[test]
    fn avx2_saxpy_kernel_runs() {
        if !host_supports_kernels() {
            return; // these bytes are not executable here — see `host_supports_kernels`
        }
        eprintln!("raw AVX2 execution enabled: Win64 + AVX2 + FMA host gate passed");
        let bytes = assemble_saxpy_probe().expect("assemble");
        assert!(!bytes.is_empty());
        if std::env::var("P4_DUMP").is_ok() {
            let path = std::env::var("P4_DUMP").unwrap();
            std::fs::write(&path, &bytes).unwrap();
            eprintln!("wrote {} bytes to {path}", bytes.len());
        }

        let isa = crate::make_isa(false).expect("isa");
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);
        let ptr = module.target_config().pointer_type();
        let mut sig = Signature::new(module.target_config().default_call_conv);
        sig.params.push(AbiParam::new(ptr)); // ptrs
        sig.params.push(AbiParam::new(ptr)); // scalars
        sig.params
            .push(AbiParam::new(cranelift_codegen::ir::types::I64)); // n
        let fid = module
            .declare_function("saxpy_probe", Linkage::Export, &sig)
            .unwrap();
        module
            .define_function_bytes(fid, 16, &bytes, &[])
            .expect("install raw bytes");
        module.finalize_definitions().expect("finalize");
        let code = module.get_finalized_function(fid);

        // fn(ptrs: *const *mut f32, scalars: *const f32, n: u64)
        let kernel: extern "C" fn(*const *mut f32, *const f32, u64) =
            unsafe { std::mem::transmute(code) };

        for n in BOUNDARY_LENGTHS {
            let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let y: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();
            let initial_out = vec![f32::from_bits(0xffc0_1234); n];
            let mut guarded_x = GuardedStream::new(&x);
            let mut guarded_y = GuardedStream::new(&y);
            let mut out = GuardedStream::new(&initial_out);
            let mut a_scalar = GuardedStream::new(&[3.0]);
            let ptrs = [
                guarded_x.as_mut_ptr(),
                guarded_y.as_mut_ptr(),
                out.as_mut_ptr(),
            ];
            kernel(ptrs.as_ptr(), a_scalar.as_mut_ptr(), n as u64);

            let mut expected = initial_out;
            for i in 0..n / 8 * 8 {
                let want = 3.0 * x[i] + y[i];
                assert_eq!(
                    out.payload()[i],
                    want,
                    "lane {i}: got {}, want {want}",
                    out.payload()[i]
                );
                expected[i] = want;
            }
            out.assert_matches(&expected, "saxpy output");
            guarded_x.assert_matches(&x, "saxpy x");
            guarded_y.assert_matches(&y, "saxpy y");
            a_scalar.assert_matches(&[3.0], "saxpy scalar");
        }
        unsafe { module.free_memory() };
    }

    /// Assemble `k`, run it on `init` streams / `scalars` over `n` elements, and assert it matches
    /// the `VecKernel::eval_lane` reference **bit-for-bit** on every full-vector lane `[0, n/8*8)`.
    /// Also assert every remaining lane and both redzones are untouched. Inputs may alias outputs
    /// (in-place); the reference mutates its own copy the same way.
    fn check(k: &VecKernel, init: &[Vec<f32>], scalars: &[f32], n: usize) {
        if !host_supports_kernels() {
            return; // these bytes are not executable here — see `host_supports_kernels`
        }
        let bytes = assemble_kernel(k).expect("assemble");

        // Reference: run eval_lane over its own mutable copy of the streams.
        let mut refs: Vec<Vec<f32>> = init.to_vec();
        let vlen = n / LANES as usize * LANES as usize;
        for i in 0..vlen {
            // snapshot loads for this lane before any store (SSA order: loads precede stores).
            let snap = refs.clone();
            k.eval_lane(
                |s| snap[s as usize][i],
                |c| scalars[c as usize],
                |s, v| refs[s as usize][i] = v,
            );
        }

        // Native: install the raw bytes and call on a fresh mutable copy.
        let mut got: Vec<GuardedStream> = init.iter().map(|v| GuardedStream::new(v)).collect();
        let mut guarded_scalars = GuardedStream::new(scalars);
        let isa = crate::make_isa(false).expect("isa");
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);
        let ptr = module.target_config().pointer_type();
        let mut sig = Signature::new(module.target_config().default_call_conv);
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.params
            .push(AbiParam::new(cranelift_codegen::ir::types::I64));
        let fid = module.declare_function("k", Linkage::Export, &sig).unwrap();
        module.define_function_bytes(fid, 16, &bytes, &[]).unwrap();
        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(fid);
        let kernel: extern "C" fn(*const *mut f32, *const f32, u64) =
            unsafe { std::mem::transmute(code) };
        let ptrs: Vec<*mut f32> = got.iter_mut().map(|v| v.as_mut_ptr()).collect();
        kernel(ptrs.as_ptr(), guarded_scalars.as_mut_ptr(), n as u64);

        for s in 0..k.streams as usize {
            for (i, expected) in refs[s][..vlen].iter().enumerate() {
                assert_eq!(
                    got[s].payload()[i].to_bits(),
                    expected.to_bits(),
                    "kernel vs eval_lane mismatch: stream {s} lane {i} (got {}, want {})",
                    got[s].payload()[i],
                    expected
                );
            }
            got[s].assert_matches(&refs[s], &format!("stream {s}, n={n}"));
        }
        guarded_scalars.assert_matches(scalars, "scalars");
        unsafe { module.free_memory() };
    }

    /// Assemble a *reduction* kernel `k`, run it over `n/8*8` elements of `streams`/`scalars`, and
    /// assert its f32 return matches [`VecKernel::eval_reduction`] **bit-for-bit** — the same
    /// reassociation on both. Reduction kernels only read their streams, so `*const` suffices.
    fn check_reduce(k: &VecKernel, streams: &[Vec<f32>], scalars: &[f32], n: usize) {
        if !host_supports_kernels() {
            return; // these bytes are not executable here — see `host_supports_kernels`
        }
        let bytes = assemble_kernel(k).expect("assemble reduction");
        let vlen = n / LANES as usize * LANES as usize;
        let want = k.eval_reduction(vlen, |s, e| streams[s as usize][e], |c| scalars[c as usize]);

        let isa = crate::make_isa(false).expect("isa");
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);
        let ptr = module.target_config().pointer_type();
        let mut sig = Signature::new(module.target_config().default_call_conv);
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.params
            .push(AbiParam::new(cranelift_codegen::ir::types::I64));
        sig.returns
            .push(AbiParam::new(cranelift_codegen::ir::types::F32));
        let fid = module
            .declare_function("kr", Linkage::Export, &sig)
            .unwrap();
        module.define_function_bytes(fid, 16, &bytes, &[]).unwrap();
        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(fid);
        let kernel: extern "C" fn(*const *const f32, *const f32, u64) -> f32 =
            unsafe { std::mem::transmute(code) };
        let mut guarded: Vec<GuardedStream> =
            streams.iter().map(|v| GuardedStream::new(v)).collect();
        let mut guarded_scalars = GuardedStream::new(scalars);
        let ptrs: Vec<*const f32> = guarded
            .iter_mut()
            .map(|v| v.as_mut_ptr().cast_const())
            .collect();
        let got = kernel(ptrs.as_ptr(), guarded_scalars.as_mut_ptr(), vlen as u64);
        for (s, (stream, original)) in guarded.iter().zip(streams).enumerate() {
            stream.assert_matches(original, &format!("reduction stream {s}, n={n}"));
        }
        guarded_scalars.assert_matches(scalars, "reduction scalars");
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "reduction kernel vs eval_reduction (n={n}): got {got}, want {want}"
        );
        unsafe { module.free_memory() };
    }

    // Deterministic inputs that exercise negatives, zeros, and non-round values (bit-exact vs the
    // f32 reference regardless of "ugliness", including sqrt(<0)=NaN and x/0=inf).
    fn ramp(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.375 - 5.0 + seed).collect()
    }

    #[test]
    fn avx2_reduce_sum_and_dot() {
        // sum: acc += a[i]
        let sum = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 1,
            scalars: 0,
            unroll: 4,
            reduce: Some(wukong_mir::VecReduce {
                op: VecRedOp::Add,
                value: 0,
                fma: None,
            }),
            ops: vec![VecOp::Load { stream: 0 }],
        };
        // dot: acc += a[i]*b[i]
        let dot = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 2,
            scalars: 0,
            unroll: 4,
            reduce: Some(wukong_mir::VecReduce {
                op: VecRedOp::Add,
                value: 2,
                fma: None,
            }),
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 0,
                    b: 1,
                },
            ],
        };
        for n in [8usize, 16, 24, 32, 64, 128, 256, 512]
            .into_iter()
            .chain(BOUNDARY_LENGTHS)
        {
            check_reduce(&sum, &[ramp(n, 0.0)], &[], n);
            check_reduce(&dot, &[ramp(n, 0.0), ramp(n, 1.0)], &[], n);
        }
    }

    #[test]
    fn avx2_reduce_fma_fused() {
        // Fused dot: acc = fma(a[i], b[i], acc) — the product is *not* a recipe op; the two loads are
        // the fma operands (kept live to the fold). One rounding, so it must match eval_reduction's
        // mul_add bit-for-bit (that agreement is the whole point of the shared reference).
        let fdot = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 2,
            scalars: 0,
            unroll: 4,
            reduce: Some(wukong_mir::VecReduce {
                op: VecRedOp::Add,
                value: 0,
                fma: Some((0, 1)),
            }),
            ops: vec![VecOp::Load { stream: 0 }, VecOp::Load { stream: 1 }],
        };
        // Fused product of two composite operands `(a+b)*(a-b)`: the loads feed both operands, so their
        // last-use must extend past the two arithmetic ops to the fold (exercises the liveness guard).
        let fcomposite = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 2,
            scalars: 0,
            unroll: 4,
            reduce: Some(wukong_mir::VecReduce {
                op: VecRedOp::Add,
                value: 2,
                fma: Some((2, 3)),
            }),
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 0,
                    b: 1,
                }, // X = a+b
                VecOp::Bin {
                    op: VecBin::Sub,
                    a: 0,
                    b: 1,
                }, // Y = a-b
            ],
        };
        for n in [8usize, 16, 24, 32, 64, 128, 256, 512] {
            check_reduce(&fdot, &[ramp(n, 0.0), ramp(n, 1.0)], &[], n);
            check_reduce(&fcomposite, &[ramp(n, 0.0), ramp(n, 1.0)], &[], n);
        }
    }

    #[test]
    fn avx2_reduce_sumsq_and_weighted() {
        // sum of squares: acc += a[i]*a[i]
        let ssq = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 1,
            scalars: 0,
            unroll: 4,
            reduce: Some(wukong_mir::VecReduce {
                op: VecRedOp::Add,
                value: 1,
                fma: None,
            }),
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 0,
                    b: 0,
                },
            ],
        };
        // weighted sum: acc += a[i] * k  (a hoisted scalar in the addend)
        let wsum = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 1,
            scalars: 1,
            unroll: 4,
            reduce: Some(wukong_mir::VecReduce {
                op: VecRedOp::Add,
                value: 2,
                fma: None,
            }),
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Splat { scalar: 0 },
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 0,
                    b: 1,
                },
            ],
        };
        for n in [8usize, 16, 24, 40, 96, 256] {
            check_reduce(&ssq, &[ramp(n, 0.0)], &[], n);
            check_reduce(&wsum, &[ramp(n, 0.0)], &[2.5], n);
        }
    }

    #[test]
    fn avx2_reduce_fmax_fmin() {
        let mk = |op| VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 1,
            scalars: 0,
            unroll: 4,
            reduce: Some(wukong_mir::VecReduce {
                op,
                value: 0,
                fma: None,
            }),
            ops: vec![VecOp::Load { stream: 0 }],
        };
        let fmax = mk(VecRedOp::Fmax);
        let fmin = mk(VecRedOp::Fmin);
        for n in [8usize, 16, 24, 32, 64, 200, 256] {
            // ramps with both signs so the extreme isn't always at an edge lane.
            check_reduce(&fmax, &[ramp(n, 0.0)], &[], n);
            check_reduce(&fmin, &[ramp(n, 3.0)], &[], n);
        }
    }

    fn sym() -> wukong_span::Symbol {
        wukong_span::Interner::new().intern("k")
    }

    #[test]
    fn avx2_kernel_saxpy_all_widths() {
        // out = a*x + y
        let k = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 3,
            scalars: 1,
            unroll: 4,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Splat { scalar: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Fma { a: 0, b: 1, c: 2 },
                VecOp::Store { stream: 2, val: 3 },
            ],
        };
        for n in [8usize, 16, 24, 32, 40, 64, 80, 100]
            .into_iter()
            .chain(BOUNDARY_LENGTHS)
        {
            check(&k, &[ramp(n, 0.0), ramp(n, 1.0), vec![0.0; n]], &[2.5], n);
        }
    }

    #[test]
    fn avx2_kernel_diff_of_squares_p4() {
        // out = (x+y)*(x-y). Peak live = 4 registers.
        let k = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 3,
            scalars: 0,
            unroll: 4,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 0,
                    b: 1,
                },
                VecOp::Bin {
                    op: VecBin::Sub,
                    a: 0,
                    b: 1,
                },
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 2,
                    b: 3,
                },
                VecOp::Store { stream: 2, val: 4 },
            ],
        };
        for n in [8usize, 16, 24, 48, 96] {
            check(&k, &[ramp(n, 0.0), ramp(n, 2.0), vec![0.0; n]], &[], n);
        }
    }

    #[test]
    fn avx2_kernel_sqrt_div() {
        // out = x / sqrt(y)  (y may be negative ⇒ NaN, or zero ⇒ inf — must match bit-for-bit)
        let k = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 3,
            scalars: 0,
            unroll: 2,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Sqrt { a: 1 },
                VecOp::Bin {
                    op: VecBin::Div,
                    a: 0,
                    b: 2,
                },
                VecOp::Store { stream: 2, val: 3 },
            ],
        };
        for n in [8usize, 24, 64] {
            check(&k, &[ramp(n, 0.0), ramp(n, 0.0), vec![0.0; n]], &[], n);
        }
    }

    #[test]
    fn avx2_kernel_store_to_load_forward() {
        // A fused body `t = 2*x + 1; o = if t > 0 { t } else { 0 }`: stream 1 (`t`) is STORED then
        // LOADED twice in the same lane, so the loads must observe the just-stored value. The
        // assembler forwards through memory; `eval_lane` forwards internally — this test pins that
        // they agree. `t` is pre-seeded with poison the forwarding must overwrite (without it, the
        // relu would read poison and the reference would diverge from the machine code).
        let k = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 3, // 0 = x (in), 1 = t (in-place scratch), 2 = o (out)
            scalars: 3, // 0 = 2.0, 1 = 1.0, 2 = 0.0
            unroll: 4,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },  // v0 = x
                VecOp::Splat { scalar: 0 }, // v1 = 2.0
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 0,
                    b: 1,
                }, // v2 = 2*x
                VecOp::Splat { scalar: 1 }, // v3 = 1.0
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 2,
                    b: 3,
                }, // v4 = 2*x + 1
                VecOp::Store { stream: 1, val: 4 }, // t = 2*x + 1
                VecOp::Load { stream: 1 },  // v6 = t   (forwarded)
                VecOp::Splat { scalar: 2 }, // v7 = 0.0
                VecOp::Cmp {
                    pred: VecCmp::Gt,
                    a: 6,
                    b: 7,
                }, // v8 = t > 0
                VecOp::Load { stream: 1 },  // v9 = t   (forwarded again)
                VecOp::Select {
                    mask: 8,
                    a: 9,
                    b: 7,
                }, // v10 = t>0 ? t : 0
                VecOp::Store { stream: 2, val: 10 }, // o = relu(t)
            ],
        };
        for n in [8usize, 16, 40, 96] {
            // stream 1 poison-seeded (-999); forwarding must overwrite it before the relu reads it.
            check(
                &k,
                &[ramp(n, 0.0), vec![-999.0; n], vec![0.0; n]],
                &[2.0, 1.0, 0.0],
                n,
            );
        }
    }

    #[test]
    fn avx2_kernel_neg_signmask() {
        // out = (-x) + y  (sign-bit flip, bit-exact for ±0)
        let k = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 3,
            scalars: 0,
            unroll: 4,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Neg { a: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 1,
                    b: 2,
                },
                VecOp::Store { stream: 2, val: 3 },
            ],
        };
        for n in [8usize, 40] {
            check(&k, &[ramp(n, 0.0), ramp(n, 1.0), vec![0.0; n]], &[], n);
        }
    }

    #[test]
    fn avx2_kernel_relu_cmp_select() {
        // out = if x > 0 { x } else { 0 }  (vcmpps + vblendvps)
        let k = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 2,
            scalars: 1,
            unroll: 4,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Splat { scalar: 0 }, // 0.0
                VecOp::Cmp {
                    pred: VecCmp::Gt,
                    a: 0,
                    b: 1,
                },
                VecOp::Select {
                    mask: 2,
                    a: 0,
                    b: 1,
                },
                VecOp::Store { stream: 1, val: 3 },
            ],
        };
        for n in [8usize, 24, 72] {
            check(&k, &[ramp(n, 0.0), vec![0.0; n]], &[0.0], n);
        }
    }

    #[test]
    fn avx2_kernel_inplace_and_four_streams() {
        // in-place scale x[i] *= s, plus a separate 4-stream body to exercise rdx-as-base.
        let k = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 1,
            scalars: 1,
            unroll: 4,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Splat { scalar: 0 },
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 0,
                    b: 1,
                },
                VecOp::Store { stream: 0, val: 2 },
            ],
        };
        check(&k, &[ramp(64, 0.0)], &[1.5], 64);

        // out = w*x + z*y  → streams x,y,z,w? No: distinct streams x(0),y(1),out(2),plus scalars.
        // Use 4 streams: a,b,c,out with out = a*b + c ... plus a 4th read stream d unused-as-out.
        let kfour = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 4,
            scalars: 0,
            unroll: 2,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Load { stream: 2 },
                VecOp::Fma { a: 0, b: 1, c: 2 }, // a*b + c
                VecOp::Store { stream: 3, val: 3 },
            ],
        };
        check(
            &kfour,
            &[ramp(48, 0.0), ramp(48, 1.0), ramp(48, 2.0), vec![0.0; 48]],
            &[],
            48,
        );
    }

    /// A value used TWICE by one op (`a*a`) must be released exactly once. The free list is a plain
    /// `Vec`, so a double-push hands the same physical register to two later allocations and the
    /// second silently clobbers the first — `o = a*a + b*c` then computes `a*a + c*c`. Three shapes,
    /// each with the repeated operand followed by further allocating ops so the duplicate is actually
    /// handed out: two loads, another repeated-operand op, and a fused `Fma` (whose three operands
    /// give the widest double-free surface).
    #[test]
    fn avx2_kernel_repeated_operand_frees_once() {
        // o = a*a + b*c   (dup-operand op, then two loads that pop the duplicate)
        let dup_then_loads = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 4,
            scalars: 0,
            unroll: 4,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 }, // v0 = a
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 0,
                    b: 0,
                }, // v1 = a*a
                VecOp::Load { stream: 1 }, // v2 = b
                VecOp::Load { stream: 2 }, // v3 = c
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 2,
                    b: 3,
                }, // v4 = b*c
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 1,
                    b: 4,
                }, // v5 = a*a + b*c
                VecOp::Store { stream: 3, val: 5 },
            ],
        };
        // o = a*a + b*b + b   (dup-operand op, then a second dup-operand op)
        let dup_then_dup = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 3,
            scalars: 0,
            unroll: 4,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 }, // v0 = a
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 0,
                    b: 0,
                }, // v1 = a*a
                VecOp::Load { stream: 1 }, // v2 = b
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 2,
                    b: 2,
                }, // v3 = b*b
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 1,
                    b: 3,
                }, // v4 = a*a + b*b
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 4,
                    b: 2,
                }, // v5 = … + b
                VecOp::Store { stream: 2, val: 5 },
            ],
        };
        // o = (a*a)*c + b*c  — `c` feeds an Fma and a later Bin, so the Fma's three `free_after`
        // calls run while `c` is still live; `a` is the repeated operand.
        let dup_then_fma = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 4,
            scalars: 0,
            unroll: 2,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 }, // v0 = a
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 0,
                    b: 0,
                }, // v1 = a*a
                VecOp::Load { stream: 1 }, // v2 = b
                VecOp::Load { stream: 2 }, // v3 = c
                VecOp::Fma { a: 1, b: 3, c: 2 }, // v4 = (a*a)*c + b
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 4,
                    b: 3,
                }, // v5 = v4 * c
                VecOp::Store { stream: 3, val: 5 },
            ],
        };
        for n in [8usize, 16, 24, 64] {
            let s = |seed| ramp(n, seed);
            check(
                &dup_then_loads,
                &[s(0.0), s(1.0), s(2.0), vec![0.0; n]],
                &[],
                n,
            );
            check(&dup_then_dup, &[s(0.0), s(1.0), vec![0.0; n]], &[], n);
            check(
                &dup_then_fma,
                &[s(0.0), s(1.0), s(2.0), vec![0.0; n]],
                &[],
                n,
            );
        }
        // Same defect on the reduction path: the fold reads its addend after the body, so a register
        // handed out twice corrupts the accumulator. `acc += (a*a) * (b*c)`.
        let red = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 3,
            scalars: 0,
            unroll: 4,
            reduce: Some(wukong_mir::VecReduce {
                op: VecRedOp::Add,
                value: 5,
                fma: None,
            }),
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 0,
                    b: 0,
                }, // a*a
                VecOp::Load { stream: 1 },
                VecOp::Load { stream: 2 },
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 2,
                    b: 3,
                }, // b*c
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 1,
                    b: 4,
                },
            ],
        };
        for n in [8usize, 16, 64, 256] {
            check_reduce(&red, &[ramp(n, 0.0), ramp(n, 1.0), ramp(n, 2.0)], &[], n);
        }
    }

    /// `assemble_kernel` promises `Err` for a body it cannot express, and callers rely on that (the
    /// message reaches the user through the driver; a panic would not). A body with exactly 256 live
    /// values used to narrow to `per_group == 0` *before* the bound was checked, so it slipped past
    /// the register guard and divided by zero computing the unroll. The vectorizer's own u32 gate
    /// rejects such a body first today, but this entry point must stand on its own.
    #[test]
    fn avx2_kernel_bails_on_register_overflow() {
        let mut ops: Vec<VecOp> = (0..256).map(|_| VecOp::Load { stream: 0 }).collect();
        ops.push(VecOp::Store { stream: 1, val: 0 });
        let k = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 2,
            scalars: 0,
            unroll: 1,
            reduce: None,
            ops,
        };
        assert_eq!(k.pressure().map(|p| p.per_group), Some(256), "test premise");
        let err = assemble_kernel(&k).expect_err("256 live values must bail, not panic");
        assert!(
            err.contains("registers") || err.contains("Win64"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn avx2_kernel_bails_past_max_streams() {
        let streams = wukong_mir::VEC_MAX_STREAMS + 1;
        let k = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams,
            scalars: 0,
            unroll: 1,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Store {
                    stream: streams - 1,
                    val: 0,
                },
            ],
        };
        assert!(
            assemble_kernel(&k).is_err(),
            "{streams} streams must bail to fallback"
        );
    }

    /// Five and more streams — the fused-epilogue shapes (`o = a*b + c*d`, residual + scale + bias)
    /// that used to drop to 128-bit at the old four-register limit — run in the callee-saved
    /// rsi/rdi/rbx/r12-r15 the prologue pushes. Every stream count from 5 to the maximum is
    /// assembled and checked against the reference, so each newly used base register (and its
    /// push/pop pairing) is exercised.
    #[test]
    fn avx2_kernel_uses_every_stream_register() {
        for streams in 5..=wukong_mir::VEC_MAX_STREAMS {
            // o = x0 + x1*x2 + x3*... : a chain over every input, so every base is read.
            let inputs = streams - 1;
            let mut ops = vec![VecOp::Load { stream: 0 }];
            let mut acc = 0u32;
            let mut s = 1;
            while s < inputs {
                ops.push(VecOp::Load { stream: s });
                let x = ops.len() as u32 - 1;
                if s + 1 < inputs {
                    ops.push(VecOp::Load { stream: s + 1 });
                    let y = ops.len() as u32 - 1;
                    ops.push(VecOp::Fma { a: x, b: y, c: acc });
                    s += 2;
                } else {
                    ops.push(VecOp::Bin {
                        op: VecBin::Sub,
                        a: acc,
                        b: x,
                    });
                    s += 1;
                }
                acc = ops.len() as u32 - 1;
            }
            ops.push(VecOp::Store {
                stream: inputs,
                val: acc,
            });
            let k = VecKernel {
                name: sym(),
                lane: VecLane::F32,
                streams,
                scalars: 0,
                unroll: 4,
                reduce: None,
                ops,
            };
            assert!(
                k.pressure().is_some_and(|p| p.fits()),
                "{streams} streams must be inside the emitter's coverage"
            );
            for n in [8usize, 17, 40, 64] {
                let mut init: Vec<Vec<f32>> = (0..inputs).map(|s| ramp(n, s as f32)).collect();
                init.push(vec![0.0; n]);
                check(&k, &init, &[], n);
            }
        }
    }

    /// The canonical tree fixes a reduction's accumulator count, so a body too wide for four banks
    /// shares register banks between unrolled copies instead of dropping accumulators. This is the
    /// pressure body `reduction_partition` pins: P=6, H=4 → one bank, four accumulators.
    #[test]
    fn avx2_reduce_shares_banks_but_keeps_four_accumulators() {
        // acc = fma(a, (b+1)*(b+2) + (b+3)*(b+4), acc)
        let k = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 2,
            scalars: 4,
            unroll: 4,
            reduce: Some(wukong_mir::VecReduce {
                op: VecRedOp::Add,
                value: 0,
                fma: Some((0, 11)),
            }),
            ops: vec![
                VecOp::Load { stream: 0 },  // 0: a
                VecOp::Load { stream: 1 },  // 1: b
                VecOp::Splat { scalar: 0 }, // 2
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 1,
                    b: 2,
                }, // 3: b+1
                VecOp::Splat { scalar: 1 }, // 4
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 1,
                    b: 4,
                }, // 5: b+2
                VecOp::Splat { scalar: 2 }, // 6
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 1,
                    b: 6,
                }, // 7: b+3
                VecOp::Splat { scalar: 3 }, // 8
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 1,
                    b: 8,
                }, // 9: b+4
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 7,
                    b: 9,
                }, // 10
                VecOp::Fma { a: 3, b: 5, c: 10 }, // 11
            ],
        };
        assert_eq!(
            k.reduction_unroll(),
            4,
            "the canonical tree has four accumulators"
        );
        assert_eq!(k.reduction_banks(), 1, "test premise: one bank only");
        assert!(k.reduction_fits());
        for n in [8usize, 16, 32, 40, 64, 96, 200, 256] {
            check_reduce(&k, &[ramp(n, 0.0), ramp(n, 1.0)], &[1.0, 2.0, 3.0, 4.0], n);
        }
    }

    // ---- lane-generic harness: f64 / i32 / i64 (and f32 on the non-temporal path) ----

    /// One lane type under test: its raw bits (compared exactly, so NaN payloads and signed zero
    /// cannot hide a mismatch), a redzone pattern, and a typed call of an assembled reduction.
    trait TestLane: wukong_mir::VecValue + std::fmt::Debug {
        fn bits(self) -> u64;
        fn canary(i: usize) -> Self;
        /// Call reduction `code` (Win64 `fn(ptrs, scalars, n) -> Self`).
        unsafe fn call_reduce(
            code: *const u8,
            ptrs: *const *const u8,
            sc: *const u8,
            n: u64,
        ) -> Self;
    }
    impl TestLane for f32 {
        fn bits(self) -> u64 {
            self.to_bits() as u64
        }
        fn canary(i: usize) -> Self {
            f32::from_bits(0x7fc0_5a00 | i as u32)
        }
        unsafe fn call_reduce(
            code: *const u8,
            ptrs: *const *const u8,
            sc: *const u8,
            n: u64,
        ) -> Self {
            let f: extern "C" fn(*const *const u8, *const u8, u64) -> f32 =
                std::mem::transmute(code);
            f(ptrs, sc, n)
        }
    }
    impl TestLane for f64 {
        fn bits(self) -> u64 {
            self.to_bits()
        }
        fn canary(i: usize) -> Self {
            f64::from_bits(0x7ff8_5a5a_0000_0000 | i as u64)
        }
        unsafe fn call_reduce(
            code: *const u8,
            ptrs: *const *const u8,
            sc: *const u8,
            n: u64,
        ) -> Self {
            let f: extern "C" fn(*const *const u8, *const u8, u64) -> f64 =
                std::mem::transmute(code);
            f(ptrs, sc, n)
        }
    }
    impl TestLane for i32 {
        fn bits(self) -> u64 {
            self as u32 as u64
        }
        fn canary(i: usize) -> Self {
            0x5a5a_0000 | i as i32
        }
        unsafe fn call_reduce(
            code: *const u8,
            ptrs: *const *const u8,
            sc: *const u8,
            n: u64,
        ) -> Self {
            let f: extern "C" fn(*const *const u8, *const u8, u64) -> i32 =
                std::mem::transmute(code);
            f(ptrs, sc, n)
        }
    }
    impl TestLane for i64 {
        fn bits(self) -> u64 {
            self as u64
        }
        fn canary(i: usize) -> Self {
            0x5a5a_5a5a_0000_0000 | i as i64
        }
        unsafe fn call_reduce(
            code: *const u8,
            ptrs: *const *const u8,
            sc: *const u8,
            n: u64,
        ) -> Self {
            let f: extern "C" fn(*const *const u8, *const u8, u64) -> i64 =
                std::mem::transmute(code);
            f(ptrs, sc, n)
        }
    }

    /// A stream with redzones whose payload starts at a chosen residue modulo 32 bytes — 0 for the
    /// alignment a non-temporal store needs, one lane for an address that must keep the cacheable
    /// path.
    struct Placed<T: TestLane> {
        storage: Vec<T>,
        start: usize,
        len: usize,
    }

    impl<T: TestLane> Placed<T> {
        const GUARD: usize = 16;

        fn new(values: &[T], misaligned: bool) -> Self {
            let size = std::mem::size_of::<T>();
            let slack = 32 / size;
            let mut storage: Vec<T> = (0..values.len() + 2 * Self::GUARD + 2 * slack)
                .map(T::canary)
                .collect();
            let base = storage.as_ptr() as usize;
            let want = if misaligned { size } else { 0 };
            let start = (Self::GUARD..Self::GUARD + slack)
                .find(|&s| (base + s * size) % 32 == want)
                .expect("an element-aligned buffer reaches every residue");
            storage[start..start + values.len()].copy_from_slice(values);
            Placed {
                storage,
                start,
                len: values.len(),
            }
        }

        fn ptr(&mut self) -> *mut u8 {
            self.storage.as_mut_ptr().wrapping_add(self.start).cast()
        }

        fn assert_matches(&self, expected: &[T], label: &str) {
            assert_eq!(self.len, expected.len());
            for (i, v) in self.storage.iter().enumerate() {
                if i < self.start || i >= self.start + self.len {
                    assert_eq!(v.bits(), T::canary(i).bits(), "{label}: redzone word {i}");
                }
            }
            for (i, (got, want)) in self.storage[self.start..self.start + self.len]
                .iter()
                .zip(expected)
                .enumerate()
            {
                assert_eq!(
                    got.bits(),
                    want.bits(),
                    "{label}: lane {i}: got {got:?}, want {want:?}"
                );
            }
        }
    }

    /// Install raw kernel bytes in a fresh JIT module. The caller frees the module after the call.
    fn install(bytes: &[u8]) -> (JITModule, *const u8) {
        let isa = crate::make_isa(false).expect("isa");
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);
        let ptr = module.target_config().pointer_type();
        let mut sig = Signature::new(module.target_config().default_call_conv);
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.params
            .push(AbiParam::new(cranelift_codegen::ir::types::I64));
        let fid = module
            .declare_function("lane_kernel", Linkage::Export, &sig)
            .unwrap();
        module.define_function_bytes(fid, 16, &bytes, &[]).unwrap();
        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(fid);
        (module, code)
    }

    /// Run elementwise kernel `k` (lane type `T`) natively over `n` elements and compare every
    /// vector-part lane with `eval_lane` bit-for-bit, and every tail lane and redzone with its
    /// original contents. `nt_min_bytes` is the non-temporal threshold to assemble with;
    /// `misaligned` offsets every stream by one lane from a 32-byte boundary.
    fn check_lanes<T: TestLane>(
        k: &VecKernel,
        init: &[Vec<T>],
        scalars: &[T],
        n: usize,
        nt_min_bytes: u64,
        misaligned: bool,
    ) {
        if !host_supports_kernels() {
            return;
        }
        assert_eq!(T::LANE, k.lane, "harness lane type");
        let bytes = assemble_kernel_with_nt_threshold(k, nt_min_bytes).expect("assemble");
        let lanes = k.lane.lanes() as usize;
        let vlen = n / lanes * lanes;
        let mut refs: Vec<Vec<T>> = init.to_vec();
        for i in 0..vlen {
            let snap: Vec<T> = refs.iter().map(|s| s[i]).collect();
            k.eval_lane(
                |s| snap[s as usize],
                |c| scalars[c as usize],
                |s, v| refs[s as usize][i] = v,
            );
        }
        let mut streams: Vec<Placed<T>> = init.iter().map(|v| Placed::new(v, misaligned)).collect();
        let mut sc = Placed::new(scalars, false);
        let ptrs: Vec<*mut u8> = streams.iter_mut().map(|s| s.ptr()).collect();
        let (module, code) = install(&bytes);
        let kernel: extern "C" fn(*const *mut u8, *const u8, u64) =
            unsafe { std::mem::transmute(code) };
        kernel(ptrs.as_ptr(), sc.ptr(), vlen as u64);
        unsafe { module.free_memory() };
        for (s, stream) in streams.iter().enumerate() {
            stream.assert_matches(
                &refs[s],
                &format!("{:?} stream {s} n={n} misaligned={misaligned}", k.lane),
            );
        }
        sc.assert_matches(scalars, "scalars");
    }

    /// Run reduction kernel `k` natively and compare its return with `eval_reduction` bit-for-bit.
    fn check_lane_reduce<T: TestLane>(k: &VecKernel, streams: &[Vec<T>], scalars: &[T], n: usize) {
        if !host_supports_kernels() {
            return;
        }
        assert_eq!(T::LANE, k.lane, "harness lane type");
        let bytes = assemble_kernel(k).expect("assemble reduction");
        let lanes = k.lane.lanes() as usize;
        let vlen = n / lanes * lanes;
        let want = k.eval_reduction(vlen, |s, e| streams[s as usize][e], |c| scalars[c as usize]);
        let mut placed: Vec<Placed<T>> = streams.iter().map(|v| Placed::new(v, true)).collect();
        let mut sc = Placed::new(scalars, false);
        let ptrs: Vec<*const u8> = placed.iter_mut().map(|s| s.ptr().cast_const()).collect();
        let (module, code) = install(&bytes);
        let got =
            unsafe { T::call_reduce(code, ptrs.as_ptr(), sc.ptr().cast_const(), vlen as u64) };
        unsafe { module.free_memory() };
        for (s, (p, original)) in placed.iter().zip(streams).enumerate() {
            p.assert_matches(original, &format!("{:?} reduction stream {s}", k.lane));
        }
        assert_eq!(
            got.bits(),
            want.bits(),
            "{:?} reduction n={n}: got {got:?}, want {want:?}",
            k.lane
        );
    }

    const LANE_LENGTHS: [usize; 14] = [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 64, 100];

    fn f64_input(n: usize, seed: f64) -> Vec<f64> {
        // Signs, zeros of both signs, a NaN and infinities among ordinary non-round values.
        (0..n)
            .map(|i| match (i + seed as usize) % 11 {
                0 => -0.0,
                1 => 0.0,
                5 => f64::NAN,
                7 => f64::INFINITY,
                _ => (i as f64) * 0.375 - 5.0 + seed,
            })
            .collect()
    }

    fn int_input<T: TryFrom<i64> + Copy>(n: usize, seed: i64, extremes: [T; 2]) -> Vec<T>
    where
        <T as TryFrom<i64>>::Error: std::fmt::Debug,
    {
        (0..n)
            .map(|i| match (i as i64 + seed) % 9 {
                0 => extremes[0],
                4 => extremes[1],
                r => T::try_from((i as i64) * 37 - 900 + r * seed).unwrap(),
            })
            .collect()
    }

    /// The body vocabulary over f64: fused multiply-add, division, square root (of negatives too),
    /// negation and a select on every compare predicate — each against `eval_lane::<f64>`.
    #[test]
    fn avx2_f64_lanes_match_reference() {
        let fma_div_sqrt = VecKernel {
            name: sym(),
            lane: VecLane::F64,
            streams: 4,
            scalars: 1,
            unroll: 4,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },       // 0: a
                VecOp::Load { stream: 1 },       // 1: b
                VecOp::Load { stream: 2 },       // 2: c
                VecOp::Fma { a: 0, b: 1, c: 2 }, // 3: a*b + c
                VecOp::Sqrt { a: 2 },            // 4: sqrt(c)
                VecOp::Splat { scalar: 0 },      // 5: k
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 4,
                    b: 5,
                }, // 6: sqrt(c) + k
                VecOp::Bin {
                    op: VecBin::Div,
                    a: 3,
                    b: 6,
                }, // 7
                VecOp::Neg { a: 7 },             // 8
                VecOp::Store { stream: 3, val: 8 },
            ],
        };
        for n in LANE_LENGTHS {
            let init = vec![
                f64_input(n, 0.0),
                f64_input(n, 1.0),
                f64_input(n, 2.0),
                vec![0.0; n],
            ];
            check_lanes(&fma_div_sqrt, &init, &[1.5], n, NT_MIN_BYTES, true);
        }
        for pred in [
            VecCmp::Lt,
            VecCmp::Le,
            VecCmp::Gt,
            VecCmp::Ge,
            VecCmp::Eq,
            VecCmp::Ne,
        ] {
            let k = VecKernel {
                name: sym(),
                lane: VecLane::F64,
                streams: 3,
                scalars: 0,
                unroll: 4,
                reduce: None,
                ops: vec![
                    VecOp::Load { stream: 0 },
                    VecOp::Load { stream: 1 },
                    VecOp::Cmp { pred, a: 0, b: 1 },
                    VecOp::Select {
                        mask: 2,
                        a: 0,
                        b: 1,
                    },
                    VecOp::Store { stream: 2, val: 3 },
                ],
            };
            for n in LANE_LENGTHS {
                let init = vec![f64_input(n, 0.0), f64_input(n, 3.0), vec![0.0; n]];
                check_lanes(&k, &init, &[], n, NT_MIN_BYTES, false);
            }
        }
    }

    /// f64 reductions: a fused dot, a plain sum of a composite, and max/min — the canonical tree at
    /// four lanes, against `eval_reduction::<f64>`.
    #[test]
    fn avx2_f64_reductions_match_reference() {
        let mk = |op, ops: Vec<VecOp>, value, fma| VecKernel {
            name: sym(),
            lane: VecLane::F64,
            streams: 2,
            scalars: 0,
            unroll: 4,
            reduce: Some(wukong_mir::VecReduce { op, value, fma }),
            ops,
        };
        let two = vec![VecOp::Load { stream: 0 }, VecOp::Load { stream: 1 }];
        let mut diff = two.clone();
        diff.push(VecOp::Bin {
            op: VecBin::Sub,
            a: 0,
            b: 1,
        });
        let kernels = [
            mk(VecRedOp::Add, two.clone(), 0, Some((0, 1))),
            mk(VecRedOp::Add, diff, 2, None),
            mk(VecRedOp::Fmax, two.clone(), 0, None),
            mk(VecRedOp::Fmin, two, 1, None),
        ];
        for k in &kernels {
            for n in [4usize, 8, 12, 16, 20, 32, 64, 100, 256] {
                // Finite inputs for the sums (a NaN would make every sum NaN and prove nothing);
                // the max/min inputs keep their NaNs and signed zeros.
                let (a, b) = if k.reduce.unwrap().op == VecRedOp::Add {
                    (
                        (0..n).map(|i| (i % 23) as f64 * 0.1 - 1.0).collect(),
                        (0..n).map(|i| (i % 7) as f64 * 1e-3 + 0.5).collect(),
                    )
                } else {
                    (f64_input(n, 0.0), f64_input(n, 4.0))
                };
                check_lane_reduce(k, &[a, b], &[], n);
            }
        }
    }

    /// i32 lanes wrap like the scalar two's-complement ops (`vpmulld` keeps the low 32 bits, `0 - MIN
    /// == MIN`), and every compare is the signed one — the inverted `!=`/`<=`/`>=` included, which
    /// the blend implements by swapping its operands.
    #[test]
    fn avx2_i32_lanes_match_reference() {
        let arith = VecKernel {
            name: sym(),
            lane: VecLane::I32,
            streams: 3,
            scalars: 1,
            unroll: 4,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 0,
                    b: 1,
                }, // 2
                VecOp::Splat { scalar: 0 },
                VecOp::Bin {
                    op: VecBin::Sub,
                    a: 2,
                    b: 3,
                }, // 4
                VecOp::Neg { a: 4 }, // 5
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 5,
                    b: 0,
                }, // 6
                VecOp::Store { stream: 2, val: 6 },
            ],
        };
        for n in LANE_LENGTHS {
            let init = vec![
                int_input(n, 1, [i32::MIN, i32::MAX]),
                int_input(n, 2, [i32::MAX, -1]),
                vec![0; n],
            ];
            check_lanes(&arith, &init, &[7], n, NT_MIN_BYTES, true);
        }
        for pred in [
            VecCmp::Lt,
            VecCmp::Le,
            VecCmp::Gt,
            VecCmp::Ge,
            VecCmp::Eq,
            VecCmp::Ne,
        ] {
            let k = VecKernel {
                name: sym(),
                lane: VecLane::I32,
                streams: 3,
                scalars: 0,
                unroll: 2,
                reduce: None,
                ops: vec![
                    VecOp::Load { stream: 0 },
                    VecOp::Load { stream: 1 },
                    VecOp::Cmp { pred, a: 0, b: 1 },
                    VecOp::Select {
                        mask: 2,
                        a: 0,
                        b: 1,
                    },
                    VecOp::Store { stream: 2, val: 3 },
                ],
            };
            for n in LANE_LENGTHS {
                // Equal pairs, both orders, and the sign boundary (MIN vs MAX compares signed).
                let a: Vec<i32> = int_input(n, 3, [i32::MIN, 0]);
                let b: Vec<i32> = (0..n)
                    .map(|i| {
                        if i % 3 == 0 {
                            a[i]
                        } else {
                            int_input(n, 5, [i32::MAX, 0])[i]
                        }
                    })
                    .collect();
                check_lanes(&k, &[a, b, vec![0; n]], &[], n, NT_MIN_BYTES, false);
            }
        }
        // A wrapping sum.
        let sum = VecKernel {
            name: sym(),
            lane: VecLane::I32,
            streams: 1,
            scalars: 0,
            unroll: 4,
            reduce: Some(wukong_mir::VecReduce {
                op: VecRedOp::Add,
                value: 0,
                fma: None,
            }),
            ops: vec![VecOp::Load { stream: 0 }],
        };
        for n in [8usize, 16, 32, 40, 96, 256] {
            check_lane_reduce(&sum, &[int_input(n, 1, [i32::MAX, i32::MAX])], &[], n);
        }
    }

    /// i64 lanes: add/sub/negate, every signed compare, and a wrapping sum (AVX2 has no `vpmullq`, so
    /// a multiply is outside the coverage and must decline rather than assemble).
    #[test]
    fn avx2_i64_lanes_match_reference() {
        let arith = VecKernel {
            name: sym(),
            lane: VecLane::I64,
            streams: 3,
            scalars: 1,
            unroll: 4,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Bin {
                    op: VecBin::Sub,
                    a: 0,
                    b: 1,
                }, // 2
                VecOp::Neg { a: 2 }, // 3
                VecOp::Splat { scalar: 0 },
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 3,
                    b: 4,
                }, // 5
                VecOp::Store { stream: 2, val: 5 },
            ],
        };
        for n in LANE_LENGTHS {
            let init = vec![
                int_input(n, 1, [i64::MIN, i64::MAX]),
                int_input(n, 2, [i64::MAX, 1]),
                vec![0; n],
            ];
            check_lanes(&arith, &init, &[-3], n, NT_MIN_BYTES, true);
        }
        for pred in [
            VecCmp::Lt,
            VecCmp::Le,
            VecCmp::Gt,
            VecCmp::Ge,
            VecCmp::Eq,
            VecCmp::Ne,
        ] {
            let k = VecKernel {
                name: sym(),
                lane: VecLane::I64,
                streams: 3,
                scalars: 0,
                unroll: 4,
                reduce: None,
                ops: vec![
                    VecOp::Load { stream: 0 },
                    VecOp::Load { stream: 1 },
                    VecOp::Cmp { pred, a: 1, b: 0 },
                    VecOp::Select {
                        mask: 2,
                        a: 1,
                        b: 0,
                    },
                    VecOp::Store { stream: 2, val: 3 },
                ],
            };
            for n in LANE_LENGTHS {
                let a: Vec<i64> = int_input(n, 4, [i64::MIN, 0]);
                let b: Vec<i64> = (0..n)
                    .map(|i| {
                        if i % 4 == 1 {
                            a[i]
                        } else {
                            int_input(n, 6, [i64::MAX, -1])[i]
                        }
                    })
                    .collect();
                check_lanes(&k, &[a, b, vec![0; n]], &[], n, NT_MIN_BYTES, false);
            }
        }
        let sum = VecKernel {
            name: sym(),
            lane: VecLane::I64,
            streams: 2,
            scalars: 0,
            unroll: 4,
            reduce: Some(wukong_mir::VecReduce {
                op: VecRedOp::Add,
                value: 2,
                fma: None,
            }),
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Bin {
                    op: VecBin::Sub,
                    a: 0,
                    b: 1,
                },
            ],
        };
        for n in [4usize, 8, 16, 20, 64, 100] {
            check_lane_reduce(
                &sum,
                &[
                    int_input(n, 1, [i64::MAX, i64::MAX]),
                    int_input(n, 3, [i64::MIN, 5]),
                ],
                &[],
                n,
            );
        }
        let mut mul = sum.clone();
        mul.reduce = None;
        mul.streams = 3;
        mul.ops[2] = VecOp::Bin {
            op: VecBin::Mul,
            a: 0,
            b: 1,
        };
        mul.ops.push(VecOp::Store { stream: 2, val: 2 });
        assert!(mul.validate().is_empty(), "an i64 multiply is well-formed");
        assert!(mul.pressure().is_none(), "but outside AVX2's coverage");
        assert!(assemble_kernel(&mul).is_err());
    }

    /// The non-temporal path: assembled with a 1-byte threshold so a small call takes it whenever
    /// every output-only stream is 32-byte aligned. Aligned and misaligned calls must both store
    /// exactly what `eval_lane` computes (the misaligned one through the cacheable copy — a
    /// `vmovnt*` there would fault), for every lane type, with an in-place stream that must keep
    /// the cacheable store even on the aligned call.
    #[test]
    fn avx2_non_temporal_stores_match_reference() {
        // o = a*k + b (o output-only → non-temporal), a = a + 1 (read and written → cacheable).
        let f32k = VecKernel {
            name: sym(),
            lane: VecLane::F32,
            streams: 3,
            scalars: 2,
            unroll: 4,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Splat { scalar: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Fma { a: 0, b: 1, c: 2 },
                VecOp::Store { stream: 2, val: 3 },
                VecOp::Splat { scalar: 1 },
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 0,
                    b: 5,
                },
                VecOp::Store { stream: 0, val: 6 },
            ],
        };
        for misaligned in [false, true] {
            for n in [0usize, 8, 15, 64, 100, 1000] {
                let init = vec![ramp(n, 0.0), ramp(n, 1.0), vec![0.0f32; n]];
                check_lanes(&f32k, &init, &[2.5, 1.0], n, 1, misaligned);
            }
        }
        let mut f64k = f32k.clone();
        f64k.lane = VecLane::F64;
        let mut i32k = f32k.clone();
        i32k.lane = VecLane::I32;
        i32k.ops[3] = VecOp::Bin {
            op: VecBin::Mul,
            a: 0,
            b: 1,
        };
        i32k.ops.insert(
            4,
            VecOp::Bin {
                op: VecBin::Add,
                a: 3,
                b: 2,
            },
        );
        // Re-point the store and later operands past the inserted op.
        i32k.ops[5] = VecOp::Store { stream: 2, val: 4 };
        i32k.ops[7] = VecOp::Bin {
            op: VecBin::Add,
            a: 0,
            b: 6,
        };
        i32k.ops[8] = VecOp::Store { stream: 0, val: 7 };
        for misaligned in [false, true] {
            for n in [0usize, 4, 9, 64, 100] {
                let init = vec![f64_input(n, 0.0), f64_input(n, 2.0), vec![0.0f64; n]];
                check_lanes(&f64k, &init, &[2.5, 1.0], n, 1, misaligned);
                let init = vec![
                    int_input(n, 1, [i32::MIN, i32::MAX]),
                    int_input(n, 2, [-1, 1]),
                    vec![0i32; n],
                ];
                check_lanes(&i32k, &init, &[3, 1], n, 1, misaligned);
            }
        }
    }
}
