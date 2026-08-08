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
//! ABI of an assembled kernel: `fn(ptrs: *const *mut u8, scalars: *const f32, n: u64)`, returning
//! nothing for an elementwise body and the `f32` horizontal fold (in XMM0) for a reduction — the
//! Cranelift side declares the two signatures accordingly. `ptrs[k]` is the base of stream `k`,
//! `scalars[k]` the k-th loop-invariant f32, `n` the (multiple-of-8) element count the caller assigns
//! to the vector part; the caller runs the scalar remainder itself. The kernel touches only volatile
//! GPRs and makes no calls, so it needs no shadow space and no frame — the only prologue/epilogue is
//! saving the low xmm halves of any of ymm6..ymm15 it allocates (callee-saved under Win64), and it
//! ends with `vzeroupper`.
//!
//! The body reads its three arguments out of **rcx/rdx/r8** — the Win64 integer argument registers —
//! and executes VEX.256 AVX2 + FMA3 encodings. Neither is negotiable here: the register mapping is
//! literal in the emitted bytes, and there is no runtime dispatch inside a kernel. So
//! [`assemble_kernel`] refuses on any host that does not supply both (see [`host_supports_kernels`]),
//! which is the only CPU-feature gate on this path — the Cranelift side declares the kernel with the
//! module's `default_call_conv` and installs the bytes verbatim, so a mismatch is silent corruption.

// Covers the test-only `assemble_saxpy_probe` reference emitter plus `GROUP_BYTES` and
// `Plan::hoist_scalars`, which the recipe path computes but no longer reads.
#![allow(dead_code)]

use iced_x86::code_asm::*;
use wukong_mir::{VecBin, VecCmp, VecKernel, VecOp, VecPressure, VecRedOp};

/// f32 lanes per YMM register (256-bit / 32-bit).
pub const LANES: u32 = wukong_mir::VEC_LANES;
const GROUP_BYTES: i32 = (LANES * 4) as i32;
/// Total architectural YMM registers.
const NREG: u32 = wukong_mir::VEC_NREG;

/// Whether this host can run the machine code this module emits: the AVX2 + FMA3 instructions the
/// body encodes, delivered under the Win64 argument-register mapping (rcx/rdx/r8) the body hardcodes.
/// Every other AVX2 consumer in the tree (`wukong_runtime`'s gemm/gemv/attention/… kernels) gates the
/// same way and keeps a scalar twin; this path has no in-kernel fallback, so the gate is the refusal
/// in [`assemble_kernel`]. `is_x86_feature_detected!` caches its CPUID probe, so this is cheap.
pub fn host_supports_kernels() -> bool {
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "windows")))]
    {
        false
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
/// Stream index → its base-pointer GPR. Loaded once before the loop; `rdx` is reused for stream 3
/// only after the invariant scalars have been broadcast out of `[rdx]`.
fn base_gpr(stream: u32) -> AsmRegister64 {
    [r9, r10, r11, rdx][stream as usize]
}
/// `vcmpps` imm8 predicate for an ordered compare that is false on NaN (except `Ne`, unordered/true
/// on NaN) — matching Rust's `<,<=,>,>=,==,!=` on the interpreter oracle side.
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

/// A register-allocation plan for a kernel body: how many interleaved groups fit, the per-group
/// live-register count, and which invariant broadcasts / sign-mask occupy fixed registers.
struct Plan {
    unroll: u32,
    per_group: u8,              // P: registers a single group needs
    hoist_scalars: Vec<u32>,    // distinct invariant scalars, each in a fixed reg
    signmask_reg: Option<u8>,   // fixed reg holding -0.0 lanes, if the body negates
    scalar_reg: Vec<(u32, u8)>, // scalar index → fixed reg
    acc_regs: Vec<u8>, // reduction accumulators, one per unroll copy (empty if elementwise)
    saved: Vec<u8>,    // xmm6..xmm15 we must preserve (callee-saved on Win64)
    last_use: Vec<usize>, // last_use[i] = last op index referencing value i
}

/// Plan the register allocation, or return `Err` (→ the caller keeps the 128-bit fallback) when the
/// body cannot be expressed in this emitter (too many streams, a `Const` the vectorizer should have
/// folded into `scalars`, or a single group that will not fit in 16 YMM registers).
fn plan_registers(k: &VecKernel) -> Result<Plan, String> {
    // P (peak live), H's parts (hoisted scalars + sign-mask) and last_use come from the
    // backend-agnostic `pressure()` in `wukong_mir`, single-sourced with the vectorizer's gate so a
    // body the vectorizer accepted never fails to assemble here.
    let pressure = k.pressure().ok_or_else(|| {
        format!(
            "avx2: body outside emitter coverage (streams={}, or Const present)",
            k.streams
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
    let (per_group, hoist) = (per_group as u8, hoist as u8);
    // A reduction additionally needs one accumulator register per unrolled copy; an elementwise
    // kernel needs none. The unroll comes from `wukong_mir` (single-sourced with the interpreter's
    // `eval_reduction`, which must pick the same number of accumulators to reassociate identically).
    let is_reduction = k.reduce.is_some();
    let unroll = if is_reduction {
        if !k.reduction_fits() {
            return Err(format!(
                "avx2: reduction body needs {per_group}+1+{hoist} > {NREG} registers"
            ));
        }
        k.reduction_unroll()
    } else {
        // Interleave as many groups as fit (dense from ymm0), capped by the requested unroll.
        k.unroll
            .max(1)
            .min((NREG - hoist as u32) / per_group as u32)
            .max(1)
    };

    // Register assignment (dense, low first, to minimize callee-saved spills): body banks occupy
    // ymm[0 .. unroll*P); a reduction's accumulators occupy the next `unroll`; the hoisted broadcasts
    // occupy the `hoist` after that.
    let mut next = unroll as u8 * per_group;
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

    Ok(Plan {
        unroll,
        per_group,
        hoist_scalars,
        signmask_reg,
        scalar_reg,
        acc_regs,
        saved,
        last_use,
    })
}

/// Assemble `k` into a self-contained AVX2 function `fn(ptrs, scalars, n)` (Win64 ABI: rcx, rdx,
/// r8). Returns the raw machine code, or `Err` if the host cannot run these bytes at all, or if the
/// body is outside this emitter's coverage.
pub fn assemble_kernel(k: &VecKernel) -> Result<Vec<u8>, String> {
    // The ISA/ABI gate comes first: the bytes below are AVX2+FMA3 under the Win64 argument
    // registers, with no in-kernel dispatch. Emitting them for a host that lacks either is a #UD at
    // the first `vmovups ymm` or a wild dereference of whatever rcx held — neither diagnosable.
    if !host_supports_kernels() {
        return Err(
            "avx2: host lacks AVX2+FMA3 under the Win64 ABI these kernels encode".to_string(),
        );
    }
    let plan = plan_registers(k)?;
    emit(k, &plan).map_err(|e| format!("avx2 encode: {e}"))
}

fn emit(k: &VecKernel, plan: &Plan) -> Result<Vec<u8>, IcedError> {
    let mut a = CodeAssembler::new(64)?;

    // --- Prologue: preserve callee-saved xmm6..xmm15 we use (low 128 bits). No calls are made, so
    // no shadow space; vmovups avoids any stack-alignment requirement. ---
    let save_bytes = plan.saved.len() as i32 * 16;
    if save_bytes > 0 {
        a.sub(rsp, save_bytes)?;
        for (i, &r) in plan.saved.iter().enumerate() {
            a.vmovups(xmmword_ptr(rsp + i as i32 * 16), xmm(r))?;
        }
    }

    // rcx = ptrs, rdx = scalars, r8 = n. Load the first three stream bases out of `ptrs`.
    for s in 0..k.streams.min(3) {
        a.mov(base_gpr(s), qword_ptr(rcx + (8 * s) as i32))?;
    }
    // Broadcast the invariant scalars (still readable via rdx) into their fixed registers.
    for &(sidx, reg) in &plan.scalar_reg {
        a.vbroadcastss(ymm(reg), dword_ptr(rdx + (4 * sidx) as i32))?;
    }
    // Build the -0.0 sign mask in-register (0x80000000 per lane) for negation, if needed.
    if let Some(r) = plan.signmask_reg {
        a.vpcmpeqd(ymm(r), ymm(r), ymm(r))?; // all ones
        a.vpslld(ymm(r), ymm(r), 31u32)?; // → 0x80000000 per lane
    }
    // Initialise the reduction accumulators to the fold identity (all lanes): 0.0 for a sum, −∞/+∞
    // for max/min so empty tail lanes never win. The ±∞ patterns are built in-register from all-ones
    // (no data section): −∞ = 0xFF800000 = all-ones << 23; +∞ = 0x7F800000 = (all-ones >> 24) << 23.
    if let Some(red) = k.reduce {
        for &r in &plan.acc_regs {
            match red.op {
                VecRedOp::Add => a.vxorps(ymm(r), ymm(r), ymm(r))?,
                VecRedOp::Fmax => {
                    a.vpcmpeqd(ymm(r), ymm(r), ymm(r))?;
                    a.vpslld(ymm(r), ymm(r), 23u32)?; // → 0xFF800000 = −∞
                }
                VecRedOp::Fmin => {
                    a.vpcmpeqd(ymm(r), ymm(r), ymm(r))?;
                    a.vpsrld(ymm(r), ymm(r), 24u32)?; // → 0x000000FF
                    a.vpslld(ymm(r), ymm(r), 23u32)?; // → 0x7F800000 = +∞
                }
            }
        }
    }
    // The 4th stream base reuses rdx now that the scalars have been read.
    if k.streams == 4 {
        a.mov(rdx, qword_ptr(rcx + 24))?;
    }
    a.xor(rcx, rcx)?; // i = 0 (rcx is free now that ptrs is consumed)

    // The single-vector loop head is the main loop's exit target too, so it carries exactly one
    // label (iced forbids two labels on one instruction).
    let mut single_head = a.create_label();
    let mut done = a.create_label();

    // --- Unrolled main loop (only when it does more than the single-vector loop). ---
    if plan.unroll > 1 {
        let step = (plan.unroll * LANES) as i32;
        let mut head = a.create_label();
        a.set_label(&mut head)?;
        a.lea(rax, qword_ptr(rcx + step))?;
        a.cmp(rax, r8)?;
        a.ja(single_head)?;
        for u in 0..plan.unroll {
            emit_group(
                &mut a,
                k,
                plan,
                u,
                (u * LANES) as i32 * 4,
                plan.acc_regs.get(u as usize).copied(),
            )?;
        }
        a.add(rcx, step)?;
        a.jmp(head)?;
    }

    // --- Single-vector cleanup loop (step 8); a reduction folds into accumulator 0. ---
    a.set_label(&mut single_head)?;
    a.lea(rax, qword_ptr(rcx + LANES as i32))?;
    a.cmp(rax, r8)?;
    a.ja(done)?;
    emit_group(&mut a, k, plan, 0, 0, plan.acc_regs.first().copied())?;
    a.add(rcx, LANES as i32)?;
    a.jmp(single_head)?;
    a.set_label(&mut done)?;

    // --- Reduction finish: combine the accumulators into acc[0], then a sequential lane-0..7
    // horizontal fold into xmm0 (the f32 return). Done before the epilogue restores the callee-saved
    // xmm halves, since an accumulator may live in one of them. ---
    if let Some(red) = k.reduce {
        let acc0 = plan.acc_regs[0];
        for &r in &plan.acc_regs[1..] {
            fold_ymm(&mut a, red.op, acc0, r)?;
        }
        // Spill the 8 lanes and fold them in order; a local 32-byte scratch keeps rsp balanced.
        a.sub(rsp, 32)?;
        a.vmovups(ymmword_ptr(rsp), ymm(acc0))?;
        a.vmovss(xmm(0), dword_ptr(rsp))?;
        for lane in 1..LANES as i32 {
            fold_ss(&mut a, red.op, dword_ptr(rsp + lane * 4))?;
        }
        a.add(rsp, 32)?;
    }

    // --- Epilogue. ---
    if save_bytes > 0 {
        for (i, &r) in plan.saved.iter().enumerate() {
            a.vmovups(xmm(r), xmmword_ptr(rsp + i as i32 * 16))?;
        }
        a.add(rsp, save_bytes)?;
    }
    a.vzeroupper()?;
    a.ret()?;
    a.assemble(0x0)
}

/// Emit one lane-group of the body: unroll copy `u` (its own register bank of `per_group` regs) at
/// byte displacement `disp` (`u * 32`) from `base + i*4`. Loads/stores are unit-stride.
///
/// Allocation rule (uniform, hazard-free): resolve operand registers, allocate the result register
/// from a free list that still excludes every live operand, emit, *then* free operands whose last
/// use is this op. So the result never aliases a still-needed operand — even across the two-op FMA
/// (`vmovaps` + `vfmadd231ps`). `per_group` was sized to admit result-plus-live-operands, so the
/// free list never underflows.
fn emit_group(
    a: &mut CodeAssembler,
    k: &VecKernel,
    plan: &Plan,
    u: u32,
    disp: i32,
    acc_reg: Option<u8>,
) -> Result<(), IcedError> {
    let bank_base = u as u8 * plan.per_group;
    let mut free: Vec<u8> = (0..plan.per_group).map(|r| bank_base + r).collect();
    let mut vreg: Vec<Option<u8>> = vec![None; k.ops.len()];

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
            a.vmovups(ymmword_ptr(base_gpr(stream) + rcx * 4 + disp), ymm(rv))?;
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
                a.vmovups(ymm(d), ymmword_ptr(base_gpr(stream) + rcx * 4 + disp))?;
                vreg[i] = Some(d);
            }
            VecOp::Bin { op, a: x, b: y } => {
                let (rx, ry) = (reg_of(&vreg, x), reg_of(&vreg, y));
                let d = free.pop().expect("free reg");
                match op {
                    VecBin::Add => a.vaddps(ymm(d), ymm(rx), ymm(ry))?,
                    VecBin::Sub => a.vsubps(ymm(d), ymm(rx), ymm(ry))?,
                    VecBin::Mul => a.vmulps(ymm(d), ymm(rx), ymm(ry))?,
                    VecBin::Div => a.vdivps(ymm(d), ymm(rx), ymm(ry))?,
                }
                vreg[i] = Some(d);
                free_after(&mut free, &mut vreg, i, x);
                free_after(&mut free, &mut vreg, i, y);
            }
            VecOp::Fma { a: x, b: y, c } => {
                let (rx, ry, rc) = (reg_of(&vreg, x), reg_of(&vreg, y), reg_of(&vreg, c));
                let d = free.pop().expect("free reg");
                a.vmovaps(ymm(d), ymm(rc))?;
                a.vfmadd231ps(ymm(d), ymm(rx), ymm(ry))?; // d = x*y + c
                vreg[i] = Some(d);
                free_after(&mut free, &mut vreg, i, x);
                free_after(&mut free, &mut vreg, i, y);
                free_after(&mut free, &mut vreg, i, c);
            }
            VecOp::Sqrt { a: x } => {
                let rx = reg_of(&vreg, x);
                let d = free.pop().expect("free reg");
                a.vsqrtps(ymm(d), ymm(rx))?;
                vreg[i] = Some(d);
                free_after(&mut free, &mut vreg, i, x);
            }
            VecOp::Neg { a: x } => {
                let rx = reg_of(&vreg, x);
                let sm = plan.signmask_reg.expect("signmask reg");
                let d = free.pop().expect("free reg");
                a.vxorps(ymm(d), ymm(rx), ymm(sm))?;
                vreg[i] = Some(d);
                free_after(&mut free, &mut vreg, i, x);
            }
            VecOp::Cmp { pred, a: x, b: y } => {
                let (rx, ry) = (reg_of(&vreg, x), reg_of(&vreg, y));
                let d = free.pop().expect("free reg");
                a.vcmpps(ymm(d), ymm(rx), ymm(ry), cmp_imm(pred))?;
                vreg[i] = Some(d);
                free_after(&mut free, &mut vreg, i, x);
                free_after(&mut free, &mut vreg, i, y);
            }
            VecOp::Select { mask, a: x, b: y } => {
                let (rm, rx, ry) = (reg_of(&vreg, mask), reg_of(&vreg, x), reg_of(&vreg, y));
                let d = free.pop().expect("free reg");
                // vblendvps dst, src1, src2, mask ⇒ dst = mask ? src2 : src1. We want mask ? x : y.
                a.vblendvps(ymm(d), ymm(ry), ymm(rx), ymm(rm))?;
                vreg[i] = Some(d);
                free_after(&mut free, &mut vreg, i, mask);
                free_after(&mut free, &mut vreg, i, x);
                free_after(&mut free, &mut vreg, i, y);
            }
            VecOp::Store { .. } | VecOp::Splat { .. } | VecOp::Const { .. } => unreachable!(),
        }
    }
    // A reduction folds this group's addend into this copy's accumulator. A fused product folds in one
    // rounding via `vfmadd231ps` (acc += X*Y), matching `eval_reduction`'s `mul_add`; otherwise the
    // plain `fold` of the addend value.
    if let (Some(acc), Some(red)) = (acc_reg, k.reduce) {
        match red.fma {
            Some((x, y)) => {
                let (rx, ry) = (reg_of(&vreg, x), reg_of(&vreg, y));
                a.vfmadd231ps(ymm(acc), ymm(rx), ymm(ry))?; // acc = X*Y + acc
            }
            None => {
                let rv = vreg[red.value as usize].expect("reduce value has a register");
                fold_ymm(a, red.op, acc, rv)?;
            }
        }
    }
    Ok(())
}

/// `acc = fold(acc, src)` on 256-bit lanes (`acc` = src1, so `vmaxps`/`vminps` keep src2 on ties/NaN,
/// matching [`VecRedOp::fold`]).
fn fold_ymm(a: &mut CodeAssembler, op: VecRedOp, acc: u8, src: u8) -> Result<(), IcedError> {
    match op {
        VecRedOp::Add => a.vaddps(ymm(acc), ymm(acc), ymm(src)),
        VecRedOp::Fmax => a.vmaxps(ymm(acc), ymm(acc), ymm(src)),
        VecRedOp::Fmin => a.vminps(ymm(acc), ymm(acc), ymm(src)),
    }
}

/// `xmm0 = fold(xmm0, [mem])` on one scalar (the horizontal reduction step).
fn fold_ss(a: &mut CodeAssembler, op: VecRedOp, src: AsmMemoryOperand) -> Result<(), IcedError> {
    match op {
        VecRedOp::Add => a.addss(xmm(0), src),
        VecRedOp::Fmax => a.maxss(xmm(0), src),
        VecRedOp::Fmin => a.minss(xmm(0), src),
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

    /// Phase A: assemble the saxpy kernel, install its raw bytes via `define_function_bytes`, call
    /// it, and check it computed `a*x + y` in true 256-bit strides. This is the whole raw-AVX2 seam.
    #[test]
    fn avx2_saxpy_kernel_runs() {
        if !host_supports_kernels() {
            return; // these bytes are not executable here — see `host_supports_kernels`
        }
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

        const N: usize = 24; // multiple of 8
        let mut x = [0f32; N];
        let mut y = [0f32; N];
        let mut out = [0f32; N];
        for i in 0..N {
            x[i] = i as f32;
            y[i] = (2 * i) as f32;
        }
        let a_scalar = [3.0f32];
        let ptrs = [x.as_mut_ptr(), y.as_mut_ptr(), out.as_mut_ptr()];
        kernel(ptrs.as_ptr(), a_scalar.as_ptr(), N as u64);

        for i in 0..N {
            let want = 3.0 * x[i] + y[i];
            assert_eq!(out[i], want, "lane {i}: got {}, want {want}", out[i]);
        }
        unsafe { module.free_memory() };
    }

    /// Assemble `k`, run it on `init` streams / `scalars` over `n` elements, and assert it matches
    /// the `VecKernel::eval_lane` reference **bit-for-bit** on every full-vector lane `[0, n/8*8)`.
    /// Inputs may alias outputs (in-place); the reference mutates its own copy the same way.
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
        let mut got: Vec<Vec<f32>> = init.to_vec();
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
        kernel(ptrs.as_ptr(), scalars.as_ptr(), n as u64);

        for s in 0..k.streams as usize {
            for i in 0..vlen {
                assert_eq!(
                    got[s][i].to_bits(),
                    refs[s][i].to_bits(),
                    "kernel vs eval_lane mismatch: stream {s} lane {i} (got {}, want {})",
                    got[s][i],
                    refs[s][i]
                );
            }
        }
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
        let ptrs: Vec<*const f32> = streams.iter().map(|v| v.as_ptr()).collect();
        let got = kernel(ptrs.as_ptr(), scalars.as_ptr(), vlen as u64);
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
        for n in [8usize, 16, 24, 32, 64, 128, 256, 512] {
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
        for n in [8usize, 16, 24, 32, 40, 64, 80, 100] {
            check(&k, &[ramp(n, 0.0), ramp(n, 1.0), vec![0.0; n]], &[2.5], n);
        }
    }

    #[test]
    fn avx2_kernel_diff_of_squares_p4() {
        // out = (x+y)*(x-y). Peak live = 4 registers.
        let k = VecKernel {
            name: sym(),
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

    #[test]
    fn avx2_kernel_bails_on_five_streams() {
        let k = VecKernel {
            name: sym(),
            streams: 5,
            scalars: 0,
            unroll: 1,
            reduce: None,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Store { stream: 4, val: 0 },
            ],
        };
        assert!(
            assemble_kernel(&k).is_err(),
            "5 streams must bail to fallback"
        );
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
}
