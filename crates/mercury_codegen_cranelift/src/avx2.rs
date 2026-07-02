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
//! flat [`VecKernel`] recipe (in `mercury_mir`, backend-agnostic) and this module assembles it into a
//! self-contained function that both the Cranelift JIT/object backend installs (via
//! `define_function_bytes`) and — for the differential oracle — the interpreter marshals lane-wise
//! from the *same* recipe. Elementwise lanes are bit-identical across the two; the gate polices it.
//!
//! ABI of an assembled kernel (Windows/SysV both pass the first args in registers we normalize to):
//! `fn(ptrs: *const *mut u8, scalars: *const f32, n: u64)`. `ptrs[k]` is the base of stream `k`,
//! `scalars[k]` the k-th loop-invariant f32, `n` the (multiple-of-8) element count the caller assigns
//! to the vector part; the caller runs the scalar remainder itself. The kernel touches only volatile
//! registers, makes no calls, and ends with `vzeroupper` — so no prologue/epilogue is needed.

#![allow(dead_code)] // Phase A: assembler proven in isolation before the vectorizer wires it in.

use iced_x86::code_asm::*;
use mercury_mir::{VecBin, VecCmp, VecKernel, VecOp};

/// f32 lanes per YMM register (256-bit / 32-bit).
pub const LANES: u32 = 8;
const GROUP_BYTES: i32 = (LANES * 4) as i32;
/// Total architectural YMM registers.
const NREG: u8 = 16;
/// The maximum number of distinct streams we keep in GPRs (`r9`, `r10`, `r11`, `rdx`-after-hoist).
const MAX_STREAMS: u32 = 4;

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
    per_group: u8,               // P: registers a single group needs
    hoist_scalars: Vec<u32>,     // distinct invariant scalars, each in a fixed reg
    signmask_reg: Option<u8>,    // fixed reg holding -0.0 lanes, if the body negates
    scalar_reg: Vec<(u32, u8)>,  // scalar index → fixed reg
    saved: Vec<u8>,              // xmm6..xmm15 we must preserve (callee-saved on Win64)
    last_use: Vec<usize>,        // last_use[i] = last op index referencing value i
}

/// The operand value-indices an op reads (for liveness); `Store`/`Load`/`Splat`/`Const` read 0..1.
fn operands(op: &VecOp) -> Vec<u32> {
    match *op {
        VecOp::Load { .. } | VecOp::Splat { .. } | VecOp::Const { .. } => vec![],
        VecOp::Bin { a, b, .. } | VecOp::Cmp { a, b, .. } => vec![a, b],
        VecOp::Fma { a, b, c } => vec![a, b, c],
        VecOp::Select { mask, a, b } => vec![mask, a, b],
        VecOp::Sqrt { a } | VecOp::Neg { a } => vec![a],
        VecOp::Store { val, .. } => vec![val],
    }
}
/// Whether an op produces a body value that needs a (non-hoisted) register.
fn produces_body_value(op: &VecOp) -> bool {
    !matches!(
        op,
        VecOp::Store { .. } | VecOp::Splat { .. } | VecOp::Const { .. }
    )
}

/// Plan the register allocation, or return `Err` (→ the caller keeps the 128-bit fallback) when the
/// body cannot be expressed in this emitter (too many streams, a `Const` the vectorizer should have
/// folded into `scalars`, or a single group that will not fit in 16 YMM registers).
fn plan_registers(k: &VecKernel) -> Result<Plan, String> {
    if k.streams == 0 || k.streams > MAX_STREAMS {
        return Err(format!("avx2: {} streams (max {MAX_STREAMS})", k.streams));
    }
    if k.ops.iter().any(|o| matches!(o, VecOp::Const { .. })) {
        return Err("avx2: Const must be folded into scalars".into());
    }
    // Distinct invariant scalars referenced by Splat, in first-seen order.
    let mut hoist_scalars: Vec<u32> = Vec::new();
    for o in &k.ops {
        if let VecOp::Splat { scalar } = *o {
            if !hoist_scalars.contains(&scalar) {
                hoist_scalars.push(scalar);
            }
        }
    }
    let needs_signmask = k.ops.iter().any(|o| matches!(o, VecOp::Neg { .. }));

    // last_use[i]: last op index that references value i.
    let mut last_use = vec![0usize; k.ops.len()];
    for (i, o) in k.ops.iter().enumerate() {
        for v in operands(o) {
            last_use[v as usize] = i;
        }
    }

    // P = peak simultaneously-live body registers (alloc result before freeing this op's dead
    // operands, mirroring the emitter's ordering, so P is an exact upper bound).
    let mut live: Vec<usize> = Vec::new();
    let mut per_group: usize = 0;
    for (i, o) in k.ops.iter().enumerate() {
        if produces_body_value(o) {
            live.push(i);
            per_group = per_group.max(live.len());
        }
        for v in operands(o) {
            // hoisted (Splat/Const) values never occupy a body register.
            let is_hoisted = matches!(k.ops[v as usize], VecOp::Splat { .. } | VecOp::Const { .. });
            if !is_hoisted && last_use[v as usize] == i {
                live.retain(|&x| x != v as usize);
            }
        }
    }
    let per_group = per_group.max(1) as u8;
    let hoist = hoist_scalars.len() as u8 + needs_signmask as u8;
    if per_group as u32 + hoist as u32 > NREG as u32 {
        return Err(format!(
            "avx2: body needs {per_group}+{hoist} > {NREG} registers"
        ));
    }
    // Interleave as many groups as fit (dense from ymm0), capped by the requested unroll.
    let unroll = k
        .unroll
        .max(1)
        .min(((NREG - hoist) / per_group) as u32)
        .max(1);

    // Register assignment (dense, low first, to minimize callee-saved spills): body banks occupy
    // ymm[0 .. unroll*P); the hoisted broadcasts occupy the next `hoist` registers.
    let mut next = unroll as u8 * per_group;
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
        saved,
        last_use,
    })
}

/// Assemble `k` into a self-contained AVX2 function `fn(ptrs, scalars, n)` (Win64 ABI: rcx, rdx,
/// r8). Returns the raw machine code, or `Err` if the body is outside this emitter's coverage (the
/// caller then keeps the differential-safe 128-bit vectorizer path).
pub fn assemble_kernel(k: &VecKernel) -> Result<Vec<u8>, String> {
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
            emit_group(&mut a, k, plan, u, (u * LANES) as i32 * 4)?;
        }
        a.add(rcx, step)?;
        a.jmp(head)?;
    }

    // --- Single-vector cleanup loop (step 8). ---
    a.set_label(&mut single_head)?;
    a.lea(rax, qword_ptr(rcx + LANES as i32))?;
    a.cmp(rax, r8)?;
    a.ja(done)?;
    emit_group(&mut a, k, plan, 0, 0)?;
    a.add(rcx, LANES as i32)?;
    a.jmp(single_head)?;
    a.set_label(&mut done)?;

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
) -> Result<(), IcedError> {
    let bank_base = u as u8 * plan.per_group;
    let mut free: Vec<u8> = (0..plan.per_group).map(|r| bank_base + r).collect();
    let mut vreg: Vec<Option<u8>> = vec![None; k.ops.len()];

    // The register currently holding value `v` (a hoisted scalar resolves to its fixed register).
    let reg_of = |vreg: &[Option<u8>], v: u32| -> u8 {
        match k.ops[v as usize] {
            VecOp::Splat { scalar } => {
                plan.scalar_reg
                    .iter()
                    .find(|(s, _)| *s == scalar)
                    .map(|(_, r)| *r)
                    .expect("hoisted scalar reg")
            }
            _ => vreg[v as usize].expect("value has a register"),
        }
    };
    // Free value `v`'s body register after op `i`, if this is its last use.
    let free_after = |free: &mut Vec<u8>, vreg: &mut [Option<u8>], i: usize, v: u32| {
        if plan.last_use[v as usize] == i {
            if let Some(r) = vreg[v as usize] {
                if !matches!(k.ops[v as usize], VecOp::Splat { .. } | VecOp::Const { .. }) {
                    free.push(r);
                }
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
    Ok(())
}

/// Bring-up kernel (Phase A): hand-written saxpy `out[i] = a*x[i] + y[i]` over `n` (multiple of 8)
/// elements, `ptrs = [x, y, out]`, `scalars = [a]`. Proves the encode → `define_function_bytes` →
/// finalize → call → result seam end-to-end with a real 256-bit `vfmadd` before the recipe compiler
/// exists. Returns the raw machine code.
pub fn assemble_saxpy_probe() -> Result<Vec<u8>, IcedError> {
    let mut a = CodeAssembler::new(64)?;

    // Args (SysV: rdi,rsi,rdx / Win64: rcx,rdx,r8). This bring-up test drives it through the host
    // C ABI directly, so we branch on target below when calling; the body uses the Win64 mapping
    // because that is this box. rcx = ptrs, rdx = scalars, r8 = n.
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
        sig.params.push(AbiParam::new(cranelift_codegen::ir::types::I64)); // n
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
        sig.params.push(AbiParam::new(cranelift_codegen::ir::types::I64));
        let fid = module
            .declare_function("k", Linkage::Export, &sig)
            .unwrap();
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

    // Deterministic inputs that exercise negatives, zeros, and non-round values (bit-exact vs the
    // f32 reference regardless of "ugliness", including sqrt(<0)=NaN and x/0=inf).
    fn ramp(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.375 - 5.0 + seed).collect()
    }

    fn sym() -> mercury_span::Symbol {
        mercury_span::Interner::new().intern("k")
    }

    #[test]
    fn avx2_kernel_saxpy_all_widths() {
        // out = a*x + y
        let k = VecKernel {
            name: sym(),
            streams: 3,
            scalars: 1,
            unroll: 4,
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
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Bin { op: VecBin::Add, a: 0, b: 1 },
                VecOp::Bin { op: VecBin::Sub, a: 0, b: 1 },
                VecOp::Bin { op: VecBin::Mul, a: 2, b: 3 },
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
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Sqrt { a: 1 },
                VecOp::Bin { op: VecBin::Div, a: 0, b: 2 },
                VecOp::Store { stream: 2, val: 3 },
            ],
        };
        for n in [8usize, 24, 64] {
            check(&k, &[ramp(n, 0.0), ramp(n, 0.0), vec![0.0; n]], &[], n);
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
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Neg { a: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Bin { op: VecBin::Add, a: 1, b: 2 },
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
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Splat { scalar: 0 }, // 0.0
                VecOp::Cmp { pred: VecCmp::Gt, a: 0, b: 1 },
                VecOp::Select { mask: 2, a: 0, b: 1 },
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
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Splat { scalar: 0 },
                VecOp::Bin { op: VecBin::Mul, a: 0, b: 1 },
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

    #[test]
    fn avx2_kernel_bails_on_five_streams() {
        let k = VecKernel {
            name: sym(),
            streams: 5,
            scalars: 0,
            unroll: 1,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Store { stream: 4, val: 0 },
            ],
        };
        assert!(assemble_kernel(&k).is_err(), "5 streams must bail to fallback");
    }
}
