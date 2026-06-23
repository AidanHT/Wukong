//! Backward (gradient) GPU kernels for the autodiff tape — the device twin of the synthesized
//! counted loops `mercury_autodiff::tape` emits on the CPU.
//!
//! The tape's forward VJPs already ride the tuned `mercury_*` kernels (matmul gradients reuse the
//! GEMM, reductions reuse `sreduce`, …). What it *synthesizes* as plain counted loops — buffer
//! transpose (for `dB = dCᵀ·A`), the elementwise activation-backward `dx = dout ⊙ f'(x)`, and the
//! per-row norm-backward combines — are CPU device-loops today. This module provides their GPU
//! kernels so the whole backward stays device-resident, plus the **flash-attention backward**
//! (dQ/dK/dV) gated against a two-pass-softmax f64 reference (increment 4).
//!
//! Each kernel mirrors the exact math the tape emits (`mercury_autodiff::tape`), so it is gated
//! tolerance-/bit-equal to that op — and the tape's CPU form is finite-difference-gated, which
//! transitively makes the device form correct. PTX is pure ASCII; target `sm_89`. Kernels are
//! grid-stride / one-CTA-per-row, so correctness is independent of the launch grid.

use crate::gpu::Gpu;
use crate::ptx_gemm::{gemm_rb_ptx, TILE_M, TILE_N};
use crate::ptx_optim::grid_stride_cfg;
use cudarc::driver::{CudaSlice, DriverError, LaunchConfig, PushKernelArg};
use std::sync::OnceLock;

// Row-norm op codes (mirror `mercury_autodiff::tape` / `mercury_runtime`).
const NORM_SOFTMAX: i64 = 0;
const NORM_LAYERNORM: i64 = 1;
const NORM_RMSNORM: i64 = 2;

// ----------------------------------------------------------------------------------------------
// Transpose: dst(n×m) = src(m×n)ᵀ  (the `dB = dCᵀ·A` path's transpose, tape.rs::transpose)
// ----------------------------------------------------------------------------------------------

/// Transpose `src (m×n)` into `dst (n×m)`: for flat index `f`, `row = f/n`, `col = f%n`,
/// `dst[col*m + row] = src[f]`. Pure data movement — **bit-exact**. Grid-stride over `m*n`.
/// (A naive scatter; a tiled/coalesced transpose is a later perf lever — transpose is a small part
/// of the backward, the `dB` GEMM dominates.)
pub const TRANSPOSE_F32_PTX: &str = r#"
.version 7.8
.target sm_89
.address_size 64

.visible .entry transpose_f32(
    .param .u64 t_src,
    .param .u64 t_dst,
    .param .u32 t_m,
    .param .u32 t_n
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<12>;
    .reg .b64   %rd<8>;
    .reg .f32   %f<2>;

    ld.param.u64    %rd1, [t_src];
    ld.param.u64    %rd2, [t_dst];
    ld.param.u32    %r1,  [t_m];
    ld.param.u32    %r2,  [t_n];
    cvta.to.global.u64  %rd1, %rd1;
    cvta.to.global.u64  %rd2, %rd2;
    mul.lo.s32      %r3, %r1, %r2;      // total = m*n

    mov.u32         %r4, %ntid.x;
    mov.u32         %r5, %ctaid.x;
    mov.u32         %r6, %tid.x;
    mad.lo.s32      %r7, %r5, %r4, %r6; // idx
    mov.u32         %r8, %nctaid.x;
    mul.lo.s32      %r9, %r4, %r8;      // stride

T_LOOP:
    setp.ge.s32     %p1, %r7, %r3;
    @%p1 bra        T_END;

    div.u32         %r10, %r7, %r2;     // row = idx / n
    mul.lo.s32      %r11, %r10, %r2;
    sub.s32         %r11, %r7, %r11;    // col = idx - row*n
    mad.lo.s32      %r11, %r11, %r1, %r10; // didx = col*m + row

    mul.wide.s32    %rd3, %r7,  4;
    add.s64         %rd4, %rd1, %rd3;   // &src[idx]
    mul.wide.s32    %rd5, %r11, 4;
    add.s64         %rd6, %rd2, %rd5;   // &dst[didx]
    ld.global.f32   %f1, [%rd4];
    st.global.f32   [%rd6], %f1;

    add.s32         %r7, %r7, %r9;
    bra             T_LOOP;
T_END:
    ret;
}
"#;

/// Transpose `src (m×n)` → `dst (n×m)` on the GPU (host-slice convenience wrapper).
pub fn transpose_f32(g: &mut Gpu, src: &[f32], m: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    assert_eq!(src.len(), m * n, "transpose: src must be m×n");
    let f = g.function("transpose_f32", TRANSPOSE_F32_PTX, "transpose_f32")?;
    let src_d = g.stream.memcpy_stod(src)?;
    let mut dst_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mu, nu) = (m as u32, n as u32);
    let cfg = grid_stride_cfg(g, (m * n) as u32);
    let mut b = g.stream.launch_builder(&f);
    b.arg(&src_d).arg(&mut dst_d).arg(&mu).arg(&nu);
    unsafe { b.launch(cfg)? };
    g.stream.memcpy_dtov(&dst_d)
}

// ----------------------------------------------------------------------------------------------
// Activation backward: dx[i] = dout[i] * f'(x[i])   (tape.rs::activation_backward)
//   relu     f'(x) = (x>0) ? 1 : 0          (reads x)
//   sigmoid  f'    = y*(1-y)                (reads forward output y)
//   tanh     f'    = 1 - y^2               (reads y)
//   exp      f'    = y                      (reads y)
// One PTX module, one entry per activation; the launcher dispatches by op code. The signature is
// uniform `(dout, x, y, dx, n)`; each entry reads only the operands it needs.
// ----------------------------------------------------------------------------------------------

/// PTX for the elementwise activation-backward kernels (relu/sigmoid/tanh/exp), grid-stride. Each
/// `dx = dout ⊙ f'(·)` in the same op order as `tape.rs::activation_backward`, all correctly-rounded
/// f32 — bit-exact to the CPU tape loop.
pub const ACT_BWD_PTX: &str = r#"
.version 7.8
.target sm_89
.address_size 64

.visible .entry relu_bwd(
    .param .u64 ab_dout, .param .u64 ab_x, .param .u64 ab_y, .param .u64 ab_dx, .param .u32 ab_n
)
{
    .reg .pred  %p<3>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<8>;
    .reg .f32   %f<6>;
    ld.param.u64    %rd1, [ab_dout];
    ld.param.u64    %rd2, [ab_x];
    ld.param.u64    %rd3, [ab_dx];
    ld.param.u32    %r1,  [ab_n];
    cvta.to.global.u64  %rd1, %rd1;
    cvta.to.global.u64  %rd2, %rd2;
    cvta.to.global.u64  %rd3, %rd3;
    mov.u32 %r2, %ntid.x; mov.u32 %r3, %ctaid.x; mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r3, %r2, %r4;
    mov.u32 %r6, %nctaid.x; mul.lo.s32 %r7, %r2, %r6;
RELU_L:
    setp.ge.s32 %p1, %r5, %r1; @%p1 bra RELU_E;
    mul.wide.s32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4; ld.global.f32 %f1, [%rd5];   // dout
    add.s64 %rd6, %rd2, %rd4; ld.global.f32 %f2, [%rd6];   // x
    setp.gt.f32 %p2, %f2, 0f00000000;
    selp.f32 %f3, 0f3F800000, 0f00000000, %p2;             // x>0 ? 1 : 0
    mul.f32 %f4, %f1, %f3;
    add.s64 %rd7, %rd3, %rd4; st.global.f32 [%rd7], %f4;
    add.s32 %r5, %r5, %r7; bra RELU_L;
RELU_E:
    ret;
}

.visible .entry sigmoid_bwd(
    .param .u64 ab_dout, .param .u64 ab_x, .param .u64 ab_y, .param .u64 ab_dx, .param .u32 ab_n
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<8>;
    .reg .f32   %f<7>;
    ld.param.u64    %rd1, [ab_dout];
    ld.param.u64    %rd2, [ab_y];
    ld.param.u64    %rd3, [ab_dx];
    ld.param.u32    %r1,  [ab_n];
    cvta.to.global.u64  %rd1, %rd1;
    cvta.to.global.u64  %rd2, %rd2;
    cvta.to.global.u64  %rd3, %rd3;
    mov.u32 %r2, %ntid.x; mov.u32 %r3, %ctaid.x; mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r3, %r2, %r4;
    mov.u32 %r6, %nctaid.x; mul.lo.s32 %r7, %r2, %r6;
SIG_L:
    setp.ge.s32 %p1, %r5, %r1; @%p1 bra SIG_E;
    mul.wide.s32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4; ld.global.f32 %f1, [%rd5];   // dout
    add.s64 %rd6, %rd2, %rd4; ld.global.f32 %f2, [%rd6];   // y
    sub.f32 %f3, 0f3F800000, %f2;                          // 1 - y
    mul.f32 %f4, %f2, %f3;                                 // y*(1-y)
    mul.f32 %f5, %f1, %f4;
    add.s64 %rd7, %rd3, %rd4; st.global.f32 [%rd7], %f5;
    add.s32 %r5, %r5, %r7; bra SIG_L;
SIG_E:
    ret;
}

.visible .entry tanh_bwd(
    .param .u64 ab_dout, .param .u64 ab_x, .param .u64 ab_y, .param .u64 ab_dx, .param .u32 ab_n
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<8>;
    .reg .f32   %f<7>;
    ld.param.u64    %rd1, [ab_dout];
    ld.param.u64    %rd2, [ab_y];
    ld.param.u64    %rd3, [ab_dx];
    ld.param.u32    %r1,  [ab_n];
    cvta.to.global.u64  %rd1, %rd1;
    cvta.to.global.u64  %rd2, %rd2;
    cvta.to.global.u64  %rd3, %rd3;
    mov.u32 %r2, %ntid.x; mov.u32 %r3, %ctaid.x; mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r3, %r2, %r4;
    mov.u32 %r6, %nctaid.x; mul.lo.s32 %r7, %r2, %r6;
TANH_L:
    setp.ge.s32 %p1, %r5, %r1; @%p1 bra TANH_E;
    mul.wide.s32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4; ld.global.f32 %f1, [%rd5];   // dout
    add.s64 %rd6, %rd2, %rd4; ld.global.f32 %f2, [%rd6];   // y
    mul.f32 %f3, %f2, %f2;                                 // y*y
    sub.f32 %f4, 0f3F800000, %f3;                          // 1 - y^2
    mul.f32 %f5, %f1, %f4;
    add.s64 %rd7, %rd3, %rd4; st.global.f32 [%rd7], %f5;
    add.s32 %r5, %r5, %r7; bra TANH_L;
TANH_E:
    ret;
}

.visible .entry exp_bwd(
    .param .u64 ab_dout, .param .u64 ab_x, .param .u64 ab_y, .param .u64 ab_dx, .param .u32 ab_n
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<8>;
    .reg .f32   %f<5>;
    ld.param.u64    %rd1, [ab_dout];
    ld.param.u64    %rd2, [ab_y];
    ld.param.u64    %rd3, [ab_dx];
    ld.param.u32    %r1,  [ab_n];
    cvta.to.global.u64  %rd1, %rd1;
    cvta.to.global.u64  %rd2, %rd2;
    cvta.to.global.u64  %rd3, %rd3;
    mov.u32 %r2, %ntid.x; mov.u32 %r3, %ctaid.x; mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r3, %r2, %r4;
    mov.u32 %r6, %nctaid.x; mul.lo.s32 %r7, %r2, %r6;
EXP_L:
    setp.ge.s32 %p1, %r5, %r1; @%p1 bra EXP_E;
    mul.wide.s32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4; ld.global.f32 %f1, [%rd5];   // dout
    add.s64 %rd6, %rd2, %rd4; ld.global.f32 %f2, [%rd6];   // y = exp(x)
    mul.f32 %f3, %f1, %f2;
    add.s64 %rd7, %rd3, %rd4; st.global.f32 [%rd7], %f3;
    add.s32 %r5, %r5, %r7; bra EXP_L;
EXP_E:
    ret;
}
"#;

/// The PTX entry name for the activation-backward of op `op` (a `VM_*` code).
fn act_bwd_entry(op: i64) -> &'static str {
    use mercury_runtime::{VM_EXP, VM_RELU, VM_SIGMOID, VM_TANH};
    match op {
        x if x == VM_RELU => "relu_bwd",
        x if x == VM_SIGMOID => "sigmoid_bwd",
        x if x == VM_TANH => "tanh_bwd",
        x if x == VM_EXP => "exp_bwd",
        _ => panic!("act_bwd op {op} not implemented on GPU"),
    }
}

/// Whether [`act_bwd_f32`] has a GPU kernel for activation-backward op `op`.
pub fn act_bwd_supported(op: i64) -> bool {
    use mercury_runtime::{VM_EXP, VM_RELU, VM_SIGMOID, VM_TANH};
    op == VM_RELU || op == VM_SIGMOID || op == VM_TANH || op == VM_EXP
}

/// Elementwise activation-backward on the GPU: `dx[i] = dout[i] * f'(·)`. `x` is the forward input
/// (used by relu); `y` is the forward output (used by sigmoid/tanh/exp). Host-slice wrapper.
pub fn act_bwd_f32(
    g: &mut Gpu,
    op: i64,
    dout: &[f32],
    x: &[f32],
    y: &[f32],
) -> Result<Vec<f32>, DriverError> {
    let n = dout.len();
    assert_eq!(x.len(), n);
    assert_eq!(y.len(), n);
    let entry = act_bwd_entry(op);
    let f = g.function("act_bwd", ACT_BWD_PTX, entry)?;
    let dout_d = g.stream.memcpy_stod(dout)?;
    let x_d = g.stream.memcpy_stod(x)?;
    let y_d = g.stream.memcpy_stod(y)?;
    let mut dx_d = g.stream.memcpy_stod(&vec![0f32; n])?;
    let n_u = n as u32;
    let cfg = grid_stride_cfg(g, n_u);
    let mut b = g.stream.launch_builder(&f);
    b.arg(&dout_d).arg(&x_d).arg(&y_d).arg(&mut dx_d).arg(&n_u);
    unsafe { b.launch(cfg)? };
    g.stream.memcpy_dtov(&dx_d)
}

// ----------------------------------------------------------------------------------------------
// Row-norm backward: softmax / LayerNorm / RMSNorm (tape.rs::{softmax_back,layernorm_back,
// rmsnorm_back}). One **warp per row** (block_dim=32, grid=rows): the 32 lanes stride the row to
// form the per-row reductions, all-reduce them with `shfl.sync.bfly` (fixed butterfly order ->
// deterministic), then a second strided pass writes `dx`. Math, per row:
//   softmax:   dx = y * (dy - sum_j dy_j*y_j)
//   layernorm: mu=mean(x), var=mean(x^2)-mu^2, sigma=sqrt(var+eps), inv=1/sigma,
//              dx = inv*(dy - mean(dy) - y*mean(dy*y))           (sigma recomputed from x)
//   rmsnorm:   r=sqrt(mean(x^2)+eps), inv=1/r,
//              dx = inv*(dy - y*mean(dy*y))
// Reductions reassociate vs the CPU tape, so this is tolerance-gated (c*sqrt(cols)*eps).
// ----------------------------------------------------------------------------------------------

/// Warp butterfly all-reduce (add) of `%{reg}`, scratch `%rt` — every lane ends with the row sum.
fn allreduce_add(reg: &str) -> String {
    let mut s = String::new();
    for off in [16, 8, 4, 2, 1] {
        s += &format!("    shfl.sync.bfly.b32 %rt, %{reg}, {off}, 0x1f, 0xffffffff;\n");
        s += &format!("    add.f32 %{reg}, %{reg}, %rt;\n");
    }
    s
}

/// A strided pass `for (i = lane; i < cols; i += 32)` running `body`, which addresses element `i`
/// via the byte offset `%off` added to a row-base pointer (`%dyp`/`%xp`/`%yp`/`%dxp`). `tag` makes
/// labels unique.
fn strided(tag: &str, body: &str) -> String {
    format!(
        "    mov.u32 %i,%lane;\nL_{tag}:\n    setp.ge.u32 %p0,%i,%cols;\n    @%p0 bra E_{tag};\n    mul.wide.u32 %off,%i,4;\n{body}    add.u32 %i,%i,32;\n    bra L_{tag};\nE_{tag}:\n"
    )
}

/// Common backward prologue: one warp per row; loads `(rows, cols, eps)` and the four buffers
/// `(dy, x, y, dx)`, computes the row-base pointers, bails if `row >= rows`.
fn bwd_prologue(name: &str) -> String {
    format!(
        r#".visible .entry {name}(
    .param .u32 pRows,
    .param .u32 pCols,
    .param .f32 pEps,
    .param .u64 pDy,
    .param .u64 pX,
    .param .u64 pY,
    .param .u64 pDx
)
{{
    .reg .pred %p0;
    .reg .f32 %rt,%vx,%vy,%vdy,%sx,%sx2,%sdy,%sdyy,%s,%mu,%ex2,%var,%denom,%inv,%mdy,%mdyy,%eps,%colsf,%t1,%res;
    .reg .b32 %rows,%cols,%row,%lane,%i,%tmp;
    .reg .b64 %DY,%X,%Y,%DX,%dyp,%xp,%yp,%dxp,%off,%a;
    ld.param.u32 %rows,[pRows];
    ld.param.u32 %cols,[pCols];
    ld.param.f32 %eps,[pEps];
    ld.param.u64 %DY,[pDy];
    ld.param.u64 %X,[pX];
    ld.param.u64 %Y,[pY];
    ld.param.u64 %DX,[pDx];
    cvta.to.global.u64 %DY,%DY;
    cvta.to.global.u64 %X,%X;
    cvta.to.global.u64 %Y,%Y;
    cvta.to.global.u64 %DX,%DX;
    mov.u32 %row,%ctaid.x;
    setp.ge.u32 %p0,%row,%rows;
    @%p0 bra RET_{name};
    mov.u32 %lane,%tid.x;
    cvt.rn.f32.u32 %colsf,%cols;
    mul.lo.s32 %tmp,%row,%cols;
    mul.wide.u32 %off,%tmp,4;
    add.s64 %dyp,%DY,%off;
    add.s64 %xp,%X,%off;
    add.s64 %yp,%Y,%off;
    add.s64 %dxp,%DX,%off;
"#
    )
}

/// softmax backward: `dx = y * (dy - sum_j dy_j*y_j)`.
fn softmax_bwd() -> String {
    let mut s = bwd_prologue("softmax_bwd");
    s += "    mov.f32 %s,0f00000000;\n";
    s += &strided(
        "sbr",
        "    add.s64 %a,%dyp,%off;\n    ld.global.f32 %vdy,[%a];\n    add.s64 %a,%yp,%off;\n    ld.global.f32 %vy,[%a];\n    fma.rn.f32 %s,%vdy,%vy,%s;\n",
    );
    s += &allreduce_add("s");
    s += &strided(
        "sbw",
        "    add.s64 %a,%dyp,%off;\n    ld.global.f32 %vdy,[%a];\n    add.s64 %a,%yp,%off;\n    ld.global.f32 %vy,[%a];\n    sub.f32 %t1,%vdy,%s;\n    mul.f32 %res,%vy,%t1;\n    add.s64 %a,%dxp,%off;\n    st.global.f32 [%a],%res;\n",
    );
    s += "RET_softmax_bwd:\n    ret;\n}\n";
    s
}

/// LayerNorm backward: `dx = (1/sigma)*(dy - mean(dy) - y*mean(dy*y))`, sigma recomputed from x.
fn layernorm_bwd() -> String {
    let mut s = bwd_prologue("layernorm_bwd");
    s += "    mov.f32 %sx,0f00000000;\n    mov.f32 %sx2,0f00000000;\n    mov.f32 %sdy,0f00000000;\n    mov.f32 %sdyy,0f00000000;\n";
    s += &strided(
        "lbr",
        "    add.s64 %a,%xp,%off;\n    ld.global.f32 %vx,[%a];\n    add.s64 %a,%dyp,%off;\n    ld.global.f32 %vdy,[%a];\n    add.s64 %a,%yp,%off;\n    ld.global.f32 %vy,[%a];\n    add.f32 %sx,%sx,%vx;\n    fma.rn.f32 %sx2,%vx,%vx,%sx2;\n    add.f32 %sdy,%sdy,%vdy;\n    fma.rn.f32 %sdyy,%vdy,%vy,%sdyy;\n",
    );
    s += &allreduce_add("sx");
    s += &allreduce_add("sx2");
    s += &allreduce_add("sdy");
    s += &allreduce_add("sdyy");
    s += "    div.rn.f32 %mu,%sx,%colsf;\n";
    s += "    div.rn.f32 %ex2,%sx2,%colsf;\n";
    s += "    mul.f32 %t1,%mu,%mu;\n    sub.f32 %var,%ex2,%t1;\n";
    s += "    add.f32 %denom,%var,%eps;\n    sqrt.rn.f32 %denom,%denom;\n";
    s += "    mov.f32 %t1,0f3F800000;\n    div.rn.f32 %inv,%t1,%denom;\n";
    s += "    div.rn.f32 %mdy,%sdy,%colsf;\n";
    s += "    div.rn.f32 %mdyy,%sdyy,%colsf;\n";
    s += &strided(
        "lbw",
        "    add.s64 %a,%dyp,%off;\n    ld.global.f32 %vdy,[%a];\n    add.s64 %a,%yp,%off;\n    ld.global.f32 %vy,[%a];\n    mul.f32 %res,%vy,%mdyy;\n    sub.f32 %t1,%vdy,%mdy;\n    sub.f32 %t1,%t1,%res;\n    mul.f32 %res,%inv,%t1;\n    add.s64 %a,%dxp,%off;\n    st.global.f32 [%a],%res;\n",
    );
    s += "RET_layernorm_bwd:\n    ret;\n}\n";
    s
}

/// RMSNorm backward: `dx = (1/r)*(dy - y*mean(dy*y))`, r recomputed from x.
fn rmsnorm_bwd() -> String {
    let mut s = bwd_prologue("rmsnorm_bwd");
    s += "    mov.f32 %sx2,0f00000000;\n    mov.f32 %sdyy,0f00000000;\n";
    s += &strided(
        "rbr",
        "    add.s64 %a,%xp,%off;\n    ld.global.f32 %vx,[%a];\n    add.s64 %a,%dyp,%off;\n    ld.global.f32 %vdy,[%a];\n    add.s64 %a,%yp,%off;\n    ld.global.f32 %vy,[%a];\n    fma.rn.f32 %sx2,%vx,%vx,%sx2;\n    fma.rn.f32 %sdyy,%vdy,%vy,%sdyy;\n",
    );
    s += &allreduce_add("sx2");
    s += &allreduce_add("sdyy");
    s += "    div.rn.f32 %ex2,%sx2,%colsf;\n";
    s += "    add.f32 %denom,%ex2,%eps;\n    sqrt.rn.f32 %denom,%denom;\n";
    s += "    mov.f32 %t1,0f3F800000;\n    div.rn.f32 %inv,%t1,%denom;\n";
    s += "    div.rn.f32 %mdyy,%sdyy,%colsf;\n";
    s += &strided(
        "rbw",
        "    add.s64 %a,%dyp,%off;\n    ld.global.f32 %vdy,[%a];\n    add.s64 %a,%yp,%off;\n    ld.global.f32 %vy,[%a];\n    mul.f32 %res,%vy,%mdyy;\n    sub.f32 %t1,%vdy,%res;\n    mul.f32 %res,%inv,%t1;\n    add.s64 %a,%dxp,%off;\n    st.global.f32 [%a],%res;\n",
    );
    s += "RET_rmsnorm_bwd:\n    ret;\n}\n";
    s
}

/// The row-norm backward module (softmax / layernorm / rmsnorm), generated once and cached.
pub fn norm_bwd_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
        m += &softmax_bwd();
        m += &layernorm_bwd();
        m += &rmsnorm_bwd();
        m
    })
    .as_str()
}

/// Whether [`norm_bwd_f32`] has a GPU kernel for row-norm backward op `op`.
pub fn norm_bwd_supported(op: i64) -> bool {
    op == NORM_SOFTMAX || op == NORM_LAYERNORM || op == NORM_RMSNORM
}

/// Row-norm backward on the GPU. `dy` is the output gradient, `x` the forward input (softmax ignores
/// it), `y` the forward normalized output; returns `dx` (`rows*cols`). `eps` matches the forward
/// norm (softmax ignores it). One warp per row.
#[allow(clippy::too_many_arguments)]
pub fn norm_bwd_f32(
    g: &mut Gpu,
    op: i64,
    dy: &[f32],
    x: &[f32],
    y: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<Vec<f32>, DriverError> {
    let n = rows * cols;
    assert_eq!(dy.len(), n, "norm_bwd: dy must be rows*cols");
    assert_eq!(x.len(), n, "norm_bwd: x must be rows*cols");
    assert_eq!(y.len(), n, "norm_bwd: y must be rows*cols");
    let entry = match op {
        NORM_SOFTMAX => "softmax_bwd",
        NORM_LAYERNORM => "layernorm_bwd",
        NORM_RMSNORM => "rmsnorm_bwd",
        _ => panic!("norm_bwd op {op} not implemented"),
    };
    let f = g.function("norm_bwd", norm_bwd_ptx(), entry)?;
    let dy_d = g.stream.memcpy_stod(dy)?;
    let x_d = g.stream.memcpy_stod(x)?;
    let y_d = g.stream.memcpy_stod(y)?;
    let mut dx_d = g.stream.memcpy_stod(&vec![0f32; n])?;
    let (rows_u, cols_u) = (rows as u32, cols as u32);
    let cfg = LaunchConfig {
        grid_dim: (rows_u, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = g.stream.launch_builder(&f);
    b.arg(&rows_u)
        .arg(&cols_u)
        .arg(&eps)
        .arg(&dy_d)
        .arg(&x_d)
        .arg(&y_d)
        .arg(&mut dx_d);
    unsafe { b.launch(cfg)? };
    g.stream.memcpy_dtov(&dx_d)
}

// ----------------------------------------------------------------------------------------------
// Training GEMM: C[m×n] = op(A)·op(B), with per-operand transpose (NN / NT / TN). The forward
// Linear is NT (y = x·Wᵀ); the tape's gradient matmuls are NN (dX = dY·W) and TN (dW = dYᵀ·X).
// One transposable kernel covers all three so the resident step needs no operand shuffling.
//
// This is a **naive** one-thread-per-output-element kernel (a k-loop of fused MACs) — kept as a
// small self-contained host reference (`gemm_f32`, gated by `train_gemm_matches_reference`). The
// resident training step does NOT use it: `gemm_device` (below) routes to the register-blocked
// `ptx_gemm` kernel (~3-5× faster) — the M8 GEMM lever. `fma.rn` accumulation in increasing k
// matches a naive CPU reference closely; tolerance-gated (the reduction reassociates).
// ----------------------------------------------------------------------------------------------

/// One naive GEMM entry for transpose mode `(ta, tb)`: `C[m×n] = opA(A)·opB(B)`, inner dim `k`.
/// `A` is `[m×k]` (ta=false) or `[k×m]` (ta=true); `B` is `[k×n]` (tb=false) or `[n×k]` (tb=true).
fn gemm_entry(tag: &str, ta: bool, tb: bool) -> String {
    let aidx = if ta {
        "    mad.lo.s32 %aidx, %l, %M, %row;\n" // A^T: A[l*M + row]
    } else {
        "    mad.lo.s32 %aidx, %row, %K, %l;\n" // A:   A[row*K + l]
    };
    let bidx = if tb {
        "    mad.lo.s32 %bidx, %col, %K, %l;\n" // B^T: B[col*K + l]
    } else {
        "    mad.lo.s32 %bidx, %l, %N, %col;\n" // B:   B[l*N + col]
    };
    format!(
        r#".visible .entry gemm_{tag}(
    .param .u64 gA, .param .u64 gB, .param .u64 gC,
    .param .u32 gM, .param .u32 gN, .param .u32 gK
)
{{
    .reg .pred %p<4>;
    .reg .b32 %M,%N,%K,%col,%row,%l,%tx,%ty,%cx,%cy,%ntx,%nty,%aidx,%bidx,%cidx;
    .reg .b64 %A,%B,%C,%off,%a;
    .reg .f32 %acc,%av,%bv;
    ld.param.u64 %A,[gA]; ld.param.u64 %B,[gB]; ld.param.u64 %C,[gC];
    ld.param.u32 %M,[gM]; ld.param.u32 %N,[gN]; ld.param.u32 %K,[gK];
    cvta.to.global.u64 %A,%A; cvta.to.global.u64 %B,%B; cvta.to.global.u64 %C,%C;
    mov.u32 %ntx,%ntid.x; mov.u32 %cx,%ctaid.x; mov.u32 %tx,%tid.x;
    mad.lo.s32 %col,%cx,%ntx,%tx;
    mov.u32 %nty,%ntid.y; mov.u32 %cy,%ctaid.y; mov.u32 %ty,%tid.y;
    mad.lo.s32 %row,%cy,%nty,%ty;
    setp.ge.u32 %p1,%row,%M; setp.ge.u32 %p2,%col,%N; or.pred %p3,%p1,%p2;
    @%p3 bra GEMM_END_{tag};
    mov.f32 %acc,0f00000000;
    mov.u32 %l,0;
GEMM_L_{tag}:
    setp.ge.u32 %p1,%l,%K; @%p1 bra GEMM_W_{tag};
{aidx}    mul.wide.u32 %off,%aidx,4; add.s64 %a,%A,%off; ld.global.f32 %av,[%a];
{bidx}    mul.wide.u32 %off,%bidx,4; add.s64 %a,%B,%off; ld.global.f32 %bv,[%a];
    fma.rn.f32 %acc,%av,%bv,%acc;
    add.u32 %l,%l,1; bra GEMM_L_{tag};
GEMM_W_{tag}:
    mad.lo.s32 %cidx,%row,%N,%col;
    mul.wide.u32 %off,%cidx,4; add.s64 %a,%C,%off; st.global.f32 [%a],%acc;
GEMM_END_{tag}:
    ret;
}}
"#
    )
}

/// The training-GEMM module (gemm_nn / gemm_nt / gemm_tn), generated once and cached.
pub fn train_gemm_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
        m += &gemm_entry("nn", false, false);
        m += &gemm_entry("nt", false, true);
        m += &gemm_entry("tn", true, false);
        m
    })
    .as_str()
}

/// PTX entry name for transpose mode `(ta, tb)` — only NN / NT / TN are generated.
pub(crate) fn gemm_entry_name(ta: bool, tb: bool) -> &'static str {
    match (ta, tb) {
        (false, false) => "gemm_nn",
        (false, true) => "gemm_nt",
        (true, false) => "gemm_tn",
        (true, true) => panic!("gemm_tt not generated"),
    }
}

/// Launch config for the 16×16-tiled GEMM grid (one thread per output element).
pub(crate) fn gemm_cfg(m: usize, n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n.div_ceil(16)) as u32, (m.div_ceil(16)) as u32, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    }
}

/// `C[m×n] = opA(A)·opB(B)` on the GPU (host-slice convenience wrapper). `ta`/`tb` transpose A/B.
pub fn gemm_f32(
    g: &mut Gpu,
    ta: bool,
    tb: bool,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, DriverError> {
    assert_eq!(a.len(), m * k, "gemm A must be m*k elements (whatever its layout)");
    assert_eq!(b.len(), k * n, "gemm B must be k*n elements");
    let f = g.function("train_gemm", train_gemm_ptx(), gemm_entry_name(ta, tb))?;
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mu, nu, ku) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&a_d).arg(&b_d).arg(&mut c_d).arg(&mu).arg(&nu).arg(&ku);
    unsafe { bld.launch(gemm_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

// ----------------------------------------------------------------------------------------------
// Forward elementwise glue the resident training step needs (relu forward; MSE-loss gradient seed).
// ----------------------------------------------------------------------------------------------

/// `relu_fwd(x, out, n)`: `out = max(x, 0)`. `mse_grad(y, t, dy, n, scale)`: `dy = scale*(y - t)` —
/// the gradient of `scale/2 * sum (y-t)^2` (scale=2 reproduces the SSD loss the tape differentiates).
pub const TRAIN_ELEM_PTX: &str = r#"
.version 7.8
.target sm_89
.address_size 64

.visible .entry relu_fwd(.param .u64 rx, .param .u64 rout, .param .u32 rn)
{
    .reg .pred %p<2>;
    .reg .b32  %r<8>;
    .reg .b64  %rd<6>;
    .reg .f32  %f<3>;
    ld.param.u64 %rd1,[rx]; ld.param.u64 %rd2,[rout]; ld.param.u32 %r1,[rn];
    cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;
    mov.u32 %r2,%ntid.x; mov.u32 %r3,%ctaid.x; mov.u32 %r4,%tid.x;
    mad.lo.s32 %r5,%r3,%r2,%r4; mov.u32 %r6,%nctaid.x; mul.lo.s32 %r7,%r2,%r6;
RF_L:
    setp.ge.s32 %p1,%r5,%r1; @%p1 bra RF_E;
    mul.wide.s32 %rd3,%r5,4;
    add.s64 %rd4,%rd1,%rd3; ld.global.f32 %f1,[%rd4];
    max.f32 %f2,%f1,0f00000000;
    add.s64 %rd5,%rd2,%rd3; st.global.f32 [%rd5],%f2;
    add.s32 %r5,%r5,%r7; bra RF_L;
RF_E:
    ret;
}

.visible .entry mse_grad(.param .u64 my, .param .u64 mt, .param .u64 mdy, .param .u32 mn, .param .f32 ms)
{
    .reg .pred %p<2>;
    .reg .b32  %r<8>;
    .reg .b64  %rd<8>;
    .reg .f32  %f<5>;
    ld.param.u64 %rd1,[my]; ld.param.u64 %rd2,[mt]; ld.param.u64 %rd3,[mdy];
    ld.param.u32 %r1,[mn]; ld.param.f32 %f1,[ms];
    cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2; cvta.to.global.u64 %rd3,%rd3;
    mov.u32 %r2,%ntid.x; mov.u32 %r3,%ctaid.x; mov.u32 %r4,%tid.x;
    mad.lo.s32 %r5,%r3,%r2,%r4; mov.u32 %r6,%nctaid.x; mul.lo.s32 %r7,%r2,%r6;
MG_L:
    setp.ge.s32 %p1,%r5,%r1; @%p1 bra MG_E;
    mul.wide.s32 %rd4,%r5,4;
    add.s64 %rd5,%rd1,%rd4; ld.global.f32 %f2,[%rd5];
    add.s64 %rd6,%rd2,%rd4; ld.global.f32 %f3,[%rd6];
    sub.f32 %f4,%f2,%f3; mul.f32 %f4,%f1,%f4;
    add.s64 %rd7,%rd3,%rd4; st.global.f32 [%rd7],%f4;
    add.s32 %r5,%r5,%r7; bra MG_L;
MG_E:
    ret;
}

.visible .entry scale_inplace(.param .u64 sx, .param .u32 sn, .param .f32 ss)
{
    .reg .pred %p<2>;
    .reg .b32  %r<8>;
    .reg .b64  %rd<5>;
    .reg .f32  %f<3>;
    ld.param.u64 %rd1,[sx]; ld.param.u32 %r1,[sn]; ld.param.f32 %f1,[ss];
    cvta.to.global.u64 %rd1,%rd1;
    mov.u32 %r2,%ntid.x; mov.u32 %r3,%ctaid.x; mov.u32 %r4,%tid.x;
    mad.lo.s32 %r5,%r3,%r2,%r4; mov.u32 %r6,%nctaid.x; mul.lo.s32 %r7,%r2,%r6;
SI_L:
    setp.ge.s32 %p1,%r5,%r1; @%p1 bra SI_E;
    mul.wide.s32 %rd2,%r5,4; add.s64 %rd3,%rd1,%rd2;
    ld.global.f32 %f2,[%rd3]; mul.f32 %f2,%f2,%f1; st.global.f32 [%rd3],%f2;
    add.s32 %r5,%r5,%r7; bra SI_L;
SI_E:
    ret;
}

.visible .entry causal_mask(.param .u64 cm, .param .u32 cs)
{
    .reg .pred %p<3>;
    .reg .b32  %r<12>;
    .reg .b64  %rd<5>;
    .reg .f32  %f<2>;
    ld.param.u64 %rd1,[cm]; ld.param.u32 %r1,[cs];
    cvta.to.global.u64 %rd1,%rd1;
    mul.lo.s32 %r2,%r1,%r1;
    mov.u32 %r3,%ntid.x; mov.u32 %r4,%ctaid.x; mov.u32 %r5,%tid.x;
    mad.lo.s32 %r6,%r4,%r3,%r5; mov.u32 %r7,%nctaid.x; mul.lo.s32 %r8,%r3,%r7;
CM_L:
    setp.ge.s32 %p1,%r6,%r2; @%p1 bra CM_E;
    div.u32 %r9,%r6,%r1; mul.lo.s32 %r10,%r9,%r1; sub.s32 %r10,%r6,%r10; // row=idx/S, col=idx-row*S
    setp.gt.s32 %p2,%r10,%r9;  // col > row -> mask
    @!%p2 bra CM_N;
    mul.wide.s32 %rd2,%r6,4; add.s64 %rd3,%rd1,%rd2;
    mov.f32 %f1,0fFF800000; st.global.f32 [%rd3],%f1;
CM_N:
    add.s32 %r6,%r6,%r8; bra CM_L;
CM_E:
    ret;
}
"#;

// ----------------------------------------------------------------------------------------------
// Device-pointer launch helpers (no host transfer) — the resident training step chains these over
// device buffers it keeps alive across forward + backward + optimizer.
// ----------------------------------------------------------------------------------------------

/// Transpose `src(m×n)` -> `dst(n×m)` over device buffers (the device twin of [`transpose_f32`]) —
/// used to turn a TN GEMM (`Aᵀ·B`) into a transpose + NN, so all matmuls ride the reg-blocked kernel.
pub fn transpose_device(
    g: &mut Gpu,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    m: usize,
    n: usize,
) -> Result<(), DriverError> {
    let f = g.function("transpose_f32", TRANSPOSE_F32_PTX, "transpose_f32")?;
    let (mu, nu) = (m as u32, n as u32);
    let cfg = grid_stride_cfg(g, (m * n) as u32);
    let mut b = g.stream.launch_builder(&f);
    b.arg(src).arg(dst).arg(&mu).arg(&nu);
    unsafe { b.launch(cfg)? };
    Ok(())
}

/// Launch a register-blocked GEMM entry (`gemm_nn_rb` / `gemm_nt_rb`) from `ptx_gemm` (read-only):
/// `C[m×n] = A·op(B)`, params `(M,N,K,A,B,C)`, one 64×64 CTA tile of 16×16 threads.
fn gemm_rb(
    g: &mut Gpu,
    entry: &str,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), DriverError> {
    let f = g.function("gemm_rb", gemm_rb_ptx(), entry)?;
    let (mu, nu, ku) = (m as u32, n as u32, k as u32);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(TILE_N), (m as u32).div_ceil(TILE_M), 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mu).arg(&nu).arg(&ku).arg(a).arg(b).arg(c);
    unsafe { bld.launch(cfg)? };
    Ok(())
}

/// `C[m×n] = opA(A)·opB(B)` over device buffers — the resident training GEMM. NN and NT ride the
/// register-blocked `ptx_gemm` kernel directly (~3-5× the naive path); TN (`Aᵀ·B`, the weight-gradient
/// shape) transposes A (`[k×m]→[m×k]`, cheap vs the O(mnk) GEMM) then NN. Gated end-to-end by the
/// MLP-gradient and attention-backward references (all three modes exercised).
#[allow(clippy::too_many_arguments)]
pub fn gemm_device(
    g: &mut Gpu,
    ta: bool,
    tb: bool,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), DriverError> {
    match (ta, tb) {
        (false, false) => gemm_rb(g, "gemm_nn_rb", a, b, c, m, n, k),
        (false, true) => gemm_rb(g, "gemm_nt_rb", a, b, c, m, n, k),
        (true, false) => {
            // C = Aᵀ·B with A stored [k×m]: transpose to At[m×k], then NN(At, B).
            let mut at = g.stream.alloc_zeros::<f32>(m * k)?;
            transpose_device(g, a, &mut at, k, m)?;
            gemm_rb(g, "gemm_nn_rb", &at, b, c, m, n, k)
        }
        (true, true) => panic!("gemm TT not supported"),
    }
}

/// `dx = dout ⊙ f'(·)` over device buffers (resident activation-backward; see [`act_bwd_f32`]).
pub fn act_bwd_device(
    g: &mut Gpu,
    op: i64,
    dout: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    y: &CudaSlice<f32>,
    dx: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), DriverError> {
    let f = g.function("act_bwd", ACT_BWD_PTX, act_bwd_entry(op))?;
    let n_u = n as u32;
    let cfg = grid_stride_cfg(g, n_u);
    let mut b = g.stream.launch_builder(&f);
    b.arg(dout).arg(x).arg(y).arg(dx).arg(&n_u);
    unsafe { b.launch(cfg)? };
    Ok(())
}

/// `out = max(x, 0)` over device buffers (resident relu forward).
pub fn relu_fwd_device(
    g: &mut Gpu,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), DriverError> {
    let f = g.function("train_elem", TRAIN_ELEM_PTX, "relu_fwd")?;
    let n_u = n as u32;
    let cfg = grid_stride_cfg(g, n_u);
    let mut b = g.stream.launch_builder(&f);
    b.arg(x).arg(out).arg(&n_u);
    unsafe { b.launch(cfg)? };
    Ok(())
}

/// `dy = scale*(y - t)` over device buffers (resident MSE-loss gradient seed).
pub fn mse_grad_device(
    g: &mut Gpu,
    y: &CudaSlice<f32>,
    t: &CudaSlice<f32>,
    dy: &mut CudaSlice<f32>,
    n: usize,
    scale: f32,
) -> Result<(), DriverError> {
    let f = g.function("train_elem", TRAIN_ELEM_PTX, "mse_grad")?;
    let n_u = n as u32;
    let cfg = grid_stride_cfg(g, n_u);
    let mut b = g.stream.launch_builder(&f);
    b.arg(y).arg(t).arg(dy).arg(&n_u).arg(&scale);
    unsafe { b.launch(cfg)? };
    Ok(())
}

/// `x *= s` over a device buffer.
pub fn scale_inplace_device(
    g: &mut Gpu,
    x: &mut CudaSlice<f32>,
    n: usize,
    s: f32,
) -> Result<(), DriverError> {
    let f = g.function("train_elem", TRAIN_ELEM_PTX, "scale_inplace")?;
    let n_u = n as u32;
    let cfg = grid_stride_cfg(g, n_u);
    let mut b = g.stream.launch_builder(&f);
    b.arg(x).arg(&n_u).arg(&s);
    unsafe { b.launch(cfg)? };
    Ok(())
}

/// Apply a causal mask to a square `s×s` score matrix in place: `S[i,j] = -inf` for `j > i` (each
/// query attends only to keys at or before its position — the decoder mask).
pub fn causal_mask_device(g: &mut Gpu, x: &mut CudaSlice<f32>, s: usize) -> Result<(), DriverError> {
    let f = g.function("train_elem", TRAIN_ELEM_PTX, "causal_mask")?;
    let s_u = s as u32;
    let cfg = grid_stride_cfg(g, (s * s) as u32);
    let mut b = g.stream.launch_builder(&f);
    b.arg(x).arg(&s_u);
    unsafe { b.launch(cfg)? };
    Ok(())
}

/// Row-softmax **forward** over device buffers. Reuses `ptx_norm`'s softmax entry (read, never
/// edited) so the attention backward recomputes P with the exact forward softmax.
pub fn softmax_fwd_device(
    g: &mut Gpu,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    rows: usize,
    cols: usize,
) -> Result<(), DriverError> {
    let f = g.function("norm_fwd", crate::ptx_norm::norm_ptx(), "softmax")?;
    let (r, c, eps) = (rows as u32, cols as u32, 0f32);
    let cfg = LaunchConfig {
        grid_dim: (r, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = g.stream.launch_builder(&f);
    b.arg(&r).arg(&c).arg(&eps).arg(x).arg(out);
    unsafe { b.launch(cfg)? };
    Ok(())
}

/// Row-norm **backward** over device buffers (see [`norm_bwd_f32`]). For softmax, `x` is ignored.
#[allow(clippy::too_many_arguments)]
pub fn norm_bwd_device(
    g: &mut Gpu,
    op: i64,
    dy: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    y: &CudaSlice<f32>,
    dx: &mut CudaSlice<f32>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<(), DriverError> {
    let entry = match op {
        NORM_SOFTMAX => "softmax_bwd",
        NORM_LAYERNORM => "layernorm_bwd",
        NORM_RMSNORM => "rmsnorm_bwd",
        _ => panic!("norm_bwd op {op} not implemented"),
    };
    let f = g.function("norm_bwd", norm_bwd_ptx(), entry)?;
    let (r, c) = (rows as u32, cols as u32);
    let cfg = LaunchConfig {
        grid_dim: (r, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = g.stream.launch_builder(&f);
    b.arg(&r).arg(&c).arg(&eps).arg(dy).arg(x).arg(y).arg(dx);
    unsafe { b.launch(cfg)? };
    Ok(())
}

// ----------------------------------------------------------------------------------------------
// Flash-attention backward (dQ/dK/dV). With the forward S = scale·Q·Kᵀ, P = softmax(S), O = P·V,
// the backward decomposes entirely into kernels we already have + the softmax backward:
//   P  = softmax(scale·Q·Kᵀ)            (recomputed: gemm_nt + scale + ptx_norm softmax)
//   dV = Pᵀ·dO                          (gemm_tn)
//   dP = dO·Vᵀ                          (gemm_nt)
//   dS = softmax_bwd(P, dP)             (the row-softmax backward kernel above)
//   dQ = scale·(dS·K)                   (gemm_nn + scale)
//   dK = scale·(dSᵀ·Q)                  (gemm_tn + scale)
// This is the materialized (non-flash) form: correct, deterministic, built from gated primitives,
// O(S²) workspace. The fused/tiled flash backward (no S×S materialization) is the perf/long-S lever.
// ----------------------------------------------------------------------------------------------

/// Flash-attention backward on the GPU. Inputs `q`/`k`/`v`/`do` are `[S×d]` row-major; `scale` is the
/// forward's softmax scale (typically `1/sqrt(d)`); `causal` applies the decoder mask. Returns
/// `(dQ, dK, dV)`, each `[S×d]`. Host-slice convenience wrapper (uploads, runs the resident kernel
/// chain, downloads).
#[allow(clippy::too_many_arguments)]
pub fn attention_backward(
    g: &mut Gpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_o: &[f32],
    s: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), DriverError> {
    let nd = s * d;
    let n2 = s * s;
    assert_eq!(q.len(), nd);
    assert_eq!(k.len(), nd);
    assert_eq!(v.len(), nd);
    assert_eq!(d_o.len(), nd);
    let st = g.stream.clone();
    let q_d = st.memcpy_stod(q)?;
    let k_d = st.memcpy_stod(k)?;
    let v_d = st.memcpy_stod(v)?;
    let do_d = st.memcpy_stod(d_o)?;
    let mut s_d = st.alloc_zeros::<f32>(n2)?;
    let mut p_d = st.alloc_zeros::<f32>(n2)?;
    let mut dp_d = st.alloc_zeros::<f32>(n2)?;
    let mut ds_d = st.alloc_zeros::<f32>(n2)?;
    let mut dq_d = st.alloc_zeros::<f32>(nd)?;
    let mut dk_d = st.alloc_zeros::<f32>(nd)?;
    let mut dv_d = st.alloc_zeros::<f32>(nd)?;

    // P = softmax(scale · Q·Kᵀ [+ causal mask])
    gemm_device(g, false, true, &q_d, &k_d, &mut s_d, s, s, d)?;
    scale_inplace_device(g, &mut s_d, n2, scale)?;
    if causal {
        causal_mask_device(g, &mut s_d, s)?;
    }
    softmax_fwd_device(g, &s_d, &mut p_d, s, s)?;
    // dV = Pᵀ·dO
    gemm_device(g, true, false, &p_d, &do_d, &mut dv_d, s, d, s)?;
    // dP = dO·Vᵀ
    gemm_device(g, false, true, &do_d, &v_d, &mut dp_d, s, s, d)?;
    // dS = softmax_bwd(P, dP)
    norm_bwd_device(g, NORM_SOFTMAX, &dp_d, &p_d, &p_d, &mut ds_d, s, s, 0.0)?;
    // dQ = scale · dS·K
    gemm_device(g, false, false, &ds_d, &k_d, &mut dq_d, s, d, s)?;
    scale_inplace_device(g, &mut dq_d, nd, scale)?;
    // dK = scale · dSᵀ·Q
    gemm_device(g, true, false, &ds_d, &q_d, &mut dk_d, s, d, s)?;
    scale_inplace_device(g, &mut dk_d, nd, scale)?;

    Ok((st.memcpy_dtov(&dq_d)?, st.memcpy_dtov(&dk_d)?, st.memcpy_dtov(&dv_d)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::{assert_close, Rng};
    use crate::gpu::gpu;
    use mercury_runtime::{VM_EXP, VM_RELU, VM_SIGMOID, VM_TANH};

    fn with_gpu(name: &str, body: impl FnOnce(&mut Gpu)) {
        let mut guard = gpu();
        match guard.as_mut() {
            Some(g) => body(g),
            None => eprintln!("[skip] {name}: no CUDA device reachable"),
        }
    }

    #[test]
    fn transpose_matches_reference() {
        with_gpu("transpose_matches_reference", |g| {
            for &(m, n) in &[(1usize, 1usize), (3, 5), (64, 64), (17, 130), (128, 1)] {
                let mut rng = Rng::new(0x7A05 ^ (m as u64) << 8 ^ n as u64);
                let src = rng.vec(m * n, -3.0, 3.0);
                let got = transpose_f32(g, &src, m, n).unwrap();
                let mut want = vec![0f32; m * n];
                for r in 0..m {
                    for c in 0..n {
                        want[c * m + r] = src[r * n + c];
                    }
                }
                // Pure data movement: bit-exact.
                for i in 0..m * n {
                    assert_eq!(got[i].to_bits(), want[i].to_bits(), "transpose {m}x{n} @ {i}");
                }
            }
        });
    }

    #[test]
    fn act_bwd_matches_reference() {
        with_gpu("act_bwd_matches_reference", |g| {
            let n = 4096usize;
            let mut rng = Rng::new(0x33AC);
            let dout = rng.vec(n, -2.0, 2.0);
            let x = rng.vec(n, -3.0, 3.0);

            for &op in &[VM_RELU, VM_SIGMOID, VM_TANH, VM_EXP] {
                // y = f(x): the forward output the smooth derivatives reuse.
                let y: Vec<f32> = x
                    .iter()
                    .map(|&v| match op {
                        o if o == VM_RELU => v.max(0.0),
                        o if o == VM_SIGMOID => 1.0 / (1.0 + (-v).exp()),
                        o if o == VM_TANH => v.tanh(),
                        o if o == VM_EXP => v.exp(),
                        _ => unreachable!(),
                    })
                    .collect();
                let got = act_bwd_f32(g, op, &dout, &x, &y).unwrap();
                // Reference: dout * f'(·), same op order as the tape.
                let want: Vec<f32> = (0..n)
                    .map(|i| {
                        let fp = match op {
                            o if o == VM_RELU => {
                                if x[i] > 0.0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            o if o == VM_SIGMOID => {
                                let yi = y[i];
                                yi * (1.0 - yi)
                            }
                            o if o == VM_TANH => 1.0 - y[i] * y[i],
                            o if o == VM_EXP => y[i],
                            _ => unreachable!(),
                        };
                        dout[i] * fp
                    })
                    .collect();
                assert_close(&format!("act_bwd op {op}"), &got, &want, 1e-6, 1e-6);
            }
        });
    }

    #[test]
    fn norm_bwd_matches_reference() {
        with_gpu("norm_bwd_matches_reference", |g| {
            let (rows, cols) = (16usize, 130usize); // cols not a multiple of 32 -> tail lanes
            let eps = 1e-5f32;
            let mut rng = Rng::new(0xBEEF);
            let x = rng.vec(rows * cols, -2.0, 2.0);
            let dy = rng.vec(rows * cols, -1.0, 1.0);

            for &op in &[NORM_SOFTMAX, NORM_LAYERNORM, NORM_RMSNORM] {
                // Forward output y per row (the value the backward reuses), computed in f64.
                let mut y = vec![0f32; rows * cols];
                for r in 0..rows {
                    let row = &x[r * cols..(r + 1) * cols];
                    match op {
                        NORM_SOFTMAX => {
                            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let exps: Vec<f64> = row.iter().map(|&v| ((v - m) as f64).exp()).collect();
                            let s: f64 = exps.iter().sum();
                            for c in 0..cols {
                                y[r * cols + c] = (exps[c] / s) as f32;
                            }
                        }
                        NORM_LAYERNORM => {
                            let mu = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
                            let var =
                                row.iter().map(|&v| (v as f64 - mu).powi(2)).sum::<f64>() / cols as f64;
                            let sigma = (var + eps as f64).sqrt();
                            for c in 0..cols {
                                y[r * cols + c] = ((row[c] as f64 - mu) / sigma) as f32;
                            }
                        }
                        NORM_RMSNORM => {
                            let ms = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / cols as f64;
                            let rr = (ms + eps as f64).sqrt();
                            for c in 0..cols {
                                y[r * cols + c] = (row[c] as f64 / rr) as f32;
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                let got = norm_bwd_f32(g, op, &dy, &x, &y, rows, cols, eps).unwrap();

                // f64 reference of the exact tape backward formulas.
                let mut want = vec![0f32; rows * cols];
                for r in 0..rows {
                    let xr = &x[r * cols..(r + 1) * cols];
                    let dyr = &dy[r * cols..(r + 1) * cols];
                    let yr = &y[r * cols..(r + 1) * cols];
                    let dot = |a: &[f32], b: &[f32]| -> f64 {
                        (0..cols).map(|c| a[c] as f64 * b[c] as f64).sum()
                    };
                    match op {
                        NORM_SOFTMAX => {
                            let s = dot(dyr, yr);
                            for c in 0..cols {
                                want[r * cols + c] = (yr[c] as f64 * (dyr[c] as f64 - s)) as f32;
                            }
                        }
                        NORM_LAYERNORM => {
                            let mu = xr.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
                            let ex2 = xr.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / cols as f64;
                            let inv = 1.0 / (ex2 - mu * mu + eps as f64).sqrt();
                            let mdy = dyr.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
                            let mdyy = dot(dyr, yr) / cols as f64;
                            for c in 0..cols {
                                let v = inv * (dyr[c] as f64 - mdy - yr[c] as f64 * mdyy);
                                want[r * cols + c] = v as f32;
                            }
                        }
                        NORM_RMSNORM => {
                            let ms = xr.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / cols as f64;
                            let inv = 1.0 / (ms + eps as f64).sqrt();
                            let mdyy = dot(dyr, yr) / cols as f64;
                            for c in 0..cols {
                                let v = inv * (dyr[c] as f64 - yr[c] as f64 * mdyy);
                                want[r * cols + c] = v as f32;
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                assert_close(&format!("norm_bwd op {op}"), &got, &want, 1e-4, 1e-4);
            }
        });
    }

    #[test]
    fn train_gemm_matches_reference() {
        with_gpu("train_gemm_matches_reference", |g| {
            // Non-multiples of 16 to exercise the grid edges.
            let (m, n, k) = (33usize, 40usize, 50usize);
            let mut rng = Rng::new(0x6E33);
            for &(ta, tb) in &[(false, false), (false, true), (true, false)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(k * n, -1.0, 1.0);
                let got = gemm_f32(g, ta, tb, &a, &b, m, n, k).unwrap();
                let aref = |i: usize, l: usize| if ta { a[l * m + i] } else { a[i * k + l] } as f64;
                let bref = |l: usize, j: usize| if tb { b[j * k + l] } else { b[l * n + j] } as f64;
                let mut want = vec![0f32; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0f64;
                        for l in 0..k {
                            acc += aref(i, l) * bref(l, j);
                        }
                        want[i * n + j] = acc as f32;
                    }
                }
                let mode = gemm_entry_name(ta, tb);
                assert_close(mode, &got, &want, 1e-4, 1e-3);
            }
        });
    }

    /// **The attention-backward gate:** GPU `attention_backward` vs an f64 two-pass-softmax
    /// reference computing dQ/dK/dV from the standard FA2 backward formulas, over the full tensors.
    #[test]
    fn attention_backward_matches_reference() {
        with_gpu("attention_backward_matches_reference", |g| {
            let (s, d) = (64usize, 64usize);
            let scale = 1.0 / (d as f32).sqrt();
            let mut rng = Rng::new(0xA77E);
            let q = rng.vec(s * d, -1.0, 1.0);
            let k = rng.vec(s * d, -1.0, 1.0);
            let v = rng.vec(s * d, -1.0, 1.0);
            let dout = rng.vec(s * d, -1.0, 1.0);
            let sc = scale as f64;
            for &causal in &[false, true] {
                let (dq, dk, dv) =
                    attention_backward(g, &q, &k, &v, &dout, s, d, scale, causal).unwrap();

                // P = softmax(scale · Q·Kᵀ [+ causal mask]), two-pass, in f64.
                let mut p = vec![0f64; s * s];
                for i in 0..s {
                    let mut row = vec![0f64; s];
                    let mut mx = f64::NEG_INFINITY;
                    for j in 0..s {
                        if causal && j > i {
                            row[j] = f64::NEG_INFINITY; // masked: P_ij = 0
                            continue;
                        }
                        let mut acc = 0f64;
                        for dd in 0..d {
                            acc += q[i * d + dd] as f64 * k[j * d + dd] as f64;
                        }
                        row[j] = sc * acc;
                        mx = mx.max(row[j]);
                    }
                    let mut sm = 0f64;
                    for j in 0..s {
                        row[j] = (row[j] - mx).exp();
                        sm += row[j];
                    }
                    for j in 0..s {
                        p[i * s + j] = row[j] / sm;
                    }
                }
                // dS_ij = P_ij (dP_ij - D_i), dP_ij = dO_i·V_j, D_i = sum_j P_ij dP_ij.
                let mut ds = vec![0f64; s * s];
                for i in 0..s {
                    let mut dp = vec![0f64; s];
                    let mut di = 0f64;
                    for j in 0..s {
                        let mut acc = 0f64;
                        for dd in 0..d {
                            acc += dout[i * d + dd] as f64 * v[j * d + dd] as f64;
                        }
                        dp[j] = acc;
                        di += p[i * s + j] * acc;
                    }
                    for j in 0..s {
                        ds[i * s + j] = p[i * s + j] * (dp[j] - di);
                    }
                }
                let mut rdv = vec![0f32; s * d];
                let mut rdq = vec![0f32; s * d];
                let mut rdk = vec![0f32; s * d];
                for dd in 0..d {
                    for j in 0..s {
                        let mut acc = 0f64; // dV_jd = sum_i P_ij dO_id
                        for i in 0..s {
                            acc += p[i * s + j] * dout[i * d + dd] as f64;
                        }
                        rdv[j * d + dd] = acc as f32;
                        let mut ak = 0f64; // dK_jd = scale sum_i dS_ij Q_id
                        for i in 0..s {
                            ak += ds[i * s + j] * q[i * d + dd] as f64;
                        }
                        rdk[j * d + dd] = (sc * ak) as f32;
                    }
                    for i in 0..s {
                        let mut aq = 0f64; // dQ_id = scale sum_j dS_ij K_jd
                        for j in 0..s {
                            aq += ds[i * s + j] * k[j * d + dd] as f64;
                        }
                        rdq[i * d + dd] = (sc * aq) as f32;
                    }
                }
                let tag = if causal { "causal" } else { "full" };
                assert_close(&format!("attn dV {tag}"), &dv, &rdv, 2e-3, 3e-3);
                assert_close(&format!("attn dQ {tag}"), &dq, &rdq, 2e-3, 3e-3);
                assert_close(&format!("attn dK {tag}"), &dk, &rdk, 2e-3, 3e-3);
            }
        });
    }
}
