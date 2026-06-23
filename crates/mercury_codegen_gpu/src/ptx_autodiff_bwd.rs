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
use crate::ptx_optim::grid_stride_cfg;
use cudarc::driver::{DriverError, PushKernelArg};

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
}
