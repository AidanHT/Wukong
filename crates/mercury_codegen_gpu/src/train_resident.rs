//! A **GPU-resident MLP training step** — forward + backward + optimizer with no per-op host
//! round-trip — assembled from the autodiff/optimizer kernels in [`crate::ptx_autodiff_bwd`] and
//! [`crate::ptx_optim`]. This is the M8 vehicle: it proves the whole training step (not just
//! inference) holds on the device, and it is the thing benched against PyTorch eager (increment 6).
//!
//! The net is a 2-layer MLP `X[B×I] -> W1 -> relu -> W2 -> Y[B×O]` with an MSE loss. The forward is
//! two NT GEMMs + a relu; the backward is the tape's VJP structure realized as direct kernel
//! launches — `dW2 = dYᵀ·H`, `dH = dY·W2`, `dH_pre = dH ⊙ relu'(H_pre)`, `dW1 = dH_preᵀ·X` — and the
//! update is the fused AdamW kernel. All intermediates live in device buffers the [`MlpTrainer`]
//! keeps alive across the step, so nothing crosses the bus except the inputs (once) and the scalar
//! loss (gate only). Gated: gradients vs an f64 closed-form backprop, and the loss strictly falls.

use crate::gpu::Gpu;
use crate::ptx_autodiff_bwd::{
    act_bwd_device, gemm_device, gemm_device_f16, mse_grad_device, relu_fwd_device,
};
use crate::ptx_optim::{adamw_step_device, hp};
use cudarc::driver::{CudaSlice, DriverError};
use mercury_runtime::VM_RELU;

/// GEMM precision for the resident step. `F32` runs the reg-blocked CUDA-core kernel (the bit-tight
/// default and oracle); `F16Mixed` runs the **fp16 tensor cores** — operands narrowed to f16
/// just-in-time, products f32-accumulated, **master weights/grads kept f32** so there is no gradient
/// underflow to chase and no loss scaling. The tensor-core path is the GEMM-bound step's headline
/// perf lever (f32 CUDA-core GEMM is far below the Ada tensor-core roofline).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Precision {
    F32,
    F16Mixed,
}

/// Dispatch a resident GEMM by [`Precision`]: `C[m×n] = opA(A)·opB(B)` over f32 device buffers.
#[allow(clippy::too_many_arguments)]
fn gemm_dispatch(
    prec: Precision,
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
    match prec {
        Precision::F32 => gemm_device(g, ta, tb, a, b, c, m, n, k),
        Precision::F16Mixed => gemm_device_f16(g, ta, tb, a, b, c, m, n, k),
    }
}

/// Optimizer hyperparameters for the resident step.
#[derive(Clone, Copy)]
pub struct AdamWCfg {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
}

impl Default for AdamWCfg {
    fn default() -> Self {
        AdamWCfg {
            lr: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            wd: 0.0,
        }
    }
}

/// A resident 2-layer MLP trainer. All weights/grads/moments/activations are device buffers; a
/// training [`step`](Self::step) issues only kernel launches (no host transfer of intermediates).
pub struct MlpTrainer {
    pub b: usize,
    pub i: usize,
    pub h: usize,
    pub o: usize,
    /// GEMM precision — `F32` (default) or `F16Mixed` (tensor cores). Set via [`with_precision`].
    pub prec: Precision,
    // parameters + their gradients + AdamW moments (per layer)
    w1: CudaSlice<f32>,
    w2: CudaSlice<f32>,
    dw1: CudaSlice<f32>,
    dw2: CudaSlice<f32>,
    m1: CudaSlice<f32>,
    v1: CudaSlice<f32>,
    m2: CudaSlice<f32>,
    v2: CudaSlice<f32>,
    // inputs + activations (forward) and adjoints (backward)
    x: CudaSlice<f32>,
    target: CudaSlice<f32>,
    h_pre: CudaSlice<f32>,
    hact: CudaSlice<f32>,
    y: CudaSlice<f32>,
    dy: CudaSlice<f32>,
    dh: CudaSlice<f32>,
    dh_pre: CudaSlice<f32>,
    target_host: Vec<f32>,
}

impl MlpTrainer {
    /// Allocate the resident buffers and upload the initial weights. `w1` is `[H×I]`, `w2` is `[O×H]`
    /// (the NT-Linear weight layout `y = x·Wᵀ`).
    pub fn new(
        g: &mut Gpu,
        b: usize,
        i: usize,
        h: usize,
        o: usize,
        w1: &[f32],
        w2: &[f32],
    ) -> Result<Self, DriverError> {
        assert_eq!(w1.len(), h * i, "w1 must be H*I");
        assert_eq!(w2.len(), o * h, "w2 must be O*H");
        let st = &g.stream;
        Ok(MlpTrainer {
            b,
            i,
            h,
            o,
            prec: Precision::F32,
            w1: st.memcpy_stod(w1)?,
            w2: st.memcpy_stod(w2)?,
            dw1: st.alloc_zeros::<f32>(h * i)?,
            dw2: st.alloc_zeros::<f32>(o * h)?,
            m1: st.alloc_zeros::<f32>(h * i)?,
            v1: st.alloc_zeros::<f32>(h * i)?,
            m2: st.alloc_zeros::<f32>(o * h)?,
            v2: st.alloc_zeros::<f32>(o * h)?,
            x: st.alloc_zeros::<f32>(b * i)?,
            target: st.alloc_zeros::<f32>(b * o)?,
            h_pre: st.alloc_zeros::<f32>(b * h)?,
            hact: st.alloc_zeros::<f32>(b * h)?,
            y: st.alloc_zeros::<f32>(b * o)?,
            dy: st.alloc_zeros::<f32>(b * o)?,
            dh: st.alloc_zeros::<f32>(b * h)?,
            dh_pre: st.alloc_zeros::<f32>(b * h)?,
            target_host: vec![0.0; b * o],
        })
    }

    /// Select the GEMM precision (builder style), e.g. `MlpTrainer::new(...)?.with_precision(F16Mixed)`.
    pub fn with_precision(mut self, p: Precision) -> Self {
        self.prec = p;
        self
    }

    /// Upload the training batch (`x` is `[B×I]`, `target` is `[B×O]`). For a fixed-batch loop this is
    /// called once; the per-step compute never re-touches the host.
    pub fn set_batch(&mut self, g: &mut Gpu, x: &[f32], target: &[f32]) -> Result<(), DriverError> {
        assert_eq!(x.len(), self.b * self.i);
        assert_eq!(target.len(), self.b * self.o);
        self.x = g.stream.memcpy_stod(x)?;
        self.target = g.stream.memcpy_stod(target)?;
        self.target_host = target.to_vec();
        Ok(())
    }

    /// Forward: `H_pre = X·W1ᵀ`, `H = relu(H_pre)`, `Y = H·W2ᵀ` — all resident.
    pub fn forward(&mut self, g: &mut Gpu) -> Result<(), DriverError> {
        let (b, i, h, o) = (self.b, self.i, self.h, self.o);
        gemm_dispatch(self.prec, g, false, true, &self.x, &self.w1, &mut self.h_pre, b, h, i)?;
        relu_fwd_device(g, &self.h_pre, &mut self.hact, b * h)?;
        gemm_dispatch(self.prec, g, false, true, &self.hact, &self.w2, &mut self.y, b, o, h)?;
        Ok(())
    }

    /// Recompute the hidden activations (`H_pre = X·W1ᵀ`, `H = relu(H_pre)`) from the current
    /// weights — the **gradient-checkpointing** path. Instead of stashing `H_pre`/`H` from the
    /// forward (the bulk of an MLP/FFN's activation memory), the backward regenerates them right
    /// before it needs them. Deterministic, so the gradients are bit-identical to the stashed path
    /// (gated). Only the small output `Y` (for the loss/`dY`) need stay resident. The attention
    /// backward is checkpointed the same way — it recomputes `P` rather than stashing the `S×S`
    /// matrix (see [`crate::ptx_autodiff_bwd::attention_backward`]).
    pub fn recompute_activations(&mut self, g: &mut Gpu) -> Result<(), DriverError> {
        let (b, i, h) = (self.b, self.i, self.h);
        gemm_dispatch(self.prec, g, false, true, &self.x, &self.w1, &mut self.h_pre, b, h, i)?;
        relu_fwd_device(g, &self.h_pre, &mut self.hact, b * h)?;
        Ok(())
    }

    /// Backward: seed `dY = scale*(Y - target)`, then `dW2 = dYᵀ·H`, `dH = dY·W2`,
    /// `dH_pre = dH ⊙ relu'(H_pre)`, `dW1 = dH_preᵀ·X` — all resident. `scale = 2` reproduces the
    /// `sum (y-t)^2` loss the autodiff tape differentiates.
    pub fn backward(&mut self, g: &mut Gpu, scale: f32) -> Result<(), DriverError> {
        let (b, i, h, o) = (self.b, self.i, self.h, self.o);
        mse_grad_device(g, &self.y, &self.target, &mut self.dy, b * o, scale)?;
        // dW2[O×H] = dYᵀ[O×B]·H[B×H]
        gemm_dispatch(self.prec, g, true, false, &self.dy, &self.hact, &mut self.dw2, o, h, b)?;
        // dH[B×H] = dY[B×O]·W2[O×H]
        gemm_dispatch(self.prec, g, false, false, &self.dy, &self.w2, &mut self.dh, b, h, o)?;
        // dH_pre = dH ⊙ relu'(H_pre)
        act_bwd_device(g, VM_RELU, &self.dh, &self.h_pre, &self.hact, &mut self.dh_pre, b * h)?;
        // dW1[H×I] = dH_preᵀ[H×B]·X[B×I]
        gemm_dispatch(self.prec, g, true, false, &self.dh_pre, &self.x, &mut self.dw1, h, i, b)?;
        Ok(())
    }

    /// Fused AdamW update of both layers (step index `t` drives the bias correction). The hp vector
    /// is the only host->device traffic (7 floats); the data buffers stay resident.
    pub fn optimize(&mut self, g: &mut Gpu, t: i32, cfg: AdamWCfg) -> Result<(), DriverError> {
        let mut hpv = vec![0.0f32; hp::LEN];
        hpv[hp::LR] = cfg.lr;
        hpv[hp::BETA1] = cfg.beta1;
        hpv[hp::BETA2] = cfg.beta2;
        hpv[hp::EPS] = cfg.eps;
        hpv[hp::WD] = cfg.wd;
        hpv[hp::BC1] = 1.0 - (cfg.beta1 as f64).powi(t) as f32;
        hpv[hp::BC2] = 1.0 - (cfg.beta2 as f64).powi(t) as f32;
        let hp_d = g.stream.memcpy_stod(&hpv)?;
        let (n1, n2) = (self.h * self.i, self.o * self.h);
        adamw_step_device(g, &mut self.w1, &self.dw1, &mut self.m1, &mut self.v1, &hp_d, n1)?;
        adamw_step_device(g, &mut self.w2, &self.dw2, &mut self.m2, &mut self.v2, &hp_d, n2)?;
        Ok(())
    }

    /// One full resident training step (forward + backward + optimizer). Returns nothing — call
    /// [`loss`](Self::loss) when the scalar is needed (it reads `Y` back; skip it in a throughput loop).
    pub fn step(&mut self, g: &mut Gpu, t: i32, cfg: AdamWCfg) -> Result<(), DriverError> {
        self.forward(g)?;
        self.backward(g, 2.0)?;
        self.optimize(g, t, cfg)?;
        Ok(())
    }

    /// One resident step (fwd + bwd + fused AdamW) with a **pre-uploaded** hp device buffer — pure
    /// kernel chain, no per-step host traffic. The throughput-measurement entry (the `bc` bias
    /// corrections are baked into `hp_d`; for a fixed-cadence timing loop they barely move).
    pub fn step_devhp(
        &mut self,
        g: &mut Gpu,
        hp_d: &CudaSlice<f32>,
    ) -> Result<(), DriverError> {
        self.forward(g)?;
        self.backward(g, 2.0)?;
        let (n1, n2) = (self.h * self.i, self.o * self.h);
        adamw_step_device(g, &mut self.w1, &self.dw1, &mut self.m1, &mut self.v1, hp_d, n1)?;
        adamw_step_device(g, &mut self.w2, &self.dw2, &mut self.m2, &mut self.v2, hp_d, n2)?;
        Ok(())
    }

    /// `sum (Y - target)^2` over the batch (reads `Y` back to the host — gate/diagnostic only).
    pub fn loss(&self, g: &Gpu) -> Result<f64, DriverError> {
        let yv: Vec<f32> = g.stream.memcpy_dtov(&self.y)?;
        Ok(yv
            .iter()
            .zip(&self.target_host)
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum())
    }

    /// Read the parameter gradients back to the host (`dW1`, `dW2`) — the gradient gate.
    pub fn grads_host(&self, g: &Gpu) -> Result<(Vec<f32>, Vec<f32>), DriverError> {
        Ok((g.stream.memcpy_dtov(&self.dw1)?, g.stream.memcpy_dtov(&self.dw2)?))
    }

    /// Read the current weights back to the host (`W1`, `W2`).
    pub fn weights_host(&self, g: &Gpu) -> Result<(Vec<f32>, Vec<f32>), DriverError> {
        Ok((g.stream.memcpy_dtov(&self.w1)?, g.stream.memcpy_dtov(&self.w2)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::{assert_close, Rng};
    use crate::gpu::gpu;

    fn with_gpu(name: &str, body: impl FnOnce(&mut Gpu)) {
        let mut guard = gpu();
        match guard.as_mut() {
            Some(g) => body(g),
            None => eprintln!("[skip] {name}: no CUDA device reachable"),
        }
    }

    /// f64 closed-form backprop reference for `loss = sum (y - t)^2` of the 2-layer relu MLP.
    /// Returns `(dW1, dW2)` to match the GPU trainer's gradient buffers.
    #[allow(clippy::too_many_arguments)]
    fn mlp_backprop_ref(
        b: usize,
        i: usize,
        h: usize,
        o: usize,
        x: &[f32],
        w1: &[f32],
        w2: &[f32],
        t: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut hpre = vec![0f64; b * h];
        let mut hh = vec![0f64; b * h];
        for bi in 0..b {
            for j in 0..h {
                let mut acc = 0f64;
                for ii in 0..i {
                    acc += x[bi * i + ii] as f64 * w1[j * i + ii] as f64;
                }
                hpre[bi * h + j] = acc;
                hh[bi * h + j] = acc.max(0.0);
            }
        }
        let mut y = vec![0f64; b * o];
        for bi in 0..b {
            for oo in 0..o {
                let mut acc = 0f64;
                for j in 0..h {
                    acc += hh[bi * h + j] * w2[oo * h + j] as f64;
                }
                y[bi * o + oo] = acc;
            }
        }
        let dy: Vec<f64> = (0..b * o).map(|k| 2.0 * (y[k] - t[k] as f64)).collect();
        // dW2[o][j] = sum_b dy[b][o] * h[b][j]
        let mut dw2 = vec![0f32; o * h];
        for oo in 0..o {
            for j in 0..h {
                let mut acc = 0f64;
                for bi in 0..b {
                    acc += dy[bi * o + oo] * hh[bi * h + j];
                }
                dw2[oo * h + j] = acc as f32;
            }
        }
        // dh[b][j] = sum_o dy[b][o] * w2[o][j]; dh_pre = dh * (hpre>0); dW1[j][i] = sum_b dh_pre[b][j]*x[b][i]
        let mut dw1 = vec![0f32; h * i];
        for j in 0..h {
            for ii in 0..i {
                let mut acc = 0f64;
                for bi in 0..b {
                    let mut dh = 0f64;
                    for oo in 0..o {
                        dh += dy[bi * o + oo] * w2[oo * h + j] as f64;
                    }
                    let dhpre = if hpre[bi * h + j] > 0.0 { dh } else { 0.0 };
                    acc += dhpre * x[bi * i + ii] as f64;
                }
                dw1[j * i + ii] = acc as f32;
            }
        }
        (dw1, dw2)
    }

    /// **The gradient gate:** the resident GPU forward+backward vs an f64 closed-form backprop of the
    /// same MLP/loss — over the full dW1, dW2 gradient tensors.
    #[test]
    fn resident_mlp_gradients_match_reference() {
        with_gpu("resident_mlp_gradients_match_reference", |g| {
            let (b, i, h, o) = (24usize, 18usize, 32usize, 12usize);
            let mut rng = Rng::new(0x9AD1);
            let w1 = rng.vec(h * i, -0.4, 0.4);
            let w2 = rng.vec(o * h, -0.4, 0.4);
            let x = rng.vec(b * i, -1.0, 1.0);
            let target = rng.vec(b * o, -1.0, 1.0);

            let mut tr = MlpTrainer::new(g, b, i, h, o, &w1, &w2).unwrap();
            tr.set_batch(g, &x, &target).unwrap();
            tr.forward(g).unwrap();
            tr.backward(g, 2.0).unwrap();
            let (dw1, dw2) = tr.grads_host(g).unwrap();

            let (rdw1, rdw2) = mlp_backprop_ref(b, i, h, o, &x, &w1, &w2, &target);
            assert_close("resident dW1", &dw1, &rdw1, 1e-3, 2e-3);
            assert_close("resident dW2", &dw2, &rdw2, 1e-3, 2e-3);
        });
    }

    /// **The fp16 gradient gate:** the resident step with `Precision::F16Mixed` (tensor-core GEMMs) vs
    /// the same f64 closed-form backprop, at an f16 tolerance — proving the mixed-precision routing
    /// produces correct gradients end-to-end through all five GEMMs (NT/NN/TN). Dims are 64-multiples
    /// so every GEMM rides the `_sm_db` tensor-core kernel (not the non-16 f32 fallback).
    #[test]
    fn resident_mlp_gradients_match_reference_f16() {
        with_gpu("resident_mlp_gradients_match_reference_f16", |g| {
            let (b, i, h, o) = (64usize, 64usize, 128usize, 64usize);
            let mut rng = Rng::new(0x16AD);
            let w1 = rng.vec(h * i, -0.4, 0.4);
            let w2 = rng.vec(o * h, -0.4, 0.4);
            let x = rng.vec(b * i, -1.0, 1.0);
            let target = rng.vec(b * o, -1.0, 1.0);

            let mut tr = MlpTrainer::new(g, b, i, h, o, &w1, &w2)
                .unwrap()
                .with_precision(Precision::F16Mixed);
            tr.set_batch(g, &x, &target).unwrap();
            tr.forward(g).unwrap();
            tr.backward(g, 2.0).unwrap();
            let (dw1, dw2) = tr.grads_host(g).unwrap();

            let (rdw1, rdw2) = mlp_backprop_ref(b, i, h, o, &x, &w1, &w2, &target);
            // f16 inputs to every GEMM (~5e-4 rel each) compound through the chain; dW1 rides 4 chained
            // f16 GEMMs and a relu whose kink amplifies a few near-threshold lanes (measured worst-case
            // ~4.3% rel). This is the genuine f16 precision cost — the GEMM kernel itself is bit-tight-
            // gated separately (`fp16_gemm_matches_f64_reference`, 3e-3 rel). The job of THIS gate is to
            // catch a miswiring of the 5 GEMMs (mode/operand/dim) — which is gross, orders past 8% rel.
            assert_close("resident-f16 dW1", &dw1, &rdw1, 2e-1, 8e-2);
            assert_close("resident-f16 dW2", &dw2, &rdw2, 2e-1, 8e-2);
        });
    }

    /// **The fp16 training gate:** the tensor-core (`F16Mixed`) resident step actually learns — several
    /// AdamW steps on a fixed teacher-generated batch drive the loss down. Confirms the mixed-precision
    /// gradients are not just close but *useful* (no silent sign/scale corruption that a one-shot
    /// gradient check at a single point could miss).
    #[test]
    fn resident_mlp_loss_decreases_f16() {
        with_gpu("resident_mlp_loss_decreases_f16", |g| {
            let (b, i, h, o) = (64usize, 64usize, 128usize, 64usize);
            let mut rng = Rng::new(0x16ED);
            let w1 = rng.vec(h * i, -0.1, 0.1);
            let w2 = rng.vec(o * h, -0.1, 0.1);
            let x = rng.vec(b * i, -1.0, 1.0);
            let tw1 = rng.vec(h * i, -0.6, 0.6);
            let tw2 = rng.vec(o * h, -0.6, 0.6);
            let target = {
                let (mut hpre, mut yv) = (vec![0f32; b * h], vec![0f32; b * o]);
                for bi in 0..b {
                    for j in 0..h {
                        let mut a = 0f32;
                        for ii in 0..i {
                            a += x[bi * i + ii] * tw1[j * i + ii];
                        }
                        hpre[bi * h + j] = a.max(0.0);
                    }
                    for oo in 0..o {
                        let mut a = 0f32;
                        for j in 0..h {
                            a += hpre[bi * h + j] * tw2[oo * h + j];
                        }
                        yv[bi * o + oo] = a;
                    }
                }
                yv
            };

            let mut tr = MlpTrainer::new(g, b, i, h, o, &w1, &w2)
                .unwrap()
                .with_precision(Precision::F16Mixed);
            tr.set_batch(g, &x, &target).unwrap();
            let cfg = AdamWCfg {
                lr: 5e-3,
                ..Default::default()
            };
            tr.forward(g).unwrap();
            let l0 = tr.loss(g).unwrap();
            let mut prev = l0;
            for t in 1..=200i32 {
                tr.step(g, t, cfg).unwrap();
                prev = tr.loss(g).unwrap();
            }
            eprintln!("resident MLP (f16): loss {l0:.4e} -> {prev:.4e} over 200 AdamW steps");
            // Looser than the f32 gate (0.2): f16 rounding floors how far the loss can fall.
            assert!(prev < l0 * 0.5, "f16 loss did not fall enough: {l0:.3e} -> {prev:.3e}");
        });
    }

    /// **The checkpointing gate:** gradients are **bit-identical** whether the hidden activations are
    /// stashed from the forward or recomputed in the backward. The recomputed path first poisons the
    /// stashed `H_pre`/`H` (only `Y` is kept), proving the backward truly reconstructs them.
    #[test]
    fn resident_mlp_checkpointing_bit_identical() {
        with_gpu("resident_mlp_checkpointing_bit_identical", |g| {
            let (b, i, h, o) = (24usize, 18usize, 32usize, 12usize);
            let mut rng = Rng::new(0xC4EC);
            let w1 = rng.vec(h * i, -0.4, 0.4);
            let w2 = rng.vec(o * h, -0.4, 0.4);
            let x = rng.vec(b * i, -1.0, 1.0);
            let target = rng.vec(b * o, -1.0, 1.0);

            let mut tr = MlpTrainer::new(g, b, i, h, o, &w1, &w2).unwrap();
            tr.set_batch(g, &x, &target).unwrap();
            // Stashed path.
            tr.forward(g).unwrap();
            tr.backward(g, 2.0).unwrap();
            let (a1, a2) = tr.grads_host(g).unwrap();
            // Checkpointed path: keep Y, poison the stashed hidden activations, recompute, backward.
            tr.h_pre = g.stream.memcpy_stod(&vec![1e30f32; b * h]).unwrap();
            tr.hact = g.stream.memcpy_stod(&vec![-7e29f32; b * h]).unwrap();
            tr.recompute_activations(g).unwrap();
            tr.backward(g, 2.0).unwrap();
            let (c1, c2) = tr.grads_host(g).unwrap();
            for (k, (s, r)) in a1.iter().zip(&c1).enumerate() {
                assert_eq!(s.to_bits(), r.to_bits(), "dW1[{k}] stashed vs checkpointed");
            }
            for (k, (s, r)) in a2.iter().zip(&c2).enumerate() {
                assert_eq!(s.to_bits(), r.to_bits(), "dW2[{k}] stashed vs checkpointed");
            }
        });
    }

    /// **The loss-decrease gate:** several resident AdamW steps on a fixed batch drive the loss down.
    #[test]
    fn resident_mlp_loss_decreases() {
        with_gpu("resident_mlp_loss_decreases", |g| {
            let (b, i, h, o) = (32usize, 16usize, 48usize, 8usize);
            let mut rng = Rng::new(0x5EED);
            // Small init weights -> the student starts well away from the (teacher-generated) target.
            let w1 = rng.vec(h * i, -0.1, 0.1);
            let w2 = rng.vec(o * h, -0.1, 0.1);
            let x = rng.vec(b * i, -1.0, 1.0);
            // A teacher MLP makes a learnable, non-trivial target.
            let tw1 = rng.vec(h * i, -0.6, 0.6);
            let tw2 = rng.vec(o * h, -0.6, 0.6);
            let target = {
                let (mut hpre, mut yv) = (vec![0f32; b * h], vec![0f32; b * o]);
                for bi in 0..b {
                    for j in 0..h {
                        let mut a = 0f32;
                        for ii in 0..i {
                            a += x[bi * i + ii] * tw1[j * i + ii];
                        }
                        hpre[bi * h + j] = a.max(0.0);
                    }
                    for oo in 0..o {
                        let mut a = 0f32;
                        for j in 0..h {
                            a += hpre[bi * h + j] * tw2[oo * h + j];
                        }
                        yv[bi * o + oo] = a;
                    }
                }
                yv
            };

            let mut tr = MlpTrainer::new(g, b, i, h, o, &w1, &w2).unwrap();
            tr.set_batch(g, &x, &target).unwrap();
            let cfg = AdamWCfg {
                lr: 5e-3,
                ..Default::default()
            };
            tr.forward(g).unwrap();
            let l0 = tr.loss(g).unwrap();
            let mut prev = l0;
            let mut worse = 0;
            for t in 1..=300i32 {
                tr.step(g, t, cfg).unwrap();
                let cur = tr.loss(g).unwrap();
                if cur > prev * 1.0001 {
                    worse += 1; // AdamW can blip; only flag sustained increases
                }
                prev = cur;
            }
            eprintln!("resident MLP: loss {l0:.4e} -> {prev:.4e} over 300 AdamW steps ({worse} up-blips)");
            assert!(prev < l0 * 0.2, "loss did not fall enough: {l0:.3e} -> {prev:.3e}");
            assert!(worse < 15, "too many loss increases ({worse}) — unstable");
        });
    }

    fn best_of(rounds: usize, mut f: impl FnMut() -> f64) -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..rounds {
            best = best.min(f());
        }
        best
    }

    /// Seconds/call for cuBLAS **f32** SGEMM (M×N×K): resident calls + warm-up + sync. The honest f32
    /// peer for Mercury's training GEMMs (transpose convention is irrelevant for a timing-only peer).
    fn time_cublas_sgemm_f32(g: &mut Gpu, m: usize, k: usize, n: usize, iters: u32) -> f64 {
        use cudarc::cublas::sys::cublasOperation_t;
        use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
        let blas = CudaBlas::new(g.stream.clone()).unwrap();
        let a_d = g.stream.memcpy_stod(&vec![0.01f32; m * k]).unwrap();
        let b_d = g.stream.memcpy_stod(&vec![0.01f32; n * k]).unwrap();
        let mut c_d = g.stream.memcpy_stod(&vec![0.0f32; m * n]).unwrap();
        let cfg = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0,
            lda: k as i32,
            ldb: k as i32,
            beta: 0.0,
            ldc: n as i32,
        };
        unsafe { blas.gemm(cfg, &b_d, &a_d, &mut c_d).unwrap() };
        g.stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            unsafe { blas.gemm(cfg, &b_d, &a_d, &mut c_d).unwrap() };
        }
        g.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// **M8 throughput** — the resident MLP/FFN training step (fwd+bwd+fused AdamW), GPU-resident, no
    /// host round-trip, at a GPT-2-FFN-ish shape. Absolute us/step and tokens/s swing with the laptop
    /// clock (~7×); the clock-invariant figure is the cuBLAS-GEMM ratio in `m8_gemm_vs_cublas`.
    #[test]
    #[ignore]
    fn m8_resident_step_throughput() {
        with_gpu("m8_resident_step_throughput", |g| {
            let (b, i, h, o) = (512usize, 768usize, 3072usize, 768usize); // GPT-2 FFN-ish
            let mut rng = Rng::new(0x6FFA);
            let w1 = rng.vec(h * i, -0.02, 0.02);
            let w2 = rng.vec(o * h, -0.02, 0.02);
            let x = rng.vec(b * i, -1.0, 1.0);
            let target = rng.vec(b * o, -1.0, 1.0);
            let mut tr = MlpTrainer::new(g, b, i, h, o, &w1, &w2).unwrap();
            tr.set_batch(g, &x, &target).unwrap();
            let mut hpv = vec![0.0f32; hp::LEN];
            hpv[hp::LR] = 1e-3;
            hpv[hp::BETA1] = 0.9;
            hpv[hp::BETA2] = 0.999;
            hpv[hp::EPS] = 1e-8;
            hpv[hp::BC1] = 1.0 - 0.9f32.powi(10);
            hpv[hp::BC2] = 1.0 - 0.999f32.powi(10);
            let hp_d = g.stream.memcpy_stod(&hpv).unwrap();
            for _ in 0..3 {
                tr.step_devhp(g, &hp_d).unwrap();
            }
            g.stream.synchronize().unwrap();
            let iters = 30u32;
            let secs = best_of(5, || {
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    tr.step_devhp(g, &hp_d).unwrap();
                }
                g.stream.synchronize().unwrap();
                t0.elapsed().as_secs_f64() / iters as f64
            });
            eprintln!(
                "M8 resident step @B={b} d={i} ff={h}: {:.0} us/step, {:.0} tokens/s (clock-variant)",
                secs * 1e6,
                b as f64 / secs
            );
        });
    }

    /// **M8 honesty frontier** — Mercury's training GEMM vs cuBLAS f32 SGEMM, same-run, at the FFN's
    /// dominant shapes. The clock-invariant % of cuBLAS says how far the (currently naive) training
    /// GEMM is from the library — the binding constraint for beating an eager cuBLAS chain. Reported,
    /// not asserted: it is a measurement, and the lever is kernel work (tiling / tensor cores).
    #[test]
    #[ignore]
    fn m8_gemm_vs_cublas() {
        with_gpu("m8_gemm_vs_cublas", |g| {
            for &(m, k, n) in &[(512usize, 768usize, 3072usize), (512, 3072, 768)] {
                let mut rng = Rng::new(0xA11 ^ (m * n * k) as u64);
                let a = rng.vec(m * k, -1.0, 1.0);
                let bb = rng.vec(k * n, -1.0, 1.0);
                let a_d = g.stream.memcpy_stod(&a).unwrap();
                let b_d = g.stream.memcpy_stod(&bb).unwrap();
                let mut c_d = g.stream.alloc_zeros::<f32>(m * n).unwrap();
                gemm_device(g, false, false, &a_d, &b_d, &mut c_d, m, n, k).unwrap();
                g.stream.synchronize().unwrap();
                let iters = 20u32;
                let mer = best_of(5, || {
                    let t0 = std::time::Instant::now();
                    for _ in 0..iters {
                        gemm_device(g, false, false, &a_d, &b_d, &mut c_d, m, n, k).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                let cub = best_of(5, || time_cublas_sgemm_f32(g, m, k, n, iters));
                let flop = 2.0 * m as f64 * n as f64 * k as f64;
                let pct = 100.0 * cub / mer;
                eprintln!(
                    "GEMM {m}x{n}x{k}: Mercury {:.1} GFLOP/s vs cuBLAS-f32 {:.1} GFLOP/s = {pct:.1}% of cuBLAS",
                    flop / mer / 1e9,
                    flop / cub / 1e9,
                );
                // Regression floor (clock-invariant ratio): the reg-blocked routing measures ~32-35%;
                // a revert to the naive kernel drops to ~7-10%. 20% catches that without flakiness.
                assert!(pct >= 20.0, "training GEMM regressed to {pct:.1}% of cuBLAS (expected >=20%)");
            }
        });
    }

    /// **M8 fp16 GEMM standing** — the tensor-core training GEMM (`gemm_device_f16`, NT, including the
    /// just-in-time f16 narrow the resident step pays) vs both the f32 reg-blocked kernel and the
    /// gold-standard cuBLAS **fp16** peer, same-run, at the FFN's dominant shapes. The clock-invariant
    /// figures are the **f16/f32 speedup** (what the tensor cores bought) and the **% of cuBLAS-fp16**
    /// (how far the kernel is from the library on the SAME precision). cuBLAS-fp16 is the honest peer
    /// now that Mercury also uses the tensor cores. Reported, not asserted — it's a measurement.
    #[test]
    #[ignore]
    fn m8_gemm_f16_vs_cublas_f16() {
        with_gpu("m8_gemm_f16_vs_cublas_f16", |g| {
            for &(m, k, n) in &[(512usize, 768usize, 3072usize), (512, 3072, 768)] {
                let mut rng = Rng::new(0xF16B ^ (m * n * k) as u64);
                let a = rng.vec(m * k, -1.0, 1.0);
                let bb = rng.vec(n * k, -1.0, 1.0); // NT: B is n×k
                let a_d = g.stream.memcpy_stod(&a).unwrap();
                let b_d = g.stream.memcpy_stod(&bb).unwrap();
                let mut c_d = g.stream.alloc_zeros::<f32>(m * n).unwrap();
                // warm up both kernels
                gemm_device(g, false, true, &a_d, &b_d, &mut c_d, m, n, k).unwrap();
                gemm_device_f16(g, false, true, &a_d, &b_d, &mut c_d, m, n, k).unwrap();
                g.stream.synchronize().unwrap();
                let iters = 20u32;
                let mer32 = best_of(5, || {
                    let t0 = std::time::Instant::now();
                    for _ in 0..iters {
                        gemm_device(g, false, true, &a_d, &b_d, &mut c_d, m, n, k).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                let mer16 = best_of(5, || {
                    let t0 = std::time::Instant::now();
                    for _ in 0..iters {
                        gemm_device_f16(g, false, true, &a_d, &b_d, &mut c_d, m, n, k).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                let cub16 = best_of(5, || {
                    crate::baselines::time_cublas_gemm_nt_f16(g, m, k, n, iters).unwrap()
                });
                let flop = 2.0 * m as f64 * n as f64 * k as f64;
                eprintln!(
                    "GEMM {m}x{n}x{k}: f32 {:.1} | f16 {:.1} | cuBLAS-f16 {:.1} GFLOP/s  \
                     => f16 is {:.2}x f32, {:.1}% of cuBLAS-f16",
                    flop / mer32 / 1e9,
                    flop / mer16 / 1e9,
                    flop / cub16 / 1e9,
                    mer32 / mer16,
                    100.0 * cub16 / mer16,
                );
            }
        });
    }

    /// **M8 resident-step f16 vs f32** — the whole resident MLP/FFN training step (fwd+bwd+fused AdamW)
    /// timed in both precisions same-run at the GPT-2-FFN shape. The clock-invariant figure is the
    /// **f16/f32 step speedup** (absolute us/step and tokens/s swing ~7× with the laptop clock). This
    /// is the honest M8 perf delta the tensor-core routing buys end-to-end (the GEMMs dominate, so the
    /// step speedup tracks the GEMM speedup minus the cast/transpose overhead the f16 path adds).
    #[test]
    #[ignore]
    fn m8_resident_step_f16_vs_f32() {
        with_gpu("m8_resident_step_f16_vs_f32", |g| {
            let (b, i, h, o) = (512usize, 768usize, 3072usize, 768usize);
            let mut rng = Rng::new(0x6FFB);
            let w1 = rng.vec(h * i, -0.02, 0.02);
            let w2 = rng.vec(o * h, -0.02, 0.02);
            let x = rng.vec(b * i, -1.0, 1.0);
            let target = rng.vec(b * o, -1.0, 1.0);
            let mut hpv = vec![0.0f32; hp::LEN];
            hpv[hp::LR] = 1e-3;
            hpv[hp::BETA1] = 0.9;
            hpv[hp::BETA2] = 0.999;
            hpv[hp::EPS] = 1e-8;
            hpv[hp::BC1] = 1.0 - 0.9f32.powi(10);
            hpv[hp::BC2] = 1.0 - 0.999f32.powi(10);
            let hp_d = g.stream.memcpy_stod(&hpv).unwrap();

            let time_step = |g: &mut Gpu, prec: Precision| -> f64 {
                let mut tr = MlpTrainer::new(g, b, i, h, o, &w1, &w2)
                    .unwrap()
                    .with_precision(prec);
                tr.set_batch(g, &x, &target).unwrap();
                for _ in 0..3 {
                    tr.step_devhp(g, &hp_d).unwrap();
                }
                g.stream.synchronize().unwrap();
                let iters = 30u32;
                best_of(5, || {
                    let t0 = std::time::Instant::now();
                    for _ in 0..iters {
                        tr.step_devhp(g, &hp_d).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                })
            };
            let s32 = time_step(g, Precision::F32);
            let s16 = time_step(g, Precision::F16Mixed);
            eprintln!(
                "M8 resident step @B={b} d={i} ff={h}: f32 {:.0} us ({:.0} tok/s) | f16 {:.0} us \
                 ({:.0} tok/s) => f16 is {:.2}x f32 (clock-variant)",
                s32 * 1e6,
                b as f64 / s32,
                s16 * 1e6,
                b as f64 / s16,
                s32 / s16,
            );
        });
    }
}
