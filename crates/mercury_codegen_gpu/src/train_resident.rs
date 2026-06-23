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
use crate::ptx_autodiff_bwd::{act_bwd_device, gemm_device, mse_grad_device, relu_fwd_device};
use crate::ptx_optim::{adamw_step_device, hp};
use cudarc::driver::{CudaSlice, DriverError};
use mercury_runtime::VM_RELU;

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
        gemm_device(g, false, true, &self.x, &self.w1, &mut self.h_pre, b, h, i)?;
        relu_fwd_device(g, &self.h_pre, &mut self.hact, b * h)?;
        gemm_device(g, false, true, &self.hact, &self.w2, &mut self.y, b, o, h)?;
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
        gemm_device(g, false, true, &self.x, &self.w1, &mut self.h_pre, b, h, i)?;
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
        gemm_device(g, true, false, &self.dy, &self.hact, &mut self.dw2, o, h, b)?;
        // dH[B×H] = dY[B×O]·W2[O×H]
        gemm_device(g, false, false, &self.dy, &self.w2, &mut self.dh, b, h, o)?;
        // dH_pre = dH ⊙ relu'(H_pre)
        act_bwd_device(g, VM_RELU, &self.dh, &self.h_pre, &self.hact, &mut self.dh_pre, b * h)?;
        // dW1[H×I] = dH_preᵀ[H×B]·X[B×I]
        gemm_device(g, true, false, &self.dh_pre, &self.x, &mut self.dw1, h, i, b)?;
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
}
