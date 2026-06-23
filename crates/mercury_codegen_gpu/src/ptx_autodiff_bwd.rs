//! Backward (gradient) GPU kernels for the autodiff tape — the device twin of the synthesized
//! counted loops `mercury_autodiff::tape` emits on the CPU.
//!
//! The tape's forward VJPs already ride the tuned `mercury_*` kernels (matmul gradients reuse the
//! GEMM, reductions reuse `sreduce`, …). What it *synthesizes* as plain counted loops — buffer
//! transpose (for `dB = dCᵀ·A`), the elementwise activation-backward `dx = dout ⊙ f'(x)`, and the
//! per-row norm-backward combines — are CPU device-loops today. This module provides their GPU
//! kernels so the whole backward stays device-resident, plus the **flash-attention backward**
//! (dQ/dK/dV) gated against a two-pass-softmax f64 reference.
//!
//! PTX is pure ASCII; target `sm_89`. Kernels are grid-stride / one-CTA-per-row so correctness is
//! independent of the launch grid. (Populated across increments 2 and 4.)
