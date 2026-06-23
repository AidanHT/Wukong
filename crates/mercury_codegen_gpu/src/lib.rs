//! `mercury_codegen_gpu` — Mercury's GPU backend.
//!
//! Mercury is a compiler, so its GPU path **emits PTX text** and **driver-JIT-loads it via
//! `cudarc`** (`cuModuleLoadData` — the NVIDIA driver's built-in PTX→SASS JIT, which needs no
//! `nvcc`/`ptxas`/`nvrtc`), launches kernels over device buffers, and copies results back. This is
//! the GPU analogue of the CPU seam: a recognizer picks a kernel symbol, this crate provides the
//! kernel (as PTX) + a host launcher, and the interpreter remains the (tolerance) oracle.
//!
//! Everything is behind the `gpu` feature so the toolchain-free CPU core stays untouched: a plain
//! `cargo test` compiles an essentially empty crate; `cargo test -p mercury_codegen_gpu --features
//! gpu` builds the `cudarc` path and runs kernels on the local device.

/// Whether this build has the GPU backend compiled in (`--features gpu`).
pub const GPU_ENABLED: bool = cfg!(feature = "gpu");

#[cfg(feature = "gpu")]
pub mod baselines;

#[cfg(feature = "gpu")]
pub mod cubin;

#[cfg(feature = "gpu")]
pub mod diff;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub mod ptx;

#[cfg(feature = "gpu")]
pub mod ptx_gemm;

#[cfg(feature = "gpu")]
pub mod ptx_wmma;

#[cfg(feature = "gpu")]
pub mod ptx_norm;

#[cfg(feature = "gpu")]
pub mod ptx_flash;

#[cfg(feature = "gpu")]
pub mod ptx_conv;

#[cfg(feature = "gpu")]
pub mod ptx_fp8;

#[cfg(feature = "gpu")]
pub mod ptx_int8;

// --- Session I (GPU-resident training / M8): owned, append-only ---------------------------------
/// Fused optimizer-step GPU kernels (AdamW / SGD) — the device twin of `mercury_autodiff::optim`.
#[cfg(feature = "gpu")]
pub mod ptx_optim;

/// Backward GPU kernels for the autodiff tape's synthesized loops (transpose / activation- and
/// norm-backward) and the flash-attention backward (dQ/dK/dV).
#[cfg(feature = "gpu")]
pub mod ptx_autodiff_bwd;

/// GPU-resident MLP training step (forward + backward + fused AdamW, no host round-trip) — the M8
/// vehicle benched against PyTorch eager.
#[cfg(feature = "gpu")]
pub mod train_resident;

#[cfg(feature = "gpu")]
pub use gpu::{available, gpu, Gpu};
