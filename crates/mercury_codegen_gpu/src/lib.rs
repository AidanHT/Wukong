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

// M7 runtime (Phase 7): device memory pool + CUDA-graph capture/replay. New, single-owner files;
// they wrap the existing launchers without touching a kernel.
#[cfg(feature = "gpu")]
pub mod pool;

/// General MIR->PTX lowering (Phase 4): a real `Backend` that consumes MIR and emits PTX, so
/// *arbitrary* Mercury programs run on the GPU — recognized ops still dispatch to the tuned kernels
/// as a fast path, everything else lowers generally. Additive to the existing recognizer-offload
/// path (`--backend=gpu`); this is the separate `--backend=gpu-native` path.
#[cfg(feature = "gpu")]
pub mod lower;

/// GPU op-graph fusion planner + megakernel **eligibility analysis** (Phase 3 general / Phase 8):
/// pure MIR analysis (no device) deciding whether an entry function can be compiled into the
/// cooperative single-block megakernel, and inventorying its recognized cooperative ops.
#[cfg(feature = "gpu")]
pub mod fusion;

/// Whole-program **cooperative megakernel** (Phase 8 / M13): compile an eligible Mercury program into
/// one persistent `.visible .entry` kernel run by a block of threads, recognized ops executed
/// cooperatively across the block (no per-op launches, activations resident in one shared frame).
#[cfg(feature = "gpu")]
pub mod megakernel;

#[cfg(feature = "gpu")]
pub mod ptx_int4;

// M7 runtime (Phase 7): device memory pool + CUDA-graph capture/replay. New, single-owner files;
// they wrap the existing launchers without touching a kernel.
#[cfg(feature = "gpu")]
pub mod pool;

#[cfg(feature = "gpu")]
pub mod graph;

#[cfg(feature = "gpu")]
pub mod ptx_int8;

// fp8 *training* kernels (Phase 6): E5M2 backward GEMM + amax + delayed scaling. Single-owner file;
// reuses the proven E4M3 m16n8k32 fragment layout without touching the forward path.
#[cfg(feature = "gpu")]
pub mod ptx_fp8_train;

// Phase 10: per-(op, shape, dtype) autotuning + on-disk config cache + regression mode. Picks the
// fastest of the int8 GEMM kernel variants (swz / hand-placed × tiles × split-K) per shape, same-run.
#[cfg(feature = "gpu")]
pub mod autotune;
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

#[cfg(feature = "gpu")]
pub use lower::{jit_run as lower_jit_run, GpuLowerBackend};
