//! `wukong_codegen_gpu` — Wukong's GPU backend.
//!
//! Wukong is a compiler, so its GPU path **emits PTX text** and **driver-JIT-loads it via
//! `cudarc`** (`cuModuleLoadData` — the NVIDIA driver's built-in PTX→SASS JIT, which needs no
//! `nvcc`/`ptxas`/`nvrtc`), launches kernels over device buffers, and copies results back. This is
//! the GPU analogue of the CPU seam: a recognizer picks a kernel symbol, this crate provides the
//! kernel (as PTX) + a host launcher, and the interpreter remains the (tolerance) oracle.
//!
//! Every module that touches `cudarc` is behind the `gpu` feature so the toolchain-free CPU core stays
//! untouched: a plain `cargo test` compiles only the two device-free modules declared below
//! (`paged_kv`, `paged_attention` — host policy, PTX text generators and f64 references, plus their
//! shape/ASCII gates); `cargo test -p wukong_codegen_gpu --features gpu` builds the `cudarc` path and
//! runs kernels on the local device. **A plain `cargo test` therefore does not type-check most of this
//! crate** — only a `--features gpu` build does, and `cargo check --features gpu --all-targets` is the
//! cheap way to get that coverage without a device.

/// Whether this build has the GPU backend compiled in (`--features gpu`).
pub const GPU_ENABLED: bool = cfg!(feature = "gpu");

/// The single source of PTX module headers (`.version`/`.target` floors per family) and of the
/// device-arch flag spellings (`compute_XX`/`sm_XX`). Un-gated on purpose: pure strings with no
/// `cudarc` dependency, consumed by the un-gated `paged_attention` generators as well as every
/// gated family, and its gates run in a plain `cargo test`.
pub mod ptx_target;

/// The **benchmark instrument** (GPU_RETARGET_PLAN.md §6.1–§6.2): the twin control, the publish
/// gate and the harness-emitted provenance header. Un-gated for the same reason `ptx_target` is —
/// the statistics, the rotation driver, the device spec table, the `nvidia-smi` parser, the
/// provenance formatting and the twin's PTX text are pure functions with no `cudarc` dependency, so
/// they are unit-tested in a plain, device-free `cargo test`. Only the launch layer (`facts_of`,
/// `open_round`, `TwinBuffers`, `PtxTwin`'s timing methods, `machine_floor`) is `gpu`-gated within.
pub mod bench_instrument;

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

// Winograd convolution F(2×2,3×3) / F(4×4,3×3): the f64 CPU reference (gate oracle) plus the GPU
// transform + batched-GEMM PTX generators. The 2.25–4× multiply-reduction lever for 3×3 stride-1.
#[cfg(feature = "gpu")]
pub mod ptx_winograd;

#[cfg(feature = "gpu")]
pub mod ptx_fp8;

// M7 runtime (Phase 7): device memory pool + CUDA-graph capture/replay. New, single-owner files;
// they wrap the existing launchers without touching a kernel.
#[cfg(feature = "gpu")]
pub mod pool;

/// General MIR->PTX lowering (Phase 4): consumes MIR and emits PTX, so *arbitrary* Wukong programs
/// run on the GPU — recognized runtime symbols lower to per-op `mrt_*` device helpers, everything else
/// lowers generally, and anything unhandled declines with `lower::UNSUPPORTED`. Additive to the
/// existing recognizer-offload path (`--backend=gpu`); this is the separate `--backend=gpu-native`
/// path, which the driver drives through `lower::jit_run` (the `Backend` impl is the declared seam,
/// not the call path).
#[cfg(feature = "gpu")]
pub mod lower;

/// GPU op-graph fusion planner + megakernel **eligibility analysis** (Phase 3 general / Phase 8):
/// pure MIR analysis (no device) deciding whether an entry function can be compiled into the
/// cooperative single-block megakernel, and inventorying its recognized cooperative ops.
#[cfg(feature = "gpu")]
pub mod fusion;

/// Whole-program **cooperative megakernel** (Phase 8 / M13): compile an eligible Wukong program into
/// one persistent `.visible .entry` kernel run by a block of threads, recognized ops executed
/// cooperatively across the block (no per-op launches, activations resident in one shared frame).
#[cfg(feature = "gpu")]
pub mod megakernel;

#[cfg(feature = "gpu")]
pub mod ptx_int4;

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
/// Fused optimizer-step GPU kernels (AdamW / SGD) — the device twin of `wukong_autodiff::optim`.
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

// --- End-to-end serving (perf/gpu-serving): paged KV-cache + continuous batching + decode graph ---
// Owned, append-only. The host block allocator (`paged_kv::BlockManager`) is pure policy with no device
// dependency, so this module is declared *un-gated*: it compiles and unit-tests in a plain (no-`gpu`)
// `cargo test`. Only the device cache (`PagedKvCache`) and the decode kernels are `#[cfg(feature =
// "gpu")]` within. `paged_attention` below is un-gated for the same reason; `serving` (the GPU
// launchers / decode loop) is `gpu`-gated whole.
pub mod paged_kv;

/// Paged decode-attention kernel: single-query attention against the paged KV-cache, gathering K/V
/// through the per-sequence block table (PagedAttention). The four PTX generators, the int8 quantizer
/// and the f64 reference are pure host code with no device dependency, so — like its `paged_kv`
/// sibling above — this module is declared **un-gated**: the PTX shape and **ASCII** gates run in a
/// plain, toolchain-free `cargo test`. Only the launchers and the device gates are
/// `#[cfg(feature = "gpu")]` within.
pub mod paged_attention;

/// Batched autoregressive decode layer/model over the paged KV-cache (`DecodeLayer`/`DecodeModel`) —
/// the serving forward pass: projections + append-to-cache + paged attention + FFN, whole-stack
/// GPU-resident, pooled/on-stream so a CUDA graph captures the whole decode step — plus the
/// continuous-batching `Scheduler` (`Request` admission, eviction, `step_graphed`) that drives it.
#[cfg(feature = "gpu")]
pub mod serving;
