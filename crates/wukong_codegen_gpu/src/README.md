# `wukong_codegen_gpu` modules

Module table from [`lib.rs`](lib.rs). **Ungated** modules compile in a plain `cargo test`; **`gpu`**
modules need `--features gpu`. This is a map, not a tour of `gpu.rs`.

| Module | Gate | One line |
|--------|------|----------|
| [`ptx_target`](ptx_target.rs) | — | Single source of PTX module headers (`.version`/`.target` floors) and `compute_XX`/`sm_XX` spellings. |
| [`bench_instrument`](bench_instrument.rs) | partial | Benchmark twin control, publish gate, provenance header, and device-free statistics; launch layer is `gpu`-gated. |
| [`tma_host`](tma_host.rs) | partial | Host half of TMA: 128-byte `CUtensorMap` args and preconditions; `TensorMap::encode` is `gpu`-gated. |
| [`ptx_wgmma`](ptx_wgmma.rs) | partial | Hopper `wgmma` + TMA GEMM PTX generator (`.target sm_90a`); `require_sm90a` is `gpu`-gated. |
| [`ptx_act`](ptx_act.rs) | — | Shared fused-activation PTX text (`relu`/`silu`/`gelu`) used by every epilogue that fuses one. |
| [`paged_kv`](paged_kv.rs) | — | Host block allocator and GQA cache geometry for paged KV — pure policy, no device. |
| [`paged_attention`](paged_attention.rs) | partial | Paged decode-attention PTX generators, int8 quantizer, and f64 reference; launchers are `gpu`-gated. |
| [`baselines`](baselines.rs) | gpu | cuBLAS / cuBLASLt / cuDNN / NVRTC peer baselines for benchmark comparison. |
| [`cubin`](cubin.rs) | gpu | On-disk cubin cache keyed by device. |
| [`diff`](diff.rs) | gpu | CPU↔GPU tolerance harness against the interpreter / runtime oracle. |
| [`gpu`](gpu.rs) | gpu | Device context, module load, launch wrappers, and the kernel zoo entry points benches call. |
| [`ptx`](ptx.rs) | gpu | Shared PTX emission helpers and register naming conventions. |
| [`ptx_gemm`](ptx_gemm.rs) | gpu | Hand-tuned GEMM PTX families (scalar and tiled). |
| [`ptx_wmma`](ptx_wmma.rs) | gpu | WMMA/tensor-core GEMM PTX generators (Ampere-class). |
| [`ptx_norm`](ptx_norm.rs) | gpu | Layer/RMS norm PTX kernels. |
| [`ptx_flash`](ptx_flash.rs) | gpu | Flash-attention forward PTX; bench/gate surface, not reachable from `.wk` today. |
| [`ptx_conv`](ptx_conv.rs) | gpu | Convolution PTX (direct and WMMA paths); bench/gate surface, not reachable from `.wk` today. |
| [`ptx_winograd`](ptx_winograd.rs) | gpu | Winograd F(2×2,3×3) / F(4×4,3×3) transform + batched-GEMM PTX. |
| [`ptx_fp8`](ptx_fp8.rs) | gpu | E4M3 forward fp8 GEMM PTX (Ada-floor families). |
| [`pool`](pool.rs) | gpu | Device memory pool and CUDA-graph capture/replay wrappers. |
| [`lower`](lower.rs) | gpu | General MIR→PTX lowering for `--backend=gpu-native`; `mrt_*` helpers, `UNSUPPORTED` = SKIP. |
| [`fusion`](fusion.rs) | gpu | Megakernel eligibility analysis: which cooperative ops fuse into one block. |
| [`megakernel`](megakernel.rs) | gpu | Whole-program cooperative megakernel: one persistent `.visible .entry` kernel. |
| [`ptx_int4`](ptx_int4.rs) | gpu | Int4-quantized GEMM PTX. |
| [`graph`](graph.rs) | gpu | CUDA graph construction helpers for repeated launch sequences. |
| [`ptx_int8`](ptx_int8.rs) | gpu | Int8 GEMM PTX with autotuned tile variants. |
| [`ptx_fp8_train`](ptx_fp8_train.rs) | gpu | fp8 training kernels: E5M2 backward GEMM, amax, delayed scaling. |
| [`autotune`](autotune.rs) | gpu | Per-(op, shape, dtype) autotune cache and regression mode for int8 GEMM variants. |
| [`ptx_optim`](ptx_optim.rs) | gpu | Fused AdamW / SGD optimizer-step device kernels. |
| [`ptx_autodiff_bwd`](ptx_autodiff_bwd.rs) | gpu | Backward kernels for autodiff tape loops and flash-attention backward. |
| [`train_resident`](train_resident.rs) | gpu | GPU-resident MLP training step (forward + backward + fused AdamW); bench vehicle, not `.wk` surface. |
| [`serving`](serving.rs) | gpu | Batched autoregressive decode over paged KV + continuous-batching scheduler; not reachable from `.wk` today. |

**Re-exports** (all `gpu`-gated): `Gpu`, `gpu()`, `available()`, `lower_jit_run`, `GpuLowerBackend`.
