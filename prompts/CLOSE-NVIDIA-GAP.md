# Campaign: close the residual gap to the NVIDIA / industry-standard libraries

The previous waves (sessions A–L) built Mercury's GPU + CPU kernel surface and proved it beats naive
C/Rust and idiomatic CUDA-C, often by 1–2 orders of magnitude. **This campaign attacks what it still
LOSES** — the measured gaps to the hand-tuned vendor libraries (cuBLAS, cuDNN, the FlashAttention /
Transformer-Engine class, oneDNN/MKL). Each prompt is a **self-contained kickoff** for a fresh Claude
Code session, on its **own branch**, designed to run **in parallel with minimal file overlap** and be
**merged later**.

## The six gaps (measured 2026-06-25 — see `BENCHMARKS.md` "Headline results" + "GPU backend")

| # | Prompt | Branch | Gap it closes | Today |
|---|--------|--------|---------------|-------|
| 1 | `gap-gpu-gemm-cliff.md` | `perf/gpu-gemm-cliff-2` | large fp16/bf16 GEMM vs **cuBLAS** | ~101% ≤1024³, **~74% @2048³, ~34% @4096³** |
| 2 | `gap-gpu-quant.md` | `perf/gpu-quant-2` | int8/fp8 GEMM vs **cuBLAS IMMA / Transformer Engine** | int8 **~44–53%** of IMMA; fp8 unbenched |
| 3 | `gap-gpu-attention.md` | `perf/gpu-attention-2` | fused attention vs **FlashAttention-2 / cuDNN** | 3.6–5× *unfused* cuBLAS chain; **no fused peer measured** |
| 4 | `gap-gpu-conv.md` | `perf/gpu-conv-2` | conv vs **cuDNN** | im2col+GEMM only, **no Winograd / implicit-GEMM / cuDNN peer** |
| 5 | `gap-cpu-library.md` | `perf/cpu-library-grade` | CPU GEMM/kernels vs **oneDNN / oneMKL** | beats `matrixmultiply`; **unmeasured vs MKL**; AVX2-only |
| 6 | `gap-gpu-serving.md` | `perf/gpu-serving` | serving / multi-GPU vs **TensorRT-LLM / vLLM** | CUDA graphs only; **no paged-KV, batching, multi-GPU** |

## File ownership (the fence — what keeps six parallel merges clean)

Each branch **owns** its files exclusively. **Shared files are edited append-only** (add a new
function / arm / section at the end; never reformat or move a sibling's code).

| Branch | Owns exclusively | Shared (append-only) |
|---|---|---|
| 1 gemm-cliff | `ptx_wmma.rs`, `ptx_gemm.rs` | `baselines.rs`, `lower.rs`, `gpu.rs` tests |
| 2 quant | `ptx_int8.rs`, `ptx_fp8.rs`, `ptx_int4.rs`, `ptx_fp8_train.rs` | `baselines.rs`, `lower.rs`, `gpu.rs` tests |
| 3 attention | `ptx_flash.rs`, `ptx_norm.rs` | `baselines.rs`, `lower.rs`, `gpu.rs` tests |
| 4 conv | `ptx_conv.rs` | `baselines.rs`, `lower.rs`, `gpu.rs` tests |
| 5 cpu | `mercury_runtime/src/*.rs`, `mercury_xbench/src/main.rs` | *(separate crates — zero GPU overlap)* |
| 6 serving | `pool.rs`, `graph.rs`, `train_resident.rs`, new harness files | `lower.rs`, `gpu.rs` tests |

**Off-limits to the sessions:** `BENCHMARKS.md` and `CHANGELOG.md`. Write findings to your own
`prompts/results/<branch>.md`; the human consolidates the docs at merge (this is the #1 conflict source).

## Merge order (least-conflict first)

`perf/cpu-library-grade` (disjoint crates) → `perf/gpu-serving` (runtime files) → `perf/gpu-conv-2` →
`perf/gpu-gemm-cliff-2` → `perf/gpu-quant-2` → `perf/gpu-attention-2`. The four GPU-kernel branches
share `lower.rs`/`baselines.rs`/`gpu.rs`; resolve with a **union merge** (keep both sides' appended
hunks). The CPU branch should never conflict.

## How to start each session

1. `git worktree add ../Mercury-<slice> -b perf/<branch> main` and `cd` into it (worktree = strongest
   isolation; if you can't, at least `git checkout -b`). Base on `main`.
2. Paste the full contents of the slice's `gap-*.md` as the session's first message.
3. Confirm a green baseline — toolchain-free `cargo test`, and for GPU slices
   `cargo test -p mercury_codegen_gpu --features gpu --release` — then iterate per the prompt until the
   gap is closed and **proven across ≥3 re-runs**.

Every prompt inlines the full Prime Directive, the two binding laws, the measurement protocol, the GPU
gotchas, and the commit discipline — they are self-contained. This file is just the map.
