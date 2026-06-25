# Close the quantized-GEMM gap vs cuBLAS IMMA / Transformer Engine (int8 + fp8)

You are a Claude Code session on **Mercury** (`mercuryc` emits PTX, driver-JIT via `cudarc`, no toolkit).
Target: **mobile RTX 4050, sm_89 (Ada), ~30–50 W**. First action:
`git worktree add ../Mercury-quant2 -b perf/gpu-quant-2 main` and work inside it (or `git checkout -b`).

## Prime Directive (how you work)

Continuously iterate through the entire codebase until Mercury performs better than C, C++, and Rust
**and closes — then beats — the NVIDIA gold-standard library on quantized GEMM**. **No scope
restrictions.** If something blocks a higher metric, **remove that blocker** and explore options **even
outside scope**. Improve everything. **No limit.** **Assume the metrics are NOT strong until you PROVE
they are** across multiple same-run iterations. **Do not stop** until the benchmarks heavily exceed
C/C++/Rust and close the cuBLAS-IMMA / Transformer-Engine gap. Think very carefully. If you CAN improve
it, **do it.** Time doesn't matter. **Run many sub-agents in parallel** for research and experiments.
**Make 10–20 commits, NO co-authored lines.** "Good enough" is failure: ~44% of cuBLAS IMMA is a **56%
deficiency**.

## Your mission

Mercury's int8 (W8A8, `u8×i8→i32`) tensor-core GEMM **crushes** naive/dp4a CUDA-C (~180–237× / ~34–58×)
but runs at only **~44% (1024³) / ~53% (2048³) of cuBLAS int8 IMMA** — a real ~2× gap. fp8 (E4M3/E5M2)
GEMM exists but is **unbenched vs the Transformer Engine class**. Targets: int8 ≥75% of cuBLAS IMMA;
fp8 measured honestly vs the best fp8 peer you can build/`dlopen`, and the fused per-channel **dequant
epilogue** (which cuBLAS *cannot* fuse) turned into a decisive end-to-end win. Then push past.

## Research first — think very carefully, spawn parallel agents

Write a plan to `prompts/results/quant.md` first. Note from project memory: int8 multi-stage was once
"the wrong lever — occupancy-bound", and `ldmatrix`+XOR-swizzle is already the int8 default win — so
**re-profile the binding constraint before assuming**. Evaluate:
- **Occupancy vs SMEM/register pressure at 1024³–2048³** — is the IMMA gap occupancy-bound (then: smaller
  tiles / more CTAs / register trimming) or pipeline-bound (then: deeper `cp.async`)? Measure first.
- **`mma.sync` shape & `ldmatrix` for the byte geometry** — confirm the `m16n8k32` int8 path uses the
  optimal fragment loads + swizzle; byte geometry == fp16 so fp16 swizzle derivations port.
- **Split-K / stream-K** for thin-M / small-N decode shapes (the real quantized-inference regime).
- **fp8**: E4M3 forward + E5M2, fragment-reuse `_mt` path (already the fastest fp8 per memory) — sweep
  tile/stage; compare vs a Transformer-Engine-style peer if one can be `dlopen`'d, else vs cuBLAS fp8
  (`cublasLtMatmul` fp8) honestly, else document that no library fp8 peer is buildable here.
- **Fused dequant epilogue** (`out = act((A·Bᵀ)·scale_a·scale_b [+ bias])`): cuBLAS outputs raw `i32` and
  needs a *second* HBM round-trip kernel to dequant — Mercury folds it for ~0 cost. **Measure the full
  quantized-inference output stage (GEMM+dequant) vs the cuBLAS GEMM+dequant *chain*** — that is where
  Mercury should *beat* cuBLAS outright (the fusion lever a library can't use). This is your headline win.
- The autotuner (`autotune.rs`) — per-shape search over int8 candidates with on-disk cache already exists;
  extend it to cover the new candidates and keep the bit-exact cross-check.

## The two binding laws (every commit)

1. **Correctness before speed.** int8 is **bit-exact** — the GPU output must `==` a wrapping-`i32` CPU
   reference over the full `M×N` (i32 add is associative mod 2³², so any order is exact — a stronger bar
   than float). fp8 uses the **tolerance** gate. Gate *before* any speed number. `cargo test` (no gpu
   feature) stays green; GPU code behind `#[cfg(feature = "gpu")]`.
2. **Honesty.** Same-run, checksum-cross-checked, **%-of-cuBLAS-IMMA** + **× vs naive/dp4a** (clock swings
   ~7×; never report absolute GOP/s). Disclose the peer. Measured ≥3× or it's a hypothesis.

## Files you own / shared (append-only)

- **Own:** `ptx_int8.rs`, `ptx_fp8.rs`, `ptx_int4.rs`, `ptx_fp8_train.rs`, and your arms of `autotune.rs`.
- **Append-only:** `baselines.rs` (cuBLAS IMMA / fp8 peers, uniquely named), `lower.rs` (your dispatch
  arm), `gpu.rs` (tests named `quant_*`), `lib.rs`, `Cargo.toml`. **Do NOT** touch
  `ptx_wmma/gemm/flash/conv.rs`, `BENCHMARKS.md`, `CHANGELOG.md`. Numbers → `prompts/results/quant.md`.

## GPU gotchas

PTX must be ASCII. `%tid`→`%tix`. ptxas errors via `jit_log` (`cuModuleLoadDataEx`). `cp.async` dst is a
shared u32 addr. cubin must load from a file (cudarc can't load in-mem cubin). Only same-run ratios count.
Redist NVRTC+cuBLAS DLLs live in gitignored `tools/cuda-redist`.

## Commit discipline

`perf(gpu):`/`feat(gpu):`/`bench(gpu):` with the measured same-run effect in the body. **NO co-author /
"Generated with" trailers.** **Never `git add -A`** — stage by name. **Run the gate as its own step and
read it before committing** (don't let `grep`'s exit code hide a failure). 10–20 green commits.

## Definition of done

int8 ≥75% of cuBLAS IMMA (or a proven roofline argument), the fused GEMM+dequant stage **beating** the
cuBLAS dequant chain, fp8 honestly characterized, every result bit-/tolerance-gated and proven across
≥3 re-runs, in `prompts/results/quant.md`, 10–20 clean commits. Then find the next bottleneck.
