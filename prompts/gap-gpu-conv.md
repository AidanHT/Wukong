# Convolution vs cuDNN — implicit-GEMM + Winograd, and a real cuDNN peer

You are a Claude Code session on **Mercury** (`mercuryc` emits PTX, driver-JIT via `cudarc`, no toolkit).
Target: **mobile RTX 4050, sm_89 (Ada), ~30–50 W**. First action:
`git worktree add ../Mercury-conv2 -b perf/gpu-conv-2 main` and work inside it (or `git checkout -b`).

## Prime Directive (how you work)

Continuously iterate through the entire codebase until Mercury performs better than C, C++, and Rust
**and closes — then beats — cuDNN on convolution**. **No scope restrictions.** Remove any blocker;
explore options **even outside scope**. Improve everything. **No limit.** **Assume the metric is NOT
strong until you PROVE it is** across multiple same-run iterations vs a real cuDNN peer. **Do not stop**
until conv is measured against cuDNN and competitive. Think very carefully. If you CAN improve it, **do
it.** Time doesn't matter. **Run many sub-agents in parallel.** **Make 10–20 commits, NO co-authored
lines.** Right now conv is measured only vs *naive* C (~6×) — that proves nothing about the real gap.

## Your mission

Mercury lowers conv as **im2col + GEMM**, benchmarked only against a naive direct-conv C kernel (~6×).
cuDNN — the industry standard — uses **implicit GEMM, Winograd, and FFT** with per-shape autotuning, and
is far ahead. **Your job:** (a) stand up a real **cuDNN peer** to measure the true gap; (b) implement the
algorithms that close it — **implicit-GEMM** (no materialized im2col buffer) and **Winograd** for 3×3 —
and a **fused conv+bias+activation** epilogue; reach competitiveness with cuDNN on the common shapes,
then beat it via fusion + static-shape specialization.

## Research first — think very carefully, spawn parallel agents

Plan to `prompts/results/conv.md`. Evaluate:
- **Real peer:** `dlopen` **cuDNN** (redist DLL — `cudnnConvolutionForward` with `cudnnFindConvolution…`
  picking the autotuned algo) for an honest Tier-B peer; keep the existing naive direct-conv as Tier A.
  Cross-check Mercury's output vs cuDNN's to tolerance.
- **Implicit GEMM** (the cuDNN/CUTLASS default): fold the im2col gather into the GEMM's *global→SMEM*
  load addressing — no `M×(C·R·S)` materialized buffer, which is most of the current cost — reusing the
  tensor-core GEMM machinery. This is the highest-leverage lever for the common case.
- **Winograd** F(2×2, 3×3) and F(4×4, 3×3): the input/filter/output transforms + the elementwise-multiply
  in the Winograd domain — a large FLOP reduction for 3×3 stride-1 convs (the CNN workhorse). Mind the
  numerical tolerance (Winograd has larger error — gate accordingly).
- **Layout**: NHWC + fp16 tensor cores (cuDNN's fast path) vs NCHW; pick what the hardware wants.
- **Fusion** cuDNN can't always do: conv+bias+BN-fold+activation in one kernel; static (N,C,H,W,K,R,S)
  specialization at compile time.
- **Coverage**: 1×1 (pointwise = GEMM), 3×3 (Winograd), strided, depthwise/grouped, dilated.

## The two binding laws (every commit)

1. **Correctness before speed.** Conv output gated **over the full output** vs the CPU reference — the
   **`c·√K·ε` tolerance** for the GEMM/implicit-GEMM path; a **looser, documented** tolerance for Winograd
   (it has inherent extra error — state the bound and why it's acceptable). Gate before any speed number.
   `cargo test` (no gpu feature) stays green; GPU code behind `#[cfg(feature = "gpu")]`.
2. **Honesty.** Same-run, checksum-cross-checked, **%-of-cuDNN** + **× vs naive** (clock swings ~7×).
   Disclose the cuDNN algo cuDNN chose. Measured ≥3×.

## Files you own / shared (append-only)

- **Own:** `ptx_conv.rs` (+ any new `ptx_winograd.rs` you add).
- **Append-only:** `baselines.rs` (cuDNN peer, uniquely named), `lower.rs` (your conv dispatch arm),
  `gpu.rs` (tests named `conv_*`), `lib.rs`, `Cargo.toml`. **Do NOT** touch
  `ptx_wmma/gemm/int8/fp8/flash.rs`, `BENCHMARKS.md`, `CHANGELOG.md`. Numbers → `prompts/results/conv.md`.

## GPU gotchas

PTX must be ASCII. `%tid`→`%tix`. ptxas errors via `jit_log`. `cp.async` dst = shared u32 addr. cubin
loads from a file. Only same-run ratios count. Redist cuDNN/NVRTC DLLs in gitignored `tools/cuda-redist`
(you may need to add the cuDNN redist to that one `pip` install — document it in your results file).

## Commit discipline

`perf(gpu):`/`feat(gpu):`/`bench(gpu):`, measured same-run effect + the cuDNN algo in the body. **NO
co-author / "Generated with" trailers.** **Never `git add -A`.** **Run the gate as its own step and read
it before committing.** 10–20 green commits.

## Definition of done

A reproducible same-run cuDNN peer, implicit-GEMM + Winograd implemented and gated, Mercury competitive
with cuDNN on 1×1/3×3/strided (or a proven roofline argument for any residual), fused conv+bias+act
beating the cuDNN+epilogue chain, proven ≥3×, results in `prompts/results/conv.md`, 10–20 clean commits.
Then find the next bottleneck.
