# Beat a *real* fused FlashAttention-2 / cuDNN attention, not just the unfused cuBLAS chain

You are a Claude Code session on **Mercury** (`mercuryc` emits PTX, driver-JIT via `cudarc`, no toolkit).
Target: **mobile RTX 4050, sm_89 (Ada), ~30–50 W**. First action:
`git worktree add ../Mercury-attn2 -b perf/gpu-attention-2 main` and work inside it (or `git checkout -b`).

## Prime Directive (how you work)

**Hold this as bedrock conviction: a solution EXISTS and there is CERTAINLY a way to improve this
metric. The gold-standard library already achieves it, so the performance is physically available on
this silicon and a path to it is guaranteed to exist — your only job is to find Mercury's path, and you
WILL find it. Treat every "this might be impossible" thought as "I have not found the lever yet." A gap
is never a ceiling; it is an unsolved problem with a known-achievable answer. NEVER conclude it can't be
done. When one lever stalls, that is the signal to switch levers — there is always another lever, and
the next one is the one that works. Improvement is not a hope here; it is a certainty you have not yet
collected.**

Continuously iterate through the entire codebase until Mercury performs better than C, C++, and Rust
**and closes — then beats — the NVIDIA gold-standard fused attention**. **No scope restrictions.** Remove
any blocker; explore options **even outside scope**. Improve everything. **No limit.** **Assume the
metric is NOT strong until you PROVE it is** across multiple same-run iterations. **Do not stop** until
Mercury's fused attention is measured against — and competitive with — a genuinely *fused* FA2-class
peer. Think very carefully. If you CAN improve it, **do it.** Time doesn't matter. **Run many sub-agents
in parallel.** **Make 10–20 commits, NO co-authored lines.** "Beating the *unfused* cuBLAS chain" is
**not** the bar — the SOTA is *fused* FA2; until you measure against it you have not proven anything.

## Your mission

Mercury's fused flash-attention (`flash_d64_mp`: register-resident `mma.sync` core + `cp.async`-pipelined
K/V SMEM prefetch) is **3.6–5× the *unfused* cuBLAS attention chain** and 205–738× naive CUDA-C — but
that chain is the *pre-FlashAttention* baseline. **A real fused FA2 kernel beats that chain too**, so
Mercury's standing vs the actual SOTA is **unknown**. The blocker: NVRTC here has no headers, so
`nvcuda::wmma` / CUTLASS / the FlashAttention source **won't compile** as an in-process peer.

**Two objectives:** (a) *find a way to benchmark against a genuinely fused FA2-class peer* — this is the
research crux; (b) make Mercury's kernel competitive with it, then beat it via levers a library can't use
(static-shape specialization, fused norm/RoPE/bias, determinism).

## Research first — think very carefully, spawn parallel agents

Plan to `prompts/results/attention.md`. The peer problem is the heart of this slice — research hard:
- **Real fused peers you might `dlopen`/subprocess** (no toolkit build needed): cuDNN's fused attention
  (`cudnnMultiHeadAttn` / the cuDNN-frontend runtime fusion engine, redist DLL); the prebuilt
  **FlashAttention** `.so`/`.pyd`; **PyTorch `scaled_dot_product_attention`** (which dispatches to the
  fused Ada kernel) via a tiny Python subprocess timing harness; TensorRT's MHA. Pick the most honest one
  you can drive same-run on this box and document exactly what it is.
- **Kernel levers** for the Mercury side: `ldmatrix` for Q/K/V fragment loads (the one untried lever per
  project memory); deeper K/V `cp.async` pipelining; head dim **D=128/256** (not just 64); GQA/MQA
  (shared K/V heads — the modern layout); efficient **causal** masking (skip upper-triangle blocks);
  the online-softmax rescaling cost and 2-pass vs streaming numerics; register pressure vs occupancy.
- **Fusion the library can't do**: fold RoPE, the QK-scale, bias/ALiBi, and the output projection into the
  one kernel; static seq/head specialization at compile time.
- **Driving a fused peer is itself a solvable problem — solve it.** There IS a way to benchmark a real
  fused kernel on this box; exhaust the options (the cuDNN DLL, a prebuilt FlashAttention `.so`, a
  PyTorch-SDPA subprocess) with full conviction before accepting anything less. Only if you have truly
  exhausted them do you report vs the unfused chain — and then be precise that the fused comparison is
  *pending*, not that you beat FA2. Your win must be real to count; a fused-peer win you didn't run isn't
  one yet, so go get the peer running. Treat "no peer" as a bug to fix, never an endpoint.

## The two binding laws (every commit)

1. **Correctness before speed.** Attention output is gated to the **`c·√K·ε` tolerance** vs the CPU
   reference over the full output (the softmax + matmuls reassociate); causal/GQA variants each gated.
   `cargo test` (no gpu feature) stays green; GPU code behind `#[cfg(feature = "gpu")]`.
2. **Honesty.** Same-run, checksum-cross-checked, reported as a ratio vs a **named** peer (clock swings
   ~7×). The peer's identity (cuDNN vs PyTorch-SDPA vs unfused chain) is part of the result. ≥3 re-runs.

## Files you own / shared (append-only)

- **Own:** `ptx_flash.rs`, `ptx_norm.rs`.
- **Append-only:** `baselines.rs` (the fused-attention peer harness, uniquely named), `lower.rs` (your
  dispatch arm), `gpu.rs` (tests named `attn_*`/`flash_*`), `lib.rs`, `Cargo.toml`. **Do NOT** touch
  `ptx_wmma/gemm/int8/fp8/conv.rs`, `BENCHMARKS.md`, `CHANGELOG.md`. Numbers → `prompts/results/attention.md`.

## GPU gotchas

PTX must be ASCII. `%tid`→`%tix`. ptxas errors via `jit_log`. `cp.async` dst = shared u32 addr. cubin
loads from a file. Only same-run ratios count. Redist DLLs in gitignored `tools/cuda-redist`.

## Commit discipline

`perf(gpu):`/`feat(gpu):`/`bench(gpu):`, measured same-run effect in the body, naming the peer. **NO
co-author / "Generated with" trailers.** **Never `git add -A`.** **Run the gate as its own step and read
it before committing.** 10–20 green commits.

## Definition of done

A reproducible same-run benchmark vs a **named, genuinely fused** FA2-class peer (or a documented,
honest reason none is drivable here + the best available proxy), Mercury competitive-or-ahead on at least
the static-shape / fused-RoPE regime, every variant tolerance-gated and proven ≥3×, results in
`prompts/results/attention.md`, 10–20 clean commits. Then push the next lever.
