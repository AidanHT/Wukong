# Beating a *genuinely fused* FlashAttention-2 peer — results

Mission: stop measuring Mercury's fused flash against the *unfused* cuBLAS chain (the
pre-FlashAttention baseline) and measure it against a **genuinely fused FA2-class kernel**, then close
and beat the gap. Target: mobile RTX 4050, sm_89 (Ada), ~30–50 W.

## 1. The peer problem — SOLVED (the research crux)

NVRTC here has no headers, so `nvcuda::wmma` / CUTLASS / the FlashAttention source won't compile as an
in-process peer. The path that works: **PyTorch's fused `scaled_dot_product_attention` backends**, driven
as a subprocess over the *same* f16 Q/K/V bytes Mercury runs.

Empirically probed on this box (`torch 2.6.0+cu124`, RTX 4050, driver 592.27):

| SDPA backend | status | what it is |
|---|---|---|
| `FLASH_ATTENTION` | ❌ not built on the Windows wheel | FlashAttention-2 |
| `EFFICIENT_ATTENTION` | ✅ **works** | cutlass mem-efficient fMHA (genuinely fused) |
| `CUDNN_ATTENTION` | ✅ **works** | **cuDNN's fused flash attention** (the gold standard the prompt named) |
| `MATH` | ✅ works | unfused, materialized softmax (the *unfused* anchor, ≈ the cuBLAS chain) |

So the named, genuinely fused peer is **cuDNN's fused attention** (with cutlass mem-efficient fMHA as a
second fused peer). Both are real FA-class kernels; the faster (cuDNN) is the headline bar. This is NOT
the unfused chain — cuDNN runs ~15–26× the MATH backend, the signature of a fused kernel.

### How it's wired (reproducible)
- `tools/fa2_sdpa_peer.py` — forces each fused SDPA backend in turn, CUDA-event-times it (so Python's
  per-call dispatch overhead is **excluded** — fair to the peer), writes the chosen backend's O (f32) +
  a flat report. A one-time `tools/torch-cuda-venv` (CUDA torch + numpy) hosts it.
- `baselines::fa2_sdpa_peer` (append-only) — dumps the identical f16 Q/K/V Mercury runs, drives the
  subprocess, parses the report, returns O + per-backend timings. `fa2_peer_available()` skips (never
  fails) when the venv is absent.
- `gpu::tests::attn_vs_fused_peer` (append-only, `#[ignore]`) — same f16 bytes to both, output
  tolerance-gated vs the per-head f64 oracle (small S) + checksum-cross-checked (all S), Mercury
  wall-clock vs the peer's CUDA-event time, REPS≥3 on a warmed clock.

Run:
```
MERCURY_FA2_PYTHON=<main>/tools/torch-cuda-venv/Scripts/python.exe \
  cargo test -p mercury_codegen_gpu --features gpu attn_vs_fused_peer -- --ignored --nocapture --test-threads=1
```

### Honesty notes
- **Cross-process clock skew** is the one residual risk (the laptop GPU clock swings ~7×). Mitigated by:
  warm the GPU first, run peer + Mercury back-to-back in one window, REPS≥3 best-of, and report the
  unfused MATH backend as a cross-anchor (it should track Mercury's known "3.6–5× the cuBLAS chain").
- The peer's CUDA-event time excludes its launch overhead; Mercury's wall-clock includes its (Rust,
  <1%) launch overhead — so **any Mercury win is the conservative direction**.
- Both sides use the identical `4·H·S²·D` FLOP count; outputs cross-checked within f16 tolerance.

## 2. Honest baseline — Mercury `flash_d64_mp` vs cuDNN fused (H=8, D=64, same-run)

First measured standing (debug build, REPS=3, warmed clock):

| S | Mercury GF/s | cuDNN GF/s | **Mercury / cuDNN** | efficient GF/s | MATH (unfused) GF/s |
|---|---|---|---|---|---|
| 512 | 5713 | 10411 | **0.55×** | 8182 | 1177 |
| 1024 | 9357 | 11730 | **0.80×** | 10304 | 782 |
| 2048 | 9518 | 18742 | **0.51×** | 11796 | 793 |
| 4096 | 8437 | 21124 | **0.40×** | 12011 | 796 |

Correctness gate at S=512: Mercury max_abs **3.6e-5**, cuDNN max_abs **5.3e-5** vs the f64 oracle ✓.

### Gap analysis (the lever map)
- Mercury is **0.40–0.80× cuDNN**, worst at long S. Mercury **plateaus at ~8–9 TFLOP/s** while cuDNN
  **scales up** with S (10.4 → 21.1 TFLOP/s). That divergence is the whole story.
- Signature: Mercury's production kernel is **1 warp / CTA, 16 query rows, 2-stage cp.async**. At long S
  each CTA streams the full K/V serially with only double-buffered prefetch and ~21% occupancy — the
  K-loop latency is **not hidden**, so throughput falls as S grows. cuDNN keeps the tensor cores fed
  (bigger per-CTA tiles / more warps / deeper pipeline), so it scales the other way.
- Mercury already runs ~10× the unfused MATH chain — consistent with the documented "3.6–5× the cuBLAS
  chain" (cuBLAS chain ≈ 2× MATH). The *fused-peer* comparison is the new, harder, honest bar.

### Levers to close → beat (in priority order)
1. **Multi-warp CTA at long S** — `flash_d64_mp4`/`mp8` already exist (4/8 warps share one staged K/V
   block: 4–8× L2 reuse + higher occupancy) but aren't dispatched. If they win at long S, dispatching
   them is a near-free recovery of the S≥2048 collapse. *(measuring next)*
3. **Deeper cp.async pipeline** (3–4 stage) to hide the K-loop latency the 2-stage pipe exposes.
4. **`ldmatrix`** for Q/K/V fragment loads (the one untried lever per project memory; ~1.2–1.5× on int8).
5. **D=128 tensor-core flash** — the fast `mma` kernels are instantiated at D=64 only; D=128 is the
   modern head dim (Llama). New regime, not just a speedup.
6. **Fusion the library can't do** — fold RoPE / QK-scale / ALiBi-bias / output-proj into the one kernel;
   static seq/head specialization. The durable lead even where raw GEMM ties.
7. **Efficient causal** (skip upper-triangle blocks) — `flash_d64_mpc` exists; measure vs cuDNN causal.

Status: **objective (a) — a named, genuinely fused FA2 peer running same-run — is DONE.** Objective (b)
— competitive-or-ahead — begins now from this honest 0.40–0.80× baseline.
