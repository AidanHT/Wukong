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
1. **Multi-warp CTA at long S** — `flash_d64_mp4` (4 warps share one staged K/V block: 4× L2 reuse).
   **MEASURED — RESOLVED: a wash ≤3072, only ~12% at S≥4096; NOT the gap-closer.** Two independent
   methodologies agree (cross-process `attn_variants_vs_fused_peer`: mp4 0.48× vs mp 0.44× cuDNN at
   S=4096; clock-cancelled in-process `flash_mp4_vs_mp`: mp4/mp = 0.99×/1.02× at S≤3072, 0.88×/0.88× at
   S=4096/8192). The single-warp grid already saturates the 20-SM GPU at long S, so packing warps into
   fewer CTAs trades parallelism for the reuse and only nets out at the very largest S. mp stays the
   production kernel; mp4 is a noted regime-aware option at S≥4096 but still leaves Mercury at ~0.48×
   cuDNN there — it does **not** close the gap.
   - **Measurement cautionary tale (kept in the `flash_mp4_vs_mp` harness):** the *first* in-process A/B
     reported an incoherent, non-monotonic mp4/mp = **0.25/1.02/0.51/0.70** — a false "4× mp4 win." Cause:
     it timed `mp` first every round right after the clock-pinning GEMMs, so `mp` ate the clock *ramp*
     while `mp4` (second) ran already-boosted. Fix: warm BOTH kernels to steady clock before timing, and
     time each in both orders (mp,mp4,mp4,mp) taking the min — which collapsed the "win" to the true
     ~1.0×/~0.88×. A clean reminder that same-process ≠ clock-cancelled unless the launch order is too.
3. **Deeper cp.async pipeline** (3–4 stage) to hide the K-loop latency the 2-stage pipe exposes.
4. **`ldmatrix`** for Q/K/V fragment loads (the one untried lever per project memory; ~1.2–1.5× on int8).
5. **D=128 tensor-core flash** — the fast `mma` kernels are instantiated at D=64 only; D=128 is the
   modern head dim (Llama). New regime, not just a speedup.
6. **Fusion the library can't do** — fold RoPE / QK-scale / ALiBi-bias / output-proj into the one kernel;
   static seq/head specialization. The durable lead even where raw GEMM ties.
7. **Efficient causal** (skip upper-triangle blocks) — `flash_d64_mpc` exists; measure vs cuDNN causal.

Status: **objective (a) — a named, genuinely fused FA2 peer running same-run — is DONE.** Objective (b)
— competitive-or-ahead — begins now from this honest 0.40–0.80× baseline.

## 3. The fused-RoPE win — the library-can't-fuse lever (objective b, regime #1)

A fused-attention library (cuDNN's fused flash, cutlass mem-efficient fMHA) takes Q/K/V and emits O in
one kernel — it has **no hook to apply RoPE inside**. So a real model running rotary embeddings must run
a *separate* elementwise RoPE pass over Q and K first (an extra HBM round-trip of both), then the fused
attention. Mercury's `flash_d64_mprope` folds the interleaved rotation into the `b32` `mma` fragments it
already loads — one `mma` register packs exactly one `(2t,2t+1)` rotation pair — so RoPE costs ~nothing on
a kernel that's already tensor-core-bound. **This is a fusion the library cannot do**, regardless of how
fast its attention is.

### The honest peer (no strawman)
`baselines::fa2_sdpa_peer_rope` drives the **full pipeline a model actually pays**: an optimized
interleaved-RoPE pass over Q,K + the same cuDNN/cutlass fused SDPA from §1. To keep the peer strong, the
harness times **two** RoPE implementations and takes the **min**: the eager `(reshape→mul→stack)` form
*and* a complex-multiply form (`view_as_complex`/`view_as_real`, the fastest RoPE torch produces here
without triton). It also reports the SDPA-only time, so the **RoPE tax** (pipeline − sdpa) is explicit.
Same f16 Q/K/V + cos/sin to both sides; Mercury's fused-rope O is gated vs an f64 *rotate-then-attend*
oracle at small S and checksum-cross-checked against the peer's rope+sdpa O at all S.

### Result (H=8, D=64, RTX 4050, 3 runs — `gpu::attn_rope_vs_fused_peer`)

**Mercury's one fused kernel vs the peer's RoPE + fused-SDPA pipeline** (median [range] over 3 runs):

| S | Mercury / (rope+sdpa) | verdict | peer RoPE tax (% of sdpa, within-process) |
|---|---|---|---|
| 256 | **2.14×** [1.83–5.71] | **win** | 608–1064% |
| 512 | **1.76×** [1.31–2.95] | **win** | 25–392% |
| 1024 | 0.66× [0.45–0.87] | lose | 101–271% |
| 2048 | 0.29× [0.28–0.32] | lose | **~52%** (48–59%, stable) |

Correctness gate (S≤512): Mercury rope-flash max_abs **3.7e-5–5.5e-5**, peer rope+sdpa **5.0e-5–7.7e-5**,
both vs the f64 oracle ✓.

### Reading it honestly
- **Mercury wins at S ≤ 512** (all six small-S data points > 1×): the library-forced RoPE pass is a large
  fixed cost there, and Mercury erases it. At S=1024–2048 Mercury loses — cuDNN's attention throughput
  (it scales to ~20 TFLOP/s vs Mercury's ~4 plateau, §2) overtakes the RoPE savings. **Crossover ≈ 512–1024.**
- **What's clock-robust vs clock-noisy.** The small-S *pipeline ratio magnitude* is cross-process
  (Mercury process vs the Python peer process, ~7× laptop-clock swing) — hence the wide [range] and why
  the table leads with the **median** and a verdict, not a point estimate. What *is* clock-robust: the
  **RoPE-tax fraction** (pipeline vs sdpa measured in the *same* peer process, same clock) and the
  **win/lose direction** (consistent across all 3 runs at every S). The honest, stable headline number is
  the **~52% RoPE tax at S=2048**: even where cuDNN's attention dominates, the library still pays a ~50%
  surcharge to rotate that Mercury doesn't.
- **Bound on the win.** The peer's RoPE tax is launch/overhead-dominated (~flat 0.2–0.6 ms across S, far
  above the ~tens-of-µs memory-bound floor of a single fused rope kernel). So the win is largest against
  an *eager* rope (a common real deployment) and narrows against a maximally-fused rope; even so, fusing
  it is strictly free for Mercury, so the direction never reverses — only the magnitude.

**Takeaway:** objective (b) is met for the **small-S / prefix / RoPE regime** — Mercury's single fused
kernel beats a *genuinely fused* cuDNN/cutlass attention + an optimized RoPE pass at S ≤ 512, a win the
library is structurally unable to match. Past S=512 the durable lever is closing the raw attention-
throughput gap itself (multi-warp dispatch at long S, D=128, deeper pipeline — §2 lever map), which is
the next front.

## 4. The causal win — Mercury beats BOTH fused peers at S=512, ties cutlass-efficient through S=1024

Causal attention (the decoder mask: query `i` attends only to keys `j ≤ i`) is the regime every LLM
actually runs. Mercury's `flash_d64_mpc` skips every all-masked K-block (the online-softmax K-loop stops
at the diagonal `kb==row`) and masks only the diagonal block — ~half the `mma` work at long S. cuDNN and
cutlass-efficient skip the upper triangle too, so this is a *fair fused-vs-fused* causal comparison
(`is_causal=true`), same f16 Q/K/V, output gated vs the per-head f64 `ref_attn_causal` oracle.

### Result (H=8, D=64, RTX 4050, `gpu::attn_causal_vs_fused_peer`, 3 runs at S≤1024 / 2 at S≥2048)

Mercury vs the **fastest** fused peer, and vs **cutlass-efficient** specifically (median [range]):

| S | Mercury / best fused peer | Mercury / cutlass-efficient | verdict |
|---|---|---|---|
| 512 | **1.12×** [1.03–1.12] (peer=cuDNN) | **1.14×** [1.01–1.16] | **beats BOTH fused peers** |
| 1024 | 0.87× [0.85–0.88] (peer=cuDNN) | **~1.01×** [1.00–1.02] | ties cutlass-efficient |
| 2048 | 0.54× [0.52–0.58] (peer=cuDNN) | 0.90× [0.83–0.93] | competitive w/ efficient |
| 4096 | 0.54× [0.52–0.62] (peer=cuDNN) | 0.77× [0.76–0.88] | competitive w/ efficient |

Correctness gate at S=512: Mercury causal max_abs **3.3e-4**, peer **4.6e-4** vs the f64 oracle ✓.

### Reading it honestly
- **Mercury beats both fused FA-class peers at S=512** (1.03–1.12× cuDNN *and* 1.01–1.16× efficient,
  consistent across all 3 runs — not a single noisy point) and **ties cutlass mem-efficient fMHA through
  S=1024**. This is a *far* stronger standing than the non-causal §2 baseline (0.40–0.80×): the clean
  triangular skip roughly doubles Mercury's effective throughput, and at short/medium S Mercury's skip is
  as efficient as the libraries' — sometimes more.
- **cuDNN's causal scales ahead at S≥2048** (to ~30 TFLOP/s full-S²) where Mercury plateaus (~16–18k
  GF/s) — the same single-warp long-S ceiling as §2. vs cutlass-efficient Mercury stays competitive
  (0.83–0.90×) even there; only cuDNN pulls away.
- **GF/s convention:** both sides use the full `4·H·S²·D` count (so the *ratio* is exact); the absolute
  number is ~2× the useful causal FLOP — a shared, disclosed convention, not a per-side advantage.
- **Honesty caveats** mirror §1/§3: cross-process (Mercury wall-clock vs peer CUDA-event time, ~7× clock
  swing) ⇒ ranges over 3 runs, with the *direction* (win/tie/trail) stable at every S; the peer's
  launch overhead is excluded so any Mercury win is conservative.

**Takeaway:** a **second** objective-(b) win, and a more important one than RoPE — in the *causal*
regime that real decoders run, Mercury's fused flash **beats both genuinely-fused peers at S=512 and
matches cutlass-efficient through S=1024**. Combined with §3 (fused RoPE ≤512) Mercury is
competitive-or-ahead of a real FA-2-class kernel across the short/medium-context regime; the residual
gap is purely cuDNN's long-S (≥2048) attention-throughput scaling.
