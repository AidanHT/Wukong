# Real GPT-2 124M forward — dispatch + correctness (branch `perf/gpt2-real-dispatch`)

Goal: take the real pretrained GPT-2 124M forward that already **matches** HuggingFace's logits
(`examples/gpt2_infer.wk`, rel ~2e-6) and make the *same* program, at a compute-bound sequence
length, run through the tuned runtime kernels — so we can honestly ask whether it also **beats**
PyTorch on CPU. The vehicle is `examples/gpt2_forward_bench.wk` (S=512); the peer is
`tools/bench_gpt2_torch.py` (real HuggingFace `GPT2Model`, identical trunk + last-position-LM workload
and identical deterministic ids, so the last-row logits cross-check exactly).

## Solid, power-independent results

**Full kernel dispatch.** Every compute-heavy op is spelled in its recognized form and lowers to a
tuned kernel — verified by `wukongc --emit=mir -O2 examples/gpt2_forward_bench.wk | grep wukong_`:

| op | kernel | count / fwd |
|----|--------|-------------|
| Q/K/V/O/W1/W2 projections (fused bias) | `wukong_sgemm_nt_epi` | 6 |
| scaled scores α·(Qh·Khᵀ) | `wukong_sgemm_nt_alpha` | 1 |
| head output P·Vᵀ (V repacked transposed) | `wukong_sgemm_nt` | 1 |
| the 3 LayerNorms (γ/β in [i]-indexed locals) | `wukong_norm_affine_f32` | 3 |
| the row softmax | `wukong_norm_f32` | 1 |
| tanh-approx GELU (`gelu()` intrinsic) | `wukong_vmath_f32` | 1 |
| residual adds | `wukong_velem_f32` | 2 |

Getting there needed one compiler fix — `emit_norm` passed a `[]f32` slice's 16-byte fat-pointer
buffer to the norm kernel instead of its data pointer (segfault); now routed through
`kernel_base_ptr` like the GEMM path (`tests/run/norm_slice.wk` locks it, differential-gated).

**Correctness — exact match with real PyTorch.** With every op dispatched, the forward still
reproduces HuggingFace GPT-2 bit-for-close:

```
wukong argmax(last) = 338   torch argmax(last) = 338   rel = 2.064e-06   -> MATCH
```

(`GELU` via the intrinsic and the NT-form P·V change float reassociation vs the scalar spelling, yet
the argmax and rel are unchanged.)

## Timing — measured on AC+charging (single-core = charging-immune, reportable)

Session note: the first measurements were on **battery** and are discarded — the identical vehicle
binary swung 1.37 s ↔ 2.16 s single-core (±58%) from battery DVFS. Once on AC+charging the vehicle was
stable to ~3% across runs; single-core is charging-immune (the all-core numbers below are throttled by
the ~25% AC-charging all-core cap and are directional only). All rows S=512, adjacent same-window,
power confirmed AC before+after, cross-checks < 1e-6.

### Single-core (the clean, reportable comparison), best min ms/forward

| rank | config (1 core) | ms | vs Wukong vehicle |
|------|-----------------|----|-------------------|
| 1 | **HF `GPT2Model`** — real PyTorch GPT-2, QKV fused as one Conv1D GEMM (strict MKL/OMP=1) | **1085** | Wukong **1.07× slower** |
| 2 | **Wukong vehicle** (real weights, fully dispatched) | **1161** | — |
| 3 | torch.compile max-autotune, manual forward (Inductor CPP GEMM broken → pinned ATEN/MKL) | 1269 | Wukong 1.01× faster |
| 4 | torch eager SDPA / manual, unfused (torch 2.12.1+cpu) | 1791–1831 | Wukong **1.45× faster** |
| 5 | C(gcc -ffast-math) | 5445 | Wukong **4.3–5.6× faster** |
| 6 | C(gcc -O3 -march=native) | ~25000 | Wukong **~20× faster** |

**Verdict: Wukong exceeds eager PyTorch (1.45×), matches compiled PyTorch, and beats optimized C by
5.6–20× — but the strongest hand-optimized PyTorch (HF's QKV-fused `GPT2Model`) is ~7% faster
single-core, so it is NOT beaten.** Honesty requires the strongest baseline, so the headline single-core
result is a ~7% loss to HF, not a win over the weaker eager/manual peers.

Root cause of the 7%: HF fuses Q/K/V into ONE `[S,768]×[768,2304]` Conv1D GEMM; the vehicle issues three
separate `[S,768]×[768,768]` NT GEMMs (~22% of the forward's FLOPs). MKL's single-core microkernel is
also simply very well tuned. NB: `wukong_xbench`'s `bench_model` peer is the *unfused* manual forward
(1831 ms eager), so its "1.45× faster than PyTorch" is against the weaker peer — real HF `GPT2Model`
(1085 ms) is 1.45× faster than *that* peer and 7% faster than Wukong.

### All-core (directional only — throttled by AC-charging cap, and shaped-vs-real)

`bench_model`'s Wukong `@parallel` GPT-2-shaped trunk was **249 ms** vs HF `GPT2Model` all-thread
**259.8 ms** (same session): ~parity, ~4% Wukong edge — but the vehicle itself is single-core (serial
kernels; only a `@parallel` function selects the multicore kernels), so this is the *shaped* trunk, not
the real-weights vehicle, and both are power-capped. Not a clean claim.

### Open levers toward a clean real-weights "exceed"
1. **QKV fusion** in the vehicle (one `[S,768]×[768,2304]` GEMM + split) — matches HF's structure; likely
   worth ~2–4% single-core, not enough alone to overcome 7% vs MKL.
2. **`@parallel` real-weights vehicle** (port `bench_model`'s block: loop-local per-head scratch, called
   12× from `main`) → measure all-core vs HF all-thread at FULL charge. Given shaped `@parallel` already
   ≈ HF all-thread, a real-weights all-core win is plausible but unproven — this is the most promising route.
