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

## Timing — NOT YET REPORTABLE (needs AC)

The machine was on **battery** for this session (51% → 25%, discharging). Battery DVFS makes
sustained-load timings non-comparable, and this session proved it directly: **the identical release
binary of the vehicle measured 1.37 s and 2.16 s single-core in two runs ~20 min apart** (±58%),
purely from battery-clock drift. An adjacent same-window read had Wukong-1c at 2.16 s vs torch eager
1-thread at 1.96 s — but a ~10% gap under ±58% program-level noise is **inconclusive**, so no speed
claim is made here. Per the project's power-state law, this does not count.

### How to get the reportable number (plug in to AC, ideally full charge)
1. `RAYON_NUM_THREADS=1 wukongc --run --backend=native examples/gpt2_forward_bench.wk` → Wukong-1c µs.
2. Immediately (adjacent): `tools/torch-venv/Scripts/python.exe tools/bench_gpt2_torch.py 512` →
   torch eager 1-thread / all-thread / compiled ms, plus the cross-check.
3. Report the **same-window ratio**, single-core (charging-immune) first; all-core only at full charge.

Prior context that makes a win plausible (to be confirmed, not assumed): the GPT-2-*shaped* trunk in
`wukong_xbench` (`bench_model`) has beaten torch eager/compiled 1-thread at S=512 in earlier AC runs.
The vehicle here is the *real-weights* version of that same forward, now at dispatch parity.
