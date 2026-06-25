# End-to-end serving & multi-GPU — paged KV-cache + continuous batching + whole-model decode graph

Branch `perf/gpu-serving` (worktree `Mercury-serving`). Target: **RTX 4050 Laptop, sm_89 (Ada), 6 GB,
ONE GPU**. Goal: close — then beat — the NVIDIA inference-serving stack on what is testable here, single
GPU; design (unmeasured) multi-GPU.

## Reality of the measurement environment (the honesty law, decided up front)

- **GPU is live**: RTX 4050 Laptop, 6141 MiB, driver 592.27, cudarc driver-JIT (no nvcc/ptxas). Verified
  this session (`int4_gemm_matches_reference` ran real device work, passed).
- **No GPU serving peer is installed**: the local PyTorch is **CPU-only** (`torch 2.12.1+cpu`,
  `torch.cuda.is_available() == False`); **vLLM not installed**; no nvcc. So a *GPU* vLLM / TensorRT-LLM /
  torch `.generate()` peer is **not available out of the box**.
- **Primary named peer = Mercury's own eager per-op decode** (no pool, no graph), measured **same-run**,
  bit-exact-gated against the paged+batched+graphed path. This is the *established* methodology in this
  repo: the megakernel headline (M13 ~285×) and `decode_stack_latency` are both measured "vs Mercury's own
  `--backend=gpu` per-op model, not a library." The pool/graph wins are **ratios** (laptop clock swings
  ~7×; only same-run ratios are reportable).
- **Stretch external peer**: attempt a CUDA-enabled torch `.generate()` as an honest floor; if a cu-wheel
  for py3.13 won't run against this driver, report a CPU-torch floor explicitly labeled CPU (a magnitude
  sanity check, not a fair GPU comparison) and document why no GPU library peer exists here.

## What exists (reuse) vs what's missing (build)

Mature & reused unchanged: `DevicePool` (bump arena), `Graph` (capture/replay), the WMMA f16 GEMMs
(`wmma_nt_f16_sm_db` + `_silu`/`_residual` fused epilogues), `rmsnorm`, `CAST_F32_F16`, flash kernels,
the `Gpu` harness (`function`/`stream`/`ctx`/`peak_hbm_gbs`/`sm_count`). **Missing entirely (this is a
clean slate, no collisions):** KV-cache (paged or dense), block table / page allocator, decode-step
(single-token) attention, append-to-cache, sampling, continuous batching, scheduler.

The decisive structural fact: **`ResidentLayerF16` is a *prefill* layer** — it processes a fixed `[S,D]`
and recomputes Q/K/V for the whole sequence every call, with **no KV-cache**. Autoregressive *decode*
(the serving throughput metric) needs a single-token step that attends over a cached past — exactly the
gap this work fills.

## Architecture

### Decode step (per layer), batch of `B` sequences, `M = Bcap` rows
1. RMSNorm `[Bcap,D]` → cast f16. (reuse `rmsnorm`, `CAST_F32_F16`)
2. Q/K/V projections: `[Bcap,D]·[D,D]ᵀ` WMMA f16 GEMM (M=Bcap=64 → reuses the proven 64×64 `_sm_db` tile,
   static shape ⇒ graph-capturable). (reuse `wmma_nt_f16_sm_db`)
3. **Append** new K,V (f16) into the **paged KV-cache** at each sequence's current position, via its block
   table. (new kernel `kv_append`)
4. **Paged decode attention**: grid `(Bcap, heads)`, one CTA per `(seq,head)`; gather K/V through the
   sequence's block table over its ragged `context_len`; online softmax; write `out[seq,head,:]`.
   (new kernel `paged_attn_decode`)
5. O-projection + residual (fused), RMSNorm, FFN up (SiLU fused), FFN down + residual (fused). (reuse the
   fused WMMA epilogues `_residual`/`_silu`)

`Bcap` (max batch) fixed = **64** (multiple of the 64×64 WMMA tile; `D`,`Dff` already %64). The number of
*active* sequences ≤ Bcap varies; inactive rows compute ignored garbage. Fixed M keeps every shape static
— required for one whole-model CUDA graph and matching how real engines bucket batch sizes.

### Paged KV-cache (`paged_kv.rs`, owned)
vLLM-style. K and V each a device slab of fixed-size **blocks**; block holds `block_size` tokens (default
16) × `heads` × `head_dim`, stored **f16** (halves footprint vs f32 — the 6 GB lever). Per-sequence
**block table** maps logical block → physical block id. Host-side **free-list allocator**: `allocate`
pops blocks, `append_token` grows a sequence (new block when the last fills), `free` returns blocks.
The block manager is pure host logic over device pointers ⇒ **CPU-unit-testable with no GPU**.

### Whole-model decode graph (`serving.rs` + `graph.rs`)
Mirror `capture_resident_stack`: pool all decode scratch, run the full N-layer decode step on a capturable
stream, capture into ONE `cuGraphLaunch`. At decode shapes (tiny per-step compute) the per-launch driver
overhead dominates ⇒ folding `N×(~15)` launches into one is the big latency lever.

### Continuous batching (`serving.rs`, owned)
Iteration-level scheduler: a slot table of ≤Bcap active sequences, each with `{block_table, context_len,
pos, done}`. Each step advances all active sequences one token; finished ones are evicted and their blocks
freed; waiting prompts are admitted (prefill) into free slots. Ragged context lengths are handled by the
paged-attn kernel's per-sequence `context_len`. Throughput lever: at batch 1 the 64-row GEMM yields 1
useful token; at batch 64 the *same* GEMM yields 64 — aggregate tokens/s scales with fill.

### Multi-GPU (design + single-GPU simulation; the 2-GPU number is unmeasured)
Megatron tensor parallelism for a decoder layer: QKV proj + FFN-up/gate **column-parallel** (split N, no
comm), attention-out proj + FFN-down **row-parallel** (split K) followed by an **all-reduce** of the
partial `[B,D]` outputs (two all-reduces per layer). Implement the GEMM column/row split + a host-summed
"all-reduce" on **one** GPU (split → 2 partials → sum), bit-exact vs the unsplit GEMM, to validate the
partition math; document the NCCL / driver-P2P all-reduce as the only unmeasured seam. Produces a real
number the instant a 2nd GPU is attached.

## Correctness gating (the first law)

- **Absolute correctness**: each new kernel tolerance-gated vs a CPU f32/f64 reference (CPU↔GPU differ in
  float reduction order — the repo's standing tolerance gate).
- **Paging is numerically invisible (paged == non-paged, bit-exact)**: the paged-attn output must be
  **bit-for-bit identical** across two *different physical block-table layouts* of the same logical
  sequence (contiguous vs fragmented free-list order). True by construction: the block table only changes
  the *load address*, never the value or the accumulation order, so the float ops are identical. This is
  the decode analogue of int8 split-K / transpose bit-exactness.
- **Graph replay**: bit-exact + deterministic vs eager (mirror `resident_stack_graphed_matches_eager`).
- KV quantization is lossy ⇒ tolerance-gated (not bit-exact), like the other quantized paths.
- `cargo test` (no gpu feature) stays green; all GPU code behind `#[cfg(feature = "gpu")]`.

## Milestones

- **P1** `paged_kv.rs`: block manager + block table + device slabs. CPU-unit-tested allocator. ✅/⏳
- **P2** `paged_attention.rs`: decode-attn PTX + launcher; tolerance gate + block-layout bit-exact gate.
- **P3** `serving.rs`: batched `DecodeLayer`/`DecodeModel` (proj + append + paged-attn + FFN), pooled/on-stream.
- **P4** whole-model decode CUDA graph; bit-exact+deterministic gate; same-run speedup vs eager (≥3× target).
- **P5** continuous batching scheduler; aggregate tokens/s vs single-sequence.
- **P6** int8 KV quantization (tolerance-gated); 6 GB-budget footprint win.
- **P7** tensor-parallel design + single-GPU partition simulation (bit-exact); NCCL/P2P documented unmeasured.

## Results (filled as measured, same-run ratios only)

### P2 — paged decode-attention kernel (correctness, the first law)
- **Absolute correctness** (`paged_attention_matches_reference`, ragged ctx `[37,0,16,100,5,64]`, heads=4,
  hd=64): paged decode-attention vs f64 full-softmax reference **max_abs = 1.79e-7, max_rel = 3.87e-5** —
  far under the 1e-2/3e-3 gate (K/V pre-rounded to f16 so only GPU `ex2.approx` + f32 order remains).
- **Paging is numerically invisible** (`paged_attention_invariant_to_block_layout`): the same logical
  sequences under **two different physical block layouts** (slot 0: A=`[0,1,2]` vs B=`[10,11,12]`) give
  **bit-for-bit identical** output. The first-law decode analogue of int8 split-K / transpose bit-exactness.

### P3 — batched decode layer/model (the serving forward, the first law)
- **KV-append scatter** (`serving_kv_append_round_trip`): appending the new token's K/V through the block
  table, read back, is **bit-exact vs f16(input)** at every address.
- **Whole decode step vs f64 reference** (`serving_decode_step_matches_reference`, Bcap=64, ragged ctx
  0..80): RMSNorm→proj→append→paged-attn→O-proj+res→RMSNorm→SiLU-FFN→res matches the f64 oracle at
  **max_abs = 4.55e-4** (every element within the 5e-2 abs gate; the high max_rel is the near-zero-element
  artifact). The integration of every reused WMMA/norm kernel + the new append/paged-attention.
- **Whole-model decode step is paging-invariant** (`serving_decode_step_invariant_to_block_layout`, 4
  layers, Bcap=64): **bit-for-bit identical** across two physical block layouts (ascending vs descending
  slot allocation) — every layer reads/writes the cache correctly; a misread would diverge.

## Multi-GPU design detail (unmeasured)

_see P7._
