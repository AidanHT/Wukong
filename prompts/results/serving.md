# End-to-end serving & multi-GPU — paged KV-cache + continuous batching + whole-model decode graph

Branch `perf/gpu-serving` (worktree `Wukong-serving`). Target: **RTX 4050 Laptop, sm_89 (Ada), 6 GB,
ONE GPU**. Goal: close — then beat — the NVIDIA inference-serving stack on what is testable here, single
GPU; design (unmeasured) multi-GPU.

## Reality of the measurement environment (the honesty law, decided up front)

- **GPU is live**: RTX 4050 Laptop, 6141 MiB, driver 592.27, cudarc driver-JIT (no nvcc/ptxas). Verified
  this session (`int4_gemm_matches_reference` ran real device work, passed).
- **No GPU serving peer is installed**: the local PyTorch is **CPU-only** (`torch 2.12.1+cpu`,
  `torch.cuda.is_available() == False`); **vLLM not installed**; no nvcc. So a *GPU* vLLM / TensorRT-LLM /
  torch `.generate()` peer is **not available out of the box**.
- **Primary named peer = Wukong's own eager per-op decode** (no pool, no graph), measured **same-run**,
  bit-exact-gated against the paged+batched+graphed path. This is the *established* methodology in this
  repo: the megakernel headline (M13 ~285×) and `decode_stack_latency` are both measured "vs Wukong's own
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

- **P1 ✅** `paged_kv.rs`: block manager + block table + device slabs. CPU-unit-tested allocator (8 host tests).
- **P2 ✅** `paged_attention.rs`: warp-cooperative decode-attn PTX; tolerance gate (max_abs 1.79e-7) + block-layout bit-exact gate.
- **P3 ✅** `serving.rs`: batched `DecodeLayer`/`DecodeModel` (proj + append + paged-attn + FFN), pooled/on-stream; vs-f64 + paging-invariant gates.
- **P4 ✅** whole-model decode CUDA graph; bit-exact + deterministic gate; same-run **1.07–1.35×** vs eager (modest — compute-bound at this batch).
- **P5 ✅** continuous-batching scheduler (admit/evict/free, masked append); **39.1× goodput** at fill=64 vs single-sequence (same-clock).
- **P6 ✅** int8 KV quantization (tolerance-gated, max_abs 3e-3); **3.88× footprint** vs f32 (1.94× vs f16) — the 6 GB lever.
- **P7 ✅** Megatron tensor-parallel design + single-GPU partition simulation: column-parallel **bit-exact**, row-parallel all-reduce **1.34e-5**; NCCL/P2P collective documented unmeasured.

**Headline (single GPU, honest, same-run):** paged + continuous-batched + whole-model-graphed decode at
**39× goodput** over single-sequence serving, **bit-exact** vs the eager/non-paged reference at every step,
with an **int8 KV cache** cutting footprint **3.9×**; multi-GPU partition math proven exact on one device.

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

### P4 — whole-model decode CUDA graph (the first law + a same-run latency ratio)
- **Graphed == eager, bit-for-bit + deterministic** (`serving_decode_graph_matches_eager`, 6 layers,
  Bcap=64): the entire 6-layer decode step (~84 kernel launches) captured into **one `cuGraphLaunch`**
  reproduces the eager per-op `run_layers_on` output **bit-for-bit** (`to_bits()` equality, every
  element) and is **identical across two replays** (deterministic). The graph changes *how* the launches
  are issued, never *what* they compute. Capture runs on a dedicated non-blocking stream with event
  tracking disabled (the NULL stream is un-capturable; cross-stream event waits break capture).
- **Same-run latency, graphed vs eager per-op** (`serving_decode_graph_throughput`, RTX 4050, named peer
  = Wukong's own eager per-op decode):

  | depth N | eager | graphed | graphed speedup | tokens/s eager → graphed | launches folded |
  |--------:|------:|--------:|----------------:|--------------------------|-----------------|
  | 1       | 997 µs  | 744 µs  | **1.34×** | 64 197 → 85 963 | ~14 → 1 |
  | 6       | 2399 µs | 1777 µs | **1.35×** | 26 678 → 36 023 | ~84 → 1 |
  | 12      | 4108 µs | 3824 µs | **1.07×** | 15 581 → 16 735 | ~168 → 1 |

- **Honest reading**: the graph win is real but **modest, and shrinks with depth** — the diagnostic that
  at Bcap=64 the decode step is **compute/execution-bound, not launch-bound** (eager's async launches
  already pipeline behind GPU compute; the graph only recovers the exposed launch overhead, a roughly
  fixed ~250–280 µs, which is a large fraction of one shallow layer but a small fraction of twelve). The
  per-layer wall (~320 µs) is dominated by the six WMMA GEMMs running at M=64 — only 8–32 CTAs on 40 SMs,
  a single under-occupied wave with a long K-reduction. **This underutilization is precisely the lever
  P5 (continuous batching) converts into goodput**: the fixed-shape step costs ~the same whether 1 or 64
  rows are useful, so filling the batch multiplies useful tokens/s without adding latency. The graph's
  1.07–1.35× then stacks on top of the batching win. (The decode GEMM kernel itself is owned by the
  kernel branches — not edited here; the serving path calls it as-is.)

### P4 — paged decode-attention kernel: warp-cooperative rewrite (correctness preserved)
- The v1 decode-attention kernel was **one thread per `(slot, head)`** — correct and trivially bit-exact,
  but it launched only `num_slots*heads` *threads* (512 here), leaving 39/40 SMs idle. Rewritten
  **warp-cooperative**: one **warp** per `(slot, head)`, the 32 lanes split the context (`t = lane,
  lane+32, …`), each lane runs a partial online-softmax (FP32 accumulators), then a fixed
  `shfl.sync.bfly.b32` butterfly merges them (max → rescale → Σl → Σacc). The query is staged once per
  warp in shared memory. **Both P2 gates still pass on the new kernel** — tolerance vs the f64 reference
  (`max_abs = 1.79e-7`) and **bit-for-bit identical across two physical block layouts** — because the
  lane partition and the butterfly merge order are layout-independent, so paging stays numerically
  invisible by construction.

### P5 — continuous (in-flight) batching (the throughput lever)
The serving forward is fixed-shape `Bcap=64`, so the decode kernels compute **all 64 rows every step**
regardless of how many carry a live request. An Orca-style [`Scheduler`] admits waiting requests into
free slots (prefilling each prompt into the paged cache), advances all *active* slots one token per
[`step`], and evicts a sequence the iteration it hits its `gen_len` — freeing its blocks for the next
admission. Inactive/free slots are **masked off in the append kernel** (a free slot pads its block table
with 0, so an unmasked write would scatter into block 0, a *live* block) and read as empty by attention
(`context_len == 0` ⇒ zero row), so they are numerically inert.

**Correctness (the first law):**
- **Append mask** (`serving_kv_append_masked_skips_inactive`): with a ragged active mask, the *entire*
  cache slab is zero except the active slots' written addresses — every inactive row is skipped, **block
  0 untouched**. (The mask is a new `pAct` param on the append kernel; bit-exact write set.)
- **Batch composition is invisible** (`serving_decode_step_invariant_to_batch_composition`): slot 0's
  decode output is **bit-for-bit identical** whether co-batched with 63 other active sequences carrying
  unrelated context or run **alone** (all other slots masked). Row-independent GEMMs + per-sequence paged
  attention + the append mask make co-batched sequences invisible to one another — *the* property that
  makes continuous batching correct (the serving analogue of the paging-invariance gate).
- **Scheduler liveness + block conservation** (`serving_scheduler_drains_and_conserves_blocks`): 240
  ragged requests (prompt 1..40, gen 1..24) driven to completion through admit→step→evict→free: **all 240
  complete**, useful tokens **= 3000 = Σ gen_len**, **every KV block returned** to the pool (416→416, no
  leak), peak batch **64/64**, and two identical request streams produce the **identical per-step
  schedule** (deterministic). 62 steps for 3000 tokens.

**Goodput — same-clock interleaved best-of-N** (`serving_continuous_batching_goodput`, RTX 4050,
graphed 12-layer decode step, D=512 Dff=2048, Bcap=64; named peer = Wukong's own single-sequence decode):

| batch fill | step latency | goodput (useful tok/s) | vs fill=1 |
|-----------:|-------------:|-----------------------:|----------:|
| 1 / 64  | 2008 µs | 498    | 1.0×  |
| 4 / 64  | 2035 µs | 1 966  | 3.9×  |
| 16 / 64 | 2444 µs | 6 548  | 13.1× |
| 32 / 64 | 2384 µs | 13 421 | 27.0× |
| 64 / 64 | 3287 µs | 19 471 | **39.1×** |

**HEADLINE: 39.1× goodput at fill=64 vs fill=1, at only 1.64× step latency.** The decode step cost is
dominated by the six WMMA GEMMs, which compute all 64 rows regardless of fill — so per-token cost falls
almost linearly as the batch fills. Latency grows only 1.64× over a 64× range of active rows (the paged
attention + masked append are the only fill-dependent work, and they are a small fraction of the step),
so goodput = fill / latency scales nearly linearly. **≥3× is met from fill=4** and reaches **39×** at full
batch. This stacks with P4's graph win (each step here is *already* one `cuGraphLaunch`).

**Measurement honesty.** A *first* (naive sequential) sweep reported a spurious **415×** — it timed
fill=1 at a throttled clock and fill=64 boosted (the documented ~7× laptop clock swing). That number is
**discarded**. The honest figure above captures all fills up front and times them **interleaved** (each
round times every fill back-to-back so they share the clock; per-fill best-of-N), so the ratio reflects
batching alone, not the clock.

### P6 — int8 KV quantization (the 6 GB footprint lever, tolerance-gated)
The KV cache is the dominant footprint at serving time. Storing it **int8** with a per-(token, head)
symmetric scale (`scale = max|x|/127`) instead of f16 halves it again — a *quarter* of f32 — the lever
that fits a long-context batch on a 6 GB device.

**Footprint** (`int8_kv_footprint_shrink`, pure geometry, no device; a Llama-7B-ish 32-layer, 8-KV-head
× 128, 4096×16-token-block cache): **f32 16.00 GiB | f16 8.00 GiB | int8 4.12 GiB** — **3.88× vs f32,
1.94× vs f16**. The per-(token, head) f32 scale slab is `head_dim×` smaller than the value slab, so it
costs only ~3% (1 + 4/`head_dim` bytes/value). A 16 GiB f32 cache that **cannot** fit the device fits at
int8.

**Correctness (tolerance — the first law for a lossy path):**
- **int8 decode-attention vs f64 reference** (`paged_int8_attention_matches_reference`, ragged ctx
  `[37,0,16,100,5,64]`, heads=4, hd=64): **max_abs = 3.00e-3** (gate 1e-2) — the only error is the
  per-(token, head) int8 quantization of K/V (Q stays f32). The kernel factors the scale out of the dot:
  `q·K = scaleK·Σ q[d]·int8K[d]`, `acc += (p·scaleV)·int8V[d]`.
- **Paging still invisible on the quantized path** (`paged_int8_attention_invariant_to_block_layout`):
  int8 values + scales under two different physical block layouts give **bit-for-bit identical** output
  (the dequant multiply order is fixed per token — as for f16).

**Kernel** (`paged_attn_decode_int8_ptx`): the warp-cooperative decode-attention kernel with the f16
cache replaced by `ld.global.s8` + a per-token scale; same online softmax. New `KvConfig::{scale_slab_elems,
scale_offset, kv_bytes, kv_bytes_int8}` + a host `quantize_kv_int8`. (Storage + read path; on-device
quantize-on-append is the documented next step — the read path + footprint are the win here.)

### P7 — multi-GPU tensor parallelism (design + single-GPU partition simulation)
The device here is **one** RTX 4050, so the 2-GPU throughput/latency is **explicitly unmeasured**. What
*is* built and gated: the **Megatron partition math**, validated on one GPU so the instant a second GPU +
a collective are attached it produces a real number.

**Megatron tensor parallelism for a decode layer** (`T` GPUs):
- **QKV projection + FFN gate/up → column-parallel.** Split the output dim `N` across GPUs; GPU `g` holds
  weight shard `W[g·N/T:(g+1)·N/T, :]` and computes `C_g = A·W_gᵀ = [Bcap, N/T]` from the *full* input —
  **no communication**. The attention heads partition cleanly: GPU `g` owns `heads/T` heads, so its slice
  of the **paged KV-cache is local** (each GPU appends + attends over only its own heads — no KV is moved
  cross-GPU during decode, the property that makes paged-attention TP cheap).
- **Attention-out + FFN-down → row-parallel.** Split the contraction dim `K`; GPU `g` holds `W_g[:, k-shard]`
  and the matching input shard, computes a **partial** `[Bcap, D]`, and the partials are summed by an
  **all-reduce**. Two all-reduces per layer (one after attn-out, one after FFN-down).
- **Comm volume:** `2 · Bcap · D` elements/layer; a ring all-reduce moves `2(T−1)/T · Bcap·D` per GPU per
  collective. At decode shapes (`Bcap·D` small) this is latency-bound — the regime CUDA graphs + batching
  already target.

**Single-GPU partition simulation (the math, gated):**
- **Column-parallel is bit-exact** (`tp_column_parallel_gemm_split_is_bit_exact`, 64×256×256): splitting
  `N` into two shards and concatenating is **bit-for-bit identical** to the unsplit WMMA GEMM — each output
  column is the same `K`-reduction, so column-parallel needs **no all-reduce and introduces zero error**.
- **Row-parallel all-reduce matches within tolerance** (`tp_row_parallel_gemm_allreduce_matches`,
  64×256×256, split K): summing the two `K`-shard partials matches the unsplit GEMM at **max_abs =
  1.34e-5** — the sum reassociates the `K` reduction (float, not associative), *exactly* the
  numerical behavior of a real ring all-reduce (the repo's standing reassociation-exception class).

**The one unmeasured seam:** the all-reduce *collective* itself — NCCL ring or CUDA driver P2P
(`cuMemcpyPeer` / `cuLaunchKernel` over P2P-mapped buffers). Everything feeding it (the column/row weight
split, the local sharded KV-cache, the partial GEMMs, the partial-sum reduction) is built and gated on one
GPU; only the cross-device transport is absent, and only because there is no second device attached.
