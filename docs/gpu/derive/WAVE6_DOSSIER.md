# WAVE 6 DERIVATION -- decode and serving on H100: the memory-bound regime

**Status: DERIVATION. Cost so far: $0.** Nothing here has executed on an H100. What it does is turn
the Wave-6 line of `ACT2_WAVE_PLAN.md` (`:168-178`) into text an implementer can write PTX and a
round manifest from, by deriving -- not assuming -- where a decode token's bytes actually go on a
132-SM part, and by refusing to carry a single one of this repo's serving ratios across the device
boundary.

**Provenance rules, inherited from this directory's README.** Every claim below is one of:

* **FACT(repo)** -- read out of this tree at a cited `file:line`, or measured in a cited round log.
  These outrank everything else.
* **FACT(ext)** -- an external document. Used only where the tree is silent, and **only to settle
  the SHAPE of a construction, never its size.**
* **DERIVED** -- arithmetic over the two above, reproducible by hand.
* **MODEL** -- arithmetic that needs a constant neither measured here nor in the tree. Every model
  below names its constant and its falsifier.
* **PREDICTION** -- falsifiable, and named as the thing a round exists to test.

---

> ## THE DEVICE-SCOPE RULE FOR THIS DOSSIER (read before quoting any number)
>
> **Every serving ratio this repository owns was measured on an RTX 4050 Laptop (sm_89, 20 SMs,
> 6 GiB, Windows/WDDM).** That includes `39.1x` / `85.6x` continuous-batching goodput, `1.07-1.35x`
> decode CUDA graph, `~6.5-6.9x` 12-layer graphed decode, `1.14-1.27x` continuous-vs-static
> scheduling, and every microsecond in `prompts/results/serving.md`. **`BENCHMARKS.md:51`** -- the
> standing index, not the results table -- already scopes them, and these are its words verbatim:
> those rounds *"predate grouped-query support, so their KV cache was sized for `q_heads`"*, and the
> internal ratios *"are **not** a Llama-class geometry"*. The results table carries a second scope
> note on the same row at `BENCHMARKS.md:366-377`, in *different* words ("before that, the cache was
> sized for `q_heads`, i.e. `g` times the real workload"); it scopes the Serving row to the 4050 but
> it is not the source of the two quotations above.
>
> **None of them appears in this dossier as an H100 expectation.** Where a 4050 number is quoted it
> is quoted as a *mechanism* -- "the step was launch-bound at depth 1 and compute-bound at depth 12"
> -- never as a size. Every H100 number below is derived from H100 facts only, and this is the
> **whole** list of them -- eight, in three classes, because a list that says "only" has to be
> closed:
>
> * **Probed on the device**, by the instrument's own provenance block: 132 SMs, 232,448 B opt-in
>   SMEM, 52,428,800 B L2, 85,017,624,576 B VRAM (section 1.1's first five rows).
> * **Measured in a round log**: the three bandwidths of `2026-08-11-h100-w2-r5-hbm.log:50-53`, and
>   `BW_L2 = 7.00 TB/s` -- which is measured rather than derived, but *indirectly*: it is what
>   `gpt_d1024_down`'s 597.6 TFLOP/s implies through `I_cta = 85.33` (`ACT2_WAVE_PLAN.md:5`, carried
>   as a fact by `WAVE3_DOSSIER.md:28`). It does exactly one job here -- 10.1's two ping-pong L2
>   roofs, restated in 10.5 and in section 13 -- and no byte budget divides by it.
> * **Not measured anywhere in this campaign**, and therefore quoted with that stamp on it every
>   time: the **989 TFLOP/s** dense-f16 tensor peak, which is a *spec-sheet* denominator
>   (`WAVE3_DOSSIER.md:30` sources it as the number that "reproduces the plan's 57.2% / 84.8%
>   peak-fractions exactly", i.e. it is what those fractions were taken against). Section 4.2 states
>   the band the repo's own measured f16 figure gives instead, and every claim that turns on it
>   carries both ends. Likewise the **256 KB unified L1/SMEM per SM** used once, in 5.4's
>   reuse-distance argument: **FACT(ext)**, the Hopper whitepaper's SM figure, of which the probed
>   232,448 B opt-in SMEM is the shared-memory share. Nothing is sized by either one; 5.4's
>   conclusion is a *distance* argument with a stated falsifier.
>
> Two H100 denominators this wave needs **do not exist yet**;
> section 1.4 names them and section 12.4 prices the measurement that creates them.

---

## 0. The one-paragraph answer

**The plan ranks Wave 6 by attention, and the byte budget says weights.** At the geometry the repo
itself prints for Llama-3-8B (`bench/gpu/h100/2026-08-10-h100-s2d-full-suite.log:2495`), one decode
step through a 32-layer stack reads **10.20 GB of f16 weights** and, at batch 16 / context 2048,
**4.29 GB of f16 KV** -- so the weight stream is **2.4x** the KV stream and is *independent of
batch*. Both terms are pure bytes: at the measured 2922.3 GB/s the whole step has a **4.96 ms
floor** and every mechanism in the wave is a fight over how close to it we land. The largest single
lever is therefore **not** a scheduling change at all -- it is Wave 5's dtypes applied to the two
byte streams: 8-bit weights **1.54x**, int4 weights **2.12x**, int8 KV **1.17x**, and int4 weights
with int8 KV **3.04x**, all of it arithmetic over byte counts with no scheduling assumption in it.
The second lever is that the decode GEMM launches **16 to 224 CTAs on 132 SMs** and cannot reach
that bandwidth: WAVE3's held-out **Stream-K trigger fires on six of six decode GEMMs**, which is the
promotion this dossier exists to record. The coalescing rewrite the plan ranks first is real but is
an **instruction/transaction** win of at most 8x on a term whose HBM bytes do not change, and
whether it converts into time is the one thing nobody has measured. And **ping-pong -- whose trigger
WAVE3 5.4 parked on exactly this row -- fires and still loses**, by a second independent route:
below M = 338 the decode GEMM's roof is `M x BW`, in which no tile shape appears.

---

## 1. THE INSTRUMENT: what is measured, on which device, and the two denominators that do not exist

### 1.1 H100 facts this wave may divide by

**FACT(repo)**, and the `source` column is the authority per row rather than any blanket sentence
over it -- **eight of these fifteen rows do not come from the instrument's provenance block.** The
first seven do: device through clock lock are the provenance lines the r3 round printed before it
timed anything. The four bandwidth rows come out of `hbm_bandwidth`'s own test output, in a
*different* log. And the last four are not device measurements at all: `BW_L2` is inferred from a
TFLOP/s through an `I_cta`, 989 TFLOP/s is a spec-sheet denominator, 256 KB is FACT(ext), and the
suite mean is a published table's summary. Read the row, not the lead-in:

| fact | value | source |
|---|---|---|
| device | NVIDIA H100 80GB HBM3, cc 9.0 | `bench/gpu/h100/2026-08-10-h100-act2-r3-bmulticast.log:517-518` |
| SMs | **132** | same log, `:519` |
| opt-in SMEM / CTA | 232,448 B | same log, `:521` |
| L2 | 52,428,800 B (**50.0 MiB**) | same log, `:522` |
| VRAM | 85,017,624,576 B (**79.18 GiB**) | same log, `:523` |
| sm clock | 1980 MHz, drift +0.00% | same log, `:529-531` |
| clock lock | **UNKNOWN (undeclared)** | same log, `:528` |
| HBM spec peak | 3352.3 GB/s | `bench/gpu/h100/2026-08-11-h100-w2-r5-hbm.log:50` |
| **copy (2N r+w), measured** | **2922.3 GB/s = 87.2% of peak** | same log, `:51` |
| saxpy (3N triad), measured | 2567.0 GB/s = 76.6% | same log, `:52` |
| reduce (1N read-only), measured | **682.1 GB/s = 20.3%** | same log, `:53` |
| `BW_L2`, measured indirectly (597.6 TFLOP/s through `I_cta = 85.33`) | 7.00 TB/s | `ACT2_WAVE_PLAN.md:5`, carried as a fact by `WAVE3_DOSSIER.md:28` |
| dense f16 tensor peak (campaign denominator) | 989 TFLOP/s | `WAVE3_DOSSIER.md:30` |
| unified L1/SMEM per SM (**FACT(ext)**, used once, in 5.4) | 256 KB | Hopper whitepaper; the probed 232,448 B opt-in SMEM is its SMEM share |
| suite mean vs cuBLAS, post-v2 | ~87.9% | `CAMPAIGN_CHECKPOINT.md:57` |

**`BW = 2922.3 GB/s` is the number every byte budget below divides by**, and the choice is
deliberate rather than convenient: it is the *only* measured H100 streaming figure that carries a
real write stream, which is what a decode step does (weights and KV in, activations out). It is
also, for a pure-read term, **conservative in the wrong direction and optimistic in the right one**
-- see 1.4.

### 1.2 The 4050 serving ledger, quarantined

**FACT(repo).** What the repo owns, with its device stamped on it:

| measured | value | device | source |
|---|---|---|---|
| continuous-batching goodput vs fill=1 | 39.1x @Bcap=64, 85.6x @Bcap=256 | **RTX 4050** | `prompts/results/serving.md:200-213`; `BENCHMARKS.md:395` |
| graph-driven scheduler vs static batching | 1.14-1.27x | **RTX 4050** | `BENCHMARKS.md:395` |
| decode CUDA graph, 1 / 6 / 12 layers | 1.34x / 1.35x / **1.07x** | **RTX 4050 (WDDM)** | `prompts/results/serving.md:139-143` |
| exposed launch overhead, 168 launches | "roughly fixed ~250-280 us" | **RTX 4050 (WDDM)** | same file, `:148` |
| 12-layer decode, eager -> graphed | ~6.5-6.9x | **RTX 4050 (WDDM)** | `BENCHMARKS.md:2490` |
| int8-KV footprint | **3.88x vs f32, 1.94x vs f16** | **device-free geometry** | `paged_kv.rs:846-878`, comment `:865` |
| int8 decode-attention vs f64 reference | max_abs 3.00e-3 (gate 1e-2) | 4050, **and green on H100** | `serving.md:233-236`; `s2d-full-suite.log:2501` |

**One row in that table is not device-scoped and it matters: the int8-KV footprint.** `3.88x` and
`1.94x` come out of `int8_kv_footprint_shrink`, a pure-geometry unit test with no device in it
(`paged_kv.rs:846`), and the underlying identity is `1 + 4/head_dim` bytes per value against 2. That
transfers to H100 unchanged, and section 6 is where it does real work. **Footprint is arithmetic;
throughput is not.** Every other row above stays on the 4050.

**One number in the ledger was internally inconsistent. It is now corrected at its own site, and
one site still carries the error.** `prompts/results/serving.md` described the 4050 as a **40**-SM
part in two places -- "only 8-32 CTAs on 40 SMs" and "39/40 SMs idle" -- where the device itself
reports **20**, and the probe print is checked into
the tree: `hbm_bandwidth` emits `theoretical peak HBM: 192.0 GB/s  (20 SMs)` straight out of
`g.sm_count()` (`gpu.rs:23493`), captured at
`bench/gpu/4050-identity/hbm_bandwidth.A.r1.txt:4` and in all seventeen sibling runs in that
directory. Two code sites agree with the probe -- `ptx_norm.rs:591` stamps its 4050 norm sweep
"20 SMs", `megakernel.rs:26` derives its idle fraction from "the 20-SM 4050" -- as does the L4
bring-up note, *"the **58-SM** L4 ... (vs the 4050's 20 SMs)"*
(`bench/gpu/l4/2026-08-09-session.md:112`). Both `serving.md` sentences now read the probed 20 and
cite that probe (`serving.md:149-152`, `:161-164`); the second one's *numerator* was wrong too, and
is re-derived there from the v1 launcher's own `PAGED_ATTN_BLOCK = 128` (4 CTAs, not 1).
**A third site is a Rust file and still says 40:** `paged_attention.rs:84`, "which left 39/40 SMs
idle" -- in the very file section 5 rewrites -- so it needs a commit that can carry the full
five-part gate rather than the docs-only one, and section 14 keeps it open. The
*mechanism* all three sentences describe -- a decode launch that covers a small fraction of the
machine -- is the one section 4 re-derives from scratch on 132 SMs, so nothing here rests on the
ledger's SM count either way.

### 1.3 The correctness floor is ALREADY GREEN on H100 -- the wave's best news, and it is free

**FACT(repo)**, `bench/gpu/h100/2026-08-10-h100-s2d-full-suite.log`. The whole serving stack ran on
Hopper and passed, in one container, at no extra cost:

| gate | line | what it proves on H100 |
|---|---|---|
| `paged_attention_matches_reference` | `:2280` | decode attention vs the f64 full-softmax reference |
| `paged_attention_invariant_to_block_layout` | `:2274` | **bit-for-bit** across two physical block layouts |
| `paged_gqa_attention_matches_reference` | `:2296` | the grouped-query path |
| `paged_gqa_attention_equals_mha_over_replicated_kv` | `:2286` | GQA == MHA over a replicated cache |
| `paged_gqa_attention_invariant_to_block_layout` | `:2288` | paging invariance survives GQA |
| `paged_int8_attention_matches_reference` | `:2501` | the int8-KV read path, tolerance |
| `paged_int8_attention_invariant_to_block_layout` | `:2408` | int8 paging invariance, bit-for-bit |
| `serving_decode_step_matches_reference` | `:2717` | the whole layer vs the f64 oracle |
| `serving_decode_step_invariant_to_block_layout` | `:2715` | whole-model paging invariance |
| `serving_decode_step_invariant_to_batch_composition` | `:2713` | co-batched sequences are invisible to each other |
| `serving_decode_graph_matches_eager` | `:2707` | the CUDA graph reproduces eager bit-for-bit |
| `serving_scheduler_graph_matches_eager` | `:2731` | the graphed scheduler ditto |
| `serving_gqa_scheduler_drains_and_conserves_blocks` | `:2721` | liveness + no block leak |
| `tp_column_parallel_gemm_split_is_bit_exact` | `:2739` | the Megatron column split |

**Every serving *performance* bench in the tree is `#[ignore]`d and has therefore never run on an
H100:** `serving_continuous_batching_goodput` (`:2494`), `serving_decode_graph_throughput`
(`:2497`), `decode_stack_latency` (`:171`), and all four megakernel benches (`:1645-1648`).

That asymmetry is the wave's whole situation in one sentence: **on H100 this stack is proven correct
and entirely unmeasured.** Wave 6 is not a bring-up. It is a measurement round with a small number
of derived levers attached.

### 1.4 The two denominators that do not exist

**(a) H100 pure-read bandwidth at a grid-sized launch.** The only read-only H100 number the repo
owns is `reduce (1N r): 682.1 GB/s` (`r5-hbm.log:53`), and it is **not a device bound** -- it is our
kernel's, at a grid of `RED_GRID = 256` CTAs of `RED_BLOCK = 256` threads (`gpu.rs:1302-1303`).
Copy and saxpy in the same test size their grids from the element count (`gpu.rs:1219-1221`,
`LaunchConfig::for_num_elems`) and land at 2922.3 and 2567.0 GB/s; the reduce is the one arm with a
fixed grid, and it is the one arm at 20.3%. 256 CTAs on 132 SMs is 1.94 CTAs/SM.

> **FACT(repo), and it changes what may be done about that.** `RED_GRID` is **not** a tuning
> constant and nothing in this tree ties its value to any device's SM count. It is a *determinism
> contract*, stated at its definition (`gpu.rs:1300-1301`) in the tree's own words: *"The grid is
> independent of input size and occupancy, so the GPU result is identical run-to-run (determinism by
> fixed decomposition, not associativity)."* The same constant sizes the **product** reduction under
> `--backend=gpu` -- `gpu::reduce` allocates `RED_GRID` partials and launches that grid
> (`gpu.rs:1361-1363`), reached from `wukong_driver::gpu_accel.rs:155` -- as well as the
> `hbm_bandwidth` bench's reduce arm (`gpu.rs:23538-23544`). **Changing `RED_GRID` therefore changes
> the offload path's reduction decomposition, not a bench knob**, and a size-dependent grid is
> exactly what that comment forbids. The measurement in `d1_readbw` is a *bench-local* grid; the
> constant, and the product reduce, stay as they are.

> **DERIVED, and it is the single most important sentence in section 1: the campaign has never
> measured H100 read bandwidth at a grid that covers the machine.** Decode is a pure-read workload.
> Every floor in this dossier is quoted at the copy figure, 2922.3 GB/s, and the honest band on any
> pure-read term is `[bytes/2922.3, bytes/682.1] GB/s` -- a factor of **4.28** -- until one launch
> closes it. Section 12.4 makes that launch row 1 of the round, and it costs a bench-local grid in
> `hbm_bandwidth`'s reduce arm -- **not** an edit to `RED_GRID`.

**(b) H100/Linux per-launch overhead.** The repo's only launch-overhead constant is 250-280 us for
168 launches (`serving.md:148`) = **1.49-1.67 us/launch on Windows/WDDM**. `BENCHMARKS.md:2464-2474`
already says out loud that those graph sizes "are WDDM sizes. Re-measure on the Linux driver before
quoting a multiple." Section 7 declines to use it and prices the replacement at seconds of H100
time.

---

## 2. THE STEP BUDGET -- where a decode token's bytes actually go

### 2.1 The geometry, taken from the repo's own H100 print

**FACT(repo)**, `bench/gpu/h100/2026-08-10-h100-s2d-full-suite.log:2495`, printed by
`serving::tests::llama3_8b_decode_geometry_is_expressible_at_a_quarter_of_the_mha_cache` **on the
H100**:

> `Llama-3-8B decode stack (32q/8kv x 128, 32L, D=4096 kv_dim=1024 Dff=14336, Bcap=64 x 8192 tok):`
> `Wk/Wv [1024, 4096] = 4x smaller than Wq; f16 KV cache 64.00 GiB with GQA vs 256.00 GiB without =`
> `exactly 4x`

So: `layers = 32`, `q_heads = 32`, `kv_heads = 8`, `head_dim = 128`, `g = 4`, `D = 4096`,
`kv_dim = 1024`, `dff = 14336`. The head geometries are pinned in `paged_kv.rs:934-940`
(`MODEL_GEOMETRIES`), and the six weight extents in `serving.rs:106-120`
(`decode_weight_extents`).

### 2.2 Weight traffic vs KV traffic -- the reordering

**DERIVED** from `decode_weight_extents` (`serving.rs:106-120`), f16 weights:

| weight | extent | elements | bytes |
|---|---|---|---|
| `wq` | `D x D` | 16,777,216 | 33,554,432 |
| `wk` | `kv_dim x D` | 4,194,304 | 8,388,608 |
| `wv` | `kv_dim x D` | 4,194,304 | 8,388,608 |
| `wo` | `D x D` | 16,777,216 | 33,554,432 |
| `w1` | `dff x D` | 58,720,256 | 117,440,512 |
| `w2` | `D x dff` | 58,720,256 | 117,440,512 |
| **per layer** | | **159,383,552** | **318,767,104 B = 318.77 MB** |
| **x 32 layers** | | | **10,200,547,328 B = 10.200 GB = 9.500 GiB** |

> Wukong's `DecodeLayer` has a **two-matrix** FFN (`w1` with a fused SiLU, then `w2`), not the
> three-matrix SwiGLU a stock Llama-3 ships. A real gate matrix adds 117.44 MB/layer = **+3.758 GB**
> per step, taking the weight term to **13.96 GB**. Every weight number below is Wukong's 10.20 GB;
> the three-matrix figure is stated wherever it changes a conclusion, and it never makes the weight
> term *smaller*.

**DERIVED**, KV traffic. `KvConfig::elem_offset` (`paged_kv.rs:139-151`) lays the cache out
`[layers, num_blocks, block_size, kv_heads, head_dim]` and **`KvConfig::heads` is the KV-head
count** (`paged_kv.rs:38-41`), so a decode step reads, per sequence:

```
    KV_bytes = 2 (K and V) * layers * ctx * kv_heads * head_dim * sizeof(elem)
             = 2 * 32 * ctx * 8 * 128 * 2
             = 131,072 * ctx      bytes per sequence per step
```

| batch B | ctx C | KV f16 | KV int8 (`x 0.515625`) |
|---|---|---|---|
| 16 | 2048 | 4,294,967,296 B = **4.295 GB** | 2,214,592,512 B = 2.215 GB |
| 64 | 2048 | 17,179,869,184 B = **17.180 GB** | 8,858,370,048 B = 8.858 GB |
| 64 | 8192 | 68,719,476,736 B = **68.719 GB = 64.00 GiB** | 35,433,480,192 B = 33.00 GiB |

The `64.00 GiB` row reproduces the repo's own H100 print to the digit, which is the check that this
formula is the tree's formula and not a re-invention.

**THE REORDERING.** At the plan's own reference point -- batch 16, ctx 2048, 32 layers -- the two
terms are **10.20 GB of weights against 4.295 GB of KV**. The weight stream is **2.37x** the KV
stream (3.25x with a real SwiGLU gate) and, unlike KV, it does not grow with batch. The crossover is

```
    B* = weight_bytes / KV_bytes_per_sequence = 10,200,547,328 / 268,435,456 = 38.0 sequences
```

so **KV only becomes the dominant term above batch 38** at ctx 2048 (above batch 9.5 at ctx 8192).
`ACT2_WAVE_PLAN.md:174` sizes Wave 6 entirely off the KV term and never states the weight term.

### 2.3 The step-floor table

**DERIVED**, `T_floor = (weight_bytes + KV_bytes) / 2922.3 GB/s`. This is a **bandwidth-only floor**:
zero compute, zero launch overhead, zero inefficiency, perfect overlap. It is an upper bound on
goodput and a lower bound on latency, and nothing in Wave 6 can beat it.

**Batch 16, ctx 2048** (KV f16 4.295 GB / int8 2.215 GB):

| weight dtype | weight bytes | KV f16 | KV int8 |
|---|---|---|---|
| f16 | 10.200 GB | **4.960 ms** (1.000x) | 4.248 ms (**1.168x**) |
| 8-bit (fp8 / int8, W5) | 5.100 GB | 3.215 ms (**1.543x**) | 2.503 ms (**1.982x**) |
| int4 (W4A16, `ptx_int4.rs`) | 2.550 GB | 2.342 ms (**2.118x**) | **1.631 ms (3.042x)** |

**Batch 64, ctx 2048** (KV f16 17.180 GB / int8 8.858 GB):

| weight dtype | KV f16 | KV int8 |
|---|---|---|
| f16 | **9.370 ms** (1.000x) | 6.522 ms (**1.437x**) |
| 8-bit | 7.624 ms (1.229x) | 4.777 ms (1.962x) |
| int4 | 6.752 ms (1.388x) | **3.904 ms (2.400x)** |

Goodput ceilings (`B / T_floor`), for scale only -- **these are floors on a 132-SM part and are not
comparable to any 4050 tok/s figure in this repo**:

| configuration | tok/s ceiling |
|---|---|
| B=16, f16 W / f16 KV | 3,225 |
| B=16, int4 W / int8 KV | 9,813 |
| B=64, f16 W / f16 KV | 6,831 |
| B=64, int4 W / int8 KV | 16,394 |

### 2.4 Capacity: what 79.18 GiB actually holds

**DERIVED** from two probed numbers -- VRAM `85,017,624,576 B = 79.18 GiB` (r3 log `:523`) and the
`64.00 GiB` f16 KV cache the s2d suite prints for `Bcap=64 x 8192 tok`:

| configuration | KV | weights (3-matrix Llama-3-8B, f16) | total | fits 79.18 GiB? |
|---|---|---|---|---|
| Bcap 64 x 8192, **f16 KV** | 64.00 GiB | 13.00 GiB | **77.00 GiB** | 2.18 GiB spare -- **no**, in practice |
| Bcap 64 x 8192, **int8 KV** | 33.00 GiB | 13.00 GiB | **46.00 GiB** | 33.2 GiB spare -- **yes** |
| Bcap 128 x 8192, int8 KV | 66.00 GiB | 13.00 GiB | 79.00 GiB | 0.18 GiB -- **no** |
| Bcap 128 x 4096, int8 KV | 33.00 GiB | 13.00 GiB | 46.00 GiB | **yes** |

> **DERIVED: on one H100, int8 KV is not a throughput lever, it is a *feasibility* lever.** At
> `Bcap = 64 x 8192` the f16 cache plus the weights leave 2.18 GiB for activations, the
> `DevicePool`, the block table and every fragment -- which is to say it does not fit. The int8 path
> makes the same configuration fit with 33 GiB to spare. This is arithmetic over two probed device
> facts, it has no ratio in it, and it is the only Wave-6 claim that needs no measurement at all.

### 2.5 What section 2 concludes

1. **The wave is a byte-count problem.** Two streams, weights and KV, and the levers that move them
   are dtype (section 3) and enough parallelism to reach bandwidth (section 4).
2. **The plan's ranking inverts the terms.** Decode attention is 29.6% of the step at batch 16 and
   62.7% at batch 64; weights are the rest. A 2x on attention is a 1.17x on the step at batch 16.
   Section 9.2 turns that into a refusal.
3. **The plan's own decode floor is 15% optimistic.** `ACT2_WAVE_PLAN.md:174` reads "4.29 GB/step =
   1.28 ms at 3.35 TB/s". 3.35 TB/s is the *spec sheet*; the measured figure is 2922.3 GB/s, which
   makes the same 4.29 GB **1.470 ms**. Divide by what was measured.

---

## 3. MECHANISM 1 -- weight and KV bytes: Wave 5's dtypes are Wave 6's biggest lever

**DERIVED.** Section 2.3's table *is* the derivation; this section is about what it costs to
collect.

| arm | what already exists | what Wave 6 must add | derived effect (B=16 / B=64) |
|---|---|---|---|
| **int8 KV** | the whole read path: `paged_attn_decode_int8_ptx` (`paged_attention.rs:408`), `kv_append_int8_ptx` (`:962`), `KvDtype::Int8` through `DecodeModel::new_with_dtype`, all four gates green on H100 | **nothing.** Measure it. | **1.168x / 1.437x** |
| **8-bit weights** | `ptx_int8.rs` (2344 lines), `ptx_fp8.rs` (1823) at `mma.sync`; W5 retypes them onto `wgmma` | a decode call site: the projections must route to an 8-bit GEMM. `serving.rs:339-341` hardcodes three f16 WMMA entries | **1.543x / 1.229x** |
| **int4 weights (W4A16)** | `ptx_int4.rs` (878 lines), `w4a16_ptx`, in the device-free module set | same call-site problem, plus a weight pre-pack | **2.118x / 1.388x** |
| **int4 W + int8 KV** | both halves above | both | **3.042x / 2.400x** |

Three things this table says that the plan does not:

1. **The int8-KV arm is already built and already correct on the target device.** It is a
   measurement, not an implementation. It should be the *first* row of the round, because it is the
   only lever in the wave whose engineering cost is zero.
2. **8-bit and int4 weights are the largest levers and neither has a decode call site.** This is the
   exact defect `ACT2_WAVE_PLAN.md:33-34` records for Wave 4 (`gemm_nt_wgmma` has no product call
   site) reappearing one wave later on a different axis: the kernels exist, the serving path calls
   `wmma_nt_f16_sm_db` unconditionally (`serving.rs:339`), and no gate can see the gap.
3. **Wave 5 is a prerequisite for Wave 6's headline, not a sibling.** `WAVE5_DOSSIER.md` derives
   that the 8-bit `wgmma` descriptor transfers byte-for-byte at `bk * dtype.size() == 128`; the
   value of that transfer, for serving, is section 2.3's `1.543x` column.

**REFUSAL, stated in advance.** A quantized-weight decode row may only be published against an
**equal-batch, equal-context f16 control taken in the same container**, and the quantization error
must be reported beside the ratio. Halving the weight bytes halves the time by construction; the
finding is not that it is faster, it is *how much accuracy that costs at a real model's weights*,
and this repo has no such measurement at decode.

---

## 4. MECHANISM 2 -- the decode GEMM is under-parallel, and this is where Stream-K fires

### 4.1 What the decode step launches today

**FACT(repo).** `DecodeLayer` loads three WMMA entries -- `wmma_nt_f16_sm_db`, `_silu`, `_residual`
(`serving.rs:338-341`) -- and launches them through `wmma_sm_cfg` (`serving.rs:85-93`):

```
    grid  = (N / SM_BN, M / SM_BM, 1)
    block = (SM_THREADS, 1, 1)
```

with `SM_BM = SM_BN = 64`, `SM_WARPS_M = SM_WARPS_N = 2`, `SM_THREADS = 128`
(`ptx_wmma.rs:167-173`). **The decode path does not touch `wgmma` at all** -- so Waves 3, 4 and 5's
entire body of work reaches serving only through a wiring change that nobody has scheduled.

**DERIVED**, CTA counts per layer at the Llama-3-8B geometry:

| GEMM | M | N | K | CTAs at Bcap=64 | % of 132 SMs | CTAs at Bcap=256 |
|---|---|---|---|---|---|---|
| `wq` | Bcap | 4096 | 4096 | **64** | 48.5% | 256 |
| `wk` | Bcap | 1024 | 4096 | **16** | **12.1%** | 64 |
| `wv` | Bcap | 1024 | 4096 | **16** | **12.1%** | 64 |
| `wo` | Bcap | 4096 | 4096 | **64** | 48.5% | 256 |
| `w1` (SiLU) | Bcap | 14336 | 4096 | **224** | 1.70 waves (84.8% eff.) | 896 |
| `w2` (residual) | Bcap | 4096 | 14336 | **64** | 48.5% | 256 |

At `Bcap = 64`, **four of the six launches cover under half the machine and two cover an eighth.**

### 4.2 The roof, and the number that kills every scheduling lever below it

**DERIVED.** For `C[M,N] = A[M,K] * B[N,K]^T` in f16 with an f32 C:

```
    FLOP  = 2*M*N*K
    bytes = 2*N*K (weights) + 2*M*K (activations) + 4*M*N (output)
    I     = FLOP / bytes  ->  I ~ M  when M << K and M << N
    roof  = I * BW
```

| M | I (FLOP/B) | roof at 2922.3 GB/s | % of 989 TFLOP/s peak |
|---|---|---|---|
| 64 | 61.13 | **178.6 TFLOP/s** | **18.1%** |
| 256 | 215.6 | **630.0 TFLOP/s** | 63.7% |
| 338.4 | 338.4 | 989 TFLOP/s | 100% |

> **DERIVED, and it is the load-bearing statement of section 4: below `M = 338` the decode
> projection GEMM's ceiling is `M x BW_HBM`, a function of the batch and the bandwidth alone. No
> tile shape, schedule, raster, cluster or epilogue appears in it.** At `Bcap = 64` the ceiling is
> 18.1% of the tensor peak; the tensor cores are idle five sixths of the time *by arithmetic*, and
> that is correct behaviour, not a defect. The only two things that move it are fewer weight bytes
> (section 3) and more rows per weight read (a bigger batch).

The crossover `M* = 989e12 / 2.9223e12 = 338.4` is worth remembering as the campaign's decode
constant: **a decode batch is "small" below 338 rows on this device and "large" above it**, and
every H100 serving configuration this wave will run is small.

### 4.3 Reaching the roof: the memory-level-parallelism gap

The roof above assumes the launch can *sustain* 2922.3 GB/s. Sixteen CTAs cannot.

**MODEL** -- Little's law, with both constants named. To sustain `BW` at HBM latency `L` a kernel
must keep `BW * L` bytes in flight. Taking `L = 600 ns` (**an external constant, not measured
here**) gives `2922.3e9 * 600e-9 = 1.753 MB`. A CTA of 128 threads issuing 16-byte loads with `q`
outstanding each carries `128 * 16 * q` bytes; at `q = 8`, 16,384 B per CTA.

| launch | CTAs | bytes in flight | fraction of 1.753 MB |
|---|---|---|---|
| `wk` / `wv` at Bcap=64 | 16 | 0.262 MB | **15%** |
| `wq` / `wo` / `w2` at Bcap=64 | 64 | 1.049 MB | 60% |
| `w1` at Bcap=64 | 224 | 3.67 MB | saturating |
| `wq` at Bcap=256 | 256 | 4.19 MB | saturating |

**Falsifier, and it is one number:** measure the achieved bandwidth of a single `wk`-shaped decode
GEMM launch on H100 (`M=64, N=1024, K=4096`, 8.39 MB of weights). If it reads at ~2900 GB/s the
model is wrong and section 4 collapses to "nothing to do"; if it reads at ~450 GB/s the model is
right and the prize is up to **6.6x on that launch**. There is no third answer, and the measurement
is one `#[ignore]`d bench away.

### 4.4 The two ways to add CTAs, and only one of them is free

**(a) Merged QKV -- bit-exact by construction, and it costs a host-side concat.** `wq`, `wk` and
`wv` consume the same `[Bcap, D]` activation and differ only in their weight rows. Concatenating
them into one `[D + 2*kv_dim, D] = [6144, 4096]` weight makes one launch of
`6144/64 = 96` CTAs where there were three launches of 64 + 16 + 16.

*Why it is bit-exact*: each output column is the same `K`-reduction over the same operands in the
same order; only the CTA that computes it changes. That is precisely the argument
`tp_column_parallel_gemm_split_is_bit_exact` (`serving.rs:3401`, green on H100 at
`s2d-full-suite.log:2739`) already proves in the other direction -- splitting `N` and concatenating
is bit-identical to the unsplit GEMM, so concatenating `N` and splitting the result is too.

Effect: 3 launches -> 1 (saving 64 launches per step at 32 layers), and 3 partial waves -> 1 fuller
wave. **DERIVED and free of any numerical risk.** The cost is that `kv_append` must read K and V at
offsets into the merged output rather than from two buffers -- pointer arithmetic, no new kernel.

**(b) Split-K / Stream-K -- the trigger WAVE3 5.3 deferred, now firing.** See section 10.2. It is
the larger lever and it is the one with a numerical precondition.

---

## 5. MECHANISM 3 -- paged decode attention: coalescing and the g-fold

### 5.1 What the kernel emits today

**FACT(repo)**, `paged_attention.rs`. One warp per `(slot, q_head)`, `PAGED_ATTN_WARPS = 4` pairs
per CTA (`:85`); lane `i` owns positions `{t : t % 32 == i}` (`:189`, `:218`); the query is staged
once per warp in shared memory (`:146`, `:177-183`). The inner loop is scalar, twice:

```
    :204   ld.shared.f32 %qv,[%qshw+4d];  ld.global.u16 %h,[%kbase+2d];  cvt.f32.f16 %kf,%h;  fma ...
    :216   ld.global.u16 %h,[%vbase+2d];  cvt.f32.f16 %vf,%h;  mul %acc{d},%acc{d},%corr;  fma ...
```

unrolled `head_dim` times each. The int8 twin is the same shape with `ld.global.s8`
(`:497`, `:510`).

### 5.2 The sector arithmetic

**DERIVED.** At a single `ld.global.u16`, the warp's 32 lanes are reading element `d` of 32
*different* context positions. Consecutive positions inside one block are `kv_heads * head_dim`
elements apart -- at the Llama-3-8B geometry `8 * 128 * 2 = 2048 B` -- and the 16-token block size
(`paged_kv.rs:63`, `DEFAULT_BLOCK_SIZE = 16`) puts lanes 16..31 in a different physical block
entirely. So:

| quantity, per warp instruction | today (`u16`) | with `ld.global.v4.b32` (16 B) |
|---|---|---|
| distinct 32-B sectors touched | 32 | 32 |
| bytes consumed per sector | **2 of 32 = 6.25%** | **16 of 32 = 50%** |
| load instructions per position (K + V, hd=128) | **256** | **32** |
| sector requests per position per warp | **8192** | **1024** |

**8x fewer memory instructions and 8x fewer sector requests per useful byte.** Two consecutive `v4`
loads land in the same 32-byte sector, so a paired emission reaches 100% sector efficiency and the
same 8x.

> **DERIVED, and this is where this dossier parts company with the plan: the `v4` rewrite does not
> change a single byte of HBM traffic.** Over the whole `d` loop a lane reads its position's 256
> contiguous bytes exactly once either way; the unique-byte count is section 2.2's formula, which
> already has no `q_heads` and no vector width in it. What changes is **issue rate and L1 tag
> pressure**. `ACT2_WAVE_PLAN.md:174` sizes this lever at "1.6-2.0x on the decode-attention term"
> from FlashInfer's and QuACK's published bandwidth-utilisation figures. That is literature sizing a
> lever, which this directory's provenance rule forbids: **literature may settle the shape of a
> construction, never its size.** The derived statement is a bound -- at most 8x, at least 1x -- and
> the discriminator is whether the kernel is LSU-bound or HBM-bound, which nobody has measured on
> any device.

**PREDICTION, falsifiable in one arm:** run the existing scalar kernel and a `v4` twin back to back
at `(Bcap=64, ctx=2048, 32q/8kv x 128)` on H100. If the ratio is inside the contender's dispersion
the kernel was already HBM-bound and the lever is dead; if it is above ~2x the kernel was
issue-bound and the plan's band was right for the wrong reason.

### 5.3 `head_dim % 8` -- the alignment law the assert does not carry

**FACT(repo).** `paged_attn_decode_ptx` asserts only
`head_dim > 0 && head_dim.is_multiple_of(2)` (`paged_attention.rs:108-111`); the int8 twin asserts
the same (`:409-412`).

**DERIVED.** A `(token, kv_head)` vector starts at element index `es * head_dim`, i.e. byte offset
`es * head_dim * sizeof(elem)` (`elem_offset`, `paged_kv.rs:139-151`). A 16-byte `ld.global.v4.b32`
requires that offset to be 16-byte aligned:

| cache dtype | condition | head_dim 64 | head_dim 128 |
|---|---|---|---|
| f16 (2 B) | `head_dim % 8 == 0` | ok | ok |
| int8 (1 B) | `head_dim % 16 == 0` | ok | ok |

Both shipped head dims clear both rules, which is exactly why this is dangerous: **the assert would
never fire in the corpus and would let a `head_dim = 12` or `head_dim = 40` model emit a kernel that
mis-addresses silently.** The law belongs in the generator, per dtype, in the same commit as the
`v4` arm (G-W6-2).

### 5.4 The g-fold: request amplification, not HBM amplification

**FACT(repo).** The warp-to-work map is `gid = ctaid * 4 + warp`, `slot = gid / q_heads`,
`head = gid % q_heads`, `kvhead = head / g` with `g = q_heads / kv_heads`
(`paged_attention.rs:164-171`).

**DERIVED.** With `PAGED_ATTN_WARPS = 4`, CTA `c` owns four *consecutive* query heads. So:

| model | q / kv | g | which warps share a KV head | where the reuse is served |
|---|---|---|---|---|
| Llama-3-8B | 32 / 8 | 4 | **all 4 warps of one CTA** | **L1** (same SM, launched together, 256 B apart) |
| Llama-3-70B | 64 / 8 | 8 | 2 CTAs | **L2** (50 MiB, both CTAs streaming in step) |
| MQA-32x1 | 32 / 1 | 32 | 8 CTAs | L2 |
| GPT-2 (MHA) | 12 / 12 | 1 | none | n/a |

The reuse *distance* is small in every row: the sharing warps run the same instruction stream over
the same positions, so the bytes one re-reads are the bytes another read a few hundred cycles ago --
256 B per position, 1 KB per CTA at `g = 4`, against a 256 KB unified L1/SMEM per SM -- and that
capacity is the one **FACT(ext)** in this section (the Hopper whitepaper's SM figure; the probed
232,448 B opt-in SMEM is its shared-memory share, `r3-bmulticast.log:521`). It is used for a
*distance*, not a size: 1 KB against 256 KB is two orders of margin, so the conclusion survives any
plausible restatement of the capacity, and the falsifier below measures the traffic directly rather
than trusting either number.

> **DERIVED, and it is a correction to the plan.** `ACT2_WAVE_PLAN.md:174` says GQA "fixes a hard
> g-fold read amplification (4x at Llama-3-8B, 8x at 70B) on the single dominant traffic term:
> batch 16 / L=2048 / 32 layers requires 4.29 GB/step ...; today we ask for 4x that." **The 4.29 GB
> is already the g-folded number** -- it is computed with `kv_heads = 8`, and section 2.2 reproduces
> it from `elem_offset`'s KV-headed layout to the digit. Asking for "4x that" would require every
> redundant read to miss both L1 and L2, which the reuse-distance argument above says it does not.
> The g-fold is a **request** amplification of `g`, absorbed by the cache hierarchy at an unmeasured
> efficiency; it is not 17.2 GB of HBM.

**Falsifier, and it is the cheapest high-value measurement in the wave:** read the decode kernel's
DRAM traffic for one step and divide by `2 * layers * ctx * kv_heads * head_dim * 2 * B`. A ratio
near **1.0** retires the plan's 4x and drops the tiled q-group to rank 5; a ratio near **g**
confirms the plan and promotes it to rank 1. Both outcomes are publishable; the current text is not.

### 5.5 The tiled q-group: the K side is free, the V side is a 512-register wall

**DERIVED.** Give a warp `(slot, kv_head)` and a tile of `g` query heads. The two halves of the
kernel behave completely differently:

**(a) The K side is free.** One K vector produces `g` scores, so the fold needs `g` extra f32
registers and nothing else. At `g = 4` that is 4 registers against the ~198 the kernel already
declares (`.reg .f32 %acc<128>` at `:147` plus ~70 scalars at `:148-152`). It removes `g - 1` of
every `g` K loads outright.

**(b) The V side does not fit.** Accumulating `g` output rows needs `g * head_dim` f32
accumulators per lane:

```
    g = 4, head_dim = 128  ->  512 registers/thread   against a 255 hard cap   DOES NOT FIT
    g = 8, head_dim = 128  ->  1024                                            DOES NOT FIT
    g = 2, head_dim = 64   ->  128 + ~70 = ~198                                fits
```

`ACT2_WAVE_PLAN.md:176` names this ("g=4 at d=128 is 512 registers -- impossible without tiling")
and calls it a design step. **The derivation says the design step has only bad options at
`head_dim = 128`:** chunking the accumulator over `head_dim` re-reads V once per chunk, which
re-introduces exactly the traffic the fold removed; buffering the scores instead needs `ctx` floats
of SMEM, which is the shared-memory logits buffer the module deliberately does not have
(`paged_attention.rs:19-20` -- its absence is why arbitrary context works with no v2-style split).

> **DERIVED, the honest scope: implement the K-side fold and stop.** It is `g` registers, it halves
> the load instruction count on its half of the traffic, it composes with the `v4` rewrite, and it
> keeps the lane partition untouched -- which section 5.6 shows is the whole correctness argument.
> The V-side fold is a different kernel with a different lane partition and a different reduction
> order, and it should be scoped as its own wave item if and only if the section-5.4 falsifier comes
> back near `g`.

### 5.6 Which rewrites keep the bit-exactness invariant

**FACT(repo)**, `paged_attention.rs:37-47`, the module's own load-bearing paragraph. Layout
invariance rests on exactly two facts: **(1)** the lane partition is a function of the *logical
position index alone* (`t % 32`), and **(2)** the cross-lane merge is a *fixed* butterfly. The doc
already names the trap: partitioning by *physical block* "still satisfies 'one fixed accumulation
order per lane'" and **breaks** the property.

**DERIVED**, applied to each Wave-6 rewrite:

| rewrite | partition still position-keyed? | merge order unchanged? | `*_invariant_to_block_layout` survives? | values bit-identical to today? |
|---|---|---|---|---|
| `v4` loads, same `t % 32` partition | **yes** | yes | **yes** | **yes** -- same values, same order |
| K-side g-fold (warp owns `(slot, kv_head)`) | **yes** | yes | **yes** | **yes** |
| V-side g-fold with a head_dim-split partition | yes (a lane owns output *elements*, positions walked serially) | **no -- different tree** | yes | **NO** -- re-derive the f64 tolerance |
| partition by physical block | **NO** | -- | **NO** | -- |
| int8 KV (shipped) | yes | yes | yes (gate green on H100) | n/a (lossy path) |

**The `v4` and K-fold arms are bit-identical to the shipped kernel**, which means the round can gate
them with `==` against the f16 kernel's own output rather than against a tolerance -- a strictly
stronger gate than the one the module ships with, and free.

---

## 6. MECHANISM 4 -- int8 KV on H100

**FACT(repo).** Everything exists and is green on the target device: the read kernel
(`paged_attn_decode_int8_ptx`, `paged_attention.rs:408`), the masked append
(`kv_append_int8_ptx`, `:962`), the host quantizer (`quantize_kv_int8`, `:692`), the GQA launchers
(`:587`, `:1169`), the footprint geometry (`paged_kv.rs:846-878`), and four H100-green gates
(`s2d-full-suite.log:2408`, `:2501`, `:2316`, `:2406`).

**DERIVED**, the three numbers, each labelled with what kind of claim it is:

| claim | value | kind |
|---|---|---|
| footprint vs f16 / vs f32 | **1.94x / 3.88x** | **device-free arithmetic** -- `1 + 4/head_dim` bytes per value against 2 or 4 |
| step-floor speedup at B=16, ctx=2048, f16 weights | **1.168x** | derived over H100 bytes and the measured 2922.3 GB/s |
| step-floor speedup at B=64, ctx=2048, f16 weights | **1.437x** | same |
| makes `Bcap=64 x 8192` fit 79.18 GiB | 77.00 -> 46.00 GiB | derived over two probed device facts |
| accuracy | max_abs **3.00e-3** vs the f64 reference, gate 1e-2 | measured on 4050 (`serving.md:229`), **gate re-run green on H100** |

**The scale slab is the part that is easy to get wrong and is already right.** One f32 per
`(token, kv_head)`, `head_dim` times smaller than the value slab (`paged_kv.rs:153-158`,
`scale_slab_elems`), and the kernel factors it out of the dot: `q.K = scaleK * sum q[d]*int8K[d]`,
`acc += (p*scaleV)*int8V[d]` (`paged_attention.rs:499`, `:508`). One multiply per position, not per
element.

**REFUSAL.** The int8-KV row is published **with its accuracy column or not at all**, and the
accuracy must be re-measured at the *serving* geometry (32 layers, real weights, ctx 2048+), not
inherited from the 4-head / hd-64 / ctx-100 unit gate. A 3e-3 per-layer error compounded through 32
layers is not a 3e-3 model error and nothing in this tree has measured what it is.

---

## 7. MECHANISM 5 -- the CUDA-graph decode loop, and a constant we do not own

**FACT(repo).** `graph.rs:1-30` captures a whole decode step into one `cuGraphLaunch`; the
`DevicePool` is its prerequisite (a synchronizing `cuMemAlloc` inside a capture is rejected); capture
runs on a dedicated non-blocking stream with event tracking disabled. `serving_decode_graph_matches_eager`
and `serving_scheduler_graph_matches_eager` are **green on H100** (`s2d-full-suite.log:2707`,
`:2731`).

**DERIVED**, the launch count: `serving.md:141-143` records ~14 launches per layer, so a 32-layer
decode step issues **~448 launches** eager and **1** graphed.

**The prize cannot be sized.** Multiply 448 by a per-launch overhead and the answer spans the whole
interesting range:

| per-launch overhead | 448 launches | as a fraction of the 4.960 ms step floor (B=16) |
|---|---|---|
| 1.5 us (the repo's WDDM figure, `serving.md:148`) | 0.672 ms | **13.5%** |
| 0.5 us | 0.224 ms | 4.5% |
| 0.3 us | 0.134 ms | 2.7% |

> **REFUSAL: the repo's 1.49-1.67 us/launch is a Windows/WDDM constant and may not be carried to a
> Linux H100.** `BENCHMARKS.md:2464-2474` already says so about the graph ratios that constant
> produced. `ACT2_WAVE_PLAN.md:174`'s "~1.3 us of dead time per kernel even under graphs (320
> launches/token-step = ~0.42 ms)" is FACT(ext) from the megakernel literature and is sizing, not
> shape -- decline it too.

**The replacement costs seconds.** Time `N` launches of an empty kernel eager against one graph
replay of the same `N`, on the H100, at `N = 448`. That is one arm, no peer, no model, and it turns
sections 7 and 9.4 from a range into a number. It is row 2 of the round.

---

## 8. MECHANISM 6 -- norms: the serving path bypasses its own planner

**FACT(repo), and it is a defect rather than a lever.** `ptx_norm.rs` ships an SM-filling entry
family `{softmax,layernorm,rmsnorm}_w{W}` for `W` in `{1,2,4,8}` with a grid-stride row loop, and a
planner `norm_launch(base_entry, sm_count, rows, cols)` (`ptx_norm.rs:667`) that sizes `W` and the
grid from the **probed** SM count. `DecodeLayer` uses none of it:

```
    serving.rs:336    let f_norm = g.function("norm", crate::ptx_norm::norm_ptx(), "rmsnorm")?;
    serving.rs:435    let norm_cfg = LaunchConfig { grid_dim: (bcap as u32, 1, 1),
    serving.rs:437                                  block_dim: (32, 1, 1), shared_mem_bytes: 0 };
```

It names the **shipped one-warp entry** and hardcodes `grid = (Bcap, 1, 1)`, `block = 32`.

**DERIVED.** At `Bcap = 64` on 132 SMs that launch is **64 warps = 2048 threads**, one warp on each
of 48.5% of the SMs and nothing on the rest. The planner, given the same shape, would choose
`W = min(prev_pow2(cols / (32 * MIN_ELEMS_PER_LANE)), 8) = min(prev_pow2(4096/512), 8) = 8`
(`ptx_norm.rs:70`, `:619-629`) and a grid-strided launch sized from `sm_count` -- because `rows = 64`
is far below `FILL_CTAS_PER_SM * sm_count = 4 * 132 = 528` (`:75`), i.e. **the decode norm sits deep
inside the underfill regime the `_w{W}` family was built for.**

**DERIVED**, what it is worth. RMSNorm reads each row twice (sum-of-squares pass at
`ptx_norm.rs:229`, normalize+store at `:235`) and writes once: 12 B per element.

```
    per layer  = 2 norms * Bcap * D * 12 = 2 * 64 * 4096 * 12 = 6,291,456 B
    per step   = x 32 layers             = 201,326,592 B = 201.3 MB
    share of the 14.495 GB step          = 1.389%
    at BW                                = 0.0689 ms of a 4.960 ms step
```

So the norm is **1.39% of the step if it runs at bandwidth**, and up to ~11% if the underfilled
launch runs 8x off it. The 4050 measured **1.6-3.5x** available from `W > 1` in exactly this regime
(`ptx_norm.rs:600-607`) -- quoted here as the *existence* of the gap on a 20-SM part, not as its
size on a 132-SM one, where `ptx_norm.rs:617-618` says out loud that "the covered regime is
genuinely open there".

> **The fix is three lines** -- route `DecodeLayer`'s norm through `norm_launch(g.sm_count(), rows,
> d)` and use the entry and grid it returns. It is worth **0 to ~9 points of the decode step** and
> it is the cheapest engineering in the wave. It is also the correct place to put the plan's
> "`MIN_ELEMS_PER_LANE` and `warps_per_row` re-derived on H100, never ported from the 20-SM 4050"
> demand (`ACT2_WAVE_PLAN.md:176`): the knob for that round already exists as `WUKONG_NORM_WARPS`
> (`ptx_norm.rs:631-636`), and it **panics** on a value outside `MW_WIDTHS` rather than silently
> measuring the default.

**REFUSAL.** `ACT2_WAVE_PLAN.md:174` proposes a `norm_vs_peers` bench against QuACK and
torch.compile. `norm_vs_peers` **does not exist** anywhere in the tree (grep: one hit, the plan line
itself), QuACK is **not staged** on the Volume (`peers.json` records CUTLASS, FA2, vLLM, torch,
ptxas -- `2026-08-11-fa2-stage.log`), and the plan's own target list already declares norms
explicitly NOT a headline (`:205`). **Build the three-line planner fix and the internal A/B; do not
build a peer bench for a 1.4%-of-step term this wave.**

---

## 9. THE PEER BAR

### 9.1 What is staged, and it is the good news

**FACT(repo)**, `bench/gpu/h100/2026-08-11-fa2-stage.log` (the `peers.json` audit block) --
everything Wave 6 needs is already on the `wukong-build` Volume and **already paid for**:

| peer | version | evidence |
|---|---|---|
| **flash-attn** | **2.8.3.post1**, archs 90, `kvcache_api=true`, `kvcache_symbol=true`, so 208.4 MB | the audit block; `_verify_fa_wheel` (`modal_app.py:1356-1413`) proves `fwd_kvcache` is in the extension's own bytes |
| **vLLM** | **0.26.0**, venv mode `symlinks`, `/persist/vllm-venv/bin/python` | same block |
| torch | **2.13.0+cu129** | same block |
| CUTLASS profiler | v4.6.1 sm90a, f16/fp8/int8 368/782/308 | `CAMPAIGN_CHECKPOINT.md:113-115` |
| ptxas | 12.9.86, archs sm_80/89/90/90a | same block |

`_verify_fa_wheel` is worth one sentence of respect: it exists because a FlashAttention wheel built
with `DISABLE_PAGEDKV` "imports and then declines the only call the round makes"
(`modal_app.py:1305-1312`), and it refuses to record such a wheel. **The decode bar is staged and
audited; the round does not need a build step.**

### 9.2 FA2 with kvcache: the honest scope, stated before the measurement

`flash_attn_with_kvcache(block_table=)` measures **the decode-attention term only**. Section 2.2
prices that term:

| batch | ctx | KV bytes | share of the step (f16 weights) | a 2x on attention is a step-level |
|---|---|---|---|---|
| 16 | 2048 | 4.295 GB | **29.6%** | **1.17x** |
| 64 | 2048 | 17.180 GB | **62.7%** | **1.46x** |
| 64 | 8192 | 68.719 GB | **87.1%** | **1.77x** |

> **REFUSAL, Wave 6's own: an attention-kernel ratio may not be published as a serving result
> without the batch and context it was taken at, and without the step-level number beside it.** The
> same 2x is a 1.17x product at batch 16 and a 1.77x product at batch 64 / ctx 8192, and the
> difference is Amdahl, not engineering. This is the decode twin of Wave 4's refusal against
> publishing a GEMM-parity fight as a fusion win.

Two fairness facts to record in the round log rather than discover afterwards: **(i)** FA2's decode
path and ours both walk a block table, so the paging model matches and neither side gets a
contiguous-cache advantage; **(ii)** our kernel keeps **f32** logits and V accumulation
(`paged_attention.rs:8-9`) -- if FA2's kvcache path accumulates narrower, the comparison is a
speed/accuracy trade and must be published as two columns, exactly as `WAVE5_DOSSIER.md:584-588`
requires for `FAST_ACCUM`.

### 9.3 vLLM: a floor, and the plan is right about why

`ACT2_WAVE_PLAN.md:178` refuses "any serving headline against `paged_attention_v2` alone", because
`benchmark_paged_attention` drives the **deprecated** v1/v2 op. Keep it as a labelled floor. The
usable vLLM column for this wave is `cutlass_scaled_mm` on the *quantized decode GEMM*
(`WAVE5_DOSSIER.md:558`), which is a peer for section 3's weight-dtype arm and is the one place
vLLM is the right bar rather than the available one.

### 9.4 The megakernel peer, and why the wave should not build it

**FACT(repo).** The launch layer for a persistent cooperative decode kernel exists in full:
`grid_barrier_ptx` (`megakernel.rs:132`), `max_resident_ctas` via
`cuOccupancyMaxActiveBlocksPerMultiprocessor x GpuTarget::sm_count` (`:294-308`), `plan_grid`
declining rather than clamping (`:320-358`), and `launch_mega` through
`cuLaunchCooperativeKernel`. **The kernel does not.** `megakernel.rs:47-49` states it plainly:
"`lower::emit_mega_ptx` still emits the block-scoped form today", and the grid-parallel arm is
exercised by exactly one test.

**DERIVED.** A persistent decode megakernel's prize is the launch overhead of section 7, which is
between 2.7% and 13.5% of the step and is **unsized because the repo has no H100 launch constant**.
Against that: the peer is a full vLLM/SGLang serving-stack build (the plan's own "most expensive
peer on this list", `:203`), the kernel is a from-scratch lowering, and a cooperative grid that is
not fully resident **deadlocks the device** (`megakernel.rs:32-36`).

> **Budget the megakernel at ZERO for Wave 6**, exactly as `WAVE3_DOSSIER.md:1412-1415` budgets the
> mainloop drain. Its trigger is computable and belongs in section 10.5: *fire the megakernel only
> if the section-7 launch-overhead measurement exceeds 10% of the measured decode step.* One number
> decides it, and that number costs seconds.

---

## 10. HELD-OUT TRIGGER CHECK -- the wave that fires the triggers

Wave 6 is the row three earlier dossiers deferred levers *onto*. Each is settled here with
arithmetic, not with a shrug.

### 10.1 Ping-pong -- the trigger FIRES, and the lever still loses

`WAVE3_DOSSIER.md:1375-1376`: *"Trigger unchanged and still correct: **a decode / skinny-M row.** The
suite's smallest M is 1024 = 8 m-tiles, so nothing qualifies."* Wave 6 adds `M = Bcap` in
`{64, 256}`. **The trigger fires.** And the lever loses three times over.

**Loss 1 -- the roof has no tile in it.** Section 4.2: below `M = 338` the decode GEMM's ceiling is
`M x BW_HBM`. Ping-pong changes the CTA tile from 128x256 to 64x256. Neither `128` nor `64` appears
in `M x BW`. **A lever that changes a quantity absent from the binding constraint cannot move it.**

**Loss 2 -- at Bcap=256 it lowers the ceiling that is not binding onto the one that is.** Using
`WAVE3_DOSSIER.md:1356-1362`'s own `I_cta` arithmetic and the measured `BW_L2 = 7.00 TB/s`:

| tile | `I_cta` (with B-multicast) | L2 roof | HBM roof at M=256 | binding roof |
|---|---|---|---|---|
| 128x256 | 128.0 | 896 TFLOP/s | 630 TFLOP/s | **630 (HBM)** |
| 64x256 (ping-pong) | 85.33 | **597 TFLOP/s** | 630 TFLOP/s | **597 (L2)** |

Ping-pong converts a bandwidth-bound shape into an L2-bound one and costs **5.2%** of the ceiling.
It does double the tile count (32 -> 64 CTAs at Bcap=256, 24% -> 48% of the device), which pushes
*toward* whichever roof binds -- so the honest verdict at Bcap=256 is "at best a wash", not "a win".

**Loss 3 -- at Bcap=64 it does not even add tiles.** `gy = ceil(64/128) = 1` at 128x256 and
`ceil(64/64) = 1` at 64x256; `gx = 16` either way. **16 tiles both ways.** The only thing ping-pong
buys at Bcap=64 is not wasting half the accumulator registers on masked rows, which is a register
argument, not a time argument, at 12.1% device occupancy.

**And it is not even reachable.** Ping-pong is a `wgmma` schedule; the decode path calls
`wmma_nt_f16_sm_db` (`serving.rs:339`) and no `wgmma` kernel has a decode call site
(`ACT2_WAVE_PLAN.md:34`).

> **VERDICT: DEAD, and the trigger is RETIRED rather than deferred.** `WAVE3_DOSSIER.md:1385` should
> gain a second reason: the decode row that its trigger names was added, and at `M <= 338` the
> binding roof is `M x BW`, in which no tile shape appears. Ping-pong now has *three* independent
> refutations -- WAVE3's L2 roof below the measured floor, WAVE4's claim on the same `X_epi`, and
> this one. Do not re-open it.

### 10.2 Stream-K / split-K -- the trigger FIRES, and this one should be PROMOTED

`WAVE3_DOSSIER.md:1336-1341` sharpened the trigger to a predicate:

> *"Stream-K fires when the best emittable tile's wave efficiency `tiles / (132 * ceil(tiles/132))`
> is below 0.90 -- i.e. when `tiles mod 132` lands in the bad band and `tiles < 264` -- with K large
> enough that splitting it amortises a fixup. No shape in the current suite qualifies."*

**DERIVED**, evaluated on the six decode GEMMs at `Bcap = 64`:

| GEMM | tiles (64x64) | wave efficiency | `tiles < 264`? | K | trigger? |
|---|---|---|---|---|---|
| `wq` | 64 | **0.485** | yes | 4096 | **FIRES** |
| `wk` | 16 | **0.121** | yes | 4096 | **FIRES** |
| `wv` | 16 | **0.121** | yes | 4096 | **FIRES** |
| `wo` | 64 | **0.485** | yes | 4096 | **FIRES** |
| `w1` | 224 | **0.848** | yes | 4096 | **FIRES** |
| `w2` | 64 | **0.485** | yes | 14336 | **FIRES** |

**Six of six.** At the `wgmma` 128x256 tile it is worse still (`wq` -> 16 tiles = 0.121).

**The numerical precondition, resolved precisely.** WAVE3 held Stream-K partly on "a changed
summation order ... the f64 tolerance arm's `c*sqrt(K)*eps` bound becomes shape-dependent and must
be re-derived". At decode that cost is smaller than it looks, and it is worth writing down which
gates survive untouched:

| serving gate | compares | split-K breaks it? |
|---|---|---|
| `paged_attention_invariant_to_block_layout` | the attention kernel against itself under two layouts | **no** -- different kernel |
| `serving_decode_step_invariant_to_block_layout` | the step against itself under two layouts | **no** -- both runs use the same split |
| `serving_decode_step_invariant_to_batch_composition` | slot 0 alone vs co-batched | **no** -- shard boundaries depend on K, not on the batch |
| `serving_decode_graph_matches_eager` | graphed vs eager, same kernels | **no** |
| `serving_decode_step_matches_reference` | the step against an **f64 oracle** | **YES** -- reassociation; re-derive the bound |

> **Four of the five bit-exactness gates survive split-K untouched, because they all compare the
> kernel against itself under a permutation of its inputs rather than against a fixed reduction
> order.** Only the f64-reference tolerance moves, and it moves into the repo's standing
> "reassociation-exception class" (`serving.md:271`) that the Megatron row-parallel all-reduce
> already lives in. That is a materially cheaper precondition than WAVE3's suite faced.

**Size.** Bounded by section 4.3's MODEL: up to **6.6x** on the 16-CTA launches and **1.7x** on the
64-CTA ones *if* the memory-parallelism model holds, and **1.0x** if it does not. **The one
measurement that decides it is the achieved bandwidth of a single decode GEMM launch.**

> **PROMOTION, recorded here for `ACT2_WAVE_PLAN.md`'s held-out list: Stream-K's trigger has fired.
> It moves from "HELD OUT" to "Wave 6, rank 2, gated on one bandwidth measurement."** Its cheap
> cousin -- the merged-QKV concat of section 4.4(a) -- should be built first regardless, because it
> is bit-exact by construction and needs no workspace, no fixup and no re-derived tolerance.

### 10.3 2x2x1 cluster and 192x256x64 -- unaffected

`WAVE3_DOSSIER.md:1275-1279` re-evaluates 2x2x1 after **Wave 4**, on the predicate "some shape's
`T_L2` exceeds its `T_floor`". Decode shapes are HBM-bound at 18-64% of the tensor peak
(section 4.2), so their `T_L2` is nowhere near binding and Wave 6 supplies no new evidence in either
direction. **Unchanged: DEFERRED to after W4.** 192x256x64 is **DEAD** on the emitter's own
CTA-M decline and the 512-thread register cap (`WAVE3_DOSSIER.md:1289-1316`); nothing about decode
touches either. **Unchanged: DEAD.**

### 10.4 Machete / W4A16 -- both of the plan's bands are entered, and only one of them is a tie

`ACT2_WAVE_PLAN.md:189` holds "Beating Machete on W4A16" out as measure-only, with a sweep at
`M = 1 / 16 / 128` and the expectation pre-registered in **two** bands: *"At M<=32 the shape is 100%
weight-bandwidth-bound (12.98 MB = 3.87 us caps a perfect M=16 kernel at 208 TFLOP/s; two kernels at
90% of HBM differ by <10%). Beating it at M=128+ needs W5's plumbing plus a weight pre-shuffle.
**Expect and publish a loss at M>=128 and a tie at M=1.**"*

Wave 6's decode batches are `M in {16, 64, 256}` (section 12.2: `B = 16` and `64` on the step rows,
`Bcap = 256` on the goodput row), and they **straddle** those bands rather than sitting inside
either. `M = 16` is inside the plan's tie band; `M = 256` is inside its pre-registered LOSS band;
`M = 64` falls in the gap between 32 and 128, where the plan states no expectation at all. **This
dossier does not overturn the pre-registered expectation -- section 4.2's roof re-derives it, and
the loss at `M >= 128` stands.**

**DERIVED, where the both-sides-bandwidth-bound band actually ends.** Section 4.2's roof with 4-bit
weights replaces the `2*N*K` weight term by `0.5*N*K` and leaves the f16 activations and f32 output
alone:

```
    FLOP  = 2*M*N*K
    bytes = 0.5*N*K (4-bit W) + 2*M*K (f16 act) + 4*M*N (f32 out)
    I     ~ 4M   when M << K and M << N     ->   M* = 338.4 / 4 = 84.6
```

At the exact `wq` decode shape (`N = K = 4096`), where the activation and output terms are not
negligible, `I(M) = 33,554,432*M / (8,388,608 + 24,576*M)` and the crossover is `M = 112.5`. Both
brackets give the same verdict at every swept point:

| M | I (FLOP/B) | roof at 2922.3 GB/s | vs the 989 TFLOP/s f16 tensor peak |
|---|---|---|---|
| 16 | 61.13 | 178.6 TFLOP/s | 18.1% -- **weight-bandwidth-bound** |
| 64 | 215.6 | 630.0 TFLOP/s | 63.7% -- **weight-bandwidth-bound** |
| 112.5 | 338.4 | 989 TFLOP/s | **100% -- the crossover** |
| 128 | 372.4 | 1088 TFLOP/s | above peak -- **compute-bound** |
| 256 | 585.1 | 1710 TFLOP/s | above peak -- **compute-bound** |

So the round publishes three separate things, and conflating any two of them is a defect:

* **`M <= 64`: expect a tie.** Both kernels read the same weight bytes under the same roof, and
  section 4.2's roof does not care whose kernel it is. This *extends* the plan's band from `M <= 32`
  to `M <= 64` on the arithmetic above; it does not reverse it.
* **`M in {128, 256}`: expect a LOSS, exactly as pre-registered.** Above the crossover the weight
  stream is no longer the binding term, so the winner is decided by tensor-core efficiency and by
  the weight pre-shuffle the plan names as the precondition -- and **Wave 6 supplies neither.**
  Section 3 records that int4 weights have no decode call site at all (`serving.rs:339` launches
  `wmma_nt_f16_sm_db` unconditionally). Publish the loss; it is the pre-registered result, not a
  surprise, and a dossier that quietly turned it into a tie would have destroyed the one thing a
  pre-registration is for.
* **vs our own f16 decode: 2.118x, and it is the wave's largest single number.** That is a *product*
  ratio against an internal control at `B = 16` (section 2.3), legitimate precisely because it is
  labelled as one. It is not a Machete comparison and must never appear beside one without the
  label.

**AMENDMENT to a held-out item, named as one.** The plan's sweep is `M = 1 / 16 / 128`; row
`p3_marlin` runs `M = 1 / 16 / 64 / 128 / 256`, a **superset**. Both plan anchors are kept so the
pre-registered expectation stays falsifiable at the two points it was registered at, and the wave's
own decode batches (64, 256) are added. Section 13 carries the amendment row.

`modal_app.py`'s `::marlin` entrypoint (`def marlin` at `:3964`) picks the kernel by device itself:
`which = kernel or ("machete" if cc == "sm_90" else "marlin")` at `:4009`, with the two off-device
strawman guards at `:4012-4017`. Use that auto-pick rather than naming a kernel by hand.

### 10.5 Summary of the trigger check

| lever | trigger status in Wave 6 | why, in one line |
|---|---|---|
| **ping-pong** | **FIRES -> DEAD. Retire the trigger.** | at `M <= 338` the roof is `M x BW`; no tile shape appears in it, and at Bcap=256 the 64x256 tile *lowers* the binding roof 630 -> 597 TFLOP/s |
| **Stream-K / split-K** | **FIRES -> PROMOTE to rank 2** | 6 of 6 decode GEMMs are under 0.90 wave efficiency, 5 under 0.50; 4 of 5 bit-exactness gates survive it untouched |
| **merged QKV** (new, this dossier) | n/a -- **build it, rank 3** | 3 launches -> 1, 96 CTAs instead of 64+16+16, and bit-exact by the same argument `tp_column_parallel_gemm_split_is_bit_exact` already proves |
| 2x2x1 cluster | unchanged: **DEFERRED to after W4** | decode is HBM-bound, so `T_L2` is not binding and Wave 6 adds no evidence |
| 192x256x64 | unchanged: **DEAD** | CTA-M 192 declines in the emitter; 512 threads cap ptxas at 128 regs |
| Machete / W4A16 | **both bands entered: tie at `M <= 64`, the plan's pre-registered LOSS at `M >= 128`** | the W4A16 roof crosses the 989 TFLOP/s f16 peak at `M ~ 112`, so only `M <= 64` is both-sides-weight-bandwidth-bound; the 2.118x is a product ratio against our own f16 decode, never a peer ratio |
| **megakernel** | **HELD, with a computable trigger** | fire only if the measured H100 launch overhead exceeds 10% of the measured decode step |

---

## 11. GUARDS -- law text

Each is written so it can become a test name and an assertion message.

### G-W6-1. The decode family is not in the ptxas census, and a spill here is silent

**FACT(repo).** The census corpus in `ptxas_reports_the_register_and_spill_budget`
(`gpu.rs:22217`, corpus built at `:22250-22335`) is exactly three families: the `wmma` deep/wide
lattice, the flash variants, and the `wgmma` rows plus two bring-up entries. **`paged_attn_decode`,
its int8 twin, `kv_append`, the norm module and the 64x64 `_sm_db` decode GEMM are not in it.** They
*are* in `device_free_modules()` (`gpu.rs:8247`, EXPECTED_MODULES = 117 at `:8624`, the paged entries
at `:8473-8488`), so they are covered by the ASCII, `.target` and `.version` laws -- and by **no
register or spill law at all**.

**DERIVED, why that is dangerous now.** `paged_attn_decode_ptx(128)` declares
`.reg .f32 %acc<128>` (`paged_attention.rs:147`) plus roughly 70 more scalars (`:148-152`) -- about
**198 virtual registers** against a 255 hard cap. The `v4` arm adds staging registers; the V-side
g-fold multiplies the accumulators by `g` (512 at `g=4`). A spill in a decode kernel is a **silent
2-10x** into local memory, and nothing in this tree would report it.

> **Law: every module the decode path loads is a row of the ptxas census, with `spill_st == 0`,
> `spill_ld == 0`, `stack == 0` and no C7511, before the H100 is rented.** Add
> `paged_attn_decode_ptx(64)`, `(128)`, both int8 twins, `kv_append_ptx`, `kv_append_int8_ptx`,
> `norm_ptx` and `wmma_f16_ptx`'s `_sm_db` entries to the census corpus. Cost: $0.02, the same
> census the wave already runs.

### G-W6-2. The `v4` alignment law, per dtype

> **`paged_attn_decode_ptx` must assert `head_dim % 8 == 0` and `paged_attn_decode_int8_ptx`
> `head_dim % 16 == 0` the moment either emits a 16-byte load**, and the message must name the
> arithmetic (`elem_offset` puts a vector at byte `es * head_dim * sizeof(elem)`; a 16-byte
> `ld.global.v4.b32` needs that 16-byte aligned).

Today both assert only `% 2` (`:109-111`, `:410-412`). Both shipped head dims (64, 128) satisfy
both rules, so **the corpus cannot catch the omission** -- which is exactly why the law must be
written rather than tested into existence.

### G-W6-3. The lane partition stays position-keyed (the property, restated as a law)

> **`paged_attention_invariant_to_block_layout` and its three siblings gate every decode-attention
> rewrite, and any rewrite that makes the lane partition a function of the physical block id is
> rejected at review, not at the gate.**

`paged_attention.rs:37-47` already derives why: the property rests on the partition being
`t % 32`-keyed and the merge being a fixed butterfly, and partitioning by physical block "still
satisfies 'one fixed accumulation order per lane'" while breaking it. Section 5.6's table is the
per-rewrite verdict; **the `v4` and K-fold arms are additionally bit-identical to the shipped
kernel**, so they get an `==` gate against it and not merely the layout gate.

### G-W6-4. Split-K's reduction order is fixed and declared

> **A split-K decode GEMM sums its `S` partials in a fixed shard order, on the device, with no
> atomics.** Determinism is not optional here: `serving_scheduler_drains_and_conserves_blocks`
> asserts that two identical request streams produce the identical per-step schedule
> (`serving.md:194-195`), and a non-deterministic reduction turns that into a flaky gate rather than
> a wrong answer -- the worst failure mode.
>
> **And the f64-reference tolerance is re-derived, never widened to fit.** The new bound is a
> reassociation of a `K`-reduction into `S` shards; it belongs in the same exception class as
> `tp_row_parallel_gemm_allreduce_matches` (`serving.md:268-271`, max_abs 1.34e-5 at K=256).

### G-W6-5. `plan_grid` is the residency bound, and a cooperative grid that is not resident hangs

> **No cooperative decode launch may bypass `megakernel::plan_grid`.** It returns `Ok(None)` -- a
> clean decline -- when the device cannot host the grid, and it declines rather than clamps
> precisely because "a clamped sweep reports a number for a grid nobody asked for"
> (`megakernel.rs:346-348`). The barrier state is a **launch parameter**, never a module-scope
> `.global`, because `Gpu::function` caches modules for the process's life and a faulted barrier
> would hang the *next* program -- that is `GRID_BARRIER_BYTES`'s own doc comment, verbatim, at
> `megakernel.rs:89-94` (the module header states the same rule at `:32-33`).

### G-W6-6. G16 applies, and the decode contender's dispersion is unknown

> **No serving ratio is published until the peer's own dispersion has been measured at the decode
> shape.** The campaign's dispersion floors are GEMM floors, and both ends of the range come out of
> round logs rather than out of the checkpoint: round 3's per-cell control floors ran `+/-0.00%` to
> `+/-3.29%` (the `+3.29%` cell is `w1_s4_off/sq2048` at
> `bench/gpu/h100/2026-08-10-h100-act2-r3-bmulticast.log:538`; `WAVE3_DOSSIER.md:1498` records the
> range and which pairs to quote), and sq1024 was **refused** when its peer A-vs-C floor came in at
> `+/-15.47%` (`bench/gpu/h100/2026-08-11-h100-w2-r10-vs-cublas-v2rule.log:142`, "GATE SHUT ...
> exceeds the pre-registered bar +/-5.00%"). `CAMPAIGN_CHECKPOINT.md:49` carries that one refusal
> rounded to `+/-15.5%` and is the only dispersion figure in the checkpoint -- it does not carry the
> `+/-3.29%`, so do not cite it for the range. A decode kernel at 12% SM
> occupancy with a ragged per-sequence context is a far noisier contender than a square GEMM, and
> **nothing in this repo has measured how noisy.** Run the two-identical-arms control (arms A and C
> both FA2) before any ratio arm, exactly as Wave 2 did.

### G-W6-7. The `sm_80` floor is a feature here -- do not let a rewrite raise it

**FACT(repo).** Every generator in `paged_attention.rs` opens with `HDR_SM80`
(`paged_attention.rs:24-27`, `:127`), and the module says why: the instruction mix is
`shfl.sync.bfly` + `red.f32` + ordinary ld/st, all Ampere-legal, so it JITs on every part from A100
up **and its PTX-shape gates run in a plain toolchain-free `cargo test`**.

> **`ld.global.v4.b32` is Ampere-legal and keeps the floor. A rewrite that reaches for
> `cp.async.bulk`, a cluster, or any `sm_90a` instruction moves this family to an
> architecture-specific header and off every non-Hopper part.** If a Hopper-only decode arm is ever
> wanted, it is a *second* module at a second floor, and `the_family_declares_no_floor_it_does_not_need`
> applies (`WAVE5_DOSSIER.md:743-754`).

### G-W6-8. Round ordering (standing rules 1 and 3)

> **The $0.02 CPU ptxas census -- now including G-W6-1's decode rows -- runs before the H100 is
> rented; the bring-up and the correctness arms run before the peer arms in the same container.**
> Wave 6 has an unusual advantage here and must not waste it: section 1.3 shows the correctness
> arms are *already green on H100*, so the bring-up cost of this wave is a re-run, not a
> bring-up.

---

## 12. RANKED SUMMARY AND THE ONE-VISIT ROUND

### 12.1 The levers, ranked by derived effect on H100

**1. WEIGHT AND KV DTYPE -- up to 3.042x on the step floor, and it is pure byte arithmetic.**
int4 weights + int8 KV takes the batch-16 step floor from 4.960 ms to 1.631 ms; 8-bit weights alone
are 1.543x; int8 KV alone is 1.168x at batch 16 and 1.437x at batch 64. The int8-KV half is **built,
gated and green on H100** and costs a measurement. The weight half needs a decode call site that
does not exist. This is Wave 5's output arriving in Wave 6's product.

**2. SPLIT-K / STREAM-K ON THE DECODE GEMM -- 1.0x to 6.6x on five of six launches, gated on one
number.** Six of six decode GEMMs fail WAVE3's own wave-efficiency predicate, two of them at 12.1%.
The prize is whatever separates the achieved bandwidth of a 16-CTA launch from 2922.3 GB/s, and that
separation has never been measured. Four of the five serving bit-exactness gates survive it.

**3. MERGED QKV -- 3 launches to 1, bit-exact by construction, host-side concat only.** 96 CTAs in
one wave instead of 64+16+16 in three, and 64 fewer launches per step. No workspace, no fixup, no
re-derived tolerance. Build it first because it cannot be wrong.

**4. `v4` COALESCING ON THE PAGED DECODE ATTENTION -- 8x fewer load instructions, an unknown
fraction of which is time.** Sector efficiency 6.25% -> 50%, 256 -> 32 loads per position, and the
values stay **bit-identical** to the shipped kernel so the gate is `==`. Whether it converts is the
LSU-vs-HBM question nobody has answered.

**5. THE K-SIDE GQA FOLD -- `g` registers, removes `(g-1)/g` of the K loads.** Free at
`g in {4, 8}`. Its V-side twin is a 512-register wall at `head_dim = 128` and should not be built
until the section-5.4 falsifier says the g-fold costs HBM bytes.

**6. THE NORM PLANNER FIX -- three lines, 0 to ~9 points of the step.** `DecodeLayer` hardcodes the
one-warp entry and a 64-CTA grid on a 132-SM part while the SM-filling family and its planner sit
unused in the same crate.

**7. THE CUDA-GRAPH / MEGAKERNEL TERM -- 2.7% to 13.5% of the step, and the constant is a WDDM
number.** Budget at **ZERO** until the launch-overhead probe runs. It costs seconds and it decides
whether the most expensive item in the plan is worth anything at all.

### 12.2 The one-visit round

**Preceded by the $0.02 CPU ptxas census**, extended per G-W6-1 to the decode family, gating on
`spill_st == 0`, `spill_ld == 0`, no C7511, `.target sm_80` on the paged family, pure ASCII.

**Then, in one container, in this order.** Rows 1-3 are the denominators; nothing after them is
interpretable without them.

```
  row              arm                                              shapes / config                    from
  --- denominators (run first; each is seconds) ---
  d1_readbw        hbm_bandwidth with a DEVICE-SIZED reduce grid    n = 64 Mi f32                      1.4(a)
                   (a BENCH-LOCAL grid = blocks_per_sm * sm_count;
                    RED_GRID stays 256 -- it is the product reduce's
                    determinism contract, gpu.rs:1300-1301, :1361-1363)
  d2_launch        448 empty launches eager vs 1 graph replay       448                                7
  d3_gemmbw        achieved bandwidth of ONE decode GEMM launch     M=64,  N=1024, K=4096 (wk-shaped)  4.3
                   (and its 64-CTA sibling)                         M=64,  N=4096, K=4096 (wq-shaped)
  --- the shipped stack, measured for the first time on H100 ---
  s1_step_f16      DecodeModel step, f16 W / f16 KV                 B=16 and 64, ctx 2048, 32L Llama-8B  2.3
  s2_step_int8kv   same, KvDtype::Int8                              same                                6
  s3_goodput       serving_continuous_batching_goodput              Bcap 64 / 256                       1.2
  s4_graph         serving_decode_graph_throughput                  depth 32                            7
  --- the derived levers, each against s1 as its control ---
  a1_v4            v4 decode attention, == gate vs the f16 kernel   B=16/64, ctx 2048 and 8192          5.2
  a2_kfold         K-side GQA fold, == gate                         same                                5.5
  a3_qkv           merged-QKV concat, == gate vs unmerged           B=64                                4.4(a)
  a4_splitk        split-K on wk/wv, S in {4, 8}                    B=64                                10.2
  a5_normplan      norm through norm_launch                         B=64, D=4096                        8
  --- peers (last; every one needs its own dispersion control first) ---
  p0_disp          FA2 vs FA2, two identical arms (G16)             the a1 shapes                       G-W6-6
  p1_fa2           flash_attn_with_kvcache(block_table=)            the a1 shapes                       9.2
  p2_vllm          benchmark_paged_attention  [LABELLED A FLOOR]    same                                9.3
  p3_marlin        ::marlin at M = 1/16/64/128/256                  Llama-8B weights                    10.4
                   (the plan's 1/16/128 anchors KEPT so its
                    pre-registered expectation stays falsifiable;
                    64/256 added as the wave's own decode batches)
```

### 12.3 Round-level refusals

* **Publish nothing until `d1_readbw` has run.** Every floor in this dossier divides by 2922.3 GB/s,
  a copy figure, on a pure-read workload whose only measured read number came off the reduce's fixed
  256-CTA determinism grid at 682.1 GB/s -- 1.94 CTAs/SM on this part. That is a 4.28x band and it
  sits under every claim.
* **No serving ratio without its batch and its context.** Section 9.2: the same attention win is
  1.17x or 1.77x of the step depending on the batch. A ratio without its shape is not a result.
* **No peer ratio before `p0_disp`.** A gain inside the contender's own dispersion is not a result
  (G16), and the decode contender's dispersion is unmeasured.
* **No vLLM headline** -- it drives the deprecated v1/v2 op and is a labelled floor
  (`ACT2_WAVE_PLAN.md:178`).
* **No quantized-decode row without its accuracy column**, measured at the serving geometry rather
  than inherited from the unit gate (section 6).
* **The Machete row publishes per `M`, against the band it was pre-registered in.** A tie at
  `M >= 128` is not a result to celebrate, it is a result to *check*: the plan registered a loss
  there and section 10.4 re-derives why. Reporting a single aggregate "tie" across the sweep, or
  quoting the internal 2.118x anywhere near the peer column, is a refusal.
* **No 4050 number anywhere in the output.** Not as a comparison, not as an expectation, not as a
  "consistent with". This dossier's device-scope rule is a round-level refusal too.
* **Lock the SM clock and declare it.** The r3 provenance still reads `clock lock: UNKNOWN`
  (`r3-bmulticast.log:528`). Decode is a low-occupancy workload and therefore a *high-boost* one --
  it is more clock-sensitive than a GEMM, not less.

### 12.4 WHAT CANNOT BE DERIVED WITHOUT A MEASUREMENT

Eight items. Each names the cheapest measurement that closes it. **Seven of the eight are
free-or-seconds and should all be in the same container.**

| # | Cannot be derived | Why | The measurement |
|---|---|---|---|
| 1 | **H100 read-only bandwidth at a machine-sized grid** | the only read number is `reduce` at the fixed 256-CTA *determinism* grid (`gpu.rs:1300-1303`); copy carries a write stream | `d1_readbw`: give the **bench's** reduce arm (`gpu.rs:23538-23544`) its own `sm_count`-sized grid. `RED_GRID` and the product reduce (`gpu.rs:1361-1363`) are not touched. One launch, seconds |
| 2 | **Whether the decode-attention kernel is LSU-bound or HBM-bound** | decides whether the `v4` rewrite is 1x or up to 8x; the plan's 1.6-2.0x is FACT(ext) sizing and is declined | `a1_v4` A/B against the scalar kernel, with an `==` gate |
| 3 | **Whether the g-fold costs HBM bytes or only L1/L2 requests** | decides whether the tiled q-group is rank 1 or rank 5, and whether the plan's "4x that" stands | measure DRAM read bytes for one step; divide by `2*L*ctx*kv_heads*hd*2*B`. Ratio ~1 or ~g |
| 4 | **Achieved bandwidth of a 16-CTA and a 64-CTA decode GEMM** | the entire size of lever 2 (split-K), 1.0x to 6.6x | `d3_gemmbw`, two launches |
| 5 | **H100/Linux per-launch overhead** | decides levers 7 and the megakernel; the repo's 1.5 us is WDDM and non-transferable | `d2_launch`: 448 empty launches vs one graph replay, seconds |
| 6 | **The norm's real share of the decode step on H100** | bounded 1.4% to ~11%; `MIN_ELEMS_PER_LANE`, `FILL_CTAS_PER_SM` and `OCCUPANCY_WARPS` were all calibrated on 20 SMs | `a5_normplan` A/B, plus a `WUKONG_NORM_WARPS` sweep at the decode shape |
| 7 | **FA2's own dispersion at decode shapes** | without it no peer ratio is publishable (G16); the campaign's floors are GEMM floors | `p0_disp`: two identical FA2 arms |
| 8 | **Register/spill for every decode module on any architecture** | the census corpus is wmma+flash+wgmma only; `%acc<128>` sits ~198 virtual registers deep and every Wave-6 rewrite adds more | the $0.02 CPU census, corpus extended per G-W6-1. **Runs before the GPU is rented** |

Two more that are *not* cheap and should be named as such rather than quietly attempted:

* **A real vLLM/SGLang end-to-end serving comparison.** A full stack build, not a `dlopen`
  (`ACT2_WAVE_PLAN.md:203`). Out of scope for one visit.
* **What a 3e-3 per-layer int8-KV error does to a 32-layer model's output distribution.** Needs a
  real checkpoint and a reference generation, which is a different kind of round entirely.

---

## 13. What this dossier changes in `ACT2_WAVE_PLAN.md`

| plan claim (`:168-178`, `:187`, `:202-203`) | this dossier |
|---|---|
| Wave 6's levers are ranked attention-first (paged decode coalescing, then GQA, then int8/fp8 KV, then norms) | the byte budget puts **weight dtype first** (up to 3.042x) and **decode-GEMM parallelism second**; attention coalescing is rank 4 and is an instruction-count lever, not a byte-count one |
| "batch 16 / L=2048 / 32 layers requires 4.29 GB/step = **1.28 ms at 3.35 TB/s**" | 3.35 TB/s is the spec sheet. At the **measured** 2922.3 GB/s the same 4.29 GB is **1.470 ms** -- 15% slower. And the step also reads **10.20 GB of weights** the plan never counts |
| "GQA fixes a hard g-fold read amplification (4x at 8B, 8x at 70B) ... today we ask for 4x that" | the 4.29 GB is **already** the g-folded number (`kv_heads = 8`). The g-fold is a **request** amplification absorbed by L1 (`g = 4`, one CTA) or L2 (`g = 8`, two CTAs) -- not 17.2 GB of HBM. Falsifier stated in 5.4 |
| "v4 plus QuACK's coalescing prerequisite puts the band at 75-85%, i.e. **1.6-2.0x** on the decode-attention term" | that is literature sizing a lever, which this directory forbids. Derived: **8x fewer load instructions**, sector efficiency 6.25% -> 50%, and a speedup bounded by an unmeasured LSU-vs-HBM split |
| "the accumulator-tiling design step (g=4 at d=128 is 512 registers -- impossible without tiling)" | correct, and the tiling has only bad options at `head_dim = 128`: chunking re-reads V; buffering scores needs the SMEM logits buffer the module deliberately lacks. **Ship the K-side fold (`g` registers) and stop** |
| "plus the `norm_vs_peers` bench that does not exist" | it still does not, QuACK is not staged, and norms are **1.39% of the decode step**. Build the three-line planner fix (`serving.rs:336` bypasses `norm_launch` entirely) and skip the peer bench |
| the owner list includes "the bench wiring in `gpu.rs`" | **`gpu.rs` is single-owner per wave** (`CAMPAIGN_CHECKPOINT.md:25-26`), and every serving bench in the tree already lives in its own module (`serving.rs:2468`, `:3101`). Delete the `gpu.rs` clause and the ownership conflict with Wave 3 disappears |
| megakernel: "Published ceiling: 78% of BW ... ~1.56x on the bandwidth-bound term" | FACT(ext) sizing. The repo's derived prize is the **launch overhead**, 2.7-13.5% of the step, and its only constant is a WDDM one. **Budget ZERO**; trigger = the measured overhead exceeds 10% of the measured step |
| ping-pong is HELD OUT with trigger "a decode row is added" (`:187`) | **the trigger fires and the lever loses.** At `M <= 338` the roof is `M x BW`, in which no tile appears; at Bcap=256 a 64x256 tile *lowers* the binding roof 630 -> 597 TFLOP/s. **RETIRE the trigger** |
| Stream-K is HELD OUT: "wave quantization is only ~3% once W3's persistence lands" | true for the GEMM suite, false for decode: **6 of 6 decode GEMMs are below 0.90 wave efficiency, 5 below 0.50.** **PROMOTE to Wave 6 rank 2**, gated on one bandwidth measurement |
| Machete W4A16 is HELD OUT at `:189` as measure-only, sweep `M = 1/16/128`, "**expect and publish a loss at M>=128 and a tie at M=1**" | **expectation UPHELD, sweep EXTENDED -- and this row is the amendment.** With 4-bit weights `I ~ 4M`, so the roof crosses the 989 TFLOP/s f16 peak at `M = 84.6` asymptotically and at `M = 112.5` at the exact `wq` shape: the both-sides-bandwidth-bound band ends near `M = 64`, not `M = 32`, and `M >= 128` stays the pre-registered LOSS. `p3_marlin` runs `M = 1/16/64/128/256` -- a **superset** keeping both plan anchors and adding the wave's own decode batches. The 2.118x of section 2.3 is an internal product ratio, not a Machete number |
| target-table row 6: "1.6-2.0x from coalescing compounded with a 4x (8B) / 8x (70B) read-amplification removal" | both factors are re-derived above and neither survives as stated. The row's honest content is the **int8/fp8-KV prize** and the **capacity** result of 2.4 |
| (absent from the plan) | **int8 KV is a feasibility lever on H100**: `Bcap=64 x 8192` is 77.00 GiB of a 79.18 GiB part at f16 and 46.00 GiB at int8 |
| (absent from the plan) | **merged QKV**: 3 launches -> 1, bit-exact by construction, no numerical precondition |
| (absent from the plan) | **the decode path never touches `wgmma`** (`serving.rs:339`), so Waves 3/4/5 reach serving only through a wiring change nobody has scheduled -- the same defect the plan already records for `gemm_nt_wgmma` |

---

## 14. SOURCES

### In-tree (FACT(repo)) -- these outrank everything below

| what | where |
|---|---|
| H100 device identity: 132 SMs, 232,448 B SMEM, 50.0 MiB L2, 79.18 GiB VRAM, clock lock UNKNOWN | `bench/gpu/h100/2026-08-10-h100-act2-r3-bmulticast.log:517-531` |
| H100 bandwidth: spec 3352.3, copy **2922.3**, saxpy 2567.0, reduce **682.1** GB/s | `bench/gpu/h100/2026-08-11-h100-w2-r5-hbm.log:50-53` |
| The campaign's GEMM dispersion floors, quoted by G-W6-6: `+3.29%` (round 3, `w1_s4_off/sq2048`) and the `+/-15.47%` sq1024 refusal | `bench/gpu/h100/2026-08-10-h100-act2-r3-bmulticast.log:538`; `bench/gpu/h100/2026-08-11-h100-w2-r10-vs-cublas-v2rule.log:142`; range restated at `docs/gpu/derive/WAVE3_DOSSIER.md:1498`, refusal restated at `CAMPAIGN_CHECKPOINT.md:49` |
| The whole serving correctness suite, **green on H100**; every serving perf bench `#[ignore]`d | `bench/gpu/h100/2026-08-10-h100-s2d-full-suite.log:171`, `:2274-2501`, `:2493-2497`, `:2707-2741` |
| The Llama-3-8B decode geometry and the 64.00 GiB f16 KV cache, printed on H100 | same log, `:2495` |
| Staged peer bar: FA2 2.8.3.post1 (`kvcache_api`/`kvcache_symbol` true), vLLM 0.26.0, torch 2.13.0+cu129, ptxas 12.9.86 | `bench/gpu/h100/2026-08-11-fa2-stage.log` (the `peers.json` audit block) |
| The paged decode-attention kernel: warp map, lane partition, scalar `ld.global.u16`, the GQA head map, the register declaration | `crates/wukong_codegen_gpu/src/paged_attention.rs:11-62`, `:85`, `:107-111`, `:146-152`, `:164-171`, `:189`, `:203-218` |
| The int8-KV kernel and its scale factoring | same file, `:396-412`, `:490-511` |
| The f64 decode references | same file, `:1220`, `:1246` |
| KV layout, `elem_offset`, scale slab, block size, model geometries, the int8 footprint test | `crates/wukong_codegen_gpu/src/paged_kv.rs:5-52`, `:63`, `:139-158`, `:846-878`, `:934-940` |
| `DecodeLayer`: weight extents, the three WMMA entries, the hardcoded one-warp norm launch | `crates/wukong_codegen_gpu/src/serving.rs:85-93`, `:106-120`, `:336-341`, `:435-450` |
| The serving bench and gate set (all in `serving.rs`'s own test module), incl. the TP splits | same file, `:1534-3450` |
| WMMA 64x64 decode tile constants | `crates/wukong_codegen_gpu/src/ptx_wmma.rs:161-173` |
| Norm planner: `MIN_ELEMS_PER_LANE`, `MAX_WARPS_PER_ROW`, `FILL_CTAS_PER_SM`, `OCCUPANCY_WARPS`, `warps_per_row`, `norm_launch`, the 4050 sweep table | `crates/wukong_codegen_gpu/src/ptx_norm.rs:1-42`, `:56`, `:70`, `:75`, `:87`, `:570-629`, `:667` |
| Megakernel launch layer: grid barrier (its state as a launch parameter at `:89-94`, its PTX at `:102-132`), `MEGA_BLOCK`, `max_resident_ctas`, `plan_grid`, and "still emits the block-scoped form today" | `crates/wukong_codegen_gpu/src/megakernel.rs:26-49`, `:61-65`, `:89-94`, `:102-132`, `:294-358` |
| CUDA graph mechanics and the pool prerequisite | `crates/wukong_codegen_gpu/src/graph.rs:1-30` |
| `RED_GRID`/`RED_BLOCK` **and the determinism rationale for the fixed grid**, the product `gpu::reduce` that consumes it, `stream_cfg`, `hbm_bandwidth` (its reduce arm at `:23538-23544`), `best_bw`, `decode_stack_latency` | `crates/wukong_codegen_gpu/src/gpu.rs:1219-1221`, `:1300-1303`, `:1361-1363`, `:23488-23595`, `:25087`; the offload caller at `crates/wukong_driver/src/gpu_accel.rs:155` |
| The device-free module set (117) and the ptxas census corpus (wmma+flash+wgmma only) | same file, `:8247`, `:8473-8488`, `:8624`; `:22217`, `:22250-22335` |
| FA2 wheel audit and the `fwd_kvcache` requirement; the `::marlin` device dispatcher (its `def` line, the `sm_90` auto-pick, the strawman guards) | `tools/cloud/modal_app.py:1297-1413`, `:3964`, `:4009`, `:4012-4017` |
| The 4050 serving ledger, quarantined in 1.2 | `prompts/results/serving.md:100-107`, `:139-157`, `:197-239`, `:268-271`; `BENCHMARKS.md:51` (the standing-index scope note quoted in the device-scope block), `:366-377` (the results-table scope note, different words), `:395`, `:2466-2490` |
| Ping-pong's trigger, Stream-K's trigger, the 2x2x1 and 192x256 verdicts, the per-tile cost model | `docs/gpu/derive/WAVE3_DOSSIER.md:19-30`, `:1252-1386` |
| The epilogue/fusion break-even model this wave inherits nothing from but must not contradict | `docs/gpu/derive/WAVE4_DOSSIER.md:188-277` |
| The 8-bit `wgmma` transfer condition, the fp8 tolerance, the two-arm exactness split, `FAST_ACCUM` fairness | `docs/gpu/derive/WAVE5_DOSSIER.md:28-38`, `:388-411`, `:565-588`, `:823-845` |
| Wave-6 brief, held-out list, target table, standing rules; and the measured `BW_L2` + cuBLAS's measured 838.7 TFLOP/s at sq4096 | `docs/gpu/derive/ACT2_WAVE_PLAN.md:168-178`, `:182-189`, `:195-205`, `:209-215`, `:5` |
| Gate rules, single-owner rule, staged peers, measured baseline | `CAMPAIGN_CHECKPOINT.md:12-30`, `:38-71`, `:107-117` |

### External (FACT(ext)) -- cited for the SHAPE of a construction, never for its size

- **PagedAttention / vLLM** -- the block-table decode design this module implements, and the reason
  vLLM's v1/v2 op needs a split past 8192 tokens (which this kernel's absent SMEM logits buffer
  avoids): Kwon et al., *Efficient Memory Management for Large Language Model Serving with
  PagedAttention*, SOSP 2023, <https://arxiv.org/abs/2309.06180>.
- **Orca / continuous batching** -- the iteration-level scheduling `Scheduler` implements: Yu et
  al., OSDI 2022.
- **FlashAttention-2 `flash_attn_with_kvcache`** -- the decode bar's API surface (a `block_table`
  argument and a paged cache), <https://github.com/Dao-AILab/flash-attention>. Its *utilisation*
  figures are deliberately not used as a size anywhere above.
- **CUDA cooperative launch residency** -- why `plan_grid` must query
  `cuOccupancyMaxActiveBlocksPerMultiprocessor` rather than assume: CUDA Programming Guide,
  Cooperative Groups / grid synchronisation.
- Where the plan cites QuACK's 3.01 TB/s norm figure, FlashInfer's "40%+ bandwidth utilization", the
  megakernel's "78% of BW / ~1.3 us per kernel", and vLLM's "ITL slope 54% of BF16": all four are
  **sizing** claims and this dossier declines every one of them. They are recorded here so a later
  reader can see they were considered and refused, not overlooked.

### Open, and deliberately left open

* Section 12.4's eight items, seven of which close inside one container.
* The 4050's SM count in the two `prompts/results/serving.md` sentences is **closed** -- both now
  read the probed 20 and cite the probe (`:149-152`, `:161-164`). **One site is still open and it is
  a Rust file:** `crates/wukong_codegen_gpu/src/paged_attention.rs:84`, "which left 39/40 SMs idle".
  Nothing above depends on it, and a docs-only commit must not touch it: it needs the full five-part
  gate, so it belongs to whichever Rust commit next opens that file -- section 5's `v4` rewrite is
  the obvious one.
* Whether a `wgmma` decode GEMM is worth wiring at all, given that at `M <= 338` the roof is
  `M x BW` and the 64x64 WMMA tile already reaches it whenever the launch has enough CTAs. **That
  question is answered by `d3_gemmbw` and by nothing else**, and it is worth asking before Wave 4's
  recognizer-to-offload wiring is extended to the serving path.
