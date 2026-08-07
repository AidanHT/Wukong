# D2 — H100 attention dossier (Wave 0, $0, no device)

Agent D2, 2026-08-06. Read-only derivation for the Wukong GPU retarget campaign
(`GPU_RETARGET_PLAN.md` §0, §2.3, §2.4, §5 Phase 3, §8 risk 5).

Every number is tagged **FACT** (with a source), **DERIVED** (arithmetic from FACTs, shown), or
**PREDICTION** (falsifiable, with the experiment that falsifies it).

---

## 1. FACTS

### 1.1 Hardware limits

| Quantity | Ada `sm_89` (RTX 4050 / L4 / L40S) | Hopper `sm_90` (H100) | Source |
|---|---|---|---|
| SMEM capacity / SM | **100 KB** | **228 KB** (+128%) | FACT — [Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html), [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) |
| Max SMEM / thread block | **99 KB** | **227 KB** | FACT — same |
| **Static** SMEM / block | 48 KB | **48 KB (unchanged)** | FACT — Hopper Tuning Guide: "static shared memory allocations remain limited to 48 KB, and an explicit opt-in is also required to enable dynamic allocations above this limit" |
| Max thread blocks / SM | **24** | **32** (+33%) | FACT — same |
| Max warps / SM | **48** | **64** (+33%) | FACT — same |
| Register file / SM | **64K × 32-bit** | **64K × 32-bit (UNCHANGED, +0%)** | FACT — both tuning guides |
| Max registers / thread | 255 | 255 | FACT — same |
| Unified L1/SMEM/tex cache | 128 KB | **256 KB** | FACT — Hopper Tuning Guide ("from 192 KB in Ampere") |
| SMEM carveout granularity | — | 0/8/16/32/64/100/132/164/196/**228 KB** | FACT — Hopper Tuning Guide, `cudaFuncAttributePreferredSharedMemoryCarveout` |
| SMs | 20 (4050) / 58 (L4) / 142 (L40S) | **132** | FACT — plan §3, [H100 whitepaper](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf) |
| FP16 tensor dense peak | ~24 TF/s (4050, est.) | **989 TFLOPS** | FACT (H100) — [PyTorch FA3 blog](https://pytorch.org/blog/flashattention-3/): "989 TFLOPS with FP16 and 1978 TFLOPS with FP8" |
| SFU (special function) throughput | 16 ops/SM/clk | 16 ops/SM/clk → **3.9 TFLOPS** on H100 | FACT — FA3 blog note 2 ("16 × 132 SMs × 1830 MHz") |
| Clusters / distributed SMEM | no | yes, portable size 8 (16 non-portable) | FACT — Hopper Tuning Guide |

**The single most load-bearing FACT for this dossier: the register file did not grow.** SMEM +128%,
warps/SM +33%, blocks/SM +33%, **registers/SM +0%**. Any "H100 has more of everything" reasoning in
the plan must be qualified by this.

### 1.2 mma.sync vs wgmma ceiling

- FACT — FA3 blog, note 1: *"Without the wgmma instruction, the older `mma.sync` instruction can only
  reach about ⅔ the peak throughput of Hopper Tensor Cores"* (citing
  [arXiv:2402.13499](https://arxiv.org/abs/2402.13499v1)). ⇒ hard ceiling ≈ **660 TFLOPS** fp16 for
  any `mma.sync`-only kernel on H100.
- FACT — FA3 blog: **FlashAttention-2 achieves only 35% of theoretical max FLOPs on H100**
  (≈346 TFLOPS), vs "up to 70% on A100". FA2 is a well-tuned `mma.sync` + `cp.async` kernel — i.e.
  exactly Wukong's programming model. **FA2-on-H100 is the honest ceiling for the Wukong flash
  family as it stands.**

### 1.3 FlashAttention-2 design summary

FACT — [FA2 paper/blog](https://tridao.me/publications/flash2/flash2.pdf),
[HazyResearch blog](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2):
1. Reduce non-matmul FLOPs (matmul throughput is up to 16× non-matmul).
2. **Parallelize over the sequence-length dimension** in addition to batch × heads — explicitly "in
   the case of long sequences (which usually means small batch sizes or small number of heads) ... to
   make better use of the multiprocessors on the GPU". This is the lever Wukong does **not** have.
3. Better warp partitioning inside a CTA: **split Q across 4 warps, keep K and V accessible by all
   warps** (vs FA1's split-K), which removes cross-warp synchronization and SMEM round-trips.
4. Result: 50–73% of theoretical peak on A100; 225 TFLOPS/A100 end-to-end GPT training.
5. FACT — FA2 supports MQA/GQA ("multiple heads of query attend to the same head of key and value").

### 1.4 FlashAttention-3 design summary (the H100 peer)

FACT — [FA3 paper](https://tridao.me/publications/flash3/flash3.pdf),
[PyTorch blog](https://pytorch.org/blog/flashattention-3/), [Tri Dao blog](https://tridao.me/blog/2024/flash3/):
1. **WGMMA** — asynchronous warpgroup MMA; the only path to full Hopper tensor throughput.
2. **TMA** — dedicated hardware for global↔shared tensor copies, handles indexing and OOB predication;
   "frees up registers, which is a valuable resource to increase tile size and efficiency".
3. **Producer/consumer warp specialization** — a producer warpgroup issues TMA and *deallocates*
   registers with `setmaxnreg`; consumer warpgroups reallocate them and run WGMMA.
4. **Pingpong scheduling** (inter-warpgroup) — `bar.sync`-driven alternation so one warpgroup's softmax
   hides in the other's GEMM. Worth **570 → 620 TFLOPS** (hdim 128, seqlen 8K, FP16).
5. **Intra-warpgroup GEMM/softmax pipelining** — worth **620 → 640–660 TFLOPS**, at higher register cost.
6. **FP8** with block quantization + incoherent processing (Hadamard with random signs), 2.6× lower
   quantization error.
7. Measured: **1.5–2.0× faster than FA2 forward** (1.5–1.75× backward); **up to 740 TFLOPS FP16 (75%
   utilization)**; **~1.2 PFLOPS FP8**. Ablation at batch 4 / seqlen 8448 / 16 heads / hdim 128 /
   non-causal: FA3 661 TFLOPS, −GEMM-softmax-pipelining 582, −warp-specialization 570.
8. FACT — benchmark grid: seqlen 512/1k/2k/4k/8k/16k, head dims 64/128/**256**, total tokens fixed at
   16k. FA3 "surpasses cuDNN for medium and long sequences (1k and above)" and is 1.5× Triton FA2.
9. FACT — FA3 supports head dims **up to 256** and MQA/GQA
   ([Dao-AILab/flash-attention](https://github.com/dao-ailab/flash-attention)).

### 1.5 PyTorch SDPA backend selection on H100

- FACT — [PyTorch SDPA docs](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html):
  backends are **FLASH_ATTENTION (FA2)**, **EFFICIENT_ATTENTION (xformers mem-efficient)**,
  **CUDNN_ATTENTION**, **MATH**; "SDPA attempts to automatically select the most optimal
  implementation based on the inputs"; overridable via `torch.nn.attention.sdpa_kernel(...)`.
- FACT — `enable_gqa` is **experimental**, "works only for Flash_attention and math kernel on CUDA
  tensor", requires `n_heads_q % n_heads_kv == 0` and `n_heads_k == n_heads_v`.
- FACT — on H100 SDPA dispatches `SDPBackend.FLASH_ATTENTION` **by default**, but manually selecting
  `SDPBackend.CUDNN_ATTENTION` has been reported to give ~10% end-to-end improvement
  ([pytorch/torchtitan#1042](https://github.com/pytorch/torchtitan/issues/1042)).
  ⇒ **A bare `F.scaled_dot_product_attention` peer on H100 is FA2-class, NOT FA3-class.** Publishing
  against the default is publishing against the weaker of the two available peers. Phase 3 must pin
  the backend explicitly and report *both*.

### 1.6 cuDNN attention (frontend v9+)

FACT — [cudnn-frontend Attention docs](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md),
[cuDNN 9 blog](https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/):
- Head dims: Ampere/Ada prefill **d ≤ 256**, decode/bprop **d ≤ 128**; Hopper/Blackwell prefill and
  decode **d ≤ 256** (also d_qk=192 / d_vo=128). d must be a multiple of 8 (fp16/bf16), 16 (fp8).
- Architectures: SM80 / SM89 / SM90 / SM100 / SM110.
- **"All attention flavors MHA, MQA, GQA are supported"** — Q/K/V may have independent head counts.
- Masking: causal (top-left **and bottom-right** alignment), sliding window (≥ 9.2.0), padding,
  bias, ALiBi, 128×128 block mask.
- **Paged attention** supported via `paged_cache_load` (Ampere + Hopper). Backward does **not**
  support paged attention.
- FP8: Hopper+, causal/padding masks only, explicit scale/descale tensors; up to **1.2 PFLOPs** at
  d=256 non-causal forward.
- Claimed up to 2× PyTorch eager in BF16, 3× in FP8.

### 1.7 Wukong's current flash geometry (FACT, from source)

`crates/wukong_codegen_gpu/src/ptx_flash.rs`, `.../gpu.rs`:

| entry | shape | SMEM/CTA | warps/CTA | grid | dispatched? |
|---|---|---|---|---|---|
| `flash_d{32,64,128}` | f32, 1 warp per query **row**, no SMEM | 0 | `FLASH_WARPS`=2 | `ceil(S/2)` | no (`FLASH_TILE_MIN=0`) |
| `flash_d{32,64,128}_t` | f32, 1 warp/row, `BK=1024/D` keys staged | **8 KB static** | `FLASH_TWARPS`=8 | `ceil(S/8)` | yes, S < 512 |
| `flash_d64_mp` | fp16 `mma.sync.m16n8k16`, **1 warp per 16 query rows**, O/m/l in registers, `cp.async` 2-stage K/V, Bc=16 | **8 KB** = 128·D | **1** | `(S/16, heads)` | yes, D=64, S ≥ 512, S < 4096 |
| `flash_d128_mp_lm` | same + `ldmatrix.x2{,.trans}` feed | **16 KB** | **1** | `(S/16, heads)` | yes, D=128, S ≥ 512, S < 4096 |
| `flash_d64_ws` | warp-specialized ping-pong, 2 disjoint query tiles, named barriers 1/2 | 8 KB | **2** | `(ceil(S/32), heads)` | yes, D=64, S ≥ 4096 |
| `flash_d128_ws3_lm` | ws + 3-stage `cp.async` ring | **24 KB** | 2 | `(ceil(S/32), heads)` | yes, D=128, S ≥ 4096 |
| `flash_d64_m1` | single-buffer probe | 4 KB | 1 | — | no (diagnostic) |
| `flash_d64_mp{4,8}` | multi-warp CTA sharing one staged K/V | 8 KB | 4 / 8 | `(S/(16W), heads)` | no (shelved) |
| `flash_d64_mpw{2,4}` | wide key tile Bc = 32 / 64 | 16 / 32 KB | 1 | `(S/16, heads)` | no (shelved) |
| `flash_d64_msp` | QKᵀ-ahead software pipeline, separate K/V pools | 8 KB | 1 | — | no (shelved) |
| `flash_d128_hs` | head-dim warp split, 2 warps / same 16 rows | 16 KB | 2 | — | no (shelved) |
| `flash_d64_mprope` | fused RoPE | 8 KB | 1 | — | yes (RoPE path) |

- SMEM formula (FACT, from the generators): `stages × 2 × Bc × D × sizeof(dtype)`. With Bc=16,
  fp16, 2 stages ⇒ **128·D bytes** (8 KB @ D=64, 16 KB @ D=128); ws3 uses 3 stages ⇒ 192·D.
- `SUPPORTED_D = [32, 64, 128]`; tensor-core flash gated to **D ∈ {64, 128}** (`ptx_flash.rs:2763-2838`).
- `flash_ptx()` header is **`.version 7.8` / `.target sm_89`** — one module for all entries.
- **No batch dimension**: `grid_dim.1 = heads`, `grid_dim.2` unused; `[H,S,D]` base folded from `ctaid.y`.
- SMEM is **static** everywhere: every `LaunchConfig` has `shared_mem_bytes: 0`; no `cuFuncSetAttribute`
  anywhere in the crate (plan §2.3, verifier-confirmed).
- Attention backward is the **materialized O(S²)** form (`ptx_autodiff_bwd.rs:10, 1246-1247`), not fused.
- `paged_kv.rs:64`: `heads` documented as "KV heads (== query heads here; GQA/MQA would make this smaller)".

### 1.8 The frozen 4050 A/B verdicts (FACT, from code + BENCHMARKS.md)

| A/B | verdict on RTX 4050 (20 SMs, 100 KB SMEM, 48 warps/SM) |
|---|---|
| `flash_tiled_vs_untiled` | tiled wins/ties at every S (0.56/0.90/0.73/0.80/1.02× @256/512/1024/2048/4096) ⇒ `FLASH_TILE_MIN = 0` |
| `flash_pipe_vs_mma` | `mp/m` 0.28→0.88× single-head S=512→4096; 0.84–0.90× at H=12 |
| `flash_lm_vs_mp` | `_lm` wins D=128 by 18–23%, D=64 by 2–4% |
| `flash_mw_vs_mp` / `flash_mp4_vs_mp` | **mp4/mp8 only marginal: ~6–8% at H=12, a regression single-head** ⇒ "not strongly K/V-bandwidth-bound" |
| `flash_wide_vs_mp` | **mpw2/mp ≈ 0.65–0.91×** (10–35% slower), worst at H=8/S=4096 |
| `flash_sp_vs_mp` | **msp = wash-to-loss: 1.043 / 1.044 / 1.010 / 1.019** @ S=512/1k/2k/4k |
| `flash_hs_vs_mp` | **hs = 11–20% slower: 1.110 / 1.108 / 1.199 / 1.193** @ S=512/1k/2k/4k, D=128 — despite occupancy doubling 6→12 warps/SM |
| `flash_single_vs_double` | m1 (50% occupancy, no overlap) loses to mp (25% occupancy, overlapped) |
| `flash_ws_vs_mp` (median-of-9, clock-cancelled) | **ws wins ONLY at S ≥ 4096** (d64 `ws` 0.944×, d128 `ws3_lm` 0.950×); **tie @2048** (1.00–1.02×); **10–52% LOSS @ S ≤ 1024** |
| vs cuDNN | Wukong flash is **0.46× / 0.43× of cuDNN at S=2048 / 4096, D=64** (~9.5 TF/s vs cuDNN's ~20); wins short/causal/RoPE regimes; beats cutlass mem-efficient fMHA at D=128, S ≤ 1024 (1.11–1.20×) |
| multi-head saturation | flat **~860 GFLOP/s across S** at H=12 (f32 path); single-head climbs 117→438 |

---

## 2. DERIVED

All arithmetic below reproduces the code's own stated occupancy numbers on Ada exactly (12 CTAs/SM
for `mp`, 24 for `m1`, 6 warps for D=128, 24 for `ws`, 16/8 for `ws3`), which validates the model
before it is applied to H100.

### 2.1 Occupancy model

```
warps/SM = min( floor(SMEM_SM / smem_cta) × warps_cta ,     # SMEM bound
                blocks_cap × warps_cta ,                     # 24 (Ada) / 32 (H100)
                floor(65536 / (32 × R)) ,                    # register bound (SAME on both)
                warps_cap )                                  # 48 (Ada) / 64 (H100)
```
with `SMEM_SM` = 102,400 B (Ada) / 233,472 B (H100) and `R` = registers/thread.

### 2.2 Existing kernels: Ada vs H100 (DERIVED, ignoring registers)

| entry | SMEM/CTA | w/CTA | Ada CTAs/SM | **Ada warps/SM (of 48)** | H100 CTAs/SM | **H100 warps/SM (of 64)** |
|---|---|---|---|---|---|---|
| `flash_d64_mp` | 8 KB | 1 | 12 | **12 (25%)** ✓ matches code | 28 | **28 (43.8%)** |
| `flash_d128_mp_lm` | 16 KB | 1 | 6 | **6 (12.5%)** ✓ | 14 | **14 (21.9%)** |
| `flash_d64_ws` | 8 KB | 2 | 12 | **24 (50%)** ✓ | 28 | **56 (87.5%)** |
| `flash_d128_ws_lm` | 16 KB | 2 | 6 | **12 (25%)** ✓ | 14 | **28 (43.8%)** |
| `flash_d64_ws3` | 12 KB | 2 | 8 | **16 (33%)** ✓ | 19 | **38 (59.4%)** |
| `flash_d128_ws3_lm` | 24 KB | 2 | 4 | **8 (16.7%)** ✓ | 9 | **18 (28.1%)** |
| `flash_d64_m1` | 4 KB | 1 | 24 (block-capped) | **24 (50%)** ✓ | 32 (block-capped) | **32 (50%)** |
| `flash_d64_mp4` | 8 KB | 4 | 12→**12 (block)** ⇒ 48 | **48 (100%)** | 16 (warp-capped) | **64 (100%)** |
| `flash_d64_mp8` | 8 KB | 8 | 6 (warp-capped) | **48 (100%)** | 8 (warp-capped) | **64 (100%)** |
| `flash_d64_mpw2` | 16 KB | 1 | 6 | **6 (12.5%)** | 14 | **14 (21.9%)** |
| `flash_d64_t` (f32) | 8 KB | 8 | 6 (warp-capped) | **48 (100%)** | 8 (warp-capped) | **64 (100%)** |

**DERIVED finding A — the 32-blocks/SM cap pins every 1-warp-per-CTA kernel at ≤ 32 warps/SM = 50%
on H100.** The production dispatch (`flash_d64_mp`, `flash_d128_mp_lm`) is 1 warp/CTA. Even with
*infinite* SMEM it can never exceed half of H100's warp slots. Warps-per-CTA — not SMEM — is the
knob that converts H100's headroom into occupancy. `GPU_RETARGET_PLAN.md` §2.4 flags
`FLASH_WARPS = 2` for re-tuning, but `FLASH_WARPS` governs the **untiled f32 kernel that is never
dispatched** (`FLASH_TILE_MIN = 0`). Re-tuning it is worth nothing. **The constant that matters is
the implicit `warps_cta = 1` in `wmma_flash_cfg` (`gpu.rs:2151` region, `block_dim: (32,1,1)`).**

### 2.3 The register wall (DERIVED)

Registers/SM is 65,536 on **both** architectures. Warps/SM ≤ `floor(65536 / (32·R))`:

| R (regs/thread) | max warps/SM | as % of Ada's 48 | as % of H100's 64 |
|---|---|---|---|
| 40 | 51 | 100% | 80% |
| 64 | 32 | 67% | 50% |
| 80 | 25 | 52% | 39% |
| **120** (D=128 `mp`, stated in `ptx_flash.rs:2820` comment) | **17** | 35% | **27%** |
| 168 | 12 | 25% | 19% |
| 255 | 8 | 17% | 13% |

**DERIVED finding B — on Ada, SMEM was the binding constraint; on H100 it flips to the register
file.** For `flash_d128_mp_lm` (R ≈ 120): Ada `min(6 SMEM, 17 reg) = 6` → SMEM binds. H100
`min(14 SMEM, 17 reg) = 14` → SMEM still just binds, but only barely; and for `flash_d64_ws`
(2 warps/CTA, R unknown ≈ 70–90): Ada `min(24, ~25) = 24`, H100 `min(56, ~25) = 25` → **register
bound**. The nominal 87.5% occupancy in §2.2 is unreachable.

**Consequence: on H100 the correct flash lever is register economy and warps-per-CTA, not shared
memory.** Which is exactly what FA3 does with `setmaxnreg` (producer warpgroups drop registers,
consumers claim them) and TMA (removes address-arithmetic registers). Wukong has neither
(`GPU_RETARGET_PLAN.md` §2.3: zero occurrences of `wgmma`/`TMA`/`setmaxnreg` crate-wide).

**R is UNMEASURED for every entry.** Measuring it costs $0 (see experiment #1 in §4.2).

### 2.4 SMEM budget as a function of (Br, Bc, stages, dtype)

Staged-K/V SMEM = `stages × 2 × Bc × D × e`; add `Br × D × e` if the Q tile is also staged.
fp16 (`e = 2`), in KB:

| D | Bc | 1 stage | 2 stages | 3 stages | 4 stages |
|---|---|---|---|---|---|
| 64 | **16 (today)** | 4 | **8** | 12 | 16 |
| 64 | 32 | 8 | 16 | 24 | 32 |
| 64 | 64 | 16 | 32 | **48** | 64 |
| 64 | 128 | 32 | 64 | 96 | 128 † |
| 64 | 256 | 64 | 128 † | 192 † | 256 ✗ |
| 128 | **16 (today)** | 8 | **16** | 24 | 32 |
| 128 | 32 | 16 | 32 | **48** | 64 |
| 128 | 64 | 32 | 64 | 96 | 128 † |
| 128 | 128 | 64 | 128 † | 192 † | 256 ✗ |
| 256 | 16 | 16 | **32** | **48** | 64 |
| 256 | 32 | 32 | 64 | 96 | 128 † |
| 256 | 64 | 64 | 128 † | 192 † | 256 ✗ |

† = fits H100's 227 KB but **not** Ada's 99 KB (H100-only, dynamic SMEM required).
✗ = fits nowhere. **Bold** = fits under the **48 KB static** cap that applies to *both* architectures
and to Wukong's code as written today.

**DERIVED finding C — the flash family does NOT need the dynamic-SMEM keystone to get a much bigger
tile.** Under the unchanged 48 KB static cap, Wukong can already go from Bc=16/2-stage to
**Bc=64/3-stage at D=64**, **Bc=32/3-stage at D=128**, or **Bc=16/3-stage at D=256** — a 4× / 2× tile
widening and a deeper pipeline, with no `cuFuncSetAttribute` plumbing at all. Dynamic SMEM is the
*GEMM* keystone (plan §2.3's eight 48 KiB asserts); for flash it only unlocks the ≥64 KB regime,
which §2.3's register wall says is the wrong direction anyway (1–2 CTAs/SM at 4–8 warps each = 8–16
warps/SM, only viable with async instructions).

**DERIVED finding D — D=256 was never SMEM-blocked, on either architecture.** At Bc=16, 2 stages,
D=256 the staging buffer is **32 KB — under even the 48 KB *static* cap**, and it fit Ada's 99 KB
too. The actual blocker is register pressure: `nto = D/8 = 32` PV n-tiles × 4 f32 accumulators =
**128 O-accumulator registers per thread** before Q fragments, scores, or addresses. At R ≥ 128 the
register model caps occupancy at ≤ 16 warps/SM. `GPU_RETARGET_PLAN.md` §2.3 lists the "Flash D-set"
gap in the same table as dynamic SMEM, implying a capacity problem — **it is a register/design
problem, and it is device-free work available today.**

### 2.5 What 227 KB actually permits that ~100 KB forbade

Purely as capacity (ignoring whether it is wise):
- **D=64**: Bc=128 @ 4 stages (128 KB), Bc=256 @ 2–3 stages (128/192 KB). Ada topped out at
  Bc=128 @ 3 stages (96 KB) with nothing left for a Q tile.
- **D=128**: Bc=64 @ 4 stages, **Bc=128 @ 2–3 stages (128/192 KB)** — this is the FA2/FA3 tile shape
  and it is the headline unlock. Ada could not stage a 128-key block at D=128 at any depth beyond 1.
- **D=256**: Bc=64 @ 2–3 stages (128/192 KB). Ada capped at Bc=32 @ 3 stages.
- Occupancy cost of using it: a 192 KB CTA is **1 CTA/SM**. With 4 warps that is 8 warps/SM (12.5%);
  with 8 warps, 16 warps/SM (25%). This only pays if the CTA's work is async-overlapped, i.e.
  wgmma + TMA. **DERIVED: 227 KB is a Phase-4 (wgmma) asset, not a Phase-3 (Act 1) asset.**

### 2.6 CTA supply — where the 20-SM verdicts invert (plan risk 5, quantified)

Warps offered by the current `mp` dispatch = `(S/16) × H` (1 warp/CTA, no batch dim). Compare against
warps the machine can *hold* (§2.2, register-capped to ~25 on H100, 12 on Ada):

| shape (D=64, `flash_d64_mp`) | warps offered | 4050: warps/SM offered vs 12 held | H100: warps/SM offered vs ~25 held |
|---|---|---|---|
| S=512, H=1 | 32 | 1.6 — underfilled | **0.24 — 1% of capacity** |
| S=512, H=12 (GPT-2) | 384 | **19.2 — OVERSUBSCRIBED (saturated)** | **2.9 — 12% of capacity** |
| S=512, H=32 (Llama-8B) | 1024 | 51 — oversubscribed | 7.8 — 31% |
| S=2048, H=12 | 1536 | 77 — oversubscribed | 11.6 — 47% |
| S=4096, H=12 | 3072 | 154 — oversubscribed | 23.3 — 93% |
| **S=8192, H=12** | 6144 | 307 | **46.5 — fully filled** |

**DERIVED finding E — the inversion is exact and it is a factor of 6.6× (132/20).** Every shape the
4050 called "saturated" at H=12 (the regime BENCHMARKS.md reports the flat ~860 GFLOP/s plateau in)
supplies **12% of H100's capacity at S=512** and only reaches full supply at **S ≈ 8192**. The
4050's small-S wins were measured on a machine that a 384-warp grid oversubscribes 1.6×; H100 needs
~3,300 warps for the same kernel.

**DERIVED finding F — the "structural on 20 SMs" long-S loss dissolves *as an occupancy claim* at
S ≈ 8192, H ≥ 12, and nowhere else.** That makes `S=8192, D=64, H=12–16, non-causal` the single
cleanest diagnostic shape in the whole campaign: it is the only place where the machine is genuinely
full, so the residual gap vs cuDNN/FA3 is *purely* instruction mix and scheduling — i.e. it **is**
the wgmma business case number the Phase-4 go/no-go memo needs (plan §5 Phase 4.1(b)).

**DERIVED finding G — the S ≥ 512 tensor-core floor (`wmma_flash_applies`) should MOVE UP on H100,
not down.** The f32 `flash_d64_t` kernel is 1 warp per query *row* (S warps total, 8 per CTA); the
tensor-core `mp` is 1 warp per *16* rows (S/16 warps, 1 per CTA). At S=512, H=1 that is 512 warps vs
32 — a **16× parallelism difference**. On 20 SMs the tensor-core path still won above S=512 because
32 warps ≈ 1.6/SM was tolerable; on 132 SMs 32 warps is 0.24/SM and the f32 path's 3.9 warps/SM may
well win despite doing 8× the arithmetic per FLOP. This is a counterintuitive, cheap, testable
inversion.

**DERIVED finding H — Wukong has no seqlen-K parallelism (no split-KV / "flash-decoding") and no
batch dimension in the grid.** FA2's item 2 (§1.3) exists precisely for the small-batch/few-head
long-sequence case. On 132 SMs this is the difference between 12% and 100% machine utilization at
S ≤ 2048. Both are device-free code changes: `grid.z = batch` is nearly free; split-KV needs a
second reduction pass over per-split `(m, l, O)` partials.

### 2.7 Where the 2-warp `ws` family stands on H100 (DERIVED)

Two opposing mechanisms, and they act at opposite ends of S:
- **Favouring `ws` at large S**: `mp` is block-capped at 32 warps/SM (finding A); `ws` is not. `ws`
  also halves K/V L2 traffic and H100's L2 is **50 MB vs the 4050's 12 MB** (plan §3), so the reuse
  is worth more, not less.
- **Favouring `mp` at small S**: `ws` grid is `ceil((S/16)/2)` — **half the CTAs**. On a 132-SM part
  the small-S underfill (finding E) is already the dominant problem; halving the CTA count makes it
  strictly worse.

**DERIVED: the ws crossover should move DOWN (win regime widens) and its small-S loss should get
WORSE.** The 4050's S ≥ 4096 threshold is a 20-SM artifact in both directions. `ws_flash_route_with`
(`gpu.rs` ~2140) must be re-derived per device, never ported.

Also: Wukong's `ws` is a **symmetric** ping-pong (both warps do QK + softmax + PV, phase-locked
anti-phase). FA3's is **asymmetric producer/consumer** (§1.4 item 3). On Ada the symmetric form was
the only option — there was no `setmaxnreg` and no TMA to make asymmetry pay. On Hopper the
asymmetric form is what the hardware rewards. Wukong's `ws` is therefore the right *scaffold* and
the wrong *shape*.

---

## 3. VERDICT LIST — shelved variants under H100 conditions

| variant | 4050 verdict (FACT, from code comments) | H100 re-open? | mechanism |
|---|---|---|---|
| **`mp4` / `mp8`** (4/8 warps per CTA sharing one staged K/V block; `entry_mma_reg_pipe_mw`) | "only marginal — ~6–8% at the H=12 filled regime and a *regression* single-head" (BENCHMARKS.md:2119-2122); "the `mp4` wash proved this kernel is tensor-core-**feed**-bound, not occupancy-bound" (`ptx_flash.rs:2788`); the `ws` doc calls mp4/mp8 "measured DEAD" (`ptx_flash.rs:2206-2209`) | **YES — highest priority** | (1) **It is the only existing variant that can exceed 32 warps/SM** — finding A shows every 1-warp-CTA kernel is block-capped at 50% occupancy on H100. mp4 reaches 64 warps/SM nominally (register-capped to ~25, still ≥ mp). (2) The single-head regression was a *20-SM grid-granularity* artifact (`grid = S/64`), not a design fault; at 132 SMs, single-head is hopeless for every variant anyway (finding E), so the regression regime is no longer a dispatch regime. (3) 4× the K/V L2 reuse against a 4.2× larger L2. (4) **4 warps = 128 threads = exactly one Hopper warpgroup** — mp4 is the natural scaffold for a wgmma port (same CTA shape, same cooperative staging, swap the mma). (5) Independently re-opened by GQA (§5). |
| **`mpw2` / `mpw4`** (wide key tile Bc=32/64; `entry_mma_reg_pipe_wide`) | **LOSS: `mpw2/mp` ≈ 0.65–0.91×**, worse at large S (~0.65× @ H=8/S=4096). Stated cause: "the wider tile's extra score/P **registers** and `nkb`× **SMEM** cut occupancy by more than the amortisation saves" (`ptx_flash.rs:1795-1801`) | **PARTIALLY — only fused with mp4** | The loss had two components. The **SMEM** component is relieved (mpw2 at 16 KB: 6 CTAs/SM on Ada → 14 on H100). The **register** component is not relieved at all (finding B: 64K regs/SM on both) and is now the binding one. At 1 warp/CTA the SMEM relief cannot reach the block cap anyway. Expect the loss to shrink toward a tie without inverting. It becomes interesting only as `mp4 × mpw` (4 warps × Bc=32–64 = Br 64 / Bc 64 — the FA2 tile), which is a new generator, not a re-run. The kernel's own doc already anticipates this: "retained only to localise the comparison should a future GPU change the occupancy/softmax balance." |
| **`msp`** (QKᵀ of tile *i+1* hoisted ahead of softmax of tile *i*; separate K/V pools, **same SMEM/occupancy**) | **WASH-to-LOSS: 1.043 / 1.044 / 1.010 / 1.019** at S=512/1k/2k/4k. Cause offered: `ptxas` already extracts what one in-order warp affords, or the rotation `mov`s cost what the overlap saves (`ptx_flash.rs:1189-1196`) | **NO now; YES only after wgmma** | Nothing in H100's *synchronous* `mma.sync` path changes the verdict — the failure is that one in-order warp cannot overlap a synchronous mma with an SFU stretch. But **FA3's "intra-warpgroup overlapping of GEMM and softmax" is exactly msp's idea done with asynchronous wgmma, and FA3 measures it worth 620 → 640–660 TFLOPS** (§1.4 item 5). So msp's *hypothesis* is vindicated by the H100 peer and its *implementation* is what was wrong. Re-open as a Phase-4 design input, not a Phase-3 A/B. Note H100's SFU throughput is 3.9 TFLOPS against 989 TFLOPS of matmul (**256:1**, FACT §1.1) — the softmax stall gets structurally *worse* on Hopper, so the lever matters more, not less. |
| **`hs`** (head-dim warp split at D=128: 2 warps, same 16 query rows, each owns half the output hdim ⇒ 32 not 64 O-accumulators) | **LOSS: 1.110 / 1.108 / 1.199 / 1.193** (11–20% slower) at S=512/1k/2k/4k, *despite* occupancy doubling 6→12 warps/SM. Cause: redundant full QKᵀ in both warps + 2 `bar.sync`/tile cost more than the extra warps buy (`ptx_flash.rs:1406-1414`) | **NO at D=128; YES as the D=256 enabler** | At D=128 on H100, `mp_lm` already gets 14 warps/SM and the register cap is ~17 — the split has almost nothing left to buy, while the redundant QKᵀ cost is unchanged. **At D=256 the calculus inverts**: 128 O-accumulators/thread (finding D) makes registers unambiguously binding, and `hs` is precisely a register-halving transform. The doc's own dismissal of the contraction-split (no redundant QKᵀ, but a cross-warp partial-score exchange) was made on D=128 evidence and does not transfer to D=256. **The `hs` code is the D=256 unlock, re-scoped.** |
| **`m1`** (single-buffered occupancy probe, 4 KB, 24 CTAs/SM = 50% on Ada) | Diagnostic only, never dispatched: the double-buffered `mp` at 25% occupancy beat it at 50% ⇒ pipeline overlap > occupancy on Ada (`ptx_flash.rs:892-897`, `:2780-2783`) | **NO as a dispatch; its diagnostic value INVERTS — keep it** | On Ada m1 had a 2× occupancy advantage (24 vs 12 warps). On H100 the block cap flattens it: m1 = min(57, **32**) = 32 warps, mp = 28 → only a 1.14× advantage. So m1 becomes a clean **control that proves the 32-block cap is binding**: if `m1/mp` on H100 is much closer to 1.0 than on the 4050, finding A is confirmed with no profiler. Cheap, high-signal, already written. |
| **`ws` / `ws3`** (warp-specialized ping-pong, 2 warps/CTA on disjoint query tiles; ws3 = 3-stage ring) | **WIN only at S ≥ 4096** (d64 `ws` 0.944×, d128 `ws3_lm` 0.950×); **tie @2048**; **10–52% LOSS at S ≤ 1024**. Default-routed at S≥4096 (`gpu.rs` `ws_flash_route_with`); a provisional S≥2048/S≥1024 regime "would have shipped a regression and is gone" | **YES — the win regime should widen at large S and the loss deepen at small S** | §2.7. Two opposing mechanisms: it is the only *dispatched* family that escapes the 32-block warp cap (favours ws everywhere), but it halves the CTA count (hurts precisely where H100 is already starved). The 3-stage `ws3` trade — "the ring trades occupancy for latency slack, and `flash_ws_vs_mp` measures which side of that trade this GPU is on" — is exactly the trade H100 changes: ws3 D=64 goes 16 → 38 warps/SM, ws3 D=128 goes 8 → 18. **`ws3` is the variant whose Ada verdict was most SMEM-limited and therefore the most likely to flip.** Also: FA3 validates warp specialization as its #1 technique (570 → 620 TFLOPS with pingpong, §1.4). Wukong's version is symmetric where Hopper rewards producer/consumer asymmetry. |

---

## 4. PREDICTIONS

### 4.1 Where the Ada flash kernels land on H100

Anchors: Wukong 4050 flash D=64 ≈ **9.5 TF/s** long-S, **0.46× / 0.43× of cuDNN @ S=2048 / 4096**
(FACT); FA2-on-H100 = **35% of peak ≈ 346 TFLOPS** (FACT); FA3 FP16 up to **740 TFLOPS** (FACT);
`mma.sync` ceiling ≈ **⅔ of peak ≈ 660 TFLOPS** (FACT).

All rows below are **PREDICTION** (fp16, `flash_d64_mp` / `flash_d128_mp_lm` / `ws` as routed today,
forward-JITted, no code change, H=12–16, no batch folding):

| shape | Wukong H100 throughput | vs PyTorch-SDPA default (FA2) | vs cuDNN / FA3 | falsified by |
|---|---|---|---|---|
| S=512, D=64, non-causal | **5–25 TFLOPS** (≈3 warps/SM, finding E) | **0.03–0.10×** | **0.02–0.06×** | the shape sweep in exp. #2 |
| S=512, D=64, **causal** | 4–20 TFLOPS | 0.03–0.10× | 0.02–0.06× | same |
| S=2048, D=64 | **40–110 TFLOPS** | 0.12–0.30× | 0.08–0.20× | same |
| S=2048, D=128 | 35–100 TFLOPS | 0.10–0.28× | 0.07–0.18× | same |
| **S=8192, D=64** (machine full) | **80–200 TFLOPS** | **0.25–0.55×** | **0.12–0.33×** | exp. #3 |
| S=8192, D=128 | 70–180 TFLOPS | 0.22–0.50× | 0.11–0.30× | exp. #3 |
| S=8192, D=64 causal | 60–160 TFLOPS | 0.20–0.50× | 0.10–0.30× | exp. #3 |

**THE HEADLINE PREDICTION.** *The flash gap vs the strongest peer gets **worse** on H100 than it was
on the 4050 — roughly 0.43–0.46× of cuDNN → **0.12–0.33×** of cuDNN/FA3 at long S — even though the
machine is now easy to fill.* Mechanism: on Ada, Wukong and cuDNN were both `mma.sync` + `cp.async`
kernels, so the 0.43–0.46× was a *tuning* gap. On Hopper the peer gains wgmma (up to 1.5× the
`mma.sync` ceiling), TMA, warp specialization and pingpong (FA3 ablation: 570 → 661 TFLOPS from
scheduling alone) while Wukong gains only SMs. **Retiring the "structural on 20 SMs" excuse does not
produce a win; it produces a larger, cleaner, correctly-attributed loss — and that loss number is
precisely the Phase-4 wgmma business case.** Plan §5 Phase 3.4 already frames Act 1 this way; this
dossier says the magnitude will be worse than the 4050 ratios suggest, and docs should be re-scoped
in that direction *before* the measurement, not after.

**Corollary predictions (all falsifiable in one sweep):**
- P1. **Every 4050 small-S "win" inverts.** The fused-RoPE S≤512 win, the causal-512 win, and the
  D=128 ≤1024 win vs cutlass mem-efficient fMHA all become losses on H100. (Mechanism: finding E;
  cutlass/cuDNN have persistent kernels and split-KV, Wukong has neither.)
- P2. **`mp4` flips from "marginal 6–8%" to a ≥1.25× win over `mp`** at S ≥ 4096, H ≥ 8 on H100
  (mechanism: finding A — mp cannot exceed 32 warps/SM, mp4 can). Falsified by `flash_mw_vs_mp`.
- P3. **The `ws` crossover falls from S ≥ 4096 to S ≈ 1024–2048** at D=64, and the S ≤ 512 loss
  *deepens* beyond the 4050's 10–52%. Falsified by `flash_ws_vs_mp`.
- P4. **`ws3` flips at D=128** (its Ada loss was SMEM-occupancy-driven: 8 warps/SM; H100 gives 18).
- P5. **`mpw2`'s 0.65–0.91× loss narrows to 0.88–1.02× but does not invert** at 1 warp/CTA.
- P6. **`hs` stays a 1.05–1.20× loss at D=128.**
- P7. **`m1/mp` moves from a clear loss toward ~1.0** — the control that proves the block cap binds.
- P8. **The `wmma_flash_applies` S ≥ 512 tensor-core floor rises on H100**, and at S ≤ 512 single-head
  the f32 `flash_d64_t` (16× more warps) may *beat* the tensor-core kernel. Counterintuitive; cheap.
- P9. **`R` (registers/thread) will be ≥ 96 for D=64 `mp` and ≥ 120 for D=128 `mp_lm`**, so measured
  occupancy on H100 will fall short of §2.2's SMEM-only table by 10–50%. Falsified by `ptxas -v`.
- P10. **PTX `.version 7.8` / `.target sm_89` will forward-JIT and run correctly on an H100 driver**
  (plan risk #1) — but performance-relevant: the JIT will target sm_90 without any Hopper-specific
  scheduling, so no PTX-level surprise either way.

### 4.2 Top-5 confirmation experiments, ranked by information value

**#1 — `ptxas -v` register/SMEM census. $0, NO GPU, runs in CI.**
Emit `ptx_flash::flash_ptx()` (there is already a `WUKONG_GPU_DUMP_PTX` knob and a `WUKONG_PTXAS`
hook), retarget the header to `sm_90`, and run `ptxas -v -arch=sm_90 -O3` (plus `sm_89` as control)
over every entry. Record **registers/thread, SMEM/CTA, spill stores/loads** for all ~30 entries.
*Why it ranks first:* finding B says the H100 flash ceiling is set by registers, and **not one
register count is currently known**. This single free step decides tile shape, warps/CTA, whether
`setmaxnreg` is needed, and whether §2.2's occupancy table is real or fantasy. It also flushes out
spills that never showed on Ada. Belongs in Phase 0, alongside the missing gpu-feature CI.

**#2 — The CTA-supply sweep, not a timing sweep (one H100 hour).**
Re-run `flash_ws_vs_mp`, `flash_mw_vs_mp`, `flash_lm_vs_mp`, `flash_single_vs_double`,
`flash_wide_vs_mp`, `flash_hs_vs_mp`, `flash_sp_vs_mp`, `flash_tiled_vs_untiled` across
D ∈ {64,128} × S ∈ {512, 1024, 2048, 4096, 8192} × H ∈ {1, 12, 32}, **recording achieved warps/SM
next to every ratio**. One pass re-derives all four frozen constants (`FLASH_TILE_MIN`, the S≥512
tensor-core floor, the S≥4096 ws crossover, `wmma_flash_entry`'s per-D winner) and tests P2–P8
simultaneously. The existing harnesses are clock-cancelled median-of-9 same-process A/Bs, so they
port unchanged. Highest ratio of decisions-resolved to dollars-spent.

**#3 — The honest peer triangle at the one FILLED shape.**
`S=8192, D ∈ {64,128}, H=16, non-causal + causal`, fp16. Wukong (`ws` route) vs **FA3**, vs **cuDNN
frontend SDPA**, vs `torch.compile`, vs PyTorch SDPA with the backend *pinned* (both
`FLASH_ATTENTION` and `CUDNN_ATTENTION` — §1.5: the default is FA2-class, so publishing against the
default alone understates the peer). This is the only shape where occupancy is not an excuse
(finding F), so **this ratio *is* the wgmma go/no-go number** (plan §5 Phase 4.1(b)). Everything
else is context.

**#4 — `mp4`-as-warpgroup probe.**
`flash_d64_mp4` (4 warps = 128 threads = one Hopper warpgroup) at S ∈ {4096, 8192}, H ∈ {8, 12, 32},
with the grid re-derived for 132 SMs. Tests P2 directly. If mp4 flips to a clear win, Phase 4's
wgmma port has a validated CTA scaffold (same staging, same barriers, swap `mma.sync` → `wgmma`); if
it doesn't, `ws` is the scaffold instead. Cheap (the kernel is written and gated) and it decides
where the largest work item in the plan starts.

**#5 — Minimal dynamic-SMEM + tile-widening probe.**
Two sub-parts, deliberately ordered: **(a) static-only**, widen `flash_d128_mp_lm` to Bc=32 @ 3
stages (48 KB — finding C says this needs *no* new capability), and `flash_d64_mp` to Bc=64 @
3 stages (48 KB); **(b) dynamic**, convert one kernel's `.shared` to `.extern .shared`, add
`cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, …)` + non-zero
`shared_mem_bytes`, and push to Bc=128 @ 2–3 stages (128/192 KB). (a) tests finding C at zero
plumbing cost; (b) proves the plan's keystone plumbing end-to-end on the family where the tile is
most obviously too small, and produces the first evidence on whether >48 KB helps flash at all
(§2.5 predicts it does not, before wgmma).

*Ranked-out but noted:* a fused flash **backward** (today: materialized O(S²), `ptx_autodiff_bwd.rs:10`)
is the single largest correctness-preserving perf item, but it is device-free work — write it on the
4050, don't spend H100 hours discovering that.

---

## 5. GQA note

**The gap (FACT):** `crates/wukong_codegen_gpu/src/paged_kv.rs:64` — `KvConfig.heads` is documented
"KV heads (**== query heads here**; GQA/MQA would make this smaller)". `layer_plane_elems`,
`slab_elems` and `elem_offset` (`paged_kv.rs:78-101`) are all keyed on that one `heads`. The prefill
flash has the same conflation: `wmma_flash_plan(d, s, heads)` sets `grid_dim.1 = heads` and every
kernel folds one `ctaid.y` offset into **Q, K, V and O alike**.

**What Llama-class geometries need.** Llama-3-8B: 32 q-heads / **8 kv-heads** (g=4), D=128.
Llama-3-70B: 64 / 8 (g=8). Mistral-7B: 32 / 8. Concretely:

1. `KvConfig` splits into `q_heads` and `kv_heads` with `q_heads % kv_heads == 0`, and **every slab
   offset uses `kv_heads`** — the cache shrinks by exactly `g` (4× for Llama-8B, 8× for 70B). That
   shrink is the entire point of GQA and is what makes long-context serving fit; today Wukong
   allocates g× more KV than the model needs.
2. `paged_attention.rs`: the warp owns a `(slot, q_head)` pair; only the **K/V gather** changes to
   `kv_head = q_head / g`. Q, O, and the lane partition are untouched.
   **The layout-invariance proof survives**: `paged_attention.rs:32-44` states the bit-exactness
   property rests on (i) the lane partition being a function of the *logical position index alone*
   (`t % 32`) and (ii) a fixed butterfly merge order. GQA changes *which head's* K/V a lane reads,
   never *which lane* a position lands in nor the merge order — so
   `paged_attention_invariant_to_block_layout` remains valid by the same argument. Worth stating
   explicitly in the commit, because it is the kind of change that *looks* like it should break it.
3. Prefill flash: K/V base offset uses `ctaid.y / g`, Q/O keep `ctaid.y`. Two extra integer ops.
4. `kv_append` and the int8-KV quantizer take the same `kv_heads` geometry.

**How the field does it.**
- FACT — **FA2/FA3** take `nheads_k` alongside `nheads_q` and require `nheads_q % nheads_k == 0`;
  MQA/GQA are supported natively rather than by materializing repeated KV.
- FACT — **PyTorch SDPA** exposes `enable_gqa=True` (experimental; **flash and math backends on CUDA
  only**), same divisibility constraint plus `n_heads_k == n_heads_v`.
- FACT — **cuDNN frontend**: "All attention flavors MHA, MQA, GQA are supported", with independent
  head counts for Q/K/V.
- FACT — **vLLM / PagedAttention**: kernels look up the correct K/V vector for a query group through
  the block table without ever allocating repeated KV.
- The performance idiom (from the Triton-kernel literature): **set the query block so that all query
  heads mapping to one KV head land in a single CTA** — "By setting BLOCK_M = num_query_heads /
  num_kv_heads, each Q block covers all query heads that map to a single KV head, which only needs
  to be loaded once per Q block."

**The convergence worth acting on:** that idiom is *literally* `flash_d64_mp{4,8}` — W warps in one
CTA sharing one cooperatively-staged K/V block, with **W = g**. Llama-8B wants W=4 (`mp4`),
Llama-70B wants W=8 (`mp8`). So **GQA re-opens the shelved multi-warp kernels for a second,
completely independent reason** (the first being finding A, the 32-block warp cap). Two unrelated
lines of reasoning converging on the same shelved artifact is the strongest signal in this dossier:
whatever else Phase 3 does, `mp4`/`mp8` should be un-shelved and re-measured first.

---

## 6. Plan-assumption deltas (what this dossier changes)

1. **§2.4's `FLASH_WARPS=2` re-tune is worthless** — that constant governs `flash_d{D}` (untiled
   f32), which `FLASH_TILE_MIN = 0` makes undispatchable. The real constant is the implicit
   **1 warp/CTA in `wmma_flash_cfg`**, which caps the production kernel at 50% occupancy on H100 via
   the 32-block limit alone (finding A).
2. **§2.3's "Flash D-set" is not a SMEM gap** — D=256 at Bc=16/2-stage is 32 KB, under even the 48 KB
   *static* cap, and fit Ada too. It is a **128-register-accumulator** design problem, device-free,
   and `flash_d128_hs` is its (re-scoped) unlock (finding D).
3. **227 KB is not the flash lever on H100** — the register file did not grow (0%), so registers, not
   SMEM, bind; and a 192 KB CTA is 1 CTA/SM, which only pays with async instructions. Dynamic SMEM is
   the *GEMM* keystone; for flash, useful tile widening (Bc 16→64, 2→3 stages) fits in **48 KB static
   with no new capability at all** (finding C). Re-scope the flash half of §2.3.
4. **Risk 5 is not "may invert" — it will, by 6.6×, and here is the number**: at S=512/H=12/D=64 the
   grid supplies 19.2 warps/SM on the 4050 (oversubscribed, hence the "saturated" plateau) and
   **2.9 warps/SM on H100 (12% of capacity)** (finding E).
5. **§2.7 item 3's premise needs inverting**: the *occupancy* half of the long-S loss dissolves at
   S ≈ 8192, but the *ISA* half (mma.sync ≈ ⅔ of wgmma peak; FA2 = 35% of H100 peak) replaces it and
   is larger. Expect the published ratio to get **worse**, and say so before measuring.
6. **Two missing structural capabilities, both device-free**: no batch dimension in the flash grid
   (`grid.y = heads` only, `grid.z` unused) and no **split-KV / flash-decoding** seqlen-K parallelism
   — FA2's explicit fix for the small-batch/few-head case (§1.3 item 2). On 132 SMs these are the
   difference between 12% and ~100% utilization below S=2048 (finding H).
7. **Phase 3's peer must pin the SDPA backend.** PyTorch's H100 default is FA2-class; cuDNN is a
   separate, stronger backend behind `sdpa_kernel(SDPBackend.CUDNN_ATTENTION)`, and FA3 is a separate
   package again. Publishing against the bare default would understate the peer — the mirror image of
   the eager-torch problem §2.7 already flags.
