> **CORRECTIONS (2026-08-09, after Phase 2 and Waves 1–3 landed).** Several §1.7 / §5 / §6 defects
> this dossier names were **already fixed by Phase 2 before anyone read it**, and two §7 open
> questions are now answered. The original text stands with the correction beside it. Index:
>
> | Where | Original | Status at HEAD |
> |---|---|---|
> | §1.7, §5.1, §5.2, §6 row 8 | *"literal 16 MiB / 48 MiB"* f16 regime thresholds | **CLOSED for f16** — `f16_regime_thresholds(l2_bytes) = (l2*2/3, l2*2)` (`gpu.rs:1039`), machine-checked to reproduce the old literals at the probed 24 MiB by `f16_regime_thresholds_reproduce_the_4050_literals`. **Still literal in the bf16 twin** (`gpu.rs:2494`) — see §5.1's note. |
> | §1.7, §5.1 | *"`l2_cache_size()` … its only consumer is a bench print"* | **CLOSED** — L2 is probed once into `GpuTarget::l2_bytes` and feeds dispatch through `f16_regime_thresholds`; `l2_cache_size()` reads the descriptor and is cross-checked against it by a device gate (`gpu.rs:7570`). |
> | §1.7 | *"`sm_count()` falls back to `.unwrap_or(20)`"* | **CLOSED** — every `GpuTarget` field is a driver query with **no defaults**; an unqueryable SM count fails `Gpu::new` (`gpu.rs:26-28`, `:130-131`, `:563`). |
> | §1.7 | *"Static 48 KiB asserts"* at eight sites | **CLOSED** — replaced by an explicit `smem_budget` parameter and the named `gpu::STATIC_SMEM_CAP`; D6 §5 is landed. |
> | §6 row 7 | generalise the `swz` XOR phase off `bk == 32` | **STILL OPEN** — `ptx_wmma.rs:2131` / `:2599` still `assert_eq!(bk, 32)`. Re-verified 2026-08-09. |
> | §7 Q2 | *"Does the driver accept a >48 KiB `.extern .shared` allocation … after `cuFuncSetAttribute`?"* | **ANSWERED, at home, $0** — see the §7 note. |
> | §7 Q3 | ptxas register allocation for A3/A4 | **Downgraded from "rented GPU" to "CPU container"**, with independent corroboration from NVIDIA's own CUTLASS build — see §2.3 and the §7 note. |
>
> Re-verify before spending money: these were checked by reading the current files on 2026-08-09,
> not by a fresh device round.

> **COORDINATOR CORRECTION (2026-08-06, before commit):** this dossier derives the L2 regime
> thresholds as `1.33x/4x` of probed L2 — arithmetic that assumed the RTX 4050's L2 is 12 MiB
> (the spec-sheet figure). The B0 device probe reads `l2_bytes = 25165824` (24 MiB) on the
> metal, so the behavior-preserving multiples are **2/3x and 2x of the probed value** (16 MiB
> and 48 MiB exactly). The de-literaling brief specifies those multiples plus an exact-identity
> unit test. Everything else in the dossier is unaffected; on H100's 50 MiB the crossovers move
> accordingly (33 MiB / 100 MiB, not 66.5 / 200).

# D1 — H100 f16/bf16 GEMM derivation dossier

**Agent:** D1, Wave 0, GPU retarget campaign. **Date:** 2026-08-06. **Cost:** $0, no device touched.
**Repo state:** all `file:line` citations re-verified by reading the current files while HEAD was
`70f901c` (a parallel Wave-0 agent advanced HEAD past `0e1b2ea` and left `tools/cloud/modal_app.py`
dirty during this run — **not this agent's doing**). **D1 made zero writes to the repository**; its
only writes were this dossier and downloaded reference sources under the session scratchpad.

Every number below is tagged **FACT** (sourced, URL given), **DERIVED** (arithmetic from FACTs plus the
repo's own generator formulas, shown so it can be checked), or **PREDICTION** (falsifiable, to be
confirmed or killed on rented silicon).

---

## 0. Executive summary of the derivation

1. **H100's register file, warps/thread, and per-thread register ceiling are IDENTICAL to Ada's.**
   Only SMEM/SM (100 KB → 228 KB), warp slots/SM (48 → 64), L2 (12–24 MB → 50 MB), SMs (20 → 132) and
   HBM (192 GB/s → 3.35 TB/s) change. Consequence: **the shipped 128×128 w24 kernel gets the same
   2 CTAs/SM on H100 as on the 4050, and its *warp-slot* occupancy actually DROPS from 33% to 25%.**
   The 4050's occupancy verdicts are therefore not merely stale — several of them invert.
2. **Three ceilings bound Act 1 (mma.sync + cp.async, no wgmma/TMA), and they are computable now:**
   the mma.sync **issue** ceiling ≈ **642 TFLOPS** (measured ratio, §1.4); the **SMEM-read** ceiling
   ≈ **659 TFLOPS** for the shipped 4×4 warp tile; and the **L2-fill** ceiling ≈ **525 TFLOPS** for the
   shipped 128×128 CTA tile. cuBLAS on H100 does **716 TFLOPS at 4096³** (FACT). So Act 1's *arithmetic*
   ceiling at 4096³ is **90% of cuBLAS**, and with the shipped tile it is **73%**.
3. **The single highest-value Act-1 change is NOT the pipeline depth — it is the CTA tile width.**
   Going 128×128 → 128×256 raises the L2-fill ceiling from 525 → 700 TFLOPS and the SMEM-read ceiling
   from 659 → 989 TFLOPS, moving the binding constraint onto the mma.sync issue rate (642) where it
   belongs. That change needs **dynamic SMEM and nothing else** — no wgmma, no TMA, no new generator.
4. **At 1024³ a 128×128 tile leaves 52% of H100 idle** (64 tiles on 132 SMs). Wave quantization, not
   instruction choice, is the small-GEMM story on this part.
5. **Every literal L2 threshold in the dispatcher is wrong by ~2× in N.** The 16/48 MiB triggers
   (`gpu.rs:621,632`) must become multiples of the probed `l2_bytes`; on H100 the crossovers land near
   **4096³ and ~7200³** instead of 2048³ and ~3550³.

---

## 1. FACTS

### 1.1 H100 SXM5 device facts

| Fact | Value | Source |
|---|---|---|
| SMs (H100 SXM5) | **132** (PCIe: 114; full GH100 die: 144) | [Hopper whitepaper, Table 3, p.39](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf) |
| Shared memory capacity / SM | **228 KB**, "a 39% increase compared to A100's 164 KB" | [CUDA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) |
| Max shared memory / thread block (opt-in dynamic) | **227 KB** | [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) |
| CUTLASS's own SM90 SMEM budget constant | `sm90_smem_capacity_bytes = 232448` (= 227 KiB exactly) | [`sm90_common.inl:57`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/builders/sm90_common.inl) |
| Combined L1 + shared / SM | **256 KB** (A100: 192 KB) | [Hopper whitepaper p.22, p.27](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf) |
| Register file / SM | **64K 32-bit registers** (= 256 KB) — *unchanged from V100/A100/Ada* | [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html); whitepaper Table 4 p.41 |
| Max registers / thread | **255** | whitepaper Table 4 p.41 |
| Max registers / thread block | **65,536** | whitepaper Table 4 p.41 |
| Max warps / SM | **64** ("remains the same as Ampere") | Hopper Tuning Guide; whitepaper Table 4 |
| Max threads / SM | **2048** | whitepaper Table 4 p.41 |
| Max thread blocks / SM | **32** | Hopper Tuning Guide; whitepaper Table 4 |
| Max CTAs / thread-block cluster | **16** (portable: 8) | whitepaper Table 4; [CUTLASS `sm90_utils.py:312-315`](https://github.com/NVIDIA/cutlass/blob/main/python/cutlass_library/sm90_utils.py) |
| Tensor cores / SM | **4** | whitepaper Table 3 p.39 |
| FP32 cores / SM | **128** (A100: 64) | whitepaper Table 3 p.39 |
| L2 cache | **50 MB** (A100: 40 MB) | [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html); whitepaper p.40 |
| Memory | **80 GB HBM3, 3.35 TB/s**, 5120-bit | [NVIDIA H100 product page](https://www.nvidia.com/en-us/data-center/h100/) |
| TDP | up to **700 W** (configurable) | [NVIDIA H100 product page](https://www.nvidia.com/en-us/data-center/h100/) |
| FP16 / BF16 Tensor Core peak | **1,979 TFLOPS \*with sparsity\*** ⇒ **989.5 TFLOPS dense** | [NVIDIA H100 product page](https://www.nvidia.com/en-us/data-center/h100/) (asterisk = with sparsity) |
| FP8 Tensor Core peak | 3,958 TFLOPS w/ sparsity ⇒ 1,979 dense | same |
| Compute capability | **9.0** | whitepaper p.41 |
| Distributed shared memory | a CTA may read/write/atomic another CTA's SMEM within its cluster | Hopper Tuning Guide |

**Ada (`sm_89`, the 4050) for contrast — FACT** ([Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)):
shared memory/SM = **100 KB**; max SMEM/block (opt-in) = **99 KB**; max warps/SM = **48**;
max blocks/SM = **24**; register file = **64K 32-bit regs/SM**; combined L1 = 128 KB.

> **The load-bearing comparison: 64K registers/SM on BOTH parts.** Ada→Hopper adds SMEM, warp slots,
> L2, SMs and bandwidth. It adds **zero registers per SM**. Any Wukong kernel whose occupancy is
> register-limited on the 4050 has *exactly* the same CTA/SM count on H100.

### 1.2 Measured microarchitecture numbers (H800 PCIe = same GH100 SM, 114 SMs @1755 MHz)

Source: **Luo et al., "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture"**,
[arXiv:2402.13499](https://arxiv.org/abs/2402.13499) (Tables IV, V, VII, VIII, X). All FACT.

| Quantity | H800 | A100 | RTX4090 |
|---|---|---|---|
| Shared-memory throughput | **127.9 B/clk/SM** | 128.0 | 127.9 |
| L2 throughput (FP32.v4) | **3942 B/clk**; FP32 **4472 B/clk** | 2007.9 / 1853.7 | 1708.0 / 1622.2 |
| Global memory throughput | 1861.5 GB/s | 1407.2 | 929.8 |
| Global memory latency | 478.8 clk | 466.3 | 541.5 |
| Shared memory latency | 29.0 clk | 29.0 | 30.1 |
| L2 latency | 263.0 clk | 261.5 | 273.0 |

**Tensor-core instruction throughput on H800 (peak FP16 quoted by the paper: 756.5 TFLOPS):**

| Instruction | Throughput | % of that part's FP16 peak |
|---|---|---|
| `mma.sync.m16n8k16` f16→**f32** dense | **490.7 TFLOPS** | **64.9%** |
| `mma.sync.m16n8k16` f16→f16 dense | 494.4 TFLOPS | 65.4% |
| `wgmma.m64n256k16` f16→f32, SS, zeros | **728.5 TFLOPS** | 96.3% |
| `wgmma.m64n256k16` f16→f32, SS, **random data** | **665.4 TFLOPS** | 87.9% |
| `wgmma.m64n128k16` SS zeros / random | 728.5 / 659.8 | 96.3 / 87.2% |
| `wgmma.m64n64k16` SS zeros / random | 719.6 / 648.3 | 95.1 / 85.7% |
| `wgmma.m64n32k16` SS / RS zeros | 477.3 / 710.3 | 63.1 / 93.9% |
| `wgmma.m64n16k16` SS zeros | 287.0 | 37.9% |
| `wgmma.m64n8k16` SS zeros | 158.2 | 20.9% |

Also FACT from the same paper (Table VI): on Hopper, **`wmma` and `mma` both compile to
`HMMA.16816`**, while `wgmma` compiles to the distinct `HGMMA.64x256x16` SASS family; **INT4 `mma`
on Hopper degenerates to `IMAD` on the CUDA cores** (relevant to `ptx_int4.rs`, not to this dossier).

> **The Act-1 ratio: `mma.sync` reaches 0.649 of the tensor peak; `wgmma` reaches 0.963 (zeros) /
> 0.879 (random). `mma.sync / wgmma` = 0.674.** This is a same-device, same-clock measurement of an
> *instruction issue* rate with no memory traffic — an upper bound on any mainloop built from it.

### 1.3 cuBLAS on H100 — the peer bar

| Shape (square, bf16 in / f32 accum) | cuBLAS TFLOPS | % of 989.5 dense peak | Source |
|---|---|---|---|
| N = 4096 | **716** | 72.4% | [`pranjalssh/fast.cu` README](https://github.com/pranjalssh/fast.cu/blob/main/README.md) |
| N = 8192 | **795** | 80.3% | same |

The same worklog's hand-written wgmma+TMA kernel reaches **764 TFLOPS @4096³ (107% of cuBLAS)** and
**808 @8192³ (101.6%)**, and reports relative gains of +2% @512, +17% @1024, +7–8% @2048
([worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)).
NVIDIA's own cuBLAS 12.0 note claims a "three-fold speedup" for large compute-bound FP16 GEMMs on
H100 SXM vs A100 PCIe and warns H100 native kernels want **≥32 MiB of workspace**
([NVIDIA blog](https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/)).

For TF32 on **H100 PCIe**, Colfax measured cuBLAS 215.6 TFLOPS vs CUTLASS TMA+WGMMA+WS 249.5 (62%
efficiency) at 4096³ ([Colfax paper](https://research.colfax-intl.com/wp-content/uploads/2023/12/colfax-gemm-kernels-hopper.pdf)) — useful as a
reminder that **cuBLAS is beatable on Hopper, but only by the wgmma+TMA+WS path**.

### 1.4 `wgmma` — the PTX ISA facts

Source: [PTX ISA §9.7.16](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) (fetched
and text-extracted; quotes verbatim).

- **Warpgroup** = "a set of four contiguous warps such that the warp-rank of the first warp is a
  multiple of 4" (128 threads).
- **Shape menu, f16 and bf16 dense:** `.m64n{8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,
  144,152,160,168,176,184,192,200,208,216,224,232,240,248,256}k16` — **M is always 64, N is every
  multiple of 8 from 8 to 256, K is always 16.**
- tf32: same N menu, **K = 8**. fp8 (`.e4m3`/`.e5m2`) and int8: same N menu, **K = 32**.
- Two forms: `d, a-desc, b-desc, ...` (**A and B both from SMEM**, "SS") and `d, a, b-desc, ...`
  (**A from registers, B from SMEM**, "RS"). **B must always come from SMEM.** Accumulator D is
  always in registers.
- `.dtype` for f16 inputs = `{.f16, .f32}`; for bf16 = `{.f32}` only.
- Extra operands `scale-d` (predicate: accumulate or overwrite), `imm-scale-a`, `imm-scale-b`
  (±1), `imm-trans-a`, `imm-trans-b`.
- **SMEM matrix descriptor** = a 64-bit register: bits 13–0 start address, 29–16 leading-dim byte
  offset, 45–32 stride-dim byte offset, 51–49 base offset, 63–62 swizzle mode
  (`0` none, `1` 128-B, `2` 64-B, `3` 32-B), all encoded as `(x & 0x3FFFF) >> 4`.
- Async proxy ops: `wgmma.fence`, `wgmma.commit_group`, `wgmma.wait_group`.
- **PTX ISA Notes: "Introduced in PTX ISA version 8.0." Target ISA Notes: "Requires `sm_90a`."**

> **`sm_90a` is architecture-specific and NOT forward-compatible.** A `wgmma` module cannot be
> emitted under plain `.target sm_90`. This is a hard fact the plan should absorb: Act 2 means a
> *separate module family* with its own target string and its own capability gate, exactly as
> §5 Phase 4 assumes — but it also means the `sm_90a` modules will not JIT onto Blackwell later.

### 1.5 TMA — the PTX ISA facts

- `cp.async.bulk.tensor.{1d..5d}.dst.src{.load_mode}.completion_mechanism{...}
  [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo}{, ctaMask}{, cache_policy}`
- `.dst = {.shared::cta, .shared::cluster, .global}`, `.src = {.global, .shared::cta}`,
  `.completion_mechanism = .mbarrier::complete_tx::bytes` (g→s) or `.bulk_group` (s→g),
  `.load_mode = {.tile, .im2col, ...}`, `.multicast = .multicast::cluster`.
- **PTX ISA Notes: "Introduced in PTX ISA version 8.0." Target ISA Notes: "Requires `sm_90` or
  higher."** — TMA itself is available on plain `sm_90`; only `.multicast::cluster` is "advised to be
  used with `.target sm_90a`".
- The descriptor (`tensorMap`) is a **128-byte opaque host-built object** (`cuTensorMapEncodeTiled`),
  not a PTX construct — so a TMA path needs a host-side driver-API call the crate does not make today.

### 1.6 The static-SMEM ceiling is a PTX rule, not a device rule

Verbatim from PTX ISA §5.1.7 (Shared State Space):

> "Maximum capacity for statically allocated shared memory is **48 KB per CTA**. Architecture specific
> targets such as `sm_90a`, support extended shared memory capacity beyond 48 KB per CTA as described
> below:" — `sm_90a`: **228 KB**; `sm_100a`/`sm_103a`: 228 KB; `sm_110a`: 228 KB; **`sm_120a`/`sm_121a`: 100 KB**.

> **Plan-changing.** The repo's eight `smem <= 48*1024` asserts are not a 4050 artifact at all — they
> encode a **PTX-ISA-level rule that still applies verbatim on plain `.target sm_90`.** Retagging the
> header to `sm_90` does **not** unlock 227 KB. Only two routes do:
> (a) **dynamic SMEM** — `.extern .shared .align 16 .b8 smem[];` + `LaunchConfig.shared_mem_bytes` +
>     `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)` — works on plain `sm_90`
>     and on `sm_89` (99 KB) and `sm_80` (163 KB); this is Phase 2 step 4 and is correct as written; or
> (b) target `sm_90a`, which allows a **static** `.shared` array up to 228 KB but forfeits forward
>     compatibility.
> **Route (a) is the right one for Act 1**, and it is also the route the 4050 can validate at home
> (99 KB opt-in on `sm_89` vs the 48 KB static cap — a 2.06× budget increase, free, today).
>
> Note also for agent D4: **`sm_120a` caps at 100 KB, not 228 KB** — the consumer-Blackwell target has
> an Ada-class SMEM budget and must not inherit H100's tile lattice.

### 1.7 Wukong's own baseline (code facts, verified in-tree at `5686f37`)

| Fact | Location |
|---|---|
| SMEM formula: `stages * (bm + bn) * (bk + pad) * 2` | `crates/wukong_codegen_gpu/src/ptx_wmma.rs:216-218` (`PipeCfg::smem_bytes`), `:303-305` (`CliffCfg::smem_bytes`) |
| Shipped f16 workhorse: `mma_nt_f16_128_bk32_s2_r16`, bm=bn=128, bk=32, w2×4, s2, r16, **pad=8** → 40 KiB | `ptx_wmma.rs:246` |
| Dispatched large-GEMM kernel is actually the **no-pad swizzle** twin (`pad: 0`, 32 KiB) built at the call site | `gpu.rs:644` |
| Warp tile: `tm = bm/(16·wm)`, `tn = bn/(8·wn)`; accumulators = `tm·tn·4` f32/thread | `ptx_wmma.rs:1283-1284`, `:1364-1369` |
| `swz` requires **bk == 32 exactly** (XOR phase derived for nc = bk/8 = 4) and `wmr%8==0`, `wnc%8==0` | `ptx_wmma.rs:1303-1306` |
| `bk % 16 == 0` and `bk/8` a power of two; `bm%(16·wm)==0`; `bn%(8·wn)==0`; raster needs bm,bn powers of 2 | `ptx_wmma.rs:1274-1280` |
| `min_ctas > 0` ⇒ emits `.maxntid {threads},1,1` + `.minnctapersm {min_ctas}`; `0` ⇒ no directive | `ptx_wmma.rs:1336-1338` |
| Static 48 KiB asserts | `ptx_wmma.rs:942, 1312, 1662`; `ptx_fp8.rs:205, 453`; `ptx_int8.rs:548`; test at `gpu.rs:5587` |
| f16 regime dispatch literals **16 MiB / 48 MiB** on `ws_bytes = (m·k + n·k)·2` | `gpu.rs:620-632` |
| `l2_cache_size()` probe exists but feeds only a bench print | `gpu.rs:118`, consumer `gpu.rs:12228` |
| `sm_count()` falls back to `.unwrap_or(20)` | `gpu.rs:111` |

> **CORRECTION (2026-08-09) — the last four rows are the defects §5 and §6 build on, and Phase 2
> closed three and a half of them before this dossier was read. Do not re-fix them.**
>
> | Row | Status at HEAD, with the citation |
> |---|---|
> | Static 48 KiB asserts | **CLOSED.** Every generator that declares `.shared` now takes an explicit `smem_budget: usize` (e.g. `entry_mma_pipe` `ptx_wmma.rs:2085`, assert `:2148`; the whole `ptx_conv` `*_budget` family). The static boundary has a name, `gpu::STATIC_SMEM_CAP` (`gpu.rs:160`), and above it the generator emits the one module-scope `.extern` window (`gpu::DSMEM_DECL`, `:174`) and returns `SmemMode::Dynamic(n)`. D6 §5 is landed. |
> | 16 MiB / 48 MiB literals | **CLOSED for f16, OPEN for bf16.** `gemm_nt_f16` now derives its bands from the probed L2 via `f16_regime_thresholds` (`gpu.rs:1039`, call site `:1107`). **`gemm_nt_bf16` still branches on a literal `16 * 1024 * 1024` (`gpu.rs:2494`)** — the bf16 twin was not de-literaled, so on a 48 MiB-L2 part it takes the large-GEMM arm at a working set the f16 path calls L2-resident. Found by re-reading the tree on 2026-08-09; no code change was made here (this is a docs-only branch). |
> | `l2_cache_size()` feeds only a bench print | **CLOSED.** `GpuTarget::l2_bytes` is probed once at `Gpu` construction (`gpu.rs:46-47`) and is a dispatch input; `l2_cache_size()` now reads the descriptor (`gpu.rs:572`) and a device gate asserts the two agree (`gpu.rs:7570`, *"l2_cache_size() diverged from the target"*). |
> | `sm_count()` `.unwrap_or(20)` | **CLOSED.** `GpuTarget` has **no defaults** — *"a made-up device fact is worse than no device"* (`gpu.rs:26-28`); a non-positive `MULTIPROCESSOR_COUNT` fails `Gpu::new` (`:130-131`), and `sm_count()` reads the descriptor (`:563`). It now feeds real dispatch: `ptx_norm::norm_launch` (`gpu.rs:2959`) and `conv_splitk_factor` (`:3595`). |

---

## 2. DERIVED — the SMEM / stage / occupancy lattice

### 2.1 The formulas

Using the repo's own layout (two `.shared` arrays, A tile `bm × ldp`, B tile `bn × ldp`, f16):

```
ldp          = bk            (swz path, pad forced to 0)
             = bk + pad      (padded hand-placed path, pad ∈ {0, 8, 16, ...})
stage_bytes  = (bm + bn) · ldp · 2
smem_bytes   = stages · stage_bytes
```

Checks against the tree: 128×128 bk32 pad8 s2 = 2·256·40·2 = **40,960 B = 40 KiB** ✔ (`ptx_wmma.rs:246`
comment "40 KiB"); swz s2 = 2·256·32·2 = **32 KiB** ✔; swz s3 = **48 KiB exactly** ✔ (`:282`);
padded s3 = **60 KiB** ✔ (`:282`).

Two derived roofline quantities per CTA tile, both independent of the instruction path:

```
CTA arithmetic intensity      I_cta = (bm·bn) / (bm + bn)          FLOP per byte of SMEM fill
L2-fill ceiling               T_L2  = BW_L2 · I_cta
Warp-tile SMEM-read intensity I_wrp = (tm·tn·4096) / (tm·512 + tn·256)   FLOP per byte read from SMEM
SMEM-read ceiling             T_smem= BW_smem · I_wrp
```

(`tm·512` = tm A-fragments × 32 lanes × 4×b32; `tn·256` = tn B-fragments × 32 lanes × 2×b32;
`tm·tn` mma's × 2·16·8·16 FLOP each.)

**DERIVED bandwidth constants for H100 SXM5**, scaling the H800 measurements (§1.2) by clock and SM count:

- `BW_smem` = 128 B/clk/SM × 132 SM × 1.83 GHz = **30.9 TB/s** aggregate.
- `BW_L2`   = 4472 B/clk × 1.83 GHz = **8.2 TB/s** aggregate. *(±15%; the L2 slice count is tied to the
  5120-bit memory interface which is identical between H800 and H100, so this should transfer well.
  **Measure it with the twin control in the first H100 hour** — it is a load-bearing constant.)*
- Effective tensor clock: 989.5e12 / (132 SM × 4 TC × 1024 FLOP/clk) = **1.83 GHz** (DERIVED).

### 2.2 The full feasible (tile, stages) lattice under 227 KB

Occupancy budget used below: `CTAs/SM(smem) = floor(233472 / (smem + 1024))`, i.e. 228 KiB per SM with
a **1 KiB per-CTA driver reservation** (the 228-vs-227 KB gap). *Confirm against
`cuOccupancyMaxActiveBlocksPerMultiprocessor` on device — the crate already calls
`occupancy_max_active_blocks` (`gpu.rs:639` comment), so this is free to verify.*

| tile (bm×bn) | bk | pad | stage B | KiB/stage | max stages @1 CTA/SM | @2 CTA/SM | @3 CTA/SM | I_cta | **T_L2 (TFLOPS)** |
|---|---|---|---|---|---|---|---|---|---|
| 64×64   | 32 | 0 | 8 192  | 8.0  | 14* | 14 | 9 | 32.0 | 262 |
| 128×64  | 32 | 0 | 12 288 | 12.0 | 18 | 9 | 6 | 42.7 | 350 |
| 64×128  | 32 | 0 | 12 288 | 12.0 | 18 | 9 | 6 | 42.7 | 350 |
| **128×128** | **32** | **0** | **16 384** | **16.0** | **14** | **7** | **4** | **64.0** | **525** |
| 128×128 | 32 | 8 | 20 480 | 20.0 | 11 | 5 | 3 | 64.0 | 525 |
| 128×128 | 64 | 8 | 36 864 | 36.0 | 6  | 3 | 2 | 64.0 | 525 |
| **128×256** | **32** | **0** | **24 576** | **24.0** | **9** | **4** | **3** | **85.3** | **700** |
| 256×128 | 32 | 0 | 24 576 | 24.0 | 9 | 4 | 3 | 85.3 | 700 |
| 128×256 | 64 | 8 | 55 296 | 54.0 | 4 | 2 | 1 | 85.3 | 700 |
| 256×256 | 32 | 0 | 32 768 | 32.0 | 7 | 3 | 2 | 128.0 | 1050 |
| 256×256 | 64 | 8 | 73 728 | 72.0 | 3 | 1 | 1 | 128.0 | 1050 |

\* 64×64 at 1 CTA/SM is capped by the 227 KB per-block limit (28 stages of SMEM would fit the SM but
not one block); it is also pointless — nobody wants a 28-deep pipeline on a tiny tile.

**Read this table as: what the 4050 could express is the 48 KiB row of each tile.** For 128×128 swz
that was `s3` and nothing more. H100 permits `s14` at 1 CTA/SM and `s7` at 2 CTAs/SM — a **3.5×
deeper prefetch window at unchanged occupancy.**

### 2.3 Occupancy — the register wall is unchanged from Ada

Register estimate for `entry_mma_pipe` (DERIVED from the generator's declarations, `ptx_wmma.rs:1342-1370`):

```
regs/thread ≈ 4·tm·tn  (D accumulators, f32)
            + 4·tm     (A fragments, b32)
            + 2·tn     (B fragments, b32)
            + ~25–30   (pointers, indices, predicates, swizzle phases, epilogue scratch)
```

| tile | warp grid | threads | tm | tn | accum regs | est. regs/thread | regs/CTA | **CTAs/SM (reg)** | warps/SM | **I_wrp** | **T_smem** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **128×128 (shipped)** | 2×4 | 256 | 4 | 4 | 64 | ~125 | 32 000 | **2** | 16 (25%) | 21.3 | **659** |
| 128×128 | 2×2 (`w22`) | 128 | 4 | 8 | 128 | ~190 | 24 320 | 2 (SMEM/warp-bound) | 8 | 32.0 | 989 |
| **128×256** | 2×4 | 256 | 4 | 8 | 128 | ~185 | 47 360 | **1** | 8 (12.5%) | 32.0 | **989** |
| 256×128 | 2×4 | 256 | 8 | 4 | 128 | ~190 | 48 640 | **1** | 8 | 25.6 | 791 |
| 128×256 | 4×4 | 512 | 2 | 8 | 64 | ~125 | 64 000 | **1** | 16 | 26.9 | 832 |
| 64×64   | 2×2 | 128 | 2 | 4 | 32 | ~70 | 8 960 | 7 (→ 8 by regs, then SMEM/warp) | ≤28 | 18.3 | 566 |

Hard limits applied: 65 536 regs/SM, 65 536 regs/CTA, 255 regs/thread, 64 warps/SM, 32 CTAs/SM.

> **DERIVED, and this is the finding that most changes the plan's intuition:** the shipped 128×128 w24
> kernel is register-limited to **2 CTAs/SM on H100 — exactly as on the 4050** — and because H100 has
> 64 warp slots instead of Ada's 48, its warp occupancy *falls* from 33% to **25%**. Wukong will arrive
> on H100 running at a quarter of the machine's warp slots with a 2-deep prefetch window while cuBLAS
> runs a warp-specialised persistent kernel with a 4-deep TMA pipeline. **The 4050's occupancy A/B
> verdicts (`_mc3` lost, `w22` a tie, s3 "measured NEGATIVE") were all taken under a 48 KiB static cap
> that no longer binds; every one of them must be re-taken, and their sign may flip.**

**CORROBORATION (added 2026-08-09, and it is not ours) — the register wall this section names is
real at exactly the tile sizes A3/A4 target, and NVIDIA's own kernels hit it.** Building the CUTLASS
`sm90a` profiler (v4.6.1) on a CPU-only Modal container emitted **112** copies of

```
ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized
                due to insufficient register resources for the wgmma pipeline in the function
                '..cutlass3x_sm90_tensorop_gemm_f16_f16_f32_.._256x128x64_.._warpspecialized_pingpong..'
```

— all in the `f16_f16_f32` family, distributed **48 at `256x256x64`, 32 at `256x128x64`, 32 at
`128x256x64`**, and by scheduler **80 warp-specialised pingpong / 32 warp-specialised cooperative**.
Log: `bench/gpu/h100/2026-08-09-preflight-build-peers.log:1074-1185`.

This is independent evidence for §2.3's claim, from a different toolchain and a different code
generator: at 256-wide CTA tiles the Hopper register file is the binding resource even for a library
that was designed around it and is using warp specialisation to escape exactly this. Two
consequences. (1) A3/A4's stated kill condition — *"if A3 loses to the 128×128 base, the 1-CTA/SM
occupancy collapse beats the intensity gain"* — is not hypothetical. (2) **Act 2 does not make the
wall go away**: `wgmma` moves the accumulator problem, it does not delete it, so §4's "the wgmma
business case is a predicted 1.4–1.6×" must be read as **conditional on register pressure**, and the
first Act-2 measurement to take is `ptxas -v` regs/thread, not TFLOPS. (This is corroboration of a
mechanism, not a measurement of Wukong's kernels — §7 Q3 is still the experiment that settles ours.)

### 2.4 Wave quantization at 132 SMs

`tiles = ceil(M/BM)·ceil(N/BN)`; `waves = tiles/132`; `eff = tiles / (132·ceil(tiles/132))`
(fraction of SM-slots doing work in the persistent/1-CTA-per-SM accounting; with 2 CTAs/SM the
under-one-wave cases get proportionally worse, the over-one-wave cases are unchanged).

| shape | 128×128 | 128×256 | 256×128 | 64×128 | 64×64 |
|---|---|---|---|---|---|
| **1024³** | **64 / 0.48 / 48%** | 32 / 0.24 / **24%** | 32 / 0.24 / **24%** | 128 / 0.97 / 97% | 256 / 1.94 / 97% |
| **2048³** | 256 / 1.94 / 97% | 128 / 0.97 / 97% | 128 / 0.97 / 97% | 512 / 3.88 / 97% | 1024 / 7.76 / 97% |
| **4096³** | 1024 / 7.76 / 97% | 512 / 3.88 / 97% | 512 / 3.88 / 97% | 2048 / 15.5 / 97% | 4096 / 31.0 / 97% |
| **8192³** | 4096 / 31.0 / 97% | 2048 / 15.5 / 97% | 2048 / 15.5 / 97% | 8192 / 62.1 / 99% | 16384 / 124 / 99% |
| GPT d=768, M=2048 (N=3072,K=768) | 384 / 2.91 / 97% | 192 / 1.45 / **73%** | 192 / 1.45 / **73%** | 768 / 5.8 / 97% | 1536 / 11.6 / 97% |
| GPT d=768, M=4096 | 768 / 5.8 / 97% | 384 / 2.9 / 97% | 384 / 2.9 / 97% | 1536 / 11.6 / 97% | 3072 / 23.3 / 97% |
| GPT d=768, M=16384 | 3072 / 23.3 / 97% | 1536 / 11.6 / 97% | 1536 / 11.6 / 97% | 6144 / 46.6 / 99% | 12288 / 93 / 99% |
| GPT d=1024, M=2048 (N=4096,K=1024) | 512 / 3.9 / 97% | 256 / 1.94 / 97% | 256 / 1.94 / 97% | 1024 / 7.8 / 97% | 2048 / 15.5 / 97% |
| GPT d=1024, M=4096 | 1024 / 7.8 / 97% | 512 / 3.9 / 97% | 512 / 3.9 / 97% | 2048 / 15.5 / 97% | 4096 / 31 / 97% |
| GPT d=1024, M=16384 | 4096 / 31 / 97% | 2048 / 15.5 / 97% | 2048 / 15.5 / 97% | 8192 / 62 / 99% | 16384 / 124 / 99% |
| GPT d=4096, M=2048 (N=16384,K=4096) | 2048 / 15.5 / 97% | 1024 / 7.8 / 97% | 1024 / 7.8 / 97% | 4096 / 31 / 97% | 8192 / 62 / 99% |
| GPT d=4096, M=4096 | 4096 / 31 / 97% | 2048 / 15.5 / 97% | 2048 / 15.5 / 97% | 8192 / 62 / 99% | 16384 / 124 / 99% |
| GPT d=4096, M=16384 | 16384 / 124 / 99% | 8192 / 62 / 99% | 8192 / 62 / 99% | 32768 / 248 / 100% | 65536 / 496 / 100% |
| GPT d=8192, M=2048 (N=32768,K=8192) | 4096 / 31 / 97% | 2048 / 15.5 / 97% | 2048 / 15.5 / 97% | 8192 / 62 / 99% | 16384 / 124 / 99% |
| GPT d=8192, M=4096 | 8192 / 62 / 99% | 4096 / 31 / 97% | 4096 / 31 / 97% | 16384 / 124 / 99% | 32768 / 248 / 100% |
| GPT d=8192, M=16384 | 32768 / 248 / 100% | 16384 / 124 / 99% | 16384 / 124 / 99% | 65536 / 496 / 100% | 131072 / 993 / 100% |

**DERIVED closed form:** a `BM×BN` tile fills H100 once iff `M·N ≥ 132·BM·BN`.
For 128×128 that is **M·N ≥ 2.16 M elements**; for 128×256, **M·N ≥ 4.33 M**; for 64×64, **M·N ≥ 0.54 M**.
1024² = 1.05 M elements — **below the 128×128 threshold by 2.06×.**

Two conclusions:

1. **Only 1024³ (and any transformer GEMM with `M·N < 2.2e6`) has a wave-quantization problem on
   H100 with a 128-wide tile.** Every transformer FFN/attention-projection shape at d ≥ 768 and
   M ≥ 2048 is ≥ 2.9 waves and quantizes at ≥ 97%. **This kills a plausible worry cheaply**: the plan
   need not spend rented hours sweeping tile sizes for wave quantization at realistic model shapes;
   it needs exactly one small-GEMM arm.
2. The only shape class in the requested set that a **256-wide** tile hurts is `d=768, M=2048`
   (73% efficiency). If the dispatcher gains a 128×256 arm, it must be gated on `M·N ≥ 4.33e6`.

### 2.5 The three Act-1 ceilings, side by side

For each candidate, `T_act1 = min(T_issue, T_smem, T_L2)` where `T_issue = 0.649 × 989.5 = 642 TFLOPS`
(§1.2 measured mma.sync fraction of peak, DERIVED onto H100 SXM's peak):

| config | T_issue | T_smem | T_L2 | **binding ceiling** | vs cuBLAS@4096³ (716) |
|---|---|---|---|---|---|
| **128×128 w24 (shipped)** | 642 | 659 | **525** | **525 (L2 fill)** | **73%** |
| 128×128 w22 | 642 | 989 | **525** | 525 (L2 fill) | 73% |
| **128×256 w24** | **642** | 989 | 700 | **642 (mma issue)** | **90%** |
| 256×128 w24 | **642** | 791 | 700 | **642 (mma issue)** | 90% |
| 256×256 | **642** | ~1300 | 1050 | **642 (mma issue)** | 90% |
| 64×64 w22 | 642 | 566 | **262** | **262 (L2 fill)** | 37% |
| 64×128 | 642 | ~700 | **350** | 350 (L2 fill) | 49% |
| — *wgmma m64n256k16 (Act 2, reference)* | *879–963* | *~1580* | *(tile-dep.)* | *~879 (random data)* | *123%* |

> **These are ceilings, not predictions.** A real kernel realizes some fraction of the binding ceiling.
> But the table already answers the campaign's most expensive question for free: **Act 1's arithmetic
> ceiling at 4096³ is 90% of cuBLAS, and only if the CTA tile is widened past 128.** With the shipped
> tile the ceiling is 73%. Both are below parity, which means **§5 Phase 3 step 4's framing ("expect
> them to trail cuBLAS at large GEMM and say so plainly") is correct and now has a number.**

### 2.6 The `.cs` streaming-store arm needs re-deriving, not re-tuning

`cliff_swz_s2_v2cs` is dispatched at `ws ≥ 48 MiB` because on a 12–24 MB L2, C at 4096²·f32 (64 MiB)
was hopeless to keep resident and evicting it protected the raster band. On H100, C = 64 MiB against a
50 MiB L2 — **marginal, not hopeless**, and the 4050's own A/B showed `.cs` is a **0.75× LOSS** when C
is L2-scale (`gpu.rs:626-629`). PREDICTION in §4.3.

---

## 3. CUTLASS SM90 ground truth — what NVIDIA's own library considers optimal

Extracted from the CUTLASS sources (not from memory). Default generator behaviour is
`instantiation_level = 131` (`generator.py:5242`), which decomposes as
`wgmma_level = 131 % 10 = 1`, `mma_level = (131//10)%10 = 3`, `cluster_level = (131//100)%10 = 1`,
`pruning_level = 0` ([`sm90_utils.py:90-103`](https://github.com/NVIDIA/cutlass/blob/main/python/cutlass_library/sm90_utils.py)).

### 3.1 The emitted f16/bf16 configuration set

**WGMMA instruction shapes** ([`sm90_shapes.py`](https://github.com/NVIDIA/cutlass/blob/main/python/cutlass_library/sm90_shapes.py),
`SM90_WGMMA_SHAPES_FP16_BF16_DENSE`, filtered at level ≤ 1):

- **`(64, 128, 16)` — level 0, the *default* shape.**
- **`(64, 256, 16)` — level 1.**
- Everything else (N = 8…248 in steps of 8) is level 2 or 3: exhaustive/opt-in only.

**MMA multipliers** (`SM90_MMA_MULTIPLIERS`, level ≤ 3): `(2,1,4)`, `(1,1,4)`, `(4,1,4)`, `(2,2,4)`.
The K multiplier is **always 4** at the default level ⇒ **CTA K-tile = 4 × 16 = 64 for f16, always.**

**Cluster sizes** (`SM90_CLUSTER_SIZES`, level ≤ 1): **`(1,2,1)` and `(2,1,1)`** — i.e. **2-CTA clusters
only** at the default level. 1×1×1 is level 2; 4- and 8-CTA clusters are levels 3–4.

**Resulting default CTA tiles for f16/bf16** (DERIVED = wgmma shape × multiplier, filtered by
`is_tile_desc_valid` at `sm90_utils.py:281`):

| WGMMA shape | multiplier | **CTA tile (M×N×K)** | valid? |
|---|---|---|---|
| 64×128×16 | (1,1,4) | 64×128×64 | valid, but pruned at level 0 ("Don't stamp out FP16/BF16 kernels smaller than or equal to 64×128×64", `sm90_utils.py:509-511`) |
| 64×128×16 | (2,1,4) | **128×128×64** | ✔ |
| 64×128×16 | (4,1,4) | **256×128×64** | ✔ |
| 64×128×16 | (2,2,4) | **128×256×64** | ✔ |
| 64×256×16 | (1,1,4) | **64×256×64** | ✔ |
| 64×256×16 | (2,1,4) | **128×256×64** | ✔ (dup) |
| 64×256×16 | (4,1,4) | **256×256×64** | ✔ |
| 64×256×16 | (2,2,4) | 128×512×64 | ✗ (`cta_shape[1] > 256`) |

Validity rules worth copying verbatim (`sm90_utils.py:320-353`): CTA-M must be a multiple of 64 and
≥ 64; CTA-N ≥ 16 and a multiple of 8; CTA-K ≥ 16 and a multiple of 8 **and at least twice the WGMMA
K**; CTA shape upper bound is `<256, 256, 256>`; a cluster may not exceed 16 CTAs.

**Stages are `stages = 0` = auto** (`sm90_utils.py:391`): CUTLASS computes them at compile time from
the SMEM budget ([`sm90_gmma_builder.inl:69-85`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl)):

```
mainloop_pipeline_bytes = sizeof(PipelineTmaAsync<1>::SharedStorage)   // 2 mbarriers = 16 B
stage_bytes = round_up(A_bytes + B_bytes, 128) + mainloop_pipeline_bytes
stages      = (232448 - round_up(epilogue_carveout,128)) / stage_bytes
```

(`PipelineTmaAsync<Stages>::SharedStorage` = `FullBarrier[Stages]` + `EmptyBarrier[Stages]`, each an
8-byte `ClusterTransactionBarrier`/`ClusterBarrier` — [`sm90_pipeline.hpp:273-283`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/pipeline/sm90_pipeline.hpp) — hence 16 B/stage.)

**DERIVED stage counts for the default f16 tiles (zero carveout):**

| CTA tile | A B / stage | stage_bytes | **stages @227 KiB** | mainloop SMEM |
|---|---|---|---|---|
| 64×128×64 | 8 192 + 16 384 | 24 592 | **9** | 221 KiB |
| **128×128×64** | 16 384 + 16 384 | 32 784 | **7** | 224 KiB |
| **128×256×64** | 16 384 + 32 768 | 49 168 | **4** | 192 KiB |
| **256×128×64** | 32 768 + 16 384 | 49 168 | **4** | 192 KiB |
| 64×256×64 | 8 192 + 32 768 | 40 976 | **5** | 200 KiB |
| 256×256×64 | 32 768 + 32 768 | 65 552 | **3** | 192 KiB |

(With a ~32 KB TMA-epilogue carveout, 128×256×64 stays at 4 and 128×128×64 drops to 6.)

### 3.2 Schedulers CUTLASS emits, and when

From `get_valid_schedules` (`sm90_utils.py:443-713`):

- `KernelScheduleType.ScheduleAuto`, `TmaWarpSpecialized` — always (non-fp8).
- **`TmaWarpSpecializedPingpong`** + `EpilogueScheduleType.TmaWarpSpecialized` — whenever the epilogue
  can use SMEM (`can_tile_desc_use_shmem_in_epilogue`: `bits(C)·M·N + bits(D)·M·N ≤ 2^20`).
- **`TmaWarpSpecializedCooperative`** — only when **CTA-M % 128 == 0** (`is_tile_desc_compatible_with_cooperative`,
  `:408-410`). So 64-row tiles get pingpong/non-persistent, 128- and 256-row tiles get cooperative.
- **Stream-K** tile scheduler is emitted alongside every cooperative schedule (`:665-668`) — CUTLASS's
  answer to wave quantization is **stream-K, not a smaller tile**.
- The epilogue SMEM heuristic explicitly reasons about the 228 KB budget: *"Hopper's max shmem size is
  228 KB, and 2^20 ~= 131 KB. Since epilogue can't possibly use ALL of the shmem available we can just
  settle on 2^20 bits (~131 KB) being the upper bound we would allow for epilogue."* (`:428-435`).

### 3.3 What NVIDIA's library therefore "considers optimal", by shape class

| shape class | CUTLASS's answer |
|---|---|
| large square / compute-bound (≥4096³) | **128×256×64 or 256×128×64, cluster 2×1×1 or 1×2×1, 4 stages, warp-specialised *cooperative*, TMA epilogue** |
| moderate (≈2048³) / more tiles wanted | **128×128×64, 6–7 stages, cooperative or pingpong** |
| skinny-M (decode, small batch) | **64×128×64 / 64×256×64, 5–9 stages, pingpong or plain `TmaWarpSpecialized`** (cooperative is illegal below CTA-M 128) |
| awkward tile counts | **the same tile + the Stream-K scheduler**, not a smaller tile |
| the K dimension | **always 64 for 16-bit** — CUTLASS never ships a BK=32 f16 SM90 mainloop |

Independent confirmation from a reproduced 107%-of-cuBLAS kernel
([worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)): **BM=128, BN=256,
BK=64; `wgmma.m64n256k16`; 3 warpgroups (1 producer + 2 consumers, 384 threads); cluster 2×1×1 with TMA
multicast; QSIZE 3–5; 168 regs/thread → 64 512 of the 65 536 per-CTA registers; Hilbert-curve tile
ordering for 83% L2 hit rate vs cuBLAS's 70%.** That is the CUTLASS default configuration, found
independently.

> **Two things Wukong's table disagrees with NVIDIA about, and both are 4050 artifacts:** BK
> (Wukong 32, CUTLASS 64 — Wukong's BK=32 was forced because BK=64 at s2 padded is 36 KiB/stage and
> two stages barely fit 48 KiB) and CTA-N (Wukong 128, CUTLASS 256 — forced by the same cap).

---

## 4. PREDICTIONS (falsifiable, numeric)

All predictions are for **H100 SXM5, clock-locked root VM, f16 or bf16 in / f32 accumulate,
`C = A·Bᵀ` (NT), same-run adjacent A/B against cuBLAS, twin control passing.**
Reference bar: cuBLAS **716 TFLOPS @4096³** and **795 @8192³** (FACT §1.3); **560 ± 80 @2048³** and
**330 ± 80 @1024³** (PREDICTION — interpolated from the worklog's relative gains and the wave-count
argument; **record cuBLAS's own absolute number in the first round, it is cheap and it anchors everything**).

### 4.1 Act-1 headline predictions (mma.sync + cp.async, forward-JIT, no wgmma/TMA)

| size | what the shipped dispatcher picks | binding ceiling | **PREDICTED % of cuBLAS** | mechanism |
|---|---|---|---|---|
| **1024³** | `wmma_nt_f16_pipe_64_s6` (64×64, WMMA, BK=16, s6, 24 KiB) — `gpu.rs:647` | **T_L2 = 262 TFLOPS** | **45%** (range 32–58%) | 64×64 fills the SMs (256 tiles) but its intensity is only 32 FLOP/byte, so the L2 fill path caps it at 262 TFLOPS ≈ 26% of peak. Instruction path is irrelevant here. |
| **2048³** | `mma_nt_f16_128_bk32_s2_r16_swz` (ws = 16.0 MiB ≥ 16 MiB) — `gpu.rs:632-645` | **T_L2 = 525** | **52%** (42–65%) | 256 tiles ≈ 1 wave at 2 CTAs/SM; whole 16 MiB working set is L2-resident on 50 MiB; 2-stage pipeline against a 479-clock global latency is the shortfall. |
| **4096³** | `cliff_swz_s2_v2cs` (ws = 64 MiB ≥ 48 MiB) — `gpu.rs:621-630` | **T_L2 = 525** | **48%** (38–60%) | Same tile; plus the `.cs` streaming-store hint is **mis-dispatched** here (§4.3), costing an estimated 5–15%. |
| **8192³** | `cliff_swz_s2_v2cs` | **T_L2 = 525** | **45%** (35–57%) | `.cs` is now *correctly* dispatched (ws = 256 MiB ≥ 4×L2); HBM at 3.35 TB/s is comfortable (§2.1); the shortfall is entirely the 128-wide tile + 2-stage pipe. |

**Mechanism statement, once, for all four:** on the 4050 the large-GEMM kernel was **HBM-bound** and the
raster fixed it. On H100 the machine balance is 989.5 TFLOPS / 3.35 TB/s = **295 FLOP/byte** vs the
4050's 121/0.192 = **630 FLOP/byte** — H100 is *relatively* **2.1× less bandwidth-starved**. With
rasterization keeping the L2 hit rate high, a wave of 264 CTAs at r16 has an effective DRAM intensity
of ~1040 FLOP/byte (DERIVED, §2.1 model), far above 295. **So the 4050's HBM cliff dissolves and the
binding constraint moves to L2→SMEM fill bandwidth and mma.sync issue rate.** Every Act-1 lever must
therefore target *intensity per SMEM byte*, not DRAM traffic.

### 4.2 Act-1 with the Phase-2 dynamic-SMEM capability landed

| size | config | **PREDICTED % of cuBLAS** | Δ vs shipped |
|---|---|---|---|
| 1024³ | 128×128 swz s6 + **split-K 4** (or 64×128 s8) | **60%** (48–72%) | +15 pts |
| 2048³ | **128×256 swz s6 r16**, `min_ctas=1` | **62%** (50–74%) | +10 pts |
| 4096³ | **128×256 swz s6 r16**, `min_ctas=1` | **66%** (55–78%) | +18 pts |
| 8192³ | **128×256 swz s6 r32**, `min_ctas=1`, `.cs` epilogue | **68%** (57–80%) | +23 pts |

**Absolute Act-1 ceiling, for the Phase-4 memo: ≤ 90% of cuBLAS at 4096³ and ≤ 81% at 8192³** (the
mma.sync issue ceiling 642 TFLOPS divided by cuBLAS's 716 / 795). **Predicted realistic best Act-1:
66–72% at 4096³.** ⇒ **the wgmma business case is a predicted 1.4–1.6× on large GEMM**, computable
today and confirmable with two rented hours.

### 4.3 Dispatch-bug predictions (all confirmable in one Modal invocation)

1. **`gpu.rs:621`'s 48 MiB literal mis-fires at 4096³ on H100.** Scaled to L2, the arm means
   "ws ≥ 2–4× L2"; on 50 MiB that is 100–200 MiB, i.e. ≥ ~6000³. At 4096³ (ws = 64 MiB = 1.28× L2)
   the `.cs` evict-first hint will forfeit reuse of a C that *nearly* fits L2 — the same failure the
   4050 measured at 2048³ (0.75×, `gpu.rs:626-629`). **PREDICTION: `cliff_swz_s2` beats
   `cliff_swz_s2_v2cs` at 4096³ on H100 by 3–15%.**
2. **`gpu.rs:632`'s 16 MiB literal mis-classifies GPT d=1024 FFN shapes.** `M=4096, N=4096, K=1024`
   has ws = exactly 16.0 MiB ⇒ trips the "large/HBM-bound" arm, but on a 50 MiB L2 it is 0.32× L2 —
   deeply L2-resident. **PREDICTION: a deep-pipeline L2-resident variant beats the raster workhorse
   on that shape by 5–20%.**
3. **`gpu.rs:647`'s `m <= 1024 && n <= 1024` literal is an SM-count proxy.** It should be
   `M·N < c · sm_count · BM · BN`. **PREDICTION: the 64×64 arm is the right choice at 1024³ on
   H100 *for SM fill* but the wrong choice *for intensity*; a 128×128 + split-K arm beats it by 20–40%.**
4. **The skinny-M gap:** decode shapes (M = batch, e.g. 8–64) fail every `m % 128 == 0` /
   `m % SM_BM == 0` guard and fall through to the un-staged `wmma_pick` single-tile path
   (`gpu.rs:656-658`). On a 132-SM part that path is ~0 flops. **PREDICTION: `--backend=gpu`
   token-decode latency on H100 is dominated by this fall-through, not by attention.** (Out of D1's
   GEMM scope, but it is the most valuable thing in this file for the serving story.)

### 4.4 THE TOP-6 ACT-1 CONFIRMATION POINTS

The plan's derive-before-you-rent rule asks for six points instead of a 40-point sweep. Here they are,
**ranked, with the hypothesis each one tests and the observation that kills it.** All six are
`entry_mma_pipe` rows — **no new generator, no new PTX family**; #2–#6 need only the Phase-2 dynamic
SMEM launch path. All are generator-legal today (asserts at `ptx_wmma.rs:1273-1315` checked by hand
below). Run them at **1024³ / 2048³ / 4096³ / 8192³ + GPT (M=4096, N=4d, K=d) at d ∈ {1024, 4096}**,
round-robin best-of-N inside ONE invocation, with the byte-identical twin control column.

---

**A1 — CONTROL: `mma_nt_f16_128_bk32_s2_r16_swz` unchanged.**
`bm128 bn128 bk32 wm2 wn4 s2 r16 swz pad0 min_ctas0` → SMEM 32 KiB, 256 threads, 2 CTAs/SM.
*Purpose:* the honest "forward-JIT an Ampere-class kernel onto Hopper, change nothing" number that
Phase 3 step 4 must publish, and the A/B base for #2–#6. *Not a candidate — a measurement.*
*Kill:* nothing kills it; if it fails to JIT at all, that is Risk #2 (the legacy 8×`.b32` WMMA
fragment spelling) firing and everything else waits.

**A2 — `..._128_bk32_s4_r16_swz` — depth at unchanged tile and occupancy.** *(rank 3)*
`s4` → SMEM 64 KiB; 2 CTAs/SM needs 128 KiB ≤ 227 ✔ (lattice §2.2 allows s7).
*Hypothesis:* the 2-stage window (tuned under the 48 KiB static cap) cannot cover H100's 479-clock
global latency at 25% warp occupancy; doubling it recovers 10–25% at 4096³/8192³. **The cheapest
possible test of the whole dynamic-SMEM capability** — one table row.
*Kill:* if `s4` ≤ `s2` + noise, the mainloop is not latency-bound and every deeper-pipeline candidate
(A5) is dead too; go straight to the tile-width levers.

**A3 — `mma_nt_f16_128x256_bk32_s6_r16_swz`, `min_ctas = 1` — THE TOP-RANKED LEVER.** *(rank 1)*
`bm128 bn256 bk32 wm2 wn4 s6 r16 swz pad0 min_ctas1` → 256 threads, tm=4, tn=8, 128 accumulator
regs/thread (~185 total), **1 CTA/SM**, SMEM 6 × 24 576 = **144 KiB**.
Legality: `bk==32` ✔; `wmr = 128/2 = 64` ✔ %8; `wnc = 256/4 = 64` ✔ %8; `bm%(16·2)=0` ✔;
`bn%(8·4)=0` ✔; `a_chunks = 128·32/(256·8) = 2 ≥ 1` ✔; `b_chunks = 4` ✔; bm,bn powers of two ✔.
*Hypothesis:* the shipped kernel is **L2-fill-bandwidth-bound at 525 TFLOPS**; widening N doubles
the warp-tile SMEM intensity (21.3 → 32 FLOP/byte) and raises the CTA intensity 64 → 85.3 FLOP/byte,
moving the binding ceiling to the mma.sync issue rate at 642. Predicted **+18 to +25 points of
cuBLAS at 4096³**. This is also exactly the tile CUTLASS defaults to (§3.3).
*Kill:* if A3 loses to A1, the 1-CTA/SM occupancy collapse (8 of 64 warp slots) dominates the
intensity gain — in which case the answer is A4's 512-thread arrangement or warp specialisation, i.e.
Act 2. **Either result is worth the rental.**

**A4 — `mma_nt_f16_256x128_bk32_s6_r16_swz`, `min_ctas = 1` — the M-major twin.** *(rank 2)*
`bm256 bn128 bk32 wm2 wn4 s6 r16 swz pad0 min_ctas1` → 256 threads, **tm = 256/32 = 8, tn = 128/32 = 4**,
128 accumulator regs/thread (~190 total), 1 CTA/SM, SMEM identical to A3 (6 × 24 576 = **144 KiB**).
Legality: `wmr = 256/2 = 128` ✔ %8; `wnc = 128/4 = 32` ✔ %8; `bm%(16·2)=0` ✔; `bn%(8·4)=0` ✔;
`a_chunks = 256·32/(256·8) = 4` ✔; `b_chunks = 2` ✔; `bk == 32` ✔.
*(A `wm4 wn2` arrangement — tm=4, tn=8 — is equally legal and gives A3's warp-tile ratio; use `wm2 wn4`
so the experiment actually varies the thing being tested. `wm2 wn2` would give tm=8, tn=8 = **256
accumulator registers — illegal, exceeds the 255/thread limit**; do not emit it.)*
*Hypothesis:* **identical CTA intensity to A3 (85.3 FLOP/byte) but a deliberately different warp-tile
SMEM intensity (25.6 vs A3's 32)**, so A4 isolates the mechanism: if A4 ≈ A3, the win is CTA *area*
(L2-fill bound); if A3 ≫ A4, the win is the **B-fragment SMEM read path** (B fragments are 2×b32/lane
vs A's 4×b32, so widening N buys reuse more cheaply than widening M) — which tells Act 2 to
prioritise `wgmma` **N**-width over M-width, and tells the dispatcher which twin to pick for `M ≫ N`
(attention output-projection) vs `N ≫ M` (FFN up-projection).
*Kill:* if both A3 and A4 lose to A1, the 1-CTA/SM register-pressure story dominates the intensity
gain, Act 1 is capped at A2's number, and warp specialisation (Act 2) is the only route past it.

**A5 — A3 with `raster = 32`.** *(rank 5)*
*Hypothesis:* r16 was swept when 40 CTAs were co-scheduled (20 SMs × 2). H100 co-schedules 132–264.
The perimeter-optimal band width is `sqrt(W)` = 16.2 for W=264 (so r16 may already be right), but the
4050's measured optimum was ~2.5× its own perimeter-optimal, which extrapolates to r≈32–40. The
capacity bound `(W/G + G)·BM·K·2 ≤ L2` admits G ∈ [6.6, 41] at 4096³.
Predicted: **r32 wins at 8192³ by 2–8% and ties or loses at ≤4096³.**
*Kill:* if r32 ≤ r16 everywhere, freeze r16 as target-independent and stop sweeping raster forever.

**A6 — the small-GEMM arm: `mma_nt_f16_128_bk32_s6_r8_swz` + split-K = 4, at 1024³.** *(rank 4)*
`bm128 bn128 bk32 wm2 wn4 s6 r8 swz` → SMEM 96 KiB, 2 CTAs/SM (192 KiB ≤ 227 ✔), split-K 4 ⇒
64 tiles × 4 = **256 CTAs** on 132 SMs. Compare against **A1's dispatched `pipe_64_s6`**.
*Hypothesis:* at 1024³ the machine is not instruction-bound or bandwidth-bound — it is **empty**
(64 tiles / 132 SMs = 48%). Split-K restores occupancy *without* forfeiting the 128×128 tile's 64
FLOP/byte intensity, so it should beat the 64×64 tile's 262-TFLOPS L2 ceiling by 20–40%.
*Kill:* if the split-K reduction epilogue eats the gain, the answer for small GEMM on H100 is a
**Stream-K scheduler** (what CUTLASS does, §3.2) and that is a scheduler change, not a tile change —
note it and move on.

**Non-candidates, and why (so nobody spends an hour on them):**
`bk = 64` — the `swz` XOR phase is asserted to `bk == 32` (`ptx_wmma.rs:1304`), so BK=64 forces the
padded hand-placed path, which the 4050 measured **1.23× slower** than swz (`gpu.rs:634-643`); the
right move is to *generalise the swizzle phase to nc = 8* (free, at home, PTX-text-gateable) and only
then test BK=64. `256×256` — 128 accumulator regs at 512 threads = 64 000 regs/CTA, one CTA/SM with
4 warp slots' worth of latency hiding and no warp specialisation to fix it; it is an Act-2 shape.
`min_ctas = 3` on 128×128 — the 4050 measured `_mc3` a loss, and on H100 the register wall is
*identical*, so the verdict transfers; retest only if A2 shows the mainloop is latency-bound.

### 4.5 THE TOP-4 ACT-2 (`wgmma` + TMA) CONFIGS

Ranked by prior, all requiring `.target sm_90a` + `.version 8.0`+, a host `cuTensorMapEncodeTiled`
call, `mbarrier` pipelines, and `setmaxnreg`.

| # | config | why |
|---|---|---|
| **W1** | **CTA 128×256×64; `wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16` (SS); cluster 2×1×1 with `.multicast::cluster` on A; 4 stages (49 168 B/stage → 192 KiB mainloop + TMA epilogue); warp-specialised **cooperative**: 1 producer warpgroup (`setmaxnreg.dec` → ~32 regs) + 2 consumer warpgroups (`setmaxnreg.inc` → ~232 regs), 384 threads, 1 CTA/SM** | **This is simultaneously CUTLASS's default emitted kernel (§3.1) and the independently reproduced 107%-of-cuBLAS point (§3.3).** Two independent sources landing on the same configuration is the strongest prior available. Predicted **95–108% of cuBLAS at 4096³–8192³.** |
| **W2** | **CTA 256×128×64; `m64n128k16`; cluster 1×2×1 (multicast on B); 4 stages; cooperative** | The M-major twin CUTLASS also emits. Wins when `M ≫ N` (GPT prefill attention-output projection, `M = B·S`, `N = d`). Predicted 92–103%. |
| **W3** | **CTA 128×128×64; `m64n128k16`; cluster 2×1×1; 6–7 stages; `TmaWarpSpecializedPingpong` (2 consumer warpgroups alternating mainloop/epilogue)** | The moderate-size arm (≈2048³, and any shape with `M·N < 4.3e6` where a 256-wide tile quantizes). 7 stages is free at 227 KiB. Predicted 90–100%. |
| **W4** | **CTA 64×256×64 (or 64×128×64); cluster 1×1×1 or 1×2×1; 5–9 stages; plain `TmaWarpSpecialized`, non-persistent** | The skinny-M / decode arm. **Cooperative is *illegal* below CTA-M 128** (`sm90_utils.py:408-410`), so this class needs a different schedule, which is a real code-shape fact for the Act-2 design. Predicted: the largest *relative* win over Act 1, because Act 1 has no working skinny-M path at all (§4.3.4). |

**Act-2 design facts that constrain the generator** (all §1.4/§1.5 FACTs, restated as requirements):
`wgmma` needs `sm_90a`; **B must live in SMEM** (no RS form for B); the SMEM descriptor is a 64-bit
register with a 3-bit base offset and a 2-bit swizzle-mode field whose 128-B mode requires the
repeating pattern to start on a **1024-byte boundary**; the accumulator lives in registers and
`scale-d` selects accumulate-vs-overwrite (which is exactly the "skip zero-init" trick the worklog
credits for a measurable gain); `wgmma.fence` / `commit_group` / `wait_group` bracket the async proxy.
TMA needs a **host-built 128-byte `tensorMap`**, which `wukong_codegen_gpu` has no equivalent of today.

---

## 5. L2-resident dispatch thresholds on a 50 MB L2

### 5.1 The defect

`gpu.rs:620-632` computes `ws_bytes = (m·k + n·k)·2` and compares against **literal 16 MiB and
48 MiB**. Those literals were swept on the 4050, whose L2 the tree itself cannot agree on: the
comments say 24 MB (`gpu.rs:608`, `ptx_wmma.rs:224`) and the census says 12 MB. **`l2_cache_size()`
exists at `gpu.rs:118` and its only consumer is a bench print** (`gpu.rs:12228`).

Both readings give the same *shape* of answer, which is why the conclusion is robust:

| interpretation | 16 MiB arm | 48 MiB arm |
|---|---|---|
| if 4050 L2 = 12 MiB | **1.33 × L2** | **4.00 × L2** |
| if 4050 L2 = 24 MiB | 0.67 × L2 | 2.00 × L2 |

> **STATUS 2026-08-09 — the f16 half of this defect is FIXED, the 24-vs-12 question is settled at
> 24 MiB by the probe, and the tree therefore took the SECOND row of that table.** `gemm_nt_f16`
> reads `f16_regime_thresholds(g.target().l2_bytes)` (`gpu.rs:1107`) and the function is
> `(l2*2/3, l2*2)` (`gpu.rs:1039`) — **not** the 1.33×/4× reading §5.2 recommends below, because that
> reading assumes a 12 MiB L2 the probe refutes. At the probed 25 165 824 B the pair is exactly
> `(16 777 216, 50 331 648)` = the old 16/48 MiB literals, so the 4050 dispatch census did not move:
> `f16_regime_thresholds_reproduce_the_4050_literals` machine-checks that identity, pins the `·2/3`
> evaluation order (`/3·2` truncates first and gives a different band edge whenever `l2 % 3 == 2`),
> and its device arm re-confirms the probe on the metal.
>
> **The bf16 twin is still literal.** `gemm_nt_bf16` (`gpu.rs:2494`) computes the same `ws_bytes` and
> compares it against a hardcoded `16 * 1024 * 1024`. On a 48 MiB-L2 part (L4 / L40S) or a 50 MiB one
> (H100) that arm fires at a working set the f16 path now correctly calls L2-resident, so **the
> training-precision GEMM would carry the 4050's tuning verdict onto every datacenter card**.
> Recorded here, unfixed: this is a docs-only branch and `gpu.rs` has one owner.
>
> **`l2_cache_size()` is no longer dead:** L2 is a probed `GpuTarget` field (`gpu.rs:46-47`) that
> feeds dispatch, and `l2_cache_size()` (`gpu.rs:572`) reads that descriptor, gated against it at
> `gpu.rs:7570`.

### 5.2 The de-literaled rule and its H100 values

Replace the literals with multiples of the probed `l2_bytes`. Recommended (taking the 12 MiB reading,
which makes 16 MiB ≈ 1.33× — the reading under which the existing 4050 thresholds are self-consistent
with "the working set stops fitting L2"):

```
raster/large arm      : ws_bytes >= (4 * l2_bytes) / 3        // 1.33 x L2
streaming-C (.cs) arm : ws_bytes >= 4 * l2_bytes              // 4.00 x L2
```

| device | L2 | 1.33× L2 | ⇒ square N | 4× L2 | ⇒ square N |
|---|---|---|---|---|---|
| RTX 4050 (as shipped) | 12 MiB | 16 MiB | **2048** | 48 MiB | **3547** |
| **H100** | **50 MiB** | **66.7 MiB** | **≈ 4180** | **200 MiB** | **≈ 7240** |
| A100 (for Phase 3b) | 40 MiB | 53.3 MiB | ≈ 3740 | 160 MiB | ≈ 6475 |
| L40S (Phase 1) | 48 MiB | 64 MiB | ≈ 4096 | 192 MiB | ≈ 7094 |

(`ws = 4N²` bytes for a square f16 GEMM, so every threshold moves as `√(L2_new/L2_old)` in N:
**H100 shifts every crossover up by √(50/12) = 2.04× in N.**)

> **CORRECTION (2026-08-09) — the shipped rule is `2/3×` and `2×`, not `1.33×` and `4×`, and the
> table above is therefore wrong in its numbers while right in its shape.** The 1.33/4 reading is
> the one that assumes a 12 MiB 4050 L2; the device probe reads **24 MiB**, and the top-of-file
> coordinator correction already said so. `f16_regime_thresholds` (`gpu.rs:1039`) implements
> `(l2·2/3, l2·2)`. Recomputed against the same `ws = 4N²`:
>
> | device | probed/spec L2 | `2/3 × L2` | ⇒ square N | `2 × L2` | ⇒ square N |
> |---|---|---|---|---|---|
> | RTX 4050 (probed) | 24 MiB | 16 MiB | **2048** | 48 MiB | **3547** |
> | **L4 (probed on the metal)** | **48 MiB** | **32 MiB** | **≈ 2896** | **96 MiB** | **≈ 5017** |
> | **H100** | **50 MiB** | **33.3 MiB** | **≈ 2956** | **100 MiB** | **5120** |
> | A100 | 40 MiB | 26.7 MiB | ≈ 2644 | 80 MiB | ≈ 4580 |
>
> The 4050 row is exactly the old literals, which is the point (§5.1's status note). **The L4 row is
> measured, not derived**: the Phase-1 L4 round probed L2 = 48 MiB and the harness printed the bands
> as `[32, 96) MiB` — *"A tree still carrying the old hardcoded literals would have mis-dispatched
> every f16 GEMM on this card"* (`bench/gpu/l4/2026-08-09-session.md`). The A100/H100 rows remain
> DERIVED. H100's crossovers move up by `√(50/24) = 1.44×` in N from the 4050, not 2.04×.
>
> §5.3's shape table below is keyed to the **old** 1.33×/4× crossovers; its `× 50 MiB L2` column is
> still correct (it is a ratio), but read the "regime" column against **0.67× and 2.0×**, not 1.33×
> and 4×. Four rows move, and the *sign* of the headline row inverts:
>
> | shape | ×L2 | §5.3 says | under the shipped rule |
> |---|---|---|---|
> | **4096³** | 1.28× | "just under the raster crossover" | **over it — raster** |
> | GPT d=1024, M=16384 | 0.80× | L2-resident | **raster** |
> | GPT d=4096, M=2048 | 2.88× | raster | **raster + streaming C** |
> | GPT d=4096, M=4096 | 3.20× | raster | **raster + streaming C** |
>
> `GPT d=768, M=16384` (0.57×) does **not** move — it stays under `2/3 × L2` — so §5.3's structural
> note 1 (A+B and C cross L2 at different shapes; the tree conflates them) survives the correction
> unchanged, and is still the open design item. Note 2 (raster width should key on `sm_count`, not a
> literal) is also still open: `sm_count()` is now a real probe, but the raster width `r16` is still
> a constant in the variant tables.

### 5.3 Where the requested shapes land on H100 (DERIVED)

| shape | ws (A+B, f16) | × 50 MiB L2 | C (f32) | H100 regime |
|---|---|---|---|---|
| 1024³ | 4.0 MiB | 0.08× | 4 MiB | deeply L2-resident |
| 2048³ | 16.0 MiB | 0.32× | 16 MiB | **L2-resident** (the 4050 called this "large") |
| **4096³** | **64.0 MiB** | **1.28×** | 64 MiB | **just under the raster crossover — the interesting boundary** |
| 8192³ | 256.0 MiB | 5.12× | 256 MiB | raster **+ streaming C** |
| GPT d=768, M=2048 | 7.5 MiB | 0.15× | 24 MiB | L2-resident |
| GPT d=768, M=16384 | 28.5 MiB | 0.57× | 192 MiB | L2-resident (but C ≫ L2 — a `.cs` *epilogue-only* candidate) |
| GPT d=1024, M=4096 | 16.0 MiB | 0.32× | 64 MiB | L2-resident — **currently mis-dispatched to the raster arm** |
| GPT d=1024, M=16384 | 40.0 MiB | 0.80× | 256 MiB | L2-resident, C ≫ L2 |
| GPT d=4096, M=2048 | 144.0 MiB | 2.88× | 128 MiB | raster |
| GPT d=4096, M=4096 | 160.0 MiB | 3.20× | 256 MiB | raster |
| GPT d=4096, M=16384 | 256.0 MiB | 5.12× | 1024 MiB | raster + streaming C |
| GPT d=8192, M=2048 | 544.0 MiB | 10.9× | 256 MiB | raster + streaming C |
| GPT d=8192, M=4096 | 576.0 MiB | 11.5× | 512 MiB | raster + streaming C |
| GPT d=8192, M=16384 | 768.0 MiB | 15.4× | 2048 MiB | raster + streaming C |

**Two structural notes the current single-threshold scheme cannot express:**

1. **The A+B working set and the C working set cross L2 at different shapes.** `GPT d=768, M=16384`
   has ws = 0.57×L2 but C = 192 MiB = 3.8×L2 — the *right* answer there is the plain mainloop with a
   `.cs`-hinted **epilogue only**. The tree's two arms conflate "big inputs" with "big output"
   because on the 4050 the two crossed at nearly the same N. **Recommend three predicates:
   `inputs_exceed_l2`, `output_exceeds_l2`, `wave_count`** — each derived from a probed number.
2. **Raster width should key on `sm_count`, not on a literal.** Perimeter-optimal band width is
   `G* = sqrt(W · BM / BN)` where `W = sm_count × CTAs_per_SM`. 4050: `W=40 → G*=6.3` (r16 shipped);
   H100: `W=264 → G*=16.2` (r16 also plausible) — **so r16 may be target-independent by accident, and
   A5 is the one point that settles it.** Do not port the literal; port the formula.

---

## 6. What this changes in `GPU_RETARGET_PLAN.md`

> **STATUS 2026-08-09 — five of these ten are now landed or answered. Read the ledger before acting
> on a row.**
>
> | Row | Status at HEAD |
> |---|---|
> | 1 (dynamic SMEM is the only unlock; two `.shared` arrays must become one `.extern`) | **LANDED.** One module-scope window (`gpu::DSMEM_DECL`), per-slab constant offsets, `SmemMode`, `Gpu::function_dyn` / `function_smem` / `dyn_launch_cfg`. Proven on the metal by `dynamic_smem_window_exceeds_the_static_48_kib_ceiling`; D6 §5 carries the seam's design. |
> | 3 (`sm_90a` is architecture-locked; the emission rule needs a third category) | **LANDED as a rule.** Module headers come from `ptx_target` at per-family floors — `sm_80` for the Ampere-legal families, `sm_89` only for fp8 — enforced crate-wide by the textual law `ptx::every_dispatched_ptx_family_opens_at_the_sm80_floor`, and the fp8 launchers are capability-gated through `Gpu::require_fp8` (`gpu.rs:410`). **The architecture-locked third category this row asked for exists**: `ptx_target::HDR_SM90A_V80` / `TARGET_SM90A`, documented as *"`sm_80`/`sm_89` are floors, `sm_90a` is a lock"*, used by the `ptx_wgmma` family and guarded by a test for the one-byte `sm_90` vs `sm_90a` trap. |
> | 7 (generalise the `swz` XOR phase off `bk == 32`) | **STILL OPEN**, re-verified: `assert_eq!(bk, 32)` at `ptx_wmma.rs:2131` and `:2599`. Still $0 and still PTX-text-gateable; still a prerequisite for CUTLASS's BK=64 tile. |
> | 8 (the 16/48 MiB literals) | **HALF LANDED.** f16 derives from probed L2; **bf16 still carries the literal** (`gpu.rs:2494`). The inputs-vs-output predicate split is **not** landed. See §5.1/§5.2's notes. |
> | 9 (`GpuTarget` must carry a *probed* `smem_per_block_optin`) | **LANDED.** `GpuTarget::smem_per_block_optin` is a driver query with no default; `Gpu::smem_budget()` is what dispatch declines against, and the wide tiles capability-skip loudly on a part that cannot hold them. |
>
> Rows 2, 4, 5, 6 and 10 are unchanged predictions/observations; row 5's Act-2 business case should
> now be read with §2.3's C7511 corroboration attached.

| # | Finding | Plan impact |
|---|---|---|
| 1 | **The 48 KiB static-SMEM cap is a PTX-ISA rule for non-`a` targets (§1.6), not a 4050 fact.** | §2.3 calls it a "capability gap" — correct, but the plan implies retagging helps. It does not. **Only the dynamic-SMEM path (Phase 2 step 4) unlocks it on plain `sm_90`.** Phase 2 step 4 is therefore not optional for *any* datacenter tile work. Also: the two `.shared` arrays must become one `.extern` array with computed A/B offsets — a real generator change for C2/C3, not a launch-config change. |
| 2 | **H100's register file, regs/thread, and regs/CTA are identical to Ada's (§1.1).** | §2.4 says the tuned constants "must be re-earned". True — but **the register-limited ones will re-earn to the *same* value**, and the warp-slot denominator gets *worse* (48 → 64 slots, same 16 warps). Predict `_mc3` still loses; predict `w22` still ties. This removes ~8 sweep points. |
| 3 | **`wgmma` requires `sm_90a` — architecture-specific, not forward-compatible (§1.4).** | §5 Phase 4 already plans `sm_90a`; make explicit that **`sm_90a` modules will not JIT onto sm_100/sm_120** and that the `GpuTarget` emission rule ("lowest legal target per module family", Phase 2 step 2) needs a third category: *architecture-locked*. |
| 4 | **TMA needs a host-side `cuTensorMapEncodeTiled` 128-byte descriptor (§1.5).** | Not mentioned in §5 Phase 4's scope list (which names "TMA descriptors" but reads as a PTX-side item). It is a **cudarc/driver-API surface** the crate does not have; check `cudarc 0.16` exposes it (Risk #3 territory). |
| 5 | **Act 1's ceiling is computable: ≤ 90% of cuBLAS @4096³, ≤ 81% @8192³, realistically 66–72% (§4.2).** | §5 Phase 4's go/no-go gate wanted the Act-1 gap "measurable in advance". **It now is.** The wgmma business case is a predicted **1.4–1.6×** on large GEMM — enough to justify Act 2 before spending the H100 hours, and enough to *scope the claim honestly in advance*. |
| 6 | **Wave quantization is a non-problem at every realistic transformer GEMM shape (§2.4).** | §5 Phase 3 step 2 lists "wave quantization" among the things to sweep. **It resolves on paper**: `M·N ≥ 132·BM·BN`. Only `1024³`-class shapes need an arm. That is one confirmation point (A6), not a sweep dimension. |
| 7 | **CUTLASS never ships a BK=32 f16 SM90 mainloop — the K-tile is always 64 (§3.1).** | Wukong's BK=32 is a 48-KiB artifact *and* its `swz` XOR phase is hard-asserted to `bk==32` (`ptx_wmma.rs:1304`). **Generalising the swizzle phase to `nc = bk/8 = 8` is $0, at home, PTX-text-gateable** and is a prerequisite for ever matching CUTLASS's tile. Add it to Phase 2 / the C-track. |
| 8 | **The 16/48 MiB dispatch literals mis-fire on H100 in two named, predicted ways (§4.3).** | Phase 2 step 6 ("dispatch de-literaling") already plans this; §5.2 above gives the exact rule and the exact H100 crossovers (N ≈ 4180 and N ≈ 7240), and §5.3 gives the shape table to gate it with. Also: **the ws predicate must split into an inputs-vs-L2 and an output-vs-L2 predicate** — one number cannot serve both. |
| 9 | **`sm_120a` caps static SMEM at 100 KB, not 228 KB (§1.6).** | Relevant to agent D4. The `GpuTarget` descriptor must carry `smem_per_block_optin` as a *probed* number and the generators must not assume "datacenter ⇒ 227 KB". |
| 10 | **The skinny-M / decode GEMM path falls through every guard to an un-staged kernel (§4.3.4).** | Not in the census's list. On a 132-SM part this is the serving story's floor. Worth a Wave-1 or Wave-2 owner. |

---

## 7. Open questions — measure these first, they are load-bearing

1. **`BW_L2` on H100 SXM5.** Everything in §2.5 keys on 8.2 TB/s (scaled from an H800 PCIe
   measurement). One `st.global`/`ld.global` L2-resident microbenchmark in the first invocation settles
   it. If it is 12 TB/s, the 128×128 tile's ceiling moves 525 → 768 and A3's predicted margin halves.
2. **Does the driver accept a >48 KiB `.extern .shared` allocation from a plain `.target sm_90`
   module after `cuFuncSetAttribute`?** The PTX ISA text (§1.6) is explicit only about *static* SMEM.
   This is the single assumption Phase 2 step 4 rests on. **It is testable on the 4050 today** (99 KB
   opt-in on `sm_89`) for $0 — do that before renting anything.

   > **ANSWERED — YES, at home, for $0 (2026-08-09).** The permanent gate is
   > **`gpu.rs::dynamic_smem_window_exceeds_the_static_48_kib_ceiling`**, and it proves it **in both
   > directions**, which a one-directional test could not:
   > * **with** the opt-in — a `.extern .shared` window of **64 KiB** (WANT = `64 * 1024`, well past
   >   the 48 KiB static ISA ceiling) is requested through `Gpu::function_dyn` + `dyn_launch_cfg`,
   >   the launch succeeds, and every lane is asserted **exact at the far end of the window** (each
   >   thread reads a slot near the top, so a window that is not really that large cannot pass);
   > * **without** it — the identical PTX under a second module key, so a fresh `CUfunction` with no
   >   attribute set, and the identical launch is **refused by the driver**. That is what proves
   >   step 1 was bought by `cuFuncSetAttribute` and not by some default.
   >
   > The probe is tagged at the **`sm_80` floor** (`HDR_SM80`), so the proof is Ampere-legal, not an
   > Ada special case. It also carries a legitimate capability skip for a device whose opt-in budget
   > is under 64 KiB. Caveat, stated rather than glossed: this is measured on `sm_89`; the question
   > as posed said *"plain `.target sm_90`"*, and no `sm_90` device has been touched. The ISA
   > argument (§1.6: the 48 KiB rule is about *static* SMEM only) plus D6 §1's device mechanics are
   > what carry it across, and the first H100 hour confirms it for free — every deep row is a
   > dynamic-window launch.
3. **Actual registers/thread ptxas allocates for A3/A4.** §2.3's estimates (~185) are derived from the
   generator's declarations, not from `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)`. If ptxas
   spills at 128 accumulators, A3 collapses and A2 becomes the top lever. **`--emit`-and-`ptxas`-check
   is free at home** (`WUKONG_PTXAS` env knob already exists).

   > **RECLASSIFIED (2026-08-09): this needs a CPU container, not a rented GPU — and not "home"
   > either.** Two corrections in one:
   > * *"free at home"* is **wrong on this box**: there is no `ptxas`, `nvcc` or `nvrtc` installed
   >   here, so `gemm_cliff_ptxas_ab`'s `WUKONG_PTXAS` knob has nothing to point at and the test
   >   takes its skip branch (`gpu.rs:17393-17399`). The production path compiles PTX→SASS with the
   >   *driver's embedded* ptxas via `cuLink`, which is not the standalone tool and does not print
   >   `-v` register counts.
   > * **`ptxas` compiles FOR an arch without needing one, and the CUDA devel image has it.** Proven,
   >   not assumed: `::build_peers --cutlass-arch 90a` built the **CUTLASS `sm90a` profiler (v4.6.1)**
   >   to completion on a **CPU-only Modal container** — *"device: NONE - CPU container (16 cores /
   >   32 GiB). nvcc compiles FOR an arch, it does not need one"* — in 1836.9 s for **$0.258** total
   >   (`bench/gpu/h100/2026-08-09-preflight-build-peers.log`; the metered line reads
   >   `[meter] build_peers: 1837.0s wall = $0.258`).
   >
   > So the honest cost of answering Q3 is **a CPU container**, roughly a rounding error against one
   > H100 minute, and it should be answered **before** any Act-1/Act-2 GPU hour is bought. The same
   > build already produced the answer's shape for the peer: 112 `(C7511)` register-pressure
   > warnings at 256-wide tiles — see §2.3's corroboration note.
4. **cuBLAS's absolute TFLOPS at 1024³ and 2048³ on the rented part.** §4's percentages at those two
   sizes rest on a predicted denominator. Recording it costs seconds and anchors four predictions.
5. **Risk #2 (the "legacy 8×`.b32`" f16 WMMA fragment spelling on sm_90 JIT).** A1 at 1024³ dispatches
   the WMMA path (`pipe_64_s6`), so **A1 is also the smoke test for Risk #2** — put it first.

---

*Sources fetched and text-extracted 2026-08-06: NVIDIA Hopper Tuning Guide; NVIDIA Ada Tuning Guide;
NVIDIA H100 Tensor Core GPU Architecture whitepaper (Tables 1–4, pp. 20/26/39–41); NVIDIA H100 product
page; PTX ISA (§5.1.7, §9.7.9.26.5.2, §9.7.16); CUTLASS `main` — `python/cutlass_library/generator.py`,
`sm90_utils.py`, `sm90_shapes.py`, `include/cutlass/gemm/collective/builders/sm90_common.inl`,
`sm90_gmma_builder.inl`, `include/cutlass/pipeline/sm90_pipeline.hpp`, `include/cutlass/arch/arch.h`;
Luo et al. arXiv:2402.13499; Bikshandi & Shah (Colfax) Hopper GEMM/CUTLASS paper; `pranjalssh/fast.cu`
README + the "Outperforming cuBLAS on H100" worklog; NVIDIA cuBLAS 12.0 Hopper blog.*
