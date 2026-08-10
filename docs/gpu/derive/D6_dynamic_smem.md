> **CORRECTIONS (2026-08-09, after Phase 2 and Waves 1–3 landed).** This dossier was written
> 2026-08-06 against `0e1b2ea`. Three things have since been *measured* or *machine-checked* and the
> original text is left standing with the correction beside it, per this repo's retraction habit:
>
> | § | Original claim | Status |
> |---|---|---|
> | §4.2 row 6 | conv: raising the 44 KiB gate "frees declined shapes" | **REFUTED** — the machine-checked census (`ptx_conv.rs::conv_smem_census_across_target_budgets`, commit `d69e4da`) shows it declined only square filters `R=S>=34`; AlexNet conv1 (11x11) and the ResNet stem (7x7) fit even the *retired* gate. See the note at that row. |
> | §2, conv row | *"gated `<= 45056` (44 KiB), 'no opt-in to the larger Ada banks'"* | **SUPERSEDED** at `d69e4da` — the gate is now `STATIC_SMEM_CAP` (48 KiB) with a `smem_budget` parameter and a dynamic-window arm. |
> | §4.1 caveat | *"The residency-dependent verdicts need L40S (142 SMs, Phase 1) or the target itself"* | **HALF WRONG** — the L40S is Ada and has the *same* per-SM SMEM and the same ~99 KiB opt-in as the dev 4050, so no residency verdict moves there. It answers **SM scaling at the same ISA** only. See the note at that paragraph. |
> | §5 (the whole migration design) | proposal for C2/C3 | **LANDED** — `smem_budget` parameters, `SmemMode`, `DSMEM_DECL`, `Gpu::function_dyn` / `function_smem` / `smem_budget` / `dyn_launch_cfg` all exist. See the §5 status note. |
>
> Nothing in §1 changed: the on-device mechanics it proved by scratchpad probe are now a permanent
> in-tree gate, `gpu.rs::dynamic_smem_window_exceeds_the_static_48_kib_ceiling`, which asserts the
> window **in both directions** (64 KiB launches and is exact at the far end; the identical launch
> without `cuFuncSetAttribute` is refused by the driver).

# D6 — Dynamic shared memory: mechanics, closed forms, the decision model, candidates, and the C2/C3 migration design

**Wave 0, GPU retarget campaign** (`GPU_RETARGET_PLAN.md` §0, §2.3 first row, §5 Phase 2 step 4).
Produced 2026-08-06 against HEAD `0e1b2ea`. **Every mechanics claim in §1 was verified empirically
on the dev box's own RTX 4050 (CC 8.9, driver-JIT, cudarc 0.16.6 — the exact production stack)** by
a scratchpad probe; its full output is in the Appendix and it is re-runnable at $0:

```
scratchpad\dsmem_probe\   (cargo run; needs only the driver + the 4050)
```

Labels: **FACT(device)** = measured by the probe on the 4050 · **FACT(code)** = read from the tree or
the vendored cudarc source at the cited file:line · **FACT(doc)** = quoted from NVIDIA docs (source
given) · **DERIVED** = computed from FACTs · **PREDICTION** = a falsifiable expectation to measure.

---

## 1. MECHANICS

### 1.1 The PTX spelling — verified on-device

- **Declaration**: `.extern .shared .align 16 .b8 dsmem[];` at **module scope**. FACT(device): the
  driver JIT (`cuModuleLoadData`) accepts it under the tree's own header `.version 7.8 / .target
  sm_89`, and the kernel runs. The same declaration placed **inside** an `.entry` body is
  **rejected** (`CUDA_ERROR_INVALID_PTX`) — the extern window must be module-scope. (This is also
  the exact spelling nvcc emits for `extern __shared__` — DERIVED corroboration.)
- **No new PTX `.version` or `.target` is needed.** FACT(device): 7.8/sm_89 suffices; dynamic SMEM
  is launch-time state, not an ISA feature.
- **Taking the address**: `mov.u32 %r, dsmem;` yields the dynamic window's base address inside the
  shared window — **not guaranteed to be 0**. FACT(device): 0 in a kernel with no statics; **3072**
  in a kernel with 3072 B of statics. The generators already form every SMEM address as
  `mov.u32 %aptr, smemX_{name}; add.u32 …` (e.g. `ptx_fp8.rs:340`, `ptx_wmma` staging), so the
  migration is: same idiom, one base symbol, per-slab **generation-time constant offsets**.
- **Mixing static + dynamic in one kernel is legal and non-aliasing.** FACT(device): statics keep
  their own symbols and addresses (statA=0, statB=1024 for 1 KiB/2 KiB `.align 16` arrays); the
  dynamic window lands immediately after the statics (3072), 16-aligned per the declared `.align`.
  Do **not** hardcode placement — always go through the symbol.
- **Static allocations stay hard-capped at 48 KiB even on an opt-in device.** FACT(device), probe 8:
  a 60 KiB static `.shared` declaration fails the JIT with
  `ptxas error: Entry function 'probe_bigstatic' uses too much shared data (0xf000 bytes, 0xc000 max)`.
  FACT(doc), all three tuning guides verbatim: *"static shared memory allocations remain limited to
  48 KB, and an explicit opt-in is also required to enable dynamic allocations above this limit."*
  **Consequence: every >48 KiB slab MUST become the extern window; the static arrays cannot grow.**
- **Multiple `.extern .shared` incomplete arrays ALIAS.** FACT(device): two module-scope externs
  both resolve to the window base (addresses equal). So per-kernel A/B/Bu buffers are carved out of
  ONE window by constant offsets (A at 0, B at `stages·tile_a`, Bu at `stages·(tile_a+tile_b)`).
  Every shipped slab size is a multiple of 1 KiB, so 16-B sub-slab alignment is trivially preserved
  — assert it anyway at generation.
- **One module-scope extern serves every entry in the module.** FACT(device): two entries shared one
  `dsmem`; each launch sizes its own window (`sharedMemBytes` is per-launch, not per-symbol).

### 1.2 The host-side sequence (cudarc 0.16.6 — vendored source cited)

The two-step sequence, both steps confirmed present in the vendored crate and exercised on-device:

1. **Once per (module, entry):**
   `f.set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dyn_bytes)`
   — FACT(code): `CudaFunction::set_attribute(CUfunction_attribute_enum, i32)` at
   `cudarc-0.16.6/src/driver/safe/core.rs:1864` → `cuFuncSetAttribute` at
   `src/driver/result.rs:193`; the enum value (`= 8`) at `src/driver/sys/mod.rs:3407`.
2. **Per launch:** `LaunchConfig { …, shared_mem_bytes: dyn_bytes }` — FACT(code): the field is
   documented *"Dynamic shared-memory size per thread block in bytes"* (`safe/launch.rs:20-24`) and
   is passed straight through as `sharedMemBytes` to `cuLaunchKernel` (`safe/launch.rs:224`).

Verified semantics (all FACT(device), probe 2/3/4):

| Behavior | Measured |
|---|---|
| dyn ≤ 48 KiB, **no** attribute | launches fine (the default cap) |
| dyn = 60 KiB, no attribute | `CUDA_ERROR_INVALID_VALUE` **at launch** |
| attribute = optin (101376), dyn = 101376 | runs; kernel touches the last word correctly |
| dyn = optin + 4 | `CUDA_ERROR_INVALID_VALUE` |
| 3072 B static in the kernel | max dyn = optin − 3072 **exactly**; even `set_attribute(optin−static+16)` itself fails |
| fresh `load_function` of the same name, no re-set | **runs at 99 KiB — the attribute persists per (module, entry)**, driver-side |

FACT(doc), CUDA driver API (`group__CUDA__EXEC`): *"The sum of this value and the function attribute
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES cannot exceed the device attribute
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN."*

The persistence fact means the **single plumbing point is `Gpu::function`** (`gpu.rs:53-66`): set
the attribute right after `load_function` on the first (cached-module) lookup. Re-setting on every
lookup is idempotent and cheap — either policy is correct.

**The budget probe for B0's `GpuTarget`** — FACT(code): all three attributes are in the vendored sys
(`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97` at `sys/mod.rs:1855`,
`…PER_MULTIPROCESSOR = 81`, `RESERVED_SHARED_MEMORY_PER_BLOCK = 111`), and `Gpu::device_attr`
(`gpu.rs:97`) already wraps `cuDeviceGetAttribute`. FACT(device), 4050 ordinal 0:

```
MAX_SHARED_MEMORY_PER_BLOCK        49152     (the non-optin default)
MAX_SHARED_MEMORY_PER_BLOCK_OPTIN  101376    (99 KiB)
MAX_SHARED_MEMORY_PER_MULTIPROC    102400    (100 KiB)
RESERVED_SHARED_MEMORY_PER_BLOCK   1024
```

**Carveout hint**: `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT` is a percent **hint**
(FACT(doc)). Not required: the occupancy probe shows the driver already accounts the full 100 KiB
carveout for a dynamic-SMEM function (§3). Recommendation: do NOT set it in v1 — one fewer variable;
revisit only if a bandwidth-bound family regresses from L1 loss on the datacenter parts.

**The cuLink / cubin-cache route needs NOTHING extra.** FACT(device), probe 5: the same
extern-window PTX compiles via `cuLinkCreate/AddData/Complete` (the verbatim `cubin.rs:27-62` path)
to a 5600-B cubin, loads from file, takes the attribute, and runs the 99 KiB launch correctly.
Static SMEM is baked into the cubin; the dynamic limit is runtime per-function state. The cache key
already hashes the full PTX text (`cubin.rs:84-93`), and any stage/tile/budget variant is different
text — **no key change needed on the cubin side**.

**Occupancy introspection for gates**: FACT(code)
`CudaFunction::occupancy_max_active_blocks_per_multiprocessor(block, dyn_bytes, None)`
(`safe/core.rs:1746`) — already used once in the tree's comments as evidence (`gpu.rs:639`). The
probe used it to validate the §3 model to the byte.

**CUDA graphs** (`graph.rs`, `serving.rs::step_graphed`): `sharedMemBytes` is part of the captured
kernel-node params; the attribute must be set **before** capture. DERIVED (capture semantics) —
flagged as a Phase-1/3 cloud gate; on the 4050 the serving path stays ≤48 KiB static initially.

### 1.3 The per-target budgets (FACT(doc) — tuning guides, fetched 2026-08-06, quoted verbatim)

| | RTX 4050 (Ada, 8.9) | A100 (8.0) | H100 (9.0) |
|---|---|---|---|
| SMEM/SM (max carveout) | **100 KiB** = 102400 B ✓measured | **164 KiB** = 167936 B | **228 KiB** = 233472 B |
| Max SMEM **per block (opt-in)** | **99 KiB** = 101376 B ✓measured | **163 KiB** = 166912 B | **227 KiB** = 232448 B |
| Static-legal (no opt-in) | 48 KiB ✓measured | 48 KiB | 48 KiB |
| Carveout menu (KB) | 0/8/16/32/64/100 | 0/8/16/32/64/100/132/164 | 0/8/16/32/64/100/132/164/196/228 |
| Reserved per block | 1 KiB ✓measured (and ✓in occupancy) | 1 KiB | 1 KiB |
| Warps / blocks / threads per SM | 48 / 24 / 1536 ✓(1536 measured) | 64 / 32 / 2048 | 64 / 32 / 2048 |
| Regfile / max regs per thread | 64K / 255 | 64K / 255 | 64K / 255 |

Sources: Ada tuning guide (*"GPUs with compute capability 8.9 can address up to 99 KB of shared
memory in a single thread block"*, *"CUDA reserves 1 KB of shared memory per thread block"*, *"The
maximum number of concurrent warps per SM is 48 … thread blocks per SM is 24"*); Ampere tuning guide
(*"the A100 GPU enables a single thread block to address up to 163 KB"*, carveout list); Hopper
tuning guide (*"shared memory capacity per SM is 228 KB … maximum shared memory per thread block is
227 KB"*, *"warps per SM remains the same as in NVIDIA Ampere … 64"*, *"thread blocks per SM is
32"*). The plan's §2.3/§3 figures (163/227) are per-block opt-in maxima; 164/228 are the per-SM
carveouts — both are needed: **per-block legality uses optin, residency arithmetic uses per-SM**.

---

## 2. CLOSED FORMS — SMEM(stages, tile, dtype, pad) per shipped family

All derived from the generators' own size computations (file:line = the computation + its 48 KiB
assert). `s` = stages, sizes in bytes. **These formulas are the contract C2/C3 must expose** (each
family already computes them; they must become `pub` accessors so the launch wrapper never
re-derives them).

| Family (generator) | Formula | Shipped instances (bytes) |
|---|---|---|
| f16/bf16 **WMMA pipe** `entry_smem_pipe` (`ptx_wmma.rs:937-945`) | `s·(BM+BN)·BK·2` | `pipe_64_s6` 6·128·16·2 = 24576; `pipe_128_s4` = 32768 |
| f16/bf16 **mma pipe** `entry_mma_pipe` (`ptx_wmma.rs:1307-1312`) | `s·(BM+BN)·ldp·2`, `ldp = swz ? BK : BK+pad` (swz pinned to BK=32) | workhorse pad8 s2 = 40960; **swz s2 = 32768; swz s3 = 49152 (48 KiB exactly — the static wall)**; padded s3 = 61440 ✗static |
| f16 **gated FFN** `entry_mma_gate` (`ptx_wmma.rs:1656-1662`) | `s·(BM+2·BN)·ldp·2` (A + Wg + Wu rings) | 128×64 pad8 s2 = 40960 |
| **fp8 pipe** `fp8_pipe_entry` (`ptx_fp8.rs:202-205`) | `s·(BM+BN)·(BK+pad)` (1 B/elem) | 128×128 BK64 pad16 s2 = 40960 (`FP8_PIPE_*`, `ptx_fp8.rs:134-146`); m64 variant (BM=64) = 15360/stage |
| **fp8 gate** `fp8_gate_entry` (`ptx_fp8.rs:450-453`) | `s·(BM+2·BN)·(BK+pad)` | 128×64 s2 = 40960 (`ptx_fp8.rs:751`) |
| **int8 swz** `gen_int8_smdb_swz_impl` (`ptx_int8.rs:548,575-576`) | `s·(BM+BN)·64` (BK=64 pinned, no pad — the XOR swizzle is the conflict fix) | w64 128×128 s2 = 32768; s3 = 49152 (measured NEGATIVE on 4050, `ptx_int8.rs:941`); 256×128 s2 = 49152 (48 KiB exactly, `ptx_int8.rs:906`) |
| **int4 W4A16** `entry_w4a16` (`ptx_int4.rs:279-280,303-304`) | `(BM+BN)·16·2` single-stage | small (≤ a few KiB) — not SMEM-bound |
| **flash pipe family** (`ptx_flash.rs:906-907,958-959`; wide `:1809-1810`; ws3 `:2535-2536,2575`) | `n_buf · 2 · (16·nkb · D · 2)` = `64·n_buf·nkb·D` (K-slab + V-slab per buffer; keys/stage = 16·nkb) | D=64 db = 8192; D=128 db = 16384; D=128 ws3 = 24576; `_mpw{nkb}` scales by nkb |
| **conv tiled f32** (`ptx_conv.rs:55-58,78,100`) | `((TP+r−1)(TQ+s−1) + KB·r·s)·4`, TP=TQ=16, KB≤8 | ~~gated `≤ 45056` (44 KiB), *"no opt-in to the larger Ada banks"* (`ptx_conv.rs:57`)~~ **SUPERSEDED `d69e4da`**: the formula is now the single source `tiled_smem_bytes`, the gate is `tiled_applies_budget(.., smem_budget)` at `STATIC_SMEM_CAP` = 49152 by default, and above that the tile moves into the `.extern` window. The 44 KiB number never had a mechanism behind it. |
| **conv WMMA implicit-GEMM** (`ptx_conv.rs:642-644,672-674`) | `BM·16·2 + 16·BN·2 + BM·BN·4` — **single-buffered, un-pipelined** | 64×64 ⇒ 4096+16384 = 20480 |
| megakernel / paged_attention / `ptx.rs` reductions | small statics (≤4 KiB) | out of scope — stay static |

Cross-checks (DERIVED, all match the in-tree comments): mma s2 padded = 40 KiB ✓(`:246`), swz s3 =
48 KiB exactly ✓(`:330`), int8 s3 ring = 3·(8+8) KiB ✓(`:939`), fp8 gate "fits 40 KiB easily"
✓(`:416`), gate 128×64 = "40 KiB (= single-B 128×128)" ✓(`:1661`).

**The launch side today**: every one of the ~60 `LaunchConfig` sites passes `shared_mem_bytes: 0`
(grep-verified; e.g. `gpu.rs:1114` `pipe_cfg`, `gpu.rs:2107` `wmma_flash_cfg`, `gpu.rs:2246-2253`
`conv_tiled_cfg` — whose comment `gpu.rs:2242` states the static-array convention). The ceiling
test is `gpu.rs:5586-5591` (asserts every `PIPE_VARIANTS` row ≤ 48 KiB inside
`wmma_pipe_matches_reference_within_tol`).

> **STATUS 2026-08-09 — no longer true, by design.** `Gpu::dyn_launch_cfg` exists (`gpu.rs:223`) and
> the dynamic families route their launches through it; `STATIC_SMEM_CAP` (`gpu.rs:160`) and
> `DSMEM_DECL` (`gpu.rs:174`) are the named constants the generators share. `shared_mem_bytes: 0`
> remains correct for every **static** entry and that is a contract, not an oversight — see §5.2's
> LANDMINE.

---

## 3. THE DECISION MODEL

### 3.1 Residency (verified to the byte on the 4050)

```
CTAs/SM = min(  floor(SMEM_sm / (S_static + S_dyn + 1024)),      ← SMEM term (1024 = CUDA reserved/block)
                floor(65536 / (threads · regs_per_thread)),       ← register term (alloc granularity ~8)
                floor(threads_sm / threads),                      ← warp-slot term
                blocks_sm_cap )                                   ← 24 (Ada) / 32 (A100/H100)
subject to  S_static + S_dyn ≤ optin_block   (and S_static ≤ 48 KiB always)
```

FACT(device), probe 6 (block=128, no statics, 4050): dyn 0→12 (=1536 threads/SM, thread-bound),
16 KiB→5, 32 KiB→3, 33 KiB→2, 49 KiB→2, 50 KiB→1, 99 KiB→1. Every value equals
`floor(102400/(dyn+1024))` clipped by the thread term — **the +1 KiB reservation is real in the
residency arithmetic**, and the driver rates a dynamic-SMEM function against the full 100 KiB
carveout without any carveout hint.

Register anchors from the tree (FACT(code)): the f16 128×128 mma tile (8 warps, 64 f32
accums/thread) runs **2 CTAs/SM register-bound** — verified via the occupancy API, swz and padded
identical (`gpu.rs:636-643`); int8 w64 (4 warps, 128 accums) keeps **~3 CTAs/SM**
(`ptx_int8.rs:925`). With a 64K regfile on all three targets, **register verdicts port unchanged**;
what changes across targets is the SMEM term and the warp-slot pool (48→64).

### 3.2 Stage count s* — the latency-coverage rule

A `cp.async` ring with `s` buffers keeps `s−1` K-slabs in flight. Let `t_k` = cycles a CTA takes to
consume one staged slab (tensor-core math on BM×BN×BK), `L` = global→SMEM fill latency of one slab
(DRAM latency + slab_bytes/achievable per-CTA bandwidth). The pipeline hides memory when

```
(s−1) · t_k  ≥  L        ⇒        s* = 1 + ceil(L / t_k)
```

**Where deeper pipelines STOP paying** — four stop conditions, each already witnessed on the 4050:

1. **Latency already covered**: once `(s−1)·t_k ≥ L`, more buffers are dead weight. FACT: int8
   BK=64 s2 already covers L (int8 mma runs at 2× the f16 rate ⇒ t_k is large); s3 measured
   negative at every size (`ptx_int8.rs:941-949`).
2. **Bandwidth-bound, not latency-bound**: when aggregate in-flight bytes ≥ BW_sm·L (Little's law),
   prefetch depth cannot create bandwidth. FACT: the f16 4096³ regime is HBM-bound; cliff s3 lost
   at both 2048³ and 4096³ on the 4050 (`gpu.rs:615-618`).
3. **The occupancy cliff**: the next stage drops CTAs/SM below what the *non-cp.async* latencies
   (mma issue, epilogue, softmax) need in warp parallelism. FACT: int8 s3 cut 3→2 CTAs/SM on the
   4050 — the stated mechanism of its loss (`ptx_int8.rs:944-946`).
4. **Short K**: buffers beyond `ceil(K/BK)` never fill (the prologue guards them; pure waste).
   Gate `s ≤ ceil(K/BK)` at dispatch.

**Spending priority for a bigger budget B** (DERIVED — this is the ranking rule that replaces a
blind sweep):

1. **Tile area first** (raises arithmetic intensity ∝ harmonic mean of BM,BN — attacks the
   bandwidth bound itself), until the register term pins CTAs/SM at 1–2;
2. **then stages** up to s* (attacks latency), preferring stage growth that does NOT change the
   CTAs/SM verdict (free depth);
3. **leftover → residency** (more CTAs/SM) only while the warp-slot pool is under-filled for the
   latency profile (≥ ~12–16 warps/SM for mma pipelines; more for softmax/epilogue-heavy flash).

**Target shift** (DERIVED): per-SM HBM bandwidth rises ~1.9× (A100) / ~2.6× (H100) over the 4050
while per-SM tensor throughput roughly doubles — the machine balance moves toward compute-bound, so
`L/t_k` grows and s* rises. That is exactly why CUTLASS's SM80 **defaults are 3-stage** —
FACT(doc), `cutlass/gemm/device/default_gemm_configuration.h` (fetched): Sm80 tensor-op defaults are
**ThreadblockShape 128×256×64, WarpShape 64×64, kStages = 3, for both f16 (m16n8k16) and int8
(m16n8k32)**. Their f16 default needs `3·(128+256)·64·2 = 147456 B = 144 KiB` — **only expressible
through the opt-in window; this is precisely the plan's "CUTLASS's SM80 floor" line, now with the
arithmetic attached.**

### 3.3 Per-CTA budget headroom at fixed residency (the "free depth" table, DERIVED)

For a kernel register-pinned at `c` CTAs/SM, the free SMEM per CTA is `SMEM_sm/c − 1024`:

| | c=1 | c=2 | c=3 |
|---|---|---|---|
| 4050 (100 KiB) | 99 KiB | 49 KiB | 32.3 KiB |
| A100 (164 KiB) | 163 KiB* | 81 KiB | 53.6 KiB |
| H100 (228 KiB) | 227 KiB* | 113 KiB | 75 KiB |

(*capped by the per-block optin.) Read: the f16 workhorse (c=2, 16 KiB/stage) gets **s3 on the
4050, s5 on A100, s7 on H100 at zero occupancy cost**; int8 w64 (c=3, 16 KiB/stage) gets **s2 on
the 4050 but s3 on A100 and s4 on H100 at an unchanged 3 CTAs/SM** — the 4050's s3-loses verdict
was an occupancy artifact that the datacenter budgets dissolve.

---

## 4. CANDIDATE TABLES

Grid legality (`M%BM, N%BN, K%BK`) and the swizzle's BK pins (f16 swz: BK=32; int8 swz: BK=64)
carry over unchanged. "CTAs" = predicted CTAs/SM from §3.1 with the tree's register anchors.

### 4.1 Budget 100 KiB — RTX 4050 opt-in, **measurable THIS WEEK at $0**

Ranked by information value **for the datacenter targets** (the flag column):

| # | Candidate | SMEM | CTAs | Predicted verdict + mechanism | Datacenter proxy value |
|---|---|---|---|---|---|
| 1 | **Variable-stage ring correctness grid**: int8 w64 s∈{2..5} (bit-exact ==), f16 swz s∈{2..4}, fp8 s∈{2..4}, at the K-corner shapes (`gpu.rs:5592-5597` pattern: 1 K-tile, K-tiles>stages wrap) | 32–96 KiB | — | must PASS — this is the gate, not a perf bet | **100 % transfers** — the exact PTX that will run on A100/H100 |
| 2 | **fp8 s3/s4**: 128-tile s3 = 60 KiB (1–2 CTAs), s4 = 80 KiB (1 CTA); m64 s4 = 60 KiB, s6 = 90 KiB | 60–90 KiB | 1–2 | **fp8 depth pays** — the fp8 sweep already measured s3 +10/+19 pts inside 48 KiB (`ptx_int8.rs:941-943`); 1 B/elem keeps slabs small so depth is cheap | **High** — Ada fp8 mma.sync ≡ Hopper's programming model; direct H100-fp8 evidence (A100 has no fp8) |
| 3 | **Iso-occupancy depth A/B** (the instrument trick): pad `shared_mem_bytes` with dummy bytes so s2 and s3 run at the SAME CTAs/SM, isolating pure pipeline-depth effect from occupancy | any | pinned | separates §3.2 conditions 1/2 from 3 — the confound that corrupted the 4050's static-cap verdicts | **High** — produces the transferable per-CTA depth curve that predicts A100/H100 where depth is occupancy-free |
| 4 | **int8 256×128 w64×64 s3** (= CUTLASS's int8 shape class) = 72 KiB | 72 KiB | 1 (reg+SMEM) | on 20 SMs, 1 CTA/SM starves — expect a loss **locally** (`ptx_int8.rs:923-925` said exactly this for s2); the per-CTA throughput number is the transferable datum | **Medium-high** with Phase-1 L40S SM-scaling as the co-variable |
| 5 | **flash mp4/mp8 × nkb∈{1,2}** re-audit under the real 100 KiB (mp8+nkb2 = 33 KiB → 3 CTAs·8 warps) | 16–66 KiB | 2–5 | the shelved mp/ws variants were "occupancy verdicts on ~100 KB SMEM" (plan §2.4) — re-run them with the budget actually opted in | **Medium** — warp-count coupling transfers, absolute verdicts don't |
| 6 | f16 swz s3 dynamic (49 152 B — now legal *with* the +1 KiB reservation, unlike static-48-exactly at 2 CTAs… identical footprint) | 48 KiB | 2 | re-confirm the cliff verdict under dynamic launch — expect unchanged (loss) per §3.2-2 | Low (already known) — serves as the dynamic-path regression control |

**Explicit 4050 caveat**: the 100 KiB budget does NOT let the 4050 imitate A100 residency (e.g.
int8 s3 still drops to 2 CTAs locally; on A100 it keeps 3). The 4050's job is #1–#3: correctness,
mechanism isolation, fp8. ~~The residency-dependent verdicts need L40S (142 SMs, Phase 1) or the
target itself.~~

> **CORRECTION (2026-08-09) — an L40S cannot answer a residency or a wide-tile question.** The L40S
> is **Ada, cc 8.9**, with the *same* per-SM shared memory and the *same* ~99 KiB per-block opt-in as
> the dev 4050 (`ptx_wmma.rs`, `WIDE_SMEM_BUDGET`'s doc: *"this laptop and the L40S both cap at
> 99 KiB"*; D2 §1's device table lists 4050 / L4 / L40S as one `sm_89` column). Per-SM SMEM,
> warps/SM (48), blocks/SM and the register file are Ada's on both parts, so **every residency
> verdict the 4050 produces, the L40S reproduces** — `int8 s3` drops to the same CTAs/SM there.
>
> What an L40S *does* answer, and it is the only thing: **SM scaling at a fixed ISA** — 142 SMs
> against the 4050's 20, same PTX, same per-SM limits. That isolates wave quantization, grid sizing,
> split-K/stream-K and any `sm_count()`-derived launch geometry from every other variable. The L4
> round already exercised the identity path at 58 SMs (`bench/gpu/l4/2026-08-09-session.md`).
>
> The wide tiles are **datacenter-only by construction**: `PIPE_WIDE_VARIANTS` (112–144 KiB) is
> generated against `WIDE_SMEM_BUDGET = 166912` (A100's opt-in), and on any 99 KiB Ada part every one
> of its four rows is a **`[skip:capability]`** line in `gpu.rs::gemm_deep_matches_reference_within_tol`
> — the gate asserts that the skipped set is *exactly* the over-budget set, in both directions. So
> **only A100 or H100 can run A3/A4.** Renting an L40S to answer a wide-tile question buys four skip
> lines.

**Correspondingly, the residency-dependent verdicts need the target itself** (A100 or H100), not an
intermediate Ada part.

### 4.2 Budget 163 KiB — A100 (no fp8 — capability-gated off; bf16/int8 carry low-precision)

| # | Candidate | SMEM | CTAs | Predicted verdict + mechanism |
|---|---|---|---|---|
| 1 | **int8 w64 128×128 s3** = 48 KiB | 48 KiB | **3 (unchanged)** | PREDICTION: the 4050's s3 loss inverts to ≥ wash — the occupancy cut (its stated mechanism) is gone; any latency-coverage gain is kept |
| 2 | **int8 256×128 (w64×64 warp) s3** = 72 KiB | 72 KiB | 1–2 | PREDICTION: the A100 int8 winner — matches the CUTLASS Sm80 int8 default shape/stages exactly; the tree's own comment says bigger CTA tiles are where the remaining ~2× lives (`ptx_int8.rs:899-905`) |
| 3 | **f16/bf16 swz 128×128 s4–s5** (64–80 KiB) at 2 CTAs | 64–80 KiB | 2 | PREDICTION: modest gain at HBM-bound sizes (s* rises with per-SM balance); bf16 twin matters most (training GEMMs, `PIPE_BF16`) |
| 4 | **f16 CUTLASS-shape 256×128 BK=64 s3** = 144 KiB | 144 KiB | 1 | the known-good A100 shape (CUTLASS default) — but **requires re-deriving the XOR swizzle for BK=64** (the phase is pinned to bk=32, `ptx_wmma.rs:1304`) → this is codegen work, not a budget flip; schedule behind #1–#3 |
| 5 | flash D=128 mp8 + nkb=2 (33 KiB → 5 CTAs·8 warps = 40 of 64 warps) | 33 KiB | 5 | PREDICTION: the flash lever on A100 is warps-per-CTA × staging width, not raw depth; norm/flash small-S underfill is the §2.7-flagged risk direction |
| 6 | conv tiled: raise the 44 KiB gate to the budget − epsilon; admit KB=8 at large r·s and 32-wide tiles | ≤160 KiB | — | ~~frees declined shapes; low effort, low risk (single formula + gate change, `ptx_conv.rs:51-59`)~~ **REFUTED — see below** |

> **CORRECTION to row 6 (2026-08-09).** *"Frees declined shapes"* is **wrong**, and the commit that
> retired the gate says so: `d69e4da` — *"gpu(conv): retire the 44 KiB SMEM gate for a real budget
> seam — and report that it was declining nothing"*. The deliverable there is a **negative result**,
> machine-checked by `ptx_conv.rs::conv_smem_census_across_target_budgets`:
>
> | budget | bytes | KB=8 | KB=1 |
> |---|---:|---|---|
> | retired 44 KiB gate | 45056 | `R=S<=33` | `R=S<=67` |
> | PTX ISA static cap | 49152 | `R=S<=34` | `R=S<=70` |
> | RTX 4050 opt-in | 101376 | `R=S<=51` | `R=S<=104` |
> | A100 opt-in | 166912 | `R=S<=66` | `R=S<=136` |
> | H100 opt-in | 232448 | `R=S<=78` | `R=S<=162` |
>
> With the tile pinned at 16x16 and `kblock <= 8`, the tiled conv's footprint is
> `((15+R)(15+S) + KB*R*S)*4` — bounded by the **filter**, not by C, H, W or K. So the 44 KiB gate
> declined exactly the square filters `R=S>=34`, and no convolution anyone runs lives in that band:
> the same test asserts that AlexNet conv1 (11x11) and the ResNet stem (7x7) fit **even the retired
> gate**. What actually binds this family is the **1024-thread block cap** (`TILE_P*TILE_Q` threads)
> and the per-thread `KB` accumulators — neither of which a shared-memory budget touches.
>
> **Where a budget IS the only road, and this row should have named it:** the implicit-GEMM CTA
> tile, whose `BM*BN*4` epilogue scratch dominates. 64x64 = 20480 B, **128x64 = 38912 B — still
> static-legal, so the cheapest widening needs no window at all**, and 128x128 = 73728 B, over the
> ISA cap on every target (1 CTA/SM on the 4050, 2 on A100, 3 on H100; all asserted in the same
> census test). The *replacement* row 6, therefore: **conv implicit-GEMM 128x128 (73728 B)**, ≤163
> KiB, 2 CTAs/SM on A100 — predicted a loss on the 4050 at 1 CTA/SM and a win on A100/H100, and
> reachable only through the window.

### 4.3 Budget 227 KiB — H100 (Act 1: Ampere-class kernels forward-JITted; wgmma is Phase 4)

| # | Candidate | SMEM | CTAs | Predicted verdict + mechanism |
|---|---|---|---|---|
| 1 | **fp8 128×128 s4–s5** (80–100 KiB) at 2 CTAs | 80–100 KiB | 2 | PREDICTION: the strongest Act-1 family — fp8 is where mma.sync is least behind cuBLAS-on-Hopper, and depth is cheapest per byte |
| 2 | **int8 w64 s4** = 64 KiB at 3 CTAs; 256×128 s3–s4 (72–96 KiB) | 64–96 KiB | 1–3 | as A100 #1–#2 with one more free stage |
| 3 | **f16/bf16 swz s5–s7** (80–112 KiB) at 2 CTAs | 80–112 KiB | 2 | PREDICTION: diminishing returns past s3–s4 (§3.2-1); publish the measured gap as the wgmma business case (plan Phase 3 step 4) — do NOT expect parity from depth alone |
| 4 | flash D=128: mp8 + nkb=4 (66 KiB → 3 CTAs·8 warps), ws3 wide | 66–99 KiB | 3 | PREDICTION: helps long-S; the real H100 flash story is TMA/wgmma (Phase 4) — keep Act-1 claims scoped |
| 5 | f16 256-wide BK=64 s3–s4 (144–192 KiB) | 144–192 KiB | 1 | only after the BK=64 swizzle derivation lands (A100 #4) |

---

## 5. MIGRATION DESIGN for C2/C3 (Phase 2.4)

> **STATUS 2026-08-09: LANDED, essentially as designed.** Kept verbatim because it is the rationale
> for a seam that now exists, and the "why" is not recorded anywhere else. What shipped, with the
> in-tree names to grep for:
>
> | §5 item | In the tree at HEAD |
> |---|---|
> | 5.1 explicit `smem_budget: usize` parameter, generators never read `GpuTarget` | `entry_mma_pipe(.., smem_budget)` (`ptx_wmma.rs:2085`, assert at `:2148`); `conv2d_ptx_budget` / `tiled_applies_budget` / `conv_wmma_ptx_budget` (`ptx_conv.rs`); the 8 hardcoded `48*1024` asserts are gone, replaced by the budget or by `gpu::STATIC_SMEM_CAP` |
> | 5.2 static ≤ 48 KiB stays byte-identical; window only beyond | `smem_mode_for` + `SmemMode::{Static,Dynamic}`, one module-scope `DSMEM_DECL` (`gpu.rs:174`). Byte-identity is pinned by digest tests (`shipped_conv_ptx_is_byte_identical`, `shipped_wino_ptx_is_byte_identical`, `shipped_entries_are_byte_identical`, `deep_grid_s2_s3_are_the_shipped_cliff_kernels`) |
> | 5.3 `function_dyn` chokepoint next to `Gpu::function` | `Gpu::function_dyn` (`gpu.rs:462`) and `Gpu::function_smem` (`:503`), plus `Gpu::smem_budget` (`:443`) and `dyn_launch_cfg` (`:223`). The over-budget request is a **loud assert at the seam**, naming the kernel, the request and the device ceiling — the driver's own rejection names none of them |
> | 5.4 tests | `dynamic_smem_window_exceeds_the_static_48_kib_ceiling` (both directions, on the metal); `gemm_deep_matches_reference_within_tol` (deep + wide rows, with a two-directional capability-skip accounting); `gemm_deep_declines_over_budget_instead_of_downshifting`; `conv2d_ptx_crosses_into_the_dynamic_window_above_the_isa_cap` (device-free) |
>
> One item is **not** landed and is deliberately still open: the `_dyn` variants do not change the
> shipped dispatch on Ada — the wide rows capability-skip there (see §4.1's correction).

### 5.1 The API decision: **explicit budget parameter — recommended**

Generators take `smem_budget: usize` (bytes) as an argument; **they do not read `GpuTarget`**.
Rationale, in order:

1. **The generators are pure text functions with device-free tests** (the ASCII/structure gates run
   without a GPU — crate hard-rule 1's whole design). Injecting a device query breaks that purity
   and couples every unit test to a probe.
2. **B0's `Gpu::target()` lives host-side**; the dispatch layer (`gpu.rs`, owned by B6 per the §5.0
   ownership rule) already picks variants per shape — it is the natural place to pick per budget
   too: `let budget = g.target().smem_per_block_optin.unwrap_or(49152);`.
3. A budget parameter keeps the generators **sweepable off-device** (candidate enumeration for §4
   tables is a pure function of the budget) and testable for A100/H100 budgets on the 4050.

Concretely per family: the `stages` parameter already exists (`gen_int8_smdb_swz_impl`,
`entry_smem_pipe`, `entry_mma_pipe`, `fp8_pipe_entry` all take `stages`); add `smem_budget` and
replace the 8 hard asserts:

```rust
// was:  assert!(smem_a + smem_b <= 48 * 1024, …)
// now:  assert!(smem_a + smem_b <= smem_budget, …)   // every existing caller passes 49152 ⇒ behavior-identical
```

Sites: `ptx_wmma.rs:942/1312/1662`, `ptx_fp8.rs:205/453`, `ptx_int8.rs:548`, `ptx_conv.rs:58` (the
44 KiB gate becomes `budget − 4096` conservative form), plus the doc lines that state the cap.

### 5.2 Emission rule: static ≤ 48 KiB stays **byte-identical**; the extern window only beyond

```
if smem_total <= 49152  → emit today's static `.shared` declarations, launch shared_mem_bytes: 0
                          (BYTE-IDENTICAL PTX ⇒ zero 4050 regression, cubin-cache warm hits preserved,
                           Phase-2 rule 7's behavioral-identity A/B trivially satisfiable)
else                    → emit ONE module-scope `.extern .shared .align 16 .b8 wk_dsmem[];`
                          + per-slab constant offsets (A=0, B=s·tile_a, Bu=s·(tile_a+tile_b));
                          return SmemMode::Dynamic(smem_total as u32)
```

Generator return contract: `(entry_name, ptx, SmemMode)` where
`enum SmemMode { Static, Dynamic(u32) }` — the launch wrapper must never re-derive the size.
**LANDMINE (verified semantics)**: for `Static`, `shared_mem_bytes` stays **0** — the launch
parameter is *additional* dynamic memory on top of statics; passing the static size again would
allocate it twice and silently halve residency.

### 5.3 Launch-wrapper changes (B6's side of the seam — C2/C3 must NOT touch `gpu.rs`)

One new chokepoint next to `Gpu::function` (`gpu.rs:53-66`):

```rust
fn function_dyn(&self, key, ptx, name, mode: SmemMode) -> Result<(CudaFunction, u32), _> {
    let f = self.function(key, ptx, name)?;
    match mode {
        SmemMode::Static => Ok((f, 0)),
        SmemMode::Dynamic(n) => {
            f.set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, n as i32)?; // idempotent; persists per (module,entry) — verified
            Ok((f, n))
        }
    }
}
```

A `set_attribute` failure is a **loud error** (device budget exceeded / exotic driver) — never a
silent clamp or fallback launch. The config helpers (`pipe_cfg` `gpu.rs:1108`, `wmma_flash_cfg`,
`conv_tiled_cfg`, …) take the returned `u32` instead of the literal `0`.

### 5.4 Tests and asserts

- **`gpu.rs:5587`** (PIPE_VARIANTS ≤ 48 KiB): keep the assert for `SmemMode::Static` rows; when the
  table grows dynamic rows, the assert becomes two-armed — static rows ≤ 49152, dynamic rows ≤ the
  probed `smem_per_block_optin` (skip-or-fail via `with_gpu` when no device). This edit is inside
  `gpu.rs` tests ⇒ **lands as a B6/Wave-2 commit together with the first dynamic table row**, per
  the one-owner rule.
- New device gate (C2/C3 deliverable, lives in the generator files' test mods): for each family,
  the **stage-grid reference gate** — s∈{2..max} at the K-corner shapes (single K-tile; K-tiles >
  stages ring wrap; rectangular multi-CTA), int8 compared `==`, f16/bf16/fp8 by `assert_close`
  against the f16-rounded f64 reference (the `wmma_pipe_matches_reference_within_tol` pattern).
- The probe's boundary checks (attribute-required, optin-exact, static-counts-against-optin) become
  one small device test so a driver regression is caught loudly.
- ASCII gates: the extern declaration is ASCII; existing per-family ASCII gates cover it
  automatically since they scan the generated text.

### 5.5 Cache keys (crate hard-rule 4) and the cubin cache

- `Gpu::function` caches per `&'static str` key and **never re-examines PTX on a hit** — a new
  stage/budget variant MUST be a new key and a new entry name. The tree already encodes variants in
  names (`_s3`, `int8_swz_{bm}x{bn}_w{wm}x{wn}`); keep that discipline: **entry name and module key
  both embed (tile, warps, stages) — and the budget class if it changes the PTX text**.
- The cubin cache needs no change: it hashes the full PTX text (`cubin.rs:88-93`), and probe 5
  proved the cuLink route composes with the extern window.
- The megakernel's `Box::leak` per-PTX key (`megakernel.rs:86`) is unaffected (megakernel stays
  static) but is the §8-risk-9 growth point if it ever grows variants — note, don't fix here.

### 5.6 Correctness risks, ranked, each with its gate

| # | Risk | Why it is the way it goes wrong | Gate |
|---|---|---|---|
| 1 | **cp.async commit-group bookkeeping at variable depth**: steady-state `cp.async.wait_group s−2` and the prologue's `s−1` commit groups; an off-by-one consumes an in-flight slab → **silently wrong C**, only at particular K counts | the immediates are format-string constants; every stage count is a different unroll | the stage-grid reference gate at the K-corners (single K-tile / wrap / rectangular); int8's `==` makes it unmissable |
| 2 | **Ring-cursor wrap**: the shipped 2-stage path toggles by XOR (`bufcA ^= tile_bytes`, power-of-two asserts `ptx_int8.rs:539`); s≥3 must use add+wrap — mixing the two corrupts stage ≥3 | int8's `_impl` already made this transition correctly once (`ptx_int8.rs:499-505`) — port its discipline, and keep its restriction asserts (`:544-547`) until split-K/raster/static-dims thread the ring | same gate as #1 with K-tiles > stages; plus generation-time asserts |
| 3 | **Barrier discipline at depth**: the 2-barrier overwrite protocol per slab must hold for any s; a missing `bar.sync` is a data race that only shows under contention | timing-dependent; passes small tests | stress iterations at both grid extremes locally; **compute-sanitizer on the Phase-1 cloud box** (toolkit available there — free on L4) |
| 4 | **Swizzle/alignment at new sizes**: the XOR phase is *derived* for bk=32 (f16) / bk=64 (int8); BK changes or new pads invalidate it (asserts exist, `ptx_wmma.rs:1304`, `ptx_int8.rs:536`); sub-slab offsets in the single window must stay 16-B aligned | the asserts make it a loud generation failure, EXCEPT a forgotten assert on window offsets | keep the BK pins; add `offset % 16 == 0` asserts; the BK=64 swizzle re-derivation (§4.2-4) is its own reviewed change with its own reference gate |
| 5 | **Module-cache aliasing** (hard-rule 4): two budget/stage variants under one key → second caller silently runs the first's kernel with a wrong-size window; with a larger launch window it still *runs* — wrong tail behavior | cache never re-reads PTX on hit | name/key embed the variant; extend the `every_dispatchable_*_entry_is_defined`-style pin gates to the new variants |
| 6 | **Graph capture ordering**: `set_attribute` during capture, or a graph instantiated before the attribute | capture bakes node params | set the attribute at function-load (5.3 does); a serving-path >48 KiB kernel gets a `step_graphed` gate before it ships |
| 7 | **4050 default-path regression**: any byte change in ≤48 KiB PTX invalidates cubin-cache warmth and risks perf drift | — | the 5.2 emission rule (static path byte-identical) + Phase-2 rule 7's before/after A/B |

### 5.7 Sequencing note for the wave

C2/C3 own the generator files and their test mods only; every `gpu.rs` edit in this design (5.3
wrapper, 5.4 ceiling test, dispatch table rows) is **B6's**, landing after C2/C3's generator API is
merged. The B0 dependency is exactly one field: `GpuTarget.smem_per_block_optin` (attr 97, default
49152 on failure).

---

## Appendix A — probe output (RTX 4050 Laptop, 2026-08-06, cudarc 0.16.6, driver-JIT)

```
=== device attributes (ordinal 0) ===
CC                                : 8.9
MAX_SHARED_MEMORY_PER_BLOCK       : 49152
MAX_SHARED_MEMORY_PER_BLOCK_OPTIN : 101376
MAX_SHARED_MEMORY_PER_MULTIPROC   : 102400
RESERVED_SHARED_MEMORY_PER_BLOCK  : 1024

=== probe 1: module-scope `.extern .shared .align 16 .b8 dsmem[];` under .version 7.8 ===
PTX JIT load: OK
launch 60 KiB dynamic, NO attribute : "DriverError(CUDA_ERROR_INVALID_VALUE, \"invalid argument\")"
launch 48 KiB dynamic, NO attribute : OK
launch 101376 B dynamic, attr=101376      : OK  out=[bytes=101376 sentinel=0x12345678 dsmem_addr=0]
launch optin+4 (101380 B)             : "DriverError(CUDA_ERROR_INVALID_VALUE, \"invalid argument\")"
fresh load_function, 99 KiB, no re-set: OK (attribute persists per module+name)

=== probe 3: static .shared (in-body) + extern dynamic window, one kernel ===
launch dyn=98304 B (optin - 3072 static): OK
  statA[0]=111 statB[0]=222 dsmem[0]=333 (expect 111/222/333 — no aliasing)
  addresses: statA=0 statB=1024 dsmem=3072
set attr to optin-static+16          : "DriverError(CUDA_ERROR_INVALID_VALUE, ...)"
launch dyn=optin-static+16           : "DriverError(CUDA_ERROR_INVALID_VALUE, ...)"

=== probe 6: occupancy (blocks/SM) vs dynamic SMEM on probe_dyn, block=128 ===
  dyn=  0 KiB -> 12 blocks/SM        dyn= 33 KiB -> 2 blocks/SM
  dyn= 16 KiB -> 5 blocks/SM         dyn= 48 KiB -> 2 blocks/SM
  dyn= 32 KiB -> 3 blocks/SM         dyn= 49 KiB -> 2 blocks/SM
                                     dyn= 50 KiB -> 1 blocks/SM
                                     dyn= 64/99 KiB -> 1 blocks/SM

=== probe 5: cuLink (cubin.rs route) on extern-shared PTX ===
cuLink PTX->cubin: OK (5600 bytes)
cubin-loaded kernel, 99 KiB dynamic: OK out=[101376 0x12345678 addr=0]

=== probe 7: declaration-placement variants ===
extern INSIDE entry body: JIT REJECTS: DriverError(CUDA_ERROR_INVALID_PTX, ...)
TWO module-scope externs: accepted; addresses dynA=0 dynB=0 (ALIAS)

=== probe 8: 60 KiB STATIC .shared declaration ===
60 KiB static: JIT REJECTS
ptxas error   : Entry function 'probe_bigstatic' uses too much shared data (0xf000 bytes, 0xc000 max)
```

Incidental re-confirmation: the probe initially tripped the known `%tid` register-name clash
(`ptxas: Unknown video selector '.x'`) — the reason the tree uses `%tix` everywhere
(`ptx_int4.rs:306-307` documents it). The jit-log diagnostic pattern (`lower.rs:4195`,
`paged_attention.rs:934`) was what surfaced it; C2/C3 should keep it at hand.

## Appendix B — external sources

- Ada / Ampere / Hopper GPU tuning guides (docs.nvidia.com/cuda/{ada,ampere,hopper}-tuning-guide),
  fetched 2026-08-06 — all §1.3 quotes.
- CUDA driver API `group__CUDA__EXEC` — cuFuncSetAttribute / MAX_DYNAMIC_SHARED_SIZE_BYTES /
  PREFERRED_SHARED_MEMORY_CARVEOUT constraint text.
- CUTLASS `include/cutlass/gemm/device/default_gemm_configuration.h` (main) — Sm80 defaults
  (128×256×64, warp 64×64, kStages=3 for f16 m16n8k16 and int8 m16n8k32).
- cudarc 0.16.6 vendored source (`~/.cargo/registry/src/index.crates.io-*/cudarc-0.16.6`) —
  `safe/core.rs:1864` (set_attribute), `safe/core.rs:1746` (occupancy), `safe/launch.rs:20-24,224`
  (LaunchConfig/sharedMemBytes), `driver/result.rs:193` (cuFuncSetAttribute),
  `sys/mod.rs:1855,3407` (attribute enums).
