# GPU Retarget Plan — from the RTX 4050 Laptop to datacenter A100/H100

**Status: PLAN, no code written.** Produced 2026-08-06 from (a) an 11-agent verified census of every
RTX-4050 coupling in the tree (5 auditors + adversarial verifier: 21/22 spot-checks confirmed at the
cited file:line, one classification corrected) and (b) live-verified cloud-GPU market research
(4 researchers + a verifier that re-fetched the six load-bearing offers the same day; one stale price
corrected). Every file:line in this document was re-checked against the tree at commit `5686f37`.

---

## Table of contents

0. [**THE MISSION — read this before doing anything**](#0-the-mission)
1. [Executive summary](#1-executive-summary)
2. [How deeply the compiler is fused to the RTX 4050 (the verified census)](#2-the-verified-census)
3. [Target hardware analysis: 4050 vs A100 vs H100, and the sm_89 stepping stones](#3-target-hardware)
4. [Cloud access strategy (prices live-verified 2026-08-06)](#4-cloud-access)
5. [The plan — **the parallel execution model (§5.0)** + six phases](#5-the-plan)
6. [Benchmark methodology on rented GPUs, and how results get recorded](#6-benchmark-methodology)
7. [Budget](#7-budget)
8. [Risks and open technical questions](#8-risks)
9. [What is explicitly OUT of scope](#9-out-of-scope)
10. [Success criteria](#10-success-criteria)
11. [Decisions needed from you](#11-decisions)

---

## 0. THE MISSION

Your mission is to **continuously iterate on this backend until Wukong's GPU story is genuinely
world-class** — a compiler that takes real `.wk` programs through the real pipeline and emits PTX
that beats cuBLAS, cuBLASLt, cuDNN, CUTLASS, FlashAttention and `torch.compile` on the workloads
that actually matter for ML/DL, on hardware real users actually run.

**Before you optimize anything, work out for yourself what those workloads are.** Do not assume they
are the ones that mattered on a 20-SM laptop. Derive it from first principles and from what people
running real models care about: GEMM at the shapes transformers actually use, attention at real
context lengths, the bandwidth-bound norm/elementwise families, quantized inference, end-to-end
latency and throughput at realistic batch sizes. The 4050 taught this project a set of priorities
fitted to a part with 20 SMs, 12 MB of L2 and 192 GB/s. Almost none of those priorities survive
contact with 132 SMs and 3.35 TB/s. **Re-derive them, then relentlessly improve them.**

### A win only counts if the compiler earned it

A number counts only when it comes from **the language and the compiler themselves**: a real Wukong
program, through the real pipeline, producing correct output and fast PTX. A number that comes from
a hand-written kernel, a special-cased shape, a benchmark tuned to the recognizer dialect, or
anything that would not hold for a general program a real user writes **does NOT count and must
never be reported as a win.**

Peers must be **strong and fairly configured**, and on a cloud box with the full toolkit there is no
longer any excuse for a weak one:

- The GEMM bar is **cuBLAS/cuBLASLt with fused epilogues**, not a naive CUDA-C nest.
- The attention bar is **cuDNN and a real FlashAttention build**, not an unfused cuBLAS chain.
- The framework bar is **`torch.compile` with Inductor+Triton**, which works natively on Linux.
  Eager PyTorch is not a bar; every claim measured against it is on notice.
- The int4/int8 bar is **Marlin/machete-class kernels**, not "no library peer exists."

If you find an existing benchmark that flatters this project unfairly, **fix it and re-measure**,
and say plainly in the commit that the old number was wrong.

### Be skeptical of this project's claims — including the ones in this document

Assume the GPU backend does **NOT** beat the vendor libraries until you have proven it, on the target
hardware, with the control column passing, reproduced across rounds. Every `%-of-cuBLAS` figure in
the tree today is an **RTX 4050 number** and predicts nothing about an H100. Several are worse than
that: §2.7 lists claims that were measured against a peer weakened by Windows, by a missing toolkit,
or by an eager-mode framework. Treat all of them as **retracted until re-earned**.

"Perfect benchmarks" does **not** mean "we win everything." It means: every published number is
controls-validated, reproducible, scoped to the device it was measured on, and either **exceeds a
strong peer or is honestly reported as not exceeding it.** A truthful loss is a finished benchmark.
A flattering win is an unfinished one. Never let the goal of winning corrupt the instrument — this
project has already caught itself doing exactly that four times, and every one of those retractions
made it stronger.

### THE GPU IS AN INSTRUMENT, NOT A WORKBENCH

**This is the hardest cost discipline in the plan and the one most likely to be violated.** Rented
GPU time is metered by the second. An hour spent *thinking* on an H100 is an hour of thinking you
paid $4 for and could have had for free.

**Everything that is not a device correctness gate or a timed measurement happens OFF the GPU:**

| Do this for free, first | Never do this on a rented GPU |
|---|---|
| Read the PTX ISA docs, CUTLASS sources, vendor tuning guides | Browse documentation at $4/hr |
| Compute occupancy by hand from SMEM/SM, regs/SM, warps/SM | Sweep blindly to discover what the math predicts |
| Derive candidate tiles/stages from the target's SMEM budget and SM count | "Try everything and see" |
| Write the code, and `cargo check --features gpu --all-targets` | Compile on a GPU box |
| Build test binaries on CPU (`::build` attaches **no GPU** — keep it that way) | Interactive iterate-compile-run loops |
| Write the *complete* batch script you intend to run | Sit at a GPU shell deciding what to do next |
| Analyze results, plot, write up, decide next steps | Leave a container idling while you think |

**Research what the architecture WANTS before you rent anything.** A100 has 164 KB of opt-in shared
memory and 108 SMs; H100 has 228 KB and 132. From those two numbers plus the register file you can
*derive on paper* which tile shapes and pipeline depths are even feasible, and rank them, before a
single dollar is spent. The 4050's shipped constants exist because 48 KB and 20 SMs forced them —
the equivalent derivation for the target is free, and it turns a blind 40-point sweep into a
6-point confirmation.

**Predict before you measure.** Write down the number you expect and why, then measure. A prediction
that fails teaches you something about the hardware; a sweep that succeeds teaches you nothing. This
is both cheaper *and* better science.

**Never pay twice for the same fact.** Persist the autotune cache, the cubin cache and every raw
round log as committed artifacts. Batch GPU work: prepare, run, tear down, analyze locally. Prefer
the cheapest device that can answer the question — an `sm_89` L4 at $0.80/hr answers most
SM-scaling and correctness questions that an H100 at $3.95/hr would.

**Rent only when the question genuinely requires silicon**: a device correctness gate, or a timed
number you intend to publish. Everything else is free.

### MAXIMIZE PARALLELISM — serial work is the exception that needs justifying

**Default to parallel. Running one agent where the work partitions is the single biggest waste in
this campaign.** The census that produced this document ran 11 agents and finished in 15 minutes
what would have taken a day serially. Apply that everywhere.

**The doctrine:**

1. **Saturate the concurrency cap, every wave.** The cap is `min(16, cores − 2)` concurrent agents.
   A wave running 3 agents on a machine that allows 12 is leaving 75% of the throughput unused.
   Before launching any wave, ask: *what else could be running right now?* — and launch that too.
2. **Find the critical path, run it alone, then fan out hard.** This campaign's critical path is
   astonishingly short: **one agent** building the `GpuTarget` probe. Almost everything else in
   Phase 2 depends on it and on nothing else, so the shape is a thin stem and a very wide crown.
   Never let the crown execute serially.
3. **Partition by FILE, not by topic.** Two agents assigned "the header work" will collide; two
   agents assigned *disjoint file sets* will not. Every parallel task must name the files it owns,
   and no file may have two owners in one wave.
4. **Pipeline, don't barrier.** Do not make wave *N+1* wait for all of wave *N*. If the fp8 gating
   agent finishes while the header agents are still running, its verifier starts immediately. Use a
   barrier only when a step genuinely needs *all* prior results (a cross-cutting merge, a
   census-wide dedup, an early-exit decision).
5. **Research runs from minute one and never blocks.** Deriving what A100/H100/`sm_120` want from
   their specs, reading CUTLASS configs, and reading the PTX ISA are **$0, need no device, and have
   no code dependencies at all.** Those agents should be in flight before the first line of code is
   written, and their output is exactly what §0's derive-before-you-rent rule consumes.
6. **Verification is parallel too.** Pair every *finding* agent with an *adversarial verifier*, and
   run the verifiers concurrently with the next wave of finders rather than after them. The census
   caught one misclassified fact this way; the CPU campaign retracted four claims this way,
   including one where an agent walked back its own predecessor's win.

**Isolation — mandatory, this repo has a known hazard:**

- **Every subagent gets its own branch AND its own git worktree.** Parallel sessions share ONE
  checkout here, so two agents editing the same working tree corrupt each other silently. Always
  `git worktree add` off the real tip (or launch with `isolation: "worktree"`), never let two agents
  share a checkout, and `git worktree prune` leftovers first — a stale `wf_*` worktree holding a
  branch blocks the next `checkout`.
- Merge **deliberately**: keep both sides' regression tests rather than asserting the merge was
  clean, and re-run the two-part gate after **every** merge, not once at the end.

**Model selection:**

- **Default: Opus 5 at `xhigh` thinking effort** — census work, mechanical retargeting, porting, doc
  sweeps, research, file-partitioned header routing.
- **Fable when the task requires extreme reasoning:** designing the **wgmma + TMA generator family**,
  the `tcgen05`/Blackwell analysis, deriving **dynamic-SMEM pipeline schedules** and their occupancy
  trade-offs, and **adversarial verification of any benchmark claim** before publication. These are
  where being wrong is expensive and subtle.
- Cheap mechanical stages (grep sweeps, mirroring an edit across files) can drop to `low` effort —
  spend the reasoning budget where correctness is subtle, not where the work is typing.

### Commit constantly

**Make 60–80 commits across this campaign**, distributed across the phases (Phase 2 alone is ~25–35).
Every commit follows the repo convention: a conventional-commit subject, a body naming **the defect,
the reproducer, and a `Verified:`/`Gate:` line with unpiped test output**, and **no `Co-Authored-By`
trailer**. Never `git add -A` — stage the files you touched. Commit the retractions too; a commit
that says "the old number was measured wrong, here is the right one" is worth more than a commit
that quietly improves it.

### Do not stop

**Iterate until the GPU backend's numbers clearly and durably exceed the vendor libraries on
general, honestly measured workloads — or until you have proven, honestly, exactly where and why
they cannot, and said so plainly.** A phase is not done when it runs; it is done when its numbers
are controls-validated and reproduced. When a measurement disappoints, find the cause, fix the
cause, and re-measure. When a claim survives, try harder to break it.

There are no scope restrictions here. If something blocks a higher number — a missing capability, a
hardcoded constant, a whole absent codegen family, an instrument that cannot resolve the difference,
a peer that was never fairly configured — **remove the blocker**, including options outside this
document's current assumptions. Nothing is off-limits. Think very carefully. Time does not matter.
Cost matters only in that it must be spent on silicon and never on thinking.

---

## 1. Executive summary

The GPU backend is not merely *tuned for* the RTX 4050 — it is **fused to it at four depths**:

1. **Hard-target:** 67 hardcoded `.target sm_89` sites across 18 files (all inside
   `wukong_codegen_gpu` — nothing outside the crate pins an arch, so this is a single-crate change).
   PTX is forward-compatible only, so **today's output loads on ZERO A100s** (sm_80 is older) and
   forward-JITs on H100 (sm_90) without using any Hopper feature. The compiler **never queries
   compute capability anywhere** — `device_attr` exists (`gpu.rs:97`) but is called for exactly four
   attributes (SM count, L2, mem clock, bus width), never CC, never SMEM, never VRAM.
2. **Tuned-constant:** every tile, warp tile, stage depth, raster width, regime threshold, split-K
   trigger and grid size was swept on a 20-SM, 12 MB-L2, 48 KiB-static-SMEM, 6 GB, power-capped
   laptop part. The autotune cache is keyed by shape only — **a 4050-tuned cache replayed on an A100
   is silently trusted** (`autotune.rs:240`).
3. **Capability-gap:** the codegen cannot express dynamic shared memory (every `LaunchConfig` passes
   `shared_mem_bytes: 0`; a test *asserts* ≤48 KiB), wgmma, TMA, clusters, or `setmaxnreg` (zero
   occurrences crate-wide) — so A100's 164 KB and H100's 228 KB SMEM and H100's entire flagship
   tensor pipeline are **unreachable by construction**, not by tuning.
4. **Instrument & docs:** the whole benchmark methodology is built for a ~7× clock-swinging laptop
   under Windows/WDDM, peers are discovered by Windows DLL names on PATH, PyTorch is compared in
   **eager** mode only because Triton doesn't install on Windows, and ~20 published headline families
   (cuBLAS parity, beats-IMMA, flash standings, 95.7%-of-192 GB/s) are 4050-only numbers.

**The strategy is a multi-target retarget, not a target swap.** The 4050 stays the free daily dev
target; the datacenter parts become additional first-class targets selected by a queried `GpuTarget`
descriptor. Sequencing follows the compatibility asymmetry and §0's rule that silicon is rented only
when silicon is genuinely required:

- **Phase 0** — Linux/cloud portability + CI hole-closing. **$0** (CI is already green on ubuntu).
- **Phase 1** — bring-up on **rented `sm_89`** (Modal L4 $0.80/hr, RunPod L40S $0.99): today's binary
  and PTX run **unmodified**, so this validates the Linux port, cudarc, the driver-JIT and the
  ~130-test device suite *before a single line of codegen changes*. Bonus: L40S has **142 SMs** — the
  same ISA with 7× the SMs — so every "does this scale past 20 SMs" question gets a ~$10 answer.
- **Phase 2** — the keystone retarget, **entirely on the laptop at $0**: `GpuTarget` probe, one
  parameterized PTX header through all 67 sites, fp8 capability-gating, dynamic-SMEM launches,
  device-keyed caches. ~25–35 commits without renting anything.
- **Phase 3** — **H100 is the competitive target** (§3): its peers are the strongest and best-tuned,
  so a win there means something. Everything except a native wgmma path forward-JITs day one.
- **Phase 3b** — **A100 for 1–2 hours only**, to prove the header retarget emits correctly toward an
  *older* ISA. `sm_89` already gave us the Ampere programming model for free, so A100 is a validation
  target, not a competitive one.
- **Phase 4** — **wgmma + TMA behind an explicit go/no-go gate.** Competitive H100 GEMM requires it;
  we will NOT pretend Ampere-era kernels can approach cuBLAS-on-Hopper. The Act-1 gap is what makes
  the investment measurable in advance.
- **Phase 5/6** — the rented-GPU benchmark instrument (controls, provenance, MIG checks) and the
  honest re-scoping of every published claim.

**Cost:** roughly **$54–108 out of pocket** (hard cap $150 excluding a separately-approved wgmma Act
2), and plausibly near **$0–50** — Modal's verified **$30/month recurring credit** is ≈7.6 free
H100-hours *per cycle*, so spreading Phase 3 across two months covers most of it, and §0's
derive-before-you-rent rule removes most of the sweep time that would otherwise dominate. There is
also a real grant path (Prime Intellect Fast Compute Grants, $500+, individuals explicitly eligible,
this project squarely in their stated remit).

---

## 2. The verified census

### 2.1 Hard-target surface (all inside `crates/wukong_codegen_gpu/src`)

| What | Where | Verified fact |
|---|---|---|
| `.target sm_89` headers | 67 sites, 18 files (ptx.rs 8, paged_attention 8, ptx_int8 7, ptx_conv 7, ptx_fp8 6, ptx_autodiff_bwd 6, ptx_wmma 5, ptx_winograd 4, ptx_fp8_train 3, ptx_int4 3, lower 2, ptx_optim 2, +1 each baselines/cubin/gpu/ptx_flash/ptx_gemm/ptx_norm) | Verifier grep-confirmed count; **zero arch strings outside the crate** |
| `.version` split | 7.8 everywhere except 8.4 in fp8/int8 families (55 version sites) | 7.8 predates wgmma/TMA (need 8.0+) |
| The gpu-native backend | `lower.rs:178` and `:258` — both headers for ALL MIR→PTX output | One shared `header()` fn is the single retarget point |
| NVRTC peers | `compute_89` at `baselines.rs:133,246,1165,1267,1416` + `gpu.rs:11862`; ptxas `-arch=sm_89` at `gpu.rs:12371`; embedded peer PTX literal `baselines.rs:2144` | 8 more sites the header helper must feed |
| Tests that pin the literal | `ptx_fp8.rs:1072`, `ptx_int8.rs:1377`, `paged_attention.rs:820/847/866/881` | **The paged_attention asserts run under plain `cargo test` (un-gated)** — the header retag must edit them in the same commit or the toolchain-free gate breaks |
| fp8 family | `mma.sync…e4m3/e5m2` (`ptx_fp8.rs:109` +4 sites; `ptx_fp8_train.rs:166`), `cvt.rn.satfinite.e{4m3,5m2}x2` (`ptx_fp8_train.rs:322`) | **Architecturally impossible on A100** — 12 forward entries + 2 backward + the quantize kernels. Works on H100 via forward-JIT |
| Everything else | f16/bf16 wmma + `mma.sync m16n8k16` + `ldmatrix` + `cp.async` (sm_80/75-legal), int8 `m16n8k32 u8.s8` (sm_80), int4 `lop3`/f16x2 wmma (sm_80) | **Header-only retag unlocks the whole non-fp8 kernel inventory on A100** |

### 2.2 The keystone gap: no device identity anywhere

- `Gpu` never queries **compute capability**, **SMEM per SM/block**, or **total VRAM** (workspace
  grep: zero hits for `COMPUTE_CAPABILITY`, `TOTAL_MEMORY`, `cuMemGetInfo`).
- `sm_count()` falls back to **`.unwrap_or(20)`** — the 4050's SM count — on query failure
  (`gpu.rs:111`).
- **The L2 probe is dead code for dispatch**: `l2_cache_size()` (`gpu.rs:118`) was added expressly to
  settle the 24-vs-12 MB L2 confusion its own comments carry, yet its only consumer is a bench
  diagnostic print (`gpu.rs:12228`); the f16 GEMM regime dispatch still uses literal 16/48 MiB
  thresholds (`gpu.rs:621/632`).
- Autotune cache key = `<dtype> <m> <n> <k>` only (`autotune.rs:240`); cubin cache covers arch only
  because `.target sm_89` happens to be inside the hashed PTX text (`cubin.rs:88`).
- Device ordinal hardcoded to 0; one singleton context; no multi-GPU (`gpu.rs:29,101`).

### 2.3 Capability gaps (what A100/H100 have that we cannot express)

| Gap | Evidence | Unlocks |
|---|---|---|
| **Dynamic SMEM** | 48 KiB asserts in 8 generators (`ptx_wmma.rs:942/1312/1662`, `ptx_fp8.rs:205/453`, `ptx_int8.rs:548`, conv's 44 KB gate `ptx_conv.rs:58`); every `LaunchConfig` has `shared_mem_bytes: 0`; no `cuFuncSetAttribute` anywhere; a test asserts the ceiling (`gpu.rs:5587`) | A100 164 KB / H100 228 KB per block → 3–5-stage pipelines, 256-wide tiles. **This is CUTLASS's SM80 floor and the main A100 GEMM lever** |
| **wgmma / TMA / clusters / setmaxnreg** | zero occurrences crate-wide (verifier-confirmed grep) | The entire Hopper flagship pipeline; without it, H100 runs Ampere-style code at a structural deficit vs cuBLAS |
| **Megakernel = ONE SM** | grid `(1,1,1)` × `MEGA_BLOCK=256`, plain `b.launch`, not `cuLaunchCooperativeKernel` (`megakernel.rs:38,109-118`); scratch caps block at 1024 (`lower.rs:2524`) | On any datacenter part it uses <1% of the machine. Multi-CTA cooperative execution is a redesign, already named "a later increment" in its own docs |
| **GQA** | `KvConfig` assumes kv_heads == q_heads (`paged_kv.rs:64`) | Llama-class serving models are GQA |
| **Flash D-set / backward** | tensor-core flash only D∈{64,128} (`ptx_flash.rs:2764-2838`); attention backward is the materialized O(S²) form, not fused (`ptx_autodiff_bwd.rs:10`) | D=256 models; long-context training |
| **Local-frame ceiling** | single-thread gpu-native puts the whole frame in `.local` (`lower.rs:476`) | Large-array programs fail regardless of VRAM |

### 2.4 Tuned constants that must be re-earned, not ported (headline subset)

- f16 workhorse `mma_nt_f16_128_bk32_s2_r16` (128×128, BK=32, 2 stages, raster 16, pad 8) and the
  whole `PIPE_VARIANTS`/`CLIFF_VARIANTS` regime table (`ptx_wmma.rs:234-342`) — stage depth 2 exists
  *because* of the 48 KiB cap; CUTLASS's SM80 floor is 3.
- int8's shipped **64×64 warp tile** exists because "bigger CTA tiles (256×128) *lose* on the 20-SM
  4050" (`ptx_int8.rs:922-926`) — the literature expects the opposite verdict at 108+ SMs, and the
  sweep hook (`int8_gemm_swz_tile_ptx`, `:909`) already generates arbitrary tiles: **re-tuning, not
  new code**. The 3-stage variant "measured NEGATIVE" is a 48-KiB artifact to re-measure.
- Flash `FLASH_WARPS=2` is derived in-comment from Ada's ~24-blocks/SM cap (`ptx_flash.rs:64-68`);
  A100/H100 allow 32 blocks and 64 warps/SM. `FLASH_TILE_MIN=0` and the ws-flash S≥4096 crossover
  (`gpu.rs:2112`) are frozen 4050 A/B verdicts; the doc itself says the constant exists for a future
  GPU. The shelved mp4/mp8/mpw/msp/hs variants were all occupancy verdicts on ~100 KB SMEM.
- Norm kernels: one warp per row, grid=(rows,1,1) — small row counts underfill 108/132 SMs
  (`ptx_norm.rs:29`).
- Serving/paged-KV: production APIs are properly parameterized (verifier-confirmed — the 4 GiB /
  "6 GB part" constants live in `#[ignore]`d benches and capacity tests, `serving.rs:1991`,
  `paged_kv.rs:626`); the retarget work is a `cuMemGetInfo` probe + ~10 doc sites, not deleting
  shipped constants.
- `gpu_accel.rs:118`: the `--backend=gpu` fused-epilogue offload declines unless M,N %64 and K %16 —
  literals mirroring the 64×64 tiling; must derive from the dispatched kernel's tile constants.
- The ONE correctly-scaling pattern to copy everywhere: `grid_stride_cfg` = 256-thread blocks ×
  `32 * sm_count()` (`ptx_optim.rs:211`).

### 2.5 The test gate (good news, mostly)

- 295 `#[test]`s in the crate; ~130 real **device-executing correctness gates** wrapped in `with_gpu`
  which skip loudly without a device and **escalate to failures under `WUKONG_GPU_REQUIRED=1`**
  (`gpu.rs:4867-4880`), plus 5 driver e2e tests (`wukong_driver/src/lib.rs:1772-2137`) and the two
  corpus-vs-oracle gates (`lower.rs` / `megakernel.rs` sweep `tests/run` at two `-O` levels).
- **The correctness infrastructure already exists and is device-gated, not device-fitted.** Cloud
  bring-up = run the existing suite with two env vars. What does NOT exist: any CI that builds the
  gpu feature — `.github/workflows/ci.yml` never runs `--features gpu` (line 44), so GPU code is
  CI-blind on both OSes.
- Gap found by the census: `ptx_fp8_train.rs` has **no ASCII gate** (crate rule violation, verified
  zero `is_ascii` hits) — add during Phase 2.

### 2.6 Portability (Windows → Linux cloud)

Verified good: CI is already green on **ubuntu-latest** (CPU core + `--emit=exe` rustc-driven link
are proven portable); C-runtime CRLF is `_WIN32`-guarded; `cudarc`'s `dynamic-loading` dlopens
`libcuda.so` on Linux.

Real blockers/adjustments, all cited:

| Item | Where | Action |
|---|---|---|
| Peer redist discovery is Windows PATH + `.dll` names; pinned pip wheels; nested `bin` dirs | `baselines.rs:61`, memory `gpu-peer-dll-path` | Linux arm: system CUDA libs or wheel `lib/` dirs via `LD_LIBRARY_PATH`; make `peer_env_hint` OS-conditional |
| FA2/torch peer paths | `baselines.rs:2295` (`Scripts/python.exe`) | `bin/python` under `cfg(unix)`; on Linux **Triton works** → the peer gets *stronger* (torch.compile) |
| Fault-recovery premise is WDDM | `gpu.rs:204,212`, `docs/roadmap.md:528` | Re-measure primary-ctx reset on the Linux driver; the record-and-skip policy may relax |
| Error text says `nvcuda.dll` | `lower.rs:157`, `wukong_driver/src/lib.rs:405` | Per-OS message |
| `{stem}.exe` naming | `wukong_driver/src/lib.rs:561` | Cosmetic cfg |
| `cudarc 0.16` feature `cuda-12060` (12.9-symbol expectations observed) | `wukong_codegen_gpu/Cargo.toml:16-19` | Verify against cloud images (r550–r580 drivers); Modal hosts run driver 580.95 / CUDA 13.0 API |
| CPU-side only (NOT needed for the GPU campaign): 256-bit AVX2 vectorizer is Win64-ABI-only (`avx2.rs:53`), thread pinning is kernel32-only (`gemm.rs:165`), MKL/vcvars discovery is Windows-only | — | **Deliberately out of scope** (§9): CPU benchmarks stay laptop-scoped; the cloud boxes are GPU instruments |

### 2.7 Published claims that are 4050-only (must be re-earned or re-scoped)

~20 headline families across `BENCHMARKS.md:1856-2421`, `README.md:103-110`, `docs/roadmap.md:437-533`,
`docs/metrics.md:48-60`, plus %-of-cuBLAS comments used as *dispatch rationale* inside the crate
(`ptx_wmma.rs:279`, `gpu.rs:626/635`, `ptx_int8.rs:922`). Three deserve special flags:

1. **"Beats PyTorch at every S"** is vs **eager** only, justified by Triton-not-on-Windows
   (`BENCHMARKS.md:2182,2230`). On Linux the honest peer is `torch.compile` — this claim must be
   re-earned against a much stronger bar and may not survive.
2. **CUDA-graph launch-overhead wins (~4.5–6.9×)** were measured under Windows/WDDM, where launch
   cost is several times Linux's (`BENCHMARKS.md:2314`). Expect the multiples to shrink on Linux
   before the GPU even changes.
3. **The long-S flash loss "structural on 20 SMs"** (`BENCHMARKS.md:2132`, `docs/metrics.md`) —
   the premise dissolves at 108/132 SMs; must be re-diagnosed, direction unknown in our favor or not.
   Conversely the small-S occupancy *wins* depend on 20 SMs being easy to fill and may invert.

Also: the **no-CUDA-toolkit premise inverts on cloud** (stated in four docs; `README.md:274` etc.) —
cloud images ship the full toolkit, which means (a) real CUTLASS/FA2/Marlin peers become buildable
(the "no honest peer possible" framings expire), and (b) an optional ptxas-backed offline-compile
route becomes available. Reframe as "toolkit-optional". `docs/internals.md` is accidentally
triplicated (identical passages at 3 offsets) — de-duplicate before editing it three times.

---

## 3. Target hardware

**For a compiler, GPUs are not a speed ladder — each distinct `sm_XX` is a separate codegen target.**
A faster part with a different ISA is *more* work, not less. That reframing drives every choice below.

| | RTX 4050 (dev) | L4 / L40S / RTX 4090 | A100 80GB | **H100 / H200** | B200 / B300 | RTX PRO 6000 |
|---|---|---|---|---|---|---|
| Arch / CC | Ada **sm_89** | Ada **sm_89** (same!) | Ampere **sm_80** | Hopper **sm_90** | Blackwell **sm_100/103** | Blackwell **sm_120** |
| SMs | 20 | 58 / 142 / 128 | 108 | 132 | 148 | 188 |
| SMEM/block (opt-in) | ~100 KB | ~100 KB | **164 KB** | **228 KB** | larger | smaller (RTX-class) |
| L2 | 12 MB | 48 / 96 / 72 MB | 40 MB | 50 MB | larger | — |
| Memory BW | ~192 GB/s | 300 GB/s – 1 TB/s | ~2 TB/s HBM2e | 3.35 / **4.8** TB/s | HBM3e+ | GDDR7 |
| Tensor-core model | mma.sync | mma.sync | mma.sync | **wgmma + TMA** | **tcgen05 / tensor memory** | mma/mma.sp (FP4/FP6) |
| fp8 | yes | yes | **NO** | yes | yes + FP4 | yes + FP4 |
| Loads today's PTX? | yes | **yes, unmodified** | **NO** (forward-compat only) | yes (forward-JIT, Hopper-blind) | yes (blind) | yes (blind) |
| Modal $/hr | — | 0.80 / 1.95 / — | 2.50 | **3.95 / 4.54** | 6.25 / 7.10 | 3.03 |

Five consequences, in the order they change the plan:

1. **H100 is the top target — not Blackwell.** Not because B200 is slower (it isn't) but because H100
   has the *strongest honest peers*: cuBLAS, cuDNN, CUTLASS and FlashAttention are maximally tuned
   there, so a win means something. It is also the most-rented and most-reproducible part, and wgmma
   is well documented. Beating an immature library on new silicon would be a hollow number.
2. **H200 is a free upgrade in codegen terms** — same GH100 die, **identical `sm_90` ISA**, ~1.4× the
   memory bandwidth and 141 GB. Zero extra codegen for a different bandwidth regime, at +15% on
   Modal. Use H100 for *published* ratios (reproducibility), H200 when a bandwidth-bound question
   needs headroom.
3. **Blackwell (B200/B300) is deferred.** `sm_100/sm_103` uses `tcgen05` tensor-memory paths — a
   **third** codegen family after `mma.sync` and `wgmma`, and the most expensive hardware to iterate
   on. Wrong order of business.
4. **A100 is demoted from "first competitive target" to backward-compat validation** (1–2 hours).
   The original reasoning was that our kernels are Ampere-native — true, but `sm_89` (Ada) uses the
   *same* `mma.sync` + `cp.async` programming model, so an L40S already gives us the Ampere-class
   model on hardware that needs **zero code change**. A100's remaining unique value is narrow: it
   proves the header parameterization works toward an *older* target (a direction sm_89→sm_90 never
   tests), and its ~2 TB/s HBM shifts bandwidth-bound conclusions.
5. **`sm_120` (RTX PRO 6000) is added as a real-user target.** It is the same ISA as an **RTX 5090** —
   what enthusiasts actually own — and is distinct from datacenter Blackwell (`sm_100`): smaller SMEM
   limits, different low-precision paths. At $3.03/hr it is the cheapest way to test the ISA that
   Wukong's eventual local users will run on.

**Revised ladder:** L4/L40S (`sm_89`, free, no change) → **H100 (`sm_90`, the competitive target)** →
A100 (`sm_80`, validation only) → RTX PRO 6000 (`sm_120`, user relevance) → Blackwell datacenter, later.

Three consequences that drive the whole plan:

1. **A100 is the natural first competitive target.** The instruction inventory minus fp8 is already
   sm_80-native; the win path is retag → dynamic SMEM → re-sweep. The fp8 family is gated off with a
   loud capability skip (never a silent fallback), and its bf16/int8 siblings carry the low-precision
   flag.
2. **H100 is a two-act story.** Act 1 (cheap): forward-JIT everything, full correctness suite, scoped
   honest numbers ("Ampere-class kernels on Hopper"). Act 2 (expensive): a new wgmma+TMA generator
   family, `.version 8.x` / `sm_90a`, mbarrier pipelines — effectively a new GEMM/flash backend.
   cuBLAS on H100 *is* wgmma+TMA; parity without it is not achievable and we will not claim otherwise.
3. **Rented sm_89 removes all retarget risk from the first cloud step** — and a 4090's 128 SMs let us
   answer the SM-scaling questions (grid sizing, split-K thresholds, wave quantization, the
   256×128-tile question, megakernel multi-CTA payoff) on the *same architecture* for ~$0.35/hr
   before any A100 hour is spent.

---

## 4. Cloud access

All prices below were **live-verified 2026-08-06** by the verifier agent against provider pages
(exceptions noted). The hyperscaler "free trials" are confirmed traps for this purpose: GCP's $300,
Azure's $200 and OCI's $300 all have **GPU quota locked at 0** until you convert to paid billing
(verified via official docs/forums); AWS's free plan is micro-instances only and sells A100/H100 in
8-GPU nodes anyway. Kaggle/Saturn never reach A100/H100. Thunder Compute is GPU-over-network
virtualization — correctness-only, never a benchmarking instrument.

### 4.1 The chosen stack

| Role | Provider | Verified terms | Why |
|---|---|---|---|
| **Free tier for everything** | **Modal** | **$30/month recurring free credit, no card** (confirmed on pricing page); A100-40GB $2.10/hr, A100-80GB $2.50/hr, H100 $3.95/hr per-second; full devices; driver 580.95.05 / CUDA 13.0 API; arbitrary `nvidia/cuda:*-devel` images (full nvcc/NVRTC/cuBLAS/cuDNN); `modal shell` interactive | ≈12–14 free A100-hrs or ≈7.5 free H100-hrs **every month**. Container (no clock control, host may change between invocations) → keep every A/B inside one invocation |
| **Cheap sm_89 + iteration hours** | **RunPod** (confirmed to the cent) or **Vast.ai** (floats) | RunPod: 4090 $0.34/hr Community / L40S $0.99 Secure; A100 PCIe 80GB $1.19; H100 PCIe $1.99. Vast: 4090 ~$0.35, A100-80 ~$0.96, H100 ~$2.01 (same-day floats) | Per-second billing, custom devel images, persistent volume. Containers → no clock locking; benchmark on Secure/verified-datacenter hosts, pin ONE host per round |
| **Canonical measurement rounds (VM + root)** | **Verda** (ex-DataCrunch) or **Hyperstack**; **Lambda** as the "known-clean" control | Verda (corrected live): H100 SXM **$3.25/hr on-demand / $1.63 spot** (flat −50%, demand-reclaimed), A100-40 $1.29/$0.645, A100-80 $1.79/$0.895 — real root-SSH VMs, CUDA preinstalled. Hyperstack: A100-80 $1.35 ($1.08 spot), H100 PCIe $2.50 ($2.00 spot), per-minute. Lambda: A100-40 $1.99, H100 PCIe $3.29 — Lambda Stack (CUDA 12.8 + cuDNN) preinstalled, most reliable, capacity-constrained | **A VM with root is the only place `nvidia-smi -lgc` clock locking can work** — canonical published rounds happen here |
| Cheapest interactive A100 backstop | Colab Pro | $9.99/mo ≈ 18 A100-40GB hrs (~5.4 CU/hr, third-party-verified); real shell via `colab console` CLI | Smoke tests only — no clock control, no persistence, no guaranteed A100 |
| Grant path (parallel, free) | **Prime Intellect Fast Compute Grants** | $500–$100k credits, "anyone from anywhere", email pitch | An open-source shape-safe tensor compiler with a MIR→PTX backend is squarely their stated remit; the $500 floor ≈ 200+ H100 marketplace hours |

**Corrections the verifier caught (do not budget on the stale numbers):** the researched
"DataCrunch H100 $2.01/hr on-demand" is a March-2026 snapshot, ~60% low — live is $3.25 ($1.63 spot).
RunPod's rumored "$5 signup / $500 randomized credit" appears nowhere on the live page — plan as
zero-free. Vast H100 "1.50–1.89" is optimistic — budget $2.01. The Vultr $250–300 coupon and
Lightning-free-A100 claims are third-party-only — nice-if-true, never load-bearing.

### 4.2 Hard requirements checklist (all satisfied by the stack above)

- CUDA **driver** present (cudarc driver-JIT needs nothing else to *run*): all providers. r525+
  accepts PTX ISA 8.x; observed fleets run r535–r580.
- **Full toolkit for peers** (nvcc/NVRTC headers, cuBLAS, cuBLASLt, cuDNN): via `nvidia/cuda:12.x-devel`
  images (containers) or preinstalled/apt (VMs). This *upgrades* peer honesty vs the laptop.
- **Full device, not MIG**: everything selected above sells whole GPUs; RunPod *Serverless* MIG tiers
  and Vultr's small "Cloud GPU" SKUs are the flagged exceptions. **Every session starts with
  `nvidia-smi` + SM-count verification against the spec** — marketplaces have sold virtualized
  GRID A100D variants as "A100".
- Rust toolchain: rustup in image/VM; bake into the Modal image / persistent volume so metered time
  never pays for compiles twice.

---

## 5. The plan

### 5.0 The parallel execution model — read before launching anything

The phases below are written as numbered steps for *readability*. **They are not an execution
order.** The real dependency graph is a thin stem and a very wide crown, and executing it serially
would waste most of the campaign's throughput.

```
                    ┌── D: architecture research (6) ────────────────┐   $0, zero deps,
                    │      derive what H100/A100/sm_120 WANT         │   START AT MINUTE ONE
   S               ├── A: Linux portability + CI (3) ───────────────┤   $0, zero deps
   T ──────────────┼── E: docs de-dup + scope banners (2) ──────────┤   $0, zero deps
   A               │                                                │
   R               └── B0: GpuTarget probe (1) ══ CRITICAL PATH ══╗  │   $0, ~1 file
   T                                                              ║  │
                    ┌─────────────────────────────────────────────╝  │
                    │                                                │
      WAVE 1        ├── B1..B6: header routing, FILE-DISJOINT (6) ───┤   $0
      (fan out      ├── C: dynamic SMEM (3, fed by D6) ──────────────┤   $0, 4050 has ~100 KiB!
       maximally)   └── P1: rented sm_89 correctness ────────────────┤   runs CONCURRENTLY —
                          (needs only A, not B) ─────────────────────┘   it tests the PRE-retarget tree
                                        │
                                        ▼
      WAVE 2        V1..V6 adversarial verifiers ‖ merge coordinator ‖ 4050 regression gate
                                        │
                                        ▼
                              Phase 3 (rented silicon) — derivation already done by D
```

**Wave 0 — launch ~12 agents immediately. Every one is $0 and needs no device.**

| Track | Agents | Work | Model |
|---|---|---|---|
| **D — Architecture research** | 6 | D1 H100 GEMM: derive feasible tile/stage/warpgroup configs from 228 KB SMEM + 132 SMs + regfile; read CUTLASS SM90 configs. D2 H100 attention/TMA staging. D3 A100 (164 KB, 108 SMs). D4 `sm_120` consumer Blackwell. D5 how to *build* CUTLASS/FA2/Marlin/`torch.compile` peers on a cloud box (exact commands). **D6 dynamic-SMEM pipeline-scheduling theory** | Opus `xhigh`; **D6 = Fable** |
| **A — Portability + CI** | 3 | A1 CI job `cargo check --features gpu --all-targets`. A2 Linux peer-discovery arms + per-OS messages + unix exe suffix. A3 the missing `ptx_fp8_train` ASCII gate | Opus `xhigh` |
| **E — Docs** | 2 | E1 **de-duplicate the triplicated `docs/internals.md` FIRST** (any doc edit before this must be applied three times). E2 device-scope banners on BENCHMARKS/README/roadmap/metrics | Opus `xhigh` |
| **B0 — CRITICAL PATH** | 1 | `GpuTarget` descriptor; kill `.unwrap_or(20)`. Small, fast, and **everything in Wave 1 waits on it** — so give it a dedicated agent and merge it the moment it gates | Opus `xhigh` |

**Wave 1 — the crown. Fan out the moment B0 merges.** The 67 header sites partition **exactly** and
disjointly, so six agents can route them with zero conflicts:

| Agent | Files owned (exclusive) | Sites | Also does |
|---|---|---|---|
| B1 | `ptx.rs`, `ptx_gemm.rs`, `ptx_wmma.rs` | 14 | — |
| B2 | `ptx_fp8.rs`, `ptx_fp8_train.rs`, `ptx_int8.rs`, `ptx_int4.rs` | 19 | fp8 capability gating; split `AMAX_PTX` out |
| B3 | `ptx_flash.rs`, `ptx_norm.rs`, `ptx_conv.rs`, `ptx_winograd.rs`, `ptx_optim.rs`, `ptx_autodiff_bwd.rs` | 21 | — |
| B4 | `paged_attention.rs`, `lower.rs`, `megakernel.rs` | 10 | **the 4 UN-GATED `sm_89` test asserts** — must change in the same commit |
| B5 | `baselines.rs`, `cubin.rs` | 2 | NVRTC `compute_XX` ×5; cubin key += arch |
| B6 | **`gpu.rs` — SOLE OWNER** | 1 | ptxas `-arch`; dispatch de-literaling; autotune cache key; **the dynamic-SMEM launch path**; serving/KV budget probes |
| C2, C3 | generator SMEM parameterization; **4050 ~100 KiB validation** | — | must NOT touch `gpu.rs` |
| P1 | *(rented L4 — see Phase 1)* | — | runs concurrently; tests the pre-retarget tree |

> ⚠ **`gpu.rs` is the contention hotspot.** It is ~4.8k host lines touched by the header work, the
> de-literaling, the launch path, the cache keys and the budget probes — five workstreams that would
> collide catastrophically. **Exactly one agent owns `gpu.rs` per wave (B6).** Everything that needs
> a `gpu.rs` change either belongs to B6 or waits for Wave 2. This single rule prevents most of the
> merge pain this campaign could generate.

**Wave 2 — verify and integrate, still parallel.** One adversarial verifier per Wave-1 branch, all
concurrent; a merge coordinator merging sequentially with the two-part gate after *each* merge (never
a big-bang merge); and the 4050 regression agent proving the dispatch census, suite results and
same-run perf A/B are unchanged. Verifiers start as their target branch lands — do **not** barrier
on all six.

**Two things that must stay serial**, and why: `B0` (everything depends on the descriptor's shape),
and the **merge sequence** (parallel merges into one branch is how you lose a regression test).
Everything else in this document is parallelizable, and should be.

### Phase 0 — Portability + CI (local + free CI; no GPU spend)

Goal: a Linux box can build, test, and run the GPU path; CI stops being blind.

1. Add the missing CI job: `cargo check --features gpu --all-targets` on ubuntu (types-only, free —
   closes the "CI never builds gpu" hole at `.github/workflows/ci.yml:44`).
2. Linux arms for peer discovery (`baselines.rs:61` hint text, `.so` names, `LD_LIBRARY_PATH`,
   `bin/python` at `baselines.rs:2295`), per-OS `nvcuda.dll`/`libcuda.so` messages
   (`lower.rs:157`, `driver/lib.rs:405`), unix `--emit=exe` suffix (`driver/lib.rs:561`).
3. Provisioning script (`tools/cloud/` + a `Dockerfile`): FROM `nvidia/cuda:12.x-devel-ubuntu22.04`,
   rustup, the repo, `cargo build --features gpu`, entry points for the suite and each bench. One
   image serves Modal, RunPod, Vast and any VM.
4. Add the missing `ptx_fp8_train` ASCII gate (crate-rule violation found by the census).

Exit: `cargo test` + `cargo check --features gpu --all-targets` green on Linux locally/CI; image builds.

### Phase 1 — sm_89 stepping stone: validate everything, change nothing ($7–14)

Goal: prove Linux + cudarc + driver-JIT + the whole device suite on rented hardware **before any
codegen change**, and measure the suite's wall-clock so later metered phases are budgetable.

Cheapest first (§0): **Modal L4 $0.80/hr** for correctness, **L40S** (142 SMs, Modal $1.95 / RunPod
$0.99) for the SM-scaling work. Both are `sm_89` — today's PTX runs unmodified.

1. **Correctness, peers OFF first.** `WUKONG_GPU_REQUIRED=1 cargo test -p wukong_codegen_gpu
   --features gpu` → all ~130 device gates must PASS (not skip). Run **without**
   `WUKONG_PEER_REQUIRED` on the first pass so a missing library is *reported* rather than failing
   the whole suite; turn it on for the second pass once you know what resolves. (`tools/cloud`'s
   `::test` defaults `--no-peers` for exactly this reason.) Then the driver e2e tests and both
   corpus-vs-oracle gates. **Record total wall time** — this number sizes every later phase.
2. Linux behavior checks the census flagged: sticky-fault recovery (WDDM premise, `gpu.rs:204`),
   CUDA-graph capture on the r550+/r580 driver (`graph.rs:132`, `serving.rs:1335`), cubin cache on
   the mounted Volume.
3. **The SM-scaling experiment** (L40S, 142 SMs vs the 4050's 20 — same ISA): re-run the f16/int8
   sweep benches and the flash A/B suite unmodified. Every "does X scale past 20 SMs" question —
   256×128 tiles, 3-stage pipelines within 48 KiB, split-K thresholds, raster widths, ws-flash
   crossover, norm-kernel underfill — gets a **same-architecture** answer for pocket change, with no
   codegen variable confounding it. These results directly seed the Phase 3 candidate grids.
4. Optional same-run scoreboard vs cuBLAS on the L40S — **not headline numbers** (different power
   class, GDDR6 not HBM), but the first datacenter-adjacent calibration of the 4050-measured
   "%-of-cuBLAS" comments.

Exit: suite green on Linux `sm_89`; suite runtime known; SM-scaling findings written up.

### Phase 2 — The keystone retarget (local dev on 4050; $0 until validation)

Goal: one device-derived `GpuTarget`, zero hardcoded arch facts. All file:line refs are §2 citations.

1. **`GpuTarget` descriptor** on `Gpu`: `{cc_major/minor, sm_count, smem_per_block_optin, total_mem,
   l2_bytes, name, driver_version}` via `cuDeviceGetAttribute`/`cuMemGetInfo`. Kill the
   `.unwrap_or(20)` (loud error instead).
2. **One `ptx_header(target)` helper**; route all 67 sites in 18 files; update the six literal-string
   test asserts **in the same commit** (the 4 paged_attention ones are un-gated); feed the 6 NVRTC
   `compute_XX` sites and the ptxas `-arch`. Emission rule: lowest legal target per module family
   (sm_80 for everything sm_80-legal, sm_89 for fp8, later sm_90a for wgmma modules), `.version`
   per family floor.
3. **Capability gating**: fp8 dispatch requires `cc >= (8,9)` — decline is loud (`UNSUPPORTED…` /
   skip-in-gates), the bf16/int8 paths are the documented A100 alternative. Split `AMAX_PTX` out of
   the fp8-train header (it is plain f32) so amax survives on A100.
4. **Dynamic SMEM launches**: `shared_mem_bytes` + `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)`;
   generators take an SMEM budget from `GpuTarget` instead of the 48 KiB literal; keep the static
   path as the ≤48 KiB fallback. New candidates: 3–5-stage f16/int8 pipelines, 256-wide tiles, larger
   flash K/V staging, conv shapes the 44 KB gate declined.

   > **This is developable, gate-able and perf-testable AT HOME for $0 — do not rent anything for
   > it.** The 4050 is Ada, with a **~100 KiB opt-in carveout** the codebase deliberately never uses:
   > `ptx_conv.rs:57` says *"no opt-in to the larger Ada banks"* and `ptx_wmma.rs:340` reasons about
   > *"the 100 KiB SM carveout"*. So the single biggest new capability in this plan — the one that
   > unlocks A100's 164 KB and H100's 228 KB — can be written and **measured** on the laptop, more
   > than doubling the SMEM budget the generators can currently express. It may even be a 4050 win in
   > its own right (the 3-stage variants the census found were rejected *because* of the 48 KiB
   > static cap). Landing dynamic SMEM before renting anything de-risks Phase 3's largest step and
   > costs nothing.
5. **Cache keys**: autotune key += device identity (name or cc+sm_count); cubin filename += arch.
   Wire a default autotune cache path so metered instances don't re-pay tuning per process.
6. **Dispatch de-literaling**: f16 regime thresholds become multiples of probed L2 (the probe is
   currently dead code); split-K triggers from `sm_count`; `gpu_accel.rs:118` divisibility derives
   from the dispatched kernel's tile constants; serving/KV budgets from `cuMemGetInfo`.
7. Regression discipline: on the **4050**, before/after this phase must be **behaviorally identical**
   (same dispatch census, same suite results, same-run perf A/B ties) — the retarget must not cost
   the existing target anything. That regression run is free, at home.

Exit: 4050 identical; the tree can *in principle* emit sm_80. (~25–35 commits.)

### Phase 3 — H100: the competitive datacenter target (~$35–65, partly Modal-free)

**Reordered per §3:** H100 is the target whose peers are strongest and therefore whose wins mean the
most. A100 is no longer the first competitive target — it drops to the 1–2 hour backward-emission
check in Phase 3b, because `sm_89` already gave us the Ampere-class programming model for free.

1. **Bring-up** (Modal H100, free credit): full suite with `WUKONG_GPU_REQUIRED=1`. Clear the flagged
   JIT unknowns first (§8): `.version 8.4` + `sm_89` forward-JIT acceptance, and the "legacy 8×`.b32`"
   f16 WMMA fragment spelling. Retag the fp8 family `sm_90`; re-run the fp8 E5M2 host-vs-device byte
   gate on Hopper silicon and add the missing E4M3 device gate (`ptx_fp8.rs:19`).
2. **Derive on paper, THEN re-tune** (§0's cost rule; this is the biggest-spend step, so it is where
   the discipline pays). First, **off-GPU and free**: from the target's opt-in SMEM, register file,
   warps/SM and SM count, compute which tile shapes and pipeline depths are even feasible, and rank
   them. Read the CUTLASS configs for the same architecture and shape class. Write down the expected
   winner and why. *Only then* rent, and run a short confirmation sweep against that prediction
   rather than a blind grid — int8 (the `int8_gemm_swz_tile_ptx` sweep hook already generates
   arbitrary tiles), f16 pipe/cliff (ignored benches exist), the flash A/B suite, conv split-K,
   gemv/norm underfill. Seed candidates from the Phase-1 SM-scaling findings + the new dynamic-SMEM
   variants. Persist the autotune cache as a committed artifact so no shape is ever tuned twice.
3. **Benchmark rounds** (root VM for clock-locking: Verda spot $1.63 / Lambda as control): the §6
   instrument, vs cuBLAS / cuBLASLt (epilogue-fused — the *honest* fusion peer, now buildable),
   cuDNN, **`torch.compile`** (Triton native on Linux), and real CUTLASS/FA2 peers.
4. **Publish Act 1 honestly**: these are *Ampere-class kernels forward-JITted onto Hopper*. Expect
   them to trail cuBLAS at large GEMM and say so plainly — that gap **is** the wgmma business case,
   and is the number Phase 4 is gated on. The families where wgmma matters less (serving, latency,
   bandwidth-bound, quantized decode) are where Act 1 can genuinely compete.
5. Record per §6; scope every claim to the device.

Exit: an H100 section in BENCHMARKS.md with controls passing, the wgmma gap quantified, and the
tune cache committed as an artifact.

### Phase 3b — A100: backward-emission validation only (~$2–4, 1–2 hours)

Not a competitive round. Its single job is to prove the Phase-2 header parameterization works toward
an **older** ISA — a direction `sm_89 → sm_90` never exercises, and the one way the retarget could be
silently wrong. Run the suite on Modal A100-40GB; expect the fp8 family to skip loudly by capability
and everything else to pass. If bandwidth-bound conclusions matter later, A100's ~2 TB/s HBM makes it
a useful second data point — but that is opportunistic, not required.

### Phase 4 — wgmma + TMA: the gated decision (+$40–80 if GO)

1. **The go/no-go gate.** Criteria to GO: (a) Phase 3 rounds published and healthy, (b) the Act-1
   H100 gap quantified so the wgmma win is *measurable in advance*, (c) explicit acceptance of the
   scope — a new generator family (`ptx_wgmma.rs`: warpgroup MMA, TMA descriptors + `mbarrier`
   pipelines, `.version 8.x` + `sm_90a`, likely `setmaxnreg`), the largest single work item in this
   document, comparable to all of Phase 2 by itself. **Use a Fable subagent for the design** (§0).
2. If **NO-GO**: H100 remains an honestly-scoped correctness + serving + bandwidth target, and the
   plan closes there. That is a legitimate ending, not a failure.
3. If **GO**: derive the warpgroup tile/TMA schedule on paper first (§0), build and gate on the 4050
   where possible (PTX text gates are device-free), then confirm on rented H100 hours.
4. (Stretch, device-independent, any time after Phase 2: GQA in paged KV, flash D=256, fused flash
   backward, megakernel multi-CTA cooperative — each is worth ~6× more on a 132-SM part than it was
   on 20 SMs, and none of it needs a rented GPU to *write*.)

### Phase 5 — The rented-GPU benchmark instrument (§6) — built during Phase 3, reused for H100.

### Phase 6 — Documentation re-scope (interleaved; final pass at the end)

1. Immediately (cheap, honest): device-scope banners on `BENCHMARKS.md` GPU section, `README.md:103`,
   `docs/roadmap.md`, `docs/metrics.md` — "measured on RTX 4050 Laptop (sm_89); datacenter retarget
   in progress, see GPU_RETARGET_PLAN.md".
2. De-duplicate the triplicated `docs/internals.md` **before** editing it.
3. As results land: per-device tables (§6 format); reframe "no-toolkit" as "toolkit-optional";
   rewrite the in-code %-of-cuBLAS dispatch rationales with per-arch tables; sweep the ~20 doc-claim
   families from §2.7. Per repo law, measurement docs change **only** with fresh same-run numbers.

---

## 6. Benchmark methodology

The laptop taught this project its instrument discipline (same-run adjacent A/B, best-of-N minima,
ratios never absolutes, a control column that measures the run's own floor). Rented GPUs have
*different* noise axes — multi-tenant hosts, unlockable clocks in containers, heterogeneous host
CPUs, MIG mislabeling — so the discipline ports, re-instrumented:

1. **Provenance header on every round** (currently only `device_name` is printed, `serving.rs:1496`):
   GPU name + CC + SM count (verified against spec — MIG/vGPU detection), driver, CUDA runtime,
   provider + SKU, clock-lock status, `nvidia-smi` clocks/temp/power before AND after. A round whose
   SM count doesn't match spec publishes nothing.
2. **The GPU twin control**: the byte-identical-PTX module loaded twice (two module handles, same
   source), timed as two columns — expected ratio 1.00, no content. Same role as `C(twin)`: the
   per-run noise floor. A cell publishes a size only if its range clears the floor; a failed control
   publishes nothing. Add a cuBLAS-called-twice control on peer rounds.
3. **Clock policy**: canonical rounds on VMs with root, `nvidia-smi -lgc` locked below throttle
   (verify `-lgc` actually works on first boot — flagged medium-confidence even on VMs). In
   containers (Modal/RunPod/Vast): no lock — warmup, back-to-back trials with NO sleeps, CUDA-event
   timing, trimmed-median over ≥100 iters, everything same-invocation. Container rounds are
   *iteration* data; VM rounds are *publication* data.
4. **Peers get stronger and that is the point**: cuBLASLt with fused epilogues (vs our fused GEMM),
   torch.compile (vs eager), real FA2/CUTLASS builds, Marlin-class int4 — the "no honest peer
   available" framings from the toolkit-free laptop expire, and every claim that survives is worth
   more.
5. **Recording**: new per-device sections in `BENCHMARKS.md` (4050 / A100 / H100 / stepping-stone
   appendix), each opening with the provenance block and the control readings; cross-device
   comparisons only as ratios-vs-that-device's-peers, never absolute-vs-absolute across devices.
   Tune caches and raw round logs committed under `bench/gpu/<device>/` as artifacts.
6. **Budget honesty**: every round's GPU-hours and $ cost recorded in the round log; suite/bench
   wall-times from Phase 1 drive session planning so no metered hour is spent discovering how long
   a gate takes.

---

## 7. Budget

Revised for the §3 ladder (H100 promoted, A100 demoted) and §0's cost rule. **All research, derivation,
code and compilation is $0** — only device gates and timed rounds appear here.

| Phase | Hardware | GPU-hrs | $ (on-demand) | Notes |
|---|---|---|---|---|
| 0 Portability / CI | none (local + free CI) | — | **$0** | |
| 1 `sm_89` bring-up + SM-scaling | Modal L4 $0.80, RunPod L40S $0.99 | 8–14 | $7–14 | free under Modal credit |
| 2 Keystone retarget | local 4050 only | — | **$0** | ~25–35 commits, no rent |
| 3 H100 bring-up + derive + retune + rounds | Modal H100 $3.95, Verda spot VM $1.63 | 16–26 | $35–65 | the competitive target |
| 3b A100 backward-target validation | Modal A100-40GB $2.10 | 1–2 | $2–4 | proves sm_80 emission only |
| 4 wgmma/TMA Act 2 (**gated — separate decision**) | H100 | +10–20 | +$40–80 | only if the Phase-4 gate says GO |
| 5–6 Regression rounds + docs | mixed | 6–10 | $10–25 | |
| **Total (excl. Act 2)** | | **~31–52 GPU-hrs** | **~$54–108** | **→ ~$0–50 net** across 2–3 Modal cycles |

Hard cap proposal: **$150** excluding a separately-approved Act 2. Two levers keep the real number at
the bottom of that range: Modal's **$30/month recurring credit** (≈7.6 free H100-hours *per cycle*, so
spreading Phase 3 across two months is close to free), and §0's derive-before-you-rent rule, which is
what turns a 40-point blind sweep into a 6-point confirmation. The Prime Intellect grant, if it lands,
covers everything.

---

## 8. Risks and open technical questions

| # | Risk | Mitigation |
|---|---|---|
| 1 | `.version 8.4` + `.target sm_89` forward-JIT acceptance on H100's deployed driver unverified | First Modal H100 hour smoke-tests it; fallback = retag those modules sm_90 immediately |
| 2 | The f16 "legacy 8×.b32" wmma fragment spelling on sm_80/sm_90 JIT ("legacy" per its own comment, `ptx_wmma.rs:34`) | Phase-3 bring-up test; PTX-ISA-defined per shape so expected fine — verify, don't assume |
| 3 | `cudarc 0.16` / `cuda-12060` vs r580/CUDA-13 hosts and Hopper API surface (cluster launch) | Phase 1 exercises cudarc on a cloud driver before any retarget work; bump cudarc if needed |
| 4 | Marketplace hardware fraud/variance (GRID-virtualized "A100"s, power-capped hosts) | Provenance gate (§6.1) refuses to publish; verified-datacenter hosts only; Lambda as known-clean control |
| 5 | Wins that will NOT transfer: WDDM-inflated graph multiples, eager-torch wins, 20-SM occupancy crossovers ("beats IMMA"), small-S flash wins | Named in advance here so shrinkage is reported as re-scoping, not regression; §2.7 |
| 6 | Corpus gates too slow for metered time (332 programs × 2 `-O` levels) | Phase 1 measures; if slow, a device-gate subset tag for cloud + full corpus nightly on the 4050 |
| 7 | Modal container model (host may change between invocations) corrupting A/B | Rule: every comparison inside ONE invocation; Modal is bring-up/iteration only, never canonical rounds |
| 8 | H100 disappointment risk: without wgmma the GEMM gap vs cuBLAS will be large | Scoped in advance (Phase 4 Act 1 framing); the gap number itself is the wgmma business case |
| 9 | Megakernel module-cache `Box::leak` growth once headers are per-device (census "missing" item) | Address in the Phase-2 header design (keyed cache, not leak-per-variant) |
| 10 | Spot reclamation (Verda −50%) mid-round | Spot for sweeps only; canonical rounds on-demand/per-minute VMs |

---

## 9. What is explicitly OUT of scope

- **CPU benchmark porting to Linux** (the Win64-only 256-bit vectorizer, thread pinning, MKL/vcvars
  peer discovery): the CPU story stays laptop-scoped; cloud boxes are GPU instruments. A SysV port of
  `avx2.rs` is a separate, worthwhile campaign — not this one.
- **Multi-GPU / NCCL / real tensor parallelism**: single-GPU targets only; ordinal selection lands
  (Phase 2.6) but TP stays simulation.
- **Blackwell (B200/GB200), MI300X, Grace-ARM hosts**: noted for the future; nothing here precludes
  them once `GpuTarget` exists.
- **Language-surface changes**: this is a backend/serving/benchmark campaign; the `.wk` language is
  untouched except where datacenter workloads need runtime features already listed (GQA geometry).

## 10. Success criteria

1. **Phase 1**: the full device suite passes with `WUKONG_GPU_REQUIRED=1` on a Linux sm_89 cloud box.
2. **Phase 2**: zero hardcoded arch facts (grep-clean for `sm_89` outside per-target tables/tests);
   4050 behaviorally identical before/after.
3. **Phase 3**: H100 suite green; a published H100 section whose controls pass, against
   cuBLAS/cuBLASLt (epilogue-fused), cuDNN, real FA2/CUTLASS and **`torch.compile`** — never eager.
   Act-1 numbers carry the *Ampere-class-kernels-on-Hopper* scope explicitly, and the wgmma gap is
   quantified rather than hand-waved. *Aspirational* (not promised): parity-class f16 GEMM at the
   L2-resident shapes, int8 within striking distance of cuBLASLt IMMA. **The success bar is honest
   controls-passing numbers, whatever they are** — a truthful loss closes the criterion; a flattering
   win does not.
4. **Phase 3b**: A100 suite green with fp8 skipping loudly by capability — proving the retarget emits
   correctly toward an older ISA. **Phase 4**: a written wgmma go/no-go memo backed by the measured
   Act-1 gap; if GO, a wgmma path that beats Act-1 on the same rounds.
5. **Phase 6**: no published claim anywhere in the repo that silently generalizes a 4050 number to
   other hardware.

## 11. Decisions needed from you

Status as of 2026-08-06, after the review pass:

| # | Item | Status |
|---|---|---|
| 1 | **Modal account connected** | ✅ **DONE** — workspace `aidanht`, token verified against the API |
| 2 | **Workspace budget cap set** at [modal.com/settings/usage](https://modal.com/settings/usage) | ⛔ **OPEN — the only blocker for Phase 1.** $30 (or $25). Also check whether a payment method is on file; if none is, leave it that way |
| 3 | **Spending approval**: ~$54–108, hard cap $150 (excl. a separately-approved wgmma Act 2) | ⛔ open |
| 4 | Second provider for cheap iteration (RunPod or Vast, ~$10 load) | ⏸ not needed until Phase 1 step 3 |
| 5 | Root-VM provider for clock-locked publication rounds (Verda / Hyperstack / Lambda) | ⏸ not needed until Phase 3 step 3 |
| 6 | **Prime Intellect grant pitch** — want me to draft the email? Free, real upside, individuals explicitly eligible | ⏸ optional, parallel |
| 7 | **wgmma Act 2** | deferred by design behind the Phase-4 gate |
| 8 | **Default in-repo target stays the 4050** (`GpuTarget` makes it a runtime fact anyway) | recommend yes — confirm |

**Nothing blocks Phase 0 or the free half of Phase 2.** Both are $0, local, and can start immediately;
only item 2 gates renting silicon.

---

*Census provenance: 5 code auditors + 1 adversarial verifier (21/22 spot-checks confirmed; the one
imprecision — serving KV budgets are bench-code, not production constants — is corrected above).
Cloud provenance: 4 researchers + 1 verifier who re-fetched Modal, RunPod, Verda, Hyperstack, Vast
and Colab the same day; corrections applied above. Both raw digests preserved in the session
scratchpad (`census_digest.md`, `cloud_digest.md`).*
