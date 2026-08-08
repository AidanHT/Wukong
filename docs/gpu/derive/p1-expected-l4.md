# P1 — predict-before-measure: what the device suite should do on a Modal L4

**Status: PREDICTION. Nothing here was measured on an L4.** Produced 2026-08-07 against the tree at
`d10d868` (branch `retarget/p1d-expectations`), entirely off-device, per `GPU_RETARGET_PLAN.md` §0
*"Write down the number you expect and why, then measure."* This document is the **acceptance
checklist** for the Phase-1 metered run: read §5 first when the run comes back, and treat every row
that disagrees as a finding to explain, not a number to accept.

Provenance convention, inherited from the Wave-0 dossiers (`docs/gpu/derive/README.md`):
**FACT** carries a `file:line` in this tree or a fetched document; **DERIVED** is arithmetic over
facts; **PREDICTION** is falsifiable and is an *input to the experiment*, never a result.

---

## 0. What is being run, and against what baseline

**FACT — the metered commands** (`tools/cloud/modal_app.py:329-357`, `::test`):

```
WUKONG_GPU_REQUIRED=1  cargo test -p wukong_codegen_gpu --features gpu     # debug profile
WUKONG_GPU_REQUIRED=1  cargo test -p wukong_driver      --features gpu     # debug profile
```

`WUKONG_PEER_REQUIRED` is deliberately **unset** on the first pass (`modal_app.py:330,340`), so a
peer that will not load is *reported* rather than failing the suite (`gpu.rs:7466` `peer_gate`).

**FACT — the local baseline** (RTX 4050 Laptop, `sm_89`, 20 SMs, L2 24 MiB = 25 165 824 B, opt-in
SMEM 101 376 B, driver 13010): `wukong_codegen_gpu` = **253 passed / 0 failed / 87 ignored**.

**FACT — that baseline is reproducible from source, not just from a log.** The crate has 340
`#[test]` attributes and 87 real `#[ignore]`s (89 textual hits minus two that are prose:
`gpu.rs:10901` and `train_resident.rs:924`), so 340 − 87 = 253. There is **no** `#[cfg(windows)]`,
`#[cfg(unix)]` or `#[cfg(target_os = …)]` anywhere in `crates/wukong_codegen_gpu/src` or
`crates/wukong_driver/src` (grep-verified; `baselines.rs:83` deliberately uses a *runtime* `cfg!`
so both arms type-check on both hosts). **Therefore the test COUNTS are OS-independent and the L4
must report the same 340/87/253 split.** Any change in the *counts* is a build-configuration bug,
not a device finding.

**DERIVED — `wukong_driver` under `--features gpu`:** 24 `#[test]`s, 0 ignored
(`lib.rs` 20, `loader.rs` 3, `gpu_accel.rs` 1). Five of them are the device e2e gates in
`mod gpu_e2e_tests` (`lib.rs:1859`, `#[cfg(all(test, feature = "gpu"))]`); the other 19 are CPU
autodiff/train/verify tests already proven on Linux by CI (`.github/workflows/ci.yml:44`,
`ubuntu-latest`).

**DERIVED — how much of the suite actually touches silicon.** Scanning every `#[test]` body for
`with_gpu(` / `with_fp8(` / `with_cap(` / `gpu::gpu()` / `device_lost()`:

| | tests | ignored | running | **running & device-executing** |
|---|---:|---:|---:|---:|
| `wukong_codegen_gpu` | 340 | 87 | 253 | **154** |
| `wukong_driver` (`--features gpu`) | 24 | 0 | 24 | **5** |

154 is the honest number for what the plan calls "~130 device gates" (`GPU_RETARGET_PLAN.md:313`);
the remaining 99 running tests in the crate are PTX-text laws, header-floor gates, pure host
allocators and cache-key algebra that pass identically on a machine with no GPU at all.

---

## 1. Device-test inventory by family, and the predicted L4 outcome

The L4 is `sm_89` Ada — **the same compute capability as the dev 4050**. That single fact carries
most of this table: `ptx_target::HDR_SM80` (`ptx_target.rs:25`) and `HDR_SM89_V84`
(`ptx_target.rs:30`) are family *floors*, and 8.9 ≥ 8.9 ≥ 8.0, so **every module in the crate loads**.
The fp8 gate is `FP8_MIN_CC = (8, 9)` (`gpu.rs:150`) and `GpuTarget::supports` is a tuple compare,
so cc 8.9 clears it exactly.

| # | Family | Running device gates (file:line of the first) | Predicted L4 outcome | Mechanism |
|---|---|---|---|---|
| 1 | **Device identity + caches** | 5 in `gpu.rs` (7145, 7278, 7400, 9377, 9446) + `cubin.rs:306,345` + `baselines.rs:2921` | **PASS (8)** | every assert is an architecture-*independent* range (`gpu.rs:7169-7212`) or a derivation, never a 4050 value — see §2 |
| 2 | **f16 / bf16 WMMA + `mma.sync` GEMM** | 26 in `gpu.rs` (8015 … 8934, 9204, 9259, 15977, 16076, 16199, 16236, 21562) | **PASS (26)** | `mma.sync m16n8k16`, `ldmatrix`, `cp.async` are sm_80-legal; identical ISA, identical SMEM budget |
| 3 | **Flash attention** | 17 in `gpu.rs` (9645 … 10107, 22129 … 24147) | **PASS (17)** | same; `FLASH_WARPS` and tile choices are compile-time constants, not device queries |
| 4 | **Conv2d / Winograd** | 11 in `gpu.rs` (10824 … 11943) | **PASS (11)**, one *coverage* change | `conv2d_wmma_padded_auto` (`gpu.rs:11440`) feeds the real `sm_count()` into `conv_splitk_factor_affine`; see §3.4 |
| 5 | **int8 / quant (incl. dynamic-SMEM stage rows)** | 12 in `gpu.rs` (19124 … 21405) | **PASS (12)** | budget-derived throughout: `smem_budget()` (`gpu.rs:443`) = `smem_per_block_optin`, predicted identical at 101 376 B |
| 6 | **int4 / W4A16** | 2 in `gpu.rs` (7484, 7570) | **PASS (2)** | `lop3` + f16x2 unpack, sm_80-legal |
| 7 | **fp8 (E4M3 / E5M2)** | 15 in `gpu.rs` (6250, 6283, 6303, 6436, 9126, 14283 … 14690, 14998, 21088) + `ptx_fp8_train.rs:601` | **PASS (16)**, but see row 8 | `with_fp8` → `require_cap("fp8", (8,9))`; cc 8.9 ≥ 8.9 so the gate must **not** decline, and `with_cap` (`gpu.rs:6374-6395`) *escalates a wrong skip to a failure* |
| 8 | **fp8 vs the cuBLASLt peer** | `gpu.rs:14998` `cublaslt_fp8_matches_reference_within_tol` (counted in row 7) | **FAIL-CANDIDATE (medium)** | it is **not** `#[ignore]`d, and the `nvidia/cuda:12.9.2-cudnn-devel` image resolves `libcublasLt.so.12` (`baselines.rs:97`), so its cuBLASLt fp8 path executes where the 4050's default `cargo test` skips it (`gpu.rs:15004-15011`). First real execution of that transpose mapping in the default gate |
| 9 | **Elementwise / reduce / cast / norms** | 8 in `gpu.rs` (7776 … 7931, 9553) + `ptx_norm.rs:263` | **PASS (9)** | grid-stride and warp-butterfly kernels; `ptx_optim.rs:219` already scales grids by `32 × sm_count()` |
| 10 | **Optimizer (AdamW / SGD)** | 4 in `ptx_optim.rs` (431, 504, 537, 589) | **PASS (4)** | same grid-stride form |
| 11 | **Autodiff backward kernels** | 7 in `ptx_autodiff_bwd.rs` (1427 … 1727) | **PASS (7)** | sm_80-floored (`ptx_autodiff_bwd.rs:1363` pins it textually) |
| 12 | **Paged attention / paged KV** | 4 in `paged_attention.rs` (1365, 1400, 1531, 1566) | **PASS (4)** | KV geometry is explicit in the test, never probed |
| 13 | **Serving stack (scheduler, batching, TP sim)** | 15 in `serving.rs` (1126 … 2843) | **PASS (15)** | the 4 GiB / "6 GB part" budgets are *inputs to pure functions* in un-gated host tests (`paged_kv.rs:676,700`), not device queries |
| 14 | **Resident training** | 5 in `train_resident.rs` (468 … 617) | **PASS (5)** | fixed shapes, tolerance gates |
| 15 | **Runtime integration: pool, CUDA graphs, multi-stream** | 4 in `gpu.rs` (18232, 18355, 18687, 19006) + 2 in `serving.rs` (1800, 2146) | **PASS (6)** — the top driver-API risk | see §4 |
| 16 | **Autotune (device sweeps)** | 3 in `autotune.rs` (1275, 1379, 1452) | **PASS (3)** | see §3.3 for cost |
| 17 | **e2e model / layer / reproducibility** | 7 in `gpu.rs` (13039 … 13271, 15605) | **PASS (7)** | fixed grids, no atomics (`gpu.rs:13260`) |
| 18 | **Roofline sanity** | 1: `gpu.rs:17572` | **PASS**, the suite's *only* timing-derived assert | asserts `1e11 < r < 5e14` FLOP/s (`gpu.rs:17576`). L4 fp16-TC peak ≈ 1.2e14, and Linux launch overhead is *lower* than Windows/WDDM, so the margin widens |
| 19 | **gpu-native corpus → interp oracle** | `lower.rs:4263` | **PASS**, `covered == 217` | see §5 |
| 20 | **Megakernel corpus → interp oracle** | `megakernel.rs:237` | **PASS**, `ran == 87` | see §5 |
| 21 | **`wukong_driver` `--backend=gpu` e2e** | `lib.rs:1943, 1999, 2063, 2151, 2236` | **PASS (5)** | `gpu_accel.rs:118`'s M,N %64 / K %16 divisibility is a *kernel-tile* fact, not a device fact |
| — | **Non-device running tests** (PTX laws, header floors, cache algebra, host allocators, fusion analysis) | 99 in `wukong_codegen_gpu` + 19 in `wukong_driver` | **PASS (118)** | no device, no OS conditionals; already green on `ubuntu-latest` under `cargo check --features gpu --all-targets` (`ci.yml:86`) |

**Predicted `[skip:capability]` count: exactly the same set of lines as the 4050 emits, byte for
byte.** Every capability skip in the running set is a comparison against `g.smem_budget()`
(`gpu.rs:14580`, `16091`, `20613`, `20717`, `7319`) and both parts are cc 8.9 with a 99 KiB opt-in
carveout. This is a sharper check than a count: *a difference in that set is a direct measurement
that the L4's opt-in SMEM is not 101 376 B*, and every stage-depth conclusion below has to be
re-derived from the real number.

---

## 2. Device-identity coupling hunt — the highest-value check

**Method.** Every non-`#[ignore]`d `#[test]` body in both crates was scanned for the couplings that
would only hold on the dev card: `sm_count`, `l2_bytes`/`l2_cache_size`, `smem_budget`, `total_mem`,
`driver_version`, `device_tag`, and the literals `20`, `101376`, `25165824`, `4050`. Twenty-three
tests matched; each was read. **Result: zero genuine device-identity couplings in the running set.**
That is a load-bearing negative, so here is the whole ledger with the reason each hit is inert.

| Hit | Verdict | Why |
|---|---|---|
| `gpu.rs:7400` `f16_regime_thresholds_reproduce_the_4050_literals` — hardcodes `L2_RTX_4050 = 25_165_824` | **PROVEN NON-ISSUE** | the literal is an *argument to a pure function* (`gpu.rs:7404`), and the device half is guarded by `if l2 == L2_RTX_4050` (`gpu.rs:7448`). On a 48 MiB L2 that branch is simply not taken; the only device assert left is `dlo < dhi` (`gpu.rs:7455`). Deliberate — the doc comment at `gpu.rs:7396` says so |
| `gpu.rs:7145` `gpu_target_is_probed_and_sane` — reads every probed field | **PROVEN NON-ISSUE** | asserts *ranges only*: `cc_major ∈ 6..=12`, `sm_count ∈ 1..=1024`, `smem_* ≥ 48 KiB`, `l2_bytes > 0`, `total_mem > 1 GiB`, `driver_version > 0` (`gpu.rs:7171-7212`). The comment at `gpu.rs:7140` states the intent: "deliberately *not* this 4050's values" |
| `autotune.rs:944, 992, 1009, 1167, 1198` — the string `"sm_89x20"` | **PROVEN NON-ISSUE** | an opaque cache-key *token* in five device-free algebra tests. `autotune.rs:1009` even pairs it with `"sm_80x108"` to prove an A100 lookup misses |
| `autotune.rs:1275, 1452` — `g.device_tag()` | **PROVEN NON-ISSUE** | the tag is *read from the device* and used for a same-process round-trip (`autotune.rs:1328-1337`); the only literal is the negative check `get_int8("sm_80x108", …).is_none()` (`autotune.rs:1340`) |
| `ptx_conv.rs:1865` `splitk_factor_picks_reasonable` — `let sm = 20;` | **PROVEN NON-ISSUE** | `sm` is an *input* to `conv_splitk_factor`, not a query. Pure function, device-free test |
| `ptx_fp8.rs:1797` `fp8_deep_over_budget_panics_at_generation` — mentions 101 376 | **PROVEN NON-ISSUE** | pure PTX-generation test; the budget is passed in |
| `gpu.rs:14577`, `16199` — "declines over budget" | **PROVEN NON-ISSUE**, and *correctly* written | both fabricate a 32-stage config and assert `over.smem_bytes() > budget` where `budget = g.smem_budget()` (`gpu.rs:14580`, `16202`). The comments say "so the check fires on every card, however large its carveout (H100's opt-in is 227 KiB)". 512–640 KiB exceeds every part |
| `gpu.rs:14460`, `16076`, `20592`, `20709`, `7278` — SMEM-budget gates | **PROVEN NON-ISSUE** | all compare generated `smem_bytes()` against the probe and *skip loudly* above it; the launch counts are floors (`ran >= 12`, `ran >= 10`, `ran >= 8`), which the same 101 376 B budget clears identically |
| `gpu.rs:11440` `conv2d_wmma_padded_auto` — `g.sm_count()` | **PROVEN NON-ISSUE for pass/fail**, coverage changes | `sk` is only *printed* (`gpu.rs:11474`); nothing asserts it. See §3.4 |
| `cubin.rs:218-278` — `drv12090` and `sm_80/89/90/120` literals | **PROVEN NON-ISSUE** | pure key algebra through `cache_path_for` (`cubin.rs:162`), which takes both as parameters |
| `cubin.rs:287` `cache_path_uses_the_probed_device_arch` | **PROVEN NON-ISSUE** | accepts `UNKNOWN_ARCH` *or* any `sm_<digits>` (`cubin.rs:290`) |
| `baselines.rs:2921` `peers_compile_for_the_probed_device` | **PROVEN NON-ISSUE** | asserts the NVRTC flag *equals the probe* (`compute_89` on both parts), and explicitly forbids a static literal (`baselines.rs:2936`) |
| `ptx_target.rs:84` `only_the_ada_floor_declares_the_r550_driver_version` | **PROVEN NON-ISSUE** | pure string law over two constants |
| `paged_kv.rs:676, 700` — `4 GiB` budget, "the 6 GB part" | **PROVEN NON-ISSUE** | `budget` is a local passed to `max_ctx_within_budget`; the census's classification (`GPU_RETARGET_PLAN.md:303`) is confirmed |

Also checked and **absent**: any `assert*` on `sm_count == 20`, on a `"sm_89x20"` device tag produced
by `device_tag()`, on 24 MiB / 25 165 824 as a probed value, on 101 376 as a probed value, on
6 GB of VRAM, or on a specific `driver_version`. The only surviving `.unwrap_or(20)` was removed —
`sm_count()` now reads the probed descriptor and an unqueryable count fails `Gpu::new`
(`gpu.rs:560-565`).

**One structural hole found, and it is the mirror of what this section looks for.** Six device tests
skip *without* the `WUKONG_GPU_REQUIRED` escalation the rest of the crate uses
(`diff.rs:23` `skip_or_fail`):

- `lower.rs:4264-4267` — `run_corpus_matches_interp_oracle`, the gpu-native corpus gate;
- `wukong_driver/src/lib.rs:1948, 2004, 2068, 2157, 2242` — **all five** driver e2e gates.

On a healthy L4 these run normally, so this is not a FAIL-candidate. It matters because it means the
*driver* leg can report `24 passed` having executed **zero** device instructions if `Gpu::new` fails
in that process — the exact failure mode `WUKONG_GPU_REQUIRED` exists to prevent. **Acceptance rule
for the driver leg: it is only believed if `gpu --backend linear …: N GPU call(s)` appears in its
output** (`lib.rs:1985`) — which requires `--nocapture` (see §6.1).

---

## 3. What legitimately changes on the L4, and what it buys for free

### 3.1 The `GpuTarget` print — the single highest-value free datum

`gpu_target_is_probed_and_sane` (`gpu.rs:7145`) `println!`s the whole descriptor (`gpu.rs:7148-7167`).
Predicted values, all falsifiable:

| Field | 4050 (FACT) | **L4 (PREDICTION)** | Consequence if wrong |
|---|---|---|---|
| `name` | `NVIDIA GeForce RTX 4050 Laptop GPU` | `NVIDIA L4` | none (only non-empty is asserted) |
| `cc` | 8.9 | **8.9** | if not 8.9, the whole fp8 family becomes a capability skip and rows 7–8 of §1 change verdict |
| `sm_count` | 20 | **58** | provenance: `modal_app.py:220` `_EXPECTED["L4"] = ("sm_89", 58)`. Anything else = MIG/vGPU → publish nothing |
| `smem_per_block_optin` | 101 376 (99.0 KiB) | **101 376** | *the* load-bearing prediction: it fixes the entire `[skip:capability]` set (§1) |
| `smem_per_sm` | 102 400 (100.0 KiB) | **102 400** | occupancy prints (§3.5) |
| `l2_bytes` | 25 165 824 (24.00 MiB) | **50 331 648 (48.00 MiB)** | changes the f16 regime bands, §3.2 |
| `total_mem` | ~6 GiB | **~22–23 GiB** | only `> 1 GiB` is asserted |
| `driver_version` | 13010 | **13000** (r580 ⇒ CUDA 13.0 API) | only `> 0` is asserted; it becomes the cubin key prefix `drv13000-` |
| `device_tag()` | `sm_89x20` | **`sm_89x58`** | `gpu.rs:378` `format!("{}x{}", sm_arch(), sm_count)` |
| `compute_arch()` / `sm_arch()` | `compute_89` / `sm_89` | **unchanged** | `ptx_target.rs:51,56` from the probed cc |

### 3.2 f16 regime thresholds on a 48 MiB L2

**DERIVED** from `f16_regime_thresholds` (`gpu.rs:1039`, `l2·2/3` and `l2·2`, evaluation order pinned
at `gpu.rs:7418`):

```
4050   L2 25 165 824  ->  bands [16 777 216, 50 331 648)  = [16, 48) MiB
L4     L2 50 331 648  ->  bands [33 554 432, 100 663 296) = [32, 96) MiB
```

**Does any test assert a regime CHOICE rather than a correct result? No.** Grep of `regime` across
`gpu.rs` returns dispatch-rationale comments plus exactly two tests: `f16_regime_thresholds_…`
(guarded, §2) and `fp8_pipe_regime_matches_reference` (`gpu.rs:14386`), which asserts tolerance over
both `bm` arms rather than which arm the dispatcher picks. **Kernel coverage does not shrink either**:
`gemm_cliff_matches_reference` (`gpu.rs:15977`) loads every `CLIFF_VARIANTS` entry *by name* and
`assert`s each one was launched, deliberately bypassing the dispatcher (`gpu.rs:15982-15987`). So the
wider bands are pure new information (printed at `gpu.rs:7440`) and cost nothing.

### 3.3 Cold caches — cheaper than they look, and one of them is not exercised at all

- **Autotune cache: NOT persisted by the suite.** Every device autotune test builds
  `AutotuneCache::new()` in memory (`autotune.rs:1286`, `1431`, `1464`); there is no default cache
  path and no `AutotuneCache::load` call outside `cache_save_load_roundtrip_through_a_file`'s own
  temp file (`autotune.rs:1220-1222`). Plan §5 Phase-2 step 5's "wire a default autotune cache path"
  is **not landed**. So "cold autotune" is a non-event for Phase 1.
- **Cold tuning cost is seconds, not minutes.** DERIVED: 12 int8 candidates (6 fixed at
  `autotune.rs:104-185`, + `w64_s3/s4/s5` at `:193`, + 3 split-K at `:213`), each timed best-of-6 × 50 iters
  (`autotune.rs:393`), over 4 tuned shapes (three at `autotune.rs:1287-1291`, one at
  `autotune.rs:1390`) plus 2 W4A16 shapes (`autotune.rs:1465`) ⇒ ~2 × 10⁴ launches of
  microsecond-scale kernels at ≤256³.
- **Cubin cache: cold, and every corpus module is a single-use entry.** On a miss,
  `load_module_cached` (`gpu.rs:523-542`) does a `cuLink` compile *and* a `write_atomic`
  (`cubin.rs:178`, temp file + rename) before loading. The two corpus gates generate a *distinct*
  PTX text per program per `-O` level, so ~900 cubins are written and **none is ever read back**.
  See §6.3 — this is the one place the Modal Volume can cost real money for no benefit.

### 3.4 58 SMs: what it actually reaches in this suite

The suite consumes `sm_count()` in exactly three production paths, and only one is inside a running
gate: `conv_splitk_factor_affine` via `conv2d_wmma_padded_auto` (`gpu.rs:11442,11459`). The other
two are `ptx_optim.rs:219`'s grid-stride sizing (correct by construction at any SM count) and
`gpu.rs:3429/3809/12531` (dispatch + `#[ignore]`d benches).

**PREDICTION — a control silently stops being a control.** `gpu.rs:11451` marks
`(64, 56, 56, 64, 3, 3, 1, 1)` as "base grid ~49 CTAs → `sk == 1` control". Split-K fires when the
base grid starves the SMs; 49 CTAs starve 58 SMs where they did not starve 20. So on the L4 that case
most likely reports `sk > 1` and the suite loses its `sk == 1` arm — **without any assertion firing,
because `sk` is only printed.** Not a failure; do read the printed `sk` values and, if every case
splits, note that the non-split conv path went untested on this device.

Everything else that "scales past 20 SMs" lives in `#[ignore]`d benches and is Phase-1 step 3 work on
the L40S, not part of this run.

### 3.5 Dynamic SMEM on Ada — the deep rings must all still fit

**DERIVED.** `INT8_STAGE_VARIANTS` (`ptx_int8.rs:1378-1411`) is 128×128 bk64 at s2/s3/s4/s5 =
32 / 48 / 64 / 80 KiB; the s4/s5 rows exist only through the `.extern` window. The standalone probe
`dynamic_smem_window_exceeds_the_static_48_kib_ceiling` (`gpu.rs:7278`) asks for 64 KiB. The big-tile
gate (`gpu.rs:20709`) asks for `3·(256+128)·64` = 72 KiB. All four are ≤ 101 376 B, so on the
predicted budget **every deep-stage gate runs and none takes the capability-skip branch** — identical
to the 4050. The 32-stage over-budget probes (640 KiB / 512 KiB) still exceed it, so both "declines
loudly" gates still exercise the decline arm.

**Free datum worth harvesting:** `occupancy_max_active_blocks_per_multiprocessor` is printed by three
running gates (`gpu.rs:16111`, `20634`, `20736`). Occupancy is a **per-SM** function of SMEM, registers
and warps, and the L4 and the 4050 are the same cc with the same per-SM limits — so **the printed
`occupancy=N CTA/SM` values should be identical on both parts.** A difference means the r580 driver's
JIT allocated registers differently from r5xx-on-Windows; that is interesting Phase-3 input, not a bug.

---

## 4. cudarc on an r580 / CUDA-13 host (plan §8 risk 3)

**FACT — the binding.** `cudarc = "0.16"`, `default-features = false`, features
`driver, dynamic-loading, cuda-12060, f16, std, cublas, cublaslt, nvrtc, cudnn`
(`crates/wukong_codegen_gpu/Cargo.toml:16-43`). `dynamic-loading` means `libcuda.so.1` is `dlopen`ed
at run time (`baselines.rs:94`), so the *build* needs no toolkit — already proven on `ubuntu-latest`
by `ci.yml:86`.

**FACT — the whole raw driver-API surface the suite touches** (`sys::cu*` grep over the crate, plus
the safe cudarc calls the gates reach):

| Surface | Where | CUDA version introduced | r580 expectation |
|---|---|---|---|
| `cuInit`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain`, `cuStreamCreate`, `cuMemAlloc/Free`, `cuMemcpyHtoD/DtoH`, `cuLaunchKernel` | cudarc `CudaContext`/`CudaStream` | ≤ 4.0 | fine |
| `cuDeviceGetAttribute` ×6, `cuDeviceGetName`, `cuDeviceTotalMem_v2`, `cuDriverGetVersion` | `gpu.rs:80,93,123,127` | ≤ 5.0 | fine; `cuDriverGetVersion` returns **13000** |
| `cuModuleLoadDataEx` (PTX JIT) | cudarc `load_module` | ≤ 4.0 | **the load-bearing one.** PTX `.version` is a *driver floor*: 7.8 needs r520+, 8.4 needs r550+ (`ptx_target.rs:38-44`). r580 clears both |
| `cuLinkCreate_v2` / `cuLinkAddData_v2` / `cuLinkComplete` / `cuLinkDestroy` | `cubin.rs:54-80` | 5.5 | fine; this is the in-driver `ptxas` |
| `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` | `gpu.rs:484` | 9.0 | fine |
| `cuOccupancyMaxActiveBlocksPerMultiprocessor` | `gpu.rs:16111`, `20634`, `20736` | 6.5 | fine |
| `cuStreamBeginCapture_v2(THREAD_LOCAL)` / `cuStreamEndCapture` / `cuGraphInstantiateWithFlags(flags=0)` / `cuGraphLaunch` / `cuGraphExecDestroy` / `cuGraphDestroy` | `graph.rs:137,144,153,171,187,190` | 10.0 / **11.4** for `…WithFlags` | fine. Note the code deliberately bypasses cudarc's safe `CudaGraph` because it forces `AUTO_FREE_ON_LAUNCH` (`graph.rs:25-30`) |
| `cuDevicePrimaryCtxReset_v2` | `gpu.rs:679` | 4.0 | fine; only reached *after* a fault |
| **Absent:** `cuLaunchCooperativeKernel`, cluster launch, `cuMemGetInfo` | — | — | the megakernel is a plain `b.launch` on a `(1,1,1)` grid (`megakernel.rs:39,100`), so no cooperative-launch surface is exercised at all |

**PREDICTION — compatible, with two named caveats.**
1. The NVIDIA driver is forward-compatible with older toolkits: an r580 `libcuda.so.1` retains every
   versioned entry point above, so `cuda-12060` bindings against a CUDA-13 driver resolve. The image
   itself is pinned to CUDA **12.9.2** precisely because cudarc 0.16.6 has no CUDA-13 bindings
   (`modal_app.py:64-69`), so nothing in the process asks for a 13-only symbol.
2. CUDA 13 dropped *architecture* support below Volta; `sm_80`/`sm_89` are unaffected, and every
   module in the tree floors at one of those two (`ptx_target.rs:25,30`).

**The Linux behaviour genuinely worth measuring here is the sticky-fault policy.** `reset_gpu`
(`gpu.rs:651-680`) degrades to marking the device *lost* because on Windows/WDDM the re-retain still
returns `CUDA_ERROR_ILLEGAL_ADDRESS` (`gpu.rs:659-664`). If no gate faults on the L4 this stays
unmeasured — which is the *good* outcome, and the honest thing to write down.

---

## 5. Diff me against the real run

### 5.1 The counts

| Metric | 4050 baseline | **L4 prediction** | Tolerance | If it differs, inspect first |
|---|---|---|---|---|
| `wukong_codegen_gpu` total | 340 | **340** | **0** | a different total means a build-config difference, not a device one — no `#[cfg(os)]` exists in the crate |
| `wukong_codegen_gpu` ignored | 87 | **87** | **0** | as above |
| `wukong_codegen_gpu` passed | 253 | **253** | **0** | §5.2 |
| `wukong_codegen_gpu` failed | 0 | **0** | **0** | §5.2, in order |
| `wukong_driver` total / passed / failed | 24 / 24 / 0 | **24 / 24 / 0** | **0** | driver leg is only believed with `--nocapture` (§2 hole) |
| `[skip] …: GPU unavailable` lines | 0 | **0** | **0** | under `WUKONG_GPU_REQUIRED=1` these are assertion failures by construction (`gpu.rs:6353`, `diff.rs:24`) — *except* in the six un-escalated sites listed in §2, which is why they are called out there |
| `[skip:capability]` lines | *(record from the 4050)* | **identical set** | **0 lines** | a delta ⇒ `smem_per_block_optin ≠ 101 376` ⇒ re-derive §3.5 |
| `[skip]` peer lines | ≥1 (`cublaslt_fp8_gate`) | **0** | — | if it still skips, the image did not resolve `libcublasLt.so.12` |
| `run_corpus_matches_interp_oracle` `covered` | 217 / 357 | **exactly 217** | **0** | floor at `lower.rs:4260`; coverage is decided by `lower::UNSUPPORTED`, a pure MIR property |
| `mega_corpus_matches_oracle` `ran` / `eligible` | 87 / 103 | **exactly 87 / 103** | **0** | floor at `megakernel.rs:229`; eligibility is `fusion::analyze`, device-free |
| driver FAULTS in either corpus gate | 0 | **0** | **0** | any fault is the run's headline finding |
| **suite wall time**, `wukong_codegen_gpu` | *unrecorded* | **10–25 min** (debug) | ±2× | **the number Phase 1 exists to produce** (plan §8 risk 6; `tools/cloud/README.md:117`) |
| **suite wall time**, `wukong_driver` | *unrecorded* | **1–4 min** (debug) | ±2× | as above |

Wall-time basis (**PREDICTION, weakest row in this document**): the two corpus gates dominate.
Together they compile 357 `.wk` programs × 2 `-O` levels **twice** (`lower.rs:4304`,
`megakernel.rs:278`) through lex→parse→sema→mir_build→opt in a **debug** build, run each through the
debug tree-walking interpreter as the oracle, and JIT ~900 one-shot PTX modules. Everything else —
154 device gates at ≤256³ shapes plus ~2 × 10⁴ autotune launches — is seconds of device time. If the
measured total lands outside 5–50 min, the model is wrong and the cause is worth a paragraph.

### 5.2 If a failure count is non-zero, inspect in this order

1. **`cublaslt_fp8_matches_reference_within_tol`** (`gpu.rs:14998`) — the one gate whose peer path
   executes for the first time on the L4. A tolerance failure here is a *peer-wiring* finding
   (column-major `Cᵀ = B̌ᵀ·Ǎ` mapping, `gpu.rs:14994`), **not** a Wukong-kernel finding. Do not let it
   contaminate the Phase-1 verdict.
2. **Either corpus gate's FAULTS list** (`lower.rs:4396`, `megakernel.rs:325`) — read this *before*
   anything else if more than one test failed. A single kernel fault poisons the shared primary
   context, `device_lost()` flips, and `mega_corpus_matches_oracle` then **panics outright**
   (`megakernel.rs:238-245`) with a message that names none of the root cause. One fault ⇒ many
   red tests ⇒ exactly one real bug.
3. **`gpu_target_is_probed_and_sane`** (`gpu.rs:7145`) — if this fails, every other verdict in this
   document is void; the probe is the single source of arch-dependent behaviour.
4. **`dynamic_smem_window_exceeds_the_static_48_kib_ceiling`** (`gpu.rs:7278`) — the `cuFuncSetAttribute`
   plumbing. It asserts *both* directions, so a failure on the negative half ("the un-opted launch was
   accepted") means the r580 driver's default window differs, and every deep-stage result needs re-reading.
5. **The four graph-capture gates** (`gpu.rs:18355`, `19006`; `serving.rs:1800`, `2146`) — plan §8
   risk 3's concrete surface. All four route through `with_event_tracking_disabled`
   (`gpu.rs:18293`, `serving.rs:1680`), so a failure is a capture-semantics change, not the known
   cudarc event-tracking trap.
6. **`roofline_kernel_runs`** (`gpu.rs:17572`) — the suite's only timing-derived assert. A low reading
   on a contended container host is a *container* finding; re-run before believing it.
7. **`a_corrupt_cache_entry_degrades_to_the_ptx_jit`** (`cubin.rs:306`) — if this is the *only*
   failure, suspect the Modal Volume, not the GPU (§6.3).
8. **Any `wukong_driver` failure** — the 19 non-GPU tests there are CI-green on `ubuntu-latest`, so a
   failure among them is a Linux regression landed since the last CI run, not a device finding.

---

## 6. Harness items to settle BEFORE spending the money

These are not code defects; they are the ways this specific run can cost money and return less than
it should. Each cites the exact line.

### 6.1 `::test` does not pass `--nocapture`, so almost all of the free data is discarded

`modal_app.py:346-351` invokes `cargo test … -- <filter>` with no `--nocapture`. libtest **captures
stdout and stderr of passing tests**. On a fully green run — the outcome this document predicts —
that means the L4 returns the string `253 passed` and *nothing else*: no `GpuTarget` descriptor
(`gpu.rs:7148`), no `occupancy=N CTA/SM` tables (`gpu.rs:16111`, `20634`, `20736`), no f16 regime
bands (`gpu.rs:7440`), no `[skip:capability]` inventory, no `covered 217/357` coverage line
(`lower.rs:4385`), no autotune rankings (`autotune.rs:1312`), and no proof the driver leg touched the
device (`lib.rs:1985`). **Every §3 and §5.1 row above becomes unverifiable.** Add `--nocapture`.
Interleaving from parallel test threads is acceptable — the lines are individually prefixed
(`[gate]`, `[skip`, `  candidate`) — and the alternative (`--test-threads=1`) buys ordering at the
cost of wall time, which is itself a metered quantity.

### 6.2 `::test` requests no CPU while `::build` requests eight

`modal_app.py:299` decorates `build` with `cpu=8.0`; `modal_app.py:329` (`test`) and `:360` (`bench`)
request none. Per §5.1 the codegen_gpu leg is dominated by **CPU** work — a debug-build front end and
a debug tree-walking interpreter over 714 program-configs, twice. If the GPU container's CPU
allocation is a hard cap, that work runs on a fraction of a core while an L4 idles at $0.80/hr, and
`std::thread::available_parallelism` (cgroup-aware on Linux) may report 1, serialising libtest as
well. Confirm Modal's default `cpu` is a soft reservation, or mirror `cpu=8.0` onto `test`.

### 6.3 `WUKONG_CUBIN_CACHE` points at the network Volume, and the corpus gates abuse it

`modal_app.py:140` sets `WUKONG_CUBIN_CACHE=/persist/cubin-cache` (a `modal.Volume`). The corpus
gates emit a distinct PTX text per program per `-O` level, so `load_module_cached` (`gpu.rs:523-542`)
performs ~900 `cuLink` compiles each followed by `write_atomic` — a temp-file write plus a `rename`
(`cubin.rs:178-192`) — into a FUSE-backed network volume, and **not one of those entries is ever read
back**, because no two programs share PTX. Then `build_vol.commit()` (`modal_app.py:353`) ships them
all. Point the cache at container-local disk for the corpus leg, or accept the cost deliberately.
Second-order: `cubin.rs:325` does `write_atomic(&path, …).expect("plant a corrupt cache entry")`, so
if the Volume refuses that rename, `a_corrupt_cache_entry_degrades_to_the_ptx_jit` fails for a reason
that has nothing to do with the GPU (see §5.2 item 7).

### 6.4 `WK_TIMEOUT` defaults to 3600 s, and the suite's runtime is the unknown

`modal_app.py:63`. If the prediction in §5.1 is low by more than ~2.4×, the container is killed
mid-suite and the entire metered invocation returns nothing. Raise it for the first run; it costs
nothing unless it is used.

### 6.5 Confirm `::test` links rather than compiles

`build` (CPU container) and `test` (GPU container) share `CARGO_TARGET_DIR=/persist/target`
(`modal_app.py:137`) while the source is a *runtime mount*, not baked in (`modal_app.py:113`,
`copy=False`). Cargo's freshness check is mtime-based. If mounted mtimes are not stable across
invocations, `::test` recompiles the whole `--features gpu` graph **on the metered GPU container** —
exactly the cost `build` exists to avoid (`modal_app.py:300-305`). Watch the first lines of `::test`:
anything beyond a final link/`Running` line means abort and fix it in `build`.

### 6.6 Run `::device_info` first, every session

`modal_app.py:236` with `_EXPECTED["L4"] = ("sm_89", 58)` (`modal_app.py:220`). It costs seconds and
is the only thing standing between a MIG/vGPU slice and a set of occupancy numbers that mean nothing
(plan §6.1). Its `smem_optin` line is also the independent confirmation of §3.1's load-bearing
101 376 B prediction, from CUDA-C rather than from the Rust probe under test.

---

## Appendix — reproducing this inventory

The per-family counts and the coupling ledger come from a mechanical scan of every `#[test]` body in
`crates/wukong_codegen_gpu/src` and `crates/wukong_driver/src`: brace-match each test function,
classify it as device-executing if the body names `with_gpu(` / `with_fp8(` / `with_cap(` /
`gpu::gpu()` / `device_lost()`, and grep the body for the device-identity patterns listed in §2. Any
number in this document that is not stamped **PREDICTION** is checkable against the tree at
`d10d868` at the cited `file:line` — and per this repo's habit, **re-verify any load-bearing fact
before spending money on it**.
