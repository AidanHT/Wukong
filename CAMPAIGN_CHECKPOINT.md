# CAMPAIGN CHECKPOINT — 2026-08-11 23:5x, state refreshed 2026-08-12 (for a cold resume by ANY harness)

This file is the single resume point for the **Wukong H100 SOTA campaign** (Act 2). It assumes the
reader has NO prior session context. Read this, then `docs/gpu/derive/ACT2_WAVE_PLAN.md` (its
DOSSIER AMENDMENTS section first), then the three wave dossiers in the same directory.

## Objective and standing constraints

Make Wukong's GPU backend heavily exceed SOTA on H100, with honest peers and a refusing
measurement instrument. Hard rules that bind every commit:

- **Gate before ANY commit, all six, unpiped, each its own step:**
  1. `cargo test` (48 suites; exit code is the verdict — NEVER pipe into grep, the pipe's exit
     status is grep's and `cargo test | grep X && git commit` commits on failure)
  2. `cargo check --features gpu --all-targets` (plain cargo test never builds the gpu feature)
  3. `RUSTFLAGS="-D warnings" cargo clippy --features gpu --all-targets`
  4. `WUKONG_GPU_REQUIRED=1 cargo test -p wukong_codegen_gpu --features gpu`  (4050 device suite;
     468 passed at the 2026-08-11 checkpoint, **496 passed / 0 failed / 95 ignored** after the
     2026-08-12 Wave-3 merge)
  5. `WUKONG_GPU_REQUIRED=1 cargo test -p wukong_driver --features gpu --lib` (the driver-side
     dispatch laws: which route a recognized GEMM takes per capability, which shapes and operand
     *values* the wgmma seam declines, the route witness. **Part 4 does not reach them** — it is
     `-p wukong_codegen_gpu` — and neither does part 1, which never passes `--features gpu`: the
     crate reports 20 tests without the feature and 47 with it. The 27 in the delta had no runner
     at all until 2026-08-12. Device-free apart from six e2e gates that `[skip]` without a device,
     so it is also the CI step `gpu-check` runs on a plain ubuntu runner.)
  6. `cargo fmt --all -- --check`
- Commits: conventional subject + prose body naming the DEFECT, the REPRODUCER, and a
  `Verified:`/`Gate:` line quoting unpiped output. **NO `Co-Authored-By` trailer, ever.**
  Never `git add -A` — stage files by name.
- PTX must be pure ASCII (one `×` in a format! is a ptxas fatal). Every generator family has an
  ASCII law.
- `crates/wukong_codegen_gpu/src/{gpu.rs, ptx_wgmma.rs}` have ONE owner per wave — never two
  agents in them concurrently.
- Metered cloud spend: ≈ **$6.1 of a $25 cap** so far. GPU rounds are Modal
  (`tools/cloud/modal_app.py`); ALWAYS `export PYTHONIOENCODING=utf-8` first (cp1252 console
  kills the CLI while the container keeps billing) and ALWAYS `modal run --detach` (a local DNS
  flake once killed a healthy 20-minute build). `modal app list` checks for orphan billing.
- Every H100 visit is preceded by the ~$0.01 CPU ptxas census:
  `modal run tools/cloud/modal_app.py::ptxas --filter ptxas_reports_the_register_and_spill_budget --tag <tag>`
  (the filter MUST name the audit test — `--filter wgmma` selects the wrong tests; there is a law).
- Bench entrypoints: `::bench --name X` reaches ONLY `#[ignore]`d tests; `::test --filter Y`
  reaches only plain `#[test]`s. Routing a name through the wrong one runs ZERO tests and exits 0
  (there is now a reachability law; the $0.006 reproducer is round log r2a).

## Measured state at this checkpoint (all logs in bench/gpu/h100/, commits on main)

**The shipped default** is `wgmma_w1_for(m,n)` (f16) / `wgmma_w1_bf16_for` (bf16) in
`crates/wukong_codegen_gpu/src/ptx_wgmma.rs`: the fused `st.global.v2.f32` epilogue on both
regime arms, 1x2x1 B-multicast cluster at/above 8e6 output elements, un-clustered below.

Published table, % of cuBLAS (f32 out), through the instrument (twin control, ±5% bar,
refusals honest), 2026-08-11:

| shape | f16 | bf16 |
|---|---|---|
| sq1024 | refused (peer floor ±15.5%) | 49.5 |
| sq2048 | **95.3** | 94.0 |
| sq4096 | **88.6** | 89.0 |
| sq8192 | **92.3** | 91.9 |
| gpt_d1024_up | 80.3 | 80.0 |
| gpt_d1024_down | **97.3** | 95.4 |
| gpt_d4096_up | 73.4 | 73.2 |

Suite mean went **61.5% → ~87.9%** in one wave; the single lever was the v2 store
(+25.2/+15.3/+10.1 points; commits fd41ebf, a1e3cd8). Other measured facts:
- **nostore diagnostic 114.0/101.1/101.8%** — the mainloop is at-or-above cuBLAS everywhere; the
  ENTIRE remaining gap is epilogue + wave overhead (= Waves 3/4's mechanisms exactly).
- K-sweep: the scalar epilogue costs a measured **17.7 µs** at M=N=2048 (intercepts 22.33 vs 4.68).
- First H100 HBM measurement: copy **2922 GB/s = 87.2%** of the 3352 spec peak.
- Fused int8-dequant **beats the cuBLAS GEMM+dequant chain 1.08–1.15×** (first outright peer win).
- int8 Ada tuning does NOT transfer (22–52% of cuBLAS IMMA) → Wave-5 int8 needs its own tile
  search. int4: no bindable library peer (documented lead).
- cuBLASLt fuses RELU/GELU/BIAS at BOTH f32 and f16 out on H100; **SiLU is absent from its
  epilogue enum** — the highest-margin Wave-4 target (`silu(x·Wᵀ+b)`, derived 1.451× at r=0.82)
  has no library peer.
- Evict-first/evict-last L2 hints: measured null (publishable null).
- Clock lock is REFUSED in Modal containers (recorded per run now); one bf16 round self-refused
  on +6.82% SM clock drift — the instrument and its `[clock]` provenance lines work.

### The 2026-08-12 Wave-3 visit (logs `2026-08-12-h100-w3-r*.log`, commit f109847; ~$0.44 total)

**SWEEP-ARM results, NOT yet the shipped rule** (the shipped `wgmma_w1_for` still selects the
wave-2 arms; the 2026-08-11 table above remains the shipped-kernel table). Config sweep: 29 rows x
7 shapes = 92 cells, every delta below outside both floors ("moved"), clock lock refused and
declared, f16/f32out:

| shape | control (w1_mcb_v2 / w1_v2) | best wave-3 arm | arm |
|---|---|---|---|
| sq8192 | 92.0 | **102.0 — ABOVE cuBLAS** | `_v2_r16_p` (raster+persistent) |
| gpt_d4096_up | 73.3 | **101.6 — AT/ABOVE cuBLAS** | `_v2_r16_p` |
| sq4096 | 88.3 | **94.3** | `_v2_p` (persistent) |
| gpt_d1024_up | 80.6 | **91.8** (persistence alone 90.0) | `_v2_r16_p` |
| sq1024 | 48.3 (w1_v2) | **79.6** | `w3d` 128x64 tile |
| sq2048 | 94.1 (w1_v2) | single-wave control: flat (93.5 p) | — |

Mechanism checks: the r32 bracket lands BETWEEN linear and r16 at gpt_d4096_up (90.9 vs 94.5) —
the raster's gain is traffic, as derived; pstop captures part but not all of the persistent gain
(99.9 vs 102.0 at sq8192) — the continuous ring is real on top of dispatch; drain arms (fh/d1)
null within ~1 point everywhere — the dossier's zero budget, measured. Bring-up + cluster
exactness green on the FIRST device execution of the persistent/rastered/drain PTX (no hang, G19
reachable). **PRE-REGISTERED FALSIFIER FIRED: gpt_d1024_down prefers NO cluster (96.8 vs 91.4) at
f_L2 = 1.000** — dossier 3.8 falsifier (i): the cluster predicate needs a `waves` term (every
wrong ON was single-wave; every multi-wave ON won). `wgmma_vs_cublas` was deliberately NOT run —
it measures the superseded rule; the promotion (winners + waves-term predicate + bf16 twins) then
ONE rule re-measure round publishes the new suite table.

## In flight at the checkpoint (STATE REFRESHED 2026-08-12 after the Wave-3 merge)

1. **Wave 3 is MERGED to main** (`9ef142b`, all four levers: cluster-index raster, persistent
   clusters + deadlock law + G19, per-shape dispatcher with the thrashing-aware cluster predicate,
   drain arms). Implemented to `WAVE3_DOSSIER.md`, adversarially reviewed to an EMPTY findings
   list, full gate green on main (48 suites / 1020 passed; device suite 496/0/95).
   EXPECTED_MODULES = 128, wgmma in-family = 34; 11 new modules all census-enumerated. **It is
   UNMEASURED** — the next metered step is: census (`::ptxas --filter
   ptxas_reports_the_register_and_spill_budget --tag act2-w3`) → `::build --release` + `::build` →
   one H100 visit (`::bench --name wgmma_config_sweep --peers`, then `::bench --name
   wgmma_vs_cublas --peers`, `WK_GPU=H100`, `--detach`, utf-8 export, clock-lock attempt DECLARED).
   Known divergence deliberately left for a docs pass: WAVE3_DOSSIER/ACT2_WAVE_PLAN state the
   cluster bracket in `T_L2/T_measured` units; the shipped `f_L2` is a predicted-denominator
   quantity with the bracket `[0.769, 0.812]` and the threshold in saving units (0.26) — every
   code site that quotes them records the divergence.
2. **The wgmma product call site is MERGED** (`6df8d91`, branch gpu/wave4-wiring): `--backend=gpu`
   on cc==9 routes recognized f16/bf16 GEMMs to `gemm_nt_wgmma` behind a witnessed
   decline-never-wrong seam; gate part 5 (driver-gpu suite, 47 tests) and the CI step exist
   because of it. No H100 number has been measured through the route yet.
3. **Two implementation lanes are RUNNING in worktrees** (Opus/xhigh, adversarial-review chains):
   `gpu/wave4-epilogue` (`.claude/worktrees/w4-epilogue` — Wave 4 fused epilogues, OWNS
   gpu.rs/ptx_wgmma.rs) and `gpu/wave6-serving-bitexact` (`.claude/worktrees/w6-serving` — Wave 6's
   bit-exact levers: merged QKV, norm planner fix, v4 paged-attention coalescing, K-side GQA fold;
   forbidden from the owned files). A cold-resuming harness: inspect those worktrees' `git log`
   and continue per their dossiers.
4. Task #21 (push + CI) remains pending on the user's request only. Before any push:
   `rustup update stable` (CI's rustc is newer than local and runs clippy -D warnings).

## Next steps after Wave 3 (the plan's order)

- Wave 4: fused epilogues (bias/ReLU/GELU vs the cuBLASLt fused peer at BOTH out-dtypes; SiLU
  unopposed). The recognizer→offload wiring LANDED 2026-08-12 (`6df8d91`); the epilogue lane is
  in flight on `gpu/wave4-epilogue`. Break-evens priced by the measured 2.92 TB/s.
- Wave 5: 8-bit wgmma. The descriptor transfers from 16-bit IFF `bk*dtype.size()==128` (BK
  64→128); K-major only; fp8 exact arm only K≤256; int8 `==` on s32. The CUTLASS f16+fp8+int8
  profiler binary is ALREADY staged on the `wukong-build` volume (peers.json is the authority) —
  do NOT rebuild it; only the H100 profiler MEASUREMENT runs remain.
- Wave 6: decode/serving (FA2 wheel with kvcache is staged on the volume). Its dossier is MERGED
  (`docs/gpu/derive/WAVE6_DOSSIER.md`, 2026-08-12): the decode step is WEIGHT-bound, not KV-bound —
  dtype levers first (int4 W + int8 KV, derived 3.042x at B=16), Stream-K promoted, ping-pong
  refuted a third time. The bit-exact lever subset is in flight on `gpu/wave6-serving-bitexact`.

## Where everything is

- Plan + dossiers: `docs/gpu/derive/ACT2_WAVE_PLAN.md`, `WAVE{3,4,5}_DOSSIER.md` (dossier beats
  plan on conflict), `README.md` (index).
- Round logs: `bench/gpu/h100/2026-08-11-h100-w2-r*.log` (r1–r12), censuses
  `2026-08-11-ptxas-act2-w2*.{log,json}`.
- Cloud harness: `tools/cloud/modal_app.py` (+ device-free selftest
  `python tools/cloud/peers/selftest_modal_helpers.py` → must print "FAILED: nothing").
- Staged peer bar (wukong-build Modal volume): CUTLASS v4.6.1 sm90a profiler (368 f16 / 782 fp8 /
  308 int8 kernels), flash-attn 2.8.3.post1 wheel (kvcache true), vLLM 0.26.0 venv, torch
  2.13.0+cu129, ptxas 12.9.86 archive. `peers.json` on the volume records it all.
- This box: RTX 4050 (sm_89) — sm_90a rows capability-skip locally; the wgmma laws are
  device-free. No clang/llc; MSYS2 gcc/g++/rustc present.
