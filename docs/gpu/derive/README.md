# GPU retarget — Wave-0 derivation dossiers

Six research dossiers produced 2026-08-06 by the Wave-0 fan-out of the datacenter retarget
campaign (`GPU_RETARGET_PLAN.md` at the repo root, §0's derive-before-you-rent rule): everything
here cost $0 and touched no rented silicon. They are the paper derivations Phase 3's confirmation
sweeps test against — a prediction that fails on the metal teaches something; a blind sweep that
succeeds teaches nothing.

**D1, D3 and D6 carry corrections** (marked † below) — see the
[corrections ledger](#corrections-ledger-2026-08-09) before quoting a row from any of them. The
dossiers themselves still cost $0; some of the *corrections* are now backed by measurement (a $1.26
L4 round, a $0.258 CPU-container build, and dev-4050 rounds), and each one says which.

| Dossier | What it derives | Headline |
|---|---|---|
| [D1_h100_gemm.md](D1_h100_gemm.md) † | H100 f16 GEMM: feasible tiles/stages from 227 KB SMEM, occupancy math, CUTLASS SM90 ground truth, the 6-point Act-1 confirmation grid | Act-1 (mma.sync) realistic ceiling 45–52% of cuBLAS as shipped, 62–68% with dynamic SMEM; the wgmma business case is a predicted 1.4–1.6× |
| [D2_h100_attention.md](D2_h100_attention.md) | Flash tile SMEM math, H100 occupancy limits, FA2/FA3 design, shelved-variant verdicts | The production 1-warp flash dispatch is hard-capped at 50% of H100 warp capacity by the 32-blocks/SM limit; `mp4/mp8` re-open as top priority; D=256 is register-bound, not SMEM-bound |
| [D3_a100.md](D3_a100.md) † | Which 4050 verdicts flip on A100 (108 SMs, 163 KB, no fp8) | 3-stage pipelines flip to free (the 4050 NEGATIVE was an occupancy tax); the 64×64 warp tile does NOT flip — split-K/stream-K is the first A100 lever |
| [D4_sm120.md](D4_sm120.md) | Consumer Blackwell (RTX 5090 / PRO 6000): ISA deltas, SMEM, forward-compat | No wgmma, no tcgen05 on sm_120 — the sm_89-floor route is structurally sufficient; found the cubin-cache device-key bug; GeForce halves fp16 tensor throughput under f32 accumulate |
| [D5_peer_builds.md](D5_peer_builds.md) | Exact build recipes for the strong peers on cloud Linux (cuBLASLt, torch.compile, FA2/FA3/FA4, CUTLASS profiler, Marlin/machete) | CUDA 12.9.2 image + torch 2.13.0+cu129; cudarc 0.16.6 has no CUDA-13 bindings; FA4 is a pure-Python wheel; ~$0.50 of metered GPU for the whole smoke battery |
| [D6_dynamic_smem.md](D6_dynamic_smem.md) † | Dynamic-SMEM mechanics **verified live on the 4050 through cudarc 0.16.6**, SMEM closed forms, the stage-depth decision model, the C2/C3 migration design | The extern window JITs under today's `.version 7.8`; `set_attribute` + `shared_mem_bytes` suffice; static >48 KiB is rejected by ptxas on every non-`a` target — dynamic SMEM is the only unlock |

## Predict-before-measure documents

Same provenance rules, different job: these are written *before* a measurement and are its
acceptance checklist. Read the relevant one first when a round comes back, and treat every row that
disagrees as a finding to explain rather than a number to accept.

| Document | Predicts | Measured yet? |
|---|---|---|
| [p1-expected-l4.md](p1-expected-l4.md) | What the device suite should do on a Modal L4 (`sm_89`, rented) | **YES — 2026-08-08/09**, `bench/gpu/l4/2026-08-09-session.md`. 253/0/87 with zero `[skip]` lines; L2 48 MiB ⇒ bands `[32,96) MiB`; plan risk #3 answered. One miss: `wukong_driver` read 26, not the predicted 24 (+2 clippy regression tests). ≈$1.26. **Iteration data — container, unlocked clocks.** |
| [p2-expected-4050-identity.md](p2-expected-4050-identity.md) | That Phase 2 cost the 4050 nothing: `0e1b2ea` vs `main`, 11 body-identical benches, self-control | **YES — 2026-08-08/09**, results appended as that file's §6; raw logs `bench/gpu/4050-identity{,-deep}/round.log`. **Six certified ties, five consistent-with-tie**; 381 invocations, 0 nonzero exits. P3 was wrong, informatively. |

Both are now *post*-measurement documents and their own opening "Status: PREDICTION" lines are
stale; each carries a status-update stamp above it saying so, rather than being rewritten.

## Corrections ledger (2026-08-09) — read this before acting on any dossier row

The six dossiers were written **2026-08-06 at `5686f37`/`0e1b2ea`**, before Phase 2 landed and before
the last three waves. They are the input to every future rented-GPU decision, so where they are now
wrong they misdirect real money. Each correction below is stamped **at the affected file**, with the
original claim left visible beside it — this repo never quietly deletes a wrong claim. Every entry
cites a test, a commit or a log in the tree; anything that could not be cited is marked *unverified*
rather than asserted.

| # | Dossier / § | Claim | Correction | Evidence |
|---|---|---|---|---|
| 1 | **D6 §4.2 row 6** (and §2's conv row) | raising the 44 KiB conv SMEM gate "frees declined shapes" | **REFUTED — it was declining nothing anyone runs.** The gate declined only square filters `R=S>=34`; AlexNet conv1 (11x11) and the ResNet stem (7x7) fit even the *retired* gate. What binds the family is the **1024-thread block cap** and the per-thread `KB` accumulators, not shared memory. The row's real content is the implicit-GEMM tile: 128x64 = 38912 B is static-legal, 128x128 = 73728 B needs the window. | commit `d69e4da`; `ptx_conv.rs::conv_smem_census_across_target_budgets` |
| 2 | **D1 §1.7 / §5.1 / §5.2 / §6 row 8** | the f16 regime thresholds are literal 16/48 MiB, `l2_cache_size()` is dead code feeding a bench print, `sm_count()` falls back to `.unwrap_or(20)`, eight hardcoded `48*1024` asserts | **CLOSED BY PHASE 2 before the dossiers were read — do not re-fix.** `f16_regime_thresholds(l2_bytes) = (l2*2/3, l2*2)`, machine-checked to reproduce the old literals at the probed 24 MiB; `GpuTarget` has no defaults and an unqueryable SM count fails `Gpu::new`; the asserts are an explicit `smem_budget` parameter. **One remnant is genuinely still open: `gemm_nt_bf16` still branches on a literal `16 * 1024 * 1024`.** §6 row 7 (the `bk == 32` swizzle) is also still open. | `gpu.rs:1039`, `:2494`, `:130-131`, `:563`, `:572`, `:7570`; `f16_regime_thresholds_reproduce_the_4050_literals`; `ptx_wmma.rs:2085/2148`, `:2131`, `:2599` |
| 3 | **D1 §7 Q2** | *"does the driver accept a >48 KiB `.extern .shared` allocation after `cuFuncSetAttribute`?"* — listed as load-bearing and unmeasured | **ANSWERED, at home, for $0, in both directions.** 64 KiB launches and is exact at the far end of the window; the identical launch *without* the opt-in is refused by the driver. Measured on `sm_89`, tagged at the `sm_80` floor. | `gpu.rs::dynamic_smem_window_exceeds_the_static_48_kib_ceiling` |
| 4 | **D1 §7 Q3** | ptxas register allocation for A3/A4 — *"free at home"*, otherwise a rented part | **Downgraded to a CPU container** (there is no `ptxas`/`nvcc` on the dev box). `ptxas` compiles *for* an arch without needing one: the CUTLASS **sm90a** profiler built to completion on a CPU-only container for **$0.258**. Same build gave independent evidence for D1 §2.3's register wall — **112** `ptxas info : (C7511) … wgmma.mma_async instructions are serialized due to insufficient register resources` from NVIDIA's *own* `f16_f16_f32` kernels: 48 at `256x256x64`, 32 at `256x128x64`, 32 at `128x256x64`; 80 warp-specialised pingpong, 32 cooperative. | `bench/gpu/h100/2026-08-09-preflight-build-peers.log` (`:1074-1185`, and the `[meter]` line) |
| 5 | **D3 §2(c)** | norm-kernel grid underfill — the occupancy tables | **A PREDICTION FAILED AND IS RECORDED AS FAILED.** The SM-fill work predicted **1.4–2.0×** in the covered regime and measured **1.11–1.16×, exactly 1.00× at 4096×4096**: the shipped kernel already runs ~168 GB/s against a ~192 GB/s pin rate, so on Ada the occupancy headroom is real and the *bandwidth* headroom is not. The genuine win is the underfill regime — **2.8–3.5× at rows=64, cols=8192**. Verdict (c) itself is not refuted; the lesson is that an occupancy table without a bandwidth denominator is not a speedup. | commits `a35892c` (measurement + protocol) and `d7ad1a8` (wiring); `norm_dispatch_reaches_every_planner_width_and_matches_the_cpu_oracle` |
| 6 | **D6 §4.1** (and anywhere else that reads L40S as a stepping stone) | *"the residency-dependent verdicts need L40S (142 SMs) or the target itself"* | **An L40S cannot execute the wide tiles or move a residency verdict.** It is Ada, cc 8.9, capping at ~99 KiB opt-in exactly like the dev 4050, so it **capability-skips every `PIPE_WIDE_VARIANTS` row (112–144 KiB)** — the same four skips the laptop already prints. L40S answers **SM scaling at the same ISA** (142 vs 20 SMs) and nothing else; **only A100/H100 can answer the wide tiles.** | `ptx_wmma.rs` `WIDE_SMEM_BUDGET` (= 166912, A100's opt-in) and `PIPE_WIDE_VARIANTS`; the two-directional skip accounting in `gpu.rs::gemm_deep_matches_reference_within_tol`; D2 §1's `sm_89` device column |

**Outside this directory, and therefore not fixed here** (flagged for whoever owns those files):
`tools/cloud/README.md` §"Start on L40S or L4, not A100" and `GPU_RETARGET_PLAN.md` §5 Phase 1
step 3 both frame the L40S as the next rung; that framing is correct **for SM scaling** and must not
be read as a route to the wide tiles or to any dynamic-SMEM residency question. `gpu.rs:2494`'s bf16
literal is a code fix with a different owner.

## Provenance and epistemic status

Every dossier separates **FACT** (with a source URL), **DERIVED** (arithmetic from facts), and
**PREDICTION** (falsifiable, to be tested on silicon). Predictions are *inputs to experiments*,
not results — nothing in here is a benchmark claim, and none of it belongs in `BENCHMARKS.md`
until measured under §6 of the plan.

Two corrections were found by same-day verification and are stamped at the top of the affected
files rather than silently edited: **D1** derived L2 multiples from the 12 MiB spec-sheet figure,
but the device probe reads 24 MiB (the multiples that preserve 4050 behavior are 2/3× and 2×);
**D5** claimed the `12.8.1-cudnn-devel` Docker tag was gone — it is not (the 12.9.2 bump stands,
for alignment reasons). The habit generalizes: these documents were produced by research agents
and verified opportunistically, so re-verify any load-bearing fact before spending money on it.
**Six more corrections were found by measurement over the following three days — the ledger above is
the index, and each is stamped at its own site.** The pattern in all eight is the same and is worth
naming: a *derivation* stays true only as long as the tree it cites does. Re-read the cited
`file:line` before you spend, and treat "this dossier says X" as a hypothesis about the code, not a
fact about it.

D6's re-runnable device probe (a small cudarc workspace) lives outside the repo in the session
scratchpad; its full output and the PTX it proved are inline in the dossier's Appendix A.
