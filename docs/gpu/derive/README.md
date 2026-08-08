# GPU retarget — Wave-0 derivation dossiers

Six research dossiers produced 2026-08-06 by the Wave-0 fan-out of the datacenter retarget
campaign (`GPU_RETARGET_PLAN.md` at the repo root, §0's derive-before-you-rent rule): everything
here cost $0 and touched no rented silicon. They are the paper derivations Phase 3's confirmation
sweeps test against — a prediction that fails on the metal teaches something; a blind sweep that
succeeds teaches nothing.

| Dossier | What it derives | Headline |
|---|---|---|
| [D1_h100_gemm.md](D1_h100_gemm.md) | H100 f16 GEMM: feasible tiles/stages from 227 KB SMEM, occupancy math, CUTLASS SM90 ground truth, the 6-point Act-1 confirmation grid | Act-1 (mma.sync) realistic ceiling 45–52% of cuBLAS as shipped, 62–68% with dynamic SMEM; the wgmma business case is a predicted 1.4–1.6× |
| [D2_h100_attention.md](D2_h100_attention.md) | Flash tile SMEM math, H100 occupancy limits, FA2/FA3 design, shelved-variant verdicts | The production 1-warp flash dispatch is hard-capped at 50% of H100 warp capacity by the 32-blocks/SM limit; `mp4/mp8` re-open as top priority; D=256 is register-bound, not SMEM-bound |
| [D3_a100.md](D3_a100.md) | Which 4050 verdicts flip on A100 (108 SMs, 163 KB, no fp8) | 3-stage pipelines flip to free (the 4050 NEGATIVE was an occupancy tax); the 64×64 warp tile does NOT flip — split-K/stream-K is the first A100 lever |
| [D4_sm120.md](D4_sm120.md) | Consumer Blackwell (RTX 5090 / PRO 6000): ISA deltas, SMEM, forward-compat | No wgmma, no tcgen05 on sm_120 — the sm_89-floor route is structurally sufficient; found the cubin-cache device-key bug; GeForce halves fp16 tensor throughput under f32 accumulate |
| [D5_peer_builds.md](D5_peer_builds.md) | Exact build recipes for the strong peers on cloud Linux (cuBLASLt, torch.compile, FA2/FA3/FA4, CUTLASS profiler, Marlin/machete) | CUDA 12.9.2 image + torch 2.13.0+cu129; cudarc 0.16.6 has no CUDA-13 bindings; FA4 is a pure-Python wheel; ~$0.50 of metered GPU for the whole smoke battery |
| [D6_dynamic_smem.md](D6_dynamic_smem.md) | Dynamic-SMEM mechanics **verified live on the 4050 through cudarc 0.16.6**, SMEM closed forms, the stage-depth decision model, the C2/C3 migration design | The extern window JITs under today's `.version 7.8`; `set_attribute` + `shared_mem_bytes` suffice; static >48 KiB is rejected by ptxas on every non-`a` target — dynamic SMEM is the only unlock |

## Predict-before-measure documents

Same provenance rules, different job: these are written *before* a measurement and are its
acceptance checklist. Read the relevant one first when a round comes back, and treat every row that
disagrees as a finding to explain rather than a number to accept.

| Document | Predicts | Measured yet? |
|---|---|---|
| [p1-expected-l4.md](p1-expected-l4.md) | What the device suite should do on a Modal L4 (`sm_89`, rented) | **No** — blocked on the workspace budget cap |
| [p2-expected-4050-identity.md](p2-expected-4050-identity.md) | That Phase 2 cost the 4050 nothing: `0e1b2ea` vs `main`, 11 body-identical benches, self-control | **No** — needs the laptop on AC power |

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

D6's re-runnable device probe (a small cudarc workspace) lives outside the repo in the session
scratchpad; its full output and the PTX it proved are inline in the dossier's Appendix A.
