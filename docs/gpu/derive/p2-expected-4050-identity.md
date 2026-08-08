# P2 — predict-before-measure: the 4050 perf-identity leg

**Status: PREDICTION. Nothing here has been measured.** Written 2026-08-08 against `8c66dc0`, on a
machine that was **on battery at the time of writing** and therefore could not time anything.
Per `GPU_RETARGET_PLAN.md` §0 — *write down the number you expect and why, then measure*.

This is the acceptance checklist for the last open Phase-2 item: plan line 614, *"on the **4050**,
before/after this phase must be behaviorally identical (same dispatch census, same suite results,
**same-run perf A/B ties**)"*. The census and suite legs were closed by the V2 wave (MIR census
byte-identical, device suite 253/0/87). **Only the timed leg is outstanding**, because the V2 agent
correctly refused to time on battery.

Provenance convention (`docs/gpu/derive/README.md`): **FACT** carries a `file:line` in this tree;
**DERIVED** is arithmetic over facts; **PREDICTION** is falsifiable and is an *input* to the
experiment, never a result.

---

## 0. The two arms and the instrument

| | commit | what it is |
|---|---|---|
| **A** | `0e1b2ea` | the pre-campaign tip (plan + Modal harness landed, no retarget code) |
| **C** | `0e1b2ea` | **the same binary as A, invoked again** — the control |
| **B** | `8c66dc0` | current `main`: Phases 0+2 landed, plus the CI/Linux fixes |

Instrument: `tools/perf/gpu_ab.ps1`. Baseline worktree at `C:\v2b` with `CARGO_TARGET_DIR=C:\v2bt`
(short paths — MAX_PATH). Both arms built `--release --features gpu`.

**Arm C is the whole point.** Two binaries cannot share a process, so "same-run adjacent A/B"
degrades to a tight alternation between processes, and process-level alternation has a noise floor
that no amount of care removes. C is byte-identical to A, so `C/A` has expected value exactly 1.00
and contains no code content: it measures **what this particular run can resolve**. A `B/A` inside
the `C/A` spread is a **tie**. This is the GPU analogue of the `C(twin)` column, and it exists
because an unpinned round on this machine once read a byte-identical binary as *"1.31x faster than
C"* (`pin-the-timing-thread`). The floor is measured per run; it is never a hardcoded threshold.

Two further protections, both encoded in the script rather than left to discipline: **rotating arm
order** (round `r` starts at arm `r mod 3`, so the first-launch GPU clock ramp does not always tax
the same arm), and a **discarded warm-up round 0** (the Phase-2 caches are device-keyed, so the two
arms write different cubin/autotune keys and the cold JIT would otherwise land entirely on whichever
arm ran first).

## 1. Which benches are even comparable

**FACT.** Each arm has exactly **87** `#[ignore]`d benches and the two *name* sets are identical —
nothing added, nothing removed.

**FACT.** Normalizing away everything `cargo fmt` can change (comments, all whitespace, trailing
commas before a closer), **64 of the 87 bodies are byte-identical across the arms and 23 differ**.
The unnormalized comparison reports 71 differing, which is an artifact of the Phase-2 `fmt` sweep,
not of the retarget.

**DERIVED — only the 64 are valid A/B rows.** A bench whose body changed is running different work
in the two arms; its two numbers are not a ratio of anything. The 23 that changed are exactly the
families Phase 2 touched by design (fp8 capability gates, int8/f16 dynamic-SMEM rows, conv
device-ceiling declines, the NVRTC peer's `compute_arch`). They must be *interpreted*, never
compared, and they are excluded from this leg.

**All 11 benches selected below are in the 64.** They are also DLL-free (no cuBLAS/cuBLASLt/cuDNN/
NVRTC/torch), because a peer proves nothing about *identity* and only adds variance:

`gemm_throughput` · `tensorcore_throughput` · `tensorcore_roofline_pct` · `flash_throughput` ·
`int8_swz_vs_handplaced` · `hbm_bandwidth` · `transformer_layer_throughput` ·
`resident_model_throughput` · `mega_vs_single_gemm` · `mega_vs_single_vmath` ·
`cubin_cache_compile_latency`

That spans `ptx.rs`, `ptx_gemm`, `ptx_wmma`, `ptx_flash`, `ptx_int8`, the megakernel and the cubin
cache. No fp8 bench is in the set, which matters for §2.1.

## 2. Causal inventory — what in Phase 2 could move a 4050 number

**2.1 PTX header floors — the one genuine risk.** FACT (`ptx_target.rs:25,30`): emission is now
centralized on `HDR_SM80` = `.version 7.8` / `.target sm_80` for **every family but fp8**, and
`HDR_SM89_V84` = `.version 8.4` / `.target sm_89` for fp8 only. FACT (`ptx_target.rs:3`): before
this module existed, **67 sites across 18 files hardcoded `.target sm_89`**. So every kernel in this
A/B now declares a *lower* target, and most a lower ISA version, than it did at `0e1b2ea`.

The driver JIT compiles PTX→SASS for the real `sm_89` device either way, and these are hand-written
PTX generators rather than compiler output, so the instruction mix is explicit in the text and the
JIT's remaining freedom is mostly scheduling and register allocation. That is the *reason to expect
a tie* — and it is also exactly the assumption that has never been measured. **This leg exists to
test it.**

**2.2 L2 de-literaling — provably identity on this device, no prediction needed.** FACT
(`gpu.rs:1039`): `f16_regime_thresholds(l2) = (l2*2/3, l2*2)`. DERIVED at the probed 24 MiB
(25 165 824 B): `(16 777 216, 50 331 648)` = exactly the old `16 MiB` / `48 MiB` literals, so the
f16 dispatch census cannot move on the 4050. FACT: `gpu.rs:7400`
`f16_regime_thresholds_reproduce_the_4050_literals` machine-checks this. **No regime flip is
possible here** — the "L2 was 12, not 24" correction changes other devices, not this one.

**2.3 Device-keyed cubin + autotune caches.** New keys ⇒ pre-change caches are clean misses ⇒ one
re-JIT and one re-tune. Neutralized by the discard round; if it leaks anywhere it is
`cubin_cache_compile_latency`.

**2.4 `Gpu::target()` probe at construction.** One extra probe per process — a fixed startup cost,
not a steady-state throughput cost.

**2.5 `sm_count()`.** The `.unwrap_or(20)` fallback became a real probe that returns **20** on this
device. Same value; no behavioral change here.

**2.6 Dynamic SMEM.** Plumbing added (`function_dyn`/`smem_budget`: 0 references in arm A, 27 in
arm B). The deep-stage rows it unlocks are opt-in variants, and every bench that selects one is in
the 23 excluded bodies — so this should not reach the 11 rows at all.

## 3. Predictions (falsifiable)

**P1 — the headline.** All 11 benches tie: `|B/A − 1| ≤ spread(C/A)`, per bench. Confidence high
for §2.2–2.6 (identity by construction or by test); the whole uncertainty is §2.1.

**P2 — the control resolves.** On AC+full and an idle machine, `C/A` lands within **±3%** per bench.
If the control spread exceeds **±5%**, the run cannot resolve a Phase-2-sized effect and **must
publish nothing** — re-run rather than reporting a wide tie as a tie.

**P3 — if anything moves, it is a tensor-core family.** `tensorcore_throughput`, `gemm_throughput`,
`flash_throughput` and `int8_swz_vs_handplaced` are the rows whose `mma`/`ldmatrix` selection is
most sensitive to the declared ISA version. A move in a *non*-mma row is more likely to be the
machine than the compiler.

**P4 — `hbm_bandwidth` is the canary and will not move.** It is a memory-bound streaming kernel from
`ptx.rs` with no `mma` at all, so a target-floor change cannot alter its instruction mix. **If
`hbm_bandwidth` moves outside the floor, suspect the instrument, not the retarget** — check power,
background load and thermals before believing any other row in the same round.

**P5 — `cubin_cache_compile_latency` may rise slightly on B** from §2.4's extra probe. Predicted
**< 1 ms**, and it is a startup finding, not a throughput finding. A larger move means the cache key
change is costing a re-JIT the discard round failed to absorb.

## 4. Abort and acceptance

**Publish nothing from a round in which any of these is true** — these are abort conditions, not
caveats to note afterwards:

- the power state changed between the before and after readings (the script records both and flags it);
- the round started on battery (the script refuses without `-Force`);
- the control spread exceeds ±5% (P2);
- any bench exited nonzero (recorded per invocation).

**Acceptance.** P1 holding across all 11 rows closes plan line 614's timed leg and the Phase-2 exit
criterion *"4050 identical"*.

## 5. Smoke run — plumbing only, on battery, NOT a measurement

The instrument was exercised end-to-end on 2026-08-08 at 66% battery, one bench (`hbm_bandwidth`),
warm-up round only. **No timing below is a perf result and none may be published**; both
observations are about the instrument.

**5.1 The discard round is load-bearing, not hygiene.** On the very first pass, arm B read
`copy 6.1 GB/s (3.2% of peak)` and `saxpy 5.9 GB/s (3.1%)` against arm A's `168.2` and `119.9` —
a 27x apparent collapse — while `reduce` was normal. Re-running arm B with its cache warm gave
`167.6 / 182.6 / 116.3`, i.e. **identical to arm A**, and 6.0 s wall instead of 70.9 s. The cause is
§2.3: arm B's device-keyed cubin keys are all cold misses, and the module JIT lands *inside* the
timed region for the first kernels touched. Without the warm-up round this leg would have reported
a spectacular false regression on its own canary bench. Do not "simplify" round 0 away.

**5.2 On battery the instrument resolves nothing, which is why the script refuses.** Arms A and C
are the same binary, and in this run they read `saxpy 119.9` vs `182.7 GB/s` — a control ratio of
**1.52x** where the expected value is 1.00. A Phase-2-sized effect is invisible under that floor.
This is the measured justification for P2's ±5% abort and for the battery refusal.

**A B-slower row outside the floor BLOCKS that exit criterion** and must be explained before Phase 3
spends money on H100 tuning — the retarget is not allowed to cost the existing target anything. A
B-*faster* row is equally a finding to explain, not a bonus to bank: nothing in §2 predicts a
speedup on this device, so an unexplained win means the instrument is wrong or the arms are not what
this document says they are.
