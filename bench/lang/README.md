# `bench/lang/` — the language ruler

`WUKONG_MASTER_PLAN` Task 0.4: the honest gpu-native baseline of a `@generic` kernel. It was skipped
when Phase 0 was executed, so Phases 1 and 3 shipped speed features with **zero** measurement and
Phase 4's own criterion ("improves vs Phase 3") had no baseline. These fixtures plus
`crates/wukong_driver/src/lang_bench.rs` are that baseline.

**Nothing in `cargo test` runs this directory.** The correctness twins are `tests/run/lang_*.wk`,
which stay tiny and print every lane; these are timing artifacts at `N = 1<<22`.

```sh
cargo test -p wukong_driver --features gpu --release --lib lang_ruler_round -- --ignored --nocapture
```

`--release` is not optional: a debug binary makes the compiler's own stages look ~7x slower
(`docs/compile-floor.md` §2), and the CPU peer arm is a debug/release difference outright.

## The shared skeleton, and why it is forced

Every fixture declares the same two `[f32; 4194304]` locals (16 MiB each, 32 MiB of frame) and runs
the same *anchor* loop, differing only in what it adds after. Three properties of gpu-native force
that, and each cost a probe to find:

1. **`alloc_f32` never reaches the device.** gpu-native declines `wukong_rt_alloc`
   (`UNSUPPORTED: call to 'wukong_rt_alloc' not yet lowered to PTX`), so the buffers must be `main`
   locals — which the megakernel homes in the shared `.global` frame it allocates for the grid.
2. **A program with no `wukong_parallel_for` and no recognized kernel is not megakernel eligible.**
   It falls through to the single-thread lowering, whose frame is per-thread `.local`; at 32 MiB the
   driver rejects that launch with a bare `DriverError(CUDA_ERROR_INVALID_VALUE)` — not an
   `UNSUPPORTED:` decline. Hence the anchor loop, whose only job is eligibility.
3. **`[0.0; N]` is not free.** It lowers to a 4x-unrolled *scalar* store loop the `@parallel`
   outliner does not claim, so each declaration is N serial device stores.

`base.wk` is the skeleton and nothing else, so **every number here is `kernel_ms(x) - kernel_ms(base)`.
A raw `kernel_ms` from this directory is not a result.**

`@generic` is on all of them. Without it the recognizers claim these loops and the number becomes a
measurement of the dispatcher — the one thing this campaign exists not to do. `lang_bench` asserts
that no `wukong_*` kernel other than `wukong_parallel_for` survives in any fixture's MIR.

| fixture | adds | regions in `--emit=mir-high` |
|---|---|---|
| `base.wk` | nothing | 1 (the anchor) |
| `write_grid.wk` | `dst[i] = (i % 1024) as f32` | 2 — the shape that **does** reach the outliner |
| `saxpy_lane.wk` | `dst[i] = src[i] * 2.0 + 1.0` | 1 — the shape that **does not** |

## Measured, RTX 4050 (20 SMs), 2026-08-22

Two rounds, adjacent, same process, same build. Box on AC and charging at 35–42% for both — the
middle of this laptop's three power states, so the CPU arm is under its ~25% all-core cap. The GPU
arms are same-run adjacent ratios, which is what makes them reportable anyway.

| | `kernel_ms` (min of 7) | round 2 |
|---|---|---|
| `base` | 133.73 | 134.38 |
| `write_grid` | 139.86 | 139.91 |
| `saxpy_lane` | 271.89 | 271.44 |

The loop alone, per element (4194304 iterations each):

| loop | ms | ns/element | GB/s | % of the 4050's ~192 GB/s |
|---|---|---|---|---|
| `write_grid` — outlined, 256 threads | 5.53–6.13 | 1.32–1.46 | 2.7–3.0 | **1.4–1.6 %** |
| `saxpy_lane` — vectorizer-claimed, 1 thread | 137.06–138.16 | 32.7–32.9 | 0.24 | **0.13 %** |

**~22–25x per iteration**, for the same trip count. The bodies differ by one load and one FMA, which
on a bandwidth-bound device is at most a 2x byte difference — so the gap is parallelism, not the body.

And against the CPU, comparing `kernel_ms` (which excludes the GPU's compile *and* module load, so it
flatters the GPU) to `wukong_codegen_cranelift::jit_run`'s whole call:

| program | gpu-native | CPU native | gpu-native is |
|---|---|---|---|
| `base` | 133.7–134.4 ms | 18.9–20.4 ms | **6.6–7.1x slower** |
| `write_grid` | 139.9 ms | 19.4–19.7 ms | **7.1–7.2x slower** |
| `saxpy_lane` | 271.4–271.9 ms | 20.5 ms | **13.3x slower** |

**A loss is a result.** Two defects explain all of it, and both were found by this round:

- **`@parallel` never reaches the canonical elementwise loop.** `FnLowerer::lower_for` tries
  `try_vectorize_for` *before* `try_emit_parallel_region`, so `dst[i] = src[i] * 2.0 + 1.0` is
  claimed by the CPU SIMD vectorizer and the region outliner is never asked. On a CPU that ordering
  is right. On gpu-native it inverts: the vectorized form runs on one device thread, four lanes at a
  time. `saxpy_lane.wk` is the fixture; the 22–25x is the price.
- **The megakernel's grid-parallel path is unreachable from any Wukong program.** `lower.rs` — the
  emitter — contains **zero** occurrences of `p_gbar` or `mrt_grid_barrier`, so `emit_mega_ptx`
  always produces the two-parameter block-scoped entry, `plan_grid` returns `grid = 1`, and the
  program runs on **one CTA of the 80 this device can hold resident**. The dumped PTX says so
  directly: `.visible .entry wukong_mega(.param .u64 p_ctx, .param .u64 p_frame)` and
  `gpu-mega: grid 1x256 (block-scoped), 4 blocks/SM x 20 SMs = 80 resident max`. The grid barrier,
  the cooperative launch and `plan_grid`'s cooperative branch are all exercised only by hand-written
  PTX in `megakernel.rs`'s own tests. So "grid-stride `@parallel`" is today a **block**-stride, and
  the ledger's and fixture headers' wording overstated it.

The round's own Phase 1 A/B — the same program at 1 CTA against a full resident wave — reads `tie`
for exactly that reason: with a block-scoped entry both arms *are* 1 CTA. The twin protocol caught
it (`C/A` floor +0.01%, `B/A` -0.01%, margin 0.29x, verdict `tie`) rather than publishing a number
for an experiment that never varied anything, which is what the control is for.

## Adding a fixture

Keep the skeleton, add one loop, and say in the header what bytes it moves *derived from the source*.
Then extend `lang_bench`'s byte table — it deliberately has no counter, so an unlisted fixture reports
`NaN` GB/s instead of a plausible wrong number.
