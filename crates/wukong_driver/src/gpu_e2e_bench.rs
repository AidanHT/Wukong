//! **The end-to-end round: a real `.wk` program, through the real pipeline, against cuBLAS.**
//!
//! # The defect this module exists to close
//!
//! The campaign's headline GEMM numbers come from `wukong_codegen_gpu::gpu::time_gemm_nt_wgmma`,
//! which launches the PTX generator's kernel **directly**: no lexer, no parser, no sema, no
//! `mir_build`, no recognizer, no interpreter, no offload seam. The recognizer -> offload route
//! landed separately (`6df8d91`), and `CAMPAIGN_CHECKPOINT.md` records the gap in one sentence:
//! *"No H100 number has been measured through the route yet."*
//!
//! Under `GPU_RETARGET_PLAN.md`'s own rule — *"a number counts only when it comes from a real Wukong
//! program, through the real pipeline, producing correct output and fast PTX"* — that makes the
//! published figure a claim about **the kernel Wukong's backend emits, launched directly**, and not
//! yet a claim about **Wukong**. This module measures the second thing.
//!
//! # What is timed, and why each quantity is named separately
//!
//! A single "end-to-end milliseconds" number would hide the interesting part, so the round reports
//! four costs and never folds them together:
//!
//! | quantity | what is inside it | how it is obtained |
//! |---|---|---|
//! | `compile_ms` | lex, parse, sema, `mir_build` (the recognizer), `wukong_opt` at `-O2` | host only, no device, best of [`COMPILE_REPS`] |
//! | `e2e_ms` | the offload seam's whole per-kernel host-to-host cost: interpreter + [`GpuAccel`] + host f32->f16 conversion + H2D + kernel + D2H + the seam's copy back into the user's buffer | `wukong_interp::run_kernel_f32_accel` |
//! | `launch_ms` | the same conversion, copies and kernel **without** the pipeline: `gpu::gemm_nt_wgmma` called directly, host slices in, host `Vec` out | the direct-launch peer of `e2e_ms` |
//! | `kernel_ms` | the kernel alone, device-resident buffers, no copies — **the campaign's published quantity** | `gpu::time_gemm_nt_wgmma` |
//!
//! `e2e_ms` is **not** the whole `--backend=gpu` CLI path and must not be described as one:
//! [`crate::compile`] additionally reads the file, runs `load_imports`, gates on
//! [`crate::verify_or_ice`], and enters the interpreter at `main` through `run_with_output_accel`.
//! This round enters at the kernel function with buffers bound by the harness, because a `main` that
//! allocates an 8192-square array as a local would make the number a measurement of the
//! interpreter's allocator. Same interpreter, same `GpuAccel`, same recognized call — one stage
//! shorter, and named that way.
//!
//! `launch_ms` and `e2e_ms` both also contain something `kernel_ms` does not: `gemm_nt_wgmma`
//! **regenerates the PTX text on every call** and hands it to a module lookup that is a cache hit —
//! it does not re-load the module, though it does re-hash the whole text as a key-collision guard
//! (`ptx_fingerprint`, GUARD G4), so the per-call cost is `O(|PTX|)` and not free — while
//! `time_gemm_nt_wgmma` resolves once outside its timed region. Both host-to-host arms pay it
//! identically, being the same function, so it cannot bias `pipeline cost`; it is part of what
//! retention measures, alongside the copies.
//!
//! **One asymmetry does bias `pipeline cost`, upward, against Wukong, and is measured rather than
//! argued.** `wukong_interp::run_kernel_f32_accel` wraps every call in `with_big_stack`, which spawns
//! and joins a fresh 512 MiB-stack thread; a real program pays that once per run, but arm B pays it
//! once per measured iteration and arm A never does. The round times an empty thread of the same
//! shape and prints it as `interp thread us`; since `pipeline cost` is a ratio, the subtraction is
//! done in the RETENTION table's milliseconds rather than on the percentage.
//!
//! From those, two published ratios and one attribution:
//!
//! - **`pipeline cost` = `e2e_ms` vs `launch_ms`** — both arms are the *same* Wukong function, so the
//!   difference is the pipeline's own addition: one interpreter entry, the seam's dispatch, the
//!   `copy_from_slice` into the user's buffer, and the thread above.
//! - **`kernel_ms` vs `peer_ms`** — both device-resident, both with their cuBLAS/module handles
//!   hoisted out of the timed region, i.e. the published table's own comparison re-measured in this
//!   same round so the two share a denominator, a clock and a container.
//! - **`kernel_ms / e2e_ms` — the RETENTION**: what fraction of the kernel-only number survives the
//!   trip through the interpreter and the offload seam. The compile is in NEITHER quantity — it runs
//!   once per shape outside both twins — so retention is about execution only. `launch_ms` splits it
//!   in two:
//!   `kernel_ms / launch_ms` is the copy-and-conversion tax the resident-buffer bench does not pay,
//!   and `launch_ms / e2e_ms` is the pipeline's own share.
//!
//! **`e2e_ms` must never be quoted against `peer_ms`.** One includes both PCIe crossings and the host
//! conversion and the other does not. The honest end-to-end peer would be a cuBLAS entry that is also
//! host-to-host *and* hoists its handle; `baselines::cublas_gemm_nt_f16_f32out` is host-to-host but
//! constructs a `CudaBlas` (a workspace allocation) on every call, so timing it as an arm would
//! charge the peer a cost Wukong's arm does not pay — a bias in *our* favour. That comparison is
//! therefore NOT made here; it is a HANDOFF for a `time_cublas_gemm_nt_f16_f32out_h2h` in
//! `baselines.rs`, and the one-shot's wall time appears only in the DIAGNOSTIC block, labelled as a
//! lone timing that includes handle creation.
//!
//! # The instrument, and what makes it refuse
//!
//! Both ratios go through `wukong_codegen_gpu::bench_instrument`'s twin protocol, unmodified: arms
//! `A` and `C` run *identical* work so `C/A` is that shape's own noise floor, arm `B` is the
//! contender, and `Round::publish` hands back a number only when the round closed, the clocks agree
//! at both ends, and the cell cleared the pre-registered bar. A shape whose floor does not clear the
//! bar prints `-- (Unresolved)`, not a number.
//!
//! Two refusals are this round's own, because the instrument cannot see either:
//!
//! 1. **The route witness.** `GpuAccel::calls` is route-blind — `sgemm_nt` bumps it identically
//!    whether the wgmma family took the call or `gpu_accel::wgmma_declines` sent it to
//!    `gpu::gemm_nt`. So before any timing, each shape runs once and must report
//!    `(gemm_wgmma_calls, gemm_existing_calls) == (calls, 0)`. Without it a declining shape would be
//!    timed as `gpu::gemm_nt` in arm B and as the wgmma kernel in arm A, and "retention" would be
//!    the ratio of two different kernels.
//! 2. **Correctness, at the bench shape.** A fast wrong answer is the failure mode a timing harness
//!    cannot feel. The fills are `X[i][k] = 1 + i % 3` and `W[j][k] = 1 + j % 5`, constant along `K`,
//!    so `Y[i][j] = K * (1 + i%3) * (1 + j%5)` **exactly**: the operand values 1..5 are exact and
//!    normal in f16, the products are integers, and `K * 15 <= 122880 < 2^24` is exact in f32 under
//!    any reassociation the tiling, split-K or raster order might choose. So it is an `==` check on
//!    **every lane** of `Y`, never a sum of them.
//!
//!    The two periods are coprime and differ (3 along rows, 5 along columns) **because an all-ones
//!    fill cannot see a layout bug**: with `X = W = 1` the answer is `K` in every lane under `A.B^T`,
//!    `A.B`, an i/j swap, or any permutation of the output buffer, and five of the seven grid rows
//!    have `M == N`, so even a whole-matrix transpose would be invisible. With these fills a
//!    transposed operand, a transposed output, an i/j swap and a row or column shift each land on a
//!    different integer somewhere in the buffer — verified numerically at all seven shapes.
//!
//!    Two limits, stated because a periodic fill has them by construction: a row shift by a multiple
//!    of 3 (or a column shift by a multiple of 5) is exactly invisible, which matters for a tile-swap
//!    bug at a stride that happens to be such a multiple — a 128-row M-tile is visible (`128 % 3 =
//!    2`) but swapping M-tiles three apart is not. And on a square shape the whole-matrix transpose
//!    is first distinguished at `j = 3`, since `x(0)*w(1) == x(1)*w(0)`.
//!
//!    What this check still cannot see, stated rather than glossed: the fills are constant along `K`,
//!    so a bug that permutes or drops terms *within* the K-reduction produces the same sum. Making
//!    the fill vary along `K` would cost `O(M*N*K)` host work at 8192 cubed, so that class stays
//!    where it is already covered — `tests/run/linear_nt_every_lane.wk` lane-by-lane on the CPU
//!    backends, and `ptx_wgmma`'s own pre-timing guard, whose two arms are an exact-integer
//!    permutation diagnostic and an independent f64 reference.
//!
//! # The program is the corpus fixture's program
//!
//! [`linear_tensor_src`] generates the same shape-typed, multi-index `Y = X.W^T` spelling that
//! `tests/run/linear_nt_every_lane.wk` pins lane-by-lane on both CPU backends, at the campaign's
//! shapes instead of a 3x6x8 one. `the_bench_program_is_the_fixtures_program` compares the two texts
//! so the timed program cannot drift into a dialect the gated fixture does not cover, and
//! `the_bench_program_is_recognized_without_a_device` proves — on a laptop, at zero cost — that the
//! generated source really does lower to `wukong_sgemm_nt`, which is the only thing that puts the
//! offload seam on the path at all.
//!
//! # Running it
//!
//! [`E2E_BENCH_INVOCATION`]. It is a `::bench` line and **must** carry `--package wukong_driver`:
//! `bench()`'s `package` parameter defaults to `wukong_codegen_gpu`, so without it the harness runs
//! `cargo test -p wukong_codegen_gpu ... gpu_e2e_route_vs_cublas`, selects zero tests, and exits 0
//! having measured nothing — the exact shape of the 2026-08-11 vacuous round.
//! `the_e2e_invocation_reaches_a_bench_that_can_fail` pins every link of that chain textually,
//! against `modal_app.py` and against this file.

use crate::gpu_accel::{self, GemmRoute, GpuAccel};
use std::time::{Duration, Instant};
use wukong_codegen_gpu::bench_instrument as bi;
use wukong_codegen_gpu::ptx_wgmma::{bench_iters, GemmPoint, HOPPER_SM_COUNT, WGMMA_BENCH_GRID};
use wukong_span::{Interner, SourceMap};

/// The bench's test-function name, so the reachability law selects on the same string the invocation
/// does rather than on a copy of it.
pub(crate) const E2E_BENCH_NAME: &str = "gpu_e2e_route_vs_cublas";

/// **The one invocation that runs this round.**
///
/// **Run `modal run tools/cloud/modal_app.py::build --release` first.** `bench()` calls
/// `_require_prebuilt(package, release=True)` while `build()`'s `release` defaults to `False`, so this
/// line on its own aborts before it measures anything — deliberately, since the alternative is
/// compiling the workspace in release mode on metered GPU time.
///
/// `::bench` (not `::test`) because the round is `#[ignore]`d: `::bench` is the only entrypoint that
/// passes `--ignored` on its own, and `::test` never passes it (though `::test --filter` splits its
/// argument, so `--filter "--ignored <name>"` could also reach an ignored test — this constant does
/// not use that route, because a sweep runner is the right home for a measurement).
/// `--package wukong_driver` because this bench does not live in `bench()`'s default package.
///
/// `--peers` sets `WUKONG_PEER_REQUIRED`, which the peers *inside* `wukong_codegen_gpu` read; it is
/// carried so this round's provisioning matches every other published GEMM round. It is **not** what
/// makes a missing cuBLAS fail here — that comes from `WUKONG_GPU_REQUIRED` through
/// `diff::skip_or_fail`, and `bench()` sets that unconditionally.
///
/// `WK_GPU=H100` books the part, because `wgmma` is architecture-locked to `sm_90a` and `WK_GPU`
/// defaults to `L40S`. It does **not** feed the capability escalation: `bench()` exports no
/// capability variable at all (only `::test` does), which is why the escalation in the round rides on
/// `WUKONG_GPU_REQUIRED` instead. That was a live defect — the escalation was inert under this very
/// line — and the two halves are pinned by
/// `the_e2e_invocation_reaches_a_bench_that_can_fail`.
///
/// `--detach` because a local DNS flake has killed a healthy round on this project before.
pub(crate) const E2E_BENCH_INVOCATION: &str = "WK_GPU=H100 modal run --detach \
     tools/cloud/modal_app.py::bench --name gpu_e2e_route_vs_cublas --package wukong_driver --peers";

/// The entry function in [`linear_tensor_src`]; its parameter order is the buffer order
/// `[x, w, y]`.
const E2E_ENTRY: &str = "linear";

/// How many times [`compile_through_pipeline`] is repeated for the compile-time reading. The compile
/// is device-free and takes single-digit milliseconds, so the reps are nearly free.
///
/// **The reported statistic is the MINIMUM across them, and that is `docs/compile-floor.md`'s regime,
/// not a shortcut**: that document measures warm steady-state best-of-N per stage and says in its own
/// section 2 that a debug binary "makes the backend look ~7x slower and the measurement is
/// worthless". Both conditions hold here only because the harness runs this round `--release`; a
/// `compile ms` column read off a debug build is not comparable to anything in that document.
///
/// One difference from it is deliberate and must be stated wherever these numbers are quoted: this
/// round times `lex..optimize` and **no backend**, because `--backend=gpu` has no codegen stage at
/// all. `docs/compile-floor.md`'s headline (0.35-1.00 ms warm, in-process) is `lex..emit_object`, in
/// which `codegen+obj` is 73.4% of the time — so the quantity here is the *front* subtotal, and
/// comparing it to that headline would understate Wukong's compile cost by roughly 4x.
pub(crate) const COMPILE_REPS: usize = 9;

/// The `wukong_*` runtime symbol the plain `A.B^T` recognizer emits, **as the MIR printer emits the
/// call** — the grep target for the manual GATE-BLIND check and what
/// `the_bench_program_is_recognized_without_a_device` asserts on.
///
/// The trailing `(` is load-bearing, not decoration. `wukong_sgemm_nt` is a PREFIX of
/// `wukong_sgemm_nt_epi`, `wukong_sgemm_nt_alpha` and both `_parallel` twins, and those route through
/// *different* `Accelerator` hooks — `sgemm_nt_epi` reaches the Act-1 wmma family and never wgmma at
/// all. A bare `contains("wukong_sgemm_nt")` would therefore stay green while the recognizer had moved
/// the nest onto a kernel this round cannot measure. Verified against the real printer output:
/// `call wukong_sgemm_nt(v0, v11, v22, v169, v109, v110, v3)`.
pub(crate) const NT_GEMM_SYMBOL: &str = "call wukong_sgemm_nt(";

/// **`Y = X . W^T` — the `nn.Linear` forward, in the spelling the language asks for.**
///
/// Shape-typed `Tensor` parameters, multi-index reads, `for i in 0..N` loops and a scalar accumulator
/// for the inner dot product. This is the body of `tests/run/linear_nt_every_lane.wk` with the
/// dimensions substituted, which is what makes the timed program and the lane-by-lane-gated program
/// the same program.
///
/// The K loop variable is named `k` and the K *dimension* appears only inside the types, so there is
/// nothing to shadow — and that separation is also why [`compile_through_pipeline`]'s cost barely
/// moves between a 1024-cube and an 8192-cube.
pub(crate) fn linear_tensor_src(m: usize, k: usize, n: usize) -> String {
    format!(
        "module bench.linear_nt\n\
         fn linear(x: Tensor[f32, {m}, {k}], w: Tensor[f32, {n}, {k}], mut y: Tensor[f32, {m}, {n}]) {{\n\
         \x20   for i in 0..{m} {{\n\
         \x20       for j in 0..{n} {{\n\
         \x20           let mut s: f32 = 0.0;\n\
         \x20           for k in 0..{k} {{\n\
         \x20               s = s + x[i, k] * w[j, k];\n\
         \x20           }}\n\
         \x20           y[i, j] = s;\n\
         \x20       }}\n\
         \x20   }}\n\
         }}\n"
    )
}

/// **The real pipeline, timed: lex -> parse -> sema -> `mir_build` -> `wukong_opt`.**
///
/// The same stages, in the same order, that [`crate::compile`] runs for `--backend=gpu`, and every
/// one of them is asserted clean — a program that stopped compiling would otherwise be timed as a
/// very fast pipeline.
///
/// **Two stages of the real CLI path are absent**, and both are named so the number is not read as
/// more than it is: `load_imports`, because the generated source has no `import` (a real user's file
/// pays it once per imported file), and `verify_or_ice`, which [`crate::compile`] runs before EVERY
/// backend entry. Neither is large, and neither is here.
///
/// Returns the lowered program, its interner, and how long the stages took. The recognizer runs
/// inside `mir_build`, so the returned MIR already carries the `wukong_sgemm_nt` call the offload
/// seam intercepts.
pub(crate) fn compile_through_pipeline(src: &str) -> (wukong_mir::Program, Interner, Duration) {
    let t0 = Instant::now();
    let mut sm = SourceMap::new();
    let id = sm.add("gpu_e2e_bench.wk".to_string(), src.to_string());
    let (tokens, ld) = wukong_lexer::tokenize(sm.source(id), id);
    assert!(!ld.iter().any(|d| d.is_error()), "lex errors");
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
    assert!(!pd.iter().any(|d| d.is_error()), "parse errors");
    let (sema, sd) = wukong_sema::check(&module, &interner);
    assert!(!sd.iter().any(|d| d.is_error()), "sema errors");
    let (mut program, md) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(!md.iter().any(|d| d.is_error()), "mir_build errors");
    wukong_opt::optimize(&mut program, 2);
    (program, interner, t0.elapsed())
}

/// The printed MIR, via the same printer `--emit=mir` uses, so a law reading it reads what a
/// developer would see.
pub(crate) fn mir_text(program: &wukong_mir::Program, interner: &Interner) -> String {
    wukong_mir::print::print_program(program, interner)
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    /// The sample-label field the host-to-host twin records into (pipeline vs direct launcher).
    const H2H_FIELD: &str = "h2h_ms";

    /// The sample-label field the resident-buffer twin records into (wgmma kernel vs cuBLAS, both with
    /// handles hoisted). A separate name from [`H2H_FIELD`] because the two twins measure genuinely
    /// different quantities, and one name across both would let a `publish` call read the wrong one.
    const RESIDENT_FIELD: &str = "resident_ms";

    /// The two quantities of one shape that are NOT twin arms: the device-free compile cost, and the
    /// host-to-host peer's one-shot wall time (a lone timing, diagnostic only — see the module doc on
    /// why it is not an arm). Every timed *device* quantity comes out of a twin verdict so it can
    /// refuse, which is why neither `e2e_ms`, `launch_ms`, `kernel_ms` nor `peer_ms` is a field here.
    #[derive(Debug, Clone, Copy)]
    struct ShapeCosts {
        compile_ms: f64,
        peer_h2h_once_ms: f64,
    }

    /// Loudly skip when there is no device, escalating under `WUKONG_GPU_REQUIRED=1` — which
    /// `::bench` always sets — through the one helper every skip in the workspace goes through.
    fn skip_no_device(name: &str) {
        wukong_codegen_gpu::diff::skip_or_fail(
            name,
            wukong_codegen_gpu::gpu::init_error().unwrap_or("no CUDA device reachable"),
        );
    }

    /// Row period of the `X` fill, and column period of the `W` fill. Coprime and different, so a
    /// transposed operand, an i/j swap or a shifted row lands on a different integer — see the module
    /// doc for why an all-ones fill cannot.
    const X_PERIOD: usize = 3;
    const W_PERIOD: usize = 5;

    /// `X[i][k] = 1 + i % X_PERIOD`, constant along K. Row-major, `m * k` long.
    fn fill_x(m: usize, k: usize) -> Vec<f32> {
        (0..m * k)
            .map(|t| (1 + (t / k) % X_PERIOD) as f32)
            .collect()
    }

    /// `W[j][k] = 1 + j % W_PERIOD`, constant along K. Row-major (`W` is `n * k`, K-major NT).
    fn fill_w(n: usize, k: usize) -> Vec<f32> {
        (0..n * k)
            .map(|t| (1 + (t / k) % W_PERIOD) as f32)
            .collect()
    }

    /// The exact expected lane: `Y[i][j] = K * (1 + i%3) * (1 + j%5)`.
    fn want_lane(i: usize, j: usize, k: usize) -> f32 {
        (k * (1 + i % X_PERIOD) * (1 + j % W_PERIOD)) as f32
    }

    /// **Every lane of `y` must equal [`want_lane`] exactly.** Reports how many lanes are wrong and
    /// the first offender with its 2-D index, because "one lane" and "every lane" are different bugs
    /// and the index is what tells a transpose from a shifted row.
    fn assert_every_lane(what: &str, y: &[f32], m: usize, k: usize, n: usize) {
        assert_eq!(y.len(), m * n, "{what}: output buffer is the wrong length");
        let mut wrong = 0usize;
        let mut first = None;
        for i in 0..m {
            for j in 0..n {
                let got = y[i * n + j];
                if got != want_lane(i, j, k) {
                    wrong += 1;
                    if first.is_none() {
                        first = Some((i, j, got));
                    }
                }
            }
        }
        if let Some((i, j, got)) = first {
            panic!(
                "{what}: {wrong} of {} lanes are wrong; first at y[{i}][{j}] = {got}, want {} \
                 (X[i][k] = 1+i%{X_PERIOD} and W[j][k] = 1+j%{W_PERIOD} are exact and normal in f16 \
                 and the product is an integer under 2^24, so every lane is exact bit-for-bit -- \
                 this is a WRONG ANSWER and no timing from this shape may be published)",
                m * n,
                want_lane(i, j, k)
            );
        }
    }

    /// **The cost `with_big_stack` adds to arm B and not to arm A**, measured rather than argued:
    /// `wukong_interp::run_kernel_f32_accel` spawns and joins a fresh 512 MiB-stack thread per call
    /// (`wukong_interp`'s `with_big_stack`), so this is the same shape with an empty body — same
    /// `thread::scope`, same `stack_size`, same `spawn_scoped`/`join`.
    ///
    /// Returns the **minimum** across reps in microseconds, not the mean: the root `CLAUDE.md` allows
    /// "best-of-N minima" and not averages, and the first spawn is cold, so a mean carries it into
    /// every later reading. Never panics — a failed reservation yields `NAN`, which the caller prints
    /// as `n/a`, because this is a diagnostic and must not redden a round that already completed.
    fn interp_thread_overhead_us() -> f64 {
        const STACK: usize = 512 * 1024 * 1024;
        const REPS: usize = 32;
        let mut best = f64::INFINITY;
        for _ in 0..REPS {
            let t0 = Instant::now();
            let ok = std::thread::scope(|s| {
                match std::thread::Builder::new()
                    .name("wukong-interp-probe".into())
                    .stack_size(STACK)
                    .spawn_scoped(s, || {})
                {
                    Ok(h) => h.join().is_ok(),
                    Err(_) => false,
                }
            });
            if !ok {
                return f64::NAN;
            }
            best = best.min(t0.elapsed().as_secs_f64() * 1e6);
        }
        best
    }

    /// **THE END-TO-END ROUND.** See the module documentation for what each column means and why
    /// `e2e_ms` may not be quoted against the resident-buffer peer.
    ///
    /// `#[ignore]` is load-bearing: [`E2E_BENCH_INVOCATION`] is a `::bench` line, `::bench` appends
    /// `--ignored`, and a plain `#[test]` here would be selected by zero tests and exit 0.
    #[test]
    #[ignore = "Act-2 end-to-end round; Hopper + cuBLAS; run explicitly (E2E_BENCH_INVOCATION)"]
    fn gpu_e2e_route_vs_cublas() {
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device(E2E_BENCH_NAME);
                return;
            }
        };

        // --- 0. capability, before a peer probe or any allocation ------------------------------
        // `sm_90a` is an architecture LOCK, so on this repo's own 4050 — and on a Blackwell part —
        // the correct outcome is a capability skip and nothing else. On a container that was SUPPOSED
        // to be Hopper it is a mis-provisioned round, and skipping quietly there is one of the ways a
        // metered round reports green having executed no wgmma instruction (`gpu_accel` names two
        // more; the provenance refusal below is a fourth, and is escalated for the same reason).
        //
        // **The escalation needs TWO levers here, and the second one is why this is not a copy of the
        // Hopper gate's single check.** That gate is a `::test --filter` line, and `::test` exports
        // `WUKONG_GPU_REQUIRE_CC` from the booked SKU, so `required_cc()` alone closes it. This bench
        // is an `#[ignore]`d `::bench --name` line, and **`bench()` exports no such variable** — only
        // `WUKONG_GPU_REQUIRED`. So on `::bench` the first lever is always `None` and, alone, would be
        // vacuous: the H100 round would print `[skip:capability]`, libtest would report `ok`, and
        // `bench()` would see rc 0.
        //
        // The second lever is `WUKONG_GPU_REQUIRED`, which `bench()` does set. It is the WRONG lever
        // for a correctness gate — `::test` sets it on every device run, including the routine
        // `WK_GPU=L4` rounds where skipping a Hopper-only gate is correct — but the reasoning that
        // makes it wrong there does not hold for THIS test: it is `#[ignore]`d, so it executes only
        // when `--ignored` was passed, and this crate has exactly one `#[ignore]`d test (the suite
        // reports `1 ignored`), so `--ignored` selects this round and nothing else. A Hopper-only
        // measurement running on a non-Hopper device is a mis-provisioned round whichever way it was
        // launched. The cost of the blunter lever is stated rather than hidden: a developer who
        // exports `WUKONG_GPU_REQUIRED=1` and sweeps `--ignored` on this repo's own 4050 gets a red
        // naming a container they never provisioned, and should drop the export or name a shape.
        //
        // `the_e2e_invocation_reaches_a_bench_that_can_fail` asserts that `bench()` still does not
        // export the capability, so if the harness ever grows that export this comment fails a test
        // instead of going stale, and the second lever can be narrowed back to the first.
        let cc = g.target().cc();
        if gpu_accel::gemm_route_for(cc, false) != GemmRoute::Wgmma {
            let declared_cc = gpu_accel::capability_skip_is_a_failure(cc, gpu_accel::required_cc());
            let declared_device = std::env::var_os("WUKONG_GPU_REQUIRED").is_some();
            assert!(
                !declared_cc && !declared_device,
                "{E2E_BENCH_NAME}: this round asked for a device it could measure on ({}={:?}, \
                 WUKONG_GPU_REQUIRED={}) but the container probed cc {}.{} ({}), which the wgmma \
                 route does NOT take. Skipping here would report a green PASS having executed no \
                 wgmma instruction — the container is not the part the round paid for.",
                gpu_accel::REQUIRED_CC_ENV,
                std::env::var(gpu_accel::REQUIRED_CC_ENV).unwrap_or_default(),
                declared_device,
                cc.0,
                cc.1,
                g.device_name()
            );
            eprintln!(
                "[skip:capability] {E2E_BENCH_NAME}: the wgmma route needs cc 9.x and this device \
                 is cc {}.{} ({}), where the plain-GEMM offload is correctly `Existing`. The round \
                 this bench exists for is:\n    {E2E_BENCH_INVOCATION}",
                cc.0,
                cc.1,
                g.device_name()
            );
            return;
        }

        // --- 1. the peer, before anything that costs device time -------------------------------
        if !wukong_codegen_gpu::baselines::peers_available(g) {
            wukong_codegen_gpu::diff::skip_or_fail(
                E2E_BENCH_NAME,
                "cuBLAS is not loadable, so this round has no bar",
            );
            eprintln!(
                "[skip] {E2E_BENCH_NAME}: no cuBLAS.\n{}",
                wukong_codegen_gpu::baselines::peer_env_hint()
            );
            return;
        }

        // --- 2. provenance ---------------------------------------------------------------------
        eprintln!("\n================ {E2E_BENCH_NAME} ================");
        eprintln!(
            "A REAL `.wk` PROGRAM THROUGH THE REAL PIPELINE. The published wgmma table comes from\n\
             `time_gemm_nt_wgmma`, which launches the generator's kernel directly. This round runs\n\
             the recognized program instead: lex -> parse -> sema -> mir_build -> opt -> interpreter\n\
             + GpuAccel -> gemm_nt_wgmma, and reports compile time, end-to-end time, direct-launch\n\
             time and kernel-only time as FOUR separate quantities.\n\
             The program is `tests/run/linear_nt_every_lane.wk`'s spelling at the grid's shapes."
        );
        let prov = bi::Provenance::new(
            E2E_BENCH_NAME,
            bi::facts_of(g),
            bi::RoundMeta::from_env(),
            bi::query_smi(),
        );
        eprint!("{}", prov.header());
        if !matches!(prov.meta.clock_lock, bi::ClockLock::Locked(_)) {
            eprintln!(
                "*** CLOCKS ARE NOT LOCKED (or the lock is undeclared), so per \
                 GPU_RETARGET_PLAN.md 6.3\n*** this round is ITERATION data, not publication data."
            );
        }
        let mut round = match bi::Round::open(prov) {
            Ok(r) => r,
            Err(why) => {
                // **Escalated, not merely printed.** `Round::open` refuses when the device does not
                // verify against `DEVICE_SPECS` — a MIG slice, a vGPU profile, an SM count off spec,
                // a part the table does not know. On a container that IS Hopper but does not verify,
                // a bare `return` here reports `ok`, hands `bench()` rc 0, and the detached round is
                // green having measured nothing: the same class as the capability skip above, past
                // the capability check, on metered silicon. `skip_or_fail` turns it into a failure
                // under `WUKONG_GPU_REQUIRED`, which `::bench` always sets.
                wukong_codegen_gpu::diff::skip_or_fail(
                    E2E_BENCH_NAME,
                    &format!("the round's provenance does not verify: {why}"),
                );
                eprintln!("[skip:provenance] {E2E_BENCH_NAME}: {why}");
                eprintln!("*** THIS ROUND PUBLISHES NOTHING ***");
                return;
            }
        };

        // --- 3. the plan, pre-registered -------------------------------------------------------
        let plan = bi::TwinPlan::default();
        eprintln!(
            "plan           : {} recorded rounds + {} discarded warm-up, bar +/-{:.2}%, median \
             across rounds",
            plan.rounds,
            plan.warmups,
            plan.bar * 100.0
        );
        match bi::machine_floor(g, &plan) {
            Ok(v) => eprint!("{}", v.table()),
            Err(e) => eprintln!("[warn] machine-floor probe unavailable: {e}"),
        }

        // --- 4. per shape: witness, correctness, then timing -----------------------------------
        // Measured before the device work, which makes it a LOWER BOUND on what arm B pays mid-round:
        // the per-shape host buffers (768 MiB at sq8192) do not exist yet, so the reservation is
        // cheaper here than it will be later. The RETENTION block says so where the number is used.
        let thread_us = interp_thread_overhead_us();
        let mut rows: Vec<(&GemmPoint, ShapeCosts, bi::BenchVerdict, bi::BenchVerdict)> =
            Vec::new();
        for p in WGMMA_BENCH_GRID {
            // `GemmPoint` names its dims `m`/`n`/`k`; every launcher below takes `(m, k, n)`.
            let (m, k, n) = (p.m, p.k, p.n);
            let cfg = gpu_accel::wgmma_cfg_for(
                gpu_accel::WGMMA_F32_SEAM_DTYPE,
                m,
                k,
                n,
                g.target().sm_count_usize(),
            )
            .unwrap_or_else(|e| {
                panic!(
                    "{}: the shipped rule DECLINES this grid shape ({e}), so arm B would run                      `gpu::gemm_nt` while arm A has no kernel. The grid and the rule have diverged.",
                    p.label
                )
            });
            let iters = bench_iters(p);
            eprintln!(
                "\n  ---- {} {} ----  arm B kernel = {}, {iters} launches/resident region",
                p.label,
                p.dims(),
                cfg.name
            );

            // The seam's own decline set, restated here so a grid row it rejects cannot be timed as
            // though the wgmma family had taken it. `every_bench_grid_shape_reaches_the_wgmma_route`
            // asserts this never fires for the shipped grid, on a laptop; this is the run-time half,
            // for a grid or a shared-memory budget that has since moved.
            if let Some(why) = gpu_accel::wgmma_declines(cfg, m, k, n, g.smem_budget()) {
                panic!(
                    "{}: the offload seam DECLINES this grid shape ({why}), so arm B would run \
                     `gpu::gemm_nt` while arm A runs the wgmma kernel — a ratio between two \
                     different kernels. The grid and the seam have diverged.",
                    p.label
                );
            }

            // --- 4a. compile, timed, device-free (T4) -----------------------------------------
            let src = linear_tensor_src(m, k, n);
            let mut compile_ms = f64::INFINITY;
            let mut built = None;
            for _ in 0..COMPILE_REPS {
                let (program, interner, dt) = compile_through_pipeline(&src);
                compile_ms = compile_ms.min(dt.as_secs_f64() * 1e3);
                built = Some((program, interner));
            }
            let (program, mut interner) = built.expect("COMPILE_REPS >= 1");
            let entry = interner.intern(E2E_ENTRY);
            assert!(
                mir_text(&program, &interner).contains(NT_GEMM_SYMBOL),
                "{}: the compiled MIR carries no `{NT_GEMM_SYMBOL}` call, so the offload seam is \
                 not on this program's path and arm B would time the interpreter's scalar nest",
                p.label
            );

            // --- 4b. the route witness and correctness, before a single timed launch ----------
            let (mut x, mut w, mut y) = (fill_x(m, k), fill_w(n, k), vec![0f32; m * n]);
            {
                let mut accel = GpuAccel::new(&mut *g);
                {
                    let mut bufs: [&mut [f32]; 3] = [&mut x, &mut w, &mut y];
                    wukong_interp::run_kernel_f32_accel(
                        &program, entry, &mut bufs, &interner, &mut accel,
                    )
                    .unwrap_or_else(|e| panic!("{}: the offload run failed: {e}", p.label));
                }
                assert!(
                    accel.calls >= 1,
                    "{}: nothing was offloaded, so arm B would time a CPU GEMM at a bench shape",
                    p.label
                );
                assert_eq!(
                    (accel.gemm_wgmma_calls, accel.gemm_existing_calls),
                    (accel.calls, 0),
                    "{}: {} of {} plain-GEMM offloads fell back to `gpu::gemm_nt`, so the \
                     retention ratio this round prints would compare two different kernels",
                    p.label,
                    accel.gemm_existing_calls,
                    accel.calls
                );
            }
            assert_every_lane(&format!("{} offload route", p.label), &y, m, k, n);

            // The direct-launch arm, checked against the same reference: a wrong kernel there would
            // hand the retention ratio a denominator it did not earn.
            let direct = wukong_codegen_gpu::gpu::gemm_nt_wgmma(g, cfg, &x, &w, m, k, n)
                .unwrap_or_else(|e| panic!("{}: direct `gemm_nt_wgmma` failed: {e}", p.label));
            assert_every_lane(&format!("{} direct launch", p.label), &direct, m, k, n);
            // The peer, host-to-host, ONCE. This is a correctness check on the bar — a transpose slip
            // in cuBLAS's arguments would hand every shape a win or a loss it did not earn — and its
            // wall time is recorded only as a DIAGNOSTIC. It is not an arm: the function constructs a
            // `CudaBlas` (a workspace allocation) on every call, which the crate's own *timing* entry
            // hoists out of its loop precisely because it is not free. Timing it as arm A would charge
            // the peer a per-call cost Wukong's arm does not pay, biasing the ratio in OUR favour.
            let peer_t0 = Instant::now();
            let peer_once =
                wukong_codegen_gpu::baselines::cublas_gemm_nt_f16_f32out(g, &x, &w, m, k, n)
                    .unwrap_or_else(|e| panic!("{}: cuBLAS one-shot failed: {e}", p.label));
            let peer_h2h_once_ms = peer_t0.elapsed().as_secs_f64() * 1e3;
            assert_every_lane(&format!("{} cuBLAS peer", p.label), &peer_once, m, k, n);
            eprintln!(
                "  [gate] route = Wgmma, and all three paths returned K*(1+i%{X_PERIOD})*(1+j%\
                 {W_PERIOD}) exactly on every one of {} lanes \u{2713}",
                m * n
            );

            // **The TIMED launcher must also write C, and this is the last check before either TWIN.**
            // (Two timed quantities do precede it — the compile at 4a and the peer one-shot just
            // above — but neither comes from `time_gemm_nt_wgmma` and neither is published; both land
            // in the DIAGNOSTIC block. What this probe guards is the resident twin.)
            // `time_gemm_nt_wgmma` never reads C back. Its own doc says why that matters: "a kernel
            // that stored nothing -- a mispredicated epilogue, a truncated grid, a transport that lost
            // its operands -- would post the fastest number in the table and pass every gate in the
            // round, because the pre-timing arms run through a *different* entry point
            // (`gemm_nt_wgmma`) with its own buffers." Every lane check above goes through exactly
            // that different entry point, so publishing the resident ratio (which this round does)
            // reopens the seam unless the TIMING launcher is itself made to read C back.
            // `wukong_codegen_gpu`'s own sweep is required by a device-free law to run this probe
            // before it times anything; this is that requirement, honoured at the shape being
            // published rather than at a guard shape, and placed here so it precedes BOTH twins rather
            // than landing as an unshared burst between them.
            let region = wukong_codegen_gpu::gpu::TimedRegion::closed_form_probe();
            const PROBE_ITERS: usize = 4;
            match wukong_codegen_gpu::gpu::time_gemm_nt_wgmma_in(g, cfg, m, k, n, PROBE_ITERS, region)
            {
                // The sentinel half of the launcher's readback is deliberately NOT claimed here:
                // `closed_form_probe()` sets `zero_c_in_region`, so the memset destroys
                // `WGMMA_C_SENTINEL` before the first launch and the launcher's `unwritten` count is
                // structurally zero. What survives is the arm that matters — `K*fill^2` is never 0,
                // so a lane the kernel never wrote reads 0.0 after the memset and is caught.
                Ok(secs) => eprintln!(
                    "  [gate] the TIMED launcher writes C at {} ({}, {PROBE_ITERS} launches): every \
                     lane is exactly K*fill^2 = {:e} ({:.3} ms/launch, NOT a measurement) \u{2713}",
                    p.dims(),
                    region.label(),
                    k as f32 * region.fill * region.fill,
                    secs * 1e3
                ),
                Err(e) => panic!(
                    "{}: the timed path's closed-form probe failed: {e}. The resident column is \
                     published from this launcher, so without the readback a kernel that stored \
                     nothing would post the fastest number in the table.",
                    p.label
                ),
            }

            // --- 4c. twin 1: the pipeline against the DIRECT LAUNCHER ------------------------
            // A and C are the same host-to-host launcher call, so C/A is this shape's own floor and
            // B/A is exactly what the pipeline adds: one interpreter call, the seam's dispatch, and
            // the `copy_from_slice` that moves the returned `Vec` into the user's buffer.
            let overhead = bi::run_rotated(
                &format!("{E2E_BENCH_NAME}@{}/pipeline_vs_launcher", p.label),
                &plan,
                |arm| match arm {
                    bi::Arm::Baseline | bi::Arm::Control => {
                        let t0 = Instant::now();
                        let out = wukong_codegen_gpu::gpu::gemm_nt_wgmma(g, cfg, &x, &w, m, k, n);
                        let dt = t0.elapsed().as_secs_f64() * 1e3;
                        vec![bi::read(H2H_FIELD, if out.is_ok() { dt } else { f64::NAN })]
                    }
                    bi::Arm::Contender => {
                        let mut accel = GpuAccel::new(&mut *g);
                        let t0 = Instant::now();
                        let r = {
                            let mut bufs: [&mut [f32]; 3] = [&mut x, &mut w, &mut y];
                            wukong_interp::run_kernel_f32_accel(
                                &program, entry, &mut bufs, &interner, &mut accel,
                            )
                        };
                        let dt = t0.elapsed().as_secs_f64() * 1e3;
                        vec![bi::read(H2H_FIELD, if r.is_ok() { dt } else { f64::NAN })]
                    }
                },
            );

            // --- 4d. twin 2: the RESIDENT pair — the published table's own comparison --------
            // Both arms are `iters` device-resident launches with their handles and modules resolved
            // outside the timed region, which is what makes them comparable to each other and to the
            // shipped wgmma table. This goes through the twin rather than two lone `time_*` calls
            // because the root CLAUDE.md is explicit that "a lone timing is not a measurement", and
            // because it is the campaign's headline quantity: it must be able to refuse.
            let resident = bi::run_rotated(
                &format!("{E2E_BENCH_NAME}@{}/resident_vs_cublas", p.label),
                &plan,
                |arm| match arm {
                    bi::Arm::Baseline | bi::Arm::Control => {
                        let s = wukong_codegen_gpu::baselines::time_cublas_gemm_nt_f16_f32out(
                            g,
                            m,
                            k,
                            n,
                            iters as u32,
                        );
                        vec![bi::read(
                            RESIDENT_FIELD,
                            s.map(|v| v * 1e3).unwrap_or(f64::NAN),
                        )]
                    }
                    bi::Arm::Contender => {
                        let s = wukong_codegen_gpu::gpu::time_gemm_nt_wgmma(g, cfg, m, k, n, iters);
                        vec![bi::read(
                            RESIDENT_FIELD,
                            s.map(|v| v * 1e3).unwrap_or(f64::NAN),
                        )]
                    }
                },
            );

            rows.push((
                p,
                ShapeCosts {
                    compile_ms,
                    peer_h2h_once_ms,
                },
                bi::analyze(&overhead, plan.bar),
                bi::analyze(&resident, plan.bar),
            ));
        }

        // --- 5. close, then let the gate decide ------------------------------------------------
        round.close(bi::query_smi());
        eprintln!("\n---- {E2E_BENCH_NAME}: round close ----");
        eprint!("{}", round.header());

        eprintln!("\n---- {E2E_BENCH_NAME}: per-shape verdicts ----");
        for (p, _, ov, res) in &rows {
            eprintln!(
                "  == {} : pipeline vs direct launcher (host-to-host) ==",
                p.label
            );
            eprint!("{}", ov.table());
            eprintln!(
                "  == {} : wgmma kernel vs cuBLAS (both resident) ==",
                p.label
            );
            eprint!("{}", res.table());
        }

        eprintln!(
            "\n---- {E2E_BENCH_NAME}: THE PUBLISHED TABLE ----\n\
             Both columns come out of `Round::publish`, so a shape whose control floor did not clear\n\
             the bar prints its refusal instead of a number.\n\
             `pipeline cost` is what the INTERPRETER and the OFFLOAD SEAM add over calling the\n\
             launcher directly -- same function in both arms, and the compile is NOT in either of\n\
             them (it runs once per shape outside the twin; see the DIAGNOSTIC block). A tie means the\n\
             route is free. Part of it is the interpreter's per-call thread; the RETENTION table below\n\
             says how to net that out in ms.\n\
             `kernel/cuBLAS` is the resident-buffer pair: the campaign's own comparison, re-measured\n\
             here in the same round, same clock, same container, same iteration counts, and its\n\
             contender's timed region is `{}`.\n",
            wukong_codegen_gpu::gpu::TimedRegion::for_cfg(gpu_accel::wgmma_cfg_for(
                gpu_accel::WGMMA_F32_SEAM_DTYPE,
                WGMMA_BENCH_GRID[0].m,
                WGMMA_BENCH_GRID[0].n
            ))
            .label()
        );
        eprintln!(
            "  {:<15} {:>18} {:>30}  kernel/cuBLAS, resident",
            "shape", "MxNxK", "pipeline cost vs launcher"
        );
        // **A round that completes and publishes NOTHING must not exit 0.** Every refusal below
        // formats as `-- (reason)` text, so a clock-drift refusal, an unresolved control floor or a
        // `Malformed` field from a launcher error would print fourteen dashes, hand `bench()` rc 0 and
        // produce a green detached round — the same defect as the provenance skip, one layer later.
        let mut published = 0usize;
        for (p, _, ov, res) in &rows {
            let cost = match round.publish(ov, H2H_FIELD) {
                Ok(pd) => {
                    published += 1;
                    format!(
                        "{:+.1}% ({}, floor +/-{:.2}%, ours {})",
                        (pd.ratio() - 1.0) * 100.0,
                        if pd.is_tie() { "tie" } else { "moved" },
                        pd.floor() * 100.0,
                        // The CONTENDER's own spread (guard G16). Arm B here is the interpreter, the least
                        // stable arm in this file, so dropping it would hide the one dispersion a reader
                        // most needs — `analyze` already refuses above the bar, but the value is evidence.
                        match pd.dispersion() {
                            Some(d) => format!("+/-{:.2}%", d * 100.0),
                            None => "n/a".to_string(),
                        }
                    )
                }
                Err(why) => format!("-- ({why})"),
            };
            let head = match round.publish(res, RESIDENT_FIELD) {
                // `ratio` is B/A on a MILLISECOND field: below 1.0 means we are faster, and the
                // throughput percentage is its reciprocal. `bench_instrument` is unit-agnostic and
                // cannot catch a ratio quoted the wrong way round, so polarity is owned here.
                Ok(pd) => {
                    published += 1;
                    format!(
                        "{:.1}% of cuBLAS ({:.4}x its ms, {}, floor +/-{:.2}%)",
                        100.0 / pd.ratio(),
                        pd.ratio(),
                        if pd.is_tie() { "tie" } else { "moved" },
                        pd.floor() * 100.0
                    )
                }
                Err(why) => format!("-- ({why})"),
            };
            eprintln!("  {:<15} {:>18} {:>30}  {head}", p.label, p.dims(), cost);
        }

        eprintln!(
            "\n---- {E2E_BENCH_NAME}: RETENTION ----\n\
             `retention` = kernel ms / e2e ms: the fraction of the KERNEL-ONLY number that survives\n\
             the trip through the INTERPRETER and the OFFLOAD SEAM -- the compile is in neither\n\
             quantity, it runs once per shape outside both twins. It is below 1 by the host f32->f16\n\
             conversion and the two PCIe crossings that the RESIDENT-buffer bench does not pay --\n\
             which is why `launch ms` sits beside it: kernel/launch is that tax alone, and\n\
             launch/e2e is the pipeline's own share.\n\
             **The four ms COLUMNS are taken from PUBLISHED twins** (`Published::baseline` and\n\
             `Published::contender`), never from a raw verdict field, so a row whose twin refused\n\
             shows the refusal rather than a number the gate would not have allowed. `retention`\n\
             itself is their QUOTIENT ACROSS TWO TWINS and therefore carries no floor of its own --\n\
             its two operands were each gated, against different denominators.\n\
             `e2e ms` includes one 512 MiB-stack thread spawn per call that a real program pays once\n\
             per run and `launch ms` never pays. So the thread-free end-to-end cost is\n\
             `e2e ms - interp thread us/1000`, and the SEAM's own addition is that minus `launch ms`.\n\
             `interp thread us` is measured before the per-shape host buffers exist, so it is a lower\n\
             bound on what arm B pays mid-round.\n"
        );
        eprintln!(
            "  {:<15} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "shape", "e2e ms", "launch ms", "kernel ms", "peer ms", "retention"
        );
        for (p, _, ov, res) in &rows {
            let h2h = round.publish(ov, H2H_FIELD);
            let resident = round.publish(res, RESIDENT_FIELD);
            match (&h2h, &resident) {
                (Ok(a), Ok(b)) => eprintln!(
                    "  {:<15} {:>11.4} {:>11.4} {:>11.4} {:>11.4} {:>10.1}%",
                    p.label,
                    a.contender(),
                    a.baseline(),
                    b.contender(),
                    b.baseline(),
                    100.0 * b.contender() / a.contender()
                ),
                (Err(why), _) | (_, Err(why)) => {
                    eprintln!("  {:<15} -- ({why})", p.label)
                }
            }
        }

        let ms = |v: f64| {
            if v.is_finite() && v > 0.0 {
                format!("{v:.4}")
            } else {
                "n/a".to_string()
            }
        };
        eprintln!(
            "\n---- {E2E_BENCH_NAME}: DIAGNOSTIC (NOT publishable) ----\n\
             `compile ms` is a ONE-TIME host cost, is not part of any ratio above, and is the\n\
             `lex..optimize` front subtotal with no backend -- see COMPILE_REPS on why it is not\n\
             comparable to docs/compile-floor.md's `lex..emit_object` headline.\n\
             `peer h2h once` is a LONE TIMING of one `cublas_gemm_nt_f16_f32out` call. It INCLUDES a\n\
             `CudaBlas` construction (a workspace allocation) that the crate's own timing entry hoists\n\
             out of its loop, AND it is the first cublasGemmEx at this shape in the process -- the peer\n\
             probe only ever built a handle -- so it also pays cuBLAS's lazy kernel load and heuristic\n\
             selection. It is here because the honest end-to-end peer does not yet exist; it is not an\n\
             arm and must not be quoted as a ratio.\n\
             `interp thread us` is the 512 MiB-stack thread `run_kernel_f32_accel` spawns per call,\n\
             which arm B pays every iteration and a real program pays once per run.\n"
        );
        eprintln!(
            "  interp thread us : {} (best of many; arm B pays it per call, a real run pays it once)",
            ms(thread_us)
        );
        eprintln!(
            "  {:<15} {:>12} {:>16}",
            "shape", "compile ms", "peer h2h once ms"
        );
        for (p, c, _, _) in &rows {
            eprintln!(
                "  {:<15} {:>12} {:>16}",
                p.label,
                ms(c.compile_ms),
                ms(c.peer_h2h_once_ms)
            );
        }
        // **The round's product is numbers, so producing none is a failure, not a result.** Both
        // tables above turn every refusal into `-- (reason)` text and keep going, which is right for a
        // per-shape refusal — a wide floor at one shape is a finding, and the log is worth its rented
        // minutes. But if NOTHING cleared, the round measured nothing publishable: a clock-drift
        // refusal, an unverified device, or a launcher error that poisoned every field would each
        // print dashes across all fourteen cells and still hand `bench()` rc 0. Escalated through the
        // same helper as every other skip, so it is loud locally and RED under `WUKONG_GPU_REQUIRED`,
        // which `::bench` always sets.
        eprintln!(
            "\n  published cells: {published} of {} (2 per shape)",
            2 * rows.len()
        );
        if published == 0 {
            wukong_codegen_gpu::diff::skip_or_fail(
                E2E_BENCH_NAME,
                "the round completed but every cell refused, so it published NOTHING — see the \
                 per-shape verdicts above for which gate shut",
            );
            eprintln!("*** THIS ROUND PUBLISHED NOTHING ***");
        }
        eprintln!("=============================================================\n");
    }

    // ------------------------------------------------------------------------------------------
    // Device-free laws. Every one runs on this laptop under
    // `cargo test -p wukong_driver --features gpu --lib`, and each pins a way the round above could
    // cost rented minutes and prove nothing.
    // ------------------------------------------------------------------------------------------

    /// The text from `decl` to the first `end` marker after it, so a `contains` check about one
    /// function cannot be satisfied by a neighbour's prose.
    ///
    /// Deliberately a local copy of `gpu_accel::tests::body_of` rather than a shared helper: that one
    /// lives inside a `#[cfg(test)] mod tests`, which a sibling module cannot reach, and widening a
    /// test module's visibility to share six lines is a worse trade than restating them.
    fn body_of<'a>(src: &'a str, decl: &str, end: &str) -> &'a str {
        let at = src
            .find(decl)
            .unwrap_or_else(|| panic!("`{decl}` is not in the scanned source"));
        let rest = &src[at..];
        let e = rest.find(end).map(|x| x + end.len()).unwrap_or(rest.len());
        &rest[..e]
    }

    /// Where a Python `@app.function`-decorated function ends.
    const PY_FN_END: &str = "\n@app.function";

    /// The `#[...]` attribute lines immediately above `fn {name}(`, and nothing else.
    ///
    /// **A byte window is not good enough here, and that is a measured fact rather than a worry.** The
    /// first version of this law took `&me[at-160..at]` and asserted it contained `#[test]`; because
    /// this file's own doc comments quote the string `` `#[test]` `` while explaining why the attribute
    /// is load-bearing, deleting the real attribute merely slid the window back onto the prose and the
    /// law stayed green — with the round no longer a test at all, which is the first failure its own
    /// docstring lists. Walking whole lines backwards and stopping at the first non-attribute line
    /// cannot be satisfied by prose, because a doc line starts with `///`.
    fn attributes_above(src: &str, name: &str) -> Vec<String> {
        let at = src
            .find(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("`fn {name}(` is not in the scanned source"));
        let mut attrs: Vec<String> = Vec::new();
        for line in src[..at].lines().rev() {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            if t.starts_with("#[") {
                // **Split the line, do not push it whole.** One line pushed as one element makes
                // `#[test] #[should_panic]` a single string that `starts_with("#[test]")` accepts, so
                // the exhaustiveness check below would pass an added attribute on the same line.
                // `rustfmt` would split that line and `cargo fmt --check` would catch it, but a law
                // whose message claims to reject extra attributes should not be relying on a
                // different gate to do it.
                attrs.extend(
                    t.split_whitespace()
                        .filter(|s| s.starts_with("#["))
                        .map(str::to_string),
                );
                continue;
            }
            break;
        }
        attrs
    }

    /// **The generated program really is recognized — proved without a device, for free.**
    ///
    /// This is the assertion that makes the whole round meaningful, and it is GATE-BLIND to every
    /// other test in the workspace: recognition happens in `mir_build`, so interp and native agree at
    /// every `-O` level whether or not the recognizer fired, and no differential or invariance gate
    /// can see it. If the recognizer's reach ever narrows past this spelling, the round above would
    /// still run — timing the interpreter walking a scalar triple loop at 8192 cubed — and its `e2e`
    /// column would be a number about the interpreter, published under the GPU's name.
    #[test]
    fn the_bench_program_is_recognized_without_a_device() {
        for p in WGMMA_BENCH_GRID {
            let (program, interner, _) =
                compile_through_pipeline(&linear_tensor_src(p.m, p.k, p.n));
            let mir = mir_text(&program, &interner);
            // The exact call, not the prefix: `wukong_sgemm_nt_epi` and `_alpha` both contain the
            // bare symbol and neither reaches wgmma. See NT_GEMM_SYMBOL.
            assert!(
                mir.contains(NT_GEMM_SYMBOL),
                "{}: the idiomatic `Y = X.W^T` at {} does not lower to `{NT_GEMM_SYMBOL}`, so the \
                 offload seam is not on its path and the end-to-end round would time the \
                 interpreter's scalar loop nest instead of the GPU",
                p.label,
                p.dims()
            );
        }
    }

    /// **The timed program and the lane-by-lane-gated fixture are the same program.**
    ///
    /// `tests/run/linear_nt_every_lane.wk` is the only thing that checks this spelling's *answers*,
    /// one lane at a time, on both CPU backends. The bench checks its *speed*. If the two texts drift
    /// — a flat index here, a multi-index there — then nothing gates the shape that gets timed, and
    /// the round becomes a benchmark tuned to whatever the recognizer happens to accept, which
    /// `GPU_RETARGET_PLAN.md` names as a number that must never be reported as a win.
    #[test]
    fn the_bench_program_is_the_fixtures_program() {
        let fixture = include_str!("../../../tests/run/linear_nt_every_lane.wk");
        // The fixture's kernel at its own dimensions, generated by the bench's own generator.
        let generated = linear_tensor_src(3, 8, 6);
        // CRLF is normalized BEFORE the body is cut out, not after: `split_whitespace` would absorb
        // a stray `\r` happily, but the `\n}\n` end marker would miss `\r\n}\r\n` entirely and this
        // law would then compare the fixture's whole file — including `main` — against one function
        // and fail for a reason that has nothing to do with the programs. An editor that rewrites
        // the fixture with Windows endings is a likely event, not a hypothetical.
        let body = |src: &str| {
            body_of(&src.replace("\r\n", "\n"), "fn linear(", "\n}\n")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        };
        assert_eq!(
            body(&generated),
            body(fixture),
            "the bench's generated kernel and `tests/run/linear_nt_every_lane.wk`'s kernel are no \
             longer the same program (whitespace-insensitive). Either the fixture stopped covering \
             what the bench times, or the bench drifted into a spelling nothing gates."
        );
    }

    /// **Every shape the round will time is one the offload seam ACCEPTS.**
    ///
    /// Pure functions of the shape, so the H100's answer is decidable here. A grid row the seam
    /// declines is the worst outcome this round has: arm B silently runs `gpu::gemm_nt` (the seam's
    /// documented, correct fallback), arm A runs the wgmma kernel, and their ratio gets printed as
    /// "retention" while comparing two different kernels. The run-time half is a `panic!` inside the
    /// round; this is the half that fires before an hour is rented.
    ///
    /// It also records the T1 determination as a law: the `M % 64 / N % 64 / K % 16` literals a
    /// census attributed to the plain-GEMM route live in `sgemm_nt_epi` — the *fused-epilogue* hook,
    /// which routes to the Act-1 wmma kernels and never to wgmma. The plain route's real rules are an
    /// even `N`, `K % 8 == 0`, `M*N` inside a `u32`, the shared-memory budget and the grid ceiling,
    /// and all seven grid rows clear every one of them.
    #[test]
    fn every_bench_grid_shape_reaches_the_wgmma_route() {
        /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` as the H100 probes it — the number
        /// the round logs print as `opt-in SMEM: 232448 B`.
        const H100_SMEM: usize = 232_448;
        assert_eq!(
            gpu_accel::gemm_route_for((9, 0), false),
            GemmRoute::Wgmma,
            "the device this round books is not eligible for the route it measures"
        );
        for p in WGMMA_BENCH_GRID {
            let (m, k, n) = (p.m, p.k, p.n);
            let cfg =
                gpu_accel::wgmma_cfg_for(gpu_accel::WGMMA_F32_SEAM_DTYPE, m, k, n, HOPPER_SM_COUNT)
                    .unwrap_or_else(|e| {
                        panic!(
                            "{} {} would be DECLINED by the shipped rule on an H100: {e}",
                            p.label,
                            p.dims()
                        )
                    });
            let why = gpu_accel::wgmma_declines(cfg, m, k, n, H100_SMEM);
            assert!(
                why.is_none(),
                "{} {} would be DECLINED by the offload seam on an H100: {}",
                p.label,
                p.dims(),
                why.unwrap_or_default()
            );
            // The rules that actually bind, asserted by name so a future grid row cannot creep past
            // them silently.
            assert!(
                n.is_multiple_of(2),
                "{}: the v2 epilogue needs an even N",
                p.label
            );
            assert!(
                k.is_multiple_of(8),
                "{}: a K-major NT operand's row stride is K*2 bytes and a tensor map needs a \
                 multiple of 16",
                p.label
            );
            assert!(
                (m as u64) * (n as u64) <= u32::MAX as u64,
                "{}: M*N overflows the epilogue's u32 element index",
                p.label
            );
        }
    }

    /// **The published table prints ONE timed-region label for all seven rows, so all seven must
    /// agree.**
    ///
    /// The header derives it from the first grid row's config
    /// (`TimedRegion::for_cfg(wgmma_cfg_for(.., GRID[0].m, GRID[0].n))`), and `wgmma_cfg_for` selects
    /// on output size — the clustered row at or above `W1_CLUSTER_MIN_OUTPUT_ELEMS`, the un-clustered
    /// one below — so the grid really does select more than one config. Today every config agrees
    /// because `WgmmaCfg::requires_zeroed_c()` is `const fn ... { false }` for all of them, which makes
    /// the header's claim TRUE BY ACCIDENT rather than by construction.
    ///
    /// That is the kind of fact that rots silently: a future row that needs a zeroed C would make the
    /// header say `memset-in-region no` over rows that had one, and `TimedRegion::label` exists
    /// precisely because "the timed region contained a memset" is not recoverable from a number. If
    /// this fails, do not widen it — move the label into the per-shape row.
    #[test]
    fn every_grid_shape_shares_the_published_timed_region() {
        let header = wukong_codegen_gpu::gpu::TimedRegion::for_cfg(gpu_accel::wgmma_cfg_for(
            gpu_accel::WGMMA_F32_SEAM_DTYPE,
            WGMMA_BENCH_GRID[0].m,
            WGMMA_BENCH_GRID[0].n,
        ))
        .label();
        for p in WGMMA_BENCH_GRID {
            let cfg = gpu_accel::wgmma_cfg_for(gpu_accel::WGMMA_F32_SEAM_DTYPE, p.m, p.n);
            let mine = wukong_codegen_gpu::gpu::TimedRegion::for_cfg(cfg).label();
            assert_eq!(
                mine, header,
                "{} selects `{}`, whose timed region is `{mine}` — the published table's header \
                 announces `{header}` for every row. Move the label into the per-shape row rather \
                 than relaxing this.",
                p.label, cfg.name
            );
        }
    }

    /// **The fill really does make correctness an `==` check at every grid shape.**
    ///
    /// [`assert_every_lane`] rests on arithmetic that is shape-dependent, so it is checked per shape
    /// rather than asserted in prose: every operand value `1..=max(X_PERIOD, W_PERIOD)` survives the
    /// seam's host `f32 -> f16` conversion exactly and stays NORMAL there (a subnormal would trip
    /// `wgmma_declines_operand_range` and send every shape to `gpu::gemm_nt`), and the largest product
    /// `K * X_PERIOD * W_PERIOD` is inside f32's exact-integer range.
    ///
    /// **The bound is the PRODUCT, not `K`.** An earlier version of this law checked `K < 2^24`, which
    /// was the right bound for the all-ones fill it was written for and is 15x too loose for this one:
    /// it would admit a grid row with `K` up to 16_777_215 while `assert_every_lane` needs
    /// `K * 15 < 2^24`, i.e. `K <= 1_118_481`. A row in that gap makes the lane check fail on a
    /// *correct* kernel, and the natural repair — loosening it to a tolerance — would delete the only
    /// thing standing between this round and publishing a fast wrong answer.
    #[test]
    fn the_fill_is_exact_at_every_grid_shape() {
        for v in 1..=X_PERIOD.max(W_PERIOD) {
            let h = half::f16::from_f32(v as f32);
            assert_eq!(
                h.to_f32(),
                v as f32,
                "operand value {v} does not survive the seam's f16 conversion exactly"
            );
            assert!(
                h.is_normal(),
                "operand value {v} is not a NORMAL f16, so `wgmma_declines_operand_range` would \
                 decline the bench data and every shape would fall back to `gpu::gemm_nt`"
            );
        }
        let span = (X_PERIOD as u64) * (W_PERIOD as u64);
        for p in WGMMA_BENCH_GRID {
            let want_max = (p.k as u64) * span;
            assert!(
                want_max < (1u64 << 24),
                "{}: the largest expected lane is K*{X_PERIOD}*{W_PERIOD} = {want_max}, past f32's \
                 exact-integer range, so `assert_every_lane` would stop being an `==` check",
                p.label
            );
            assert_eq!(
                (want_max as f32) as u64,
                want_max,
                "{}: the largest expected lane is not exactly representable in f32",
                p.label
            );
        }
    }

    /// **The invocation reaches a bench that can fail, and selects it in the package it lives in.**
    ///
    /// Three independent ways this round could bill an H100 and prove nothing, each of which has
    /// happened in this repo or was one flag away from it:
    ///
    /// - **the wrong entrypoint.** `::bench` appends `--ignored` on its own, so it reaches
    ///   `#[ignore]`d tests; `::test` does not pass that flag, so a bare `--filter <name>` there
    ///   selects zero tests, prints `0 passed` and exits 0 — the 2026-08-11 vacuous round, $0.006.
    ///   (`::test` splits its filter, so `--filter "--ignored <name>"` *could* reach one; that is a
    ///   correctness entrypoint, not a measurement one.)
    /// - **the wrong package.** `bench()`'s `package` defaults to `wukong_codegen_gpu`, and this
    ///   bench is in `wukong_driver`. Without `--package wukong_driver` the harness selects zero
    ///   tests in a crate that does not contain it.
    /// - **an entrypoint that cannot fail.** `bench()` once ran its child `check=False` and returned
    ///   with no `sys.exit`, so a failing assertion produced a green `modal run --detach`.
    #[test]
    fn the_e2e_invocation_reaches_a_bench_that_can_fail() {
        let inv = E2E_BENCH_INVOCATION;
        assert!(
            inv.is_ascii() && inv.lines().count() == 1,
            "the invocation is meant to be pasted into a shell: {inv}"
        );
        assert!(
            inv.contains("modal_app.py::bench"),
            "an `#[ignore]`d round is only reachable from ::bench, which appends --ignored: {inv}"
        );
        assert!(
            !inv.contains("::test") && !inv.contains("--filter "),
            "`--filter` is ::test's selector and ::test does not pass --ignored itself, so a bare \
             `--filter <name>` selects ZERO tests here and exits 0. (`::test` splits its filter, so \
             `--filter \"--ignored <name>\"` could reach an ignored test; a sweep runner is still the \
             right home for a measurement.): {inv}"
        );
        assert!(
            inv.contains("--package wukong_driver"),
            "`bench()`'s package defaults to wukong_codegen_gpu, which does not contain this bench \
             — without `--package wukong_driver` the round selects zero tests: {inv}"
        );
        // `--peers` is belt-and-braces here, and saying so is the point: the cuBLAS branch in this
        // round calls `diff::skip_or_fail`, which escalates on `WUKONG_GPU_REQUIRED` — a variable
        // `bench()` sets unconditionally — so a missing peer already fails loudly without it.
        // `--peers` additionally sets `WUKONG_PEER_REQUIRED`, which the peers *inside*
        // `wukong_codegen_gpu` read. Keeping it costs nothing and keeps this round's provisioning
        // identical to every other published GEMM round; claiming it is what makes the round fail
        // would be false.
        assert!(
            inv.contains("--peers"),
            "keep --peers so this round's peer provisioning matches every other published GEMM \
             round (the loud failure itself comes from WUKONG_GPU_REQUIRED, which ::bench always \
             sets): {inv}"
        );
        // The selector must name a test that exists here, IS `#[ignore]`d, and is the round.
        let name = inv
            .split("--name ")
            .nth(1)
            .expect("::bench selects with --name")
            .split_whitespace()
            .next()
            .expect("--name carries no test name");
        assert_eq!(
            name, E2E_BENCH_NAME,
            "the invocation and the bench name have drifted apart"
        );
        let me = include_str!("gpu_e2e_bench.rs");
        let decl = format!("fn {name}()");
        assert!(
            me.contains(&decl),
            "the invocation names `{name}`, which is not a test in this file"
        );
        // The attribute LINES immediately above the declaration — see `attributes_above` for why a
        // byte window was not enough.
        let attrs = attributes_above(me, name);
        assert!(
            attrs.iter().any(|a| a.starts_with("#[test]")),
            "`{name}` is named by an invocation but is not a #[test]: {attrs:?}"
        );
        assert!(
            attrs.iter().any(|a| a.starts_with("#[ignore")),
            "`{name}` must be #[ignore]d: ::bench appends --ignored and would otherwise select zero \
             tests and exit 0: {attrs:?}"
        );
        // **The set must be EXACTLY those two, because an ADDED attribute is the dangerous one.**
        // Two mutations pass both `any` checks above while destroying the property outright:
        // `#[cfg(any())]` compiles the round out entirely, so `::bench --name ...` selects zero tests
        // and exits 0 — the 2026-08-11 vacuous round reached a fourth way — and `#[should_panic]`
        // inverts every assertion in it, turning the route witness, the lane check and the capability
        // escalation from guards into pass conditions. Neither is visible to an existential check.
        assert!(
            attrs
                .iter()
                .all(|a| a.starts_with("#[test]") || a.starts_with("#[ignore")),
            "`{name}` carries an attribute beyond #[test] and #[ignore]. `#[cfg(..)]` can compile the \
             round out (zero tests selected, exit 0) and `#[should_panic]` inverts every assertion in \
             it; both pass an existential check. Saw: {attrs:?}"
        );
        // The round must be able to fail on the things the instrument cannot see. Scoped to the
        // round's own body, which ends at its own closing brace alone on a 4-space-indented line —
        // not at the next `#[test]`, which overshot by the whole `body_of` helper and let a
        // neighbour's prose satisfy these checks. **Comments are stripped**, because the round's own
        // prose names most of these symbols while explaining why they matter: without the strip,
        // deleting `assert_every_lane(..)` and leaving the comment above it green-lights the law, which
        // is precisely the class of defect the byte-window version of `attributes_above` had.
        let round_src = body_of(me, &decl, "\n    }\n");
        let round: String = round_src
            .lines()
            .map(|l| match l.find("//") {
                Some(at) => &l[..at],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            round.contains("(accel.gemm_wgmma_calls, accel.gemm_existing_calls),"),
            "`{name}` no longer compares the per-route counters, so it cannot tell a wgmma launch \
             from a fallback and would publish a ratio between two different kernels"
        );
        assert!(
            round.contains("assert_every_lane("),
            "`{name}` no longer checks every output lane, so a fast WRONG answer could publish"
        );
        assert!(
            round.contains("round.publish("),
            "`{name}` no longer publishes through the instrument's gate, so a round whose control \
             floor never cleared the bar would print a number anyway"
        );
        // **The closed-form probe must precede BOTH twins.** `gpu.rs` pins exactly this ordering for
        // its own sweep ("afterwards it is a post-mortem") but its law scans only that crate's
        // sources, so nothing reached this file: deleting the probe, or sliding it below the resident
        // twin, would leave every gate in the tree green while the published `kernel/cuBLAS` column
        // came from a launcher that never reads C back.
        let probe = round
            .find("TimedRegion::closed_form_probe()")
            .expect("the round must probe the TIMED launcher's writeback");
        let first_timed = round
            .find("bi::run_rotated(")
            .expect("the round must time something");
        assert!(
            probe < first_timed,
            "`{name}` runs the closed-form probe AFTER its first timed region — afterwards it is a \
             post-mortem, and the resident column it guards is published"
        );
        assert!(
            round.contains("published == 0"),
            "`{name}` no longer counts its publications, so a round where every cell refused would \
             print dashes and still exit 0"
        );
        // **The counter is not the property; the ESCALATION is.** `if published == 0 { eprintln!(..) }`
        // satisfies the check above while restoring the exact rc-0 defect, and so does hoisting the
        // increment out of the `Ok(..)` arms to count attempts instead of publications. Pin the
        // `skip_or_fail` argument, which only the escalating form can carry.
        assert!(
            round.contains("the round completed but every cell refused"),
            "`{name}` counts publications but no longer ESCALATES when there are none, so a \
             clock-drift refusal would print fourteen dashes and hand `bench()` rc 0"
        );
        // **The capability escalation, and the reason it needs a second lever on `::bench`.** The
        // round escalates on `WUKONG_GPU_REQUIRED` *as well as* the declared capability, because
        // `bench()` exports only the former. If the harness ever grows the `_cc_for_sku` export that
        // `test()` has, this assertion fails — deliberately — and whoever sees it can narrow the
        // round's second lever back to the precise one instead of leaving a stale comment behind.
        assert!(
            round.contains("std::env::var_os(\"WUKONG_GPU_REQUIRED\").is_some()"),
            "`{name}` no longer escalates its capability skip on WUKONG_GPU_REQUIRED. That is the \
             ONLY lever ::bench gives it — a non-Hopper container would print [skip:capability], \
             libtest would report ok, and the metered round would be green having executed no wgmma \
             instruction."
        );
        // The harness half: the entrypoint must exist, take a package, pass it, and carry the status.
        let modal = include_str!("../../../tools/cloud/modal_app.py");
        let bench_body = body_of(modal, "def bench(", PY_FN_END);
        // The other half of the escalation fact: `bench()` must still NOT export the capability, or
        // the round's blunt second lever should be narrowed to the precise one. `test()` is checked
        // as the positive control, so this cannot pass merely because the variable was renamed.
        let cc_export = format!("env[\"{}\"] = ", gpu_accel::REQUIRED_CC_ENV);
        assert!(
            body_of(modal, "def test(", PY_FN_END).contains(&cc_export),
            "`::test` no longer exports {}, so this law's reading of which entrypoint carries the \
             capability has gone stale and must be re-derived from the harness",
            gpu_accel::REQUIRED_CC_ENV
        );
        assert!(
            !bench_body.contains(&cc_export),
            "`::bench` now exports {} from the booked SKU. Good — that is the precise lever. Narrow \
             `{name}`'s second, blunter lever (WUKONG_GPU_REQUIRED) to it and delete the comment \
             explaining why the blunt one was needed.",
            gpu_accel::REQUIRED_CC_ENV
        );
        assert!(
            bench_body.contains("env[\"WUKONG_GPU_REQUIRED\"] = \"1\""),
            "`::bench` no longer sets WUKONG_GPU_REQUIRED, which is `{name}`'s only capability lever \
             — the round could then skip itself green on a non-Hopper container"
        );
        assert!(
            modal.contains("def bench(name: str = \"\", package: str = \"wukong_codegen_gpu\""),
            "`::bench` no longer takes a `package`, so `--package wukong_driver` would be rejected \
             and this bench would be unreachable from the harness entirely"
        );
        assert!(
            bench_body.contains("\"-p\", package"),
            "`::bench` no longer passes `package` to cargo, so `--package` is accepted and ignored \
             — the worst of the three failures, because the command line still looks right"
        );
        assert!(
            bench_body.contains("--ignored"),
            "`::bench` no longer passes --ignored, so it can no longer select this #[ignore]d round"
        );
        assert!(
            bench_body.contains("sys.exit(rc)"),
            "`::bench` no longer carries its child's exit status, so a failing witness or a wrong \
             answer would produce a green `modal run --detach`"
        );
        // `::build --release` must stage THIS package's release test binary, or `_require_prebuilt`
        // aborts the round before it starts.
        let build_body = body_of(modal, "def build(", PY_FN_END);
        assert!(
            build_body.contains("\"-p\", \"wukong_driver\"") && build_body.contains("--no-run"),
            "`::build` no longer builds the wukong_driver test binary, so `_require_prebuilt` will \
             refuse this round: {E2E_BENCH_INVOCATION}"
        );
        // That cargo line is guarded by `if driver:`, so the DEFAULT is load-bearing: flip it and the
        // documented `::build --release` stops staging this package while the assertion above stays
        // green, and every round then aborts inside `_require_prebuilt`.
        assert!(
            modal.contains("def build(release: bool = False, driver: bool = True)"),
            "`::build` no longer builds the driver package by default, so `{E2E_BENCH_INVOCATION}` \
             needs a `--driver` flag the documented prerequisite does not pass"
        );
    }
}
