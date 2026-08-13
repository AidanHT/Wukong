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
//! | `e2e_ms` | the `--backend=gpu` execution: interpreter + [`GpuAccel`] + host f32->f16 conversion + H2D + kernel + D2H + the seam's copy back into the user's buffer | `wukong_interp::run_kernel_f32_accel` |
//! | `launch_ms` | the same conversion, copies and kernel **without** the pipeline: `gpu::gemm_nt_wgmma` called directly, host slices in, host `Vec` out | the direct-launch peer of `e2e_ms` |
//! | `kernel_ms` | the kernel alone, device-resident buffers, no copies — **the campaign's published quantity** | `gpu::time_gemm_nt_wgmma` |
//!
//! `launch_ms` and `e2e_ms` both also contain something `kernel_ms` does not, and it is worth naming
//! rather than discovering later: `gemm_nt_wgmma` **regenerates the PTX text and re-resolves the
//! module on every call**, while `time_gemm_nt_wgmma` does that once outside its timed region. The two
//! host-to-host arms pay it identically — they are the same function — so it cannot bias `pipeline
//! cost vs launcher`; it is simply part of what "retention" is measuring, alongside the copies.
//!
//! From those, two ratios and one attribution:
//!
//! - **`e2e_ms` vs `peer_h2h_ms`** — the honest end-to-end comparison, because
//!   `baselines::cublas_gemm_nt_f16_f32out` is *also* host-slices-in / host-`Vec`-out and therefore
//!   also pays the conversion and both PCIe crossings. This is the only ratio here that compares
//!   like with like at the whole-program level.
//! - **`kernel_ms` vs `peer_ms`** — both device-resident, i.e. the published table's own comparison,
//!   re-measured in this same round so the two share a denominator, a clock and a container.
//! - **`kernel_ms / e2e_ms` — the RETENTION**: what fraction of the direct-launch number survives the
//!   trip through the compiler and the offload seam. `launch_ms` splits it in two —
//!   `kernel_ms / launch_ms` is the copy tax, `launch_ms / e2e_ms` is the pipeline's own share.
//!
//! **`e2e_ms` must never be quoted against `peer_ms`.** One includes both PCIe crossings and the
//! other does not; the round prints them in separate blocks for exactly that reason.
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
//!    cannot feel. The operands are all-ones, for which `Y[i][j] = sum over K of 1*1 = K`
//!    **exactly** — 1.0 is exact in f16, and a sum of `K <= 8192` ones is exact in f32 under any
//!    reassociation — so this is an `==` check, not a tolerance. **Every lane** of `Y` is checked,
//!    never a sum of them: a buffer with one row right and the rest stale passes a reduction and
//!    fails this.
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
use wukong_codegen_gpu::ptx_wgmma::{bench_iters, GemmPoint, WGMMA_BENCH_GRID};
use wukong_span::{Interner, SourceMap};

/// The bench's test-function name, so the reachability law selects on the same string the invocation
/// does rather than on a copy of it.
pub(crate) const E2E_BENCH_NAME: &str = "gpu_e2e_route_vs_cublas";

/// **The one invocation that runs this round.**
///
/// `::bench` (not `::test`) because the round is `#[ignore]`d — `::bench` appends `--ignored`, which
/// is the only way to select it, and `::test` never passes that flag. `--package wukong_driver`
/// because this bench does not live in `bench()`'s default package. `--peers` so a container that
/// cannot load cuBLAS **fails** instead of printing `[skip]` over a round with no bar. `WK_GPU=H100`
/// because `wgmma` is architecture-locked to `sm_90a`, and the harness exports the declared SKU so
/// the capability skip below can escalate rather than report a green PASS on the wrong container.
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

/// The `wukong_*` runtime symbol the plain `A.B^T` recognizer emits — the grep target for the manual
/// GATE-BLIND check, and what `the_bench_program_is_recognized_without_a_device` asserts on.
pub(crate) const NT_GEMM_SYMBOL: &str = "wukong_sgemm_nt";

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
/// very fast pipeline. `load_imports` is the one stage absent, because the generated source has no
/// `import`; a real user's file pays it once per imported file.
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

    /// The sample-label field both twins record into. One name, so the two `analyze` calls and the
    /// two `publish` calls cannot drift onto different fields.
    const H2H_FIELD: &str = "h2h_ms";

    /// The two resident-buffer readings of one shape, plus its compile cost. `e2e_ms` and
    /// `launch_ms` are not here: they come out of the twin verdict, so there is one source for them.
    #[derive(Debug, Clone, Copy)]
    struct ShapeCosts {
        compile_ms: f64,
        kernel_ms: f64,
        peer_ms: f64,
    }

    /// Loudly skip when there is no device, escalating under `WUKONG_GPU_REQUIRED=1` — which
    /// `::bench` always sets — through the one helper every skip in the workspace goes through.
    fn skip_no_device(name: &str) {
        wukong_codegen_gpu::diff::skip_or_fail(
            name,
            wukong_codegen_gpu::gpu::init_error().unwrap_or("no CUDA device reachable"),
        );
    }

    /// All-ones operands: the fill that makes correctness an `==` check at a bench shape. `1.0` is
    /// exact in f16, and `Y[i][j] = sum over K of 1*1 = K` is exact in f32 for every `K` in the grid
    /// under any reassociation the kernel's tiling, split-K or raster order might choose.
    fn ones(n: usize) -> Vec<f32> {
        vec![1.0f32; n]
    }

    /// Every lane of `y` must be exactly `k`. Reports the first offender with its 2-D index and how
    /// many lanes are wrong, because "one lane" and "every lane" are different bugs.
    fn assert_every_lane_is_k(what: &str, y: &[f32], m: usize, k: usize, n: usize) {
        let want = k as f32;
        assert_eq!(y.len(), m * n, "{what}: output buffer is the wrong length");
        let wrong = y.iter().filter(|&&v| v != want).count();
        if wrong == 0 {
            return;
        }
        let at = y.iter().position(|&v| v != want).expect("wrong > 0");
        panic!(
            "{what}: {wrong} of {} lanes are not exactly K={k}; first at y[{}][{}] = {} (all-ones \
             operands make every lane exactly K, bit-for-bit, so this is a WRONG ANSWER and no \
             timing from this shape may be published)",
            m * n,
            at / n,
            at % n,
            y[at]
        );
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
        // the correct outcome is a capability skip and nothing else. But the invocation books an
        // H100 and the harness exports the declared SKU, so a skip on a container that DECLARED a
        // Hopper part is a mis-provisioned round, not a correct skip.
        let cc = g.target().cc();
        if gpu_accel::gemm_route_for(cc, false) != GemmRoute::Wgmma {
            assert!(
                !gpu_accel::capability_skip_is_a_failure(cc, gpu_accel::required_cc()),
                "{E2E_BENCH_NAME}: this round declared {}={} — a part the wgmma route TAKES — but \
                 the container probed cc {}.{} ({}). Publishing nothing here would be a green run \
                 that measured no wgmma instruction.",
                gpu_accel::REQUIRED_CC_ENV,
                std::env::var(gpu_accel::REQUIRED_CC_ENV).unwrap_or_default(),
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
        let mut rows: Vec<(&GemmPoint, ShapeCosts, bi::BenchVerdict, bi::BenchVerdict)> =
            Vec::new();
        for p in WGMMA_BENCH_GRID {
            // `GemmPoint` names its dims `m`/`n`/`k`; every launcher below takes `(m, k, n)`.
            let (m, k, n) = (p.m, p.k, p.n);
            let cfg = gpu_accel::wgmma_cfg_for(gpu_accel::WGMMA_F32_SEAM_DTYPE, m, n);
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
            let (mut x, mut w, mut y) = (ones(m * k), ones(n * k), vec![0f32; m * n]);
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
            assert_every_lane_is_k(&format!("{} offload route", p.label), &y, m, k, n);

            // The direct-launch arm, checked against the same reference: a wrong kernel there would
            // hand the retention ratio a denominator it did not earn.
            let direct = wukong_codegen_gpu::gpu::gemm_nt_wgmma(g, cfg, &x, &w, m, k, n)
                .unwrap_or_else(|e| panic!("{}: direct `gemm_nt_wgmma` failed: {e}", p.label));
            assert_every_lane_is_k(&format!("{} direct launch", p.label), &direct, m, k, n);
            // And the peer, host-to-host — arm A of the end-to-end ratio.
            let peer_once =
                wukong_codegen_gpu::baselines::cublas_gemm_nt_f16_f32out(g, &x, &w, m, k, n)
                    .unwrap_or_else(|e| panic!("{}: cuBLAS one-shot failed: {e}", p.label));
            assert_every_lane_is_k(&format!("{} cuBLAS peer", p.label), &peer_once, m, k, n);
            eprintln!(
                "  [gate] route = Wgmma, and all three paths returned exactly K={k} on every one \
                 of {} lanes \u{2713}",
                m * n
            );

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

            // --- 4d. twin 2: the pipeline against the cuBLAS PEER, both host-to-host ---------
            let vs_peer = bi::run_rotated(
                &format!("{E2E_BENCH_NAME}@{}/pipeline_vs_cublas", p.label),
                &plan,
                |arm| match arm {
                    bi::Arm::Baseline | bi::Arm::Control => {
                        let t0 = Instant::now();
                        let out = wukong_codegen_gpu::baselines::cublas_gemm_nt_f16_f32out(
                            g, &x, &w, m, k, n,
                        );
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

            // --- 4e. the resident-buffer pair: the published table's own comparison ----------
            let kernel_ms = wukong_codegen_gpu::gpu::time_gemm_nt_wgmma(g, cfg, m, k, n, iters)
                .map(|s| s * 1e3)
                .unwrap_or(f64::NAN);
            let peer_ms = wukong_codegen_gpu::baselines::time_cublas_gemm_nt_f16_f32out(
                g,
                m,
                k,
                n,
                iters as u32,
            )
            .map(|s| s * 1e3)
            .unwrap_or(f64::NAN);

            rows.push((
                p,
                ShapeCosts {
                    compile_ms,
                    kernel_ms,
                    peer_ms,
                },
                bi::analyze(&overhead, plan.bar),
                bi::analyze(&vs_peer, plan.bar),
            ));
        }

        // --- 5. close, then let the gate decide ------------------------------------------------
        round.close(bi::query_smi());
        eprintln!("\n---- {E2E_BENCH_NAME}: round close ----");
        eprint!("{}", round.header());

        eprintln!("\n---- {E2E_BENCH_NAME}: per-shape verdicts ----");
        for (p, _, ov, vp) in &rows {
            eprintln!("  == {} : pipeline vs direct launcher ==", p.label);
            eprint!("{}", ov.table());
            eprintln!(
                "  == {} : pipeline vs cuBLAS (both host-to-host) ==",
                p.label
            );
            eprint!("{}", vp.table());
        }

        eprintln!(
            "\n---- {E2E_BENCH_NAME}: THE PUBLISHED TABLE ----\n\
             `route/cuBLAS` compares two host-to-host paths: both convert f32->f16 on the host and\n\
             both cross PCIe twice. `pipeline cost` is what the compiler and interpreter add over\n\
             calling the launcher directly; a tie there means the route is free.\n"
        );
        eprintln!(
            "  {:<15} {:>18} {:>28}  route/cuBLAS, host-to-host",
            "shape", "MxNxK", "pipeline cost vs launcher"
        );
        for (p, _, ov, vp) in &rows {
            let cost = match round.publish(ov, H2H_FIELD) {
                Ok(pd) => format!(
                    "{:+.1}% ({}, floor +/-{:.2}%)",
                    (pd.ratio() - 1.0) * 100.0,
                    if pd.is_tie() { "tie" } else { "moved" },
                    pd.floor() * 100.0
                ),
                Err(why) => format!("-- ({why})"),
            };
            let head = match round.publish(vp, H2H_FIELD) {
                // `ratio` is B/A on a MILLISECOND field: below 1.0 means we are faster, and the
                // throughput percentage is its reciprocal. `bench_instrument` is unit-agnostic and
                // cannot catch a ratio quoted the wrong way round, so polarity is owned here.
                Ok(pd) => format!(
                    "{:.1}% of cuBLAS ({:.4}x its ms, {}, floor +/-{:.2}%)",
                    100.0 / pd.ratio(),
                    pd.ratio(),
                    if pd.is_tie() { "tie" } else { "moved" },
                    pd.floor() * 100.0
                ),
                Err(why) => format!("-- ({why})"),
            };
            eprintln!("  {:<15} {:>18} {:>28}  {head}", p.label, p.dims(), cost);
        }

        eprintln!(
            "\n---- {E2E_BENCH_NAME}: RETENTION, and the compile/execute split ----\n\
             `retention` = kernel_ms / e2e_ms: the fraction of the DIRECT-LAUNCH number that\n\
             survives the pipeline. It is below 1 by the copies the direct-launch bench does not\n\
             pay, which is why `launch ms` sits beside it — kernel/launch is the copy tax alone and\n\
             launch/e2e is the pipeline's own share.\n\
             `compile ms` is a ONE-TIME host cost and is NOT part of any ratio above (T4 /\n\
             docs/compile-floor.md); reporting the two apart is the whole point.\n"
        );
        eprintln!(
            "  {:<15} {:>11} {:>11} {:>11} {:>11} {:>10} {:>11}",
            "shape", "compile ms", "e2e ms", "launch ms", "kernel ms", "peer ms", "retention"
        );
        let num = |x: Option<f64>| match x {
            Some(v) if v.is_finite() && v > 0.0 => format!("{v:.4}"),
            _ => "n/a".to_string(),
        };
        let ms = |v: f64| {
            if v.is_finite() && v > 0.0 {
                format!("{v:.4}")
            } else {
                "n/a".to_string()
            }
        };
        for (p, c, ov, _) in &rows {
            let f = ov.field(H2H_FIELD);
            let launch = f.and_then(|x| x.a);
            let e2e = f.and_then(|x| x.b);
            let retention = match e2e {
                Some(e) if e > 0.0 && c.kernel_ms.is_finite() => {
                    format!("{:.1}%", 100.0 * c.kernel_ms / e)
                }
                _ => "n/a".to_string(),
            };
            eprintln!(
                "  {:<15} {:>11} {:>11} {:>11} {:>11} {:>10} {:>11}",
                p.label,
                ms(c.compile_ms),
                num(e2e),
                num(launch),
                ms(c.kernel_ms),
                ms(c.peer_ms),
                retention
            );
        }
        eprintln!(
            "\n  The resident-buffer pair, for comparison with the published wgmma table (same\n\
             round, same clock, same container, same denominator):"
        );
        eprintln!("  {:<15} {:>22}", "shape", "kernel/peer, resident");
        for (p, c, _, _) in &rows {
            let pct = if c.kernel_ms.is_finite() && c.kernel_ms > 0.0 && c.peer_ms.is_finite() {
                format!("{:.1}% of cuBLAS", 100.0 * c.peer_ms / c.kernel_ms)
            } else {
                "n/a".to_string()
            };
            eprintln!("  {:<15} {:>22}", p.label, pct);
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
            assert!(
                mir_text(&program, &interner).contains(NT_GEMM_SYMBOL),
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
            let cfg = gpu_accel::wgmma_cfg_for(gpu_accel::WGMMA_F32_SEAM_DTYPE, m, n);
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

    /// **The all-ones fill really does make correctness an `==` check at every grid shape.**
    ///
    /// The round's strongest guard rests on two arithmetic facts, both shape-dependent, so they are
    /// checked per shape rather than asserted in prose: `1.0` survives the seam's host `f32 -> f16`
    /// conversion exactly, and a sum of `K` ones is exact in f32 (which needs `K < 2^24`). If a future
    /// grid row broke either, `assert_every_lane_is_k` would start failing on a *correct* kernel, and
    /// the natural repair — loosening it to a tolerance — would delete the only thing standing
    /// between this round and publishing a fast wrong answer.
    #[test]
    fn the_ones_fill_is_exact_at_every_grid_shape() {
        assert_eq!(
            half::f16::from_f32(1.0).to_f32(),
            1.0,
            "1.0 must survive the seam's f16 conversion exactly"
        );
        assert!(
            half::f16::from_f32(1.0).is_normal(),
            "the ones fill must be a NORMAL f16, or `wgmma_declines_operand_range` declines it and \
             every shape falls back to `gpu::gemm_nt`"
        );
        for p in WGMMA_BENCH_GRID {
            assert!(
                (p.k as u64) < (1u64 << 24),
                "{}: K={} is past f32's exact-integer range, so a sum of K ones is no longer an \
                 `==` check",
                p.label,
                p.k
            );
            assert_eq!(
                (p.k as f32) as f64,
                p.k as f64,
                "{}: K is not exactly representable in f32",
                p.label
            );
        }
    }

    /// **The invocation reaches a bench that can fail, and selects it in the package it lives in.**
    ///
    /// Three independent ways this round could bill an H100 and prove nothing, each of which has
    /// happened in this repo or was one flag away from it:
    ///
    /// - **the wrong entrypoint.** `::bench` appends `--ignored`, so it reaches ONLY `#[ignore]`d
    ///   tests; `::test` reaches only plain ones. A name routed through the wrong one selects zero
    ///   tests, prints `0 passed` and exits 0 — the 2026-08-11 vacuous round, $0.006.
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
            "`--filter` is ::test's selector and ::test never passes --ignored, so it would select \
             ZERO tests here and exit 0: {inv}"
        );
        assert!(
            inv.contains("--package wukong_driver"),
            "`bench()`'s package defaults to wukong_codegen_gpu, which does not contain this bench \
             — without `--package wukong_driver` the round selects zero tests: {inv}"
        );
        assert!(
            inv.contains("--peers"),
            "without --peers a container that cannot load cuBLAS prints [skip] and reports green \
             over a round with no bar: {inv}"
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
        let at = me
            .find(&decl)
            .unwrap_or_else(|| panic!("the invocation names `{name}`, not a test in this file"));
        // The attributes immediately above the declaration, not anywhere in the file.
        let attrs = &me[at.saturating_sub(160)..at];
        assert!(
            attrs.contains("#[test]") && attrs.contains("#[ignore"),
            "`{name}` must be an #[ignore]d #[test]: ::bench passes --ignored and would otherwise \
             select zero tests. Saw: {attrs:?}"
        );
        // The round must be able to fail on the two things the instrument cannot see. Scoped to the
        // round's own body: it ends where the next `#[test]` begins.
        let round = body_of(me, &decl, "\n    #[test]");
        assert!(
            round.contains("(accel.gemm_wgmma_calls, accel.gemm_existing_calls),"),
            "`{name}` no longer compares the per-route counters, so it cannot tell a wgmma launch \
             from a fallback and would publish a ratio between two different kernels"
        );
        assert!(
            round.contains("assert_every_lane_is_k"),
            "`{name}` no longer checks every output lane, so a fast WRONG answer could publish"
        );
        assert!(
            round.contains("round.publish("),
            "`{name}` no longer publishes through the instrument's gate, so a round whose control \
             floor never cleared the bar would print a number anyway"
        );

        // The harness half: the entrypoint must exist, take a package, pass it, and carry the status.
        let modal = include_str!("../../../tools/cloud/modal_app.py");
        let bench_body = body_of(modal, "def bench(", PY_FN_END);
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
    }
}
