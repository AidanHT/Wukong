//! The GPU implementation of [`wukong_interp::Accelerator`] — the bridge that makes `--backend=gpu`
//! run recognized kernel calls on the device while the rest of the program is still tree-walked by
//! the interpreter (so control flow, buffer layout, and every non-kernel op are bit-identical to the
//! CPU oracle; only the recognized GEMM/norm/activation/reduction calls move to the GPU). The offload
//! menu is exactly the five [`Accelerator`] hooks implemented below: `sgemm_nt`, the fused
//! `sgemm_nt_epi` epilogue (`act(x·Wᵀ [+ bias])`), `vmath`, `norm` and `sreduce`.
//!
//! Behind the driver's `gpu` feature, so the default toolchain-free build never compiles `cudarc`.
//! A GPU error is surfaced as `Some(Err(..))`, never `None`: declining (`None`) means "fall back to
//! the CPU kernel", and silently running on the CPU when the user asked for `--backend=gpu` would be
//! dishonest. `None` is reserved for shapes/ops the GPU wrappers structurally don't cover. There is
//! a third outcome that is neither, and it belongs in the same sentence: a launch that never retires
//! ends the process from inside `wukong_codegen_gpu::gpu`'s launch wait (`std::process::exit(70)`
//! after the hung-kernel diagnosis), because a wedged context blocks every later driver call. See
//! [`GemmRoute`] for both non-decline outcomes stated against the route that can reach them.
//!
//! # The Hopper `wgmma` route (Act 2, wave 4)
//!
//! [`GemmRoute`] is the second dispatch axis this file carries: on a Hopper part the recognized plain
//! `C = A·Bᵀ` call is routed to [`wukong_codegen_gpu::gpu::gemm_nt_wgmma`] — the warpgroup-MMA + TMA
//! family the whole Act-2 campaign measures — instead of the pre-Hopper launcher. Until this seam
//! existed that kernel had **no product call site at all**: a repo-wide search found two callers, both
//! inside `gpu.rs`'s own test module, so no `.wk` program could reach it
//! (`docs/gpu/derive/WAVE4_DOSSIER.md` §5.0, §5.4 gap 4). Everything about the route is stated as a
//! pure function of the *probed* [`wukong_codegen_gpu::gpu::GpuTarget`] plus the shape, so it is
//! decidable device-free and is unit-tested that way below.
//!
//! Because a fallback is *silent by design* — the same answer, from the other kernel — the route also
//! carries a **witness**: [`GpuAccel::gemm_wgmma_calls`] / [`GpuAccel::gemm_existing_calls`] split the
//! offload count by route, [`GpuAccel::gemm_route_taken`] reads them, and [`ROUTE_LOG_ENV`] prints
//! each dispatch with its decline reason. Without one, nothing anywhere distinguishes "the wgmma route
//! ran" from "the wgmma route declined and the pre-Hopper launcher ran", and a round on rented silicon
//! can measure one kernel while reporting the other's name.
//!
//! # The one command that runs the laws in this file
//!
//! ```text
//! cargo test -p wukong_driver --features gpu --lib
//! ```
//!
//! Nothing else reaches them, and for a while nothing did. The workspace `cargo test` never passes
//! `--features gpu`, so this whole module is `cfg`'d out of it (the crate reports 20 tests without the
//! feature and 38 with it); the device suite is scoped `-p wukong_codegen_gpu`; and CI's `gpu-check` job
//! ran `cargo check`/`cargo clippy --features gpu`, which *compile* a test without running it. Deleting
//! the odd-`N` decline, the operand-range decline or the route witness therefore left every gate part and
//! every CI job green. That command is now a `gpu-check` step (it is device-free — the module below never
//! constructs a [`Gpu`], and the end-to-end gates in `lib.rs` `[skip]` without a device) and a named part
//! of the pre-commit gate in `CONTRIBUTING.md`. Under `WUKONG_GPU_REQUIRED=1` the skips become failures,
//! so the same command is the device gate on real silicon.

use wukong_codegen_gpu::ptx_wgmma::{self, WgmmaCfg, WgmmaDtype};
use wukong_codegen_gpu::Gpu;
use wukong_interp::Accelerator;

/// Wraps the process-wide [`Gpu`] context and routes recognized kernels to its launch wrappers.
/// `calls` counts kernels that actually ran on the device — the tolerance test asserts it is `> 0`
/// so a silent CPU fallback can never masquerade as a passing GPU run.
///
/// The two `gemm_*_calls` fields split that total by GEMM **route**, and they exist because `calls`
/// alone cannot tell the two apart: [`Accelerator::sgemm_nt`] bumps it identically whether the wgmma
/// family took the call or declined it to the pre-Hopper launcher. Without the split, an H100 round
/// can time `gpu::gemm_nt` and report it as wgmma — see [`GpuAccel::gemm_route_taken`].
pub struct GpuAccel<'g> {
    pub gpu: &'g mut Gpu,
    pub calls: u32,
    /// Plain `C = A·Bᵀ` offloads the **wgmma** family took (a launch was issued, whether or not the
    /// driver then failed it). Part of `calls`.
    pub gemm_wgmma_calls: u32,
    /// Plain `C = A·Bᵀ` offloads that ran on [`GemmRoute::Existing`] — `gpu::gemm_nt`. Part of
    /// `calls`.
    pub gemm_existing_calls: u32,
}

impl<'g> GpuAccel<'g> {
    /// Construct over a bound GPU context with zeroed offload counters.
    pub fn new(gpu: &'g mut Gpu) -> Self {
        GpuAccel {
            gpu,
            calls: 0,
            gemm_wgmma_calls: 0,
            gemm_existing_calls: 0,
        }
    }

    /// The route this context's device is **eligible** for on a recognized `C = A·Bᵀ`, including the
    /// [`WGMMA_OFF_ENV`] switch. The one place the pure decision meets the probed device, and
    /// therefore the one a device gate must ask rather than re-deriving from `cc`.
    ///
    /// **This is an eligibility, not a witness.** It is `cc` and the env switch and nothing else, so
    /// on a Hopper part it answers `Wgmma` for *every* shape — including the ones [`wgmma_declines`]
    /// then sends to the other launcher. Anything that needs to know what actually ran must ask
    /// [`GpuAccel::gemm_route_taken`], which reads the counters.
    pub(crate) fn gemm_route(&self) -> GemmRoute {
        gemm_route_for(self.gpu.target().cc(), wgmma_disabled_by_env())
    }

    /// **The route that actually RAN, or `None` if no plain GEMM was offloaded at all.**
    ///
    /// The witness [`GpuAccel::gemm_route`] cannot be. Every decline in [`wgmma_declines`] — an
    /// unencodable `K`, an odd `N`, an output past the epilogue's `u32` index, a ring over the
    /// device's shared-memory budget — and every `UNSUPPORTED` the generator itself returns leaves
    /// the eligibility at `Wgmma` while `gpu::gemm_nt` does the work. A gate or a bench that asks
    /// only the eligibility therefore cannot distinguish "the wgmma route ran" from "the wgmma route
    /// declined and the pre-Hopper launcher ran", which on rented silicon means measuring one kernel
    /// and publishing the other's name.
    ///
    /// `#[cfg(test)]` for the same reason [`HOPPER_GATE_SHAPES`] is: the gates are its only callers
    /// today, and the counters it reads are `pub` fields any of them can also compare directly.
    /// Promoting it to product API means making [`GemmRoute`] `pub` with it.
    #[cfg(test)]
    pub(crate) fn gemm_route_taken(&self) -> Option<GemmRoute> {
        gemm_route_taken_from(self.gemm_wgmma_calls, self.gemm_existing_calls)
    }

    /// **`C = A·Bᵀ` through the Hopper wgmma family, or `None` to fall back.**
    ///
    /// `None` at every step is a routing decision: not Hopper, the switch is set, the shape has no
    /// encodable descriptor, the ring does not fit, or the generator itself declined. Only a real
    /// driver failure becomes `Some(Err(..))`, matching this module's contract that a GPU error is
    /// never silently downgraded to a CPU run.
    ///
    /// # This changes the arithmetic of the plain-GEMM offload on Hopper, deliberately, and only there
    ///
    /// [`wukong_codegen_gpu::gpu::gemm_nt`] — the [`GemmRoute::Existing`] arm — is an f32 kernel;
    /// `gemm_nt_wgmma` converts both operands to f16 on the host and accumulates in f32, so the two
    /// arms do not agree bit-for-bit and the wgmma arm carries an f16 input rounding the other does
    /// not. That is inside the `--backend=gpu` contract rather than a violation of it: CPU↔GPU
    /// agreement in this backend is a **tolerance** gate (`c·√K·ε`), and the fused-epilogue hook below
    /// has run f16 tensor cores against the f32 CPU oracle since Act 1. What it is *not* is invisible —
    /// a tolerance band sized for f32 would fail on this route, so the driver's device gate reads
    /// [`GpuAccel::gemm_route_taken`], the per-route counters, and picks its band from the route that
    /// actually ran (never from the device, which is eligible for shapes it then declines).
    ///
    /// The conversion is **not only** a rounding, and the part that is not is handled rather than
    /// tolerated. f16 has about four decades less dynamic range than f32 at *each* end, and both
    /// ends change the **answer** rather than widening the error bar: an operand past 65504 enters
    /// the GEMM as an infinity and leaves as `inf`/`NaN`, and a **row** whose whole magnitude range
    /// is under f16's smallest normal (6.104e-5) enters as subnormals or exact zeros and leaves an
    /// entire row (A) or column (B) of `C` as zeros — where the f32 arm returns the finite, non-zero
    /// matrix the CPU oracle returns. [`wgmma_declines_operand_range`] cuts both, so what the
    /// tolerance band has to cover really is the rounding, with the one residual edge stated at
    /// [`WGMMA_F32_SEAM_DTYPE`]: inside a *row* whose peak is normal, individual lanes far below
    /// that peak still flush, and their error is bounded by their own magnitude against a sum that
    /// holds something larger.
    fn try_gemm_nt_wgmma(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<Result<Vec<f32>, String>> {
        if self.gemm_route() != GemmRoute::Wgmma {
            let cc = self.gpu.target().cc();
            route_log(format_args!(
                "{m}x{k}x{n}: Existing (cc {}.{} is not eligible for wgmma, or {WGMMA_OFF_ENV} is set)",
                cc.0, cc.1
            ));
            return None;
        }
        let cfg = wgmma_cfg_for(WGMMA_F32_SEAM_DTYPE, m, n);
        // The launcher's own preconditions are asserts; decide them here so a shape it would abort on
        // simply takes the other path.
        if let Some(why) = wgmma_declines(cfg, m, k, n, self.gpu.smem_budget()) {
            route_log(format_args!("{m}x{k}x{n}: Existing (declined: {why})"));
            return None;
        }
        if a.len() != m * k || b.len() != n * k {
            route_log(format_args!(
                "{m}x{k}x{n}: Existing (operand lengths {}/{} are not m*k / n*k)",
                a.len(),
                b.len()
            ));
            return None;
        }
        // The only decline that reads the DATA rather than the shape, and the only one that is about
        // the arithmetic class rather than the launch: f16 carries four decades less dynamic range
        // than f32 at *each* end, so an element past 65504 would enter this GEMM as an infinity and
        // a ROW wholly under 6.104e-5 would enter it as zeros. `k` is the row length of both
        // operands (A is `m x k` row-major, B is `n x k`), which the length check immediately above
        // is what makes exact.
        if let Some(why) = wgmma_declines_operand_range(cfg.dtype, a, b, k) {
            route_log(format_args!("{m}x{k}x{n}: Existing (declined: {why})"));
            return None;
        }
        match wukong_codegen_gpu::gpu::gemm_nt_wgmma(self.gpu, cfg, a, b, m, k, n) {
            Ok(out) => {
                route_log(format_args!("{m}x{k}x{n}: Wgmma ({})", cfg.name));
                Some(Ok(out))
            }
            // A capability or encodability decline is the generator saying "not this shape", which is
            // a routing fact, not a failure — `UNSUPPORTED` means SKIP everywhere else in this
            // backend and it means fall-back here.
            Err(e) if e.unsupported().is_some() => {
                route_log(format_args!(
                    "{m}x{k}x{n}: Existing ({} declined at launch: {e})",
                    cfg.name
                ));
                None
            }
            Err(e) => Some(Err(format!("GPU {} failed: {e}", cfg.name))),
        }
    }
}

/// **The witness, as a pure function of the two counters** — so the algebra that decides "did the
/// wgmma route run" is testable on a laptop with no device, which is where every other rule in this
/// file is pinned.
///
/// A *mixed* run answers [`GemmRoute::Wgmma`]: one wgmma call is enough to put an f16 input rounding
/// into the buffers the run is judged on, so anything sizing a tolerance from this must take the
/// wider band. It is deliberately not a third variant — a caller that needs "were they *all* wgmma"
/// (the Hopper gate does) compares the counters directly and gets a better message for it.
#[cfg(test)]
pub(crate) fn gemm_route_taken_from(wgmma_calls: u32, existing_calls: u32) -> Option<GemmRoute> {
    match (wgmma_calls, existing_calls) {
        (0, 0) => None,
        (0, _) => Some(GemmRoute::Existing),
        _ => Some(GemmRoute::Wgmma),
    }
}

/// Whether `wukong_codegen_gpu::gpu::norm` has a PTX entry for this `NORM_*` op code — the driver-side
/// mirror of `gpu::norm_supported`, itself the sibling of `gpu::vmath_supported` /
/// `gpu::reduce_supported`. This copy and that predicate must agree op-for-op.
///
/// MIRROR: the op codes are defined in `wukong_runtime/src/norm.rs` — SOFTMAX=0, LAYERNORM=1,
/// RMSNORM=2, LOGSOFTMAX=3, L2NORM=4 — and `gpu::norm` maps only the first three, ending its match
/// in `panic!("norm op {op} not implemented on GPU yet")`. The recognizer in `wukong_mir_build`
/// emits all five (`tests/run/log_softmax_fused.wk` lowers to op 3, `tests/run/l2norm.wk` to op 4),
/// so without this gate those two programs abort the process under `--backend=gpu` instead of
/// falling back to the CPU kernel. Extending `gpu::norm` must extend this list.
fn norm_supported(op: i64) -> bool {
    (0..=2).contains(&op)
}

// --- the Hopper wgmma route ------------------------------------------------------------------------

/// **Which GEMM family a recognized `C = A·Bᵀ` offload is sent to.**
///
/// A routing decision, never an error: [`GemmRoute::Existing`] is the launcher this file called
/// before wave 4 and its results are unchanged, so every reason the wgmma family cannot take a call
/// that is **decidable before the launch** — the architecture, the shape, an unencodable tensor map,
/// a shared-memory budget, a grid past the driver's `gridDim.y` ceiling, an operand outside the seam
/// dtype's range, and the generator's own `UNSUPPORTED` — resolves to the same working path rather
/// than to a failure.
///
/// # The two outcomes that are NOT that, named rather than implied
///
/// The sentence above used to read "*every* reason", full stop, and that was wider than the code.
/// Two things a wgmma dispatch can do are neither a decline nor a result, and a reader deciding
/// whether to route a new caller here has to know both:
///
/// 1. **A driver rejection at the launch is `Some(Err(..))` — a hard error on a call that had a
///    working fallback.** [`GpuAccel::try_gemm_nt_wgmma`] falls back only on
///    `GpuError::unsupported()`, and `GpuError` has exactly two variants: everything else is
///    `GpuError::Driver` and is surfaced. The concrete one is a module load —
///    `gemm_nt_wgmma` -> `raw_function_dyn` -> `cuModuleLoadData` — refusing an `sm_90a` binary on a
///    driver too old to have that virtual architecture, since [`ptx_wgmma::Sm90aLicense`] gates on
///    the *device's* capability and not on the driver's. That is deliberate rather than an
///    oversight: this module's older and wider contract is that a GPU **failure** under
///    `--backend=gpu` is never silently answered on the CPU, and a driver that rejects a module the
///    device is capable of is a broken installation, not a shape this family declines. What it is
///    not is invisible — which is why it is written here.
/// 2. **A launch that never retires ends the PROCESS with exit code 70.** `gpu::sync_with_deadline`
///    (the wait every launcher in that crate goes through) prints the hung-kernel diagnosis and
///    calls `std::process::exit(70)` when `cuStreamQuery` is still `NOT_READY` at the deadline,
///    because a wedged context blocks every later driver call anyway. A `.wk` program that reaches
///    this seam therefore has a third exit path that neither returns nor falls back. The mechanism
///    is the mbarrier/cluster kind of bug, i.e. a defect in the family rather than a property of the
///    caller's shape, and `WUKONG_GPU_LAUNCH_TIMEOUT_MS` sets or disables the deadline.
///
/// `pub(crate)` because the driver's own device gates must size their tolerance band by *route*
/// rather than by device — see [`GpuAccel::try_gemm_nt_wgmma`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GemmRoute {
    /// `gpu::gemm_nt` (and, for the fused epilogue, the Act-1 `gemm_nt_f16_sm_db*` wmma family).
    /// Correct on every device this backend supports, Hopper included.
    Existing,
    /// `gpu::gemm_nt_wgmma` over a [`WgmmaCfg`] from [`wgmma_cfg_for`] — Hopper only.
    Wgmma,
}

/// Set this to anything to force [`GemmRoute::Existing`] on a Hopper part.
///
/// The route changes both the kernel and the arithmetic (see [`GpuAccel::try_gemm_nt_wgmma`]), so a
/// switch that flips it **inside one binary** is worth a line: an A/B whose two arms are two builds is
/// the hazard this repo has already been bitten by (two `cargo` profiles producing byte-identical
/// binaries under different names), and one binary run twice cannot have it.
const WGMMA_OFF_ENV: &str = "WUKONG_GPU_NO_WGMMA";

/// Whether [`WGMMA_OFF_ENV`] is set. Read at each dispatch rather than cached: the cost is one
/// `getenv` against a kernel launch, and a cached value would make the switch depend on which test in
/// a shared process happened to run first.
fn wgmma_disabled_by_env() -> bool {
    std::env::var_os(WGMMA_OFF_ENV).is_some()
}

/// Set this to anything to have every recognized `C = A·Bᵀ` offload print the route it took, and —
/// when the wgmma family declined — the reason, on stderr.
///
/// The counters ([`GpuAccel::gemm_route_taken`]) are the machine-readable witness and a test asserts
/// on them; this is the human one, for a round on rented silicon where the failure mode is "the
/// numbers are fine and they came from the other kernel". Off by default so a `.wk` run's stderr
/// stays diagnostics.
const ROUTE_LOG_ENV: &str = "WUKONG_GPU_ROUTE_LOG";

/// One `[wukong gpu route]` line per plain-GEMM offload when [`ROUTE_LOG_ENV`] is set.
/// `format_args!` at the call sites, so a disabled log formats nothing.
fn route_log(args: std::fmt::Arguments<'_>) {
    if std::env::var_os(ROUTE_LOG_ENV).is_some() {
        eprintln!("[wukong gpu route] {args}");
    }
}

/// **The routing decision, as a pure function of the probed compute capability.** No device, no
/// shape, no I/O — so the H100 point is decidable on this laptop and is asserted below.
///
/// # It is `major == 9`, not `>= (9, 0)`, and that is the whole point of the function
///
/// Every other capability gate in the GPU backend is a lower bound because the PTX ISA is forward
/// compatible. `sm_90a` is not: it is an *architecture lock*, and an `sm_90a` module fails to load on
/// `sm_100` exactly as it fails on this `sm_89` laptop. The kernel side already knows this —
/// [`ptx_wgmma::Sm90aLicense::for_probed_cc`] admits `cc.0 == 9` and rejects both directions — and
/// `gemm_nt_wgmma` takes that license as its first statement. So a `>=` rule here would hand a
/// Blackwell part a module it cannot run and turn a routing decision into a
/// `GpuError::Unsupported` the caller must recover from. `the_route_never_outruns_the_sm90a_license`
/// pins the two together over the whole capability grid, so widening the license without widening
/// this fails a test instead of failing a launch.
pub(crate) fn gemm_route_for(cc: (i32, i32), wgmma_off: bool) -> GemmRoute {
    if wgmma_off || cc.0 != 9 {
        return GemmRoute::Existing;
    }
    GemmRoute::Wgmma
}

/// **THE CONFIG SEAM. One function, and wave 3's per-shape dispatcher replaces exactly it.**
///
/// Today it is a delegation to the shipped regime rule in the generator crate
/// ([`ptx_wgmma::wgmma_w1_for`] / [`ptx_wgmma::wgmma_w1_bf16_for`]: the 1×2×1 B-multicast `v2`-store
/// row at or above 8e6 output elements, the un-clustered `v2` row below it), and it delegates rather
/// than re-deriving so the driver can never dispatch on a threshold the generator has since
/// re-measured. When wave 3 lands the per-shape dispatcher — which also chooses the *tile*, something
/// this rule structurally cannot express because both arms are 128×256 — the body of this function is
/// the one place that changes.
fn wgmma_cfg_for(dtype: WgmmaDtype, m: usize, n: usize) -> &'static WgmmaCfg {
    match dtype {
        WgmmaDtype::F16 => ptx_wgmma::wgmma_w1_for(m, n),
        WgmmaDtype::Bf16 => ptx_wgmma::wgmma_w1_bf16_for(m, n),
    }
}

/// **The shapes the Hopper end-to-end gate runs, shared so it and the device-free law below cannot
/// drift.**
///
/// `gpu_backend_linear_routes_through_wgmma_on_hopper` (driver `lib.rs`) can only reach the launch
/// on rented silicon — everywhere else it capability-`[skip]`s — and a claim in its doc comment — "this one crosses the cluster threshold, so it is the
/// only shape that exercises the clustered launch" — is exactly the kind of statement that rots
/// silently when the threshold moves. [`tests::the_hopper_gate_shapes_reach_both_regime_arms`] checks
/// that claim **here**, on this laptop, against
/// [`ptx_wgmma::W1_CLUSTER_MIN_OUTPUT_ELEMS`] itself rather than a copy of its value; both the gate
/// and the law read this one list.
#[cfg(test)]
pub(crate) const HOPPER_GATE_SHAPES: [(usize, usize, usize); 3] = [
    // A whole tile grid: 2 tiles of M x 2 of N, nothing ragged.
    (256, 1024, 512),
    // Ragged in M, N **and** K at once — the shape that proves the seam needs no alignment gate.
    (130, 1032, 258),
    // M*N = 8_388_608, just past the cluster threshold: the only one that returns the CLUSTERED row.
    (4096, 256, 2048),
];

/// **The one invocation that runs the Hopper end-to-end gate, pinned where a law can check it.**
///
/// `gpu_backend_linear_routes_through_wgmma_on_hopper` (driver `lib.rs`) is the only proof anywhere
/// that the wgmma family *executed* rather than declined, and its witness assertion —
/// `(gemm_wgmma_calls, gemm_existing_calls) == (calls, 0)` — is worth the rented minutes only if a
/// failure can turn the run red. Two independent things had to be true for that, and neither was
/// checked anywhere:
///
/// - **the entrypoint must propagate the failure.** `::bench` ended in
///   `_run(args, env, check=False)` and then returned, with no `sys.exit` anywhere in the function,
///   so it exited 0 whatever libtest did — a failed witness produced a green `modal run`. `::test`
///   ends `if rc1 or rc2: sys.exit(1)`, which is where a correctness gate belongs. (`bench()` now
///   propagates too, matching every other measurement entrypoint in that file, but a gate whose
///   product is a verdict rather than a table does not belong on the sweep runner.)
/// - **the entrypoint must SELECT the test.** `::bench` appends `--ignored`, which runs ONLY
///   `#[ignore]`d tests; `::test` never reaches one. Routing a name through the wrong one selects
///   zero tests, prints `0 passed` and exits 0 — the 2026-08-11 vacuous bring-up run, $0.006 of
///   H100, for which `ptx_wgmma`'s own visit constant grew the same law.
///
/// So the gate is a plain `#[test]` behind a capability skip and this is a `::test --filter` line.
/// `--release` because `::test` then requires the release test binaries `::build --release` stages,
/// which is what the campaign already builds; `--detach` because a local DNS flake has killed a
/// healthy round before. [`tests::the_hopper_gate_invocation_reaches_a_gate_that_can_fail`] checks
/// all of it textually, on a laptop, against both files.
#[cfg(test)]
pub(crate) const HOPPER_GATE_INVOCATION: &str = "WK_GPU=H100 modal run --detach \
     tools/cloud/modal_app.py::test \
     --filter gpu_backend_linear_routes_through_wgmma_on_hopper --release";

/// The 16-bit input type the **f32** seam feeds `wgmma`.
///
/// f16 and bf16 are the same width and trade eight bits of significand against eight of exponent, and
/// the recognized `sgemm_nt` call carries f32 operands, so the seam is choosing which f32 fact to
/// give up — *both* of them, not just precision:
///
/// - **Precision**, and the reason f16 wins: 11 significand bits against bf16's 8, three decimal
///   digits more at every magnitude an activation or a weight occupies.
/// - **Range**, and the reason that win is not free: bf16 carries f32's exponent field, so it holds
///   everything up to 3.39e38, while f16 stops at 65504 and reaches zero below ~6e-8. Choosing f16
///   gives up roughly four decades at each end — and the top end is not a wider error bar but a
///   different **class** of answer: a finite f32 above 65504 enters the GEMM as `inf` and comes back
///   as `inf`/`NaN` where the pre-Hopper f32 launcher returns a number.
///
/// **Both** ends are therefore closed by [`wgmma_declines_operand_range`] — such a call takes the f32
/// launcher and its results are unchanged, so f16's range cost is paid only in precision — and they
/// are closed by deliberately *different* rules, because the two failures do not propagate alike:
///
/// - **Overflow is per element.** One `inf` contaminates without bound (`inf + finite = inf`,
///   `inf * 0 = NaN`), so a single element past 65504 poisons every output lane its row or column
///   touches. Any element over the ceiling declines.
/// - **Underflow is per ROW, on that row's maximum**, because a row is the granularity at which a
///   flush changes an output lane. `C[i][j] = Σ_k A[i][k]·B[j][k]`, so row `i` of A feeds row `i` of
///   `C` and *nothing else*, and row `j` of B feeds column `j`. A flushed element therefore removes
///   only its own contribution to sums that its **own row's** larger elements also feed — which is
///   the bounded, tolerable case — while a row that is *wholly* under f16's smallest normal enters
///   the GEMM as subnormals or zeros and returns that entire row (A) or column (B) of `C` as flushed
///   lanes: by ~1e-8 exactly zero, where `gpu::gemm_nt` returns the finite matrix the CPU oracle
///   returns. So the rule is `0 < max|A[i][·]| < f16::MIN_POSITIVE` for any row `i` (and the same
///   for `B`), and the granularity is load-bearing rather than fussy: a single scalar peak over the
///   whole operand **cannot see a quiet row inside a loud one** — A with one row of 1.0s and one row
///   of 1e-8s has peak 1.0, passes a per-operand rule, and returns half of `C` as zeros.
///
///   It is per row rather than per element for the reason a per-element rule is useless: a `U(-1,1)`
///   buffer of a few million elements very probably holds one below f16's round-to-zero point
///   (`2^-25` = 2.98e-8) by pure chance, so that rule would decline nearly every real GEMM. The row
///   rule cannot fire on such data either — `P(max of K uniforms < 6.1e-5) = (6.1e-5)^K`, about
///   1e-34 at `K = 8` — costs the same single pass the overflow scan already makes (the `max` fold
///   simply restarts at each row boundary), and exempts an exactly-zero row, which converts
///   *exactly* and needs no fallback. What it does newly decline is a genuinely tiny row inside a
///   normal matrix, and that is the conservative direction: the cost is one fallback, the cost of
///   the other direction is a row of zeros returned as a result.
///
/// The residual edge, stated rather than hidden: inside a *row* whose peak is normal, lanes far
/// below that peak keep only f16's subnormal precision, so a result **dominated** by those lanes is
/// the shape this backend's GEMM band does not promise to cover. That much is the arithmetic class
/// the whole f16 tensor-core path has carried since Act 1 — the fused `gemm_nt_f16_sm_db*` epilogue
/// converts its operands exactly the same way — and it is bounded by lanes the same sum also holds,
/// unlike a whole-row flush.
///
/// bf16 becomes reachable when the lowp `wukong_sgemm_{bf16,f16}_nt_epi` symbols get an
/// `Accelerator` hook at all (WAVE4_DOSSIER §5.4 gap 3); [`wgmma_cfg_for`] already routes it, and it
/// is the arm that would need none of the range decline.
const WGMMA_F32_SEAM_DTYPE: WgmmaDtype = WgmmaDtype::F16;

/// **The largest magnitude the seam's 16-bit input type carries as a finite value**, asked of the
/// crate `gpu::gemm_nt_wgmma` converts with rather than typed as a literal — so a `half` upgrade
/// cannot move the conversion without moving the decline. f16 stops at 65504; bf16 has f32's
/// exponent field and stops at 3.39e38, one mantissa band short of `f32::MAX`.
///
/// Deliberately the **conservative** edge of the true overflow point: round-to-nearest-even carries
/// magnitudes just above this back *down* to it, so a thin band is declined that would in fact have
/// survived. An unnecessary fallback costs a fallback; the error in the other direction costs an
/// infinity.
fn seam_dtype_max_finite(dtype: WgmmaDtype) -> f32 {
    match dtype {
        WgmmaDtype::F16 => half::f16::MAX.to_f32(),
        WgmmaDtype::Bf16 => half::bf16::MAX.to_f32(),
    }
}

/// **The smallest magnitude the seam's 16-bit input type carries at full precision** — the underflow
/// twin of [`seam_dtype_max_finite`], asked of the same crate for the same reason. f16's smallest
/// normal is 6.104e-5; bf16 has f32's exponent field, so its is 1.18e-38.
///
/// Also the **conservative** edge, in the same direction as its twin: f16 does hold magnitudes below
/// this as subnormals, all the way down to 5.96e-8, so an operand whose peak sits in that band is
/// declined although it would have carried a few bits. The error in the other direction is a matrix
/// of zeros returned as a result, which is why the threshold is the smallest *normal* rather than
/// the smallest representable.
fn seam_dtype_min_positive_normal(dtype: WgmmaDtype) -> f32 {
    match dtype {
        WgmmaDtype::F16 => half::f16::MIN_POSITIVE.to_f32(),
        WgmmaDtype::Bf16 => half::bf16::MIN_POSITIVE.to_f32(),
    }
}

/// **The dynamic-range decline: an operand the seam's 16-bit type cannot hold at either end.**
///
/// `gemm_nt_wgmma` runs `half::f16::from_f32` over every element of both operands, and f16 spans
/// 6.104e-5 to 65504 ([`seam_dtype_min_positive_normal`], [`seam_dtype_max_finite`]) against f32's
/// 1.18e-38 to 3.4e38 — about four decades narrower at *each* end. Both ends produce a different
/// **answer**, not a wider error bar, where [`GemmRoute::Existing`] (`gpu::gemm_nt`, f32 in and f32
/// out) returns exactly what the CPU oracle returns:
///
/// - past the ceiling, the element becomes `inf` and propagates through the accumulation as
///   `inf`/`NaN`;
/// - beneath the floor, it becomes a subnormal or an exact zero, and a **row** that is *wholly*
///   beneath it comes back as a row (A) or column (B) of zeros — at ~1e-8 every element of that row
///   is under f16's round-to-zero point, so those lanes of `C` are identically 0 where the oracle is
///   finite and non-zero, and at ~1e-7 the surviving one or two bits carry 20-100% relative error.
///
/// No tolerance band covers either, and no gate here would catch either: every band gate in this
/// repo seeds `rng.vec(.., -1.0, 1.0)`, and `diff::assert_close` passes a lane on
/// `abs <= abs_tol || rel <= rel_tol`, so with the wgmma band's `abs_tol = 5e-2` **any** result
/// whose true magnitude is below 5e-2 passes at 100% relative error — a zeroed 1e-7-scale output
/// included. So both join the other reasons this seam cannot take a call, with the same consequence
/// — the pre-Hopper path, results unchanged.
///
/// The two rules are **not** symmetric, and [`WGMMA_F32_SEAM_DTYPE`] says why: overflow is per
/// element because one `inf` contaminates without bound; underflow is per **row**, on that row's
/// maximum, because `C[i][j] = Σ_k A[i][k]·B[j][k]` makes a row the unit that feeds an output lane —
/// a flushed element costs only its own contribution to sums its own row's larger elements also feed,
/// while a wholly-flushed row costs the whole lane. A per-*operand* maximum cannot express that: it
/// is blind to a quiet row inside a loud operand, which is a deterministically zeroed row of `C`
/// rather than a bounded error. A per-*element* rule would fire by chance on ordinary `U(-1,1)`
/// buffers and decline nearly every real GEMM.
///
/// **Cost, stated rather than hidden.** One read-only pass over each operand — an overflow exits it
/// early, nothing else can — so it is `O(m·k + n·k)`, the same order as the launcher's very next
/// act, which maps every one of those elements to `u16`, allocates the result and copies it to the
/// device. Row granularity does not change that: the underflow rule is still a `max` fold over a
/// scan that was already full-length in the common (no-decline) case, with the fold restarting at
/// each row boundary rather than a second pass. A constant factor on an already-linear host stage,
/// not a new order of work, and it disappears the day the seam takes operands that are already 16
/// bit (the lowp `nt_epi` hook, WAVE4_DOSSIER §5.4 gap 3).
///
/// **The bf16 arm effectively never declines at either end**, which is the range half of
/// [`WGMMA_F32_SEAM_DTYPE`]'s trade made operational: bf16 has f32's exponent field, so only the
/// last mantissa band below `f32::MAX` and magnitudes under 1.18e-38 fail it.
///
/// `k` is the shared row length: `a` is `m x k` row-major and `b` is `n x k`, which is what the NT
/// form means. [`GpuAccel::try_gemm_nt_wgmma`] checks `a.len() == m*k` and `b.len() == n*k` before
/// calling, so the chunking below is exact rather than approximate.
fn wgmma_declines_operand_range(
    dtype: WgmmaDtype,
    a: &[f32],
    b: &[f32],
    k: usize,
) -> Option<String> {
    // A per operand, then B, so the message names the one that actually failed and each ROW is
    // judged against its OWN peak — the underflow rule is a statement about one row of one matrix,
    // not about the pair and not about the matrix.
    operand_range_declines(dtype, "A", a, k).or_else(|| operand_range_declines(dtype, "B", b, k))
}

/// One operand, one pass, both ends — the ceiling per element, the floor per row of `k`. See
/// [`wgmma_declines_operand_range`].
fn operand_range_declines(dtype: WgmmaDtype, which: &str, xs: &[f32], k: usize) -> Option<String> {
    // Hoisted out of the scan: two conversions for the whole operand, then plain f32 compares.
    let ceiling = seam_dtype_max_finite(dtype);
    let floor = seam_dtype_min_positive_normal(dtype);
    // `k == 0` cannot reach here from the seam — `wgmma_declines` rejects a zero extent through the
    // tensor-map validator before this runs — but `chunks(0)` panics, so fold the whole operand as
    // one row rather than aborting a user's program on a shape that had a working fallback.
    let row_len = if k == 0 { xs.len().max(1) } else { k };
    for (r, row) in xs.chunks(row_len).enumerate() {
        // The largest FINITE magnitude in this row. Both comparisons below are `>`, which settles
        // the non-finite cases the way the routes actually differ: an infinity trips the ceiling and
        // declines (falling back reproduces the CPU oracle's own infinity exactly), while a NaN
        // trips neither — every comparison against it is false — so it never becomes the peak and
        // never declines, because it is a NaN on both routes and the route changes nothing about it.
        let mut peak = 0.0f32;
        for &x in row {
            let mag = x.abs();
            if mag > ceiling {
                return Some(format!(
                    "operand {which} holds {x:e}, which {dtype:?} cannot carry as a finite value \
                     (the seam converts both operands to {dtype:?} on the host, so this element \
                     would enter the GEMM as an infinity while the f32 launcher carries it \
                     unchanged)"
                ));
            }
            if mag > peak {
                peak = mag;
            }
        }
        // `peak > 0.0` exempts an all-zero (or all-NaN) row: it converts EXACTLY, so there is
        // nothing to decline and a fallback would buy nothing.
        if peak > 0.0 && peak < floor {
            return Some(format!(
                "operand {which} row {r}'s largest magnitude is {peak:e}, under {dtype:?}'s \
                 smallest normal {floor:e} (the seam converts both operands to {dtype:?} on the \
                 host, so EVERY element of this row enters the GEMM as a subnormal or as zero — and \
                 since C[i][j] = sum_k A[i][k]*B[j][k], that row of A feeds one whole row of C and \
                 that row of B one whole column, which come back as flushed lanes while the f32 \
                 launcher carries them unchanged)"
            ));
        }
    }
    None
}

/// **Every precondition of `gpu::gemm_nt_wgmma`, restated as a decline** — the ones it asserts, and
/// the one it does not. `None` = the wgmma route may take this call; `Some(reason)` = fall back,
/// with the reason.
///
/// This exists because the launcher's preconditions are **panics**, not errors: `k >= 1`,
/// `m*n <= u32::MAX` (the epilogue forms its element index with `mad.lo.s32`) and
/// `dyn_smem_bytes <= smem_budget` all abort the process. A panic is the right answer for a launcher
/// whose caller is a bench that chose the shape; it is the wrong answer for an offload seam whose
/// caller is a user's `.wk` program, where the same shape must simply run on the other path. So the
/// same facts are decided here, first, and the launcher's asserts become unreachable rather than
/// redundant.
///
/// Two rules here are **nobody's** precondition, and they are the two that would otherwise leave
/// [`GemmRoute`]'s contract false: an odd `N` (below) and CUDA's grid ceiling. Neither the launcher
/// nor the generator checks either, and both turn into a driver failure — a sticky
/// `CUDA_ERROR_MISALIGNED_ADDRESS` and a `CUDA_ERROR_INVALID_VALUE` — on a call that had a working
/// fallback.
///
/// The tensor-map rules are **not** restated — [`ptx_wgmma::WgmmaCfg::tensor_map_a`] is asked to build
/// the very descriptor the launcher will build and
/// [`wukong_codegen_gpu::tma_host::TensorMapArgs::validate`] is asked whether it is encodable. One
/// authority. The rule that fires in practice is the row stride: a K-major NT operand has a row
/// stride of `K * 2` bytes, `cuTensorMapEncodeTiled` requires a multiple of 16, so `K % 8 != 0` is not
/// a ragged shape but an *unencodable* one. Ragged M, N and K (against the 128×256×64 tile) are all
/// fine and deliberately not declined: TMA zero-fills out-of-range elements and the epilogue
/// predicates every store on `row < M && col < N`, which `wgmma_hopper_bringup` item 7 proves exact
/// down to a 1×1 output. **An odd `N` under the v2 store is the one exception**, and it is not a
/// raggedness rule — see below.
/// CUDA's `gridDim.x` ceiling — `CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X`, `2^31 - 1` on every
/// architecture this backend supports, which is why it is a constant rather than a probe.
const MAX_GRID_X: u32 = i32::MAX as u32;

/// CUDA's `gridDim.y` / `gridDim.z` ceiling — 65535, four orders of magnitude under the x one and
/// the half that is actually reachable here, because [`ptx_wgmma::LaunchPlan::grid`] puts the M
/// tiles on y.
const MAX_GRID_YZ: u32 = 65_535;

fn wgmma_declines(
    cfg: &WgmmaCfg,
    m: usize,
    k: usize,
    n: usize,
    smem_budget: usize,
) -> Option<String> {
    // **The v2 epilogue's alignment precondition, and the sharpest decline in this function.**
    // `st.global.v2.f32` writes an 8-byte pair at `C + 4*(row*N + ctan + 2*(lane&3)) + 32j`; every
    // term but `row*N` is even by construction, so the pair is 8-byte aligned iff `N` is even. An
    // odd `N` is NOT a wrong number and NOT a ragged edge: it is `CUDA_ERROR_MISALIGNED_ADDRESS` on
    // the first store, and on this platform that leaves the context STICKILY errored, so every later
    // GPU call in the process fails identically (the generator states this at `EpilogueStore::V2`).
    //
    // It is checked here and not merely inherited because `gemm_nt_wgmma` — unlike its timing
    // sibling `time_gemm_nt_wgmma_in`, which asserts it — does NOT guard it, and both arms
    // `wgmma_cfg_for` can return carry the v2 store. Without this line the first `.wk` matmul with
    // an odd N on a Hopper part poisons the process.
    if cfg.epilogue.requires_even_n() && !n.is_multiple_of(2) {
        return Some(format!(
            "{}'s v2 epilogue needs an EVEN N and N={n} is odd (st.global.v2.f32 would be \
             misaligned on every row, a sticky context error rather than a wrong answer)",
            cfg.name
        ));
    }
    // The elided-store diagnostic arm computes a GEMM and throws the answer away, so it would return
    // the zeros the launcher allocated and call them results. `wgmma_cfg_for` cannot return it today;
    // this is here so that a later edit to the seam cannot make it reachable silently.
    if cfg.epilogue.is_diagnostic_only() {
        return Some(format!(
            "{} is the elided-store DIAGNOSTIC arm: it writes no C and would return zeros",
            cfg.name
        ));
    }
    // The epilogue's own index arithmetic, and the reason an 8192-square output is fine while a
    // 65536-square one is not.
    if (m as u64) * (n as u64) > u32::MAX as u64 {
        return Some(format!(
            "M*N = {m}*{n} overflows the u32 element index the epilogue forms with mad.lo.s32"
        ));
    }
    let plan = cfg.launch_plan();
    if plan.dyn_smem_bytes > smem_budget {
        return Some(format!(
            "{} needs {} B of dynamic shared memory, this device grants {smem_budget} B",
            cfg.name, plan.dyn_smem_bytes
        ));
    }
    // **The driver's grid ceiling, which nothing upstream checks.** `LaunchPlan::grid` indexes N
    // tiles on x and M tiles on y, and CUDA caps y and z at 65535 while x gets the full 2^31-1
    // (`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_{X,Y,Z}`, invariant across every architecture this backend
    // supports). `cluster_launch` asserts only that the grid is a multiple of the cluster, so
    // `M > 8_388_480` with an `N` small enough to keep `M*N` inside the `u32` index above reaches
    // `cuLaunchKernelEx` as `gridDim.y > 65535` and comes back `CUDA_ERROR_INVALID_VALUE` — a
    // `GpuError::Driver`, i.e. `Some(Err(..))` on a call the pre-Hopper launcher would have run.
    // It is decidable here, so it is decided here, and [`GemmRoute`]'s contract stays true.
    let grid = plan.grid(m, n);
    if grid.0 > MAX_GRID_X || grid.1 > MAX_GRID_YZ || grid.2 > MAX_GRID_YZ {
        return Some(format!(
            "{}: a {m}x{n} output needs a CTA grid of {grid:?}, past CUDA's grid ceiling \
             ({MAX_GRID_X}, {MAX_GRID_YZ}, {MAX_GRID_YZ}) — the launch would be rejected as \
             CUDA_ERROR_INVALID_VALUE, which is an error rather than a fallback",
            cfg.name
        ));
    }
    // Both descriptors, from the config that will build them, checked by the encoder's own predicate.
    // This is also what rejects `M == 0`, `N == 0` and `K == 0` (a zero global dimension), so the
    // launcher's `k >= 1` assert is covered here rather than spelled a second time.
    if let Err(e) = cfg.tensor_map_a(m, k).validate() {
        return Some(format!("A[{m}x{k}] has no encodable tensor map: {e}"));
    }
    if let Err(e) = cfg.tensor_map_b(n, k).validate() {
        return Some(format!("B[{n}x{k}] has no encodable tensor map: {e}"));
    }
    None
}

impl Accelerator for GpuAccel<'_> {
    fn sgemm_nt(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<Result<(), String>> {
        // Hopper first: this is `gemm_nt_wgmma`'s product call site (WAVE4_DOSSIER §5.0). A decline
        // falls through to the launcher every pre-Hopper device has always used.
        //
        // The two arms bump `calls` identically, which is exactly why they must also bump a counter
        // that names the route: a fallback here is invisible to `calls`, to the tolerance band and to
        // any bench reading either, so without the split an H100 round can time `gpu::gemm_nt` and
        // publish it under the wgmma family's name.
        if let Some(res) = self.try_gemm_nt_wgmma(a, b, m, k, n) {
            self.calls += 1;
            self.gemm_wgmma_calls += 1;
            return Some(res.map(|out| c.copy_from_slice(&out)));
        }
        self.calls += 1;
        self.gemm_existing_calls += 1;
        Some(
            wukong_codegen_gpu::gpu::gemm_nt(self.gpu, a, b, m, k, n)
                .map(|out| c.copy_from_slice(&out))
                .map_err(|e| format!("GPU gemm_nt failed: {e:?}")),
        )
    }

    fn vmath(&mut self, op: i64, x: &[f32], out: &mut [f32]) -> Option<Result<(), String>> {
        // The GPU vmath kernel only implements a subset of the activation op codes; an unsupported
        // one would panic, so guard with `vmath_supported` and decline (CPU fallback) otherwise.
        if !wukong_codegen_gpu::gpu::vmath_supported(op) {
            return None;
        }
        self.calls += 1;
        Some(
            wukong_codegen_gpu::gpu::vmath(self.gpu, op, x)
                .map(|res| out.copy_from_slice(&res))
                .map_err(|e| format!("GPU vmath failed: {e:?}")),
        )
    }

    fn norm(
        &mut self,
        op: i64,
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Option<Result<(), String>> {
        // Same guard as `vmath`/`sreduce`: `gpu::norm` panics on an op code it has no PTX entry for,
        // and the recognizer really does emit the two it lacks, so decline them to the CPU kernel.
        if !norm_supported(op) {
            return None;
        }
        self.calls += 1;
        Some(
            wukong_codegen_gpu::gpu::norm(self.gpu, op, x, rows, cols, eps)
                .map(|res| out.copy_from_slice(&res))
                .map_err(|e| format!("GPU norm failed: {e:?}")),
        )
    }

    fn sgemm_nt_epi(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        beta: i64,
        bias: Option<&[f32]>,
        act: i64,
    ) -> Option<Result<(), String>> {
        // The fused tensor-core kernel covers the overwrite form (`beta == 0`), the relu/gelu/silu
        // activations (ACT_RELU=1, ACT_GELU=2, ACT_SILU=3 in `wukong_runtime`; ACT_IDENTITY=0), an
        // **optional per-column bias** (`act(x·Wᵀ + bias)` — the canonical `nn.Linear`/FFN epilogue, via
        // the `_sm_db_bias*` kernels that route the tile through SMEM to add bias by explicit column),
        // and the `_sm_db` tiling's aligned shapes (M,N multiples of 64; K a multiple of 16). Anything
        // else declines to the CPU kernel. This is the fp16 tensor-core path, so it's tolerance-gated
        // against the f32 CPU oracle (the same `--backend=gpu` differential contract), not bit-exact.
        //
        // **WAVE 4'S FUSED EPILOGUE PLUGS IN HERE, AND DELIBERATELY DOES NOT YET.**
        // `docs/gpu/derive/WAVE4_DOSSIER.md` §5.5 item 3 asks for a wgmma arm in the `(bias, act)`
        // match below — no new tag, no new trait method, no `mir_build` change — and §5.1 confirms the
        // tag this hook already carries (`beta`, `bias`, `act`) is exactly the one a fused wgmma
        // kernel would need. What is missing is the KERNEL: `ptx_wgmma::WgmmaCfg` has no activation,
        // bias or output-dtype axis (dossier §5.3 rows 1-3: the config, `derived_name`, `PARAM_ORDER`
        // and the epilogue emitter), so there is no wgmma module computing `act(A·Bᵀ + bias)` to route
        // to. Composing the plain wgmma GEMM with a second pointwise launch would be precisely the
        // unfused chain the wave exists to delete, timed under the fused heading — dossier refusal 7.
        // So on Hopper this hook keeps the Act-1 wmma arms below, which are correct there
        // (`mma.sync.m16n8k16` is legal on sm_90), and R1/R2 turns this comment into the arm.
        //
        // Two of the guards below are wmma's and would be wrong for a wgmma arm, so they must not be
        // shared when it lands: the `m%64 / n%64 / k%16` alignment is a `_sm_db` tiling constraint —
        // wgmma predicates its own ragged edge and needs no alignment gate at all — and `beta != 0`
        // is a hard decline only because no fused residual kernel is wired (dossier §5.4 gap 1).
        if beta != 0 || !m.is_multiple_of(64) || !n.is_multiple_of(64) || !k.is_multiple_of(16) {
            return None;
        }
        use wukong_codegen_gpu::gpu as g;
        let res = match (bias, act) {
            // Bias-free: the activation-only fused kernels. Identity-without-bias is a plain GEMM (no
            // epilogue to fuse) — decline so the regular `sgemm_nt` path handles it.
            (None, 1) => g::gemm_nt_f16_sm_db_relu(self.gpu, a, b, m, k, n),
            (None, 2) => g::gemm_nt_f16_sm_db_gelu(self.gpu, a, b, m, k, n),
            (None, 3) => g::gemm_nt_f16_sm_db_silu(self.gpu, a, b, m, k, n),
            (None, _) => return None,
            // With bias: `act(x·Wᵀ + bias)`. ACT_IDENTITY=0 is the affine Linear `x·Wᵀ + bias`.
            (Some(bias), 0) => g::gemm_nt_f16_sm_db_bias(self.gpu, a, b, bias, m, k, n),
            (Some(bias), 1) => g::gemm_nt_f16_sm_db_bias_relu(self.gpu, a, b, bias, m, k, n),
            (Some(bias), 2) => g::gemm_nt_f16_sm_db_bias_gelu(self.gpu, a, b, bias, m, k, n),
            (Some(bias), 3) => g::gemm_nt_f16_sm_db_bias_silu(self.gpu, a, b, bias, m, k, n),
            (Some(_), _) => return None,
        };
        self.calls += 1;
        Some(
            res.map(|out| c.copy_from_slice(&out))
                .map_err(|e| format!("GPU sgemm_nt_epi failed: {e:?}")),
        )
    }

    fn sreduce(&mut self, op: i64, x: &[f32], y: &[f32]) -> Option<Result<f32, String>> {
        // GPU reduce covers sum/dot/max; decline the rest so the CPU kernel handles them.
        if !wukong_codegen_gpu::gpu::reduce_supported(op) {
            return None;
        }
        let yopt = if wukong_codegen_gpu::gpu::reduce_needs_y(op) {
            Some(y)
        } else {
            None
        };
        self.calls += 1;
        Some(
            wukong_codegen_gpu::gpu::reduce(self.gpu, op, x, yopt)
                .map_err(|e| format!("GPU reduce failed: {e:?}")),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` as the H100 probes it — the number the
    /// round logs print as `opt-in SMEM: 232448 B`.
    const H100_SMEM: usize = 232_448;
    /// The same attribute on this repo's RTX 4050, for the budget decline.
    const ADA_4050_SMEM: usize = 101_376;

    /// The gate must cover exactly the three op codes `gpu::norm` has PTX entries for. The two the
    /// recognizer also emits (LOGSOFTMAX=3, L2NORM=4) reach `panic!` inside `gpu::norm`, so they
    /// must decline here; needs no CUDA device, unlike the end-to-end fallback test.
    #[test]
    fn norm_gate_covers_only_the_implemented_ptx_entries() {
        for op in [0i64, 1, 2] {
            assert!(norm_supported(op), "norm op {op} has a PTX entry");
        }
        for op in [3i64, 4, 5, -1] {
            assert!(!norm_supported(op), "norm op {op} would panic in gpu::norm");
        }
    }

    /// The two devices this campaign actually owns, plus the two edges that are easy to get wrong.
    /// Device-free: the routing rule takes a capability, not a `Gpu`.
    #[test]
    fn the_wgmma_route_is_hopper_only() {
        assert_eq!(
            gemm_route_for((8, 9), false),
            GemmRoute::Existing,
            "the RTX 4050 this repo develops on is sm_89 and must keep the pre-Hopper launcher"
        );
        assert_eq!(
            gemm_route_for((9, 0), false),
            GemmRoute::Wgmma,
            "H100 is the whole point of the family"
        );
        assert_eq!(
            gemm_route_for((10, 0), false),
            GemmRoute::Existing,
            "sm_90a is an architecture LOCK: a Blackwell part cannot load the module, so routing it \
             there would turn a working GEMM into a decline the caller has to recover from"
        );
        for cc in [(6, 1), (7, 0), (7, 5), (8, 0), (8, 6), (8, 9), (9, 1)] {
            let want = if cc.0 == 9 {
                GemmRoute::Wgmma
            } else {
                GemmRoute::Existing
            };
            assert_eq!(gemm_route_for(cc, false), want, "cc {cc:?}");
        }
    }

    /// The kill switch is a *forcing* switch, not a hint: it must beat the capability on the one
    /// device where the capability says yes, because its whole job is a same-binary A/B.
    #[test]
    fn the_wgmma_kill_switch_beats_the_capability() {
        assert_eq!(gemm_route_for((9, 0), true), GemmRoute::Existing);
        assert_eq!(gemm_route_for((8, 9), true), GemmRoute::Existing);
    }

    /// **Eligibility is not a witness, and this is the test that says so.**
    ///
    /// `gemm_route_for` answers [`GemmRoute::Wgmma`] for *every* shape on a Hopper part — it never
    /// sees one — while `wgmma_declines` sends a real subset of them to `gpu::gemm_nt`. So a gate or
    /// a bench that reads the eligibility and concludes "the wgmma family ran" is reading a fact
    /// about the device, not about the call: it cannot tell a wgmma launch from a fallback, and on
    /// rented silicon that is measuring one kernel under the other's name. The counters can tell
    /// them apart, and this pins their algebra device-free.
    #[test]
    fn the_route_that_ran_is_a_separate_fact_from_the_route_the_device_is_eligible_for() {
        // Hopper is eligible at every shape...
        assert_eq!(gemm_route_for((9, 0), false), GemmRoute::Wgmma);
        // ...including ones the seam then declines, which is the whole gap.
        let cfg = wgmma_cfg_for(WGMMA_F32_SEAM_DTYPE, 64, 64);
        for &(m, k, n) in &[(64usize, 100usize, 64usize), (64, 64, 63)] {
            assert!(
                wgmma_declines(cfg, m, k, n, H100_SMEM).is_some(),
                "{m}x{k}x{n} must decline, or this test proves nothing"
            );
        }

        assert_eq!(
            gemm_route_taken_from(0, 0),
            None,
            "no plain GEMM was offloaded at all — neither route is a witness to anything yet"
        );
        assert_eq!(gemm_route_taken_from(4, 0), Some(GemmRoute::Wgmma));
        assert_eq!(
            gemm_route_taken_from(0, 4),
            Some(GemmRoute::Existing),
            "four fallbacks on a Hopper part: the eligibility still says Wgmma and the witness must \
             not"
        );
        assert_eq!(
            gemm_route_taken_from(1, 3),
            Some(GemmRoute::Wgmma),
            "one wgmma call puts the f16 input rounding in the output, so a mixed run is judged by \
             the wider band"
        );
    }

    /// **The driver's threshold is the launcher's conversion, asked directly.**
    ///
    /// `gemm_nt_wgmma` converts with `half::f16::from_f32` / `half::bf16::from_f32`, so the decline
    /// threshold must be that crate's own ceiling and not a literal someone typed — a `half` upgrade
    /// must not be able to move one without the other. This also pins the two facts the seam's dtype
    /// choice rests on, and the direction the threshold errs in.
    #[test]
    fn the_range_decline_matches_the_launcher_conversion() {
        let f16_max = seam_dtype_max_finite(WgmmaDtype::F16);
        assert_eq!(f16_max, 65_504.0, "f16 max finite is (2 - 2^-10)*2^15");
        assert!(half::f16::from_f32(f16_max).is_finite());
        assert!(
            half::f16::from_f32(65_520.0).is_infinite(),
            "round-to-nearest-even sends the midpoint up to 2^16, which f16 cannot hold"
        );
        assert!(
            half::f16::from_f32(65_510.0).is_finite(),
            "the threshold errs toward declining: this one is over it and would have survived"
        );

        // bf16 has f32's exponent field, so its ceiling is four decades of f32 range above f16's —
        // it is `f32::MAX` less the mantissa bits bf16 does not have, not `f32::MAX` itself.
        let bf16_max = seam_dtype_max_finite(WgmmaDtype::Bf16);
        assert!(half::bf16::from_f32(bf16_max).is_finite());
        assert!(bf16_max > 3.3e38 && bf16_max < f32::MAX);
        assert!(
            half::bf16::from_f32(f32::MAX).is_infinite(),
            "even bf16 rounds the very top of the f32 range up to infinity — which is why the bf16 \
             arm gets the same decline rather than a claim that it needs none"
        );
    }

    /// **The class change the tolerance band cannot absorb, and the values every other gate misses.**
    ///
    /// Every band gate in the repo seeds `rng.vec(.., -1.0, 1.0)`, so an operand past f16's range is
    /// never exercised by one — and it would not be a tolerance failure if it were, it would be
    /// `inf` where the f32 launcher returns a number. This is the device-free proof that such a call
    /// declines instead, that it declines on the DATA and not on the shape, and that the bf16 arm
    /// (which has f32's exponent range) declines nothing finite.
    #[test]
    fn an_operand_past_f16s_range_declines_to_the_f32_launcher() {
        let f16_max = seam_dtype_max_finite(WgmmaDtype::F16);
        // One row of K = 6 in both operands, so every element below shares a row with the peak.
        const K: usize = 6;
        let ok = [1.0f32, -1.0, 0.0, f16_max, -f16_max, 1e-9];
        assert!(
            wgmma_declines_operand_range(WgmmaDtype::F16, &ok, &ok, K).is_none(),
            "everything here is a finite f16, including the ceiling itself — and the lone 1e-9, \
             because the underflow rule is on the ROW's maximum and this row's is 65504: a single \
             flushed lane costs its own contribution to a sum its own row's 65504 also feeds, which \
             the band does cover"
        );

        for bad in [65_536.0f32, -65_536.0, 1e30, f32::MAX, f32::INFINITY] {
            let why = wgmma_declines_operand_range(WgmmaDtype::F16, &[1.0, bad], &ok, K)
                .unwrap_or_else(|| panic!("{bad:e} in A must decline"));
            assert!(why.starts_with("operand A"), "{why}");
            let why = wgmma_declines_operand_range(WgmmaDtype::F16, &ok, &[1.0, bad], K)
                .unwrap_or_else(|| panic!("{bad:e} in B must decline"));
            assert!(why.starts_with("operand B"), "{why}");
        }

        // A NaN is a NaN on both routes, so the route changes nothing and there is nothing to decline.
        assert!(wgmma_declines_operand_range(WgmmaDtype::F16, &[f32::NAN], &ok, K).is_none());

        // The bf16 arm: everything f16 refuses here is ordinary, because bf16 has f32's exponent
        // field. Only the very top of the f32 range and an infinity fail it.
        for x in [1e30f32, -1e30, 65_536.0, 1e38] {
            assert!(
                wgmma_declines_operand_range(WgmmaDtype::Bf16, &[x], &ok, K).is_none(),
                "bf16 carries f32's exponent range, so {x:e} is finite there"
            );
        }
        for x in [f32::MAX, -f32::MAX, f32::INFINITY] {
            assert!(
                wgmma_declines_operand_range(WgmmaDtype::Bf16, &[x], &ok, K).is_some(),
                "{x:e} is past bf16's own ceiling"
            );
        }

        // And it is a DATA decline, not a shape one: the shape these values ride on is fine.
        let cfg = wgmma_cfg_for(WGMMA_F32_SEAM_DTYPE, 64, 64);
        assert!(wgmma_declines(cfg, 64, 64, 64, H100_SMEM).is_none());
    }

    /// **The underflow threshold is the launcher's conversion too, asked the same way.**
    ///
    /// The ceiling's twin: `gemm_nt_wgmma` converts with `half::f16::from_f32`, so the floor must be
    /// that crate's own smallest normal and not a literal. This also pins what the conversion
    /// actually *does* below it — which is the whole reason the underflow end is a class change and
    /// not a rounding — and the direction the threshold errs in.
    #[test]
    fn the_underflow_decline_matches_the_launcher_conversion() {
        let f16_min = seam_dtype_min_positive_normal(WgmmaDtype::F16);
        assert_eq!(f16_min, 6.103_515_6e-5, "f16's smallest normal is 2^-14");
        assert!(half::f16::from_f32(f16_min).is_normal());

        // Below the floor f16 still represents, but only as subnormals — and then not at all.
        let sub = half::f16::from_f32(1e-5);
        assert!(
            !sub.is_normal() && sub.to_f32() != 0.0,
            "1e-5 is under the smallest normal: representable as a subnormal, at reduced precision"
        );
        assert_eq!(
            half::f16::from_f32(1e-8).to_f32(),
            0.0,
            "1e-8 is under 2^-25, f16's round-to-zero point: the element ENTERS the GEMM as zero"
        );
        assert_eq!(half::f16::from_f32(-1e-8).to_f32(), 0.0);
        // And the band between them keeps one or two bits, which is not a tolerance story either.
        let one_e_minus_7 = half::f16::from_f32(1e-7).to_f32();
        assert!(
            one_e_minus_7 != 0.0 && (one_e_minus_7 - 1e-7).abs() / 1e-7 > 0.15,
            "1e-7 survives as {one_e_minus_7:e}, >15% off — a different answer, not a wider bar"
        );

        // The threshold errs toward declining, exactly like the ceiling: this one is under it and
        // would still have carried real bits.
        assert!(half::f16::from_f32(5e-5).to_f32() != 0.0);

        // bf16 carries f32's exponent field at this end too, so its floor is four decades of f32
        // range BELOW f16's and the bf16 arm effectively never underflow-declines.
        let bf16_min = seam_dtype_min_positive_normal(WgmmaDtype::Bf16);
        assert!(half::bf16::from_f32(bf16_min).is_normal());
        assert!(bf16_min < 1.2e-38 && bf16_min > f32::MIN_POSITIVE * 0.9);
        assert!(bf16_min < f16_min * 1e-30);
    }

    /// **The other class change the tolerance band cannot absorb — and the one no gate could see.**
    ///
    /// Round 1 closed the overflow end; this is its twin. A row whose whole magnitude range is
    /// under f16's smallest normal converts to subnormals or to exact zeros, so the lanes of `C` it
    /// feeds come back dominated by flushed lanes — at ~1e-8, identically zero — where
    /// `gpu::gemm_nt` returns the finite matrix the CPU oracle returns. That is invisible to every
    /// gate in this repo twice over: the band gates all seed `rng.vec(.., -1.0, 1.0)` so they never
    /// produce such an operand, and `diff::assert_close` passes a lane on
    /// `abs <= abs_tol || rel <= rel_tol`, so under the wgmma band's `abs_tol = 5e-2` a zeroed
    /// 1e-7-scale output passes at 100% relative error.
    ///
    /// Device-free, on the same shape the overflow proof uses, so the decline is on the DATA.
    #[test]
    fn an_operand_wholly_under_f16s_normal_range_declines_to_the_f32_launcher() {
        // One row of K = 4 per operand, so "the row" and "the operand" coincide here and this test
        // stays about the THRESHOLD; the granularity is
        // `a_quiet_row_inside_a_loud_operand_declines_although_the_operand_peak_is_normal`.
        const K: usize = 4;
        let ok = [1.0f32, -1.0, 0.5, 0.25];

        // The concrete failure: every element flushes to zero, so C would be all zeros.
        for scale in [1e-8f32, -1e-8, 1e-7, 1e-9, 5e-5, f32::MIN_POSITIVE] {
            let tiny = [scale, scale * 0.5, -scale, scale * 0.25];
            let why = wgmma_declines_operand_range(WgmmaDtype::F16, &tiny, &ok, K)
                .unwrap_or_else(|| panic!("an A whose peak is {scale:e} must decline"));
            assert!(why.starts_with("operand A"), "{why}");
            assert!(why.contains("smallest normal"), "{why}");
            let why = wgmma_declines_operand_range(WgmmaDtype::F16, &ok, &tiny, K)
                .unwrap_or_else(|| panic!("a B whose peak is {scale:e} must decline"));
            assert!(why.starts_with("operand B"), "{why}");
        }

        // A row whose peak IS normal is not declined, however small the peak is.
        let f16_min = seam_dtype_min_positive_normal(WgmmaDtype::F16);
        for peak in [f16_min, 1e-4f32, 1e-3] {
            assert!(
                wgmma_declines_operand_range(WgmmaDtype::F16, &[peak, 1e-30], &ok, K).is_none(),
                "peak {peak:e} is normal in f16; its small lanes cost their own magnitude"
            );
        }

        // An exactly-zero row converts EXACTLY. Declining it would buy a fallback and nothing
        // else, so it is not declined — and it is the one case a naive `max < floor` gets wrong.
        assert!(
            wgmma_declines_operand_range(WgmmaDtype::F16, &[0.0, -0.0, 0.0], &ok, K).is_none(),
            "0.0 -> f16 0.0 is exact on both routes"
        );
        // An all-NaN row likewise: a NaN is a NaN on both routes and never becomes the peak.
        assert!(
            wgmma_declines_operand_range(WgmmaDtype::F16, &[f32::NAN, f32::NAN], &ok, K).is_none()
        );

        // The bf16 arm needs none of this: its floor is 1.18e-38.
        for scale in [1e-8f32, 1e-20, 1e-30] {
            assert!(
                wgmma_declines_operand_range(WgmmaDtype::Bf16, &[scale], &ok, K).is_none(),
                "bf16 carries f32's exponent range, so {scale:e} is normal there"
            );
        }
        assert!(
            wgmma_declines_operand_range(WgmmaDtype::Bf16, &[1e-40], &ok, K).is_some(),
            "1e-40 is under bf16's own smallest normal"
        );

        // And a DATA decline, not a shape one: this shape is fine.
        let cfg = wgmma_cfg_for(WGMMA_F32_SEAM_DTYPE, 64, 64);
        assert!(wgmma_declines(cfg, 64, 64, 64, H100_SMEM).is_none());
    }

    /// **The granularity, and the case a per-operand maximum structurally cannot see.**
    ///
    /// `C[i][j] = sum_k A[i][k]*B[j][k]`, so row `i` of A feeds row `i` of `C` and nothing else, and
    /// row `j` of B feeds column `j`. A single scalar peak folded over the whole operand therefore
    /// answers a question about the *matrix* when the flush that changes an answer acts on a *row*:
    /// an operand with one loud row and one quiet row has an O(1) peak, passes a per-operand rule
    /// unchanged, and returns the quiet row's entire row of `C` as exact zeros where the f32
    /// launcher — and the CPU oracle — return finite values.
    ///
    /// The reproducer is the finding's: `M=2, K=64, N=64` on Hopper, A row 0 all 1.0 and A row 1 all
    /// 1e-8, B all 1.0. Every other decline accepts it (N even, `K % 8 == 0`, `M*N = 128`, the ring
    /// fits the H100 carveout), so before this rule the route was `Wgmma` and `C[1][j]` came back 0
    /// against the oracle's 6.4e-7 — which no band gate could see, since `abs_tol = 5e-2` passes a
    /// zeroed 6.4e-7 lane at 100% relative error. Device-free, because the decline is the whole
    /// difference between the two routes.
    #[test]
    fn a_quiet_row_inside_a_loud_operand_declines_although_the_operand_peak_is_normal() {
        const M: usize = 2;
        const K: usize = 64;
        const N: usize = 64;
        let f16_min = seam_dtype_min_positive_normal(WgmmaDtype::F16);

        // The arithmetic the decline exists for: this row does not round, it VANISHES.
        assert_eq!(
            half::f16::from_f32(1e-8).to_f32(),
            0.0,
            "1e-8 is under f16's round-to-zero point, so the quiet row enters the GEMM as zeros"
        );

        let mut a = vec![1.0f32; M * K];
        a[K..].fill(1e-8);
        let b = vec![1.0f32; N * K];

        // The old rule's own input, computed here so the test states what it is replacing: the
        // operand's single peak is 1.0, comfortably normal, and blind to the row below it.
        let operand_peak = a.iter().fold(0.0f32, |p, x| p.max(x.abs()));
        assert!(
            operand_peak > f16_min,
            "the whole-operand maximum is {operand_peak:e}, which a per-operand rule passes — that \
             is exactly why this case needs a per-row one"
        );

        let why = wgmma_declines_operand_range(WgmmaDtype::F16, &a, &b, K)
            .expect("a quiet ROW inside a loud operand must decline to the f32 launcher");
        assert!(why.starts_with("operand A row 1"), "{why}");
        assert!(why.contains("smallest normal"), "{why}");

        // The B twin: a quiet row of B zeroes a whole COLUMN of C.
        let mut bq = vec![1.0f32; N * K];
        bq[3 * K..4 * K].fill(1e-8);
        let loud = vec![1.0f32; M * K];
        let why = wgmma_declines_operand_range(WgmmaDtype::F16, &loud, &bq, K)
            .expect("a quiet row of B must decline too");
        assert!(why.starts_with("operand B row 3"), "{why}");

        // The control, one row apart: the same shape with both rows loud is taken by the seam, so
        // the decline above is caused by the DATA and by nothing else.
        assert!(
            wgmma_declines_operand_range(WgmmaDtype::F16, &loud, &b, K).is_none(),
            "the control must not decline, or this test proves nothing about the quiet row"
        );
        let cfg = wgmma_cfg_for(WGMMA_F32_SEAM_DTYPE, M, N);
        assert!(
            wgmma_declines(cfg, M, K, N, H100_SMEM).is_none(),
            "M={M} K={K} N={N} is accepted by every shape decline, so this reproducer really does \
             reach the wgmma launcher on Hopper: {:?}",
            wgmma_declines(cfg, M, K, N, H100_SMEM)
        );

        // And the row rule still exempts an exactly-zero row — a zero-padded operand is the common
        // way to meet a tile, and it converts exactly on both routes.
        let mut padded = vec![1.0f32; M * K];
        padded[K..].fill(0.0);
        assert!(
            wgmma_declines_operand_range(WgmmaDtype::F16, &padded, &b, K).is_none(),
            "a zero row converts EXACTLY; declining it would buy a fallback and nothing else"
        );
    }

    /// **Why the underflow rule is on a row's maximum and the overflow rule is per element.**
    ///
    /// The asymmetry is the load-bearing design decision, and it is the reason the commit that
    /// closed the overflow end gave for leaving this one open ("a few elements of any `U(-1,1)`
    /// buffer are below 6.104e-5, so the rule would decline nearly every real GEMM"). That argument
    /// rules out a *per-element* underflow rule only. This pins both halves: a per-element rule
    /// really would fire on an ordinary buffer, and the row-max one really does not — at any row
    /// length — while still catching the wholly-flushed row a hair below the same threshold.
    #[test]
    fn the_underflow_rule_is_per_row_because_a_per_element_one_would_decline_everything() {
        let f16_min = seam_dtype_min_positive_normal(WgmmaDtype::F16);

        // A plausible `U(-1,1)` buffer: peak near 1, and one element that a per-element rule (and
        // even the round-to-zero point, 2^-25 = 2.98e-8) would refuse. `rng.vec(m*k, -1.0, 1.0)`
        // over a few million elements draws one of these with probability ~40%.
        let mut buf: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0) - 1.0).collect();
        buf[7] = 1e-9;
        buf[9] = -3e-8;
        assert!(
            buf.iter().any(|x| x.abs() > 0.9),
            "peak is O(1) by construction"
        );
        assert!(
            buf.iter().any(|x| *x != 0.0 && x.abs() < f16_min),
            "a per-element rule would have something to fire on"
        );
        // At every row length that divides it, including the ones that put the two tiny elements in
        // rows of their own company: an ordinary buffer has a normal peak in EVERY row, which is why
        // narrowing the rule from the operand to the row costs no real GEMM. (`P(max of K uniforms
        // < 6.1e-5) = (6.1e-5)^K`, about 1e-34 at K = 8.)
        for k in [2usize, 4, 8, 16, 32, 64] {
            assert!(
                wgmma_declines_operand_range(WgmmaDtype::F16, &buf, &buf, k).is_none(),
                "K={k}: the row-max rule must NOT fire on an ordinary U(-1,1) buffer, or the wgmma \
                 route declines nearly every real GEMM and the whole wave measures the f32 launcher"
            );
        }

        // One `inf` in the same buffer still declines: overflow IS per element, because an infinity
        // propagates through the accumulation into every output lane its row or column touches.
        let mut over = buf.clone();
        over[3] = 70_000.0;
        assert!(wgmma_declines_operand_range(WgmmaDtype::F16, &over, &buf, 64).is_some());

        // And the boundary of the row-max rule, from both sides, against the crate's own constant.
        let just_under = [f16_min * 0.5, f16_min * 0.25];
        let just_over = [f16_min, f16_min * 0.25];
        assert!(
            wgmma_declines_operand_range(WgmmaDtype::F16, &just_under, &just_over, 2).is_some()
        );
        assert!(wgmma_declines_operand_range(WgmmaDtype::F16, &just_over, &just_over, 2).is_none());

        // `k == 0` cannot reach the seam (a zero extent has no encodable tensor map), but the scan
        // must not panic on it either — `chunks(0)` does. It degrades to the whole operand as one
        // row, which is the conservative reading.
        assert!(wgmma_declines_operand_range(WgmmaDtype::F16, &buf, &buf, 0).is_none());
        assert!(wgmma_declines_operand_range(WgmmaDtype::F16, &[], &[], 0).is_none());
    }

    /// **The law that keeps this file honest against the generator.** `gemm_nt_wgmma`'s first
    /// statement is `require_sm90a`, so a capability this router sends to the wgmma family and the
    /// license refuses is not a fallback — it is an `Unsupported` error on a call that had a working
    /// path. Assert the two agree over the whole grid, so widening one without the other fails here
    /// rather than on rented silicon.
    #[test]
    fn the_route_never_outruns_the_sm90a_license() {
        for major in 5..=12 {
            for minor in 0..=9 {
                let cc = (major, minor);
                let licensed = ptx_wgmma::Sm90aLicense::for_probed_cc(cc).is_ok();
                assert_eq!(
                    gemm_route_for(cc, false) == GemmRoute::Wgmma,
                    licensed,
                    "cc {cc:?}: the route and the sm_90a license disagree"
                );
            }
        }
    }

    /// The seam **delegates**; it does not re-derive. Pointer identity, because the regime rule
    /// returns `&'static` rows: a driver-side copy of the 8e6-element threshold would keep dispatching
    /// on it after the generator re-measured it.
    #[test]
    fn the_config_seam_delegates_to_the_shipped_regime_rule() {
        for &(m, n) in &[(1usize, 1usize), (2048, 2048), (4096, 4096), (8192, 8192)] {
            assert!(
                std::ptr::eq(
                    wgmma_cfg_for(WgmmaDtype::F16, m, n),
                    ptx_wgmma::wgmma_w1_for(m, n)
                ),
                "f16 {m}x{n}"
            );
            assert!(
                std::ptr::eq(
                    wgmma_cfg_for(WgmmaDtype::Bf16, m, n),
                    ptx_wgmma::wgmma_w1_bf16_for(m, n)
                ),
                "bf16 {m}x{n}"
            );
        }
        // And the seam's f16 arm is what the f32 offload actually asks for.
        assert_eq!(WGMMA_F32_SEAM_DTYPE, WgmmaDtype::F16);
    }

    /// **The whole route, up to the launch, on a laptop that cannot launch it.**
    ///
    /// Everything between "a `.wk` matmul was recognized" and `cuLaunchKernelEx` is a pure function
    /// of the shape and the probed device: the route, the config the seam selects, the declines, and
    /// the [`ptx_wgmma::LaunchPlan`] that config yields. So the only part of this seam that needs
    /// Hopper is the launch itself, and this asserts the rest at `cargo test` speed for the three
    /// shapes `gpu_backend_linear_routes_through_wgmma_on_hopper` will run there.
    ///
    /// # What it is really guarding: the H100 gate covering only ONE regime arm and nobody noticing
    ///
    /// [`wgmma_cfg_for`] is a two-arm regime rule split at
    /// [`ptx_wgmma::W1_CLUSTER_MIN_OUTPUT_ELEMS`], and the two arms are **not** interchangeable at the
    /// launch: the clustered row compiles `.reqnctapercluster` into the entry, and a compiled cluster
    /// requirement the launch does not match is a *launch failure*, not a wrong number. A device gate
    /// whose shapes all sat below the threshold would therefore prove nothing about the arm that has
    /// the extra launch mechanism, while looking complete. Asserting the coverage against the
    /// generator's own constant — never a copy of `8_000_000` — means a re-measured threshold breaks
    /// this test on this laptop instead of quietly narrowing a rented-silicon round.
    #[test]
    fn the_hopper_gate_shapes_reach_both_regime_arms() {
        let mut clustered = 0usize;
        for &(m, k, n) in &HOPPER_GATE_SHAPES {
            // Both dtypes: the f32 seam asks for f16 today, and `wgmma_cfg_for` already routes bf16
            // for the day the lowp `nt_epi` symbols get an `Accelerator` hook (dossier §5.4 gap 3).
            for dtype in [WgmmaDtype::F16, WgmmaDtype::Bf16] {
                let cfg = wgmma_cfg_for(dtype, m, n);
                assert!(
                    wgmma_declines(cfg, m, k, n, H100_SMEM).is_none(),
                    "{dtype:?} {m}x{k}x{n}: the Hopper gate's own shape declines, so that gate would \
                     silently measure the pre-Hopper launcher: {:?}",
                    wgmma_declines(cfg, m, k, n, H100_SMEM)
                );
                let plan = cfg.launch_plan();
                assert!(plan.dyn_smem_bytes <= H100_SMEM);
                // No such thing as a partial cluster: every grid axis is a multiple of its cluster
                // axis, including on the shape that is ragged in M, N and K at once.
                let grid = plan.grid(m, n);
                assert_eq!(
                    (
                        grid.0 % plan.cluster.0,
                        grid.1 % plan.cluster.1,
                        grid.2 % plan.cluster.2
                    ),
                    (0, 0, 0),
                    "{}: grid {grid:?} is not a multiple of cluster {:?}",
                    cfg.name,
                    plan.cluster
                );
                assert!(grid.0 >= 1 && grid.1 >= 1 && grid.2 >= 1);
                // The regime split, stated against the generator's constant rather than its value.
                let want_cluster = m * n >= ptx_wgmma::W1_CLUSTER_MIN_OUTPUT_ELEMS;
                assert_eq!(
                    plan.cluster_ctas() > 1,
                    want_cluster,
                    "{}: M*N = {} against the shipped threshold {}",
                    cfg.name,
                    m * n,
                    ptx_wgmma::W1_CLUSTER_MIN_OUTPUT_ELEMS
                );
                if dtype == WGMMA_F32_SEAM_DTYPE && want_cluster {
                    clustered += 1;
                }
            }
        }
        assert_eq!(
            clustered,
            1,
            "exactly one Hopper gate shape must cross {} output elements — zero leaves the \
             `.reqnctapercluster` launch arm untested on the device, and all three would leave the \
             un-clustered arm untested",
            ptx_wgmma::W1_CLUSTER_MIN_OUTPUT_ELEMS
        );
    }

    /// The `#[..]` attribute lines immediately above `fn {name}(` in `src`, innermost last.
    ///
    /// A backward walk over whole lines rather than a fixed-width window, because the window form
    /// (`ptx_wgmma`'s, 220 bytes) reads the tail of the doc comment too: a gate whose prose explains
    /// *why it is no longer* `#[ignore]`d would fail its own law.
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
                attrs.push(t.to_string());
            } else {
                break;
            }
        }
        attrs.reverse();
        attrs
    }

    /// **The Hopper gate is only a gate if the invocation reaches it AND the entrypoint can fail.**
    ///
    /// Both halves have burned rented time in this repo already. The *selector* half is the
    /// 2026-08-11 vacuous bring-up (`::bench` appends `--ignored` and the gate was a plain `#[test]`,
    /// so zero tests ran and the run exited 0 for $0.006); `ptx_wgmma`'s visit constant grew a law
    /// against it, and this is that law for the driver's own gate. The *exit-status* half is
    /// narrower and worse: `modal_app.py`'s `bench()` ran its child `check=False` and then returned
    /// without a `sys.exit`, so even a selected, executed, **failing** assertion produced a green
    /// `modal run --detach`. The witness assertion this gate exists for could therefore fail on H100
    /// and be reported as a completed round.
    ///
    /// Textual and device-free, over the two files that carry the fact:
    /// [`HOPPER_GATE_INVOCATION`] names a test that exists in `lib.rs`, is not `#[ignore]`d, and is
    /// routed through the entrypoint whose selector reaches a plain `#[test]` — and that entrypoint,
    /// in `modal_app.py`, still ends in a `sys.exit`.
    #[test]
    fn the_hopper_gate_invocation_reaches_a_gate_that_can_fail() {
        let inv = HOPPER_GATE_INVOCATION;
        assert!(
            inv.is_ascii() && inv.lines().count() == 1,
            "the invocation is meant to be pasted into a shell: {inv}"
        );
        assert!(
            inv.contains("modal_app.py::test"),
            "a correctness gate belongs on ::test, the entrypoint that exits non-zero: {inv}"
        );
        assert!(
            !inv.contains("::bench") && !inv.contains("--name "),
            "`--name` is ::bench's selector and ::bench appends --ignored, which would select ZERO \
             tests here and exit 0: {inv}"
        );

        // The selector, and the test it must reach.
        let name = inv
            .split("--filter ")
            .nth(1)
            .expect("::test selects with --filter")
            .split_whitespace()
            .next()
            .expect("--filter carries no test name, only flags");
        let lib = include_str!("lib.rs");
        assert!(
            lib.contains(&format!("fn {name}(")),
            "the invocation names `{name}`, which is not a test in the driver's lib.rs"
        );
        let attrs = attributes_above(lib, name);
        assert!(
            attrs.iter().any(|a| a.starts_with("#[test]")),
            "`{name}` is named by an invocation but is not a #[test]: {attrs:?}"
        );
        assert!(
            !attrs.iter().any(|a| a.contains("ignore")),
            "`{name}` is #[ignore]d, so `::test` — which never passes --ignored — would select zero \
             tests and exit 0 having proven nothing: {attrs:?}"
        );
        // ...and it is the gate we think it is: the witness assertion, not just any test.
        assert!(
            lib.contains("(accel.gemm_wgmma_calls, accel.gemm_existing_calls),"),
            "`{name}` no longer compares the per-route counters, so the invocation is pinned to a \
             gate that cannot tell a wgmma launch from a fallback"
        );

        // The entrypoint half: a Modal function that runs a gate must be able to fail. `bench` is
        // included because it is where this gate used to be routed and where every sweep still is.
        let modal = include_str!("../../../tools/cloud/modal_app.py");
        for entry in ["def test(", "def bench("] {
            let at = modal
                .find(entry)
                .unwrap_or_else(|| panic!("`{entry}` is not in modal_app.py"));
            let body = &modal[at..];
            let end = body.find("\n@app.function").unwrap_or(body.len());
            assert!(
                body[..end].contains("sys.exit"),
                "`{entry}` runs cargo test with check=False and never exits non-zero, so a FAILED \
                 assertion inside it completes as a green `modal run` — the defect this law exists \
                 for"
            );
        }
    }

    /// **A tripwire for wave 4's G21, aimed at the one caller that cannot see it fire.**
    ///
    /// The dossier's §5.3 row 2 turns `PARAM_ORDER` from a constant into `param_order()` the moment a
    /// bias pointer exists, and `LaunchPlan::params` becomes per-variant. The driver does not build
    /// the argument array — `gpu::gemm_nt_wgmma` does — so the driver would keep compiling and keep
    /// dispatching while two configs it can return declared different signatures. Pinning "both arms
    /// of the seam declare the same list, and it is the family's" makes that a failing test in this
    /// file rather than a short argument array on rented silicon, which the driver does not report as
    /// an error at all: it reads whatever follows on the host stack as the next pointer.
    #[test]
    fn both_seam_arms_declare_one_parameter_list() {
        for dtype in [WgmmaDtype::F16, WgmmaDtype::Bf16] {
            let small = wgmma_cfg_for(dtype, 64, 64).launch_plan();
            let large = wgmma_cfg_for(dtype, 4096, 4096).launch_plan();
            assert_eq!(small.params, ptx_wgmma::PARAM_ORDER, "{}", small.entry);
            assert_eq!(large.params, ptx_wgmma::PARAM_ORDER, "{}", large.entry);
        }
    }

    /// Every `assert!` inside `gemm_nt_wgmma` must already be a `Some(..)` here, or the offload path
    /// aborts a user's program on a shape that had a working fallback.
    #[test]
    fn every_launcher_assert_is_a_driver_side_decline() {
        let cfg = wgmma_cfg_for(WgmmaDtype::F16, 4096, 4096);

        assert!(
            wgmma_declines(cfg, 4096, 4096, 4096, H100_SMEM).is_none(),
            "sq4096 is the shape the campaign measures; it must not decline"
        );
        // Ragged against the 128x256x64 tile is NOT a decline — TMA zero-fills and the epilogue
        // predicates, which bring-up item 7 proved exact down to 1x1. (Every N here is even; the odd
        // ones are the next case, and they are an alignment rule rather than a raggedness one.)
        for &(m, k, n) in &[(1usize, 8usize, 2usize), (17, 24, 34), (129, 176, 258)] {
            assert!(
                wgmma_declines(cfg, m, k, n, H100_SMEM).is_none(),
                "{m}x{k}x{n} is ragged, not unencodable"
            );
        }

        // **The one the launcher does not check.** `gemm_nt_wgmma` has no even-N assert (only its
        // timing sibling does), and BOTH rows `wgmma_cfg_for` returns carry the v2 store, so an odd
        // N reaching it is `CUDA_ERROR_MISALIGNED_ADDRESS` — which sticks to the context and fails
        // every later GPU call in the process, not just this one.
        // BOTH arms of the seam, because the small-shape row is the one a first real `.wk` matmul
        // hits and it would be the one to poison the context.
        let small = wgmma_cfg_for(WgmmaDtype::F16, 64, 64);
        for c in [cfg, small] {
            assert!(c.epilogue.requires_even_n(), "{} is a v2 row", c.name);
            for &(m, k, n) in &[(1usize, 8usize, 1usize), (129, 176, 257), (256, 64, 4095)] {
                let why = wgmma_declines(c, m, k, n, H100_SMEM).unwrap_or_else(|| {
                    panic!(
                        "{}: {m}x{k}x{n}: an odd N under a v2 store must decline",
                        c.name
                    )
                });
                assert!(why.contains("EVEN N"), "{why}");
            }
        }

        // K % 8 != 0: the NT row stride is K*2 B and a tensor map needs a multiple of 16.
        let why = wgmma_declines(cfg, 64, 100, 64, H100_SMEM).expect("K=100 has no tensor map");
        assert!(why.contains("multiple of 16"), "{why}");
        // A zero extent: the launcher's `k >= 1` assert, and its M/N twins.
        for &(m, k, n) in &[(64usize, 0usize, 64usize), (0, 64, 64), (64, 64, 0)] {
            assert!(
                wgmma_declines(cfg, m, k, n, H100_SMEM).is_some(),
                "{m}x{k}x{n} has a zero extent"
            );
        }
        // M*N past the epilogue's u32 element index.
        assert!(wgmma_declines(cfg, 65536, 64, 65536, H100_SMEM).is_some());

        // **The grid ceiling: nobody's assert, and an ERROR rather than a fallback without this.**
        // `LaunchPlan::grid` puts M tiles on y, which CUDA caps at 65535 (x gets 2^31-1, so only y
        // is reachable). `N` is small on purpose: `M*N` has to stay inside the u32 element index
        // checked above, or that decline shadows this one and the test proves nothing.
        //
        // The boundary is found by asking the PLAN, never by arithmetic on a tile size or a cluster
        // shape this test would then have to know — the clustered row rounds the y axis up to a
        // multiple of 2, and hard-coding `128 * 65535` lands on the wrong side of that.
        let gplan = cfg.launch_plan();
        let n_small = 510usize;
        let (mut lo, mut hi) = (1usize, 1usize << 30);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if gplan.grid(mid, n_small).1 <= MAX_GRID_YZ {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let (fits_m, over_m) = (lo, lo + 1);
        assert!(
            (over_m as u64) * (n_small as u64) <= u32::MAX as u64,
            "M*N = {over_m}*{n_small} must pass the u32 index check, or the grid rule is shadowed"
        );
        assert!(
            gplan.grid(over_m, n_small).1 > MAX_GRID_YZ
                && gplan.grid(fits_m, n_small).1 <= MAX_GRID_YZ,
            "the search must straddle gridDim.y, or this test is vacuous"
        );
        let why = wgmma_declines(cfg, over_m, 64, n_small, H100_SMEM)
            .expect("a grid past CUDA's gridDim.y ceiling must decline, not reach the driver");
        assert!(why.contains("grid ceiling"), "{why}");
        // ...and the tallest output whose grid fits must NOT decline: this is a ceiling, not a size
        // limit, and a rule that also refused the shapes CUDA accepts would be a silent narrowing.
        assert!(
            wgmma_declines(cfg, fits_m, 64, n_small, H100_SMEM).is_none(),
            "the largest grid CUDA accepts must still take the wgmma route: {:?}",
            wgmma_declines(cfg, fits_m, 64, n_small, H100_SMEM)
        );

        // And the budget: the ring that fits Hopper's carveout does not fit Ada's, which is the
        // assert that would fire first if the capability gate above were ever relaxed.
        let why = wgmma_declines(cfg, 4096, 4096, 4096, ADA_4050_SMEM)
            .expect("196 KiB of ring does not fit 99 KiB of carveout");
        assert!(why.contains("shared memory"), "{why}");
    }
}
