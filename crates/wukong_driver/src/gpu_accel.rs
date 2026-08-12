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
//! dishonest. `None` is reserved for shapes/ops the GPU wrappers structurally don't cover.
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

use wukong_codegen_gpu::ptx_wgmma::{self, WgmmaCfg, WgmmaDtype};
use wukong_codegen_gpu::Gpu;
use wukong_interp::Accelerator;

/// Wraps the process-wide [`Gpu`] context and routes recognized kernels to its launch wrappers.
/// `calls` counts kernels that actually ran on the device — the tolerance test asserts it is `> 0`
/// so a silent CPU fallback can never masquerade as a passing GPU run.
pub struct GpuAccel<'g> {
    pub gpu: &'g mut Gpu,
    pub calls: u32,
}

impl<'g> GpuAccel<'g> {
    /// Construct over a bound GPU context with a zeroed offload counter.
    pub fn new(gpu: &'g mut Gpu) -> Self {
        GpuAccel { gpu, calls: 0 }
    }

    /// The route this context's device takes for a recognized `C = A·Bᵀ`, including the
    /// [`WGMMA_OFF_ENV`] switch. The one place the pure decision meets the probed device, and
    /// therefore the one a device gate must ask rather than re-deriving from `cc`.
    pub(crate) fn gemm_route(&self) -> GemmRoute {
        gemm_route_for(self.gpu.target().cc(), wgmma_disabled_by_env())
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
    /// [`GpuAccel::gemm_route`] and picks its band from the route rather than from the device.
    fn try_gemm_nt_wgmma(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<Result<Vec<f32>, String>> {
        if self.gemm_route() != GemmRoute::Wgmma {
            return None;
        }
        let cfg = wgmma_cfg_for(WGMMA_F32_SEAM_DTYPE, m, n);
        // The launcher's own preconditions are asserts; decide them here so a shape it would abort on
        // simply takes the other path.
        if wgmma_declines(cfg, m, k, n, self.gpu.smem_budget()).is_some() {
            return None;
        }
        if a.len() != m * k || b.len() != n * k {
            return None;
        }
        match wukong_codegen_gpu::gpu::gemm_nt_wgmma(self.gpu, cfg, a, b, m, k, n) {
            Ok(out) => Some(Ok(out)),
            // A capability or encodability decline is the generator saying "not this shape", which is
            // a routing fact, not a failure — `UNSUPPORTED` means SKIP everywhere else in this
            // backend and it means fall-back here.
            Err(e) if e.unsupported().is_some() => None,
            Err(e) => Some(Err(format!("GPU {} failed: {e}", cfg.name))),
        }
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
/// before wave 4 and its results are unchanged, so *every* reason the wgmma family cannot take a call
/// — the architecture, the shape, an unencodable tensor map, a shared-memory budget — resolves to the
/// same working path rather than to a failure.
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
fn gemm_route_for(cc: (i32, i32), wgmma_off: bool) -> GemmRoute {
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

/// The 16-bit input type the **f32** seam feeds `wgmma`.
///
/// f16 and bf16 are the same width and eight bits apart in precision; the recognized `sgemm_nt` call
/// carries f32 operands, so the choice is purely "which 16-bit type loses least", and f16's 11
/// significand bits beat bf16's 8 at every magnitude an activation or a weight occupies. bf16 becomes
/// reachable when the lowp `wukong_sgemm_{bf16,f16}_nt_epi` symbols get an `Accelerator` hook at all
/// (WAVE4_DOSSIER §5.4 gap 3); [`wgmma_cfg_for`] already routes it.
const WGMMA_F32_SEAM_DTYPE: WgmmaDtype = WgmmaDtype::F16;

/// **Every precondition `gpu::gemm_nt_wgmma` enforces with an `assert!`, restated as a decline.**
/// `None` = the wgmma route may take this call; `Some(reason)` = fall back, with the reason.
///
/// This exists because the launcher's preconditions are **panics**, not errors: `k >= 1`,
/// `m*n <= u32::MAX` (the epilogue forms its element index with `mad.lo.s32`) and
/// `dyn_smem_bytes <= smem_budget` all abort the process. A panic is the right answer for a launcher
/// whose caller is a bench that chose the shape; it is the wrong answer for an offload seam whose
/// caller is a user's `.wk` program, where the same shape must simply run on the other path. So the
/// same facts are decided here, first, and the launcher's asserts become unreachable rather than
/// redundant.
///
/// The tensor-map rules are **not** restated — [`ptx_wgmma::WgmmaCfg::tensor_map_a`] is asked to build
/// the very descriptor the launcher will build and
/// [`wukong_codegen_gpu::tma_host::TensorMapArgs::validate`] is asked whether it is encodable. One
/// authority. The rule that fires in practice is the row stride: a K-major NT operand has a row
/// stride of `K * 2` bytes, `cuTensorMapEncodeTiled` requires a multiple of 16, so `K % 8 != 0` is not
/// a ragged shape but an *unencodable* one. Ragged M, N and K (against the 128×256×64 tile) are all
/// fine and deliberately not declined: TMA zero-fills out-of-range elements and the epilogue
/// predicates every store on `row < M && col < N`, which `wgmma_hopper_bringup` item 7 proves exact
/// down to a 1×1 output.
fn wgmma_declines(
    cfg: &WgmmaCfg,
    m: usize,
    k: usize,
    n: usize,
    smem_budget: usize,
) -> Option<String> {
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
        if let Some(res) = self.try_gemm_nt_wgmma(a, b, m, k, n) {
            self.calls += 1;
            return Some(res.map(|out| c.copy_from_slice(&out)));
        }
        self.calls += 1;
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

    /// Every `assert!` inside `gemm_nt_wgmma` must already be a `Some(..)` here, or the offload path
    /// aborts a user's program on a shape that had a working fallback.
    #[test]
    fn every_launcher_assert_is_a_driver_side_decline() {
        // The H100's `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`, and the 4050's, as probed.
        const H100_SMEM: usize = 232_448;
        const ADA_4050_SMEM: usize = 101_376;
        let cfg = wgmma_cfg_for(WgmmaDtype::F16, 4096, 4096);

        assert!(
            wgmma_declines(cfg, 4096, 4096, 4096, H100_SMEM).is_none(),
            "sq4096 is the shape the campaign measures; it must not decline"
        );
        // Ragged against the 128x256x64 tile is NOT a decline — TMA zero-fills and the epilogue
        // predicates, which bring-up item 7 proved exact down to 1x1.
        for &(m, k, n) in &[(1usize, 8usize, 1usize), (17, 24, 33), (129, 176, 257)] {
            assert!(
                wgmma_declines(cfg, m, k, n, H100_SMEM).is_none(),
                "{m}x{k}x{n} is ragged, not unencodable"
            );
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
        // And the budget: the ring that fits Hopper's carveout does not fit Ada's, which is the
        // assert that would fire first if the capability gate above were ever relaxed.
        let why = wgmma_declines(cfg, 4096, 4096, 4096, ADA_4050_SMEM)
            .expect("196 KiB of ring does not fit 99 KiB of carveout");
        assert!(why.contains("shared memory"), "{why}");
    }
}
