//! Whole-program **cooperative megakernel** (Phase 8 / M13) — the compiler-only end-to-end lever a
//! kernel library cannot pull: compile an eligible Wukong program into **one** persistent
//! `.visible .entry` kernel run by a *block* of threads, with recognized ops executed cooperatively
//! across the block (no per-op kernel launches, activations resident in one shared `.global` frame).
//!
//! ## How it works (the single-block increment)
//! [`crate::fusion::analyze`] proves a program is *megakernel-safe* (data-independent control flow, so
//! the SPMD threads never diverge and deadlock a `bar.sync`; single entry function; recognized ops
//! only). [`crate::lower::emit_mega_ptx`] then lowers the entry SPMD: the alloca frame moves to one
//! shared `.global` buffer the whole block sub-allocates, every store / side effect is `tid==0`-only,
//! and each recognized op is `bar.sync`-bracketed and run *cooperatively* — a block-wide tree for
//! reductions, or chunked over disjoint per-thread sub-ranges of the serial helper for elementwise /
//! GEMM / norm — with the remaining ops `tid==0`-serial. [`try_run`] launches it as a single block of [`MEGA_BLOCK`]
//! threads and decodes the same print/exit context buffer the single-thread path uses — so the output
//! is byte-identical, while the cooperative ops use the whole block instead of one lane.
//!
//! This is **additive and opt-in**: [`try_run`] returns `Ok(None)` for any program it can't accelerate
//! (the caller falls back to the correct single-thread `--backend=gpu-native` path — set
//! `WUKONG_GPU_DUMP_PTX` to have [`decline`] print *which* of the three reasons it was), and the existing
//! offload `--backend=gpu` path and plain `cargo test` are untouched. The megakernel is the spine the
//! op-graph fusion ([`crate::fusion`]) and the resident-chain comparison ([M13]) build on; GEMM, vmath
//! and norm now have chunked-cooperative bodies (see `lower::lower_call_mega`), so what remains for
//! later increments is cooperative bodies for the leftover ops (axpby, the bf16/f16 reductions).
//!
//! ## The multi-CTA increment (this module's launch layer)
//! One CTA is one SM. On the 20-SM 4050 that leaves 95% of the part idle; on a 132-SM H100 it is
//! under 1%, which makes "the whole program in one kernel" a claim about a single SM rather than
//! about the machine. Scaling past that needs three things, and **this module owns all three**:
//!
//!  1. [`grid_barrier_ptx`] — a **grid-wide** barrier. `bar.sync` synchronizes a CTA and nothing
//!     more; across cooperative CTAs the rendezvous is a sense-reversing counter in `.global` plus
//!     device-scope fences. Its state is a *launch parameter*, never a module-scope `.global`, so it
//!     cannot inherit a poisoned generation from the previous launch of a cached module.
//!  2. [`plan_grid`] — the **residency bound**. A cooperative grid must be simultaneously resident or
//!     the barrier deadlocks the device, so the grid is
//!     `cuOccupancyMaxActiveBlocksPerMultiprocessor x GpuTarget::sm_count` — a query and a probed
//!     device fact, never a literal. A grid that will not fit declines to `Ok(None)`.
//!  3. [`launch_mega`] — the launch itself, through `cuLaunchCooperativeKernel`
//!     (`LaunchArgs::launch_cooperative`), which is the only launch API that *guarantees* co-residency.
//!
//! **The kernel opts in through its own ABI**, read back out of the PTX by [`mega_abi`]: a
//! block-scoped entry declares `(p_ctx, p_frame)` and is launched with exactly one CTA, exactly as
//! before; a grid-parallel entry declares `(p_ctx, p_frame, p_gbar)` *and* defines
//! [`GRID_BARRIER_FN`], and is launched cooperatively across the resident grid. Neither half can
//! drift: the launcher pushes as many arguments as the entry declares, and the two markers must agree
//! or the load is a hard error. `lower::emit_mega_ptx` still emits the block-scoped form today, so the
//! observable behaviour of every corpus program is unchanged — the grid-parallel arm is exercised by
//! [`tests::grid_barrier_orders_writes_across_cooperative_ctas`], which runs the same barrier over the
//! full resident grid on the real device.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use cudarc::driver::{sys, CudaFunction, LaunchConfig, PushKernelArg};

use wukong_mir::Program;
use wukong_span::{Interner, Symbol};

use crate::lower::{self, MEGA_KERNEL_NAME};

/// The block size the megakernel launches: one CTA of this many threads. A power of two (the
/// cooperative reduction tree halves the block each step) and at most 1024 (`lower.rs`'s
/// `mrt_red_smem[1024]` is indexed by `%tid.x`, and 1024 is also the hardware ceiling on a CTA);
/// 256 is a good occupancy point. How many *CTAs* run is not a constant — see [`plan_grid`].
pub const MEGA_BLOCK: u32 = 256;

// Both halves of that sentence are load-bearing, and both are cheap to prove here rather than in a
// `CUDA_ERROR_MISALIGNED_ADDRESS` six calls deep: an odd block size makes the reduction tree drop a
// lane on its last halving, and a block above 1024 walks `mrt_red_smem` off its end.
const _: () = assert!(
    MEGA_BLOCK.is_power_of_two() && MEGA_BLOCK <= 1024,
    "MEGA_BLOCK must be a power of two <= 1024 (the reduction tree and mrt_red_smem[1024])"
);

/// The `.func` name of the grid-wide barrier ([`grid_barrier_ptx`]). Its presence in a mega module is
/// one of the two markers that make the module grid-parallel — see [`mega_abi`].
pub const GRID_BARRIER_FN: &str = "mrt_grid_barrier";

/// The mega entry's first two parameters, in order: the print/assert/exit record buffer and the
/// shared `.global` frame. [`launch_mega`] pushes them in this order, so the names are part of the
/// contract, not decoration — a rename in `lower.rs` without one here is a hard error, by design.
pub const MEGA_CTX_PARAM: &str = "p_ctx";
/// See [`MEGA_CTX_PARAM`].
pub const MEGA_FRAME_PARAM: &str = "p_frame";

/// The mega entry's third parameter: the grid-barrier state pointer. The other marker.
pub const MEGA_GBAR_PARAM: &str = "p_gbar";

/// Bytes of `.global` state one grid barrier needs: `u32` arrival counter + `u32` generation.
///
/// The host allocates it **zeroed, per launch**, and passes it in. A module-scope `.global` would be
/// the obvious alternative and is wrong here: [`crate::gpu::Gpu::function`] caches modules for the
/// life of the process, so a kernel that faulted mid-barrier would leave a non-zero counter behind
/// and hang the *next* program instead of the one with the bug.
pub const GRID_BARRIER_BYTES: usize = 8;

/// The environment override for the cooperative grid (`WUKONG_MEGA_GRID=<n>`), for SM-scaling sweeps.
/// A value above the residency bound is a decline, not a clamp: silently shrinking it would report a
/// measurement for a grid the caller did not ask for.
pub const GRID_ENV: &str = "WUKONG_MEGA_GRID";

/// **The grid-wide barrier, as PTX.** A `.func` spliced into a mega module (no header of its own —
/// the module already carries `ptx_target::HDR_SM80`, and every instruction here is Ampere-legal).
///
/// ```text
/// mrt_grid_barrier(state)   // state: .global u32[2] = { arrived, generation }
/// ```
///
/// **Why not `bar.sync`.** `bar.sync` is a CTA rendezvous; it says nothing about the other CTAs of the
/// grid. The grid-level protocol is the classic sense-reversing counter: each CTA elects thread 0,
/// which reads the current generation, atomically increments the arrival count, and then either (last
/// one in) resets the count and publishes `generation + 1`, or spins until it observes the new
/// generation. Reading the generation *before* arriving is what makes it safe to reuse: the flip
/// cannot happen until this CTA has arrived, so the value read is always the pre-flip one, and the
/// count is already back to zero before any CTA can arrive for the next round.
///
/// **Why the fences are per-thread and on both sides.** The CUDA-sanctioned shape is
/// `__syncthreads(); if (tid==0) { __threadfence(); atomic... }`, which leans on `bar.sync` to make
/// one CTA's writes visible to its own thread 0 and on that thread's fence to push them device-wide.
/// Every thread here executes `membar.gl` *before* the CTA rendezvous and again after it, which is
/// strictly stronger and removes the "does a fence by thread 0 order thread 5's stores?" question
/// entirely — for two extra fences per barrier, on a construct that is already a device-wide spin.
///
/// **Deadlock is a residency property, not a code property**: every CTA of the grid must be resident
/// simultaneously or the spin never ends. That is [`plan_grid`]'s job, and it is why a grid-parallel
/// module may only be launched through `cuLaunchCooperativeKernel`.
///
/// **No geometry precondition.** The arrival target is `nctaid.x*y*z` and the CTA's representative is
/// thread `(0,0,0)`, not `tid.x == 0` — six extra ALU instructions to make the barrier correct under
/// any launch shape instead of correct-only-if-the-caller-kept-it-1-D. A barrier whose contract is
/// "and also please never use a 2-D grid" is a barrier that eventually deadlocks a caller who did.
pub fn grid_barrier_ptx() -> String {
    format!(
        r#".func {GRID_BARRIER_FN} (.param .b64 p_bar)
{{
    .reg .b64 %rd<4>;
    .reg .b32 %r<12>;
    .reg .pred %p<4>;
    ld.param.u64 %rd0, [p_bar];
    cvta.to.global.u64 %rd1, %rd0;
    add.s64 %rd2, %rd1, 4;
    membar.gl;
    bar.sync 0;
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %tid.y;
    or.b32 %r0, %r0, %r1;
    mov.u32 %r1, %tid.z;
    or.b32 %r0, %r0, %r1;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra GB_CTA;
    ld.volatile.global.u32 %r2, [%rd2];
    mov.u32 %r3, 1;
    atom.global.add.u32 %r4, [%rd1], %r3;
    mov.u32 %r5, %nctaid.x;
    mov.u32 %r6, %nctaid.y;
    mul.lo.u32 %r5, %r5, %r6;
    mov.u32 %r6, %nctaid.z;
    mul.lo.u32 %r5, %r5, %r6;
    sub.u32 %r5, %r5, 1;
    setp.ne.u32 %p1, %r4, %r5;
    @%p1 bra GB_SPIN;
    mov.u32 %r7, 0;
    st.volatile.global.u32 [%rd1], %r7;
    membar.gl;
    add.u32 %r8, %r2, 1;
    st.volatile.global.u32 [%rd2], %r8;
    bra GB_CTA;
GB_SPIN:
    ld.volatile.global.u32 %r9, [%rd2];
    setp.eq.u32 %p2, %r9, %r2;
    @%p2 bra GB_SPIN;
GB_CTA:
    bar.sync 0;
    membar.gl;
    ret;
}}
"#
    )
}

/// How a mega PTX module wants to be launched, read out of the module text itself.
///
/// This exists because of hard rule 2 of this crate: *the argument list pushed must match the
/// kernel's declared `.param` count derived from the same source as the entry name*. Pushing short
/// makes the driver read adjacent host stack as a pointer. `lower.rs` decides the entry's shape and
/// this file decides the launch, so the launch reads the shape back out of the artifact instead of
/// assuming it — the two files cannot desync silently.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MegaAbi {
    /// The entry's `.param` names, in declaration order.
    pub params: Vec<String>,
    /// Is this module safe to run on more than one CTA? True iff it declares [`MEGA_GBAR_PARAM`]
    /// **and** defines [`GRID_BARRIER_FN`].
    pub grid_parallel: bool,
}

/// Read [`MegaAbi`] out of a mega PTX module.
///
/// Two recognized shapes, and nothing else:
///  - `(p_ctx, p_frame)` — **block-scoped**. Side effects are `tid==0`-guarded and recognized ops are
///    bracketed by `bar.sync`, both of which are per-CTA facts, so a second CTA would duplicate every
///    `print` and race the shared frame. Exactly one CTA.
///  - `(p_ctx, p_frame, p_gbar)` + a [`GRID_BARRIER_FN`] definition — **grid-parallel**. Cooperative,
///    multi-CTA, grid derived from residency.
///
/// Anything else is an `Err`, deliberately not a decline: the only way to get here is for `lower.rs`
/// and this file to disagree about the ABI, and that is precisely the failure a silent fallback would
/// hide.
pub fn mega_abi(ptx: &str) -> Result<MegaAbi, String> {
    let anchor = format!(".visible .entry {MEGA_KERNEL_NAME}");
    let at = ptx
        .find(&anchor)
        .ok_or_else(|| format!("mega PTX declares no `{anchor}`"))?;
    let after = &ptx[at + anchor.len()..];
    let open = after
        .find('(')
        .ok_or_else(|| format!("mega entry `{MEGA_KERNEL_NAME}` has no parameter list"))?;
    let close = after
        .find(')')
        .ok_or_else(|| format!("mega entry `{MEGA_KERNEL_NAME}` parameter list is unterminated"))?;
    if close < open {
        return Err(format!(
            "mega entry `{MEGA_KERNEL_NAME}` parameter list is malformed"
        ));
    }
    let params: Vec<String> = after[open + 1..close]
        .split(',')
        .map(|p| p.split_whitespace().last().unwrap_or("").to_string())
        .filter(|p| !p.is_empty())
        .collect();

    let defines_barrier = ptx.contains(&format!(".func {GRID_BARRIER_FN} "));
    let head_ok = params.len() >= 2 && params[0] == MEGA_CTX_PARAM && params[1] == MEGA_FRAME_PARAM;
    let takes_bar = params.len() == 3 && params[2] == MEGA_GBAR_PARAM;
    match (head_ok, params.len(), takes_bar, defines_barrier) {
        (true, 2, _, false) => Ok(MegaAbi {
            params,
            grid_parallel: false,
        }),
        (true, 3, true, true) => Ok(MegaAbi {
            params,
            grid_parallel: true,
        }),
        _ => Err(format!(
            "mega PTX ABI is inconsistent: entry params {params:?}, defines `{GRID_BARRIER_FN}` = \
             {defines_barrier}. The only two shapes are block-scoped `({MEGA_CTX_PARAM}, \
             {MEGA_FRAME_PARAM})` with no barrier, and grid-parallel `({MEGA_CTX_PARAM}, \
             {MEGA_FRAME_PARAM}, {MEGA_GBAR_PARAM})` WITH one. Half of a grid-parallel kernel either \
             deadlocks (a barrier no launch made resident) or duplicates every side effect (a \
             multi-CTA launch of `tid==0`-guarded stores)."
        )),
    }
}

/// The launch geometry for one mega module: how many CTAs, and by which launch API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridPlan {
    /// CTAs in the (1-D) grid.
    pub grid: u32,
    /// Threads per CTA — always [`MEGA_BLOCK`].
    pub block: u32,
    /// Must this go through `cuLaunchCooperativeKernel`? True for every `grid > 1` plan.
    pub cooperative: bool,
    /// `cuOccupancyMaxActiveBlocksPerMultiprocessor` for this entry at this block size.
    pub blocks_per_sm: u32,
    /// The probed `GpuTarget::sm_count`.
    pub sm_count: u32,
    /// `blocks_per_sm * sm_count` — the hard ceiling on a cooperative grid.
    pub max_resident: u32,
}

/// Does this device support `cuLaunchCooperativeKernel` at all?
/// (`CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`; false on very old parts and under some virtualization.)
pub fn supports_cooperative_launch(g: &crate::gpu::Gpu) -> bool {
    let dev = g.ctx.cu_device();
    let mut v: i32 = 0;
    let ok = unsafe {
        sys::cuDeviceGetAttribute(
            &mut v,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
            dev,
        )
        .result()
    };
    ok.is_ok() && v != 0
}

/// **The residency bound**: how many CTAs of `f` at `block` threads can be co-resident on this device.
///
/// Returns `(blocks_per_sm, blocks_per_sm * sm_count)`. This is the same product
/// `cuLaunchCooperativeKernel` itself checks, which is why it is computed here rather than trusted to
/// the driver's error: a plan that exceeds it must decline *before* anything is allocated or launched,
/// and having our own number lets a gate assert the two agree.
pub fn max_resident_ctas(
    g: &crate::gpu::Gpu,
    f: &CudaFunction,
    block: u32,
) -> Result<(u32, u32), String> {
    // `cuOccupancyMaxActiveBlocksPerMultiprocessor` needs a context current on THIS thread; libtest
    // runs these on many threads and `cudarc` binds inside launch/load but not inside the query.
    let _ = g.ctx.bind_to_thread();
    let per_sm = f
        .occupancy_max_active_blocks_per_multiprocessor(block, 0, None)
        .map_err(|e| format!("cuOccupancyMaxActiveBlocksPerMultiprocessor failed: {e:?}"))?;
    let sms = u32::try_from(g.target().sm_count)
        .map_err(|_| format!("probed SM count {} is not usable", g.target().sm_count))?;
    Ok((per_sm, per_sm.saturating_mul(sms)))
}

/// Derive the launch geometry for a module of the given [`MegaAbi`].
///
/// `Ok(None)` is the clean fallback — the caller declines to `Ok(None)` and the single-thread path
/// (still the correctness reference) runs the program. It is returned when the module is
/// grid-parallel but this machine cannot host it: no cooperative-launch support, an occupancy of zero
/// CTAs per SM, or an explicit [`GRID_ENV`] request above the residency bound.
///
/// A block-scoped module always plans `grid = 1` and a plain launch, which is byte-for-byte what this
/// file did before the multi-CTA work — `desired` is ignored there and saying so is the point: one
/// CTA is a *correctness* requirement of that ABI, not a tuning default.
pub fn plan_grid(
    g: &crate::gpu::Gpu,
    f: &CudaFunction,
    grid_parallel: bool,
    desired: Option<u32>,
) -> Result<Option<GridPlan>, String> {
    let (blocks_per_sm, max_resident) = max_resident_ctas(g, f, MEGA_BLOCK)?;
    let sm_count = u32::try_from(g.target().sm_count).unwrap_or(0);
    let base = GridPlan {
        grid: 1,
        block: MEGA_BLOCK,
        cooperative: false,
        blocks_per_sm,
        sm_count,
        max_resident,
    };
    if !grid_parallel {
        return Ok(Some(base));
    }
    if !supports_cooperative_launch(g) {
        return Ok(None);
    }
    if max_resident == 0 {
        return Ok(None);
    }
    let grid = match desired {
        // Above the bound the grid cannot be made resident, so the barrier would spin forever.
        // Decline rather than clamp: a clamped sweep reports a number for a grid nobody asked for.
        Some(n) if n == 0 || n > max_resident => return Ok(None),
        Some(n) => n,
        // The persistent-kernel default: exactly one full wave of CTAs, no more.
        None => max_resident,
    };
    Ok(Some(GridPlan {
        grid,
        cooperative: true,
        ..base
    }))
}

/// Parse [`GRID_ENV`], or `None` for "one full resident wave". A malformed value is an error, not a
/// silently-ignored knob.
fn env_grid() -> Result<Option<u32>, String> {
    match std::env::var(GRID_ENV) {
        Err(_) => Ok(None),
        Ok(s) => s
            .trim()
            .parse::<u32>()
            .map(Some)
            .map_err(|_| format!("{GRID_ENV}={s:?} is not a CTA count")),
    }
}

/// Report why the megakernel declined, under the existing `WUKONG_GPU_DUMP_PTX` knob, and return the
/// `Ok(None)` the caller falls back on.
///
/// All four decline paths below collapse to the same bare `Ok(None)`, so from the outside "this
/// program has data-dependent control flow", "the emitter hit an op it cannot lower" and "there is no
/// GPU in this machine" are indistinguishable — and a *silent* fallback still passes
/// `mega_corpus_matches_oracle`, because falling back to the single-thread oracle is by construction
/// correct. That is exactly how the `Op::Iota` lowering hole survived: it declined 39 corpus programs
/// on both gpu-native paths while every gate stayed green, and it only surfaced when a test asserted
/// eligibility separately. Coverage is the headline for this phase, so a coverage loss must be
/// *legible*, not merely harmless.
fn decline(why: impl std::fmt::Display) -> Result<Option<(i64, Vec<u8>)>, String> {
    if std::env::var_os("WUKONG_GPU_DUMP_PTX").is_some() {
        eprintln!("gpu-mega: declined ({why})");
    }
    Ok(None)
}

/// Try to run `program`'s `entry` as the cooperative megakernel. Returns:
///  - `Ok(Some((exit, stdout)))` — it ran on the megakernel (eligible + launched);
///  - `Ok(None)` — not megakernel-eligible, an op declined (`UNSUPPORTED:`), or no device: the caller
///    must fall back to the single-thread path (which stays the correctness reference). Set
///    `WUKONG_GPU_DUMP_PTX` to have [`decline`] name which of the three it was;
///  - `Err(_)` — a genuine JIT / launch / readback failure of an *eligible* program.
///
/// The single-thread path remains the oracle: this never changes a program's result, only how fast an
/// eligible one computes it (proven by `mega_corpus_matches_oracle`).
pub fn try_run(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> Result<Option<(i64, Vec<u8>)>, String> {
    let plan = crate::fusion::analyze(program, entry, interner);
    if !plan.eligible {
        // `MegaPlan` already carries a human-readable reason; surface it instead of dropping it.
        return decline(format_args!("ineligible: {}", plan.reason));
    }
    let Some(entry_fn) = program.function(entry) else {
        return decline(format_args!(
            "no entry function `{}`",
            interner.resolve(entry)
        ));
    };
    let frame_bytes = lower::mega_frame_bytes(entry_fn);

    let ptx = match lower::emit_mega_ptx(program, entry, interner) {
        Ok(p) => p,
        // A recognized op declined (e.g. an unsupported vmath op code) -> fall back, don't fail.
        Err(e) if e.starts_with(lower::UNSUPPORTED) => {
            return decline(format_args!("emitter declined: {e}"))
        }
        Err(e) => return Err(e),
    };

    let mut guard = crate::gpu::gpu();
    let Some(g) = guard.as_mut() else {
        // No device: let the single-thread path surface the helpful error.
        return decline("no device");
    };
    match launch_mega(g, &ptx, frame_bytes)? {
        Some(r) => Ok(Some(r)),
        // The module is grid-parallel but this machine cannot host its grid (no cooperative launch,
        // or it will not fit resident). Falling back is correct; falling back *silently* is how a
        // coverage loss hides, so it goes through `decline` like every other one.
        None => decline("cooperative grid not launchable here (see plan_grid)"),
    }
}

/// JIT + launch one mega PTX module, returning the decoded `(exit_code, stdout)` — or `Ok(None)` if
/// the module's grid cannot be made resident on this device (the caller declines).
///
/// The geometry comes from the module, not from a constant: [`mega_abi`] reads the entry's ABI back
/// out of the PTX and [`plan_grid`] turns it into a grid. A block-scoped entry runs as one CTA of
/// [`MEGA_BLOCK`] threads through a plain launch; a grid-parallel entry runs as a full resident wave
/// through `cuLaunchCooperativeKernel`, with a fresh zeroed [`GRID_BARRIER_BYTES`] state buffer.
///
/// **Launch-seam preconditions** (hard rule 2 — the argument list must match the entry's declared
/// `.param` count, derived from the same source as the entry name):
///  - the number of arguments pushed equals `abi.params.len()`, and both come from the same parse of
///    the same PTX text;
///  - a `grid > 1` plan is always cooperative — a plain multi-CTA launch of a grid-barriered kernel
///    is the deadlock this whole path exists to avoid.
///
/// The grid is emitted 1-D; the barrier does not require that (it counts `nctaid.x*y*z`), but the
/// chunked-cooperative bodies index work by a flat thread id, so a 1-D launch keeps the two aligned.
fn launch_mega(
    g: &mut crate::gpu::Gpu,
    ptx: &str,
    frame_bytes: u64,
) -> Result<Option<(i64, Vec<u8>)>, String> {
    let mut h = DefaultHasher::new();
    ptx.hash(&mut h);
    let key: &'static str = Box::leak(format!("mega_{:016x}", h.finish()).into_boxed_str());

    if std::env::var_os("WUKONG_GPU_DUMP_PTX").is_some() {
        eprintln!("--- gpu-mega PTX ---\n{ptx}\n--- end PTX ---");
    }

    let abi = mega_abi(ptx)?;
    let f = g.function(key, ptx, MEGA_KERNEL_NAME).map_err(|e| {
        let p = std::env::temp_dir().join(format!("{key}.ptx"));
        let _ = std::fs::write(&p, ptx);
        format!(
            "gpu-mega JIT/load failed: {e:?}\n  (PTX written to {})",
            p.display()
        )
    })?;
    let Some(plan) = plan_grid(g, &f, abi.grid_parallel, env_grid()?)? else {
        return Ok(None);
    };
    assert!(
        plan.grid == 1 || plan.cooperative,
        "gpu-mega: a {}-CTA grid must be launched cooperatively — a plain launch does not \
         guarantee co-residency, and a grid barrier without co-residency spins forever",
        plan.grid
    );
    if std::env::var_os("WUKONG_GPU_DUMP_PTX").is_some() {
        eprintln!(
            "gpu-mega: grid {}x{} ({}), {} blocks/SM x {} SMs = {} resident max",
            plan.grid,
            plan.block,
            if plan.cooperative {
                "cooperative"
            } else {
                "block-scoped"
            },
            plan.blocks_per_sm,
            plan.sm_count,
            plan.max_resident
        );
    }

    let host = lower::new_ctx_host();
    let mut ctx_d = g
        .stream
        .memcpy_stod(&host)
        .map_err(|e| format!("gpu-mega ctx alloc failed: {e:?}"))?;
    // One shared frame for the whole grid (>=8 bytes so a frame-less program still allocs cleanly).
    let mut frame_d = g
        .stream
        .alloc_zeros::<u8>(frame_bytes.max(8) as usize)
        .map_err(|e| format!("gpu-mega frame alloc failed: {e:?}"))?;
    // Fresh and zeroed per launch — see `GRID_BARRIER_BYTES` for why it is not a module `.global`.
    // Only for a grid-parallel entry: the block-scoped path must keep its exact per-launch driver
    // traffic, because `mega_vs_chain_reduce` measures precisely that.
    let mut bar_d = if abi.grid_parallel {
        Some(
            g.stream
                .alloc_zeros::<u8>(GRID_BARRIER_BYTES)
                .map_err(|e| format!("gpu-mega barrier alloc failed: {e:?}"))?,
        )
    } else {
        None
    };

    let cfg = LaunchConfig {
        grid_dim: (plan.grid, 1, 1),
        block_dim: (plan.block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = g.stream.launch_builder(&f);
    let mut pushed = 0usize;
    b.arg(&mut ctx_d);
    pushed += 1;
    b.arg(&mut frame_d);
    pushed += 1;
    if let Some(bar) = bar_d.as_mut() {
        b.arg(bar);
        pushed += 1;
    }
    assert_eq!(
        pushed,
        abi.params.len(),
        "gpu-mega: pushed {pushed} kernel arguments for an entry declaring {:?} — a short push makes \
         the driver read adjacent host stack as a device pointer",
        abi.params
    );
    unsafe {
        if plan.cooperative {
            b.launch_cooperative(cfg)
                .map_err(|e| format!("gpu-mega cooperative launch failed: {e:?}"))?;
        } else {
            b.launch(cfg)
                .map_err(|e| format!("gpu-mega launch failed: {e:?}"))?;
        }
    }

    let out = g
        .stream
        .memcpy_dtov(&ctx_d)
        .map_err(|e| format!("gpu-mega readback failed: {e:?}"))?;
    lower::decode_ctx(&out).map(Some)
}

// `pub(crate)` so `MEGA_CORPUS_COVERAGE_FLOOR` has exactly one definition: the device-free
// invariance gate in `lower.rs` checks the same constant this device gate ratchets on, and two
// copies of a floor is how a floor silently stops meaning anything.
#[cfg(all(test, feature = "gpu"))]
pub(crate) mod tests {
    use super::*;
    use std::path::PathBuf;

    // The corpus builder is single-sourced in `lower.rs` and normalizes away the host's 256-bit
    // raw-AVX2 loop vectorizer. That normalization is what makes `MEGA_CORPUS_COVERAGE_FLOOR` one
    // number instead of one per operating system (see `crate::lower::hostvec`), and sharing the
    // function rather than copying it is what keeps this sweep and the single-thread sweep looking
    // at the same MIR — their floors are asserted together, so a private copy here would let the
    // two silently diverge while both stayed green.
    use crate::lower::hostvec::build;

    fn line_matches(g: &str, c: &str) -> bool {
        if g == c {
            return true;
        }
        match (g.trim().parse::<f64>(), c.trim().parse::<f64>()) {
            (Ok(a), Ok(b)) => {
                let diff = (a - b).abs();
                diff <= 1e-3 || diff <= 1e-2 * b.abs().max(a.abs())
            }
            _ => false,
        }
    }

    fn outputs_match(gpu: &[u8], cpu: &[u8]) -> bool {
        let gs = String::from_utf8_lossy(gpu);
        let cs = String::from_utf8_lossy(cpu);
        let gl: Vec<&str> = gs.lines().collect();
        let cl: Vec<&str> = cs.lines().collect();
        gl.len() == cl.len() && gl.iter().zip(&cl).all(|(a, b)| line_matches(a, b))
    }

    /// The **coverage ratchet** for `mega_corpus_matches_oracle`: how many `tests/run`
    /// program-configs (one per fixture per `-O` level) must be fusion-eligible AND actually launch
    /// cooperatively AND match the interpreter oracle.
    ///
    /// This gate needs the floor even more than the single-thread one does, because a megakernel
    /// decline is *doubly* invisible: `analyze(..).eligible == false` `continue`s without a word, and
    /// a launch-time decline (`Ok(None)`) is explicitly excused as "not a miscompile". Both paths
    /// shrink the set being checked while the gate stays green. The precedent is the single-thread
    /// path's own history — the loop-vectorizer campaign silently declined 39 corpus programs on one
    /// missing `Op::Iota` arm and no gate noticed.
    ///
    /// Deliberately a **count**, not a ratio: a new fixture that is ineligible leaves `ran`
    /// unchanged, so corpus growth can never trip this. And because `ran` is a subset of `eligible`,
    /// one floor catches both regressions — an eligibility loss in `fusion::analyze` and a decline at
    /// launch.
    ///
    /// **The count is host-invariant**, which it was not until 2026-08-09. Like its single-thread
    /// sibling it is measured over MIR built with the host CPU's 256-bit raw-AVX2 loop vectorizer
    /// suppressed (see [`crate::lower::hostvec`]): that vectorizer exists only on Win64, and
    /// `lower.rs` declines the `Op::VecKernelCall` it emits, which `try_run` turns into a silent
    /// `Ok(None)`. The old floor of 87 was a Windows number — the same sweep read 90 on a Modal L4
    /// only because Linux never emitted the op, so on Linux the ratchet was vacuous.
    ///
    /// Recorded 2026-08-09 by this gate on an RTX 4050 / Windows: **90 ran / 103 eligible**
    /// program-configs. The pre-normalization Modal L4 (Linux) sweep read exactly 90/103, because
    /// Linux was already in this condition; and the device-free
    /// `lower::tests::corpus_lowering_floors_are_host_vectorizer_invariant` counts 90 mega-lowerable
    /// configs normalized against 87 host-native on this Win64 box, which is what attributes the +3
    /// to the host CPU vectorizer rather than to the megakernel. Raise it whenever coverage grows;
    /// lower it ONLY in the same commit as the intentional decline, with the reason.
    pub(crate) const MEGA_CORPUS_COVERAGE_FLOOR: usize = 90;

    /// Every **megakernel-eligible** `tests/run` program, run through the cooperative megakernel,
    /// matches the interpreter oracle (tolerance for floats, exact otherwise) at both -O0 and -O3.
    /// Ineligible programs are skipped here (the single-thread gate covers them); this proves the
    /// cooperative path is correct on the subset it accelerates — the prerequisite to wiring it into
    /// `jit_run` as the default for eligible programs.
    #[test]
    fn mega_corpus_matches_oracle() {
        if crate::gpu::device_lost() {
            // Another test in this process already hit an unrecoverable device fault (sticky CUDA
            // error). Say so loudly — this is NOT a healthy pass, and NOT "no CUDA device".
            panic!(
                "mega_corpus_matches_oracle: CUDA device lost after an earlier in-process kernel \
                 fault — fix that fault and re-run (nothing here was tested)"
            );
        }
        if crate::gpu::gpu().is_none() {
            crate::diff::skip_or_fail(
                "mega_corpus_matches_oracle",
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            );
            return;
        }
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/run");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("read tests/run")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "wk").unwrap_or(false))
            .collect();
        files.sort();

        let mut eligible = 0usize;
        let mut ran = 0usize;
        let mut mismatches: Vec<String> = Vec::new();
        // Genuine device faults (ILLEGAL_ADDRESS etc.), kept distinct from miscompiles. A fault
        // poisons the shared primary context; `crate::gpu::reset_gpu` attempts recovery, but on this
        // driver error 700 is process-fatal, so recovery degrades to record-and-skip: the remaining
        // programs land on `lost` with a loud count instead of cascading as false failures.
        let mut faults: Vec<String> = Vec::new();
        let mut lost: Vec<String> = Vec::new();

        for path in &files {
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            if crate::gpu::device_lost() {
                lost.push(name);
                continue;
            }
            let src = std::fs::read_to_string(path).unwrap();
            for opt in [0u8, 3u8] {
                let Some((program, mut interner)) = build(&src, opt) else {
                    continue;
                };
                let entry = interner.intern("main");
                if program.function(entry).is_none() {
                    continue;
                }
                if !crate::fusion::analyze(&program, entry, &interner).eligible {
                    continue;
                }
                eligible += 1;
                let cpu = wukong_interp::run_with_output(&program, entry, &interner);
                let gpu = try_run(&program, entry, &interner);
                match (&cpu, &gpu) {
                    (Err(_), Ok(None)) | (Ok(_), Ok(None)) => {} // declined at launch: not a miscompile
                    (Err(_), Err(_)) => {} // both error (e.g. assert) -> agree
                    (Ok((ce, co)), Ok(Some((ge, go)))) => {
                        ran += 1;
                        if ce != ge || !outputs_match(go, co) {
                            mismatches.push(format!(
                                "{name}@O{opt}: cpu=({ce},{:?}) mega=({ge},{:?})",
                                String::from_utf8_lossy(co),
                                String::from_utf8_lossy(go)
                            ));
                        }
                    }
                    (Ok(_), Err(e)) => {
                        // A genuine mega launch/JIT/readback fault poisons the shared context: record
                        // it on the fault ledger and attempt a reset. If the reset fails (sticky
                        // process-level error), `device_lost()` flips and the loop above skips the
                        // remaining programs loudly instead of cascading.
                        faults.push(format!("{name}@O{opt}: mega errored: {e}"));
                        crate::gpu::reset_gpu();
                    }
                    (Err(ce), Ok(Some((ge, _)))) => {
                        mismatches.push(format!("{name}@O{opt}: cpu err `{ce}` but mega ok ({ge})"))
                    }
                }
            }
        }

        eprintln!(
            "\n=== megakernel: {ran} ran / {eligible} eligible program-configs match the interp oracle ===",
        );
        // Both counts are raw — no program-config is excluded from `eligible`. What is normalized is
        // the MIR presented to the sweep, so this log is comparable line-for-line with a run on
        // another OS.
        eprintln!(
            "    (host 256-bit raw-AVX2 loop vectorizer suppressed for every build in this sweep, \
             so both counts are host-invariant; this host would otherwise emit VecKernelCall: {}. \
             See `lower::tests::corpus_lowering_floors_are_host_vectorizer_invariant`.)",
            wukong_mir::host_supports_vec_kernels()
        );
        if !faults.is_empty() {
            eprintln!(
                "-- driver FAULTS ({}, root causes only — no cascade):",
                faults.len()
            );
            for f in &faults {
                eprintln!("   {f}");
            }
        }
        if !lost.is_empty() {
            eprintln!(
                "-- NOT RUN ({}): device lost after the fault above (sticky CUDA error; restart to \
                 re-test): {}",
                lost.len(),
                lost.join(", ")
            );
        }
        assert!(
            mismatches.is_empty(),
            "megakernel disagrees with the interpreter oracle:\n{}",
            mismatches.join("\n")
        );
        assert!(
            faults.is_empty(),
            "megakernel programs faulted on the device ({} root fault(s); {} later programs were \
             skipped after device loss, NOT failed — these are genuine kernel faults, not \
             miscompiles):\n{}",
            faults.len(),
            lost.len(),
            faults.join("\n")
        );
        assert!(
            ran > 0,
            "no eligible program ran on the megakernel — pipeline broken"
        );
        assert!(
            ran >= MEGA_CORPUS_COVERAGE_FLOOR,
            "megakernel corpus coverage regressed below the recorded floor: {ran} program-configs \
             ran and matched the oracle (of {eligible} eligible), floor is \
             {MEGA_CORPUS_COVERAGE_FLOOR} ({} lost). This count is measured over \
             host-vectorizer-normalized MIR (see `crate::lower::hostvec`), so it is the SAME number \
             on Windows and on a Linux datacenter box — a shortfall here is a real regression, \
             never an OS difference. Either `fusion::analyze` stopped finding programs eligible or \
             the megakernel started declining them at launch — both are silent here. If the decline \
             is INTENTIONAL, update MEGA_CORPUS_COVERAGE_FLOOR in the SAME commit and say why in \
             its comment. If it is not, you just silently lost megakernel corpus coverage — a \
             decline is a skip here, so no other gate in this workspace would ever have told you.",
            MEGA_CORPUS_COVERAGE_FLOOR - ran
        );
    }

    /// **The `@parallel` activation shape runs cooperatively and is correct** (full output, not a
    /// checksum). At -O2 a multi-statement `@parallel` region is recognized as
    /// `wukong_vmath_f32_parallel` and inlined into `main`; `fusion::classify_call` did not know that
    /// spelling, so `analyze` reported
    /// "entry calls unrecognized `wukong_vmath_f32_parallel`" and the megakernel silently declined the
    /// very shape the user asked to parallelize. This asserts eligibility *and* that the cooperative
    /// run reproduces every printed line — exactly against the single-thread GPU lowering (same kernel,
    /// so bit-identical) and within tolerance against the interpreter oracle (an independent CPU
    /// evaluator, not a GPU-vs-GPU self-check).
    #[test]
    fn mega_parallel_activation_matches_single_and_oracle() {
        const N: usize = 1024;
        // A *multi-statement* @parallel body is what makes mir_build run the region with
        // `parallel_fn = true` and intern `wukong_vmath_f32_parallel`, inlined into `main` (the trailing
        // `tail[0] = ..` is what makes the body multi-statement; a single-statement body is outlined to
        // `wukong_parallel_for` instead, which is a different — and still ineligible — shape). x[i]
        // spans negatives, zero and positives so silu's sign/saturation regions are all covered, and 13
        // sampled outputs are printed so the comparison is over real per-element values, not a checksum.
        let src = format!(
            r#"module mega_par
@parallel
fn act(x: [f32; {N}], mut o: [f32; {N}], mut tail: [f32; 1]) {{
    for i in 0..{N} {{ o[i] = silu(x[i]); }}
    tail[0] = o[{last}];
}}
fn main() -> i32 {{
    let mut x: [f32; {N}] = [0.0; {N}];
    let mut o: [f32; {N}] = [0.0; {N}];
    let mut tail: [f32; 1] = [0.0; 1];
    for i in 0..{N} {{ x[i] = ((i as f32) - 512.0) / 128.0; }}
    act(x, o, tail);
    let mut j: i32 = 0;
    while j < 12 {{
        print((o[(j * 71) as usize] * 10000.0) as i32);
        j = j + 1;
    }}
    print((tail[0] * 10000.0) as i32);
    return 0;
}}
"#,
            last = N - 1
        );
        let (program, mut interner) = build(&src, 2).expect("frontend ok");
        let entry = interner.intern("main");
        let plan = crate::fusion::analyze(&program, entry, &interner);
        assert!(
            plan.eligible,
            "the @parallel activation shape must be megakernel-eligible, got: {}",
            plan.reason
        );
        assert!(
            plan.coop_ops
                .iter()
                .any(|c| c.name == "wukong_vmath_f32_parallel"),
            "expected the @parallel vmath twin, got {:?}",
            plan.coop_ops
                .iter()
                .map(|c| c.name.clone())
                .collect::<Vec<_>>()
        );
        let oracle = wukong_interp::run_with_output(&program, entry, &interner).expect("interp");
        if crate::gpu::gpu().is_none() {
            crate::diff::skip_or_fail(
                "mega_parallel_activation_matches_single_and_oracle",
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            );
            return;
        }
        let mega = try_run(&program, entry, &interner)
            .expect("mega run")
            .expect("mega eligible");
        let single = lower::jit_run_single(&program, entry, &interner).expect("single run");
        assert_eq!(mega.0, single.0, "exit codes differ");
        // Same PTX helper, same math: mega and single must agree byte-for-byte, no tolerance.
        assert_eq!(
            String::from_utf8_lossy(&mega.1),
            String::from_utf8_lossy(&single.1),
            "cooperative and single-thread output differ"
        );
        assert!(
            outputs_match(&mega.1, &oracle.1),
            "mega vs interpreter oracle differ:\nmega:   {}\noracle: {}",
            String::from_utf8_lossy(&mega.1).trim(),
            String::from_utf8_lossy(&oracle.1).trim()
        );
        eprintln!(
            "[gate] @parallel activation cooperative == single == oracle: {}",
            String::from_utf8_lossy(&mega.1).trim().replace('\n', " ")
        );
    }

    /// Same-run latency A/B: the cooperative megakernel vs the single-thread lowering on a
    /// reduction-heavy program (a `[N]` dot reduced `REPEAT` times). Both produce byte-identical
    /// output (checksum cross-check) before any timing counts; we then report the clock-invariant
    /// ratio (single / mega) best-of-N back-to-back in one process — never an absolute ms (≈7× clock
    /// swing on this part, per the honesty law). The cooperative tree turns each reduction's O(N)
    /// serial fold into O(N/block + log block), so the win grows with N·REPEAT.
    #[test]
    #[ignore = "perf bench; run with --ignored --nocapture"]
    fn mega_vs_single_reduce() {
        use std::time::Instant;
        if crate::gpu::gpu().is_none() {
            crate::diff::skip_or_fail(
                "mega_vs_single_reduce",
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            );
            return;
        }
        // N small enough for the single-thread .local frame, REPEAT large enough that compute
        // dominates fixed launch/alloc overhead. dot(ones,ones)=N, acc=N*REPEAT.
        const N: usize = 4096;
        const REPEAT: usize = 2000;
        let src = format!(
            r#"module bench
@parallel
fn dotp(x: [f32; {N}], y: [f32; {N}], mut o: [f32; 1]) {{
    let mut s: f32 = 0.0;
    for k in 0..{N} {{ s = s + x[k] * y[k]; }}
    o[0] = s;
}}
fn main() -> i32 {{
    let mut x: [f32; {N}] = [1.0; {N}];
    let mut y: [f32; {N}] = [1.0; {N}];
    let mut o: [f32; 1] = [0.0; 1];
    let mut acc: f32 = 0.0;
    let mut r: i32 = 0;
    while r < {REPEAT} {{ dotp(x, y, o); acc = acc + o[0]; r = r + 1; }}
    print(acc as i32);
    return 0;
}}
"#
        );
        // -O2 so the @parallel `dotp` inlines into `main` (-> main calls the recognized reduce
        // directly, making it megakernel-eligible). The opaque `wukong_sreduce_*` call has memory
        // side effects, so LICM keeps it in the loop -> the reduce work happens REPEAT times (the
        // timing below confirms it scales with REPEAT).
        let (program, mut interner) = build(&src, 2).expect("frontend ok");
        let entry = interner.intern("main");
        assert!(
            crate::fusion::analyze(&program, entry, &interner).eligible,
            "bench program must be megakernel-eligible"
        );

        // Correctness cross-check FIRST: mega and single-thread must agree (and with the oracle).
        let mega0 = try_run(&program, entry, &interner)
            .expect("mega run")
            .expect("mega eligible");
        let single0 = lower::jit_run_single(&program, entry, &interner).expect("single run");
        let oracle = wukong_interp::run_with_output(&program, entry, &interner).expect("interp");
        assert_eq!(mega0.0, single0.0, "exit codes differ");
        assert!(
            outputs_match(&mega0.1, &single0.1),
            "mega vs single output differs"
        );
        assert!(
            outputs_match(&mega0.1, &oracle.1),
            "mega vs oracle output differs"
        );
        eprintln!(
            "checksum (mega==single==oracle): {}",
            String::from_utf8_lossy(&mega0.1).trim()
        );

        // Warm the JIT/module caches for both paths, then best-of-N (smallest = least contention).
        let _ = try_run(&program, entry, &interner);
        let _ = lower::jit_run_single(&program, entry, &interner);
        let iters = 10;
        let best = |f: &dyn Fn()| {
            let mut b = f64::INFINITY;
            for _ in 0..iters {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let mega_t = best(&|| {
            let _ = try_run(&program, entry, &interner).unwrap().unwrap();
        });
        let single_t = best(&|| {
            let _ = lower::jit_run_single(&program, entry, &interner).unwrap();
        });
        eprintln!(
            "\n=== mega_vs_single_reduce  N={N} REPEAT={REPEAT} ({} reductions, {} fma) ===\n\
             single-thread: {:.3} ms   megakernel(256t): {:.3} ms   speedup: {:.2}x (same-run ratio)",
            REPEAT,
            (N * REPEAT) as f64,
            single_t * 1e3,
            mega_t * 1e3,
            single_t / mega_t,
        );
    }

    /// Same-run A/B for the **chunked-cooperative elementwise** path: a SiLU activation over `[N]`
    /// applied `REPEAT` times (a feedback store keeps the optimizer from hoisting it). Output is
    /// cross-checked mega==single==oracle (tolerance) before timing; the ratio (single/mega) is the
    /// clock-invariant win of running the elementwise kernel across the block vs one thread.
    #[test]
    #[ignore = "perf bench; run with --ignored --nocapture"]
    fn mega_vs_single_vmath() {
        use std::time::Instant;
        if crate::gpu::gpu().is_none() {
            crate::diff::skip_or_fail(
                "mega_vs_single_vmath",
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            );
            return;
        }
        const N: usize = 8192;
        const REPEAT: usize = 600;
        // o[i] = silu(x[i]) is recognized as wukong_vmath_f32; the feedback x[0]=o[0] makes the
        // REPEAT loop non-hoistable so the elementwise work runs REPEAT times.
        let src = format!(
            r#"module bench
fn main() -> i32 {{
    let mut x: [f32; {N}] = [0.5; {N}];
    let mut o: [f32; {N}] = [0.0; {N}];
    let mut r: i32 = 0;
    while r < {REPEAT} {{
        for i in 0..{N} {{ o[i] = silu(x[i]); }}
        x[0] = o[0];
        r = r + 1;
    }}
    print((o[1] * 1000.0) as i32);
    return 0;
}}
"#
        );
        let (program, mut interner) = build(&src, 2).expect("frontend ok");
        let entry = interner.intern("main");
        if !crate::fusion::analyze(&program, entry, &interner).eligible {
            crate::diff::skip_or_fail(
                "mega_vs_single_vmath",
                "the bench program is not megakernel-eligible",
            );
            return;
        }
        let mega0 = try_run(&program, entry, &interner)
            .expect("mega")
            .expect("eligible");
        let single0 = lower::jit_run_single(&program, entry, &interner).expect("single");
        let oracle = wukong_interp::run_with_output(&program, entry, &interner).expect("interp");
        assert!(outputs_match(&mega0.1, &single0.1), "mega vs single differ");
        assert!(outputs_match(&mega0.1, &oracle.1), "mega vs oracle differ");
        eprintln!(
            "checksum (mega==single==oracle): {}",
            String::from_utf8_lossy(&mega0.1).trim()
        );

        let _ = try_run(&program, entry, &interner);
        let _ = lower::jit_run_single(&program, entry, &interner);
        let best = |f: &dyn Fn()| {
            let mut b = f64::INFINITY;
            for _ in 0..10 {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let mega_t = best(&|| {
            let _ = try_run(&program, entry, &interner).unwrap().unwrap();
        });
        let single_t = best(&|| {
            let _ = lower::jit_run_single(&program, entry, &interner).unwrap();
        });
        eprintln!(
            "\n=== mega_vs_single_vmath  N={N} REPEAT={REPEAT} ({} silu) ===\n\
             single-thread: {:.3} ms   megakernel(256t): {:.3} ms   speedup: {:.2}x (same-run ratio)",
            (N * REPEAT) as f64,
            single_t * 1e3,
            mega_t * 1e3,
            single_t / mega_t,
        );
    }

    /// Same-run A/B for the **row-chunked-cooperative GEMM** path: `C[M,N] = A[M,K]·Bᵀ` reduced
    /// `REPEAT` times. The megakernel partitions the M output rows across the block (each thread runs
    /// the serial `mrt_sgemm_nt` over its rows); the single-thread path does all M rows in one lane.
    /// Output cross-checked mega==single==oracle before the ratio is reported.
    #[test]
    #[ignore = "perf bench; run with --ignored --nocapture"]
    fn mega_vs_single_gemm() {
        use std::time::Instant;
        if crate::gpu::gpu().is_none() {
            crate::diff::skip_or_fail(
                "mega_vs_single_gemm",
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            );
            return;
        }
        // M rows chunked across the 256-thread block; K/N kept modest so the single-thread .local
        // frame fits. Constant inputs -> out[i,j]=K, acc=REPEAT*K exactly (no precision drift).
        const M: usize = 256;
        const K: usize = 64;
        const N: usize = 64;
        const REPEAT: usize = 16;
        let src = format!(
            r#"module bench
fn linear(x: [f32; {mk}], w: [f32; {nk}], mut out: [f32; {mn}]) {{
    for i in 0..{M} {{
        for j in 0..{N} {{
            let mut s: f32 = 0.0;
            for p in 0..{K} {{ s = s + x[i * {K} + p] * w[j * {K} + p]; }}
            out[i * {N} + j] = s;
        }}
    }}
}}
fn main() -> i32 {{
    let x: [f32; {mk}] = [1.0; {mk}];
    let w: [f32; {nk}] = [1.0; {nk}];
    let mut out: [f32; {mn}] = [0.0; {mn}];
    let mut acc: f32 = 0.0;
    let mut r: i32 = 0;
    while r < {REPEAT} {{ linear(x, w, out); acc = acc + out[0]; r = r + 1; }}
    print(acc as i32);
    return 0;
}}
"#,
            mk = M * K,
            nk = N * K,
            mn = M * N,
        );
        let (program, mut interner) = build(&src, 2).expect("frontend ok");
        let entry = interner.intern("main");
        if !crate::fusion::analyze(&program, entry, &interner).eligible {
            crate::diff::skip_or_fail(
                "mega_vs_single_gemm",
                "the bench program is not megakernel-eligible",
            );
            return;
        }
        let mega0 = try_run(&program, entry, &interner)
            .expect("mega")
            .expect("eligible");
        let single0 = lower::jit_run_single(&program, entry, &interner).expect("single");
        let oracle = wukong_interp::run_with_output(&program, entry, &interner).expect("interp");
        assert!(outputs_match(&mega0.1, &single0.1), "mega vs single differ");
        assert!(outputs_match(&mega0.1, &oracle.1), "mega vs oracle differ");
        eprintln!(
            "checksum (mega==single==oracle): {}",
            String::from_utf8_lossy(&mega0.1).trim()
        );

        let _ = try_run(&program, entry, &interner);
        let _ = lower::jit_run_single(&program, entry, &interner);
        let best = |f: &dyn Fn()| {
            let mut b = f64::INFINITY;
            for _ in 0..8 {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let mega_t = best(&|| {
            let _ = try_run(&program, entry, &interner).unwrap().unwrap();
        });
        let single_t = best(&|| {
            let _ = lower::jit_run_single(&program, entry, &interner).unwrap();
        });
        eprintln!(
            "\n=== mega_vs_single_gemm  M={M} K={K} N={N} REPEAT={REPEAT} ({} fma) ===\n\
             single-thread: {:.3} ms   megakernel(256t): {:.3} ms   speedup: {:.2}x (same-run ratio)",
            (M * K * N * REPEAT) as f64,
            single_t * 1e3,
            mega_t * 1e3,
            single_t / mega_t,
        );
    }

    /// **M13 — the megakernel's structural win: one launch vs the per-op library chain.** A workload
    /// of `REPEAT` reductions is run two ways, same-run, same launcher (`try_run`), so the only
    /// difference is launch structure:
    ///  - **megakernel**: the whole `REPEAT`-loop program in ONE launch, data resident in the shared
    ///    frame across every reduction (zero per-op launches / re-marshaling);
    ///  - **per-op chain (the offload / library model)**: a single-reduction program launched `REPEAT`
    ///    times — each op its own launch that re-materializes its inputs, exactly what the
    ///    `--backend=gpu` `GpuAccel` path does (per-call H2D + launch + D2H, no residency).
    ///
    /// The ratio (chain / mega) is the launch-overhead + residency win a kernel library *cannot* get
    /// without fusing the whole graph (Mirage/FlashFormer-class). Reported as a clock-invariant ratio
    /// only; correctness is the loop program's `acc = REPEAT*N` vs the chain's per-launch `N` summed.
    #[test]
    #[ignore = "perf bench (M13); run with --ignored --nocapture"]
    fn mega_vs_chain_reduce() {
        use std::time::Instant;
        if crate::gpu::gpu().is_none() {
            crate::diff::skip_or_fail(
                "mega_vs_chain_reduce",
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            );
            return;
        }
        const N: usize = 4096;
        const REPEAT: usize = 400;
        // Loop program: the whole chain in one megakernel launch. acc = REPEAT*N (dot of ones).
        let loop_src = format!(
            r#"module bench
@parallel
fn dotp(x: [f32; {N}], y: [f32; {N}], mut o: [f32; 1]) {{
    let mut s: f32 = 0.0;
    for k in 0..{N} {{ s = s + x[k] * y[k]; }}
    o[0] = s;
}}
fn main() -> i32 {{
    let mut x: [f32; {N}] = [1.0; {N}];
    let mut y: [f32; {N}] = [1.0; {N}];
    let mut o: [f32; 1] = [0.0; 1];
    let mut acc: f32 = 0.0;
    let mut r: i32 = 0;
    while r < {REPEAT} {{ dotp(x, y, o); acc = acc + o[0]; r = r + 1; }}
    print(acc as i32);
    return 0;
}}
"#
        );
        // Single-op program: one reduction per launch (the per-op chain element). prints N.
        let one_src = format!(
            r#"module bench
@parallel
fn dotp(x: [f32; {N}], y: [f32; {N}], mut o: [f32; 1]) {{
    let mut s: f32 = 0.0;
    for k in 0..{N} {{ s = s + x[k] * y[k]; }}
    o[0] = s;
}}
fn main() -> i32 {{
    let mut x: [f32; {N}] = [1.0; {N}];
    let mut y: [f32; {N}] = [1.0; {N}];
    let mut o: [f32; 1] = [0.0; 1];
    dotp(x, y, o);
    print(o[0] as i32);
    return 0;
}}
"#
        );
        let (loop_p, mut li) = build(&loop_src, 2).expect("frontend");
        let loop_e = li.intern("main");
        let (one_p, mut oi) = build(&one_src, 2).expect("frontend");
        let one_e = oi.intern("main");
        assert!(crate::fusion::analyze(&loop_p, loop_e, &li).eligible);
        assert!(crate::fusion::analyze(&one_p, one_e, &oi).eligible);

        // Correctness: megakernel computes the whole chain (acc=REPEAT*N); each chain element computes
        // N; the two agree when the chain elements are summed -> the megakernel fused them losslessly.
        let mega0 = try_run(&loop_p, loop_e, &li)
            .expect("mega")
            .expect("eligible");
        let one0 = try_run(&one_p, one_e, &oi).expect("one").expect("eligible");
        let mega_val: i64 = String::from_utf8_lossy(&mega0.1).trim().parse().unwrap();
        let one_val: i64 = String::from_utf8_lossy(&one0.1).trim().parse().unwrap();
        assert_eq!(
            mega_val,
            one_val * REPEAT as i64,
            "mega chain != sum of per-op results"
        );
        eprintln!("checksum: megakernel acc={mega_val} == {REPEAT} * per-op {one_val}");

        // Warm both, then best-of-N.
        let _ = try_run(&loop_p, loop_e, &li);
        let _ = try_run(&one_p, one_e, &oi);
        let best = |f: &dyn Fn()| {
            let mut b = f64::INFINITY;
            for _ in 0..6 {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let mega_t = best(&|| {
            let _ = try_run(&loop_p, loop_e, &li).unwrap().unwrap();
        });
        // The per-op chain: REPEAT separate launches, each re-marshaling its inputs (offload model).
        let chain_t = best(&|| {
            for _ in 0..REPEAT {
                let _ = try_run(&one_p, one_e, &oi).unwrap().unwrap();
            }
        });
        eprintln!(
            "\n=== M13 mega_vs_chain_reduce  N={N} REPEAT={REPEAT} ===\n\
             per-op chain ({REPEAT} launches): {:.3} ms   megakernel (1 launch): {:.3} ms   \
             win: {:.2}x (same-run ratio)",
            chain_t * 1e3,
            mega_t * 1e3,
            chain_t / mega_t,
        );
    }

    // ==========================================================================================
    // Multi-CTA: the grid barrier, the residency bound, and the ABI seam between this file and
    // `lower.rs`. Everything below is about *how* the megakernel is launched, not what it computes.
    // ==========================================================================================

    /// Rounds the grid-barrier probe runs. Two barriers per round, so 32 rounds is 64 grid
    /// rendezvous — enough that a barrier which only works once (a counter reset with no generation
    /// flip) fails on round two rather than passing by luck.
    const PROBE_ROUNDS: u64 = 32;

    /// A **self-contained cooperative kernel that can only pass if the grid barrier really works.**
    ///
    /// Each round `r`, CTA `b` writes `buf[b] = b*1000 + r + 1`, rendezvouses grid-wide, then reads
    /// **every** slot `buf[c]` and checks it against the exact value CTA `c` must have written *this
    /// round*. A second barrier closes the round so the next write cannot race the read. `out[b]`
    /// accumulates mismatches and `out[G+b]` keeps the last value read, so a vacuous pass (nothing
    /// ran, everything zero) is distinguishable from a real one.
    ///
    /// It is a **per-lane** check, deliberately not a checksum: a sum over the slots could cancel a
    /// CTA that is one round behind against one that is one round ahead, which is exactly the state a
    /// half-working barrier produces. Every slot is compared to its own closed form instead, so the
    /// mismatch count is the number of CTAs that were out of step, per round.
    ///
    /// A `bar.sync` in place of the grid barrier cannot pass this: it orders CTA `b` against itself
    /// and says nothing about the other `G-1` CTAs, so the scan lands on stale or unwritten slots.
    fn gridbar_probe_ptx() -> String {
        format!(
            "{}{}{}",
            crate::ptx_target::HDR_SM80,
            grid_barrier_ptx(),
            r#"
.visible .entry wk_gridbar_probe (
    .param .u64 p_buf,
    .param .u64 p_out,
    .param .u64 p_bar,
    .param .u64 p_rounds
)
{
    .reg .b64 %rd<12>;
    .reg .b32 %r<12>;
    .reg .pred %p<4>;
    ld.param.u64 %rd0, [p_buf];
    ld.param.u64 %rd1, [p_out];
    ld.param.u64 %rd2, [p_bar];
    ld.param.u64 %rd3, [p_rounds];
    cvta.to.global.u64 %rd4, %rd0;
    cvta.to.global.u64 %rd5, %rd1;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %nctaid.x;
    mov.u32 %r2, %tid.x;
    mul.wide.u32 %rd6, %r0, 4;
    add.s64 %rd6, %rd4, %rd6;
    mul.wide.u32 %rd8, %r0, 4;
    add.s64 %rd8, %rd5, %rd8;
    add.u32 %r4, %r1, %r0;
    mul.wide.u32 %rd9, %r4, 4;
    add.s64 %rd9, %rd5, %rd9;
    mov.u32 %r5, 0;
    mov.u32 %r6, 0;
    mov.b64 %rd10, 0;
    setp.ne.u32 %p1, %r2, 0;
GBP_LOOP:
    setp.ge.s64 %p0, %rd10, %rd3;
    @%p0 bra GBP_DONE;
    @%p1 bra GBP_W;
    cvt.u32.u64 %r7, %rd10;
    mul.lo.u32 %r8, %r0, 1000;
    add.u32 %r8, %r8, %r7;
    add.u32 %r8, %r8, 1;
    st.global.u32 [%rd6], %r8;
GBP_W:
    {
    .param .b64 _b0;
    st.param.b64 [_b0], %rd2;
    call.uni mrt_grid_barrier, (_b0);
    }
    @%p1 bra GBP_R;
    mov.u32 %r3, 0;
    mov.u64 %rd7, %rd4;
GBP_SCAN:
    setp.ge.u32 %p2, %r3, %r1;
    @%p2 bra GBP_R;
    ld.global.u32 %r9, [%rd7];
    cvt.u32.u64 %r7, %rd10;
    mul.lo.u32 %r10, %r3, 1000;
    add.u32 %r10, %r10, %r7;
    add.u32 %r10, %r10, 1;
    setp.ne.u32 %p3, %r9, %r10;
    @%p3 add.u32 %r5, %r5, 1;
    mov.u32 %r6, %r9;
    add.u32 %r3, %r3, 1;
    add.s64 %rd7, %rd7, 4;
    bra GBP_SCAN;
GBP_R:
    {
    .param .b64 _b1;
    st.param.b64 [_b1], %rd2;
    call.uni mrt_grid_barrier, (_b1);
    }
    add.s64 %rd10, %rd10, 1;
    bra GBP_LOOP;
GBP_DONE:
    @%p1 bra GBP_END;
    st.global.u32 [%rd8], %r5;
    st.global.u32 [%rd9], %r6;
GBP_END:
    ret;
}
"#
        )
    }

    /// **The barrier's ASCII + shape law** (crate hard rule 1: one non-ASCII byte anywhere in a PTX
    /// string is a `ptxas fatal` at `cuModuleLoadData`, and the Rust prose around this generator is
    /// full of arrows and multiplication signs). Also pins the pieces the protocol cannot lose: the
    /// two `bar.sync`es that bracket the grid phase, the three device-scope fences, the arrival
    /// atomic, the CTA count it compares against, and the fact that it is a spliceable `.func`
    /// fragment with no `.version` header of its own.
    #[test]
    fn grid_barrier_ptx_is_pure_ascii_and_ampere_legal() {
        let p = grid_barrier_ptx();
        assert!(
            p.is_ascii(),
            "grid barrier PTX must be pure ASCII: {:?}",
            p.chars().find(|c| !c.is_ascii())
        );
        assert!(!p.contains('\t'), "PTX must not contain tabs");
        assert!(p.starts_with(&format!(".func {GRID_BARRIER_FN} (.param .b64 p_bar)")));
        assert!(
            !p.contains(".version") && !p.contains(".target"),
            "the barrier is a fragment spliced into a module that already carries HDR_SM80"
        );
        assert_eq!(
            p.matches("bar.sync 0;").count(),
            2,
            "the CTA rendezvous brackets the grid phase on both sides"
        );
        assert_eq!(
            p.matches("membar.gl;").count(),
            3,
            "release fence, the last-CTA reset fence, and the acquire fence"
        );
        assert!(p.contains("atom.global.add.u32"));
        // The arrival target is the WHOLE grid and the representative is thread (0,0,0), so the
        // barrier carries no "must be launched 1-D" precondition for a caller to violate.
        for d in ["%nctaid.x", "%nctaid.y", "%nctaid.z", "%tid.y", "%tid.z"] {
            assert!(p.contains(d), "barrier must be geometry-agnostic: missing {d}");
        }
        assert_eq!(
            p.matches("ld.volatile.global.u32").count(),
            2,
            "the generation read and the spin load must both be volatile or the JIT hoists them"
        );
        // Nothing here postdates PTX ISA 7.8 / sm_80, which is what lets the fragment sit in an
        // `HDR_SM80` module (see `ptx_target`, and gpu.rs's `.version` law).
        for above_78 in ["wgmma", "stmatrix", "elect.sync", "cp.async.bulk", "tcgen05"] {
            assert!(!p.contains(above_78), "{above_78} would raise the ISA floor");
        }
    }

    /// **The ABI seam has exactly two legal shapes, and half of one is an error.**
    ///
    /// `lower.rs` decides the mega entry's parameter list and this file decides the launch. Hard rule
    /// 2 says the argument list pushed must be derived from the same source as the entry name, so the
    /// launcher parses the shape back out of the PTX; this pins that parse, including the two ways
    /// the halves can drift — a barrier parameter with no barrier `.func` (deadlock: a rendezvous no
    /// launch made resident) and a barrier `.func` with no parameter (duplicated side effects, since
    /// the entry's `tid==0` guards are per-CTA).
    #[test]
    fn mega_abi_recognizes_exactly_two_shapes() {
        let block = format!(".visible .entry {MEGA_KERNEL_NAME}(.param .u64 p_ctx, .param .u64 p_frame)\n{{\nret;\n}}\n");
        let a = mega_abi(&block).expect("block-scoped ABI");
        assert_eq!(a.params, ["p_ctx", "p_frame"]);
        assert!(!a.grid_parallel);

        let entry3 = format!(
            ".visible .entry {MEGA_KERNEL_NAME}(.param .u64 p_ctx, .param .u64 p_frame, \
             .param .u64 {MEGA_GBAR_PARAM})\n{{\nret;\n}}\n"
        );
        let grid = format!("{}{}", grid_barrier_ptx(), entry3);
        let b = mega_abi(&grid).expect("grid-parallel ABI");
        assert_eq!(b.params, ["p_ctx", "p_frame", MEGA_GBAR_PARAM]);
        assert!(b.grid_parallel);

        // Half a grid-parallel kernel, both directions.
        assert!(mega_abi(&entry3).is_err());
        let func_no_param = format!("{}{}", grid_barrier_ptx(), block);
        assert!(mega_abi(&func_no_param).is_err());

        // The first two names are the contract, not decoration: the launcher pushes ctx then frame
        // and nothing downstream would notice them swapped or renamed.
        let renamed = format!(
            ".visible .entry {MEGA_KERNEL_NAME}(.param .u64 p_frame, .param .u64 p_ctx)\n{{\nret;\n}}\n"
        );
        assert!(mega_abi(&renamed).is_err());

        assert!(mega_abi("nothing here").is_err());
    }

    /// **What `lower::emit_mega_ptx` actually emits agrees with what the launcher pushes.**
    ///
    /// The ratchet on the seam above: it reads a real emitted module rather than a hand-written
    /// string, so the day `lower.rs` grows the third parameter this test sees it, and it fails if
    /// only one of the two markers lands. It deliberately accepts *either* shape — the block-scoped
    /// one is today's, the grid-parallel one is the follow-up — and pins the invariant that binds
    /// them: barrier parameter iff barrier `.func`, and one CTA iff block-scoped.
    #[test]
    fn emitted_mega_abi_agrees_with_the_launcher() {
        // The same `@parallel` shape `mega_parallel_activation_matches_single_and_oracle` pins as
        // eligible (multi-statement body -> `wukong_vmath_f32_parallel`, inlined into `main` at -O2).
        let src = r#"module abi
@parallel
fn act(x: [f32; 256], mut o: [f32; 256], mut tail: [f32; 1]) {
    for i in 0..256 { o[i] = silu(x[i]); }
    tail[0] = o[255];
}
fn main() -> i32 {
    let mut x: [f32; 256] = [0.5; 256];
    let mut o: [f32; 256] = [0.0; 256];
    let mut tail: [f32; 1] = [0.0; 1];
    act(x, o, tail);
    print((tail[0] * 1000.0) as i32);
    return 0;
}
"#;
        let (program, mut interner) = build(src, 2).expect("frontend ok");
        let entry = interner.intern("main");
        assert!(
            crate::fusion::analyze(&program, entry, &interner).eligible,
            "the ABI probe program must be megakernel-eligible"
        );
        let ptx = lower::emit_mega_ptx(&program, entry, &interner).expect("mega ptx");
        let abi = mega_abi(&ptx).expect("emitted mega PTX must have a recognized ABI");
        assert_eq!(
            abi.grid_parallel,
            ptx.contains(&format!(".func {GRID_BARRIER_FN} ")),
            "the two grid-parallel markers must agree"
        );
        assert_eq!(abi.params[0], "p_ctx");
        assert_eq!(abi.params[1], "p_frame");
        assert_eq!(
            abi.params.len(),
            if abi.grid_parallel { 3 } else { 2 },
            "the launcher pushes ctx + frame (+ barrier state when grid-parallel)"
        );
        eprintln!(
            "[gate] emit_mega_ptx ABI = {:?} (grid_parallel = {})",
            abi.params, abi.grid_parallel
        );
    }

    /// **The grid-wide barrier really orders writes across cooperative CTAs, on the real device.**
    ///
    /// This is the correctness risk the whole multi-CTA increment turns on: `bar.sync` synchronizes a
    /// CTA, a grid barrier is a different construct, and getting it wrong deadlocks the part. The
    /// probe ([`gridbar_probe_ptx`]) makes every CTA read values only its *peers* can have written
    /// *this round*, over the **full resident grid** derived from occupancy — 20 SMs is a small
    /// machine but a genuinely multi-CTA one, and a broken barrier reads the wrong value here exactly
    /// as it would on 132 SMs.
    ///
    /// **The probe is measured to be sensitive, not assumed to be.** Replacing the grid barrier's
    /// body with a bare `bar.sync 0; ret;` (a CTA rendezvous and nothing more — the construct this
    /// whole increment exists to replace) and re-running on this 4050 reports *120 of 120 CTAs bad,
    /// 426961 stale slot-reads of 460800*. So a pass here is a statement about the grid protocol, not
    /// about 120 CTAs happening to run in lockstep.
    #[test]
    fn grid_barrier_orders_writes_across_cooperative_ctas() {
        let mut guard = crate::gpu::gpu();
        let Some(g) = guard.as_mut() else {
            crate::diff::skip_or_fail(
                "grid_barrier_orders_writes_across_cooperative_ctas",
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            );
            return;
        };
        if !supports_cooperative_launch(g) {
            crate::diff::skip_or_fail(
                "grid_barrier_orders_writes_across_cooperative_ctas",
                "device reports no CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH",
            );
            return;
        }
        let ptx = gridbar_probe_ptx();
        assert!(ptx.is_ascii(), "probe module must be pure ASCII");
        assert_eq!(
            ptx.matches(&format!("call.uni {GRID_BARRIER_FN},")).count(),
            2,
            "the probe must call the barrier twice per round"
        );
        let f = g
            .function("mega_gridbar_probe_v1", &ptx, "wk_gridbar_probe")
            .expect("gridbar probe JIT");
        let plan = plan_grid(g, &f, true, None)
            .expect("plan")
            .expect("a cooperative grid must be plannable on this device");
        assert!(plan.cooperative);
        assert_eq!(plan.grid, plan.max_resident);
        assert!(
            plan.grid > 1,
            "a one-CTA grid would make this test vacuous (blocks/SM {} x SMs {})",
            plan.blocks_per_sm,
            plan.sm_count
        );
        let gsz = plan.grid as usize;

        let mut buf = g.stream.alloc_zeros::<u32>(gsz).expect("buf");
        let mut out = g.stream.alloc_zeros::<u32>(2 * gsz).expect("out");
        let mut bar = g
            .stream
            .alloc_zeros::<u8>(GRID_BARRIER_BYTES)
            .expect("barrier state");
        let rounds: u64 = PROBE_ROUNDS;
        let cfg = LaunchConfig {
            grid_dim: (plan.grid, 1, 1),
            block_dim: (plan.block, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut b = g.stream.launch_builder(&f);
            b.arg(&mut buf);
            b.arg(&mut out);
            b.arg(&mut bar);
            b.arg(&rounds);
            unsafe {
                b.launch_cooperative(cfg)
                    .expect("cooperative launch of the grid-barrier probe");
            }
        }
        let host = g.stream.memcpy_dtov(&out).expect("readback");
        let mismatches: u32 = host[..gsz].iter().sum();
        assert_eq!(
            mismatches,
            0,
            "grid barrier did not order cross-CTA writes: {} of {gsz} CTA(s) saw a stale or \
             unwritten peer slot while scanning all {gsz} slots over {rounds} rounds ({mismatches} \
             bad slot-reads in total)",
            host[..gsz].iter().filter(|&&m| m != 0).count()
        );
        // Not-vacuous: every CTA's LAST read is slot G-1 in the LAST round, a value only a real
        // cross-CTA read after a real rendezvous can produce (0 would mean it never got there).
        let want = (gsz as u32 - 1) * 1000 + (rounds as u32 - 1) + 1;
        for b in 0..gsz {
            assert_eq!(
                host[gsz + b],
                want,
                "CTA {b} last read {} from slot {}, expected {want}",
                host[gsz + b],
                gsz - 1
            );
        }
        eprintln!(
            "[gate] grid barrier: {} cooperative CTAs x {} threads ({} blocks/SM x {} SMs), \
             {rounds} rounds, every CTA scanned all {} peer slots each round, \
             0 cross-CTA ordering violations \u{2713}",
            plan.grid, plan.block, plan.blocks_per_sm, plan.sm_count, plan.grid
        );
    }

    /// **The residency bound is enforced before anything is launched, and it is the driver's bound.**
    ///
    /// A cooperative grid that is not simultaneously resident does not run slowly — it hangs, because
    /// the CTAs that never got scheduled never arrive at the barrier. So [`plan_grid`] declines above
    /// `cuOccupancyMaxActiveBlocksPerMultiprocessor x sm_count` instead of clamping, and this checks
    /// that our number is the same one `cuLaunchCooperativeKernel` itself validates against.
    ///
    /// The over-subscribed launch is run with `rounds = 0`, so the probe calls no barrier at all: if
    /// some driver were to accept the launch instead of refusing it, the kernel returns immediately
    /// and this test fails loudly rather than wedging the device.
    #[test]
    fn cooperative_grid_declines_above_the_residency_bound() {
        let mut guard = crate::gpu::gpu();
        let Some(g) = guard.as_mut() else {
            crate::diff::skip_or_fail(
                "cooperative_grid_declines_above_the_residency_bound",
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            );
            return;
        };
        if !supports_cooperative_launch(g) {
            crate::diff::skip_or_fail(
                "cooperative_grid_declines_above_the_residency_bound",
                "device reports no CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH",
            );
            return;
        }
        let ptx = gridbar_probe_ptx();
        let f = g
            .function("mega_gridbar_probe_v1", &ptx, "wk_gridbar_probe")
            .expect("gridbar probe JIT");
        let (per_sm, max_resident) = max_resident_ctas(g, &f, MEGA_BLOCK).expect("occupancy");
        assert!(per_sm > 0, "occupancy query returned 0 blocks/SM");
        assert_eq!(max_resident, per_sm * g.target().sm_count as u32);

        // The plan declines above the bound, and at zero, instead of quietly clamping.
        assert!(plan_grid(g, &f, true, Some(max_resident + 1))
            .expect("plan")
            .is_none());
        assert!(plan_grid(g, &f, true, Some(0)).expect("plan").is_none());
        let at = plan_grid(g, &f, true, Some(max_resident))
            .expect("plan")
            .expect("exactly the bound must be plannable");
        assert_eq!(at.grid, max_resident);
        // A block-scoped module is one CTA regardless of what anyone asks for: that is a correctness
        // property of `tid==0`-guarded side effects, not a tuning default.
        let bs = plan_grid(g, &f, false, Some(max_resident))
            .expect("plan")
            .expect("block-scoped always plans");
        assert_eq!((bs.grid, bs.cooperative), (1, false));

        // ...and the driver agrees. `rounds = 0` means the probe never reaches a barrier, so an
        // unexpectedly-accepted launch terminates instead of hanging.
        let mut buf = g.stream.alloc_zeros::<u32>(1).expect("buf");
        let mut out = g.stream.alloc_zeros::<u32>(2).expect("out");
        let mut bar = g
            .stream
            .alloc_zeros::<u8>(GRID_BARRIER_BYTES)
            .expect("barrier state");
        let rounds: u64 = 0;
        let too_big = LaunchConfig {
            grid_dim: (max_resident + 1, 1, 1),
            block_dim: (MEGA_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let refused = {
            let mut b = g.stream.launch_builder(&f);
            b.arg(&mut buf);
            b.arg(&mut out);
            b.arg(&mut bar);
            b.arg(&rounds);
            unsafe { b.launch_cooperative(too_big) }
        };
        assert!(
            refused.is_err(),
            "cuLaunchCooperativeKernel accepted {} CTAs while occupancy says only {max_resident} \
             fit — our residency bound and the driver's disagree, so a real grid barrier at this \
             size would hang",
            max_resident + 1
        );
        // The refusal is a launch-validation error, not a sticky fault: the context must still work.
        g.stream
            .synchronize()
            .expect("context healthy after a refused cooperative launch");
        eprintln!(
            "[gate] residency bound: {per_sm} blocks/SM x {} SMs = {max_resident} CTAs; \
             {} refused by the driver ({:?}) \u{2713}",
            g.target().sm_count,
            max_resident + 1,
            refused.unwrap_err()
        );
    }
}
