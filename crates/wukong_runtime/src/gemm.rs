//! Single-precision GEMM — the heavy ML kernel.
//!
//! Computes `C[M,N] = A[M,K] · B[K,N]` (row-major f32) with a `beta` flag: `beta == 0` overwrites
//! `C`, `beta == 1` accumulates into it. This is the tuned microkernel the Wukong compiler lowers a
//! recognized matmul loop nest to (see `wukong_mir_build`'s matmul recognizer) — analogous to how
//! XLA/TVM/oneDNN lower a matmul op to an optimized microkernel rather than emitting the naive nest.
//!
//! Structure (the classic BLIS five-loop GEMM):
//!   * register-block: a `MR×NR` tile of C is held in vector registers across the K loop;
//!   * cache-block: `MC×KC` panels of A and `KC×NC` panels of B are packed into contiguous,
//!     aligned scratch so the microkernel streams them with unit stride;
//!   * the microkernel is AVX2 + FMA (true 256-bit, 8 f32/lane) — the width Cranelift can't emit —
//!     with a portable scalar fallback selected at runtime via `is_x86_feature_detected!`.
//!
//! The native backend calls this directly; the interpreter marshals its abstract memory into real
//! buffers and calls the *same* function, so the differential oracle stays bit-for-bit exact.

// Register-block tile: 6 rows × 16 cols. 16 cols = two 256-bit lanes, so the kernel keeps 12 `__m256`
// accumulators live (of 16 ymm regs) — the proven Haswell/Zen sweet spot.
const MR: usize = 6;
const NR: usize = 16;
// Cache-block sizes (multiples of MR/NR). The A panel (MC×KC) targets L2, the B panel (KC×NC) the L3.
// MC=144 (≈144 KB A-block) measured the single-core sweet spot here: small enough to stay L2-resident
// while B streams. A bigger MC regressed — the larger A-block gets evicted by the streaming B and its
// micropanels then miss to L3 (1008 ≈ -16% at 2048³; 216/288 lost too). KC=384 is the *cap* for the
// size-adaptive K-block `select_kc`: the largest KC×NR B-micropanel (24 KB) + MR×KC A-micropanel
// (9 KB) that stays L1-resident on this 48 KB L1 — KC=512 (44 KB) thrashes L1 and is ~30% slower,
// while the old KC=256 left ~10% on the table at ≥1024³. NC=4080 keeps a KC×NC B-panel (≤6 MB) in L3.
const MC: usize = 144;
const KC: usize = 384;
const NC: usize = 4080;

// Minimum multiply-accumulate count (`m·n·k`) before the parallel kernel is worth its threading
// overhead. Was 2^26 (≈67M) when the only parallel grid was 3×nworkers slivers — 256³ (~17M) ran
// faster serial then. The work-scaled shared-pack grid (few large blocks near the gate) moved the
// crossover: 256³ measured ~1.7–2× over serial through it (2026-07-09, adjacent same-run), so the
// gate now sits at 2^23 ≈ 8.4M, below which cross-core wake/sync on this P+E hybrid still costs
// more than it saves.
const PAR_MIN_MACS: u64 = 1 << 23;

/// The parallel gate, env-probe-able: `WUKONG_GEMM_PAR_MIN_MACS` (a MAC count, read once)
/// overrides [`PAR_MIN_MACS`] so the serial↔parallel crossover — e.g. whether 256³ (~17M MACs)
/// pays for threads under the work-scaled 2D grid — can be swept without rebuilds. Gate-only:
/// which path runs never changes the bits (serial == parallel bit-for-bit).
fn par_min_macs() -> u64 {
    use std::sync::OnceLock;
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("WUKONG_GEMM_PAR_MIN_MACS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(PAR_MIN_MACS)
    })
}

/// Per-task MAC budget for the shared-pack 2D path's **work-scaled** load-balance target: the
/// effective worker count is `clamp(macs / min_task_macs(), 1, nworkers)`, so a problem near the
/// parallel gate gets a few large blocks (each still worth its scheduling) instead of `3×nworkers`
/// slivers. The coded default is 4 Mi. (The 2026-07-09 sweep at 256³ read 1M/4M/16M → 209/146/90
/// GF/s, i.e. 1 Mi measured best; the default was never moved to it and the shared path no longer
/// runs by DEFAULT at any size — see [`gemm_2d_shared`] — so this budget only governs the
/// `=1`/`=band` instrument settings and the 1 Mi point stays an un-landed measurement.)
/// Env-overridable
/// (`WUKONG_GEMM_MIN_TASK_MACS`, read once) so the small-problem crossover can be swept together
/// with `WUKONG_GEMM_PAR_MIN_MACS`. Throughput-only: the grid shape never changes the bits.
#[cfg(target_arch = "x86_64")]
fn min_task_macs() -> u64 {
    use std::sync::OnceLock;
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        positive_or_unset(std::env::var("WUKONG_GEMM_MIN_TASK_MACS").ok()).unwrap_or(4 << 20)
    })
}

/// Sanitize a knob value that is used as a **divisor**: zero — like an unparsable string — is
/// ignored (= unset). `WUKONG_GEMM_MIN_TASK_MACS=0` otherwise reached the `macs / min_task_macs()`
/// in [`sgemm_2d_shared`] and aborted the compiled program with `attempt to divide by zero` raised
/// inside an `extern "C"` runtime kernel; a sweep script that walks a budget range down through 0
/// must fall back to the default instead. Pure (takes the already-read variable) so the policy is
/// unit-testable without process-global env state; [`gemm_task_macs`] enforces the same rule inline.
#[cfg(target_arch = "x86_64")]
fn positive_or_unset(raw: Option<String>) -> Option<u64> {
    raw.and_then(|s| s.parse::<u64>().ok()).filter(|&v| v > 0)
}

#[inline]
fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

/// Size-adaptive K-block height (`kc`). KC=384 is the largest K-block whose `KC×NR` B-micropanel
/// (24 KB) + `MR×KC` A-micropanel (9 KB) stays L1-resident on this 48 KB-L1 P-core; KC=512 (44 KB)
/// thrashes L1 (~30% slower) and KC=256 under-amortizes the B-micropanel loads (and doubles the C
/// re-stream passes), leaving ~10% on the table at ≥1024³. K is split into the fewest equal-ish blocks
/// ≤ the cap, so k=512 → 2×256 (even) instead of 384+128 (a thin tail costing ~4%), while large K gets
/// the full ~384. `kc` only changes how the K reduction is *grouped*: both kernels call this so serial
/// stays bit-identical to parallel, and the grouping differs from KC=256 only in low f32 bits (within
/// the √k·ε tolerance the gemm tests assert vs a naive reference). Result ∈ [4, KC].
#[inline]
fn select_kc(k: usize) -> usize {
    let nblocks = k.div_ceil(KC).max(1);
    round_up(k.div_ceil(nblocks), 4) // multiple of the ×4 K-unroll; ≤ KC since k/nblocks ≤ KC
}

/// A private rayon pool for the parallel GEMM, sized to the machine's **physical** core count instead
/// of rayon's default (one worker per *logical* core). A compute-bound AVX2-FMA GEMM already saturates
/// a core's two FMA pipes with a single thread, so the HyperThread sibling adds only scheduling
/// contention: measured at the mid sizes where scaling lags hardest, 16 physical threads scale ~30-40%
/// better than 22 logical (512³ 2.2×→3.1×, 1024³ 3.2×→3.9× over single core). A *private* pool (not a
/// global resize) leaves the default pool — and the thread-count-derived striping of the reduction
/// kernels that run on it — untouched. The GEMM result is independent of how panels are distributed
/// (the per-(i,j) K-order is fixed), so this changes throughput only, never the bits: serial stays
/// bit-identical to parallel. `None` (⇒ caller uses the default pool) when physical ≥ logical (no HT
/// to avoid) or the pool can't be built.
/// Since 2026-07-10 this is the crate's UNIFIED kernel pool: `run_on_wuk_pool` (lib.rs) routes
/// the `_parallel` norms/velem/reductions and `wukong_parallel_for` regions here too (default;
/// `WUKONG_POOL_UNIFY=0` opts out), so a transformer forward stays on ONE warm pool instead of
/// bouncing private↔global between adjacent kernels. Worker stacks are 16 MiB for the outlined
/// region bodies' privatized frames (same rationale as `ensure_global_pool`). The routed kernels
/// all chunk independently of worker count (fixed-chunk velem/reductions, per-row norms,
/// per-index regions), so the striping caveat above is moot for them — bits never change.
pub(crate) fn gemm_pool() -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        // Configure the global pool FIRST: `current_num_threads()` below instantiates the default
        // registry if none exists, and it must come up with the runtime's required configuration
        // (16 MiB stacks, RAYON_NUM_THREADS honored) rather than rayon's bare defaults.
        crate::ensure_global_pool();
        let physical = num_cpus::get_physical().max(1);
        if physical >= rayon::current_num_threads() {
            return None; // no HyperThreads to shed (or single pool already this small)
        }
        let mut b = rayon::ThreadPoolBuilder::new()
            .num_threads(physical)
            .stack_size(16 * 1024 * 1024)
            .thread_name(|i| format!("wukong-gemm-{i}"));
        // A/B probe (`WUKONG_GEMM_AFFINITY=1`): pin worker i to one logical CPU — i<6 to the
        // P-cores' primary siblings (logical 2i on this 6P+8E+2LPE part), the rest to the E/LP-E
        // CPUs (6+i). MEASURED 25–35% SLOWER than free migration (ABBA adjacent runs, stable
        // serial baseline): Windows' scheduler + Thread Director places better than static pins
        // on this hybrid. Kept, off by default, purely as the recorded instrument so the negative
        // stays reproducible — do not flip this on expecting an MKL-style KMP_AFFINITY win.
        if std::env::var("WUKONG_GEMM_AFFINITY").is_ok_and(|v| v == "1") {
            b = b.start_handler(|i| pin_worker_to_cpu(if i < 6 { 2 * i } else { 6 + i }));
        }
        b.build().ok()
    })
    .as_ref()
}

/// Pin the calling worker thread to one logical CPU (Windows; no-op elsewhere). Scheduling-only:
/// affects which core runs a worker, never what it computes.
#[cfg(windows)]
fn pin_worker_to_cpu(cpu: usize) {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadAffinityMask(h: isize, mask: usize) -> usize;
    }
    if cpu < usize::BITS as usize {
        // SAFETY: plain affinity syscall on the current thread handle.
        unsafe {
            SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu);
        }
    }
}
#[cfg(not(windows))]
fn pin_worker_to_cpu(_cpu: usize) {}

// A mid-size-regime pool with fewer workers (physical/2, "shed the E/LP-E stragglers") was tried
// here and MEASURED SLOWER on the P+E dev box (adjacent same-run sweep, gemm_scaling example,
// WUKONG_GEMM_MID_THREADS ∈ {6,8,12,16}): the full 16-worker physical pool won at every mid
// shape (1024³ 410 vs 388 GF/s, 512×768×3072 377 vs 352; bare 512³ was the one ~5% exception).
// Work-stealing over the many row-panel tasks absorbs the slow cores; shrinking the pool just
// discards their throughput. Don't re-add a thread-count regime without new adjacent-run evidence.

/// A/B kill-switch (`WUKONG_GEMM_FORKJOIN=1`, read once): route the parallel GEMM through the
/// legacy per-block fork-join shape ([`sgemm_avx2_parallel`]) instead of the persistent broadcast
/// region ([`sgemm_persistent_region`]), so the region win stays measurable adjacent-run on any
/// machine — the same instrument discipline as `WUKONG_PACK_SPLIT_REGIONS` (see [`pack_ab_par`]).
#[cfg(target_arch = "x86_64")]
fn gemm_forkjoin() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("WUKONG_GEMM_FORKJOIN").is_ok_and(|v| v == "1"))
}

/// The BLIS-style **2D block-parallel** path ([`sgemm_2d_blocks`]) — per-thread L2-resident C
/// blocks with per-worker packing and no barriers — is the DEFAULT parallel GEMM: it A/B-measured
/// a decisive win over the row-panel/shared-B paths in BOTH ABBA orderings (gemm_scaling,
/// 2026-07-08: 512³ ~182→~305 GF/s ≈1.65×, 1024³ ~365→~460 ≈1.26×, 512×768×3072 ~290→~473),
/// exactly the per-thread-B-reuse mechanism the three refuted scheduling levers pointed to.
/// `WUKONG_GEMM_2D=0` (read once) opts OUT to the persistent-region path, keeping the win
/// adjacent-run-measurable — the same instrument discipline as `WUKONG_GEMM_FORKJOIN`.
#[cfg(target_arch = "x86_64")]
fn gemm_2d() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| !std::env::var("WUKONG_GEMM_2D").is_ok_and(|v| v == "0"))
}

/// The (retired) shared-pack band ceiling: kept for the `WUKONG_GEMM_2D_SHARED=band` instrument
/// setting, which restores the 2026-07-09 size-keyed dispatch for A/B archaeology. See
/// [`gemm_2d_shared`] — per-block is now the default at EVERY size.
const SHARED_MAX_MACS: u64 = 1 << 26;

/// Which 2D path a parallel GEMM takes: **per-block packing ([`sgemm_2d_blocks`]) everywhere** —
/// each worker packs exactly the panels it consumes into its own L2, no cross-core panel reads,
/// no per-K-block fork-join barrier. The 2026-07-09 measurement had a size-keyed band (shared
/// pack below [`SHARED_MAX_MACS`], where 256³ then read ~209 vs ~185 GF/s in shared's favor),
/// but that basis did NOT survive this session's pool unification: re-measured 2026-07-10/11
/// (AC, quiet machine, adjacent ABBA ×2 plus the prior window's 3 observations — per-block won
/// all 6 pairings): 256³ shared 137-159 GF/s (53-56% of adjacent MKL-all) vs per-block
/// 176-230 GF/s (65-70%, one earlier read 97%). Mid/large sizes were per-block all along
/// (512³ ~315 vs ~278, 1024³ ~462 vs ~385, skinny FFN up to 1.75×). `WUKONG_GEMM_2D_SHARED=1`
/// forces shared everywhere, `=0` per-block everywhere (now the default), `=band` restores the
/// retired size-keyed dispatch (read once) — adjacent-run A/B instruments, same discipline as
/// `WUKONG_GEMM_2D` / `WUKONG_GEMM_FORKJOIN`.
#[cfg(target_arch = "x86_64")]
fn gemm_2d_shared(macs: u64) -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<Option<bool>> = OnceLock::new();
    match V.get_or_init(|| match std::env::var("WUKONG_GEMM_2D_SHARED").as_deref() {
        Ok("1") => Some(true),
        Ok("0") => Some(false),
        Ok("band") => None,
        _ => Some(false),
    }) {
        Some(v) => *v,
        None => macs < SHARED_MAX_MACS,
    }
}

/// Dynamic block scheduling (**default ON** since 2026-07-11; `WUKONG_GEMM_DYN=0` opts out to
/// the static range-split path, read once): route the per-block 2D path through
/// [`sgemm_2d_blocks_dyn`] — a shared atomic work queue (`fetch_add` over the fixed block
/// enumeration) instead of rayon's range-split work-stealing — so faster cores automatically
/// claim more C blocks on this asymmetric 6P+8E+2LPE part (a P-core that finishes its share keeps
/// pulling blocks instead of idling while E-cores straggle). Per-worker packing semantics are
/// preserved: a worker packs exactly what it consumes, into its own [`PACK_SCRATCH_2D`] — the
/// locality the shared-pack refutation showed is the win.
///
/// **The central adjacent-run A/B that flipped it** (2026-07-11, AC+full healthy state, S/D/D/S
/// process quartets, `gemm_var` + `gemm_scaling` instruments): the box's dominant mid-size
/// symptom is 100-500 ms *straggler episodes* — per-call min stays fast while the whole
/// distribution widens (a worker stuck on a slow/preempted core drags every call's join; MKL
/// adjacent under the identical regime shows no episodes). Dynamic claiming halves that
/// variance (512³ solo CoV 14.3/11.8% → 8.6/7.5%, batch floor 154/177 → 198/206 GF/s), wins the
/// vs-MKL(all) medians at 256³ (~94-96 → ~103-110%) and 512³ (~93-97 → ~98-99%), washes-to-wins
/// 1024³, washes 2048³, and wins 4 of 6 transformer skinny NT shapes (128×768·768ᵀ par 292/315 →
/// 337/339 GF/s; both S=128 FFN shapes up) with the other 2 a wash. Scheduling-only: WHO computes
/// a block never changes the bits — serial == dynamic bit-for-bit (pinned by
/// `sgemm_2d_dyn_matches_serial`).
#[cfg(target_arch = "x86_64")]
fn gemm_dyn() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| !std::env::var("WUKONG_GEMM_DYN").is_ok_and(|v| v == "0"))
}

/// Steal-order probe (`WUKONG_GEMM_STEAL_ORDER=1`, read once, **default OFF**): with dynamic
/// scheduling ([`gemm_dyn`]), enumerate the C blocks full-size-first / edge-tails-last (the
/// LPT-style order [`dyn_block_order`] builds) instead of plain row-major, so the largest work is
/// claimed early — and the last blocks claimed (the ones that bound the join straggle) are the
/// SMALL edge remainders, which even an E-core drains quickly. A no-op under
/// `WUKONG_GEMM_DYN=0` (the static rayon path has no claim queue to order). A permutation of the
/// same block set, each block still computed exactly once by one owner — throughput-only, never
/// the bits.
#[cfg(target_arch = "x86_64")]
fn gemm_steal_order() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("WUKONG_GEMM_STEAL_ORDER").is_ok_and(|v| v == "1"))
}

/// Shared-A pack for single-block-row grids (`WUKONG_GEMM_SKINNY_A=1`, read once, **default OFF
/// — REFUTED as a win, kept as the recorded instrument**): when the 2D grid has ONE block row
/// (`nbi == 1` — the skinny-M rule, every transformer activation GEMM at S ≤ 144), all blocks'
/// A slices are the SAME whole-A rows, and per-block packing re-packs them once per column block
/// — at 128×768·768ᵀ that is 24 redundant ~0.4 MB A packs on a ~380 µs call. This regime packs A
/// ONCE per call into a shared call-lifetime buffer (parallel pack phase, joined before any
/// compute reads it), keeping per-block B packing as-is.
///
/// **Measured 2026-07-11 (ON/OFF/OFF/ON adjacent quartet, gemm_scaling, AC+full healthy state):
/// the traffic argument LOSES.** 128×768×3072 par 413/407 GF/s shared vs 468/483 per-block
/// (~14% loss, both pairings, far beyond drift); 128×768·768ᵀ a wash (356/334 vs 335/380);
/// 128×3072×768 within noise (415/430 vs 422/393). Why: the per-block "redundant" A pack is also
/// a PREFETCH — each worker packs the chunk into its own L2 right before computing with it, with
/// zero synchronization — while the shared pack demotes every A read to L3 and stalls the whole
/// call on a pack-phase join straggler (this hybrid's E/LPE cores make phase joins expensive,
/// the same tax the persistent-region wash exposed). Do not re-flip without a mechanism that
/// keeps the packed A L2-local per worker. Packed bytes, K grouping, and per-(i,j) order are
/// unchanged either way — serial == dyn bit-for-bit on both settings (the differential tests
/// sweep the parameter).
#[cfg(target_arch = "x86_64")]
fn gemm_skinny_a() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("WUKONG_GEMM_SKINNY_A").is_ok_and(|v| v == "1"))
}

/// Grain-size tuning surface (`WUKONG_GEMM_TASK_MACS`, read once, **default unset = current
/// behavior**): when set to a per-task MAC budget `T`, the **per-block** 2D paths
/// ([`sgemm_2d_blocks`] / [`sgemm_2d_blocks_dyn`]) aim for `clamp(macs / T, 1, 2^20)` grid blocks
/// instead of the default `3 × nworkers`, so the block granularity can be swept externally
/// (smaller tasks improve P/E-core balance but raise pack redundancy — every A element is packed
/// once per block column and every B element once per block row — plus queue/scheduling traffic;
/// the sweep decides, not this code). Scope: the per-block paths only — the shared-pack small
/// band keeps its own budget (`WUKONG_GEMM_MIN_TASK_MACS`, see [`min_task_macs`]), so the two
/// knobs never fight over one path. Zero/unparsable values are ignored (= unset). Grid shape is
/// throughput-only: bit-exactness holds for every value (see [`select_2d_block_shape`]).
#[cfg(target_arch = "x86_64")]
fn gemm_task_macs() -> Option<u64> {
    use std::sync::OnceLock;
    static V: OnceLock<Option<u64>> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("WUKONG_GEMM_TASK_MACS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&v| v > 0)
    })
}

/// The load-balance target block count for the per-block 2D paths: `default_target`
/// (`3 × nworkers` at both call sites) unless [`gemm_task_macs`] overrides it. Pure in
/// `task_macs` so the policy is unit-testable without env state; the `2^20` cap keeps a
/// pathological budget from asking [`select_2d_block_shape`] for an absurd grid (the policy's
/// smallest candidate bounds the real block count by the matrix anyway).
#[cfg(target_arch = "x86_64")]
fn blocks_target_from(macs: u64, task_macs: Option<u64>, default_target: usize) -> usize {
    match task_macs {
        Some(t) => (macs / t).clamp(1, 1 << 20) as usize,
        None => default_target,
    }
}

/// C-tile prefetch in the [`micro_6x16`] prologue is ON by default: the up-to-6 C rows the
/// writeback reads/modifies sit `ldc` apart (a page-scale stride at large N), so no hardware
/// prefetcher covers them, and at ≥2048³ C is far out of cache — every micro-tile writeback
/// otherwise eats ~6–12 demand misses that the ~kc-deep FMA loop could have hidden.
/// `WUKONG_GEMM_PF_C=0` (read once) opts out, keeping the win adjacent-run-measurable — the
/// same instrument discipline as the path gates above. A prefetch is architecturally a hint:
/// the computed bits are identical either way.
#[cfg(target_arch = "x86_64")]
fn gemm_pf_c() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| !std::env::var("WUKONG_GEMM_PF_C").is_ok_and(|v| v == "0"))
}

/// Send-able bundle of the raw-pointer GEMM arguments, so they can cross into [`gemm_pool`]'s worker
/// (`ThreadPool::install` requires `Send`). Sound: `install` runs the closure to completion before it
/// returns, so the pointers outlive the call — identical to the lifetime discipline of the kernel's
/// own internal rayon closures, which already smuggle these pointers across as `usize`.
#[cfg(target_arch = "x86_64")]
struct GemmArgs {
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    beta: f32,
    bt: bool,
    epi: Option<Epilogue>,
}
#[cfg(target_arch = "x86_64")]
unsafe impl Send for GemmArgs {}

#[cfg(target_arch = "x86_64")]
impl GemmArgs {
    /// Run the **legacy fork-join** parallel kernel from this bundle (the `WUKONG_GEMM_FORKJOIN=1`
    /// instrument path; the default is [`sgemm_persistent_region`]). Taking `self` by value forces a
    /// closure that calls it to capture the whole (`Send`) `GemmArgs`, not the individual `!Send`
    /// raw-pointer fields that edition-2021 disjoint capture would otherwise grab.
    ///
    /// # Safety
    /// Same operand-size contract as [`sgemm_avx2_parallel`]; AVX2/FMA must be available.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run(self) {
        unsafe {
            sgemm_avx2_parallel(
                self.a, self.b, self.c, self.m, self.k, self.n, self.beta, self.bt, self.epi,
            )
        }
    }
}

// Reusable per-thread pack scratch for the serial kernel. The A/B panels are fully overwritten by
// the packers (real data + edge-padding zeros) on every block, so reusing the buffers across calls
// needs no re-zeroing — and removes a per-call malloc+zero of several hundred KB that was a
// measurable fraction of a *small* GEMM (e.g. 256³, where it left Wukong ~8% behind a tuned
// library; large GEMMs are compute-bound and never noticed it). Taken out and put back around the
// kernel so no borrow is held across it.
#[cfg(target_arch = "x86_64")]
thread_local! {
    static PACK_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

// Fused-epilogue activation codes (shared with the compiler's recognizer in `wukong_mir_build`).
// `ACT_IDENTITY` is the implicit passthrough (the kernel only branches on the others), but it names
// the protocol value 0 the compiler emits, so it is kept for clarity and used by the tests.
#[allow(dead_code)]
const ACT_IDENTITY: u32 = 0;
const ACT_RELU: u32 = 1;
const ACT_GELU: u32 = 2;
const ACT_SILU: u32 = 3;

/// A fused GEMM epilogue, applied to each `C` element **on the final K-block writeback only**:
/// `c = act(alpha·(A·Bᵀ) + bias[col])`. `bias` is null for no bias; `act` is one of [`ACT_IDENTITY`],
/// [`ACT_RELU`], [`ACT_GELU`], [`ACT_SILU`] (the transformer FFN activations); `alpha` is a
/// loop-invariant scalar applied to the matmul result before the bias-add (the attention score scale
/// `QKᵀ/√d` and every scaled projection). `alpha == 1.0` is the identity — the multiply is skipped so
/// every existing caller (which passes `1.0`) stays byte-identical.
/// Folding it here means `C` is written once with the scale+bias+activation already applied, instead of
/// separate passes over `C` — the memory traffic a `α·linear → bias → act` otherwise pays. Both backends
/// call the identical kernel, so the fused result stays bit-for-bit exact.
#[derive(Clone, Copy)]
struct Epilogue {
    bias: *const f32,
    act: u32,
    alpha: f32,
}

impl Epilogue {
    /// Apply the epilogue to value `x` at tile-local column `j` (relative to this epilogue's bias
    /// base): `act(alpha·x + bias[j])`. `# Safety`: `bias`, when non-null, must be valid at index `j`.
    #[inline]
    unsafe fn apply(&self, x: f32, j: usize) -> f32 {
        // The α scale multiplies the matmul result before the bias-add. `alpha == 1.0` skips the
        // multiply so the fused-epilogue callers (`nt_epi`, which pass 1.0) stay byte-identical.
        let mut y = if self.alpha != 1.0 { self.alpha * x } else { x };
        if !self.bias.is_null() {
            y += *self.bias.add(j);
        }
        // GELU/SiLU reuse `vmath`'s scalar form so a fused `act(x·Wᵀ+b)` is bit-identical to the
        // unfused `{ t = x·Wᵀ+b; act(t) }`. The epilogue is applied scalar in the (always-taken-when-
        // present) general writeback path, so one scalar call per active C element is the whole cost.
        match self.act {
            ACT_RELU => y = y.max(0.0),
            ACT_GELU => y = crate::vmath::gelu1(y),
            ACT_SILU => y = crate::vmath::silu1(y),
            _ => {}
        }
        y
    }

    /// This epilogue with its bias base advanced by `cols` columns (null stays null; `act`/`alpha`
    /// are scalar and carry unchanged).
    #[inline]
    fn shift(&self, cols: usize) -> Epilogue {
        Epilogue {
            // SAFETY: callers shift only within the C column range the bias array covers.
            bias: if self.bias.is_null() {
                self.bias
            } else {
                unsafe { self.bias.add(cols) }
            },
            act: self.act,
            alpha: self.alpha,
        }
    }
}

/// `C = A·B` with `beta` (0 = overwrite, else accumulate). Single-threaded. Row-major.
///
/// # Safety
/// `a`, `b`, `c` must be valid for `m*k`, `k*n`, `m*n` `f32` elements respectively.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    gemm_dispatch(a, b, c, m, k, n, beta, false, false, None);
}

/// `C = A·Bᵀ` (B is row-major `[n, k]`), the `nn.Linear` / `x @ Wᵀ` form — the most common matmul
/// in deep learning. Same `beta` rule, single-threaded.
///
/// # Safety
/// `a`, `b`, `c` valid for `m*k`, `n*k`, `m*n` `f32` elements respectively.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_nt(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    gemm_dispatch(a, b, c, m, k, n, beta, true, false, None);
}

/// `C = act(A·Bᵀ + bias)` — the `nn.Linear` form with a fused bias-add + activation epilogue, folded
/// into the GEMM's C-tile writeback so `C` is written once (no separate read-modify-write pass). The
/// compiler lowers a `matmul → bias-add → activation` chain to this. `bias` may be null (no bias) and
/// must otherwise be valid for `n` `f32`; `act` is 0 (identity), 1 (ReLU), 2 (GELU), or 3 (SiLU).
/// `beta` rule as usual.
/// Single-threaded (the epilogue folds into the serial writeback).
///
/// # Safety
/// `a`, `b`, `c` valid for `m*k`, `n*k`, `m*n` `f32`; `bias` null or valid for `n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_nt_epi(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    bias: *const f32,
    act: i64,
) {
    let epi = Epilogue {
        bias,
        act: act as u32,
        alpha: 1.0,
    };
    gemm_dispatch(a, b, c, m, k, n, beta, true, false, Some(epi));
}

/// `C = alpha·(A·Bᵀ)` — the **α-scaled `nn.Linear`**: the attention score matmul `QKᵀ/√d` (and every
/// scaled projection), where a loop-invariant scalar `alpha` multiplies the dot. The compiler lowers a
/// matmul nest whose store is `c[i,j] = alpha·s` to this. `alpha` is folded into the C-tile writeback on
/// the final K-block (no second pass over C), reusing the fused-epilogue machinery with a null bias and
/// identity activation — so `C` is written once as `alpha·(A·Bᵀ)`. `beta` rule as usual. Single-threaded.
///
/// The α multiply is one exact f32 op applied to the fully-reduced dot, so the result equals the naive
/// nest's `alpha·s` under the documented matmul reassociation (both backends call this identical kernel,
/// and it is gated against an f64 reference).
///
/// # Safety
/// `a` valid for `m*k`, `b` for `n*k`, `c` for `m*n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_nt_alpha(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    alpha: f32,
) {
    let epi = Epilogue {
        bias: std::ptr::null(),
        act: ACT_IDENTITY,
        alpha,
    };
    gemm_dispatch(a, b, c, m, k, n, beta, true, false, Some(epi));
}

/// Multi-threaded `C = alpha·(A·Bᵀ)` — the α-scaled `nn.Linear` across cores (the `@parallel` attention
/// score / scaled projection). The scale folds into each tile's final-K-block writeback, and each C tile
/// is owned by exactly one task with the same per-(i,j) accumulation order as the serial kernel — so it
/// is bit-identical to the serial `wukong_sgemm_nt_alpha` the interpreter oracle calls.
///
/// # Safety
/// `a` valid for `m*k`, `b` for `n*k`, `c` for `m*n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_nt_alpha_parallel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    alpha: f32,
) {
    let epi = Epilogue {
        bias: std::ptr::null(),
        act: ACT_IDENTITY,
        alpha,
    };
    gemm_dispatch(a, b, c, m, k, n, beta, true, true, Some(epi));
}

/// Pick AVX2 vs scalar and serial vs parallel; `bt` selects `C = A·Bᵀ`.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm`] / [`wukong_sgemm_nt`].
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_dispatch(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    bt: bool,
    par: bool,
    epi: Option<Epilogue>,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let beta = beta as f32;
    // Multi-thread only above a work threshold: below it, cross-core wake/sync (worse on this
    // P+E-core hybrid, where the E-cores are slow to spin up) costs more than it saves and the
    // parallel path is a net loss — empirically ~256^3 runs faster on one core than across all of
    // them. The serial AVX2 kernel is the fast path for everything smaller. The epilogue folds into
    // the per-tile writeback on the final K-block, so the parallel path carries it too (each C tile
    // is owned by exactly one task) — a `@parallel` fused FFN runs the bias+activation across cores.
    let macs = m as u64 * n as u64 * k as u64;
    let par = par && macs >= par_min_macs();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features just checked; dims validated by the caller contract.
            unsafe {
                match (par, gemm_pool()) {
                    // Parallel: the DEFAULT is the BLIS-style 2D block-parallel path with
                    // PER-BLOCK packing at every size — see `gemm_2d_shared` for the measured
                    // basis (the 2026-07-09 shared-pack small band did not survive pool
                    // unification) — scheduled by the atomic claim queue
                    // (`sgemm_2d_blocks_dyn`; the default since the 2026-07-11 central A/B — see
                    // `gemm_dyn`). `WUKONG_GEMM_DYN=0` swaps the scheduler back to rayon's
                    // static range-split (`sgemm_2d_blocks`, the adjacent-run baseline), with
                    // `WUKONG_GEMM_STEAL_ORDER=1` layering the full-blocks-first enumeration on
                    // top of the queue, and `WUKONG_GEMM_TASK_MACS` sweeping the grain of both
                    // per-block schedulers. `WUKONG_GEMM_2D_SHARED=1|band`
                    // forces the shared packing shape everywhere / restores the retired
                    // size-keyed band, `WUKONG_GEMM_2D=0` opts out to the
                    // persistent-broadcast-region shape (`sgemm_persistent_region`), and
                    // `WUKONG_GEMM_FORKJOIN=1` further routes to the legacy fork-join shape —
                    // all retained as adjacent-run A/B instruments.
                    // Width==1 serial fast-path: a 1-worker pool (RAYON_NUM_THREADS=1) can win
                    // nothing from the parallel schedules' per-block packing/claim machinery —
                    // run the tuned serial kernel instead (bit-identical by the standing
                    // serial == parallel law, so this is throughput-only dispatch).
                    (true, pool)
                        if pool.map_or_else(rayon::current_num_threads, |p| {
                            p.current_num_threads()
                        }) <= 1 =>
                    {
                        sgemm_avx2(a, b, c, m, k, n, beta, bt, epi)
                    }
                    (true, pool) => {
                        let args = GemmArgs { a, b, c, m, k, n, beta, bt, epi };
                        if gemm_2d() {
                            if gemm_2d_shared(macs) {
                                sgemm_2d_shared(pool, args);
                            } else if gemm_dyn() {
                                sgemm_2d_blocks_dyn(pool, args, gemm_steal_order(), gemm_skinny_a());
                            } else {
                                sgemm_2d_blocks(pool, args);
                            }
                        } else if gemm_forkjoin() {
                            match pool {
                                // Legacy: run the whole fork-join kernel inside the private pool so
                                // its nested `into_par_iter`s (pack + compute) use physical-core
                                // workers, not the default logical-core pool. `args.run()` moves the
                                // whole bundle, so the closure captures the `Send` `GemmArgs` rather
                                // than its `!Send` fields (edition-2021 disjoint capture); the
                                // closure runs to completion before `install` returns.
                                Some(p) => p.install(move || args.run()),
                                None => args.run(),
                            }
                        } else {
                            sgemm_persistent_region(pool, args);
                        }
                    }
                    (false, _) => sgemm_avx2(a, b, c, m, k, n, beta, bt, epi),
                }
            }
            return;
        }
    }
    sgemm_scalar(a, b, c, m, k, n, beta, bt, epi);
}

/// Multi-threaded `C = A·B`. For each `KC` contraction block it packs the *whole* A column-panel and
/// B row-panel once (shared, read-only), then runs every `MR×NR` register tile of the C grid in
/// parallel. Per-(i,j) accumulation order is identical to [`wukong_sgemm`], so the two agree
/// bit-for-bit (the interpreter oracle calls the serial one for both). Packing is shared, not
/// re-done per worker, so it scales to high core counts.
///
/// # Safety
/// Same contract as [`wukong_sgemm`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_parallel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    gemm_dispatch(a, b, c, m, k, n, beta, false, true, None);
}

/// Multi-threaded `C = A·Bᵀ` (the `nn.Linear` form).
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_nt`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_nt_parallel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    gemm_dispatch(a, b, c, m, k, n, beta, true, true, None);
}

/// Multi-threaded `C = act(A·Bᵀ + bias)` — the fused-epilogue `nn.Linear` across cores (the
/// `@parallel` transformer FFN). Same epilogue contract as [`wukong_sgemm_nt_epi`]; the bias +
/// activation folds into each tile's final-K-block writeback, and each C tile is owned by exactly one
/// task with the same per-(i,j) accumulation order as the serial kernel — so it is bit-identical to
/// the serial `wukong_sgemm_nt_epi` the interpreter oracle calls. Below the work threshold it runs
/// the serial fused path. This lets a `@parallel` FFN combine the fusion *and* all the cores (before,
/// the epilogue forced serial, so a fused FFN could use one or the other but not both).
///
/// # Safety
/// `a`, `b`, `c` valid for `m*k`, `n*k`, `m*n` `f32`; `bias` null or valid for `n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_nt_epi_parallel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    bias: *const f32,
    act: i64,
) {
    let epi = Epilogue {
        bias,
        act: act as u32,
        alpha: 1.0,
    };
    gemm_dispatch(a, b, c, m, k, n, beta, true, true, Some(epi));
}

/// Transpose a row-major `[rows, cols]` matrix `src` into a row-major `[cols, rows]` matrix `dst`
/// (`dst[c*rows + r] = src[r*cols + c]`). Reads each source row contiguously; a one-time
/// O(rows·cols) reorg used by the `C = Aᵀ·B` GEMM below.
///
/// # Safety
/// `src` and `dst` must each be valid for `rows*cols` `f32`, and must not overlap.
#[inline]
unsafe fn transpose_into(src: *const f32, dst: *mut f32, rows: usize, cols: usize) {
    for r in 0..rows {
        let s = src.add(r * cols);
        for c in 0..cols {
            *dst.add(c * rows + r) = *s.add(c);
        }
    }
}

/// `C = Aᵀ·B` — A is stored row-major as `[k, m]` (`a[p*m + i]`), B row-major `[k, n]` (`b[p*n + j]`),
/// C is `[m, n]`. This is the **weight-gradient** GEMM of a training backward pass (`dW = dYᵀ·X`): the
/// contraction axis (the batch) is the *outer* index of both inputs, so A's logical `[m, k]` operand
/// is the transpose of its storage. gcc/rustc compile the naive nest with column-strided A reads that
/// defeat vectorization (one cache line per element); Wukong transposes A into scratch once — O(m·k),
/// ~1/n of the O(m·n·k) GEMM — then runs the identical tuned `C = A·B` kernel. Reusing *that exact
/// kernel* means it is bit-for-bit the differential contract the interpreter marshals (no new
/// accumulation order to keep in sync). `beta`: 0 overwrites C, else accumulates.
///
/// # Safety
/// `a` valid for `k*m`, `b` for `k*n`, `c` for `m*n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_tn(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (mu, ku) = (m as usize, k as usize);
    let mut at = vec![0.0f32; mu * ku];
    transpose_into(a, at.as_mut_ptr(), ku, mu); // [k, m] -> [m, k]
    gemm_dispatch(at.as_ptr(), b, c, m, k, n, beta, false, false, None);
}

/// Multi-threaded `C = Aᵀ·B` (the `@parallel` weight-gradient GEMM). Transposes A serially (the small
/// O(m·k) prepass), then runs the multicore `C = A·B` kernel — whose per-(i,j) accumulation order
/// matches the serial one, so serial == parallel == interpreter bit-for-bit (the differential gate).
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_tn`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_tn_parallel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (mu, ku) = (m as usize, k as usize);
    let mut at = vec![0.0f32; mu * ku];
    transpose_into(a, at.as_mut_ptr(), ku, mu);
    gemm_dispatch(at.as_ptr(), b, c, m, k, n, beta, false, true, None);
}

/// Widen `n` half-width values (`bf16`/`f16` stored bits) into an `f32` scratch buffer. The widen is
/// **lossless** (`bf16` is the top 16 bits of an f32; `f16`→f32 is an exact super-range), so the
/// widened matrix holds exactly the values the half storage represents.
///
/// # Safety
/// `src` valid for `n` `u16`, `dst` valid for `n` `f32`, non-overlapping.
#[inline]
unsafe fn widen_into_f32(src: *const u16, dst: *mut f32, n: usize, widen: fn(u16) -> f32) {
    for i in 0..n {
        *dst.add(i) = widen(*src.add(i));
    }
}

/// `C = A·Bᵀ` (the `nn.Linear` form) with **half-precision inputs** (`bf16`/`f16`, stored as `u16`)
/// and an **f32 accumulator** — the standard mixed-precision transformer matmul. A is `[m, k]`, B is
/// `[n, k]` (its rows are the weight vectors), C is `[m, n]` f32.
///
/// Like the `Aᵀ·B` kernel, this is a cheap prepass (widen A and B into f32 scratch — `O(m·k + n·k)`,
/// dwarfed by the `O(m·n·k)` GEMM) feeding the *identical* tuned `C = A·Bᵀ` microkernel. Because the
/// widen is lossless, the result is bit-for-bit the f32 GEMM on the widened values — exactly the
/// differential contract the interpreter marshals (no new accumulation order). gcc/rustc compile the
/// naive half-precision nest with a per-element widen inside the triple loop that they cannot
/// vectorize; dispatching the whole nest to the packed AVX2 kernel after one vectorizable widen pass
/// is the win. (The half inputs are a footprint win on load; the GEMM itself runs in f32.)
///
/// # Safety
/// `a` valid for `m*k`, `b` for `n*k` `u16`; `c` for `m*n` `f32`.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn gemm_lowp_nt(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    par: bool,
    widen: fn(u16) -> f32,
    epi: Option<Epilogue>,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (mu, ku, nu) = (m as usize, k as usize, n as usize);
    let mut af = vec![0.0f32; mu * ku];
    let mut bf = vec![0.0f32; nu * ku];
    widen_into_f32(a, af.as_mut_ptr(), mu * ku, widen);
    widen_into_f32(b, bf.as_mut_ptr(), nu * ku, widen);
    gemm_dispatch(af.as_ptr(), bf.as_ptr(), c, m, k, n, beta, true, par, epi);
}

/// `C = A·Bᵀ` with `bf16` inputs (f32 accumulate), single-threaded. See [`gemm_lowp_nt`].
///
/// # Safety
/// `a` valid for `m*k`, `b` for `n*k` `bf16` (`u16`); `c` for `m*n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_bf16_nt(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    gemm_lowp_nt(a, b, c, m, k, n, beta, false, crate::bf16_bits_to_f32, None);
}

/// Multi-threaded `C = A·Bᵀ` with `bf16` inputs. The widen prepass is serial (small); the GEMM runs
/// across cores with the same per-(i,j) accumulation order as the serial kernel, so serial == parallel
/// == interpreter bit-for-bit.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_bf16_nt`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_bf16_nt_parallel(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    gemm_lowp_nt(a, b, c, m, k, n, beta, true, crate::bf16_bits_to_f32, None);
}

/// `C = A·Bᵀ` with IEEE `f16` inputs (f32 accumulate), single-threaded. The widen is F16C-exact
/// (`half` crate); otherwise identical to the bf16 twin. See [`gemm_lowp_nt`].
///
/// # Safety
/// `a` valid for `m*k`, `b` for `n*k` `f16` (`u16`); `c` for `m*n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_f16_nt(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    gemm_lowp_nt(a, b, c, m, k, n, beta, false, crate::f16_bits_to_f32, None);
}

/// Multi-threaded `C = A·Bᵀ` with `f16` inputs.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_f16_nt`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_f16_nt_parallel(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    gemm_lowp_nt(a, b, c, m, k, n, beta, true, crate::f16_bits_to_f32, None);
}

/// `C = act(A·Bᵀ + bias)` with **bf16/f16 inputs** (f32 accumulate) — the mixed-precision transformer
/// FFN projection, fused. Same widen prepass as [`gemm_lowp_nt`], but the f32 GEMM folds the bias-add
/// and activation into its C-tile writeback (the `Epilogue`), so C is written once. `bias` may be null
/// and must otherwise be valid for `n` `f32`; `act` is 0 (identity), 1 (ReLU), 2 (GELU), or 3 (SiLU).
/// Reuses the exact `wukong_sgemm_nt_epi` epilogue, so the fused result equals the unfused
/// matmul → [bias →] activation bit-for-bit (the differential contract the interpreter marshals).
///
/// # Safety
/// `a` valid for `m*k`, `b` for `n*k` `u16`; `c` for `m*n` `f32`; `bias` null or valid for `n` `f32`.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn gemm_lowp_nt_epi(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    bias: *const f32,
    act: i64,
    par: bool,
    widen: fn(u16) -> f32,
) {
    let epi = Epilogue {
        bias,
        act: act as u32,
        alpha: 1.0,
    };
    gemm_lowp_nt(a, b, c, m, k, n, beta, par, widen, Some(epi));
}

/// Fused-epilogue `C = act(A·Bᵀ + bias)` with `bf16` inputs, single-threaded. See [`gemm_lowp_nt_epi`].
///
/// # Safety
/// `a`/`b` valid for `m*k`/`n*k` `bf16` (`u16`); `c` for `m*n` `f32`; `bias` null or valid for `n` `f32`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wukong_sgemm_bf16_nt_epi(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    bias: *const f32,
    act: i64,
) {
    gemm_lowp_nt_epi(a, b, c, m, k, n, beta, bias, act, false, crate::bf16_bits_to_f32);
}

/// Multi-threaded fused-epilogue `C = act(A·Bᵀ + bias)` with `bf16` inputs.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_bf16_nt_epi`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wukong_sgemm_bf16_nt_epi_parallel(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    bias: *const f32,
    act: i64,
) {
    gemm_lowp_nt_epi(a, b, c, m, k, n, beta, bias, act, true, crate::bf16_bits_to_f32);
}

/// Fused-epilogue `C = act(A·Bᵀ + bias)` with `f16` inputs, single-threaded. See [`gemm_lowp_nt_epi`].
///
/// # Safety
/// `a`/`b` valid for `m*k`/`n*k` `f16` (`u16`); `c` for `m*n` `f32`; `bias` null or valid for `n` `f32`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wukong_sgemm_f16_nt_epi(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    bias: *const f32,
    act: i64,
) {
    gemm_lowp_nt_epi(a, b, c, m, k, n, beta, bias, act, false, crate::f16_bits_to_f32);
}

/// Multi-threaded fused-epilogue `C = act(A·Bᵀ + bias)` with `f16` inputs.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_f16_nt_epi`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wukong_sgemm_f16_nt_epi_parallel(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    bias: *const f32,
    act: i64,
) {
    gemm_lowp_nt_epi(a, b, c, m, k, n, beta, bias, act, true, crate::f16_bits_to_f32);
}

/// `C = Aᵀ·B` with **half-precision inputs** (`bf16`/`f16`, stored as `u16`) and an **f32 accumulator**
/// — the mixed-precision **weight-gradient** GEMM of a training backward pass (`dW = dYᵀ·X`). A is
/// stored `[k, m]` (its logical `[m, k]` operand is the transpose of storage), B is `[k, n]`, C is
/// `[m, n]` f32.
///
/// Like the f32 `Aᵀ·B` and the half `A·Bᵀ` kernels, this is a cheap prepass feeding the *identical*
/// proven kernel: widen A and B into f32 scratch (lossless — `O(k·m + k·n)`, dwarfed by the
/// `O(m·n·k)` GEMM), then delegate to the f32 `wukong_sgemm_tn`, which transposes A once and runs the
/// tuned `C = A·B` microkernel. Because the widen is lossless, the result is bit-for-bit the f32 TN
/// GEMM on the widened values — exactly the differential contract the interpreter marshals (no new
/// accumulation order). gcc/rustc lose twice on the naive half nest: the inline `bf16→f32` widen won't
/// vectorize *and* A's column-strided reads (the contraction is A's outer index) defeat vectorization.
///
/// # Safety
/// `a` valid for `k*m`, `b` for `k*n` `u16`; `c` for `m*n` `f32`.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn gemm_lowp_tn(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    par: bool,
    widen: fn(u16) -> f32,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (mu, ku, nu) = (m as usize, k as usize, n as usize);
    let mut af = vec![0.0f32; ku * mu];
    let mut bf = vec![0.0f32; ku * nu];
    widen_into_f32(a, af.as_mut_ptr(), ku * mu, widen);
    widen_into_f32(b, bf.as_mut_ptr(), ku * nu, widen);
    // Delegate to the f32 TN entry point on the widened operands (transpose A prepass + the tuned
    // C = A·B kernel) — so the half TN GEMM is bit-for-bit the f32 TN GEMM on the widened values.
    if par {
        wukong_sgemm_tn_parallel(af.as_ptr(), bf.as_ptr(), c, m, k, n, beta);
    } else {
        wukong_sgemm_tn(af.as_ptr(), bf.as_ptr(), c, m, k, n, beta);
    }
}

/// `C = Aᵀ·B` with `bf16` inputs (f32 accumulate), single-threaded. See [`gemm_lowp_tn`].
///
/// # Safety
/// `a` valid for `k*m`, `b` for `k*n` `bf16` (`u16`); `c` for `m*n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_bf16_tn(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    gemm_lowp_tn(a, b, c, m, k, n, beta, false, crate::bf16_bits_to_f32);
}

/// Multi-threaded `C = Aᵀ·B` with `bf16` inputs. The widen + transpose prepass is serial (small); the
/// GEMM runs across cores with the same per-(i,j) order as the serial kernel, so serial == parallel
/// == interpreter bit-for-bit.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_bf16_tn`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_bf16_tn_parallel(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    gemm_lowp_tn(a, b, c, m, k, n, beta, true, crate::bf16_bits_to_f32);
}

/// `C = Aᵀ·B` with IEEE `f16` inputs (f32 accumulate), single-threaded. The widen is F16C-exact;
/// otherwise identical to the bf16 twin. See [`gemm_lowp_tn`].
///
/// # Safety
/// `a` valid for `k*m`, `b` for `k*n` `f16` (`u16`); `c` for `m*n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_f16_tn(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    gemm_lowp_tn(a, b, c, m, k, n, beta, false, crate::f16_bits_to_f32);
}

/// Multi-threaded `C = Aᵀ·B` with `f16` inputs.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_f16_tn`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemm_f16_tn_parallel(
    a: *const u16,
    b: *const u16,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
) {
    gemm_lowp_tn(a, b, c, m, k, n, beta, true, crate::f16_bits_to_f32);
}

/// The **legacy fork-join** parallel kernel: one rayon fork-join for the fused pack region plus one
/// for the compute region, per K-block, per NC-block. Kept byte-identical behind
/// `WUKONG_GEMM_FORKJOIN=1` (see [`gemm_forkjoin`]) as the adjacent-run A/B instrument for the
/// persistent-region shape ([`sgemm_persistent_region`]), which replaces those per-block fork-joins
/// — each of which re-wakes parked workers and pays a join-straggler tax on this hybrid P/E-core
/// box — with one `broadcast` region per call and in-region barriers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_avx2_parallel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    beta: f32,
    bt: bool,
    epi: Option<Epilogue>,
) {
    use rayon::prelude::*;
    // Pack scratch allocated ONCE and reused across every cache block, instead of a fresh
    // `vec![0.0; …]` (malloc + zero of ~2 MB) per K-block per call — pure overhead the old code paid
    // on every block. Each block fully repacks its region (the pack writes real data *and* the
    // edge-padding zeros for every element it owns), so reuse needs no re-zeroing. Sized to the
    // largest block: full M rows × KC, and an NC-wide × KC B panel.
    let ap_cap = round_up(m, MR) * k.min(KC);
    let bp_cap = round_up(n.min(NC), NR) * k.min(KC);
    let mut ap = vec![0.0f32; ap_cap];
    let mut bp = vec![0.0f32; bp_cap];
    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let mut pc = 0;
        while pc < k {
            let kc = (k - pc).min(select_kc(k));
            let beta_eff = if pc == 0 { beta } else { 1.0 };
            // The epilogue (bias + activation) is folded into the C writeback on the FINAL K-block
            // only — exactly as the serial path does. Captured as Send-safe primitives (a raw `bias`
            // pointer can't cross into the rayon closure), reconstructed per tile at its global column.
            let is_last_k = pc + kc == k;
            let (epi_on, epi_bias_addr, epi_act, epi_alpha) = match epi {
                Some(e) if is_last_k => (true, e.bias as usize, e.act, e.alpha),
                _ => (false, 0usize, 0u32, 1.0f32),
            };
            // Pack the full A column-panel (m×kc) and B row-panel (kc×nc) — both in ONE parallel
            // region (see `pack_ab_par`): with the C compute spread across every core a serial pack
            // would dominate (Amdahl), but every extra fork-join is a real barrier tax at the mid
            // sizes, so the A- and B-panels share a single region (and a tiny block packs serially).
            pack_ab_par(
                a.add(pc),
                k,
                m,
                b,
                k,
                n,
                pc,
                jc,
                kc,
                nc,
                bt,
                ap.as_mut_ptr(),
                bp.as_mut_ptr(),
            );

            let mpanels = m.div_ceil(MR);
            let npanels = nc.div_ceil(NR);
            let (ap_addr, bp_addr, c_addr) =
                (ap.as_ptr() as usize, bp.as_ptr() as usize, c as usize);
            // Parallelize over C *row panels* (one task per MR rows), each sweeping all column
            // panels. Coarser tasks than per-tile (e.g. ~170 vs ~10880 at 1024³) cut rayon's
            // scheduling overhead, and a task reuses its A micro-panel (MR×kc, L1-resident) across
            // every column while streaming B. The per-(i,j) k-accumulation order is unchanged, so
            // serial, parallel, and the interpreter stay bit-identical (the differential gate).
            // Many more tasks than cores lets work-stealing absorb this P+E hybrid's core imbalance.
            (0..mpanels).into_par_iter().for_each(|ip| {
                let i0 = ip * MR;
                let mrv = (m - i0).min(MR);
                for jp in 0..npanels {
                    let j0 = jp * NR;
                    let nrv = (nc - j0).min(NR);
                    // Reconstruct the epilogue at this tile's global column (jc + j0); `None` off the
                    // final K-block. A null bias-addr (0) stays null through `shift`. Matches the serial
                    // path's double shift (jc, then j0) bit-for-bit, so parallel-fused == serial-fused.
                    let tile_epi = epi_on.then(|| {
                        Epilogue {
                            bias: epi_bias_addr as *const f32,
                            act: epi_act,
                            alpha: epi_alpha,
                        }
                        .shift(jc + j0)
                    });
                    // SAFETY: disjoint C rows; shared read-only packed panels; avx2 verified above.
                    unsafe {
                        micro_6x16(
                            kc,
                            (ap_addr as *const f32).add(ip * kc * MR),
                            (bp_addr as *const f32).add(jp * kc * NR),
                            (c_addr as *mut f32).add(i0 * n + (jc + j0)),
                            n,
                            beta_eff,
                            mrv,
                            nrv,
                            tile_epi,
                        );
                    }
                }
            });
            pc += kc;
        }
        jc += NC;
    }
}

/// Shared state of one persistent GEMM region, referenced by every broadcast worker. The raw
/// pointers cross thread boundaries by design; see the `Sync` impl for the aliasing argument.
#[cfg(target_arch = "x86_64")]
struct GemmRegion {
    args: GemmArgs,
    /// Pack scratch (call-lifetime: the broadcast joins before the owning `Vec`s drop).
    ap: *mut f32,
    bp: *mut f32,
    /// Per-(jc,pc)-block claim counters: `.0` hands out pack-panel indices over the combined A+B
    /// panel space, `.1` hands out C row-panel (`ip`) indices. One fresh pair per block — cheaper
    /// and simpler than resetting two shared counters, which would need its *own* barrier to order
    /// the reset against the previous phase's overshooting `fetch_add` stragglers.
    counters: Vec<(std::sync::atomic::AtomicUsize, std::sync::atomic::AtomicUsize)>,
    /// In-region phase barrier, sized to the exact broadcast width (asserted before first use).
    barrier: std::sync::Barrier,
    nworkers: usize,
}
// SAFETY: shared across the broadcast workers by reference. Every mutable target is written by
// exactly one worker per phase — packed panels and C row-panels are claimed via `fetch_add` (unique
// indices) and are disjoint — and the barrier orders all pack writes before any compute read. This
// is the same aliasing discipline the fork-join kernel has always used; it smuggled the very same
// pointers across its rayon closures as `usize`.
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for GemmRegion {}

/// One **persistent parallel region per GEMM call** — the GotoBLAS structure. The legacy shape
/// ([`sgemm_avx2_parallel`]) pays one rayon fork-join for the fused pack region plus one for the
/// compute region, per K-block, per NC-block — at 512³ that is 2 K-blocks × 2 regions = 4 full
/// fork-joins per call, and each one re-wakes parked workers and eats a join straggler tax on this
/// hybrid 6P+8E+2LPE pool (measured same-run: parallel scaling stuck at ~1.7×-vs-serial at 512³
/// while MKL scales ~3.7×). Here the pool is entered ONCE via `ThreadPool::broadcast` and the same
/// serial jc/pc block schedule runs *inside* the region on every worker, phase-separated by a
/// lightweight in-region [`std::sync::Barrier`]:
///
/// ```text
/// per block:  pack   — claim A+B panels off an atomic counter        → barrier
///             compute — claim C row-panels off a second counter      → barrier
///             (the final block's trailing barrier is elided: the broadcast join is that sync)
/// ```
///
/// so a 512³ call pays 1 broadcast fork-join + 3 barrier waits instead of 4 full fork-joins.
///
/// **Measured standing (honesty note):** adjacent A/B against the fork-join path
/// (`WUKONG_GEMM_FORKJOIN=1`, gemm_scaling, both orderings) is a **wash** — whichever config runs
/// first wins by more than any real delta, so the fork-join tax is NOT the dominant mid-size loss
/// this design targeted. Kept because it is structurally cheaper (1 wake per call, not 4+) with no
/// measured regression; three sibling hypotheses for the 512³-vs-MKL gap are now all refuted by
/// adjacent measurement (smaller pool: slower; this region fusion: wash; hard worker pinning
/// `WUKONG_GEMM_AFFINITY=1`: 25–35% SLOWER). The residual gap vs threaded MKL at 512³ is its
/// mid-size parallel *algorithm* (per-thread L2-blocked 2D C ownership), not our scheduling.
///
/// **Why a barrier under `broadcast` is sound** (and would deadlock under `scope`/`spawn`):
/// `broadcast` runs **exactly one closure instance on every pool worker** (rayon-core injects
/// `num_threads` jobs, one into each worker's dedicated broadcast queue, and its join latch counts
/// all of them), so a barrier sized to that width always has every participant scheduled on its own
/// thread — plain scope/spawn tasks can be executed serially on fewer workers by work-stealing, and
/// a barrier would then wait on a participant that can never start. Two further rayon-core facts
/// close the argument: (a) `inject_broadcast` pushes under a single registry-wide mutex into
/// per-worker FIFO queues, so concurrent broadcasts land in the SAME order on every worker — two
/// racing barrier-broadcasts cannot interleave in opposite orders on different workers and
/// cross-deadlock; (b) a worker drains its broadcast queue both when idle and while latch-waiting
/// inside other rayon work (`wait_until_cold` → `take_local_job`), so every participant eventually
/// arrives even on a busy pool.
///
/// **Bit-exactness:** the jc/pc schedule, `kc` grouping, `beta_eff` accumulate rule, final-K-block
/// epilogue rule, packed panel bytes, and each tile's `micro_6x16` sweep are identical to the
/// serial kernel — the atomics only change *which worker* handles a given (disjoint) panel, which
/// no result byte depends on. serial == parallel bit-for-bit, exactly as before (pinned by
/// `sgemm_persistent_region_matches_serial` and the differential gate).
///
/// # Safety
/// Operand-size contract of [`sgemm_avx2_parallel`]; AVX2/FMA must be available (verified by
/// [`gemm_dispatch`]).
#[cfg(target_arch = "x86_64")]
unsafe fn sgemm_persistent_region(pool: Option<&rayon::ThreadPool>, args: GemmArgs) {
    use std::sync::atomic::AtomicUsize;
    let (m, k, n) = (args.m, args.k, args.n);
    // Barrier width must equal broadcast width. `Some(pool)` broadcasts on the private pool (its
    // thread count is fixed at construction); `None` falls back to the free-function broadcast,
    // which lands on the caller's current pool (the global one when called from outside any pool)
    // — the same registry `rayon::current_num_threads()` reads, so the two always agree. A
    // mismatch would hang the barrier, so it is also asserted per-worker before the first wait.
    let nworkers = pool.map_or_else(rayon::current_num_threads, |p| p.current_num_threads());
    // Pack scratch allocated once per call and reused across every block, exactly as the fork-join
    // shape does (each block's pack fully overwrites the region it owns, padding included).
    let ap_cap = round_up(m, MR) * k.min(KC);
    let bp_cap = round_up(n.min(NC), NR) * k.min(KC);
    let mut ap = vec![0.0f32; ap_cap];
    let mut bp = vec![0.0f32; bp_cap];
    // The jc/pc schedule is a pure function of (m, k, n): every worker walks it identically, so
    // the block count — and each block's fresh counter pair — is known up front.
    let nblocks = n.div_ceil(NC) * k.div_ceil(select_kc(k));
    let region = GemmRegion {
        args,
        ap: ap.as_mut_ptr(),
        bp: bp.as_mut_ptr(),
        counters: (0..nblocks)
            .map(|_| (AtomicUsize::new(0), AtomicUsize::new(0)))
            .collect(),
        barrier: std::sync::Barrier::new(nworkers),
        nworkers,
    };
    let worker = |ctx: rayon::BroadcastContext<'_>| {
        // A width mismatch must fail loudly, not hang the barrier. Both sides are per-call
        // constants, uniform across workers — either every instance panics or none does, so no
        // worker is ever left waiting on a sibling that panicked before reaching the barrier.
        assert_eq!(
            ctx.num_threads(),
            region.nworkers,
            "gemm broadcast width != barrier width"
        );
        // SAFETY: avx2+fma verified by `gemm_dispatch`; `region`'s pointers stay valid for the
        // whole broadcast (it joins before `ap`/`bp` drop below).
        unsafe { gemm_region_worker(&region, ctx.index()) };
    };
    match pool {
        Some(p) => {
            p.broadcast(worker);
        }
        None => {
            // No-HT fallback (`gemm_pool() == None`): broadcast on the caller's *current* pool — the
            // global one when called from outside any pool. If that caller is itself a default-pool
            // rayon worker (a `@parallel` outer loop that dispatched this GEMM), this re-enters the
            // SAME pool via `broadcast`. That does not deadlock *only* because rayon-core hands every
            // pool worker the broadcast jobs in one uniform FIFO order and workers keep draining the
            // broadcast queue while latch-waiting — so no worker blocks on a broadcast job a sibling
            // is holding. That argument rests on rayon-core INTERNALS, not a public API guarantee
            // (verified against rayon-core 1.13.0 / rayon 1.12.0, the versions pinned in Cargo.lock).
            // A rayon bump must re-verify it, or route this arm to the legacy fork-join path
            // (`WUKONG_GEMM_FORKJOIN=1`, `gemm_forkjoin()` above), which never nests a broadcast.
            rayon::broadcast(worker);
        }
    }
}

/// One broadcast worker's walk of the whole block schedule (see [`sgemm_persistent_region`]). All
/// workers run this same function with the same (m,k,n)-derived loop bounds; the per-block atomics
/// distribute panels, and `Barrier::wait` — a synchronizing operation — publishes every pack write
/// before any compute read (which is why `Relaxed` suffices on the claim counters: they only need
/// unique indices, not ordering).
///
/// # Safety
/// Called only from [`sgemm_persistent_region`]'s broadcast; avx2+fma available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_region_worker(rg: &GemmRegion, widx: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    let (a, b, c) = (rg.args.a, rg.args.b, rg.args.c);
    let (m, k, n) = (rg.args.m, rg.args.k, rg.args.n);
    let (beta, bt, epi) = (rg.args.beta, rg.args.bt, rg.args.epi);
    let (ap, bp) = (rg.ap, rg.bp);
    let mut block = 0usize;
    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let mut pc = 0;
        while pc < k {
            let kc = (k - pc).min(select_kc(k));
            // First K-block honors the caller's beta; later blocks accumulate the partials — and
            // the epilogue folds into the C writeback on the FINAL K-block only. Identical rules,
            // in identical `kc` grouping, to both existing kernels.
            let beta_eff = if pc == 0 { beta } else { 1.0 };
            let block_epi = if pc + kc == k { epi } else { None };
            let (pack_ctr, comp_ctr) = &rg.counters[block];
            let mpanels = m.div_ceil(MR);
            let npanels = nc.div_ceil(NR);

            // Phase 1 — pack this block's A column-panel + B row-panel, over the same combined
            // A+B panel index space (and byte-gate) as the legacy `pack_ab_par`: below the
            // threshold the copy is too small to amortize cross-core distribution, so worker 0
            // packs both serially while the rest fall straight through to the barrier.
            let total_bytes = (round_up(m, MR) + round_up(nc, NR)) * kc * 4;
            if total_bytes < pack_par_min_bytes() {
                if widx == 0 {
                    pack_a(a.add(pc), k, m, kc, ap);
                    pack_b_block(b, k, n, pc, jc, kc, nc, bt, bp);
                }
            } else {
                // Resolve B's base pointer + leading dim once (mirrors `pack_b_block`'s dispatch).
                let (b_base, ldb) = if bt {
                    (b.add(jc * k + pc), k)
                } else {
                    (b.add(pc * n + jc), n)
                };
                loop {
                    let t = pack_ctr.fetch_add(1, Relaxed);
                    if t >= mpanels + npanels {
                        break;
                    }
                    // SAFETY: `fetch_add` hands each panel index to exactly one worker; panels
                    // are disjoint. Same bytes regardless of who packs what.
                    if t < mpanels {
                        pack_a_panel(a.add(pc), k, m, kc, t, ap.add(t * kc * MR));
                    } else {
                        let jp = t - mpanels;
                        let panel = bp.add(jp * kc * NR);
                        if bt {
                            pack_b_trans_panel(b_base, ldb, kc, nc, jp, panel);
                        } else {
                            pack_b_panel(b_base, ldb, kc, nc, jp, panel);
                        }
                    }
                }
            }
            // Every pack write must be complete — and published — before any worker's compute
            // reads the panels. This is the fork-join the broadcast region replaces, at the cost
            // of one barrier wait instead of a full worker wake + join.
            rg.barrier.wait();

            // Phase 2 — compute: claim C row-panels dynamically (`fetch_add` = the same
            // self-scheduling load balance work-stealing gave the fork-join shape; many more
            // panels than workers absorbs this P/E hybrid's core-speed imbalance). Each claimed
            // `ip` sweeps every column panel with the identical `micro_6x16` arguments and order
            // the legacy kernel used, so per-(i,j) K-accumulation is untouched.
            loop {
                let ip = comp_ctr.fetch_add(1, Relaxed);
                if ip >= mpanels {
                    break;
                }
                let i0 = ip * MR;
                let mrv = (m - i0).min(MR);
                for jp in 0..npanels {
                    let j0 = jp * NR;
                    let nrv = (nc - j0).min(NR);
                    // Bias base advanced to this tile's global column (jc + j0); `None` off the
                    // final K-block — matches the serial path's double shift bit-for-bit.
                    let tile_epi = block_epi.map(|e| e.shift(jc + j0));
                    // SAFETY: disjoint C rows per claimed `ip`; shared read-only packed panels
                    // (ordered by the barrier above); avx2 verified by the dispatcher.
                    micro_6x16(
                        kc,
                        ap.add(ip * kc * MR),
                        bp.add(jp * kc * NR),
                        c.add(i0 * n + (jc + j0)),
                        n,
                        beta_eff,
                        mrv,
                        nrv,
                        tile_epi,
                    );
                }
            }
            pc += kc;
            block += 1;
            // The next block's pack overwrites the shared panels, so compute must drain first.
            // The FINAL block needs no trailing barrier: the broadcast's own join is that sync.
            if block < rg.counters.len() {
                rg.barrier.wait();
            }
        }
        jc += NC;
    }
}

// --- 2D block-parallel path (WUKONG_GEMM_2D=1) ------------------------------------------------

// Reusable per-WORKER pack scratch for the 2D block path: each rayon worker thread keeps one
// (A-slice, B-slice) buffer pair and reuses it across every block it claims (and across calls —
// pool workers are process-lifetime threads), so a block costs zero mallocs after each worker's
// first. Same take/put-back discipline as `PACK_SCRATCH`; a separate thread-local keeps the two
// paths' buffer sizing independent. The packers fully overwrite every element they own (real data
// + edge-padding zeros), so reuse needs no re-zeroing.
#[cfg(target_arch = "x86_64")]
thread_local! {
    static PACK_SCRATCH_2D: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

// Reusable CALLER-side pack scratch for the shared-pack 2D path: the calling thread owns one
// (A, B) buffer pair holding the whole-matrix packed panels of the current K-block, shared
// read-only by every worker (pack writes are ordered before compute reads by the phase join —
// see `sgemm_2d_shared`). Same take/put-back discipline as `PACK_SCRATCH`; a separate
// thread-local keeps the shared and per-block paths' buffer sizing independent so either can be
// A/B-selected without resizing churn. The packers fully overwrite every element they own (real
// data + edge-padding zeros), so reuse needs no re-zeroing.
#[cfg(target_arch = "x86_64")]
thread_local! {
    static PACK_SCRATCH_2D_SHARED: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

// Caller-side scratch for the skinny shared-A pack ([`gemm_skinny_a`]): ALL K-chunks of A packed
// once per call, shared read-only by every worker (pack writes are ordered before compute reads
// by the pack region's join). Its own thread-local so the shared-A and per-block paths' buffer
// sizing stays independent — the same take/put-back discipline as the siblings above. The pack
// fully overwrites every element it owns (real data + edge-padding zeros), so reuse needs no
// re-zeroing.
#[cfg(target_arch = "x86_64")]
thread_local! {
    static PACK_SCRATCH_SKINNY_A: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Cap on the skinny shared-A buffer: `round_up(m, MR) × k` f32 for the WHOLE packed A (every
/// K-chunk at once, vs one `bm×KC` chunk in the per-block path). 8 MB covers every transformer
/// shape (132×3072×4 ≈ 1.6 MB) while staying L3-resident — the whole point is that every worker
/// re-reads the one packed copy from shared L3 instead of re-packing it; a buffer past L3 would
/// lose that and pay the pack allocation besides, so bigger K falls back to per-block packing.
#[cfg(target_arch = "x86_64")]
const SKINNY_A_MAX_BYTES: usize = 8 << 20;

/// Block-shape policy for the 2D block-parallel paths: pick `(BM, BN)` — the C-block height/width
/// — for an `m×n` output, aiming for at least `target_blocks` grid blocks (the caller's
/// load-balance budget: `3×` its effective worker count, so rayon work-stealing can absorb this
/// asymmetric 6P+8E+2LPE pool's core-speed imbalance). **The one tunable of the 2D paths**; pure,
/// so an A/B sweep can retune it without touching the kernels.
///
/// Constraints, in priority order:
/// * `BM % MR == 0` and `BN % NR == 0` — every interior block boundary lands on a micropanel
///   boundary, which is what makes each block's pack slice byte-identical to the serial kernel's
///   full-panel packs (zero-padding can then only occur at the true matrix edge; see
///   [`sgemm_2d_blocks`] / [`sgemm_2d_shared`]). Bit-exactness holds REGARDLESS of the shape
///   picked here — this policy is throughput-only.
/// * **L2 residency**: the block's packed B slice (`BN×kc` f32, `kc ≤ KC = 384`) plus A slice
///   (`BM×kc`) must fit a per-core L2 with room left for the C-tile stream. The dev box's P-core
///   L2 is 2 MB (not hardcoded — the candidates are sized conservatively so they also fit the
///   E-core cluster's per-core share): the largest candidate (192, 256) packs at most
///   192·384·4 ≈ 288 KB of A + 256·384·4 ≈ 384 KB of B ≈ 672 KB.
/// * **Skinny M** (`m ≤ MC = 144`): one row of blocks (`BM = round_up(m, MR)`) — a second row
///   would split the K reuse of an already-L2-resident A stripe for no locality gain — with `BN`
///   the largest NR multiple still yielding `~target_blocks/2` column blocks. The general
///   candidates mis-shape these: 128×768 output picked (24, 48) → a 6×16 grid whose blocks are
///   too short to amortize anything; the rule gives 1×24 at (132, 32).
/// * **Load balance**: candidates are tried largest-first (bigger blocks reuse their packed
///   slices longer) and the first yielding `>= target_blocks` wins; if even the smallest can't
///   (small m·n), the smallest is used — below that the block count is bounded by the matrix, not
///   the policy. Both chosen dims are then rebalanced to `round_up(ceil(dim / nblocks), unit)`,
///   which equalizes block heights AND widths without changing the grid (e.g. 1024³: (144, 192)
///   gave 7×144 + a 16-row straggler; rebalanced BM = 132 gives 7×132 + 100, and rebalanced
///   BN = 176 gives 5×176 + 144 instead of 5×192 + a 64-wide straggler column). Tail balancing is
///   an M/N-dimension-only move — it can never regroup the K reduction, which is what keeps it
///   unconditionally bit-safe (a K-dimension rebalance would change accumulation order and is
///   forbidden).
#[cfg(target_arch = "x86_64")]
fn select_2d_block_shape(m: usize, n: usize, target_blocks: usize) -> (usize, usize) {
    // (BM, BN) regimes, largest first. All BM ∈ 24..=192 are multiples of MR=6, all BN ∈ 48..=256
    // multiples of NR=16. Examples on the 16-worker dev pool (target = 48 blocks):
    // 2048³ → (192, 256); 1024³ → (144, 192) → 8×6 grid; 512³ → (48, 96) → 11×6.
    //
    // The square candidate (96, 96) sits between (96, 128) and (48, 96): it is the least-B-repack
    // grid for the **wide-M / narrow-N** rectangle that the pure-square ladder mis-shapes. A
    // 512×768 output (the GPT-2 attention/output projection at S=512) skips (96, 128) (only 36
    // blocks < 48) and lands on (48, 96) → an 11×8 grid whose 11 short row-blocks each re-pack the
    // full B column stripe eleven times — and B here is the transpose-gathered nn.Linear operand,
    // the expensive pack. (96, 96) instead gives a 6×8 grid (rebalanced (90, 96)): the same 48-block
    // load balance with **6** row-blocks, cutting the redundant B transpose-packs nearly in half.
    // Measured (adjacent-run ABBA, same power state, per-block default): 512×768·(768×768)ᵀ 75→89%
    // and 512×3072·(768×3072)ᵀ 76→87% of MKL(all), with the S=512 model forward ~8% faster; the
    // insertion never fires for any square (256/512/1024/2048³ each still pick their prior
    // candidate — (96, 96) yields < target blocks at every cube), so the cube standings are
    // unchanged (pinned in `select_2d_block_shape_policy`).
    const CANDIDATES: &[(usize, usize)] =
        &[(192, 256), (144, 192), (96, 128), (96, 96), (48, 96), (24, 48)];
    let target = target_blocks.max(1);
    // Equalize block sizes along one dim without changing its block count: `nb` is preserved
    // (round_up(ceil(d/nb), unit) ≤ bd since bd is a unit multiple with nb·bd ≥ d, and
    // nb·round_up(ceil(d/nb), unit) ≥ d, so ceil(d/·) lands back on nb) — the remainder is spread
    // across every block instead of one thin straggler that pays a whole task dispatch (and, under
    // dynamic scheduling, a whole claim) for a sliver of work.
    let rebalance = |d: usize, bd: usize, unit: usize| round_up(d.div_ceil(d.div_ceil(bd)), unit);
    if m <= MC {
        // Skinny M: nbi = 1; the largest NR-multiple BN with >= ~target/2 column blocks. Column
        // blocks alone must carry the load balance, so the divisor is halved (1.5× workers at the
        // callers' 3× budget). Floored at NR; capped at round_up(n, NR) (whole-matrix block).
        // (A 2-row skinny split to cut column-block A re-packing was A/B-refuted: it doubles the
        // costlier B transpose-repack, a wash-to-loss on 128×768 — the single L2-resident A stripe
        // stays the better trade.)
        let col_target = target.div_ceil(2);
        let bn = ((n / col_target) / NR * NR).clamp(NR, round_up(n, NR));
        return (round_up(m, MR), rebalance(n, bn, NR));
    }
    let (bm, bn) = CANDIDATES
        .iter()
        .copied()
        .find(|&(bm, bn)| m.div_ceil(bm) * n.div_ceil(bn) >= target)
        .unwrap_or(*CANDIDATES.last().unwrap());
    (rebalance(m, bm, MR), rebalance(n, bn, NR))
}

/// The **BLIS/MKL-style 2D block-parallel** GEMM with per-block packing under rayon's STATIC
/// range-split scheduling (the `WUKONG_GEMM_DYN=0` instrument path; the default since 2026-07-11
/// is the same grid/block body under the atomic claim queue — see [`sgemm_2d_blocks_dyn`] /
/// [`gemm_dyn`] — keeping this shape intact as its adjacent-run A/B baseline).
///
/// **Why this exists.** The shipped parallel decomposition splits C by MR=6-row panels: every
/// worker streams the ENTIRE shared packed B panel from L3 on every K-block — 16 workers
/// hammering shared L3 with zero per-thread L2 reuse of B. Three scheduling hypotheses for the
/// mid-size gap vs threaded MKL (42–46% @512³, 67–72% @1024³ same-run) were refuted by adjacent
/// A/B (fewer workers: slower; persistent broadcast region: wash; hard pinning: 25–35% slower —
/// all documented above), leaving the *structural* explanation: MKL/BLIS give each thread a 2D C
/// block whose B slice stays L2-resident. This path implements that:
///
/// * C is partitioned into `BM×BN` blocks ([`select_2d_block_shape`]) on MR/NR-aligned
///   boundaries; a flat `0..nbi·nbj` index space is one rayon task per block on the existing
///   private pool — plain work-stealing, **no barriers, no phase sync**: each block is fully
///   independent because its owner runs the WHOLE ascending-`pc` K loop for its block.
/// * Per K-block the owner packs THIS block's A slice (`BM×kc`) and B slice (`BN×kc`, honoring
///   `bt`) into per-worker scratch ([`PACK_SCRATCH_2D`]) with the same `pack_a`/`pack_b_block`
///   building blocks the serial kernel uses, offset to the block's row/col origin, then runs the
///   same [`macro_kernel`] with `ldc = n`. Packing is redundant across blocks sharing rows/cols
///   (an O(n²) cost buying O(n³) locality) — that is the deliberate BLIS trade.
///
/// **Bit-exactness (the hard invariant).** Each C element is accumulated by exactly ONE owner,
/// over the same [`select_kc`] groups, in the same ascending-`pc` order, with the same
/// `beta_eff = if pc == 0 { beta } else { 1.0 }` rule, the same final-K-block-only epilogue
/// shifted to the block's global column origin (the serial `shift(jc + j0)` discipline), through
/// the same [`micro_6x16`] — on packed panels whose BYTES are identical to the serial kernel's
/// for those rows/cols: `BM`/`BN` are MR/NR multiples, so every interior block boundary is a
/// micropanel boundary and `pack_a_panel`/`pack_b_panel`/`pack_b_trans_panel` zero-pad only at
/// the true matrix edge — exactly where the serial full-panel pack pads. Serial == 2D-parallel
/// bit-for-bit REGARDLESS of block shape (pinned by `sgemm_2d_blocks_matches_serial`).
///
/// # Safety
/// Operand-size contract of [`sgemm_avx2_parallel`]; AVX2/FMA must be available (verified by
/// [`gemm_dispatch`]).
#[cfg(target_arch = "x86_64")]
unsafe fn sgemm_2d_blocks(pool: Option<&rayon::ThreadPool>, args: GemmArgs) {
    use rayon::prelude::*;
    let (m, k, n) = (args.m, args.k, args.n);
    let nworkers = pool.map_or_else(rayon::current_num_threads, |p| p.current_num_threads());
    // Load-balance target: 3 blocks per worker, unless `WUKONG_GEMM_TASK_MACS` overrides the
    // grain (see [`gemm_task_macs`]) — the default is byte-for-byte the old `3 × nworkers`.
    let macs = m as u64 * n as u64 * k as u64;
    let target = blocks_target_from(macs, gemm_task_macs(), 3 * nworkers.max(1));
    let (bm, bn) = select_2d_block_shape(m, n, target);
    let (nbi, nbj) = (m.div_ceil(bm), n.div_ceil(bn));
    // Per-worker scratch caps: one block's A/B slices (not whole-matrix panels), rounded up to
    // full micropanels exactly as the other kernels size their scratch.
    let ap_cap = round_up(bm.min(m), MR) * k.min(KC);
    let bp_cap = round_up(bn.min(n), NR) * k.min(KC);
    // Send-safe scalar captures — the same raw-pointer-as-usize idiom `sgemm_avx2_parallel` and
    // `pack_ab_par` already use (the closure outlives nothing: `install`/`for_each` join before
    // this frame returns).
    let (a_addr, b_addr, c_addr) = (args.a as usize, args.b as usize, args.c as usize);
    let (beta, bt) = (args.beta, args.bt);
    let (epi_on, epi_bias_addr, epi_act, epi_alpha) = match args.epi {
        Some(e) => (true, e.bias as usize, e.act, e.alpha),
        None => (false, 0usize, 0u32, 1.0f32),
    };
    let body = move || {
        (0..nbi * nbj).into_par_iter().for_each(|t| {
            let (bi, bj) = (t / nbj, t % nbj);
            let (i0, j0) = (bi * bm, bj * bn);
            // A null bias-addr (0) stays null; the block body shifts to its global column origin.
            let epi = epi_on.then(|| Epilogue {
                bias: epi_bias_addr as *const f32,
                act: epi_act,
                alpha: epi_alpha,
            });
            // This worker's reusable scratch, taken out and put back so no borrow spans the kernel.
            let (mut ap, mut bp) = PACK_SCRATCH_2D.with(|s| {
                let mut g = s.borrow_mut();
                (std::mem::take(&mut g.0), std::mem::take(&mut g.1))
            });
            if ap.len() < ap_cap {
                ap.resize(ap_cap, 0.0);
            }
            if bp.len() < bp_cap {
                bp.resize(bp_cap, 0.0);
            }
            // SAFETY: each (bi, bj) C block has exactly one owner (disjoint writes); A/B are read-
            // only; scratch is worker-private; avx2+fma verified by `gemm_dispatch`.
            unsafe {
                sgemm_2d_block(
                    a_addr as *const f32,
                    b_addr as *const f32,
                    c_addr as *mut f32,
                    k,
                    n,
                    beta,
                    bt,
                    epi,
                    i0,
                    j0,
                    (m - i0).min(bm),
                    (n - j0).min(bn),
                    ap.as_mut_ptr(),
                    bp.as_mut_ptr(),
                );
            }
            PACK_SCRATCH_2D.with(|s| {
                let mut g = s.borrow_mut();
                g.0 = ap;
                g.1 = bp;
            });
        });
    };
    match pool {
        Some(p) => p.install(body),
        None => body(),
    }
}

/// One 2D block's whole-K GEMM: the owner walks every K-block ascending — pack this block's A/B
/// slices (offset to the block's row/col origin `i0`/`j0`), then [`macro_kernel`] into the block's
/// C region with `ldc = n`. `beta` on the first K-block, accumulate after, epilogue (shifted to
/// the block's global column origin) on the final K-block only — the serial kernel's exact rules,
/// so per-(i,j) accumulation is bit-identical (see [`sgemm_2d_blocks`]).
///
/// # Safety
/// `a`/`b` valid per the [`wukong_sgemm`]/[`wukong_sgemm_nt`] contract with `k`/`n` the full
/// matrix dims; `[i0, i0+bm) × [j0, j0+bn)` in range of C; `ap`/`bp` sized for
/// `round_up(bm,MR)·min(k,KC)` / `round_up(bn,NR)·min(k,KC)` f32; avx2+fma available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_2d_block(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    beta: f32,
    bt: bool,
    epi: Option<Epilogue>,
    i0: usize,
    j0: usize,
    bm: usize,
    bn: usize,
    ap: *mut f32,
    bp: *mut f32,
) {
    let mut pc = 0;
    while pc < k {
        let kc = (k - pc).min(select_kc(k));
        // First K-block honors the caller's beta; later blocks accumulate — and the epilogue folds
        // into the C writeback on the FINAL K-block only, bias base shifted to this block's global
        // column origin (macro_kernel adds the tile-local j0 shift, mirroring the serial
        // `shift(jc)`-then-`shift(j0)` double shift bit-for-bit).
        let beta_eff = if pc == 0 { beta } else { 1.0 };
        let block_epi = if pc + kc == k {
            epi.map(|e| e.shift(j0))
        } else {
            None
        };
        // The serial kernel's own packers, offset to this block's origin: `pack_a` reads rows
        // [i0, i0+bm) × cols [pc, pc+kc) of A (lda = k), `pack_b_block` reads the (pc, j0) slice of
        // B honoring `bt` — both contracts take an arbitrary base/leading-dim, and bm/bn smaller
        // than m/n only shortens their row/col loops. Same source values, same panel layout, same
        // matrix-edge-only zero padding ⇒ byte-identical micropanels.
        pack_a(a.add(i0 * k + pc), k, bm, kc, ap);
        pack_b_block(b, k, n, pc, j0, kc, bn, bt, bp);
        macro_kernel(bm, bn, kc, ap, bp, c.add(i0 * n + j0), n, beta_eff, block_epi);
        pc += kc;
    }
}

/// [`sgemm_2d_block`] for the skinny shared-A regime ([`gemm_skinny_a`]): the block's row origin
/// is 0 (single block row) and its packed A comes pre-packed from the call-shared buffer — the
/// `chunks` table gives each K-chunk's `(pc, kc, packed offset)`, the same [`select_kc`] walk the
/// per-block body does, over panels [`pack_a_panel`] laid out exactly as `pack_a` would. Only the
/// B slice is packed here (per block, per K-chunk — zero redundancy in a one-row grid: each B
/// column slice has exactly one owner). `beta` / final-K epilogue rules are byte-identical to the
/// per-block body, so serial == shared-A bit-for-bit.
///
/// # Safety
/// `ap_shared` valid packed-A for the whole `chunks` walk (`round_up(m,MR)·k` f32, fully packed
/// before this call — the pack phase's join); `b` per the [`wukong_sgemm_nt`] contract;
/// `[0, m) × [j0, j0+bn)` in range of C; `bp` sized `round_up(bn,NR)·min(k,KC)` f32; avx2+fma.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_2d_block_shared_a(
    ap_shared: *const f32,
    chunks: &[(usize, usize, usize)],
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    beta: f32,
    bt: bool,
    epi: Option<Epilogue>,
    j0: usize,
    m: usize,
    bn: usize,
    bp: *mut f32,
) {
    for &(pc, kc, off) in chunks {
        let beta_eff = if pc == 0 { beta } else { 1.0 };
        let block_epi = if pc + kc == k {
            epi.map(|e| e.shift(j0))
        } else {
            None
        };
        pack_b_block(b, k, n, pc, j0, kc, bn, bt, bp);
        macro_kernel(m, bn, kc, ap_shared.add(off), bp, c.add(j0), n, beta_eff, block_epi);
    }
}

/// A cache-line-isolated claim counter for the dynamic scheduler: 128-byte alignment (a 64 B line
/// plus its adjacent-line-prefetch pair) keeps the hot `fetch_add` word from false-sharing with
/// whatever the caller's stack frame packs around it. The counter is the ONE cross-worker write
/// of the whole dynamic path, so its line must bounce only for the claims themselves.
#[cfg(target_arch = "x86_64")]
#[repr(align(128))]
struct PaddedCounter(std::sync::atomic::AtomicUsize);

/// The steal-order block enumeration (`WUKONG_GEMM_STEAL_ORDER=1` — see [`gemm_steal_order`]): a
/// fixed, deterministic permutation of the row-major block indices `0..nbi·nbj` that lists every
/// FULL `bm×bn` block first (row-major among themselves), then the M/N edge-remainder blocks last
/// (row-major among themselves). Under dynamic claiming the queue drains front-to-back, so the
/// large blocks are in flight early — on whichever cores pull fastest, i.e. the P-cores,
/// naturally, with no pinning (pinning is the refuted lever) — and the join straggle is bounded
/// by a small remainder block, not a full one (LPT scheduling). A permutation of the same set:
/// each index appears exactly once (pinned by `dyn_block_order_is_a_permutation`), so every C
/// block keeps exactly one owner and the bits cannot change.
#[cfg(target_arch = "x86_64")]
fn dyn_block_order(m: usize, n: usize, bm: usize, bn: usize, nbi: usize, nbj: usize) -> Vec<u32> {
    let full = |t: usize| {
        let (bi, bj) = (t / nbj, t % nbj);
        (bi + 1) * bm <= m && (bj + 1) * bn <= n
    };
    let mut order = Vec::with_capacity(nbi * nbj);
    order.extend((0..nbi * nbj).filter(|&t| full(t)).map(|t| t as u32));
    order.extend((0..nbi * nbj).filter(|&t| !full(t)).map(|t| t as u32));
    order
}

/// The **dynamic-queue** variant of [`sgemm_2d_blocks`] (the DEFAULT parallel-GEMM scheduler
/// since the 2026-07-11 central A/B — see [`gemm_dyn`] for the measured basis;
/// `WUKONG_GEMM_DYN=0` opts back out): the same 2D C-block grid, block body
/// ([`sgemm_2d_block`]) and per-worker packing, but blocks are handed out by a shared atomic
/// claim counter instead of rayon's range-split work-stealing. Rayon's parallel iterator deals
/// the range out in contiguous chunks and rebalances by stealing *halves* of chunks; on this
/// asymmetric 6P+8E+2LPE pool that still leaves coarse ownership — an E-core can sit on a
/// multi-block chunk while a finished P-core has nothing left to steal but another chunk's half.
/// The claim queue is block-granular self-scheduling: every worker pulls ONE block at a time, so
/// core-speed asymmetry translates directly into more blocks on faster cores, and the tail
/// imbalance is at most one block per worker (with `WUKONG_GEMM_STEAL_ORDER=1`, at most one
/// *small* edge block — see [`dyn_block_order`]).
///
/// **Per-call overhead audit** (deliberately lower than the static path's):
/// * scheduling — ONE `Relaxed` `fetch_add` per block on a [`PaddedCounter`] (no locks, no
///   per-block rayon job push/steal, no split tree); one `pool.install` fork-join per call and
///   one plain fork-join over `nworkers` claim-loop tasks inside it;
/// * allocations — ZERO per call on the default path (the pack scratch is the same per-worker
///   [`PACK_SCRATCH_2D`] reuse; the caller-frame counter is stack); the steal-order probe's
///   permutation `Vec` is the one exception, and it is off by default;
/// * TLS traffic — the scratch take/put runs ONCE per worker per call, not once per block as the
///   static path's per-task `with` does.
/// For reference, the rest of the entry path is already lean: `gemm_dispatch` is OnceLock reads +
/// cached feature tests, and the generic `wukong_parallel_for` broadcast machinery (lib.rs) is
/// NOT on this path — recognized GEMMs call these kernels directly.
///
/// **Bit-exactness (the hard invariant).** Identical to [`sgemm_2d_blocks`]'s argument: the grid,
/// the enumeration `t → (t/nbj, t%nbj)`, and each block's whole-K ascending-`pc` body are the
/// same — the counter (and the optional steal-order permutation) only choose WHICH worker runs a
/// block and WHEN, and each index is claimed exactly once (`fetch_add` uniqueness). No result
/// byte depends on claim order: blocks write disjoint C regions and each block's per-(i,j)
/// K-accumulation order is fixed inside [`sgemm_2d_block`] / [`sgemm_2d_block_shared_a`].
/// serial == dynamic bit-for-bit for every (thread count × steal order × shared-A × grain)
/// combination (pinned by `sgemm_2d_dyn_matches_serial` / `sgemm_2d_dyn_thread_sweep`).
///
/// `steal_order` and `skinny_shared_a` are parameters (not env reads) so the differential tests
/// can exercise every combination regardless of process env; [`gemm_dispatch`] passes
/// [`gemm_steal_order`] / [`gemm_skinny_a`].
///
/// # Safety
/// Operand-size contract of [`sgemm_avx2_parallel`]; AVX2/FMA must be available (verified by
/// [`gemm_dispatch`]).
#[cfg(target_arch = "x86_64")]
unsafe fn sgemm_2d_blocks_dyn(
    pool: Option<&rayon::ThreadPool>,
    args: GemmArgs,
    steal_order: bool,
    skinny_shared_a: bool,
) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let (m, k, n) = (args.m, args.k, args.n);
    let nworkers = pool
        .map_or_else(rayon::current_num_threads, |p| p.current_num_threads())
        .max(1);
    // Same grid policy as the static path (grain override included), so an A/B between the two
    // schedulers compares scheduling alone, never block shape.
    let macs = m as u64 * n as u64 * k as u64;
    let target = blocks_target_from(macs, gemm_task_macs(), 3 * nworkers);
    let (bm, bn) = select_2d_block_shape(m, n, target);
    let (nbi, nbj) = (m.div_ceil(bm), n.div_ceil(bn));
    let nblocks = nbi * nbj;
    let ap_cap = round_up(bm.min(m), MR) * k.min(KC);
    let bp_cap = round_up(bn.min(n), NR) * k.min(KC);
    // Skinny shared-A regime (INSTRUMENT, default OFF — refuted as a win; see the measured
    // basis at [`gemm_skinny_a`]): ONE block row ⇒ every block's A slice is the same whole-A
    // rows, so per-block packing is redundant traffic — pack A once per call instead of once per
    // column block. A single-block grid gains nothing, and past the buffer cap the packed copy
    // would leave L3 (then per-block chunk packing is the better trade).
    let a_rows = round_up(m, MR);
    let use_shared_a =
        skinny_shared_a && nbi == 1 && nblocks > 1 && a_rows * k * 4 <= SKINNY_A_MAX_BYTES;
    // The K-chunk walk (pc, kc, packed-A offset) shared by the pack phase and every block body —
    // the same `select_kc` grouping `sgemm_2d_block` walks, so packed bytes and K order match the
    // per-block path exactly. Empty (no allocation) outside the shared-A regime.
    let mut chunks: Vec<(usize, usize, usize)> = Vec::new();
    if use_shared_a {
        let (mut pc, mut off) = (0usize, 0usize);
        while pc < k {
            let kc = (k - pc).min(select_kc(k));
            chunks.push((pc, kc, off));
            off += a_rows * kc;
            pc += kc;
        }
    }
    // Call-lifetime shared-A buffer, reused across calls via the caller-side thread-local. The
    // Vec stays in THIS frame (the workers see only the address): `install` joins before the
    // frame returns, and the buffer is put back after the join.
    let mut ap_shared: Vec<f32> = Vec::new();
    if use_shared_a {
        ap_shared = PACK_SCRATCH_SKINNY_A.with(|s| std::mem::take(&mut *s.borrow_mut()));
        if ap_shared.len() < a_rows * k {
            ap_shared.resize(a_rows * k, 0.0);
        }
    }
    let ap_shared_addr = ap_shared.as_mut_ptr() as usize;
    // Optional LPT-style enumeration; `None` = identity (row-major), allocation-free.
    let order: Option<Vec<u32>> = steal_order.then(|| dyn_block_order(m, n, bm, bn, nbi, nbj));
    let order_ref: Option<&[u32]> = order.as_deref();
    // The claim counter lives in this frame: `install`/`for_each` join before it drops, so
    // sharing it by reference is sound (the lifetime discipline every kernel here uses).
    let queue = PaddedCounter(AtomicUsize::new(0));
    let queue_ref = &queue;
    // Send-safe scalar captures — the raw-pointer-as-usize idiom of the sibling kernels.
    let (a_addr, b_addr, c_addr) = (args.a as usize, args.b as usize, args.c as usize);
    let (beta, bt) = (args.beta, args.bt);
    let (epi_on, epi_bias_addr, epi_act, epi_alpha) = match args.epi {
        Some(e) => (true, e.bias as usize, e.act, e.alpha),
        None => (false, 0usize, 0u32, 1.0f32),
    };
    let body = move || {
        // Shared-A pack phase (shared-A regime only): (K-chunk × MR-panel) granules across the
        // pool, `pack_a_panel` writing the same bytes the per-block `pack_a` would. The region's
        // implicit join orders every packed byte before any claim-loop read (the same
        // happens-before discipline `sgemm_2d_shared` uses — no barriers). Below the fork-join
        // crossover the whole pack runs serially on one worker: same bytes, no worker wake.
        if use_shared_a {
            let mpanels = m.div_ceil(MR);
            if a_rows * k * 4 < pack_par_min_bytes() {
                for &(pc, kc, off) in &chunks {
                    // SAFETY: buffer sized `a_rows·k` above; A read-only; single writer here.
                    unsafe {
                        pack_a(
                            (a_addr as *const f32).add(pc),
                            k,
                            m,
                            kc,
                            (ap_shared_addr as *mut f32).add(off),
                        );
                    }
                }
            } else {
                let chunks_ref = &chunks;
                (0..chunks_ref.len() * mpanels).into_par_iter().for_each(|u| {
                    let (ci, ip) = (u / mpanels, u % mpanels);
                    let (pc, kc, off) = chunks_ref[ci];
                    // SAFETY: each (chunk, panel) granule owns a disjoint `MR·kc` destination
                    // slice; A is read-only; buffer sized `a_rows·k` above.
                    unsafe {
                        pack_a_panel(
                            (a_addr as *const f32).add(pc),
                            k,
                            m,
                            kc,
                            ip,
                            (ap_shared_addr as *mut f32).add(off + ip * kc * MR),
                        );
                    }
                });
            }
        }
        // One claim-loop task per worker — plain fork-join tasks, NOT `broadcast`: the loop never
        // waits on a sibling, so it is correct even if work-stealing runs several loops serially
        // on one thread (they just drain the queue in turn); none of the barrier-width reasoning
        // the persistent region needs applies here, and no nested-broadcast rayon-internals
        // caveat is incurred.
        let chunks_ref = &chunks;
        (0..nworkers).into_par_iter().for_each(|_| {
            // A null bias-addr (0) stays null; each block shifts to its global column origin.
            let epi = epi_on.then(|| Epilogue {
                bias: epi_bias_addr as *const f32,
                act: epi_act,
                alpha: epi_alpha,
            });
            // Per-WORKER scratch, claimed ONCE per claim loop (the static path pays this per
            // block). The loop between take and put-back never re-enters rayon or
            // PACK_SCRATCH_2D, so take/put pairs cannot interleave on a thread. The shared-A
            // regime skips the A half — its packed A is the call-shared buffer above.
            let (mut ap, mut bp) = PACK_SCRATCH_2D.with(|s| {
                let mut g = s.borrow_mut();
                (std::mem::take(&mut g.0), std::mem::take(&mut g.1))
            });
            if !use_shared_a && ap.len() < ap_cap {
                ap.resize(ap_cap, 0.0);
            }
            if bp.len() < bp_cap {
                bp.resize(bp_cap, 0.0);
            }
            loop {
                // The whole scheduler: one Relaxed fetch_add per block. Relaxed suffices — the
                // counter only hands out unique indices; a block's pack + compute happen entirely
                // on the claiming worker, and the fork-join's join publishes C to the caller.
                let claim = queue_ref.0.fetch_add(1, Relaxed);
                if claim >= nblocks {
                    break;
                }
                let t = order_ref.map_or(claim, |o| o[claim] as usize);
                let (bi, bj) = (t / nbj, t % nbj);
                let (i0, j0) = (bi * bm, bj * bn);
                // SAFETY: `fetch_add` hands each block index to exactly one worker (disjoint C
                // writes); A/B are read-only; scratch is worker-private (the shared packed A is
                // read-only past its phase join); avx2+fma verified by `gemm_dispatch`.
                unsafe {
                    if use_shared_a {
                        sgemm_2d_block_shared_a(
                            ap_shared_addr as *const f32,
                            chunks_ref,
                            b_addr as *const f32,
                            c_addr as *mut f32,
                            k,
                            n,
                            beta,
                            bt,
                            epi,
                            j0,
                            m,
                            (n - j0).min(bn),
                            bp.as_mut_ptr(),
                        );
                    } else {
                        sgemm_2d_block(
                            a_addr as *const f32,
                            b_addr as *const f32,
                            c_addr as *mut f32,
                            k,
                            n,
                            beta,
                            bt,
                            epi,
                            i0,
                            j0,
                            (m - i0).min(bm),
                            (n - j0).min(bn),
                            ap.as_mut_ptr(),
                            bp.as_mut_ptr(),
                        );
                    }
                }
            }
            PACK_SCRATCH_2D.with(|s| {
                let mut g = s.borrow_mut();
                g.0 = ap;
                g.1 = bp;
            });
        });
    };
    match pool {
        Some(p) => p.install(body),
        None => body(),
    }
    if use_shared_a {
        PACK_SCRATCH_SKINNY_A.with(|s| *s.borrow_mut() = ap_shared);
    }
}

/// The **shared-pack 2D block-parallel** GEMM — RETIRED as a default (see [`gemm_2d_shared`]:
/// per-block packing won every adjacent pairing after pool unification, 2026-07-10/11), retained
/// fully tested as the `WUKONG_GEMM_2D_SHARED=1|band` A/B instrument.
///
/// **Why this exists.** The per-block 2D path re-packs each K-block's operands per C block: every
/// A element is packed once per block *column* (`nbj`×) and every B element once per block *row*
/// (`nbi`×). Measured grids: 512³ → (48,96) → 11×6 (A 6×, B 11×; ~17 MiB packed vs the ~2 MiB
/// minimum); 1024³ → (144,192) → 8×6 (56 vs 8 MiB); 128×768×768 → 6×16 (A 16×, B 6× — packing a
/// plausible 30–50% of the call); 128×3072×768 → A 16× + B 6× ≈ 80 MiB packed for 604 MFLOP.
/// This path keeps the 2D grid's per-thread L2-resident C blocks but packs each K-block's
/// whole-matrix A/B panels ONCE:
///
/// * The K loop runs sequentially (ascending `pc`, [`select_kc`] groups) inside ONE
///   `pool.install`. Per K-block, phase 1 packs the full `m×kc` A and `n×kc` B panels
///   cooperatively via [`pack_ab_par`] (the fused A+B rayon region, `jc = 0` / `nc = n`, with its
///   serial small-pack fallback) into caller-owned shared buffers ([`PACK_SCRATCH_2D_SHARED`]);
///   phase 2 is one rayon task per `BM×BN` C block, each pointed at its slice of the shared
///   packs. Both phases are synchronous `for_each` regions, so every pack write happens-before
///   every compute read (the implicit join) — no barriers, no phase counters.
/// * A block's slice of the shared pack is CONTIGUOUS: `BM`/`BN` are MR/NR multiples
///   ([`select_2d_block_shape`]), so block row `bi`'s A micropanels are exactly global panels
///   `bi·BM/MR ..` at `ap + (bi·BM/MR)·MR·kc` and block column `bj`'s B panels are global panels
///   `bj·BN/NR ..` at `bp + (bj·BN/NR)·NR·kc` — [`macro_kernel`] runs on those offsets unchanged.
///
/// **Bit-exactness (the hard invariant).** The per-block path's argument, with the pack hoisted:
/// the shared buffers hold the SAME BYTES `pack_a`/`pack_b_block` produce per block (same
/// `pack_a_panel`/`pack_b_panel`/`pack_b_trans_panel` on the same sources; interior block
/// boundaries are micropanel boundaries, so zero-padding occurs only at the true matrix edge —
/// exactly where the serial full-panel pack pads). Each C element is accumulated by exactly one
/// owner per K-block, over the same [`select_kc`] groups in ascending `pc`, with
/// `beta_eff = if pc == 0 { beta } else { 1.0 }`, the epilogue folded on the final K-block only
/// and shifted to the block's global column origin (the serial `shift` discipline), through the
/// same [`micro_6x16`]. Serial == shared-2D bit-for-bit REGARDLESS of block shape (pinned by
/// `sgemm_2d_shared_matches_serial`).
///
/// # Safety
/// Operand-size contract of [`sgemm_avx2_parallel`]; AVX2/FMA must be available (verified by
/// [`gemm_dispatch`]).
#[cfg(target_arch = "x86_64")]
unsafe fn sgemm_2d_shared(pool: Option<&rayon::ThreadPool>, args: GemmArgs) {
    use rayon::prelude::*;
    let (m, k, n) = (args.m, args.k, args.n);
    let nworkers = pool.map_or_else(rayon::current_num_threads, |p| p.current_num_threads());
    // Work-scaled load-balance target: a problem near the parallel gate can't feed 3×nworkers
    // blocks each worth a task dispatch, so the effective worker count is capped by the per-task
    // MAC budget ([`min_task_macs`]). The cap binds from the parallel gate up: at [`PAR_MIN_MACS`]
    // (2^23 MACs) and the 4 Mi default budget it yields w_eff == 2, and only a shape ≳ 2^26 MACs
    // reaches w_eff == nworkers on the 16-worker pool — so small problems near the gate (including
    // the 256³ crossover sweep under a lowered `WUKONG_GEMM_PAR_MIN_MACS`) get a few large blocks.
    let macs = m as u64 * n as u64 * k as u64;
    let w_eff = (macs / min_task_macs()).clamp(1, nworkers.max(1) as u64) as usize;
    let (bm, bn) = select_2d_block_shape(m, n, 3 * w_eff);
    let (nbi, nbj) = (m.div_ceil(bm), n.div_ceil(bn));
    // The shared buffers hold the WHOLE matrices' panels for one K-block — not one C block's
    // slice, and (unlike the NC-blocked kernels) not capped at NC: every block column reads the
    // same pack, so it spans all of n.
    let ap_cap = round_up(m, MR) * k.min(KC);
    let bp_cap = round_up(n, NR) * k.min(KC);
    let (mut ap, mut bp) = PACK_SCRATCH_2D_SHARED.with(|s| {
        let mut g = s.borrow_mut();
        (std::mem::take(&mut g.0), std::mem::take(&mut g.1))
    });
    if ap.len() < ap_cap {
        ap.resize(ap_cap, 0.0);
    }
    if bp.len() < bp_cap {
        bp.resize(bp_cap, 0.0);
    }
    // Send-safe scalar captures — the raw-pointer-as-usize idiom every parallel kernel here uses
    // (`install`/`for_each` join before this frame returns, so the pointers outlive every task).
    let (a_addr, b_addr, c_addr) = (args.a as usize, args.b as usize, args.c as usize);
    let (ap_addr, bp_addr) = (ap.as_mut_ptr() as usize, bp.as_mut_ptr() as usize);
    let (beta, bt) = (args.beta, args.bt);
    let (epi_on, epi_bias_addr, epi_act, epi_alpha) = match args.epi {
        Some(e) => (true, e.bias as usize, e.act, e.alpha),
        None => (false, 0usize, 0u32, 1.0f32),
    };
    let body = move || {
        let mut pc = 0;
        while pc < k {
            let kc = (k - pc).min(select_kc(k));
            // First K-block honors the caller's beta; later blocks accumulate — and the epilogue
            // folds into the C writeback on the FINAL K-block only (shifted per block below).
            let beta_eff = if pc == 0 { beta } else { 1.0 };
            let is_last_k = pc + kc == k;
            // Phase 1 — pack the whole m×kc A and n×kc B panels once, cooperatively. The shared
            // buffer bytes equal what `pack_a`/`pack_b_block` produce for the blocks they feed:
            // same panel functions, same sources, same layout.
            unsafe {
                pack_ab_par(
                    (a_addr as *const f32).add(pc),
                    k,
                    m,
                    b_addr as *const f32,
                    k,
                    n,
                    pc,
                    0,
                    kc,
                    n,
                    bt,
                    ap_addr as *mut f32,
                    bp_addr as *mut f32,
                );
            }
            // Phase 2 — one task per C block over the shared packs. Phase 1's region joined
            // above, so all pack writes happen-before these reads; this `for_each` joins before
            // the next K-block's pack overwrites the buffers.
            (0..nbi * nbj).into_par_iter().for_each(|t| {
                let (bi, bj) = (t / nbj, t % nbj);
                let (i0, j0) = (bi * bm, bj * bn);
                // A null bias-addr (0) stays null; shift to this block's global column origin
                // (macro_kernel adds the tile-local shift — the serial double-shift discipline).
                let block_epi = (epi_on && is_last_k).then(|| {
                    Epilogue {
                        bias: epi_bias_addr as *const f32,
                        act: epi_act,
                        alpha: epi_alpha,
                    }
                    .shift(j0)
                });
                // SAFETY: each (bi, bj) C block has exactly one owner (disjoint writes); the
                // packed panels are shared read-only (pack writes ordered by the phase join);
                // block boundaries are MR/NR-aligned so the offsets below address whole global
                // panels; avx2+fma verified by `gemm_dispatch`.
                unsafe {
                    macro_kernel(
                        (m - i0).min(bm),
                        (n - j0).min(bn),
                        kc,
                        (ap_addr as *const f32).add((i0 / MR) * kc * MR),
                        (bp_addr as *const f32).add((j0 / NR) * kc * NR),
                        (c_addr as *mut f32).add(i0 * n + j0),
                        n,
                        beta_eff,
                        block_epi,
                    );
                }
            });
            pc += kc;
        }
    };
    match pool {
        Some(p) => p.install(body),
        None => body(),
    }
    PACK_SCRATCH_2D_SHARED.with(|s| {
        let mut g = s.borrow_mut();
        g.0 = ap;
        g.1 = bp;
    });
}

/// Portable reference: a straight triple loop (`C = A·B`, or `C = A·Bᵀ` when `bt`). Correct for any
/// target; also the no-AVX fallback. Accumulation order differs from the blocked kernel, so it is
/// only used where the AVX path is unavailable (the two agree within f32 tolerance in tests).
#[allow(clippy::too_many_arguments)]
fn sgemm_scalar(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    beta: f32,
    bt: bool,
    epi: Option<Epilogue>,
) {
    unsafe {
        for i in 0..m {
            let crow = c.add(i * n);
            if beta == 0.0 {
                for j in 0..n {
                    *crow.add(j) = 0.0;
                }
            }
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let bjp = if bt {
                        *b.add(j * k + p)
                    } else {
                        *b.add(p * n + j)
                    };
                    acc += *a.add(i * k + p) * bjp;
                }
                let mut val = *crow.add(j) + acc;
                // No K-blocking here, so the sum is final: fold in the epilogue (j is the global col).
                if let Some(e) = epi {
                    val = e.apply(val, j);
                }
                *crow.add(j) = val;
            }
        }
    }
}

// --- AVX2 + FMA path -------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_avx2(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    beta: f32,
    bt: bool,
    epi: Option<Epilogue>,
) {
    // Pack scratch, sized to the actual blocks needed (never larger than the cache-block caps).
    // Reused across calls via a per-thread buffer (see PACK_SCRATCH): the packers overwrite every
    // element they read back (data + edge padding), so no re-zeroing is needed, and we avoid a
    // per-call malloc+zero that was a measurable slice of small-GEMM time. Taken out here and put
    // back at the end; the body between is panic-free so the buffers are always returned.
    let kc_max = k.min(KC);
    let mc_max = m.min(MC);
    let nc_max = n.min(NC);
    let ap_cap = round_up(mc_max, MR) * kc_max;
    let bp_cap = round_up(nc_max, NR) * kc_max;
    let (mut ap, mut bp) = PACK_SCRATCH.with(|c| {
        let mut g = c.borrow_mut();
        (std::mem::take(&mut g.0), std::mem::take(&mut g.1))
    });
    if ap.len() < ap_cap {
        ap.resize(ap_cap, 0.0);
    }
    if bp.len() < bp_cap {
        bp.resize(bp_cap, 0.0);
    }

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let mut pc = 0;
        while pc < k {
            let kc = (k - pc).min(select_kc(k));
            // First K-block honors the caller's beta; later blocks must accumulate the partial sums.
            let beta_eff = if pc == 0 { beta } else { 1.0 };
            // The fused epilogue applies only once the K reduction is complete — i.e. on the final
            // K-block — and is shifted to this column block's bias entries.
            let is_last_k = pc + kc == k;
            let block_epi = if is_last_k {
                epi.map(|e| e.shift(jc))
            } else {
                None
            };
            pack_b_block(b, k, n, pc, jc, kc, nc, bt, bp.as_mut_ptr());
            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(MC);
                pack_a(a.add(ic * k + pc), k, mc, kc, ap.as_mut_ptr());
                macro_kernel(
                    mc,
                    nc,
                    kc,
                    ap.as_ptr(),
                    bp.as_ptr(),
                    c.add(ic * n + jc),
                    n,
                    beta_eff,
                    block_epi,
                );
                ic += MC;
            }
            pc += kc;
        }
        jc += NC;
    }

    // Return the (now larger) buffers to the per-thread cache for the next call to reuse.
    PACK_SCRATCH.with(|c| {
        let mut g = c.borrow_mut();
        g.0 = ap;
        g.1 = bp;
    });
}

/// Pack the B panel for cache block `(pc, jc)` into `bp`, honoring the `C = A·Bᵀ` layout when `bt`.
/// Both layouts produce the same `[kc][NR]`-per-panel packed form, so the microkernel is unchanged.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pack_b_block(
    b: *const f32,
    k: usize,
    n: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    bt: bool,
    bp: *mut f32,
) {
    if bt {
        // B is [n, k] row-major; column j of Bᵀ is row j of B (stride k).
        pack_b_trans(b.add(jc * k + pc), k, kc, nc, bp);
    } else {
        pack_b(b.add(pc * n + jc), n, kc, nc, bp);
    }
}

/// Pack one `NR`-wide column panel `jp` of a row-major B slice into `[kc][NR]` contiguous form.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_b_panel(
    b: *const f32,
    ldb: usize,
    kc: usize,
    nc: usize,
    jp: usize,
    panel: *mut f32,
) {
    let j0 = jp * NR;
    let ncols = (nc - j0).min(NR);
    let mut dst = panel;
    for p in 0..kc {
        let src = b.add(p * ldb + j0);
        for j in 0..ncols {
            *dst.add(j) = *src.add(j);
        }
        for j in ncols..NR {
            *dst.add(j) = 0.0;
        }
        dst = dst.add(NR);
    }
}

/// Pack a `KC×NC` slice of B (row-major, leading dim `ldb`) into `NR`-wide column panels: panel `jp`
/// is `[kc][NR]` contiguous, zero-padded if the slice's last panel is partial.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_b(b: *const f32, ldb: usize, kc: usize, nc: usize, bp: *mut f32) {
    let npanels = nc.div_ceil(NR);
    for jp in 0..npanels {
        pack_b_panel(b, ldb, kc, nc, jp, bp.add(jp * kc * NR));
    }
}

/// Pack a `KC×NC` slice of Bᵀ — read B as `[nc rows × kc cols]` (leading dim `ldb`) and write the
/// `[kc][NR]`-per-panel form the microkernel expects. Reads run *contiguously* down each B row (the
/// `p` loop is innermost) and scatter into the small, L1-resident packed panel; the naive transpose
/// order (strided B reads) thrashes cache and dominates runtime, so this order is essential.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_b_trans_panel(
    b: *const f32,
    ldb: usize,
    kc: usize,
    nc: usize,
    jp: usize,
    panel: *mut f32,
) {
    let j0 = jp * NR;
    let ncols = (nc - j0).min(NR);
    for r in 0..ncols {
        let src = b.add((j0 + r) * ldb); // row (j0+r) of B, contiguous over the contraction
        for p in 0..kc {
            *panel.add(p * NR + r) = *src.add(p);
        }
    }
    for r in ncols..NR {
        for p in 0..kc {
            *panel.add(p * NR + r) = 0.0;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_b_trans(b: *const f32, ldb: usize, kc: usize, nc: usize, bp: *mut f32) {
    let npanels = nc.div_ceil(NR);
    for jp in 0..npanels {
        pack_b_trans_panel(b, ldb, kc, nc, jp, bp.add(jp * kc * NR));
    }
}

/// Pack an `MC×KC` slice of A (row-major, leading dim `lda`) into `MR`-wide row panels: panel `ip`
/// is `[kc][MR]` contiguous (so the microkernel broadcasts each row with unit stride), zero-padded.
/// Reads each A row *contiguously* over the contraction (`p` innermost) and scatters into the small
/// L1-resident panel; the transposed order (strided A reads, one cache line per element) thrashes
/// cache, so this order matters as much as it does for `pack_b_trans`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_a_panel(
    a: *const f32,
    lda: usize,
    mc: usize,
    kc: usize,
    ip: usize,
    panel: *mut f32,
) {
    let i0 = ip * MR;
    let nrows = (mc - i0).min(MR);
    for r in 0..nrows {
        let src = a.add((i0 + r) * lda); // row (i0+r) of A, contiguous over the contraction
        for p in 0..kc {
            *panel.add(p * MR + r) = *src.add(p);
        }
    }
    for r in nrows..MR {
        for p in 0..kc {
            *panel.add(p * MR + r) = 0.0;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_a(a: *const f32, lda: usize, mc: usize, kc: usize, ap: *mut f32) {
    let mpanels = mc.div_ceil(MR);
    for ip in 0..mpanels {
        pack_a_panel(a, lda, mc, kc, ip, ap.add(ip * kc * MR));
    }
}

// Below ~this much total packed data per K-block, even one fork-join's fixed cost (worker wake +
// join barrier — worst on this P+E hybrid, where a parked E-core is slow to arrive at either end)
// outweighs the whole copy: ~256 KB packs serially at cache speed in the same tens of µs a
// 16-worker region costs just to open and close. Env-overridable (`WUKONG_PACK_PAR_MIN_KB`,
// read once) so the crossover stays A/B-measurable on other machines.
fn pack_par_min_bytes() -> usize {
    use std::sync::OnceLock;
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| pack_min_bytes_from_kb(std::env::var("WUKONG_PACK_PAR_MIN_KB").ok()))
}

/// The `WUKONG_PACK_PAR_MIN_KB` policy: KiB → bytes with a **saturating** scale. A plain `* 1024`
/// overflowed usize for a knob value near `usize::MAX` — wrapping silently in release and aborting
/// a debug build with `attempt to multiply with overflow` raised inside the packer of an
/// `extern "C"` GEMM. Zero is a legitimate setting here (it means "always pack in parallel"), so
/// unlike the divisor knobs it is NOT filtered out. Pure (takes the already-read variable) so the
/// policy is unit-testable without process-global env state.
fn pack_min_bytes_from_kb(raw: Option<String>) -> usize {
    raw.and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256)
        .saturating_mul(1024)
}

/// Pack one K-block's A column-panel and B row-panel in a **single** parallel region: one task per
/// `MR`-row A panel plus one per `NR`-col B panel, over the combined index space. The previous
/// shape — `pack_a_par(); pack_b_block_par();` — paid two full fork-joins per K-block before the
/// compute region's third; at the mid sizes (512³–1024³, where a K-block's entire pack is ~1 MB)
/// those extra barriers were a leading term in the 60–70%-of-MKL-threaded scaling loss. One region
/// halves the pack barriers and lets work-stealing balance A- against B-panels; below
/// [`pack_par_min_bytes`] the copy is too small to amortize even one region, so both pack serially
/// (the compute region is still parallel). The packed bytes are identical however the panels are
/// distributed — layout and values never depend on the split — so serial == parallel bit-for-bit
/// and the differential oracle is untouched.
///
/// # Safety
/// Operand contracts of [`pack_a`] and [`pack_b_block`]; `ap`/`bp` sized as in
/// [`sgemm_avx2_parallel`] (`round_up(mc,MR)·kc` / `round_up(nc,NR)·kc` f32); avx2 verified by the
/// caller.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn pack_ab_par(
    a: *const f32,
    lda: usize,
    mc: usize,
    b: *const f32,
    k: usize,
    n: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    bt: bool,
    ap: *mut f32,
    bp: *mut f32,
) {
    use rayon::prelude::*;
    let mpanels = mc.div_ceil(MR);
    let npanels = nc.div_ceil(NR);
    let total_bytes = (round_up(mc, MR) + round_up(nc, NR)) * kc * 4;
    if total_bytes < pack_par_min_bytes() {
        pack_a(a, lda, mc, kc, ap);
        pack_b_block(b, k, n, pc, jc, kc, nc, bt, bp);
        return;
    }
    // Resolve B's base pointer + leading dim once (mirrors pack_b_block's bt dispatch).
    let (b_base, ldb) = if bt {
        (b.add(jc * k + pc) as usize, k)
    } else {
        (b.add(pc * n + jc) as usize, n)
    };
    let (a_addr, ap_addr, bp_addr) = (a as usize, ap as usize, bp as usize);
    // A/B kill-switch (`WUKONG_PACK_SPLIT_REGIONS=1`, read once): the pre-fusion two-region
    // shape — one fork-join per operand pack — kept so the fused-region win stays measurable
    // adjacent-run on any machine (the same instrument discipline as WUKONG_P4_NO_256).
    fn split_regions() -> bool {
        use std::sync::OnceLock;
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| std::env::var("WUKONG_PACK_SPLIT_REGIONS").is_ok_and(|v| v == "1"))
    }
    if split_regions() {
        (0..mpanels).into_par_iter().for_each(|ip| {
            // SAFETY: disjoint output panel per task; pointers re-derived per task.
            unsafe {
                pack_a_panel(
                    a_addr as *const f32,
                    lda,
                    mc,
                    kc,
                    ip,
                    (ap_addr as *mut f32).add(ip * kc * MR),
                );
            }
        });
        (0..npanels).into_par_iter().for_each(|jp| {
            let panel = (bp_addr as *mut f32).add(jp * kc * NR);
            // SAFETY: disjoint output panel per task; pointers re-derived per task.
            unsafe {
                if bt {
                    pack_b_trans_panel(b_base as *const f32, ldb, kc, nc, jp, panel);
                } else {
                    pack_b_panel(b_base as *const f32, ldb, kc, nc, jp, panel);
                }
            }
        });
        return;
    }
    (0..mpanels + npanels).into_par_iter().for_each(|t| {
        // SAFETY: every task writes a disjoint packed panel; pointers re-derived per task.
        unsafe {
            if t < mpanels {
                pack_a_panel(
                    a_addr as *const f32,
                    lda,
                    mc,
                    kc,
                    t,
                    (ap_addr as *mut f32).add(t * kc * MR),
                );
            } else {
                let jp = t - mpanels;
                let panel = (bp_addr as *mut f32).add(jp * kc * NR);
                if bt {
                    pack_b_trans_panel(b_base as *const f32, ldb, kc, nc, jp, panel);
                } else {
                    pack_b_panel(b_base as *const f32, ldb, kc, nc, jp, panel);
                }
            }
        }
    });
}

/// Run the `MR×NR` microkernel over every register tile of an `mc×nc` macro-block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn macro_kernel(
    mc: usize,
    nc: usize,
    kc: usize,
    ap: *const f32,
    bp: *const f32,
    c: *mut f32,
    ldc: usize,
    beta: f32,
    epi: Option<Epilogue>,
) {
    let mpanels = mc.div_ceil(MR);
    let npanels = nc.div_ceil(NR);
    for jp in 0..npanels {
        let j0 = jp * NR;
        let nrv = (nc - j0).min(NR);
        let bpanel = bp.add(jp * kc * NR);
        // Each column panel's bias starts `j0` columns into this macro-block's epilogue.
        let panel_epi = epi.map(|e| e.shift(j0));
        for ip in 0..mpanels {
            let i0 = ip * MR;
            let mrv = (mc - i0).min(MR);
            let apanel = ap.add(ip * kc * MR);
            micro_6x16(
                kc,
                apanel,
                bpanel,
                c.add(i0 * ldc + j0),
                ldc,
                beta,
                mrv,
                nrv,
                panel_epi,
            );
        }
    }
}

/// The `6×16` AVX2/FMA microkernel: 12 live `__m256` accumulators, 12 FMAs per K step. Computes the
/// full 6×16 tile (packed panels are zero-padded), then writes back the valid `mr×nr` corner with
/// the `beta` rule.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_6x16(
    kc: usize,
    ap: *const f32,
    bp: *const f32,
    c: *mut f32,
    ldc: usize,
    beta: f32,
    mr: usize,
    nr: usize,
    epi: Option<Epilogue>,
) {
    use std::arch::x86_64::*;
    // Prefetch the C tile this call's writeback will read/modify: `mr` rows of `nr` f32, rows
    // `ldc` apart — a stride no hardware prefetcher tracks — touching the first and last line of
    // each row. Issued before the K loop so the demand misses resolve under ~kc·12 FMAs of shadow
    // instead of stalling the writeback (the dominant uncovered miss at ≥2048³, where C never
    // fits cache and every K-block re-reads it).
    if gemm_pf_c() {
        for r in 0..mr {
            _mm_prefetch::<_MM_HINT_T0>(c.add(r * ldc) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(c.add(r * ldc + nr - 1) as *const i8);
        }
    }
    // AVX-512 fast path: the full-tile, no-epilogue case (the bulk of a large GEMM's tiles). Wider
    // 512-bit lanes, half the FMA instructions, and BIT-IDENTICAL accumulation order to the AVX2 body
    // below (see `micro_6x16_avx512`). Gated on `avx512f` — FALSE on this development box, so the
    // branch is dead code here (and the `&&` short-circuits to a single predicted-not-taken compare on
    // the AVX2 path, after the full-tile checks the AVX2 fast path already makes). On AVX-512 silicon
    // it takes over the hot tiles; the width win is a labelled projection, never measured here.
    if epi.is_none() && mr == MR && nr == NR && is_x86_feature_detected!("avx512f") {
        micro_6x16_avx512(kc, ap, bp, c, ldc, beta);
        return;
    }
    let (mut c0, mut c1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c2, mut c3) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c4, mut c5) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c6, mut c7) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c8, mut c9) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c10, mut c11) = (_mm256_setzero_ps(), _mm256_setzero_ps());

    let mut ap = ap;
    let mut bp = bp;
    // One K-step: load the 16-wide B row (two 256-bit lanes), then for each of the 6 A rows
    // load-broadcast it straight from the packed panel (`vbroadcastss [mem]`, 1 uop — frees the
    // shuffle/ALU ports for the FMAs) and FMA into the two lanes. 12 FMAs, 12 live accumulators.
    macro_rules! kstep {
        () => {{
            let b0 = _mm256_loadu_ps(bp);
            let b1 = _mm256_loadu_ps(bp.add(8));
            let a0 = _mm256_broadcast_ss(&*ap);
            c0 = _mm256_fmadd_ps(a0, b0, c0);
            c1 = _mm256_fmadd_ps(a0, b1, c1);
            let a1 = _mm256_broadcast_ss(&*ap.add(1));
            c2 = _mm256_fmadd_ps(a1, b0, c2);
            c3 = _mm256_fmadd_ps(a1, b1, c3);
            let a2 = _mm256_broadcast_ss(&*ap.add(2));
            c4 = _mm256_fmadd_ps(a2, b0, c4);
            c5 = _mm256_fmadd_ps(a2, b1, c5);
            let a3 = _mm256_broadcast_ss(&*ap.add(3));
            c6 = _mm256_fmadd_ps(a3, b0, c6);
            c7 = _mm256_fmadd_ps(a3, b1, c7);
            let a4 = _mm256_broadcast_ss(&*ap.add(4));
            c8 = _mm256_fmadd_ps(a4, b0, c8);
            c9 = _mm256_fmadd_ps(a4, b1, c9);
            let a5 = _mm256_broadcast_ss(&*ap.add(5));
            c10 = _mm256_fmadd_ps(a5, b0, c10);
            c11 = _mm256_fmadd_ps(a5, b1, c11);
            ap = ap.add(MR);
            bp = bp.add(NR);
        }};
    }
    // Unroll the K loop by 4: one loop branch + counter per 4 steps instead of per step (the
    // increment/compare otherwise contends with the FMAs for ports 0/1), and the scheduler gets a
    // wider window to overlap loads with the in-flight FMA chains. One prefetch per 4 steps keeps
    // the next B panel rows warm in L1 without flooding the load ports.
    //
    // `wrapping_add`, not `add`: the 8-K-step lookahead deliberately runs off the end of the
    // packed-B buffer near the end of the last micropanel (instrumented against the packed-B
    // length: exactly 256 bytes past, on every `sgemm_matches_naive_various_sizes` shape).
    // `<*const T>::add`'s precondition is that the result stays inside the same allocated object,
    // so it lowers to `getelementptr inbounds` on an address that is not — poison — even though
    // the prefetch itself is architecturally a hint that issues no load and faults on nothing.
    // `wrapping_add` carries no such precondition and lowers to the same `lea`, so the emitted
    // address and every computed bit are unchanged.
    let mut p = 0;
    while p + 4 <= kc {
        _mm_prefetch::<_MM_HINT_T0>(bp.wrapping_add(NR * 8) as *const i8);
        kstep!();
        kstep!();
        kstep!();
        kstep!();
        p += 4;
    }
    while p < kc {
        kstep!();
        p += 1;
    }

    // Fast path: a full 6×16 tile with no fused epilogue — store the 12 accumulators straight to C
    // (two 256-bit stores per row) with the beta rule, skipping the tmp spill + 96-element scalar
    // copy. The K loop dominates only at large K; at the small K of conv/attention (and the n=256
    // GEMM) the writeback is a real fraction, so this is a measurable win there and free elsewhere.
    if epi.is_none() && mr == MR && nr == NR {
        macro_rules! wb {
            ($lo:expr, $hi:expr, $r:expr) => {{
                let row = c.add($r * ldc);
                if beta == 0.0 {
                    _mm256_storeu_ps(row, $lo);
                    _mm256_storeu_ps(row.add(8), $hi);
                } else {
                    _mm256_storeu_ps(row, _mm256_add_ps(_mm256_loadu_ps(row), $lo));
                    _mm256_storeu_ps(row.add(8), _mm256_add_ps(_mm256_loadu_ps(row.add(8)), $hi));
                }
            }};
        }
        wb!(c0, c1, 0);
        wb!(c2, c3, 1);
        wb!(c4, c5, 2);
        wb!(c6, c7, 3);
        wb!(c8, c9, 4);
        wb!(c10, c11, 5);
        return;
    }

    // Fused-epilogue fast path: a FULL 6×16 tile on the final K-block. Apply the beta rule, the per-
    // column bias (16 cols = two 256-bit lanes, broadcast identically across the 6 rows), and the
    // activation — all 256-bit, instead of the 96-element scalar spill+writeback below. Bit-identical
    // to that scalar epilogue: `_mm256_max_ps`==`f32::max`, and `gelu8`/`silu8` mirror `gelu1`/`silu1`
    // lane-for-lane (the vmath tail-match invariant), so the interpreter's scalar oracle still agrees.
    // The bias add is *skipped* (not added-as-zero) when null, matching `Epilogue::apply` on ±0.
    if let Some(e) = epi {
        if mr == MR && nr == NR {
            let has_bias = !e.bias.is_null();
            let (bias_lo, bias_hi) = if has_bias {
                (_mm256_loadu_ps(e.bias), _mm256_loadu_ps(e.bias.add(8)))
            } else {
                (_mm256_setzero_ps(), _mm256_setzero_ps())
            };
            // α scale (skipped when 1.0 so the fused-epilogue callers stay byte-identical), broadcast
            // to all 8 lanes; applied to the matmul result before the bias-add, mirroring `apply`.
            let apply_alpha = e.alpha != 1.0;
            let alpha8 = _mm256_set1_ps(e.alpha);
            macro_rules! act8 {
                ($v:expr) => {{
                    match e.act {
                        ACT_RELU => _mm256_max_ps($v, _mm256_setzero_ps()),
                        ACT_GELU => crate::vmath::gelu8($v),
                        ACT_SILU => crate::vmath::silu8($v),
                        _ => $v,
                    }
                }};
            }
            macro_rules! wbe {
                ($lo:expr, $hi:expr, $r:expr) => {{
                    let row = c.add($r * ldc);
                    let mut lo = if beta == 0.0 {
                        $lo
                    } else {
                        _mm256_add_ps(_mm256_loadu_ps(row), $lo)
                    };
                    let mut hi = if beta == 0.0 {
                        $hi
                    } else {
                        _mm256_add_ps(_mm256_loadu_ps(row.add(8)), $hi)
                    };
                    if apply_alpha {
                        lo = _mm256_mul_ps(lo, alpha8);
                        hi = _mm256_mul_ps(hi, alpha8);
                    }
                    if has_bias {
                        lo = _mm256_add_ps(lo, bias_lo);
                        hi = _mm256_add_ps(hi, bias_hi);
                    }
                    _mm256_storeu_ps(row, act8!(lo));
                    _mm256_storeu_ps(row.add(8), act8!(hi));
                }};
            }
            wbe!(c0, c1, 0);
            wbe!(c2, c3, 1);
            wbe!(c4, c5, 2);
            wbe!(c6, c7, 3);
            wbe!(c8, c9, 4);
            wbe!(c10, c11, 5);
            return;
        }
    }

    // General path (partial edge tiles + the fused epilogue): spill the tile, then write back the
    // valid corner with the beta rule (and, on the final K-block, the fused bias+activation).
    let mut tmp = [0.0f32; MR * NR];
    _mm256_storeu_ps(tmp.as_mut_ptr(), c0);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(8), c1);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(16), c2);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(24), c3);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(32), c4);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(40), c5);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(48), c6);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(56), c7);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(64), c8);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(72), c9);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(80), c10);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(88), c11);
    for r in 0..mr {
        for j in 0..nr {
            let cp = c.add(r * ldc + j);
            let v = tmp[r * NR + j];
            let acc = if beta == 0.0 { v } else { *cp + v };
            // On the final K-block (epi present) fold bias + activation into this single store,
            // so C is never read back for a separate epilogue pass. `j` is the tile-local column.
            *cp = match epi {
                Some(e) => e.apply(acc, j),
                None => acc,
            };
        }
    }
}

/// The **AVX-512 twin** of [`micro_6x16`]'s K-accumulation + plain (no-epilogue) full-tile writeback.
/// Each of the 6 A-rows gets ONE 512-bit accumulator holding all `NR=16` of that row's C-columns —
/// versus the AVX2 kernel's two 256-bit halves (`c{2r}`, `c{2r+1}`). Per K-step: one 16-wide B load,
/// 6 A-broadcasts, 6 FMAs — **half** the AVX2 kernel's 12 FMAs for the same flops.
///
/// **Correctness is by construction, not measurement.** Lane `j` of accumulator `r` sums `a[r,p]·b[p,j]`
/// over `p` ascending — exactly the sequence `micro_6x16` accumulates into `c{2r}[j]` (`j<8`) /
/// `c{2r+1}[j−8]` (`j≥8`). Widening 2×ymm → 1×zmm changes only the register width, never which products
/// reach `C[i,j]` nor their order, so this is **bit-identical** to the AVX2 kernel — the very argument
/// the differential gate already makes for SIMD lane width (a wider vector holds *different output
/// elements*, not partial sums of one). `avx512f` is **false on this development box**, so this is DEAD
/// CODE here: it can touch no gate and no measured number. On AVX-512 silicon `micro_6x16_avx512_twin`
/// (a `#[test]`) asserts the bit-equality directly; the width win is reported only as a labelled
/// PROJECTION (see `prompts/results/cpu-library.md`), never measured on hardware that cannot run it.
///
/// Scope is deliberately the full-tile, no-epilogue case — the bulk of a large GEMM's tiles and the
/// part whose AVX-512 form is a trivial lane-width swap. Partial edge tiles and the fused bias+act
/// epilogue (which would need AVX-512 `gelu16`/`silu16` vmath that does not exist yet) fall back to the
/// proven AVX2 [`micro_6x16`], bounding the untestable surface to this small, structurally-trivial core.
///
/// # Safety
/// `avx512f` must be available (caller runtime-checks). Packed-panel/`ldc` contract of [`micro_6x16`],
/// restricted to a full `MR×NR` tile.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn micro_6x16_avx512(
    kc: usize,
    ap: *const f32,
    bp: *const f32,
    c: *mut f32,
    ldc: usize,
    beta: f32,
) {
    use std::arch::x86_64::*;
    let (mut c0, mut c1, mut c2) =
        (_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps());
    let (mut c3, mut c4, mut c5) =
        (_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps());
    let mut ap = ap;
    let mut bp = bp;
    // One K-step: load the 16-wide B row into a single zmm, then for each of the 6 A rows broadcast its
    // packed value and FMA into that row's accumulator. Identical (a, b) values and ascending-`p` order
    // as `micro_6x16::kstep`, so the bits match.
    macro_rules! kstep {
        () => {{
            let b = _mm512_loadu_ps(bp);
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(*ap), b, c0);
            c1 = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(1)), b, c1);
            c2 = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(2)), b, c2);
            c3 = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(3)), b, c3);
            c4 = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(4)), b, c4);
            c5 = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(5)), b, c5);
            ap = ap.add(MR);
            bp = bp.add(NR);
        }};
    }
    // Same ×4 K-unroll + one prefetch per 4 steps as the AVX2 kernel — including its
    // `wrapping_add` (the lookahead leaves the packed-B allocation on the last micropanel; see
    // [`micro_6x16`] for why `add` would be `inbounds` poison there).
    let mut p = 0;
    while p + 4 <= kc {
        _mm_prefetch::<_MM_HINT_T0>(bp.wrapping_add(NR * 8) as *const i8);
        kstep!();
        kstep!();
        kstep!();
        kstep!();
        p += 4;
    }
    while p < kc {
        kstep!();
        p += 1;
    }
    // Plain full-tile writeback, one 512-bit store per row, with the beta rule — the zmm analogue of
    // the AVX2 fast-path `wb!` (which stored each row as `[c{2r} | c{2r+1}]`).
    macro_rules! wb {
        ($acc:expr, $r:expr) => {{
            let row = c.add($r * ldc);
            if beta == 0.0 {
                _mm512_storeu_ps(row, $acc);
            } else {
                _mm512_storeu_ps(row, _mm512_add_ps(_mm512_loadu_ps(row), $acc));
            }
        }};
    }
    wb!(c0, 0);
    wb!(c1, 1);
    wb!(c2, 2);
    wb!(c3, 3);
    wb!(c4, 4);
    wb!(c5, 5);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two GEMM env knobs whose raw value reaches arithmetic that can trap. Both crashed a
    /// compiled program before this gate, from inside an `extern "C"` runtime kernel:
    /// `WUKONG_GEMM_2D_SHARED=1 WUKONG_GEMM_MIN_TASK_MACS=0 wukongc --run --backend=native -O2
    /// tests/run/scaled_gemm_parallel.wk` → `attempt to divide by zero` at the `macs /
    /// min_task_macs()` in `sgemm_2d_shared`, and `WUKONG_GEMM_2D=0
    /// WUKONG_PACK_PAR_MIN_KB=18446744073709551615` (debug build) → `attempt to multiply with
    /// overflow` in the KiB→byte scale. The policies are pure functions so this gate needs no
    /// process-global env state.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn gemm_env_knobs_survive_hostile_values() {
        // Divisor knob: 0 and unparsable are both "unset", so the caller's default stands.
        assert_eq!(positive_or_unset(Some("0".to_string())), None);
        assert_eq!(positive_or_unset(Some(String::new())), None);
        assert_eq!(positive_or_unset(Some("-1".to_string())), None);
        assert_eq!(positive_or_unset(Some("not-a-number".to_string())), None);
        assert_eq!(positive_or_unset(None), None);
        assert_eq!(positive_or_unset(Some("1".to_string())), Some(1));
        assert_eq!(positive_or_unset(Some("4194304".to_string())), Some(4 << 20));
        // The invariant the `macs / min_task_macs()` use site depends on, under this process's env.
        assert!(min_task_macs() > 0, "min_task_macs is used as a divisor");

        // Byte-threshold knob: 0 is a legitimate setting ("always pack in parallel") and must be
        // preserved, but the KiB→byte scale must saturate rather than wrap or trap.
        assert_eq!(pack_min_bytes_from_kb(Some("0".to_string())), 0);
        assert_eq!(pack_min_bytes_from_kb(None), 256 * 1024);
        assert_eq!(pack_min_bytes_from_kb(Some("junk".to_string())), 256 * 1024);
        assert_eq!(pack_min_bytes_from_kb(Some("1".to_string())), 1024);
        assert_eq!(
            pack_min_bytes_from_kb(Some(usize::MAX.to_string())),
            usize::MAX
        );
    }

    /// Naive reference, distinct from both production paths, for validation.
    fn naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    c[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        c
    }

    fn fill(seed: u64, n: usize) -> Vec<f32> {
        // Deterministic pseudo-random-ish values in [-1, 1).
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    fn check(m: usize, k: usize, n: usize) {
        let a = fill(1, m * k);
        let b = fill(2, k * n);
        let want = naive(&a, &b, m, k, n);
        let mut got = vec![0.0f32; m * n];
        unsafe {
            wukong_sgemm(
                a.as_ptr(),
                b.as_ptr(),
                got.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
                0,
            );
        }
        // Tolerance scales with K (the number of accumulated products).
        let tol = 1e-3 * (k as f32).sqrt();
        for i in 0..m * n {
            assert!(
                (got[i] - want[i]).abs() <= tol + 1e-4 * want[i].abs(),
                "({m}x{k}x{n}) idx {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn sgemm_matches_naive_various_sizes() {
        // Sizes that exercise every MR=6 / NR=16 / MC=72 / KC=256 remainder, incl. 1-wide dims.
        for &(m, k, n) in &[
            (1, 1, 1),
            (1, 256, 1),
            (5, 5, 5),
            (6, 16, 1),
            (7, 17, 13),
            (15, 31, 17),
            (64, 64, 64),
            (72, 256, 80),
            (73, 257, 15),
            (100, 100, 100),
            (128, 256, 512),
            (200, 200, 200),
        ] {
            check(m, k, n);
        }
    }

    #[test]
    fn sgemm_beta_accumulates() {
        let (m, k, n) = (40, 50, 60);
        let a = fill(3, m * k);
        let b = fill(4, k * n);
        let base = fill(5, m * n);
        let want: Vec<f32> = {
            let prod = naive(&a, &b, m, k, n);
            base.iter().zip(&prod).map(|(x, y)| x + y).collect()
        };
        let mut got = base.clone();
        unsafe {
            wukong_sgemm(
                a.as_ptr(),
                b.as_ptr(),
                got.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
                1,
            );
        }
        let tol = 1e-3 * (k as f32).sqrt();
        for i in 0..m * n {
            assert!((got[i] - want[i]).abs() <= tol + 1e-4 * want[i].abs());
        }
    }

    /// Throughput probe (run: `cargo test -p wukong_runtime --release -- --ignored --nocapture`).
    /// Sweeps square sizes so the parallel scaling (which improves with size, as the packs amortize
    /// and each core gets more compute per K-block) is visible, not just the small-matrix corner.
    /// Pin the current thread to one P-core (logical CPU 0) and raise priority for a repeatable
    /// single-core measurement; returns the previous affinity mask to restore before the parallel
    /// benches. On this hybrid laptop the single-core GFLOP/s otherwise swings ±30% with P/E
    /// scheduling and turbo, swamping microkernel changes. No-op (returns 0) off Windows.
    #[cfg(windows)]
    fn pin_pcore() -> usize {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn GetCurrentProcess() -> isize;
            fn SetThreadAffinityMask(h: isize, mask: usize) -> usize;
            fn SetThreadPriority(h: isize, prio: i32) -> i32;
            fn SetPriorityClass(h: isize, class: u32) -> i32;
        }
        unsafe {
            SetPriorityClass(GetCurrentProcess(), 0x0000_0080); // HIGH_PRIORITY_CLASS
            SetThreadPriority(GetCurrentThread(), 15); // THREAD_PRIORITY_TIME_CRITICAL
            SetThreadAffinityMask(GetCurrentThread(), 0x1) // logical CPU 0 (a P-core)
        }
    }
    #[cfg(windows)]
    fn restore_affinity(mask: usize) {
        if mask == 0 {
            return;
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadAffinityMask(h: isize, mask: usize) -> usize;
        }
        unsafe {
            SetThreadAffinityMask(GetCurrentThread(), mask);
        }
    }
    #[cfg(not(windows))]
    fn pin_pcore() -> usize {
        0
    }
    #[cfg(not(windows))]
    fn restore_affinity(_: usize) {}

    #[test]
    #[ignore]
    fn sgemm_throughput() {
        use std::time::Instant;
        for &n in &[256usize, 512, 1024, 2048] {
            let a = fill(1, n * n);
            let b = fill(2, n * n);
            let mut c = vec![0.0f32; n * n];
            let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
            let flops = 2.0 * (n as f64).powi(3);
            // Best (min) of many batches: on a busy box the fastest run is the least-interfered
            // estimate. Single-core benches run pinned (see `pin_pcore`) for repeatability.
            let bench = |label: &str, f: &dyn Fn()| {
                for _ in 0..5 {
                    f();
                }
                let mut best = f64::INFINITY;
                for _ in 0..50 {
                    let t = Instant::now();
                    f();
                    best = best.min(t.elapsed().as_secs_f64());
                }
                println!(
                    "n={n:<4} {label:<22}: {:6.1} GFLOP/s ({:.3} ms)",
                    flops / best / 1e9,
                    best * 1e3
                );
            };
            let prev = pin_pcore();
            bench("sgemm (1 core)", &|| unsafe {
                wukong_sgemm(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
            bench("sgemm_nt (1 core)", &|| unsafe {
                wukong_sgemm_nt(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
            restore_affinity(prev); // parallel benches want all cores
            bench("sgemm (parallel)", &|| unsafe {
                wukong_sgemm_parallel(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
            bench("sgemm_nt (parallel)", &|| unsafe {
                wukong_sgemm_nt_parallel(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
        }
    }

    /// Adjacent A/B: the SAME parallel GEMM run in the default (logical-core) rayon pool vs a private
    /// physical-core pool, measured back-to-back best-of-N so the laptop's thermal drift cancels in the
    /// ratio. Cross-RUN comparison of the two pools is hopeless (the power state swings ~3×, and the
    /// 1c-vs-par self-scaling ratio is dominated by *where* in the thermal cycle each leg is sampled);
    /// only this interleaved same-run ratio reliably answers whether shedding the HyperThread siblings
    /// lifts throughput on this compute-bound kernel. Run: `cargo test -p wukong_runtime --release
    /// sgemm_pool_ab -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn sgemm_pool_ab() {
        use std::time::Instant;
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            println!("no avx2/fma — skipping");
            return;
        }
        let logical = rayon::current_num_threads();
        let physical = num_cpus::get_physical().max(1);
        let ppool = rayon::ThreadPoolBuilder::new()
            .num_threads(physical)
            .build()
            .unwrap();
        println!("pools: logical={logical} physical={physical}");
        for &n in &[512usize, 1024, 2048, 4096] {
            let a = fill(1, n * n);
            let b = fill(2, n * n);
            let mut c = vec![0.0f32; n * n];
            let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
            let flops = 2.0 * (n as f64).powi(3);
            let logical_run = || unsafe { sgemm_avx2_parallel(ap, bp, cp, n, n, n, 0.0, false, None) };
            let physical_run = || {
                let args = GemmArgs { a: ap, b: bp, c: cp, m: n, k: n, n, beta: 0.0, bt: false, epi: None };
                ppool.install(move || unsafe { args.run() });
            };
            for _ in 0..3 {
                logical_run();
                physical_run();
            }
            let (mut bl, mut bphys) = (f64::INFINITY, f64::INFINITY);
            for _ in 0..12 {
                let t = Instant::now();
                logical_run();
                bl = bl.min(t.elapsed().as_secs_f64());
                let t = Instant::now();
                physical_run();
                bphys = bphys.min(t.elapsed().as_secs_f64());
            }
            let (lg, pg) = (flops / bl / 1e9, flops / bphys / 1e9);
            println!(
                "n={n:<4} logical(×{logical})={lg:6.1}  physical(×{physical})={pg:6.1} GFLOP/s  →  physical/logical = {:.2}×",
                pg / lg
            );
        }
    }

    /// Twin check for the AVX-512 microkernel: it must produce **bit-identical** output to the proven
    /// AVX2 [`micro_6x16`] on the same packed panels (full 6×16 tile, no epilogue, both beta rules).
    /// On AVX-512 silicon this asserts the equality directly; on this development box (no `avx512f`) it
    /// is skipped — but the body still **compiles**, exercising the kernel's form, and its correctness
    /// rests on the structural-twin argument documented on `micro_6x16_avx512`. This is the honest
    /// shape of an "AVX-512 twin test" on hardware that cannot execute AVX-512.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn micro_6x16_avx512_twin() {
        let have = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma");
        if !have {
            println!("micro_6x16_avx512_twin: host lacks avx512f — compile-checked, run skipped");
            return;
        }
        for &kc in &[1usize, 4, 7, 64] {
            let ap = fill(1, kc * MR);
            let bp = fill(2, kc * NR);
            for &beta in &[0.0f32, 1.0] {
                let cinit = fill(3, MR * NR);
                let mut c_avx2 = cinit.clone();
                let mut c_512 = cinit.clone();
                // SAFETY: features just checked; full MR×NR tile, ldc=NR, no epilogue — the AVX-512
                // path's supported case.
                unsafe {
                    micro_6x16(
                        kc, ap.as_ptr(), bp.as_ptr(), c_avx2.as_mut_ptr(), NR, beta, MR, NR, None,
                    );
                    micro_6x16_avx512(kc, ap.as_ptr(), bp.as_ptr(), c_512.as_mut_ptr(), NR, beta);
                }
                for i in 0..MR * NR {
                    assert_eq!(
                        c_avx2[i].to_bits(),
                        c_512[i].to_bits(),
                        "kc={kc} beta={beta} idx={i}: avx2={} avx512={}",
                        c_avx2[i],
                        c_512[i]
                    );
                }
            }
        }
    }

    /// `naive` for `C = A·Bᵀ` (B is `[n, k]`).
    fn naive_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[j * k + p];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn sgemm_nt_matches_naive() {
        for (m, k, n) in [
            (1, 1, 1),
            (7, 17, 13),
            (64, 64, 64),
            (100, 130, 96),
            (128, 256, 512),
        ] {
            let a = fill(11, m * k);
            let b = fill(12, n * k);
            let want = naive_nt(&a, &b, m, k, n);
            let mut got = vec![0.0f32; m * n];
            let mut got_par = vec![0.0f32; m * n];
            unsafe {
                wukong_sgemm_nt(
                    a.as_ptr(),
                    b.as_ptr(),
                    got.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                );
                wukong_sgemm_nt_parallel(
                    a.as_ptr(),
                    b.as_ptr(),
                    got_par.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                );
            }
            let tol = 1e-3 * (k as f32).sqrt();
            for i in 0..m * n {
                assert!(
                    (got[i] - want[i]).abs() <= tol + 1e-4 * want[i].abs(),
                    "nt ({m}x{k}x{n}) idx {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
            assert_eq!(got, got_par, "nt serial vs parallel ({m}x{k}x{n})");
        }
    }

    /// Naive `C = Aᵀ·B`: A stored row-major `[k, m]`, B `[k, n]`, C `[m, n]`.
    fn naive_tn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[p * m + i] * b[p * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    #[test]
    fn sgemm_tn_matches_naive_and_transposed_nn() {
        // The last tuple exceeds PAR_MIN_MACS (~74M MACs) so the multicore tn path actually runs;
        // the rest straddle the MR=6 / NR=16 remainders.
        for (m, k, n) in [
            (1, 1, 1),
            (5, 7, 3),
            (64, 64, 64),
            (100, 130, 96),
            (128, 256, 64),
            (520, 264, 540),
        ] {
            let a = fill(21, k * m); // A stored [k, m]
            let b = fill(22, k * n); // B stored [k, n]
            let want = naive_tn(&a, &b, m, k, n);
            let mut got = vec![0.0f32; m * n];
            let mut got_par = vec![0.0f32; m * n];
            unsafe {
                wukong_sgemm_tn(
                    a.as_ptr(),
                    b.as_ptr(),
                    got.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                );
                wukong_sgemm_tn_parallel(
                    a.as_ptr(),
                    b.as_ptr(),
                    got_par.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                );
            }
            let tol = 1e-3 * (k as f32).sqrt();
            for i in 0..m * n {
                assert!(
                    (got[i] - want[i]).abs() <= tol + 1e-4 * want[i].abs(),
                    "tn ({m}x{k}x{n}) idx {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
            // Bit-exact against the NN kernel on a manually-transposed A — proves the only difference
            // from the differential-contract kernel is the (deterministic) transpose prepass.
            let mut at = vec![0.0f32; m * k];
            for p in 0..k {
                for i in 0..m {
                    at[i * k + p] = a[p * m + i];
                }
            }
            let mut nn = vec![0.0f32; m * n];
            unsafe {
                wukong_sgemm(
                    at.as_ptr(),
                    b.as_ptr(),
                    nn.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                );
            }
            assert_eq!(got, nn, "tn must equal NN on manually-transposed A ({m}x{k}x{n})");
            assert_eq!(got, got_par, "tn serial vs parallel ({m}x{k}x{n})");
        }
    }

    /// `C = A·Bᵀ` with bf16/f16 inputs must equal the f32 NT kernel on the **losslessly-widened**
    /// operands (the widen prepass is the only difference from the differential-contract kernel), and
    /// serial must equal parallel — bit-for-bit, deterministically.
    #[test]
    fn sgemm_lowp_nt_matches_widened_f32_and_parallel() {
        for (m, k, n) in [
            (1, 1, 1),
            (5, 7, 3),
            (64, 64, 64),
            (100, 130, 96),
            (128, 256, 64),
            (520, 264, 540), // > PAR_MIN_MACS so the multicore path runs
        ] {
            let a = fill(31, m * k);
            let b = fill(32, n * k);
            // bf16: round each operand to bf16, keep both the stored bits and the widened-back f32.
            let a_bf: Vec<u16> = a.iter().map(|&x| crate::f32_to_bf16_bits(x)).collect();
            let b_bf: Vec<u16> = b.iter().map(|&x| crate::f32_to_bf16_bits(x)).collect();
            let a_bf_f32: Vec<f32> = a.iter().map(|&x| crate::round_bf16(x)).collect();
            let b_bf_f32: Vec<f32> = b.iter().map(|&x| crate::round_bf16(x)).collect();
            // f16 twin.
            let a_h: Vec<u16> = a.iter().map(|&x| crate::f32_to_f16_bits(x)).collect();
            let b_h: Vec<u16> = b.iter().map(|&x| crate::f32_to_f16_bits(x)).collect();
            let a_h_f32: Vec<f32> = a.iter().map(|&x| crate::round_f16(x)).collect();
            let b_h_f32: Vec<f32> = b.iter().map(|&x| crate::round_f16(x)).collect();

            let mut ref_bf = vec![0.0f32; m * n];
            let mut ref_h = vec![0.0f32; m * n];
            let mut got_bf = vec![0.0f32; m * n];
            let mut got_bf_par = vec![0.0f32; m * n];
            let mut got_h = vec![0.0f32; m * n];
            let mut got_h_par = vec![0.0f32; m * n];
            let (mi, ki, ni) = (m as i64, k as i64, n as i64);
            unsafe {
                // Reference: the exact f32 NT kernel on the widened operands.
                wukong_sgemm_nt(a_bf_f32.as_ptr(), b_bf_f32.as_ptr(), ref_bf.as_mut_ptr(), mi, ki, ni, 0);
                wukong_sgemm_nt(a_h_f32.as_ptr(), b_h_f32.as_ptr(), ref_h.as_mut_ptr(), mi, ki, ni, 0);
                // bf16 / f16 kernels on the stored half-width bits.
                wukong_sgemm_bf16_nt(a_bf.as_ptr(), b_bf.as_ptr(), got_bf.as_mut_ptr(), mi, ki, ni, 0);
                wukong_sgemm_bf16_nt_parallel(a_bf.as_ptr(), b_bf.as_ptr(), got_bf_par.as_mut_ptr(), mi, ki, ni, 0);
                wukong_sgemm_f16_nt(a_h.as_ptr(), b_h.as_ptr(), got_h.as_mut_ptr(), mi, ki, ni, 0);
                wukong_sgemm_f16_nt_parallel(a_h.as_ptr(), b_h.as_ptr(), got_h_par.as_mut_ptr(), mi, ki, ni, 0);
            }
            assert_eq!(got_bf, ref_bf, "bf16 nt must equal f32 nt on widened operands ({m}x{k}x{n})");
            assert_eq!(got_bf, got_bf_par, "bf16 nt serial vs parallel ({m}x{k}x{n})");
            assert_eq!(got_h, ref_h, "f16 nt must equal f32 nt on widened operands ({m}x{k}x{n})");
            assert_eq!(got_h, got_h_par, "f16 nt serial vs parallel ({m}x{k}x{n})");
        }
    }

    /// `C = Aᵀ·B` with bf16/f16 inputs (the mixed-precision weight-gradient GEMM) must equal the f32 TN
    /// kernel on the **losslessly-widened** operands (the widen prepass is the only difference from the
    /// differential-contract kernel), and serial must equal parallel — bit-for-bit, deterministically.
    #[test]
    fn sgemm_lowp_tn_matches_widened_f32_and_parallel() {
        for (m, k, n) in [
            (1, 1, 1),
            (5, 7, 3),
            (64, 64, 64),
            (100, 130, 96),
            (128, 256, 64),
            (520, 264, 540), // > PAR_MIN_MACS so the multicore path runs
        ] {
            let a = fill(51, k * m); // A stored [k, m]
            let b = fill(52, k * n); // B stored [k, n]
            let a_bf: Vec<u16> = a.iter().map(|&x| crate::f32_to_bf16_bits(x)).collect();
            let b_bf: Vec<u16> = b.iter().map(|&x| crate::f32_to_bf16_bits(x)).collect();
            let a_bf_f32: Vec<f32> = a.iter().map(|&x| crate::round_bf16(x)).collect();
            let b_bf_f32: Vec<f32> = b.iter().map(|&x| crate::round_bf16(x)).collect();
            let a_h: Vec<u16> = a.iter().map(|&x| crate::f32_to_f16_bits(x)).collect();
            let b_h: Vec<u16> = b.iter().map(|&x| crate::f32_to_f16_bits(x)).collect();
            let a_h_f32: Vec<f32> = a.iter().map(|&x| crate::round_f16(x)).collect();
            let b_h_f32: Vec<f32> = b.iter().map(|&x| crate::round_f16(x)).collect();

            let mut ref_bf = vec![0.0f32; m * n];
            let mut ref_h = vec![0.0f32; m * n];
            let mut got_bf = vec![0.0f32; m * n];
            let mut got_bf_par = vec![0.0f32; m * n];
            let mut got_h = vec![0.0f32; m * n];
            let mut got_h_par = vec![0.0f32; m * n];
            let (mi, ki, ni) = (m as i64, k as i64, n as i64);
            unsafe {
                // Reference: the exact f32 TN kernel on the widened operands.
                wukong_sgemm_tn(a_bf_f32.as_ptr(), b_bf_f32.as_ptr(), ref_bf.as_mut_ptr(), mi, ki, ni, 0);
                wukong_sgemm_tn(a_h_f32.as_ptr(), b_h_f32.as_ptr(), ref_h.as_mut_ptr(), mi, ki, ni, 0);
                // bf16 / f16 TN kernels on the stored half-width bits.
                wukong_sgemm_bf16_tn(a_bf.as_ptr(), b_bf.as_ptr(), got_bf.as_mut_ptr(), mi, ki, ni, 0);
                wukong_sgemm_bf16_tn_parallel(a_bf.as_ptr(), b_bf.as_ptr(), got_bf_par.as_mut_ptr(), mi, ki, ni, 0);
                wukong_sgemm_f16_tn(a_h.as_ptr(), b_h.as_ptr(), got_h.as_mut_ptr(), mi, ki, ni, 0);
                wukong_sgemm_f16_tn_parallel(a_h.as_ptr(), b_h.as_ptr(), got_h_par.as_mut_ptr(), mi, ki, ni, 0);
            }
            assert_eq!(got_bf, ref_bf, "bf16 tn must equal f32 tn on widened operands ({m}x{k}x{n})");
            assert_eq!(got_bf, got_bf_par, "bf16 tn serial vs parallel ({m}x{k}x{n})");
            assert_eq!(got_h, ref_h, "f16 tn must equal f32 tn on widened operands ({m}x{k}x{n})");
            assert_eq!(got_h, got_h_par, "f16 tn serial vs parallel ({m}x{k}x{n})");
        }
    }

    /// The fused-epilogue bf16/f16 GEMM (`C = act(A·Bᵀ + bias)`) must equal the f32 `nt_epi` kernel on
    /// the losslessly-widened operands, for every (activation, bias-present) combination — the widen is
    /// the only difference — and serial must equal parallel.
    #[test]
    fn sgemm_lowp_nt_epi_matches_widened_f32() {
        let (m, k, n) = (40usize, 72, 48);
        let a = fill(41, m * k);
        let b = fill(42, n * k);
        let bias = fill(43, n);
        let a_bf: Vec<u16> = a.iter().map(|&x| crate::f32_to_bf16_bits(x)).collect();
        let b_bf: Vec<u16> = b.iter().map(|&x| crate::f32_to_bf16_bits(x)).collect();
        let a_bf_f32: Vec<f32> = a.iter().map(|&x| crate::round_bf16(x)).collect();
        let b_bf_f32: Vec<f32> = b.iter().map(|&x| crate::round_bf16(x)).collect();
        let a_h: Vec<u16> = a.iter().map(|&x| crate::f32_to_f16_bits(x)).collect();
        let b_h: Vec<u16> = b.iter().map(|&x| crate::f32_to_f16_bits(x)).collect();
        let a_h_f32: Vec<f32> = a.iter().map(|&x| crate::round_f16(x)).collect();
        let b_h_f32: Vec<f32> = b.iter().map(|&x| crate::round_f16(x)).collect();
        let (mi, ki, ni) = (m as i64, k as i64, n as i64);
        for act in 0i64..4 {
            for use_bias in [false, true] {
                let bptr = if use_bias { bias.as_ptr() } else { std::ptr::null() };
                let mut ref_bf = vec![0.0f32; m * n];
                let mut got_bf = vec![0.0f32; m * n];
                let mut got_bf_par = vec![0.0f32; m * n];
                let mut ref_h = vec![0.0f32; m * n];
                let mut got_h = vec![0.0f32; m * n];
                let mut got_h_par = vec![0.0f32; m * n];
                unsafe {
                    // Reference: the f32 fused-epilogue kernel on the widened operands.
                    wukong_sgemm_nt_epi(a_bf_f32.as_ptr(), b_bf_f32.as_ptr(), ref_bf.as_mut_ptr(), mi, ki, ni, 0, bptr, act);
                    wukong_sgemm_bf16_nt_epi(a_bf.as_ptr(), b_bf.as_ptr(), got_bf.as_mut_ptr(), mi, ki, ni, 0, bptr, act);
                    wukong_sgemm_bf16_nt_epi_parallel(a_bf.as_ptr(), b_bf.as_ptr(), got_bf_par.as_mut_ptr(), mi, ki, ni, 0, bptr, act);
                    wukong_sgemm_nt_epi(a_h_f32.as_ptr(), b_h_f32.as_ptr(), ref_h.as_mut_ptr(), mi, ki, ni, 0, bptr, act);
                    wukong_sgemm_f16_nt_epi(a_h.as_ptr(), b_h.as_ptr(), got_h.as_mut_ptr(), mi, ki, ni, 0, bptr, act);
                    wukong_sgemm_f16_nt_epi_parallel(a_h.as_ptr(), b_h.as_ptr(), got_h_par.as_mut_ptr(), mi, ki, ni, 0, bptr, act);
                }
                assert_eq!(got_bf, ref_bf, "bf16 nt_epi act={act} bias={use_bias}");
                assert_eq!(got_bf, got_bf_par, "bf16 nt_epi serial vs parallel act={act} bias={use_bias}");
                assert_eq!(got_h, ref_h, "f16 nt_epi act={act} bias={use_bias}");
                assert_eq!(got_h, got_h_par, "f16 nt_epi serial vs parallel act={act} bias={use_bias}");
            }
        }
    }

    #[test]
    fn sgemm_parallel_matches_serial() {
        // The size must exceed PAR_MIN_MACS, or wukong_sgemm_parallel falls back to the serial
        // kernel (the work threshold) and this would degenerate into a serial-vs-serial no-op. These
        // dims also straddle the MR=6 / NR=16 remainders so the cross-core packing + edge tiles run.
        let (m, k, n) = (520usize, 264, 540); // ~74M MACs > 2^26
        assert!(
            (m * k * n) as u64 >= PAR_MIN_MACS,
            "size must exercise the multi-core path"
        );
        let a = fill(6, m * k);
        let b = fill(7, k * n);
        let mut serial = vec![0.0f32; m * n];
        let mut par = vec![0.0f32; m * n];
        unsafe {
            wukong_sgemm(
                a.as_ptr(),
                b.as_ptr(),
                serial.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
                0,
            );
            wukong_sgemm_parallel(
                a.as_ptr(),
                b.as_ptr(),
                par.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
                0,
            );
        }
        // Identical per-(i,j) accumulation order ⇒ bit-for-bit identical results.
        assert_eq!(serial, par);

        // Same for the nn.Linear (C = A·Bᵀ) parallel path, which otherwise has no correctness test.
        let mut serial_nt = vec![0.0f32; m * n];
        let mut par_nt = vec![0.0f32; m * n];
        unsafe {
            wukong_sgemm_nt(
                a.as_ptr(),
                b.as_ptr(),
                serial_nt.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
                0,
            );
            wukong_sgemm_nt_parallel(
                a.as_ptr(),
                b.as_ptr(),
                par_nt.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
                0,
            );
        }
        assert_eq!(serial_nt, par_nt);
    }

    /// The persistent-broadcast-region parallel GEMM (one `ThreadPool::broadcast` per call with
    /// in-region phase barriers — see [`sgemm_persistent_region`]) must be **bit-for-bit** identical
    /// to the serial kernel, and so must the legacy fork-join shape it replaced (the
    /// `WUKONG_GEMM_FORKJOIN=1` instrument path). Shapes exercise what the smaller
    /// `sgemm_parallel_matches_serial` shape cannot: `n > NC = 4080` (two NC blocks → two full jc
    /// iterations of the region schedule, the second with a non-NR-multiple `nc`), `k >
    /// select_kc(k)` (multiple K-blocks → the mid-schedule barriers and the `beta_eff` accumulate
    /// rule), a `k` whose equal-split leaves a thin remainder block (901 → 304+304+293), and an
    /// MR-remainder `m`. Every shape trips `PAR_MIN_MACS` (asserted) so the public entry really
    /// takes the parallel path. Both internal entries are called directly, making the test immune
    /// to the process-wide `WUKONG_GEMM_FORKJOIN` OnceLock state; the public entry is checked too
    /// (it routes to one of the two, and both must equal serial). The epilogue leg pins the
    /// final-K-block-only bias+activation rule across multiple K *and* NC blocks — a per-block
    /// epilogue bug would apply the bias three times over.
    #[test]
    fn sgemm_persistent_region_matches_serial() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            println!("no avx2/fma — skipping");
            return;
        }
        for &(m, k, n) in &[(520usize, 900usize, 4200usize), (214, 901, 4160)] {
            assert!((m * k * n) as u64 >= PAR_MIN_MACS, "shape must trip the parallel gate");
            assert!(n > NC, "shape must span multiple NC blocks");
            assert!(k > select_kc(k), "shape must span multiple K blocks");
            let a = fill(71, m * k);
            let b = fill(72, k * n);
            let (mi, ki, ni) = (m as i64, k as i64, n as i64);
            let mut serial = vec![0.0f32; m * n];
            unsafe {
                wukong_sgemm(a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(), mi, ki, ni, 0);
            }
            // Persistent region, called directly with the same pool selection gemm_dispatch uses.
            let mut per = vec![0.0f32; m * n];
            unsafe {
                let args = GemmArgs {
                    a: a.as_ptr(), b: b.as_ptr(), c: per.as_mut_ptr(),
                    m, k, n, beta: 0.0, bt: false, epi: None,
                };
                sgemm_persistent_region(gemm_pool(), args);
            }
            assert_eq!(serial, per, "persistent region != serial (m{m} k{k} n{n})");
            // Legacy fork-join path (what the WUKONG_GEMM_FORKJOIN=1 kill-switch routes to).
            let mut fj = vec![0.0f32; m * n];
            unsafe {
                sgemm_avx2_parallel(a.as_ptr(), b.as_ptr(), fj.as_mut_ptr(), m, k, n, 0.0, false, None);
            }
            assert_eq!(serial, fj, "fork-join != serial (m{m} k{k} n{n})");
            // Public entry: routes to whichever path the process env selected — must equal serial
            // either way.
            let mut public = vec![0.0f32; m * n];
            unsafe {
                wukong_sgemm_parallel(a.as_ptr(), b.as_ptr(), public.as_mut_ptr(), mi, ki, ni, 0);
            }
            assert_eq!(serial, public, "public parallel entry != serial (m{m} k{k} n{n})");
        }
        // Fused-epilogue leg: bias + ReLU on the nt form, across 3 K-blocks (uneven split) and 2 NC
        // blocks, on both parallel shapes.
        {
            let (m, k, n) = (520usize, 901usize, 4200usize);
            assert!((m * k * n) as u64 >= PAR_MIN_MACS && n > NC && k > select_kc(k));
            let a = fill(73, m * k);
            let b = fill(74, n * k);
            let bias = fill(75, n);
            let epi = Epilogue { bias: bias.as_ptr(), act: ACT_RELU, alpha: 1.0 };
            let mut serial = vec![0.0f32; m * n];
            let mut per = vec![0.0f32; m * n];
            let mut fj = vec![0.0f32; m * n];
            unsafe {
                wukong_sgemm_nt_epi(
                    a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(),
                    m as i64, k as i64, n as i64, 0, bias.as_ptr(), ACT_RELU as i64,
                );
                let args = GemmArgs {
                    a: a.as_ptr(), b: b.as_ptr(), c: per.as_mut_ptr(),
                    m, k, n, beta: 0.0, bt: true, epi: Some(epi),
                };
                sgemm_persistent_region(gemm_pool(), args);
                sgemm_avx2_parallel(a.as_ptr(), b.as_ptr(), fj.as_mut_ptr(), m, k, n, 0.0, true, Some(epi));
            }
            assert_eq!(serial, per, "persistent nt+epi != serial");
            assert_eq!(serial, fj, "fork-join nt+epi != serial");
        }
    }

    /// The 2D block-parallel path (`WUKONG_GEMM_2D=1` routes here — see [`sgemm_2d_blocks`],
    /// called DIRECTLY so the test is immune to the process env) must be **bit-for-bit** identical
    /// to the serial kernel: each C block has one owner running the whole ascending-pc K loop with
    /// the serial kernel's `select_kc` grouping, packers, and epilogue discipline. Shapes exercise
    /// multiple blocks in both grid dims for every policy candidate (520×4200: even the largest
    /// (192, 256) regime gives a 3×17 grid), MR/NR remainders at the matrix edges (520 % 6 = 4,
    /// 4200 % 16 = 8; 214 % 6 = 4, 616 % 16 = 8), a `k` spanning multiple UNEVEN K-blocks
    /// (901 → 304+304+293), `bt` false and true, `beta` 0 and 1 (accumulate onto a nonzero C), a
    /// single-block degenerate grid, and a skinny-m one-row-block grid. The epilogue leg pins the
    /// final-K-block-only bias+activation rule and the block-origin bias shift across all four
    /// activations, with k > KC.
    #[test]
    fn sgemm_2d_blocks_matches_serial() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            println!("no avx2/fma — skipping");
            return;
        }
        // (m, k, n, bt combos, beta combos): the big multi-block shape runs once (bt=false,
        // beta=0); the mid shape sweeps the full bt × beta matrix; two degenerate grids.
        let shapes: &[(usize, usize, usize, &[bool], &[i64])] = &[
            (520, 901, 4200, &[false], &[0]),          // multi-block both dims, 3 uneven K-blocks
            (214, 901, 616, &[false, true], &[0, 1]),  // edge remainders, full bt × beta sweep
            (100, 130, 96, &[false, true], &[0, 1]),   // single/degenerate grid
            (13, 700, 530, &[false, true], &[0, 1]),   // skinny m: 1 row-block, several col-blocks
        ];
        for &(m, k, n, bts, betas) in shapes {
            for &bt in bts {
                for &beta in betas {
                    let a = fill(81, m * k);
                    let b = fill(82, k * n); // same length either way: bt reads it as [n, k]
                    let base = fill(83, m * n); // beta=1 must accumulate onto a nonzero C
                    let mut serial = base.clone();
                    let mut got2d = base.clone();
                    let (mi, ki, ni) = (m as i64, k as i64, n as i64);
                    unsafe {
                        if bt {
                            wukong_sgemm_nt(a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(), mi, ki, ni, beta);
                        } else {
                            wukong_sgemm(a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(), mi, ki, ni, beta);
                        }
                        let args = GemmArgs {
                            a: a.as_ptr(), b: b.as_ptr(), c: got2d.as_mut_ptr(),
                            m, k, n, beta: beta as f32, bt, epi: None,
                        };
                        sgemm_2d_blocks(gemm_pool(), args);
                    }
                    assert_eq!(serial, got2d, "2d != serial (m{m} k{k} n{n} bt{bt} beta{beta})");
                }
            }
        }
        // Fused-epilogue leg: bias + every activation on the nt form, k = 901 > KC (3 uneven
        // K-blocks) — a per-K-block epilogue bug would fold the bias/activation three times, and a
        // missing block-origin shift would read the wrong bias entries in every column block.
        {
            let (m, k, n) = (214usize, 901usize, 616usize);
            assert!(k > select_kc(k), "epi leg must span multiple K-blocks");
            let a = fill(84, m * k);
            let b = fill(85, n * k);
            let bias = fill(86, n);
            for &act in &[ACT_IDENTITY, ACT_RELU, ACT_GELU, ACT_SILU] {
                for use_bias in [false, true] {
                    let bias_ptr = if use_bias { bias.as_ptr() } else { std::ptr::null() };
                    let mut serial = vec![0.0f32; m * n];
                    let mut got2d = vec![0.0f32; m * n];
                    unsafe {
                        wukong_sgemm_nt_epi(
                            a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(),
                            m as i64, k as i64, n as i64, 0, bias_ptr, act as i64,
                        );
                        let args = GemmArgs {
                            a: a.as_ptr(), b: b.as_ptr(), c: got2d.as_mut_ptr(),
                            m, k, n, beta: 0.0, bt: true,
                            epi: Some(Epilogue { bias: bias_ptr, act, alpha: 1.0 }),
                        };
                        sgemm_2d_blocks(gemm_pool(), args);
                    }
                    assert_eq!(serial, got2d, "2d nt+epi != serial (act{act} bias{use_bias})");
                }
            }
        }
    }

    /// [`select_2d_block_shape`] structural invariants (a pure policy fn; bit-exactness never
    /// depends on its answer, so this pins only what the kernels' pack-slice arithmetic relies
    /// on): MR/NR-aligned block dims — the contiguous-shared-slice offsets in
    /// [`sgemm_2d_shared`] and the byte-identical per-block packs both require it — and the
    /// skinny-M rule's single row of blocks. The documented example grids are pinned exactly, so
    /// a policy retune shows up as a deliberate diff here.
    #[test]
    fn select_2d_block_shape_policy() {
        for &m in &[13usize, 100, 128, 144, 214, 300, 512, 520, 1024, 2048] {
            for &n in &[96usize, 128, 530, 616, 768, 1024, 3072, 4200] {
                for &target in &[1usize, 3, 12, 48, 96] {
                    let (bm, bn) = select_2d_block_shape(m, n, target);
                    assert!(bm >= MR && bm % MR == 0, "BM {bm} not MR-aligned (m{m} n{n} t{target})");
                    assert!(bn >= NR && bn % NR == 0, "BN {bn} not NR-aligned (m{m} n{n} t{target})");
                    if m <= MC {
                        assert_eq!(m.div_ceil(bm), 1, "skinny m must give one block row (m{m} n{n} t{target})");
                    }
                    // Tail balance: after the rebalance, every non-edge block is full-size and the
                    // edge remainder is at least `block − unit` short of one extra block — i.e. the
                    // grid can't shrink a whole block dim further without changing the block count.
                    let (nbi, nbj) = (m.div_ceil(bm), n.div_ceil(bn));
                    assert!(nbi * bm >= m && nbj * bn >= n, "grid must cover the matrix (m{m} n{n} t{target})");
                    assert!(
                        (bm - MR) * nbi < m,
                        "BM {bm} not minimal for its row count (m{m} n{n} t{target})"
                    );
                    assert!(
                        (bn - NR) * nbj < n,
                        "BN {bn} not minimal for its column count (m{m} n{n} t{target})"
                    );
                }
            }
        }
        // --- Cube grids at the 16-worker default target (48) are UNCHANGED by the (96, 96)
        // insertion (it yields < 48 blocks at every square, so each cube still selects its prior
        // candidate): 256³ → (24, 48) → 11×6; 512³ → (48, 96) → 11×6; 1024³ rebalances
        // (144, 192) → (132, 176) → still 8×6, but 7×132 + 100 instead of 7×144 + a 16-row
        // straggler, and 5×176 + 144 instead of 5×192 + a 64-wide straggler column; 2048³ →
        // (192, 256) → 11×8. Pinned so a cube regression shows up here, not only in the noisy bench.
        assert_eq!(select_2d_block_shape(256, 256, 48), (24, 48));
        assert_eq!(select_2d_block_shape(512, 512, 48), (48, 96));
        assert_eq!(select_2d_block_shape(1024, 1024, 48), (132, 176));
        assert_eq!(select_2d_block_shape(2048, 2048, 48), (192, 256));
        // --- The six skinny transformer bench shapes (m, n), the shapes `matmul_skinny` reports.
        // The two 128-row shapes stay on the skinny single-row rule; the two 512×768 shapes are
        // the ones the (96, 96) insertion reshapes from an 11×8 (48, 96) grid (B repacked 11×) to
        // a 6×8 (90, 96) grid (B repacked 6×) — the measured 512×768 win. 512×3072 is untouched
        // (it selects (144, 192) before the ladder ever reaches (96, 96)).
        assert_eq!(select_2d_block_shape(128, 768, 48), (132, 32)); // 128×768·(768/3072×768)ᵀ
        assert_eq!(select_2d_block_shape(128, 3072, 48), (132, 128)); // 128×768·(3072×768)ᵀ
        assert_eq!(select_2d_block_shape(512, 768, 48), (90, 96)); // 512×768·(768/3072×768)ᵀ
        assert_eq!(select_2d_block_shape(512, 3072, 48), (132, 192)); // 512×768·(3072×768)ᵀ
    }

    /// [`blocks_target_from`] — the `WUKONG_GEMM_TASK_MACS` grain policy, tested through its pure
    /// form (the env is resolved once by the caller): unset keeps the caller's default exactly
    /// (the byte-for-byte "current behavior" contract of the knob), set divides the problem's
    /// MACs by the budget with a floor of 1 block and the 2^20 sanity cap.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn blocks_target_grain_policy() {
        // Unset → the caller's default (3 × nworkers at both call sites), regardless of size.
        assert_eq!(blocks_target_from(1 << 30, None, 48), 48);
        assert_eq!(blocks_target_from(1, None, 7), 7);
        // Set → macs / budget: a 2^30-MAC problem at a 2^24 budget is 64 tasks.
        assert_eq!(blocks_target_from(1 << 30, Some(1 << 24), 48), 64);
        // Budget above the whole problem floors at one block (never zero).
        assert_eq!(blocks_target_from(1 << 20, Some(1 << 30), 48), 1);
        // Pathological budget hits the cap instead of demanding a u64-sized grid.
        assert_eq!(blocks_target_from(u64::MAX, Some(1), 48), 1 << 20);
    }

    /// [`dyn_block_order`] structural invariants: the steal-order enumeration must be a
    /// **permutation** of `0..nbi·nbj` (each block exactly once — one owner per C block is the
    /// bit-exactness precondition) in which every full `bm×bn` block strictly precedes every M/N
    /// edge-remainder block (the LPT property the probe exists for), and the full blocks keep
    /// their row-major relative order (determinism: the enumeration is a pure function of the
    /// grid, never of timing).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dyn_block_order_is_a_permutation() {
        for &(m, n, target) in &[
            (520usize, 4200usize, 48usize), // multi-block both dims, both edges ragged
            (214, 616, 48),                 // edge remainders both dims
            (13, 530, 48),                  // skinny: one block row
            (300, 300, 91),                 // odd square, fine grid
            (192, 256, 1),                  // exactly one (full) block — no tails at all
        ] {
            let (bm, bn) = select_2d_block_shape(m, n, target);
            let (nbi, nbj) = (m.div_ceil(bm), n.div_ceil(bn));
            let order = dyn_block_order(m, n, bm, bn, nbi, nbj);
            assert_eq!(order.len(), nbi * nbj, "not a full enumeration (m{m} n{n} t{target})");
            let mut seen = vec![false; nbi * nbj];
            for &t in &order {
                assert!(!seen[t as usize], "block {t} claimed twice (m{m} n{n} t{target})");
                seen[t as usize] = true;
            }
            let full = |t: usize| {
                let (bi, bj) = (t / nbj, t % nbj);
                (bi + 1) * bm <= m && (bj + 1) * bn <= n
            };
            // Full blocks first, tails last — once the first tail appears, no full block follows.
            if let Some(ft) = order.iter().position(|&t| !full(t as usize)) {
                assert!(
                    order[ft..].iter().all(|&t| !full(t as usize)),
                    "full block after a tail block (m{m} n{n} t{target})"
                );
                // Within each class the row-major relative order is preserved (sorted indices).
                assert!(order[..ft].windows(2).all(|w| w[0] < w[1]));
                assert!(order[ft..].windows(2).all(|w| w[0] < w[1]));
            }
        }
    }

    /// The dynamic-queue 2D path ([`sgemm_2d_blocks_dyn`] — the default parallel scheduler;
    /// called DIRECTLY so the test is immune to the process-wide env OnceLock state) must be
    /// **bit-for-bit** identical to the serial kernel under BOTH block enumerations
    /// (`steal_order` false/true): the claim counter and the permutation only choose which worker
    /// runs a block and when, never the per-(i,j) K order. Shapes: two below the shared-pack band
    /// (`SHARED_MAX_MACS` = 2^26 MACs) and two at/above it — the band keying happens in
    /// `gemm_dispatch`, not here, but the sweep proves the mechanism on both regimes' geometries
    /// — all with non-divisible M/N tails: odd 300³, the skinny ragged (128, 700, 530), the
    /// edge-remainder + uneven-K (214, 901, 616) with the full bt × beta matrix (beta=1 must
    /// accumulate onto a nonzero C), and the skinny FFN shape 128×3072×768 (one block row, 8
    /// K-blocks) on both bt legs. The epilogue leg pins the final-K-block-only bias/activation
    /// rule, the block-origin bias shift, the null-bias shift, and the α scale through the
    /// dynamic claim loop.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sgemm_2d_dyn_matches_serial() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            println!("no avx2/fma — skipping");
            return;
        }
        // (m, k, n, bt combos, beta combos) — band notes relative to SHARED_MAX_MACS = 2^26.
        let shapes: &[(usize, usize, usize, &[bool], &[i64])] = &[
            (300, 300, 300, &[false], &[0]),          // below band; odd square, ragged both dims
            (128, 700, 530, &[false], &[0]),          // below band; skinny M, ragged n
            (214, 901, 616, &[false, true], &[0, 1]), // above band; edge remainders, 3 uneven K-blocks
            (128, 3072, 768, &[false, true], &[0]),   // above band; skinny FFN, long K
        ];
        for &(m, k, n, bts, betas) in shapes {
            for &bt in bts {
                for &beta in betas {
                    let a = fill(101, m * k);
                    let b = fill(102, k * n); // same length either way: bt reads it as [n, k]
                    let base = fill(103, m * n); // beta=1 must accumulate onto a nonzero C
                    let mut serial = base.clone();
                    let (mi, ki, ni) = (m as i64, k as i64, n as i64);
                    unsafe {
                        if bt {
                            wukong_sgemm_nt(a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(), mi, ki, ni, beta);
                        } else {
                            wukong_sgemm(a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(), mi, ki, ni, beta);
                        }
                    }
                    for steal_order in [false, true] {
                        for skinny_a in [false, true] {
                            let mut got = base.clone();
                            unsafe {
                                let args = GemmArgs {
                                    a: a.as_ptr(), b: b.as_ptr(), c: got.as_mut_ptr(),
                                    m, k, n, beta: beta as f32, bt, epi: None,
                                };
                                sgemm_2d_blocks_dyn(gemm_pool(), args, steal_order, skinny_a);
                            }
                            assert_eq!(
                                serial, got,
                                "dyn != serial (m{m} k{k} n{n} bt{bt} beta{beta} order{steal_order} skinnyA{skinny_a})"
                            );
                        }
                    }
                }
            }
        }
        // Epilogue leg: final-K-only rule + bias shifts + α through the dynamic claim loop, with
        // 3 uneven K-blocks (901 → 304+304+293) so a per-K-block epilogue bug folds three times.
        // Runs at TWO shapes: the edge-remainder grid (nbi > 1, per-block body) and a skinny
        // single-block-row twin (m = 128 ⇒ nbi == 1) whose `skinny_a` leg pins the epilogue rules
        // through the shared-A body ([`sgemm_2d_block_shared_a`]) — a per-K-chunk epilogue bug
        // there would fold the bias three times over.
        for (m, k, n) in [(214usize, 901usize, 616usize), (128, 901, 616)] {
            assert!(k > select_kc(k), "epi leg must span multiple K-blocks");
            let a = fill(104, m * k);
            let b = fill(105, n * k);
            let bias = fill(106, n);
            let run_dyn = |epi: Epilogue, serial_ref: &[f32], tag: &str| {
                for steal_order in [false, true] {
                    for skinny_a in [false, true] {
                        let mut got = vec![0.0f32; m * n];
                        unsafe {
                            let args = GemmArgs {
                                a: a.as_ptr(), b: b.as_ptr(), c: got.as_mut_ptr(),
                                m, k, n, beta: 0.0, bt: true, epi: Some(epi),
                            };
                            sgemm_2d_blocks_dyn(gemm_pool(), args, steal_order, skinny_a);
                        }
                        assert_eq!(
                            serial_ref,
                            &got[..],
                            "dyn nt+epi != serial ({tag} m{m} order{steal_order} skinnyA{skinny_a})"
                        );
                    }
                }
            };
            let mut serial = vec![0.0f32; m * n];
            unsafe {
                wukong_sgemm_nt_epi(
                    a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(),
                    m as i64, k as i64, n as i64, 0, bias.as_ptr(), ACT_RELU as i64,
                );
            }
            run_dyn(Epilogue { bias: bias.as_ptr(), act: ACT_RELU, alpha: 1.0 }, &serial, "relu bias");
            // No bias: the null base must stay null through both shifts.
            let mut serial_nb = vec![0.0f32; m * n];
            unsafe {
                wukong_sgemm_nt_epi(
                    a.as_ptr(), b.as_ptr(), serial_nb.as_mut_ptr(),
                    m as i64, k as i64, n as i64, 0, std::ptr::null(), ACT_SILU as i64,
                );
            }
            run_dyn(
                Epilogue { bias: std::ptr::null(), act: ACT_SILU, alpha: 1.0 },
                &serial_nb,
                "silu nobias",
            );
            // α-scaled (the attention QKᵀ/√d form): folded on the final K-block only.
            let mut serial_alpha = vec![0.0f32; m * n];
            unsafe {
                wukong_sgemm_nt_alpha(
                    a.as_ptr(), b.as_ptr(), serial_alpha.as_mut_ptr(),
                    m as i64, k as i64, n as i64, 0, 0.125,
                );
            }
            run_dyn(
                Epilogue { bias: std::ptr::null(), act: ACT_IDENTITY, alpha: 0.125 },
                &serial_alpha,
                "alpha 0.125",
            );
        }
    }

    /// The dynamic queue must stay bit-identical to serial at EVERY pool width — redistributing
    /// blocks across variable worker sets is its whole point — so this sweeps test-local rayon
    /// pools of 1..=16 threads (1 = the degenerate one-worker-drains-everything case; 16 = the
    /// dev box's physical-core pool width) with both enumerations, plus the static rayon path
    /// ([`sgemm_2d_blocks`], the shipped dynamic-OFF default) as a cross-check at the same
    /// widths. Note the grid policy's `3 × nworkers` target makes the block shape itself vary
    /// with the width, so this also sweeps geometries. Shapes alternate by width parity — odd
    /// widths get the above-band edge-remainder shape (150×901×530: ragged both edges, 3 uneven
    /// K-blocks), even widths the below-band skinny shape (128×700×530) — covering thread-count
    /// × geometry without doubling the debug-build cost of the suite.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sgemm_2d_dyn_thread_sweep() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            println!("no avx2/fma — skipping");
            return;
        }
        let shapes = [(150usize, 901usize, 530usize), (128, 700, 530)];
        // One serial reference per shape, reused across every width.
        let refs: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = shapes
            .iter()
            .map(|&(m, k, n)| {
                let a = fill(111, m * k);
                let b = fill(112, k * n);
                let mut s = vec![0.0f32; m * n];
                unsafe {
                    wukong_sgemm(a.as_ptr(), b.as_ptr(), s.as_mut_ptr(), m as i64, k as i64, n as i64, 0);
                }
                (a, b, s)
            })
            .collect();
        for t in 1..=16usize {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .expect("test pool");
            let (m, k, n) = shapes[(t - 1) % 2];
            let (a, b, serial) = &refs[(t - 1) % 2];
            for steal_order in [false, true] {
                for skinny_a in [false, true] {
                    let mut got = vec![0.0f32; m * n];
                    unsafe {
                        let args = GemmArgs {
                            a: a.as_ptr(), b: b.as_ptr(), c: got.as_mut_ptr(),
                            m, k, n, beta: 0.0, bt: false, epi: None,
                        };
                        sgemm_2d_blocks_dyn(Some(&pool), args, steal_order, skinny_a);
                    }
                    assert_eq!(
                        serial, &got,
                        "dyn != serial (threads {t} order {steal_order} skinnyA{skinny_a} m{m} k{k} n{n})"
                    );
                }
            }
            // The static range-split scheduler (the `WUKONG_GEMM_DYN=0` instrument) at the same width.
            let mut st = vec![0.0f32; m * n];
            unsafe {
                let args = GemmArgs {
                    a: a.as_ptr(), b: b.as_ptr(), c: st.as_mut_ptr(),
                    m, k, n, beta: 0.0, bt: false, epi: None,
                };
                sgemm_2d_blocks(Some(&pool), args);
            }
            assert_eq!(serial, &st, "static 2d != serial (threads {t} m{m} k{k} n{n})");
        }
    }

    /// The shared-pack 2D path ([`sgemm_2d_shared`] — retired as a default, kept as the
    /// `WUKONG_GEMM_2D_SHARED=1|band` instrument; called DIRECTLY
    /// so the test is immune to the process-wide `WUKONG_GEMM_2D*` OnceLock state) must be
    /// **bit-for-bit** identical to the serial kernel: the shared buffers hold the same packed
    /// bytes the serial packers produce, each C block has exactly one owner per K-block, and the
    /// `beta_eff` / final-K-epilogue rules are the serial kernel's. Shapes exercise: an odd
    /// square (300³ — nothing block-aligned), a `k = 1024` spanning THREE `select_kc` groups
    /// (344+344+336), the skinny-M single-block-row grids the policy special-cases —
    /// (128, 768, 768) and long-K (128, 3072, 768), both `bt` legs — a tall grid (768, 768, 128),
    /// an `m = 13` ragged edge, and the full bt × beta matrix on an edge-remainder shape
    /// (214 % 6 = 4, 616 % 16 = 8) with 3 UNEVEN K-blocks (901 → 304+304+293). The kill-switch
    /// leg pins the per-block path ([`sgemm_2d_blocks`], what `WUKONG_GEMM_2D_SHARED=0` routes
    /// to) on a skinny shape whose grid the new policy changed — direct calls, since the env is
    /// read once per process and cannot be flipped in-test. The epilogue leg pins the
    /// final-K-block-only rule, the block-origin bias shift, the null-bias shift, and the α scale
    /// through the shared path's phase-2 writeback.
    #[test]
    fn sgemm_2d_shared_matches_serial() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            println!("no avx2/fma — skipping");
            return;
        }
        // (m, k, n, bt combos, beta combos).
        let shapes: &[(usize, usize, usize, &[bool], &[i64])] = &[
            (300, 300, 300, &[false], &[0]),          // odd square, nothing block-aligned
            (512, 1024, 512, &[false], &[0]),         // k spans 3 select_kc groups (344+344+336)
            (128, 768, 768, &[false, true], &[0]),    // skinny M: one block row (policy rule)
            (128, 3072, 768, &[false, true], &[0]),   // skinny M, long K: 8 shared-pack rounds
            (768, 768, 128, &[false], &[0]),          // tall: many block rows, few columns
            (13, 700, 530, &[false, true], &[0]),     // ragged m = 13 (one 18-high block row)
            (214, 901, 616, &[false, true], &[0, 1]), // edge remainders, uneven K, full matrix
        ];
        for &(m, k, n, bts, betas) in shapes {
            for &bt in bts {
                for &beta in betas {
                    let a = fill(91, m * k);
                    let b = fill(92, k * n); // same length either way: bt reads it as [n, k]
                    let base = fill(93, m * n); // beta=1 must accumulate onto a nonzero C
                    let mut serial = base.clone();
                    let mut shared = base.clone();
                    let (mi, ki, ni) = (m as i64, k as i64, n as i64);
                    unsafe {
                        if bt {
                            wukong_sgemm_nt(a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(), mi, ki, ni, beta);
                        } else {
                            wukong_sgemm(a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(), mi, ki, ni, beta);
                        }
                        let args = GemmArgs {
                            a: a.as_ptr(), b: b.as_ptr(), c: shared.as_mut_ptr(),
                            m, k, n, beta: beta as f32, bt, epi: None,
                        };
                        sgemm_2d_shared(gemm_pool(), args);
                    }
                    assert_eq!(serial, shared, "2d-shared != serial (m{m} k{k} n{n} bt{bt} beta{beta})");
                }
            }
        }
        // Kill-switch leg: the per-block path must still match serial on a skinny grid the new
        // policy shapes (1 block row × 24 columns) — the adjacent-run A/B instrument stays honest.
        {
            let (m, k, n) = (128usize, 3072usize, 768usize);
            let a = fill(94, m * k);
            let b = fill(95, k * n);
            let (mut serial, mut blocks) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
            unsafe {
                wukong_sgemm_nt(a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(), m as i64, k as i64, n as i64, 0);
                let args = GemmArgs {
                    a: a.as_ptr(), b: b.as_ptr(), c: blocks.as_mut_ptr(),
                    m, k, n, beta: 0.0, bt: true, epi: None,
                };
                sgemm_2d_blocks(gemm_pool(), args);
            }
            assert_eq!(serial, blocks, "2d per-block (kill-switch path) != serial on the skinny grid");
        }
        // Epilogue leg: bias + every activation, plus the no-bias and α-scaled forms, on the nt
        // shape with 3 uneven K-blocks — a per-K-block epilogue bug folds the bias/activation
        // three times over, and a missing block-origin shift reads the wrong bias entries in
        // every column block past the first.
        {
            let (m, k, n) = (214usize, 901usize, 616usize);
            assert!(k > select_kc(k), "epi leg must span multiple K-blocks");
            let a = fill(96, m * k);
            let b = fill(97, n * k);
            let bias = fill(98, n);
            let run_shared = |epi: Epilogue, serial_ref: &[f32], tag: &str| {
                let mut shared = vec![0.0f32; m * n];
                unsafe {
                    let args = GemmArgs {
                        a: a.as_ptr(), b: b.as_ptr(), c: shared.as_mut_ptr(),
                        m, k, n, beta: 0.0, bt: true, epi: Some(epi),
                    };
                    sgemm_2d_shared(gemm_pool(), args);
                }
                assert_eq!(serial_ref, &shared[..], "2d-shared nt+epi != serial ({tag})");
            };
            for &act in &[ACT_IDENTITY, ACT_RELU, ACT_GELU, ACT_SILU] {
                let mut serial = vec![0.0f32; m * n];
                unsafe {
                    wukong_sgemm_nt_epi(
                        a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(),
                        m as i64, k as i64, n as i64, 0, bias.as_ptr(), act as i64,
                    );
                }
                run_shared(
                    Epilogue { bias: bias.as_ptr(), act, alpha: 1.0 },
                    &serial,
                    &format!("act{act} bias"),
                );
            }
            // No bias: the null bias base must stay null through both shifts.
            let mut serial = vec![0.0f32; m * n];
            unsafe {
                wukong_sgemm_nt_epi(
                    a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(),
                    m as i64, k as i64, n as i64, 0, std::ptr::null(), ACT_RELU as i64,
                );
            }
            run_shared(
                Epilogue { bias: std::ptr::null(), act: ACT_RELU, alpha: 1.0 },
                &serial,
                "relu nobias",
            );
            // α-scaled (the attention QKᵀ/√d form): folded on the final K-block only.
            let mut serial_alpha = vec![0.0f32; m * n];
            unsafe {
                wukong_sgemm_nt_alpha(
                    a.as_ptr(), b.as_ptr(), serial_alpha.as_mut_ptr(),
                    m as i64, k as i64, n as i64, 0, 0.125,
                );
            }
            run_shared(
                Epilogue { bias: std::ptr::null(), act: ACT_IDENTITY, alpha: 0.125 },
                &serial_alpha,
                "alpha 0.125",
            );
        }
    }

    /// Probe (run: `cargo test -p wukong_runtime --release -- --ignored --nocapture epi_throughput`).
    /// Fused `nt_epi` vs the unfused `nt` GEMM + a separate bias+ReLU pass over C — the traffic the
    /// fold eliminates (C is written once instead of written, then read-modify-written). The win is a
    /// fraction of the C-pass traffic, so it grows as K shrinks (the GEMM stops dominating).
    #[test]
    #[ignore]
    fn epi_throughput() {
        use std::time::Instant;
        let bench = |f: &dyn Fn()| {
            for _ in 0..3 {
                f();
            }
            let mut best = f64::INFINITY;
            for _ in 0..30 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best * 1e3
        };
        for &(m, k, n) in &[
            (512usize, 512usize, 512usize),
            (512, 64, 2048),
            (512, 32, 4096),
        ] {
            let a = fill(1, m * k);
            let b = fill(2, n * k);
            let bias = fill(3, n);
            let mut c = vec![0.0f32; m * n];
            let (ap, bp, biasp, cp) = (a.as_ptr(), b.as_ptr(), bias.as_ptr(), c.as_mut_ptr());
            let sep = bench(&|| unsafe {
                wukong_sgemm_nt(ap, bp, cp, m as i64, k as i64, n as i64, 0);
                // Separate bias+ReLU pass over C (what the compiler emits as a standalone loop).
                for i in 0..m {
                    for j in 0..n {
                        let v = *cp.add(i * n + j) + *biasp.add(j);
                        *cp.add(i * n + j) = v.max(0.0);
                    }
                }
                std::hint::black_box(cp);
            });
            let fused = bench(&|| unsafe {
                wukong_sgemm_nt_epi(ap, bp, cp, m as i64, k as i64, n as i64, 0, biasp, 1);
                std::hint::black_box(cp);
            });
            // The `@parallel` fused FFN: same fold, spread across cores (used to be serial-only).
            let fused_par = bench(&|| unsafe {
                wukong_sgemm_nt_epi_parallel(ap, bp, cp, m as i64, k as i64, n as i64, 0, biasp, 1);
                std::hint::black_box(cp);
            });
            println!(
                "m{m} k{k} n{n}: separate {sep:7.3} ms | fused {fused:7.3} ms ({:.2}x) | \
                 fused@parallel {fused_par:7.3} ms ({:.2}x vs serial-fused)",
                sep / fused,
                fused / fused_par
            );
        }
    }

    /// Fused-epilogue `C = act(A·Bᵀ + bias)` matches computing the GEMM then applying the epilogue
    /// separately, for {identity, ReLU} × {bias, no-bias}. The `k = 300`/`257` sizes exceed `KC=256`
    /// so the K loop blocks — verifying the epilogue is applied exactly once, on the final K-block,
    /// not per-block (a per-block bug would add the bias / clamp repeatedly).
    #[test]
    fn sgemm_nt_epi_matches_reference() {
        for &(m, k, n) in &[(7, 17, 13), (64, 64, 64), (40, 300, 48), (50, 257, 80)] {
            let a = fill(11, m * k);
            let b = fill(12, n * k);
            let bias = fill(13, n);
            let base = naive_nt(&a, &b, m, k, n); // C = A·Bᵀ
            for &act in &[ACT_IDENTITY, ACT_RELU, ACT_GELU, ACT_SILU] {
                for use_bias in [false, true] {
                    let mut want = base.clone();
                    for i in 0..m {
                        for j in 0..n {
                            let mut v = want[i * n + j];
                            if use_bias {
                                v += bias[j];
                            }
                            v = match act {
                                ACT_RELU => v.max(0.0),
                                ACT_GELU => crate::vmath::gelu1(v),
                                ACT_SILU => crate::vmath::silu1(v),
                                _ => v,
                            };
                            want[i * n + j] = v;
                        }
                    }
                    let mut got = vec![0.0f32; m * n];
                    let bias_ptr = if use_bias {
                        bias.as_ptr()
                    } else {
                        std::ptr::null()
                    };
                    unsafe {
                        wukong_sgemm_nt_epi(
                            a.as_ptr(),
                            b.as_ptr(),
                            got.as_mut_ptr(),
                            m as i64,
                            k as i64,
                            n as i64,
                            0,
                            bias_ptr,
                            act as i64,
                        );
                    }
                    let tol = 1e-3 * (k as f32).sqrt();
                    for idx in 0..m * n {
                        assert!(
                            (got[idx] - want[idx]).abs() <= tol + 1e-4 * want[idx].abs(),
                            "epi (m{m} k{k} n{n} act{act} bias{use_bias}) idx {idx}: got {} want {}",
                            got[idx],
                            want[idx]
                        );
                    }
                }
            }
        }
    }

    /// α-scaled `C = alpha·(A·Bᵀ)` must equal the plain `nt` GEMM followed by a scalar `alpha·c` pass
    /// (the α multiply is one exact f32 op on the fully-reduced dot — the *only* difference from the
    /// unscaled kernel), for both `alpha == 1.0` (must be **byte-identical** to `nt`) and a general
    /// `alpha`; and the serial and parallel α kernels must be bit-for-bit identical. `k = 300` exceeds
    /// `KC` so the K loop blocks — verifying α is applied once, on the final K-block, not per-block.
    #[test]
    fn sgemm_nt_alpha_matches_scaled_nt_and_parallel() {
        for &(m, k, n) in &[(7, 17, 13), (64, 64, 64), (40, 300, 48), (520, 300, 400)] {
            let a = fill(61, m * k);
            let b = fill(62, n * k);
            let (mi, ki, ni) = (m as i64, k as i64, n as i64);
            for &alpha in &[1.0f32, 0.125, -2.5] {
                // Reference: the exact `nt` GEMM, then a scalar `alpha·c` pass (same single f32 multiply
                // the α kernel folds into the writeback).
                let mut want = vec![0.0f32; m * n];
                unsafe {
                    wukong_sgemm_nt(a.as_ptr(), b.as_ptr(), want.as_mut_ptr(), mi, ki, ni, 0);
                }
                for v in &mut want {
                    *v *= alpha;
                }
                let mut got = vec![0.0f32; m * n];
                let mut got_par = vec![0.0f32; m * n];
                unsafe {
                    wukong_sgemm_nt_alpha(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), mi, ki, ni, 0, alpha);
                    wukong_sgemm_nt_alpha_parallel(
                        a.as_ptr(), b.as_ptr(), got_par.as_mut_ptr(), mi, ki, ni, 0, alpha,
                    );
                }
                assert_eq!(got, want, "nt_alpha != alpha·nt (m{m} k{k} n{n} alpha{alpha})");
                assert_eq!(got, got_par, "nt_alpha serial vs parallel (m{m} k{k} n{n} alpha{alpha})");
            }
            // alpha == 1.0 must be byte-identical to the plain `nt` kernel (the multiply is skipped).
            let mut plain = vec![0.0f32; m * n];
            let mut a1 = vec![0.0f32; m * n];
            unsafe {
                wukong_sgemm_nt(a.as_ptr(), b.as_ptr(), plain.as_mut_ptr(), mi, ki, ni, 0);
                wukong_sgemm_nt_alpha(a.as_ptr(), b.as_ptr(), a1.as_mut_ptr(), mi, ki, ni, 0, 1.0);
            }
            assert_eq!(plain, a1, "nt_alpha(1.0) must be byte-identical to nt (m{m} k{k} n{n})");
        }
    }

    /// The parallel fused epilogue (`wukong_sgemm_nt_epi_parallel`) must be **bit-for-bit** identical
    /// to the serial `wukong_sgemm_nt_epi` (each C tile is owned by one task and the per-(i,j)
    /// accumulation order is unchanged) — this is what lets the interpreter call the serial fused kernel
    /// as the oracle for a `@parallel` FFN and stay exact. Sizes exceed `PAR_MIN_MACS` so the parallel
    /// path is really taken, with `k > KC` so the epilogue-on-final-K-block logic is exercised.
    #[test]
    fn sgemm_nt_epi_parallel_matches_serial() {
        for &(m, k, n) in &[(512, 300, 512), (640, 300, 400)] {
            assert!((m * k * n) as u64 >= PAR_MIN_MACS, "size must trip the parallel path");
            let a = fill(21, m * k);
            let b = fill(22, n * k);
            let bias = fill(23, n);
            for &act in &[ACT_IDENTITY, ACT_RELU, ACT_GELU, ACT_SILU] {
                for use_bias in [false, true] {
                    let bias_ptr = if use_bias {
                        bias.as_ptr()
                    } else {
                        std::ptr::null()
                    };
                    let (mut serial, mut par) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
                    unsafe {
                        wukong_sgemm_nt_epi(
                            a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(),
                            m as i64, k as i64, n as i64, 0, bias_ptr, act as i64,
                        );
                        wukong_sgemm_nt_epi_parallel(
                            a.as_ptr(), b.as_ptr(), par.as_mut_ptr(),
                            m as i64, k as i64, n as i64, 0, bias_ptr, act as i64,
                        );
                    }
                    assert_eq!(serial, par, "par-fused != serial-fused (m{m} k{k} n{n} act{act} bias{use_bias})");
                }
            }
        }
    }
}
