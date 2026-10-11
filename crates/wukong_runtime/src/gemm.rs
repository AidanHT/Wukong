//! Single-precision GEMM — the heavy ML kernel.
//!
//! Computes `C[M,N] = A[M,K] · B[K,N]` (row-major f32) with a `beta` flag: `beta == 0` overwrites
//! `C`, any other `beta` accumulates into it (it is a flag, never a scale). This is the tuned
//! microkernel the Wukong compiler lowers a recognized matmul loop nest to (see `wukong_mir_build`'s
//! matmul recognizer) — analogous to how XLA/TVM/oneDNN lower a matmul op to an optimized
//! microkernel rather than emitting the naive nest.
//!
//! The whole exported family lives here: `nn`/`nt` (`A·Bᵀ`)/`tn` (`Aᵀ·B`, A packed straight from its
//! `[k, m]` storage), the fused epilogue `_nt_epi` (`act(A·Bᵀ + bias)`), the α-scaled `_nt_alpha`,
//! bf16/f16 input forms (lossless SIMD widen prepass, into reused per-thread scratch, feeding the
//! same f32 kernel), and a `_parallel` twin of each. LANDMINE:
//! `alpha` folds into the same final-K-block writeback that performs the accumulate, so it scales the
//! beta-accumulated value — only `beta == 0` is supported with a non-unit α (see [`Epilogue`] and the
//! `debug_assert` in [`gemm_dispatch`]). This module also owns [`gemm_pool`], the crate's unified
//! physical-core worker pool.
//!
//! Structure (the classic BLIS five-loop GEMM):
//!   * register-block: a `MR×NR` tile of C is held in vector registers across the K loop;
//!   * cache-block: `MC×KC` panels of A and `KC×NC` panels of B are packed into contiguous,
//!     aligned scratch so the microkernel streams them with unit stride;
//!   * the microkernel is AVX2 + FMA (true 256-bit, 8 f32/lane) — the width Cranelift can't emit —
//!     with a portable scalar fallback selected at runtime via `is_x86_feature_detected!`, and a
//!     bit-identical AVX-512 twin ([`micro_6x16_avx512`]) that takes over only the full-tile,
//!     no-epilogue case where `avx512f` is detected (false on this box, so it is dead code here).
//!
//! The native backend calls this directly; the interpreter marshals its abstract memory into real
//! buffers and calls the *same* function, so the differential oracle stays bit-for-bit exact.
//!
//! # The numeric contract
//!
//! `interp == native` holds only because both sides call one function. What that function *computes*
//! is this, and it is a property of the build, not of the language:
//!
//!   * **Order.** For each `(i, j)` independently, K is split by [`select_kc`] into
//!     `nb = ceil(k / select_kc(k))` blocks. Each block accumulates ascending `p` from a fresh `+0.0`
//!     with **one rounding per step** (`vfmadd` on the AVX2 path, [`f32::mul_add`] in
//!     [`sgemm_scalar`]), and is folded into `C` with one further rounding
//!     (`beta_eff = 1.0` on every block after the first). `MC`, `NC`, the `MR×NR` register tile, the
//!     packing and every parallel schedule only decide *which core computes which `(i, j)`* — none of
//!     them touches the order, which is why serial, parallel and the AVX-512 twin are bit-identical.
//!   * **Consequences.** The result is a function of `KC` (retuning it changes the low bits of every
//!     GEMM in the project — hence `select_kc_partition_is_pinned`), and it is NOT a function of the
//!     core count, the scheduler, the env knobs, or whether the AVX2 path or [`sgemm_scalar`] ran.
//!   * **Error bound.** Every reference test asserts the bound this order earns, against an **f64**
//!     reference, via the one helper `assert_within_derived_bound` (the fused and scalar epilogue
//!     legs assert something STRICTER on top of that — bit-identity with the unfused sum — and no
//!     test in the file asserts an absolute tolerance):
//!     `|got − exact| ≤ γ_r · (Σ_p |a_ip|·|b_pj| + |β·c_ij|)` with
//!     `r = kc + nb − 1 (+1 when β ≠ 0)` and `γ_r = r·u/(1−r·u)`, `u = 2⁻²⁴`. The `Σ|a||b|` growth
//!     factor is netlib's Level-3 BLAS test criterion (`SMMCH` in `sblat3.f`, which normalizes by
//!     `G(i) = Σ |A(i,k)|·|B(k,j)|`); `γ_r` is Higham, *Accuracy and Stability of Numerical
//!     Algorithms*, 2nd ed., §3.1 (Lemma 3.1), specialized to this kernel's known rounding count
//!     instead of netlib's implementation-agnostic `√u`.
//!   * **The epilogue is exact, not approximate.** `act(α·(A·Bᵀ) + bias)` fused into the final-K-block
//!     writeback is required to be **bit-identical** to the unfused `{ t = A·Bᵀ; act(α·t + bias) }`
//!     (`sgemm_nt_epi_matches_reference`), which is what lets the interpreter use the serial fused
//!     kernel as the oracle for a `@parallel` FFN.

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

/// The run-time row/column cache blocking of the five-loop GEMM (P13.4): the A-panel height `mc`
/// and the B-panel width `nc`. [`MC`]/[`NC`] are the measured values on the reference part (the
/// 2 MiB-L2 / 24 MiB-L3 Meteor Lake this crate was tuned on); [`gemm_blocking`] resolves the
/// values a process actually uses -- `WUKONG_GEMM_MC` / `WUKONG_GEMM_NC` when set and valid, else
/// the reference scaled to the probed L2/L3.
///
/// **Blocking-only, never numerics.** `mc` and `nc` choose which rows and columns of C are swept
/// together; every C element is still accumulated over the same [`select_kc`] K-groups in
/// ascending order through the same microkernel, so no `mc`/`nc` changes a result bit -- pinned by
/// `mc_nc_blocking_never_changes_a_bit` (serial, every pack schedule, epilogue, transposes) and
/// `legacy_parallel_nc_never_changes_a_bit`. That proof is what licenses deriving them from the
/// host. [`KC`] and `reduce.rs`'s `RCHUNK` are the opposite case: numeric-contract constants that
/// fix a summation order, so they are NOT host- or env-tunable (P13.9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Blocking {
    /// A-panel height: a positive multiple of [`MR`].
    mc: usize,
    /// B-panel width: a positive multiple of [`NR`].
    nc: usize,
}

impl Blocking {
    /// The measured blocking, [`MC`] x [`NC`].
    const REFERENCE: Blocking = Blocking { mc: MC, nc: NC };
}

/// The private L2 the reference [`MC`] was measured against (an `MC x KC` A panel is ~1/9 of it).
const REF_L2_BYTES: usize = 2 << 20;
/// The shared L3 the reference [`NC`] was measured against (a `KC x NC` B panel is ~1/4 of it).
const REF_L3_BYTES: usize = 24 << 20;
/// Largest `WUKONG_GEMM_MC`/`WUKONG_GEMM_NC` accepted: bounds the pack scratch an override can ask
/// for (a 65536 x 384 f32 panel is 96 MiB).
const MAX_BLOCK_OVERRIDE: usize = 1 << 16;

/// Scale a reference block size by `have / reference_cache`, keeping the panel's measured share of
/// that cache level, rounded down to a multiple of `unit` and clamped to `[reference/4,
/// reference*4]` -- the derivation extrapolates a measurement, so it stays within 4x of it.
fn scale_block(reference: usize, have: usize, reference_cache: usize, unit: usize) -> usize {
    let scaled = (reference as u128 * have as u128 / reference_cache as u128) as usize;
    let lo = (reference / 4 / unit).max(1) * unit;
    let hi = reference * 4;
    (scaled / unit * unit).clamp(lo, hi)
}

/// The host-derived default: `mc` scales with the probed L2, `nc` with the probed L3; an unknown
/// level keeps the reference value. Identity on the reference part.
fn derive_blocking(l2: Option<usize>, l3: Option<usize>) -> Blocking {
    Blocking {
        mc: l2
            .filter(|&b| b > 0)
            .map_or(MC, |b| scale_block(MC, b, REF_L2_BYTES, MR)),
        nc: l3
            .filter(|&b| b > 0)
            .map_or(NC, |b| scale_block(NC, b, REF_L3_BYTES, NR)),
    }
}

/// `WUKONG_GEMM_MC` / `WUKONG_GEMM_NC`: a plain ASCII decimal that is a positive multiple of `unit`
/// ([`MR`] / [`NR`]) and at most [`MAX_BLOCK_OVERRIDE`]; anything else is ignored.
fn parse_block_override(raw: Option<&str>, unit: usize) -> Option<usize> {
    let s = raw?;
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<usize>()
        .ok()
        .filter(|&v| v >= unit && v.is_multiple_of(unit) && v <= MAX_BLOCK_OVERRIDE)
}

/// The process's GEMM blocking, resolved once: each of `WUKONG_GEMM_MC` / `WUKONG_GEMM_NC` when set
/// and valid (see [`parse_block_override`]), else [`derive_blocking`] from the probed host caches.
/// The crate's own unit-test build pins [`Blocking::REFERENCE`], so the shape tables written against
/// `MC`/`NC` keep their meaning on any CI host; the bit-invariance tests drive other blockings
/// explicitly.
fn gemm_blocking() -> Blocking {
    if cfg!(test) {
        return Blocking::REFERENCE;
    }
    use std::sync::OnceLock;
    static V: OnceLock<Blocking> = OnceLock::new();
    *V.get_or_init(|| {
        let t = crate::host_topology();
        let derived = derive_blocking(t.l2_bytes, t.l3_bytes);
        let var = |k: &str| std::env::var(k).ok();
        Blocking {
            mc: parse_block_override(var("WUKONG_GEMM_MC").as_deref(), MR).unwrap_or(derived.mc),
            nc: parse_block_override(var("WUKONG_GEMM_NC").as_deref(), NR).unwrap_or(derived.nc),
        }
    })
}

/// Upper bound on `m` for the 2D block-parallel grid's **skinny-M** rule — one row of C blocks, see
/// [`select_2d_block_shape`]. This is a *policy* boundary, not a cache size: it answers "is A short
/// enough that one L2-resident A stripe should serve every column block", a question measured at
/// the transformer activation shapes (S ≤ 144). It equals [`MC`] only because both were tuned on
/// the same 2 MiB-L2 part. Until 2026-10 the policy read `MC` itself, so retuning the A-panel cache
/// block for a different L2 would also have reclassified, say, a 200-row GEMM as skinny and sent it
/// down a single-block-row grid that was never measured there, while the policy test's
/// `m <= MC` assertion stayed green with a silently changed meaning. Now `MC` means only the A-panel
/// height and this constant is pinned by value in `select_2d_block_shape_policy`.
#[cfg(target_arch = "x86_64")]
const SKINNY_M_MAX: usize = 144;

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
/// the full ~384. Result ∈ [4, KC].
///
/// **This is part of the kernel's observable result, not just its speed.** `kc` fixes how the K
/// reduction is *grouped*: each block accumulates from a fresh zero and is folded into C with one
/// extra rounding (`beta_eff = 1.0`), so retuning `KC` changes the low f32 bits of every GEMM in the
/// project. Every kernel — serial, parallel, AVX2, [`sgemm_scalar`] — calls this one function, which
/// is why they agree bit-for-bit; nothing makes one `KC` agree with another. The partition is
/// therefore pinned BY VALUE in `select_kc_partition_is_pinned` so a retune is a reviewed diff, and
/// the reference tests assert the bound this grouping actually earns (`γ_r·Σ|a||b|`, see the module
/// header) — not the `1e-3·√k` absolute slack that stood here until 2026-09-13, which contained no ε
/// at all and was 131-334× looser than the rounding scale of its own test data.
#[inline]
pub(crate) fn select_kc(k: usize) -> usize {
    let nblocks = k.div_ceil(KC).max(1);
    round_up(k.div_ceil(nblocks), 4) // multiple of the ×4 K-unroll; ≤ KC since k/nblocks ≤ KC
}

static GEMM_POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();

/// Usable L2 for one serial GEMM column block. The Redwood Cove P-core on this Meteor Lake part
/// has a **2 MiB** private L2; this is the whole of it, because the three arrays budgeted against
/// it in [`select_nc`] (the packed B panel, the packed A block, the live C column strip) are the
/// only large working set the serial five-loop has in flight, and the B source rows stream through
/// without reuse. Env-overridable (`WUKONG_GEMM_L2_BYTES`, read once) so the derivation stays
/// A/B-measurable on a machine with a different L2. Blocking-only: it never changes the bits.
const L2_BYTES: usize = 2 * 1024 * 1024;

fn l2_bytes() -> usize {
    use std::sync::OnceLock;
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("WUKONG_GEMM_L2_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v >= 64 * 1024)
            .unwrap_or(L2_BYTES)
    })
}

/// [`select_nc_with`] under the process blocking ([`gemm_blocking`]) -- the shape tables' spelling.
#[cfg(test)]
fn select_nc(m: usize, k: usize, n: usize) -> usize {
    select_nc_with(m, k, n, gemm_blocking())
}

/// Column-block width (`nc`) for the SERIAL five-loop — [`NC`] except when the row loop has no
/// reuse to sell.
///
/// The packed B panel (`kc × nc`) is written once per `(jc, pc)` and read once per `ic` block.
/// With several `ic` blocks the classic BLIS choice wins: make `NC` as wide as the L3 allows, so
/// A is packed once and every extra `ic` pass re-reads an already-resident panel. That is what
/// `NC = 4080` is for and it stays.
///
/// With a **single** `ic` block — `m <= MC`, the skinny/small-M regime — there is no reuse at all:
/// every byte packed into B is read exactly once, so an L3-resident panel makes the pack a pure
/// L2→L3→L2 round trip on top of the copy. MEASURED at `m=128, k=768, n=3072` (`kc=384`), where
/// `nc = 3072` gives a 4.7 MiB panel: capping to `nc = 768` (a 1.18 MiB panel) took the
/// macro-kernel from 95.5 to 107.7 GF/s and the whole phase sum from 7055 to 6245 us, level with
/// the other `m=128` shapes instead of 12% below them (AGENT-LOCAL, `gemm_phase_cost`).
///
/// The cap solves `4·kc·nc + 4·mc·kc + 4·mc·nc <= L2` for `nc` — B panel, A block and the C column
/// strip the tile writebacks touch — then splits `n` into the fewest equal-ish blocks under it, the
/// same "no thin tail" rule [`select_kc`] uses for `k`. Result is a multiple of [`NR`] and `>= NR`.
///
/// Blocking-only: `nc` changes which C columns are computed together, never the per-`(i, j)` order
/// of the K reduction, so every output bit is unchanged and serial stays bit-identical to parallel.
///
/// Under an explicit [`Blocking`]: `blk.mc` decides the single-row-block regime and `blk.nc` is
/// the classic width.
fn select_nc_with(m: usize, k: usize, n: usize, blk: Blocking) -> usize {
    let (mc_blk, nc_blk) = (blk.mc, blk.nc);
    if m > mc_blk {
        return nc_blk;
    }
    let (mc, kc) = (m.min(mc_blk), select_kc(k));
    let l2 = l2_bytes();
    let a_block = 4 * round_up(mc, MR) * kc;
    // Saturating: a pathological `WUKONG_GEMM_L2_BYTES` smaller than the A block alone must fall
    // back to one NR-wide panel, not wrap.
    let cap = l2.saturating_sub(a_block) / (4 * (kc + mc));
    let cap = (cap & !(NR - 1)).clamp(NR, nc_blk);
    if n <= cap {
        return nc_blk; // one block either way; leave the classic path untouched
    }
    let nblocks = n.div_ceil(cap);
    round_up(n.div_ceil(nblocks), NR).clamp(NR, nc_blk)
}

/// A private rayon pool for the parallel GEMM, sized to the machine's **physical** core count instead
/// of rayon's default (one worker per *logical* core). A compute-bound AVX2-FMA GEMM already saturates
/// a core's two FMA pipes with a single thread, so the HyperThread sibling adds only scheduling
/// contention: measured at the mid sizes where scaling lags hardest, 16 physical threads scale ~30-40%
/// better than 22 logical (512³ 2.2×→3.1×, 1024³ 3.2×→3.9× over single core). A *private* pool (not a
/// global resize) leaves the default pool — and the thread-count-derived striping of the reduction
/// kernels that run on it — untouched. The GEMM result is independent of how panels are distributed
/// (the per-(i,j) K-order is fixed), so this changes throughput only, never the bits: serial stays
/// bit-identical to parallel. `None` (⇒ caller uses the default pool) when the usable physical
/// cores are at least the global pool's width (no HT to avoid — see
/// `topology::private_pool_width`) or the pool can't be built.
/// Since 2026-07-10 this is the crate's UNIFIED kernel pool: `run_on_wuk_pool` (lib.rs) routes
/// the `_parallel` norms/velem/reductions and `wukong_parallel_for` regions here too (default;
/// `WUKONG_POOL_UNIFY=0` opts out), so a transformer forward stays on ONE warm pool instead of
/// bouncing private↔global between adjacent kernels. Worker stacks are 16 MiB for the outlined
/// region bodies' privatized frames (same rationale as `ensure_global_pool`). The routed kernels
/// all chunk independently of worker count (fixed-chunk velem/reductions, per-row norms,
/// per-index regions), so the striping caveat above is moot for them — bits never change.
pub(crate) fn gemm_pool() -> Option<&'static rayon::ThreadPool> {
    GEMM_POOL
        .get_or_init(|| {
            // Configure the global pool FIRST: `current_num_threads()` below instantiates the default
            // registry if none exists, and it must come up with the runtime's required configuration
            // (16 MiB stacks, RAYON_NUM_THREADS honored) rather than rayon's bare defaults.
            crate::ensure_global_pool();
            // Physical cores the process can actually use (affinity mask and cgroup quota
            // applied), not the machine-wide `num_cpus::get_physical()`, which compared a whole
            // machine's cores against a masked pool width and built nothing under `taskset`.
            // `None`: no SMT siblings to shed, so the global pool is already one worker per core.
            let physical = crate::topology::private_pool_width(
                crate::topology::host_topology().physical_available,
                rayon::current_num_threads(),
            )?;
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

/// Fused B micropanel packing (**default ON**; `WUKONG_GEMM_FUSED_PACK=0` opts out, read once).
///
/// The classic five-loop packs a whole `kc×nc` B panel, then sweeps it once per `ic` row block.
/// When there is exactly ONE row-block sweep — the serial kernel's `m <= MC` regime, and every 2D
/// block-parallel block, whose height never exceeds one sweep — every packed byte is written to
/// the panel, evicted toward L2, and read back exactly once: the pack buys unit stride and
/// nothing else, and its source reads (the one unavoidable B traffic) run as a burst with no
/// compute to hide behind. P9.6 measured that burst at ~11% of the m=128 `nn.Linear` shapes
/// (`gemm_phase_cost`: packB 698 us against a 5608 us macro-kernel at 128×768·(3072×768)ᵀ).
///
/// [`macro_kernel_packb`] instead packs each `KC×NR` micropanel into a 24 KB L1-resident buffer
/// immediately before the row-panel sweep that consumes it, and threads a [`PfWalk`] through that
/// sweep's FMA loops so the NEXT micropanel's source lines arrive in L2 under the current one's
/// compute (BLIS's `b_next` prefetch). The packed bytes, the micro-tile calls and every
/// per-`(i, j)` K order are identical — it changes where the panel lives and when its source is
/// fetched, never a result bit (pinned by `fused_b_pack_is_bit_identical_to_the_classic_pack`).
/// UNMEASURED at landing (the build was deferred); the knob keeps the classic panel one
/// adjacent-run A/B away, the same instrument discipline as the path gates above.
#[cfg(target_arch = "x86_64")]
fn gemm_fused_pack() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| !std::env::var("WUKONG_GEMM_FUSED_PACK").is_ok_and(|v| v == "0"))
}

/// A software-prefetch schedule [`micro_6x16`] walks inside its K loop: one `PREFETCHT1` per
/// four-step iteration, over `rows` source rows `row_stride` floats apart, `lines` 64-byte lines
/// each, starting at `ptr`. [`macro_kernel_packb`] hands each row-panel call one slice of the NEXT
/// B micropanel's source ([`PfWalk::for_b_panel`]), so that source is L2-resident by the time it is
/// packed. Issuing at the FMA loop's own cadence (one line per ~24 cycles) rather than as a burst
/// before each call keeps the prefetches from occupying every L1 fill buffer at once while the
/// micro-tile streams its A panel from L2.
///
/// Architecturally a hint: the address is formed with `wrapping_add` and never dereferenced, so a
/// walk may run past the end of B (its last line can) without UB or a fault, and no result bit
/// depends on it. [`PfWalk::NONE`] is the no-prefetch walk every other caller passes. Not threaded
/// into the AVX-512 twin: on that path the walk is simply skipped.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct PfWalk {
    ptr: *const f32,
    row_stride: usize,
    lines: usize,
    rows: usize,
}

#[cfg(target_arch = "x86_64")]
impl PfWalk {
    /// No prefetch.
    const NONE: PfWalk = PfWalk {
        ptr: std::ptr::null(),
        row_stride: 0,
        lines: 0,
        rows: 0,
    };

    /// Slice `part` of `nparts` of the source rows that packing B micropanel `jp` of the `kc×nc`
    /// slice at `b_src` (leading dimension `ldb`; the [`b_origin`] of this K/column block) reads.
    /// With `bt` (B stored `[n, k]`) the micropanel's source is `ncols` rows of `kc` contiguous
    /// floats, `ldb` apart; otherwise (B stored `[k, n]`) it is `kc` rows of `ncols` floats. The
    /// parts partition the rows (`rows·part/nparts .. rows·(part+1)/nparts`), so every source row is
    /// assigned to exactly one row-panel call; a row of `row_floats` floats at an arbitrary
    /// alignment spans at most `ceil(4·row_floats / 64) + 1` lines.
    #[allow(clippy::too_many_arguments)]
    fn for_b_panel(
        b_src: *const f32,
        ldb: usize,
        bt: bool,
        kc: usize,
        nc: usize,
        jp: usize,
        part: usize,
        nparts: usize,
    ) -> PfWalk {
        let j0 = jp * NR;
        let ncols = (nc - j0).min(NR);
        let (rows, row_floats, first_row) = if bt {
            (ncols, kc, b_src.wrapping_add(j0 * ldb))
        } else {
            (kc, ncols, b_src.wrapping_add(j0))
        };
        let (lo, hi) = (rows * part / nparts, rows * (part + 1) / nparts);
        PfWalk {
            ptr: first_row.wrapping_add(lo * ldb),
            row_stride: ldb,
            lines: (row_floats * 4).div_ceil(64) + 1,
            rows: hi - lo,
        }
    }
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
    /// A stored transposed, `[k, m]` (the `C = Aᵀ·B` entries); see [`a_origin`].
    at: bool,
    bt: bool,
    epi: Option<Epilogue>,
}
#[cfg(target_arch = "x86_64")]
// SAFETY: `GemmArgs` bundles raw operand pointers; it is only moved into the fork-join closure, where each worker writes a DISJOINT C region, so sending it across threads is race-free.
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
        // SAFETY: the `#[target_feature(avx2,fma)]` on `run` guarantees the ISA; the field pointers satisfy sgemm_avx2_parallel's operand-size contract (GemmArgs' construction invariant).
        unsafe {
            sgemm_avx2_parallel(
                self.a, self.b, self.c, self.m, self.k, self.n, self.beta, self.at, self.bt,
                self.epi,
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
/// `c = act(alpha·(A·Bᵀ) + bias[col])` — for `beta == 0`, which is the only rule the compiler emits
/// (see the `debug_assert` in [`gemm_dispatch`]). With `beta != 0` the epilogue necessarily sees the
/// beta-accumulated value, i.e. `act(alpha·(C_old + A·Bᵀ) + bias[col])`: the final-K-block writeback
/// reads C once and cannot tell the caller's `C_old` apart from the partial sums the earlier
/// K-blocks left there (both arrive as the same `beta_eff == 1` read), so `C_old + alpha·(A·Bᵀ)`
/// is not expressible here without a second pass over C.
/// `bias` is null for no bias; `act` is one of [`ACT_IDENTITY`],
/// [`ACT_RELU`], [`ACT_GELU`], [`ACT_SILU`] (the transformer FFN activations); `alpha` is a
/// loop-invariant scalar applied to the matmul result before the bias-add (the attention score scale
/// `QKᵀ/√d` and every scaled projection). `alpha == 1.0` is the identity — the multiply is skipped so
/// every existing caller (which passes `1.0`) stays byte-identical.
/// Folding it here means `C` is written once with the scale+bias+activation already applied, instead of
/// separate passes over `C` — the memory traffic a `α·linear → bias → act` otherwise pays. Both backends
/// call the identical kernel, so the fused result stays bit-for-bit exact.
#[derive(Clone, Copy)]
pub(crate) struct Epilogue {
    bias: *const f32,
    act: u32,
    alpha: f32,
}

impl Epilogue {
    /// Apply the epilogue to value `x` at tile-local column `j` (relative to this epilogue's bias
    /// base): `act(alpha·x + bias[j])`. `# Safety`: `bias`, when non-null, must be valid at index `j`.
    #[inline]
    pub(crate) unsafe fn apply(&self, x: f32, j: usize) -> f32 {
        // The α scale multiplies the matmul result before the bias-add. `alpha == 1.0` skips the
        // multiply so the fused-epilogue callers (`nt_epi`, which pass 1.0) stay byte-identical.
        let mut y = if self.alpha != 1.0 { self.alpha * x } else { x };
        if !self.bias.is_null() {
            y += *self.bias.add(j);
        }
        // GELU/SiLU reuse `vmath`'s scalar form so a fused `act(x·Wᵀ+b)` is bit-identical to the
        // unfused `{ t = x·Wᵀ+b; act(t) }`. This is the scalar epilogue: the portable twin's
        // (`sgemm_scalar`), and the reference the AVX2 writebacks (full-tile and masked edge-tile,
        // both 256-bit since 2026-10) are pinned against lane for lane.
        match self.act {
            // Match vector MAXPS: either zero and NaN select the +0 bound.
            ACT_RELU => y = if y > 0.0 { y } else { 0.0 },
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
                // SAFETY: `shift` is only called with `cols` inside the column range the bias array covers (see the note above), so `add(cols)` stays in bounds.
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
/// **Aliasing: all operands disjoint** — `c` must overlap neither `a` nor `b`: `C` is written in
/// tiled order while both are still read (`a` and `b` may share storage; both are only read). A
/// `beta = 1` call reads `c` itself, in place by construction. The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, k, n);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_dispatch(a, b, c, m, k, n, beta, false, false, false, None);
}

/// `C = A·Bᵀ` (B is row-major `[n, k]`), the `nn.Linear` / `x @ Wᵀ` form — the most common matmul
/// in deep learning. Same `beta` rule, single-threaded.
///
/// # Safety
/// `a`, `b`, `c` valid for `m*k`, `n*k`, `m*n` `f32` elements respectively.
/// **Aliasing: all operands disjoint** — `c` must overlap neither `a` nor `b`: `C` is written in
/// tiled order while both are still read (`a` and `b` may share storage; both are only read). A
/// `beta = 1` call reads `c` itself, in place by construction. The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_dispatch(a, b, c, m, k, n, beta, false, true, false, None);
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
/// When non-null, the `n`-element bias range must be disjoint from the `m*n`-element
/// output range; rows may read bias while other output elements are written.
/// **Aliasing: all operands disjoint** — `c` overlaps none of `a`, `b` and the epilogue bias (`C`
/// is written in tiled order while they are still read).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_extent_opt(bias, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), bias, n);
    }
    let epi = Epilogue {
        bias,
        act: act as u32,
        alpha: 1.0,
    };
    gemm_dispatch(a, b, c, m, k, n, beta, false, true, false, Some(epi));
}

/// `C = alpha·(A·Bᵀ)` — the **α-scaled `nn.Linear`**: the attention score matmul `QKᵀ/√d` (and every
/// scaled projection), where a loop-invariant scalar `alpha` multiplies the dot. The compiler lowers a
/// matmul nest whose store is `c[i,j] = alpha·s` to this. `alpha` is folded into the C-tile writeback on
/// the final K-block (no second pass over C), reusing the fused-epilogue machinery with a null bias and
/// identity activation — so `C` is written once as `alpha·(A·Bᵀ)`. Single-threaded.
///
/// **`beta` is NOT the usual rule here.** `alpha` is folded into the same final-K-block writeback
/// that performs the accumulate, so `beta != 0` yields `alpha·(C_old + A·Bᵀ)`, not the BLAS
/// `C_old + alpha·(A·Bᵀ)` — see [`Epilogue`] for why the writeback cannot separate the two. The
/// only supported combination is `beta == 0`; [`gemm_dispatch`] `debug_assert`s it.
///
/// The α multiply is one exact f32 op applied to the fully-reduced dot, so the result equals the naive
/// nest's `alpha·s` under the documented matmul reassociation (both backends call this identical kernel,
/// and it is gated against an f64 reference).
///
/// # Safety
/// `a` valid for `m*k`, `b` for `n*k`, `c` for `m*n` `f32`.
/// **Aliasing: all operands disjoint** — `c` must overlap neither `a` nor `b`: `C` is written in
/// tiled order while both are still read (`a` and `b` may share storage; both are only read). A
/// `beta = 1` call reads `c` itself, in place by construction. The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    let epi = Epilogue {
        bias: std::ptr::null(),
        act: ACT_IDENTITY,
        alpha,
    };
    gemm_dispatch(a, b, c, m, k, n, beta, false, true, false, Some(epi));
}

/// Multi-threaded `C = alpha·(A·Bᵀ)` — the α-scaled `nn.Linear` across cores (the `@parallel` attention
/// score / scaled projection). The scale folds into each tile's final-K-block writeback, and each C tile
/// is owned by exactly one task with the same per-(i,j) accumulation order as the serial kernel — so it
/// is bit-identical to the serial `wukong_sgemm_nt_alpha` the interpreter oracle calls. Same `beta`
/// caveat as the serial entry point: only `beta == 0` is supported.
///
/// # Safety
/// `a` valid for `m*k`, `b` for `n*k`, `c` for `m*n` `f32`.
/// **Aliasing: all operands disjoint** — `c` must overlap neither `a` nor `b`: `C` is written in
/// tiled order while both are still read (`a` and `b` may share storage; both are only read). A
/// `beta = 1` call reads `c` itself, in place by construction. The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    let epi = Epilogue {
        bias: std::ptr::null(),
        act: ACT_IDENTITY,
        alpha,
    };
    gemm_dispatch(a, b, c, m, k, n, beta, false, true, true, Some(epi));
}

/// Pick AVX2 vs scalar and serial vs parallel; `at` reads A as stored `[k, m]` (`C = Aᵀ·B`, packed
/// straight from that layout), `bt` reads B as stored `[n, k]` (`C = A·Bᵀ`).
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm`] / [`wukong_sgemm_nt`] / [`wukong_sgemm_tn`].
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_dispatch(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    beta: i64,
    at: bool,
    bt: bool,
    par: bool,
    epi: Option<Epilogue>,
) {
    // `alpha` scales the beta-ACCUMULATED value, not the matmul result alone — see [`Epilogue`].
    // A non-unit `alpha` is therefore only meaningful with `beta == 0`. Every recognizer arm
    // honours that today (`wukong_mir_build` pins `beta: 0` on the α-peeling arm and
    // `alpha: None` on the β=1 residual arm), so this cannot fire in the current compiler; it is
    // here to fail loudly on the day a new arm peels α off a residual store instead of silently
    // gaining a spurious `alpha·` on the residual term.
    debug_assert!(
        beta == 0 || epi.is_none_or(|e| e.alpha == 1.0),
        "gemm: alpha is applied AFTER the beta accumulate — alpha != 1 requires beta == 0"
    );
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
        if crate::isa::avx2() && crate::isa::fma() {
            // SAFETY: features just checked; dims validated by the caller contract.
            unsafe {
                // Serial calls, including parallel entries below the work threshold, must not
                // initialize either worker pool merely to select the serial kernel.
                let pool = if par { gemm_pool() } else { None };
                match (par, pool) {
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
                        sgemm_avx2(a, b, c, m, k, n, beta, at, bt, epi)
                    }
                    (true, pool) => {
                        let args = GemmArgs {
                            a,
                            b,
                            c,
                            m,
                            k,
                            n,
                            beta,
                            at,
                            bt,
                            epi,
                        };
                        if gemm_2d() {
                            if gemm_2d_shared(macs) {
                                sgemm_2d_shared(pool, args);
                            } else if gemm_dyn() {
                                sgemm_2d_blocks_dyn(
                                    pool,
                                    args,
                                    gemm_steal_order(),
                                    gemm_skinny_a(),
                                );
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
                    (false, _) => sgemm_avx2(a, b, c, m, k, n, beta, at, bt, epi),
                }
            }
            return;
        }
    }
    sgemm_scalar(a, b, c, m, k, n, beta, at, bt, epi);
}

/// Multi-threaded `C = A·B`. For each `KC` contraction block it packs the *whole* A column-panel and
/// B row-panel once (shared, read-only), then runs every `MR×NR` register tile of the C grid in
/// parallel. Per-(i,j) accumulation order is identical to [`wukong_sgemm`], so the two agree
/// bit-for-bit (the interpreter oracle calls the serial one for both). Packing is shared, not
/// re-done per worker, so it scales to high core counts.
///
/// # Safety
/// Same contract as [`wukong_sgemm`].
/// **Aliasing: all operands disjoint** — `c` must overlap neither `a` nor `b`: `C` is written in
/// tiled order while both are still read (`a` and `b` may share storage; both are only read). A
/// `beta = 1` call reads `c` itself, in place by construction. The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, k, n);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_dispatch(a, b, c, m, k, n, beta, false, false, true, None);
}

/// Multi-threaded `C = A·Bᵀ` (the `nn.Linear` form).
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_nt`].
/// **Aliasing: all operands disjoint** — `c` must overlap neither `a` nor `b`: `C` is written in
/// tiled order while both are still read (`a` and `b` may share storage; both are only read). A
/// `beta = 1` call reads `c` itself, in place by construction. The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_dispatch(a, b, c, m, k, n, beta, false, true, true, None);
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
/// When non-null, the `n`-element bias range must be disjoint from the `m*n`-element
/// output range; rows may read bias while other output elements are written.
/// **Aliasing: all operands disjoint** — `c` overlaps none of `a`, `b` and the epilogue bias (`C`
/// is written in tiled order while they are still read).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_extent_opt(bias, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), bias, n);
    }
    let epi = Epilogue {
        bias,
        act: act as u32,
        alpha: 1.0,
    };
    gemm_dispatch(a, b, c, m, k, n, beta, false, true, true, Some(epi));
}

/// `C = Aᵀ·B` — A is stored row-major as `[k, m]` (`a[p*m + i]`), B row-major `[k, n]` (`b[p*n + j]`),
/// C is `[m, n]`. This is the **weight-gradient** GEMM of a training backward pass (`dW = dYᵀ·X`): the
/// contraction axis (the batch) is the *outer* index of both inputs, so A's logical `[m, k]` operand
/// is the transpose of its storage. gcc/rustc compile the naive nest with column-strided A reads that
/// defeat vectorization (one cache line per element).
///
/// Wukong packs A **straight from its `[k, m]` storage** (`gemm_dispatch(.., at = true, ..)`): the
/// `[kc][MR]` A micropanel the kernel wants is, for each contraction step, six CONSECUTIVE stored
/// elements — a contiguous copy, cheaper than the row-major packer's transpose. Until 2026-10 every
/// call instead allocated and zeroed an `m·k` f32 scratch matrix, transposed A into it and ran the
/// NN kernel on the copy (P9.19): `m·k` floats of first-touch page faults and an extra read+write
/// pass per training step. The packed bytes are identical either way, so the result is bit-for-bit
/// the NN kernel on the explicitly transposed A (`sgemm_tn_matches_naive_and_transposed_nn`) — the
/// same `select_kc` grouping and per-`(i, j)` order, no new accumulation order to keep in sync.
/// `beta`: 0 overwrites C, else accumulates.
///
/// # Safety
/// `a` valid for `k*m`, `b` for `k*n`, `c` for `m*n` `f32`.
/// **Aliasing: all operands disjoint** — `c` must overlap neither `a` nor `b`: `C` is written in
/// tiled order while both are still read (`a` and `b` may share storage; both are only read). A
/// `beta = 1` call reads `c` itself, in place by construction. The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, k, m);
        crate::safety::debug_extent_2d(b, k, n);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_dispatch(a, b, c, m, k, n, beta, true, false, false, None);
}

/// Multi-threaded `C = Aᵀ·B` (the `@parallel` weight-gradient GEMM). Every parallel schedule packs
/// A from its `[k, m]` storage exactly as the serial kernel does, and keeps the serial per-(i,j)
/// accumulation order, so serial == parallel == interpreter bit-for-bit (the differential gate).
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_tn`].
/// **Aliasing: all operands disjoint** — `c` must overlap neither `a` nor `b`: `C` is written in
/// tiled order while both are still read (`a` and `b` may share storage; both are only read). A
/// `beta = 1` call reads `c` itself, in place by construction. The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, k, m);
        crate::safety::debug_extent_2d(b, k, n);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_dispatch(a, b, c, m, k, n, beta, true, false, true, None);
}

/// The half-precision storage a mixed-precision GEMM entry widens from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HalfKind {
    Bf16,
    F16,
}

impl HalfKind {
    /// The scalar widen: the crate-root definition the interpreter and every other kernel share.
    #[inline]
    pub(crate) fn widen1(self, bits: u16) -> f32 {
        match self {
            HalfKind::Bf16 => crate::bf16_bits_to_f32(bits),
            HalfKind::F16 => crate::f16_bits_to_f32(bits),
        }
    }
}

// Reusable per-thread widen scratch for the bf16/f16 GEMM entries (the widened A and B operands),
// the `PACK_SCRATCH` discipline applied to the prepass (P9.19): until 2026-10 every half GEMM
// allocated and zeroed `(m·k + n·k)` fresh f32 per call — for a 768x3072 bf16 projection at batch
// 512, ~15 MB of first-touch page faults per call — although the widen overwrites every element the
// GEMM reads. A buffer is taken out for the call and put back after it (the GEMM, serial or
// parallel, has joined by then), never re-zeroed; growth allocates fresh zeroed storage instead of
// copying stale contents across.
thread_local! {
    static LOWP_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// The largest widened operand (in f32 elements, per buffer) the per-thread [`LOWP_SCRATCH`]
/// keeps between calls: 8 Mi f32 = 32 MiB covers every transformer projection (a 768x3072 operand
/// is 2.4 Mi) while a one-off huge operand — a 50257x768 LM head is 38.6 Mi — is freed after its
/// call instead of being pinned for the thread's lifetime. Retention-only: never changes a bit.
const LOWP_SCRATCH_KEEP: usize = 8 << 20;

/// Run `f` with two f32 buffers of at least `na` and `nb` elements from the calling thread's
/// [`LOWP_SCRATCH`]. Their contents on entry are unspecified — `f` must write every element it
/// reads, which the widen prepasses do — and they are returned to the cache afterwards unless
/// larger than [`LOWP_SCRATCH_KEEP`].
fn with_lowp_scratch(na: usize, nb: usize, f: impl FnOnce(*mut f32, *mut f32)) {
    let (mut sa, mut sb) = LOWP_SCRATCH.with(|s| {
        let mut g = s.borrow_mut();
        (std::mem::take(&mut g.0), std::mem::take(&mut g.1))
    });
    if sa.len() < na {
        sa = vec![0.0f32; na];
    }
    if sb.len() < nb {
        sb = vec![0.0f32; nb];
    }
    f(sa.as_mut_ptr(), sb.as_mut_ptr());
    LOWP_SCRATCH.with(|s| {
        let mut g = s.borrow_mut();
        if sa.len() <= LOWP_SCRATCH_KEEP {
            g.0 = sa;
        }
        if sb.len() <= LOWP_SCRATCH_KEEP {
            g.1 = sb;
        }
    });
}

/// Widen `n` half-width values (`bf16`/`f16` stored bits) into an `f32` buffer. The widen is
/// **lossless** (`bf16` is the top 16 bits of an f32; `f16`→f32 is an exact super-range), so the
/// widened matrix holds exactly the values the half storage represents.
///
/// Eight lanes at a time where the host allows — bf16 through `lowp::widen_bf16` (`vpmovzxwd` +
/// `vpslld 16`, the identical bit-extend the scalar `(bits as u32) << 16` performs) and f16 through
/// `lowp::widen_f16` (F16C `vcvtph2ps`, which is what the `half` crate itself dispatches to on an
/// F16C host and which its software fallback reproduces bit for bit, quieted-NaN payloads
/// included) — with the scalar [`HalfKind::widen1`] for the tail and every other host. Selected
/// once per call by `kind`, not through a per-element function pointer. Pinned over all 65,536
/// bit patterns of both formats by `lowp_widen_simd_matches_scalar_for_every_bit_pattern`.
///
/// # Safety
/// `src` valid for `n` `u16`, `dst` valid for `n` `f32`, non-overlapping.
unsafe fn widen_into_f32(src: *const u16, dst: *mut f32, n: usize, kind: HalfKind) {
    #[cfg(target_arch = "x86_64")]
    {
        match kind {
            HalfKind::Bf16 if crate::isa::avx2() => {
                widen_bf16_into_avx2(src, dst, n);
                return;
            }
            HalfKind::F16 if crate::isa::avx2() && crate::isa::f16c() => {
                widen_f16_into_f16c(src, dst, n);
                return;
            }
            _ => {}
        }
    }
    widen_into_f32_scalar(src, dst, n, kind);
}

/// The scalar widen loop: the no-SIMD path, the SIMD tails, and the twin the SIMD bodies match.
///
/// # Safety
/// As [`widen_into_f32`].
#[inline]
unsafe fn widen_into_f32_scalar(src: *const u16, dst: *mut f32, n: usize, kind: HalfKind) {
    for i in 0..n {
        *dst.add(i) = kind.widen1(*src.add(i));
    }
}

/// The 8-lane bf16 widen body of [`widen_into_f32`].
///
/// # Safety
/// As [`widen_into_f32`]; `avx2` available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn widen_bf16_into_avx2(src: *const u16, dst: *mut f32, n: usize) {
    let mut i = 0;
    while i + 8 <= n {
        std::arch::x86_64::_mm256_storeu_ps(dst.add(i), crate::lowp::widen_bf16(src.add(i)));
        i += 8;
    }
    widen_into_f32_scalar(src.add(i), dst.add(i), n - i, HalfKind::Bf16);
}

/// The 8-lane f16 widen body of [`widen_into_f32`].
///
/// # Safety
/// As [`widen_into_f32`]; `avx2` and `f16c` available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn widen_f16_into_f16c(src: *const u16, dst: *mut f32, n: usize) {
    let mut i = 0;
    while i + 8 <= n {
        std::arch::x86_64::_mm256_storeu_ps(dst.add(i), crate::lowp::widen_f16(src.add(i)));
        i += 8;
    }
    widen_into_f32_scalar(src.add(i), dst.add(i), n - i, HalfKind::F16);
}

/// `C = A·Bᵀ` (the `nn.Linear` form) with **half-precision inputs** (`bf16`/`f16`, stored as `u16`)
/// and an **f32 accumulator** — the standard mixed-precision transformer matmul. A is `[m, k]`, B is
/// `[n, k]` (its rows are the weight vectors), C is `[m, n]` f32.
///
/// This is a cheap prepass (widen A and B into reused per-thread f32 scratch — `O(m·k + n·k)`,
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
    kind: HalfKind,
    epi: Option<Epilogue>,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (mu, ku, nu) = (m as usize, k as usize, n as usize);
    if mu <= crate::gemv_half::SMALL_M {
        // Decode-shaped (a batch of one to four tokens): stream the half `B` once in its own storage
        // and widen it in registers (`gemv_half.rs`), instead of materializing an f32 copy of the
        // whole weight matrix for a product that reads each weight once. Same per-(i, j) order as
        // the GEMM below, so the bits do not change.
        with_lowp_scratch(mu * ku, 0, |af, _| {
            widen_into_f32(a, af, mu * ku, kind);
            crate::gemv_half::half_nt_small_m(af, b, c, mu, ku, nu, beta as f32, kind, epi, par);
        });
        return;
    }
    with_lowp_scratch(mu * ku, nu * ku, |af, bf| {
        widen_into_f32(a, af, mu * ku, kind);
        widen_into_f32(b, bf, nu * ku, kind);
        gemm_dispatch(af, bf, c, m, k, n, beta, false, true, par, epi);
    });
}

/// `C = A·Bᵀ` with `bf16` inputs (f32 accumulate), single-threaded. See [`gemm_lowp_nt`].
///
/// # Safety
/// `a` valid for `m*k`, `b` for `n*k` `bf16` (`u16`); `c` for `m*n` `f32`.
/// **Aliasing: all operands disjoint** — the f32 `c` must overlap neither half-precision operand
/// (`C` is written in tiled order while `a`/`b` are still read). The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_lowp_nt(a, b, c, m, k, n, beta, false, HalfKind::Bf16, None);
}

/// Multi-threaded `C = A·Bᵀ` with `bf16` inputs. The widen prepass is serial (small); the GEMM runs
/// across cores with the same per-(i,j) accumulation order as the serial kernel, so serial == parallel
/// == interpreter bit-for-bit.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_bf16_nt`].
/// **Aliasing: all operands disjoint** — the f32 `c` must overlap neither half-precision operand
/// (`C` is written in tiled order while `a`/`b` are still read). The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_lowp_nt(a, b, c, m, k, n, beta, true, HalfKind::Bf16, None);
}

/// `C = A·Bᵀ` with IEEE `f16` inputs (f32 accumulate), single-threaded. The widen is F16C-exact
/// (`half` crate); otherwise identical to the bf16 twin. See [`gemm_lowp_nt`].
///
/// # Safety
/// `a` valid for `m*k`, `b` for `n*k` `f16` (`u16`); `c` for `m*n` `f32`.
/// **Aliasing: all operands disjoint** — the f32 `c` must overlap neither half-precision operand
/// (`C` is written in tiled order while `a`/`b` are still read). The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_lowp_nt(a, b, c, m, k, n, beta, false, HalfKind::F16, None);
}

/// Multi-threaded `C = A·Bᵀ` with `f16` inputs.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_f16_nt`].
/// **Aliasing: all operands disjoint** — the f32 `c` must overlap neither half-precision operand
/// (`C` is written in tiled order while `a`/`b` are still read). The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_lowp_nt(a, b, c, m, k, n, beta, true, HalfKind::F16, None);
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
    kind: HalfKind,
) {
    let epi = Epilogue {
        bias,
        act: act as u32,
        alpha: 1.0,
    };
    gemm_lowp_nt(a, b, c, m, k, n, beta, par, kind, Some(epi));
}

/// Fused-epilogue `C = act(A·Bᵀ + bias)` with `bf16` inputs, single-threaded. See [`gemm_lowp_nt_epi`].
///
/// # Safety
/// `a`/`b` valid for `m*k`/`n*k` `bf16` (`u16`); `c` for `m*n` `f32`; `bias` null or valid for `n` `f32`.
/// When non-null, the `n`-element bias range must be disjoint from the `m*n`-element
/// output range; rows may read bias while other output elements are written.
/// **Aliasing: all operands disjoint** — `c` overlaps none of `a`, `b` and the epilogue bias (`C`
/// is written in tiled order while they are still read).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_extent_opt(bias, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), bias, n);
    }
    gemm_lowp_nt_epi(a, b, c, m, k, n, beta, bias, act, false, HalfKind::Bf16);
}

/// Multi-threaded fused-epilogue `C = act(A·Bᵀ + bias)` with `bf16` inputs.
///
/// # Safety
/// Operand-size and non-overlap contracts of [`wukong_sgemm_bf16_nt_epi`].
/// **Aliasing: all operands disjoint** — `c` overlaps none of `a`, `b` and the epilogue bias (`C`
/// is written in tiled order while they are still read).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_extent_opt(bias, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), bias, n);
    }
    gemm_lowp_nt_epi(a, b, c, m, k, n, beta, bias, act, true, HalfKind::Bf16);
}

/// Fused-epilogue `C = act(A·Bᵀ + bias)` with `f16` inputs, single-threaded. See [`gemm_lowp_nt_epi`].
///
/// # Safety
/// `a`/`b` valid for `m*k`/`n*k` `f16` (`u16`); `c` for `m*n` `f32`; `bias` null or valid for `n` `f32`.
/// When non-null, the `n`-element bias range must be disjoint from the `m*n`-element
/// output range; rows may read bias while other output elements are written.
/// **Aliasing: all operands disjoint** — `c` overlaps none of `a`, `b` and the epilogue bias (`C`
/// is written in tiled order while they are still read).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_extent_opt(bias, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), bias, n);
    }
    gemm_lowp_nt_epi(a, b, c, m, k, n, beta, bias, act, false, HalfKind::F16);
}

/// Multi-threaded fused-epilogue `C = act(A·Bᵀ + bias)` with `f16` inputs.
///
/// # Safety
/// Operand-size and non-overlap contracts of [`wukong_sgemm_f16_nt_epi`].
/// **Aliasing: all operands disjoint** — `c` overlaps none of `a`, `b` and the epilogue bias (`C`
/// is written in tiled order while they are still read).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_extent_opt(bias, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), bias, n);
    }
    gemm_lowp_nt_epi(a, b, c, m, k, n, beta, bias, act, true, HalfKind::F16);
}

/// `C = Aᵀ·B` with **half-precision inputs** (`bf16`/`f16`, stored as `u16`) and an **f32 accumulator**
/// — the mixed-precision **weight-gradient** GEMM of a training backward pass (`dW = dYᵀ·X`). A is
/// stored `[k, m]` (its logical `[m, k]` operand is the transpose of storage), B is `[k, n]`, C is
/// `[m, n]` f32.
///
/// Like the half `A·Bᵀ` kernel, this is a cheap prepass feeding the *identical* proven kernel: widen
/// A and B **in their stored layouts** (`[k, m]` and `[k, n]`, lossless, `O(k·m + k·n)`, into reused
/// per-thread scratch) and dispatch the f32 `C = Aᵀ·B` kernel, which packs the widened A straight
/// from its `[k, m]` layout (see [`wukong_sgemm_tn`]) — no transpose pass and no `[m, k]` copy.
/// Because each conversion is exact, the result is bit-for-bit the f32 TN GEMM on the widened values
/// — exactly the differential contract the interpreter marshals (no new accumulation order).
/// gcc/rustc lose twice on the naive half nest: the inline `bf16→f32` widen won't vectorize *and*
/// A's column-strided reads (the contraction is A's outer index) defeat vectorization.
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
    kind: HalfKind,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (mu, ku, nu) = (m as usize, k as usize, n as usize);
    with_lowp_scratch(ku * mu, ku * nu, |af, bf| {
        // af[p*m + i] == widen(a[p*m + i]): A keeps its stored [k, m] layout and the GEMM packs it
        // as the transposed operand, the same bytes the old widen-then-transpose copy produced.
        widen_into_f32(a, af, ku * mu, kind);
        widen_into_f32(b, bf, ku * nu, kind);
        gemm_dispatch(af, bf, c, m, k, n, beta, true, false, par, None);
    });
}

/// `C = Aᵀ·B` with `bf16` inputs (f32 accumulate), single-threaded. See [`gemm_lowp_tn`].
///
/// # Safety
/// `a` valid for `k*m`, `b` for `k*n` `bf16` (`u16`); `c` for `m*n` `f32`.
/// **Aliasing: all operands disjoint** — the f32 `c` must overlap neither half-precision operand
/// (`C` is written in tiled order while `a`/`b` are still read). The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, k, m);
        crate::safety::debug_extent_2d(b, k, n);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_lowp_tn(a, b, c, m, k, n, beta, false, HalfKind::Bf16);
}

/// Multi-threaded `C = Aᵀ·B` with `bf16` inputs. The widen prepass is serial (small); the
/// GEMM runs across cores with the same per-(i,j) order as the serial kernel, so serial == parallel
/// == interpreter bit-for-bit.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_bf16_tn`].
/// **Aliasing: all operands disjoint** — the f32 `c` must overlap neither half-precision operand
/// (`C` is written in tiled order while `a`/`b` are still read). The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, k, m);
        crate::safety::debug_extent_2d(b, k, n);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_lowp_tn(a, b, c, m, k, n, beta, true, HalfKind::Bf16);
}

/// `C = Aᵀ·B` with IEEE `f16` inputs (f32 accumulate), single-threaded. The widen is F16C-exact;
/// otherwise identical to the bf16 twin. See [`gemm_lowp_tn`].
///
/// # Safety
/// `a` valid for `k*m`, `b` for `k*n` `f16` (`u16`); `c` for `m*n` `f32`.
/// **Aliasing: all operands disjoint** — the f32 `c` must overlap neither half-precision operand
/// (`C` is written in tiled order while `a`/`b` are still read). The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, k, m);
        crate::safety::debug_extent_2d(b, k, n);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_lowp_tn(a, b, c, m, k, n, beta, false, HalfKind::F16);
}

/// Multi-threaded `C = Aᵀ·B` with `f16` inputs.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemm_f16_tn`].
/// **Aliasing: all operands disjoint** — the f32 `c` must overlap neither half-precision operand
/// (`C` is written in tiled order while `a`/`b` are still read). The compiler versions a recognized
/// call on a run-time overlap test (`wukong_mir_build`'s `kernel_guard.rs`).
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && k > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, k, m);
        crate::safety::debug_extent_2d(b, k, n);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    gemm_lowp_tn(a, b, c, m, k, n, beta, true, HalfKind::F16);
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
    at: bool,
    bt: bool,
    epi: Option<Epilogue>,
) {
    sgemm_avx2_parallel_nc(a, b, c, m, k, n, beta, at, bt, epi, gemm_blocking().nc);
}

/// [`sgemm_avx2_parallel`] with the column block `nc_blk` explicit (a multiple of [`NR`]), so its
/// bit-invariance under any `nc` is testable regardless of the process env.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_avx2_parallel_nc(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    beta: f32,
    at: bool,
    bt: bool,
    epi: Option<Epilogue>,
    nc_blk: usize,
) {
    use rayon::prelude::*;
    // Pack scratch allocated ONCE and reused across every cache block, instead of a fresh
    // `vec![0.0; …]` (malloc + zero of ~2 MB) per K-block per call — pure overhead the old code paid
    // on every block. Each block fully repacks its region (the pack writes real data *and* the
    // edge-padding zeros for every element it owns), so reuse needs no re-zeroing. Sized to the
    // largest block: full M rows × KC, and an NC-wide × KC B panel.
    let ap_cap = round_up(m, MR) * k.min(KC);
    let bp_cap = round_up(n.min(nc_blk), NR) * k.min(KC);
    let mut ap = vec![0.0f32; ap_cap];
    let mut bp = vec![0.0f32; bp_cap];
    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(nc_blk);
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
            let (a_blk, lda) = a_origin(a, m, k, at, 0, pc);
            pack_ab_par(
                a_blk,
                lda,
                at,
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
                            PfWalk::NONE,
                        );
                    }
                }
            });
            pc += kc;
        }
        jc += nc_blk;
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
    counters: Vec<(
        std::sync::atomic::AtomicUsize,
        std::sync::atomic::AtomicUsize,
    )>,
    /// In-region phase barrier, sized to the exact broadcast width (asserted before first use).
    barrier: std::sync::Barrier,
    nworkers: usize,
    /// The column block every worker walks ([`Blocking::nc`]).
    nc_blk: usize,
}
// SAFETY: shared across the broadcast workers by reference. Every mutable target is written by
// exactly one worker per phase — packed panels and C row-panels are claimed via `fetch_add` (unique
// indices) and are disjoint — and the barrier orders all pack writes before any compute read. This
// is the same aliasing discipline the fork-join kernel has always used; it smuggled the very same
// pointers across its rayon closures as `usize`.
#[cfg(target_arch = "x86_64")]
// SAFETY: as documented above -- each mutable target is written by exactly one worker per phase (fetch_add-claimed disjoint indices) and the barrier orders packs before compute reads.
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
    sgemm_persistent_region_nc(pool, args, gemm_blocking().nc);
}

/// [`sgemm_persistent_region`] with the column block `nc_blk` explicit (a multiple of [`NR`]).
///
/// # Safety
/// That of [`sgemm_persistent_region`].
#[cfg(target_arch = "x86_64")]
unsafe fn sgemm_persistent_region_nc(
    pool: Option<&rayon::ThreadPool>,
    args: GemmArgs,
    nc_blk: usize,
) {
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
    let bp_cap = round_up(n.min(nc_blk), NR) * k.min(KC);
    let mut ap = vec![0.0f32; ap_cap];
    let mut bp = vec![0.0f32; bp_cap];
    // The jc/pc schedule is a pure function of (m, k, n, nc): every worker walks it identically, so
    // the block count — and each block's fresh counter pair — is known up front.
    let nblocks = n.div_ceil(nc_blk) * k.div_ceil(select_kc(k));
    let region = GemmRegion {
        args,
        ap: ap.as_mut_ptr(),
        bp: bp.as_mut_ptr(),
        counters: (0..nblocks)
            .map(|_| (AtomicUsize::new(0), AtomicUsize::new(0)))
            .collect(),
        barrier: std::sync::Barrier::new(nworkers),
        nworkers,
        nc_blk,
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
    let (beta, at, bt, epi) = (rg.args.beta, rg.args.at, rg.args.bt, rg.args.epi);
    let (ap, bp) = (rg.ap, rg.bp);
    let mut block = 0usize;
    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(rg.nc_blk);
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
            let (a_blk, lda) = a_origin(a, m, k, at, 0, pc);
            if total_bytes < pack_par_min_bytes() {
                if widx == 0 {
                    pack_a(a_blk, lda, at, m, kc, ap);
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
                        pack_a_panel(a_blk, lda, at, m, kc, t, ap.add(t * kc * MR));
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
                        PfWalk::NONE,
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
        jc += rg.nc_blk;
    }
}

// --- 2D block-parallel path (the DEFAULT; `WUKONG_GEMM_2D=0` opts out) --------------------------

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
/// * **Skinny M** (`m ≤ SKINNY_M_MAX = 144`, a policy constant deliberately separate from the
///   `MC` cache block): one row of blocks (`BM = round_up(m, MR)`) — a second row
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
    const CANDIDATES: &[(usize, usize)] = &[
        (192, 256),
        (144, 192),
        (96, 128),
        (96, 96),
        (48, 96),
        (24, 48),
    ];
    let target = target_blocks.max(1);
    // Equalize block sizes along one dim without changing its block count: `nb` is preserved
    // (round_up(ceil(d/nb), unit) ≤ bd since bd is a unit multiple with nb·bd ≥ d, and
    // nb·round_up(ceil(d/nb), unit) ≥ d, so ceil(d/·) lands back on nb) — the remainder is spread
    // across every block instead of one thin straggler that pays a whole task dispatch (and, under
    // dynamic scheduling, a whole claim) for a sliver of work.
    let rebalance = |d: usize, bd: usize, unit: usize| round_up(d.div_ceil(d.div_ceil(bd)), unit);
    if m <= SKINNY_M_MAX {
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
    // Every block is one row-block sweep, so its B slice is packed a micropanel at a time unless
    // the fused pack is switched off (`gemm_fused_pack`); read once per call, not per block.
    let fused = gemm_fused_pack();
    // Per-worker scratch caps: one block's A/B slices (not whole-matrix panels), rounded up to
    // full micropanels exactly as the other kernels size their scratch — the B half shrinks to one
    // `kc×NR` micropanel under the fused pack.
    let ap_cap = round_up(bm.min(m), MR) * k.min(KC);
    let bp_cap = if fused {
        NR * k.min(KC)
    } else {
        round_up(bn.min(n), NR) * k.min(KC)
    };
    // Send-safe scalar captures — the same raw-pointer-as-usize idiom `sgemm_avx2_parallel` and
    // `pack_ab_par` already use (the closure outlives nothing: `install`/`for_each` join before
    // this frame returns).
    let (a_addr, b_addr, c_addr) = (args.a as usize, args.b as usize, args.c as usize);
    let (beta, at, bt) = (args.beta, args.at, args.bt);
    let (epi_on, epi_bias_addr, epi_act, epi_alpha) = match args.epi {
        Some(e) => (true, e.bias as usize, e.act, e.alpha),
        None => (false, 0usize, 0u32, 1.0f32),
    };
    let body = move || {
        (0..nbi * nbj).into_par_iter().for_each(|t| {
            let (bi, bj) = (t / nbj, t % nbj);
            let (i0, j0) = (bi * bm, bj * bn);
            // A null bias-addr (0) stays null; the block body shifts to its global column origin.
            let epi = epi_on.then_some(Epilogue {
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
                    m,
                    k,
                    n,
                    beta,
                    at,
                    bt,
                    epi,
                    i0,
                    j0,
                    (m - i0).min(bm),
                    (n - j0).min(bn),
                    ap.as_mut_ptr(),
                    bp.as_mut_ptr(),
                    fused,
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
/// so per-(i,j) accumulation is bit-identical (see [`sgemm_2d_blocks`]). A block is always one
/// row-block sweep, so with `fused` its B slice is packed a micropanel at a time
/// ([`macro_kernel_packb`]) — the same bytes and the same micro-tile calls.
///
/// # Safety
/// `a`/`b` valid per the [`wukong_sgemm`]/[`wukong_sgemm_nt`]/[`wukong_sgemm_tn`] contract (`at`,
/// `bt` naming the stored layouts) with `m`/`k`/`n` the full matrix dims; `[i0, i0+bm) × [j0, j0+bn)`
/// in range of C; `ap`/`bp` sized for
/// `round_up(bm,MR)·min(k,KC)` / `round_up(bn,NR)·min(k,KC)` f32 (`bp`: `NR·min(k,KC)` when
/// `fused`); avx2+fma available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_2d_block(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    beta: f32,
    at: bool,
    bt: bool,
    epi: Option<Epilogue>,
    i0: usize,
    j0: usize,
    bm: usize,
    bn: usize,
    ap: *mut f32,
    bp: *mut f32,
    fused: bool,
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
        // [i0, i0+bm) × cols [pc, pc+kc) of the logical A (from its stored layout, `a_origin`),
        // `pack_b_block` reads the (pc, j0) slice of
        // B honoring `bt` — both contracts take an arbitrary base/leading-dim, and bm/bn smaller
        // than m/n only shortens their row/col loops. Same source values, same panel layout, same
        // matrix-edge-only zero padding ⇒ byte-identical micropanels.
        let (a_blk, lda) = a_origin(a, m, k, at, i0, pc);
        pack_a(a_blk, lda, at, bm, kc, ap);
        if fused {
            let (b_src, ldb) = b_origin(b, k, n, bt, pc, j0);
            macro_kernel_packb(
                bm,
                bn,
                kc,
                ap,
                b_src,
                ldb,
                bt,
                bp,
                c.add(i0 * n + j0),
                n,
                beta_eff,
                block_epi,
            );
        } else {
            pack_b_block(b, k, n, pc, j0, kc, bn, bt, bp);
            macro_kernel(
                bm,
                bn,
                kc,
                ap,
                bp,
                c.add(i0 * n + j0),
                n,
                beta_eff,
                block_epi,
            );
        }
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
/// `[0, m) × [j0, j0+bn)` in range of C; `bp` sized `round_up(bn,NR)·min(k,KC)` f32
/// (`NR·min(k,KC)` when `fused`, which packs B a micropanel at a time as [`sgemm_2d_block`]
/// does); avx2+fma.
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
    fused: bool,
) {
    for &(pc, kc, off) in chunks {
        let beta_eff = if pc == 0 { beta } else { 1.0 };
        let block_epi = if pc + kc == k {
            epi.map(|e| e.shift(j0))
        } else {
            None
        };
        if fused {
            let (b_src, ldb) = b_origin(b, k, n, bt, pc, j0);
            macro_kernel_packb(
                m,
                bn,
                kc,
                ap_shared.add(off),
                b_src,
                ldb,
                bt,
                bp,
                c.add(j0),
                n,
                beta_eff,
                block_epi,
            );
        } else {
            pack_b_block(b, k, n, pc, j0, kc, bn, bt, bp);
            macro_kernel(
                m,
                bn,
                kc,
                ap_shared.add(off),
                bp,
                c.add(j0),
                n,
                beta_eff,
                block_epi,
            );
        }
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
///
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
    // Same fused-pack choice as the static path (read once per call); the B scratch is one
    // `kc×NR` micropanel under it.
    let fused = gemm_fused_pack();
    let ap_cap = round_up(bm.min(m), MR) * k.min(KC);
    let bp_cap = if fused {
        NR * k.min(KC)
    } else {
        round_up(bn.min(n), NR) * k.min(KC)
    };
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
    let (beta, at, bt) = (args.beta, args.at, args.bt);
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
                        let (a_blk, lda) = a_origin(a_addr as *const f32, m, k, at, 0, pc);
                        pack_a(a_blk, lda, at, m, kc, (ap_shared_addr as *mut f32).add(off));
                    }
                }
            } else {
                let chunks_ref = &chunks;
                crate::par::for_each_unit(chunks_ref.len() * mpanels, |u| {
                    let (ci, ip) = (u / mpanels, u % mpanels);
                    let (pc, kc, off) = chunks_ref[ci];
                    // SAFETY: each (chunk, panel) granule owns a disjoint `MR·kc` destination
                    // slice; A is read-only; buffer sized `a_rows·k` above.
                    unsafe {
                        let (a_blk, lda) = a_origin(a_addr as *const f32, m, k, at, 0, pc);
                        pack_a_panel(
                            a_blk,
                            lda,
                            at,
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
            let epi = epi_on.then_some(Epilogue {
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
                            fused,
                        );
                    } else {
                        sgemm_2d_block(
                            a_addr as *const f32,
                            b_addr as *const f32,
                            c_addr as *mut f32,
                            m,
                            k,
                            n,
                            beta,
                            at,
                            bt,
                            epi,
                            i0,
                            j0,
                            (m - i0).min(bm),
                            (n - j0).min(bn),
                            ap.as_mut_ptr(),
                            bp.as_mut_ptr(),
                            fused,
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
    let (beta, at, bt) = (args.beta, args.at, args.bt);
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
            // SAFETY: `a_addr`/`b_addr` are valid for the m*k / k*n operands; a_origin/pack_ab_par touch only the current K-block [pc, pc+kc).
            unsafe {
                let (a_blk, lda) = a_origin(a_addr as *const f32, m, k, at, 0, pc);
                pack_ab_par(
                    a_blk,
                    lda,
                    at,
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

/// The portable **bit-exact twin** of [`sgemm_avx2`] (`C = A·B`; `Aᵀ` read from `[k, m]` storage
/// when `at`, `Bᵀ` from `[n, k]` when `bt`) — the
/// no-AVX fallback, and the ONLY GEMM path on a target without AVX2+FMA, since every non-`x86_64`
/// build compiles the fast path out entirely.
///
/// **It was not a twin until 2026-09-13.** It summed the whole K into one non-fused accumulator with
/// separate `*` and `+=` (Rust never contracts), while [`sgemm_avx2`] splits K by [`select_kc`] and
/// uses `vfmadd`; the doc comment here recorded the divergence in a parenthetical ("the two agree
/// within f32 tolerance in tests") and no test in the crate ever ran the two paths on the same
/// inputs. That made the single most-used kernel in the project the one exception to the crate's
/// invariant #1 — a machine without AVX2, or any `aarch64` build, computed a *different* matmul from
/// the same `.wk` source — and because the interpreter and the native backend both go through the
/// one dispatcher, no differential or invariance gate in the repo could observe it. It now mirrors
/// the fast path step for step:
///
///   * the same K partition, `kc = (k − pc).min(select_kc(k))`, accumulating each block from a
///     fresh `+0.0` (the microkernel's `_mm256_setzero_ps()`);
///   * one rounding per K step: [`f32::mul_add`] is the portable spelling of `vfmadd` (`fmadd` on
///     aarch64; a correctly-rounded `fmaf` call on an x86-64 build without the `fma` target
///     feature, measured bit-identical to `_mm256_fmadd_ps` over 2M random accumulator chains on
///     this host);
///   * the same writeback rule, `beta_eff = if pc == 0 { beta } else { 1.0 }`, in which any
///     non-zero `beta` means *accumulate* (it is a flag, never a scale);
///   * the epilogue applied exactly once, on the final K-block, at the global column `j`.
///
/// Pinned by `sgemm_scalar_matches_avx2_bit_for_bit` (`to_bits()` equality across
/// shape × `bt` × `beta` × epilogue, including two- and three-K-block shapes).
///
/// **The cost, stated plainly, because it is not zero for the population this path exists for.**
/// [`f32::mul_add`] lowers to a `vfmadd`/`fmadd` instruction only where the target feature is on.
/// This workspace has no `.cargo/config.toml`, no `build.rs` and no `+fma` in `RUSTFLAGS`, so on a
/// stock x86-64 build it lowers to a **`fmaf` CALL into `libm`, one per K step**, where the
/// pre-2026-09-13 body issued two inline FP ops (`mulss` + `addss`) per MAC. On any AVX2 host that
/// is free — this function is unreachable outside its own tests (the runtime probe in
/// [`gemm_dispatch`] takes the other branch) — and on `aarch64` it is free as well (`fmadd`). It
/// is NOT free on a non-AVX2 x86-64 host, which is exactly the host this function is the only GEMM
/// on. The trade is taken deliberately: a machine with no FMA unit has no faster
/// correctly-rounded option, and a portable GEMM that computes a different answer from the one
/// every gate in the repo checks is worse than a slow one. A build that wants both can set
/// `-C target-feature=+fma`, which turns the call back into one instruction with identical bits.
#[allow(clippy::too_many_arguments)]
fn sgemm_scalar(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    beta: f32,
    at: bool,
    bt: bool,
    epi: Option<Epilogue>,
) {
    // SAFETY: `a`/`b`/`c` satisfy the m,k,n operand-size contract; `c.add(i*n + j)` indexes within c's m*n elements.
    unsafe {
        for i in 0..m {
            let crow = c.add(i * n);
            for j in 0..n {
                let cij = crow.add(j);
                let mut pc = 0;
                while pc < k {
                    let kc = (k - pc).min(select_kc(k));
                    let mut acc = 0.0f32;
                    for p in pc..pc + kc {
                        let aip = if at {
                            *a.add(p * m + i)
                        } else {
                            *a.add(i * k + p)
                        };
                        let bpj = if bt {
                            *b.add(j * k + p)
                        } else {
                            *b.add(p * n + j)
                        };
                        acc = aip.mul_add(bpj, acc);
                    }
                    // First K-block honors the caller's beta; later blocks accumulate the partials.
                    let beta_eff = if pc == 0 { beta } else { 1.0 };
                    let val = if beta_eff == 0.0 { acc } else { *cij + acc };
                    // The epilogue applies only once the K reduction is complete (`j` is global).
                    *cij = match (pc + kc == k, epi) {
                        (true, Some(e)) => e.apply(val, j),
                        _ => val,
                    };
                    pc += kc;
                }
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
    at: bool,
    bt: bool,
    epi: Option<Epilogue>,
) {
    sgemm_avx2_impl(
        a,
        b,
        c,
        m,
        k,
        n,
        beta,
        at,
        bt,
        epi,
        gemm_fused_pack(),
        gemm_blocking(),
    );
}

/// [`sgemm_avx2`] with the [`gemm_fused_pack`] choice and the [`Blocking`] explicit, so the
/// bit-identity of the two B packing schedules and of every `mc`/`nc` is testable regardless of the
/// process env.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_avx2_impl(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    beta: f32,
    at: bool,
    bt: bool,
    epi: Option<Epilogue>,
    fused: bool,
    blk: Blocking,
) {
    // Pack scratch, sized to the actual blocks needed (never larger than the cache-block caps).
    // Reused across calls via a per-thread buffer (see PACK_SCRATCH): the packers overwrite every
    // element they read back (data + edge padding), so no re-zeroing is needed, and we avoid a
    // per-call malloc+zero that was a measurable slice of small-GEMM time. Taken out here and put
    // back at the end; the body between is panic-free so the buffers are always returned.
    let kc_max = k.min(KC);
    let mc_max = m.min(blk.mc);
    // Column-block width: `blk.nc` except in the single-row-block regime, where the packed B panel
    // has no ic-loop reuse and must stay in L2 (see `select_nc`).
    let nc_step = select_nc_with(m, k, n, blk);
    let nc_max = n.min(nc_step);
    // One `ic` block (`m <= mc`): every packed B byte would be read by exactly one row sweep, so
    // pack it one micropanel at a time instead (see `gemm_fused_pack`); the B scratch is then a
    // single `kc×NR` micropanel.
    let fuse_b = fused && m <= blk.mc;
    let ap_cap = round_up(mc_max, MR) * kc_max;
    let bp_cap = if fuse_b {
        NR * kc_max
    } else {
        round_up(nc_max, NR) * kc_max
    };
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
        let nc = (n - jc).min(nc_step);
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
            if fuse_b {
                // The single row block (ic = 0, mc = m): the same A pack, then the micropanel-at-a-
                // time B pack interleaved with the sweeps that consume it.
                let (a_blk, lda) = a_origin(a, m, k, at, 0, pc);
                pack_a(a_blk, lda, at, m, kc, ap.as_mut_ptr());
                let (b_src, ldb) = b_origin(b, k, n, bt, pc, jc);
                macro_kernel_packb(
                    m,
                    nc,
                    kc,
                    ap.as_ptr(),
                    b_src,
                    ldb,
                    bt,
                    bp.as_mut_ptr(),
                    c.add(jc),
                    n,
                    beta_eff,
                    block_epi,
                );
                pc += kc;
                continue;
            }
            pack_b_block(b, k, n, pc, jc, kc, nc, bt, bp.as_mut_ptr());
            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(blk.mc);
                let (a_blk, lda) = a_origin(a, m, k, at, ic, pc);
                pack_a(a_blk, lda, at, mc, kc, ap.as_mut_ptr());
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
                ic += blk.mc;
            }
            pc += kc;
        }
        jc += nc_step;
    }

    // Return the (now larger) buffers to the per-thread cache for the next call to reuse.
    PACK_SCRATCH.with(|c| {
        let mut g = c.borrow_mut();
        g.0 = ap;
        g.1 = bp;
    });
}

/// The stored location of the logical `A[i0, p0]` and the stored leading dimension: A is row-major
/// `[m, k]` (`a[i*k + p]`) unless `at`, in which case it is stored transposed, `[k, m]`
/// (`a[p*m + i]`) — the weight-gradient `C = Aᵀ·B` operand, which [`pack_a_panel`] packs straight
/// from that storage rather than from a transposed scratch copy (P9.19). Every A pack site goes
/// through this one function so the two layouts cannot drift apart between the schedules.
///
/// # Safety
/// `(i0, p0)` must address an element of A (`i0 < m`, `p0 < k`).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn a_origin(
    a: *const f32,
    m: usize,
    k: usize,
    at: bool,
    i0: usize,
    p0: usize,
) -> (*const f32, usize) {
    if at {
        (a.add(p0 * m + i0), m)
    } else {
        (a.add(i0 * k + p0), k)
    }
}

/// The stored location of the B slice whose logical origin is contraction step `pc`, column `jc`,
/// and the stored leading dimension: B is `[n, k]` when `bt` (column `j` of `Bᵀ` is row `j` of B),
/// `[k, n]` otherwise — exactly the base `pack_b_block` hands its whole-panel packers, so a
/// micropanel packed from this origin holds the same bytes as that panel's slice.
///
/// # Safety
/// `(pc, jc)` must address an element of B (`pc < k`, `jc < n`).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn b_origin(
    b: *const f32,
    k: usize,
    n: usize,
    bt: bool,
    pc: usize,
    jc: usize,
) -> (*const f32, usize) {
    if bt {
        (b.add(jc * k + pc), k)
    } else {
        (b.add(pc * n + jc), n)
    }
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
///
/// The bulk is done eight contraction steps at a time by [`transpose8x8_into`], the standard AVX2
/// in-register 8x8 transpose (BLIS's and oneDNN's packing kernels are the same shape). The scalar
/// order below it is retained verbatim as the **twin**: it defines the packed bytes, the vector
/// path must reproduce them exactly, and `pack_b_trans_panel_simd_matches_scalar_twin` asserts
/// that bit-for-bit — including on NaN and signed zero, which a pure data movement must carry
/// through unchanged (no comparison, no arithmetic, so no `_mm256_max_ps`-style tie hazard here).
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
    // Vector path: full NR-wide panels only, eight contraction steps per iteration. Two 8x8
    // transposes fill one 8-step slab of the [kc][NR] panel — rows j0..j0+8 into packed lanes
    // 0..8, rows j0+8..j0+16 into lanes 8..16. This replaces 128 four-byte stores at a 64-byte
    // stride (which is what the scalar twin below issues, one store per element, never
    // coalescable) with 16 thirty-two-byte stores, plus 16 loads and 48 shuffles.
    let mut p0 = 0;
    if ncols == NR {
        while p0 + 8 <= kc {
            transpose8x8_into(b.add(j0 * ldb + p0), ldb, panel.add(p0 * NR));
            transpose8x8_into(b.add((j0 + 8) * ldb + p0), ldb, panel.add(p0 * NR + 8));
            p0 += 8;
        }
    }
    // Scalar twin: the contraction tail (`kc % 8`), and every partial (< NR) column panel.
    for r in 0..ncols {
        let src = b.add((j0 + r) * ldb); // row (j0+r) of B, contiguous over the contraction
        for p in p0..kc {
            *panel.add(p * NR + r) = *src.add(p);
        }
    }
    for r in ncols..NR {
        for p in 0..kc {
            *panel.add(p * NR + r) = 0.0;
        }
    }
}

/// Transpose an 8x8 f32 block: read eight rows of eight contraction-contiguous floats at
/// `src + r*ldb`, write eight packed rows of eight floats at `dst + p*NR`, so that
/// `dst[p*NR + r] == src[r*ldb + p]`.
///
/// The classic AVX2 sequence: 8 `unpack` (pair rows within each 128-bit half), 8 `shuffle`
/// (quad them), 8 `permute2f128` (swap the halves) — 24 shuffle-port ops for 64 elements. It is
/// pure lane movement: every output bit is an input bit, so NaN payloads and signed zeros are
/// carried through unchanged and the result is byte-identical to the element-at-a-time twin.
///
/// # Safety
/// `avx2` (caller has the `target_feature`). `src` must be readable for eight rows of eight f32
/// at stride `ldb`; `dst` writable for eight rows of eight f32 at stride `NR`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose8x8_into(src: *const f32, ldb: usize, dst: *mut f32) {
    use std::arch::x86_64::*;
    let r0 = _mm256_loadu_ps(src);
    let r1 = _mm256_loadu_ps(src.add(ldb));
    let r2 = _mm256_loadu_ps(src.add(2 * ldb));
    let r3 = _mm256_loadu_ps(src.add(3 * ldb));
    let r4 = _mm256_loadu_ps(src.add(4 * ldb));
    let r5 = _mm256_loadu_ps(src.add(5 * ldb));
    let r6 = _mm256_loadu_ps(src.add(6 * ldb));
    let r7 = _mm256_loadu_ps(src.add(7 * ldb));
    // Pair adjacent rows inside each 128-bit half.
    let t0 = _mm256_unpacklo_ps(r0, r1);
    let t1 = _mm256_unpackhi_ps(r0, r1);
    let t2 = _mm256_unpacklo_ps(r2, r3);
    let t3 = _mm256_unpackhi_ps(r2, r3);
    let t4 = _mm256_unpacklo_ps(r4, r5);
    let t5 = _mm256_unpackhi_ps(r4, r5);
    let t6 = _mm256_unpacklo_ps(r6, r7);
    let t7 = _mm256_unpackhi_ps(r6, r7);
    // Quad them: 0x44 takes lanes (0,1) of each source, 0xEE takes lanes (2,3).
    let u0 = _mm256_shuffle_ps::<0x44>(t0, t2);
    let u1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
    let u2 = _mm256_shuffle_ps::<0x44>(t1, t3);
    let u3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
    let u4 = _mm256_shuffle_ps::<0x44>(t4, t6);
    let u5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
    let u6 = _mm256_shuffle_ps::<0x44>(t5, t7);
    let u7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
    // Cross the 128-bit halves: 0x20 = (A_lo, B_lo) -> contraction steps 0..4,
    // 0x31 = (A_hi, B_hi) -> steps 4..8.
    _mm256_storeu_ps(dst, _mm256_permute2f128_ps::<0x20>(u0, u4));
    _mm256_storeu_ps(dst.add(NR), _mm256_permute2f128_ps::<0x20>(u1, u5));
    _mm256_storeu_ps(dst.add(2 * NR), _mm256_permute2f128_ps::<0x20>(u2, u6));
    _mm256_storeu_ps(dst.add(3 * NR), _mm256_permute2f128_ps::<0x20>(u3, u7));
    _mm256_storeu_ps(dst.add(4 * NR), _mm256_permute2f128_ps::<0x31>(u0, u4));
    _mm256_storeu_ps(dst.add(5 * NR), _mm256_permute2f128_ps::<0x31>(u1, u5));
    _mm256_storeu_ps(dst.add(6 * NR), _mm256_permute2f128_ps::<0x31>(u2, u6));
    _mm256_storeu_ps(dst.add(7 * NR), _mm256_permute2f128_ps::<0x31>(u3, u7));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_b_trans(b: *const f32, ldb: usize, kc: usize, nc: usize, bp: *mut f32) {
    let npanels = nc.div_ceil(NR);
    for jp in 0..npanels {
        pack_b_trans_panel(b, ldb, kc, nc, jp, bp.add(jp * kc * NR));
    }
}

/// Transpose a 6x8 f32 block — six A rows of eight contraction-contiguous floats at `src + r*lda`
/// — into eight packed `MR`-float rows at `dst + p*MR`, so that `dst[p*MR + r] == src[r*lda + p]`.
///
/// The [`transpose8x8_into`] network with rows 6 and 7 fixed at zero: output vector `p` holds
/// `[r0[p] .. r5[p], 0, 0]`, and it is stored as eight floats at `dst + p*MR`, so its two zero lanes
/// land on `dst[(p+1)*MR]`, `dst[(p+1)*MR + 1]`. The stores run in ascending `p`, each overwriting
/// the previous one's overhang, which leaves exactly two floats written past the slab, at the
/// start of contraction row `p + 8` — the caller ([`pack_a_panel`]) guarantees that row exists and
/// is rewritten later. Pure lane movement: every packed bit is an input bit.
///
/// # Safety
/// `avx2`; `src` readable for six rows of eight f32 at stride `lda`; `dst` writable for
/// `8*MR + 2` f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose6x8_into(src: *const f32, lda: usize, dst: *mut f32) {
    use std::arch::x86_64::*;
    let z = _mm256_setzero_ps();
    let r0 = _mm256_loadu_ps(src);
    let r1 = _mm256_loadu_ps(src.add(lda));
    let r2 = _mm256_loadu_ps(src.add(2 * lda));
    let r3 = _mm256_loadu_ps(src.add(3 * lda));
    let r4 = _mm256_loadu_ps(src.add(4 * lda));
    let r5 = _mm256_loadu_ps(src.add(5 * lda));
    let t0 = _mm256_unpacklo_ps(r0, r1);
    let t1 = _mm256_unpackhi_ps(r0, r1);
    let t2 = _mm256_unpacklo_ps(r2, r3);
    let t3 = _mm256_unpackhi_ps(r2, r3);
    let t4 = _mm256_unpacklo_ps(r4, r5);
    let t5 = _mm256_unpackhi_ps(r4, r5);
    // The 8x8 network's t6/t7 are unpacks of the two zero rows, i.e. zero.
    let u0 = _mm256_shuffle_ps::<0x44>(t0, t2);
    let u1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
    let u2 = _mm256_shuffle_ps::<0x44>(t1, t3);
    let u3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
    let u4 = _mm256_shuffle_ps::<0x44>(t4, z);
    let u5 = _mm256_shuffle_ps::<0xEE>(t4, z);
    let u6 = _mm256_shuffle_ps::<0x44>(t5, z);
    let u7 = _mm256_shuffle_ps::<0xEE>(t5, z);
    // Ascending `p`: each store's two-lane overhang is overwritten by the next.
    _mm256_storeu_ps(dst, _mm256_permute2f128_ps::<0x20>(u0, u4));
    _mm256_storeu_ps(dst.add(MR), _mm256_permute2f128_ps::<0x20>(u1, u5));
    _mm256_storeu_ps(dst.add(2 * MR), _mm256_permute2f128_ps::<0x20>(u2, u6));
    _mm256_storeu_ps(dst.add(3 * MR), _mm256_permute2f128_ps::<0x20>(u3, u7));
    _mm256_storeu_ps(dst.add(4 * MR), _mm256_permute2f128_ps::<0x31>(u0, u4));
    _mm256_storeu_ps(dst.add(5 * MR), _mm256_permute2f128_ps::<0x31>(u1, u5));
    _mm256_storeu_ps(dst.add(6 * MR), _mm256_permute2f128_ps::<0x31>(u2, u6));
    _mm256_storeu_ps(dst.add(7 * MR), _mm256_permute2f128_ps::<0x31>(u3, u7));
}

/// Pack an `MC×KC` slice of A (row-major, leading dim `lda`) into `MR`-wide row panels: panel `ip`
/// is `[kc][MR]` contiguous (so the microkernel broadcasts each row with unit stride), zero-padded.
/// Reads each A row *contiguously* over the contraction (`p` innermost) and scatters into the small
/// L1-resident panel; the transposed order (strided A reads, one cache line per element) thrashes
/// cache, so this order matters as much as it does for `pack_b_trans`.
///
/// Full panels go eight contraction steps at a time through [`transpose6x8_into`] (the
/// `pack_b_trans_panel` 8x8 network with two zero rows), replacing 48 four-byte stores at a
/// 24-byte stride per slab with eight 32-byte stores. Its stores overhang each slab by two floats
/// into the next contraction row, so the vector loop stops while a following row still exists to
/// absorb them; that row and the `kc % 8` tail, plus every partial (`< MR`-row) panel, take the
/// scalar order below, which is retained verbatim as the **twin** that defines the packed bytes
/// (`pack_a_panel_simd_matches_scalar_twin` asserts them bit-for-bit, NaN payloads and signed
/// zeros included — pure data movement, so no comparison or arithmetic is involved).
///
/// With `at` the slice is stored transposed (`a` is the [`a_origin`] of the block, `lda == m`) and
/// the panel comes from [`pack_a_t_panel`] — same bytes, no transpose needed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_a_panel(
    a: *const f32,
    lda: usize,
    at: bool,
    mc: usize,
    kc: usize,
    ip: usize,
    panel: *mut f32,
) {
    if at {
        pack_a_t_panel(a, lda, mc, kc, ip, panel);
        return;
    }
    let i0 = ip * MR;
    let nrows = (mc - i0).min(MR);
    let mut p0 = 0;
    if nrows == MR {
        // `p0 + 8 < kc`, not `<=`: row `p0 + 8` must exist to take the two-float overhang, and it
        // is always rewritten afterwards (by the next slab, or by the scalar tail below, which
        // writes all MR lanes of every row from the final `p0` on).
        while p0 + 8 < kc {
            transpose6x8_into(a.add(i0 * lda + p0), lda, panel.add(p0 * MR));
            p0 += 8;
        }
    }
    for r in 0..nrows {
        let src = a.add((i0 + r) * lda); // row (i0+r) of A, contiguous over the contraction
        for p in p0..kc {
            *panel.add(p * MR + r) = *src.add(p);
        }
    }
    for r in nrows..MR {
        for p in 0..kc {
            *panel.add(p * MR + r) = 0.0;
        }
    }
}

/// [`pack_a_panel`] for an A slice stored transposed (`at`): `a` is the stored element of logical
/// `A[block row 0, pc]` and `lda == m`, so contraction step `p` of logical rows `i0 .. i0 + MR` is
/// the CONTIGUOUS run `a[p*lda + i0 ..]` and each `[kc][MR]` panel row is a straight copy of six
/// consecutive floats — cheaper than the row-major packer, which has to transpose. Rows past `mc`
/// are zero, exactly as the row-major packer pads, so the panel bytes equal those of the explicitly
/// transposed matrix packed by the row-major path (`pack_a_transposed_matches_the_pack_of_the_transpose`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_a_t_panel(
    a: *const f32,
    lda: usize,
    mc: usize,
    kc: usize,
    ip: usize,
    panel: *mut f32,
) {
    let i0 = ip * MR;
    let nrows = (mc - i0).min(MR);
    if nrows == MR {
        for p in 0..kc {
            std::ptr::copy_nonoverlapping(a.add(p * lda + i0), panel.add(p * MR), MR);
        }
    } else {
        for p in 0..kc {
            let (src, dst) = (a.add(p * lda + i0), panel.add(p * MR));
            for r in 0..MR {
                *dst.add(r) = if r < nrows { *src.add(r) } else { 0.0 };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_a(a: *const f32, lda: usize, at: bool, mc: usize, kc: usize, ap: *mut f32) {
    let mpanels = mc.div_ceil(MR);
    for ip in 0..mpanels {
        pack_a_panel(a, lda, at, mc, kc, ip, ap.add(ip * kc * MR));
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
/// Operand contracts of [`pack_a`] (`a`/`lda`/`at`: the [`a_origin`] of the K-block's column 0) and
/// [`pack_b_block`]; `ap`/`bp` sized as in
/// [`sgemm_avx2_parallel`] (`round_up(mc,MR)·kc` / `round_up(nc,NR)·kc` f32); avx2 verified by the
/// caller.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn pack_ab_par(
    a: *const f32,
    lda: usize,
    at: bool,
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
    let mpanels = mc.div_ceil(MR);
    let npanels = nc.div_ceil(NR);
    let total_bytes = (round_up(mc, MR) + round_up(nc, NR)) * kc * 4;
    if total_bytes < pack_par_min_bytes() {
        pack_a(a, lda, at, mc, kc, ap);
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
        crate::par::for_each_unit(mpanels, |ip| {
            // SAFETY: disjoint output panel per task; pointers re-derived per task.
            unsafe {
                pack_a_panel(
                    a_addr as *const f32,
                    lda,
                    at,
                    mc,
                    kc,
                    ip,
                    (ap_addr as *mut f32).add(ip * kc * MR),
                );
            }
        });
        crate::par::for_each_unit(npanels, |jp| {
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
    crate::par::for_each_unit(mpanels + npanels, |t| {
        // SAFETY: every task writes a disjoint packed panel; pointers re-derived per task.
        unsafe {
            if t < mpanels {
                pack_a_panel(
                    a_addr as *const f32,
                    lda,
                    at,
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
                PfWalk::NONE,
            );
        }
    }
}

/// [`macro_kernel`] for a macro-block whose packed B panel would be swept exactly once (see
/// [`gemm_fused_pack`]): instead of reading a pre-packed `kc×nc` panel, it packs B micropanel `jp`
/// from the source slice at `b_src` (leading dimension `ldb`, the [`b_origin`] of this K/column
/// block) into the `kc×NR` buffer `bpanel` right before the `mpanels` micro-tiles that consume it,
/// and hands each of those micro-tiles one [`PfWalk`] slice of micropanel `jp + 1`'s source.
///
/// Bit-identical to `pack_b_block` + [`macro_kernel`] by construction: the same panel function
/// (`pack_b_trans_panel` / `pack_b_panel`) on the same source and `jp` writes the same `[kc][NR]`
/// bytes the whole-panel pack places at `bp + jp·kc·NR`; the micro-tiles then run in the same
/// order with the same `(kc, A panel, C tile, ldc, beta, mr, nr, epilogue shift)` arguments.
///
/// # Safety
/// `ap` holds the packed `mc×kc` A block; `b_src`/`ldb` address the `kc×nc` B slice in the layout
/// `bt` names; `bpanel` is writable for `kc·NR` f32 and disjoint from every operand; `c` is the
/// macro-block's C origin with leading dimension `ldc`; avx2+fma available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn macro_kernel_packb(
    mc: usize,
    nc: usize,
    kc: usize,
    ap: *const f32,
    b_src: *const f32,
    ldb: usize,
    bt: bool,
    bpanel: *mut f32,
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
        if bt {
            pack_b_trans_panel(b_src, ldb, kc, nc, jp, bpanel);
        } else {
            pack_b_panel(b_src, ldb, kc, nc, jp, bpanel);
        }
        // Each column panel's bias starts `j0` columns into this macro-block's epilogue.
        let panel_epi = epi.map(|e| e.shift(j0));
        let has_next = jp + 1 < npanels;
        for ip in 0..mpanels {
            let i0 = ip * MR;
            let mrv = (mc - i0).min(MR);
            let pf = if has_next {
                PfWalk::for_b_panel(b_src, ldb, bt, kc, nc, jp + 1, ip, mpanels)
            } else {
                PfWalk::NONE
            };
            micro_6x16(
                kc,
                ap.add(ip * kc * MR),
                bpanel,
                c.add(i0 * ldc + j0),
                ldc,
                beta,
                mrv,
                nrv,
                panel_epi,
                pf,
            );
        }
    }
}

/// The `6×16` AVX2/FMA microkernel: 12 live `__m256` accumulators, 12 FMAs per K step. Accumulates
/// the tile's `mr` valid rows over all 16 columns (packed panels are zero-padded), then writes back
/// the valid `mr×nr` corner with the `beta` rule — 256-bit throughout, masked at a column edge.
/// `pf` is a software-prefetch walk issued one line per four-step K iteration ([`PfWalk`];
/// [`PfWalk::NONE`] for none) — a hint that never changes a result bit.
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
    pf: PfWalk,
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
    if epi.is_none() && mr == MR && nr == NR && crate::isa::avx512f() {
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
    // One K-step over the listed rows: load the 16-wide B row (two 256-bit lanes), then for each
    // listed A row load-broadcast it straight from the packed panel (`vbroadcastss [mem]`, 1 uop —
    // frees the shuffle/ALU ports for the FMAs) and FMA into that row's two accumulators. The full
    // tile lists all six rows: 12 FMAs, 12 live accumulators. The packed A panel keeps its `MR`
    // stride whichever rows are listed.
    macro_rules! kstep {
        ($($r:tt => $lo:ident, $hi:ident);+) => {{
            let b0 = _mm256_loadu_ps(bp);
            let b1 = _mm256_loadu_ps(bp.add(8));
            $(
                let a = _mm256_broadcast_ss(&*ap.add($r));
                $lo = _mm256_fmadd_ps(a, b0, $lo);
                $hi = _mm256_fmadd_ps(a, b1, $hi);
            )+
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
    //
    // The `pf` walk rides the same cadence: at most one `PREFETCHT1` per four K steps, so a caller's
    // look-ahead (the next B micropanel's source, see `macro_kernel_packb`) trickles into L2 under
    // the FMAs instead of arriving as a burst. `NONE` costs one predictable compare per iteration.
    macro_rules! kloop {
        ($($rows:tt)+) => {{
            let mut pf = pf;
            let mut pf_line = 0usize;
            let mut p = 0;
            while p + 4 <= kc {
                _mm_prefetch::<_MM_HINT_T0>(bp.wrapping_add(NR * 8) as *const i8);
                if pf.rows != 0 {
                    _mm_prefetch::<_MM_HINT_T1>(pf.ptr.wrapping_add(pf_line * 16) as *const i8);
                    pf_line += 1;
                    if pf_line == pf.lines {
                        pf_line = 0;
                        pf.ptr = pf.ptr.wrapping_add(pf.row_stride);
                        pf.rows -= 1;
                    }
                }
                kstep!($($rows)+);
                kstep!($($rows)+);
                kstep!($($rows)+);
                kstep!($($rows)+);
                p += 4;
            }
            while p < kc {
                kstep!($($rows)+);
                p += 1;
            }
        }};
    }
    // Row-trimmed K loops (P9.20): a tile with `mr < MR` valid rows accumulates only those rows.
    // Until 2026-10 every edge tile ran all six — at m = 128 (21 full row panels + one 2-row panel
    // per column panel) that is 4 of every 132 FMA rows computed and thrown away. Each listed row's
    // accumulator sees exactly the products, in exactly the ascending-`p` order, that the full
    // loop gives it (the rows never interact), so the valid rows are bit-identical either way; the
    // unlisted accumulators stay zero and are never written back. With ≤ 4 rows the loop is bound
    // by FMA latency (≤ 8 independent chains) rather than the two FMA ports, so the trimmed loop
    // costs ~4 cycles per step against the full tile's ~6, not mr/6 of it.
    match mr {
        1 => kloop!(0 => c0, c1),
        2 => kloop!(0 => c0, c1; 1 => c2, c3),
        3 => kloop!(0 => c0, c1; 1 => c2, c3; 2 => c4, c5),
        4 => kloop!(0 => c0, c1; 1 => c2, c3; 2 => c4, c5; 3 => c6, c7),
        5 => kloop!(0 => c0, c1; 1 => c2, c3; 2 => c4, c5; 3 => c6, c7; 4 => c8, c9),
        _ => kloop!(
            0 => c0, c1; 1 => c2, c3; 2 => c4, c5; 3 => c6, c7; 4 => c8, c9; 5 => c10, c11
        ),
    }

    // Fast path: a full 6×16 tile with no fused epilogue — store the 12 accumulators straight to C
    // (two 256-bit stores per row) with the beta rule and no mask or row test. The K loop
    // dominates only at large K; at the small K of conv/attention (and the n=256 GEMM) the
    // writeback is a real fraction, so this is a measurable win there and free elsewhere.
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
    // activation — all 256-bit (the edge-tile writeback below is the masked form of this same
    // sequence). Bit-identical to the scalar `Epilogue::apply`, and
    // `nt_epi_epilogue_is_bit_exact_on_non_finite_data` +
    // `sgemm_nt_epi_matches_reference` now pin that rather than asserting it. The ReLU step needs
    // care, because the crate's invariant #2 says `_mm256_max_ps` is NOT `f32::max` in general:
    // `MAXPS` is `SRC1 > SRC2 ? SRC1 : SRC2`, so it returns SRC2 on a tie or a NaN. Here SRC2 is the
    // zero vector, and on that one operand pair the two DO coincide for every f32 — `v > 0 ? v : 0`
    // sends every NaN, every negative and both zeros to `+0.0`, which is exactly what Rust's
    // `f32::max(v, 0.0)` returns (pinned over the hostile set by
    // `relu_epilogue_vector_fold_matches_the_scalar_fold`). `gelu8`/`silu8` mirror `gelu1`/`silu1`
    // lane-for-lane (the vmath tail-match invariant), so the interpreter's scalar oracle agrees.
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

    // Edge tiles — `mr < MR` rows or `nr < NR` columns, with or without the epilogue (P9.20). Until
    // 2026-10 these spilled all twelve accumulators to a stack tile and wrote the valid corner
    // back one element at a time — a scalar load, add, `Epilogue::apply` (with its per-element
    // `match` on the activation) and store per C element — so every tile of a matrix whose M is
    // not a multiple of 6 or whose N is not a multiple of 16, and EVERY fused-epilogue edge tile,
    // ran a scalar writeback while the full tiles above ran 256-bit. Now each valid row takes the
    // full-tile path's own vector sequence — beta add, α, bias, activation, in that order and with
    // the same operand order — straight from the accumulator registers, and the column edge is a
    // lane mask: `vmaskmovps` loads (C on the beta path, the bias) read nothing at or past column
    // `nr`, and masked stores write nothing there, so no byte outside the `mr×nr` corner is touched
    // (masked-out lanes cannot fault). A full 8-lane half keeps the plain load/store. Lane for
    // lane this is the full-tile writeback, which `Epilogue::apply` and the scalar twin match
    // (`relu_epilogue_vector_fold_matches_the_scalar_fold`, `sgemm_scalar_matches_avx2_bit_for_bit`).
    let zero = _mm256_setzero_ps();
    let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let (full_lo, has_hi, full_hi) = (nr >= 8, nr > 8, nr == NR);
    let mask_lo = _mm256_cmpgt_epi32(_mm256_set1_epi32(nr.min(8) as i32), lane);
    let mask_hi = _mm256_cmpgt_epi32(_mm256_set1_epi32(nr.saturating_sub(8) as i32), lane);
    macro_rules! ld {
        ($p:expr, $full:expr, $mask:expr) => {
            if $full {
                _mm256_loadu_ps($p)
            } else {
                _mm256_maskload_ps($p, $mask)
            }
        };
    }
    macro_rules! st {
        ($p:expr, $full:expr, $mask:expr, $v:expr) => {
            if $full {
                _mm256_storeu_ps($p, $v)
            } else {
                _mm256_maskstore_ps($p, $mask, $v)
            }
        };
    }
    // The epilogue's per-tile constants, hoisted out of the rows exactly as the full-tile path
    // hoists them. No bias lane at or past column `nr` is loaded (the hi half only when `nr > 8`).
    let (has_bias, bias_lo, bias_hi, apply_alpha, alpha8, act) = match epi {
        Some(e) => {
            let has_bias = !e.bias.is_null();
            let bias_lo = if has_bias {
                ld!(e.bias, full_lo, mask_lo)
            } else {
                zero
            };
            let bias_hi = if has_bias && has_hi {
                ld!(e.bias.add(8), full_hi, mask_hi)
            } else {
                zero
            };
            let alpha8 = _mm256_set1_ps(e.alpha);
            (
                has_bias,
                bias_lo,
                bias_hi,
                e.alpha != 1.0,
                alpha8,
                Some(e.act),
            )
        }
        None => (false, zero, zero, false, zero, None),
    };
    macro_rules! act8 {
        ($v:expr) => {
            match act {
                Some(ACT_RELU) => _mm256_max_ps($v, zero),
                Some(ACT_GELU) => crate::vmath::gelu8($v),
                Some(ACT_SILU) => crate::vmath::silu8($v),
                _ => $v,
            }
        };
    }
    // One 8-lane half of a C row: the beta rule, then the epilogue when present, then the store.
    macro_rules! half {
        ($p:expr, $acc:expr, $full:expr, $mask:expr, $bias:expr) => {{
            let mut v = if beta == 0.0 {
                $acc
            } else {
                _mm256_add_ps(ld!($p, $full, $mask), $acc)
            };
            if act.is_some() {
                if apply_alpha {
                    v = _mm256_mul_ps(v, alpha8);
                }
                if has_bias {
                    v = _mm256_add_ps(v, $bias);
                }
                v = act8!(v);
            }
            st!($p, $full, $mask, v);
        }};
    }
    macro_rules! wb_edge {
        ($lo:expr, $hi:expr, $r:expr) => {{
            if $r < mr {
                let row = c.add($r * ldc);
                half!(row, $lo, full_lo, mask_lo, bias_lo);
                if has_hi {
                    half!(row.add(8), $hi, full_hi, mask_hi, bias_hi);
                }
            }
        }};
    }
    wb_edge!(c0, c1, 0);
    wb_edge!(c2, c3, 1);
    wb_edge!(c4, c5, 2);
    wb_edge!(c6, c7, 3);
    wb_edge!(c8, c9, 4);
    wb_edge!(c10, c11, 5);
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
    let (mut c0, mut c1, mut c2) = (
        _mm512_setzero_ps(),
        _mm512_setzero_ps(),
        _mm512_setzero_ps(),
    );
    let (mut c3, mut c4, mut c5) = (
        _mm512_setzero_ps(),
        _mm512_setzero_ps(),
        _mm512_setzero_ps(),
    );
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

    #[test]
    fn serial_gemm_does_not_initialize_pool() {
        const CHILD: &str = "WUKONG_TEST_SERIAL_GEMM_CHILD";
        if std::env::var_os(CHILD).is_none() {
            // Other runtime tests deliberately initialize pools. A fresh process makes the
            // absence of initialization observable regardless of test order or thread count.
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "gemm::tests::serial_gemm_does_not_initialize_pool",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .env("WUKONG_GEMM_PAR_MIN_MACS", PAR_MIN_MACS.to_string())
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "cold serial GEMM failed: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }

        assert!(GEMM_POOL.get().is_none());
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        for parallel in [false, true] {
            for transposed in [false, true] {
                for beta in [0, 1] {
                    let mut c = [1.0; 4];
                    // SAFETY: all buffers contain the four elements required by 2x2 matrices.
                    unsafe {
                        match (parallel, transposed) {
                            (false, false) => {
                                wukong_sgemm(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2, beta)
                            }
                            (false, true) => wukong_sgemm_nt(
                                a.as_ptr(),
                                b.as_ptr(),
                                c.as_mut_ptr(),
                                2,
                                2,
                                2,
                                beta,
                            ),
                            (true, false) => wukong_sgemm_parallel(
                                a.as_ptr(),
                                b.as_ptr(),
                                c.as_mut_ptr(),
                                2,
                                2,
                                2,
                                beta,
                            ),
                            (true, true) => wukong_sgemm_nt_parallel(
                                a.as_ptr(),
                                b.as_ptr(),
                                c.as_mut_ptr(),
                                2,
                                2,
                                2,
                                beta,
                            ),
                        }
                    }
                    let product = if transposed {
                        [17.0, 23.0, 39.0, 53.0]
                    } else {
                        [19.0, 22.0, 43.0, 50.0]
                    };
                    assert_eq!(c, product.map(|x| x + beta as f32));
                    assert!(
                        GEMM_POOL.get().is_none(),
                        "serial work initialized pools (parallel={parallel}, transposed={transposed}, beta={beta})"
                    );
                }
            }
        }
    }

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
        assert_eq!(
            positive_or_unset(Some("4194304".to_string())),
            Some(4 << 20)
        );
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

    // ---- the numeric contract: an f64 reference and a bound DERIVED from the accumulation order --
    //
    // Until 2026-09-13 every vs-reference GEMM assertion in this file was
    // `|got - want| <= 1e-3*sqrt(k) + 1e-4*|want|` against an **f32** `naive` that accumulated in the
    // same precision as the thing it checked, so the oracle carried the same order of error as the
    // kernel; `f32::EPSILON` appeared nowhere in the file and `as f64` appeared only inside two
    // `#[ignore]`d throughput probes. Measured on the shipped shapes, that tolerance was 131x
    // (128,256,512) to 334x (100,100,100) looser than the rounding scale of its own test data, and a
    // GEMM that silently truncated every input to half precision passed it at every shape (0/65536,
    // 0/1095, 0/10000 failing entries) - see `the_derived_bound_rejects_a_half_precision_gemm`.
    //
    // The replacement has two independently citable halves:
    //
    //   * the SCALING is netlib's Level-3 BLAS test criterion. `SMMCH` in `sblat3.f` accumulates a
    //     growth factor `G(I) = G(I) + ABS(A(I,K))*ABS(B(K,J))` and normalizes the error by it. That
    //     is the only scaling that survives cancellation - `|C|` itself does not, and with the fixed
    //     `fill` below the C entries now cancel;
    //   * the CONSTANT is Higham's `gamma_r = r*u/(1-r*u)` (Accuracy and Stability of Numerical
    //     Algorithms, 2nd ed., Sec. 3.1, Lemma 3.1) with `r` the number of roundings THIS kernel's
    //     accumulation order actually performs. netlib must use an implementation-agnostic `sqrt(u)`
    //     because it cannot know the blocking; we do know it, so this bound is ~15x tighter than
    //     netlib's at k=384 and 60-70x tighter than the constant it replaced.

    /// Unit roundoff for binary32: `u = 2^-24`. (`f32::EPSILON` is `2^-23`, i.e. `2u`.)
    fn unit_roundoff() -> f64 {
        f64::from(f32::EPSILON) / 2.0
    }

    /// Higham's `gamma_r = r*u/(1-r*u)`: the relative bound on a product of `r` roundings.
    fn gamma(r: usize) -> f64 {
        let ru = r as f64 * unit_roundoff();
        assert!(ru < 1.0, "gamma_r is undefined for r = {r} in binary32");
        ru / (1.0 - ru)
    }

    /// How many roundings one product `a_ip*b_pj` passes through on its way into `C[i][j]`, read
    /// straight off the kernel: `select_kc(k)` fused steps inside its own K-block (each `vfmadd` /
    /// [`f32::mul_add`] rounds once), then one C-readback rounding per later K-block
    /// (`beta_eff = 1.0`), plus one more when the caller's `beta` is non-zero so block 0 reads C too.
    /// Retuning `KC` moves this number, which is the point: the bound records the grouping.
    fn gemm_rounding_steps(k: usize, beta_nonzero: bool) -> usize {
        let kc = select_kc(k).min(k);
        let nb = k.div_ceil(kc);
        kc + (nb - 1) + usize::from(beta_nonzero)
    }

    /// f64 reference for `C = A*B` (`bt == false`, B is `[k, n]`) or `C = A*B^T` (`bt == true`, B is
    /// `[n, k]`), returning the result AND netlib `SMMCH`'s growth factor
    /// `g[i*n+j] = sum_p |a_ip|*|b_pj|`. f64 carries 29 more mantissa bits than f32, so at the `k`
    /// these tests reach the reference's own error is ~2^-29 of the bound it is used to police.
    fn reference_f64(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        bt: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut c = vec![0.0f64; m * n];
        let mut g = vec![0.0f64; m * n];
        if bt {
            // Row `j` of B is contiguous: keep `p` innermost so both operand reads are unit-stride.
            for i in 0..m {
                for j in 0..n {
                    let (mut acc, mut gacc) = (0.0f64, 0.0f64);
                    for p in 0..k {
                        let (av, bv) = (f64::from(a[i * k + p]), f64::from(b[j * k + p]));
                        acc += av * bv;
                        gacc += av.abs() * bv.abs();
                    }
                    c[i * n + j] = acc;
                    g[i * n + j] = gacc;
                }
            }
        } else {
            // B is `[k, n]`: keep `j` innermost so the B row, `c` and `g` are all unit-stride.
            for i in 0..m {
                for p in 0..k {
                    let av = f64::from(a[i * k + p]);
                    let aav = av.abs();
                    for j in 0..n {
                        let bv = f64::from(b[p * n + j]);
                        c[i * n + j] += av * bv;
                        g[i * n + j] += aav * bv.abs();
                    }
                }
            }
        }
        (c, g)
    }

    /// [`reference_f64`] for `C = A^T*B`: A stored row-major `[k, m]`, B `[k, n]`, C `[m, n]`.
    fn reference_tn_f64(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut c = vec![0.0f64; m * n];
        let mut g = vec![0.0f64; m * n];
        for p in 0..k {
            for i in 0..m {
                let av = f64::from(a[p * m + i]);
                let aav = av.abs();
                for j in 0..n {
                    let bv = f64::from(b[p * n + j]);
                    c[i * n + j] += av * bv;
                    g[i * n + j] += aav * bv.abs();
                }
            }
        }
        (c, g)
    }

    /// The GEMM numeric contract: `|got - exact| <= gamma_r * (sum_p |a_ip||b_pj| + |beta*c_ij|)`.
    /// `beta_c` is `Some(c_init)` exactly when the call ran with a non-zero `beta`, which both adds a
    /// rounding and puts the incoming C into the growth factor (netlib scales `G` by
    /// `ABS(BETA)*ABS(C)` for the same reason). Where the growth factor is zero the bound is zero and
    /// exact equality is required: an all-zero set of products sums to exactly zero in any order.
    #[allow(clippy::too_many_arguments)]
    fn assert_within_derived_bound(
        got: &[f32],
        want: &[f64],
        growth: &[f64],
        m: usize,
        k: usize,
        n: usize,
        beta_c: Option<&[f32]>,
        what: &str,
    ) {
        let r = gemm_rounding_steps(k, beta_c.is_some());
        let g = gamma(r);
        for i in 0..m * n {
            let growth_i = growth[i] + beta_c.map_or(0.0, |c0| f64::from(c0[i].abs()));
            let bound = g * growth_i;
            let err = (f64::from(got[i]) - want[i]).abs();
            assert!(
                err <= bound,
                "{what} ({m}x{k}x{n}) idx {i}: got {} want {} -> |err| {err:e} exceeds the derived \
                 bound gamma_{r} * (sum|a||b| + |beta*c|) = {g:e} * {growth_i:e} = {bound:e}. \
                 Either the accumulation order changed, or `gemm_rounding_steps` no longer matches \
                 `select_kc`.",
                got[i],
                want[i]
            );
        }
    }

    /// The fused epilogue `act(alpha*x + bias[j])`, transcribed independently of
    /// [`Epilogue::apply`] so the two are a real pair rather than one expression called twice.
    fn epilogue_ref(x: f32, j: usize, alpha: f32, bias: Option<&[f32]>, act: u32) -> f32 {
        let mut y = if alpha != 1.0 { alpha * x } else { x };
        if let Some(bs) = bias {
            y += bs[j];
        }
        match act {
            ACT_RELU => y.max(0.0),
            ACT_GELU => crate::vmath::gelu1(y),
            ACT_SILU => crate::vmath::silu1(y),
            _ => y,
        }
    }

    /// Deterministic pseudo-random values in `[-1, 1)`.
    ///
    /// The comment said `[-1, 1)` and the expression produced `[-1, 0)` until 2026-09-13: `s >> 33`
    /// on a `u64` lies in `[0, 2^31)`, so `(s >> 33) / 2^31 - 1.0` never reached zero (measured over
    /// 1000 samples: min -0.99923, max -0.00213, 0 of them >= 0). Every A and B entry being negative
    /// made every product positive and every C entry a large positive sum, with two consequences that
    /// silently deleted coverage: the fused-epilogue **ReLU was provably the identity** at every
    /// shipped shape (0 of 65536 entries clamped, worst delta exactly 0.0), so the four activation
    /// legs were four copies of a passthrough test; and the reduction never once cancelled, a
    /// same-signed sum being the numerically easiest case there is - the second reason the retired
    /// `1e-3*sqrt(k)` tolerance never bit. Only the scale changes here (`2^30`), so the same LCG
    /// stream now spans the documented range.
    fn fill(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                fill_value(s)
            })
            .collect()
    }

    /// One LCG state -> one sample. Split out of [`fill`] so `[-1, 1)` is a property of the
    /// FUNCTION, pinnable at its endpoints, rather than of whichever seeds and lengths the suite
    /// happens to use today.
    ///
    /// 24 bits of state, not 31, and the upper endpoint is the reason. `(s >> 33)` reaches
    /// `2^31 - 1`; f32's ulp at `2^31` is 128, so every numerator `>= 2^31 - 64` rounds UP to `2^31`
    /// on the `as f32` and the sample is exactly `2^31/2^30 - 1.0 == 1.0` - outside the documented
    /// half-open range, and a violation of [`fill_spans_the_documented_range`]'s `max < 1.0`. No
    /// seed in the suite reaches it today, which is exactly what makes it a trap: a shape or seed
    /// added later would trip an assertion in a test that has nothing to do with the generator.
    /// Widening the division to f64 does NOT fix it - `1 - 2^-30` is inside a half-ulp of `1.0` in
    /// f32 and rounds back to `1.0` on the way down. With `s >> 40` the numerator is `< 2^24` and
    /// therefore EXACT in f32, the division by `2^23` is exact (a power of two), and `u/2^23 - 1.0`
    /// is exact by Sterbenz, so the closed form of the range is `[-1, 1 - 2^-23]` with no rounding
    /// step anywhere in it. (The LCG's high bits are also the good ones, so taking 24 of them rather
    /// than 31 costs nothing but the granularity.)
    fn fill_value(s: u64) -> f32 {
        ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }

    /// The generator's range is load-bearing for the epilogue and cancellation coverage above, so it
    /// is pinned rather than trusted to a comment.
    #[test]
    fn fill_spans_the_documented_range() {
        let v = fill(7, 4096);
        let min = v.iter().copied().fold(f32::INFINITY, f32::min);
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(min >= -1.0, "fill left [-1, 1): min {min}");
        assert!(max < 1.0, "fill left [-1, 1): max {max}");
        assert!(min < -0.9, "fill never approaches -1: min {min}");
        assert!(max > 0.9, "fill never approaches +1: max {max}");
        let negatives = v.iter().filter(|x| **x < 0.0).count();
        assert!(
            negatives > v.len() / 4 && negatives < 3 * v.len() / 4,
            "fill is not sign-balanced ({negatives} of {} below zero) - the epilogue and \
             cancellation coverage depend on it",
            v.len()
        );
        // The endpoints, pinned against the FUNCTION and not against the seeds this suite happens
        // to use: a sampled `max < 1.0` only says today's seeds miss the top of the range. The
        // retired `(s >> 33) / 2^30` form fails the first line below - it returns exactly `1.0` for
        // every numerator `>= 2^31 - 64` - and so does the f64-division "fix" for it.
        for &s in &[
            u64::MAX,
            u64::MAX - 1,
            !0u64 << 40,
            (!0u64 << 40) | 0xff_ffff_ffff,
        ] {
            let x = fill_value(s);
            assert!(
                x < 1.0,
                "fill_value({s:#x}) = {x} reached the open upper bound"
            );
            assert!(x >= -1.0, "fill_value({s:#x}) = {x} left [-1, 1)");
        }
        assert_eq!(
            fill_value(0).to_bits(),
            (-1.0f32).to_bits(),
            "the closed lower endpoint moved: fill_value(0) = {}",
            fill_value(0)
        );
    }

    /// The K partition is part of the kernel's observable result (see [`select_kc`]), so it is pinned
    /// BY VALUE: a `KC` retune must show up as a reviewed diff here, not as a silent change to the
    /// low bits of every GEMM in the project. Before this test the only in-test uses of `select_kc`
    /// were `assert!(k > select_kc(k))` preconditions, which fail only if `KC` is RAISED.
    ///
    /// Every value below is DERIVED, not recorded from output: all 12 `select_kc` results and all 7
    /// block counts were recomputed independently from `round_up(k.div_ceil(k.div_ceil(384).max(1)),
    /// 4)` (twice, by two readers) before being written here. The test can fail - changing the
    /// `round_up` multiple from 4 to 8 breaks `select_kc(1)`, `select_kc(257)` and `select_kc(300)`,
    /// and any `KC` retune breaks the first `assert_eq!` on the constants.
    #[test]
    fn select_kc_partition_is_pinned() {
        assert_eq!((MR, NR, MC, KC, NC), (6, 16, 144, 384, 4080));
        for &(k, kc) in &[
            (1usize, 4usize),
            (4, 4),
            (257, 260),
            (300, 300),
            (383, 384),
            (384, 384),
            (385, 196),
            (391, 196),
            (512, 256),
            (805, 272),
            (901, 304),
            (1024, 344),
        ] {
            assert_eq!(select_kc(k), kc, "select_kc({k})");
            assert!(
                (4..=KC).contains(&select_kc(k)),
                "select_kc({k}) left [4, KC]"
            );
        }
        // The block count the derived error bound is built from, at the shapes the reference tests
        // use. `select_kc(k).min(k)` is the real block height (`select_kc(1) == 4 > k`).
        for &(k, nb) in &[
            (1usize, 1usize),
            (300, 1),
            (384, 1),
            (385, 2),
            (391, 2),
            (805, 3),
            (901, 3),
        ] {
            assert_eq!(
                k.div_ceil(select_kc(k).min(k)),
                nb,
                "K-block count for k={k}"
            );
        }
    }

    fn check(m: usize, k: usize, n: usize) {
        let a = fill(1, m * k);
        let b = fill(2, k * n);
        let (want, growth) = reference_f64(&a, &b, m, k, n, false);
        let mut got = vec![0.0f32; m * n];
        // SAFETY: `a`/`b` sized m*k/k*n and `got` m*n (reference comparison).
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
        assert_within_derived_bound(&got, &want, &growth, m, k, n, None, "sgemm");
    }

    /// The blocking-boundary shapes every reference test should reach, DERIVED from the tuning
    /// constants rather than restating them. The 2026-07 bump from KC=256 to KC=384 silently
    /// vacated every "the K loop blocks" claim in this file because the shapes were literals sized
    /// against the old constant; a size written as `KC + 7` follows any retune instead.
    /// [`assert_blocking_coverage`] is the other half: it fails if a retune (or an edit to a list)
    /// leaves a regime unreached, instead of letting the coverage lapse quietly.
    fn blocking_boundary_shapes() -> Vec<(usize, usize, usize)> {
        vec![
            (MR - 1, KC + 7, NR - 1),              // 2 K-blocks, a single partial tile
            (2 * MR + 1, 2 * KC + 37, 2 * NR + 3), // 3 uneven K-blocks over full 6x16 tiles + edges
            (MC + 1, 9, NC + NR + 3), // 2 MC blocks (ragged) x 2 NC blocks on the classic path
            (MC + MR, KC + 5, 3 * NR + 1), // 2 MC blocks, 2 K-blocks, partial last column panel
            (MC, KC, NR * 64),        // the single-row-block regime at its upper edge
        ]
    }

    /// Fail loudly if `shapes` no longer reaches every blocking regime of the serial five-loop:
    /// multiple (and uneven) K-blocks, a second ragged MC block, a second NC block on the classic
    /// path, the `select_nc` L2-capped column split, full 6x16 tiles, and partial MR/NR edges.
    fn assert_blocking_coverage(shapes: &[(usize, usize, usize)], what: &str) {
        let k_blocks = |k: usize| k.div_ceil(select_kc(k).min(k));
        let reaches = |pred: &dyn Fn(usize, usize, usize) -> bool| {
            shapes.iter().any(|&(m, k, n)| pred(m, k, n))
        };
        let regimes: [(&str, &dyn Fn(usize, usize, usize) -> bool); 7] = [
            ("three or more K-blocks", &|_, k, _| k_blocks(k) >= 3),
            ("a ragged second MC block", &|m, _, _| {
                m > MC && !m.is_multiple_of(MC)
            }),
            ("a second NC block on the classic path", &|m, _, n| {
                m > MC && n > NC
            }),
            ("the L2-capped column split", &|m, k, n| {
                m <= MC && select_nc(m, k, n) < n
            }),
            ("a full 6x16 tile", &|m, _, n| m >= MR && n >= NR),
            ("a partial MR edge", &|m, _, _| !m.is_multiple_of(MR)),
            ("a partial NR edge", &|_, _, n| !n.is_multiple_of(NR)),
        ];
        for (name, pred) in regimes {
            assert!(
                reaches(pred),
                "{what}: no shape reaches {name} at MR={MR} NR={NR} MC={MC} KC={KC} NC={NC} - a \
                 retune or an edit vacated this coverage"
            );
        }
    }

    #[test]
    fn sgemm_matches_naive_various_sizes() {
        // Small literal sizes for the MR=6 / NR=16 remainders and 1-wide dims, then the derived
        // blocking-boundary shapes. Before 2026-09-13 every reference-checked GEMM shape in the crate
        // had k <= 300 < KC, `select_kc` always returned k, and the multi-K-block accumulation path -
        // the `beta_eff = 1.0` C readback and the `is_last_k` epilogue gate - was never compared
        // against anything but itself; before 2026-10 no serial shape crossed NC at all.
        let mut shapes = vec![
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
            // Small m with n past the L2 column cap: `select_nc` splits these into several jc
            // blocks (see the `select_nc_*` tests).
            (12, 384, 1400),
            (6, 384, 1400),
            (145, 384, 1400), // m just OVER MC: must stay on the classic single-NC-block path
        ];
        shapes.extend(blocking_boundary_shapes());
        assert_blocking_coverage(&shapes, "sgemm_matches_naive_various_sizes");
        for &(m, k, n) in &shapes {
            check(m, k, n);
        }
    }

    /// The `A·Bᵀ` and `Aᵀ·B` forms at the same derived blocking boundaries, against the f64
    /// reference at the derived bound. The NT form drives the transposing B-packer across every
    /// regime; the TN form drives the transposed-A packer (A stored `[k, m]`). Each must also equal
    /// its own parallel entry bit-for-bit.
    #[test]
    fn sgemm_nt_and_tn_match_reference_at_blocking_boundaries() {
        let shapes = blocking_boundary_shapes();
        assert_blocking_coverage(
            &shapes,
            "sgemm_nt_and_tn_match_reference_at_blocking_boundaries",
        );
        for &(m, k, n) in &shapes {
            let (mi, ki, ni) = (m as i64, k as i64, n as i64);
            // NT: B stored [n, k].
            let a = fill(401, m * k);
            let b = fill(402, n * k);
            let (want, growth) = reference_f64(&a, &b, m, k, n, true);
            let (mut got, mut got_par) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
            // SAFETY: `a`/`b` sized m*k/k*n; `got`/`got_par` m*n.
            unsafe {
                wukong_sgemm_nt(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), mi, ki, ni, 0);
                wukong_sgemm_nt_parallel(
                    a.as_ptr(),
                    b.as_ptr(),
                    got_par.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
            }
            assert_within_derived_bound(&got, &want, &growth, m, k, n, None, "nt boundary");
            assert_eq!(got, got_par, "nt serial vs parallel ({m}x{k}x{n})");
            // TN: A stored [k, m], B stored [k, n].
            let at = fill(403, k * m);
            let bn = fill(404, k * n);
            let (want, growth) = reference_tn_f64(&at, &bn, m, k, n);
            let (mut got, mut got_par) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
            // SAFETY: `at`/`bn` sized k*m/k*n (TN) and `got`/`got_par` m*n.
            unsafe {
                wukong_sgemm_tn(at.as_ptr(), bn.as_ptr(), got.as_mut_ptr(), mi, ki, ni, 0);
                wukong_sgemm_tn_parallel(
                    at.as_ptr(),
                    bn.as_ptr(),
                    got_par.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
            }
            assert_within_derived_bound(&got, &want, &growth, m, k, n, None, "tn boundary");
            assert_eq!(got, got_par, "tn serial vs parallel ({m}x{k}x{n})");
        }
    }

    /// `select_nc`'s policy, as a pure function: the classic `NC` everywhere except the
    /// single-row-block regime, and there a multiple of `NR` that keeps the packed B panel, the
    /// packed A block and the live C column strip inside L2.
    #[test]
    fn select_nc_keeps_the_classic_block_outside_the_small_m_regime() {
        // m > MC: several ic blocks amortise an L3-resident panel - unchanged.
        for &(m, k, n) in &[
            (145usize, 384usize, 3072usize),
            (512, 3072, 3072),
            (2048, 2048, 2048),
        ] {
            assert_eq!(select_nc(m, k, n), NC, "m={m} k={k} n={n}");
        }
        // m <= MC but n already inside the cap: still one block, so leave it alone.
        assert_eq!(select_nc(128, 768, 768), NC);
        assert_eq!(select_nc(1, 1, 1), NC);
    }

    #[test]
    fn select_nc_caps_the_panel_to_l2_when_there_is_one_row_block() {
        // The measured case: m=128, kc=384, n=3072 -> four 768-wide blocks (1.18 MiB panel)
        // instead of one 3072-wide one (4.7 MiB, which does not fit a 2 MiB L2).
        assert_eq!(
            select_nc(128, 768, 3072),
            768,
            "expected the four-block split"
        );
        for &(m, k, n) in &[
            (128usize, 768usize, 3072usize),
            (128, 3072, 3072),
            (6, 384, 1400),
            (144, 4096, 8192),
            (37, 512, 4000),
        ] {
            let nc = select_nc(m, k, n);
            let (mc, kc) = (m.min(MC), select_kc(k));
            assert!(
                nc >= NR && nc.is_multiple_of(NR),
                "nc={nc} must be a positive NR multiple"
            );
            assert!(nc <= NC, "nc={nc} must not exceed NC");
            // The budget the cap is derived from actually holds.
            let bytes = 4 * (kc * nc + round_up(mc, MR) * kc + mc * nc);
            assert!(
                bytes <= L2_BYTES,
                "m={m} k={k} n={n}: nc={nc} needs {bytes} B of a {L2_BYTES} B L2"
            );
            // No thin tail: the last block is at least half a full one.
            let last = n - (n.div_ceil(nc) - 1) * nc;
            assert!(
                last * 2 >= nc,
                "m={m} k={k} n={n}: tail block {last} of {nc}"
            );
        }
    }

    /// The column-block split must be **blocking only**. `nc` decides which C columns are
    /// computed together and nothing else; the per-`(i, j)` K reduction order is untouched, so
    /// the serial kernel must still equal the parallel one bit-for-bit at a shape that actually
    /// takes the new multi-jc-block path (`m <= MC`, `n` past the L2 cap). The parallel path does
    /// its own blocking and this change does not touch it, so it is the independent witness.
    #[test]
    fn small_m_wide_n_serial_matches_parallel_bit_for_bit() {
        let (m, k, n) = (128usize, 384usize, 1400usize);
        assert!(
            select_nc(m, k, n) < n,
            "shape must exercise the multi-block path"
        );
        let a = fill(3, m * k);
        let b = fill(4, n * k);
        let mut serial = vec![0.0f32; m * n];
        let mut par = vec![0.0f32; m * n];
        // SAFETY: `serial`/`par` sized m*n; inputs sized for the shape.
        unsafe {
            wukong_sgemm_nt(
                a.as_ptr(),
                b.as_ptr(),
                serial.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
                0,
            );
            wukong_sgemm_nt_parallel(
                a.as_ptr(),
                b.as_ptr(),
                par.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
                0,
            );
        }
        assert_eq!(serial, par, "serial != parallel at {m}x{k}x{n}");
    }

    /// The derived bound must have teeth. Rounding EVERY A and B element to fp16 (11-bit mantissa)
    /// before multiplying passed the retired `1e-3*sqrt(k)` tolerance at every shipped shape - 0 of
    /// 65536 failing entries at (128,256,512), worst |err| 0.00575 against a tolerance of 0.016 -
    /// i.e. a GEMM that silently truncated all of its inputs to half precision was indistinguishable
    /// from a correct one. `gamma_r * sum|a||b|` must reject it.
    ///
    /// It drives [`assert_within_derived_bound`] itself rather than re-deriving `g * growth[i]`
    /// inline, and that is the whole point of the `should_panic`: the mutation this test exists to
    /// catch is a LOOSENING OF THE HELPER (multiply `bound` by 1000 at its comparison, or drop the
    /// `beta_c` term), and an inline transcription here would stay green under exactly that. The
    /// helper panics on the first entry it rejects, so "it panicked" is "it rejected at least one",
    /// and the `expected` string pins the panic to the helper's own message rather than to any
    /// other way this body could fail. The positive control - the same helper ACCEPTING the
    /// un-truncated GEMM at this shape - is `sgemm_matches_naive_various_sizes`, whose shape list
    /// contains `(128, 256, 512)`.
    #[test]
    #[should_panic(expected = "exceeds the derived bound")]
    fn the_derived_bound_rejects_a_half_precision_gemm() {
        let (m, k, n) = (128usize, 256usize, 512usize);
        let a = fill(1, m * k);
        let b = fill(2, k * n);
        let a_h: Vec<f32> = a.iter().map(|&x| crate::round_f16(x)).collect();
        let b_h: Vec<f32> = b.iter().map(|&x| crate::round_f16(x)).collect();
        let (want, growth) = reference_f64(&a, &b, m, k, n, false);
        let mut got = vec![0.0f32; m * n];
        // SAFETY: `a`/`b` sized m*k/k*n and `got` m*n.
        unsafe {
            wukong_sgemm(
                a_h.as_ptr(),
                b_h.as_ptr(),
                got.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
                0,
            );
        }
        assert_within_derived_bound(&got, &want, &growth, m, k, n, None, "fp16-truncated sgemm");
    }

    /// `beta != 0` accumulates onto the incoming C (it is a flag, never a scale). The second shape
    /// is DERIVED from `KC` so the accumulate crosses K blocks - with `beta != 0` the FIRST block
    /// reads C as well, and every later block re-reads the partial sum it just wrote
    /// (`beta_eff = 1.0`). That is exactly the `+(nb-1) +1` half of the bound's rounding count, and
    /// no reference test reached it while every checked shape had k <= 300 < KC = 384.
    #[test]
    fn sgemm_beta_accumulates() {
        for &(m, k, n) in &[(40usize, 50usize, 60usize), (40, 2 * KC + 37, 48)] {
            let a = fill(3, m * k);
            let b = fill(4, k * n);
            let base = fill(5, m * n);
            let (prod, growth) = reference_f64(&a, &b, m, k, n, false);
            let want: Vec<f64> = base
                .iter()
                .zip(&prod)
                .map(|(x, y)| f64::from(*x) + y)
                .collect();
            let mut got = base.clone();
            // SAFETY: `got` is a clone sized m*n (in-place beta accumulate).
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
            assert_within_derived_bound(&got, &want, &growth, m, k, n, Some(&base), "sgemm beta=1");
        }
    }

    /// Every piece of scheduling state [`pin_pcore`] overwrites, so [`restore_sched`] can put ALL
    /// of it back. `pin_pcore` narrows the calling thread's affinity *and* raises the priority of
    /// the whole PROCESS and of the calling thread; restoring only the mask left both elevations
    /// in force for the rest of the test process, so the parallel rows of [`sgemm_throughput`] —
    /// and every other `#[ignore]` probe sharing the process — were measured with the submitting
    /// thread at TIME_CRITICAL and the process at HIGH while the rayon workers doing the actual
    /// work stayed at NORMAL. On this hybrid P/E laptop that changes core placement, so the
    /// printed parallel GFLOP/s were not reproducible by a run that did not include this probe.
    // On non-Windows the restore path is a no-op, so the fields are written but never read there.
    #[cfg_attr(not(windows), allow(dead_code))]
    #[derive(Clone, Copy)]
    struct PrevSched {
        /// Thread affinity mask; 0 means "nothing to restore" (the non-Windows no-op).
        affinity: usize,
        /// Process priority class. `GetPriorityClass` returns 0 on failure — never pushed back.
        priority_class: u32,
        /// Thread priority. `GetThreadPriority` returns `THREAD_PRIORITY_ERROR_RETURN` on failure.
        thread_priority: i32,
    }
    /// `GetThreadPriority`'s documented failure return.
    const THREAD_PRIORITY_ERROR_RETURN: i32 = 0x7FFF_FFFF;

    /// Throughput probe (run: `cargo test -p wukong_runtime --release sgemm_throughput --
    /// --ignored --nocapture --test-threads=1`). The filter and `--test-threads=1` are part of the
    /// instrument: libtest's default `--test-threads` is the logical CPU count, so an unfiltered
    /// `--ignored` run starts all nine ignored probes in this crate at once and the `sgemm
    /// (1 core)` row below would be measured against an all-core 4096³ GEMM and two 64 MiB
    /// bandwidth streams. Sweeps square sizes so the parallel scaling (which improves with size, as
    /// the packs amortize and each core gets more compute per K-block) is visible, not just the
    /// small-matrix corner.
    /// Pin the current thread to one P-core (logical CPU 0) and raise priority for a repeatable
    /// single-core measurement; returns the previous scheduling state to restore before the
    /// parallel benches. On this hybrid laptop the single-core GFLOP/s otherwise swings ±30% with
    /// P/E scheduling and turbo, swamping microkernel changes. No-op off Windows.
    #[cfg(windows)]
    fn pin_pcore() -> PrevSched {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn GetCurrentProcess() -> isize;
            fn SetThreadAffinityMask(h: isize, mask: usize) -> usize;
            fn SetThreadPriority(h: isize, prio: i32) -> i32;
            fn GetThreadPriority(h: isize) -> i32;
            fn SetPriorityClass(h: isize, class: u32) -> i32;
            fn GetPriorityClass(h: isize) -> u32;
        }
        // SAFETY: plain scheduling syscalls on the current thread/process pseudo-handles, which
        // are always valid and need no close; every argument is a by-value integer.
        unsafe {
            // Read the state back BEFORE overwriting it: the elevation is meant to last only as
            // long as the pinned single-core rows.
            let priority_class = GetPriorityClass(GetCurrentProcess());
            let thread_priority = GetThreadPriority(GetCurrentThread());
            SetPriorityClass(GetCurrentProcess(), 0x0000_0080); // HIGH_PRIORITY_CLASS
            SetThreadPriority(GetCurrentThread(), 15); // THREAD_PRIORITY_TIME_CRITICAL
            let affinity = SetThreadAffinityMask(GetCurrentThread(), 0x1); // logical CPU 0 (a P-core)
            PrevSched {
                affinity,
                priority_class,
                thread_priority,
            }
        }
    }
    /// Undo [`pin_pcore`] — all three pieces, not just the affinity mask.
    #[cfg(windows)]
    fn restore_sched(prev: PrevSched) {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn GetCurrentProcess() -> isize;
            fn SetThreadAffinityMask(h: isize, mask: usize) -> usize;
            fn SetThreadPriority(h: isize, prio: i32) -> i32;
            fn SetPriorityClass(h: isize, class: u32) -> i32;
        }
        // SAFETY: the same current-thread/current-process pseudo-handle syscalls [`pin_pcore`]
        // used, with the values it read back from them.
        unsafe {
            // Each getter has a documented failure return that is not a valid setting; pushing it
            // back would leave the process in a state it was never in.
            if prev.affinity != 0 {
                SetThreadAffinityMask(GetCurrentThread(), prev.affinity);
            }
            if prev.priority_class != 0 {
                SetPriorityClass(GetCurrentProcess(), prev.priority_class);
            }
            if prev.thread_priority != THREAD_PRIORITY_ERROR_RETURN {
                SetThreadPriority(GetCurrentThread(), prev.thread_priority);
            }
        }
    }
    #[cfg(not(windows))]
    fn pin_pcore() -> PrevSched {
        PrevSched {
            affinity: 0,
            priority_class: 0,
            thread_priority: THREAD_PRIORITY_ERROR_RETURN,
        }
    }
    #[cfg(not(windows))]
    fn restore_sched(_: PrevSched) {}

    /// `pin_pcore` must hand back everything it took. It sets three pieces of state — the process
    /// priority class, the calling thread's priority, and the thread affinity mask — and the old
    /// `restore_affinity` put back only the mask. Observed before, in this exact sequence:
    /// `BEFORE class=0x20 thread_prio=0` → `PINNED class=0x80 thread_prio=15` →
    /// `RESTORED class=0x80 thread_prio=15`, i.e. HIGH_PRIORITY_CLASS and TIME_CRITICAL leaked for
    /// the remainder of the process and every later measurement in it. Gated here rather than in
    /// [`sgemm_throughput`] itself because that probe is `#[ignore]`d, so it never runs in CI.
    #[test]
    #[cfg(windows)]
    fn pin_pcore_restores_every_piece_of_scheduling_state() {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn GetCurrentProcess() -> isize;
            fn GetThreadPriority(h: isize) -> i32;
            fn GetPriorityClass(h: isize) -> u32;
            fn SetThreadAffinityMask(h: isize, mask: usize) -> usize;
        }
        // SAFETY: read-only scheduling queries on the current thread/process pseudo-handles, plus
        // one `SetThreadAffinityMask` that re-applies the mask already in force (see below).
        unsafe {
            let class0 = GetPriorityClass(GetCurrentProcess());
            let prio0 = GetThreadPriority(GetCurrentThread());
            let prev = pin_pcore();
            assert_eq!(
                GetPriorityClass(GetCurrentProcess()),
                0x0000_0080,
                "pin_pcore is supposed to raise the process to HIGH_PRIORITY_CLASS"
            );
            restore_sched(prev);
            assert_eq!(
                GetPriorityClass(GetCurrentProcess()),
                class0,
                "process priority class leaked past restore_sched"
            );
            assert_eq!(
                GetThreadPriority(GetCurrentThread()),
                prio0,
                "thread priority leaked past restore_sched"
            );
            // No `GetThreadAffinityMask` exists: setting the mask returns the one that was in
            // force, so re-setting the restored value both checks it and leaves it unchanged.
            assert_eq!(
                SetThreadAffinityMask(GetCurrentThread(), prev.affinity),
                prev.affinity,
                "thread affinity mask not restored"
            );
        }
    }

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
            // SAFETY: bench: ap/bp/cp address n*n buffers (square timing).
            bench("sgemm (1 core)", &|| unsafe {
                wukong_sgemm(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
            // SAFETY: bench: ap/bp/cp address n*n buffers (nt timing).
            bench("sgemm_nt (1 core)", &|| unsafe {
                wukong_sgemm_nt(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
            restore_sched(prev); // parallel benches want all cores, at the process's own priority
                                 // SAFETY: bench: ap/bp/cp address n*n buffers (parallel timing).
            bench("sgemm (parallel)", &|| unsafe {
                wukong_sgemm_parallel(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
            // SAFETY: bench: ap/bp/cp address n*n buffers (nt parallel timing).
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
        if !(crate::isa::avx2() && crate::isa::fma()) {
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
            let logical_run =
                // SAFETY: sgemm_avx2_parallel over ap/bp/cp, each an n*n buffer.
                || unsafe { sgemm_avx2_parallel(ap, bp, cp, n, n, n, 0.0, false, false, None) };
            let physical_run = || {
                let args = GemmArgs {
                    a: ap,
                    b: bp,
                    c: cp,
                    m: n,
                    k: n,
                    n,
                    beta: 0.0,
                    at: false,
                    bt: false,
                    epi: None,
                };
                // SAFETY: `args` wraps ap/bp/cp (n*n buffers); run() executes on the pool.
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
        let have = crate::isa::avx512f() && crate::isa::avx2() && crate::isa::fma();
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
                        kc,
                        ap.as_ptr(),
                        bp.as_ptr(),
                        c_avx2.as_mut_ptr(),
                        NR,
                        beta,
                        MR,
                        NR,
                        None,
                        PfWalk::NONE,
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

    /// The `A·Bᵀ` (`nn.Linear`) form against the f64 reference at the derived bound, plus the
    /// standing serial == parallel law. The last shape is DERIVED from `KC` so the K loop really
    /// blocks; every shape here had k <= 256 < KC = 384 before 2026-09-13.
    #[test]
    fn sgemm_nt_matches_naive() {
        for (m, k, n) in [
            (1usize, 1usize, 1usize),
            (7, 17, 13),
            (64, 64, 64),
            (100, 130, 96),
            (128, 256, 512),
            (40, 2 * KC + 37, 48),
        ] {
            let a = fill(11, m * k);
            let b = fill(12, n * k);
            let (want, growth) = reference_f64(&a, &b, m, k, n, true);
            let mut got = vec![0.0f32; m * n];
            let mut got_par = vec![0.0f32; m * n];
            // SAFETY: `a`/`b` sized m*k/k*n; `got`/`got_par` m*n.
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
            assert_within_derived_bound(&got, &want, &growth, m, k, n, None, "sgemm_nt");
            assert_eq!(got, got_par, "nt serial vs parallel ({m}x{k}x{n})");
        }
    }

    #[test]
    fn sgemm_tn_matches_naive_and_transposed_nn() {
        // (520,264,540) exceeds PAR_MIN_MACS (~74M MACs) so the multicore tn path actually runs, and
        // (40, 2*KC+37, 48) is DERIVED from `KC` so the K loop blocks; the rest straddle the MR=6 /
        // NR=16 remainders.
        for (m, k, n) in [
            (1usize, 1usize, 1usize),
            (5, 7, 3),
            (64, 64, 64),
            (100, 130, 96),
            (128, 256, 64),
            (40, 2 * KC + 37, 48),
            (520, 264, 540),
        ] {
            let a = fill(21, k * m); // A stored [k, m]
            let b = fill(22, k * n); // B stored [k, n]
            let (want, growth) = reference_tn_f64(&a, &b, m, k, n);
            let mut got = vec![0.0f32; m * n];
            let mut got_par = vec![0.0f32; m * n];
            // SAFETY: `a`/`b` sized m*k/k*n; `got`/`got_par` m*n.
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
            assert_within_derived_bound(&got, &want, &growth, m, k, n, None, "sgemm_tn");
            // Bit-exact against the NN kernel on a manually-transposed A — proves packing A straight
            // from its [k, m] storage yields exactly the NN kernel's bytes and K order.
            let mut at = vec![0.0f32; m * k];
            for p in 0..k {
                for i in 0..m {
                    at[i * k + p] = a[p * m + i];
                }
            }
            let mut nn = vec![0.0f32; m * n];
            // SAFETY: `nn` sized m*n; inputs sized for the shape.
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
            assert_eq!(
                got, nn,
                "tn must equal NN on manually-transposed A ({m}x{k}x{n})"
            );
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
            // SAFETY: the half-precision operands and `got_h`/`got_h_par` are sized m*k/k*n/m*n.
            unsafe {
                // Reference: the exact f32 NT kernel on the widened operands.
                wukong_sgemm_nt(
                    a_bf_f32.as_ptr(),
                    b_bf_f32.as_ptr(),
                    ref_bf.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
                wukong_sgemm_nt(
                    a_h_f32.as_ptr(),
                    b_h_f32.as_ptr(),
                    ref_h.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
                // bf16 / f16 kernels on the stored half-width bits.
                wukong_sgemm_bf16_nt(
                    a_bf.as_ptr(),
                    b_bf.as_ptr(),
                    got_bf.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
                wukong_sgemm_bf16_nt_parallel(
                    a_bf.as_ptr(),
                    b_bf.as_ptr(),
                    got_bf_par.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
                wukong_sgemm_f16_nt(
                    a_h.as_ptr(),
                    b_h.as_ptr(),
                    got_h.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
                wukong_sgemm_f16_nt_parallel(
                    a_h.as_ptr(),
                    b_h.as_ptr(),
                    got_h_par.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
            }
            assert_eq!(
                got_bf, ref_bf,
                "bf16 nt must equal f32 nt on widened operands ({m}x{k}x{n})"
            );
            assert_eq!(
                got_bf, got_bf_par,
                "bf16 nt serial vs parallel ({m}x{k}x{n})"
            );
            assert_eq!(
                got_h, ref_h,
                "f16 nt must equal f32 nt on widened operands ({m}x{k}x{n})"
            );
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
            // SAFETY: the half-precision operands and `got_h`/`got_h_par` are sized m*k/k*n/m*n.
            unsafe {
                // Reference: the exact f32 TN kernel on the widened operands.
                wukong_sgemm_tn(
                    a_bf_f32.as_ptr(),
                    b_bf_f32.as_ptr(),
                    ref_bf.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
                wukong_sgemm_tn(
                    a_h_f32.as_ptr(),
                    b_h_f32.as_ptr(),
                    ref_h.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
                // bf16 / f16 TN kernels on the stored half-width bits.
                wukong_sgemm_bf16_tn(
                    a_bf.as_ptr(),
                    b_bf.as_ptr(),
                    got_bf.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
                wukong_sgemm_bf16_tn_parallel(
                    a_bf.as_ptr(),
                    b_bf.as_ptr(),
                    got_bf_par.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
                wukong_sgemm_f16_tn(
                    a_h.as_ptr(),
                    b_h.as_ptr(),
                    got_h.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
                wukong_sgemm_f16_tn_parallel(
                    a_h.as_ptr(),
                    b_h.as_ptr(),
                    got_h_par.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0,
                );
            }
            assert_eq!(
                got_bf, ref_bf,
                "bf16 tn must equal f32 tn on widened operands ({m}x{k}x{n})"
            );
            assert_eq!(
                got_bf, got_bf_par,
                "bf16 tn serial vs parallel ({m}x{k}x{n})"
            );
            assert_eq!(
                got_h, ref_h,
                "f16 tn must equal f32 tn on widened operands ({m}x{k}x{n})"
            );
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
                let bptr = if use_bias {
                    bias.as_ptr()
                } else {
                    std::ptr::null()
                };
                let mut ref_bf = vec![0.0f32; m * n];
                let mut got_bf = vec![0.0f32; m * n];
                let mut got_bf_par = vec![0.0f32; m * n];
                let mut ref_h = vec![0.0f32; m * n];
                let mut got_h = vec![0.0f32; m * n];
                let mut got_h_par = vec![0.0f32; m * n];
                // SAFETY: `got_h`/`got_h_par` sized m*n; inputs sized for the shape.
                unsafe {
                    // Reference: the f32 fused-epilogue kernel on the widened operands.
                    wukong_sgemm_nt_epi(
                        a_bf_f32.as_ptr(),
                        b_bf_f32.as_ptr(),
                        ref_bf.as_mut_ptr(),
                        mi,
                        ki,
                        ni,
                        0,
                        bptr,
                        act,
                    );
                    wukong_sgemm_bf16_nt_epi(
                        a_bf.as_ptr(),
                        b_bf.as_ptr(),
                        got_bf.as_mut_ptr(),
                        mi,
                        ki,
                        ni,
                        0,
                        bptr,
                        act,
                    );
                    wukong_sgemm_bf16_nt_epi_parallel(
                        a_bf.as_ptr(),
                        b_bf.as_ptr(),
                        got_bf_par.as_mut_ptr(),
                        mi,
                        ki,
                        ni,
                        0,
                        bptr,
                        act,
                    );
                    wukong_sgemm_nt_epi(
                        a_h_f32.as_ptr(),
                        b_h_f32.as_ptr(),
                        ref_h.as_mut_ptr(),
                        mi,
                        ki,
                        ni,
                        0,
                        bptr,
                        act,
                    );
                    wukong_sgemm_f16_nt_epi(
                        a_h.as_ptr(),
                        b_h.as_ptr(),
                        got_h.as_mut_ptr(),
                        mi,
                        ki,
                        ni,
                        0,
                        bptr,
                        act,
                    );
                    wukong_sgemm_f16_nt_epi_parallel(
                        a_h.as_ptr(),
                        b_h.as_ptr(),
                        got_h_par.as_mut_ptr(),
                        mi,
                        ki,
                        ni,
                        0,
                        bptr,
                        act,
                    );
                }
                assert_eq!(got_bf, ref_bf, "bf16 nt_epi act={act} bias={use_bias}");
                assert_eq!(
                    got_bf, got_bf_par,
                    "bf16 nt_epi serial vs parallel act={act} bias={use_bias}"
                );
                assert_eq!(got_h, ref_h, "f16 nt_epi act={act} bias={use_bias}");
                assert_eq!(
                    got_h, got_h_par,
                    "f16 nt_epi serial vs parallel act={act} bias={use_bias}"
                );
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
        // SAFETY: `serial`/`par` sized m*n.
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
        // SAFETY: `serial_nt`/`par_nt` sized m*n.
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
        if !(crate::isa::avx2() && crate::isa::fma()) {
            println!("no avx2/fma — skipping");
            return;
        }
        for &(m, k, n) in &[(520usize, 900usize, 4200usize), (214, 901, 4160)] {
            assert!(
                (m * k * n) as u64 >= PAR_MIN_MACS,
                "shape must trip the parallel gate"
            );
            assert!(n > NC, "shape must span multiple NC blocks");
            assert!(k > select_kc(k), "shape must span multiple K blocks");
            let a = fill(71, m * k);
            let b = fill(72, k * n);
            let (mi, ki, ni) = (m as i64, k as i64, n as i64);
            let mut serial = vec![0.0f32; m * n];
            // SAFETY: `serial` sized m*n; inputs sized for the shape.
            unsafe {
                wukong_sgemm(a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(), mi, ki, ni, 0);
            }
            // Persistent region, called directly with the same pool selection gemm_dispatch uses.
            let mut per = vec![0.0f32; m * n];
            // SAFETY: `per` sized m*n (persistent region path).
            unsafe {
                let args = GemmArgs {
                    a: a.as_ptr(),
                    b: b.as_ptr(),
                    c: per.as_mut_ptr(),
                    m,
                    k,
                    n,
                    beta: 0.0,
                    at: false,
                    bt: false,
                    epi: None,
                };
                sgemm_persistent_region(gemm_pool(), args);
            }
            assert_eq!(serial, per, "persistent region != serial (m{m} k{k} n{n})");
            // Legacy fork-join path (what the WUKONG_GEMM_FORKJOIN=1 kill-switch routes to).
            let mut fj = vec![0.0f32; m * n];
            // SAFETY: `fj` sized m*n (fork-join path).
            unsafe {
                sgemm_avx2_parallel(
                    a.as_ptr(),
                    b.as_ptr(),
                    fj.as_mut_ptr(),
                    m,
                    k,
                    n,
                    0.0,
                    false,
                    false,
                    None,
                );
            }
            assert_eq!(serial, fj, "fork-join != serial (m{m} k{k} n{n})");
            // Public entry: routes to whichever path the process env selected — must equal serial
            // either way.
            let mut public = vec![0.0f32; m * n];
            // SAFETY: `public` sized m*n (public entry).
            unsafe {
                wukong_sgemm_parallel(a.as_ptr(), b.as_ptr(), public.as_mut_ptr(), mi, ki, ni, 0);
            }
            assert_eq!(
                serial, public,
                "public parallel entry != serial (m{m} k{k} n{n})"
            );
        }
        // Fused-epilogue leg: bias + ReLU on the nt form, across 3 K-blocks (uneven split) and 2 NC
        // blocks, on both parallel shapes.
        {
            let (m, k, n) = (520usize, 901usize, 4200usize);
            assert!((m * k * n) as u64 >= PAR_MIN_MACS && n > NC && k > select_kc(k));
            let a = fill(73, m * k);
            let b = fill(74, n * k);
            let bias = fill(75, n);
            let epi = Epilogue {
                bias: bias.as_ptr(),
                act: ACT_RELU,
                alpha: 1.0,
            };
            let mut serial = vec![0.0f32; m * n];
            let mut per = vec![0.0f32; m * n];
            let mut fj = vec![0.0f32; m * n];
            // SAFETY: `per`/`fj` sized m*n.
            unsafe {
                wukong_sgemm_nt_epi(
                    a.as_ptr(),
                    b.as_ptr(),
                    serial.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                    bias.as_ptr(),
                    ACT_RELU as i64,
                );
                let args = GemmArgs {
                    a: a.as_ptr(),
                    b: b.as_ptr(),
                    c: per.as_mut_ptr(),
                    m,
                    k,
                    n,
                    beta: 0.0,
                    at: false,
                    bt: true,
                    epi: Some(epi),
                };
                sgemm_persistent_region(gemm_pool(), args);
                sgemm_avx2_parallel(
                    a.as_ptr(),
                    b.as_ptr(),
                    fj.as_mut_ptr(),
                    m,
                    k,
                    n,
                    0.0,
                    false,
                    true,
                    Some(epi),
                );
            }
            assert_eq!(serial, per, "persistent nt+epi != serial");
            assert_eq!(serial, fj, "fork-join nt+epi != serial");
        }
    }

    /// The static-scheduled 2D block-parallel path (reached with `WUKONG_GEMM_DYN=0` — see
    /// [`sgemm_2d_blocks`],
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
        if !(crate::isa::avx2() && crate::isa::fma()) {
            println!("no avx2/fma — skipping");
            return;
        }
        // (m, k, n, bt combos, beta combos): the big multi-block shape runs once (bt=false,
        // beta=0); the mid shape sweeps the full bt × beta matrix; two degenerate grids.
        let shapes: &[(usize, usize, usize, &[bool], &[i64])] = &[
            (520, 901, 4200, &[false], &[0]), // multi-block both dims, 3 uneven K-blocks
            (214, 901, 616, &[false, true], &[0, 1]), // edge remainders, full bt × beta sweep
            (100, 130, 96, &[false, true], &[0, 1]), // single/degenerate grid
            (13, 700, 530, &[false, true], &[0, 1]), // skinny m: 1 row-block, several col-blocks
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
                    // SAFETY: `got2d` is a clone sized m*n (2D scheduler).
                    unsafe {
                        if bt {
                            wukong_sgemm_nt(
                                a.as_ptr(),
                                b.as_ptr(),
                                serial.as_mut_ptr(),
                                mi,
                                ki,
                                ni,
                                beta,
                            );
                        } else {
                            wukong_sgemm(
                                a.as_ptr(),
                                b.as_ptr(),
                                serial.as_mut_ptr(),
                                mi,
                                ki,
                                ni,
                                beta,
                            );
                        }
                        let args = GemmArgs {
                            a: a.as_ptr(),
                            b: b.as_ptr(),
                            c: got2d.as_mut_ptr(),
                            m,
                            k,
                            n,
                            beta: beta as f32,
                            at: false,
                            bt,
                            epi: None,
                        };
                        sgemm_2d_blocks(gemm_pool(), args);
                    }
                    assert_eq!(
                        serial, got2d,
                        "2d != serial (m{m} k{k} n{n} bt{bt} beta{beta})"
                    );
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
                    let bias_ptr = if use_bias {
                        bias.as_ptr()
                    } else {
                        std::ptr::null()
                    };
                    let mut serial = vec![0.0f32; m * n];
                    let mut got2d = vec![0.0f32; m * n];
                    // SAFETY: `serial`/`got2d` sized m*n.
                    unsafe {
                        wukong_sgemm_nt_epi(
                            a.as_ptr(),
                            b.as_ptr(),
                            serial.as_mut_ptr(),
                            m as i64,
                            k as i64,
                            n as i64,
                            0,
                            bias_ptr,
                            act as i64,
                        );
                        let args = GemmArgs {
                            a: a.as_ptr(),
                            b: b.as_ptr(),
                            c: got2d.as_mut_ptr(),
                            m,
                            k,
                            n,
                            beta: 0.0,
                            at: false,
                            bt: true,
                            epi: Some(Epilogue {
                                bias: bias_ptr,
                                act,
                                alpha: 1.0,
                            }),
                        };
                        sgemm_2d_blocks(gemm_pool(), args);
                    }
                    assert_eq!(
                        serial, got2d,
                        "2d nt+epi != serial (act{act} bias{use_bias})"
                    );
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
        // The skinny-M boundary is its own policy constant, pinned BY VALUE here so a retune is a
        // reviewed diff — and deliberately NOT `MC`: the A-panel cache block may move with the L2
        // without reclassifying which GEMMs get a single block row.
        assert_eq!(SKINNY_M_MAX, 144, "skinny-M policy boundary moved");
        for &m in &[13usize, 100, 128, 144, 145, 214, 300, 512, 520, 1024, 2048] {
            for &n in &[96usize, 128, 530, 616, 768, 1024, 3072, 4200] {
                for &target in &[1usize, 3, 12, 48, 96] {
                    let (bm, bn) = select_2d_block_shape(m, n, target);
                    assert!(
                        bm >= MR && bm % MR == 0,
                        "BM {bm} not MR-aligned (m{m} n{n} t{target})"
                    );
                    assert!(
                        bn >= NR && bn % NR == 0,
                        "BN {bn} not NR-aligned (m{m} n{n} t{target})"
                    );
                    if m <= SKINNY_M_MAX {
                        assert_eq!(
                            m.div_ceil(bm),
                            1,
                            "skinny m must give one block row (m{m} n{n} t{target})"
                        );
                    }
                    // Tail balance: after the rebalance, every non-edge block is full-size and the
                    // edge remainder is at least `block − unit` short of one extra block — i.e. the
                    // grid can't shrink a whole block dim further without changing the block count.
                    let (nbi, nbj) = (m.div_ceil(bm), n.div_ceil(bn));
                    assert!(
                        nbi * bm >= m && nbj * bn >= n,
                        "grid must cover the matrix (m{m} n{n} t{target})"
                    );
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
            assert_eq!(
                order.len(),
                nbi * nbj,
                "not a full enumeration (m{m} n{n} t{target})"
            );
            let mut seen = vec![false; nbi * nbj];
            for &t in &order {
                assert!(
                    !seen[t as usize],
                    "block {t} claimed twice (m{m} n{n} t{target})"
                );
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
        if !(crate::isa::avx2() && crate::isa::fma()) {
            println!("no avx2/fma — skipping");
            return;
        }
        // (m, k, n, bt combos, beta combos) — band notes relative to SHARED_MAX_MACS = 2^26.
        let shapes: &[(usize, usize, usize, &[bool], &[i64])] = &[
            (300, 300, 300, &[false], &[0]), // below band; odd square, ragged both dims
            (128, 700, 530, &[false], &[0]), // below band; skinny M, ragged n
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
                    // SAFETY: `serial` is a clone sized m*n.
                    unsafe {
                        if bt {
                            wukong_sgemm_nt(
                                a.as_ptr(),
                                b.as_ptr(),
                                serial.as_mut_ptr(),
                                mi,
                                ki,
                                ni,
                                beta,
                            );
                        } else {
                            wukong_sgemm(
                                a.as_ptr(),
                                b.as_ptr(),
                                serial.as_mut_ptr(),
                                mi,
                                ki,
                                ni,
                                beta,
                            );
                        }
                    }
                    for steal_order in [false, true] {
                        for skinny_a in [false, true] {
                            let mut got = base.clone();
                            // SAFETY: `got` is a clone sized m*n (skinny-A variants).
                            unsafe {
                                let args = GemmArgs {
                                    a: a.as_ptr(),
                                    b: b.as_ptr(),
                                    c: got.as_mut_ptr(),
                                    m,
                                    k,
                                    n,
                                    beta: beta as f32,
                                    at: false,
                                    bt,
                                    epi: None,
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
                        // SAFETY: `got` sized m*n (skinny-A variants).
                        unsafe {
                            let args = GemmArgs {
                                a: a.as_ptr(),
                                b: b.as_ptr(),
                                c: got.as_mut_ptr(),
                                m,
                                k,
                                n,
                                beta: 0.0,
                                at: false,
                                bt: true,
                                epi: Some(epi),
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
            // SAFETY: `serial` sized m*n (bias shift test).
            unsafe {
                wukong_sgemm_nt_epi(
                    a.as_ptr(),
                    b.as_ptr(),
                    serial.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                    bias.as_ptr(),
                    ACT_RELU as i64,
                );
            }
            run_dyn(
                Epilogue {
                    bias: bias.as_ptr(),
                    act: ACT_RELU,
                    alpha: 1.0,
                },
                &serial,
                "relu bias",
            );
            // No bias: the null base must stay null through both shifts.
            let mut serial_nb = vec![0.0f32; m * n];
            // SAFETY: `serial_nb` sized m*n (null bias stays null).
            unsafe {
                wukong_sgemm_nt_epi(
                    a.as_ptr(),
                    b.as_ptr(),
                    serial_nb.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                    std::ptr::null(),
                    ACT_SILU as i64,
                );
            }
            run_dyn(
                Epilogue {
                    bias: std::ptr::null(),
                    act: ACT_SILU,
                    alpha: 1.0,
                },
                &serial_nb,
                "silu nobias",
            );
            // α-scaled (the attention QKᵀ/√d form): folded on the final K-block only.
            let mut serial_alpha = vec![0.0f32; m * n];
            // SAFETY: `serial_alpha` sized m*n (alpha-scaled NT).
            unsafe {
                wukong_sgemm_nt_alpha(
                    a.as_ptr(),
                    b.as_ptr(),
                    serial_alpha.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                    0.125,
                );
            }
            run_dyn(
                Epilogue {
                    bias: std::ptr::null(),
                    act: ACT_IDENTITY,
                    alpha: 0.125,
                },
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
        if !(crate::isa::avx2() && crate::isa::fma()) {
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
                // SAFETY: the operands and output are sized for the shape.
                unsafe {
                    wukong_sgemm(
                        a.as_ptr(),
                        b.as_ptr(),
                        s.as_mut_ptr(),
                        m as i64,
                        k as i64,
                        n as i64,
                        0,
                    );
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
                    // SAFETY: `got` sized m*n (skinny-A variants).
                    unsafe {
                        let args = GemmArgs {
                            a: a.as_ptr(),
                            b: b.as_ptr(),
                            c: got.as_mut_ptr(),
                            m,
                            k,
                            n,
                            beta: 0.0,
                            at: false,
                            bt: false,
                            epi: None,
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
            // SAFETY: `st` sized m*n (static range-split scheduler).
            unsafe {
                let args = GemmArgs {
                    a: a.as_ptr(),
                    b: b.as_ptr(),
                    c: st.as_mut_ptr(),
                    m,
                    k,
                    n,
                    beta: 0.0,
                    at: false,
                    bt: false,
                    epi: None,
                };
                sgemm_2d_blocks(Some(&pool), args);
            }
            assert_eq!(
                serial, &st,
                "static 2d != serial (threads {t} m{m} k{k} n{n})"
            );
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
        if !(crate::isa::avx2() && crate::isa::fma()) {
            println!("no avx2/fma — skipping");
            return;
        }
        // (m, k, n, bt combos, beta combos).
        let shapes: &[(usize, usize, usize, &[bool], &[i64])] = &[
            (300, 300, 300, &[false], &[0]),  // odd square, nothing block-aligned
            (512, 1024, 512, &[false], &[0]), // k spans 3 select_kc groups (344+344+336)
            (128, 768, 768, &[false, true], &[0]), // skinny M: one block row (policy rule)
            (128, 3072, 768, &[false, true], &[0]), // skinny M, long K: 8 shared-pack rounds
            (768, 768, 128, &[false], &[0]),  // tall: many block rows, few columns
            (13, 700, 530, &[false, true], &[0]), // ragged m = 13 (one 18-high block row)
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
                    // SAFETY: `shared` is a clone sized m*n (2D shared).
                    unsafe {
                        if bt {
                            wukong_sgemm_nt(
                                a.as_ptr(),
                                b.as_ptr(),
                                serial.as_mut_ptr(),
                                mi,
                                ki,
                                ni,
                                beta,
                            );
                        } else {
                            wukong_sgemm(
                                a.as_ptr(),
                                b.as_ptr(),
                                serial.as_mut_ptr(),
                                mi,
                                ki,
                                ni,
                                beta,
                            );
                        }
                        let args = GemmArgs {
                            a: a.as_ptr(),
                            b: b.as_ptr(),
                            c: shared.as_mut_ptr(),
                            m,
                            k,
                            n,
                            beta: beta as f32,
                            at: false,
                            bt,
                            epi: None,
                        };
                        sgemm_2d_shared(gemm_pool(), args);
                    }
                    assert_eq!(
                        serial, shared,
                        "2d-shared != serial (m{m} k{k} n{n} bt{bt} beta{beta})"
                    );
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
            // SAFETY: `serial`/`blocks` sized m*n; `b` sized k*n.
            unsafe {
                wukong_sgemm_nt(
                    a.as_ptr(),
                    b.as_ptr(),
                    serial.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                );
                let args = GemmArgs {
                    a: a.as_ptr(),
                    b: b.as_ptr(),
                    c: blocks.as_mut_ptr(),
                    m,
                    k,
                    n,
                    beta: 0.0,
                    at: false,
                    bt: true,
                    epi: None,
                };
                sgemm_2d_blocks(gemm_pool(), args);
            }
            assert_eq!(
                serial, blocks,
                "2d per-block (kill-switch path) != serial on the skinny grid"
            );
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
                // SAFETY: `shared` sized m*n (epilogue shift).
                unsafe {
                    let args = GemmArgs {
                        a: a.as_ptr(),
                        b: b.as_ptr(),
                        c: shared.as_mut_ptr(),
                        m,
                        k,
                        n,
                        beta: 0.0,
                        at: false,
                        bt: true,
                        epi: Some(epi),
                    };
                    sgemm_2d_shared(gemm_pool(), args);
                }
                assert_eq!(
                    serial_ref,
                    &shared[..],
                    "2d-shared nt+epi != serial ({tag})"
                );
            };
            for &act in &[ACT_IDENTITY, ACT_RELU, ACT_GELU, ACT_SILU] {
                let mut serial = vec![0.0f32; m * n];
                // SAFETY: `serial` sized m*n (activation variants).
                unsafe {
                    wukong_sgemm_nt_epi(
                        a.as_ptr(),
                        b.as_ptr(),
                        serial.as_mut_ptr(),
                        m as i64,
                        k as i64,
                        n as i64,
                        0,
                        bias.as_ptr(),
                        act as i64,
                    );
                }
                run_shared(
                    Epilogue {
                        bias: bias.as_ptr(),
                        act,
                        alpha: 1.0,
                    },
                    &serial,
                    &format!("act{act} bias"),
                );
            }
            // No bias: the null bias base must stay null through both shifts.
            let mut serial = vec![0.0f32; m * n];
            // SAFETY: `serial` sized m*n (null bias).
            unsafe {
                wukong_sgemm_nt_epi(
                    a.as_ptr(),
                    b.as_ptr(),
                    serial.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                    std::ptr::null(),
                    ACT_RELU as i64,
                );
            }
            run_shared(
                Epilogue {
                    bias: std::ptr::null(),
                    act: ACT_RELU,
                    alpha: 1.0,
                },
                &serial,
                "relu nobias",
            );
            // α-scaled (the attention QKᵀ/√d form): folded on the final K-block only.
            let mut serial_alpha = vec![0.0f32; m * n];
            // SAFETY: `serial_alpha` sized m*n (alpha-scaled).
            unsafe {
                wukong_sgemm_nt_alpha(
                    a.as_ptr(),
                    b.as_ptr(),
                    serial_alpha.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                    0.125,
                );
            }
            run_shared(
                Epilogue {
                    bias: std::ptr::null(),
                    act: ACT_IDENTITY,
                    alpha: 0.125,
                },
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
            // SAFETY: bench: ap/bp/biasp/cp address buffers sized m*k/k*n/>=2*n/m*n (separate).
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
            // SAFETY: bench: the same ap/bp/biasp/cp buffers, fused epilogue.
            let fused = bench(&|| unsafe {
                wukong_sgemm_nt_epi(ap, bp, cp, m as i64, k as i64, n as i64, 0, biasp, 1);
                std::hint::black_box(cp);
            });
            // The `@parallel` fused FFN: same fold, spread across cores (used to be serial-only).
            // SAFETY: bench: the same buffers, parallel fused.
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

    /// The fused epilogue is **bit-identical** to the unfused sequence, not merely close to it:
    /// `wukong_sgemm_nt_epi(A, B, bias, act)` must equal `act(wukong_sgemm_nt(A, B) + bias)` to the
    /// last bit, for {identity, ReLU, GELU, SiLU} x {bias, no-bias}. That is the law the interpreter
    /// depends on when it uses the serial fused kernel as the oracle for a `@parallel` FFN, and it
    /// holds by construction: with `beta = 0` the value the final K-block writes back before the
    /// epilogue is exactly what the plain `nt` kernel stores, so fusing may not perturb it.
    ///
    /// It replaces a comparison against an f32 `naive_nt` at `1e-3*sqrt(k)`, which was blind twice
    /// over. (1) The tolerance could not see the vector epilogue at all: the full-6x16 writeback
    /// applies `_mm256_max_ps` / `gelu8` / `silu8` where the edge-tile path applies `f32::max` /
    /// `gelu1` / `silu1`, and a lane-for-lane divergence between those is far below `1e-3*sqrt(k)`.
    /// (2) With the pre-fix `fill` producing values in `[-1, 0)`, every C entry was a large positive
    /// sum and ReLU was *provably the identity* at every shape here (min base 2.54 / 9.85 / 63.02 /
    /// 52.70; 0 of 91, 4096, 1920 and 4000 entries clamped in both the bias and no-bias legs), so
    /// the four activation legs were four copies of a passthrough test - hence the explicit
    /// `clamped > 0` guard below.
    ///
    /// The `k = 300` / `257` shapes do NOT block the K loop and have not since KC went 256 -> 384
    /// (`select_kc(300) == 300`, `select_kc(257) == 260`), which silently vacated this test's stated
    /// purpose; `(40, 2*KC + 37, 48)` is derived from the live constant so a retune cannot vacate it
    /// again.
    #[test]
    fn sgemm_nt_epi_matches_reference() {
        for &(m, k, n) in &[
            (7usize, 17usize, 13usize),
            (64, 64, 64),
            (40, 300, 48),
            (50, 257, 80),
            (40, 2 * KC + 37, 48), // 3 K-blocks: the epilogue must fire on the LAST one only
        ] {
            let a = fill(11, m * k);
            let b = fill(12, n * k);
            let bias = fill(13, n);
            let (mi, ki, ni) = (m as i64, k as i64, n as i64);
            // The unfused leg - WITH its own external oracle. Without the bound below, every
            // assertion in this test compares one kernel to itself: the (identity, no-bias) leg is
            // then a self-comparison by construction (same kernel, same accumulator, so no
            // arithmetic defect in the sum can make it fail), and `(40, 300, 48)` and
            // `(50, 257, 80)` - which appear in no other reference-checked shape list in this crate
            // - would have no f64 reference anywhere in the repo. `sgemm_nt_matches_naive` does
            // this job at ITS shapes, not at these. With the bound in, the sum is checked against
            // f64 and the fusion law is checked against the sum, which is the pair this test needs.
            let (want_f64, growth) = reference_f64(&a, &b, m, k, n, true);
            let mut base = vec![0.0f32; m * n];
            // SAFETY: `base` sized m*n; inputs sized for the shape.
            unsafe {
                wukong_sgemm_nt(a.as_ptr(), b.as_ptr(), base.as_mut_ptr(), mi, ki, ni, 0);
            }
            assert_within_derived_bound(&base, &want_f64, &growth, m, k, n, None, "nt_epi base");
            for &act in &[ACT_IDENTITY, ACT_RELU, ACT_GELU, ACT_SILU] {
                for use_bias in [false, true] {
                    let bias_ptr = if use_bias {
                        bias.as_ptr()
                    } else {
                        std::ptr::null()
                    };
                    let bias_ref = if use_bias { Some(&bias[..]) } else { None };
                    let mut got = vec![0.0f32; m * n];
                    // SAFETY: `got` sized m*n; optional bias sized for the shape.
                    unsafe {
                        wukong_sgemm_nt_epi(
                            a.as_ptr(),
                            b.as_ptr(),
                            got.as_mut_ptr(),
                            mi,
                            ki,
                            ni,
                            0,
                            bias_ptr,
                            act as i64,
                        );
                    }
                    let mut clamped = 0usize;
                    for i in 0..m {
                        for j in 0..n {
                            let idx = i * n + j;
                            let pre = base[idx] + bias_ref.map_or(0.0, |bs| bs[j]);
                            if pre < 0.0 {
                                clamped += 1;
                            }
                            let want = epilogue_ref(base[idx], j, 1.0, bias_ref, act);
                            assert_eq!(
                                got[idx].to_bits(),
                                want.to_bits(),
                                "fused != unfused (m{m} k{k} n{n} act{act} bias{use_bias}) idx \
                                 {idx}: got {} want {want}",
                                got[idx]
                            );
                        }
                    }
                    assert!(
                        clamped > 0,
                        "no entry of C is negative at (m{m} k{k} n{n} bias{use_bias}) - the ReLU \
                         leg of this test is the identity and every activation is untested, which \
                         is exactly the state the `[-1, 0)` `fill` left it in"
                    );
                }
            }
        }
    }

    /// The fused epilogue on non-finite data, over a 12x32 C so every tile is a FULL 6x16 and the
    /// 256-bit vector writeback (`_mm256_add_ps` for the bias, `_mm256_max_ps` for the ReLU) is what
    /// runs - the path no test could reach before, because the generator could not produce a
    /// non-positive C, let alone a NaN - and over a 13x37 C whose row, column and corner edge tiles
    /// take the masked vector writeback (a 1-row tile, a 5-column `vmaskmovps` half, and both).
    ///
    /// Non-finite values live only in A and B, never in `bias`: both legs then compute the SAME
    /// accumulator bits (same kernel, same inputs), so any NaN reaching the epilogue comes from the
    /// accumulator and no comparison depends on which operand of an `addps` a payload is taken from.
    /// Restricted to identity and ReLU on purpose: whether `gelu8`/`silu8` mirror `gelu1`/`silu1` on
    /// NaN and +/-inf is `vmath`'s tail-match invariant, not this kernel's, and `vmath.rs` is not
    /// this change's to touch.
    #[test]
    fn nt_epi_epilogue_is_bit_exact_on_non_finite_data() {
        for (m, k, n) in [(12usize, 5usize, 32usize), (13, 5, 37)] {
            nt_epi_non_finite_case(m, k, n);
        }
    }

    fn nt_epi_non_finite_case(m: usize, k: usize, n: usize) {
        let mut a = fill(301, m * k);
        let mut b = fill(302, n * k);
        a[0] = f32::NAN;
        a[3] = f32::INFINITY;
        a[k] = f32::NEG_INFINITY;
        a[2 * k + 1] = -f32::NAN;
        b[2] = f32::INFINITY;
        b[k + 1] = f32::NEG_INFINITY;
        b[3 * k] = f32::NAN;
        let mut bias = fill(303, n);
        bias[0] = 0.0;
        bias[1] = -0.0;
        bias[2] = f32::MIN_POSITIVE;
        bias[3] = -f32::MIN_POSITIVE;
        let (mi, ki, ni) = (m as i64, k as i64, n as i64);
        let mut base = vec![0.0f32; m * n];
        // SAFETY: `base` sized m*n; inputs sized for the shape.
        unsafe {
            wukong_sgemm_nt(a.as_ptr(), b.as_ptr(), base.as_mut_ptr(), mi, ki, ni, 0);
        }
        assert!(
            base.iter().any(|v| v.is_nan()) && base.iter().any(|v| v.is_infinite()),
            "the non-finite operands did not reach C - this test would be vacuous"
        );
        for &act in &[ACT_IDENTITY, ACT_RELU] {
            for use_bias in [false, true] {
                let bias_ptr = if use_bias {
                    bias.as_ptr()
                } else {
                    std::ptr::null()
                };
                let bias_ref = if use_bias { Some(&bias[..]) } else { None };
                let mut got = vec![0.0f32; m * n];
                // SAFETY: `got` sized m*n; optional bias sized for the shape.
                unsafe {
                    wukong_sgemm_nt_epi(
                        a.as_ptr(),
                        b.as_ptr(),
                        got.as_mut_ptr(),
                        mi,
                        ki,
                        ni,
                        0,
                        bias_ptr,
                        act as i64,
                    );
                }
                for i in 0..m {
                    for j in 0..n {
                        let idx = i * n + j;
                        let want = epilogue_ref(base[idx], j, 1.0, bias_ref, act);
                        assert_eq!(
                            got[idx].to_bits(),
                            want.to_bits(),
                            "non-finite fused epilogue (act{act} bias{use_bias}) idx {idx}: got \
                             {} ({:08x}) want {want} ({:08x})",
                            got[idx],
                            got[idx].to_bits(),
                            want.to_bits()
                        );
                    }
                }
            }
        }
    }

    /// The vector epilogue's ReLU is `_mm256_max_ps(v, 0)` while the scalar-twin epilogue
    /// (`Epilogue::apply`) is `v > 0 ? v : 0`, and the crate's invariant #2 says `MAXPS` is not
    /// `f32::max` in general: it is `SRC1 > SRC2 ? SRC1 : SRC2`, returning SRC2 on a tie or a NaN, which
    /// is why `cumminmax.rs` writes its twin `(a > b) ? a : b` rather than `f32::max`. Against the
    /// one operand pair the GEMM actually uses - SRC2 = `+0.0` - they do coincide for every f32, and
    /// that is what the fused-vs-unfused law rests on, so it is pinned here over the hostile set
    /// rather than asserted in a comment.
    ///
    /// (`-0.0` is separately unreachable as an epilogue input through any public entry: the
    /// accumulator starts at `+0.0` and `fma(a, b, +0.0)` cannot return `-0.0`, `x + (+0.0)` is
    /// never `-0.0`, and `nt_epi` fixes `alpha = 1`. It is checked anyway.)
    ///
    /// Two parts, because the law alone is not the pin. Part (1) pins the LAW over the hostile set.
    /// Part (2) pins the CALL SITES by driving [`micro_6x16`] itself - the full-tile `wbe!` and the
    /// masked edge-tile writeback - against [`Epilogue::apply`] on the kernel's own accumulator
    /// bits. Until 2026-09-13 only part (1) existed, and it constructs its own intrinsic call, so
    /// an operand-order flip inside `act8!`
    /// (`_mm256_max_ps(zero, v)`, which returns every NaN unchanged) left this test - the one the
    /// production comment names as the pin - green.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn relu_epilogue_vector_fold_matches_the_scalar_fold() {
        if !(crate::isa::avx2() && crate::isa::fma()) {
            return; // no vector epilogue on this host; the scalar fold is the only one that runs
        }
        // (1) The law: over the one operand pair the GEMM uses (SRC2 = +0.0), MAXPS and
        // `f32::max` agree on every f32, including the two `-0.0` cases part (2) cannot deliver.
        let epi = Epilogue {
            bias: std::ptr::null(),
            act: ACT_RELU,
            alpha: 1.0,
        };
        for &v in &[
            0.0f32,
            -0.0,
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            f32::MAX,
            f32::MIN,
        ] {
            // SAFETY: avx2 just feature-detected; `bias` is null so `apply` dereferences nothing.
            let (scalar, vector) = unsafe {
                use std::arch::x86_64::*;
                let r = _mm256_max_ps(_mm256_set1_ps(v), _mm256_setzero_ps());
                let mut lanes = [0.0f32; 8];
                _mm256_storeu_ps(lanes.as_mut_ptr(), r);
                (epi.apply(v, 0), lanes[0])
            };
            assert_eq!(
                scalar.to_bits(),
                vector.to_bits(),
                "relu fold disagrees on {v} ({:08x}): scalar {scalar} ({:08x}) vs _mm256_max_ps \
                 {vector} ({:08x})",
                v.to_bits(),
                scalar.to_bits(),
                vector.to_bits()
            );
        }

        // (2) The PRODUCTION call sites, which part (1) cannot see. Part (1) builds its OWN
        // `_mm256_max_ps`, so flipping the operand order in `micro_6x16`'s `act8!` macros to
        // `_mm256_max_ps(_mm256_setzero_ps(), $v)` leaves it green while every NaN rides through the
        // real fused epilogue as a NaN. So drive the kernel instead. `c_raw` holds the accumulators
        // themselves (no epilogue, beta = 0, a full tile: the plain vector store). Then every
        // writeback the kernel has must equal `Epilogue::apply` - the scalar fold - applied to those
        // accumulator bits: the full-tile 256-bit `wbe!` (mr = MR, nr = NR) and the masked edge-tile
        // writeback on a row edge (mr = MR - 1), a column edge that masks the hi half (nr = NR - 3)
        // and a corner that masks the lo half and skips hi (mr = 2, nr = 5). Since 2026-10 the edge
        // tiles are 256-bit as well, so `apply` is computed here explicitly rather than taken from an
        // edge-tile call. A sentinel C pins that the masked stores write nothing outside the
        // `mr x nr` corner. Row-trimmed K loops leave each valid row's accumulator bit-identical to
        // the full tile's, which `c_raw` relies on.
        //
        // `kc` steps of `a = 1.0` against a B panel that is zero except on the LAST step put the
        // hostile value itself into lane `j`'s accumulator (`fma(1, 0, acc) = acc`, then
        // `fma(1, v, +0.0) = v`). `-0.0` is the one member of the set this cannot deliver, because
        // `fma(a, b, +0.0)` never returns `-0.0` - which is the same reason it is unreachable as an
        // epilogue input through any public entry, and why part (1) covers it directly.
        // GELU/SiLU are deliberately out: whether `gelu8`/`silu8` mirror `gelu1`/`silu1` on NaN and
        // +/-inf is `vmath`'s tail-match invariant, not this kernel's (same restriction as
        // `nt_epi_epilogue_is_bit_exact_on_non_finite_data`).
        let hostile: [f32; NR] = [
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            f32::MAX,
            f32::MIN,
            3.5,
            -3.5,
        ];
        const SENTINEL: u32 = 0x7FBA_D0D0; // a signalling-NaN pattern no writeback can produce
        let bias = fill(97, NR);
        for kc in [1usize, 3, 9] {
            let ap = vec![1.0f32; MR * kc];
            let mut bp = vec![0.0f32; NR * kc];
            bp[NR * (kc - 1)..].copy_from_slice(&hostile);
            let mut c_raw = vec![0.0f32; MR * NR];
            // SAFETY: avx2+fma feature-detected above; `ap`/`bp` hold the MR-per-step and NR-per-step
            // packed panels the kernel's contract requires for `kc` steps, and `ldc = NR` matches
            // the MR*NR C buffer.
            unsafe {
                micro_6x16(
                    kc,
                    ap.as_ptr(),
                    bp.as_ptr(),
                    c_raw.as_mut_ptr(),
                    NR,
                    0.0,
                    MR,
                    NR,
                    None,
                    PfWalk::NONE,
                );
            }
            for use_bias in [false, true] {
                for &act in &[ACT_IDENTITY, ACT_RELU] {
                    let e = Epilogue {
                        bias: if use_bias {
                            bias.as_ptr()
                        } else {
                            std::ptr::null()
                        },
                        act,
                        alpha: 1.0,
                    };
                    for &(mr, nr) in &[(MR, NR), (MR - 1, NR), (MR, NR - 3), (2, 5)] {
                        let mut c = vec![f32::from_bits(SENTINEL); MR * NR];
                        // SAFETY: as above; `bias`, when non-null, is NR long.
                        unsafe {
                            micro_6x16(
                                kc,
                                ap.as_ptr(),
                                bp.as_ptr(),
                                c.as_mut_ptr(),
                                NR,
                                0.0,
                                mr,
                                nr,
                                Some(e),
                                PfWalk::NONE,
                            );
                        }
                        for r in 0..MR {
                            // `hostile` is `[f32; NR]`, so this walks exactly `j in 0..NR`.
                            for (j, &lane) in hostile.iter().enumerate() {
                                let idx = r * NR + j;
                                let want = if r < mr && j < nr {
                                    // SAFETY: `j < NR` indexes the NR-long bias when non-null.
                                    unsafe { e.apply(c_raw[idx], j) }.to_bits()
                                } else {
                                    SENTINEL
                                };
                                assert_eq!(
                                    c[idx].to_bits(),
                                    want,
                                    "micro_6x16 writeback (mr{mr} nr{nr} kc{kc} act{act} \
                                     bias{use_bias}) row {r} col {j}: got {} ({:08x}), want \
                                     {want:08x}; accumulator {} from hostile lane {}",
                                    c[idx],
                                    c[idx].to_bits(),
                                    c_raw[idx],
                                    lane
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// **Invariant #1 for the flagship kernel.** Every other AVX2 family in this crate has a scalar
    /// twin it agrees with bit-for-bit (`reduce.rs`'s `scalar_matches_avx2_bit_for_bit` asserts
    /// `to_bits()` equality), and the GEMM was the one exception: [`sgemm_scalar`] summed the whole
    /// K into a single non-fused accumulator while [`sgemm_avx2`] blocks K by [`select_kc`] and uses
    /// `vfmadd`, its own doc comment recorded the divergence, and `grep sgemm_scalar` found exactly
    /// one test - against `naive` at `1e-3*sqrt(k)`, never against the AVX2 path. A machine without
    /// AVX2, or any `aarch64` build, therefore computed a different matmul from the same `.wk`
    /// source, and no gate in the repo could see it because interp and native share the dispatcher.
    ///
    /// Both kernels are private and take the same argument list, so this drives them directly over
    /// shape x `at` x `bt` x `beta` x epilogue and asserts `to_bits()` equality (`at` reads A from its
    /// `[k, m]` storage: the scalar twin indexes it in place, the AVX2 kernel packs it in place). The shapes cover a 1x1x1,
    /// remainder-only tiles, exactly one full 6x16 tile, an `m` past `MC = 144`, and two- and
    /// three-K-block reductions derived from `KC`. `alpha != 1` is paired only with `beta == 0`, the
    /// combination the ABI supports.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sgemm_scalar_matches_avx2_bit_for_bit() {
        if !(crate::isa::avx2() && crate::isa::fma()) {
            return; // nothing to compare against: this host runs `sgemm_scalar` for everything
        }
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (7, 17, 13), // remainder tiles only
            (6, 16, 5),  // exactly one FULL 6x16 tile (the vector writeback fast path)
            (13, 19, 4),
            (150, 20, 20),         // m past MC = 144
            (13, KC + 7, 19),      // 2 K-blocks
            (13, 2 * KC + 37, 19), // 3 K-blocks
        ] {
            let a = fill(201, m * k); // same length either way: `at` reads it as [k, m]
            let b = fill(202, k * n); // same length either way: `bt` reads it as [n, k]
            let bias = fill(203, n);
            let c_init = fill(204, m * n);
            for (at, bt) in [(false, false), (false, true), (true, false), (true, true)] {
                for &beta in &[0.0f32, 1.0] {
                    let mut epis: Vec<Option<Epilogue>> = vec![
                        None,
                        Some(Epilogue {
                            bias: std::ptr::null(),
                            act: ACT_RELU,
                            alpha: 1.0,
                        }),
                        Some(Epilogue {
                            bias: bias.as_ptr(),
                            act: ACT_IDENTITY,
                            alpha: 1.0,
                        }),
                        Some(Epilogue {
                            bias: bias.as_ptr(),
                            act: ACT_RELU,
                            alpha: 1.0,
                        }),
                        Some(Epilogue {
                            bias: bias.as_ptr(),
                            act: ACT_GELU,
                            alpha: 1.0,
                        }),
                        Some(Epilogue {
                            bias: bias.as_ptr(),
                            act: ACT_SILU,
                            alpha: 1.0,
                        }),
                    ];
                    if beta == 0.0 {
                        // alpha scales the beta-accumulated value, so a non-unit alpha is only
                        // meaningful with beta == 0 (see the debug_assert in `gemm_dispatch`).
                        epis.push(Some(Epilogue {
                            bias: bias.as_ptr(),
                            act: ACT_RELU,
                            alpha: -2.5,
                        }));
                    }
                    for epi in epis {
                        let mut cs = c_init.clone();
                        let mut cv = c_init.clone();
                        sgemm_scalar(
                            a.as_ptr(),
                            b.as_ptr(),
                            cs.as_mut_ptr(),
                            m,
                            k,
                            n,
                            beta,
                            at,
                            bt,
                            epi,
                        );
                        // SAFETY: avx2+fma just feature-detected; the buffers are sized m*k, k*n,
                        // m*n exactly as the kernel contract requires.
                        unsafe {
                            sgemm_avx2(
                                a.as_ptr(),
                                b.as_ptr(),
                                cv.as_mut_ptr(),
                                m,
                                k,
                                n,
                                beta,
                                at,
                                bt,
                                epi,
                            );
                        }
                        let (act, has_bias, alpha) = epi.map_or((u32::MAX, false, 1.0), |e| {
                            (e.act, !e.bias.is_null(), e.alpha)
                        });
                        for idx in 0..m * n {
                            assert_eq!(
                                cs[idx].to_bits(),
                                cv[idx].to_bits(),
                                "scalar twin != avx2 (m{m} k{k} n{n} at{at} bt{bt} beta{beta} act{act} \
                                 bias{has_bias} alpha{alpha}) idx {idx}: scalar {} ({:08x}) vs \
                                 avx2 {} ({:08x})",
                                cs[idx],
                                cs[idx].to_bits(),
                                cv[idx],
                                cv[idx].to_bits()
                            );
                        }
                    }
                }
            }
        }
    }

    /// [`sgemm_scalar`] - the portable twin, and the ONLY GEMM path on a target without AVX2+FMA
    /// (every non-`x86_64` build compiles the whole fast path out) - had no test at all before
    /// 2026-07: its sole caller is the feature dispatch in [`gemm_dispatch`], which on any AVX2 host
    /// takes the other branch, so `cargo test -p wukong_runtime` never executed a single line of it.
    /// Verified by mutation: transposing the two arms of its `bt` select (`*b.add(j*k+p)` vs
    /// `*b.add(p*n+j)`) left all 197 tests green while every `nn.Linear` off-AVX2 would return
    /// garbage. This calls the private kernel directly, so it runs on every target -
    /// `sgemm_scalar_matches_avx2_bit_for_bit` is the x86-only other half.
    ///
    /// Two contracts, checked separately, so that neither needs a fudge factor:
    ///
    ///   1. **the sum** - `epi = None` against the f64 reference at the derived
    ///      `gamma_r * (sum|a||b| + |beta*c|)` bound. What stood here was `1e-3*sqrt(k)` absolute
    ///      against an f32 `naive`, which contains no epsilon and accepts a GEMM that has truncated
    ///      every one of its inputs to half precision;
    ///   2. **the epilogue** - `epi = Some(..)` must be **bit-identical** to applying the same
    ///      `act(alpha*x + bias[j])` to the `epi = None` result, i.e. applied exactly once, after
    ///      the full K reduction, at the global column. No tolerance is involved.
    ///
    /// Covers the reachable argument space: `bt` x `beta in {0, 1}` x the four activations x
    /// bias/no-bias x `alpha`, at shapes exercising the 1-wide dims, the MR/NR remainders, and a
    /// three-K-block reduction derived from `KC` so a retune cannot vacate it. `alpha != 1` is
    /// paired only with `beta == 0`, the combination the ABI supports (see the `debug_assert` in
    /// [`gemm_dispatch`]).
    #[test]
    fn sgemm_scalar_fallback_matches_reference() {
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (7, 17, 13),
            (64, 64, 64),
            (13, 2 * KC + 37, 19),
        ] {
            let a = fill(81, m * k);
            let b = fill(82, k * n);
            let bias = fill(83, n);
            let c_init = fill(84, m * n);
            for bt in [false, true] {
                // `bt` reinterprets the same k*n elements as `[n, k]` rather than `[k, n]`.
                let (dot, growth) = reference_f64(&a, &b, m, k, n, bt);
                for &beta in &[0.0f32, 1.0] {
                    // (1) the sum.
                    let mut base = c_init.clone();
                    sgemm_scalar(
                        a.as_ptr(),
                        b.as_ptr(),
                        base.as_mut_ptr(),
                        m,
                        k,
                        n,
                        beta,
                        false,
                        bt,
                        None,
                    );
                    let want: Vec<f64> = if beta == 0.0 {
                        dot.clone()
                    } else {
                        dot.iter()
                            .zip(&c_init)
                            .map(|(d, c0)| d + f64::from(*c0))
                            .collect()
                    };
                    let beta_c = if beta == 0.0 { None } else { Some(&c_init[..]) };
                    assert_within_derived_bound(
                        &base,
                        &want,
                        &growth,
                        m,
                        k,
                        n,
                        beta_c,
                        &format!("sgemm_scalar (bt{bt} beta{beta})"),
                    );
                    // (2) the epilogue, bit-exactly, on top of that same sum.
                    let alphas: &[f32] = if beta == 0.0 { &[1.0, 0.125] } else { &[1.0] };
                    for &alpha in alphas {
                        for &act in &[ACT_IDENTITY, ACT_RELU, ACT_GELU, ACT_SILU] {
                            for use_bias in [false, true] {
                                if act == ACT_IDENTITY && !use_bias && alpha == 1.0 {
                                    continue; // that IS `epi = None`, already checked as (1)
                                }
                                let bias_ptr = if use_bias {
                                    bias.as_ptr()
                                } else {
                                    std::ptr::null()
                                };
                                let bias_ref = if use_bias { Some(&bias[..]) } else { None };
                                let mut got = c_init.clone();
                                sgemm_scalar(
                                    a.as_ptr(),
                                    b.as_ptr(),
                                    got.as_mut_ptr(),
                                    m,
                                    k,
                                    n,
                                    beta,
                                    false,
                                    bt,
                                    Some(Epilogue {
                                        bias: bias_ptr,
                                        act,
                                        alpha,
                                    }),
                                );
                                for i in 0..m {
                                    for j in 0..n {
                                        let idx = i * n + j;
                                        let expect =
                                            epilogue_ref(base[idx], j, alpha, bias_ref, act);
                                        assert_eq!(
                                            got[idx].to_bits(),
                                            expect.to_bits(),
                                            "scalar epilogue (m{m} k{k} n{n} bt{bt} beta{beta} \
                                             alpha{alpha} act{act} bias{use_bias}) idx {idx}: got \
                                             {} want {expect}",
                                            got[idx]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// α-scaled `C = alpha·(A·Bᵀ)` must equal the plain `nt` GEMM followed by a scalar `alpha·c` pass
    /// (the α multiply is one exact f32 op on the fully-reduced dot — the *only* difference from the
    /// unscaled kernel), for both `alpha == 1.0` (must be **byte-identical** to `nt`) and a general
    /// `alpha`; and the serial and parallel α kernels must be bit-for-bit identical.
    ///
    /// This doc used to claim `k = 300` "exceeds `KC` so the K loop blocks — verifying α is applied
    /// once, on the final K-block, not per-block". It has not since KC went 256 -> 384:
    /// `select_kc(300) == 300`, so `is_last_k` was true on the only iteration and the stated
    /// property was vacuous - a regression applying α per K-block would have passed. The
    /// `(40, 2*KC + 37, 48)` shape is derived from the live constant, so a retune cannot vacate it
    /// again.
    #[test]
    fn sgemm_nt_alpha_matches_scaled_nt_and_parallel() {
        for &(m, k, n) in &[
            (7usize, 17usize, 13usize),
            (64, 64, 64),
            (40, 300, 48),
            (40, 2 * KC + 37, 48), // 3 K-blocks: α must be applied on the last one only
            (520, 300, 400),
        ] {
            let a = fill(61, m * k);
            let b = fill(62, n * k);
            let (mi, ki, ni) = (m as i64, k as i64, n as i64);
            for &alpha in &[1.0f32, 0.125, -2.5] {
                // Reference: the exact `nt` GEMM, then a scalar `alpha·c` pass (same single f32 multiply
                // the α kernel folds into the writeback).
                let mut want = vec![0.0f32; m * n];
                // SAFETY: `want` sized m*n (alpha reference).
                unsafe {
                    wukong_sgemm_nt(a.as_ptr(), b.as_ptr(), want.as_mut_ptr(), mi, ki, ni, 0);
                }
                for v in &mut want {
                    *v *= alpha;
                }
                let mut got = vec![0.0f32; m * n];
                let mut got_par = vec![0.0f32; m * n];
                // SAFETY: `got`/`got_par` sized m*n.
                unsafe {
                    wukong_sgemm_nt_alpha(
                        a.as_ptr(),
                        b.as_ptr(),
                        got.as_mut_ptr(),
                        mi,
                        ki,
                        ni,
                        0,
                        alpha,
                    );
                    wukong_sgemm_nt_alpha_parallel(
                        a.as_ptr(),
                        b.as_ptr(),
                        got_par.as_mut_ptr(),
                        mi,
                        ki,
                        ni,
                        0,
                        alpha,
                    );
                }
                assert_eq!(
                    got, want,
                    "nt_alpha != alpha·nt (m{m} k{k} n{n} alpha{alpha})"
                );
                assert_eq!(
                    got, got_par,
                    "nt_alpha serial vs parallel (m{m} k{k} n{n} alpha{alpha})"
                );
            }
            // alpha == 1.0 must be byte-identical to the plain `nt` kernel (the multiply is skipped).
            let mut plain = vec![0.0f32; m * n];
            let mut a1 = vec![0.0f32; m * n];
            // SAFETY: `plain`/`a1` sized m*n (alpha==1 vs plain).
            unsafe {
                wukong_sgemm_nt(a.as_ptr(), b.as_ptr(), plain.as_mut_ptr(), mi, ki, ni, 0);
                wukong_sgemm_nt_alpha(a.as_ptr(), b.as_ptr(), a1.as_mut_ptr(), mi, ki, ni, 0, 1.0);
            }
            assert_eq!(
                plain, a1,
                "nt_alpha(1.0) must be byte-identical to nt (m{m} k{k} n{n})"
            );
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
            assert!(
                (m * k * n) as u64 >= PAR_MIN_MACS,
                "size must trip the parallel path"
            );
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
                    // SAFETY: `serial`/`par` sized m*n.
                    unsafe {
                        wukong_sgemm_nt_epi(
                            a.as_ptr(),
                            b.as_ptr(),
                            serial.as_mut_ptr(),
                            m as i64,
                            k as i64,
                            n as i64,
                            0,
                            bias_ptr,
                            act as i64,
                        );
                        wukong_sgemm_nt_epi_parallel(
                            a.as_ptr(),
                            b.as_ptr(),
                            par.as_mut_ptr(),
                            m as i64,
                            k as i64,
                            n as i64,
                            0,
                            bias_ptr,
                            act as i64,
                        );
                    }
                    assert_eq!(
                        serial, par,
                        "par-fused != serial-fused (m{m} k{k} n{n} act{act} bias{use_bias})"
                    );
                }
            }
        }
    }

    /// The transposing B-packer's AVX2 8x8 path must be **byte-identical** to the scalar
    /// element-at-a-time twin it replaced. Packing is pure data movement, so "byte-identical"
    /// is the whole contract: every packed f32 must carry its exact bit pattern, NaN payloads,
    /// signed zeros and denormals included, or the microkernel's products change and the
    /// interpreter-vs-native differential oracle breaks.
    ///
    /// The twin here is a literal transcription of the pre-vectorisation loop. Sizes cover the
    /// vector body (`kc` a multiple of 8), the scalar contraction tail (`kc % 8 != 0`, incl.
    /// `kc < 8` where the vector path never runs), full `NR`-wide panels and the partial last
    /// panel (which stays scalar and must still zero-fill lanes `ncols..NR`).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn pack_b_trans_panel_simd_matches_scalar_twin() {
        if !crate::isa::avx2() || !crate::isa::fma() {
            return;
        }
        /// The pre-vectorisation packer, verbatim.
        unsafe fn twin(
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
                let src = b.add((j0 + r) * ldb);
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
        // Values whose BITS must survive: quiet and signalling NaN with distinct payloads,
        // both infinities, both zeros, a denormal, and ordinary magnitudes.
        let specials = [
            f32::from_bits(0x7FC0_0000), // qNaN
            f32::from_bits(0x7FC0_DEAD), // qNaN, distinct payload
            f32::from_bits(0xFFA0_0BAD), // sNaN, sign set
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0f32,
            -0.0f32,
            f32::from_bits(0x0000_0001), // smallest denormal
            -1.5,
            3.25,
        ];
        for &(kc, nc, ldb) in &[
            (8usize, 16usize, 8usize),
            (8, 16, 37),
            (16, 32, 16),
            (24, 16, 100),
            (7, 16, 7),   // kc < 8: vector path never runs
            (13, 16, 64), // 8 vector steps + 5 scalar
            (32, 16, 32),
            (32, 20, 40), // partial last panel (ncols = 4)
            (17, 33, 64), // partial last panel (ncols = 1) + scalar tail
            (64, 48, 71),
        ] {
            let src: Vec<f32> = (0..nc * ldb)
                .map(|i| {
                    if i % 5 == 0 {
                        specials[(i / 5) % specials.len()]
                    } else {
                        (i as f32) * 0.5 - 7.0
                    }
                })
                .collect();
            for jp in 0..nc.div_ceil(NR) {
                let mut got = vec![f32::from_bits(0xDEAD_BEEF); kc * NR];
                let mut want = vec![f32::from_bits(0xDEAD_BEEF); kc * NR];
                // SAFETY: `got`/`want` hold kc*NR f32 (a packed B panel).
                unsafe {
                    pack_b_trans_panel(src.as_ptr(), ldb, kc, nc, jp, got.as_mut_ptr());
                    twin(src.as_ptr(), ldb, kc, nc, jp, want.as_mut_ptr());
                }
                let gb: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
                let wb: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
                assert_eq!(
                    gb, wb,
                    "packed bytes differ (kc={kc} nc={nc} ldb={ldb} jp={jp})"
                );
            }
        }
    }

    /// The A-packer's AVX2 6x8 path must be **byte-identical** to the scalar element-at-a-time
    /// twin it replaced — a literal transcription of the pre-vectorisation loop — over NaN
    /// payloads, signed zeros, infinities and a denormal. Each slab's stores overhang two floats
    /// into the next contraction row, so the panel buffer is pre-filled with a sentinel and
    /// extended eight rows past the packed region: a lane the packer forgot to write, or an
    /// overhang that escaped the panel, both show up as a byte difference against the twin.
    /// Sizes cover `kc < 8` (no slab), `kc == 8` and `kc == 16` (the last slab has no following row,
    /// so it must stay scalar), slabs plus a tail, and partial (`< MR`-row) panels.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn pack_a_panel_simd_matches_scalar_twin() {
        if !crate::isa::avx2() || !crate::isa::fma() {
            return;
        }
        /// The pre-vectorisation packer, verbatim.
        unsafe fn twin(
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
                let src = a.add((i0 + r) * lda);
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
        let specials = [
            f32::from_bits(0x7FC0_0000), // qNaN
            f32::from_bits(0x7FC0_DEAD), // qNaN, distinct payload
            f32::from_bits(0xFFA0_0BAD), // sNaN, sign set
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0f32,
            -0.0f32,
            f32::from_bits(0x0000_0001), // smallest denormal
            -1.5,
            3.25,
        ];
        const SENTINEL: u32 = 0xDEAD_BEEF;
        for &(kc, mc, lda) in &[
            (1usize, 6usize, 1usize),
            (7, 6, 7),    // kc < 8: no slab at all
            (8, 6, 8),    // one slab fits but has no following row: scalar only
            (9, 6, 9),    // one slab + a one-row tail that absorbs the overhang
            (16, 6, 19),  // two slabs fit; the second has no following row
            (17, 12, 40), // two slabs + tail, two full panels
            (24, 13, 24), // full panels + a one-row partial panel
            (33, 11, 50), // a five-row partial panel
            (64, 6, 71),
            (384, 18, 400),
        ] {
            let src: Vec<f32> = (0..mc * lda)
                .map(|i| {
                    if i % 5 == 0 {
                        specials[(i / 5) % specials.len()]
                    } else {
                        (i as f32) * 0.25 - 9.0
                    }
                })
                .collect();
            for ip in 0..mc.div_ceil(MR) {
                let mut got = vec![f32::from_bits(SENTINEL); (kc + 8) * MR];
                let mut want = vec![f32::from_bits(SENTINEL); (kc + 8) * MR];
                // SAFETY: `got`/`want` hold (kc+8)*MR f32 (a packed A panel).
                unsafe {
                    pack_a_panel(src.as_ptr(), lda, false, mc, kc, ip, got.as_mut_ptr());
                    twin(src.as_ptr(), lda, mc, kc, ip, want.as_mut_ptr());
                }
                let gb: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
                let wb: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
                assert_eq!(
                    gb, wb,
                    "packed A bytes differ (kc={kc} mc={mc} lda={lda} ip={ip})"
                );
                assert!(
                    gb[kc * MR..].iter().all(|&x| x == SENTINEL),
                    "the A pack wrote past its panel (kc={kc} mc={mc} ip={ip})"
                );
            }
        }
    }

    /// P13.4's licence: the row/column cache blocking (`mc`, `nc`) never changes a result bit. The
    /// serial five-loop is driven under the reference blocking and under blockings far from it --
    /// one micro-panel tall / wide, ragged multiples, and the 4x-reference corner the host
    /// derivation can reach -- through both B-pack schedules, both transposes, both `beta` rules and
    /// a bias+ReLU epilogue, over shapes with several row blocks, several column blocks and several
    /// K-blocks. Every output must match the reference bit-for-bit. Only after this holds may `mc`
    /// and `nc` follow the host (`derive_blocking`) or the environment (`WUKONG_GEMM_MC`/`_NC`).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn mc_nc_blocking_never_changes_a_bit() {
        if !(crate::isa::avx2() && crate::isa::fma()) {
            return;
        }
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let blockings = [
            Blocking { mc: MR, nc: NR },
            Blocking {
                mc: 2 * MR,
                nc: 3 * NR,
            },
            Blocking {
                mc: 7 * MR,
                nc: 5 * NR,
            },
            Blocking {
                mc: 4 * MC,
                nc: 4 * NC,
            },
            Blocking {
                mc: MC / 4,
                nc: 64 * NR,
            },
        ];
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (13, 37, 70),
            (61, 20, 500),
            (2 * MR + 5, KC + 41, 3 * NR + 9), // two K-blocks
            (150, 2 * KC + 9, 40),             // three K-blocks, m past MC
        ] {
            let a = fill(611, m * k);
            let b = fill(612, k * n);
            let bias = fill(613, n);
            let c_init = fill(614, m * n);
            for (at, bt, beta, with_epi) in [
                (false, false, 0.0f32, false),
                (true, true, 1.0f32, true),
                (false, true, 0.0f32, true),
            ] {
                let epi = with_epi.then_some(Epilogue {
                    bias: bias.as_ptr(),
                    act: ACT_RELU,
                    alpha: 1.0,
                });
                for fused in [false, true] {
                    let run = |blk: Blocking| {
                        let mut c = c_init.clone();
                        // SAFETY: avx2+fma detected above; a is m*k, b k*n, c m*n, bias n.
                        unsafe {
                            sgemm_avx2_impl(
                                a.as_ptr(),
                                b.as_ptr(),
                                c.as_mut_ptr(),
                                m,
                                k,
                                n,
                                beta,
                                at,
                                bt,
                                epi,
                                fused,
                                blk,
                            );
                        }
                        bits(&c)
                    };
                    let want = run(Blocking::REFERENCE);
                    for blk in blockings {
                        assert_eq!(
                            run(blk),
                            want,
                            "blocking {blk:?} moved a bit (m{m} k{k} n{n} at{at} bt{bt} \
                             beta{beta} epi{with_epi} fused{fused})"
                        );
                    }
                }
            }
        }
    }

    /// The two legacy parallel schedules walk `nc`-wide column blocks too (`WUKONG_GEMM_FORKJOIN=1`
    /// and the persistent region): under a non-reference `nc` they must still equal the serial
    /// reference bit-for-bit, epilogue included.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn legacy_parallel_nc_never_changes_a_bit() {
        if !(crate::isa::avx2() && crate::isa::fma()) {
            return;
        }
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let (m, k, n) = (37usize, KC + 23, 7 * NR + 3);
        let a = fill(621, m * k);
        let b = fill(622, n * k);
        let bias = fill(623, n);
        let epi = Some(Epilogue {
            bias: bias.as_ptr(),
            act: ACT_RELU,
            alpha: 1.0,
        });
        let mut want = vec![0.0f32; m * n];
        // SAFETY: avx2+fma detected above; a is m*k, b n*k (bt), c m*n, bias n.
        unsafe {
            sgemm_avx2_impl(
                a.as_ptr(),
                b.as_ptr(),
                want.as_mut_ptr(),
                m,
                k,
                n,
                0.0,
                false,
                true,
                epi,
                false,
                Blocking::REFERENCE,
            );
        }
        for nc in [NR, 3 * NR, 5 * NR, NC] {
            let mut fj = vec![0.0f32; m * n];
            let mut pr = vec![0.0f32; m * n];
            // SAFETY: as above; both legacy kernels take the same operand contract.
            unsafe {
                sgemm_avx2_parallel_nc(
                    a.as_ptr(),
                    b.as_ptr(),
                    fj.as_mut_ptr(),
                    m,
                    k,
                    n,
                    0.0,
                    false,
                    true,
                    epi,
                    nc,
                );
                let args = GemmArgs {
                    a: a.as_ptr(),
                    b: b.as_ptr(),
                    c: pr.as_mut_ptr(),
                    m,
                    k,
                    n,
                    beta: 0.0,
                    at: false,
                    bt: true,
                    epi,
                };
                sgemm_persistent_region_nc(gemm_pool(), args, nc);
            }
            assert_eq!(bits(&fj), bits(&want), "fork-join nc={nc}");
            assert_eq!(bits(&pr), bits(&want), "persistent region nc={nc}");
        }
    }

    /// `WUKONG_GEMM_MC` / `WUKONG_GEMM_NC` accept only positive multiples of `MR` / `NR` (bounded),
    /// and the host derivation is the identity on the reference part, scales with L2 (`mc`) and L3
    /// (`nc`), stays on the `MR`/`NR` grid and within 4x of the measured values.
    #[test]
    fn blocking_overrides_and_host_derivation() {
        assert_eq!(parse_block_override(Some("144"), MR), Some(144));
        assert_eq!(parse_block_override(Some("6"), MR), Some(6));
        assert_eq!(parse_block_override(Some("4080"), NR), Some(4080));
        for bad in [
            None,
            Some(""),
            Some("0"),
            Some("145"),
            Some("-6"),
            Some("+6"),
            Some(" 6"),
            Some("6k"),
            Some("70000"),
        ] {
            assert_eq!(parse_block_override(bad, MR), None, "{bad:?}");
        }
        assert_eq!(
            parse_block_override(Some("24"), NR),
            None,
            "24 is not a multiple of NR"
        );
        // Identity on the reference part, and on an unknown cache level.
        assert_eq!(
            derive_blocking(Some(REF_L2_BYTES), Some(REF_L3_BYTES)),
            Blocking::REFERENCE
        );
        assert_eq!(derive_blocking(None, None), Blocking::REFERENCE);
        assert_eq!(derive_blocking(Some(0), Some(0)), Blocking::REFERENCE);
        // Half the caches, half the panels (rounded down to the NR grid); the 4x clamps.
        assert_eq!(
            derive_blocking(Some(1 << 20), Some(12 << 20)),
            Blocking { mc: 72, nc: 2032 }
        );
        assert_eq!(
            derive_blocking(Some(1 << 40), Some(1 << 40)),
            Blocking {
                mc: 4 * MC,
                nc: 4 * NC
            }
        );
        assert_eq!(
            derive_blocking(Some(1), Some(1)),
            Blocking { mc: 36, nc: 1008 }
        );
        for l2 in [256 << 10, 1 << 20, 1280 << 10, 2 << 20, 3 << 20, 8 << 20] {
            for l3 in [4 << 20, 16 << 20, 24 << 20, 32 << 20, 96 << 20] {
                let d = derive_blocking(Some(l2), Some(l3));
                assert!(d.mc.is_multiple_of(MR) && d.mc >= MR, "{d:?}");
                assert!(d.nc.is_multiple_of(NR) && d.nc >= NR, "{d:?}");
            }
        }
        // The unit-test build pins the reference, so the shape tables above keep their meaning.
        assert_eq!(gemm_blocking(), Blocking::REFERENCE);
    }

    /// The fused micropanel-at-a-time B pack ([`macro_kernel_packb`], the default) must be
    /// **bit-identical** to the classic whole-panel pack, through every body that can take it: the
    /// serial kernel's single-row-block regime — both `bt` layouts, both `beta` rules, plain /
    /// bias+ReLU / bias-free GELU / α-scaled epilogues, one-, two- and three-K-block shapes, one and
    /// several `jc` blocks — and the 2D block body on an interior block. Both schedules are driven
    /// explicitly, so this does not depend on `WUKONG_GEMM_FUSED_PACK`. The `m > MC` shape is the
    /// control: the serial kernel never fuses there, so the two calls run identical code.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn fused_b_pack_is_bit_identical_to_the_classic_pack() {
        if !(crate::isa::avx2() && crate::isa::fma()) {
            return;
        }
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (5, 17, 13),
            (MR, KC + 7, 2 * NR),
            (2 * MR + 1, 2 * KC + 37, 3 * NR + 5),
            (2 * MR + 1, KC, 1400), // one row block, several `select_nc` column blocks
            (MC, 9, 40),            // the fused regime's upper edge, m == MC
            (MC + 1, 40, 70),
        ] {
            let a = fill(501, m * k); // same length either way: `at` reads it as [k, m]
            let b = fill(502, k * n); // same length either way: `bt` reads it as [n, k]
            let bias = fill(503, n);
            let c_init = fill(504, m * n);
            for (at, bt) in [(false, false), (false, true), (true, false)] {
                for &beta in &[0.0f32, 1.0] {
                    let mut epis = vec![
                        None,
                        Some(Epilogue {
                            bias: bias.as_ptr(),
                            act: ACT_RELU,
                            alpha: 1.0,
                        }),
                        Some(Epilogue {
                            bias: std::ptr::null(),
                            act: ACT_GELU,
                            alpha: 1.0,
                        }),
                    ];
                    if beta == 0.0 {
                        epis.push(Some(Epilogue {
                            bias: std::ptr::null(),
                            act: ACT_IDENTITY,
                            alpha: 0.125,
                        }));
                    }
                    for epi in epis {
                        let (mut fused, mut classic) = (c_init.clone(), c_init.clone());
                        // SAFETY: avx2+fma detected above; buffers sized m*k, k*n, m*n, bias n.
                        unsafe {
                            sgemm_avx2_impl(
                                a.as_ptr(),
                                b.as_ptr(),
                                fused.as_mut_ptr(),
                                m,
                                k,
                                n,
                                beta,
                                at,
                                bt,
                                epi,
                                true,
                                Blocking::REFERENCE,
                            );
                            sgemm_avx2_impl(
                                a.as_ptr(),
                                b.as_ptr(),
                                classic.as_mut_ptr(),
                                m,
                                k,
                                n,
                                beta,
                                at,
                                bt,
                                epi,
                                false,
                                Blocking::REFERENCE,
                            );
                        }
                        assert_eq!(
                            bits(&fused),
                            bits(&classic),
                            "fused != classic B pack (m{m} k{k} n{n} at{at} bt{bt} beta{beta} epi{})",
                            epi.is_some()
                        );
                    }
                }
            }
        }
        // The 2D block body on an interior block — nonzero row AND column origin, ragged extents,
        // two K-blocks — fused against classic, with everything outside the block untouched.
        let (m, k, n) = (40usize, KC + 9, 100usize);
        let (i0, j0, bm, bn) = (2 * MR, 2 * NR, 3 * MR + 2, 3 * NR + 2);
        let a = fill(505, m * k);
        let b = fill(506, k * n);
        let bias = fill(507, n);
        let c_init = fill(508, m * n);
        let epi = Some(Epilogue {
            bias: bias.as_ptr(),
            act: ACT_SILU,
            alpha: 1.0,
        });
        for bt in [false, true] {
            let mut ap = vec![0.0f32; round_up(bm, MR) * k.min(KC)];
            let mut bp = vec![0.0f32; round_up(bn, NR) * k.min(KC)];
            let (mut fused, mut classic) = (c_init.clone(), c_init.clone());
            for (out, flag) in [(&mut fused, true), (&mut classic, false)] {
                // SAFETY: avx2+fma detected above; the block lies inside the m×n C; `ap`/`bp`
                // are sized for the classic (larger) schedule.
                unsafe {
                    sgemm_2d_block(
                        a.as_ptr(),
                        b.as_ptr(),
                        out.as_mut_ptr(),
                        m,
                        k,
                        n,
                        1.0,
                        false,
                        bt,
                        epi,
                        i0,
                        j0,
                        bm,
                        bn,
                        ap.as_mut_ptr(),
                        bp.as_mut_ptr(),
                        flag,
                    );
                }
            }
            assert_eq!(
                bits(&fused),
                bits(&classic),
                "2D block: fused != classic B pack (bt{bt})"
            );
        }
    }

    /// [`PfWalk::for_b_panel`]'s parts must partition, in order, exactly the source rows the
    /// packer reads for that micropanel, and each row's walk must cover the row's whole byte span
    /// at any alignment. A wrong walk cannot change a result — it is a prefetch — which is exactly
    /// why no other test would ever notice one; this pins the address arithmetic directly. No
    /// address here is dereferenced.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn pf_walk_partitions_the_packed_source() {
        let buf = [0.0f32; 4];
        let base = buf.as_ptr().wrapping_add(3); // deliberately not 64-byte aligned
        for &(bt, kc, nc, ldb) in &[
            (true, 384usize, 50usize, 768usize),
            (true, 7, 16, 9),
            (false, 384, 50, 3072),
            (false, 13, 33, 33),
        ] {
            for jp in 0..nc.div_ceil(NR) {
                let j0 = jp * NR;
                let ncols = (nc - j0).min(NR);
                let (want, row_bytes): (Vec<usize>, usize) = if bt {
                    (
                        (0..ncols)
                            .map(|r| base.wrapping_add((j0 + r) * ldb) as usize)
                            .collect(),
                        4 * kc,
                    )
                } else {
                    (
                        (0..kc)
                            .map(|p| base.wrapping_add(p * ldb + j0) as usize)
                            .collect(),
                        4 * ncols,
                    )
                };
                for nparts in [1usize, 3, 22, 1000] {
                    let mut got: Vec<usize> = Vec::new();
                    for part in 0..nparts {
                        let w = PfWalk::for_b_panel(base, ldb, bt, kc, nc, jp, part, nparts);
                        assert_eq!(w.row_stride, ldb);
                        for r in 0..w.rows {
                            let row = w.ptr.wrapping_add(r * w.row_stride) as usize;
                            let (first_line, last_line) = (row / 64, (row + row_bytes - 1) / 64);
                            assert!(
                                last_line - first_line < w.lines,
                                "walk of {} lines misses the tail of a {row_bytes}-byte row \
                                 (bt{bt} kc{kc} nc{nc} jp{jp})",
                                w.lines
                            );
                            got.push(row);
                        }
                    }
                    assert_eq!(
                        got, want,
                        "walk parts do not partition the packed rows (bt{bt} kc{kc} nc{nc} jp{jp} \
                         nparts{nparts})"
                    );
                }
            }
        }
    }

    /// The transposed-A packer ([`pack_a_t_panel`], reached through `pack_a_panel(.., at = true, ..)`)
    /// must produce exactly the bytes the row-major packer produces from the explicitly transposed
    /// matrix — that byte identity is the whole reason the `Aᵀ·B` entries no longer build a
    /// transposed scratch copy. Driven from interior origins (nonzero row and contraction offsets,
    /// as the 2D blocks and later K-blocks use) with NaN payloads, signed zeros, infinities and a
    /// denormal in the data, over full and partial (`< MR`-row) panels.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn pack_a_transposed_matches_the_pack_of_the_transpose() {
        if !crate::isa::avx2() || !crate::isa::fma() {
            return;
        }
        let specials = [
            f32::from_bits(0x7FC0_DEAD),
            f32::from_bits(0xFFA0_0BAD),
            f32::INFINITY,
            -0.0f32,
            0.0,
            f32::from_bits(0x0000_0001),
        ];
        // (m, k) of the whole stored matrix; (i0, p0, mc, kc) of the packed block.
        for &(m, k, i0, p0, mc, kc) in &[
            (6usize, 1usize, 0usize, 0usize, 6usize, 1usize),
            (13, 20, 0, 0, 13, 20),
            (40, 50, 12, 7, 20, 33),
            (37, 400, 6, 16, 31, 384),
            (5, 9, 0, 1, 5, 8),
        ] {
            // A stored [k, m]; its explicit transpose [m, k].
            let stored: Vec<f32> = (0..k * m)
                .map(|t| {
                    if t % 7 == 0 {
                        specials[(t / 7) % specials.len()]
                    } else {
                        t as f32 * 0.5 - 11.0
                    }
                })
                .collect();
            let mut rowmajor = vec![0.0f32; m * k];
            for p in 0..k {
                for i in 0..m {
                    rowmajor[i * k + p] = stored[p * m + i];
                }
            }
            for ip in 0..mc.div_ceil(MR) {
                let mut got = vec![f32::from_bits(0xDEAD_BEEF); kc * MR];
                let mut want = vec![f32::from_bits(0xDEAD_BEEF); kc * MR];
                // SAFETY: avx2+fma detected; both origins address the same logical block inside
                // their buffers, and each panel buffer holds `kc*MR` floats.
                unsafe {
                    let (t_org, t_ld) = a_origin(stored.as_ptr(), m, k, true, i0, p0);
                    let (r_org, r_ld) = a_origin(rowmajor.as_ptr(), m, k, false, i0, p0);
                    assert_eq!((t_ld, r_ld), (m, k));
                    pack_a_panel(t_org, t_ld, true, mc, kc, ip, got.as_mut_ptr());
                    pack_a_panel(r_org, r_ld, false, mc, kc, ip, want.as_mut_ptr());
                }
                let gb: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
                let wb: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
                assert_eq!(
                    gb, wb,
                    "transposed-A pack != pack of the transpose (m{m} k{k} i0{i0} p0{p0} mc{mc} \
                     kc{kc} ip{ip})"
                );
            }
        }
    }

    /// `Aᵀ·B` packs A straight from its `[k, m]` storage in EVERY schedule — serial (fused and
    /// classic B pack), the scalar twin, the fork-join and persistent-region kernels, the static
    /// and dynamic 2D block paths (with every steal-order × shared-A combination) and the shared-pack
    /// 2D path — and each must be bit-identical to the plain NN kernel run on the explicitly
    /// transposed A, which is exactly what the TN entries computed when they still built that copy.
    /// Internal entries are driven directly, so no env knob or work threshold can route around one.
    /// Shapes: m > MC with two uneven K-blocks and ragged edges, a skinny single-block-row grid (the
    /// shared-A instrument's regime), and a tiny one; `beta` 0 and 1.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn tn_direct_pack_matches_nn_on_the_transpose_in_every_schedule() {
        if !(crate::isa::avx2() && crate::isa::fma()) {
            return;
        }
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        for &(m, k, n) in &[(150usize, KC + 7, 70usize), (128, 200, 260), (13, 9, 40)] {
            let stored = fill(601, k * m); // A stored [k, m]
            let mut rowmajor = vec![0.0f32; m * k];
            for p in 0..k {
                for i in 0..m {
                    rowmajor[i * k + p] = stored[p * m + i];
                }
            }
            let b = fill(602, k * n);
            let c_init = fill(603, m * n);
            for &beta in &[0.0f32, 1.0] {
                let mut want = c_init.clone();
                // SAFETY: (this block and the ones below) avx2+fma detected above; every buffer is
                // sized for the m×k / k×n / m×n operands its kernel contract names.
                unsafe {
                    sgemm_avx2_impl(
                        rowmajor.as_ptr(),
                        b.as_ptr(),
                        want.as_mut_ptr(),
                        m,
                        k,
                        n,
                        beta,
                        false,
                        false,
                        None,
                        false,
                        Blocking::REFERENCE,
                    );
                }
                let want = bits(&want);
                let check = |tag: &str, run: &mut dyn FnMut(*mut f32)| {
                    let mut got = c_init.clone();
                    run(got.as_mut_ptr());
                    assert_eq!(
                        bits(&got),
                        want,
                        "TN {tag} != NN on the transpose (m{m} k{k} n{n} beta{beta})"
                    );
                };
                let (a, bp) = (stored.as_ptr(), b.as_ptr());
                let args = |c: *mut f32| GemmArgs {
                    a,
                    b: bp,
                    c,
                    m,
                    k,
                    n,
                    beta,
                    at: true,
                    bt: false,
                    epi: None,
                };
                for fused in [false, true] {
                    // SAFETY: `a`/`bp` sized for m,k,n; `c` is the m*n output check provides (serial).
                    check(&format!("serial fused{fused}"), &mut |c| unsafe {
                        sgemm_avx2_impl(
                            a,
                            bp,
                            c,
                            m,
                            k,
                            n,
                            beta,
                            true,
                            false,
                            None,
                            fused,
                            Blocking::REFERENCE,
                        )
                    });
                }
                check("scalar twin", &mut |c| {
                    sgemm_scalar(a, bp, c, m, k, n, beta, true, false, None)
                });
                // SAFETY: `a`/`bp` sized for m,k,n; `c` is the m*n output (fork-join).
                check("fork-join", &mut |c| unsafe {
                    sgemm_avx2_parallel(a, bp, c, m, k, n, beta, true, false, None)
                });
                // SAFETY: `c` is the m*n output (persistent region).
                check("persistent region", &mut |c| unsafe {
                    sgemm_persistent_region(gemm_pool(), args(c))
                });
                // SAFETY: `c` is the m*n output (2D static).
                check("2D static", &mut |c| unsafe {
                    sgemm_2d_blocks(gemm_pool(), args(c))
                });
                for steal_order in [false, true] {
                    for skinny_a in [false, true] {
                        check(
                            &format!("2D dyn order{steal_order} skinnyA{skinny_a}"),
                            // SAFETY: `c` is the m*n output (2D dyn variants).
                            &mut |c| unsafe {
                                sgemm_2d_blocks_dyn(gemm_pool(), args(c), steal_order, skinny_a)
                            },
                        );
                    }
                }
                // SAFETY: `c` is the m*n output (2D shared).
                check("2D shared", &mut |c| unsafe {
                    sgemm_2d_shared(gemm_pool(), args(c))
                });
            }
        }
    }

    /// The SIMD widen bodies behind the bf16/f16 GEMM prepass must equal the scalar crate-root
    /// widen on EVERY one of the 65,536 bit patterns of both formats — signed zeros, every
    /// subnormal, both infinities, and every NaN payload (signalling ones included, which the f16
    /// conversion quiets identically on both paths). The length is not a multiple of 8, so the
    /// scalar tail runs too. On a host without the SIMD features this compares the scalar path
    /// with itself, which is the path that host runs.
    #[test]
    fn lowp_widen_simd_matches_scalar_for_every_bit_pattern() {
        let src: Vec<u16> = (0..=u16::MAX)
            .chain([0x8000, 0x7C01, 0xFF81, 1, 0x7F80])
            .collect();
        for kind in [HalfKind::Bf16, HalfKind::F16] {
            let mut simd = vec![f32::from_bits(0xDEAD_BEEF); src.len()];
            let mut scalar = vec![f32::from_bits(0xDEAD_BEEF); src.len()];
            // SAFETY: both destinations hold `src.len()` f32 and do not overlap the source.
            unsafe {
                widen_into_f32(src.as_ptr(), simd.as_mut_ptr(), src.len(), kind);
                widen_into_f32_scalar(src.as_ptr(), scalar.as_mut_ptr(), src.len(), kind);
            }
            for (t, (&x, &y)) in simd.iter().zip(&scalar).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{kind:?} widen of {:#06x}: simd {:08x} vs scalar {:08x}",
                    src[t],
                    x.to_bits(),
                    y.to_bits()
                );
            }
        }
    }

    /// **Dev instrument, `#[ignore]`d** — the phase breakdown of the serial NT five-loop at the
    /// skinny shapes the 2026-09 CPU evaluation lost on. Times the transposing B-pack, the
    /// A-pack and the macro-kernel separately, at exactly the block sizes `sgemm_avx2` would
    /// use, so "packing costs X% of this shape" is a measured quantity rather than a model.
    ///
    /// The timing thread is pinned to logical CPU 0 (a P-core on this 6P+8E+2LP-E part): an
    /// unpinned single-thread loop migrates onto a half-FMA-width Crestmont E-core mid-run and
    /// read the SAME macro-kernel call at 622 us and 727 us in one process before the pin.
    ///
    /// The `fused us` column is the `WUKONG_GEMM_FUSED_PACK` A/B in one process: packA plus the
    /// micropanel-at-a-time schedule ([`macro_kernel_packb`], B packed just before the sweep that
    /// consumes it, the next micropanel's source prefetched under it), directly comparable with
    /// `total us`; `-` where `m > MC`, where the serial kernel never fuses.
    ///
    /// Not a correctness gate and not a campaign number: it prints, asserts nothing, and is
    /// `#[ignore]`d so `cargo test --workspace` never spends time in it. Run it deliberately:
    /// `cargo test --release -p wukong_runtime --lib -- --ignored --nocapture gemm_phase_cost`
    #[test]
    #[ignore = "dev perf instrument; prints a phase breakdown, asserts nothing"]
    #[cfg(target_arch = "x86_64")]
    fn gemm_phase_cost() {
        use std::time::Instant;
        if !crate::isa::avx2() || !crate::isa::fma() {
            return;
        }
        pin_worker_to_cpu(0);
        println!("AGENT-LOCAL, DIRECTIONAL ONLY (best-of-5 batches per phase, warm-up discarded)");
        println!(
            "  {:<16} {:>4} {:>9} {:>9} {:>9} {:>9} {:>7} {:>9} {:>9}",
            "M x K x N",
            "jcb",
            "packB us",
            "packA us",
            "macro us",
            "total us",
            "pack%",
            "macro GF",
            "fused us"
        );
        for &(m, k, n, nc_cap) in &[
            (128usize, 768usize, 768usize, 4080usize),
            (128, 768, 3072, 4080),
            (128, 768, 3072, 768), // the L2-capped B-panel variant of the row above
            (128, 3072, 768, 4080),
            (512, 768, 768, 4080),
            (512, 3072, 768, 4080),
            (1536, 3072, 768, 4080),
        ] {
            let a = fill(1, m * k);
            let b = fill(2, n * k);
            let mut c = vec![0.0f32; m * n];
            let kc = select_kc(k);
            // Equal-ish column blocks under the cap, the way `sgemm_avx2` would walk them.
            let jcb = n.div_ceil(nc_cap.min(NC));
            let nc = round_up(n.div_ceil(jcb), NR).min(n);
            let mc = m.min(MC);
            let mut ap = vec![0.0f32; round_up(mc, MR) * kc];
            let mut bp = vec![0.0f32; round_up(nc, NR) * kc];
            let nkb = k.div_ceil(kc);
            let nib = m.div_ceil(MC);
            // Call counts the real five-loop makes: B once per (jc, pc), A and the macro
            // kernel once per (jc, pc, ic).
            let (cb, ca) = (jcb * nkb, jcb * nkb * nib);

            let best = |reps: usize, f: &mut dyn FnMut()| -> f64 {
                f();
                let mut lo = f64::INFINITY;
                for _ in 0..5 {
                    let t = Instant::now();
                    for _ in 0..reps {
                        f();
                    }
                    lo = lo.min(t.elapsed().as_secs_f64() / reps as f64);
                }
                lo
            };
            // SAFETY: the pack-B timing reads b (k*n) and writes a panel buffer sized for the shape.
            let tb = unsafe {
                best(cb.max(2), &mut || {
                    pack_b_block(b.as_ptr(), k, n, 0, 0, kc, nc, true, bp.as_mut_ptr())
                })
            } * cb as f64;
            // SAFETY: the pack-A timing reads a (m*k) and writes a panel buffer sized for the shape.
            let ta = unsafe {
                best(ca.max(2), &mut || {
                    pack_a(a.as_ptr(), k, false, mc, kc, ap.as_mut_ptr())
                })
            } * ca as f64;
            // SAFETY: the micro-timing operands are sized for the shape.
            let tm1 = unsafe {
                best(ca.max(2), &mut || {
                    macro_kernel(
                        mc,
                        nc,
                        kc,
                        ap.as_ptr(),
                        bp.as_ptr(),
                        c.as_mut_ptr(),
                        n,
                        1.0,
                        None,
                    )
                })
            };
            let tm = tm1 * ca as f64;
            let total = tb + ta + tm;
            // Macro-kernel throughput on the work it actually issues (mc x nc x kc per call),
            // NOT on the shape's nominal flops: the last ic block is shorter than `mc`.
            let macro_gf = 2.0 * mc as f64 * nc as f64 * kc as f64 / tm1 / 1e9;
            let fused = if m <= MC {
                let mut bpanel = vec![0.0f32; NR * kc];
                // SAFETY: `b` holds k*n f32; b_origin computes the panel origin within it.
                let (b_src, ldb) = unsafe { b_origin(b.as_ptr(), k, n, true, 0, 0) };
                // SAFETY: the fused micro-timing operands are sized for the shape.
                let tf1 = unsafe {
                    best(ca.max(2), &mut || {
                        macro_kernel_packb(
                            mc,
                            nc,
                            kc,
                            ap.as_ptr(),
                            b_src,
                            ldb,
                            true,
                            bpanel.as_mut_ptr(),
                            c.as_mut_ptr(),
                            n,
                            1.0,
                            None,
                        )
                    })
                };
                format!("{:9.0}", (tf1 * ca as f64 + ta) * 1e6)
            } else {
                format!("{:>9}", "-")
            };
            println!(
                "  {:<16} {jcb:>4} {:9.0} {:9.0} {:9.0} {:9.0} {:6.1}% {macro_gf:9.1} {fused}",
                format!("{m}x{k}x{n}"),
                tb * 1e6,
                ta * 1e6,
                tm * 1e6,
                total * 1e6,
                (tb + ta) / total * 100.0,
            );
        }
    }
}

#[cfg(test)]
mod relu_zero_contract_tests {
    use super::*;
    // Input bits and independently specified MAXPS(x, +0) result bits.
    const CASES: [(u32, u32); 12] = [
        (0x0000_0000, 0),
        (0x8000_0000, 0),
        (0x7fc1_2345, 0),
        (0xffc1_2345, 0),
        (0x7f80_0001, 0),
        (0xff80_0001, 0),
        (0x7f80_0000, 0x7f80_0000),
        (0xff80_0000, 0),
        (0x0000_0001, 0x0000_0001),
        (0x8000_0001, 0),
        (0x3fc0_0000, 0x3fc0_0000),
        (0xbfc0_0000, 0),
    ];

    #[test]
    fn relu_zero_contract_direct_epilogue() {
        // Direct application keeps -0 intact: a GEMM dot starting at +0 could erase it.
        let bias = [-0.0f32];
        for bias_ptr in [std::ptr::null(), bias.as_ptr()] {
            let epi = Epilogue {
                bias: bias_ptr,
                act: ACT_RELU,
                alpha: 1.0,
            };
            for (input, expected) in CASES {
                assert_eq!(
                    // SAFETY: apply reads only its scalar argument and the epilogue's own fields; no memory access.
                    unsafe { epi.apply(f32::from_bits(input), 0) }.to_bits(),
                    expected
                );
            }
            let scaled = Epilogue { alpha: 2.0, ..epi };
            // SAFETY: apply reads only its scalar argument and the epilogue's own fields; no memory access.
            assert_eq!(unsafe { scaled.apply(-0.0, 0) }.to_bits(), 0);
        }
    }
}
