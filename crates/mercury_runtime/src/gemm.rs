//! Single-precision GEMM — the heavy ML kernel.
//!
//! Computes `C[M,N] = A[M,K] · B[K,N]` (row-major f32) with a `beta` flag: `beta == 0` overwrites
//! `C`, `beta == 1` accumulates into it. This is the tuned microkernel the Mercury compiler lowers a
//! recognized matmul loop nest to (see `mercury_mir_build`'s matmul recognizer) — analogous to how
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
// overhead. 2^26 ≈ 67M MACs sits between 256^3 (~17M, faster serial) and 512^3 (~134M, ~2.4× on
// threads) on this machine.
const PAR_MIN_MACS: u64 = 1 << 26;

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
fn gemm_pool() -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let physical = num_cpus::get_physical().max(1);
        if physical >= rayon::current_num_threads() {
            return None; // no HyperThreads to shed (or single pool already this small)
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(physical)
            .thread_name(|i| format!("mercury-gemm-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// Smaller pool for the mid-size parallel regime (see [`gemm_pool_for`]). At 512³–1024³ every
/// fork-join waits for its slowest participant, and on a P+E hybrid that straggler is an E/LP-E
/// core arriving late and computing slowly once there — the barrier tax grows with worker count
/// while the compute win saturates once the fast cores are busy. Worker count from
/// `MERCURY_GEMM_MID_THREADS` (read once; the A/B knob), defaulting to `physical/2` (min 4) —
/// on the 6P+8E+2LPE dev box that is 8: the P-cores plus a little headroom, few enough that the
/// slowest cores never gate a barrier. Thread distribution never changes the bits (row-split,
/// fixed per-(i,j) K-order), so this is throughput-only. `None` ⇒ caller falls back to
/// [`gemm_pool`].
fn mid_pool() -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let physical = num_cpus::get_physical().max(1);
        let nt = std::env::var("MERCURY_GEMM_MID_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or((physical / 2).max(4));
        if nt >= physical {
            return None; // nothing to shed — use the full physical pool
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .thread_name(|i| format!("mercury-gemm-mid-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

// Above this many MACs the kernel has enough compute per K-block to feed every physical core
// past the barrier/straggler tax; below it the smaller `mid_pool` scales better. 2^31 puts
// 512³ (2^27) and 1024³ (2^30) in the mid regime and 2048³ (2^33) on the full pool — the sizes
// where each pool measured fastest against threaded oneMKL.
const MID_MAX_MACS: u64 = 1 << 31;

/// Regime-aware pool choice for one GEMM call: the mid-size pool below [`MID_MAX_MACS`], the full
/// physical pool above. Pool choice affects scheduling only, never results (see [`mid_pool`]).
fn gemm_pool_for(macs: u64) -> Option<&'static rayon::ThreadPool> {
    if macs < MID_MAX_MACS {
        if let Some(p) = mid_pool() {
            return Some(p);
        }
    }
    gemm_pool()
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
    /// Run the parallel kernel from this bundle. Taking `self` by value forces a closure that calls it
    /// to capture the whole (`Send`) `GemmArgs`, not the individual `!Send` raw-pointer fields that
    /// edition-2021 disjoint capture would otherwise grab.
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
// measurable fraction of a *small* GEMM (e.g. 256³, where it left Mercury ~8% behind a tuned
// library; large GEMMs are compute-bound and never noticed it). Taken out and put back around the
// kernel so no borrow is held across it.
#[cfg(target_arch = "x86_64")]
thread_local! {
    static PACK_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

// Fused-epilogue activation codes (shared with the compiler's recognizer in `mercury_mir_build`).
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
pub unsafe extern "C" fn mercury_sgemm(
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
pub unsafe extern "C" fn mercury_sgemm_nt(
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
pub unsafe extern "C" fn mercury_sgemm_nt_epi(
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
pub unsafe extern "C" fn mercury_sgemm_nt_alpha(
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
/// is bit-identical to the serial `mercury_sgemm_nt_alpha` the interpreter oracle calls.
///
/// # Safety
/// `a` valid for `m*k`, `b` for `n*k`, `c` for `m*n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn mercury_sgemm_nt_alpha_parallel(
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
/// Operand-size contract of [`mercury_sgemm`] / [`mercury_sgemm_nt`].
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
    let par = par && macs >= PAR_MIN_MACS;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features just checked; dims validated by the caller contract.
            unsafe {
                match (par, gemm_pool_for(macs)) {
                    // Parallel, with a private physical-core pool: run the whole kernel inside it so
                    // its nested `into_par_iter`s (pack + compute) use physical-core workers, not the
                    // default logical-core pool. The args cross the `install` boundary via `GemmArgs`
                    // (Send-wrapped); the closure runs to completion before `install` returns.
                    (true, Some(pool)) => {
                        let args = GemmArgs { a, b, c, m, k, n, beta, bt, epi };
                        // `args.run()` moves the whole bundle, so the closure captures the `Send`
                        // `GemmArgs` rather than its `!Send` fields (edition-2021 disjoint capture).
                        pool.install(move || args.run());
                    }
                    (true, None) => sgemm_avx2_parallel(a, b, c, m, k, n, beta, bt, epi),
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
/// parallel. Per-(i,j) accumulation order is identical to [`mercury_sgemm`], so the two agree
/// bit-for-bit (the interpreter oracle calls the serial one for both). Packing is shared, not
/// re-done per worker, so it scales to high core counts.
///
/// # Safety
/// Same contract as [`mercury_sgemm`].
#[no_mangle]
pub unsafe extern "C" fn mercury_sgemm_parallel(
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
/// Operand-size contract of [`mercury_sgemm_nt`].
#[no_mangle]
pub unsafe extern "C" fn mercury_sgemm_nt_parallel(
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
/// `@parallel` transformer FFN). Same epilogue contract as [`mercury_sgemm_nt_epi`]; the bias +
/// activation folds into each tile's final-K-block writeback, and each C tile is owned by exactly one
/// task with the same per-(i,j) accumulation order as the serial kernel — so it is bit-identical to
/// the serial `mercury_sgemm_nt_epi` the interpreter oracle calls. Below the work threshold it runs
/// the serial fused path. This lets a `@parallel` FFN combine the fusion *and* all the cores (before,
/// the epilogue forced serial, so a fused FFN could use one or the other but not both).
///
/// # Safety
/// `a`, `b`, `c` valid for `m*k`, `n*k`, `m*n` `f32`; `bias` null or valid for `n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn mercury_sgemm_nt_epi_parallel(
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
/// defeat vectorization (one cache line per element); Mercury transposes A into scratch once — O(m·k),
/// ~1/n of the O(m·n·k) GEMM — then runs the identical tuned `C = A·B` kernel. Reusing *that exact
/// kernel* means it is bit-for-bit the differential contract the interpreter marshals (no new
/// accumulation order to keep in sync). `beta`: 0 overwrites C, else accumulates.
///
/// # Safety
/// `a` valid for `k*m`, `b` for `k*n`, `c` for `m*n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn mercury_sgemm_tn(
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
/// Operand-size contract of [`mercury_sgemm_tn`].
#[no_mangle]
pub unsafe extern "C" fn mercury_sgemm_tn_parallel(
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
pub unsafe extern "C" fn mercury_sgemm_bf16_nt(
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
/// Operand-size contract of [`mercury_sgemm_bf16_nt`].
#[no_mangle]
pub unsafe extern "C" fn mercury_sgemm_bf16_nt_parallel(
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
pub unsafe extern "C" fn mercury_sgemm_f16_nt(
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
/// Operand-size contract of [`mercury_sgemm_f16_nt`].
#[no_mangle]
pub unsafe extern "C" fn mercury_sgemm_f16_nt_parallel(
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
/// Reuses the exact `mercury_sgemm_nt_epi` epilogue, so the fused result equals the unfused
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
pub unsafe extern "C" fn mercury_sgemm_bf16_nt_epi(
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
/// Operand-size contract of [`mercury_sgemm_bf16_nt_epi`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn mercury_sgemm_bf16_nt_epi_parallel(
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
pub unsafe extern "C" fn mercury_sgemm_f16_nt_epi(
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
/// Operand-size contract of [`mercury_sgemm_f16_nt_epi`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn mercury_sgemm_f16_nt_epi_parallel(
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
/// `O(m·n·k)` GEMM), then delegate to the f32 `mercury_sgemm_tn`, which transposes A once and runs the
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
        mercury_sgemm_tn_parallel(af.as_ptr(), bf.as_ptr(), c, m, k, n, beta);
    } else {
        mercury_sgemm_tn(af.as_ptr(), bf.as_ptr(), c, m, k, n, beta);
    }
}

/// `C = Aᵀ·B` with `bf16` inputs (f32 accumulate), single-threaded. See [`gemm_lowp_tn`].
///
/// # Safety
/// `a` valid for `k*m`, `b` for `k*n` `bf16` (`u16`); `c` for `m*n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn mercury_sgemm_bf16_tn(
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
/// Operand-size contract of [`mercury_sgemm_bf16_tn`].
#[no_mangle]
pub unsafe extern "C" fn mercury_sgemm_bf16_tn_parallel(
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
pub unsafe extern "C" fn mercury_sgemm_f16_tn(
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
/// Operand-size contract of [`mercury_sgemm_f16_tn`].
#[no_mangle]
pub unsafe extern "C" fn mercury_sgemm_f16_tn_parallel(
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
// 16-worker region costs just to open and close. Env-overridable (`MERCURY_PACK_PAR_MIN_KB`,
// read once) so the crossover stays A/B-measurable on other machines.
fn pack_par_min_bytes() -> usize {
    use std::sync::OnceLock;
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("MERCURY_PACK_PAR_MIN_KB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(256)
            * 1024
    })
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
    let mut p = 0;
    while p + 4 <= kc {
        _mm_prefetch::<_MM_HINT_T0>(bp.add(NR * 8) as *const i8);
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
    // Same ×4 K-unroll + one prefetch per 4 steps as the AVX2 kernel.
    let mut p = 0;
    while p + 4 <= kc {
        _mm_prefetch::<_MM_HINT_T0>(bp.add(NR * 8) as *const i8);
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
            mercury_sgemm(
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
            mercury_sgemm(
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

    /// Throughput probe (run: `cargo test -p mercury_runtime --release -- --ignored --nocapture`).
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
                mercury_sgemm(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
            bench("sgemm_nt (1 core)", &|| unsafe {
                mercury_sgemm_nt(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
            restore_affinity(prev); // parallel benches want all cores
            bench("sgemm (parallel)", &|| unsafe {
                mercury_sgemm_parallel(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
            bench("sgemm_nt (parallel)", &|| unsafe {
                mercury_sgemm_nt_parallel(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
        }
    }

    /// Adjacent A/B: the SAME parallel GEMM run in the default (logical-core) rayon pool vs a private
    /// physical-core pool, measured back-to-back best-of-N so the laptop's thermal drift cancels in the
    /// ratio. Cross-RUN comparison of the two pools is hopeless (the power state swings ~3×, and the
    /// 1c-vs-par self-scaling ratio is dominated by *where* in the thermal cycle each leg is sampled);
    /// only this interleaved same-run ratio reliably answers whether shedding the HyperThread siblings
    /// lifts throughput on this compute-bound kernel. Run: `cargo test -p mercury_runtime --release
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
                mercury_sgemm_nt(
                    a.as_ptr(),
                    b.as_ptr(),
                    got.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                );
                mercury_sgemm_nt_parallel(
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
                mercury_sgemm_tn(
                    a.as_ptr(),
                    b.as_ptr(),
                    got.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                );
                mercury_sgemm_tn_parallel(
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
                mercury_sgemm(
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
                mercury_sgemm_nt(a_bf_f32.as_ptr(), b_bf_f32.as_ptr(), ref_bf.as_mut_ptr(), mi, ki, ni, 0);
                mercury_sgemm_nt(a_h_f32.as_ptr(), b_h_f32.as_ptr(), ref_h.as_mut_ptr(), mi, ki, ni, 0);
                // bf16 / f16 kernels on the stored half-width bits.
                mercury_sgemm_bf16_nt(a_bf.as_ptr(), b_bf.as_ptr(), got_bf.as_mut_ptr(), mi, ki, ni, 0);
                mercury_sgemm_bf16_nt_parallel(a_bf.as_ptr(), b_bf.as_ptr(), got_bf_par.as_mut_ptr(), mi, ki, ni, 0);
                mercury_sgemm_f16_nt(a_h.as_ptr(), b_h.as_ptr(), got_h.as_mut_ptr(), mi, ki, ni, 0);
                mercury_sgemm_f16_nt_parallel(a_h.as_ptr(), b_h.as_ptr(), got_h_par.as_mut_ptr(), mi, ki, ni, 0);
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
                mercury_sgemm_tn(a_bf_f32.as_ptr(), b_bf_f32.as_ptr(), ref_bf.as_mut_ptr(), mi, ki, ni, 0);
                mercury_sgemm_tn(a_h_f32.as_ptr(), b_h_f32.as_ptr(), ref_h.as_mut_ptr(), mi, ki, ni, 0);
                // bf16 / f16 TN kernels on the stored half-width bits.
                mercury_sgemm_bf16_tn(a_bf.as_ptr(), b_bf.as_ptr(), got_bf.as_mut_ptr(), mi, ki, ni, 0);
                mercury_sgemm_bf16_tn_parallel(a_bf.as_ptr(), b_bf.as_ptr(), got_bf_par.as_mut_ptr(), mi, ki, ni, 0);
                mercury_sgemm_f16_tn(a_h.as_ptr(), b_h.as_ptr(), got_h.as_mut_ptr(), mi, ki, ni, 0);
                mercury_sgemm_f16_tn_parallel(a_h.as_ptr(), b_h.as_ptr(), got_h_par.as_mut_ptr(), mi, ki, ni, 0);
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
                    mercury_sgemm_nt_epi(a_bf_f32.as_ptr(), b_bf_f32.as_ptr(), ref_bf.as_mut_ptr(), mi, ki, ni, 0, bptr, act);
                    mercury_sgemm_bf16_nt_epi(a_bf.as_ptr(), b_bf.as_ptr(), got_bf.as_mut_ptr(), mi, ki, ni, 0, bptr, act);
                    mercury_sgemm_bf16_nt_epi_parallel(a_bf.as_ptr(), b_bf.as_ptr(), got_bf_par.as_mut_ptr(), mi, ki, ni, 0, bptr, act);
                    mercury_sgemm_nt_epi(a_h_f32.as_ptr(), b_h_f32.as_ptr(), ref_h.as_mut_ptr(), mi, ki, ni, 0, bptr, act);
                    mercury_sgemm_f16_nt_epi(a_h.as_ptr(), b_h.as_ptr(), got_h.as_mut_ptr(), mi, ki, ni, 0, bptr, act);
                    mercury_sgemm_f16_nt_epi_parallel(a_h.as_ptr(), b_h.as_ptr(), got_h_par.as_mut_ptr(), mi, ki, ni, 0, bptr, act);
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
        // The size must exceed PAR_MIN_MACS, or mercury_sgemm_parallel falls back to the serial
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
            mercury_sgemm(
                a.as_ptr(),
                b.as_ptr(),
                serial.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
                0,
            );
            mercury_sgemm_parallel(
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
            mercury_sgemm_nt(
                a.as_ptr(),
                b.as_ptr(),
                serial_nt.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
                0,
            );
            mercury_sgemm_nt_parallel(
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

    /// Probe (run: `cargo test -p mercury_runtime --release -- --ignored --nocapture epi_throughput`).
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
                mercury_sgemm_nt(ap, bp, cp, m as i64, k as i64, n as i64, 0);
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
                mercury_sgemm_nt_epi(ap, bp, cp, m as i64, k as i64, n as i64, 0, biasp, 1);
                std::hint::black_box(cp);
            });
            // The `@parallel` fused FFN: same fold, spread across cores (used to be serial-only).
            let fused_par = bench(&|| unsafe {
                mercury_sgemm_nt_epi_parallel(ap, bp, cp, m as i64, k as i64, n as i64, 0, biasp, 1);
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
                        mercury_sgemm_nt_epi(
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
                    mercury_sgemm_nt(a.as_ptr(), b.as_ptr(), want.as_mut_ptr(), mi, ki, ni, 0);
                }
                for v in &mut want {
                    *v *= alpha;
                }
                let mut got = vec![0.0f32; m * n];
                let mut got_par = vec![0.0f32; m * n];
                unsafe {
                    mercury_sgemm_nt_alpha(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), mi, ki, ni, 0, alpha);
                    mercury_sgemm_nt_alpha_parallel(
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
                mercury_sgemm_nt(a.as_ptr(), b.as_ptr(), plain.as_mut_ptr(), mi, ki, ni, 0);
                mercury_sgemm_nt_alpha(a.as_ptr(), b.as_ptr(), a1.as_mut_ptr(), mi, ki, ni, 0, 1.0);
            }
            assert_eq!(plain, a1, "nt_alpha(1.0) must be byte-identical to nt (m{m} k{k} n{n})");
        }
    }

    /// The parallel fused epilogue (`mercury_sgemm_nt_epi_parallel`) must be **bit-for-bit** identical
    /// to the serial `mercury_sgemm_nt_epi` (each C tile is owned by one task and the per-(i,j)
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
                        mercury_sgemm_nt_epi(
                            a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(),
                            m as i64, k as i64, n as i64, 0, bias_ptr, act as i64,
                        );
                        mercury_sgemm_nt_epi_parallel(
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
