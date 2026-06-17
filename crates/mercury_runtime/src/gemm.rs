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
// Cache-block sizes: A panel (MC×KC) targets L2, B panel (KC×NC) targets L3. Multiples of MR/NR.
const MC: usize = 72;
const KC: usize = 256;
const NC: usize = 4080;

// Minimum multiply-accumulate count (`m·n·k`) before the parallel kernel is worth its threading
// overhead. 2^26 ≈ 67M MACs sits between 256^3 (~17M, faster serial) and 512^3 (~134M, ~2.4× on
// threads) on this machine.
const PAR_MIN_MACS: u64 = 1 << 26;

#[inline]
fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
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
    gemm_dispatch(a, b, c, m, k, n, beta, false, false);
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
    gemm_dispatch(a, b, c, m, k, n, beta, true, false);
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
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let beta = beta as f32;
    // Multi-thread only above a work threshold: below it, cross-core wake/sync (worse on this
    // P+E-core hybrid, where the E-cores are slow to spin up) costs more than it saves and the
    // parallel path is a net loss — empirically ~256^3 runs faster on one core than across all of
    // them. The serial AVX2 kernel is the fast path for everything smaller.
    let par = par && (m as u64 * n as u64 * k as u64) >= PAR_MIN_MACS;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features just checked; dims validated by the caller contract.
            unsafe {
                if par {
                    sgemm_avx2_parallel(a, b, c, m, k, n, beta, bt);
                } else {
                    sgemm_avx2(a, b, c, m, k, n, beta, bt);
                }
            }
            return;
        }
    }
    sgemm_scalar(a, b, c, m, k, n, beta, bt);
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
    gemm_dispatch(a, b, c, m, k, n, beta, false, true);
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
    gemm_dispatch(a, b, c, m, k, n, beta, true, true);
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
) {
    use rayon::prelude::*;
    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let mut pc = 0;
        while pc < k {
            let kc = (k - pc).min(KC);
            let beta_eff = if pc == 0 { beta } else { 1.0 };
            // Pack the full A column-panel (m×kc) and B row-panel (kc×nc) once — in parallel, since
            // with the C compute spread across every core the serial pack would dominate (Amdahl).
            let mut ap = vec![0.0f32; round_up(m, MR) * kc];
            let mut bp = vec![0.0f32; round_up(nc, NR) * kc];
            pack_a_par(a.add(pc), k, m, kc, ap.as_mut_ptr());
            pack_b_block_par(b, k, n, pc, jc, kc, nc, bt, bp.as_mut_ptr());

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
                        );
                    }
                }
            });
            pc += KC;
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
                *crow.add(j) += acc;
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
) {
    // Pack scratch, sized to the actual blocks needed (never larger than the cache-block caps).
    let kc_max = k.min(KC);
    let mc_max = m.min(MC);
    let nc_max = n.min(NC);
    let mut ap = vec![0.0f32; round_up(mc_max, MR) * kc_max];
    let mut bp = vec![0.0f32; round_up(nc_max, NR) * kc_max];

    let mut jc = 0;
    while jc < n {
        let nc = (n - jc).min(NC);
        let mut pc = 0;
        while pc < k {
            let kc = (k - pc).min(KC);
            // First K-block honors the caller's beta; later blocks must accumulate the partial sums.
            let beta_eff = if pc == 0 { beta } else { 1.0 };
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
                );
                ic += MC;
            }
            pc += KC;
        }
        jc += NC;
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

// Below ~this many panels, rayon's task overhead outweighs the copy; pack serially instead.
const PACK_PAR_THRESHOLD: usize = 16;

/// Parallel A pack: each `MR`-row panel writes a disjoint `[kc·MR]` region, so pack them across cores.
/// Once the C compute is spread over every core, this serial-pack step is the Amdahl bottleneck.
///
/// # Safety
/// Same operand contract as [`pack_a`]; `ap` must hold `round_up(mc, MR) * kc` f32.
#[cfg(target_arch = "x86_64")]
unsafe fn pack_a_par(a: *const f32, lda: usize, mc: usize, kc: usize, ap: *mut f32) {
    use rayon::prelude::*;
    let mpanels = mc.div_ceil(MR);
    if mpanels < PACK_PAR_THRESHOLD {
        pack_a(a, lda, mc, kc, ap);
        return;
    }
    let (a_addr, ap_addr) = (a as usize, ap as usize);
    (0..mpanels).into_par_iter().for_each(|ip| {
        // SAFETY: disjoint output panel; avx2 verified by the caller; pointers re-derived per task.
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
}

/// Parallel B-block pack (handles the `C = A·Bᵀ` layout); each `NR`-col panel is independent.
///
/// # Safety
/// Same operand contract as [`pack_b_block`]; `bp` must hold `round_up(nc, NR) * kc` f32.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn pack_b_block_par(
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
    use rayon::prelude::*;
    let npanels = nc.div_ceil(NR);
    if npanels < PACK_PAR_THRESHOLD {
        pack_b_block(b, k, n, pc, jc, kc, nc, bt, bp);
        return;
    }
    // Resolve base pointer + leading dim once (mirrors pack_b_block's bt dispatch).
    let (b_base, ldb) = if bt {
        (b.add(jc * k + pc) as usize, k)
    } else {
        (b.add(pc * n + jc) as usize, n)
    };
    let bp_addr = bp as usize;
    (0..npanels).into_par_iter().for_each(|jp| {
        let panel = (bp_addr as *mut f32).add(jp * kc * NR);
        // SAFETY: disjoint output panel; avx2 verified by the caller; pointers re-derived per task.
        unsafe {
            if bt {
                pack_b_trans_panel(b_base as *const f32, ldb, kc, nc, jp, panel);
            } else {
                pack_b_panel(b_base as *const f32, ldb, kc, nc, jp, panel);
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
) {
    let mpanels = mc.div_ceil(MR);
    let npanels = nc.div_ceil(NR);
    for jp in 0..npanels {
        let j0 = jp * NR;
        let nrv = (nc - j0).min(NR);
        let bpanel = bp.add(jp * kc * NR);
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
) {
    use std::arch::x86_64::*;
    let (mut c0, mut c1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c2, mut c3) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c4, mut c5) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c6, mut c7) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c8, mut c9) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c10, mut c11) = (_mm256_setzero_ps(), _mm256_setzero_ps());

    let mut ap = ap;
    let mut bp = bp;
    for _ in 0..kc {
        // Prefetch a few K-steps ahead so the next B column / A row are warm in L1.
        _mm_prefetch::<_MM_HINT_T0>(bp.add(NR * 8) as *const i8);
        let b0 = _mm256_loadu_ps(bp);
        let b1 = _mm256_loadu_ps(bp.add(8));
        // Load-broadcast each A element straight from the packed panel (vbroadcastss [mem], 1 uop)
        // rather than a scalar load + register broadcast, freeing shuffle/ALU ports for the FMAs.
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
    }

    // Spill the tile, then write back the valid corner with the beta rule. The spill happens once
    // per tile (the K loop above dominates), so a scalar write-back costs nothing measurable.
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
            *cp = if beta == 0.0 { v } else { *cp + v };
        }
    }
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
    #[test]
    #[ignore]
    fn sgemm_throughput() {
        use std::time::Instant;
        for &n in &[256usize, 512, 1024] {
            let a = fill(1, n * n);
            let b = fill(2, n * n);
            let mut c = vec![0.0f32; n * n];
            let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
            let flops = 2.0 * (n as f64).powi(3);
            let bench = |label: &str, f: &dyn Fn()| {
                for _ in 0..3 {
                    f();
                }
                let mut best = f64::INFINITY;
                for _ in 0..20 {
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
            bench("sgemm (1 core)", &|| unsafe {
                mercury_sgemm(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
            bench("sgemm (parallel)", &|| unsafe {
                mercury_sgemm_parallel(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
            bench("sgemm_nt (1 core)", &|| unsafe {
                mercury_sgemm_nt(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
            bench("sgemm_nt (parallel)", &|| unsafe {
                mercury_sgemm_nt_parallel(ap, bp, cp, n as i64, n as i64, n as i64, 0);
            });
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
}
