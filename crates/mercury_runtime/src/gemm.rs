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
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let beta = beta as f32;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features just checked; dims validated by the caller contract.
            unsafe { sgemm_avx2(a, b, c, m, k, n, beta) };
            return;
        }
    }
    sgemm_scalar(a, b, c, m, k, n, beta);
}

/// Multi-threaded `C = A·B`: split the M dimension into row bands, each band an independent GEMM
/// over the shared `B`. Embarrassingly parallel (disjoint C rows), good locality (B reused within a
/// band's own cache blocking). Pack scratch is per-thread.
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
    use rayon::prelude::*;
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (mu, ku, nu) = (m as usize, k as usize, n as usize);
    // One band per worker (at least MR rows so the microkernel stays full), capped at the row count.
    let workers = rayon::current_num_threads().max(1);
    let band = round_up(mu.div_ceil(workers).max(MR), MR);
    let nbands = mu.div_ceil(band);
    let a_addr = a as usize;
    let b_addr = b as usize;
    let c_addr = c as usize;
    (0..nbands).into_par_iter().for_each(|w| {
        let i0 = w * band;
        if i0 >= mu {
            return;
        }
        let rows = (mu - i0).min(band);
        // SAFETY: bands address disjoint A/C row ranges; B is shared read-only.
        unsafe {
            let a_sub = (a_addr as *const f32).add(i0 * ku);
            let c_sub = (c_addr as *mut f32).add(i0 * nu);
            mercury_sgemm(
                a_sub,
                b_addr as *const f32,
                c_sub,
                rows as i64,
                k,
                n,
                beta,
            );
        }
    });
}

/// Portable reference: a straight `ikj` triple loop. Correct for any target; also the small-/odd-
/// size fallback. Accumulation order differs from the blocked kernel, so it is only used where the
/// AVX path is unavailable (the two are validated to agree within f32 tolerance in tests).
fn sgemm_scalar(a: *const f32, b: *const f32, c: *mut f32, m: usize, k: usize, n: usize, beta: f32) {
    unsafe {
        for i in 0..m {
            let crow = c.add(i * n);
            if beta == 0.0 {
                for j in 0..n {
                    *crow.add(j) = 0.0;
                }
            }
            for p in 0..k {
                let aik = *a.add(i * k + p);
                let brow = b.add(p * n);
                for j in 0..n {
                    *crow.add(j) += aik * *brow.add(j);
                }
            }
        }
    }
}

// --- AVX2 + FMA path -------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sgemm_avx2(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    beta: f32,
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
            pack_b(b.add(pc * n + jc), n, kc, nc, bp.as_mut_ptr());
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

/// Pack a `KC×NC` slice of B (row-major, leading dim `ldb`) into `NR`-wide column panels: panel `jp`
/// is `[kc][NR]` contiguous, zero-padded if the slice's last panel is partial.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_b(b: *const f32, ldb: usize, kc: usize, nc: usize, bp: *mut f32) {
    let mut dst = bp;
    let npanels = nc.div_ceil(NR);
    for jp in 0..npanels {
        let j0 = jp * NR;
        let ncols = (nc - j0).min(NR);
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
}

/// Pack an `MC×KC` slice of A (row-major, leading dim `lda`) into `MR`-wide row panels: panel `ip`
/// is `[kc][MR]` contiguous (so the microkernel broadcasts each row with unit stride), zero-padded.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_a(a: *const f32, lda: usize, mc: usize, kc: usize, ap: *mut f32) {
    let mut dst = ap;
    let mpanels = mc.div_ceil(MR);
    for ip in 0..mpanels {
        let i0 = ip * MR;
        let nrows = (mc - i0).min(MR);
        for p in 0..kc {
            for r in 0..nrows {
                *dst.add(r) = *a.add((i0 + r) * lda + p);
            }
            for r in nrows..MR {
                *dst.add(r) = 0.0;
            }
            dst = dst.add(MR);
        }
    }
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
            micro_6x16(kc, apanel, bpanel, c.add(i0 * ldc + j0), ldc, beta, mrv, nrv);
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
        let b0 = _mm256_loadu_ps(bp);
        let b1 = _mm256_loadu_ps(bp.add(8));
        let a0 = _mm256_set1_ps(*ap);
        c0 = _mm256_fmadd_ps(a0, b0, c0);
        c1 = _mm256_fmadd_ps(a0, b1, c1);
        let a1 = _mm256_set1_ps(*ap.add(1));
        c2 = _mm256_fmadd_ps(a1, b0, c2);
        c3 = _mm256_fmadd_ps(a1, b1, c3);
        let a2 = _mm256_set1_ps(*ap.add(2));
        c4 = _mm256_fmadd_ps(a2, b0, c4);
        c5 = _mm256_fmadd_ps(a2, b1, c5);
        let a3 = _mm256_set1_ps(*ap.add(3));
        c6 = _mm256_fmadd_ps(a3, b0, c6);
        c7 = _mm256_fmadd_ps(a3, b1, c7);
        let a4 = _mm256_set1_ps(*ap.add(4));
        c8 = _mm256_fmadd_ps(a4, b0, c8);
        c9 = _mm256_fmadd_ps(a4, b1, c9);
        let a5 = _mm256_set1_ps(*ap.add(5));
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
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
        // Includes sizes that exercise the MR/NR/MC/KC/NC remainders.
        check(1, 1, 1);
        check(6, 16, 1);
        check(7, 17, 13);
        check(64, 64, 64);
        check(72, 256, 80);
        check(100, 100, 100);
        check(128, 256, 512);
        check(200, 200, 200);
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
    #[test]
    #[ignore]
    fn sgemm_throughput() {
        use std::time::Instant;
        let n = 512usize;
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
            println!("{label}: {:.1} GFLOP/s ({:.3} ms)", flops / best / 1e9, best * 1e3);
        };
        bench("sgemm (1 core)", &|| unsafe {
            mercury_sgemm(ap, bp, cp, n as i64, n as i64, n as i64, 0);
        });
        bench("sgemm (parallel)", &|| unsafe {
            mercury_sgemm_parallel(ap, bp, cp, n as i64, n as i64, n as i64, 0);
        });
    }

    #[test]
    fn sgemm_parallel_matches_serial() {
        let (m, k, n) = (256, 192, 320);
        let a = fill(6, m * k);
        let b = fill(7, k * n);
        let mut serial = vec![0.0f32; m * n];
        let mut par = vec![0.0f32; m * n];
        unsafe {
            mercury_sgemm(a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(), m as i64, k as i64, n as i64, 0);
            mercury_sgemm_parallel(a.as_ptr(), b.as_ptr(), par.as_mut_ptr(), m as i64, k as i64, n as i64, 0);
        }
        // Same blocking per band ⇒ identical results.
        assert_eq!(serial, par);
    }
}
