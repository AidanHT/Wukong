//! Single-precision GEMV — the matrix-times-vector kernel `y[i] = Σ_j A[i,j]·x[j]`.
//!
//! `A` is row-major `[M, N]`, `x` is length `N`, `y` is length `M`. This is the decode-time attention
//! /projection shape (batch 1): a `[M,N]·[N]` product where the matmul's 3-loop register-blocking
//! collapses — there is no reuse of `A` (each element is read once and multiplied once), so GEMV is
//! **memory-bound**, streaming `A` from DRAM while `x` stays cache-resident. The lever a scalar C/Rust
//! compiler can't pull at `-O3 -march=native` *without* `-ffast-math` is reassociating the row dot: gcc
//! keeps `s += A[i,j]·x[j]` a strict in-order scalar/`vfmadd…ss` chain (one FMA per ~4–5 cycles,
//! latency-bound far below bandwidth), while this kernel folds each row 8-wide across **four independent
//! `__m256` accumulators** — enough in-flight FMAs to hit the A-streaming roofline — and the `_parallel`
//! form maps the independent rows across cores to saturate aggregate memory bandwidth.
//!
//! **Differential contract.** The compiler lowers a recognized GEMV nest to this symbol and the
//! interpreter marshals its abstract memory into real buffers and calls the *identical* function, so the
//! two backends stay bit-for-bit exact by construction. Each row's dot reassociates the sum (the
//! documented reassociated-reduction exception — the kernel *is* the oracle both backends run); the
//! `#[test]`s additionally gate it against an independent **f64 reference** over the whole output. Rows
//! are independent and every row is computed by the *same* per-row routine, so serial == parallel
//! bit-for-bit (no cross-row combine).

/// Horizontal sum of the 8 lanes of a `__m256` into one `f32`, in a fixed order (so serial == parallel).
///
/// # Safety
/// AVX must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // Fold the high 128 into the low 128, then sum the 4 lanes: (0+2)+(1+3) via two hadds. A fixed,
    // deterministic reduction tree — the same every call, so it introduces no serial/parallel divergence.
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// One row's dot `y[i] = Σ_{j<n} a[j]·x[j]`, AVX2 + FMA, four 8-wide accumulators for ILP + a scalar
/// tail. The four accumulators break the loop-carried FMA dependency (four chains in flight ≈ the
/// memory-streaming roofline), the win gcc/rustc leave on the table by refusing to reassociate an f32
/// reduction without `-ffast-math`.
///
/// # Safety
/// `a` and `x` valid for `n` `f32`; AVX2 + FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_row_avx2(a: *const f32, x: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let (mut c0, mut c1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c2, mut c3) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut j = 0;
    // 32 elements per step: four independent FMA chains, streaming A contiguously and re-reading the
    // (cache-resident) x. `_mm256_fmadd_ps(a, x, c)` == the scalar `x.mul_add`/`fmadd` lane-for-lane.
    while j + 32 <= n {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(j)), _mm256_loadu_ps(x.add(j)), c0);
        c1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.add(j + 8)),
            _mm256_loadu_ps(x.add(j + 8)),
            c1,
        );
        c2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.add(j + 16)),
            _mm256_loadu_ps(x.add(j + 16)),
            c2,
        );
        c3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.add(j + 24)),
            _mm256_loadu_ps(x.add(j + 24)),
            c3,
        );
        j += 32;
    }
    // Remaining full 8-blocks fold into the first accumulator.
    while j + 8 <= n {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(j)), _mm256_loadu_ps(x.add(j)), c0);
        j += 8;
    }
    // Combine the four accumulators in a fixed tree, then horizontally sum.
    let acc = _mm256_add_ps(_mm256_add_ps(c0, c1), _mm256_add_ps(c2, c3));
    let mut s = hsum256(acc);
    // Scalar tail (< 8 elements), folded after the vector sum — deterministic order.
    while j < n {
        s += *a.add(j) * *x.add(j);
        j += 1;
    }
    s
}

/// Scalar reference / no-AVX2 fallback: the straight in-order row dot. Used only where AVX2 is absent
/// (its accumulation order differs from the AVX2 path, so the two agree only within f32 tolerance — the
/// AVX2 path is the one the differential gate exercises on this target).
///
/// # Safety
/// `a` and `x` valid for `n` `f32`.
unsafe fn gemv_row_scalar(a: *const f32, x: *const f32, n: usize) -> f32 {
    let mut s = 0.0f32;
    for j in 0..n {
        s += *a.add(j) * *x.add(j);
    }
    s
}

/// One row's dot, dispatching to the AVX2 path when available.
///
/// # Safety
/// `a` and `x` valid for `n` `f32`.
#[inline]
unsafe fn gemv_row(a: *const f32, x: *const f32, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return gemv_row_avx2(a, x, n);
        }
    }
    gemv_row_scalar(a, x, n)
}

/// `y[i] = Σ_j A[i,j]·x[j]` for a row-major `A[M,N]`, `x[N]`, `y[M]` — single-threaded.
///
/// # Safety
/// `a` valid for `m*n` `f32`, `x` for `n`, `y` for `m`; `y` must not overlap `a` or `x`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemv(a: *const f32, x: *const f32, y: *mut f32, m: i64, n: i64) {
    if m <= 0 || n <= 0 {
        return;
    }
    let (m, n) = (m as usize, n as usize);
    for i in 0..m {
        *y.add(i) = gemv_row(a.add(i * n), x, n);
    }
}

/// `y[i] = alpha · Σ_j A[i,j]·x[j]` — the **α-scaled GEMV**: the decode-time attention score
/// `scores = (K·q)·(1/√d)` and every scaled single-token projection, where a loop-invariant scalar
/// `alpha` multiplies the row dot on the store (`y[i] = s * c`). The compiler lowers a recognized
/// GEMV nest with a scaled store to this. Each row's dot is the *identical* [`gemv_row`] the plain
/// [`wukong_sgemv`] computes; `alpha` is applied **exactly once per output element** on the store —
/// the same single f32 multiply the scalar nest's `y[i] = s * c` performs — and `alpha == 1.0` skips
/// the multiply (byte-identical to the plain kernel). Single-threaded.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemv`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemv_alpha(
    a: *const f32,
    x: *const f32,
    y: *mut f32,
    m: i64,
    n: i64,
    alpha: f32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    let (m, n) = (m as usize, n as usize);
    for i in 0..m {
        let s = gemv_row(a.add(i * n), x, n);
        // One multiply on the store, skipped for α == 1.0 (the Epilogue convention) so the unscaled
        // form stays byte-identical to `wukong_sgemv`.
        *y.add(i) = if alpha != 1.0 { s * alpha } else { s };
    }
}

/// Multi-threaded α-scaled GEMV: independent rows across cores, each the identical per-row routine
/// (dot + one store multiply) — **bit-identical to [`wukong_sgemv_alpha`]** (no cross-row combine).
/// Below the row threshold it runs the serial kernel.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemv`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemv_alpha_parallel(
    a: *const f32,
    x: *const f32,
    y: *mut f32,
    m: i64,
    n: i64,
    alpha: f32,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    let (mu, nu) = (m as usize, n as usize);
    if mu < GEMV_PAR_MIN_ROWS {
        wukong_sgemv_alpha(a, x, y, m, n, alpha);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        use rayon::prelude::*;
        // A `@parallel` GEMV can be the process's FIRST rayon touch (a decode-step LM head runs
        // before any outlined region), so it must configure the global pool before forking — the
        // invariant `ensure_global_pool` states: whoever forks first decides the registry, and a
        // default 2 MiB-stack registry built here makes the runtime's later 16 MiB `build_global`
        // lose the race, leaving `@parallel` region bodies (~1.5 MiB of privatized scratch) on
        // 2 MiB stacks. Pool configuration only — the row split below is worker-count independent,
        // so the bits are unchanged.
        crate::ensure_global_pool();
        let (a_addr, x_addr, y_addr) = (a as usize, x as usize, y as usize);
        (0..mu).into_par_iter().for_each(|i| {
            // SAFETY: disjoint output element y[i]; shared read-only a-row / x; pointers re-derived.
            unsafe {
                let s = gemv_row((a_addr as *const f32).add(i * nu), x_addr as *const f32, nu);
                *(y_addr as *mut f32).add(i) = if alpha != 1.0 { s * alpha } else { s };
            }
        });
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    wukong_sgemv_alpha(a, x, y, m, n, alpha);
}

/// Row count below which the parallel GEMV just runs serially (thread wake/sync would dominate a tiny
/// matrix). GEMV is memory-bound, so the parallel win comes from spreading the row stream across cores
/// to saturate aggregate DRAM bandwidth.
const GEMV_PAR_MIN_ROWS: usize = 64;

/// Multi-threaded GEMV: the independent rows are mapped across cores, each computed by the identical
/// per-row routine — so the result is **bit-identical to [`wukong_sgemv`]** (no cross-row combine).
/// Below the row threshold it runs the serial kernel.
///
/// # Safety
/// Operand-size contract of [`wukong_sgemv`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemv_parallel(
    a: *const f32,
    x: *const f32,
    y: *mut f32,
    m: i64,
    n: i64,
) {
    if m <= 0 || n <= 0 {
        return;
    }
    let (mu, nu) = (m as usize, n as usize);
    if mu < GEMV_PAR_MIN_ROWS {
        wukong_sgemv(a, x, y, m, n);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        use rayon::prelude::*;
        // Configure the global pool before forking, for the reason spelled out in
        // [`wukong_sgemv_alpha_parallel`] (this entry can equally be the first rayon touch).
        crate::ensure_global_pool();
        let (a_addr, x_addr, y_addr) = (a as usize, x as usize, y as usize);
        (0..mu).into_par_iter().for_each(|i| {
            // SAFETY: disjoint output element y[i]; shared read-only a-row / x; pointers re-derived.
            unsafe {
                *(y_addr as *mut f32).add(i) =
                    gemv_row((a_addr as *const f32).add(i * nu), x_addr as *const f32, nu);
            }
        });
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    wukong_sgemv(a, x, y, m, n);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(seed: u64, n: usize) -> Vec<f32> {
        // Deterministic pseudo-random values in [-1, 1).
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

    /// `y = A·x` matches an independent **f64 reference** (the reassociated 8-wide sum stays within
    /// √n·ε of the exact dot), across sizes that straddle the 32/8-element unroll edges and 1-wide dims;
    /// and the serial and parallel kernels are **bit-for-bit identical** (rows independent, same routine).
    #[test]
    fn sgemv_matches_f64_reference_and_parallel() {
        for &(m, n) in &[
            (1, 1),
            (1, 7),
            (3, 8),
            (5, 31),
            (7, 32),
            (9, 33),
            (16, 100),
            (64, 64),
            (100, 257),
            (128, 512),
            // Exceeds GEMV_PAR_MIN_ROWS so the multicore path really runs, with a non-multiple-of-32 N.
            (300, 517),
        ] {
            let a = fill(1, m * n);
            let x = fill(2, n);
            // f64 reference over the whole output.
            let mut want = vec![0.0f64; m];
            for i in 0..m {
                let mut acc = 0.0f64;
                for j in 0..n {
                    acc += a[i * n + j] as f64 * x[j] as f64;
                }
                want[i] = acc;
            }
            let mut got = vec![0.0f32; m];
            let mut got_par = vec![0.0f32; m];
            unsafe {
                wukong_sgemv(a.as_ptr(), x.as_ptr(), got.as_mut_ptr(), m as i64, n as i64);
                wukong_sgemv_parallel(
                    a.as_ptr(),
                    x.as_ptr(),
                    got_par.as_mut_ptr(),
                    m as i64,
                    n as i64,
                );
            }
            let tol = 1e-3 * (n as f32).sqrt();
            for i in 0..m {
                assert!(
                    (got[i] as f64 - want[i]).abs() as f32 <= tol + 1e-4 * want[i].abs() as f32,
                    "gemv ({m}x{n}) row {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
            assert_eq!(got, got_par, "gemv serial vs parallel ({m}x{n})");
        }
    }

    /// The α-scaled GEMV must equal the plain kernel followed by one scalar `* alpha` per output
    /// element (**exact** equality — the kernel performs that identical single multiply on the
    /// store); `alpha == 1.0` must be **byte-identical** to the plain kernel (the multiply is
    /// skipped); and serial == parallel bit-for-bit (rows independent, same per-row routine).
    /// Sizes straddle the 32/8-element unroll edges and `GEMV_PAR_MIN_ROWS`.
    #[test]
    fn sgemv_alpha_matches_scaled_plain_and_parallel() {
        for &(m, n) in &[
            (1usize, 1usize),
            (3, 8),
            (7, 32),
            (9, 33),
            (128, 64),
            (300, 517),
        ] {
            let a = fill(5, m * n);
            let x = fill(6, n);
            let mut plain = vec![0.0f32; m];
            unsafe {
                wukong_sgemv(
                    a.as_ptr(),
                    x.as_ptr(),
                    plain.as_mut_ptr(),
                    m as i64,
                    n as i64,
                );
            }
            for &alpha in &[1.0f32, 0.125, -2.5] {
                let want: Vec<f32> = plain
                    .iter()
                    .map(|&s| if alpha != 1.0 { s * alpha } else { s })
                    .collect();
                let mut got = vec![0.0f32; m];
                let mut got_par = vec![0.0f32; m];
                unsafe {
                    wukong_sgemv_alpha(
                        a.as_ptr(),
                        x.as_ptr(),
                        got.as_mut_ptr(),
                        m as i64,
                        n as i64,
                        alpha,
                    );
                    wukong_sgemv_alpha_parallel(
                        a.as_ptr(),
                        x.as_ptr(),
                        got_par.as_mut_ptr(),
                        m as i64,
                        n as i64,
                        alpha,
                    );
                }
                assert_eq!(
                    got, want,
                    "sgemv_alpha != alpha·sgemv ({m}x{n} alpha={alpha})"
                );
                assert_eq!(
                    got, got_par,
                    "sgemv_alpha serial vs parallel ({m}x{n} alpha={alpha})"
                );
            }
            // alpha == 1.0 must be byte-identical to the plain kernel.
            let mut a1 = vec![0.0f32; m];
            unsafe {
                wukong_sgemv_alpha(
                    a.as_ptr(),
                    x.as_ptr(),
                    a1.as_mut_ptr(),
                    m as i64,
                    n as i64,
                    1.0,
                );
            }
            assert_eq!(
                plain, a1,
                "sgemv_alpha(1.0) must be byte-identical to sgemv ({m}x{n})"
            );
        }
    }

    /// The portable fallback [`gemv_row_scalar`] is the ONLY row dot on a target without AVX2+FMA
    /// (and the whole kernel on a non-x86_64 build, where the `#[cfg(target_arch = "x86_64")]` fast
    /// path is compiled out), but on an AVX2 host every public entry takes the AVX2 branch, so no
    /// other test in this module ever executes it — deleting the `* x[j]` factor left the suite
    /// green. Pin it directly against an independent f64 dot, across the sizes the AVX2 twin's
    /// 32/8-element unroll edges are keyed to. Tolerance is the standard in-order f32 accumulation
    /// bound `n·u·Σ|a·x|` with slack (this path does NOT reassociate, so it is the loosest of the
    /// two paths only in chain length).
    #[test]
    fn gemv_row_scalar_matches_f64_reference() {
        for &n in &[1usize, 7, 8, 31, 32, 33] {
            let a = fill(11, n);
            let x = fill(12, n);
            let (mut want, mut mag) = (0.0f64, 0.0f64);
            for j in 0..n {
                want += a[j] as f64 * x[j] as f64;
                mag += (a[j] as f64 * x[j] as f64).abs();
            }
            // SAFETY: `a` and `x` are each `n` `f32` long — the kernel's whole precondition.
            let got = unsafe { gemv_row_scalar(a.as_ptr(), x.as_ptr(), n) };
            let tol = 4.0 * n as f64 * f32::EPSILON as f64 * mag;
            assert!(
                (got as f64 - want).abs() <= tol,
                "gemv_row_scalar n={n}: got {got} want {want} tol {tol}"
            );
        }
    }

    /// Throughput probe (run: `cargo test -p wukong_runtime --release -- --ignored --nocapture
    /// sgemv_throughput`). GEMV is memory-bound, so the figure of merit is A-streaming GB/s, not GFLOP/s.
    #[test]
    #[ignore]
    fn sgemv_throughput() {
        use std::time::Instant;
        for &(m, n) in &[(4096usize, 4096usize), (8192, 2048), (2048, 8192)] {
            let a = fill(1, m * n);
            let x = fill(2, n);
            let mut y = vec![0.0f32; m];
            let (ap, xp, yp) = (a.as_ptr(), x.as_ptr(), y.as_mut_ptr());
            let bytes = (m * n * 4) as f64; // A is streamed once; the roofline traffic.
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
                    "m={m:<5} n={n:<5} {label:<18}: {:6.1} GB/s ({:.3} ms)",
                    bytes / best / 1e9,
                    best * 1e3
                );
            };
            bench("sgemv (1 core)", &|| unsafe {
                wukong_sgemv(ap, xp, yp, m as i64, n as i64);
            });
            bench("sgemv (parallel)", &|| unsafe {
                wukong_sgemv_parallel(ap, xp, yp, m as i64, n as i64);
            });
        }
    }
}
