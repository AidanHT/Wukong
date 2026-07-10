//! Vector-times-matrix `out = wᵀ·A` — the weighted row-sum `out[j] = α · Σ_i w[i]·A[i,j]` over a
//! `[rows, cols]` row-major `A` (`wukong_sgevm_f32[_parallel]`): the single-token KV-decode
//! attention read-out `out = scoresᵀ·V`, and any right-multiplication of a vector by a matrix.
//!
//! The naive spelling `for j { let s = 0; for i { s += w[i]*a[i*N+j] }; out[j] = s }` reads `A` with
//! stride `N` — a **strided reduction** gcc/rustc leave scalar at `-O3 -march=native` (the same
//! column-outer family as `colreduce.rs`). This kernel restructures to **i-outer / j-inner**:
//! `for i { out[j..] += w[i]·A[i, j..] }`, streaming `A` row-major contiguously and folding 8
//! columns per step with a 256-bit FMA into the cache-resident `out[]`.
//!
//! **Bit-exactness.** For each fixed column `j` the fold is `out[j] = fma(w[i], a[i,j], out[j])` in
//! **ascending i from 0.0** — the identical operation and order as the source's FMA-contracted
//! inner loop (`s = s + w[i]*a[..]` contracts to `Op::Fma`; `f32::mul_add` == `vfmadd` lane-for-
//! lane), so there is **no reassociation at all**: the AVX2 lanes, the scalar twin/tail, the serial
//! kernel, and *any* column split of the parallel form compute the same bits per element (each
//! column's chain is self-contained — no cross-column combine). The optional α scale (the
//! scaled-store form `out[j] = s * c`) multiplies each output element **exactly once** after its
//! full fold — the same single f32 multiply the scalar store performs; `alpha == 1.0` skips the
//! pass entirely (byte-identical to the unscaled fold). The interpreter marshals this same serial
//! kernel as the oracle, so the differential gate holds by construction; the `#[test]`s pin the
//! kernel against a literal scalar `mul_add` reference (exact equality) and an independent f64
//! reference.

/// Scale `out[j0..j1]` by `alpha` — the per-column finalize of a scaled store `out[j] = s * c`.
/// One f32 multiply per element (the same multiply the scalar store performs), applied after the
/// column's full fold; a no-op for `alpha == 1.0` so the unscaled fold is byte-identical to a
/// never-scaled kernel. Plain scalar arithmetic, shared by the AVX2 and scalar paths and
/// independent of the column split, so it does not perturb serial == parallel bit-equality.
///
/// # Safety
/// `out` valid for `[j0, j1)` f32.
#[inline]
unsafe fn gevm_finalize(out: *mut f32, j0: usize, j1: usize, alpha: f32) {
    if alpha != 1.0 {
        for j in j0..j1 {
            *out.add(j) *= alpha;
        }
    }
}

/// Fold rows `[0, rows)` into the column range `[j0, j1)` of `out`
/// (`out[j] = α · Σ_i w[i]·a[i*cols+j]`), AVX2 + FMA, 8 columns/step + a scalar tail. Zero-seeds the
/// range, then streams `a` row-major, broadcasting `w[i]` and folding with `_mm256_fmadd_ps` — the
/// exact `mul_add` the scalar twin uses, so the two agree lane-for-lane.
///
/// # Safety
/// `w` valid for `rows`, `a` for `rows*cols`, `out` for `cols` f32; `j0 <= j1 <= cols`;
/// AVX2 + FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gevm_avx2(
    w: *const f32,
    a: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
    j0: usize,
    j1: usize,
    alpha: f32,
) {
    use std::arch::x86_64::*;
    for j in j0..j1 {
        *out.add(j) = 0.0;
    }
    for i in 0..rows {
        let wi = *w.add(i);
        let wv = _mm256_set1_ps(wi);
        let ar = a.add(i * cols);
        let mut j = j0;
        while j + 8 <= j1 {
            let acc = _mm256_loadu_ps(out.add(j));
            let v = _mm256_loadu_ps(ar.add(j));
            // out[j..] = w[i]·a[i,j..] + out[j..] — one fused multiply-add per element, the same
            // `fma(w, a, acc)` chain (ascending i) the source's contracted scalar loop runs.
            _mm256_storeu_ps(out.add(j), _mm256_fmadd_ps(wv, v, acc));
            j += 8;
        }
        while j < j1 {
            // `mul_add` == `vfmadd` bit-for-bit (this fn has the `fma` target feature, so it is a
            // hardware fused multiply-add, matching the vector lanes above).
            *out.add(j) = wi.mul_add(*ar.add(j), *out.add(j));
            j += 1;
        }
    }
    gevm_finalize(out, j0, j1, alpha);
}

/// Scalar twin of [`gevm_avx2`] (the no-AVX2 fallback and the bit-exact reference). The same
/// i-ascending per-column `mul_add` fold from `0.0` (`f32::mul_add` is a correctly-rounded fused
/// multiply-add — libm `fmaf` where no hardware FMA exists — so it equals the AVX2 lanes
/// bit-for-bit), then the same single-α finalize.
///
/// # Safety
/// `w` valid for `rows`, `a` for `rows*cols`, `out` for `cols` f32; `j0 <= j1 <= cols`.
unsafe fn gevm_scalar(
    w: *const f32,
    a: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
    j0: usize,
    j1: usize,
    alpha: f32,
) {
    for j in j0..j1 {
        *out.add(j) = 0.0;
    }
    for i in 0..rows {
        let wi = *w.add(i);
        let ar = a.add(i * cols);
        for j in j0..j1 {
            *out.add(j) = wi.mul_add(*ar.add(j), *out.add(j));
        }
    }
    gevm_finalize(out, j0, j1, alpha);
}

/// Dispatch AVX2+FMA vs scalar for the column range `[j0, j1)`.
///
/// # Safety
/// Operand-size contract of [`gevm_avx2`].
#[inline]
unsafe fn gevm_range(
    w: *const f32,
    a: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
    j0: usize,
    j1: usize,
    alpha: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            gevm_avx2(w, a, out, rows, cols, j0, j1, alpha);
            return;
        }
    }
    gevm_scalar(w, a, out, rows, cols, j0, j1, alpha);
}

/// `out[j] = alpha · Σ_i w[i]·a[i, j]` over a `[rows, cols]` row-major `a`, single-threaded.
/// `alpha == 1.0` is the plain (unscaled) vector·matrix product.
///
/// # Safety
/// `w` valid for `rows`, `a` for `rows*cols`, `out` for `cols` f32; `out` must not overlap `w` or
/// `a` (it is the running accumulator while both are re-read).
#[no_mangle]
pub unsafe extern "C" fn wukong_sgevm_f32(
    w: *const f32,
    a: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
    alpha: f32,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    gevm_range(w, a, out, rows as usize, cols as usize, 0, cols as usize, alpha);
}

/// Column count below which the parallel vector·matrix product just runs serially (thread wake/sync
/// would dominate a small output).
const GEVM_PAR_MIN: usize = 256;

/// Multi-threaded `out[j] = alpha · Σ_i w[i]·a[i, j]`: the **output columns** are split into
/// disjoint stripes across cores, each core folding its column range over **all** rows (the full
/// i sweep, ascending) — never rows-across-cores, which would need per-thread partials and
/// reassociate the sum. Each stripe writes a disjoint slice of `out` and every column folds in the
/// same i-ascending order regardless of the split, so the result is **bit-identical to
/// [`wukong_sgevm_f32`]** (no cross-stripe combine). Below [`GEVM_PAR_MIN`] columns it runs serial.
///
/// # Safety
/// Operand-size contract of [`wukong_sgevm_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgevm_f32_parallel(
    w: *const f32,
    a: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
    alpha: f32,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if c < GEVM_PAR_MIN {
        gevm_range(w, a, out, r, c, 0, c, alpha);
        return;
    }
    use rayon::prelude::*;
    // One stripe per core, each a multiple of 8 columns (keep the AVX2 main loop aligned to the
    // stripe boundary); the last stripe absorbs the remainder. The split does not affect the bits
    // (per-column chains are self-contained), only the work distribution.
    let nthreads = rayon::current_num_threads().max(1);
    let per = (c.div_ceil(nthreads)).next_multiple_of(8).max(8);
    let nstripes = c.div_ceil(per);
    let (w_addr, a_addr, out_addr) = (w as usize, a as usize, out as usize);
    (0..nstripes).into_par_iter().for_each(|s| {
        let j0 = s * per;
        let j1 = (j0 + per).min(c);
        // SAFETY: disjoint out[] stripe per task; pointers re-derived from the captured addresses.
        unsafe {
            gevm_range(
                w_addr as *const f32,
                a_addr as *const f32,
                out_addr as *mut f32,
                r,
                c,
                j0,
                j1,
                alpha,
            );
        }
    });
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

    /// The kernel must equal the literal scalar reference — `out[j] = fold_{i ascending}
    /// mul_add(w[i], a[i,j], ·)` from `0.0`, then one `* alpha` — **exactly** (`assert_eq!`, no
    /// tolerance): the kernel performs the identical per-element chain, so it is bit-for-bit its own
    /// scalar oracle. Serial and parallel must also be bit-identical (disjoint column stripes, full
    /// i sweep each). Sizes straddle the 8-lane edge, the stripe boundary, and `GEVM_PAR_MIN`.
    #[test]
    fn sgevm_matches_fma_reference_and_parallel() {
        for &(rows, cols) in &[
            (1usize, 1usize),
            (1, 7),
            (3, 8),
            (5, 31),
            (7, 32),
            (128, 64), // the KV-decode read-out shape (S=128, D=64)
            (9, 100),
            (16, 257),
            (64, 300), // exceeds GEVM_PAR_MIN so the multicore path really runs
        ] {
            let w = fill(1, rows);
            let a = fill(2, rows * cols);
            for &alpha in &[1.0f32, 0.125, -2.5] {
                let mut want = vec![0.0f32; cols];
                for i in 0..rows {
                    for j in 0..cols {
                        want[j] = w[i].mul_add(a[i * cols + j], want[j]);
                    }
                }
                if alpha != 1.0 {
                    for v in &mut want {
                        *v *= alpha;
                    }
                }
                let mut got = vec![0.0f32; cols];
                let mut got_par = vec![0.0f32; cols];
                unsafe {
                    wukong_sgevm_f32(
                        w.as_ptr(),
                        a.as_ptr(),
                        got.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                        alpha,
                    );
                    wukong_sgevm_f32_parallel(
                        w.as_ptr(),
                        a.as_ptr(),
                        got_par.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                        alpha,
                    );
                }
                assert_eq!(got, want, "sgevm {rows}x{cols} alpha={alpha} vs scalar fma reference");
                assert_eq!(got, got_par, "sgevm serial vs parallel {rows}x{cols} alpha={alpha}");
            }
        }
    }

    /// Independent **f64 reference** over the whole output (no `mul_add`, no call into any
    /// `wukong_*` kernel): the fused fold stays within `√rows·ε` of the exact weighted sum. This is
    /// the genuine external oracle — the recognizer path is gate-blind (both backends marshal this
    /// identical kernel), so interp==native alone proves nothing about the kernel's arithmetic.
    #[test]
    fn sgevm_matches_f64_reference() {
        for &(rows, cols) in &[(1usize, 1usize), (3, 7), (128, 64), (100, 257), (300, 300)] {
            let w = fill(3, rows);
            let a = fill(4, rows * cols);
            let alpha = 0.125f32;
            let mut want = vec![0.0f64; cols];
            for j in 0..cols {
                let mut acc = 0.0f64;
                for i in 0..rows {
                    acc += w[i] as f64 * a[i * cols + j] as f64;
                }
                want[j] = acc * alpha as f64;
            }
            let mut got = vec![0.0f32; cols];
            unsafe {
                wukong_sgevm_f32(
                    w.as_ptr(),
                    a.as_ptr(),
                    got.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                    alpha,
                );
            }
            let tol = 1e-3 * (rows as f32).sqrt();
            for j in 0..cols {
                assert!(
                    (got[j] as f64 - want[j]).abs() as f32 <= tol + 1e-4 * want[j].abs() as f32,
                    "sgevm ({rows}x{cols}) col {j}: got {} want {}",
                    got[j],
                    want[j]
                );
            }
        }
    }
}
