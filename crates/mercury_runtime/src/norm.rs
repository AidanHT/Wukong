//! Fused single-pass row-wise normalizations — **softmax / LayerNorm / RMSNorm** over the last axis
//! of a `[rows, cols]` f32 matrix. These are the per-token normalizations every transformer block
//! runs (softmax in attention; LayerNorm/RMSNorm around the sublayers), and they are *memory-bound*:
//! a naive implementation streams each row from memory two or three times (max+exp+sum+normalize for
//! softmax; mean+var+normalize for LayerNorm). The win here is **fusion** — each row is loaded once
//! into registers/L1 and all passes run on it before moving on — plus the 256-bit AVX2 width (and,
//! for softmax, the same hand-vectorized `exp` the activation kernel uses, which Cranelift can't emit).
//!
//! The compiler recognizes the canonical row-wise norm loop nest and lowers it to one
//! [`mercury_norm_f32`] call (the same play as matmul→GEMM, activation→`mercury_vmath_f32`, and
//! reduction→`mercury_sreduce_f32`). The interpreter marshals its abstract memory through the
//! **identical** kernel, so the differential oracle stays bit-for-bit exact.
//!
//! **Determinism.** Rows are independent, so the `_parallel` entry just maps the identical per-row
//! routine across rows — `serial == parallel` bit-for-bit with no cross-row combine. Within a row the
//! two reductions (softmax's max+sum, LayerNorm's mean+var, RMSNorm's mean-square) use a fixed 8-lane
//! accumulator and a fixed-order horizontal combine, and the AVX2 path stores its accumulator to the
//! same `[f32; 8]` the scalar twin builds and calls the *same* combine — so the AVX2 kernel and the
//! scalar fallback agree bit-for-bit too (pinned by a unit test across partial-chunk / tail sizes).

use crate::vmath::exp1;
#[cfg(target_arch = "x86_64")]
use crate::vmath::exp8;
use rayon::prelude::*;

// Norm op codes (shared with the recognizer in `mercury_mir_build`).
pub const NORM_SOFTMAX: i64 = 0; // out = softmax(x) over the row (numerically stable)
pub const NORM_LAYERNORM: i64 = 1; // out = (x - mean) / sqrt(var + eps)
pub const NORM_RMSNORM: i64 = 2; // out = x / sqrt(mean(x^2) + eps)

/// Fixed-order horizontal sum of 8 lane accumulators — a balanced tree, identical in the scalar twin
/// and the AVX2 path (which stores its `__m256` to `[f32; 8]` and calls this), so both give the same
/// bits. Same shape as the reduction kernel's `hcombine8`.
#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

/// Fixed-order horizontal max of 8 lane accumulators (balanced tree). On finite inputs this matches
/// `_mm256_max_ps` lane-for-lane; softmax never sees NaN/±0 rows in practice and the twin test pins
/// the agreement on finite data.
#[inline(always)]
fn hmax8(a: [f32; 8]) -> f32 {
    (a[0].max(a[1]).max(a[2].max(a[3]))).max(a[4].max(a[5]).max(a[6].max(a[7])))
}

// --- scalar twins (the AVX2 tail + the no-AVX2 fallback) ------------------------------------------

/// Numerically-stable softmax of one row, scalar reference. `x` and `out` may alias (in-place).
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
unsafe fn softmax_row_scalar(x: *const f32, out: *mut f32, n: usize) {
    let nb = n / 8;
    let t = nb * 8;
    // 1) row max (8 lanes; lane j folds elements ≡ j (mod 8), tail into lanes 0..).
    let mut mx = [f32::NEG_INFINITY; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, mxj) in mx.iter_mut().enumerate() {
            *mxj = mxj.max(*x.add(b + j));
        }
    }
    for (j, mxj) in mx.iter_mut().enumerate().take(n - t) {
        *mxj = mxj.max(*x.add(t + j));
    }
    let m = hmax8(mx);
    // 2) e = exp(x - m); write out; accumulate the sum in the same lane structure.
    let mut sm = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, smj) in sm.iter_mut().enumerate() {
            let e = exp1(*x.add(b + j) - m);
            *out.add(b + j) = e;
            *smj += e;
        }
    }
    for (j, smj) in sm.iter_mut().enumerate().take(n - t) {
        let e = exp1(*x.add(t + j) - m);
        *out.add(t + j) = e;
        *smj += e;
    }
    let inv = 1.0 / hsum8(sm);
    // 3) normalize (reads the just-written e values; independent multiply, order-immaterial).
    for i in 0..n {
        *out.add(i) *= inv;
    }
}

/// LayerNorm of one row (gamma=1, beta=0), scalar reference. `x`/`out` may alias.
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
unsafe fn layernorm_row_scalar(x: *const f32, out: *mut f32, n: usize, eps: f32) {
    let nb = n / 8;
    let t = nb * 8;
    let invn = 1.0 / (n as f32);
    // mean
    let mut sm = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, smj) in sm.iter_mut().enumerate() {
            *smj += *x.add(b + j);
        }
    }
    for (j, smj) in sm.iter_mut().enumerate().take(n - t) {
        *smj += *x.add(t + j);
    }
    let mean = hsum8(sm) * invn;
    // variance = mean of squared deviations (the addend FMA-contracts, matching the AVX2 fmadd)
    let mut vv = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, vvj) in vv.iter_mut().enumerate() {
            let d = *x.add(b + j) - mean;
            *vvj = d.mul_add(d, *vvj);
        }
    }
    for (j, vvj) in vv.iter_mut().enumerate().take(n - t) {
        let d = *x.add(t + j) - mean;
        *vvj = d.mul_add(d, *vvj);
    }
    let inv = 1.0 / (hsum8(vv) * invn + eps).sqrt();
    for i in 0..n {
        *out.add(i) = (*x.add(i) - mean) * inv;
    }
}

/// RMSNorm of one row (gamma=1), scalar reference. `x`/`out` may alias.
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
unsafe fn rmsnorm_row_scalar(x: *const f32, out: *mut f32, n: usize, eps: f32) {
    let nb = n / 8;
    let t = nb * 8;
    let invn = 1.0 / (n as f32);
    let mut ss = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, ssj) in ss.iter_mut().enumerate() {
            let v = *x.add(b + j);
            *ssj = v.mul_add(v, *ssj);
        }
    }
    for (j, ssj) in ss.iter_mut().enumerate().take(n - t) {
        let v = *x.add(t + j);
        *ssj = v.mul_add(v, *ssj);
    }
    let inv = 1.0 / (hsum8(ss) * invn + eps).sqrt();
    for i in 0..n {
        *out.add(i) = *x.add(i) * inv;
    }
}

// --- AVX2 kernels (mirror the scalar twins lane-for-lane on finite inputs) -------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_row_avx2(x: *const f32, out: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    // 1) max
    let mut mxv = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 8 <= n {
        mxv = _mm256_max_ps(mxv, _mm256_loadu_ps(x.add(i)));
        i += 8;
    }
    let mut mx = [0.0f32; 8];
    _mm256_storeu_ps(mx.as_mut_ptr(), mxv);
    for (j, mxj) in mx.iter_mut().enumerate().take(n - i) {
        *mxj = mxj.max(*x.add(i + j));
    }
    let m = hmax8(mx);
    let mb = _mm256_set1_ps(m);
    // 2) exp(x - m) + sum
    let mut sv = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb));
        _mm256_storeu_ps(out.add(i), e);
        sv = _mm256_add_ps(sv, e);
        i += 8;
    }
    let mut sm = [0.0f32; 8];
    _mm256_storeu_ps(sm.as_mut_ptr(), sv);
    for (j, smj) in sm.iter_mut().enumerate().take(n - i) {
        let e = exp1(*x.add(i + j) - m);
        *out.add(i + j) = e;
        *smj += e;
    }
    let inv = 1.0 / hsum8(sm);
    let ivb = _mm256_set1_ps(inv);
    // 3) normalize
    i = 0;
    while i + 8 <= n {
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(_mm256_loadu_ps(out.add(i)), ivb));
        i += 8;
    }
    while i < n {
        *out.add(i) *= inv;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layernorm_row_avx2(x: *const f32, out: *mut f32, n: usize, eps: f32) {
    use std::arch::x86_64::*;
    let invn = 1.0 / (n as f32);
    // mean
    let mut sv = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        sv = _mm256_add_ps(sv, _mm256_loadu_ps(x.add(i)));
        i += 8;
    }
    let mut sm = [0.0f32; 8];
    _mm256_storeu_ps(sm.as_mut_ptr(), sv);
    for (j, smj) in sm.iter_mut().enumerate().take(n - i) {
        *smj += *x.add(i + j);
    }
    let mean = hsum8(sm) * invn;
    let mb = _mm256_set1_ps(mean);
    // variance
    let mut vv = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        let d = _mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb);
        vv = _mm256_fmadd_ps(d, d, vv);
        i += 8;
    }
    let mut va = [0.0f32; 8];
    _mm256_storeu_ps(va.as_mut_ptr(), vv);
    for (j, vaj) in va.iter_mut().enumerate().take(n - i) {
        let d = *x.add(i + j) - mean;
        *vaj = d.mul_add(d, *vaj);
    }
    let inv = 1.0 / (hsum8(va) * invn + eps).sqrt();
    let ivb = _mm256_set1_ps(inv);
    // normalize
    i = 0;
    while i + 8 <= n {
        let d = _mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb);
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(d, ivb));
        i += 8;
    }
    while i < n {
        *out.add(i) = (*x.add(i) - mean) * inv;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rmsnorm_row_avx2(x: *const f32, out: *mut f32, n: usize, eps: f32) {
    use std::arch::x86_64::*;
    let invn = 1.0 / (n as f32);
    let mut ssv = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.add(i));
        ssv = _mm256_fmadd_ps(v, v, ssv);
        i += 8;
    }
    let mut ss = [0.0f32; 8];
    _mm256_storeu_ps(ss.as_mut_ptr(), ssv);
    for (j, ssj) in ss.iter_mut().enumerate().take(n - i) {
        let v = *x.add(i + j);
        *ssj = v.mul_add(v, *ssj);
    }
    let inv = 1.0 / (hsum8(ss) * invn + eps).sqrt();
    let ivb = _mm256_set1_ps(inv);
    i = 0;
    while i + 8 <= n {
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(_mm256_loadu_ps(x.add(i)), ivb));
        i += 8;
    }
    while i < n {
        *out.add(i) = *x.add(i) * inv;
        i += 1;
    }
}

/// One row through the selected norm, AVX2 when available, else the scalar twin.
///
/// # Safety
/// `x` and `out` valid for `n` `f32`; they may alias.
#[inline]
unsafe fn norm_row(x: *const f32, out: *mut f32, n: usize, eps: f32, op: i64) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            match op {
                NORM_SOFTMAX => return softmax_row_avx2(x, out, n),
                NORM_LAYERNORM => return layernorm_row_avx2(x, out, n, eps),
                NORM_RMSNORM => return rmsnorm_row_avx2(x, out, n, eps),
                _ => return,
            }
        }
    }
    match op {
        NORM_SOFTMAX => softmax_row_scalar(x, out, n),
        NORM_LAYERNORM => layernorm_row_scalar(x, out, n, eps),
        NORM_RMSNORM => rmsnorm_row_scalar(x, out, n, eps),
        _ => {}
    }
}

/// Row-wise norm over a `[rows, cols]` matrix (serial). `eps_bits` is `eps.to_bits()` (the all-integer
/// ABI mirrors the other dispatch kernels; softmax ignores it). `x`/`out` may alias for in-place.
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn mercury_norm_f32(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
    eps_bits: i64,
    op: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    let eps = f32::from_bits(eps_bits as u32);
    for r in 0..rows {
        // SAFETY: row r occupies [r*cols, r*cols + cols) ⊆ [0, rows*cols).
        unsafe { norm_row(x.add(r * cols), out.add(r * cols), cols, eps, op) };
    }
}

/// Multicore row-wise norm — **bit-identical** to [`mercury_norm_f32`]. Rows are independent, so each
/// is computed by the same per-row routine regardless of which thread runs it; there is no cross-row
/// combine, so the result does not depend on thread count and the interpreter (serial) agrees with the
/// native `@parallel` path (this one) exactly.
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn mercury_norm_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
    eps_bits: i64,
    op: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    let eps = f32::from_bits(eps_bits as u32);
    // Raw pointers cross the rayon boundary as integers (same pattern as the parallel GEMM/reduce);
    // each row is a disjoint sub-slice.
    let (xa, oa) = (x as usize, out as usize);
    (0..rows).into_par_iter().for_each(|r| {
        // SAFETY: disjoint row; pointers valid for rows*cols by contract.
        unsafe {
            norm_row(
                (xa as *const f32).add(r * cols),
                (oa as *mut f32).add(r * cols),
                cols,
                eps,
                op,
            )
        };
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPS: [i64; 3] = [NORM_SOFTMAX, NORM_LAYERNORM, NORM_RMSNORM];
    const EPS: f32 = 1e-5;

    // Deterministic, mildly varied input (no RNG — reproducible).
    fn fill(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.017).sin() * 2.3 - 0.4)
            .collect()
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        // Sizes exercising the 8-lane body, a non-multiple-of-8 tail, and rows shorter than 8.
        for &n in &[1usize, 3, 7, 8, 9, 15, 16, 17, 31, 64, 100, 257] {
            let x = fill(n);
            for &op in &OPS {
                let mut a = vec![0.0f32; n];
                let mut b = vec![0.0f32; n];
                unsafe {
                    match op {
                        NORM_SOFTMAX => {
                            softmax_row_scalar(x.as_ptr(), a.as_mut_ptr(), n);
                            softmax_row_avx2(x.as_ptr(), b.as_mut_ptr(), n);
                        }
                        NORM_LAYERNORM => {
                            layernorm_row_scalar(x.as_ptr(), a.as_mut_ptr(), n, EPS);
                            layernorm_row_avx2(x.as_ptr(), b.as_mut_ptr(), n, EPS);
                        }
                        _ => {
                            rmsnorm_row_scalar(x.as_ptr(), a.as_mut_ptr(), n, EPS);
                            rmsnorm_row_avx2(x.as_ptr(), b.as_mut_ptr(), n, EPS);
                        }
                    }
                }
                for i in 0..n {
                    assert_eq!(
                        a[i].to_bits(),
                        b[i].to_bits(),
                        "scalar != avx2 at n={n} op={op} i={i}: {} vs {}",
                        a[i],
                        b[i]
                    );
                }
            }
        }
    }

    #[test]
    fn serial_matches_parallel_bit_for_bit() {
        let (rows, cols) = (37usize, 100usize);
        let x = fill(rows * cols);
        for &op in &OPS {
            let mut s = vec![0.0f32; rows * cols];
            let mut p = vec![0.0f32; rows * cols];
            unsafe {
                mercury_norm_f32(
                    x.as_ptr(),
                    s.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                    EPS.to_bits() as i64,
                    op,
                );
                mercury_norm_f32_parallel(
                    x.as_ptr(),
                    p.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                    EPS.to_bits() as i64,
                    op,
                );
            }
            for i in 0..rows * cols {
                assert_eq!(
                    s[i].to_bits(),
                    p[i].to_bits(),
                    "serial != parallel op={op} i={i}"
                );
            }
        }
    }

    #[test]
    fn in_place_matches_out_of_place() {
        // The recognizer allows in-place (x == out); it must equal the out-of-place result.
        let n = 130usize;
        let x = fill(n);
        for &op in &OPS {
            let mut oop = vec![0.0f32; n];
            let mut ip = x.clone();
            unsafe {
                mercury_norm_f32(
                    x.as_ptr(),
                    oop.as_mut_ptr(),
                    1,
                    n as i64,
                    EPS.to_bits() as i64,
                    op,
                );
                mercury_norm_f32(
                    ip.as_ptr(),
                    ip.as_mut_ptr(),
                    1,
                    n as i64,
                    EPS.to_bits() as i64,
                    op,
                );
            }
            for i in 0..n {
                assert_eq!(
                    oop[i].to_bits(),
                    ip[i].to_bits(),
                    "in-place != out-of-place op={op} i={i}"
                );
            }
        }
    }

    #[test]
    fn matches_f64_reference() {
        // Tolerance check against an f64 row computation for each norm.
        let n = 512usize;
        let x = fill(n);
        let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();

        // softmax
        let mut sm = vec![0.0f32; n];
        unsafe {
            mercury_norm_f32(x.as_ptr(), sm.as_mut_ptr(), 1, n as i64, 0, NORM_SOFTMAX);
        }
        let mx = xd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let den: f64 = xd.iter().map(|&v| (v - mx).exp()).sum();
        let mut tot = 0.0f64;
        for i in 0..n {
            let want = (xd[i] - mx).exp() / den;
            assert!((sm[i] as f64 - want).abs() < 1e-5, "softmax i={i}");
            tot += sm[i] as f64;
        }
        assert!((tot - 1.0).abs() < 1e-4, "softmax sums to {tot}");

        // layernorm
        let mut ln = vec![0.0f32; n];
        unsafe {
            mercury_norm_f32(
                x.as_ptr(),
                ln.as_mut_ptr(),
                1,
                n as i64,
                EPS.to_bits() as i64,
                NORM_LAYERNORM,
            );
        }
        let mean = xd.iter().sum::<f64>() / n as f64;
        let var = xd.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        let inv = 1.0 / (var + EPS as f64).sqrt();
        for i in 0..n {
            let want = (xd[i] - mean) * inv;
            assert!(
                (ln[i] as f64 - want).abs() < 1e-3,
                "layernorm i={i}: {} vs {want}",
                ln[i]
            );
        }

        // rmsnorm
        let mut rn = vec![0.0f32; n];
        unsafe {
            mercury_norm_f32(
                x.as_ptr(),
                rn.as_mut_ptr(),
                1,
                n as i64,
                EPS.to_bits() as i64,
                NORM_RMSNORM,
            );
        }
        let ms = xd.iter().map(|&v| v * v).sum::<f64>() / n as f64;
        let invr = 1.0 / (ms + EPS as f64).sqrt();
        for i in 0..n {
            let want = xd[i] * invr;
            assert!(
                (rn[i] as f64 - want).abs() < 1e-3,
                "rmsnorm i={i}: {} vs {want}",
                rn[i]
            );
        }
    }
}
