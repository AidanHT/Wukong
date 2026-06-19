//! Single-precision reductions — dot product and friends, with a **deterministic multicore** variant.
//!
//! Computes `reduce_i f(x[i], y[i])` for `f` in `{ x·y (dot), (x−y)² (ssd), x (sum), x·x (sumsq) }`
//! folded by `+`, and `{ x (max), x (min) }` folded by `fmax`/`fmin`. These are the reductions
//! transformer math leans on: attention scores and projections (dot), the L2 loss (ssd),
//! LayerNorm/RMSNorm mean & variance (sum, sumsq), and the per-tensor **max/absmax** that softmax
//! stability and dynamic int8 quantization scale-computation need (max, min). The compiler recognizes
//! the reduction loop in a `@parallel` function and lowers it to one of these calls — the same play as
//! the matmul→GEMM and activation→`mercury_vmath_f32` dispatch. The interpreter marshals its abstract
//! memory through the **identical serial kernel**, so the differential oracle stays bit-for-bit exact.
//!
//! **Determinism is the whole game for a *parallel* reduction** — the result must not depend on how
//! many cores ran it. So the array is cut into FIXED-size chunks (count independent of thread count);
//! each chunk is reduced to a partial by the identical [`reduce_chunk`]; the partials are folded in
//! ascending chunk order. The serial and parallel entries call the same per-chunk function and combine
//! in the same order, so `serial == parallel == interpreter`, bit for bit, on any machine. (`fmax`/
//! `fmin` are *not* associative on NaN/±0, but determinism here rests on the FIXED decomposition and
//! ascending combine, not on associativity — the serial and parallel forms evaluate the identical
//! expression tree.) Within a chunk the AVX2 path and the scalar twin are also bit-identical (lane `j`
//! folds elements `≡ j (mod 8)`, then a fixed-order horizontal combine; `f32::mul_add` is the same
//! fused op as `_mm256_fmadd`, and the scalar `(a > b) ? a : b` is the same as `_mm256_max_ps`).
//!
//! At `N = 2^20` a dot is *memory-bound* (a single core already saturates load bandwidth at ~44
//! GB/s), so one 8-lane accumulator per chunk is enough to be bandwidth-bound; the win is spreading
//! chunks across cores to reach *aggregate* bandwidth — exactly how `saxpy@parallel` hits ~130 GB/s.

use rayon::prelude::*;

// Reduction op codes (shared with the recognizer in `mercury_mir_build`). For the unary ops (`SUM`,
// `SUMSQ`) the recognizer passes `y == x` so `y` is always a valid pointer and the kernel simply
// never reads it — no null-pointer marshalling anywhere.
pub const RED_DOT: i64 = 0; // sum(x[i] * y[i])
pub const RED_SSD: i64 = 1; // sum((x[i] - y[i])^2)
pub const RED_SUM: i64 = 2; // sum(x[i])
pub const RED_SUMSQ: i64 = 3; // sum(x[i] * x[i])
pub const RED_MAX: i64 = 4; // max(x[i])  — fold by fmax
pub const RED_MIN: i64 = 5; // min(x[i])  — fold by fmin
pub const RED_MAXABS: i64 = 6; // max(|x[i]|) — abs each element, fold by fmax (symmetric int8 quant)

/// The fold identity: `0.0` for the additive ops, `∓∞` for max/min/maxabs so the first real element
/// wins (`maxabs` folds by max, identity `−∞`).
#[inline(always)]
fn ident(op: i64) -> f32 {
    match op {
        RED_MAX | RED_MAXABS => f32::NEG_INFINITY,
        RED_MIN => f32::INFINITY,
        _ => 0.0,
    }
}

/// Fold two partials under the reduction's combine: `a + b` (additive), or `(a > b) ? a : b` /
/// `(a < b) ? a : b` for max/min — the exact semantics of `_mm256_max_ps`/`_mm256_min_ps` (and of
/// the MIR `Cmp(Fogt/Folt)+Select` the recognizer emits to combine the kernel result), so the AVX2
/// lanes, the scalar twin, and the compiler's outer fold all agree bit-for-bit.
#[inline(always)]
fn fold2(a: f32, b: f32, op: i64) -> f32 {
    match op {
        // maxabs folds its (already abs'd) partials by plain max.
        RED_MAX | RED_MAXABS => {
            if a > b {
                a
            } else {
                b
            }
        }
        RED_MIN => {
            if a < b {
                a
            } else {
                b
            }
        }
        _ => a + b,
    }
}

// Fixed chunk size in elements — independent of thread count, which is what makes the parallel
// partial decomposition deterministic. 8192 f32 = 32 KB (an L1's worth) per chunk; at N=2^20 that is
// 128 chunks, plenty for rayon work-stealing to balance the P+E hybrid, and the 128-entry partial
// array combines in negligible time. Must stay constant for serial/parallel/interp to agree.
const RCHUNK: usize = 8192;

/// One element's contribution folded into accumulator `a` (the fused-multiply-add form, so the AVX2
/// `_mm256_fmadd_ps` lanes match this lane-for-lane). `yi` is ignored for the unary ops.
#[inline(always)]
fn contrib(a: f32, xi: f32, yi: f32, op: i64) -> f32 {
    match op {
        RED_DOT => xi.mul_add(yi, a),
        RED_SSD => {
            let d = xi - yi;
            d.mul_add(d, a)
        }
        RED_SUM => a + xi,
        RED_SUMSQ => xi.mul_add(xi, a),
        RED_MAX | RED_MIN => fold2(a, xi, op), // `(a > xi) ? a : xi` ≡ `_mm256_max_ps(a, xi)`
        RED_MAXABS => fold2(a, xi.abs(), RED_MAX), // `f32::abs` clears the sign bit ≡ `andnot(-0, xi)`
        _ => a,
    }
}

/// Fixed-order horizontal combine of the 8 lane accumulators: a balanced tree built from [`fold2`],
/// identical in the AVX2 and scalar paths so both produce the same bits. For the additive ops this is
/// exactly `((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))` (unchanged); for max/min it is the same tree under
/// `fmax`/`fmin`.
#[inline(always)]
fn hcombine8(a: [f32; 8], op: i64) -> f32 {
    fold2(
        fold2(fold2(a[0], a[1], op), fold2(a[2], a[3], op), op),
        fold2(fold2(a[4], a[5], op), fold2(a[6], a[7], op), op),
        op,
    )
}

/// Reduce `x[lo..hi]` (and `y[lo..hi]` for the binary ops) to a scalar partial. Pure function of its
/// chunk, so it returns the same bits no matter which thread (or the serial loop) calls it.
///
/// # Safety
/// `x` and `y` must be valid for reads on `[lo, hi)`.
#[inline]
unsafe fn reduce_chunk(x: *const f32, y: *const f32, lo: usize, hi: usize, op: i64) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; range validity is the caller's contract.
            return unsafe { reduce_chunk_avx2(x, y, lo, hi, op) };
        }
    }
    // SAFETY: range validity is the caller's contract.
    unsafe { reduce_chunk_scalar(x, y, lo, hi, op) }
}

/// Portable reference: 8 scalar lane accumulators mirroring the AVX2 register lane-for-lane. Used on
/// non-AVX2 targets; the unit tests pin it bit-identical to the AVX2 path where that is available.
///
/// # Safety
/// `x`/`y` valid on `[lo, hi)`.
unsafe fn reduce_chunk_scalar(x: *const f32, y: *const f32, lo: usize, hi: usize, op: i64) -> f32 {
    let mut acc = [ident(op); 8];
    let len = hi - lo;
    let nsteps = len / 8;
    for s in 0..nsteps {
        let base = lo + s * 8;
        for (j, a) in acc.iter_mut().enumerate() {
            let i = base + j;
            // SAFETY: i < hi.
            *a = contrib(*a, unsafe { *x.add(i) }, unsafe { *y.add(i) }, op);
        }
    }
    // Tail (< 8 elements) folds into lanes 0.. in order — identical to the AVX2 store-then-tail.
    let tail = lo + nsteps * 8;
    for (j, a) in acc.iter_mut().enumerate().take(len - nsteps * 8) {
        let i = tail + j;
        // SAFETY: i < hi.
        *a = contrib(*a, unsafe { *x.add(i) }, unsafe { *y.add(i) }, op);
    }
    hcombine8(acc, op)
}

/// AVX2 chunk reduce: one `__m256` accumulator (lane `j` sums elements `≡ j (mod 8)` over the chunk),
/// then the tail and horizontal combine in scalar so the result is bit-identical to the scalar twin.
///
/// # Safety
/// `x`/`y` valid on `[lo, hi)`; AVX2+FMA must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn reduce_chunk_avx2(x: *const f32, y: *const f32, lo: usize, hi: usize, op: i64) -> f32 {
    use std::arch::x86_64::*;
    // `set1(ident)` is `setzero` for the additive ops (+0.0 = all-zero bits), and ∓∞ for max/min.
    let mut acc = _mm256_set1_ps(ident(op));
    let len = hi - lo;
    let nsteps = len / 8;
    for s in 0..nsteps {
        let i = lo + s * 8;
        let xv = _mm256_loadu_ps(x.add(i));
        acc = match op {
            RED_DOT => _mm256_fmadd_ps(xv, _mm256_loadu_ps(y.add(i)), acc),
            RED_SSD => {
                let d = _mm256_sub_ps(xv, _mm256_loadu_ps(y.add(i)));
                _mm256_fmadd_ps(d, d, acc)
            }
            RED_SUM => _mm256_add_ps(xv, acc),
            RED_SUMSQ => _mm256_fmadd_ps(xv, xv, acc),
            RED_MAX => _mm256_max_ps(acc, xv), // `(acc > xv) ? acc : xv`, lane-wise
            RED_MIN => _mm256_min_ps(acc, xv),
            // |xv| via `andnot(-0.0, xv)` (clear the sign bit) ≡ the scalar `f32::abs`, then max.
            RED_MAXABS => _mm256_max_ps(acc, _mm256_andnot_ps(_mm256_set1_ps(-0.0), xv)),
            _ => acc,
        };
    }
    let mut tmp = [ident(op); 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    // Scalar tail into the same lanes — `contrib` matches the AVX2 lane op exactly (mul_add↔fmadd,
    // `(a > xi) ? a : xi` ↔ max_ps).
    let tail = lo + nsteps * 8;
    for (j, t) in tmp.iter_mut().enumerate().take(len - nsteps * 8) {
        let i = tail + j;
        *t = contrib(*t, *x.add(i), *y.add(i), op);
    }
    hcombine8(tmp, op)
}

/// `sum_i f(x[i], y[i])` for `i in 0..n` (serial). For `RED_SUM`/`RED_SUMSQ`, `y` is ignored (callers
/// pass `y == x`). The chunk decomposition matches [`mercury_sreduce_f32_parallel`] exactly.
///
/// # Safety
/// `x` and `y` must each be valid for `n` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn mercury_sreduce_f32(x: *const f32, y: *const f32, n: i64, op: i64) -> f32 {
    if n <= 0 {
        return ident(op);
    }
    let n = n as usize;
    let nchunks = n.div_ceil(RCHUNK);
    let mut acc = ident(op);
    for c in 0..nchunks {
        let lo = c * RCHUNK;
        let hi = ((c + 1) * RCHUNK).min(n);
        // SAFETY: [lo, hi) ⊆ [0, n); buffers valid for n by contract.
        acc = fold2(acc, unsafe { reduce_chunk(x, y, lo, hi, op) }, op);
    }
    acc
}

/// Multicore `sum_i f(x[i], y[i])` — **bit-identical** to [`mercury_sreduce_f32`]. Each fixed chunk is
/// reduced independently (rayon), the partials are collected *in index order* (rayon's indexed
/// `map().collect()` preserves order regardless of which thread computed each), then summed ascending.
/// So the float result is the same whether one core or sixteen ran it, and the interpreter (which
/// calls the serial form) agrees with the native `@parallel` path that calls this one.
///
/// # Safety
/// `x` and `y` must each be valid for `n` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn mercury_sreduce_f32_parallel(
    x: *const f32,
    y: *const f32,
    n: i64,
    op: i64,
) -> f32 {
    if n <= 0 {
        return ident(op);
    }
    let n = n as usize;
    let nchunks = n.div_ceil(RCHUNK);
    if nchunks < 2 {
        // SAFETY: same contract.
        return unsafe { mercury_sreduce_f32(x, y, n as i64, op) };
    }
    // Raw pointers cross the rayon closure boundary as integers (same pattern as the parallel GEMM);
    // every access is a disjoint read-only chunk.
    let (xa, ya) = (x as usize, y as usize);
    let partials: Vec<f32> = (0..nchunks)
        .into_par_iter()
        .map(|c| {
            let lo = c * RCHUNK;
            let hi = ((c + 1) * RCHUNK).min(n);
            // SAFETY: disjoint read-only chunk; pointers valid for n by contract.
            unsafe { reduce_chunk(xa as *const f32, ya as *const f32, lo, hi, op) }
        })
        .collect();
    let mut acc = ident(op);
    for p in partials {
        acc = fold2(acc, p, op);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    // A deterministic, mildly varied input (no RNG — keeps the test reproducible across runs).
    fn fill(n: usize) -> (Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.013).sin() * 1.7 + 0.3)
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.027).cos() * 0.9 - 0.2)
            .collect();
        (x, y)
    }

    fn naive(x: &[f32], y: &[f32], op: i64) -> f64 {
        // f64 reference for the tolerance check. For max/min the fold picks an actual element, so the
        // f64 result equals the f32 kernel's bit-for-bit on the finite, non-NaN test data.
        let mut s = ident(op) as f64;
        for i in 0..x.len() {
            let (xi, yi) = (x[i] as f64, y[i] as f64);
            match op {
                RED_DOT => s += xi * yi,
                RED_SSD => s += (xi - yi) * (xi - yi),
                RED_SUM => s += xi,
                RED_SUMSQ => s += xi * xi,
                RED_MAX => s = if s > xi { s } else { xi },
                RED_MIN => s = if s < xi { s } else { xi },
                RED_MAXABS => {
                    let axi = xi.abs();
                    s = if s > axi { s } else { axi }
                }
                _ => {}
            };
        }
        s
    }

    const OPS: [i64; 7] = [
        RED_DOT, RED_SSD, RED_SUM, RED_SUMSQ, RED_MAX, RED_MIN, RED_MAXABS,
    ];

    // The unary ops read only `x`; the recognizer passes `y == x` for them.
    fn unary(op: i64) -> bool {
        matches!(op, RED_SUM | RED_SUMSQ | RED_MAX | RED_MIN | RED_MAXABS)
    }

    #[test]
    fn serial_matches_parallel_bit_for_bit() {
        // Sizes that exercise multiple chunks, a partial final chunk, and a non-multiple-of-8 tail.
        for &n in &[1usize, 7, 8, 9, 8192, 8193, 3 * 8192 + 13, 100_003] {
            let (x, y) = fill(n);
            for &op in &OPS {
                // For the unary ops the recognizer passes y == x; mirror that here.
                let yp = if unary(op) { x.as_ptr() } else { y.as_ptr() };
                let s = unsafe { mercury_sreduce_f32(x.as_ptr(), yp, n as i64, op) };
                let p = unsafe { mercury_sreduce_f32_parallel(x.as_ptr(), yp, n as i64, op) };
                assert_eq!(
                    s.to_bits(),
                    p.to_bits(),
                    "serial != parallel at n={n} op={op}: {s} vs {p}"
                );
            }
        }
    }

    #[test]
    fn within_tolerance_of_f64_reference() {
        let n = 100_003;
        let (x, y) = fill(n);
        for &op in &OPS {
            let yp = if unary(op) { x.as_ptr() } else { y.as_ptr() };
            let got = unsafe { mercury_sreduce_f32(x.as_ptr(), yp, n as i64, op) } as f64;
            let want = naive(&x, &y, op);
            let rel = (got - want).abs() / want.abs().max(1.0);
            assert!(rel < 1e-4, "op={op}: got {got}, want {want}, rel {rel}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for &n in &[8usize, 9, 17, 8191, 8192, 8193] {
            let (x, y) = fill(n);
            for &op in &OPS {
                let s = unsafe { reduce_chunk_scalar(x.as_ptr(), y.as_ptr(), 0, n, op) };
                let v = unsafe { reduce_chunk_avx2(x.as_ptr(), y.as_ptr(), 0, n, op) };
                assert_eq!(s.to_bits(), v.to_bits(), "scalar != avx2 at n={n} op={op}");
            }
        }
    }
}
