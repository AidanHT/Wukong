//! Fused, batched **Shannon entropy** over the rows of a row-major `[rows, cols]` probability matrix
//! `p` — the policy entropy of an RL actor (the exploration bonus `H(π(·|s))`) and the per-token
//! uncertainty of a language model. Per row (`C = cols`):
//!
//! ```text
//! out[r] = − Σ_i p[r,i] · log(p[r,i])
//! ```
//!
//! `out` has length `rows`. The body is a per-row reduction of `p·log(p)` capped with a single
//! negation — a transcendental reduction that **C/Rust keep scalar**: a `Σ p·logf(p)` loop won't
//! vectorize (libm has no AVX2 `logf`), so the 256-bit fused kernel — one row streamed once, the
//! `log` evaluated 8 lanes at a time through the *same* hand-vectorized polynomial the activation /
//! log-softmax kernels use — wins for the same reason the log-softmax / cross-entropy dispatch does.
//!
//! **Reuse for bit-exactness.** The `log` is the shared [`crate::vmath::log8`] (AVX2) / [`crate::vmath::log1`]
//! (scalar tail + no-AVX2 fallback) — never a private copy — so the result is bit-identical to a
//! `p·log(p)` reduction computed the standard way, and the scalar twin and the AVX2 path agree
//! **bit-for-bit**: the reduction keeps a fixed 8-lane accumulator (lane `j` folds elements
//! `≡ j (mod 8)`, the tail into lanes `0..`), the per-element contribution `p·log(p)` is a *mul then
//! add* (one product rounding, then accumulate — `_mm256_mul_ps`+`_mm256_add_ps` == scalar `*`+`+`,
//! deliberately **not** an FMA, so no fused-vs-split rounding mismatch), and both store the 8 lanes
//! to `[f32; 8]` and call the same balanced [`hsum8`], with the lone final negation applied to that
//! one scalar. Pinned by a unit test across non-multiple-of-8 `cols`.
//!
//! *The `p=0` convention.* The kernel computes `p·log(p)` **unconditionally** — there is no
//! special-case branch, so the scalar twin and the AVX2 path see the identical expression and stay
//! bit-for-bit (whatever `0·log(0)` yields, both yield the same bits). A recognizer matches the
//! idiomatic nest `s += p·log(p)` directly; callers pass strictly-positive probabilities (a valid
//! distribution), for which every term is finite.
//!
//! **Determinism.** Rows are independent, so the `_parallel` entry just maps the identical per-row
//! routine across rows — `serial == parallel` bit-for-bit with no cross-row combine, so the result
//! does not depend on core count and the interpreter's serial call agrees with the native `@parallel`
//! path exactly. (The per-row reduction reassociates across lanes — the documented
//! reassociated-reduction exception: every backend runs this same kernel, so they agree.)

use crate::vmath::log1;
#[cfg(target_arch = "x86_64")]
use crate::vmath::log8;
use rayon::prelude::*;

/// Fixed-order horizontal sum of 8 lane accumulators — a balanced tree, identical in the scalar twin
/// and the AVX2 path (which stores its `__m256` to `[f32; 8]` and calls this), so both give the same
/// bits. Same shape as `norm::hsum8` / the reduction kernel's `hcombine8`.
#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

// --- scalar twin (the AVX2 tail + the no-AVX2 fallback) -------------------------------------------

/// Shannon entropy of one row, scalar reference / AVX2 tail / no-AVX2 fallback.
/// Returns `− Σ_i p_i·log(p_i)`, accumulating `Σ p·log(p)` in a fixed 8-lane accumulator (lane `j`
/// folds elements `≡ j (mod 8)`, tail into lanes `0..`) exactly as the AVX2 path does, then one
/// `hsum8` and a single negation. Each term is `p·log1(p)` — a product (one rounding) then add, **not**
/// an FMA, matching the AVX2 `mul`+`add`.
///
/// # Safety
/// `p` must be valid for `n` `f32`.
#[inline]
unsafe fn entropy_row_scalar(p: *const f32, n: usize) -> f32 {
    let nb = n / 8;
    let t = nb * 8;
    // acc[j] = Σ over chunks of p·log(p) for elements ≡ j (mod 8) — same lane structure as the AVX2 path.
    let mut acc = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, accj) in acc.iter_mut().enumerate() {
            let v = *p.add(b + j);
            *accj += v * log1(v); // mul then add (one product rounding), matching _mm256_mul_ps + _mm256_add_ps.
        }
    }
    for (j, accj) in acc.iter_mut().enumerate().take(n - t) {
        let v = *p.add(t + j);
        *accj += v * log1(v);
    }
    -hsum8(acc) // entropy = −Σ p·log(p); the lone negation, applied to the one combined scalar.
}

// --- AVX2 kernel (mirrors the scalar twin lane-for-lane on finite inputs) --------------------------

/// Shannon entropy of one row, AVX2/FMA. Keeps one `__m256` accumulator whose lane `j` folds elements
/// `≡ j (mod 8)` exactly as the scalar twin does — each step adds `p · log8(p)` (an `_mm256_mul_ps`
/// then `_mm256_add_ps`, the same split rounding as the scalar `*`+`+`, **not** an FMA) — then stores
/// to `[f32; 8]`, calls the same [`hsum8`], and negates that one scalar. `log8` is the shared 8-lane
/// log, so it agrees with the scalar `log1` lane-for-lane.
///
/// # Safety
/// `p` valid for `n` `f32`; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn entropy_row_avx2(p: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut accv = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(p.add(i));
        // p·log(p), mul then add — identical split rounding to the scalar twin (no FMA).
        accv = _mm256_add_ps(accv, _mm256_mul_ps(v, log8(v)));
        i += 8;
    }
    let mut acc = [0.0f32; 8];
    _mm256_storeu_ps(acc.as_mut_ptr(), accv);
    // Tail: fold into the SAME lanes, same ops, as the scalar twin.
    for j in 0..(n - i) {
        let v = *p.add(i + j);
        acc[j] += v * log1(v);
    }
    -hsum8(acc) // same hsum8 + negation as the scalar twin, on the same lane bits.
}

/// One row through Shannon entropy, AVX2 when available, else the scalar twin.
///
/// # Safety
/// `p` valid for `n` `f32`.
#[inline]
unsafe fn entropy_row(p: *const f32, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return entropy_row_avx2(p, n);
        }
    }
    entropy_row_scalar(p, n)
}

// --- C-ABI entry points ---------------------------------------------------------------------------

/// Per-row **Shannon entropy** `out[r] = − Σ_i p[r,i]·log(p[r,i])` over a `[rows, cols]` row-major
/// probability matrix, single-threaded. `out` has length `rows`.
///
/// # Safety
/// `p` must be valid for `rows * cols` `f32`; `out` must be valid for `rows` `f32`.
#[no_mangle]
pub unsafe extern "C" fn mercury_entropy_f32(p: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    for row in 0..r {
        // SAFETY: row `row` occupies [row*c, row*c+c) ⊆ [0, rows*cols); out[row] ∈ [0, rows).
        *out.add(row) = entropy_row(p.add(row * c), c);
    }
}

/// Row count below which the parallel variant just runs serially (rayon's per-row overhead isn't worth
/// it for a handful of rows).
const ENTROPY_PAR_MIN: usize = 8;

/// Multicore **Shannon entropy** — **bit-identical** to [`mercury_entropy_f32`]. Rows are independent,
/// so each is computed by the same per-row routine regardless of which thread runs it; there is no
/// cross-row combine, so the result does not depend on thread count and the interpreter's serial call
/// agrees with this `@parallel` path exactly.
///
/// # Safety
/// `p` must be valid for `rows * cols` `f32`; `out` must be valid for `rows` `f32`.
#[no_mangle]
pub unsafe extern "C" fn mercury_entropy_f32_parallel(
    p: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r < ENTROPY_PAR_MIN {
        mercury_entropy_f32(p, out, rows, cols);
        return;
    }
    // Raw pointers cross the rayon boundary as integers (same pattern as the parallel norm/reduce);
    // each row reads a disjoint p-slice and writes one disjoint out slot.
    let (pa, oa) = (p as usize, out as usize);
    (0..r).into_par_iter().for_each(|row| {
        // SAFETY: disjoint row data / out slot; pointers re-derived from the captured addresses, valid
        // for rows*cols / rows.
        unsafe { *(oa as *mut f32).add(row) = entropy_row((pa as *const f32).add(row * c), c) };
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // A deterministic, strictly-positive probability row of length `cols` (no RNG — reproducible),
    // normalized to sum 1 so it is a real distribution. `sin`/`cos` of the index give a non-uniform
    // but always-positive shape; the `+ 0.5` floor keeps every entry > 0 (so `p·log(p)` is finite).
    fn fill_probs(rows: usize, cols: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let off = r * cols;
            let mut sum = 0.0f64;
            for i in 0..cols {
                // strictly positive, varied per (row, col).
                let w = (((r * 31 + i) as f32) * 0.017).sin() * 0.4
                    + (((i * 7 + 1) as f32) * 0.009).cos() * 0.3
                    + 1.0; // ∈ roughly [0.3, 1.7], always > 0.
                v[off + i] = w;
                sum += w as f64;
            }
            // normalize the row to sum 1 (a valid probability distribution).
            let inv = (1.0 / sum) as f32;
            for i in 0..cols {
                v[off + i] *= inv;
            }
        }
        v
    }

    /// (a) scalar path == AVX2 path bit-for-bit, across cols straddling the 8-lane edge (incl.
    /// non-multiples of 8), for several rows. This is the agreement the differential gate rests on.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for &cols in &[1usize, 7, 8, 9, 16, 17, 33, 64, 100, 257, 1000] {
            for &rows in &[1usize, 3, 5] {
                let p = fill_probs(rows, cols);
                for row in 0..rows {
                    let off = row * cols;
                    let a = unsafe { entropy_row_scalar(p[off..].as_ptr(), cols) };
                    let b = unsafe { entropy_row_avx2(p[off..].as_ptr(), cols) };
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "entropy scalar != avx2 rows={rows} cols={cols} row={row}: {a} vs {b}"
                    );
                }
            }
        }
    }

    /// (b) the kernel ≈ an independent **f64** reference within a tight tolerance (< 1e-5). An honest
    /// double-precision recompute of `−Σ p·log(p)` guards the *formula* (not just scalar==avx2).
    #[test]
    fn matches_f64_reference() {
        for &(rows, cols) in &[(1usize, 1usize), (3, 7), (4, 64), (5, 257), (8, 512)] {
            let p = fill_probs(rows, cols);
            let mut got = vec![0.0f32; rows];
            unsafe {
                mercury_entropy_f32(p.as_ptr(), got.as_mut_ptr(), rows as i64, cols as i64);
            }
            for r in 0..rows {
                let row: Vec<f64> = (0..cols).map(|i| p[r * cols + i] as f64).collect();
                let want = -row.iter().map(|&v| v * v.ln()).sum::<f64>();
                let denom = want.abs().max(1.0);
                assert!(
                    ((got[r] as f64 - want).abs() / denom) <= 1e-5,
                    "entropy f64 ref r={r} cols={cols}: {} vs {want}",
                    got[r]
                );
            }
        }
    }

    /// (c) serial == parallel bit-for-bit, with rows ≥ the parallel threshold (so the rayon path
    /// actually runs). Several cols straddle the 8-lane edge.
    #[test]
    fn serial_matches_parallel_bit_for_bit() {
        for &(rows, cols) in &[
            (ENTROPY_PAR_MIN, 1usize),
            (ENTROPY_PAR_MIN, 7),
            (ENTROPY_PAR_MIN, 8),
            (17, 33),
            (40, 100),
            (64, 257),
            (37, 1000),
        ] {
            let p = fill_probs(rows, cols);
            let mut s = vec![0.0f32; rows];
            let mut par = vec![0.0f32; rows];
            unsafe {
                mercury_entropy_f32(p.as_ptr(), s.as_mut_ptr(), rows as i64, cols as i64);
                mercury_entropy_f32_parallel(p.as_ptr(), par.as_mut_ptr(), rows as i64, cols as i64);
            }
            for r in 0..rows {
                assert_eq!(
                    s[r].to_bits(),
                    par[r].to_bits(),
                    "entropy serial != parallel rows={rows} cols={cols} r={r}"
                );
            }
        }
    }

    /// (d) closed-form sanity: a *uniform* row (every `p_i = 1/C`) has entropy `−Σ (1/C)·log(1/C) =
    /// log(C)`, the maximum-entropy distribution. Checked within f32 tolerance across several `cols`.
    #[test]
    fn uniform_row_entropy_is_log_cols() {
        for &cols in &[1usize, 2, 5, 8, 9, 16, 17, 100, 1000] {
            let p = vec![1.0f32 / cols as f32; cols];
            let mut got = [0.0f32; 1];
            unsafe {
                mercury_entropy_f32(p.as_ptr(), got.as_mut_ptr(), 1, cols as i64);
            }
            let want = (cols as f32).ln();
            assert!(
                (got[0] - want).abs() <= 1e-5 * want.max(1.0),
                "uniform entropy cols={cols}: {} vs log(C)={want}",
                got[0]
            );
        }
    }

    /// Edge: zero/negative rows or cols are a no-op (don't write, don't panic) — for both entries.
    #[test]
    fn degenerate_shapes_are_noops() {
        let p = fill_probs(1, 8);
        let mut buf = vec![42.0f32; 8];
        unsafe {
            mercury_entropy_f32(p.as_ptr(), buf.as_mut_ptr(), 0, 4);
            mercury_entropy_f32(p.as_ptr(), buf.as_mut_ptr(), 2, 0);
            mercury_entropy_f32_parallel(p.as_ptr(), buf.as_mut_ptr(), -1, 4);
            mercury_entropy_f32_parallel(p.as_ptr(), buf.as_mut_ptr(), 3, -2);
        }
        assert!(buf.iter().all(|&v| v == 42.0), "no-op must not write");
    }
}
