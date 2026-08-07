//! Per-row **Kullback–Leibler divergence** `KL(p ‖ q)` over two row-major `[rows, cols]` probability
//! matrices `p`, `q` — the knowledge-distillation loss (student `q` vs teacher `p`) and the VAE /
//! variational objective. One scalar per row:
//!
//! ```text
//! out[r] = Σ_i p[r,i] · (log(p[r,i]) − log(q[r,i]))
//! ```
//!
//! Each row is a reduction whose per-element term `p · (log p − log q)` evaluates **two** `log`s.
//! C/Rust keep `logf` scalar (a `logf` reduction loop won't vectorize — no AVX2 transcendental in
//! libm), so the fused 256-bit kernel — one row streamed once, the two `log`s + the dot in registers
//! with the same hand-vectorized `log` the rest of the suite uses — wins like the softmax / log-softmax
//! / cross-entropy dispatch.
//!
//! **Reuse for bit-exactness.** The `log` is the shared vmath one — AVX2 [`crate::vmath::log8`] on the
//! 8-lane body, scalar [`crate::vmath::log1`] on the tail and the no-AVX2 fallback — so the term is
//! bit-identical with the rest of the suite (cross-entropy's `log`, log-softmax's `log`). The Σ uses
//! the **same fixed 8-lane accumulator + balanced horizontal combine** the other per-row reductions use
//! (lane `j` folds elements `≡ j (mod 8)` in the same order, the same `hsum8`), so the AVX2 kernel and
//! the scalar twin agree **bit-for-bit** (pinned by a unit test across non-multiple-of-8 `cols`). The
//! per-element term is the single multiply `p · (log p − log q)` in both paths (no FMA: the natural
//! shape is a multiply of a difference, identical lane-for-lane).
//!
//! **Determinism.** Rows are independent, so the `_parallel` entry just maps the identical per-row
//! routine across rows — `serial == parallel` bit-for-bit with no cross-row combine (thread count is
//! irrelevant, so the interpreter's serial call agrees with the native `@parallel` path exactly). The
//! per-row Σ reassociates across lanes — the documented reassociated-reduction exception: every backend
//! runs this same kernel, so they agree.

use crate::vmath::log1;
#[cfg(target_arch = "x86_64")]
use crate::vmath::log8;
use rayon::prelude::*;

/// Fixed-order horizontal sum of 8 lane accumulators — a balanced tree, identical in the scalar twin
/// and the AVX2 path (which stores its `__m256` to `[f32; 8]` and calls this), so both give the same
/// bits. Same shape as `xent::hsum8` / `norm::hsum8`.
#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

// --- scalar twin (the AVX2 tail + the no-AVX2 fallback) -------------------------------------------

/// KL divergence of one row, scalar reference / AVX2 tail / no-AVX2 fallback. Returns
/// `Σ_i p_i · (log p_i − log q_i)`.
///
/// The Σ uses 8 lanes (lane `j` folds elements `≡ j (mod 8)`, tail into lanes `0..`) with the SAME
/// `hsum8` the other per-row reductions use, and the per-element term is the same single multiply
/// `p · (log1(p) − log1(q))` the AVX2 body computes — so the sum is bit-identical across paths.
///
/// # Safety
/// `p`, `q` valid for `n` `f32`.
unsafe fn kldiv_row_scalar(p: *const f32, q: *const f32, n: usize) -> f32 {
    let nb = n / 8;
    let t = nb * 8;
    // s = Σ p·(log p − log q) — one fixed 8-lane accumulator, same lane structure as the dot kernels.
    let mut sm = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, smj) in sm.iter_mut().enumerate() {
            let pv = *p.add(b + j);
            // term = p · (log p − log q); the single multiply the AVX2 body does (no FMA).
            *smj += pv * (log1(pv) - log1(*q.add(b + j)));
        }
    }
    for (j, smj) in sm.iter_mut().enumerate().take(n - t) {
        let pv = *p.add(t + j);
        *smj += pv * (log1(pv) - log1(*q.add(t + j)));
    }
    hsum8(sm)
}

// --- AVX2 kernel (mirrors the scalar twin lane-for-lane on finite inputs) --------------------------

/// KL divergence of one row, AVX2/FMA. One `__m256` accumulator whose lane `j` folds elements
/// `≡ j (mod 8)` exactly as the scalar twin, with the term `p · (log8(p) − log8(q))` per lane (the same
/// single multiply of a `log` difference); then store to `[f32; 8]` and call the same `hsum8`. The tail
/// folds into the SAME lanes via `log1`, so the sum matches the scalar twin bit-for-bit.
///
/// # Safety
/// `p`, `q` valid for `n` `f32`; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn kldiv_row_avx2(p: *const f32, q: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    // s = Σ p·(log p − log q) — one accumulator, same lane structure as the scalar twin.
    let mut sv = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let pv = _mm256_loadu_ps(p.add(i));
        let qv = _mm256_loadu_ps(q.add(i));
        // term = p · (log p − log q): the shared vmath log8 on both, one subtract, one multiply.
        let term = _mm256_mul_ps(pv, _mm256_sub_ps(log8(pv), log8(qv)));
        sv = _mm256_add_ps(sv, term);
        i += 8;
    }
    let mut sm = [0.0f32; 8];
    _mm256_storeu_ps(sm.as_mut_ptr(), sv);
    // Tail: fold into the SAME lanes, same ops (log1 == log8 lane-for-lane), as the scalar twin.
    for (j, smj) in sm.iter_mut().enumerate().take(n - i) {
        let pv = *p.add(i + j);
        *smj += pv * (log1(pv) - log1(*q.add(i + j)));
    }
    hsum8(sm)
}

/// One row through KL divergence, AVX2 when available, else the scalar twin.
///
/// # Safety
/// `p`, `q` valid for `n` `f32`.
#[inline]
unsafe fn kldiv_row(p: *const f32, q: *const f32, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return kldiv_row_avx2(p, q, n);
        }
    }
    kldiv_row_scalar(p, q, n)
}

/// Per-row KL divergence `out[r] = Σ_i p[r,i]·(log p[r,i] − log q[r,i])` over two `[rows, cols]`
/// row-major probability matrices, single-threaded. `out` has length `rows`.
///
/// # Safety
/// `p`, `q` valid for `rows*cols` f32; `out` valid for `rows` f32.
#[no_mangle]
pub unsafe extern "C" fn wukong_kldiv_f32(
    p: *const f32,
    q: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    for row in 0..r {
        // SAFETY: row `row` occupies [row*c, row*c+c) ⊆ [0, rows*cols); out[row] ∈ [0, rows).
        *out.add(row) = kldiv_row(p.add(row * c), q.add(row * c), c);
    }
}

/// Row count below which the parallel KL divergence just runs serially.
const KLDIV_PAR_MIN: usize = 8;

/// Multi-threaded KL divergence: rows are mapped across cores, each computed by the identical per-row
/// routine — so the result is **bit-identical to [`wukong_kldiv_f32`]** (rows are independent, no
/// cross-row combine, so thread count is irrelevant and the interpreter's serial call agrees with this
/// `@parallel` path exactly).
///
/// # Safety
/// Operand-size contract of [`wukong_kldiv_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_kldiv_f32_parallel(
    p: *const f32,
    q: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r < KLDIV_PAR_MIN {
        wukong_kldiv_f32(p, q, out, rows, cols);
        return;
    }
    // Raw pointers cross the rayon boundary as integers; each row reads disjoint p/q sub-slices and
    // writes one disjoint out slot (same pattern as the parallel xent / norm).
    let (p_addr, q_addr, o_addr) = (p as usize, q as usize, out as usize);
    // This fork can be the process's FIRST rayon touch, so it must provision the global pool first —
    // [`crate::ensure_global_pool`]'s stated precondition. Forking bare builds rayon's default
    // 2 MiB-stack registry, so the runtime's later 16 MiB `build_global` silently loses the race and
    // outlined `@parallel` region bodies are left on undersized stacks. Idempotent (`Once`) and
    // provisioning-only: the work split below is unchanged, so serial == parallel stays bit-exact.
    crate::ensure_global_pool();
    (0..r).into_par_iter().for_each(|row| {
        let off = row * c;
        // SAFETY: disjoint row slices; pointers re-derived from the captured addresses, valid for
        // rows*cols (p/q) and rows (out) by contract.
        unsafe {
            *(o_addr as *mut f32).add(row) = kldiv_row(
                (p_addr as *const f32).add(off),
                (q_addr as *const f32).add(off),
                c,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic, strictly-positive "probabilities" (no RNG — reproducible). Each row is normalized
    // to sum 1 (realistic distributions) so `p`/`q` are valid pmfs; the offsets/phases keep p ≠ q.
    fn fill_probs(rows: usize, cols: usize, phase: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let off = r * cols;
            // Strictly positive, varied across columns and rows.
            for i in 0..cols {
                v[off + i] =
                    (((i as f32) * 0.13 + (r as f32) * 0.07 + phase).sin() * 0.5 + 1.0) + 0.1;
            }
            // Normalize the row to sum 1.0 (a proper distribution).
            let s: f32 = v[off..off + cols].iter().sum();
            for i in 0..cols {
                v[off + i] /= s;
            }
        }
        v
    }

    /// (a) scalar == AVX2 bit-for-bit across several `cols` (incl. non-multiples of 8) and rows. The
    /// reduction-tail agreement that underwrites the differential gate.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        // Sizes exercising the 8-lane body, non-multiple-of-8 tails, and rows shorter than 8.
        for &cols in &[1usize, 7, 8, 9, 16, 17, 33, 64, 100, 257] {
            for &rows in &[1usize, 3, 5] {
                let p = fill_probs(rows, cols, 0.0);
                let q = fill_probs(rows, cols, 1.7);
                for row in 0..rows {
                    let off = row * cols;
                    let a = unsafe { kldiv_row_scalar(p[off..].as_ptr(), q[off..].as_ptr(), cols) };
                    let b = unsafe { kldiv_row_avx2(p[off..].as_ptr(), q[off..].as_ptr(), cols) };
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "scalar != avx2 rows={rows} cols={cols} row={row}: {a} vs {b}"
                    );
                }
            }
        }
    }

    /// (b) the kernel ≈ an independent **f64** reference within a tight tolerance (< 1e-5). An honest
    /// double-precision recompute of `Σ p·(log p − log q)` guards the *formula* (not just scalar==avx2).
    #[test]
    fn matches_f64_reference() {
        for &cols in &[1usize, 7, 8, 9, 17, 64, 257, 1000] {
            let rows = 4usize;
            let p = fill_probs(rows, cols, 0.0);
            let q = fill_probs(rows, cols, 1.7);
            let mut got = vec![0.0f32; rows];
            unsafe {
                wukong_kldiv_f32(
                    p.as_ptr(),
                    q.as_ptr(),
                    got.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for row in 0..rows {
                let off = row * cols;
                let want: f64 = (0..cols)
                    .map(|i| {
                        let pv = p[off + i] as f64;
                        let qv = q[off + i] as f64;
                        pv * (pv.ln() - qv.ln())
                    })
                    .sum();
                let denom = want.abs().max(1.0);
                assert!(
                    ((got[row] as f64 - want).abs() / denom) <= 1e-5,
                    "f64 ref rows={rows} cols={cols} row={row}: {} vs {want}",
                    got[row]
                );
            }
        }
    }

    /// (c) serial == parallel bit-for-bit at rows ≥ the parallel threshold (rows are independent, no
    /// cross-row combine, so thread count is irrelevant).
    #[test]
    fn serial_matches_parallel_bit_for_bit() {
        for (rows, cols) in [(8usize, 1usize), (37, 7), (64, 100), (128, 257)] {
            let p = fill_probs(rows, cols, 0.0);
            let q = fill_probs(rows, cols, 1.7);
            let mut s = vec![0.0f32; rows];
            let mut par = vec![0.0f32; rows];
            unsafe {
                wukong_kldiv_f32(
                    p.as_ptr(),
                    q.as_ptr(),
                    s.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
                wukong_kldiv_f32_parallel(
                    p.as_ptr(),
                    q.as_ptr(),
                    par.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for row in 0..rows {
                assert_eq!(
                    s[row].to_bits(),
                    par[row].to_bits(),
                    "serial != parallel rows={rows} cols={cols} row={row}"
                );
            }
        }
    }

    /// (d) KL(p ‖ p) ≈ 0 — the known property (a distribution has zero divergence from itself). Checked
    /// within a tight f32 tolerance across several `cols` (the per-element `p·(log p − log p)` is `p·0`
    /// in exact arithmetic; rounding of the two independent `log p` evaluations leaves a tiny residual).
    #[test]
    fn kl_of_self_is_zero() {
        for &cols in &[1usize, 2, 5, 8, 9, 16, 17, 100, 1000] {
            let rows = 3usize;
            let p = fill_probs(rows, cols, 0.4);
            let mut got = vec![0.0f32; rows];
            unsafe {
                // q == p.
                wukong_kldiv_f32(
                    p.as_ptr(),
                    p.as_ptr(),
                    got.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for row in 0..rows {
                assert!(
                    got[row].abs() <= 1e-6,
                    "KL(p‖p) != 0 cols={cols} row={row}: {}",
                    got[row]
                );
            }
        }
    }

    /// Edge: zero/negative rows or cols are a no-op (don't write, don't panic).
    #[test]
    fn degenerate_shapes_are_noops() {
        let p = fill_probs(2, 4, 0.0);
        let q = fill_probs(2, 4, 1.0);
        let mut out = vec![42.0f32; 2];
        unsafe {
            wukong_kldiv_f32(p.as_ptr(), q.as_ptr(), out.as_mut_ptr(), 0, 4);
            wukong_kldiv_f32(p.as_ptr(), q.as_ptr(), out.as_mut_ptr(), 2, 0);
            wukong_kldiv_f32_parallel(p.as_ptr(), q.as_ptr(), out.as_mut_ptr(), -1, 4);
        }
        assert!(out.iter().all(|&v| v == 42.0), "no-op must not write out");
    }
}
