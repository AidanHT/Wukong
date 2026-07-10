//! Fused **soft-label cross-entropy** — the knowledge-distillation loss — over a row-major
//! `[rows, cols]` f32 logits matrix `x` and a per-row **soft target distribution** `q` (each row a
//! probability vector summing to 1). It is the cross-entropy of the student's softmax against a
//! teacher-provided soft label, the loss that trains a distilled model.
//!
//! The numerically-stable per-row form (`C = cols`):
//!
//! ```text
//! m      = max_i x[r,i]
//! lse    = m + log(Σ_i exp(x[r,i] − m))     // log-sum-exp (the stable softmax denominator)
//! out[r] = Σ_i q[r,i] · (lse − x[r,i])      // = −Σ_i q[r,i]·log_softmax(x[r,·])[i]
//! ```
//!
//! This is exactly softmax's stabilizing **row max + Σexp(x−m)** reduction (the same two passes
//! `norm.rs`'s softmax and the hard-label [`crate::wukong_xent_fwd_f32`] run), capped with one `log`
//! of the row sum and then a **second reduction** `Σ q·(lse − x)`. It **generalizes the hard-label
//! cross-entropy** in `xent.rs`: a one-hot `q` (all mass on class `t`) collapses the final sum to
//! `lse − x[t]`, the hard-label loss. C/Rust keep `expf`/`logf` scalar (a loop with the libm calls
//! won't vectorize), so the 256-bit fused reduction wins for the same reason the softmax/log-softmax
//! dispatch does.
//!
//! **Reuse for bit-exactness.** The max pass uses the same fixed 8-lane accumulator + balanced
//! horizontal combine softmax uses, and the Σexp pass reuses the *exact* shared `exp` polynomials —
//! AVX2 [`crate::vmath::exp8`] on the 8-lane body, the scalar [`crate::vmath::exp1`] on the tail and
//! the no-AVX2 fallback — with the row sum's `log` taken through the shared [`crate::vmath::log1`].
//! The final `Σ q·(lse − x)` pass uses a second fixed 8-lane accumulator with the same balanced
//! `hsum8` (one FMA per lane: `q·lse − q·x` accumulated). So the loss is bit-identical to the
//! hard-label kernel's `lse` computation, and the AVX2 path and the scalar twin agree **bit-for-bit**
//! (pinned by a unit test across non-multiple-of-8 `cols`).
//!
//! **Determinism.** Rows are independent, so the `_parallel` entry just maps the identical per-row
//! routine across rows — `serial == parallel` bit-for-bit with no cross-row combine (thread count is
//! irrelevant, so the interpreter's serial call agrees with the native `@parallel` path exactly). The
//! per-row max, Σexp and the final q-weighted sum reassociate across lanes — the documented
//! reassociated-reduction exception: every backend runs this same kernel, so they agree.

use crate::vmath::{exp1, log1};
#[cfg(target_arch = "x86_64")]
use crate::vmath::exp8;
use rayon::prelude::*;

/// Fixed-order horizontal sum of 8 lane accumulators — a balanced tree, identical in the scalar twin
/// and the AVX2 path (which stores its `__m256` to `[f32; 8]` and calls this), so both give the same
/// bits. Same shape as `xent::hsum8` / `norm::hsum8`.
#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

/// Fixed-order horizontal max of 8 lane accumulators (balanced tree). On finite inputs this matches
/// `_mm256_max_ps` lane-for-lane; the same `hmax8` softmax / `xent` uses. The twin test pins the
/// agreement on finite data.
#[inline(always)]
fn hmax8(a: [f32; 8]) -> f32 {
    (a[0].max(a[1]).max(a[2].max(a[3]))).max(a[4].max(a[5]).max(a[6].max(a[7])))
}

// --- scalar twin (the AVX2 tail + the no-AVX2 fallback) -------------------------------------------

/// Soft-label cross-entropy of one row, scalar reference / AVX2 tail / no-AVX2 fallback. Returns
/// `Σ_i q_i·(lse − x_i)` where `lse = m + log(Σ exp(x − m))`, `m = max(x)`.
///
/// The max pass (8 lanes; lane `j` folds elements `≡ j (mod 8)`, tail into lanes `0..`) and the Σexp
/// pass are byte-identical to softmax's / `xent`'s in `norm.rs`/`xent.rs` — same `hmax8`, same
/// `exp1`, same `hsum8` — so `m` and the `lse` are bit-identical to the hard-label kernel's. The new
/// pass folds `q·(lse − x)` with the same 8-lane layout and `hsum8`: per lane one `mul_add` of
/// `q·(lse − x)` computed as `q.mul_add(lse, -(q*x))` accumulated — the same op order the AVX2 path
/// uses.
///
/// # Safety
/// `x`, `q` valid for `n` `f32`.
unsafe fn kd_loss_row_scalar(x: *const f32, q: *const f32, n: usize) -> f32 {
    let nb = n / 8;
    let t = nb * 8;
    // 1) row max — identical lane structure to softmax_row_scalar / xent_row_scalar.
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
    // 2) s = Σ exp(x − m) — same lane structure as softmax's / xent's sum.
    let mut sm = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, smj) in sm.iter_mut().enumerate() {
            *smj += exp1(*x.add(b + j) - m);
        }
    }
    for (j, smj) in sm.iter_mut().enumerate().take(n - t) {
        *smj += exp1(*x.add(t + j) - m);
    }
    let lse = m + log1(hsum8(sm)); // log-sum-exp; ONE scalar log — same op/bits as xent's `lse`.
    // 3) out = Σ q·(lse − x) — a second 8-lane reduction with the same hsum8. Per lane:
    //    acc = q·lse + (−q·x)  via  q.mul_add(lse, -(q*x))  — the exact op order the AVX2 path uses.
    let mut acc = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, accj) in acc.iter_mut().enumerate() {
            let qj = *q.add(b + j);
            *accj += qj.mul_add(lse, -(qj * *x.add(b + j)));
        }
    }
    for (j, accj) in acc.iter_mut().enumerate().take(n - t) {
        let qj = *q.add(t + j);
        *accj += qj.mul_add(lse, -(qj * *x.add(t + j)));
    }
    hsum8(acc)
}

// --- AVX2 kernel (mirrors the scalar twin lane-for-lane on finite inputs) --------------------------

/// Soft-label cross-entropy of one row, AVX2/FMA. The max + Σexp passes are byte-identical to
/// `softmax_row_avx2` / `xent_row_avx2` (so `m` and `lse` match bit-for-bit), capped with one scalar
/// `log1`; then the q-weighted sum keeps one `__m256` accumulator whose lane `j` folds elements
/// `≡ j (mod 8)` exactly as the scalar twin does (`fmadd(q, lse, -(q*x))` == the scalar `mul_add`),
/// stores to `[f32; 8]` and calls the same `hsum8`.
///
/// # Safety
/// `x`, `q` valid for `n` `f32`; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn kd_loss_row_avx2(x: *const f32, q: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    // 1) max — identical to softmax_row_avx2 / xent_row_avx2.
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
    // 2) s = Σ exp(x − m) — exp8 lanes + exp1 tail, identical to softmax's / xent's accumulation.
    let mut sv = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        sv = _mm256_add_ps(sv, exp8(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb)));
        i += 8;
    }
    let mut sm = [0.0f32; 8];
    _mm256_storeu_ps(sm.as_mut_ptr(), sv);
    for (j, smj) in sm.iter_mut().enumerate().take(n - i) {
        *smj += exp1(*x.add(i + j) - m);
    }
    let lse = m + log1(hsum8(sm)); // same scalar log on the same sum bits as the scalar twin.
    // 3) out = Σ q·(lse − x): one accumulator, lane `j` folds elements ≡ j (mod 8). Per lane the op
    //    is fmadd(q, lse, -(q*x)) == the scalar twin's q.mul_add(lse, -(q*x)).
    let lsev = _mm256_set1_ps(lse);
    let zero = _mm256_setzero_ps();
    let mut av = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        let qv = _mm256_loadu_ps(q.add(i));
        let xv = _mm256_loadu_ps(x.add(i));
        // av += q*lse - q*x  ==  fmadd(q, lse, -(q*x)) with -(q*x) = 0 - (q*x).
        let nqx = _mm256_sub_ps(zero, _mm256_mul_ps(qv, xv));
        av = _mm256_add_ps(av, _mm256_fmadd_ps(qv, lsev, nqx));
        i += 8;
    }
    let mut acc = [0.0f32; 8];
    _mm256_storeu_ps(acc.as_mut_ptr(), av);
    // Tail: fold into the SAME lanes, same op, as the scalar twin.
    for (j, accj) in acc.iter_mut().enumerate().take(n - i) {
        let qj = *q.add(i + j);
        *accj += qj.mul_add(lse, -(qj * *x.add(i + j)));
    }
    hsum8(acc)
}

/// One row through soft-label cross-entropy, AVX2 when available, else the scalar twin.
///
/// # Safety
/// `x`, `q` valid for `n` `f32`.
#[inline]
unsafe fn kd_loss_row(x: *const f32, q: *const f32, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return kd_loss_row_avx2(x, q, n);
        }
    }
    kd_loss_row_scalar(x, q, n)
}

/// Fused soft-label cross-entropy `out[r] = Σ_i q[r,i]·(lse_r − x[r,i])` over a `[rows, cols]`
/// row-major logits matrix `x` and a matching soft-target matrix `q`, single-threaded. `out` has
/// length `rows`; each row of `q` should be a probability vector (the formula is computed verbatim
/// regardless, so callers own the normalization).
///
/// # Safety
/// `x`, `q` valid for `rows*cols` f32; `out` valid for `rows`.
#[no_mangle]
pub unsafe extern "C" fn wukong_kd_loss_f32(
    x: *const f32,
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
        // SAFETY: row `row` occupies [row*c, row*c+c) ⊆ [0, rows*cols) for both x and q.
        let off = row * c;
        *out.add(row) = kd_loss_row(x.add(off), q.add(off), c);
    }
}

/// Row count below which the parallel soft-label cross-entropy just runs serially.
const KD_LOSS_PAR_MIN: usize = 8;

/// Multi-threaded soft-label cross-entropy: rows are mapped across cores, each computed by the
/// identical per-row routine — so the result is **bit-identical to [`wukong_kd_loss_f32`]** (rows
/// are independent, no cross-row combine, so thread count is irrelevant and the interpreter's serial
/// call agrees with this `@parallel` path exactly).
///
/// # Safety
/// Operand-size contract of [`wukong_kd_loss_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_kd_loss_f32_parallel(
    x: *const f32,
    q: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r < KD_LOSS_PAR_MIN {
        wukong_kd_loss_f32(x, q, out, rows, cols);
        return;
    }
    // Raw pointers cross the rayon boundary as integers; each row is a disjoint sub-slice of x/q, and
    // `out` is indexed by row.
    let (x_addr, q_addr, o_addr) = (x as usize, q as usize, out as usize);
    (0..r).into_par_iter().for_each(|row| {
        let off = row * c;
        // SAFETY: disjoint row slices; pointers re-derived from the captured addresses.
        unsafe {
            *(o_addr as *mut f32).add(row) = kd_loss_row(
                (x_addr as *const f32).add(off),
                (q_addr as *const f32).add(off),
                c,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wukong_xent_fwd_f32;

    // Deterministic, mildly varied logits (no RNG — reproducible). Same stream shape as xent.rs's
    // `fill` so the row reductions exercise realistic magnitudes.
    fn fill(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.017).sin() * 2.3 - 0.4)
            .collect()
    }

    // A deterministic, strictly-positive soft target for each row, normalized to sum 1. The raw
    // weights are an exp of a distinct stream so every entry is > 0; each row is divided by its
    // (f64-accumulated) sum so it sums to 1.
    fn fill_q(rows: usize, cols: usize) -> Vec<f32> {
        let mut q = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let off = r * cols;
            let mut sum = 0.0f64;
            for i in 0..cols {
                // Strictly-positive raw weight (exp keeps it > 0), distinct per (r, i).
                let w = ((((r * 31 + i * 7) as f32) * 0.013 + 0.3).cos() * 0.9 + 1.4).exp();
                q[off + i] = w;
                sum += w as f64;
            }
            let inv = (1.0f64 / sum) as f32;
            for i in 0..cols {
                q[off + i] *= inv;
            }
        }
        q
    }

    // A one-hot soft target: all mass on class `t(r)` in each row (row sums to exactly 1.0).
    fn fill_onehot(rows: usize, cols: usize, t: impl Fn(usize) -> usize) -> Vec<f32> {
        let mut q = vec![0.0f32; rows * cols];
        for r in 0..rows {
            q[r * cols + t(r)] = 1.0;
        }
        q
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
                let x = fill(rows * cols);
                let q = fill_q(rows, cols);
                for row in 0..rows {
                    let off = row * cols;
                    let a =
                        unsafe { kd_loss_row_scalar(x[off..].as_ptr(), q[off..].as_ptr(), cols) };
                    let b = unsafe { kd_loss_row_avx2(x[off..].as_ptr(), q[off..].as_ptr(), cols) };
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
    /// double-precision recompute of `Σ q·(lse − x)` guards the *formula* (not just scalar==avx2).
    #[test]
    fn matches_f64_reference() {
        for &cols in &[1usize, 7, 8, 9, 17, 64, 257, 1000] {
            let rows = 4usize;
            let x = fill(rows * cols);
            let q = fill_q(rows, cols);
            let mut got = vec![0.0f32; rows];
            unsafe {
                wukong_kd_loss_f32(
                    x.as_ptr(),
                    q.as_ptr(),
                    got.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for row in 0..rows {
                let off = row * cols;
                let xd: Vec<f64> = x[off..off + cols].iter().map(|&v| v as f64).collect();
                let qd: Vec<f64> = q[off..off + cols].iter().map(|&v| v as f64).collect();
                let m = xd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let s: f64 = xd.iter().map(|&v| (v - m).exp()).sum();
                let lse = m + s.ln();
                let want: f64 = (0..cols).map(|i| qd[i] * (lse - xd[i])).sum();
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
            let x = fill(rows * cols);
            let q = fill_q(rows, cols);
            let mut s = vec![0.0f32; rows];
            let mut p = vec![0.0f32; rows];
            unsafe {
                wukong_kd_loss_f32(
                    x.as_ptr(),
                    q.as_ptr(),
                    s.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
                wukong_kd_loss_f32_parallel(
                    x.as_ptr(),
                    q.as_ptr(),
                    p.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for row in 0..rows {
                assert_eq!(
                    s[row].to_bits(),
                    p[row].to_bits(),
                    "serial != parallel rows={rows} cols={cols} row={row}"
                );
            }
        }
    }

    /// (d) consistency with the hard-label cross-entropy: a one-hot `q` (all mass on class `t`)
    /// reduces the soft loss to `lse − x[t]`, which is exactly what `wukong_xent_fwd_f32` computes
    /// for target `t`. With one-hot mass the final sum is `1·(lse − x[t]) + Σ_{i≠t} 0·(…) = lse −
    /// x[t]` — and `0.0 * anything` is `0.0` exactly in IEEE-754 for the finite logits here, so the
    /// two kernels agree to a tight tolerance (both compute the *same* `lse`; only the final fold
    /// differs in shape).
    #[test]
    fn onehot_matches_hard_label_xent() {
        for &cols in &[1usize, 7, 8, 9, 16, 17, 33, 64, 257] {
            let rows = 5usize;
            let x = fill(rows * cols);
            // Same deterministic targets xent.rs uses: target[r] = (r*7 + 3) % cols.
            let tgt = |r: usize| (r * 7 + 3) % cols;
            let q = fill_onehot(rows, cols, tgt);
            let targets: Vec<i32> = (0..rows).map(|r| tgt(r) as i32).collect();

            let mut soft = vec![0.0f32; rows];
            let mut hard = vec![0.0f32; rows];
            unsafe {
                wukong_kd_loss_f32(
                    x.as_ptr(),
                    q.as_ptr(),
                    soft.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
                wukong_xent_fwd_f32(
                    x.as_ptr(),
                    targets.as_ptr(),
                    hard.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for row in 0..rows {
                let denom = (hard[row].abs()).max(1.0);
                assert!(
                    ((soft[row] - hard[row]).abs() / denom) <= 1e-5,
                    "one-hot KD != hard xent cols={cols} row={row}: {} vs {}",
                    soft[row],
                    hard[row]
                );
            }
        }
    }

    /// (e) closed-form sanity: for a row of all-equal logits the log-softmax is `−log(C)` for every
    /// class, so `out = Σ q·log(C) = log(C)·Σq = log(C)` for any normalized `q`. Checked within f32
    /// tolerance across several `cols`.
    #[test]
    fn all_equal_logits_loss_is_log_cols() {
        for &cols in &[1usize, 2, 5, 8, 9, 16, 17, 100, 1000] {
            // Constant row (non-zero, to confirm the stabilizing subtraction): every logit == 0.7.
            let x = vec![0.7f32; cols];
            let q = fill_q(1, cols); // a normalized strictly-positive soft target
            let mut got = [0.0f32; 1];
            unsafe {
                wukong_kd_loss_f32(x.as_ptr(), q.as_ptr(), got.as_mut_ptr(), 1, cols as i64);
            }
            let want = (cols as f32).ln();
            assert!(
                (got[0] - want).abs() <= 1e-5 * want.max(1.0),
                "all-equal loss cols={cols}: {} vs log(C)={want}",
                got[0]
            );
        }
    }

    /// Edge: zero/negative rows or cols are a no-op (don't write, don't panic).
    #[test]
    fn degenerate_shapes_are_noops() {
        let x = fill(8);
        let q = fill_q(2, 4);
        let mut out = vec![42.0f32; 2];
        unsafe {
            wukong_kd_loss_f32(x.as_ptr(), q.as_ptr(), out.as_mut_ptr(), 0, 4);
            wukong_kd_loss_f32(x.as_ptr(), q.as_ptr(), out.as_mut_ptr(), 2, 0);
            wukong_kd_loss_f32_parallel(x.as_ptr(), q.as_ptr(), out.as_mut_ptr(), -1, 4);
        }
        assert!(out.iter().all(|&v| v == 42.0), "no-op must not write out");
    }
}
