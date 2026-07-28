//! Fused softmax cross-entropy **forward loss** — the training loss of every classifier and language
//! model — over a row-major `[rows, classes]` f32 logits matrix with one integer target label per row.
//!
//! The numerically-stable per-row form (`C = cols`, `t = target[r] ∈ [0, C)`):
//!
//! ```text
//! m       = max_i x[r,i]
//! lse     = m + log(Σ_i exp(x[r,i] − m))     // log-sum-exp (the stable softmax denominator)
//! loss[r] = lse − x[r, t]                     // = −log(softmax(x[r,·])[t])
//! ```
//!
//! This is exactly softmax's stabilizing **row max + Σexp(x−m)** reduction (the same two passes
//! `norm.rs`'s softmax runs), capped with one `log` of the row sum and a single scalar subtraction of
//! the target logit. C/Rust keep `expf`/`logf` scalar (a loop with the libm calls won't vectorize), so
//! the 256-bit fused reduction wins for the same reason the softmax/log-softmax dispatch does.
//!
//! **Reuse for bit-exactness.** The max pass uses the same fixed 8-lane accumulator + balanced
//! horizontal combine softmax uses, and the Σexp pass reuses the *exact* shared `exp` polynomials —
//! AVX2 [`crate::vmath::exp8`] on the 8-lane body, the scalar [`crate::vmath::exp1`] on the tail and the
//! no-AVX2 fallback — with the row sum's `log` taken through the shared [`crate::vmath::log1`]. So the
//! loss is bit-identical to a softmax/log-softmax computed the standard way, and the AVX2 path and the
//! scalar twin agree **bit-for-bit** (pinned by a unit test across non-multiple-of-8 `cols`). The
//! `− x[r, target[r]]` is a single scalar gather load in both paths.
//!
//! **Out-of-range labels.** A `target[r]` outside `[0, C)` — negative (PyTorch's `ignore_index = -100`
//! idiom) or `>= C` — has no logit to gather, so the kernel writes `NaN` into `loss[r]` rather than
//! reading past the row. Same house rule as `embedding.rs`, which zeroes the output row for an
//! out-of-range id: a stated, deterministic value, identical in the AVX2, scalar and parallel paths.
//!
//! **Determinism.** Rows are independent, so the `_parallel` entry just maps the identical per-row
//! routine across rows — `serial == parallel` bit-for-bit with no cross-row combine (thread count is
//! irrelevant, so the interpreter's serial call agrees with the native `@parallel` path exactly). The
//! per-row max and Σexp reassociate across lanes — the documented reassociated-reduction exception:
//! every backend runs this same kernel, so they agree.

use crate::vmath::{exp1, log1};
#[cfg(target_arch = "x86_64")]
use crate::vmath::exp8;
use rayon::prelude::*;

/// Fixed-order horizontal sum of 8 lane accumulators — a balanced tree, identical in the scalar twin
/// and the AVX2 path (which stores its `__m256` to `[f32; 8]` and calls this), so both give the same
/// bits. Same shape as `norm::hsum8`.
#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

/// Fixed-order horizontal max of 8 lane accumulators (balanced tree). On finite inputs this matches
/// `_mm256_max_ps` lane-for-lane; the same `hmax8` softmax uses. The twin test pins the agreement on
/// finite data.
#[inline(always)]
fn hmax8(a: [f32; 8]) -> f32 {
    (a[0].max(a[1]).max(a[2].max(a[3]))).max(a[4].max(a[5]).max(a[6].max(a[7])))
}

// --- scalar twin (the AVX2 tail + the no-AVX2 fallback) -------------------------------------------

/// Cross-entropy loss of one row, scalar reference / AVX2 tail / no-AVX2 fallback. Returns
/// `lse − x[t]` where `lse = m + log(Σ exp(x − m))`, `m = max(x)`, `t = target`.
///
/// The max pass (8 lanes; lane `j` folds elements `≡ j (mod 8)`, tail into lanes `0..`) and the Σexp
/// pass are byte-identical to softmax's in `norm.rs` — same `hmax8`, same `exp1`, same `hsum8` — so `m`
/// and the sum are bit-identical to softmax's; the only new ops are one scalar `log1` on the row sum
/// and the single scalar subtraction of the target logit.
///
/// # Safety
/// `x` valid for `n` `f32`. `target` is unconstrained: a value outside `[0, n)` yields `NaN`.
unsafe fn xent_row_scalar(x: *const f32, target: usize, n: usize) -> f32 {
    let nb = n / 8;
    let t = nb * 8;
    // 1) row max — identical lane structure to softmax_row_scalar.
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
    // 2) s = Σ exp(x − m) — same lane structure as softmax's sum (no `out` store; we only need the sum).
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
    let lse = m + log1(hsum8(sm)); // log-sum-exp; ONE scalar log — same op/bits as log-softmax's `off`.
    // 3) loss = lse − x[target] (a single scalar gather load, identical in the AVX2 path). An
    // out-of-range label has no logit to gather, so the loss is NaN instead of a read past the row.
    if target < n {
        lse - *x.add(target)
    } else {
        f32::NAN
    }
}

// --- AVX2 kernel (mirrors the scalar twin lane-for-lane on finite inputs) --------------------------

/// Cross-entropy loss of one row, AVX2/FMA. The max + Σexp passes are byte-identical to
/// `softmax_row_avx2` in `norm.rs` (so `m` and the sum match bit-for-bit), capped with one scalar
/// `log1` and the single scalar subtraction of `x[target]`.
///
/// # Safety
/// `x` valid for `n` `f32`; AVX2+FMA available. `target` is unconstrained: a value outside `[0, n)`
/// yields `NaN`, exactly as in the scalar twin.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xent_row_avx2(x: *const f32, target: usize, n: usize) -> f32 {
    use std::arch::x86_64::*;
    // 1) max — identical to softmax_row_avx2.
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
    // 2) s = Σ exp(x − m) — exp8 lanes + exp1 tail, identical to softmax's accumulation (no out store).
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
    // 3) loss = lse − x[target] — the same single scalar load, and the same out-of-range guard, as
    // the scalar twin.
    if target < n {
        lse - *x.add(target)
    } else {
        f32::NAN
    }
}

/// One row through cross-entropy forward, AVX2 when available, else the scalar twin.
///
/// # Safety
/// `x` valid for `n` `f32`. `target` is unconstrained: a value outside `[0, n)` yields `NaN`.
#[inline]
unsafe fn xent_row(x: *const f32, target: usize, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return xent_row_avx2(x, target, n);
        }
    }
    xent_row_scalar(x, target, n)
}

/// Fused softmax cross-entropy forward loss `loss[r] = lse_r − x[r, target[r]]` over a `[rows, cols]`
/// row-major logits matrix, single-threaded. `loss` has length `rows`; a `target[r]` outside
/// `[0, cols)` yields `loss[r] = NaN` (see the module's out-of-range rule).
///
/// # Safety
/// `x` valid for `rows*cols` f32; `target` and `loss` valid for `rows`. The label values are
/// unconstrained — the kernel bounds them itself.
#[no_mangle]
pub unsafe extern "C" fn wukong_xent_fwd_f32(
    x: *const f32,
    target: *const i32,
    loss: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    for row in 0..r {
        // SAFETY: row `row` occupies [row*c, row*c+c) ⊆ [0, rows*cols); the label is bounded by
        // `xent_row`'s own `target < c` check — a negative i32 sign-extends to a `usize` far above
        // `c`, so both out-of-range directions fail it — and the gather stays inside the row.
        let t = *target.add(row) as usize;
        *loss.add(row) = xent_row(x.add(row * c), t, c);
    }
}

/// Row count below which the parallel cross-entropy just runs serially.
const XENT_PAR_MIN: usize = 8;

/// Multi-threaded cross-entropy forward: rows are mapped across cores, each computed by the identical
/// per-row routine — so the result is **bit-identical to [`wukong_xent_fwd_f32`]** (rows are
/// independent, no cross-row combine, so thread count is irrelevant and the interpreter's serial call
/// agrees with this `@parallel` path exactly).
///
/// # Safety
/// Operand-size contract of [`wukong_xent_fwd_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_xent_fwd_f32_parallel(
    x: *const f32,
    target: *const i32,
    loss: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r < XENT_PAR_MIN {
        wukong_xent_fwd_f32(x, target, loss, rows, cols);
        return;
    }
    // Raw pointers cross the rayon boundary as integers (the `target` i32 pointer and the `loss`
    // pointer cross as `usize` too); each row is a disjoint sub-slice, `target`/`loss` indexed by row.
    let (x_addr, t_addr, l_addr) = (x as usize, target as usize, loss as usize);
    (0..r).into_par_iter().for_each(|row| {
        // SAFETY: disjoint row slices; pointers re-derived from the captured addresses; the label is
        // bounded by `xent_row`'s own `target < c` check, exactly as in the serial entry.
        unsafe {
            let t = *(t_addr as *const i32).add(row) as usize;
            *(l_addr as *mut f32).add(row) = xent_row((x_addr as *const f32).add(row * c), t, c);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic, mildly varied logits (no RNG — reproducible). Same stream shape as norm.rs's
    // `fill` so the row reductions exercise realistic magnitudes.
    fn fill(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.017).sin() * 2.3 - 0.4)
            .collect()
    }

    // Deterministic targets in [0, cols): `target[r] = (r*7 + 3) % cols`.
    fn fill_targets(rows: usize, cols: usize) -> Vec<i32> {
        (0..rows).map(|r| ((r * 7 + 3) % cols) as i32).collect()
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
        for &cols in &[1usize, 7, 8, 9, 16, 17, 31, 33, 64, 100, 257] {
            for &rows in &[1usize, 3, 5] {
                let x = fill(rows * cols);
                let tg = fill_targets(rows, cols);
                for row in 0..rows {
                    let off = row * cols;
                    let t = tg[row] as usize;
                    let a = unsafe { xent_row_scalar(x[off..].as_ptr(), t, cols) };
                    let b = unsafe { xent_row_avx2(x[off..].as_ptr(), t, cols) };
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
    /// double-precision recompute of `lse − x[t]` guards the *formula* (not just scalar==avx2).
    #[test]
    fn matches_f64_reference() {
        for &cols in &[1usize, 7, 8, 9, 17, 64, 257, 1000] {
            let rows = 4usize;
            let x = fill(rows * cols);
            let tg = fill_targets(rows, cols);
            let mut got = vec![0.0f32; rows];
            unsafe {
                wukong_xent_fwd_f32(
                    x.as_ptr(),
                    tg.as_ptr(),
                    got.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for row in 0..rows {
                let off = row * cols;
                let xd: Vec<f64> = x[off..off + cols].iter().map(|&v| v as f64).collect();
                let m = xd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let s: f64 = xd.iter().map(|&v| (v - m).exp()).sum();
                let lse = m + s.ln();
                let want = lse - xd[tg[row] as usize];
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
            let tg = fill_targets(rows, cols);
            let mut s = vec![0.0f32; rows];
            let mut p = vec![0.0f32; rows];
            unsafe {
                wukong_xent_fwd_f32(
                    x.as_ptr(),
                    tg.as_ptr(),
                    s.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
                wukong_xent_fwd_f32_parallel(
                    x.as_ptr(),
                    tg.as_ptr(),
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

    /// (d) closed-form sanity: for a row of all-equal logits the softmax is uniform, so
    /// `loss = −log(1/C) = log(C)` regardless of the (equal) logit value or the target. Checked within
    /// f32 tolerance across several `cols`.
    #[test]
    fn all_equal_logits_loss_is_log_cols() {
        for &cols in &[1usize, 2, 5, 8, 9, 16, 17, 100, 1000] {
            // Constant row (non-zero, to confirm the stabilizing subtraction): every logit == 0.7.
            let x = vec![0.7f32; cols];
            // Sweep a few targets — the loss must not depend on which class is the target.
            for &t in &[0usize, cols / 2, cols - 1] {
                let tg = [t as i32];
                let mut got = [0.0f32; 1];
                unsafe {
                    wukong_xent_fwd_f32(x.as_ptr(), tg.as_ptr(), got.as_mut_ptr(), 1, cols as i64);
                }
                let want = (cols as f32).ln();
                assert!(
                    (got[0] - want).abs() <= 1e-5 * want.max(1.0),
                    "all-equal loss cols={cols} target={t}: {} vs log(C)={want}",
                    got[0]
                );
            }
        }
    }

    /// (e) A label past the end of the row must not gather past the end of the row. The logits sit at
    /// the front of a padded buffer whose padding holds a recognizable sentinel, so a past-the-end
    /// gather shows up as `lse − sentinel` here instead of touching unmapped memory in the field.
    #[test]
    fn past_the_end_target_does_not_gather_past_the_row() {
        const PAD: usize = 8;
        for &cols in &[1usize, 7, 8, 9, 17, 64] {
            let mut buf = vec![777.0f32; cols + PAD];
            buf[..cols].copy_from_slice(&fill(cols));
            for t in [cols, cols + 3] {
                let tg = [t as i32];
                let mut got = [0.0f32; 1];
                unsafe {
                    wukong_xent_fwd_f32(buf.as_ptr(), tg.as_ptr(), got.as_mut_ptr(), 1, cols as i64);
                }
                assert!(
                    got[0].is_nan(),
                    "target {t} past cols={cols} must not gather the row's neighbour: {}",
                    got[0]
                );
            }
        }
    }

    /// (f) Every out-of-range label is total and agrees between the serial and parallel entries:
    /// negative (PyTorch's `ignore_index = -100`), `i32::MIN` and `i32::MAX` (whose `as usize` land
    /// nowhere near the row) and `== cols` all yield `NaN`, while the in-range rows are untouched.
    /// Only sound because the kernel now bounds the label itself — each of these used to be an
    /// unchecked gather at that offset.
    #[test]
    fn out_of_range_labels_are_nan_serial_and_parallel() {
        for &cols in &[1usize, 7, 8, 9, 17, 64] {
            let rows = 16usize; // ≥ XENT_PAR_MIN, so the parallel entry really forks
            let x = fill(rows * cols);
            let mut tg = fill_targets(rows, cols);
            let bad = [-1, -100, i32::MIN, cols as i32, i32::MAX];
            tg[..bad.len()].copy_from_slice(&bad);
            let mut s = vec![0.0f32; rows];
            let mut p = vec![0.0f32; rows];
            unsafe {
                wukong_xent_fwd_f32(
                    x.as_ptr(),
                    tg.as_ptr(),
                    s.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
                wukong_xent_fwd_f32_parallel(
                    x.as_ptr(),
                    tg.as_ptr(),
                    p.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for row in 0..bad.len() {
                assert!(
                    s[row].is_nan() && p[row].is_nan(),
                    "cols={cols} target={}: serial {} / parallel {} must both be NaN",
                    tg[row],
                    s[row],
                    p[row]
                );
            }
            for row in bad.len()..rows {
                assert_eq!(
                    s[row].to_bits(),
                    p[row].to_bits(),
                    "serial != parallel cols={cols} row={row}"
                );
                assert!(s[row].is_finite(), "in-range row {row}: {}", s[row]);
            }
        }
    }

    /// Edge: zero/negative rows or cols are a no-op (don't write, don't panic).
    #[test]
    fn degenerate_shapes_are_noops() {
        let x = fill(8);
        let tg = [0i32; 2];
        let mut loss = vec![42.0f32; 2];
        unsafe {
            wukong_xent_fwd_f32(x.as_ptr(), tg.as_ptr(), loss.as_mut_ptr(), 0, 4);
            wukong_xent_fwd_f32(x.as_ptr(), tg.as_ptr(), loss.as_mut_ptr(), 2, 0);
            wukong_xent_fwd_f32_parallel(x.as_ptr(), tg.as_ptr(), loss.as_mut_ptr(), -1, 4);
        }
        assert!(loss.iter().all(|&v| v == 42.0), "no-op must not write loss");
    }
}
