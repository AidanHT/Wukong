//! Fused softmax **cross-entropy backward** (the input gradient) — the gradient that pairs with the
//! [`crate::xent`] forward loss, flowing through every classifier / language-model head in training.
//!
//! Per row `r` of a row-major `[rows, classes]` logits matrix (`C = cols`, `t = target[r] ∈ [0, C)`):
//!
//! ```text
//! m       = max_i x[r,i]
//! Z       = Σ_i exp(x[r,i] − m)
//! dx[r,i] = exp(x[r,i] − m) / Z − (i == t ? 1.0 : 0.0)     // = softmax(x)[i] − onehot(t)[i]
//! ```
//!
//! (Unscaled — the caller multiplies by `1/rows` when averaging.) This is *exactly* softmax's
//! stabilizing **row max + Σexp(x−m)** reduction followed by the normalize (multiply by `1/Z`), with a
//! single scalar `− 1.0` fixup at the target column. So on finite logits the writeback is byte-identical
//! to softmax's
//! (`norm::softmax_row_avx2`): the max pass uses the same fixed 8-lane accumulator + balanced
//! horizontal combine, the Σexp pass reuses the *exact* shared `exp` polynomials — AVX2
//! [`crate::vmath::exp8`] on the 8-lane body, the scalar [`crate::vmath::exp1`] on the tail and the
//! no-AVX2 fallback — and the normalize multiplies by the same `inv = 1/hsum8(sum)`. C/Rust keep the
//! `expf` reduction + per-element exp scalar (a loop with the libm call won't vectorize), so the fused
//! 256-bit kernel wins for the same reason the softmax / cross-entropy-forward dispatch does.
//!
//! **Bit-exactness.** Because the per-element value is the standard softmax computed the standard way,
//! `dx` (before the onehot fixup) is bit-identical to a dispatched `softmax` **on finite logits**; the
//! `− 1.0` at column `t` is a single exact scalar subtraction (a write-after-read at one index). The
//! AVX2 path and the scalar twin agree **bit-for-bit** (pinned by a unit test across non-multiple-of-8
//! `cols`, and on NaN rows by test (g)).
//!
//! **NaN divergence from `norm.rs`.** Like [`crate::xent`], this module resolves the `MAXPS`-vs-`maxNum`
//! hazard the *opposite* way from [`crate::norm`]: the scalar twin folds the row max with `f32::max`
//! (drops a NaN) and `xent_bwd_row_avx2` blends its accumulator back over the unordered lanes so
//! `_mm256_max_ps` cannot erase them. `norm::softmax_row_scalar` instead mirrors the raw NaN-absorbing
//! `maxps` chain. Both files are internally twin-consistent but pick **different** row maxima on a row
//! containing a NaN, so this kernel is *not* byte-identical to `NORM_SOFTMAX` on such a row.
//!
//! **Out-of-range labels.** `onehot(t)[i]` is `i == t`, so a `target[r]` outside `[0, C)` — negative
//! (PyTorch's `ignore_index = -100` idiom) or `>= C` — matches no column and the row is the plain
//! softmax, exactly as the formula above reads. The kernel does not write past the row for it.
//! Identical in the AVX2, scalar and parallel paths.
//!
//! **Determinism.** Rows are independent, so the `_parallel` entry just maps the identical per-row
//! routine across rows — `serial == parallel` bit-for-bit with no cross-row combine (thread count is
//! irrelevant, so the interpreter's serial call agrees with the native `@parallel` path exactly). The
//! per-row max and Σexp reassociate across lanes — the documented reassociated-reduction exception:
//! every backend runs this same kernel, so they agree.

use crate::vmath::exp1;
#[cfg(target_arch = "x86_64")]
use crate::vmath::exp8;
use rayon::prelude::*;

/// Fixed-order horizontal sum of 8 lane accumulators — a balanced tree, identical in the scalar twin
/// and the AVX2 path (which stores its `__m256` to `[f32; 8]` and calls this), so both give the same
/// bits. Same shape as `norm::hsum8` / `xent::hsum8`.
#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

/// Fixed-order horizontal max of 8 lane accumulators (balanced tree), folded with `f32::max`
/// (= `maxNum`, which drops a NaN) — the same expression `norm::hmax8` / `xent::hmax8` use, and called by
/// both of this module's paths, so the two agree by construction on every input including NaN rows
/// (pinned by test (g)). The *lane* accumulation feeding it is `f32::max` here too, unlike `norm.rs`'s
/// `maxps` fold; see the module header.
#[inline(always)]
fn hmax8(a: [f32; 8]) -> f32 {
    (a[0].max(a[1]).max(a[2].max(a[3]))).max(a[4].max(a[5]).max(a[6].max(a[7])))
}

// --- scalar twin (the AVX2 tail + the no-AVX2 fallback) -------------------------------------------

/// Cross-entropy backward for one row, scalar reference / AVX2 tail / no-AVX2 fallback. Writes
/// `dx[i] = softmax(x)[i] − onehot(target)[i]`.
///
/// The max pass (8 lanes; lane `j` folds elements `≡ j (mod 8)`, tail into lanes `0..`), the Σexp pass
/// (exp into `dx`, accumulate the sum), and the normalize (multiply by `1/Z`) are byte-identical to
/// softmax's in `norm.rs` — same `hmax8`, same `exp1`, same `hsum8`, same `1.0 / hsum8(sum)` — so the
/// pre-fixup `dx` is bit-identical to a dispatched softmax. The only new op is the single scalar
/// `dx[target] -= 1.0` (an exact subtraction at one index).
///
/// # Safety
/// `x` valid for `n` `f32`; `dx` valid for `n` `f32` (may alias `x` — each `x[i]` is read into `dx[i]`
/// before being overwritten). `target` is unconstrained: a value outside `[0, n)` matches no column,
/// so the onehot fixup is skipped.
unsafe fn xent_bwd_row_scalar(x: *const f32, target: usize, dx: *mut f32, n: usize) {
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
    // 2) dx[i] = exp(x[i] − m); accumulate Z with the same 8-lane sum as softmax (store-then-sum).
    let mut sm = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, smj) in sm.iter_mut().enumerate() {
            let e = exp1(*x.add(b + j) - m);
            *dx.add(b + j) = e;
            *smj += e;
        }
    }
    for (j, smj) in sm.iter_mut().enumerate().take(n - t) {
        let e = exp1(*x.add(t + j) - m);
        *dx.add(t + j) = e;
        *smj += e;
    }
    // 3) normalize by 1/Z — same `inv` and per-element multiply as softmax's final pass.
    let inv = 1.0 / hsum8(sm);
    for i in 0..n {
        *dx.add(i) *= inv;
    }
    // 4) subtract the onehot at the target column — a single exact scalar fixup. `onehot(t)[i]` is
    // `i == t`, so an out-of-range label matches no column: skip the fixup instead of writing past
    // the row, leaving the plain softmax the formula prescribes.
    if target < n {
        *dx.add(target) -= 1.0;
    }
}

// --- AVX2 kernel (mirrors the scalar twin lane-for-lane on finite inputs) --------------------------

/// Cross-entropy backward for one row, AVX2/FMA. The max + Σexp + normalize passes are byte-identical
/// to `softmax_row_avx2` in `norm.rs` (so the pre-fixup `dx` matches softmax bit-for-bit), capped with
/// the single scalar `dx[target] -= 1.0`.
///
/// # Safety
/// `x` valid for `n` `f32`; `dx` valid for `n` `f32` (may alias `x`); AVX2+FMA available. `target` is
/// unconstrained: a value outside `[0, n)` skips the onehot fixup, exactly as in the scalar twin.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn xent_bwd_row_avx2(x: *const f32, target: usize, dx: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    // 1) max — identical to softmax_row_avx2.
    let mut mxv = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.add(i));
        // `_mm256_max_ps(a, b)` yields `b` for an unordered pair, so a NaN logit would ERASE the
        // lane's running maximum; the scalar twin's `f32::max` ignores NaN and keeps it. Blend the
        // accumulator back over the NaN lanes so both twins fold the same values. On a row with no
        // NaN the mask is all-zero and the max — and its bits — are exactly as before.
        mxv = _mm256_blendv_ps(
            _mm256_max_ps(mxv, v),
            mxv,
            _mm256_cmp_ps::<_CMP_UNORD_Q>(v, v),
        );
        i += 8;
    }
    let mut mx = [0.0f32; 8];
    _mm256_storeu_ps(mx.as_mut_ptr(), mxv);
    for (j, mxj) in mx.iter_mut().enumerate().take(n - i) {
        *mxj = mxj.max(*x.add(i + j));
    }
    let m = hmax8(mx);
    let mb = _mm256_set1_ps(m);
    // 2) dx = exp(x − m) + sum — exp8 lanes + exp1 tail, store-then-accumulate, identical to softmax.
    let mut sv = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb));
        _mm256_storeu_ps(dx.add(i), e);
        sv = _mm256_add_ps(sv, e);
        i += 8;
    }
    let mut sm = [0.0f32; 8];
    _mm256_storeu_ps(sm.as_mut_ptr(), sv);
    for (j, smj) in sm.iter_mut().enumerate().take(n - i) {
        let e = exp1(*x.add(i + j) - m);
        *dx.add(i + j) = e;
        *smj += e;
    }
    // 3) normalize by 1/Z — same `inv` and multiply as softmax's final pass.
    let inv = 1.0 / hsum8(sm);
    let ivb = _mm256_set1_ps(inv);
    i = 0;
    while i + 8 <= n {
        _mm256_storeu_ps(dx.add(i), _mm256_mul_ps(_mm256_loadu_ps(dx.add(i)), ivb));
        i += 8;
    }
    while i < n {
        *dx.add(i) *= inv;
        i += 1;
    }
    // 4) subtract the onehot at the target column — the same single scalar fixup, and the same
    // out-of-range guard, as the scalar twin.
    if target < n {
        *dx.add(target) -= 1.0;
    }
}

/// One row through cross-entropy backward, AVX2 when available, else the scalar twin.
///
/// # Safety
/// `x` valid for `n` `f32`; `dx` valid for `n` `f32` (may alias `x`). `target` is unconstrained: a
/// value outside `[0, n)` skips the onehot fixup.
#[inline]
unsafe fn xent_bwd_row(x: *const f32, target: usize, dx: *mut f32, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return xent_bwd_row_avx2(x, target, dx, n);
        }
    }
    xent_bwd_row_scalar(x, target, dx, n);
}

/// Fused softmax cross-entropy backward `dx[r,i] = softmax(x[r,·])[i] − onehot(target[r])[i]` over a
/// `[rows, cols]` row-major logits matrix, single-threaded. `dx` is `[rows, cols]` and **may alias
/// `x`** (each row's `x` is read into `dx` before the onehot fixup); a `target[r]` outside `[0, cols)`
/// matches no column and leaves that row the plain softmax (see the module's out-of-range rule).
/// Unscaled — the caller multiplies by `1/rows` for the mean reduction.
///
/// # Safety
/// `x` valid for `rows*cols` f32; `dx` valid for `rows*cols` f32 (may alias `x`); `target` valid for
/// `rows`. The label values are unconstrained — the kernel bounds them itself.
#[no_mangle]
pub unsafe extern "C" fn wukong_xent_bwd_f32(
    x: *const f32,
    target: *const i32,
    dx: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    for row in 0..r {
        // SAFETY: row `row` occupies [row*c, row*c+c) ⊆ [0, rows*cols); the label is bounded by
        // `xent_bwd_row`'s own `target < c` check — a negative i32 sign-extends to a `usize` far
        // above `c`, so both out-of-range directions fail it — and the fixup stays inside the row.
        let t = *target.add(row) as usize;
        xent_bwd_row(x.add(row * c), t, dx.add(row * c), c);
    }
}

/// Row count below which the parallel cross-entropy backward just runs serially.
const XENT_BWD_PAR_MIN: usize = 8;

/// Multi-threaded cross-entropy backward: rows are mapped across cores, each computed by the identical
/// per-row routine — so the result is **bit-identical to [`wukong_xent_bwd_f32`]** (rows are
/// independent, no cross-row combine, so thread count is irrelevant and the interpreter's serial call
/// agrees with this `@parallel` path exactly).
///
/// # Safety
/// Operand-size contract of [`wukong_xent_bwd_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_xent_bwd_f32_parallel(
    x: *const f32,
    target: *const i32,
    dx: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r < XENT_BWD_PAR_MIN {
        wukong_xent_bwd_f32(x, target, dx, rows, cols);
        return;
    }
    // This fork can be the process's FIRST rayon touch, and `ensure_global_pool`'s contract is that
    // such a path configures the global pool before forking — otherwise rayon builds its default
    // 2 MiB-stack registry here and the later 16 MiB `build_global()` silently loses the race,
    // leaving every outlined `@parallel` region body on a stack too small for its privatized
    // scratch. Scheduling only: rows are mapped one per index regardless of worker count, so the
    // bits cannot change.
    crate::ensure_global_pool();
    // Raw pointers cross the rayon boundary as integers; each row is a disjoint sub-slice, `target`
    // indexed by row. (`dx` may alias `x`, but the per-row slices are disjoint across rows.)
    let (x_addr, t_addr, dx_addr) = (x as usize, target as usize, dx as usize);
    (0..r).into_par_iter().for_each(|row| {
        // SAFETY: disjoint row slices; pointers re-derived from the captured addresses; the label is
        // bounded by `xent_bwd_row`'s own `target < c` check, exactly as in the serial entry.
        unsafe {
            let t = *(t_addr as *const i32).add(row) as usize;
            xent_bwd_row(
                (x_addr as *const f32).add(row * c),
                t,
                (dx_addr as *mut f32).add(row * c),
                c,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic, mildly varied logits (no RNG — reproducible). Same stream shape as xent.rs's
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
    /// writeback-tail + onehot-fixup agreement that underwrites the differential gate.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        // Sizes exercising the 8-lane body, non-multiple-of-8 tails, and rows shorter than 8.
        for &cols in &[1usize, 7, 8, 9, 16, 17, 33] {
            for &rows in &[1usize, 3, 5] {
                let x = fill(rows * cols);
                let tg = fill_targets(rows, cols);
                for (row, &tgr) in tg.iter().enumerate() {
                    let off = row * cols;
                    let t = tgr as usize;
                    let mut a = vec![0.0f32; cols];
                    let mut b = vec![0.0f32; cols];
                    unsafe {
                        xent_bwd_row_scalar(x[off..].as_ptr(), t, a.as_mut_ptr(), cols);
                        xent_bwd_row_avx2(x[off..].as_ptr(), t, b.as_mut_ptr(), cols);
                    }
                    for i in 0..cols {
                        assert_eq!(
                            a[i].to_bits(),
                            b[i].to_bits(),
                            "scalar != avx2 rows={rows} cols={cols} row={row} i={i}: {} vs {}",
                            a[i],
                            b[i]
                        );
                    }
                }
            }
        }
    }

    /// (b) the kernel ≈ an independent **f64** reference within a tight tolerance (< 1e-5). An honest
    /// double-precision recompute of `softmax(x)[i] − onehot(t)[i]` guards the *formula*.
    #[test]
    fn matches_f64_reference() {
        for &cols in &[1usize, 7, 8, 9, 16, 17, 33, 64, 257, 1000] {
            let rows = 4usize;
            let x = fill(rows * cols);
            let tg = fill_targets(rows, cols);
            let mut got = vec![0.0f32; rows * cols];
            unsafe {
                wukong_xent_bwd_f32(
                    x.as_ptr(),
                    tg.as_ptr(),
                    got.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for (row, &tgr) in tg.iter().enumerate() {
                let off = row * cols;
                let xd: Vec<f64> = x[off..off + cols].iter().map(|&v| v as f64).collect();
                let m = xd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = xd.iter().map(|&v| (v - m).exp()).sum();
                let tgt = tgr as usize;
                for i in 0..cols {
                    let want = (xd[i] - m).exp() / z - if i == tgt { 1.0 } else { 0.0 };
                    let denom = want.abs().max(1.0);
                    assert!(
                        ((got[off + i] as f64 - want).abs() / denom) <= 1e-5,
                        "f64 ref rows={rows} cols={cols} row={row} i={i}: {} vs {want}",
                        got[off + i]
                    );
                }
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
            let mut s = vec![0.0f32; rows * cols];
            let mut p = vec![0.0f32; rows * cols];
            unsafe {
                wukong_xent_bwd_f32(
                    x.as_ptr(),
                    tg.as_ptr(),
                    s.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
                wukong_xent_bwd_f32_parallel(
                    x.as_ptr(),
                    tg.as_ptr(),
                    p.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for i in 0..rows * cols {
                assert_eq!(
                    s[i].to_bits(),
                    p[i].to_bits(),
                    "serial != parallel rows={rows} cols={cols} i={i}"
                );
            }
        }
    }

    /// (d) the gradient **sums to ≈ 0 per row** (Σ_i softmax = 1 and exactly one onehot subtracts 1),
    /// the defining structural property of the softmax-CE input gradient. Checked within f32
    /// tolerance across several `cols` (and exercising in-place aliasing of `dx` onto a copy of `x`).
    #[test]
    fn gradient_sums_to_zero_per_row() {
        for &cols in &[1usize, 2, 5, 8, 9, 16, 17, 100, 1000] {
            let rows = 4usize;
            let x = fill(rows * cols);
            let tg = fill_targets(rows, cols);
            // In-place: dx aliases x (a copy), the documented supported aliasing.
            let mut dx = x.clone();
            unsafe {
                wukong_xent_bwd_f32(
                    x.as_ptr(),
                    tg.as_ptr(),
                    dx.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for row in 0..rows {
                let off = row * cols;
                let sum: f64 = dx[off..off + cols].iter().map(|&v| v as f64).sum();
                assert!(
                    sum.abs() <= 1e-5,
                    "row sum != 0 cols={cols} row={row}: {sum}"
                );
            }
        }
    }

    /// (e) A label past the end of the row must not write past the end of the row. `dx` sits at the
    /// front of a padded buffer holding a recognizable sentinel, so the onehot fixup's stray `-= 1.0`
    /// is observable here instead of corrupting unrelated memory in the field. No column matches an
    /// out-of-range label, so the row is the plain softmax (`Σ dx = 1`, not 0).
    #[test]
    fn past_the_end_target_does_not_write_past_the_row() {
        const PAD: usize = 8;
        for &cols in &[1usize, 7, 8, 9, 17, 64] {
            let x = fill(cols);
            for t in [cols, cols + 3] {
                let tg = [t as i32];
                let mut dx = vec![777.0f32; cols + PAD];
                unsafe {
                    wukong_xent_bwd_f32(x.as_ptr(), tg.as_ptr(), dx.as_mut_ptr(), 1, cols as i64);
                }
                for (i, &v) in dx[cols..].iter().enumerate() {
                    assert_eq!(
                        v.to_bits(),
                        777.0f32.to_bits(),
                        "target {t} past cols={cols} wrote past the row at +{i}: {v}"
                    );
                }
                let s: f64 = dx[..cols].iter().map(|&v| v as f64).sum();
                assert!(
                    (s - 1.0).abs() <= 1e-5,
                    "cols={cols} target={t}: softmax row sum {s} != 1"
                );
            }
        }
    }

    /// (f) Every out-of-range label is total and agrees between the serial and parallel entries:
    /// negative (PyTorch's `ignore_index = -100`), `i32::MIN` and `i32::MAX` (whose `as usize` land
    /// nowhere near the row) and `== cols` all leave the row the plain softmax (`Σ dx = 1`), while
    /// the in-range rows still carry the onehot (`Σ dx = 0`). Only sound because the kernel now
    /// bounds the label itself — each of these used to be an unchecked `-= 1.0` at that offset.
    #[test]
    fn out_of_range_labels_keep_the_plain_softmax_serial_and_parallel() {
        for &cols in &[1usize, 7, 8, 9, 17, 64] {
            let rows = 16usize; // ≥ XENT_BWD_PAR_MIN, so the parallel entry really forks
            let x = fill(rows * cols);
            let mut tg = fill_targets(rows, cols);
            let bad = [-1, -100, i32::MIN, cols as i32, i32::MAX];
            tg[..bad.len()].copy_from_slice(&bad);
            let mut s = vec![0.0f32; rows * cols];
            let mut p = vec![0.0f32; rows * cols];
            unsafe {
                wukong_xent_bwd_f32(
                    x.as_ptr(),
                    tg.as_ptr(),
                    s.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
                wukong_xent_bwd_f32_parallel(
                    x.as_ptr(),
                    tg.as_ptr(),
                    p.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for i in 0..rows * cols {
                assert_eq!(
                    s[i].to_bits(),
                    p[i].to_bits(),
                    "serial != parallel cols={cols} i={i}"
                );
            }
            for (row, &tgr) in tg.iter().enumerate() {
                let off = row * cols;
                let sum: f64 = s[off..off + cols].iter().map(|&v| v as f64).sum();
                let want = if row < bad.len() { 1.0 } else { 0.0 };
                assert!(
                    (sum - want).abs() <= 1e-5,
                    "cols={cols} row={row} target={tgr}: row sum {sum} != {want}"
                );
            }
        }
    }

    /// (g) A NaN logit must not split the twins. `_mm256_max_ps(a, b)` yields `b` for an unordered
    /// pair, so a NaN lane ERASES that lane's running maximum, while the scalar twin's `f32::max`
    /// ignores NaN and keeps it — the two then stabilize with different `m`, and since `m` scales the
    /// whole row through `exp(x − m)/Z` the entire gradient row diverges on AVX2 hosts. The row here
    /// puts the maximum in lane 0 and sweeps the NaN across every column.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn nan_logit_keeps_scalar_and_avx2_in_agreement() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for &cols in &[9usize, 16, 17, 33, 64] {
            for nan_at in 0..cols {
                let mut x = fill(cols);
                x[0] = 100.0; // the row max, in lane 0 — a NaN must not erase it
                x[nan_at] = f32::NAN;
                let mut a = vec![0.0f32; cols];
                let mut b = vec![0.0f32; cols];
                unsafe {
                    xent_bwd_row_scalar(x.as_ptr(), 0, a.as_mut_ptr(), cols);
                    xent_bwd_row_avx2(x.as_ptr(), 0, b.as_mut_ptr(), cols);
                }
                for i in 0..cols {
                    assert!(
                        (a[i].is_nan() && b[i].is_nan()) || a[i].to_bits() == b[i].to_bits(),
                        "scalar != avx2 cols={cols} nan_at={nan_at} i={i}: {} vs {}",
                        a[i],
                        b[i]
                    );
                }
            }
        }
    }

    /// Edge: zero/negative rows or cols are a no-op (don't write, don't panic).
    #[test]
    fn degenerate_shapes_are_noops() {
        let x = fill(8);
        let tg = [0i32; 2];
        let mut dx = vec![42.0f32; 8];
        unsafe {
            wukong_xent_bwd_f32(x.as_ptr(), tg.as_ptr(), dx.as_mut_ptr(), 0, 4);
            wukong_xent_bwd_f32(x.as_ptr(), tg.as_ptr(), dx.as_mut_ptr(), 2, 0);
            wukong_xent_bwd_f32_parallel(x.as_ptr(), tg.as_ptr(), dx.as_mut_ptr(), -1, 4);
        }
        assert!(dx.iter().all(|&v| v == 42.0), "no-op must not write dx");
    }
}
