//! Fused, batched **log-softmax** and **log-sum-exp** over the last axis of a row-major
//! `[rows, cols]` f32 matrix — the numerically-stable transcendental ops at the heart of every
//! classifier / language-model loss (cross-entropy is `−logsoftmax(logits)[target]`; the partition
//! function is `logsumexp`). Both are *compute-bound* on the `exp`/`log` they evaluate, and C/Rust
//! keep `expf`/`logf` **scalar** (no AVX2 transcendental in libm), so the 256-bit fused kernel — one
//! row loaded once, max + Σexp + writeback in registers, with the same hand-vectorized `exp`/`log`
//! the activation kernel uses — wins big.
//!
//! Per row (`C = cols`, `m = max_i x_i`, `s = Σ_i exp(x_i − m)`):
//!
//! ```text
//! log-softmax:  out_i = x_i − m − log(s)        // length C per row
//! log-sum-exp:  out   = m + log(s)              // ONE scalar per row
//! ```
//!
//! The `m` (row max) and `s` (Σexp) passes are **byte-identical** to `norm.rs`'s softmax /
//! log-softmax — same fixed 8-lane accumulator, same `hmax8`/`hsum8` balanced combine, the same
//! `exp8`/`exp1` from `vmath` — so these kernels are bit-for-bit consistent with the existing
//! `NORM_LOGSOFTMAX` dispatch (pinned by `matches_norm_logsoftmax_bit_for_bit`). A hand-composed
//! `exp`/`log` expression in source is *not* bit-identical — it has no 8-lane accumulator — and only
//! the f64-reference test holds these to it, to a tolerance. log-softmax then writes
//! `x_i − off` with `off = m + log(s)` (one subtract, folding the two −m, −log s into one rounding,
//! matching `norm.rs`); log-sum-exp writes the single `off`.
//!
//! **Only the log-sum-exp pair is wired into the pipeline.** `wukong_logsumexp_f32[_parallel]` has a
//! recognizer (`match_logsumexp`), an `RT_LOGSUMEXP` Cranelift symbol and an interpreter marshal arm;
//! `wukong_logsoftmax_f32[_parallel]` has **none of the three** — every recognized log-softmax is
//! emitted as `norm::wukong_norm_f32(.., NORM_LOGSOFTMAX)` instead. The log-softmax entries here are
//! exported for external C callers and held to the live `norm.rs` copy by
//! `matches_norm_logsoftmax_bit_for_bit`, which is what keeps the duplicated core from drifting.
//!
//! **The differential gate (non-negotiable).** For the log-sum-exp pair the interpreter marshals its
//! abstract memory through these exact kernels (always the *serial* entry, even for a `_parallel`
//! symbol), so:
//! - the **scalar twin equals the AVX2 path bit-for-bit** (same 8-lane reduction layout + horizontal
//!   combine in both; the only transcendental ops are `exp8`/`exp1` which already agree, and one
//!   scalar `log1` on the *same* sum bits) — pinned by a unit test across non-multiple-of-8 tails;
//! - the **`_parallel` variant equals the serial one bit-for-bit** — rows are independent, so the
//!   parallel entry just maps the identical per-row routine across rows (no cross-row combine, no
//!   thread-count-dependent chunking), so the result does not depend on core count; below
//!   `LOGSOFTMAX_PAR_MIN` rows it tail-calls the serial entry outright, same bits again.
//!
//! A non-positive `rows` or `cols` is a no-op at all four entries. Unlike `norm.rs`, these `_parallel`
//! entries fork with a bare `into_par_iter()` on the **global** pool (after `ensure_global_pool()`)
//! rather than through `run_on_wuk_pool`, so they are not part of the unified-pool routing — which is
//! a scheduling difference only, by the row-independence above.
//! (Each reduction's lane reassociation is the documented reassociated-reduction exception: every
//! backend runs this same kernel, so they agree.)

#[cfg(target_arch = "x86_64")]
use crate::vmath::exp8;
use crate::vmath::{exp1, log1};
use rayon::prelude::*;

/// Fixed-order horizontal sum of 8 lane accumulators — a balanced tree, identical in the scalar twin
/// and the AVX2 path (which stores its `__m256` to `[f32; 8]` and calls this), so both give the same
/// bits. Same shape as `norm::hsum8`.
#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

/// Fixed-order horizontal max of 8 lane accumulators (balanced tree). It folds with `f32::max`
/// (= `maxNum`, which *drops* a NaN), **not** the [`maxps`] semantics the 8-lane accumulation uses —
/// safe only because it is the same function on both paths (the AVX2 body stores its `__m256` to
/// `[f32; 8]` and calls this), so the two row maxima agree by construction on NaN rows as well as
/// finite ones; `scalar_matches_avx2_on_nan_rows` pins that. Same shape as `norm::hmax8`.
#[inline(always)]
fn hmax8(a: [f32; 8]) -> f32 {
    (a[0].max(a[1]).max(a[2].max(a[3]))).max(a[4].max(a[5]).max(a[6].max(a[7])))
}

/// `_mm256_max_ps(a, b)` semantics spelled out: `a > b ? a : b`. **Not** `f32::max` (= `maxNum`),
/// which returns the non-NaN operand — MAXPS returns its *second* source whenever the compare is
/// unordered, so a NaN in the freshly loaded operand poisons the accumulator lane while a NaN already
/// in the accumulator is dropped. The scalar row-max twin folds with this so it mirrors the AVX2 body
/// bit-for-bit on NaN rows too (pinned by `scalar_matches_avx2_on_nan_rows`). Same shape as
/// `norm::maxps` — keep the two identical.
#[inline(always)]
fn maxps(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

// --- scalar twins (the AVX2 tail + the no-AVX2 fallback) ------------------------------------------

/// `off = m + log(Σ exp(x_i − m))` for one row — the shared core of both kernels, scalar reference.
/// Computes the row max then `s = Σ exp(x − m)` in the SAME fixed 8-lane structure the AVX2 path uses
/// (so `m`, `s`, and hence `off` are bit-identical across paths), then one scalar `log1`. This is
/// exactly the `off` `norm.rs`'s `logsoftmax_row_scalar` derives.
///
/// # Safety
/// `x` must be valid for `n` `f32` elements.
#[inline]
unsafe fn lse_off_scalar(x: *const f32, n: usize) -> f32 {
    let nb = n / 8;
    let t = nb * 8;
    // 1) row max — lane j folds elements ≡ j (mod 8), tail into lanes 0.. (identical to softmax).
    let mut mx = [f32::NEG_INFINITY; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, mxj) in mx.iter_mut().enumerate() {
            *mxj = maxps(*mxj, *x.add(b + j));
        }
    }
    for (j, mxj) in mx.iter_mut().enumerate().take(n - t) {
        *mxj = mxj.max(*x.add(t + j));
    }
    let m = hmax8(mx);
    // 2) s = Σ exp(x − m) — same lane structure as softmax's sum, same `exp1` per element.
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
    m + log1(hsum8(sm)) // m + log-sum-exp; ONE scalar log — the bit-exact pivot.
}

/// Numerically-stable log-softmax of one row, scalar reference. `x`/`out` may alias.
/// `out_i = x_i − m − log(s) == x_i − off`, with `off = m + log(s)`.
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
unsafe fn logsoftmax_row_scalar(x: *const f32, out: *mut f32, n: usize) {
    let off = lse_off_scalar(x, n);
    // out_i = x_i − off. Reads `x` (not `out`), so an in-place x==out row stays correct.
    for i in 0..n {
        *out.add(i) = *x.add(i) - off;
    }
}

/// Numerically-stable log-sum-exp of one row → one scalar, scalar reference.
/// `out[0] = m + log(s) == off`.
///
/// # Safety
/// `x` valid for `n` `f32`; `out` valid for 1 `f32`.
#[inline]
unsafe fn logsumexp_row_scalar(x: *const f32, out: *mut f32, n: usize) {
    *out = lse_off_scalar(x, n);
}

// --- AVX2 kernels (mirror the scalar twins lane-for-lane on finite inputs) -------------------------

/// `off = m + log(Σ exp(x_i − m))` for one row, AVX2/FMA. The max and Σexp passes are byte-identical
/// to `norm.rs`'s `softmax_row_avx2` accumulation (8-wide `_mm256_max_ps` / `exp8` body + an `exp1`
/// scalar tail folded into the same lanes), then store to `[f32; 8]` and call the same `hmax8`/`hsum8`
/// — so `m`, `s`, and `off` match the scalar twin bit-for-bit; the only transcendental beyond `exp` is
/// one scalar `log1` on the same sum bits.
///
/// # Safety
/// `x` valid for `n` `f32`; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn lse_off_avx2(x: *const f32, n: usize) -> f32 {
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
    // 2) s = Σ exp(x − m) — exp8 lanes + exp1 tail, identical to softmax's accumulation, no out store.
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
    m + log1(hsum8(sm)) // same scalar log on the same sum bits as the scalar twin.
}

/// AVX2 log-softmax of one row. Derives `off` via [`lse_off_avx2`] (so `m`, `s`, `off` match the
/// scalar twin), then writes `out = x − off` (8-wide subtract + scalar tail).
///
/// # Safety
/// `x`/`out` valid for `n` `f32`; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn logsoftmax_row_avx2(x: *const f32, out: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    let off = lse_off_avx2(x, n);
    let offb = _mm256_set1_ps(off);
    // out = x − off (matches the scalar twin's single subtract per element).
    let mut i = 0;
    while i + 8 <= n {
        _mm256_storeu_ps(out.add(i), _mm256_sub_ps(_mm256_loadu_ps(x.add(i)), offb));
        i += 8;
    }
    while i < n {
        *out.add(i) = *x.add(i) - off;
        i += 1;
    }
}

/// AVX2 log-sum-exp of one row → one scalar. Just stores [`lse_off_avx2`]; bit-identical to the
/// scalar twin (the writeback is a single store, no SIMD).
///
/// # Safety
/// `x` valid for `n` `f32`; `out` valid for 1 `f32`; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn logsumexp_row_avx2(x: *const f32, out: *mut f32, n: usize) {
    *out = lse_off_avx2(x, n);
}

// --- per-row dispatch (AVX2 when available, else the scalar twin) ---------------------------------

/// One row of log-softmax, AVX2 when available, else the scalar twin.
///
/// # Safety
/// `x` and `out` valid for `n` `f32`; they may alias.
#[inline]
unsafe fn logsoftmax_row(x: *const f32, out: *mut f32, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return logsoftmax_row_avx2(x, out, n);
        }
    }
    logsoftmax_row_scalar(x, out, n);
}

/// One row of log-sum-exp (→ one scalar), AVX2 when available, else the scalar twin.
///
/// # Safety
/// `x` valid for `n` `f32`; `out` valid for 1 `f32`.
#[inline]
unsafe fn logsumexp_row(x: *const f32, out: *mut f32, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return logsumexp_row_avx2(x, out, n);
        }
    }
    logsumexp_row_scalar(x, out, n);
}

// --- C-ABI entry points ---------------------------------------------------------------------------

/// Row count below which the parallel variants just run serially (rayon's per-row overhead isn't worth
/// it for a handful of rows).
const LOGSOFTMAX_PAR_MIN: usize = 8;

/// Batched **log-softmax** over a `[rows, cols]` row-major matrix (serial): for each row,
/// `out[r,i] = x[r,i] − m_r − log(Σ_i exp(x[r,i] − m_r))`. `x`/`out` may alias (in-place); each row's
/// reductions read `x` before that row's `out` is written.
///
/// **Not reachable from the recognizer.** Every recognized log-softmax is emitted as
/// `wukong_norm_f32(.., NORM_LOGSOFTMAX)`; no `RT_*` constant, `GemmSyms` field or interpreter
/// marshal arm names this symbol. It is exported for external C callers, and
/// `matches_norm_logsoftmax_bit_for_bit` pins it bit-for-bit against the live `norm.rs` copy so the
/// two cannot drift apart.
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn wukong_logsoftmax_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    for row in 0..r {
        let off = row * c;
        // SAFETY: row `row` occupies [off, off+c) ⊆ [0, rows*cols).
        logsoftmax_row(x.add(off), out.add(off), c);
    }
}

/// Multicore **log-softmax** — **bit-identical** to [`wukong_logsoftmax_f32`]. Rows are independent,
/// so each is computed by the same per-row routine regardless of which thread runs it; there is no
/// cross-row combine, so the result does not depend on thread count and the interpreter's serial call
/// agrees with this `@parallel` path exactly.
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn wukong_logsoftmax_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r < LOGSOFTMAX_PAR_MIN {
        wukong_logsoftmax_f32(x, out, rows, cols);
        return;
    }
    // Configure the global pool BEFORE the first fork: this entry can be a program's first rayon
    // touch, and `ensure_global_pool`'s contract is that whoever forks first must have built the
    // 16 MiB-stack registry — otherwise rayon builds its default 2 MiB one and a later outlined
    // `@parallel` region (which carries ~1.5 MiB of per-iteration scratch) overflows its stack.
    crate::ensure_global_pool();
    // Raw pointers cross the rayon boundary as integers (same pattern as the parallel norm/reduce);
    // each row is a disjoint sub-slice.
    let (xa, oa) = (x as usize, out as usize);
    (0..r).into_par_iter().for_each(|row| {
        let off = row * c;
        // SAFETY: disjoint row; pointers re-derived from the captured addresses, valid for rows*cols.
        unsafe { logsoftmax_row((xa as *const f32).add(off), (oa as *mut f32).add(off), c) };
    });
}

/// Batched **log-sum-exp** over a `[rows, cols]` row-major matrix (serial): for each row,
/// `out[r] = m_r + log(Σ_i exp(x[r,i] − m_r))` — one scalar per row, so `out` has length `rows`.
///
/// # Safety
/// `x` must be valid for `rows * cols` `f32`; `out` must be valid for `rows` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_logsumexp_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    for row in 0..r {
        // SAFETY: row `row`'s data is [row*c, row*c+c) ⊆ [0, rows*cols); out[row] ∈ [0, rows).
        logsumexp_row(x.add(row * c), out.add(row), c);
    }
}

/// Multicore **log-sum-exp** — **bit-identical** to [`wukong_logsumexp_f32`] (rows independent, no
/// cross-row combine, so thread count is irrelevant and the interpreter's serial call agrees with this
/// `@parallel` path exactly).
///
/// # Safety
/// `x` must be valid for `rows * cols` `f32`; `out` must be valid for `rows` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_logsumexp_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r < LOGSOFTMAX_PAR_MIN {
        wukong_logsumexp_f32(x, out, rows, cols);
        return;
    }
    // Same first-touch obligation as `wukong_logsoftmax_f32_parallel`: build the 16 MiB-stack global
    // registry before forking, or rayon's default 2 MiB one wins the race for the whole process.
    crate::ensure_global_pool();
    // Pointers cross the rayon boundary as integers; each row reads a disjoint x-slice and writes one
    // disjoint out slot.
    let (xa, oa) = (x as usize, out as usize);
    (0..r).into_par_iter().for_each(|row| {
        // SAFETY: disjoint row data / out slot; pointers re-derived from the captured addresses.
        unsafe {
            logsumexp_row(
                (xa as *const f32).add(row * c),
                (oa as *mut f32).add(row),
                c,
            )
        };
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS_PAR: usize = LOGSOFTMAX_PAR_MIN;

    // Deterministic, mildly varied input (no RNG — reproducible). `sin`/`cos` of the index, scaled and
    // shifted, so values span a useful range (incl. negatives) without being trivially uniform.
    fn fill(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.017).sin() * 2.3 - 0.4 + ((i as f32) * 0.003).cos() * 1.1)
            .collect()
    }

    /// (a) scalar path == AVX2 path bit-for-bit, across cols straddling the 8-lane edge (incl.
    /// non-multiples of 8), for BOTH kernels. This is the agreement the differential gate rests on.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for &n in &[
            1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 257, 1000,
        ] {
            let x = fill(n);
            // log-softmax: full row of outputs.
            let mut a = vec![0.0f32; n];
            let mut b = vec![0.0f32; n];
            unsafe {
                logsoftmax_row_scalar(x.as_ptr(), a.as_mut_ptr(), n);
                logsoftmax_row_avx2(x.as_ptr(), b.as_mut_ptr(), n);
            }
            for i in 0..n {
                assert_eq!(
                    a[i].to_bits(),
                    b[i].to_bits(),
                    "logsoftmax scalar != avx2 n={n} i={i}: {} vs {}",
                    a[i],
                    b[i]
                );
            }
            // log-sum-exp: one scalar.
            let mut sa = [0.0f32; 1];
            let mut sb = [0.0f32; 1];
            unsafe {
                logsumexp_row_scalar(x.as_ptr(), sa.as_mut_ptr(), n);
                logsumexp_row_avx2(x.as_ptr(), sb.as_mut_ptr(), n);
            }
            assert_eq!(
                sa[0].to_bits(),
                sb[0].to_bits(),
                "logsumexp scalar != avx2 n={n}: {} vs {}",
                sa[0],
                sb[0]
            );
        }
    }

    /// (a2) the same agreement on a row whose maximum shares an 8-lane slot with a later NaN. The
    /// AVX2 body folds with `_mm256_max_ps` (= `a > b ? a : b`, so the freshly loaded NaN wins and
    /// poisons the lane); the scalar twin folds with `f32::max` (= `maxNum`, which drops the NaN and
    /// keeps the peak) — so the two pick a different `m`. `exp1`/`exp8` saturate NaN to `exp(EXP_HI)`
    /// instead of propagating it, so the divergent `m` survives into `off` rather than washing the
    /// answer out to NaN. `peak = nan_at - 8` is the only arrangement that exposes it: a NaN in any
    /// other lane poisons a lane that does not hold the row maximum and `hmax8` then drops it.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_on_nan_rows() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for &n in &[16usize, 17, 24, 33, 100] {
            for nan_at in 8..n {
                let mut x = vec![0.0f32; n];
                x[nan_at - 8] = 1.0; // the row maximum, same lane as the NaN, earlier chunk
                x[nan_at] = f32::NAN;
                let (sa, sb) =
                    unsafe { (lse_off_scalar(x.as_ptr(), n), lse_off_avx2(x.as_ptr(), n)) };
                assert_eq!(
                    sa.to_bits(),
                    sb.to_bits(),
                    "lse_off scalar != avx2 on NaN row n={n} nan_at={nan_at}: {sa} vs {sb}"
                );
            }
        }
    }

    /// (b) both kernels ≈ an independent f64 reference within a tight tolerance — guards the *formula*
    /// (not just scalar==avx2). Also (d): the log-softmax row, re-exponentiated, sums to ≈1.
    #[test]
    fn matches_f64_reference_and_normalizes() {
        for &(rows, cols) in &[(1usize, 1usize), (3, 7), (4, 64), (5, 257), (8, 512)] {
            let x = fill(rows * cols);
            let mut ls = vec![0.0f32; rows * cols];
            let mut lse = vec![0.0f32; rows];
            unsafe {
                wukong_logsoftmax_f32(x.as_ptr(), ls.as_mut_ptr(), rows as i64, cols as i64);
                wukong_logsumexp_f32(x.as_ptr(), lse.as_mut_ptr(), rows as i64, cols as i64);
            }
            for r in 0..rows {
                let row: Vec<f64> = (0..cols).map(|i| x[r * cols + i] as f64).collect();
                let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let s: f64 = row.iter().map(|&v| (v - m).exp()).sum();
                let want_lse = m + s.ln();
                // log-sum-exp scalar.
                assert!(
                    (lse[r] as f64 - want_lse).abs() < 1e-5,
                    "logsumexp r={r} cols={cols}: {} vs {want_lse}",
                    lse[r]
                );
                // log-softmax row + the sum-to-≈0-under-exp invariant.
                let mut exp_sum = 0.0f64;
                for i in 0..cols {
                    let want = (row[i] - m) - s.ln();
                    assert!(
                        (ls[r * cols + i] as f64 - want).abs() < 1e-5,
                        "logsoftmax r={r} cols={cols} i={i}: {} vs {want}",
                        ls[r * cols + i]
                    );
                    exp_sum += (ls[r * cols + i] as f64).exp();
                }
                assert!(
                    (exp_sum - 1.0).abs() < 1e-4,
                    "Σ exp(logsoftmax) != 1 for r={r} cols={cols}: {exp_sum}"
                );
                // logsumexp and logsoftmax are consistent: out_i = x_i − logsumexp_r.
                for i in 0..cols {
                    let want = x[r * cols + i] - lse[r];
                    assert!(
                        (ls[r * cols + i] - want).abs() <= 1e-5 * want.abs().max(1.0),
                        "logsoftmax != x − logsumexp r={r} i={i}"
                    );
                }
            }
        }
    }

    /// (c) serial == parallel bit-for-bit, for BOTH kernels, with rows ≥ the parallel threshold (so the
    /// rayon path actually runs). Several cols straddle the 8-lane edge.
    #[test]
    fn serial_matches_parallel_bit_for_bit() {
        for &(rows, cols) in &[
            (EPS_PAR, 1usize),
            (EPS_PAR, 7),
            (EPS_PAR, 8),
            (17, 33),
            (40, 100),
            (64, 257),
            (37, 1000),
        ] {
            let x = fill(rows * cols);
            // log-softmax.
            let mut s = vec![0.0f32; rows * cols];
            let mut p = vec![0.0f32; rows * cols];
            unsafe {
                wukong_logsoftmax_f32(x.as_ptr(), s.as_mut_ptr(), rows as i64, cols as i64);
                wukong_logsoftmax_f32_parallel(
                    x.as_ptr(),
                    p.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for i in 0..rows * cols {
                assert_eq!(
                    s[i].to_bits(),
                    p[i].to_bits(),
                    "logsoftmax serial != parallel rows={rows} cols={cols} i={i}"
                );
            }
            // log-sum-exp.
            let mut sl = vec![0.0f32; rows];
            let mut pl = vec![0.0f32; rows];
            unsafe {
                wukong_logsumexp_f32(x.as_ptr(), sl.as_mut_ptr(), rows as i64, cols as i64);
                wukong_logsumexp_f32_parallel(
                    x.as_ptr(),
                    pl.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            for r in 0..rows {
                assert_eq!(
                    sl[r].to_bits(),
                    pl[r].to_bits(),
                    "logsumexp serial != parallel rows={rows} cols={cols} r={r}"
                );
            }
        }
    }

    /// log-softmax in-place (x == out) must equal the out-of-place result (each row reads `x` before
    /// writing `out`, so aliasing is safe — the recognizer is allowed to emit in-place).
    #[test]
    fn logsoftmax_in_place_matches_out_of_place() {
        let (rows, cols) = (5usize, 130usize);
        let x = fill(rows * cols);
        let mut oop = vec![0.0f32; rows * cols];
        let mut ip = x.clone();
        unsafe {
            wukong_logsoftmax_f32(x.as_ptr(), oop.as_mut_ptr(), rows as i64, cols as i64);
            wukong_logsoftmax_f32(ip.as_ptr(), ip.as_mut_ptr(), rows as i64, cols as i64);
        }
        for i in 0..rows * cols {
            assert_eq!(
                oop[i].to_bits(),
                ip[i].to_bits(),
                "logsoftmax in-place != out-of-place i={i}"
            );
        }
    }

    /// The cross-file gate. Stable log-softmax exists TWICE: here, and in `norm.rs` as the
    /// `NORM_LOGSOFTMAX` arm of `wukong_norm_f32` — which is the copy every recognized log-softmax
    /// actually runs (`mir_build` emits `wukong_norm_f32(.., NORM_LOGSOFTMAX)`; the entries in this
    /// file are not reachable from the recognizer). Both files' doc comments assert the copies are
    /// byte-identical, and nothing pinned it: each module's own tests only compare that module's
    /// scalar/AVX2 pair against each other and against a loose f64 reference, so a change to one core
    /// alone leaves every test green.
    ///
    /// Two asserts, both bit-for-bit:
    ///  * `wukong_norm_f32(.., NORM_LOGSOFTMAX)` == `wukong_logsoftmax_f32` — the duplicated cores;
    ///  * `logsoftmax[r,i]` == `x[r,i] − logsumexp[r]` — the identity a .wk program can observe by
    ///    computing both over the same row, which go through *different* kernels
    ///    (`wukong_norm_f32` vs `wukong_logsumexp_f32`). Exact, not approximate: both derive the same
    ///    `off = m + log(s)` and both finish with one IEEE subtract.
    #[test]
    fn matches_norm_logsoftmax_bit_for_bit() {
        use crate::norm::{wukong_norm_f32, NORM_LOGSOFTMAX};
        for &(rows, cols) in &[
            (1usize, 1usize),
            (1, 3),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 16),
            (1, 17),
            (1, 31),
            (1, 64),
            (1, 100),
            (1, 257),
            (3, 33),
            (9, 512),
        ] {
            let x = fill(rows * cols);
            let mut via_norm = vec![0.0f32; rows * cols];
            let mut via_logsoftmax = vec![0.0f32; rows * cols];
            let mut lse = vec![0.0f32; rows];
            unsafe {
                wukong_norm_f32(
                    x.as_ptr(),
                    via_norm.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                    0,
                    NORM_LOGSOFTMAX,
                );
                wukong_logsoftmax_f32(
                    x.as_ptr(),
                    via_logsoftmax.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
                wukong_logsumexp_f32(x.as_ptr(), lse.as_mut_ptr(), rows as i64, cols as i64);
            }
            for (r, &lser) in lse.iter().enumerate() {
                for i in 0..cols {
                    let k = r * cols + i;
                    assert_eq!(
                        via_norm[k].to_bits(),
                        via_logsoftmax[k].to_bits(),
                        "norm::NORM_LOGSOFTMAX != logsoftmax.rs rows={rows} cols={cols} k={k}: {} vs {}",
                        via_norm[k],
                        via_logsoftmax[k]
                    );
                    assert_eq!(
                        via_norm[k].to_bits(),
                        (x[k] - lser).to_bits(),
                        "logsoftmax != x − logsumexp rows={rows} cols={cols} k={k}: {} vs {}",
                        via_norm[k],
                        x[k] - lser
                    );
                }
            }
        }
    }

    /// Edge: zero/negative rows or cols are a no-op (don't write, don't panic) — for all four entries.
    #[test]
    fn degenerate_shapes_are_noops() {
        let x = fill(8);
        let mut buf = vec![42.0f32; 8];
        unsafe {
            wukong_logsoftmax_f32(x.as_ptr(), buf.as_mut_ptr(), 0, 4);
            wukong_logsoftmax_f32(x.as_ptr(), buf.as_mut_ptr(), 2, 0);
            wukong_logsoftmax_f32_parallel(x.as_ptr(), buf.as_mut_ptr(), -1, 4);
            wukong_logsumexp_f32(x.as_ptr(), buf.as_mut_ptr(), 0, 4);
            wukong_logsumexp_f32_parallel(x.as_ptr(), buf.as_mut_ptr(), 3, -2);
        }
        assert!(buf.iter().all(|&v| v == 42.0), "no-op must not write");
    }
}
