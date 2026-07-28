//! Inclusive per-row **prefix sum** (scan) over the last axis of a `[rows, cols]` row-major f32
//! matrix: `out[r, i] = Σ_{k=0..=i} x[r, k]` for each row `r` and each `i` in `0..cols`. This is the
//! cumulative-sum primitive behind running totals, prefix-sum-based sampling (e.g. the CDF a
//! top-p / nucleus sampler bisects), segment offsets, and integral images along one axis.
//!
//! **Why this is a win.** A prefix sum is a *loop-carried dependency* — `out[i] = out[i-1] + x[i]` —
//! and gcc `-O3 -march=native` / rustc keep it **scalar** (a serial `vaddss` chain; the recurrence
//! defeats their auto-vectorizer). This kernel breaks the chain with an in-register **Hillis-Steele**
//! scan: each 8-element block is scanned with three shift-add steps (a balanced reduction tree), then a
//! running scalar `carry` (the row's running total) is broadcast-added to the block and updated from
//! the block's last lane. So the per-row work is SIMD-width instead of one-add-at-a-time.
//!
//! **Bit-exactness contract.** The AVX2 in-lane scan sums each block as a **balanced tree**
//! (`v += v>>1; v += v>>2; v += v>>4`), whereas the scalar twin sums **left-to-right**. Float add is
//! not associative, so the AVX2 path is *not* bit-identical to a strict left-to-right prefix sum — it is
//! the project's documented **reassociated-reduction** case: the AVX2 kernel is the oracle that BOTH
//! backends run (the interpreter marshals its memory through this exact function), so the relevant
//! correctness checks are (1) AVX2 ≈ scalar/`f64` prefix sum within a tight *relative* tolerance (it is
//! numerically the prefix sum, not a permutation bug) and (2) **serial == parallel bit-for-bit** — rows
//! are independent, the parallel path just maps the identical per-row routine across cores, so there is
//! *no new reassociation* between serial and parallel; that one IS exact (`assert_eq!` on the bits).

use rayon::prelude::*;

/// Scalar twin / no-AVX2 fallback / numerical reference: strict left-to-right inclusive prefix sum of
/// one row. `acc` starts at `0.0` and folds `x[base..base+cols]` in ascending order. The
/// reassociated AVX2 path is checked *against* this within a relative tolerance (not bit-for-bit).
///
/// # Safety
/// `x` valid for `cols` f32 from `base`, `out` valid for `cols` f32 from `base`; the two ranges may be
/// distinct buffers (the kernel does not assume aliasing).
#[inline]
unsafe fn cumsum_row_scalar(x: *const f32, out: *mut f32, base: usize, cols: usize) {
    let mut acc = 0.0f32;
    for i in 0..cols {
        acc += *x.add(base + i);
        *out.add(base + i) = acc;
    }
}

/// In-register inclusive scan of the 8 lanes of `v` via the Hillis-Steele shift-add pattern, returning a
/// vector whose lane `i` holds `Σ_{k=0..=i} v[k]` (a balanced tree over the block).
///
/// The three steps shift the lanes right by 1, 2, then 4 — **filling the vacated low lanes with 0.0** —
/// and add. The shift is a full 8-lane cross-128-bit shift (NOT the per-128-bit `_mm256_slli_si256`),
/// done with `_mm256_permutevar8x32_ps` to gather each lane from `j - s` and `_mm256_blendv_ps` to zero
/// the `s` low lanes that have no source. `permutevar8x32` is a single cross-lane shuffle, so the
/// 128-bit boundary is crossed correctly by construction; the blend masks pin the zero-fill (the
/// `scalar_vs_avx2_lane_shift` test verifies the exact `[1,2,3,4,5,6,7,8]`-style pattern on integers).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn inclusive_scan8(v: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // Permute indices: lane j reads source lane (j - s), saturated at 0 for j < s (those lanes are then
    // overwritten with 0.0 by the blend, so the saturated index value is irrelevant).
    // s = 1: [0,0,1,2,3,4,5,6]; s = 2: [0,0,0,1,2,3,4,5]; s = 4: [0,0,0,0,0,1,2,3].
    let idx1 = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    let idx2 = _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5);
    let idx4 = _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 2, 3);
    // Blend masks: high bit set on the lanes to take from the *shifted* (zero) operand, i.e. the first s
    // lanes. `_mm256_blendv_ps(a, b, mask)` picks b where mask's sign bit is set.
    let zero = _mm256_setzero_ps();
    let m1 = _mm256_castsi256_ps(_mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0));
    let m2 = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0));
    let m4 = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0));

    // step 1: v += shift_right_1(v)
    let s1 = _mm256_blendv_ps(_mm256_permutevar8x32_ps(v, idx1), zero, m1);
    let v = _mm256_add_ps(v, s1);
    // step 2: v += shift_right_2(v)
    let s2 = _mm256_blendv_ps(_mm256_permutevar8x32_ps(v, idx2), zero, m2);
    let v = _mm256_add_ps(v, s2);
    // step 4: v += shift_right_4(v)
    let s4 = _mm256_blendv_ps(_mm256_permutevar8x32_ps(v, idx4), zero, m4);
    _mm256_add_ps(v, s4)
}

/// AVX2 inclusive prefix sum of one row. Processes the row in 8-element blocks: in-lane scan
/// ([`inclusive_scan8`]), add the running `carry` to all lanes, store, then update `carry` from lane 7
/// (the block's full inclusive sum). A scalar tail folds `cols % 8` left-to-right (`carry += x; out =
/// carry`) — matching `cumsum_row_scalar`'s recurrence exactly for the tail, and continuing the same
/// `carry` so the row stays a single prefix sum.
///
/// # Safety
/// `x`/`out` valid for `cols` f32 from `base` (distinct or aliasing); AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cumsum_row_avx2(x: *const f32, out: *mut f32, base: usize, cols: usize) {
    use std::arch::x86_64::*;
    let xb = x.add(base);
    let ob = out.add(base);
    let mut carry = 0.0f32;
    let mut i = 0usize;
    while i + 8 <= cols {
        let v = _mm256_loadu_ps(xb.add(i));
        let scan = inclusive_scan8(v);
        let res = _mm256_add_ps(scan, _mm256_set1_ps(carry));
        _mm256_storeu_ps(ob.add(i), res);
        // carry = last lane of res = running total through this block. Extract lane 7 by storing the
        // upper 128 and taking its top element (a plain scalar read, no horizontal op needed).
        let hi = _mm256_extractf128_ps(res, 1);
        carry = _mm_cvtss_f32(_mm_shuffle_ps(hi, hi, 0b11_11_11_11));
        i += 8;
    }
    // Scalar tail — strict left-to-right, continuing the same carry (identical to the scalar twin's
    // recurrence on these elements).
    while i < cols {
        carry += *xb.add(i);
        *ob.add(i) = carry;
        i += 1;
    }
}

/// One row through the prefix sum: AVX2 when available, else the scalar twin.
///
/// # Safety
/// `x`/`out` valid for `cols` f32 from `base` (distinct or aliasing).
#[inline]
unsafe fn cumsum_row(x: *const f32, out: *mut f32, base: usize, cols: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            cumsum_row_avx2(x, out, base, cols);
            return;
        }
    }
    cumsum_row_scalar(x, out, base, cols);
}

/// Row count below which the parallel scan just runs serially (per-row work is light; the rayon
/// dispatch only pays off across many rows).
const CUMSUM_PAR_MIN: usize = 256;

/// `out[r, i] = Σ_{k=0..=i} x[r, k]` over a `[rows, cols]` row-major matrix, single-threaded. Each row
/// is an independent inclusive prefix sum (carry resets to 0 at every row).
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` f32; they may be distinct or alias.
#[no_mangle]
pub unsafe extern "C" fn wukong_cumsum_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    for r in 0..rows {
        // SAFETY: row r occupies [r*cols, r*cols + cols) ⊆ [0, rows*cols).
        unsafe { cumsum_row(x, out, r * cols, cols) };
    }
}

/// Multicore `out[r, i] = Σ_{k=0..=i} x[r, k]` — **bit-identical** to [`wukong_cumsum_f32`]. Rows are
/// independent (each row's carry starts at 0 and never crosses rows), so the parallel path maps the
/// *identical* per-row routine across cores with no cross-row combine: the result does not depend on
/// thread count and equals the serial kernel the interpreter marshals, bit-for-bit. Below
/// `CUMSUM_PAR_MIN` rows it runs serial.
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` f32; non-overlapping rows (each row written by one
/// task).
#[no_mangle]
pub unsafe extern "C" fn wukong_cumsum_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    if rows < CUMSUM_PAR_MIN {
        for r in 0..rows {
            unsafe { cumsum_row(x, out, r * cols, cols) };
        }
        return;
    }
    // This fork can be the process's FIRST rayon touch, and rayon builds its global registry lazily
    // there: without this, the DEFAULT registry (std-sized worker stacks) is what gets built, and the
    // runtime's own 16 MiB `build_global` then loses the race for the rest of the process — its `Err`
    // is discarded, so outlined `@parallel` region bodies end up on undersized stacks. See
    // [`crate::ensure_global_pool`], whose doc states this as a precondition on every parallel path.
    // Idempotent (`Once`) and scheduling-only: the row remains the unit of work, so the bits are
    // unchanged and serial == parallel still holds bit-for-bit.
    crate::ensure_global_pool();
    // Raw pointers cross the rayon boundary as integers (same pattern as the parallel GEMM/norm); each
    // row is a disjoint sub-slice of out.
    let (xa, oa) = (x as usize, out as usize);
    (0..rows).into_par_iter().for_each(|r| {
        // SAFETY: disjoint row r; pointers valid for rows*cols by contract.
        unsafe { cumsum_row(xa as *const f32, oa as *mut f32, r * cols, cols) };
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic, non-RNG input with a spread of signs/magnitudes so the prefix sums are
    /// non-trivial (and partial cancellation exercises the reassociation).
    fn fill(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 31 + 7) % 101) as f32 * 0.25 - 12.0)
            .collect()
    }

    /// f64 left-to-right inclusive prefix sum, per row — the independent numerical reference.
    fn cumsum_f64_ref(x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; rows * cols];
        for r in 0..rows {
            let mut acc = 0.0f64;
            for i in 0..cols {
                acc += x[r * cols + i] as f64;
                out[r * cols + i] = acc;
            }
        }
        out
    }

    /// f32 scalar-twin prefix sum, per row (the strict left-to-right reference / fallback).
    fn cumsum_scalar_ref(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            unsafe { cumsum_row_scalar(x.as_ptr(), out.as_mut_ptr(), r * cols, cols) };
        }
        out
    }

    /// Relative tolerance the reassociated AVX2 scan is held to vs the (left-to-right) scalar/f64
    /// reference: `|a - b| <= REL_TOL * max(1, |b|)`.
    const REL_TOL: f32 = 1e-4;

    #[inline]
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= REL_TOL * b.abs().max(1.0)
    }

    /// The AVX2 prefix sum must match the scalar twin AND an independent f64 left-to-right prefix sum
    /// within a tight relative tolerance, across shapes straddling the 8-lane block edge and the tail.
    /// (Tolerance, not bit-equality: the in-lane tree scan reassociates the float sum — the documented
    /// reassociated-reduction exception.)
    #[test]
    fn cumsum_matches_scalar_within_tol() {
        for &rows in &[1usize, 3, 64, 500] {
            for &cols in &[1usize, 7, 8, 9, 15, 16, 17, 33, 64, 100, 1000] {
                let x = fill(rows * cols);
                let mut got = vec![0.0f32; rows * cols];
                unsafe {
                    wukong_cumsum_f32(x.as_ptr(), got.as_mut_ptr(), rows as i64, cols as i64);
                }
                let want_f32 = cumsum_scalar_ref(&x, rows, cols);
                let want_f64 = cumsum_f64_ref(&x, rows, cols);
                for t in 0..rows * cols {
                    assert!(
                        close(got[t], want_f32[t]),
                        "vs scalar {rows}x{cols} t={t}: {} vs {}",
                        got[t],
                        want_f32[t]
                    );
                    assert!(
                        (got[t] as f64 - want_f64[t]).abs()
                            <= REL_TOL as f64 * want_f64[t].abs().max(1.0),
                        "vs f64 {rows}x{cols} t={t}: {} vs {}",
                        got[t],
                        want_f64[t]
                    );
                }
            }
        }
    }

    /// NaN must reach the output at exactly the positions the strict left-to-right twin puts it. The
    /// additive tree gives lane `j` the sum of `v[0..=j]` with every element counted exactly once — only
    /// the *order* differs — so a NaN at index `p` poisons `out[p..]` on both paths and nothing before
    /// it. This is precisely what makes the reassociated scan safe where the sibling running-max/min
    /// scan needed an explicit guard: `+` propagates NaN under any association, `(a > b) ? a : b` does
    /// not. Positions before `p` are an ordinary reassociated prefix sum and are held to `REL_TOL`.
    #[test]
    fn cumsum_nan_propagates_like_the_scalar_twin() {
        for &cols in &[8usize, 9, 17, 33, 64] {
            for p in 0..cols {
                let mut x = fill(cols);
                x[p] = f32::NAN;
                let mut got = vec![0.0f32; cols];
                unsafe {
                    wukong_cumsum_f32(x.as_ptr(), got.as_mut_ptr(), 1, cols as i64);
                }
                let want = cumsum_scalar_ref(&x, 1, cols);
                for t in 0..cols {
                    assert_eq!(
                        got[t].is_nan(),
                        want[t].is_nan(),
                        "NaN reach {cols} p={p} t={t}: {} vs {}",
                        got[t],
                        want[t]
                    );
                    if t < p {
                        assert!(
                            close(got[t], want[t]),
                            "prefix {cols} p={p} t={t}: {} vs {}",
                            got[t],
                            want[t]
                        );
                    }
                }
            }
        }
    }

    /// Serial and parallel must be **bit-for-bit identical** (rows independent, no new reassociation —
    /// the parallel path just maps the same per-row routine across cores). Exercised above the parallel
    /// threshold so the rayon path actually runs.
    #[test]
    fn serial_equals_parallel_bit_for_bit() {
        let rows = CUMSUM_PAR_MIN + 137; // > threshold so the multicore path runs
        for &cols in &[1usize, 8, 17, 64, 333] {
            let x = fill(rows * cols);
            let mut s = vec![0.0f32; rows * cols];
            let mut p = vec![0.0f32; rows * cols];
            unsafe {
                wukong_cumsum_f32(x.as_ptr(), s.as_mut_ptr(), rows as i64, cols as i64);
                wukong_cumsum_f32_parallel(x.as_ptr(), p.as_mut_ptr(), rows as i64, cols as i64);
            }
            assert_eq!(s, p, "serial != parallel {rows}x{cols}");
        }
    }

    /// Focused check on the cross-128-lane in-lane scan with integer-valued f32 (exact, no rounding):
    /// all-ones → `[1,2,3,4,5,6,7,8]`, and an ascending ramp `[1..=8]` → its prefix sums. This pins the
    /// `_mm256_permutevar8x32_ps` shift + zero-fill blend (the one tricky part) on a single block.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_vs_avx2_lane_shift() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use std::arch::x86_64::*;
        unsafe {
            // all ones -> running count 1..=8
            let ones = _mm256_set1_ps(1.0);
            let mut got = [0.0f32; 8];
            _mm256_storeu_ps(got.as_mut_ptr(), inclusive_scan8(ones));
            let want = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            assert_eq!(got, want, "scan of all-ones");

            // ascending ramp 1..=8 -> prefix sums 1,3,6,10,15,21,28,36
            let ramp = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            let mut got2 = [0.0f32; 8];
            _mm256_storeu_ps(got2.as_mut_ptr(), inclusive_scan8(ramp));
            let want2 = [1.0f32, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0];
            assert_eq!(got2, want2, "scan of ascending ramp");

            // a third pattern with distinct values to catch a lane-misroute the symmetric ones miss.
            let mixed = _mm256_setr_ps(10.0, -3.0, 2.0, 0.0, 5.0, -1.0, 4.0, 8.0);
            let mut got3 = [0.0f32; 8];
            _mm256_storeu_ps(got3.as_mut_ptr(), inclusive_scan8(mixed));
            // prefix: 10,7,9,9,14,13,17,25
            let want3 = [10.0f32, 7.0, 9.0, 9.0, 14.0, 13.0, 17.0, 25.0];
            assert_eq!(got3, want3, "scan of mixed values");
        }
    }
}
