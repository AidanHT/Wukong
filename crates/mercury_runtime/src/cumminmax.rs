//! Inclusive per-row **running max** (`cummax`) and **running min** (`cummin`) over the last axis of a
//! `[rows, cols]` row-major f32 matrix: `out[r, i] = max_{k=0..=i} x[r, k]` (and `min` for `cummin`) for
//! each row `r` and each `i` in `0..cols`. These are the cumulative-extremum primitives behind running
//! peaks/troughs, watermark tracking, and the value half of `torch.cummax` / `torch.cummin` (we return
//! only the running values, not the argmax/argmin indices).
//!
//! **Why this is a win.** A running max/min is a *loop-carried dependency* — `out[i] = fmax(out[i-1],
//! x[i])` — and gcc `-O3 -march=native` / rustc keep it **scalar** (a serial `vmaxss` chain; the
//! recurrence defeats their auto-vectorizer). This kernel breaks the chain with an in-register
//! **Hillis-Steele** scan, exactly as [`super::cumsum`] does for the prefix *sum*, but folding each block
//! with `_mm256_max_ps` / `_mm256_min_ps` instead of add: each 8-element block is scanned with three
//! shift-fold steps (a balanced reduction tree), then a running scalar `carry` (the row's running
//! extremum) is broadcast-folded into the block and updated from the block's last lane. So the per-row
//! work is SIMD-width instead of one-compare-at-a-time.
//!
//! **Bit-exactness contract — exact, not tolerance.** Unlike the prefix sum, `max`/`min` are
//! **idempotent and associative** on the values here (no rounding — the result is always one of the
//! inputs), so the in-lane balanced-tree fold gives *exactly* the same value as a strict left-to-right
//! scan. This kernel is therefore **bit-identical to its scalar twin** (`assert_eq!`, not a tolerance):
//! there is no reassociated-reduction exception. The fold uses `(a > b) ? a : b` for max and `(a < b) ?
//! a : b` for min — the exact lane semantics of `_mm256_max_ps`/`_mm256_min_ps` (which on a tie / NaN /
//! ±0 return the second operand) — so the AVX2 lanes, the scalar twin, and the cross-lane shift all
//! agree on every bit. Standard finite data has no ambiguity. And **serial == parallel bit-for-bit** —
//! rows are independent, the parallel path just maps the identical per-row routine across cores, so there
//! is no cross-row combine and the result does not depend on thread count.

use rayon::prelude::*;

/// The two extremum folds this kernel supports, selecting the identity and the scalar/AVX2 fold pair.
#[derive(Clone, Copy)]
enum Ext {
    Max,
    Min,
}

impl Ext {
    /// Fold identity: the value that leaves the other operand unchanged. `-∞` for max (any real beats
    /// it), `+∞` for min. Vacated lanes in the cross-lane shift are filled with this so the tree fold
    /// is correct, and the row `carry` is seeded to it.
    #[inline]
    fn ident(self) -> f32 {
        match self {
            Ext::Max => f32::NEG_INFINITY,
            Ext::Min => f32::INFINITY,
        }
    }

    /// Scalar fold matching `_mm256_max_ps`/`_mm256_min_ps` lane semantics: max is `(a > b) ? a : b`,
    /// min is `(a < b) ? a : b`. (On a tie/NaN this returns `b`, exactly as the AVX2 op does, so the
    /// twin agrees bit-for-bit.)
    #[inline]
    fn fold(self, a: f32, b: f32) -> f32 {
        match self {
            Ext::Max => {
                if a > b {
                    a
                } else {
                    b
                }
            }
            Ext::Min => {
                if a < b {
                    a
                } else {
                    b
                }
            }
        }
    }
}

/// Scalar twin / no-AVX2 fallback / numerical reference: strict left-to-right inclusive running extremum
/// of one row. `acc` starts at the identity (`-∞` for max, `+∞` for min) and folds `x[base..base+cols]`
/// in ascending order with [`Ext::fold`]. The AVX2 path is checked *against* this **bit-for-bit** (max/min
/// don't reassociate the value).
///
/// # Safety
/// `x` valid for `cols` f32 from `base`, `out` valid for `cols` f32 from `base`; the two ranges may be
/// distinct buffers (the kernel does not assume aliasing).
#[inline]
unsafe fn cummm_row_scalar(x: *const f32, out: *mut f32, base: usize, cols: usize, ext: Ext) {
    let mut acc = ext.ident();
    for i in 0..cols {
        acc = ext.fold(*x.add(base + i), acc);
        *out.add(base + i) = acc;
    }
}

/// In-register inclusive scan of the 8 lanes of `v` via the Hillis-Steele shift-fold pattern, returning a
/// vector whose lane `i` holds `fold_{k=0..=i} v[k]` (a balanced tree over the block, `fold` = max or min).
///
/// The three steps shift the lanes right by 1, 2, then 4 — **filling the vacated low lanes with the
/// identity** (`-∞` for max, `+∞` for min, so they don't perturb the fold) — and fold. The shift is a
/// full 8-lane cross-128-bit shift (NOT the per-128-bit `_mm256_slli_si256`), done with
/// `_mm256_permutevar8x32_ps` to gather each lane from `j - s` and `_mm256_blendv_ps` to identity-fill the
/// `s` low lanes that have no source. `permutevar8x32` is a single cross-lane shuffle, so the 128-bit
/// boundary is crossed correctly by construction; the blend masks pin the identity-fill (mirrors
/// [`super::cumsum::inclusive_scan8`], which fills with `0.0` for the additive scan).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn inclusive_scan8(v: std::arch::x86_64::__m256, ext: Ext) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // Permute indices: lane j reads source lane (j - s), saturated at 0 for j < s (those lanes are then
    // overwritten with the identity by the blend, so the saturated index value is irrelevant).
    // s = 1: [0,0,1,2,3,4,5,6]; s = 2: [0,0,0,1,2,3,4,5]; s = 4: [0,0,0,0,0,1,2,3].
    let idx1 = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    let idx2 = _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5);
    let idx4 = _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 2, 3);
    // Blend masks: high bit set on the lanes to take from the *identity* operand, i.e. the first s lanes.
    // `_mm256_blendv_ps(a, b, mask)` picks b where mask's sign bit is set.
    let id = _mm256_set1_ps(ext.ident());
    let m1 = _mm256_castsi256_ps(_mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0));
    let m2 = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0));
    let m4 = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0));

    // Pick the lane fold to match the scalar twin exactly. `_mm256_max_ps(a, b)` / `_mm256_min_ps(a, b)`
    // return `b` on a tie/NaN — the same as `(a>b)?a:b` / `(a<b)?a:b` — so AVX2 == twin bit-for-bit.
    #[inline(always)]
    unsafe fn foldv(ext: Ext, a: __m256, b: __m256) -> __m256 {
        match ext {
            Ext::Max => _mm256_max_ps(a, b),
            Ext::Min => _mm256_min_ps(a, b),
        }
    }

    // step 1: v = fold(v, shift_right_1(v))
    let s1 = _mm256_blendv_ps(_mm256_permutevar8x32_ps(v, idx1), id, m1);
    let v = foldv(ext, v, s1);
    // step 2: v = fold(v, shift_right_2(v))
    let s2 = _mm256_blendv_ps(_mm256_permutevar8x32_ps(v, idx2), id, m2);
    let v = foldv(ext, v, s2);
    // step 4: v = fold(v, shift_right_4(v))
    let s4 = _mm256_blendv_ps(_mm256_permutevar8x32_ps(v, idx4), id, m4);
    foldv(ext, v, s4)
}

/// AVX2 inclusive running extremum of one row. Processes the row in 8-element blocks: in-lane scan
/// ([`inclusive_scan8`]), fold the running `carry` into all lanes, store, then update `carry` from lane 7
/// (the block's full inclusive extremum). A scalar tail folds `cols % 8` left-to-right (`carry =
/// fold(x, carry); out = carry`) — matching `cummm_row_scalar`'s recurrence exactly for the tail, and
/// continuing the same `carry` so the row stays a single running extremum.
///
/// # Safety
/// `x`/`out` valid for `cols` f32 from `base` (distinct or aliasing); AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cummm_row_avx2(x: *const f32, out: *mut f32, base: usize, cols: usize, ext: Ext) {
    use std::arch::x86_64::*;
    let xb = x.add(base);
    let ob = out.add(base);
    let mut carry = ext.ident();
    let mut i = 0usize;
    while i + 8 <= cols {
        let v = _mm256_loadu_ps(xb.add(i));
        let scan = inclusive_scan8(v, ext);
        // fold(scan, carry) lane-wise — carry broadcast. max/min(a, set1(carry)).
        let cv = _mm256_set1_ps(carry);
        let res = match ext {
            Ext::Max => _mm256_max_ps(scan, cv),
            Ext::Min => _mm256_min_ps(scan, cv),
        };
        _mm256_storeu_ps(ob.add(i), res);
        // carry = last lane of res = running extremum through this block. Extract lane 7 by storing the
        // upper 128 and taking its top element (a plain scalar read, no horizontal op needed).
        let hi = _mm256_extractf128_ps(res, 1);
        carry = _mm_cvtss_f32(_mm_shuffle_ps(hi, hi, 0b11_11_11_11));
        i += 8;
    }
    // Scalar tail — strict left-to-right, continuing the same carry (identical to the scalar twin's
    // recurrence on these elements).
    while i < cols {
        carry = ext.fold(*xb.add(i), carry);
        *ob.add(i) = carry;
        i += 1;
    }
}

/// One row through the running extremum: AVX2 when available, else the scalar twin.
///
/// # Safety
/// `x`/`out` valid for `cols` f32 from `base` (distinct or aliasing).
#[inline]
unsafe fn cummm_row(x: *const f32, out: *mut f32, base: usize, cols: usize, ext: Ext) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            cummm_row_avx2(x, out, base, cols, ext);
            return;
        }
    }
    cummm_row_scalar(x, out, base, cols, ext);
}

/// Row count below which the parallel scan just runs serially (per-row work is light; the rayon dispatch
/// only pays off across many rows). Same threshold as the prefix-sum sibling.
const CUMMINMAX_PAR_MIN: usize = 256;

/// Serial driver over all rows for one extremum.
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` f32; they may be distinct or alias.
#[inline]
unsafe fn cummm_serial(x: *const f32, out: *mut f32, rows: usize, cols: usize, ext: Ext) {
    for r in 0..rows {
        // SAFETY: row r occupies [r*cols, r*cols + cols) ⊆ [0, rows*cols).
        unsafe { cummm_row(x, out, r * cols, cols, ext) };
    }
}

/// Parallel driver over all rows for one extremum — **bit-identical** to [`cummm_serial`]. Rows are
/// independent (each row's carry starts at the identity and never crosses rows), so the parallel path maps
/// the *identical* per-row routine across cores with no cross-row combine: the result does not depend on
/// thread count. Below `CUMMINMAX_PAR_MIN` rows it runs serial.
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` f32; non-overlapping rows (each row written by one
/// task).
#[inline]
unsafe fn cummm_parallel(x: *const f32, out: *mut f32, rows: usize, cols: usize, ext: Ext) {
    if rows < CUMMINMAX_PAR_MIN {
        cummm_serial(x, out, rows, cols, ext);
        return;
    }
    // Raw pointers cross the rayon boundary as integers (same pattern as the parallel GEMM/norm/cumsum);
    // each row is a disjoint sub-slice of out.
    let (xa, oa) = (x as usize, out as usize);
    (0..rows).into_par_iter().for_each(|r| {
        // SAFETY: disjoint row r; pointers valid for rows*cols by contract.
        unsafe { cummm_row(xa as *const f32, oa as *mut f32, r * cols, cols, ext) };
    });
}

/// `out[r, i] = max_{k=0..=i} x[r, k]` over a `[rows, cols]` row-major matrix, single-threaded. Each row
/// is an independent inclusive running max (carry resets to `-∞` at every row).
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` f32; they may be distinct or alias.
#[no_mangle]
pub unsafe extern "C" fn mercury_cummax_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    cummm_serial(x, out, rows as usize, cols as usize, Ext::Max);
}

/// `out[r, i] = min_{k=0..=i} x[r, k]` over a `[rows, cols]` row-major matrix, single-threaded. Each row
/// is an independent inclusive running min (carry resets to `+∞` at every row).
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` f32; they may be distinct or alias.
#[no_mangle]
pub unsafe extern "C" fn mercury_cummin_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    cummm_serial(x, out, rows as usize, cols as usize, Ext::Min);
}

/// Multicore `out[r, i] = max_{k=0..=i} x[r, k]` — **bit-identical** to [`mercury_cummax_f32`]. Rows are
/// independent (each row's carry starts at `-∞` and never crosses rows), so the parallel path maps the
/// *identical* per-row routine across cores with no cross-row combine; the result does not depend on
/// thread count and equals the serial kernel the interpreter marshals, bit-for-bit. Below
/// `CUMMINMAX_PAR_MIN` rows it runs serial.
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` f32; non-overlapping rows (each row written by one
/// task).
#[no_mangle]
pub unsafe extern "C" fn mercury_cummax_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    cummm_parallel(x, out, rows as usize, cols as usize, Ext::Max);
}

/// Multicore `out[r, i] = min_{k=0..=i} x[r, k]` — **bit-identical** to [`mercury_cummin_f32`] (same
/// row-independent mapping as the max sibling, identity `+∞`).
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` f32; non-overlapping rows (each row written by one
/// task).
#[no_mangle]
pub unsafe extern "C" fn mercury_cummin_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    cummm_parallel(x, out, rows as usize, cols as usize, Ext::Min);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic, non-RNG input with both signs and duplicates (the `% 101` band repeats values, so
    /// ties are exercised) so the running max/min are non-trivial.
    fn fill(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 47 + 13) % 101) as f32 - 50.0).collect()
    }

    /// Independent naive left-to-right inclusive running extremum, per row (the cross-check reference).
    /// `is_max` selects `(a>b)?a:b` (max) vs `(a<b)?a:b` (min) — the `_mm256_max_ps`/`_mm256_min_ps`
    /// tie/NaN semantics, so it agrees with the kernel bit-for-bit.
    fn naive_ref(x: &[f32], rows: usize, cols: usize, is_max: bool) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let mut acc = if is_max {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
            for i in 0..cols {
                let v = x[r * cols + i];
                acc = if is_max {
                    if v > acc {
                        v
                    } else {
                        acc
                    }
                } else if v < acc {
                    v
                } else {
                    acc
                };
                out[r * cols + i] = acc;
            }
        }
        out
    }

    /// Scalar-twin running extremum, per row (the `Ext::fold` reference / no-AVX2 fallback path).
    fn scalar_ref(x: &[f32], rows: usize, cols: usize, ext: Ext) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            unsafe { cummm_row_scalar(x.as_ptr(), out.as_mut_ptr(), r * cols, cols, ext) };
        }
        out
    }

    /// The AVX2 running max/min must equal BOTH the scalar twin AND an independent naive left-to-right
    /// reference **bit-for-bit** (max/min are idempotent + associative on the values — the in-lane tree
    /// fold is exact, no reassociation), across shapes straddling the 8-lane block edge and the tail.
    #[test]
    fn cummax_cummin_match_scalar_exact() {
        for &rows in &[1usize, 3, 64, 500] {
            for &cols in &[1usize, 7, 8, 9, 15, 16, 17, 33, 64, 100, 1000] {
                let x = fill(rows * cols);

                // cummax
                let mut got_max = vec![0.0f32; rows * cols];
                unsafe {
                    mercury_cummax_f32(x.as_ptr(), got_max.as_mut_ptr(), rows as i64, cols as i64);
                }
                let twin_max = scalar_ref(&x, rows, cols, Ext::Max);
                let naive_max = naive_ref(&x, rows, cols, true);
                assert_eq!(got_max, twin_max, "cummax vs twin {rows}x{cols}");
                assert_eq!(got_max, naive_max, "cummax vs naive {rows}x{cols}");

                // cummin
                let mut got_min = vec![0.0f32; rows * cols];
                unsafe {
                    mercury_cummin_f32(x.as_ptr(), got_min.as_mut_ptr(), rows as i64, cols as i64);
                }
                let twin_min = scalar_ref(&x, rows, cols, Ext::Min);
                let naive_min = naive_ref(&x, rows, cols, false);
                assert_eq!(got_min, twin_min, "cummin vs twin {rows}x{cols}");
                assert_eq!(got_min, naive_min, "cummin vs naive {rows}x{cols}");
            }
        }
    }

    /// Serial and parallel must be **bit-for-bit identical** (rows independent, no new reassociation — the
    /// parallel path just maps the same per-row routine across cores). Exercised above the parallel
    /// threshold so the rayon path actually runs, for both cummax and cummin.
    #[test]
    fn serial_equals_parallel() {
        let rows = CUMMINMAX_PAR_MIN + 137; // > threshold so the multicore path runs
        for &cols in &[1usize, 8, 17, 64, 333] {
            let x = fill(rows * cols);

            let mut s_max = vec![0.0f32; rows * cols];
            let mut p_max = vec![0.0f32; rows * cols];
            let mut s_min = vec![0.0f32; rows * cols];
            let mut p_min = vec![0.0f32; rows * cols];
            unsafe {
                mercury_cummax_f32(x.as_ptr(), s_max.as_mut_ptr(), rows as i64, cols as i64);
                mercury_cummax_f32_parallel(x.as_ptr(), p_max.as_mut_ptr(), rows as i64, cols as i64);
                mercury_cummin_f32(x.as_ptr(), s_min.as_mut_ptr(), rows as i64, cols as i64);
                mercury_cummin_f32_parallel(x.as_ptr(), p_min.as_mut_ptr(), rows as i64, cols as i64);
            }
            assert_eq!(s_max, p_max, "cummax serial != parallel {rows}x{cols}");
            assert_eq!(s_min, p_min, "cummin serial != parallel {rows}x{cols}");
        }
    }

    /// Focused check on the cross-128-lane in-lane scan with known data (exact, no rounding) on a single
    /// 8-block. An ascending ramp → cummax is the ramp itself and cummin is the first element repeated; a
    /// descending ramp → the opposite. This pins the `_mm256_permutevar8x32_ps` shift + identity-fill
    /// blend (the one tricky part) and that the carry seeds to the right identity.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scan_known_values() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // ascending ramp 1..=8
        let asc: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut max_asc = [0.0f32; 8];
        let mut min_asc = [0.0f32; 8];
        unsafe {
            mercury_cummax_f32(asc.as_ptr(), max_asc.as_mut_ptr(), 1, 8);
            mercury_cummin_f32(asc.as_ptr(), min_asc.as_mut_ptr(), 1, 8);
        }
        assert_eq!(max_asc, asc, "cummax of ascending ramp = the ramp");
        assert_eq!(min_asc, [1.0f32; 8], "cummin of ascending ramp = first repeated");

        // descending ramp 8..=1
        let desc: [f32; 8] = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut max_desc = [0.0f32; 8];
        let mut min_desc = [0.0f32; 8];
        unsafe {
            mercury_cummax_f32(desc.as_ptr(), max_desc.as_mut_ptr(), 1, 8);
            mercury_cummin_f32(desc.as_ptr(), min_desc.as_mut_ptr(), 1, 8);
        }
        assert_eq!(max_desc, [8.0f32; 8], "cummax of descending ramp = first repeated");
        assert_eq!(min_desc, desc, "cummin of descending ramp = the ramp");

        // a mixed pattern with distinct values + a duplicate to catch a lane-misroute the monotone
        // ramps miss (running max: 3,3,9,9,9,9,9,9 ; running min: 3,1,1,1,-4,-4,-4,-4).
        let mixed: [f32; 8] = [3.0, 1.0, 9.0, 2.0, -4.0, 7.0, 0.0, 9.0];
        let mut max_mx = [0.0f32; 8];
        let mut min_mx = [0.0f32; 8];
        unsafe {
            mercury_cummax_f32(mixed.as_ptr(), max_mx.as_mut_ptr(), 1, 8);
            mercury_cummin_f32(mixed.as_ptr(), min_mx.as_mut_ptr(), 1, 8);
        }
        assert_eq!(max_mx, [3.0, 3.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0], "cummax mixed");
        assert_eq!(min_mx, [3.0, 1.0, 1.0, 1.0, -4.0, -4.0, -4.0, -4.0], "cummin mixed");
    }
}
