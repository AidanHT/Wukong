//! Selection and sampling kernels for decoding: `topk_f32`, `argsort_f32` / `argsort_desc_f32`,
//! and `sample_categorical` — the pieces temperature / top-k / top-p sampling is written from.
//!
//! **One total order.** Every result here is defined by a *strict total* order on indices, so it is
//! unique — independent of the selection algorithm, the thread count and the backend:
//! * descending (`topk_f32`, `argsort_desc_f32`): larger value first, **NaN largest of all**, ties
//!   (including `-0.0` against `+0.0`, and NaN against NaN) to the **lower index**;
//! * ascending (`argsort_f32`): smaller value first, **NaN last**, ties to the lower index — a
//!   *stable* sort.
//!
//! These are `torch.topk` / `torch.sort(stable=True)` orders. The interpreter marshals its slots
//! into host buffers and calls the very same safe functions the `extern "C"` entries wrap, so the
//! two backends agree bit for bit by construction.
//!
//! **Aliasing.** The entries read the whole input before writing any output, so an output buffer
//! that overlaps the input (`topk_f32(x, x, idx)`) sees the result of selecting from the original
//! `x` — the same answer the interpreter's copy-in/copy-out gives.

use std::cmp::Ordering;

/// The descending order: NaN first, then larger values, ties to the lower index.
fn desc(x: &[f32], a: usize, b: usize) -> Ordering {
    let (xa, xb) = (x[a], x[b]);
    let by_value = match (xa.is_nan(), xb.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        // Neither is NaN, so `partial_cmp` is total; `-0.0 == +0.0` is a tie.
        (false, false) => xb.partial_cmp(&xa).unwrap_or(Ordering::Equal),
    };
    by_value.then(a.cmp(&b))
}

/// The ascending order: smaller values first, NaN last, ties to the lower index.
fn asc(x: &[f32], a: usize, b: usize) -> Ordering {
    let (xa, xb) = (x[a], x[b]);
    let by_value = match (xa.is_nan(), xb.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => xa.partial_cmp(&xb).unwrap_or(Ordering::Equal),
    };
    by_value.then(a.cmp(&b))
}

/// The first `m` indices of `x` under `order`, in order: a linear-time selection of the `m`
/// smallest, then a sort of just those (`O(n + m log m)`). `m` must be `<= x.len()`.
fn ranked_prefix(x: &[f32], m: usize, order: fn(&[f32], usize, usize) -> Ordering) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    if m == 0 {
        return Vec::new();
    }
    if m < idx.len() {
        idx.select_nth_unstable_by(m - 1, |&a, &b| order(x, a, b));
        idx.truncate(m);
    }
    // The order is total, so an unstable sort is deterministic.
    idx.sort_unstable_by(|&a, &b| order(x, a, b));
    idx
}

/// The `k = min(x.len(), vals.len(), idx.len())` largest elements of `x` in descending order:
/// `vals[j]` is the value and `idx[j]` its index. Elements of `vals`/`idx` past `k` are untouched.
/// Returns `k`.
pub fn topk(x: &[f32], vals: &mut [f32], idx: &mut [i64]) -> usize {
    let k = x.len().min(vals.len()).min(idx.len());
    let order = ranked_prefix(x, k, desc);
    for (j, &i) in order.iter().enumerate() {
        vals[j] = x[i];
        idx[j] = i as i64;
    }
    k
}

/// Write into `idx` the first `m = min(x.len(), idx.len())` positions of the stable sort of `x`
/// (descending when `descending`). With `idx.len() >= x.len()` that is the full argsort; with a
/// shorter `idx`, its prefix. Returns `m`.
pub fn argsort(x: &[f32], idx: &mut [i64], descending: bool) -> usize {
    let m = x.len().min(idx.len());
    let order = ranked_prefix(x, m, if descending { desc } else { asc });
    for (j, &i) in order.iter().enumerate() {
        idx[j] = i as i64;
    }
    m
}

/// Draw one index from the categorical distribution with unnormalized weights `p`, by inverse CDF
/// at the uniform variate `u`.
///
/// A weight's mass is the weight itself when it is positive and finite, and zero otherwise (a
/// negative, zero, NaN or infinite weight can never be drawn). With `total` the sum of the masses
/// in index order (in `f64`), the result is the first index whose running mass exceeds `u · total`
/// — so `u = 0` draws the first index with mass, and `u = 1` (or a `u` that rounding leaves past
/// the last running sum) the last. `u` is clamped to `[0, 1]`, NaN counting as 0. Returns `-1` when
/// no weight has mass (including an empty `p`). Feed it `rand_f32(seed, step)` and the draw is a
/// pure function of `(probs, seed, step)`.
pub fn sample_categorical(p: &[f32], u: f32) -> i64 {
    let mass = |w: f32| -> f64 {
        if w > 0.0 && w.is_finite() {
            f64::from(w)
        } else {
            0.0
        }
    };
    let mut total = 0.0f64;
    for &w in p {
        total += mass(w);
    }
    if total <= 0.0 {
        return -1;
    }
    let u = if u.is_nan() { 0.0 } else { u.clamp(0.0, 1.0) };
    let target = f64::from(u) * total;
    let mut acc = 0.0f64;
    let mut last = -1i64;
    for (i, &w) in p.iter().enumerate() {
        let m = mass(w);
        if m > 0.0 {
            acc += m;
            last = i as i64;
            if acc > target {
                return last;
            }
        }
    }
    last
}

/// An `f32` buffer argument, or the empty slice for a non-positive length (a zero-length
/// `alloc_*` slice carries a null data pointer, which `from_raw_parts` must never see).
///
/// # Safety
/// For `n > 0`, `p` must address `n` readable elements.
unsafe fn input<'a>(p: *const f32, n: i64) -> &'a [f32] {
    if n <= 0 || p.is_null() {
        return &[];
    }
    // SAFETY: forwarded from this function's contract.
    unsafe { std::slice::from_raw_parts(p, n as usize) }
}

/// The mutable twin of [`input`].
///
/// # Safety
/// For `n > 0`, `p` must address `n` writable elements, aliased by no live reference.
unsafe fn output<'a, T>(p: *mut T, n: i64) -> &'a mut [T] {
    if n <= 0 || p.is_null() {
        return &mut [];
    }
    // SAFETY: forwarded from this function's contract.
    unsafe { std::slice::from_raw_parts_mut(p, n as usize) }
}

/// `wukong_topk_f32(x, n, vals, nv, idx, ni) -> i64` — see [`topk`]. The input is copied before
/// either output is written, so an output may overlap it.
///
/// # Safety
/// Each pointer must address its count of elements (`x`: `n` readable `f32`; `vals`: `nv` writable
/// `f32`; `idx`: `ni` writable `i64`); `vals` and `idx` must not overlap each other.
#[no_mangle]
pub unsafe extern "C" fn wukong_topk_f32(
    x: *const f32,
    n: i64,
    vals: *mut f32,
    nv: i64,
    idx: *mut i64,
    ni: i64,
) -> i64 {
    // SAFETY: forwarded from this function's contract; the copy ends the input borrow before the
    // outputs are created, so an overlapping output never coexists with it.
    let xs = unsafe { input(x, n) }.to_vec();
    let vals = unsafe { output(vals, nv) };
    let idx = unsafe { output(idx, ni) };
    topk(&xs, vals, idx) as i64
}

/// `wukong_argsort_f32(x, n, idx, ni, descending) -> i64` — see [`argsort`]; `descending != 0`
/// selects the descending order.
///
/// # Safety
/// `x` must address `n` readable `f32`s and `idx` `ni` writable `i64`s.
#[no_mangle]
pub unsafe extern "C" fn wukong_argsort_f32(
    x: *const f32,
    n: i64,
    idx: *mut i64,
    ni: i64,
    descending: i64,
) -> i64 {
    // SAFETY: forwarded from this function's contract (an `i64` output cannot overlap an `f32`
    // input in a well-typed program, but the copy keeps the entry sound even if it did).
    let xs = unsafe { input(x, n) }.to_vec();
    let idx = unsafe { output(idx, ni) };
    argsort(&xs, idx, descending != 0) as i64
}

/// `wukong_sample_categorical_f32(p, n, u) -> i64` — see [`sample_categorical`].
///
/// # Safety
/// `p` must address `n` readable `f32`s.
#[no_mangle]
pub unsafe extern "C" fn wukong_sample_categorical_f32(p: *const f32, n: i64, u: f32) -> i64 {
    // SAFETY: forwarded from this function's contract.
    sample_categorical(unsafe { input(p, n) }, u)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A brute-force reference: sort every index by the order and take the prefix.
    fn reference(x: &[f32], m: usize, order: fn(&[f32], usize, usize) -> Ordering) -> Vec<usize> {
        let mut all: Vec<usize> = (0..x.len()).collect();
        all.sort_by(|&a, &b| order(x, a, b));
        all.truncate(m);
        all
    }

    #[test]
    fn topk_orders_by_value_nan_first_ties_to_lower_index() {
        let x = [1.0, f32::NAN, 3.0, -0.0, 3.0, 0.0, f32::NAN, -5.0];
        let mut vals = [9.0f32; 5];
        let mut idx = [9i64; 5];
        assert_eq!(topk(&x, &mut vals, &mut idx), 5);
        assert_eq!(idx, [1, 6, 2, 4, 0]);
        assert!(vals[0].is_nan() && vals[1].is_nan());
        assert_eq!(&vals[2..], &[3.0, 3.0, 1.0]);
        // k is bounded by the shortest buffer; the rest stays untouched.
        let mut v2 = [9.0f32; 3];
        let mut i2 = [9i64; 2];
        assert_eq!(topk(&x[..4], &mut v2, &mut i2), 2);
        assert_eq!((i2, v2[2]), ([1, 2], 9.0));
        // -0.0 and +0.0 tie; the lower index wins.
        let z = [0.0, -0.0, 0.0];
        let mut zv = [1.0f32; 3];
        let mut zi = [7i64; 3];
        topk(&z, &mut zv, &mut zi);
        assert_eq!(zi, [0, 1, 2]);
        assert_eq!(zv[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(topk(&[], &mut zv, &mut zi), 0);
    }

    #[test]
    fn argsort_is_stable_both_ways_with_nan_last_ascending() {
        let x = [2.0, f32::NAN, -1.0, 2.0, 0.5, -1.0];
        let mut idx = [0i64; 6];
        assert_eq!(argsort(&x, &mut idx, false), 6);
        assert_eq!(idx, [2, 5, 4, 0, 3, 1]);
        assert_eq!(argsort(&x, &mut idx, true), 6);
        assert_eq!(idx, [1, 0, 3, 4, 2, 5]);
        // A short index buffer receives the prefix of the full order.
        let mut short = [9i64; 2];
        assert_eq!(argsort(&x, &mut short, false), 2);
        assert_eq!(short, [2, 5]);
        let mut long = [9i64; 8];
        assert_eq!(argsort(&x[..3], &mut long, false), 3);
        assert_eq!(long, [2, 0, 1, 9, 9, 9, 9, 9]);
    }

    #[test]
    fn selection_matches_a_full_sort_on_awkward_inputs() {
        // Many duplicates, signed zeros and NaNs, every prefix length.
        let x: Vec<f32> = (0..97u32)
            .map(|i| match i % 7 {
                0 => f32::NAN,
                1 => -0.0,
                2 => 0.0,
                _ => ((i * 37) % 11) as f32 - 5.0,
            })
            .collect();
        for m in 0..=x.len() {
            for order in [desc as fn(&[f32], usize, usize) -> Ordering, asc] {
                assert_eq!(
                    ranked_prefix(&x, m, order),
                    reference(&x, m, order),
                    "m={m}"
                );
            }
        }
    }

    #[test]
    fn categorical_sampling_is_inverse_cdf_over_positive_mass() {
        let p = [0.0, 1.0, f32::NAN, 2.0, -4.0, f32::INFINITY, 1.0];
        // Masses: [0, 1, 0, 2, 0, 0, 1], total 4: [0,0.25) -> 1, [0.25,0.75) -> 3, [0.75,1] -> 6.
        assert_eq!(sample_categorical(&p, 0.0), 1);
        assert_eq!(sample_categorical(&p, 0.2499), 1);
        assert_eq!(sample_categorical(&p, 0.25), 3);
        assert_eq!(sample_categorical(&p, 0.7499), 3);
        assert_eq!(sample_categorical(&p, 0.75), 6);
        assert_eq!(sample_categorical(&p, 1.0), 6);
        assert_eq!(sample_categorical(&p, 7.0), 6);
        assert_eq!(sample_categorical(&p, -3.0), 1);
        assert_eq!(sample_categorical(&p, f32::NAN), 1);
        assert_eq!(sample_categorical(&[0.0, -1.0, f32::NAN], 0.5), -1);
        assert_eq!(sample_categorical(&[], 0.5), -1);
    }

    #[test]
    fn entries_tolerate_overlap_and_empty_buffers() {
        let mut x = [0.5f32, 2.0, 1.0];
        let mut idx = [0i64; 3];
        let p = x.as_mut_ptr();
        // vals overlaps x: the selection still sees the original x.
        let k = unsafe { wukong_topk_f32(p, 3, p, 3, idx.as_mut_ptr(), 3) };
        assert_eq!((k, x, idx), (3, [2.0, 1.0, 0.5], [1, 2, 0]));
        let k = unsafe {
            wukong_topk_f32(
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                0,
                idx.as_mut_ptr(),
                3,
            )
        };
        assert_eq!(k, 0);
        let m = unsafe { wukong_argsort_f32(x.as_ptr(), 3, idx.as_mut_ptr(), 3, 0) };
        assert_eq!((m, idx), (3, [2, 1, 0]));
        assert_eq!(
            unsafe { wukong_sample_categorical_f32(std::ptr::null(), 0, 0.5) },
            -1
        );
        assert_eq!(
            unsafe { wukong_sample_categorical_f32(x.as_ptr(), 3, 0.0) },
            0
        );
    }
}
