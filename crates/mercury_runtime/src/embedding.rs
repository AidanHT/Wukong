//! Embedding lookup (gather rows by index) — the **first layer of every LLM**: each token id selects
//! one row of the embedding table. Over a `[V, H]` row-major weight table `weight` and a length-`T`
//! `i32` index array `ids`, produce a `[T, H]` row-major output where
//!
//! ```text
//! out[t, :] = weight[ids[t], :]      (t in [0, T), each row H contiguous f32)
//! ```
//!
//! Pure **data movement** — every output element is a verbatim copy of a weight element, no
//! arithmetic — so the result is **bit-identical to the naive scalar gather on every backend** (like
//! the cache-blocked transpose, the differential gate is trivial: there is no float reassociation to
//! keep in sync). The 256-bit `_mm256_loadu_ps`/`_mm256_storeu_ps` copy just moves the bytes a cache
//! line at a time; a scalar tail handles `H % 8`.
//!
//! Each output row `t` is independent (it reads a disjoint `weight` row — possibly the *same* row as
//! another `t`, but reads never conflict — and writes its own disjoint `out` row), so the `_parallel`
//! sibling maps the `T` rows across cores with **no cross-row combine**: `serial == parallel`
//! bit-for-bit, thread count and order irrelevant (the interpreter marshals the serial form as the
//! oracle, the native `@parallel` path runs the parallel one, and the differential gate compares them).
//!
//! **Out-of-range guard.** A negative or `>= v` id has no valid weight row; rather than read out of
//! bounds the kernel writes that output row as all-zeros (a safe, deterministic fallback identical in
//! the AVX2, scalar, and parallel paths). In-range ids — the only case a well-typed program produces —
//! copy exactly.

/// Copy one weight row into one output row: `out_row[0..h] = weight_row[0..h]`, 8 f32 per step with a
/// scalar tail. AVX2 when available, else the scalar copy — both move the identical bytes (a copy is
/// precision/representation-agnostic), so they agree bit-for-bit.
///
/// # Safety
/// `out_row` valid for `h` f32 (writable), `weight_row` valid for `h` f32 (readable), non-overlapping.
#[inline]
unsafe fn copy_row(out_row: *mut f32, weight_row: *const f32, h: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            copy_row_avx2(out_row, weight_row, h);
            return;
        }
    }
    copy_row_scalar(out_row, weight_row, h);
}

/// AVX2 row copy: 8 f32 (`_mm256_loadu_ps`/`_mm256_storeu_ps`) per step, scalar tail for `h % 8`.
///
/// # Safety
/// Operand-size contract of [`copy_row`]; AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_row_avx2(out_row: *mut f32, weight_row: *const f32, h: usize) {
    use std::arch::x86_64::*;
    let mut j = 0;
    while j + 8 <= h {
        let v = _mm256_loadu_ps(weight_row.add(j));
        _mm256_storeu_ps(out_row.add(j), v);
        j += 8;
    }
    while j < h {
        *out_row.add(j) = *weight_row.add(j);
        j += 1;
    }
}

/// Scalar row copy (the no-AVX2 fallback and the bit-exact reference): element-by-element, identical
/// bytes to the AVX2 path.
///
/// # Safety
/// Operand-size contract of [`copy_row`].
#[inline]
unsafe fn copy_row_scalar(out_row: *mut f32, weight_row: *const f32, h: usize) {
    for j in 0..h {
        *out_row.add(j) = *weight_row.add(j);
    }
}

/// Zero one output row (`out_row[0..h] = 0`), the out-of-range fallback. 8 f32 per step + scalar tail.
///
/// # Safety
/// `out_row` valid for `h` f32 (writable).
#[inline]
unsafe fn zero_row(out_row: *mut f32, h: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            use std::arch::x86_64::*;
            let z = _mm256_setzero_ps();
            let mut j = 0;
            while j + 8 <= h {
                _mm256_storeu_ps(out_row.add(j), z);
                j += 8;
            }
            while j < h {
                *out_row.add(j) = 0.0;
                j += 1;
            }
            return;
        }
    }
    for j in 0..h {
        *out_row.add(j) = 0.0;
    }
}

/// Gather output rows `[t0, t1)`: for each `t`, copy `weight[ids[t], :]` into `out[t, :]` (or zero the
/// row if `ids[t]` is out of `[0, v)`). Shared by the serial and parallel kernels; each call writes only
/// `out` rows in `[t0, t1)`, which are disjoint across row ranges → the parallel split is race-free and
/// order-independent (reads of `weight` may overlap, which is fine — they never mutate).
///
/// # Safety
/// `out` valid for `t*h` f32, `weight` valid for `v*h` f32, `ids` valid for `t` `i32`; `t0 <= t1 <= t`.
#[inline]
unsafe fn embedding_rows(
    out: *mut f32,
    weight: *const f32,
    ids: *const i32,
    h: usize,
    v: usize,
    t0: usize,
    t1: usize,
) {
    for t in t0..t1 {
        let id = *ids.add(t);
        let out_row = out.add(t * h);
        // In-range ids (the only case a well-typed program produces) copy the weight row; an
        // out-of-range id writes zeros rather than read out of bounds.
        if id >= 0 && (id as usize) < v {
            copy_row(out_row, weight.add(id as usize * h), h);
        } else {
            zero_row(out_row, h);
        }
    }
}

/// Row count below which the parallel gather just runs serially (the rayon split is not worth its
/// overhead on a small lookup).
const EMBEDDING_PAR_MIN: usize = 64;

/// `out[t, :] = weight[ids[t], :]` over a `[T, H]` output, `[V, H]` weight table, and length-`T` `i32`
/// `ids` — embedding lookup, single-threaded.
///
/// # Safety
/// `out` valid for `t*h` f32, `weight` valid for `v*h` f32, `ids` valid for `t` `i32`, non-overlapping
/// `out`/`weight`.
#[no_mangle]
pub unsafe extern "C" fn mercury_embedding_f32(
    out: *mut f32,
    weight: *const f32,
    ids: *const i32,
    t: usize,
    h: usize,
    v: usize,
) {
    if t == 0 || h == 0 {
        return;
    }
    embedding_rows(out, weight, ids, h, v, 0, t);
}

/// Multi-threaded `out[t, :] = weight[ids[t], :]`: the independent `T` output rows are spread across
/// cores. Each row writes a disjoint `out` slice and only reads `weight`/`ids`, so the result is
/// **bit-identical to [`mercury_embedding_f32`]** (a permutation/gather, no accumulation — thread
/// count and order are irrelevant). Below `EMBEDDING_PAR_MIN` rows it runs the serial path.
///
/// # Safety
/// Operand-size contract of [`mercury_embedding_f32`].
#[no_mangle]
pub unsafe extern "C" fn mercury_embedding_f32_parallel(
    out: *mut f32,
    weight: *const f32,
    ids: *const i32,
    t: usize,
    h: usize,
    v: usize,
) {
    if t == 0 || h == 0 {
        return;
    }
    if t < EMBEDDING_PAR_MIN {
        embedding_rows(out, weight, ids, h, v, 0, t);
        return;
    }
    use rayon::prelude::*;
    // One chunk of rows per core; the last chunk absorbs the remainder. Raw pointers cross the rayon
    // boundary as integers (the pointees outlive this blocking call; chunks touch disjoint `out` rows).
    let nthreads = rayon::current_num_threads().max(1);
    let per = t.div_ceil(nthreads).max(1);
    let nchunks = t.div_ceil(per);
    let (out_addr, w_addr, ids_addr) = (out as usize, weight as usize, ids as usize);
    (0..nchunks).into_par_iter().for_each(|c| {
        let t0 = c * per;
        let t1 = (t0 + per).min(t);
        // SAFETY: disjoint `out` row range per chunk; pointers re-derived from the captured addresses
        // (so they are Send); reads of `weight`/`ids` are shared and immutable.
        unsafe {
            embedding_rows(
                out_addr as *mut f32,
                w_addr as *const f32,
                ids_addr as *const i32,
                h,
                v,
                t0,
                t1,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive scalar gather — the bit-exact reference. `ids[t]` out of `[0, v)` zeros the row, matching
    /// the kernel's guard.
    fn naive(weight: &[f32], ids: &[i32], t: usize, h: usize, v: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; t * h];
        for (ti, &id) in ids.iter().enumerate() {
            if id >= 0 && (id as usize) < v {
                let src = id as usize * h;
                out[ti * h..ti * h + h].copy_from_slice(&weight[src..src + h]);
            }
            // else: row already zero.
        }
        out
    }

    /// The kernel (AVX2 serial and the multicore chunk split) must equal the naive gather **exactly** —
    /// it is a copy/permutation of the inputs, so "exactly" is literal bit equality, no tolerance.
    /// Shapes straddle the 8-lane H edge (non-multiples, thin, wide) and the parallel row threshold,
    /// with ids that repeat, hit row 0, and hit the last row.
    #[test]
    fn embedding_matches_naive_and_parallel() {
        for &(t, h, v) in &[
            (1usize, 1usize, 1usize),
            (3, 5, 4),
            (8, 8, 8),
            (7, 9, 5),
            (17, 31, 11),
            (4, 64, 3),
            (100, 70, 23),
            (1, 257, 2),
            (200, 16, 50), // > EMBEDDING_PAR_MIN so the multicore path runs
        ] {
            // Distinct, exactly-representable weight values so any mis-gather shows as a bit mismatch.
            let weight: Vec<f32> = (0..v * h).map(|i| i as f32 * 0.5 - 3.0).collect();
            // Deterministic ids in [0, v): repeats, row 0, and the last row all appear.
            let ids: Vec<i32> = (0..t).map(|i| ((i * 7 + 1) % v) as i32).collect();
            let want = naive(&weight, &ids, t, h, v);
            let mut got = vec![0.0f32; t * h];
            let mut got_par = vec![0.0f32; t * h];
            unsafe {
                mercury_embedding_f32(
                    got.as_mut_ptr(),
                    weight.as_ptr(),
                    ids.as_ptr(),
                    t,
                    h,
                    v,
                );
                mercury_embedding_f32_parallel(
                    got_par.as_mut_ptr(),
                    weight.as_ptr(),
                    ids.as_ptr(),
                    t,
                    h,
                    v,
                );
            }
            assert_eq!(got, want, "embedding {t}x{h} v={v} vs naive");
            assert_eq!(got, got_par, "embedding serial vs parallel {t}x{h} v={v}");
        }
    }

    /// Out-of-range ids (negative and `>= v`) zero their output row in both the serial and parallel
    /// paths (the safe fallback), and in-range rows around them still copy exactly.
    #[test]
    fn out_of_range_ids_zero_the_row() {
        let (t, h, v) = (6usize, 12usize, 4usize);
        let weight: Vec<f32> = (0..v * h).map(|i| i as f32 + 1.0).collect();
        // Mix in-range with a negative and an over-large id.
        let ids: Vec<i32> = vec![0, -1, 2, v as i32, 3, -100];
        let want = naive(&weight, &ids, t, h, v);
        let mut got = vec![9.0f32; t * h];
        let mut got_par = vec![9.0f32; t * h];
        unsafe {
            mercury_embedding_f32(got.as_mut_ptr(), weight.as_ptr(), ids.as_ptr(), t, h, v);
            mercury_embedding_f32_parallel(
                got_par.as_mut_ptr(),
                weight.as_ptr(),
                ids.as_ptr(),
                t,
                h,
                v,
            );
        }
        assert_eq!(got, want, "out-of-range gather vs naive");
        assert_eq!(got, got_par, "out-of-range serial vs parallel");
        // Spot-check: the out-of-range rows (t=1,3,5) are all zero.
        for &ti in &[1usize, 3, 5] {
            assert!(
                got[ti * h..ti * h + h].iter().all(|&x| x == 0.0),
                "row {ti} should be zeroed"
            );
        }
    }

    /// Edge: zero rows or zero hidden is a no-op (don't write, don't panic).
    #[test]
    fn degenerate_shapes_are_noops() {
        let weight = vec![1.0f32; 8];
        let ids = [0i32; 2];
        let mut out = vec![42.0f32; 8];
        unsafe {
            mercury_embedding_f32(out.as_mut_ptr(), weight.as_ptr(), ids.as_ptr(), 0, 4, 2);
            mercury_embedding_f32(out.as_mut_ptr(), weight.as_ptr(), ids.as_ptr(), 2, 0, 2);
            mercury_embedding_f32_parallel(out.as_mut_ptr(), weight.as_ptr(), ids.as_ptr(), 0, 4, 2);
        }
        assert!(out.iter().all(|&x| x == 42.0), "no-op must not write out");
    }
}
