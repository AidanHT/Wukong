//! Scatter-add (embedding-gradient backward) — the **dual of the embedding gather** (`embedding.rs`):
//! where the forward pass selects `grad_w` rows by token id, the backward pass routes each output-row
//! gradient back to the weight row its token came from and **accumulates**. Over a `[T, H]` row-major
//! upstream gradient `grad_out`, a length-`T` `i32` `ids` array, and a `[V, H]` row-major weight-gradient
//! table `grad_w` (**pre-zeroed by the caller** — this kernel only adds), it computes
//!
//! ```text
//! grad_w[ids[t], :] += grad_out[t, :]      (t in [0, T), each row H contiguous f32)
//! ```
//!
//! Unlike the gather, this is **accumulation, not pure data movement**: several tokens routinely carry
//! the *same* id (a word that occurs twice in a sequence), so several `grad_out` rows **collide** on one
//! `grad_w` row and must sum. Collisions are the whole point — this kernel *is* the sum-over-occurrences
//! that an embedding's gradient is defined to be.
//!
//! **Why it is still bit-exact across every backend.** Float add is not associative, so a sum's bits
//! depend on the *order* of its addends. This kernel pins that order: row `r` accumulates the colliding
//! `grad_out` rows in **ascending token index `ti`** — always, on every path. The interpreter marshals
//! its memory through the serial kernel as the oracle; the native `@parallel` path runs the parallel
//! kernel; and because both fold each row in the identical ascending-`ti` order, they agree
//! **bit-for-bit** (there is no float-reassociation exception to keep in sync — the order *is* fixed).
//! The 256-bit `_mm256_add_ps(load grad_w, load grad_out) -> store grad_w` adds 8 lanes of `H` at a time
//! (a scalar tail handles `H % 8`); the 8-wide is across the hidden dimension, never across `ti`, so it
//! never perturbs the accumulation order. Each lane is one rounded IEEE add — the same one the scalar
//! twin performs (`_mm256_add_ps` == `+`) — so AVX2 and scalar agree lane-for-lane.
//!
//! **Deterministic parallelism — the hard part, and Wukong's structural advantage.** A C author
//! parallelizing this would split the *tokens* across threads and let them `+=` into shared `grad_w`
//! rows; colliding tokens on different threads then race, forcing **atomic** float adds — slow, and
//! worse, *non-deterministic in float order* (whichever thread wins the race adds first), so the result's
//! low bits wobble run to run. Wukong instead splits the **output rows**: each rayon chunk owns a
//! disjoint range of `grad_w` rows `[r0, r1)`, scans **all** `T` tokens in ascending `ti`, and applies
//! `grad_out[ti]` to `grad_w[id]` **only** when `r0 <= id < r1`. Disjoint output rows mean no two threads
//! ever write the same `grad_w` element — no atomics, no races — and each row is still folded by a single
//! thread in ascending `ti`, so the result is **bit-identical to serial**, independent of thread count and
//! scheduling. The cost is that every chunk re-scans the `T` ids (cheap integer compares); the payoff is a
//! lock-free, run-to-run-stable gradient the differential gate can hold to the serial oracle. Below
//! `SCATTER_PAR_MIN` output rows the split is not worth its overhead and the parallel entry runs serial.
//!
//! **Out-of-range guard.** A negative or `>= v` id names no weight row; like the gather's safe guard,
//! such a token is simply **skipped** (it contributes nothing — there is no row to accumulate into),
//! identically in the AVX2, scalar, and parallel paths. In-range ids — the only case a well-typed program
//! produces — accumulate exactly.

/// Accumulate one gradient row into one weight-gradient row: `grad_w_row[0..h] += grad_out_row[0..h]`,
/// 8 f32 per step with a scalar tail. AVX2 when available, else the scalar twin — both perform the
/// identical lane-wise IEEE add (`_mm256_add_ps` == `+`, one rounding per lane), so they agree
/// bit-for-bit.
///
/// # Safety
/// `grad_w_row` valid for `h` f32 (readable+writable), `grad_out_row` valid for `h` f32 (readable),
/// non-overlapping.
#[inline]
unsafe fn add_row(grad_w_row: *mut f32, grad_out_row: *const f32, h: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            add_row_avx2(grad_w_row, grad_out_row, h);
            return;
        }
    }
    add_row_scalar(grad_w_row, grad_out_row, h);
}

/// AVX2 row accumulate: 8 f32 (`_mm256_add_ps` of two `_mm256_loadu_ps`, then `_mm256_storeu_ps` back
/// into `grad_w_row`) per step, scalar tail for `h % 8`.
///
/// # Safety
/// Operand-size contract of [`add_row`]; AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_row_avx2(grad_w_row: *mut f32, grad_out_row: *const f32, h: usize) {
    use std::arch::x86_64::*;
    let mut j = 0;
    while j + 8 <= h {
        let acc = _mm256_loadu_ps(grad_w_row.add(j));
        let inc = _mm256_loadu_ps(grad_out_row.add(j));
        _mm256_storeu_ps(grad_w_row.add(j), _mm256_add_ps(acc, inc));
        j += 8;
    }
    while j < h {
        *grad_w_row.add(j) += *grad_out_row.add(j);
        j += 1;
    }
}

/// Scalar row accumulate (the no-AVX2 fallback and the bit-exact reference): element-by-element
/// `grad_w_row[j] += grad_out_row[j]`, the identical lane-wise add the AVX2 path performs.
///
/// # Safety
/// Operand-size contract of [`add_row`].
#[inline]
unsafe fn add_row_scalar(grad_w_row: *mut f32, grad_out_row: *const f32, h: usize) {
    for j in 0..h {
        *grad_w_row.add(j) += *grad_out_row.add(j);
    }
}

/// Scatter-add the tokens whose id lands in the output-row range `[r0, r1)`: scan **all** `t` tokens in
/// ascending `ti`, and for each in-range id with `r0 <= id < r1` accumulate `grad_out[ti, :]` into
/// `grad_w[id, :]`. Shared by the serial and parallel kernels — the serial call owns every row
/// (`r0 = 0, r1 = v`), so its filter is exactly the `0 <= id < v` guard; a parallel chunk owns a disjoint
/// sub-range, so chunks write disjoint `grad_w` rows (race-free) while each row still folds its colliding
/// tokens in ascending `ti` (== serial). A negative id, or one whose row this range does not own, is
/// skipped. Because `r1 <= v`, the `id < r1` test is also the in-table `id < v` bound.
///
/// # Safety
/// `grad_w` valid for `v*h` f32 (only rows in `[r0, r1)` are written), `grad_out` valid for `t*h` f32,
/// `ids` valid for `t` `i32`; `r0 <= r1 <= v`; `grad_w`/`grad_out` non-overlapping.
#[inline]
unsafe fn scatter_rows(
    grad_w: *mut f32,
    grad_out: *const f32,
    ids: *const i32,
    t: usize,
    h: usize,
    r0: usize,
    r1: usize,
) {
    for ti in 0..t {
        let id = *ids.add(ti);
        // Out-of-range ids (`< 0`) contribute nothing; an in-range id accumulates, but only into the
        // grad_w rows this chunk owns (`r0 <= id < r1`) — so chunks touch disjoint output rows, and the
        // additive fold of each row stays in ascending `ti` order.
        if id >= 0 {
            let idu = id as usize;
            if idu >= r0 && idu < r1 {
                add_row(grad_w.add(idu * h), grad_out.add(ti * h), h);
            }
        }
    }
}

/// Output-row count (`V`) below which the parallel scatter just runs serially: the row-range split has to
/// re-scan all `T` ids per chunk, so on a small weight-gradient table the serial kernel wins.
const SCATTER_PAR_MIN: usize = 64;

/// `grad_w[ids[t], :] += grad_out[t, :]` accumulated over a `[T, H]` gradient `grad_out`, length-`T` `i32`
/// `ids`, and a `[V, H]` weight-gradient table `grad_w` — the embedding backward, single-threaded.
/// `grad_w` must be **pre-zeroed by the caller** (this kernel only adds); colliding ids sum in ascending
/// `ti` order, out-of-range ids are skipped.
///
/// # Safety
/// `grad_w` valid for `v*h` f32 (readable+writable), `grad_out` valid for `t*h` f32, `ids` valid for `t`
/// `i32`, non-overlapping `grad_w`/`grad_out`.
#[no_mangle]
pub unsafe extern "C" fn wukong_scatter_add_f32(
    grad_w: *mut f32,
    grad_out: *const f32,
    ids: *const i32,
    t: usize,
    h: usize,
    v: usize,
) {
    if t == 0 || h == 0 || v == 0 {
        return;
    }
    // Serial owns the whole table: the `[0, v)` filter is exactly the `0 <= id < v` guard.
    scatter_rows(grad_w, grad_out, ids, t, h, 0, v);
}

/// Multi-threaded `grad_w[ids[t], :] += grad_out[t, :]`: the `V` **output rows** of `grad_w` are split
/// into disjoint ranges across cores (NOT the tokens), each core scanning all `T` ids and accumulating
/// only those landing in its range. Disjoint output rows make it lock-free and race-free; per-row
/// ascending-`ti` folding makes it **bit-identical to [`wukong_scatter_add_f32`]** regardless of thread
/// count or scheduling (the interpreter runs the serial form as the oracle, native `@parallel` runs this
/// one, and the differential gate compares them). Below `SCATTER_PAR_MIN` rows it runs the serial path.
/// `grad_w` must be pre-zeroed by the caller.
///
/// # Safety
/// Operand-size contract of [`wukong_scatter_add_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_scatter_add_f32_parallel(
    grad_w: *mut f32,
    grad_out: *const f32,
    ids: *const i32,
    t: usize,
    h: usize,
    v: usize,
) {
    if t == 0 || h == 0 || v == 0 {
        return;
    }
    if v < SCATTER_PAR_MIN {
        scatter_rows(grad_w, grad_out, ids, t, h, 0, v);
        return;
    }
    use rayon::prelude::*;
    // One chunk of *output rows* per core; the last chunk absorbs the remainder. Each chunk owns a
    // disjoint `grad_w` row range `[r0, r1)` and re-scans all `T` ids, so writes never overlap across
    // chunks. Raw pointers cross the rayon boundary as integers (the pointees outlive this blocking call;
    // the chunks together cover `[0, v)` exactly once, so every in-range token is applied exactly once).
    // This fork can be the process's FIRST rayon touch, so it must provision the global pool first —
    // [`crate::ensure_global_pool`]'s stated precondition. Forking bare builds rayon's default
    // 2 MiB-stack registry, so the runtime's later 16 MiB `build_global` silently loses the race and
    // outlined `@parallel` region bodies are left on undersized stacks. Idempotent (`Once`) and
    // provisioning-only: the `current_num_threads()` read below resolves `RAYON_NUM_THREADS` the same
    // way either way, so the chunk count — and with it every chunk boundary — is unchanged.
    crate::ensure_global_pool();
    let nthreads = rayon::current_num_threads().max(1);
    let per = v.div_ceil(nthreads).max(1);
    let nchunks = v.div_ceil(per);
    let (gw_addr, go_addr, ids_addr) = (grad_w as usize, grad_out as usize, ids as usize);
    (0..nchunks).into_par_iter().for_each(|c| {
        let r0 = c * per;
        let r1 = (r0 + per).min(v);
        // SAFETY: disjoint `grad_w` row range per chunk (writes never overlap across chunks); pointers
        // re-derived from the captured addresses (so they are Send); reads of `grad_out`/`ids` are shared
        // and immutable.
        unsafe {
            scatter_rows(
                gw_addr as *mut f32,
                go_addr as *const f32,
                ids_addr as *const i32,
                t,
                h,
                r0,
                r1,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive scalar scatter-add — the bit-exact reference. Starts from a zeroed `[V, H]` table and folds
    /// `grad_out[ti, :]` into `grad_w[ids[ti], :]` in ascending `ti`; an id out of `[0, v)` is skipped,
    /// matching the kernel's guard.
    fn naive(grad_out: &[f32], ids: &[i32], t: usize, h: usize, v: usize) -> Vec<f32> {
        let mut grad_w = vec![0.0f32; v * h];
        for (ti, &id) in ids.iter().enumerate().take(t) {
            if id >= 0 && (id as usize) < v {
                let dst = id as usize * h;
                let src = ti * h;
                for d in 0..h {
                    grad_w[dst + d] += grad_out[src + d];
                }
            }
            // else: out-of-range id contributes nothing.
        }
        grad_w
    }

    /// The kernel (AVX2 serial and the multicore output-row split) must equal the naive scatter-add
    /// **exactly**. The grad_out values are small integers, so every per-row partial sum is exactly
    /// representable in f32 (integers below 2^24) and ascending-`ti` accumulation is genuinely
    /// order-stable — hence `assert_eq` on the raw bits is sound. Shapes straddle the 8-lane H edge
    /// (non-multiples, thin, wide) and the parallel row threshold, with ids that repeat heavily (T > V),
    /// hit row 0, and hit the last row.
    #[test]
    fn scatter_matches_naive_and_parallel() {
        for &(t, h, v) in &[
            (1usize, 1usize, 1usize),
            (3, 5, 4),
            (8, 8, 8),
            (7, 9, 5),
            (17, 31, 11),
            (4, 64, 3),
            (100, 70, 23),
            (1, 257, 2),
            (300, 16, 80),  // v >= SCATTER_PAR_MIN so the multicore path runs
            (500, 17, 64),  // v == SCATTER_PAR_MIN (boundary, multicore) with an H tail
            (260, 70, 96),  // multicore + wide H with a tail; T >> V => heavy collisions
        ] {
            // Small integers (1..=9), varying by flat index so any mis-scatter — wrong row, wrong token,
            // dropped or duplicated token, or an off-by-one within the row — changes a sum and shows as a
            // bit mismatch; every partial sum stays an exactly-representable integer.
            let grad_out: Vec<f32> = (0..t * h).map(|i| ((i % 9) + 1) as f32).collect();
            // Deterministic ids in [0, v): with T > V they repeat heavily; row 0 and the last row appear.
            let ids: Vec<i32> = (0..t).map(|i| ((i * 7 + 1) % v) as i32).collect();
            let want = naive(&grad_out, &ids, t, h, v);
            // The caller pre-zeros the accumulator table; the kernel only adds.
            let mut got = vec![0.0f32; v * h];
            let mut got_par = vec![0.0f32; v * h];
            unsafe {
                wukong_scatter_add_f32(
                    got.as_mut_ptr(),
                    grad_out.as_ptr(),
                    ids.as_ptr(),
                    t,
                    h,
                    v,
                );
                wukong_scatter_add_f32_parallel(
                    got_par.as_mut_ptr(),
                    grad_out.as_ptr(),
                    ids.as_ptr(),
                    t,
                    h,
                    v,
                );
            }
            assert_eq!(got, want, "scatter {t}x{h} v={v} vs naive");
            assert_eq!(got, got_par, "scatter serial vs parallel {t}x{h} v={v}");
        }
    }

    /// The determinism stress: every id pattern below — all tokens on one row (row 0, the last row, an
    /// interior row), two dense rows, an even spread, a pseudo-random spread, and a mix that also drops
    /// out-of-range ids — runs through the **real multicore path** (V >= SCATTER_PAR_MIN) and must equal
    /// both the naive reference and the serial kernel bit-for-bit. The all-equal patterns are the worst
    /// case for the output-row split: every collision lands in one chunk, which must fold all `T` tokens
    /// in ascending `ti` exactly as serial does, while the other chunks scan the ids and add nothing.
    #[test]
    fn heavy_collisions_serial_equals_parallel() {
        let (t, h, v) = (256usize, 24usize, 96usize); // v >= SCATTER_PAR_MIN => parallel path
        let grad_out: Vec<f32> = (0..t * h).map(|i| ((i % 9) + 1) as f32).collect();
        let id_patterns: Vec<Vec<i32>> = vec![
            vec![0i32; t],                                       // all collide on row 0
            vec![(v - 1) as i32; t],                             // all collide on the last row
            vec![7i32; t],                                       // all collide on an interior row
            (0..t).map(|i| (i % 2) as i32).collect(),            // two rows, dense repeats
            (0..t).map(|i| (i % v) as i32).collect(),            // even spread, ~T/V hits per row
            (0..t).map(|i| ((i * 13 + 5) % v) as i32).collect(), // pseudo-random repeats
            // Mixed: some tokens out of range (negative and >= v), the rest in-range repeats — exercises
            // the skip guard on the genuine multicore path.
            (0..t)
                .map(|i| {
                    if i % 5 == 0 {
                        -1
                    } else if i % 7 == 0 {
                        (v + 3) as i32
                    } else {
                        ((i * 13 + 5) % v) as i32
                    }
                })
                .collect(),
        ];
        for ids in &id_patterns {
            let want = naive(&grad_out, ids, t, h, v);
            let mut got = vec![0.0f32; v * h];
            let mut got_par = vec![0.0f32; v * h];
            unsafe {
                wukong_scatter_add_f32(got.as_mut_ptr(), grad_out.as_ptr(), ids.as_ptr(), t, h, v);
                wukong_scatter_add_f32_parallel(
                    got_par.as_mut_ptr(),
                    grad_out.as_ptr(),
                    ids.as_ptr(),
                    t,
                    h,
                    v,
                );
            }
            assert_eq!(got, want, "collision scatter vs naive");
            assert_eq!(got, got_par, "collision serial vs parallel (bit-exact determinism)");
        }
    }

    /// Out-of-range ids (negative and `>= v`) are skipped — they accumulate nothing — in both the serial
    /// and parallel paths, while the in-range tokens around them still accumulate exactly.
    #[test]
    fn out_of_range_ids_are_skipped() {
        let (t, h, v) = (6usize, 12usize, 4usize);
        let grad_out: Vec<f32> = (0..t * h).map(|i| ((i % 7) + 1) as f32).collect();
        // ids: row0, skip, row2, skip(=v), row3, skip. So row 1 stays untouched (zero).
        let ids: Vec<i32> = vec![0, -1, 2, v as i32, 3, -100];
        let want = naive(&grad_out, &ids, t, h, v);
        let mut got = vec![0.0f32; v * h];
        let mut got_par = vec![0.0f32; v * h];
        unsafe {
            wukong_scatter_add_f32(got.as_mut_ptr(), grad_out.as_ptr(), ids.as_ptr(), t, h, v);
            wukong_scatter_add_f32_parallel(
                got_par.as_mut_ptr(),
                grad_out.as_ptr(),
                ids.as_ptr(),
                t,
                h,
                v,
            );
        }
        assert_eq!(got, want, "out-of-range scatter vs naive");
        assert_eq!(got, got_par, "out-of-range serial vs parallel");
        // Spot-check: each in-range token landed on its row, and the untouched row (1) is all zero.
        for d in 0..h {
            assert_eq!(got[d], grad_out[d], "row 0 = token 0");
            assert_eq!(got[2 * h + d], grad_out[2 * h + d], "row 2 = token 2");
            assert_eq!(got[3 * h + d], grad_out[4 * h + d], "row 3 = token 4");
            assert_eq!(got[h + d], 0.0, "row 1 was never targeted");
        }
    }

    /// Edge: zero tokens, zero hidden, or an empty table (`v == 0`) is a no-op — don't write, don't panic.
    #[test]
    fn degenerate_shapes_are_noops() {
        let grad_out = vec![1.0f32; 8];
        let ids = [0i32; 2];
        let mut grad_w = vec![42.0f32; 8];
        unsafe {
            wukong_scatter_add_f32(grad_w.as_mut_ptr(), grad_out.as_ptr(), ids.as_ptr(), 0, 4, 2);
            wukong_scatter_add_f32(grad_w.as_mut_ptr(), grad_out.as_ptr(), ids.as_ptr(), 2, 0, 2);
            wukong_scatter_add_f32(grad_w.as_mut_ptr(), grad_out.as_ptr(), ids.as_ptr(), 2, 4, 0);
            wukong_scatter_add_f32_parallel(
                grad_w.as_mut_ptr(),
                grad_out.as_ptr(),
                ids.as_ptr(),
                0,
                4,
                2,
            );
        }
        assert!(grad_w.iter().all(|&x| x == 42.0), "no-op must not write grad_w");
    }
}
