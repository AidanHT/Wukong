//! First-order **linear recurrence** / selective scan over the last axis of a `[rows, cols]` row-major
//! f32 matrix — the time-folded state update at the heart of state-space sequence models (Mamba / S4 /
//! RWKV linear attention) and the exponential moving average (EMA):
//!
//! ```text
//! for each row r:  h_{-1} = 0;  h_t = a[r,t] * h_{t-1} + b[r,t];  out[r,t] = h_t   (t = 0..cols)
//! ```
//!
//! `a` is the per-step decay/gate (the "A" / forget term), `b` the per-step input (the "B·x" term); each
//! row is an **independent sequence** whose hidden state `h` resets to 0 at `t = 0`. EMA is the special
//! case `a ≡ α`, `b_t = (1−α)·x_t`; a selective SSM scan is this with data-dependent `a_t`, `b_t`.
//!
//! **Why this is a win.** The inner time loop is a *true loop-carried dependency* — `h_t` needs `h_{t-1}`
//! — so gcc `-O3 -march=native` / rustc cannot auto-vectorize it (the recurrence defeats them, exactly
//! like the prefix-sum scan in `cumsum.rs`): they emit **one serial multiply-add chain** per row. That
//! chain is **latency-bound, not bandwidth-bound** — each step waits on the previous step's `mul`+`add`
//! latency, so a lone chain sustains only a few GB/s, far below DRAM. The lever is **instruction-level
//! parallelism across independent rows** (`lrscan_rows`): the rows are independent recurrences, so
//! processing **4 rows at once** — interleaving their four `h`-updates down a shared `t` — keeps four
//! independent `mul`+`add` chains in flight and fills the arithmetic/load-store ports the single chain
//! leaves ~3/4 idle. Each row stays its own contiguous in-order recurrence (no SIMD: 4 consecutive rows
//! at a fixed `t` are strided by `cols`, so we keep four contiguous streams, not one strided 8-lane
//! gather/scatter), so a row's arithmetic is unchanged. gcc/rustc cannot legally re-order an f32
//! recurrence across rows without `-ffast-math`, so this 4-way ILP is the single-core lever, and the
//! `_parallel` map of independent row *chunks* across cores stacks on top of it.
//!
//! **Bit-exactness contract.** The step is a plain `mul` then `add` (`*a * h + *b`, two roundings —
//! matching the `FMul`/`FAdd` the front-end lowers `a*h + b` to, *not* `f32::mul_add`'s single fused
//! rounding), and the rows are independent with no cross-row combine — so the per-row expression tree is
//! identical in the serial entry, the parallel entry (disjoint row chunks), and the 4-row-interleaved
//! block (which changes only *issue order*, never a row's own arithmetic): there is **no reassociation
//! anywhere**. Serial == parallel **bit-for-bit** on any thread count, and both equal the naive scalar
//! reference exactly (`assert_eq!` on the bits, not a tolerance). The interpreter marshals the serial
//! form, so the differential oracle stays exact with no documented exception.

use rayon::prelude::*;

/// Scan one row in place along time: `h ← a[t]·h + b[t]; out[t] ← h` for `t = 0..cols`, with `h` starting
/// at `0.0` and carried in a register. The tail helper (rows the 4-wide block doesn't cover) and the
/// building block the interleaved scan reduces to.
///
/// The step is `*a.add(t) * h + *b.add(t)` — a multiply then an add (**two** roundings), matching the
/// `FMul`/`FAdd` the front-end lowers `a*h + b` to; it is intentionally *not* `f32::mul_add` (which on a
/// target without the `fma` feature is a libm `fmaf` *call* that serializes the chain). The interpreter
/// reproduces it by calling this exact kernel, so the oracle stays bit-for-bit.
///
/// # Safety
/// `a` and `b` valid for `cols` f32 and `out` valid for `cols` f32; `out` may alias `a` and/or `b` —
/// each `a[t]`/`b[t]` is consumed before `out[t]` is written and later steps only read higher indices,
/// so the scan is safe **in place**.
#[inline]
unsafe fn lrscan_row(a: *const f32, b: *const f32, out: *mut f32, cols: usize) {
    let mut h = 0.0f32;
    for t in 0..cols {
        h = *a.add(t) * h + *b.add(t);
        *out.add(t) = h;
    }
}

/// Scan rows `[r0, r1)` of a `[_, cols]` matrix, **4 independent rows interleaved** so four `mul`+`add`
/// chains are in flight at once — the single-core lever (see the module header: the scan is latency-bound
/// on the per-row carry, so 4-way ILP across independent rows fills the ports a single serial chain leaves
/// idle). Each row remains its own in-order recurrence, so the result is **bit-identical** to the 1-row
/// [`lrscan_row`] — serial == parallel == interpreter. The `< 4` leftover rows fall to `lrscan_row`.
///
/// # Safety
/// `a`, `b`, `out` valid for at least `r1 * cols` f32; rows `[r0, r1)` lie within that range.
#[inline]
unsafe fn lrscan_rows(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    cols: usize,
    r0: usize,
    r1: usize,
) {
    let mut r = r0;
    // 4 rows at a time: four independent `h` carries down a shared `t`, four `mul`+`add` chains in flight.
    while r + 4 <= r1 {
        let (o0, o1, o2, o3) = (r * cols, (r + 1) * cols, (r + 2) * cols, (r + 3) * cols);
        let (mut h0, mut h1, mut h2, mut h3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for t in 0..cols {
            h0 = *a.add(o0 + t) * h0 + *b.add(o0 + t);
            h1 = *a.add(o1 + t) * h1 + *b.add(o1 + t);
            h2 = *a.add(o2 + t) * h2 + *b.add(o2 + t);
            h3 = *a.add(o3 + t) * h3 + *b.add(o3 + t);
            *out.add(o0 + t) = h0;
            *out.add(o1 + t) = h1;
            *out.add(o2 + t) = h2;
            *out.add(o3 + t) = h3;
        }
        r += 4;
    }
    // Tail rows (fewer than 4): the plain single-row scan, same arithmetic.
    while r < r1 {
        let off = r * cols;
        lrscan_row(a.add(off), b.add(off), out.add(off), cols);
        r += 1;
    }
}

/// Row count below which the parallel scan just runs serially: each row is a light scalar pass, so the
/// rayon split only pays for itself once there are many independent rows to spread across cores.
const LRSCAN_PAR_MIN: usize = 64;

/// `out[r,t] = h_t` where `h_t = a[r,t]·h_{t-1} + b[r,t]`, `h_{-1} = 0`, over a `[rows, cols]` row-major
/// matrix — single-threaded (4-row-interleaved for ILP). Each row is an independent first-order linear
/// recurrence (the hidden state resets to 0 at the start of every row).
///
/// # Safety
/// `a`, `b`, and `out` must each be valid for `rows * cols` f32; `out` may be distinct from or alias
/// `a`/`b` (the scan is in-place-safe per row).
#[no_mangle]
pub unsafe extern "C" fn mercury_lrscan_f32(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    lrscan_rows(a, b, out, cols, 0, rows);
}

/// Multicore `out[r,t] = a[r,t]·h_{t-1} + b[r,t]` — **bit-identical** to [`mercury_lrscan_f32`]. Rows are
/// independent (each row's `h` starts at 0 and never crosses rows), so the parallel path slices the rows
/// into one contiguous chunk per core and runs the *identical* 4-row-interleaved [`lrscan_rows`] on each
/// — the result is independent of thread count and equals the serial kernel the interpreter marshals,
/// bit-for-bit. Below `LRSCAN_PAR_MIN` rows it runs serial.
///
/// # Safety
/// `a`, `b`, and `out` must each be valid for `rows * cols` f32; each row is written by exactly one task
/// (disjoint row sub-slices), and `out` may alias `a`/`b` in place per row.
#[no_mangle]
pub unsafe extern "C" fn mercury_lrscan_f32_parallel(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    if rows < LRSCAN_PAR_MIN {
        lrscan_rows(a, b, out, cols, 0, rows);
        return;
    }
    // Raw pointers cross the rayon boundary as integers (same pattern as the parallel cumsum / norm);
    // each task owns a disjoint *chunk* of rows (so it keeps the 4-row interleave), not one row.
    let (aa, ba, oa) = (a as usize, b as usize, out as usize);
    let nthreads = rayon::current_num_threads().max(1);
    let per = rows.div_ceil(nthreads).max(1);
    let nchunks = rows.div_ceil(per);
    (0..nchunks).into_par_iter().for_each(|c| {
        let r0 = c * per;
        let r1 = (r0 + per).min(rows);
        // SAFETY: rows [r0, r1) are disjoint across chunks; pointers re-derived from captured addresses.
        unsafe {
            lrscan_rows(
                aa as *const f32,
                ba as *const f32,
                oa as *mut f32,
                cols,
                r0,
                r1,
            )
        };
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-step decay `a`, **bounded in (−1, 1)** (values in {−0.8, −0.6, …, 0.8}) so the recurrence is a
    /// contraction and `h` stays finite and modestly sized — keeping every intermediate an ordinary
    /// finite f32 (no inf/NaN, so `assert_eq!` on the bits is meaningful and the comparison is honest).
    fn fill_a(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 7 + 3) % 9) as f32 - 4.0) * 0.2)
            .collect()
    }

    /// Per-step input `b`, **small integers** in [−3, 3] (exactly representable), the "B·x" drive term.
    fn fill_b(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 3 + 1) % 7) as f32 - 3.0).collect()
    }

    /// The independent reference: the literal scalar recurrence, `mul` then `add`, per row — exactly the
    /// arithmetic `lrscan_row` / `lrscan_rows` perform, so a correct kernel matches it **bit-for-bit**.
    fn lrscan_naive(a: &[f32], b: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let mut h = 0.0f32;
            for t in 0..cols {
                h = a[r * cols + t] * h + b[r * cols + t];
                out[r * cols + t] = h;
            }
        }
        out
    }

    /// Serial kernel == naive per-row recurrence, **bit-for-bit**, across shapes straddling the small-`cols`
    /// edges and a spread of row counts that exercise the 4-row interleave (8, 64, 200) **and** its `< 4`
    /// tail (1, 2, 3). No reassociation — each row is the same scalar expression tree, and the interleave
    /// changes only issue order — so equality is exact, not toleranced.
    #[test]
    fn lrscan_matches_naive() {
        for &rows in &[1usize, 2, 3, 8, 64, 200] {
            for &cols in &[1usize, 2, 7, 8, 9, 16, 17, 33, 64, 100, 1000] {
                let a = fill_a(rows * cols);
                let b = fill_b(rows * cols);
                let mut got = vec![0.0f32; rows * cols];
                unsafe {
                    mercury_lrscan_f32(
                        a.as_ptr(),
                        b.as_ptr(),
                        got.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                    );
                }
                let want = lrscan_naive(&a, &b, rows, cols);
                assert_eq!(got, want, "lrscan {rows}x{cols} vs naive");
            }
        }
    }

    /// Serial and parallel must be **bit-for-bit identical** (rows independent, no cross-row combine — the
    /// parallel path runs the same 4-row-interleaved scan over disjoint row chunks). Exercised above
    /// `LRSCAN_PAR_MIN` so the rayon path actually runs, across varied `cols` (short rows included).
    #[test]
    fn serial_equals_parallel_bit_for_bit() {
        let rows = LRSCAN_PAR_MIN + 137; // > threshold so the multicore path runs
        for &cols in &[1usize, 7, 8, 17, 64, 333] {
            let a = fill_a(rows * cols);
            let b = fill_b(rows * cols);
            let mut s = vec![0.0f32; rows * cols];
            let mut p = vec![0.0f32; rows * cols];
            unsafe {
                mercury_lrscan_f32(a.as_ptr(), b.as_ptr(), s.as_mut_ptr(), rows as i64, cols as i64);
                mercury_lrscan_f32_parallel(
                    a.as_ptr(),
                    b.as_ptr(),
                    p.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                );
            }
            assert_eq!(s, p, "serial != parallel {rows}x{cols}");
        }
    }

    /// Hand-computed cases pinning the recurrence semantics on exactly-representable values: a constant
    /// decay (EMA-style) and `a = 0` rows that prove the state both **resets per row** and that `h_{t-1}`
    /// is genuinely multiplied in (a zero gate erases all history). Eight rows exercises a full 4-row block.
    #[test]
    fn lrscan_known_values() {
        // Single row, a ≡ 0.5, b ≡ 1:  h = 1, 1.5, 1.75, 1.875  (all exact binary fractions).
        let a = [0.5f32, 0.5, 0.5, 0.5];
        let b = [1.0f32, 1.0, 1.0, 1.0];
        let mut got = [0.0f32; 4];
        unsafe {
            mercury_lrscan_f32(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), 1, 4);
        }
        assert_eq!(got, [1.0f32, 1.5, 1.75, 1.875]);

        // Eight rows, cols = 2 (a full 4-row block twice). Even rows a≡0.5,b≡2 → [2,3]; odd rows a≡0,
        // b=[3,4] → [3,4] (the a=0 rows: h resets to 0 and 0·h kills the carry, so out == b exactly).
        let mut a2 = [0.0f32; 16];
        let mut b2 = [0.0f32; 16];
        let mut want = [0.0f32; 16];
        for r in 0..8 {
            let o = r * 2;
            if r % 2 == 0 {
                a2[o] = 0.5;
                a2[o + 1] = 0.5;
                b2[o] = 2.0;
                b2[o + 1] = 2.0;
                want[o] = 2.0;
                want[o + 1] = 3.0;
            } else {
                b2[o] = 3.0;
                b2[o + 1] = 4.0;
                want[o] = 3.0;
                want[o + 1] = 4.0;
            }
        }
        let mut got2 = [0.0f32; 16];
        unsafe {
            mercury_lrscan_f32(a2.as_ptr(), b2.as_ptr(), got2.as_mut_ptr(), 8, 2);
        }
        assert_eq!(got2, want);
    }
}
