//! Inclusive per-row **prefix product** (cumulative product / scan) over the last axis of a
//! `[rows, cols]` row-major f32 matrix: `out[r, i] = Π_{k=0..=i} x[r, k]` for each row `r` and each `i`
//! in `0..cols`. The cumulative-product primitive behind autoregressive normalizing-flow Jacobians,
//! cumulative gates (a chain of forget/decay factors), survival/hazard products, and running products.
//!
//! **Why this is a win.** A prefix product is a *loop-carried dependency* — `out[i] = out[i-1] * x[i]` —
//! and gcc `-O3 -march=native` / rustc keep it **scalar** (one serial `mulss` chain per row; the
//! recurrence defeats their auto-vectorizer, exactly like the prefix sum in `cumsum.rs`). That chain is
//! *latency-bound* (each step waits on the previous multiply). The lever is **instruction-level
//! parallelism across independent rows** (`cumprod_rows`): the rows are independent, so processing **4
//! rows at once** — interleaving their four running products down a shared `i` — keeps four independent
//! `mul` chains in flight and fills the ports the single chain leaves idle. gcc/rustc may not legally
//! re-order an f32 product across rows without `-ffast-math`, so this 4-way ILP is a genuine single-core
//! win, and the `_parallel` map of independent row *chunks* across cores stacks on top.
//!
//! **Bit-exactness contract.** Unlike the prefix *sum*, there is **no reassociation here** — each row's
//! product folds strictly left-to-right (`p *= x[i]` in ascending `i`), and the 4-row interleave changes
//! only *issue order*, never a row's fold order. A scalar `mul` is not fused into anything (no `fma`
//! contraction applies to a bare product), so the kernel is **bit-identical** to the naive scalar nest —
//! `assert_eq!` on the bits, not a tolerance — and to the strict left-to-right reference gcc/rustc run,
//! so even the cross-language check is exact. Serial == parallel bit-for-bit (rows independent, no
//! cross-row combine); the interpreter marshals the serial form, so the differential oracle is exact.

use rayon::prelude::*;

/// Inclusive prefix product of one row in place: `p ← p * x[i]; out[i] ← p` for `i = 0..cols`, with `p`
/// starting at `1.0` (the multiplicative identity) carried in a register. The tail helper (rows the
/// 4-wide block doesn't cover) and the building block the interleaved scan reduces to.
///
/// # Safety
/// `x` valid for `cols` f32, `out` valid for `cols` f32; `out` may alias `x` (each `x[i]` is read before
/// `out[i]` is written and later steps read only higher indices, so the scan is in-place-safe).
#[inline]
unsafe fn cumprod_row(x: *const f32, out: *mut f32, cols: usize) {
    let mut p = 1.0f32;
    for i in 0..cols {
        p *= *x.add(i);
        *out.add(i) = p;
    }
}

/// Scan rows `[r0, r1)` of a `[_, cols]` matrix, **4 independent rows interleaved** so four `mul` chains
/// are in flight at once — the single-core lever (the prefix product is latency-bound on the per-row
/// carry, so 4-way ILP across independent rows fills the ports a single serial chain leaves idle). Each
/// row remains its own strictly-left-to-right product, so the result is **bit-identical** to the 1-row
/// [`cumprod_row`] — serial == parallel == interpreter. The `< 4` leftover rows fall to `cumprod_row`.
///
/// # Safety
/// `x`, `out` valid for at least `r1 * cols` f32; rows `[r0, r1)` lie within that range.
#[inline]
unsafe fn cumprod_rows(x: *const f32, out: *mut f32, cols: usize, r0: usize, r1: usize) {
    let mut r = r0;
    while r + 4 <= r1 {
        let (o0, o1, o2, o3) = (r * cols, (r + 1) * cols, (r + 2) * cols, (r + 3) * cols);
        let (mut p0, mut p1, mut p2, mut p3) = (1.0f32, 1.0f32, 1.0f32, 1.0f32);
        for i in 0..cols {
            p0 *= *x.add(o0 + i);
            p1 *= *x.add(o1 + i);
            p2 *= *x.add(o2 + i);
            p3 *= *x.add(o3 + i);
            *out.add(o0 + i) = p0;
            *out.add(o1 + i) = p1;
            *out.add(o2 + i) = p2;
            *out.add(o3 + i) = p3;
        }
        r += 4;
    }
    while r < r1 {
        let off = r * cols;
        cumprod_row(x.add(off), out.add(off), cols);
        r += 1;
    }
}

/// Row count below which the parallel scan just runs serially.
const CUMPROD_PAR_MIN: usize = 64;

/// `out[r, i] = Π_{k<=i} x[r, k]` over a `[rows, cols]` row-major matrix — single-threaded
/// (4-row-interleaved for ILP). Each row is an independent inclusive prefix product.
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` f32; `out` may alias `x` (in-place-safe per row).
#[no_mangle]
pub unsafe extern "C" fn mercury_cumprod_f32(x: *const f32, out: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    cumprod_rows(x, out, cols, 0, rows);
}

/// Multicore prefix product — **bit-identical** to [`mercury_cumprod_f32`]. Rows are independent, so the
/// parallel path slices the rows into one contiguous chunk per core and runs the *identical*
/// 4-row-interleaved [`cumprod_rows`] on each — independent of thread count, bit-equal to serial. Below
/// `CUMPROD_PAR_MIN` rows it runs serial.
///
/// # Safety
/// `x`, `out` valid for `rows * cols` f32; each row written by exactly one task (disjoint sub-slices).
#[no_mangle]
pub unsafe extern "C" fn mercury_cumprod_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    if rows < CUMPROD_PAR_MIN {
        cumprod_rows(x, out, cols, 0, rows);
        return;
    }
    let (xa, oa) = (x as usize, out as usize);
    let nthreads = rayon::current_num_threads().max(1);
    let per = rows.div_ceil(nthreads).max(1);
    let nchunks = rows.div_ceil(per);
    (0..nchunks).into_par_iter().for_each(|c| {
        let r0 = c * per;
        let r1 = (r0 + per).min(rows);
        // SAFETY: rows [r0, r1) disjoint across chunks; pointers re-derived from captured addresses.
        unsafe { cumprod_rows(xa as *const f32, oa as *mut f32, cols, r0, r1) };
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inputs near 1 (in {0.5, 0.75, 1.0, 1.25, 1.5}) so the running product stays finite and modestly
    /// sized over the longest test row (no over/underflow), keeping every intermediate an ordinary finite
    /// f32 — `assert_eq!` on the bits is then meaningful.
    fn fill(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 + 0.25 * ((i * 5 + 2) % 5) as f32 * 0.5)
            .collect()
    }

    /// Naive scalar prefix product — the bit-exact reference (strict left-to-right, `p` from 1.0), exactly
    /// the arithmetic `cumprod_row` / `cumprod_rows` perform.
    fn naive(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let mut p = 1.0f32;
            for i in 0..cols {
                p *= x[r * cols + i];
                out[r * cols + i] = p;
            }
        }
        out
    }

    /// Serial kernel == naive left-to-right prefix product, **bit-for-bit**, across shapes that exercise
    /// the 4-row interleave (8, 64, 200) and its `< 4` tail (1, 2, 3). No reassociation (a product is not
    /// fused), so equality is exact.
    #[test]
    fn cumprod_matches_naive() {
        for &rows in &[1usize, 2, 3, 8, 64, 200] {
            for &cols in &[1usize, 2, 7, 8, 9, 16, 17, 33, 64, 100, 1000] {
                let x = fill(rows * cols);
                let mut got = vec![0.0f32; rows * cols];
                unsafe {
                    mercury_cumprod_f32(x.as_ptr(), got.as_mut_ptr(), rows as i64, cols as i64);
                }
                let want = naive(&x, rows, cols);
                assert_eq!(got, want, "cumprod {rows}x{cols} vs naive");
            }
        }
    }

    /// Serial and parallel must be **bit-for-bit identical** (rows independent). Exercised above
    /// `CUMPROD_PAR_MIN` so the rayon path runs, across varied `cols`.
    #[test]
    fn serial_equals_parallel_bit_for_bit() {
        let rows = CUMPROD_PAR_MIN + 137;
        for &cols in &[1usize, 7, 8, 17, 64, 333] {
            let x = fill(rows * cols);
            let mut s = vec![0.0f32; rows * cols];
            let mut p = vec![0.0f32; rows * cols];
            unsafe {
                mercury_cumprod_f32(x.as_ptr(), s.as_mut_ptr(), rows as i64, cols as i64);
                mercury_cumprod_f32_parallel(x.as_ptr(), p.as_mut_ptr(), rows as i64, cols as i64);
            }
            assert_eq!(s, p, "serial != parallel {rows}x{cols}");
        }
    }

    /// Hand-computed case on exactly-representable values: a row of halves (running powers of 1/2) and a
    /// row with a 0 that zeroes every later prefix (proves the carry is genuinely multiplied through).
    #[test]
    fn cumprod_known_values() {
        // Row 0: [2, 0.5, 4, 0.25] -> prefixes [2, 1, 4, 1]; Row 1: [3, 0, 5, 7] -> [3, 0, 0, 0].
        let x = [2.0f32, 0.5, 4.0, 0.25, 3.0, 0.0, 5.0, 7.0];
        let mut got = [0.0f32; 8];
        unsafe {
            mercury_cumprod_f32(x.as_ptr(), got.as_mut_ptr(), 2, 4);
        }
        assert_eq!(got, [2.0f32, 1.0, 4.0, 1.0, 3.0, 0.0, 0.0, 0.0]);
    }
}
