//! Cache-blocked matrix transpose: `dst[j, i] = src[i, j]` (a `[rows, cols]` row-major matrix into a
//! `[cols, rows]` row-major one). Pure data movement — no arithmetic — so the result is a permutation
//! of the input, **bit-identical to the naive nested-loop transpose on every backend** (the
//! differential gate is trivial: there is no float reassociation to keep in sync).
//!
//! The win over a naive C/Rust transpose is **cache blocking**: the naive `for i { for j { dst[j*r+i]
//! = src[i*c+j] } }` writes `dst` with stride `r` — a fresh cache line per element once `r` is large,
//! so the working set thrashes — while the blocked form keeps a `B×B` tile of *both* `src` and `dst`
//! L1-resident, turning the strided stream into B sequential runs. gcc/rustc do **not** tile a
//! transpose at `-O3` (loop tiling is a polyhedral pass outside `-O3`), so this is a real algorithmic
//! win on the memory-bound op, independent of SIMD width.

/// Block edge: a `B×B` f32 tile is `B²·4` bytes; `B = 32` → 4 KB per tile, so both the `src` and `dst`
/// tiles sit comfortably in a 32 KB L1 together.
const B: usize = 32;

/// Transpose the row range `[i0, i1)` of a `[rows, cols]` row-major `src` into a `[cols, rows]`
/// row-major `dst` (`dst[j*rows + i] = src[i*cols + j]`), cache-blocked. Shared by the serial and
/// parallel kernels; each call writes only the `dst` entries with `i ∈ [i0, i1)` (`dst[*, i]`), which
/// are disjoint across row ranges → the parallel split is data-race-free and order-independent.
///
/// Generic over the element type `T` (any `Copy` value moves the same way — transpose is pure data
/// movement), so one core serves the f32 and the 16-bit (`bf16`/`f16`, stored as `u16`) entry points.
///
/// # Safety
/// `src` valid for `rows*cols`, `dst` for `cols*rows` `T`; `i0 <= i1 <= rows`.
#[inline]
unsafe fn transpose_rows<T: Copy>(
    src: *const T,
    dst: *mut T,
    rows: usize,
    cols: usize,
    i0: usize,
    i1: usize,
) {
    let mut ib = i0;
    while ib < i1 {
        let imax = (ib + B).min(i1);
        let mut jb = 0;
        while jb < cols {
            let jmax = (jb + B).min(cols);
            // One B×B tile: read `src` contiguously along each row, scatter into the `dst` column —
            // both tiles stay L1-resident, so the only strided traffic is within a cached block.
            for i in ib..imax {
                let srow = src.add(i * cols);
                for j in jb..jmax {
                    *dst.add(j * rows + i) = *srow.add(j);
                }
            }
            jb += B;
        }
        ib += B;
    }
}

/// Element count below which the parallel transpose just runs serially (the rayon split is not worth
/// its overhead on a small, already-cache-resident matrix).
const TRANSPOSE_PAR_MIN: usize = 1 << 16;

/// Multi-threaded `dst = srcᵀ` (generic over `T`): the independent `B`-row blocks are spread across
/// cores. Each block writes a disjoint set of `dst` entries (`dst[*, i]` for `i` in the block), so the
/// result is **bit-identical to the serial transpose** the interpreter marshals as the oracle — there
/// is no accumulation, just a permutation, so thread count and order are irrelevant. Below
/// `TRANSPOSE_PAR_MIN` it runs the serial path.
///
/// # Safety
/// `src` valid for `rows*cols`, `dst` for `cols*rows` `T`, non-overlapping.
#[inline]
unsafe fn transpose_parallel<T: Copy + Send + Sync>(
    src: *const T,
    dst: *mut T,
    rows: i64,
    cols: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r * c < TRANSPOSE_PAR_MIN {
        transpose_rows(src, dst, r, c, 0, r);
        return;
    }
    use rayon::prelude::*;
    let nblocks = r.div_ceil(B);
    let (src_addr, dst_addr) = (src as usize, dst as usize);
    (0..nblocks).into_par_iter().for_each(|blk| {
        let i0 = blk * B;
        let i1 = (i0 + B).min(r);
        // SAFETY: disjoint dst-row range per block (dst[*, i] for i in [i0, i1)); pointers re-derived
        // per task from the captured addresses (so they are Send).
        unsafe {
            transpose_rows(src_addr as *const T, dst_addr as *mut T, r, c, i0, i1);
        }
    });
}

/// `dst = srcᵀ` — transpose a `[rows, cols]` row-major f32 matrix into a `[cols, rows]` row-major one,
/// cache-blocked, single-threaded.
///
/// # Safety
/// `src` valid for `rows*cols`, `dst` for `cols*rows` f32, non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn wukong_transpose_f32(src: *const f32, dst: *mut f32, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    transpose_rows(src, dst, rows as usize, cols as usize, 0, rows as usize);
}

/// Multi-threaded f32 `dst = srcᵀ`. See [`transpose_parallel`].
///
/// # Safety
/// Operand-size contract of [`wukong_transpose_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_transpose_f32_parallel(
    src: *const f32,
    dst: *mut f32,
    rows: i64,
    cols: i64,
) {
    transpose_parallel(src, dst, rows, cols);
}

/// `dst = srcᵀ` for 16-bit elements (`bf16`/`f16`, stored as `u16`) — the same cache-blocked transpose
/// as f32, half the bytes per element (so ~2× the elements per cache line). A transpose moves the raw
/// bits, so it is precision-agnostic: one `u16` kernel serves both bf16 and f16.
///
/// # Safety
/// `src` valid for `rows*cols`, `dst` for `cols*rows` `u16`, non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn wukong_transpose_u16(src: *const u16, dst: *mut u16, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    transpose_rows(src, dst, rows as usize, cols as usize, 0, rows as usize);
}

/// Multi-threaded 16-bit `dst = srcᵀ`. See [`transpose_parallel`].
///
/// # Safety
/// Operand-size contract of [`wukong_transpose_u16`].
#[no_mangle]
pub unsafe extern "C" fn wukong_transpose_u16_parallel(
    src: *const u16,
    dst: *mut u16,
    rows: i64,
    cols: i64,
) {
    transpose_parallel(src, dst, rows, cols);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_transpose(src: &[f32], r: usize, c: usize) -> Vec<f32> {
        let mut dst = vec![0.0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                dst[j * r + i] = src[i * c + j];
            }
        }
        dst
    }

    /// The blocked transpose (serial and parallel) must equal the naive transpose exactly — it is a
    /// permutation of the inputs, so "exactly" is literal bit equality, no tolerance. Shapes straddle
    /// the B=32 block edge (non-multiples, thin, square, and big enough to trip the parallel path).
    #[test]
    fn transpose_matches_naive_and_parallel() {
        for (r, c) in [
            (1, 1),
            (3, 5),
            (8, 8),
            (32, 32),
            (33, 31),
            (100, 70),
            (1, 257),
            (257, 1),
            (300, 400), // > TRANSPOSE_PAR_MIN so the multicore path runs
        ] {
            let src: Vec<f32> = (0..r * c).map(|i| i as f32 * 0.5 - 3.0).collect();
            let want = naive_transpose(&src, r, c);
            let mut got = vec![0.0f32; r * c];
            let mut got_par = vec![0.0f32; r * c];
            unsafe {
                wukong_transpose_f32(src.as_ptr(), got.as_mut_ptr(), r as i64, c as i64);
                wukong_transpose_f32_parallel(src.as_ptr(), got_par.as_mut_ptr(), r as i64, c as i64);
            }
            assert_eq!(got, want, "transpose {r}x{c} vs naive");
            assert_eq!(got, got_par, "transpose serial vs parallel {r}x{c}");
        }
    }

    /// The 16-bit (`bf16`/`f16`) transpose must equal the naive transpose exactly too — it moves the
    /// raw `u16` bits, a permutation, so bit equality holds for any stored values. Same shapes.
    #[test]
    fn transpose_u16_matches_naive_and_parallel() {
        for (r, c) in [
            (1, 1),
            (3, 5),
            (8, 8),
            (32, 32),
            (33, 31),
            (100, 70),
            (1, 257),
            (257, 1),
            (300, 400), // > TRANSPOSE_PAR_MIN so the multicore path runs
        ] {
            let src: Vec<u16> = (0..r * c).map(|i| (i as u16).wrapping_mul(2719)).collect();
            let mut want = vec![0u16; r * c];
            for i in 0..r {
                for j in 0..c {
                    want[j * r + i] = src[i * c + j];
                }
            }
            let mut got = vec![0u16; r * c];
            let mut got_par = vec![0u16; r * c];
            unsafe {
                wukong_transpose_u16(src.as_ptr(), got.as_mut_ptr(), r as i64, c as i64);
                wukong_transpose_u16_parallel(src.as_ptr(), got_par.as_mut_ptr(), r as i64, c as i64);
            }
            assert_eq!(got, want, "u16 transpose {r}x{c} vs naive");
            assert_eq!(got, got_par, "u16 transpose serial vs parallel {r}x{c}");
        }
    }
}
