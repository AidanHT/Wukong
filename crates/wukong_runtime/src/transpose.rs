//! Cache-blocked matrix transpose: `dst[j, i] = src[i, j]` (a `[rows, cols]` row-major matrix into a
//! `[cols, rows]` row-major one). Pure data movement — no arithmetic — so the result is a permutation
//! of the input, **bit-identical to the naive nested-loop transpose on every backend** (the
//! differential gate is trivial: there is no float reassociation to keep in sync).
//!
//! The first lever is **cache blocking**: the naive `for i { for j { dst[j*r+i] = src[i*c+j] } }`
//! writes `dst` with stride `r` — a fresh cache line per element once `r` is large, so the working set
//! thrashes — while the blocked form keeps a `B×B` tile of *both* `src` and `dst` L1-resident, turning
//! the strided stream into B sequential runs. gcc/rustc do **not** tile a transpose at `-O3` (loop
//! tiling is a polyhedral pass outside `-O3`), so it is a real algorithmic win over a naive peer.
//!
//! Blocking alone is not an edge over a *competent* peer, though: a C programmer writing a transpose
//! by hand blocks it too, and against a 32-blocked `__restrict__` C peer this kernel was a tie. The
//! f32 path therefore now also carries an **8×8 in-register AVX2 kernel**
//! ([`transpose_rows_f32_avx2`]) inside the same `B`-square block, which turns 64 scalar moves per
//! tile into 8 vector loads / 24 shuffles / 8 vector stores *and* — the larger effect — drops the
//! number of simultaneously-live `dst` rows from 32 to 8. On a power-of-two `rows` the `dst` rows are
//! `rows*4` bytes apart and all land in the same L1 set, so a 32-row block needs 32 ways of a 12-way
//! L1 and collapses; eight fit. That is why the scalar-blocked form and the identically-blocked C
//! peer both run an order of magnitude below their non-power-of-two rate at 1024²/2048², and why the
//! 8×8 kernel is 5–7× faster there and ~1.4× at sizes where nobody thrashes.
//!
//! The 16-bit (`bf16`/`f16`) entries still use the generic scalar `transpose_rows<T: Copy>` core,
//! which also serves as the f32 fallback on a host without AVX2. Both produce the identical `dst`,
//! since a transpose is a permutation — it never inspects, compares or combines a value, so every
//! path here is bit-identical **by construction**, for any bit pattern including NaN payloads.
//!
//! Four C entries: `wukong_transpose_f32[_parallel]` and `wukong_transpose_u16[_parallel]`. A
//! non-positive `rows` or `cols` is a no-op. The `_parallel` forms spread the independent `B`-row
//! blocks across cores, each owning a disjoint set of `dst` entries, so they are bit-identical to the
//! serial form and run serial below `TRANSPOSE_PAR_MIN` elements.

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

/// f32 transpose of the row range `[i0, i1)` with an **8x8 in-register AVX2 kernel** inside the same
/// `B`-square cache block as [`transpose_rows`]. Eight `_mm256_loadu_ps` read eight source rows; three
/// stages of `unpacklo/unpackhi`, `shuffle` and `permute2f128` transpose them in registers; eight
/// `_mm256_storeu_ps` write eight destination rows. Pure data movement — the eight results are exactly
/// the same 64 floats [`transpose_rows`] would move one at a time, so this is **bit-identical by
/// construction**, for any bit pattern including NaN payloads (nothing is compared or arithmetically
/// combined).
///
/// TWO THINGS ARE FIXED HERE, and the second is the bigger one:
///  * 64 scalar loads + 64 scalar stores per 8x8 tile become 8 vector loads + 8 vector stores.
///  * The tile only has **eight** `dst` rows live at a time instead of `B` = 32. Consecutive `dst`
///    rows are `rows*4` bytes apart, so on a power-of-two `rows` they all collide in the same L1 set:
///    at `rows = 1024` the stride is exactly 4096 B, and a 32-row block needs 32 ways of a 12-way L1.
///    That is why the 32-block scalar transpose (and the identically-blocked C peer, which also
///    collapses) runs an order of magnitude below its non-power-of-two rate; eight rows fit.
///
/// Measured against a bounds-correct 32-blocked `__restrict__` C peer at gcc `-O3 -march=native`
/// (own TU, shared library, ABBA, best-of-N, pinned to one P-core): 1024² **7.2x**, 1536² **5.9x**,
/// 2048² **5.3x**; at the non-colliding 1000² / 1031² where nobody thrashes, **1.42x**.
///
/// The `rows % 8` trailing rows and `cols % 8` trailing columns of each block fall back to the same
/// scalar moves [`transpose_rows`] uses.
///
/// # Safety
/// `src` valid for `rows*cols`, `dst` for `cols*rows` f32; `i0 <= i1 <= rows`; AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_rows_f32_avx2(
    src: *const f32,
    dst: *mut f32,
    rows: usize,
    cols: usize,
    i0: usize,
    i1: usize,
) {
    use std::arch::x86_64::*;
    let r8 = i1 & !7; // last row index the 8-row strip may start at
    let c8 = cols & !7;
    let mut ib = i0;
    while ib < i1 {
        let imax = (ib + B).min(i1);
        let mut jb = 0;
        while jb < cols {
            let jmax = (jb + B).min(cols);
            let i8max = imax.min(r8);
            let j8max = jmax.min(c8);
            let mut i = ib;
            while i + 8 <= i8max {
                let mut j = jb;
                while j + 8 <= j8max {
                    let s = src.add(i * cols + j);
                    let r0 = _mm256_loadu_ps(s);
                    let r1 = _mm256_loadu_ps(s.add(cols));
                    let r2 = _mm256_loadu_ps(s.add(2 * cols));
                    let r3 = _mm256_loadu_ps(s.add(3 * cols));
                    let r4 = _mm256_loadu_ps(s.add(4 * cols));
                    let r5 = _mm256_loadu_ps(s.add(5 * cols));
                    let r6 = _mm256_loadu_ps(s.add(6 * cols));
                    let r7 = _mm256_loadu_ps(s.add(7 * cols));
                    // Stage 1: interleave adjacent rows within each 128-bit half.
                    let t0 = _mm256_unpacklo_ps(r0, r1);
                    let t1 = _mm256_unpackhi_ps(r0, r1);
                    let t2 = _mm256_unpacklo_ps(r2, r3);
                    let t3 = _mm256_unpackhi_ps(r2, r3);
                    let t4 = _mm256_unpacklo_ps(r4, r5);
                    let t5 = _mm256_unpackhi_ps(r4, r5);
                    let t6 = _mm256_unpacklo_ps(r6, r7);
                    let t7 = _mm256_unpackhi_ps(r6, r7);
                    // Stage 2: gather 64-bit pairs into 4x4 transposed halves.
                    let u0 = _mm256_shuffle_ps(t0, t2, 0b01_00_01_00);
                    let u1 = _mm256_shuffle_ps(t0, t2, 0b11_10_11_10);
                    let u2 = _mm256_shuffle_ps(t1, t3, 0b01_00_01_00);
                    let u3 = _mm256_shuffle_ps(t1, t3, 0b11_10_11_10);
                    let u4 = _mm256_shuffle_ps(t4, t6, 0b01_00_01_00);
                    let u5 = _mm256_shuffle_ps(t4, t6, 0b11_10_11_10);
                    let u6 = _mm256_shuffle_ps(t5, t7, 0b01_00_01_00);
                    let u7 = _mm256_shuffle_ps(t5, t7, 0b11_10_11_10);
                    // Stage 3: swap the 128-bit lanes to finish the 8x8.
                    let d = dst.add(j * rows + i);
                    _mm256_storeu_ps(d, _mm256_permute2f128_ps(u0, u4, 0x20));
                    _mm256_storeu_ps(d.add(rows), _mm256_permute2f128_ps(u1, u5, 0x20));
                    _mm256_storeu_ps(d.add(2 * rows), _mm256_permute2f128_ps(u2, u6, 0x20));
                    _mm256_storeu_ps(d.add(3 * rows), _mm256_permute2f128_ps(u3, u7, 0x20));
                    _mm256_storeu_ps(d.add(4 * rows), _mm256_permute2f128_ps(u0, u4, 0x31));
                    _mm256_storeu_ps(d.add(5 * rows), _mm256_permute2f128_ps(u1, u5, 0x31));
                    _mm256_storeu_ps(d.add(6 * rows), _mm256_permute2f128_ps(u2, u6, 0x31));
                    _mm256_storeu_ps(d.add(7 * rows), _mm256_permute2f128_ps(u3, u7, 0x31));
                    j += 8;
                }
                // cols % 8 trailing columns of this 8-row strip.
                while j < jmax {
                    for ii in i..i + 8 {
                        *dst.add(j * rows + ii) = *src.add(ii * cols + j);
                    }
                    j += 1;
                }
                i += 8;
            }
            // rows % 8 trailing rows of this block.
            while i < imax {
                let srow = src.add(i * cols);
                for j in jb..jmax {
                    *dst.add(j * rows + i) = *srow.add(j);
                }
                i += 1;
            }
            jb += B;
        }
        ib += B;
    }
}

/// f32 row-range transpose: the AVX2 8x8 kernel where available, the generic scalar block otherwise.
/// The two produce identical `dst` bytes (both are the same permutation), so the choice is invisible
/// to every caller and to the differential gate.
///
/// # Safety
/// `src` valid for `rows*cols`, `dst` for `cols*rows` f32; `i0 <= i1 <= rows`.
#[inline]
unsafe fn transpose_rows_f32(
    src: *const f32,
    dst: *mut f32,
    rows: usize,
    cols: usize,
    i0: usize,
    i1: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            transpose_rows_f32_avx2(src, dst, rows, cols, i0, i1);
            return;
        }
    }
    transpose_rows(src, dst, rows, cols, i0, i1);
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
    // This fork can be the process's FIRST rayon touch, so it must provision the global pool first —
    // [`crate::ensure_global_pool`]'s stated precondition. Forking bare builds rayon's default
    // 2 MiB-stack registry, so the runtime's later 16 MiB `build_global` silently loses the race and
    // outlined `@parallel` region bodies are left on undersized stacks. Idempotent (`Once`) and
    // provisioning-only: the work split below is unchanged, so serial == parallel stays bit-exact.
    crate::ensure_global_pool();
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
    transpose_rows_f32(src, dst, rows as usize, cols as usize, 0, rows as usize);
}

/// Multi-threaded f32 `dst = srcᵀ` — the f32 twin of [`transpose_parallel`], splitting the same
/// independent `B`-row blocks across cores but running [`transpose_rows_f32`] (the AVX2 8x8 kernel)
/// on each. Each block writes a disjoint set of `dst` entries, and a transpose is a permutation with
/// no accumulation, so it is bit-identical to the serial form regardless of thread count. Below
/// [`TRANSPOSE_PAR_MIN`] elements it runs serial. The row blocks are multiples of `B` = 32, hence of
/// 8, so every task's range starts on an 8-row strip boundary.
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
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r * c < TRANSPOSE_PAR_MIN {
        transpose_rows_f32(src, dst, r, c, 0, r);
        return;
    }
    use rayon::prelude::*;
    // Same first-rayon-touch precondition as [`transpose_parallel`]: provision the global pool before
    // forking so the runtime's 16 MiB `build_global` is not beaten by rayon's lazy 2 MiB default.
    crate::ensure_global_pool();
    let nblocks = r.div_ceil(B);
    let (src_addr, dst_addr) = (src as usize, dst as usize);
    (0..nblocks).into_par_iter().for_each(|blk| {
        let i0 = blk * B;
        let i1 = (i0 + B).min(r);
        // SAFETY: disjoint dst-row range per block (dst[*, i] for i in [i0, i1)); pointers re-derived
        // per task from the captured addresses (so they are Send).
        unsafe {
            transpose_rows_f32(src_addr as *const f32, dst_addr as *mut f32, r, c, i0, i1);
        }
    });
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
    /// the B=32 block edge AND the 8x8 AVX2 tile edge (non-multiples, thin, square, and big enough to
    /// trip the parallel path).
    #[test]
    fn transpose_matches_naive_and_parallel() {
        for (r, c) in [
            (1, 1),
            (3, 5),
            (7, 7), // below one 8x8 tile: the whole block is the scalar tail
            (8, 8),
            (9, 9),   // one tile + a 1-row / 1-column tail
            (8, 33),  // exactly one 8-row strip, columns past a 32-block
            (33, 8),  // rows past a 32-block, exactly one 8-column tile
            (15, 17), // both tails, both under a block
            (32, 32),
            (33, 31),
            (40, 40), // one full block + one 8-row strip
            (100, 70),
            (1, 257),
            (257, 1),
            (300, 400), // > TRANSPOSE_PAR_MIN so the multicore path runs
            (128, 128), // power-of-two stride: the L1-set-collision shape the 8x8 tile exists for
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

    /// The AVX2 8x8 kernel must move **the same bits** as the generic scalar block for every shape,
    /// and for every bit pattern — a transpose never inspects a value, so this must hold for NaN
    /// payloads, signalling NaNs and ±0 exactly as for ordinary floats. The source is therefore built
    /// from RAW BIT PATTERNS (`f32::from_bits` over a sweep that includes both zeros, both infinities
    /// and several NaN payloads) and compared by bits — `assert_eq!` on `f32` would let a NaN column
    /// pass vacuously, and would call `+0.0` and `-0.0` equal.
    ///
    /// The row ranges exercised also cover a partial `[i0, i1)` window, which is what the parallel
    /// entry hands each task.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_tile_moves_the_same_bits_as_the_scalar_block() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        for (r, c) in [
            (1usize, 1usize),
            (7, 7),
            (8, 8),
            (9, 15),
            (16, 24),
            (17, 33),
            (32, 32),
            (33, 41),
            (64, 64),
            (70, 100),
            (128, 128),
        ] {
            // Raw bit patterns, cycling through the awkward ones. A transpose must not care.
            let pool: [u32; 10] = [
                0x0000_0000, // +0.0
                0x8000_0000, // -0.0
                0x7f80_0000, // +inf
                0xff80_0000, // -inf
                0x7fc0_0000, // quiet NaN
                0x7fc0_1234, // quiet NaN, other payload
                0x7f80_0001, // signalling NaN
                0x0000_0001, // smallest subnormal
                0x3f80_0000, // 1.0
                0xbf80_0000, // -1.0
            ];
            let src: Vec<f32> = (0..r * c)
                .map(|t| {
                    if t % 3 == 0 {
                        f32::from_bits(pool[t % pool.len()])
                    } else {
                        (t as f32) * 0.25 - 3.0
                    }
                })
                .collect();
            let mut s = vec![0.0f32; r * c];
            let mut v = vec![0.0f32; r * c];
            unsafe {
                transpose_rows(src.as_ptr(), s.as_mut_ptr(), r, c, 0, r);
                transpose_rows_f32_avx2(src.as_ptr(), v.as_mut_ptr(), r, c, 0, r);
            }
            assert_eq!(bits(&s), bits(&v), "avx2 != scalar block at {r}x{c}");
            // Split into two row windows the way the parallel entry does, and demand the same bits.
            if r > B {
                let mut w = vec![0.0f32; r * c];
                unsafe {
                    transpose_rows_f32_avx2(src.as_ptr(), w.as_mut_ptr(), r, c, 0, B);
                    transpose_rows_f32_avx2(src.as_ptr(), w.as_mut_ptr(), r, c, B, r);
                }
                assert_eq!(bits(&s), bits(&w), "avx2 windowed != whole at {r}x{c}");
            }
        }
    }
}
