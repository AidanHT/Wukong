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
//! The 16-bit (`bf16`/`f16`) entries carry the same kind of in-register kernel
//! ([`transpose_rows_u16_avx2`]): a 16-row × 8-column `u16` tile transposed with three in-lane
//! unpack stages on 256-bit registers, which writes each destination row as one 32-byte store and
//! keeps the same **eight** destination rows live per j-tile. The generic scalar
//! `transpose_rows<T: Copy>` core remains the fallback for both widths on a host without AVX2. Every
//! path produces the identical `dst`, since a transpose is a permutation — it never inspects,
//! compares or combines a value, so every path here is bit-identical **by construction**, for any
//! bit pattern including NaN payloads.
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
/// TRAVERSAL INSIDE THE BLOCK: **the 8-column j-tile is the OUTER of the two inner loops and the
/// 8-row i-strip the inner one**, which is the opposite of the obvious nesting and is worth about
/// 1.09-1.26x on its own. The reason is the burst length of destination store MISSES. Every
/// destination line is written by two 32-byte stores (`i` and `i+8`), so what matters is how many
/// *distinct* destination lines are in flight between the two halves:
///
///  * i-strip outer: one pass over the block stores into all `B` = 32 destination rows before `i`
///    advances, so 32 lines are opened back to back, all missing, and only then does the next strip
///    close them. A burst of 32 concurrent store misses against roughly a dozen L1 fill buffers
///    stalls the in-order store queue.
///  * j-tile outer: only the tile's **eight** destination rows are open while all four i-strips of
///    the block run, so the misses arrive in bursts of 8 — inside the fill-buffer count — and every
///    second strip hits a line the previous strip already owns.
///
/// That this is the burst length and not store *adjacency* was checked directly: a 16-wide-in-`i`
/// tile that writes each destination line with two ADJACENT `vmovups` while keeping the i-outer
/// nesting recovers none of it (measured slower than the plain 8x8 at 1031² and 1100x950), while the
/// loop swap alone recovers all of it.
///
/// Prefetching the next j-tile's eight destination rows starts their fill about one tile group
/// (four 8x8 tiles, roughly a hundred cycles of shuffle work) early, and is worth a further ~1.06x.
/// The hint asked for is `_MM_HINT_ET0`, the *write* intent, because these lines are about to be
/// stored to — but this crate is built for baseline x86-64 and only turns on `avx2`, and `PRFCHW`
/// is a separate feature that Haswell (AVX2, no PRFCHW) does not have, so LLVM lowers it to
/// **`prefetcht0`** and the release assembly of this function contains 22 `prefetcht0` and zero
/// `prefetchw`. That is deliberate, not an oversight: emitting `prefetchw` would need its own
/// `prfchw` probe arm (std cannot detect it; `crate::isa` has none) or it would be a `#UD` on
/// Haswell, and the plain read hint measured within noise of the write hint on this part (worth
/// 2-5%, not worth a second dispatch). The write intent is kept in the source so that a future
/// `prfchw`-gated arm is a one-line change.
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
            // Does this block write into the SECOND cache line (byte 64) of a destination row?
            // Only then is that line worth prefetching, and only then is `ib + 16` inside `dst`.
            let pf_two_lines = ib + 16 < imax;
            // The j-tile is the OUTER loop and the i-strip the inner one — see the header: it caps
            // the concurrent destination store misses at eight instead of `B`.
            let mut j = jb;
            while j + 8 <= j8max {
                // Start the read-for-ownership of the NEXT j-tile's eight destination rows now.
                if j + 16 <= j8max {
                    let p = dst.add((j + 8) * rows + ib) as *const i8;
                    let mut r = 0;
                    while r < 8 {
                        let q = p.add(r * rows * 4);
                        _mm_prefetch::<_MM_HINT_ET0>(q);
                        if pf_two_lines {
                            _mm_prefetch::<_MM_HINT_ET0>(q.add(64));
                        }
                        r += 1;
                    }
                }
                let mut i = ib;
                while i + 8 <= i8max {
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
                    i += 8;
                }
                j += 8;
            }
            // cols % 8 trailing columns of this block, over the rows the 8-row strips covered.
            while j < jmax {
                let mut i = ib;
                while i < i8max {
                    *dst.add(j * rows + i) = *src.add(i * cols + j);
                    i += 1;
                }
                j += 1;
            }
            // rows % 8 trailing rows of this block, full width. `i8max < ib` when the whole block
            // sits past the last multiple of 8, so clamp rather than start below `ib`.
            let mut i = i8max.max(ib);
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

/// 16-bit (`bf16`/`f16` bits) transpose of the row range `[i0, i1)` with an in-register AVX2 kernel
/// inside the same `B`-square cache block as [`transpose_rows`] — the `u16` sibling of
/// [`transpose_rows_f32_avx2`], with the same two levers and the same traversal.
///
/// THE TILE is 16 source rows × 8 source columns. Source rows `i + k` and `i + 8 + k` (`k ∈ 0..8`)
/// are loaded as the low and high 128-bit halves of one `ymm` (8 `u16` each); three stages of
/// in-lane `unpack{lo,hi}_epi16` → `_epi32` → `_epi64` then transpose BOTH 8×8 halves at once,
/// because AVX2's unpacks operate per 128-bit lane. Output vector `c` holds destination row `j + c`
/// at columns `i .. i + 16` — 16 contiguous `u16` — so each destination row of the tile is written by
/// ONE 32-byte store: 16 loads, 24 shuffles and 8 stores move 128 elements that the scalar block
/// moves with 128 loads and 128 stores. There is no lane-crossing shuffle anywhere; the cross-half
/// work is done by the loads (`vinserti128` from memory is a load plus an ALU uop, not a shuffle).
///
/// LIVE DESTINATION ROWS stay at eight, as in the f32 kernel: the j-tile is 8 columns wide, so one
/// tile writes 8 destination rows, and on a power-of-two `rows` those rows (`rows·2` bytes apart)
/// all collide in one L1 set exactly as the f32 header describes — a 16-column tile would keep 16
/// live and exceed a 12-way L1. The j-tile is the OUTER loop and the row strip the inner one (same
/// burst-length argument), and the next j-tile's eight destination rows are prefetched with the same
/// `_MM_HINT_ET0` (lowered to `prefetcht0` on this crate's baseline, see the f32 kernel).
///
/// Rows of a block beyond the last 16-row strip take one 8-row strip if at least 8 remain (the same
/// shuffles on the low halves only, stored as 16-byte rows); the `< 8` trailing rows and the
/// `cols % 8` trailing columns fall back to the scalar moves [`transpose_rows`] uses.
///
/// # Safety
/// `src` valid for `rows*cols`, `dst` for `cols*rows` `u16`; `i0 <= i1 <= rows`; AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_rows_u16_avx2(
    src: *const u16,
    dst: *mut u16,
    rows: usize,
    cols: usize,
    i0: usize,
    i1: usize,
) {
    use std::arch::x86_64::*;
    // In-lane 8x8 transpose of eight 16-bit rows, applied to both 128-bit halves at once: input `v[k]`
    // holds row k of each half; output `c[q]` holds column q of each half.
    macro_rules! tr8x8_epi16 {
        ($v:expr) => {{
            let v: [__m256i; 8] = $v;
            let t0 = _mm256_unpacklo_epi16(v[0], v[1]);
            let t1 = _mm256_unpackhi_epi16(v[0], v[1]);
            let t2 = _mm256_unpacklo_epi16(v[2], v[3]);
            let t3 = _mm256_unpackhi_epi16(v[2], v[3]);
            let t4 = _mm256_unpacklo_epi16(v[4], v[5]);
            let t5 = _mm256_unpackhi_epi16(v[4], v[5]);
            let t6 = _mm256_unpacklo_epi16(v[6], v[7]);
            let t7 = _mm256_unpackhi_epi16(v[6], v[7]);
            let u0 = _mm256_unpacklo_epi32(t0, t2);
            let u1 = _mm256_unpackhi_epi32(t0, t2);
            let u2 = _mm256_unpacklo_epi32(t1, t3);
            let u3 = _mm256_unpackhi_epi32(t1, t3);
            let u4 = _mm256_unpacklo_epi32(t4, t6);
            let u5 = _mm256_unpackhi_epi32(t4, t6);
            let u6 = _mm256_unpacklo_epi32(t5, t7);
            let u7 = _mm256_unpackhi_epi32(t5, t7);
            [
                _mm256_unpacklo_epi64(u0, u4),
                _mm256_unpackhi_epi64(u0, u4),
                _mm256_unpacklo_epi64(u1, u5),
                _mm256_unpackhi_epi64(u1, u5),
                _mm256_unpacklo_epi64(u2, u6),
                _mm256_unpackhi_epi64(u2, u6),
                _mm256_unpacklo_epi64(u3, u7),
                _mm256_unpackhi_epi64(u3, u7),
            ]
        }};
    }
    let c8 = cols & !7;
    let mut ib = i0;
    while ib < i1 {
        let imax = (ib + B).min(i1);
        // Rows this block covers with 16-row strips, then with at most one 8-row strip.
        let i16max = ib + (imax - ib) / 16 * 16;
        let i8max = if imax - i16max >= 8 {
            i16max + 8
        } else {
            i16max
        };
        // Does the block reach the second half (32 bytes on) of a destination row's segment?
        let pf_two = ib + 16 < imax;
        let mut jb = 0;
        while jb < cols {
            let jmax = (jb + B).min(cols);
            let j8max = jmax.min(c8);
            let mut j = jb;
            while j + 8 <= j8max {
                // Start the read-for-ownership of the NEXT j-tile's eight destination rows now.
                if j + 16 <= j8max {
                    let p = dst.wrapping_add((j + 8) * rows + ib) as *const i8;
                    for r in 0..8 {
                        let q = p.wrapping_add(r * rows * 2);
                        _mm_prefetch::<_MM_HINT_ET0>(q);
                        if pf_two {
                            _mm_prefetch::<_MM_HINT_ET0>(q.wrapping_add(32));
                        }
                    }
                }
                let mut i = ib;
                while i < i16max {
                    let s = src.add(i * cols + j);
                    let mut v = [_mm256_setzero_si256(); 8];
                    for (k, vk) in v.iter_mut().enumerate() {
                        let lo = _mm_loadu_si128(s.add(k * cols) as *const __m128i);
                        let hi = _mm_loadu_si128(s.add((k + 8) * cols) as *const __m128i);
                        *vk = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
                    }
                    let c = tr8x8_epi16!(v);
                    let d = dst.add(j * rows + i);
                    for (q, cq) in c.iter().enumerate() {
                        _mm256_storeu_si256(d.add(q * rows) as *mut __m256i, *cq);
                    }
                    i += 16;
                }
                if i < i8max {
                    // One 8-row strip: the same shuffles on the low halves; the high halves are
                    // never read back.
                    let s = src.add(i * cols + j);
                    let mut v = [_mm256_setzero_si256(); 8];
                    for (k, vk) in v.iter_mut().enumerate() {
                        let lo = _mm_loadu_si128(s.add(k * cols) as *const __m128i);
                        *vk = _mm256_castsi128_si256(lo);
                    }
                    let c = tr8x8_epi16!(v);
                    let d = dst.add(j * rows + i);
                    for (q, cq) in c.iter().enumerate() {
                        _mm_storeu_si128(
                            d.add(q * rows) as *mut __m128i,
                            _mm256_castsi256_si128(*cq),
                        );
                    }
                }
                j += 8;
            }
            // cols % 8 trailing columns of this block, over the rows the strips covered.
            while j < jmax {
                let mut i = ib;
                while i < i8max {
                    *dst.add(j * rows + i) = *src.add(i * cols + j);
                    i += 1;
                }
                j += 1;
            }
            // Rows past the last strip of this block (< 8 of them), full width.
            let mut i = i8max;
            while i < imax {
                let srow = src.add(i * cols);
                for jj in jb..jmax {
                    *dst.add(jj * rows + i) = *srow.add(jj);
                }
                i += 1;
            }
            jb += B;
        }
        ib += B;
    }
}

/// 16-bit row-range transpose: the AVX2 tile kernel where available, the generic scalar block
/// otherwise. Both are the same permutation, so the choice is invisible to every caller.
///
/// # Safety
/// `src` valid for `rows*cols`, `dst` for `cols*rows` `u16`; `i0 <= i1 <= rows`.
#[inline]
unsafe fn transpose_rows_u16(
    src: *const u16,
    dst: *mut u16,
    rows: usize,
    cols: usize,
    i0: usize,
    i1: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::isa::avx2() {
            transpose_rows_u16_avx2(src, dst, rows, cols, i0, i1);
            return;
        }
    }
    transpose_rows(src, dst, rows, cols, i0, i1);
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
        if crate::isa::avx2() {
            transpose_rows_f32_avx2(src, dst, rows, cols, i0, i1);
            return;
        }
    }
    transpose_rows(src, dst, rows, cols, i0, i1);
}

/// Element count below which the parallel transpose just runs serially (the rayon split is not worth
/// its overhead on a small, already-cache-resident matrix).
const TRANSPOSE_PAR_MIN: usize = 1 << 16;

/// `dst = srcᵀ` — transpose a `[rows, cols]` row-major f32 matrix into a `[cols, rows]` row-major one,
/// cache-blocked, single-threaded.
///
/// # Safety
/// `src` valid for `rows*cols`, `dst` for `cols*rows` f32, non-overlapping.
/// **Aliasing: all operands disjoint** — `dst` does not overlap `src`.
#[no_mangle]
pub unsafe extern "C" fn wukong_transpose_f32(
    src: *const f32,
    dst: *mut f32,
    rows: i64,
    cols: i64,
) {
    // P14.5: src valid for rows*cols, dst for cols*rows, disjoint (documented all-disjoint).
    if cfg!(debug_assertions) && rows > 0 && cols > 0 {
        crate::safety::debug_extent_2d(src, rows, cols);
        crate::safety::debug_extent_2d(dst, rows, cols);
        crate::safety::debug_disjoint(
            dst,
            rows.saturating_mul(cols),
            src,
            rows.saturating_mul(cols),
        );
    }
    if rows <= 0 || cols <= 0 {
        return;
    }
    transpose_rows_f32(src, dst, rows as usize, cols as usize, 0, rows as usize);
}

/// Multi-threaded f32 `dst = srcᵀ`: the independent `B`-row blocks are spread across cores, each
/// running [`transpose_rows_f32`] (the AVX2 8x8 kernel). Each block writes a disjoint set of `dst` entries, and a transpose is a permutation with
/// no accumulation, so it is bit-identical to the serial form regardless of thread count. Below
/// [`TRANSPOSE_PAR_MIN`] elements it runs serial. The row blocks are multiples of `B` = 32, hence of
/// 8, so every task's range starts on an 8-row strip boundary.
///
/// # Safety
/// Operand-size contract of [`wukong_transpose_f32`].
/// **Aliasing: all operands disjoint** — `dst` does not overlap `src`.
#[no_mangle]
pub unsafe extern "C" fn wukong_transpose_f32_parallel(
    src: *const f32,
    dst: *mut f32,
    rows: i64,
    cols: i64,
) {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && rows > 0 && cols > 0 {
        crate::safety::debug_extent_2d(src, rows, cols);
        crate::safety::debug_extent_2d(dst, rows, cols);
        crate::safety::debug_disjoint(
            dst,
            rows.saturating_mul(cols),
            src,
            rows.saturating_mul(cols),
        );
    }
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r * c < TRANSPOSE_PAR_MIN {
        transpose_rows_f32(src, dst, r, c, 0, r);
        return;
    }
    // This fork can be the process's FIRST rayon touch, so it must provision the global pool first —
    // [`crate::ensure_global_pool`]'s stated precondition. Forking bare builds rayon's default
    // 2 MiB-stack registry, so the runtime's later 16 MiB `build_global` silently loses the race and
    // outlined `@parallel` region bodies are left on undersized stacks. Idempotent (`Once`) and
    // provisioning-only: the work split below is unchanged, so serial == parallel stays bit-exact.
    crate::ensure_global_pool();
    let nblocks = r.div_ceil(B);
    let (src_addr, dst_addr) = (src as usize, dst as usize);
    crate::par::for_each_unit(nblocks, |blk| {
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
/// **Aliasing: all operands disjoint** — `dst` does not overlap `src`.
#[no_mangle]
pub unsafe extern "C" fn wukong_transpose_u16(
    src: *const u16,
    dst: *mut u16,
    rows: i64,
    cols: i64,
) {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && rows > 0 && cols > 0 {
        crate::safety::debug_extent_2d(src, rows, cols);
        crate::safety::debug_extent_2d(dst, rows, cols);
        crate::safety::debug_disjoint(
            dst,
            rows.saturating_mul(cols),
            src,
            rows.saturating_mul(cols),
        );
    }
    if rows <= 0 || cols <= 0 {
        return;
    }
    transpose_rows_u16(src, dst, rows as usize, cols as usize, 0, rows as usize);
}

/// Multi-threaded 16-bit `dst = srcᵀ`: the same independent `B`-row blocks as
/// [`wukong_transpose_f32_parallel`], each running [`transpose_rows_u16`] (the AVX2 tile kernel).
/// A permutation with no accumulation, so it is bit-identical to the serial form regardless of
/// thread count; below [`TRANSPOSE_PAR_MIN`] elements it runs serial. Task ranges start on multiples
/// of `B` = 32, so every 16-row strip lies inside one task.
///
/// # Safety
/// Operand-size contract of [`wukong_transpose_u16`].
/// **Aliasing:** that of [`wukong_transpose_u16`] — all operands disjoint: `dst` does not overlap
/// `src`.
#[no_mangle]
pub unsafe extern "C" fn wukong_transpose_u16_parallel(
    src: *const u16,
    dst: *mut u16,
    rows: i64,
    cols: i64,
) {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && rows > 0 && cols > 0 {
        crate::safety::debug_extent_2d(src, rows, cols);
        crate::safety::debug_extent_2d(dst, rows, cols);
        crate::safety::debug_disjoint(
            dst,
            rows.saturating_mul(cols),
            src,
            rows.saturating_mul(cols),
        );
    }
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (r, c) = (rows as usize, cols as usize);
    if r * c < TRANSPOSE_PAR_MIN {
        transpose_rows_u16(src, dst, r, c, 0, r);
        return;
    }
    // Same first-rayon-touch precondition as the f32 entry: provision the global pool before forking.
    crate::ensure_global_pool();
    let nblocks = r.div_ceil(B);
    let (src_addr, dst_addr) = (src as usize, dst as usize);
    crate::par::for_each_unit(nblocks, |blk| {
        let i0 = blk * B;
        let i1 = (i0 + B).min(r);
        // SAFETY: disjoint dst-row range per block (dst[*, i] for i in [i0, i1)); pointers re-derived
        // per task from the captured addresses (so they are Send).
        unsafe {
            transpose_rows_u16(src_addr as *const u16, dst_addr as *mut u16, r, c, i0, i1);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_extents_are_safe() {
        // P14.6: a [2,3] -> [3,2] transpose; dst[j*rows + i] = src[i*cols + j].
        let (rows, cols) = (2usize, 3usize);
        let src = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![f32::from_bits(0x7fc0_0001); rows * cols];
        // SAFETY: `src` sized rows*cols, `dst` sized cols*rows; a nonpositive rows is a no-op.
        unsafe {
            wukong_transpose_f32(src.as_ptr(), dst.as_mut_ptr(), 0, cols as i64);
        }
        assert!(
            dst.iter().all(|v| v.to_bits() == 0x7fc0_0001),
            "nonpositive rows must not write dst"
        );
        // SAFETY: `src` sized rows*cols, `dst` sized cols*rows; the positive call writes the transpose.
        unsafe {
            wukong_transpose_f32(src.as_ptr(), dst.as_mut_ptr(), rows as i64, cols as i64);
        }
        assert_eq!(dst, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

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
            (35, 35),   // a block wholly past the last multiple of 8 (the row-tail clamp)
            (40, 41),   // a column tail beside two row blocks
            (72, 72),   // a last column block of exactly one j-tile (no next-tile prefetch)
            (120, 96),  // many blocks, no tail in either dimension
        ] {
            let src: Vec<f32> = (0..r * c).map(|i| i as f32 * 0.5 - 3.0).collect();
            let want = naive_transpose(&src, r, c);
            let mut got = vec![0.0f32; r * c];
            let mut got_par = vec![0.0f32; r * c];
            // SAFETY: `got`/`got_par` are sized r*c for the dispatched f32 transpose.
            unsafe {
                wukong_transpose_f32(src.as_ptr(), got.as_mut_ptr(), r as i64, c as i64);
                wukong_transpose_f32_parallel(
                    src.as_ptr(),
                    got_par.as_mut_ptr(),
                    r as i64,
                    c as i64,
                );
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
            (16, 8),  // exactly one 16x8 tile
            (16, 16), // two j-tiles, one 16-row strip
            (24, 40), // a 16-row strip + an 8-row strip, a column tail
            (31, 17), // 16 + 8 + 7 trailing rows
            (32, 32),
            (33, 31),
            (40, 24), // a full block + one 8-row strip
            (47, 9),  // second block: 8-row strip + 7-row tail
            (100, 70),
            (128, 128), // power-of-two stride (the L1-set-collision shape)
            (1, 257),
            (257, 1),
            (300, 400), // > TRANSPOSE_PAR_MIN so the multicore path runs
            (520, 136), // > TRANSPOSE_PAR_MIN with a ragged last block
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
            // SAFETY: `got`/`got_par` are sized r*c for the dispatched u16 transpose.
            unsafe {
                wukong_transpose_u16(src.as_ptr(), got.as_mut_ptr(), r as i64, c as i64);
                wukong_transpose_u16_parallel(
                    src.as_ptr(),
                    got_par.as_mut_ptr(),
                    r as i64,
                    c as i64,
                );
            }
            assert_eq!(got, want, "u16 transpose {r}x{c} vs naive");
            assert_eq!(got, got_par, "u16 transpose serial vs parallel {r}x{c}");
        }
    }

    /// The AVX2 16-bit tile kernel must move **the same bits** as the generic scalar block, for every
    /// strip/tail combination a `B` = 32 block can hold (16-row strips, the single 8-row strip, the
    /// `< 8` row tail, the `cols % 8` column tail) and over partial `[i0, i1)` windows as the parallel
    /// entry hands them out. Every `u16` value is distinct within a run, so a misrouted element
    /// cannot hide behind an equal neighbour.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_u16_tile_moves_the_same_bits_as_the_scalar_block() {
        if !crate::isa::avx2() {
            return;
        }
        for (r, c) in [
            (1usize, 1usize),
            (7, 7),
            (8, 8),
            (15, 9),
            (16, 8),
            (16, 24),
            (17, 33),
            (23, 16),
            (24, 8),
            (31, 31),
            (32, 32),
            (33, 41),
            (40, 41),
            (48, 40),
            (56, 64),
            (64, 64),
            (72, 72),
            (95, 100),
            (120, 96),
            (128, 128),
        ] {
            let src: Vec<u16> = (0..r * c).map(|t| (t as u16).wrapping_mul(40503)).collect();
            let mut s = vec![0u16; r * c];
            let mut v = vec![0u16; r * c];
            // SAFETY: `s`/`v` are sized r*c (serial vs parallel u16).
            unsafe {
                transpose_rows(src.as_ptr(), s.as_mut_ptr(), r, c, 0, r);
                transpose_rows_u16_avx2(src.as_ptr(), v.as_mut_ptr(), r, c, 0, r);
            }
            assert_eq!(s, v, "u16 avx2 != scalar block at {r}x{c}");
            if r > B {
                let mut w = vec![0u16; r * c];
                // SAFETY: `w` is sized r*c (parallel path above the threshold).
                unsafe {
                    transpose_rows_u16_avx2(src.as_ptr(), w.as_mut_ptr(), r, c, 0, B);
                    transpose_rows_u16_avx2(src.as_ptr(), w.as_mut_ptr(), r, c, B, r);
                }
                assert_eq!(s, w, "u16 avx2 windowed != whole at {r}x{c}");
            }
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
        if !crate::isa::avx2() {
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
            // The j-tile-outer traversal hoisted both tails out of the strip loop, so pin the
            // shapes where a hoisted tail can go wrong: a block whose rows are ENTIRELY past the
            // last multiple of 8 (35, 36 -> the second block is rows 32..35 with no 8-row strip at
            // all, the `i8max.max(ib)` clamp), a column tail beside several row blocks (40x41,
            // 48x40), a last column block of exactly one j-tile so the next-tile prefetch is
            // skipped (72x72), and a many-block shape with no tail anywhere (120x96).
            (35, 35),
            (36, 72),
            (40, 41),
            (48, 40),
            (72, 72),
            (120, 96),
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
            // SAFETY: `s`/`v` are sized r*c (serial vs parallel f32).
            unsafe {
                transpose_rows(src.as_ptr(), s.as_mut_ptr(), r, c, 0, r);
                transpose_rows_f32_avx2(src.as_ptr(), v.as_mut_ptr(), r, c, 0, r);
            }
            assert_eq!(bits(&s), bits(&v), "avx2 != scalar block at {r}x{c}");
            // Split into two row windows the way the parallel entry does, and demand the same bits.
            if r > B {
                let mut w = vec![0.0f32; r * c];
                // SAFETY: `w` is sized r*c (parallel path above the threshold).
                unsafe {
                    transpose_rows_f32_avx2(src.as_ptr(), w.as_mut_ptr(), r, c, 0, B);
                    transpose_rows_f32_avx2(src.as_ptr(), w.as_mut_ptr(), r, c, B, r);
                }
                assert_eq!(bits(&s), bits(&w), "avx2 windowed != whole at {r}x{c}");
            }
        }
    }
}
