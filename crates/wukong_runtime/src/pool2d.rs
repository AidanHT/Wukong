//! 2D pooling — the spatial down-samplers of every convolutional stack: **max-pooling** and
//! **average-pooling** over a `kh×kw` window with stride `sh×sw` and no padding, on an
//! `[channels, h, w]` row-major (NCHW with N folded into the channel axis) input:
//!
//! ```text
//! oh = (h - kh)/sh + 1                 ow = (w - kw)/sw + 1
//! out[c, oy, ox] = ⊕  over dy∈[0,kh), dx∈[0,kw)  of  x[c, oy*sh+dy, ox*sw+dx]
//! ```
//!
//! where `⊕` is `max` for max-pool and `sum/(kh·kw)` for average-pool (every window is full — there is
//! no padding — so the divisor is always the whole window count, making `count_include_pad` moot).
//!
//! The naive spelling is a 4-deep nest `for c { for oy { for ox { fold the kh×kw window } } }` whose
//! window reads stride by `w` down the rows; gcc/rustc do not vectorize across the output-x dimension
//! for it. This kernel instead, for the common **unit horizontal stride** (`sw == 1`, where windows for
//! eight consecutive output columns overlap and read **contiguous** input), folds **8 output columns at
//! once** with AVX2: for a fixed window cell `(dy, dx)` the eight output columns `[ox, ox+8)` read the
//! contiguous input slice `[ox+dx, ox+dx+8)`, one `_mm256_loadu_ps`. For the **dominant downsampling
//! stride** (`sw == 2`, the 2×2/s2 and 3×3/s2 of every CNN), the eight output columns `[ox, ox+8)` map
//! to input columns `2*ox + dx, 2*ox+2 + dx, …, 2*ox+14 + dx` — strided by two. We still fold 8 output
//! columns at once: load the 16 contiguous input f32 at `[2*ox+dx, 2*ox+dx+16)` as two `__m256`
//! (`v0`=[0..7], `v1`=[8..15]) and **deinterleave the even lanes** (a cross-lane
//! `_mm256_permutevar8x32_ps` with index `{0,2,4,6,0,2,4,6}` on each, then blend the 128-bit halves) to
//! get exactly `{v0[0],v0[2],v0[4],v0[6], v1[0],v1[2],v1[4],v1[6]}` = the eight strided-by-2 columns.
//! For any other `sw` (> 2, or 0) the per-window reads are strided, so a clean scalar path streams each
//! window contiguously.
//!
//! **Bit-exactness.** Max is idempotent and associative, so any window-traversal order (and any
//! serial/parallel channel split) agrees. The average **sum order is fixed** — the window is folded in
//! `(dy, dx)` ascending order in *both* the scalar twin and the AVX2 path (the 8-wide path accumulates
//! into eight independent lanes in the same `(dy, dx)` sequence, so lane `l` receives exactly the
//! scalar fold for output column `ox+l`), then a single divide by `kh·kw` — so the SIMD path equals the
//! scalar nest bit-for-bit. Channels are independent (each output plane is disjoint), so the
//! `_parallel` form maps channels across cores and is **bit-identical** to the serial kernel (no
//! cross-channel combine); the interpreter marshals the serial form, so the differential oracle stays
//! exact.

/// Which pooling fold: `Max` seeds the window's first cell `x[…dy=0, dx=0]` and folds the rest by
/// `max` (idempotent — equivalent to seeding `-∞`); `Avg` seeds `0`, folds by `+` in `(dy, dx)`
/// ascending order, then divides by the full window count `kh·kw`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PoolKind {
    Max,
    Avg,
}

/// Pool one channel plane `[h, w] → [oh, ow]` with AVX2 along the output-x axis, **requires
/// `sw == 1`** (so the eight windows for output columns `[ox, ox+8)` read the contiguous input slice
/// `[ox+dx, ox+dx+8)`). Folds the `kh×kw` window in `(dy, dx)` ascending order into eight lanes; for
/// `Avg` divides each lane by `kh·kw` after the fold. The per-lane fold is the exact
/// `_mm256_{max,add}_ps` whose scalar twin is in [`pool_channel_scalar`], so the two agree
/// lane-for-lane.
///
/// # Safety
/// `xc` valid for `h*w`, `oc` for `oh*ow` f32; `sw == 1`; the output dims `oh`/`ow` consistent with
/// `h`/`w`/`kh`/`kw`/`sh`; AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn pool_channel_avx2_sw1(
    xc: *const f32,
    oc: *mut f32,
    w: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    kind: PoolKind,
) {
    use std::arch::x86_64::*;
    // The window count as a divisor — a true vector DIVIDE (`_mm256_div_ps`), matching the scalar twin's
    // `s / (kh*kw)` bit-for-bit. A reciprocal-multiply `x * (1/d)` would NOT agree (different rounding).
    let cnt = _mm256_set1_ps((kh * kw) as f32);
    for oy in 0..oh {
        let iy0 = oy * sh; // top input row of this window band
        let orow = oc.add(oy * ow);
        let mut ox = 0;
        // Main band: eight output columns at a time.
        while ox + 8 <= ow {
            // Seed: Max -> the window's first cell (dy=0, dx=0) for the 8 columns; Avg -> 0.
            let mut acc = match kind {
                PoolKind::Max => _mm256_loadu_ps(xc.add(iy0 * w + ox)),
                PoolKind::Avg => _mm256_setzero_ps(),
            };
            // Fold the window in (dy, dx) ascending order. For Max the (0,0) cell is the seed; skip it.
            for dy in 0..kh {
                let rowp = xc.add((iy0 + dy) * w);
                for dx in 0..kw {
                    if kind == PoolKind::Max && dy == 0 && dx == 0 {
                        continue;
                    }
                    // 8 windows, columns [ox, ox+8): input columns [ox+dx, ox+dx+8) — contiguous.
                    let v = _mm256_loadu_ps(rowp.add(ox + dx));
                    acc = match kind {
                        PoolKind::Max => _mm256_max_ps(acc, v),
                        PoolKind::Avg => _mm256_add_ps(acc, v),
                    };
                }
            }
            if kind == PoolKind::Avg {
                acc = _mm256_div_ps(acc, cnt);
            }
            _mm256_storeu_ps(orow.add(ox), acc);
            ox += 8;
        }
        // Scalar tail for the last < 8 output columns (identical fold order).
        while ox < ow {
            *orow.add(ox) = pool_window_scalar(xc, w, iy0, ox /* sw==1 */, kh, kw, kind);
            ox += 1;
        }
    }
}

/// Gather the eight **even-indexed** lanes of two consecutive `__m256` (the 16 contiguous input f32 at
/// `[base, base+16)`) into one vector: `{v0[0],v0[2],v0[4],v0[6], v1[0],v1[2],v1[4],v1[6]}`. For an
/// `sw == 2` window cell these are the input columns `2*ox+dx + {0,2,4,6,8,10,12,14}` — i.e. the eight
/// strided-by-2 columns feeding output columns `[ox, ox+8)`. The permute index `{0,2,4,6,0,2,4,6}`
/// applied per source pulls the four evens into *both* 128-bit halves of each result; the
/// `0b1111_0000` blend then keeps `v0`'s low half and `v1`'s high half. Plain lane shuffles — no
/// arithmetic — so the gathered values are bit-identical to the scalar loads.
///
/// # Safety
/// AVX2 available (callers are `#[target_feature(enable = "avx2")]`).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn deinterleave_even_sw2(v0: std::arch::x86_64::__m256, v1: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    let e0 = _mm256_permutevar8x32_ps(v0, idx); // [v0[0],v0[2],v0[4],v0[6], (repeat)]
    let e1 = _mm256_permutevar8x32_ps(v1, idx); // [v1[0],v1[2],v1[4],v1[6], (repeat)]
    // Keep low 128 from e0, high 128 from e1 -> {v0 evens, v1 evens}.
    _mm256_blend_ps(e0, e1, 0b1111_0000)
}

/// Pool one channel plane `[h, w] → [oh, ow]` with AVX2 along the output-x axis for **`sw == 2`** (the
/// canonical CNN downsampling stride). Output columns `[ox, ox+8)` map to input columns `2*ox + dx +
/// {0,2,…,14}`; for each window cell `(dy, dx)` we load the 16 contiguous input f32 at
/// `[2*ox+dx, 2*ox+dx+16)` and [`deinterleave_even_sw2`] them to the eight strided columns, then fold
/// in `(dy, dx)` ascending order into eight lanes exactly as the `sw == 1` path does (for `Avg`, one
/// vector divide by `kh·kw` after the fold). So lane `l` receives precisely the scalar window fold for
/// output column `ox+l`, bit-for-bit equal to [`pool_window_scalar`]. A scalar tail handles `OW % 8`.
///
/// # Safety
/// `xc` valid for `h*w`, `oc` for `oh*ow` f32; `sw == 2`; the output dims `oh`/`ow` consistent with
/// `h`/`w`/`kh`/`kw`/`sh`; AVX2 available. The 8-wide band's 16-f32 load `[2*ox+dx, 2*ox+dx+16)` is
/// kept in-bounds by the `ow_band` cutoff (`2*ox + (kw-1) + 16 <= w`); the rest is the scalar tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn pool_channel_avx2_sw2(
    xc: *const f32,
    oc: *mut f32,
    w: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    kind: PoolKind,
) {
    use std::arch::x86_64::*;
    let cnt = _mm256_set1_ps((kh * kw) as f32);
    // A band starting at output column `ox` (covering `[ox, ox+8)`) reads, for window cell `(dy, dx)`,
    // the 16 contiguous input f32 `[2*ox+dx, 2*ox+dx+16)` (two `_mm256_loadu_ps` feeding the even-lane
    // gather; the odd lanes — up to input col `2*ox+dx+15` — are loaded then discarded). All eight
    // output columns of the band share this same load offset `2*ox+dx`, so the band is in-bounds iff
    // the widest cell `dx = kw-1` is: `2*ox + (kw-1) + 16 <= w`. `ow_band` is the number of leading
    // output columns covered by full 8-wide bands satisfying that; the rest go to the scalar tail.
    let mut ow_band = 0usize;
    while ow_band + 8 <= ow && 2 * ow_band + (kw - 1) + 16 <= w {
        ow_band += 8;
    }
    for oy in 0..oh {
        let iy0 = oy * sh;
        let orow = oc.add(oy * ow);
        let mut ox = 0;
        // Main band: eight output columns at a time (sw==2 deinterleaved).
        while ox < ow_band {
            let base_col = 2 * ox; // input column of output column ox at dx=0
            let mut acc = match kind {
                PoolKind::Max => {
                    let v0 = _mm256_loadu_ps(xc.add(iy0 * w + base_col));
                    let v1 = _mm256_loadu_ps(xc.add(iy0 * w + base_col + 8));
                    deinterleave_even_sw2(v0, v1)
                }
                PoolKind::Avg => _mm256_setzero_ps(),
            };
            for dy in 0..kh {
                let rowp = xc.add((iy0 + dy) * w);
                for dx in 0..kw {
                    if kind == PoolKind::Max && dy == 0 && dx == 0 {
                        continue;
                    }
                    let off = base_col + dx;
                    let v0 = _mm256_loadu_ps(rowp.add(off));
                    let v1 = _mm256_loadu_ps(rowp.add(off + 8));
                    let v = deinterleave_even_sw2(v0, v1);
                    acc = match kind {
                        PoolKind::Max => _mm256_max_ps(acc, v),
                        PoolKind::Avg => _mm256_add_ps(acc, v),
                    };
                }
            }
            if kind == PoolKind::Avg {
                acc = _mm256_div_ps(acc, cnt);
            }
            _mm256_storeu_ps(orow.add(ox), acc);
            ox += 8;
        }
        // Scalar tail for the remaining output columns (identical fold order), sw==2 -> ix0 = ox*2.
        while ox < ow {
            *orow.add(ox) = pool_window_scalar(xc, w, iy0, ox * 2, kh, kw, kind);
            ox += 1;
        }
    }
}

/// Fold one `kh×kw` window of a single channel at output cell whose top-left input is row `iy0`,
/// column `ix0`, in `(dy, dx)` ascending order. The shared scalar definition used by the scalar
/// kernel, the AVX2 tail, and the test reference, so all three agree bit-for-bit.
///
/// # Safety
/// `xc` valid for the channel plane; `iy0+kh-1` and `ix0+kw-1` in bounds for the plane width `w`.
#[inline]
unsafe fn pool_window_scalar(
    xc: *const f32,
    w: usize,
    iy0: usize,
    ix0: usize,
    kh: usize,
    kw: usize,
    kind: PoolKind,
) -> f32 {
    match kind {
        PoolKind::Max => {
            // Seed the first cell (dy=0, dx=0), fold the rest by max (idempotent).
            let mut m = *xc.add(iy0 * w + ix0);
            for dy in 0..kh {
                let rowp = xc.add((iy0 + dy) * w + ix0);
                for dx in 0..kw {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let v = *rowp.add(dx);
                    if v > m {
                        m = v;
                    }
                }
            }
            m
        }
        PoolKind::Avg => {
            // Sum in (dy, dx) ascending order, then divide by the full window count.
            let mut s = 0.0f32;
            for dy in 0..kh {
                let rowp = xc.add((iy0 + dy) * w + ix0);
                for dx in 0..kw {
                    s += *rowp.add(dx);
                }
            }
            s / (kh * kw) as f32
        }
    }
}

/// Scalar twin / no-AVX2 fallback: pool one channel plane `[h, w] → [oh, ow]` for any stride, folding
/// each window via [`pool_window_scalar`] (the same `(dy, dx)` ascending order the AVX2 path uses).
///
/// # Safety
/// `xc` valid for `h*w`, `oc` for `oh*ow` f32; the output dims consistent with the input/window/stride.
#[allow(clippy::too_many_arguments)]
unsafe fn pool_channel_scalar(
    xc: *const f32,
    oc: *mut f32,
    w: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    kind: PoolKind,
) {
    for oy in 0..oh {
        let iy0 = oy * sh;
        let orow = oc.add(oy * ow);
        for ox in 0..ow {
            let ix0 = ox * sw;
            *orow.add(ox) = pool_window_scalar(xc, w, iy0, ix0, kh, kw, kind);
        }
    }
}

/// Pool one channel plane, dispatching the AVX2 fast path for `sw ∈ {1, 2}` when available (the
/// unit-stride contiguous band and the stride-2 deinterleaved band — the two dominant pooling
/// configurations), else the scalar twin.
///
/// # Safety
/// `xc` valid for `h*w`, `oc` for `oh*ow` f32; the output dims consistent with the input/window/stride.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn pool_channel(
    xc: *const f32,
    oc: *mut f32,
    w: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    kind: PoolKind,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            if sw == 1 {
                pool_channel_avx2_sw1(xc, oc, w, oh, ow, kh, kw, sh, kind);
                return;
            }
            if sw == 2 {
                pool_channel_avx2_sw2(xc, oc, w, oh, ow, kh, kw, sh, kind);
                return;
            }
        }
    }
    pool_channel_scalar(xc, oc, w, oh, ow, kh, kw, sh, sw, kind);
}

/// Compute the output spatial dims `(oh, ow)` for the no-padding pooling, or `None` if the window does
/// not fit (`kh > h` or `kw > w`, or any non-positive dim) — in which case the kernel writes nothing.
#[inline]
fn out_dims(h: i64, w: i64, kh: i64, kw: i64, sh: i64, sw: i64) -> Option<(usize, usize)> {
    if h <= 0 || w <= 0 || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 || kh > h || kw > w {
        return None;
    }
    let oh = (h - kh) / sh + 1;
    let ow = (w - kw) / sw + 1;
    if oh <= 0 || ow <= 0 {
        return None;
    }
    Some((oh as usize, ow as usize))
}

/// Serial pooling over all `channels` planes: `out[c, oy, ox] = ⊕ window`.
///
/// # Safety
/// `x` valid for `channels*h*w`, `out` for `channels*oh*ow` f32, non-overlapping.
#[allow(clippy::too_many_arguments)]
unsafe fn pool2d_serial(
    x: *const f32,
    out: *mut f32,
    channels: i64,
    h: i64,
    w: i64,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
    kind: PoolKind,
) {
    let Some((oh, ow)) = out_dims(h, w, kh, kw, sh, sw) else {
        return;
    };
    if channels <= 0 {
        return;
    }
    let (c_n, hh, ww) = (channels as usize, h as usize, w as usize);
    let (khu, kwu, shu, swu) = (kh as usize, kw as usize, sh as usize, sw as usize);
    let in_plane = hh * ww;
    let out_plane = oh * ow;
    for c in 0..c_n {
        pool_channel(
            x.add(c * in_plane),
            out.add(c * out_plane),
            ww,
            oh,
            ow,
            khu,
            kwu,
            shu,
            swu,
            kind,
        );
    }
}

/// Channel count below which the parallel pooling just runs serially.
const POOL_PAR_MIN: usize = 4;

/// Multi-threaded pooling: the **channels** are mapped across cores (each channel's output plane is
/// disjoint), every plane pooled by the identical per-channel routine — so the result is
/// **bit-identical to [`pool2d_serial`]** (no cross-channel combine). Below `POOL_PAR_MIN` channels it
/// runs serial.
///
/// # Safety
/// `x` valid for `channels*h*w`, `out` for `channels*oh*ow` f32, non-overlapping.
#[allow(clippy::too_many_arguments)]
unsafe fn pool2d_parallel(
    x: *const f32,
    out: *mut f32,
    channels: i64,
    h: i64,
    w: i64,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
    kind: PoolKind,
) {
    let Some((oh, ow)) = out_dims(h, w, kh, kw, sh, sw) else {
        return;
    };
    if channels <= 0 {
        return;
    }
    let c_n = channels as usize;
    if c_n < POOL_PAR_MIN {
        pool2d_serial(x, out, channels, h, w, kh, kw, sh, sw, kind);
        return;
    }
    let (hh, ww) = (h as usize, w as usize);
    let (khu, kwu, shu, swu) = (kh as usize, kw as usize, sh as usize, sw as usize);
    let in_plane = hh * ww;
    let out_plane = oh * ow;
    use rayon::prelude::*;
    let (x_addr, out_addr) = (x as usize, out as usize);
    (0..c_n).into_par_iter().for_each(|c| {
        // SAFETY: disjoint output plane per channel; pointers re-derived from the captured addresses.
        unsafe {
            pool_channel(
                (x_addr as *const f32).add(c * in_plane),
                (out_addr as *mut f32).add(c * out_plane),
                ww,
                oh,
                ow,
                khu,
                kwu,
                shu,
                swu,
                kind,
            );
        }
    });
}

/// 2D **max-pooling** `out[c, oy, ox] = max over the kh×kw window of x[c, oy*sh+dy, ox*sw+dx]`, over an
/// `[channels, h, w]` row-major input, no padding, single-threaded. Output is `[channels, oh, ow]` with
/// `oh = (h-kh)/sh + 1`, `ow = (w-kw)/sw + 1`. A window that does not fit (`kh > h` or `kw > w`) writes
/// nothing.
///
/// # Safety
/// `x` valid for `channels*h*w`, `out` for `channels*oh*ow` f32, non-overlapping.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wukong_maxpool2d_f32(
    x: *const f32,
    out: *mut f32,
    channels: i64,
    h: i64,
    w: i64,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
) {
    pool2d_serial(x, out, channels, h, w, kh, kw, sh, sw, PoolKind::Max);
}

/// Multi-threaded 2D max-pooling (channels across cores; **bit-identical to**
/// [`wukong_maxpool2d_f32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_maxpool2d_f32`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wukong_maxpool2d_f32_parallel(
    x: *const f32,
    out: *mut f32,
    channels: i64,
    h: i64,
    w: i64,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
) {
    pool2d_parallel(x, out, channels, h, w, kh, kw, sh, sw, PoolKind::Max);
}

/// 2D **average-pooling** `out[c, oy, ox] = (Σ over the kh×kw window of x[c, oy*sh+dy, ox*sw+dx]) /
/// (kh·kw)`, over an `[channels, h, w]` row-major input, no padding, single-threaded. The window sum is
/// folded in `(dy, dx)` ascending order, then divided by the full window count `kh·kw`. Output dims as
/// [`wukong_maxpool2d_f32`].
///
/// # Safety
/// `x` valid for `channels*h*w`, `out` for `channels*oh*ow` f32, non-overlapping.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wukong_avgpool2d_f32(
    x: *const f32,
    out: *mut f32,
    channels: i64,
    h: i64,
    w: i64,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
) {
    pool2d_serial(x, out, channels, h, w, kh, kw, sh, sw, PoolKind::Avg);
}

/// Multi-threaded 2D average-pooling (channels across cores; **bit-identical to**
/// [`wukong_avgpool2d_f32`]).
///
/// # Safety
/// Operand-size contract of [`wukong_avgpool2d_f32`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wukong_avgpool2d_f32_parallel(
    x: *const f32,
    out: *mut f32,
    channels: i64,
    h: i64,
    w: i64,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
) {
    pool2d_parallel(x, out, channels, h, w, kh, kw, sh, sw, PoolKind::Avg);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive 4-deep nested reference in the **same** `(dy, dx)` ascending window fold order the kernels
    /// use — so equality is literal (max idempotent, avg's fixed sum order), pinning AVX2 == scalar and
    /// serial == parallel.
    fn naive(
        x: &[f32],
        channels: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        kind: PoolKind,
    ) -> Vec<f32> {
        let oh = (h - kh) / sh + 1;
        let ow = (w - kw) / sw + 1;
        let mut out = vec![0.0f32; channels * oh * ow];
        for c in 0..channels {
            for oy in 0..oh {
                for ox in 0..ow {
                    let iy0 = oy * sh;
                    let ix0 = ox * sw;
                    let val = match kind {
                        PoolKind::Max => {
                            let mut m = x[c * h * w + iy0 * w + ix0];
                            for dy in 0..kh {
                                for dx in 0..kw {
                                    if dy == 0 && dx == 0 {
                                        continue;
                                    }
                                    let v = x[c * h * w + (iy0 + dy) * w + (ix0 + dx)];
                                    if v > m {
                                        m = v;
                                    }
                                }
                            }
                            m
                        }
                        PoolKind::Avg => {
                            let mut s = 0.0f32;
                            for dy in 0..kh {
                                for dx in 0..kw {
                                    s += x[c * h * w + (iy0 + dy) * w + (ix0 + dx)];
                                }
                            }
                            s / (kh * kw) as f32
                        }
                    };
                    out[c * oh * ow + oy * ow + ox] = val;
                }
            }
        }
        out
    }

    /// Max/avg pooling (AVX2 `sw==1` serial, the scalar `sw>1` path, and the multicore channel split)
    /// must equal the naive 4-deep nest **exactly** — same `(dy, dx)` window order — so it is literal
    /// bit equality. Configs straddle the 8-output-column edge, unit and non-unit strides, and the
    /// `POOL_PAR_MIN` channel threshold; several have `oh`/`ow` that don't evenly divide the input.
    #[test]
    fn pool2d_matches_naive_and_parallel() {
        // (channels, h, w, kh, kw, sh, sw)
        let configs: &[(usize, usize, usize, usize, usize, usize, usize)] = &[
            (1, 4, 4, 2, 2, 2, 2),
            (3, 8, 8, 2, 2, 2, 2),
            (2, 7, 7, 3, 3, 2, 2),   // 7 doesn't divide: oh=ow=3
            (4, 16, 16, 2, 2, 1, 1), // sw==1, wide -> AVX2 8-col band + tail
            (1, 5, 9, 2, 3, 1, 2),   // non-square window, mixed stride, ragged
            (8, 10, 10, 3, 3, 3, 3), // > POOL_PAR_MIN channels, 10 doesn't divide by 3
            (5, 1, 13, 1, 5, 1, 1),  // single input row, sw==1, AVX2 band crosses 8
            (2, 9, 9, 3, 3, 1, 1),   // sw==1 overlapping windows, ragged tail (ow=7)
            (6, 12, 1, 4, 1, 2, 1),  // single column, sw==1 but ow=1 -> all tail
        ];
        for &(channels, h, w, kh, kw, sh, sw) in configs {
            let n = channels * h * w;
            // A spread of distinct values so max has a unique winner; an exact-ish range so any
            // reordering (if a path had one) would surface as a bit mismatch.
            let x: Vec<f32> = (0..n).map(|t| ((t * 13 + 7) % 97) as f32 - 48.0).collect();
            for kind in [PoolKind::Max, PoolKind::Avg] {
                let want = naive(&x, channels, h, w, kh, kw, sh, sw, kind);
                let mut got = vec![0.0f32; want.len()];
                let mut got_par = vec![0.0f32; want.len()];
                let (f, fp): (
                    unsafe extern "C" fn(*const f32, *mut f32, i64, i64, i64, i64, i64, i64, i64),
                    unsafe extern "C" fn(*const f32, *mut f32, i64, i64, i64, i64, i64, i64, i64),
                ) = match kind {
                    PoolKind::Max => (wukong_maxpool2d_f32, wukong_maxpool2d_f32_parallel),
                    PoolKind::Avg => (wukong_avgpool2d_f32, wukong_avgpool2d_f32_parallel),
                };
                unsafe {
                    f(
                        x.as_ptr(),
                        got.as_mut_ptr(),
                        channels as i64,
                        h as i64,
                        w as i64,
                        kh as i64,
                        kw as i64,
                        sh as i64,
                        sw as i64,
                    );
                    fp(
                        x.as_ptr(),
                        got_par.as_mut_ptr(),
                        channels as i64,
                        h as i64,
                        w as i64,
                        kh as i64,
                        kw as i64,
                        sh as i64,
                        sw as i64,
                    );
                }
                let tag = if kind == PoolKind::Max { "max" } else { "avg" };
                assert_eq!(
                    got, want,
                    "{tag}pool ({channels},{h},{w},{kh},{kw},{sh},{sw}) vs naive"
                );
                assert_eq!(
                    got, got_par,
                    "{tag}pool serial vs parallel ({channels},{h},{w},{kh},{kw},{sh},{sw})"
                );
            }
        }
    }

    /// The `sw == 2` AVX2 deinterleaved band must equal [`pool_channel_scalar`] **bit-for-bit** for
    /// both max and avg, across widths that straddle the 8-output-column band edge and leave a < 8
    /// scalar tail (W ∈ {15,16,17,33,64,113}), windows {2×2, 3×3, 2×3}, and vertical strides {1, 2}.
    /// We drive `pool_channel_avx2_sw2` directly (the unit under test) and also pin the public
    /// serial==parallel. Guarded on runtime AVX2 (the band path is only taken when detected).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pool2d_sw2_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available; skipping pool2d_sw2_avx2_matches_scalar");
            return;
        }
        // (h, w, kh, kw, sh) with sw fixed to 2. Widths straddle the band edge / tail; some heights
        // don't divide evenly (ragged oh); 2 channels so the public-API parallel still maps per channel.
        let widths = [15usize, 16, 17, 33, 64, 113];
        let windows = [(2usize, 2usize), (3, 3), (2, 3)];
        let shs = [1usize, 2usize];
        let channels = 2usize;
        let h = 20usize; // tall enough for both sh and every window
        let sw = 2usize;
        for &w in &widths {
            for &(kh, kw) in &windows {
                for &sh in &shs {
                    let oh = (h - kh) / sh + 1;
                    let ow = (w - kw) / sw + 1;
                    if ow == 0 {
                        continue;
                    }
                    let n = channels * h * w;
                    let x: Vec<f32> =
                        (0..n).map(|t| ((t * 31 + 5) % 101) as f32 - 50.0).collect();
                    let in_plane = h * w;
                    let out_plane = oh * ow;
                    for kind in [PoolKind::Max, PoolKind::Avg] {
                        // Reference: the scalar twin, per channel.
                        let mut want = vec![0.0f32; channels * out_plane];
                        // Unit under test: the AVX2 sw==2 band path, per channel.
                        let mut got = vec![0.0f32; channels * out_plane];
                        unsafe {
                            for c in 0..channels {
                                pool_channel_scalar(
                                    x.as_ptr().add(c * in_plane),
                                    want.as_mut_ptr().add(c * out_plane),
                                    w,
                                    oh,
                                    ow,
                                    kh,
                                    kw,
                                    sh,
                                    sw,
                                    kind,
                                );
                                pool_channel_avx2_sw2(
                                    x.as_ptr().add(c * in_plane),
                                    got.as_mut_ptr().add(c * out_plane),
                                    w,
                                    oh,
                                    ow,
                                    kh,
                                    kw,
                                    sh,
                                    kind,
                                );
                            }
                        }
                        let tag = if kind == PoolKind::Max { "max" } else { "avg" };
                        assert_eq!(
                            got, want,
                            "{tag}pool sw2 AVX2 vs scalar (h={h},w={w},kh={kh},kw={kw},sh={sh})"
                        );

                        // And the public API: serial dispatch (which takes the AVX2 path here) ==
                        // the scalar reference, and serial == parallel.
                        let (f, fp): (
                            unsafe extern "C" fn(
                                *const f32, *mut f32, i64, i64, i64, i64, i64, i64, i64,
                            ),
                            unsafe extern "C" fn(
                                *const f32, *mut f32, i64, i64, i64, i64, i64, i64, i64,
                            ),
                        ) = match kind {
                            PoolKind::Max => {
                                (wukong_maxpool2d_f32, wukong_maxpool2d_f32_parallel)
                            }
                            PoolKind::Avg => {
                                (wukong_avgpool2d_f32, wukong_avgpool2d_f32_parallel)
                            }
                        };
                        let mut pub_got = vec![0.0f32; channels * out_plane];
                        let mut pub_par = vec![0.0f32; channels * out_plane];
                        unsafe {
                            f(
                                x.as_ptr(),
                                pub_got.as_mut_ptr(),
                                channels as i64,
                                h as i64,
                                w as i64,
                                kh as i64,
                                kw as i64,
                                sh as i64,
                                sw as i64,
                            );
                            fp(
                                x.as_ptr(),
                                pub_par.as_mut_ptr(),
                                channels as i64,
                                h as i64,
                                w as i64,
                                kh as i64,
                                kw as i64,
                                sh as i64,
                                sw as i64,
                            );
                        }
                        assert_eq!(
                            pub_got, want,
                            "{tag}pool sw2 public vs scalar (h={h},w={w},kh={kh},kw={kw},sh={sh})"
                        );
                        assert_eq!(
                            pub_got, pub_par,
                            "{tag}pool sw2 serial vs parallel (h={h},w={w},kh={kh},kw={kw},sh={sh})"
                        );
                    }
                }
            }
        }
    }
}
