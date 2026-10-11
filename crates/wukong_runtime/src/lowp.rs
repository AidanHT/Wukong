//! Low-precision (bf16 / f16) CPU reduction kernels with an **f32 accumulator** — the standard
//! mixed-precision contract. On this AVX2+F16C box there is no native bf16/f16 MAC, so low precision
//! is a **bandwidth / footprint** win, not a FLOP/s one: a reduction over half-width inputs streams
//! half the bytes, so a memory-bound reduction runs ~2× faster than its f32 twin. We widen on load
//! (F16C `vcvtph2ps` for f16; a `<<16` bit-extend for bf16 — both *lossless*) and accumulate in f32.
//!
//! # The accumulation contract for `sum` and `dot`
//!
//! The widen is exact, so the **only** float rounding in a bf16/f16 reduction is the f32
//! accumulation, and that accumulation is a *fixed, documented reassociation* of the mathematical
//! sum. It is deliberately **not** the strictly sequential f32 sum a naive `for (k) s += x[k];`
//! C loop performs, and it is deliberately more accurate than one.
//!
//! - **bf16 `sum`/`dot`** ([`sum_bf16_avx`], [`dot_bf16_avx`]) partition the input into
//!   `K = 8` accumulator vectors × 8 f32 lanes = **64 logical lanes**. Element `i` belongs to
//!   group `g = i / 16` and position `t = i % 16`; it lands in accumulator `j = (g % 4) + 4·(t % 2)`
//!   at lane `l = t / 2` (the even/odd split is what the branch-free widen below produces). Each
//!   lane accumulates its subsequence in ascending `i`; the 8 accumulator vectors are folded by a
//!   fixed binary tree (`acc[j] += acc[j+w]` for `w = 4, 2, 1`), then the 8 lanes by
//!   [`hcombine8`]. The `n % 16` trailing elements are folded in ascending order afterwards.
//! - **f16 `sum`/`dot`** ([`sum_f16_avx`], [`dot_f16_avx`]) also carry `K = 8` accumulator vectors,
//!   in the plain (not de-interleaved) map — F16C's `vcvtph2ps` already widens 8 *contiguous*
//!   elements, so there is no even/odd split to follow. The input is walked in 64-element blocks;
//!   block element `64·b + 8·t + l` lands in accumulator `t`, lane `l`, in ascending `b`. The `< 64`
//!   remainder runs in 8-element steps into accumulator 0 (lane `l` takes element `+ l`), the 8
//!   vectors fold by the same fixed tree as bf16, the lanes by [`hcombine8`], and the `n % 8` trailing
//!   elements fold in ascending order afterwards. For `n < 64` the blocked body never runs, so short
//!   inputs keep the exact bits of the earlier single-accumulator form.
//! - The max-family folds ([`wukong_reduce_bf16`], [`wukong_reduce_f16`]) round nothing, so they
//!   are the *exact* reduction of the widened values regardless of order.
//!
//! The partition is a pure function of `n` — never of the thread count, the ISA tier or the call
//! site — so the SIMD path, the scalar twin, the interpreter and the native backend all return
//! identical bits. NOTE the `_parallel` reduction twins fold over the fixed `reduce::RCHUNK`
//! decomposition, which reassociates against the flat serial kernel: a `@parallel` reduction here
//! is deterministic on any thread count but **not** bit-equal to its serial sibling (see
//! [`par_chunk_reduce`]).
//!
//! ## Why the lane count is load-bearing, not an implementation detail
//!
//! A sequential f32 accumulator over an ML-sized reduction is *badly* wrong. Once its running total
//! passes `2^23` the f32 ULP is 1.0, and every `O(1)` addend then rounds to a whole ULP — a
//! one-sided bias, not a random walk. Measured against an independent in-order **f64** reference
//! over the `wukong_xbench` bf16 inputs (all-positive, `O(1)` magnitude), relative error:
//!
//! | N | `sum` seq-f32 | `sum` here | `dot` seq-f32 | `dot` here |
//! |---|---|---|---|---|
//! | 2^20 | 8.80e-4 | **0** (exact) | 2.42e-4 | **2.21e-7** |
//! | 2^22 | 1.97e-3 | **1.03e-8** | 5.37e-4 | **1.08e-5** |
//! | 2^24 | 1.407e-2 | **1.69e-4** | 2.333e-2 | **1.62e-4** |
//!
//! So when a cross-language check reports a ~1e-2 disagreement between these kernels and a C/C++/
//! Rust peer compiled at honest flags, **it is reporting the peer's error, not ours** — the peers
//! agree with each other to the last bit precisely because all three emit the same in-order sum.
//! `lowp_reductions_beat_the_sequential_f32_sum` pins this ruling; do not "fix" a drift warning by
//! matching the peer.
//!
//! Beyond the reductions this module also owns the half-precision **elementwise** family: the
//! mixed-precision `axpby` (`wukong_axpby_{bf16,f16}`, half in / f32 out) and the narrowing-output
//! kernels (`wukong_axpby_{bf16,f16}_out`, `wukong_vmath_{bf16,f16}_out`) that compute in f32 and
//! round the store through the shared `f32_to_{bf16,f16}_bits` shim. Those are pure maps — no
//! accumulation to reorder — so their SIMD lanes equal the scalar twin element-for-element, and the
//! narrowing store's non-temporal path (the `stream_narrow!` skeleton) writes the same bits as the cacheable
//! one. None of the elementwise entries has a `_parallel` twin.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::bf16_bits_to_f32;

/// IEEE f16 (stored bits) → f32. Lossless, so it equals the F16C `vcvtph2ps` result exactly. Thin
/// alias for the crate-root [`crate::f16_bits_to_f32`] — one shared definition of the f16 widen.
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    crate::f16_bits_to_f32(h)
}

/// Fixed 8-lane horizontal combine tree (the same order the scalar twin and the SIMD path use).
#[inline]
fn hcombine8(l: &[f32; 8]) -> f32 {
    ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]))
}

/// Element kind for the generic scalar twin.
#[derive(Clone, Copy)]
enum Half {
    F16,
    Bf16,
}

impl Half {
    #[inline]
    fn widen(self, bits: u16) -> f32 {
        match self {
            Half::F16 => f16_to_f32(bits),
            Half::Bf16 => bf16_bits_to_f32(bits),
        }
    }

    /// Round an `f32` to this half type's stored bits — the shared `wukong_f32_to_{bf16,f16}_bits`
    /// shim the interpreter and the native narrowing store both use, so a half-**output** kernel is
    /// bit-for-bit consistent with a scalar `<f32> as bf16/f16` store (round at the value, not the
    /// store — the sweep-6 lesson).
    #[inline]
    fn narrow(self, v: f32) -> u16 {
        match self {
            Half::F16 => crate::f32_to_f16_bits(v),
            Half::Bf16 => crate::f32_to_bf16_bits(v),
        }
    }
}

/// 256-bit loads issued per unrolled step of the bf16 `sum`/`dot` inner loop.
const BF_H: usize = 4;
/// Independent f32 accumulator *vectors* the bf16 `sum`/`dot` keep (`2 × BF_H`, one per widen half),
/// i.e. `8 × BF_K = 64` logical lanes. Two things ride on this number and it is tuned for both:
/// it is the reduction's dependency-chain depth divider (one 4-cycle `vfmadd` chain per accumulator,
/// so `BF_K = 8` turns a latency-bound loop into a port-bound one), and it is the accuracy
/// parameter of the module-header contract table. Changing it changes the kernel's *result bits*;
/// the scalar twins below must be changed with it or `simd_equals_scalar_twin_bit_for_bit` goes red.
const BF_K: usize = 2 * BF_H;
/// Extra unroll of the bf16 `sum` group loop. **Scheduling only** — the lane map is `acc[g % BF_H]`
/// for every `BF_SUM_U`, so the result bits do not depend on it and the scalar twin does not
/// mention it. It exists because `sum` at N = 2^24 is DRAM bound, where the loop's *bytes requested
/// per iteration* (here 16 loads = 512 B = 8 cache lines) is what sets streaming throughput: at
/// `BF_SUM_U = 1` the new kernel measured ~0.96x of the old one on that row across nine runs, at
/// `BF_SUM_U = 4` it measured 1.10-1.29x in every one of four. `dot` needs no such unroll (it
/// already issues two streams' worth of loads per group) and measured very slightly worse with one.
const BF_SUM_U: usize = 4;

/// Scalar twin of the bf16 SIMD sum: the identical 64-lane de-interleaved partition documented in
/// the module header, folded by the identical fixed tree. The reference the SIMD path must match
/// bit-for-bit, and the no-AVX2 fallback.
fn sum_scalar_bf16(x: &[u16]) -> f32 {
    let n = x.len();
    let groups = n / 16;
    let mut acc = [[0.0f32; 8]; BF_K];
    for g in 0..groups {
        let j = g % BF_H;
        let base = g * 16;
        for l in 0..8 {
            acc[j][l] += bf16_bits_to_f32(x[base + 2 * l]);
            acc[j + BF_H][l] += bf16_bits_to_f32(x[base + 2 * l + 1]);
        }
    }
    let mut s = hcombine8(&fold_tree(acc));
    for &v in &x[groups * 16..] {
        s += bf16_bits_to_f32(v);
    }
    s
}

/// Scalar twin of the bf16 SIMD dot (f32-accumulated, fused like the SIMD `fmadd`), over the same
/// 64-lane partition as [`sum_scalar_bf16`].
fn dot_scalar_bf16(x: &[u16], y: &[u16]) -> f32 {
    let n = x.len();
    let groups = n / 16;
    let mut acc = [[0.0f32; 8]; BF_K];
    for g in 0..groups {
        let base = g * 16;
        // `split_at_mut` so the even-half and odd-half accumulators of the same `j` are both live.
        let (evens, odds) = acc.split_at_mut(BF_H);
        let (ae, ao) = (&mut evens[g % BF_H], &mut odds[g % BF_H]);
        for (l, (ael, aol)) in ae.iter_mut().zip(ao.iter_mut()).enumerate() {
            let (e, o) = (base + 2 * l, base + 2 * l + 1);
            *ael = bf16_bits_to_f32(x[e]).mul_add(bf16_bits_to_f32(y[e]), *ael);
            *aol = bf16_bits_to_f32(x[o]).mul_add(bf16_bits_to_f32(y[o]), *aol);
        }
    }
    let mut s = hcombine8(&fold_tree(acc));
    for i in groups * 16..n {
        s = bf16_bits_to_f32(x[i]).mul_add(bf16_bits_to_f32(y[i]), s);
    }
    s
}

/// The fixed binary tree over the `BF_K` accumulator vectors — `acc[j] += acc[j+w]` for
/// `w = BF_K/2, …, 1`, elementwise per lane. Mirrors the SIMD fold exactly.
#[inline]
fn fold_tree(mut acc: [[f32; 8]; BF_K]) -> [f32; 8] {
    let mut w = BF_K;
    while w > 1 {
        let h = w / 2;
        let (lo, hi) = acc.split_at_mut(h);
        for (a, b) in lo.iter_mut().zip(hi.iter()) {
            for (al, bl) in a.iter_mut().zip(b.iter()) {
                *al += *bl;
            }
        }
        w = h;
    }
    acc[0]
}

/// Elements per blocked step of the f16 `sum`/`dot`: one 8-lane vector per accumulator.
const F16_BLOCK: usize = 8 * BF_K;

/// Scalar twin of the f16 SIMD sum: the `BF_K`-accumulator 64-element-block partition documented in
/// the module header, the same fixed tree and lane combine, then the scalar tail. This is the
/// reference the SIMD path must match bit-for-bit, and the no-F16C fallback.
fn sum_scalar_f16(x: &[u16]) -> f32 {
    let n = x.len();
    let mut acc = [[0.0f32; 8]; BF_K];
    let mut i = 0;
    while i + F16_BLOCK <= n {
        for (t, at) in acc.iter_mut().enumerate() {
            for (l, a) in at.iter_mut().enumerate() {
                *a += f16_to_f32(x[i + 8 * t + l]);
            }
        }
        i += F16_BLOCK;
    }
    while i + 8 <= n {
        for (l, a) in acc[0].iter_mut().enumerate() {
            *a += f16_to_f32(x[i + l]);
        }
        i += 8;
    }
    let mut s = hcombine8(&fold_tree(acc));
    for &v in &x[i..] {
        s += f16_to_f32(v);
    }
    s
}

/// Scalar twin of the f16 SIMD dot (f32-accumulated, fused like the SIMD `fmadd`), over the same
/// partition as [`sum_scalar_f16`].
fn dot_scalar_f16(x: &[u16], y: &[u16]) -> f32 {
    let n = x.len();
    let mut acc = [[0.0f32; 8]; BF_K];
    let mut i = 0;
    while i + F16_BLOCK <= n {
        for (t, at) in acc.iter_mut().enumerate() {
            for (l, a) in at.iter_mut().enumerate() {
                let k = i + 8 * t + l;
                *a = f16_to_f32(x[k]).mul_add(f16_to_f32(y[k]), *a);
            }
        }
        i += F16_BLOCK;
    }
    while i + 8 <= n {
        for (l, a) in acc[0].iter_mut().enumerate() {
            *a = f16_to_f32(x[i + l]).mul_add(f16_to_f32(y[i + l]), *a);
        }
        i += 8;
    }
    let mut s = hcombine8(&fold_tree(acc));
    for k in i..n {
        s = f16_to_f32(x[k]).mul_add(f16_to_f32(y[k]), s);
    }
    s
}

// ---- SIMD widen helpers (8 lanes) ----

/// `pub(crate)` so `vmath.rs`'s f16-input activation kernel widens with the *identical* F16C
/// `vcvtph2ps` this module's f16 reductions use — one source of truth for the f16 widen.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
#[inline]
pub(crate) unsafe fn widen_f16(p: *const u16) -> __m256 {
    _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
}

/// `pub(crate)` so `vmath.rs`'s bf16-input activation kernel (`wukong_vmath_bf16`) widens with the
/// *identical* lossless `<<16` bit-extend this module's reductions use — one source of truth for the
/// bf16→f32 widen keeps every bf16 dispatch path bit-for-bit consistent.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn widen_bf16(p: *const u16) -> __m256 {
    // zero-extend 8×u16 → 8×u32, shift the bf16 bits into the f32 high half, reinterpret.
    let lo = _mm_loadu_si128(p as *const __m128i);
    let w = _mm256_cvtepu16_epi32(lo);
    _mm256_castsi256_ps(_mm256_slli_epi32(w, 16))
}

// ---- The branch-free bf16 widen the *reductions* use (16 elements per 256-bit load) ----
//
// `widen_bf16` above costs one `vpmovzxwd` per 8 elements, and `vpmovzxwd ymm, xmm` is a SHUFFLE:
// it issues only on the single shuffle port, so the widen alone caps the loop at one 8-element step
// per cycle no matter how much FMA throughput is left over. The pair below avoids the shuffle port
// entirely. Load 16 bf16 as one `__m256i`; dword `k` then holds element `2k` in its LOW half and
// element `2k+1` in its HIGH half (little endian), and a bf16's f32 value is exactly its 16 bits in
// the f32 high half:
//
//   `vpslld(v, 16)`             -> f32 lanes of the EVEN elements (the odd half is shifted out)
//   `vpand (v, 0xFFFF_0000)`    -> f32 lanes of the ODD  elements (the even half is masked out)
//
// Both are plain integer ALU ops (p0/p1, and p5 as well for `vpand`), both are exact, and together
// they widen 16 elements for two uops instead of two shuffle-port uops for the same 16. The price is
// that the two halves come out de-interleaved — which is why the reductions' lane map is the
// even/odd one documented in the module header rather than `8k+l`.

/// f32 lanes of the **even**-indexed bf16 elements of a 16-element load. Exact.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn widen_bf16_even(v: __m256i) -> __m256 {
    _mm256_castsi256_ps(_mm256_slli_epi32(v, 16))
}

/// f32 lanes of the **odd**-indexed bf16 elements of a 16-element load. Exact.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn widen_bf16_odd(v: __m256i) -> __m256 {
    _mm256_castsi256_ps(_mm256_and_si256(
        v,
        _mm256_set1_epi32(0xFFFF_0000u32 as i32),
    ))
}

// ---- SIMD narrowing store helpers (8 f32 lanes → 8 contiguous half bits) — the reverse of the widen
//      helpers, used by the half-**output** streaming kernels. Each is bit-identical to the scalar
//      `f32_to_{bf16,f16}_bits` shim the interpreter narrows through (twin-tested), so a half-output
//      kernel stays bit-for-bit consistent with the differential oracle. `pub(crate)` so `velem.rs`'s
//      and `bias.rs`'s narrowing stores share the one definition. ----

/// Round 8 f32 lanes to bf16 stored bits, returned as 8× `u32` (each bf16 value in its low 16 bits) —
/// the store-free core shared by [`narrow_bf16`] (8-lane store) and [`narrow_bf16_pack`] (16-lane NT
/// pack). Round-to-nearest-even `(bits + 0x7fff + ((bits>>16)&1)) >> 16`, with a blend to the
/// quieted-NaN form `(bits>>16)|0x40` on unordered lanes — bit-identical to [`crate::f32_to_bf16_bits`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bf16_round8(v: __m256) -> __m256i {
    let bits = _mm256_castps_si256(v);
    let lsb = _mm256_and_si256(_mm256_srli_epi32::<16>(bits), _mm256_set1_epi32(1));
    let bias = _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7fff));
    let rounded = _mm256_srli_epi32::<16>(_mm256_add_epi32(bits, bias));
    let nan_res = _mm256_or_si256(_mm256_srli_epi32::<16>(bits), _mm256_set1_epi32(0x40));
    let is_nan = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_UNORD_Q>(v, v));
    _mm256_blendv_epi8(rounded, nan_res, is_nan) // per-32-bit-lane mask (all-ones on NaN)
}

/// Round 8 f32 lanes to bf16 stored bits and store them as 8 contiguous `u16` at `out` — bit-identical
/// to [`crate::f32_to_bf16_bits`] applied lane-by-lane. Used for the ≤8-lane tail of the streaming
/// kernels (the bulk goes through [`narrow_bf16_pack`] + a 256-bit store).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn narrow_bf16(v: __m256, out: *mut u16) {
    let res = bf16_round8(v);
    // Pack the 8× u32 (each a u16 in its low half) to 8× u16, then gather the two low quadwords.
    let packed = _mm256_packus_epi32(res, res); // [r0..3 r0..3 | r4..7 r4..7] as u16
    let lo = _mm256_permute4x64_epi64::<0b0000_1000>(packed); // qwords [q0, q2, _, _] = r0..7
    _mm_storeu_si128(out as *mut __m128i, _mm256_castsi256_si128(lo));
}

/// Round **16** f32 lanes (two `__m256`) to 16 contiguous bf16 stored bits packed in one `__m256i` — the
/// 256-bit core of the streaming half-output kernel, so a whole cache-line-friendly 32-byte block is
/// produced per store (enables a single non-temporal `vmovntdq`). Each 8-lane half is
/// [`bf16_round8`], so it is bit-identical to [`narrow_bf16`] (and thus the scalar shim) per lane.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn narrow_bf16_pack(v0: __m256, v1: __m256) -> __m256i {
    let r0 = bf16_round8(v0);
    let r1 = bf16_round8(v1);
    // packus interleaves per 128-bit lane: [r0.0-3 r1.0-3 | r0.4-7 r1.4-7]; permute qwords (0,2,1,3)
    // → [r0.0-7 r1.0-7], 16 contiguous bf16.
    let packed = _mm256_packus_epi32(r0, r1);
    _mm256_permute4x64_epi64::<0b11_01_10_00>(packed)
}

/// Round 8 f32 lanes to IEEE-f16 stored bits and store them as 8 contiguous `u16` — F16C `vcvtps2ph`
/// (round to nearest even), which the `simd_equals_scalar_twin` test pins == [`crate::f32_to_f16_bits`]
/// (the `half` crate). So the f16 narrowing store matches the shim the interpreter uses. Used for the
/// ≤8-lane tail (the bulk goes through [`narrow_f16_pack`] + a 256-bit store).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
#[inline]
pub(crate) unsafe fn narrow_f16(v: __m256, out: *mut u16) {
    // `vcvtps2ph` rounding imm is 3 bits; round-to-nearest-even (`_MM_FROUND_TO_NEAREST_INT` = 0) is
    // what `half::f16::from_f32` uses (the narrow twin test pins F16C == the shim bit-for-bit).
    let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
    _mm_storeu_si128(out as *mut __m128i, h);
}

/// Round **16** f32 lanes (two `__m256`) to 16 contiguous f16 stored bits packed in one `__m256i` — two
/// `vcvtps2ph` (each 8 lanes → a `__m128i`) joined low/high, so a 32-byte block is produced per store
/// (single non-temporal `vmovntdq`). Bit-identical to [`narrow_f16`] per lane.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
#[inline]
unsafe fn narrow_f16_pack(v0: __m256, v1: __m256) -> __m256i {
    let h0 = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v0); // 8 halves (low block)
    let h1 = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v1); // 8 halves (high block)
    _mm256_set_m128i(h1, h0) // [h0 (low 128) | h1 (high 128)] = 16 contiguous f16
}

/// IEEE-f16 `Σ x[i]` with an f32 accumulator, over the `BF_K`-chain partition of the module header.
/// Twin: [`sum_scalar_f16`].
///
/// The single-accumulator predecessor had both defects `sum_bf16_avx` lists: one loop-carried
/// `vaddps` per 8 elements (latency bound) and, the f16-specific half, only one 128-bit load per
/// step. Now a 64-element block issues eight independent 16-byte loads into eight chains. The tree is
/// [`fold_tree_avx`] — the same `acc[j] + acc[j + w]` fold as the bf16 kernels and [`fold_tree`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn sum_f16_avx(x: &[u16]) -> f32 {
    let n = x.len();
    let p = x.as_ptr();
    let mut acc = [_mm256_setzero_ps(); BF_K];
    let mut i = 0usize;
    while i + F16_BLOCK <= n {
        for (t, at) in acc.iter_mut().enumerate() {
            *at = _mm256_add_ps(*at, widen_f16(p.add(i + 8 * t)));
        }
        i += F16_BLOCK;
    }
    while i + 8 <= n {
        acc[0] = _mm256_add_ps(acc[0], widen_f16(p.add(i)));
        i += 8;
    }
    let mut s = fold_tree_avx(&mut acc);
    for &v in &x[i..] {
        s += f16_to_f32(v);
    }
    s
}

/// The fixed binary fold over the `BF_K` accumulator vectors, then the 8 lanes. Mirrors
/// [`fold_tree`] exactly (`acc[j] += acc[j+w]` for `w = BF_K/2 … 1`, elementwise).
///
/// Enables only `avx` (it issues nothing but `vaddps` and a store): the f16 reductions are dispatched
/// on `avx` + `f16c`, not `avx2`, and must not reach an AVX2-compiled body on a host without AVX2.
/// The bf16 callers enable a superset, so it still inlines into them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn fold_tree_avx(acc: &mut [__m256; BF_K]) -> f32 {
    let mut w = BF_K;
    while w > 1 {
        let h = w / 2;
        for j in 0..h {
            acc[j] = _mm256_add_ps(acc[j], acc[j + h]);
        }
        w = h;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc[0]);
    hcombine8(&lanes)
}

/// bf16 `Σ x[i]` with an f32 accumulator, over the 64-lane de-interleaved partition of the module
/// header. Twin: [`sum_scalar_bf16`].
///
/// Two defects of the previous one-accumulator form are fixed here, and both are countable:
///   * the loop-carried `vaddps` chain is now `BF_K = 8` independent chains, so the step rate is set
///     by issue throughput rather than by add latency;
///   * the widen no longer touches the shuffle port ([`widen_bf16_even`] / [`widen_bf16_odd`]),
///     removing one shuffle uop per 8 elements — previously the loop's hard ceiling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_bf16_avx(x: &[u16]) -> f32 {
    let n = x.len();
    let groups = n / 16; // whole 256-bit loads
    let p = x.as_ptr();
    let mut acc = [_mm256_setzero_ps(); BF_K];
    let steps = groups / (BF_H * BF_SUM_U);
    let mut g = 0usize;
    for _ in 0..steps {
        for u in 0..BF_SUM_U {
            for j in 0..BF_H {
                let v = _mm256_loadu_si256(p.add((g + u * BF_H + j) * 16) as *const __m256i);
                acc[j] = _mm256_add_ps(acc[j], widen_bf16_even(v));
                acc[j + BF_H] = _mm256_add_ps(acc[j + BF_H], widen_bf16_odd(v));
            }
        }
        g += BF_H * BF_SUM_U;
    }
    // `< BF_H * BF_SUM_U` leftover groups, still `acc[group % BF_H]`. Both loop bounds stay
    // constant (`g` is a multiple of `BF_H` here, so `j` *is* `g % BF_H`): a variable accumulator
    // index would make LLVM give up and keep the whole array on the stack.
    while g < groups {
        for j in 0..BF_H {
            if g + j < groups {
                let v = _mm256_loadu_si256(p.add((g + j) * 16) as *const __m256i);
                acc[j] = _mm256_add_ps(acc[j], widen_bf16_even(v));
                acc[j + BF_H] = _mm256_add_ps(acc[j + BF_H], widen_bf16_odd(v));
            }
        }
        g += BF_H;
    }
    let mut s = fold_tree_avx(&mut acc);
    for &v in &x[groups * 16..] {
        s += bf16_bits_to_f32(v);
    }
    s
}

/// IEEE-f16 `Σ x[i]·y[i]` with an f32 accumulator, over the same partition as [`sum_f16_avx`].
/// Twin: [`dot_scalar_f16`]. Eight independent `vfmadd` chains where the predecessor ran one, so the
/// step rate is set by FMA/load issue rather than by the 4-cycle FMA latency.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c,fma")]
unsafe fn dot_f16_avx(x: &[u16], y: &[u16]) -> f32 {
    let n = x.len();
    let (px, py) = (x.as_ptr(), y.as_ptr());
    let mut acc = [_mm256_setzero_ps(); BF_K];
    let mut i = 0usize;
    while i + F16_BLOCK <= n {
        for (t, at) in acc.iter_mut().enumerate() {
            let o = i + 8 * t;
            *at = _mm256_fmadd_ps(widen_f16(px.add(o)), widen_f16(py.add(o)), *at);
        }
        i += F16_BLOCK;
    }
    while i + 8 <= n {
        acc[0] = _mm256_fmadd_ps(widen_f16(px.add(i)), widen_f16(py.add(i)), acc[0]);
        i += 8;
    }
    let mut s = fold_tree_avx(&mut acc);
    for k in i..n {
        s = f16_to_f32(x[k]).mul_add(f16_to_f32(y[k]), s);
    }
    s
}

/// bf16 `Σ x[i]·y[i]` with an f32 accumulator, over the same 64-lane partition as
/// [`sum_bf16_avx`]. Twin: [`dot_scalar_bf16`].
///
/// The single-accumulator predecessor was **`vfmadd` latency bound**: one dependent FMA per 8
/// elements at 4 cycles of latency is 0.5 cycles/element no matter how much of the machine is idle.
/// `BF_K = 8` independent chains plus the shuffle-port-free widen make it issue bound instead.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16_avx(x: &[u16], y: &[u16]) -> f32 {
    let n = x.len();
    let groups = n / 16;
    let (px, py) = (x.as_ptr(), y.as_ptr());
    let mut acc = [_mm256_setzero_ps(); BF_K];
    let (full, rest) = (groups / BF_H, groups % BF_H);
    let mut g = 0usize;
    for _ in 0..full {
        for j in 0..BF_H {
            let a = _mm256_loadu_si256(px.add((g + j) * 16) as *const __m256i);
            let b = _mm256_loadu_si256(py.add((g + j) * 16) as *const __m256i);
            acc[j] = _mm256_fmadd_ps(widen_bf16_even(a), widen_bf16_even(b), acc[j]);
            acc[j + BF_H] = _mm256_fmadd_ps(widen_bf16_odd(a), widen_bf16_odd(b), acc[j + BF_H]);
        }
        g += BF_H;
    }
    for j in 0..BF_H {
        if j < rest {
            let a = _mm256_loadu_si256(px.add((g + j) * 16) as *const __m256i);
            let b = _mm256_loadu_si256(py.add((g + j) * 16) as *const __m256i);
            acc[j] = _mm256_fmadd_ps(widen_bf16_even(a), widen_bf16_even(b), acc[j]);
            acc[j + BF_H] = _mm256_fmadd_ps(widen_bf16_odd(a), widen_bf16_odd(b), acc[j + BF_H]);
        }
    }
    let mut s = fold_tree_avx(&mut acc);
    for i in groups * 16..n {
        s = bf16_bits_to_f32(x[i]).mul_add(bf16_bits_to_f32(y[i]), s);
    }
    s
}

// ---- bf16 max / min / absmax (the max-family folds; widen is lossless and max/min round nothing,
//      so the result is the *exact* reduction of the widened values — no f32-accumulator caveat) ----

/// One widened element folded into a max/min/maxabs accumulator, matching `reduce.rs::contrib` for
/// these ops so the bf16 max-family fold is consistent with the f32 reduction's semantics (and the
/// scalar twin matches the AVX2 lanes: `f32::abs` clears the sign bit ≡ `andnot(-0.0, v)`).
#[inline(always)]
fn rfold(a: f32, xi: f32, op: i64) -> f32 {
    use crate::reduce::{fold2, RED_MAX, RED_MAXABS};
    if op == RED_MAXABS {
        fold2(a, xi.abs(), RED_MAX)
    } else {
        fold2(a, xi, op)
    }
}

/// Scalar twin of the bf16/f16 max/min/maxabs reduction: 8 logical lane accumulators (lane `l` folds
/// elements `8k+l`), the fixed [`crate::reduce::hcombine8`] combine, then a scalar tail — bit-identical
/// to the AVX2 path (a test pins it) and the no-AVX2 fallback. Generic over the half kind.
fn reduce_minmax_scalar(kind: Half, op: i64, x: &[u16]) -> f32 {
    use crate::reduce::{hcombine8, ident};
    let n = x.len();
    let chunks = n / 8;
    let mut acc = [ident(op); 8];
    for c in 0..chunks {
        for (l, a) in acc.iter_mut().enumerate() {
            *a = rfold(*a, kind.widen(x[c * 8 + l]), op);
        }
    }
    let mut s = hcombine8(acc, op);
    for &v in &x[chunks * 8..] {
        s = rfold(s, kind.widen(v), op);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_minmax_bf16_avx(op: i64, x: &[u16]) -> f32 {
    use crate::reduce::{hcombine8, ident, RED_MAXABS, RED_MIN};
    let n = x.len();
    let chunks = n / 8;
    let mut acc = _mm256_set1_ps(ident(op));
    let absmask = _mm256_set1_ps(-0.0); // andnot(-0.0, v) clears the sign bit = |v|
    for c in 0..chunks {
        let mut v = widen_bf16(x.as_ptr().add(c * 8));
        if op == RED_MAXABS {
            v = _mm256_andnot_ps(absmask, v);
        }
        acc = if op == RED_MIN {
            _mm256_min_ps(acc, v)
        } else {
            _mm256_max_ps(acc, v) // RED_MAX and the (already-abs'd) RED_MAXABS both fold by max
        };
    }
    let mut lanes = [ident(op); 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = hcombine8(lanes, op);
    for &v in &x[chunks * 8..] {
        s = rfold(s, bf16_bits_to_f32(v), op);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn reduce_minmax_f16_avx(op: i64, x: &[u16]) -> f32 {
    use crate::reduce::{hcombine8, ident, RED_MAXABS, RED_MIN};
    let n = x.len();
    let chunks = n / 8;
    let mut acc = _mm256_set1_ps(ident(op));
    let absmask = _mm256_set1_ps(-0.0);
    for c in 0..chunks {
        let mut v = widen_f16(x.as_ptr().add(c * 8)); // F16C vcvtph2ps, lossless
        if op == RED_MAXABS {
            v = _mm256_andnot_ps(absmask, v);
        }
        acc = if op == RED_MIN {
            _mm256_min_ps(acc, v)
        } else {
            _mm256_max_ps(acc, v)
        };
    }
    let mut lanes = [ident(op); 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = hcombine8(lanes, op);
    for &v in &x[chunks * 8..] {
        s = rfold(s, f16_to_f32(v), op);
    }
    s
}

/// `reduce_i widen(x[i])` over `n` bf16 values for the **max-family** ops — `RED_MAX` (per-tensor
/// max, e.g. the softmax-stability shift), `RED_MIN`, and `RED_MAXABS` (the symmetric-quantization
/// absmax scale a `[bf16]` weight tensor's int8 export needs). f32 result; half the bytes of the f32
/// reduction. The widen is lossless and the folds round nothing, so this is the *exact* reduction of
/// the widened values — the interpreter marshals through this very kernel, so interp == native.
///
/// # Safety
/// `x` must point to `n` readable `u16`.
/// **Aliasing: none constrained** — every operand is only read and the result is returned by
/// value, so the inputs may share storage freely.
#[no_mangle]
pub unsafe extern "C" fn wukong_reduce_bf16(x: *const u16, n: i64, op: i64) -> f32 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(x, n);
    if n <= 0 {
        return crate::reduce::ident(op);
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::avx2() {
        return reduce_minmax_bf16_avx(op, x);
    }
    reduce_minmax_scalar(Half::Bf16, op, x)
}

/// `reduce_i widen(x[i])` over `n` IEEE-f16 values for the max-family ops (RED_MAX/RED_MIN/RED_MAXABS),
/// f32 result. The F16C twin of [`wukong_reduce_bf16`] — widens with `vcvtph2ps` (lossless); the
/// folds round nothing, so it is the exact reduction. Interp marshals through this very kernel.
///
/// # Safety
/// `x` must point to `n` readable `u16`.
/// **Aliasing: none constrained** — every operand is only read and the result is returned by
/// value, so the inputs may share storage freely.
#[no_mangle]
pub unsafe extern "C" fn wukong_reduce_f16(x: *const u16, n: i64, op: i64) -> f32 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(x, n);
    if n <= 0 {
        return crate::reduce::ident(op);
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::f16c() && crate::isa::avx() {
        return reduce_minmax_f16_avx(op, x);
    }
    reduce_minmax_scalar(Half::F16, op, x)
}

// ---- Public C-ABI entry points (f32-accumulated) ----

/// `sum(widen(x[i]))` over `n` IEEE-f16 values (stored as `u16` bits), accumulated in f32.
///
/// # Safety
/// `x` must point to `n` readable `u16`.
/// **Aliasing: none constrained** — every operand is only read and the result is returned by
/// value, so the inputs may share storage freely.
#[no_mangle]
pub unsafe extern "C" fn wukong_sum_f16(x: *const u16, n: i64) -> f32 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(x, n);
    if n <= 0 {
        return 0.0;
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::f16c() && crate::isa::avx() {
        return sum_f16_avx(x);
    }
    sum_scalar_f16(x)
}

/// `sum(widen(x[i]))` over `n` bf16 values (stored as `u16` bits), accumulated in f32.
///
/// # Safety
/// `x` must point to `n` readable `u16`.
/// **Aliasing: none constrained** — every operand is only read and the result is returned by
/// value, so the inputs may share storage freely.
#[no_mangle]
pub unsafe extern "C" fn wukong_sum_bf16(x: *const u16, n: i64) -> f32 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(x, n);
    if n <= 0 {
        return 0.0;
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::avx2() {
        return sum_bf16_avx(x);
    }
    sum_scalar_bf16(x)
}

/// `sum(widen(x[i])·widen(y[i]))` over `n` IEEE-f16 values, accumulated in f32.
///
/// # Safety
/// `x` and `y` must each point to `n` readable `u16`.
/// **Aliasing: none constrained** — every operand is only read and the result is returned by
/// value, so the inputs may share storage freely.
#[no_mangle]
pub unsafe extern "C" fn wukong_dot_f16(x: *const u16, y: *const u16, n: i64) -> f32 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(x, n);
    crate::safety::debug_extent(y, n);
    if n <= 0 {
        return 0.0;
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    let y = std::slice::from_raw_parts(y, n as usize);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::f16c() && crate::isa::fma() {
        return dot_f16_avx(x, y);
    }
    dot_scalar_f16(x, y)
}

/// `sum(widen(x[i])·widen(y[i]))` over `n` bf16 values, accumulated in f32.
///
/// # Safety
/// `x` and `y` must each point to `n` readable `u16`.
/// **Aliasing: none constrained** — every operand is only read and the result is returned by
/// value, so the inputs may share storage freely.
#[no_mangle]
pub unsafe extern "C" fn wukong_dot_bf16(x: *const u16, y: *const u16, n: i64) -> f32 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(x, n);
    crate::safety::debug_extent(y, n);
    if n <= 0 {
        return 0.0;
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    let y = std::slice::from_raw_parts(y, n as usize);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::avx2() && crate::isa::fma() {
        return dot_bf16_avx(x, y);
    }
    dot_scalar_bf16(x, y)
}

// ---- Parallel reductions (`@parallel`): deterministic, bit-identical on any thread count ----

/// Cut `[0, n)` into fixed `crate::reduce::RCHUNK` chunks, reduce each with `per_chunk(lo, hi)` (one of
/// the serial kernels over the sub-range — a pure function of its chunk), then fold the partials in
/// **ascending chunk order** from `ident`. rayon's indexed `collect()` preserves chunk order regardless
/// of which thread computed each, so the result is identical on every call no matter the core count:
/// the native `@parallel` path and the interpreter (which marshals this very function) agree
/// bit-for-bit. NOTE this chunked fold reassociates vs the *flat* whole-array serial kernel
/// (`wukong_sum_bf16` et al.), so a `@parallel` reduction is **not** bit-equal to its non-`@parallel`
/// twin. Unlike these half-precision serial entries, the f32 reduction entries already share the
/// same fixed-chunk decomposition between serial and parallel calls.
#[inline]
fn par_chunk_reduce(
    n: usize,
    ident: f32,
    per_chunk: impl Fn(usize, usize) -> f32 + Sync,
    combine: impl Fn(f32, f32) -> f32,
) -> f32 {
    let nchunks = n.div_ceil(crate::reduce::RCHUNK);
    // With one worker, a parallel map cannot reduce the work. Fold directly without allocating
    // partials or handing off to a pool. Keep every RCHUNK boundary AND the identity-seeded fold:
    // calling the flat serial sibling would change rounding, and returning an uncombined single
    // partial could change signed-zero/NaN behavior for the supplied combine operation.
    if nchunks < 2 || crate::wuk_pool_width() <= 1 {
        let mut acc = ident;
        for c in 0..nchunks {
            let lo = c * crate::reduce::RCHUNK;
            let hi = ((c + 1) * crate::reduce::RCHUNK).min(n);
            acc = combine(acc, per_chunk(lo, hi));
        }
        return acc;
    }
    // Fork on the unified kernel pool, never on rayon's implicit global registry: this is a path that
    // can be a process's FIRST rayon touch (a bf16 model whose first parallel op is a reduction), and
    // `run_on_wuk_pool` is what installs the runtime's 16 MiB worker stacks before anything can build
    // rayon's 2 MiB default (`crate::ensure_global_pool`, lib.rs:324-329). Scheduling-only: the fixed
    // `RCHUNK` decomposition and the ordered fold below are what fix the bits, so which pool runs the
    // map cannot change them. `per_chunk` is borrowed, not moved — `&P` is `Send` because `P: Sync`.
    let per_chunk = &per_chunk;
    // `par::map_units`: partial `c` in slot `c` under either scheduler (`WUKONG_KERNEL_DYN`), so the
    // index-order fold below is unchanged.
    let partials: Vec<f32> = crate::run_on_wuk_pool(move || {
        crate::par::map_units(nchunks, |c| {
            let lo = c * crate::reduce::RCHUNK;
            let hi = ((c + 1) * crate::reduce::RCHUNK).min(n);
            per_chunk(lo, hi)
        })
    });
    let mut acc = ident;
    for p in partials {
        acc = combine(acc, p);
    }
    acc
}

/// Select the ISA before calling this helper, so workers invoke the selected kernel directly.
/// Short inputs retain the public entries' flat result, without an extra identity fold.
#[inline]
fn par_half_reduce(
    x: &[u16],
    ident: f32,
    per_chunk: impl Fn(&[u16]) -> f32 + Sync,
    combine: impl Fn(f32, f32) -> f32,
) -> f32 {
    if x.len() <= crate::reduce::RCHUNK {
        return per_chunk(x);
    }
    par_chunk_reduce(x.len(), ident, |lo, hi| per_chunk(&x[lo..hi]), combine)
}

#[inline]
fn par_half_dot(x: &[u16], y: &[u16], per_chunk: impl Fn(&[u16], &[u16]) -> f32 + Sync) -> f32 {
    if x.len() <= crate::reduce::RCHUNK {
        return per_chunk(x, y);
    }
    par_chunk_reduce(
        x.len(),
        0.0,
        |lo, hi| per_chunk(&x[lo..hi], &y[lo..hi]),
        |a, b| a + b,
    )
}

/// Multicore bf16 sum — the `@parallel` twin of [`wukong_sum_bf16`]. Each fixed chunk is summed by the
/// serial kernel and the partials added ascending, so the result is deterministic (interp == native).
///
/// # Safety
/// `x` must point to `n` readable `u16`.
/// **Aliasing: none constrained** — every operand is only read and the result is returned by
/// value, so the inputs may share storage freely.
#[no_mangle]
pub unsafe extern "C" fn wukong_sum_bf16_parallel(x: *const u16, n: i64) -> f32 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(x, n);
    if n <= 0 {
        return 0.0;
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::avx2() {
        // SAFETY: feature detected once; borrowed chunks are read-only valid subslices.
        return par_half_reduce(x, 0.0, |chunk| sum_bf16_avx(chunk), |a, b| a + b);
    }
    par_half_reduce(x, 0.0, sum_scalar_bf16, |a, b| a + b)
}

/// Multicore IEEE-f16 sum — the `@parallel` twin of [`wukong_sum_f16`].
///
/// # Safety
/// `x` must point to `n` readable `u16`.
/// **Aliasing: none constrained** — every operand is only read and the result is returned by
/// value, so the inputs may share storage freely.
#[no_mangle]
pub unsafe extern "C" fn wukong_sum_f16_parallel(x: *const u16, n: i64) -> f32 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(x, n);
    if n <= 0 {
        return 0.0;
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::f16c() && crate::isa::avx() {
        // SAFETY: features detected once; borrowed chunks are read-only valid subslices.
        return par_half_reduce(x, 0.0, |chunk| sum_f16_avx(chunk), |a, b| a + b);
    }
    par_half_reduce(x, 0.0, sum_scalar_f16, |a, b| a + b)
}

/// Multicore bf16 dot — the `@parallel` twin of [`wukong_dot_bf16`].
///
/// # Safety
/// `x` and `y` must each point to `n` readable `u16`.
/// **Aliasing: none constrained** — every operand is only read and the result is returned by
/// value, so the inputs may share storage freely.
#[no_mangle]
pub unsafe extern "C" fn wukong_dot_bf16_parallel(x: *const u16, y: *const u16, n: i64) -> f32 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(x, n);
    crate::safety::debug_extent(y, n);
    if n <= 0 {
        return 0.0;
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    let y = std::slice::from_raw_parts(y, n as usize);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::avx2() && crate::isa::fma() {
        // SAFETY: features detected once; x/y may overlap because both are read-only.
        return par_half_dot(x, y, |xc, yc| dot_bf16_avx(xc, yc));
    }
    par_half_dot(x, y, dot_scalar_bf16)
}

/// Multicore IEEE-f16 dot — the `@parallel` twin of [`wukong_dot_f16`].
///
/// # Safety
/// `x` and `y` must each point to `n` readable `u16`.
/// **Aliasing: none constrained** — every operand is only read and the result is returned by
/// value, so the inputs may share storage freely.
#[no_mangle]
pub unsafe extern "C" fn wukong_dot_f16_parallel(x: *const u16, y: *const u16, n: i64) -> f32 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(x, n);
    crate::safety::debug_extent(y, n);
    if n <= 0 {
        return 0.0;
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    let y = std::slice::from_raw_parts(y, n as usize);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::f16c() && crate::isa::fma() {
        // SAFETY: features detected once; x/y may overlap because both are read-only.
        return par_half_dot(x, y, |xc, yc| dot_f16_avx(xc, yc));
    }
    par_half_dot(x, y, dot_scalar_f16)
}

/// Multicore bf16 max-family reduction (`RED_MAX`/`RED_MIN`/`RED_MAXABS`) — the `@parallel` twin of
/// [`wukong_reduce_bf16`]. Partials fold with the op's `fold2`/`ident` (the same `Cmp+Select` the
/// recognizer emits), so max/min/absmax stay deterministic over the fixed chunk decomposition.
///
/// # Safety
/// `x` must point to `n` readable `u16`.
/// **Aliasing: none constrained** — every operand is only read and the result is returned by
/// value, so the inputs may share storage freely.
#[no_mangle]
pub unsafe extern "C" fn wukong_reduce_bf16_parallel(x: *const u16, n: i64, op: i64) -> f32 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(x, n);
    if n <= 0 {
        return crate::reduce::ident(op);
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::avx2() {
        // Keep the same operand order within each kernel and in the ordered partial fold.
        return par_half_reduce(
            x,
            crate::reduce::ident(op),
            |chunk| reduce_minmax_bf16_avx(op, chunk),
            |a, b| crate::reduce::fold2(a, b, op),
        );
    }
    par_half_reduce(
        x,
        crate::reduce::ident(op),
        |chunk| reduce_minmax_scalar(Half::Bf16, op, chunk),
        |a, b| crate::reduce::fold2(a, b, op),
    )
}

/// Multicore IEEE-f16 max-family reduction — the `@parallel` twin of [`wukong_reduce_f16`].
///
/// # Safety
/// `x` must point to `n` readable `u16`.
/// **Aliasing: none constrained** — every operand is only read and the result is returned by
/// value, so the inputs may share storage freely.
#[no_mangle]
pub unsafe extern "C" fn wukong_reduce_f16_parallel(x: *const u16, n: i64, op: i64) -> f32 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(x, n);
    if n <= 0 {
        return crate::reduce::ident(op);
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::f16c() && crate::isa::avx() {
        // Keep the same operand order within each kernel and in the ordered partial fold.
        return par_half_reduce(
            x,
            crate::reduce::ident(op),
            |chunk| reduce_minmax_f16_avx(op, chunk),
            |a, b| crate::reduce::fold2(a, b, op),
        );
    }
    par_half_reduce(
        x,
        crate::reduce::ident(op),
        |chunk| reduce_minmax_scalar(Half::F16, op, chunk),
        |a, b| crate::reduce::fold2(a, b, op),
    )
}

/// Scalar twin of the bf16→f32 axpby: `out[i] = a·widen(x[i]) + b·widen(y[i])`. The op order
/// (`t = a·xw` then `fma(b, yw, t)`) is chosen to match the AVX2 path's `mul`+`fmadd` lane-for-lane,
/// so SIMD == scalar bit-for-bit (the twin test pins this); it's also the no-AVX2 fallback.
fn axpby_bf16_scalar(kind: Half, x: &[u16], y: &[u16], out: &mut [f32], a: f32, b: f32) {
    for i in 0..out.len() {
        let xw = kind.widen(x[i]);
        let yw = kind.widen(y[i]);
        out[i] = b.mul_add(yw, a * xw); // b·yw + a·xw, the same op order as the fmadd path
    }
}

/// Compare the differently sized byte ranges without forming references or end pointers.
/// Even equal input/output addresses have a cross-element dependency when stores widen.
pub(crate) fn half_input_overlaps_f32_output(x: *const u16, out: *mut f32, n: usize) -> bool {
    let (input, output) = (x as usize, out as usize);
    if input <= output {
        output - input < n.saturating_mul(std::mem::size_of::<u16>())
    } else {
        input - output < n.saturating_mul(std::mem::size_of::<f32>())
    }
}

/// Read both current half values before the widening store, then advance one element.
/// Later reads deliberately observe bytes overwritten by earlier stores on overlap.
unsafe fn axpby_widen_overlapping(
    kind: Half,
    x: *const u16,
    y: *const u16,
    out: *mut f32,
    n: usize,
    a: f32,
    b: f32,
) {
    for i in 0..n {
        let xw = kind.widen(*x.add(i));
        let yw = kind.widen(*y.add(i));
        *out.add(i) = b.mul_add(yw, a * xw);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpby_bf16_avx(x: &[u16], y: &[u16], out: &mut [f32], a: f32, b: f32) {
    let n = out.len();
    let chunks = n / 8;
    let av = _mm256_set1_ps(a);
    let bv = _mm256_set1_ps(b);
    for c in 0..chunks {
        let xw = widen_bf16(x.as_ptr().add(c * 8));
        let yw = widen_bf16(y.as_ptr().add(c * 8));
        let t = _mm256_mul_ps(av, xw);
        let r = _mm256_fmadd_ps(bv, yw, t); // b·yw + a·xw
        _mm256_storeu_ps(out.as_mut_ptr().add(c * 8), r);
    }
    for i in chunks * 8..n {
        let xw = bf16_bits_to_f32(x[i]);
        let yw = bf16_bits_to_f32(y[i]);
        out[i] = b.mul_add(yw, a * xw);
    }
}

/// `out[i] = a·widen(x[i]) + b·widen(y[i])` over `n` **bf16** inputs (stored as `u16` bits) with an
/// **f32 output** — the mixed-precision streaming elementwise (saxpy/axpby) twin of the f32
/// `wukong_velem_f32`. bf16 in + f32 out streams 8 bytes/elem vs the f32 kernel's 12, so on a
/// memory-bound elementwise it runs ~1.5× faster. Compute is f32 (widen is lossless), so the only
/// rounding is the f32 math — identical in the AVX2 and scalar paths (twin-tested).
/// Overlapping output follows ascending element order: each step reads both half values before
/// writing its f32 result. Widening stores may overwrite subsequent inputs; no snapshot is taken.
///
/// # Safety
/// `x`/`y` must point to `n` readable `u16`; `out` to `n` writable `f32`.
/// **Aliasing: all operands disjoint** — the f32 `out` overlaps neither half-precision input (any
/// overlap is a mixed-width pun the source loop does not perform).
#[no_mangle]
pub unsafe extern "C" fn wukong_axpby_bf16(
    x: *const u16,
    y: *const u16,
    out: *mut f32,
    n: i64,
    a: f32,
    b: f32,
) {
    // P14.5: debug-build contract checks (the `# Safety` section above); no aliasing assert: the
    // kernel detects a mixed-width overlap itself and defines it (`half_input_overlaps_f32_output`
    // -> `axpby_widen_overlapping`, pinned by `mixed_width_overlap_matches_live_byte_reference`),
    // so an overlap is not UB here.
    crate::safety::debug_extent(x, n);
    crate::safety::debug_extent(y, n);
    crate::safety::debug_extent(out, n);
    if n <= 0 {
        return;
    }
    let n = n as usize;
    if half_input_overlaps_f32_output(x, out, n) || half_input_overlaps_f32_output(y, out, n) {
        return axpby_widen_overlapping(Half::Bf16, x, y, out, n, a, b);
    }
    let x = std::slice::from_raw_parts(x, n);
    let y = std::slice::from_raw_parts(y, n);
    let out = std::slice::from_raw_parts_mut(out, n);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::avx2() && crate::isa::fma() {
        return axpby_bf16_avx(x, y, out, a, b);
    }
    axpby_bf16_scalar(Half::Bf16, x, y, out, a, b);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn axpby_f16_avx(x: &[u16], y: &[u16], out: &mut [f32], a: f32, b: f32) {
    let n = out.len();
    let chunks = n / 8;
    let av = _mm256_set1_ps(a);
    let bv = _mm256_set1_ps(b);
    for c in 0..chunks {
        let xw = widen_f16(x.as_ptr().add(c * 8)); // F16C vcvtph2ps
        let yw = widen_f16(y.as_ptr().add(c * 8));
        let t = _mm256_mul_ps(av, xw);
        let r = _mm256_fmadd_ps(bv, yw, t); // b·yw + a·xw, same op order as the scalar twin
        _mm256_storeu_ps(out.as_mut_ptr().add(c * 8), r);
    }
    for i in chunks * 8..n {
        let xw = f16_to_f32(x[i]);
        let yw = f16_to_f32(y[i]);
        out[i] = b.mul_add(yw, a * xw);
    }
}

/// `out[i] = a·widen(x[i]) + b·widen(y[i])` over `n` **IEEE-f16** inputs with an **f32 output** — the
/// F16C twin of [`wukong_axpby_bf16`] (widens with `vcvtph2ps`). 8 bytes/elem vs an all-f32 axpby's
/// 12; the only rounding is the f32 math, identical in the AVX2 and scalar paths (twin-tested).
/// Overlap follows the same ascending read-before-write semantics as [`wukong_axpby_bf16`].
///
/// # Safety
/// `x`/`y` must point to `n` readable `u16`; `out` to `n` writable `f32`.
/// **Aliasing: all operands disjoint** — the f32 `out` overlaps neither half-precision input (any
/// overlap is a mixed-width pun the source loop does not perform).
#[no_mangle]
pub unsafe extern "C" fn wukong_axpby_f16(
    x: *const u16,
    y: *const u16,
    out: *mut f32,
    n: i64,
    a: f32,
    b: f32,
) {
    // P14.5: debug-build contract checks (the `# Safety` section above); no aliasing assert: the
    // kernel detects a mixed-width overlap itself and defines it (`half_input_overlaps_f32_output`
    // -> `axpby_widen_overlapping`, pinned by `mixed_width_overlap_matches_live_byte_reference`),
    // so an overlap is not UB here.
    crate::safety::debug_extent(x, n);
    crate::safety::debug_extent(y, n);
    crate::safety::debug_extent(out, n);
    if n <= 0 {
        return;
    }
    let n = n as usize;
    if half_input_overlaps_f32_output(x, out, n) || half_input_overlaps_f32_output(y, out, n) {
        return axpby_widen_overlapping(Half::F16, x, y, out, n, a, b);
    }
    let x = std::slice::from_raw_parts(x, n);
    let y = std::slice::from_raw_parts(y, n);
    let out = std::slice::from_raw_parts_mut(out, n);
    #[cfg(target_arch = "x86_64")]
    if crate::isa::f16c() && crate::isa::avx2() && crate::isa::fma() {
        return axpby_f16_avx(x, y, out, a, b);
    }
    axpby_bf16_scalar(Half::F16, x, y, out, a, b);
}

// ---- Half-precision **output** axpby: `out[i] = a·widen(x[i]) + b·widen(y[i])` narrowed to bf16/f16.
//      The all-half twin of the bf16-in/f32-out axpby above — bf16/f16 in AND out streams 6 bytes/elem
//      vs an all-f32 axpby's 12, so on a memory-bound elementwise it runs ~2× faster (the narrowing
//      store halves the write traffic that dominates). Compute is f32 (widen is lossless), and the only
//      new rounding is the narrowing store, which goes through the shared `Half::narrow` shim (== the
//      interpreter's), so interp == native bit-for-bit and `-O0` == `-O2`. ----

/// Scalar twin / no-AVX2 fallback: `out[i] = narrow(b·yw + a·xw)`, the same op order (`t = a·xw` then
/// `fma(b, yw, t)`) and narrowing shim as the AVX2 path, so SIMD == scalar bit-for-bit.
unsafe fn axpby_narrow_scalar(
    kind: Half,
    x: *const u16,
    y: *const u16,
    out: *mut u16,
    n: usize,
    a: f32,
    b: f32,
) {
    for i in 0..n {
        let xw = kind.widen(*x.add(i));
        let yw = kind.widen(*y.add(i));
        *out.add(i) = kind.narrow(b.mul_add(yw, a * xw));
    }
}

/// Exact in-place inputs have no cross-element dependency. Shifted overlap must retain
/// ascending scalar reads/writes; a vector load would otherwise snapshot future inputs.
fn axpby_narrow_shifted_overlap(x: *const u16, y: *const u16, out: *mut u16, n: usize) -> bool {
    let bytes = n.saturating_mul(std::mem::size_of::<u16>());
    [x, y].into_iter().any(|p| {
        let distance = (p as usize).abs_diff(out as usize);
        distance != 0 && distance < bytes
    })
}

/// Preserve the all-half traffic model: x/y reads plus output write total six bytes per element.
#[inline]
fn use_nt_halfout(n: usize) -> bool {
    n.saturating_mul(6) >= crate::nt_policy::nt_min_bytes()
}

/// Elements ahead to software-prefetch the `x`/`y` reads in the streaming path. The output is
/// non-temporal (not prefetched — we never read it back here); the two half-width input streams are the
/// DRAM-read-bound side, so pulling them in a few lines early hides the miss latency. The last few
/// iterations of every bulk loop deliberately address past the end of `x`/`y` — a prefetch past the
/// buffer end is a hint the hardware drops, so no end guard is needed. That is a statement about the
/// *hardware*; the Rust-level obligation is separate, so the prefetch sites form the address with
/// `wrapping_add`, NOT `add`: `<*const T>::add` requires its result to stay inside (or one past) the
/// same allocated object, which this by construction does not. `wrapping_add` carries no such
/// precondition and lowers to the identical address arithmetic; the pointer is only ever handed to
/// `_mm_prefetch`, which never dereferences it.
const HALFOUT_PF_AHEAD: usize = 256;

/// The streaming-narrow skeleton shared by the half-output kernels (axpby, activations): write
/// `out[i] = narrow(⟨op8 i⟩)` for `i in 0..n`. The bulk runs **32 lanes/iter** — 4 independent 8-lane
/// chains packed into 2 stores — and when `6·n` spills L3 the stores are **non-temporal** (`vmovntdq`,
/// with a scalar alignment prologue + `sfence`), else cacheable; an 8-lane + scalar tail finishes.
/// `$op8(i)` yields the 8 f32 lanes at element `i` (reads its own inputs); `$sc(i)` the scalar `u16`
/// (prologue/tail); `$pf(i)` software-prefetches the inputs ahead; `$pack`/`$narrow8` are the
/// precision's 16-/8-lane narrowing stores. Expanded *inside* the caller's `#[target_feature]` fn so
/// the intrinsics and closures share its feature context (the same idiom as `bias.rs`'s `run!`).
macro_rules! stream_narrow {
    ($outp:expr, $n:expr, $op8:expr, $sc:expr, $pf:expr, $pack:expr, $narrow8:expr) => {{
        let outp = $outp;
        let n = $n;
        let op8 = $op8;
        let sc = $sc;
        let pf = $pf;
        let mut i = 0usize;
        if use_nt_halfout(n) {
            // Scalar prologue until `out` is 32-byte aligned (`vmovntdq` faults otherwise).
            while i < n && (outp.add(i) as usize) & 31 != 0 {
                *outp.add(i) = sc(i);
                i += 1;
            }
            while i + 32 <= n {
                pf(i);
                let p0 = $pack(op8(i), op8(i + 8));
                let p1 = $pack(op8(i + 16), op8(i + 24));
                crate::velem::assert_nt_aligned(outp.add(i) as *mut __m256i);
                _mm256_stream_si256(outp.add(i) as *mut __m256i, p0);
                crate::velem::assert_nt_aligned(outp.add(i + 16) as *mut __m256i);
                _mm256_stream_si256(outp.add(i + 16) as *mut __m256i, p1);
                i += 32;
            }
            while i + 16 <= n {
                crate::velem::assert_nt_aligned(outp.add(i) as *mut __m256i);
                _mm256_stream_si256(outp.add(i) as *mut __m256i, $pack(op8(i), op8(i + 8)));
                i += 16;
            }
            _mm_sfence(); // NT stores are weakly ordered; fence before the buffer is read back.
        } else {
            // L3-resident: cacheable 256-bit stores (no NT — that would evict useful lines).
            while i + 16 <= n {
                _mm256_storeu_si256(outp.add(i) as *mut __m256i, $pack(op8(i), op8(i + 8)));
                i += 16;
            }
        }
        while i + 8 <= n {
            $narrow8(op8(i), outp.add(i));
            i += 8;
        }
        for j in i..n {
            *outp.add(j) = sc(j);
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpby_narrow_bf16_avx(
    xp: *const u16,
    yp: *const u16,
    out: *mut u16,
    n: usize,
    a: f32,
    b: f32,
) {
    let (av, bv) = (_mm256_set1_ps(a), _mm256_set1_ps(b));
    // 8-lane axpby `b·widen(y) + a·widen(x)` (same op order / FMA as the scalar tail).
    stream_narrow!(
        out,
        n,
        |i: usize| _mm256_fmadd_ps(
            bv,
            widen_bf16(yp.add(i)),
            _mm256_mul_ps(av, widen_bf16(xp.add(i)))
        ),
        |i: usize| crate::f32_to_bf16_bits(b.mul_add(
            bf16_bits_to_f32(*yp.add(i)),
            a * bf16_bits_to_f32(*xp.add(i))
        )),
        |i: usize| {
            _mm_prefetch(
                xp.wrapping_add(i + HALFOUT_PF_AHEAD) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                yp.wrapping_add(i + HALFOUT_PF_AHEAD) as *const i8,
                _MM_HINT_T0,
            );
        },
        narrow_bf16_pack,
        narrow_bf16
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn axpby_narrow_f16_avx(
    xp: *const u16,
    yp: *const u16,
    out: *mut u16,
    n: usize,
    a: f32,
    b: f32,
) {
    let (av, bv) = (_mm256_set1_ps(a), _mm256_set1_ps(b));
    stream_narrow!(
        out,
        n,
        |i: usize| _mm256_fmadd_ps(
            bv,
            widen_f16(yp.add(i)),
            _mm256_mul_ps(av, widen_f16(xp.add(i)))
        ),
        |i: usize| crate::f32_to_f16_bits(
            b.mul_add(f16_to_f32(*yp.add(i)), a * f16_to_f32(*xp.add(i)))
        ),
        |i: usize| {
            _mm_prefetch(
                xp.wrapping_add(i + HALFOUT_PF_AHEAD) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                yp.wrapping_add(i + HALFOUT_PF_AHEAD) as *const i8,
                _MM_HINT_T0,
            );
        },
        narrow_f16_pack,
        narrow_f16
    );
}

/// Scalar twin / no-AVX2 fallback for the half-output activation: `out[i] = narrow(act(widen(x[i])))`,
/// the same shared `apply1` activation and narrowing shim as the AVX2 lanes, so SIMD == scalar.
unsafe fn vmath_narrow_scalar(kind: Half, x: *const u16, out: *mut u16, n: usize, op: i64) {
    for i in 0..n {
        *out.add(i) = kind.narrow(crate::vmath::apply1(op, kind.widen(*x.add(i))));
    }
}

/// AVX2 dispatcher for [`wukong_vmath_bf16_out`]: [`crate::vmath::with_act8`] instantiates
/// [`vmath_narrow_bf16_loop`] once per activation kernel with the polynomial inlined — it used to call
/// a `vmath8_for` function pointer per 8-lane vector, four times per unrolled step of the streaming
/// skeleton (P9.3). An op with no 8-lane kernel runs the scalar twin for every element.
///
/// # Safety
/// `xp` valid for `n` `u16`, `out` for `n` `u16`; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vmath_narrow_bf16_avx(xp: *const u16, out: *mut u16, n: usize, op: i64) {
    struct Body {
        xp: *const u16,
        out: *mut u16,
        n: usize,
        op: i64,
    }
    impl crate::vmath::Act8Body for Body {
        unsafe fn run<F: Fn(__m256) -> __m256>(self, f: F) {
            // SAFETY: the dispatcher's contract — buffers valid for `n`, AVX2+FMA checked.
            unsafe { vmath_narrow_bf16_loop(self.xp, self.out, self.n, self.op, f) }
        }
    }
    if !crate::vmath::with_act8(op, Body { xp, out, n, op }) {
        vmath_narrow_scalar(Half::Bf16, xp, out, n, op);
    }
}

/// The bf16 half-output activation stream for one kernel `f` (the [`stream_narrow`] skeleton).
///
/// # Safety
/// `xp` valid for `n` `u16`, `out` for `n` `u16`; AVX2+FMA available; `f` an avx2+fma kernel equal
/// lane-for-lane to `apply1(op, ·)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vmath_narrow_bf16_loop<F: Fn(__m256) -> __m256>(
    xp: *const u16,
    out: *mut u16,
    n: usize,
    op: i64,
    f: F,
) {
    stream_narrow!(
        out,
        n,
        |i: usize| f(widen_bf16(xp.add(i))),
        |i: usize| crate::f32_to_bf16_bits(crate::vmath::apply1(op, bf16_bits_to_f32(*xp.add(i)))),
        |i: usize| _mm_prefetch(
            xp.wrapping_add(i + HALFOUT_PF_AHEAD) as *const i8,
            _MM_HINT_T0
        ),
        narrow_bf16_pack,
        narrow_bf16
    );
}

/// AVX2/F16C dispatcher for [`wukong_vmath_f16_out`] — [`vmath_narrow_bf16_avx`]'s shape with the
/// F16C widen and narrow.
///
/// # Safety
/// `xp` valid for `n` `u16`, `out` for `n` `u16`; AVX2+F16C+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn vmath_narrow_f16_avx(xp: *const u16, out: *mut u16, n: usize, op: i64) {
    struct Body {
        xp: *const u16,
        out: *mut u16,
        n: usize,
        op: i64,
    }
    impl crate::vmath::Act8Body for Body {
        unsafe fn run<F: Fn(__m256) -> __m256>(self, f: F) {
            // SAFETY: the dispatcher's contract — buffers valid for `n`, AVX2+F16C+FMA checked.
            unsafe { vmath_narrow_f16_loop(self.xp, self.out, self.n, self.op, f) }
        }
    }
    if !crate::vmath::with_act8(op, Body { xp, out, n, op }) {
        vmath_narrow_scalar(Half::F16, xp, out, n, op);
    }
}

/// The f16 half-output activation stream for one kernel `f` (the [`stream_narrow`] skeleton).
///
/// # Safety
/// `xp` valid for `n` `u16`, `out` for `n` `u16`; AVX2+F16C+FMA available; `f` an avx2+fma kernel
/// equal lane-for-lane to `apply1(op, ·)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn vmath_narrow_f16_loop<F: Fn(__m256) -> __m256>(
    xp: *const u16,
    out: *mut u16,
    n: usize,
    op: i64,
    f: F,
) {
    stream_narrow!(
        out,
        n,
        |i: usize| f(widen_f16(xp.add(i))),
        |i: usize| crate::f32_to_f16_bits(crate::vmath::apply1(op, f16_to_f32(*xp.add(i)))),
        |i: usize| _mm_prefetch(
            xp.wrapping_add(i + HALFOUT_PF_AHEAD) as *const i8,
            _MM_HINT_T0
        ),
        narrow_f16_pack,
        narrow_f16
    );
}

/// `out[i] = round_bf16(a·widen(x[i]) + b·widen(y[i]))` over `n` **bf16** inputs with a **bf16 output**
/// — the all-half streaming axpby. 6 bytes/elem vs an all-f32 axpby's 12; the compute is f32 (lossless
/// widen) and the store rounds through the shared `f32_to_bf16_bits` shim, so interp == native.
/// Overlap is supported with ascending element-order semantics, including exact in-place output.
///
/// # Safety
/// `x`/`y` must point to `n` readable `u16`; `out` to `n` writable `u16`.
/// **Aliasing: exactly in place, otherwise disjoint** — `out` may be EXACTLY `x` or `y`; the
/// compiler dispatches no other overlap.
#[no_mangle]
pub unsafe extern "C" fn wukong_axpby_bf16_out(
    x: *const u16,
    y: *const u16,
    out: *mut u16,
    n: i64,
    a: f32,
    b: f32,
) {
    // P14.5: debug-build contract checks (the `# Safety` section above); no aliasing assert: a
    // shifted overlap is detected and defined in-kernel (`axpby_narrow_shifted_overlap`, pinned by
    // `axpby_narrow_overlap_preserves_ascending_reads_and_writes`).
    crate::safety::debug_extent(x, n);
    crate::safety::debug_extent(y, n);
    crate::safety::debug_extent(out, n);
    if n <= 0 {
        return;
    }
    let n = n as usize;
    if axpby_narrow_shifted_overlap(x, y, out, n) {
        return axpby_narrow_scalar(Half::Bf16, x, y, out, n, a, b);
    }
    #[cfg(target_arch = "x86_64")]
    if crate::isa::avx2() && crate::isa::fma() {
        return axpby_narrow_bf16_avx(x, y, out, n, a, b);
    }
    axpby_narrow_scalar(Half::Bf16, x, y, out, n, a, b);
}

/// `out[i] = round_f16(a·widen(x[i]) + b·widen(y[i]))` over `n` **f16** inputs with an **f16 output** —
/// the F16C twin of [`wukong_axpby_bf16_out`] (widen `vcvtph2ps`, narrow `vcvtps2ph`). 6 bytes/elem.
/// Overlap is supported with ascending element-order semantics, including exact in-place output.
///
/// # Safety
/// `x`/`y` must point to `n` readable `u16`; `out` to `n` writable `u16`.
/// **Aliasing: exactly in place, otherwise disjoint** — `out` may be EXACTLY `x` or `y`; the
/// compiler dispatches no other overlap.
#[no_mangle]
pub unsafe extern "C" fn wukong_axpby_f16_out(
    x: *const u16,
    y: *const u16,
    out: *mut u16,
    n: i64,
    a: f32,
    b: f32,
) {
    // P14.5: debug-build contract checks (the `# Safety` section above); no aliasing assert: a
    // shifted overlap is detected and defined in-kernel (`axpby_narrow_shifted_overlap`, pinned by
    // `axpby_narrow_overlap_preserves_ascending_reads_and_writes`).
    crate::safety::debug_extent(x, n);
    crate::safety::debug_extent(y, n);
    crate::safety::debug_extent(out, n);
    if n <= 0 {
        return;
    }
    let n = n as usize;
    if axpby_narrow_shifted_overlap(x, y, out, n) {
        return axpby_narrow_scalar(Half::F16, x, y, out, n, a, b);
    }
    #[cfg(target_arch = "x86_64")]
    if crate::isa::f16c() && crate::isa::avx2() && crate::isa::fma() {
        return axpby_narrow_f16_avx(x, y, out, n, a, b);
    }
    axpby_narrow_scalar(Half::F16, x, y, out, n, a, b);
}

/// `out[i] = round_bf16(act(widen(x[i])))` over `n` **bf16** inputs with a **bf16 output** (`op` a
/// `crate::vmath::VM_*` activation code) — a **half-output activation**: the transformer
/// activation-store / KV-cache-write path computes in f32 but stores bf16. 4 bytes/elem (2 in + 2 out)
/// vs the half-in/f32-out activation's 6 and the all-f32's 8, so on a memory-bound activation the
/// narrowing store is the bandwidth win (the >L3 non-temporal path via [`stream_narrow`]). `act` is the
/// shared `vmath` kernel and the store the shared `f32_to_bf16_bits` shim, so it equals a scalar
/// `apply1` + narrow lane-for-lane — interp == native, `-O0` == `-O2`.
/// Overlap is supported with ascending element-order semantics, including exact in-place output.
///
/// # Safety
/// `x` must point to `n` readable `u16` (bf16 bits); `out` to `n` writable `u16`.
/// **Aliasing: exactly in place, otherwise disjoint** — `out` may be EXACTLY `x`; the compiler
/// dispatches no other overlap.
#[no_mangle]
pub unsafe extern "C" fn wukong_vmath_bf16_out(x: *const u16, out: *mut u16, n: i64, op: i64) {
    // P14.5: debug-build contract checks (the `# Safety` section above); no aliasing assert: a
    // shifted overlap is detected and defined in-kernel (`axpby_narrow_shifted_overlap`, pinned by
    // `vmath_narrow_overlap_preserves_ascending_reads_and_writes`).
    crate::safety::debug_extent(x, n);
    crate::safety::debug_extent(out, n);
    if n <= 0 {
        return;
    }
    let n = n as usize;
    if axpby_narrow_shifted_overlap(x, x, out, n) {
        return vmath_narrow_scalar(Half::Bf16, x, out, n, op);
    }
    #[cfg(target_arch = "x86_64")]
    if crate::isa::avx2() && crate::isa::fma() {
        return vmath_narrow_bf16_avx(x, out, n, op);
    }
    vmath_narrow_scalar(Half::Bf16, x, out, n, op);
}

/// `out[i] = round_f16(act(widen(x[i])))` over `n` **f16** inputs with an **f16 output** — the F16C twin
/// of [`wukong_vmath_bf16_out`] (widen `vcvtph2ps`, narrow `vcvtps2ph`). 4 bytes/elem.
/// Overlap is supported with ascending element-order semantics, including exact in-place output.
///
/// # Safety
/// `x` must point to `n` readable `u16` (f16 bits); `out` to `n` writable `u16`.
/// **Aliasing: exactly in place, otherwise disjoint** — `out` may be EXACTLY `x`; the compiler
/// dispatches no other overlap.
#[no_mangle]
pub unsafe extern "C" fn wukong_vmath_f16_out(x: *const u16, out: *mut u16, n: i64, op: i64) {
    // P14.5: debug-build contract checks (the `# Safety` section above); no aliasing assert: a
    // shifted overlap is detected and defined in-kernel (`axpby_narrow_shifted_overlap`, pinned by
    // `vmath_narrow_overlap_preserves_ascending_reads_and_writes`).
    crate::safety::debug_extent(x, n);
    crate::safety::debug_extent(out, n);
    if n <= 0 {
        return;
    }
    let n = n as usize;
    if axpby_narrow_shifted_overlap(x, x, out, n) {
        return vmath_narrow_scalar(Half::F16, x, out, n, op);
    }
    #[cfg(target_arch = "x86_64")]
    if crate::isa::f16c() && crate::isa::avx2() && crate::isa::fma() {
        return vmath_narrow_f16_avx(x, out, n, op);
    }
    vmath_narrow_scalar(Half::F16, x, out, n, op);
}

#[cfg(test)]
mod tests {
    // See vmath.rs: the release-only probes guard with a deliberately constant
    // `assert!(!cfg!(debug_assertions), …)` whose message says how to rerun.
    #![allow(clippy::assertions_on_constants)]

    use super::*;

    #[test]
    fn boundary_extents_are_safe() {
        // P14.6: bf16 bits 0x3F80 == 1.0; axpby with a=1,b=0 copies x; reduce sums to n.
        let n = 3usize;
        let x = vec![0x3F80u16; n];
        let y = vec![0u16; n];
        let mut out = vec![f32::from_bits(0x7fc0_0001); n];
        // SAFETY: `x`/`y`/`out` sized n; a nonpositive n is a no-op.
        unsafe {
            wukong_axpby_bf16(x.as_ptr(), y.as_ptr(), out.as_mut_ptr(), 0, 1.0, 0.0);
        }
        assert!(
            out.iter().all(|v| v.to_bits() == 0x7fc0_0001),
            "nonpositive n must not write out"
        );
        // SAFETY: `x`/`y`/`out` sized n; the positive call writes out.
        unsafe {
            wukong_axpby_bf16(x.as_ptr(), y.as_ptr(), out.as_mut_ptr(), n as i64, 1.0, 0.0);
        }
        assert_eq!(out, vec![1.0, 1.0, 1.0]);
        // SAFETY: `x` holds n bf16 bits; reduce reads n.
        let s = unsafe { wukong_reduce_bf16(x.as_ptr(), n as i64, crate::reduce::RED_SUM) };
        assert_eq!(s, 3.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn nt_alignment_narrow_peel() {
        if !(crate::isa::avx2() && crate::isa::fma() && crate::isa::f16c()) {
            return;
        }
        let n = crate::nt_policy::test_min_bytes() / 6 + 35;
        assert!(use_nt_halfout(n));
        for (kernel, value) in [
            (
                wukong_axpby_bf16_out
                    as unsafe extern "C" fn(*const u16, *const u16, *mut u16, i64, f32, f32),
                bf16_bits(2.0),
            ),
            (wukong_axpby_f16_out, f16_bits(2.0)),
        ] {
            let x = vec![value; n];
            // SAFETY: `check_nt_output` supplies an out of n f32; `x` holds n bf16 bits.
            crate::velem::check_nt_output(n, 0xdead, value, |out| unsafe {
                kernel(x.as_ptr(), x.as_ptr(), out, n as i64, 0.5, 0.5);
            });
        }
    }

    fn bf16_bits(x: f32) -> u16 {
        crate::f32_to_bf16_bits(x)
    }
    fn f16_bits(x: f32) -> u16 {
        half::f16::from_f32(x).to_bits()
    }

    #[test]
    fn simd_equals_scalar_twin_bit_for_bit() {
        // Every shape of partial step the bf16 kernels can take: an empty input, a pure ≤15-element
        // scalar tail, each `rest ∈ 0..BF_H` count of leftover 16-element groups (n/16 mod 4), and
        // each of those with a tail on top — plus the f16 kernels' non-multiple-of-8 tails.
        for n in [
            0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 79, 80, 95, 100,
            127, 128, 129, 1000, 4099,
        ] {
            let xs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013 - 3.1).sin()).collect();
            let ys: Vec<f32> = (0..n).map(|i| (i as f32 * 0.019 + 1.7).cos()).collect();
            let xf16: Vec<u16> = xs.iter().map(|&v| f16_bits(v)).collect();
            let yf16: Vec<u16> = ys.iter().map(|&v| f16_bits(v)).collect();
            let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
            let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();

            // SAFETY: the bf16 inputs and out buffer are sized n for the axpby.
            unsafe {
                assert_eq!(
                    wukong_sum_f16(xf16.as_ptr(), n as i64).to_bits(),
                    sum_scalar_f16(&xf16).to_bits(),
                    "sum_f16 n={n}"
                );
                assert_eq!(
                    wukong_sum_bf16(xbf.as_ptr(), n as i64).to_bits(),
                    sum_scalar_bf16(&xbf).to_bits(),
                    "sum_bf16 n={n}"
                );
                assert_eq!(
                    wukong_dot_f16(xf16.as_ptr(), yf16.as_ptr(), n as i64).to_bits(),
                    dot_scalar_f16(&xf16, &yf16).to_bits(),
                    "dot_f16 n={n}"
                );
                assert_eq!(
                    wukong_dot_bf16(xbf.as_ptr(), ybf.as_ptr(), n as i64).to_bits(),
                    dot_scalar_bf16(&xbf, &ybf).to_bits(),
                    "dot_bf16 n={n}"
                );
            }
        }
    }

    /// The f16 `sum`/`dot` partition, pinned from both sides.
    ///
    /// (a) Below one 64-element block the blocked body never runs, so both kernels must return the
    ///     exact bits of the single-accumulator form they replaced — spelled out independently here
    ///     (lane `i % 8`, then the lane combine, then the ordered tail).
    /// (b) At 2^20 elements of the `wukong_xbench`-style input the eight-chain partition is clearly
    ///     closer to the f64 sum than one chain. Checked offline with an exact f32 emulation of both
    ///     partitions: relative error 2.84e-5 against the one-chain 1.60e-4, so the factor-of-two
    ///     margin asserted has ~2.8x of slack.
    #[test]
    fn f16_reductions_keep_short_bits_and_beat_one_chain() {
        let one_chain_sum = |x: &[u16]| {
            let mut l = [0.0f32; 8];
            let c = x.len() / 8;
            for k in 0..c {
                for (j, lj) in l.iter_mut().enumerate() {
                    *lj += f16_to_f32(x[k * 8 + j]);
                }
            }
            let mut s = hcombine8(&l);
            for &v in &x[c * 8..] {
                s += f16_to_f32(v);
            }
            s
        };
        let one_chain_dot = |x: &[u16], y: &[u16]| {
            let mut l = [0.0f32; 8];
            let c = x.len() / 8;
            for k in 0..c {
                for (j, lj) in l.iter_mut().enumerate() {
                    *lj = f16_to_f32(x[k * 8 + j]).mul_add(f16_to_f32(y[k * 8 + j]), *lj);
                }
            }
            let mut s = hcombine8(&l);
            for k in c * 8..x.len() {
                s = f16_to_f32(x[k]).mul_add(f16_to_f32(y[k]), s);
            }
            s
        };
        for n in 0usize..F16_BLOCK {
            let x: Vec<u16> = (0..n)
                .map(|i| f16_bits((i as f32 * 0.37 - 2.1).sin() * 3.0))
                .collect();
            let y: Vec<u16> = (0..n)
                .map(|i| f16_bits((i as f32 * 0.23 + 0.4).cos()))
                .collect();
            // SAFETY: the f16 inputs and out buffer are sized n.
            unsafe {
                assert_eq!(
                    wukong_sum_f16(x.as_ptr(), n as i64).to_bits(),
                    one_chain_sum(&x).to_bits(),
                    "sum_f16 short n={n}"
                );
                assert_eq!(
                    wukong_dot_f16(x.as_ptr(), y.as_ptr(), n as i64).to_bits(),
                    one_chain_dot(&x, &y).to_bits(),
                    "dot_f16 short n={n}"
                );
            }
        }

        let n = 1usize << 20;
        let x: Vec<u16> = (0..n)
            .map(|i| f16_bits((i % 17) as f32 * 0.05 + 0.5))
            .collect();
        let want: f64 = x.iter().map(|&v| f16_to_f32(v) as f64).sum();
        // SAFETY: `x` holds n f16 bits; wukong_sum_f16 reads n.
        let ours = unsafe { wukong_sum_f16(x.as_ptr(), n as i64) };
        let old = one_chain_sum(&x);
        let e_ours = (ours as f64 - want).abs() / want;
        let e_old = (old as f64 - want).abs() / want;
        assert!(
            e_ours * 2.0 <= e_old,
            "f16 sum: partition rel err {e_ours:.3e} not clearly below one chain {e_old:.3e}"
        );
    }

    #[test]
    fn reduce_bf16_simd_equals_scalar_and_reference() {
        use crate::reduce::{ident, RED_MAX, RED_MAXABS, RED_MIN};
        // fold matching `reduce.rs::fold2` exactly, so the reference picks the same element on ties.
        let fold = |a: f32, v: f32, op: i64| -> f32 {
            match op {
                RED_MIN => {
                    if a < v {
                        a
                    } else {
                        v
                    }
                }
                _ => {
                    if a > v {
                        a
                    } else {
                        v
                    }
                }
            }
        };
        for n in [0usize, 1, 7, 8, 9, 100, 1000, 4099] {
            let xs: Vec<f32> = (0..n)
                .map(|i| (i as f32 * 0.013 - 7.0).sin() * 3.0)
                .collect();
            let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
            let xf16: Vec<u16> = xs.iter().map(|&v| f16_bits(v)).collect();
            for op in [RED_MAX, RED_MIN, RED_MAXABS] {
                for (kind, xb, got) in [
                    // SAFETY: `xbf` holds n bf16 bits; reduce reads n.
                    (Half::Bf16, &xbf, unsafe {
                        wukong_reduce_bf16(xbf.as_ptr(), n as i64, op)
                    }),
                    // SAFETY: `xf16` holds n f16 bits; reduce reads n.
                    (Half::F16, &xf16, unsafe {
                        wukong_reduce_f16(xf16.as_ptr(), n as i64, op)
                    }),
                ] {
                    // AVX2/F16C == scalar twin, bit-for-bit.
                    let twin = reduce_minmax_scalar(kind, op, xb);
                    assert_eq!(got.to_bits(), twin.to_bits(), "twin op {op} n {n}");
                    // == the exact reduction of the widened values (max/min round nothing).
                    let mut want = ident(op);
                    for &b in xb {
                        let v = kind.widen(b);
                        let v = if op == RED_MAXABS { v.abs() } else { v };
                        want = fold(want, v, if op == RED_MAXABS { RED_MAX } else { op });
                    }
                    assert_eq!(got.to_bits(), want.to_bits(), "ref op {op} n {n}");
                }
            }
        }
    }

    #[test]
    fn axpby_lowp_simd_equals_scalar_twin() {
        for n in [0usize, 1, 7, 8, 9, 100, 1000, 4099] {
            let xs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011 - 2.3).sin()).collect();
            let ys: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017 + 0.9).cos()).collect();
            let (a, b) = (1.5f32, -0.75f32);
            for (kind, conv) in [
                (Half::Bf16, bf16_bits as fn(f32) -> u16),
                (Half::F16, f16_bits as fn(f32) -> u16),
            ] {
                let xb: Vec<u16> = xs.iter().map(|&v| conv(v)).collect();
                let yb: Vec<u16> = ys.iter().map(|&v| conv(v)).collect();
                let mut got = vec![0f32; n];
                // SAFETY: `xb`/`yb` hold n half bits and `got` n f32.
                unsafe {
                    match kind {
                        Half::Bf16 => wukong_axpby_bf16(
                            xb.as_ptr(),
                            yb.as_ptr(),
                            got.as_mut_ptr(),
                            n as i64,
                            a,
                            b,
                        ),
                        Half::F16 => wukong_axpby_f16(
                            xb.as_ptr(),
                            yb.as_ptr(),
                            got.as_mut_ptr(),
                            n as i64,
                            a,
                            b,
                        ),
                    }
                }
                let mut want = vec![0f32; n];
                axpby_bf16_scalar(kind, &xb, &yb, &mut want, a, b);
                for i in 0..n {
                    assert_eq!(got[i].to_bits(), want[i].to_bits(), "axpby n={n} i={i}");
                }
            }
        }
    }

    #[test]
    fn mixed_width_overlap_matches_live_byte_reference() {
        for kind in [Half::Bf16, Half::F16] {
            for n in [0usize, 1, 7, 8, 9, 16, 17, 33] {
                // Byte offsets: identical starts, shifted in each direction, y-only overlap,
                // both inputs overlapping, and adjacency beyond the wider output range.
                for (xi, yi, oi) in [
                    (0, 8 * n + 8, 0),
                    (0, 8 * n + 8, 4),
                    (4, 8 * n + 8, 0),
                    (8 * n + 8, 0, 0),
                    (8 * n + 8, 2, 0),
                    (0, 2, 4),
                    (2 * n, 2 * n + 2, 0),
                    (4 * n, 4 * n + 2, 0),
                ] {
                    for op in [
                        None,
                        Some(crate::vmath::VM_RELU),
                        Some(crate::vmath::VM_SIGMOID),
                        Some(-1),
                    ] {
                        // u32 storage gives every f32 store its required alignment. Both u16
                        // and f32 accept every initialized bit pattern, including rewritten input.
                        let mut got: Vec<u32> = (0..3 * n + 16)
                            .map(|i| {
                                let lo = kind.narrow((i % 5) as f32 / 4.0);
                                let hi = kind.narrow((i % 7) as f32 / 4.0);
                                let [a, b] = lo.to_ne_bytes();
                                let [c, d] = hi.to_ne_bytes();
                                u32::from_ne_bytes([a, b, c, d])
                            })
                            .collect();
                        let mut want: Vec<u8> = got.iter().flat_map(|v| v.to_ne_bytes()).collect();
                        for i in 0..n {
                            let x = kind.widen(u16::from_ne_bytes([
                                want[xi + 2 * i],
                                want[xi + 2 * i + 1],
                            ]));
                            let value = if let Some(op) = op {
                                crate::vmath::apply1(op, x)
                            } else {
                                let y = kind.widen(u16::from_ne_bytes([
                                    want[yi + 2 * i],
                                    want[yi + 2 * i + 1],
                                ]));
                                0.25f32.mul_add(y, 0.5 * x)
                            };
                            want[oi + 4 * i..oi + 4 * i + 4].copy_from_slice(&value.to_ne_bytes());
                        }
                        let p = got.as_mut_ptr().cast::<u8>();
                        // SAFETY: `got` has n f32 (= n*4 bytes); the u8-cast pointer addresses that same buffer.
                        unsafe {
                            let x = p.add(xi).cast::<u16>();
                            let y = p.add(yi).cast::<u16>();
                            let out = p.add(oi).cast::<f32>();
                            match (kind, op) {
                                (Half::Bf16, None) => {
                                    wukong_axpby_bf16(x, y, out, n as i64, 0.5, 0.25)
                                }
                                (Half::F16, None) => {
                                    wukong_axpby_f16(x, y, out, n as i64, 0.5, 0.25)
                                }
                                (Half::Bf16, Some(op)) => {
                                    crate::wukong_vmath_bf16(x, out, n as i64, op)
                                }
                                (Half::F16, Some(op)) => {
                                    crate::wukong_vmath_f16(x, out, n as i64, op)
                                }
                            }
                        }
                        let got: Vec<u8> = got.iter().flat_map(|v| v.to_ne_bytes()).collect();
                        assert_eq!(got, want, "n={n}, offsets=({xi},{yi},{oi}), op={op:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn mixed_width_disjoint_output_allows_shared_read_only_inputs() {
        for kind in [Half::Bf16, Half::F16] {
            for n in [0usize, 1, 7, 8, 9, 16, 17, 33] {
                let input: Vec<u16> = (0..n + 1)
                    .map(|i| kind.narrow((i % 13) as f32 / 4.0 - 1.0))
                    .collect();
                for offset in [0usize, 1] {
                    let mut got = vec![0.0f32; n];
                    // SAFETY: `got` holds n f32; the offset variants write within it.
                    unsafe {
                        let kernel = match kind {
                            Half::Bf16 => wukong_axpby_bf16,
                            Half::F16 => wukong_axpby_f16,
                        };
                        kernel(
                            input.as_ptr(),
                            input.as_ptr().add(offset),
                            got.as_mut_ptr(),
                            n as i64,
                            0.5,
                            0.25,
                        );
                    }
                    for i in 0..n {
                        // Dyadic inputs/coefficient arithmetic is exact in both f64 and f32.
                        let want = (0.5 * kind.widen(input[i]) as f64
                            + 0.25 * kind.widen(input[i + offset]) as f64)
                            as f32;
                        assert_eq!(got[i].to_bits(), want.to_bits());
                    }
                }
            }
        }
    }

    /// The SIMD narrowing store (`narrow_bf16`/`narrow_f16`, 8 lanes at once) must be **bit-identical**
    /// to the scalar `f32_to_{bf16,f16}_bits` shim the interpreter narrows through — across ordinary
    /// values, negatives, subnormals, ±inf, and NaN (the round's every branch). This is the gate that
    /// keeps a half-**output** kernel bit-for-bit consistent with the differential oracle.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn narrow_simd_equals_scalar_shim() {
        if !crate::isa::avx2() || !crate::isa::f16c() {
            return;
        }
        // A spread of tricky f32s: 0/±0, small/large, subnormal, ties, ±inf, NaN, negatives.
        let mut vals: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            std::f32::consts::PI,
            -std::f32::consts::E,
            1e-40,
            -1e-40, // subnormal after narrowing
            65504.0,
            -65504.0, // f16 max
            1e30,
            -1e30, // overflows f16 → inf
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            1.0000001,
            1.9999999,
            0.999999, // near ties
        ];
        // Add a deterministic sweep that hits many mantissa bits, then pad to a multiple of 8.
        for i in 0..128 {
            vals.push((i as f32 * 0.013 - 0.4).sin() * 123.456);
        }
        while !vals.len().is_multiple_of(8) {
            vals.push(0.0);
        }
        let chunks = vals.len() / 8;
        for c in 0..chunks {
            let mut gb = [0u16; 8];
            let mut gf = [0u16; 8];
            // SAFETY: `gb`/`gf` hold 8 u16 each.
            unsafe {
                let v = _mm256_loadu_ps(vals.as_ptr().add(c * 8));
                narrow_bf16(v, gb.as_mut_ptr());
                narrow_f16(v, gf.as_mut_ptr());
            }
            for l in 0..8 {
                let x = vals[c * 8 + l];
                assert_eq!(gb[l], crate::f32_to_bf16_bits(x), "bf16 narrow x={x}");
                assert_eq!(gf[l], crate::f32_to_f16_bits(x), "f16 narrow x={x}");
            }
        }
        // The 16-lane pack helpers (the 256-bit NT-store core) must produce the identical bits as two
        // 8-lane narrows concatenated — same round, just a wider pack.
        while !vals.len().is_multiple_of(16) {
            vals.push(0.0);
        }
        for c in 0..vals.len() / 16 {
            let mut gb = [0u16; 16];
            let mut gf = [0u16; 16];
            // SAFETY: `gb`/`gf` hold 16 u16 each.
            unsafe {
                let v0 = _mm256_loadu_ps(vals.as_ptr().add(c * 16));
                let v1 = _mm256_loadu_ps(vals.as_ptr().add(c * 16 + 8));
                _mm256_storeu_si256(gb.as_mut_ptr() as *mut __m256i, narrow_bf16_pack(v0, v1));
                _mm256_storeu_si256(gf.as_mut_ptr() as *mut __m256i, narrow_f16_pack(v0, v1));
            }
            for l in 0..16 {
                let x = vals[c * 16 + l];
                assert_eq!(gb[l], crate::f32_to_bf16_bits(x), "bf16 pack x={x}");
                assert_eq!(gf[l], crate::f32_to_f16_bits(x), "f16 pack x={x}");
            }
        }
    }

    /// The half-**output** axpby (`wukong_axpby_{bf16,f16}_out`) SIMD path must equal its scalar twin
    /// bit-for-bit across tail sizes — the narrowing store rounds through the same shim, so the two
    /// agree lane-for-lane (and it is the no-AVX2 fallback).
    #[test]
    fn axpby_narrow_out_simd_equals_scalar_twin() {
        // Keep the existing sizes and grow the largest NT case with the shared policy.
        // Its mod-32 residue retains the alignment prologue, packed stores and scalar tail.
        for n in [
            0usize,
            1,
            7,
            8,
            9,
            16,
            17,
            100,
            1000,
            4099,
            1_749_000,
            crate::nt_policy::test_len(6, 2_000_003),
        ] {
            let xs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011 - 2.3).sin()).collect();
            let ys: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017 + 0.9).cos()).collect();
            let (a, b) = (1.5f32, -0.75f32);
            for (kind, conv) in [
                (Half::Bf16, bf16_bits as fn(f32) -> u16),
                (Half::F16, f16_bits as fn(f32) -> u16),
            ] {
                let xb: Vec<u16> = xs.iter().map(|&v| conv(v)).collect();
                let yb: Vec<u16> = ys.iter().map(|&v| conv(v)).collect();
                let mut got = vec![0u16; n];
                // SAFETY: `xb`/`yb` hold n half bits and `got` n u16.
                unsafe {
                    match kind {
                        Half::Bf16 => wukong_axpby_bf16_out(
                            xb.as_ptr(),
                            yb.as_ptr(),
                            got.as_mut_ptr(),
                            n as i64,
                            a,
                            b,
                        ),
                        Half::F16 => wukong_axpby_f16_out(
                            xb.as_ptr(),
                            yb.as_ptr(),
                            got.as_mut_ptr(),
                            n as i64,
                            a,
                            b,
                        ),
                    }
                }
                let mut want = vec![0u16; n];
                // SAFETY: `want` holds n u16; the reference kernel writes n.
                unsafe {
                    axpby_narrow_scalar(kind, xb.as_ptr(), yb.as_ptr(), want.as_mut_ptr(), n, a, b);
                }
                for i in 0..n {
                    assert_eq!(got[i], want[i], "narrow axpby n={n} i={i}");
                }
            }
        }
    }

    #[test]
    fn axpby_narrow_overlap_preserves_ascending_reads_and_writes() {
        for kind in [Half::Bf16, Half::F16] {
            let kernel = match kind {
                Half::Bf16 => wukong_axpby_bf16_out,
                Half::F16 => wukong_axpby_f16_out,
            };
            for n in [0usize, 1, 7, 8, 9, 16, 17, 33, 65] {
                // Disjoint, x in-place, y in-place, all equal, then shifted overlap
                // in each direction with both one and two overlapping input ranges.
                for (xi, yi, oi) in [
                    (0, n + 4, 2 * n + 8),
                    (0, n + 4, 0),
                    (0, n + 4, n + 4),
                    (0, 0, 0),
                    (0, n + 4, 1),
                    (1, n + 4, 0),
                    (n + 4, 0, 1),
                    (n + 4, 1, 0),
                    (0, 2, 1),
                ] {
                    let mut got: Vec<u16> = (0..3 * n + 16)
                        .map(|i| kind.narrow((i % 9) as f32 / 8.0))
                        .collect();
                    let mut want = got.clone();
                    for i in 0..n {
                        // These dyadic inputs and coefficients are exact in f64 and f32;
                        // narrow after each write so subsequent overlapping reads see it.
                        let x = kind.widen(want[xi + i]) as f64;
                        let y = kind.widen(want[yi + i]) as f64;
                        want[oi + i] = kind.narrow((0.5 * x + 0.25 * y) as f32);
                    }
                    let p = got.as_mut_ptr();
                    // SAFETY: `p.add(xi)`/`add(yi)`/`add(oi)` stay within `got` (overlap/aliasing probe).
                    unsafe { kernel(p.add(xi), p.add(yi), p.add(oi), n as i64, 0.5, 0.25) };
                    assert_eq!(got, want, "n={n}, offsets=({xi},{yi},{oi})");
                }
            }
            // SAFETY: the operands address within the allocated buffers for the disjoint control.
            unsafe {
                kernel(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    -1,
                    1.0,
                    1.0,
                )
            };
        }
    }

    #[test]
    fn axpby_narrow_in_place_streaming_and_special_values() {
        for kind in [Half::Bf16, Half::F16] {
            let kernel = match kind {
                Half::Bf16 => wukong_axpby_bf16_out,
                Half::F16 => wukong_axpby_f16_out,
            };
            // Cross the non-temporal threshold with a scalar prologue and tail.
            let n = crate::nt_policy::test_len(6, 1_747_661);
            let mut got = vec![kind.narrow(0.5); n + 1];
            let base = if got.as_ptr() as usize & 31 == 0 {
                1
            } else {
                0
            };
            // SAFETY: `base` keeps `p` within `got` (in-place probe).
            let p = unsafe { got.as_mut_ptr().add(base) };
            // SAFETY: `p` is all three operands (exact in-place), within `got`.
            unsafe { kernel(p, p, p, n as i64, 0.5, 0.25) };
            assert!(got[base..base + n].iter().all(|&v| v == kind.narrow(0.375)));

            let values = [
                0.0,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                -f32::NAN,
                1.0,
                -1.0,
            ];
            let mut got: Vec<u16> = values
                .into_iter()
                .cycle()
                .take(33)
                .map(|v| kind.narrow(v))
                .collect();
            let want: Vec<u16> = got
                .iter()
                .map(|&v| {
                    let v = kind.widen(v);
                    kind.narrow(1.0f32.mul_add(v, v))
                })
                .collect();
            let p = got.as_mut_ptr();
            // SAFETY: `p` is all three operands (exact in-place) over got.len() elements.
            unsafe { kernel(p, p, p, got.len() as i64, 1.0, 1.0) };
            assert_eq!(got, want);
        }
    }

    #[test]
    fn vmath_narrow_overlap_preserves_ascending_reads_and_writes() {
        use crate::vmath::{VM_RELU, VM_SIGMOID, VM_TANH};
        for kind in [Half::Bf16, Half::F16] {
            let kernel = match kind {
                Half::Bf16 => wukong_vmath_bf16_out,
                Half::F16 => wukong_vmath_f16_out,
            };
            for op in [VM_RELU, VM_SIGMOID, VM_TANH, -1] {
                for n in [0usize, 1, 7, 8, 9, 16, 17, 33, 65] {
                    for (xi, oi) in [(0, n + 1), (0, 0), (0, 1), (1, 0)] {
                        let mut got: Vec<u16> = (0..2 * n + 2)
                            .map(|i| kind.narrow((i % 13) as f32 / 4.0 - 1.5))
                            .collect();
                        let mut want = got.clone();
                        for i in 0..n {
                            want[oi + i] =
                                kind.narrow(crate::vmath::apply1(op, kind.widen(want[xi + i])));
                        }
                        let p = got.as_mut_ptr();
                        // SAFETY: the operands address within `got` for this width.
                        unsafe {
                            kernel(p.add(xi), p.add(oi), n as i64, op);
                        }
                        assert_eq!(got, want, "op={op}, n={n}, offsets=({xi},{oi})");
                    }
                }
            }
            // SAFETY: the operands address within the allocated buffers.
            unsafe {
                kernel(std::ptr::null(), std::ptr::null_mut(), -1, VM_RELU);
            }
        }
    }

    #[test]
    fn vmath_narrow_in_place_streaming_and_special_values() {
        use crate::vmath::{VM_RELU, VM_TANH};
        for kind in [Half::Bf16, Half::F16] {
            let kernel = match kind {
                Half::Bf16 => wukong_vmath_bf16_out,
                Half::F16 => wukong_vmath_f16_out,
            };
            let n = crate::nt_policy::test_len(6, 1_747_661);
            let mut got = vec![kind.narrow(-0.5); n + 1];
            let base = if got.as_ptr() as usize & 31 == 0 {
                1
            } else {
                0
            };
            // SAFETY: `base` keeps `p` within `got` (in-place probe).
            let p = unsafe { got.as_mut_ptr().add(base) };
            // SAFETY: `p` is all three operands (exact in-place), within `got`.
            unsafe {
                kernel(p, p, n as i64, VM_RELU);
            }
            assert!(got[base..base + n].iter().all(|&v| v == kind.narrow(0.0)));
            for op in [VM_RELU, VM_TANH, -1] {
                let values = [
                    0.0,
                    -0.0,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                    -f32::NAN,
                    1.0,
                    -1.0,
                ];
                let mut got: Vec<u16> = values
                    .into_iter()
                    .cycle()
                    .take(33)
                    .map(|v| kind.narrow(v))
                    .collect();
                let mut want = vec![0; got.len()];
                // SAFETY: `want` sized got.len(); the reference writes that span.
                unsafe {
                    vmath_narrow_scalar(kind, got.as_ptr(), want.as_mut_ptr(), got.len(), op);
                }
                let p = got.as_mut_ptr();
                // SAFETY: `p` addresses `got` over its full length.
                unsafe {
                    kernel(p, p, got.len() as i64, op);
                }
                assert_eq!(got, want, "op={op}");
            }
        }
    }

    #[test]
    fn vmath_narrow_out_every_op_instantiation_matches_the_scalar_twin() {
        // Each `VM_*` code has its own monomorphized instantiation of the half-output stream (the
        // kernel inlined through `vmath::with_act8`), so every one is driven here, plus two
        // off-table codes that must take the scalar twin. 45 = 32-lane body + 8 + a 5-element tail.
        // Mixed-sign input crosses the domain edges of log/acosh/atanh/asin/...; a NaN result is
        // compared as NaN-ness (the narrowing of a NaN payload is the shims' business, pinned
        // elsewhere), every other result bit-for-bit.
        use crate::vmath::VM_CBRT;
        let n = 45usize;
        let xs: Vec<f32> = (0..n).map(|i| (i as f32 - 22.0) * 0.21).collect();
        let ops: Vec<i64> = (0..=VM_CBRT).chain([-1, VM_CBRT + 1]).collect();
        for &op in &ops {
            for (kind, conv) in [
                (Half::Bf16, bf16_bits as fn(f32) -> u16),
                (Half::F16, f16_bits as fn(f32) -> u16),
            ] {
                let xb: Vec<u16> = xs.iter().map(|&v| conv(v)).collect();
                let mut got = vec![0u16; n];
                let mut want = vec![0u16; n];
                // SAFETY: `got`/`want` hold n u16.
                unsafe {
                    match kind {
                        Half::Bf16 => {
                            wukong_vmath_bf16_out(xb.as_ptr(), got.as_mut_ptr(), n as i64, op)
                        }
                        Half::F16 => {
                            wukong_vmath_f16_out(xb.as_ptr(), got.as_mut_ptr(), n as i64, op)
                        }
                    }
                    vmath_narrow_scalar(kind, xb.as_ptr(), want.as_mut_ptr(), n, op);
                }
                for i in 0..n {
                    let (g, w) = (kind.widen(got[i]), kind.widen(want[i]));
                    if w.is_nan() {
                        assert!(g.is_nan(), "narrow op={op} i={i}: {g} vs NaN");
                    } else {
                        assert_eq!(got[i], want[i], "narrow op={op} i={i}: {g} vs {w}");
                    }
                }
            }
        }
    }

    /// The half-**output** activation (`wukong_vmath_{bf16,f16}_out`) SIMD path must equal its scalar
    /// twin bit-for-bit across activations and tail sizes — the vector activation is the twin of
    /// `apply1` and the narrowing store the shared shim, so it agrees lane-for-lane (this is the gate
    /// that keeps the half-output activation consistent with the differential oracle). The large sizes
    /// drive the non-temporal streaming path (alignment prologue + 16-lane packs + tail).
    #[test]
    fn vmath_narrow_out_simd_equals_scalar_twin() {
        use crate::vmath::{VM_ERF, VM_EXP, VM_GELU, VM_RELU, VM_SIGMOID, VM_SILU, VM_TANH};
        for op in [
            VM_RELU, VM_SILU, VM_GELU, VM_EXP, VM_SIGMOID, VM_TANH, VM_ERF,
        ] {
            for n in [
                0usize,
                1,
                7,
                8,
                9,
                16,
                17,
                100,
                1000,
                4099,
                crate::nt_policy::test_len(6, 1_749_000),
            ] {
                // A spread over sign/magnitude (activations bend near 0 and saturate for large |x|).
                let xs: Vec<f32> = (0..n)
                    .map(|i| (i as f32 * 0.011 - 3.7).sin() * 4.0)
                    .collect();
                for (kind, conv) in [
                    (Half::Bf16, bf16_bits as fn(f32) -> u16),
                    (Half::F16, f16_bits as fn(f32) -> u16),
                ] {
                    let xb: Vec<u16> = xs.iter().map(|&v| conv(v)).collect();
                    let mut got = vec![0u16; n];
                    // SAFETY: `xb` holds n half bits and `got` n u16.
                    unsafe {
                        match kind {
                            Half::Bf16 => {
                                wukong_vmath_bf16_out(xb.as_ptr(), got.as_mut_ptr(), n as i64, op)
                            }
                            Half::F16 => {
                                wukong_vmath_f16_out(xb.as_ptr(), got.as_mut_ptr(), n as i64, op)
                            }
                        }
                    }
                    let mut want = vec![0u16; n];
                    // SAFETY: `want` holds n u16.
                    unsafe {
                        vmath_narrow_scalar(kind, xb.as_ptr(), want.as_mut_ptr(), n, op);
                    }
                    for i in 0..n {
                        assert_eq!(got[i], want[i], "narrow vmath op={op} n={n} i={i}");
                    }
                }
            }
        }
    }

    #[test]
    fn axpby_bf16_within_ulp_of_f64_reference() {
        let n = 1 << 14;
        let xs: Vec<f32> = (0..n).map(|i| ((i % 89) as f32) * 0.03 - 1.0).collect();
        let ys: Vec<f32> = (0..n).map(|i| ((i % 61) as f32) * 0.02 - 0.5).collect();
        let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
        let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();
        let (a, b) = (0.6f32, 1.3f32);
        let mut got = vec![0f32; n];
        // SAFETY: `got` holds n f32; inputs sized n (axpby a, b).
        unsafe {
            wukong_axpby_bf16(xbf.as_ptr(), ybf.as_ptr(), got.as_mut_ptr(), n as i64, a, b);
        }
        // f64 reference over the SAME widened bf16 inputs (so this isolates the f32 math error).
        let mut max_rel = 0.0f64;
        for i in 0..n {
            let r = a as f64 * bf16_bits_to_f32(xbf[i]) as f64
                + b as f64 * bf16_bits_to_f32(ybf[i]) as f64;
            let rel = (got[i] as f64 - r).abs() / r.abs().max(1.0);
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 1e-6, "axpby_bf16 rel {max_rel:.2e}");
    }

    /// Bandwidth win: a reduction over half-width inputs streams half the bytes, so for a working
    /// set well past L3 it runs ~2× faster than the f32 reduction. Compares Wukong's SIMD bf16/f16
    /// sum to Wukong's tuned f32 sum AND to a plain (rustc-autovectorized) Rust f32/bf16 sum.
    /// Run: `cargo test -p wukong_runtime --release lowp_bandwidth -- --ignored --nocapture`.
    #[test]
    #[ignore = "bandwidth bench; run explicitly in --release"]
    fn lowp_bandwidth() {
        use std::time::Instant;
        assert!(
            !cfg!(debug_assertions),
            "this is a throughput measurement, not a test — rebuild with --release \
             (cargo test -p wukong_runtime --release lowp_bandwidth -- --ignored --nocapture)"
        );
        let n = 32 << 20; // 32M elements — 128 MB f32, 64 MB half, both ≫ L3
        let xs: Vec<f32> = (0..n).map(|i| ((i % 251) as f32) * 0.001).collect();
        let xf16: Vec<u16> = xs.iter().map(|&v| f16_bits(v)).collect();
        let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();

        let best = |iters: usize, mut f: Box<dyn FnMut() -> f32>| -> f64 {
            let _ = f(); // warmup
            let mut t = f64::INFINITY;
            for _ in 0..iters {
                let t0 = Instant::now();
                std::hint::black_box(f());
                t = t.min(t0.elapsed().as_secs_f64());
            }
            t
        };

        let xs2 = xs.clone();
        let t_rust_f32 = best(5, Box::new(move || xs2.iter().copied().sum()));
        let xp = xs.as_ptr() as usize;
        let t_merc_f32 = best(
            5,
            // SAFETY: the sum closure reads n half bits via the captured pointer.
            Box::new(move || unsafe {
                crate::wukong_sreduce_f32(
                    xp as *const f32,
                    xp as *const f32,
                    n as i64,
                    crate::RED_SUM,
                )
            }),
        );
        let xbf2 = xbf.clone();
        let t_rust_bf16 = best(
            5,
            Box::new(move || xbf2.iter().map(|&b| bf16_bits_to_f32(b)).sum()),
        );
        let bp = xbf.as_ptr() as usize;
        let t_merc_bf16 = best(
            5,
            // SAFETY: the bf16 sum reads n elements via the captured pointer `bp`.
            Box::new(move || unsafe { wukong_sum_bf16(bp as *const u16, n as i64) }),
        );
        let fp = xf16.as_ptr() as usize;
        let t_merc_f16 = best(
            5,
            // SAFETY: the f16 sum reads n elements via the captured pointer `fp`.
            Box::new(move || unsafe { wukong_sum_f16(fp as *const u16, n as i64) }),
        );

        let gbps = |bytes: f64, t: f64| bytes / t / 1e9;
        let nf = n as f64;
        eprintln!("sum over {n} elements (best of 5):");
        eprintln!(
            "  rust f32 (autovec): {:.2} ms  {:.0} GB/s",
            t_rust_f32 * 1e3,
            gbps(4.0 * nf, t_rust_f32)
        );
        eprintln!(
            "  merc f32 (AVX2):    {:.2} ms  {:.0} GB/s",
            t_merc_f32 * 1e3,
            gbps(4.0 * nf, t_merc_f32)
        );
        eprintln!(
            "  rust bf16 (autovec):{:.2} ms  {:.0} GB/s",
            t_rust_bf16 * 1e3,
            gbps(2.0 * nf, t_rust_bf16)
        );
        eprintln!(
            "  merc bf16 (AVX2):   {:.2} ms  {:.0} GB/s  → {:.2}× vs merc-f32",
            t_merc_bf16 * 1e3,
            gbps(2.0 * nf, t_merc_bf16),
            t_merc_f32 / t_merc_bf16
        );
        eprintln!(
            "  merc f16  (F16C):   {:.2} ms  {:.0} GB/s  → {:.2}× vs merc-f32",
            t_merc_f16 * 1e3,
            gbps(2.0 * nf, t_merc_f16),
            t_merc_f32 / t_merc_f16
        );
    }

    /// Bandwidth win for the streaming **axpby** (`out = a·x + b·y`): bf16 inputs + f32 output move
    /// 8 bytes/elem vs an all-f32 axpby's 12, so on a working set ≫ L3 the bf16-in kernel runs ~1.5×
    /// faster. Compares `wukong_axpby_bf16` to a plain (rustc-autovectorized) f32 axpby.
    /// Run: `cargo test -p wukong_runtime --release axpby_bf16_bandwidth -- --ignored --nocapture`.
    #[test]
    #[ignore = "bandwidth bench; run explicitly in --release"]
    fn axpby_bf16_bandwidth() {
        use std::time::Instant;
        assert!(
            !cfg!(debug_assertions),
            "this is a throughput measurement, not a test — rebuild with --release \
             (cargo test -p wukong_runtime --release axpby_bf16_bandwidth -- --ignored --nocapture)"
        );
        let n = 32 << 20; // 32M elems — bf16 in 128 MB, f32 in 256 MB, both ≫ L3
        let xs: Vec<f32> = (0..n).map(|i| ((i % 251) as f32) * 0.001).collect();
        let ys: Vec<f32> = (0..n).map(|i| ((i % 199) as f32) * 0.002).collect();
        let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
        let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();
        let (a, b) = (1.5f32, 0.75f32);
        let best = |iters: usize, mut f: Box<dyn FnMut()>| -> f64 {
            f();
            let mut t = f64::INFINITY;
            for _ in 0..iters {
                let t0 = Instant::now();
                f();
                t = t.min(t0.elapsed().as_secs_f64());
            }
            t
        };

        let (xp, yp) = (xbf.as_ptr() as usize, ybf.as_ptr() as usize);
        let mut out_bf = vec![0f32; n];
        let obp = out_bf.as_mut_ptr() as usize;
        let t_bf = best(
            5,
            // SAFETY: the bf16 bench closure reads n elements via captured pointers.
            Box::new(move || unsafe {
                wukong_axpby_bf16(
                    xp as *const u16,
                    yp as *const u16,
                    obp as *mut f32,
                    n as i64,
                    a,
                    b,
                );
                std::hint::black_box(obp);
            }),
        );
        let (xfp, yfp) = (xs.as_ptr() as usize, ys.as_ptr() as usize);
        let mut out_f = vec![0f32; n];
        let ofp = out_f.as_mut_ptr() as usize;
        let t_f = best(
            5,
            // SAFETY: the f16 bench closure reads n elements via captured pointers.
            Box::new(move || unsafe {
                let x = std::slice::from_raw_parts(xfp as *const f32, n);
                let y = std::slice::from_raw_parts(yfp as *const f32, n);
                let o = std::slice::from_raw_parts_mut(ofp as *mut f32, n);
                for i in 0..n {
                    o[i] = a * x[i] + b * y[i];
                }
                std::hint::black_box(ofp);
            }),
        );
        let gbps = |bytes: f64, t: f64| bytes / t / 1e9;
        let nf = n as f64;
        eprintln!("axpby out=a·x+b·y over {n} elements (best of 5):");
        eprintln!(
            "  rust f32  (autovec, 12 B/elem): {:.2} ms  {:.0} GB/s",
            t_f * 1e3,
            gbps(12.0 * nf, t_f)
        );
        eprintln!(
            "  merc bf16-in/f32-out (8 B/elem): {:.2} ms  {:.0} GB/s  → {:.2}× faster",
            t_bf * 1e3,
            gbps(8.0 * nf, t_bf),
            t_f / t_bf
        );
    }

    #[test]
    fn within_ulp_tolerance_of_f64_reference() {
        let n = 1 << 16;
        let xs: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.01).collect();
        let ys: Vec<f32> = (0..n).map(|i| ((i % 53) as f32) * 0.02).collect();
        // f16 sum
        let xf16: Vec<u16> = xs.iter().map(|&v| f16_bits(v)).collect();
        let ref_sum: f64 = xf16.iter().map(|&b| f16_to_f32(b) as f64).sum();
        // SAFETY: `xf16` holds n f16 bits; wukong_sum_f16 reads n.
        let got = unsafe { wukong_sum_f16(xf16.as_ptr(), n as i64) };
        let rel = (got as f64 - ref_sum).abs() / ref_sum.abs().max(1.0);
        assert!(rel < 1e-3, "f16 sum rel {rel:.2e}");
        // bf16 dot
        let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
        let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();
        let ref_dot: f64 = xbf
            .iter()
            .zip(&ybf)
            .map(|(&a, &b)| bf16_bits_to_f32(a) as f64 * bf16_bits_to_f32(b) as f64)
            .sum();
        // SAFETY: `xbf`/`ybf` hold n bf16 bits; dot reads n.
        let got = unsafe { wukong_dot_bf16(xbf.as_ptr(), ybf.as_ptr(), n as i64) };
        let rel = (got as f64 - ref_dot).abs() / ref_dot.abs().max(1.0);
        assert!(rel < 1e-2, "bf16 dot rel {rel:.2e}");
    }

    /// **The bf16 accumulation ruling, pinned.** The cross-language benchmark reports the bf16 `dot`
    /// and `sum` disagreeing with C, C++ and Rust by ~1e-2 at N = 2^24. All three peers print the
    /// *same* value to the last bit, which is the signature of one strictly sequential f32
    /// accumulator — and an in-order f64 oracle says that accumulator, not this kernel, is the side
    /// that is wrong: at 2^24 the peers are 1.407e-2 (sum) / 2.333e-2 (dot) from the truth while
    /// these kernels are 1.69e-4 / 1.62e-4. Once a running f32 total passes 2^23 its ULP is 1.0, so
    /// every O(1) addend rounds to a whole ULP — a one-sided bias, not a random walk.
    ///
    /// This test pins the ruling at a size a debug build can afford (2^20, where the same mechanism
    /// already separates the two by >1000x on `sum`). Two directions are asserted, and a future
    /// change that quietly makes these kernels agree with the peers fails BOTH:
    ///   * these kernels stay within 1e-6 of the f64 reference, and
    ///   * the sequential f32 sum they deliberately do not perform stays past 1e-4 of it.
    ///
    /// LANDMINE: the right response to a bf16 "scalar drift" warning is to check this test, not to
    /// widen a tolerance. The warning is correctly reporting a real disagreement in the peer's
    /// favour of neither side — it is our accuracy that is the better one.
    #[test]
    fn lowp_reductions_beat_the_sequential_f32_sum() {
        // Two shapes. (a) is exactly the `wukong_xbench` bf16-reduction input (all-positive,
        // O(1), period 17 / 13); (b) is badly conditioned on purpose — one big value in three,
        // tiny signed values between — so the ruling is not an artifact of one benign pattern.
        let n = 1usize << 20;
        let bench_x: Vec<u16> = (0..n)
            .map(|i| bf16_bits((i as f32 % 17.0) * 0.05 + 0.5))
            .collect();
        let bench_y: Vec<u16> = (0..n)
            .map(|i| bf16_bits((i as f32 % 13.0) * 0.03 + 0.25))
            .collect();
        let hostile_x: Vec<u16> = (0..n)
            .map(|i| {
                bf16_bits(if i % 3 == 0 {
                    1000.0
                } else {
                    (i as f32 % 11.0) * 0.001 - 0.005
                })
            })
            .collect();

        // The independent oracle: the identical widened values (the widen is exact) summed in
        // ascending index order in f64, no reassociation. Every partial stays far below 2^53, so
        // this is the truth to well under one f32 ULP.
        let f64_sum = |x: &[u16]| {
            x.iter()
                .fold(0.0f64, |a, &v| a + bf16_bits_to_f32(v) as f64)
        };
        let f64_dot = |x: &[u16], y: &[u16]| {
            x.iter().zip(y).fold(0.0f64, |a, (&u, &v)| {
                a + bf16_bits_to_f32(u) as f64 * bf16_bits_to_f32(v) as f64
            })
        };
        // The accumulation gcc / g++ / rustc emit at honest flags: one sequential f32 accumulator.
        let seq_sum = |x: &[u16]| x.iter().fold(0.0f32, |a, &v| a + bf16_bits_to_f32(v));
        let seq_dot = |x: &[u16], y: &[u16]| {
            x.iter().zip(y).fold(0.0f32, |a, (&u, &v)| {
                a + bf16_bits_to_f32(u) * bf16_bits_to_f32(v)
            })
        };
        let rel = |got: f32, want: f64| ((got as f64 - want) / want).abs();

        for (name, x) in [("bench", &bench_x), ("hostile", &hostile_x)] {
            let want = f64_sum(x);
            // SAFETY: `x` holds n bf16 bits; wukong_sum_bf16 reads n.
            let ours = rel(unsafe { wukong_sum_bf16(x.as_ptr(), n as i64) }, want);
            let peer = rel(seq_sum(x), want);
            assert!(ours <= 1e-6, "{name}: bf16 sum rel {ours:.3e} > 1e-6");
            assert!(
                peer > 1e-4,
                "{name}: the sequential f32 sum is only {peer:.3e} off — the ruling's premise \
                 (a sequential accumulator loses badly at this size) no longer holds for this data"
            );
        }
        let want = f64_dot(&bench_x, &bench_y);
        let ours = rel(
            // SAFETY: `bench_x`/`bench_y` hold n bf16 bits; dot reads n.
            unsafe { wukong_dot_bf16(bench_x.as_ptr(), bench_y.as_ptr(), n as i64) },
            want,
        );
        let peer = rel(seq_dot(&bench_x, &bench_y), want);
        assert!(ours <= 1e-6, "bf16 dot rel {ours:.3e} > 1e-6");
        assert!(peer > 1e-4, "sequential f32 dot only {peer:.3e} off");
    }

    /// The IDENTICAL chunk fold the `_parallel` kernels perform, computed sequentially (no rayon): cut
    /// `[0, n)` on the fixed `RCHUNK` boundary, reduce each chunk with the serial kernel, fold the
    /// partials ascending from `ident(op)`. The parallel kernel must equal this bit-for-bit — that is
    /// its thread-count-independence contract — and it reassociates vs the flat whole-array serial.
    fn seq_chunked(x: &[u16], y: &[u16], op: i64, is_f16: bool) -> f32 {
        use crate::reduce::{fold2, ident, RCHUNK, RED_DOT, RED_SUM};
        let n = x.len();
        let nchunks = n.div_ceil(RCHUNK).max(1);
        let mut acc = ident(op);
        for c in 0..nchunks {
            let lo = c * RCHUNK;
            let hi = ((c + 1) * RCHUNK).min(n);
            let len = (hi - lo) as i64;
            // SAFETY: [lo, hi) ⊆ [0, n); the slices are valid for `len` u16.
            let p = unsafe {
                match (op, is_f16) {
                    (RED_DOT, false) => wukong_dot_bf16(x[lo..].as_ptr(), y[lo..].as_ptr(), len),
                    (RED_DOT, true) => wukong_dot_f16(x[lo..].as_ptr(), y[lo..].as_ptr(), len),
                    (RED_SUM, false) => wukong_sum_bf16(x[lo..].as_ptr(), len),
                    (RED_SUM, true) => wukong_sum_f16(x[lo..].as_ptr(), len),
                    (_, false) => wukong_reduce_bf16(x[lo..].as_ptr(), len, op),
                    (_, true) => wukong_reduce_f16(x[lo..].as_ptr(), len, op),
                }
            };
            acc = fold2(acc, p, op);
        }
        acc
    }

    #[test]
    fn parallel_reductions_match_sequential_chunked() {
        use crate::reduce::{RED_DOT, RED_MAX, RED_MAXABS, RED_MIN, RED_SUM};
        // Sizes straddling the fixed RCHUNK (8192) boundary: the < 2-chunk guard path, an exact chunk,
        // a one-over partial, and several full chunks + a non-mult-of-8 tail.
        for &n in &[1usize, 8, 8191, 8192, 8193, 3 * 8192 + 13] {
            let xs: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.013 - 0.5).collect();
            let ys: Vec<f32> = (0..n).map(|i| (i % 53) as f32 * 0.017 - 0.3).collect();
            let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
            let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();
            let xf16: Vec<u16> = xs.iter().map(|&v| f16_bits(v)).collect();
            let yf16: Vec<u16> = ys.iter().map(|&v| f16_bits(v)).collect();
            let nn = n as i64;
            // SAFETY: the half inputs and outputs are sized n for the axpby family.
            unsafe {
                assert_eq!(
                    wukong_sum_bf16_parallel(xbf.as_ptr(), nn).to_bits(),
                    seq_chunked(&xbf, &xbf, RED_SUM, false).to_bits(),
                    "sum_bf16 n={n}"
                );
                assert_eq!(
                    wukong_sum_f16_parallel(xf16.as_ptr(), nn).to_bits(),
                    seq_chunked(&xf16, &xf16, RED_SUM, true).to_bits(),
                    "sum_f16 n={n}"
                );
                assert_eq!(
                    wukong_dot_bf16_parallel(xbf.as_ptr(), ybf.as_ptr(), nn).to_bits(),
                    seq_chunked(&xbf, &ybf, RED_DOT, false).to_bits(),
                    "dot_bf16 n={n}"
                );
                assert_eq!(
                    wukong_dot_f16_parallel(xf16.as_ptr(), yf16.as_ptr(), nn).to_bits(),
                    seq_chunked(&xf16, &yf16, RED_DOT, true).to_bits(),
                    "dot_f16 n={n}"
                );
                for &op in &[RED_MAX, RED_MIN, RED_MAXABS] {
                    let gb = wukong_reduce_bf16_parallel(xbf.as_ptr(), nn, op);
                    assert_eq!(
                        gb.to_bits(),
                        seq_chunked(&xbf, &xbf, op, false).to_bits(),
                        "reduce_bf16 op={op} n={n}"
                    );
                    // Determinism: a second call returns the identical bits.
                    assert_eq!(
                        gb.to_bits(),
                        wukong_reduce_bf16_parallel(xbf.as_ptr(), nn, op).to_bits(),
                        "reduce_bf16 nondeterministic op={op} n={n}"
                    );
                    let gf = wukong_reduce_f16_parallel(xf16.as_ptr(), nn, op);
                    assert_eq!(
                        gf.to_bits(),
                        seq_chunked(&xf16, &xf16, op, true).to_bits(),
                        "reduce_f16 op={op} n={n}"
                    );
                }
            }
        }
    }

    #[test]
    fn parallel_reductions_fork_on_the_unified_pool() {
        // `par_chunk_reduce` can be a process's FIRST rayon touch (a bf16 model whose first parallel
        // op is a reduction). Forking `into_par_iter()` straight from there builds rayon's DEFAULT
        // global registry — 2 MiB worker stacks — and the runtime's later `ensure_global_pool()` then
        // loses the race silently (`build_global` → GlobalPoolAlreadyInitialized, discarded at
        // lib.rs), so every subsequent outlined `@parallel` region body runs on a stack an eighth of
        // the required 16 MiB. That first-touch ordering is process-global, so a shared-process test
        // harness cannot pin it directly; what it CAN pin, order-independently, is the cause: the
        // chunks must execute on the unified kernel pool, i.e. this fork must go through
        // `run_on_wuk_pool`, which is what configures the global pool before anything else can.
        //
        // On a part with no HyperThreads to shed, `gemm_pool()` is None and the two pools coincide,
        // so the comparison is vacuous there (and under `WUKONG_POOL_UNIFY=0`, and at
        // `RAYON_NUM_THREADS=1`) — it still cannot regress. On this 6P+8E+2LPE machine the widths are
        // 16 vs 22 and it is a real discriminator.
        let want = crate::wuk_pool_width();
        let n = 4 * crate::reduce::RCHUNK + 7;
        let seen = std::sync::Mutex::new(std::collections::BTreeSet::new());
        let got = par_chunk_reduce(
            n,
            0.0,
            |lo, hi| {
                seen.lock().unwrap().insert(rayon::current_num_threads());
                (hi - lo) as f32
            },
            |a, b| a + b,
        );
        // Every chunk ran exactly once (so the widths below are the whole story, not a sample).
        assert_eq!(got, n as f32, "chunk coverage");
        assert_eq!(
            seen.into_inner().unwrap(),
            std::collections::BTreeSet::from([want]),
            "par_chunk_reduce forked off the unified kernel pool (width {want})"
        );
    }

    #[test]
    fn parallel_reductions_nest_inside_a_pool_worker() {
        // A bf16 `_parallel` reduction is emitted inside `@parallel` functions, so it can be reached
        // from an already-outlined region body — i.e. from a worker of the very pool the fix above
        // now installs onto. rayon runs a nested `install` on the same pool inline, so this must
        // complete and return the same bits as the un-nested call; if it ever deadlocked, the gate
        // would hang rather than fail, so pin it explicitly.
        let n = 4 * crate::reduce::RCHUNK + 7;
        let flat = |lo: usize, hi: usize| (hi - lo) as f32;
        let outer = par_chunk_reduce(n, 0.0, flat, |a, b| a + b);
        let inner = crate::run_on_wuk_pool(|| par_chunk_reduce(n, 0.0, flat, |a, b| a + b));
        assert_eq!(outer.to_bits(), inner.to_bits(), "nested != top-level");
        assert_eq!(outer, n as f32, "chunk coverage");
    }

    #[test]
    fn parallel_reductions_within_f64_tolerance() {
        use crate::reduce::{RED_MAX, RED_MAXABS, RED_MIN};
        // A size that exercises several real chunks (RCHUNK = 8192), so rayon actually runs.
        let n = 3 * 8192 + 251;
        let xs: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.01).collect();
        let ys: Vec<f32> = (0..n).map(|i| ((i % 53) as f32) * 0.02).collect();
        let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
        let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();
        let xf16: Vec<u16> = xs.iter().map(|&v| f16_bits(v)).collect();
        // SAFETY: the bf16/f16 inputs and outputs are sized n.
        unsafe {
            // bf16 / f16 sum: f32-accumulation error within ULP tolerance of the f64 reference.
            let want: f64 = xbf.iter().map(|&b| bf16_bits_to_f32(b) as f64).sum();
            let got = wukong_sum_bf16_parallel(xbf.as_ptr(), n as i64) as f64;
            assert!(
                (got - want).abs() / want.abs().max(1.0) < 1e-2,
                "bf16 sum got {got} want {want}"
            );
            let want: f64 = xf16.iter().map(|&b| f16_to_f32(b) as f64).sum();
            let got = wukong_sum_f16_parallel(xf16.as_ptr(), n as i64) as f64;
            assert!(
                (got - want).abs() / want.abs().max(1.0) < 1e-3,
                "f16 sum got {got} want {want}"
            );
            // bf16 dot.
            let want: f64 = xbf
                .iter()
                .zip(&ybf)
                .map(|(&a, &b)| bf16_bits_to_f32(a) as f64 * bf16_bits_to_f32(b) as f64)
                .sum();
            let got = wukong_dot_bf16_parallel(xbf.as_ptr(), ybf.as_ptr(), n as i64) as f64;
            assert!(
                (got - want).abs() / want.abs().max(1.0) < 1e-2,
                "bf16 dot got {got} want {want}"
            );
            // max / min / absmax round nothing and the widen is lossless, so the parallel result is
            // the *exact* reduction of the widened values — assert bit-for-bit, no tolerance.
            let widen =
                |b: &[u16]| -> Vec<f32> { b.iter().map(|&v| bf16_bits_to_f32(v)).collect() };
            let xw = widen(&xbf);
            let want_max = xw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let want_min = xw.iter().copied().fold(f32::INFINITY, f32::min);
            let want_amx = xw.iter().map(|v| v.abs()).fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(
                wukong_reduce_bf16_parallel(xbf.as_ptr(), n as i64, RED_MAX).to_bits(),
                want_max.to_bits(),
                "bf16 max"
            );
            assert_eq!(
                wukong_reduce_bf16_parallel(xbf.as_ptr(), n as i64, RED_MIN).to_bits(),
                want_min.to_bits(),
                "bf16 min"
            );
            assert_eq!(
                wukong_reduce_bf16_parallel(xbf.as_ptr(), n as i64, RED_MAXABS).to_bits(),
                want_amx.to_bits(),
                "bf16 absmax"
            );
        }
    }
}
