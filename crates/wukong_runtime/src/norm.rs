//! Fused single-pass row-wise normalizations — **softmax / log-softmax / LayerNorm / RMSNorm /
//! L2-norm** (the five `NORM_*` op codes below) over the last axis of a `[rows, cols]` f32 matrix.
//! These are the per-token normalizations every transformer block
//! runs (softmax in attention; LayerNorm/RMSNorm around the sublayers), and they are *memory-bound*:
//! a naive implementation streams each row from memory two or three times (max+exp+sum+normalize for
//! softmax; mean+var+normalize for LayerNorm). The win here is **fusion** — each row is loaded once
//! into registers/L1 and all passes run on it before moving on — plus the 256-bit AVX2 width (and,
//! for softmax, the same hand-vectorized `exp` the activation kernel uses, which Cranelift can't emit).
//!
//! Four C entries: [`wukong_norm_f32`] / [`wukong_norm_f32_parallel`] for the plain forms, and
//! [`wukong_norm_affine_f32`] / [`wukong_norm_affine_f32_parallel`] for LayerNorm/RMSNorm with a
//! per-column `gamma` scale and optional `beta` shift (either pointer may be null; the affine entries
//! accept **only** `NORM_LAYERNORM`/`NORM_RMSNORM` and silently do nothing for any other code).
//!
//! The compiler recognizes the canonical row-wise norm loop nest and lowers it to one of those calls
//! (the same play as matmul→GEMM, activation→`wukong_vmath_f32`, and
//! reduction→`wukong_sreduce_f32`). The interpreter marshals its abstract memory through the
//! **identical** kernel, so the differential oracle stays bit-for-bit exact.
//!
//! **Determinism.** Rows are independent, so the `_parallel` entries just map the identical per-row
//! routine across rows — `serial == parallel` bit-for-bit with no cross-row combine (and at
//! `wuk_pool_width() <= 1` they skip the pool entirely and tail-call the serial sibling, which is the
//! same bits by that argument). A non-positive `rows` or `cols` is a no-op at every entry, never a
//! walk off the end. Within a row the
//! reductions (softmax's max+sum, LayerNorm's mean+var, RMSNorm's mean-square) use a **fixed**
//! accumulator layout and a fixed-order combine, so the result depends on `n` alone and never on the
//! machine or the call site. Every row reduction except softmax's exp-sum runs `UNROLL` = 8
//! accumulators over `BLOCK` = 64-element steps (see the block above [`ssq8_scalar`] for why — the
//! one-accumulator form was FMA-*latency* bound — and the block above [`rowmax8_scalar`] for the
//! softmax / log-softmax max pass, which is a pure `vmaxps` chain), with a single-accumulator
//! remainder and a ragged scalar tail. The exp-sum keeps one 8-lane accumulator: its `vaddps` chain
//! hides behind `exp8`, so splitting it would change bits and buy nothing. Both forms end by storing
//! the 8 lanes to an `[f32; 8]` the scalar twin builds identically and calling the *same* horizontal
//! combine, so the
//! AVX2 kernel and the scalar fallback agree bit-for-bit (pinned by `scalar_matches_avx2_bit_for_bit`
//! across 25 sizes covering every block, remainder and tail seam, and by
//! `tests/run/norm_blocked_every_lane.wk`, which prints every lane of a dispatched 100- and
//! 112-element row against expectations derived from the scalar reading of the loop).

//! Softmax and log-softmax use ordinary exceptional arithmetic: a row containing NaN or +∞,
//! or a fully masked row of -∞, yields NaNs. In particular, -∞ - -∞ is NaN and exp propagates
//! it (W16/FINDING-05); there is no special fully masked-row policy. A row with one finite
//! element and otherwise -∞ gives exactly one/zero probabilities and zero/-∞ log-probabilities.

#[cfg(target_arch = "x86_64")]
use crate::vmath::exp8;
use crate::vmath::{exp1, log1};

// Norm op codes (shared with the recognizer in `wukong_mir_build`).
pub const NORM_SOFTMAX: i64 = 0; // out = softmax(x) over the row (numerically stable)
pub const NORM_LAYERNORM: i64 = 1; // out = (x - mean) / sqrt(var + eps)
pub const NORM_RMSNORM: i64 = 2; // out = x / sqrt(mean(x^2) + eps)
pub const NORM_LOGSOFTMAX: i64 = 3; // out = (x - m) - log(sum(exp(x - m))) — stable log-softmax
pub const NORM_L2NORM: i64 = 4; // out = x / sqrt(sum(x^2) + eps) — L2 / unit-norm (no mean divisor)

/// Fixed-order horizontal sum of 8 lane accumulators — a balanced tree, identical in the scalar twin
/// and the AVX2 path (which stores its `__m256` to `[f32; 8]` and calls this), so both give the same
/// bits. Same shape as the reduction kernel's `hcombine8`.
#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

/// Fixed-order horizontal max of 8 lane accumulators (balanced tree). Note it folds with `f32::max`
/// (= `maxNum`, which *drops* a NaN), **not** the [`maxps`] semantics the 8-lane accumulation below
/// uses — which is safe only because this is the very same function on both paths: the AVX2 body
/// stores its `__m256` to `[f32; 8]` and calls this, so the two row maxima agree by construction, on
/// NaN rows as well as finite ones (pinned by `scalar_matches_avx2_on_nan_rows`, which places the NaN
/// past the first 8-block so it reaches this fold through a lane accumulator).
#[inline(always)]
fn hmax8(a: [f32; 8]) -> f32 {
    (a[0].max(a[1]).max(a[2].max(a[3]))).max(a[4].max(a[5]).max(a[6].max(a[7])))
}

/// `_mm256_max_ps(a, b)` semantics spelled out: `a > b ? a : b`. **Not** `f32::max` (= `maxNum`),
/// which returns the non-NaN operand — MAXPS returns its *second* source whenever the compare is
/// unordered, so a NaN in the freshly loaded operand poisons the accumulator lane while a NaN already
/// in the accumulator is dropped. The scalar row-max twin's 8-block loop folds with this, not with
/// `f32::max`, so it mirrors the AVX2 `_mm256_max_ps` chain bit-for-bit on NaN rows too and not only on
/// finite data (its ragged tail *does* use `f32::max`, but so does the AVX2 tail — same fold both
/// sides, and [`hmax8`] likewise; pinned by
/// `scalar_matches_avx2_on_nan_rows`). The reduction order remains shared even though NaNs
/// now propagate through `exp1`/`exp8` into the normalized row.
#[inline(always)]
fn maxps(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

// --- row reductions: 8 independent accumulators -------------------------------------------------
//
// The reduction pass of every non-softmax norm used to run ONE accumulator:
//
//     while i + 8 <= n { let v = load(x+i); ssv = fmadd(v, v, ssv); i += 8; }
//
// which is a serial dependency chain through `ssv`, so the loop retires one 8-lane step per FMA
// *latency* (4 cycles on Golden/Redwood Cove) rather than per FMA *throughput* (2 per cycle). At the
// single-row sizes a transformer actually uses — a 768-wide row is 96 steps, a 4096-wide row 512 —
// that chain **is** the kernel: 96 x 4 = 384 cycles for a pass whose throughput bound is 48. gcc's
// `-ffast-math` peer gets a 2x head start here for free, because it emits a separate `vmulps` +
// `vaddps` and only the 2-cycle `vaddps` is on its chain.
//
// So the block below runs `UNROLL` = 8 accumulators over `BLOCK` = 64 elements per step, which puts
// 8 independent FMA chains in flight and makes the pass throughput-bound. `UNROLL` = 8 (not 4) is
// what FMA latency x issue width asks for: 4 cycles x 2 per cycle = 8 in flight to saturate.
//
// **Each reduction is a matched scalar/AVX2 pair, written adjacently and with the identical block
// structure**, because the module's contract is that the AVX2 kernel and the scalar twin agree
// bit-for-bit — and an accumulator layout is exactly the kind of thing that silently desyncs. Lane
// `j` of scalar accumulator `t` folds the same elements as lane `j` of vector accumulator `t`, and
// the final combine is the same explicit halving tree on both sides, so they agree by construction.
//
// Two consequences worth stating:
//   * For `n < 64` the blocked loop never runs and everything falls through the single-accumulator
//     remainder, so **short rows keep byte-identical results to the pre-unroll kernel**.
//   * For `n >= 64` the summation order changes. It changes toward pairwise summation, so the error
//     bound improves rather than degrades (`matches_f64_reference` covers this); it is not, and was
//     never, bit-compatible with a strictly sequential reduction.

/// Accumulators in flight over the reduction's blocked body.
const UNROLL: usize = 8;
/// Elements consumed per blocked step (`UNROLL` 8-lane vectors).
const BLOCK: usize = UNROLL * 8;

/// Fold the 8 lane-accumulator arrays down to one with an explicit halving tree. The AVX2 twins run
/// the identical tree on `__m256` values, and `_mm256_add_ps` is lane-wise, so lane `j` here is lane
/// `j` there.
#[inline(always)]
fn combine8(a: &mut [[f32; 8]; UNROLL]) {
    // Halve three times: (0,4)(1,5)(2,6)(3,7), then (0,2)(1,3), then (0,1) — the pairing and the
    // order the AVX2 twins spell out explicitly with `_mm256_add_ps`.
    let mut w = UNROLL;
    while w > 1 {
        let h = w / 2;
        let (lo, hi) = a[..w].split_at_mut(h);
        for (l, r) in lo.iter_mut().zip(hi.iter()) {
            for (lj, rj) in l.iter_mut().zip(r.iter()) {
                *lj += *rj;
            }
        }
        w = h;
    }
}

/// `Σ x[i]²` as 8 lane accumulators (RMSNorm / L2-norm), scalar twin of [`ssq8_avx2`].
///
/// # Safety
/// `x` valid for `n` `f32`.
unsafe fn ssq8_scalar(x: *const f32, n: usize) -> [f32; 8] {
    let mut a = [[0.0f32; 8]; UNROLL];
    let mut i = 0;
    while i + BLOCK <= n {
        for (t, at) in a.iter_mut().enumerate() {
            for (j, atj) in at.iter_mut().enumerate() {
                let v = *x.add(i + 8 * t + j);
                *atj = v.mul_add(v, *atj);
            }
        }
        i += BLOCK;
    }
    while i + 8 <= n {
        for (j, a0j) in a[0].iter_mut().enumerate() {
            let v = *x.add(i + j);
            *a0j = v.mul_add(v, *a0j);
        }
        i += 8;
    }
    combine8(&mut a);
    let mut ss = a[0];
    for (j, ssj) in ss.iter_mut().enumerate().take(n - i) {
        let v = *x.add(i + j);
        *ssj = v.mul_add(v, *ssj);
    }
    ss
}

/// `Σ x[i]` as 8 lane accumulators (the LayerNorm mean pass), scalar twin of [`sum8_avx2`].
///
/// # Safety
/// `x` valid for `n` `f32`.
unsafe fn sum8_scalar(x: *const f32, n: usize) -> [f32; 8] {
    let mut a = [[0.0f32; 8]; UNROLL];
    let mut i = 0;
    while i + BLOCK <= n {
        for (t, at) in a.iter_mut().enumerate() {
            for (j, atj) in at.iter_mut().enumerate() {
                *atj += *x.add(i + 8 * t + j);
            }
        }
        i += BLOCK;
    }
    while i + 8 <= n {
        for (j, a0j) in a[0].iter_mut().enumerate() {
            *a0j += *x.add(i + j);
        }
        i += 8;
    }
    combine8(&mut a);
    let mut sm = a[0];
    for (j, smj) in sm.iter_mut().enumerate().take(n - i) {
        *smj += *x.add(i + j);
    }
    sm
}

/// `Σ (x[i] - mean)²` as 8 lane accumulators (the LayerNorm variance pass), scalar twin of
/// [`dev2_8_avx2`]. The addend is a `mul_add`, matching the AVX2 `fmadd`.
///
/// # Safety
/// `x` valid for `n` `f32`.
unsafe fn dev2_8_scalar(x: *const f32, n: usize, mean: f32) -> [f32; 8] {
    let mut a = [[0.0f32; 8]; UNROLL];
    let mut i = 0;
    while i + BLOCK <= n {
        for (t, at) in a.iter_mut().enumerate() {
            for (j, atj) in at.iter_mut().enumerate() {
                let d = *x.add(i + 8 * t + j) - mean;
                *atj = d.mul_add(d, *atj);
            }
        }
        i += BLOCK;
    }
    while i + 8 <= n {
        for (j, a0j) in a[0].iter_mut().enumerate() {
            let d = *x.add(i + j) - mean;
            *a0j = d.mul_add(d, *a0j);
        }
        i += 8;
    }
    combine8(&mut a);
    let mut va = a[0];
    for (j, vaj) in va.iter_mut().enumerate().take(n - i) {
        let d = *x.add(i + j) - mean;
        *vaj = d.mul_add(d, *vaj);
    }
    va
}

/// `Σ x[i]²` as 8 lane accumulators, AVX2 twin of [`ssq8_scalar`] (same block structure, same tree).
///
/// # Safety
/// `x` valid for `n` `f32`; AVX2+FMA detected by the caller.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ssq8_avx2(x: *const f32, n: usize) -> [f32; 8] {
    use std::arch::x86_64::*;
    let mut a = [_mm256_setzero_ps(); UNROLL];
    let mut i = 0;
    while i + BLOCK <= n {
        for (t, at) in a.iter_mut().enumerate() {
            let v = _mm256_loadu_ps(x.add(i + 8 * t));
            *at = _mm256_fmadd_ps(v, v, *at);
        }
        i += BLOCK;
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.add(i));
        a[0] = _mm256_fmadd_ps(v, v, a[0]);
        i += 8;
    }
    a[0] = _mm256_add_ps(a[0], a[4]);
    a[1] = _mm256_add_ps(a[1], a[5]);
    a[2] = _mm256_add_ps(a[2], a[6]);
    a[3] = _mm256_add_ps(a[3], a[7]);
    a[0] = _mm256_add_ps(a[0], a[2]);
    a[1] = _mm256_add_ps(a[1], a[3]);
    a[0] = _mm256_add_ps(a[0], a[1]);
    let mut ss = [0.0f32; 8];
    _mm256_storeu_ps(ss.as_mut_ptr(), a[0]);
    for (j, ssj) in ss.iter_mut().enumerate().take(n - i) {
        let v = *x.add(i + j);
        *ssj = v.mul_add(v, *ssj);
    }
    ss
}

/// `Σ x[i]` as 8 lane accumulators, AVX2 twin of [`sum8_scalar`].
///
/// # Safety
/// `x` valid for `n` `f32`; AVX2+FMA detected by the caller.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum8_avx2(x: *const f32, n: usize) -> [f32; 8] {
    use std::arch::x86_64::*;
    let mut a = [_mm256_setzero_ps(); UNROLL];
    let mut i = 0;
    while i + BLOCK <= n {
        for (t, at) in a.iter_mut().enumerate() {
            *at = _mm256_add_ps(*at, _mm256_loadu_ps(x.add(i + 8 * t)));
        }
        i += BLOCK;
    }
    while i + 8 <= n {
        a[0] = _mm256_add_ps(a[0], _mm256_loadu_ps(x.add(i)));
        i += 8;
    }
    a[0] = _mm256_add_ps(a[0], a[4]);
    a[1] = _mm256_add_ps(a[1], a[5]);
    a[2] = _mm256_add_ps(a[2], a[6]);
    a[3] = _mm256_add_ps(a[3], a[7]);
    a[0] = _mm256_add_ps(a[0], a[2]);
    a[1] = _mm256_add_ps(a[1], a[3]);
    a[0] = _mm256_add_ps(a[0], a[1]);
    let mut sm = [0.0f32; 8];
    _mm256_storeu_ps(sm.as_mut_ptr(), a[0]);
    for (j, smj) in sm.iter_mut().enumerate().take(n - i) {
        *smj += *x.add(i + j);
    }
    sm
}

/// `Σ (x[i] - mean)²` as 8 lane accumulators, AVX2 twin of [`dev2_8_scalar`].
///
/// # Safety
/// `x` valid for `n` `f32`; AVX2+FMA detected by the caller.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dev2_8_avx2(x: *const f32, n: usize, mean: f32) -> [f32; 8] {
    use std::arch::x86_64::*;
    let mb = _mm256_set1_ps(mean);
    let mut a = [_mm256_setzero_ps(); UNROLL];
    let mut i = 0;
    while i + BLOCK <= n {
        for (t, at) in a.iter_mut().enumerate() {
            let d = _mm256_sub_ps(_mm256_loadu_ps(x.add(i + 8 * t)), mb);
            *at = _mm256_fmadd_ps(d, d, *at);
        }
        i += BLOCK;
    }
    while i + 8 <= n {
        let d = _mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb);
        a[0] = _mm256_fmadd_ps(d, d, a[0]);
        i += 8;
    }
    a[0] = _mm256_add_ps(a[0], a[4]);
    a[1] = _mm256_add_ps(a[1], a[5]);
    a[2] = _mm256_add_ps(a[2], a[6]);
    a[3] = _mm256_add_ps(a[3], a[7]);
    a[0] = _mm256_add_ps(a[0], a[2]);
    a[1] = _mm256_add_ps(a[1], a[3]);
    a[0] = _mm256_add_ps(a[0], a[1]);
    let mut va = [0.0f32; 8];
    _mm256_storeu_ps(va.as_mut_ptr(), a[0]);
    for (j, vaj) in va.iter_mut().enumerate().take(n - i) {
        let d = *x.add(i + j) - mean;
        *vaj = d.mul_add(d, *vaj);
    }
    va
}

// --- the softmax row max: the same 8 chains, folded by MAXPS ---------------------------------------
//
// The softmax / log-softmax max pass is a pure `load -> vmaxps` stream with nothing else in the loop,
// so on one accumulator it retires one 8-lane step per `vmaxps` *latency* — the single most
// latency-bound pass in the module (the exp-sum pass that follows hides its one `vaddps` chain
// behind `exp8`'s far longer per-vector work, so it keeps its single accumulator: splitting it would
// change result bits for no throughput). Eight chains over `BLOCK` = 64 elements make it load-bound.
//
// The fold is MAXPS (`(a > b) ? a : b`, [`maxps`]) in both twins, and the halving tree folds
// `maxps(a[t + h], a[t])` — the higher chain as the FIRST operand. That order is what makes the
// change free of observable effect:
//   * An untouched chain holds `-inf`, and `maxps(-inf, v)` returns `v` for every `v`, NaN
//     included, so for `n < BLOCK` the lanes come out bit-identical to the single-chain kernel.
//   * For `n >= BLOCK` on NaN-free data the row max is the same *value* under any fold order; only
//     the sign of a zero maximum can move, and `exp(x - (+0)) == exp(x - (-0))` for every `x` (and
//     log-softmax's `m + log(s)` is `+0`-insensitive), so every softmax / log-softmax output bit is
//     unchanged. On a row holding a NaN the outputs are all NaN under either order.
// The ragged tail keeps its `f32::max` fold and the result keeps its [`hmax8`] combine, both shared
// by the twins exactly as before. `pub(crate)` so `logsoftmax.rs`'s duplicated core uses the same
// pass and stays bit-identical to `NORM_LOGSOFTMAX`.

/// Row max as 8 lane maxima over `UNROLL` MAXPS chains, scalar twin of [`rowmax8_avx2`].
///
/// # Safety
/// `x` valid for `n` `f32`.
pub(crate) unsafe fn rowmax8_scalar(x: *const f32, n: usize) -> [f32; 8] {
    let mut a = [[f32::NEG_INFINITY; 8]; UNROLL];
    let mut i = 0;
    while i + BLOCK <= n {
        for (t, at) in a.iter_mut().enumerate() {
            for (j, atj) in at.iter_mut().enumerate() {
                *atj = maxps(*atj, *x.add(i + 8 * t + j));
            }
        }
        i += BLOCK;
    }
    while i + 8 <= n {
        for (j, a0j) in a[0].iter_mut().enumerate() {
            *a0j = maxps(*a0j, *x.add(i + j));
        }
        i += 8;
    }
    // Halving tree (0,4)(1,5)(2,6)(3,7), (0,2)(1,3), (0,1), higher chain first.
    let mut w = UNROLL;
    while w > 1 {
        let h = w / 2;
        let (lo, hi) = a[..w].split_at_mut(h);
        for (l, r) in lo.iter_mut().zip(hi.iter()) {
            for (lj, rj) in l.iter_mut().zip(r.iter()) {
                *lj = maxps(*rj, *lj);
            }
        }
        w = h;
    }
    let mut mx = a[0];
    for (j, mxj) in mx.iter_mut().enumerate().take(n - i) {
        *mxj = mxj.max(*x.add(i + j));
    }
    mx
}

/// Row max as 8 lane maxima, AVX2 twin of [`rowmax8_scalar`] (same chains, same tree, same tail).
///
/// # Safety
/// `x` valid for `n` `f32`; AVX2+FMA detected by the caller.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn rowmax8_avx2(x: *const f32, n: usize) -> [f32; 8] {
    use std::arch::x86_64::*;
    let mut a = [_mm256_set1_ps(f32::NEG_INFINITY); UNROLL];
    let mut i = 0;
    while i + BLOCK <= n {
        for (t, at) in a.iter_mut().enumerate() {
            *at = _mm256_max_ps(*at, _mm256_loadu_ps(x.add(i + 8 * t)));
        }
        i += BLOCK;
    }
    while i + 8 <= n {
        a[0] = _mm256_max_ps(a[0], _mm256_loadu_ps(x.add(i)));
        i += 8;
    }
    a[0] = _mm256_max_ps(a[4], a[0]);
    a[1] = _mm256_max_ps(a[5], a[1]);
    a[2] = _mm256_max_ps(a[6], a[2]);
    a[3] = _mm256_max_ps(a[7], a[3]);
    a[0] = _mm256_max_ps(a[2], a[0]);
    a[1] = _mm256_max_ps(a[3], a[1]);
    a[0] = _mm256_max_ps(a[1], a[0]);
    let mut mx = [0.0f32; 8];
    _mm256_storeu_ps(mx.as_mut_ptr(), a[0]);
    for (j, mxj) in mx.iter_mut().enumerate().take(n - i) {
        *mxj = mxj.max(*x.add(i + j));
    }
    mx
}

// --- scalar twins (the AVX2 tail + the no-AVX2 fallback) ------------------------------------------

/// Numerically-stable softmax of one row, scalar reference. `x` and `out` may alias (in-place).
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
unsafe fn softmax_row_scalar(x: *const f32, out: *mut f32, n: usize) {
    // 1) row max: the 8-chain MAXPS pass (see the block above `rowmax8_scalar`).
    softmax_row_scalar_from_max(x, out, n, hmax8(rowmax8_scalar(x, n)));
}

/// Passes 2 and 3 of [`softmax_row_scalar`] for a given row max `m`. Split out so a test can feed
/// it the row max a *different* fold order produces and prove the output bits do not move.
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
unsafe fn softmax_row_scalar_from_max(x: *const f32, out: *mut f32, n: usize, m: f32) {
    let nb = n / 8;
    let t = nb * 8;
    // 2) e = exp(x - m); write out; accumulate the sum (lane j folds elements ≡ j (mod 8), tail
    //    into lanes 0..).
    let mut sm = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, smj) in sm.iter_mut().enumerate() {
            let e = exp1(*x.add(b + j) - m);
            *out.add(b + j) = e;
            *smj += e;
        }
    }
    for (j, smj) in sm.iter_mut().enumerate().take(n - t) {
        let e = exp1(*x.add(t + j) - m);
        *out.add(t + j) = e;
        *smj += e;
    }
    let inv = 1.0 / hsum8(sm);
    // 3) normalize (reads the just-written e values; independent multiply, order-immaterial).
    for i in 0..n {
        *out.add(i) *= inv;
    }
}

/// Numerically-stable log-softmax of one row, scalar reference. `x`/`out` may alias.
/// `out[i] = (x[i] - m) - log(s)`, where `m = max(x)` and `s = Σ exp(x[i]-m)`.
///
/// The max and sum passes are byte-identical to [`softmax_row_scalar`] (same `hmax8`, same `exp1`,
/// same `hsum8`), so `m` and `s` are bit-identical to softmax's; the only new op is **one** scalar
/// `log1(s)`. The final pass folds the two subtracts into `x[i] - (m + ls)` (one rounding), and the
/// AVX2 twin uses the identical `off = m + ls` splat — so scalar and AVX2 agree lane-for-lane.
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
unsafe fn logsoftmax_row_scalar(x: *const f32, out: *mut f32, n: usize) {
    // 1) row max — identical to softmax_row_scalar.
    logsoftmax_row_scalar_from_max(x, out, n, hmax8(rowmax8_scalar(x, n)));
}

/// Passes 2 and 3 of [`logsoftmax_row_scalar`] for a given row max `m` (split out for the same test
/// as [`softmax_row_scalar_from_max`]).
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
unsafe fn logsoftmax_row_scalar_from_max(x: *const f32, out: *mut f32, n: usize, m: f32) {
    let nb = n / 8;
    let t = nb * 8;
    // 2) s = Σ exp(x - m) — same lane structure as softmax's sum, but do NOT write `out` here (the
    //    final pass below overwrites it reading `x`, so an in-place x==out row stays correct).
    let mut sm = [0.0f32; 8];
    for s in 0..nb {
        let b = s * 8;
        for (j, smj) in sm.iter_mut().enumerate() {
            *smj += exp1(*x.add(b + j) - m);
        }
    }
    for (j, smj) in sm.iter_mut().enumerate().take(n - t) {
        *smj += exp1(*x.add(t + j) - m);
    }
    let off = m + log1(hsum8(sm)); // m + log-sum-exp; ONE scalar log — the bit-exact pivot.
                                   // 3) out[i] = (x[i] - m) - ls == x[i] - off.
    for i in 0..n {
        *out.add(i) = *x.add(i) - off;
    }
}

/// LayerNorm of one row (gamma=1, beta=0), scalar reference. `x`/`out` may alias.
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
unsafe fn layernorm_row_scalar(x: *const f32, out: *mut f32, n: usize, eps: f32) {
    let invn = 1.0 / (n as f32);
    let mean = hsum8(sum8_scalar(x, n)) * invn;
    // variance = mean of squared deviations (the addend FMA-contracts, matching the AVX2 fmadd)
    let inv = 1.0 / (hsum8(dev2_8_scalar(x, n, mean)) * invn + eps).sqrt();
    for i in 0..n {
        *out.add(i) = (*x.add(i) - mean) * inv;
    }
}

/// RMSNorm of one row (gamma=1), scalar reference. `x`/`out` may alias.
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
unsafe fn rmsnorm_row_scalar(x: *const f32, out: *mut f32, n: usize, eps: f32) {
    let invn = 1.0 / (n as f32);
    let inv = 1.0 / (hsum8(ssq8_scalar(x, n)) * invn + eps).sqrt();
    for i in 0..n {
        *out.add(i) = *x.add(i) * inv;
    }
}

/// L2-normalize one row, scalar reference: `out[i] = x[i] / sqrt(Σ x[i]² + eps)` — the unit-norm
/// projection (cosine similarity, normalized embeddings, retrieval keys). Identical to
/// [`rmsnorm_row_scalar`] **without** the `1/n` mean factor on the sum of squares: RMSNorm divides by
/// the root-*mean*-square, L2 by the root-*sum*-square. The sum-of-squares reduction is byte-for-byte
/// the same as RMSNorm's, so the AVX2 twin agrees the same way. `x`/`out` may alias (in-place).
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
unsafe fn l2norm_row_scalar(x: *const f32, out: *mut f32, n: usize, eps: f32) {
    let inv = 1.0 / (hsum8(ssq8_scalar(x, n)) + eps).sqrt();
    for i in 0..n {
        *out.add(i) = *x.add(i) * inv;
    }
}

// --- AVX2 kernels (mirror the scalar twins lane-for-lane on finite inputs) -------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_row_avx2(x: *const f32, out: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    // 1) max: the 8-chain MAXPS pass, twin of the scalar path's.
    let m = hmax8(rowmax8_avx2(x, n));
    let mb = _mm256_set1_ps(m);
    // 2) exp(x - m) + sum
    let mut sv = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb));
        _mm256_storeu_ps(out.add(i), e);
        sv = _mm256_add_ps(sv, e);
        i += 8;
    }
    let mut sm = [0.0f32; 8];
    _mm256_storeu_ps(sm.as_mut_ptr(), sv);
    for (j, smj) in sm.iter_mut().enumerate().take(n - i) {
        let e = exp1(*x.add(i + j) - m);
        *out.add(i + j) = e;
        *smj += e;
    }
    let inv = 1.0 / hsum8(sm);
    let ivb = _mm256_set1_ps(inv);
    // 3) normalize
    i = 0;
    while i + 8 <= n {
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(_mm256_loadu_ps(out.add(i)), ivb));
        i += 8;
    }
    while i < n {
        *out.add(i) *= inv;
        i += 1;
    }
}

/// AVX2 log-softmax of one row. Max + Σexp passes are byte-identical to `softmax_row_avx2` (so `m`,
/// `s` match bit-for-bit); the only new op is one scalar `log1` on the row sum. Final pass is `x - off`
/// (`off = m + log(s)`), matching the scalar twin's single subtract per element.
///
/// # Safety
/// `x`/`out` valid for `n` `f32`; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn logsoftmax_row_avx2(x: *const f32, out: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    // 1) max — identical to softmax_row_avx2.
    let m = hmax8(rowmax8_avx2(x, n));
    let mb = _mm256_set1_ps(m);
    // 2) s = Σ exp(x - m) — exp8 lanes + exp1 tail, identical to softmax's accumulation, no out store.
    let mut sv = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        sv = _mm256_add_ps(sv, exp8(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb)));
        i += 8;
    }
    let mut sm = [0.0f32; 8];
    _mm256_storeu_ps(sm.as_mut_ptr(), sv);
    for (j, smj) in sm.iter_mut().enumerate().take(n - i) {
        *smj += exp1(*x.add(i + j) - m);
    }
    let off = m + log1(hsum8(sm)); // same scalar log on the same sum bits as the scalar twin.
    let offb = _mm256_set1_ps(off);
    // 3) out = x - off.
    i = 0;
    while i + 8 <= n {
        _mm256_storeu_ps(out.add(i), _mm256_sub_ps(_mm256_loadu_ps(x.add(i)), offb));
        i += 8;
    }
    while i < n {
        *out.add(i) = *x.add(i) - off;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layernorm_row_avx2(x: *const f32, out: *mut f32, n: usize, eps: f32) {
    use std::arch::x86_64::*;
    let invn = 1.0 / (n as f32);
    let mean = hsum8(sum8_avx2(x, n)) * invn;
    let mb = _mm256_set1_ps(mean);
    let inv = 1.0 / (hsum8(dev2_8_avx2(x, n, mean)) * invn + eps).sqrt();
    let ivb = _mm256_set1_ps(inv);
    // normalize — unrolled x4 so four independent load->sub->mul->store chains are in flight (the
    // writeback is otherwise one dependent 8-lane step per iteration, the same stall the reduction
    // above had).
    let mut i = 0;
    while i + 32 <= n {
        for t in 0..4 {
            let d = _mm256_sub_ps(_mm256_loadu_ps(x.add(i + 8 * t)), mb);
            _mm256_storeu_ps(out.add(i + 8 * t), _mm256_mul_ps(d, ivb));
        }
        i += 32;
    }
    while i + 8 <= n {
        let d = _mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb);
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(d, ivb));
        i += 8;
    }
    while i < n {
        *out.add(i) = (*x.add(i) - mean) * inv;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rmsnorm_row_avx2(x: *const f32, out: *mut f32, n: usize, eps: f32) {
    use std::arch::x86_64::*;
    let invn = 1.0 / (n as f32);
    let inv = 1.0 / (hsum8(ssq8_avx2(x, n)) * invn + eps).sqrt();
    scale_row_avx2(x, out, n, _mm256_set1_ps(inv), inv);
}

/// `out[i] = x[i] * inv` over a row, unrolled x4 (the shared writeback of RMSNorm and L2-norm).
/// `ivb` must be `_mm256_set1_ps(inv)`; both are passed so the ragged tail uses the identical scalar
/// `inv` the lanes were built from.
///
/// # Safety
/// `x` and `out` valid for `n` `f32` (may alias); AVX2 detected by the caller.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_row_avx2(
    x: *const f32,
    out: *mut f32,
    n: usize,
    ivb: std::arch::x86_64::__m256,
    inv: f32,
) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 32 <= n {
        for t in 0..4 {
            _mm256_storeu_ps(
                out.add(i + 8 * t),
                _mm256_mul_ps(_mm256_loadu_ps(x.add(i + 8 * t)), ivb),
            );
        }
        i += 32;
    }
    while i + 8 <= n {
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(_mm256_loadu_ps(x.add(i)), ivb));
        i += 8;
    }
    while i < n {
        *out.add(i) = *x.add(i) * inv;
        i += 1;
    }
}

/// L2-normalize one row, AVX2 twin of [`l2norm_row_scalar`]. Identical to [`rmsnorm_row_avx2`] minus
/// the `1/n` mean factor — the `fmadd` sum-of-squares accumulator and horizontal `hsum8` are the same,
/// so it agrees with the scalar twin bit-for-bit (pinned by `scalar_matches_avx2_bit_for_bit`).
///
/// # Safety
/// `x` and `out` must each be valid for `n` `f32` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2norm_row_avx2(x: *const f32, out: *mut f32, n: usize, eps: f32) {
    use std::arch::x86_64::*;
    let inv = 1.0 / (hsum8(ssq8_avx2(x, n)) + eps).sqrt();
    scale_row_avx2(x, out, n, _mm256_set1_ps(inv), inv);
}

// --- affine variants: `out = norm(x) * gamma[i] (+ beta[i])` ---------------------------------------
// These mirror the plain twins above *exactly through the reduction* (mean/inv for LayerNorm,
// mean-square/inv for RMSNorm); they differ only in the final normalize step, which folds the per-
// column scale `gamma` and optional shift `beta` into the writeback via one FMA. Scalar `mul_add` ==
// AVX2 `fmadd`, so the affine scalar twin and AVX2 path agree bit-for-bit (pinned by the same
// `scalar_matches_avx2` test). `gamma` null means scale by 1.0, `beta` null means no shift (RMSNorm
// has scale but no shift); both, when present, are per-column arrays of length `n` shared across rows.
// The reductions are duplicated from the plain twins deliberately — keep them identical if either
// changes (the `affine_gamma1_matches_plain` test guards against drift).

/// Affine LayerNorm of one row: `out = (x-mean)/sqrt(var+eps) * gamma + beta`, scalar reference.
///
/// # Safety
/// `x`/`out` valid for `n` f32 (may alias); `gamma`/`beta` null or valid for `n` f32.
unsafe fn layernorm_affine_row_scalar(
    x: *const f32,
    out: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    n: usize,
    eps: f32,
) {
    let invn = 1.0 / (n as f32);
    let mean = hsum8(sum8_scalar(x, n)) * invn;
    let inv = 1.0 / (hsum8(dev2_8_scalar(x, n, mean)) * invn + eps).sqrt();
    for i in 0..n {
        let norm = (*x.add(i) - mean) * inv;
        let g = if gamma.is_null() { 1.0 } else { *gamma.add(i) };
        let b = if beta.is_null() { 0.0 } else { *beta.add(i) };
        *out.add(i) = norm.mul_add(g, b);
    }
}

/// Affine RMSNorm of one row: `out = x/sqrt(mean(x^2)+eps) * gamma (+ beta)`, scalar reference.
///
/// # Safety
/// `x`/`out` valid for `n` f32 (may alias); `gamma`/`beta` null or valid for `n` f32.
unsafe fn rmsnorm_affine_row_scalar(
    x: *const f32,
    out: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    n: usize,
    eps: f32,
) {
    let invn = 1.0 / (n as f32);
    let inv = 1.0 / (hsum8(ssq8_scalar(x, n)) * invn + eps).sqrt();
    for i in 0..n {
        let norm = *x.add(i) * inv;
        let g = if gamma.is_null() { 1.0 } else { *gamma.add(i) };
        let b = if beta.is_null() { 0.0 } else { *beta.add(i) };
        *out.add(i) = norm.mul_add(g, b);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layernorm_affine_row_avx2(
    x: *const f32,
    out: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    n: usize,
    eps: f32,
) {
    use std::arch::x86_64::*;
    let invn = 1.0 / (n as f32);
    // mean / variance — the shared 8-accumulator reductions (identical to layernorm_row_avx2)
    let mean = hsum8(sum8_avx2(x, n)) * invn;
    let mb = _mm256_set1_ps(mean);
    let inv = 1.0 / (hsum8(dev2_8_avx2(x, n, mean)) * invn + eps).sqrt();
    let ivb = _mm256_set1_ps(inv);
    // affine normalize: out = ((x-mean)*inv) * gamma + beta, via fmadd; unrolled x4
    let (g_null, b_null) = (gamma.is_null(), beta.is_null());
    let ones = _mm256_set1_ps(1.0);
    let zeros = _mm256_setzero_ps();
    let mut i = 0;
    while i + 32 <= n {
        for t in 0..4 {
            let k = i + 8 * t;
            let norm = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.add(k)), mb), ivb);
            let g = if g_null {
                ones
            } else {
                _mm256_loadu_ps(gamma.add(k))
            };
            let b = if b_null {
                zeros
            } else {
                _mm256_loadu_ps(beta.add(k))
            };
            _mm256_storeu_ps(out.add(k), _mm256_fmadd_ps(norm, g, b));
        }
        i += 32;
    }
    while i + 8 <= n {
        let norm = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb), ivb);
        let g = if g_null {
            ones
        } else {
            _mm256_loadu_ps(gamma.add(i))
        };
        let b = if b_null {
            zeros
        } else {
            _mm256_loadu_ps(beta.add(i))
        };
        _mm256_storeu_ps(out.add(i), _mm256_fmadd_ps(norm, g, b));
        i += 8;
    }
    while i < n {
        let norm = (*x.add(i) - mean) * inv;
        let g = if g_null { 1.0 } else { *gamma.add(i) };
        let b = if b_null { 0.0 } else { *beta.add(i) };
        *out.add(i) = norm.mul_add(g, b);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rmsnorm_affine_row_avx2(
    x: *const f32,
    out: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    n: usize,
    eps: f32,
) {
    use std::arch::x86_64::*;
    let invn = 1.0 / (n as f32);
    let inv = 1.0 / (hsum8(ssq8_avx2(x, n)) * invn + eps).sqrt();
    let ivb = _mm256_set1_ps(inv);
    let (g_null, b_null) = (gamma.is_null(), beta.is_null());
    let ones = _mm256_set1_ps(1.0);
    let zeros = _mm256_setzero_ps();
    let mut i = 0;
    while i + 32 <= n {
        for t in 0..4 {
            let k = i + 8 * t;
            let norm = _mm256_mul_ps(_mm256_loadu_ps(x.add(k)), ivb);
            let g = if g_null {
                ones
            } else {
                _mm256_loadu_ps(gamma.add(k))
            };
            let b = if b_null {
                zeros
            } else {
                _mm256_loadu_ps(beta.add(k))
            };
            _mm256_storeu_ps(out.add(k), _mm256_fmadd_ps(norm, g, b));
        }
        i += 32;
    }
    while i + 8 <= n {
        let norm = _mm256_mul_ps(_mm256_loadu_ps(x.add(i)), ivb);
        let g = if g_null {
            ones
        } else {
            _mm256_loadu_ps(gamma.add(i))
        };
        let b = if b_null {
            zeros
        } else {
            _mm256_loadu_ps(beta.add(i))
        };
        _mm256_storeu_ps(out.add(i), _mm256_fmadd_ps(norm, g, b));
        i += 8;
    }
    while i < n {
        let norm = *x.add(i) * inv;
        let g = if g_null { 1.0 } else { *gamma.add(i) };
        let b = if b_null { 0.0 } else { *beta.add(i) };
        *out.add(i) = norm.mul_add(g, b);
        i += 1;
    }
}

/// One row through the selected affine norm (LayerNorm/RMSNorm only; softmax has no affine), AVX2 when
/// available, else the scalar twin.
///
/// # Safety
/// `x`/`out` valid for `n` f32 (may alias); `gamma`/`beta` null or valid for `n` f32.
#[inline]
unsafe fn norm_affine_row(
    x: *const f32,
    out: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    n: usize,
    eps: f32,
    op: i64,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::isa::avx2() && crate::isa::fma() {
            match op {
                NORM_LAYERNORM => return layernorm_affine_row_avx2(x, out, gamma, beta, n, eps),
                NORM_RMSNORM => return rmsnorm_affine_row_avx2(x, out, gamma, beta, n, eps),
                _ => return,
            }
        }
    }
    match op {
        NORM_LAYERNORM => layernorm_affine_row_scalar(x, out, gamma, beta, n, eps),
        NORM_RMSNORM => rmsnorm_affine_row_scalar(x, out, gamma, beta, n, eps),
        _ => {}
    }
}

/// One row through the selected norm, AVX2 when available, else the scalar twin.
///
/// # Safety
/// `x` and `out` valid for `n` `f32`; they may alias.
#[inline]
unsafe fn norm_row(x: *const f32, out: *mut f32, n: usize, eps: f32, op: i64) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::isa::avx2() && crate::isa::fma() {
            match op {
                NORM_SOFTMAX => return softmax_row_avx2(x, out, n),
                NORM_LOGSOFTMAX => return logsoftmax_row_avx2(x, out, n),
                NORM_LAYERNORM => return layernorm_row_avx2(x, out, n, eps),
                NORM_RMSNORM => return rmsnorm_row_avx2(x, out, n, eps),
                NORM_L2NORM => return l2norm_row_avx2(x, out, n, eps),
                _ => return,
            }
        }
    }
    match op {
        NORM_SOFTMAX => softmax_row_scalar(x, out, n),
        NORM_LOGSOFTMAX => logsoftmax_row_scalar(x, out, n),
        NORM_LAYERNORM => layernorm_row_scalar(x, out, n, eps),
        NORM_RMSNORM => rmsnorm_row_scalar(x, out, n, eps),
        NORM_L2NORM => l2norm_row_scalar(x, out, n, eps),
        _ => {}
    }
}

/// Row-wise norm over a `[rows, cols]` matrix (serial). `eps_bits` is `eps.to_bits()` (the all-integer
/// ABI mirrors the other dispatch kernels; softmax ignores it). `x`/`out` may alias for in-place.
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` `f32` elements.
/// **Aliasing: exactly in place, otherwise disjoint** — `out` may be EXACTLY `x`, never a shifted
/// view of it.
#[no_mangle]
pub unsafe extern "C" fn wukong_norm_f32(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
    eps_bits: i64,
    op: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    let eps = f32::from_bits(eps_bits as u32);
    for r in 0..rows {
        // SAFETY: row r occupies [r*cols, r*cols + cols) ⊆ [0, rows*cols).
        unsafe { norm_row(x.add(r * cols), out.add(r * cols), cols, eps, op) };
    }
}

/// Per-element cost weight of a norm row for the shared fork gate (`par::fork_rows`): softmax and
/// log-softmax evaluate an `exp` per element; LayerNorm/RMSNorm/L2 make two or three passes with a
/// reduction between. Scheduling-only — never part of a result.
fn norm_weight(op: i64) -> usize {
    match op {
        NORM_SOFTMAX | NORM_LOGSOFTMAX => crate::par::W_TRANSCENDENTAL,
        _ => crate::par::W_ROWPASS,
    }
}

/// Multicore row-wise norm — **bit-identical** to [`wukong_norm_f32`]. Rows are independent, so each
/// is computed by the same per-row routine regardless of which thread runs it; there is no cross-row
/// combine, so the result does not depend on thread count and the interpreter (serial) agrees with the
/// native `@parallel` path (this one) exactly.
///
/// # Safety
/// `x` and `out` must each be valid for `rows * cols` `f32` elements.
/// **Aliasing:** that of [`wukong_norm_f32`] — `out` may be EXACTLY `x`, never a shifted view of
/// it.
#[no_mangle]
pub unsafe extern "C" fn wukong_norm_f32_parallel(
    x: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
    eps_bits: i64,
    op: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    // Work gate + width==1 serial fast-path (`par::fork_rows`): a one- or two-row norm, or any
    // norm whose rows x cols x per-element cost cannot amortize a fork, runs the serial sibling
    // inline instead of paying the pool handoff (bit-identical: per-row work, no cross-row combine).
    if !crate::par::fork_rows(rows as usize, cols as usize, norm_weight(op)) {
        // SAFETY: same contract as this function.
        return unsafe { wukong_norm_f32(x, out, rows, cols, eps_bits, op) };
    }
    let (rows, cols) = (rows as usize, cols as usize);
    let eps = f32::from_bits(eps_bits as u32);
    // Raw pointers cross the rayon boundary as integers (same pattern as the parallel GEMM/reduce);
    // each row is a disjoint sub-slice. Rows are mapped on the unified kernel pool by
    // `par::for_each_row` (rayon's per-row split, or the claim queue under `WUKONG_KERNEL_DYN=1`) —
    // per-row work is self-contained, so neither the pool nor the schedule touches the bits.
    let (xa, oa) = (x as usize, out as usize);
    crate::par::for_each_row(rows, move |r| {
        // SAFETY: disjoint row; pointers valid for rows*cols by contract.
        unsafe {
            norm_row(
                (xa as *const f32).add(r * cols),
                (oa as *mut f32).add(r * cols),
                cols,
                eps,
                op,
            )
        };
    });
}

/// Row-wise *affine* norm (LayerNorm/RMSNorm with a per-column `gamma` scale and optional `beta`
/// shift) over a `[rows, cols]` matrix (serial). `gamma`/`beta` are length `cols`, shared across all
/// rows (the standard transformer layout); either may be null (`gamma` null ⇒ scale 1, `beta` null ⇒
/// no shift). `eps_bits` is `eps.to_bits()`. `x`/`out` may alias for in-place. softmax never reaches
/// here (it has no affine parameters).
///
/// # Safety
/// `x`/`out` valid for `rows*cols` f32; `gamma`/`beta` null or valid for `cols` f32.
/// **Aliasing: exactly in place, otherwise disjoint** — `out` may be EXACTLY `x`, never a shifted
/// view of it; `gamma`/`beta` must not overlap `out`.
#[no_mangle]
pub unsafe extern "C" fn wukong_norm_affine_f32(
    x: *const f32,
    out: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    rows: i64,
    cols: i64,
    eps_bits: i64,
    op: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    let eps = f32::from_bits(eps_bits as u32);
    for r in 0..rows {
        // SAFETY: row r occupies [r*cols, r*cols+cols); gamma/beta indexed within [0,cols).
        unsafe {
            norm_affine_row(
                x.add(r * cols),
                out.add(r * cols),
                gamma,
                beta,
                cols,
                eps,
                op,
            )
        };
    }
}

/// Multicore affine row-wise norm — **bit-identical** to [`wukong_norm_affine_f32`] (rows are
/// independent, no cross-row combine, so thread count is irrelevant and the interpreter's serial call
/// agrees with this `@parallel` path exactly).
///
/// # Safety
/// `x`/`out` valid for `rows*cols` f32; `gamma`/`beta` null or valid for `cols` f32.
/// **Aliasing:** that of [`wukong_norm_affine_f32`] — `out` may be EXACTLY `x`; `gamma`/`beta` must
/// not overlap `out`.
#[no_mangle]
pub unsafe extern "C" fn wukong_norm_affine_f32_parallel(
    x: *const f32,
    out: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    rows: i64,
    cols: i64,
    eps_bits: i64,
    op: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    // Work gate + width==1 serial fast-path, as in `wukong_norm_f32_parallel`.
    if !crate::par::fork_rows(rows as usize, cols as usize, norm_weight(op)) {
        // SAFETY: same contract as this function.
        return unsafe { wukong_norm_affine_f32(x, out, gamma, beta, rows, cols, eps_bits, op) };
    }
    let (rows, cols) = (rows as usize, cols as usize);
    let eps = f32::from_bits(eps_bits as u32);
    // Pointers cross the rayon boundary as integers (null round-trips through 0); rows are disjoint,
    // gamma/beta are shared read-only. Rows are mapped on the unified kernel pool by
    // `par::for_each_row`, as in `wukong_norm_f32_parallel`.
    let (xa, oa, ga, ba) = (x as usize, out as usize, gamma as usize, beta as usize);
    crate::par::for_each_row(rows, move |r| {
        // SAFETY: disjoint row; gamma/beta valid for cols by contract.
        unsafe {
            norm_affine_row(
                (xa as *const f32).add(r * cols),
                (oa as *mut f32).add(r * cols),
                ga as *const f32,
                ba as *const f32,
                cols,
                eps,
                op,
            )
        };
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPS: [i64; 5] = [
        NORM_SOFTMAX,
        NORM_LOGSOFTMAX,
        NORM_LAYERNORM,
        NORM_RMSNORM,
        NORM_L2NORM,
    ];
    const EPS: f32 = 1e-5;

    // Deterministic, mildly varied input (no RNG — reproducible).
    fn fill(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.017).sin() * 2.3 - 0.4)
            .collect()
    }

    #[test]
    fn softmax_exp_log_exceptional_rows() {
        for n in [1usize, 7, 8, 9, 31, 32, 33, 65] {
            for op in [NORM_SOFTMAX, NORM_LOGSOFTMAX] {
                // Independent expected classes: every fully masked row is undefined, including
                // n=1, which formerly returned approximately one through exp(NaN) saturation.
                for special in [
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    f32::from_bits(0x7fc0_1234),
                ] {
                    let x = vec![special; n];
                    let mut scalar = vec![0.0; n];
                    // SAFETY: `x`/`scalar` sized n (special-value scalar twin).
                    unsafe {
                        if op == NORM_SOFTMAX {
                            softmax_row_scalar(x.as_ptr(), scalar.as_mut_ptr(), n);
                        } else {
                            logsoftmax_row_scalar(x.as_ptr(), scalar.as_mut_ptr(), n);
                        }
                    }
                    assert!(
                        scalar.iter().all(|x| x.is_nan()),
                        "op={op} n={n} x={special}"
                    );
                    #[cfg(target_arch = "x86_64")]
                    if crate::isa::avx2() && crate::isa::fma() {
                        let mut lanes = vec![0.0; n];
                        // SAFETY: AVX2+FMA confirmed by the enclosing feature guard; `x`/`lanes` sized n.
                        unsafe {
                            if op == NORM_SOFTMAX {
                                softmax_row_avx2(x.as_ptr(), lanes.as_mut_ptr(), n);
                            } else {
                                logsoftmax_row_avx2(x.as_ptr(), lanes.as_mut_ptr(), n);
                            }
                        }
                        for (a, b) in scalar.iter().zip(lanes) {
                            assert_eq!(a.to_bits(), b.to_bits(), "op={op} n={n} x={special}");
                        }
                    }
                }
                let mut x = vec![f32::NEG_INFINITY; n];
                x[n / 2] = 3.0;
                let mut out = vec![0.0; n];
                // SAFETY: `x`/`out` sized n (a single row).
                unsafe { wukong_norm_f32(x.as_ptr(), out.as_mut_ptr(), 1, n as i64, 0, op) };
                for (i, actual) in out.into_iter().enumerate() {
                    let expected: f32 = match (op, i == n / 2) {
                        (NORM_SOFTMAX, true) => 1.0,
                        (NORM_SOFTMAX, false) | (NORM_LOGSOFTMAX, true) => 0.0,
                        _ => f32::NEG_INFINITY,
                    };
                    assert_eq!(actual.to_bits(), expected.to_bits(), "op={op} n={n} i={i}");
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !(crate::isa::avx2() && crate::isa::fma()) {
            return;
        }
        // Sizes exercising the 8-lane body, a non-multiple-of-8 tail, and rows shorter than 8 —
        // plus, since the reduction runs `UNROLL` = 8 accumulators over `BLOCK` = 64 elements and
        // the writeback unrolls x4 (32 elements), every boundary those two blockings introduce:
        // just under / on / just over a 64-block (63, 64, 65, 71, 72, 127..129, 191..193), a size
        // with a full block plus a partial 32-group plus a ragged tail (100, 200), and the 768 the
        // transformer-row benchmarks actually use.
        for &n in &[
            1usize, 3, 7, 8, 9, 15, 16, 17, 31, 63, 64, 65, 71, 72, 100, 127, 128, 129, 191, 192,
            193, 200, 257, 512, 768,
        ] {
            let x = fill(n);
            for &op in &OPS {
                let mut a = vec![0.0f32; n];
                let mut b = vec![0.0f32; n];
                // SAFETY: `x`/`a`/`b` sized n; the scalar and avx2 twins each read n (AVX2+FMA guarded above).
                unsafe {
                    match op {
                        NORM_SOFTMAX => {
                            softmax_row_scalar(x.as_ptr(), a.as_mut_ptr(), n);
                            softmax_row_avx2(x.as_ptr(), b.as_mut_ptr(), n);
                        }
                        NORM_LOGSOFTMAX => {
                            logsoftmax_row_scalar(x.as_ptr(), a.as_mut_ptr(), n);
                            logsoftmax_row_avx2(x.as_ptr(), b.as_mut_ptr(), n);
                        }
                        NORM_LAYERNORM => {
                            layernorm_row_scalar(x.as_ptr(), a.as_mut_ptr(), n, EPS);
                            layernorm_row_avx2(x.as_ptr(), b.as_mut_ptr(), n, EPS);
                        }
                        NORM_L2NORM => {
                            l2norm_row_scalar(x.as_ptr(), a.as_mut_ptr(), n, EPS);
                            l2norm_row_avx2(x.as_ptr(), b.as_mut_ptr(), n, EPS);
                        }
                        _ => {
                            rmsnorm_row_scalar(x.as_ptr(), a.as_mut_ptr(), n, EPS);
                            rmsnorm_row_avx2(x.as_ptr(), b.as_mut_ptr(), n, EPS);
                        }
                    }
                }
                for i in 0..n {
                    assert_eq!(
                        a[i].to_bits(),
                        b[i].to_bits(),
                        "scalar != avx2 at n={n} op={op} i={i}: {} vs {}",
                        a[i],
                        b[i]
                    );
                }
            }
        }
    }

    /// A row whose maximum shares an 8-lane slot with a later NaN. The AVX2 body folds the max with
    /// `_mm256_max_ps` (= `a > b ? a : b`, so a NaN in the freshly *loaded* operand wins and poisons
    /// the lane). The scalar twin uses the same maxps rule; its former f32::max rule kept the peak
    /// and picked a different row maximum. Both paths must retain the same reduction order,
    /// including now that exp propagates NaNs into the output.
    ///
    /// `peak = nan_at - 8` puts the peak in the same lane and an earlier chunk, which is the only
    /// arrangement that exposes it: a NaN elsewhere merely poisons a lane that does not hold the row
    /// maximum, and `hmax8` then drops it.
    /// The lengths every blocking seam in this module lives on: shorter than one 8-lane vector,
    /// around the `BLOCK` = 64 body, around the x4-unrolled (32-element) writeback groups, and a
    /// spread of ragged tails. Shared by the twin tests so a new seam is added in one place.
    const SEAM_LENS: [usize; 25] = [
        1, 3, 7, 8, 9, 15, 16, 17, 31, 63, 64, 65, 71, 72, 100, 127, 128, 129, 191, 192, 193, 200,
        257, 512, 768,
    ];

    /// Deterministic pathological rows: every value class an f32 reduction can round differently on.
    /// Each entry is `(name, f(i) -> f32)`; the row is that function over `0..n`, so every length in
    /// `SEAM_LENS` lands the special values in different lanes, different accumulators and different
    /// blocks. Sign-alternating patterns are used deliberately: with 8 accumulators, cancellation is
    /// where a lane-layout desync between the twins would first show.
    #[allow(clippy::type_complexity)]
    fn pathological() -> Vec<(&'static str, fn(usize) -> f32)> {
        vec![
            ("signed-zero", |i| if i % 2 == 0 { 0.0 } else { -0.0 }),
            ("zero-and-one", |i| if i % 3 == 0 { -0.0 } else { 1.0 }),
            ("nan-sparse", |i| if i % 11 == 7 { f32::NAN } else { 1.5 }),
            ("nan-dense", |i| if i % 2 == 0 { f32::NAN } else { -2.25 }),
            ("pos-inf", |i| if i % 13 == 5 { f32::INFINITY } else { 0.5 }),
            ("both-inf", |i| match i % 7 {
                3 => f32::INFINITY,
                5 => f32::NEG_INFINITY,
                _ => 1.0,
            }),
            // Subnormals: the smallest positive, the largest subnormal, and their negatives.
            ("subnormal", |i| match i % 4 {
                0 => f32::from_bits(1),
                1 => f32::from_bits(0x007f_ffff),
                2 => -f32::from_bits(1),
                _ => -f32::from_bits(0x0040_0000),
            }),
            // Subnormals mixed with normals, so `x*x` underflows to zero inside a live accumulator.
            ("subnormal-mixed", |i| {
                if i % 5 == 0 {
                    f32::from_bits(3)
                } else {
                    1.0
                }
            }),
            // Round-to-nearest-EVEN ties: 2^24 + 1 is exactly halfway between two f32, and so is
            // 2^24 + 3. Which way each rounds depends only on the accumulator's current low bit, so a
            // twin that folded elements into a different lane would diverge here first.
            ("tie-2p24", |i| if i % 8 == 0 { 16777216.0 } else { 1.0 }),
            (
                "tie-2p24-neg",
                |i| {
                    if i % 8 == 4 {
                        -16777216.0
                    } else {
                        1.0
                    }
                },
            ),
            // Squares that overflow to +inf partway through a live accumulator.
            ("overflow-sq", |i| if i % 9 == 2 { 2.0e19 } else { 3.0 }),
            // Wide dynamic range: every addend is absorbed by the running total.
            ("absorbed", |i| if i % 6 == 0 { 1.0e20 } else { 1.0e-20 }),
            // Exact cancellation to a signed zero in some lanes and not others.
            ("cancel", |i| if i % 2 == 0 { 1.0e7 } else { -1.0e7 }),
            // --- patterns that pin the COMBINE TREE, not the value classes -------------------
            // Every pattern above holds values whose accumulator sums are exact (small integers,
            // powers of two) or saturated (NaN/inf), so they are insensitive to the order in which
            // the eight accumulators are folded — a twin that used a different but equally valid
            // halving tree would pass all of them. These two do not: full random mantissas make
            // every partial sum inexact, and the second one additionally spreads the accumulators
            // across 2^56 of dynamic range so that which pair is added first decides what is lost.
            // Verified by mutation: swapping the AVX2 combine to the (0,1)(2,3)(4,5)(6,7) tree is
            // bit-invisible to every other pattern here and to `scalar_matches_avx2_bit_for_bit`,
            // and is caught by these.
            ("random-mantissa", |i| rand_mantissa(i)),
            ("mixed-scale", |i| {
                let t = ((i / 8) % 8) as i32;
                rand_mantissa(i) * exp2i(4 * t - 14)
            }),
        ]
    }

    /// A full-mantissa f32 in `[1, 2)` from a multiplicative hash of `i` — deterministic, no RNG.
    /// The point is the 23 significand bits: sums of these round differently under different
    /// association, which is what pins the accumulator combine tree.
    fn rand_mantissa(i: usize) -> f32 {
        f32::from_bits(0x3f80_0000 | ((i as u32).wrapping_mul(2_654_435_761) >> 9))
    }

    /// `2^k` exactly, for `|k| < 127` — an exact scale, so it moves the exponent without perturbing
    /// the mantissa the pattern is carrying.
    fn exp2i(k: i32) -> f32 {
        f32::from_bits((((k + 127) as u32) & 0xff) << 23)
    }

    /// The three reductions the multi-accumulator rewrite replaced — `ssq8` (RMSNorm / L2-norm),
    /// `sum8` (the LayerNorm mean pass) and `dev2_8` (the LayerNorm variance pass) — must agree
    /// BIT-FOR-BIT between the AVX2 kernel and its scalar twin on every value class an f32 reduction
    /// can round differently on: NaN, +/-0, +/-inf, subnormals, round-to-even ties, overflow to inf
    /// and total cancellation. All eight lane accumulators are compared, not the folded scalar: a
    /// lane-layout desync between the twins can cancel in the horizontal sum and is invisible in the
    /// final row.
    ///
    /// This is the coverage `scalar_matches_avx2_bit_for_bit` does not give (its input is a smooth
    /// finite sine, and it checks the normalized row rather than the accumulators) and that
    /// `scalar_matches_avx2_on_nan_rows` does not give either (it exercises only softmax and
    /// log-softmax, whose max pass these three reductions are not part of).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn reduction_twins_agree_bit_for_bit_on_pathological_rows() {
        if !(crate::isa::avx2() && crate::isa::fma()) {
            return;
        }
        // Means chosen so `x - mean` itself produces the hard cases: inf - inf = NaN, a NaN mean
        // poisoning every lane, and an exact subtraction that cancels to zero.
        let means: [f32; 6] = [
            0.0,
            -0.0,
            1.0,
            f32::NAN,
            f32::INFINITY,
            f32::from_bits(0x007f_ffff),
        ];
        for (name, gen) in pathological() {
            for &n in &SEAM_LENS {
                let x: Vec<f32> = (0..n).map(gen).collect();
                let cmp = |what: &str, a: [f32; 8], b: [f32; 8], extra: String| {
                    for j in 0..8 {
                        assert_eq!(
                            a[j].to_bits(),
                            b[j].to_bits(),
                            "{what} scalar != avx2 on '{name}' n={n} lane={j}{extra}: \
                             {:#010x} ({}) vs {:#010x} ({})",
                            a[j].to_bits(),
                            a[j],
                            b[j].to_bits(),
                            b[j]
                        );
                    }
                };
                // SAFETY: both twins read exactly `n` f32 from `x`; AVX2+FMA detected above.
                unsafe {
                    cmp(
                        "ssq8",
                        ssq8_scalar(x.as_ptr(), n),
                        ssq8_avx2(x.as_ptr(), n),
                        String::new(),
                    );
                    cmp(
                        "sum8",
                        sum8_scalar(x.as_ptr(), n),
                        sum8_avx2(x.as_ptr(), n),
                        String::new(),
                    );
                    for &m in &means {
                        cmp(
                            "dev2_8",
                            dev2_8_scalar(x.as_ptr(), n, m),
                            dev2_8_avx2(x.as_ptr(), n, m),
                            format!(" mean={m}"),
                        );
                    }
                }
            }
        }
    }

    /// The whole row, end to end, on the same pathological inputs: the three ops whose reductions
    /// the rewrite touched (LayerNorm, RMSNorm, L2-norm) must produce bit-identical output from the
    /// AVX2 kernel and the scalar twin. This covers the finalize (`1/sqrt`, the mean divide) and the
    /// writeback, which the accumulator test above deliberately stops short of, and it is where a
    /// NaN or an inf that survived the reduction has to keep propagating identically.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn norm_rows_agree_bit_for_bit_on_pathological_rows() {
        if !(crate::isa::avx2() && crate::isa::fma()) {
            return;
        }
        for (name, gen) in pathological() {
            for &n in &SEAM_LENS {
                let x: Vec<f32> = (0..n).map(gen).collect();
                for &op in &[
                    NORM_LAYERNORM,
                    NORM_RMSNORM,
                    NORM_L2NORM,
                    NORM_SOFTMAX,
                    NORM_LOGSOFTMAX,
                ] {
                    let mut a = vec![0.0f32; n];
                    let mut b = vec![0.0f32; n];
                    // SAFETY: both write exactly `n` f32; AVX2+FMA detected above.
                    unsafe {
                        match op {
                            NORM_SOFTMAX => {
                                softmax_row_scalar(x.as_ptr(), a.as_mut_ptr(), n);
                                softmax_row_avx2(x.as_ptr(), b.as_mut_ptr(), n);
                            }
                            NORM_LOGSOFTMAX => {
                                logsoftmax_row_scalar(x.as_ptr(), a.as_mut_ptr(), n);
                                logsoftmax_row_avx2(x.as_ptr(), b.as_mut_ptr(), n);
                            }
                            NORM_LAYERNORM => {
                                layernorm_row_scalar(x.as_ptr(), a.as_mut_ptr(), n, EPS);
                                layernorm_row_avx2(x.as_ptr(), b.as_mut_ptr(), n, EPS);
                            }
                            NORM_RMSNORM => {
                                rmsnorm_row_scalar(x.as_ptr(), a.as_mut_ptr(), n, EPS);
                                rmsnorm_row_avx2(x.as_ptr(), b.as_mut_ptr(), n, EPS);
                            }
                            _ => {
                                l2norm_row_scalar(x.as_ptr(), a.as_mut_ptr(), n, EPS);
                                l2norm_row_avx2(x.as_ptr(), b.as_mut_ptr(), n, EPS);
                            }
                        }
                    }
                    for i in 0..n {
                        assert_eq!(
                            a[i].to_bits(),
                            b[i].to_bits(),
                            "scalar != avx2 on '{name}' n={n} op={op} i={i}: {:#010x} ({}) vs {:#010x} ({})",
                            a[i].to_bits(),
                            a[i],
                            b[i].to_bits(),
                            b[i]
                        );
                    }
                }
            }
        }
    }

    /// The row max is a fold over MAXPS chains, and its *lanes* must agree bit-for-bit between the
    /// twins on every pathological row (NaN in a non-zero chain included) — the accumulator-level
    /// check `norm_rows_agree_bit_for_bit_on_pathological_rows` stops one step short of. On NaN-free
    /// rows the folded maximum must also equal the plain maximum as a value.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn rowmax_twins_agree_on_pathological_rows() {
        if !(crate::isa::avx2() && crate::isa::fma()) {
            return;
        }
        let mut rows: Vec<(&str, fn(usize) -> f32)> = pathological();
        rows.push(("finite", |i| ((i as f32) * 0.017).sin() * 2.3 - 0.4));
        for (name, gen) in rows {
            for &n in &SEAM_LENS {
                let x: Vec<f32> = (0..n).map(gen).collect();
                // SAFETY: both twins read exactly `n` f32; AVX2+FMA detected above.
                let (s, v) =
                    // SAFETY: both twins read exactly n f32; AVX2+FMA detected above.
                    unsafe { (rowmax8_scalar(x.as_ptr(), n), rowmax8_avx2(x.as_ptr(), n)) };
                for j in 0..8 {
                    assert_eq!(
                        s[j].to_bits(),
                        v[j].to_bits(),
                        "rowmax lane {j} scalar != avx2 on '{name}' n={n}"
                    );
                }
                if x.iter().all(|v| !v.is_nan()) {
                    let want = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let got = hmax8(s);
                    assert!(got == want, "rowmax on '{name}' n={n}: {got} vs {want}");
                }
            }
        }
    }

    /// The eight-chain max pass must not move a single softmax / log-softmax output bit on NaN-free
    /// data. Feed the passes after the max the row max the *single-chain* kernel computed — spelled
    /// out here independently — and compare against the shipped kernel. The signed-zero rows are the
    /// sharp case: there the two fold orders can disagree on the SIGN of a zero maximum, and the
    /// claim is that `exp(x - (+0))`, `exp(x - (-0))` and `m + log(s)` cannot see it.
    #[test]
    fn eight_chain_row_max_leaves_softmax_bits_unchanged() {
        let one_chain_max = |x: &[f32]| {
            let n = x.len();
            let t = n / 8 * 8;
            let mut mx = [f32::NEG_INFINITY; 8];
            for s in 0..n / 8 {
                for (j, mxj) in mx.iter_mut().enumerate() {
                    *mxj = maxps(*mxj, x[s * 8 + j]);
                }
            }
            for (j, mxj) in mx.iter_mut().enumerate().take(n - t) {
                *mxj = mxj.max(x[t + j]);
            }
            hmax8(mx)
        };
        let mut rows: Vec<(&str, fn(usize) -> f32)> = pathological();
        rows.push(("finite", |i| ((i as f32) * 0.017).sin() * 2.3 - 0.4));
        rows.push(("zero-peak", |i| match i % 5 {
            0 => 0.0,
            1 => -0.0,
            _ => -(i as f32) * 0.25,
        }));
        for (name, gen) in rows {
            for &n in &SEAM_LENS {
                let x: Vec<f32> = (0..n).map(gen).collect();
                if x.iter().any(|v| v.is_nan()) {
                    continue;
                }
                let m_old = one_chain_max(&x);
                let (mut a, mut b) = (vec![0.0f32; n], vec![0.0f32; n]);
                // SAFETY: every call reads `n` f32 from `x` and writes `n` f32.
                unsafe {
                    softmax_row_scalar(x.as_ptr(), a.as_mut_ptr(), n);
                    softmax_row_scalar_from_max(x.as_ptr(), b.as_mut_ptr(), n, m_old);
                }
                for i in 0..n {
                    assert_eq!(
                        a[i].to_bits(),
                        b[i].to_bits(),
                        "softmax moved on '{name}' n={n} i={i}"
                    );
                }
                // SAFETY: as above.
                unsafe {
                    logsoftmax_row_scalar(x.as_ptr(), a.as_mut_ptr(), n);
                    logsoftmax_row_scalar_from_max(x.as_ptr(), b.as_mut_ptr(), n, m_old);
                }
                for i in 0..n {
                    assert_eq!(
                        a[i].to_bits(),
                        b[i].to_bits(),
                        "log-softmax moved on '{name}' n={n} i={i}"
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_on_nan_rows() {
        if !(crate::isa::avx2() && crate::isa::fma()) {
            return;
        }
        for &n in &[16usize, 17, 24, 33, 100] {
            for nan_at in 8..n {
                let mut x = vec![0.0f32; n];
                x[nan_at - 8] = 1.0; // the row maximum, same lane as the NaN, earlier chunk
                x[nan_at] = f32::NAN;
                for &op in &[NORM_SOFTMAX, NORM_LOGSOFTMAX] {
                    let mut a = vec![0.0f32; n];
                    let mut b = vec![0.0f32; n];
                    // SAFETY: `x`/`a`/`b` sized n; the scalar and avx2 twins each read n (AVX2+FMA guarded above).
                    unsafe {
                        if op == NORM_SOFTMAX {
                            softmax_row_scalar(x.as_ptr(), a.as_mut_ptr(), n);
                            softmax_row_avx2(x.as_ptr(), b.as_mut_ptr(), n);
                        } else {
                            logsoftmax_row_scalar(x.as_ptr(), a.as_mut_ptr(), n);
                            logsoftmax_row_avx2(x.as_ptr(), b.as_mut_ptr(), n);
                        }
                    }
                    for i in 0..n {
                        assert_eq!(
                            a[i].to_bits(),
                            b[i].to_bits(),
                            "scalar != avx2 on NaN row n={n} nan_at={nan_at} op={op} i={i}: {} vs {}",
                            a[i],
                            b[i]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn serial_matches_parallel_bit_for_bit() {
        let (rows, cols) = (37usize, 100usize);
        let x = fill(rows * cols);
        for &op in &OPS {
            let mut s = vec![0.0f32; rows * cols];
            let mut p = vec![0.0f32; rows * cols];
            // SAFETY: `s`/`p` sized rows*cols (serial vs parallel).
            unsafe {
                wukong_norm_f32(
                    x.as_ptr(),
                    s.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                    EPS.to_bits() as i64,
                    op,
                );
                wukong_norm_f32_parallel(
                    x.as_ptr(),
                    p.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                    EPS.to_bits() as i64,
                    op,
                );
            }
            for i in 0..rows * cols {
                assert_eq!(
                    s[i].to_bits(),
                    p[i].to_bits(),
                    "serial != parallel op={op} i={i}"
                );
            }
        }
    }

    #[test]
    fn in_place_matches_out_of_place() {
        // The recognizer allows in-place (x == out); it must equal the out-of-place result.
        let n = 130usize;
        let x = fill(n);
        for &op in &OPS {
            let mut oop = vec![0.0f32; n];
            let mut ip = x.clone();
            // SAFETY: `oop` sized n (out-of-place) and `ip` an in-place clone.
            unsafe {
                wukong_norm_f32(
                    x.as_ptr(),
                    oop.as_mut_ptr(),
                    1,
                    n as i64,
                    EPS.to_bits() as i64,
                    op,
                );
                wukong_norm_f32(
                    ip.as_ptr(),
                    ip.as_mut_ptr(),
                    1,
                    n as i64,
                    EPS.to_bits() as i64,
                    op,
                );
            }
            for i in 0..n {
                assert_eq!(
                    oop[i].to_bits(),
                    ip[i].to_bits(),
                    "in-place != out-of-place op={op} i={i}"
                );
            }
        }
    }

    #[test]
    fn matches_f64_reference() {
        // Tolerance check against an f64 row computation for each norm.
        let n = 512usize;
        let x = fill(n);
        let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();

        // softmax
        let mut sm = vec![0.0f32; n];
        // SAFETY: `x`/`sm` sized n (softmax).
        unsafe {
            wukong_norm_f32(x.as_ptr(), sm.as_mut_ptr(), 1, n as i64, 0, NORM_SOFTMAX);
        }
        let mx = xd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let den: f64 = xd.iter().map(|&v| (v - mx).exp()).sum();
        let mut tot = 0.0f64;
        for i in 0..n {
            let want = (xd[i] - mx).exp() / den;
            assert!((sm[i] as f64 - want).abs() < 1e-5, "softmax i={i}");
            tot += sm[i] as f64;
        }
        assert!((tot - 1.0).abs() < 1e-4, "softmax sums to {tot}");

        // log-softmax — reuse `mx` and `den` (= Σ exp(x-mx)) from the softmax block; ls = log(den).
        let mut lsm = vec![0.0f32; n];
        // SAFETY: `x`/`lsm` sized n (log-softmax).
        unsafe {
            wukong_norm_f32(
                x.as_ptr(),
                lsm.as_mut_ptr(),
                1,
                n as i64,
                0,
                NORM_LOGSOFTMAX,
            );
        }
        let ls = den.ln();
        for i in 0..n {
            let want = (xd[i] - mx) - ls;
            assert!(
                (lsm[i] as f64 - want).abs() < 1e-4,
                "logsoftmax i={i}: {} vs {want}",
                lsm[i]
            );
            // exp(log-softmax) must equal softmax — the two are consistent.
            assert!(
                ((lsm[i] as f64).exp() - sm[i] as f64).abs() < 1e-4,
                "exp(logsoftmax)!=softmax i={i}"
            );
        }

        // layernorm
        let mut ln = vec![0.0f32; n];
        // SAFETY: `x`/`ln` sized n (layernorm).
        unsafe {
            wukong_norm_f32(
                x.as_ptr(),
                ln.as_mut_ptr(),
                1,
                n as i64,
                EPS.to_bits() as i64,
                NORM_LAYERNORM,
            );
        }
        let mean = xd.iter().sum::<f64>() / n as f64;
        let var = xd.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        let inv = 1.0 / (var + EPS as f64).sqrt();
        for i in 0..n {
            let want = (xd[i] - mean) * inv;
            assert!(
                (ln[i] as f64 - want).abs() < 1e-3,
                "layernorm i={i}: {} vs {want}",
                ln[i]
            );
        }

        // rmsnorm
        let mut rn = vec![0.0f32; n];
        // SAFETY: `x`/`rn` sized n (rmsnorm).
        unsafe {
            wukong_norm_f32(
                x.as_ptr(),
                rn.as_mut_ptr(),
                1,
                n as i64,
                EPS.to_bits() as i64,
                NORM_RMSNORM,
            );
        }
        let ms = xd.iter().map(|&v| v * v).sum::<f64>() / n as f64;
        let invr = 1.0 / (ms + EPS as f64).sqrt();
        for i in 0..n {
            let want = xd[i] * invr;
            assert!(
                (rn[i] as f64 - want).abs() < 1e-3,
                "rmsnorm i={i}: {} vs {want}",
                rn[i]
            );
        }

        // l2norm — `out[i] = x[i] / sqrt(Σ x[i]² + eps)` (RMSNorm without the mean divisor).
        let mut l2 = vec![0.0f32; n];
        // SAFETY: `x`/`l2` sized n (l2norm).
        unsafe {
            wukong_norm_f32(
                x.as_ptr(),
                l2.as_mut_ptr(),
                1,
                n as i64,
                EPS.to_bits() as i64,
                NORM_L2NORM,
            );
        }
        let ssq = xd.iter().map(|&v| v * v).sum::<f64>();
        let invl = 1.0 / (ssq + EPS as f64).sqrt();
        for i in 0..n {
            let want = xd[i] * invl;
            assert!(
                (l2[i] as f64 - want).abs() < 1e-4,
                "l2norm i={i}: {} vs {want}",
                l2[i]
            );
        }
    }

    /// Edge: zero or negative `rows`/`cols` are a no-op (don't write, don't panic) — for all four
    /// C-ABI entries and every op code. The guard is load-bearing for memory safety, not tidiness:
    /// `rows = -1` past it would reach `rows as usize` = 2^64-1 and walk `x.add(r * cols)` off the end
    /// of the buffer. A negative shape is reachable from a .wk program with a runtime loop bound, so
    /// the guard must not be silently deletable. (logsoftmax.rs pins the same property for its four
    /// entries; norm.rs had no such test.)
    #[test]
    fn degenerate_shapes_are_noops() {
        let x = fill(8);
        let gamma = fill_off(8, 0.5);
        let beta = fill_off(8, 1.3);
        let mut buf = vec![42.0f32; 8];
        for &(rows, cols) in &[(0i64, 4i64), (2, 0), (-1, 4), (3, -2), (0, 0), (-1, -1)] {
            for &op in &OPS {
                // SAFETY: a degenerate rows/cols pair is a no-op; no pointer is read.
                unsafe {
                    wukong_norm_f32(
                        x.as_ptr(),
                        buf.as_mut_ptr(),
                        rows,
                        cols,
                        EPS.to_bits() as i64,
                        op,
                    );
                    wukong_norm_f32_parallel(
                        x.as_ptr(),
                        buf.as_mut_ptr(),
                        rows,
                        cols,
                        EPS.to_bits() as i64,
                        op,
                    );
                    wukong_norm_affine_f32(
                        x.as_ptr(),
                        buf.as_mut_ptr(),
                        gamma.as_ptr(),
                        beta.as_ptr(),
                        rows,
                        cols,
                        EPS.to_bits() as i64,
                        op,
                    );
                    wukong_norm_affine_f32_parallel(
                        x.as_ptr(),
                        buf.as_mut_ptr(),
                        gamma.as_ptr(),
                        beta.as_ptr(),
                        rows,
                        cols,
                        EPS.to_bits() as i64,
                        op,
                    );
                }
                assert!(
                    buf.iter().all(|&v| v == 42.0),
                    "no-op must not write: rows={rows} cols={cols} op={op}"
                );
            }
        }
    }

    // --- affine (gamma/beta) variants ----------------------------------------------------------

    // A second deterministic stream for gamma/beta (distinct from `fill` so the affine params are not
    // accidentally equal to the data or to each other).
    fn fill_off(n: usize, off: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.011 + off).cos() * 1.7 + 0.3)
            .collect()
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn affine_scalar_matches_avx2_bit_for_bit() {
        if !(crate::isa::avx2() && crate::isa::fma()) {
            return;
        }
        // Same boundary set as the plain twin: the affine kernels share its 8-accumulator
        // reductions and carry their own x4-unrolled writeback, so both blockings need covering.
        for &n in &[
            1usize, 3, 7, 8, 9, 15, 16, 17, 31, 63, 64, 65, 71, 72, 100, 127, 128, 129, 191, 192,
            193, 200, 257, 512, 768,
        ] {
            let x = fill(n);
            let gamma = fill_off(n, 0.5);
            let beta = fill_off(n, 1.3);
            // LayerNorm exercises gamma+beta; RMSNorm exercises gamma with a null beta.
            for &(op, with_beta) in &[(NORM_LAYERNORM, true), (NORM_RMSNORM, false)] {
                let bp = if with_beta {
                    beta.as_ptr()
                } else {
                    std::ptr::null()
                };
                let mut a = vec![0.0f32; n];
                let mut b = vec![0.0f32; n];
                // SAFETY: `x`/`a`/`b` sized n; the scalar and avx2 affine twins each read n (AVX2+FMA guarded above).
                unsafe {
                    if op == NORM_LAYERNORM {
                        layernorm_affine_row_scalar(
                            x.as_ptr(),
                            a.as_mut_ptr(),
                            gamma.as_ptr(),
                            bp,
                            n,
                            EPS,
                        );
                        layernorm_affine_row_avx2(
                            x.as_ptr(),
                            b.as_mut_ptr(),
                            gamma.as_ptr(),
                            bp,
                            n,
                            EPS,
                        );
                    } else {
                        rmsnorm_affine_row_scalar(
                            x.as_ptr(),
                            a.as_mut_ptr(),
                            gamma.as_ptr(),
                            bp,
                            n,
                            EPS,
                        );
                        rmsnorm_affine_row_avx2(
                            x.as_ptr(),
                            b.as_mut_ptr(),
                            gamma.as_ptr(),
                            bp,
                            n,
                            EPS,
                        );
                    }
                }
                for i in 0..n {
                    assert_eq!(
                        a[i].to_bits(),
                        b[i].to_bits(),
                        "affine scalar != avx2 n={n} op={op} i={i}: {} vs {}",
                        a[i],
                        b[i]
                    );
                }
            }
        }
    }

    #[test]
    fn affine_serial_matches_parallel_bit_for_bit() {
        let (rows, cols) = (37usize, 100usize);
        let x = fill(rows * cols);
        let gamma = fill_off(cols, 0.5);
        let beta = fill_off(cols, 1.3);
        for &op in &[NORM_LAYERNORM, NORM_RMSNORM] {
            let mut s = vec![0.0f32; rows * cols];
            let mut p = vec![0.0f32; rows * cols];
            // SAFETY: `s`/`p` sized rows*cols (serial vs parallel).
            unsafe {
                wukong_norm_affine_f32(
                    x.as_ptr(),
                    s.as_mut_ptr(),
                    gamma.as_ptr(),
                    beta.as_ptr(),
                    rows as i64,
                    cols as i64,
                    EPS.to_bits() as i64,
                    op,
                );
                wukong_norm_affine_f32_parallel(
                    x.as_ptr(),
                    p.as_mut_ptr(),
                    gamma.as_ptr(),
                    beta.as_ptr(),
                    rows as i64,
                    cols as i64,
                    EPS.to_bits() as i64,
                    op,
                );
            }
            for i in 0..rows * cols {
                assert_eq!(
                    s[i].to_bits(),
                    p[i].to_bits(),
                    "affine serial != parallel op={op} i={i}"
                );
            }
        }
    }

    #[test]
    fn affine_matches_f64_reference() {
        let n = 512usize;
        let x = fill(n);
        let gamma = fill_off(n, 0.5);
        let beta = fill_off(n, 1.3);
        let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();

        // LayerNorm: (x-mean)/sqrt(var+eps) * gamma + beta
        let mut ln = vec![0.0f32; n];
        // SAFETY: `x`/`ln` sized n; gamma/beta sized cols for the affine layernorm.
        unsafe {
            wukong_norm_affine_f32(
                x.as_ptr(),
                ln.as_mut_ptr(),
                gamma.as_ptr(),
                beta.as_ptr(),
                1,
                n as i64,
                EPS.to_bits() as i64,
                NORM_LAYERNORM,
            );
        }
        let mean = xd.iter().sum::<f64>() / n as f64;
        let var = xd.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        let inv = 1.0 / (var + EPS as f64).sqrt();
        for i in 0..n {
            let want = (xd[i] - mean) * inv * gamma[i] as f64 + beta[i] as f64;
            assert!(
                (ln[i] as f64 - want).abs() < 2e-3,
                "affine layernorm i={i}: {} vs {want}",
                ln[i]
            );
        }

        // RMSNorm: x/sqrt(mean(x^2)+eps) * gamma  (no beta)
        let mut rn = vec![0.0f32; n];
        // SAFETY: `x`/`rn` sized n; gamma sized cols for the affine rmsnorm.
        unsafe {
            wukong_norm_affine_f32(
                x.as_ptr(),
                rn.as_mut_ptr(),
                gamma.as_ptr(),
                std::ptr::null(),
                1,
                n as i64,
                EPS.to_bits() as i64,
                NORM_RMSNORM,
            );
        }
        let ms = xd.iter().map(|&v| v * v).sum::<f64>() / n as f64;
        let invr = 1.0 / (ms + EPS as f64).sqrt();
        for i in 0..n {
            let want = xd[i] * invr * gamma[i] as f64;
            assert!(
                (rn[i] as f64 - want).abs() < 2e-3,
                "affine rmsnorm i={i}: {} vs {want}",
                rn[i]
            );
        }
    }

    #[test]
    fn affine_gamma1_beta0_matches_plain() {
        // gamma=ones, beta=zeros affine must equal the plain norm (within fma-vs-mul rounding / signed
        // zero). This pins the duplicated affine reductions to the plain twins: if either drifts, the
        // mean/inv diverge and this fails.
        let n = 200usize;
        let x = fill(n);
        let ones = vec![1.0f32; n];
        let zeros = vec![0.0f32; n];
        for &op in &[NORM_LAYERNORM, NORM_RMSNORM] {
            let mut plain = vec![0.0f32; n];
            let mut aff = vec![0.0f32; n];
            // SAFETY: `plain`/`aff` sized n (plain vs affine).
            unsafe {
                wukong_norm_f32(
                    x.as_ptr(),
                    plain.as_mut_ptr(),
                    1,
                    n as i64,
                    EPS.to_bits() as i64,
                    op,
                );
                wukong_norm_affine_f32(
                    x.as_ptr(),
                    aff.as_mut_ptr(),
                    ones.as_ptr(),
                    zeros.as_ptr(),
                    1,
                    n as i64,
                    EPS.to_bits() as i64,
                    op,
                );
            }
            for i in 0..n {
                assert!(
                    (plain[i] - aff[i]).abs() <= 1e-6 * plain[i].abs().max(1.0),
                    "affine(γ1,β0) != plain op={op} i={i}: {} vs {}",
                    aff[i],
                    plain[i]
                );
            }
        }
    }
}
