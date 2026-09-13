//! Fused scaled-dot-product attention — the flash-attention kernel.
//!
//! Computes, for one head, `O[S,D] = softmax(scale · Q·Kᵀ [+ causal mask]) · V` with `Q,K,V,O`
//! row-major `[S,D]` f32. The softmax runs **online** (a running max `m`, running denominator `l`,
//! and a rescaled `O(D)` accumulator — the flash-attention recurrence), so the `S×S` score matrix is
//! **never materialized**: only a `BR×BC` score tile and the `BR×D` accumulator are live.
//!
//! One C entry, [`wukong_attention_f32`], and — unlike almost every other family in this crate —
//! **no `_parallel` twin**: multi-head work is spread by the caller (an outlined `@parallel` head
//! loop), not inside the kernel. A non-positive `s` or `d` is a no-op.
//!
//! The native backend calls this directly; the interpreter marshals its abstract memory into real
//! buffers and calls the *same* function, so the differential oracle stays bit-for-bit exact (the
//! online recurrence reassociates the softmax, so — like the GEMM kernel and the reassociated
//! reductions — the shared kernel *is* the oracle: both backends run it and must agree). Note the
//! twin relation *inside* the kernel is weaker than elsewhere: [`attention_avx2`] reassociates the
//! `q_i·k_j` dot across lanes and sums a whole key block before rescaling, where
//! [`attention_scalar`] sums sequentially, so those two agree only to tolerance — the naive
//! materialized reference in the tests is what gates both.
//!
//! # Why it is shaped this way (2026-09, workstream P5)
//!
//! The previous kernel was a per-`(query, key)` nest and lost to Wukong's own materialized
//! GEMM+softmax+GEMM path at every tested shape. It had three countable defects, all fixed here:
//!
//! 1. **Two platform `expf` calls per `(query, key)`** — `2·S²` scalar libm calls, where every
//!    other softmax in this crate uses the crate's own 8-lane [`crate::vmath::exp8`]. At `S=1024`
//!    that is 2.1M scalar transcendentals, tens of cycles each; it dominated everything else in the
//!    kernel by roughly an order of magnitude. Now: `S²` *vectorized* exps (~2 SIMD ops/lane) plus
//!    one scalar [`crate::vmath::exp1`] per `(query, key-block)` for the rescale factor, and that
//!    one is skipped outright when the running max did not move.
//! 2. **One horizontal reduce per `(query, key)`** — a `hsum256` (~9 shuffle/add uops on a serial
//!    dependency chain) for every single score, over a 1-FMA-per-2-loads inner dot with a single
//!    accumulator chain that could not cover FMA latency. Now: a 2-query × 4-key register-blocked
//!    microkernel — 8 independent accumulator chains (what it takes to saturate both FMA ports at
//!    4-cycle latency), 6 loads per 8 FMAs, and one 4-way [`red4`] tree per *four* scores.
//! 3. **A full `D`-wide accumulator rescale on every key** — `acc = acc·corr + p·v_j` round-tripped
//!    `D` floats through L1 for each of the `S²` steps: at `S=1024, D=64` that is 8.4M 32-byte
//!    stores, on a port that retires one per cycle. Now the rescale is folded into the accumulator
//!    *load* once per key block, and the block's `P·V` runs as an 8-row × 8-lane microkernel with
//!    the accumulator held in registers across all `BC` keys — `BC×` fewer accumulator round trips,
//!    and the `S²·D` extra multiplies of the per-key rescale disappear with them.
//!
//! **What fusion can and cannot buy on this CPU.** On a GPU, flash-attention wins because the
//! `S×S` intermediate spills out of a ~100 KB SRAM into HBM at an arithmetic intensity of ~200
//! FLOP/byte. Here the ratio is ~1.6 FLOP/byte and the L3 is 24 MB, so at `D=64` the materialized
//! path's extra traffic (write scores, read+write them in the softmax pass, re-read and pack them
//! for the second GEMM ≈ `6·S²·4` bytes) against its `4·S²·D` FLOPs is only ~0.094 bytes/FLOP —
//! about 7% of the run time while the intermediate still fits in L3, ~11% once it does not.
//! **Non-causal fusion is therefore a single-digit-percent effect here, not a GPU-style multiple**,
//! and any larger claim is measuring a weak control. The structural win is **causal**: the fused
//! form skips whole key blocks above the diagonal, where a dense two-GEMM control computes and
//! softmaxes the full `S²` and throws half of it away. `attention_throughput` below measures
//! against both a scalar-exp control and a same-exp, triangular-blocked strong control so the
//! difference between those two statements stays visible.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// `O = softmax(scale · Q·Kᵀ [+ causal mask]) · V` for one head. Row-major `[S,D]`.
///
/// `causal != 0` masks key `j > i` (autoregressive / decoder attention): query `i` attends only to
/// keys `0..=i`. `O` is fully overwritten.
///
/// # Safety
/// `q`, `k`, `v` must be valid for `s*d` `f32` reads each; `o` valid for `s*d` `f32` writes.
#[no_mangle]
pub unsafe extern "C" fn wukong_attention_f32(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    o: *mut f32,
    s: i64,
    d: i64,
    scale: f32,
    causal: i64,
) {
    if s <= 0 || d <= 0 {
        return;
    }
    let (s, d) = (s as usize, d as usize);
    let causal = causal != 0;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features just checked; dims validated by the caller contract.
            unsafe {
                attention_avx2(q, k, v, o, s, d, scale, causal);
            }
            return;
        }
    }
    // SAFETY: dims validated above; the operand-size contract is this entry point's caller's.
    unsafe {
        attention_scalar(q, k, v, o, s, d, scale, causal);
    }
}

/// `exp` through the crate's own kernel — the same one `norm.rs`'s fused softmax and a dispatched
/// `exp` loop use — but **NaN-transparent**.
///
/// [`crate::vmath::exp1`] opens with `x.min(EXP_HI).max(EXP_LO)`, and IEEE `minNum`/`maxNum` return
/// *the other operand* for a NaN, so a NaN is silently replaced by the clamp bound and `exp(NaN)`
/// comes back as roughly 2.4e38 (the crate-wide behaviour recorded as W16, unfixed, and
/// deliberately not touched from inside this file). Attention has always propagated NaN — the old
/// kernel called the platform `expf`, which does — and a diverged attention head that quietly
/// returns confident finite numbers is exactly the failure this crate's NaN discipline exists to
/// prevent. So the NaN is restored here, at the one call site, rather than by changing the shared
/// transcendental. [`exp_row`] does the same thing with a `vcmpps`/`vblendvps` pair, lane-for-lane.
#[inline]
fn exp_keep_nan(x: f32) -> f32 {
    if x.is_nan() {
        x
    } else {
        crate::vmath::exp1(x)
    }
}

/// Portable reference: the same online-softmax recurrence, scalar, one key at a time. Used where
/// AVX2 is unavailable; the AVX2 path computes the same quantity but tiled and reassociated (it
/// reduces the `q_i·k_j` dot across lanes and defers the accumulator rescale to a key-block
/// boundary), so the two agree to tolerance, not bit-for-bit. Both are gated against the naive
/// materialized reference, which is the actual oracle.
///
/// # Safety
/// Operand-size contract of [`wukong_attention_f32`]: `q`, `k`, `v` must each be valid for `s*d`
/// `f32` reads and `o` valid for `s*d` `f32` writes, with `o` not overlapping `q`/`k`/`v`.
#[allow(clippy::needless_range_loop)]
unsafe fn attention_scalar(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    o: *mut f32,
    s: usize,
    d: usize,
    scale: f32,
    causal: bool,
) {
    // SAFETY: every pointer read/write below stays inside `s*d` elements of the respective operand,
    // which the caller guarantees above.
    unsafe {
        let mut acc = vec![0.0f32; d];
        for i in 0..s {
            let qi = q.add(i * d);
            let jmax = if causal { i + 1 } else { s };
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            acc.iter_mut().for_each(|a| *a = 0.0);
            for j in 0..jmax {
                let kj = k.add(j * d);
                let mut x = 0.0f32;
                for t in 0..d {
                    x += *qi.add(t) * *kj.add(t);
                }
                x *= scale;
                // Online softmax: fold key j in, rescaling the running stats to the new max.
                let m_new = m.max(x);
                // `corr` is exactly 1 whenever the max did not move — the common case — so test for
                // that instead of paying an `exp`. The equality also keeps the `m = m_new = -inf`
                // corner (a fully masked prefix) out of `exp(NaN)`; on key 0 of a live row
                // `m = -inf < m_new` and `exp(-inf) = +0.0` zeroes the accumulator as before.
                let corr = if m == m_new {
                    1.0
                } else {
                    exp_keep_nan(m - m_new)
                };
                let p = exp_keep_nan(x - m_new);
                l = l * corr + p;
                let vj = v.add(j * d);
                for t in 0..d {
                    acc[t] = acc[t] * corr + p * *vj.add(t);
                }
                m = m_new;
            }
            let inv = if l != 0.0 { 1.0 / l } else { 0.0 };
            let oi = o.add(i * d);
            for t in 0..d {
                *oi.add(t) = acc[t] * inv;
            }
        }
    }
}

/// Query rows per tile. The block's running stats and its `BR×D` accumulator stay L1-resident and
/// each key loaded in the inner loop is reused across all `BR` queries.
const ATT_BR: usize = 32;
/// Keys per tile. 64 is the block width the CPU/vector flash-attention literature converges on
/// (arXiv 2608.18656 tunes to 64×64 against an L1/L2 pair the size of this host's): the score tile
/// is `BR·BC·4` = 8 KB and the K/V tiles `BC·D·4` = 16 KB each at `D=64`, so the working set of
/// either inner microkernel sits inside a 48 KB L1D. It is also the rescale period — the
/// accumulator is corrected once per `BC` keys instead of once per key.
const ATT_BC: usize = 64;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// Horizontal max of one vector, mirroring `_mm256_max_ps`'s `(a > b) ? a : b` tie/NaN rule so the
/// vector body and any scalar tail fold identically.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hmax256(v: std::arch::x86_64::__m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let a = _mm_max_ps(lo, hi);
    let b = _mm_max_ps(a, _mm_movehl_ps(a, a));
    let c = _mm_max_ss(b, _mm_shuffle_ps(b, b, 0x55));
    _mm_cvtss_f32(c)
}

/// Reduce **four** accumulators to their four scalar sums in one 5-op tree (3 `vhaddps`, an
/// `extractf128`, an `addps`), landing them contiguously so the caller stores them with a single
/// `movups`. That is under 2 uops per dot against `hsum256`'s ~9 — the single biggest reason the
/// score tile is computed four keys at a time rather than one.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn red4(
    a0: std::arch::x86_64::__m256,
    a1: std::arch::x86_64::__m256,
    a2: std::arch::x86_64::__m256,
    a3: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m128 {
    // hadd(a,b) = [a01, a23, b01, b23 | a45, a67, b45, b67], so two levels leave each input's
    // half-sums in one lane pair and the final 128-bit add closes them.
    let t0 = _mm256_hadd_ps(a0, a1);
    let t1 = _mm256_hadd_ps(a2, a3);
    let t2 = _mm256_hadd_ps(t0, t1);
    _mm_add_ps(_mm256_castps256_ps128(t2), _mm256_extractf128_ps(t2, 1))
}

/// One `D`-long dot, 8-wide with a scalar remainder. Only the `kb % 4` and `qb % 2` tile edges use
/// it; the body runs [`dot2x4`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot1(a: *const f32, b: *const f32, d: usize, dv: usize) -> f32 {
    let mut sv = _mm256_setzero_ps();
    let mut t = 0;
    while t < dv {
        sv = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(t)), _mm256_loadu_ps(b.add(t)), sv);
        t += 8;
    }
    let mut x = hsum256(sv);
    while t < d {
        x += *a.add(t) * *b.add(t);
        t += 1;
    }
    x
}

/// One query against four keys: 4 accumulators, 5 loads per 4 FMAs, one [`red4`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot1x4(
    a: *const f32,
    k0: *const f32,
    k1: *const f32,
    k2: *const f32,
    k3: *const f32,
    d: usize,
    dv: usize,
) -> std::arch::x86_64::__m128 {
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let mut t = 0;
    while t < dv {
        let av = _mm256_loadu_ps(a.add(t));
        a0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(k0.add(t)), a0);
        a1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(k1.add(t)), a1);
        a2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(k2.add(t)), a2);
        a3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(k3.add(t)), a3);
        t += 8;
    }
    let mut r = red4(a0, a1, a2, a3);
    if t < d {
        let mut s = [0.0f32; 4];
        while t < d {
            let x = *a.add(t);
            s[0] += x * *k0.add(t);
            s[1] += x * *k1.add(t);
            s[2] += x * *k2.add(t);
            s[3] += x * *k3.add(t);
            t += 1;
        }
        r = _mm_add_ps(r, _mm_loadu_ps(s.as_ptr()));
    }
    r
}

/// The score microkernel: **two queries x four keys**, 8 independent accumulator chains.
///
/// Eight chains is the register-blocking floor for this machine, not a round number — an AVX2 FMA
/// has 4-cycle latency on two ports, so fewer than 8 in flight cannot retire 2/cycle. Per 8-lane
/// step it issues 6 loads (2 query, 4 key) for 8 FMAs, against the old per-pair dot's 2 loads per
/// 1 FMA. This is the standard dot-product-form register tile — the BLIS microkernel shape, the
/// same one `gemm.rs` uses for the NT case.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot2x4(
    qa: *const f32,
    qc: *const f32,
    k0: *const f32,
    k1: *const f32,
    k2: *const f32,
    k3: *const f32,
    d: usize,
    dv: usize,
) -> (std::arch::x86_64::__m128, std::arch::x86_64::__m128) {
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let mut b0 = _mm256_setzero_ps();
    let mut b1 = _mm256_setzero_ps();
    let mut b2 = _mm256_setzero_ps();
    let mut b3 = _mm256_setzero_ps();
    let mut t = 0;
    while t < dv {
        let av = _mm256_loadu_ps(qa.add(t));
        let bv = _mm256_loadu_ps(qc.add(t));
        let kv0 = _mm256_loadu_ps(k0.add(t));
        let kv1 = _mm256_loadu_ps(k1.add(t));
        let kv2 = _mm256_loadu_ps(k2.add(t));
        let kv3 = _mm256_loadu_ps(k3.add(t));
        a0 = _mm256_fmadd_ps(av, kv0, a0);
        a1 = _mm256_fmadd_ps(av, kv1, a1);
        a2 = _mm256_fmadd_ps(av, kv2, a2);
        a3 = _mm256_fmadd_ps(av, kv3, a3);
        b0 = _mm256_fmadd_ps(bv, kv0, b0);
        b1 = _mm256_fmadd_ps(bv, kv1, b1);
        b2 = _mm256_fmadd_ps(bv, kv2, b2);
        b3 = _mm256_fmadd_ps(bv, kv3, b3);
        t += 8;
    }
    let mut ra = red4(a0, a1, a2, a3);
    let mut rb = red4(b0, b1, b2, b3);
    if t < d {
        let mut sa = [0.0f32; 4];
        let mut sb = [0.0f32; 4];
        while t < d {
            let xa = *qa.add(t);
            let xb = *qc.add(t);
            let y0 = *k0.add(t);
            let y1 = *k1.add(t);
            let y2 = *k2.add(t);
            let y3 = *k3.add(t);
            sa[0] += xa * y0;
            sa[1] += xa * y1;
            sa[2] += xa * y2;
            sa[3] += xa * y3;
            sb[0] += xb * y0;
            sb[1] += xb * y1;
            sb[2] += xb * y2;
            sb[3] += xb * y3;
            t += 1;
        }
        ra = _mm_add_ps(ra, _mm_loadu_ps(sa.as_ptr()));
        rb = _mm_add_ps(rb, _mm_loadu_ps(sb.as_ptr()));
    }
    (ra, rb)
}

/// Fill one `qb x kb` score tile with `scale * q_i.k_j`, rows strided by [`ATT_BC`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn qk_tile(
    q: *const f32,
    k: *const f32,
    sc: *mut f32,
    i0: usize,
    j0: usize,
    qb: usize,
    kb: usize,
    d: usize,
    dv: usize,
    scale: f32,
) {
    let sv = _mm_set1_ps(scale);
    let kb4 = kb & !3;
    let mut ii = 0;
    while ii + 2 <= qb {
        let qa = q.add((i0 + ii) * d);
        let qc = q.add((i0 + ii + 1) * d);
        let ra = sc.add(ii * ATT_BC);
        let rc = sc.add((ii + 1) * ATT_BC);
        let mut jj = 0;
        while jj < kb4 {
            let kp = k.add((j0 + jj) * d);
            let (va, vb) = dot2x4(qa, qc, kp, kp.add(d), kp.add(2 * d), kp.add(3 * d), d, dv);
            _mm_storeu_ps(ra.add(jj), _mm_mul_ps(va, sv));
            _mm_storeu_ps(rc.add(jj), _mm_mul_ps(vb, sv));
            jj += 4;
        }
        while jj < kb {
            let kp = k.add((j0 + jj) * d);
            *ra.add(jj) = dot1(qa, kp, d, dv) * scale;
            *rc.add(jj) = dot1(qc, kp, d, dv) * scale;
            jj += 1;
        }
        ii += 2;
    }
    if ii < qb {
        let qa = q.add((i0 + ii) * d);
        let ra = sc.add(ii * ATT_BC);
        let mut jj = 0;
        while jj < kb4 {
            let kp = k.add((j0 + jj) * d);
            let va = dot1x4(qa, kp, kp.add(d), kp.add(2 * d), kp.add(3 * d), d, dv);
            _mm_storeu_ps(ra.add(jj), _mm_mul_ps(va, sv));
            jj += 4;
        }
        while jj < kb {
            *ra.add(jj) = dot1(qa, k.add((j0 + jj) * d), d, dv) * scale;
            jj += 1;
        }
    }
}

/// Running max of `row[..n]` folded with `init`, vector body plus scalar tail on the same tie rule.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_max(row: *const f32, n: usize, init: f32) -> f32 {
    let nv = n & !7;
    let mut best = init;
    let mut t = 0;
    if nv != 0 {
        let mut mv = _mm256_set1_ps(init);
        while t < nv {
            mv = _mm256_max_ps(mv, _mm256_loadu_ps(row.add(t)));
            t += 8;
        }
        best = hmax256(mv);
    }
    while t < n {
        let x = *row.add(t);
        // `(x > best) ? x : best` — `_mm256_max_ps`'s rule, not `f32::max`'s.
        if x > best {
            best = x;
        }
        t += 1;
    }
    best
}

/// In place: `row[j] := exp(row[j] - mn)` for `j < n`, returning the sum of the results.
///
/// This is the fix for the kernel's dominant cost. It is the crate's own 8-lane exp — the very same
/// [`crate::vmath::exp8`] `norm.rs`'s fused softmax and a dispatched `exp` loop call — at ~16 SIMD
/// ops per 8 lanes, replacing two platform `expf` calls per `(query, key)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_row(row: *mut f32, n: usize, mn: f32) -> f32 {
    let nv = n & !7;
    let mnv = _mm256_set1_ps(mn);
    let mut sumv = _mm256_setzero_ps();
    let mut t = 0;
    while t < nv {
        let a = _mm256_sub_ps(_mm256_loadu_ps(row.add(t)), mnv);
        // NaN transparency, lane-for-lane with `exp_keep_nan`: `exp8` clamps a NaN to `EXP_HI`
        // (crate-wide, W16), so select the original NaN back — payload and sign intact — instead of
        // letting a diverged head return a confident finite number.
        let p = _mm256_blendv_ps(
            crate::vmath::exp8(a),
            a,
            _mm256_cmp_ps::<_CMP_UNORD_Q>(a, a),
        );
        _mm256_storeu_ps(row.add(t), p);
        sumv = _mm256_add_ps(sumv, p);
        t += 8;
    }
    let mut rs = if nv != 0 { hsum256(sumv) } else { 0.0 };
    while t < n {
        let p = exp_keep_nan(*row.add(t) - mn);
        *row.add(t) = p;
        rs += p;
        t += 1;
    }
    rs
}

/// `acc[r] := acc[r]*corr[r] + sum_j P[r][j]*V[j]` for **eight** query rows at once, 8 lanes of `D`
/// at a time.
///
/// The accumulator is loaded once, rescaled while it is already in the register, driven through the
/// whole key block, and stored once — so the per-key `D`-wide load/multiply/FMA/store round trip
/// that dominated the old kernel's store port becomes one round trip per `BC` keys. Eight rows (not
/// four) because eight independent chains are what it takes to keep both FMA ports busy at 4-cycle
/// latency; the cost is 8 `vbroadcastss` per shared `V` load, which the three load ports absorb.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::needless_range_loop)]
unsafe fn pv8(
    v: *const f32,
    p: *const f32,
    acc: *mut f32,
    c: *const f32,
    j0: usize,
    kb: usize,
    d: usize,
    dv: usize,
) {
    let mut t = 0;
    while t < dv {
        let mut a0 = _mm256_mul_ps(_mm256_loadu_ps(acc.add(t)), _mm256_set1_ps(*c));
        let mut a1 = _mm256_mul_ps(_mm256_loadu_ps(acc.add(d + t)), _mm256_set1_ps(*c.add(1)));
        let mut a2 = _mm256_mul_ps(
            _mm256_loadu_ps(acc.add(2 * d + t)),
            _mm256_set1_ps(*c.add(2)),
        );
        let mut a3 = _mm256_mul_ps(
            _mm256_loadu_ps(acc.add(3 * d + t)),
            _mm256_set1_ps(*c.add(3)),
        );
        let mut a4 = _mm256_mul_ps(
            _mm256_loadu_ps(acc.add(4 * d + t)),
            _mm256_set1_ps(*c.add(4)),
        );
        let mut a5 = _mm256_mul_ps(
            _mm256_loadu_ps(acc.add(5 * d + t)),
            _mm256_set1_ps(*c.add(5)),
        );
        let mut a6 = _mm256_mul_ps(
            _mm256_loadu_ps(acc.add(6 * d + t)),
            _mm256_set1_ps(*c.add(6)),
        );
        let mut a7 = _mm256_mul_ps(
            _mm256_loadu_ps(acc.add(7 * d + t)),
            _mm256_set1_ps(*c.add(7)),
        );
        let mut jj = 0;
        while jj < kb {
            let vv = _mm256_loadu_ps(v.add((j0 + jj) * d + t));
            a0 = _mm256_fmadd_ps(_mm256_set1_ps(*p.add(jj)), vv, a0);
            a1 = _mm256_fmadd_ps(_mm256_set1_ps(*p.add(ATT_BC + jj)), vv, a1);
            a2 = _mm256_fmadd_ps(_mm256_set1_ps(*p.add(2 * ATT_BC + jj)), vv, a2);
            a3 = _mm256_fmadd_ps(_mm256_set1_ps(*p.add(3 * ATT_BC + jj)), vv, a3);
            a4 = _mm256_fmadd_ps(_mm256_set1_ps(*p.add(4 * ATT_BC + jj)), vv, a4);
            a5 = _mm256_fmadd_ps(_mm256_set1_ps(*p.add(5 * ATT_BC + jj)), vv, a5);
            a6 = _mm256_fmadd_ps(_mm256_set1_ps(*p.add(6 * ATT_BC + jj)), vv, a6);
            a7 = _mm256_fmadd_ps(_mm256_set1_ps(*p.add(7 * ATT_BC + jj)), vv, a7);
            jj += 1;
        }
        _mm256_storeu_ps(acc.add(t), a0);
        _mm256_storeu_ps(acc.add(d + t), a1);
        _mm256_storeu_ps(acc.add(2 * d + t), a2);
        _mm256_storeu_ps(acc.add(3 * d + t), a3);
        _mm256_storeu_ps(acc.add(4 * d + t), a4);
        _mm256_storeu_ps(acc.add(5 * d + t), a5);
        _mm256_storeu_ps(acc.add(6 * d + t), a6);
        _mm256_storeu_ps(acc.add(7 * d + t), a7);
        t += 8;
    }
    while t < d {
        let mut sums = [0.0f32; 8];
        for r in 0..8 {
            sums[r] = *acc.add(r * d + t) * *c.add(r);
        }
        let mut jj = 0;
        while jj < kb {
            let vx = *v.add((j0 + jj) * d + t);
            for r in 0..8 {
                sums[r] += *p.add(r * ATT_BC + jj) * vx;
            }
            jj += 1;
        }
        for r in 0..8 {
            *acc.add(r * d + t) = sums[r];
        }
        t += 1;
    }
}

/// The `qb % 8` edge of [`pv8`], one query row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pv1(
    v: *const f32,
    p: *const f32,
    acc: *mut f32,
    c: f32,
    j0: usize,
    kb: usize,
    d: usize,
    dv: usize,
) {
    let cv = _mm256_set1_ps(c);
    let mut t = 0;
    while t < dv {
        let mut a = _mm256_mul_ps(_mm256_loadu_ps(acc.add(t)), cv);
        let mut jj = 0;
        while jj < kb {
            a = _mm256_fmadd_ps(
                _mm256_set1_ps(*p.add(jj)),
                _mm256_loadu_ps(v.add((j0 + jj) * d + t)),
                a,
            );
            jj += 1;
        }
        _mm256_storeu_ps(acc.add(t), a);
        t += 8;
    }
    while t < d {
        let mut a = *acc.add(t) * c;
        let mut jj = 0;
        while jj < kb {
            a += *p.add(jj) * *v.add((j0 + jj) * d + t);
            jj += 1;
        }
        *acc.add(t) = a;
        t += 1;
    }
}

/// AVX2/FMA path: FlashAttention-2's two-level block loop (Dao 2023, Alg. 1) — queries outer, keys
/// inner, running `(m, l)` per query row, the un-normalized accumulator divided by `l` once at the
/// very end rather than per block, and whole key blocks above the diagonal skipped under a causal
/// mask. Inside a block it is three straight-line microkernels: [`qk_tile`], the row softmax
/// ([`row_max`] + [`exp_row`]), and [`pv8`].
///
/// # Safety
/// Operand-size contract of [`wukong_attention_f32`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attention_avx2(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    o: *mut f32,
    s: usize,
    d: usize,
    scale: f32,
    causal: bool,
) {
    let dv = d & !7; // largest multiple of 8 <= d
    let mut sc = vec![0.0f32; ATT_BR * ATT_BC];
    let mut acc = vec![0.0f32; ATT_BR * d];
    let mut m = [f32::NEG_INFINITY; ATT_BR];
    let mut l = [0.0f32; ATT_BR];
    let mut corr = [0.0f32; ATT_BR];

    let mut i0 = 0;
    while i0 < s {
        let qb = (s - i0).min(ATT_BR);
        m[..qb].iter_mut().for_each(|x| *x = f32::NEG_INFINITY);
        l[..qb].iter_mut().for_each(|x| *x = 0.0);
        acc[..qb * d].iter_mut().for_each(|x| *x = 0.0);
        // The largest key any query in this block attends to is `i0 + qb - 1` under a causal mask,
        // so every key block past that is skipped whole — the structural half of the causal win.
        let jhi = if causal { i0 + qb } else { s };
        let mut j0 = 0;
        while j0 < jhi {
            let kb = (jhi - j0).min(ATT_BC);
            qk_tile(q, k, sc.as_mut_ptr(), i0, j0, qb, kb, d, dv, scale);
            for ii in 0..qb {
                let row = sc.as_mut_ptr().add(ii * ATT_BC);
                // Per-row causal limit inside the tile. Everything past it is zeroed rather than
                // masked to -inf, so `pv8` below can run the block densely and `exp` never sees an
                // infinity it would have to special-case.
                let ke = if causal {
                    (i0 + ii + 1).saturating_sub(j0).min(kb)
                } else {
                    kb
                };
                let mut z = ke;
                while z < kb {
                    *row.add(z) = 0.0;
                    z += 1;
                }
                if ke == 0 {
                    // Whole row above the diagonal in this block: stats and accumulator unchanged.
                    corr[ii] = 1.0;
                    continue;
                }
                let mn = row_max(row, ke, m[ii]);
                corr[ii] = if m[ii] == mn {
                    1.0
                } else {
                    exp_keep_nan(m[ii] - mn)
                };
                l[ii] = l[ii] * corr[ii] + exp_row(row, ke, mn);
                m[ii] = mn;
            }
            // One accumulator rescale and one P*V pass for the whole key block.
            let mut ii = 0;
            while ii + 8 <= qb {
                pv8(
                    v,
                    sc.as_ptr().add(ii * ATT_BC),
                    acc.as_mut_ptr().add(ii * d),
                    corr.as_ptr().add(ii),
                    j0,
                    kb,
                    d,
                    dv,
                );
                ii += 8;
            }
            while ii < qb {
                pv1(
                    v,
                    sc.as_ptr().add(ii * ATT_BC),
                    acc.as_mut_ptr().add(ii * d),
                    corr[ii],
                    j0,
                    kb,
                    d,
                    dv,
                );
                ii += 1;
            }
            j0 += ATT_BC;
        }
        // O_i = acc_i / l_i for each query in the block — the single normalization FA-2 defers to
        // here instead of renormalizing the accumulator at every block boundary.
        for (ii, &lv) in l.iter().take(qb).enumerate() {
            let inv = if lv != 0.0 { 1.0 / lv } else { 0.0 };
            let oi = o.add((i0 + ii) * d);
            let ap = acc.as_ptr().add(ii * d);
            let invv = _mm256_set1_ps(inv);
            let mut t = 0;
            while t < dv {
                _mm256_storeu_ps(oi.add(t), _mm256_mul_ps(_mm256_loadu_ps(ap.add(t)), invv));
                t += 8;
            }
            while t < d {
                *oi.add(t) = *ap.add(t) * inv;
                t += 1;
            }
        }
        i0 += ATT_BR;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shapes every correctness test walks. They straddle the 8-wide `D` remainder, the `qb % 2`
    /// and `kb % 4` score-tile edges, the `qb % 8` P*V edge, and — the ones the pre-P5 list could
    /// not reach, because the old kernel had no key blocking — the `ATT_BR = 32` query-block and
    /// `ATT_BC = 64` key-block boundaries in both directions.
    const SHAPES: &[(usize, usize)] = &[
        (1, 1),
        (2, 2),
        (4, 8),
        (5, 7),
        (8, 16),
        (16, 8),
        (33, 17),
        (64, 64),
        (32, 64),
        (33, 64),
        (63, 8),
        (65, 16),
        (96, 33),
        (129, 8),
        (130, 24),
        (200, 17),
    ];

    /// Naive reference: materialize the score row, softmax it (max-subtract), then weight V — the
    /// textbook attention, independent of the online recurrence under test.
    fn naive(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        s: usize,
        d: usize,
        scale: f32,
        causal: bool,
    ) -> Vec<f32> {
        let mut o = vec![0.0f32; s * d];
        for i in 0..s {
            let jmax = if causal { i + 1 } else { s };
            let mut sc = vec![0.0f32; jmax];
            let mut mx = f32::NEG_INFINITY;
            for (j, scj) in sc.iter_mut().enumerate() {
                let mut x = 0.0f32;
                for t in 0..d {
                    x += q[i * d + t] * k[j * d + t];
                }
                *scj = x * scale;
                mx = mx.max(*scj);
            }
            let mut sum = 0.0f32;
            for scj in sc.iter_mut() {
                *scj = (*scj - mx).exp();
                sum += *scj;
            }
            for (j, &scj) in sc.iter().enumerate() {
                let p = scj / sum;
                for t in 0..d {
                    o[i * d + t] += p * v[j * d + t];
                }
            }
        }
        o
    }

    fn fill(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    fn check(s: usize, d: usize, causal: bool) {
        let q = fill(1, s * d);
        let k = fill(2, s * d);
        let v = fill(3, s * d);
        let scale = 1.0 / (d as f32).sqrt();
        let want = naive(&q, &k, &v, s, d, scale, causal);
        let mut got = vec![0.0f32; s * d];
        unsafe {
            wukong_attention_f32(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                got.as_mut_ptr(),
                s as i64,
                d as i64,
                scale,
                causal as i64,
            );
        }
        for i in 0..s * d {
            assert!(
                (got[i] - want[i]).abs() <= 1e-4 + 1e-4 * want[i].abs(),
                "({s}x{d}, causal={causal}) idx {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn attention_matches_naive() {
        for &(s, d) in SHAPES {
            check(s, d, false);
            check(s, d, true);
        }
    }

    /// The scalar twin is the only path on a machine without AVX2/FMA and on every non-x86_64 build,
    /// but `wukong_attention_f32`'s runtime dispatch sends every CI and developer machine down the
    /// AVX2 path — so `attention_matches_naive` above never executes `attention_scalar` and a
    /// regression in it (a dropped `corr` rescale, a lost `l == 0` guard) would ship green. Drive it
    /// directly over the same shapes.
    #[test]
    fn attention_scalar_matches_naive() {
        for &(s, d) in SHAPES {
            for &causal in &[false, true] {
                let q = fill(1, s * d);
                let k = fill(2, s * d);
                let v = fill(3, s * d);
                let scale = 1.0 / (d as f32).sqrt();
                let want = naive(&q, &k, &v, s, d, scale, causal);
                let mut got = vec![0.0f32; s * d];
                // SAFETY: q/k/v are s*d long and readable, got is s*d long and writable — the
                // operand contract of `wukong_attention_f32` restated on `attention_scalar`.
                unsafe {
                    attention_scalar(
                        q.as_ptr(),
                        k.as_ptr(),
                        v.as_ptr(),
                        got.as_mut_ptr(),
                        s,
                        d,
                        scale,
                        causal,
                    );
                }
                for i in 0..s * d {
                    assert!(
                        (got[i] - want[i]).abs() <= 1e-4 + 1e-4 * want[i].abs(),
                        "scalar ({s}x{d}, causal={causal}) idx {i}: got {} want {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }

    /// A NaN in an operand must reach the output. The old kernel got this for free from the
    /// platform `expf`; the crate's own `exp` clamps a NaN to a finite `EXP_HI` (W16), so both the
    /// vector body and the scalar tail restore it explicitly and this pins that they do.
    ///
    /// Both probes are chosen so the answer does not depend on which of `f32::max` /
    /// `_mm256_max_ps` the running max uses on a NaN:
    /// * a NaN at `V[0][0]` reaches column 0 of **every** output row, because key 0 is in range for
    ///   every query under either mask;
    /// * a NaN at `Q[1][0]` makes **every** score of row 1 a NaN, so the whole row is NaN, while
    ///   row 0 — which never reads `Q[1]` — must stay finite. That second half is also a
    ///   cross-row-contamination check on the 2x4 and 8-row register tiles.
    #[test]
    fn attention_propagates_nan() {
        for &(s, d) in &[(4usize, 8usize), (5, 7), (40, 64), (70, 16)] {
            for &causal in &[false, true] {
                let scale = 1.0 / (d as f32).sqrt();
                let q = fill(1, s * d);
                let k = fill(2, s * d);

                let mut vn = fill(3, s * d);
                vn[0] = f32::NAN;
                let mut got = vec![0.0f32; s * d];
                unsafe {
                    wukong_attention_f32(
                        q.as_ptr(),
                        k.as_ptr(),
                        vn.as_ptr(),
                        got.as_mut_ptr(),
                        s as i64,
                        d as i64,
                        scale,
                        causal as i64,
                    );
                }
                for i in 0..s {
                    assert!(
                        got[i * d].is_nan(),
                        "V NaN lost ({s}x{d}, causal={causal}) at row {i}: {}",
                        got[i * d]
                    );
                }

                let mut qn = q.clone();
                qn[d] = f32::NAN; // Q row 1, lane 0
                let v = fill(3, s * d);
                let mut got = vec![0.0f32; s * d];
                unsafe {
                    wukong_attention_f32(
                        qn.as_ptr(),
                        k.as_ptr(),
                        v.as_ptr(),
                        got.as_mut_ptr(),
                        s as i64,
                        d as i64,
                        scale,
                        causal as i64,
                    );
                }
                for t in 0..d {
                    assert!(
                        got[d + t].is_nan(),
                        "Q NaN lost ({s}x{d}, causal={causal}) at row 1 lane {t}: {}",
                        got[d + t]
                    );
                    assert!(
                        got[t].is_finite(),
                        "row 0 contaminated ({s}x{d}, causal={causal}) at lane {t}: {}",
                        got[t]
                    );
                }

                // Same probe through the portable twin.
                let mut got = vec![0.0f32; s * d];
                // SAFETY: operand contract of `wukong_attention_f32`.
                unsafe {
                    attention_scalar(
                        qn.as_ptr(),
                        k.as_ptr(),
                        v.as_ptr(),
                        got.as_mut_ptr(),
                        s,
                        d,
                        scale,
                        causal,
                    );
                }
                for t in 0..d {
                    assert!(
                        got[d + t].is_nan(),
                        "scalar Q NaN lost ({s}x{d}, causal={causal}) lane {t}: {}",
                        got[d + t]
                    );
                    assert!(
                        got[t].is_finite(),
                        "scalar row 0 contaminated ({s}x{d}, causal={causal}) lane {t}: {}",
                        got[t]
                    );
                }
            }
        }
    }

    /// The pre-P5 AVX2 kernel, verbatim, kept as a second independent implementation: it is the
    /// A/B arm in `attention_throughput` and the peer in `attention_matches_legacy_kernel`. Its
    /// defects are performance-only — two scalar `expf` per (query, key), one `hsum256` per score,
    /// a `D`-wide accumulator round trip per key — so it is a legitimate correctness oracle even
    /// though it is the thing being replaced.
    ///
    /// # Safety
    /// Operand-size contract of [`wukong_attention_f32`].
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn attention_avx2_legacy(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        o: *mut f32,
        s: usize,
        d: usize,
        scale: f32,
        causal: bool,
    ) {
        const ATT_QB: usize = 32;
        let dv = d & !7;
        let mut m = [f32::NEG_INFINITY; ATT_QB];
        let mut l = [0.0f32; ATT_QB];
        let mut acc = vec![0.0f32; ATT_QB * d];
        let mut i0 = 0;
        while i0 < s {
            let qb = (s - i0).min(ATT_QB);
            m[..qb].iter_mut().for_each(|x| *x = f32::NEG_INFINITY);
            l[..qb].iter_mut().for_each(|x| *x = 0.0);
            acc[..qb * d].iter_mut().for_each(|x| *x = 0.0);
            let jmax = if causal { i0 + qb } else { s };
            for j in 0..jmax {
                let kj = k.add(j * d);
                let vj = v.add(j * d);
                for ii in 0..qb {
                    let i = i0 + ii;
                    if causal && j > i {
                        continue;
                    }
                    let qi = q.add(i * d);
                    let mut sv = _mm256_setzero_ps();
                    let mut t = 0;
                    while t < dv {
                        sv = _mm256_fmadd_ps(
                            _mm256_loadu_ps(qi.add(t)),
                            _mm256_loadu_ps(kj.add(t)),
                            sv,
                        );
                        t += 8;
                    }
                    let mut x = hsum256(sv);
                    while t < d {
                        x += *qi.add(t) * *kj.add(t);
                        t += 1;
                    }
                    x *= scale;
                    let m_new = m[ii].max(x);
                    let corr = (m[ii] - m_new).exp();
                    let p = (x - m_new).exp();
                    l[ii] = l[ii] * corr + p;
                    let accp = acc.as_mut_ptr().add(ii * d);
                    let corrv = _mm256_set1_ps(corr);
                    let pv = _mm256_set1_ps(p);
                    let mut t = 0;
                    while t < dv {
                        let a = _mm256_mul_ps(_mm256_loadu_ps(accp.add(t)), corrv);
                        _mm256_storeu_ps(
                            accp.add(t),
                            _mm256_fmadd_ps(pv, _mm256_loadu_ps(vj.add(t)), a),
                        );
                        t += 8;
                    }
                    while t < d {
                        *accp.add(t) = *accp.add(t) * corr + p * *vj.add(t);
                        t += 1;
                    }
                    m[ii] = m_new;
                }
            }
            for (ii, &lv) in l.iter().take(qb).enumerate() {
                let inv = if lv != 0.0 { 1.0 / lv } else { 0.0 };
                let oi = o.add((i0 + ii) * d);
                let accp = acc.as_ptr().add(ii * d);
                let invv = _mm256_set1_ps(inv);
                let mut t = 0;
                while t < dv {
                    _mm256_storeu_ps(oi.add(t), _mm256_mul_ps(_mm256_loadu_ps(accp.add(t)), invv));
                    t += 8;
                }
                while t < d {
                    *oi.add(t) = *accp.add(t) * inv;
                    t += 1;
                }
            }
            i0 += ATT_QB;
        }
    }

    /// The tiled kernel must agree with the kernel it replaces. Both are already gated against
    /// `naive`, but that gate is per-path; this one is direct, and it is the check that would catch
    /// a tile-edge (`qb % 2`, `kb % 4`, `qb % 8`, `d % 8`) or causal-limit bug that happened to
    /// stay inside the naive tolerance on one path only. Same `1e-4` band the naive gate uses —
    /// both paths reassociate, so an exact match is not available to either.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn attention_matches_legacy_kernel() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for &(s, d) in SHAPES {
            for &causal in &[false, true] {
                let q = fill(1, s * d);
                let k = fill(2, s * d);
                let v = fill(3, s * d);
                let scale = 1.0 / (d as f32).sqrt();
                let mut want = vec![0.0f32; s * d];
                let mut got = vec![0.0f32; s * d];
                // SAFETY: avx2+fma checked above; operand contract of `wukong_attention_f32`.
                unsafe {
                    attention_avx2_legacy(
                        q.as_ptr(),
                        k.as_ptr(),
                        v.as_ptr(),
                        want.as_mut_ptr(),
                        s,
                        d,
                        scale,
                        causal,
                    );
                    wukong_attention_f32(
                        q.as_ptr(),
                        k.as_ptr(),
                        v.as_ptr(),
                        got.as_mut_ptr(),
                        s as i64,
                        d as i64,
                        scale,
                        causal as i64,
                    );
                }
                for i in 0..s * d {
                    assert!(
                        (got[i] - want[i]).abs() <= 1e-4 + 1e-4 * want[i].abs(),
                        "vs legacy ({s}x{d}, causal={causal}) idx {i}: got {} want {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }

    /// In-place row softmax through the crate's own `exp` — `exp8` where AVX2 is present, `exp1`
    /// otherwise. This is what makes the strong control in `attention_throughput` honest: it pays
    /// the same vectorized exps the fused kernel pays, so the two differ only in the
    /// materialization, not in their transcendental.
    fn softmax_row_fast(row: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: features just checked; all accesses are inside `row`.
                unsafe {
                    softmax_row_avx2(row);
                }
                return;
            }
        }
        softmax_row_scalar(row);
    }

    fn softmax_row_scalar(row: &mut [f32]) {
        let mut mx = f32::NEG_INFINITY;
        for &x in row.iter() {
            if x > mx {
                mx = x;
            }
        }
        let mut sum = 0.0f32;
        for x in row.iter_mut() {
            *x = exp_keep_nan(*x - mx);
            sum += *x;
        }
        let inv = 1.0 / sum;
        for x in row.iter_mut() {
            *x *= inv;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn softmax_row_avx2(row: &mut [f32]) {
        let n = row.len();
        let p = row.as_mut_ptr();
        let mx = row_max(p, n, f32::NEG_INFINITY);
        let sum = exp_row(p, n, mx);
        let inv = 1.0 / sum;
        let invv = _mm256_set1_ps(inv);
        let nv = n & !7;
        let mut t = 0;
        while t < nv {
            _mm256_storeu_ps(p.add(t), _mm256_mul_ps(_mm256_loadu_ps(p.add(t)), invv));
            t += 8;
        }
        while t < n {
            *p.add(t) *= inv;
            t += 1;
        }
    }

    /// Throughput probe (run: `cargo test -p wukong_runtime --release --lib attention_throughput --
    /// --ignored --nocapture --test-threads=1`). The filter and `--test-threads=1` are load-bearing:
    /// an unfiltered `--ignored` run starts all of the crate's throughput probes at once on
    /// libtest's default thread pool, so each one measures a machine saturated by the others.
    ///
    /// Four columns, because the answer depends entirely on which control you pick and an honest
    /// report needs all of them:
    ///
    /// * `mat-libm` — Wukong's materialized path with a scalar-`expf` row softmax. This is the
    ///   control the pre-P5 W15 row was measured against, kept so the old number stays comparable.
    /// * `mat-exp8` — the same two GEMMs with the crate's *vectorized* row softmax, and, under a
    ///   causal mask, a triangular panel loop that shrinks the NT GEMM's `N` and the second GEMM's
    ///   `K` to `i0 + panel` so it never computes the half it would throw away. **This is the
    ///   strong control**: same exp, same GEMM microkernels, same triangular saving, so the delta
    ///   against it is purely the materialization.
    /// * `legacy` — the pre-P5 fused kernel, for a same-process A/B of the rewrite itself.
    /// * `fused` — the kernel in this file.
    ///
    /// Every buffer, the `S x S` score matrix included, is allocated **outside** the timed region.
    /// The pre-P5 probe allocated and freed a `4·S²`-byte zeroed `Vec` inside each timed iteration,
    /// which charged the control 4096 demand-zero page faults at `S = 2048` that the fused path,
    /// whose whole scratch is 16 KB, never pays. That is an instrument artifact in the control's
    /// disfavour, and a fused-vs-materialized number is worthless if the control is handicapped.
    #[test]
    #[ignore]
    fn attention_throughput() {
        use crate::{wukong_sgemm, wukong_sgemm_nt};
        use std::time::Instant;

        /// Two full GEMMs plus a row softmax — the non-causal materialized control. Writes through
        /// `scores` (`s*s`) and `o` (`s*d`), both caller-owned so no allocation is timed.
        ///
        /// # Safety
        /// `scores` valid for `s*s` writes, `o` for `s*d` writes; `q`/`k`/`v` are `s*d` long.
        unsafe fn materialized(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            scores: *mut f32,
            o: *mut f32,
            s: usize,
            d: usize,
            scale: f32,
            libm_exp: bool,
        ) {
            wukong_sgemm_nt(
                q.as_ptr(),
                k.as_ptr(),
                scores,
                s as i64,
                d as i64,
                s as i64,
                0,
            );
            for i in 0..s {
                let row = std::slice::from_raw_parts_mut(scores.add(i * s), s);
                for r in row.iter_mut() {
                    *r *= scale;
                }
                if libm_exp {
                    let mut mx = f32::NEG_INFINITY;
                    for &r in row.iter() {
                        mx = mx.max(r);
                    }
                    let mut sum = 0.0f32;
                    for r in row.iter_mut() {
                        *r = (*r - mx).exp();
                        sum += *r;
                    }
                    let inv = 1.0 / sum;
                    for r in row.iter_mut() {
                        *r *= inv;
                    }
                } else {
                    softmax_row_fast(row);
                }
            }
            wukong_sgemm(scores, v.as_ptr(), o, s as i64, s as i64, d as i64, 0);
        }

        /// Causal materialized control, panelled so the GEMMs see only the triangle: for a panel of
        /// `PANEL` query rows starting at `i0`, no query attends past key `i0 + pb - 1`, so the NT
        /// GEMM's `N` and the second GEMM's `K` are `i0 + pb` rather than `S`. Without this the
        /// control does 2x the necessary flops and the comparison is vacuous.
        ///
        /// # Safety
        /// As [`materialized`]; `scores` needs `PANEL*s` writes, which `s*s` covers for `s >= PANEL`.
        unsafe fn materialized_causal(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            scores: *mut f32,
            o: *mut f32,
            s: usize,
            d: usize,
            scale: f32,
        ) {
            const PANEL: usize = 64;
            let mut i0 = 0;
            while i0 < s {
                let pb = (s - i0).min(PANEL);
                let jmax = i0 + pb;
                wukong_sgemm_nt(
                    q.as_ptr().add(i0 * d),
                    k.as_ptr(),
                    scores,
                    pb as i64,
                    d as i64,
                    jmax as i64,
                    0,
                );
                for ii in 0..pb {
                    let row = std::slice::from_raw_parts_mut(scores.add(ii * jmax), jmax);
                    let ke = i0 + ii + 1;
                    for r in row.iter_mut().take(ke) {
                        *r *= scale;
                    }
                    softmax_row_fast(&mut row[..ke]);
                    for r in row.iter_mut().skip(ke) {
                        *r = 0.0;
                    }
                }
                wukong_sgemm(
                    scores,
                    v.as_ptr(),
                    o.add(i0 * d),
                    pb as i64,
                    jmax as i64,
                    d as i64,
                    0,
                );
                i0 += PANEL;
            }
        }

        let bench = |f: &dyn Fn()| {
            for _ in 0..3 {
                f();
            }
            let mut best = f64::INFINITY;
            for _ in 0..20 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best * 1e3
        };

        let avx = {
            #[cfg(target_arch = "x86_64")]
            {
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        };

        for &causal in &[false, true] {
            println!("--- causal={causal} (ms, best of 20; ratios are control/fused) ---");
            for &(s, d) in &[(256usize, 64usize), (512, 64), (1024, 64), (2048, 64)] {
                let q = fill(1, s * d);
                let k = fill(2, s * d);
                let v = fill(3, s * d);
                let scale = 1.0 / (d as f32).sqrt();
                let mut o = vec![0.0f32; s * d];
                let mut oc = vec![0.0f32; s * d];
                let mut scores = vec![0.0f32; s * s];
                let (qp, kp, vp, op) = (q.as_ptr(), k.as_ptr(), v.as_ptr(), o.as_mut_ptr());
                let ocp = oc.as_mut_ptr();
                let scp = scores.as_mut_ptr();

                let mat_libm = if causal {
                    f64::NAN
                } else {
                    bench(&|| {
                        // SAFETY: scores is s*s, oc is s*d, q/k/v are s*d.
                        unsafe {
                            materialized(&q, &k, &v, scp, ocp, s, d, scale, true);
                        }
                        std::hint::black_box(ocp);
                    })
                };
                let mat_v8 = bench(&|| {
                    // SAFETY: as above; the causal arm needs only PANEL*s of `scores`.
                    unsafe {
                        if causal {
                            materialized_causal(&q, &k, &v, scp, ocp, s, d, scale);
                        } else {
                            materialized(&q, &k, &v, scp, ocp, s, d, scale, false);
                        }
                    }
                    std::hint::black_box(ocp);
                });
                let legacy = if avx {
                    bench(&|| {
                        #[cfg(target_arch = "x86_64")]
                        // SAFETY: avx2+fma checked; every buffer is s*d long.
                        unsafe {
                            attention_avx2_legacy(qp, kp, vp, op, s, d, scale, causal);
                        }
                    })
                } else {
                    f64::NAN
                };
                let fused = bench(&|| unsafe {
                    wukong_attention_f32(qp, kp, vp, op, s as i64, d as i64, scale, causal as i64);
                });

                // The control and the fused kernel must be computing the same thing, or the
                // timings compare two different programs. Checked once per shape, outside the
                // timed region, on the buffers the runs just left behind.
                for i in 0..s * d {
                    assert!(
                        (o[i] - oc[i]).abs() <= 1e-4 + 1e-4 * oc[i].abs(),
                        "control disagrees at ({s}x{d}, causal={causal}) idx {i}: fused {} \
                         materialized {}",
                        o[i],
                        oc[i]
                    );
                }

                println!(
                    "S={s:<5} D={d}: mat-libm {mat_libm:8.3} | mat-exp8 {mat_v8:8.3} | legacy \
                     {legacy:8.3} | fused {fused:8.3} || vs-libm {:5.2}x  vs-exp8 {:5.2}x  \
                     vs-legacy {:5.2}x",
                    mat_libm / fused,
                    mat_v8 / fused,
                    legacy / fused
                );
            }
        }
    }
}
