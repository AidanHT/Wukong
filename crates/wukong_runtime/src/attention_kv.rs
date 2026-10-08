//! Decode attention over a KV cache — the runtime of the `sdpa_kv` and `kv_append` builtins.
//!
//! [`wukong_attention_f32`](crate::wukong_attention_f32) is the square, single-head kernel: one
//! `S` for both the queries and the keys, the causal mask `j > i`, packed `[S, D]` rows. An
//! autoregressive decode step is none of those — one new query (or a chunk of them) against a cache
//! of every earlier key, at a position offset, with fewer key/value heads than query heads (GQA /
//! MQA) and heads interleaved in each token's row. [`wukong_attention_kv_f32`] is that operation,
//! the shape of llama.cpp's `ggml_flash_attn_ext` and ONNX Runtime's `GroupQueryAttention`:
//!
//! * `q` and `o` hold `sq` query rows of stride `q_stride`; head `h` of row `i` is the `d` elements
//!   at `i·q_stride + h·d`. `k` and `v` hold `skv` cached rows of stride `kv_stride`; KV head `g` of
//!   key `j` is at `j·kv_stride + g·d`.
//! * Query head `h` attends with KV head `h / (n_heads / n_kv_heads)` (`n_kv_heads` divides
//!   `n_heads`; `n_kv_heads == n_heads` is ordinary multi-head attention, `1` is MQA).
//! * Query row `i` sits at absolute position `q_pos0 + i`; with `causal != 0` it sees keys
//!   `j <= q_pos0 + i` (and never a key past `skv`), so a decode step is `sq = 1,
//!   q_pos0 = skv - 1`, and a chunked prefill against a cache is `q_pos0 = skv - sq`.
//! * `o = softmax(scale · q·kᵀ) · v` per (row, head), written at `o`'s head offsets only: elements
//!   between heads or past a row's heads (a stride wider than `n_heads·d`) are left untouched.
//!
//! The entry validates every extent against the operand lengths before it reads anything
//! ([`kv_attention_plan`], shared with the interpreter so both backends return the same status), and
//! an output overlapping an input is computed into scratch (value semantics — the interpreter's
//! copy-in/copy-out answer).
//!
//! # Numeric contract
//!
//! Each (row, head) runs one online-softmax recurrence over its visible keys in blocks of
//! [`KV_BC`] = 64, starting at key 0 — the same blocks whatever `sq`, `q_pos0` or the thread count,
//! so a token decoded one step at a time gets **bit-for-bit** the row a causal prefill of the whole
//! sequence produces, and `_parallel` (rows split across cores) equals serial. Per key block:
//!
//! * each score is `q·k` as eight FMA lanes over `d` (lane `l` takes `t ≡ l (mod 8)` below
//!   `d & !7`), folded `((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7))`, plus a sequential FMA tail over the
//!   last `d % 8` elements when there is one, times `scale`;
//! * the block max folds `(a > b) ? a : b` (`_mm256_max_ps`'s rule) eight lanes wide from the
//!   running max, then a fixed tree, then the tail; the rescale factor is `1` when the max did not
//!   move and `exp(m_old - m_new)` otherwise;
//! * the weights are `exp(s - m)` on the crate's own `exp` ([`crate::vmath::exp1`], whose 8-lane
//!   twin [`crate::vmath::exp8`] is bit-identical), summed eight lanes wide then by a fixed tree;
//! * `acc[t] = acc[t]·corr`, then one FMA per key in key order; finally `o = acc · (1/l)`.
//!
//! The AVX2 path and the portable twin perform exactly those operations, so they agree **bit for
//! bit** (`avx2_and_scalar_twins_agree_bit_for_bit`) — stricter than the square kernel, whose
//! tiles reassociate. Exceptional values propagate as in `attention.rs`: a NaN score or value
//! poisons the row it reaches, and a causal row never reads a key or value past its position.

#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::gemv_half::bytes_overlap;
use crate::vmath::exp1;

/// Status: success (including a well-formed call that has nothing to compute).
pub const KV_OK: i64 = 0;
/// Status: a negative extent, `n_kv_heads` zero or not dividing `n_heads`, or a stride narrower
/// than its row of heads. Nothing was read or written.
pub const KV_BAD_GEOMETRY: i64 = -1;
/// Status: an operand shorter than the rows the geometry addresses. Nothing was read or written.
pub const KV_SHORT_OPERAND: i64 = -2;

/// Keys per online-softmax block: the rescale period, and the length of the score scratch.
pub const KV_BC: usize = 64;

/// A validated decode-attention call (see the module docs for the layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPlan {
    pub sq: usize,
    pub skv: usize,
    pub d: usize,
    pub q_pos0: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub q_stride: usize,
    pub kv_stride: usize,
    /// Elements of `q` the call reads, and of `o` it may write: `(sq-1)·q_stride + n_heads·d`.
    pub q_extent: usize,
    /// Elements of `k` and of `v` the call reads: `(skv-1)·kv_stride + n_kv_heads·d`, `0` when the
    /// cache is empty.
    pub kv_extent: usize,
}

impl KvPlan {
    /// The offset of query row `i`, head `h` in `q` and in `o`.
    pub fn q_off(&self, i: usize, h: usize) -> usize {
        i * self.q_stride + h * self.d
    }

    /// The KV head query head `h` reads.
    pub fn kv_head(&self, h: usize) -> usize {
        h / (self.n_heads / self.n_kv_heads)
    }

    /// The number of leading keys query row `i` attends to.
    pub fn keys(&self, i: usize, causal: bool) -> usize {
        if causal {
            self.skv
                .min(self.q_pos0.saturating_add(i).saturating_add(1))
        } else {
            self.skv
        }
    }
}

/// Validate a `wukong_attention_kv_f32` call. `geo` is `[sq, skv, d, q_pos0, n_heads, n_kv_heads,
/// q_stride, kv_stride]`. `Ok(None)` is a well-formed call with nothing to compute (`sq`, `n_heads`
/// or `d` is zero); `Err` carries [`KV_BAD_GEOMETRY`] or [`KV_SHORT_OPERAND`].
pub fn kv_attention_plan(
    q_len: i64,
    k_len: i64,
    v_len: i64,
    o_len: i64,
    geo: [i64; 8],
) -> Result<Option<KvPlan>, i64> {
    let [sq, skv, d, q_pos0, n_heads, n_kv_heads, q_stride, kv_stride] = geo;
    if geo.iter().any(|&g| g < 0) || [q_len, k_len, v_len, o_len].iter().any(|&l| l < 0) {
        return Err(KV_BAD_GEOMETRY);
    }
    if sq == 0 || n_heads == 0 || d == 0 {
        return Ok(None);
    }
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(KV_BAD_GEOMETRY);
    }
    let q_row = i128::from(n_heads) * i128::from(d);
    let kv_row = i128::from(n_kv_heads) * i128::from(d);
    if i128::from(q_stride) < q_row || (skv > 0 && i128::from(kv_stride) < kv_row) {
        return Err(KV_BAD_GEOMETRY);
    }
    let q_extent = (i128::from(sq) - 1) * i128::from(q_stride) + q_row;
    let kv_extent = if skv == 0 {
        0
    } else {
        (i128::from(skv) - 1) * i128::from(kv_stride) + kv_row
    };
    if q_extent > i128::from(q_len)
        || q_extent > i128::from(o_len)
        || kv_extent > i128::from(k_len)
        || kv_extent > i128::from(v_len)
    {
        return Err(KV_SHORT_OPERAND);
    }
    // Every extent is now bounded by an operand length, so each fits a `usize`.
    Ok(Some(KvPlan {
        sq: sq as usize,
        skv: skv as usize,
        d: d as usize,
        q_pos0: q_pos0 as usize,
        n_heads: n_heads as usize,
        n_kv_heads: n_kv_heads as usize,
        q_stride: q_stride as usize,
        kv_stride: kv_stride as usize,
        q_extent: q_extent as usize,
        kv_extent: kv_extent as usize,
    }))
}

/// One (row, head)'s view of the cache: its KV head's first key and value, the row stride, and how
/// many leading keys it attends to.
#[derive(Clone, Copy)]
struct KvRow {
    k: *const f32,
    v: *const f32,
    stride: usize,
    keys: usize,
}

// ---- The portable twin ---------------------------------------------------------------------

/// `q·k` by the module's dot contract (eight FMA lanes, the fixed tree, the FMA tail).
///
/// # Safety
/// `a` and `b` valid for `d` f32 reads; `dv == d & !7`.
unsafe fn dot_scalar(a: *const f32, b: *const f32, d: usize, dv: usize) -> f32 {
    let mut lane = [0.0f32; 8];
    let mut t = 0;
    while t < dv {
        for l in 0..8 {
            lane[l] = (*a.add(t + l)).mul_add(*b.add(t + l), lane[l]);
        }
        t += 8;
    }
    let tree =
        ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7]));
    if t == d {
        return tree;
    }
    let mut tail = 0.0f32;
    while t < d {
        tail = (*a.add(t)).mul_add(*b.add(t), tail);
        t += 1;
    }
    tree + tail
}

/// `_mm256_max_ps`'s lane rule: the first operand only when it is strictly greater.
#[inline]
fn max_rule(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

/// The block max folded from `init`, emulating the AVX2 lane layout and reduction tree exactly.
fn row_max_scalar(row: &[f32], init: f32) -> f32 {
    let nv = row.len() & !7;
    let mut best = init;
    if nv != 0 {
        let mut lane = [init; 8];
        for c in (0..nv).step_by(8) {
            for l in 0..8 {
                lane[l] = max_rule(lane[l], row[c + l]);
            }
        }
        let a = [
            max_rule(lane[0], lane[4]),
            max_rule(lane[1], lane[5]),
            max_rule(lane[2], lane[6]),
            max_rule(lane[3], lane[7]),
        ];
        best = max_rule(max_rule(a[0], a[2]), max_rule(a[1], a[3]));
    }
    for &x in &row[nv..] {
        best = max_rule(best, x);
    }
    best
}

/// In place `row[j] = exp(row[j] - mn)`, returning their sum in the AVX2 lane layout and tree.
fn exp_row_scalar(row: &mut [f32], mn: f32) -> f32 {
    let nv = row.len() & !7;
    let mut sv = [0.0f32; 8];
    for c in (0..nv).step_by(8) {
        for l in 0..8 {
            let p = exp1(row[c + l] - mn);
            row[c + l] = p;
            sv[l] += p;
        }
    }
    let mut rs = if nv != 0 {
        let s = [sv[0] + sv[4], sv[1] + sv[5], sv[2] + sv[6], sv[3] + sv[7]];
        (s[0] + s[1]) + (s[2] + s[3])
    } else {
        0.0
    };
    for x in &mut row[nv..] {
        let p = exp1(*x - mn);
        *x = p;
        rs += p;
    }
    rs
}

/// One (row, head) by the module's numeric contract — the portable twin of [`attend_row_avx2`].
///
/// # Safety
/// `qv` valid for `d` reads, `out` for `d` writes; key and value `j < kv.keys` valid for `d` reads
/// at `kv.k + j·stride`, `kv.v + j·stride`; `acc.len() >= d`.
unsafe fn attend_row_scalar(
    qv: *const f32,
    kv: KvRow,
    d: usize,
    scale: f32,
    out: *mut f32,
    acc: &mut [f32],
    sc: &mut [f32; KV_BC],
) {
    let dv = d & !7;
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    acc[..d].fill(0.0);
    let mut j0 = 0;
    while j0 < kv.keys {
        let kb = (kv.keys - j0).min(KV_BC);
        for jj in 0..kb {
            sc[jj] = dot_scalar(qv, kv.k.add((j0 + jj) * kv.stride), d, dv) * scale;
        }
        let mn = row_max_scalar(&sc[..kb], m);
        let corr = if m == mn { 1.0 } else { exp1(m - mn) };
        let sum = exp_row_scalar(&mut sc[..kb], mn);
        l = l * corr + sum;
        m = mn;
        for t in 0..d {
            let mut a = acc[t] * corr;
            for jj in 0..kb {
                a = sc[jj].mul_add(*kv.v.add((j0 + jj) * kv.stride + t), a);
            }
            acc[t] = a;
        }
        j0 += kb;
    }
    let inv = if l != 0.0 { 1.0 / l } else { 0.0 };
    for t in 0..d {
        *out.add(t) = acc[t] * inv;
    }
}

// ---- The AVX2 path -------------------------------------------------------------------------

/// Four keys' dots against one query: lanes, tree and tail exactly as [`dot_scalar`].
///
/// # Safety
/// `a` and each `k[r]` valid for `d` f32 reads; `dv == d & !7`; AVX2 + FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4_avx2(a: *const f32, k: [*const f32; 4], d: usize, dv: usize) -> __m128 {
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut t = 0;
    while t < dv {
        let av = _mm256_loadu_ps(a.add(t));
        for r in 0..4 {
            acc[r] = _mm256_fmadd_ps(av, _mm256_loadu_ps(k[r].add(t)), acc[r]);
        }
        t += 8;
    }
    // hadd(x, y) = [x0+x1, x2+x3, y0+y1, y2+y3 | x4+x5, x6+x7, y4+y5, y6+y7], so two levels and a
    // 128-bit add leave key r's ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)) in lane r.
    let t0 = _mm256_hadd_ps(acc[0], acc[1]);
    let t1 = _mm256_hadd_ps(acc[2], acc[3]);
    let t2 = _mm256_hadd_ps(t0, t1);
    let mut sums = _mm_add_ps(_mm256_castps256_ps128(t2), _mm256_extractf128_ps(t2, 1));
    if t < d {
        let mut tails = [0.0f32; 4];
        while t < d {
            let x = *a.add(t);
            for r in 0..4 {
                tails[r] = x.mul_add(*k[r].add(t), tails[r]);
            }
            t += 1;
        }
        sums = _mm_add_ps(sums, _mm_loadu_ps(tails.as_ptr()));
    }
    sums
}

/// One key's dot: lanes, tree and tail exactly as [`dot_scalar`].
///
/// # Safety
/// `a` and `b` valid for `d` f32 reads; `dv == d & !7`; AVX2 + FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot1_avx2(a: *const f32, b: *const f32, d: usize, dv: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut t = 0;
    while t < dv {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(t)), _mm256_loadu_ps(b.add(t)), acc);
        t += 8;
    }
    let h = _mm256_hadd_ps(acc, acc);
    let h = _mm256_hadd_ps(h, h);
    let tree = _mm_cvtss_f32(_mm_add_ss(
        _mm256_castps256_ps128(h),
        _mm256_extractf128_ps(h, 1),
    ));
    if t == d {
        return tree;
    }
    let mut tail = 0.0f32;
    while t < d {
        tail = (*a.add(t)).mul_add(*b.add(t), tail);
        t += 1;
    }
    tree + tail
}

/// [`row_max_scalar`]'s vector form.
///
/// # Safety
/// `row` valid for `n` reads; AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_max_avx2(row: *const f32, n: usize, init: f32) -> f32 {
    let nv = n & !7;
    let mut best = init;
    let mut t = 0;
    if nv != 0 {
        let mut mv = _mm256_set1_ps(init);
        while t < nv {
            mv = _mm256_max_ps(mv, _mm256_loadu_ps(row.add(t)));
            t += 8;
        }
        let a = _mm_max_ps(_mm256_castps256_ps128(mv), _mm256_extractf128_ps(mv, 1));
        let b = _mm_max_ps(a, _mm_movehl_ps(a, a));
        best = _mm_cvtss_f32(_mm_max_ss(b, _mm_shuffle_ps(b, b, 0x55)));
    }
    while t < n {
        best = max_rule(best, *row.add(t));
        t += 1;
    }
    best
}

/// [`exp_row_scalar`]'s vector form.
///
/// # Safety
/// `row` valid for `n` reads and writes; AVX2 + FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_row_avx2(row: *mut f32, n: usize, mn: f32) -> f32 {
    let nv = n & !7;
    let mnv = _mm256_set1_ps(mn);
    let mut sumv = _mm256_setzero_ps();
    let mut t = 0;
    while t < nv {
        let p = crate::vmath::exp8(_mm256_sub_ps(_mm256_loadu_ps(row.add(t)), mnv));
        _mm256_storeu_ps(row.add(t), p);
        sumv = _mm256_add_ps(sumv, p);
        t += 8;
    }
    let mut rs = if nv != 0 {
        let s = _mm_add_ps(_mm256_castps256_ps128(sumv), _mm256_extractf128_ps(sumv, 1));
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(_mm_hadd_ps(s, s))
    } else {
        0.0
    };
    while t < n {
        let p = exp1(*row.add(t) - mn);
        *row.add(t) = p;
        rs += p;
        t += 1;
    }
    rs
}

/// One (row, head) on AVX2 — the operations of [`attend_row_scalar`], eight lanes at a time.
///
/// # Safety
/// The contract of [`attend_row_scalar`], with `acc` valid for `d` and `sc` for [`KV_BC`] f32;
/// AVX2 + FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attend_row_avx2(
    qv: *const f32,
    kv: KvRow,
    d: usize,
    scale: f32,
    out: *mut f32,
    acc: *mut f32,
    sc: *mut f32,
) {
    let dv = d & !7;
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    for t in 0..d {
        *acc.add(t) = 0.0;
    }
    let scv = _mm_set1_ps(scale);
    let mut j0 = 0;
    while j0 < kv.keys {
        let kb = (kv.keys - j0).min(KV_BC);
        let mut jj = 0;
        while jj + 4 <= kb {
            let base = kv.k.add((j0 + jj) * kv.stride);
            let keys = [
                base,
                base.add(kv.stride),
                base.add(2 * kv.stride),
                base.add(3 * kv.stride),
            ];
            _mm_storeu_ps(sc.add(jj), _mm_mul_ps(dot4_avx2(qv, keys, d, dv), scv));
            jj += 4;
        }
        while jj < kb {
            *sc.add(jj) = dot1_avx2(qv, kv.k.add((j0 + jj) * kv.stride), d, dv) * scale;
            jj += 1;
        }
        let mn = row_max_avx2(sc, kb, m);
        let corr = if m == mn { 1.0 } else { exp1(m - mn) };
        let sum = exp_row_avx2(sc, kb, mn);
        l = l * corr + sum;
        m = mn;
        // acc = acc·corr, then one FMA per key in key order, eight output lanes at a time.
        let cv = _mm256_set1_ps(corr);
        let mut t = 0;
        while t < dv {
            let mut a = _mm256_mul_ps(_mm256_loadu_ps(acc.add(t)), cv);
            for jj in 0..kb {
                let vv = _mm256_loadu_ps(kv.v.add((j0 + jj) * kv.stride + t));
                a = _mm256_fmadd_ps(_mm256_set1_ps(*sc.add(jj)), vv, a);
            }
            _mm256_storeu_ps(acc.add(t), a);
            t += 8;
        }
        while t < d {
            let mut a = *acc.add(t) * corr;
            for jj in 0..kb {
                a = (*sc.add(jj)).mul_add(*kv.v.add((j0 + jj) * kv.stride + t), a);
            }
            *acc.add(t) = a;
            t += 1;
        }
        j0 += kb;
    }
    let inv = if l != 0.0 { 1.0 / l } else { 0.0 };
    let invv = _mm256_set1_ps(inv);
    let mut t = 0;
    while t < dv {
        _mm256_storeu_ps(out.add(t), _mm256_mul_ps(_mm256_loadu_ps(acc.add(t)), invv));
        t += 8;
    }
    while t < d {
        *out.add(t) = *acc.add(t) * inv;
        t += 1;
    }
}

#[cfg(target_arch = "x86_64")]
fn simd_available() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
fn simd_available() -> bool {
    false
}

/// The AVX2 row where it is compiled in.
///
/// # Safety
/// The contract of [`attend_row_scalar`]; AVX2 + FMA available (checked by the caller).
#[cfg(target_arch = "x86_64")]
unsafe fn attend_row_simd(
    qv: *const f32,
    kv: KvRow,
    d: usize,
    scale: f32,
    out: *mut f32,
    acc: &mut [f32],
    sc: &mut [f32; KV_BC],
) {
    attend_row_avx2(qv, kv, d, scale, out, acc.as_mut_ptr(), sc.as_mut_ptr());
}

/// The portable build has no vector row; [`simd_available`] is `false` there.
///
/// # Safety
/// The contract of [`attend_row_scalar`].
#[cfg(not(target_arch = "x86_64"))]
unsafe fn attend_row_simd(
    qv: *const f32,
    kv: KvRow,
    d: usize,
    scale: f32,
    out: *mut f32,
    acc: &mut [f32],
    sc: &mut [f32; KV_BC],
) {
    attend_row_scalar(qv, kv, d, scale, out, acc, sc);
}

/// Rows `span` of the flattened `(i, h)` row space (`r = i·n_heads + h`).
///
/// # Safety
/// The extents of `plan` hold for `q`, `k`, `v` and `o`; `o` overlaps none of them; distinct calls
/// that run concurrently cover disjoint spans.
#[allow(clippy::too_many_arguments)]
unsafe fn rows_span(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    o: *mut f32,
    plan: &KvPlan,
    scale: f32,
    causal: bool,
    span: (usize, usize),
) {
    let simd = simd_available();
    let mut acc = vec![0.0f32; plan.d];
    let mut sc = [0.0f32; KV_BC];
    for r in span.0..span.1 {
        let (i, h) = (r / plan.n_heads, r % plan.n_heads);
        let g = plan.kv_head(h);
        // `wrapping_add`: with an empty cache `k`/`v` may be null and are never read, so their
        // head offset is only an address computation.
        let kv = KvRow {
            k: k.wrapping_add(g * plan.d),
            v: v.wrapping_add(g * plan.d),
            stride: plan.kv_stride,
            keys: plan.keys(i, causal),
        };
        let (qv, out) = (q.add(plan.q_off(i, h)), o.add(plan.q_off(i, h)));
        if simd {
            attend_row_simd(qv, kv, plan.d, scale, out, &mut acc, &mut sc);
        } else {
            attend_row_scalar(qv, kv, plan.d, scale, out, &mut acc, &mut sc);
        }
    }
}

/// Every (row, head) of a validated call, `par` splitting the rows across the kernel pool
/// (bit-identical: rows are independent and each runs the same recurrence).
///
/// # Safety
/// As [`rows_span`] for the whole row space.
#[allow(clippy::too_many_arguments)]
unsafe fn attention_kv_into(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    o: *mut f32,
    plan: &KvPlan,
    scale: f32,
    causal: bool,
    par: bool,
) {
    let rows = plan.sq * plan.n_heads;
    let work = rows
        .saturating_mul(plan.skv.max(1))
        .saturating_mul(plan.d)
        .saturating_mul(2);
    if par && crate::par::fork_units(rows, work) {
        let (qa, ka, va, oa) = (q as usize, k as usize, v as usize, o as usize);
        let plan = *plan;
        crate::par::claim_chunks(rows, 1, true, move |lo, hi| {
            // SAFETY: each span owns the disjoint output rows `lo..hi`; the operand contract is
            // this function's own.
            unsafe {
                rows_span(
                    qa as *const f32,
                    ka as *const f32,
                    va as *const f32,
                    oa as *mut f32,
                    &plan,
                    scale,
                    causal,
                    (lo, hi),
                );
            }
        });
        return;
    }
    rows_span(q, k, v, o, plan, scale, causal, (0, rows));
}

/// The shared body of the serial and parallel entries.
///
/// # Safety
/// Each pointer addresses its length of f32 (`q_len`, `k_len`, `v_len`, `o_len`); a zero-length
/// operand is never dereferenced.
#[allow(clippy::too_many_arguments)]
unsafe fn attention_kv_entry(
    q: *const f32,
    q_len: i64,
    k: *const f32,
    k_len: i64,
    v: *const f32,
    v_len: i64,
    o: *mut f32,
    o_len: i64,
    geo: [i64; 8],
    scale: f32,
    causal: bool,
    par: bool,
) -> i64 {
    let plan = match kv_attention_plan(q_len, k_len, v_len, o_len, geo) {
        Ok(Some(plan)) => plan,
        Ok(None) => return KV_OK,
        Err(code) => return code,
    };
    let ob = plan.q_extent * 4;
    let kvb = plan.kv_extent * 4;
    let hazard = bytes_overlap(o as usize, ob, q as usize, ob)
        || bytes_overlap(o as usize, ob, k as usize, kvb)
        || bytes_overlap(o as usize, ob, v as usize, kvb);
    if hazard {
        let mut tmp = vec![0.0f32; plan.q_extent];
        attention_kv_into(q, k, v, tmp.as_mut_ptr(), &plan, scale, causal, par);
        for i in 0..plan.sq {
            for h in 0..plan.n_heads {
                let off = plan.q_off(i, h);
                std::ptr::copy_nonoverlapping(tmp.as_ptr().add(off), o.add(off), plan.d);
            }
        }
    } else {
        attention_kv_into(q, k, v, o, &plan, scale, causal, par);
    }
    KV_OK
}

/// Decode attention over a KV cache — the `sdpa_kv` builtin. See the module docs for the layout and
/// the numeric contract. Returns [`KV_OK`], or [`KV_BAD_GEOMETRY`] / [`KV_SHORT_OPERAND`] with
/// nothing read or written.
///
/// # Safety
/// `q`, `k`, `v` address `q_len`, `k_len`, `v_len` readable f32 and `o` `o_len` writable f32 (a
/// zero-length operand may be null). An `o` overlapping an input is handled (value semantics).
#[no_mangle]
pub unsafe extern "C" fn wukong_attention_kv_f32(
    q: *const f32,
    q_len: i64,
    k: *const f32,
    k_len: i64,
    v: *const f32,
    v_len: i64,
    o: *mut f32,
    o_len: i64,
    sq: i64,
    skv: i64,
    d: i64,
    q_pos0: i64,
    n_heads: i64,
    n_kv_heads: i64,
    q_stride: i64,
    kv_stride: i64,
    scale: f32,
    causal: i64,
) -> i64 {
    let geo = [sq, skv, d, q_pos0, n_heads, n_kv_heads, q_stride, kv_stride];
    attention_kv_entry(
        q,
        q_len,
        k,
        k_len,
        v,
        v_len,
        o,
        o_len,
        geo,
        scale,
        causal != 0,
        false,
    )
}

/// Multi-threaded [`wukong_attention_kv_f32`]: the (row, head) pairs across cores, bit-identical to
/// the serial entry.
///
/// # Safety
/// The contract of [`wukong_attention_kv_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_attention_kv_f32_parallel(
    q: *const f32,
    q_len: i64,
    k: *const f32,
    k_len: i64,
    v: *const f32,
    v_len: i64,
    o: *mut f32,
    o_len: i64,
    sq: i64,
    skv: i64,
    d: i64,
    q_pos0: i64,
    n_heads: i64,
    n_kv_heads: i64,
    q_stride: i64,
    kv_stride: i64,
    scale: f32,
    causal: i64,
) -> i64 {
    let geo = [sq, skv, d, q_pos0, n_heads, n_kv_heads, q_stride, kv_stride];
    attention_kv_entry(
        q,
        q_len,
        k,
        k_len,
        v,
        v_len,
        o,
        o_len,
        geo,
        scale,
        causal != 0,
        true,
    )
}

/// Validate a `kv_append(cache, row, pos, stride)` call: the row of `row_len` elements lands at
/// `pos·stride`. `Ok(None)` for an empty row (nothing to copy), `Ok(Some(offset))` otherwise;
/// [`KV_BAD_GEOMETRY`] for a negative value or a stride narrower than the row, [`KV_SHORT_OPERAND`]
/// when the cache cannot hold the row there.
pub fn kv_append_plan(
    cache_len: i64,
    row_len: i64,
    pos: i64,
    stride: i64,
) -> Result<Option<usize>, i64> {
    if cache_len < 0 || row_len < 0 || pos < 0 || stride < 0 {
        return Err(KV_BAD_GEOMETRY);
    }
    if row_len == 0 {
        return Ok(None);
    }
    if stride < row_len {
        return Err(KV_BAD_GEOMETRY);
    }
    let off = i128::from(pos) * i128::from(stride);
    if off + i128::from(row_len) > i128::from(cache_len) {
        return Err(KV_SHORT_OPERAND);
    }
    Ok(Some(off as usize))
}

/// `cache[pos·stride .. pos·stride + row_len] = row` — append one token's keys (or values) to a KV
/// cache, the `kv_append` builtin. A `memmove`, so an overlapping `row` is copied as it was before
/// the call. Returns [`KV_OK`] or a [`kv_append_plan`] status with nothing written.
///
/// # Safety
/// `cache` addresses `cache_len` writable f32 and `row` `row_len` readable f32 (a zero-length operand
/// may be null).
#[no_mangle]
pub unsafe extern "C" fn wukong_kv_append_f32(
    cache: *mut f32,
    cache_len: i64,
    row: *const f32,
    row_len: i64,
    pos: i64,
    stride: i64,
) -> i64 {
    match kv_append_plan(cache_len, row_len, pos, stride) {
        Err(code) => code,
        Ok(None) => KV_OK,
        Ok(Some(off)) => {
            std::ptr::copy(row, cache.add(off), row_len as usize);
            KV_OK
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// One decode-attention geometry: `(sq, skv, d, q_pos0, n_heads, n_kv_heads, q_pad, kv_pad)`,
    /// the strides being the packed row plus the pad.
    type Geo = (usize, usize, usize, usize, usize, usize, usize, usize);

    /// Shapes covering MHA / GQA / MQA, a decode step (`sq = 1`), a chunked prefill against a
    /// cache, the 8-wide `d` tail, the 64-key block boundary, and padded (strided) rows.
    const GEOS: &[Geo] = &[
        (1, 1, 1, 0, 1, 1, 0, 0),
        (1, 5, 8, 4, 1, 1, 0, 0),
        (1, 70, 16, 69, 4, 2, 0, 0),
        (3, 9, 7, 6, 2, 1, 3, 1),
        (4, 130, 33, 126, 6, 3, 0, 5),
        (2, 64, 64, 62, 2, 2, 0, 0),
        (5, 5, 12, 0, 4, 4, 2, 0),
        (2, 200, 9, 150, 8, 1, 0, 0),
    ];

    struct Case {
        plan: KvPlan,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        scale: f32,
    }

    fn case(g: Geo, seed: u64) -> Case {
        let (sq, skv, d, q_pos0, nh, nkv, q_pad, kv_pad) = g;
        let (qs, kvs) = (nh * d + q_pad, nkv * d + kv_pad);
        let q_len = (sq - 1) * qs + nh * d;
        let kv_len = (skv - 1) * kvs + nkv * d;
        let geo = [sq, skv, d, q_pos0, nh, nkv, qs, kvs].map(|x| x as i64);
        let plan = kv_attention_plan(
            q_len as i64,
            kv_len as i64,
            kv_len as i64,
            q_len as i64,
            geo,
        )
        .unwrap()
        .unwrap();
        Case {
            plan,
            q: fill(seed, q_len),
            k: fill(seed + 1, kv_len),
            v: fill(seed + 2, kv_len),
            scale: 1.0 / (d as f32).sqrt(),
        }
    }

    type KvEntry = unsafe extern "C" fn(
        *const f32,
        i64,
        *const f32,
        i64,
        *const f32,
        i64,
        *mut f32,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        f32,
        i64,
    ) -> i64;

    /// Call the entry (serial or parallel) on `c` with an output pre-filled with a sentinel.
    fn run(c: &Case, causal: bool, par: bool) -> Vec<f32> {
        let p = c.plan;
        let mut o = vec![-7.5f32; p.q_extent];
        let entry: KvEntry = if par {
            wukong_attention_kv_f32_parallel
        } else {
            wukong_attention_kv_f32
        };
        // SAFETY: every buffer holds exactly the extent the plan validated.
        let code = unsafe {
            entry(
                c.q.as_ptr(),
                c.q.len() as i64,
                c.k.as_ptr(),
                c.k.len() as i64,
                c.v.as_ptr(),
                c.v.len() as i64,
                o.as_mut_ptr(),
                o.len() as i64,
                p.sq as i64,
                p.skv as i64,
                p.d as i64,
                p.q_pos0 as i64,
                p.n_heads as i64,
                p.n_kv_heads as i64,
                p.q_stride as i64,
                p.kv_stride as i64,
                c.scale,
                causal as i64,
            )
        };
        assert_eq!(code, KV_OK);
        o
    }

    /// Naive f64 reference: materialized scores, max-subtracted softmax, weighted values.
    fn naive(c: &Case, causal: bool) -> Vec<f64> {
        let p = c.plan;
        let mut o = vec![f64::NAN; p.q_extent];
        for i in 0..p.sq {
            for h in 0..p.n_heads {
                let g = p.kv_head(h);
                let qo = p.q_off(i, h);
                let keys = p.keys(i, causal);
                let mut s = vec![0.0f64; keys];
                for (j, sj) in s.iter_mut().enumerate() {
                    let ko = j * p.kv_stride + g * p.d;
                    let mut x = 0.0f64;
                    for t in 0..p.d {
                        x += c.q[qo + t] as f64 * c.k[ko + t] as f64;
                    }
                    *sj = x * c.scale as f64;
                }
                let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = s.iter().map(|&x| (x - mx).exp()).collect();
                let sum: f64 = w.iter().sum();
                for t in 0..p.d {
                    let mut acc = 0.0f64;
                    for (j, &wj) in w.iter().enumerate() {
                        acc += wj * c.v[j * p.kv_stride + g * p.d + t] as f64;
                    }
                    o[qo + t] = if keys == 0 { 0.0 } else { acc / sum };
                }
            }
        }
        o
    }

    #[test]
    fn matches_a_naive_reference_and_leaves_the_gaps_alone() {
        for (n, &g) in GEOS.iter().enumerate() {
            let c = case(g, 10 * n as u64 + 1);
            for causal in [false, true] {
                let got = run(&c, causal, false);
                let want = naive(&c, causal);
                for (e, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
                    if wv.is_nan() {
                        // A gap between heads / rows: never written.
                        assert_eq!(gv, -7.5, "{g:?} causal={causal}: gap {e} written");
                    } else {
                        assert!(
                            (gv as f64 - wv).abs() <= 1e-4 + 1e-4 * wv.abs(),
                            "{g:?} causal={causal} [{e}]: got {gv} want {wv}"
                        );
                    }
                }
            }
        }
    }

    /// The AVX2 row and the portable twin perform the same operations in the same order: compare
    /// every output bit, including the wide score spreads that flush most weights to zero.
    #[test]
    fn avx2_and_scalar_twins_agree_bit_for_bit() {
        if !simd_available() {
            return;
        }
        for (n, &g) in GEOS.iter().enumerate() {
            for spread in [1.0f32, 300.0] {
                let mut c = case(g, 7 * n as u64 + 3);
                for x in c.q.iter_mut() {
                    *x *= spread;
                }
                for causal in [false, true] {
                    let p = c.plan;
                    let mut acc = vec![0.0f32; p.d];
                    let mut sc = [0.0f32; KV_BC];
                    for i in 0..p.sq {
                        for h in 0..p.n_heads {
                            let gh = p.kv_head(h);
                            let kv = KvRow {
                                k: c.k.as_ptr().wrapping_add(gh * p.d),
                                v: c.v.as_ptr().wrapping_add(gh * p.d),
                                stride: p.kv_stride,
                                keys: p.keys(i, causal),
                            };
                            let qv = c.q.as_ptr().wrapping_add(p.q_off(i, h));
                            let mut a = vec![0.0f32; p.d];
                            let mut b = vec![0.0f32; p.d];
                            // SAFETY: the plan validated every row these pointers address.
                            unsafe {
                                attend_row_scalar(
                                    qv,
                                    kv,
                                    p.d,
                                    c.scale,
                                    a.as_mut_ptr(),
                                    &mut acc,
                                    &mut sc,
                                );
                                attend_row_simd(
                                    qv,
                                    kv,
                                    p.d,
                                    c.scale,
                                    b.as_mut_ptr(),
                                    &mut acc,
                                    &mut sc,
                                );
                            }
                            let (ab, bb): (Vec<u32>, Vec<u32>) = (
                                a.iter().map(|x| x.to_bits()).collect(),
                                b.iter().map(|x| x.to_bits()).collect(),
                            );
                            assert_eq!(ab, bb, "{g:?} spread={spread} causal={causal} [{i}][{h}]");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn parallel_equals_serial_bit_for_bit() {
        for (n, &g) in GEOS.iter().enumerate() {
            let c = case(g, 5 * n as u64 + 2);
            for causal in [false, true] {
                let a: Vec<u32> = run(&c, causal, false).iter().map(|x| x.to_bits()).collect();
                let b: Vec<u32> = run(&c, causal, true).iter().map(|x| x.to_bits()).collect();
                assert_eq!(a, b, "{g:?} causal={causal}");
            }
        }
    }

    /// Decoding one token at a time against a growing cache (`sq = 1`, `q_pos0 = t`, the cache
    /// filled by `wukong_kv_append_f32`) yields bit-for-bit the rows of one causal prefill of the
    /// whole sequence — the key blocks start at key 0 whatever the query offset.
    #[test]
    fn incremental_decode_equals_causal_prefill() {
        let (t_len, d, nh, nkv) = (70usize, 12usize, 4usize, 2usize);
        let (qs, kvs) = (nh * d, nkv * d);
        let q = fill(41, t_len * qs);
        let k = fill(42, t_len * kvs);
        let v = fill(43, t_len * kvs);
        let scale = 0.25f32;
        let mut full = vec![0.0f32; t_len * qs];
        // SAFETY: packed buffers sized to the geometry.
        let code = unsafe {
            wukong_attention_kv_f32(
                q.as_ptr(),
                q.len() as i64,
                k.as_ptr(),
                k.len() as i64,
                v.as_ptr(),
                v.len() as i64,
                full.as_mut_ptr(),
                full.len() as i64,
                t_len as i64,
                t_len as i64,
                d as i64,
                0,
                nh as i64,
                nkv as i64,
                qs as i64,
                kvs as i64,
                scale,
                1,
            )
        };
        assert_eq!(code, KV_OK);
        let mut kc = vec![0.0f32; t_len * kvs];
        let mut vc = vec![0.0f32; t_len * kvs];
        for t in 0..t_len {
            let mut o = vec![0.0f32; qs];
            // SAFETY: the cache holds `t_len` rows; each call addresses rows `..=t`.
            unsafe {
                let row = |b: &[f32]| b.as_ptr().add(t * kvs);
                assert_eq!(
                    wukong_kv_append_f32(
                        kc.as_mut_ptr(),
                        kc.len() as i64,
                        row(&k),
                        kvs as i64,
                        t as i64,
                        kvs as i64
                    ),
                    KV_OK
                );
                assert_eq!(
                    wukong_kv_append_f32(
                        vc.as_mut_ptr(),
                        vc.len() as i64,
                        row(&v),
                        kvs as i64,
                        t as i64,
                        kvs as i64
                    ),
                    KV_OK
                );
                let code = wukong_attention_kv_f32(
                    q.as_ptr().add(t * qs),
                    qs as i64,
                    kc.as_ptr(),
                    ((t + 1) * kvs) as i64,
                    vc.as_ptr(),
                    ((t + 1) * kvs) as i64,
                    o.as_mut_ptr(),
                    qs as i64,
                    1,
                    (t + 1) as i64,
                    d as i64,
                    t as i64,
                    nh as i64,
                    nkv as i64,
                    qs as i64,
                    kvs as i64,
                    scale,
                    1,
                );
                assert_eq!(code, KV_OK);
            }
            let want: Vec<u32> = full[t * qs..(t + 1) * qs]
                .iter()
                .map(|x| x.to_bits())
                .collect();
            let got: Vec<u32> = o.iter().map(|x| x.to_bits()).collect();
            assert_eq!(got, want, "step {t}");
        }
    }

    /// `n_heads = n_kv_heads = 1` with packed rows and `q_pos0 = 0` is the square kernel's problem;
    /// the two kernels block differently, so they agree to the square kernel's own tolerance band.
    #[test]
    fn the_square_case_agrees_with_the_flash_kernel() {
        for &(s, d) in &[(1usize, 1usize), (5, 7), (33, 17), (65, 16), (130, 24)] {
            let q = fill(51, s * d);
            let k = fill(52, s * d);
            let v = fill(53, s * d);
            let scale = 1.0 / (d as f32).sqrt();
            for causal in [0i64, 1] {
                let mut a = vec![0.0f32; s * d];
                let mut b = vec![0.0f32; s * d];
                let n = (s * d) as i64;
                // SAFETY: every buffer is s*d.
                unsafe {
                    crate::wukong_attention_f32(
                        q.as_ptr(),
                        k.as_ptr(),
                        v.as_ptr(),
                        a.as_mut_ptr(),
                        s as i64,
                        d as i64,
                        scale,
                        causal,
                    );
                    let code = wukong_attention_kv_f32(
                        q.as_ptr(),
                        n,
                        k.as_ptr(),
                        n,
                        v.as_ptr(),
                        n,
                        b.as_mut_ptr(),
                        n,
                        s as i64,
                        s as i64,
                        d as i64,
                        0,
                        1,
                        1,
                        d as i64,
                        d as i64,
                        scale,
                        causal,
                    );
                    assert_eq!(code, KV_OK);
                }
                for e in 0..s * d {
                    assert!(
                        (a[e] - b[e]).abs() <= 1e-4 + 1e-4 * a[e].abs(),
                        "({s}x{d}, causal={causal}) [{e}]: flash {} kv {}",
                        a[e],
                        b[e]
                    );
                }
            }
        }
    }

    /// An output aliasing the query is computed from the query as it was before the call.
    #[test]
    fn an_output_aliasing_the_query_has_value_semantics() {
        let c = case((3, 9, 8, 6, 2, 1, 0, 0), 77);
        let want = run(&c, true, false);
        let p = c.plan;
        let mut buf = c.q.clone();
        let ptr = buf.as_mut_ptr();
        // SAFETY: `buf` holds q's extent; the entry handles the overlap.
        let code = unsafe {
            wukong_attention_kv_f32(
                ptr,
                buf.len() as i64,
                c.k.as_ptr(),
                c.k.len() as i64,
                c.v.as_ptr(),
                c.v.len() as i64,
                ptr,
                buf.len() as i64,
                p.sq as i64,
                p.skv as i64,
                p.d as i64,
                p.q_pos0 as i64,
                p.n_heads as i64,
                p.n_kv_heads as i64,
                p.q_stride as i64,
                p.kv_stride as i64,
                c.scale,
                1,
            )
        };
        assert_eq!(code, KV_OK);
        assert_eq!(
            buf.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn malformed_geometry_is_refused_before_anything_is_touched() {
        // [sq, skv, d, q_pos0, n_heads, n_kv_heads, q_stride, kv_stride]
        let ok = [1i64, 4, 8, 3, 4, 2, 32, 16];
        assert!(matches!(kv_attention_plan(32, 64, 64, 32, ok), Ok(Some(_))));
        // Nothing to compute.
        assert_eq!(
            kv_attention_plan(0, 0, 0, 0, [0, 4, 8, 3, 4, 2, 32, 16]),
            Ok(None)
        );
        // Heads that do not group, a zero KV-head count, a negative value, a narrow stride.
        let bad = |g: [i64; 8]| kv_attention_plan(1 << 20, 1 << 20, 1 << 20, 1 << 20, g);
        assert_eq!(bad([1, 4, 8, 3, 4, 3, 32, 24]), Err(KV_BAD_GEOMETRY));
        assert_eq!(bad([1, 4, 8, 3, 4, 0, 32, 16]), Err(KV_BAD_GEOMETRY));
        assert_eq!(bad([1, 4, 8, -3, 4, 2, 32, 16]), Err(KV_BAD_GEOMETRY));
        assert_eq!(bad([1, 4, 8, 3, 4, 2, 31, 16]), Err(KV_BAD_GEOMETRY));
        assert_eq!(bad([1, 4, 8, 3, 4, 2, 32, 15]), Err(KV_BAD_GEOMETRY));
        // Each operand one element short.
        assert_eq!(kv_attention_plan(31, 64, 64, 32, ok), Err(KV_SHORT_OPERAND));
        assert_eq!(kv_attention_plan(32, 63, 64, 32, ok), Err(KV_SHORT_OPERAND));
        assert_eq!(kv_attention_plan(32, 64, 63, 32, ok), Err(KV_SHORT_OPERAND));
        assert_eq!(kv_attention_plan(32, 64, 64, 31, ok), Err(KV_SHORT_OPERAND));
        // A refused call writes nothing.
        let mut o = [9.0f32; 32];
        let x = [0.5f32; 64];
        // SAFETY: the entry validates before reading; the buffers cover what it would read.
        let code = unsafe {
            wukong_attention_kv_f32(
                x.as_ptr(),
                32,
                x.as_ptr(),
                63,
                x.as_ptr(),
                64,
                o.as_mut_ptr(),
                32,
                1,
                4,
                8,
                3,
                4,
                2,
                32,
                16,
                1.0,
                1,
            )
        };
        assert_eq!(code, KV_SHORT_OPERAND);
        assert!(o.iter().all(|&v| v == 9.0));
        // An empty cache yields zero rows.
        let mut o = [9.0f32; 8];
        // SAFETY: the cache is empty, so `k`/`v` are never read.
        let code = unsafe {
            wukong_attention_kv_f32(
                x.as_ptr(),
                8,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                o.as_mut_ptr(),
                8,
                1,
                0,
                8,
                0,
                1,
                1,
                8,
                8,
                1.0,
                0,
            )
        };
        assert_eq!(code, KV_OK);
        assert_eq!(o, [0.0f32; 8]);
    }

    #[test]
    fn kv_append_validates_and_copies_one_row() {
        assert_eq!(kv_append_plan(12, 3, 2, 4), Ok(Some(8)));
        assert_eq!(kv_append_plan(12, 0, 99, 0), Ok(None));
        assert_eq!(kv_append_plan(12, 3, 3, 4), Err(KV_SHORT_OPERAND));
        assert_eq!(kv_append_plan(12, 5, 0, 4), Err(KV_BAD_GEOMETRY));
        assert_eq!(kv_append_plan(12, 3, -1, 4), Err(KV_BAD_GEOMETRY));
        assert_eq!(
            kv_append_plan(i64::MAX, 3, i64::MAX, 4),
            Err(KV_SHORT_OPERAND)
        );
        let mut cache = [0.0f32; 12];
        let row = [1.0f32, 2.0, 3.0];
        // SAFETY: sized buffers.
        let code = unsafe { wukong_kv_append_f32(cache.as_mut_ptr(), 12, row.as_ptr(), 3, 2, 4) };
        assert_eq!(code, KV_OK);
        assert_eq!(
            cache,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0]
        );
        // An overlapping source is copied as it was (memmove).
        let mut buf = [1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0];
        let p = buf.as_mut_ptr();
        // SAFETY: both ranges lie inside `buf`.
        let code = unsafe { wukong_kv_append_f32(p, 6, p.wrapping_add(1), 3, 1, 3) };
        assert_eq!(code, KV_OK);
        assert_eq!(buf, [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
    }
}
