//! Decode attention over a KV cache — the runtime of the `sdpa_kv`, `sdpa_kv_batch` and `kv_append`
//! builtins.
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
//! # A batch of sequences, padded
//!
//! [`wukong_attention_kv_batch_f32`] runs `batch` independent sequences in one call — the batched
//! decode step of a server, or a batch of chunked prefills — over packed tensors: `q` and `o` are
//! `[batch, sq, n_heads, d]`, `k` and `v` are `[batch, skv, n_kv_heads, d]`, where `skv` is the
//! cache *capacity* every sequence is padded to. Two `i32` vectors of `batch` entries describe the
//! sequences (the `cache_seqlens` of FlashAttention's `flash_attn_with_kvcache`, plus an explicit
//! position):
//!
//! * `kv_lens[b]` (`0 <= kv_lens[b] <= skv`) is how many leading cache rows of sequence `b` hold
//!   real keys; the rest is padding. **A masked key contributes exactly zero probability**: it is
//!   never read at all, so whatever the padding holds — stale rows, NaN, ±inf — cannot reach an
//!   output, and a padded sequence's rows are bit-for-bit the rows of [`wukong_attention_kv_f32`]
//!   over that sequence alone (`skv = kv_lens[b]`, `q_pos0 = q_pos[b]`).
//! * `q_pos[b]` is the absolute position of sequence `b`'s first query row, so with `causal != 0`
//!   row `i` sees keys `j <= q_pos[b] + i` (and `j < kv_lens[b]`). A decode step is `q_pos[b] =
//!   kv_lens[b] - 1`; FlashAttention's bottom-right-aligned mask is `q_pos[b] = kv_lens[b] - sq`; a
//!   right-padded prompt prefill is `q_pos[b] = 0`. A negative position is allowed: those rows see
//!   nothing yet. Non-causal rows ignore `q_pos` and see all `kv_lens[b]` keys.
//! * **A fully masked row** — one that sees no key: `kv_lens[b] == 0`, or a causal row whose
//!   position is negative — is defined to be **exactly `+0.0` in every lane** (FlashAttention's
//!   convention, not the NaN of a naive `softmax` over an empty row). The softmax denominator of a
//!   row with at least one visible key is at least `1`, so `0` is never a real output's sentinel.
//!
//! The entries validate every extent (and, for the batch, every `kv_lens[b]`) against the operand
//! lengths before anything is read or written ([`kv_attention_plan`] / [`kv_batch_plan`] +
//! [`kv_batch_check_lens`], shared with the interpreter so both backends return the same status),
//! and an output overlapping an input is computed into scratch (value semantics — the
//! interpreter's copy-in/copy-out answer). The batch reads `kv_lens` and `q_pos` once, first.
//!
//! # Numeric contract
//!
//! Each (row, head) runs one online-softmax recurrence over its visible keys in blocks of
//! [`KV_BC`] = 64, starting at key 0 — the same blocks whatever `sq`, the position, the batch or
//! the thread count, so a token decoded one step at a time gets **bit-for-bit** the row a causal
//! prefill of the whole sequence produces, a padded batch row equals its unpadded sequence, and
//! `_parallel` (work split across cores) equals serial. Per key block:
//!
//! * each score is `q·k` as eight FMA lanes over `d` (lane `l` takes `t ≡ l (mod 8)` below
//!   `d & !7`), folded `((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7))`, plus a sequential FMA tail over the
//!   last `d % 8` elements when there is one, times `scale`;
//! * the block max folds `(a > b) ? a : b` (`_mm256_max_ps`'s rule) eight lanes wide from the
//!   running max, then a fixed tree, then the tail; the rescale factor is `1` when the max did not
//!   move and `exp(m_old - m_new)` otherwise;
//! * the weights are `exp(s - m)` on the crate's own `exp` ([`crate::vmath::exp1`], whose 8-lane
//!   twin [`crate::vmath::exp8`] is bit-identical), summed eight lanes wide then by a fixed tree;
//! * `acc[t] = acc[t]·corr`, then one FMA per key in key order; finally `o = acc · (1/l)` (`0` for
//!   a row with no visible key).
//!
//! # GQA-group blocking
//!
//! The query heads that share a KV head read the same cached rows, so the AVX2 path processes them
//! together — up to [`KV_GROUP_BLOCK`] = 8 heads of one group per pass: each key row is loaded
//! once and dotted against every head's query in the same register tile, and each value row is
//! loaded once and accumulated into every head's output. A GQA model with `G` heads per KV head
//! reads its cache `min(G, 8)` times less often than one head at a time, which is what bounds a
//! long-context decode step. The tile shapes keep at most eight accumulators live: `4 keys x 1-2
//! heads`, `2 x 3-4` or `1 x 5-8` for the scores, and `8/G` lane chunks per head for the values
//! (plain MHA gets the wider value tile too). Blocking interleaves independent recurrences and
//! never reorders one: every (row, head) performs exactly the operations listed above, so the AVX2
//! path and the portable per-head twin agree **bit for bit**
//! (`grouped_avx2_matches_the_per_head_scalar_twin_bit_for_bit`) — stricter than the square
//! kernel, whose tiles reassociate. Exceptional values propagate as in `attention.rs`: a NaN score
//! or value poisons the row it reaches, and a causal or padded row never reads a key or value past
//! its window.

#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::gemv_half::bytes_overlap;
use crate::vmath::exp1;

/// Status: success (including a well-formed call that has nothing to compute).
pub const KV_OK: i64 = 0;
/// Status: a negative extent, `n_kv_heads` zero or not dividing `n_heads`, a stride narrower than
/// its row of heads, or (batched) a `kv_lens[b]` outside `0..=skv`. Nothing was written.
pub const KV_BAD_GEOMETRY: i64 = -1;
/// Status: an operand shorter than the rows the geometry addresses. Nothing was read or written.
pub const KV_SHORT_OPERAND: i64 = -2;

/// Keys per online-softmax block: the rescale period, and the length of the score scratch.
pub const KV_BC: usize = 64;

/// Query heads of one GQA group the AVX2 path processes in one pass (see the module docs).
pub const KV_GROUP_BLOCK: usize = 8;

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

/// A validated `wukong_attention_kv_batch_f32` call: packed `[batch, sq, n_heads, d]` queries and
/// outputs, packed `[batch, skv, n_kv_heads, d]` key and value caches (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvBatchPlan {
    pub batch: usize,
    pub sq: usize,
    pub skv: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d: usize,
    /// Elements of `q` the call reads and of `o` it writes: `batch·sq·n_heads·d`.
    pub q_extent: usize,
    /// Elements of `k` and of `v` the call may read: `batch·skv·n_kv_heads·d`.
    pub kv_extent: usize,
}

impl KvBatchPlan {
    /// The offset of sequence `b`'s query row `i`, head `h`, in `q` and in `o`.
    pub fn q_off(&self, b: usize, i: usize, h: usize) -> usize {
        ((b * self.sq + i) * self.n_heads + h) * self.d
    }

    /// The offset of sequence `b`'s cached row `j`, KV head `g`, in `k` and in `v`.
    pub fn kv_off(&self, b: usize, j: usize, g: usize) -> usize {
        ((b * self.skv + j) * self.n_kv_heads + g) * self.d
    }

    /// The KV head query head `h` reads.
    pub fn kv_head(&self, h: usize) -> usize {
        h / (self.n_heads / self.n_kv_heads)
    }

    /// The number of leading keys row `i` of a sequence with `kv_len` valid keys at position
    /// `q_pos` attends to (`0`: a fully masked row).
    pub fn keys(kv_len: i32, q_pos: i32, i: usize, causal: bool) -> usize {
        Seq::new(kv_len, q_pos).keys(i, causal)
    }
}

/// The product of non-negative factors, `None` past `i128`.
fn extent(factors: &[i64]) -> Option<i128> {
    factors
        .iter()
        .try_fold(1i128, |acc, &f| acc.checked_mul(i128::from(f)))
}

/// Validate a `wukong_attention_kv_batch_f32` call's shapes. `geo` is `[batch, sq, skv, n_heads,
/// n_kv_heads, d]`; `lens_len` / `pos_len` are the lengths of the `kv_lens` / `q_pos` operands.
/// `Ok(None)` is a well-formed call with nothing to compute (`batch`, `sq`, `n_heads` or `d` is
/// zero); `Err` carries [`KV_BAD_GEOMETRY`] or [`KV_SHORT_OPERAND`]. The per-sequence lengths are
/// checked separately, once they have been read ([`kv_batch_check_lens`]).
pub fn kv_batch_plan(
    q_len: i64,
    k_len: i64,
    v_len: i64,
    o_len: i64,
    lens_len: i64,
    pos_len: i64,
    geo: [i64; 6],
) -> Result<Option<KvBatchPlan>, i64> {
    let [batch, sq, skv, n_heads, n_kv_heads, d] = geo;
    let lens = [q_len, k_len, v_len, o_len, lens_len, pos_len];
    if geo.iter().any(|&g| g < 0) || lens.iter().any(|&l| l < 0) {
        return Err(KV_BAD_GEOMETRY);
    }
    if batch == 0 || sq == 0 || n_heads == 0 || d == 0 {
        return Ok(None);
    }
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(KV_BAD_GEOMETRY);
    }
    let q_extent = extent(&[batch, sq, n_heads, d]);
    let kv_extent = extent(&[batch, skv, n_kv_heads, d]);
    let fits = |e: Option<i128>, len: i64| e.is_some_and(|e| e <= i128::from(len));
    if !fits(q_extent, q_len)
        || !fits(q_extent, o_len)
        || !fits(kv_extent, k_len)
        || !fits(kv_extent, v_len)
        || batch > lens_len
        || batch > pos_len
    {
        return Err(KV_SHORT_OPERAND);
    }
    // Every extent is now bounded by an operand length, so each fits a `usize`.
    Ok(Some(KvBatchPlan {
        batch: batch as usize,
        sq: sq as usize,
        skv: skv as usize,
        n_heads: n_heads as usize,
        n_kv_heads: n_kv_heads as usize,
        d: d as usize,
        q_extent: q_extent.unwrap_or(0) as usize,
        kv_extent: kv_extent.unwrap_or(0) as usize,
    }))
}

/// Every sequence length of a validated batch call lies in `0..=skv`, else [`KV_BAD_GEOMETRY`].
/// `kv_lens` holds the call's first `batch` entries.
pub fn kv_batch_check_lens(plan: &KvBatchPlan, kv_lens: &[i32]) -> Result<(), i64> {
    let ok = kv_lens.len() >= plan.batch
        && kv_lens[..plan.batch]
            .iter()
            .all(|&l| l >= 0 && l as usize <= plan.skv);
    if ok {
        Ok(())
    } else {
        Err(KV_BAD_GEOMETRY)
    }
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

/// One sequence's key window: `len` valid cached keys, query row `i` at absolute position
/// `pos + i`.
#[derive(Clone, Copy, Debug)]
struct Seq {
    len: usize,
    pos: i64,
}

impl Seq {
    /// A batch entry; `kv_len` is non-negative (validated by [`kv_batch_check_lens`]).
    fn new(kv_len: i32, q_pos: i32) -> Seq {
        Seq {
            len: kv_len.max(0) as usize,
            pos: i64::from(q_pos),
        }
    }

    /// How many leading keys row `i` sees: every valid key, or under the causal mask those at or
    /// before its position — none for a row before position 0.
    fn keys(self, i: usize, causal: bool) -> usize {
        if !causal {
            return self.len;
        }
        let visible = i128::from(self.pos) + i as i128 + 1;
        visible.clamp(0, self.len as i128) as usize
    }
}

/// The work of one call: `batch` sequences of `sq` query rows, each sequence `q_bstride` /
/// `kv_bstride` elements after the previous one in `q`/`o` and in `k`/`v`. The unit of work is one
/// (sequence, query row, KV head, block of up to [`KV_GROUP_BLOCK`] of that head's query heads).
#[derive(Clone, Copy)]
struct Walk {
    batch: usize,
    sq: usize,
    d: usize,
    n_heads: usize,
    n_kv_heads: usize,
    q_stride: usize,
    kv_stride: usize,
    q_bstride: usize,
    kv_bstride: usize,
}

impl Walk {
    fn for_plan(p: &KvPlan) -> Walk {
        Walk {
            batch: 1,
            sq: p.sq,
            d: p.d,
            n_heads: p.n_heads,
            n_kv_heads: p.n_kv_heads,
            q_stride: p.q_stride,
            kv_stride: p.kv_stride,
            q_bstride: p.q_extent,
            kv_bstride: p.kv_extent,
        }
    }

    fn for_batch(p: &KvBatchPlan) -> Walk {
        let (q_row, kv_row) = (p.n_heads * p.d, p.n_kv_heads * p.d);
        Walk {
            batch: p.batch,
            sq: p.sq,
            d: p.d,
            n_heads: p.n_heads,
            n_kv_heads: p.n_kv_heads,
            q_stride: q_row,
            kv_stride: kv_row,
            q_bstride: p.sq * q_row,
            kv_bstride: p.skv * kv_row,
        }
    }

    /// Query heads per KV head.
    fn group(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Head blocks per KV head.
    fn blocks(&self) -> usize {
        self.group().div_ceil(KV_GROUP_BLOCK)
    }

    fn units(&self) -> usize {
        self.batch * self.sq * self.n_kv_heads * self.blocks()
    }

    /// The offset of sequence `b`'s query row `i`, head `h`, in `q` and in `o`.
    fn q_off(&self, b: usize, i: usize, h: usize) -> usize {
        b * self.q_bstride + i * self.q_stride + h * self.d
    }
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

/// One (row, head) by the module's numeric contract — the portable twin of the grouped AVX2 pass
/// [`attend_group_avx2`], one head at a time.
///
/// # Safety
/// `qv` valid for `d` reads, `out` for `d` writes; key and value `j < kv.keys` valid for `d` reads
/// at `kv.k + j·stride`, `kv.v + j·stride`; `acc.len() >= d`, `sc.len() >= KV_BC`.
unsafe fn attend_row_scalar(
    qv: *const f32,
    kv: KvRow,
    d: usize,
    scale: f32,
    out: *mut f32,
    acc: &mut [f32],
    sc: &mut [f32],
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
    // No visible key (`l == 0` with `acc` still zero): the fully masked row is exactly `+0.0`.
    let inv = if l != 0.0 { 1.0 / l } else { 0.0 };
    for t in 0..d {
        *out.add(t) = acc[t] * inv;
    }
}

// ---- The AVX2 path -------------------------------------------------------------------------

/// Four accumulators' `((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7))`, accumulator `r` in lane `r`:
/// `hadd(x, y) = [x0+x1, x2+x3, y0+y1, y2+y3 | x4+x5, x6+x7, y4+y5, y6+y7]`, so two levels and a
/// 128-bit add leave exactly [`dot_scalar`]'s tree. A macro, so it inlines into the
/// `#[target_feature]` body that uses it.
#[cfg(target_arch = "x86_64")]
macro_rules! hsum4 {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {{
        let t0 = _mm256_hadd_ps($a, $b);
        let t1 = _mm256_hadd_ps($c, $d);
        let t2 = _mm256_hadd_ps(t0, t1);
        _mm_add_ps(_mm256_castps256_ps128(t2), _mm256_extractf128_ps(t2, 1))
    }};
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

/// `NG` query heads of one GQA group (head `g` at `q + g·d`, its output at `out + g·d`) against
/// their shared KV head — the operations of [`attend_row_scalar`] for each head, with every key and
/// value row loaded once per block for all `NG` heads (see the module docs for the tiles).
///
/// # Safety
/// The contract of [`attend_row_scalar`] for each of the `NG` heads; `acc` valid for `NG·d` and
/// `sc` for `NG·KV_BC` f32; `1 <= NG <= 8`; AVX2 + FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attend_group_avx2<const NG: usize>(
    q: *const f32,
    kv: KvRow,
    d: usize,
    scale: f32,
    out: *mut f32,
    acc: *mut f32,
    sc: *mut f32,
) {
    // Keys per score tile and 8-lane chunks per value tile: at most eight live accumulators each.
    let nk: usize = if NG <= 2 {
        4
    } else if NG <= 4 {
        2
    } else {
        1
    };
    let nt: usize = (8 / NG).max(1);
    let dv = d & !7;
    for x in 0..NG * d {
        *acc.add(x) = 0.0;
    }
    let mut m = [f32::NEG_INFINITY; NG];
    let mut l = [0.0f32; NG];
    let mut corr = [1.0f32; NG];
    let mut j0 = 0;
    while j0 < kv.keys {
        let kb = (kv.keys - j0).min(KV_BC);

        // Scores: an `nk`-key x `NG`-head tile of dots per step. A short last tile repeats its
        // last live key; those dots are computed and dropped.
        let mut jj = 0;
        while jj < kb {
            let live = (kb - jj).min(nk);
            let mut kp = [kv.k; 4];
            for r in 0..nk {
                kp[r] = kv.k.add((j0 + jj + r.min(live - 1)) * kv.stride);
            }
            let mut a = [_mm256_setzero_ps(); 8];
            let mut t = 0;
            while t < dv {
                let mut kk = [_mm256_setzero_ps(); 4];
                for r in 0..nk {
                    kk[r] = _mm256_loadu_ps(kp[r].add(t));
                }
                for g in 0..NG {
                    let qv = _mm256_loadu_ps(q.add(g * d + t));
                    for r in 0..nk {
                        a[r * NG + g] = _mm256_fmadd_ps(qv, kk[r], a[r * NG + g]);
                    }
                }
                t += 8;
            }
            let mut tree = [0.0f32; 8];
            _mm_storeu_ps(tree.as_mut_ptr(), hsum4!(a[0], a[1], a[2], a[3]));
            if nk * NG > 4 {
                _mm_storeu_ps(tree.as_mut_ptr().add(4), hsum4!(a[4], a[5], a[6], a[7]));
            }
            for r in 0..live {
                for g in 0..NG {
                    let mut s = tree[r * NG + g];
                    if dv < d {
                        let mut tail = 0.0f32;
                        for t in dv..d {
                            tail = (*q.add(g * d + t)).mul_add(*kp[r].add(t), tail);
                        }
                        s += tail;
                    }
                    *sc.add(g * KV_BC + jj + r) = s * scale;
                }
            }
            jj += live;
        }

        // Per head: the block max, the rescale factor, the weights and their sum.
        for g in 0..NG {
            let row = sc.add(g * KV_BC);
            let mn = row_max_avx2(row, kb, m[g]);
            corr[g] = if m[g] == mn { 1.0 } else { exp1(m[g] - mn) };
            let sum = exp_row_avx2(row, kb, mn);
            l[g] = l[g] * corr[g] + sum;
            m[g] = mn;
        }

        // Values: acc = acc·corr, then one FMA per key in key order, `nt` chunks x `NG` heads per
        // tile, then one chunk at a time, then the scalar lanes past `d & !7`.
        let vrow = |jj: usize, t: usize| kv.v.add((j0 + jj) * kv.stride + t);
        let mut t = 0;
        while t + 8 * nt <= dv {
            let mut a = [_mm256_setzero_ps(); 8];
            for c in 0..nt {
                for g in 0..NG {
                    let cur = _mm256_loadu_ps(acc.add(g * d + t + 8 * c));
                    a[c * NG + g] = _mm256_mul_ps(cur, _mm256_set1_ps(corr[g]));
                }
            }
            for jj in 0..kb {
                for c in 0..nt {
                    let vv = _mm256_loadu_ps(vrow(jj, t + 8 * c));
                    for g in 0..NG {
                        let w = _mm256_set1_ps(*sc.add(g * KV_BC + jj));
                        a[c * NG + g] = _mm256_fmadd_ps(w, vv, a[c * NG + g]);
                    }
                }
            }
            for c in 0..nt {
                for g in 0..NG {
                    _mm256_storeu_ps(acc.add(g * d + t + 8 * c), a[c * NG + g]);
                }
            }
            t += 8 * nt;
        }
        while t < dv {
            let mut a = [_mm256_setzero_ps(); NG];
            for g in 0..NG {
                let cur = _mm256_loadu_ps(acc.add(g * d + t));
                a[g] = _mm256_mul_ps(cur, _mm256_set1_ps(corr[g]));
            }
            for jj in 0..kb {
                let vv = _mm256_loadu_ps(vrow(jj, t));
                for g in 0..NG {
                    let w = _mm256_set1_ps(*sc.add(g * KV_BC + jj));
                    a[g] = _mm256_fmadd_ps(w, vv, a[g]);
                }
            }
            for g in 0..NG {
                _mm256_storeu_ps(acc.add(g * d + t), a[g]);
            }
            t += 8;
        }
        while t < d {
            for g in 0..NG {
                let mut x = *acc.add(g * d + t) * corr[g];
                for jj in 0..kb {
                    x = (*sc.add(g * KV_BC + jj)).mul_add(*vrow(jj, t), x);
                }
                *acc.add(g * d + t) = x;
            }
            t += 1;
        }
        j0 += kb;
    }
    for g in 0..NG {
        // No visible key (`l == 0` with `acc` still zero): the fully masked row is exactly `+0.0`.
        let inv = if l[g] != 0.0 { 1.0 / l[g] } else { 0.0 };
        let invv = _mm256_set1_ps(inv);
        let (src, dst) = (acc.add(g * d), out.add(g * d));
        let mut t = 0;
        while t < dv {
            _mm256_storeu_ps(dst.add(t), _mm256_mul_ps(_mm256_loadu_ps(src.add(t)), invv));
            t += 8;
        }
        while t < d {
            *dst.add(t) = *src.add(t) * inv;
            t += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn simd_available() -> bool {
    crate::isa::avx2() && crate::isa::fma()
}

#[cfg(not(target_arch = "x86_64"))]
fn simd_available() -> bool {
    false
}

/// `nh` (`1..=8`) heads of one group on the AVX2 path where it is compiled in.
///
/// # Safety
/// The contract of [`attend_group_avx2`] for `nh` heads; AVX2 + FMA available (checked by the
/// caller).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn attend_group_simd(
    q: *const f32,
    nh: usize,
    kv: KvRow,
    d: usize,
    scale: f32,
    out: *mut f32,
    acc: &mut [f32],
    sc: &mut [f32],
) {
    debug_assert!(acc.len() >= nh * d && sc.len() >= nh * KV_BC);
    let (a, s) = (acc.as_mut_ptr(), sc.as_mut_ptr());
    match nh {
        1 => attend_group_avx2::<1>(q, kv, d, scale, out, a, s),
        2 => attend_group_avx2::<2>(q, kv, d, scale, out, a, s),
        3 => attend_group_avx2::<3>(q, kv, d, scale, out, a, s),
        4 => attend_group_avx2::<4>(q, kv, d, scale, out, a, s),
        5 => attend_group_avx2::<5>(q, kv, d, scale, out, a, s),
        6 => attend_group_avx2::<6>(q, kv, d, scale, out, a, s),
        7 => attend_group_avx2::<7>(q, kv, d, scale, out, a, s),
        _ => attend_group_avx2::<8>(q, kv, d, scale, out, a, s),
    }
}

/// The portable build has no vector path; [`simd_available`] is `false` there.
///
/// # Safety
/// The contract of [`attend_row_scalar`] for each of the `nh` heads.
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn attend_group_simd(
    q: *const f32,
    nh: usize,
    kv: KvRow,
    d: usize,
    scale: f32,
    out: *mut f32,
    acc: &mut [f32],
    sc: &mut [f32],
) {
    for h in 0..nh {
        attend_row_scalar(q.add(h * d), kv, d, scale, out.add(h * d), acc, sc);
    }
}

/// Units `span` of `w` (unit `u` = ((sequence·sq + row)·n_kv_heads + KV head)·blocks + block).
/// `simd` picks the grouped AVX2 pass over the per-head portable twin; the two agree bit for bit.
///
/// # Safety
/// `q`/`o` hold every query row `w` addresses and `k`/`v` every key row `seqs` makes visible; `o`
/// overlaps none of them; distinct calls that run concurrently cover disjoint spans; `simd` only
/// when [`simd_available`].
#[allow(clippy::too_many_arguments)]
unsafe fn units_span(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    o: *mut f32,
    w: &Walk,
    seqs: &[Seq],
    scale: f32,
    causal: bool,
    simd: bool,
    span: (usize, usize),
) {
    let (group, blocks) = (w.group(), w.blocks());
    let per_row = w.n_kv_heads * blocks;
    let mut acc = vec![0.0f32; KV_GROUP_BLOCK * w.d];
    let mut sc = vec![0.0f32; KV_GROUP_BLOCK * KV_BC];
    for u in span.0..span.1 {
        let (row, rem) = (u / per_row, u % per_row);
        let (b, i) = (row / w.sq, row % w.sq);
        let (g, blk) = (rem / blocks, rem % blocks);
        let h0 = g * group + blk * KV_GROUP_BLOCK;
        let nh = (group - blk * KV_GROUP_BLOCK).min(KV_GROUP_BLOCK);
        // `wrapping_add`: with an empty cache `k`/`v` may be null and are never read, so their
        // offset is only an address computation.
        let base = b * w.kv_bstride + g * w.d;
        let kv = KvRow {
            k: k.wrapping_add(base),
            v: v.wrapping_add(base),
            stride: w.kv_stride,
            keys: seqs[b].keys(i, causal),
        };
        let off = w.q_off(b, i, h0);
        let (qv, out) = (q.add(off), o.add(off));
        if simd {
            attend_group_simd(qv, nh, kv, w.d, scale, out, &mut acc, &mut sc);
        } else {
            for h in 0..nh {
                attend_row_scalar(
                    qv.add(h * w.d),
                    kv,
                    w.d,
                    scale,
                    out.add(h * w.d),
                    &mut acc,
                    &mut sc,
                );
            }
        }
    }
}

/// Every unit of `w`, `par` splitting them across the kernel pool (bit-identical: units are
/// independent and each (row, head) runs the same recurrence whichever unit holds it).
///
/// # Safety
/// As [`units_span`] for the whole unit space.
#[allow(clippy::too_many_arguments)]
unsafe fn attention_kv_into(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    o: *mut f32,
    w: &Walk,
    seqs: &[Seq],
    scale: f32,
    causal: bool,
    simd: bool,
    par: bool,
) {
    let units = w.units();
    let keys = seqs.iter().map(|s| s.len).max().unwrap_or(0);
    let work = (w.batch * w.sq * w.n_heads)
        .saturating_mul(keys.max(1))
        .saturating_mul(w.d)
        .saturating_mul(2);
    if par && crate::par::fork_units(units, work) {
        let (qa, ka, va, oa) = (q as usize, k as usize, v as usize, o as usize);
        let w = *w;
        crate::par::claim_chunks(units, 1, true, move |lo, hi| {
            // SAFETY: each span owns the disjoint output rows of units `lo..hi`; the operand
            // contract is this function's own.
            unsafe {
                units_span(
                    qa as *const f32,
                    ka as *const f32,
                    va as *const f32,
                    oa as *mut f32,
                    &w,
                    seqs,
                    scale,
                    causal,
                    simd,
                    (lo, hi),
                );
            }
        });
        return;
    }
    units_span(q, k, v, o, w, seqs, scale, causal, simd, (0, units));
}

/// [`attention_kv_into`] with value semantics: an `o` overlapping an input is computed into
/// scratch, and only the elements the call defines (every head of every row) are copied out.
///
/// # Safety
/// As [`attention_kv_into`], except that `o` may overlap `q`, `k` or `v`; `q_extent` / `kv_extent`
/// bound what the call reads of `q` and of `k`/`v`.
#[allow(clippy::too_many_arguments)]
unsafe fn attention_kv_valued(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    o: *mut f32,
    w: &Walk,
    seqs: &[Seq],
    extents: (usize, usize),
    scale: f32,
    causal: bool,
    par: bool,
) {
    let simd = simd_available();
    let (ob, kvb) = (extents.0 * 4, extents.1 * 4);
    let hazard = bytes_overlap(o as usize, ob, q as usize, ob)
        || bytes_overlap(o as usize, ob, k as usize, kvb)
        || bytes_overlap(o as usize, ob, v as usize, kvb);
    if !hazard {
        attention_kv_into(q, k, v, o, w, seqs, scale, causal, simd, par);
        return;
    }
    let mut tmp = vec![0.0f32; extents.0];
    attention_kv_into(q, k, v, tmp.as_mut_ptr(), w, seqs, scale, causal, simd, par);
    for b in 0..w.batch {
        for i in 0..w.sq {
            for h in 0..w.n_heads {
                let off = w.q_off(b, i, h);
                std::ptr::copy_nonoverlapping(tmp.as_ptr().add(off), o.add(off), w.d);
            }
        }
    }
}

/// The shared body of the serial and parallel single-sequence entries.
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
    // `q_pos0` came from a non-negative `i64`, so it converts back losslessly.
    let seqs = [Seq {
        len: plan.skv,
        pos: plan.q_pos0 as i64,
    }];
    let w = Walk::for_plan(&plan);
    let extents = (plan.q_extent, plan.kv_extent);
    attention_kv_valued(q, k, v, o, &w, &seqs, extents, scale, causal, par);
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
    // P14.5: debug-build contract checks (the `# Safety` section above); operand lengths are
    // explicit; an `o` overlapping an input is handled (value semantics).
    crate::safety::debug_extent(q, q_len);
    crate::safety::debug_extent(k, k_len);
    crate::safety::debug_extent(v, v_len);
    crate::safety::debug_extent(o, o_len);
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

/// Multi-threaded [`wukong_attention_kv_f32`]: the (row, KV head, head block) units across cores,
/// bit-identical to the serial entry.
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
    // P14.5: debug-build contract checks (the `# Safety` section above); operand lengths are
    // explicit; an `o` overlapping an input is handled (value semantics).
    crate::safety::debug_extent(q, q_len);
    crate::safety::debug_extent(k, k_len);
    crate::safety::debug_extent(v, v_len);
    crate::safety::debug_extent(o, o_len);
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

/// The operands of a batched call, each with its length (the C ABI order).
#[derive(Clone, Copy)]
struct BatchOperands {
    q: (*const f32, i64),
    k: (*const f32, i64),
    v: (*const f32, i64),
    o: (*mut f32, i64),
    kv_lens: (*const i32, i64),
    q_pos: (*const i32, i64),
}

/// The shared body of the serial and parallel batched entries.
///
/// # Safety
/// Each pointer addresses its length of elements; a zero-length operand is never dereferenced.
unsafe fn attention_kv_batch_entry(
    ops: BatchOperands,
    geo: [i64; 6],
    scale: f32,
    causal: bool,
    par: bool,
) -> i64 {
    let plan = match kv_batch_plan(
        ops.q.1,
        ops.k.1,
        ops.v.1,
        ops.o.1,
        ops.kv_lens.1,
        ops.q_pos.1,
        geo,
    ) {
        Ok(Some(plan)) => plan,
        Ok(None) => return KV_OK,
        Err(code) => return code,
    };
    // Read the sequence descriptions once, before anything is written: an output that overlaps
    // them cannot change them mid-call.
    let lens = std::slice::from_raw_parts(ops.kv_lens.0, plan.batch).to_vec();
    let pos = std::slice::from_raw_parts(ops.q_pos.0, plan.batch).to_vec();
    if let Err(code) = kv_batch_check_lens(&plan, &lens) {
        return code;
    }
    let seqs: Vec<Seq> = lens
        .iter()
        .zip(&pos)
        .map(|(&l, &p)| Seq::new(l, p))
        .collect();
    let w = Walk::for_batch(&plan);
    let extents = (plan.q_extent, plan.kv_extent);
    let (q, k, v, o) = (ops.q.0, ops.k.0, ops.v.0, ops.o.0);
    attention_kv_valued(q, k, v, o, &w, &seqs, extents, scale, causal, par);
    KV_OK
}

/// Batched, padded decode attention — the `sdpa_kv_batch` builtin: `batch` sequences over packed
/// `[batch, sq, n_heads, d]` queries/outputs and `[batch, skv, n_kv_heads, d]` caches, sequence `b`
/// seeing its first `kv_lens[b]` cached keys, its query row `i` at position `q_pos[b] + i` (see
/// the module docs, including the fully masked row). Returns [`KV_OK`], or [`KV_BAD_GEOMETRY`] /
/// [`KV_SHORT_OPERAND`] with nothing written.
///
/// # Safety
/// `q`, `k`, `v` address `q_len`, `k_len`, `v_len` readable f32, `o` `o_len` writable f32, and
/// `kv_lens` / `q_pos` `kv_lens_len` / `q_pos_len` readable i32 (a zero-length operand may be null).
/// An `o` overlapping an input is handled (value semantics).
#[no_mangle]
pub unsafe extern "C" fn wukong_attention_kv_batch_f32(
    q: *const f32,
    q_len: i64,
    k: *const f32,
    k_len: i64,
    v: *const f32,
    v_len: i64,
    o: *mut f32,
    o_len: i64,
    kv_lens: *const i32,
    kv_lens_len: i64,
    q_pos: *const i32,
    q_pos_len: i64,
    batch: i64,
    sq: i64,
    skv: i64,
    n_heads: i64,
    n_kv_heads: i64,
    d: i64,
    scale: f32,
    causal: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); operand lengths are
    // explicit; an `o` overlapping an input is handled (value semantics).
    crate::safety::debug_extent(q, q_len);
    crate::safety::debug_extent(k, k_len);
    crate::safety::debug_extent(v, v_len);
    crate::safety::debug_extent(o, o_len);
    crate::safety::debug_extent(kv_lens, kv_lens_len);
    crate::safety::debug_extent(q_pos, q_pos_len);
    let ops = BatchOperands {
        q: (q, q_len),
        k: (k, k_len),
        v: (v, v_len),
        o: (o, o_len),
        kv_lens: (kv_lens, kv_lens_len),
        q_pos: (q_pos, q_pos_len),
    };
    let geo = [batch, sq, skv, n_heads, n_kv_heads, d];
    attention_kv_batch_entry(ops, geo, scale, causal != 0, false)
}

/// Multi-threaded [`wukong_attention_kv_batch_f32`]: the (sequence, row, KV head, head block) units
/// across cores, bit-identical to the serial entry.
///
/// # Safety
/// The contract of [`wukong_attention_kv_batch_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_attention_kv_batch_f32_parallel(
    q: *const f32,
    q_len: i64,
    k: *const f32,
    k_len: i64,
    v: *const f32,
    v_len: i64,
    o: *mut f32,
    o_len: i64,
    kv_lens: *const i32,
    kv_lens_len: i64,
    q_pos: *const i32,
    q_pos_len: i64,
    batch: i64,
    sq: i64,
    skv: i64,
    n_heads: i64,
    n_kv_heads: i64,
    d: i64,
    scale: f32,
    causal: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); operand lengths are
    // explicit; an `o` overlapping an input is handled (value semantics).
    crate::safety::debug_extent(q, q_len);
    crate::safety::debug_extent(k, k_len);
    crate::safety::debug_extent(v, v_len);
    crate::safety::debug_extent(o, o_len);
    crate::safety::debug_extent(kv_lens, kv_lens_len);
    crate::safety::debug_extent(q_pos, q_pos_len);
    let ops = BatchOperands {
        q: (q, q_len),
        k: (k, k_len),
        v: (v, v_len),
        o: (o, o_len),
        kv_lens: (kv_lens, kv_lens_len),
        q_pos: (q_pos, q_pos_len),
    };
    let geo = [batch, sq, skv, n_heads, n_kv_heads, d];
    attention_kv_batch_entry(ops, geo, scale, causal != 0, true)
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(cache, cache_len);
    crate::safety::debug_extent(row, row_len);
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
    /// cache, the 8-wide `d` tail, the 64-key block boundary, padded (strided) rows, and GQA ratios
    /// 1, 2, 3, 4, 5, 7, 8, 12 and 16 (one, partial and several head blocks per KV head).
    const GEOS: &[Geo] = &[
        (1, 1, 1, 0, 1, 1, 0, 0),
        (1, 5, 8, 4, 1, 1, 0, 0),
        (1, 70, 16, 69, 4, 2, 0, 0),
        (3, 9, 7, 6, 2, 1, 3, 1),
        (4, 130, 33, 126, 6, 3, 0, 5),
        (2, 64, 64, 62, 2, 2, 0, 0),
        (5, 5, 12, 0, 4, 4, 2, 0),
        (2, 200, 9, 150, 8, 1, 0, 0),
        (1, 130, 64, 129, 32, 8, 0, 0),
        (2, 77, 13, 75, 16, 2, 0, 3),
        (1, 65, 24, 64, 12, 4, 0, 0),
        (3, 40, 5, 37, 10, 2, 1, 0),
        (1, 99, 17, 98, 14, 2, 0, 0),
        (1, 70, 8, 69, 12, 1, 0, 0),
        (2, 66, 40, 64, 16, 1, 4, 0),
        (1, 3, 1, 2, 6, 3, 0, 0),
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

    /// The grouped AVX2 pass and the per-head portable twin perform the same operations in the same
    /// order for every (row, head): compare every output bit through the whole driver, over GQA
    /// ratios 1-16 (one, partial and several head blocks per KV head), every `d % 8` tail, key
    /// counts on both sides of the 64-key block, and wide score spreads that flush most weights to
    /// zero.
    #[test]
    fn grouped_avx2_matches_the_per_head_scalar_twin_bit_for_bit() {
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
                    let bits = |simd| -> Vec<u32> {
                        run_path(&c, causal, simd)
                            .iter()
                            .map(|x| x.to_bits())
                            .collect()
                    };
                    assert_eq!(
                        bits(true),
                        bits(false),
                        "{g:?} spread={spread} causal={causal}"
                    );
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
    /// the two kernels block differently, so they agree to twice the band each holds to the naive
    /// reference.
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
                        (a[e] - b[e]).abs() <= 2e-4 + 2e-4 * a[e].abs(),
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

    /// `c` through the internal driver on the chosen path (`simd` = the grouped AVX2 pass, else the
    /// per-head portable twin), serially, into an output pre-filled with a sentinel.
    fn run_path(c: &Case, causal: bool, simd: bool) -> Vec<f32> {
        let p = c.plan;
        let mut o = vec![-7.5f32; p.q_extent];
        let w = Walk::for_plan(&p);
        let seqs = [Seq {
            len: p.skv,
            pos: p.q_pos0 as i64,
        }];
        // SAFETY: every buffer holds exactly the extent the plan validated; `o` is disjoint.
        unsafe {
            attention_kv_into(
                c.q.as_ptr(),
                c.k.as_ptr(),
                c.v.as_ptr(),
                o.as_mut_ptr(),
                &w,
                &seqs,
                c.scale,
                causal,
                simd,
                false,
            );
        }
        o
    }

    /// One batched geometry: `(batch, sq, skv, n_heads, n_kv_heads, d)`, per-sequence `kv_lens`
    /// and `q_pos`.
    type BatchGeo = ([usize; 6], &'static [i32], &'static [i32]);

    /// Batched decode with an empty sequence, bottom-right chunks over 64-key blocks with GQA 8,
    /// negative positions (fully masked leading rows) next to a top-left prefill, a llama-shaped
    /// GQA-4 step, MHA at `d = 1`, odd tails and ratios 3 and 12.
    const BATCH_GEOS: &[BatchGeo] = &[
        ([3, 1, 5, 4, 2, 8], &[5, 2, 0], &[4, 1, 0]),
        ([2, 3, 70, 8, 1, 7], &[70, 66], &[67, 63]),
        ([2, 4, 9, 6, 2, 13], &[9, 3], &[-2, 0]),
        ([1, 2, 130, 32, 8, 64], &[129], &[127]),
        ([4, 1, 3, 2, 2, 1], &[0, 1, 2, 3], &[-1, 0, 1, 2]),
        ([2, 2, 66, 9, 3, 17], &[65, 66], &[63, 0]),
        ([3, 1, 12, 12, 1, 9], &[12, 7, 1], &[11, 6, 0]),
    ];

    struct BatchCase {
        plan: KvBatchPlan,
        lens: Vec<i32>,
        pos: Vec<i32>,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        scale: f32,
    }

    /// A batched case whose cache padding (rows `j >= kv_lens[b]`) is NaN, so a single masked key
    /// that leaked into a row would poison it.
    fn batch_case(g: BatchGeo, seed: u64) -> BatchCase {
        let ([b, sq, skv, nh, nkv, d], lens, pos) = g;
        let (q_len, kv_len) = (b * sq * nh * d, b * skv * nkv * d);
        let geo = [b, sq, skv, nh, nkv, d].map(|x| x as i64);
        let plan = kv_batch_plan(
            q_len as i64,
            kv_len as i64,
            kv_len as i64,
            q_len as i64,
            b as i64,
            b as i64,
            geo,
        )
        .unwrap()
        .unwrap();
        let mut k = fill(seed + 1, kv_len);
        let mut v = fill(seed + 2, kv_len);
        for s in 0..b {
            for j in lens[s] as usize..skv {
                for gg in 0..nkv {
                    let off = plan.kv_off(s, j, gg);
                    k[off..off + d].fill(f32::NAN);
                    v[off..off + d].fill(f32::NAN);
                }
            }
        }
        BatchCase {
            plan,
            lens: lens.to_vec(),
            pos: pos.to_vec(),
            q: fill(seed, q_len),
            k,
            v,
            scale: 1.0 / (d as f32).sqrt(),
        }
    }

    type KvBatchEntry = unsafe extern "C" fn(
        *const f32,
        i64,
        *const f32,
        i64,
        *const f32,
        i64,
        *mut f32,
        i64,
        *const i32,
        i64,
        *const i32,
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

    /// Call the batched entry on `c` with an output pre-filled with a sentinel; the status too.
    fn run_batch_status(c: &BatchCase, lens: &[i32], causal: bool, par: bool) -> (i64, Vec<f32>) {
        let p = c.plan;
        let mut o = vec![-7.5f32; p.q_extent];
        let entry: KvBatchEntry = if par {
            wukong_attention_kv_batch_f32_parallel
        } else {
            wukong_attention_kv_batch_f32
        };
        // SAFETY: every buffer holds the extent the plan validated (`lens` at least `batch`).
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
                lens.as_ptr(),
                lens.len() as i64,
                c.pos.as_ptr(),
                c.pos.len() as i64,
                p.batch as i64,
                p.sq as i64,
                p.skv as i64,
                p.n_heads as i64,
                p.n_kv_heads as i64,
                p.d as i64,
                c.scale,
                causal as i64,
            )
        };
        (code, o)
    }

    fn run_batch(c: &BatchCase, causal: bool, par: bool) -> Vec<f32> {
        let (code, o) = run_batch_status(c, &c.lens, causal, par);
        assert_eq!(code, KV_OK);
        o
    }

    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter().map(|v| v.to_bits()).collect()
    }

    /// Masking is exact: each batched row is bit-for-bit the single-sequence kernel's row over that
    /// sequence's `kv_lens[b]` keys at position `q_pos[b] + i` (so the NaN padding was never read),
    /// and a row that sees no key is exactly `+0.0` in every lane.
    #[test]
    fn batched_rows_equal_their_unpadded_sequences_bit_for_bit() {
        for (n, &g) in BATCH_GEOS.iter().enumerate() {
            let c = batch_case(g, 13 * n as u64 + 5);
            let p = c.plan;
            let (q_row, kv_row) = (p.n_heads * p.d, p.n_kv_heads * p.d);
            for causal in [false, true] {
                let got = run_batch(&c, causal, false);
                for b in 0..p.batch {
                    let len = c.lens[b] as usize;
                    for i in 0..p.sq {
                        let row = &got[p.q_off(b, i, 0)..p.q_off(b, i, 0) + q_row];
                        let keys = KvBatchPlan::keys(c.lens[b], c.pos[b], i, causal);
                        if keys == 0 {
                            assert!(
                                row.iter().all(|x| x.to_bits() == 0),
                                "{g:?} causal={causal} [{b}][{i}]: a fully masked row is +0.0"
                            );
                            continue;
                        }
                        // The same row alone: one query against the sequence's own keys.
                        let q_pos0 = if causal {
                            (i64::from(c.pos[b]) + i as i64).max(0)
                        } else {
                            0
                        };
                        let mut want = vec![0.0f32; q_row];
                        let kv = p.kv_off(b, 0, 0);
                        // SAFETY: the row and the sequence's `len` cached rows lie inside the case.
                        let code = unsafe {
                            wukong_attention_kv_f32(
                                c.q.as_ptr().add(p.q_off(b, i, 0)),
                                q_row as i64,
                                c.k.as_ptr().add(kv),
                                (len * kv_row) as i64,
                                c.v.as_ptr().add(kv),
                                (len * kv_row) as i64,
                                want.as_mut_ptr(),
                                q_row as i64,
                                1,
                                len as i64,
                                p.d as i64,
                                q_pos0,
                                p.n_heads as i64,
                                p.n_kv_heads as i64,
                                q_row as i64,
                                kv_row as i64,
                                c.scale,
                                causal as i64,
                            )
                        };
                        assert_eq!(code, KV_OK);
                        assert_eq!(bits(row), bits(&want), "{g:?} causal={causal} [{b}][{i}]");
                    }
                }
            }
        }
    }

    /// The batched kernel against a naive f64 reference that materializes each row's visible
    /// scores (and never looks at the padding).
    #[test]
    fn batched_attention_matches_a_naive_masked_reference() {
        for (n, &g) in BATCH_GEOS.iter().enumerate() {
            let c = batch_case(g, 3 * n as u64 + 11);
            let p = c.plan;
            for causal in [false, true] {
                let got = run_batch(&c, causal, false);
                for b in 0..p.batch {
                    for i in 0..p.sq {
                        let keys = KvBatchPlan::keys(c.lens[b], c.pos[b], i, causal);
                        for h in 0..p.n_heads {
                            let gh = p.kv_head(h);
                            let qo = p.q_off(b, i, h);
                            let s: Vec<f64> = (0..keys)
                                .map(|j| {
                                    let ko = p.kv_off(b, j, gh);
                                    let dot: f64 = (0..p.d)
                                        .map(|t| c.q[qo + t] as f64 * c.k[ko + t] as f64)
                                        .sum();
                                    dot * c.scale as f64
                                })
                                .collect();
                            let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let w: Vec<f64> = s.iter().map(|&x| (x - mx).exp()).collect();
                            let sum: f64 = w.iter().sum();
                            for t in 0..p.d {
                                let want = if keys == 0 {
                                    0.0
                                } else {
                                    w.iter()
                                        .enumerate()
                                        .map(|(j, &wj)| wj * c.v[p.kv_off(b, j, gh) + t] as f64)
                                        .sum::<f64>()
                                        / sum
                                };
                                let gv = got[qo + t] as f64;
                                assert!(
                                    (gv - want).abs() <= 1e-4 + 1e-4 * want.abs(),
                                    "{g:?} causal={causal} [{b}][{i}][{h}][{t}]: got {gv} want \
                                     {want}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn batched_parallel_equals_serial_bit_for_bit() {
        for (n, &g) in BATCH_GEOS.iter().enumerate() {
            let c = batch_case(g, 17 * n as u64 + 1);
            for causal in [false, true] {
                assert_eq!(
                    bits(&run_batch(&c, causal, false)),
                    bits(&run_batch(&c, causal, true)),
                    "{g:?} causal={causal}"
                );
            }
        }
    }

    /// The batched entry rides the same grouped pass: forcing the per-head twin through the batched
    /// walk changes no bit.
    #[test]
    fn batched_grouped_pass_matches_the_per_head_twin_bit_for_bit() {
        if !simd_available() {
            return;
        }
        for (n, &g) in BATCH_GEOS.iter().enumerate() {
            let c = batch_case(g, 29 * n as u64 + 7);
            let p = c.plan;
            let w = Walk::for_batch(&p);
            let seqs: Vec<Seq> = c
                .lens
                .iter()
                .zip(&c.pos)
                .map(|(&l, &q)| Seq::new(l, q))
                .collect();
            for causal in [false, true] {
                let path = |simd| {
                    let mut o = vec![-7.5f32; p.q_extent];
                    // SAFETY: packed buffers sized by the plan; `o` is disjoint.
                    unsafe {
                        attention_kv_into(
                            c.q.as_ptr(),
                            c.k.as_ptr(),
                            c.v.as_ptr(),
                            o.as_mut_ptr(),
                            &w,
                            &seqs,
                            c.scale,
                            causal,
                            simd,
                            false,
                        );
                    }
                    bits(&o)
                };
                assert_eq!(path(true), path(false), "{g:?} causal={causal}");
            }
        }
    }

    /// An output aliasing the batch's queries is computed from the queries as they were.
    #[test]
    fn a_batched_output_aliasing_the_query_has_value_semantics() {
        let c = batch_case(BATCH_GEOS[2], 91);
        let want = run_batch(&c, true, false);
        let p = c.plan;
        let mut buf = c.q.clone();
        let ptr = buf.as_mut_ptr();
        // SAFETY: `buf` holds q's extent; the entry handles the overlap.
        let code = unsafe {
            wukong_attention_kv_batch_f32(
                ptr,
                buf.len() as i64,
                c.k.as_ptr(),
                c.k.len() as i64,
                c.v.as_ptr(),
                c.v.len() as i64,
                ptr,
                buf.len() as i64,
                c.lens.as_ptr(),
                c.lens.len() as i64,
                c.pos.as_ptr(),
                c.pos.len() as i64,
                p.batch as i64,
                p.sq as i64,
                p.skv as i64,
                p.n_heads as i64,
                p.n_kv_heads as i64,
                p.d as i64,
                c.scale,
                1,
            )
        };
        assert_eq!(code, KV_OK);
        assert_eq!(bits(&buf), bits(&want));
    }

    #[test]
    fn malformed_batches_are_refused_before_anything_is_written() {
        // [batch, sq, skv, n_heads, n_kv_heads, d]
        let ok = [2i64, 1, 4, 4, 2, 8];
        let (ql, kvl) = (2 * 4 * 8, 2 * 4 * 2 * 8);
        assert!(matches!(
            kv_batch_plan(ql, kvl, kvl, ql, 2, 2, ok),
            Ok(Some(_))
        ));
        // Nothing to compute, whatever the operands hold.
        assert_eq!(
            kv_batch_plan(0, 0, 0, 0, 0, 0, [0, 1, 4, 4, 2, 8]),
            Ok(None)
        );
        let bad = |g: [i64; 6]| kv_batch_plan(1 << 20, 1 << 20, 1 << 20, 1 << 20, 64, 64, g);
        assert_eq!(bad([2, 1, 4, 4, 3, 8]), Err(KV_BAD_GEOMETRY));
        assert_eq!(bad([2, 1, 4, 4, 0, 8]), Err(KV_BAD_GEOMETRY));
        assert_eq!(bad([2, 1, -4, 4, 2, 8]), Err(KV_BAD_GEOMETRY));
        assert_eq!(
            bad([i64::MAX, i64::MAX, i64::MAX, 4, 2, i64::MAX]),
            Err(KV_SHORT_OPERAND)
        );
        // Each operand one element short, and each sequence vector one entry short.
        let short = |l: [i64; 6]| kv_batch_plan(l[0], l[1], l[2], l[3], l[4], l[5], ok);
        assert_eq!(short([ql - 1, kvl, kvl, ql, 2, 2]), Err(KV_SHORT_OPERAND));
        assert_eq!(short([ql, kvl - 1, kvl, ql, 2, 2]), Err(KV_SHORT_OPERAND));
        assert_eq!(short([ql, kvl, kvl - 1, ql, 2, 2]), Err(KV_SHORT_OPERAND));
        assert_eq!(short([ql, kvl, kvl, ql - 1, 2, 2]), Err(KV_SHORT_OPERAND));
        assert_eq!(short([ql, kvl, kvl, ql, 1, 2]), Err(KV_SHORT_OPERAND));
        assert_eq!(short([ql, kvl, kvl, ql, 2, 1]), Err(KV_SHORT_OPERAND));
        // A sequence length outside `0..=skv` is refused and writes nothing.
        let c = batch_case(BATCH_GEOS[0], 5);
        for lens in [[5, 6, 0], [5, 2, -1]] {
            let (code, o) = run_batch_status(&c, &lens, true, false);
            assert_eq!(code, KV_BAD_GEOMETRY, "{lens:?}");
            assert!(o.iter().all(|&x| x == -7.5), "{lens:?}: written");
        }
        let plan = c.plan;
        assert_eq!(kv_batch_check_lens(&plan, &[5, 5, 5]), Ok(()));
        assert_eq!(kv_batch_check_lens(&plan, &[5, 5]), Err(KV_BAD_GEOMETRY));
        // An empty cache: every row is fully masked.
        let (q, mut o) = ([0.5f32; 8], [9.0f32; 8]);
        let (lens, pos) = ([0i32], [0i32]);
        // SAFETY: the cache is empty, so `k`/`v` are never read.
        let code = unsafe {
            wukong_attention_kv_batch_f32(
                q.as_ptr(),
                8,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                o.as_mut_ptr(),
                8,
                lens.as_ptr(),
                1,
                pos.as_ptr(),
                1,
                1,
                1,
                0,
                1,
                1,
                8,
                1.0,
                1,
            )
        };
        assert_eq!(code, KV_OK);
        assert_eq!(bits(&o), [0u32; 8]);
    }

    /// The GQA-blocking A/B instrument, not a gate: one decode step (`sq = 1`, `d = 128`, a long
    /// cache) through the grouped AVX2 pass versus the same code one head at a time (blocks of one,
    /// the per-head schedule every head used before), single-threaded, alternating the two arms and
    /// keeping each arm's best of 15 — report the ratio only, never an absolute time, and read it on
    /// a release build with the power state noted:
    /// `cargo test -p wukong_runtime --release --lib gqa_group_blocking_ab -- --ignored --nocapture`.
    /// Both arms must also agree bit for bit.
    #[test]
    #[ignore]
    fn gqa_group_blocking_ab() {
        if !simd_available() {
            return;
        }
        let d = 128usize;
        for &(nh, nkv, skv) in &[
            (32usize, 32usize, 4096usize),
            (32, 8, 4096),
            (32, 4, 4096),
            (28, 4, 4096),
            (32, 1, 4096),
        ] {
            let group = nh / nkv;
            let (q_row, kv_row) = (nh * d, nkv * d);
            let q = fill(1, q_row);
            let k = fill(2, skv * kv_row);
            let v = fill(3, skv * kv_row);
            let scale = 1.0 / (d as f32).sqrt();
            let mut acc = vec![0.0f32; KV_GROUP_BLOCK * d];
            let mut sc = vec![0.0f32; KV_GROUP_BLOCK * KV_BC];
            let kv = |g: usize| KvRow {
                k: k.as_ptr().wrapping_add(g * d),
                v: v.as_ptr().wrapping_add(g * d),
                stride: kv_row,
                keys: skv,
            };
            // `block` heads per call: the grouped schedule (8) or one head at a time (1).
            let mut run = |block: usize, o: &mut [f32]| {
                let t0 = std::time::Instant::now();
                for g in 0..nkv {
                    let mut h = 0;
                    while h < group {
                        let nb = (group - h).min(block);
                        let off = (g * group + h) * d;
                        // SAFETY: every pointer stays inside the buffers sized above.
                        unsafe {
                            attend_group_simd(
                                q.as_ptr().add(off),
                                nb,
                                kv(g),
                                d,
                                scale,
                                o.as_mut_ptr().add(off),
                                &mut acc,
                                &mut sc,
                            );
                        }
                        h += nb;
                    }
                }
                t0.elapsed()
            };
            let (mut og, mut oh) = (vec![0.0f32; q_row], vec![0.0f32; q_row]);
            let (mut best_g, mut best_h) = (std::time::Duration::MAX, std::time::Duration::MAX);
            for _ in 0..15 {
                best_g = best_g.min(run(KV_GROUP_BLOCK, &mut og));
                best_h = best_h.min(run(1, &mut oh));
            }
            assert_eq!(bits(&og), bits(&oh), "H={nh} HKV={nkv}");
            println!(
                "decode step H={nh} HKV={nkv} (G={group}) skv={skv} d={d}: per-head / grouped = \
                 {:.2}x",
                best_h.as_secs_f64() / best_g.as_secs_f64()
            );
        }
    }
}
