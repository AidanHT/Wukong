//! Single-precision reductions — dot product and friends, with a **deterministic multicore** variant.
//!
//! Computes `reduce_i f(x[i], y[i])` for `f` in `{ x·y (dot), (x−y)² (ssd), x (sum), x·x (sumsq),
//! |x| (abssum / L1 norm), |x−y| (absdiff / MAE) }` folded by `+`, and `{ x (max), x (min), |x|
//! (maxabs) }` folded by `fmax`/`fmin`. These are the reductions transformer math leans on: attention
//! scores and projections (dot), the L2 loss (ssd), LayerNorm/RMSNorm mean & variance (sum, sumsq),
//! the **L1 norm / mean-absolute-error** (abssum, absdiff), and the per-tensor **max/absmax** that
//! softmax stability and dynamic int8 quantization scale-computation need (max, min). Those nine ride
//! [`wukong_sreduce_f32`] (`-> f32`); the two **arg**-reductions `RED_ARGMAX`/`RED_ARGMIN` return an
//! *index* and so ride a separate `-> i64` entry, [`wukong_argreduce_f32`], with `-1` reserved for
//! `n <= 0`. The compiler recognizes
//! the reduction loop in a `@parallel` function and lowers it to one of these calls — the same play as
//! the matmul→GEMM and activation→`wukong_vmath_f32` dispatch. The interpreter marshals its abstract
//! memory through the **identical serial kernel**, so the differential oracle stays bit-for-bit exact.
//!
//! **Determinism is the whole game for a *parallel* reduction** — the result must not depend on how
//! many cores ran it. So the array is cut into FIXED-size chunks (count independent of thread count);
//! each chunk is reduced to a partial by the identical [`reduce_chunk`]; the partials are folded in
//! ascending chunk order. The serial and parallel entries call the same per-chunk function and combine
//! in the same order, so `serial == parallel == interpreter`, bit for bit, on any machine (the arg
//! entries do the same with [`argreduce_chunk`] and [`arg_fold`], whose lowest-index-wins tie-break is
//! what makes *that* fold decomposition-independent; and both `_parallel` entries just call their
//! serial sibling when there is one chunk or `wuk_pool_width() <= 1`). (`fmax`/
//! `fmin` are *not* associative on NaN/±0, but determinism here rests on the FIXED decomposition and
//! ascending combine, not on associativity — the serial and parallel forms evaluate the identical
//! expression tree.) Within a chunk the AVX2 path and the scalar twin are also bit-identical (lane `j`
//! folds elements `≡ j (mod 8)`, then a fixed-order horizontal combine; `f32::mul_add` is the same
//! fused op as `_mm256_fmadd`, and the scalar `(a > b) ? a : b` is the same as `_mm256_max_ps`).
//!
//! At `N = 2^20` a dot is *memory-bound* (a single core already saturates load bandwidth at ~44
//! GB/s), so one 8-lane accumulator per chunk is enough to be bandwidth-bound; the win is spreading
//! chunks across cores to reach *aggregate* bandwidth — exactly how `saxpy@parallel` hits ~130 GB/s.

use rayon::prelude::*;

// Reduction op codes (shared with the recognizer in `wukong_mir_build`). For the unary ops (`SUM`,
// `SUMSQ`) the recognizer passes `y == x` so `y` is always a valid pointer and the kernel simply
// never reads it — no null-pointer marshalling anywhere.
pub const RED_DOT: i64 = 0; // sum(x[i] * y[i])
pub const RED_SSD: i64 = 1; // sum((x[i] - y[i])^2)
pub const RED_SUM: i64 = 2; // sum(x[i])
pub const RED_SUMSQ: i64 = 3; // sum(x[i] * x[i])
pub const RED_MAX: i64 = 4; // max(x[i])  — fold by fmax
pub const RED_MIN: i64 = 5; // min(x[i])  — fold by fmin
pub const RED_MAXABS: i64 = 6; // max(|x[i]|) — abs each element, fold by fmax (symmetric int8 quant)

// Arg-reductions: return an *index* (not a value), so they ride a separate `-> i64` ABI
// (`wukong_argreduce_f32`), not the `-> f32` `wukong_sreduce_f32`. Lowest index wins on ties.
pub const RED_ARGMAX: i64 = 7; // argmax_i x[i] — greedy decode / classification top-1
pub const RED_ARGMIN: i64 = 8; // argmin_i x[i]

// More `-> f32` reductions on `wukong_sreduce_f32` (numbered after the arg codes 7/8 so those keep
// their values — the recognizer and GPU paths pin RED_ARGMAX/MIN by number). Both are additive folds
// like `RED_SUM`, so `ident`/`fold2`/`hcombine8` need no new arm; only `contrib` + the AVX2 loop do.
pub const RED_SUMABS: i64 = 9; // sum(|x[i]|) — L1 norm / abssum (unary; recognizer passes y == x)
pub const RED_ABSDIFF: i64 = 10; // sum(|x[i] - y[i]|) — MAE / SAD numerator (binary, like RED_SSD)

/// The fold identity: `0.0` for the additive ops, `∓∞` for max/min/maxabs so the first real element
/// wins (`maxabs` folds by max, identity `−∞`).
#[inline(always)]
pub(crate) fn ident(op: i64) -> f32 {
    match op {
        RED_MAX | RED_MAXABS => f32::NEG_INFINITY,
        RED_MIN => f32::INFINITY,
        _ => 0.0,
    }
}

/// Fold two partials under the reduction's combine: `a + b` (additive), or `(a > b) ? a : b` /
/// `(a < b) ? a : b` for max/min — the exact semantics of `_mm256_max_ps`/`_mm256_min_ps` (and of
/// the MIR `Cmp(Fogt/Folt)+Select` the recognizer emits to combine the kernel result), so the AVX2
/// lanes, the scalar twin, and the compiler's outer fold all agree bit-for-bit.
#[inline(always)]
pub(crate) fn fold2(a: f32, b: f32, op: i64) -> f32 {
    match op {
        // maxabs folds its (already abs'd) partials by plain max.
        RED_MAX | RED_MAXABS => {
            if a > b {
                a
            } else {
                b
            }
        }
        RED_MIN => {
            if a < b {
                a
            } else {
                b
            }
        }
        _ => a + b,
    }
}

// Fixed chunk size in elements — independent of thread count, which is what makes the parallel
// partial decomposition deterministic. 8192 f32 = 32 KB (an L1's worth) per chunk; at N=2^20 that is
// 128 chunks, plenty for rayon work-stealing to balance the P+E hybrid, and the 128-entry partial
// array combines in negligible time. Must stay constant for serial/parallel/interp to agree.
// `pub(crate)` so the bf16/f16 parallel reductions in `lowp.rs` cut on the *same* fixed boundary.
pub(crate) const RCHUNK: usize = 8192;

/// One element's contribution folded into accumulator `a` (the fused-multiply-add form, so the AVX2
/// `_mm256_fmadd_ps` lanes match this lane-for-lane). `yi` is ignored for the unary ops.
#[inline(always)]
fn contrib(a: f32, xi: f32, yi: f32, op: i64) -> f32 {
    match op {
        RED_DOT => xi.mul_add(yi, a),
        RED_SSD => {
            let d = xi - yi;
            d.mul_add(d, a)
        }
        RED_SUM => a + xi,
        RED_SUMSQ => xi.mul_add(xi, a),
        RED_MAX | RED_MIN => fold2(a, xi, op), // `(a > xi) ? a : xi` ≡ `_mm256_max_ps(a, xi)`
        RED_MAXABS => fold2(a, xi.abs(), RED_MAX), // `f32::abs` clears the sign bit ≡ `andnot(-0, xi)`
        RED_SUMABS => a + xi.abs(), // Σ|x|: same sign-bit clear, folded by + (mirrors RED_SUM)
        RED_ABSDIFF => {
            let d = xi - yi; // same subtract as RED_SSD, then |d| instead of d²
            a + d.abs()
        }
        _ => a,
    }
}

/// Fixed-order horizontal combine of the 8 lane accumulators: a balanced tree built from [`fold2`],
/// identical in the AVX2 and scalar paths so both produce the same bits. For the additive ops this is
/// exactly `((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))` (unchanged); for max/min it is the same tree under
/// `fmax`/`fmin`.
#[inline(always)]
pub(crate) fn hcombine8(a: [f32; 8], op: i64) -> f32 {
    fold2(
        fold2(fold2(a[0], a[1], op), fold2(a[2], a[3], op), op),
        fold2(fold2(a[4], a[5], op), fold2(a[6], a[7], op), op),
        op,
    )
}

/// Reduce `x[lo..hi]` (and `y[lo..hi]` for the binary ops) to a scalar partial. Pure function of its
/// chunk, so it returns the same bits no matter which thread (or the serial loop) calls it.
///
/// # Safety
/// `x` and `y` must be valid for reads on `[lo, hi)`.
#[inline]
unsafe fn reduce_chunk(x: *const f32, y: *const f32, lo: usize, hi: usize, op: i64) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; range validity is the caller's contract.
            return unsafe { reduce_chunk_avx2(x, y, lo, hi, op) };
        }
    }
    // SAFETY: range validity is the caller's contract.
    unsafe { reduce_chunk_scalar(x, y, lo, hi, op) }
}

/// Portable reference: 8 scalar lane accumulators mirroring the AVX2 register lane-for-lane. Used on
/// non-AVX2 targets; the unit tests pin it bit-identical to the AVX2 path where that is available.
///
/// # Safety
/// `x`/`y` valid on `[lo, hi)`.
unsafe fn reduce_chunk_scalar(x: *const f32, y: *const f32, lo: usize, hi: usize, op: i64) -> f32 {
    let mut acc = [ident(op); 8];
    let len = hi - lo;
    let nsteps = len / 8;
    for s in 0..nsteps {
        let base = lo + s * 8;
        for (j, a) in acc.iter_mut().enumerate() {
            let i = base + j;
            // SAFETY: i < hi.
            *a = contrib(*a, unsafe { *x.add(i) }, unsafe { *y.add(i) }, op);
        }
    }
    // Tail (< 8 elements) folds into lanes 0.. in order — identical to the AVX2 store-then-tail.
    let tail = lo + nsteps * 8;
    for (j, a) in acc.iter_mut().enumerate().take(len - nsteps * 8) {
        let i = tail + j;
        // SAFETY: i < hi.
        *a = contrib(*a, unsafe { *x.add(i) }, unsafe { *y.add(i) }, op);
    }
    hcombine8(acc, op)
}

/// AVX2 chunk reduce: one `__m256` accumulator (lane `j` sums elements `≡ j (mod 8)` over the chunk),
/// then the tail and horizontal combine in scalar so the result is bit-identical to the scalar twin.
///
/// # Safety
/// `x`/`y` valid on `[lo, hi)`; AVX2+FMA must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn reduce_chunk_avx2(x: *const f32, y: *const f32, lo: usize, hi: usize, op: i64) -> f32 {
    use std::arch::x86_64::*;
    // `set1(ident)` is `setzero` for the additive ops (+0.0 = all-zero bits), and ∓∞ for max/min.
    let mut acc = _mm256_set1_ps(ident(op));
    let len = hi - lo;
    let nsteps = len / 8;
    for s in 0..nsteps {
        let i = lo + s * 8;
        let xv = _mm256_loadu_ps(x.add(i));
        acc = match op {
            RED_DOT => _mm256_fmadd_ps(xv, _mm256_loadu_ps(y.add(i)), acc),
            RED_SSD => {
                let d = _mm256_sub_ps(xv, _mm256_loadu_ps(y.add(i)));
                _mm256_fmadd_ps(d, d, acc)
            }
            RED_SUM => _mm256_add_ps(xv, acc),
            RED_SUMSQ => _mm256_fmadd_ps(xv, xv, acc),
            RED_MAX => _mm256_max_ps(acc, xv), // `(acc > xv) ? acc : xv`, lane-wise
            RED_MIN => _mm256_min_ps(acc, xv),
            // |xv| via `andnot(-0.0, xv)` (clear the sign bit) ≡ the scalar `f32::abs`, then max.
            RED_MAXABS => _mm256_max_ps(acc, _mm256_andnot_ps(_mm256_set1_ps(-0.0), xv)),
            // Σ|x|: clear the sign bit (andnot -0.0) then add — bit-identical to `contrib`'s xi.abs().
            RED_SUMABS => _mm256_add_ps(acc, _mm256_andnot_ps(_mm256_set1_ps(-0.0), xv)),
            // Σ|x−y|: subtract (as RED_SSD) then clear the sign bit, then add — matches `contrib`.
            RED_ABSDIFF => {
                let d = _mm256_sub_ps(xv, _mm256_loadu_ps(y.add(i)));
                _mm256_add_ps(acc, _mm256_andnot_ps(_mm256_set1_ps(-0.0), d))
            }
            _ => acc,
        };
    }
    let mut tmp = [ident(op); 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    // Scalar tail into the same lanes — `contrib` matches the AVX2 lane op exactly (mul_add↔fmadd,
    // `(a > xi) ? a : xi` ↔ max_ps).
    let tail = lo + nsteps * 8;
    for (j, t) in tmp.iter_mut().enumerate().take(len - nsteps * 8) {
        let i = tail + j;
        *t = contrib(*t, *x.add(i), *y.add(i), op);
    }
    hcombine8(tmp, op)
}

/// `sum_i f(x[i], y[i])` for `i in 0..n` (serial). For `RED_SUM`/`RED_SUMSQ`, `y` is ignored (callers
/// pass `y == x`). The chunk decomposition matches [`wukong_sreduce_f32_parallel`] exactly.
///
/// # Safety
/// `x` and `y` must each be valid for `n` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn wukong_sreduce_f32(x: *const f32, y: *const f32, n: i64, op: i64) -> f32 {
    if n <= 0 {
        return ident(op);
    }
    let n = n as usize;
    let nchunks = n.div_ceil(RCHUNK);
    let mut acc = ident(op);
    for c in 0..nchunks {
        let lo = c * RCHUNK;
        let hi = ((c + 1) * RCHUNK).min(n);
        // SAFETY: [lo, hi) ⊆ [0, n); buffers valid for n by contract.
        acc = fold2(acc, unsafe { reduce_chunk(x, y, lo, hi, op) }, op);
    }
    acc
}

/// Multicore `sum_i f(x[i], y[i])` — **bit-identical** to [`wukong_sreduce_f32`]. Each fixed chunk is
/// reduced independently (rayon), the partials are collected *in index order* (rayon's indexed
/// `map().collect()` preserves order regardless of which thread computed each), then summed ascending.
/// So the float result is the same whether one core or sixteen ran it, and the interpreter (which
/// calls the serial form) agrees with the native `@parallel` path that calls this one.
///
/// # Safety
/// `x` and `y` must each be valid for `n` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn wukong_sreduce_f32_parallel(
    x: *const f32,
    y: *const f32,
    n: i64,
    op: i64,
) -> f32 {
    if n <= 0 {
        return ident(op);
    }
    let n = n as usize;
    let nchunks = n.div_ceil(RCHUNK);
    // One chunk, or a width-1 pool (RAYON_NUM_THREADS=1): nothing to parallelize — run the serial
    // sibling (bit-identical: fixed chunks + ordered fold).
    if nchunks < 2 || crate::wuk_pool_width() <= 1 {
        // SAFETY: same contract.
        return unsafe { wukong_sreduce_f32(x, y, n as i64, op) };
    }
    // Raw pointers cross the rayon closure boundary as integers (same pattern as the parallel GEMM);
    // every access is a disjoint read-only chunk. Forks on the unified kernel pool
    // (`run_on_wuk_pool`) — the fixed-`RCHUNK` decomposition and the ordered serial fold below are
    // what make the result thread-count- and pool-independent.
    let (xa, ya) = (x as usize, y as usize);
    let partials: Vec<f32> = crate::run_on_wuk_pool(move || {
        (0..nchunks)
            .into_par_iter()
            .map(|c| {
                let lo = c * RCHUNK;
                let hi = ((c + 1) * RCHUNK).min(n);
                // SAFETY: disjoint read-only chunk; pointers valid for n by contract.
                unsafe { reduce_chunk(xa as *const f32, ya as *const f32, lo, hi, op) }
            })
            .collect()
    });
    let mut acc = ident(op);
    for p in partials {
        acc = fold2(acc, p, op);
    }
    acc
}

// --- arg-reductions (argmax / argmin) -------------------------------------------------------------
//
// These return the INDEX of the extreme element, with **lowest index winning on a value tie** — the
// tie-break that makes the fold associative over any chunk decomposition, so the fixed-`RCHUNK`
// serial form, the multicore form, and the interpreter all return the same index regardless of how
// many chunks/threads ran (the same determinism contract as `wukong_sreduce_f32`). NaN never
// displaces the running candidate (strict `>`/`<`), matching the value reductions' NaN behavior.

/// Combine two `(value, index)` candidates. A strictly-better value wins; on an exact value tie the
/// LOWER index wins; else keep `a`. `is_max`: argmax (larger wins) vs argmin (smaller). Strict compare
/// + explicit lower-index tie = a total order on the (unique-index) pairs, so the fold is associative
/// and serial == parallel == interp.
#[inline(always)]
pub(crate) fn arg_fold(a: (f32, usize), b: (f32, usize), is_max: bool) -> (f32, usize) {
    let b_better = if is_max { b.0 > a.0 } else { b.0 < a.0 };
    if b_better {
        b
    } else if a.0 == b.0 && b.1 < a.1 {
        b
    } else {
        a
    }
}

/// Reduce `x[lo..hi]` to its `(extreme value, index)` under argmax/argmin. Pure function of the chunk
/// (ascending scan + lowest-index tie-break), so every thread or the serial loop returns the same pair.
/// Dispatches to a 256-bit AVX2 path when available (the `(value, index)` bookkeeping gcc/rustc won't
/// auto-vectorize — but a hand-written `cmp + blendv` sweep can), falling back to the scalar twin below
/// on non-AVX2 targets. Both return the **same** `(extreme, lowest-index)` pair, so the differential
/// oracle stays bit-exact (the SIMD candidates collapse through the very same scalar [`arg_fold`]).
///
/// # Safety
/// `x` valid for reads on `[lo, hi)`.
#[inline]
unsafe fn argreduce_chunk(x: *const f32, lo: usize, hi: usize, is_max: bool) -> (f32, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        // The kernel tracks indices in i32 SIMD lanes; fall back to the scalar twin if a chunk index
        // could exceed i32 (it never does for realistic logit/vocab tensors — RCHUNK·nchunks ≤ n).
        if is_x86_feature_detected!("avx2") && hi <= i32::MAX as usize {
            use std::arch::x86_64::{_CMP_GT_OQ, _CMP_LT_OQ};
            // SAFETY: feature detected; range validity is the caller's contract.
            return if is_max {
                unsafe { argreduce_chunk_avx2::<_CMP_GT_OQ>(x, lo, hi, f32::NEG_INFINITY, true) }
            } else {
                unsafe { argreduce_chunk_avx2::<_CMP_LT_OQ>(x, lo, hi, f32::INFINITY, false) }
            };
        }
    }
    // SAFETY: range validity is the caller's contract.
    unsafe { argreduce_chunk_scalar(x, lo, hi, is_max) }
}

/// Portable reference: 8 scalar lane candidates (ILP) collapsed ascending. The AVX2 path is pinned
/// bit-identical to this by the unit tests; this also backs non-AVX2 targets.
///
/// # Safety
/// `x` valid for reads on `[lo, hi)`.
unsafe fn argreduce_chunk_scalar(x: *const f32, lo: usize, hi: usize, is_max: bool) -> (f32, usize) {
    let ident_v = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let mut acc = [(ident_v, usize::MAX); 8];
    let len = hi - lo;
    let nsteps = len / 8;
    for s in 0..nsteps {
        let base = lo + s * 8;
        for (j, a) in acc.iter_mut().enumerate() {
            let i = base + j;
            *a = arg_fold(*a, (*x.add(i), i), is_max);
        }
    }
    let tail = lo + nsteps * 8;
    for (j, a) in acc.iter_mut().enumerate().take(len - nsteps * 8) {
        let i = tail + j;
        *a = arg_fold(*a, (*x.add(i), i), is_max);
    }
    // Fixed-order ascending lane collapse (0→7): on an all-equal chunk lane 0 holds the lowest index,
    // and ascending `arg_fold` keeps it — so the lowest-index rule holds through the horizontal combine.
    let mut r = acc[0];
    for a in &acc[1..] {
        r = arg_fold(r, *a, is_max);
    }
    r
}

/// AVX2 chunk arg-reduce: **4 independent accumulators** (4×8 = 32 elements/iteration) each tracking 8
/// `f32` extreme-value lanes + 8 `i32` index lanes. Per element it does one `_mm256_cmp_ps::<PRED>`
/// (strict `>` for argmax / `<` for argmin — ordered, non-signalling, so NaN never displaces, matching
/// the scalar twin) and two `blendv` (value + index) — the bookkeeping gcc/rustc leave scalar. The
/// strict compare gives the lowest-index tie-break **for free** (a tie never fires the update, so each
/// lane keeps its earlier index). The 32 lane candidates then collapse through the *same* scalar
/// [`arg_fold`] (a total order on unique indices, so order-independent), and a `< 32` scalar tail folds
/// in after — so the result is bit-identical to [`argreduce_chunk_scalar`], with the one span the
/// strict compare cannot express (a lane that saw only NaN/identity values) delegated to the twin
/// outright, see below. Four accumulators hide the `cmp→blendv` latency while staying inside 16 ymm
/// registers.
///
/// # Safety
/// `x` valid for reads on `[lo, hi)`; AVX2 available; `hi ≤ i32::MAX` (indices ride i32 lanes).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn argreduce_chunk_avx2<const PRED: i32>(
    x: *const f32,
    lo: usize,
    hi: usize,
    ident_v: f32,
    is_max: bool,
) -> (f32, usize) {
    use std::arch::x86_64::*;
    let len = hi - lo;
    let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut vext = [_mm256_set1_ps(ident_v); 4]; // running extreme value per lane
    let mut vidx = [_mm256_set1_epi32(-1); 4]; // running index per lane (-1 = unconsumed sentinel)
    // Absolute index of lane 0 of each accumulator at the current step.
    let mut ibase = [
        _mm256_add_epi32(_mm256_set1_epi32(lo as i32), lane),
        _mm256_add_epi32(_mm256_set1_epi32((lo + 8) as i32), lane),
        _mm256_add_epi32(_mm256_set1_epi32((lo + 16) as i32), lane),
        _mm256_add_epi32(_mm256_set1_epi32((lo + 24) as i32), lane),
    ];
    let bump = _mm256_set1_epi32(32);
    let nsteps = len / 32;
    for s in 0..nsteps {
        let base = lo + s * 32;
        for a in 0..4 {
            let xv = _mm256_loadu_ps(x.add(base + a * 8));
            // mask = all-ones where xv strictly beats the running extreme (NaN -> 0, never updates).
            let mask = _mm256_cmp_ps::<PRED>(xv, vext[a]);
            vext[a] = _mm256_blendv_ps(vext[a], xv, mask);
            // The f32 compare mask is all-ones/all-zeros per 32-bit lane, so a byte-blend selects the
            // whole index lane exactly.
            vidx[a] = _mm256_blendv_epi8(vidx[a], ibase[a], _mm256_castps_si256(mask));
            ibase[a] = _mm256_add_epi32(ibase[a], bump);
        }
    }
    // Collapse the 32 lane candidates through the scalar arg_fold (preserves the lowest-index tie-break
    // exactly).
    let mut vals = [0f32; 32];
    let mut idxs = [0i32; 32];
    for a in 0..4 {
        _mm256_storeu_ps(vals.as_mut_ptr().add(a * 8), vext[a]);
        _mm256_storeu_si256(idxs.as_mut_ptr().add(a * 8) as *mut __m256i, vidx[a]);
    }
    // A lane still holding the `-1` sentinel never fired the strict compare, i.e. every element it saw
    // was NaN or *exactly* the identity — and there the scalar twin does NOT agree: its `(ident,
    // usize::MAX)` seed makes `arg_fold`'s `a.0 == b.0 && b.1 < a.1` arm fire on the first
    // identity-valued element, so the twin reports that element's index while the lane reports
    // "nothing". (Seeding the lanes from real data instead would fix that but change the NaN
    // semantics, which are pinned to the identity seed.) Hand such a span to the twin outright: it is
    // the reference, so bit-identity is exact by construction. Only a span that is entirely ±inf/NaN
    // in some lane can take this branch (a fully masked logits row) — never real data, and `nsteps ==
    // 0` (len < 32, every lane unconsumed) falls in here too, matching `rowarg_avx2`'s `cols < 8`
    // delegation.
    if idxs.iter().any(|&k| k < 0) {
        // SAFETY: the same `[lo, hi)` read validity this function was called under.
        return unsafe { argreduce_chunk_scalar(x, lo, hi, is_max) };
    }
    let mut r = (ident_v, usize::MAX);
    for k in 0..32 {
        r = arg_fold(r, (vals[k], idxs[k] as usize), is_max);
    }
    // Scalar tail (< 32 elements), ascending — composes with the SIMD candidates under the same fold.
    let tail = lo + nsteps * 32;
    for i in tail..hi {
        r = arg_fold(r, (*x.add(i), i), is_max);
    }
    r
}

/// `argmax`/`argmin` over `x[0..n]` (serial), lowest index on ties. Returns the index as `i64`
/// (`-1` if `n <= 0`). Same fixed-`RCHUNK` decomposition as [`wukong_sreduce_f32`], so it is
/// bit-identical to [`wukong_argreduce_f32_parallel`] and the interpreter.
///
/// # Safety
/// `x` valid for `n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_argreduce_f32(x: *const f32, n: i64, op: i64) -> i64 {
    if n <= 0 {
        return -1;
    }
    let n = n as usize;
    let is_max = op == RED_ARGMAX;
    let ident_v = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let nchunks = n.div_ceil(RCHUNK);
    let mut acc = (ident_v, usize::MAX);
    for c in 0..nchunks {
        let lo = c * RCHUNK;
        let hi = ((c + 1) * RCHUNK).min(n);
        // SAFETY: [lo, hi) ⊆ [0, n).
        acc = arg_fold(acc, unsafe { argreduce_chunk(x, lo, hi, is_max) }, is_max);
    }
    acc.1 as i64
}

/// Multicore `argmax`/`argmin` — **bit-identical** to [`wukong_argreduce_f32`]. Partials are collected
/// in chunk order (rayon's indexed `collect`) and folded ascending — the identical tree as the serial
/// form — so the returned index never depends on thread count.
///
/// # Safety
/// `x` valid for `n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_argreduce_f32_parallel(x: *const f32, n: i64, op: i64) -> i64 {
    if n <= 0 {
        return -1;
    }
    let n = n as usize;
    let is_max = op == RED_ARGMAX;
    let ident_v = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let nchunks = n.div_ceil(RCHUNK);
    // One chunk, or a width-1 pool (RAYON_NUM_THREADS=1): nothing to parallelize — run the serial
    // sibling (bit-identical: fixed chunks + ordered fold).
    if nchunks < 2 || crate::wuk_pool_width() <= 1 {
        // SAFETY: same contract.
        return unsafe { wukong_argreduce_f32(x, n as i64, op) };
    }
    let xa = x as usize;
    // Unified kernel pool; fixed-`RCHUNK` chunks + lowest-index tie-break keep the result
    // pool- and thread-count-independent (see the module note above).
    let partials: Vec<(f32, usize)> = crate::run_on_wuk_pool(move || {
        (0..nchunks)
            .into_par_iter()
            .map(|c| {
                let lo = c * RCHUNK;
                let hi = ((c + 1) * RCHUNK).min(n);
                // SAFETY: disjoint read-only chunk; pointer valid for n.
                unsafe { argreduce_chunk(xa as *const f32, lo, hi, is_max) }
            })
            .collect()
    });
    let mut acc = (ident_v, usize::MAX);
    for p in partials {
        acc = arg_fold(acc, p, is_max);
    }
    acc.1 as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    // A deterministic, mildly varied input (no RNG — keeps the test reproducible across runs).
    fn fill(n: usize) -> (Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.013).sin() * 1.7 + 0.3)
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.027).cos() * 0.9 - 0.2)
            .collect();
        (x, y)
    }

    fn naive(x: &[f32], y: &[f32], op: i64) -> f64 {
        // f64 reference for the tolerance check. For max/min the fold picks an actual element, so the
        // f64 result equals the f32 kernel's bit-for-bit on the finite, non-NaN test data.
        let mut s = ident(op) as f64;
        for i in 0..x.len() {
            let (xi, yi) = (x[i] as f64, y[i] as f64);
            match op {
                RED_DOT => s += xi * yi,
                RED_SSD => s += (xi - yi) * (xi - yi),
                RED_SUM => s += xi,
                RED_SUMSQ => s += xi * xi,
                RED_MAX => s = if s > xi { s } else { xi },
                RED_MIN => s = if s < xi { s } else { xi },
                RED_MAXABS => {
                    let axi = xi.abs();
                    s = if s > axi { s } else { axi }
                }
                RED_SUMABS => s += xi.abs(),
                RED_ABSDIFF => s += (xi - yi).abs(),
                _ => {}
            };
        }
        s
    }

    const OPS: [i64; 9] = [
        RED_DOT, RED_SSD, RED_SUM, RED_SUMSQ, RED_MAX, RED_MIN, RED_MAXABS, RED_SUMABS, RED_ABSDIFF,
    ];

    // The unary ops read only `x`; the recognizer passes `y == x` for them. RED_ABSDIFF is binary
    // (reads y), so it stays OUT of this set — the tests must marshal a real y for it, like RED_SSD.
    fn unary(op: i64) -> bool {
        matches!(op, RED_SUM | RED_SUMSQ | RED_MAX | RED_MIN | RED_MAXABS | RED_SUMABS)
    }

    #[test]
    fn serial_matches_parallel_bit_for_bit() {
        // Sizes that exercise multiple chunks, a partial final chunk, and a non-multiple-of-8 tail.
        for &n in &[1usize, 7, 8, 9, 8192, 8193, 3 * 8192 + 13, 100_003] {
            let (x, y) = fill(n);
            for &op in &OPS {
                // For the unary ops the recognizer passes y == x; mirror that here.
                let yp = if unary(op) { x.as_ptr() } else { y.as_ptr() };
                let s = unsafe { wukong_sreduce_f32(x.as_ptr(), yp, n as i64, op) };
                let p = unsafe { wukong_sreduce_f32_parallel(x.as_ptr(), yp, n as i64, op) };
                assert_eq!(
                    s.to_bits(),
                    p.to_bits(),
                    "serial != parallel at n={n} op={op}: {s} vs {p}"
                );
            }
        }
    }

    #[test]
    fn within_tolerance_of_f64_reference() {
        let n = 100_003;
        let (x, y) = fill(n);
        for &op in &OPS {
            let yp = if unary(op) { x.as_ptr() } else { y.as_ptr() };
            let got = unsafe { wukong_sreduce_f32(x.as_ptr(), yp, n as i64, op) } as f64;
            let want = naive(&x, &y, op);
            let rel = (got - want).abs() / want.abs().max(1.0);
            assert!(rel < 1e-4, "op={op}: got {got}, want {want}, rel {rel}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for &n in &[8usize, 9, 17, 8191, 8192, 8193] {
            let (x, y) = fill(n);
            for &op in &OPS {
                let s = unsafe { reduce_chunk_scalar(x.as_ptr(), y.as_ptr(), 0, n, op) };
                let v = unsafe { reduce_chunk_avx2(x.as_ptr(), y.as_ptr(), 0, n, op) };
                assert_eq!(s.to_bits(), v.to_bits(), "scalar != avx2 at n={n} op={op}");
            }
        }
    }

    #[test]
    fn argreduce_serial_matches_parallel_and_naive() {
        // Deterministic argmax/argmin: serial == parallel, and == a naive ascending scan with the same
        // lowest-index tie-break — independent of how many RCHUNK chunks/threads ran.
        for &n in &[1usize, 7, 8, 9, 8192, 8193, 3 * 8192 + 13, 100_003] {
            let (x, _) = fill(n);
            for &op in &[RED_ARGMAX, RED_ARGMIN] {
                let s = unsafe { wukong_argreduce_f32(x.as_ptr(), n as i64, op) };
                let p = unsafe { wukong_argreduce_f32_parallel(x.as_ptr(), n as i64, op) };
                assert_eq!(s, p, "serial != parallel at n={n} op={op}");
                let is_max = op == RED_ARGMAX;
                let mut best = (
                    if is_max {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    },
                    usize::MAX,
                );
                for (i, &v) in x.iter().enumerate() {
                    best = arg_fold(best, (v, i), is_max);
                }
                assert_eq!(s, best.1 as i64, "kernel != naive at n={n} op={op}");
            }
        }
    }

    #[test]
    fn argreduce_duplicate_maxima_return_lowest_index() {
        // The global max 9.0 appears at indices spanning multiple RCHUNK chunks; argmax must return
        // the LOWEST such index, serial and parallel (the tie-break determinism guarantee).
        let n = 5 * 8192;
        let mut x = vec![1.0f32; n];
        for &i in &[3usize, 8192 + 17, 2 * 8192 + 5, 4 * 8192 + 100] {
            x[i] = 9.0;
        }
        let s = unsafe { wukong_argreduce_f32(x.as_ptr(), n as i64, RED_ARGMAX) };
        let p = unsafe { wukong_argreduce_f32_parallel(x.as_ptr(), n as i64, RED_ARGMAX) };
        assert_eq!(s, 3, "argmax lowest index (serial)");
        assert_eq!(p, 3, "argmax lowest index (parallel)");
        // argmin: minimum 1.0 first appears at index 0.
        assert_eq!(
            unsafe { wukong_argreduce_f32(x.as_ptr(), n as i64, RED_ARGMIN) },
            0,
            "argmin lowest index"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn argreduce_scalar_matches_avx2_bit_for_bit() {
        // The twin gate for the ARG reduction (`scalar_matches_avx2_bit_for_bit` above pins only the
        // VALUE reductions). `argreduce_chunk_avx2` is a separate 4-accumulator algorithm whose doc
        // block claims bit-identity with `argreduce_chunk_scalar`; this is what checks it — including
        // the DEGENERATE value sets the finite `fill()` data can never reach: a span made entirely of
        // the fold identity (∓∞), of NaN, or of a mix. An all-(-∞) span is a real input class (a
        // fully masked attention/logits row fed to a top-1).
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use std::arch::x86_64::{_CMP_GT_OQ, _CMP_LT_OQ};
        let sets: [(&str, fn(usize) -> Vec<f32>); 6] = [
            ("finite", |n| fill(n).0),
            ("all -inf", |n| vec![f32::NEG_INFINITY; n]),
            ("all +inf", |n| vec![f32::INFINITY; n]),
            ("all NaN", |n| vec![f32::NAN; n]),
            ("NaN/finite", |n| {
                (0..n)
                    .map(|i| if i % 3 == 0 { f32::NAN } else { i as f32 * 0.5 - 3.0 })
                    .collect()
            }),
            ("-inf/finite", |n| {
                (0..n)
                    .map(|i| {
                        if i % 5 == 0 {
                            i as f32 * 0.25 - 1.0
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect()
            }),
        ];
        for &n in &[1usize, 7, 8, 31, 32, 33, 63, 64, 8191, 8192] {
            for (name, make) in sets {
                let x = make(n);
                for is_max in [true, false] {
                    let s = unsafe { argreduce_chunk_scalar(x.as_ptr(), 0, n, is_max) };
                    let v = unsafe {
                        if is_max {
                            argreduce_chunk_avx2::<_CMP_GT_OQ>(
                                x.as_ptr(),
                                0,
                                n,
                                f32::NEG_INFINITY,
                                true,
                            )
                        } else {
                            argreduce_chunk_avx2::<_CMP_LT_OQ>(x.as_ptr(), 0, n, f32::INFINITY, false)
                        }
                    };
                    assert_eq!(
                        (s.0.to_bits(), s.1),
                        (v.0.to_bits(), v.1),
                        "scalar != avx2 arg-reduce: set={name} n={n} is_max={is_max}"
                    );
                }
            }
        }
    }

    #[test]
    fn argreduce_all_identity_span_returns_index_zero() {
        // End-to-end contract for the degenerate span the twin gate above isolates: with every element
        // equal to the fold identity, "lowest index wins on ties" makes the answer 0 — not the `-1`
        // that `wukong_argreduce_f32` reserves for `n <= 0`. Sized to span several RCHUNK chunks so
        // serial and parallel both exercise the multi-chunk fold.
        for &n in &[32usize, 33, 8192, 3 * 8192 + 13] {
            for &(op, ident_v) in &[
                (RED_ARGMAX, f32::NEG_INFINITY),
                (RED_ARGMIN, f32::INFINITY),
            ] {
                let x = vec![ident_v; n];
                let s = unsafe { wukong_argreduce_f32(x.as_ptr(), n as i64, op) };
                let p = unsafe { wukong_argreduce_f32_parallel(x.as_ptr(), n as i64, op) };
                assert_eq!(s, 0, "all-identity argreduce n={n} op={op} (serial)");
                assert_eq!(s, p, "serial != parallel n={n} op={op}");
            }
        }
    }

    #[test]
    fn argreduce_avx2_lane_ties_and_tails() {
        // Pin the AVX2 path's two tricky cases against the naive ascending scan: (a) duplicate maxima
        // in different SIMD lanes *within one 32-wide span* must collapse to the lower lane index, and
        // (b) the extreme living in a `< 32` scalar tail of a non-multiple-of-32 length.
        let naive = |x: &[f32], is_max: bool| {
            let mut best = (
                if is_max { f32::NEG_INFINITY } else { f32::INFINITY },
                usize::MAX,
            );
            for (i, &v) in x.iter().enumerate() {
                best = arg_fold(best, (v, i), is_max);
            }
            best.1 as i64
        };
        // (a) ties at lanes 5 and 13 of the first span — argmax must return 5.
        let mut x = vec![0.5f32; 64];
        x[5] = 7.0;
        x[13] = 7.0;
        x[40] = 7.0;
        assert_eq!(
            unsafe { wukong_argreduce_f32(x.as_ptr(), 64, RED_ARGMAX) },
            5
        );
        assert_eq!(naive(&x, true), 5);
        // (b) extreme in the scalar tail of several non-mult-of-32 lengths, both ops.
        for &n in &[33usize, 47, 63, 65, 95, 1000] {
            let mut x = vec![0.0f32; n];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i * 31 % 17) as f32) * 0.25 - 2.0; // varied finite data
            }
            x[n - 1] = 100.0; // max in the tail
            x[n - 2] = -100.0; // min in the tail
            assert_eq!(
                unsafe { wukong_argreduce_f32(x.as_ptr(), n as i64, RED_ARGMAX) },
                naive(&x, true),
                "argmax tail n={n}"
            );
            assert_eq!(
                unsafe { wukong_argreduce_f32(x.as_ptr(), n as i64, RED_ARGMIN) },
                naive(&x, false),
                "argmin tail n={n}"
            );
        }
    }
}
