//! Fused scaled-dot-product attention — the flash-attention kernel.
//!
//! Computes, for one head, `O[S,D] = softmax(scale · Q·Kᵀ [+ causal mask]) · V` with `Q,K,V,O`
//! row-major `[S,D]` f32. The softmax runs **online** (a running max `m`, running denominator `l`,
//! and a rescaled `O(D)` accumulator — the flash-attention recurrence), so the `S×S` score matrix is
//! **never materialized**: only one score and the `D`-wide accumulator are live per (query,key) step.
//!
//! That is the whole point. A naive `scores = Q·Kᵀ; softmax(scores); O = scores·V` allocates an
//! `S×S`-per-head intermediate (1M f32 = 4 MB at `S=1024`) and streams it through memory three times;
//! the working set blows past cache and the kernel goes memory-bound. The fused form keeps the live
//! state at `O(D)` and reads `Q`/`K`/`V` straight through, which is exactly the memory wall
//! flash-attention was built to break — the win grows with the sequence length.
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
//! `q_i·k_j` dot across 8 lanes where [`attention_scalar`] sums sequentially, so those two agree only
//! to tolerance — the naive materialized reference in the tests is what gates both.

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

/// Portable reference: the same online-softmax recurrence, scalar. Used where AVX2 is unavailable;
/// also the algorithm the AVX2 path vectorizes (over `D`). The vectorization *does* reassociate the
/// `q_i·k_j` dot — the AVX2 path accumulates 8 partial sums and horizontally reduces them (:161-175)
/// where this one sums sequentially — so the two agree to tolerance, not bit-for-bit. Both are gated
/// against the naive materialized reference, which is the actual oracle.
///
/// # Safety
/// Operand-size contract of [`wukong_attention_f32`]: `q`, `k`, `v` must each be valid for `s*d`
/// `f32` reads and `o` valid for `s*d` `f32` writes, with `o` not overlapping `q`/`k`/`v`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
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
                let corr = (m - m_new).exp(); // exp(old_max - new_max) ∈ (0,1]; exp(-inf)=0 on key 0
                let p = (x - m_new).exp();
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// Query tile: a block of `ATT_QB` query rows is kept hot together so each loaded `k_j`/`v_j` (read
/// once per key) is reused across all of them — turning the per-query kernel's `S`× re-streaming of
/// K/V into one L2-resident pass per query block. The block's running stats + `O(QB·D)` accumulator
/// stay L1-resident.
const ATT_QB: usize = 32;

/// AVX2/FMA path: the online-softmax recurrence, **query-block tiled** for cache reuse. The keys are
/// the *outer* loop and the queries within a block the *inner* loop, so `k_j`/`v_j` load once and
/// feed every query in the block. The two `D`-wide inner loops — the `q_i·k_j` dot and the
/// `acc = acc·corr + p·v_j` rescale-accumulate — are vectorized 8-wide (scalar remainder for `D % 8`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
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
    use std::arch::x86_64::*;
    let dv = d & !7; // largest multiple of 8 ≤ d
    let mut m = [f32::NEG_INFINITY; ATT_QB];
    let mut l = [0.0f32; ATT_QB];
    let mut acc = vec![0.0f32; ATT_QB * d];
    let mut i0 = 0;
    while i0 < s {
        let qb = (s - i0).min(ATT_QB);
        m[..qb].iter_mut().for_each(|x| *x = f32::NEG_INFINITY);
        l[..qb].iter_mut().for_each(|x| *x = 0.0);
        acc[..qb * d].iter_mut().for_each(|x| *x = 0.0);
        // The largest key any query in this block attends to is `i0 + qb - 1` under a causal mask.
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
                // x = scale · (q_i · k_j)
                let mut sv = _mm256_setzero_ps();
                let mut t = 0;
                while t < dv {
                    sv =
                        _mm256_fmadd_ps(_mm256_loadu_ps(qi.add(t)), _mm256_loadu_ps(kj.add(t)), sv);
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
                // acc_ii = acc_ii·corr + p·v_j
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
        // O_i = acc_i / l_i for each query in the block.
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

#[cfg(test)]
mod tests {
    use super::*;

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
        // Sizes straddling the 8-wide D remainder and the causal/full mask.
        for &(s, d) in &[
            (1, 1),
            (2, 2),
            (4, 8),
            (5, 7),
            (8, 16),
            (16, 8),
            (33, 17),
            (64, 64),
        ] {
            check(s, d, false);
            check(s, d, true);
        }
    }

    /// The scalar twin is the only path on a machine without AVX2/FMA and on every non-x86_64 build,
    /// but `wukong_attention_f32`'s runtime dispatch (:44) sends every CI and developer machine down
    /// the AVX2 path — so `attention_matches_naive` above never executes `attention_scalar` and a
    /// regression in it (a dropped `corr` rescale, a lost `l == 0` guard) would ship green. Drive it
    /// directly over the same shapes.
    #[test]
    fn attention_scalar_matches_naive() {
        for &(s, d) in &[
            (1usize, 1usize),
            (2, 2),
            (4, 8),
            (5, 7),
            (8, 16),
            (16, 8),
            (33, 17),
            (64, 64),
        ] {
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

    /// Throughput probe (run: `cargo test -p wukong_runtime --release attention_throughput --
    /// --ignored --nocapture --test-threads=1`). The filter and `--test-threads=1` are load-bearing:
    /// an unfiltered `--ignored` run starts all nine of the crate's throughput probes at once on
    /// libtest's default thread pool, so each one measures a machine saturated by the other eight.
    /// Compares the fused kernel against Wukong's *own* strongest non-fused path: materialize
    /// `scores = Q·Kᵀ` with the tuned AVX2 GEMM, softmax the rows, then `O = P·V` with the GEMM —
    /// the two-matmul + S×S-intermediate shape. Both paths use the same AVX2 primitives and the same
    /// S² scalar `exp`s, so the delta is purely the fusion / no-materialization effect. That baseline
    /// is *stronger* than idiomatic C attention (which uses unblocked matmuls), so a win here implies
    /// a win over C.
    #[test]
    #[ignore]
    fn attention_throughput() {
        use crate::{wukong_sgemm, wukong_sgemm_nt};
        use std::time::Instant;

        // Wukong's best non-fused attention: GEMM + materialized softmax + GEMM.
        fn materialized(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            s: usize,
            d: usize,
            scale: f32,
            causal: bool,
        ) -> Vec<f32> {
            let mut scores = vec![0.0f32; s * s];
            unsafe {
                wukong_sgemm_nt(
                    q.as_ptr(),
                    k.as_ptr(),
                    scores.as_mut_ptr(),
                    s as i64,
                    d as i64,
                    s as i64,
                    0,
                );
            }
            for i in 0..s {
                let row = &mut scores[i * s..i * s + s];
                let jmax = if causal { i + 1 } else { s };
                let mut mx = f32::NEG_INFINITY;
                for r in row.iter_mut().take(jmax) {
                    *r *= scale;
                    mx = mx.max(*r);
                }
                let mut sum = 0.0f32;
                for r in row.iter_mut().take(jmax) {
                    *r = (*r - mx).exp();
                    sum += *r;
                }
                let inv = 1.0 / sum;
                for r in row.iter_mut().take(jmax) {
                    *r *= inv;
                }
                for r in row.iter_mut().skip(jmax) {
                    *r = 0.0;
                }
            }
            let mut o = vec![0.0f32; s * d];
            unsafe {
                wukong_sgemm(
                    scores.as_ptr(),
                    v.as_ptr(),
                    o.as_mut_ptr(),
                    s as i64,
                    s as i64,
                    d as i64,
                    0,
                );
            }
            o
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

        for &(s, d) in &[(256usize, 64usize), (512, 64), (1024, 64), (2048, 64)] {
            let q = fill(1, s * d);
            let k = fill(2, s * d);
            let v = fill(3, s * d);
            let scale = 1.0 / (d as f32).sqrt();
            let mut o = vec![0.0f32; s * d];
            let (qp, kp, vp, op) = (q.as_ptr(), k.as_ptr(), v.as_ptr(), o.as_mut_ptr());
            let mat = bench(&|| {
                std::hint::black_box(materialized(&q, &k, &v, s, d, scale, false));
            });
            let fused = bench(&|| unsafe {
                wukong_attention_f32(qp, kp, vp, op, s as i64, d as i64, scale, 0);
            });
            println!(
                "S={s:<5} D={d}: materialized {mat:7.3} ms | fused {fused:7.3} ms | {:.2}x",
                mat / fused
            );
        }
    }
}
