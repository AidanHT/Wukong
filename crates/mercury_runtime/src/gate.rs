//! Gated-FFN activation — the SwiGLU / GeGLU feed-forward gate of Llama / PaLM / Gemma (and the
//! GLU-family generally). The transformer FFN's first projection produces two halves `a` and `b`;
//! the gated activation combines them elementwise:
//!
//! ```text
//! out[i] = act(a[i]) * b[i]        act ∈ { silu (SwiGLU), gelu (GeGLU) }
//! ```
//!
//! `a` is the *gate* branch (run through the activation), `b` the *linear* (value) branch — both the
//! length-`n` projected halves. The activation folds an `exp` (`silu = x·σ(x)`, `gelu`'s tanh-approx),
//! exactly the libm wall C/Rust keep **scalar** (a loop calling `siluf`/`geluf` won't vectorize), so
//! fusing the gate into one 256-bit AVX2 pass — the activation across 8 lanes, then one `mulps` by the
//! linear branch — wins for the same reason the forward activation dispatch (`mercury_vmath_f32`) does.
//!
//! **Reuse, not reinvention.** The activation is the *identical* [`crate::vmath::silu1`]/[`gelu1`]
//! (scalar) and [`silu8`]/[`gelu8`] (AVX2) the forward dispatch and the GEMM fused epilogue use, with
//! the gate's only extra op a multiply by `b`. So `mercury_gate_f32(SILU, a, ones, ·)` is bit-for-bit
//! the forward `silu(a)`, and the gate composes the same ≈1-ULP `exp` the whole activation family
//! shares. The scalar twin (`act1(a)·b`) backs the < 8 tail and the no-AVX2 fallback, so every lane of
//! every path agrees **bit-for-bit** (pinned by a twin test at non-multiple-of-8 `n`).
//!
//! **Determinism.** This is a *pure elementwise* map — each `out[i]` depends only on `a[i]`, `b[i]` —
//! so there is no reduction and no reassociation. The `_parallel` entry cuts `[0, n)` into FIXED-size
//! chunks (count independent of thread count), runs the *identical* per-element routine on each, and
//! maps them across cores with rayon: `serial == parallel` bit-for-bit on any machine, and the
//! interpreter (which marshals the serial form) agrees with the native `@parallel` path exactly.

use crate::vmath::{gelu1, silu1};
use rayon::prelude::*;

// --- op codes (which activation runs on the gate branch `a`) --------------------------------------
/// SwiGLU: `out[i] = silu(a[i]) * b[i]`, `silu(x) = x·σ(x)` (Llama / PaLM).
pub const GATE_SILU: i64 = 0;
/// GeGLU: `out[i] = gelu(a[i]) * b[i]`, tanh-approx GELU (Gemma and GLU-variant FFNs).
pub const GATE_GELU: i64 = 1;

/// `n` below which `_parallel` just runs serially — the rayon fan-out isn't worth it for tiny gates.
const GATE_PAR_MIN: usize = 4096;

/// Fixed chunk size in elements — independent of thread count, which is what makes the parallel
/// decomposition deterministic (each chunk runs the identical per-element routine, so the result is
/// bit-identical regardless of how many chunks/threads ran). 8192 f32 = 32 KB (an L1's worth) per
/// chunk; plenty of chunks for rayon work-stealing at the sizes that matter. Must stay constant for
/// serial / parallel / interp to agree. Matches the `RCHUNK` reduction convention.
const GCHUNK: usize = 8192;

/// One element, scalar: `act_op(a) * b` (the AVX2 tail + the no-AVX2 fallback). Reuses the shared
/// activation twins, so it is bit-identical to the forward `silu`/`gelu` dispatch.
#[inline]
fn gate1(op: i64, a: f32, b: f32) -> f32 {
    match op {
        GATE_SILU => silu1(a) * b,
        GATE_GELU => gelu1(a) * b,
        _ => a * b,
    }
}

/// Select the 8-lane activation for `op`, or `None` for an unrecognized code. The kernel applies it
/// across 8 lanes and multiplies by the linear branch — `act8` is the *exact* function the forward
/// dispatch uses, so the gate is bit-identical to it.
#[cfg(target_arch = "x86_64")]
#[inline]
fn gate8_for(
    op: i64,
) -> Option<unsafe fn(std::arch::x86_64::__m256) -> std::arch::x86_64::__m256> {
    Some(match op {
        GATE_SILU => crate::vmath::silu8,
        GATE_GELU => crate::vmath::gelu8,
        _ => return None,
    })
}

/// `out[lo..hi] = act_op(a) * b`, scalar — the per-element routine, identical in the serial loop, the
/// AVX2 tail, and each parallel chunk (so serial == parallel == the interpreter's marshalled form).
///
/// # Safety
/// `a`, `b`, `out` valid for `[lo, hi)` f32 (`out` may alias `a` or `b` — each element reads `a[i]`
/// and `b[i]` before writing `out[i]`, so in-place is sound).
#[inline]
unsafe fn gate_chunk_scalar(a: *const f32, b: *const f32, out: *mut f32, lo: usize, hi: usize, op: i64) {
    let mut i = lo;
    while i < hi {
        *out.add(i) = gate1(op, *a.add(i), *b.add(i));
        i += 1;
    }
}

/// `out[lo..hi] = act_op(a) * b`, AVX2/FMA — 8 lanes/step (the shared `act8` then one `_mm256_mul_ps`
/// by the linear branch) + a scalar tail via [`gate_chunk_scalar`]. The lanes mirror the scalar twin
/// op-for-op (`act8` == `act1` bit-for-bit, and `mulps` == the scalar `*`), so the whole chunk is
/// bit-identical to the scalar path.
///
/// # Safety
/// `a`, `b`, `out` valid for `[lo, hi)` f32 (`out` may alias `a`/`b`); AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gate_chunk_avx2(a: *const f32, b: *const f32, out: *mut f32, lo: usize, hi: usize, op: i64) {
    use std::arch::x86_64::*;
    let Some(act8) = gate8_for(op) else {
        // Unrecognized op: fall back to the scalar identity twin (a*b) for the whole chunk.
        return gate_chunk_scalar(a, b, out, lo, hi, op);
    };
    let mut i = lo;
    while i + 8 <= hi {
        let av = _mm256_loadu_ps(a.add(i));
        let bv = _mm256_loadu_ps(b.add(i));
        // act(a) * b — the activation across 8 lanes, then one multiply by the linear branch.
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(act8(av), bv));
        i += 8;
    }
    // Scalar tail (same activation twin as the lanes) for the final < 8 elements of the chunk.
    while i < hi {
        *out.add(i) = gate1(op, *a.add(i), *b.add(i));
        i += 1;
    }
}

/// `out[lo..hi] = act_op(a) * b`, AVX2 when available else the scalar twin. The one per-chunk routine
/// the serial loop and every parallel chunk share.
///
/// # Safety
/// `a`, `b`, `out` valid for `[lo, hi)` f32 (`out` may alias `a`/`b`).
#[inline]
unsafe fn gate_chunk(a: *const f32, b: *const f32, out: *mut f32, lo: usize, hi: usize, op: i64) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; buffers valid for [lo, hi) by the caller contract.
            return unsafe { gate_chunk_avx2(a, b, out, lo, hi, op) };
        }
    }
    unsafe { gate_chunk_scalar(a, b, out, lo, hi, op) };
}

/// Gated-FFN activation `out[i] = act_op(a[i]) * b[i]` for `i in 0..n`, `act` selected by `op` (see the
/// `GATE_*` codes). `a` is the gate branch, `b` the linear branch. Uses the 256-bit AVX2/FMA path when
/// available (8 lanes/step + a scalar tail), else the scalar fallback. `out` may alias `a` or `b` (the
/// recognizer allows an in-place gate).
///
/// # Safety
/// `a`, `b`, and `out` must each be valid for `n` `f32` elements.
#[no_mangle]
pub unsafe extern "C" fn mercury_gate_f32(a: *const f32, b: *const f32, out: *mut f32, n: i64, op: i64) {
    if n <= 0 {
        return;
    }
    // SAFETY: one chunk over the whole range; buffers valid for n by contract.
    unsafe { gate_chunk(a, b, out, 0, n as usize, op) };
}

/// Multicore gated-FFN activation — **bit-identical** to [`mercury_gate_f32`]. `[0, n)` is cut into
/// fixed-size [`GCHUNK`] chunks (count independent of thread count); each chunk runs the *identical*
/// per-element routine and the chunks are mapped across cores. Because the map is pure elementwise (no
/// cross-element state, no reduction), the result does not depend on thread count — `serial ==
/// parallel` exactly, so the interpreter's serial call agrees with this `@parallel` path bit-for-bit.
///
/// # Safety
/// Operand-size contract of [`mercury_gate_f32`].
#[no_mangle]
pub unsafe extern "C" fn mercury_gate_f32_parallel(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    n: i64,
    op: i64,
) {
    if n <= 0 {
        return;
    }
    let n = n as usize;
    if n < GATE_PAR_MIN {
        // SAFETY: same contract; the serial form for small n.
        return unsafe { mercury_gate_f32(a, b, out, n as i64, op) };
    }
    let nchunks = n.div_ceil(GCHUNK);
    // Raw pointers cross the rayon boundary as integers; each chunk is a disjoint output sub-range, and
    // `a`/`b` are read-only, so there is no data race.
    let (a_addr, b_addr, out_addr) = (a as usize, b as usize, out as usize);
    (0..nchunks).into_par_iter().for_each(|c| {
        let lo = c * GCHUNK;
        let hi = ((c + 1) * GCHUNK).min(n);
        // SAFETY: disjoint output chunk [lo, hi); pointers re-derived from the captured addresses and
        // valid for n by contract.
        unsafe { gate_chunk(a_addr as *const f32, b_addr as *const f32, out_addr as *mut f32, lo, hi, op) };
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic, mildly varied streams (no RNG — reproducible). Distinct multipliers/phases so a
    // and b are not accidentally equal, and `a` spans both signs (exercising the activation's tails).
    fn fill_a(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.023).sin() * 3.1 - 0.6)
            .collect()
    }
    fn fill_b(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.017 + 0.4).cos() * 1.7 + 0.3)
            .collect()
    }

    const OPS: [i64; 2] = [GATE_SILU, GATE_GELU];

    /// f64 reference activations (independent of the kernel's f32 polynomials) for the formula check.
    fn silu_f64(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }
    fn gelu_f64(x: f64) -> f64 {
        // tanh approximation (matches the forward `gelu1` family).
        let c0 = (2.0f64 / std::f64::consts::PI).sqrt();
        0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
    }
    fn act_f64(op: i64, x: f64) -> f64 {
        match op {
            GATE_SILU => silu_f64(x),
            GATE_GELU => gelu_f64(x),
            _ => unreachable!(),
        }
    }

    /// (a) scalar path == AVX2 path bit-for-bit, both ops, across `n` straddling the 8-lane edge —
    /// the tail agreement that underwrites the differential gate.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for &n in &[1usize, 7, 8, 9, 16, 17, 1000] {
            let a = fill_a(n);
            let b = fill_b(n);
            for &op in &OPS {
                let mut scal = vec![0.0f32; n];
                let mut avx = vec![0.0f32; n];
                unsafe {
                    gate_chunk_scalar(a.as_ptr(), b.as_ptr(), scal.as_mut_ptr(), 0, n, op);
                    gate_chunk_avx2(a.as_ptr(), b.as_ptr(), avx.as_mut_ptr(), 0, n, op);
                }
                for i in 0..n {
                    assert_eq!(
                        scal[i].to_bits(),
                        avx[i].to_bits(),
                        "scalar != avx2 op={op} n={n} i={i}: {} vs {}",
                        scal[i],
                        avx[i]
                    );
                }
            }
        }
    }

    /// (b) ≈ f64 reference `act_f64(a) * b` for both ops — guards the *formula* (not just
    /// scalar==avx2): an independent double-precision recompute.
    #[test]
    fn matches_f64_reference() {
        for &n in &[1usize, 7, 8, 9, 16, 17, 1000] {
            let a = fill_a(n);
            let b = fill_b(n);
            for &op in &OPS {
                let mut got = vec![0.0f32; n];
                unsafe {
                    mercury_gate_f32(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), n as i64, op);
                }
                for i in 0..n {
                    let want = act_f64(op, a[i] as f64) * (b[i] as f64);
                    let tol = 1e-4 * want.abs() + 1e-4;
                    assert!(
                        ((got[i] as f64) - want).abs() <= tol,
                        "f64 ref op={op} n={n} i={i}: got {} want {want} (a={}, b={})",
                        got[i],
                        a[i],
                        b[i]
                    );
                }
            }
        }
    }

    /// (c) serial == parallel bit-for-bit, both ops, across `n` straddling `GCHUNK` and the parallel
    /// threshold. Rests on the fixed-chunk decomposition being thread-count-independent.
    #[test]
    fn serial_matches_parallel_bit_for_bit() {
        for &n in &[1usize, 7, 8, 9, 17, 1000, 4095, 4096, 8192, 8193, 20000] {
            let a = fill_a(n);
            let b = fill_b(n);
            for &op in &OPS {
                let mut ser = vec![0.0f32; n];
                let mut par = vec![0.0f32; n];
                unsafe {
                    mercury_gate_f32(a.as_ptr(), b.as_ptr(), ser.as_mut_ptr(), n as i64, op);
                    mercury_gate_f32_parallel(a.as_ptr(), b.as_ptr(), par.as_mut_ptr(), n as i64, op);
                }
                for i in 0..n {
                    assert_eq!(
                        ser[i].to_bits(),
                        par[i].to_bits(),
                        "serial != parallel op={op} n={n} i={i}: {} vs {}",
                        ser[i],
                        par[i]
                    );
                }
            }
        }
    }

    /// (d) `GATE_SILU` with `b == 1` equals plain `silu(a)` — i.e. the gate reuses the *exact* forward
    /// activation (`mercury_vmath_f32(VM_SILU)`), so a SwiGLU with a unit linear branch is the forward
    /// pass bit-for-bit.
    #[test]
    fn silu_gate_with_unit_b_equals_forward_silu() {
        for &n in &[1usize, 8, 9, 17, 1000] {
            let a = fill_a(n);
            let ones = vec![1.0f32; n];
            let mut gated = vec![0.0f32; n];
            let mut fwd = vec![0.0f32; n];
            unsafe {
                mercury_gate_f32(a.as_ptr(), ones.as_ptr(), gated.as_mut_ptr(), n as i64, GATE_SILU);
                // The forward activation dispatch (the very kernel a `silu(a[i])` loop lowers to).
                crate::vmath::mercury_vmath_f32(a.as_ptr(), fwd.as_mut_ptr(), n as i64, crate::vmath::VM_SILU);
            }
            for i in 0..n {
                assert_eq!(
                    gated[i].to_bits(),
                    fwd[i].to_bits(),
                    "silu-gate(b=1) != forward silu n={n} i={i}: {} vs {}",
                    gated[i],
                    fwd[i]
                );
            }
        }
    }

    /// Edge: zero/negative `n` is a no-op (don't write, don't panic) on both entries.
    #[test]
    fn degenerate_n_is_noop() {
        let a = fill_a(8);
        let b = fill_b(8);
        let mut out = vec![42.0f32; 8];
        unsafe {
            mercury_gate_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 0, GATE_SILU);
            mercury_gate_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), -3, GATE_GELU);
            mercury_gate_f32_parallel(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 0, GATE_SILU);
            mercury_gate_f32_parallel(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), -1, GATE_GELU);
        }
        assert!(out.iter().all(|&v| v == 42.0), "no-op must not write out");
    }
}
