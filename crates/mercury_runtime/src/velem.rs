//! Streaming elementwise affine + activation — `out[i] = act(a·x[i] + b·y[i] + c)` — the **256-bit
//! AVX2** kernel a recognized streaming map loop (saxpy / scale / residual-add / bias / ReLU / ReLU6)
//! lowers to. Two reasons it beats the Cranelift-vectorized form gcc/rustc also emit:
//!
//!  1. **256-bit width.** Cranelift's generic vectorizer is stuck at 128-bit SSE (`f32x8` does not
//!     legalize), so a Mercury saxpy ran ~15% *behind* gcc's 256-bit AVX2. This kernel restores the
//!     width parity, the same play as `vmath`/`gemm`.
//!  2. **Non-temporal (streaming) stores.** Once the working set spills L3 (≥ [`NT_MIN_BYTES`] across
//!     all live arrays) the store goes out `vmovntps`, which skips the **read-for-ownership** every
//!     cacheable store pays (the CPU must pull the target line into cache before overwriting it).
//!     gcc/rustc cannot emit this automatically — they can't prove the array is large and write-once —
//!     but Mercury's domain-aware lowering *knows* the loop streams a whole tensor. The threshold is on
//!     the *total* bytes touched, not the length, so a cache-resident map keeps its normal store (where
//!     a needless `vmovntps` would lose): measured ~1.1–1.4× over gcc at real (>L3) activation-tensor
//!     sizes, and a clean tie at the 4 MiB benchmark size where the output still lives in L3.
//!
//! The interpreter marshals its abstract memory through this **identical** kernel (like `vmath`), so
//! the differential oracle stays bit-for-bit exact. NT and cacheable stores write the *same bits* (the
//! float value is identical; only the cache path differs), and the scalar tail/fallback mirrors the
//! AVX2 lanes op-for-op (the `act_avx`/`act1` pair share the `maxps`/`minps` semantics), so every lane
//! of every path agrees — pinned by `velem_tail_matches_lanes` and `velem_scalar_matches_avx`.

// --- op codes (shared with the recognizer in mercury_mir_build) ------------------------------------
// The low byte is the activation; `VE_USE_Y` (bit 8) flags that `y` is read (so a scale/ReLU that
// never touches `y` can pass a null `y` pointer the kernel will not dereference).
/// `out = a·x (+ b·y) + c`, no activation.
pub const VE_ID: i64 = 0;
/// `out = max(a·x (+ b·y) + c, 0)` — ReLU.
pub const VE_RELU: i64 = 1;
/// `out = min(max(a·x (+ b·y) + c, 0), 6)` — ReLU6.
pub const VE_RELU6: i64 = 2;
/// OR'd into `op` when the kernel must read `y` (`b` may be non-zero).
pub const VE_USE_Y: i64 = 256;

/// Total streamed bytes (all live arrays) at/above which the store goes non-temporal. Non-temporal
/// stores pay off only once the working set spills L3: below it a normal cacheable store keeps the
/// output hot — C/Rust win there, and a per-core `@parallel` chunk may be re-read — while above it
/// `vmovntps` streams the array out once and skips the read-for-ownership traffic the cacheable store
/// pays. The crossover is the **total** bytes touched, not the element count: a 2-stream map
/// (`x → out`) at 1<<20 is 8 MiB and fits L3, but a 3-stream map (`x, y → out`) at the *same* length
/// is 12 MiB and spills it — so the two want opposite store policies at one length (measured: relu
/// regressed under non-temporal stores at 1<<20 where saxpy gained). ~10 MiB ≈ this machine's L3.
const NT_MIN_BYTES: usize = 10 * 1024 * 1024;

/// Whether a kernel touching `streams` arrays of `n` f32 each should use non-temporal stores — true
/// once the working set spills L3 (see [`NT_MIN_BYTES`]). `streams` counts every live array (the
/// output plus each input read), since they all compete for cache residency.
#[inline]
fn use_nt(n: usize, streams: usize) -> bool {
    streams.saturating_mul(n).saturating_mul(4) >= NT_MIN_BYTES
}

/// Software-prefetch distance (elements ahead). A prefetch of an address past the buffer end is a
/// hint the hardware silently drops — never a fault — so the last iterations need no guard.
const PF_AHEAD: usize = 128;

/// Scalar activation, mirroring the AVX2 `maxps`/`minps` semantics exactly: `maxps(v,0)` is
/// `(v > 0) ? v : 0` (returns the second operand for `±0`/NaN), so the `if` form agrees lane-for-lane.
#[inline]
fn act1(op: i64, v: f32) -> f32 {
    match op & 0xff {
        VE_RELU => {
            if v > 0.0 {
                v
            } else {
                0.0
            }
        }
        VE_RELU6 => {
            let v = if v > 0.0 { v } else { 0.0 };
            if v < 6.0 {
                v
            } else {
                6.0
            }
        }
        _ => v,
    }
}

/// One element `act(a·x + b·y + c)`. `b·y + c` is fused (`mul_add`) then `a·x + that` is fused, so the
/// chain is exactly two roundings — matching the AVX2 `fmadd` pair, and (for saxpy `a·x + y`, i.e.
/// `b = 1, c = 0`) the single `fma` gcc emits under `-ffp-contract=fast`. `y` is only read when the
/// caller set `VE_USE_Y`.
///
/// **Identity-affine fast path.** When `a == 1`, `c == 0` and `y` is unread, `a·x + c` is just `x`, so
/// the FMA is skipped entirely — the case a recognized `relu`/`relu6`/copy hits (`out[i] =
/// max(x[i], 0)`). For `max(1·x+0, 0)` == `max(x, 0)` bit-for-bit (the FMA's only observable effect,
/// flipping `-0.0` to `+0.0`, is erased by the following `max`), and a bare copy preserves the input.
/// The AVX2 lanes apply the identical (loop-invariant) test, so the scalar tail and the vector body
/// still agree lane-for-lane — what `velem_tail_matches_lanes` pins. This removes the wasted multiply
/// that left a pure `relu` a touch *behind* gcc's bare `maxps` loop at L3-resident sizes.
#[inline]
fn elem1(op: i64, x: f32, y: f32, a: f32, b: f32, c: f32) -> f32 {
    let base = if op & VE_USE_Y == 0 && a == 1.0 && c == 0.0 {
        x
    } else {
        let inner = if op & VE_USE_Y != 0 {
            b.mul_add(y, c)
        } else {
            c
        };
        a.mul_add(x, inner)
    };
    act1(op, base)
}

/// `out[i] = act(a·x[i] + b·y[i] + c)` for `i in 0..n`. Uses the 256-bit AVX2/FMA kernel when
/// available (8 lanes/step + a scalar tail), with non-temporal stores for a large output; else a
/// scalar fallback. `x`/`y`/`out` may alias (the recognizer allows in-place). `y` is dereferenced
/// only when `op & VE_USE_Y` is set.
///
/// # Safety
/// `x` and `out` must be valid for `n` `f32`; `y` must be valid for `n` `f32` when `op & VE_USE_Y`.
#[no_mangle]
pub unsafe extern "C" fn mercury_velem_f32(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: i64,
    a: f32,
    b: f32,
    c: f32,
    op: i64,
) {
    if n <= 0 {
        return;
    }
    let n = n as usize;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; buffers valid for n by the caller contract.
            unsafe { velem_avx2(x, y, out, n, a, b, c, op) };
            return;
        }
    }
    let use_y = op & VE_USE_Y != 0;
    for i in 0..n {
        // SAFETY: i < n; buffers valid for n (y only when use_y).
        unsafe {
            let yi = if use_y { *y.add(i) } else { 0.0 };
            *out.add(i) = elem1(op, *x.add(i), yi, a, b, c);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn velem_avx2(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
    a: f32,
    b: f32,
    c: f32,
    op: i64,
) {
    use std::arch::x86_64::*;
    let va = _mm256_set1_ps(a);
    let vb = _mm256_set1_ps(b);
    let vc = _mm256_set1_ps(c);
    let zero = _mm256_setzero_ps();
    let six = _mm256_set1_ps(6.0);
    let use_y = op & VE_USE_Y != 0;
    let act = op & 0xff;
    // Identity affine (`a == 1`, `c == 0`, no `y`): the FMA degenerates to `x`, so skip it — the case a
    // bare `relu`/`relu6`/copy hits. Hoisted here so the hot loop branches on a single invariant flag
    // (matching `elem1`'s scalar fast path), turning a recognized ReLU into gcc's bare `maxps` loop at
    // true 256-bit width instead of paying a wasted multiply per 8 lanes.
    let id_affine = !use_y && a == 1.0 && c == 0.0;
    // Streams = output + x (+ y when read). The non-temporal decision keys on the whole working set,
    // so a 2-input saxpy spills L3 (and wants `vmovntps`) at a length where a 1-input map still fits.
    let nt = use_nt(n, if use_y { 3 } else { 2 });
    let mut i = 0usize;
    // For the streaming store, peel a scalar prologue until `out` is 32-byte aligned (vmovntps faults
    // on a misaligned address); after that each 8-lane step keeps it aligned.
    if nt {
        while i < n && (out.add(i) as usize) & 31 != 0 {
            let yi = if use_y { *y.add(i) } else { 0.0 };
            *out.add(i) = elem1(op, *x.add(i), yi, a, b, c);
            i += 1;
        }
    }
    // One 8-lane result vector for the element at `i + $off`: the affine `a·x (+ b·y) + c` followed by
    // the optional ReLU/ReLU6 clamp. Factored so the main loop can run four of them with no
    // inter-vector dependency.
    macro_rules! compute {
        ($off:expr) => {{
            let xv = _mm256_loadu_ps(x.add(i + $off));
            let mut r = if id_affine {
                xv
            } else {
                let inner = if use_y {
                    _mm256_fmadd_ps(vb, _mm256_loadu_ps(y.add(i + $off)), vc)
                } else {
                    vc
                };
                _mm256_fmadd_ps(va, xv, inner)
            };
            if act == VE_RELU {
                r = _mm256_max_ps(r, zero);
            } else if act == VE_RELU6 {
                r = _mm256_min_ps(_mm256_max_ps(r, zero), six);
            }
            r
        }};
    }
    macro_rules! store {
        ($p:expr, $v:expr) => {
            if nt {
                _mm256_stream_ps($p, $v);
            } else {
                _mm256_storeu_ps($p, $v);
            }
        };
    }
    // Unroll ×4 (32 elements/step) so four independent load→fma→max→store chains are in flight,
    // hiding the ~4-cycle FMA / load latency. A single 8-lane step leaves the pipe stalling on the
    // dependent store and lost to gcc's unrolled relu; four chains restore the throughput.
    while i + 32 <= n {
        // Software prefetch only when streaming from DRAM (the non-temporal regime). For an
        // L3-resident map the hardware prefetcher already has the lines, so an explicit `prefetcht0`
        // just burns an issue slot — which is part of why a cache-resident `relu` trailed gcc.
        if nt {
            _mm_prefetch(x.add(i + PF_AHEAD) as *const i8, _MM_HINT_T0);
            if use_y {
                _mm_prefetch(y.add(i + PF_AHEAD) as *const i8, _MM_HINT_T0);
            }
        }
        let r0 = compute!(0);
        let r1 = compute!(8);
        let r2 = compute!(16);
        let r3 = compute!(24);
        store!(out.add(i), r0);
        store!(out.add(i + 8), r1);
        store!(out.add(i + 16), r2);
        store!(out.add(i + 24), r3);
        i += 32;
    }
    while i + 8 <= n {
        let r = compute!(0);
        store!(out.add(i), r);
        i += 8;
    }
    // The non-temporal stores are weakly ordered; fence before the buffer is read back by anyone.
    if nt {
        _mm_sfence();
    }
    while i < n {
        let yi = if use_y { *y.add(i) } else { 0.0 };
        *out.add(i) = elem1(op, *x.add(i), yi, a, b, c);
        i += 1;
    }
}

/// One element of a Horner polynomial `((c[0]·x + c[1])·x + …)·x + c[n-1]`, the fused `mul_add` chain
/// matching the AVX2 lanes and the `r = r·x + c` source (which contracts to one `fma` per step under
/// gcc's `-ffp-contract=fast`).
#[inline]
fn horner1(x: f32, coeffs: &[f32]) -> f32 {
    let mut r = coeffs[0];
    for &c in &coeffs[1..] {
        r = r.mul_add(x, c);
    }
    r
}

/// `out[i] = poly(x[i])` for `i in 0..n`, where `poly` is the Horner evaluation of `coeffs` (highest
/// degree first; `ncoeff` terms). The 256-bit AVX2/FMA kernel a recognized `r = c0; r = r*x + c1; …;
/// out[i] = r` loop lowers to — Cranelift's vectorizer is stuck at 128-bit, and (like saxpy) the
/// streamed output goes out non-temporal for a large array, the store path gcc won't emit. `x` and
/// `out` may alias. Degenerate `ncoeff <= 1` writes the constant `coeffs[0]`.
///
/// # Safety
/// `x`/`out` valid for `n` f32; `coeffs` valid for `ncoeff` f32.
#[no_mangle]
pub unsafe extern "C" fn mercury_vhorner_f32(
    x: *const f32,
    out: *mut f32,
    n: i64,
    coeffs: *const f32,
    ncoeff: i64,
) {
    if n <= 0 || ncoeff <= 0 {
        return;
    }
    let n = n as usize;
    // SAFETY: caller guarantees coeffs valid for ncoeff f32.
    let coeffs = unsafe { std::slice::from_raw_parts(coeffs, ncoeff as usize) };
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; buffers valid for n by the caller contract.
            unsafe { vhorner_avx2(x, out, n, coeffs) };
            return;
        }
    }
    for i in 0..n {
        // SAFETY: i < n; buffers valid for n.
        unsafe { *out.add(i) = horner1(*x.add(i), coeffs) };
    }
}

/// Largest polynomial degree the AVX2 path keeps its splatted coefficients on the stack for (degree
/// MAX_HORNER-1). Far beyond any real activation/approximation polynomial; a longer one falls back to
/// the scalar path. Stack storage avoids a per-call heap allocation — which, plus the runtime-length
/// inner loop, was what made the first cut *lose* to gcc on the degree-4 poly.
const MAX_HORNER: usize = 32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vhorner_avx2(x: *const f32, out: *mut f32, n: usize, coeffs: &[f32]) {
    use std::arch::x86_64::*;
    let nc = coeffs.len();
    if nc > MAX_HORNER {
        for i in 0..n {
            *out.add(i) = horner1(*x.add(i), coeffs);
        }
        return;
    }
    // Pre-splat the coefficients once onto the **stack** (no per-call heap alloc) so the FMA chain
    // has no per-iteration broadcasts and stays L1/register-resident.
    let mut cv = [_mm256_setzero_ps(); MAX_HORNER];
    for k in 0..nc {
        cv[k] = _mm256_set1_ps(coeffs[k]);
    }
    let cv = &cv[..nc];
    let c0 = cv[0];
    // Two streams (x in, out): non-temporal once the pair spills L3.
    let nt = use_nt(n, 2);
    let mut i = 0usize;
    if nt {
        while i < n && (out.add(i) as usize) & 31 != 0 {
            *out.add(i) = horner1(*x.add(i), coeffs);
            i += 1;
        }
    }
    // Horner is a *latency-bound* dependent chain (`r = r·x + c`): each step waits on the previous
    // FMA's ~4-cycle latency. A single accumulator stalls; four chains (the old unroll) keep only four
    // FMAs in flight, under-filling the two FMA ports that retire 2/cycle (need ~8 in flight to hide the
    // latency) — so the degree-4 poly only tied gcc's own 256-bit autovec. Run **six** independent
    // 8-lane chains (48 elements/step): six accumulators + six `x` vectors + the broadcast `ck` = 13 of
    // the 16 YMM registers, so it lifts ILP toward the port limit *without* the spills eight chains hit.
    macro_rules! store {
        ($p:expr, $v:expr) => {
            if nt {
                _mm256_stream_ps($p, $v);
            } else {
                _mm256_storeu_ps($p, $v);
            }
        };
    }
    while i + 48 <= n {
        // Prefetch only when streaming from DRAM; an L3-resident poly is served by the HW prefetcher.
        if nt {
            _mm_prefetch(x.add(i + PF_AHEAD) as *const i8, _MM_HINT_T0);
        }
        let x0 = _mm256_loadu_ps(x.add(i));
        let x1 = _mm256_loadu_ps(x.add(i + 8));
        let x2 = _mm256_loadu_ps(x.add(i + 16));
        let x3 = _mm256_loadu_ps(x.add(i + 24));
        let x4 = _mm256_loadu_ps(x.add(i + 32));
        let x5 = _mm256_loadu_ps(x.add(i + 40));
        let (mut r0, mut r1, mut r2) = (c0, c0, c0);
        let (mut r3, mut r4, mut r5) = (c0, c0, c0);
        for ck in &cv[1..] {
            let ck = *ck;
            r0 = _mm256_fmadd_ps(r0, x0, ck);
            r1 = _mm256_fmadd_ps(r1, x1, ck);
            r2 = _mm256_fmadd_ps(r2, x2, ck);
            r3 = _mm256_fmadd_ps(r3, x3, ck);
            r4 = _mm256_fmadd_ps(r4, x4, ck);
            r5 = _mm256_fmadd_ps(r5, x5, ck);
        }
        store!(out.add(i), r0);
        store!(out.add(i + 8), r1);
        store!(out.add(i + 16), r2);
        store!(out.add(i + 24), r3);
        store!(out.add(i + 32), r4);
        store!(out.add(i + 40), r5);
        i += 48;
    }
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let mut r = c0;
        for ck in &cv[1..] {
            r = _mm256_fmadd_ps(r, xv, *ck);
        }
        store!(out.add(i), r);
        i += 8;
    }
    if nt {
        _mm_sfence();
    }
    while i < n {
        *out.add(i) = horner1(*x.add(i), coeffs);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The AVX2 lanes and the scalar tail/fallback must agree element-for-element, so a length that is
    /// not a multiple of 8 (and that crosses the NT alignment prologue) produces a consistent result.
    #[test]
    fn velem_tail_matches_lanes() {
        let n = 1_500_003usize; // 2-stream case is 12 MiB > NT_MIN_BYTES: forces NT, a tail, a prologue
        let x: Vec<f32> = (0..n).map(|i| (i as f32 % 19.0) - 7.0).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32 % 11.0) * 0.5 - 2.0).collect();
        let cases: &[(i64, f32, f32, f32)] = &[
            (VE_ID | VE_USE_Y, 2.0, 1.0, 0.0), // saxpy
            (VE_ID, 3.5, 0.0, 0.0),            // scale
            (VE_ID | VE_USE_Y, 1.0, 1.0, 0.0), // residual add
            (VE_ID, 1.0, 0.0, 0.75),           // bias
            (VE_RELU, 1.0, 0.0, 0.0),          // relu
            (VE_RELU, 2.0, 0.0, 1.0),          // fused linear→relu
            (VE_RELU6, 1.0, 0.0, 0.0),         // relu6
            (VE_RELU6 | VE_USE_Y, 1.0, 1.0, 0.0),
        ];
        for &(op, a, b, c) in cases {
            let mut got = vec![0.0f32; n];
            unsafe {
                mercury_velem_f32(
                    x.as_ptr(),
                    y.as_ptr(),
                    got.as_mut_ptr(),
                    n as i64,
                    a,
                    b,
                    c,
                    op,
                );
            }
            for i in 0..n {
                let yi = if op & VE_USE_Y != 0 { y[i] } else { 0.0 };
                let want = elem1(op, x[i], yi, a, b, c);
                assert_eq!(got[i].to_bits(), want.to_bits(), "op {op} i {i}");
            }
        }
    }

    /// saxpy `a·x + y` must equal the fused `mul_add` (the same single rounding gcc's
    /// `-ffp-contract=fast` produces), so the cross-language check stays exact.
    #[test]
    fn velem_saxpy_is_fma() {
        let n = 64usize; // small → cacheable path, exercising the non-NT branch
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.3 - 5.0).collect();
        let y: Vec<f32> = (0..n).map(|i| i as f32 * -0.7 + 2.0).collect();
        let mut got = vec![0.0f32; n];
        unsafe {
            mercury_velem_f32(
                x.as_ptr(),
                y.as_ptr(),
                got.as_mut_ptr(),
                n as i64,
                2.0,
                1.0,
                0.0,
                VE_ID | VE_USE_Y,
            );
        }
        for i in 0..n {
            assert_eq!(
                got[i].to_bits(),
                2.0f32.mul_add(x[i], y[i]).to_bits(),
                "i {i}"
            );
        }
    }

    /// The Horner kernel's AVX2 lanes and scalar tail must agree element-for-element across the NT
    /// boundary and a misaligned tail, for several degrees (incl. the degenerate constant `ncoeff=1`).
    #[test]
    fn vhorner_tail_matches_lanes() {
        let n = 1_500_005usize; // 12 MiB (2 streams) > NT_MIN_BYTES: forces NT, prologue, mult-of-8 tail
        let x: Vec<f32> = (0..n).map(|i| (i as f32 % 23.0) * 0.1 - 1.1).collect();
        for coeffs in [
            vec![3.0f32],                          // constant
            vec![2.0f32, -1.0],                    // linear
            vec![1e-5f32, 1e-4, 1e-3, 1e-2, 1e-1], // the deg-4 poly benchmark
        ] {
            let mut got = vec![0.0f32; n];
            unsafe {
                mercury_vhorner_f32(
                    x.as_ptr(),
                    got.as_mut_ptr(),
                    n as i64,
                    coeffs.as_ptr(),
                    coeffs.len() as i64,
                );
            }
            for i in 0..n {
                assert_eq!(
                    got[i].to_bits(),
                    horner1(x[i], &coeffs).to_bits(),
                    "deg {} i {i}",
                    coeffs.len()
                );
            }
        }
    }

    /// A scale (`b = 0`, no `VE_USE_Y`) must never dereference `y`: pass a dangling `y` and confirm
    /// the result is `a·x` with no crash.
    #[test]
    fn velem_scale_ignores_y() {
        let n = 70_000usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 % 23.0) - 11.0).collect();
        let mut got = vec![0.0f32; n];
        unsafe {
            mercury_velem_f32(
                x.as_ptr(),
                std::ptr::null(),
                got.as_mut_ptr(),
                n as i64,
                0.25,
                0.0,
                0.0,
                VE_ID,
            );
        }
        for i in 0..n {
            assert_eq!(got[i].to_bits(), (0.25f32 * x[i]).to_bits(), "i {i}");
        }
    }
}
