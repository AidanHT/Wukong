//! Streaming elementwise affine + activation — `out[i] = act(a·x[i] + b·y[i] + c)` — the **256-bit
//! AVX2** kernel a recognized streaming map loop (saxpy / scale / residual-add / bias / ReLU / ReLU6)
//! lowers to. Two reasons it beats the Cranelift-vectorized form gcc/rustc also emit:
//!
//!  1. **256-bit width.** Cranelift's generic vectorizer is stuck at 128-bit SSE (`f32x8` does not
//!     legalize), so a Wukong saxpy ran ~15% *behind* gcc's 256-bit AVX2. This kernel restores the
//!     width parity, the same play as `vmath`/`gemm`.
//!  2. **Non-temporal (streaming) stores.** Once the working set spills L3 (≥ [`NT_MIN_BYTES`] across
//!     all live arrays) the store goes out `vmovntps`, which skips the **read-for-ownership** every
//!     cacheable store pays (the CPU must pull the target line into cache before overwriting it).
//!     gcc/rustc cannot emit this automatically — they can't prove the array is large and write-once —
//!     but Wukong's domain-aware lowering *knows* the loop streams a whole tensor. The threshold is on
//!     the *total* bytes touched, not the length, so a cache-resident map keeps its normal store (where
//!     a needless `vmovntps` would lose): measured ~1.1–1.4× over gcc at real (>L3) activation-tensor
//!     sizes, and a clean tie at the 4 MiB benchmark size where the output still lives in L3.
//!
//! The interpreter marshals its abstract memory through this **identical** kernel (like `vmath`), so
//! the differential oracle stays bit-for-bit exact. NT and cacheable stores write the *same bits* (the
//! float value is identical; only the cache path differs), and the scalar tail/fallback mirrors the
//! AVX2 lanes op-for-op (the `act_avx`/`act1` pair share the `maxps`/`minps` semantics), so every lane
//! of every path agrees — pinned by `velem_tail_matches_lanes` and `velem_scalar_matches_avx`.
//!
//! **Determinism (the `_parallel` twin).** The map is pure elementwise (no reduction), so element `i`
//! depends only on its own `x[i]`/`y[i]`/`a`/`b`/`c`/`op`. The multicore entry just cuts `[0, n)` into
//! FIXED-size chunks (count independent of thread count, the same discipline as reduce.rs's `RCHUNK`)
//! and runs the identical per-span routine over each — so `serial == parallel` bit-for-bit with no
//! cross-chunk combine (pinned by `serial_matches_parallel_bit_for_bit`), and the interpreter (which
//! calls the serial form) agrees regardless of core count.

use rayon::prelude::*;

// --- op codes (shared with the recognizer in wukong_mir_build) ------------------------------------
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
/// `out = act(x · y)` — the elementwise **Hadamard product** (gating, attention masks, residual
/// scaling, RoPE). A distinct compute mode (a product of two arrays, not the affine `a·x+b·y+c`),
/// so it bypasses the affine FMA. Implies the kernel reads `y` (no need to also set `VE_USE_Y`).
/// Bit-exact across backends: `_mm256_mul_ps` lane == scalar `f32 * f32` (one IEEE rounding).
pub const VE_HADAMARD: i64 = 512;
/// `out = act(x / y)` — elementwise quotient (normalize-by-per-element-scale). Same convention as
/// [`VE_HADAMARD`]; `_mm256_div_ps` lane == scalar `f32 / f32`, both correctly rounded.
pub const VE_DIV: i64 = 1024;

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

/// The number of live f32 streams a call touches — the output and `x`, plus `y` when the op reads it
/// (`VE_USE_Y` / Hadamard / Div) — the count the non-temporal-store decision keys on (see [`use_nt`]).
/// Pulled out so the serial entry, the parallel entry, and the AVX2 kernel all agree on the NT policy.
#[inline]
fn velem_streams(op: i64) -> usize {
    if op & (VE_USE_Y | VE_HADAMARD | VE_DIV) != 0 {
        3
    } else {
        2
    }
}

/// Fixed chunk size (elements) for the multicore [`wukong_velem_f32_parallel`] — a constant
/// independent of thread count, so the chunk decomposition is deterministic (the same discipline as
/// reduce.rs's `RCHUNK`). For a pure elementwise map bit-exactness does not actually depend on the
/// boundary (there is no cross-chunk combine), so this is purely a load-balancing knob: a multiple of
/// 8 keeps every chunk boundary 8-lane aligned (and 32-byte aligned when the base is), so the AVX2
/// body needs no extra per-chunk realignment. 8192 f32 = 32 KB gives plenty of chunks for rayon's
/// work-stealing to balance the P+E hybrid at the sizes past [`velem_par_min`].
const VCHUNK: usize = 8192;

/// Minimum element count for [`wukong_velem_f32_parallel`] to actually spread work across cores.
/// Below it the fork-join wake/join cost (worst on the P+E hybrid, where a parked E-core is slow to
/// arrive) outweighs a single core's streaming map — which already saturates a good fraction of store
/// bandwidth on its own — so the call runs the serial kernel instead. Env-overridable
/// (`WUKONG_VELEM_PAR_MIN`, read once) so the crossover stays A/B-measurable on other machines;
/// default ~256 Ki elements (a ~1 MiB output), to be tuned centrally later.
fn velem_par_min() -> usize {
    use std::sync::OnceLock;
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("WUKONG_VELEM_PAR_MIN")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(256 * 1024)
    })
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
    let base = if op & VE_HADAMARD != 0 {
        x * y
    } else if op & VE_DIV != 0 {
        x / y
    } else if op & VE_USE_Y == 0 && a == 1.0 && c == 0.0 {
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
pub unsafe extern "C" fn wukong_velem_f32(
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
    // The non-temporal-store decision keys on the whole working set (see `use_nt`), so make it once
    // here from the total length and pass it down — the parallel entry makes the *same* decision from
    // the same total `n`, keeping the store policy identical across the serial and parallel forms.
    let nt = use_nt(n, velem_streams(op));
    // SAFETY: buffers valid for n by the caller contract (y only when the op reads it).
    unsafe { velem_span(x, y, out, n, a, b, c, op, nt) };
}

/// Run the streaming map over `[0, n)` at the given base pointers, with the non-temporal store
/// decision `nt` supplied by the caller (decided from the *total* working set — see [`use_nt`] — so a
/// parallel call can split `[0, n)` into chunks that all inherit the whole-array store policy rather
/// than each small chunk re-deciding on its own sub-length). Dispatches to the 256-bit AVX2/FMA kernel
/// when available, else a scalar fallback (which never streams, so it ignores `nt`). The result is
/// bit-for-bit identical for every element however `[0, n)` is chunked: the op is elementwise, so
/// element `i` depends only on `x[i]`/`y[i]`/`a`/`b`/`c`/`op`, the AVX2 lanes agree with the scalar
/// tail (see [`elem1`]), and `nt` only changes the cache path, never the stored bits.
///
/// # Safety
/// `x`/`out` valid for `n` f32; `y` valid for `n` f32 when the op reads it (`VE_USE_Y`/Hadamard/Div).
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn velem_span(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
    a: f32,
    b: f32,
    c: f32,
    op: i64,
    nt: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; buffers valid for n by the caller contract.
            unsafe { velem_avx2(x, y, out, n, a, b, c, op, nt) };
            return;
        }
    }
    // The scalar fallback never streams, so `nt` is consulted only by the AVX2 path above; bind it so
    // the parameter is used on every target (no-op on x86_64, silences the unused warning elsewhere).
    let _ = nt;
    let use_y = op & (VE_USE_Y | VE_HADAMARD | VE_DIV) != 0;
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
#[allow(clippy::too_many_arguments)]
unsafe fn velem_avx2(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
    a: f32,
    b: f32,
    c: f32,
    op: i64,
    nt: bool,
) {
    use std::arch::x86_64::*;
    let va = _mm256_set1_ps(a);
    let vb = _mm256_set1_ps(b);
    let vc = _mm256_set1_ps(c);
    let zero = _mm256_setzero_ps();
    let six = _mm256_set1_ps(6.0);
    // Binary compute modes (Hadamard `x·y` / quotient `x/y`) read `y` and bypass the affine FMA.
    let had = op & VE_HADAMARD != 0;
    let div = op & VE_DIV != 0;
    let use_y = had || div || op & VE_USE_Y != 0;
    let act = op & 0xff;
    // Identity affine (`a == 1`, `c == 0`, no `y`): the FMA degenerates to `x`, so skip it — the case a
    // bare `relu`/`relu6`/copy hits. Hoisted here so the hot loop branches on a single invariant flag
    // (matching `elem1`'s scalar fast path), turning a recognized ReLU into gcc's bare `maxps` loop at
    // true 256-bit width instead of paying a wasted multiply per 8 lanes. (Never set for a binary op.)
    let id_affine = !use_y && a == 1.0 && c == 0.0;
    // `nt` (whether to use non-temporal stores) is decided by the caller from the *total* working set
    // — output + x (+ y when read) — so a serial call and every chunk of a parallel call stream with
    // the identical store policy (see `velem_streams`/`use_nt`).
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
            let mut r = if had {
                _mm256_mul_ps(xv, _mm256_loadu_ps(y.add(i + $off)))
            } else if div {
                _mm256_div_ps(xv, _mm256_loadu_ps(y.add(i + $off)))
            } else if id_affine {
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

/// Multicore streaming elementwise map — **bit-identical** to [`wukong_velem_f32`]. The op is pure
/// elementwise (`out[i] = act(a·x[i] (+ b·y[i]) + c)`, no reduction), so element `i`'s result depends
/// only on its own inputs. Cutting `[0, n)` into FIXED-size [`VCHUNK`] chunks (count independent of
/// thread count — the reduce.rs `RCHUNK` discipline) and running each chunk through the same
/// [`velem_span`] therefore reproduces, for every element, the exact bits the whole-range serial pass
/// writes: there is no cross-chunk combine, so the result never depends on how many cores ran it, and
/// the interpreter (which calls the serial form) agrees with this `@parallel` path. The non-temporal
/// store decision is made ONCE from the total `n` (as in the serial entry) and shared by every chunk,
/// so the store policy matches too — and NT stores write the same bits regardless.
///
/// Below [`velem_par_min`] elements the fork-join wake/join cost outweighs a single-core streaming
/// map, so the call falls back to the serial kernel (still bit-identical).
///
/// # Safety
/// `x`/`out` valid for `n` f32; `y` valid for `n` f32 when the op reads it (`VE_USE_Y`/Hadamard/Div).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wukong_velem_f32_parallel(
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
    // Small arrays don't amortize the pool wake — run the serial kernel (bit-identical).
    if n < velem_par_min() {
        // SAFETY: same contract as this function.
        unsafe { wukong_velem_f32(x, y, out, n as i64, a, b, c, op) };
        return;
    }
    let use_y = op & (VE_USE_Y | VE_HADAMARD | VE_DIV) != 0;
    // Whole-array NT decision, shared by every chunk (see the serial entry).
    let nt = use_nt(n, velem_streams(op));
    let nchunks = n.div_ceil(VCHUNK);
    // Raw pointers cross the rayon closure boundary as integers (the same pattern as the parallel
    // reduce/norm); each chunk touches a disjoint output sub-slice and `y` is shared read-only.
    let (xa, ya, oa) = (x as usize, y as usize, out as usize);
    (0..nchunks).into_par_iter().for_each(|ck| {
        let lo = ck * VCHUNK;
        let hi = ((ck + 1) * VCHUNK).min(n);
        // SAFETY: disjoint output sub-slice [lo, hi) ⊆ [0, n); x/out valid for n. `y` is offset only
        // when the op reads it — a scale/relu passes a null (or aliased) `y` the kernel never touches,
        // and `null.add(lo)` would be UB, so an unread `y` is left unoffset.
        unsafe {
            let xp = (xa as *const f32).add(lo);
            let yp = if use_y {
                (ya as *const f32).add(lo)
            } else {
                ya as *const f32
            };
            let outp = (oa as *mut f32).add(lo);
            velem_span(xp, yp, outp, hi - lo, a, b, c, op, nt);
        }
    });
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
pub unsafe extern "C" fn wukong_vhorner_f32(
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
            (VE_HADAMARD, 1.0, 1.0, 0.0),      // hadamard x*y (a/b/c ignored)
            (VE_RELU | VE_HADAMARD, 1.0, 1.0, 0.0), // relu(x*y)
        ];
        for &(op, a, b, c) in cases {
            let mut got = vec![0.0f32; n];
            unsafe {
                wukong_velem_f32(
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
                // Binary ops (Hadamard/Div) read `y` too, matching the kernel's `use_y`.
                let yi = if op & (VE_USE_Y | VE_HADAMARD | VE_DIV) != 0 {
                    y[i]
                } else {
                    0.0
                };
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
            wukong_velem_f32(
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
                wukong_vhorner_f32(
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
            wukong_velem_f32(
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

    /// Elementwise quotient `out = x / y`: the AVX2 `_mm256_div_ps` lanes and the scalar `f32 / f32`
    /// twin must agree bit-for-bit (both correctly-rounded IEEE division). `y` is kept away from zero
    /// so the test exercises the ordinary quotient (the kernel handles `x/0` → ±inf identically too,
    /// but that is not the point being pinned here). Crosses the NT boundary + a non-mult-of-8 tail.
    #[test]
    fn velem_div_matches_lanes() {
        let n = 1_500_001usize; // 3-stream (x,y,out) = 18 MiB > NT_MIN_BYTES: NT + prologue + tail
        let x: Vec<f32> = (0..n).map(|i| (i as f32 % 17.0) - 8.0).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32 % 13.0) + 1.5).collect(); // 1.5..=13.5, never 0
        let mut got = vec![0.0f32; n];
        unsafe {
            wukong_velem_f32(
                x.as_ptr(),
                y.as_ptr(),
                got.as_mut_ptr(),
                n as i64,
                1.0,
                1.0,
                0.0,
                VE_DIV,
            );
        }
        for i in 0..n {
            let want = elem1(VE_DIV, x[i], y[i], 1.0, 1.0, 0.0);
            assert_eq!(got[i].to_bits(), want.to_bits(), "div i {i}");
        }
    }

    /// The multicore entry must equal the serial one **bit-for-bit** for every element and op — the
    /// pure-elementwise map has no cross-chunk combine, so chunking is transparent. Covers: `n` below
    /// the parallel gate (falls back to serial), `n` at/above it (real multicore), ragged tails
    /// (non-multiple-of-8), lengths whose last `VCHUNK` chunk is partial (chunk-boundary straddling),
    /// and the NT threshold crossed differently by 2-stream vs 3-stream ops. Runs with the default
    /// `WUKONG_VELEM_PAR_MIN` (~256 Ki) so both the serial-fallback and the parallel branch fire.
    #[test]
    fn serial_matches_parallel_bit_for_bit() {
        // Reuse the op set from `velem_tail_matches_lanes`, plus a Div (3-stream binary) case.
        let cases: &[(i64, f32, f32, f32)] = &[
            (VE_ID | VE_USE_Y, 2.0, 1.0, 0.0), // saxpy (3 streams → crosses NT sooner)
            (VE_ID, 3.5, 0.0, 0.0),            // scale (2 streams; y unread)
            (VE_ID | VE_USE_Y, 1.0, 1.0, 0.0), // residual add — the GPT-2 layer's hot loop
            (VE_ID, 1.0, 0.0, 0.75),           // bias (2 streams)
            (VE_RELU, 1.0, 0.0, 0.0),          // relu (identity-affine fast path)
            (VE_RELU, 2.0, 0.0, 1.0),          // fused linear→relu
            (VE_RELU6, 1.0, 0.0, 0.0),         // relu6
            (VE_RELU6 | VE_USE_Y, 1.0, 1.0, 0.0),
            (VE_HADAMARD, 1.0, 1.0, 0.0),      // hadamard x*y
            (VE_RELU | VE_HADAMARD, 1.0, 1.0, 0.0), // relu(x*y)
            (VE_DIV, 1.0, 1.0, 0.0),           // quotient x/y
        ];
        // Sizes: below the ~256 Ki gate (serial fallback), then above it — 300 003 is past the gate
        // but under NT for both stream counts; 900 001 crosses NT for 3-stream ops (10.8 MiB) but not
        // 2-stream (7.2 MiB); 1 500 001 crosses NT for both. Odd sizes force ragged tails and partial
        // final chunks; the divisor never hits 0 (y is kept ≥ 1.5).
        for &n in &[
            1_000usize, 8_195, 65_537, 262_144, 262_151, 300_003, 900_001, 1_500_001,
        ] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 % 19.0) - 7.0).collect();
            let y: Vec<f32> = (0..n).map(|i| (i as f32 % 13.0) + 1.5).collect(); // 1.5..=14.5
            for &(op, a, b, c) in cases {
                let mut s = vec![0.0f32; n];
                let mut p = vec![0.0f32; n];
                unsafe {
                    wukong_velem_f32(x.as_ptr(), y.as_ptr(), s.as_mut_ptr(), n as i64, a, b, c, op);
                    wukong_velem_f32_parallel(
                        x.as_ptr(),
                        y.as_ptr(),
                        p.as_mut_ptr(),
                        n as i64,
                        a,
                        b,
                        c,
                        op,
                    );
                }
                for i in 0..n {
                    assert_eq!(
                        s[i].to_bits(),
                        p[i].to_bits(),
                        "serial != parallel n={n} op={op} i={i}"
                    );
                }
            }
        }
    }
}
