//! Integer-input **dequantization** — `out[i] = act((q[i] as f32)·scale)` — the 256-bit AVX2 kernel a
//! recognized dequant loop lowers to. This is the ubiquitous boundary of the quantized dataflow: the
//! `(q as f32)·scale` that turns an `[i8]`/`[u8]` quantized weight/activation or an `[i32]` GEMM
//! accumulator back into `f32`, plus the **per-channel** sibling (`scale[c]` per output column, the
//! shape a quantized `nn.Linear` writeback takes).
//!
//! **Why it beats gcc/rustc.** The dequant is a *widen* (`i8`/`i32` → `i32` → `f32`) followed by a
//! scale multiply. gcc/rustc `-O3 -march=native` leave the widening `cvt` chain **scalar** (the load
//! type is narrower than the compute type, and the compiler does not prove the pattern is a clean
//! streaming map), so the loop retires one element per `cvtsi2ss`+`mulss`. This kernel folds the whole
//! widen+scale into one 256-bit pass — `_mm256_cvtepi8_epi32` / `_mm256_cvtepi32_ps` / `_mm256_mul_ps`,
//! eight lanes per step — and, once the working set spills L3, streams the `f32` output with
//! **non-temporal stores** (`vmovntps`), skipping the read-for-ownership traffic a cacheable store
//! pays. gcc cannot emit that automatically; Mercury's domain-aware lowering *knows* the loop streams a
//! whole tensor once.
//!
//! **Bit-exactness — the cleanest correctness story there is.** `int → f32` is *exact* for the
//! magnitudes that arise here and correctly rounded (round-to-nearest-even) otherwise, and
//! `_mm256_cvtepi32_ps` rounds identically to Rust's `as f32`; `_mm256_mul_ps` is one IEEE rounding, the
//! same as scalar `f32 * f32`. So the 256-bit lanes equal the scalar twin **bit-for-bit** — no
//! reassociation, no tolerance (there is no accumulation to reorder). The interpreter marshals its
//! abstract memory through this **identical** kernel (unmarshalling the `[i8]`/`[i32]` array to real
//! bytes for the call), so the differential oracle stays exact. NT and cacheable stores write the same
//! bits (only the cache path differs). The `id`/`ReLU` activations are applied in-vector (`_mm256_max_ps`
//! matches the scalar `if v>0 {v} else {0}` lane-for-lane, the same equivalence `velem` relies on);
//! `GELU`/`SiLU` reuse the shared scalar twins `vmath::{gelu1,silu1}` per lane on the dequantized bits,
//! so the fused activation equals the unfused one exactly. Pinned by `simd_equals_scalar_twin` and
//! `dequant_matches_rational_reference`.

use rayon::prelude::*;

// --- op codes (shared with the recognizer in mercury_mir_build) ------------------------------------
// The low byte is the activation; a higher nibble selects the input element width. A recognized dequant
// loop's element type and (optional) activation wrapper choose the code; the interpreter and the native
// backend pass the identical value, so the two agree by construction.
/// `out = (q as f32)·scale`, no activation.
pub const DQ_ID: i64 = 0;
/// `out = max((q as f32)·scale, 0)` — ReLU.
pub const DQ_RELU: i64 = 1;
/// `out = gelu((q as f32)·scale)` — tanh-approx GELU (reuses `vmath::gelu1`).
pub const DQ_GELU: i64 = 2;
/// `out = silu((q as f32)·scale)` — SiLU / swish (reuses `vmath::silu1`).
pub const DQ_SILU: i64 = 3;
/// Mask isolating the activation byte.
const DQ_ACT_MASK: i64 = 0xff;

/// Input element is signed `i8` (quantized weights, symmetric quant). Widen with `_mm256_cvtepi8_epi32`.
pub const DQ_I8: i64 = 0 << 8;
/// Input element is unsigned `u8` (asymmetric-quant activations). Widen with `_mm256_cvtepu8_epi32`.
pub const DQ_U8: i64 = 1 << 8;
/// Input element is `i32` (a quantized GEMM accumulator). Loaded directly, no byte-widen.
pub const DQ_I32: i64 = 2 << 8;
/// Mask isolating the input-width field.
const DQ_WIDTH_MASK: i64 = 0xff << 8;

/// Total streamed bytes (input + output) at/above which the store goes non-temporal. Mirrors the
/// crossover `velem` uses: below it a cacheable store keeps the output hot (a per-core `@parallel` chunk
/// may be re-read); above it `vmovntps` streams the array out once and skips the read-for-ownership the
/// cacheable store pays. ~10 MiB ≈ this machine's L3.
const NT_MIN_BYTES: usize = 10 * 1024 * 1024;

/// Scalar activation, mirroring the AVX2 `maxps` semantics exactly: `maxps(v,0)` is `(v > 0) ? v : 0`
/// (returns the second operand for `±0`/NaN), so the `if` form agrees lane-for-lane; GELU/SiLU are the
/// shared `vmath` scalar twins the fused kernel applies per lane. `act` is the pre-masked activation byte.
#[inline]
fn act1(act: i64, v: f32) -> f32 {
    match act {
        DQ_RELU => {
            if v > 0.0 {
                v
            } else {
                0.0
            }
        }
        DQ_GELU => crate::vmath::gelu1(v),
        DQ_SILU => crate::vmath::silu1(v),
        _ => v,
    }
}

/// Dequantize one element given its already-widened `i32` value: `act((qi as f32)·scale)`. The single
/// source of truth the scalar path and the AVX2 tail share (and the AVX2 body proves equivalent to).
#[inline]
fn deq1(act: i64, qi: i32, scale: f32) -> f32 {
    act1(act, (qi as f32) * scale)
}

/// Read the `i`-th input element of `q` as an `i32`, per the width field of `op`. `q` is a raw byte
/// pointer (the kernel ABI takes `*const u8` so one signature covers all widths); the width code casts
/// it to the right element type before indexing, so the stride matches the source array's layout.
///
/// # Safety
/// `q` must be valid for the element at `i` in the width `op` selects.
#[inline]
unsafe fn load_i32(q: *const u8, i: usize, width: i64) -> i32 {
    match width {
        DQ_U8 => *q.add(i) as i32,
        DQ_I32 => *(q as *const i32).add(i),
        // DQ_I8 (and any unknown code, defensively) → signed byte.
        _ => *(q as *const i8).add(i) as i32,
    }
}

/// `out[i] = act((q[i] as f32)·scale)` for `i in 0..n`, scalar. The no-AVX2 fallback and the tail of the
/// AVX2 path; the differential reference for the unit tests.
///
/// # Safety
/// `q` valid for `n` elements of the width `op` selects; `out` valid for `n` `f32`.
unsafe fn dequant_scalar(q: *const u8, out: *mut f32, n: usize, scale: f32, op: i64) {
    let act = op & DQ_ACT_MASK;
    let width = op & DQ_WIDTH_MASK;
    for i in 0..n {
        *out.add(i) = deq1(act, load_i32(q, i, width), scale);
    }
}

/// Whether an `n`-element dequant whose input element is `in_bytes` wide should use non-temporal stores
/// — true once input + output spill L3 (see [`NT_MIN_BYTES`]).
#[inline]
fn use_nt(n: usize, in_bytes: usize) -> bool {
    n.saturating_mul(in_bytes + 4) >= NT_MIN_BYTES
}

/// Widen eight input elements at `q.add(i)` to an `__m256i` of eight `i32`, per the width field. Used by
/// the compute-bound `GELU`/`SiLU` paths (the memory-bound `id`/`ReLU` paths widen inline via macros).
///
/// # Safety
/// `q` valid for 8 elements at `i`; requires `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn widen8(q: *const u8, i: usize, width: i64) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    match width {
        DQ_U8 => _mm256_cvtepu8_epi32(_mm_loadl_epi64((q as *const u8).add(i) as *const __m128i)),
        DQ_I32 => _mm256_loadu_si256((q as *const i32).add(i) as *const __m256i),
        _ => _mm256_cvtepi8_epi32(_mm_loadl_epi64((q as *const i8).add(i) as *const __m128i)),
    }
}

/// The AVX2 dequant: `out[i] = act((q[i] as f32)·scale)`, 8 lanes/step, 4×-unrolled (32 elements/step)
/// so four independent widen→cvt→mul→store chains hide the cvt/load latency, with non-temporal stores
/// for a large output. Bit-equal to [`dequant_scalar`].
///
/// The `id`/`ReLU` hot path (the memory-bound one) is written as **macro-expanded flat loops** — one per
/// input width — so the widen intrinsic is fixed at compile time and there is no per-element helper call
/// (a `#[target_feature]` `fn` will not inline on stable Rust, and an out-of-line vector call every 8
/// elements serializes the load/store stream, capping memory-level parallelism well below the gcc loop).
/// The `nt`/`relu` flags are loop-invariant, so LLVM unswitches them. `GELU`/`SiLU` are compute-bound
/// (the transcendental dominates), so they keep the simpler per-lane scalar-twin path.
///
/// # Safety
/// `q`/`out` valid for `n` elements; requires `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequant_avx2(q: *const u8, out: *mut f32, n: usize, scale: f32, op: i64) {
    use std::arch::x86_64::*;
    let act = op & DQ_ACT_MASK;
    let width = op & DQ_WIDTH_MASK;
    let in_bytes = if width == DQ_I32 { 4 } else { 1 };
    let vscale = _mm256_set1_ps(scale);
    let zero = _mm256_setzero_ps();
    let nt = use_nt(n, in_bytes);

    // Load 8 input elements at index `$i` and widen to eight `i32` lanes — the width is fixed per arm, so
    // the match on width is hoisted out of the loop (no per-element branch, no call).
    macro_rules! w_i8 {
        ($i:expr) => {
            _mm256_cvtepi8_epi32(_mm_loadl_epi64((q as *const i8).add($i) as *const __m128i))
        };
    }
    macro_rules! w_u8 {
        ($i:expr) => {
            _mm256_cvtepu8_epi32(_mm_loadl_epi64((q as *const u8).add($i) as *const __m128i))
        };
    }
    macro_rules! w_i32 {
        ($i:expr) => {
            _mm256_loadu_si256((q as *const i32).add($i) as *const __m256i)
        };
    }
    // One full pass for a fixed widen macro `$w`. `relu`/`nt` are loop-invariant (LLVM unswitches them).
    macro_rules! deq_pass {
        ($w:ident, $relu:expr) => {{
            let mut i = 0usize;
            // Peel a scalar prologue until `out` is 32-byte aligned (`vmovntps` faults if misaligned).
            if nt {
                while i < n && (out.add(i) as usize) & 31 != 0 {
                    *out.add(i) = deq1(act, load_i32(q, i, width), scale);
                    i += 1;
                }
            }
            while i + 32 <= n {
                let mut d0 = _mm256_mul_ps(_mm256_cvtepi32_ps($w!(i)), vscale);
                let mut d1 = _mm256_mul_ps(_mm256_cvtepi32_ps($w!(i + 8)), vscale);
                let mut d2 = _mm256_mul_ps(_mm256_cvtepi32_ps($w!(i + 16)), vscale);
                let mut d3 = _mm256_mul_ps(_mm256_cvtepi32_ps($w!(i + 24)), vscale);
                if $relu {
                    d0 = _mm256_max_ps(d0, zero);
                    d1 = _mm256_max_ps(d1, zero);
                    d2 = _mm256_max_ps(d2, zero);
                    d3 = _mm256_max_ps(d3, zero);
                }
                if nt {
                    _mm256_stream_ps(out.add(i), d0);
                    _mm256_stream_ps(out.add(i + 8), d1);
                    _mm256_stream_ps(out.add(i + 16), d2);
                    _mm256_stream_ps(out.add(i + 24), d3);
                } else {
                    _mm256_storeu_ps(out.add(i), d0);
                    _mm256_storeu_ps(out.add(i + 8), d1);
                    _mm256_storeu_ps(out.add(i + 16), d2);
                    _mm256_storeu_ps(out.add(i + 24), d3);
                }
                i += 32;
            }
            while i + 8 <= n {
                let mut d = _mm256_mul_ps(_mm256_cvtepi32_ps($w!(i)), vscale);
                if $relu {
                    d = _mm256_max_ps(d, zero);
                }
                if nt {
                    _mm256_stream_ps(out.add(i), d);
                } else {
                    _mm256_storeu_ps(out.add(i), d);
                }
                i += 8;
            }
            if nt {
                _mm_sfence();
            }
            while i < n {
                *out.add(i) = deq1(act, load_i32(q, i, width), scale);
                i += 1;
            }
        }};
    }

    if act == DQ_ID || act == DQ_RELU {
        let relu = act == DQ_RELU;
        match width {
            DQ_U8 => deq_pass!(w_u8, relu),
            DQ_I32 => deq_pass!(w_i32, relu),
            _ => deq_pass!(w_i8, relu), // DQ_I8 (and any unknown code, defensively)
        }
    } else {
        // GELU / SiLU — compute-bound: the transcendental twin per lane dominates, so a cacheable store
        // and the shared `widen8` helper are fine. Bit-exact with the unfused activation.
        let mut i = 0usize;
        while i + 8 <= n {
            let d = _mm256_mul_ps(_mm256_cvtepi32_ps(widen8(q, i, width)), vscale);
            let mut t = [0f32; 8];
            _mm256_storeu_ps(t.as_mut_ptr(), d);
            for v in &mut t {
                *v = act1(act, *v);
            }
            _mm256_storeu_ps(out.add(i), _mm256_loadu_ps(t.as_ptr()));
            i += 8;
        }
        while i < n {
            *out.add(i) = deq1(act, load_i32(q, i, width), scale);
            i += 1;
        }
    }
}

/// Dequantize `n` integer elements to `f32`: `out[i] = act((q[i] as f32)·scale)`. Uses the 256-bit AVX2
/// kernel when available (8 lanes/step + non-temporal stores for a large output), else a scalar
/// fallback. `q` is a raw byte pointer whose element width is encoded in `op` (see `DQ_I8`/`DQ_U8`/
/// `DQ_I32`); `scale` is the per-tensor dequant scale; the low byte of `op` is the activation.
///
/// # Safety
/// `q` must be valid for `n` elements of the width `op` selects; `out` valid for `n` `f32`.
#[no_mangle]
pub unsafe extern "C" fn mercury_dequant_f32(
    q: *const u8,
    out: *mut f32,
    n: i64,
    scale: f32,
    op: i64,
) {
    if n <= 0 {
        return;
    }
    let n = n as usize;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detected; buffers valid for n by the caller contract.
            unsafe { dequant_avx2(q, out, n, scale, op) };
            return;
        }
    }
    dequant_scalar(q, out, n, scale, op);
}

/// A raw address shuttled across the rayon boundary as an integer (the disjoint-chunk pattern the other
/// `_parallel` kernels use).
#[derive(Clone, Copy)]
struct Addr(usize);
unsafe impl Send for Addr {}
unsafe impl Sync for Addr {}

/// Multicore dequant — **bit-identical** to [`mercury_dequant_f32`]. The map is elementwise, so `[0,n)`
/// splits into disjoint contiguous chunks across cores, each running the identical per-chunk kernel; the
/// result is independent of thread count and the interpreter (which calls the serial form) agrees
/// exactly. Each chunk is 8-aligned so its NT-store prologue is trivial and never overlaps a neighbor.
///
/// # Safety
/// Same as [`mercury_dequant_f32`].
#[no_mangle]
pub unsafe extern "C" fn mercury_dequant_f32_parallel(
    q: *const u8,
    out: *mut f32,
    n: i64,
    scale: f32,
    op: i64,
) {
    if n <= 0 {
        return;
    }
    let n = n as usize;
    let in_bytes = if op & DQ_WIDTH_MASK == DQ_I32 { 4 } else { 1 };
    let (qa, oa) = (Addr(q as usize), Addr(out as usize));
    let workers = rayon::current_num_threads().max(1).min(n.div_ceil(8).max(1));
    // 8-aligned chunk bounds so each worker's output sub-slice starts 32-byte-aligned iff `out` is
    // (the NT prologue then peels ≤7 elements, disjointly from the previous chunk's tail).
    let chunk = n.div_ceil(workers);
    let chunk = chunk.div_ceil(8) * 8;
    (0..workers).into_par_iter().for_each(|w| {
        let start = w * chunk;
        if start >= n {
            return;
        }
        let end = (start + chunk).min(n);
        // SAFETY: disjoint output range [start,end); pointers valid for the declared extent.
        unsafe {
            let q = qa.0 as *const u8;
            let out = oa.0 as *mut f32;
            let len = (end - start) as i64;
            mercury_dequant_f32(q.add(start * in_bytes), out.add(start), len, scale, op);
        }
    });
}

// --- per-channel dequant -----------------------------------------------------------------------------

/// One row of the per-channel dequant: `out[j] = act((q[j] as f32)·scale[j])` for `j in 0..cols`, where
/// `scale` is the per-column vector (length `cols`). Scalar reference / AVX2 tail.
///
/// # Safety
/// `q`/`out` valid for `cols` elements; `scale` for `cols` `f32`.
unsafe fn perchan_row_scalar(q: *const u8, out: *mut f32, cols: usize, scale: *const f32, op: i64) {
    let act = op & DQ_ACT_MASK;
    let width = op & DQ_WIDTH_MASK;
    for j in 0..cols {
        *out.add(j) = deq1(act, load_i32(q, j, width), *scale.add(j));
    }
}

/// One row of the per-channel dequant on the AVX2 path: like [`dequant_avx2`] but the scale is a
/// per-column **vector** loaded from `scale.add(j)` (the norm-affine γ broadcast pattern) rather than a
/// splatted scalar. Flat macro-expanded loop per input width (no per-element `#[target_feature]` call —
/// see [`dequant_avx2`]); bit-equal to [`perchan_row_scalar`].
///
/// # Safety
/// `q`/`out` valid for `cols` elements; `scale` for `cols` `f32`; requires `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn perchan_row_avx2(
    q: *const u8,
    out: *mut f32,
    cols: usize,
    scale: *const f32,
    op: i64,
    nt: bool,
) {
    use std::arch::x86_64::*;
    let act = op & DQ_ACT_MASK;
    let width = op & DQ_WIDTH_MASK;
    let zero = _mm256_setzero_ps();

    macro_rules! w_i8 {
        ($i:expr) => {
            _mm256_cvtepi8_epi32(_mm_loadl_epi64((q as *const i8).add($i) as *const __m128i))
        };
    }
    macro_rules! w_u8 {
        ($i:expr) => {
            _mm256_cvtepu8_epi32(_mm_loadl_epi64((q as *const u8).add($i) as *const __m128i))
        };
    }
    macro_rules! w_i32 {
        ($i:expr) => {
            _mm256_loadu_si256((q as *const i32).add($i) as *const __m256i)
        };
    }
    macro_rules! row_pass {
        ($w:ident, $relu:expr) => {{
            let mut j = 0usize;
            if nt {
                while j < cols && (out.add(j) as usize) & 31 != 0 {
                    *out.add(j) = deq1(act, load_i32(q, j, width), *scale.add(j));
                    j += 1;
                }
            }
            while j + 8 <= cols {
                let mut d =
                    _mm256_mul_ps(_mm256_cvtepi32_ps($w!(j)), _mm256_loadu_ps(scale.add(j)));
                if $relu {
                    d = _mm256_max_ps(d, zero);
                }
                if nt {
                    _mm256_stream_ps(out.add(j), d);
                } else {
                    _mm256_storeu_ps(out.add(j), d);
                }
                j += 8;
            }
            if nt {
                _mm_sfence();
            }
            while j < cols {
                *out.add(j) = deq1(act, load_i32(q, j, width), *scale.add(j));
                j += 1;
            }
        }};
    }

    if act == DQ_ID || act == DQ_RELU {
        let relu = act == DQ_RELU;
        match width {
            DQ_U8 => row_pass!(w_u8, relu),
            DQ_I32 => row_pass!(w_i32, relu),
            _ => row_pass!(w_i8, relu),
        }
    } else {
        // GELU / SiLU — compute-bound: scalar-twin per lane (bit-exact with the unfused activation).
        let mut j = 0usize;
        while j + 8 <= cols {
            let d = _mm256_mul_ps(_mm256_cvtepi32_ps(widen8(q, j, width)), _mm256_loadu_ps(scale.add(j)));
            let mut t = [0f32; 8];
            _mm256_storeu_ps(t.as_mut_ptr(), d);
            for v in &mut t {
                *v = act1(act, *v);
            }
            _mm256_storeu_ps(out.add(j), _mm256_loadu_ps(t.as_ptr()));
            j += 8;
        }
        while j < cols {
            *out.add(j) = deq1(act, load_i32(q, j, width), *scale.add(j));
            j += 1;
        }
    }
}

/// Per-channel dequant of a `[rows, cols]` row-major matrix: `out[i*cols+j] = act((q[i*cols+j] as f32)·
/// scale[j])`. `scale` is the per-output-channel weight scale (length `cols`), broadcast down every row
/// — the shape a quantized `nn.Linear` / per-channel-quant writeback takes. Detects AVX2 once and runs
/// the per-row kernel; NT stores kick in once the whole matrix spills L3.
///
/// # Safety
/// `q`/`out` valid for `rows*cols` elements; `scale` for `cols` `f32`.
#[no_mangle]
pub unsafe extern "C" fn mercury_dequant_perchan_f32(
    q: *const u8,
    out: *mut f32,
    rows: i64,
    cols: i64,
    scale: *const f32,
    op: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    let in_bytes = if op & DQ_WIDTH_MASK == DQ_I32 { 4 } else { 1 };
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            let nt = use_nt(rows * cols, in_bytes);
            for i in 0..rows {
                // SAFETY: row i in bounds; features detected.
                perchan_row_avx2(q.add(i * cols * in_bytes), out.add(i * cols), cols, scale, op, nt);
            }
            return;
        }
    }
    for i in 0..rows {
        perchan_row_scalar(q.add(i * cols * in_bytes), out.add(i * cols), cols, scale, op);
    }
}

/// Multicore per-channel dequant — **bit-identical** to [`mercury_dequant_perchan_f32`] (rows are
/// independent; each maps to a core running the identical per-row kernel with the same `scale` vector).
/// The interpreter calls the serial form, so serial == parallel == interp.
///
/// # Safety
/// Same as [`mercury_dequant_perchan_f32`].
#[no_mangle]
pub unsafe extern "C" fn mercury_dequant_perchan_f32_parallel(
    q: *const u8,
    out: *mut f32,
    rows: i64,
    cols: i64,
    scale: *const f32,
    op: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    let in_bytes = if op & DQ_WIDTH_MASK == DQ_I32 { 4 } else { 1 };
    let (qa, oa, sa) = (Addr(q as usize), Addr(out as usize), Addr(scale as usize));
    #[cfg(target_arch = "x86_64")]
    let avx2 = is_x86_feature_detected!("avx2");
    #[cfg(target_arch = "x86_64")]
    let nt = use_nt(rows * cols, in_bytes);
    (0..rows).into_par_iter().for_each(|i| {
        // SAFETY: disjoint output row i; pointers valid for the declared extents by contract.
        unsafe {
            let q = (qa.0 as *const u8).add(i * cols * in_bytes);
            let out = (oa.0 as *mut f32).add(i * cols);
            let scale = sa.0 as *const f32;
            #[cfg(target_arch = "x86_64")]
            {
                if avx2 {
                    perchan_row_avx2(q, out, cols, scale, op, nt);
                    return;
                }
            }
            perchan_row_scalar(q, out, cols, scale, op);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic, varied fills (no RNG). i8 spans the full signed range; u8 the full unsigned range;
    // i32 includes magnitudes past 2^24 so the `i32 → f32` rounding path (round-to-nearest-even) is
    // exercised, pinning that `_mm256_cvtepi32_ps` rounds like Rust's `as f32`.
    fn fill_i8(n: usize) -> Vec<i8> {
        (0..n).map(|i| (((i * 53 + 7) % 256) as i32 - 128) as i8).collect()
    }
    fn fill_u8(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * 37 + 11) % 256) as u8).collect()
    }
    fn fill_i32(n: usize) -> Vec<i32> {
        // A mix of small values and large magnitudes (both signs) past the 2^24 exact-f32 boundary.
        (0..n)
            .map(|i| {
                let base = (i as i32 % 97) - 48;
                if i % 5 == 0 {
                    base.wrapping_mul(1_000_003).wrapping_add(16_777_600)
                } else {
                    base
                }
            })
            .collect()
    }

    /// The heart of the gate: the AVX2 lanes, the scalar twin, and the `_parallel` kernel must agree
    /// **bit-for-bit** for every width, activation, and a length that crosses the NT prologue + tail.
    #[test]
    fn simd_equals_scalar_twin() {
        // 1<<21 i32 elements → 8 MiB in + 8 MiB out = 16 MiB > NT_MIN_BYTES: forces NT + prologue; the
        // odd remainder forces a scalar tail. i8/u8 at the same length are 5 MiB total (no NT) — so both
        // store regimes are covered across the width sweep.
        let n = 2_097_155usize;
        let qi8 = fill_i8(n);
        let qu8 = fill_u8(n);
        let qi32 = fill_i32(n);
        let scale = 0.0123f32;
        for &act in &[DQ_ID, DQ_RELU, DQ_GELU, DQ_SILU] {
            for &(width, qptr) in &[
                (DQ_I8, qi8.as_ptr() as *const u8),
                (DQ_U8, qu8.as_ptr() as *const u8),
                (DQ_I32, qi32.as_ptr() as *const u8),
            ] {
                let op = act | width;
                let mut want = vec![0f32; n];
                unsafe { dequant_scalar(qptr, want.as_mut_ptr(), n, scale, op) };
                let mut got = vec![0f32; n];
                let mut got_par = vec![0f32; n];
                unsafe {
                    mercury_dequant_f32(qptr, got.as_mut_ptr(), n as i64, scale, op);
                    mercury_dequant_f32_parallel(qptr, got_par.as_mut_ptr(), n as i64, scale, op);
                }
                // Bit-exact equality (compare the raw bits so a NaN from a pathological scale still pins).
                for i in 0..n {
                    assert_eq!(got[i].to_bits(), want[i].to_bits(), "avx2!=scalar w={width} act={act} i={i}");
                    assert_eq!(got_par[i].to_bits(), want[i].to_bits(), "par!=scalar w={width} act={act} i={i}");
                }
            }
        }
    }

    /// The pure (`id`) dequant equals an independent **rational/f64 reference**: `q·scale` computed by
    /// promoting each integer to `f64` and multiplying, then demoting. For `|q| ≤ 2^24` the `i32→f32` is
    /// exact and this is `(q as f32)*scale` bit-for-bit; the large-magnitude entries additionally pin
    /// that both the kernel and the reference round the widen identically (RNE). A hard equality — there
    /// is no accumulation to reassociate.
    #[test]
    fn dequant_matches_rational_reference() {
        let n = 4099usize;
        let qi32 = fill_i32(n);
        let scale = 0.007f32;
        let mut got = vec![0f32; n];
        unsafe { mercury_dequant_f32(qi32.as_ptr() as *const u8, got.as_mut_ptr(), n as i64, scale, DQ_ID | DQ_I32) };
        for i in 0..n {
            // Reference: widen through f32 (round-to-nearest-even, as the kernel's cvt does), then one
            // f32 multiply — the exact operation the source `(q as f32)*scale` denotes.
            let want = (qi32[i] as f32) * scale;
            assert_eq!(got[i].to_bits(), want.to_bits(), "i={i}");
        }
    }

    /// Per-channel: AVX2 == scalar == parallel, bit-for-bit, across widths/activations, with a per-column
    /// scale and a non-multiple-of-8 `cols` (forces the tail) over several rows.
    #[test]
    fn perchan_simd_equals_scalar_twin() {
        let (rows, cols) = (37usize, 653usize); // cols not a multiple of 8; rows>1
        let n = rows * cols;
        let qi8 = fill_i8(n);
        let qi32 = fill_i32(n);
        let scale: Vec<f32> = (0..cols).map(|j| 0.002 + (j % 11) as f32 * 0.0005).collect();
        for &act in &[DQ_ID, DQ_RELU, DQ_GELU, DQ_SILU] {
            for &(width, qptr) in &[
                (DQ_I8, qi8.as_ptr() as *const u8),
                (DQ_I32, qi32.as_ptr() as *const u8),
            ] {
                let op = act | width;
                let mut want = vec![0f32; n];
                for i in 0..rows {
                    let qrow = unsafe { qptr.add(i * cols * if width == DQ_I32 { 4 } else { 1 }) };
                    unsafe { perchan_row_scalar(qrow, want.as_mut_ptr().add(i * cols), cols, scale.as_ptr(), op) };
                }
                let mut got = vec![0f32; n];
                let mut got_par = vec![0f32; n];
                unsafe {
                    mercury_dequant_perchan_f32(qptr, got.as_mut_ptr(), rows as i64, cols as i64, scale.as_ptr(), op);
                    mercury_dequant_perchan_f32_parallel(qptr, got_par.as_mut_ptr(), rows as i64, cols as i64, scale.as_ptr(), op);
                }
                for t in 0..n {
                    assert_eq!(got[t].to_bits(), want[t].to_bits(), "perchan avx2!=scalar w={width} act={act} t={t}");
                    assert_eq!(got_par[t].to_bits(), want[t].to_bits(), "perchan par!=scalar w={width} act={act} t={t}");
                }
            }
        }
    }

    /// Small-length edge cases: n < 8 (no vector step), n == 8 exactly, and the degenerate n == 0.
    #[test]
    fn small_and_degenerate_lengths() {
        let scale = 0.5f32;
        for &n in &[0usize, 1, 7, 8, 9] {
            let q = fill_i8(n.max(1));
            let mut got = vec![0f32; n];
            let mut want = vec![0f32; n];
            unsafe {
                if n > 0 {
                    dequant_scalar(q.as_ptr() as *const u8, want.as_mut_ptr(), n, scale, DQ_RELU | DQ_I8);
                }
                mercury_dequant_f32(q.as_ptr() as *const u8, got.as_mut_ptr(), n as i64, scale, DQ_RELU | DQ_I8);
            }
            assert_eq!(got, want, "n={n}");
        }
    }
}
