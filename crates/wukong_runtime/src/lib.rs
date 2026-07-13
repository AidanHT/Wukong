//! `wukong_runtime` — the minimal runtime that Wukong programs link against.
//!
//! Kept deliberately tiny and allocation-explicit, matching the language's philosophy. The
//! interpreter calls these implementations directly; native (LLVM) builds link the same logic
//! compiled as a static library. Today it provides a bump [`Arena`], a CPU [`parallel_for`], and a
//! tuned [`wukong_sgemm`] (the matmul microkernel the compiler lowers a matmul nest to).

mod attention;
mod gemm;
pub use attention::wukong_attention_f32;
pub use gemm::{
    wukong_sgemm, wukong_sgemm_bf16_nt, wukong_sgemm_bf16_nt_epi,
    wukong_sgemm_bf16_nt_epi_parallel, wukong_sgemm_bf16_nt_parallel, wukong_sgemm_bf16_tn,
    wukong_sgemm_bf16_tn_parallel, wukong_sgemm_f16_nt, wukong_sgemm_f16_nt_epi,
    wukong_sgemm_f16_nt_epi_parallel, wukong_sgemm_f16_nt_parallel, wukong_sgemm_f16_tn,
    wukong_sgemm_f16_tn_parallel, wukong_sgemm_nt, wukong_sgemm_nt_alpha,
    wukong_sgemm_nt_alpha_parallel, wukong_sgemm_nt_epi, wukong_sgemm_nt_epi_parallel,
    wukong_sgemm_nt_parallel, wukong_sgemm_parallel, wukong_sgemm_tn, wukong_sgemm_tn_parallel,
};

mod gemv;
pub use gemv::{
    wukong_sgemv, wukong_sgemv_alpha, wukong_sgemv_alpha_parallel, wukong_sgemv_parallel,
};

mod gevm;
pub use gevm::{wukong_sgevm_f32, wukong_sgevm_f32_parallel};

mod vmath;
pub use vmath::{
    wukong_vmath2_f32, wukong_vmath_bf16, wukong_vmath_f16, wukong_vmath_f32, VM_ACOS,
    VM_ACOSH, VM_ASIN, VM_ASINH, VM_ATAN, VM_ATANH, VM_CBRT, VM_COS, VM_COSH, VM_ERF, VM_EXP,
    VM_EXP10, VM_EXP2, VM_EXPM1, VM_GELU, VM_LOG, VM_LOG10, VM_LOG1P, VM_LOG2, VM_LOGSIGMOID,
    VM_RELU, VM_SIGMOID, VM_SILU, VM_SIN, VM_SINH, VM_SOFTSIGN, VM_TAN, VM_TANH,
};

mod velem;
pub use velem::{
    wukong_velem_f32, wukong_velem_f32_parallel, wukong_vhorner_f32, VE_ID, VE_RELU, VE_RELU6,
    VE_USE_Y,
};

mod reduce;
pub use reduce::{
    wukong_argreduce_f32, wukong_argreduce_f32_parallel, wukong_sreduce_f32,
    wukong_sreduce_f32_parallel, RED_ARGMAX, RED_ARGMIN, RED_DOT, RED_SSD, RED_SUM, RED_SUMSQ,
};

mod norm;
pub use norm::{
    wukong_norm_affine_f32, wukong_norm_affine_f32_parallel, wukong_norm_f32,
    wukong_norm_f32_parallel, NORM_LAYERNORM, NORM_LOGSOFTMAX, NORM_RMSNORM, NORM_SOFTMAX,
};

mod i8gemm;
pub use i8gemm::{
    wukong_i8gemm_nt, wukong_i8gemm_nt_deq, wukong_i8gemm_nt_deq_parallel,
    wukong_i8gemm_nt_parallel,
};

mod dequant;
pub use dequant::{
    wukong_dequant_f32, wukong_dequant_f32_parallel, wukong_dequant_perchan_f32,
    wukong_dequant_perchan_f32_parallel, DQ_GELU, DQ_I32, DQ_I8, DQ_ID, DQ_RELU, DQ_SILU, DQ_U8,
};

mod lowp;
pub use lowp::{
    wukong_axpby_bf16, wukong_axpby_bf16_out, wukong_axpby_f16, wukong_axpby_f16_out,
    wukong_dot_bf16, wukong_dot_bf16_parallel, wukong_dot_f16, wukong_dot_f16_parallel,
    wukong_reduce_bf16, wukong_reduce_bf16_parallel, wukong_reduce_f16,
    wukong_reduce_f16_parallel, wukong_sum_bf16, wukong_sum_bf16_parallel, wukong_sum_f16,
    wukong_sum_f16_parallel, wukong_vmath_bf16_out, wukong_vmath_f16_out,
};

mod transpose;
pub use transpose::{
    wukong_transpose_f32, wukong_transpose_f32_parallel, wukong_transpose_u16,
    wukong_transpose_u16_parallel,
};

mod colreduce;
pub use colreduce::{
    wukong_coll2_f32, wukong_coll2_f32_parallel, wukong_colmax_f32, wukong_colmax_f32_parallel,
    wukong_colmaxabs_f32, wukong_colmaxabs_f32_parallel, wukong_colmean_f32,
    wukong_colmean_f32_parallel, wukong_colmin_f32, wukong_colmin_f32_parallel,
    wukong_colrms_f32, wukong_colrms_f32_parallel, wukong_colsum_f32, wukong_colsum_f32_parallel,
    wukong_colsumsq_f32, wukong_colsumsq_f32_parallel,
};

mod softmax_bwd;
pub use softmax_bwd::{wukong_softmax_bwd_f32, wukong_softmax_bwd_f32_parallel};

mod rmsnorm_bwd;
pub use rmsnorm_bwd::{wukong_rmsnorm_bwd_f32, wukong_rmsnorm_bwd_f32_parallel};

mod layernorm_bwd;
pub use layernorm_bwd::{wukong_layernorm_bwd_f32, wukong_layernorm_bwd_f32_parallel};

mod pool2d;
pub use pool2d::{
    wukong_avgpool2d_f32, wukong_avgpool2d_f32_parallel, wukong_maxpool2d_f32,
    wukong_maxpool2d_f32_parallel,
};

mod xent;
pub use xent::{wukong_xent_fwd_f32, wukong_xent_fwd_f32_parallel};

mod embedding;
pub use embedding::{wukong_embedding_f32, wukong_embedding_f32_parallel};

mod rope;
pub use rope::{wukong_rope_f32, wukong_rope_f32_parallel};

mod logsoftmax;
pub use logsoftmax::{
    wukong_logsoftmax_f32, wukong_logsoftmax_f32_parallel, wukong_logsumexp_f32,
    wukong_logsumexp_f32_parallel,
};

mod xent_bwd;
pub use xent_bwd::{wukong_xent_bwd_f32, wukong_xent_bwd_f32_parallel};

mod rope_bwd;
pub use rope_bwd::{wukong_rope_bwd_f32, wukong_rope_bwd_f32_parallel};

mod kldiv;
pub use kldiv::{wukong_kldiv_f32, wukong_kldiv_f32_parallel};

mod entropy;
pub use entropy::{wukong_entropy_f32, wukong_entropy_f32_parallel};

mod kd_loss;
pub use kd_loss::{wukong_kd_loss_f32, wukong_kd_loss_f32_parallel};

mod rowarg;
pub use rowarg::{
    wukong_rowargmax_i32, wukong_rowargmax_i32_parallel, wukong_rowargmin_i32,
    wukong_rowargmin_i32_parallel,
};

mod colarg;
pub use colarg::{
    wukong_colargmax_i32, wukong_colargmax_i32_parallel, wukong_colargmin_i32,
    wukong_colargmin_i32_parallel,
};

mod cumsum;
pub use cumsum::{wukong_cumsum_f32, wukong_cumsum_f32_parallel};

mod cumminmax;
pub use cumminmax::{
    wukong_cummax_f32, wukong_cummax_f32_parallel, wukong_cummin_f32, wukong_cummin_f32_parallel,
};

mod lrscan;
pub use lrscan::{wukong_lrscan_f32, wukong_lrscan_f32_parallel};

mod scatter;
pub use scatter::{wukong_scatter_add_f32, wukong_scatter_add_f32_parallel};

mod cumprod;
pub use cumprod::{wukong_cumprod_f32, wukong_cumprod_f32_parallel};

mod bias;
pub use bias::{wukong_bias_bcast_f32, wukong_bias_bcast_f32_parallel, BIAS_ACT_NONE};

/// `bf16` (the "brain float": an `f32` truncated to its top 16 bits) round-to-nearest-even from an
/// `f32`, returning the 16 stored bits. This is the single shared definition the interpreter, the
/// native backend (which emits the identical integer arithmetic in CLIF), and the bf16 GEMM packer
/// all use, so every path agrees bit-for-bit. Matches the rounding TensorFlow/PyTorch use.
#[inline]
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        // Keep NaN a NaN (rounding could otherwise carry it to inf); force the quiet bit.
        return ((bits >> 16) as u16) | 0x0040;
    }
    // Round to nearest even: bias by 0x7fff plus the lsb of the surviving mantissa, then truncate.
    let rounding_bias = 0x0000_7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

/// Widen `bf16` stored bits back to the `f32` they represent (exact: the low 16 bits are zero).
#[inline]
pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `x` rounded to `bf16` precision, as the `f32` an `f32 -> bf16 -> f32` round-trip yields. The
/// value `[bf16; N]` storage observes; the native backend computes the identical result.
#[inline]
pub fn round_bf16(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(x))
}

/// IEEE `f16` (half precision) round-to-nearest-even from an `f32`, returning the 16 stored bits.
/// Backed by the `half` crate, which matches the F16C `vcvtps2ph` the runtime f16 reduction kernels
/// use (the `simd_equals_scalar_twin` test pins F16C == `half`). The single shared definition the
/// interpreter and the native backend both use (the latter via the `wukong_f32_to_f16_bits` shim),
/// so the two agree bit-for-bit. Unlike bf16 (the top 16 bits of an f32, a cheap inline round), f16
/// has a different exponent/mantissa layout, so the conversion is a function call, not inline math.
#[inline]
pub fn f32_to_f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

/// Widen `f16` stored bits to the `f32` they represent (lossless; equals F16C `vcvtph2ps`).
#[inline]
pub fn f16_bits_to_f32(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

/// `x` rounded to `f16` precision, as an `f32 -> f16 -> f32` round-trip yields — the value `[f16; N]`
/// storage observes; the native backend computes the identical result (it calls these shims).
#[inline]
pub fn round_f16(x: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(x))
}

/// C-ABI shim: round an `f32` to `f16`, returning the 16 stored bits (in an `i32` for a clean call
/// ABI). The Cranelift backend calls this for an `f16` store/cast, so native and interp — which uses
/// [`f32_to_f16_bits`] directly — round through the identical code.
///
/// # Safety
/// Pure; `extern "C"` only for the JIT symbol table.
#[no_mangle]
pub unsafe extern "C" fn wukong_f32_to_f16_bits(x: f32) -> i32 {
    f32_to_f16_bits(x) as i32
}

/// C-ABI shim: widen `f16` stored bits (low 16 of `b`) to `f32`. The Cranelift backend calls this for
/// an `f16` load/cast.
///
/// # Safety
/// Pure; `extern "C"` only for the JIT symbol table.
#[no_mangle]
pub unsafe extern "C" fn wukong_f16_bits_to_f32(b: i32) -> f32 {
    f16_bits_to_f32(b as u16)
}

/// A bump (arena) allocator over an owned byte buffer. Allocation is a pointer bump; freeing is
/// all-at-once via [`Arena::reset`]. This is the idiomatic allocator for kernel scratch space:
/// no per-object bookkeeping, no fragmentation.
pub struct Arena {
    buf: Vec<u8>,
    offset: usize,
}

impl Arena {
    /// Create an arena with `capacity` bytes of backing store.
    pub fn with_capacity(capacity: usize) -> Arena {
        Arena {
            buf: vec![0u8; capacity],
            offset: 0,
        }
    }

    /// Bytes handed out so far.
    pub fn used(&self) -> usize {
        self.offset
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Allocate `size` bytes aligned to `align` (a power of two). Returns the byte offset of the
    /// allocation, or `None` if the arena is exhausted.
    pub fn alloc(&mut self, size: usize, align: usize) -> Option<usize> {
        debug_assert!(align.is_power_of_two());
        let start = (self.offset + align - 1) & !(align - 1);
        let end = start.checked_add(size)?;
        if end > self.buf.len() {
            return None;
        }
        self.offset = end;
        Some(start)
    }

    /// A mutable view of a previously-allocated region.
    pub fn slice_mut(&mut self, offset: usize, len: usize) -> &mut [u8] {
        &mut self.buf[offset..offset + len]
    }

    /// Free everything at once.
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}

/// Run `body(i)` for `i` in `lo, lo+step, ...` while `i < hi`.
///
/// The execution order is deterministic and sequential, so it matches the interpreter and serves
/// as a correct reference; a work-stealing thread pool can replace the body later without changing
/// observable results for associative reductions.
pub fn parallel_for(lo: i64, hi: i64, step: i64, mut body: impl FnMut(i64)) {
    if step <= 0 {
        return;
    }
    let mut i = lo;
    while i < hi {
        body(i);
        i += step;
    }
}

/// A raw `env` address shuttled across worker threads. Sending the address is sound here: the
/// pointee outlives the (blocking) `wukong_parallel_for` call, and worker chunks touch disjoint
/// output indices, so there is no data race.
#[derive(Clone, Copy)]
struct EnvAddr(usize);
unsafe impl Send for EnvAddr {}
unsafe impl Sync for EnvAddr {}

/// Build the GLOBAL rayon pool exactly once, with the runtime's required configuration: 16 MiB
/// worker stacks (a JIT'd parallel-for body carries its per-iteration scratch as ordinary stack
/// allocas — an outlined `@parallel` head-attention region privatizes ~1.5 MiB of qh/kh/vt/scores
/// buffers at S=512 — which does not fit reliably in rayon's default 2 MiB stacks; Cranelift
/// emits inline stack probes, so a big frame is safe exactly when the reserve is big enough) and
/// `RAYON_NUM_THREADS` honored explicitly (the core-count sweep instrument).
///
/// EVERY parallel path in this crate that can be the process's FIRST rayon touch must call this
/// before forking: the pre-2026-07-10 code initialized only inside `wukong_parallel_for`, but in
/// a real transformer forward the first parallel op is a `_parallel` norm — rayon then built the
/// default registry (2 MiB stacks) first and the 16 MiB `build_global` silently lost the race, so
/// region bodies ran on 2 MiB stacks. Best-effort as ever: if another rayon user in the host
/// process won the race anyway, `build_global` errors and behavior is as before.
pub(crate) fn ensure_global_pool() {
    use std::sync::Once;
    static POOL_INIT: Once = Once::new();
    POOL_INIT.call_once(|| {
        let mut b = rayon::ThreadPoolBuilder::new().stack_size(16 * 1024 * 1024);
        if let Some(t) = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&t| t > 0)
        {
            b = b.num_threads(t);
        }
        let _ = b.build_global();
    });
}

/// Pool-unification opt-out (`WUKONG_POOL_UNIFY=0`, read once, **default ON**): when ON, the
/// model-path kernels (`_parallel` norms/velem/reductions and `wukong_parallel_for` regions) run
/// on the SAME private physical-core pool the parallel GEMM uses, instead of the logical-core
/// global pool. One pool = no private↔global park/unpark churn between adjacent kernels of a
/// forward pass (~6–7 pool bounces per transformer layer before this), and no HT-sibling
/// contention for the compute-bound kernels. Scheduling-only: every kernel routed here chunks its
/// work independently of the worker count (fixed-chunk reductions/velem, per-row norms,
/// per-index regions), so serial == parallel stays bit-exact under either pool. `=0` restores the
/// old global-pool routing as the adjacent-run A/B instrument.
pub(crate) fn pool_unify() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("WUKONG_POOL_UNIFY").map_or(true, |v| v != "0"))
}

/// Run `f` on the unified kernel pool: the private physical-core GEMM pool when unification is ON
/// and the pool exists (see [`pool_unify`] / `gemm::gemm_pool`), the global pool otherwise. The
/// global pool is configured first in every case ([`ensure_global_pool`]), so a fallback fork
/// lands on properly-sized stacks. Nested calls from a worker of the same pool run inline
/// (rayon's `install` semantics) — safe for kernels invoked inside an outlined region body.
pub(crate) fn run_on_wuk_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    ensure_global_pool();
    #[cfg(target_arch = "x86_64")]
    if pool_unify() {
        if let Some(p) = crate::gemm::gemm_pool() {
            return p.install(f);
        }
    }
    f()
}

/// Worker count of the unified kernel pool ([`run_on_wuk_pool`]'s target) as seen from the calling
/// thread. `1` means a `_parallel` kernel entry has no second core to win with — the width==1
/// serial fast-path: run the serial sibling kernel inline instead of paying the pool handoff plus
/// the parallel schedule's per-chunk/per-block overheads (measured 0.62–0.80× of the serial kernel
/// at `RAYON_NUM_THREADS=1`). Throughput-only dispatch: every substituted serial sibling is
/// bit-identical to its parallel form by the standing gate-pinned law, so the bits cannot change.
/// NOT used by `wukong_parallel_for`: outlined region bodies carry multi-hundred-KB frames that
/// must stay on the pool's 16 MiB worker stacks, and its width-1 shape (one par_iter job) is
/// already minimal.
pub(crate) fn wuk_pool_width() -> usize {
    ensure_global_pool();
    #[cfg(target_arch = "x86_64")]
    if pool_unify() {
        if let Some(p) = crate::gemm::gemm_pool() {
            return p.current_num_threads();
        }
    }
    rayon::current_num_threads()
}

/// The C-ABI parallel-for the native backend lowers `@parallel for` to. Splits `[0, n)` into one
/// contiguous chunk per worker of the unified kernel pool ([`run_on_wuk_pool`]) and runs
/// `body(start, end, env)` on each concurrently, returning only once every chunk has completed.
///
/// Bodies must be data-parallel: each index is processed exactly once and the chunks must not have
/// cross-iteration dependencies (the interpreter runs the whole range sequentially and must agree).
/// Chunk boundaries derive from the worker count / claim order and are therefore pool- and
/// schedule-dependent — legal exactly because of that independence contract (bits never depend on
/// which chunk ran an index).
///
/// **Scheduling** (`WUKONG_PFOR_DYN=0` opts out to the old static split, read once): iterations
/// are claimed DYNAMICALLY from a shared atomic counter in granules of `ceil(n / (workers·8))`
/// (min 1), so on this asymmetric 6P+8E+2LPE part a P-core that finishes its granule pulls the
/// next one instead of idling while E-cores straggle — the same self-scheduling shape as the
/// GEMM's `WUKONG_GEMM_DYN` claim queue, at one `fetch_add` per granule. When `n ≤ workers`
/// dynamic and static degenerate to the same one-granule-per-worker shape, so the knob only
/// matters when there is imbalance to absorb.
///
/// # Safety
/// `body` must be a valid `extern "C" fn(i64, i64, *const u8)` and `env` valid for the call.
#[no_mangle]
pub unsafe extern "C" fn wukong_parallel_for(
    n: i64,
    body: extern "C" fn(i64, i64, *const u8),
    env: *const u8,
) {
    use rayon::prelude::*;
    if n <= 0 {
        return;
    }
    let n = n as usize;
    let env = EnvAddr(env as usize);
    run_on_wuk_pool(move || {
        // Read the worker count INSIDE the installed context so it reflects the pool actually
        // running the chunks (the private physical-core pool under unification).
        let workers = rayon::current_num_threads().max(1).min(n);
        if pfor_dyn() && workers > 1 {
            // Dynamic claim queue: granules small enough to absorb P/E-core speed asymmetry,
            // big enough that the per-granule fetch_add is noise even for huge n. The counter is
            // cache-line-isolated (128 B: line + adjacent-line-prefetch pair) so the one shared
            // write bounces only for the claims themselves.
            #[repr(align(128))]
            struct Claims(std::sync::atomic::AtomicUsize);
            let g = n.div_ceil(workers * 8).max(1);
            let ntasks = n.div_ceil(g);
            let claims = Claims(std::sync::atomic::AtomicUsize::new(0));
            let claims_ref = &claims;
            (0..workers).into_par_iter().for_each(|_| loop {
                // Relaxed suffices: the counter only hands out unique granules; the fork-join's
                // join publishes every write the body made.
                let c = claims_ref.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if c >= ntasks {
                    break;
                }
                let start = c * g;
                let end = (start + g).min(n);
                body(start as i64, end as i64, env.0 as *const u8);
            });
        } else {
            let chunk = n.div_ceil(workers);
            (0..workers).into_par_iter().for_each(|w| {
                let start = w * chunk;
                if start >= n {
                    return;
                }
                let end = (start + chunk).min(n);
                body(start as i64, end as i64, env.0 as *const u8);
            });
        }
    });
}

/// `WUKONG_PFOR_DYN` (read once, default ON): dynamic granule claiming in
/// [`wukong_parallel_for`]; `=0` restores the static one-chunk-per-worker split as the
/// adjacent-run A/B instrument. Scheduling-only — the independence contract makes the bits
/// identical either way.
fn pfor_dyn() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("WUKONG_PFOR_DYN").map_or(true, |v| v != "0"))
}

/// The hidden allocation header preceding every `wukong_rt_alloc` data pointer: 16 bytes storing
/// the total layout size (so `wukong_rt_free` can reconstruct the `Layout` Rust's allocator API
/// requires at deallocation) while keeping the returned data pointer 16-byte aligned.
const HEAP_HDR: usize = 16;

/// Zero-initialized heap allocation — the runtime backing of the Wukong `alloc_<T>(n)` builtins.
///
/// Allocates `count * elem_size` bytes, **zeroed** (the determinism contract: both backends must
/// observe identical initial contents — zero bits decode to `0`/`0.0` for every Wukong element
/// type), and returns a 16-byte-aligned pointer to the data. `elem_is_float` is part of the shared
/// call ABI but consumed only by the interpreter's typed-zero slot model; it is ignored here.
///
/// Returns **null** for a zero/negative byte count (a zero-length slice has no dereferenceable
/// element, so the null is never read through by a well-formed program), on multiply overflow, or
/// on allocator exhaustion. The compiler clamps a negative `count` to 0 before the call; the clamp
/// here is defense in depth.
#[no_mangle]
pub extern "C" fn wukong_rt_alloc(count: i64, elem_size: i64, _elem_is_float: i64) -> *mut u8 {
    let bytes = (count.max(0) as u128).saturating_mul(elem_size.max(0) as u128);
    if bytes == 0 || bytes > (isize::MAX as u128) - (HEAP_HDR as u128) {
        return std::ptr::null_mut();
    }
    let total = bytes as usize + HEAP_HDR;
    let Ok(layout) = std::alloc::Layout::from_size_align(total, HEAP_HDR) else {
        return std::ptr::null_mut();
    };
    // SAFETY: `layout` has a non-zero size and a valid power-of-two alignment.
    unsafe {
        let base = std::alloc::alloc_zeroed(layout);
        if base.is_null() {
            return std::ptr::null_mut();
        }
        (base as *mut usize).write(total);
        base.add(HEAP_HDR)
    }
}

/// Release an allocation previously returned by [`wukong_rt_alloc`] (the Wukong `free(s)`
/// builtin). Null — the zero-length or failed allocation — is a no-op. Passing any other pointer,
/// double-freeing, or touching the slice after the free is **undefined behavior** on the native
/// backend; the interpreter's mark-and-forget model (its run-scoped memory is never reclaimed)
/// keeps such programs from crashing there, but they are outside the differential contract.
#[no_mangle]
pub extern "C" fn wukong_rt_free(data: *mut u8) {
    if data.is_null() {
        return;
    }
    // SAFETY: `data` came from `wukong_rt_alloc`, whose header records the total layout size.
    unsafe {
        let base = data.sub(HEAP_HDR);
        let total = (base as *const usize).read();
        if let Ok(layout) = std::alloc::Layout::from_size_align(total, HEAP_HDR) {
            std::alloc::dealloc(base, layout);
        }
    }
}

/// Reconstruct an owned filesystem path from the NUL-terminated `*const u8` C string the file-I/O
/// intrinsics receive as their `path` argument. Returns `None` for a null pointer or a byte
/// sequence that is not valid UTF-8 — either case is reported to the program as an open failure
/// (`-1`), never a silent open of the wrong file.
///
/// # Safety
/// `path`, when non-null, must point at a NUL-terminated byte string that stays valid for the
/// duration of the call (the compiler passes a pointer into the program's static/heap string data).
unsafe fn rt_path(path: *const u8) -> Option<std::path::PathBuf> {
    if path.is_null() {
        return None;
    }
    // `CStr::from_ptr` performs the NUL walk; `to_str` validates UTF-8.
    let cstr = std::ffi::CStr::from_ptr(path as *const std::os::raw::c_char);
    cstr.to_str().ok().map(std::path::PathBuf::from)
}

/// Defines one `read`/`write` pair of file-I/O intrinsics for element type `$T`, following the
/// frozen contract: **headerless, raw contiguous little-endian** elements (no magic, no length
/// prefix). Both backends (native here, interpreter elsewhere) MUST produce byte-identical files
/// and identical `-1` / `-2` / `n` return codes, so every element is (de)serialized with
/// `from_le_bytes` / `to_le_bytes` regardless of host endianness.
macro_rules! rt_file_io {
    ($read:ident, $write:ident, $T:ty) => {
        /// Read up to `len` little-endian `
        #[doc = stringify!($T)]
        /// ` elements from `path` into `data`.
        ///
        /// Opens `path` read-only (returns `-1` if that fails, leaving `data` untouched), then
        /// reads `n = min(len, file_size / sizeof)` elements — the file's trailing partial element,
        /// if any, is ignored. `data[0..n]` receives the decoded values; `data[n..]` is left as the
        /// caller allocated it (`alloc_*` pre-zeroes). Returns `n` on success, `-2` on a mid-read
        /// I/O error, and `0` (without touching `data`) when `len <= 0`.
        ///
        /// # Safety
        /// `data` must address at least `len` writable `
        #[doc = stringify!($T)]
        /// ` slots, and `path` must satisfy [`rt_path`]'s contract.
        #[no_mangle]
        pub extern "C" fn $read(path: *const u8, data: *mut $T, len: i64) -> i64 {
            use std::io::Read;
            const SZ: usize = std::mem::size_of::<$T>();
            if len <= 0 {
                return 0;
            }
            // SAFETY: forwarded from this function's own safety contract.
            let Some(path) = (unsafe { rt_path(path) }) else {
                return -1;
            };
            let mut file = match std::fs::File::open(&path) {
                Ok(f) => f,
                Err(_) => return -1,
            };
            let avail = match file.metadata() {
                Ok(m) => (m.len() / SZ as u64) as usize,
                Err(_) => return -2,
            };
            let n = (len as usize).min(avail);
            if n == 0 {
                return 0;
            }
            let mut bytes = vec![0u8; n * SZ];
            if file.read_exact(&mut bytes).is_err() {
                return -2;
            }
            for (i, chunk) in bytes.chunks_exact(SZ).enumerate() {
                let v = <$T>::from_le_bytes(chunk.try_into().unwrap());
                // SAFETY: `i < n <= len` and the caller guarantees `len` writable slots at `data`.
                unsafe { *data.add(i) = v };
            }
            n as i64
        }

        /// Write all `len` `
        #[doc = stringify!($T)]
        /// ` elements of `data` to `path` as little-endian bytes.
        ///
        /// Creates/truncates `path` (returns `-1` if that fails). Returns `len` on success and
        /// `-2` on a write error. A `len <= 0` writes an empty (truncated) file and returns `0`.
        ///
        /// # Safety
        /// `data` must address at least `len` readable `
        #[doc = stringify!($T)]
        /// ` slots, and `path` must satisfy [`rt_path`]'s contract.
        #[no_mangle]
        pub extern "C" fn $write(path: *const u8, data: *const $T, len: i64) -> i64 {
            use std::io::Write;
            // SAFETY: forwarded from this function's own safety contract.
            let Some(path) = (unsafe { rt_path(path) }) else {
                return -1;
            };
            let mut file = match std::fs::File::create(&path) {
                Ok(f) => f,
                Err(_) => return -1,
            };
            let count = len.max(0) as usize;
            let mut bytes = Vec::with_capacity(count * std::mem::size_of::<$T>());
            for i in 0..count {
                // SAFETY: `i < count == len` and the caller guarantees `len` readable slots.
                let v = unsafe { *data.add(i) };
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            if file.write_all(&bytes).is_err() {
                return -2;
            }
            len.max(0)
        }
    };
}

rt_file_io!(wukong_rt_read_f32, wukong_rt_write_f32, f32);
rt_file_io!(wukong_rt_read_f64, wukong_rt_write_f64, f64);
rt_file_io!(wukong_rt_read_i32, wukong_rt_write_i32, i32);
rt_file_io!(wukong_rt_read_i64, wukong_rt_write_i64, i64);
rt_file_io!(wukong_rt_read_i8, wukong_rt_write_i8, i8);
rt_file_io!(wukong_rt_read_u8, wukong_rt_write_u8, u8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt_alloc_zeroes_and_frees() {
        // A fresh allocation is fully zeroed, 16-byte aligned, writable, and freeable.
        let n = 1000usize;
        let p = wukong_rt_alloc(n as i64, 4, 1);
        assert!(!p.is_null());
        assert_eq!(p as usize % 16, 0, "data pointer must be 16-byte aligned");
        // SAFETY: p points at n*4 zeroed bytes owned by this test.
        unsafe {
            let f = p as *mut f32;
            for i in 0..n {
                assert_eq!(*f.add(i), 0.0, "alloc must zero-initialize");
            }
            for i in 0..n {
                *f.add(i) = i as f32;
            }
            assert_eq!(*f.add(n - 1), (n - 1) as f32);
        }
        wukong_rt_free(p);
    }

    #[test]
    fn rt_alloc_degenerate_counts_are_null_and_free_ignores_null() {
        assert!(wukong_rt_alloc(0, 4, 0).is_null());
        assert!(wukong_rt_alloc(-5, 8, 1).is_null());
        assert!(wukong_rt_alloc(i64::MAX, i64::MAX, 0).is_null(), "overflow must yield null");
        wukong_rt_free(std::ptr::null_mut()); // must be a no-op, not a crash
    }

    #[test]
    fn bf16_exact_values_roundtrip() {
        // Values exactly representable in bf16 survive a round-trip unchanged.
        for &x in &[0.0f32, 1.0, -2.0, 0.5, 256.0, -0.015625] {
            assert_eq!(round_bf16(x), x, "{x} should be bf16-exact");
        }
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        // 1 + 2^-8 sits exactly between two bf16 values (mantissa step is 2^-7); RNE picks the even
        // one, which is 1.0 (mantissa bits 0). 1 + 2^-7 is exact.
        assert_eq!(round_bf16(1.0 + 2f32.powi(-8)), 1.0);
        assert_eq!(round_bf16(1.0 + 2f32.powi(-7)), 1.0 + 2f32.powi(-7));
        // 1/3 rounds to the nearest of the two surrounding bf16 grid points.
        let third = round_bf16(1.0 / 3.0);
        assert_eq!(third.to_bits() & 0xffff, 0, "low 16 bits must be zero");
        assert!((third - 1.0 / 3.0).abs() < 2f32.powi(-7));
    }

    #[test]
    fn bf16_specials() {
        assert_eq!(round_bf16(f32::INFINITY), f32::INFINITY);
        assert_eq!(round_bf16(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(round_bf16(f32::NAN).is_nan());
        assert_eq!(bf16_bits_to_f32(f32_to_bf16_bits(0.0)), 0.0);
    }

    extern "C" fn fill_squares(start: i64, end: i64, env: *const u8) {
        let out = env as *mut i64;
        for i in start..end {
            unsafe { *out.add(i as usize) = i * i };
        }
    }

    #[test]
    fn parallel_for_covers_every_index_once() {
        let n = 10_000usize;
        let mut buf = vec![0i64; n];
        unsafe {
            wukong_parallel_for(n as i64, fill_squares, buf.as_mut_ptr() as *const u8);
        }
        assert!(buf
            .iter()
            .enumerate()
            .all(|(i, &v)| v == (i as i64) * (i as i64)));
    }

    #[test]
    fn arena_alignment_and_exhaustion() {
        let mut a = Arena::with_capacity(64);
        let p0 = a.alloc(1, 1).unwrap();
        assert_eq!(p0, 0);
        // next 8-aligned allocation rounds up past the 1-byte one.
        let p1 = a.alloc(16, 8).unwrap();
        assert_eq!(p1, 8);
        assert_eq!(a.used(), 24);
        // too big -> None, and the arena is unchanged.
        assert!(a.alloc(1024, 8).is_none());
        assert_eq!(a.used(), 24);
        a.reset();
        assert_eq!(a.used(), 0);
    }

    #[test]
    fn parallel_for_sequential_reduction() {
        let mut sum = 0i64;
        parallel_for(0, 10, 2, |i| sum += i);
        assert_eq!(sum, 2 + 4 + 6 + 8);
    }

    #[test]
    fn parallel_for_edge_cases() {
        // Empty range and non-positive step do nothing.
        let mut n = 0;
        parallel_for(5, 5, 1, |_| n += 1);
        parallel_for(0, 10, 0, |_| n += 1);
        parallel_for(0, 10, -2, |_| n += 1);
        assert_eq!(n, 0);

        // A step larger than the range runs exactly once (the lo iteration).
        let mut hits = Vec::new();
        parallel_for(3, 10, 100, |i| hits.push(i));
        assert_eq!(hits, vec![3]);
    }

    #[test]
    fn arena_reset_reuses_storage() {
        let mut a = Arena::with_capacity(32);
        let first = a.alloc(16, 8).unwrap();
        a.slice_mut(first, 16).fill(0xAB);
        a.reset();
        // After reset, the same offset is handed out again (storage reused, not grown).
        let second = a.alloc(16, 8).unwrap();
        assert_eq!(first, second);
        assert_eq!(a.capacity(), 32);
    }

    // ---- file-I/O intrinsics (wukong_rt_{read,write}_{f32,i32,i64,u8}) ----

    /// A collision-free temp path (process id + monotonic counter) so parallel test threads and
    /// repeated runs never share a file. Not created on disk — the caller writes it.
    fn io_tmp(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let id = CTR.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!("wukong_io_{}_{}_{}.bin", tag, std::process::id(), id));
        p
    }

    /// NUL-terminated `*const u8` view of a path — the ABI the intrinsics expect. The returned
    /// `CString` owns the bytes and must outlive every use of the pointer.
    fn cpath(p: &std::path::Path) -> std::ffi::CString {
        std::ffi::CString::new(p.to_str().unwrap()).unwrap()
    }

    #[test]
    fn file_io_roundtrip_f32() {
        let path = io_tmp("f32");
        let cp = cpath(&path);
        let ptr = cp.as_ptr() as *const u8;
        let src = [1.5f32, -2.25, 0.0, 3.141_592_7, 6.022e23, -1.0e-9];
        assert_eq!(wukong_rt_write_f32(ptr, src.as_ptr(), src.len() as i64), 6);
        let mut dst = [0f32; 6];
        assert_eq!(wukong_rt_read_f32(ptr, dst.as_mut_ptr(), 6), 6);
        assert_eq!(dst, src);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_io_roundtrip_f64() {
        let path = io_tmp("f64");
        let cp = cpath(&path);
        let ptr = cp.as_ptr() as *const u8;
        let src = [1.5f64, -2.25, 0.0, std::f64::consts::PI, 6.022e23, -1.0e-9];
        assert_eq!(wukong_rt_write_f64(ptr, src.as_ptr(), src.len() as i64), 6);
        let mut dst = [0f64; 6];
        assert_eq!(wukong_rt_read_f64(ptr, dst.as_mut_ptr(), 6), 6);
        assert_eq!(dst, src);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_io_roundtrip_i32() {
        let path = io_tmp("i32");
        let cp = cpath(&path);
        let ptr = cp.as_ptr() as *const u8;
        let src = [0i32, -1, i32::MIN, i32::MAX, 123_456];
        assert_eq!(wukong_rt_write_i32(ptr, src.as_ptr(), src.len() as i64), 5);
        let mut dst = [0i32; 5];
        assert_eq!(wukong_rt_read_i32(ptr, dst.as_mut_ptr(), 5), 5);
        assert_eq!(dst, src);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_io_roundtrip_i64() {
        let path = io_tmp("i64");
        let cp = cpath(&path);
        let ptr = cp.as_ptr() as *const u8;
        let src = [0i64, -1, i64::MIN, i64::MAX, 9_876_543_210];
        assert_eq!(wukong_rt_write_i64(ptr, src.as_ptr(), src.len() as i64), 5);
        let mut dst = [0i64; 5];
        assert_eq!(wukong_rt_read_i64(ptr, dst.as_mut_ptr(), 5), 5);
        assert_eq!(dst, src);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_io_roundtrip_i8() {
        let path = io_tmp("i8");
        let cp = cpath(&path);
        let ptr = cp.as_ptr() as *const u8;
        let src = [0i8, -1, i8::MIN, i8::MAX, 42];
        assert_eq!(wukong_rt_write_i8(ptr, src.as_ptr(), src.len() as i64), 5);
        let mut dst = [0i8; 5];
        assert_eq!(wukong_rt_read_i8(ptr, dst.as_mut_ptr(), 5), 5);
        assert_eq!(dst, src);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_io_roundtrip_u8() {
        let path = io_tmp("u8");
        let cp = cpath(&path);
        let ptr = cp.as_ptr() as *const u8;
        let src = [0u8, 1, 127, 128, 255, 42];
        assert_eq!(wukong_rt_write_u8(ptr, src.as_ptr(), src.len() as i64), 6);
        let mut dst = [0u8; 6];
        assert_eq!(wukong_rt_read_u8(ptr, dst.as_mut_ptr(), 6), 6);
        assert_eq!(dst, src);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_is_little_endian_on_disk() {
        // A round-trip alone would pass even if both sides used big-endian; pin the on-disk bytes.
        let path = io_tmp("le");
        let cp = cpath(&path);
        let src = [0x0102_0304i32, -1];
        assert_eq!(wukong_rt_write_i32(cp.as_ptr() as *const u8, src.as_ptr(), 2), 2);
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw, vec![0x04, 0x03, 0x02, 0x01, 0xff, 0xff, 0xff, 0xff]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_honors_min_len_avail() {
        let path = io_tmp("minlen");
        let cp = cpath(&path);
        let ptr = cp.as_ptr() as *const u8;
        let src = [10i64, 20, 30, 40];
        assert_eq!(wukong_rt_write_i64(ptr, src.as_ptr(), 4), 4);

        // buf larger than file: reads the 4 available, leaves the tail untouched.
        let mut big = [-1i64; 6];
        assert_eq!(wukong_rt_read_i64(ptr, big.as_mut_ptr(), 6), 4);
        assert_eq!(&big[..4], &src);
        assert_eq!(&big[4..], &[-1i64, -1]);

        // buf smaller than file: reads only the first 2.
        let mut small = [0i64; 2];
        assert_eq!(wukong_rt_read_i64(ptr, small.as_mut_ptr(), 2), 2);
        assert_eq!(small, [10, 20]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_ignores_trailing_partial_element() {
        // A file whose length is not a whole multiple of sizeof: the partial tail is dropped.
        let path = io_tmp("partial");
        let cp = cpath(&path);
        let ptr = cp.as_ptr() as *const u8;
        // 9 bytes = two whole i32 (8 bytes) + 1 stray byte.
        std::fs::write(&path, [1u8, 0, 0, 0, 2, 0, 0, 0, 0xEE]).unwrap();
        let mut dst = [-1i32; 4];
        assert_eq!(wukong_rt_read_i32(ptr, dst.as_mut_ptr(), 4), 2);
        assert_eq!(&dst[..2], &[1, 2]);
        assert_eq!(&dst[2..], &[-1, -1]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_nonexistent_returns_minus_one() {
        let path = io_tmp("nope"); // never created
        let cp = cpath(&path);
        let ptr = cp.as_ptr() as *const u8;
        let mut bf = [0f32; 4];
        assert_eq!(wukong_rt_read_f32(ptr, bf.as_mut_ptr(), 4), -1);
        assert_eq!(bf, [0.0; 4], "buf must be untouched on open failure");
        let mut bi = [0i32; 2];
        assert_eq!(wukong_rt_read_i32(ptr, bi.as_mut_ptr(), 2), -1);
        let mut bl = [0i64; 2];
        assert_eq!(wukong_rt_read_i64(ptr, bl.as_mut_ptr(), 2), -1);
        let mut bu = [0u8; 2];
        assert_eq!(wukong_rt_read_u8(ptr, bu.as_mut_ptr(), 2), -1);
    }

    #[test]
    fn read_nonpositive_len_returns_zero_without_touching_buf() {
        let path = io_tmp("zerolen");
        let cp = cpath(&path);
        let ptr = cp.as_ptr() as *const u8;
        assert_eq!(wukong_rt_write_u8(ptr, [7u8, 8, 9].as_ptr(), 3), 3);
        let mut buf = [0xAAu8; 3];
        assert_eq!(wukong_rt_read_u8(ptr, buf.as_mut_ptr(), 0), 0);
        assert_eq!(buf, [0xAA; 3], "buf must be untouched on len == 0");
        assert_eq!(wukong_rt_read_u8(ptr, buf.as_mut_ptr(), -5), 0);
        assert_eq!(buf, [0xAA; 3], "buf must be untouched on negative len");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_empty_truncates_and_returns_zero() {
        // Pre-populate the path, then a zero-length write must leave an empty file and return 0.
        let path = io_tmp("empty");
        let cp = cpath(&path);
        let ptr = cp.as_ptr() as *const u8;
        assert_eq!(wukong_rt_write_i32(ptr, [1i32, 2, 3].as_ptr(), 3), 3);
        let empty: [i32; 0] = [];
        assert_eq!(wukong_rt_write_i32(ptr, empty.as_ptr(), 0), 0);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0, "file must be truncated");
        std::fs::remove_file(&path).ok();
    }
}
