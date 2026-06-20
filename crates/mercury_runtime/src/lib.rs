//! `mercury_runtime` — the minimal runtime that Mercury programs link against.
//!
//! Kept deliberately tiny and allocation-explicit, matching the language's philosophy. The
//! interpreter calls these implementations directly; native (LLVM) builds link the same logic
//! compiled as a static library. Today it provides a bump [`Arena`], a CPU [`parallel_for`], and a
//! tuned [`mercury_sgemm`] (the matmul microkernel the compiler lowers a matmul nest to).

mod gemm;
pub use gemm::{
    mercury_sgemm, mercury_sgemm_nt, mercury_sgemm_nt_epi, mercury_sgemm_nt_epi_parallel,
    mercury_sgemm_nt_parallel, mercury_sgemm_parallel,
};

mod vmath;
pub use vmath::{
    mercury_vmath_bf16, mercury_vmath_f32, VM_ACOSH, VM_ASINH, VM_ATAN, VM_ATANH, VM_COS, VM_COSH,
    VM_ERF, VM_EXP, VM_EXP2, VM_EXPM1, VM_GELU, VM_LOG, VM_LOG1P, VM_LOG2, VM_RELU, VM_SIGMOID,
    VM_SILU, VM_SIN, VM_SINH, VM_TANH,
};

mod velem;
pub use velem::{mercury_velem_f32, mercury_vhorner_f32, VE_ID, VE_RELU, VE_RELU6, VE_USE_Y};

mod reduce;
pub use reduce::{
    mercury_sreduce_f32, mercury_sreduce_f32_parallel, RED_DOT, RED_SSD, RED_SUM, RED_SUMSQ,
};

mod norm;
pub use norm::{
    mercury_norm_affine_f32, mercury_norm_affine_f32_parallel, mercury_norm_f32,
    mercury_norm_f32_parallel, NORM_LAYERNORM, NORM_RMSNORM, NORM_SOFTMAX,
};

mod i8gemm;
pub use i8gemm::{mercury_i8gemm_nt, mercury_i8gemm_nt_parallel};

mod lowp;
pub use lowp::{
    mercury_axpby_bf16, mercury_dot_bf16, mercury_dot_f16, mercury_sum_bf16, mercury_sum_f16,
};

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
/// pointee outlives the (blocking) `mercury_parallel_for` call, and worker chunks touch disjoint
/// output indices, so there is no data race.
#[derive(Clone, Copy)]
struct EnvAddr(usize);
unsafe impl Send for EnvAddr {}
unsafe impl Sync for EnvAddr {}

/// The C-ABI parallel-for the native backend lowers `@parallel for` to. Splits `[0, n)` into one
/// contiguous chunk per host CPU and runs `body(start, end, env)` on each concurrently (via a
/// persistent thread pool), returning only once every chunk has completed.
///
/// Bodies must be data-parallel: each index is processed exactly once and the chunks must not have
/// cross-iteration dependencies (the interpreter runs the whole range sequentially and must agree).
///
/// # Safety
/// `body` must be a valid `extern "C" fn(i64, i64, *const u8)` and `env` valid for the call.
#[no_mangle]
pub unsafe extern "C" fn mercury_parallel_for(
    n: i64,
    body: extern "C" fn(i64, i64, *const u8),
    env: *const u8,
) {
    use rayon::prelude::*;
    if n <= 0 {
        return;
    }
    let n = n as usize;
    let workers = rayon::current_num_threads().max(1).min(n);
    let chunk = n.div_ceil(workers);
    let env = EnvAddr(env as usize);
    (0..workers).into_par_iter().for_each(|w| {
        let start = w * chunk;
        if start >= n {
            return;
        }
        let end = (start + chunk).min(n);
        body(start as i64, end as i64, env.0 as *const u8);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

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
            mercury_parallel_for(n as i64, fill_squares, buf.as_mut_ptr() as *const u8);
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
}
