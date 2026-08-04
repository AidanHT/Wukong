//! Device memory **pool** — bump-arena sub-allocation over one big device slab (milestone M7).
//!
//! Unpooled, every resident op draws its scratch from `stream.alloc_zeros::<T>(n)`, which in `cudarc`
//! 0.16 is a stream-ordered `cuMemAllocAsync` + a `cuMemsetD8Async` (see `CudaStream::alloc_zeros`),
//! and frees it again on drop with `cuMemFreeAsync` (see `CudaSlice::drop`). At decode / small-batch
//! sizes — where each kernel touches only a few KB and the layer is replayed thousands of times —
//! that per-op allocate/free traffic dominates the actual compute.
//!
//! [`DevicePool`] removes it. One `cuMemAllocAsync` reserves a slab up front; [`alloc`](DevicePool::alloc)
//! hands out a sub-range by bumping a cursor (no driver call), and [`reset`](DevicePool::reset)
//! reclaims **everything** in O(1) for the next iteration. A high-water mark tracks the real peak so
//! the 6 GB part's budget stays visible.
//!
//! ## Why this is also the key that unlocks CUDA-graph capture
//! `cuStreamBeginCapture` forbids the synchronizing allocations a per-op `alloc_zeros` issues — a
//! `cuMemAlloc*` inside the captured region invalidates the capture. Pre-allocating **all** of a
//! layer's scratch from this pool *before* capture leaves the captured region containing nothing but
//! kernel launches, which records cleanly into a replayable graph (see [`crate::graph`]). So the pool
//! is not merely an optimization; it is the prerequisite for the launch-overhead win on top of it.
//!
//! ## Safety model
//! The slab is the single owner of the device memory. A handed-out [`PoolBuf`] is a *view* into the
//! slab wrapped in a real `CudaSlice<T>` (via the documented `leak()` → `upgrade_device_ptr()` round
//! trip), so it launches through the existing wrappers unchanged — **but on drop it `leak()`s rather
//! than frees**, because `CudaSlice::drop`'s `cuMemFreeAsync` would free a sub-pointer the driver
//! never handed out as its own allocation (a corruption / double-free). The slab itself is freed
//! exactly once when the `DevicePool` drops.

use std::sync::Arc;

use cudarc::driver::{result, sys, CudaSlice, CudaStream, DriverError};

/// Default sub-allocation alignment: 256 B. Comfortably covers every device type we hand out
/// (f32/f16/u8/…) and keeps each buffer's base on a 128-byte memory-transaction sector boundary, so
/// pooled buffers coalesce exactly like a fresh `cuMemAlloc` (which is 256-byte aligned) would.
pub const DEFAULT_ALIGN: usize = 256;

/// Aligned bump arithmetic, factored out as a pure function so the offset/alignment/overflow logic is
/// unit-testable with no device present. `align` must be a power of two. Returns the `[start, end)`
/// byte sub-range the next allocation of `bytes` occupies after aligning `offset` up.
#[inline]
fn aligned_range(offset: usize, align: usize, bytes: usize) -> (usize, usize) {
    debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
    let start = (offset + align - 1) & !(align - 1);
    (start, start + bytes)
}

/// A sub-allocation handed out by [`DevicePool`]. Derefs to a `CudaSlice<T>`, so it feeds the existing
/// `launch_builder(...).arg(&*buf)` / `arg(&mut *buf)` path with no change to any kernel launcher.
///
/// **Drop leaks, it does not free.** The backing bytes belong to the pool's slab; this handle must not
/// issue the `cuMemFreeAsync` that a normal `CudaSlice` drop would (that would free a slab interior
/// pointer). See the module docs.
pub struct PoolBuf<T> {
    inner: Option<CudaSlice<T>>,
}

impl<T> PoolBuf<T> {
    /// The number of elements of `T` in this sub-buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.as_ref().map_or(0, CudaSlice::len)
    }

    /// True iff the sub-buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> std::ops::Deref for PoolBuf<T> {
    type Target = CudaSlice<T>;
    #[inline]
    fn deref(&self) -> &CudaSlice<T> {
        // `inner` is `Some` for the whole lifetime; only `Drop` takes it.
        self.inner.as_ref().expect("PoolBuf used after drop")
    }
}

impl<T> std::ops::DerefMut for PoolBuf<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut CudaSlice<T> {
        self.inner.as_mut().expect("PoolBuf used after drop")
    }
}

impl<T> Drop for PoolBuf<T> {
    fn drop(&mut self) {
        if let Some(s) = self.inner.take() {
            // `leak` forgets the slice (no `cuMemFreeAsync`) and balances the `Arc<CudaStream>`
            // increment `upgrade_device_ptr` did — the slab still owns these bytes.
            let _ = s.leak();
        }
    }
}

/// A bump-allocated device-memory arena. One slab, O(1) reset, single high-water mark. Not `Clone`;
/// not internally synchronized (the whole GPU harness already funnels through one `Mutex`).
pub struct DevicePool {
    stream: Arc<CudaStream>,
    /// Base device pointer of the slab (owned raw — freed once in `Drop`, never per-op).
    base: sys::CUdeviceptr,
    cap: usize,
    offset: usize,
    high_water: usize,
    align: usize,
    served: u64,
}

impl DevicePool {
    /// Reserve a `cap_bytes` slab on `stream` (one `cuMemAllocAsync`, zeroed once). Sub-allocations
    /// then come from the slab with no further driver allocation until [`reset`](Self::reset) +
    /// re-fill or `Drop`.
    pub fn new(stream: Arc<CudaStream>, cap_bytes: usize) -> Result<Self, DriverError> {
        Self::with_align(stream, cap_bytes, DEFAULT_ALIGN)
    }

    /// As [`new`](Self::new) with an explicit power-of-two `align`.
    pub fn with_align(
        stream: Arc<CudaStream>,
        cap_bytes: usize,
        align: usize,
    ) -> Result<Self, DriverError> {
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        // One real allocation, zeroed once. `leak()` hands us the raw pointer and suppresses the
        // automatic free, so the pool owns the bytes until `Drop` frees them exactly once.
        let slab = stream.alloc_zeros::<u8>(cap_bytes)?;
        let base = slab.leak();
        Ok(Self {
            stream,
            base,
            cap: cap_bytes,
            offset: 0,
            high_water: 0,
            align,
            served: 0,
        })
    }

    /// Bump the cursor by `n * size_of::<T>()` (aligned) and return the sub-range base pointer.
    /// `CUDA_ERROR_OUT_OF_MEMORY` if it would overflow the slab — surface it and grow `cap`.
    fn bump<T>(&mut self, n: usize) -> Result<sys::CUdeviceptr, DriverError> {
        let bytes = n * std::mem::size_of::<T>();
        let (start, end) = aligned_range(self.offset, self.align, bytes);
        if end > self.cap {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY));
        }
        self.offset = end;
        if end > self.high_water {
            self.high_water = end;
        }
        self.served += 1;
        Ok(self.base + start as sys::CUdeviceptr)
    }

    /// Hand out an **uninitialized** `[T; n]` sub-buffer (its contents are whatever the previous
    /// iteration left there). Correct for any kernel that fully overwrites its output — the common
    /// case — and the cheapest path (no memset). For an accumulated / partially-written output, use
    /// [`alloc_zeros`](Self::alloc_zeros).
    pub fn alloc<T>(&mut self, n: usize) -> Result<PoolBuf<T>, DriverError> {
        let ptr = self.bump::<T>(n)?;
        // SAFETY: `ptr` is a valid, `align`-aligned sub-range of the live slab with room for
        // `n * size_of::<T>()` bytes; `PoolBuf::drop` leaks (never frees) it, so the slab keeps
        // sole ownership.
        let s = unsafe { self.stream.upgrade_device_ptr::<T>(ptr, n) };
        Ok(PoolBuf { inner: Some(s) })
    }

    /// Hand out a **zeroed** `[T; n]` sub-buffer (a `cuMemsetD8Async` on the pool's stream, exactly
    /// what `alloc_zeros` does). Use only where the kernel reads or accumulates into its output, so a
    /// pooled run stays bit-identical to the per-op-`alloc_zeros` baseline.
    pub fn alloc_zeros<T>(&mut self, n: usize) -> Result<PoolBuf<T>, DriverError> {
        let ptr = self.bump::<T>(n)?;
        let bytes = n * std::mem::size_of::<T>();
        // SAFETY: `ptr`/`bytes` describe a live sub-range of the slab; the memset is ordered on the
        // pool's stream ahead of any later launch into this buffer.
        unsafe { result::memset_d8_async(ptr, 0, bytes, self.stream.cu_stream())? };
        let s = unsafe { self.stream.upgrade_device_ptr::<T>(ptr, n) };
        Ok(PoolBuf { inner: Some(s) })
    }

    /// Reclaim the whole arena in O(1). Every previously handed-out [`PoolBuf`] must already be
    /// dropped (each leaks, not frees); the same byte ranges are then re-handed-out next fill — which
    /// is exactly why a captured graph can replay against stable pointers.
    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Fill the **entire** slab with `byte` (a `cuMemsetD8Async` on the pool's stream). A correctness
    /// aid: poison with a NaN-ish pattern (e.g. `0xFF`) before a pooled run so any buffer the run reads
    /// *without* first fully writing it surfaces as a NaN mismatch against the per-op-`alloc_zeros`
    /// baseline — i.e. it proves the [`alloc`](Self::alloc) (uninitialized) fast path is used only for
    /// genuinely full-overwrite outputs. Steady-state replay leaves the slab dirty with the previous
    /// iteration's data anyway, so this just makes that condition deterministic and hostile.
    pub fn poison(&mut self, byte: u8) -> Result<(), DriverError> {
        // SAFETY: `base`/`cap` describe the whole live slab; the memset is ordered on the pool's
        // stream ahead of any later launch.
        unsafe { result::memset_d8_async(self.base, byte, self.cap, self.stream.cu_stream()) }
    }

    /// Peak bytes ever simultaneously live (the real footprint to budget against the 6 GB part).
    #[inline]
    pub fn high_water_bytes(&self) -> usize {
        self.high_water
    }

    /// Bytes currently bumped (live this fill).
    #[inline]
    pub fn in_use_bytes(&self) -> usize {
        self.offset
    }

    /// Total slab capacity.
    #[inline]
    pub fn capacity_bytes(&self) -> usize {
        self.cap
    }

    /// Count of sub-allocations served over the pool's life (a launch/alloc-pressure stat).
    #[inline]
    pub fn served(&self) -> u64 {
        self.served
    }
}

impl Drop for DevicePool {
    fn drop(&mut self) {
        // Free the slab exactly once. Synchronize first so no in-flight kernel still reads it, and
        // bind the context (a raw `cuMemFree` needs it current on this thread). Both are best-effort
        // at teardown — record nothing, the process is releasing the device anyway.
        let _ = self.stream.synchronize();
        let _ = self.stream.context().bind_to_thread();
        // SAFETY: `base` came from a single slab allocation and is freed exactly once here; all
        // sub-buffers were leaked (never freed), so this is the sole free of these bytes.
        unsafe {
            let _ = result::free_sync(self.base);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::aligned_range;

    // Pure bump/alignment math — no device required, so it runs even on a GPU-less box (when the
    // crate is built `--features gpu`).
    #[test]
    fn aligned_range_rounds_start_up_and_packs_tightly() {
        // From 0, aligned start is 0.
        assert_eq!(aligned_range(0, 256, 100), (0, 100));
        // Next alloc rounds 100 up to 256 before placing 40 bytes.
        assert_eq!(aligned_range(100, 256, 40), (256, 296));
        // Already-aligned offset is untouched.
        assert_eq!(aligned_range(512, 256, 8), (512, 520));
        // Tight alignment (1) never pads.
        assert_eq!(aligned_range(7, 1, 5), (7, 12));
        // Power-of-two alignments other than 256.
        assert_eq!(aligned_range(1, 16, 16), (16, 32));
        assert_eq!(aligned_range(33, 32, 1), (64, 65));
    }
}
