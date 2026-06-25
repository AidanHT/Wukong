//! CUDA **graph capture / replay** (milestone M7) — turn a fixed launch sequence into one driver
//! call.
//!
//! A resident transformer layer issues ~13–16 individual kernel launches per forward. At decode /
//! small-batch sizes each kernel runs for only microseconds, so the per-launch driver overhead
//! (`cuLaunchKernel` × 16) is a large fraction of the layer's wall time. CUDA graphs remove it:
//! capture the sequence **once** into a `CUgraph`, instantiate an executable `CUgraphExec`, then
//! **replay** the whole thing with a single `cuGraphLaunch`.
//!
//! ## Why the pool is a prerequisite
//! `cuStreamBeginCapture` rejects any *synchronizing* call inside the captured region — most of all a
//! `cuMemAlloc*`. A per-op `alloc_zeros` forward therefore cannot be captured. Drawing all scratch
//! from a [`DevicePool`](crate::pool) *before* capture (a host-side cursor bump issues no driver call)
//! leaves the region containing nothing but launches, which records cleanly. So pool + graph compose:
//! the pool removes the alloc overhead, the graph removes the launch overhead.
//!
//! ## Why a dedicated stream
//! `cudarc`'s `default_stream()` is the **legacy NULL stream**, and CUDA forbids capturing it. So
//! capture/replay run on a dedicated non-blocking stream ([`CudaContext::new_stream`]). Because a
//! non-blocking stream does *not* implicitly synchronize with the NULL stream, the caller must ensure
//! any prior NULL-stream work (weight upload, input `memcpy`) is visible before replay — a single
//! context synchronize suffices and is done once, outside the timed region.
//!
//! ## Surface
//! `cudarc` 0.16 *does* expose a safe `CudaGraph`, but its `end_capture` forces the
//! `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH` flag (the enum has no zero variant). This graph
//! has no graph-ordered allocations, so we want plain `flags = 0`; we therefore drive the raw driver
//! entry points (`result::stream::{begin,end}_capture`, `cuGraphInstantiateWithFlags`,
//! `result::graph::{launch,exec_destroy,destroy}`) directly — the same raw-`sys` style `cubin.rs` uses
//! for `cuLink*` — and own the `Drop`.

use std::sync::Arc;

use cudarc::driver::{
    result, sys, CudaContext, CudaStream, DeviceRepr, DriverError, PinnedHostSlice, ValidAsZeroBits,
};

/// A page-locked (**pinned**) host staging buffer for truly-async H2D/D2H.
///
/// Copies from *pageable* host memory force the driver to bounce through an internal pinned staging
/// buffer and typically serialize against compute; copies from **pinned** memory run concurrently on a
/// dedicated copy stream while another stream computes — the basis of [copy/compute
/// overlap](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#asynchronous-transfers-and-overlapping-transfers-with-computation).
/// Wraps cudarc's `alloc_pinned` (write-combined — ideal for the H2D upload direction; D2H readback
/// also works, only host *reads* of the result pay the write-combined penalty, outside any timed loop).
pub struct PinnedBuf<T> {
    inner: PinnedHostSlice<T>,
}

impl<T: DeviceRepr + ValidAsZeroBits> PinnedBuf<T> {
    /// Allocate `len` elements of page-locked host memory.
    pub fn alloc(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, DriverError> {
        // SAFETY: the memory is left uninitialized; we always fill it (`copy_from_slice`) before any
        // device read, and only read it back after a D2H completes.
        let inner = unsafe { ctx.alloc_pinned::<T>(len)? };
        Ok(Self { inner })
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True iff empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<T: DeviceRepr + ValidAsZeroBits + Copy> PinnedBuf<T> {
    /// Fill the pinned buffer from a host slice (host-side copy; syncs the buffer's event first).
    pub fn copy_from_slice(&mut self, src: &[T]) -> Result<(), DriverError> {
        assert_eq!(src.len(), self.inner.len(), "PinnedBuf::copy_from_slice length mismatch");
        let dst = self.inner.as_mut_ptr()?;
        // SAFETY: `dst` has room for `len` elements (asserted); `src` is a valid host slice.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        Ok(())
    }

    /// Read the pinned buffer into a host `Vec` (host-side copy; syncs the buffer's event first).
    pub fn to_vec(&self) -> Result<Vec<T>, DriverError> {
        let src = self.inner.as_ptr()?;
        let mut out = Vec::with_capacity(self.inner.len());
        // SAFETY: `src` points at `len` initialized elements once a prior D2H has completed (the
        // caller synchronizes the copy stream before calling this).
        unsafe {
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), self.inner.len());
            out.set_len(self.inner.len());
        }
        Ok(out)
    }
}

impl<T> std::ops::Deref for PinnedBuf<T> {
    type Target = PinnedHostSlice<T>;
    fn deref(&self) -> &PinnedHostSlice<T> {
        &self.inner
    }
}

impl<T> std::ops::DerefMut for PinnedBuf<T> {
    fn deref_mut(&mut self) -> &mut PinnedHostSlice<T> {
        &mut self.inner
    }
}

/// A captured, instantiated, replayable launch sequence. Replay with [`launch`](Self::launch); the
/// `CUgraph` + `CUgraphExec` are destroyed on drop. Not internally synchronized (the whole GPU harness
/// already funnels through one `Mutex`, and CUDA graph objects must not be touched concurrently).
pub struct Graph {
    stream: Arc<CudaStream>,
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
}

impl Graph {
    /// Capture the launches that `record` issues **on `stream`** into a replayable graph. `record` must
    /// issue only stream-ordered work (kernel launches, memsets) and **no synchronizing allocation** —
    /// draw every buffer from a [`DevicePool`](crate::pool) beforehand. `stream` must be a real
    /// (non-NULL) stream — see the module docs.
    ///
    /// Capture is always ended even if `record` fails, so the stream is never left in capturing state;
    /// the `record` error is then surfaced.
    pub fn capture(
        stream: Arc<CudaStream>,
        record: impl FnOnce() -> Result<(), DriverError>,
    ) -> Result<Self, DriverError> {
        stream.context().bind_to_thread()?;
        let raw = stream.cu_stream();
        // THREAD_LOCAL: capture only the work this thread issues on the stream (everything is already
        // serialized under the harness mutex, so this is the strict, race-free choice).
        unsafe {
            result::stream::begin_capture(
                raw,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?
        };
        let rec = record();
        // End capture regardless of `rec`, so a failed record cannot strand the stream mid-capture.
        let ended = unsafe { result::stream::end_capture(raw) };
        rec?;
        let graph = ended?;
        if graph.is_null() {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_ILLEGAL_STATE));
        }
        // Instantiate with flags = 0 (no auto-free; this graph owns no graph-ordered memory). Raw call
        // because cudarc's safe wrapper hard-codes the AUTO_FREE flag.
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let inst = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0).result() };
        if let Err(e) = inst {
            unsafe {
                let _ = result::graph::destroy(graph);
            }
            return Err(e);
        }
        Ok(Self { stream, graph, exec })
    }

    /// Replay the whole captured sequence with **one** `cuGraphLaunch` on the capture stream. The
    /// caller synchronizes the stream when it needs the results.
    pub fn launch(&self) -> Result<(), DriverError> {
        self.stream.context().bind_to_thread()?;
        unsafe { result::graph::launch(self.exec, self.stream.cu_stream()) }
    }

    /// The stream this graph was captured on and replays on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // Destroy the executable first, then the graph template (mirrors cudarc's CudaGraph drop).
        // Best-effort at teardown.
        let _ = self.stream.context().bind_to_thread();
        unsafe {
            if !self.exec.is_null() {
                let _ = result::graph::exec_destroy(self.exec);
            }
            if !self.graph.is_null() {
                let _ = result::graph::destroy(self.graph);
            }
        }
    }
}
