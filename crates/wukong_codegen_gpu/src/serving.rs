//! Batched autoregressive **decode** layer/model over the [paged KV-cache](crate::paged_kv) — the
//! serving forward pass.
//!
//! [`crate::gpu::ResidentLayerF16`] is a *prefill* layer: it takes a fixed `[S,D]` and recomputes Q/K/V
//! for the whole sequence every call, with no cache. Serving throughput is a *decode* metric — one new
//! token per sequence, attending over a cached past — so this module adds the missing half:
//!
//! - [`DecodeLayer`] — one pre-norm transformer layer run on a **batch of `Bcap` single-token rows**.
//!   The Q/K/V/O projections and the FFN are the *same* tuned WMMA f16 GEMMs the prefill layer uses
//!   (M = `Bcap`, a fixed multiple of 64 ⇒ static shapes, graph-capturable, matching how real engines
//!   bucket batch sizes). The new step: the projected K/V are **appended to the paged cache**
//!   ([`crate::paged_attention::launch_kv_append`]) and attention is the **paged decode kernel**
//!   ([`crate::paged_attention::launch_paged_attn_decode`]) over each sequence's ragged context — *not*
//!   a full-sequence flash. Selective batching (Orca): the token-wise ops batch across all sequences,
//!   attention is per-sequence against its own block table.
//! - [`DecodeModel`] — an `N`-layer stack owning the [`PagedKvCache`], a [`DevicePool`] for per-layer
//!   scratch, the per-step block-table / context-length / write-position device buffers, and the
//!   `[Bcap,D]` ping-pong activations. One [`step_on`](DecodeModel::step_on) advances every active
//!   sequence by one token: append a cache slot per sequence (host), upload the metadata once (shared
//!   across layers), then run all `N` layers GPU-resident. This is the unit a whole-model CUDA graph
//!   captures (P4) and the scheduler below drives (P5).
//! - `Scheduler` — Orca-style **continuous (in-flight) batching** over one fixed-`Bcap` `DecodeModel`:
//!   `Request`s are admitted into free slots (first-fit with bounded look-ahead), `Scheduler::step`
//!   advances every active slot one token, and a finished sequence is evicted and its slot refilled the
//!   same iteration. `Scheduler::step_graphed` replays the whole-step CUDA graph, captured once
//!   (admission/eviction only changes device-buffer *contents*). `Scheduler::new_static` is the same
//!   machinery with static-batching admission — the honest peer the goodput comparison is made against.
//!
//! Cache storage is f16 by default; `new_with_dtype` swaps in the int8 append/attention kernel pair
//! (lossy, tolerance-gated) with everything else — including graph capture — unchanged.
//!
//! Pooled + on-an-explicit-stream throughout ([`DecodeLayer::forward_step_on`]) so the whole decode
//! step records cleanly into one [`crate::graph::Graph`] (no synchronizing alloc inside the capture).

use std::sync::Arc;

use cudarc::driver::{
    sys, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use half::f16;

use crate::gpu::{Gpu, TransformerWeights};
use crate::paged_attention::{
    kv_append_int8_ptx, kv_append_ptx, launch_kv_append, launch_kv_append_int8,
    launch_paged_attn_decode, launch_paged_attn_decode_int8, paged_attn_decode_int8_ptx,
    paged_attn_decode_ptx, KV_APPEND_ENTRY, KV_APPEND_INT8_ENTRY, PAGED_ATTN_ENTRY,
    PAGED_ATTN_INT8_ENTRY,
};
use crate::paged_kv::{KvConfig, KvDtype, KvStorage, PagedKvCache};
use crate::pool::{DevicePool, PoolBuf};

/// Launch config for the shared-memory-staged WMMA kernels (`*_sm_db`), one CTA per `SM_BM×SM_BN`
/// output tile. Replicated read-only from `gpu.rs`'s private helper (the tile constants are `pub`).
fn wmma_sm_cfg(m: usize, n: usize) -> LaunchConfig {
    use crate::ptx_wmma::{SM_BM, SM_BN, SM_THREADS};
    LaunchConfig {
        grid_dim: ((n / SM_BN) as u32, (m / SM_BM) as u32, 1),
        block_dim: (SM_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One pre-norm transformer **decode** layer: RMSNorm → Q/K/V proj → append-to-cache → paged attention
/// → O-proj(+residual) → RMSNorm → SiLU FFN-up → FFN-down(+residual), all on a batch of `Bcap` rows.
/// Weights upload (f16) and kernels load once at construction.
pub struct DecodeLayer {
    f_norm: CudaFunction,
    f_cast: CudaFunction,
    f_gemm: CudaFunction,
    f_silu: CudaFunction,
    f_resid: CudaFunction,
    f_attn: CudaFunction,
    f_append: CudaFunction,
    wq: CudaSlice<f16>,
    wk: CudaSlice<f16>,
    wv: CudaSlice<f16>,
    wo: CudaSlice<f16>,
    w1: CudaSlice<f16>,
    w2: CudaSlice<f16>,
    cfg: KvConfig,
    dff: usize,
    eps: f32,
    scale: f32,
    /// Cache storage dtype this layer's append/attention kernels were loaded for — must match the
    /// [`KvStorage`] arm handed to [`forward_step_on`](Self::forward_step_on).
    dtype: KvDtype,
}

impl DecodeLayer {
    /// `D = heads * head_dim` (the hidden size).
    #[inline]
    pub fn d(&self) -> usize {
        self.cfg.heads * self.cfg.head_dim
    }
    /// `Bcap` — the fixed decode batch (== cache slots).
    #[inline]
    pub fn bcap(&self) -> usize {
        self.cfg.num_slots
    }

    /// Upload the weights (narrowed to f16) and load every kernel for **f16 cache storage** (the
    /// default, bit-exact path). `cfg` is the cache geometry; `dff` the FFN inner dim. Requires
    /// `D % 64 == 0`, `Dff % 64 == 0`, `Bcap % 64 == 0` (the 64×64 WMMA tile), and
    /// `head_dim ∈ {64, 128}` (the generated paged-attn kernels).
    pub fn new(g: &mut Gpu, w: &TransformerWeights, cfg: KvConfig, dff: usize) -> Result<Self, DriverError> {
        Self::new_with_dtype(g, w, cfg, dff, KvDtype::F16)
    }

    /// As [`new`](Self::new) with an explicit cache storage dtype: `Int8` loads the int8
    /// append/attention kernel pair instead (distinct module-cache keys — a shared key would hand
    /// back the wrong cached kernel). Everything else (weights, GEMMs, norms) is dtype-independent.
    pub fn new_with_dtype(
        g: &mut Gpu,
        w: &TransformerWeights,
        cfg: KvConfig,
        dff: usize,
        dtype: KvDtype,
    ) -> Result<Self, DriverError> {
        let d = cfg.heads * cfg.head_dim;
        assert!(d % 64 == 0 && dff % 64 == 0, "D and Dff must be multiples of 64 (WMMA tile)");
        assert!(cfg.num_slots % 64 == 0, "Bcap (num_slots) must be a multiple of 64 (WMMA M tile)");
        assert!(matches!(cfg.head_dim, 64 | 128), "head_dim must be 64 or 128 (generated paged-attn kernels)");
        for (name, wt, len) in [
            ("wq", w.wq, d * d),
            ("wk", w.wk, d * d),
            ("wv", w.wv, d * d),
            ("wo", w.wo, d * d),
            ("w1", w.w1, dff * d),
            ("w2", w.w2, d * dff),
        ] {
            assert_eq!(wt.len(), len, "{name} wrong size");
        }
        let f_norm = g.function("norm", crate::ptx_norm::norm_ptx(), "rmsnorm")?;
        let f_cast = g.function("cast", crate::ptx::CAST_F32_F16, "cast_f32_f16")?;
        let wmma = crate::ptx_wmma::wmma_f16_ptx();
        let f_gemm = g.function("wmma_f16", wmma, "wmma_nt_f16_sm_db")?;
        let f_silu = g.function("wmma_f16", wmma, "wmma_nt_f16_sm_db_silu")?;
        let f_resid = g.function("wmma_f16", wmma, "wmma_nt_f16_sm_db_residual")?;
        let (f_attn, f_append) = match dtype {
            KvDtype::F16 => {
                let attn_key: &'static str = match cfg.head_dim {
                    64 => "paged_attn_d64",
                    128 => "paged_attn_d128",
                    _ => unreachable!(),
                };
                (
                    g.function(attn_key, &paged_attn_decode_ptx(cfg.head_dim), PAGED_ATTN_ENTRY)?,
                    g.function("kv_append", &kv_append_ptx(), KV_APPEND_ENTRY)?,
                )
            }
            KvDtype::Int8 => {
                let attn_key: &'static str = match cfg.head_dim {
                    64 => "paged_attn_int8_d64",
                    128 => "paged_attn_int8_d128",
                    _ => unreachable!(),
                };
                (
                    g.function(attn_key, &paged_attn_decode_int8_ptx(cfg.head_dim), PAGED_ATTN_INT8_ENTRY)?,
                    g.function("kv_append_int8", &kv_append_int8_ptx(), KV_APPEND_INT8_ENTRY)?,
                )
            }
        };
        let to16 = |wt: &[f32]| -> Vec<f16> { wt.iter().map(|&v| f16::from_f32(v)).collect() };
        let wq = g.stream.memcpy_stod(&to16(w.wq))?;
        let wk = g.stream.memcpy_stod(&to16(w.wk))?;
        let wv = g.stream.memcpy_stod(&to16(w.wv))?;
        let wo = g.stream.memcpy_stod(&to16(w.wo))?;
        let w1 = g.stream.memcpy_stod(&to16(w.w1))?;
        let w2 = g.stream.memcpy_stod(&to16(w.w2))?;
        Ok(Self {
            f_norm,
            f_cast,
            f_gemm,
            f_silu,
            f_resid,
            f_attn,
            f_append,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            cfg,
            dff,
            eps: 1e-5,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            dtype,
        })
    }

    /// Run one decode step for this layer on `stream`, all scratch from `pool`. `x_d` is the `[Bcap,D]`
    /// input activation; the new K/V are appended into `kv` at `wpos_d[slot]` (which must already be
    /// reserved) for layer plane `layer`; attention reads the cache over `cl_d[slot]` (= post-append
    /// context). `kv`'s storage dtype must match the dtype this layer was built with (the kernels are
    /// loaded per dtype — asserted). The `[Bcap,D]` result is written into `out`. Every pooled buffer
    /// is a full-overwrite output (uninit `alloc` is safe — proven by the poison gate).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step_on(
        &self,
        stream: &Arc<CudaStream>,
        pool: &mut DevicePool,
        kv: &mut KvStorage,
        x_d: &CudaSlice<f32>,
        bt_d: &CudaSlice<u32>,
        cl_d: &CudaSlice<u32>,
        wpos_d: &CudaSlice<u32>,
        active_d: &CudaSlice<u32>,
        layer: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), DriverError> {
        assert_eq!(self.dtype, kv.dtype(), "cache storage dtype must match the layer's kernels");
        let (bcap, d, dff, eps) = (self.bcap(), self.d(), self.dff, self.eps);
        debug_assert_eq!(x_d.len(), bcap * d);
        debug_assert_eq!(out.len(), bcap * d);
        let norm_cfg = LaunchConfig { grid_dim: (bcap as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };

        let norm = |pool: &mut DevicePool, src: &CudaSlice<f32>, rows: usize| -> Result<PoolBuf<f32>, DriverError> {
            let mut o = pool.alloc::<f32>(rows * d)?;
            let (r, c) = (rows as u32, d as u32);
            let mut b = stream.launch_builder(&self.f_norm);
            b.arg(&r).arg(&c).arg(&eps).arg(src).arg(&mut *o);
            unsafe { b.launch(norm_cfg)? };
            Ok(o)
        };
        let cast = |pool: &mut DevicePool, src: &CudaSlice<f32>, n: usize| -> Result<PoolBuf<f16>, DriverError> {
            let mut dst = pool.alloc::<f16>(n)?;
            let nn = n as u32;
            let mut b = stream.launch_builder(&self.f_cast);
            b.arg(&nn).arg(src).arg(&mut *dst);
            unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
            Ok(dst)
        };
        let gemm16 = |pool: &mut DevicePool, f: &CudaFunction, a: &CudaSlice<f16>, b: &CudaSlice<f16>, m: usize, k: usize, n: usize| -> Result<PoolBuf<f32>, DriverError> {
            let mut c = pool.alloc::<f32>(m * n)?;
            let (mm, nn, kk) = (m as u32, n as u32, k as u32);
            let mut bld = stream.launch_builder(f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a).arg(b).arg(&mut *c);
            unsafe { bld.launch(wmma_sm_cfg(m, n))? };
            Ok(c)
        };
        let resid_gemm = |pool: &mut DevicePool, a: &CudaSlice<f16>, b: &CudaSlice<f16>, residual: &CudaSlice<f32>, m: usize, k: usize, n: usize| -> Result<PoolBuf<f32>, DriverError> {
            let mut c = pool.alloc::<f32>(m * n)?;
            let (mm, nn, kk) = (m as u32, n as u32, k as u32);
            let mut bld = stream.launch_builder(&self.f_resid);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a).arg(b).arg(&mut *c).arg(residual);
            unsafe { bld.launch(wmma_sm_cfg(m, n))? };
            Ok(c)
        };

        // --- attention: norm → fp16 Q/K/V projections → append new K/V → paged decode attention ---
        let h1 = norm(pool, x_d, bcap)?;
        let h1_16 = cast(pool, &h1, bcap * d)?;
        let q = gemm16(pool, &self.f_gemm, &h1_16, &self.wq, bcap, d, d)?;
        let k = gemm16(pool, &self.f_gemm, &h1_16, &self.wk, bcap, d, d)?;
        let v = gemm16(pool, &self.f_gemm, &h1_16, &self.wv, bcap, d, d)?;
        // Append the new token's K/V into this layer's cache plane at each slot's write position,
        // then paged decode attention over the (now-updated) cache — the one dtype-dispatched pair.
        // `attn` is pooled *before* the match so the bump sequence is identical on both arms (a
        // captured graph bakes the pooled addresses).
        let mut attn = pool.alloc::<f32>(bcap * d)?;
        match kv {
            KvStorage::F16 { k: kc, v: vc } => {
                launch_kv_append(stream, &self.f_append, &k, &v, kc, vc, bt_d, wpos_d, active_d, &self.cfg, layer, bcap)?;
                launch_paged_attn_decode(stream, &self.f_attn, &q, kc, vc, &mut attn, bt_d, cl_d, &self.cfg, layer, bcap, self.scale)?;
            }
            KvStorage::Int8 { k: kc, v: vc, ksc, vsc } => {
                launch_kv_append_int8(stream, &self.f_append, &k, &v, kc, vc, ksc, vsc, bt_d, wpos_d, active_d, &self.cfg, layer, bcap)?;
                launch_paged_attn_decode_int8(stream, &self.f_attn, &q, kc, vc, ksc, vsc, &mut attn, bt_d, cl_d, &self.cfg, layer, bcap, self.scale)?;
            }
        }
        let attn_16 = cast(pool, &attn, bcap * d)?;
        let x1 = resid_gemm(pool, &attn_16, &self.wo, x_d, bcap, d, d)?; // x + attn·Woᵀ (residual 1)

        // --- FFN: RMSNorm → SiLU up-projection (fused) → down-projection with residual into `out` ---
        let h2 = norm(pool, &x1, bcap)?;
        let h2_16 = cast(pool, &h2, bcap * d)?;
        let f1 = gemm16(pool, &self.f_silu, &h2_16, &self.w1, bcap, d, dff)?; // SiLU(h2·W1ᵀ)
        let f1_16 = cast(pool, &f1, bcap * dff)?;
        {
            let (mm, nn, kk) = (bcap as u32, d as u32, dff as u32);
            let mut bld = stream.launch_builder(&self.f_resid);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(&*f1_16).arg(&self.w2).arg(&mut *out).arg(&*x1);
            unsafe { bld.launch(wmma_sm_cfg(bcap, d))? };
        }
        Ok(())
    }
}

/// An `N`-layer decode model, whole-stack GPU-resident, owning the paged KV-cache. One
/// [`step_on`](Self::step_on) advances every slot by one token.
pub struct DecodeModel {
    layers: Vec<DecodeLayer>,
    cache: PagedKvCache,
    pool: DevicePool,
    bt_d: CudaSlice<u32>,
    cl_d: CudaSlice<u32>,
    wpos_d: CudaSlice<u32>,
    /// Per-slot active mask (u32 0/1) the append kernel reads — a free/padding slot is skipped so it
    /// can't scatter into a live block. All-1 for the all-slots-active [`step_on`]; the P5 scheduler
    /// uploads a ragged mask via [`advance_and_upload_masked`](Self::advance_and_upload_masked).
    active_d: CudaSlice<u32>,
    /// `[Bcap,D]` ping-pong activations between layers (persistent — outside the pool).
    bufs: [CudaSlice<f32>; 2],
    cfg: KvConfig,
    /// [`BlockManager::layout_epoch`](crate::paged_kv::BlockManager::layout_epoch) at the last
    /// `bt_d` upload (`None` = never uploaded). A steady-state decode step whose appends stay
    /// inside their current blocks leaves the epoch unchanged, so
    /// [`advance_and_upload_masked`](Self::advance_and_upload_masked) skips the (large) flat
    /// block-table upload and pushes only the per-slot ctx/wpos/mask vectors.
    uploaded_epoch: Option<u64>,
}

impl DecodeModel {
    /// `D = heads*head_dim`.
    #[inline]
    pub fn d(&self) -> usize {
        self.cfg.heads * self.cfg.head_dim
    }
    /// Decode batch (== cache slots).
    #[inline]
    pub fn bcap(&self) -> usize {
        self.cfg.num_slots
    }
    /// Layer count.
    #[inline]
    pub fn depth(&self) -> usize {
        self.layers.len()
    }
    /// The paged KV-cache (host populate / manager access for tests + the scheduler).
    pub fn cache_mut(&mut self) -> &mut PagedKvCache {
        &mut self.cache
    }

    /// Read access to the paged cache (footprint + block accounting for the scheduler / gates).
    pub fn cache(&self) -> &PagedKvCache {
        &self.cache
    }

    /// Build the `N` decode layers (one weight set each), the paged cache (`cfg`), a `pool_bytes`
    /// scratch arena, and the per-step metadata + ping-pong buffers. All share `g`'s stream. F16
    /// cache storage (the default, bit-exact path); int8 via [`new_with_dtype`](Self::new_with_dtype).
    pub fn new(
        g: &mut Gpu,
        weights: &[TransformerWeights],
        cfg: KvConfig,
        dff: usize,
        pool_bytes: usize,
    ) -> Result<Self, DriverError> {
        Self::new_with_dtype(g, weights, cfg, dff, pool_bytes, KvDtype::F16)
    }

    /// As [`new`](Self::new) with an explicit cache storage dtype (`Int8` = half the KV footprint of
    /// f16, lossy, tolerance-gated; the append/attention kernels are swapped per dtype, everything
    /// else is unchanged — including graph capture, which sees the same pooled-launch shape).
    pub fn new_with_dtype(
        g: &mut Gpu,
        weights: &[TransformerWeights],
        cfg: KvConfig,
        dff: usize,
        pool_bytes: usize,
        dtype: KvDtype,
    ) -> Result<Self, DriverError> {
        assert!(!weights.is_empty(), "model needs at least one layer");
        assert_eq!(weights.len(), cfg.layers, "cfg.layers must equal the number of weight sets");
        let d = cfg.heads * cfg.head_dim;
        let mut layers = Vec::with_capacity(weights.len());
        for w in weights {
            layers.push(DecodeLayer::new_with_dtype(g, w, cfg, dff, dtype)?);
        }
        let cache = PagedKvCache::new_with_dtype(g.stream.clone(), cfg, dtype)?;
        let pool = DevicePool::new(g.stream.clone(), pool_bytes)?;
        let bt_d = g.stream.alloc_zeros::<u32>(cfg.num_slots * cfg.max_blocks_per_seq)?;
        let cl_d = g.stream.alloc_zeros::<u32>(cfg.num_slots)?;
        let wpos_d = g.stream.alloc_zeros::<u32>(cfg.num_slots)?;
        let active_d = g.stream.memcpy_stod(&vec![1u32; cfg.num_slots])?;
        let bufs = [g.stream.alloc_zeros::<f32>(cfg.num_slots * d)?, g.stream.alloc_zeros::<f32>(cfg.num_slots * d)?];
        Ok(Self { layers, cache, pool, bt_d, cl_d, wpos_d, active_d, bufs, cfg, uploaded_epoch: None })
    }

    /// Advance **every** slot by one token: append a cache position per slot (host), upload the block
    /// table + post-append context lengths + write positions (once, shared across layers), then run all
    /// `N` layers GPU-resident on `stream`, ping-ponging the `[Bcap,D]` activation and writing the final
    /// hidden state into `out`. Errors `OUT_OF_MEMORY` if the cache is full (the scheduler's signal to
    /// preempt). All slots are assumed active (the P5 scheduler adds ragged admission).
    pub fn step_on(
        &mut self,
        stream: &Arc<CudaStream>,
        x_d: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), DriverError> {
        let active = vec![true; self.cfg.num_slots];
        self.advance_and_upload_masked(stream, &active)?;
        self.run_layers_on(stream, x_d, out)
    }

    /// Run the `N` layers over the **already-uploaded** metadata (the seam a CUDA graph captures: pure
    /// launches, no host append / upload). `x_d` → layer 0 → … → `out`.
    pub fn run_layers_on(
        &mut self,
        stream: &Arc<CudaStream>,
        x_d: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), DriverError> {
        // Disjoint field borrows (the launchers need &mut cache storage + &mut pool + & metadata at once).
        let Self { layers, cache, pool, bt_d, cl_d, wpos_d, active_d, bufs, .. } = self;
        let n = layers.len();
        let kv = cache.storage_mut();
        if n == 1 {
            pool.reset();
            return layers[0].forward_step_on(stream, pool, kv, x_d, bt_d, cl_d, wpos_d, active_d, 0, out);
        }
        pool.reset();
        layers[0].forward_step_on(stream, pool, kv, x_d, bt_d, cl_d, wpos_d, active_d, 0, &mut bufs[0])?;
        let mut cur = 0usize; // layer i-1's output lives in bufs[cur]
        for (i, layer) in layers.iter().enumerate().take(n - 1).skip(1) {
            pool.reset();
            let (a, b) = bufs.split_at_mut(1);
            let (src, dst) = if cur == 0 { (&a[0], &mut b[0]) } else { (&b[0], &mut a[0]) };
            layer.forward_step_on(stream, pool, kv, src, bt_d, cl_d, wpos_d, active_d, i, dst)?;
            cur = 1 - cur;
        }
        pool.reset();
        // Last layer reads the current buffer, writes the caller's `out`.
        let src = &bufs[cur];
        layers[n - 1].forward_step_on(stream, pool, kv, src, bt_d, cl_d, wpos_d, active_d, n - 1, out)
    }

    /// Upload the current host block table / context lengths / write positions to the device metadata
    /// buffers **without** running the layers — the host half of [`step_on`](Self::step_on), exposed so
    /// a graph can capture only the launch half ([`run_layers_on`](Self::run_layers_on)). Returns the
    /// per-slot write positions it appended.
    pub fn advance_and_upload(&mut self, stream: &Arc<CudaStream>) -> Result<Vec<u32>, DriverError> {
        let active = vec![true; self.cfg.num_slots];
        self.advance_and_upload_masked(stream, &active)
    }

    /// Like [`advance_and_upload`](Self::advance_and_upload) but only the `active[slot]` slots advance
    /// (append a token + grow context); inactive slots are frozen and **masked off** in the device
    /// `active_d` buffer so the append kernel skips them (it cannot write a free slot's padding-0 block
    /// table without corrupting block 0). This is the host half of one continuous-batching step — the
    /// [`Scheduler`] calls it, then [`run_layers_on`](Self::run_layers_on) (the graph-capturable launch
    /// half). Returns the per-slot pre-append write positions.
    ///
    /// **Atomic**: on `Err` (the cache cannot grow every active slot by one token) *nothing* has been
    /// mutated — no context length advanced, no block popped, no upload issued — so the caller may
    /// preempt and retry with a different mask without the host allocator and the device metadata
    /// having drifted apart.
    pub fn advance_and_upload_masked(
        &mut self,
        stream: &Arc<CudaStream>,
        active: &[bool],
    ) -> Result<Vec<u32>, DriverError> {
        let bcap = self.cfg.num_slots;
        assert_eq!(active.len(), bcap, "active mask must be one bool per slot");
        // **Feasibility pre-pass — the advance must be all-or-nothing.** The loop below mutates the
        // host allocator slot by slot, but the four device metadata uploads happen only after it
        // completes; a mid-loop failure would leave the earlier slots' context lengths advanced with
        // *nothing* uploaded, and the caller's documented recovery (preempt, retry) would then take
        // those inflated lengths as the next write positions — writing each survivor's next token one
        // position past a hole the attention kernel still covers (stale bytes from whichever sequence
        // last owned that block). `can_grow` tests exactly the two conditions `push_block` can fail on
        // (free-list exhaustion and the per-sequence table-width cap), so once it passes for every
        // active slot no `append` below can fail.
        {
            let mgr = self.cache.manager_ref();
            for b in 0..bcap {
                if active[b] && !mgr.can_grow(b, 1) {
                    return Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY));
                }
            }
        }
        let mut wpos = vec![0u32; bcap];
        let mut mask = vec![0u32; bcap];
        for b in 0..bcap {
            wpos[b] = self.cache.manager().context_len(b) as u32;
            if active[b] {
                mask[b] = 1;
                self.cache
                    .manager()
                    .append(b)
                    .map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
            }
        }
        // The flat block table only changes when a table gains or loses a block (admission, eviction,
        // block-boundary growth) — the layout epoch tracks exactly that, so the steady-state decode
        // step (every append inside its current block) uploads just the three per-slot vectors.
        let epoch = self.cache.manager_ref().layout_epoch();
        if self.uploaded_epoch != Some(epoch) {
            let table = self.cache.manager_ref().flat_block_table();
            stream.memcpy_htod(&table, &mut self.bt_d)?;
            self.uploaded_epoch = Some(epoch);
        }
        let lens = self.cache.manager_ref().ctx_lens();
        stream.memcpy_htod(&lens, &mut self.cl_d)?;
        stream.memcpy_htod(&wpos, &mut self.wpos_d)?;
        stream.memcpy_htod(&mask, &mut self.active_d)?;
        Ok(wpos)
    }

    /// Upload **caller-crafted** metadata contents into the four device buffers the kernels read —
    /// the bench/gate seam for staging a specific fill/mask scenario without mutating the host
    /// allocator. Contents-only: a captured graph bakes these buffers' *pointers*, so a re-upload
    /// re-steers every subsequent replay (a fill level is data, not a shape). Lengths must match the
    /// `Bcap` geometry.
    pub fn upload_metadata(
        &mut self,
        stream: &Arc<CudaStream>,
        table: &[u32],
        cl: &[u32],
        wpos: &[u32],
        active: &[u32],
    ) -> Result<(), DriverError> {
        let bcap = self.cfg.num_slots;
        assert_eq!(table.len(), bcap * self.cfg.max_blocks_per_seq, "flat block table must be [Bcap, max_blocks_per_seq]");
        assert_eq!(cl.len(), bcap, "context lengths must be one per slot");
        assert_eq!(wpos.len(), bcap, "write positions must be one per slot");
        assert_eq!(active.len(), bcap, "active mask must be one per slot");
        stream.memcpy_htod(table, &mut self.bt_d)?;
        stream.memcpy_htod(cl, &mut self.cl_d)?;
        stream.memcpy_htod(wpos, &mut self.wpos_d)?;
        stream.memcpy_htod(active, &mut self.active_d)?;
        // The device table no longer mirrors the manager — force the next masked advance to re-push.
        self.uploaded_epoch = None;
        Ok(())
    }

    /// The two ping-pong activation buffers (a graph bakes their pointers — keep alive).
    pub fn buffers(&self) -> &[CudaSlice<f32>; 2] {
        &self.bufs
    }
}

/// A serving request: a `prompt_len`-token prefill followed by `gen_len` decode tokens.
///
/// **Both lengths are floored at 1 on admission.** A slot must own at least one cache block (a
/// zero-length prefill would leave the block table padded with 0 — the live block the append kernel's
/// `active` mask exists to protect), and a sequence must emit at least one token (`gen_len == 0` would
/// underflow the in-flight `remaining` counter on its very first retire).
#[derive(Clone, Copy, Debug)]
pub struct Request {
    pub prompt_len: usize,
    pub gen_len: usize,
}

/// Per-slot in-flight sequence state.
#[derive(Clone, Copy)]
struct Inflight {
    /// Decode tokens still to emit before this sequence finishes and frees its slot.
    remaining: usize,
}

/// **Continuous-batching (in-flight) scheduler** over a fixed-`Bcap` [`DecodeModel`] — Orca-style
/// iteration-level scheduling with selective batching. A waiting [`Request`] is admitted into any free
/// slot (prefilling its prompt into the paged cache); every [`step`](Self::step) advances all *active*
/// slots by one decode token; a sequence that reaches its `gen_len` is **evicted the same iteration**,
/// its blocks freed back to the pool and a waiting request admitted into the freed slot.
///
/// The throughput lever: the fixed-shape decode kernel computes all `Bcap` rows *regardless* of how many
/// carry a live request, so the per-step latency is **independent of the active count**. A server that
/// runs one sequence at a time wastes `Bcap-1` rows of compute every step; continuous batching fills
/// them, so **goodput (useful tokens/s) scales with batch fill** at ~constant latency. Inactive slots
/// are masked off in the append kernel (they hold no blocks — a write would corrupt block 0) and read as
/// empty by attention (`context_len == 0` ⇒ zero row), so they are numerically inert.
pub struct Scheduler {
    model: DecodeModel,
    /// Per-slot occupancy (`None` = free); length `Bcap`.
    slots: Vec<Option<Inflight>>,
    /// FIFO of requests not yet admitted (no free slot, or insufficient blocks).
    waiting: std::collections::VecDeque<Request>,
    /// Cumulative useful decode tokens emitted (one per active slot per step).
    emitted: usize,
    /// Cumulative requests admitted / completed (for accounting gates).
    admitted: usize,
    completed: usize,
    /// The captured whole-step graph [`step_graphed`](Self::step_graphed) replays, plus the raw
    /// device pointers of the `x_d`/`out` it was captured against (the graph bakes them — later
    /// calls must pass the same buffers). Captured once, never re-captured: the active mask, block
    /// table, context lengths, and write positions are all *contents* of device buffers the kernels
    /// re-read every replay, and every grid shape is a function of the fixed `Bcap` alone — so no
    /// admission/eviction/growth ever invalidates the recording.
    graph: Option<(crate::graph::Graph, u64, u64)>,
    /// **Static batching** (the classic peer continuous batching is measured against): admit a full
    /// batch, then admit nothing more until the whole batch has drained — no mid-flight refill of
    /// freed slots. Same kernels, same step; only the admission policy differs, which is exactly
    /// what makes the continuous-vs-static goodput ratio the honest scheduling win.
    static_batching: bool,
}

impl Scheduler {
    /// Wrap a built [`DecodeModel`]; all `Bcap` slots start free. Continuous batching (the default
    /// policy: freed slots are refilled every iteration).
    pub fn new(model: DecodeModel) -> Self {
        let bcap = model.bcap();
        Self {
            model,
            slots: vec![None; bcap],
            waiting: std::collections::VecDeque::new(),
            emitted: 0,
            admitted: 0,
            completed: 0,
            graph: None,
            static_batching: false,
        }
    }

    /// As [`new`](Self::new) but with **static batching** (see the field docs): the honest peer the
    /// goodput bench measures continuous batching against — identical kernels and step machinery,
    /// admission only when the previous batch has fully drained.
    pub fn new_static(model: DecodeModel) -> Self {
        Self { static_batching: true, ..Self::new(model) }
    }

    /// Queue a request for admission.
    pub fn enqueue(&mut self, req: Request) {
        self.waiting.push_back(req);
    }

    /// The underlying model (cache footprint, geometry).
    pub fn model(&self) -> &DecodeModel {
        &self.model
    }

    /// Active (occupied) slot count this instant.
    pub fn num_active(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Requests still waiting for a slot.
    pub fn pending(&self) -> usize {
        self.waiting.len()
    }

    /// Cumulative useful tokens emitted; admitted / completed request counts.
    pub fn emitted(&self) -> usize {
        self.emitted
    }
    pub fn admitted(&self) -> usize {
        self.admitted
    }
    pub fn completed(&self) -> usize {
        self.completed
    }

    /// Physical KV blocks currently free in the pool (block-conservation gates).
    pub fn free_blocks(&self) -> usize {
        self.model.cache().manager_ref().free_blocks()
    }

    /// No active sequences and nothing waiting — the drain loop's stop condition.
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.num_active() == 0
    }

    /// Bounded look-ahead of the first-fit admission scan: how deep into the waiting queue
    /// [`admit`](Self::admit) searches for a request that fits when the queue front doesn't. Keeps
    /// the per-step admission cost O(free_slots · LOOKAHEAD) and the reordering window finite.
    pub const ADMIT_LOOKAHEAD: usize = 64;

    /// Admit waiting requests into free slots, prefilling each prompt into the paged cache.
    /// **Deterministic first-fit with bounded look-ahead**: for each free slot, scan the first
    /// [`ADMIT_LOOKAHEAD`](Self::ADMIT_LOOKAHEAD) waiting requests front-to-back and admit the first
    /// that fits the block pool — so one large request at the queue head no longer head-of-line-blocks
    /// every smaller request behind it into idle slots. FIFO is preserved among requests that fit
    /// (the lowest-index fitting request always wins). **Fairness trade-off (standard for
    /// first-fit):** under sustained block pressure a large head request can be overtaken repeatedly
    /// by smaller arrivals until pressure eases — the price of maximizing fill; the bounded window
    /// keeps the scan cheap and the reordering finite. Returns the number admitted this call.
    pub fn admit(&mut self) -> Result<usize, DriverError> {
        // Static batching: no mid-flight refill — wait for the whole batch to drain.
        if self.static_batching && self.num_active() > 0 {
            return Ok(0);
        }
        let bcap = self.model.bcap();
        let mut n = 0;
        for slot in 0..bcap {
            if self.slots[slot].is_some() {
                continue;
            }
            if self.waiting.is_empty() {
                break;
            }
            let mgr = self.model.cache_mut().manager();
            // First fitting request within the look-ahead window. Every free slot has an empty
            // table, so fit depends only on the request — a whole-window miss is a miss for every
            // remaining free slot, and the scan stops.
            let Some(pick) = self
                .waiting
                .iter()
                .take(Self::ADMIT_LOOKAHEAD)
                .position(|req| mgr.can_grow(slot, req.prompt_len.max(1)))
            else {
                break; // nothing in the window fits → leave the queue intact
            };
            let req = self.waiting.remove(pick).expect("position() index is in range");
            // Prefill: bulk-reserve the prompt's cache positions (≥1 so the slot owns a block; the
            // attention/append kernels then see a non-empty, non-padding row).
            self.model
                .cache_mut()
                .manager()
                .reserve(slot, req.prompt_len.max(1))
                .map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
            // `gen_len.max(1)`: the same normalization `prompt_len` gets just above. A `remaining` of 0
            // would go negative on this sequence's first `retire_finished` — panicking in a debug build
            // and wrapping to `usize::MAX` in a release one, pinning the slot and its blocks forever.
            self.slots[slot] = Some(Inflight { remaining: req.gen_len.max(1) });
            self.admitted += 1;
            n += 1;
        }
        Ok(n)
    }

    /// One continuous-batching iteration on `stream`: admit waiting requests, advance every active slot
    /// by one decode token (masked append + one metadata upload, then the `N`-layer launch over the
    /// already-uploaded metadata), then retire any sequence that has emitted its `gen_len` tokens
    /// (freeing its blocks). `x_d`/`out` are `[Bcap,D]`. Returns the useful tokens emitted this step
    /// (= the active slot count). When idle, runs nothing and returns 0.
    pub fn step(
        &mut self,
        stream: &Arc<CudaStream>,
        x_d: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<usize, DriverError> {
        self.admit()?;
        let active: Vec<bool> = self.slots.iter().map(|s| s.is_some()).collect();
        let n_active = active.iter().filter(|&&a| a).count();
        if n_active == 0 {
            return Ok(0);
        }
        self.model.advance_and_upload_masked(stream, &active)?;
        self.model.run_layers_on(stream, x_d, out)?;
        self.retire_finished();
        Ok(n_active)
    }

    /// [`step`](Self::step) with the `N`-layer launch half replayed as **one cached
    /// `cuGraphLaunch`** — the production decode loop: per iteration the host does admission, the
    /// masked append bookkeeping, three small per-slot uploads (the block table only when the layout
    /// epoch moved), and a single graph replay. The first active call runs one eager warmup (which
    /// both executes that step and stabilizes the pool sub-allocation addresses the recording bakes
    /// in) and captures; every later call replays. No recapture is ever needed — see the `graph`
    /// field docs. Requirements the caller owns (as for [`crate::graph::Graph::capture`]): `stream`
    /// is a dedicated non-NULL stream, event tracking is disabled, `x_d`/`out` are the same buffers
    /// every call (asserted), and any prior NULL-stream work (weight upload, cache zeroing at model
    /// construction) has been synchronized before the first call — a non-blocking stream does *not*
    /// implicitly wait on the NULL stream, so an unsynchronized first step would race the uploads.
    pub fn step_graphed(
        &mut self,
        stream: &Arc<CudaStream>,
        x_d: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<usize, DriverError> {
        use cudarc::driver::DevicePtr;
        self.admit()?;
        let active: Vec<bool> = self.slots.iter().map(|s| s.is_some()).collect();
        let n_active = active.iter().filter(|&&a| a).count();
        if n_active == 0 {
            return Ok(0);
        }
        self.model.advance_and_upload_masked(stream, &active)?;
        let (xp, op) = {
            let (xp, _gx) = x_d.device_ptr(stream);
            let (op, _go) = out.device_ptr(stream);
            (xp as u64, op as u64)
        };
        if let Some((graph, gx, go)) = &self.graph {
            assert_eq!((*gx, *go), (xp, op), "step_graphed: x_d/out must be the buffers the graph was captured with");
            graph.launch()?;
        } else {
            // Warmup executes this step eagerly (idempotent kernels: re-running would rewrite the
            // same K/V bytes and recompute the same output), then the capture records the identical
            // launch sequence without executing it — so the step still runs exactly once.
            self.model.run_layers_on(stream, x_d, out)?;
            stream.synchronize()?;
            let graph = crate::graph::Graph::capture(stream.clone(), || {
                self.model.run_layers_on(stream, x_d, out)
            })?;
            self.graph = Some((graph, xp, op));
        }
        self.retire_finished();
        Ok(n_active)
    }

    /// Retire every sequence that has emitted its `gen_len` tokens this step: decrement each active
    /// slot's remaining count, bump the goodput counter, and free finished slots' blocks.
    fn retire_finished(&mut self) {
        for slot in 0..self.model.bcap() {
            let remaining = match &mut self.slots[slot] {
                Some(inf) => {
                    inf.remaining -= 1;
                    inf.remaining
                }
                None => continue,
            };
            self.emitted += 1;
            if remaining == 0 {
                self.model.cache_mut().manager().free(slot);
                self.slots[slot] = None;
                self.completed += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::{
        kv_append_int8_ptx, kv_append_ptx, launch_kv_append, launch_kv_append_int8,
        quantize_kv_int8, reference_decode_attn, KV_APPEND_ENTRY, KV_APPEND_INT8_ENTRY,
    };
    use crate::paged_kv::BlockManager;

    fn with_gpu(name: &str, body: impl FnOnce(&mut Gpu)) {
        let mut guard = crate::gpu();
        match guard.as_mut() {
            Some(g) => body(g),
            None => eprintln!("[skip] {name}: no CUDA device reachable"),
        }
    }

    fn f16r(x: f32) -> f32 {
        f16::from_f32(x).to_f32()
    }

    // ---- f64 references for one decode step (the tolerance oracle) ----
    fn ref_rmsnorm_row(x: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();
        let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        x.iter().map(|&v| (v as f64 * inv) as f32).collect()
    }
    /// `C[m,n] = Σ_k f16(a[m,k])·f16(b[n,k])` (NT, weight `b` is `[N,K]`), f64 accumulate — matches the
    /// WMMA f16-in/f32-out GEMM up to accumulation precision.
    fn ref_nt_f16(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += f16r(a[i * k + kk]) as f64 * f16r(b[j * k + kk]) as f64;
                }
                c[i * n + j] = acc as f32;
            }
        }
        c
    }
    fn ref_silu(x: f32) -> f32 {
        let x = x as f64;
        (x / (1.0 + (-x).exp())) as f32
    }

    /// f64 reference for **one decode step** of a single layer: RMSNorm → projections → attention over
    /// (`past` ++ new token) → O-proj+residual → RMSNorm → SiLU FFN → residual. `past_k`/`past_v[b]` are
    /// slot `b`'s cached context `[ctx0_b, D]` (already f16-rounded). Mirrors `DecodeLayer::forward_step`.
    #[allow(clippy::too_many_arguments)]
    fn ref_decode_step(
        x: &[f32],
        w: &TransformerWeights,
        past_k: &[Vec<f32>],
        past_v: &[Vec<f32>],
        ctx0: &[usize],
        heads: usize,
        hd: usize,
        dff: usize,
        eps: f32,
    ) -> Vec<f32> {
        // f16 cache: the new token's K/V round through f16 on the way into the cache.
        let f16_row = |row: &[f32]| -> Vec<f32> { row.iter().map(|&z| f16r(z)).collect() };
        ref_decode_step_with(x, w, past_k, past_v, ctx0, heads, hd, dff, eps, &f16_row)
    }

    /// [`ref_decode_step`] with the **cache rounding of the new token** as a parameter: the f16 path
    /// rounds through f16, the int8 path quantizes/dequantizes per (token, head) — `past_k`/`past_v`
    /// carry whatever effective (already-rounded/dequantized) values the cache stores for the past.
    #[allow(clippy::too_many_arguments)]
    fn ref_decode_step_with(
        x: &[f32],
        w: &TransformerWeights,
        past_k: &[Vec<f32>],
        past_v: &[Vec<f32>],
        ctx0: &[usize],
        heads: usize,
        hd: usize,
        dff: usize,
        eps: f32,
        cache_round: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> Vec<f32> {
        let bcap = ctx0.len();
        let d = heads * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0f32; bcap * d];
        for b in 0..bcap {
            let xb = &x[b * d..(b + 1) * d];
            let h1 = ref_rmsnorm_row(xb, eps);
            let q = ref_nt_f16(&h1, w.wq, 1, d, d);
            let kk = ref_nt_f16(&h1, w.wk, 1, d, d);
            let vv = ref_nt_f16(&h1, w.wv, 1, d, d);
            // Full attention context: the cached past ++ the new token (rounded as the cache stores).
            let ctx = ctx0[b];
            let mut kfull = past_k[b].clone();
            let mut vfull = past_v[b].clone();
            kfull.extend(cache_round(&kk));
            vfull.extend(cache_round(&vv));
            let attn = reference_decode_attn(&q, &[kfull], &[vfull], &[ctx + 1], heads, hd, scale);
            let o = ref_nt_f16(&attn, w.wo, 1, d, d);
            let x1: Vec<f32> = xb.iter().zip(&o).map(|(&a, &b)| a + b).collect();
            let h2 = ref_rmsnorm_row(&x1, eps);
            let f1 = ref_nt_f16(&h2, w.w1, 1, d, dff);
            let f1act: Vec<f32> = f1.iter().map(|&z| ref_silu(z)).collect();
            let f2 = ref_nt_f16(&f1act, w.w2, 1, dff, d);
            for i in 0..d {
                out[b * d + i] = x1[i] + f2[i];
            }
        }
        out
    }

    /// **KV-append scatter correctness.** Appending the new token's K/V via the kernel, then reading the
    /// cache back, must reproduce the f16-rounded inputs at exactly the block-table address the host
    /// allocator hands out — bit-for-bit (the scatter is a pure narrow + place, no arithmetic).
    #[test]
    fn serving_kv_append_round_trip() {
        with_gpu("serving_kv_append_round_trip", |g| {
            let (heads, hd, bsz, bcap) = (4usize, 64usize, 16usize, 8usize);
            let d = heads * hd;
            let wpos = [5usize, 0, 16, 31, 3, 20, 12, 47]; // write positions across block boundaries
            let max_bps = wpos.iter().copied().max().unwrap().div_ceil(bsz) + 2;
            let num_blocks = bcap * max_bps + 4;
            let cfg = KvConfig { layers: 1, heads, head_dim: hd, block_size: bsz, num_blocks, num_slots: bcap, max_blocks_per_seq: max_bps };
            let mut mgr = BlockManager::new(num_blocks, bsz, bcap, max_bps);
            for b in 0..bcap {
                mgr.reserve(b, wpos[b] + 1).unwrap(); // ensure position wpos[b] has a block
            }
            let mut rng = crate::diff::Rng::new(0xA99E);
            let knew = rng.vec(bcap * d, -1.0, 1.0);
            let vnew = rng.vec(bcap * d, -2.0, 2.0);
            let knew_d = g.stream.memcpy_stod(&knew).unwrap();
            let vnew_d = g.stream.memcpy_stod(&vnew).unwrap();
            let mut k_d = g.stream.alloc_zeros::<f16>(cfg.slab_elems()).unwrap();
            let mut v_d = g.stream.alloc_zeros::<f16>(cfg.slab_elems()).unwrap();
            let bt_d = g.stream.memcpy_stod(&mgr.flat_block_table()).unwrap();
            let wpos_u: Vec<u32> = wpos.iter().map(|&p| p as u32).collect();
            let wpos_d = g.stream.memcpy_stod(&wpos_u).unwrap();
            let active_d = g.stream.memcpy_stod(&vec![1u32; bcap]).unwrap(); // all slots active
            let func = g.function("kv_append", &kv_append_ptx(), KV_APPEND_ENTRY).unwrap();
            launch_kv_append(&g.stream, &func, &knew_d, &vnew_d, &mut k_d, &mut v_d, &bt_d, &wpos_d, &active_d, &cfg, 0, bcap).unwrap();
            g.stream.synchronize().unwrap();
            let kh = g.stream.memcpy_dtov(&k_d).unwrap();
            let vh = g.stream.memcpy_dtov(&v_d).unwrap();
            for b in 0..bcap {
                let (phys, off) = mgr.locate(b, wpos[b]);
                for h in 0..heads {
                    for dh in 0..hd {
                        let idx = cfg.elem_offset(0, phys, off, h, dh);
                        let src = (h * hd + dh) + b * d;
                        assert_eq!(kh[idx].to_f32(), f16r(knew[src]), "K append slot {b} h{h} dh{dh}");
                        assert_eq!(vh[idx].to_f32(), f16r(vnew[src]), "V append slot {b} h{h} dh{dh}");
                    }
                }
            }
            eprintln!("kv_append: {bcap} slots scattered to their block-table addresses, bit-exact vs f16(input)");
        });
    }

    /// Build `n` independent layers' weights (owned) + a `TransformerWeights` view per layer.
    #[allow(clippy::type_complexity)]
    fn layer_weights(rng: &mut crate::diff::Rng, n: usize, d: usize, dff: usize) -> Vec<[Vec<f32>; 6]> {
        (0..n)
            .map(|_| {
                [
                    rng.vec(d * d, -0.1, 0.1),
                    rng.vec(d * d, -0.1, 0.1),
                    rng.vec(d * d, -0.1, 0.1),
                    rng.vec(d * d, -0.1, 0.1),
                    rng.vec(dff * d, -0.1, 0.1),
                    rng.vec(d * dff, -0.1, 0.1),
                ]
            })
            .collect()
    }
    fn weights_view(wdata: &[[Vec<f32>; 6]]) -> Vec<TransformerWeights<'_>> {
        wdata
            .iter()
            .map(|wl| TransformerWeights { wq: &wl[0], wk: &wl[1], wv: &wl[2], wo: &wl[3], w1: &wl[4], w2: &wl[5] })
            .collect()
    }

    /// Populate a 1-layer model's cache plane with each slot's f16-rounded past context, returning the
    /// `(past_k, past_v)` the reference reuses. Reserves `ctx0[b]` tokens per slot first.
    fn populate_one_layer(
        g: &mut Gpu,
        model: &mut DecodeModel,
        ctx0: &[usize],
        heads: usize,
        hd: usize,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let d = heads * hd;
        let cfg = *model.cache_mut().config();
        let mut rng = crate::diff::Rng::new(seed);
        let mut past_k = Vec::new();
        let mut past_v = Vec::new();
        let mut kh = vec![f16::from_f32(0.0); cfg.slab_elems()];
        let mut vh = vec![f16::from_f32(0.0); cfg.slab_elems()];
        for b in 0..ctx0.len() {
            let pk: Vec<f32> = rng.vec(ctx0[b] * d, -1.0, 1.0).iter().map(|&x| f16r(x)).collect();
            let pv: Vec<f32> = rng.vec(ctx0[b] * d, -1.0, 1.0).iter().map(|&x| f16r(x)).collect();
            if ctx0[b] > 0 {
                model.cache_mut().manager().reserve(b, ctx0[b]).unwrap();
            }
            for t in 0..ctx0[b] {
                let (phys, off) = model.cache_mut().manager_ref().locate(b, t);
                for h in 0..heads {
                    for dh in 0..hd {
                        let idx = cfg.elem_offset(0, phys, off, h, dh);
                        kh[idx] = f16::from_f32(pk[(t * heads + h) * hd + dh]);
                        vh[idx] = f16::from_f32(pv[(t * heads + h) * hd + dh]);
                    }
                }
            }
            past_k.push(pk);
            past_v.push(pv);
        }
        let (ks, vs) = model.cache_mut().slabs_mut();
        g.stream.memcpy_htod(&kh, ks).unwrap();
        g.stream.memcpy_htod(&vh, vs).unwrap();
        (past_k, past_v)
    }

    /// **Whole decode-step numerical gate (the first law).** One full decode step (RMSNorm → proj →
    /// append → paged attention → O-proj+res → RMSNorm → SiLU FFN → res) over a batch of ragged-context
    /// sequences must match an f64 reference within tolerance — the integration of every reused kernel +
    /// the new append/paged-attn. Pre-existing context is host-populated; the step appends one token.
    #[test]
    fn serving_decode_step_matches_reference() {
        with_gpu("serving_decode_step_matches_reference", |g| {
            let (heads, hd, dff, bsz, bcap) = (4usize, 64usize, 256usize, 16usize, 64usize);
            let d = heads * hd;
            let ctx0: Vec<usize> = (0..bcap).map(|b| (b * 13) % 81).collect(); // ragged incl. 0
            let max_bps = ctx0.iter().copied().max().unwrap().div_ceil(bsz) + 2;
            let num_blocks = bcap * max_bps + 8;
            let cfg = KvConfig { layers: 1, heads, head_dim: hd, block_size: bsz, num_blocks, num_slots: bcap, max_blocks_per_seq: max_bps };
            let mut rng = crate::diff::Rng::new(0x5723);
            let wdata = layer_weights(&mut rng, 1, d, dff);
            let weights = weights_view(&wdata);
            let mut model = DecodeModel::new(g, &weights, cfg, dff, 64 * 1024 * 1024).unwrap();
            let (past_k, past_v) = populate_one_layer(g, &mut model, &ctx0, heads, hd, 0x1234);
            let x = rng.vec(bcap * d, -1.0, 1.0);
            let x_d = g.stream.memcpy_stod(&x).unwrap();
            let mut out_d = g.stream.alloc_zeros::<f32>(bcap * d).unwrap();
            model.step_on(&g.stream.clone(), &x_d, &mut out_d).unwrap();
            g.stream.synchronize().unwrap();
            let got = g.stream.memcpy_dtov(&out_d).unwrap();
            let refv = ref_decode_step(&x, &weights[0], &past_k, &past_v, &ctx0, heads, hd, dff, 1e-5);
            let s = crate::diff::assert_close("decode_step", &got, &refv, 5e-2, 5e-2);
            eprintln!(
                "decode step vs f64 ref: max_abs={:.2e} max_rel={:.2e} (Bcap={bcap}, heads={heads}, hd={hd}, Dff={dff}, ragged ctx 0..80)",
                s.max_abs, s.max_rel
            );
        });
    }

    /// Populate a 1-layer **int8** model's cache plane with each slot's per-(token, head) quantized
    /// past (values + scales, the [`quantize_kv_int8`] scheme), returning the **dequantized**
    /// `(past_k, past_v)` — the effective values the reference attends over. Reserves `ctx0[b]`
    /// tokens per slot first. The int8 twin of [`populate_one_layer`].
    fn populate_one_layer_int8(
        g: &mut Gpu,
        model: &mut DecodeModel,
        ctx0: &[usize],
        heads: usize,
        hd: usize,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let d = heads * hd;
        let cfg = *model.cache_mut().config();
        let mut rng = crate::diff::Rng::new(seed);
        let (mut past_k, mut past_v) = (Vec::new(), Vec::new());
        let mut kq = vec![0i8; cfg.slab_elems()];
        let mut vq = vec![0i8; cfg.slab_elems()];
        let mut ks = vec![0f32; cfg.scale_slab_elems()];
        let mut vs = vec![0f32; cfg.scale_slab_elems()];
        for b in 0..ctx0.len() {
            let pk = rng.vec(ctx0[b] * d, -1.0, 1.0);
            let pv = rng.vec(ctx0[b] * d, -1.0, 1.0);
            if ctx0[b] > 0 {
                model.cache_mut().manager().reserve(b, ctx0[b]).unwrap();
            }
            let (kqi, ksi) = quantize_kv_int8(&pk, ctx0[b], heads, hd);
            let (vqi, vsi) = quantize_kv_int8(&pv, ctx0[b], heads, hd);
            for t in 0..ctx0[b] {
                let (phys, off) = model.cache_mut().manager_ref().locate(b, t);
                for h in 0..heads {
                    ks[cfg.scale_offset(0, phys, off, h)] = ksi[t * heads + h];
                    vs[cfg.scale_offset(0, phys, off, h)] = vsi[t * heads + h];
                    for dh in 0..hd {
                        let idx = cfg.elem_offset(0, phys, off, h, dh);
                        let src = (t * heads + h) * hd + dh;
                        kq[idx] = kqi[src];
                        vq[idx] = vqi[src];
                    }
                }
            }
            // Dequantized effective past ([t, heads, hd] row-major ⇒ scale index = i / hd).
            past_k.push((0..ctx0[b] * d).map(|i| kqi[i] as f32 * ksi[i / hd]).collect());
            past_v.push((0..ctx0[b] * d).map(|i| vqi[i] as f32 * vsi[i / hd]).collect());
        }
        match model.cache_mut().storage_mut() {
            KvStorage::Int8 { k, v, ksc, vsc } => {
                g.stream.memcpy_htod(&kq, k).unwrap();
                g.stream.memcpy_htod(&vq, v).unwrap();
                g.stream.memcpy_htod(&ks, ksc).unwrap();
                g.stream.memcpy_htod(&vs, vsc).unwrap();
            }
            KvStorage::F16 { .. } => unreachable!("int8 populate on a non-int8 cache"),
        }
        (past_k, past_v)
    }

    /// **int8 KV-append round-trip + mask (exact).** The device append must quantize exactly like
    /// the ties-even host mirror — int8 values equal, f32 scales bit-equal (IEEE `div.rn` + order-
    /// independent amax make this deterministic) — at exactly the block-table addresses, with
    /// inactive slots leaving the slabs untouched (the block-0 guard) and zeros everywhere else.
    #[test]
    fn serving_kv_append_int8_round_trip_and_mask() {
        with_gpu("serving_kv_append_int8_round_trip_and_mask", |g| {
            let (heads, hd, bsz, bcap) = (4usize, 64usize, 16usize, 8usize);
            let d = heads * hd;
            let active = [true, false, true, true, false, true, true, false];
            let wpos = [5usize, 0, 16, 31, 0, 3, 20, 0]; // across block boundaries
            let max_bps = wpos.iter().copied().max().unwrap().div_ceil(bsz) + 2;
            let num_blocks = bcap * max_bps + 4;
            let cfg = KvConfig { layers: 1, heads, head_dim: hd, block_size: bsz, num_blocks, num_slots: bcap, max_blocks_per_seq: max_bps };
            let mut mgr = BlockManager::new(num_blocks, bsz, bcap, max_bps);
            for b in 0..bcap {
                if active[b] {
                    mgr.reserve(b, wpos[b] + 1).unwrap();
                }
            }
            let mut rng = crate::diff::Rng::new(0x18A9);
            let knew = rng.vec(bcap * d, -1.0, 1.0);
            let vnew = rng.vec(bcap * d, -2.0, 2.0);
            let knew_d = g.stream.memcpy_stod(&knew).unwrap();
            let vnew_d = g.stream.memcpy_stod(&vnew).unwrap();
            let mut k_d = g.stream.alloc_zeros::<i8>(cfg.slab_elems()).unwrap();
            let mut v_d = g.stream.alloc_zeros::<i8>(cfg.slab_elems()).unwrap();
            let mut ksc_d = g.stream.alloc_zeros::<f32>(cfg.scale_slab_elems()).unwrap();
            let mut vsc_d = g.stream.alloc_zeros::<f32>(cfg.scale_slab_elems()).unwrap();
            let bt_d = g.stream.memcpy_stod(&mgr.flat_block_table()).unwrap();
            let wpos_u: Vec<u32> = wpos.iter().map(|&p| p as u32).collect();
            let wpos_d = g.stream.memcpy_stod(&wpos_u).unwrap();
            let act_u: Vec<u32> = active.iter().map(|&a| a as u32).collect();
            let act_d = g.stream.memcpy_stod(&act_u).unwrap();
            let func = g.function("kv_append_int8", &kv_append_int8_ptx(), KV_APPEND_INT8_ENTRY).unwrap();
            launch_kv_append_int8(
                &g.stream, &func, &knew_d, &vnew_d, &mut k_d, &mut v_d, &mut ksc_d, &mut vsc_d, &bt_d, &wpos_d,
                &act_d, &cfg, 0, bcap,
            )
            .unwrap();
            g.stream.synchronize().unwrap();
            let kh = g.stream.memcpy_dtov(&k_d).unwrap();
            let vh = g.stream.memcpy_dtov(&v_d).unwrap();
            let ksch = g.stream.memcpy_dtov(&ksc_d).unwrap();
            let vsch = g.stream.memcpy_dtov(&vsc_d).unwrap();
            // Ties-even host mirror of the device quantization (see the kernel's rounding note).
            let quant = |row: &[f32]| -> (Vec<i8>, f32) {
                let amax = row.iter().fold(0f32, |m, &x| m.max(x.abs()));
                let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                (row.iter().map(|&x| (x / scale).round_ties_even().clamp(-127.0, 127.0) as i8).collect(), scale)
            };
            let mut exp_k = vec![0i8; cfg.slab_elems()];
            let mut exp_v = vec![0i8; cfg.slab_elems()];
            let mut exp_ks = vec![0f32; cfg.scale_slab_elems()];
            let mut exp_vs = vec![0f32; cfg.scale_slab_elems()];
            for b in 0..bcap {
                if !active[b] {
                    continue;
                }
                let (phys, off) = mgr.locate(b, wpos[b]);
                for h in 0..heads {
                    let (qk, sk) = quant(&knew[b * d + h * hd..b * d + (h + 1) * hd]);
                    let (qv, sv) = quant(&vnew[b * d + h * hd..b * d + (h + 1) * hd]);
                    exp_ks[cfg.scale_offset(0, phys, off, h)] = sk;
                    exp_vs[cfg.scale_offset(0, phys, off, h)] = sv;
                    for dh in 0..hd {
                        exp_k[cfg.elem_offset(0, phys, off, h, dh)] = qk[dh];
                        exp_v[cfg.elem_offset(0, phys, off, h, dh)] = qv[dh];
                    }
                }
            }
            for i in 0..cfg.slab_elems() {
                assert_eq!(kh[i], exp_k[i], "int8 K slab elem {i} (inactive leak or quant mismatch?)");
                assert_eq!(vh[i], exp_v[i], "int8 V slab elem {i}");
            }
            for i in 0..cfg.scale_slab_elems() {
                assert_eq!(ksch[i].to_bits(), exp_ks[i].to_bits(), "K scale {i} not bit-equal");
                assert_eq!(vsch[i].to_bits(), exp_vs[i].to_bits(), "V scale {i} not bit-equal");
            }
            let written = active.iter().filter(|&&a| a).count();
            eprintln!(
                "int8 kv_append: {written}/{bcap} active slots quantized+scattered EXACTLY (values == host mirror, \
                 scales bit-equal); every inactive row skipped"
            );
        });
    }

    /// **int8-KV whole decode-step tolerance gate (the first law, lossy path).** One full decode step
    /// on an int8-cache model (device-quantized append + int8 paged attention) must match the f64
    /// reference attending over the dequantized cache within tolerance — int8 is lossy, so this is
    /// the tolerance sibling of the bit-oriented f16 gates; the f16 default path keeps every
    /// bit-exact invariance gate. Also checks the footprint halves vs f16.
    #[test]
    fn serving_decode_step_int8_matches_reference() {
        with_gpu("serving_decode_step_int8_matches_reference", |g| {
            let (heads, hd, dff, bsz, bcap) = (4usize, 64usize, 256usize, 16usize, 64usize);
            let d = heads * hd;
            let ctx0: Vec<usize> = (0..bcap).map(|b| (b * 13) % 81).collect(); // ragged incl. 0
            let max_bps = ctx0.iter().copied().max().unwrap().div_ceil(bsz) + 2;
            let num_blocks = bcap * max_bps + 8;
            let cfg = KvConfig { layers: 1, heads, head_dim: hd, block_size: bsz, num_blocks, num_slots: bcap, max_blocks_per_seq: max_bps };
            let mut rng = crate::diff::Rng::new(0x1D8A);
            let wdata = layer_weights(&mut rng, 1, d, dff);
            let weights = weights_view(&wdata);
            let mut model =
                DecodeModel::new_with_dtype(g, &weights, cfg, dff, 64 * 1024 * 1024, KvDtype::Int8).unwrap();
            let (past_k, past_v) = populate_one_layer_int8(g, &mut model, &ctx0, heads, hd, 0x9876);
            let x = rng.vec(bcap * d, -1.0, 1.0);
            let x_d = g.stream.memcpy_stod(&x).unwrap();
            let mut out_d = g.stream.alloc_zeros::<f32>(bcap * d).unwrap();
            model.step_on(&g.stream.clone(), &x_d, &mut out_d).unwrap();
            g.stream.synchronize().unwrap();
            let got = g.stream.memcpy_dtov(&out_d).unwrap();
            // Reference: past = dequantized cache; the new token rounds through int8 quant/dequant.
            // (Device rounds ties-even vs the host's half-away — ≤1 int8 LSB apart, far inside tolerance.)
            let int8_round = |row: &[f32]| -> Vec<f32> {
                let (q, sc) = quantize_kv_int8(row, 1, heads, hd);
                (0..row.len()).map(|i| q[i] as f32 * sc[i / hd]).collect()
            };
            let refv =
                ref_decode_step_with(&x, &weights[0], &past_k, &past_v, &ctx0, heads, hd, dff, 1e-5, &int8_round);
            let s = crate::diff::assert_close("decode_step_int8", &got, &refv, 5e-2, 5e-2);
            let (i8b, f16b) = (model.cache().footprint_bytes(), cfg.kv_bytes(2));
            assert!(i8b < f16b, "int8 cache must be smaller than f16");
            eprintln!(
                "int8-KV decode step vs f64 ref: max_abs={:.2e} max_rel={:.2e} (Bcap={bcap}, ragged ctx 0..80); \
                 cache {:.1} MiB vs f16 {:.1} MiB ({:.2}x)",
                s.max_abs,
                s.max_rel,
                i8b as f64 / (1 << 20) as f64,
                f16b as f64 / (1 << 20) as f64,
                f16b as f64 / i8b as f64
            );
        });
    }

    /// **Whole-model decode step is paging-invariant (the first law).** A 4-layer decode step over the
    /// SAME logical sequences laid into two different physical block layouts (ascending vs descending
    /// slot allocation) must produce **bit-for-bit identical** output — proving every layer reads/writes
    /// the cache through the block table correctly (a misread would diverge). No reference needed.
    #[test]
    fn serving_decode_step_invariant_to_block_layout() {
        with_gpu("serving_decode_step_invariant_to_block_layout", |g| {
            let (heads, hd, dff, bsz, bcap, depth) = (4usize, 64usize, 256usize, 16usize, 64usize, 4usize);
            let d = heads * hd;
            let ctx0: Vec<usize> = (0..bcap).map(|b| (b * 11) % 67).collect();
            let max_bps = ctx0.iter().copied().max().unwrap().div_ceil(bsz) + 2;
            let num_blocks = bcap * max_bps + 8;
            let cfg = KvConfig { layers: depth, heads, head_dim: hd, block_size: bsz, num_blocks, num_slots: bcap, max_blocks_per_seq: max_bps };
            let mut rng = crate::diff::Rng::new(0x9001);
            let wdata = layer_weights(&mut rng, depth, d, dff);
            let weights = weights_view(&wdata);
            let x = rng.vec(bcap * d, -1.0, 1.0);

            // Run a decode step on a model whose slots are reserved in `order`, populating every layer's
            // cache plane with the same logical past (seed per layer); return the output.
            let run = |g: &mut Gpu, ascending: bool| -> (Vec<f32>, Vec<u32>) {
                let mut model = DecodeModel::new(g, &weights, cfg, dff, 128 * 1024 * 1024).unwrap();
                // Reserve in the chosen slot order ⇒ different physical block ids per sequence.
                let order: Vec<usize> = if ascending { (0..bcap).collect() } else { (0..bcap).rev().collect() };
                for &b in &order {
                    if ctx0[b] > 0 {
                        model.cache_mut().manager().reserve(b, ctx0[b]).unwrap();
                    }
                }
                // Populate every layer plane with the same logical past (deterministic per layer).
                let mut khv = vec![f16::from_f32(0.0); cfg.slab_elems()];
                let mut vhv = vec![f16::from_f32(0.0); cfg.slab_elems()];
                for layer in 0..depth {
                    let mut lrng = crate::diff::Rng::new(0xAB00 + layer as u64);
                    for b in 0..bcap {
                        let pk: Vec<f32> = lrng.vec(ctx0[b] * d, -1.0, 1.0).iter().map(|&x| f16r(x)).collect();
                        let pv: Vec<f32> = lrng.vec(ctx0[b] * d, -1.0, 1.0).iter().map(|&x| f16r(x)).collect();
                        for t in 0..ctx0[b] {
                            let (phys, off) = model.cache_mut().manager_ref().locate(b, t);
                            for h in 0..heads {
                                for dh in 0..hd {
                                    let idx = cfg.elem_offset(layer, phys, off, h, dh);
                                    khv[idx] = f16::from_f32(pk[(t * heads + h) * hd + dh]);
                                    vhv[idx] = f16::from_f32(pv[(t * heads + h) * hd + dh]);
                                }
                            }
                        }
                    }
                }
                let table = model.cache_mut().manager_ref().flat_block_table();
                {
                    let (ks, vs) = model.cache_mut().slabs_mut();
                    g.stream.memcpy_htod(&khv, ks).unwrap();
                    g.stream.memcpy_htod(&vhv, vs).unwrap();
                }
                let x_d = g.stream.memcpy_stod(&x).unwrap();
                let mut out_d = g.stream.alloc_zeros::<f32>(bcap * d).unwrap();
                model.step_on(&g.stream.clone(), &x_d, &mut out_d).unwrap();
                g.stream.synchronize().unwrap();
                (g.stream.memcpy_dtov(&out_d).unwrap(), table)
            };

            let (out_a, table_a) = run(g, true);
            let (out_b, table_b) = run(g, false);
            assert_ne!(table_a, table_b, "the two layouts must physically differ");
            assert_eq!(out_a.len(), out_b.len());
            for i in 0..out_a.len() {
                assert_eq!(out_a[i].to_bits(), out_b[i].to_bits(), "decode step diverged across block layouts at {i}");
            }
            eprintln!("{depth}-layer decode step BIT-IDENTICAL across 2 physical block layouts (Bcap={bcap}) — paging invisible end-to-end");
        });
    }

    // ============================ P4: whole-model decode CUDA graph ==================================
    use cudarc::driver::CudaContext;

    /// Disable cudarc's default event tracking for the duration of `f` (it inserts cross-stream waits a
    /// CUDA-graph capture rejects). Mirrors the gpu.rs harness helper.
    fn with_event_tracking_disabled(g: &mut Gpu, f: impl FnOnce(&mut Gpu)) {
        struct Reenable(Arc<CudaContext>);
        impl Drop for Reenable {
            fn drop(&mut self) {
                unsafe { self.0.enable_event_tracking() };
            }
        }
        let _guard = Reenable(g.ctx.clone());
        unsafe { g.ctx.disable_event_tracking() };
        f(g);
    }

    /// Best (lowest) per-iteration wall time of `run` in seconds, syncing `s` to retire the work. Warms
    /// the boost clock, then takes the fastest of several timed rounds — the least-throttled measurement.
    fn min_latency(s: &Arc<CudaStream>, mut run: impl FnMut()) -> f64 {
        use std::time::Instant;
        const WARMUP: usize = 20;
        const ROUNDS: usize = 10;
        const ITERS: usize = 30;
        for _ in 0..WARMUP {
            run();
        }
        s.synchronize().unwrap();
        let mut best = f64::MAX;
        for _ in 0..ROUNDS {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                run();
            }
            s.synchronize().unwrap();
            best = best.min(t0.elapsed().as_secs_f64() / ITERS as f64);
        }
        best
    }

    /// Build an `N`-layer `DecodeModel`, reserve+populate every layer's cache plane with `ctx0[b]`
    /// f16-rounded past tokens per slot, and advance one token (host append + metadata upload) so the
    /// model is primed for exactly one decode step.
    fn primed_model(
        g: &mut Gpu,
        weights: &[TransformerWeights],
        cfg: KvConfig,
        dff: usize,
        ctx0: &[usize],
        pool_bytes: usize,
        seed: u64,
    ) -> DecodeModel {
        let d = cfg.heads * cfg.head_dim;
        let mut model = DecodeModel::new(g, weights, cfg, dff, pool_bytes).unwrap();
        for b in 0..cfg.num_slots {
            if ctx0[b] > 0 {
                model.cache_mut().manager().reserve(b, ctx0[b]).unwrap();
            }
        }
        let mut kh = vec![f16::from_f32(0.0); cfg.slab_elems()];
        let mut vh = vec![f16::from_f32(0.0); cfg.slab_elems()];
        for layer in 0..cfg.layers {
            let mut lrng = crate::diff::Rng::new(seed + layer as u64);
            for b in 0..cfg.num_slots {
                let pk: Vec<f32> = lrng.vec(ctx0[b] * d, -1.0, 1.0).iter().map(|&x| f16r(x)).collect();
                let pv: Vec<f32> = lrng.vec(ctx0[b] * d, -1.0, 1.0).iter().map(|&x| f16r(x)).collect();
                for t in 0..ctx0[b] {
                    let (phys, off) = model.cache_mut().manager_ref().locate(b, t);
                    for h in 0..cfg.heads {
                        for dh in 0..cfg.head_dim {
                            let idx = cfg.elem_offset(layer, phys, off, h, dh);
                            kh[idx] = f16::from_f32(pk[(t * cfg.heads + h) * cfg.head_dim + dh]);
                            vh[idx] = f16::from_f32(pv[(t * cfg.heads + h) * cfg.head_dim + dh]);
                        }
                    }
                }
            }
        }
        {
            let (ks, vs) = model.cache_mut().slabs_mut();
            g.stream.memcpy_htod(&kh, ks).unwrap();
            g.stream.memcpy_htod(&vh, vs).unwrap();
        }
        model.advance_and_upload(&g.stream.clone()).unwrap();
        g.stream.synchronize().unwrap();
        model
    }

    fn graph_cfg(depth: usize) -> (KvConfig, usize, Vec<usize>, usize, usize) {
        graph_cfg_at(depth, 64)
    }

    /// [`graph_cfg`] at an arbitrary decode batch (the Bcap-scaling sweep): same layer geometry
    /// (D=512, Dff=2048), block pool and ragged 32..95 contexts sized off `bcap`.
    fn graph_cfg_at(depth: usize, bcap: usize) -> (KvConfig, usize, Vec<usize>, usize, usize) {
        let (heads, hd, dff, bsz) = (8usize, 64usize, 2048usize, 16usize);
        let ctx0: Vec<usize> = (0..bcap).map(|b| 32 + (b * 7) % 64).collect();
        let max_bps = ctx0.iter().copied().max().unwrap().div_ceil(bsz) + 2;
        let num_blocks = bcap * max_bps + 8;
        let cfg = KvConfig { layers: depth, heads, head_dim: hd, block_size: bsz, num_blocks, num_slots: bcap, max_blocks_per_seq: max_bps };
        (cfg, dff, ctx0, heads * hd, bcap)
    }

    /// **Whole-model decode CUDA-graph gate (the first law).** The entire `N`-layer decode step captured
    /// into one `cuGraphLaunch` must produce output **bit-for-bit identical** to the eager per-op
    /// `run_layers_on`, and be deterministic across two replays — the graph changes *how* the launches
    /// are issued, not *what* they compute. Capture runs on a dedicated non-blocking stream with event
    /// tracking disabled (the NULL stream is un-capturable; events break capture).
    #[test]
    fn serving_decode_graph_matches_eager() {
        with_gpu("serving_decode_graph_matches_eager", |g| {
            with_event_tracking_disabled(g, |g| {
                let depth = 6usize;
                let (cfg, dff, ctx0, d, bcap) = graph_cfg(depth);
                let mut rng = crate::diff::Rng::new(0xC0F0);
                let wdata = layer_weights(&mut rng, depth, d, dff);
                let weights = weights_view(&wdata);
                let mut model = primed_model(g, &weights, cfg, dff, &ctx0, 64 * 1024 * 1024, 0xC0DE);
                let x = rng.vec(bcap * d, -1.0, 1.0);
                let x_d = g.stream.memcpy_stod(&x).unwrap();

                // Eager reference on the default stream.
                let mut out_e = g.stream.alloc_zeros::<f32>(bcap * d).unwrap();
                model.run_layers_on(&g.stream.clone(), &x_d, &mut out_e).unwrap();
                g.stream.synchronize().unwrap();
                let ref_host = g.stream.memcpy_dtov(&out_e).unwrap();

                // Capture the whole decode step on a dedicated capturable stream.
                let cap = g.ctx.new_stream().unwrap();
                let mut out_c = cap.alloc_zeros::<f32>(bcap * d).unwrap();
                model.run_layers_on(&cap, &x_d, &mut out_c).unwrap(); // warmup (stable pool pointers)
                cap.synchronize().unwrap();
                let graph = crate::graph::Graph::capture(cap.clone(), || {
                    model.run_layers_on(&cap, &x_d, &mut out_c)
                })
                .unwrap();
                graph.launch().unwrap();
                cap.synchronize().unwrap();
                let g1 = cap.memcpy_dtov(&out_c).unwrap();
                graph.launch().unwrap();
                cap.synchronize().unwrap();
                let g2 = cap.memcpy_dtov(&out_c).unwrap();

                assert_eq!(g1.len(), ref_host.len());
                for i in 0..ref_host.len() {
                    assert_eq!(g1[i].to_bits(), ref_host[i].to_bits(), "graphed != eager at {i}");
                    assert_eq!(g1[i].to_bits(), g2[i].to_bits(), "graph replay non-deterministic at {i}");
                }
                eprintln!(
                    "{depth}-layer decode step graphed==eager bit-identical & deterministic (Bcap={bcap} D={d} Dff={dff}); \
                     whole decode step = one cuGraphLaunch (~{} launches folded)",
                    depth * 14
                );
            });
        });
    }

    /// **P4 throughput — whole-model decode: eager per-op vs one graph replay.** At decode shape every
    /// kernel runs for microseconds, so the per-launch driver overhead (`cuLaunchKernel` × ~14 × depth)
    /// dominates; folding the whole step into one `cuGraphLaunch` collapses it. Same-run ratio (the
    /// honesty law — clocks swing ~7×; named peer = **Wukong's own eager per-op decode**, the project's
    /// standing GPU baseline). Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "perf bench; needs a GPU. Run with --ignored --nocapture"]
    fn serving_decode_graph_throughput() {
        with_gpu("serving_decode_graph_throughput", |g| {
            eprintln!("device: {}", g.device_name());
            with_event_tracking_disabled(g, |g| {
                for &depth in &[1usize, 6, 12] {
                    let (cfg, dff, ctx0, d, bcap) = graph_cfg(depth);
                    let mut rng = crate::diff::Rng::new(0x7000 + depth as u64);
                    let wdata = layer_weights(&mut rng, depth, d, dff);
                    let weights = weights_view(&wdata);
                    let mut model = primed_model(g, &weights, cfg, dff, &ctx0, 128 * 1024 * 1024, 0xBEE5);
                    let x = rng.vec(bcap * d, -1.0, 1.0);
                    let x_d = g.stream.memcpy_stod(&x).unwrap();

                    // Eager: per-op launches on the default stream (named peer = Wukong's own eager decode).
                    let mut out_e = g.stream.alloc_zeros::<f32>(bcap * d).unwrap();
                    let gs = g.stream.clone();
                    let eager = min_latency(&gs, || {
                        model.run_layers_on(&gs, &x_d, &mut out_e).unwrap();
                    });

                    // Graphed: the whole decode step as one cuGraphLaunch.
                    let cap = g.ctx.new_stream().unwrap();
                    let mut out_c = cap.alloc_zeros::<f32>(bcap * d).unwrap();
                    model.run_layers_on(&cap, &x_d, &mut out_c).unwrap();
                    cap.synchronize().unwrap();
                    let graph = crate::graph::Graph::capture(cap.clone(), || {
                        model.run_layers_on(&cap, &x_d, &mut out_c)
                    })
                    .unwrap();
                    let graphed = min_latency(&cap, || {
                        graph.launch().unwrap();
                    });

                    eprintln!(
                        "depth N={depth:2} (Bcap={bcap} D={d} Dff={dff}): eager {:8.1} us | graphed {:7.1} us → \
                         graphed {:.2}x  | tokens/s eager {:.0} graphed {:.0}  (~{} launches → 1 cuGraphLaunch)",
                        eager * 1e6,
                        graphed * 1e6,
                        eager / graphed,
                        bcap as f64 / eager,
                        bcap as f64 / graphed,
                        depth * 14
                    );
                }
            });
        });
    }

    // ============================ P5: continuous batching ==========================================

    /// **Append-mask gate.** A slot with `active==0` must be skipped entirely — its padding block-table
    /// entry (0) would otherwise scatter K/V into block 0 (a live block). Mark a subset inactive and
    /// assert the *whole* cache slab is zero except the active slots' written addresses (so no inactive
    /// row touched anything, block 0 included).
    #[test]
    fn serving_kv_append_masked_skips_inactive() {
        with_gpu("serving_kv_append_masked_skips_inactive", |g| {
            let (heads, hd, bsz, bcap) = (4usize, 64usize, 16usize, 8usize);
            let d = heads * hd;
            let active = [true, false, true, true, false, false, true, false];
            let wpos = [3usize, 0, 7, 16, 0, 0, 20, 0];
            let max_bps = wpos.iter().copied().max().unwrap().div_ceil(bsz) + 2;
            let num_blocks = bcap * max_bps + 4;
            let cfg = KvConfig { layers: 1, heads, head_dim: hd, block_size: bsz, num_blocks, num_slots: bcap, max_blocks_per_seq: max_bps };
            let mut mgr = BlockManager::new(num_blocks, bsz, bcap, max_bps);
            for b in 0..bcap {
                if active[b] {
                    mgr.reserve(b, wpos[b] + 1).unwrap();
                }
            }
            let mut rng = crate::diff::Rng::new(0x5A1D);
            let knew = rng.vec(bcap * d, -1.0, 1.0);
            let vnew = rng.vec(bcap * d, -1.0, 1.0);
            let knew_d = g.stream.memcpy_stod(&knew).unwrap();
            let vnew_d = g.stream.memcpy_stod(&vnew).unwrap();
            let mut k_d = g.stream.alloc_zeros::<f16>(cfg.slab_elems()).unwrap();
            let mut v_d = g.stream.alloc_zeros::<f16>(cfg.slab_elems()).unwrap();
            let bt_d = g.stream.memcpy_stod(&mgr.flat_block_table()).unwrap();
            let wpos_u: Vec<u32> = wpos.iter().map(|&p| p as u32).collect();
            let wpos_d = g.stream.memcpy_stod(&wpos_u).unwrap();
            let act_u: Vec<u32> = active.iter().map(|&a| a as u32).collect();
            let act_d = g.stream.memcpy_stod(&act_u).unwrap();
            let func = g.function("kv_append", &kv_append_ptx(), KV_APPEND_ENTRY).unwrap();
            launch_kv_append(&g.stream, &func, &knew_d, &vnew_d, &mut k_d, &mut v_d, &bt_d, &wpos_d, &act_d, &cfg, 0, bcap).unwrap();
            g.stream.synchronize().unwrap();
            let kh = g.stream.memcpy_dtov(&k_d).unwrap();
            // Exact slab expectation: zeros everywhere except each active slot's written channels.
            let mut expect = vec![0f32; cfg.slab_elems()];
            for b in 0..bcap {
                if !active[b] {
                    continue;
                }
                let (phys, off) = mgr.locate(b, wpos[b]);
                for h in 0..heads {
                    for dh in 0..hd {
                        expect[cfg.elem_offset(0, phys, off, h, dh)] = f16r(knew[(h * hd + dh) + b * d]);
                    }
                }
            }
            for i in 0..cfg.slab_elems() {
                assert_eq!(kh[i].to_f32(), expect[i], "slab elem {i} (inactive-row leak?)");
            }
            let written = active.iter().filter(|&&a| a).count();
            eprintln!("kv_append mask: {written}/{bcap} active slots written; every inactive row skipped (block 0 intact)");
        });
    }

    /// **Batch-composition invariance (the first law for serving).** A sequence's decode output must be
    /// bit-for-bit independent of which *other* sequences share its batch — the property that makes
    /// continuous batching correct. Run slot 0's sequence (a) co-batched with 63 other active sequences
    /// carrying unrelated context, and (b) alone (all other slots masked inactive); slot 0's output row
    /// must be identical to the bit (row-independent GEMMs + per-sequence paged attention + append mask).
    #[test]
    fn serving_decode_step_invariant_to_batch_composition() {
        with_gpu("serving_decode_step_invariant_to_batch_composition", |g| {
            let (heads, hd, dff, bsz, bcap) = (4usize, 64usize, 256usize, 16usize, 64usize);
            let d = heads * hd;
            let l0 = 37usize; // slot 0's prefilled context length
            let max_bps = (l0 + 80).div_ceil(bsz) + 2;
            let num_blocks = bcap * max_bps + 8;
            let cfg = KvConfig { layers: 1, heads, head_dim: hd, block_size: bsz, num_blocks, num_slots: bcap, max_blocks_per_seq: max_bps };
            let mut rng = crate::diff::Rng::new(0x4242);
            let wdata = layer_weights(&mut rng, 1, d, dff);
            let weights = weights_view(&wdata);
            let x = rng.vec(bcap * d, -1.0, 1.0); // same input both runs; row 0 = the sequence under test
            let x_d = g.stream.memcpy_stod(&x).unwrap();

            let run = |g: &mut Gpu, ctx0: &[usize], active: &[bool]| -> Vec<f32> {
                let mut model = DecodeModel::new(g, &weights, cfg, dff, 64 * 1024 * 1024).unwrap();
                let _ = populate_one_layer(g, &mut model, ctx0, heads, hd, 0xABCD);
                let mut out = g.stream.alloc_zeros::<f32>(bcap * d).unwrap();
                model.advance_and_upload_masked(&g.stream.clone(), active).unwrap();
                model.run_layers_on(&g.stream.clone(), &x_d, &mut out).unwrap();
                g.stream.synchronize().unwrap();
                g.stream.memcpy_dtov(&out).unwrap()
            };

            // (a) co-batched: every slot active with its own (slot-0-shared) context.
            let ctx_co: Vec<usize> = (0..bcap).map(|b| if b == 0 { l0 } else { 8 + (b * 5) % 72 }).collect();
            let out_co = run(g, &ctx_co, &vec![true; bcap]);
            // (b) alone: only slot 0 active; others free + masked.
            let mut ctx_alone = vec![0usize; bcap];
            ctx_alone[0] = l0;
            let mut act_alone = vec![false; bcap];
            act_alone[0] = true;
            let out_alone = run(g, &ctx_alone, &act_alone);

            for i in 0..d {
                assert_eq!(out_co[i].to_bits(), out_alone[i].to_bits(), "slot 0 dim {i}: co-batched != alone");
            }
            eprintln!("slot-0 decode output bit-identical: co-batched (64 active) == alone (1 active) — batch composition is invisible");
        });
    }

    /// **Scheduler liveness + block conservation.** Drive >Bcap requests of ragged prompt/gen lengths to
    /// completion: every request finishes, useful tokens == Σ gen_len, every KV block returns to the pool
    /// (no leak), and two identical request streams produce the identical per-step active-count schedule
    /// (deterministic). Exercises admit → masked step → evict → free over thousands of iterations.
    #[test]
    fn serving_scheduler_drains_and_conserves_blocks() {
        with_gpu("serving_scheduler_drains_and_conserves_blocks", |g| {
            let (heads, hd, dff, bsz, bcap, depth) = (4usize, 64usize, 256usize, 16usize, 64usize, 2usize);
            let d = heads * hd;
            let max_bps = (40usize + 24).div_ceil(bsz) + 2;
            let num_blocks = bcap * max_bps + 32;
            let cfg = KvConfig { layers: depth, heads, head_dim: hd, block_size: bsz, num_blocks, num_slots: bcap, max_blocks_per_seq: max_bps };
            let mut rng = crate::diff::Rng::new(0x77AA);
            let wdata = layer_weights(&mut rng, depth, d, dff);
            let weights = weights_view(&wdata);
            let x = rng.vec(bcap * d, -1.0, 1.0);

            const NREQ: usize = 240;
            let reqs: Vec<Request> =
                (0..NREQ).map(|i| Request { prompt_len: 1 + (i * 7) % 40, gen_len: 1 + (i * 5) % 24 }).collect();
            let total_gen: usize = reqs.iter().map(|r| r.gen_len).sum();

            let drive = |g: &mut Gpu| -> (usize, usize, usize, usize, Vec<usize>) {
                let model = DecodeModel::new(g, &weights, cfg, dff, 64 * 1024 * 1024).unwrap();
                let init_free = model.cache().manager_ref().free_blocks();
                let mut sched = Scheduler::new(model);
                for &r in &reqs {
                    sched.enqueue(r);
                }
                let x_d = g.stream.memcpy_stod(&x).unwrap();
                let mut out = g.stream.alloc_zeros::<f32>(bcap * d).unwrap();
                let mut trace = Vec::new();
                let mut steps = 0;
                while !sched.is_idle() {
                    let n = sched.step(&g.stream.clone(), &x_d, &mut out).unwrap();
                    trace.push(n);
                    steps += 1;
                    assert!(steps < 100_000, "scheduler failed to drain (liveness)");
                }
                g.stream.synchronize().unwrap();
                (sched.completed(), sched.emitted(), sched.free_blocks(), init_free, trace)
            };

            let (completed, emitted, free_end, init_free, trace1) = drive(g);
            assert_eq!(completed, NREQ, "every request completes");
            assert_eq!(emitted, total_gen, "useful tokens == Σ gen_len");
            assert_eq!(free_end, init_free, "all KV blocks returned to the pool (no leak)");
            let peak = trace1.iter().copied().max().unwrap();
            let (_, _, _, _, trace2) = drive(g);
            assert_eq!(trace1, trace2, "scheduler schedule is deterministic");
            eprintln!(
                "scheduler drained {NREQ} reqs in {} steps: {emitted} tokens (=Σgen_len), peak batch {peak}/{bcap}, blocks conserved {init_free}→{free_end}",
                trace1.len()
            );
        });
    }

    /// **Scheduler graph==eager gate (the first law for the production decode loop).** Driving the
    /// SAME request stream to drain twice — once with the eager per-op [`Scheduler::step`], once with
    /// the cached whole-step graph [`Scheduler::step_graphed`] — must produce the identical per-step
    /// active-count schedule, identical accounting, and **bit-for-bit identical** decode output at
    /// every step (compared by per-step FNV digest over the output bits). The graph changes how the
    /// launches are issued, never what they compute: masks / block tables / context lengths are
    /// re-read from device buffers at every replay, so ONE capture serves admissions, evictions, and
    /// block growth alike — the property that lets the real scheduler loop run graph-driven.
    #[test]
    fn serving_scheduler_graph_matches_eager() {
        with_gpu("serving_scheduler_graph_matches_eager", |g| {
            with_event_tracking_disabled(g, |g| {
                let (heads, hd, dff, bsz, bcap, depth) = (4usize, 64usize, 256usize, 16usize, 64usize, 2usize);
                let d = heads * hd;
                let max_bps = (40usize + 24).div_ceil(bsz) + 2;
                let num_blocks = bcap * max_bps + 32;
                let cfg = KvConfig { layers: depth, heads, head_dim: hd, block_size: bsz, num_blocks, num_slots: bcap, max_blocks_per_seq: max_bps };
                let mut rng = crate::diff::Rng::new(0x6A6A);
                let wdata = layer_weights(&mut rng, depth, d, dff);
                let weights = weights_view(&wdata);
                let x = rng.vec(bcap * d, -1.0, 1.0);
                const NREQ: usize = 96;
                let reqs: Vec<Request> =
                    (0..NREQ).map(|i| Request { prompt_len: 1 + (i * 7) % 40, gen_len: 1 + (i * 5) % 24 }).collect();

                // FNV-1a over the output bits: digest equality at every step == bit-equality.
                let digest = |v: &[f32]| -> u64 {
                    let mut h = 0xcbf29ce484222325u64;
                    for f in v {
                        for b in f.to_bits().to_le_bytes() {
                            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
                        }
                    }
                    h
                };
                // Drive a full drain; per-step (active, out-digest) trace + accounting.
                let drive = |g: &mut Gpu, graphed: bool| -> (Vec<(usize, u64)>, usize, usize, usize, usize) {
                    let model = DecodeModel::new(g, &weights, cfg, dff, 64 * 1024 * 1024).unwrap();
                    let init_free = model.cache().manager_ref().free_blocks();
                    let mut sched = Scheduler::new(model);
                    for &r in &reqs {
                        sched.enqueue(r);
                    }
                    // The graphed loop needs a dedicated capturable stream; eager runs on the
                    // default. Retire the construction-time NULL-stream work (weight upload, slab
                    // zeroing) first — a non-blocking stream does not implicitly wait on it.
                    g.stream.synchronize().unwrap();
                    let stream = if graphed { g.ctx.new_stream().unwrap() } else { g.stream.clone() };
                    let x_d = stream.memcpy_stod(&x).unwrap();
                    let mut out = stream.alloc_zeros::<f32>(bcap * d).unwrap();
                    let mut trace = Vec::new();
                    while !sched.is_idle() {
                        let n = if graphed {
                            sched.step_graphed(&stream, &x_d, &mut out).unwrap()
                        } else {
                            sched.step(&stream, &x_d, &mut out).unwrap()
                        };
                        stream.synchronize().unwrap();
                        let bits = stream.memcpy_dtov(&out).unwrap();
                        trace.push((n, digest(&bits)));
                        assert!(trace.len() < 100_000, "scheduler failed to drain (liveness)");
                    }
                    (trace, sched.completed(), sched.emitted(), sched.free_blocks(), init_free)
                };

                let (trace_e, comp_e, emit_e, free_e, init_e) = drive(g, false);
                let (trace_g, comp_g, emit_g, free_g, init_g) = drive(g, true);
                assert_eq!(comp_e, NREQ, "eager completes every request");
                assert_eq!(comp_g, NREQ, "graphed completes every request");
                assert_eq!(emit_e, emit_g, "useful-token accounting diverges");
                assert_eq!(free_e, init_e, "eager leaks blocks");
                assert_eq!(free_g, init_g, "graphed leaks blocks");
                assert_eq!(trace_e.len(), trace_g.len(), "step counts diverge");
                for (i, (e, gr)) in trace_e.iter().zip(&trace_g).enumerate() {
                    assert_eq!(e.0, gr.0, "active count diverges at step {i}");
                    assert_eq!(e.1, gr.1, "decode output diverges at step {i} (graphed != eager)");
                }
                eprintln!(
                    "scheduler drain graph-driven == eager: {} steps, {NREQ} reqs, {emit_e} tokens — \
                     per-step output BIT-IDENTICAL (digest), one capture across admit/evict/growth",
                    trace_e.len()
                );
            });
        });
    }

    /// **A legal `gen_len == 0` request must not wedge the scheduler.** [`Request`] is a public struct
    /// of two plain `usize`s, so `gen_len == 0` is a legal-typed input that [`Scheduler::admit`]
    /// accepts (it only floors `prompt_len`). Un-floored, the first `retire_finished` decrements
    /// `remaining` from 0: a debug build panics with `attempt to subtract with overflow`, and a
    /// *release* build wraps to `usize::MAX`, so the slot never retires, its KV blocks never return to
    /// the pool, and `is_idle()` never becomes true — the standard drain loop spins until its liveness
    /// guard fires (or forever, in a server loop that has none). Admission floors `gen_len` at one
    /// token exactly as it already floors `prompt_len`.
    #[test]
    fn serving_scheduler_retires_zero_gen_len_request() {
        with_gpu("serving_scheduler_retires_zero_gen_len_request", |g| {
            let (heads, hd, dff, bsz, bcap, depth) = (4usize, 64usize, 128usize, 16usize, 64usize, 1usize);
            let d = heads * hd;
            let cfg = KvConfig {
                layers: depth,
                heads,
                head_dim: hd,
                block_size: bsz,
                num_blocks: 160,
                num_slots: bcap,
                max_blocks_per_seq: 4,
            };
            let mut rng = crate::diff::Rng::new(0x0E20);
            let wdata = layer_weights(&mut rng, depth, d, dff);
            let weights = weights_view(&wdata);
            let x = rng.vec(bcap * d, -1.0, 1.0);
            let model = DecodeModel::new(g, &weights, cfg, dff, 32 * 1024 * 1024).unwrap();
            let init_free = model.cache().manager_ref().free_blocks();
            let mut sched = Scheduler::new(model);
            sched.enqueue(Request { prompt_len: 8, gen_len: 0 });
            let x_d = g.stream.memcpy_stod(&x).unwrap();
            let mut out = g.stream.alloc_zeros::<f32>(bcap * d).unwrap();
            let mut steps = 0usize;
            while !sched.is_idle() {
                sched.step(&g.stream.clone(), &x_d, &mut out).unwrap();
                steps += 1;
                assert!(steps <= 4, "gen_len==0 never retired: `remaining` underflowed and pinned the slot");
            }
            g.stream.synchronize().unwrap();
            assert_eq!(steps, 1, "a gen_len==0 request emits its one floored token and retires the same step");
            assert_eq!(sched.completed(), 1, "the request completes");
            assert_eq!(sched.emitted(), 1, "gen_len is floored at one token");
            assert_eq!(sched.free_blocks(), init_free, "the retired slot's KV blocks return to the pool");
            eprintln!(
                "gen_len==0 request retired in 1 step, blocks conserved {init_free}->{} (no `remaining` underflow)",
                sched.free_blocks()
            );
        });
    }

    /// **A failed decode advance must leave the host allocator untouched.**
    /// [`DecodeModel::advance_and_upload_masked`] mutates the [`BlockManager`] slot by slot but uploads
    /// the four device metadata buffers only *after* the whole loop, so a mid-loop `OutOfBlocks` used to
    /// leave the earlier slots' context lengths already advanced with nothing uploaded. The caller's
    /// documented recovery (preempt, retry) would then re-read those inflated lengths as the next write
    /// positions, writing each survivor's next token one slot past a hole the attention kernel still
    /// covers — stale f16 bytes from whichever sequence last owned that block, silently wrong logits.
    /// The advance is now all-or-nothing: a `can_grow` feasibility pass runs before any mutation.
    #[test]
    fn serving_failed_advance_leaves_allocator_unchanged() {
        with_gpu("serving_failed_advance_leaves_allocator_unchanged", |g| {
            let (heads, hd, dff, bsz, bcap, depth) = (4usize, 64usize, 128usize, 16usize, 64usize, 1usize);
            let d = heads * hd;
            // max_blocks_per_seq = 2 ⇒ a sequence tops out at 32 cached tokens; the pool itself is
            // roomy, so the only way to fail is the per-sequence table-width cap.
            let cfg = KvConfig {
                layers: depth,
                heads,
                head_dim: hd,
                block_size: bsz,
                num_blocks: 160,
                num_slots: bcap,
                max_blocks_per_seq: 2,
            };
            let mut rng = crate::diff::Rng::new(0x0A70);
            let wdata = layer_weights(&mut rng, depth, d, dff);
            let weights = weights_view(&wdata);
            let mut model = DecodeModel::new(g, &weights, cfg, dff, 32 * 1024 * 1024).unwrap();
            // Slots 0 and 1 have room; slot 3 sits exactly at its per-sequence cap (2 blocks, 32 tokens).
            model.cache_mut().manager().reserve(0, 5).unwrap();
            model.cache_mut().manager().reserve(1, 5).unwrap();
            model.cache_mut().manager().reserve(3, 32).unwrap();
            let before: Vec<usize> = (0..bcap).map(|b| model.cache().manager_ref().context_len(b)).collect();
            let free_before = model.cache().manager_ref().free_blocks();
            let epoch_before = model.cache().manager_ref().layout_epoch();

            let stream = g.stream.clone();
            assert!(
                model.advance_and_upload_masked(&stream, &vec![true; bcap]).is_err(),
                "slot 3 is at its max_blocks_per_seq cap ⇒ the advance must fail"
            );
            let after: Vec<usize> = (0..bcap).map(|b| model.cache().manager_ref().context_len(b)).collect();
            assert_eq!(before, after, "a failed advance must not advance any slot's context length");
            assert_eq!(
                free_before,
                model.cache().manager_ref().free_blocks(),
                "a failed advance must not consume physical blocks"
            );
            assert_eq!(
                epoch_before,
                model.cache().manager_ref().layout_epoch(),
                "a failed advance must not move the block-table layout epoch"
            );
            // With the capped slot masked off, the same call succeeds and advances exactly the active slots.
            let mut mask = vec![true; bcap];
            mask[3] = false;
            model.advance_and_upload_masked(&stream, &mask).unwrap();
            stream.synchronize().unwrap();
            for b in 0..bcap {
                let want = before[b] + usize::from(b != 3);
                assert_eq!(model.cache().manager_ref().context_len(b), want, "slot {b} advance");
            }
            eprintln!(
                "failed decode advance is atomic: ctx lengths, free blocks ({free_before}) and layout epoch \
                 ({epoch_before}) all unchanged; masking the capped slot then advances the rest"
            );
        });
    }

    /// **Static-batching policy gate.** A [`Scheduler::new_static`] scheduler must (a) admit ONLY
    /// when the previous batch has fully drained — never refill a freed slot mid-flight (the policy
    /// that defines the goodput bench's honest peer), and (b) still drain every request and conserve
    /// every block. Verified by snapshotting the admission counter each step and requiring the
    /// active count to have been zero whenever it moves.
    #[test]
    fn serving_static_batching_admits_only_when_drained() {
        with_gpu("serving_static_batching_admits_only_when_drained", |g| {
            let (heads, hd, dff, bsz, bcap, depth) = (4usize, 64usize, 256usize, 16usize, 64usize, 1usize);
            let d = heads * hd;
            let max_bps = (40usize + 24).div_ceil(bsz) + 2;
            let num_blocks = bcap * max_bps + 32;
            let cfg = KvConfig { layers: depth, heads, head_dim: hd, block_size: bsz, num_blocks, num_slots: bcap, max_blocks_per_seq: max_bps };
            let mut rng = crate::diff::Rng::new(0x57A7);
            let wdata = layer_weights(&mut rng, depth, d, dff);
            let weights = weights_view(&wdata);
            let x = rng.vec(bcap * d, -1.0, 1.0);
            const NREQ: usize = 96;
            let reqs: Vec<Request> =
                (0..NREQ).map(|i| Request { prompt_len: 1 + (i * 7) % 40, gen_len: 1 + (i * 5) % 24 }).collect();
            let total_gen: usize = reqs.iter().map(|r| r.gen_len).sum();

            let model = DecodeModel::new(g, &weights, cfg, dff, 64 * 1024 * 1024).unwrap();
            let init_free = model.cache().manager_ref().free_blocks();
            let mut sched = Scheduler::new_static(model);
            for &r in &reqs {
                sched.enqueue(r);
            }
            let x_d = g.stream.memcpy_stod(&x).unwrap();
            let mut out = g.stream.alloc_zeros::<f32>(bcap * d).unwrap();
            let (mut steps, mut batches) = (0usize, 0usize);
            let mut prev_active_after = 0usize; // active slots after the previous step's retirements
            let mut prev_admitted = 0usize;
            while !sched.is_idle() {
                sched.step(&g.stream.clone(), &x_d, &mut out).unwrap();
                if sched.admitted() > prev_admitted {
                    assert_eq!(prev_active_after, 0, "static batching refilled a slot mid-flight");
                    prev_admitted = sched.admitted();
                    batches += 1;
                }
                prev_active_after = sched.num_active();
                steps += 1;
                assert!(steps < 100_000, "static scheduler failed to drain (liveness)");
            }
            g.stream.synchronize().unwrap();
            assert_eq!(sched.completed(), NREQ, "every request completes");
            assert_eq!(sched.emitted(), total_gen, "useful tokens == Σ gen_len");
            assert_eq!(sched.free_blocks(), init_free, "all KV blocks returned to the pool");
            assert!(batches >= NREQ / bcap, "at least ceil(NREQ/Bcap) admission waves");
            eprintln!(
                "static batching drained {NREQ} reqs in {steps} steps across {batches} full-drain batches — \
                 zero mid-flight refills (the honest peer policy holds)"
            );
        });
    }

    /// **P5 throughput — continuous-batching goodput vs batch fill, swept over Bcap.** The fixed-shape
    /// decode step computes all `Bcap` rows regardless of how many carry a live request, and at these
    /// sizes it is weight-HBM-bound (every GEMM streams the same weights whatever M is), so per-step
    /// latency grows far slower than the batch: goodput scales ~linearly with fill *and keeps scaling
    /// as Bcap itself grows* until M reaches the low hundreds. The sweep measures both levers: fill at
    /// fixed Bcap (the continuous-batching win) and Bcap itself (the batch-shape headroom).
    ///
    /// **Measurement honesty (the laptop clock swings ~7×).** One model + ONE captured graph per Bcap —
    /// a fill level is *contents* of the metadata buffers (ctx/wpos/active), not a shape, so every fill
    /// replays the same graph after a small re-upload. All (Bcap, fill) points are then timed
    /// **interleaved, best-of-N**: each round times every point back-to-back so they share the same
    /// clock state, and the per-point minimum picks its boosted time. A naive sequential sweep is
    /// *invalid* here — it catches early points cold and late points boosted, skewing every ratio.
    /// Named peer = Wukong's own single-sequence (fill=1) decode at the same Bcap. --ignored.
    #[test]
    #[ignore = "perf bench; needs a GPU. Run with --ignored --nocapture"]
    fn serving_continuous_batching_goodput() {
        with_gpu("serving_continuous_batching_goodput", |g| {
            eprintln!("device: {}", g.device_name());
            with_event_tracking_disabled(g, |g| {
                use std::time::Instant;
                let depth = 12usize;
                // KV budget: cap the resident f16 slabs well under the 6 GB part (weights, pools,
                // activations, and the desktop share the rest). All three Bcaps stay resident at once
                // (~1.4 GiB of KV total) so the timing rounds can interleave under one clock.
                const KV_BUDGET: usize = 4 << 30;
                let bcaps = [64usize, 128, 256];
                let mut rng = crate::diff::Rng::new(0x9100);
                // Same layer geometry at every Bcap (D=512, Dff=2048) ⇒ one weight set serves all.
                let (_, dff, _, d, _) = graph_cfg_at(depth, bcaps[0]);
                let wdata = layer_weights(&mut rng, depth, d, dff);
                let weights = weights_view(&wdata);
                let x_all = rng.vec(bcaps[bcaps.len() - 1] * d, -1.0, 1.0);

                // A Bcap's model + its one captured graph + per-fill best latencies, kept alive.
                struct HeldB {
                    bcap: usize,
                    cap: Arc<CudaStream>,
                    graph: crate::graph::Graph,
                    model: DecodeModel,
                    _x: CudaSlice<f32>,
                    _out: CudaSlice<f32>,
                    table: Vec<u32>,
                    ctx0: Vec<usize>,
                    fills: Vec<usize>,
                    best: Vec<f64>,
                }
                impl HeldB {
                    /// Steer the shared graph to `fill` active slots: slots < fill carry their ragged
                    /// context (+1 for the appended token), the rest read as empty and masked.
                    fn set_fill(&mut self, fill: usize) {
                        let cl: Vec<u32> =
                            (0..self.bcap).map(|b| if b < fill { self.ctx0[b] as u32 + 1 } else { 0 }).collect();
                        let wpos: Vec<u32> = self.ctx0.iter().map(|&c| c as u32).collect();
                        let act: Vec<u32> = (0..self.bcap).map(|b| (b < fill) as u32).collect();
                        self.model.upload_metadata(&self.cap, &self.table, &cl, &wpos, &act).unwrap();
                    }
                }
                let mut held: Vec<HeldB> = Vec::new();
                for &bcap in &bcaps {
                    let (cfg, dff, ctx0, d, _) = graph_cfg_at(depth, bcap);
                    let kv = cfg.assert_kv_budget(2, KV_BUDGET);
                    let pool_bytes = 48 * 1024 * 1024 * (bcap / 64);
                    let mut model = DecodeModel::new(g, &weights, cfg, dff, pool_bytes).unwrap();
                    for b in 0..bcap {
                        model.cache_mut().manager().reserve(b, ctx0[b]).unwrap();
                    }
                    let table = model.cache().manager_ref().flat_block_table();
                    let cap = g.ctx.new_stream().unwrap();
                    let x_d = g.stream.memcpy_stod(&x_all[..bcap * d]).unwrap();
                    let mut out_c = cap.alloc_zeros::<f32>(bcap * d).unwrap();
                    // Stage all-active metadata, warm up once (stable pool pointers), capture. The
                    // graph bakes buffer pointers + Bcap-shaped grids; the fill stays re-steerable.
                    {
                        let cl: Vec<u32> = ctx0.iter().map(|&c| c as u32 + 1).collect();
                        let wpos: Vec<u32> = ctx0.iter().map(|&c| c as u32).collect();
                        let act = vec![1u32; bcap];
                        model.upload_metadata(&cap, &table, &cl, &wpos, &act).unwrap();
                    }
                    // Retire the NULL-stream construction work (weights, slabs, x) before the
                    // dedicated stream reads it — no implicit NULL-stream ordering here.
                    g.stream.synchronize().unwrap();
                    model.run_layers_on(&cap, &x_d, &mut out_c).unwrap();
                    cap.synchronize().unwrap();
                    let graph = crate::graph::Graph::capture(cap.clone(), || model.run_layers_on(&cap, &x_d, &mut out_c)).unwrap();
                    eprintln!(
                        "  Bcap={bcap:3}: KV slabs {:6.1} MiB (budget {:.1} GiB) | pool {} MiB | ctx 32..95",
                        kv as f64 / (1 << 20) as f64,
                        KV_BUDGET as f64 / (1u64 << 30) as f64,
                        pool_bytes / (1 << 20)
                    );
                    let fills = vec![1usize, bcap / 4, bcap / 2, bcap];
                    let n_fills = fills.len();
                    held.push(HeldB { bcap, cap, graph, model, _x: x_d, _out: out_c, table, ctx0, fills, best: vec![f64::MAX; n_fills] });
                }

                // Global warmup on the largest point to lock the boost clock high before any timing.
                {
                    let last = held.len() - 1;
                    let full = *held[last].fills.last().unwrap();
                    held[last].set_fill(full);
                    for _ in 0..200 {
                        held[last].graph.launch().unwrap();
                    }
                    held[last].cap.synchronize().unwrap();
                }

                // Interleaved best-of-N: every round times all (Bcap, fill) points adjacently (shared
                // clock), min per point. The fill re-upload happens outside the timed region.
                const ROUNDS: usize = 15;
                const ITERS: usize = 20;
                for _ in 0..ROUNDS {
                    for hi in 0..held.len() {
                        for fi in 0..held[hi].fills.len() {
                            let fill = held[hi].fills[fi];
                            held[hi].set_fill(fill);
                            held[hi].cap.synchronize().unwrap();
                            let t = Instant::now();
                            for _ in 0..ITERS {
                                held[hi].graph.launch().unwrap();
                            }
                            held[hi].cap.synchronize().unwrap();
                            let dt = t.elapsed().as_secs_f64() / ITERS as f64;
                            if dt < held[hi].best[fi] {
                                held[hi].best[fi] = dt;
                            }
                        }
                    }
                }

                eprintln!(
                    "continuous-batching goodput (graphed {depth}-layer decode step, D={d} Dff={dff}; \
                     interleaved best-of-N across every (Bcap, fill) point):"
                );
                for h in &held {
                    let l1 = h.best[0];
                    for (fi, &fill) in h.fills.iter().enumerate() {
                        let l = h.best[fi];
                        let gp = fill as f64 / l;
                        eprintln!(
                            "  Bcap={:3} fill {:3}: step {:7.1} us | goodput {:8.0} tok/s | {:5.1}x vs fill=1 (step latency {:.2}x)",
                            h.bcap,
                            fill,
                            l * 1e6,
                            gp,
                            gp * l1,
                            l / l1
                        );
                    }
                }
                // Bcap-scaling table: each Bcap's full-fill point against Bcap=64's.
                let base = *held[0].best.last().unwrap();
                eprintln!("Bcap scaling (full fill):");
                for h in &held {
                    let l = *h.best.last().unwrap();
                    let gp = h.bcap as f64 / l;
                    eprintln!(
                        "  Bcap={:3}: step {:7.1} us | goodput {:8.0} tok/s | {:.2}x vs Bcap=64 (step latency {:.2}x)",
                        h.bcap,
                        l * 1e6,
                        gp,
                        gp / (64.0 / base),
                        l / base
                    );
                }
                // Sweep-derived legacy multiples (full fill vs fill=1 per Bcap), then free the
                // sweep's models/graphs before the drains below re-allocate.
                let sweep: Vec<(usize, f64)> = held
                    .iter()
                    .map(|h| {
                        let lf = *h.best.last().unwrap();
                        (h.bcap, (h.bcap as f64 / lf) * h.best[0])
                    })
                    .collect();
                drop(held);

                // ---- Continuous vs STATIC batching over the REAL scheduler loop (same kernels) ----
                // The fill sweep above prices the M-amortization win (a fuller fixed batch beats an
                // emptier one). The *scheduling* win is separate: continuous batching refills freed
                // slots mid-flight; static batching (the classic peer) admits a full batch and waits
                // for its LAST straggler before refilling, idling slots on ragged gen lengths.
                // Identical kernels, identical graph-driven step, identical request stream — the
                // ratio below is continuous batching's true scheduling contribution, with all host
                // scheduling + metadata-upload costs included (this times the real Scheduler drain,
                // not a bare graph replay). Adjacent same-run A/B per Bcap (the clock-honesty rule).
                eprintln!("continuous vs static batching (graph-driven Scheduler drain, depth {depth}, ragged gen 8..63):");
                let mut last_line = None;
                for &bcap in &bcaps {
                    let (cfg, dff, _, d, _) = graph_cfg_at(depth, bcap);
                    let nreq = 3 * bcap;
                    let reqs: Vec<Request> = (0..nreq)
                        .map(|i| Request { prompt_len: 8 + (i * 11) % 56, gen_len: 8 + (i * 13) % 56 })
                        .collect();
                    let mut drain = |static_batching: bool| -> (f64, usize, usize) {
                        let model = DecodeModel::new(g, &weights, cfg, dff, 48 * 1024 * 1024 * (bcap / 64)).unwrap();
                        let mut sched =
                            if static_batching { Scheduler::new_static(model) } else { Scheduler::new(model) };
                        for &r in &reqs {
                            sched.enqueue(r);
                        }
                        g.stream.synchronize().unwrap(); // retire NULL-stream construction work
                        let stream = g.ctx.new_stream().unwrap();
                        let x_d = stream.memcpy_stod(&x_all[..bcap * d]).unwrap();
                        let mut out = stream.alloc_zeros::<f32>(bcap * d).unwrap();
                        // Exclude the first WARM steps (graph capture + clock ramp) from the timed
                        // region; tokens are counted from the same instant.
                        const WARM: usize = 16;
                        let mut steps = 0usize;
                        let mut t0 = Instant::now();
                        let mut tok0 = 0usize;
                        while !sched.is_idle() {
                            sched.step_graphed(&stream, &x_d, &mut out).unwrap();
                            steps += 1;
                            if steps == WARM {
                                stream.synchronize().unwrap();
                                t0 = Instant::now();
                                tok0 = sched.emitted();
                            }
                            assert!(steps < 1_000_000, "drain liveness");
                        }
                        stream.synchronize().unwrap();
                        (t0.elapsed().as_secs_f64(), sched.emitted() - tok0, steps)
                    };
                    let (dt_c, tok_c, steps_c) = drain(false);
                    let (dt_s, tok_s, steps_s) = drain(true);
                    let (gp_c, gp_s) = (tok_c as f64 / dt_c, tok_s as f64 / dt_s);
                    eprintln!(
                        "  Bcap={bcap:3}: continuous {gp_c:8.0} tok/s ({steps_c:4} steps) | static {gp_s:8.0} tok/s ({steps_s:4} steps) | continuous/static {:.2}x",
                        gp_c / gp_s
                    );
                    last_line = Some((bcap, gp_c, gp_c / gp_s));
                }

                // The three-number headline: the absolute (real scheduler loop, host costs
                // included), the M-amortization multiple (legacy fill=1 peer, interleaved sweep),
                // and the honest scheduling multiple (static-batching peer, same kernels).
                let (bcap_h, gp_h, vs_static) = last_line.unwrap();
                let vs_fill1 = sweep.iter().find(|s| s.0 == bcap_h).unwrap().1;
                eprintln!(
                    "HEADLINE: {gp_h:.0} tok/s decode goodput at Bcap={bcap_h} (graph-driven Scheduler drain, {depth}-layer D={d} model) | \
                     {vs_fill1:.1}x vs fill=1 same-Bcap (M-amortization, same-clock interleaved) | \
                     {vs_static:.2}x vs static batching (the scheduling win, same kernels)"
                );
            });
        });
    }

    // ===================== P7: tensor-parallel partition simulation (single GPU) ====================

    /// Run the tuned WMMA f16 GEMM `C[m,n] = A[m,k]·B[n,k]ᵀ` (the projection kernel TP would split),
    /// f16 inputs / f32 output. Calls — does not modify — `wmma_nt_f16_sm_db`.
    fn wmma_gemm_nt(g: &mut Gpu, a16: &[f16], b16: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
        let a_d = g.stream.memcpy_stod(a16).unwrap();
        let b_d = g.stream.memcpy_stod(b16).unwrap();
        let mut c_d = g.stream.alloc_zeros::<f32>(m * n).unwrap();
        let func = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_sm_db").unwrap();
        let (mm, nn, kk) = (m as u32, n as u32, k as u32);
        let mut b = g.stream.launch_builder(&func);
        b.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
        unsafe { b.launch(wmma_sm_cfg(m, n)).unwrap() };
        g.stream.synchronize().unwrap();
        g.stream.memcpy_dtov(&c_d).unwrap()
    }

    /// **TP column-parallel partition is bit-exact (validates the Megatron partition math on one GPU).**
    /// QKV / FFN-up projections are *column-parallel*: split the output dim `N` across GPUs, each computing
    /// its `N/T` shard from the full input with **no communication**, then concat. Simulated here by
    /// splitting `N` into two halves and reassembling — the result is **bit-for-bit identical** to the
    /// unsplit GEMM, because each output column is the *same* reduction (splitting `N` never changes the
    /// `K`-accumulation). The 2-GPU number is unmeasured; the partition math is proven exact.
    #[test]
    fn tp_column_parallel_gemm_split_is_bit_exact() {
        with_gpu("tp_column_parallel_gemm_split_is_bit_exact", |g| {
            let (m, n, k) = (64usize, 256usize, 256usize); // N%128==0 ⇒ N/2 a multiple of the 64 tile
            let mut rng = crate::diff::Rng::new(0x701);
            let a: Vec<f16> = rng.vec(m * k, -1.0, 1.0).iter().map(|&x| f16::from_f32(x)).collect();
            let b: Vec<f16> = rng.vec(n * k, -1.0, 1.0).iter().map(|&x| f16::from_f32(x)).collect();
            let full = wmma_gemm_nt(g, &a, &b, m, n, k);
            // Two "GPUs", each owning N/2 output columns (B rows): C_g = A · B_g^T, no comm.
            let nh = n / 2;
            let c0 = wmma_gemm_nt(g, &a, &b[0..nh * k], m, nh, k);
            let c1 = wmma_gemm_nt(g, &a, &b[nh * k..n * k], m, nh, k);
            let mut part = vec![0f32; m * n];
            for mm in 0..m {
                for j in 0..nh {
                    part[mm * n + j] = c0[mm * nh + j];
                    part[mm * n + nh + j] = c1[mm * nh + j];
                }
            }
            for i in 0..m * n {
                assert_eq!(part[i].to_bits(), full[i].to_bits(), "column-parallel split != unsplit at {i}");
            }
            eprintln!("TP column-parallel ({m}×{n}×{k}, split N {n}→2×{nh}): concat == unsplit BIT-IDENTICAL (no all-reduce needed)");
        });
    }

    /// **TP row-parallel all-reduce matches (validates the partition math; the sum is the only comm).**
    /// Attention-out / FFN-down projections are *row-parallel*: split the contraction dim `K` across GPUs,
    /// each computing a partial `[M,N]`, then **all-reduce (sum)** the partials. Simulated by splitting `K`
    /// into two halves and summing. The sum reassociates the `K` reduction (float, not associative), so it
    /// matches the unsplit GEMM within tolerance — *exactly* the numerical behavior of a real ring
    /// all-reduce. (The repo's reassociation-exception class; the all-reduce itself is the unmeasured seam.)
    #[test]
    fn tp_row_parallel_gemm_allreduce_matches() {
        with_gpu("tp_row_parallel_gemm_allreduce_matches", |g| {
            let (m, n, k) = (64usize, 256usize, 256usize); // K/2 a multiple of the 16-wide K tile
            let mut rng = crate::diff::Rng::new(0x702);
            let a: Vec<f16> = rng.vec(m * k, -1.0, 1.0).iter().map(|&x| f16::from_f32(x)).collect();
            let b: Vec<f16> = rng.vec(n * k, -1.0, 1.0).iter().map(|&x| f16::from_f32(x)).collect();
            let full = wmma_gemm_nt(g, &a, &b, m, n, k);
            // Each "GPU" owns a K-shard of both operands and computes a full-[M,N] partial.
            let kh = k / 2;
            let slice_k = |src: &[f16], rows: usize, lo: usize, hi: usize| -> Vec<f16> {
                let mut out = Vec::with_capacity(rows * (hi - lo));
                for r in 0..rows {
                    out.extend_from_slice(&src[r * k + lo..r * k + hi]);
                }
                out
            };
            let c0 = wmma_gemm_nt(g, &slice_k(&a, m, 0, kh), &slice_k(&b, n, 0, kh), m, n, kh);
            let c1 = wmma_gemm_nt(g, &slice_k(&a, m, kh, k), &slice_k(&b, n, kh, k), m, n, kh);
            // All-reduce (sum) the two partials — the one inter-GPU collective in row-parallel TP.
            let summed: Vec<f32> = c0.iter().zip(&c1).map(|(&x, &y)| x + y).collect();
            let s = crate::diff::assert_close("tp_row_parallel", &summed, &full, 1e-2, 1e-2);
            eprintln!(
                "TP row-parallel ({m}×{n}×{k}, split K {k}→2×{kh}): summed partials vs unsplit max_abs={:.2e} max_rel={:.2e} \
                 (all-reduce reassociates the K reduction — float, not bit-exact, as a real ring all-reduce)",
                s.max_abs, s.max_rel
            );
        });
    }
}
