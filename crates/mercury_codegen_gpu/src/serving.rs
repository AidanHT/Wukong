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
//!   captures (P4) and a continuous-batching scheduler drives (P5).
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
    kv_append_ptx, launch_kv_append, launch_paged_attn_decode, paged_attn_decode_ptx,
    KV_APPEND_ENTRY, PAGED_ATTN_ENTRY,
};
use crate::paged_kv::{KvConfig, PagedKvCache};
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

    /// Upload the weights (narrowed to f16) and load every kernel. `cfg` is the cache geometry; `dff`
    /// the FFN inner dim. Requires `D % 64 == 0`, `Dff % 64 == 0`, `Bcap % 64 == 0` (the 64×64 WMMA
    /// tile), and `head_dim ∈ {64, 128}` (the generated paged-attn kernels).
    pub fn new(g: &mut Gpu, w: &TransformerWeights, cfg: KvConfig, dff: usize) -> Result<Self, DriverError> {
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
        let attn_key: &'static str = match cfg.head_dim {
            64 => "paged_attn_d64",
            128 => "paged_attn_d128",
            _ => unreachable!(),
        };
        let f_attn = g.function(attn_key, &paged_attn_decode_ptx(cfg.head_dim), PAGED_ATTN_ENTRY)?;
        let f_append = g.function("kv_append", &kv_append_ptx(), KV_APPEND_ENTRY)?;
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
        })
    }

    /// Run one decode step for this layer on `stream`, all scratch from `pool`. `x_d` is the `[Bcap,D]`
    /// input activation; the new K/V are appended into `(k_cache, v_cache)` at `wpos_d[slot]` (which
    /// must already be reserved) for layer plane `layer`; attention reads the cache over `cl_d[slot]`
    /// (= post-append context). The `[Bcap,D]` result is written into `out`. Every pooled buffer is a
    /// full-overwrite output (uninit `alloc` is safe — proven by the poison gate).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step_on(
        &self,
        stream: &Arc<CudaStream>,
        pool: &mut DevicePool,
        k_cache: &mut CudaSlice<f16>,
        v_cache: &mut CudaSlice<f16>,
        x_d: &CudaSlice<f32>,
        bt_d: &CudaSlice<u32>,
        cl_d: &CudaSlice<u32>,
        wpos_d: &CudaSlice<u32>,
        layer: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), DriverError> {
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
        // Append the new token's K/V into this layer's cache plane at each slot's write position.
        launch_kv_append(stream, &self.f_append, &k, &v, k_cache, v_cache, bt_d, wpos_d, &self.cfg, layer, bcap)?;
        // Paged decode attention over the (now-updated) cache.
        let mut attn = pool.alloc::<f32>(bcap * d)?;
        launch_paged_attn_decode(stream, &self.f_attn, &q, k_cache, v_cache, &mut attn, bt_d, cl_d, &self.cfg, layer, bcap, self.scale)?;
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
    /// `[Bcap,D]` ping-pong activations between layers (persistent — outside the pool).
    bufs: [CudaSlice<f32>; 2],
    cfg: KvConfig,
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

    /// Build the `N` decode layers (one weight set each), the paged cache (`cfg`), a `pool_bytes`
    /// scratch arena, and the per-step metadata + ping-pong buffers. All share `g`'s stream.
    pub fn new(
        g: &mut Gpu,
        weights: &[TransformerWeights],
        cfg: KvConfig,
        dff: usize,
        pool_bytes: usize,
    ) -> Result<Self, DriverError> {
        assert!(!weights.is_empty(), "model needs at least one layer");
        assert_eq!(weights.len(), cfg.layers, "cfg.layers must equal the number of weight sets");
        let d = cfg.heads * cfg.head_dim;
        let mut layers = Vec::with_capacity(weights.len());
        for w in weights {
            layers.push(DecodeLayer::new(g, w, cfg, dff)?);
        }
        let cache = PagedKvCache::new(g.stream.clone(), cfg)?;
        let pool = DevicePool::new(g.stream.clone(), pool_bytes)?;
        let bt_d = g.stream.alloc_zeros::<u32>(cfg.num_slots * cfg.max_blocks_per_seq)?;
        let cl_d = g.stream.alloc_zeros::<u32>(cfg.num_slots)?;
        let wpos_d = g.stream.alloc_zeros::<u32>(cfg.num_slots)?;
        let bufs = [g.stream.alloc_zeros::<f32>(cfg.num_slots * d)?, g.stream.alloc_zeros::<f32>(cfg.num_slots * d)?];
        Ok(Self { layers, cache, pool, bt_d, cl_d, wpos_d, bufs, cfg })
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
        let bcap = self.cfg.num_slots;
        // Host: append a token to every slot, recording each slot's pre-append write position.
        let mut wpos = vec![0u32; bcap];
        for (b, w) in wpos.iter_mut().enumerate() {
            *w = self.cache.manager().context_len(b) as u32;
            self.cache
                .manager()
                .append(b)
                .map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
        }
        // Upload the shared per-step metadata once (block table may have grown during append).
        let table = self.cache.manager_ref().flat_block_table();
        let lens = self.cache.manager_ref().ctx_lens();
        stream.memcpy_htod(&table, &mut self.bt_d)?;
        stream.memcpy_htod(&lens, &mut self.cl_d)?;
        stream.memcpy_htod(&wpos, &mut self.wpos_d)?;
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
        // Disjoint field borrows (the launchers need &mut cache slabs + &mut pool + & metadata at once).
        let Self { layers, cache, pool, bt_d, cl_d, wpos_d, bufs, .. } = self;
        let n = layers.len();
        let (ks, vs) = cache.slabs_mut();
        if n == 1 {
            pool.reset();
            return layers[0].forward_step_on(stream, pool, ks, vs, x_d, bt_d, cl_d, wpos_d, 0, out);
        }
        pool.reset();
        layers[0].forward_step_on(stream, pool, ks, vs, x_d, bt_d, cl_d, wpos_d, 0, &mut bufs[0])?;
        let mut cur = 0usize; // layer i-1's output lives in bufs[cur]
        for (i, layer) in layers.iter().enumerate().take(n - 1).skip(1) {
            pool.reset();
            let (a, b) = bufs.split_at_mut(1);
            let (src, dst) = if cur == 0 { (&a[0], &mut b[0]) } else { (&b[0], &mut a[0]) };
            layer.forward_step_on(stream, pool, ks, vs, src, bt_d, cl_d, wpos_d, i, dst)?;
            cur = 1 - cur;
        }
        pool.reset();
        // Last layer reads the current buffer, writes the caller's `out`.
        let src = &bufs[cur];
        layers[n - 1].forward_step_on(stream, pool, ks, vs, src, bt_d, cl_d, wpos_d, n - 1, out)
    }

    /// Upload the current host block table / context lengths / write positions to the device metadata
    /// buffers **without** running the layers — the host half of [`step_on`](Self::step_on), exposed so
    /// a graph can capture only the launch half ([`run_layers_on`](Self::run_layers_on)). Returns the
    /// per-slot write positions it appended.
    pub fn advance_and_upload(&mut self, stream: &Arc<CudaStream>) -> Result<Vec<u32>, DriverError> {
        let bcap = self.cfg.num_slots;
        let mut wpos = vec![0u32; bcap];
        for (b, w) in wpos.iter_mut().enumerate() {
            *w = self.cache.manager().context_len(b) as u32;
            self.cache
                .manager()
                .append(b)
                .map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
        }
        let table = self.cache.manager_ref().flat_block_table();
        let lens = self.cache.manager_ref().ctx_lens();
        stream.memcpy_htod(&table, &mut self.bt_d)?;
        stream.memcpy_htod(&lens, &mut self.cl_d)?;
        stream.memcpy_htod(&wpos, &mut self.wpos_d)?;
        Ok(wpos)
    }

    /// The two ping-pong activation buffers (a graph bakes their pointers — keep alive).
    pub fn buffers(&self) -> &[CudaSlice<f32>; 2] {
        &self.bufs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::{
        kv_append_ptx, launch_kv_append, reference_decode_attn, KV_APPEND_ENTRY,
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
            // Full attention context: the cached past ++ the new token (rounded f16, as the cache stores).
            let ctx = ctx0[b];
            let mut kfull = past_k[b].clone();
            let mut vfull = past_v[b].clone();
            kfull.extend(kk.iter().map(|&z| f16r(z)));
            vfull.extend(vv.iter().map(|&z| f16r(z)));
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
            let func = g.function("kv_append", &kv_append_ptx(), KV_APPEND_ENTRY).unwrap();
            launch_kv_append(&g.stream, &func, &knew_d, &vnew_d, &mut k_d, &mut v_d, &bt_d, &wpos_d, &cfg, 0, bcap).unwrap();
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
        let (heads, hd, dff, bsz, bcap) = (8usize, 64usize, 2048usize, 16usize, 64usize);
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
    /// honesty law — clocks swing ~7×; named peer = **Mercury's own eager per-op decode**, the project's
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

                    // Eager: per-op launches on the default stream (named peer = Mercury's own eager decode).
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
}
