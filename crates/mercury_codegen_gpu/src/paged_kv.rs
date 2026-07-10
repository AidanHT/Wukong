//! Paged KV-cache (serving / M-serving) — a vLLM-style block-table-indexed key/value cache.
//!
//! Autoregressive decode attends, at every step, over **all** previously generated tokens. Storing
//! each sequence's K/V contiguously forces a worst-case `max_seq_len` reservation per sequence and
//! fragments the 6 GB device badly. **PagedAttention** instead chops the cache into fixed-size
//! **blocks** (`block_size` tokens each) that live anywhere in one big slab; a per-sequence **block
//! table** maps a sequence's *logical* block `i` to whatever *physical* block the allocator handed
//! out. A sequence grows one block at a time, blocks are recycled on free, and two sequences never
//! need to be contiguous — so the only reservation is the actual token count, rounded up to a block.
//!
//! This module is split so the *policy* is testable with no device:
//! - [`BlockManager`] — **pure host logic**: the free-list allocator + per-slot block tables +
//!   context lengths. Every `allocate`/`append`/`free`/`flatten` path is unit-tested on a GPU-less box.
//! - [`PagedKvCache`] — the **device storage**: the K and V f16 slabs (`[layers, num_blocks,
//!   block_size, heads, head_dim]`) plus the host→device upload of the block table + context lengths
//!   the [paged-attention kernel](crate::paged_attention) reads. Built only with a live `Gpu`.
//!
//! ### Why f16 storage
//! The cache is the dominant device footprint at serving time; storing it f16 (not f32) halves it,
//! matches the tensor-core projection dtype, and is the precision the decode-attention kernel widens
//! from. [`PagedKvCache::footprint_bytes`] reports the slab size against the 6 GB budget.
//!
//! ### The first-law property this layout guarantees
//! The block table changes only *where* a token's K/V is fetched from, never the value or the order it
//! is accumulated. So a paged read is **bit-for-bit identical** to a contiguous read of the same
//! logical sequence — paging is numerically invisible. The [`crate::paged_attention`] gate proves it
//! by running the same logical sequence under two different physical block layouts and asserting the
//! output matches to the bit (the decode analogue of int8 split-K / transpose bit-exactness).

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaSlice, CudaStream, DriverError};

/// Default tokens per block. 16 is vLLM's default — small enough to keep internal fragmentation (the
/// half-empty last block) low, large enough that the block table stays short and the per-block loop in
/// the attention kernel amortizes its bookkeeping.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// The allocator ran out of physical blocks (the device cache is full). The scheduler reacts by
/// admitting fewer sequences / preempting, exactly as a real serving engine does under memory pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutOfBlocks;

impl std::fmt::Display for OutOfBlocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("paged KV-cache out of physical blocks")
    }
}
impl std::error::Error for OutOfBlocks {}

/// The static geometry of a paged KV-cache, shared by [`BlockManager`], [`PagedKvCache`], and the
/// attention kernel so they agree on every stride. All counts are in elements/tokens, not bytes.
#[derive(Debug, Clone, Copy)]
pub struct KvConfig {
    /// Transformer layers (each layer has its own K and V planes in the slab).
    pub layers: usize,
    /// KV heads (== query heads here; GQA/MQA would make this smaller).
    pub heads: usize,
    /// Per-head dimension. `heads * head_dim == D`.
    pub head_dim: usize,
    /// Tokens per physical block.
    pub block_size: usize,
    /// Physical blocks in the pool (the cache capacity = `num_blocks * block_size` tokens, per layer).
    pub num_blocks: usize,
    /// Decode batch slots (the fixed `Bcap` the graph captures at).
    pub num_slots: usize,
    /// Block-table columns: the most logical blocks any one sequence can hold
    /// (`ceil(max_seq_len / block_size)`). Fixes the padded table width the kernel reads.
    pub max_blocks_per_seq: usize,
}

impl KvConfig {
    /// f16 elements in one layer's K (or V) plane: `num_blocks * block_size * heads * head_dim`.
    #[inline]
    pub fn layer_plane_elems(&self) -> usize {
        self.num_blocks * self.block_size * self.heads * self.head_dim
    }

    /// f16 elements in the whole K (or V) slab across all layers.
    #[inline]
    pub fn slab_elems(&self) -> usize {
        self.layers * self.layer_plane_elems()
    }

    /// Flat element offset of `K[layer][phys_block][tok][head][dh]` (row-major), the single indexing
    /// rule the append and attention kernels both compute. `tok` is the in-block token (`0..block_size`).
    #[inline]
    pub fn elem_offset(&self, layer: usize, phys_block: u32, tok: usize, head: usize, dh: usize) -> usize {
        (((layer * self.num_blocks + phys_block as usize) * self.block_size + tok) * self.heads + head)
            * self.head_dim
            + dh
    }

    /// f32 scale entries in one int8 K (or V) **scale slab**: one per `(token, head)` =
    /// `layers * num_blocks * block_size * heads` — a factor `head_dim` smaller than the value slab.
    #[inline]
    pub fn scale_slab_elems(&self) -> usize {
        self.layers * self.num_blocks * self.block_size * self.heads
    }

    /// Flat index of `scale[layer][phys_block][tok][head]` — the per-(token, head) dequant scale the
    /// [int8 attention kernel](crate::paged_attention::paged_attn_decode_int8_ptx) reads. Equals
    /// `elem_offset(..., dh=0) / head_dim`.
    #[inline]
    pub fn scale_offset(&self, layer: usize, phys_block: u32, tok: usize, head: usize) -> usize {
        ((layer * self.num_blocks + phys_block as usize) * self.block_size + tok) * self.heads + head
    }

    /// Bytes for the whole K **and** V cache at `elem_size` bytes/element (4 = f32, 2 = f16): the
    /// dense-storage footprint against the 6 GB budget.
    #[inline]
    pub fn kv_bytes(&self, elem_size: usize) -> usize {
        2 * self.slab_elems() * elem_size
    }

    /// Bytes for the **int8** K and V cache: 1 byte/value plus the two per-(token, head) f32 scale slabs.
    #[inline]
    pub fn kv_bytes_int8(&self) -> usize {
        2 * self.slab_elems() + 2 * self.scale_slab_elems() * 4
    }

    /// Geometry for a serving cache in which **every one** of `num_slots` sequences can reach
    /// `max_ctx` tokens simultaneously (the worst case a scheduler must plan for): the block-table
    /// width is `ceil(max_ctx / block_size)` and the pool holds exactly `num_slots` such sequences.
    /// Size the result against the device budget with [`assert_kv_budget`](Self::assert_kv_budget).
    pub fn for_serving(
        layers: usize,
        heads: usize,
        head_dim: usize,
        block_size: usize,
        num_slots: usize,
        max_ctx: usize,
    ) -> KvConfig {
        let max_blocks_per_seq = max_ctx.div_ceil(block_size).max(1);
        KvConfig {
            layers,
            heads,
            head_dim,
            block_size,
            num_blocks: num_slots * max_blocks_per_seq,
            num_slots,
            max_blocks_per_seq,
        }
    }

    /// Assert the K+V cache at `elem_size` bytes/element ([`kv_bytes`](Self::kv_bytes)) fits inside
    /// `budget_bytes` of device memory, returning the bytes needed. The Bcap-scaling lever's guard:
    /// growing `num_slots` (or `max_ctx` via `num_blocks`) must stay inside the 6 GB part's budget
    /// *before* the slabs are allocated, not fail as a mid-run `CUDA_ERROR_OUT_OF_MEMORY`.
    pub fn assert_kv_budget(&self, elem_size: usize, budget_bytes: usize) -> usize {
        let need = self.kv_bytes(elem_size);
        assert!(
            need <= budget_bytes,
            "KV cache needs {need} B ({:.2} GiB) for {} slots x {} blocks/seq x {} layers at {elem_size} B/elem \
             — exceeds the {budget_bytes} B ({:.2} GiB) device budget",
            need as f64 / (1u64 << 30) as f64,
            self.num_slots,
            self.max_blocks_per_seq,
            self.layers,
            budget_bytes as f64 / (1u64 << 30) as f64,
        );
        need
    }

    /// Largest per-sequence context (tokens, rounded down to whole blocks) for which
    /// `for_serving(..)`'s cache still fits `budget_bytes` at `elem_size` bytes/element — the
    /// Bcap↔context trade-off table in one call. Returns 0 if even one block per slot is over budget.
    pub fn max_ctx_within_budget(
        layers: usize,
        heads: usize,
        head_dim: usize,
        block_size: usize,
        num_slots: usize,
        elem_size: usize,
        budget_bytes: usize,
    ) -> usize {
        // Bytes per cached token position (K and V, all layers).
        let per_tok = 2 * layers * heads * head_dim * elem_size;
        let per_slot_blocks = budget_bytes / (per_tok * block_size * num_slots);
        per_slot_blocks * block_size
    }
}

/// The **host-side** block allocator + per-slot block tables. No device handle — this is pure policy,
/// so the whole free-list/append/fragmentation surface is unit-tested with no GPU.
///
/// A *slot* is one of the `num_slots` decode batch positions. Each slot owns a growable list of
/// physical block ids (its block table) and a context length (tokens cached). The free list is a LIFO
/// stack of physical block ids; its order is what makes two valid layouts *physically different* (the
/// bit-exact-across-layouts gate exploits this).
#[derive(Debug, Clone)]
pub struct BlockManager {
    block_size: usize,
    num_blocks: usize,
    max_blocks_per_seq: usize,
    /// Free physical block ids (LIFO).
    free: Vec<u32>,
    /// Per slot: logical block index → physical block id.
    tables: Vec<Vec<u32>>,
    /// Per slot: tokens currently cached.
    ctx_len: Vec<usize>,
    /// Monotone counter bumped whenever any slot's block *table* changes (block pushed or freed) —
    /// context lengths alone don't move it. A device block-table upload is stale iff this differs
    /// from the epoch it was taken at, so steady-state decode steps (in-block appends) skip the
    /// `num_slots * max_blocks_per_seq` table re-upload entirely.
    layout_epoch: u64,
}

impl BlockManager {
    /// New manager owning `num_blocks` physical blocks of `block_size` tokens, for `num_slots` batch
    /// slots, each able to grow to `max_blocks_per_seq` logical blocks. The free list starts full and
    /// **ascending-popped** (block 0 first) so a fresh, un-fragmented run lays sequences out contiguously.
    pub fn new(num_blocks: usize, block_size: usize, num_slots: usize, max_blocks_per_seq: usize) -> Self {
        assert!(block_size > 0 && num_blocks > 0 && num_slots > 0 && max_blocks_per_seq > 0);
        // Push descending so `pop()` (LIFO) hands out 0,1,2,… ascending on a fresh pool.
        let free: Vec<u32> = (0..num_blocks as u32).rev().collect();
        Self {
            block_size,
            num_blocks,
            max_blocks_per_seq,
            free,
            tables: vec![Vec::new(); num_slots],
            ctx_len: vec![0; num_slots],
            layout_epoch: 0,
        }
    }

    /// The current block-table layout epoch (see the field docs): compare against the epoch of the
    /// last device upload to decide whether the flat block table must be re-uploaded.
    #[inline]
    pub fn layout_epoch(&self) -> u64 {
        self.layout_epoch
    }

    /// Tokens per block.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Block-table width (max logical blocks per sequence).
    #[inline]
    pub fn max_blocks_per_seq(&self) -> usize {
        self.max_blocks_per_seq
    }

    /// Physical blocks currently free.
    #[inline]
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// Physical blocks currently handed out.
    #[inline]
    pub fn used_blocks(&self) -> usize {
        self.num_blocks - self.free.len()
    }

    /// Tokens cached in `slot`.
    #[inline]
    pub fn context_len(&self, slot: usize) -> usize {
        self.ctx_len[slot]
    }

    /// This slot's block table (logical → physical), un-padded.
    #[inline]
    pub fn table(&self, slot: usize) -> &[u32] {
        &self.tables[slot]
    }

    /// Physical block id holding logical block `logical` of `slot`.
    #[inline]
    pub fn physical(&self, slot: usize, logical: usize) -> u32 {
        self.tables[slot][logical]
    }

    /// Logical blocks a sequence of `tokens` needs (`ceil(tokens / block_size)`).
    #[inline]
    pub fn blocks_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_size)
    }

    /// Could `tokens` *more* tokens be appended to `slot` right now (free blocks + table width)?
    pub fn can_grow(&self, slot: usize, tokens: usize) -> bool {
        let have = self.tables[slot].len() * self.block_size;
        let need_total = self.ctx_len[slot] + tokens;
        if need_total > have {
            let new_blocks = self.blocks_for(need_total) - self.tables[slot].len();
            if new_blocks > self.free.len() {
                return false;
            }
            if self.tables[slot].len() + new_blocks > self.max_blocks_per_seq {
                return false;
            }
        }
        true
    }

    /// Pop one physical block onto `slot`'s table. Internal; callers use [`append`](Self::append) /
    /// [`reserve`](Self::reserve).
    fn push_block(&mut self, slot: usize) -> Result<u32, OutOfBlocks> {
        if self.tables[slot].len() >= self.max_blocks_per_seq {
            return Err(OutOfBlocks);
        }
        let b = self.free.pop().ok_or(OutOfBlocks)?;
        self.tables[slot].push(b);
        self.layout_epoch += 1;
        Ok(b)
    }

    /// Append **one** token to `slot`, allocating a new physical block iff the current last block is
    /// full. Returns `(physical_block, in_block_offset)` — exactly where the [append
    /// kernel](crate::paged_attention) must write this token's K/V. Increments the context length.
    pub fn append(&mut self, slot: usize) -> Result<(u32, usize), OutOfBlocks> {
        let pos = self.ctx_len[slot];
        let logical = pos / self.block_size;
        let off = pos % self.block_size;
        if logical >= self.tables[slot].len() {
            self.push_block(slot)?;
        }
        let phys = self.tables[slot][logical];
        self.ctx_len[slot] = pos + 1;
        Ok((phys, off))
    }

    /// Reserve blocks so `slot` can hold `tokens` *additional* tokens, and advance its context length
    /// by `tokens` (the prefill bulk-allocate: the kernel then fills all `tokens` positions). Returns
    /// the starting position the new tokens occupy.
    pub fn reserve(&mut self, slot: usize, tokens: usize) -> Result<usize, OutOfBlocks> {
        if !self.can_grow(slot, tokens) {
            return Err(OutOfBlocks);
        }
        let start = self.ctx_len[slot];
        let need_total = start + tokens;
        while self.tables[slot].len() * self.block_size < need_total {
            self.push_block(slot)?;
        }
        self.ctx_len[slot] = need_total;
        Ok(start)
    }

    /// Free every block of `slot` back to the pool and reset its context length to 0. The returned
    /// blocks go back on the LIFO free list, so a later run reuses them in a *different* order than a
    /// fresh pool would — which is precisely the physical re-layout the bit-exact gate wants.
    pub fn free(&mut self, slot: usize) {
        let blocks = std::mem::take(&mut self.tables[slot]);
        if !blocks.is_empty() {
            self.layout_epoch += 1;
        }
        for b in blocks {
            self.free.push(b);
        }
        self.ctx_len[slot] = 0;
    }

    /// Map `(slot, in-sequence position)` → `(physical_block, in_block_offset)` for a position already
    /// cached. The address rule the attention kernel reproduces per context position on device.
    #[inline]
    pub fn locate(&self, slot: usize, pos: usize) -> (u32, usize) {
        let logical = pos / self.block_size;
        let off = pos % self.block_size;
        (self.tables[slot][logical], off)
    }

    /// Flatten all slots' block tables into one `[num_slots * max_blocks_per_seq]` row-major `u32`
    /// array (rows padded with 0) for a single host→device upload the kernel indexes by
    /// `slot * max_blocks_per_seq + logical`.
    pub fn flat_block_table(&self) -> Vec<u32> {
        let w = self.max_blocks_per_seq;
        let mut out = vec![0u32; self.tables.len() * w];
        for (s, tbl) in self.tables.iter().enumerate() {
            out[s * w..s * w + tbl.len()].copy_from_slice(tbl);
        }
        out
    }

    /// Per-slot context lengths as `u32` (kernel reads one per CTA row).
    pub fn ctx_lens(&self) -> Vec<u32> {
        self.ctx_len.iter().map(|&l| l as u32).collect()
    }

    /// Number of batch slots.
    #[inline]
    pub fn num_slots(&self) -> usize {
        self.tables.len()
    }
}

/// The **device** paged KV-cache: one f16 K slab + one f16 V slab laid out
/// `[layers, num_blocks, block_size, heads, head_dim]`, plus the [`BlockManager`] policy and reusable
/// device buffers for the block table / context lengths the attention kernel reads. Built only with a
/// live `Gpu` (the host policy in [`BlockManager`] is what the unit tests cover).
/// Which dtype the device cache stores the K/V values in. `F16` is the default and the bit-exact
/// path every existing gate runs; `Int8` (per-(token, head) scaled — the
/// [`crate::paged_attention::quantize_kv_int8`] scheme) halves the footprint again, lossy and
/// tolerance-gated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvDtype {
    #[default]
    F16,
    Int8,
}

/// The device K/V slabs, by storage dtype. `Int8` adds the two per-(token, head) f32 **scale slabs**
/// ([`KvConfig::scale_slab_elems`] each) the int8 append/attention kernels write/read. The value
/// slabs have the identical `[layers, num_blocks, block_size, heads, head_dim]` layout in both arms.
#[cfg(feature = "gpu")]
pub enum KvStorage {
    F16 { k: CudaSlice<half::f16>, v: CudaSlice<half::f16> },
    Int8 { k: CudaSlice<i8>, v: CudaSlice<i8>, ksc: CudaSlice<f32>, vsc: CudaSlice<f32> },
}

#[cfg(feature = "gpu")]
impl KvStorage {
    /// The storage dtype of this slab set.
    pub fn dtype(&self) -> KvDtype {
        match self {
            KvStorage::F16 { .. } => KvDtype::F16,
            KvStorage::Int8 { .. } => KvDtype::Int8,
        }
    }
}

#[cfg(feature = "gpu")]
pub struct PagedKvCache {
    cfg: KvConfig,
    mgr: BlockManager,
    stream: Arc<CudaStream>,
    /// K and V slabs (+ int8 scale slabs), `[layers, num_blocks, block_size, heads, head_dim]`.
    storage: KvStorage,
    /// Reusable device upload of the flattened block table (`num_slots * max_blocks_per_seq` u32).
    block_table_d: CudaSlice<u32>,
    /// Reusable device upload of per-slot context lengths (`num_slots` u32).
    ctx_len_d: CudaSlice<u32>,
}

#[cfg(feature = "gpu")]
impl PagedKvCache {
    /// Allocate the f16 K/V slabs (zeroed) and the block-table / context-length upload buffers on
    /// `stream`. The slabs are `footprint_bytes()` of device memory — sized by the caller against the
    /// 6 GB budget. This is the default (bit-exact) storage; int8 opts in via
    /// [`new_with_dtype`](Self::new_with_dtype).
    pub fn new(stream: Arc<CudaStream>, cfg: KvConfig) -> Result<Self, DriverError> {
        Self::new_with_dtype(stream, cfg, KvDtype::F16)
    }

    /// As [`new`](Self::new) with an explicit storage dtype. `Int8` additionally allocates the two
    /// per-(token, head) f32 scale slabs (zeroed).
    pub fn new_with_dtype(stream: Arc<CudaStream>, cfg: KvConfig, dtype: KvDtype) -> Result<Self, DriverError> {
        let mgr = BlockManager::new(cfg.num_blocks, cfg.block_size, cfg.num_slots, cfg.max_blocks_per_seq);
        let storage = match dtype {
            KvDtype::F16 => KvStorage::F16 {
                k: stream.alloc_zeros::<half::f16>(cfg.slab_elems())?,
                v: stream.alloc_zeros::<half::f16>(cfg.slab_elems())?,
            },
            KvDtype::Int8 => KvStorage::Int8 {
                k: stream.alloc_zeros::<i8>(cfg.slab_elems())?,
                v: stream.alloc_zeros::<i8>(cfg.slab_elems())?,
                ksc: stream.alloc_zeros::<f32>(cfg.scale_slab_elems())?,
                vsc: stream.alloc_zeros::<f32>(cfg.scale_slab_elems())?,
            },
        };
        let block_table_d = stream.alloc_zeros::<u32>(cfg.num_slots * cfg.max_blocks_per_seq)?;
        let ctx_len_d = stream.alloc_zeros::<u32>(cfg.num_slots)?;
        Ok(Self { cfg, mgr, stream, storage, block_table_d, ctx_len_d })
    }

    /// Total device bytes the K + V storage occupies across all layers (int8 includes its scale
    /// slabs) — the serving footprint against the 6 GB budget.
    pub fn footprint_bytes(&self) -> usize {
        match self.storage {
            KvStorage::F16 { .. } => self.cfg.kv_bytes(2),
            KvStorage::Int8 { .. } => self.cfg.kv_bytes_int8(),
        }
    }

    /// The storage dtype.
    pub fn dtype(&self) -> KvDtype {
        self.storage.dtype()
    }

    /// The K/V storage (all dtypes).
    pub fn storage(&self) -> &KvStorage {
        &self.storage
    }

    /// Mutable K/V storage — the split-borrow seam the decode step hands to the append/attention
    /// launchers (dtype-dispatched in `DecodeLayer::forward_step_on`).
    pub fn storage_mut(&mut self) -> &mut KvStorage {
        &mut self.storage
    }

    /// The static geometry.
    pub fn config(&self) -> &KvConfig {
        &self.cfg
    }

    /// The host allocator/policy (mutable — the scheduler drives `append`/`reserve`/`free`).
    pub fn manager(&mut self) -> &mut BlockManager {
        &mut self.mgr
    }

    /// Immutable view of the host allocator.
    pub fn manager_ref(&self) -> &BlockManager {
        &self.mgr
    }

    /// Device K slab `[layers, num_blocks, block_size, heads, head_dim]` (f16 storage only — the
    /// int8 arm is reached through [`storage`](Self::storage); a wrong-dtype access is a logic bug,
    /// so it panics rather than corrupting).
    pub fn k(&self) -> &CudaSlice<half::f16> {
        match &self.storage {
            KvStorage::F16 { k, .. } => k,
            KvStorage::Int8 { .. } => panic!("f16 slab accessor on an int8 cache — use storage()"),
        }
    }
    /// Device V slab (same layout; f16 storage only).
    pub fn v(&self) -> &CudaSlice<half::f16> {
        match &self.storage {
            KvStorage::F16 { v, .. } => v,
            KvStorage::Int8 { .. } => panic!("f16 slab accessor on an int8 cache — use storage()"),
        }
    }
    /// Mutable device K slab (for the append kernel; f16 storage only).
    pub fn k_mut(&mut self) -> &mut CudaSlice<half::f16> {
        match &mut self.storage {
            KvStorage::F16 { k, .. } => k,
            KvStorage::Int8 { .. } => panic!("f16 slab accessor on an int8 cache — use storage_mut()"),
        }
    }
    /// Mutable device V slab (f16 storage only).
    pub fn v_mut(&mut self) -> &mut CudaSlice<half::f16> {
        match &mut self.storage {
            KvStorage::F16 { v, .. } => v,
            KvStorage::Int8 { .. } => panic!("f16 slab accessor on an int8 cache — use storage_mut()"),
        }
    }

    /// Both f16 slabs mutably at once (`&mut K`, `&mut V`) via a split borrow — the decode step needs
    /// to pass both to the append kernel in one launch, which two separate `k_mut`/`v_mut` calls
    /// (each a full `&mut self`) cannot express. F16 storage only (tests/population helpers);
    /// dtype-generic code goes through [`storage_mut`](Self::storage_mut).
    pub fn slabs_mut(&mut self) -> (&mut CudaSlice<half::f16>, &mut CudaSlice<half::f16>) {
        match &mut self.storage {
            KvStorage::F16 { k, v } => (k, v),
            KvStorage::Int8 { .. } => panic!("f16 slab accessor on an int8 cache — use storage_mut()"),
        }
    }

    /// Push the current host block table + context lengths to the device buffers the attention kernel
    /// reads, returning `(&block_table_d, &ctx_len_d)`. Call after any `append`/`reserve`/`free` that
    /// changed the layout, before launching paged attention.
    pub fn sync_metadata(&mut self) -> Result<(&CudaSlice<u32>, &CudaSlice<u32>), DriverError> {
        let table = self.mgr.flat_block_table();
        let lens = self.mgr.ctx_lens();
        self.stream.memcpy_htod(&table, &mut self.block_table_d)?;
        self.stream.memcpy_htod(&lens, &mut self.ctx_len_d)?;
        Ok((&self.block_table_d, &self.ctx_len_d))
    }

    /// The device block-table buffer (valid after [`sync_metadata`](Self::sync_metadata)).
    pub fn block_table_device(&self) -> &CudaSlice<u32> {
        &self.block_table_d
    }
    /// The device context-length buffer (valid after [`sync_metadata`](Self::sync_metadata)).
    pub fn ctx_len_device(&self) -> &CudaSlice<u32> {
        &self.ctx_len_d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **int8 KV footprint win (the 6 GB lever).** Pure geometry: int8 storage is ~half of f16 and ~a
    /// quarter of f32 — the per-(token, head) f32 scale slab is `head_dim×` smaller than the value slab,
    /// so it barely dents the win. No device needed.
    #[test]
    fn int8_kv_footprint_shrink() {
        // A Llama-7B-ish KV geometry: 32 layers, 8 KV heads × 128, 4096 blocks of 16.
        let cfg = KvConfig {
            layers: 32,
            heads: 8,
            head_dim: 128,
            block_size: 16,
            num_blocks: 4096,
            num_slots: 64,
            max_blocks_per_seq: 256,
        };
        let (f32b, f16b, i8b) = (cfg.kv_bytes(4), cfg.kv_bytes(2), cfg.kv_bytes_int8());
        assert!(i8b < f16b && f16b < f32b, "int8 < f16 < f32");
        let vs_f16 = f16b as f64 / i8b as f64;
        let vs_f32 = f32b as f64 / i8b as f64;
        // head_dim=128 ⇒ int8 = 1 + 4/128 bytes/value ⇒ ~1.94× vs f16, ~3.88× vs f32.
        assert!(vs_f16 > 1.9 && vs_f16 < 2.0, "int8 ~half of f16 (got {vs_f16:.3}x)");
        assert!(vs_f32 > 3.8 && vs_f32 < 4.0, "int8 ~quarter of f32 (got {vs_f32:.3}x)");
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "KV footprint (32L, 8h×128, 4096×16 blocks): f32 {:.2} GiB | f16 {:.2} GiB | int8 {:.2} GiB → {:.2}x vs f16, {:.2}x vs f32",
            gib(f32b), gib(f16b), gib(i8b), vs_f16, vs_f32
        );
    }

    /// **Bcap-scaling budget table (the 6 GB lever, pure geometry).** For the goodput-bench layer
    /// geometry (12 layers, 8 heads × 64, 16-token blocks, f16 KV), every swept Bcap must fit its
    /// worst-case simultaneous context inside a 4 GiB KV budget with room for the bench's 96-token
    /// contexts, and the exact `for_serving` config at that context must round-trip its own budget
    /// assert. No device needed.
    #[test]
    fn serving_bcap_budget_table() {
        let budget = 4usize << 30; // KV slice of the 6 GB part (weights/pool/desktop take the rest)
        let (layers, heads, hd, bsz) = (12, 8, 64, 16);
        for &bcap in &[64usize, 128, 256] {
            let max_ctx = KvConfig::max_ctx_within_budget(layers, heads, hd, bsz, bcap, 2, budget);
            assert!(max_ctx >= 96, "Bcap={bcap} must fit the bench's 96-token contexts (got {max_ctx})");
            let cfg = KvConfig::for_serving(layers, heads, hd, bsz, bcap, max_ctx);
            let bytes = cfg.assert_kv_budget(2, budget);
            eprintln!(
                "Bcap={bcap:3}: max simultaneous ctx {max_ctx:5} tok/seq → KV {:.2} GiB of {:.1} GiB budget",
                bytes as f64 / (1u64 << 30) as f64,
                budget as f64 / (1u64 << 30) as f64
            );
        }
    }

    /// The budget assert must fire (panic, pre-allocation) when the requested geometry exceeds the
    /// device budget — the guard that turns a mid-run `CUDA_ERROR_OUT_OF_MEMORY` into a clear error.
    #[test]
    #[should_panic(expected = "exceeds")]
    fn over_budget_kv_config_is_rejected() {
        // One block over: for_serving at max_ctx+block_size cannot fit the same budget it saturates.
        let budget = 4usize << 30;
        let (layers, heads, hd, bsz) = (12, 8, 64, 16);
        let max_ctx = KvConfig::max_ctx_within_budget(layers, heads, hd, bsz, 256, 2, budget);
        let cfg = KvConfig::for_serving(layers, heads, hd, bsz, 256, max_ctx + bsz);
        cfg.assert_kv_budget(2, budget);
    }

    // ---- pure host allocator: runs on a GPU-less box (the policy is device-independent) ----

    #[test]
    fn append_allocates_a_new_block_only_on_block_boundary() {
        let mut m = BlockManager::new(8, 4, 2, 8); // 8 blocks of 4 tokens, 2 slots
        assert_eq!(m.free_blocks(), 8);
        // First token of slot 0: allocates block, offset 0. Fresh pool pops ascending → block 0.
        assert_eq!(m.append(0).unwrap(), (0, 0));
        assert_eq!(m.free_blocks(), 7);
        // Tokens 1..3 fill the same block, no new allocation.
        assert_eq!(m.append(0).unwrap(), (0, 1));
        assert_eq!(m.append(0).unwrap(), (0, 2));
        assert_eq!(m.append(0).unwrap(), (0, 3));
        assert_eq!(m.free_blocks(), 7);
        // Token 4 crosses the boundary → second block (block 1), offset 0.
        assert_eq!(m.append(0).unwrap(), (1, 0));
        assert_eq!(m.free_blocks(), 6);
        assert_eq!(m.context_len(0), 5);
        assert_eq!(m.table(0), &[0, 1]);
    }

    #[test]
    fn reserve_bulk_allocates_prefill() {
        let mut m = BlockManager::new(8, 4, 2, 8);
        let start = m.reserve(0, 10).unwrap(); // 10 tokens → ceil(10/4)=3 blocks
        assert_eq!(start, 0);
        assert_eq!(m.context_len(0), 10);
        assert_eq!(m.table(0).len(), 3);
        assert_eq!(m.free_blocks(), 5);
        // A subsequent decode append continues from position 10 (block index 2, offset 2).
        assert_eq!(m.append(0).unwrap(), (m.physical(0, 2), 2));
        assert_eq!(m.context_len(0), 11);
    }

    #[test]
    fn free_returns_blocks_and_relayouts_on_reuse() {
        let mut m = BlockManager::new(4, 4, 2, 8);
        m.reserve(0, 8).unwrap(); // blocks [0,1]
        assert_eq!(m.table(0), &[0, 1]);
        assert_eq!(m.free_blocks(), 2); // [3,2] left on the stack (top=2)
        m.reserve(1, 4).unwrap(); // block [2]
        assert_eq!(m.table(1), &[2]);
        // Free slot 0 → blocks 0,1 pushed back (stack now top=1).
        m.free(0);
        assert_eq!(m.free_blocks(), 3);
        assert_eq!(m.context_len(0), 0);
        // Re-reserving slot 0 now pops 1 then 0 → a *different physical layout* of the same logical
        // sequence (the property the bit-exact-across-layouts attention gate exploits).
        m.reserve(0, 8).unwrap();
        assert_eq!(m.table(0), &[1, 0]);
    }

    #[test]
    fn out_of_blocks_is_reported_not_panicked() {
        let mut m = BlockManager::new(2, 4, 4, 8); // only 2 blocks
        assert!(m.reserve(0, 8).is_ok()); // uses both
        assert_eq!(m.free_blocks(), 0);
        assert_eq!(m.append(1), Err(OutOfBlocks));
        assert_eq!(m.reserve(2, 1), Err(OutOfBlocks));
        assert!(!m.can_grow(1, 1));
    }

    #[test]
    fn table_width_cap_is_enforced() {
        let mut m = BlockManager::new(16, 4, 1, 2); // max 2 logical blocks/seq
        assert!(m.reserve(0, 8).is_ok()); // exactly 2 blocks
        assert_eq!(m.append(0), Err(OutOfBlocks)); // 9th token needs a 3rd block > cap
        assert!(!m.can_grow(0, 1));
    }

    #[test]
    fn flat_block_table_is_padded_row_major() {
        let mut m = BlockManager::new(8, 4, 3, 4);
        m.reserve(0, 5).unwrap(); // 2 blocks → [0,1]
        m.reserve(2, 1).unwrap(); // 1 block  → [2]
        let flat = m.flat_block_table();
        assert_eq!(flat.len(), 3 * 4);
        assert_eq!(&flat[0..4], &[0, 1, 0, 0]); // slot 0: two blocks then pad
        assert_eq!(&flat[4..8], &[0, 0, 0, 0]); // slot 1: empty
        assert_eq!(&flat[8..12], &[2, 0, 0, 0]); // slot 2: one block then pad
        assert_eq!(m.ctx_lens(), vec![5, 0, 1]);
    }

    #[test]
    fn locate_matches_append_addresses() {
        let mut m = BlockManager::new(8, 4, 1, 8);
        let mut appended = Vec::new();
        for _ in 0..10 {
            appended.push(m.append(0).unwrap());
        }
        // locate(pos) must reproduce exactly the (phys, off) append handed out for that position.
        for (pos, &(phys, off)) in appended.iter().enumerate() {
            assert_eq!(m.locate(0, pos), (phys, off), "pos {pos}");
        }
    }

    #[test]
    fn kv_config_offset_is_row_major_and_in_bounds() {
        let cfg = KvConfig {
            layers: 2,
            heads: 3,
            head_dim: 4,
            block_size: 5,
            num_blocks: 6,
            num_slots: 2,
            max_blocks_per_seq: 4,
        };
        // Offset 0 is the first element; the last addressable element is slab_elems-1.
        assert_eq!(cfg.elem_offset(0, 0, 0, 0, 0), 0);
        let last = cfg.elem_offset(cfg.layers - 1, (cfg.num_blocks - 1) as u32, cfg.block_size - 1, cfg.heads - 1, cfg.head_dim - 1);
        assert_eq!(last, cfg.slab_elems() - 1);
        // Adjacent dh elements are contiguous (row-major innermost).
        assert_eq!(cfg.elem_offset(0, 0, 0, 0, 1) - cfg.elem_offset(0, 0, 0, 0, 0), 1);
        // Adjacent head steps by head_dim.
        assert_eq!(cfg.elem_offset(0, 0, 0, 1, 0) - cfg.elem_offset(0, 0, 0, 0, 0), cfg.head_dim);
        // Adjacent layer steps by one full plane.
        assert_eq!(cfg.elem_offset(1, 0, 0, 0, 0) - cfg.elem_offset(0, 0, 0, 0, 0), cfg.layer_plane_elems());
    }
}
