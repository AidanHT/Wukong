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
        }
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
#[cfg(feature = "gpu")]
pub struct PagedKvCache {
    cfg: KvConfig,
    mgr: BlockManager,
    stream: Arc<CudaStream>,
    /// K and V slabs, f16, `[layers, num_blocks, block_size, heads, head_dim]`.
    k: CudaSlice<half::f16>,
    v: CudaSlice<half::f16>,
    /// Reusable device upload of the flattened block table (`num_slots * max_blocks_per_seq` u32).
    block_table_d: CudaSlice<u32>,
    /// Reusable device upload of per-slot context lengths (`num_slots` u32).
    ctx_len_d: CudaSlice<u32>,
}

#[cfg(feature = "gpu")]
impl PagedKvCache {
    /// Allocate the K/V slabs (zeroed) and the block-table / context-length upload buffers on
    /// `stream`. The slabs are `footprint_bytes()` of device memory — sized by the caller against the
    /// 6 GB budget.
    pub fn new(stream: Arc<CudaStream>, cfg: KvConfig) -> Result<Self, DriverError> {
        let mgr = BlockManager::new(cfg.num_blocks, cfg.block_size, cfg.num_slots, cfg.max_blocks_per_seq);
        let k = stream.alloc_zeros::<half::f16>(cfg.slab_elems())?;
        let v = stream.alloc_zeros::<half::f16>(cfg.slab_elems())?;
        let block_table_d = stream.alloc_zeros::<u32>(cfg.num_slots * cfg.max_blocks_per_seq)?;
        let ctx_len_d = stream.alloc_zeros::<u32>(cfg.num_slots)?;
        Ok(Self { cfg, mgr, stream, k, v, block_table_d, ctx_len_d })
    }

    /// Total device bytes the two f16 slabs occupy (K + V across all layers) — the serving footprint.
    pub fn footprint_bytes(&self) -> usize {
        2 * self.cfg.slab_elems() * std::mem::size_of::<half::f16>()
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

    /// Device K slab `[layers, num_blocks, block_size, heads, head_dim]` (f16).
    pub fn k(&self) -> &CudaSlice<half::f16> {
        &self.k
    }
    /// Device V slab (same layout).
    pub fn v(&self) -> &CudaSlice<half::f16> {
        &self.v
    }
    /// Mutable device K slab (for the append kernel).
    pub fn k_mut(&mut self) -> &mut CudaSlice<half::f16> {
        &mut self.k
    }
    /// Mutable device V slab.
    pub fn v_mut(&mut self) -> &mut CudaSlice<half::f16> {
        &mut self.v
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
