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
//! - [`PagedKvCache`] — the **device storage**: the K and V slabs (`[layers, num_blocks,
//!   block_size, heads, head_dim]`) plus the host→device upload of the block table + context lengths
//!   the [paged-attention kernel](crate::paged_attention) reads. Built only with a live `Gpu`.
//!
//! ### Why f16 storage
//! The cache is the dominant device footprint at serving time; storing it f16 (not f32) halves it,
//! matches the tensor-core projection dtype, and is the precision the decode-attention kernel widens
//! from. [`PagedKvCache::footprint_bytes`] reports the slab size against the 6 GB budget. `KvDtype::Int8`
//! storage halves it again — i8 values plus two per-(token, head) f32 scale slabs — at the cost of being
//! lossy (tolerance-gated, not bit-exact vs f32 K/V); f16 stays the default.
//!
//! ### The first-law property this layout guarantees
//! The block table changes only *where* a token's K/V is fetched from, never the value or the order it
//! is accumulated. So a paged read is **bit-for-bit identical** to a contiguous read of the same
//! logical sequence — paging is numerically invisible. The [`crate::paged_attention`] gate proves it
//! by running the same logical sequence under two different physical block layouts and asserting the
//! output matches to the bit (the decode analogue of int8 split-K / transpose bit-exactness).
//!
//! ### Grouped-query attention (GQA/MQA)
//! Every Llama-class model attends with **more query heads than KV heads** — Llama-3-8B is 32 q over
//! 8 kv, Llama-3-70B 64 over 8, Mistral-7B 32 over 8 — and the whole point of that architecture is
//! that the *cache* shrinks by the group size `g = q_heads / kv_heads`. This module therefore keeps
//! the two counts strictly apart:
//!
//! - **[`KvConfig::heads`] is the KV-head count**, and it is the only head count any slab offset here
//!   is keyed on ([`layer_plane_elems`](KvConfig::layer_plane_elems),
//!   [`elem_offset`](KvConfig::elem_offset), [`scale_offset`](KvConfig::scale_offset)). Sizing the
//!   cache off the *query* head count is exactly the `g`x over-allocation GQA exists to remove.
//! - **[`GqaConfig`] pairs that cache geometry with `q_heads`** and is the type the attention path
//!   takes. Its constructor is the one place `q_heads % kv_heads == 0` is enforced, and its fields are
//!   private so no caller can assemble a geometry that skipped the check.
//! - `q_heads == kv_heads` ([`GqaConfig::mha`]) is plain multi-head attention and is bit-for-bit the
//!   behaviour that shipped before GQA existed — the group size is 1 and `kv_head_of(h) == h`.
//!
//! **Paging invariance survives GQA**, and it is worth saying explicitly because the change *looks*
//! like it should break it: the bit-exactness property rests on the lane partition being a function of
//! the logical position index alone plus a fixed merge order (see [`crate::paged_attention`]). GQA
//! changes *which head's* K/V a lane reads and nothing else — not which lane a position lands in, not
//! the merge order. The `paged_gqa_*_invariant_to_block_layout` gates prove it rather than assume it.

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
///
/// This is the **cache** geometry, so every head count in it is a **KV**-head count. The query-head
/// count is not part of it — it does not size, stride or address anything here — and lives in
/// [`GqaConfig`], which pairs a `KvConfig` with `q_heads` and validates the grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvConfig {
    /// Transformer layers (each layer has its own K and V planes in the slab).
    pub layers: usize,
    /// **KV heads** — the head count the cache is sized and indexed by. Under GQA/MQA this is
    /// *smaller* than the query-head count by the group size `g` (Llama-3-8B: 8 here, 32 query
    /// heads), and that difference is the entire memory win: the slab shrinks by exactly `g`.
    /// Prefer the named [`kv_heads`](Self::kv_heads) accessor in new code, and reach for
    /// [`GqaConfig`] whenever the query-head count is also in play.
    pub heads: usize,
    /// Per-head dimension. `heads * head_dim` is the **KV** width of one token (`GqaConfig::kv_dim`),
    /// which equals the model's hidden size `D` only when `q_heads == kv_heads`.
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
    /// The **KV**-head count — the same number as the [`heads`](Self::heads) field, under the name
    /// that says which of the two head counts it is. Every offset helper below is keyed on it.
    #[inline]
    pub fn kv_heads(&self) -> usize {
        self.heads
    }

    /// Width in elements of one cached token's K (or V) vector: `kv_heads * head_dim`. Under GQA this
    /// is `g`x narrower than the model's hidden size, which is where the cache saving comes from.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    /// f16 elements in one layer's K (or V) plane: `num_blocks * block_size * kv_heads * head_dim`.
    #[inline]
    pub fn layer_plane_elems(&self) -> usize {
        self.num_blocks * self.block_size * self.heads * self.head_dim
    }

    /// f16 elements in the whole K (or V) slab across all layers.
    #[inline]
    pub fn slab_elems(&self) -> usize {
        self.layers * self.layer_plane_elems()
    }

    /// Flat element offset of `K[layer][phys_block][tok][kv_head][dh]` (row-major), the single indexing
    /// rule the append and attention kernels both compute. `tok` is the in-block token
    /// (`0..block_size`) and `head` is a **KV** head (`0..kv_heads`) — under GQA a query head `h`
    /// reaches its row through [`GqaConfig::kv_head_of`], never with `h` itself.
    #[inline]
    pub fn elem_offset(
        &self,
        layer: usize,
        phys_block: u32,
        tok: usize,
        head: usize,
        dh: usize,
    ) -> usize {
        (((layer * self.num_blocks + phys_block as usize) * self.block_size + tok) * self.heads
            + head)
            * self.head_dim
            + dh
    }

    /// f32 scale entries in one int8 K (or V) **scale slab**: one per `(token, kv_head)` =
    /// `layers * num_blocks * block_size * kv_heads` — a factor `head_dim` smaller than the value slab.
    #[inline]
    pub fn scale_slab_elems(&self) -> usize {
        self.layers * self.num_blocks * self.block_size * self.heads
    }

    /// Flat index of `scale[layer][phys_block][tok][kv_head]` — the per-(token, KV head) dequant scale the
    /// [int8 attention kernel](crate::paged_attention::paged_attn_decode_int8_ptx) reads. Equals
    /// `elem_offset(..., dh=0) / head_dim`.
    #[inline]
    pub fn scale_offset(&self, layer: usize, phys_block: u32, tok: usize, head: usize) -> usize {
        ((layer * self.num_blocks + phys_block as usize) * self.block_size + tok) * self.heads
            + head
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
    ///
    /// `heads` is the **KV**-head count. For a GQA model pass `kv_heads`, not `q_heads` — or, better,
    /// call [`GqaConfig::for_serving`], which takes both and cannot be given them the wrong way round.
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
    ///
    /// `heads` is the **KV**-head count, so a GQA model reaches exactly `g = q_heads / kv_heads` times
    /// the context of its MHA twin inside the same budget — the serving consequence of the shrink.
    ///
    /// **Precondition:** every geometry parameter (`layers`, `heads`, `head_dim`, `block_size`,
    /// `num_slots`, `elem_size`) must be non-zero — a zero anywhere makes the divisor zero. Asserted
    /// with a message naming the offending values, rather than surfacing a bare `attempt to divide by
    /// zero` from inside this function.
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
        assert!(
            per_tok > 0 && block_size > 0 && num_slots > 0,
            "max_ctx_within_budget: layers/heads/head_dim/elem_size/block_size/num_slots must all be non-zero \
             (got layers={layers} heads={heads} head_dim={head_dim} elem_size={elem_size} block_size={block_size} \
             num_slots={num_slots})"
        );
        let per_slot_blocks = budget_bytes / (per_tok * block_size * num_slots);
        per_slot_blocks * block_size
    }
}

/// The **attention head geometry** of a paged cache: `q_heads` query heads reading a [`KvConfig`]
/// whose `heads` is the `kv_heads` count, with `q_heads % kv_heads == 0`.
///
/// This is the type the paged attention path takes, and the *only* place the two head counts are
/// allowed to meet. Its fields are **private** and its constructors validate, so a `GqaConfig` that
/// exists is a grouping that passed the check — there is no struct-literal back door the way there is
/// for `KvConfig`. The two counts do genuinely different jobs and confusing them is silent:
///
/// | quantity | keyed on |
/// |---|---|
/// | the decode grid, the `(slot, head)` warp assignment, the Q row, the O row | `q_heads` |
/// | the K/V slabs, the int8 scale slabs, every `elem_offset` / `scale_offset`, the K/V projection width | `kv_heads` |
///
/// `q_heads == kv_heads` is multi-head attention: [`group_size`](Self::group_size) is 1,
/// [`kv_head_of`](Self::kv_head_of) is the identity, and every byte count matches what the cache
/// allocated before GQA existed.
///
/// ```
/// use wukong_codegen_gpu::paged_kv::GqaConfig;
/// // Llama-3-8B: 32 query heads over 8 KV heads, head_dim 128.
/// let g = GqaConfig::for_serving(32, 32, 8, 128, 16, 8, 4096);
/// assert_eq!(g.group_size(), 4);
/// assert_eq!(g.kv_head_of(0), 0);
/// assert_eq!(g.kv_head_of(7), 1); // query heads 4..8 all read KV head 1
/// // The cache is a quarter of what the same model would need without grouping.
/// assert_eq!(g.mha_equivalent_kv_bytes(2), 4 * g.kv().kv_bytes(2));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GqaConfig {
    /// The cache geometry. `kv.heads` is the **KV**-head count.
    kv: KvConfig,
    /// Query heads. A whole multiple of `kv.heads`.
    q_heads: usize,
}

impl GqaConfig {
    /// Pair a cache geometry with a query-head count, **validating the grouping loudly**. Panics
    /// unless `q_heads > 0`, `kv.heads > 0`, `kv.head_dim > 0` and `q_heads % kv.heads == 0` — an
    /// indivisible pair has no meaning (some KV head would serve a fractional number of query heads)
    /// and every downstream index would silently address the wrong row rather than fail.
    ///
    /// Note `q_heads >= kv_heads` needs no separate check: a positive `q_heads` divisible by
    /// `kv_heads` is already at least `kv_heads`.
    pub fn new(kv: KvConfig, q_heads: usize) -> Self {
        assert!(
            q_heads > 0 && kv.heads > 0 && kv.head_dim > 0,
            "GQA geometry: q_heads, kv_heads and head_dim must all be non-zero \
             (got q_heads={q_heads} kv_heads={} head_dim={})",
            kv.heads,
            kv.head_dim
        );
        assert!(
            q_heads.is_multiple_of(kv.heads),
            "GQA geometry: q_heads ({q_heads}) must be a whole multiple of kv_heads ({}) — every KV \
             head serves exactly q_heads/kv_heads query heads, so an indivisible pair has no grouping",
            kv.heads
        );
        Self { kv, q_heads }
    }

    /// Plain multi-head attention over `kv`: `q_heads == kv.heads`, group size 1. Bit-for-bit the
    /// pre-GQA behaviour, and what every `KvConfig`-taking entry point in this crate assumes.
    #[inline]
    pub fn mha(kv: KvConfig) -> Self {
        Self::new(kv, kv.heads)
    }

    /// Build a serving geometry directly from a model's head counts — the GQA-aware
    /// [`KvConfig::for_serving`], which cannot be handed `q_heads` where it wanted `kv_heads`.
    /// The cache is sized for `kv_heads`, so it is `g`x smaller than the query-head count suggests.
    #[allow(clippy::too_many_arguments)]
    pub fn for_serving(
        layers: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_slots: usize,
        max_ctx: usize,
    ) -> Self {
        Self::new(
            KvConfig::for_serving(layers, kv_heads, head_dim, block_size, num_slots, max_ctx),
            q_heads,
        )
    }

    /// The cache geometry (its `heads` is `kv_heads`).
    #[inline]
    pub fn kv(&self) -> &KvConfig {
        &self.kv
    }

    /// Query heads.
    #[inline]
    pub fn q_heads(&self) -> usize {
        self.q_heads
    }

    /// KV heads.
    #[inline]
    pub fn kv_heads(&self) -> usize {
        self.kv.heads
    }

    /// Per-head dimension (shared by Q and KV).
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.kv.head_dim
    }

    /// `g = q_heads / kv_heads` — how many query heads share each KV head. 1 is MHA; `kv_heads == 1`
    /// is MQA.
    #[inline]
    pub fn group_size(&self) -> usize {
        self.q_heads / self.kv.heads
    }

    /// True when there is no grouping (`g == 1`).
    #[inline]
    pub fn is_mha(&self) -> bool {
        self.group_size() == 1
    }

    /// The KV head query head `q_head` reads: `q_head / group_size`, the **contiguous** grouping
    /// (query heads `[j*g, (j+1)*g)` all read KV head `j`) that FA2/FA3, cuDNN, vLLM and PyTorch's
    /// `enable_gqa` all use. This is the one mapping the decode kernel reproduces on device.
    #[inline]
    pub fn kv_head_of(&self, q_head: usize) -> usize {
        debug_assert!(
            q_head < self.q_heads,
            "query head {q_head} out of range (q_heads = {})",
            self.q_heads
        );
        q_head / self.group_size()
    }

    /// Width of one token's **query** row: `q_heads * head_dim` — the model's hidden size `D`, and the
    /// stride of the `[bcap, D]` Q and output buffers.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.q_heads * self.kv.head_dim
    }

    /// Width of one token's **KV** row: `kv_heads * head_dim` — the K/V projection width and the
    /// per-token cache footprint. `g`x narrower than [`q_dim`](Self::q_dim).
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.kv.kv_dim()
    }

    /// K+V bytes the *equivalent MHA* cache would need — the same geometry with `kv_heads` raised to
    /// `q_heads`. Exactly `group_size()` times [`KvConfig::kv_bytes`], and the number that makes the
    /// GQA saving quotable instead of implied.
    #[inline]
    pub fn mha_equivalent_kv_bytes(&self, elem_size: usize) -> usize {
        self.group_size() * self.kv.kv_bytes(elem_size)
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
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_slots: usize,
        max_blocks_per_seq: usize,
    ) -> Self {
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
    F16 {
        k: CudaSlice<half::f16>,
        v: CudaSlice<half::f16>,
    },
    Int8 {
        k: CudaSlice<i8>,
        v: CudaSlice<i8>,
        ksc: CudaSlice<f32>,
        vsc: CudaSlice<f32>,
    },
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

/// The **device** paged KV-cache: the K and V slabs (f16, or int8 + scale slabs — see [`KvStorage`])
/// laid out `[layers, num_blocks, block_size, heads, head_dim]`, plus the [`BlockManager`] policy and
/// reusable device buffers for the block table / context lengths the attention kernel reads. Built only
/// with a live `Gpu` (the host policy in [`BlockManager`] is what the unit tests cover).
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
    pub fn new_with_dtype(
        stream: Arc<CudaStream>,
        cfg: KvConfig,
        dtype: KvDtype,
    ) -> Result<Self, DriverError> {
        let mgr = BlockManager::new(
            cfg.num_blocks,
            cfg.block_size,
            cfg.num_slots,
            cfg.max_blocks_per_seq,
        );
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
        Ok(Self {
            cfg,
            mgr,
            stream,
            storage,
            block_table_d,
            ctx_len_d,
        })
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
            KvStorage::Int8 { .. } => {
                panic!("f16 slab accessor on an int8 cache — use storage_mut()")
            }
        }
    }
    /// Mutable device V slab (f16 storage only).
    pub fn v_mut(&mut self) -> &mut CudaSlice<half::f16> {
        match &mut self.storage {
            KvStorage::F16 { v, .. } => v,
            KvStorage::Int8 { .. } => {
                panic!("f16 slab accessor on an int8 cache — use storage_mut()")
            }
        }
    }

    /// Both f16 slabs mutably at once (`&mut K`, `&mut V`) via a split borrow — the decode step needs
    /// to pass both to the append kernel in one launch, which two separate `k_mut`/`v_mut` calls
    /// (each a full `&mut self`) cannot express. F16 storage only (tests/population helpers);
    /// dtype-generic code goes through [`storage_mut`](Self::storage_mut).
    pub fn slabs_mut(&mut self) -> (&mut CudaSlice<half::f16>, &mut CudaSlice<half::f16>) {
        match &mut self.storage {
            KvStorage::F16 { k, v } => (k, v),
            KvStorage::Int8 { .. } => {
                panic!("f16 slab accessor on an int8 cache — use storage_mut()")
            }
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
        assert!(
            vs_f16 > 1.9 && vs_f16 < 2.0,
            "int8 ~half of f16 (got {vs_f16:.3}x)"
        );
        assert!(
            vs_f32 > 3.8 && vs_f32 < 4.0,
            "int8 ~quarter of f32 (got {vs_f32:.3}x)"
        );
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
            assert!(
                max_ctx >= 96,
                "Bcap={bcap} must fit the bench's 96-token contexts (got {max_ctx})"
            );
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

    /// A zero anywhere in the geometry makes `max_ctx_within_budget`'s divisor zero. The public API
    /// must name the offending parameters instead of surfacing a bare `attempt to divide by zero`
    /// from inside the function. (`num_slots = 0` here; `block_size` and every factor of `per_tok`
    /// reach the same guard.)
    #[test]
    #[should_panic(expected = "must all be non-zero")]
    fn zero_geometry_in_max_ctx_within_budget_is_named() {
        KvConfig::max_ctx_within_budget(12, 8, 64, 16, 0, 2, 4usize << 30);
    }

    // ================== GQA geometry (pure arithmetic — no device, no feature gate) ==================

    /// Real head geometries, as the models ship them: `(name, layers, q_heads, kv_heads, head_dim)`.
    /// Two GQA models, one MQA-adjacent extreme and one genuine MHA control, so every assertion below
    /// is exercised at `g = 1` as well as `g > 1`.
    const MODEL_GEOMETRIES: [(&str, usize, usize, usize, usize); 5] = [
        ("Llama-3-8B", 32, 32, 8, 128),
        ("Llama-3-70B", 80, 64, 8, 128),
        ("Mistral-7B", 32, 32, 8, 128),
        ("MQA-32x1", 32, 32, 1, 128),
        ("GPT-2 (MHA control)", 12, 12, 12, 64),
    ];

    /// The grouping is validated **at construction**, loudly. 32 query heads cannot be split across 7
    /// KV heads: some KV head would serve a fractional number of query heads, and every index derived
    /// from `q_head / g` would then be silently wrong rather than absent.
    #[test]
    #[should_panic(expected = "must be a whole multiple of kv_heads")]
    fn gqa_construction_rejects_an_indivisible_grouping() {
        let kv = KvConfig::for_serving(2, 7, 64, 16, 4, 256);
        GqaConfig::new(kv, 32);
    }

    /// A zero head count is named, not surfaced as a divide-by-zero from inside `group_size`.
    #[test]
    #[should_panic(expected = "must all be non-zero")]
    fn gqa_construction_rejects_zero_query_heads() {
        let kv = KvConfig::for_serving(2, 8, 64, 16, 4, 256);
        GqaConfig::new(kv, 0);
    }

    /// `kv_heads > q_heads` is caught by the same divisibility rule (a positive `q_heads` divisible by
    /// `kv_heads` is necessarily `>= kv_heads`), so there is no second check to forget to write.
    #[test]
    #[should_panic(expected = "must be a whole multiple of kv_heads")]
    fn gqa_construction_rejects_more_kv_heads_than_query_heads() {
        let kv = KvConfig::for_serving(2, 16, 64, 16, 4, 256);
        GqaConfig::new(kv, 8);
    }

    /// **Nothing regresses at `kv_heads == q_heads`.** `GqaConfig::mha` must be the exact identity
    /// case: group size 1, `kv_head_of` the identity on every head, Q and KV widths equal, and the
    /// "MHA equivalent" byte count equal to the cache's own — i.e. no saving claimed where none exists.
    #[test]
    fn mha_is_the_group_size_one_special_case() {
        for heads in [1usize, 3, 8, 12, 32] {
            let kv = KvConfig::for_serving(4, heads, 64, 16, 2, 128);
            let g = GqaConfig::mha(kv);
            assert!(g.is_mha(), "heads={heads}");
            assert_eq!(g.group_size(), 1, "heads={heads}");
            assert_eq!(g.q_heads(), heads);
            assert_eq!(g.kv_heads(), heads);
            assert_eq!(g.q_dim(), g.kv_dim());
            assert_eq!(g.kv(), &kv, "mha() must not disturb the cache geometry");
            for h in 0..heads {
                assert_eq!(g.kv_head_of(h), h, "MHA head mapping must be the identity");
            }
            for esz in [1usize, 2, 4] {
                assert_eq!(g.mha_equivalent_kv_bytes(esz), kv.kv_bytes(esz));
            }
        }
    }

    /// **The headline: a Llama-3-8B KV cache is exactly a quarter of the MHA equivalent.** 32 query
    /// heads over 8 KV heads is `g = 4`, and the cache — which is what has to fit in device memory —
    /// shrinks by exactly that factor, at every storage dtype. This is the whole point of the
    /// architecture, and until `kv_heads` existed as its own quantity Wukong allocated all four
    /// copies: every capacity, paging and throughput number was being measured on a 4x-too-large
    /// workload, and no comparison against a vLLM- or TRT-LLM-class peer was measuring the same thing.
    #[test]
    fn llama3_8b_kv_cache_is_exactly_a_quarter_of_the_mha_equivalent() {
        let (layers, q_heads, kv_heads, hd) = (32usize, 32usize, 8usize, 128usize);
        let (bsz, slots, max_ctx) = (16usize, 8usize, 8192usize);
        let gqa = GqaConfig::for_serving(layers, q_heads, kv_heads, hd, bsz, slots, max_ctx);
        // The same pool geometry with the grouping removed — identical everywhere but the head axis.
        let mha = KvConfig::for_serving(layers, q_heads, hd, bsz, slots, max_ctx);

        assert_eq!(gqa.group_size(), 4);
        assert_eq!(gqa.q_dim(), 4096, "hidden size is query-headed");
        assert_eq!(
            gqa.kv_dim(),
            1024,
            "the cached row is KV-headed — a quarter as wide"
        );
        // Everything but the head axis must be identical, or the comparison below is not a comparison.
        assert_eq!(gqa.kv().num_blocks, mha.num_blocks);
        assert_eq!(gqa.kv().max_blocks_per_seq, mha.max_blocks_per_seq);
        assert_eq!(gqa.kv().layers, mha.layers);

        // Exactly 4x, in elements and at every dtype — not "about" 4x.
        assert_eq!(4 * gqa.kv().slab_elems(), mha.slab_elems());
        assert_eq!(4 * gqa.kv().scale_slab_elems(), mha.scale_slab_elems());
        for esz in [1usize, 2, 4] {
            assert_eq!(
                4 * gqa.kv().kv_bytes(esz),
                mha.kv_bytes(esz),
                "f{} KV bytes must be exactly a quarter",
                esz * 8
            );
        }
        assert_eq!(4 * gqa.kv().kv_bytes_int8(), mha.kv_bytes_int8());
        // And the config reports the saving itself, so it can be quoted without re-deriving it.
        assert_eq!(gqa.mha_equivalent_kv_bytes(2), mha.kv_bytes(2));

        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "Llama-3-8B ({q_heads}q/{kv_heads}kv x {hd}, {layers}L, {slots} slots x {max_ctx} tok): \
             f16 KV {:.2} GiB with GQA vs {:.2} GiB without = exactly {}x smaller",
            gib(gqa.kv().kv_bytes(2)),
            gib(mha.kv_bytes(2)),
            gqa.group_size()
        );
    }

    /// The shrink is `g` for every real geometry, not just Llama-3-8B — including `g = 1`, where it
    /// must be exactly 1 (an MHA model must not be told it saved anything).
    #[test]
    fn every_model_geometry_shrinks_the_cache_by_exactly_its_group_size() {
        for (name, layers, q_heads, kv_heads, hd) in MODEL_GEOMETRIES {
            let gqa = GqaConfig::for_serving(layers, q_heads, kv_heads, hd, 16, 8, 4096);
            let mha = KvConfig::for_serving(layers, q_heads, hd, 16, 8, 4096);
            let g = gqa.group_size();
            assert_eq!(g, q_heads / kv_heads, "{name}");
            assert_eq!(g * gqa.kv().kv_bytes(2), mha.kv_bytes(2), "{name}");
            assert_eq!(gqa.mha_equivalent_kv_bytes(2), mha.kv_bytes(2), "{name}");
            assert_eq!(gqa.is_mha(), q_heads == kv_heads, "{name}");
            eprintln!(
                "{name}: {q_heads}q/{kv_heads}kv (g={g}) -> KV {:.2} GiB vs {:.2} GiB ungrouped",
                gqa.kv().kv_bytes(2) as f64 / (1u64 << 30) as f64,
                mha.kv_bytes(2) as f64 / (1u64 << 30) as f64
            );
        }
    }

    /// `kv_head_of` must be the **contiguous equal partition** every peer uses (FA2/FA3, cuDNN, vLLM,
    /// PyTorch `enable_gqa`): query heads `[j*g, (j+1)*g)` read KV head `j`. Checked exhaustively over
    /// every query head of every geometry — monotone, surjective onto `0..kv_heads`, in range, and
    /// each KV head serving exactly `g` query heads. A round-robin partition (`h % kv_heads`) would
    /// satisfy "in range" and "surjective" and be wrong; the per-group census is what excludes it.
    #[test]
    fn kv_head_mapping_is_a_contiguous_equal_partition() {
        for (name, layers, q_heads, kv_heads, hd) in MODEL_GEOMETRIES {
            let gqa = GqaConfig::for_serving(layers, q_heads, kv_heads, hd, 16, 2, 64);
            let g = gqa.group_size();
            let mut census = vec![0usize; kv_heads];
            let mut prev = 0usize;
            for h in 0..q_heads {
                let kvh = gqa.kv_head_of(h);
                assert!(kvh < kv_heads, "{name}: head {h} -> {kvh} out of range");
                assert!(
                    kvh >= prev,
                    "{name}: mapping must be monotone in the query head"
                );
                assert_eq!(
                    kvh,
                    h / g,
                    "{name}: head {h} must map to the contiguous group"
                );
                census[kvh] += 1;
                prev = kvh;
            }
            for (j, &c) in census.iter().enumerate() {
                assert_eq!(
                    c, g,
                    "{name}: KV head {j} must serve exactly g={g} query heads"
                );
            }
            assert_eq!(gqa.kv_head_of(0), 0, "{name}");
            assert_eq!(gqa.kv_head_of(q_heads - 1), kv_heads - 1, "{name}");
        }
    }

    /// **The slab is KV-sized, so it must be indexed with a KV head.** Exhaustively: every
    /// `(layer, block, token, kv_head_of(q_head), dh)` offset lands inside `slab_elems()`, and the last
    /// one lands exactly on the last element. Then the negative half — indexing the same slab with the
    /// *query* head is not a benign over-count, it is an out-of-bounds device write: at the final
    /// token the offset for any head `>= kv_heads` is past the end of the slab.
    #[test]
    fn gqa_slab_offsets_stay_inside_the_kv_sized_slab() {
        let gqa = GqaConfig::new(
            KvConfig {
                layers: 2,
                heads: 2, // kv_heads
                head_dim: 4,
                block_size: 3,
                num_blocks: 5,
                num_slots: 2,
                max_blocks_per_seq: 3,
            },
            8, // q_heads -> g = 4
        );
        let cfg = *gqa.kv();
        assert_eq!(gqa.group_size(), 4);
        let n = cfg.slab_elems();
        let mut seen = vec![false; n];
        for layer in 0..cfg.layers {
            for blk in 0..cfg.num_blocks {
                for tok in 0..cfg.block_size {
                    for q_head in 0..gqa.q_heads() {
                        for dh in 0..cfg.head_dim {
                            let off =
                                cfg.elem_offset(layer, blk as u32, tok, gqa.kv_head_of(q_head), dh);
                            assert!(
                                off < n,
                                "q_head {q_head} escaped the KV slab ({off} >= {n})"
                            );
                            seen[off] = true;
                        }
                    }
                }
            }
        }
        // The g query heads of a group share one row, so the mapped offsets cover the slab exactly.
        assert!(seen.iter().all(|&s| s), "the KV slab must be fully covered");
        assert_eq!(
            cfg.elem_offset(
                cfg.layers - 1,
                (cfg.num_blocks - 1) as u32,
                cfg.block_size - 1,
                cfg.kv_heads() - 1,
                cfg.head_dim - 1
            ),
            n - 1
        );
        // Negative: the query head is not a valid slab index once g > 1.
        for bad_head in cfg.kv_heads()..gqa.q_heads() {
            let off = cfg.elem_offset(
                cfg.layers - 1,
                (cfg.num_blocks - 1) as u32,
                cfg.block_size - 1,
                bad_head,
                0,
            );
            assert!(
                off >= n,
                "indexing a KV-sized slab with query head {bad_head} must run past its end \
                 (got {off} < {n}) — this is the out-of-bounds the KV-head axis prevents"
            );
        }
    }

    /// The serving consequence of the shrink: inside one fixed device budget a GQA model reaches
    /// exactly `g` times the simultaneous context its MHA twin does. `max_ctx_within_budget` takes the
    /// **KV**-head count, which is what makes that fall out rather than needing a correction factor.
    #[test]
    fn gqa_buys_exactly_g_times_the_context_in_one_budget() {
        let budget = 4usize << 30;
        let (layers, q_heads, kv_heads, hd, bsz, slots) = (32usize, 32usize, 8usize, 128, 16, 8);
        let ctx_gqa = KvConfig::max_ctx_within_budget(layers, kv_heads, hd, bsz, slots, 2, budget);
        let ctx_mha = KvConfig::max_ctx_within_budget(layers, q_heads, hd, bsz, slots, 2, budget);
        assert!(ctx_mha > 0 && ctx_gqa > 0);
        assert_eq!(
            ctx_gqa,
            (q_heads / kv_heads) * ctx_mha,
            "GQA must fit exactly g times the context in the same budget"
        );
        // And the geometry that claims that context must actually fit the budget it was derived from.
        let gqa = GqaConfig::for_serving(layers, q_heads, kv_heads, hd, bsz, slots, ctx_gqa);
        let bytes = gqa.kv().assert_kv_budget(2, budget);
        eprintln!(
            "4 GiB KV budget, Llama-3-8B shape: {ctx_gqa} tok/seq x {slots} slots with GQA \
             ({:.2} GiB used) vs {ctx_mha} tok/seq without",
            bytes as f64 / (1u64 << 30) as f64
        );
    }

    /// The block allocator carries **no** head geometry — it deals in blocks, slots and tokens — so a
    /// GQA cache and its MHA twin must produce byte-identical block tables and context lengths for the
    /// same request pattern. The saving is entirely in the slab, and this pins that it is not silently
    /// also changing paging policy (which would make the two configurations incomparable).
    #[test]
    fn the_block_manager_is_head_geometry_free() {
        let gqa = GqaConfig::for_serving(4, 32, 8, 64, 4, 3, 32);
        let mha = KvConfig::for_serving(4, 32, 64, 4, 3, 32);
        let mk = |c: &KvConfig| {
            BlockManager::new(
                c.num_blocks,
                c.block_size,
                c.num_slots,
                c.max_blocks_per_seq,
            )
        };
        let (mut a, mut b) = (mk(gqa.kv()), mk(&mha));
        for (slot, toks) in [(0usize, 9usize), (2, 4), (1, 17)] {
            assert_eq!(a.reserve(slot, toks), b.reserve(slot, toks));
        }
        a.free(2);
        b.free(2);
        assert_eq!(a.append(2), b.append(2));
        assert_eq!(a.flat_block_table(), b.flat_block_table());
        assert_eq!(a.ctx_lens(), b.ctx_lens());
        assert_eq!(a.layout_epoch(), b.layout_epoch());
        assert_eq!(a.free_blocks(), b.free_blocks());
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
        let last = cfg.elem_offset(
            cfg.layers - 1,
            (cfg.num_blocks - 1) as u32,
            cfg.block_size - 1,
            cfg.heads - 1,
            cfg.head_dim - 1,
        );
        assert_eq!(last, cfg.slab_elems() - 1);
        // Adjacent dh elements are contiguous (row-major innermost).
        assert_eq!(
            cfg.elem_offset(0, 0, 0, 0, 1) - cfg.elem_offset(0, 0, 0, 0, 0),
            1
        );
        // Adjacent head steps by head_dim.
        assert_eq!(
            cfg.elem_offset(0, 0, 0, 1, 0) - cfg.elem_offset(0, 0, 0, 0, 0),
            cfg.head_dim
        );
        // Adjacent layer steps by one full plane.
        assert_eq!(
            cfg.elem_offset(1, 0, 0, 0, 0) - cfg.elem_offset(0, 0, 0, 0, 0),
            cfg.layer_plane_elems()
        );
    }
}
