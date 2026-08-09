//! Paged decode-attention kernel (serving) — single-query attention against the [paged
//! KV-cache](crate::paged_kv), gathering each sequence's K/V through its block table.
//!
//! At decode the model has exactly **one new query token per sequence** and must attend over that
//! sequence's **entire cached past**. This is the PagedAttention decode kernel: for each
//! `(slot, head)` it streams the sequence's `context_len` cached K/V vectors — fetched through the
//! per-sequence block table, so the physical blocks need not be contiguous — and runs a numerically
//! stable **online softmax** with **FP32 accumulators** (the precision vLLM/TRT-LLM use for the logits
//! and the V-accumulation).
//!
//! ## Kernel shape (warp-cooperative)
//! One **warp per `(slot, head)`**, [`PAGED_ATTN_WARPS`] pairs per CTA — grid =
//! `ceil(num_slots*heads / PAGED_ATTN_WARPS)` CTAs of `32 * PAGED_ATTN_WARPS` threads. Each warp
//! stages its query vector once into **shared memory** (`qsh`, one `head_dim` row per warp) and its 32
//! lanes then split the context by position — lane `i` owns `{t : t % 32 == i}` — each carrying its own
//! online-softmax state (`m`, `l`, and a `head_dim`-wide register accumulator `%acc0..`, unrolled). A
//! fixed `shfl.sync.bfly` butterfly (offsets `16,8,4,2,1`) merges the lanes at the end: `max` for `m`,
//! a rescale by `exp(m−M)`, then `add` for `l` and for every accumulator; lane 0 writes the normalized
//! row. There is **no shared-memory logits buffer**, so arbitrary context length works with no v2-style
//! split-K (vLLM needs v2 past 8192 tokens precisely to dodge that SMEM shortage). The block-table walk
//! is *not* incremental — a `div.u32` per position recovers `logical`/`offset`; making it incremental
//! is still an unclaimed perf lever.
//!
//! Every generator here is tagged with the **`sm_80` floor** ([`crate::ptx_target::HDR_SM80`]): the
//! instruction mix is `shfl.sync.bfly` + `red.f32` + ordinary ld/st and f32 math, all Ampere-legal,
//! so the module driver-JITs on every part from A100 up. The `assert_floor` gate below pins it, and
//! being un-gated it runs in a plain toolchain-free `cargo test`.
//!
//! ## The first-law property the gates prove
//! - **Absolute correctness**: tolerance-gated vs an f64 full-softmax CPU reference
//!   ([`reference_decode_attn`]) — the only legitimate error is the GPU's `ex2.approx` exp + f32 order.
//! - **Paging is numerically invisible**: the kernel output is **bit-for-bit identical** when the same
//!   logical sequences are laid into two *different* physical block layouts (contiguous vs a fragmented
//!   free-list order); the `paged_attention_invariant_to_block_layout` gate proves it to the bit (the
//!   decode analogue of int8 split-K / transpose bit-exactness).
//!
//! ### The invariant that property rests on (load-bearing — read before touching the loop)
//! It is **not** "one thread, one fixed accumulation order". It is two separate facts:
//! 1. the lane partition is a function of the **logical position index alone** (`t % 32`), so no
//!    physical block id can influence *which* lane a value lands in; and
//! 2. the cross-lane merge order is a **fixed** butterfly sequence.
//!
//! The block table therefore only changes *where* a value is read, never which partial accumulates it
//! nor in what order. Re-partitioning the lanes for coalescing stays safe only while it remains
//! position-keyed: partitioning by *physical block* — the obvious next optimization, and one that still
//! satisfies "one fixed accumulation order per lane" — makes the accumulation order a function of the
//! layout and breaks bit-exactness.
//!
//! ## Grouped-query attention
//! The decode kernels take **two** head counts (see [`crate::paged_kv::GqaConfig`]): `pHeads`, the
//! **query**-head count, sizes the grid and addresses the Q and O rows; `pKvHeads` addresses the cache
//! and nothing else. A warp still owns one `(slot, q_head)` pair and simply gathers from KV head
//! `q_head / (q_heads / kv_heads)` — Q, O and the lane partition are untouched. `kv_append` and the
//! int8 quantizer are KV-headed throughout and needed no change at all, because the thing they write
//! *is* the K/V projection.
//!
//! **The layout-invariance proof survives GQA by the argument above, not by luck**: the property rests
//! on the lane partition being a function of the logical position index alone (`t % 32`) and on a fixed
//! butterfly merge order. GQA changes which *head's* K/V a lane reads and neither of those two things,
//! so `paged_gqa_attention_invariant_to_block_layout` holds for the same reason its MHA twin does. It
//! is worth saying out loud because this is exactly the kind of change that looks like it should break
//! it — and the gate is run anyway rather than the argument trusted.

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

// Only the (gpu-gated) launchers and device gates take a `KvConfig`/`GqaConfig`; the generators, the
// quantizer and the f64 reference are geometry-free, which is what lets this module compile un-gated.
#[cfg(feature = "gpu")]
use crate::paged_kv::{GqaConfig, KvConfig};

// The module header every generator below opens with. `ptx_target` is deliberately un-gated (pure
// strings, no `cudarc`) precisely so this un-gated module — whose PTX-shape gates run in a plain,
// toolchain-free `cargo test` — can route through it like the gpu-gated families do.
use crate::ptx_target::HDR_SM80;

/// `(slot, head)` pairs per CTA — one **warp** each (block_dim = `32 * PAGED_ATTN_WARPS`). The warp's
/// 32 lanes cooperatively stream the context, so the launch fills the GPU (`num_slots*heads` warps)
/// instead of the one-thread-per-pair version's `num_slots*heads` *threads* (which left 39/40 SMs idle).
pub const PAGED_ATTN_WARPS: u32 = 4;

/// PTX entry name for the paged decode-attention kernel.
pub const PAGED_ATTN_ENTRY: &str = "paged_attn_decode";

/// Generate the paged decode-attention PTX, **specialized to `head_dim`** (baked so the V accumulator
/// unrolls into named registers `%acc0..` and the per-position dot unrolls). **Warp-cooperative**: one
/// warp computes `out[slot,head,:] = softmax(scale · q · Kᵀ) · V` over the sequence's context — the 32
/// lanes split the context positions (`lane, lane+32, …`), each keeping a partial online-softmax state,
/// then a fixed shfl-butterfly merge combines them (`m`→max, rescale, `l`/`acc`→sum). The query vector
/// lives in shared memory (one copy per warp, read by every lane). `PAGED_ATTN_WARPS` `(slot,head)`
/// pairs per CTA. Layout matches [`crate::paged_kv::KvConfig::elem_offset`]: `[layers, num_blocks, block_size, kv_heads,
/// head_dim]`, f16 cache, f32 query/out. **Bit-exact across block layouts** (the lane partition + merge
/// order are layout-independent; the block table only changes the load address).
///
/// **GQA:** the warp owns a `(slot, q_head)` pair and reads KV head `q_head / (q_heads / kv_heads)`.
/// `pHeads` is the **query**-head count (it drives the grid, the `(slot, head)` split and the Q/O row);
/// `pKvHeads` is the **KV**-head count (it drives the cache offset alone). Both are runtime `.param`s
/// rather than baked constants precisely so one PTX per `head_dim` still serves every grouping — a
/// per-`g` variant would need a per-`g` module-cache key, and [`crate::Gpu::function`] caches on the
/// key alone and never re-examines the PTX on a hit. `q_heads == kv_heads` makes the group size 1 and
/// the mapping the identity, i.e. exactly the pre-GQA kernel.
pub fn paged_attn_decode_ptx(head_dim: usize) -> String {
    assert!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "head_dim must be a positive even number"
    );
    let hd = head_dim;
    let w = PAGED_ATTN_WARPS as usize;
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    // Finite negative sentinel for the online-softmax max init (avoids the -inf − -inf = NaN that the
    // cross-lane merge would hit when an entire warp's context is empty).
    let neg_big = format!("0f{:08X}", (-1.0e30f32).to_bits());
    // Warp butterfly all-reduce of one f32 register under `op` (every lane ends with the full result).
    let bfly = |reg: &str, op: &str| -> String {
        let mut t = String::new();
        for off in [16, 8, 4, 2, 1] {
            t += &format!("    shfl.sync.bfly.b32 %rt,{reg},{off},0x1f,0xffffffff;\n    {op}.f32 {reg},{reg},%rt;\n");
        }
        t
    };
    let mut s = String::new();
    s += HDR_SM80;
    s += "\n";
    s += &format!(
        ".visible .entry {PAGED_ATTN_ENTRY}(\n\
        \x20   .param .u64 pQ,\n\
        \x20   .param .u64 pK,\n\
        \x20   .param .u64 pV,\n\
        \x20   .param .u64 pO,\n\
        \x20   .param .u64 pBT,\n\
        \x20   .param .u64 pCL,\n\
        \x20   .param .f32 pScale,\n\
        \x20   .param .u32 pBcap,\n\
        \x20   .param .u32 pHeads,\n\
        \x20   .param .u32 pKvHeads,\n\
        \x20   .param .u32 pBsz,\n\
        \x20   .param .u32 pNblk,\n\
        \x20   .param .u32 pMbps,\n\
        \x20   .param .u32 pLayer\n)\n{{\n"
    );
    s += &format!("    .shared .f32 qsh[{}];\n", w * hd);
    s += &format!("    .reg .f32 %acc<{hd}>;\n");
    s += "    .reg .f32 %score,%m,%l,%newm,%p,%corr,%kf,%vf,%qv,%invl,%scale,%factor,%M,%L,%t0,%rt;\n";
    s += "    .reg .b16 %h;\n";
    s += "    .reg .b32 %tix,%warp,%lane,%gid,%slot,%head,%ctx,%t,%logical,%off,%phys,%D,%qidx,%nq,%tmp,%bcap,%heads,%bsz,%nblk,%mbps,%layer,%e,%dd,%kvheads,%kvhead,%grp;\n";
    s += "    .reg .b64 %Q,%K,%V,%O,%BT,%CL,%addr,%qrow,%obase,%kbase,%vbase,%offb,%qshw;\n";
    s += "    .reg .pred %p0,%p1,%p2;\n";
    s += "    ld.param.u64 %Q,[pQ];   cvta.to.global.u64 %Q,%Q;\n";
    s += "    ld.param.u64 %K,[pK];   cvta.to.global.u64 %K,%K;\n";
    s += "    ld.param.u64 %V,[pV];   cvta.to.global.u64 %V,%V;\n";
    s += "    ld.param.u64 %O,[pO];   cvta.to.global.u64 %O,%O;\n";
    s += "    ld.param.u64 %BT,[pBT]; cvta.to.global.u64 %BT,%BT;\n";
    s += "    ld.param.u64 %CL,[pCL]; cvta.to.global.u64 %CL,%CL;\n";
    s += "    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u32 %bcap,[pBcap];\n    ld.param.u32 %heads,[pHeads];\n    ld.param.u32 %kvheads,[pKvHeads];\n    ld.param.u32 %bsz,[pBsz];\n";
    s += "    ld.param.u32 %nblk,[pNblk];\n    ld.param.u32 %mbps,[pMbps];\n    ld.param.u32 %layer,[pLayer];\n";
    // tid = tid.x ; warp = tid/32 ; lane = tid%32 ; gid = ctaid.x*WARPS + warp.
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warp,%tix,5;\n    and.b32 %lane,%tix,31;\n";
    s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mad.lo.s32 %gid,%tmp,{w},%warp;\n");
    s += "    mul.lo.s32 %nq,%bcap,%heads;\n    setp.ge.u32 %p0,%gid,%nq;\n    @%p0 bra DONE;\n";
    // slot = gid/heads ; head = gid - slot*heads ; ctx = CL[slot].  (`heads` is the QUERY head count.)
    s += "    div.u32 %slot,%gid,%heads;\n    mul.lo.s32 %tmp,%slot,%heads;\n    sub.u32 %head,%gid,%tmp;\n";
    // GQA: grp = q_heads/kv_heads ; kvhead = head/grp. Both warp-uniform and loop-invariant, so this
    // is two integer divides per warp, once. The host validates q_heads % kv_heads == 0 (GqaConfig),
    // which is what makes `grp` exact and non-zero; MHA gives grp = 1 and kvhead = head.
    s += "    div.u32 %grp,%heads,%kvheads;\n    div.u32 %kvhead,%head,%grp;\n";
    s += "    mul.wide.u32 %offb,%slot,4;\n    add.s64 %addr,%CL,%offb;\n    ld.global.u32 %ctx,[%addr];\n";
    // qidx = slot*D + head*hd ; qrow = Q + qidx*4 ; obase = O + qidx*4.
    s += &format!("    mul.lo.s32 %D,%heads,{hd};\n    mul.lo.s32 %qidx,%slot,%D;\n    mul.lo.s32 %tmp,%head,{hd};\n    add.u32 %qidx,%qidx,%tmp;\n");
    s += "    mul.wide.u32 %offb,%qidx,4;\n    add.s64 %qrow,%Q,%offb;\n    add.s64 %obase,%O,%offb;\n";
    // qshw = &qsh[warp*hd] (this warp's query staging in shared).
    s += &format!("    mov.u64 %qshw,qsh;\n    mul.wide.u32 %offb,%warp,{};\n    add.s64 %qshw,%qshw,%offb;\n", hd * 4);
    // Cooperatively stage q into shared: lane stores d = lane, lane+32, … ; warp-sync before the dot.
    s +=
        &format!("    mov.u32 %dd,%lane;\nQL:\n    setp.ge.u32 %p1,%dd,{hd};\n    @%p1 bra QLE;\n");
    s += "    mul.wide.u32 %offb,%dd,4;\n    add.s64 %addr,%qrow,%offb;\n    ld.global.f32 %qv,[%addr];\n";
    s += "    add.s64 %addr,%qshw,%offb;\n    st.shared.f32 [%addr],%qv;\n    add.u32 %dd,%dd,32;\n    bra QL;\nQLE:\n";
    s += "    bar.warp.sync 0xffffffff;\n";
    // Per-lane online softmax over positions {t : t%32 == lane}.
    s += &format!("    mov.f32 %m,{neg_big};\n    mov.f32 %l,0f00000000;\n");
    for d in 0..hd {
        s += &format!("    mov.f32 %acc{d},0f00000000;\n");
    }
    s += "    mov.u32 %t,%lane;\nLOOP:\n    setp.ge.u32 %p1,%t,%ctx;\n    @%p1 bra ENDLOOP;\n";
    // logical = t/bsz ; off = t - logical*bsz ; phys = BT[slot*mbps + logical].
    s += "    div.u32 %logical,%t,%bsz;\n    mul.lo.s32 %off,%logical,%bsz;\n    sub.u32 %off,%t,%off;\n";
    s += "    mul.lo.s32 %tmp,%slot,%mbps;\n    add.u32 %tmp,%tmp,%logical;\n    mul.wide.u32 %offb,%tmp,4;\n    add.s64 %addr,%BT,%offb;\n    ld.global.u32 %phys,[%addr];\n";
    // e = ((((layer*nblk)+phys)*bsz + off)*kvheads + kvhead)*hd ; kbase/vbase = slab + e*2.
    // The cache is KV-headed: this chain reproduces `KvConfig::elem_offset`, whose head axis is
    // `kv_heads` wide. Using the query head here would index past the (g-times-smaller) slab.
    s += "    mul.lo.s32 %e,%layer,%nblk;\n    add.u32 %e,%e,%phys;\n    mul.lo.s32 %e,%e,%bsz;\n    add.u32 %e,%e,%off;\n";
    s += &format!(
        "    mul.lo.s32 %e,%e,%kvheads;\n    add.u32 %e,%e,%kvhead;\n    mul.lo.s32 %e,%e,{hd};\n"
    );
    s += "    mul.wide.u32 %offb,%e,2;\n    add.s64 %kbase,%K,%offb;\n    add.s64 %vbase,%V,%offb;\n";
    // score = scale * dot(q, K[t])  (q from shared, K widened from f16).
    s += "    mov.f32 %score,0f00000000;\n";
    for d in 0..hd {
        s += &format!("    ld.shared.f32 %qv,[%qshw+{}];\n    ld.global.u16 %h,[%kbase+{}];\n    cvt.f32.f16 %kf,%h;\n    fma.rn.f32 %score,%qv,%kf,%score;\n", d * 4, d * 2);
    }
    s += "    mul.f32 %score,%score,%scale;\n";
    s += "    max.f32 %newm,%m,%score;\n";
    s += &format!(
        "    sub.f32 %t0,%m,%newm;\n    mul.f32 %t0,%t0,{log2e};\n    ex2.approx.f32 %corr,%t0;\n"
    );
    s += &format!(
        "    sub.f32 %t0,%score,%newm;\n    mul.f32 %t0,%t0,{log2e};\n    ex2.approx.f32 %p,%t0;\n"
    );
    s += "    mul.f32 %l,%l,%corr;\n    add.f32 %l,%l,%p;\n";
    for d in 0..hd {
        s += &format!("    ld.global.u16 %h,[%vbase+{}];\n    cvt.f32.f16 %vf,%h;\n    mul.f32 %acc{d},%acc{d},%corr;\n    fma.rn.f32 %acc{d},%p,%vf,%acc{d};\n", d * 2);
    }
    s += "    mov.f32 %m,%newm;\n    add.u32 %t,%t,32;\n    bra LOOP;\nENDLOOP:\n";
    // Cross-lane merge (fixed butterfly order): M = max(m) ; rescale by exp(m−M) ; L = Σl ; ACC = Σacc.
    s += "    mov.f32 %M,%m;\n";
    s += &bfly("%M", "max");
    s += &format!(
        "    sub.f32 %t0,%m,%M;\n    mul.f32 %t0,%t0,{log2e};\n    ex2.approx.f32 %factor,%t0;\n"
    );
    s += "    mul.f32 %l,%l,%factor;\n";
    for d in 0..hd {
        s += &format!("    mul.f32 %acc{d},%acc{d},%factor;\n");
    }
    s += "    mov.f32 %L,%l;\n";
    s += &bfly("%L", "add");
    for d in 0..hd {
        s += &bfly(&format!("%acc{d}"), "add");
    }
    // invL = (L>0) ? 1/L : 0  (the empty/inactive-sequence guard) ; lane 0 writes out[d] = ACC[d]·invL.
    s += "    rcp.rn.f32 %invl,%L;\n    setp.gt.f32 %p2,%L,0f00000000;\n    selp.f32 %invl,%invl,0f00000000,%p2;\n";
    s += "    setp.ne.u32 %p0,%lane,0;\n    @%p0 bra DONE;\n";
    for d in 0..hd {
        s += &format!(
            "    mul.f32 %t0,%acc{d},%invl;\n    st.global.f32 [%obase+{}],%t0;\n",
            d * 4
        );
    }
    s += "DONE:\n    ret;\n}\n";
    s
}

/// Launch the paged decode-attention kernel for **one layer** over a batch of `bcap` slots, on
/// `stream`, with a **preloaded** `func` (so the resident/graph path issues no JIT). All buffers are
/// device-resident:
/// - `q_d`: f32 `[bcap, D]` — the new token's query per slot (`D = heads*head_dim`).
/// - `k_d`/`v_d`: the f16 KV slabs (`cfg.slab_elems()` each), `[layers, num_blocks, block_size, heads,
///   head_dim]`.
/// - `out_d`: f32 `[bcap, D]` — written.
/// - `bt_d`: u32 `[bcap, cfg.max_blocks_per_seq]` block table; `cl_d`: u32 `[bcap]` context lengths.
///
/// `scale` is `1/sqrt(head_dim)`. Drawn-from-`alloc` (uninitialized) `out_d` is safe — the kernel fully
/// overwrites every active slot's row (inactive slots get zeros).
///
/// This is the **multi-head** entry point: it treats `cfg.heads` as both the query- and the KV-head
/// count. A grouped-query model must go through [`launch_paged_attn_decode_gqa`], which takes the two
/// counts separately; this function is exactly that one at `GqaConfig::mha(*cfg)`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn launch_paged_attn_decode(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    q_d: &CudaSlice<f32>,
    k_d: &CudaSlice<half::f16>,
    v_d: &CudaSlice<half::f16>,
    out_d: &mut CudaSlice<f32>,
    bt_d: &CudaSlice<u32>,
    cl_d: &CudaSlice<u32>,
    cfg: &KvConfig,
    layer: usize,
    bcap: usize,
    scale: f32,
) -> Result<(), DriverError> {
    launch_paged_attn_decode_gqa(
        stream,
        func,
        q_d,
        k_d,
        v_d,
        out_d,
        bt_d,
        cl_d,
        &GqaConfig::mha(*cfg),
        layer,
        bcap,
        scale,
    )
}

/// Launch the **grouped-query** paged decode-attention kernel for one layer — [`launch_paged_attn_decode`]
/// with the query- and KV-head counts held apart by [`GqaConfig`]:
/// - `q_d` / `out_d`: f32 `[bcap, gqa.q_dim()]` — **query**-headed (`q_heads * head_dim`, the hidden size).
/// - `k_d` / `v_d`: the f16 KV slabs (`gqa.kv().slab_elems()` each), **KV**-headed
///   `[layers, num_blocks, block_size, kv_heads, head_dim]`.
///
/// Query head `h` reads KV head `h / gqa.group_size()`; the kernel recomputes that mapping on device
/// from the two `.param` head counts. MQA (`kv_heads == 1`) and MHA (`kv_heads == q_heads`) are the
/// endpoints of the same code path, not special cases.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn launch_paged_attn_decode_gqa(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    q_d: &CudaSlice<f32>,
    k_d: &CudaSlice<half::f16>,
    v_d: &CudaSlice<half::f16>,
    out_d: &mut CudaSlice<f32>,
    bt_d: &CudaSlice<u32>,
    cl_d: &CudaSlice<u32>,
    gqa: &GqaConfig,
    layer: usize,
    bcap: usize,
    scale: f32,
) -> Result<(), DriverError> {
    let cfg = gqa.kv();
    debug_assert_eq!(
        q_d.len(),
        bcap * gqa.q_dim(),
        "q must be [bcap, q_heads*head_dim] (query-headed, not KV-headed)"
    );
    debug_assert_eq!(
        out_d.len(),
        bcap * gqa.q_dim(),
        "out must be [bcap, q_heads*head_dim] (query-headed, not KV-headed)"
    );
    debug_assert_eq!(
        k_d.len(),
        cfg.slab_elems(),
        "k slab must be cfg.slab_elems() (KV-headed)"
    );
    debug_assert_eq!(
        v_d.len(),
        cfg.slab_elems(),
        "v slab must be cfg.slab_elems() (KV-headed)"
    );
    debug_assert_eq!(
        bt_d.len(),
        bcap * cfg.max_blocks_per_seq,
        "block table must be [bcap, cfg.max_blocks_per_seq]"
    );
    debug_assert_eq!(cl_d.len(), bcap, "context lengths must be one u32 per slot");
    // One warp per (slot, QUERY head) — the grid is query-headed even though the cache is not.
    let nq = (bcap * gqa.q_heads()) as u32;
    let cfg_launch = LaunchConfig {
        grid_dim: (nq.div_ceil(PAGED_ATTN_WARPS), 1, 1),
        block_dim: (32 * PAGED_ATTN_WARPS, 1, 1),
        shared_mem_bytes: 0,
    };
    let (heads, kvheads, bsz, nblk, mbps, layer_u) = (
        gqa.q_heads() as u32,
        gqa.kv_heads() as u32,
        cfg.block_size as u32,
        cfg.num_blocks as u32,
        cfg.max_blocks_per_seq as u32,
        layer as u32,
    );
    let bcap_u = bcap as u32;
    let mut b = stream.launch_builder(func);
    b.arg(q_d)
        .arg(k_d)
        .arg(v_d)
        .arg(out_d)
        .arg(bt_d)
        .arg(cl_d)
        .arg(&scale);
    b.arg(&bcap_u)
        .arg(&heads)
        .arg(&kvheads)
        .arg(&bsz)
        .arg(&nblk)
        .arg(&mbps)
        .arg(&layer_u);
    // SAFETY: the pushed arguments match `PAGED_ATTN_ENTRY`'s parameter list in order and width (six
    // .u64 pointers, one .f32, **seven** .u32 — `pKvHeads` sits between `pHeads` and `pBsz`; see
    // `paged_attn_decode_ptx`, and `attention_entry_param_counts_match_the_launcher_push_lists` pins
    // the counts textually against this list). `func` was loaded from PTX generated for
    // `cfg.head_dim` (the entry is head_dim-specialized). Every buffer is at least as long as the
    // largest index this kernel's address arithmetic can produce for `bcap` slots at `layer`: the
    // `%e` chain reproduces `KvConfig::elem_offset` at the **KV** head axis, bounded by
    // `slab_elems()`; `BT[slot*mbps+logical]` by `bcap * max_blocks_per_seq`; `CL[slot]` and the
    // query-headed `Q`/`O` rows by the shapes asserted above — which the caller guarantees and the
    // debug asserts check at the call site. The device `div.u32 %grp,%heads,%kvheads` is exact and
    // non-zero because `GqaConfig`'s constructor rejects a non-divisible or zero pair, so `%kvhead`
    // cannot exceed `kv_heads - 1` and the offset cannot leave the slab.
    unsafe { b.launch(cfg_launch)? };
    Ok(())
}

/// PTX entry name for the **int8** paged decode-attention kernel.
pub const PAGED_ATTN_INT8_ENTRY: &str = "paged_attn_decode_int8";

/// Generate the **int8-KV** paged decode-attention PTX — the [`paged_attn_decode_ptx`] kernel with the
/// f16 cache replaced by an **int8** cache plus a **per-(token, head) f32 scale**. K/V are stored
/// `int8 ≈ value / scale`; the kernel reads the int8 byte (`ld.global.s8`), and because one scale covers
/// a whole head_dim vector it **factors the scale out of the dot**: `q·K = scaleK · Σ q[d]·int8K[d]` and
/// `acc += (p·scaleV)·int8V[d]`. Halves the cache footprint vs f16 (a quarter of f32); the tiny scale
/// slab is `head_dim×` smaller than the K slab. Same warp-cooperative online softmax; **bit-exact across
/// block layouts** (the dequant multiply order is fixed per token). Tolerance-gated (lossy), not bit-exact.
///
/// **GQA:** identical treatment to the f16 twin — `pHeads` is the query-head count, `pKvHeads` the KV
/// one, and query head `h` reads KV head `h / (q_heads / kv_heads)`. Both the int8 value index and the
/// per-(token, head) *scale* index are KV-headed, since the scale slab has one entry per cached
/// `(token, kv_head)`.
pub fn paged_attn_decode_int8_ptx(head_dim: usize) -> String {
    assert!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "head_dim must be a positive even number"
    );
    let hd = head_dim;
    let w = PAGED_ATTN_WARPS as usize;
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let neg_big = format!("0f{:08X}", (-1.0e30f32).to_bits());
    let bfly = |reg: &str, op: &str| -> String {
        let mut t = String::new();
        for off in [16, 8, 4, 2, 1] {
            t += &format!("    shfl.sync.bfly.b32 %rt,{reg},{off},0x1f,0xffffffff;\n    {op}.f32 {reg},{reg},%rt;\n");
        }
        t
    };
    let mut s = String::new();
    s += HDR_SM80;
    s += "\n";
    s += &format!(
        ".visible .entry {PAGED_ATTN_INT8_ENTRY}(\n\
        \x20   .param .u64 pQ,\n\
        \x20   .param .u64 pK,\n\
        \x20   .param .u64 pV,\n\
        \x20   .param .u64 pKsc,\n\
        \x20   .param .u64 pVsc,\n\
        \x20   .param .u64 pO,\n\
        \x20   .param .u64 pBT,\n\
        \x20   .param .u64 pCL,\n\
        \x20   .param .f32 pScale,\n\
        \x20   .param .u32 pBcap,\n\
        \x20   .param .u32 pHeads,\n\
        \x20   .param .u32 pKvHeads,\n\
        \x20   .param .u32 pBsz,\n\
        \x20   .param .u32 pNblk,\n\
        \x20   .param .u32 pMbps,\n\
        \x20   .param .u32 pLayer\n)\n{{\n"
    );
    s += &format!("    .shared .f32 qsh[{}];\n", w * hd);
    s += &format!("    .reg .f32 %acc<{hd}>;\n");
    s += "    .reg .f32 %score,%m,%l,%newm,%p,%corr,%kf,%vf,%qv,%invl,%scale,%factor,%M,%L,%t0,%rt,%scK,%scV,%pv;\n";
    s += "    .reg .b32 %tix,%warp,%lane,%gid,%slot,%head,%ctx,%t,%logical,%off,%phys,%D,%qidx,%nq,%tmp,%bcap,%heads,%bsz,%nblk,%mbps,%layer,%e,%es,%ki,%dd,%kvheads,%kvhead,%grp;\n";
    s += "    .reg .b64 %Q,%K,%V,%Ksc,%Vsc,%O,%BT,%CL,%addr,%qrow,%obase,%kbase,%vbase,%offb,%qshw;\n";
    s += "    .reg .pred %p0,%p1,%p2;\n";
    s += "    ld.param.u64 %Q,[pQ];     cvta.to.global.u64 %Q,%Q;\n";
    s += "    ld.param.u64 %K,[pK];     cvta.to.global.u64 %K,%K;\n";
    s += "    ld.param.u64 %V,[pV];     cvta.to.global.u64 %V,%V;\n";
    s += "    ld.param.u64 %Ksc,[pKsc]; cvta.to.global.u64 %Ksc,%Ksc;\n";
    s += "    ld.param.u64 %Vsc,[pVsc]; cvta.to.global.u64 %Vsc,%Vsc;\n";
    s += "    ld.param.u64 %O,[pO];     cvta.to.global.u64 %O,%O;\n";
    s += "    ld.param.u64 %BT,[pBT];   cvta.to.global.u64 %BT,%BT;\n";
    s += "    ld.param.u64 %CL,[pCL];   cvta.to.global.u64 %CL,%CL;\n";
    s += "    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u32 %bcap,[pBcap];\n    ld.param.u32 %heads,[pHeads];\n    ld.param.u32 %kvheads,[pKvHeads];\n    ld.param.u32 %bsz,[pBsz];\n";
    s += "    ld.param.u32 %nblk,[pNblk];\n    ld.param.u32 %mbps,[pMbps];\n    ld.param.u32 %layer,[pLayer];\n";
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warp,%tix,5;\n    and.b32 %lane,%tix,31;\n";
    s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mad.lo.s32 %gid,%tmp,{w},%warp;\n");
    s += "    mul.lo.s32 %nq,%bcap,%heads;\n    setp.ge.u32 %p0,%gid,%nq;\n    @%p0 bra DONE;\n";
    s += "    div.u32 %slot,%gid,%heads;\n    mul.lo.s32 %tmp,%slot,%heads;\n    sub.u32 %head,%gid,%tmp;\n";
    // GQA head mapping, exactly as in the f16 twin: kvhead = q_head / (q_heads / kv_heads).
    s += "    div.u32 %grp,%heads,%kvheads;\n    div.u32 %kvhead,%head,%grp;\n";
    s += "    mul.wide.u32 %offb,%slot,4;\n    add.s64 %addr,%CL,%offb;\n    ld.global.u32 %ctx,[%addr];\n";
    s += &format!("    mul.lo.s32 %D,%heads,{hd};\n    mul.lo.s32 %qidx,%slot,%D;\n    mul.lo.s32 %tmp,%head,{hd};\n    add.u32 %qidx,%qidx,%tmp;\n");
    s += "    mul.wide.u32 %offb,%qidx,4;\n    add.s64 %qrow,%Q,%offb;\n    add.s64 %obase,%O,%offb;\n";
    s += &format!("    mov.u64 %qshw,qsh;\n    mul.wide.u32 %offb,%warp,{};\n    add.s64 %qshw,%qshw,%offb;\n", hd * 4);
    s +=
        &format!("    mov.u32 %dd,%lane;\nQL:\n    setp.ge.u32 %p1,%dd,{hd};\n    @%p1 bra QLE;\n");
    s += "    mul.wide.u32 %offb,%dd,4;\n    add.s64 %addr,%qrow,%offb;\n    ld.global.f32 %qv,[%addr];\n";
    s += "    add.s64 %addr,%qshw,%offb;\n    st.shared.f32 [%addr],%qv;\n    add.u32 %dd,%dd,32;\n    bra QL;\nQLE:\n";
    s += "    bar.warp.sync 0xffffffff;\n";
    s += &format!("    mov.f32 %m,{neg_big};\n    mov.f32 %l,0f00000000;\n");
    for d in 0..hd {
        s += &format!("    mov.f32 %acc{d},0f00000000;\n");
    }
    s += "    mov.u32 %t,%lane;\nLOOP:\n    setp.ge.u32 %p1,%t,%ctx;\n    @%p1 bra ENDLOOP;\n";
    s += "    div.u32 %logical,%t,%bsz;\n    mul.lo.s32 %off,%logical,%bsz;\n    sub.u32 %off,%t,%off;\n";
    s += "    mul.lo.s32 %tmp,%slot,%mbps;\n    add.u32 %tmp,%tmp,%logical;\n    mul.wide.u32 %offb,%tmp,4;\n    add.s64 %addr,%BT,%offb;\n    ld.global.u32 %phys,[%addr];\n";
    // es = (token, KV head) linear index (scale slab) ; e = es*hd (int8 element index). Both axes are
    // KV-headed — the scale slab holds one entry per cached (token, kv_head), not per query head.
    s += "    mul.lo.s32 %e,%layer,%nblk;\n    add.u32 %e,%e,%phys;\n    mul.lo.s32 %e,%e,%bsz;\n    add.u32 %e,%e,%off;\n";
    s += "    mul.lo.s32 %e,%e,%kvheads;\n    add.u32 %e,%e,%kvhead;\n    mov.u32 %es,%e;\n";
    s += &format!("    mul.lo.s32 %e,%e,{hd};\n");
    // Per-token-head dequant scales.
    s += "    mul.wide.u32 %offb,%es,4;\n    add.s64 %addr,%Ksc,%offb;\n    ld.global.f32 %scK,[%addr];\n    add.s64 %addr,%Vsc,%offb;\n    ld.global.f32 %scV,[%addr];\n";
    // int8 K/V bases (1 byte/elem).
    s += "    mul.wide.u32 %offb,%e,1;\n    add.s64 %kbase,%K,%offb;\n    add.s64 %vbase,%V,%offb;\n";
    // raw int dot, then scale by scaleK and the attention scale.
    s += "    mov.f32 %score,0f00000000;\n";
    for d in 0..hd {
        s += &format!("    ld.shared.f32 %qv,[%qshw+{}];\n    ld.global.s8 %ki,[%kbase+{}];\n    cvt.rn.f32.s32 %kf,%ki;\n    fma.rn.f32 %score,%qv,%kf,%score;\n", d * 4, d);
    }
    s += "    mul.f32 %score,%score,%scK;\n    mul.f32 %score,%score,%scale;\n";
    s += "    max.f32 %newm,%m,%score;\n";
    s += &format!(
        "    sub.f32 %t0,%m,%newm;\n    mul.f32 %t0,%t0,{log2e};\n    ex2.approx.f32 %corr,%t0;\n"
    );
    s += &format!(
        "    sub.f32 %t0,%score,%newm;\n    mul.f32 %t0,%t0,{log2e};\n    ex2.approx.f32 %p,%t0;\n"
    );
    s += "    mul.f32 %l,%l,%corr;\n    add.f32 %l,%l,%p;\n";
    s += "    mul.f32 %pv,%p,%scV;\n";
    for d in 0..hd {
        s += &format!("    ld.global.s8 %ki,[%vbase+{}];\n    cvt.rn.f32.s32 %vf,%ki;\n    mul.f32 %acc{d},%acc{d},%corr;\n    fma.rn.f32 %acc{d},%pv,%vf,%acc{d};\n", d);
    }
    s += "    mov.f32 %m,%newm;\n    add.u32 %t,%t,32;\n    bra LOOP;\nENDLOOP:\n";
    s += "    mov.f32 %M,%m;\n";
    s += &bfly("%M", "max");
    s += &format!(
        "    sub.f32 %t0,%m,%M;\n    mul.f32 %t0,%t0,{log2e};\n    ex2.approx.f32 %factor,%t0;\n"
    );
    s += "    mul.f32 %l,%l,%factor;\n";
    for d in 0..hd {
        s += &format!("    mul.f32 %acc{d},%acc{d},%factor;\n");
    }
    s += "    mov.f32 %L,%l;\n";
    s += &bfly("%L", "add");
    for d in 0..hd {
        s += &bfly(&format!("%acc{d}"), "add");
    }
    s += "    rcp.rn.f32 %invl,%L;\n    setp.gt.f32 %p2,%L,0f00000000;\n    selp.f32 %invl,%invl,0f00000000,%p2;\n";
    s += "    setp.ne.u32 %p0,%lane,0;\n    @%p0 bra DONE;\n";
    for d in 0..hd {
        s += &format!(
            "    mul.f32 %t0,%acc{d},%invl;\n    st.global.f32 [%obase+{}],%t0;\n",
            d * 4
        );
    }
    s += "DONE:\n    ret;\n}\n";
    s
}

/// Launch the **int8-KV** paged decode-attention kernel. Like [`launch_paged_attn_decode`] but `k_d`/`v_d`
/// are `int8` slabs and `ksc_d`/`vsc_d` the per-(token, head) f32 scale slabs (`cfg.scale_slab_elems()`
/// each). `scale` is `1/sqrt(head_dim)`. Full-overwrite of every active slot's output row.
///
/// The multi-head entry point; a grouped-query model wants
/// [`launch_paged_attn_decode_int8_gqa`], of which this is the `GqaConfig::mha(*cfg)` case.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn launch_paged_attn_decode_int8(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    q_d: &CudaSlice<f32>,
    k_d: &CudaSlice<i8>,
    v_d: &CudaSlice<i8>,
    ksc_d: &CudaSlice<f32>,
    vsc_d: &CudaSlice<f32>,
    out_d: &mut CudaSlice<f32>,
    bt_d: &CudaSlice<u32>,
    cl_d: &CudaSlice<u32>,
    cfg: &KvConfig,
    layer: usize,
    bcap: usize,
    scale: f32,
) -> Result<(), DriverError> {
    launch_paged_attn_decode_int8_gqa(
        stream,
        func,
        q_d,
        k_d,
        v_d,
        ksc_d,
        vsc_d,
        out_d,
        bt_d,
        cl_d,
        &GqaConfig::mha(*cfg),
        layer,
        bcap,
        scale,
    )
}

/// Launch the **grouped-query int8-KV** paged decode-attention kernel — the int8 twin of
/// [`launch_paged_attn_decode_gqa`]. `q_d`/`out_d` are query-headed `[bcap, gqa.q_dim()]`; the int8
/// value slabs and both f32 scale slabs are KV-headed, and query head `h` dequants KV head
/// `h / gqa.group_size()` with that head's own per-(token, head) scale.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn launch_paged_attn_decode_int8_gqa(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    q_d: &CudaSlice<f32>,
    k_d: &CudaSlice<i8>,
    v_d: &CudaSlice<i8>,
    ksc_d: &CudaSlice<f32>,
    vsc_d: &CudaSlice<f32>,
    out_d: &mut CudaSlice<f32>,
    bt_d: &CudaSlice<u32>,
    cl_d: &CudaSlice<u32>,
    gqa: &GqaConfig,
    layer: usize,
    bcap: usize,
    scale: f32,
) -> Result<(), DriverError> {
    let cfg = gqa.kv();
    debug_assert_eq!(
        q_d.len(),
        bcap * gqa.q_dim(),
        "q must be [bcap, q_heads*head_dim] (query-headed, not KV-headed)"
    );
    debug_assert_eq!(
        out_d.len(),
        bcap * gqa.q_dim(),
        "out must be [bcap, q_heads*head_dim] (query-headed, not KV-headed)"
    );
    debug_assert_eq!(
        k_d.len(),
        cfg.slab_elems(),
        "int8 k slab must be cfg.slab_elems()"
    );
    debug_assert_eq!(
        v_d.len(),
        cfg.slab_elems(),
        "int8 v slab must be cfg.slab_elems()"
    );
    debug_assert_eq!(
        ksc_d.len(),
        cfg.scale_slab_elems(),
        "k scale slab must be cfg.scale_slab_elems()"
    );
    debug_assert_eq!(
        vsc_d.len(),
        cfg.scale_slab_elems(),
        "v scale slab must be cfg.scale_slab_elems()"
    );
    debug_assert_eq!(
        bt_d.len(),
        bcap * cfg.max_blocks_per_seq,
        "block table must be [bcap, cfg.max_blocks_per_seq]"
    );
    debug_assert_eq!(cl_d.len(), bcap, "context lengths must be one u32 per slot");
    // One warp per (slot, QUERY head).
    let nq = (bcap * gqa.q_heads()) as u32;
    let cfg_launch = LaunchConfig {
        grid_dim: (nq.div_ceil(PAGED_ATTN_WARPS), 1, 1),
        block_dim: (32 * PAGED_ATTN_WARPS, 1, 1),
        shared_mem_bytes: 0,
    };
    let (heads, kvheads, bsz, nblk, mbps, layer_u) = (
        gqa.q_heads() as u32,
        gqa.kv_heads() as u32,
        cfg.block_size as u32,
        cfg.num_blocks as u32,
        cfg.max_blocks_per_seq as u32,
        layer as u32,
    );
    let bcap_u = bcap as u32;
    let mut b = stream.launch_builder(func);
    b.arg(q_d)
        .arg(k_d)
        .arg(v_d)
        .arg(ksc_d)
        .arg(vsc_d)
        .arg(out_d)
        .arg(bt_d)
        .arg(cl_d)
        .arg(&scale);
    b.arg(&bcap_u)
        .arg(&heads)
        .arg(&kvheads)
        .arg(&bsz)
        .arg(&nblk)
        .arg(&mbps)
        .arg(&layer_u);
    // SAFETY: the pushed arguments match `PAGED_ATTN_INT8_ENTRY`'s parameter list in order and width
    // (eight .u64 pointers, one .f32, **seven** .u32 — `pKvHeads` between `pHeads` and `pBsz`; see
    // `paged_attn_decode_int8_ptx`, textually pinned by
    // `attention_entry_param_counts_match_the_launcher_push_lists`), and `func` was loaded from PTX
    // generated for `cfg.head_dim`. Every buffer is at least as long as the largest index the kernel
    // can produce for `bcap` slots at `layer`: `%e` reproduces `KvConfig::elem_offset` and `%es`
    // `KvConfig::scale_offset`, both at the **KV** head axis (bounded by `slab_elems()` /
    // `scale_slab_elems()`); the table/lengths/query-headed rows are bounded by the shapes asserted
    // above, which the caller guarantees and the debug asserts check at the call site. `GqaConfig`'s
    // constructor is what makes the device `div.u32 %grp,%heads,%kvheads` exact and non-zero.
    unsafe { b.launch(cfg_launch)? };
    Ok(())
}

/// **Host reference quantizer**: per-slot f32 K/V (`[ctx, kv_heads, head_dim]` row-major) → int8
/// values plus per-(token, KV head) f32 scales (`scale = max_d |x| / 127`, `0 → 1`),
/// `int8 = round(x / scale)` clamped. The device int8 cache stores exactly this;
/// [`paged_attn_decode_int8_ptx`] dequants `int8 · scale`. `heads` is a **KV**-head count: the cache
/// and its scales have no query-head axis.
pub fn quantize_kv_int8(
    slot: &[f32],
    ctx: usize,
    heads: usize,
    head_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; ctx * heads * head_dim];
    let mut sc = vec![0f32; ctx * heads];
    for t in 0..ctx {
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            let amax = (0..head_dim).fold(0f32, |m, d| m.max(slot[base + d].abs()));
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            sc[t * heads + h] = scale;
            for d in 0..head_dim {
                let q_val = (slot[base + d] / scale).round().clamp(-127.0, 127.0);
                q[base + d] = q_val as i8;
            }
        }
    }
    (q, sc)
}

/// PTX entry name for the KV-append (scatter) kernel.
pub const KV_APPEND_ENTRY: &str = "kv_append";

/// Generate the **KV-append** PTX: scatter each slot's just-projected new token K and V (f32
/// `[bcap, kv_heads*head_dim]`) into the paged f16 cache at the slot's write position, through its
/// block table. One thread per `(slot, channel)` element. `head_dim` is a runtime param (no unroll
/// needed — it's a pure scatter). This is the device twin of
/// [`crate::paged_kv::BlockManager::append`]'s `(phys, off)` address.
///
/// **GQA needs no change here, and that is a fact worth stating rather than leaving to inference.**
/// The append writes the K/V *projection*, which a grouped-query model emits at `kv_heads*head_dim`
/// wide — so `pHeads` is a KV-head count, `pHd*pHeads` is the source row stride, and the query-head
/// count never enters this kernel at all. Passing a query-headed `knew`/`vnew` is therefore the
/// GQA bug this kernel cannot detect for itself; [`launch_kv_append_gqa`] exists to assert the shape
/// at the seam where both counts are still in scope.
pub fn kv_append_ptx() -> String {
    let mut s = String::new();
    s += HDR_SM80;
    s += "\n";
    s += &format!(
        ".visible .entry {KV_APPEND_ENTRY}(\n\
        \x20   .param .u64 pKnew,\n\
        \x20   .param .u64 pVnew,\n\
        \x20   .param .u64 pK,\n\
        \x20   .param .u64 pV,\n\
        \x20   .param .u64 pBT,\n\
        \x20   .param .u64 pWPos,\n\
        \x20   .param .u64 pAct,\n\
        \x20   .param .u32 pBcap,\n\
        \x20   .param .u32 pHeads,\n\
        \x20   .param .u32 pHd,\n\
        \x20   .param .u32 pBsz,\n\
        \x20   .param .u32 pNblk,\n\
        \x20   .param .u32 pMbps,\n\
        \x20   .param .u32 pLayer\n)\n{{\n"
    );
    s += "    .reg .pred %p0;\n";
    s += "    .reg .b16 %hk,%hv;\n";
    s += "    .reg .f32 %fk,%fv;\n";
    s += "    .reg .b32 %gid,%b,%d,%hh,%dh,%D,%total,%pos,%logical,%off,%phys,%e,%tmp,%bcap,%heads,%hd,%bsz,%nblk,%mbps,%layer;\n";
    s += "    .reg .b64 %Knew,%Vnew,%K,%V,%BT,%WP,%ACT,%addr,%o64;\n";
    s += "    ld.param.u64 %Knew,[pKnew]; cvta.to.global.u64 %Knew,%Knew;\n";
    s += "    ld.param.u64 %Vnew,[pVnew]; cvta.to.global.u64 %Vnew,%Vnew;\n";
    s += "    ld.param.u64 %K,[pK];       cvta.to.global.u64 %K,%K;\n";
    s += "    ld.param.u64 %V,[pV];       cvta.to.global.u64 %V,%V;\n";
    s += "    ld.param.u64 %BT,[pBT];     cvta.to.global.u64 %BT,%BT;\n";
    s += "    ld.param.u64 %WP,[pWPos];   cvta.to.global.u64 %WP,%WP;\n";
    s += "    ld.param.u64 %ACT,[pAct];   cvta.to.global.u64 %ACT,%ACT;\n";
    s += "    ld.param.u32 %bcap,[pBcap];\n    ld.param.u32 %heads,[pHeads];\n    ld.param.u32 %hd,[pHd];\n";
    s += "    ld.param.u32 %bsz,[pBsz];\n    ld.param.u32 %nblk,[pNblk];\n    ld.param.u32 %mbps,[pMbps];\n    ld.param.u32 %layer,[pLayer];\n";
    // gid = ctaid.x*ntid.x + tid.x ; D = heads*hd ; total = bcap*D ; bail if gid>=total.
    s += "    mov.u32 %tmp,%ctaid.x;\n    mov.u32 %gid,%ntid.x;\n    mov.u32 %b,%tid.x;\n    mad.lo.s32 %gid,%tmp,%gid,%b;\n";
    s += "    mul.lo.s32 %D,%heads,%hd;\n    mul.lo.s32 %total,%bcap,%D;\n";
    s += "    setp.ge.u32 %p0,%gid,%total;\n    @%p0 bra DONE;\n";
    // b = gid/D ; d = gid - b*D ; hh = d/hd ; dh = d - hh*hd.
    s += "    div.u32 %b,%gid,%D;\n    mul.lo.s32 %tmp,%b,%D;\n    sub.u32 %d,%gid,%tmp;\n";
    // Skip inactive (padding) rows: a free slot pads its block table with 0, so appending it would
    // scatter into block 0 (a live block). Act[b]==0 ⇒ this row carries no request ⇒ no write.
    s += "    mul.wide.u32 %o64,%b,4;\n    add.s64 %addr,%ACT,%o64;\n    ld.global.u32 %tmp,[%addr];\n    setp.eq.u32 %p0,%tmp,0;\n    @%p0 bra DONE;\n";
    s += "    div.u32 %hh,%d,%hd;\n    mul.lo.s32 %tmp,%hh,%hd;\n    sub.u32 %dh,%d,%tmp;\n";
    // pos = WP[b] ; logical = pos/bsz ; off = pos - logical*bsz.
    s += "    mul.wide.u32 %o64,%b,4;\n    add.s64 %addr,%WP,%o64;\n    ld.global.u32 %pos,[%addr];\n";
    s += "    div.u32 %logical,%pos,%bsz;\n    mul.lo.s32 %tmp,%logical,%bsz;\n    sub.u32 %off,%pos,%tmp;\n";
    // phys = BT[b*mbps + logical].
    s += "    mul.lo.s32 %tmp,%b,%mbps;\n    add.u32 %tmp,%tmp,%logical;\n    mul.wide.u32 %o64,%tmp,4;\n    add.s64 %addr,%BT,%o64;\n    ld.global.u32 %phys,[%addr];\n";
    // e = ((((layer*nblk)+phys)*bsz+off)*heads+hh)*hd + dh.
    s += "    mul.lo.s32 %e,%layer,%nblk;\n    add.u32 %e,%e,%phys;\n    mul.lo.s32 %e,%e,%bsz;\n    add.u32 %e,%e,%off;\n";
    s += "    mul.lo.s32 %e,%e,%heads;\n    add.u32 %e,%e,%hh;\n    mul.lo.s32 %e,%e,%hd;\n    add.u32 %e,%e,%dh;\n";
    // load Knew[gid]/Vnew[gid], narrow to f16, store at slab[e].
    s += "    mul.wide.u32 %o64,%gid,4;\n    add.s64 %addr,%Knew,%o64;\n    ld.global.f32 %fk,[%addr];\n    add.s64 %addr,%Vnew,%o64;\n    ld.global.f32 %fv,[%addr];\n";
    s += "    cvt.rn.f16.f32 %hk,%fk;\n    cvt.rn.f16.f32 %hv,%fv;\n";
    s += "    mul.wide.u32 %o64,%e,2;\n    add.s64 %addr,%K,%o64;\n    st.global.b16 [%addr],%hk;\n    add.s64 %addr,%V,%o64;\n    st.global.b16 [%addr],%hv;\n";
    s += "DONE:\n    ret;\n}\n";
    s
}

/// Threads per CTA for the KV-append scatter.
pub const KV_APPEND_BLOCK: u32 = 256;

/// Launch the KV-append scatter for **one layer**: write the new token K/V (`knew_d`/`vnew_d`, f32
/// `[bcap, kv_heads*head_dim]`) into the f16 cache slabs at each slot's `wpos_d[slot]` position, via
/// the block table. (`cfg.heads` is the KV-head count; see [`launch_kv_append_gqa`] for the
/// grouped-query seam that asserts the projection width.)
/// `wpos_d[slot]` must be the slot's pre-append position and its block must already be reserved (the
/// host [`BlockManager::append`](crate::paged_kv::BlockManager::append) does both). `active_d[slot]`
/// (u32 0/1) gates the write: a `0` (free/padding) slot is skipped entirely — its block table pads to
/// block 0, so writing it would corrupt a live block. Full-overwrite of one f16 element per active
/// channel ⇒ safe on pooled (uninit) `knew/vnew`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn launch_kv_append(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    knew_d: &CudaSlice<f32>,
    vnew_d: &CudaSlice<f32>,
    k_d: &mut CudaSlice<half::f16>,
    v_d: &mut CudaSlice<half::f16>,
    bt_d: &CudaSlice<u32>,
    wpos_d: &CudaSlice<u32>,
    active_d: &CudaSlice<u32>,
    cfg: &KvConfig,
    layer: usize,
    bcap: usize,
) -> Result<(), DriverError> {
    debug_assert_eq!(
        knew_d.len(),
        bcap * cfg.heads * cfg.head_dim,
        "knew must be [bcap, heads*head_dim]"
    );
    debug_assert_eq!(
        vnew_d.len(),
        bcap * cfg.heads * cfg.head_dim,
        "vnew must be [bcap, heads*head_dim]"
    );
    debug_assert_eq!(
        k_d.len(),
        cfg.slab_elems(),
        "k slab must be cfg.slab_elems()"
    );
    debug_assert_eq!(
        v_d.len(),
        cfg.slab_elems(),
        "v slab must be cfg.slab_elems()"
    );
    debug_assert_eq!(
        bt_d.len(),
        bcap * cfg.max_blocks_per_seq,
        "block table must be [bcap, cfg.max_blocks_per_seq]"
    );
    debug_assert_eq!(
        wpos_d.len(),
        bcap,
        "write positions must be one u32 per slot"
    );
    debug_assert_eq!(active_d.len(), bcap, "active mask must be one u32 per slot");
    let total = (bcap * cfg.heads * cfg.head_dim) as u32;
    let launch = LaunchConfig {
        grid_dim: (total.div_ceil(KV_APPEND_BLOCK), 1, 1),
        block_dim: (KV_APPEND_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bcap_u, heads, hd, bsz, nblk, mbps, layer_u) = (
        bcap as u32,
        cfg.heads as u32,
        cfg.head_dim as u32,
        cfg.block_size as u32,
        cfg.num_blocks as u32,
        cfg.max_blocks_per_seq as u32,
        layer as u32,
    );
    let mut b = stream.launch_builder(func);
    b.arg(knew_d)
        .arg(vnew_d)
        .arg(k_d)
        .arg(v_d)
        .arg(bt_d)
        .arg(wpos_d)
        .arg(active_d);
    b.arg(&bcap_u)
        .arg(&heads)
        .arg(&hd)
        .arg(&bsz)
        .arg(&nblk)
        .arg(&mbps)
        .arg(&layer_u);
    // SAFETY: the pushed arguments match `KV_APPEND_ENTRY`'s parameter list in order and width (seven
    // .u64 pointers, seven .u32 — see `kv_append_ptx`). Every buffer is at least as long as the largest
    // index the kernel can produce for `bcap` slots at `layer`: the store index `%e` reproduces
    // `KvConfig::elem_offset` (bounded by `slab_elems()`), the load index is the thread's own `gid <
    // bcap*heads*hd`, and `BT`/`WP`/`ACT` are bounded by `bcap * max_blocks_per_seq` and `bcap`. That,
    // plus each active slot's write position already having a reserved block (the caller's contract,
    // established by `BlockManager::append`), is what keeps the scatter inside the slabs. The debug
    // asserts above check the lengths at the call site.
    unsafe { b.launch(launch)? };
    Ok(())
}

/// [`launch_kv_append`] for a **grouped-query** model. The append is KV-headed end to end, so this is
/// `launch_kv_append(.., gqa.kv(), ..)` plus the one assertion that catches the GQA mistake the kernel
/// cannot: `knew_d`/`vnew_d` must be `[bcap, gqa.kv_dim()]` — the **K/V projection** width — and never
/// `[bcap, gqa.q_dim()]`. Getting that wrong is silent, not a fault: a query-headed source row is `g`
/// times too long, so every slot past the first reads its neighbour's channels and the cache fills
/// with plausible garbage that no tolerance gate downstream can attribute.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn launch_kv_append_gqa(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    knew_d: &CudaSlice<f32>,
    vnew_d: &CudaSlice<f32>,
    k_d: &mut CudaSlice<half::f16>,
    v_d: &mut CudaSlice<half::f16>,
    bt_d: &CudaSlice<u32>,
    wpos_d: &CudaSlice<u32>,
    active_d: &CudaSlice<u32>,
    gqa: &GqaConfig,
    layer: usize,
    bcap: usize,
) -> Result<(), DriverError> {
    debug_assert_eq!(
        knew_d.len(),
        bcap * gqa.kv_dim(),
        "knew must be [bcap, kv_heads*head_dim] — the K/V projection width, not the hidden size"
    );
    debug_assert_eq!(
        vnew_d.len(),
        bcap * gqa.kv_dim(),
        "vnew must be [bcap, kv_heads*head_dim] — the K/V projection width, not the hidden size"
    );
    launch_kv_append(
        stream,
        func,
        knew_d,
        vnew_d,
        k_d,
        v_d,
        bt_d,
        wpos_d,
        active_d,
        gqa.kv(),
        layer,
        bcap,
    )
}

/// PTX entry name for the **int8** KV-append (quantize + scatter) kernel.
pub const KV_APPEND_INT8_ENTRY: &str = "kv_append_int8";

/// `(slot, head)` pairs per CTA for the int8 append — one **warp** each (block_dim = `32 * WARPS`),
/// because quantization needs a per-(slot, head) amax *reduction* before any value can be written
/// (the f16 append's thread-per-element scatter has no reduction and needs none).
pub const KV_APPEND_INT8_WARPS: u32 = 4;

/// Generate the **int8 KV-append** PTX: quantize each active slot's just-projected new-token K and V
/// (f32 `[bcap, D]`) into the int8 cache at the slot's write position, through its block table — the
/// device twin of the host [`quantize_kv_int8`] scheme, feeding [`paged_attn_decode_int8_ptx`]. One
/// **warp per `(slot, head)`**: the 32 lanes stride the head_dim computing a per-lane `|x|` max, a
/// fixed `shfl.sync.bfly` butterfly merges it (f32 max is order-independent ⇒ bit-equal to the host
/// fold), lane 0 stores the two f32 scales (`amax/127`, `0 → 1`, IEEE `div.rn` — bit-equal to the
/// host `/`), then the lanes quantize and scatter `int8 = clamp(rni(x / scale), ±127)`.
///
/// **Rounding note:** the device rounds ties-to-even (`cvt.rni`) where the host [`quantize_kv_int8`]
/// rounds half-away-from-zero — they can differ by 1 int8 LSB only at exact-`.5` quotients
/// (immaterial under the dequant tolerance; the round-trip gate's host mirror uses
/// `round_ties_even` so *that* comparison is exact). Inactive (`active_d == 0`) slots are skipped
/// whole — the padding-block-0 guard, exactly as in the f16 append. `head_dim` stays a runtime param
/// (strided lane loops need no unroll), so one PTX serves hd = 64 and 128.
pub fn kv_append_int8_ptx() -> String {
    let w = KV_APPEND_INT8_WARPS as usize;
    // f32 literals: 127.0, -127.0, 1.0.
    let (p127, n127, one) = ("0f42FE0000", "0fC2FE0000", "0f3F800000");
    // Warp butterfly all-reduce of one f32 register under `op` (every lane ends with the result).
    let bfly = |reg: &str, op: &str| -> String {
        let mut t = String::new();
        for off in [16, 8, 4, 2, 1] {
            t += &format!("    shfl.sync.bfly.b32 %rt,{reg},{off},0x1f,0xffffffff;\n    {op}.f32 {reg},{reg},%rt;\n");
        }
        t
    };
    let mut s = String::new();
    s += HDR_SM80;
    s += "\n";
    s += &format!(
        ".visible .entry {KV_APPEND_INT8_ENTRY}(\n\
        \x20   .param .u64 pKnew,\n\
        \x20   .param .u64 pVnew,\n\
        \x20   .param .u64 pK,\n\
        \x20   .param .u64 pV,\n\
        \x20   .param .u64 pKsc,\n\
        \x20   .param .u64 pVsc,\n\
        \x20   .param .u64 pBT,\n\
        \x20   .param .u64 pWPos,\n\
        \x20   .param .u64 pAct,\n\
        \x20   .param .u32 pBcap,\n\
        \x20   .param .u32 pHeads,\n\
        \x20   .param .u32 pHd,\n\
        \x20   .param .u32 pBsz,\n\
        \x20   .param .u32 pNblk,\n\
        \x20   .param .u32 pMbps,\n\
        \x20   .param .u32 pLayer\n)\n{{\n"
    );
    s += "    .reg .pred %p0,%p1;\n";
    s += "    .reg .f32 %x,%q,%amk,%amv,%sck,%scv,%rt;\n";
    s += "    .reg .b32 %tix,%warp,%lane,%gid,%slot,%head,%pos,%logical,%off,%phys,%es,%e,%src,%dd,%tmp,%nq,%qi,%bcap,%heads,%hd,%bsz,%nblk,%mbps,%layer;\n";
    s += "    .reg .b64 %Knew,%Vnew,%K,%V,%Ksc,%Vsc,%BT,%WP,%ACT,%addr,%offb,%kdst,%vdst,%ksrc,%vsrc;\n";
    s += "    ld.param.u64 %Knew,[pKnew]; cvta.to.global.u64 %Knew,%Knew;\n";
    s += "    ld.param.u64 %Vnew,[pVnew]; cvta.to.global.u64 %Vnew,%Vnew;\n";
    s += "    ld.param.u64 %K,[pK];       cvta.to.global.u64 %K,%K;\n";
    s += "    ld.param.u64 %V,[pV];       cvta.to.global.u64 %V,%V;\n";
    s += "    ld.param.u64 %Ksc,[pKsc];   cvta.to.global.u64 %Ksc,%Ksc;\n";
    s += "    ld.param.u64 %Vsc,[pVsc];   cvta.to.global.u64 %Vsc,%Vsc;\n";
    s += "    ld.param.u64 %BT,[pBT];     cvta.to.global.u64 %BT,%BT;\n";
    s += "    ld.param.u64 %WP,[pWPos];   cvta.to.global.u64 %WP,%WP;\n";
    s += "    ld.param.u64 %ACT,[pAct];   cvta.to.global.u64 %ACT,%ACT;\n";
    s += "    ld.param.u32 %bcap,[pBcap];\n    ld.param.u32 %heads,[pHeads];\n    ld.param.u32 %hd,[pHd];\n";
    s += "    ld.param.u32 %bsz,[pBsz];\n    ld.param.u32 %nblk,[pNblk];\n    ld.param.u32 %mbps,[pMbps];\n    ld.param.u32 %layer,[pLayer];\n";
    // gid = ctaid.x*WARPS + warp ; one warp per (slot, head) ⇒ every lane of a warp shares the gid,
    // so the early exits below are warp-uniform (shfl full-mask stays safe).
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warp,%tix,5;\n    and.b32 %lane,%tix,31;\n";
    s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mad.lo.s32 %gid,%tmp,{w},%warp;\n");
    s += "    mul.lo.s32 %nq,%bcap,%heads;\n    setp.ge.u32 %p0,%gid,%nq;\n    @%p0 bra DONE;\n";
    s += "    div.u32 %slot,%gid,%heads;\n    mul.lo.s32 %tmp,%slot,%heads;\n    sub.u32 %head,%gid,%tmp;\n";
    // Skip inactive (padding) rows — a free slot's table pads to block 0 (a live block).
    s += "    mul.wide.u32 %offb,%slot,4;\n    add.s64 %addr,%ACT,%offb;\n    ld.global.u32 %tmp,[%addr];\n    setp.eq.u32 %p0,%tmp,0;\n    @%p0 bra DONE;\n";
    // pos = WP[slot] ; logical = pos/bsz ; off = pos - logical*bsz ; phys = BT[slot*mbps + logical].
    s += "    add.s64 %addr,%WP,%offb;\n    ld.global.u32 %pos,[%addr];\n";
    s += "    div.u32 %logical,%pos,%bsz;\n    mul.lo.s32 %tmp,%logical,%bsz;\n    sub.u32 %off,%pos,%tmp;\n";
    s += "    mul.lo.s32 %tmp,%slot,%mbps;\n    add.u32 %tmp,%tmp,%logical;\n    mul.wide.u32 %offb,%tmp,4;\n    add.s64 %addr,%BT,%offb;\n    ld.global.u32 %phys,[%addr];\n";
    // es = (((layer*nblk + phys)*bsz + off)*heads + head)  (scale-slab index) ; e = es*hd ;
    // src = gid*hd  (the (slot, head) row base in the [bcap, D] f32 input).
    s += "    mul.lo.s32 %es,%layer,%nblk;\n    add.u32 %es,%es,%phys;\n    mul.lo.s32 %es,%es,%bsz;\n    add.u32 %es,%es,%off;\n";
    s += "    mul.lo.s32 %es,%es,%heads;\n    add.u32 %es,%es,%head;\n    mul.lo.s32 %e,%es,%hd;\n    mul.lo.s32 %src,%gid,%hd;\n";
    s += "    mul.wide.u32 %offb,%src,4;\n    add.s64 %ksrc,%Knew,%offb;\n    add.s64 %vsrc,%Vnew,%offb;\n";
    s += "    mul.wide.u32 %offb,%e,1;\n    add.s64 %kdst,%K,%offb;\n    add.s64 %vdst,%V,%offb;\n";
    // Per-lane strided |x| maxima over the head_dim (fold starts at 0, as the host's does).
    s += "    mov.f32 %amk,0f00000000;\n    mov.f32 %amv,0f00000000;\n";
    s += "    mov.u32 %dd,%lane;\nAML:\n    setp.ge.u32 %p1,%dd,%hd;\n    @%p1 bra AME;\n";
    s += "    mul.wide.u32 %offb,%dd,4;\n";
    s += "    add.s64 %addr,%ksrc,%offb;\n    ld.global.f32 %x,[%addr];\n    abs.f32 %x,%x;\n    max.f32 %amk,%amk,%x;\n";
    s += "    add.s64 %addr,%vsrc,%offb;\n    ld.global.f32 %x,[%addr];\n    abs.f32 %x,%x;\n    max.f32 %amv,%amv,%x;\n";
    s += "    add.u32 %dd,%dd,32;\n    bra AML;\nAME:\n";
    s += &bfly("%amk", "max");
    s += &bfly("%amv", "max");
    // scale = amax > 0 ? amax/127 : 1  (IEEE div — bit-equal to the host `/`).
    s += &format!("    div.rn.f32 %sck,%amk,{p127};\n    setp.gt.f32 %p1,%amk,0f00000000;\n    selp.f32 %sck,%sck,{one},%p1;\n");
    s += &format!("    div.rn.f32 %scv,%amv,{p127};\n    setp.gt.f32 %p1,%amv,0f00000000;\n    selp.f32 %scv,%scv,{one},%p1;\n");
    // Lane 0 stores the two scales at the (token, head) scale-slab index.
    s += "    setp.ne.u32 %p1,%lane,0;\n    @%p1 bra QNT;\n";
    s += "    mul.wide.u32 %offb,%es,4;\n    add.s64 %addr,%Ksc,%offb;\n    st.global.f32 [%addr],%sck;\n    add.s64 %addr,%Vsc,%offb;\n    st.global.f32 [%addr],%scv;\n";
    s += "QNT:\n    mov.u32 %dd,%lane;\nQL:\n    setp.ge.u32 %p1,%dd,%hd;\n    @%p1 bra DONE;\n";
    s += "    mul.wide.u32 %offb,%dd,4;\n    add.s64 %addr,%ksrc,%offb;\n    ld.global.f32 %x,[%addr];\n";
    s += &format!("    div.rn.f32 %q,%x,%sck;\n    cvt.rni.f32.f32 %q,%q;\n    min.f32 %q,%q,{p127};\n    max.f32 %q,%q,{n127};\n    cvt.rzi.s32.f32 %qi,%q;\n");
    s += "    mul.wide.u32 %offb,%dd,1;\n    add.s64 %addr,%kdst,%offb;\n    st.global.s8 [%addr],%qi;\n";
    s += "    mul.wide.u32 %offb,%dd,4;\n    add.s64 %addr,%vsrc,%offb;\n    ld.global.f32 %x,[%addr];\n";
    s += &format!("    div.rn.f32 %q,%x,%scv;\n    cvt.rni.f32.f32 %q,%q;\n    min.f32 %q,%q,{p127};\n    max.f32 %q,%q,{n127};\n    cvt.rzi.s32.f32 %qi,%q;\n");
    s += "    mul.wide.u32 %offb,%dd,1;\n    add.s64 %addr,%vdst,%offb;\n    st.global.s8 [%addr],%qi;\n";
    s += "    add.u32 %dd,%dd,32;\n    bra QL;\n";
    s += "DONE:\n    ret;\n}\n";
    s
}

/// Launch the **int8** KV-append for one layer: quantize + scatter the new token K/V (`knew_d`/
/// `vnew_d`, f32 `[bcap, kv_heads*head_dim]`) into the int8 slabs and the per-(token, KV head) f32 scale slabs at each
/// slot's `wpos_d[slot]` position, via the block table. Same contract as [`launch_kv_append`]
/// (reserved positions, `active_d` mask gating every write); one warp per `(slot, head)`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn launch_kv_append_int8(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    knew_d: &CudaSlice<f32>,
    vnew_d: &CudaSlice<f32>,
    k_d: &mut CudaSlice<i8>,
    v_d: &mut CudaSlice<i8>,
    ksc_d: &mut CudaSlice<f32>,
    vsc_d: &mut CudaSlice<f32>,
    bt_d: &CudaSlice<u32>,
    wpos_d: &CudaSlice<u32>,
    active_d: &CudaSlice<u32>,
    cfg: &KvConfig,
    layer: usize,
    bcap: usize,
) -> Result<(), DriverError> {
    debug_assert_eq!(
        knew_d.len(),
        bcap * cfg.heads * cfg.head_dim,
        "knew must be [bcap, heads*head_dim]"
    );
    debug_assert_eq!(
        vnew_d.len(),
        bcap * cfg.heads * cfg.head_dim,
        "vnew must be [bcap, heads*head_dim]"
    );
    debug_assert_eq!(
        k_d.len(),
        cfg.slab_elems(),
        "int8 k slab must be cfg.slab_elems()"
    );
    debug_assert_eq!(
        v_d.len(),
        cfg.slab_elems(),
        "int8 v slab must be cfg.slab_elems()"
    );
    debug_assert_eq!(
        ksc_d.len(),
        cfg.scale_slab_elems(),
        "k scale slab must be cfg.scale_slab_elems()"
    );
    debug_assert_eq!(
        vsc_d.len(),
        cfg.scale_slab_elems(),
        "v scale slab must be cfg.scale_slab_elems()"
    );
    debug_assert_eq!(
        bt_d.len(),
        bcap * cfg.max_blocks_per_seq,
        "block table must be [bcap, cfg.max_blocks_per_seq]"
    );
    debug_assert_eq!(
        wpos_d.len(),
        bcap,
        "write positions must be one u32 per slot"
    );
    debug_assert_eq!(active_d.len(), bcap, "active mask must be one u32 per slot");
    let nq = (bcap * cfg.heads) as u32;
    let launch = LaunchConfig {
        grid_dim: (nq.div_ceil(KV_APPEND_INT8_WARPS), 1, 1),
        block_dim: (32 * KV_APPEND_INT8_WARPS, 1, 1),
        shared_mem_bytes: 0,
    };
    let (bcap_u, heads, hd, bsz, nblk, mbps, layer_u) = (
        bcap as u32,
        cfg.heads as u32,
        cfg.head_dim as u32,
        cfg.block_size as u32,
        cfg.num_blocks as u32,
        cfg.max_blocks_per_seq as u32,
        layer as u32,
    );
    let mut b = stream.launch_builder(func);
    b.arg(knew_d)
        .arg(vnew_d)
        .arg(k_d)
        .arg(v_d)
        .arg(ksc_d)
        .arg(vsc_d)
        .arg(bt_d)
        .arg(wpos_d)
        .arg(active_d);
    b.arg(&bcap_u)
        .arg(&heads)
        .arg(&hd)
        .arg(&bsz)
        .arg(&nblk)
        .arg(&mbps)
        .arg(&layer_u);
    // SAFETY: the pushed arguments match `KV_APPEND_INT8_ENTRY`'s parameter list in order and width
    // (nine .u64 pointers, seven .u32 — see `kv_append_int8_ptx`). Every buffer is at least as long as
    // the largest index the kernel can produce for `bcap` slots at `layer`: `%e` reproduces
    // `KvConfig::elem_offset` (bounded by `slab_elems()`), `%es` reproduces `KvConfig::scale_offset`
    // (bounded by `scale_slab_elems()`), the source row base is `gid*hd < bcap*heads*hd`, and
    // `BT`/`WP`/`ACT` are bounded by `bcap * max_blocks_per_seq` and `bcap`. As for the f16 append, the
    // caller also guarantees every active slot's write position has a reserved block. The debug asserts
    // above check the lengths at the call site.
    unsafe { b.launch(launch)? };
    Ok(())
}

/// [`launch_kv_append_int8`] for a **grouped-query** model — the int8 twin of
/// [`launch_kv_append_gqa`], and KV-headed for the same reason: one warp per `(slot, kv_head)`, one
/// amax reduction and one f32 scale pair per cached `(token, kv_head)`. The added assertion is again
/// that `knew_d`/`vnew_d` carry the **K/V projection** width `gqa.kv_dim()`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn launch_kv_append_int8_gqa(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    knew_d: &CudaSlice<f32>,
    vnew_d: &CudaSlice<f32>,
    k_d: &mut CudaSlice<i8>,
    v_d: &mut CudaSlice<i8>,
    ksc_d: &mut CudaSlice<f32>,
    vsc_d: &mut CudaSlice<f32>,
    bt_d: &CudaSlice<u32>,
    wpos_d: &CudaSlice<u32>,
    active_d: &CudaSlice<u32>,
    gqa: &GqaConfig,
    layer: usize,
    bcap: usize,
) -> Result<(), DriverError> {
    debug_assert_eq!(
        knew_d.len(),
        bcap * gqa.kv_dim(),
        "knew must be [bcap, kv_heads*head_dim] — the K/V projection width, not the hidden size"
    );
    debug_assert_eq!(
        vnew_d.len(),
        bcap * gqa.kv_dim(),
        "vnew must be [bcap, kv_heads*head_dim] — the K/V projection width, not the hidden size"
    );
    launch_kv_append_int8(
        stream,
        func,
        knew_d,
        vnew_d,
        k_d,
        v_d,
        ksc_d,
        vsc_d,
        bt_d,
        wpos_d,
        active_d,
        gqa.kv(),
        layer,
        bcap,
    )
}

/// **f64 full-softmax CPU reference** for the decode attention — the tolerance oracle. `q` is
/// `[bcap, D]`; `k_slots`/`v_slots[b]` are slot `b`'s contiguous context, each `[ctx_b, heads,
/// head_dim]` row-major (`ctx_b == ctx_lens[b]`). Returns `out` `[bcap, D]` f32. A slot with `ctx_b ==
/// 0` (inactive) produces a zero row — exactly what the kernel emits.
///
/// Multi-head only: for a grouped-query model use [`reference_decode_attn_gqa`], of which this is the
/// `q_heads == kv_heads` case (and bit-identical to it, gated).
pub fn reference_decode_attn(
    q: &[f32],
    k_slots: &[Vec<f32>],
    v_slots: &[Vec<f32>],
    ctx_lens: &[usize],
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    reference_decode_attn_gqa(q, k_slots, v_slots, ctx_lens, heads, heads, head_dim, scale)
}

/// **f64 full-softmax CPU reference for grouped-query decode attention** — the tolerance oracle for
/// the GQA path. `q` is `[bcap, q_heads*head_dim]`; `k_slots`/`v_slots[b]` are slot `b`'s contiguous
/// context, each `[ctx_b, kv_heads, head_dim]` row-major. Returns `out` `[bcap, q_heads*head_dim]`.
/// Query head `h` attends over KV head `h / (q_heads / kv_heads)` — the contiguous grouping. A slot
/// with `ctx_b == 0` produces a zero row, as the kernel does.
///
/// **This is an independent oracle, deliberately.** It shares no code with the launch path: it does
/// not build the KV layout with [`crate::paged_kv::KvConfig::elem_offset`] (its `k_slots` are plain
/// contiguous per-slot contexts, and the *caller* is what scatters them into the paged slab), it
/// re-derives the head mapping from the definition of GQA rather than calling
/// [`crate::paged_kv::GqaConfig::kv_head_of`], and it computes a full materialized softmax in f64
/// where the kernel runs a streaming online softmax in f32. Routing it through the same host encoder
/// the kernel path uses is the circular oracle this crate has already been bitten by (`bb52f08`).
#[allow(clippy::too_many_arguments)]
pub fn reference_decode_attn_gqa(
    q: &[f32],
    k_slots: &[Vec<f32>],
    v_slots: &[Vec<f32>],
    ctx_lens: &[usize],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    assert!(
        q_heads > 0 && kv_heads > 0 && q_heads.is_multiple_of(kv_heads),
        "reference_decode_attn_gqa: q_heads ({q_heads}) must be a positive whole multiple of \
         kv_heads ({kv_heads})"
    );
    // The GQA definition, written out here rather than borrowed from the geometry helper: query heads
    // [j*group, (j+1)*group) all read KV head j.
    let group = q_heads / kv_heads;
    let bcap = ctx_lens.len();
    let qd = q_heads * head_dim;
    let kvd = kv_heads * head_dim;
    let mut out = vec![0f32; bcap * qd];
    for b in 0..bcap {
        let ctx = ctx_lens[b];
        if ctx == 0 {
            continue;
        }
        let kb = &k_slots[b];
        let vb = &v_slots[b];
        assert_eq!(
            kb.len(),
            ctx * kvd,
            "slot {b}: K context must be [ctx, kv_heads, head_dim]"
        );
        assert_eq!(
            vb.len(),
            ctx * kvd,
            "slot {b}: V context must be [ctx, kv_heads, head_dim]"
        );
        for h in 0..q_heads {
            let kvh = h / group;
            // scores[t] = scale * dot(q[b,h], K[b,t,kvh])  (f64 accumulate).
            let mut scores = vec![0f64; ctx];
            let mut mx = f64::NEG_INFINITY;
            for (t, sc) in scores.iter_mut().enumerate() {
                let mut acc = 0f64;
                for dh in 0..head_dim {
                    let qv = q[b * qd + h * head_dim + dh] as f64;
                    let kv = kb[(t * kv_heads + kvh) * head_dim + dh] as f64;
                    acc += qv * kv;
                }
                *sc = acc * scale as f64;
                if *sc > mx {
                    mx = *sc;
                }
            }
            // softmax + weighted V.
            let mut denom = 0f64;
            for sc in &scores {
                denom += (*sc - mx).exp();
            }
            for dh in 0..head_dim {
                let mut acc = 0f64;
                for (t, sc) in scores.iter().enumerate() {
                    let w = (*sc - mx).exp();
                    acc += w * vb[(t * kv_heads + kvh) * head_dim + dh] as f64;
                }
                out[b * qd + h * head_dim + dh] = (acc / denom) as f32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both head dims `serving::DecodeLayer` will accept (it asserts `head_dim ∈ {64, 128}`). The
    /// 128-wide lane had no gate at all until this loop: nothing generated it, so a shape or ASCII
    /// regression there surfaced first as a `CUDA_ERROR_INVALID_PTX` at model construction.
    const GATED_HEAD_DIMS: [usize; 2] = [64, 128];

    /// Every generator in this module must open at the **`sm_80` floor**, not at the device's own
    /// architecture. The instruction mix here is `shfl.sync.bfly` + `red.f32` + ordinary ld/st and
    /// f32 math — all Ampere-legal — and PTX is forward-compatible only, so a module tagged `sm_89`
    /// would load on zero A100s while buying nothing on Ada. Pinning the whole header (not just the
    /// `.target`) also catches a `.version` drift, and the negative check keeps a stray `sm_89`
    /// out of the *body* — a `.target` can only be declared once, so a second mention would be a
    /// hand-written directive someone slipped into a `format!`.
    ///
    /// This module is un-gated, so these run under a plain toolchain-free `cargo test`: they are
    /// the only floor gate for the serving family that does not need a GPU.
    fn assert_floor(ptx: &str, what: &str) {
        assert!(
            ptx.starts_with(crate::ptx_target::HDR_SM80),
            "{what}: must open with exactly ptx_target::HDR_SM80 (the family's lowest legal target)"
        );
        assert!(
            !ptx.contains(crate::ptx_target::TARGET_SM89),
            "{what}: nothing here needs an Ada-only instruction, so nothing may pin sm_89"
        );
    }

    // PTX shape sanity (no device): the generator emits the entry, an unrolled query cache + accumulator
    // sized to head_dim, the exp recurrence, and a single bit-exact-friendly output normalize.
    #[test]
    fn ptx_generator_is_well_formed() {
        for hd in GATED_HEAD_DIMS {
            let ptx = paged_attn_decode_ptx(hd);
            assert!(ptx.contains(".visible .entry paged_attn_decode("));
            assert_floor(&ptx, "paged_attn_decode");
            assert!(
                ptx.contains(&format!("%acc{}", hd - 1)),
                "head dim {hd} must unroll the V accumulator"
            );
            assert!(
                !ptx.contains(&format!("%acc{hd}")),
                "must not over-unroll past head_dim {hd}"
            );
            assert!(
                ptx.contains(&format!("qsh[{}]", PAGED_ATTN_WARPS as usize * hd)),
                "per-warp query staging in shared (WARPS*head_dim)"
            );
            assert!(
                ptx.contains("ld.shared.f32"),
                "query read from shared in the dot"
            );
            assert!(
                ptx.contains("shfl.sync.bfly.b32"),
                "cross-lane online-softmax merge"
            );
            assert!(ptx.contains("bar.warp.sync"), "warp sync after staging q");
            assert!(ptx.contains("ex2.approx.f32"), "online softmax exp");
            assert!(ptx.contains("cvt.f32.f16"), "f16 cache widened to f32");
            // The output uses rcp + select (the empty-sequence NaN guard).
            assert!(ptx.contains("rcp.rn.f32") && ptx.contains("selp.f32"));
            assert_gqa_shape(&ptx, "paged_attn_decode");
            assert!(ptx.is_ascii(), "PTX must be pure ASCII (head_dim {hd})");
            // Balanced braces.
            assert_eq!(ptx.matches('{').count(), ptx.matches('}').count());
        }
    }

    /// The two head counts must be *structurally* separated in the generated PTX, not merely intended
    /// to be. The query count drives the grid and the Q/O row; the KV count drives the cache offset;
    /// the bridge between them is `kvhead = head / (heads / kvheads)`. A regression that reverted the
    /// cache offset to `%heads`/`%head` would still produce loadable PTX and still pass every MHA gate
    /// (they are equal there) — it would only fail once `g > 1`, on device. This catches it in a plain
    /// `cargo test`.
    fn assert_gqa_shape(ptx: &str, what: &str) {
        assert!(
            ptx.contains(".param .u32 pKvHeads"),
            "{what}: the KV-head count must be a runtime .param (a baked constant would need a \
             per-group module-cache key, and Gpu::function never re-examines the PTX on a hit)"
        );
        assert!(
            ptx.contains("ld.param.u32 %kvheads,[pKvHeads];"),
            "{what}: pKvHeads must actually be loaded"
        );
        assert!(
            ptx.contains("div.u32 %grp,%heads,%kvheads;"),
            "{what}: the group size must be derived from both counts"
        );
        assert!(
            ptx.contains("div.u32 %kvhead,%head,%grp;"),
            "{what}: the query head must be mapped to its KV head"
        );
        // The cache offset chain must fold the KV head, and must not fold the query head.
        assert!(
            ptx.contains("mul.lo.s32 %e,%e,%kvheads;\n    add.u32 %e,%e,%kvhead;"),
            "{what}: the slab offset must step by kv_heads and add the KV head"
        );
        assert!(
            !ptx.contains("add.u32 %e,%e,%head;"),
            "{what}: the slab offset must never fold the QUERY head — the slab has no such axis"
        );
        // Q/O addressing stays query-headed: D = heads*hd, and the grid is bcap*heads warps.
        assert!(
            ptx.contains("mul.lo.s32 %D,%heads,"),
            "{what}: the Q/O row stride must stay query-headed"
        );
        assert!(
            ptx.contains("mul.lo.s32 %nq,%bcap,%heads;"),
            "{what}: the warp grid must stay query-headed"
        );
    }

    /// **Launch-seam law (crate rule 2).** The argument list each launcher pushes must match its
    /// entry's declared `.param` count, derived from the same text. `pKvHeads` moved every attention
    /// entry from six `.u32` params to seven; pushing the old six would make the driver read adjacent
    /// host stack as the layer index. The counts below are the ones the four `SAFETY` comments state,
    /// so this test is what keeps those comments honest.
    #[test]
    fn attention_entry_param_counts_match_the_launcher_push_lists() {
        let count = |ptx: &str, ty: &str| ptx.matches(&format!(".param .{ty} ")).count();
        for (what, ptx, u64s, f32s, u32s) in [
            // (pointers, scale, [bcap, heads, kv_heads, bsz, nblk, mbps, layer])
            ("paged_attn_decode", paged_attn_decode_ptx(64), 6, 1, 7),
            (
                "paged_attn_decode_int8",
                paged_attn_decode_int8_ptx(64),
                8,
                1,
                7,
            ),
            // The appends are KV-headed and gained nothing: [bcap, heads, hd, bsz, nblk, mbps, layer].
            ("kv_append", kv_append_ptx(), 7, 0, 7),
            ("kv_append_int8", kv_append_int8_ptx(), 9, 0, 7),
        ] {
            assert_eq!(count(&ptx, "u64"), u64s, "{what}: .u64 param count");
            assert_eq!(count(&ptx, "f32"), f32s, "{what}: .f32 param count");
            assert_eq!(count(&ptx, "u32"), u32s, "{what}: .u32 param count");
        }
        // The appends must NOT have grown a KV-head param: they never see a query head at all.
        for (what, ptx) in [
            ("kv_append", kv_append_ptx()),
            ("kv_append_int8", kv_append_int8_ptx()),
        ] {
            assert!(
                !ptx.contains("pKvHeads"),
                "{what}: the append writes the K/V projection, so its `pHeads` already IS the \
                 KV-head count — a second one would be two encodings of one fact"
            );
        }
    }

    // int8 decode-attention PTX shape sanity (no device), at both gated head dims: int8 cache loads
    // (never an f16 widen), the accumulator unroll, the cross-lane merge, ASCII, balanced braces.
    #[test]
    fn int8_attn_ptx_is_well_formed() {
        for hd in GATED_HEAD_DIMS {
            let ptx = paged_attn_decode_int8_ptx(hd);
            assert!(ptx.contains(".visible .entry paged_attn_decode_int8("));
            assert_floor(&ptx, "paged_attn_decode_int8");
            assert!(ptx.contains("ld.global.s8"), "int8 cache read");
            assert!(
                !ptx.contains("cvt.f32.f16"),
                "the int8 kernel stores no f16 — nothing to widen"
            );
            assert!(
                ptx.contains(&format!("%acc{}", hd - 1)),
                "head dim {hd} must unroll the V accumulator"
            );
            assert!(
                !ptx.contains(&format!("%acc{hd}")),
                "must not over-unroll past head_dim {hd}"
            );
            assert!(ptx.contains(&format!("qsh[{}]", PAGED_ATTN_WARPS as usize * hd)));
            assert!(
                ptx.contains("shfl.sync.bfly.b32"),
                "cross-lane online-softmax merge"
            );
            assert!(ptx.contains("ex2.approx.f32"), "online softmax exp");
            assert_gqa_shape(&ptx, "paged_attn_decode_int8");
            // The per-(token, head) dequant scale index is KV-headed too — the scale slab has one
            // entry per cached (token, kv_head), so folding the query head there would read a
            // neighbouring head's scale and silently mis-dequant the whole row.
            assert!(
                ptx.contains("mov.u32 %es,%e;"),
                "the scale index must be taken from the KV-headed offset chain"
            );
            assert!(ptx.is_ascii(), "PTX must be pure ASCII (head_dim {hd})");
            assert_eq!(ptx.matches('{').count(), ptx.matches('}').count());
        }
    }

    // f16 KV-append PTX shape sanity (no device): the entry, the narrowing store, and the
    // inactive-slot early-out that keeps a padding row (block-table 0) off a live block.
    #[test]
    fn kv_append_ptx_is_well_formed() {
        let ptx = kv_append_ptx();
        assert!(ptx.contains(".visible .entry kv_append("));
        assert_floor(&ptx, "kv_append");
        assert!(
            ptx.contains("cvt.rn.f16.f32"),
            "f32 input narrowed into the f16 cache"
        );
        assert!(ptx.contains("st.global.b16"), "f16 value store");
        assert!(ptx.contains("setp.eq.u32 %p0,%tmp,0;"), "active-mask test");
        assert!(
            ptx.contains("@%p0 bra DONE;"),
            "inactive rows exit before any store"
        );
        assert!(ptx.is_ascii(), "PTX must be pure ASCII");
        assert_eq!(ptx.matches('{').count(), ptx.matches('}').count());
    }

    // int8-append PTX shape sanity (no device): warp amax merge, IEEE division (not approx),
    // ties-even rounding, int8 value + f32 scale stores, ASCII-only, balanced braces.
    #[test]
    fn int8_append_ptx_is_well_formed() {
        let ptx = kv_append_int8_ptx();
        assert!(ptx.contains(".visible .entry kv_append_int8("));
        assert_floor(&ptx, "kv_append_int8");
        assert!(
            ptx.contains("shfl.sync.bfly.b32"),
            "warp-cooperative amax merge"
        );
        assert!(
            ptx.contains("div.rn.f32"),
            "IEEE-exact scale + quotient (approx would break the host match)"
        );
        assert!(
            !ptx.contains("div.approx"),
            "no approximate division anywhere"
        );
        assert!(ptx.contains("cvt.rni.f32.f32"), "ties-even rounding");
        assert!(ptx.contains("st.global.s8"), "int8 value store");
        assert!(ptx.contains("st.global.f32"), "f32 scale store");
        assert!(ptx.is_ascii(), "PTX must be pure ASCII");
        assert_eq!(ptx.matches('{').count(), ptx.matches('}').count());
    }

    // The reference zeros inactive (ctx==0) slots and is shape-correct.
    #[test]
    fn reference_zeros_inactive_slots() {
        let (heads, hd) = (2usize, 4usize);
        let d = heads * hd;
        let bcap = 2;
        let q = vec![0.5f32; bcap * d];
        let k_slots = vec![vec![1.0f32; 3 * heads * hd], Vec::new()];
        let v_slots = vec![vec![2.0f32; 3 * heads * hd], Vec::new()];
        let out = reference_decode_attn(&q, &k_slots, &v_slots, &[3, 0], heads, hd, 0.5);
        assert_eq!(out.len(), bcap * d);
        // Slot 0: every value V is 2.0 ⇒ weighted average is 2.0 for every dim.
        for &o in &out[0..d] {
            assert!((o - 2.0).abs() < 1e-5, "active slot averages V=2.0");
        }
        // Slot 1 inactive ⇒ zeros.
        for &o in &out[d..2 * d] {
            assert_eq!(o, 0.0);
        }
    }

    // ==================== GQA reference oracle (no device, no feature gate) ====================

    /// A tiny deterministic generator, so the reference tests below need neither `diff::Rng` (which is
    /// `gpu`-gated) nor a device. Values are spread across a couple of octaves so the softmax is not
    /// degenerate.
    fn seq(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// `reference_decode_attn` must be **bit-for-bit** `reference_decode_attn_gqa` at
    /// `q_heads == kv_heads`. The MHA entry point is now a delegation, and this is what proves the
    /// delegation changed no arithmetic — every existing tolerance gate in this crate and in
    /// `serving.rs` is anchored on the old function, so a drift here would move all of them at once.
    #[test]
    fn gqa_reference_is_bit_identical_to_the_mha_reference_at_group_size_one() {
        for (heads, hd) in [(1usize, 2usize), (3, 4), (4, 8)] {
            let ctx = vec![5usize, 0, 2];
            let d = heads * hd;
            let q = seq(ctx.len() * d, 7 + heads as u32);
            let k: Vec<Vec<f32>> = ctx.iter().map(|&c| seq(c * d, 11 + c as u32)).collect();
            let v: Vec<Vec<f32>> = ctx.iter().map(|&c| seq(c * d, 23 + c as u32)).collect();
            let scale = 1.0 / (hd as f32).sqrt();
            let mha = reference_decode_attn(&q, &k, &v, &ctx, heads, hd, scale);
            let gqa = reference_decode_attn_gqa(&q, &k, &v, &ctx, heads, heads, hd, scale);
            assert_eq!(mha.len(), gqa.len());
            for i in 0..mha.len() {
                assert_eq!(
                    mha[i].to_bits(),
                    gqa[i].to_bits(),
                    "heads={heads} hd={hd}: element {i} drifted"
                );
            }
        }
    }

    /// **The head mapping, proved analytically rather than by agreeing with the kernel.** Give KV head
    /// `j` a value vector that is the constant `j + 1` at every position and channel. Then whatever the
    /// scores are, the softmax weights sum to 1 and the output of *every* query head in group `j` is
    /// exactly `j + 1`. Reading the wrong KV head therefore shows up as an integer-valued output that
    /// is off by a whole group — no tolerance argument required.
    #[test]
    fn gqa_reference_reads_the_grouped_kv_head() {
        for (q_heads, kv_heads, hd) in [(8usize, 2usize, 4usize), (32, 8, 2), (4, 1, 6), (6, 6, 2)]
        {
            let group = q_heads / kv_heads;
            let (ctx, bcap) = (7usize, 2usize);
            let kvd = kv_heads * hd;
            // K varies (so the scores are non-degenerate); V is constant per KV head.
            let k: Vec<Vec<f32>> = (0..bcap).map(|b| seq(ctx * kvd, 31 + b as u32)).collect();
            let v: Vec<Vec<f32>> = (0..bcap)
                .map(|_| {
                    (0..ctx)
                        .flat_map(|_| {
                            (0..kv_heads).flat_map(|j| (0..hd).map(move |_| (j + 1) as f32))
                        })
                        .collect()
                })
                .collect();
            let q = seq(bcap * q_heads * hd, 97);
            let out = reference_decode_attn_gqa(
                &q,
                &k,
                &v,
                &[ctx; 2],
                q_heads,
                kv_heads,
                hd,
                1.0 / (hd as f32).sqrt(),
            );
            for b in 0..bcap {
                for h in 0..q_heads {
                    let want = (h / group + 1) as f32;
                    for dh in 0..hd {
                        let got = out[b * q_heads * hd + h * hd + dh];
                        assert!(
                            (got - want).abs() < 1e-5,
                            "{q_heads}q/{kv_heads}kv: slot {b} query head {h} must read KV head \
                             {} (expected {want}, got {got})",
                            h / group
                        );
                    }
                }
            }
        }
    }

    /// A slot's context must be `[ctx, kv_heads, head_dim]`. Handing the reference a query-headed
    /// context (`g` times too long) is the single most likely GQA mistake, and it would otherwise
    /// silently read every value from the wrong offset and still return finite numbers.
    #[test]
    #[should_panic(expected = "K context must be")]
    fn gqa_reference_rejects_a_query_headed_context() {
        let (q_heads, kv_heads, hd, ctx) = (8usize, 2usize, 4usize, 3usize);
        let wrong = vec![0.5f32; ctx * q_heads * hd]; // query-headed, not KV-headed
        let slots = std::slice::from_ref(&wrong);
        reference_decode_attn_gqa(
            &vec![0.5f32; q_heads * hd],
            slots,
            slots,
            &[ctx],
            q_heads,
            kv_heads,
            hd,
            0.5,
        );
    }

    /// The reference validates the grouping itself rather than dividing by a bogus group size.
    #[test]
    #[should_panic(expected = "whole multiple of")]
    fn gqa_reference_rejects_an_indivisible_grouping() {
        reference_decode_attn_gqa(
            &[0.0; 8],
            &[vec![0.0; 6]],
            &[vec![0.0; 6]],
            &[1],
            8,
            3,
            2,
            1.0,
        );
    }

    // ============================ GPU gates (need a device; skip if none) ============================
    #[cfg(feature = "gpu")]
    use crate::paged_kv::BlockManager;

    /// Local copy of the harness `with_gpu` (the gpu.rs one is private to its test module): runs `body`
    /// with the process-wide `Gpu`, or skips cleanly when no device is reachable.
    ///
    /// The skip goes through [`crate::diff::skip_or_fail`], exactly like `gpu.rs`'s original: this
    /// copy printed and returned unconditionally, so under `WUKONG_GPU_REQUIRED=1` — the invocation
    /// a rented-GPU run uses — these gates reported `ok` having touched no device at all.
    #[cfg(feature = "gpu")]
    fn with_gpu(name: &str, body: impl FnOnce(&mut crate::Gpu)) {
        let mut guard = crate::gpu();
        match guard.as_mut() {
            Some(g) => body(g),
            None => crate::diff::skip_or_fail(
                name,
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            ),
        }
    }

    /// Diagnostic: print the driver JIT error log for the paged-attn PTX (the `ptxas` line behind a bare
    /// `CUDA_ERROR_INVALID_PTX`). Run:
    /// `cargo test -p wukong_codegen_gpu --features gpu paged_attn_jit_log -- --ignored --nocapture`.
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "diagnostic; prints the driver JIT log for the paged-attn PTX module"]
    fn paged_attn_jit_log() {
        with_gpu("paged_attn_jit_log", |g| {
            use cudarc::driver::sys;
            g.ctx.bind_to_thread().unwrap();
            let ptx = paged_attn_decode_ptx(64);
            let ptx_c = std::ffi::CString::new(ptx).unwrap();
            let mut log = vec![0u8; 32768];
            let mut opts = [
                sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
                sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            ];
            let mut vals: [*mut std::ffi::c_void; 2] =
                [log.as_mut_ptr() as *mut _, log.len() as *mut _];
            let mut module: sys::CUmodule = std::ptr::null_mut();
            let res = unsafe {
                sys::cuModuleLoadDataEx(
                    &mut module,
                    ptx_c.as_ptr() as *const _,
                    2,
                    opts.as_mut_ptr(),
                    vals.as_mut_ptr(),
                )
            };
            let s = String::from_utf8_lossy(&log);
            eprintln!(
                "=== paged-attn JIT result {:?} ===\n{}",
                res,
                s.trim_end_matches('\0')
            );
        });
    }

    /// Round a per-slot f32 K/V set to f16-and-back — the *effective* inputs the f16 cache stores, so the
    /// CPU reference sees exactly what the kernel reads (the only residual error is then GPU f32 order +
    /// `ex2.approx`, not f16 rounding).
    #[cfg(feature = "gpu")]
    fn f16_round(slots: &[Vec<f32>]) -> Vec<Vec<f32>> {
        slots
            .iter()
            .map(|s| s.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect())
            .collect()
    }

    /// Populate a layer's f16 K/V slab from per-slot contiguous K/V using `mgr`'s block layout, upload
    /// everything, launch the paged decode-attention kernel, and return the `[bcap, D]` f32 output. The
    /// slab placement is exactly [`KvConfig::elem_offset`], the address the kernel reconstructs.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn run_paged_attn(
        g: &mut crate::Gpu,
        mgr: &BlockManager,
        cfg: &KvConfig,
        layer: usize,
        q: &[f32],
        k_slots: &[Vec<f32>],
        v_slots: &[Vec<f32>],
        scale: f32,
    ) -> Vec<f32> {
        run_paged_attn_gqa(
            g,
            mgr,
            &GqaConfig::mha(*cfg),
            layer,
            q,
            k_slots,
            v_slots,
            scale,
        )
    }

    /// [`run_paged_attn`] for a grouped-query geometry: `q` is `[bcap, q_heads*head_dim]` and each
    /// slot's `k_slots[b]`/`v_slots[b]` is `[ctx_b, kv_heads, head_dim]` — the **KV**-headed context
    /// that actually lives in the cache.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn run_paged_attn_gqa(
        g: &mut crate::Gpu,
        mgr: &BlockManager,
        gqa: &GqaConfig,
        layer: usize,
        q: &[f32],
        k_slots: &[Vec<f32>],
        v_slots: &[Vec<f32>],
        scale: f32,
    ) -> Vec<f32> {
        use half::f16;
        let cfg = gqa.kv();
        let bcap = cfg.num_slots;
        let (kvh_n, hd) = (gqa.kv_heads(), cfg.head_dim);
        let mut kh = vec![f16::from_f32(0.0); cfg.slab_elems()];
        let mut vh = vec![f16::from_f32(0.0); cfg.slab_elems()];
        for b in 0..bcap {
            for t in 0..mgr.context_len(b) {
                let (phys, off) = mgr.locate(b, t);
                for h in 0..kvh_n {
                    for dh in 0..hd {
                        let idx = cfg.elem_offset(layer, phys, off, h, dh);
                        let src = (t * kvh_n + h) * hd + dh;
                        kh[idx] = f16::from_f32(k_slots[b][src]);
                        vh[idx] = f16::from_f32(v_slots[b][src]);
                    }
                }
            }
        }
        let q_d = g.stream.memcpy_stod(q).unwrap();
        let k_d = g.stream.memcpy_stod(&kh).unwrap();
        let v_d = g.stream.memcpy_stod(&vh).unwrap();
        let bt_d = g.stream.memcpy_stod(&mgr.flat_block_table()).unwrap();
        let cl_d = g.stream.memcpy_stod(&mgr.ctx_lens()).unwrap();
        let mut out_d = g.stream.alloc_zeros::<f32>(bcap * gqa.q_dim()).unwrap();
        // `Gpu::function` caches on the key alone and never re-examines the PTX on a hit, so a key
        // shared by two head dims silently hands the second one the first's compiled kernel. The
        // head *counts* are runtime `.param`s, so one key per head_dim still covers every grouping —
        // which is exactly why they were not baked. No catch-all: an unmapped head dim must fail
        // loudly, as the production path does (serving.rs).
        let key: &'static str = match hd {
            64 => "paged_attn_d64",
            128 => "paged_attn_d128",
            other => panic!(
                "paged-attn test harness: no module-cache key for head_dim {other}; add an arm"
            ),
        };
        let func = g
            .function(key, &paged_attn_decode_ptx(hd), PAGED_ATTN_ENTRY)
            .unwrap();
        launch_paged_attn_decode_gqa(
            &g.stream, &func, &q_d, &k_d, &v_d, &mut out_d, &bt_d, &cl_d, gqa, layer, bcap, scale,
        )
        .unwrap();
        g.stream.synchronize().unwrap();
        g.stream.memcpy_dtov(&out_d).unwrap()
    }

    /// Build a fixture: ragged context lengths (incl. an inactive 0 and non-block-multiple lengths),
    /// f16-rounded per-slot K/V, a query, and the `KvConfig`. Returns `(cfg, ctx, q, k, v, scale)`.
    #[cfg(feature = "gpu")]
    #[allow(clippy::type_complexity)]
    fn fixture(
        seed: u64,
        heads: usize,
        hd: usize,
        block_size: usize,
        ctx: Vec<usize>,
    ) -> (
        KvConfig,
        Vec<usize>,
        Vec<f32>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        f32,
    ) {
        let (gqa, ctx, q, k, v, scale) = fixture_gqa(seed, heads, heads, hd, block_size, ctx);
        (*gqa.kv(), ctx, q, k, v, scale)
    }

    /// [`fixture`] for a grouped-query geometry. `q` is `[bcap, q_heads*hd]`; each slot's K/V context
    /// is `[ctx_b, kv_heads, hd]` (f16-rounded, so only the GPU's f32 order and `ex2.approx` remain as
    /// error against the reference). The cache is sized for `kv_heads` — which is the point.
    #[cfg(feature = "gpu")]
    #[allow(clippy::type_complexity)]
    fn fixture_gqa(
        seed: u64,
        q_heads: usize,
        kv_heads: usize,
        hd: usize,
        block_size: usize,
        ctx: Vec<usize>,
    ) -> (
        GqaConfig,
        Vec<usize>,
        Vec<f32>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        f32,
    ) {
        let mut rng = crate::diff::Rng::new(seed);
        let bcap = ctx.len();
        let kvd = kv_heads * hd;
        let max_bps = ctx
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .div_ceil(block_size)
            .max(1)
            + 1;
        let num_blocks = bcap * max_bps + 8;
        let gqa = GqaConfig::new(
            KvConfig {
                layers: 1,
                heads: kv_heads,
                head_dim: hd,
                block_size,
                num_blocks,
                num_slots: bcap,
                max_blocks_per_seq: max_bps,
            },
            q_heads,
        );
        let k: Vec<Vec<f32>> = (0..bcap)
            .map(|b| rng.vec(ctx[b] * kvd, -1.0, 1.0))
            .collect();
        let v: Vec<Vec<f32>> = (0..bcap)
            .map(|b| rng.vec(ctx[b] * kvd, -1.0, 1.0))
            .collect();
        let q = rng.vec(bcap * q_heads * hd, -1.0, 1.0);
        let scale = 1.0 / (hd as f32).sqrt();
        (gqa, ctx, q, f16_round(&k), f16_round(&v), scale)
    }

    /// Replicate a KV-headed per-slot context into the **query-headed** context an equivalent MHA
    /// model would have to store: query head `h` gets a verbatim copy of KV head `h / g`. This is the
    /// definition of GQA written as data, and it is what the bit-exactness gate compares against.
    /// Deliberately *not* built from any launcher-side helper — it is an independent restatement.
    #[cfg(feature = "gpu")]
    fn replicate_kv_heads(
        slots: &[Vec<f32>],
        ctx: &[usize],
        q_heads: usize,
        kv_heads: usize,
        hd: usize,
    ) -> Vec<Vec<f32>> {
        let group = q_heads / kv_heads;
        slots
            .iter()
            .zip(ctx)
            .map(|(s, &c)| {
                let mut out = vec![0f32; c * q_heads * hd];
                for t in 0..c {
                    for h in 0..q_heads {
                        let (src, dst) = ((t * kv_heads + h / group) * hd, (t * q_heads + h) * hd);
                        out[dst..dst + hd].copy_from_slice(&s[src..src + hd]);
                    }
                }
                out
            })
            .collect()
    }

    /// **Absolute-correctness gate (the first law).** The paged decode-attention kernel must reproduce an
    /// f64 full-softmax reference within tolerance — the only legitimate error is the GPU's `ex2.approx`
    /// exp and f32 accumulation order (K/V are pre-rounded to f16 so storage precision cancels). Ragged
    /// context lengths (incl. an inactive 0, a single block, and non-block-multiple lengths) exercise the
    /// block-table walk and the empty-sequence guard. Run at **both** head dims the serving layer
    /// advertises: `head_dim = 128` doubles the register accumulator (~190 virtual regs/thread) and had
    /// never been JIT-loaded, let alone compared to the reference.
    #[cfg(feature = "gpu")]
    #[test]
    fn paged_attention_matches_reference() {
        with_gpu("paged_attention_matches_reference", |g| {
            for hd in GATED_HEAD_DIMS {
                let (heads, block_size) = (4usize, 16usize);
                let (cfg, ctx, q, k, v, scale) =
                    fixture(0x5E13, heads, hd, block_size, vec![37, 0, 16, 100, 5, 64]);
                let mut mgr = BlockManager::new(
                    cfg.num_blocks,
                    block_size,
                    cfg.num_slots,
                    cfg.max_blocks_per_seq,
                );
                for b in 0..cfg.num_slots {
                    if ctx[b] > 0 {
                        mgr.reserve(b, ctx[b]).unwrap();
                    }
                }
                let got = run_paged_attn(g, &mgr, &cfg, 0, &q, &k, &v, scale);
                let refv = reference_decode_attn(&q, &k, &v, &ctx, heads, hd, scale);
                let s = crate::diff::assert_close("paged_attn_decode", &got, &refv, 1e-2, 3e-3);
                eprintln!(
                    "paged decode-attn vs f64 ref: max_abs={:.2e} max_rel={:.2e} (bcap={}, heads={heads}, hd={hd}, ragged ctx {:?})",
                    s.max_abs, s.max_rel, cfg.num_slots, ctx
                );
            }
        });
    }

    /// **Paging-is-numerically-invisible gate (the first law).** The SAME logical sequences laid into two
    /// *different physical block layouts* (slot-order vs reverse-order allocation ⇒ different physical
    /// block ids per sequence) must produce **bit-for-bit identical** output. True by construction: the
    /// block table only changes the load address, never the value or the accumulation order. This is the
    /// decode analogue of int8 split-K / transpose bit-exactness.
    #[cfg(feature = "gpu")]
    #[test]
    fn paged_attention_invariant_to_block_layout() {
        with_gpu("paged_attention_invariant_to_block_layout", |g| {
            let (heads, hd, block_size) = (4usize, 64usize, 16usize);
            let (cfg, ctx, q, k, v, scale) =
                fixture(0xB10C, heads, hd, block_size, vec![40, 7, 0, 96, 33]);
            // Layout A: allocate slots in ascending order.
            let mut a = BlockManager::new(
                cfg.num_blocks,
                block_size,
                cfg.num_slots,
                cfg.max_blocks_per_seq,
            );
            for b in 0..cfg.num_slots {
                if ctx[b] > 0 {
                    a.reserve(b, ctx[b]).unwrap();
                }
            }
            // Layout B: allocate in descending order ⇒ each active slot gets *different* physical blocks.
            let mut bm = BlockManager::new(
                cfg.num_blocks,
                block_size,
                cfg.num_slots,
                cfg.max_blocks_per_seq,
            );
            for b in (0..cfg.num_slots).rev() {
                if ctx[b] > 0 {
                    bm.reserve(b, ctx[b]).unwrap();
                }
            }
            // Prove the layouts actually differ for an active slot (else the gate is vacuous).
            let first_active = (0..cfg.num_slots).find(|&b| ctx[b] > 0).unwrap();
            assert_ne!(
                a.table(first_active),
                bm.table(first_active),
                "the two layouts must physically differ for the gate to be meaningful"
            );
            let out_a = run_paged_attn(g, &a, &cfg, 0, &q, &k, &v, scale);
            let out_b = run_paged_attn(g, &bm, &cfg, 0, &q, &k, &v, scale);
            assert_eq!(out_a.len(), out_b.len());
            for i in 0..out_a.len() {
                assert_eq!(
                    out_a[i].to_bits(),
                    out_b[i].to_bits(),
                    "paged attention changed under a different physical block layout at element {i}"
                );
            }
            eprintln!(
                "paged decode-attn BIT-IDENTICAL across 2 physical block layouts (slot {first_active}: A={:?} vs B={:?}) \
                 — paging is numerically invisible",
                a.table(first_active),
                bm.table(first_active)
            );
        });
    }

    // ============================== GQA device gates (grouped-query) ==============================

    /// The GQA shapes worth running on real silicon: `(q_heads, kv_heads, head_dim)`. Llama-3-8B's
    /// grouping (g=4) at both gated head dims, the 70B grouping (g=8), MQA (every query head on one
    /// KV head — the extreme the divide has to survive), and an MHA control so the same harness proves
    /// `g = 1` still works through the new code path.
    #[cfg(feature = "gpu")]
    const GQA_SHAPES: [(usize, usize, usize); 5] = [
        (8, 2, 64),  // g = 4, Llama-3-8B's grouping
        (8, 1, 64),  // MQA
        (8, 8, 64),  // MHA control through the GQA path
        (8, 2, 128), // g = 4 at the wider head dim (doubles the register accumulator)
        (16, 2, 64), // g = 8, Llama-3-70B's grouping
    ];

    /// **GQA absolute-correctness gate.** Grouped-query paged decode attention must reproduce the f64
    /// full-softmax reference within tolerance at every grouping — including MQA and the `g = 1`
    /// control. Ragged contexts (an inactive 0, a sub-block length, a multi-block length) exercise the
    /// block-table walk and the empty-sequence guard at the same time.
    #[cfg(feature = "gpu")]
    #[test]
    fn paged_gqa_attention_matches_reference() {
        with_gpu("paged_gqa_attention_matches_reference", |g| {
            for (q_heads, kv_heads, hd) in GQA_SHAPES {
                let block_size = 16usize;
                let (gqa, ctx, q, k, v, scale) = fixture_gqa(
                    0x69A0,
                    q_heads,
                    kv_heads,
                    hd,
                    block_size,
                    vec![37, 0, 16, 100, 5, 64],
                );
                let cfg = *gqa.kv();
                let mut mgr = BlockManager::new(
                    cfg.num_blocks,
                    block_size,
                    cfg.num_slots,
                    cfg.max_blocks_per_seq,
                );
                for (b, &c) in ctx.iter().enumerate() {
                    if c > 0 {
                        mgr.reserve(b, c).unwrap();
                    }
                }
                let got = run_paged_attn_gqa(g, &mgr, &gqa, 0, &q, &k, &v, scale);
                let refv =
                    reference_decode_attn_gqa(&q, &k, &v, &ctx, q_heads, kv_heads, hd, scale);
                let s = crate::diff::assert_close("paged_attn_decode_gqa", &got, &refv, 1e-2, 3e-3);
                eprintln!(
                    "GQA decode-attn {q_heads}q/{kv_heads}kv x {hd} (g={}) vs f64 ref: max_abs={:.2e} \
                     max_rel={:.2e} — cache is {}x smaller than the MHA equivalent",
                    gqa.group_size(),
                    s.max_abs,
                    s.max_rel,
                    gqa.group_size()
                );
            }
        });
    }

    /// **The head mapping, proved on device against a second kernel run rather than against a host
    /// helper.** A `g`-grouped cache must give **bit-for-bit** the output an ungrouped cache gives when
    /// its K/V is the same data with each KV head replicated `g` times — that is what GQA *means*, and
    /// it is bit-exact rather than tolerance-bounded because the two runs feed identical values to
    /// identical lanes in identical order (only the load address differs, exactly as for paging).
    ///
    /// This is the gate that would catch a mapping that is in-range but wrong (round-robin
    /// `h % kv_heads` instead of `h / g`), which the tolerance gate above cannot distinguish from
    /// noise once the values are random.
    #[cfg(feature = "gpu")]
    #[test]
    fn paged_gqa_attention_equals_mha_over_replicated_kv() {
        with_gpu("paged_gqa_attention_equals_mha_over_replicated_kv", |g| {
            for (q_heads, kv_heads, hd) in GQA_SHAPES {
                let block_size = 16usize;
                let ctx = vec![41usize, 0, 16, 77];
                let (gqa, ctx, q, k, v, scale) =
                    fixture_gqa(0xC0FFEE, q_heads, kv_heads, hd, block_size, ctx);
                let cfg = *gqa.kv();
                // The ungrouped twin: same pool, same block policy, q_heads KV heads.
                let mha = GqaConfig::mha(KvConfig {
                    heads: q_heads,
                    ..cfg
                });
                let k_rep = replicate_kv_heads(&k, &ctx, q_heads, kv_heads, hd);
                let v_rep = replicate_kv_heads(&v, &ctx, q_heads, kv_heads, hd);
                // One block layout drives both runs, so the only difference is the head axis.
                let mut mgr = BlockManager::new(
                    cfg.num_blocks,
                    block_size,
                    cfg.num_slots,
                    cfg.max_blocks_per_seq,
                );
                for (b, &c) in ctx.iter().enumerate() {
                    if c > 0 {
                        mgr.reserve(b, c).unwrap();
                    }
                }
                let grouped = run_paged_attn_gqa(g, &mgr, &gqa, 0, &q, &k, &v, scale);
                let ungrouped = run_paged_attn_gqa(g, &mgr, &mha, 0, &q, &k_rep, &v_rep, scale);
                assert_eq!(grouped.len(), ungrouped.len());
                for i in 0..grouped.len() {
                    assert_eq!(
                        grouped[i].to_bits(),
                        ungrouped[i].to_bits(),
                        "{q_heads}q/{kv_heads}kv: element {i} differs from the replicated-KV MHA run \
                         — the query head is reading the wrong KV head"
                    );
                }
                eprintln!(
                    "GQA {q_heads}q/{kv_heads}kv x {hd}: BIT-IDENTICAL to MHA over {}x-replicated KV, \
                     from a cache holding {} of the elements ({} vs {})",
                    gqa.group_size(),
                    if gqa.is_mha() { "all" } else { "a fraction" },
                    cfg.slab_elems(),
                    mha.kv().slab_elems()
                );
            }
        });
    }

    /// **GQA does not disturb paging invariance.** Same logical sequences, two different physical
    /// block layouts, bit-identical output — the argument in this module's header says GQA changes
    /// which head's K/V a lane reads and neither the position→lane partition nor the merge order, so
    /// the property must survive. Run it rather than trust it.
    #[cfg(feature = "gpu")]
    #[test]
    fn paged_gqa_attention_invariant_to_block_layout() {
        with_gpu("paged_gqa_attention_invariant_to_block_layout", |g| {
            let (q_heads, kv_heads, hd, block_size) = (8usize, 2usize, 64usize, 16usize);
            let (gqa, ctx, q, k, v, scale) = fixture_gqa(
                0xB10C6A4,
                q_heads,
                kv_heads,
                hd,
                block_size,
                vec![40, 7, 0, 96, 33],
            );
            let cfg = *gqa.kv();
            let mk = || {
                BlockManager::new(
                    cfg.num_blocks,
                    block_size,
                    cfg.num_slots,
                    cfg.max_blocks_per_seq,
                )
            };
            let (mut a, mut bm) = (mk(), mk());
            for (b, &c) in ctx.iter().enumerate() {
                if c > 0 {
                    a.reserve(b, c).unwrap();
                }
            }
            for b in (0..cfg.num_slots).rev() {
                if ctx[b] > 0 {
                    bm.reserve(b, ctx[b]).unwrap();
                }
            }
            let first_active = (0..cfg.num_slots).find(|&b| ctx[b] > 0).unwrap();
            assert_ne!(
                a.table(first_active),
                bm.table(first_active),
                "the two layouts must physically differ for the gate to be meaningful"
            );
            let out_a = run_paged_attn_gqa(g, &a, &gqa, 0, &q, &k, &v, scale);
            let out_b = run_paged_attn_gqa(g, &bm, &gqa, 0, &q, &k, &v, scale);
            for i in 0..out_a.len() {
                assert_eq!(
                    out_a[i].to_bits(),
                    out_b[i].to_bits(),
                    "GQA paged attention changed under a different physical block layout at {i}"
                );
            }
            eprintln!(
                "GQA decode-attn ({q_heads}q/{kv_heads}kv) BIT-IDENTICAL across 2 physical block \
                 layouts (slot {first_active}: A={:?} vs B={:?}) — paging stays invisible under grouping",
                a.table(first_active),
                bm.table(first_active)
            );
        });
    }

    /// **`kv_append` must mask free slots — re-pinned at a GQA shape.** A free slot's block table pads
    /// with 0, so an unmasked append scatters into physical block 0, which is a *live* block belonging
    /// to some other sequence. The whole slab is compared against an exact expectation, so an inactive
    /// row that wrote anything anywhere fails.
    ///
    /// The GQA half: the append is KV-headed, so `knew`/`vnew` are `[bcap, kv_heads*head_dim]` and the
    /// expectation is built over `kv_heads` rows. Passing a query-headed projection here is the GQA
    /// mistake [`launch_kv_append_gqa`]'s debug assert exists to catch, and the exact-slab comparison
    /// is what would catch it if the assert were compiled out.
    #[cfg(feature = "gpu")]
    #[test]
    fn gqa_kv_append_masks_free_slots() {
        with_gpu("gqa_kv_append_masks_free_slots", |g| {
            use half::f16;
            let (q_heads, kv_heads, hd, bsz, bcap) = (8usize, 2usize, 64usize, 16usize, 8usize);
            let active = [true, false, true, true, false, false, true, false];
            let wpos = [3usize, 0, 7, 16, 0, 0, 20, 0];
            let max_bps = wpos.iter().copied().max().unwrap().div_ceil(bsz) + 2;
            let num_blocks = bcap * max_bps + 4;
            let gqa = GqaConfig::new(
                KvConfig {
                    layers: 1,
                    heads: kv_heads,
                    head_dim: hd,
                    block_size: bsz,
                    num_blocks,
                    num_slots: bcap,
                    max_blocks_per_seq: max_bps,
                },
                q_heads,
            );
            let cfg = *gqa.kv();
            let kvd = gqa.kv_dim();
            assert_eq!(kvd * gqa.group_size(), gqa.q_dim(), "fixture sanity");
            let mut mgr = BlockManager::new(num_blocks, bsz, bcap, max_bps);
            for (b, &a) in active.iter().enumerate() {
                if a {
                    mgr.reserve(b, wpos[b] + 1).unwrap();
                }
            }
            let mut rng = crate::diff::Rng::new(0x5A1D6A4);
            let knew = rng.vec(bcap * kvd, -1.0, 1.0);
            let vnew = rng.vec(bcap * kvd, -1.0, 1.0);
            let knew_d = g.stream.memcpy_stod(&knew).unwrap();
            let vnew_d = g.stream.memcpy_stod(&vnew).unwrap();
            let mut k_d = g.stream.alloc_zeros::<f16>(cfg.slab_elems()).unwrap();
            let mut v_d = g.stream.alloc_zeros::<f16>(cfg.slab_elems()).unwrap();
            let bt_d = g.stream.memcpy_stod(&mgr.flat_block_table()).unwrap();
            let wpos_u: Vec<u32> = wpos.iter().map(|&p| p as u32).collect();
            let wpos_d = g.stream.memcpy_stod(&wpos_u).unwrap();
            let act_u: Vec<u32> = active.iter().map(|&a| a as u32).collect();
            let act_d = g.stream.memcpy_stod(&act_u).unwrap();
            let func = g
                .function("kv_append", &kv_append_ptx(), KV_APPEND_ENTRY)
                .unwrap();
            launch_kv_append_gqa(
                &g.stream, &func, &knew_d, &vnew_d, &mut k_d, &mut v_d, &bt_d, &wpos_d, &act_d,
                &gqa, 0, bcap,
            )
            .unwrap();
            g.stream.synchronize().unwrap();
            let kh = g.stream.memcpy_dtov(&k_d).unwrap();
            let vh = g.stream.memcpy_dtov(&v_d).unwrap();
            // Exact slab expectation: zeros everywhere except each ACTIVE slot's KV-headed channels.
            let mut ek = vec![0f32; cfg.slab_elems()];
            let mut ev = vec![0f32; cfg.slab_elems()];
            for b in 0..bcap {
                if !active[b] {
                    continue;
                }
                let (phys, off) = mgr.locate(b, wpos[b]);
                for h in 0..kv_heads {
                    for dh in 0..hd {
                        let idx = cfg.elem_offset(0, phys, off, h, dh);
                        let src = b * kvd + h * hd + dh;
                        ek[idx] = f16::from_f32(knew[src]).to_f32();
                        ev[idx] = f16::from_f32(vnew[src]).to_f32();
                    }
                }
            }
            for i in 0..cfg.slab_elems() {
                assert_eq!(
                    kh[i].to_f32(),
                    ek[i],
                    "K slab elem {i} (inactive-row leak?)"
                );
                assert_eq!(
                    vh[i].to_f32(),
                    ev[i],
                    "V slab elem {i} (inactive-row leak?)"
                );
            }
            let written = active.iter().filter(|&&a| a).count();
            eprintln!(
                "GQA kv_append ({q_heads}q/{kv_heads}kv, {kvd}-wide projection): {written}/{bcap} \
                 active slots written at their KV-headed addresses; every inactive row skipped \
                 (block 0 intact)"
            );
        });
    }

    // ===================== P6: int8-KV quantization (tolerance-gated footprint win) =================

    /// Quantize each slot's f32 K/V to the int8 cache (per-(token, head) scale), lay them + the scales
    /// into the slabs via the block table, run the int8 decode-attention kernel, return the `[bcap, D]`
    /// output. The device twin of [`run_paged_attn`] with int8 storage.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn run_paged_attn_int8(
        g: &mut crate::Gpu,
        mgr: &BlockManager,
        cfg: &KvConfig,
        layer: usize,
        q: &[f32],
        k_slots: &[Vec<f32>],
        v_slots: &[Vec<f32>],
        scale: f32,
    ) -> Vec<f32> {
        run_paged_attn_int8_gqa(
            g,
            mgr,
            &GqaConfig::mha(*cfg),
            layer,
            q,
            k_slots,
            v_slots,
            scale,
        )
    }

    /// [`run_paged_attn_int8`] for a grouped-query geometry: the int8 values *and* both f32 scale
    /// slabs are KV-headed (one scale per cached `(token, kv_head)`), while `q`/`out` stay
    /// query-headed.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn run_paged_attn_int8_gqa(
        g: &mut crate::Gpu,
        mgr: &BlockManager,
        gqa: &GqaConfig,
        layer: usize,
        q: &[f32],
        k_slots: &[Vec<f32>],
        v_slots: &[Vec<f32>],
        scale: f32,
    ) -> Vec<f32> {
        let cfg = gqa.kv();
        let bcap = cfg.num_slots;
        let (kvh_n, hd) = (gqa.kv_heads(), cfg.head_dim);
        let d = gqa.q_dim();
        let mut kq = vec![0i8; cfg.slab_elems()];
        let mut vq = vec![0i8; cfg.slab_elems()];
        let mut ks = vec![0f32; cfg.scale_slab_elems()];
        let mut vs = vec![0f32; cfg.scale_slab_elems()];
        for b in 0..bcap {
            let ctx = mgr.context_len(b);
            let (kqi, ksi) = quantize_kv_int8(&k_slots[b], ctx, kvh_n, hd);
            let (vqi, vsi) = quantize_kv_int8(&v_slots[b], ctx, kvh_n, hd);
            for t in 0..ctx {
                let (phys, off) = mgr.locate(b, t);
                for h in 0..kvh_n {
                    ks[cfg.scale_offset(layer, phys, off, h)] = ksi[t * kvh_n + h];
                    vs[cfg.scale_offset(layer, phys, off, h)] = vsi[t * kvh_n + h];
                    for dh in 0..hd {
                        let idx = cfg.elem_offset(layer, phys, off, h, dh);
                        let src = (t * kvh_n + h) * hd + dh;
                        kq[idx] = kqi[src];
                        vq[idx] = vqi[src];
                    }
                }
            }
        }
        let q_d = g.stream.memcpy_stod(q).unwrap();
        let k_d = g.stream.memcpy_stod(&kq).unwrap();
        let v_d = g.stream.memcpy_stod(&vq).unwrap();
        let ksc_d = g.stream.memcpy_stod(&ks).unwrap();
        let vsc_d = g.stream.memcpy_stod(&vs).unwrap();
        let bt_d = g.stream.memcpy_stod(&mgr.flat_block_table()).unwrap();
        let cl_d = g.stream.memcpy_stod(&mgr.ctx_lens()).unwrap();
        let mut out_d = g.stream.alloc_zeros::<f32>(bcap * d).unwrap();
        // No catch-all — see the f16 twin in `run_paged_attn_gqa`: one cache key must mean one PTX.
        let key: &'static str = match hd {
            64 => "paged_attn_int8_d64",
            128 => "paged_attn_int8_d128",
            other => panic!("paged-attn int8 test harness: no module-cache key for head_dim {other}; add an arm"),
        };
        let func = g
            .function(key, &paged_attn_decode_int8_ptx(hd), PAGED_ATTN_INT8_ENTRY)
            .unwrap();
        launch_paged_attn_decode_int8_gqa(
            &g.stream, &func, &q_d, &k_d, &v_d, &ksc_d, &vsc_d, &mut out_d, &bt_d, &cl_d, gqa,
            layer, bcap, scale,
        )
        .unwrap();
        g.stream.synchronize().unwrap();
        g.stream.memcpy_dtov(&out_d).unwrap()
    }

    /// **int8-KV tolerance gate (the first law, lossy path).** The int8-cache decode-attention must match
    /// the f64 full-precision reference within tolerance — the only error is the per-(token, head) int8
    /// quantization of K/V (Q stays f32). Ragged contexts exercise the block-table walk + empty guard.
    #[cfg(feature = "gpu")]
    #[test]
    fn paged_int8_attention_matches_reference() {
        with_gpu("paged_int8_attention_matches_reference", |g| {
            for hd in GATED_HEAD_DIMS {
                let (heads, block_size) = (4usize, 16usize);
                let (cfg, ctx, q, k, v, scale) =
                    fixture(0x171AB, heads, hd, block_size, vec![37, 0, 16, 100, 5, 64]);
                let mut mgr = BlockManager::new(
                    cfg.num_blocks,
                    block_size,
                    cfg.num_slots,
                    cfg.max_blocks_per_seq,
                );
                for b in 0..cfg.num_slots {
                    if ctx[b] > 0 {
                        mgr.reserve(b, ctx[b]).unwrap();
                    }
                }
                let got = run_paged_attn_int8(g, &mgr, &cfg, 0, &q, &k, &v, scale);
                let refv = reference_decode_attn(&q, &k, &v, &ctx, heads, hd, scale);
                // int8-KV achieves max_abs ~3e-3 here; gate at 1e-2 (3× headroom) to catch regressions.
                let s = crate::diff::assert_close("paged_attn_int8", &got, &refv, 1e-2, 5e-2);
                eprintln!(
                    "int8-KV decode-attn vs f64 ref: max_abs={:.2e} max_rel={:.2e} (hd={hd}, per-(token,head) int8 \
                     K/V, f32 Q; ragged ctx {:?})",
                    s.max_abs, s.max_rel, ctx
                );
            }
        });
    }

    /// **int8-KV paging invariance.** The int8 values + scales placed under two *different physical block
    /// layouts* must give **bit-for-bit identical** output — the dequant multiply order is fixed per
    /// token, so paging stays invisible even on the quantized path (as for f16).
    #[cfg(feature = "gpu")]
    #[test]
    fn paged_int8_attention_invariant_to_block_layout() {
        with_gpu("paged_int8_attention_invariant_to_block_layout", |g| {
            let (heads, hd, block_size) = (4usize, 64usize, 16usize);
            let (cfg, ctx, q, k, v, scale) =
                fixture(0x9C0DE, heads, hd, block_size, vec![40, 7, 0, 96, 33]);
            let mut a = BlockManager::new(
                cfg.num_blocks,
                block_size,
                cfg.num_slots,
                cfg.max_blocks_per_seq,
            );
            for b in 0..cfg.num_slots {
                if ctx[b] > 0 {
                    a.reserve(b, ctx[b]).unwrap();
                }
            }
            let mut bm = BlockManager::new(
                cfg.num_blocks,
                block_size,
                cfg.num_slots,
                cfg.max_blocks_per_seq,
            );
            for b in (0..cfg.num_slots).rev() {
                if ctx[b] > 0 {
                    bm.reserve(b, ctx[b]).unwrap();
                }
            }
            let out_a = run_paged_attn_int8(g, &a, &cfg, 0, &q, &k, &v, scale);
            let out_b = run_paged_attn_int8(g, &bm, &cfg, 0, &q, &k, &v, scale);
            for i in 0..out_a.len() {
                assert_eq!(
                    out_a[i].to_bits(),
                    out_b[i].to_bits(),
                    "int8 attention changed under a different block layout at {i}"
                );
            }
            eprintln!("int8-KV decode-attn BIT-IDENTICAL across 2 physical block layouts — paging invisible on the quantized path");
        });
    }

    /// **GQA on the int8 path.** The two footprint levers compose: grouping divides the cache by `g`
    /// and int8 storage halves what is left, so a Llama-3-8B-shaped cache lands at ~1/8 of an f16 MHA
    /// one. Correctness-wise this exercises the piece the f16 kernel does not have — the per-(token,
    /// **KV** head) dequant scale — where folding a query head into `%es` would read a neighbouring
    /// head's scale and mis-scale a whole row.
    #[cfg(feature = "gpu")]
    #[test]
    fn paged_gqa_int8_attention_matches_reference() {
        with_gpu("paged_gqa_int8_attention_matches_reference", |g| {
            for (q_heads, kv_heads, hd) in GQA_SHAPES {
                let block_size = 16usize;
                let (gqa, ctx, q, k, v, scale) = fixture_gqa(
                    0x171AB6A4,
                    q_heads,
                    kv_heads,
                    hd,
                    block_size,
                    vec![37, 0, 16, 100, 5, 64],
                );
                let cfg = *gqa.kv();
                let mut mgr = BlockManager::new(
                    cfg.num_blocks,
                    block_size,
                    cfg.num_slots,
                    cfg.max_blocks_per_seq,
                );
                for (b, &c) in ctx.iter().enumerate() {
                    if c > 0 {
                        mgr.reserve(b, c).unwrap();
                    }
                }
                let got = run_paged_attn_int8_gqa(g, &mgr, &gqa, 0, &q, &k, &v, scale);
                let refv =
                    reference_decode_attn_gqa(&q, &k, &v, &ctx, q_heads, kv_heads, hd, scale);
                // Same tolerance as the MHA int8 gate: the error is the int8 K/V quantization, and
                // grouping changes which head is read, not how precisely it is stored.
                let s = crate::diff::assert_close("paged_attn_int8_gqa", &got, &refv, 1e-2, 5e-2);
                eprintln!(
                    "GQA int8-KV decode-attn {q_heads}q/{kv_heads}kv x {hd} (g={}) vs f64 ref: \
                     max_abs={:.2e} max_rel={:.2e} — KV bytes {} vs {} f16-ungrouped",
                    gqa.group_size(),
                    s.max_abs,
                    s.max_rel,
                    cfg.kv_bytes_int8(),
                    gqa.mha_equivalent_kv_bytes(2)
                );
            }
        });
    }

    /// **GQA on the int8 path is the replicated-KV MHA run, to the bit** — the int8 twin of
    /// [`paged_gqa_attention_equals_mha_over_replicated_kv`]. Quantization is per (token, head) and the
    /// replicated heads are byte-identical copies, so each replica quantizes to the same int8 values
    /// and the same scale; any difference in the output is the head mapping, not the quantizer.
    #[cfg(feature = "gpu")]
    #[test]
    fn paged_gqa_int8_attention_equals_mha_over_replicated_kv() {
        with_gpu(
            "paged_gqa_int8_attention_equals_mha_over_replicated_kv",
            |g| {
                let (q_heads, kv_heads, hd, block_size) = (8usize, 2usize, 64usize, 16usize);
                let (gqa, ctx, q, k, v, scale) = fixture_gqa(
                    0x9C0DE6A4,
                    q_heads,
                    kv_heads,
                    hd,
                    block_size,
                    vec![40, 7, 0, 96, 33],
                );
                let cfg = *gqa.kv();
                let mha = GqaConfig::mha(KvConfig {
                    heads: q_heads,
                    ..cfg
                });
                let k_rep = replicate_kv_heads(&k, &ctx, q_heads, kv_heads, hd);
                let v_rep = replicate_kv_heads(&v, &ctx, q_heads, kv_heads, hd);
                let mut mgr = BlockManager::new(
                    cfg.num_blocks,
                    block_size,
                    cfg.num_slots,
                    cfg.max_blocks_per_seq,
                );
                for (b, &c) in ctx.iter().enumerate() {
                    if c > 0 {
                        mgr.reserve(b, c).unwrap();
                    }
                }
                let grouped = run_paged_attn_int8_gqa(g, &mgr, &gqa, 0, &q, &k, &v, scale);
                let ungrouped =
                    run_paged_attn_int8_gqa(g, &mgr, &mha, 0, &q, &k_rep, &v_rep, scale);
                for i in 0..grouped.len() {
                    assert_eq!(
                        grouped[i].to_bits(),
                        ungrouped[i].to_bits(),
                        "int8 GQA element {i} differs from the replicated-KV MHA run"
                    );
                }
                eprintln!(
                "GQA int8-KV {q_heads}q/{kv_heads}kv: BIT-IDENTICAL to MHA over {}x-replicated KV \
                 (values and per-(token,head) scales alike)",
                gqa.group_size()
            );
            },
        );
    }
}
