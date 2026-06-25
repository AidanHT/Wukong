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
//! ## Kernel shape (correctness-first v1)
//! One **thread per `(slot, head)`** (grid = `ceil(num_slots*heads / BLOCK)` CTAs of `BLOCK` threads),
//! each thread fully sequential over its context. This trades the warp-cooperative coalescing of
//! vLLM's kernel for two decisive simplifications: it is **trivially bit-exact across physical block
//! layouts** (one thread, one fixed accumulation order — the block table only changes *where* a value
//! is read, never the value or the order), and it has **no shared-memory logits buffer**, so it handles
//! arbitrary context length with no v2-style split-K (vLLM needs v2 past 8192 tokens precisely to dodge
//! that SMEM shortage). The head dim is **unrolled into registers** and the query vector is **cached in
//! registers** across the whole context loop (read once, reused every position); the block-table walk
//! is **incremental** (track `logical`/`offset` as the position advances) so there is no per-position
//! integer divide. A warp-cooperative rewrite is the documented next perf lever.
//!
//! ## The first-law property the gates prove
//! - **Absolute correctness**: tolerance-gated vs an f64 full-softmax CPU reference
//!   ([`reference_decode_attn`]) — the only legitimate error is the GPU's `ex2.approx` exp + f32 order.
//! - **Paging is numerically invisible**: the kernel output is **bit-for-bit identical** when the same
//!   logical sequences are laid into two *different* physical block layouts (contiguous vs a fragmented
//!   free-list order). True by construction; the `paged_attention_invariant_to_block_layout` gate proves
//!   it to the bit (the decode analogue of int8 split-K / transpose bit-exactness).

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg};

use crate::paged_kv::KvConfig;

/// Threads per CTA for the decode-attention launch. Each thread owns one `(slot, head)`; threads are
/// independent (no cooperation), so this is just an occupancy knob.
pub const PAGED_ATTN_BLOCK: u32 = 128;

/// PTX entry name for the paged decode-attention kernel.
pub const PAGED_ATTN_ENTRY: &str = "paged_attn_decode";

/// Generate the paged decode-attention PTX, **specialized to `head_dim`** (the per-head dimension is
/// baked so the query vector and the V accumulator unroll into named registers `%q0..` / `%acc0..`).
/// One thread computes `out[slot,head,:] = softmax(scale · q · Kᵀ) · V` over the sequence's context,
/// gathering K/V via the block table. Layout matches [`KvConfig::elem_offset`]:
/// `[layers, num_blocks, block_size, heads, head_dim]`, f16 cache, f32 query/out.
pub fn paged_attn_decode_ptx(head_dim: usize) -> String {
    assert!(head_dim > 0 && head_dim % 2 == 0, "head_dim must be a positive even number");
    let hd = head_dim;
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let mut s = String::new();
    s += ".version 7.8\n.target sm_89\n.address_size 64\n\n";
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
        \x20   .param .u32 pBsz,\n\
        \x20   .param .u32 pNblk,\n\
        \x20   .param .u32 pMbps,\n\
        \x20   .param .u32 pLayer\n)\n{{\n"
    );
    // Registers. %q<hd>/%acc<hd> are the unrolled query cache + V accumulator.
    s += &format!("    .reg .f32 %q<{hd}>;\n");
    s += &format!("    .reg .f32 %acc<{hd}>;\n");
    s += "    .reg .f32 %score,%m,%l,%newm,%p,%corr,%kf,%vf,%invl,%scale,%t0;\n";
    s += "    .reg .b16 %h;\n";
    s += "    .reg .b32 %gid,%slot,%head,%ctx,%t,%logical,%rem,%phys,%D,%qidx,%nq,%tmp,%bcap,%heads,%bsz,%nblk,%mbps,%layer,%e;\n";
    s += "    .reg .b64 %Q,%K,%V,%O,%BT,%CL,%addr,%qrow,%obase,%kbase,%vbase,%off;\n";
    s += "    .reg .pred %p0,%p1,%p2,%p3;\n";
    // Load params.
    s += "    ld.param.u64 %Q,[pQ];   cvta.to.global.u64 %Q,%Q;\n";
    s += "    ld.param.u64 %K,[pK];   cvta.to.global.u64 %K,%K;\n";
    s += "    ld.param.u64 %V,[pV];   cvta.to.global.u64 %V,%V;\n";
    s += "    ld.param.u64 %O,[pO];   cvta.to.global.u64 %O,%O;\n";
    s += "    ld.param.u64 %BT,[pBT]; cvta.to.global.u64 %BT,%BT;\n";
    s += "    ld.param.u64 %CL,[pCL]; cvta.to.global.u64 %CL,%CL;\n";
    s += "    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u32 %bcap,[pBcap];\n";
    s += "    ld.param.u32 %heads,[pHeads];\n";
    s += "    ld.param.u32 %bsz,[pBsz];\n";
    s += "    ld.param.u32 %nblk,[pNblk];\n";
    s += "    ld.param.u32 %mbps,[pMbps];\n";
    s += "    ld.param.u32 %layer,[pLayer];\n";
    // gid = ctaid.x*ntid.x + tid.x ; bail if gid >= bcap*heads.
    s += "    mov.u32 %tmp,%ctaid.x;\n    mov.u32 %gid,%ntid.x;\n    mov.u32 %slot,%tid.x;\n";
    s += "    mad.lo.s32 %gid,%tmp,%gid,%slot;\n";
    s += "    mul.lo.s32 %nq,%bcap,%heads;\n";
    s += "    setp.ge.u32 %p0,%gid,%nq;\n    @%p0 bra DONE;\n";
    // slot = gid / heads ; head = gid - slot*heads.
    s += "    div.u32 %slot,%gid,%heads;\n";
    s += "    mul.lo.s32 %tmp,%slot,%heads;\n    sub.u32 %head,%gid,%tmp;\n";
    // ctx = CL[slot].
    s += "    mul.wide.u32 %off,%slot,4;\n    add.s64 %addr,%CL,%off;\n    ld.global.u32 %ctx,[%addr];\n";
    // D = heads*hd ; qidx = slot*D + head*hd ; qrow = Q + qidx*4 ; obase = O + qidx*4.
    s += &format!("    mul.lo.s32 %D,%heads,{hd};\n");
    s += "    mul.lo.s32 %qidx,%slot,%D;\n";
    s += &format!("    mul.lo.s32 %tmp,%head,{hd};\n    add.u32 %qidx,%qidx,%tmp;\n");
    s += "    mul.wide.u32 %off,%qidx,4;\n    add.s64 %qrow,%Q,%off;\n    add.s64 %obase,%O,%off;\n";
    // Cache the query vector in registers (read once, reused every context position).
    for d in 0..hd {
        s += &format!("    ld.global.f32 %q{d},[%qrow+{}];\n", d * 4);
    }
    // Init online-softmax state: acc=0, m=-inf, l=0.
    for d in 0..hd {
        s += &format!("    mov.f32 %acc{d},0f00000000;\n");
    }
    s += "    mov.f32 %m,0fFF800000;\n    mov.f32 %l,0f00000000;\n";
    // Context loop with incremental block-table walk (no per-position divide).
    s += "    mov.u32 %t,0;\n    mov.u32 %logical,0;\n    mov.u32 %rem,0;\n";
    s += "LOOP:\n    setp.ge.u32 %p1,%t,%ctx;\n    @%p1 bra ENDLOOP;\n";
    // (Re)load phys = BT[slot*mbps + logical] only at a block boundary (rem==0).
    s += "    setp.ne.u32 %p3,%rem,0;\n    @%p3 bra HAVEPHYS;\n";
    s += "    mul.lo.s32 %tmp,%slot,%mbps;\n    add.u32 %tmp,%tmp,%logical;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %addr,%BT,%off;\n    ld.global.u32 %phys,[%addr];\n";
    s += "HAVEPHYS:\n";
    // e = ((((layer*nblk)+phys)*bsz + rem)*heads + head)*hd  (element index of this token's head, d=0).
    s += "    mul.lo.s32 %e,%layer,%nblk;\n    add.u32 %e,%e,%phys;\n";
    s += "    mul.lo.s32 %e,%e,%bsz;\n    add.u32 %e,%e,%rem;\n";
    s += "    mul.lo.s32 %e,%e,%heads;\n    add.u32 %e,%e,%head;\n";
    s += &format!("    mul.lo.s32 %e,%e,{hd};\n");
    s += "    mul.wide.u32 %off,%e,2;\n    add.s64 %kbase,%K,%off;\n    add.s64 %vbase,%V,%off;\n";
    // score = scale * dot(q, K[t]).
    s += "    mov.f32 %score,0f00000000;\n";
    for d in 0..hd {
        s += &format!("    ld.global.u16 %h,[%kbase+{}];\n    cvt.f32.f16 %kf,%h;\n    fma.rn.f32 %score,%q{d},%kf,%score;\n", d * 2);
    }
    s += "    mul.f32 %score,%score,%scale;\n";
    // Online-softmax recurrence: newm=max(m,score); corr=exp(m-newm); p=exp(score-newm); l=l*corr+p.
    s += "    max.f32 %newm,%m,%score;\n";
    s += &format!("    sub.f32 %t0,%m,%newm;\n    mul.f32 %t0,%t0,{log2e};\n    ex2.approx.f32 %corr,%t0;\n");
    s += &format!("    sub.f32 %t0,%score,%newm;\n    mul.f32 %t0,%t0,{log2e};\n    ex2.approx.f32 %p,%t0;\n");
    s += "    mul.f32 %l,%l,%corr;\n    add.f32 %l,%l,%p;\n";
    // acc[d] = acc[d]*corr + p*V[t][d].
    for d in 0..hd {
        s += &format!("    ld.global.u16 %h,[%vbase+{}];\n    cvt.f32.f16 %vf,%h;\n    mul.f32 %acc{d},%acc{d},%corr;\n    fma.rn.f32 %acc{d},%p,%vf,%acc{d};\n", d * 2);
    }
    s += "    mov.f32 %m,%newm;\n";
    // Advance position; cross to the next block when the current one fills.
    s += "    add.u32 %t,%t,1;\n    add.u32 %rem,%rem,1;\n";
    s += "    setp.lt.u32 %p3,%rem,%bsz;\n    @%p3 bra LOOP;\n";
    s += "    mov.u32 %rem,0;\n    add.u32 %logical,%logical,1;\n    bra LOOP;\n";
    s += "ENDLOOP:\n";
    // out[d] = acc[d] / l (0 if l==0, i.e. empty/inactive sequence). rcp.rn + select avoids NaN.
    s += "    rcp.rn.f32 %invl,%l;\n    setp.gt.f32 %p2,%l,0f00000000;\n    selp.f32 %invl,%invl,0f00000000,%p2;\n";
    for d in 0..hd {
        s += &format!("    mul.f32 %t0,%acc{d},%invl;\n    st.global.f32 [%obase+{}],%t0;\n", d * 4);
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
    debug_assert_eq!(q_d.len(), bcap * cfg.heads * cfg.head_dim, "q must be [bcap, heads*head_dim]");
    debug_assert_eq!(out_d.len(), bcap * cfg.heads * cfg.head_dim, "out must be [bcap, heads*head_dim]");
    let nq = (bcap * cfg.heads) as u32;
    let cfg_launch = LaunchConfig {
        grid_dim: (nq.div_ceil(PAGED_ATTN_BLOCK), 1, 1),
        block_dim: (PAGED_ATTN_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (heads, bsz, nblk, mbps, layer_u) =
        (cfg.heads as u32, cfg.block_size as u32, cfg.num_blocks as u32, cfg.max_blocks_per_seq as u32, layer as u32);
    let bcap_u = bcap as u32;
    let mut b = stream.launch_builder(func);
    b.arg(q_d).arg(k_d).arg(v_d).arg(out_d).arg(bt_d).arg(cl_d).arg(&scale);
    b.arg(&bcap_u).arg(&heads).arg(&bsz).arg(&nblk).arg(&mbps).arg(&layer_u);
    unsafe { b.launch(cfg_launch)? };
    Ok(())
}

/// **f64 full-softmax CPU reference** for the decode attention — the tolerance oracle. `q` is
/// `[bcap, D]`; `k_slots`/`v_slots[b]` are slot `b`'s contiguous context, each `[ctx_b, heads,
/// head_dim]` row-major (`ctx_b == ctx_lens[b]`). Returns `out` `[bcap, D]` f32. A slot with `ctx_b ==
/// 0` (inactive) produces a zero row — exactly what the kernel emits.
pub fn reference_decode_attn(
    q: &[f32],
    k_slots: &[Vec<f32>],
    v_slots: &[Vec<f32>],
    ctx_lens: &[usize],
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let bcap = ctx_lens.len();
    let d = heads * head_dim;
    let mut out = vec![0f32; bcap * d];
    for b in 0..bcap {
        let ctx = ctx_lens[b];
        if ctx == 0 {
            continue;
        }
        let kb = &k_slots[b];
        let vb = &v_slots[b];
        for h in 0..heads {
            // scores[t] = scale * dot(q[b,h], K[b,t,h])  (f64 accumulate).
            let mut scores = vec![0f64; ctx];
            let mut mx = f64::NEG_INFINITY;
            for (t, sc) in scores.iter_mut().enumerate() {
                let mut acc = 0f64;
                for dh in 0..head_dim {
                    let qv = q[b * d + h * head_dim + dh] as f64;
                    let kv = kb[(t * heads + h) * head_dim + dh] as f64;
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
                    acc += w * vb[(t * heads + h) * head_dim + dh] as f64;
                }
                out[b * d + h * head_dim + dh] = (acc / denom) as f32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // PTX shape sanity (no device): the generator emits the entry, an unrolled query cache + accumulator
    // sized to head_dim, the exp recurrence, and a single bit-exact-friendly output normalize.
    #[test]
    fn ptx_generator_is_well_formed() {
        let ptx = paged_attn_decode_ptx(64);
        assert!(ptx.contains(".visible .entry paged_attn_decode("));
        assert!(ptx.contains(".target sm_89"));
        assert!(ptx.contains("%q63") && ptx.contains("%acc63"), "head dim must unroll to 64 regs");
        assert!(!ptx.contains("%q64"), "must not over-unroll past head_dim");
        assert!(ptx.contains("ex2.approx.f32"), "online softmax exp");
        assert!(ptx.contains("cvt.f32.f16"), "f16 cache widened to f32");
        // The output uses rcp + select (the empty-sequence NaN guard).
        assert!(ptx.contains("rcp.rn.f32") && ptx.contains("selp.f32"));
        // Balanced braces.
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

    // ============================ GPU gates (need a device; skip if none) ============================
    #[cfg(feature = "gpu")]
    use crate::paged_kv::BlockManager;

    /// Local copy of the harness `with_gpu` (the gpu.rs one is private to its test module): runs `body`
    /// with the process-wide `Gpu`, or skips cleanly when no device is reachable.
    #[cfg(feature = "gpu")]
    fn with_gpu(name: &str, body: impl FnOnce(&mut crate::Gpu)) {
        let mut guard = crate::gpu();
        match guard.as_mut() {
            Some(g) => body(g),
            None => eprintln!("[skip] {name}: no CUDA device reachable"),
        }
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
        use half::f16;
        let bcap = cfg.num_slots;
        let d = cfg.heads * cfg.head_dim;
        let mut kh = vec![f16::from_f32(0.0); cfg.slab_elems()];
        let mut vh = vec![f16::from_f32(0.0); cfg.slab_elems()];
        for b in 0..bcap {
            for t in 0..mgr.context_len(b) {
                let (phys, off) = mgr.locate(b, t);
                for h in 0..cfg.heads {
                    for dh in 0..cfg.head_dim {
                        let idx = cfg.elem_offset(layer, phys, off, h, dh);
                        let src = (t * cfg.heads + h) * cfg.head_dim + dh;
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
        let mut out_d = g.stream.alloc_zeros::<f32>(bcap * d).unwrap();
        let key: &'static str = match cfg.head_dim {
            64 => "paged_attn_d64",
            128 => "paged_attn_d128",
            _ => "paged_attn_dX",
        };
        let func = g.function(key, &paged_attn_decode_ptx(cfg.head_dim), PAGED_ATTN_ENTRY).unwrap();
        launch_paged_attn_decode(&g.stream, &func, &q_d, &k_d, &v_d, &mut out_d, &bt_d, &cl_d, cfg, layer, bcap, scale)
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
    ) -> (KvConfig, Vec<usize>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>, f32) {
        let mut rng = crate::diff::Rng::new(seed);
        let bcap = ctx.len();
        let d = heads * hd;
        let max_bps = ctx.iter().copied().max().unwrap_or(0).div_ceil(block_size).max(1) + 1;
        let num_blocks = bcap * max_bps + 8;
        let cfg = KvConfig {
            layers: 1,
            heads,
            head_dim: hd,
            block_size,
            num_blocks,
            num_slots: bcap,
            max_blocks_per_seq: max_bps,
        };
        let k: Vec<Vec<f32>> = (0..bcap).map(|b| rng.vec(ctx[b] * d, -1.0, 1.0)).collect();
        let v: Vec<Vec<f32>> = (0..bcap).map(|b| rng.vec(ctx[b] * d, -1.0, 1.0)).collect();
        let q = rng.vec(bcap * d, -1.0, 1.0);
        let scale = 1.0 / (hd as f32).sqrt();
        (cfg, ctx, q, f16_round(&k), f16_round(&v), scale)
    }

    /// **Absolute-correctness gate (the first law).** The paged decode-attention kernel must reproduce an
    /// f64 full-softmax reference within tolerance — the only legitimate error is the GPU's `ex2.approx`
    /// exp and f32 accumulation order (K/V are pre-rounded to f16 so storage precision cancels). Ragged
    /// context lengths (incl. an inactive 0, a single block, and non-block-multiple lengths) exercise the
    /// block-table walk and the empty-sequence guard.
    #[cfg(feature = "gpu")]
    #[test]
    fn paged_attention_matches_reference() {
        with_gpu("paged_attention_matches_reference", |g| {
            let (heads, hd, block_size) = (4usize, 64usize, 16usize);
            let (cfg, ctx, q, k, v, scale) =
                fixture(0x5E13, heads, hd, block_size, vec![37, 0, 16, 100, 5, 64]);
            let mut mgr = BlockManager::new(cfg.num_blocks, block_size, cfg.num_slots, cfg.max_blocks_per_seq);
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
            let mut a = BlockManager::new(cfg.num_blocks, block_size, cfg.num_slots, cfg.max_blocks_per_seq);
            for b in 0..cfg.num_slots {
                if ctx[b] > 0 {
                    a.reserve(b, ctx[b]).unwrap();
                }
            }
            // Layout B: allocate in descending order ⇒ each active slot gets *different* physical blocks.
            let mut bm = BlockManager::new(cfg.num_blocks, block_size, cfg.num_slots, cfg.max_blocks_per_seq);
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
}
