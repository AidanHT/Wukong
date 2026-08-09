//! 2D **convolution** on the GPU (the op category attention/GEMM don't cover). Single batch:
//! input `X[C,H,W]`, weights `W[K,C,R,S]`, output `O[K,P,Q]` -- the valid cross-correlation
//! deep-learning frameworks call conv2d. The f32 kernels are stride-1 / no-padding
//! (`P=H-R+1`, `Q=W-S+1`); the fp16 tensor-core family below also covers the general affine case
//! (`stride`, `pad`: `P=(H+2·pad-R)/stride+1`).
//!
//! Wukong knows `C,H,W,K,R,S` at *compile* time (shapes live in the type system), so this is a
//! **generator**, not a fixed kernel: every function here emits a kernel **specialized to the exact
//! shape** -- the `r,s` window fully unrolled, every extent a baked-in literal, no dynamic bounds, no
//! tail logic. The lever no shape-agnostic library has.
//!
//! ## Kernels (the launcher in `gpu.rs` picks; `*_applies` gates each fast path)
//! * [`CONV2D`] -- the original **naive** one-thread-per-output kernel (looping `c,r,s` from global),
//!   kept as the honest worst-case reference and the fallback for shapes the tiled path rejects.
//! * [`conv2d_ptx`] -- **SMEM-tiled, static-shape-specialized direct f32 conv.** One CTA computes a
//!   `TILE_P x TILE_Q` output tile for [`kblock`]`(K)` output channels at once (grid
//!   `= (ceil(Q/TQ), ceil(P/TP), K/KB)`, `KB` f32 accumulators per thread). Per input channel `c` the
//!   CTA cooperatively stages the input **halo** `(TP+R-1) x (TQ+S-1)` and the `KB*R*S` weights into
//!   shared memory **once**, then every thread reuses them across the whole tile -- killing the naive
//!   kernel's dominant cost (each input pixel re-read from global `R*S` times). The `r,s` reduction is
//!   fully unrolled into an `fma.rn.f32` chain over `c`.
//! * The fp16 tensor-core **implicit-GEMM** family (the cuDNN-class path, see the section comment
//!   further down): [`conv_wmma_ptx`] and its split-K ([`conv_wmma_splitk_ptx`] +
//!   [`conv_splitk_reduce_ptx`]), fused bias/activation ([`conv_wmma_epi_ptx`]), strided
//!   ([`conv_wmma_strided_ptx`]), padded ([`conv_wmma_pad_ptx`], [`conv_wmma_pad_splitk_ptx`]) and
//!   *register*-double-buffered ([`conv_wmma_db_ptx`]) variants, plus the standalone
//!   [`pad_nchw_copy_ptx`] scatter and the unfused [`bias_relu_ptx`] baseline.
//!
//! ## Shared memory: the closed forms, the budget seam, and what actually binds
//!
//! Each family's footprint has a closed form, exposed here so the dispatch layer never re-derives a
//! size the generator already computed: [`tiled_smem_bytes`] for the direct f32 tiled conv and
//! [`conv_wmma_smem_bytes`] for the implicit-GEMM family. Both the *gate* ([`tiled_applies_budget`],
//! [`wmma_applies_budget`]) and the *declaration* ([`conv2d_ptx_budget`]) read the same function, so
//! an admitted footprint and a declared one cannot drift apart.
//!
//! `smem_budget` (bytes) is passed **in** by the dispatch layer from `Gpu::smem_budget()` — the
//! generators stay pure text functions with device-free tests, and an A100/H100 budget is therefore
//! enumerable on a laptop. Its second job is to select the *emission form* via
//! [`crate::gpu::smem_mode_for`]: at or below the PTX ISA's 48 KiB **static** `.shared` cap
//! ([`crate::gpu::STATIC_SMEM_CAP`]) the historical `.shared` array is emitted **byte-identically**;
//! beyond it the tile is carved out of the one module-scope [`crate::gpu::DSMEM_DECL`] window, which
//! the launch wrapper must opt into (`Gpu::function_smem` + `dyn_launch_cfg`).
//!
//! **What the budget actually buys here is small, and saying so is the point.** With the tile pinned
//! at [`TILE_P`]×[`TILE_Q`] and [`kblock`] capped at 8, the tiled conv's footprint is bounded by the
//! *filter*: `((15+R)(15+S) + KB·R·S)·4`. Square filters fit up to `R=S=34` inside the static cap and
//! only up to 51 / 66 / 78 at the 4050 / A100 / H100 opt-in windows — all far outside any real conv
//! (7×7 is a ResNet stem, 11×11 an AlexNet conv1). The gate this replaced was 44 KiB, i.e. `R=S≤33`:
//! **it declined nothing anyone runs.** What binds this family is the 1024-thread block cap
//! (`TILE_P·TILE_Q` threads) and the `KB` accumulators per thread, not shared memory. The one conv
//! configuration in this file where a bigger budget is the *only* road is widening the implicit-GEMM
//! CTA tile: 64×64 costs 20480 B, 128×64 costs 38912 B (still static-legal, so the cheapest widening
//! needs no window at all), and 128×128 costs 73728 B — over the ISA cap on every target. See
//! `conv_smem_census_across_target_budgets` in this file's tests for the machine-checked table.

// Every generator in this file opens its module with the shared `sm_80` header. The conv family
// emits f32 arithmetic, shared memory, `cp.async`, `ldmatrix` and `wmma`/`mma.sync` fragments —
// all Ampere-legal — and PTX is forward-compatible only, so the module is tagged with the LOWEST
// legal target, never with the device's own arch (an `sm_89` tag loads on ZERO A100s).
use crate::gpu::{smem_mode_for, SmemMode, DSMEM_DECL, DSMEM_SYM, STATIC_SMEM_CAP};
use crate::ptx_target::HDR_SM80;

/// Output-tile height a tiled-conv CTA computes (threads in `y`).
pub const TILE_P: usize = 16;
/// Output-tile width a tiled-conv CTA computes (threads in `x`).
pub const TILE_Q: usize = 16;

/// Output channels each thread accumulates in registers (channel register-block). The loaded input
/// halo is **independent of `k`**, so blocking `KB` output channels per CTA reuses each staged input
/// pixel across `KB` weights from registers and amortizes the per-input-channel barriers `KB×` — the
/// lever that turns the SMEM tile from a loss into a win. Largest power-of-two-ish divisor of `K`
/// (≤8) so the grid divides evenly with no `k` tail.
pub fn kblock(k: usize) -> usize {
    for kb in [8usize, 4, 2] {
        if k % kb == 0 {
            return kb;
        }
    }
    1
}

/// **The tiled generator's shared-memory footprint in bytes — the single source.**
///
/// `((TILE_P+r-1)·(TILE_Q+s-1) + kb·r·s)·4`: the staged input halo (one f32 per element, shared by all
/// `kb` output channels) plus the `kb` weight windows of `r·s` taps. The gate
/// ([`tiled_applies_budget`]) and the declaration ([`conv2d_ptx_budget`]) both read *this* function.
/// They used to be two hand-written copies of the expression, and a desync there is not a build error:
/// it admits a shape whose entry then declares more shared memory than the gate cleared.
pub const fn tiled_smem_bytes(kb: usize, r: usize, s: usize) -> usize {
    ((TILE_P + r - 1) * (TILE_Q + s - 1) + kb * r * s) * 4
}

/// Whether the SMEM-tiled generator applies to this shape **within `smem_budget` bytes**: a valid conv
/// (`H>=R`, `W>=S`) whose [`tiled_smem_bytes`] fits. Callers that cannot spend the dynamic window pass
/// [`STATIC_SMEM_CAP`]; a caller that can pass `Gpu::smem_budget()`. Everything it declines falls back
/// to [`CONV2D`], so every shape stays runnable either way.
pub fn tiled_applies_budget(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    smem_budget: usize,
) -> bool {
    if h < r || w < s || c < 1 || k < 1 {
        return false;
    }
    tiled_smem_bytes(kblock(k), r, s) <= smem_budget
}

/// [`tiled_applies_budget`] at the **static** ceiling — the verdict for a launcher that declares its
/// tile the historical way (`.shared` array, `shared_mem_bytes: 0`), which is every conv launcher in
/// `gpu.rs` today.
///
/// This replaces a hardcoded `44 * 1024` whose comment claimed there was *"no opt-in to the larger Ada
/// banks"*. That stopped being true when the dynamic-SMEM window landed (`Gpu::function_dyn` /
/// `Gpu::smem_budget`), and the 44 KiB number never had a mechanism behind it either — the real static
/// boundary is the PTX ISA's [`STATIC_SMEM_CAP`], and the real device boundary is the opt-in window.
/// **The only verdict this changes is the 44–48 KiB band**, which at `KB=8` is reached solely by square
/// filters `R=S=34` (`R=S∈{68,69,70}` at `KB=1`) — shapes that now take the tiled kernel instead of the
/// naive fallback, computing the identical convolution.
pub fn tiled_applies(c: usize, h: usize, w: usize, k: usize, r: usize, s: usize) -> bool {
    tiled_applies_budget(c, h, w, k, r, s, STATIC_SMEM_CAP)
}

/// Emit the SMEM-tiled, **channel-register-blocked** conv kernel specialized to
/// `[C,H,W] (*) [K,C,R,S] -> [K,P,Q]`, spending at most [`STATIC_SMEM_CAP`] of shared memory. Entry
/// name is `conv2d`. One CTA computes a `TILE_P×TILE_Q` output tile for `KB=`[`kblock`]`(K)` output
/// channels at once; each thread holds `KB` f32 accumulators. Launch with block `(TILE_Q, TILE_P, 1)`
/// and grid `(ceil(Q/TQ), ceil(P/TP), K/KB)`, `shared_mem_bytes: 0` (the tile is a static `.shared`).
///
/// Panics if the shape does not fit the static cap — pair it with [`tiled_applies`], or use
/// [`conv2d_ptx_budget`] to spend a device window.
pub fn conv2d_ptx(c: usize, h: usize, w: usize, k: usize, r: usize, s: usize) -> String {
    let (ptx, mode) = conv2d_ptx_budget(c, h, w, k, r, s, STATIC_SMEM_CAP);
    debug_assert_eq!(
        mode,
        SmemMode::Static,
        "the static cap cannot yield a window"
    );
    ptx
}

/// [`conv2d_ptx`] against an explicit shared-memory budget, returning the module **and the
/// [`SmemMode`] its launch must honour** (`Gpu::function_smem` consumes exactly this pair, so no
/// caller re-derives a size the generator already computed).
///
/// `smem_budget` is the ceiling this entry may spend — pass [`STATIC_SMEM_CAP`] for the historical
/// static form, or `Gpu::smem_budget()` (the probed `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`: 99 KiB on this
/// Ada laptop, 163 on A100, 227 on H100) to let a large-filter shape through. It also selects the
/// *emission form* via [`smem_mode_for`]: at or below the ISA's static cap the historical
/// `.shared .align 4 .b8 smem[N]` array is emitted **byte-for-byte**, and only beyond it does the tile
/// move into the one module-scope [`DSMEM_DECL`] window. Nothing else about the kernel changes — the
/// generator already forms every shared address as `mov.u32 %r10,<base>` plus a constant offset, so the
/// window swap is one symbol.
///
/// Panics (loudly, at generation) when the footprint exceeds `smem_budget`: the decline belongs in
/// [`tiled_applies_budget`] at dispatch, and the driver's own rejection
/// (`ptxas error: Entry function 'conv2d' uses too much shared data`) names neither conv, the shape,
/// nor the budget.
pub fn conv2d_ptx_budget(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    smem_budget: usize,
) -> (String, SmemMode) {
    use std::fmt::Write as _;
    let (tp, tq) = (TILE_P, TILE_Q);
    let kb = kblock(k);
    let p = h - r + 1;
    let q = w - s + 1;
    let halo_h = tp + r - 1;
    let halo_w = tq + s - 1;
    let halo = halo_h * halo_w; // input elements staged per channel
    let nthreads = tp * tq;
    let rs = r * s;
    let c_rs = c * rs; // stride between successive output channels in W
    let pq = p * q;
    // The one closed form the gate reads too. The 48 KiB ISA cap decides the FORM, the budget the
    // CEILING — two different boundaries, and conflating them is what the old 44 KiB constant did.
    let smem_bytes = tiled_smem_bytes(kb, r, s);
    let mode = smem_mode_for(smem_bytes);
    assert!(
        smem_bytes <= smem_budget,
        "conv2d: SMEM {smem_bytes} B (tile {tp}x{tq}, kblock {kb}, halo {halo_h}x{halo_w}, \
         R={r} S={s}) exceeds the budget {smem_budget} B"
    );
    // Static: the entry's own `.shared` array, the historical spelling emitted verbatim. Dynamic: the
    // single module-scope `wk_dsmem` window (its base is NOT guaranteed to be 0 — always go through
    // the symbol, which this generator already did).
    let sym = if mode.is_dynamic() { DSMEM_SYM } else { "smem" };
    let row_iters = halo_h.div_ceil(tp); // halo rows each thread strides over
    let col_iters = halo_w.div_ceil(tq);
    let w_iters = rs.div_ceil(nthreads); // weight loads per (kk, thread)

    let mut b = String::new();
    let _ = writeln!(b, "{HDR_SM80}");
    if mode.is_dynamic() {
        // MODULE SCOPE, not inside the entry — the identical line in an entry body is
        // CUDA_ERROR_INVALID_PTX (measured; D6 §1.1).
        b.push_str(DSMEM_DECL);
    }
    let _ = writeln!(
        b,
        "// SMEM-tiled conv2d specialized to C={c} H={h} W={w} K={k} R={r} S={s}"
    );
    let _ = writeln!(
        b,
        "// tile {tp}x{tq}, kblock {kb}, halo {halo_h}x{halo_w}, smem {smem_bytes} B"
    );
    let _ = writeln!(b, ".visible .entry conv2d(");
    let _ = writeln!(b, "    .param .u64 pXin,");
    let _ = writeln!(b, "    .param .u64 pWt,");
    let _ = writeln!(b, "    .param .u64 pOut");
    let _ = writeln!(b, ")");
    let _ = writeln!(b, "{{");
    let _ = writeln!(b, "    .reg .pred %p<8>;");
    let _ = writeln!(b, "    .reg .b32  %r<64>;");
    let _ = writeln!(b, "    .reg .f32  %f<32>;");
    let _ = writeln!(b, "    .reg .b64  %rd<12>;");
    if !mode.is_dynamic() {
        let _ = writeln!(b, "    .shared .align 4 .b8 smem[{smem_bytes}];");
    }
    let _ = writeln!(b);
    // global base pointers
    let _ = writeln!(b, "    ld.param.u64 %rd1,[pXin];");
    let _ = writeln!(b, "    ld.param.u64 %rd2,[pWt];");
    let _ = writeln!(b, "    ld.param.u64 %rd3,[pOut];");
    let _ = writeln!(b, "    cvta.to.global.u64 %rd1,%rd1;");
    let _ = writeln!(b, "    cvta.to.global.u64 %rd2,%rd2;");
    let _ = writeln!(b, "    cvta.to.global.u64 %rd3,%rd3;");
    let _ = writeln!(b);
    // thread / block coords
    let _ = writeln!(b, "    mov.u32 %r1,%tid.x;          // tx in [0,{tq})");
    let _ = writeln!(b, "    mov.u32 %r2,%tid.y;          // ty in [0,{tp})");
    let _ = writeln!(b, "    mov.u32 %r3,%ctaid.x;");
    let _ = writeln!(b, "    mov.u32 %r4,%ctaid.y;");
    let _ = writeln!(b, "    mov.u32 %r5,%ctaid.z;        // k-block index");
    let _ = writeln!(b, "    mul.lo.s32 %r6,%r3,{tq};     // q0");
    let _ = writeln!(b, "    mul.lo.s32 %r7,%r4,{tp};     // p0");
    let _ = writeln!(b, "    add.s32 %r8,%r6,%r1;         // oq = q0+tx");
    let _ = writeln!(b, "    add.s32 %r9,%r7,%r2;         // op = p0+ty");
    let _ = writeln!(
        b,
        "    mov.u32 %r10,{sym};           // smem base addr (shared u32)"
    );
    // compute base addr in smem for this thread's window origin (ty*halo_w + tx)*4
    let _ = writeln!(
        b,
        "    mad.lo.s32 %r11,%r2,{halo_w},%r1;  // ty*halo_w + tx"
    );
    let _ = writeln!(b, "    shl.b32 %r11,%r11,2;");
    let _ = writeln!(
        b,
        "    add.s32 %r12,%r10,%r11;      // smem compute base for (ty,tx)"
    );
    // tlin = ty*tq + tx
    let _ = writeln!(b, "    mad.lo.s32 %r13,%r2,{tq},%r1;      // tlin");
    // k0 = kblock * KB  ;  woff_k0 = k0*C*R*S
    let _ = writeln!(
        b,
        "    mul.lo.s32 %r14,%r5,{};      // woff_k0 = (kb_idx*KB)*C*R*S",
        kb * c_rs
    );
    // accumulators acc[kk] = %f{8+kk} := 0
    for kk in 0..kb {
        let _ = writeln!(b, "    mov.f32 %f{},0f00000000;", 8 + kk);
    }
    let _ = writeln!(b, "    mov.u32 %r15,0;              // c = 0");
    let _ = writeln!(b, "CLOOP:");
    let _ = writeln!(b, "    setp.ge.u32 %p0,%r15,{c};");
    let _ = writeln!(b, "    @%p0 bra CEND;");
    let _ = writeln!(b, "    mul.lo.s32 %r16,%r15,{};     // rXc = c*H*W", h * w);
    let _ = writeln!(
        b,
        "    mad.lo.s32 %r17,%r15,{rs},%r14;    // base_c = woff_k0 + c*R*S"
    );
    let _ = writeln!(b);
    let _ = writeln!(
        b,
        "    // ---- cooperative halo load (X channel c, shared by all KB channels) ----"
    );
    for i in 0..row_iters {
        let hy_base = i * tp;
        let _ = writeln!(b, "    add.s32 %r18,%r2,{hy_base};        // hy");
        let _ = writeln!(b, "    setp.lt.u32 %p1,%r18,{halo_h};     // rowok");
        let _ = writeln!(b, "    add.s32 %r19,%r7,%r18;            // gy = p0+hy");
        let _ = writeln!(b, "    setp.lt.u32 %p2,%r19,{h};         // gyok");
        let _ = writeln!(b, "    mul.lo.s32 %r20,%r19,{w};         // gy*W");
        let _ = writeln!(b, "    add.s32 %r20,%r20,%r16;           // gy*W + rXc");
        for j in 0..col_iters {
            let hx_base = j * tq;
            let _ = writeln!(b, "    add.s32 %r21,%r1,{hx_base};       // hx");
            let _ = writeln!(b, "    setp.lt.u32 %p3,%r21,{halo_w};    // colok");
            let _ = writeln!(
                b,
                "    and.pred %p3,%p3,%p1;            // storeok = colok && rowok"
            );
            let _ = writeln!(b, "    add.s32 %r22,%r6,%r21;           // gx = q0+hx");
            let _ = writeln!(b, "    setp.lt.u32 %p4,%r22,{w};        // gxok");
            let _ = writeln!(
                b,
                "    and.pred %p4,%p4,%p2;            // valok = gxok && gyok"
            );
            let _ = writeln!(
                b,
                "    add.s32 %r23,%r20,%r22;          // Xelem = gy*W+rXc+gx"
            );
            let _ = writeln!(b, "    mul.wide.s32 %rd4,%r23,4;");
            let _ = writeln!(b, "    add.s64 %rd4,%rd1,%rd4;");
            let _ = writeln!(b, "    mov.f32 %f1,0f00000000;");
            let _ = writeln!(b, "    @%p4 ld.global.f32 %f1,[%rd4];");
            let _ = writeln!(
                b,
                "    mad.lo.s32 %r24,%r18,{halo_w},%r21;  // smem elem = hy*halo_w+hx"
            );
            let _ = writeln!(b, "    shl.b32 %r24,%r24,2;");
            let _ = writeln!(b, "    add.s32 %r24,%r10,%r24;");
            let _ = writeln!(b, "    @%p3 st.shared.f32 [%r24],%f1;");
        }
    }
    let _ = writeln!(b);
    let _ = writeln!(
        b,
        "    // ---- cooperative weight load: KB windows of R*S, channel c ----"
    );
    for kk in 0..kb {
        // global base of (k0+kk, c) window = base_c + kk*(C*R*S)
        let _ = writeln!(
            b,
            "    add.s32 %r28,%r17,{};        // wbase (k0+{kk},c)",
            kk * c_rs
        );
        let smem_w_base = halo + kk * rs; // smem element offset of this window
        for j in 0..w_iters {
            let base = j * nthreads;
            let _ = writeln!(b, "    add.s32 %r25,%r13,{base};        // weight slot");
            let _ = writeln!(b, "    setp.lt.u32 %p1,%r25,{rs};");
            let _ = writeln!(
                b,
                "    add.s32 %r26,%r25,%r28;         // global weight elem"
            );
            let _ = writeln!(b, "    mul.wide.s32 %rd5,%r26,4;");
            let _ = writeln!(b, "    add.s64 %rd5,%rd2,%rd5;");
            let _ = writeln!(b, "    mov.f32 %f1,0f00000000;");
            let _ = writeln!(b, "    @%p1 ld.global.f32 %f1,[%rd5];");
            let _ = writeln!(
                b,
                "    add.s32 %r27,%r25,{smem_w_base};    // smem weight elem"
            );
            let _ = writeln!(b, "    shl.b32 %r27,%r27,2;");
            let _ = writeln!(b, "    add.s32 %r27,%r10,%r27;");
            let _ = writeln!(b, "    @%p1 st.shared.f32 [%r27],%f1;");
        }
    }
    let _ = writeln!(b);
    let _ = writeln!(b, "    bar.sync 0;");
    let _ = writeln!(
        b,
        "    // ---- unrolled R*S reduction; each X reused across KB channels ----"
    );
    for rr in 0..r {
        for ss in 0..s {
            let x_off = (rr * halo_w + ss) * 4; // bytes from compute base %r12
            let _ = writeln!(
                b,
                "    ld.shared.f32 %f1,[%r12+{x_off}];   // X[ty+{rr},tx+{ss}]"
            );
            for kk in 0..kb {
                let w_off = (halo + kk * rs + rr * s + ss) * 4; // bytes from smem base %r10
                let _ = writeln!(b, "    ld.shared.f32 %f2,[%r10+{w_off}];");
                let _ = writeln!(b, "    fma.rn.f32 %f{a},%f1,%f2,%f{a};", a = 8 + kk);
            }
        }
    }
    let _ = writeln!(b, "    bar.sync 0;");
    let _ = writeln!(b, "    add.u32 %r15,%r15,1;");
    let _ = writeln!(b, "    bra CLOOP;");
    let _ = writeln!(b, "CEND:");
    // store guard: op<P && oq<Q
    let _ = writeln!(b, "    setp.lt.u32 %p0,%r9,{p};");
    let _ = writeln!(b, "    setp.lt.u32 %p1,%r8,{q};");
    let _ = writeln!(b, "    and.pred %p0,%p0,%p1;");
    let _ = writeln!(b, "    @!%p0 bra RET;");
    // spat = op*Q + oq  ;  o0 = (k0)*P*Q + spat  ;  k0 = kb_idx*KB
    let _ = writeln!(
        b,
        "    mad.lo.s32 %r30,%r9,{q},%r8;       // spat = op*Q + oq"
    );
    let _ = writeln!(
        b,
        "    mul.lo.s32 %r31,%r5,{};           // k0*P*Q",
        kb * pq
    );
    let _ = writeln!(
        b,
        "    add.s32 %r31,%r31,%r30;           // o0 = k0*P*Q + spat"
    );
    for kk in 0..kb {
        let _ = writeln!(
            b,
            "    add.s32 %r32,%r31,{};         // oidx for channel +{kk}",
            kk * pq
        );
        let _ = writeln!(b, "    mul.wide.s32 %rd6,%r32,4;");
        let _ = writeln!(b, "    add.s64 %rd6,%rd3,%rd6;");
        let _ = writeln!(b, "    st.global.f32 [%rd6],%f{};", 8 + kk);
    }
    let _ = writeln!(b, "RET:");
    let _ = writeln!(b, "    ret;");
    let _ = writeln!(b, "}}");
    (b, mode)
}

/// Standalone **bias + ReLU** pointwise pass `O[i] = max(O[i] + bias[i/(P·Q)], 0)` over `O[K,P,Q]` (f32,
/// in-place), `bias[K]`. The *unfused* baseline for [`conv_wmma_epi_ptx`]: a plain conv writes `O`, then
/// this separate kernel re-reads + rewrites the whole `K·P·Q` output — the extra HBM round-trip + launch
/// that the fused epilogue elides. Entry `bias_relu`, params `(pOut, pBias)`, 1-D launch over `K·P·Q`.
pub fn bias_relu_ptx(k: usize, pq: usize) -> String {
    let total = k * pq;
    format!(
        r#"{HDR_SM80}
.visible .entry bias_relu(
    .param .u64 pOut,
    .param .u64 pBias
)
{{
    .reg .pred %p0;
    .reg .b32 %idx,%t,%n,%kc;
    .reg .f32 %v,%bv;
    .reg .b64 %O,%B,%off,%ptr;
    ld.param.u64 %O,[pOut];
    ld.param.u64 %B,[pBias];
    cvta.to.global.u64 %O,%O;
    cvta.to.global.u64 %B,%B;
    mov.u32 %t,%ctaid.x;
    mov.u32 %n,%ntid.x;
    mov.u32 %idx,%tid.x;
    mad.lo.s32 %idx,%t,%n,%idx;
    setp.ge.u32 %p0,%idx,{total};
    @%p0 bra RET;
    div.u32 %kc,%idx,{pq};
    mul.wide.u32 %off,%idx,4;
    add.s64 %ptr,%O,%off;
    ld.global.f32 %v,[%ptr];
    mul.wide.u32 %off,%kc,4;
    add.s64 %ptr,%B,%off;
    ld.global.f32 %bv,[%ptr];
    add.f32 %v,%v,%bv;
    max.f32 %v,%v,0f00000000;
    mul.wide.u32 %off,%idx,4;
    add.s64 %ptr,%O,%off;
    st.global.f32 [%ptr],%v;
RET:
    ret;
}}
"#
    )
}

/// `conv2d(C,H,W,K,R,S,P,Q, Xin,Wt,Out)` -- the original **naive** one-thread-per-output kernel. Output
/// index `idx = (k*P+p)*Q+q` is the linear thread id, so the store address is just `Out + idx`. Kept as
/// the honest worst-case reference and the fallback for shapes the tiled generator rejects.
/// (Header floor: this literal opens with exactly [`crate::ptx_target::HDR_SM80`] — it is a `const`
/// consumed as a `&'static str` by `gpu.rs`, so it cannot interpolate the constant; the
/// `conv_headers_are_at_the_sm80_floor` gate below asserts the two agree byte-for-byte.)
pub const CONV2D: &str = r#".version 7.8
.target sm_80
.address_size 64

.visible .entry conv2d(
    .param .u32 pC,
    .param .u32 pH,
    .param .u32 pW,
    .param .u32 pK,
    .param .u32 pR,
    .param .u32 pS,
    .param .u32 pP,
    .param .u32 pQ,
    .param .u64 pXin,
    .param .u64 pWt,
    .param .u64 pOut
)
{
    .reg .pred %p0,%pr,%ps;
    .reg .f32 %acc,%xv,%wv;
    .reg .b32 %C,%H,%W,%K,%R,%S,%P,%Q,%idx,%total,%t,%k,%p,%q,%c,%r,%s,%ih,%iw,%xidx,%widx,%tmp;
    .reg .b64 %X,%Wt,%O,%addr,%off;

    ld.param.u32 %C,[pC];
    ld.param.u32 %H,[pH];
    ld.param.u32 %W,[pW];
    ld.param.u32 %K,[pK];
    ld.param.u32 %R,[pR];
    ld.param.u32 %S,[pS];
    ld.param.u32 %P,[pP];
    ld.param.u32 %Q,[pQ];
    ld.param.u64 %X,[pXin];
    ld.param.u64 %Wt,[pWt];
    ld.param.u64 %O,[pOut];
    cvta.to.global.u64 %X,%X;
    cvta.to.global.u64 %Wt,%Wt;
    cvta.to.global.u64 %O,%O;

    // idx = blockIdx.x*blockDim.x + threadIdx.x
    mov.u32 %tmp,%ntid.x;
    mov.u32 %t,%ctaid.x;
    mul.lo.s32 %idx,%t,%tmp;
    mov.u32 %tmp,%tid.x;
    add.s32 %idx,%idx,%tmp;
    // total = K*P*Q ; bounds guard
    mul.lo.s32 %tmp,%K,%P;
    mul.lo.s32 %total,%tmp,%Q;
    setp.ge.u32 %p0,%idx,%total;
    @%p0 bra RET;

    // decode idx = (k*P + p)*Q + q
    div.u32 %t,%idx,%Q;
    mul.lo.s32 %tmp,%t,%Q;
    sub.s32 %q,%idx,%tmp;       // q = idx % Q
    div.u32 %k,%t,%P;
    mul.lo.s32 %tmp,%k,%P;
    sub.s32 %p,%t,%tmp;         // p = t % P, and k = t / P

    mov.f32 %acc,0f00000000;
    mov.u32 %c,0;
CLOOP:
    setp.ge.u32 %p0,%c,%C;
    @%p0 bra CEND;
    mov.u32 %r,0;
RLOOP:
    setp.ge.u32 %pr,%r,%R;
    @%pr bra REND;
    mov.u32 %s,0;
SLOOP:
    setp.ge.u32 %ps,%s,%S;
    @%ps bra SEND;
    // ih = p + r ; iw = q + s   (stride 1, no padding)
    add.s32 %ih,%p,%r;
    add.s32 %iw,%q,%s;
    // xidx = (c*H + ih)*W + iw
    mul.lo.s32 %xidx,%c,%H;
    add.s32 %xidx,%xidx,%ih;
    mul.lo.s32 %xidx,%xidx,%W;
    add.s32 %xidx,%xidx,%iw;
    // widx = ((k*C + c)*R + r)*S + s
    mul.lo.s32 %widx,%k,%C;
    add.s32 %widx,%widx,%c;
    mul.lo.s32 %widx,%widx,%R;
    add.s32 %widx,%widx,%r;
    mul.lo.s32 %widx,%widx,%S;
    add.s32 %widx,%widx,%s;
    // acc += X[xidx] * Wt[widx]
    mul.wide.u32 %off,%xidx,4;
    add.s64 %addr,%X,%off;
    ld.global.f32 %xv,[%addr];
    mul.wide.u32 %off,%widx,4;
    add.s64 %addr,%Wt,%off;
    ld.global.f32 %wv,[%addr];
    fma.rn.f32 %acc,%xv,%wv,%acc;
    add.u32 %s,%s,1;
    bra SLOOP;
SEND:
    add.u32 %r,%r,1;
    bra RLOOP;
REND:
    add.u32 %c,%c,1;
    bra CLOOP;
CEND:
    // O[idx] = acc   (idx already equals (k*P+p)*Q+q)
    mul.wide.u32 %off,%idx,4;
    add.s64 %addr,%O,%off;
    st.global.f32 [%addr],%acc;
RET:
    ret;
}
"#;

// ===================================================================================================
// fp16 tensor-core **implicit GEMM** conv -- the cuDNN-class path.
// ===================================================================================================
//
// conv2d is a GEMM `O[M,N] = A[M,GK] . B[GK,N]` with M=K (output channels), N=P*Q (output spatial),
// GK=C*R*S (the reduction), where:
//   * A = weights, exactly the contiguous `W[K,C,R,S]` reinterpreted as `[K, C*R*S]` (row-major).
//   * B = the **im2col** of X: `B[gk, n] = X[c, p+r, q+s]` with `gk -> (c,r,s)` and `n -> (p,q)`.
//   * O is contiguous `[K,P,Q] == [M, P*Q]`, so the GEMM store `O[m*N+n]` lands exactly right.
// We never materialize the im2col buffer in HBM: a CTA stages a `BM x 16` weight tile and a
// `16 x BN` im2col tile into shared memory each K-step, computing the im2col gather addresses on the
// fly (every divisor -- RS, S, Q -- is a compile-time literal, so ptxas lowers the div/rem to
// multiply-shift). The MMA runs `m16n16k16` fp16 with f32 accumulate. M/N/GK that aren't tile
// multiples are zero-padded by the staging guards and masked at the store, so any shape is legal.

/// WMMA implicit-GEMM CTA output-tile rows (M = output channels K direction).
pub const WMMA_BM: usize = 64;
/// WMMA implicit-GEMM CTA output-tile cols (N = output spatial P*Q direction).
pub const WMMA_BN: usize = 64;
/// Warp grid in the M direction (each warp owns `WMMA_BM/WMMA_WM` rows).
pub const WMMA_WM: usize = 2;
/// Warp grid in the N direction (each warp owns `WMMA_BN/WMMA_WN` cols). Power of two (`warpCol` via mask).
pub const WMMA_WN: usize = 2;
/// CTA thread count for the WMMA implicit-GEMM conv (one warp per (WM,WN) cell).
pub const WMMA_THREADS: usize = WMMA_WM * WMMA_WN * 32;

/// **The implicit-GEMM family's shared-memory footprint in bytes — the single source.**
///
/// `bufs·(BM·16·2) + bufs·(16·BN·2) + BM·BN·4`: the staged fp16 `BM×16` weight tile and `16×BN` im2col
/// tile (`bufs=2` for the register-double-buffered variant, which alternates two of each), plus the f32
/// `BM×BN` epilogue scratch `smemC`. At the shipped `WMMA_BM=WMMA_BN=64` that is 20480 B single-buffered
/// and 24576 B double-buffered — 42 %/50 % of the ISA's static cap, so this family has by far the most
/// headroom in the file, and widening its CTA tile is the only conv configuration here that a bigger
/// device budget actually unlocks (128×64 = 38912 B still fits the static cap; 128×128 = 73728 B does
/// not, on any target).
///
/// LANDMINE for a future widening: `smemC` is only live *after* the K-loop, so it could alias
/// `smemA`+`smemB` and cut the 64×64 footprint from 20480 to 16384 B. That is a real lever and a
/// separate change — it would alter the emitted PTX of every shipped conv, so it does not belong in a
/// budget-plumbing commit.
pub const fn conv_wmma_smem_bytes(bm: usize, bn: usize, bufs: usize) -> usize {
    bufs * (bm * 16 * 2) + bufs * (16 * bn * 2) + bm * bn * 4
}

/// Whether the fp16 tensor-core implicit-GEMM conv is worth dispatching for this shape **within
/// `smem_budget` bytes**. It is *correct* for any valid conv (guards zero-pad partial tiles), but only
/// pays off once the GEMM has enough reduction depth and output tiles to feed the tensor cores; tiny
/// `GK` makes the per-K-step staging overhead dominate. Heuristic: reduction `C*R*S >= 16` and a
/// non-trivial spatial extent — plus [`conv_wmma_smem_bytes`] fitting the budget.
///
/// The SMEM term is constant today (the CTA tile is [`WMMA_BM`]×[`WMMA_BN`] compile-time constants, so
/// it is always 20480 B and always fits): it is here so the gate and the generator's assert read the
/// same closed form, and so a widened tile is declined at *dispatch* rather than at `cuModuleLoadData`.
pub fn wmma_applies_budget(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    smem_budget: usize,
) -> bool {
    if h < r || w < s || k < 1 || c < 1 {
        return false;
    }
    if conv_wmma_smem_bytes(WMMA_BM, WMMA_BN, 1) > smem_budget {
        return false;
    }
    let gk = c * r * s;
    let n = (h - r + 1) * (w - s + 1);
    gk >= 16 && n >= 16 && k >= 16
}

/// [`wmma_applies_budget`] at the **static** ceiling — the verdict for the launchers in `gpu.rs`, all
/// of which declare their tile as a `.shared` array and launch with `shared_mem_bytes: 0`.
pub fn wmma_applies(c: usize, h: usize, w: usize, k: usize, r: usize, s: usize) -> bool {
    wmma_applies_budget(c, h, w, k, r, s, STATIC_SMEM_CAP)
}

/// Emit the fp16 tensor-core implicit-GEMM conv specialized to `[C,H,W] (*) [K,C,R,S] -> [K,P,Q]`.
/// Entry `conv2d_wmma`. **Inputs are f16** (`X`,`W`); output is f32. A CTA of `WMMA_WM×WMMA_WN` warps
/// stages a `WMMA_BM×16` weight tile and a `16×WMMA_BN` im2col tile into SMEM each K-step (reused by
/// all warps), then every warp computes its `tm×tn` grid of `m16n16k16` tiles out of SMEM. Launch with
/// block `(WMMA_THREADS,1,1)` and grid `(ceil(N/WMMA_BN), ceil(M/WMMA_BM), 1)` where `M=K`, `N=P*Q`.
pub fn conv_wmma_ptx(c: usize, h: usize, w: usize, k: usize, r: usize, s: usize) -> String {
    conv_wmma_ptx_budget(c, h, w, k, r, s, STATIC_SMEM_CAP).0
}

/// [`conv_wmma_ptx`] against an explicit shared-memory budget, returning the module **and the
/// [`SmemMode`] its launch must honour** — the seam every other variant in this family routes through
/// (`conv_wmma_ptx_impl` takes the same budget).
///
/// The CTA tile is [`WMMA_BM`]×[`WMMA_BN`] compile-time constants, so the footprint
/// ([`conv_wmma_smem_bytes`]) is 20480 B for *every* shape and the mode is always
/// [`SmemMode::Static`] — the budget is a ceiling assert here, not a switch. It exists so that the day
/// the tile is parameterized (`gpu.rs`'s `conv_wmma_cfg` grid/block must move in the same commit) the
/// ceiling is already checked at generation instead of surfacing as an opaque
/// `ptxas error: uses too much shared data` out of `cuModuleLoadData`.
pub fn conv_wmma_ptx_budget(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    smem_budget: usize,
) -> (String, SmemMode) {
    let ptx = conv_wmma_ptx_impl(
        c,
        h,
        w,
        k,
        r,
        s,
        1,
        crate::ptx_wmma::Act::None,
        false,
        1,
        0,
        smem_budget,
    );
    (
        ptx,
        smem_mode_for(conv_wmma_smem_bytes(WMMA_BM, WMMA_BN, 1)),
    )
}

/// **Split-K** variant of [`conv_wmma_ptx`] (entry `conv2d_wmma_splitk`): the `GK=C*R*S` reduction is
/// sliced across `gridDim.z = sk`, each CTA accumulating only its `GK/sk` slice into its **own** `M×N`
/// plane of a `sk·M·N` partial buffer (disjoint planes — no atomics), then summed by
/// [`conv_splitk_reduce_ptx`] in a fixed ascending-`z` order (deterministic, M12-safe). The lever for
/// **deep-channel, small-spatial** convs whose base grid under-fills the SMs (e.g. C256 14×14 → only
/// `ceil(256/64)·ceil(144/64)=12` CTAs; `sk` multiplies the resident-CTA count that hides the deep-GK
/// latency). Requires `sk` to divide `GK`.
pub fn conv_wmma_splitk_ptx(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    sk: usize,
) -> String {
    conv_wmma_ptx_impl(
        c,
        h,
        w,
        k,
        r,
        s,
        sk,
        crate::ptx_wmma::Act::None,
        false,
        1,
        0,
        STATIC_SMEM_CAP,
    )
}

/// **Fused conv + bias + activation** implicit-GEMM (entry `conv2d_wmma`): the same tensor-core conv as
/// [`conv_wmma_ptx`], but the f32-accumulate store epilogue folds in a per-output-channel bias
/// (`out += bias[k]`, when `bias`) and an activation (`out = act(out)`) — `act(x·Wᵀ + bias)` in one
/// kernel, racing cuDNN's `cudnnConvolutionBiasActivationForward`. **Free**: the epilogue runs on the
/// f32 accumulators already in the smemC store-back, so an unfused conv + a separate bias/act pass (an
/// extra full HBM round-trip of the `K·P·Q` output) collapses to a few instructions. Only `sk=1`
/// (fusion must apply *after* a split-K reduce, not per-slice — the recognizer keeps split-K unfused).
pub fn conv_wmma_epi_ptx(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    act: crate::ptx_wmma::Act,
    bias: bool,
) -> String {
    conv_wmma_ptx_impl(c, h, w, k, r, s, 1, act, bias, 1, 0, STATIC_SMEM_CAP)
}

/// **Strided** implicit-GEMM conv (entry `conv2d_wmma`): downsampling conv with `stride>1`, output
/// `P=⌊(H-R)/stride⌋+1`, `Q=⌊(W-S)/stride⌋+1`. Same tensor-core kernel as [`conv_wmma_ptx`]; only the
/// im2col gather changes — output pixel `(p,q)` reads input `(p·stride+r, q·stride+s)` (the hoisted
/// `xpart` scales by `stride`). Single-pass (no split-K / fusion threaded here — orthogonal). `stride=1`
/// reproduces the dense kernel byte-for-byte.
pub fn conv_wmma_strided_ptx(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    stride: usize,
) -> String {
    conv_wmma_ptx_impl(
        c,
        h,
        w,
        k,
        r,
        s,
        1,
        crate::ptx_wmma::Act::None,
        false,
        stride,
        0,
        STATIC_SMEM_CAP,
    )
}

/// **Strided + zero-padded** implicit-GEMM conv (entry `conv2d_wmma`): the general affine conv — output
/// `P=⌊(H+2·pad-R)/stride⌋+1`, `Q=⌊(W+2·pad-S)/stride⌋+1`; output pixel `(p,q)` reads input
/// `(p·stride+r-pad, q·stride+s-pad)`, with any coordinate outside `[0,H)×[0,W)` contributing 0. This is
/// the canonical "same" conv (e.g. 3×3 pad-1, or a ResNet stride-2 pad-1 downsample). `pad>0` switches
/// the im2col gather from the linear `xpart` fold to per-`(r,s)` **signed bounds checks** (the fold would
/// row-wrap once a coord leaves the image). `pad=0` is byte-identical to [`conv_wmma_strided_ptx`].
pub fn conv_wmma_pad_ptx(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    stride: usize,
    pad: usize,
) -> String {
    conv_wmma_ptx_impl(
        c,
        h,
        w,
        k,
        r,
        s,
        1,
        crate::ptx_wmma::Act::None,
        false,
        stride,
        pad,
        STATIC_SMEM_CAP,
    )
}

/// **Split-K** affine conv (entry `conv2d_wmma_splitk`): the strided+padded [`conv_wmma_pad_ptx`] with
/// its `GK=C·R·S` reduction sliced across `gridDim.z = sk` (disjoint `M×N` partial planes, summed by
/// [`conv_splitk_reduce_ptx`] in fixed ascending-`z` order — deterministic). The occupancy lever for the
/// **deep-channel, small-spatial downsamples** (e.g. C128 28²→14², whose base grid is only ≈8 CTAs).
/// Split-K and padding compose freely — the per-tap bounds check is independent of the K-slice. Requires
/// `sk | GK` and `GK/sk` a multiple of 16.
#[allow(clippy::too_many_arguments)]
pub fn conv_wmma_pad_splitk_ptx(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    stride: usize,
    pad: usize,
    sk: usize,
) -> String {
    conv_wmma_ptx_impl(
        c,
        h,
        w,
        k,
        r,
        s,
        sk,
        crate::ptx_wmma::Act::None,
        false,
        stride,
        pad,
        STATIC_SMEM_CAP,
    )
}

/// **Explicit zero-pad scatter** (entry `pad_nchw_copy`): copy `X[C,H,W]` (fp16) into the interior of a
/// pre-zeroed `Xpad[C, H+2·pad, W+2·pad]` (fp16), i.e. `Xpad[c, i+pad, j+pad] = X[c, i, j]`. Lets a padded
/// conv run as the **dense valid kernel** ([`conv_wmma_strided_ptx`] on the padded dims) — no per-tap
/// bounds checks, no extra hoisted registers in the GEMM. (Measured ~equal to the single-kernel
/// bounds-checked gather; see [`super::gpu::conv2d_wmma_padded_explicit`].) The scatter is one streamed
/// copy of `C·H·W` halfwords (<1% of the conv). Launch with grid `ceil(C·H·W / 256)`, block 256.
/// `Xpad` must be zeroed (e.g. `alloc_zeros`) so the border stays 0.
pub fn pad_nchw_copy_ptx(c: usize, h: usize, w: usize, pad: usize) -> String {
    use std::fmt::Write as _;
    let hw = h * w;
    let total = c * hw;
    let hp = h + 2 * pad;
    let wp = w + 2 * pad;
    let mut b = String::new();
    let _ = writeln!(b, "{HDR_SM80}");
    let _ = writeln!(
        b,
        "// zero-pad scatter: X[C{c} H{h} W{w}] -> Xpad[C {hp} {wp}] interior (fp16)"
    );
    let _ = writeln!(b, ".visible .entry pad_nchw_copy(");
    let _ = writeln!(b, "    .param .u64 pXin,");
    let _ = writeln!(b, "    .param .u64 pXpad");
    let _ = writeln!(b, ")");
    let _ = writeln!(b, "{{");
    let _ = writeln!(b, "    .reg .pred %p0;");
    let _ = writeln!(b, "    .reg .b16 %hv;");
    let _ = writeln!(b, "    .reg .b32 %gid,%cc,%rem,%ii,%jj,%dst,%ntx;");
    let _ = writeln!(b, "    .reg .b64 %X,%Xp,%off,%ptr;");
    let _ = writeln!(b);
    let _ = writeln!(b, "    ld.param.u64 %X,[pXin];");
    let _ = writeln!(b, "    ld.param.u64 %Xp,[pXpad];");
    let _ = writeln!(b, "    cvta.to.global.u64 %X,%X;");
    let _ = writeln!(b, "    cvta.to.global.u64 %Xp,%Xp;");
    // global thread id = ctaid.x*ntid.x + tid.x  (avoid naming a reg %tid -- collides with %tid.x)
    let _ = writeln!(b, "    mov.u32 %gid,%ctaid.x;");
    let _ = writeln!(b, "    mov.u32 %ntx,%ntid.x;");
    let _ = writeln!(b, "    mul.lo.s32 %gid,%gid,%ntx;");
    let _ = writeln!(b, "    mov.u32 %rem,%tid.x;");
    let _ = writeln!(
        b,
        "    add.u32 %gid,%gid,%rem;            // global thread id"
    );
    let _ = writeln!(b, "    setp.ge.u32 %p0,%gid,{total};");
    let _ = writeln!(b, "    @%p0 bra DONE;");
    // decode gid -> (c, i, j)
    let _ = writeln!(b, "    div.u32 %cc,%gid,{hw};            // c = gid/(H*W)");
    let _ = writeln!(
        b,
        "    rem.u32 %rem,%gid,{hw};           // rem = gid%(H*W)"
    );
    let _ = writeln!(b, "    div.u32 %ii,%rem,{w};             // i = rem/W");
    let _ = writeln!(b, "    rem.u32 %jj,%rem,{w};             // j = rem%W");
    // dst = c*(Hp*Wp) + (i+pad)*Wp + (j+pad)
    let _ = writeln!(b, "    add.u32 %ii,%ii,{pad};            // i+pad");
    let _ = writeln!(b, "    add.u32 %jj,%jj,{pad};            // j+pad");
    let _ = writeln!(
        b,
        "    mad.lo.s32 %dst,%ii,{wp},%jj;     // (i+pad)*Wp + (j+pad)"
    );
    let _ = writeln!(
        b,
        "    mad.lo.s32 %dst,%cc,{},%dst;      // + c*Hp*Wp",
        hp * wp
    );
    // load X[gid], store Xpad[dst]
    let _ = writeln!(b, "    mul.wide.u32 %off,%gid,2;");
    let _ = writeln!(b, "    add.s64 %ptr,%X,%off;");
    let _ = writeln!(b, "    ld.global.u16 %hv,[%ptr];");
    let _ = writeln!(b, "    mul.wide.u32 %off,%dst,2;");
    let _ = writeln!(b, "    add.s64 %ptr,%Xp,%off;");
    let _ = writeln!(b, "    st.global.u16 [%ptr],%hv;");
    let _ = writeln!(b, "DONE:");
    let _ = writeln!(b, "    ret;");
    let _ = writeln!(b, "}}");
    b
}

#[allow(clippy::too_many_arguments)]
fn conv_wmma_ptx_impl(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    sk: usize,
    act: crate::ptx_wmma::Act,
    bias: bool,
    stride: usize,
    pad: usize,
    // The SMEM ceiling this entry may spend, in bytes. Every public wrapper passes `STATIC_SMEM_CAP`
    // (the PTX ISA's static `.shared` limit); a device budget arrives through `conv_wmma_ptx_budget`.
    // Constant-footprint today — see `conv_wmma_smem_bytes` — so this is a ceiling assert, kept here
    // because generation is the only place that can name the family, the tile and the byte count.
    smem_budget: usize,
) -> String {
    use std::fmt::Write as _;
    assert!(stride >= 1, "stride must be >= 1");
    // Strided/padded conv: output P=floor((H+2*pad-R)/stride)+1, the im2col gather reads input pixel
    // (p*stride+r-pad, q*stride+s-pad). pad==0,stride==1 is the dense conv (hoisted xpart fast path);
    // pad>0 switches the gather to per-(r,s) signed bounds checks (zero-pad OOB) since the linear
    // index fold would row-wrap once a coordinate leaves [0,H)×[0,W).
    assert!(
        h + 2 * pad >= r && w + 2 * pad >= s,
        "kernel larger than padded input"
    );
    let p = (h + 2 * pad - r) / stride + 1;
    let q = (w + 2 * pad - s) / stride + 1;
    let m = k; // GEMM M
    let n = p * q; // GEMM N
    let gk = c * r * s; // GEMM K (reduction)
    assert!(
        sk >= 1 && gk % sk == 0,
        "split-K factor {sk} must divide GK={gk}"
    );
    // Fusion composes only with the single-pass kernel: a split-K conv writes partial M×N planes that
    // are summed later, so a per-slice bias/activation would be applied sk× (and act before the sum).
    assert!(
        sk == 1 || (matches!(act, crate::ptx_wmma::Act::None) && !bias),
        "split-K conv cannot fuse bias/act"
    );
    let gk_per = gk / sk; // reduction length per z-slice
                          // The K-loop advances in 16-wide WMMA tiles, so each slice must be a whole number of them — else a
                          // slice over-reads into the next slice's range (the staging guards against the full GK, not ktend).
    assert!(
        sk == 1 || gk_per % 16 == 0,
        "split-K slice GK/sk={gk_per} must be a multiple of 16"
    );
    let mn = m * n; // one output plane (for the z-plane store offset)
    let entry = if sk > 1 {
        "conv2d_wmma_splitk"
    } else {
        "conv2d_wmma"
    };
    let rs = r * s;
    let hw = h * w;
    let (bm, bn) = (WMMA_BM, WMMA_BN);
    let (warps_m, warps_n) = (WMMA_WM, WMMA_WN);
    let threads = WMMA_THREADS;
    let wm = bm / warps_m; // rows owned by a warp
    let wn = bn / warps_n; // cols owned by a warp
    let tm = wm / 16; // 16×16 tiles per warp, M
    let tn = wn / 16; // 16×16 tiles per warp, N
    let wn_shift = warps_n.trailing_zeros(); // warpId / warps_n
    let bn_shift = bn.trailing_zeros();
    let a_per = bm * 16 / threads; // A elements staged per thread
    let b_per = 16 * bn / threads; // B elements staged per thread
    let c_per = bm * bn / threads; // C elements written per thread
    let smem_a = bm * 16 * 2; // f16 bytes
    let smem_b = 16 * bn * 2;
    let smem_c = bm * bn * 4; // f32 store scratch
                              // The family's closed form, from the one place both the gate and this declaration read. The
                              // budget is the CEILING; `STATIC_SMEM_CAP` is the boundary between the two emission forms —
                              // this family is on the static side at every shipped tile, which the second assert pins.
    let smem_total = conv_wmma_smem_bytes(bm, bn, 1);
    debug_assert_eq!(smem_total, smem_a + smem_b + smem_c);
    assert!(
        smem_total <= smem_budget,
        "conv2d_wmma: SMEM {smem_total} B ({bm}x{bn} tile) exceeds the budget {smem_budget} B"
    );
    assert!(
        !smem_mode_for(smem_total).is_dynamic(),
        "conv2d_wmma: {smem_total} B needs the dynamic window, which this generator does not emit \
         (widening the CTA tile must land with the `.extern` window AND gpu.rs's conv_wmma_cfg)"
    );

    let veclist = |pre: &str| -> String {
        let regs: Vec<String> = (0..8).map(|i| format!("%{pre}{i}")).collect();
        format!("{{{}}}", regs.join(","))
    };

    let mut b = String::new();
    let _ = writeln!(b, "{HDR_SM80}");
    let _ = writeln!(
        b,
        "// fp16 tensor-core implicit-GEMM conv: C{c} H{h} W{w} K{k} R{r} S{s}"
    );
    let _ = writeln!(
        b,
        "// M={m} N={n} GK={gk}; CTA tile {bm}x{bn}, {warps_m}x{warps_n} warps, per-warp {wm}x{wn}"
    );
    if sk > 1 {
        let _ = writeln!(
            b,
            "// split-K: gridDim.z={sk} z-slices of GK/{sk}={gk_per}, disjoint M*N partial planes"
        );
    }
    let _ = writeln!(b, ".visible .entry {entry}(");
    let _ = writeln!(b, "    .param .u64 pXin,");
    let _ = writeln!(b, "    .param .u64 pWt,");
    if bias {
        let _ = writeln!(b, "    .param .u64 pOut,");
        let _ = writeln!(b, "    .param .u64 pBias");
    } else {
        let _ = writeln!(b, "    .param .u64 pOut");
    }
    let _ = writeln!(b, ")");
    let _ = writeln!(b, "{{");
    let _ = writeln!(b, "    .shared .align 16 .b8 smemA[{smem_a}];");
    let _ = writeln!(b, "    .shared .align 16 .b8 smemB[{smem_b}];");
    let _ = writeln!(b, "    .shared .align 16 .b8 smemC[{smem_c}];");
    let _ = writeln!(b, "    .reg .pred %p0,%pv;");
    let _ = writeln!(b, "    .reg .b16 %hv;");
    let _ = writeln!(
        b,
        "    .reg .b32 %tix,%m0,%n0,%kt,%e,%mm,%gkk,%ncol,%gkv,%nn,%cc,%rem,%rr,%ss,%pp,%qq,%ih,%iw,%xidx,%widx,%tmp,%tmp2,%saddr,%warpId,%wrb,%wcb;"
    );
    // Fused-epilogue scratch: bias value + activation temporaries (dropped by ptxas when unused).
    let _ = writeln!(b, "    .reg .f32 %bv,%act0,%act1;");
    if bias {
        let _ = writeln!(b, "    .reg .b64 %Bias;");
    }
    if sk > 1 {
        let _ = writeln!(b, "    .reg .b32 %zsl,%ktend;");
    }
    // accumulator + a/b fragments
    let mut decl = String::new();
    for ti in 0..tm {
        for tj in 0..tn {
            for rr in 0..8 {
                decl += &format!("%c{ti}_{tj}_{rr},");
            }
        }
    }
    for ti in 0..tm {
        for rr in 0..8 {
            decl += &format!("%a{ti}_{rr},");
        }
    }
    for tj in 0..tn {
        for rr in 0..8 {
            decl += &format!("%b{tj}_{rr},");
        }
    }
    // Per B-staging slot: the K-independent im2col decode (n=n0+ncol, then p,q) hoisted out of the
    // K-loop -- each slot's (ncol) is fixed across K-steps, so p,q (a div+rem) need computing once.
    // pad==0 folds (p,q) into a single linear xpart=p*W+q; pad>0 keeps the signed top-left input
    // coords (ph=p*stride-pad, pw=q*stride-pad) so the K-loop can bounds-check ih=ph+r, iw=pw+s.
    for li in 0..b_per {
        decl += &format!("%bnv{li},");
        if pad == 0 {
            decl += &format!("%bxp{li},");
        } else {
            decl += &format!("%bph{li},%bpw{li},");
        }
    }
    let _ = writeln!(b, "    .reg .f32 %cf;");
    let _ = writeln!(b, "    .reg .b32 {};", decl.trim_end_matches(','));
    let _ = writeln!(b, "    .reg .b64 %X,%W,%O,%off,%gp,%ptr;");
    let _ = writeln!(b);
    let _ = writeln!(b, "    ld.param.u64 %X,[pXin];");
    let _ = writeln!(b, "    ld.param.u64 %W,[pWt];");
    let _ = writeln!(b, "    ld.param.u64 %O,[pOut];");
    let _ = writeln!(b, "    cvta.to.global.u64 %X,%X;");
    let _ = writeln!(b, "    cvta.to.global.u64 %W,%W;");
    let _ = writeln!(b, "    cvta.to.global.u64 %O,%O;");
    let _ = writeln!(b, "    mov.u32 %tix,%tid.x;");
    let _ = writeln!(b, "    mov.u32 %tmp,%ctaid.y;");
    let _ = writeln!(b, "    mul.lo.s32 %m0,%tmp,{bm};      // CTA row base (M)");
    let _ = writeln!(b, "    mov.u32 %tmp,%ctaid.x;");
    let _ = writeln!(b, "    mul.lo.s32 %n0,%tmp,{bn};      // CTA col base (N)");
    // warp partition: warpId -> (warpRow, warpCol); warpRowBase = warpRow*wm, warpColBase = warpCol*wn
    let _ = writeln!(b, "    shr.u32 %warpId,%tix,5;");
    let _ = writeln!(b, "    shr.u32 %tmp,%warpId,{wn_shift};       // warpRow");
    let _ = writeln!(
        b,
        "    mul.lo.s32 %wrb,%tmp,{wm};            // warpRowBase"
    );
    let _ = writeln!(
        b,
        "    and.b32 %tmp,%warpId,{};             // warpCol",
        warps_n - 1
    );
    let _ = writeln!(
        b,
        "    mul.lo.s32 %wcb,%tmp,{wn};            // warpColBase"
    );
    // zero accumulators
    for ti in 0..tm {
        for tj in 0..tn {
            for rr in 0..8 {
                let _ = writeln!(b, "    mov.f32 %c{ti}_{tj}_{rr},0f00000000;");
            }
        }
    }
    // Hoisted im2col decode: per B-staging slot compute nn=n0+ncol and xpart=p*W+q once (K-independent).
    let _ = writeln!(
        b,
        "    // ---- hoist K-independent im2col decode (n -> p,q -> xpart) ----"
    );
    for li in 0..b_per {
        let off = li * threads;
        let _ = writeln!(b, "    add.u32 %e,%tix,{off};");
        let _ = writeln!(b, "    and.b32 %ncol,%e,{};         // ncol = e%BN", bn - 1);
        let _ = writeln!(b, "    add.u32 %bnv{li},%n0,%ncol;   // nn = n0+ncol");
        let _ = writeln!(b, "    div.u32 %pp,%bnv{li},{q};     // p = nn/Q");
        let _ = writeln!(b, "    rem.u32 %qq,%bnv{li},{q};     // q = nn%Q");
        if pad == 0 {
            if stride == 1 {
                let _ = writeln!(
                    b,
                    "    mad.lo.s32 %bxp{li},%pp,{w},%qq;   // xpart = p*W + q"
                );
            } else {
                // strided: input pixel (p*stride+r, q*stride+s) -> xpart = (p*stride)*W + q*stride
                let _ = writeln!(b, "    mul.lo.s32 %qq,%qq,{stride};   // q*stride");
                let _ = writeln!(
                    b,
                    "    mad.lo.s32 %bxp{li},%pp,{},%qq;   // (p*stride)*W + q*stride",
                    stride * w
                );
            }
        } else {
            // padded: hoist signed top-left input coords ph=p*stride-pad, pw=q*stride-pad (may be <0).
            let _ = writeln!(b, "    mul.lo.s32 %tmp,%pp,{stride};");
            let _ = writeln!(
                b,
                "    sub.s32 %bph{li},%tmp,{pad};   // ph = p*stride - pad"
            );
            let _ = writeln!(b, "    mul.lo.s32 %tmp,%qq,{stride};");
            let _ = writeln!(
                b,
                "    sub.s32 %bpw{li},%tmp,{pad};   // pw = q*stride - pad"
            );
        }
    }
    if sk > 1 {
        let _ = writeln!(b, "    mov.u32 %zsl,%ctaid.z;");
        let _ = writeln!(b, "    mul.lo.s32 %kt,%zsl,{gk_per};   // gk0 = z*GK/sk");
        let _ = writeln!(
            b,
            "    add.u32 %ktend,%kt,{gk_per};    // gk_end = gk0+GK/sk"
        );
    } else {
        let _ = writeln!(b, "    mov.u32 %kt,0;              // gk0");
    }
    let _ = writeln!(b, "KLOOP:");
    if sk > 1 {
        let _ = writeln!(b, "    setp.ge.u32 %p0,%kt,%ktend;");
    } else {
        let _ = writeln!(b, "    setp.ge.u32 %p0,%kt,{gk};");
    }
    let _ = writeln!(b, "    @%p0 bra KEND;");
    let _ = writeln!(b);
    let _ = writeln!(
        b,
        "    // ---- stage A (weights [M,GK]) into smemA[BM][16] ----"
    );
    for li in 0..a_per {
        let off = li * threads;
        let _ = writeln!(b, "    add.u32 %e,%tix,{off};");
        let _ = writeln!(b, "    shr.u32 %mm,%e,4;             // m = e/16");
        let _ = writeln!(b, "    and.b32 %gkk,%e,15;          // gkk = e%16");
        let _ = writeln!(b, "    add.u32 %tmp,%m0,%mm;        // gm = m0+m");
        let _ = writeln!(b, "    add.u32 %tmp2,%kt,%gkk;      // gc = gk0+gkk");
        let _ = writeln!(b, "    setp.lt.u32 %pv,%tmp,{m};");
        let _ = writeln!(b, "    setp.lt.u32 %p0,%tmp2,{gk};");
        let _ = writeln!(b, "    and.pred %pv,%pv,%p0;");
        let _ = writeln!(b, "    mad.lo.s32 %widx,%tmp,{gk},%tmp2;   // gm*GK + gc");
        let _ = writeln!(b, "    mul.wide.u32 %off,%widx,2;");
        let _ = writeln!(b, "    add.s64 %ptr,%W,%off;");
        let _ = writeln!(b, "    mov.u16 %hv,0;");
        let _ = writeln!(b, "    @%pv ld.global.u16 %hv,[%ptr];");
        let _ = writeln!(b, "    mov.u32 %saddr,smemA;");
        let _ = writeln!(b, "    shl.b32 %tmp,%e,1;");
        let _ = writeln!(b, "    add.u32 %saddr,%saddr,%tmp;");
        let _ = writeln!(b, "    st.shared.u16 [%saddr],%hv;");
    }
    let _ = writeln!(b);
    let _ = writeln!(
        b,
        "    // ---- stage B (im2col of X) into smemB[16][BN]; n-decode is hoisted ----"
    );
    for li in 0..b_per {
        let off = li * threads;
        let _ = writeln!(b, "    add.u32 %e,%tix,{off};");
        let _ = writeln!(b, "    shr.u32 %gkk,%e,{bn_shift};          // gkk = e/BN");
        let _ = writeln!(b, "    add.u32 %gkv,%kt,%gkk;       // gk = gk0+gkk");
        let _ = writeln!(b, "    setp.lt.u32 %pv,%gkv,{gk};");
        let _ = writeln!(
            b,
            "    setp.lt.u32 %p0,%bnv{li},{n};   // nn<N (nn precomputed)"
        );
        let _ = writeln!(b, "    and.pred %pv,%pv,%p0;");
        // decode only gk -> (c,r,s); (p,q) already folded into xpart=p*W+q.
        let _ = writeln!(b, "    div.u32 %cc,%gkv,{rs};       // c = gk/(R*S)");
        let _ = writeln!(b, "    rem.u32 %rem,%gkv,{rs};      // rem = gk%(R*S)");
        let _ = writeln!(b, "    div.u32 %rr,%rem,{s};        // r = rem/S");
        let _ = writeln!(b, "    rem.u32 %ss,%rem,{s};        // s = rem%S");
        if pad == 0 {
            // xidx = c*H*W + (p*stride+r)*W + (q*stride+s) = c*H*W + xpart + r*W + s
            let _ = writeln!(
                b,
                "    mad.lo.s32 %xidx,%cc,{hw},%bxp{li};   // c*H*W + xpart"
            );
            let _ = writeln!(b, "    mad.lo.s32 %xidx,%rr,{w},%xidx;  // + r*W");
            let _ = writeln!(b, "    add.u32 %xidx,%xidx,%ss;     // + s");
        } else {
            // padded: ih=ph+r, iw=pw+s (signed); AND in-bounds [0,H)x[0,W) into %pv (OOB -> hv=0 zero-pad).
            // Single UNSIGNED compare per axis: 0<=ih<H  <=>  (u32)ih < H (a negative ih wraps to a huge
            // u32 >= H, so it fails) -- halves the bounds chain (2 setp+2 and.pred, not 4).
            let _ = writeln!(b, "    add.s32 %ih,%bph{li},%rr;    // ih = ph + r");
            let _ = writeln!(b, "    add.s32 %iw,%bpw{li},%ss;    // iw = pw + s");
            let _ = writeln!(
                b,
                "    setp.lt.u32 %p0,%ih,{h};     // 0<=ih<H (unsigned wrap)"
            );
            let _ = writeln!(b, "    and.pred %pv,%pv,%p0;");
            let _ = writeln!(
                b,
                "    setp.lt.u32 %p0,%iw,{w};     // 0<=iw<W (unsigned wrap)"
            );
            let _ = writeln!(b, "    and.pred %pv,%pv,%p0;");
            let _ = writeln!(b, "    mad.lo.s32 %xidx,%ih,{w},%iw;    // ih*W + iw");
            let _ = writeln!(b, "    mad.lo.s32 %xidx,%cc,{hw},%xidx; // + c*H*W");
        }
        let _ = writeln!(b, "    mul.wide.u32 %off,%xidx,2;");
        let _ = writeln!(b, "    add.s64 %ptr,%X,%off;");
        let _ = writeln!(b, "    mov.u16 %hv,0;");
        let _ = writeln!(b, "    @%pv ld.global.u16 %hv,[%ptr];");
        let _ = writeln!(b, "    mov.u32 %saddr,smemB;");
        let _ = writeln!(b, "    shl.b32 %tmp,%e,1;");
        let _ = writeln!(b, "    add.u32 %saddr,%saddr,%tmp;");
        let _ = writeln!(b, "    st.shared.u16 [%saddr],%hv;");
    }
    let _ = writeln!(b);
    let _ = writeln!(b, "    bar.sync 0;");
    // load A fragments (row, ldm=16): warp row base = warpRowBase + ti*16
    let _ = writeln!(b, "    mov.u32 %tmp,16;");
    for ti in 0..tm {
        let _ = writeln!(
            b,
            "    add.u32 %tmp2,%wrb,{};        // smem row = warpRowBase+{}",
            ti * 16,
            ti * 16
        );
        let _ = writeln!(b, "    mul.lo.s32 %tmp2,%tmp2,32;    // *16*2 bytes");
        let _ = writeln!(b, "    mov.u32 %saddr,smemA;");
        let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,%saddr;");
        let _ = writeln!(b, "    cvt.u64.u32 %gp,%tmp2;");
        let _ = writeln!(b, "    cvta.shared.u64 %gp,%gp;");
        let ra = veclist(&format!("a{ti}_"));
        let _ = writeln!(
            b,
            "    wmma.load.a.sync.aligned.m16n16k16.row.f16 {ra}, [%gp], %tmp;"
        );
    }
    // load B fragments (row, ldm=BN): warp col base = warpColBase + tj*16
    let _ = writeln!(b, "    mov.u32 %tmp,{bn};");
    for tj in 0..tn {
        let _ = writeln!(
            b,
            "    add.u32 %tmp2,%wcb,{};        // smem col = warpColBase+{}",
            tj * 16,
            tj * 16
        );
        let _ = writeln!(b, "    shl.b32 %tmp2,%tmp2,1;        // *2 bytes");
        let _ = writeln!(b, "    mov.u32 %saddr,smemB;");
        let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,%saddr;");
        let _ = writeln!(b, "    cvt.u64.u32 %gp,%tmp2;");
        let _ = writeln!(b, "    cvta.shared.u64 %gp,%gp;");
        let rb = veclist(&format!("b{tj}_"));
        let _ = writeln!(
            b,
            "    wmma.load.b.sync.aligned.m16n16k16.row.f16 {rb}, [%gp], %tmp;"
        );
    }
    for ti in 0..tm {
        let ra = veclist(&format!("a{ti}_"));
        for tj in 0..tn {
            let rb = veclist(&format!("b{tj}_"));
            let cc = veclist(&format!("c{ti}_{tj}_"));
            let _ = writeln!(
                b,
                "    wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {cc}, {ra}, {rb}, {cc};"
            );
        }
    }
    let _ = writeln!(b, "    bar.sync 0;");
    let _ = writeln!(b, "    add.u32 %kt,%kt,16;");
    let _ = writeln!(b, "    bra KLOOP;");
    let _ = writeln!(b, "KEND:");
    // store each warp tile to smemC[BM][BN] (ldm=BN) at (warpRowBase+ti*16, warpColBase+tj*16).
    let _ = writeln!(b, "    mov.u32 %tmp,{bn};");
    for ti in 0..tm {
        for tj in 0..tn {
            let _ = writeln!(
                b,
                "    add.u32 %tmp2,%wrb,{};        // row = warpRowBase+{}",
                ti * 16,
                ti * 16
            );
            let _ = writeln!(b, "    mul.lo.s32 %tmp2,%tmp2,{bn};   // row*BN");
            let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,%wcb;     // + warpColBase");
            let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,{};        // + tj*16", tj * 16);
            let _ = writeln!(b, "    shl.b32 %tmp2,%tmp2,2;        // *4 bytes");
            let _ = writeln!(b, "    mov.u32 %saddr,smemC;");
            let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,%saddr;");
            let _ = writeln!(b, "    cvt.u64.u32 %gp,%tmp2;");
            let _ = writeln!(b, "    cvta.shared.u64 %gp,%gp;");
            let cc = veclist(&format!("c{ti}_{tj}_"));
            let _ = writeln!(
                b,
                "    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%gp], {cc}, %tmp;"
            );
        }
    }
    let _ = writeln!(b, "    bar.sync 0;");
    if bias {
        let _ = writeln!(b, "    ld.param.u64 %Bias,[pBias];");
        let _ = writeln!(b, "    cvta.to.global.u64 %Bias,%Bias;");
    }
    for li in 0..c_per {
        let off = li * threads;
        let _ = writeln!(b, "    add.u32 %e,%tix,{off};");
        let _ = writeln!(b, "    shr.u32 %mm,%e,{bn_shift};           // m = e/BN");
        let _ = writeln!(b, "    and.b32 %ncol,%e,{};         // n = e%BN", bn - 1);
        let _ = writeln!(b, "    add.u32 %tmp,%m0,%mm;        // gm");
        let _ = writeln!(b, "    add.u32 %nn,%n0,%ncol;       // gn");
        let _ = writeln!(b, "    setp.lt.u32 %pv,%tmp,{m};");
        let _ = writeln!(b, "    setp.lt.u32 %p0,%nn,{n};");
        let _ = writeln!(b, "    and.pred %pv,%pv,%p0;");
        let _ = writeln!(b, "    mov.u32 %saddr,smemC;");
        let _ = writeln!(b, "    shl.b32 %tmp2,%e,2;");
        let _ = writeln!(b, "    add.u32 %saddr,%saddr,%tmp2;");
        let _ = writeln!(b, "    ld.shared.f32 %cf,[%saddr];");
        if bias {
            // out += bias[gm] (gm = %tmp = the output channel). Guarded so padded gm>=K never faults.
            let _ = writeln!(b, "    mul.wide.u32 %off,%tmp,4;");
            let _ = writeln!(b, "    add.s64 %ptr,%Bias,%off;");
            let _ = writeln!(b, "    mov.f32 %bv,0f00000000;");
            let _ = writeln!(b, "    @%pv ld.global.f32 %bv,[%ptr];");
            let _ = writeln!(b, "    add.f32 %cf,%cf,%bv;");
        }
        b.push_str(&act.epilogue("%cf")); // out = act(out)  (Act::None -> nothing)
        let _ = writeln!(b, "    mad.lo.s32 %xidx,%tmp,{n},%nn;   // gm*N + gn");
        if sk > 1 {
            let _ = writeln!(
                b,
                "    mad.lo.s32 %xidx,%zsl,{mn},%xidx;   // + z*M*N (disjoint plane)"
            );
        }
        let _ = writeln!(b, "    mul.wide.u32 %off,%xidx,4;");
        let _ = writeln!(b, "    add.s64 %ptr,%O,%off;");
        let _ = writeln!(b, "    @%pv st.global.f32 [%ptr],%cf;");
    }
    let _ = writeln!(b, "    ret;");
    let _ = writeln!(b, "}}");
    b
}

/// Reduce the `sk` disjoint `M×N` partial planes [`conv_wmma_splitk_ptx`] produced into the final
/// `O[M,N]` (`= [K,P,Q]`), summing in **fixed ascending-`z` order** so the result is bit-reproducible
/// (M12 determinism — a float `atomicAdd` split-K could not be). One thread per output element; `sk`
/// and `mn = M*N` are baked in (the per-plane stride is a small constant `add.s64`, never a giant
/// immediate offset). Entry `conv_splitk_reduce`, params `(pPartial, pOut)`, launch 1-D over `M*N`.
pub fn conv_splitk_reduce_ptx(mn: usize, sk: usize) -> String {
    use std::fmt::Write as _;
    let plane = mn * 4; // bytes between successive z-planes
    let mut b = String::new();
    let _ = writeln!(b, "{HDR_SM80}");
    let _ = writeln!(
        b,
        "// split-K reduce: sum {sk} planes of {mn} f32 -> O, fixed z-order (deterministic)"
    );
    let _ = writeln!(b, ".visible .entry conv_splitk_reduce(");
    let _ = writeln!(b, "    .param .u64 pPartial,");
    let _ = writeln!(b, "    .param .u64 pOut");
    let _ = writeln!(b, ")");
    let _ = writeln!(b, "{{");
    let _ = writeln!(b, "    .reg .pred %p0;");
    let _ = writeln!(b, "    .reg .b32 %idx,%t,%tmp;");
    let _ = writeln!(b, "    .reg .f32 %acc,%v;");
    let _ = writeln!(b, "    .reg .b64 %P,%O,%base,%ptr,%off;");
    let _ = writeln!(b, "    ld.param.u64 %P,[pPartial];");
    let _ = writeln!(b, "    ld.param.u64 %O,[pOut];");
    let _ = writeln!(b, "    cvta.to.global.u64 %P,%P;");
    let _ = writeln!(b, "    cvta.to.global.u64 %O,%O;");
    let _ = writeln!(b, "    mov.u32 %tmp,%ntid.x;");
    let _ = writeln!(b, "    mov.u32 %t,%ctaid.x;");
    let _ = writeln!(b, "    mul.lo.s32 %idx,%t,%tmp;");
    let _ = writeln!(b, "    mov.u32 %tmp,%tid.x;");
    let _ = writeln!(b, "    add.s32 %idx,%idx,%tmp;");
    let _ = writeln!(b, "    setp.ge.u32 %p0,%idx,{mn};");
    let _ = writeln!(b, "    @%p0 bra RET;");
    let _ = writeln!(b, "    mul.wide.u32 %off,%idx,4;");
    let _ = writeln!(
        b,
        "    add.s64 %base,%P,%off;          // &partial[0*MN+idx]"
    );
    let _ = writeln!(b, "    mov.f32 %acc,0f00000000;");
    for z in 0..sk {
        let _ = writeln!(b, "    ld.global.f32 %v,[%base];");
        let _ = writeln!(b, "    add.f32 %acc,%acc,%v;       // += plane {z}");
        if z + 1 < sk {
            let _ = writeln!(b, "    add.s64 %base,%base,{plane};   // -> next z-plane");
        }
    }
    let _ = writeln!(b, "    add.s64 %ptr,%O,%off;");
    let _ = writeln!(b, "    st.global.f32 [%ptr],%acc;");
    let _ = writeln!(b, "RET:");
    let _ = writeln!(b, "    ret;");
    let _ = writeln!(b, "}}");
    b
}

/// Choose a split-K factor for the implicit-GEMM conv: enough z-slices to raise the (often tiny,
/// single-batch `N=1`) `M×N` base grid to ~one full wave of resident CTAs on the device, but no more —
/// over-splitting adds reduce overhead and shrinks each slice's arithmetic intensity. Returns `1` when
/// the base grid already fills the SMs, or the shape can't be split cleanly (`GK` not a multiple of the
/// 16-wide WMMA K-tile, e.g. a 3-channel first layer). Pure (takes the SM count) so it is unit-testable
/// without a device. Every returned `sk>1` satisfies the [`conv_wmma_splitk_ptx`] contract: `sk | GK`
/// and `GK/sk` a multiple of 16.
pub fn conv_splitk_factor(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    sm_count: usize,
) -> usize {
    let n = (h - r + 1) * (w - s + 1); // valid-conv output P*Q
    splitk_factor_for_n(c * r * s, n, k, sm_count)
}

/// Split-K factor for the **affine** conv (strided + zero-padded): same occupancy heuristic as
/// [`conv_splitk_factor`] but over the downsampled/padded output `P=⌊(H+2·pad-R)/stride⌋+1`,
/// `Q=⌊(W+2·pad-S)/stride⌋+1` — the grid the affine kernel actually launches. The deep-channel,
/// small-spatial downsamples (e.g. C128 28²→14²) are exactly where the base grid starves the SMs and
/// split-K helps; `stride=1, pad=0` reduces to [`conv_splitk_factor`].
#[allow(clippy::too_many_arguments)]
pub fn conv_splitk_factor_affine(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    stride: usize,
    pad: usize,
    sm_count: usize,
) -> usize {
    let p = (h + 2 * pad - r) / stride + 1;
    let q = (w + 2 * pad - s) / stride + 1;
    splitk_factor_for_n(c * r * s, p * q, k, sm_count)
}

/// Core split-K occupancy heuristic shared by the valid and affine variants: given the GEMM reduction
/// length `gk = C·R·S` and output `n = P·Q`, pick the largest clean split that lands the CTA count in a
/// useful occupancy window. Only splits when the base grid is starved AND a clean split reaches it:
/// `lo` is the "worth bothering" floor, `hi` ~one full wave (conv tile ~20 KB SMEM ⇒ ~5 CTAs/SM). A
/// shape whose largest valid split can't even reach `lo` (e.g. a 5×5 with a tiny base) keeps sk=1 —
/// over-splitting a near-full grid only adds the reduce pass and loses (measured).
fn splitk_factor_for_n(gk: usize, n: usize, k: usize, sm_count: usize) -> usize {
    if gk % 16 != 0 {
        return 1; // can't carve whole 16-wide K-tiles (e.g. C3 R3 S3 -> GK=27)
    }
    let base = k.div_ceil(WMMA_BM) * n.div_ceil(WMMA_BN);
    let units16 = gk / 16; // whole 16-wide K-tiles available to slice across z
    let lo = (sm_count * 2).max(40);
    let hi = (sm_count * 5).max(96);
    if base >= lo {
        return 1; // base grid already feeds the SMs
    }
    let mut best = 1;
    for &sk in &[2usize, 3, 4, 6, 8] {
        let ctas = base * sk;
        if units16 % sk == 0 && gk / sk >= 64 && (lo..=hi).contains(&ctas) {
            best = sk; // ascending ⇒ keeps the largest sk that still lands within one wave
        }
    }
    best
}

/// **Register double-buffered** implicit-GEMM conv (same entry names as [`conv_wmma_ptx`] /
/// [`conv_wmma_splitk_ptx`], so the launcher swaps generators with no launch-config change). The
/// single-buffer kernel exposes the full global→shared staging latency: it stages A+B, `bar.sync`, then
/// MMAs, with the tensor cores idle while the loads land — exactly the standalone-GEMM cliff. This
/// **software-pipelines** the K-loop: it prefetches the *next* K-slice's weights + im2col-gather into
/// **registers** (the `ld.global`s are issued *before* the MMAs of the current slice, so their latency
/// flies under the tensor cores), then publishes them to the alternate of **two** SMEM buffers
/// *after* the MMAs — **one `bar.sync` per K-step** instead of two. Unlike a `cp.async` port this needs
/// no 16-byte-contiguous global runs, so it is correct for **any** `R,S` (the conv im2col gather is not
/// 8-f16-contiguous for small filters); the trade-off is a single prefetch depth + a register round-trip
/// (vs `cp.async`'s deeper DMA pipeline), which is the right lever for these **L2-resident** convs where
/// the staging stall is L2- not HBM-latency. `a_per+b_per` extra f16 regs/thread (here 16) hold the
/// in-flight slice. Split-K (`sk`) is orthogonal and threaded through identically to the single-buffer
/// kernel. Requires the per-buffer tile bytes to be powers of two (the buffer toggle is an XOR).
pub fn conv_wmma_db_ptx(c: usize, h: usize, w: usize, k: usize, r: usize, s: usize) -> String {
    conv_wmma_db_ptx_impl(c, h, w, k, r, s, 1, STATIC_SMEM_CAP)
}

/// Split-K variant of [`conv_wmma_db_ptx`] (entry `conv2d_wmma_splitk`) — the double-buffered pipeline
/// applied to each `GK/sk` z-slice. Pairs with the same [`conv_splitk_reduce_ptx`].
pub fn conv_wmma_db_splitk_ptx(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    sk: usize,
) -> String {
    conv_wmma_db_ptx_impl(c, h, w, k, r, s, sk, STATIC_SMEM_CAP)
}

#[allow(clippy::too_many_arguments)]
fn conv_wmma_db_ptx_impl(
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    sk: usize,
    // The SMEM ceiling, as in `conv_wmma_ptx_impl` — but over `conv_wmma_smem_bytes(bm, bn, 2)`, since
    // this variant carries TWO A rings and TWO B rings (the register-double-buffered pipeline).
    smem_budget: usize,
) -> String {
    use std::fmt::Write as _;
    let p = h - r + 1;
    let q = w - s + 1;
    let m = k; // GEMM M
    let n = p * q; // GEMM N
    let gk = c * r * s; // GEMM K (reduction)
    assert!(
        sk >= 1 && gk % sk == 0,
        "split-K factor {sk} must divide GK={gk}"
    );
    let gk_per = gk / sk;
    assert!(
        sk == 1 || gk_per % 16 == 0,
        "split-K slice GK/sk={gk_per} must be a multiple of 16"
    );
    let mn = m * n;
    let entry = if sk > 1 {
        "conv2d_wmma_splitk"
    } else {
        "conv2d_wmma"
    };
    let rs = r * s;
    let hw = h * w;
    let (bm, bn) = (WMMA_BM, WMMA_BN);
    let (warps_m, warps_n) = (WMMA_WM, WMMA_WN);
    let threads = WMMA_THREADS;
    let wm = bm / warps_m;
    let wn = bn / warps_n;
    let tm = wm / 16;
    let tn = wn / 16;
    let wn_shift = warps_n.trailing_zeros();
    let bn_shift = bn.trailing_zeros();
    let a_per = bm * 16 / threads; // A elements staged per thread
    let b_per = 16 * bn / threads; // B elements staged per thread
    let c_per = bm * bn / threads;
    let tile_a = bm * 16 * 2; // one A buffer, f16 bytes
    let tile_b = 16 * bn * 2; // one B buffer
    assert!(
        tile_a.is_power_of_two() && tile_b.is_power_of_two(),
        "buffer toggle is an XOR"
    );
    let smem_a = 2 * tile_a; // double-buffered
    let smem_b = 2 * tile_b;
    let smem_c = bm * bn * 4; // f32 store scratch (single, epilogue-only)
    let smem_total = conv_wmma_smem_bytes(bm, bn, 2);
    debug_assert_eq!(smem_total, smem_a + smem_b + smem_c);
    assert!(
        smem_total <= smem_budget,
        "conv2d_wmma (db): SMEM {smem_total} B ({bm}x{bn} tile, 2 buffers) exceeds the budget \
         {smem_budget} B"
    );
    assert!(
        !smem_mode_for(smem_total).is_dynamic(),
        "conv2d_wmma (db): {smem_total} B needs the dynamic window, which this generator does not emit"
    );

    let veclist = |pre: &str| -> String {
        let regs: Vec<String> = (0..8).map(|i| format!("%{pre}{i}")).collect();
        format!("{{{}}}", regs.join(","))
    };

    // ---- staging emitters (load global -> reg ; store reg -> shared), reused by prologue + K-loop ----
    // A-load: weights W[M,GK]; slot e=tix+li*T -> (m=e/16, gkk=e%16), gc=<ktreg>+gkk. OOB -> 0.
    let a_load = |kt: &str, s: &mut String| {
        for li in 0..a_per {
            let off = li * threads;
            let _ = writeln!(s, "    add.u32 %e,%tix,{off};");
            let _ = writeln!(s, "    shr.u32 %mm,%e,4;");
            let _ = writeln!(s, "    and.b32 %gkk,%e,15;");
            let _ = writeln!(s, "    add.u32 %tmp,%m0,%mm;       // gm");
            let _ = writeln!(s, "    add.u32 %tmp2,{kt},%gkk;    // gc");
            let _ = writeln!(s, "    setp.lt.u32 %pv,%tmp,{m};");
            let _ = writeln!(s, "    setp.lt.u32 %p0,%tmp2,{gk};");
            let _ = writeln!(s, "    and.pred %pv,%pv,%p0;");
            let _ = writeln!(s, "    mad.lo.s32 %widx,%tmp,{gk},%tmp2;");
            let _ = writeln!(s, "    mul.wide.u32 %off,%widx,2;");
            let _ = writeln!(s, "    add.s64 %ptr,%W,%off;");
            let _ = writeln!(s, "    mov.u16 %na{li},0;");
            let _ = writeln!(s, "    @%pv ld.global.u16 %na{li},[%ptr];");
        }
    };
    // A-store: publish %na{li} into smemA at byte offset <buf> + e*2.
    let a_store = |buf: &str, s: &mut String| {
        for li in 0..a_per {
            let off = li * threads;
            let _ = writeln!(s, "    add.u32 %e,%tix,{off};");
            let _ = writeln!(s, "    mov.u32 %saddr,smemA;");
            let _ = writeln!(s, "    add.u32 %saddr,%saddr,{buf};");
            let _ = writeln!(s, "    shl.b32 %tmp,%e,1;");
            let _ = writeln!(s, "    add.u32 %saddr,%saddr,%tmp;");
            let _ = writeln!(s, "    st.shared.u16 [%saddr],%na{li};");
        }
    };
    // B-load: im2col of X; slot e -> gkk=e/BN, gkv=<ktreg>+gkk; (nn,xpart) are hoisted (K-independent).
    let b_load = |kt: &str, out: &mut String| {
        for li in 0..b_per {
            let off = li * threads;
            let _ = writeln!(out, "    add.u32 %e,%tix,{off};");
            let _ = writeln!(out, "    shr.u32 %gkk,%e,{bn_shift};");
            let _ = writeln!(out, "    add.u32 %gkv,{kt},%gkk;");
            let _ = writeln!(out, "    setp.lt.u32 %pv,%gkv,{gk};");
            let _ = writeln!(out, "    setp.lt.u32 %p0,%bnv{li},{n};");
            let _ = writeln!(out, "    and.pred %pv,%pv,%p0;");
            let _ = writeln!(out, "    div.u32 %cc,%gkv,{rs};");
            let _ = writeln!(out, "    rem.u32 %rem,%gkv,{rs};");
            let _ = writeln!(out, "    div.u32 %rr,%rem,{s};");
            let _ = writeln!(out, "    rem.u32 %ss,%rem,{s};");
            let _ = writeln!(out, "    mad.lo.s32 %xidx,%cc,{hw},%bxp{li};");
            let _ = writeln!(out, "    mad.lo.s32 %xidx,%rr,{w},%xidx;");
            let _ = writeln!(out, "    add.u32 %xidx,%xidx,%ss;");
            let _ = writeln!(out, "    mul.wide.u32 %off,%xidx,2;");
            let _ = writeln!(out, "    add.s64 %ptr,%X,%off;");
            let _ = writeln!(out, "    mov.u16 %nb{li},0;");
            let _ = writeln!(out, "    @%pv ld.global.u16 %nb{li},[%ptr];");
        }
    };
    let b_store = |buf: &str, out: &mut String| {
        for li in 0..b_per {
            let off = li * threads;
            let _ = writeln!(out, "    add.u32 %e,%tix,{off};");
            let _ = writeln!(out, "    mov.u32 %saddr,smemB;");
            let _ = writeln!(out, "    add.u32 %saddr,%saddr,{buf};");
            let _ = writeln!(out, "    shl.b32 %tmp,%e,1;");
            let _ = writeln!(out, "    add.u32 %saddr,%saddr,%tmp;");
            let _ = writeln!(out, "    st.shared.u16 [%saddr],%nb{li};");
        }
    };

    let mut b = String::new();
    let _ = writeln!(b, "{HDR_SM80}");
    let _ = writeln!(b, "// fp16 tensor-core implicit-GEMM conv (register double-buffered): C{c} H{h} W{w} K{k} R{r} S{s}");
    let _ = writeln!(b, "// M={m} N={n} GK={gk}; CTA tile {bm}x{bn}, {warps_m}x{warps_n} warps, per-warp {wm}x{wn}; 2 SMEM buffers");
    if sk > 1 {
        let _ = writeln!(
            b,
            "// split-K: gridDim.z={sk} z-slices of GK/{sk}={gk_per}, disjoint M*N partial planes"
        );
    }
    let _ = writeln!(b, ".visible .entry {entry}(");
    let _ = writeln!(b, "    .param .u64 pXin,");
    let _ = writeln!(b, "    .param .u64 pWt,");
    let _ = writeln!(b, "    .param .u64 pOut");
    let _ = writeln!(b, ")");
    let _ = writeln!(b, "{{");
    let _ = writeln!(b, "    .shared .align 16 .b8 smemA[{smem_a}];");
    let _ = writeln!(b, "    .shared .align 16 .b8 smemB[{smem_b}];");
    let _ = writeln!(b, "    .shared .align 16 .b8 smemC[{smem_c}];");
    let _ = writeln!(b, "    .reg .pred %p0,%pv;");
    // prefetch value registers (hold the next K-slice in flight across the MMAs)
    let mut hv = String::from("%hv,");
    for li in 0..a_per {
        hv += &format!("%na{li},");
    }
    for li in 0..b_per {
        hv += &format!("%nb{li},");
    }
    let _ = writeln!(b, "    .reg .b16 {};", hv.trim_end_matches(','));
    let _ = writeln!(
        b,
        "    .reg .b32 %tix,%m0,%n0,%kt,%ktn,%e,%mm,%gkk,%ncol,%gkv,%nn,%cc,%rem,%rr,%ss,%pp,%qq,%xidx,%widx,%tmp,%tmp2,%saddr,%warpId,%wrb,%wcb,%bufA,%bufB,%bufWA,%bufWB;"
    );
    if sk > 1 {
        let _ = writeln!(b, "    .reg .b32 %zsl,%ktend;");
    }
    let mut decl = String::new();
    for ti in 0..tm {
        for tj in 0..tn {
            for rr in 0..8 {
                decl += &format!("%c{ti}_{tj}_{rr},");
            }
        }
    }
    for ti in 0..tm {
        for rr in 0..8 {
            decl += &format!("%a{ti}_{rr},");
        }
    }
    for tj in 0..tn {
        for rr in 0..8 {
            decl += &format!("%b{tj}_{rr},");
        }
    }
    for li in 0..b_per {
        decl += &format!("%bnv{li},%bxp{li},");
    }
    let _ = writeln!(b, "    .reg .f32 %cf;");
    let _ = writeln!(b, "    .reg .b32 {};", decl.trim_end_matches(','));
    let _ = writeln!(b, "    .reg .b64 %X,%W,%O,%off,%gp,%ptr;");
    let _ = writeln!(b);
    let _ = writeln!(b, "    ld.param.u64 %X,[pXin];");
    let _ = writeln!(b, "    ld.param.u64 %W,[pWt];");
    let _ = writeln!(b, "    ld.param.u64 %O,[pOut];");
    let _ = writeln!(b, "    cvta.to.global.u64 %X,%X;");
    let _ = writeln!(b, "    cvta.to.global.u64 %W,%W;");
    let _ = writeln!(b, "    cvta.to.global.u64 %O,%O;");
    let _ = writeln!(b, "    mov.u32 %tix,%tid.x;");
    let _ = writeln!(b, "    mov.u32 %tmp,%ctaid.y;");
    let _ = writeln!(b, "    mul.lo.s32 %m0,%tmp,{bm};");
    let _ = writeln!(b, "    mov.u32 %tmp,%ctaid.x;");
    let _ = writeln!(b, "    mul.lo.s32 %n0,%tmp,{bn};");
    let _ = writeln!(b, "    shr.u32 %warpId,%tix,5;");
    let _ = writeln!(b, "    shr.u32 %tmp,%warpId,{wn_shift};");
    let _ = writeln!(b, "    mul.lo.s32 %wrb,%tmp,{wm};");
    let _ = writeln!(b, "    and.b32 %tmp,%warpId,{};", warps_n - 1);
    let _ = writeln!(b, "    mul.lo.s32 %wcb,%tmp,{wn};");
    for ti in 0..tm {
        for tj in 0..tn {
            for rr in 0..8 {
                let _ = writeln!(b, "    mov.f32 %c{ti}_{tj}_{rr},0f00000000;");
            }
        }
    }
    // Hoist K-independent im2col decode: per B-staging slot, nn=n0+ncol and xpart=p*W+q (used by every kt).
    let _ = writeln!(
        b,
        "    // ---- hoist K-independent im2col decode (n -> p,q -> xpart) ----"
    );
    for li in 0..b_per {
        let off = li * threads;
        let _ = writeln!(b, "    add.u32 %e,%tix,{off};");
        let _ = writeln!(b, "    and.b32 %ncol,%e,{};", bn - 1);
        let _ = writeln!(b, "    add.u32 %bnv{li},%n0,%ncol;");
        let _ = writeln!(b, "    div.u32 %pp,%bnv{li},{q};");
        let _ = writeln!(b, "    rem.u32 %qq,%bnv{li},{q};");
        let _ = writeln!(b, "    mad.lo.s32 %bxp{li},%pp,{w},%qq;");
    }
    // K-slice bounds (sk>1 carves a z-slice; the store adds z*M*N for the disjoint partial plane).
    if sk > 1 {
        let _ = writeln!(b, "    mov.u32 %zsl,%ctaid.z;");
        let _ = writeln!(b, "    mul.lo.s32 %kt,%zsl,{gk_per};");
        let _ = writeln!(b, "    add.u32 %ktend,%kt,{gk_per};");
    } else {
        let _ = writeln!(b, "    mov.u32 %kt,0;");
    }
    // ---- prologue: stage the first slice into buffer 0 ----
    let _ = writeln!(b, "    mov.u32 %bufA,0;");
    let _ = writeln!(b, "    mov.u32 %bufB,0;");
    let _ = writeln!(
        b,
        "    // ---- prologue: load+publish slice gk0 into buffer 0 ----"
    );
    a_load("%kt", &mut b);
    b_load("%kt", &mut b);
    a_store("%bufA", &mut b);
    b_store("%bufB", &mut b);
    let _ = writeln!(b, "    bar.sync 0;");
    let _ = writeln!(b);
    let _ = writeln!(b, "KLOOP:");
    if sk > 1 {
        let _ = writeln!(b, "    setp.ge.u32 %p0,%kt,%ktend;");
    } else {
        let _ = writeln!(b, "    setp.ge.u32 %p0,%kt,{gk};");
    }
    let _ = writeln!(b, "    @%p0 bra KEND;");
    let _ = writeln!(b, "    add.u32 %ktn,%kt,16;          // next K-slice base");
    let _ = writeln!(b);
    // Prefetch the next slice into registers FIRST (program order before the MMAs) so the global loads
    // are in flight while the tensor cores consume the current buffer. OOB lanes load 0 (gc/gkv>=GK).
    let _ = writeln!(
        b,
        "    // ---- prefetch next slice (gk0+16) into registers ----"
    );
    a_load("%ktn", &mut b);
    b_load("%ktn", &mut b);
    let _ = writeln!(b);
    // Load A/B fragments from the current READ buffer (%bufA/%bufB).
    let _ = writeln!(
        b,
        "    // ---- consume current buffer: load fragments + MMA ----"
    );
    let _ = writeln!(b, "    mov.u32 %tmp,16;");
    for ti in 0..tm {
        let _ = writeln!(b, "    add.u32 %tmp2,%wrb,{};", ti * 16);
        let _ = writeln!(b, "    mul.lo.s32 %tmp2,%tmp2,32;");
        let _ = writeln!(b, "    mov.u32 %saddr,smemA;");
        let _ = writeln!(b, "    add.u32 %saddr,%saddr,%bufA;");
        let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,%saddr;");
        let _ = writeln!(b, "    cvt.u64.u32 %gp,%tmp2;");
        let _ = writeln!(b, "    cvta.shared.u64 %gp,%gp;");
        let ra = veclist(&format!("a{ti}_"));
        let _ = writeln!(
            b,
            "    wmma.load.a.sync.aligned.m16n16k16.row.f16 {ra}, [%gp], %tmp;"
        );
    }
    let _ = writeln!(b, "    mov.u32 %tmp,{bn};");
    for tj in 0..tn {
        let _ = writeln!(b, "    add.u32 %tmp2,%wcb,{};", tj * 16);
        let _ = writeln!(b, "    shl.b32 %tmp2,%tmp2,1;");
        let _ = writeln!(b, "    mov.u32 %saddr,smemB;");
        let _ = writeln!(b, "    add.u32 %saddr,%saddr,%bufB;");
        let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,%saddr;");
        let _ = writeln!(b, "    cvt.u64.u32 %gp,%tmp2;");
        let _ = writeln!(b, "    cvta.shared.u64 %gp,%gp;");
        let rb = veclist(&format!("b{tj}_"));
        let _ = writeln!(
            b,
            "    wmma.load.b.sync.aligned.m16n16k16.row.f16 {rb}, [%gp], %tmp;"
        );
    }
    for ti in 0..tm {
        let ra = veclist(&format!("a{ti}_"));
        for tj in 0..tn {
            let rb = veclist(&format!("b{tj}_"));
            let cc = veclist(&format!("c{ti}_{tj}_"));
            let _ = writeln!(
                b,
                "    wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {cc}, {ra}, {rb}, {cc};"
            );
        }
    }
    let _ = writeln!(b);
    // Publish the prefetched slice into the WRITE buffer (the alternate of the two), then one barrier.
    let _ = writeln!(
        b,
        "    // ---- publish prefetched slice into the alternate buffer ----"
    );
    let _ = writeln!(b, "    xor.b32 %bufWA,%bufA,{tile_a};");
    let _ = writeln!(b, "    xor.b32 %bufWB,%bufB,{tile_b};");
    a_store("%bufWA", &mut b);
    b_store("%bufWB", &mut b);
    let _ = writeln!(b, "    bar.sync 0;");
    let _ = writeln!(b, "    mov.u32 %bufA,%bufWA;");
    let _ = writeln!(b, "    mov.u32 %bufB,%bufWB;");
    let _ = writeln!(b, "    add.u32 %kt,%kt,16;");
    let _ = writeln!(b, "    bra KLOOP;");
    let _ = writeln!(b, "KEND:");
    // Epilogue: drain warp tiles to smemC, then cooperative guarded global store (identical to single-buf).
    let _ = writeln!(b, "    mov.u32 %tmp,{bn};");
    for ti in 0..tm {
        for tj in 0..tn {
            let _ = writeln!(b, "    add.u32 %tmp2,%wrb,{};", ti * 16);
            let _ = writeln!(b, "    mul.lo.s32 %tmp2,%tmp2,{bn};");
            let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,%wcb;");
            let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,{};", tj * 16);
            let _ = writeln!(b, "    shl.b32 %tmp2,%tmp2,2;");
            let _ = writeln!(b, "    mov.u32 %saddr,smemC;");
            let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,%saddr;");
            let _ = writeln!(b, "    cvt.u64.u32 %gp,%tmp2;");
            let _ = writeln!(b, "    cvta.shared.u64 %gp,%gp;");
            let cc = veclist(&format!("c{ti}_{tj}_"));
            let _ = writeln!(
                b,
                "    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%gp], {cc}, %tmp;"
            );
        }
    }
    let _ = writeln!(b, "    bar.sync 0;");
    for li in 0..c_per {
        let off = li * threads;
        let _ = writeln!(b, "    add.u32 %e,%tix,{off};");
        let _ = writeln!(b, "    shr.u32 %mm,%e,{bn_shift};");
        let _ = writeln!(b, "    and.b32 %ncol,%e,{};", bn - 1);
        let _ = writeln!(b, "    add.u32 %tmp,%m0,%mm;");
        let _ = writeln!(b, "    add.u32 %nn,%n0,%ncol;");
        let _ = writeln!(b, "    setp.lt.u32 %pv,%tmp,{m};");
        let _ = writeln!(b, "    setp.lt.u32 %p0,%nn,{n};");
        let _ = writeln!(b, "    and.pred %pv,%pv,%p0;");
        let _ = writeln!(b, "    mov.u32 %saddr,smemC;");
        let _ = writeln!(b, "    shl.b32 %tmp2,%e,2;");
        let _ = writeln!(b, "    add.u32 %saddr,%saddr,%tmp2;");
        let _ = writeln!(b, "    ld.shared.f32 %cf,[%saddr];");
        let _ = writeln!(b, "    mad.lo.s32 %xidx,%tmp,{n},%nn;");
        if sk > 1 {
            let _ = writeln!(b, "    mad.lo.s32 %xidx,%zsl,{mn},%xidx;");
        }
        let _ = writeln!(b, "    mul.wide.u32 %off,%xidx,4;");
        let _ = writeln!(b, "    add.s64 %ptr,%O,%off;");
        let _ = writeln!(b, "    @%pv st.global.f32 [%ptr],%cf;");
    }
    let _ = writeln!(b, "    ret;");
    let _ = writeln!(b, "}}");
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert a generated module is pure ASCII, naming the offending line if not. A single non-ASCII
    /// byte anywhere in a PTX string is a `ptxas fatal`, and on the device path it surfaces only as an
    /// opaque `DriverError` out of `cuModuleLoadData` with no hint that one byte is at fault (§3A P1).
    pub(super) fn assert_ptx_ascii(what: &str, ptx: &str) {
        if let Some((i, line)) = ptx.lines().enumerate().find(|(_, l)| !l.is_ascii()) {
            panic!(
                "{what}: PTX line {} is not ASCII (ptxas fatal): {line:?}",
                i + 1
            );
        }
        assert!(!ptx.is_empty(), "{what}: generated an empty module");
    }

    /// **§3A P1 gate, device-free.** Every generator in this file interleaves `writeln!` PTX text with
    /// surrounding Rust doc comments full of non-ASCII (`×`, `→`, `≥`, `α`, `ᵀ`), so copying an
    /// adjacent comment into an emitted line is a one-keystroke way to break every conv on the device.
    /// Runs without a CUDA device, so it holds in every configuration. Each generator is exercised on
    /// every shape in the sweep — no `continue`, so no variant can silently go unchecked.
    #[test]
    fn every_conv_generator_emits_ascii_ptx() {
        for (c, h, w, k, r, s) in [
            (3usize, 32usize, 32usize, 16usize, 3usize, 3usize),
            (64, 56, 56, 64, 1, 1),
            (8, 16, 16, 48, 5, 5),
            (128, 28, 28, 128, 3, 3),
        ] {
            let pq = (h - r + 1) * (w - s + 1);
            assert_ptx_ascii("conv2d_ptx", &conv2d_ptx(c, h, w, k, r, s));
            assert_ptx_ascii("conv_wmma_ptx", &conv_wmma_ptx(c, h, w, k, r, s));
            assert_ptx_ascii(
                "conv_wmma_strided_ptx",
                &conv_wmma_strided_ptx(c, h, w, k, r, s, 2),
            );
            assert_ptx_ascii(
                "conv_wmma_pad_ptx",
                &conv_wmma_pad_ptx(c, h, w, k, r, s, 2, 1),
            );
            assert_ptx_ascii("conv_wmma_db_ptx", &conv_wmma_db_ptx(c, h, w, k, r, s));
            for act in [
                crate::ptx_wmma::Act::None,
                crate::ptx_wmma::Act::Relu,
                crate::ptx_wmma::Act::Silu,
                crate::ptx_wmma::Act::Gelu,
            ] {
                assert_ptx_ascii(
                    "conv_wmma_epi_ptx",
                    &conv_wmma_epi_ptx(c, h, w, k, r, s, act, true),
                );
            }
            assert_ptx_ascii("bias_relu_ptx", &bias_relu_ptx(k, pq));
            assert_ptx_ascii("pad_nchw_copy_ptx", &pad_nchw_copy_ptx(c, h, w, 1));
        }
        // Split-K variants need `sk | GK` with `GK/sk` a multiple of 16, so they carry their own
        // shape list rather than being `continue`d out of the sweep above.
        for (c, h, w, k, r, s, sk) in [
            (256usize, 14usize, 14usize, 256usize, 3usize, 3usize, 4usize),
            (128, 28, 28, 128, 1, 1, 2),
        ] {
            let gk = c * r * s;
            assert_eq!(gk % sk, 0);
            assert_eq!((gk / sk) % 16, 0);
            assert_ptx_ascii(
                "conv_wmma_splitk_ptx",
                &conv_wmma_splitk_ptx(c, h, w, k, r, s, sk),
            );
            assert_ptx_ascii(
                "conv_wmma_pad_splitk_ptx",
                &conv_wmma_pad_splitk_ptx(c, h, w, k, r, s, 1, 1, sk),
            );
            assert_ptx_ascii(
                "conv_wmma_db_splitk_ptx",
                &conv_wmma_db_splitk_ptx(c, h, w, k, r, s, sk),
            );
            assert_ptx_ascii(
                "conv_splitk_reduce_ptx",
                &conv_splitk_reduce_ptx(k * 196, sk),
            );
        }
        // The budget seam's second emission form: the module-scope `.extern .shared` window. It is a
        // different text path (an extra declaration line, a different base symbol), so it needs its own
        // pass through the ASCII gate — the whole point of the gate is that no variant goes unchecked.
        let window_rows: [(usize, usize, usize, usize, usize, usize, usize); 3] = [
            (2, 64, 64, 8, 40, 40, 101_376),
            (4, 96, 96, 8, 48, 48, 166_912),
            (1, 128, 128, 4, 60, 60, 232_448),
        ];
        for (c, h, w, k, r, s, budget) in window_rows {
            let (ptx, mode) = conv2d_ptx_budget(c, h, w, k, r, s, budget);
            assert!(mode.is_dynamic(), "this row must exercise the window");
            assert_ptx_ascii("conv2d_ptx_budget (dynamic window)", &ptx);
        }
        // ...and the budget-carrying entry points on the static side.
        assert_ptx_ascii(
            "conv2d_ptx_budget (static)",
            &conv2d_ptx_budget(64, 28, 28, 64, 3, 3, STATIC_SMEM_CAP).0,
        );
        assert_ptx_ascii(
            "conv_wmma_ptx_budget",
            &conv_wmma_ptx_budget(64, 28, 28, 64, 3, 3, STATIC_SMEM_CAP).0,
        );
    }

    /// **Header-floor gate, device-free.** Every conv module — the `CONV2D` literal included — must
    /// open with exactly [`crate::ptx_target::HDR_SM80`]. PTX is forward-compatible *only*: a module
    /// tagged with the development device's `sm_89` buys nothing on Ada and fails
    /// `cuModuleLoadData` on every A100. Nothing in this family needs Ada (f32 arithmetic, shared
    /// memory, `cp.async`, `ldmatrix`, `wmma`/`mma.sync.m16n8k16.f16` are all Ampere-legal), so the
    /// floor *is* `sm_80`. `CONV2D` is a `const` consumed as a `&'static str` from `gpu.rs`, so it
    /// cannot interpolate the constant — this assert is what keeps the copy honest.
    #[test]
    fn conv_headers_are_at_the_sm80_floor() {
        use crate::ptx_target::{HDR_SM80, TARGET_SM89};
        let (c, h, w, k, r, s) = (64usize, 28usize, 28usize, 64usize, 3usize, 3usize);
        let mods: Vec<(&str, String)> = vec![
            ("CONV2D", CONV2D.to_string()),
            ("conv2d_ptx", conv2d_ptx(c, h, w, k, r, s)),
            ("conv_wmma_ptx", conv_wmma_ptx(c, h, w, k, r, s)),
            (
                "conv_wmma_strided_ptx",
                conv_wmma_strided_ptx(c, h, w, k, r, s, 2),
            ),
            (
                "conv_wmma_pad_ptx",
                conv_wmma_pad_ptx(c, h, w, k, r, s, 2, 1),
            ),
            ("conv_wmma_db_ptx", conv_wmma_db_ptx(c, h, w, k, r, s)),
            (
                "conv_wmma_epi_ptx",
                conv_wmma_epi_ptx(c, h, w, k, r, s, crate::ptx_wmma::Act::Relu, true),
            ),
            (
                "conv_wmma_splitk_ptx",
                conv_wmma_splitk_ptx(c, h, w, k, r, s, 2),
            ),
            (
                "conv_wmma_db_splitk_ptx",
                conv_wmma_db_splitk_ptx(c, h, w, k, r, s, 2),
            ),
            (
                "conv_wmma_pad_splitk_ptx",
                conv_wmma_pad_splitk_ptx(c, h, w, k, r, s, 1, 1, 2),
            ),
            ("conv_splitk_reduce_ptx", conv_splitk_reduce_ptx(k * 676, 2)),
            ("bias_relu_ptx", bias_relu_ptx(k, 676)),
            ("pad_nchw_copy_ptx", pad_nchw_copy_ptx(c, h, w, 1)),
        ];
        for (what, ptx) in &mods {
            assert!(
                ptx.starts_with(HDR_SM80),
                "{what}: must open with ptx_target::HDR_SM80, got {:?}",
                &ptx[..ptx.len().min(64)]
            );
            assert!(
                !ptx.contains(TARGET_SM89),
                "{what}: an Ampere-legal module must not claim the Ada floor"
            );
        }
        assert_eq!(
            mods.len(),
            13,
            "every conv generator must be covered by the header gate"
        );
    }

    /// FNV-1a 64 over the raw bytes — a dependency-free, deterministic digest of a PTX module.
    /// Used only by [`shipped_conv_ptx_is_byte_identical`]; it never reaches the device.
    pub(super) fn ptx_digest(s: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Every conv module this crate can dispatch today, at the shapes the dispatch tables and benches
    /// actually use. `(label, ptx)`.
    pub(super) fn shipped_conv_modules() -> Vec<(String, String)> {
        use crate::ptx_wmma::Act;
        let mut out: Vec<(String, String)> = vec![("CONV2D".into(), CONV2D.to_string())];
        // The bench/dispatch corpus (`gpu.rs` conv_vs_peers): a ResNet-ish 3x3 stack, a 1x1, a 5x5.
        let corpus = [
            (3usize, 64usize, 64usize, 64usize, 3usize, 3usize),
            (64, 56, 56, 64, 3, 3),
            (128, 28, 28, 128, 3, 3),
            (256, 14, 14, 256, 3, 3),
            (32, 32, 32, 32, 5, 5),
            (64, 56, 56, 64, 1, 1),
            (8, 16, 16, 48, 5, 5),
        ];
        for (c, h, w, k, r, s) in corpus {
            let tag = format!("C{c}_{h}x{w}_K{k}_{r}x{s}");
            out.push((format!("conv2d_ptx/{tag}"), conv2d_ptx(c, h, w, k, r, s)));
            out.push((
                format!("conv_wmma_ptx/{tag}"),
                conv_wmma_ptx(c, h, w, k, r, s),
            ));
            out.push((
                format!("conv_wmma_db_ptx/{tag}"),
                conv_wmma_db_ptx(c, h, w, k, r, s),
            ));
            out.push((
                format!("conv_wmma_strided_ptx.s2/{tag}"),
                conv_wmma_strided_ptx(c, h, w, k, r, s, 2),
            ));
            out.push((
                format!("conv_wmma_pad_ptx.s1p1/{tag}"),
                conv_wmma_pad_ptx(c, h, w, k, r, s, 1, 1),
            ));
            out.push((
                format!("conv_wmma_epi_ptx.relu_bias/{tag}"),
                conv_wmma_epi_ptx(c, h, w, k, r, s, Act::Relu, true),
            ));
            out.push((
                format!("conv_wmma_epi_ptx.silu_nobias/{tag}"),
                conv_wmma_epi_ptx(c, h, w, k, r, s, Act::Silu, false),
            ));
            out.push((
                format!("bias_relu_ptx/{tag}"),
                bias_relu_ptx(k, (h - r + 1) * (w - s + 1)),
            ));
            out.push((
                format!("pad_nchw_copy_ptx.p1/{tag}"),
                pad_nchw_copy_ptx(c, h, w, 1),
            ));
        }
        // Split-K needs sk | GK and GK/sk a multiple of 16, so it carries its own shape list.
        for (c, h, w, k, r, s, sk) in [
            (256usize, 14usize, 14usize, 256usize, 3usize, 3usize, 4usize),
            (128, 28, 28, 128, 1, 1, 2),
        ] {
            let tag = format!("C{c}_{h}x{w}_K{k}_{r}x{s}_sk{sk}");
            out.push((
                format!("conv_wmma_splitk_ptx/{tag}"),
                conv_wmma_splitk_ptx(c, h, w, k, r, s, sk),
            ));
            out.push((
                format!("conv_wmma_db_splitk_ptx/{tag}"),
                conv_wmma_db_splitk_ptx(c, h, w, k, r, s, sk),
            ));
            out.push((
                format!("conv_wmma_pad_splitk_ptx.s1p1/{tag}"),
                conv_wmma_pad_splitk_ptx(c, h, w, k, r, s, 1, 1, sk),
            ));
            out.push((
                format!("conv_splitk_reduce_ptx/{tag}"),
                conv_splitk_reduce_ptx(k * (h - r + 1) * (w - s + 1), sk),
            ));
        }
        out
    }

    /// `(label, byte length, FNV-1a 64)` of every module in [`shipped_conv_modules`], **recorded from
    /// the binary built at this commit's parent** — i.e. before the budget seam existed. It is the
    /// "before" half of [`shipped_conv_ptx_is_byte_identical`].
    #[rustfmt::skip]
    const SHIPPED_CONV_PTX: &[(&str, usize, u64)] = &[
        ("CONV2D", 2681, 0x2478944dab44b71b),
        ("conv2d_ptx/C3_64x64_K64_3x3", 15478, 0xc55713bbeb38db50),
        ("conv_wmma_ptx/C3_64x64_K64_3x3", 37774, 0x992109419c26dd6b),
        ("conv_wmma_db_ptx/C3_64x64_K64_3x3", 42524, 0x8b9e4b569acbcf2e),
        ("conv_wmma_strided_ptx.s2/C3_64x64_K64_3x3", 38093, 0xb4185595828a2946),
        ("conv_wmma_pad_ptx.s1p1/C3_64x64_K64_3x3", 40366, 0x8795e80d2a84cc97),
        ("conv_wmma_epi_ptx.relu_bias/C3_64x64_K64_3x3", 43614, 0xecfb044e86e6a5b5),
        ("conv_wmma_epi_ptx.silu_nobias/C3_64x64_K64_3x3", 42926, 0xde31156424c0589f),
        ("bias_relu_ptx/C3_64x64_K64_3x3", 848, 0x08ea69e48422d2e8),
        ("pad_nchw_copy_ptx.p1/C3_64x64_K64_3x3", 1251, 0x2b6f7c61101429ba),
        ("conv2d_ptx/C64_56x56_K64_3x3", 15489, 0x9d13067b423d3b41),
        ("conv_wmma_ptx/C64_56x56_K64_3x3", 37801, 0x9f7d0ed3ae661be3),
        ("conv_wmma_db_ptx/C64_56x56_K64_3x3", 42575, 0xed10b28d61da8ff2),
        ("conv_wmma_strided_ptx.s2/C64_56x56_K64_3x3", 38120, 0x9ce481ea5e1d625b),
        ("conv_wmma_pad_ptx.s1p1/C64_56x56_K64_3x3", 40393, 0xbf271347c42407f8),
        ("conv_wmma_epi_ptx.relu_bias/C64_56x56_K64_3x3", 43641, 0x8dbb2a91bb381b7d),
        ("conv_wmma_epi_ptx.silu_nobias/C64_56x56_K64_3x3", 42953, 0x2a928ae207aa5aef),
        ("bias_relu_ptx/C64_56x56_K64_3x3", 848, 0xcdd53372c4b734ed),
        ("pad_nchw_copy_ptx.p1/C64_56x56_K64_3x3", 1253, 0x72cceabe1c414046),
        ("conv2d_ptx/C128_28x28_K128_3x3", 15486, 0x602e1a975e40a25e),
        ("conv_wmma_ptx/C128_28x28_K128_3x3", 37789, 0xe2524f5d6fdf25b1),
        ("conv_wmma_db_ptx/C128_28x28_K128_3x3", 42579, 0xbebc7d534b8d2878),
        ("conv_wmma_strided_ptx.s2/C128_28x28_K128_3x3", 38173, 0xfeff07e3e9c6f5bc),
        ("conv_wmma_pad_ptx.s1p1/C128_28x28_K128_3x3", 40381, 0x9d4698d4ce94a4d3),
        ("conv_wmma_epi_ptx.relu_bias/C128_28x28_K128_3x3", 43629, 0xd40956a4eb5c71e3),
        ("conv_wmma_epi_ptx.silu_nobias/C128_28x28_K128_3x3", 42941, 0x80544b88cc3ba3f5),
        ("bias_relu_ptx/C128_28x28_K128_3x3", 846, 0xfdb8b5b15615b67c),
        ("pad_nchw_copy_ptx.p1/C128_28x28_K128_3x3", 1251, 0xbaa3e6db29409ad0),
        ("conv2d_ptx/C256_14x14_K256_3x3", 15485, 0x2b70dae3b35632e0),
        ("conv_wmma_ptx/C256_14x14_K256_3x3", 37789, 0x45959b6b5991f3a9),
        ("conv_wmma_db_ptx/C256_14x14_K256_3x3", 42579, 0x4116acfffea063d0),
        ("conv_wmma_strided_ptx.s2/C256_14x14_K256_3x3", 38084, 0x19a37d642b8e34d9),
        ("conv_wmma_pad_ptx.s1p1/C256_14x14_K256_3x3", 40381, 0xd7dcf8950d49568a),
        ("conv_wmma_epi_ptx.relu_bias/C256_14x14_K256_3x3", 43629, 0x51803d1871fe3507),
        ("conv_wmma_epi_ptx.silu_nobias/C256_14x14_K256_3x3", 42941, 0x0f43d9297846ade5),
        ("bias_relu_ptx/C256_14x14_K256_3x3", 846, 0x9bdf232b419385b2),
        ("pad_nchw_copy_ptx.p1/C256_14x14_K256_3x3", 1250, 0xde981de277df1376),
        ("conv2d_ptx/C32_32x32_K32_5x5", 25088, 0x85d2d4d10d7d3eae),
        ("conv_wmma_ptx/C32_32x32_K32_5x5", 37744, 0xc24d7c9e2f092193),
        ("conv_wmma_db_ptx/C32_32x32_K32_5x5", 42526, 0xb33d64987dd50b2e),
        ("conv_wmma_strided_ptx.s2/C32_32x32_K32_5x5", 38128, 0xf3213daec84745c6),
        ("conv_wmma_pad_ptx.s1p1/C32_32x32_K32_5x5", 40336, 0x9eae397d9df3cadd),
        ("conv_wmma_epi_ptx.relu_bias/C32_32x32_K32_5x5", 43584, 0x071f5c3fb5f1dc35),
        ("conv_wmma_epi_ptx.silu_nobias/C32_32x32_K32_5x5", 42896, 0xde0540107bff7d9f),
        ("bias_relu_ptx/C32_32x32_K32_5x5", 846, 0x5c16c428cc2b53be),
        ("pad_nchw_copy_ptx.p1/C32_32x32_K32_5x5", 1252, 0xb99f776ed5978107),
        ("conv2d_ptx/C64_56x56_K64_1x1", 8522, 0xa36b699f3d3b93c6),
        ("conv_wmma_ptx/C64_56x56_K64_1x1", 37775, 0x32a1b841b84c1020),
        ("conv_wmma_db_ptx/C64_56x56_K64_1x1", 42525, 0xec92827e48410e9b),
        ("conv_wmma_strided_ptx.s2/C64_56x56_K64_1x1", 38094, 0x3c6c81e8b32aa96a),
        ("conv_wmma_pad_ptx.s1p1/C64_56x56_K64_1x1", 40367, 0x34ca804c7a99c6fd),
        ("conv_wmma_epi_ptx.relu_bias/C64_56x56_K64_1x1", 43615, 0x85031476cfdc7136),
        ("conv_wmma_epi_ptx.silu_nobias/C64_56x56_K64_1x1", 42927, 0xb34821c9bfe541dc),
        ("bias_relu_ptx/C64_56x56_K64_1x1", 848, 0xa286ec87eb8bbb92),
        ("pad_nchw_copy_ptx.p1/C64_56x56_K64_1x1", 1253, 0x72cceabe1c414046),
        ("conv2d_ptx/C8_16x16_K48_5x5", 25077, 0x2a050096777c59c1),
        ("conv_wmma_ptx/C8_16x16_K48_5x5", 37735, 0x5fbe2b7b54483400),
        ("conv_wmma_db_ptx/C8_16x16_K48_5x5", 42509, 0xaf0a679a363db66f),
        ("conv_wmma_strided_ptx.s2/C8_16x16_K48_5x5", 38030, 0x2aace79cb0677f76),
        ("conv_wmma_pad_ptx.s1p1/C8_16x16_K48_5x5", 40327, 0x534bd2bb404263cf),
        ("conv_wmma_epi_ptx.relu_bias/C8_16x16_K48_5x5", 43575, 0x09cd2bd59ab9d06a),
        ("conv_wmma_epi_ptx.silu_nobias/C8_16x16_K48_5x5", 42887, 0xce8fa0fd505bff74),
        ("bias_relu_ptx/C8_16x16_K48_5x5", 845, 0x555ffd2f111d7e21),
        ("pad_nchw_copy_ptx.p1/C8_16x16_K48_5x5", 1247, 0xe8ccec08f7600aaa),
        ("conv_wmma_splitk_ptx/C256_14x14_K256_3x3_sk4", 40196, 0x38006233d0e2e3b7),
        ("conv_wmma_db_splitk_ptx/C256_14x14_K256_3x3_sk4", 44002, 0x574eff00b28b9577),
        ("conv_wmma_pad_splitk_ptx.s1p1/C256_14x14_K256_3x3_sk4", 42788, 0x5aead7fc27957832),
        ("conv_splitk_reduce_ptx/C256_14x14_K256_3x3_sk4", 1277, 0x202d6b83efd2def7),
        ("conv_wmma_splitk_ptx/C128_28x28_K128_1x1_sk2", 40200, 0x150fd26f887ca829),
        ("conv_wmma_db_splitk_ptx/C128_28x28_K128_1x1_sk2", 43982, 0xe878b3e17fa66807),
        ("conv_wmma_pad_splitk_ptx.s1p1/C128_28x28_K128_1x1_sk2", 42792, 0x2fe7e7088ba1f6ff),
        ("conv_splitk_reduce_ptx/C128_28x28_K128_1x1_sk2", 1021, 0xb77596a632faa18c),
    ];

    /// **The before/after gate for the budget seam, device-free.** Threading `smem_budget` through the
    /// conv generators must leave every *shipped* module byte-identical: the emission rule is that at or
    /// below [`STATIC_SMEM_CAP`] the historical static `.shared` spelling is emitted verbatim, and every
    /// caller in `gpu.rs` passes exactly that. Byte-identity is not cosmetic — it keeps the persistent
    /// cubin cache (keyed on the PTX hash) warm and makes "no measured conv number moved" a claim about
    /// the machine rather than about the change.
    ///
    /// The pins in [`SHIPPED_CONV_PTX`] were recorded from the binary built at this commit's **parent**,
    /// so this really is an across-the-change comparison and not a self-consistency tautology. A
    /// deliberate future change to a generator updates them from the printed table below.
    #[test]
    fn shipped_conv_ptx_is_byte_identical() {
        let got = shipped_conv_modules();
        let mut bad = Vec::new();
        assert_eq!(
            got.len(),
            SHIPPED_CONV_PTX.len(),
            "the shipped-module list and its pinned digests must stay in step"
        );
        for ((label, ptx), (plabel, plen, pdig)) in got.iter().zip(SHIPPED_CONV_PTX) {
            assert_eq!(label, plabel, "pin order must match the generated order");
            let (len, dig) = (ptx.len(), ptx_digest(ptx));
            if len != *plen || dig != *pdig {
                bad.push(format!(
                    "  {label}: pinned ({plen} B, 0x{pdig:016x}) != got ({len} B, 0x{dig:016x})"
                ));
            }
        }
        if !bad.is_empty() {
            for (label, ptx) in &got {
                println!("(\"{label}\", {}, 0x{:016x}),", ptx.len(), ptx_digest(ptx));
            }
            panic!(
                "{} shipped conv module(s) changed byte-for-byte:\n{}\nIf intended, replace \
                 SHIPPED_CONV_PTX with the table printed above.",
                bad.len(),
                bad.join("\n")
            );
        }
    }

    /// Largest square filter `R=S=f` the tiled generator can stage inside `budget` bytes at channel
    /// block `kb`. Monotone in `f` (the footprint is strictly increasing), so a linear scan is exact.
    fn max_square_filter(kb: usize, budget: usize) -> usize {
        (1usize..4096)
            .take_while(|&f| tiled_smem_bytes(kb, f, f) <= budget)
            .last()
            .unwrap_or(0)
    }

    // The three device budgets: `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`, 99 / 163 / 227 KiB (D6 §1.3 —
    // the 4050 row is the value probed on this box, the other two are the vendor tuning guides).
    const OPTIN_ADA_4050: usize = 101_376;
    const OPTIN_A100: usize = 166_912;
    const OPTIN_H100: usize = 232_448;
    // `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`, 100 / 164 / 228 KiB — the residency *denominator*, and a
    // different number from the per-block ceiling above. Both are needed: per-block legality uses the
    // opt-in maximum, residency arithmetic uses the per-SM carveout.
    const SMEM_SM_ADA_4050: usize = 102_400;
    const SMEM_SM_A100: usize = 167_936;
    const SMEM_SM_H100: usize = 233_472;
    // CUDA reserves 1 KiB of shared memory per block, and it is real in the residency arithmetic
    // (measured, D6 §3.1 probe 6).
    const SMEM_RESERVED_PER_BLOCK: usize = 1024;

    /// **The census the budget lift is really about — device-free arithmetic, no sweep.**
    ///
    /// Reading it: the retired gate was a hardcoded 44 KiB whose comment claimed the larger banks could
    /// not be opted into. Both halves are now false, but the honest finding is that **the gate was
    /// declining nothing anyone runs.** With the tile pinned at 16×16 and `KB ≤ 8`, the tiled conv's
    /// footprint is bounded by the *filter*, so what the budget buys is bigger `R=S` — and 33 → 34 (the
    /// ISA cap) → 51 / 66 / 78 (the three device windows) are all far outside the DL envelope, where
    /// 11×11 is an AlexNet conv1 and 7×7 a ResNet stem. What binds this family is the 1024-thread block
    /// cap (`TILE_P·TILE_Q` threads) and `KB` accumulators per thread, not shared memory.
    ///
    /// The one conv configuration in this file where a bigger budget is the *only* road is widening the
    /// implicit-GEMM CTA tile, whose `BM·BN·4` epilogue scratch dominates: 64×64 = 20 KiB, 128×64 = 38
    /// KiB (still static-legal — so the cheapest widening needs no window at all, which is itself the
    /// evidence that the budget was not the blocker), 128×128 = 72 KiB (over the ISA cap on every
    /// target, and at 1 CTA/SM on the 4050 vs 2 on A100 and 3 on H100).
    #[test]
    fn conv_smem_census_across_target_budgets() {
        // (1) The direct tiled conv: what each budget admits, in square-filter extent.
        const RETIRED_GATE: usize = 44 * 1024;
        let census = [
            ("retired 44 KiB gate", RETIRED_GATE, 33usize, 67usize),
            ("PTX ISA static cap", STATIC_SMEM_CAP, 34, 70),
            ("RTX 4050 opt-in", OPTIN_ADA_4050, 51, 104),
            ("A100 opt-in", OPTIN_A100, 66, 136),
            ("H100 opt-in", OPTIN_H100, 78, 162),
        ];
        for (what, budget, kb8, kb1) in census {
            assert_eq!(max_square_filter(8, budget), kb8, "{what}: KB=8 R=S extent");
            assert_eq!(max_square_filter(1, budget), kb1, "{what}: KB=1 R=S extent");
            println!("{what:22} {budget:7} B -> R=S<={kb8:3} (KB=8), <={kb1:3} (KB=1)");
        }

        // (2) ...and none of it reaches a real conv: every shape the dispatch corpus and the benches
        // use already fits the *static* cap, so the device window frees zero shipped shapes. This is
        // the negative result, asserted rather than asserted-away.
        for (c, h, w, k, r, s) in [
            (3usize, 64usize, 64usize, 64usize, 3usize, 3usize),
            (64, 56, 56, 64, 3, 3),
            (128, 28, 28, 128, 3, 3),
            (256, 14, 14, 256, 3, 3),
            (32, 32, 32, 32, 5, 5),
            (64, 56, 56, 64, 1, 1),
            (8, 16, 16, 48, 5, 5),
            (3, 224, 224, 64, 7, 7),   // ResNet stem
            (3, 227, 227, 96, 11, 11), // AlexNet conv1 (the widest filter in common use)
        ] {
            assert!(
                tiled_applies_budget(c, h, w, k, r, s, STATIC_SMEM_CAP),
                "C{c} {h}x{w} K{k} {r}x{s} should already fit the static cap"
            );
            assert!(
                tiled_smem_bytes(kblock(k), r, s) <= RETIRED_GATE,
                "C{c} {h}x{w} K{k} {r}x{s} fit even the retired 44 KiB gate"
            );
        }

        // (3) The implicit-GEMM family: the tile widths a budget admits, and the residency they imply.
        assert_eq!(conv_wmma_smem_bytes(WMMA_BM, WMMA_BN, 1), 20480);
        assert_eq!(conv_wmma_smem_bytes(WMMA_BM, WMMA_BN, 2), 24576); // register-double-buffered
        assert_eq!(conv_wmma_smem_bytes(128, 64, 1), 38912);
        assert_eq!(conv_wmma_smem_bytes(128, 128, 1), 73728);
        assert!(
            conv_wmma_smem_bytes(128, 64, 1) <= STATIC_SMEM_CAP,
            "128x64 is static-legal today: the cheapest widening needs no window"
        );
        assert!(
            conv_wmma_smem_bytes(128, 128, 1) > STATIC_SMEM_CAP,
            "128x128 is the first conv tile that REQUIRES the dynamic window"
        );
        for budget in [OPTIN_ADA_4050, OPTIN_A100, OPTIN_H100] {
            assert!(conv_wmma_smem_bytes(128, 128, 1) <= budget);
        }
        // CTAs/SM from the SMEM term alone (D6 §3.1): floor(SMEM_sm / (bytes + reserved)).
        let ctas = |smem_sm: usize, bytes: usize| smem_sm / (bytes + SMEM_RESERVED_PER_BLOCK);
        assert_eq!(ctas(SMEM_SM_ADA_4050, 73728), 1, "4050 starves at 128x128");
        assert_eq!(ctas(SMEM_SM_A100, 73728), 2);
        assert_eq!(ctas(SMEM_SM_H100, 73728), 3);
        // The shipped 64x64 tile is nowhere near SMEM-bound on any target.
        assert_eq!(ctas(SMEM_SM_ADA_4050, 20480), 4);
        assert_eq!(ctas(SMEM_SM_A100, 20480), 7);
        assert_eq!(ctas(SMEM_SM_H100, 20480), 10);
    }

    /// **The window arm, device-free.** Above [`STATIC_SMEM_CAP`] the tile must move into the ONE
    /// module-scope `.extern .shared` window: declared before the entry (the identical line inside an
    /// entry body is `CUDA_ERROR_INVALID_PTX`), no static `.shared` array left behind, every address
    /// still formed through the symbol (the window base is not guaranteed to be 0), and the returned
    /// [`SmemMode`] carrying the exact byte count the launch must pass.
    #[test]
    fn conv2d_ptx_crosses_into_the_dynamic_window_above_the_isa_cap() {
        // A 40x40 filter at KB=8: 15100 elems -> 60400 B, past the ISA cap, inside the 4050's window.
        let (c, h, w, k, r, s) = (2usize, 64usize, 64usize, 8usize, 40usize, 40usize);
        let bytes = tiled_smem_bytes(kblock(k), r, s);
        assert!(
            bytes > STATIC_SMEM_CAP && bytes <= OPTIN_ADA_4050,
            "{bytes}"
        );
        assert!(
            !tiled_applies(c, h, w, k, r, s),
            "declined by the static gate"
        );
        assert!(
            tiled_applies_budget(c, h, w, k, r, s, OPTIN_ADA_4050),
            "admitted once the device window is the budget"
        );

        let (ptx, mode) = conv2d_ptx_budget(c, h, w, k, r, s, OPTIN_ADA_4050);
        assert_eq!(mode, SmemMode::Dynamic(bytes as u32));
        assert_eq!(mode.launch_bytes(), bytes);
        // The window arm is a second text path, so it needs the header floor and the ASCII gate too —
        // an `sm_89` tag or one non-ASCII byte here would be just as fatal as in the static arm.
        assert!(
            ptx.starts_with(HDR_SM80),
            "the floor still leads the module"
        );
        assert!(!ptx.contains(crate::ptx_target::TARGET_SM89));
        assert_ptx_ascii("conv2d_ptx_budget (dynamic)", &ptx);
        // The declaration, verbatim and at module scope: the identical line inside an entry body is
        // CUDA_ERROR_INVALID_PTX, and two module-scope externs would ALIAS (both measured, D6 §1.1).
        assert!(ptx.contains(DSMEM_DECL), "the exact window declaration");
        assert_eq!(
            ptx.matches(".extern .shared").count(),
            1,
            "exactly ONE window"
        );
        let decl = ptx.find(".extern .shared").expect("window declaration");
        let entry = ptx.find(".visible .entry").expect("entry");
        assert!(decl < entry, "the window must be declared at MODULE scope");
        assert!(
            !ptx.contains(".shared .align 4 .b8 smem["),
            "no static array beside the window"
        );
        assert!(
            ptx.contains(&format!("mov.u32 %r10,{DSMEM_SYM};")),
            "the tile base must come from the window symbol"
        );

        // ...and the static arm must never touch the window.
        let (small, small_mode) = conv2d_ptx_budget(64, 28, 28, 64, 3, 3, OPTIN_H100);
        assert_eq!(small_mode, SmemMode::Static);
        assert_eq!(
            small_mode.launch_bytes(),
            0,
            "a static tile launches with 0"
        );
        assert!(
            !small.contains(DSMEM_SYM),
            "static kernels keep their array"
        );
        assert_eq!(
            small,
            conv2d_ptx(64, 28, 28, 64, 3, 3),
            "same text either way"
        );
    }

    /// An over-budget shape must fail **at generation**, naming the family, the tile and both byte
    /// counts. The alternative is what the tree had: no assert at all in the generator, one hardcoded
    /// number in a separate gate function the caller must remember, and — when they forget — a
    /// `ptxas error: Entry function 'conv2d' uses too much shared data` surfacing as an opaque
    /// `DriverError` out of `cuModuleLoadData` that names neither conv nor the shape.
    #[test]
    #[should_panic(expected = "exceeds the budget")]
    fn conv2d_ptx_over_budget_panics_at_generation() {
        // 100x100 at KB=8: (115*115 + 8*10000)*4 = 372900 B — past even the H100 window.
        let _ = conv2d_ptx_budget(1, 128, 128, 8, 100, 100, OPTIN_H100);
    }

    /// The same tripwire on the implicit-GEMM family. Its footprint is a compile-time constant today,
    /// so the only way to reach it is an absurd budget — which is exactly the point: the assert lives
    /// where a widened CTA tile would first be visible.
    #[test]
    #[should_panic(expected = "exceeds the budget")]
    fn conv_wmma_ptx_over_budget_panics_at_generation() {
        let _ = conv_wmma_ptx_budget(64, 28, 28, 64, 3, 3, 4096);
    }

    #[test]
    fn splitk_factor_picks_reasonable() {
        // ~20-SM device (RTX 4050). Heavily-starved deep-channel shapes split; full grids don't.
        let sm = 20;
        // C256 14x14 K256 (base = 4*3 = 12) is badly starved -> a large split.
        assert!(conv_splitk_factor(256, 14, 14, 256, 3, 3, sm) >= 4);
        // C128 28x28 K128 (base = 2*11 = 22) -> a moderate split.
        assert!(conv_splitk_factor(128, 28, 28, 128, 3, 3, sm) >= 2);
        // C64 56x56 K64 (base = 1*46 = 46 ≥ 2·SM) already feeds the SMs -> no split.
        assert_eq!(conv_splitk_factor(64, 56, 56, 64, 3, 3, sm), 1);
        // C32 32x32 5x5 (base = 13) — its only clean split (sk2 -> 26 CTAs) can't reach 2 waves, and
        // over-splitting a near-half-wave grid measured *slower* (reduce overhead) -> keep sk=1.
        assert_eq!(conv_splitk_factor(32, 32, 32, 32, 5, 5, sm), 1);
        // First layer C3 R3 S3 -> GK=27, not a multiple of 16 -> cannot split.
        assert_eq!(conv_splitk_factor(3, 64, 64, 64, 3, 3, sm), 1);
        // Every split it returns must satisfy the kernel contract (sk|GK, GK/sk % 16 == 0).
        for (c, h, w, k, r, s) in [
            (256usize, 14, 14, 256, 3, 3),
            (128, 28, 28, 128, 3, 3),
            (256, 14, 14, 256, 1, 1),
        ] {
            let sk = conv_splitk_factor(c, h, w, k, r, s, sm);
            let gk = c * r * s;
            assert_eq!(gk % sk, 0);
            if sk > 1 {
                assert_eq!(
                    (gk / sk) % 16,
                    0,
                    "GK/sk must be a multiple of the WMMA K-tile"
                );
            }
        }
    }
}
