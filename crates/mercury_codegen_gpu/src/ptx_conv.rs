//! 2D **convolution** on the GPU (the op category attention/GEMM don't cover). Single batch:
//! input `X[C,H,W]`, weights `W[K,C,R,S]`, output `O[K,P,Q]` with `P=H-R+1`, `Q=W-S+1` (stride 1,
//! no padding) -- the valid cross-correlation deep-learning frameworks call conv2d.
//!
//! Mercury knows `C,H,W,K,R,S` at *compile* time (shapes live in the type system), so this is a
//! **generator**, not a fixed kernel: [`conv2d_ptx`] emits a kernel **specialized to the exact shape**
//! -- the `r,s` window fully unrolled, every extent a baked-in literal, no dynamic bounds, no tail
//! logic. The lever no shape-agnostic library has.
//!
//! ## Kernels (best-effort fastest first; the launcher in `gpu.rs` picks)
//! * [`CONV2D`] -- the original **naive** one-thread-per-output kernel (looping `c,r,s` from global),
//!   kept as the honest worst-case reference and the fallback for shapes the tiled path rejects.
//! * [`conv2d_ptx`] -- **SMEM-tiled, static-shape-specialized direct conv.** One CTA computes a
//!   `TILE_P x TILE_Q` output tile for one output channel `k` (grid `= (ceil(Q/TQ), ceil(P/TP), K)`).
//!   Per input channel `c` the CTA cooperatively stages the input **halo** `(TP+R-1) x (TQ+S-1)` and
//!   the `R*S` weights into shared memory **once**, then every thread reuses them across the whole
//!   tile -- killing the naive kernel's dominant cost (each input pixel re-read from global `R*S`
//!   times). The `r,s` reduction is fully unrolled into an `fma.rn.f32` chain over `c`.

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

/// Whether the SMEM-tiled generator applies to this shape. The staged halo + `KB` weight windows must
/// fit a conservative shared-memory budget, and `H>=R`, `W>=S` (a valid conv). Falls back to [`CONV2D`]
/// otherwise so every shape stays runnable.
pub fn tiled_applies(c: usize, h: usize, w: usize, k: usize, r: usize, s: usize) -> bool {
    if h < r || w < s || c < 1 || k < 1 {
        return false;
    }
    let halo = (TILE_P + r - 1) * (TILE_Q + s - 1);
    let smem_bytes = (halo + kblock(k) * r * s) * 4;
    // Stay well under the 48 KB default static-SMEM ceiling (no opt-in to the larger Ada banks).
    smem_bytes <= 44 * 1024
}

/// Emit the SMEM-tiled, **channel-register-blocked** conv kernel specialized to
/// `[C,H,W] (*) [K,C,R,S] -> [K,P,Q]`. Entry name is `conv2d`. One CTA computes a `TILE_P×TILE_Q`
/// output tile for `KB=`[`kblock`]`(K)` output channels at once; each thread holds `KB` f32
/// accumulators. Launch with block `(TILE_Q, TILE_P, 1)` and grid `(ceil(Q/TQ), ceil(P/TP), K/KB)`.
pub fn conv2d_ptx(c: usize, h: usize, w: usize, k: usize, r: usize, s: usize) -> String {
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
    let smem_bytes = (halo + kb * rs) * 4;
    let row_iters = halo_h.div_ceil(tp); // halo rows each thread strides over
    let col_iters = halo_w.div_ceil(tq);
    let w_iters = rs.div_ceil(nthreads); // weight loads per (kk, thread)

    let mut b = String::new();
    let _ = writeln!(b, ".version 7.8");
    let _ = writeln!(b, ".target sm_89");
    let _ = writeln!(b, ".address_size 64");
    let _ = writeln!(b);
    let _ = writeln!(b, "// SMEM-tiled conv2d specialized to C={c} H={h} W={w} K={k} R={r} S={s}");
    let _ = writeln!(b, "// tile {tp}x{tq}, kblock {kb}, halo {halo_h}x{halo_w}, smem {smem_bytes} B");
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
    let _ = writeln!(b, "    .shared .align 4 .b8 smem[{smem_bytes}];");
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
    let _ = writeln!(b, "    mov.u32 %r10,smem;           // smem base addr (shared u32)");
    // compute base addr in smem for this thread's window origin (ty*halo_w + tx)*4
    let _ = writeln!(b, "    mad.lo.s32 %r11,%r2,{halo_w},%r1;  // ty*halo_w + tx");
    let _ = writeln!(b, "    shl.b32 %r11,%r11,2;");
    let _ = writeln!(b, "    add.s32 %r12,%r10,%r11;      // smem compute base for (ty,tx)");
    // tlin = ty*tq + tx
    let _ = writeln!(b, "    mad.lo.s32 %r13,%r2,{tq},%r1;      // tlin");
    // k0 = kblock * KB  ;  woff_k0 = k0*C*R*S
    let _ = writeln!(b, "    mul.lo.s32 %r14,%r5,{};      // woff_k0 = (kb_idx*KB)*C*R*S", kb * c_rs);
    // accumulators acc[kk] = %f{8+kk} := 0
    for kk in 0..kb {
        let _ = writeln!(b, "    mov.f32 %f{},0f00000000;", 8 + kk);
    }
    let _ = writeln!(b, "    mov.u32 %r15,0;              // c = 0");
    let _ = writeln!(b, "CLOOP:");
    let _ = writeln!(b, "    setp.ge.u32 %p0,%r15,{c};");
    let _ = writeln!(b, "    @%p0 bra CEND;");
    let _ = writeln!(b, "    mul.lo.s32 %r16,%r15,{};     // rXc = c*H*W", h * w);
    let _ = writeln!(b, "    mad.lo.s32 %r17,%r15,{rs},%r14;    // base_c = woff_k0 + c*R*S");
    let _ = writeln!(b);
    let _ = writeln!(b, "    // ---- cooperative halo load (X channel c, shared by all KB channels) ----");
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
            let _ = writeln!(b, "    and.pred %p3,%p3,%p1;            // storeok = colok && rowok");
            let _ = writeln!(b, "    add.s32 %r22,%r6,%r21;           // gx = q0+hx");
            let _ = writeln!(b, "    setp.lt.u32 %p4,%r22,{w};        // gxok");
            let _ = writeln!(b, "    and.pred %p4,%p4,%p2;            // valok = gxok && gyok");
            let _ = writeln!(b, "    add.s32 %r23,%r20,%r22;          // Xelem = gy*W+rXc+gx");
            let _ = writeln!(b, "    mul.wide.s32 %rd4,%r23,4;");
            let _ = writeln!(b, "    add.s64 %rd4,%rd1,%rd4;");
            let _ = writeln!(b, "    mov.f32 %f1,0f00000000;");
            let _ = writeln!(b, "    @%p4 ld.global.f32 %f1,[%rd4];");
            let _ = writeln!(b, "    mad.lo.s32 %r24,%r18,{halo_w},%r21;  // smem elem = hy*halo_w+hx");
            let _ = writeln!(b, "    shl.b32 %r24,%r24,2;");
            let _ = writeln!(b, "    add.s32 %r24,%r10,%r24;");
            let _ = writeln!(b, "    @%p3 st.shared.f32 [%r24],%f1;");
        }
    }
    let _ = writeln!(b);
    let _ = writeln!(b, "    // ---- cooperative weight load: KB windows of R*S, channel c ----");
    for kk in 0..kb {
        // global base of (k0+kk, c) window = base_c + kk*(C*R*S)
        let _ = writeln!(b, "    add.s32 %r28,%r17,{};        // wbase (k0+{kk},c)", kk * c_rs);
        let smem_w_base = halo + kk * rs; // smem element offset of this window
        for j in 0..w_iters {
            let base = j * nthreads;
            let _ = writeln!(b, "    add.s32 %r25,%r13,{base};        // weight slot");
            let _ = writeln!(b, "    setp.lt.u32 %p1,%r25,{rs};");
            let _ = writeln!(b, "    add.s32 %r26,%r25,%r28;         // global weight elem");
            let _ = writeln!(b, "    mul.wide.s32 %rd5,%r26,4;");
            let _ = writeln!(b, "    add.s64 %rd5,%rd2,%rd5;");
            let _ = writeln!(b, "    mov.f32 %f1,0f00000000;");
            let _ = writeln!(b, "    @%p1 ld.global.f32 %f1,[%rd5];");
            let _ = writeln!(b, "    add.s32 %r27,%r25,{smem_w_base};    // smem weight elem");
            let _ = writeln!(b, "    shl.b32 %r27,%r27,2;");
            let _ = writeln!(b, "    add.s32 %r27,%r10,%r27;");
            let _ = writeln!(b, "    @%p1 st.shared.f32 [%r27],%f1;");
        }
    }
    let _ = writeln!(b);
    let _ = writeln!(b, "    bar.sync 0;");
    let _ = writeln!(b, "    // ---- unrolled R*S reduction; each X reused across KB channels ----");
    for rr in 0..r {
        for ss in 0..s {
            let x_off = (rr * halo_w + ss) * 4; // bytes from compute base %r12
            let _ = writeln!(b, "    ld.shared.f32 %f1,[%r12+{x_off}];   // X[ty+{rr},tx+{ss}]");
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
    let _ = writeln!(b, "    mad.lo.s32 %r30,%r9,{q},%r8;       // spat = op*Q + oq");
    let _ = writeln!(b, "    mul.lo.s32 %r31,%r5,{};           // k0*P*Q", kb * pq);
    let _ = writeln!(b, "    add.s32 %r31,%r31,%r30;           // o0 = k0*P*Q + spat");
    for kk in 0..kb {
        let _ = writeln!(b, "    add.s32 %r32,%r31,{};         // oidx for channel +{kk}", kk * pq);
        let _ = writeln!(b, "    mul.wide.s32 %rd6,%r32,4;");
        let _ = writeln!(b, "    add.s64 %rd6,%rd3,%rd6;");
        let _ = writeln!(b, "    st.global.f32 [%rd6],%f{};", 8 + kk);
    }
    let _ = writeln!(b, "RET:");
    let _ = writeln!(b, "    ret;");
    let _ = writeln!(b, "}}");
    b
}

/// `conv2d(C,H,W,K,R,S,P,Q, Xin,Wt,Out)` -- the original **naive** one-thread-per-output kernel. Output
/// index `idx = (k*P+p)*Q+q` is the linear thread id, so the store address is just `Out + idx`. Kept as
/// the honest worst-case reference and the fallback for shapes the tiled generator rejects.
pub const CONV2D: &str = r#".version 7.8
.target sm_89
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
pub const WMMA_BM: usize = 32;
/// WMMA implicit-GEMM CTA output-tile cols (N = output spatial P*Q direction).
pub const WMMA_BN: usize = 32;

/// Whether the fp16 tensor-core implicit-GEMM conv is worth dispatching for this shape. It is *correct*
/// for any valid conv (guards zero-pad partial tiles), but only pays off once the GEMM has enough
/// reduction depth and output tiles to feed the tensor cores; tiny `GK` makes the per-K-step staging
/// overhead dominate. Heuristic: reduction `C*R*S >= 16` and a non-trivial spatial extent.
pub fn wmma_applies(c: usize, h: usize, w: usize, k: usize, r: usize, s: usize) -> bool {
    if h < r || w < s || k < 1 || c < 1 {
        return false;
    }
    let gk = c * r * s;
    let n = (h - r + 1) * (w - s + 1);
    gk >= 16 && n >= 16 && k >= 16
}

/// Emit the fp16 tensor-core implicit-GEMM conv specialized to `[C,H,W] (*) [K,C,R,S] -> [K,P,Q]`.
/// Entry `conv2d_wmma`. **Inputs are f16** (`X`,`W`); output is f32. One warp per CTA owns a
/// `WMMA_BM x WMMA_BN` output tile as a `2x2` grid of `m16n16k16` tiles. Launch with block `(32,1,1)`
/// and grid `(ceil(N/WMMA_BN), ceil(M/WMMA_BM), 1)` where `M=K`, `N=P*Q`.
pub fn conv_wmma_ptx(c: usize, h: usize, w: usize, k: usize, r: usize, s: usize) -> String {
    use std::fmt::Write as _;
    let p = h - r + 1;
    let q = w - s + 1;
    let m = k; // GEMM M
    let n = p * q; // GEMM N
    let gk = c * r * s; // GEMM K (reduction)
    let rs = r * s;
    let hw = h * w;
    let (bm, bn) = (WMMA_BM, WMMA_BN);
    let tm = bm / 16; // 2
    let tn = bn / 16; // 2
    let a_per = bm * 16 / 32; // A elements staged per thread (BM*BK / 32 threads)
    let b_per = 16 * bn / 32; // B elements staged per thread
    let c_per = bm * bn / 32; // C elements written per thread
    let smem_a = bm * 16 * 2; // f16 bytes
    let smem_b = 16 * bn * 2;
    let smem_c = bm * bn * 4; // f32 store scratch

    let veclist = |pre: &str| -> String {
        let regs: Vec<String> = (0..8).map(|i| format!("%{pre}{i}")).collect();
        format!("{{{}}}", regs.join(","))
    };

    let mut b = String::new();
    let _ = writeln!(b, ".version 7.8");
    let _ = writeln!(b, ".target sm_89");
    let _ = writeln!(b, ".address_size 64");
    let _ = writeln!(b);
    let _ = writeln!(b, "// fp16 tensor-core implicit-GEMM conv: C{c} H{h} W{w} K{k} R{r} S{s}");
    let _ = writeln!(b, "// M={m} N={n} GK={gk}; CTA tile {bm}x{bn} (one warp, 2x2 m16n16k16)");
    let _ = writeln!(b, ".visible .entry conv2d_wmma(");
    let _ = writeln!(b, "    .param .u64 pXin,");
    let _ = writeln!(b, "    .param .u64 pWt,");
    let _ = writeln!(b, "    .param .u64 pOut");
    let _ = writeln!(b, ")");
    let _ = writeln!(b, "{{");
    let _ = writeln!(b, "    .shared .align 16 .b8 smemA[{smem_a}];");
    let _ = writeln!(b, "    .shared .align 16 .b8 smemB[{smem_b}];");
    let _ = writeln!(b, "    .shared .align 16 .b8 smemC[{smem_c}];");
    let _ = writeln!(b, "    .reg .pred %p0,%pv;");
    let _ = writeln!(b, "    .reg .b16 %hv;");
    let _ = writeln!(
        b,
        "    .reg .b32 %tix,%m0,%n0,%kt,%e,%li,%mm,%gkk,%ncol,%gkv,%nn,%cc,%rem,%rr,%ss,%pp,%qq,%ih,%iw,%xidx,%widx,%tmp,%tmp2,%saddr;"
    );
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
    // zero accumulators
    for ti in 0..tm {
        for tj in 0..tn {
            for rr in 0..8 {
                let _ = writeln!(b, "    mov.f32 %c{ti}_{tj}_{rr},0f00000000;");
            }
        }
    }
    let _ = writeln!(b, "    mov.u32 %kt,0;              // gk0");
    let _ = writeln!(b, "KLOOP:");
    let _ = writeln!(b, "    setp.ge.u32 %p0,%kt,{gk};");
    let _ = writeln!(b, "    @%p0 bra KEND;");
    let _ = writeln!(b);
    let _ = writeln!(b, "    // ---- stage A (weights [M,GK]) into smemA[BM][16] ----");
    for li in 0..a_per {
        let off = li * 32;
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
    let _ = writeln!(b, "    // ---- stage B (im2col of X) into smemB[16][BN] ----");
    for li in 0..b_per {
        let off = li * 32;
        let _ = writeln!(b, "    add.u32 %e,%tix,{off};");
        let _ = writeln!(b, "    shr.u32 %gkk,%e,{};          // gkk = e/BN", bn.trailing_zeros());
        let _ = writeln!(b, "    and.b32 %ncol,%e,{};         // ncol = e%BN", bn - 1);
        let _ = writeln!(b, "    add.u32 %gkv,%kt,%gkk;       // gk = gk0+gkk");
        let _ = writeln!(b, "    add.u32 %nn,%n0,%ncol;       // n = n0+ncol");
        let _ = writeln!(b, "    setp.lt.u32 %pv,%gkv,{gk};");
        let _ = writeln!(b, "    setp.lt.u32 %p0,%nn,{n};");
        let _ = writeln!(b, "    and.pred %pv,%pv,%p0;");
        // decode gk -> (c,r,s); n -> (p,q)
        let _ = writeln!(b, "    div.u32 %cc,%gkv,{rs};       // c = gk/(R*S)");
        let _ = writeln!(b, "    rem.u32 %rem,%gkv,{rs};      // rem = gk%(R*S)");
        let _ = writeln!(b, "    div.u32 %rr,%rem,{s};        // r = rem/S");
        let _ = writeln!(b, "    rem.u32 %ss,%rem,{s};        // s = rem%S");
        let _ = writeln!(b, "    div.u32 %pp,%nn,{q};         // p = n/Q");
        let _ = writeln!(b, "    rem.u32 %qq,%nn,{q};         // q = n%Q");
        let _ = writeln!(b, "    add.u32 %ih,%pp,%rr;         // ih = p+r");
        let _ = writeln!(b, "    add.u32 %iw,%qq,%ss;         // iw = q+s");
        let _ = writeln!(b, "    mad.lo.s32 %xidx,%cc,{hw},0;   // c*H*W");
        let _ = writeln!(b, "    mad.lo.s32 %xidx,%ih,{w},%xidx;  // + ih*W");
        let _ = writeln!(b, "    add.u32 %xidx,%xidx,%iw;     // + iw");
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
    // load A fragments (row, ldm=16)
    let _ = writeln!(b, "    mov.u32 %tmp,16;");
    for ti in 0..tm {
        let _ = writeln!(b, "    mov.u32 %tmp2,smemA;");
        let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,{};", ti * 16 * 16 * 2);
        let _ = writeln!(b, "    cvt.u64.u32 %gp,%tmp2;");
        let _ = writeln!(b, "    cvta.shared.u64 %gp,%gp;");
        let ra = veclist(&format!("a{ti}_"));
        let _ = writeln!(b, "    wmma.load.a.sync.aligned.m16n16k16.row.f16 {ra}, [%gp], %tmp;");
    }
    // load B fragments (row, ldm=BN)
    let _ = writeln!(b, "    mov.u32 %tmp,{bn};");
    for tj in 0..tn {
        let _ = writeln!(b, "    mov.u32 %tmp2,smemB;");
        let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,{};", tj * 16 * 2);
        let _ = writeln!(b, "    cvt.u64.u32 %gp,%tmp2;");
        let _ = writeln!(b, "    cvta.shared.u64 %gp,%gp;");
        let rb = veclist(&format!("b{tj}_"));
        let _ = writeln!(b, "    wmma.load.b.sync.aligned.m16n16k16.row.f16 {rb}, [%gp], %tmp;");
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
    // store each tile to smemC (row-major BM x BN, ldm=BN), then guarded copy to global O.
    let _ = writeln!(b, "    mov.u32 %tmp,{bn};");
    for ti in 0..tm {
        for tj in 0..tn {
            let _ = writeln!(b, "    mov.u32 %tmp2,smemC;");
            let elem = (ti * 16) * bn + tj * 16;
            let _ = writeln!(b, "    add.u32 %tmp2,%tmp2,{};", elem * 4);
            let _ = writeln!(b, "    cvt.u64.u32 %gp,%tmp2;");
            let _ = writeln!(b, "    cvta.shared.u64 %gp,%gp;");
            let cc = veclist(&format!("c{ti}_{tj}_"));
            let _ = writeln!(b, "    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%gp], {cc}, %tmp;");
        }
    }
    let _ = writeln!(b, "    bar.sync 0;");
    for li in 0..c_per {
        let off = li * 32;
        let _ = writeln!(b, "    add.u32 %e,%tix,{off};");
        let _ = writeln!(b, "    shr.u32 %mm,%e,{};           // m = e/BN", bn.trailing_zeros());
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
        let _ = writeln!(b, "    mad.lo.s32 %xidx,%tmp,{n},%nn;   // gm*N + gn");
        let _ = writeln!(b, "    mul.wide.u32 %off,%xidx,4;");
        let _ = writeln!(b, "    add.s64 %ptr,%O,%off;");
        let _ = writeln!(b, "    @%pv st.global.f32 [%ptr],%cf;");
    }
    let _ = writeln!(b, "    ret;");
    let _ = writeln!(b, "}}");
    b
}
