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

/// Whether the SMEM-tiled generator applies to this shape. The staged halo + weights must fit a
/// conservative shared-memory budget, and `H>=R`, `W>=S` (a valid conv). Falls back to [`CONV2D`]
/// otherwise so every shape stays runnable.
pub fn tiled_applies(c: usize, h: usize, w: usize, _k: usize, r: usize, s: usize) -> bool {
    if h < r || w < s {
        return false;
    }
    let halo = (TILE_P + r - 1) * (TILE_Q + s - 1);
    let smem_bytes = (halo + r * s) * 4;
    // Stay well under the 48 KB default static-SMEM ceiling (no opt-in to the larger Ada banks).
    c >= 1 && smem_bytes <= 44 * 1024
}

/// Emit the SMEM-tiled conv kernel specialized to `[C,H,W] (*) [K,C,R,S] -> [K,P,Q]`. Entry name is
/// `conv2d`. Launch with block `(TILE_Q, TILE_P, 1)` and grid `(ceil(Q/TILE_Q), ceil(P/TILE_P), K)`.
pub fn conv2d_ptx(c: usize, h: usize, w: usize, k: usize, r: usize, s: usize) -> String {
    use std::fmt::Write as _;
    let (tp, tq) = (TILE_P, TILE_Q);
    let p = h - r + 1;
    let q = w - s + 1;
    let halo_h = tp + r - 1;
    let halo_w = tq + s - 1;
    let halo = halo_h * halo_w; // input elements staged per channel
    let nthreads = tp * tq;
    let rs = r * s;
    let smem_elems = halo + rs;
    let smem_bytes = smem_elems * 4;
    let row_iters = halo_h.div_ceil(tp); // halo rows each thread strides over
    let col_iters = halo_w.div_ceil(tq);
    let w_iters = rs.div_ceil(nthreads); // weight loads per thread

    let mut b = String::new();
    let _ = writeln!(b, ".version 7.8");
    let _ = writeln!(b, ".target sm_89");
    let _ = writeln!(b, ".address_size 64");
    let _ = writeln!(b);
    let _ = writeln!(b, "// SMEM-tiled conv2d specialized to C={c} H={h} W={w} K={k} R={r} S={s}");
    let _ = writeln!(b, "// tile {tp}x{tq}, halo {halo_h}x{halo_w}, smem {smem_bytes} B");
    let _ = writeln!(b, ".visible .entry conv2d(");
    let _ = writeln!(b, "    .param .u64 pXin,");
    let _ = writeln!(b, "    .param .u64 pWt,");
    let _ = writeln!(b, "    .param .u64 pOut");
    let _ = writeln!(b, ")");
    let _ = writeln!(b, "{{");
    let _ = writeln!(b, "    .reg .pred %p<8>;");
    let _ = writeln!(b, "    .reg .b32  %r<48>;");
    let _ = writeln!(b, "    .reg .f32  %f<8>;");
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
    let _ = writeln!(b, "    mov.u32 %r5,%ctaid.z;        // k (output channel)");
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
    // weight base for this k: k*C*R*S
    let _ = writeln!(b, "    mul.lo.s32 %r14,%r5,{};      // woff_k = k*C*R*S", c * rs);
    let _ = writeln!(b, "    mov.f32 %f1,0f00000000;      // acc");
    let _ = writeln!(b, "    mov.u32 %r15,0;              // c = 0");
    let _ = writeln!(b, "CLOOP:");
    let _ = writeln!(b, "    setp.ge.u32 %p0,%r15,{c};");
    let _ = writeln!(b, "    @%p0 bra CEND;");
    let _ = writeln!(b, "    mul.lo.s32 %r16,%r15,{};     // rXc = c*H*W", h * w);
    let _ = writeln!(b, "    mad.lo.s32 %r17,%r15,{rs},%r14;    // wbase_c = woff_k + c*R*S");
    let _ = writeln!(b);
    let _ = writeln!(b, "    // ---- cooperative halo load (X channel c) ----");
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
            let _ = writeln!(b, "    mov.f32 %f2,0f00000000;");
            let _ = writeln!(b, "    @%p4 ld.global.f32 %f2,[%rd4];");
            let _ = writeln!(b, "    mad.lo.s32 %r24,%r18,{halo_w},%r21;  // smem elem = hy*halo_w+hx");
            let _ = writeln!(b, "    shl.b32 %r24,%r24,2;");
            let _ = writeln!(b, "    add.s32 %r24,%r10,%r24;");
            let _ = writeln!(b, "    @%p3 st.shared.f32 [%r24],%f2;");
        }
    }
    let _ = writeln!(b);
    let _ = writeln!(b, "    // ---- cooperative weight load (k,c window) ----");
    for j in 0..w_iters {
        let base = j * nthreads;
        let _ = writeln!(b, "    add.s32 %r25,%r13,{base};         // weight slot = tlin + {base}");
        let _ = writeln!(b, "    setp.lt.u32 %p1,%r25,{rs};");
        let _ = writeln!(b, "    add.s32 %r26,%r25,%r17;          // global weight elem");
        let _ = writeln!(b, "    mul.wide.s32 %rd5,%r26,4;");
        let _ = writeln!(b, "    add.s64 %rd5,%rd2,%rd5;");
        let _ = writeln!(b, "    mov.f32 %f3,0f00000000;");
        let _ = writeln!(b, "    @%p1 ld.global.f32 %f3,[%rd5];");
        let _ = writeln!(b, "    add.s32 %r27,%r25,{halo};        // smem weight elem");
        let _ = writeln!(b, "    shl.b32 %r27,%r27,2;");
        let _ = writeln!(b, "    add.s32 %r27,%r10,%r27;");
        let _ = writeln!(b, "    @%p1 st.shared.f32 [%r27],%f3;");
    }
    let _ = writeln!(b);
    let _ = writeln!(b, "    bar.sync 0;");
    let _ = writeln!(b, "    // ---- unrolled R*S reduction from SMEM ----");
    for rr in 0..r {
        for ss in 0..s {
            let x_off = (rr * halo_w + ss) * 4; // bytes from compute base %r12
            let w_off = (halo + rr * s + ss) * 4; // bytes from smem base %r10
            let _ = writeln!(b, "    ld.shared.f32 %f2,[%r12+{x_off}];");
            let _ = writeln!(b, "    ld.shared.f32 %f3,[%r10+{w_off}];");
            let _ = writeln!(b, "    fma.rn.f32 %f1,%f2,%f3,%f1;");
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
    // oidx = (k*P + op)*Q + oq
    let _ = writeln!(b, "    mad.lo.s32 %r28,%r5,{p},%r9;       // k*P + op");
    let _ = writeln!(b, "    mad.lo.s32 %r28,%r28,{q},%r8;      // *Q + oq");
    let _ = writeln!(b, "    mul.wide.s32 %rd6,%r28,4;");
    let _ = writeln!(b, "    add.s64 %rd6,%rd3,%rd6;");
    let _ = writeln!(b, "    st.global.f32 [%rd6],%f1;");
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
