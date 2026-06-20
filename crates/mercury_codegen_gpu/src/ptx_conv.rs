//! Direct 2D **convolution** on the GPU (the op category attention/GEMM don't cover). Single batch:
//! input `X[C,H,W]`, weights `W[K,C,R,S]`, output `O[K,P,Q]` with `P=H-R+1`, `Q=W-S+1` (stride 1,
//! no padding) — the valid cross-correlation deep-learning frameworks call conv2d. **One thread per
//! output element** `(k,p,q)`, looping `c,r,s` with `fma.rn` into an f32 accumulator. Naive but
//! correct and fully fused (no im2col buffer in HBM); tolerance-gated vs an f64 CPU reference.
//!
//! All extents are runtime params, so the three accumulation loops can't be unrolled at gen time —
//! this is a single hand-written PTX entry (`conv2d`) rather than a generator.

/// `conv2d(C,H,W,K,R,S,P,Q, Xin,Wt,Out)` — see module docs. Output index `idx = (k·P+p)·Q+q` is the
/// linear thread id, so the store address is just `Out + idx`.
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
