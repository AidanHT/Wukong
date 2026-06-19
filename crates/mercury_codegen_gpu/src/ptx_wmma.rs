//! Tensor-core GEMM via `wmma` PTX — the FLOP/s headline. On Ada (`sm_89`) the 4th-gen tensor cores
//! do bf16/fp16 multiplies with **f32 accumulate** (the standard mixed-precision contract), at many
//! times the f32 CUDA-core rate. This is where low precision stops being a footprint trick and buys
//! real throughput.
//!
//! Layout: one warp (32 threads) computes one 16×16 C tile via the `m16n16k16` WMMA fragments
//! (opaque, distributed across the warp). We accumulate over K in 16-wide steps. `A·Bᵀ` (nn.Linear):
//! A is M×K row-major, B is N×K row-major; the `Bᵀ` tile is exactly the `.col` layout of B with
//! leading dim K, so A loads `.row` and B loads `.col`, both with stride K — no host transpose.
//!
//! Requires M, N, K to be multiples of 16 (WMMA loads full 16×16×16 tiles); callers pad otherwise.
//! Inputs are f16/bf16; the accumulator and C are f32. f16·f16→f32 and bf16·bf16→f32 are exact per
//! product, so the only deviation from an f64 reference is the f32 accumulation order (tolerance-gated).

/// One WMMA GEMM entry for element type `ty` ("f16" or "bf16"), name `name`, computing `C = A·Bᵀ`.
fn entry(name: &str, ty: &str) -> String {
    // f16 multiplicands use the short mma form (`.f32.f32`); bf16/tf32 must name the input types
    // explicitly (`.f32.bf16.bf16.f32`) since f16 is the legacy default.
    let mma_ty = if ty == "f16" {
        "f32.f32".to_string()
    } else {
        format!("f32.{ty}.{ty}.f32")
    };
    // a/b fragment register count for m16n16k16: f16 uses the legacy 8×.b32 layout (replicated);
    // bf16 uses 4×.b32. The f32 accumulator is always 8×.f32.
    let nab = if ty == "f16" { 8 } else { 4 };
    let veclist = |prefix: &str, n: usize| -> String {
        let regs: Vec<String> = (0..n).map(|i| format!("%{prefix}{i}")).collect();
        format!("{{{}}}", regs.join(","))
    };
    let ra = veclist("ra", nab);
    let rb = veclist("rb", nab);
    let cc = veclist("c", 8);
    format!(
        r#"
.visible .entry {name}(
    .param .u32 pM,
    .param .u32 pN,
    .param .u32 pK,
    .param .u64 pA,
    .param .u64 pB,
    .param .u64 pC
)
{{
    .reg .pred %p0;
    .reg .b32 %ra<{nab}>, %rb<{nab}>;
    .reg .f32 %c<8>;
    .reg .b32 %M,%N,%K,%tileRow,%tileCol,%kt,%tmp;
    .reg .b64 %A,%B,%C,%aptr,%bptr,%cptr,%off;

    ld.param.u32 %M,[pM];
    ld.param.u32 %N,[pN];
    ld.param.u32 %K,[pK];
    ld.param.u64 %A,[pA];
    ld.param.u64 %B,[pB];
    ld.param.u64 %C,[pC];
    cvta.to.global.u64 %A,%A;
    cvta.to.global.u64 %B,%B;
    cvta.to.global.u64 %C,%C;

    mov.u32 %tmp,%ctaid.y;
    mul.lo.s32 %tileRow,%tmp,16;
    mov.u32 %tmp,%ctaid.x;
    mul.lo.s32 %tileCol,%tmp,16;

    mov.f32 %c0,0f00000000;
    mov.f32 %c1,0f00000000;
    mov.f32 %c2,0f00000000;
    mov.f32 %c3,0f00000000;
    mov.f32 %c4,0f00000000;
    mov.f32 %c5,0f00000000;
    mov.f32 %c6,0f00000000;
    mov.f32 %c7,0f00000000;

    // aptr = A + tileRow*K*2  (f16/bf16 = 2 bytes)
    mul.lo.s32 %tmp,%tileRow,%K;
    mul.wide.u32 %off,%tmp,2;
    add.s64 %aptr,%A,%off;
    // bptr = B + tileCol*K*2
    mul.lo.s32 %tmp,%tileCol,%K;
    mul.wide.u32 %off,%tmp,2;
    add.s64 %bptr,%B,%off;

    mov.u32 %kt,0;
KLOOP:
    setp.ge.u32 %p0,%kt,%K;
    @%p0 bra KEND;
    wmma.load.a.sync.aligned.m16n16k16.row.{ty} {ra}, [%aptr], %K;
    wmma.load.b.sync.aligned.m16n16k16.col.{ty} {rb}, [%bptr], %K;
    wmma.mma.sync.aligned.row.col.m16n16k16.{mma_ty} {cc}, {ra}, {rb}, {cc};
    add.s64 %aptr,%aptr,32;
    add.s64 %bptr,%bptr,32;
    add.u32 %kt,%kt,16;
    bra KLOOP;
KEND:
    mad.lo.s32 %tmp,%tileRow,%N,%tileCol;
    mul.wide.u32 %off,%tmp,4;
    add.s64 %cptr,%C,%off;
    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%cptr], {cc}, %N;
    ret;
}}
"#
    )
}

/// The fp16 tensor-core GEMM module (`wmma_nt_f16`).
pub fn wmma_f16_ptx() -> &'static str {
    use std::sync::OnceLock;
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
        m += &entry("wmma_nt_f16", "f16");
        m
    })
    .as_str()
}

/// The bf16 tensor-core GEMM module (`wmma_nt_bf16`).
pub fn wmma_bf16_ptx() -> &'static str {
    use std::sync::OnceLock;
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
        m += &entry("wmma_nt_bf16", "bf16");
        m
    })
    .as_str()
}
