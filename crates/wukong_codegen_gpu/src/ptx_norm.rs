//! Fused row-wise normalizations on the GPU — softmax / LayerNorm / RMSNorm — the GPU analogue of
//! `wukong_norm_f32`. One **warp per row**: the 32 lanes stride over the row, accumulate per-lane,
//! then do a warp all-reduce via `shfl.sync.bfly` (no shared memory). The reduction order is fixed
//! (butterfly tree), so results are deterministic run-to-run; tolerance-gated vs the CPU oracle
//! (exp/sqrt use SFU approximations). `eps` rides in as an f32 param.

use std::sync::OnceLock;

/// Emit a warp butterfly all-reduce of `%{reg}` under `op` ("add" or "max"); every lane ends with the
/// full-warp result. Uses a scratch f32 reg `%rt`.
fn allreduce(reg: &str, op: &str) -> String {
    let mut s = String::new();
    for off in [16, 8, 4, 2, 1] {
        s += &format!("    shfl.sync.bfly.b32 %rt, %{reg}, {off}, 0x1f, 0xffffffff;\n");
        s += &format!("    {op}.f32 %{reg}, %{reg}, %rt;\n");
    }
    s
}

/// A strided pass `for (i = lane; i < cols; i += 32)` with `body` (which may use `%i` and must leave
/// the per-element address in `%addr = xptr + i*4`). `tag` makes labels unique.
fn strided(tag: &str, body: &str) -> String {
    format!(
        "    mov.u32 %i,%lane;\nL_{tag}:\n    setp.ge.u32 %p0,%i,%cols;\n    @%p0 bra E_{tag};\n    mul.wide.u32 %off,%i,4;\n    add.s64 %addr,%xptr,%off;\n{body}    add.u32 %i,%i,32;\n    bra L_{tag};\nE_{tag}:\n"
    )
}

fn header() -> String {
    String::from(".version 7.8\n.target sm_89\n.address_size 64\n")
}

/// Common prologue: one warp per row (block_dim=32, grid=rows). Sets `%row,%lane,%cols`, the f32
/// `%colsf`, and base pointers `%xptr`/`%optr`. Bails if `row >= rows`.
fn prologue(name: &str) -> String {
    format!(
        r#".visible .entry {name}(
    .param .u32 pRows,
    .param .u32 pCols,
    .param .f32 pEps,
    .param .u64 pX,
    .param .u64 pOut
)
{{
    .reg .pred %p0;
    .reg .f32 %rt,%v,%e,%m,%s,%s2,%mean,%var,%denom,%eps,%colsf,%inv;
    .reg .b32 %rows,%cols,%row,%lane,%i,%tmp;
    .reg .b64 %X,%Out,%xptr,%optr,%addr,%off;
    ld.param.u32 %rows,[pRows];
    ld.param.u32 %cols,[pCols];
    ld.param.f32 %eps,[pEps];
    ld.param.u64 %X,[pX];
    ld.param.u64 %Out,[pOut];
    cvta.to.global.u64 %X,%X;
    cvta.to.global.u64 %Out,%Out;
    mov.u32 %row,%ctaid.x;
    setp.ge.u32 %p0,%row,%rows;
    @%p0 bra RET_{name};
    mov.u32 %lane,%tid.x;
    cvt.rn.f32.u32 %colsf,%cols;
    mul.lo.s32 %tmp,%row,%cols;
    mul.wide.u32 %off,%tmp,4;
    add.s64 %xptr,%X,%off;
    add.s64 %optr,%Out,%off;
"#
    )
}

/// softmax(row) = exp(x - max) / sum(exp(x - max)), numerically stable.
fn softmax() -> String {
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let mut s = prologue("softmax");
    // pass 1: row max
    s += "    mov.f32 %m,0fFF800000;\n";
    s += &strided(
        "smax",
        "    ld.global.f32 %v,[%addr];\n    max.f32 %m,%m,%v;\n",
    );
    s += &allreduce("m", "max");
    // pass 2: sum of exp(x - m)
    s += "    mov.f32 %s,0f00000000;\n";
    s += &strided(
        "ssum",
        &format!("    ld.global.f32 %v,[%addr];\n    sub.f32 %v,%v,%m;\n    mul.f32 %v,%v,{log2e};\n    ex2.approx.f32 %e,%v;\n    add.f32 %s,%s,%e;\n"),
    );
    s += &allreduce("s", "add");
    // pass 3: out = exp(x - m) / s
    s += &strided(
        "swr",
        &format!("    ld.global.f32 %v,[%addr];\n    sub.f32 %v,%v,%m;\n    mul.f32 %v,%v,{log2e};\n    ex2.approx.f32 %e,%v;\n    div.rn.f32 %e,%e,%s;\n    mul.wide.u32 %off,%i,4;\n    add.s64 %addr,%optr,%off;\n    st.global.f32 [%addr],%e;\n"),
    );
    s += "RET_softmax:\n    ret;\n}\n";
    s
}

/// layernorm(row) = (x - mean) / sqrt(var + eps), mean/var over the row.
fn layernorm() -> String {
    let mut s = prologue("layernorm");
    // pass 1: s1 = sum(x), s2 = sum(x^2)
    s += "    mov.f32 %s,0f00000000;\n    mov.f32 %s2,0f00000000;\n";
    s += &strided(
        "lsum",
        "    ld.global.f32 %v,[%addr];\n    add.f32 %s,%s,%v;\n    fma.rn.f32 %s2,%v,%v,%s2;\n",
    );
    s += &allreduce("s", "add");
    s += &allreduce("s2", "add");
    // mean = s/cols ; var = s2/cols - mean^2 ; denom = sqrt(var+eps)
    s += "    div.rn.f32 %mean,%s,%colsf;\n";
    s += "    div.rn.f32 %var,%s2,%colsf;\n";
    s += "    mul.f32 %v,%mean,%mean;\n    sub.f32 %var,%var,%v;\n";
    s += "    add.f32 %denom,%var,%eps;\n    sqrt.rn.f32 %denom,%denom;\n";
    // pass 2: out = (x - mean) / denom
    s += &strided(
        "lwr",
        "    ld.global.f32 %v,[%addr];\n    sub.f32 %v,%v,%mean;\n    div.rn.f32 %v,%v,%denom;\n    mul.wide.u32 %off,%i,4;\n    add.s64 %addr,%optr,%off;\n    st.global.f32 [%addr],%v;\n",
    );
    s += "RET_layernorm:\n    ret;\n}\n";
    s
}

/// rmsnorm(row) = x / sqrt(mean(x^2) + eps).
fn rmsnorm() -> String {
    let mut s = prologue("rmsnorm");
    s += "    mov.f32 %s2,0f00000000;\n";
    s += &strided(
        "rsum",
        "    ld.global.f32 %v,[%addr];\n    fma.rn.f32 %s2,%v,%v,%s2;\n",
    );
    s += &allreduce("s2", "add");
    s += "    div.rn.f32 %v,%s2,%colsf;\n    add.f32 %denom,%v,%eps;\n    sqrt.rn.f32 %denom,%denom;\n";
    s += &strided(
        "rwr",
        "    ld.global.f32 %v,[%addr];\n    div.rn.f32 %v,%v,%denom;\n    mul.wide.u32 %off,%i,4;\n    add.s64 %addr,%optr,%off;\n    st.global.f32 [%addr],%v;\n",
    );
    s += "RET_rmsnorm:\n    ret;\n}\n";
    s
}

/// The norm module (softmax / layernorm / rmsnorm), generated once and cached.
pub fn norm_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = header();
        m += &softmax();
        m += &layernorm();
        m += &rmsnorm();
        m
    })
    .as_str()
}
