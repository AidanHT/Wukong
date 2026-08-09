//! Fused row-wise normalizations on the GPU — softmax / LayerNorm / RMSNorm — the GPU analogue of
//! `wukong_norm_f32`.
//!
//! Two entry families live in this one module (one PTX text, one `"norm"` module key):
//!
//! * **the shipped one-warp entries** `softmax` / `layernorm` / `rmsnorm`. One **warp per row**: the
//!   32 lanes stride over the row, accumulate per-lane, then do a warp all-reduce via
//!   `shfl.sync.bfly` (no shared memory). Launched `grid = (rows,1,1)`, `block = (32,1,1)`; the
//!   kernel reads `%ctaid.x` as its row index and does no grid-stride, so that geometry is the *only*
//!   legal one for them. Their bytes are pinned by `shipped_entries_are_byte_identical`.
//! * **the SM-filling entries** `{softmax,layernorm,rmsnorm}_w{W}` for `W` in [`MW_WIDTHS`], added
//!   2026-08-09. Identical parameter list, identical arithmetic; `W` warps cooperate on one row (a
//!   CTA-stride pass plus, for `W > 1`, a shared-memory cross-warp fold on top of the same
//!   butterfly), and the rows are covered by a **grid-stride** `row = ctaid.x; row += nctaid.x` so
//!   the CTA count can be sized from the device instead of from the row count. [`norm_launch`] picks
//!   `W` and the grid from the probed `sm_count`, and returns the *shipped* entry whenever the plan
//!   comes out at today's geometry.
//!
//! **`W` is baked into the entry, not read from `%ntid.x`** — the entry name *is* the width. The
//! first cut of this family did read the CTA stride out of `%ntid.x`, which made one PTX text serve
//! every launch and looked strictly better. It measured **0.53-0.82x of the shipped kernel at W=1 on
//! an identical launch**: a runtime induction step defeats ptxas's unroller and its immediate-offset
//! address folding, so the strided loop drops to one outstanding load per warp and the whole family
//! starts ~1.9x in the hole. Baking `32*W` as an immediate restores it. The cost is 12 entries
//! instead of 3 and a larger module; the cubin cache makes that a one-time JIT.
//!
//! The reduction order is fixed in both families (butterfly tree within a warp; for `W > 1`, then a
//! serial left-to-right fold over the warp partials in warp-index order), so results are
//! deterministic run-to-run at a fixed `W`. Across different `W` the *order* differs, so the values
//! differ by a few ulps — see [`allreduce_w`] for what that does to the f32 error. Both families are
//! tolerance-gated against the CPU oracle (exp/sqrt use SFU approximations). `eps` rides in as an f32
//! param.
//!
//! **Launch-seam preconditions for the `_w{W}` entries** (the planner guarantees all three; a
//! hand-rolled launch must too):
//! 1. `block.x == 32 * W` for the `W` in the entry's own name — nothing else. Threads at
//!    `tid.x >= 32*W` would fold a shared slot that no warp wrote; threads short of it would leave a
//!    slot stale, and every warp executes `shfl.sync.bfly ... 0xffffffff`, so a partially populated
//!    warp is undefined behaviour either way.
//! 2. `block.y == block.z == 1` and `grid.y == grid.z == 1` — the kernel indexes `.x` only.
//! 3. `grid.x >= 1`. Any `grid.x` is correct (the grid-stride covers the rest); `grid.x > rows` just
//!    leaves the surplus CTAs retiring immediately.

use std::sync::OnceLock;

/// Warp width. Every geometry in this file is a multiple of it.
const WARP: u32 = 32;

/// The warps-per-row widths this module generates an entry for, and therefore the only values
/// [`warps_per_row`] may return. Powers of two so the shared fold is a flat unrolled chain and the
/// CTA is a legal shape; capped at 8 (a 256-thread CTA, the same width
/// `ptx_optim::grid_stride_cfg` uses for the other memory-bound family in this crate).
pub const MW_WIDTHS: [u32; 4] = [1, 2, 4, 8];

/// The most warps [`norm_launch`] will put on one row — the last of [`MW_WIDTHS`].
pub const MAX_WARPS_PER_ROW: u32 = 8;

/// Fewest row elements a lane must still own before the planner adds another warp to the row. Below
/// this the two `bar.sync`s per reduction (times two for softmax/LayerNorm) stop being amortized by
/// the loads they parallelize, and a warp with no elements at all still pays them.
///
/// **16, not 4, and the difference is measured.** `sweep_norm_sm_fill` on this 4050 found the only
/// consistent regression in the whole sweep at `rows=16384, cols=768`, where a floor of 4 would pick
/// `W = 4` (6 elements per lane) and every width came back **0.93-0.97x** of the shipped kernel in
/// every round. That shape is also the only one whose working set (50 MB) exceeds the card's 24 MiB
/// L2 — the only genuinely HBM-bound one — and it was already running at 149-172 GB/s against a
/// 192 GB/s pin rate. At a floor of 16 the same row yields `W = 1` and stays on the shipped launch.
/// The cost of the conservatism is the `rows=32, cols=512..768` rows of the table in
/// [`warps_per_row`], where `W = 4` was worth 1.06-1.16x and the floor declines it.
pub const MIN_ELEMS_PER_LANE: usize = 16;

/// CTAs per SM the planner wants before it calls the grid "already covering the machine". Below this
/// the row count alone cannot fill the device and the only lever left is warps-per-row; at or above
/// it, `W` is capped at [`OCCUPANCY_WARPS`].
const FILL_CTAS_PER_SM: usize = 4;

/// Warps per row once the grid already covers the machine.
///
/// The occupancy tables set the ceiling: one row per CTA means the CTA count is fixed at `rows`, so
/// warps/SM is `min(max_ctas_per_sm, warp_cap/W) * W`, and at `W = 1` that saturates at
/// `max_ctas_per_sm` — **16 of Ada's 48 warps/SM** (cc 8.6/8.7/8.9 cap 16 CTAs/SM) and **32 of the 64
/// on A100/H100** (cc 8.0/9.0 cap 32 CTAs/SM). `W = 4` reaches the warp ceiling on all three (Ada
/// 12x4 = 48, A100/H100 16x4 = 64), and going past it buys no residency.
///
/// Measured, that ceiling is also where the returns stop: at `rows >= 4*sm` this card gave
/// 1.11-1.16x at `W = 2..4` and never more at 8.
const OCCUPANCY_WARPS: u32 = 4;

/// Grid cap in CTAs per SM, mirroring `ptx_optim::grid_stride_cfg`'s `32 * SM` oversubscription. Past
/// this the grid-stride loop takes over, so a million-row launch becomes a bounded slate of CTAs each
/// walking many rows instead of a million one-row CTAs.
const GRID_CTAS_PER_SM: usize = 32;

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

/// The module header, from the single source ([`crate::ptx_target`]). This family emits only
/// `shfl.sync` / `bar.sync` / static `.shared` / plain f32 / SFU approximations, all legal at the
/// **`sm_80` floor** — tagging it with the device's own arch would make the module unloadable on any
/// *older* part (PTX is forward-compatible only).
fn header() -> String {
    String::from(crate::ptx_target::HDR_SM80)
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
///
/// The variance is the **two-pass** `mean((x - mean)^2)`, not the one-pass identity
/// `E[x^2] - mean^2`. The one-pass form catastrophically cancels in f32 on any row whose mean is
/// large relative to its spread: at `mean = 1e4` both terms are `~1e8`, whose f32 ulp is 8, while the
/// true variance is `O(1)` — so the subtraction returns rounding noise, and when that noise is
/// negative `sqrt.rn.f32` makes the whole row NaN (measured: 1024/1024 lanes NaN). Both siblings of
/// this kernel already use the two-pass form — the CPU runtime it replaces
/// (`wukong_runtime::norm::layernorm_row_scalar`/`_avx2`) and the gpu-native lowering
/// (`lower::PTX_NORM` LN_S1/LN_S2) — so this was also a CPU-vs-GPU band divergence. The extra pass
/// over the row is the same cost the CPU kernel already pays.
fn layernorm() -> String {
    let mut s = prologue("layernorm");
    // pass 1: s = sum(x) ; mean = s/cols
    s += "    mov.f32 %s,0f00000000;\n";
    s += &strided(
        "lsum",
        "    ld.global.f32 %v,[%addr];\n    add.f32 %s,%s,%v;\n",
    );
    s += &allreduce("s", "add");
    s += "    div.rn.f32 %mean,%s,%colsf;\n";
    // pass 2: s2 = sum((x - mean)^2) ; var = s2/cols ; denom = sqrt(var+eps)
    s += "    mov.f32 %s2,0f00000000;\n";
    s += &strided(
        "lvar",
        "    ld.global.f32 %v,[%addr];\n    sub.f32 %v,%v,%mean;\n    fma.rn.f32 %s2,%v,%v,%s2;\n",
    );
    s += &allreduce("s2", "add");
    s += "    div.rn.f32 %var,%s2,%colsf;\n";
    s += "    add.f32 %denom,%var,%eps;\n    sqrt.rn.f32 %denom,%denom;\n";
    // pass 3: out = (x - mean) / denom
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

// -------------------------------------------------------------------------------------------------
// The SM-filling family (`{softmax,layernorm,rmsnorm}_w{W}`).
// -------------------------------------------------------------------------------------------------

/// The entry name for `base` at `W` warps per row. The width is part of the name because it is part
/// of the *code* — see the module header on why it is not read from `%ntid.x`.
fn mw_name(base: &str, w: u32) -> String {
    format!("{base}_w{w}")
}

/// Prologue of a `_w{W}` entry: the same five params, `W` baked in, and the row grid-stride opened.
///
/// * `%lane` = `tid.x & 31`; for `W > 1` also `%warp` = `tid.x >> 5` and `%sa`, this warp's slot in
///   `red_{name}` (exactly `4*W` bytes — the slate is sized to the width the entry *is*, so an
///   out-of-range slot is not expressible).
/// * `%nctas = %nctaid.x` — the row grid-stride, the one piece of geometry that stays dynamic. Its
///   loop trip count is tiny and uniform, so it costs nothing to leave in a register, and leaving it
///   there is what lets the host size the grid from the device.
///
/// Opens the **row grid-stride loop** (`ROW_{name}`); [`epilogue_w`] closes it. Everything between
/// runs once per row this CTA owns, so per-row state (`%m`, `%s`, `%mean`, ...) is re-initialized
/// inside the loop by each kernel body.
///
/// The row base offset is `mul.wide.u32 %off,%row,%cols` — a genuine 64-bit product, unlike the
/// shipped entries' `mul.lo.s32 %tmp,%row,%cols` which truncates past 2^32 elements. Same result for
/// every shape either kernel is launched at today; strictly wider here because it was free.
fn prologue_w(base: &str, w: u32) -> String {
    let name = mw_name(base, w);
    let mut s = format!(
        r#".visible .entry {name}(
    .param .u32 pRows,
    .param .u32 pCols,
    .param .f32 pEps,
    .param .u64 pX,
    .param .u64 pOut
)
{{
"#
    );
    if w > 1 {
        s += &format!("    .shared .align 4 .b8 red_{name}[{}];\n", 4 * w);
    }
    s += "    .reg .pred %p0;\n";
    s += "    .reg .f32 %rt,%v,%e,%m,%s,%s2,%mean,%var,%denom,%eps,%colsf,%acc;\n";
    s += "    .reg .b32 %rows,%cols,%row,%lane,%warp,%i,%tmp,%nctas,%sa;\n";
    s += "    .reg .b64 %X,%Out,%xptr,%optr,%addr,%off;\n";
    s += "    ld.param.u32 %rows,[pRows];\n";
    s += "    ld.param.u32 %cols,[pCols];\n";
    s += "    ld.param.f32 %eps,[pEps];\n";
    s += "    ld.param.u64 %X,[pX];\n";
    s += "    ld.param.u64 %Out,[pOut];\n";
    s += "    cvta.to.global.u64 %X,%X;\n";
    s += "    cvta.to.global.u64 %Out,%Out;\n";
    s += "    cvt.rn.f32.u32 %colsf,%cols;\n";
    s += "    mov.u32 %tmp,%tid.x;\n";
    s += "    and.b32 %lane,%tmp,31;\n";
    if w > 1 {
        s += "    shr.u32 %warp,%tmp,5;\n";
        s += &format!("    mov.u32 %sa,red_{name};\n");
        s += "    mad.lo.u32 %sa,%warp,4,%sa;\n";
    }
    s += "    mov.u32 %nctas,%nctaid.x;\n";
    s += "    mov.u32 %row,%ctaid.x;\n";
    s += &format!("ROW_{name}:\n");
    s += "    setp.ge.u32 %p0,%row,%rows;\n";
    s += &format!("    @%p0 bra RET_{name};\n");
    s += "    mul.wide.u32 %off,%row,%cols;\n";
    s += "    shl.b64 %off,%off,2;\n";
    s += "    add.s64 %xptr,%X,%off;\n";
    s += "    add.s64 %optr,%Out,%off;\n";
    s
}

/// Close the row grid-stride loop opened by [`prologue_w`] and return.
fn epilogue_w(name: &str) -> String {
    format!("    add.u32 %row,%row,%nctas;\n    bra ROW_{name};\nRET_{name}:\n    ret;\n}}\n")
}

/// A CTA-strided pass `for (i = tid.x; i < cols; i += 32*W)` — the `_w{W}` twin of [`strided`].
/// `body` may use `%i` and must leave the per-element address in `%addr = xptr + i*4`. `tag` makes
/// labels unique.
///
/// **The stride is an immediate.** That is the whole reason this family is generated per width
/// instead of reading `%ntid.x`: with a constant induction step ptxas unrolls the loop and folds the
/// resulting `+128`/`+256`/... into the load's addressing mode, so a warp keeps several loads in
/// flight; with a register step it does neither and the loop degrades to one outstanding load. On
/// this RTX 4050 that difference measured **0.53-0.82x** across the shape sweep, at a launch
/// otherwise identical to the shipped kernel's.
///
/// At `W == 1` this is [`strided`] verbatim (`tid.x == lane` in a 32-thread CTA, stride 32), which is
/// what makes `one_warp_entries_are_bit_identical_to_the_shipped_kernel` a meaningful A/B.
fn strided_w(tag: &str, w: u32, body: &str) -> String {
    let stride = WARP * w;
    format!(
        "    mov.u32 %i,%tid.x;\nL_{tag}:\n    setp.ge.u32 %p0,%i,%cols;\n    @%p0 bra E_{tag};\n    mul.wide.u32 %off,%i,4;\n    add.s64 %addr,%xptr,%off;\n{body}    add.u32 %i,%i,{stride};\n    bra L_{tag};\nE_{tag}:\n"
    )
}

/// CTA-wide all-reduce of `%{reg}` under `op` ("add" or "max"), leaving **every thread of the CTA**
/// with the identical result. Butterfly within the warp ([`allreduce`]) and — only when `W > 1` —
/// then one f32 per warp staged in `red_{name}` and folded, **fully unrolled**, by every thread.
///
/// At `W == 1` this emits nothing but the butterfly: no shared memory, no barrier, no predicate. The
/// `_w1` entry is therefore the shipped kernel plus a row grid-stride and nothing else.
///
/// Why every thread folds instead of one warp folding and broadcasting: LayerNorm's `%mean` and
/// `%denom` are consumed by every lane in the writeback pass, so they must be bit-identical CTA-wide.
/// An unrolled fold of the same `W <= 8` slots in the same order on every thread is identical by
/// construction, needs no second broadcast, and is `W-1` shared loads with immediate offsets.
///
/// **Synchronization.** Two `bar.sync 0`s bracket the staging: the first orders the per-warp stores
/// before the folds, the second keeps a fast warp from overwriting `red_*` for the *next* reduction
/// (or the next row of the grid-stride loop) while a slow warp is still folding this one. Both sit
/// outside the `lane != 0` divergence, and the row loop is uniform across the CTA (`%row` comes from
/// `%ctaid.x`/`%nctaid.x`, `%rows` is a param) — which is the property that makes a `bar.sync` inside
/// a loop legal here at all.
///
/// **What this does to the f32 error.** `max` is exact and associative, so the softmax max pass is
/// bit-identical at every `W`. The `add` reductions are *reassociated*: at `W` warps each lane
/// serially accumulates `cols/(32W)` terms, then a depth-5 butterfly, then a `W`-long fold. Raising
/// `W` therefore *shortens* the sequential chain (the dominant `O(n)*eps` term) and lengthens only
/// the short tail, so the worst-case forward error is **non-increasing** in `W` — a wider reduction
/// tree is the good direction, and this is a deliberate choice, not an accident. It is still a
/// different order, so results differ from the shipped kernel by a few ulps; that is inside the
/// `c*sqrt(K)*eps` band the CPU<->GPU gate already uses, and it is *not* licence to touch the
/// LayerNorm variance, which stays the two-pass `mean((x-mean)^2)` (see [`layernorm_w`]).
fn allreduce_w(tag: &str, name: &str, w: u32, reg: &str, op: &str) -> String {
    let mut s = allreduce(reg, op);
    if w == 1 {
        return s;
    }
    assert!(
        op == "add" || op == "max",
        "allreduce_w: unknown fold op `{op}`"
    );
    s += "    setp.ne.u32 %p0,%lane,0;\n";
    s += &format!("    @%p0 bra XS_{tag};\n");
    s += &format!("    st.shared.f32 [%sa],%{reg};\n");
    s += &format!("XS_{tag}:\n");
    s += "    bar.sync 0;\n";
    s += &format!("    ld.shared.f32 %acc,[red_{name}];\n");
    for k in 1..w {
        s += &format!("    ld.shared.f32 %rt,[red_{name}+{}];\n", 4 * k);
        s += &format!("    {op}.f32 %acc,%acc,%rt;\n");
    }
    s += &format!("    mov.f32 %{reg},%acc;\n");
    s += "    bar.sync 0;\n";
    s
}

/// `softmax` at `W` warps per row with a row grid-stride. Arithmetic identical to [`softmax`]:
/// max, then `sum(exp(x - max))`, then `exp(x - max) / sum`.
fn softmax_w(w: u32) -> String {
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = mw_name("softmax", w);
    let mut s = prologue_w("softmax", w);
    // pass 1: row max
    s += "    mov.f32 %m,0fFF800000;\n";
    s += &strided_w(
        &format!("smax_{name}"),
        w,
        "    ld.global.f32 %v,[%addr];\n    max.f32 %m,%m,%v;\n",
    );
    s += &allreduce_w(&format!("smax_{name}"), &name, w, "m", "max");
    // pass 2: sum of exp(x - m)
    s += "    mov.f32 %s,0f00000000;\n";
    s += &strided_w(
        &format!("ssum_{name}"),
        w,
        &format!("    ld.global.f32 %v,[%addr];\n    sub.f32 %v,%v,%m;\n    mul.f32 %v,%v,{log2e};\n    ex2.approx.f32 %e,%v;\n    add.f32 %s,%s,%e;\n"),
    );
    s += &allreduce_w(&format!("ssum_{name}"), &name, w, "s", "add");
    // pass 3: out = exp(x - m) / s
    s += &strided_w(
        &format!("swr_{name}"),
        w,
        &format!("    ld.global.f32 %v,[%addr];\n    sub.f32 %v,%v,%m;\n    mul.f32 %v,%v,{log2e};\n    ex2.approx.f32 %e,%v;\n    div.rn.f32 %e,%e,%s;\n    mul.wide.u32 %off,%i,4;\n    add.s64 %addr,%optr,%off;\n    st.global.f32 [%addr],%e;\n"),
    );
    s += &epilogue_w(&name);
    s
}

/// `layernorm` at `W` warps per row with a row grid-stride.
///
/// **The variance is still the two-pass `mean((x - mean)^2)`** — pass 1 sums `x` to a mean, pass 2
/// sums the squared deviations from *that* mean. Widening the reduction is exactly the change that
/// tempts a rewrite to the one-pass `E[x^2] - mean^2` (it needs one pass and one reduction instead of
/// two of each, which looks like a free halving of the barriers). It is not free: it cancels
/// catastrophically in f32 and turns whole rows NaN through `sqrt.rn.f32` — measured 1024/1024 lanes
/// on this card at `mean = 1e4`. See [`layernorm`] for the full account, and
/// `sm_filling_layernorm_survives_the_large_mean_row` for the gate that would catch a regression at
/// every `W`.
fn layernorm_w(w: u32) -> String {
    let name = mw_name("layernorm", w);
    let mut s = prologue_w("layernorm", w);
    // pass 1: s = sum(x) ; mean = s/cols
    s += "    mov.f32 %s,0f00000000;\n";
    s += &strided_w(
        &format!("lsum_{name}"),
        w,
        "    ld.global.f32 %v,[%addr];\n    add.f32 %s,%s,%v;\n",
    );
    s += &allreduce_w(&format!("lsum_{name}"), &name, w, "s", "add");
    s += "    div.rn.f32 %mean,%s,%colsf;\n";
    // pass 2: s2 = sum((x - mean)^2) ; var = s2/cols ; denom = sqrt(var+eps)
    s += "    mov.f32 %s2,0f00000000;\n";
    s += &strided_w(
        &format!("lvar_{name}"),
        w,
        "    ld.global.f32 %v,[%addr];\n    sub.f32 %v,%v,%mean;\n    fma.rn.f32 %s2,%v,%v,%s2;\n",
    );
    s += &allreduce_w(&format!("lvar_{name}"), &name, w, "s2", "add");
    s += "    div.rn.f32 %var,%s2,%colsf;\n";
    s += "    add.f32 %denom,%var,%eps;\n    sqrt.rn.f32 %denom,%denom;\n";
    // pass 3: out = (x - mean) / denom
    s += &strided_w(
        &format!("lwr_{name}"),
        w,
        "    ld.global.f32 %v,[%addr];\n    sub.f32 %v,%v,%mean;\n    div.rn.f32 %v,%v,%denom;\n    mul.wide.u32 %off,%i,4;\n    add.s64 %addr,%optr,%off;\n    st.global.f32 [%addr],%v;\n",
    );
    s += &epilogue_w(&name);
    s
}

/// `rmsnorm` at `W` warps per row with a row grid-stride. One reduction, two passes.
fn rmsnorm_w(w: u32) -> String {
    let name = mw_name("rmsnorm", w);
    let mut s = prologue_w("rmsnorm", w);
    s += "    mov.f32 %s2,0f00000000;\n";
    s += &strided_w(
        &format!("rsum_{name}"),
        w,
        "    ld.global.f32 %v,[%addr];\n    fma.rn.f32 %s2,%v,%v,%s2;\n",
    );
    s += &allreduce_w(&format!("rsum_{name}"), &name, w, "s2", "add");
    s += "    div.rn.f32 %v,%s2,%colsf;\n    add.f32 %denom,%v,%eps;\n    sqrt.rn.f32 %denom,%denom;\n";
    s += &strided_w(
        &format!("rwr_{name}"),
        w,
        "    ld.global.f32 %v,[%addr];\n    div.rn.f32 %v,%v,%denom;\n    mul.wide.u32 %off,%i,4;\n    add.s64 %addr,%optr,%off;\n    st.global.f32 [%addr],%v;\n",
    );
    s += &epilogue_w(&name);
    s
}

/// The header plus the three **shipped** one-warp entries, in their shipped order — the byte-exact
/// prefix of [`norm_ptx`] that `shipped_entries_are_byte_identical` pins.
fn shipped_entries() -> String {
    let mut m = header();
    m += &softmax();
    m += &layernorm();
    m += &rmsnorm();
    m
}

/// The norm module — the three shipped one-warp entries, then `{softmax,layernorm,rmsnorm}_w{W}` for
/// every `W` in [`MW_WIDTHS`] — generated once and cached.
///
/// **One module, one `"norm"` key.** `Gpu::function` caches on the key alone and never re-examines
/// the PTX, so two modules under one key would silently share whichever loaded first; and every
/// caller in the crate (`gpu::norm`, the resident transformer layers, `serving`, `ptx_autodiff_bwd`,
/// the cuBLAS attention peer) already asks for `"norm"`. Putting the new entries in the same text
/// means switching a launch to the SM-filling family is an entry-name change and nothing else, and it
/// puts them inside the crate's existing corpus laws for free: `ptx::every_dispatched_ptx_family_is_
/// pure_ascii` and `gpu::no_module_declares_a_driver_floor_its_instructions_do_not_need` both already
/// enumerate `ptx_norm::norm_ptx`. The costs are that the module's text changed (so the first run
/// after this lands is a cold miss in the device-keyed cubin cache) and that it is now 12 entries
/// larger; `norm_module_stays_within_its_jit_budget` pins the size so that growth is deliberate.
pub fn norm_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = shipped_entries();
        for w in MW_WIDTHS {
            m += &softmax_w(w);
            m += &layernorm_w(w);
            m += &rmsnorm_w(w);
        }
        m
    })
    .as_str()
}

/// The entry name + launch geometry [`norm_launch`] chose for one normalization.
///
/// `grid`/`block` are the `.x` extents; `.y`/`.z` are 1 by construction (the kernels index `.x` only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormLaunch {
    /// `"softmax"` / `"layernorm"` / `"rmsnorm"` when the plan came out at today's shipped geometry,
    /// else the `_w{W}` sibling for this plan's `warps_per_row`.
    pub entry: String,
    /// CTAs. Each CTA owns one row per grid-stride step.
    pub grid: u32,
    /// Threads per CTA — always `32 * warps_per_row`, always in `32..=256`.
    pub block: u32,
    /// `W`: warps cooperating on one row. Always a member of [`MW_WIDTHS`].
    pub warps_per_row: u32,
}

impl NormLaunch {
    /// Whether this plan needs the SM-filling entry family (i.e. is not the shipped geometry).
    pub fn is_sm_filling(&self) -> bool {
        self.entry != strip_width(&self.entry)
    }
}

/// `"layernorm_w4"` -> `"layernorm"`; a name with no `_w<digits>` tail is returned unchanged. The
/// single place the entry-name encoding is decoded, so the launch-seam assertion
/// (`block == 32 * W`) reads `W` from the same source that wrote it.
pub fn strip_width(entry: &str) -> &str {
    match entry.rsplit_once("_w") {
        Some((base, w)) if !w.is_empty() && w.bytes().all(|b| b.is_ascii_digit()) => base,
        _ => entry,
    }
}

/// The `W` an entry name encodes: `"layernorm_w4"` -> `Some(4)`, a shipped name -> `None`.
pub fn entry_width(entry: &str) -> Option<u32> {
    let (_, w) = entry.rsplit_once("_w")?;
    w.parse().ok()
}

/// `n` rounded **down** to a power of two; `prev_pow2(0) == 1`.
fn prev_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    1usize << (usize::BITS - 1 - n.leading_zeros())
}

/// Warps per row for a `[rows, cols]` normalization on a device with `sm_count` SMs — the pure
/// heuristic, with no environment and no device in it so it is unit-testable on any box.
///
/// One bound, then a regime split:
///
/// * **Useful width.** A row of `cols` elements can feed at most `cols / (32 * MIN_ELEMS_PER_LANE)`
///   warps before lanes run dry; floored to a power of two and capped at [`MAX_WARPS_PER_ROW`].
///   Short rows therefore stay at `W = 1` and keep the shipped launch exactly.
/// * **Regime.** With one row per CTA the CTA count *is* the row count, so `rows` alone decides
///   whether the grid covers the machine. Below `FILL_CTAS_PER_SM * sm_count` it does not, and no
///   choice of `W` can add CTAs — but every extra warp is extra memory parallelism on the SMs that
///   *are* busy, so the planner takes the full useful width. At or above it, `W` is capped at
///   [`OCCUPANCY_WARPS`], past which no residency is bought.
///
/// **What this cannot do.** SMs busy is `min(ctas, sm_count)` and one CTA cannot span SMs, so a
/// launch with `rows < sm_count` leaves `sm_count - rows` SMs idle at every `W`. Lighting them needs
/// more than one CTA per row, i.e. a cross-CTA reduction (a partials buffer plus either a second
/// launch or a cooperative grid sync) — a different kernel signature and a host change, deliberately
/// out of scope here. What this *does* fix is that those `min(rows, sm_count)` busy SMs stop running
/// at one warp each.
///
/// **Measured on the 4050** (`sweep_norm_sm_fill`, LayerNorm f32, 20 SMs, AC+charging, round-robin
/// arms, best-of-50 minima, ratio against the shipped launch). Ranges over the three rounds whose
/// `C(twin)` control landed inside 5% — the magnitudes move by up to 1.5x between rounds on this
/// laptop because other work on the box inflates launch+sync latency (one round had the shipped
/// baseline itself at 19 ms where another had 0.8 ms), so only the **direction and the ordering** are
/// claimed here. A quiet datacenter box is where point values should be taken.
///
/// | rows | cols | regime | W chosen | measured | best available |
/// |---|---|---|---|---|---|
/// | 8 | 4096 | underfill | 8 | **1.7-2.5x** | W8 |
/// | 32 | 256 | underfill | 1 | 1.00x | flat, nothing to take |
/// | 32 | 512 | underfill | 1 | 1.00x | W4 1.06-1.16x |
/// | 32 | 768 | underfill | 1 | 1.00x | W4 1.07x |
/// | 32 | 1024 | underfill | 2 | **1.1-1.7x** | W8, +3% over W2 |
/// | 32 | 2048 | underfill | 4 | **1.3-1.5x** | W8, +3% over W4 |
/// | 32 | 4096 | underfill | 8 | **1.6-2.4x** | W8 |
/// | 64 | 8192 | underfill | 8 | **2.8-3.5x** | W8 |
/// | 128 | 1024 | covered | 2 | 1.13x | W4, +0% |
/// | 512 | 1024 | covered | 2 | 1.11-1.16x | W4, +1-10% |
/// | 4096 | 4096 | covered | 4 | 1.00x | nothing; already 168 GB/s |
/// | 16384 | 768 | covered | 1 | 1.00x | **shipped** — every W>1 was 0.93-0.97x |
///
/// Two things that table is worth reading twice. First, the covered regime is nearly tapped out on
/// this part: at `rows >= 4*sm` the shipped kernel already runs at 149-176 GB/s against a 192 GB/s
/// pin rate, so the occupancy argument above buys ~1.1x and at `cols = 768` it buys **less than
/// nothing** — which is what [`MIN_ELEMS_PER_LANE`] is set to exclude. Second, that ceiling is an
/// Ada fact, not a law: A100/H100 have 2x the warp slots per SM and vastly more bandwidth to feed,
/// so the covered regime is genuinely open there. `WUKONG_NORM_WARPS` is the knob for that round.
pub fn warps_per_row(sm_count: i32, rows: usize, cols: usize) -> u32 {
    let sm = sm_count.max(1) as usize;
    let useful = prev_pow2(
        (cols / (WARP as usize * MIN_ELEMS_PER_LANE)).clamp(1, MAX_WARPS_PER_ROW as usize),
    ) as u32;
    if rows >= FILL_CTAS_PER_SM.saturating_mul(sm) {
        useful.min(OCCUPANCY_WARPS)
    } else {
        useful
    }
}

/// The `WUKONG_NORM_WARPS` override, for calibrating [`warps_per_row`] on silicon this box does not
/// have. Unset (or empty) means "use the heuristic". A value outside [`MW_WIDTHS`] is a **panic**,
/// not a silent fallback: there is no entry for a width this module did not generate, and a
/// measurement round driven by a typo'd knob that quietly measured the default is worse than no
/// round.
fn warps_override() -> Option<u32> {
    let raw = std::env::var("WUKONG_NORM_WARPS").ok()?;
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let w: u32 = raw.parse().unwrap_or_else(|_| {
        panic!("WUKONG_NORM_WARPS={raw:?} is not a number (want one of {MW_WIDTHS:?})")
    });
    assert!(
        MW_WIDTHS.contains(&w),
        "WUKONG_NORM_WARPS={w} is not one of the generated widths {MW_WIDTHS:?}"
    );
    Some(w)
}

/// Pick the entry name and launch geometry for a `[rows, cols]` normalization of `base_entry`
/// (`"softmax"` / `"layernorm"` / `"rmsnorm"`) on a device with `sm_count` SMs.
///
/// **`W == 1` is today's launch, always.** One warp per row means there is nothing to gain and the
/// grid cap only costs: capping `grid` and grid-striding instead measured **0.90-1.00x** across the
/// shape sweep (it removes independent CTAs the scheduler was using to hide latency and serializes
/// those rows inside a CTA). So at `W == 1` this returns the shipped entry with `grid = rows,
/// block = 32` — byte-for-byte what `gpu::norm` performs today, uncapped. Wiring this in is
/// therefore provably a no-op for every shape it does not intend to change, and every existing
/// device gate must stay green across the wiring commit.
///
/// At `W > 1` it returns the `_w{W}` sibling with `block = 32 * W` and the grid capped at
/// `GRID_CTAS_PER_SM * sm_count`; the `_w{W}` grid-stride covers any surplus rows, which is exactly
/// the property the shipped entries lack (they read `%ctaid.x` as *the* row index and never advance
/// it, so launching one at `grid < rows` would silently leave every row past the grid untouched).
pub fn norm_launch(base_entry: &str, sm_count: i32, rows: usize, cols: usize) -> NormLaunch {
    assert!(
        matches!(base_entry, "softmax" | "layernorm" | "rmsnorm"),
        "norm_launch: `{base_entry}` is not a norm entry in this module"
    );
    let sm = sm_count.max(1) as usize;
    let grid_cap = GRID_CTAS_PER_SM.saturating_mul(sm).max(1);
    let w = warps_override().unwrap_or_else(|| warps_per_row(sm_count, rows, cols));
    debug_assert!(MW_WIDTHS.contains(&w));
    // A grid of 0 is a driver rejection, and a row count past u32 is not a shape this module's u32
    // `pRows` param can express in the first place.
    let rows_u = u32::try_from(rows).expect("norm_launch: rows must fit in the u32 `pRows` param");
    if w == 1 {
        return NormLaunch {
            entry: base_entry.to_string(),
            grid: rows_u.max(1),
            block: WARP,
            warps_per_row: 1,
        };
    }
    NormLaunch {
        entry: mw_name(base_entry, w),
        grid: rows_u.min(grid_cap as u32).max(1),
        block: WARP * w,
        warps_per_row: w,
    }
}

#[cfg(test)]
mod tests {
    use cudarc::driver::{DriverError, LaunchConfig, PushKernelArg};

    /// **Header-floor gate, device-free.** The norm module must carry the `sm_80` floor from
    /// [`crate::ptx_target`] — never the device's own arch. PTX is forward-compatible only, so an
    /// `sm_89` tag here would fail `cuModuleLoadData` on every A100 while changing nothing on Ada.
    #[test]
    fn norm_module_is_tagged_at_the_sm80_floor() {
        let ptx = super::norm_ptx();
        assert!(
            ptx.starts_with(crate::ptx_target::HDR_SM80),
            "norm PTX must open with ptx_target::HDR_SM80, got: {:?}",
            &ptx[..ptx.len().min(64)]
        );
        assert!(
            !ptx.contains(crate::ptx_target::TARGET_SM89),
            "an Ampere-legal module must not claim the Ada floor"
        );
    }

    /// FNV-1a 64 over bytes — a fixed, dependency-free hash so a pinned constant means one exact
    /// string and nothing else.
    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    /// **The byte-identity gate for the shipped geometry.** Two halves, and both are needed:
    ///
    /// 1. `norm_ptx()` still *opens* with exactly `header() + softmax() + layernorm() + rmsnorm()`,
    ///    so the `_w{W}` entries were appended and did not perturb a shipped one. This is a real string
    ///    comparison of the prefix, but on its own it is circular — it compares the generators to
    ///    themselves, so an edit inside `softmax()` moves both sides together.
    /// 2. The pinned length and FNV-1a-64 of that prefix break the circularity: they are a fixed
    ///    value recorded from the pre-change module, so any edit to the shipped text fails here.
    ///
    /// If you deliberately change a shipped entry, the failure prints the new pair — update both
    /// constants in the same commit, and say in the body why the shipped bytes moved. Every caller
    /// of the `"norm"` key (`gpu::norm`, the resident layers, `serving`, `ptx_autodiff_bwd`, the
    /// cuBLAS attention peer) launches these entries at `grid=(rows,1,1) block=(32,1,1)`.
    #[test]
    fn shipped_entries_are_byte_identical() {
        /// Bytes of `header() + softmax() + layernorm() + rmsnorm()` at `0e1b2ea`, the parent commit
        /// of the SM-fill change. (Recorded by reading the value out of this assertion's own failure
        /// message after confirming, via `git diff`, that not one line of `header`/`prologue`/
        /// `strided`/`allreduce`/`softmax`/`layernorm`/`rmsnorm` outside a doc comment had moved.)
        const SHIPPED_LEN: usize = 7233;
        /// FNV-1a-64 of those bytes.
        const SHIPPED_FNV1A64: u64 = 0x7848_a8b9_c501_5e74;

        let shipped = super::shipped_entries();
        let ptx = super::norm_ptx();
        assert!(
            ptx.len() > shipped.len(),
            "norm_ptx() must be the shipped entries plus the width-specialized family"
        );
        assert_eq!(
            &ptx[..shipped.len()],
            shipped.as_str(),
            "the width-specialized entries must be appended after the shipped ones, not interleaved"
        );
        assert_eq!(
            (shipped.len(), fnv1a64(shipped.as_bytes())),
            (SHIPPED_LEN, SHIPPED_FNV1A64),
            "the shipped one-warp entries changed. If that was deliberate, re-pin SHIPPED_LEN and \
             SHIPPED_FNV1A64 to ({}, {:#018x}) in the same commit and justify it; if not, you just \
             changed the kernel every `\"norm\"` caller launches today.",
            shipped.len(),
            fnv1a64(shipped.as_bytes())
        );
    }

    /// **ASCII gate for this family.** `ptx::every_dispatched_ptx_family_is_pure_ascii` already scans
    /// `norm_ptx()`, but that list lives in another file and this module's prose is full of `<->` and
    /// `*` math that a copy-paste could turn into the real characters. One non-ASCII byte is a
    /// `ptxas fatal` at `cuModuleLoadData`, invisible to any device-free gate that does not look.
    #[test]
    fn norm_ptx_is_pure_ascii() {
        let ptx = super::norm_ptx();
        if let Some((i, line)) = ptx.lines().enumerate().find(|(_, l)| !l.is_ascii()) {
            panic!(
                "ptx_norm: PTX must be pure ASCII (the driver rejects the module) -- line {}: {line}",
                i + 1
            );
        }
    }

    /// The text of one entry: everything from its `.visible .entry` line to the next one.
    fn entry_body<'a>(ptx: &'a str, name: &str) -> &'a str {
        let after = ptx
            .split(&format!(".visible .entry {name}(\n"))
            .nth(1)
            .unwrap_or_else(|| panic!("{name}: entry not found in the norm module"));
        after.split(".visible .entry").next().unwrap()
    }

    /// **Structure gate for the SM-filling family, device-free.** For every generated width: the
    /// entry exists, strides by the **immediate** `32*W` (the whole point of generating per width —
    /// a register stride measured 0.53-0.82x), carries the row grid-stride, and stages exactly `4*W`
    /// bytes of shared memory behind exactly two barriers per fold. `W == 1` must carry none of the
    /// staging at all.
    #[test]
    fn sm_filling_entries_bake_their_width() {
        let ptx = super::norm_ptx();
        let mut barriers = 0usize;
        for w in super::MW_WIDTHS {
            for base in ["softmax", "layernorm", "rmsnorm"] {
                let name = format!("{base}_w{w}");
                let body = entry_body(ptx, &name);
                assert!(
                    body.contains(&format!("ROW_{name}:")),
                    "{name}: missing the row grid-stride loop"
                );
                assert!(
                    body.contains("mov.u32 %nctas,%nctaid.x;"),
                    "{name}: the row stride must come from the grid, not the row count"
                );
                // The CTA stride is an immediate, and it is the only one in the entry.
                assert!(
                    body.contains(&format!("    add.u32 %i,%i,{};\n", 32 * w)),
                    "{name}: the CTA stride must be the immediate {}",
                    32 * w
                );
                assert!(
                    !body.contains("%ntid.x"),
                    "{name}: reading the width back out of the launch is what cost 1.9x"
                );
                let folds = body.matches("bar.sync 0;").count();
                if w == 1 {
                    assert_eq!(folds, 0, "{name}: one warp needs no barrier");
                    assert!(!body.contains(".shared"), "{name}: one warp needs no SMEM");
                } else {
                    let reductions = if base == "rmsnorm" { 1 } else { 2 };
                    assert_eq!(
                        folds,
                        2 * reductions,
                        "{name}: every cross-warp fold needs exactly two barriers"
                    );
                    assert!(
                        body.contains(&format!(".shared .align 4 .b8 red_{name}[{}];", 4 * w)),
                        "{name}: the staging slate must be exactly one f32 per warp"
                    );
                    // The unrolled fold must read every slot the CTA writes, and no other.
                    assert!(body.contains(&format!("ld.shared.f32 %acc,[red_{name}];")));
                    for k in 1..w {
                        assert!(
                            body.contains(&format!("ld.shared.f32 %rt,[red_{name}+{}];", 4 * k)),
                            "{name}: fold skips warp {k}"
                        );
                    }
                    assert!(
                        !body.contains(&format!("[red_{name}+{}]", 4 * w)),
                        "{name}: fold reads past the slate"
                    );
                }
                barriers += folds;
            }
        }
        // 3 widths above 1, five reductions each across the three ops, two barriers per fold.
        assert_eq!(barriers, 3 * 5 * 2);
        // The shipped entries must not have grown a barrier or a shared slate.
        let shipped = super::shipped_entries();
        assert!(!shipped.contains("bar.sync"));
        assert!(!shipped.contains(".shared"));
    }

    /// **The module is bigger now; keep that deliberate.** 12 extra entries is a real JIT cost paid
    /// by every caller of the `"norm"` key, amortized by the on-disk cubin cache. Pinning the size
    /// means a future width, or a body that grew by accident, has to be argued for rather than
    /// noticed on a slow machine. Bounds, not an equality, so a doc-only edit does not fail it.
    ///
    /// Measured on this box (`sweep_norm_sm_fill` prints it): the whole 47 KB module loads in
    /// **13-41 ms with a cold cubin cache** (the driver's `cuLink` SASS compile, once per machine per
    /// PTX text) and **1-4 ms warm** (once per process). Against a 7 KB module before. That is the
    /// price of specializing the stride per width, and it buys 1.7-3.5x on the underfilled shapes.
    #[test]
    fn norm_module_stays_within_its_jit_budget() {
        let n = super::norm_ptx().len();
        assert!(
            (40_000..=56_000).contains(&n),
            "the norm module is {n} bytes of PTX; it was 47_546 with 3 shipped + 12 generated \
             entries. If a width or a body was added deliberately, move the bound."
        );
        let entries = super::norm_ptx().matches(".visible .entry ").count();
        assert_eq!(entries, 3 + 3 * super::MW_WIDTHS.len());
    }

    /// **The numerics contract, read off the PTX text, at every width.** LayerNorm's variance must be
    /// the two-pass `mean((x - mean)^2)`: pass 2 subtracts the mean and accumulates the square of the
    /// deviation. The one-pass `E[x^2] - mean^2` would show up as a `%mean`-squared term and no
    /// `sub.f32` before the `fma`. Widening the reduction is precisely the edit that invites that
    /// "simplification", so the law is asserted structurally as well as numerically (the device gate
    /// below).
    #[test]
    fn sm_filling_layernorm_keeps_the_two_pass_variance() {
        let ptx = super::norm_ptx();
        for w in super::MW_WIDTHS {
            let name = format!("layernorm_w{w}");
            let body = entry_body(ptx, &name);
            assert!(
                body.contains("sub.f32 %v,%v,%mean;\n    fma.rn.f32 %s2,%v,%v,%s2;"),
                "{name} must accumulate squared deviations from the mean (two-pass variance)"
            );
            assert!(
                !body.contains("mul.f32 %mean,%mean,%mean")
                    && !body.contains("fma.rn.f32 %var,%mean,%mean"),
                "{name} must never form E[x^2] - mean^2: it cancels in f32 and NaNs the row"
            );
        }
    }

    /// **The planner, device-free.** Every row of this table is either a shape
    /// `sweep_norm_sm_fill` measured on the 4050 (see [`super::warps_per_row`]'s doc table) or a
    /// boundary of the two rules, so a future retune has to disagree with a measurement out loud.
    #[test]
    fn warps_per_row_matches_the_documented_regimes() {
        use super::warps_per_row as w;
        // Short rows can never feed 16 elements per lane on two warps: W stays 1 at every SM count,
        // which is what keeps those shapes on the byte-identical shipped launch.
        for sm in [20, 58, 132] {
            assert_eq!(w(sm, 4096, 64), 1, "sm={sm}");
            assert_eq!(w(sm, 4096, 512), 1, "sm={sm}");
            assert_eq!(w(sm, 8, 512), 1, "sm={sm}");
            // The measured regression: 768 wide is 1.5 warps' worth of work, so it stays shipped.
            assert_eq!(w(sm, 16384, 768), 1, "sm={sm}");
        }
        // Grid already covers the machine (rows >= 4*sm) => useful width capped at 4.
        assert_eq!(w(20, 128, 1024), 2);
        assert_eq!(w(20, 512, 1024), 2);
        assert_eq!(w(20, 4096, 1024), 2);
        assert_eq!(w(20, 4096, 4096), 4);
        assert_eq!(w(132, 4096, 4096), 4);
        assert_eq!(w(20, 4096, 16384), 4, "the cap binds, not the row length");
        // Underfilled grid (rows < 4*sm) => take the full useful width.
        assert_eq!(w(20, 8, 4096), 8);
        assert_eq!(w(20, 32, 4096), 8);
        assert_eq!(w(20, 64, 8192), 8);
        assert_eq!(w(132, 32, 4096), 8);
        assert_eq!(w(132, 32, 1024), 2);
        assert_eq!(w(132, 32, 2048), 4);
        // The regime boundary is CTAs per SM, not "big machine": 80 rows is 4 CTAs/SM on 20 SMs.
        assert_eq!(w(20, 79, 4096), 8);
        assert_eq!(w(20, 80, 4096), 4);
        // Powers of two only: a 1536-wide row can feed 3 warps, so it gets 2.
        assert_eq!(w(132, 32, 1536), 2);
        // Degenerate inputs must not panic or return 0.
        assert_eq!(w(0, 0, 0), 1);
        assert_eq!(w(-1, 1, 1), 1);
        // The heuristic may only name a width this module generated an entry for.
        for sm in [1, 20, 58, 132] {
            for rows in [0usize, 1, 31, 512, 1 << 20] {
                for cols in [1usize, 63, 333, 4096, 50257] {
                    assert!(super::MW_WIDTHS.contains(&w(sm, rows, cols)));
                }
            }
        }
    }

    /// The entry-name encoding round-trips, and does not mangle a shipped name.
    #[test]
    fn entry_names_encode_their_width() {
        for w in super::MW_WIDTHS {
            for base in ["softmax", "layernorm", "rmsnorm"] {
                let n = super::mw_name(base, w);
                assert_eq!(super::strip_width(&n), base);
                assert_eq!(super::entry_width(&n), Some(w));
            }
        }
        for base in ["softmax", "layernorm", "rmsnorm"] {
            assert_eq!(super::strip_width(base), base);
            assert_eq!(super::entry_width(base), None);
        }
    }

    /// **The wiring contract**: a plan at today's geometry must *be* today's launch, name and all —
    /// and, just as important, every plan that is *not* at today's geometry must say so by name, so
    /// the shipped entries are never launched at a grid they cannot handle.
    #[test]
    fn the_shipped_geometry_round_trips_unchanged() {
        // Every W=1 plan is byte-for-byte `gpu::norm`'s launch today, at any row count: the grid is
        // NOT capped there, because capping it measured 0.90-1.00x and buys nothing at one warp.
        for base in ["softmax", "layernorm", "rmsnorm"] {
            for rows in [1u32, 512, 640, 641, 16384, 1 << 20] {
                let p = super::norm_launch(base, 20, rows as usize, 128);
                assert_eq!(
                    p,
                    super::NormLaunch {
                        entry: base.to_string(),
                        grid: rows,
                        block: 32,
                        warps_per_row: 1,
                    },
                    "a W=1 plan must reproduce gpu::norm's launch exactly, uncapped"
                );
                assert!(!p.is_sm_filling());
            }
        }
        // W > 1 switches to the width-specialized sibling and widens the CTA.
        let p = super::norm_launch("rmsnorm", 132, 32, 4096);
        assert_eq!(p.entry, "rmsnorm_w8");
        assert_eq!((p.grid, p.block, p.warps_per_row), (32, 256, 8));
        assert!(p.is_sm_filling());
        // Only there does the grid cap apply, and only there is it safe: the `_w{W}` entries carry
        // the row grid-stride, while the shipped ones read `%ctaid.x` as *the* row index and never
        // advance it, so launching one at `grid < rows` would silently leave the tail untouched.
        let p = super::norm_launch("layernorm", 20, 100_000, 4096);
        assert_eq!(
            (p.entry.as_str(), p.grid, p.block),
            ("layernorm_w4", 640, 128)
        );
        // Every plan is a launchable one: a CTA of exactly the width its entry name encodes, a
        // non-zero grid, and an entry this module actually generated.
        let ptx = super::norm_ptx();
        for sm in [1, 20, 58, 132] {
            for &rows in &[0usize, 1, 7, 32, 4096, 1 << 20] {
                for &cols in &[1usize, 31, 64, 333, 1024, 50257] {
                    for base in ["softmax", "layernorm", "rmsnorm"] {
                        let p = super::norm_launch(base, sm, rows, cols);
                        assert_eq!(p.block, 32 * p.warps_per_row);
                        assert!((32..=256).contains(&p.block));
                        assert!(p.grid >= 1);
                        assert_eq!(super::strip_width(&p.entry), base);
                        assert_eq!(
                            super::entry_width(&p.entry).unwrap_or(1),
                            p.warps_per_row,
                            "the entry name must encode the width the block was sized for"
                        );
                        assert!(
                            ptx.contains(&format!(".visible .entry {}(", p.entry)),
                            "the planner named an entry the module does not contain: {}",
                            p.entry
                        );
                    }
                }
            }
        }
    }

    /// Independent **f64** two-pass LayerNorm over `[rows, cols]` — the oracle, computed in a wider
    /// type and in a different order than either the CPU kernel or the GPU kernel, so it is not a
    /// circular check on either one.
    fn ref_layernorm_f64(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
            let var = row
                .iter()
                .map(|&v| (v as f64 - mean) * (v as f64 - mean))
                .sum::<f64>()
                / cols as f64;
            let denom = (var + eps as f64).sqrt();
            for (i, &v) in row.iter().enumerate() {
                out[r * cols + i] = ((v as f64 - mean) / denom) as f32;
            }
        }
        out
    }

    /// Independent **f64** row softmax — same role as [`ref_layernorm_f64`].
    fn ref_softmax_f64(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let m = row.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v as f64));
            let s: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
            for (i, &v) in row.iter().enumerate() {
                out[r * cols + i] = ((v as f64 - m).exp() / s) as f32;
            }
        }
        out
    }

    /// Independent **f64** RMSNorm — same role as [`ref_layernorm_f64`].
    fn ref_rmsnorm_f64(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let ms = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / cols as f64;
            let denom = (ms + eps as f64).sqrt();
            for (i, &v) in row.iter().enumerate() {
                out[r * cols + i] = (v as f64 / denom) as f32;
            }
        }
        out
    }

    /// The CPU runtime kernel this GPU entry replaces — the band-divergence partner.
    fn cpu_layernorm(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        unsafe {
            wukong_runtime::wukong_norm_f32(
                x.as_ptr(),
                out.as_mut_ptr(),
                rows as i64,
                cols as i64,
                eps.to_bits() as i64,
                wukong_runtime::NORM_LAYERNORM,
            )
        };
        out
    }

    /// Run `body` with the process-wide GPU. A skip is a *passing* libtest test whose stderr libtest
    /// captures, so `WUKONG_GPU_REQUIRED=1` turns "no device" into a failure instead of a green run
    /// that measured nothing (§3A P3).
    fn with_gpu(name: &str, body: impl FnOnce(&mut crate::gpu::Gpu)) {
        let mut guard = crate::gpu::gpu();
        match guard.as_mut() {
            Some(g) => body(g),
            None => {
                let why = crate::gpu::init_error().unwrap_or("no CUDA device reachable");
                assert!(
                    !crate::gpu::gpu_required(),
                    "{name}: WUKONG_GPU_REQUIRED is set but the GPU is unusable: {why}"
                );
                eprintln!("[skip] {name}: GPU unavailable: {why}");
            }
        }
    }

    /// Launch one norm entry at an **explicit** `(grid, block)` and copy the result back. The test
    /// harness's own launcher, deliberately not `gpu::norm` — the whole point is to drive geometries
    /// `gpu::norm` cannot express today.
    ///
    /// Launch-seam preconditions, asserted here because the driver would not name them: the argument
    /// list is the module's five `.param`s in order, the buffer is exactly `rows*cols`, the grid is
    /// non-empty, and — the one that actually bites — **the CTA is exactly the width the entry's own
    /// name encodes**, read back through [`super::entry_width`] rather than from a second parameter.
    #[allow(clippy::too_many_arguments)]
    fn launch_norm(
        g: &mut crate::gpu::Gpu,
        entry: &str,
        x: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
        grid: u32,
        block: u32,
    ) -> Result<Vec<f32>, DriverError> {
        assert_eq!(x.len(), rows * cols, "{entry}: buffer must be rows*cols");
        assert!(grid >= 1, "{entry}: empty grid");
        match super::entry_width(entry) {
            Some(w) => assert_eq!(
                block,
                32 * w,
                "{entry}: the entry name encodes W={w}, so the CTA must be exactly {} threads",
                32 * w
            ),
            None => assert!(
                grid as usize == rows && block == 32,
                "{entry}: the shipped entries have no grid-stride and assume one warp per row"
            ),
        }
        let f = g.function("norm", super::norm_ptx(), entry)?;
        let x_d = g.stream.memcpy_stod(x)?;
        let mut out_d = g.stream.memcpy_stod(&vec![0f32; x.len()])?;
        let (r, c) = (rows as u32, cols as u32);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bld = g.stream.launch_builder(&f);
        bld.arg(&r).arg(&c).arg(&eps).arg(&x_d).arg(&mut out_d);
        unsafe { bld.launch(cfg)? };
        g.stream.memcpy_dtov(&out_d)
    }

    /// **The SM-filling correctness gate.** Every entry, at every generated width, over shapes chosen
    /// to break the geometry rather than to flatter it:
    ///
    /// * `cols` shorter than one warp, and shorter than the whole CTA — most warps get **no**
    ///   elements at all and must still stage an identity-free slot and reach both barriers;
    /// * `cols` not a multiple of `32*W` — a ragged tail on the last CTA-stride step;
    /// * `grid < rows` — the row grid-stride must cover every row, including one CTA doing many rows
    ///   in sequence (which is also what re-uses the shared slate across rows, the hazard the second
    ///   `bar.sync` exists for);
    /// * `grid > rows` — surplus CTAs must retire without writing anything.
    ///
    /// Compared against an **f64** reference, not against the shipped kernel: the reference is a
    /// different algorithm in a wider type, so it cannot hide a shared mistake. The band is
    /// `c*sqrt(K)*eps` (`diff::assert_close`), because the two orders genuinely differ.
    #[test]
    fn sm_filling_norms_match_the_f64_reference_at_every_width() {
        with_gpu("sm_filling_norms", |g| {
            let mut rng = crate::diff::Rng::new(0xB6_51_1A);
            let eps = 1e-5f32;
            // (rows, cols, grid_override)
            let shapes: &[(usize, usize, Option<u32>)] = &[
                (1, 1, None),
                (3, 17, None),       // cols < one warp: warps 1.. get nothing
                (5, 100, None),      // ragged tail at every W
                (8, 333, None),      // prime-ish tail
                (64, 1024, None),    // the common transformer row
                (7, 4096, None),     // long row, few rows: the underfill shape
                (100, 129, Some(3)), // grid << rows: the grid-stride, many rows per CTA
                (4, 512, Some(64)),  // grid >> rows: surplus CTAs must retire clean
            ];
            let mut checked = 0usize;
            for &(rows, cols, grid_over) in shapes {
                let x: Vec<f32> = (0..rows * cols).map(|_| rng.f32_range(-3.0, 3.0)).collect();
                let want_sm = ref_softmax_f64(&x, rows, cols);
                let want_ln = ref_layernorm_f64(&x, rows, cols, eps);
                let want_rms = ref_rmsnorm_f64(&x, rows, cols, eps);
                for w in super::MW_WIDTHS {
                    let grid = grid_over.unwrap_or(rows as u32).max(1);
                    let block = 32 * w;
                    for (base, want) in [
                        ("softmax", &want_sm),
                        ("layernorm", &want_ln),
                        ("rmsnorm", &want_rms),
                    ] {
                        let entry = super::mw_name(base, w);
                        let got = launch_norm(g, &entry, &x, rows, cols, eps, grid, block)
                            .unwrap_or_else(|e| panic!("{entry} {rows}x{cols}: {e:?}"));
                        // c*sqrt(K)*eps, calibrated so cols=1024 lands on the 1e-4 the sibling
                        // gate `gpu::norms_match_cpu_oracle_within_tol` already uses.
                        let abs_tol = 32.0 * (cols as f64).sqrt() * f32::EPSILON as f64;
                        crate::diff::assert_close(
                            &format!("{entry} grid={grid} {rows}x{cols}"),
                            &got,
                            want,
                            abs_tol,
                            1e-3,
                        );
                        checked += 1;
                    }
                }
            }
            eprintln!(
                "[gate] {checked} (entry, width, shape) SM-filling norms match the f64 reference"
            );
        });
    }

    /// **The A/B against the shipped kernel.** The `_w1` entries execute the same access pattern with
    /// the same immediate stride and the same reduction order as the shipped ones (`strided_w(.., 1,
    /// ..)` is `strided` verbatim, and `allreduce_w` at `W == 1` emits nothing but the butterfly), so
    /// at the shipped geometry their outputs must be **bit-identical**, not merely close. That is the
    /// strongest available statement that the new family is the same kernel plus a grid-stride, and
    /// it is what makes the `W == 1` arm of [`super::norm_launch`] safe to wire in.
    #[test]
    fn one_warp_entries_are_bit_identical_to_the_shipped_kernel() {
        with_gpu("w1_vs_shipped", |g| {
            let mut rng = crate::diff::Rng::new(0x51_1A_B6);
            let eps = 1e-5f32;
            for &(rows, cols) in &[(1usize, 1usize), (3, 17), (5, 100), (64, 1024), (7, 4096)] {
                let x: Vec<f32> = (0..rows * cols).map(|_| rng.f32_range(-3.0, 3.0)).collect();
                for base in ["softmax", "layernorm", "rmsnorm"] {
                    let w1 = super::mw_name(base, 1);
                    let a = launch_norm(g, base, &x, rows, cols, eps, rows as u32, 32)
                        .unwrap_or_else(|e| panic!("{base} {rows}x{cols}: {e:?}"));
                    let b = launch_norm(g, &w1, &x, rows, cols, eps, rows as u32, 32)
                        .unwrap_or_else(|e| panic!("{w1} {rows}x{cols}: {e:?}"));
                    assert_eq!(
                        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{w1} must be bit-identical to {base} at the shipped geometry, {rows}x{cols}"
                    );
                }
            }
            eprintln!("[gate] the _w1 entries are bit-identical to the shipped ones");
        });
    }

    /// **The large-mean LayerNorm gate.** `norms_match_cpu_oracle_within_tol` only ever feeds rows
    /// drawn from `[-3, 3]`, where `E[x^2] - mu^2` happens to be well-conditioned. A row whose mean
    /// is large relative to its spread — the ordinary case for un-centred activations — is where the
    /// one-pass identity catastrophically cancels: at `mu = 1e4` both terms are `~1e8`, whose f32 ulp
    /// is 8, while the true variance is `O(1)`.
    ///
    /// Two assertions, in increasing strength:
    ///
    /// 1. **No NaN/Inf, at any conditioning.** Tolerance-free. The one-pass form fails this outright
    ///    — measured on the RTX 4050, rows with `mu = 1e4` and `1e5` came back 1024/1024 NaN, because
    ///    the cancelled variance landed negative and `sqrt.rn.f32` of a negative is NaN.
    /// 2. **Per-row agreement with the f64 oracle within `2·eps32·(|mu|/sigma)`** (floored at 1e-5).
    ///    That is the *unavoidable* f32 forward-error floor for this problem: the row mean can only be
    ///    computed to a relative `~eps32`, i.e. an absolute `~eps32·|mu|`, and the output divides the
    ///    deviations by `sigma`, so `eps32·|mu|/sigma` is what any correct f32 implementation costs.
    ///    Measured, the two-pass GPU kernel sits at 0.005–0.45× that bound at every row, and so does
    ///    the CPU kernel — which is also asserted here, making this a CPU-vs-GPU band gate. The
    ///    one-pass form exceeded the same bound by 28× (`mu=1e2`), 190× (`mu=1e3`) and 4× (`mu=1e6`,
    ///    where it returned an all-zero row), on top of the NaN rows.
    #[test]
    fn layernorm_large_mean_matches_f64_reference() {
        with_gpu("layernorm_large_mean", |g| {
            let (rows, cols) = (8usize, 1024usize);
            let eps = 1e-5f32;
            let mut rng = crate::diff::Rng::new(0x1A7E);
            // Row r has mean ~ base[r] and spread ~ +-1: increasingly ill-conditioned for E[x^2]-mu^2.
            let bases = [0.0f32, 1.0, 1e2, 1e3, 1e4, 1e4, 1e5, 1e6];
            let mut x = vec![0.0f32; rows * cols];
            for (r, &b) in bases.iter().enumerate() {
                for i in 0..cols {
                    x[r * cols + i] = b + rng.f32_range(-1.0, 1.0);
                }
            }

            let got = crate::gpu::norm(g, wukong_runtime::NORM_LAYERNORM, &x, rows, cols, eps)
                .expect("gpu layernorm");
            let oracle = ref_layernorm_f64(&x, rows, cols, eps);
            let cpu = cpu_layernorm(&x, rows, cols, eps);

            // (1) Tolerance-free: a normalized row is finite everywhere.
            let bad = got.iter().filter(|v| !v.is_finite()).count();
            assert_eq!(
                bad,
                0,
                "gpu layernorm produced {bad}/{} non-finite lanes",
                got.len()
            );

            for (r, &b) in bases.iter().enumerate() {
                let (lo, hi) = (r * cols, (r + 1) * cols);
                let mean = x[lo..hi].iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
                let sigma = (x[lo..hi]
                    .iter()
                    .map(|&v| (v as f64 - mean).powi(2))
                    .sum::<f64>()
                    / cols as f64)
                    .sqrt();
                let cond = mean.abs() / sigma;
                let tol = (2.0 * f32::EPSILON as f64 * cond).max(1e-5);
                let gs = crate::diff::err_stats(&got[lo..hi], &oracle[lo..hi]);
                let cs = crate::diff::err_stats(&cpu[lo..hi], &oracle[lo..hi]);
                eprintln!(
                    "layernorm row {r} mean~{b:<9.0} cond={cond:>9.1}: gpu max_abs={:.3e} | cpu max_abs={:.3e} | tol={tol:.3e}",
                    gs.max_abs, cs.max_abs
                );
                // (2) Both f32 implementations must sit at the f32 forward-error floor.
                assert!(
                    gs.max_abs <= tol,
                    "gpu layernorm row {r} (mean~{b}, cond {cond:.0}): max_abs {:.3e} > {tol:.3e} \
                     at lane {} — the variance is not being computed in the stable two-pass form",
                    gs.max_abs,
                    gs.at
                );
                assert!(
                    cs.max_abs <= tol,
                    "cpu layernorm row {r} (mean~{b}, cond {cond:.0}): max_abs {:.3e} > {tol:.3e} \
                     — the ORACLE or the CPU kernel regressed, not the GPU",
                    cs.max_abs
                );
            }
            eprintln!(
                "[gate] gpu layernorm is finite and at the f32 error floor for means 0..1e6 ✓"
            );
        });
    }

    /// **The same large-mean law, at every generated width.** Widening the reduction is exactly the
    /// change that tempts a rewrite to the cancelling one-pass variance, so the gate that catches it
    /// has to run at the new widths too — the structural check above can only see the instruction
    /// mix, not what the hardware does with it.
    #[test]
    fn sm_filling_layernorm_survives_the_large_mean_row() {
        with_gpu("w_layernorm_large_mean", |g| {
            let (rows, cols) = (8usize, 1024usize);
            let eps = 1e-5f32;
            let mut rng = crate::diff::Rng::new(0x1A7E);
            let bases = [0.0f32, 1.0, 1e2, 1e3, 1e4, 1e4, 1e5, 1e6];
            let mut x = vec![0.0f32; rows * cols];
            for (r, &b) in bases.iter().enumerate() {
                for i in 0..cols {
                    x[r * cols + i] = b + rng.f32_range(-1.0, 1.0);
                }
            }
            let oracle = ref_layernorm_f64(&x, rows, cols, eps);
            for w in super::MW_WIDTHS {
                let entry = super::mw_name("layernorm", w);
                let got = launch_norm(g, &entry, &x, rows, cols, eps, rows as u32, 32 * w)
                    .unwrap_or_else(|e| panic!("{entry}: {e:?}"));
                let bad = got.iter().filter(|v| !v.is_finite()).count();
                assert_eq!(bad, 0, "{entry}: {bad}/{} non-finite", got.len());
                for (r, &b) in bases.iter().enumerate() {
                    let (lo, hi) = (r * cols, (r + 1) * cols);
                    let mean = x[lo..hi].iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
                    let sigma = (x[lo..hi]
                        .iter()
                        .map(|&v| (v as f64 - mean).powi(2))
                        .sum::<f64>()
                        / cols as f64)
                        .sqrt();
                    let cond = mean.abs() / sigma;
                    let tol = (2.0 * f32::EPSILON as f64 * cond).max(1e-5);
                    let gs = crate::diff::err_stats(&got[lo..hi], &oracle[lo..hi]);
                    assert!(
                        gs.max_abs <= tol,
                        "{entry} row {r} (mean~{b}, cond {cond:.0}): max_abs {:.3e} > \
                         {tol:.3e} at lane {} — the two-pass variance regressed",
                        gs.max_abs,
                        gs.at
                    );
                }
            }
            eprintln!(
                "[gate] layernorm_w* holds the f32 error floor at W = {:?}",
                super::MW_WIDTHS
            );
        });
    }

    /// **SM-fill sweep** (`#[ignore]`, run by hand). Same-run A/B of the shipped one-warp launch
    /// against `layernorm_w{W}` at every generated width, over the shapes the planner distinguishes.
    /// This is how [`super::warps_per_row`]'s regime split and [`super::MIN_ELEMS_PER_LANE`] were
    /// calibrated, and re-running it is the first thing to do on a datacenter part.
    ///
    /// **The protocol is the load-bearing part**, and it is the second one written here — the first
    /// measured all of arm A, then all of arm B, and produced a round in which the *shipped baseline
    /// itself* moved 28% between shapes (281 us -> 219 us) as the clocks ramped, turning a 1.00x into
    /// anything between 0.48x and 1.57x depending on launch order. What is here now:
    ///
    /// * **Round-robin, not arm-at-a-time.** Every rep runs all arms once, in order, so a clock ramp
    ///   or a thermal step lands on every arm equally instead of on whichever ran first.
    /// * **A twin control.** The last arm is a byte-identical repeat of the shipped arm. `C(twin)`
    ///   must come back ~1.00x; if it does not, the round measured the machine and not the kernel,
    ///   and every other ratio in that line is void. (This is the control that caught an unpinned CPU
    ///   harness publishing an identical binary as "1.31x faster" — see the repo's bench notes.)
    /// * **Best-of-N minima**, three warm-up rounds before any timing, and ratios only. A lone
    ///   timing is not a measurement and this laptop's power state is part of the instrument.
    #[test]
    #[ignore = "perf sweep; run explicitly"]
    fn sweep_norm_sm_fill() {
        with_gpu("sweep_norm_sm_fill", |g| {
            let sm = g.sm_count();
            let eps = 1e-5f32;
            let reps = 50u32;
            eprintln!(
                "device: {} ({} SMs)  reps={reps}  [round-robin arms, best-of-N minima, C(twin) control]",
                g.device_name(),
                sm
            );
            // The one-off cost of the 12 specialized entries, paid once per process by the first
            // caller of the "norm" key. Warm on a second run in the same cubin cache.
            let t_load = std::time::Instant::now();
            g.function("norm", super::norm_ptx(), "layernorm")
                .expect("load");
            eprintln!(
                "norm module: {} bytes of PTX, loaded in {:.2} ms",
                super::norm_ptx().len(),
                t_load.elapsed().as_secs_f64() * 1e3
            );
            // Underfill first (rows < 4*SM on a 20-SM part), sweeping the row length so the sweep
            // calibrates MIN_ELEMS_PER_LANE and not just the regime split; then the covered regime.
            let shapes: &[(usize, usize)] = &[
                (8, 4096),
                (32, 256),
                (32, 512),
                (32, 768),
                (32, 1024),
                (32, 2048),
                (32, 4096),
                (64, 8192),
                (128, 1024),
                (512, 1024),
                (4096, 1024),
                (4096, 4096),
                (16384, 768),
            ];
            for &(rows, cols) in shapes {
                let mut rng = crate::diff::Rng::new(0xF11 ^ ((rows as u64) << 20) ^ (cols as u64));
                let x: Vec<f32> = (0..rows * cols).map(|_| rng.f32_range(-3.0, 3.0)).collect();
                let plan = super::norm_launch("layernorm", sm, rows, cols);
                let capped = (rows as u32).min(32 * sm as u32).max(1);
                // (label, entry, grid, block); arm 0 and the last arm are the same launch twice.
                let mut arms: Vec<(String, String, u32, u32)> =
                    vec![("shipped".into(), "layernorm".into(), rows as u32, 32)];
                for w in super::MW_WIDTHS {
                    arms.push((
                        format!("W{w}"),
                        super::mw_name("layernorm", w),
                        capped,
                        32 * w,
                    ));
                }
                arms.push(("C(twin)".into(), "layernorm".into(), rows as u32, 32));

                let funcs: Vec<_> = arms
                    .iter()
                    .map(|(_, e, _, _)| {
                        g.function("norm", super::norm_ptx(), e.as_str())
                            .expect("fn")
                    })
                    .collect();
                let x_d = g.stream.memcpy_stod(&x).expect("upload");
                let mut out_d = g.stream.alloc_zeros::<f32>(x.len()).expect("alloc");
                let (r, c) = (rows as u32, cols as u32);
                let cfgs: Vec<LaunchConfig> = arms
                    .iter()
                    .map(|&(_, _, grid, block)| LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .collect();
                let mut best = vec![f64::INFINITY; arms.len()];
                for round in 0..(reps + 3) {
                    for i in 0..arms.len() {
                        let t0 = std::time::Instant::now();
                        {
                            let mut b = g.stream.launch_builder(&funcs[i]);
                            b.arg(&r).arg(&c).arg(&eps).arg(&x_d).arg(&mut out_d);
                            unsafe { b.launch(cfgs[i]) }.expect("launch");
                        }
                        g.stream.synchronize().expect("sync");
                        if round >= 3 {
                            best[i] = best[i].min(t0.elapsed().as_secs_f64());
                        }
                    }
                }
                let base = best[0];
                let bytes = (x.len() * 4 * 2) as f64; // one read pass + one write
                let mut line = format!(
                    "rows={rows:<6} cols={cols:<6} shipped={:>8.1}us {:>6.1}GB/s |",
                    base * 1e6,
                    bytes / base / 1e9
                );
                for (i, (label, ..)) in arms.iter().enumerate().skip(1) {
                    line += &format!(" {label}={:>5.2}x", base / best[i]);
                }
                eprintln!("{line}  planner -> {} W{}", plan.entry, plan.warps_per_row);
            }
        });
    }
}
