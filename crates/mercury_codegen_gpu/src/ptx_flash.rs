//! Fused **flash-attention** on the GPU — the kernel that *lost* on CPU (≈2× slower than the
//! GEMM-dispatch path on AVX2, because materialized attention is compute-bound on the tuned GEMM and
//! the S² scores traffic isn't the bottleneck) but is GPU-shaped: it never materializes S = Q·Kᵀ in
//! HBM, using the online-softmax recurrence to stream K/V once.
//!
//! Layout: single head, Q/K/V/O are `[S, D]` row-major, `D` a multiple of 32. **One warp per query
//! row**; the 32 lanes split the head dim (lane `t` owns d ∈ {t, t+32, …}, i.e. `R = D/32` values).
//! For each key j: each lane computes its partial of Q[i]·K[j], a warp butterfly all-reduce gives
//! the full score, then the online-softmax update rescales the running denominator `l` and the
//! per-lane output accumulators `acc[r]`. `R` is unrolled at PTX-gen time (acc/q in registers), so a
//! kernel is generated per supported D. exp uses `ex2.approx`; tolerance-gated vs a CPU f64 reference.
//!
//! **Two kernels** (`gpu::flash_plan` selects; see [`FLASH_TILE_MIN`]):
//!
//!  * `flash_d{D}_t` — *key-block tiled*, **the default at every S**. [`FLASH_TWARPS`]=8 query-row warps
//!    per CTA **cooperatively stage a block of `BK = 1024/D` keys** (K and V) into shared memory once,
//!    then each warp scores its row against the resident block. Each K/V block is read from L2 once per
//!    CTA and reused by all 8 rows, cutting K/V global traffic `2·S²·D → 2·S²·D/8`. `BK·D = 1024` for
//!    every supported D, so the staging buffer is a uniform **8 KB** (`smemK` at byte 0, `smemV` at byte
//!    4096). The same-process A/B (`flash_tiled_vs_untiled`) shows this wins or ties at all measured S —
//!    the 8× traffic cut and the coalesced cooperative loads beat per-row streaming even at short S,
//!    where the staging cost (a cooperative load + two `bar.sync`s per block) was expected to dominate.
//!  * `flash_d{D}` — *untiled* baseline. [`FLASH_WARPS`]=2 query-row warps per CTA, each independently
//!    streaming all of K/V from L2 (2 warps/CTA just fills past the blocks-per-SM cap). Retained as the
//!    A/B reference and the gate's cross-check; not used in production while the crossover is 0.
//!
//! The per-key arithmetic and the ascending key order are **identical** between the two kernels, so for
//! any S they produce **bit-identical** output — tiling is a pure data-movement optimisation, and the
//! tolerance gate (which runs *both*) confirms the SMEM plumbing is correct.
//!
//! **No-deadlock (tiled).** Threads are `32·FLASH_TWARPS` per CTA, `row = ctaid·W + warpId`. The
//! cooperative load and the two per-block `bar.sync`s run on *every* thread; only the per-row score/
//! store is predicated on `row < S`. So the warps of a ragged final CTA (when `FLASH_TWARPS ∤ S`) still
//! help stage K/V and still reach both barriers — they just skip their own compute — so the barriers
//! can never deadlock. Shared memory is static, so the launch config reserves no dynamic SMEM.

use std::sync::OnceLock;

/// Query-row warps per CTA for the **untiled** kernel (`flash_d{D}`). W=2 on Ada (sm_89): a 64-thread
/// CTA already breaks the ~24-blocks-per-SM cap that limited one-warp CTAs to ~half occupancy (2 warps
/// × 24 blocks = the full 48 warps/SM) while keeping the grid as fine as possible. (Total warps in
/// flight = S regardless of W — packing changes CTA count, not the warp supply.)
pub const FLASH_WARPS: u32 = 2;

/// Query-row warps per CTA for the **tiled** kernel (`flash_d{D}_t`) — *also the K/V SMEM reuse factor*
/// (each staged key block is read from L2 once and reused by this many rows). W=8: a 256-thread CTA
/// gives an 8× L2-traffic cut, the long-sequence lever, while 8 KB of static SMEM/CTA stays small
/// enough for high occupancy.
pub const FLASH_TWARPS: u32 = 8;

/// Sequence-length crossover: `S >= FLASH_TILE_MIN` dispatches the tiled kernel, else the untiled one.
/// The same-process A/B bench `flash_tiled_vs_untiled` (the only honest flash comparison — a cross-*run*
/// one is corrupted by the ~7× laptop-GPU clock swing) found the **tiled kernel wins or ties at every
/// measured S** on the RTX 4050: `tiled/untiled` ≈ 0.56× @256, 0.90× @512, 0.73× @1024, 0.80× @2048,
/// and 1.02× (noise) @4096 — its 8× L2-traffic cut and coalesced cooperative loads beat the untiled
/// per-row streaming even at short S, where the staging overhead was expected to dominate but doesn't.
/// So the crossover is **0**: tiled is always selected. The untiled kernel and this constant are kept
/// only to document the comparison and to localize a change should a future GPU show a high-S untiled
/// regime. See `gpu::flash_plan`.
pub const FLASH_TILE_MIN: usize = 0;

/// Generate the **untiled** flash kernel for head dim `d` (multiple of 32). Name = `flash_d{d}`.
/// `warps` independent query-row warps per CTA (occupancy packing; the launcher must match).
fn entry_untiled(d: usize, warps: u32) -> String {
    assert!(d % 32 == 0, "D must be a multiple of 32");
    let r = d / 32; // values per lane
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = format!("flash_d{d}");

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0;\n";
    // scalars + per-lane register arrays
    let mut fregs = String::from("%scale,%m,%l,%s,%sfull,%mnew,%corr,%pp,%partial,%vv,%kv,%rt");
    for i in 0..r {
        fregs += &format!(",%q{i},%acc{i}");
    }
    s += &format!("    .reg .f32 {fregs};\n");
    s += "    .reg .b32 %S,%row,%lane,%warpid,%j,%tmp;\n";
    s += "    .reg .b64 %Q,%K,%V,%O,%qbase,%kbase,%vbase,%obase,%laneoff,%off;\n";

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    // row = ctaid.x*W + warpId (W independent rows per CTA); lane = tid.x & 31.
    s += &format!(
        "    mov.u32 %tmp,%tid.x;\n    shr.u32 %warpid,%tmp,5;\n    and.b32 %lane,%tmp,31;\n    mov.u32 %row,%ctaid.x;\n    mul.lo.u32 %row,%row,{warps};\n    add.u32 %row,%row,%warpid;\n    setp.ge.u32 %p0,%row,%S;\n    @%p0 bra RET_{name};\n"
    );
    s += "    mul.wide.u32 %laneoff,%lane,4;\n";
    // qbase = Q + row*D*4 ; preload q[r]
    s += &format!("    mul.lo.s32 %tmp,%row,{d};\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %qbase,%Q,%off;\n    add.s64 %qbase,%qbase,%laneoff;\n");
    for i in 0..r {
        s += &format!("    ld.global.f32 %q{i},[%qbase+{}];\n", i * 128);
    }
    // obase = O + row*D*4
    s += &format!("    mul.lo.s32 %tmp,%row,{d};\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %obase,%O,%off;\n    add.s64 %obase,%obase,%laneoff;\n");
    // init running state
    s += "    mov.f32 %m,0fFF800000;\n    mov.f32 %l,0f00000000;\n";
    for i in 0..r {
        s += &format!("    mov.f32 %acc{i},0f00000000;\n");
    }

    // for j in 0..S
    s += "    mov.u32 %j,0;\n";
    s += &format!("J_{name}:\n    setp.ge.u32 %p0,%j,%S;\n    @%p0 bra DONE_{name};\n");
    // kbase = K + j*D*4 + laneoff
    s += &format!("    mul.lo.s32 %tmp,%j,{d};\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %kbase,%K,%off;\n    add.s64 %kbase,%kbase,%laneoff;\n");
    // partial = sum_r q[r]*K[j][lane+32r]
    s += "    mov.f32 %partial,0f00000000;\n";
    for i in 0..r {
        s += &format!(
            "    ld.global.f32 %kv,[%kbase+{}];\n    fma.rn.f32 %partial,%q{i},%kv,%partial;\n",
            i * 128
        );
    }
    // warp all-reduce add -> sfull
    s += "    mov.f32 %sfull,%partial;\n";
    for off in [16, 8, 4, 2, 1] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%sfull,{off},0x1f,0xffffffff;\n    add.f32 %sfull,%sfull,%rt;\n");
    }
    // s = sfull*scale ; online softmax update
    s += "    mul.f32 %s,%sfull,%scale;\n";
    s += "    max.f32 %mnew,%m,%s;\n";
    // corr = exp(m - mnew) ; pp = exp(s - mnew)
    s += &format!("    sub.f32 %corr,%m,%mnew;\n    mul.f32 %corr,%corr,{log2e};\n    ex2.approx.f32 %corr,%corr;\n");
    s += &format!(
        "    sub.f32 %pp,%s,%mnew;\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 %pp,%pp;\n"
    );
    // l = l*corr + pp
    s += "    fma.rn.f32 %l,%l,%corr,%pp;\n";
    // vbase = V + j*D*4 + laneoff ; acc[r] = acc[r]*corr + pp*V[j][lane+32r]
    s += &format!("    mul.lo.s32 %tmp,%j,{d};\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %vbase,%V,%off;\n    add.s64 %vbase,%vbase,%laneoff;\n");
    for i in 0..r {
        s += &format!("    ld.global.f32 %vv,[%vbase+{}];\n    mul.f32 %acc{i},%acc{i},%corr;\n    fma.rn.f32 %acc{i},%pp,%vv,%acc{i};\n", i * 128);
    }
    s += "    mov.f32 %m,%mnew;\n";
    s += &format!("    add.u32 %j,%j,1;\n    bra J_{name};\n");

    // O[row][lane+32r] = acc[r] / l
    s += &format!("DONE_{name}:\n");
    for i in 0..r {
        s += &format!(
            "    div.rn.f32 %acc{i},%acc{i},%l;\n    st.global.f32 [%obase+{}],%acc{i};\n",
            i * 128
        );
    }
    s += &format!("RET_{name}:\n    ret;\n}}\n");
    s
}

/// Generate the **key-block-tiled** flash kernel for head dim `d` (multiple of 32, must divide 1024).
/// Name = `flash_d{d}_t`. `warps` query-row warps per CTA cooperatively stage a `BK = 1024/D` key block
/// into shared memory and reuse it (the launcher must match the block dim `32·warps`).
fn entry_tiled(d: usize, warps: u32) -> String {
    assert!(d % 32 == 0, "D must be a multiple of 32");
    assert!(1024 % d == 0, "D must divide 1024 (key-block tiling: BK = 1024/D)");
    let r = d / 32; // values per lane
    let bk = 1024 / d; // keys per shared-memory block (BK·D = 1024 ⇒ 8 KB SMEM, uniform across D)
    let nthreads = 32 * warps; // threads per CTA
    let smem_v_off = bk * d * 4; // byte offset of smemV within the staging buffer (= 4096, uniform)
    let smem_bytes = 2 * bk * d * 4; // total staging bytes (= 8192, uniform across supported D)
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = format!("flash_d{d}_t");

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0,%active;\n";
    let mut fregs =
        String::from("%scale,%m,%l,%s,%sfull,%mnew,%corr,%pp,%partial,%vv,%kv,%rt,%tf0,%tf1");
    for i in 0..r {
        fregs += &format!(",%q{i},%acc{i}");
    }
    s += &format!("    .reg .f32 {fregs};\n");
    s += "    .reg .b32 %S,%row,%lane,%warpid,%kblock,%jj,%kb,%kbD,%kblockD,%idx,%nt,%tmp,%sa,%sa2,%gidx;\n";
    s += "    .reg .b64 %Q,%K,%V,%O,%qbase,%obase,%laneoff,%off,%ga,%ga2,%goff;\n";
    // staging buffer: smemK at [0, 4096), smemV at [4096, 8192) — uniform 8 KB for every supported D.
    s += &format!("    .shared .align 16 .b8 smem_{name}[{smem_bytes}];\n");

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    // row = ctaid.x*W + warpId (W query rows per CTA); lane = tid.x & 31. active = row < S.
    s += &format!(
        "    mov.u32 %tmp,%tid.x;\n    shr.u32 %warpid,%tmp,5;\n    and.b32 %lane,%tmp,31;\n    mov.u32 %row,%ctaid.x;\n    mul.lo.u32 %row,%row,{warps};\n    add.u32 %row,%row,%warpid;\n    setp.lt.u32 %active,%row,%S;\n    mov.u32 %nt,{nthreads};\n"
    );
    s += "    mul.wide.u32 %laneoff,%lane,4;\n";
    // qbase = Q + row*D*4 + laneoff ; preload q[r] only if this warp owns a real row
    s += &format!("    mul.lo.s32 %tmp,%row,{d};\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %qbase,%Q,%off;\n    add.s64 %qbase,%qbase,%laneoff;\n");
    s += &format!("    @!%active bra SKIPQ_{name};\n");
    for i in 0..r {
        s += &format!("    ld.global.f32 %q{i},[%qbase+{}];\n", i * 128);
    }
    s += &format!("SKIPQ_{name}:\n");
    // obase = O + row*D*4 + laneoff
    s += &format!("    mul.lo.s32 %tmp,%row,{d};\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %obase,%O,%off;\n    add.s64 %obase,%obase,%laneoff;\n");
    // init running state
    s += "    mov.f32 %m,0fFF800000;\n    mov.f32 %l,0f00000000;\n";
    for i in 0..r {
        s += &format!("    mov.f32 %acc{i},0f00000000;\n");
    }

    // for kblock in 0..S step BK
    s += "    mov.u32 %kblock,0;\n";
    s += &format!("KB_{name}:\n    setp.ge.u32 %p0,%kblock,%S;\n    @%p0 bra KBDONE_{name};\n");
    // kb = min(BK, S - kblock) ; kbD = kb*D ; kblockD = kblock*D
    s += &format!("    sub.u32 %kb,%S,%kblock;\n    min.u32 %kb,%kb,{bk};\n    mul.lo.u32 %kbD,%kb,{d};\n    mul.lo.u32 %kblockD,%kblock,{d};\n");
    // cooperative load: smemK[idx]=K[kblockD+idx], smemV[idx]=V[kblockD+idx], for idx=tid; idx<kbD; idx+=nt
    s += "    mov.u32 %idx,%tid.x;\n";
    s += &format!("LD_{name}:\n    setp.ge.u32 %p0,%idx,%kbD;\n    @%p0 bra LDDONE_{name};\n");
    s += "    add.u32 %gidx,%kblockD,%idx;\n    mul.wide.u32 %goff,%gidx,4;\n";
    s += &format!("    add.s64 %ga,%K,%goff;\n    ld.global.f32 %tf0,[%ga];\n    mov.u32 %sa,smem_{name};\n    mad.lo.u32 %sa,%idx,4,%sa;\n    st.shared.f32 [%sa],%tf0;\n");
    s += &format!("    add.s64 %ga2,%V,%goff;\n    ld.global.f32 %tf1,[%ga2];\n    add.u32 %sa2,%sa,{smem_v_off};\n    st.shared.f32 [%sa2],%tf1;\n");
    s += &format!("    add.u32 %idx,%idx,%nt;\n    bra LD_{name};\n");
    s += &format!("LDDONE_{name}:\n    bar.sync 0;\n");
    // compute: each active warp scores its row against the kb staged keys (online softmax)
    s += &format!("    @!%active bra AFTER_{name};\n");
    s += &format!("    mov.u32 %jj,0;\nJJ_{name}:\n    setp.ge.u32 %p0,%jj,%kb;\n    @%p0 bra AFTER_{name};\n");
    // sa = &smemK[jj*D + lane]  (then +128*r reaches lane+32r) ; partial = Σ_r q[r]·smemK[jj*D+lane+32r]
    s += &format!("    mul.lo.u32 %tmp,%jj,{d};\n    add.u32 %tmp,%tmp,%lane;\n    shl.b32 %tmp,%tmp,2;\n    mov.u32 %sa,smem_{name};\n    add.u32 %sa,%sa,%tmp;\n");
    s += "    mov.f32 %partial,0f00000000;\n";
    for i in 0..r {
        s += &format!(
            "    ld.shared.f32 %kv,[%sa+{}];\n    fma.rn.f32 %partial,%q{i},%kv,%partial;\n",
            i * 128
        );
    }
    // warp all-reduce add -> sfull
    s += "    mov.f32 %sfull,%partial;\n";
    for off in [16, 8, 4, 2, 1] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%sfull,{off},0x1f,0xffffffff;\n    add.f32 %sfull,%sfull,%rt;\n");
    }
    // s = sfull*scale ; online softmax update
    s += "    mul.f32 %s,%sfull,%scale;\n";
    s += "    max.f32 %mnew,%m,%s;\n";
    // corr = exp(m - mnew) ; pp = exp(s - mnew)
    s += &format!("    sub.f32 %corr,%m,%mnew;\n    mul.f32 %corr,%corr,{log2e};\n    ex2.approx.f32 %corr,%corr;\n");
    s += &format!(
        "    sub.f32 %pp,%s,%mnew;\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 %pp,%pp;\n"
    );
    // l = l*corr + pp
    s += "    fma.rn.f32 %l,%l,%corr,%pp;\n";
    // sa2 = &smemV[jj*D+lane] = sa + 4096 ; acc[r] = acc[r]*corr + pp*smemV[jj*D+lane+32r]
    s += &format!("    add.u32 %sa2,%sa,{smem_v_off};\n");
    for i in 0..r {
        s += &format!("    ld.shared.f32 %vv,[%sa2+{}];\n    mul.f32 %acc{i},%acc{i},%corr;\n    fma.rn.f32 %acc{i},%pp,%vv,%acc{i};\n", i * 128);
    }
    s += "    mov.f32 %m,%mnew;\n";
    s += &format!("    add.u32 %jj,%jj,1;\n    bra JJ_{name};\n");
    // barrier: every thread (active rows post-compute, inactive rows straight here) before next overwrite
    s += &format!("AFTER_{name}:\n    bar.sync 0;\n    add.u32 %kblock,%kblock,{bk};\n    bra KB_{name};\n");

    // O[row][lane+32r] = acc[r] / l   (only real rows store)
    s += &format!("KBDONE_{name}:\n    @!%active bra RET_{name};\n");
    for i in 0..r {
        s += &format!(
            "    div.rn.f32 %acc{i},%acc{i},%l;\n    st.global.f32 [%obase+{}],%acc{i};\n",
            i * 128
        );
    }
    s += &format!("RET_{name}:\n    ret;\n}}\n");
    s
}

/// Comma-joined `{%pfx0,…,%pfx{n-1}}` WMMA fragment register vector.
fn frag(pfx: &str, n: usize) -> String {
    let r: Vec<String> = (0..n).map(|i| format!("%{pfx}{i}")).collect();
    format!("{{{}}}", r.join(","))
}

/// Generate the **tensor-core (WMMA) flash** kernel for head dim `d` (multiple of 16). Name =
/// `flash_d{d}_w`. **EXPERIMENT** — the online-softmax attention with its two matmuls (`Q·Kᵀ` and
/// `P·V`) on the Ada tensor cores (f16 in, f32 accumulate), vs the hand `fma`+shuffle `_t`/untiled
/// kernels. ONE warp handles a 16-query-row block (`row = ctaid·16 + 0..15`); requires `S % 16 == 0`
/// (no ragged key tail). Inputs Q/K/V are **f16** (the tensor-core dtype), O is f32.
///
/// The WMMA accumulator fragment→(row,col) map is opaque, so every fragment is immediately
/// `wmma.store.d`'d to shared memory and the per-row work (softmax, the running-max rescale of O) is
/// done with an explicit `lane==row` mapping in SMEM — the same trick the GEMM bias epilogue uses. Per
/// 16-key block: (1) `S = Q·Kᵀ` (nt WMMA, f32 acc) → `smemS`; (2) online softmax in `smemS` (lane owns
/// row=lane, `m`/`l` in registers) writing `smemP` (f16) + a per-row correction `corr`; (3) `O += P·V`
/// (nn WMMA) → `smemPV`, then `smemO = smemO·corr + smemPV`. Final `O = smemO / l`. Tolerance-gated;
/// f16 inputs mean a looser tol than the f32 kernels (consistent with the fp16 WMMA layer).
fn entry_wmma(d: usize) -> String {
    assert!(d % 16 == 0, "WMMA flash needs D % 16 == 0");
    let kt = d / 16; // Q·Kᵀ contraction tiles (over the head dim) AND P·V output n-tiles (over D)
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = format!("flash_d{d}_w");
    // SMEM layout (one warp, Br=16): smemS[16·16] f32 | smemP[16·16] f16 | smemPV[16·D] f32 | smemO[16·D] f32
    let off_s = 0usize;
    let off_p = off_s + 16 * 16 * 4;
    let off_pv = off_p + 16 * 16 * 2;
    let off_o = off_pv + 16 * d * 4;
    let smem_bytes = off_o + 16 * d * 4;

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0;\n";
    // f32: scalars + S accumulator (8) + PV accumulators (kt tiles × 8)
    let mut fr = String::from("%scale,%mlane,%llane,%corr,%rmax,%mnew,%scv,%pp,%psum,%ov,%pvv");
    for r in 0..8 {
        fr += &format!(",%s{r}");
    }
    for n in 0..kt {
        for r in 0..8 {
            fr += &format!(",%pv{n}_{r}");
        }
    }
    s += &format!("    .reg .f32 {fr};\n");
    // b32: WMMA fragments (qa: kt×8, kb: kt×8, pa: 8, vb: kt×8) + scalars
    let mut br = String::new();
    for n in 0..kt {
        for r in 0..8 {
            br += &format!("%qa{n}_{r},%kb{n}_{r},%vb{n}_{r},");
        }
    }
    for r in 0..8 {
        br += &format!("%pa_{r},");
    }
    s += &format!(
        "    .reg .b32 {}%S,%lane,%row,%kb,%c,%sa,%tmp,%tmp2,%pf,%st64,%st16;\n",
        br
    );
    s += "    .reg .b64 %Q,%K,%V,%O,%qap,%kbp,%vbp,%gp,%off,%optr;\n";
    s += &format!("    .shared .align 16 .b8 smem_{name}[{smem_bytes}];\n");

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    // wmma.load/.store strides must be REGISTERS, not immediates (an immediate JITs but mis-addresses).
    s += &format!("    mov.u32 %st64,{d};\n    mov.u32 %st16,16;\n");
    s += "    mov.u32 %tmp,%tid.x;\n    and.b32 %lane,%tmp,31;\n    mov.u32 %tmp,%ctaid.x;\n    shl.b32 %row,%tmp,4;\n"; // row = ctaid*16

    // Load Q (A, f16, .row) once: kt k-tiles at Q + (row*D + 16k)*2, stride D.
    for n in 0..kt {
        s += &format!("    mul.lo.s32 %tmp,%row,{d};\n    add.u32 %tmp,%tmp,{};\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %qap,%Q,%off;\n", n * 16);
        s += &format!(
            "    wmma.load.a.sync.aligned.m16n16k16.row.f16 {}, [%qap], %st64;\n",
            frag(&format!("qa{n}_"), 8)
        );
    }
    // m=-inf, l=0 (per-lane regs, harmless on all lanes); smemO[lane][0..D]=0 ONLY on lanes 0..15
    // (lane==row, only 16 rows of smemO exist — an unguarded init would write past SMEM on lanes 16..31).
    s += "    mov.f32 %mlane,0fFF800000;\n    mov.f32 %llane,0f00000000;\n";
    s += &format!("    setp.ge.u32 %p0,%lane,16;\n    @%p0 bra INITDONE_{name};\n");
    s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{off_o};\n    mul.lo.s32 %tmp2,%lane,{};\n    add.u32 %tmp,%tmp,%tmp2;\n", d * 4);
    for c in 0..d {
        s += &format!("    st.shared.f32 [%tmp+{}],0f00000000;\n", c * 4);
    }
    s += &format!("INITDONE_{name}:\n");

    // for kb in 0..S step 16
    s += "    mov.u32 %kb,0;\n";
    s += &format!("KB_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra KBDONE_{name};\n");

    // S = Q · K[kb..]ᵀ  (nt WMMA): load K (B, .col) kt tiles, accumulate kt mma into %s0..7
    for r in 0..8 {
        s += &format!("    mov.f32 %s{r},0f00000000;\n");
    }
    for n in 0..kt {
        s += &format!("    mul.lo.s32 %tmp,%kb,{d};\n    add.u32 %tmp,%tmp,{};\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %kbp,%K,%off;\n", n * 16);
        s += &format!(
            "    wmma.load.b.sync.aligned.m16n16k16.col.f16 {}, [%kbp], %st64;\n",
            frag(&format!("kb{n}_"), 8)
        );
    }
    for n in 0..kt {
        s += &format!(
            "    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {}, {}, {}, {};\n",
            frag("s", 8),
            frag(&format!("qa{n}_"), 8),
            frag(&format!("kb{n}_"), 8),
            frag("s", 8)
        );
    }
    // store S → smemS (f32, stride 16)
    s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{off_s};\n    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%gp], {}, %st16;\n", frag("s", 8));
    s += "    bar.sync 0;\n";

    // online softmax over smemS row (lane==row, lanes 0..15): rmax→m→corr→P(f16)→l
    s += &format!("    setp.ge.u32 %p0,%lane,16;\n    @%p0 bra SOFTDONE_{name};\n");
    s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{off_s};\n    mul.lo.s32 %tmp2,%lane,64;\n    add.u32 %sa,%tmp,%tmp2;\n"); // %sa := &smemS[lane][0] (row stride 16 f32 = 64 B)
    // rmax = max_c scale*smemS[lane][c]
    s += "    mov.f32 %rmax,0fFF800000;\n";
    for c in 0..16 {
        s += &format!("    ld.shared.f32 %scv,[%sa+{}];\n    mul.f32 %scv,%scv,%scale;\n    max.f32 %rmax,%rmax,%scv;\n", c * 4);
    }
    s += "    max.f32 %mnew,%mlane,%rmax;\n";
    s += &format!("    sub.f32 %corr,%mlane,%mnew;\n    mul.f32 %corr,%corr,{log2e};\n    ex2.approx.f32 %corr,%corr;\n");
    s += "    mul.f32 %llane,%llane,%corr;\n    mov.f32 %psum,0f00000000;\n";
    // P[c] = exp(scale*S - mnew) → smemP (f16) ; psum += P
    s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{off_p};\n    mul.lo.s32 %tmp2,%lane,32;\n    add.u32 %pf,%tmp,%tmp2;\n"); // &smemP[lane][0] (16 f16 = 32 B)
    for c in 0..16 {
        s += &format!("    ld.shared.f32 %scv,[%sa+{}];\n    mul.f32 %scv,%scv,%scale;\n    sub.f32 %pp,%scv,%mnew;\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 %pp,%pp;\n    add.f32 %psum,%psum,%pp;\n    cvt.rn.f16.f32 %tmp2,%pp;\n    st.shared.b16 [%pf+{}],%tmp2;\n", c * 4, c * 2);
    }
    s += "    add.f32 %llane,%llane,%psum;\n    mov.f32 %mlane,%mnew;\n";
    s += &format!("SOFTDONE_{name}:\n    bar.sync 0;\n");

    // O += P·V  (nn WMMA): load P (A, .row, from smemP) + V (B, .row) kt n-tiles, mma into pv, store smemPV
    s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{off_p};\n    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n    wmma.load.a.sync.aligned.m16n16k16.row.f16 {}, [%gp], %st16;\n", frag("pa_", 8));
    for n in 0..kt {
        s += &format!("    mul.lo.s32 %tmp,%kb,{d};\n    add.u32 %tmp,%tmp,{};\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %vbp,%V,%off;\n", n * 16);
        s += &format!(
            "    wmma.load.b.sync.aligned.m16n16k16.row.f16 {}, [%vbp], %st64;\n",
            frag(&format!("vb{n}_"), 8)
        );
        for r in 0..8 {
            s += &format!("    mov.f32 %pv{n}_{r},0f00000000;\n");
        }
        s += &format!(
            "    wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {}, {}, {}, {};\n",
            frag(&format!("pv{n}_"), 8),
            frag("pa_", 8),
            frag(&format!("vb{n}_"), 8),
            frag(&format!("pv{n}_"), 8)
        );
        // store this n-tile → smemPV at col 16n (stride D)
        s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{};\n    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%gp], {}, %st64;\n", off_pv + n * 16 * 4, frag(&format!("pv{n}_"), 8));
    }
    s += "    bar.sync 0;\n";
    // smemO[lane][c] = smemO[lane][c]*corr + smemPV[lane][c]  (lane==row, lanes 0..15)
    s += &format!("    setp.ge.u32 %p0,%lane,16;\n    @%p0 bra RESDONE_{name};\n");
    s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{off_pv};\n    mul.lo.s32 %tmp2,%lane,{};\n    add.u32 %tmp,%tmp,%tmp2;\n", d * 4); // &smemPV[lane]
    s += &format!("    mov.u32 %tmp2,smem_{name};\n    add.u32 %tmp2,%tmp2,{off_o};\n    mul.lo.s32 %sa,%lane,{};\n    add.u32 %tmp2,%tmp2,%sa;\n", d * 4); // &smemO[lane]  (%sa scratch)
    for c in 0..d {
        s += &format!("    ld.shared.f32 %pvv,[%tmp+{0}];\n    ld.shared.f32 %ov,[%tmp2+{0}];\n    fma.rn.f32 %ov,%ov,%corr,%pvv;\n    st.shared.f32 [%tmp2+{0}],%ov;\n", c * 4);
    }
    s += &format!("RESDONE_{name}:\n    bar.sync 0;\n");

    s += &format!("    add.u32 %kb,%kb,16;\n    bra KB_{name};\n");

    // O[row][c] = smemO[lane][c] / l   (lane==row, lanes 0..15)
    s += &format!("KBDONE_{name}:\n    setp.ge.u32 %p0,%lane,16;\n    @%p0 bra RET_{name};\n");
    s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{off_o};\n    mul.lo.s32 %tmp2,%lane,{0};\n    add.u32 %tmp,%tmp,%tmp2;\n    add.u32 %c,%row,%lane;\n    mul.lo.s32 %c,%c,{d};\n    mul.wide.u32 %off,%c,4;\n    add.s64 %optr,%O,%off;\n", d * 4);
    for c in 0..d {
        s += &format!("    ld.shared.f32 %ov,[%tmp+{0}];\n    div.rn.f32 %ov,%ov,%llane;\n    st.global.f32 [%optr+{0}],%ov;\n", c * 4);
    }
    s += &format!("RET_{name}:\n    ret;\n}}\n");
    s
}

/// **Wide-key-tile** tensor-core flash (`flash_d{d}_w{nkb}`) — identical online-softmax math to
/// [`entry_wmma`] but processes `WK = 16·nkb` keys per softmax step instead of 16. The narrow kernel's
/// KB loop is a serial dependency chain of `S/16` iterations, each paying four `bar.sync`s and several
/// SMEM round-trips (store S → load for softmax → store P → store PV → accumulate O); under the kernel's
/// ~21% occupancy (1 warp/CTA, SMEM-capped) those per-iteration latencies are not hidden, so at long
/// context the kernel runs at a fraction of roofline (~0.67 TFLOP/s @ S=4096). Widening to `WK` keys cuts
/// the iteration count — and thus the round-trip count — by `nkb×`; the online softmax is associative
/// over any tile width so O is unchanged to f32 rounding. Cost: `smemS`/`smemP` grow `nkb×`, trimming
/// occupancy, so the net is an empirical A/B (`flash_tiled_vs_untiled`). Requires `S % WK == 0` (no
/// ragged key tail) — fine for the layer, whose WMMA path is already `S % 64 == 0`.
fn entry_wmma_wide(d: usize, nkb: usize) -> String {
    assert!(d % 16 == 0, "WMMA flash needs D % 16 == 0");
    assert!(nkb >= 1, "nkb must be >= 1");
    let kt = d / 16; // Q·Kᵀ contraction tiles (over the head dim) AND P·V output n-tiles (over D)
    let wk = 16 * nkb; // keys staged + softmaxed per KB step
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = format!("flash_d{d}_w{nkb}");
    // SMEM (one warp, Br=16): smemS[16·WK] f32 | smemP[16·WK] f16 | smemPV[16·D] f32 | smemO[16·D] f32
    let off_s = 0usize;
    let off_p = off_s + 16 * wk * 4;
    let off_pv = off_p + 16 * wk * 2;
    let off_o = off_pv + 16 * d * 4;
    let smem_bytes = off_o + 16 * d * 4;

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0;\n";
    let mut fr = String::from("%scale,%mlane,%llane,%corr,%rmax,%mnew,%scv,%pp,%psum,%ov,%pvv");
    for r in 0..8 {
        fr += &format!(",%s{r}");
    }
    for n in 0..kt {
        for r in 0..8 {
            fr += &format!(",%pv{n}_{r}");
        }
    }
    s += &format!("    .reg .f32 {fr};\n");
    // WMMA fragments: qa (kt×8, loaded once), kb (kt×8, reused per sub-tile), pa+vb (8 each, reused).
    let mut br = String::new();
    for n in 0..kt {
        for r in 0..8 {
            br += &format!("%qa{n}_{r},%kb{n}_{r},");
        }
    }
    for r in 0..8 {
        br += &format!("%pa_{r},%vb_{r},");
    }
    s += &format!(
        "    .reg .b32 {}%S,%lane,%row,%kb,%c,%sa,%tmp,%tmp2,%pf,%st64,%stwk;\n",
        br
    );
    s += "    .reg .b64 %Q,%K,%V,%O,%qap,%kbp,%vbp,%gp,%off,%optr;\n";
    s += &format!("    .shared .align 16 .b8 smem_{name}[{smem_bytes}];\n");

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    s += &format!("    mov.u32 %st64,{d};\n    mov.u32 %stwk,{wk};\n");
    s += "    mov.u32 %tmp,%tid.x;\n    and.b32 %lane,%tmp,31;\n    mov.u32 %tmp,%ctaid.x;\n    shl.b32 %row,%tmp,4;\n"; // row = ctaid*16

    // Load Q (A, f16, .row) once: kt k-tiles at Q + (row*D + 16k)*2, stride D.
    for n in 0..kt {
        s += &format!("    mul.lo.s32 %tmp,%row,{d};\n    add.u32 %tmp,%tmp,{};\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %qap,%Q,%off;\n", n * 16);
        s += &format!(
            "    wmma.load.a.sync.aligned.m16n16k16.row.f16 {}, [%qap], %st64;\n",
            frag(&format!("qa{n}_"), 8)
        );
    }
    s += "    mov.f32 %mlane,0fFF800000;\n    mov.f32 %llane,0f00000000;\n";
    s += &format!("    setp.ge.u32 %p0,%lane,16;\n    @%p0 bra INITDONE_{name};\n");
    s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{off_o};\n    mul.lo.s32 %tmp2,%lane,{};\n    add.u32 %tmp,%tmp,%tmp2;\n", d * 4);
    for c in 0..d {
        s += &format!("    st.shared.f32 [%tmp+{}],0f00000000;\n", c * 4);
    }
    s += &format!("INITDONE_{name}:\n");

    // for kb in 0..S step WK
    s += "    mov.u32 %kb,0;\n";
    s += &format!("KB_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra KBDONE_{name};\n");

    // For each 16-key sub-tile j: S_j = Q · K[kb+16j]ᵀ (kt mma) → smemS[:,16j] (stride WK).
    for j in 0..nkb {
        for r in 0..8 {
            s += &format!("    mov.f32 %s{r},0f00000000;\n");
        }
        for n in 0..kt {
            s += &format!("    add.u32 %tmp,%kb,{};\n    mul.lo.s32 %tmp,%tmp,{d};\n    add.u32 %tmp,%tmp,{};\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %kbp,%K,%off;\n", j * 16, n * 16);
            s += &format!(
                "    wmma.load.b.sync.aligned.m16n16k16.col.f16 {}, [%kbp], %st64;\n",
                frag(&format!("kb{n}_"), 8)
            );
        }
        for n in 0..kt {
            s += &format!(
                "    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {}, {}, {}, {};\n",
                frag("s", 8),
                frag(&format!("qa{n}_"), 8),
                frag(&format!("kb{n}_"), 8),
                frag("s", 8)
            );
        }
        s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{};\n    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%gp], {}, %stwk;\n", off_s + j * 16 * 4, frag("s", 8));
    }
    s += "    bar.sync 0;\n";

    // online softmax over the full WK-wide smemS row (lane==row, lanes 0..15)
    s += &format!("    setp.ge.u32 %p0,%lane,16;\n    @%p0 bra SOFTDONE_{name};\n");
    s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{off_s};\n    mul.lo.s32 %tmp2,%lane,{};\n    add.u32 %sa,%tmp,%tmp2;\n", wk * 4); // &smemS[lane][0]
    s += "    mov.f32 %rmax,0fFF800000;\n";
    for c in 0..wk {
        s += &format!("    ld.shared.f32 %scv,[%sa+{}];\n    mul.f32 %scv,%scv,%scale;\n    max.f32 %rmax,%rmax,%scv;\n", c * 4);
    }
    s += "    max.f32 %mnew,%mlane,%rmax;\n";
    s += &format!("    sub.f32 %corr,%mlane,%mnew;\n    mul.f32 %corr,%corr,{log2e};\n    ex2.approx.f32 %corr,%corr;\n");
    s += "    mul.f32 %llane,%llane,%corr;\n    mov.f32 %psum,0f00000000;\n";
    s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{off_p};\n    mul.lo.s32 %tmp2,%lane,{};\n    add.u32 %pf,%tmp,%tmp2;\n", wk * 2); // &smemP[lane][0]
    for c in 0..wk {
        s += &format!("    ld.shared.f32 %scv,[%sa+{}];\n    mul.f32 %scv,%scv,%scale;\n    sub.f32 %pp,%scv,%mnew;\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 %pp,%pp;\n    add.f32 %psum,%psum,%pp;\n    cvt.rn.f16.f32 %tmp2,%pp;\n    st.shared.b16 [%pf+{}],%tmp2;\n", c * 4, c * 2);
    }
    s += "    add.f32 %llane,%llane,%psum;\n    mov.f32 %mlane,%mnew;\n";
    s += &format!("SOFTDONE_{name}:\n    bar.sync 0;\n");

    // O += P·V : for each output d-tile n, accumulate nkb k-steps (P[:,16j]·V[kb+16j, 16n]) → smemPV[:,16n]
    for n in 0..kt {
        for r in 0..8 {
            s += &format!("    mov.f32 %pv{n}_{r},0f00000000;\n");
        }
    }
    for n in 0..kt {
        for j in 0..nkb {
            // pa = smemP[:,16j] (A,.row, base off_p+16j·2, stride WK)
            s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{};\n    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n    wmma.load.a.sync.aligned.m16n16k16.row.f16 {}, [%gp], %stwk;\n", off_p + j * 16 * 2, frag("pa_", 8));
            // vb = V[(kb+16j)·D + 16n] (B,.row, stride D)
            s += &format!("    add.u32 %tmp,%kb,{};\n    mul.lo.s32 %tmp,%tmp,{d};\n    add.u32 %tmp,%tmp,{};\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %vbp,%V,%off;\n", j * 16, n * 16);
            s += &format!(
                "    wmma.load.b.sync.aligned.m16n16k16.row.f16 {}, [%vbp], %st64;\n",
                frag("vb_", 8)
            );
            s += &format!(
                "    wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {}, {}, {}, {};\n",
                frag(&format!("pv{n}_"), 8),
                frag("pa_", 8),
                frag("vb_", 8),
                frag(&format!("pv{n}_"), 8)
            );
        }
        s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{};\n    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%gp], {}, %st64;\n", off_pv + n * 16 * 4, frag(&format!("pv{n}_"), 8));
    }
    s += "    bar.sync 0;\n";
    // smemO[lane][c] = smemO[lane][c]·corr + smemPV[lane][c] (lane==row, lanes 0..15)
    s += &format!("    setp.ge.u32 %p0,%lane,16;\n    @%p0 bra RESDONE_{name};\n");
    s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{off_pv};\n    mul.lo.s32 %tmp2,%lane,{};\n    add.u32 %tmp,%tmp,%tmp2;\n", d * 4);
    s += &format!("    mov.u32 %tmp2,smem_{name};\n    add.u32 %tmp2,%tmp2,{off_o};\n    mul.lo.s32 %sa,%lane,{};\n    add.u32 %tmp2,%tmp2,%sa;\n", d * 4);
    for c in 0..d {
        s += &format!("    ld.shared.f32 %pvv,[%tmp+{0}];\n    ld.shared.f32 %ov,[%tmp2+{0}];\n    fma.rn.f32 %ov,%ov,%corr,%pvv;\n    st.shared.f32 [%tmp2+{0}],%ov;\n", c * 4);
    }
    s += &format!("RESDONE_{name}:\n    bar.sync 0;\n");

    s += &format!("    add.u32 %kb,%kb,{wk};\n    bra KB_{name};\n");

    // O[row][c] = smemO[lane][c] / l (lane==row, lanes 0..15)
    s += &format!("KBDONE_{name}:\n    setp.ge.u32 %p0,%lane,16;\n    @%p0 bra RET_{name};\n");
    s += &format!("    mov.u32 %tmp,smem_{name};\n    add.u32 %tmp,%tmp,{off_o};\n    mul.lo.s32 %tmp2,%lane,{0};\n    add.u32 %tmp,%tmp,%tmp2;\n    add.u32 %c,%row,%lane;\n    mul.lo.s32 %c,%c,{d};\n    mul.wide.u32 %off,%c,4;\n    add.s64 %optr,%O,%off;\n", d * 4);
    for c in 0..d {
        s += &format!("    ld.shared.f32 %ov,[%tmp+{0}];\n    div.rn.f32 %ov,%ov,%llane;\n    st.global.f32 [%optr+{0}],%ov;\n", c * 4);
    }
    s += &format!("RET_{name}:\n    ret;\n}}\n");
    s
}

/// Generate the **register-resident `mma.sync` flash** kernel for head dim `d` (multiple of 16).
/// Name = `flash_d{d}_m` (or `flash_d{d}_mc` when `causal`). The FlashAttention-2 form on the Ada
/// tensor cores with the output `O`, the
/// running max `m`, and the denominator `l` held in **registers** across the whole K-loop — *no*
/// per-step SMEM round-trip of S/P/PV/O. The [`entry_wmma`]/[`entry_wmma_wide`] kernels `wmma.store.d`
/// every fragment to SMEM and keep `O` in SMEM (4 `bar.sync`s + several round-trips per key block),
/// which caps them at ~21% occupancy and serializes on the SMEM traffic; this kernel removes all of it.
///
/// Built on the hand-placed `mma.sync.m16n8k16.row.col.f32.f16.f16.f32` layout **proven** by
/// `gpu::tests::mma_m16n8k16_layout_verifies` (`grp=lane>>2`, `tg=lane&3`, `tg2=tg*2`). ONE warp owns a
/// 16-query-row block (`qrow0=ctaid.x*16+grp`, `qrow1=qrow0+8`); `S%16==0` (the layer is `%64`). Q/K/V
/// are f16 (the tensor-core dtype), O is f32. **Zero SMEM** in this first cut — K/V `mma` B-fragments are
/// loaded straight from global per 16-key block (uncoalesced; the coalescing SMEM stage is the perf
/// follow-up). The win here is purely the register-resident accumulator.
///
/// Per 16-key block (`kb`):
///  1. `S = Q·Kᵀ`: `kt=D/16` contraction tiles × 2 key n-tiles of `mma.sync` → 8 score scalars/lane in
///     the **D-accumulator layout** (lane owns rows {grp,grp+8} × keys {tg2,tg2+1, 8+tg2,8+tg2+1}). Q is
///     the A.row operand (loaded once into registers); K is the `.col` B operand — K's natural
///     `[key][hdim]` row-major *is* the required `[N=key][K=hdim]` layout, so each B-fragment is one
///     clean `b32` global load.
///  2. **Online softmax on the register scores**: per-row local max over the lane's 4 keys, then a
///     `shfl.sync.bfly` reduction across the 4 lanes sharing `grp` (offsets 1,2) for the true row max;
///     `corr=ex2((m-mnew)·log2e)`; **rescale the register O in place** (`o*=corr`); `l=l·corr+Σp`.
///  3. **P→A with no reformat**: the score D-layout *is* the A.row layout that `O+=P·V` needs (the two
///     key n-tiles' `d0..d3` map exactly onto `a0..a3`), so the softmax probabilities are just
///     `cvt.rn.f16.f32`+packed in registers into the P A-fragment — **no SMEM bounce**.
///  4. `O += P·V`: `D/8` output n-tiles of `mma.sync` accumulate directly into the register O. V is the
///     `.col` B operand `[hdim][key]` (the transpose of V's natural layout), so each B-fragment is two
///     `u16` loads packed — the only uncoalesced cost, removed by the SMEM stage later.
///  Final `O[row][c] = o / l_row`. Tolerance-gated vs `ref_attn` (f16 in ⇒ same ~2e-2 rel as the WMMA
///  flash; a mis-mapped fragment would scatter O(0.1+), which the gate catches).
///
/// **Causal** (`flash_d{d}_mc`): a decoder masks key `j > query i`. Two parts — (a) **skip** every key
/// block strictly above the query block's diagonal (the loop stops at `kb == row`, the ~½-work FA2 win
/// for long context), and (b) within the diagonal block set `s = -inf` for the per-lane scores whose key
/// exceeds their query (a `setp`/`selp` per score); the online softmax then drops them (`ex2(-inf)=0`).
/// Every query keeps its diagonal key (`key = query`), so `l > 0` always. Gated vs `ref_attn_causal`.
fn entry_mma_reg(d: usize, causal: bool) -> String {
    assert!(d % 16 == 0, "mma flash needs D % 16 == 0");
    let ktq = d / 16; // Q·Kᵀ contraction tiles (over hdim)
    let nto = d / 8; // P·V output n-tiles (over hdim)
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = if causal {
        format!("flash_d{d}_mc")
    } else {
        format!("flash_d{d}_m")
    };

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0;\n";
    // f32: scalars + 8 scores + 4 prob temps + O accumulators (nto×4)
    let mut fr = String::from(
        "%scale,%m0,%m1,%mnew0,%mnew1,%corr0,%corr1,%l0,%l1,%lmax0,%lmax1,%rt,%psum0,%psum1,%pp,%tp0,%tp1,%tp2,%tp3",
    );
    for nk in 0..2 {
        for r in 0..4 {
            fr += &format!(",%s{nk}_{r}");
        }
    }
    for nt in 0..nto {
        for r in 0..4 {
            fr += &format!(",%o{nt}_{r}");
        }
    }
    s += &format!("    .reg .f32 {fr};\n");
    // b32: Q A-fragments (ktq×4, loaded once) + P A-frag (4) + B-frag (2) + pack temps + indices
    let mut br = String::new();
    for kt in 0..ktq {
        for r in 0..4 {
            br += &format!("%qa{kt}_{r},");
        }
    }
    br += "%a0,%a1,%a2,%a3,%b0,%b1,%h0,%h1,";
    if causal {
        br += "%ck0,%ck1,%ck8,%ck9,";
    }
    s += &format!(
        "    .reg .b32 {br}%S,%lane,%grp,%tg,%tg2,%row,%qr0,%qr1,%kb,%gkey,%idx,%tmp,%hoff;\n"
    );
    s += "    .reg .b64 %Q,%K,%V,%O,%base,%off;\n";

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    // multi-head: head = ctaid.y, whose Q/K/V/O start at element ctaid.y·S·D ([H,S,D] layout). Fold the
    // head base into the (cvta'd) pointers so the per-row addressing below is unchanged. Single-head
    // callers launch grid.y=1 ⇒ ctaid.y=0 ⇒ headoff=0, so this is a no-op for them.
    s += &format!("    mov.u32 %hoff,%ctaid.y;\n    mul.lo.u32 %hoff,%hoff,%S;\n    mul.lo.u32 %hoff,%hoff,{d};\n");
    s += "    mul.wide.u32 %off,%hoff,2;\n    add.s64 %Q,%Q,%off;\n    add.s64 %K,%K,%off;\n    add.s64 %V,%V,%off;\n";
    s += "    mul.wide.u32 %off,%hoff,4;\n    add.s64 %O,%O,%off;\n";
    // lane decomposition + the two query rows this lane owns
    s += "    mov.u32 %lane,%tid.x;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg2,%tg,1;\n";
    s += "    mov.u32 %row,%ctaid.x;\n    shl.b32 %row,%row,4;\n    add.u32 %qr0,%row,%grp;\n    add.u32 %qr1,%qr0,8;\n";

    // Load Q A-fragments once (reused across every key block). qa{kt}_0=Q[qr0][16kt+tg2..],
    // _1=Q[qr1][16kt+tg2..], _2=Q[qr0][16kt+tg2+8..], _3=Q[qr1][16kt+tg2+8..]  (each a packed f16 pair).
    for kt in 0..ktq {
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_0,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_2,[%base];\n");
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_1,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_3,[%base];\n");
    }
    // init O=0, m=-inf, l=0
    for nt in 0..nto {
        for r in 0..4 {
            s += &format!("    mov.f32 %o{nt}_{r},0f00000000;\n");
        }
    }
    s += "    mov.f32 %m0,0fFF800000;\n    mov.f32 %m1,0fFF800000;\n    mov.f32 %l0,0f00000000;\n    mov.f32 %l1,0f00000000;\n";

    // for kb in 0..S step 16 (causal: stop at the diagonal block kb==row — skip the masked-only blocks).
    s += "    mov.u32 %kb,0;\n";
    if causal {
        s += &format!("KB_{name}:\n    setp.gt.u32 %p0,%kb,%row;\n    @%p0 bra DONE_{name};\n");
    } else {
        s += &format!("KB_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra DONE_{name};\n");
    }

    // 1. S = Q·Kᵀ : two key n-tiles, each accumulating ktq contraction tiles into %s{nk}_{0..3}.
    for nk in 0..2 {
        for r in 0..4 {
            s += &format!("    mov.f32 %s{nk}_{r},0f00000000;\n");
        }
        // gkey = kb + 8*nk + grp  (the key column n=grp for this n-tile)
        s += &format!("    add.u32 %gkey,%kb,{};\n    add.u32 %gkey,%gkey,%grp;\n", nk * 8);
        for kt in 0..ktq {
            // b0=pack(K[gkey][16kt+tg2], K[gkey][16kt+tg2+1]); b1=pack(K[gkey][16kt+tg2+8],..+9)
            s += &format!("    mul.lo.s32 %tmp,%gkey,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%K,%off;\n    ld.global.b32 %b0,[%base];\n", kt * 16);
            s += "    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%K,%off;\n    ld.global.b32 %b1,[%base];\n";
            s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}};\n");
        }
    }

    // causal mask: set s = -inf where the key index exceeds the query index. Keys this lane owns are
    // k0=kb+tg2, k1=kb+tg2+1 (n-tile 0) and k8=kb+8+tg2, k9=kb+8+tg2+1 (n-tile 1); queries are qr0,qr1.
    // s{0,1}_{0,1} are qr0, s{0,1}_{2,3} are qr1 (the D-fragment row map). Only the diagonal block kb==row
    // actually trips a mask (lower blocks have every key < query); the loop already skips kb>row.
    if causal {
        s += "    add.u32 %ck0,%kb,%tg2;\n    add.u32 %ck1,%ck0,1;\n    add.u32 %ck8,%ck0,8;\n    add.u32 %ck9,%ck8,1;\n";
        for (sreg, key, qr) in [
            ("%s0_0", "%ck0", "%qr0"), ("%s0_1", "%ck1", "%qr0"),
            ("%s0_2", "%ck0", "%qr1"), ("%s0_3", "%ck1", "%qr1"),
            ("%s1_0", "%ck8", "%qr0"), ("%s1_1", "%ck9", "%qr0"),
            ("%s1_2", "%ck8", "%qr1"), ("%s1_3", "%ck9", "%qr1"),
        ] {
            s += &format!("    setp.gt.u32 %p0,{key},{qr};\n    selp.f32 {sreg},0fFF800000,{sreg},%p0;\n");
        }
    }

    // 2. online softmax (register-resident). Lane owns 4 keys per row; the row max/sum need the 4 lanes
    //    sharing grp (tg=0..3 cover all 16 keys) — reduce with shfl.bfly offsets 1,2 (stays in-group).
    // qrow0 = scores _0,_1 ; qrow1 = scores _2,_3.
    s += "    max.f32 %lmax0,%s0_0,%s0_1;\n    max.f32 %lmax0,%lmax0,%s1_0;\n    max.f32 %lmax0,%lmax0,%s1_1;\n    mul.f32 %lmax0,%lmax0,%scale;\n";
    s += "    max.f32 %lmax1,%s0_2,%s0_3;\n    max.f32 %lmax1,%lmax1,%s1_2;\n    max.f32 %lmax1,%lmax1,%s1_3;\n    mul.f32 %lmax1,%lmax1,%scale;\n";
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax0,{off},0x1f,0xffffffff;\n    max.f32 %lmax0,%lmax0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax1,{off},0x1f,0xffffffff;\n    max.f32 %lmax1,%lmax1,%rt;\n");
    }
    s += &format!("    max.f32 %mnew0,%m0,%lmax0;\n    sub.f32 %corr0,%m0,%mnew0;\n    mul.f32 %corr0,%corr0,{log2e};\n    ex2.approx.f32 %corr0,%corr0;\n");
    s += &format!("    max.f32 %mnew1,%m1,%lmax1;\n    sub.f32 %corr1,%m1,%mnew1;\n    mul.f32 %corr1,%corr1,{log2e};\n    ex2.approx.f32 %corr1,%corr1;\n");
    // rescale register O (rows _0,_1 by corr0; _2,_3 by corr1)
    for nt in 0..nto {
        s += &format!("    mul.f32 %o{nt}_0,%o{nt}_0,%corr0;\n    mul.f32 %o{nt}_1,%o{nt}_1,%corr0;\n    mul.f32 %o{nt}_2,%o{nt}_2,%corr1;\n    mul.f32 %o{nt}_3,%o{nt}_3,%corr1;\n");
    }
    // probabilities p = ex2((scale·s - mnew)·log2e); pack into the P A-fragment; accumulate row sums.
    // emit one prob into %tpX from score %sREG with running-max %mnewR:
    let prob = |dst: &str, sreg: &str, mnew: &str| -> String {
        format!("    mul.f32 %pp,{sreg},%scale;\n    sub.f32 %pp,%pp,{mnew};\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 {dst},%pp;\n")
    };
    // pack two f32 probs (lo,hi) into a b32 f16 pair (lo in low 16) for the mma A operand.
    let pack = |dst: &str, lo: &str, hi: &str| -> String {
        format!("    cvt.rn.f16.f32 %h0,{lo};\n    and.b32 %h0,%h0,65535;\n    cvt.rn.f16.f32 %h1,{hi};\n    shl.b32 %h1,%h1,16;\n    or.b32 {dst},%h0,%h1;\n")
    };
    // qrow0: tp0=p(s0_0), tp1=p(s0_1), tp2=p(s1_0), tp3=p(s1_1)
    s += &prob("%tp0", "%s0_0", "%mnew0");
    s += &prob("%tp1", "%s0_1", "%mnew0");
    s += &prob("%tp2", "%s1_0", "%mnew0");
    s += &prob("%tp3", "%s1_1", "%mnew0");
    s += "    add.f32 %psum0,%tp0,%tp1;\n    add.f32 %psum0,%psum0,%tp2;\n    add.f32 %psum0,%psum0,%tp3;\n";
    s += &pack("%a0", "%tp0", "%tp1"); // n-tile0 keys tg2,tg2+1 (row grp)
    s += &pack("%a2", "%tp2", "%tp3"); // n-tile1 keys 8+tg2,8+tg2+1 (row grp)
    // qrow1: tp0=p(s0_2), tp1=p(s0_3), tp2=p(s1_2), tp3=p(s1_3)
    s += &prob("%tp0", "%s0_2", "%mnew1");
    s += &prob("%tp1", "%s0_3", "%mnew1");
    s += &prob("%tp2", "%s1_2", "%mnew1");
    s += &prob("%tp3", "%s1_3", "%mnew1");
    s += "    add.f32 %psum1,%tp0,%tp1;\n    add.f32 %psum1,%psum1,%tp2;\n    add.f32 %psum1,%psum1,%tp3;\n";
    s += &pack("%a1", "%tp0", "%tp1"); // n-tile0 keys tg2,tg2+1 (row grp+8)
    s += &pack("%a3", "%tp2", "%tp3"); // n-tile1 keys 8+tg2,8+tg2+1 (row grp+8)
    // reduce the per-lane partial row sums across the tg-group, then l = l·corr + Σp.
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum0,{off},0x1f,0xffffffff;\n    add.f32 %psum0,%psum0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum1,{off},0x1f,0xffffffff;\n    add.f32 %psum1,%psum1,%rt;\n");
    }
    s += "    fma.rn.f32 %l0,%l0,%corr0,%psum0;\n    fma.rn.f32 %l1,%l1,%corr1,%psum1;\n";
    s += "    mov.f32 %m0,%mnew0;\n    mov.f32 %m1,%mnew1;\n";

    // 4. O += P·V : nto output n-tiles, each one k-tile (16 keys). V is the .col B operand [hdim][key];
    //    b0=pack(V[kb+tg2][8nt+grp], V[kb+tg2+1][8nt+grp]); b1=pack(V[kb+tg2+8][..], V[kb+tg2+9][..]).
    for nt in 0..nto {
        s += &format!("    add.u32 %idx,%grp,{};\n", nt * 8); // hdim column = 8nt+grp
        // b0: keys kb+tg2, kb+tg2+1 (stride d in global ⇒ two u16 loads)
        s += "    add.u32 %gkey,%kb,%tg2;\n";
        s += &format!("    mul.lo.s32 %tmp,%gkey,{d};\n    add.u32 %tmp,%tmp,%idx;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%V,%off;\n    ld.global.u16 %h0,[%base];\n");
        s += &format!("    add.u32 %tmp,%tmp,{d};\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%V,%off;\n    ld.global.u16 %h1,[%base];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b0,%h0,%h1;\n");
        // b1: keys kb+tg2+8, kb+tg2+9
        s += "    add.u32 %gkey,%kb,%tg2;\n    add.u32 %gkey,%gkey,8;\n";
        s += &format!("    mul.lo.s32 %tmp,%gkey,{d};\n    add.u32 %tmp,%tmp,%idx;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%V,%off;\n    ld.global.u16 %h0,[%base];\n");
        s += &format!("    add.u32 %tmp,%tmp,{d};\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%V,%off;\n    ld.global.u16 %h1,[%base];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b1,%h0,%h1;\n");
        s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
    }
    s += &format!("    add.u32 %kb,%kb,16;\n    bra KB_{name};\n");

    // store O[row][c] = o / l_row.  o_nt_{0,1}=qr0 cols 8nt+{tg2,tg2+1}; o_nt_{2,3}=qr1 cols 8nt+{tg2,tg2+1}.
    s += &format!("DONE_{name}:\n");
    for nt in 0..nto {
        s += &format!("    div.rn.f32 %o{nt}_0,%o{nt}_0,%l0;\n    div.rn.f32 %o{nt}_1,%o{nt}_1,%l0;\n    div.rn.f32 %o{nt}_2,%o{nt}_2,%l1;\n    div.rn.f32 %o{nt}_3,%o{nt}_3,%l1;\n");
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_0;\n    st.global.f32 [%base+4],%o{nt}_1;\n", nt * 8);
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_2;\n    st.global.f32 [%base+4],%o{nt}_3;\n", nt * 8);
    }
    s += "    ret;\n}\n";
    s
}

/// Flash-attention module: untiled `flash_d{D}` + tiled `flash_d{D}_t` per supported head dim, the
/// tensor-core `flash_d64_w` (16-key tile) and wide-key `flash_d64_w4` (64-key tile), plus the
/// register-resident `flash_d64_m` (hand-placed `mma.sync`, O/m/l in registers — no SMEM round-trip).
pub fn flash_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
        for &d in &SUPPORTED_D {
            m += &entry_untiled(d, FLASH_WARPS);
            m += &entry_tiled(d, FLASH_TWARPS);
        }
        m += &entry_wmma(64);
        m += &entry_wmma_wide(64, WMMA_FLASH_NKB);
        m += &entry_mma_reg(64, false);
        m += &entry_mma_reg(64, true);
        m
    })
    .as_str()
}

/// 16-key sub-tiles staged per online-softmax step in the wide WMMA flash (`flash_d64_w{N}`). `WK = 16·N`
/// keys per step ⇒ `S % WK == 0` required; 4 → 64-key tile, divides every layer seq (which is `S%64==0`).
pub const WMMA_FLASH_NKB: usize = 4;

/// Head dims with a generated kernel (the common transformer values).
pub const SUPPORTED_D: [usize; 3] = [32, 64, 128];
