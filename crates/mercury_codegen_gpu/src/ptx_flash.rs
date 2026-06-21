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

/// Flash-attention module: untiled `flash_d{D}` + tiled `flash_d{D}_t` per supported head dim, plus the
/// experimental tensor-core `flash_d64_w`.
pub fn flash_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
        for &d in &SUPPORTED_D {
            m += &entry_untiled(d, FLASH_WARPS);
            m += &entry_tiled(d, FLASH_TWARPS);
        }
        m += &entry_wmma(64);
        m
    })
    .as_str()
}

/// Head dims with a generated kernel (the common transformer values).
pub const SUPPORTED_D: [usize; 3] = [32, 64, 128];
