//! Fused **flash-attention** on the GPU — the kernel that *lost* on CPU (≈2× slower than the
//! GEMM-dispatch path on AVX2, because materialized attention is compute-bound on the tuned GEMM and
//! the S² scores traffic isn't the bottleneck) but is GPU-shaped: it never materializes S = Q·Kᵀ in
//! HBM, using the online-softmax recurrence to stream K/V once.
//!
//! Layout: single head, Q/K/V/O are `[S, D]` row-major (f32 for the CUDA-core kernels, f16 for the
//! tensor-core ones), `D` a multiple of 32. Every kernel here streams K/V in ascending key order under
//! the same online-softmax recurrence; `exp` is `ex2.approx`, so all of them are tolerance-gated vs a
//! CPU f64 reference (`ref_attn`), never bit-exact against it.
//!
//! ## The f32 CUDA-core pair (`gpu::flash_plan` selects; see [`FLASH_TILE_MIN`])
//!
//! **One warp per query row**; the 32 lanes split the head dim (lane `t` owns d ∈ {t, t+32, …}, i.e.
//! `R = D/32` values). For each key j: each lane computes its partial of Q[i]·K[j], a warp butterfly
//! all-reduce gives the full score, then the online-softmax update rescales the running denominator `l`
//! and the per-lane output accumulators `acc[r]`. `R` is unrolled at PTX-gen time (acc/q in registers),
//! so a kernel is generated per supported D.
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
//! The per-key arithmetic and the ascending key order are **identical** between these two kernels, so
//! for any S they produce **bit-identical** output — tiling is a pure data-movement optimisation, and the
//! tolerance gate (which runs *both*) confirms the SMEM plumbing is correct.
//!
//! **No-deadlock (tiled).** Threads are `32·FLASH_TWARPS` per CTA, `row = ctaid·W + warpId`. The
//! cooperative load and the two per-block `bar.sync`s run on *every* thread; only the per-row score/
//! store is predicated on `row < S`. So the warps of a ragged final CTA (when `FLASH_TWARPS ∤ S`) still
//! help stage K/V and still reach both barriers — they just skip their own compute — so the barriers
//! can never deadlock. Shared memory is static, so the launch config reserves no dynamic SMEM.
//!
//! ## The fp16 tensor-core family (`gpu::wmma_flash_entry` / `gpu::ws_flash_route` select)
//!
//! Everything else in this file is a *tensor-core* flash generator for a 16-query tile per warp, gated
//! to `D ∈ {64, 128}`. In rough order of the levers they add (each generator's own doc comment carries
//! the derivation and the A/B verdict):
//!
//!  * `flash_d64_w` / `flash_d64_w4` — opaque `wmma` QKᵀ and PV, 16- and 64-key tiles.
//!  * `flash_d64_m` / `flash_d64_mc` (causal) — hand-placed `mma.sync.m16n8k16`, O/m/l
//!    **register-resident** (no SMEM round-trip of the accumulator).
//!  * `flash_d{64,128}_mp{,c}{,_lm}` — the production family: the `_m` kernel plus a `cp.async` K/V
//!    pipeline; `c` = causal, `_lm` = `ldmatrix` SMEM feed (the D=128 default). Plus `flash_d64_m1`
//!    (single-buffer occupancy probe), `flash_d64_mp{4,8}` (multi-warp) and `flash_d64_mpw{2,4}`
//!    (wide key tile).
//!  * `flash_d64_msp` (softmax/QKᵀ software pipeline), `flash_d128_hs` (head-dim warp split),
//!    `flash_d64_mprope` (fused RoPE, two extra `cos`/`sin` params).
//!  * `flash_d64_ws{,c}`, `flash_d128_ws{,c}_lm` and the 3-stage `flash_d64_ws3` /
//!    `flash_d128_ws3_lm` — FA2/FA3-style **warp-specialized ping-pong** over disjoint query tiles
//!    sharing one staged K/V stream; dispatched only under `WUKONG_FLASH_WS=1`.
//!
//! `every_dispatchable_flash_entry_is_defined` pins each routing decision to a defined entry.
//!
//! ## The depth × staging-width grid ([`FLASH_STAGE_VARIANTS`], [`flash_stage_ptx`])
//!
//! Everything above lives in the one `flash_ptx()` module and is capped by the PTX ISA's **48 KiB
//! static `.shared` limit** (PTX §5.1.7 — a rule about *statically declared* shared memory on every
//! target, not a device fact: it is the same 48 KiB on this Ada card's 99 KiB carveout, on A100's 164
//! and on H100's 228). [`entry_mma_reg_pipe`] and [`entry_mma_reg_pipe_wide`] therefore take a
//! `stages` ring depth and an `smem_budget`, and switch to the module-scope
//! [`crate::gpu::DSMEM_DECL`] window above the cap — the same `smem_mode_for`/[`crate::gpu::SmemMode`]
//! emission rule `ptx_int8` and `ptx_fp8` already use. Every combination beyond the shipped
//! `(stages, feed)` points is enumerated as a grid row with its own entry name and module-cache key,
//! generated on demand rather than added to `flash_ptx()` (which every flash launch JITs).
//!
//! At the shipped arguments the emitted text is **byte-identical** to the pre-window tree, pinned by
//! `flash_ptx_shipped_generators_are_byte_identical` against digests taken before the change.

use std::sync::OnceLock;

use crate::gpu::{smem_mode_for, SmemMode, DSMEM_DECL, DSMEM_SYM, STATIC_SMEM_CAP};

/// **(max warps per SM, max thread blocks per SM)** for a compute capability — the two hardware
/// occupancy limits every "how many warps per CTA?" packing decision in this file derives from.
///
/// FACT(doc): CUDA C Programming Guide, *Compute Capabilities* / *Technical Specifications per
/// Compute Capability*. The two columns move independently across parts, which is precisely why a
/// warps-per-CTA constant derived on one architecture cannot be assumed on another: Ada (8.9) allows
/// 24 blocks and 48 warps, A100/H100 allow **32 blocks and 64 warps**, and GA10x (8.6) allows only 16
/// blocks against the same 48 warps.
///
/// An unknown / newer capability falls back to the datacenter row `(64, 32)`. That is the
/// *conservative* direction for everything here: it derives the SMALLEST warps-per-CTA, i.e. the
/// finest possible grid, which is always legal — it can under-fill the warp pool but can never
/// over-subscribe a CTA or a barrier.
pub const fn occupancy_limits(cc_major: i32, cc_minor: i32) -> (u32, u32) {
    match (cc_major, cc_minor) {
        (7, 0) | (7, 2) => (64, 32), // Volta / Xavier
        (7, 5) => (32, 16),          // Turing
        (8, 0) => (64, 32),          // A100
        (8, 6) | (8, 7) => (48, 16), // GA10x / Orin
        (8, 9) => (48, 24),          // Ada — this dev box's RTX 4050
        (9, 0) => (64, 32),          // H100
        (12, _) => (48, 24),         // Blackwell consumer
        _ => (64, 32),               // Blackwell datacenter and anything newer
    }
}

/// **Query-row warps per CTA that exactly fill a target's warp pool when the CTA count binds** —
/// `ceil(max_warps_per_sm / max_blocks_per_sm)` from [`occupancy_limits`].
///
/// The untiled flash kernel puts **one independent query row per warp**, so the total warps in flight
/// is `S` no matter how they are packed: `W` changes the CTA count, not the warp supply. `W` therefore
/// matters for exactly one reason — a device caps *resident blocks per SM* independently of *resident
/// warps per SM*, so a one-warp CTA leaves `1 - blocks_cap/warps_cap` of the warp pool unreachable.
/// `W = ceil(warps_cap / blocks_cap)` is the smallest packing that closes that gap, and staying at the
/// smallest keeps the grid fine (better tail behaviour at small `S`).
pub const fn flash_untiled_warps(cc_major: i32, cc_minor: i32) -> u32 {
    let (warps_sm, blocks_sm) = occupancy_limits(cc_major, cc_minor);
    warps_sm.div_ceil(blocks_sm)
}

/// Query-row warps per CTA for the **untiled** kernel (`flash_d{D}`), baked into both the generated
/// PTX (`row = ctaid.x*W + warpId`) and `gpu::flash_plan_forced`'s launch geometry.
///
/// **This is [`flash_untiled_warps`] evaluated at Ada, and the retarget re-examined whether it may
/// stay a constant. It may — here is the argument, because the old one was wrong.** The historical
/// comment derived W=2 from "Ada's ~24-blocks/SM cap", which is an Ada fact and does not transfer:
/// A100/H100 allow 32 blocks and 64 warps per SM. What transfers is the *ratio*, and it is the same:
/// `ceil(48/24) = ceil(64/32) = 2` on Ada, A100, H100, Volta and Turing alike. The one shipped
/// capability where the derivation disagrees is GA10x/Orin (8.6/8.7: `ceil(48/16) = 3`), and no
/// dispatch path can reach it — [`FLASH_TILE_MIN`] is 0, so the tiled kernel is selected at every `S`
/// and this kernel survives only as the A/B reference.
///
/// **If a future dispatcher does want a device-derived W, both sides must read
/// [`flash_untiled_warps`], not this constant.** `flash_d{D}` bakes W into its row indexing while
/// `gpu::flash_plan_forced` computes `block_dim = 32·W` from the same constant; deriving one side
/// alone silently computes the wrong rows (no error, wrong output). `flash_warps_is_the_derived_ada_
/// packing` pins the derivation, the table, and this equality.
pub const FLASH_WARPS: u32 = flash_untiled_warps(8, 9);

/// Query-row warps per CTA for the **tiled** kernel (`flash_d{D}_t`) — *also the K/V SMEM reuse factor*
/// (each staged key block is read from L2 once and reused by this many rows). W=8: a 256-thread CTA
/// gives an 8× L2-traffic cut, the long-sequence lever, while 8 KB of static SMEM/CTA stays small
/// enough for high occupancy.
///
/// **Unlike [`FLASH_WARPS`] this is not an occupancy-derived quantity and must not become one** — it
/// is the *reuse factor*, chosen for L2 traffic, and the occupancy limits only have to not veto it.
/// They don't, anywhere: `32·8 = 256` threads is a quarter of the 1024-thread CTA maximum, and the CTA
/// count it implies (`warps_cap/8` = 6 on Ada, 8 on A100/H100) is under every part's blocks-per-SM cap
/// with room to spare. `flash_twarps_is_legal_on_every_target` pins that.
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
    assert!(
        1024 % d == 0,
        "D must divide 1024 (key-block tiling: BK = 1024/D)"
    );
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
    s += &format!(
        "AFTER_{name}:\n    bar.sync 0;\n    add.u32 %kblock,%kblock,{bk};\n    bra KB_{name};\n"
    );

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
        s += &format!(
            "    add.u32 %gkey,%kb,{};\n    add.u32 %gkey,%gkey,%grp;\n",
            nk * 8
        );
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
            ("%s0_0", "%ck0", "%qr0"),
            ("%s0_1", "%ck1", "%qr0"),
            ("%s0_2", "%ck0", "%qr1"),
            ("%s0_3", "%ck1", "%qr1"),
            ("%s1_0", "%ck8", "%qr0"),
            ("%s1_1", "%ck9", "%qr0"),
            ("%s1_2", "%ck8", "%qr1"),
            ("%s1_3", "%ck9", "%qr1"),
        ] {
            s += &format!(
                "    setp.gt.u32 %p0,{key},{qr};\n    selp.f32 {sreg},0fFF800000,{sreg},%p0;\n"
            );
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

/// Generate the **`cp.async`-pipelined, SMEM-staged** register-resident flash kernel for head dim `d`.
/// Name = `flash_d{d}_mp` (or `flash_d{d}_mpc` when `causal`). Identical online-softmax math, key order,
/// and per-key arithmetic to [`entry_mma_reg`] — so it is **bit-identical** output (the gate cross-checks
/// it against the same `ref_attn`) — but it closes the kernel's one remaining inefficiency: at the
/// multi-head-filled occupancy where attention is **per-warp latency-bound**, [`entry_mma_reg`] stalls
/// twice per key block on uncached global loads (K as B-fragments, V as strided `u16` pairs), with
/// nothing to hide the ~400-cycle latency behind.
///
/// This kernel **software-pipelines the K-loop** the way [`super::ptx_wmma::entry_smem_db`] pipelines the
/// GEMM K-loop: each 16-key block's K and V are a *contiguous* `16·d·2`-byte slab in global (16 rows ×
/// full width), so the warp stages them into shared memory with clean 16-byte `cp.async.cg` copies, and
/// **prefetches block `kb+1` into the alternate buffer while the tensor cores consume block `kb`** —
/// overlapping the global-load latency with the QKᵀ/softmax/PV compute. Two K+V buffers (`bufc` current,
/// `bufp` prefetch) ping-pong; `cp.async.wait_group 1` blocks only on the older (current) copy. The MMA
/// B-fragments are then read from SMEM (`ld.shared`) instead of global — K as one clean `b32`, V as the
/// same strided `u16` pair but now at ~30-cycle SMEM latency and already resident.
///
/// **Causal** (`flash_d{d}_mpc`): same diagonal mask + upper-block skip as [`entry_mma_reg`]; the prefetch
/// is guarded so the diagonal block (the last one processed) does not stage an out-of-range successor.
///
/// **`stages == 1`** (`flash_d{d}_m1`, non-causal + hand-packed only): a *diagnostic* single-buffered
/// twin. It stages ONE K+V slab (`smem = bufsz = 2·ksz` — 4 KB/CTA at D=64, hitting the 24-block/SM
/// occupancy cap, ~50% vs ~25% for the double-buffered `mp`) and forgoes the `cp.async` compute/load
/// overlap: each block is stage → drain → compute in series. Bit-identical online-softmax math to the
/// `mp` kernel (same QKᵀ/softmax/PV emission), so `flash_single_vs_double` can A/B occupancy against
/// pipeline overlap at long S. Not a shipped dispatch — a probe for the warp-specialization decision.
///
/// **`stages >= 3`** (`flash_d{d}_mp{c}{_lm}_s{stages}`): the `stages`-deep `cp.async` ring, the depth
/// lever the 48 KiB static cap and Ada's occupancy budget kept out of reach. `stages == 2`'s prefetch
/// distance is exactly one block: the copy for block `i+1` is issued at the top of body `i` and must
/// land by that same body's `wait_group 1`, so it gets one QKᵀ+softmax+PV stretch to cover a full HBM
/// round-trip. A `stages`-buffer ring issues block `i+stages-1` instead, giving each copy `stages-1`
/// bodies of slack for the same per-body work (D6 §3.2: `s* = 1 + ceil(L/t_k)`, and `L/t_k` *rises* on
/// the datacenter parts because per-SM tensor throughput grows faster than per-SM bandwidth). The ring
/// is **add + wrap**, never the two-buffer XOR/swap: at `stages > 2` a toggle cycles two of the buffers
/// and silently corrupts every deeper one. Prologue stages blocks `0..stages-2` (one `commit_group`
/// each, the stage itself guarded against a short `S` / the causal diagonal) so the steady-state
/// invariant `wait_group stages-1` holds from body 0; the tail commits **empty** groups so the
/// per-thread group count stays exactly one per body — the same discipline `entry_mma_reg_pipe_ws3`
/// already uses at `stages == 3`. Math, key order and register layout are untouched, so a deeper ring
/// is bit-identical to `mp` at the same `(d, causal, pv_ldmatrix)`.
///
/// **`smem_budget` (bytes) is the ceiling this entry may spend, and it also selects the emission form**
/// ([`crate::gpu::smem_mode_for`]): at or below the PTX ISA's 48 KiB **static** cap the ring stays a
/// `.shared` array declared inside the entry — **byte-identical text**, so every shipped kernel, its
/// cubin-cache warmth and its measured behaviour are untouched — and beyond it the whole ring moves
/// into the single module-scope [`crate::gpu::DSMEM_DECL`] window (this kernel has exactly one SMEM
/// object, so the window needs no sub-slab offsets). The budget is *passed in* by the dispatch layer
/// from `Gpu::smem_budget()`, never probed here, so the whole depth grid is enumerable and gateable
/// off-device and an A100/H100 budget is testable on a laptop. Over-budget is a **loud panic at
/// generation**: a decline belongs in the dispatcher's applicability test, never in a clamped launch.
fn entry_mma_reg_pipe(
    d: usize,
    causal: bool,
    pv_ldmatrix: bool,
    stages: usize,
    smem_budget: usize,
) -> String {
    assert!(d % 16 == 0, "mma flash needs D % 16 == 0");
    assert!(stages >= 1, "flash pipeline needs >= 1 buffer");
    let single_buf = stages == 1;
    assert!(
        !single_buf || (!causal && !pv_ldmatrix),
        "single_buf flash is defined only for the non-causal hand-packed kernel (flash_d{d}_m1)"
    );
    let ktq = d / 16; // Q.Kt contraction tiles (over hdim)
    let nto = d / 8; // P.V output n-tiles (over hdim)
    let ksz = 16 * d * 2; // bytes of one staged K block (== one V block): 16 keys * d hdim * 2 (f16)
    let bufsz = 2 * ksz; // K+V slab per pipeline buffer
    let cpl = d / 16; // 16-byte cp.async chunks per lane per tensor (16*d*2/16/32 = d/16)
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    // SMEM(stages) = stages * 2 * (16 * d * 2) = 64 * stages * d — the family's closed form. The 48 KiB
    // ISA cap decides the FORM; `smem_budget` is the ceiling and is asserted, not clamped.
    let smem_total = stages * bufsz;
    let mode = smem_mode_for(smem_total);
    let name = if single_buf {
        format!("flash_d{d}_m1")
    } else {
        let base = match (causal, pv_ldmatrix) {
            (false, false) => format!("flash_d{d}_mp"),
            (true, false) => format!("flash_d{d}_mpc"),
            (false, true) => format!("flash_d{d}_mp_lm"),
            (true, true) => format!("flash_d{d}_mpc_lm"),
        };
        // The depth is part of the entry symbol AND therefore of the module-cache key: `Gpu::function`
        // never re-examines PTX on a key hit, so two depths under one name would silently run the first
        // one's ring with the second one's launch window. `stages == 2` keeps the historical spelling.
        if stages == 2 {
            base
        } else {
            format!("{base}_s{stages}")
        }
    };
    assert!(
        smem_total <= smem_budget,
        "{name}: flash SMEM {smem_total} B (stages={stages}, D={d}) exceeds the budget {smem_budget} B"
    );
    // Where the ring lives. Static: its own `.shared` array (the historical spelling, emitted verbatim).
    // Dynamic: the ONE module-scope window — two module-scope externs alias, and this kernel has a
    // single SMEM object anyway, so the base offset is 0 and every address is formed through the symbol
    // exactly as before (the window's base is NOT guaranteed to be 0; it starts after any statics).
    let sym = match mode {
        SmemMode::Static => format!("smem_{name}"),
        SmemMode::Dynamic(_) => DSMEM_SYM.to_string(),
    };

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0,%pnext;\n";
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
    // pipeline scratch: bufc/bufp = current/prefetch buffer byte-offsets; next = kb+16; sbase/sk/chunk/
    // lkey/bswap = SMEM address arithmetic.
    s += &format!(
        "    .reg .b32 {br}%S,%lane,%grp,%tg,%tg2,%row,%qr0,%qr1,%kb,%gkey,%idx,%tmp,%hoff,%bufc,%bufp,%next,%sbase,%sk,%chunk,%lkey,%bswap;\n"
    );
    s += "    .reg .b64 %Q,%K,%V,%O,%base,%off;\n";
    // Double-buffered: two K+V slabs to ping-pong (prefetch vs compute). Single-buffered probe: one
    // slab. Deep ring: `stages` slabs. A dynamic ring declares nothing here — its window is the
    // module-scope `.extern .shared`, which `flash_stage_ptx` emits (the identical line inside an
    // entry body is CUDA_ERROR_INVALID_PTX).
    if !mode.is_dynamic() {
        s += &format!("    .shared .align 16 .b8 {sym}[{smem_total}];\n");
    }

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    // multi-head head base offset (ctaid.y), identical to entry_mma_reg.
    s += &format!("    mov.u32 %hoff,%ctaid.y;\n    mul.lo.u32 %hoff,%hoff,%S;\n    mul.lo.u32 %hoff,%hoff,{d};\n");
    s += "    mul.wide.u32 %off,%hoff,2;\n    add.s64 %Q,%Q,%off;\n    add.s64 %K,%K,%off;\n    add.s64 %V,%V,%off;\n";
    s += "    mul.wide.u32 %off,%hoff,4;\n    add.s64 %O,%O,%off;\n";
    s += "    mov.u32 %lane,%tid.x;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg2,%tg,1;\n";
    s += "    mov.u32 %row,%ctaid.x;\n    shl.b32 %row,%row,4;\n    add.u32 %qr0,%row,%grp;\n    add.u32 %qr1,%qr0,8;\n";

    // Load Q A-fragments once (reused across every key block) — identical to entry_mma_reg.
    for kt in 0..ktq {
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_0,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_2,[%base];\n");
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_1,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_3,[%base];\n");
    }
    for nt in 0..nto {
        for r in 0..4 {
            s += &format!("    mov.f32 %o{nt}_{r},0f00000000;\n");
        }
    }
    s += "    mov.f32 %m0,0fFF800000;\n    mov.f32 %m1,0fFF800000;\n    mov.f32 %l0,0f00000000;\n    mov.f32 %l1,0f00000000;\n";

    // Emit the cooperative cp.async stage of the 16-key K+V slab at global key index `kbreg` into the
    // SMEM buffer at byte-offset register `bufreg`. The slab is contiguous in global (16 rows * d), so
    // each lane copies `cpl` 16-byte chunks of K and the matching chunks of V.
    let stage = |kbreg: &str, bufreg: &str| -> String {
        let mut t = String::new();
        for ci in 0..cpl {
            t += &format!("    add.u32 %chunk,%lane,{};\n", ci * 32);
            // global element offset = kb*d + chunk*8 ; byte offset in %off (shared by K and V src)
            t += &format!("    mul.lo.u32 %tmp,{kbreg},{d};\n    shl.b32 %sk,%chunk,3;\n    add.u32 %tmp,%tmp,%sk;\n    mul.wide.u32 %off,%tmp,2;\n");
            // SMEM K dst = smem + bufreg + chunk*16
            t += &format!("    mov.u32 %sbase,{sym};\n    add.u32 %sbase,%sbase,{bufreg};\n    shl.b32 %sk,%chunk,4;\n    add.u32 %sbase,%sbase,%sk;\n");
            t += "    add.s64 %base,%K,%off;\n    cp.async.cg.shared.global [%sbase],[%base],16;\n";
            // SMEM V dst = K dst + ksz
            t += &format!("    add.u32 %sbase,%sbase,{ksz};\n    add.s64 %base,%V,%off;\n    cp.async.cg.shared.global [%sbase],[%base],16;\n");
        }
        t
    };

    if single_buf {
        // Single-buffered probe: no prologue/prefetch. Each iteration stages the current block into the
        // sole buffer (bufc=0), drains it fully, and computes — losing the `cp.async` compute/load
        // overlap the double-buffered path gets, but halving SMEM to 4 KB (D=64) so ~24 CTAs/SM fit.
        s += "    mov.u32 %bufc,0;\n    mov.u32 %kb,0;\n";
        s += &format!("KB_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra DONE_{name};\n");
        // WAR: a warp re-staging into the one buffer must wait for the prior iteration's SMEM reads
        // (QKᵀ/PV) to retire before overwriting. Harmless on the first iteration.
        s += "    bar.sync 0;\n";
        s += &stage("%kb", "%bufc");
        s += "    cp.async.commit_group;\n    cp.async.wait_group 0;\n    bar.sync 0;\n";
    } else if stages == 2 {
        // init pipeline + prologue: stage block 0 into bufc=0, commit. (S>=16 ⇒ block 0 always exists.)
        s += &format!("    mov.u32 %bufc,0;\n    mov.u32 %bufp,{bufsz};\n    mov.u32 %kb,0;\n");
        s += &stage("%kb", "%bufc");
        s += "    cp.async.commit_group;\n";

        // for kb in 0..S step 16 (causal: stop at the diagonal block kb==row).
        if causal {
            s += &format!("KB_{name}:\n    setp.gt.u32 %p0,%kb,%row;\n    @%p0 bra DONE_{name};\n");
        } else {
            s += &format!("KB_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra DONE_{name};\n");
        }
        // prefetch block kb+16 into bufp iff it is in range; else just drain the current copy.
        s += "    add.u32 %next,%kb,16;\n";
        if causal {
            s += "    setp.gt.u32 %pnext,%next,%row;\n";
        } else {
            s += "    setp.ge.u32 %pnext,%next,%S;\n";
        }
        s += &format!("    @%pnext bra NOPF_{name};\n");
        s += &stage("%next", "%bufp");
        s += &format!(
            "    cp.async.commit_group;\n    cp.async.wait_group 1;\n    bra PFDONE_{name};\n"
        );
        s += &format!("NOPF_{name}:\n    cp.async.wait_group 0;\n");
        s += &format!("PFDONE_{name}:\n    bar.sync 0;\n");
    } else {
        // ---- the `stages`-deep cp.async ring (add + wrap; a two-buffer XOR/swap is invalid here) ----
        //
        // Group bookkeeping is the whole correctness argument, so state it: buffer slot = block index
        // mod `stages`, and *exactly one* `cp.async` group is committed per staged block AND per body,
        // so block `j`'s group is always committed group `j`. The prologue commits `stages-1` groups
        // (blocks 0..stages-2); body `j` commits one more (block `j+stages-1`, or an EMPTY group in the
        // tail), making `stages+j` committed when body `j` waits. `wait_group W` retires all but the
        // last `W`, so `W = stages-1` retires groups `0..j` — exactly the one filling the buffer this
        // body reads, and no more (a deeper wait would serialize the ring).
        //
        // WAR: body `j` overwrites the slot of block `j-1`, whose only reader was body `j-1`'s
        // QKᵀ/PV — one body earlier in this single warp's program order. That is precisely the
        // `stages == 2` hazard (there too the prefetch target is the buffer the previous body read),
        // so the deeper ring adds no new ordering requirement.
        s += "    mov.u32 %kb,0;\n";
        s += &stage("%kb", "0");
        s += "    cp.async.commit_group;\n";
        for i in 1..stages - 1 {
            s += "    add.u32 %kb,%kb,16;\n";
            if causal {
                s += "    setp.gt.u32 %p0,%kb,%row;\n";
            } else {
                s += "    setp.ge.u32 %p0,%kb,%S;\n";
            }
            // The stage is guarded (a short S / the causal diagonal must not read out of range); the
            // commit is NOT — an empty group still counts, and dropping it would shift every later
            // group index and make `wait_group` wait on the wrong copy.
            s += &format!("    @%p0 bra PS{i}_{name};\n");
            s += &stage("%kb", &format!("{}", i * bufsz));
            s += &format!("PS{i}_{name}:\n    cp.async.commit_group;\n");
        }
        // bufc = slot of block 0; bufp = slot of block stages-1 (== the slot block -1 would have used).
        s += &format!(
            "    mov.u32 %kb,0;\n    mov.u32 %bufc,0;\n    mov.u32 %bufp,{};\n",
            (stages - 1) * bufsz
        );
        if causal {
            s += &format!("KB_{name}:\n    setp.gt.u32 %p0,%kb,%row;\n    @%p0 bra DONE_{name};\n");
        } else {
            s += &format!("KB_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra DONE_{name};\n");
        }
        s += &format!("    add.u32 %next,%kb,{};\n", (stages - 1) * 16);
        if causal {
            s += "    setp.gt.u32 %pnext,%next,%row;\n";
        } else {
            s += "    setp.ge.u32 %pnext,%next,%S;\n";
        }
        s += &format!("    @%pnext bra NOPF_{name};\n");
        s += &stage("%next", "%bufp");
        s += &format!("NOPF_{name}:\n    cp.async.commit_group;\n");
        s += &format!("    cp.async.wait_group {};\n    bar.sync 0;\n", stages - 1);
    }

    // 1. S = Q.Kt : two key n-tiles, K read from SMEM at bufc (local key = 8nk+grp, contiguous hdim).
    for nk in 0..2 {
        for r in 0..4 {
            s += &format!("    mov.f32 %s{nk}_{r},0f00000000;\n");
        }
        if pv_ldmatrix {
            // K via `ldmatrix.x2` (no trans): K is staged [key][hdim] = col-major B for mma.row.col — the
            // SAME layout as the gemm-NT B operand — so no transpose is needed. Per kt the source row is
            // key = 8nk + (lane&7); the chunk selector ((lane>>3)&1)·8 picks the k-low / k-high half of the
            // 16-hdim contraction tile. One conflict-free warp-collective load replaces the 2 strided
            // `ld.shared.b32` (which 8-way bank-conflict: a grp's lanes read keys 0,2,4,6 at fixed hdim).
            for kt in 0..ktq {
                s += &format!(
                    "    and.b32 %lkey,%lane,7;\n    add.u32 %lkey,%lkey,{};\n",
                    nk * 8
                );
                s += &format!("    mul.lo.u32 %tmp,%lkey,{d};\n");
                s += "    shr.u32 %idx,%lane,3;\n    and.b32 %idx,%idx,1;\n    shl.b32 %idx,%idx,3;\n";
                s += &format!("    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%idx;\n    shl.b32 %tmp,%tmp,1;\n", kt * 16);
                s += &format!("    mov.u32 %sbase,{sym};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,%tmp;\n");
                s += "    ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%b0,%b1},[%sbase];\n";
                s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}};\n");
            }
        } else {
            s += &format!("    add.u32 %lkey,%grp,{};\n", nk * 8);
            for kt in 0..ktq {
                s += &format!("    mul.lo.u32 %tmp,%lkey,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,{sym};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,%tmp;\n", kt * 16);
                s += "    ld.shared.b32 %b0,[%sbase];\n    ld.shared.b32 %b1,[%sbase+16];\n";
                s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}};\n");
            }
        }
    }

    // causal mask (identical to entry_mma_reg: global key indices vs query indices).
    if causal {
        s += "    add.u32 %ck0,%kb,%tg2;\n    add.u32 %ck1,%ck0,1;\n    add.u32 %ck8,%ck0,8;\n    add.u32 %ck9,%ck8,1;\n";
        for (sreg, key, qr) in [
            ("%s0_0", "%ck0", "%qr0"),
            ("%s0_1", "%ck1", "%qr0"),
            ("%s0_2", "%ck0", "%qr1"),
            ("%s0_3", "%ck1", "%qr1"),
            ("%s1_0", "%ck8", "%qr0"),
            ("%s1_1", "%ck9", "%qr0"),
            ("%s1_2", "%ck8", "%qr1"),
            ("%s1_3", "%ck9", "%qr1"),
        ] {
            s += &format!(
                "    setp.gt.u32 %p0,{key},{qr};\n    selp.f32 {sreg},0fFF800000,{sreg},%p0;\n"
            );
        }
    }

    // 2. online softmax (register-resident) — identical to entry_mma_reg.
    s += "    max.f32 %lmax0,%s0_0,%s0_1;\n    max.f32 %lmax0,%lmax0,%s1_0;\n    max.f32 %lmax0,%lmax0,%s1_1;\n    mul.f32 %lmax0,%lmax0,%scale;\n";
    s += "    max.f32 %lmax1,%s0_2,%s0_3;\n    max.f32 %lmax1,%lmax1,%s1_2;\n    max.f32 %lmax1,%lmax1,%s1_3;\n    mul.f32 %lmax1,%lmax1,%scale;\n";
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax0,{off},0x1f,0xffffffff;\n    max.f32 %lmax0,%lmax0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax1,{off},0x1f,0xffffffff;\n    max.f32 %lmax1,%lmax1,%rt;\n");
    }
    s += &format!("    max.f32 %mnew0,%m0,%lmax0;\n    sub.f32 %corr0,%m0,%mnew0;\n    mul.f32 %corr0,%corr0,{log2e};\n    ex2.approx.f32 %corr0,%corr0;\n");
    s += &format!("    max.f32 %mnew1,%m1,%lmax1;\n    sub.f32 %corr1,%m1,%mnew1;\n    mul.f32 %corr1,%corr1,{log2e};\n    ex2.approx.f32 %corr1,%corr1;\n");
    for nt in 0..nto {
        s += &format!("    mul.f32 %o{nt}_0,%o{nt}_0,%corr0;\n    mul.f32 %o{nt}_1,%o{nt}_1,%corr0;\n    mul.f32 %o{nt}_2,%o{nt}_2,%corr1;\n    mul.f32 %o{nt}_3,%o{nt}_3,%corr1;\n");
    }
    let prob = |dst: &str, sreg: &str, mnew: &str| -> String {
        format!("    mul.f32 %pp,{sreg},%scale;\n    sub.f32 %pp,%pp,{mnew};\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 {dst},%pp;\n")
    };
    let pack = |dst: &str, lo: &str, hi: &str| -> String {
        format!("    cvt.rn.f16.f32 %h0,{lo};\n    and.b32 %h0,%h0,65535;\n    cvt.rn.f16.f32 %h1,{hi};\n    shl.b32 %h1,%h1,16;\n    or.b32 {dst},%h0,%h1;\n")
    };
    s += &prob("%tp0", "%s0_0", "%mnew0");
    s += &prob("%tp1", "%s0_1", "%mnew0");
    s += &prob("%tp2", "%s1_0", "%mnew0");
    s += &prob("%tp3", "%s1_1", "%mnew0");
    s += "    add.f32 %psum0,%tp0,%tp1;\n    add.f32 %psum0,%psum0,%tp2;\n    add.f32 %psum0,%psum0,%tp3;\n";
    s += &pack("%a0", "%tp0", "%tp1");
    s += &pack("%a2", "%tp2", "%tp3");
    s += &prob("%tp0", "%s0_2", "%mnew1");
    s += &prob("%tp1", "%s0_3", "%mnew1");
    s += &prob("%tp2", "%s1_2", "%mnew1");
    s += &prob("%tp3", "%s1_3", "%mnew1");
    s += "    add.f32 %psum1,%tp0,%tp1;\n    add.f32 %psum1,%psum1,%tp2;\n    add.f32 %psum1,%psum1,%tp3;\n";
    s += &pack("%a1", "%tp0", "%tp1");
    s += &pack("%a3", "%tp2", "%tp3");
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum0,{off},0x1f,0xffffffff;\n    add.f32 %psum0,%psum0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum1,{off},0x1f,0xffffffff;\n    add.f32 %psum1,%psum1,%rt;\n");
    }
    s += "    fma.rn.f32 %l0,%l0,%corr0,%psum0;\n    fma.rn.f32 %l1,%l1,%corr1,%psum1;\n";
    s += "    mov.f32 %m0,%mnew0;\n    mov.f32 %m1,%mnew1;\n";

    // 4. O += P.V : V read from SMEM at bufc+ksz.
    if pv_ldmatrix {
        // `ldmatrix.x2.trans` gathers the PV B=V fragment in ONE warp-collective instruction. V is staged
        // row-major [key][hdim]; the mma wants it col-major [hdim][key], so `.trans` does the 8×8 transpose
        // in hardware. This replaces the hand-packed gather below — nto×(4 `ld.shared.u16` + 2 shl + 2 or)
        // whose key-strided addresses 8-way bank-conflict (lanes of one grp read keys tg2=0,2,4,6 at a fixed
        // hdim ⇒ same bank) — the documented "strided SMEM V-load" ceiling. Each of lanes 0..15 supplies the
        // address of one key row (key = lane&15, 8 contiguous hdim at nt*8); lanes 16..31 alias 0..15 (the
        // x2 form ignores their address but they still participate). Output {%b0,%b1} is exactly the mma B
        // fragment, so the PV mma is unchanged. Bit-identical to the hand path up to nothing (same values).
        for nt in 0..nto {
            s += &format!("    and.b32 %idx,%lane,15;\n    mul.lo.u32 %tmp,%idx,{d};\n    add.u32 %tmp,%tmp,{};\n    shl.b32 %tmp,%tmp,1;\n", nt * 8);
            s += &format!("    mov.u32 %sbase,{sym};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,{ksz};\n    add.u32 %sbase,%sbase,%tmp;\n");
            s += "    ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%b0,%b1},[%sbase];\n";
            s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
        }
    } else {
        for nt in 0..nto {
            s += &format!("    add.u32 %idx,%grp,{};\n", nt * 8);
            // base = &smemV[key=tg2][hdim=idx] = smem + bufc + ksz + (tg2*d + idx)*2
            s += &format!("    mul.lo.u32 %tmp,%tg2,{d};\n    add.u32 %tmp,%tmp,%idx;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,{sym};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,{ksz};\n    add.u32 %sbase,%sbase,%tmp;\n");
            // b0: keys tg2, tg2+1 (the next key is +d elements = +2d bytes in the staged [key][hdim] slab)
            s += &format!("    ld.shared.u16 %h0,[%sbase];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b0,%h0,%h1;\n", 2 * d);
            // b1: keys tg2+8, tg2+9
            s += &format!("    ld.shared.u16 %h0,[%sbase+{}];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b1,%h0,%h1;\n", 16 * d, 18 * d);
            s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
        }
    }

    // advance to the next 16-key block.
    if single_buf {
        // Single buffer: no swap, no precomputed %next — just step kb by 16.
        s += &format!("    add.u32 %kb,%kb,16;\n    bra KB_{name};\n");
    } else if stages == 2 {
        // Double buffer: swap current/prefetch, kb = next (already prefetched into the new bufc).
        s += "    mov.u32 %bswap,%bufc;\n    mov.u32 %bufc,%bufp;\n    mov.u32 %bufp,%bswap;\n";
        s += &format!("    mov.u32 %kb,%next;\n    bra KB_{name};\n");
    } else {
        // Deep ring: both cursors step ONE slot and wrap at the end of the ring — never the two-buffer
        // XOR/swap, which would only ever cycle two of the `stages` buffers. `%next` is `stages-1`
        // blocks ahead, so `kb` advances by 16 on its own (`%bswap` goes unused at this depth).
        let ring = stages * bufsz;
        for reg in ["%bufc", "%bufp"] {
            s += &format!("    add.u32 {reg},{reg},{bufsz};\n    setp.ge.u32 %p0,{reg},{ring};\n    @%p0 sub.u32 {reg},{reg},{ring};\n");
        }
        s += &format!("    add.u32 %kb,%kb,16;\n    bra KB_{name};\n");
    }

    // store O[row][c] = o / l_row (identical to entry_mma_reg).
    s += &format!("DONE_{name}:\n");
    for nt in 0..nto {
        s += &format!("    div.rn.f32 %o{nt}_0,%o{nt}_0,%l0;\n    div.rn.f32 %o{nt}_1,%o{nt}_1,%l0;\n    div.rn.f32 %o{nt}_2,%o{nt}_2,%l1;\n    div.rn.f32 %o{nt}_3,%o{nt}_3,%l1;\n");
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_0;\n    st.global.f32 [%base+4],%o{nt}_1;\n", nt * 8);
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_2;\n    st.global.f32 [%base+4],%o{nt}_3;\n", nt * 8);
    }
    s += "    ret;\n}\n";
    s
}

/// **Software-pipelined** register-resident `mma.sync` flash (`flash_d{d}_msp`, non-causal). Same
/// online-softmax math and one-warp-per-16-query-row layout as [`entry_mma_reg_pipe`], but it hoists the
/// QKᵀ of tile *i+1* to issue **before — and so overlap with — the softmax of tile *i***. In the base
/// kernel each tile is strictly QKᵀ → softmax → PV, so during the softmax (the `ex2.approx`/`shfl` SFU
/// sequence — the documented ~13 TFLOP/s plateau) the tensor cores idle: there is no independent `mma` to
/// issue. QKᵀ(i+1) depends only on K(i+1), not on softmax(i), so emitting it just before softmax(i) gives
/// `ptxas` a tensor-core stream to hide the SFU latency behind.
///
/// The enabler is **separate K and V pipeline pools** (2 buffers each, `4·ksz` SMEM — the SAME footprint
/// as the base kernel's 2-slab double-buffer, so **occupancy is unchanged**; this is what distinguishes
/// it from the occupancy-losing `mp4`/`mpw` experiments). K is staged **one tile further ahead** than V
/// (`K(i+2)` + `V(i+1)` per step, in one `cp.async` group) so that at step *i*: QKᵀ-ahead reads a ready
/// `K(i+1)` (in `%kahead`), PV reads `V(i)` (in `%vcur`), and the `K(i+2)`/`V(i+1)` prefetch is in flight —
/// the `cp.async.wait_group 1` "1 group in flight at step start" invariant is exactly the base kernel's.
/// Buffer offsets toggle by `XOR ksz` (ksz is a power of two; the V pool base `2·ksz` has the ksz bit
/// clear, so the toggle stays within each pool). Score registers rotate `%sn`→`%s`. Gated vs `ref_attn`.
///
/// **MEASURED NEGATIVE RESULT (kept as a documented A/B — `flash_sp_vs_mp`).** Clock-cancelled `sp/base`
/// (median of 9, H=8) is a **wash-to-slight-loss**: `1.043 / 1.044 / 1.010 / 1.019` at S=512/1024/2048/
/// 4096. The manual QKᵀ-ahead did not beat the softmax stall — either `ptxas` already extracts what little
/// mma/SFU overlap one warp affords, or the rotation `mov`s + separate-pool address arithmetic cost as much
/// as the overlap saves. So the long-S ceiling is not closable by software-pipelining a single warp's QKᵀ;
/// it joins `mp4`/`mpw` as a measured occupancy/overlap negative. Non-causal; not extended, given the wash.
fn entry_mma_reg_pipe_sp(d: usize) -> String {
    assert!(d % 16 == 0, "mma flash needs D % 16 == 0");
    let ktq = d / 16;
    let nto = d / 8;
    let ksz = 16 * d * 2; // one tensor's 16-key slab (K slab == V slab)
    let cpl = d / 16; // 16-byte cp.async chunks per lane per tensor
    let vbase = 2 * ksz; // V pool starts after the two K buffers
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = format!("flash_d{d}_msp");

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0,%pa,%ppf2;\n";
    let mut fr = String::from(
        "%scale,%m0,%m1,%mnew0,%mnew1,%corr0,%corr1,%l0,%l1,%lmax0,%lmax1,%rt,%psum0,%psum1,%pp,%tp0,%tp1,%tp2,%tp3",
    );
    for nk in 0..2 {
        for r in 0..4 {
            fr += &format!(",%s{nk}_{r},%sn{nk}_{r}");
        }
    }
    for nt in 0..nto {
        for r in 0..4 {
            fr += &format!(",%o{nt}_{r}");
        }
    }
    s += &format!("    .reg .f32 {fr};\n");
    let mut br = String::new();
    for kt in 0..ktq {
        for r in 0..4 {
            br += &format!("%qa{kt}_{r},");
        }
    }
    br += "%a0,%a1,%a2,%a3,%b0,%b1,%h0,%h1,";
    s += &format!(
        "    .reg .b32 {br}%S,%lane,%grp,%tg,%tg2,%row,%qr0,%qr1,%kb,%idx,%tmp,%hoff,%kahead,%vcur,%kdst,%vdst,%next1,%next2,%sbase,%sk,%chunk,%lkey;\n"
    );
    s += "    .reg .b64 %Q,%K,%V,%O,%base,%off;\n";
    s += &format!("    .shared .align 16 .b8 smem_{name}[{}];\n", 4 * ksz);

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    s += &format!("    mov.u32 %hoff,%ctaid.y;\n    mul.lo.u32 %hoff,%hoff,%S;\n    mul.lo.u32 %hoff,%hoff,{d};\n");
    s += "    mul.wide.u32 %off,%hoff,2;\n    add.s64 %Q,%Q,%off;\n    add.s64 %K,%K,%off;\n    add.s64 %V,%V,%off;\n";
    s += "    mul.wide.u32 %off,%hoff,4;\n    add.s64 %O,%O,%off;\n";
    s += "    mov.u32 %lane,%tid.x;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg2,%tg,1;\n";
    s += "    mov.u32 %row,%ctaid.x;\n    shl.b32 %row,%row,4;\n    add.u32 %qr0,%row,%grp;\n    add.u32 %qr1,%qr0,8;\n";

    // Q A-fragments (once; reused across all key tiles).
    for kt in 0..ktq {
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_0,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_2,[%base];\n");
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_1,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_3,[%base];\n");
    }
    for nt in 0..nto {
        for r in 0..4 {
            s += &format!("    mov.f32 %o{nt}_{r},0f00000000;\n");
        }
    }
    s += "    mov.f32 %m0,0fFF800000;\n    mov.f32 %m1,0fFF800000;\n    mov.f32 %l0,0f00000000;\n    mov.f32 %l1,0f00000000;\n";

    // Stage one tensor's 16-key slab (global tile `kbreg`) into SMEM byte-offset `dstreg`.
    let stage_one = |gptr: &str, kbreg: &str, dstreg: &str| -> String {
        let mut t = String::new();
        for ci in 0..cpl {
            t += &format!("    add.u32 %chunk,%lane,{};\n", ci * 32);
            t += &format!("    mul.lo.u32 %tmp,{kbreg},{d};\n    shl.b32 %sk,%chunk,3;\n    add.u32 %tmp,%tmp,%sk;\n    mul.wide.u32 %off,%tmp,2;\n");
            t += &format!("    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,{dstreg};\n    shl.b32 %sk,%chunk,4;\n    add.u32 %sbase,%sbase,%sk;\n");
            t += &format!("    add.s64 %base,{gptr},%off;\n    cp.async.cg.shared.global [%sbase],[%base],16;\n");
        }
        t
    };

    // QKᵀ of one key tile into score-reg set `sreg` ("s" or "sn"), K read from SMEM offset `kbufreg`.
    let qkt = |sreg: &str, kbufreg: &str| -> String {
        let mut t = String::new();
        for nk in 0..2 {
            for r in 0..4 {
                t += &format!("    mov.f32 %{sreg}{nk}_{r},0f00000000;\n");
            }
            t += &format!("    add.u32 %lkey,%grp,{};\n", nk * 8);
            for kt in 0..ktq {
                t += &format!("    mul.lo.u32 %tmp,%lkey,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,{kbufreg};\n    add.u32 %sbase,%sbase,%tmp;\n", kt * 16);
                t += "    ld.shared.b32 %b0,[%sbase];\n    ld.shared.b32 %b1,[%sbase+16];\n";
                t += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%{sreg}{nk}_0,%{sreg}{nk}_1,%{sreg}{nk}_2,%{sreg}{nk}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%{sreg}{nk}_0,%{sreg}{nk}_1,%{sreg}{nk}_2,%{sreg}{nk}_3}};\n");
            }
        }
        t
    };

    // PROLOGUE: stage K(0)→Kbuf0; wait; QKᵀ(0)→%s. Then stage K(16)→Kbuf1 (if it exists) + V(0)→Vbuf0,
    // one group — the first "in flight at step start" group the loop invariant expects.
    s += "    mov.u32 %kb,0;\n";
    s += &stage_one("%K", "%kb", "0");
    s += "    cp.async.commit_group;\n    cp.async.wait_group 0;\n    bar.sync 0;\n";
    s += "    mov.u32 %kahead,0;\n";
    s += &qkt("s", "%kahead");
    s += &format!("    mov.u32 %vcur,{vbase};\n    mov.u32 %kahead,{ksz};\n");
    s += &format!(
        "    mov.u32 %next1,16;\n    setp.lt.u32 %p0,%next1,%S;\n    @!%p0 bra PNK1_{name};\n"
    );
    s += &stage_one("%K", "%next1", "%kahead");
    s += &format!("PNK1_{name}:\n");
    s += &stage_one("%V", "%kb", "%vcur");
    s += "    cp.async.commit_group;\n";

    // MAIN LOOP over tile i (kb = 16·i): consume tile i (softmax+PV from %s / %vcur), compute QKᵀ(i+1)
    // ahead into %sn (from %kahead), prefetch K(i+2)/V(i+1).
    s += &format!("LOOP_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra DONE_{name};\n");
    s += "    add.u32 %next1,%kb,16;\n    add.u32 %next2,%kb,32;\n";
    s += "    setp.lt.u32 %pa,%next1,%S;\n    setp.lt.u32 %ppf2,%next2,%S;\n";
    s += &format!("    xor.b32 %kdst,%kahead,{ksz};\n    xor.b32 %vdst,%vcur,{ksz};\n");
    // prefetch K(i+2) into the other K buffer (if exists) and V(i+1) into the other V buffer (if exists).
    s += &format!("    @!%ppf2 bra SKPK_{name};\n");
    s += &stage_one("%K", "%next2", "%kdst");
    s += &format!("SKPK_{name}:\n    @!%pa bra SKPV_{name};\n");
    s += &stage_one("%V", "%next1", "%vdst");
    s += &format!(
        "SKPV_{name}:\n    @!%pa bra NOCM_{name};\n    cp.async.commit_group;\nNOCM_{name}:\n"
    );
    // wait: not-last ⇒ 2 groups in flight, wait_group 1 (drain tile i's group); last ⇒ 1, wait_group 0.
    s += &format!("    @%pa bra W1_{name};\n    cp.async.wait_group 0;\n    bra WD_{name};\nW1_{name}:\n    cp.async.wait_group 1;\nWD_{name}:\n    bar.sync 0;\n");
    // QKᵀ-ahead (tile i+1) → %sn, BEFORE softmax(i) so the score mma overlaps the softmax SFU work.
    s += &format!("    @!%pa bra SKQK_{name};\n");
    s += &qkt("sn", "%kahead");
    s += &format!("SKQK_{name}:\n");

    // softmax(i) on %s — identical to entry_mma_reg_pipe.
    s += "    max.f32 %lmax0,%s0_0,%s0_1;\n    max.f32 %lmax0,%lmax0,%s1_0;\n    max.f32 %lmax0,%lmax0,%s1_1;\n    mul.f32 %lmax0,%lmax0,%scale;\n";
    s += "    max.f32 %lmax1,%s0_2,%s0_3;\n    max.f32 %lmax1,%lmax1,%s1_2;\n    max.f32 %lmax1,%lmax1,%s1_3;\n    mul.f32 %lmax1,%lmax1,%scale;\n";
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax0,{off},0x1f,0xffffffff;\n    max.f32 %lmax0,%lmax0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax1,{off},0x1f,0xffffffff;\n    max.f32 %lmax1,%lmax1,%rt;\n");
    }
    s += &format!("    max.f32 %mnew0,%m0,%lmax0;\n    sub.f32 %corr0,%m0,%mnew0;\n    mul.f32 %corr0,%corr0,{log2e};\n    ex2.approx.f32 %corr0,%corr0;\n");
    s += &format!("    max.f32 %mnew1,%m1,%lmax1;\n    sub.f32 %corr1,%m1,%mnew1;\n    mul.f32 %corr1,%corr1,{log2e};\n    ex2.approx.f32 %corr1,%corr1;\n");
    for nt in 0..nto {
        s += &format!("    mul.f32 %o{nt}_0,%o{nt}_0,%corr0;\n    mul.f32 %o{nt}_1,%o{nt}_1,%corr0;\n    mul.f32 %o{nt}_2,%o{nt}_2,%corr1;\n    mul.f32 %o{nt}_3,%o{nt}_3,%corr1;\n");
    }
    let prob = |dst: &str, sreg: &str, mnew: &str| -> String {
        format!("    mul.f32 %pp,{sreg},%scale;\n    sub.f32 %pp,%pp,{mnew};\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 {dst},%pp;\n")
    };
    let pack = |dst: &str, lo: &str, hi: &str| -> String {
        format!("    cvt.rn.f16.f32 %h0,{lo};\n    and.b32 %h0,%h0,65535;\n    cvt.rn.f16.f32 %h1,{hi};\n    shl.b32 %h1,%h1,16;\n    or.b32 {dst},%h0,%h1;\n")
    };
    s += &prob("%tp0", "%s0_0", "%mnew0");
    s += &prob("%tp1", "%s0_1", "%mnew0");
    s += &prob("%tp2", "%s1_0", "%mnew0");
    s += &prob("%tp3", "%s1_1", "%mnew0");
    s += "    add.f32 %psum0,%tp0,%tp1;\n    add.f32 %psum0,%psum0,%tp2;\n    add.f32 %psum0,%psum0,%tp3;\n";
    s += &pack("%a0", "%tp0", "%tp1");
    s += &pack("%a2", "%tp2", "%tp3");
    s += &prob("%tp0", "%s0_2", "%mnew1");
    s += &prob("%tp1", "%s0_3", "%mnew1");
    s += &prob("%tp2", "%s1_2", "%mnew1");
    s += &prob("%tp3", "%s1_3", "%mnew1");
    s += "    add.f32 %psum1,%tp0,%tp1;\n    add.f32 %psum1,%psum1,%tp2;\n    add.f32 %psum1,%psum1,%tp3;\n";
    s += &pack("%a1", "%tp0", "%tp1");
    s += &pack("%a3", "%tp2", "%tp3");
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum0,{off},0x1f,0xffffffff;\n    add.f32 %psum0,%psum0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum1,{off},0x1f,0xffffffff;\n    add.f32 %psum1,%psum1,%rt;\n");
    }
    s += "    fma.rn.f32 %l0,%l0,%corr0,%psum0;\n    fma.rn.f32 %l1,%l1,%corr1,%psum1;\n";
    s += "    mov.f32 %m0,%mnew0;\n    mov.f32 %m1,%mnew1;\n";

    // PV(i): V read from SMEM at %vcur (separate V pool — no +ksz offset).
    for nt in 0..nto {
        s += &format!("    add.u32 %idx,%grp,{};\n", nt * 8);
        s += &format!("    mul.lo.u32 %tmp,%tg2,{d};\n    add.u32 %tmp,%tmp,%idx;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%vcur;\n    add.u32 %sbase,%sbase,%tmp;\n");
        s += &format!("    ld.shared.u16 %h0,[%sbase];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b0,%h0,%h1;\n", 2 * d);
        s += &format!("    ld.shared.u16 %h0,[%sbase+{}];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b1,%h0,%h1;\n", 16 * d, 18 * d);
        s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
    }

    // rotate scores %s ← %sn (the ahead tile becomes the consume tile), toggle buffers, advance.
    s += &format!("    @!%pa bra SKROT_{name};\n");
    for nk in 0..2 {
        for r in 0..4 {
            s += &format!("    mov.f32 %s{nk}_{r},%sn{nk}_{r};\n");
        }
    }
    s += &format!("SKROT_{name}:\n");
    s += &format!("    xor.b32 %kahead,%kahead,{ksz};\n    xor.b32 %vcur,%vcur,{ksz};\n    add.u32 %kb,%kb,16;\n    bra LOOP_{name};\n");

    // store O[row][c] = o / l_row.
    s += &format!("DONE_{name}:\n");
    for nt in 0..nto {
        s += &format!("    div.rn.f32 %o{nt}_0,%o{nt}_0,%l0;\n    div.rn.f32 %o{nt}_1,%o{nt}_1,%l0;\n    div.rn.f32 %o{nt}_2,%o{nt}_2,%l1;\n    div.rn.f32 %o{nt}_3,%o{nt}_3,%l1;\n");
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_0;\n    st.global.f32 [%base+4],%o{nt}_1;\n", nt * 8);
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_2;\n    st.global.f32 [%base+4],%o{nt}_3;\n", nt * 8);
    }
    s += "    ret;\n}\n";
    s
}

/// **Head-dim warp-split** flash (`flash_d{d}_hs`, non-causal, intended for D=128). D=128's
/// register-resident kernel plateaus at ~8 TFLOP/s — *half* of D=64's ~16 — because its **64 f32
/// O-accumulators** (nto=16) plus the 16 KB double-buffer cap it at ~6 warps/SM (~12.5% occupancy): the
/// tensor cores starve for warps. This kernel runs **2 warps / CTA on the SAME 16 query rows**, each warp
/// owning **half the output head dim** (warp `w` owns hdim `[w·d/2, w·d/2 + d/2)`). Both warps redundantly
/// compute the full QKᵀ + softmax (the score is tiny and both read the same staged K), but each does only
/// **half the PV**, so each holds **32 O-accumulators, not 64** — halving the binding register pressure.
/// With the 16 KB K/V buffer now shared by the 2 warps (not duplicated), occupancy ≈ doubles toward the
/// ~12 warps/SM that lifted D=64; that gap *is* the documented D=128 bound. The two warps cooperatively
/// stage the full K+V slab (64-thread `cp.async`) and `bar.sync` on it before reading, and `bar.sync`
/// again before the prefetch overwrites the buffer. The redundant QKᵀ is the price; the occupancy is the
/// bet. Gated vs `ref_attn`; A/B'd same-run vs `flash_d{d}_mp` by `flash_hs_vs_mp` (isolates the occupancy
/// effect — both hand-packed). Non-causal.
///
/// **MEASURED NEGATIVE RESULT (kept as a documented A/B — `flash_hs_vs_mp`).** Clock-cancelled `hs/mp`
/// (median of 9, H=8, D=128) = **1.110 / 1.108 / 1.199 / 1.193** at S=512/1024/2048/4096 — the split is
/// **11–20% SLOWER**. Occupancy *did* roughly double (SMEM-bound 6→12 warps/SM, verified by the halved
/// O-accumulators), but the bet **lost**: the redundant full QKᵀ (both warps recompute the whole score)
/// plus the two per-tile `bar.sync`s cost more than the extra warps buy. So D=128's plateau is **not**
/// occupancy-bound the way the register count suggested — doubling warps does not help. It joins
/// `mp4`/`mpw`/`msp` as a measured occupancy/overlap negative; the contraction-split (no redundant QKᵀ,
/// but a cross-warp partial-score exchange) is not pursued given this evidence. The banked D=128 win
/// stays the `ldmatrix` SMEM-feed (`flash_d128_mp_lm`), which beats cutlass-efficient ≤1024.
fn entry_mma_reg_pipe_hs(d: usize) -> String {
    assert!(d % 32 == 0, "head-split needs (d/2) % 16 == 0");
    let ktq = d / 16; // full QKᵀ contraction tiles (both warps compute these, redundantly)
    let hh = d / 2; // head-dim half each warp owns
    let nto = hh / 8; // PV n-tiles per warp (half the output)
    let ksz = 16 * d * 2; // full K slab (== V slab)
    let bufsz = 2 * ksz; // K+V per pipeline buffer
    let cpl = 2 * d / 64; // 16-byte cp.async chunks per thread per tensor over 64 threads
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = format!("flash_d{d}_hs");

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0,%pnext;\n";
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
    let mut br = String::new();
    for kt in 0..ktq {
        for r in 0..4 {
            br += &format!("%qa{kt}_{r},");
        }
    }
    br += "%a0,%a1,%a2,%a3,%b0,%b1,%hr0,%hr1,";
    s += &format!(
        "    .reg .b32 {br}%S,%tix,%lane,%warpid,%grp,%tg,%tg2,%row,%qr0,%qr1,%hbase,%kb,%idx,%tmp,%hoff,%bufc,%bufp,%next,%sbase,%sk,%chunk,%lkey,%bswap;\n"
    );
    s += "    .reg .b64 %Q,%K,%V,%O,%base,%off;\n";
    s += &format!("    .shared .align 16 .b8 smem_{name}[{}];\n", 2 * bufsz);

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    s += &format!("    mov.u32 %hoff,%ctaid.y;\n    mul.lo.u32 %hoff,%hoff,%S;\n    mul.lo.u32 %hoff,%hoff,{d};\n");
    s += "    mul.wide.u32 %off,%hoff,2;\n    add.s64 %Q,%Q,%off;\n    add.s64 %K,%K,%off;\n    add.s64 %V,%V,%off;\n";
    s += "    mul.wide.u32 %off,%hoff,4;\n    add.s64 %O,%O,%off;\n";
    // tid → warpid/lane; lane → grp/tg/tg2. hbase = warpid·(d/2) (this warp's output hdim base).
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpid,%tix,5;\n    and.b32 %lane,%tix,31;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg2,%tg,1;\n";
    s += &format!("    mul.lo.u32 %hbase,%warpid,{hh};\n");
    s += "    mov.u32 %row,%ctaid.x;\n    shl.b32 %row,%row,4;\n    add.u32 %qr0,%row,%grp;\n    add.u32 %qr1,%qr0,8;\n";

    // FULL Q A-fragments (both warps load all d — needed for the full QKᵀ contraction).
    for kt in 0..ktq {
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_0,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_2,[%base];\n");
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_1,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_3,[%base];\n");
    }
    for nt in 0..nto {
        for r in 0..4 {
            s += &format!("    mov.f32 %o{nt}_{r},0f00000000;\n");
        }
    }
    s += "    mov.f32 %m0,0fFF800000;\n    mov.f32 %m1,0fFF800000;\n    mov.f32 %l0,0f00000000;\n    mov.f32 %l1,0f00000000;\n";

    // Cooperative stage (64 threads) of the full K+V slab at global tile `kbreg` into buffer `bufreg`.
    let stage = |kbreg: &str, bufreg: &str| -> String {
        let mut t = String::new();
        for ci in 0..cpl {
            t += &format!("    add.u32 %chunk,%tix,{};\n", ci * 64);
            t += &format!("    mul.lo.u32 %tmp,{kbreg},{d};\n    shl.b32 %sk,%chunk,3;\n    add.u32 %tmp,%tmp,%sk;\n    mul.wide.u32 %off,%tmp,2;\n");
            t += &format!("    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,{bufreg};\n    shl.b32 %sk,%chunk,4;\n    add.u32 %sbase,%sbase,%sk;\n");
            t += "    add.s64 %base,%K,%off;\n    cp.async.cg.shared.global [%sbase],[%base],16;\n";
            t += &format!("    add.u32 %sbase,%sbase,{ksz};\n    add.s64 %base,%V,%off;\n    cp.async.cg.shared.global [%sbase],[%base],16;\n");
        }
        t
    };

    s += &format!("    mov.u32 %bufc,0;\n    mov.u32 %bufp,{bufsz};\n    mov.u32 %kb,0;\n");
    s += &stage("%kb", "%bufc");
    s += "    cp.async.commit_group;\n";

    s += &format!("KB_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra DONE_{name};\n");
    s += "    add.u32 %next,%kb,16;\n    setp.ge.u32 %pnext,%next,%S;\n";
    s += &format!("    @%pnext bra NOPF_{name};\n");
    s += &stage("%next", "%bufp");
    s += &format!(
        "    cp.async.commit_group;\n    cp.async.wait_group 1;\n    bra PFDONE_{name};\n"
    );
    s += &format!("NOPF_{name}:\n    cp.async.wait_group 0;\n");
    s += &format!("PFDONE_{name}:\n    bar.sync 0;\n");

    // 1. S = Q.Kt (FULL contraction, both warps) — K read from SMEM at bufc.
    for nk in 0..2 {
        for r in 0..4 {
            s += &format!("    mov.f32 %s{nk}_{r},0f00000000;\n");
        }
        s += &format!("    add.u32 %lkey,%grp,{};\n", nk * 8);
        for kt in 0..ktq {
            s += &format!("    mul.lo.u32 %tmp,%lkey,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,%tmp;\n", kt * 16);
            s += "    ld.shared.b32 %b0,[%sbase];\n    ld.shared.b32 %b1,[%sbase+16];\n";
            s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}};\n");
        }
    }

    // 2. online softmax (full, both warps identical) — copied from entry_mma_reg_pipe.
    s += "    max.f32 %lmax0,%s0_0,%s0_1;\n    max.f32 %lmax0,%lmax0,%s1_0;\n    max.f32 %lmax0,%lmax0,%s1_1;\n    mul.f32 %lmax0,%lmax0,%scale;\n";
    s += "    max.f32 %lmax1,%s0_2,%s0_3;\n    max.f32 %lmax1,%lmax1,%s1_2;\n    max.f32 %lmax1,%lmax1,%s1_3;\n    mul.f32 %lmax1,%lmax1,%scale;\n";
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax0,{off},0x1f,0xffffffff;\n    max.f32 %lmax0,%lmax0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax1,{off},0x1f,0xffffffff;\n    max.f32 %lmax1,%lmax1,%rt;\n");
    }
    s += &format!("    max.f32 %mnew0,%m0,%lmax0;\n    sub.f32 %corr0,%m0,%mnew0;\n    mul.f32 %corr0,%corr0,{log2e};\n    ex2.approx.f32 %corr0,%corr0;\n");
    s += &format!("    max.f32 %mnew1,%m1,%lmax1;\n    sub.f32 %corr1,%m1,%mnew1;\n    mul.f32 %corr1,%corr1,{log2e};\n    ex2.approx.f32 %corr1,%corr1;\n");
    for nt in 0..nto {
        s += &format!("    mul.f32 %o{nt}_0,%o{nt}_0,%corr0;\n    mul.f32 %o{nt}_1,%o{nt}_1,%corr0;\n    mul.f32 %o{nt}_2,%o{nt}_2,%corr1;\n    mul.f32 %o{nt}_3,%o{nt}_3,%corr1;\n");
    }
    let prob = |dst: &str, sreg: &str, mnew: &str| -> String {
        format!("    mul.f32 %pp,{sreg},%scale;\n    sub.f32 %pp,%pp,{mnew};\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 {dst},%pp;\n")
    };
    let pack = |dst: &str, lo: &str, hi: &str| -> String {
        format!("    cvt.rn.f16.f32 %hr0,{lo};\n    and.b32 %hr0,%hr0,65535;\n    cvt.rn.f16.f32 %hr1,{hi};\n    shl.b32 %hr1,%hr1,16;\n    or.b32 {dst},%hr0,%hr1;\n")
    };
    s += &prob("%tp0", "%s0_0", "%mnew0");
    s += &prob("%tp1", "%s0_1", "%mnew0");
    s += &prob("%tp2", "%s1_0", "%mnew0");
    s += &prob("%tp3", "%s1_1", "%mnew0");
    s += "    add.f32 %psum0,%tp0,%tp1;\n    add.f32 %psum0,%psum0,%tp2;\n    add.f32 %psum0,%psum0,%tp3;\n";
    s += &pack("%a0", "%tp0", "%tp1");
    s += &pack("%a2", "%tp2", "%tp3");
    s += &prob("%tp0", "%s0_2", "%mnew1");
    s += &prob("%tp1", "%s0_3", "%mnew1");
    s += &prob("%tp2", "%s1_2", "%mnew1");
    s += &prob("%tp3", "%s1_3", "%mnew1");
    s += "    add.f32 %psum1,%tp0,%tp1;\n    add.f32 %psum1,%psum1,%tp2;\n    add.f32 %psum1,%psum1,%tp3;\n";
    s += &pack("%a1", "%tp0", "%tp1");
    s += &pack("%a3", "%tp2", "%tp3");
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum0,{off},0x1f,0xffffffff;\n    add.f32 %psum0,%psum0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum1,{off},0x1f,0xffffffff;\n    add.f32 %psum1,%psum1,%rt;\n");
    }
    s += "    fma.rn.f32 %l0,%l0,%corr0,%psum0;\n    fma.rn.f32 %l1,%l1,%corr1,%psum1;\n";
    s += "    mov.f32 %m0,%mnew0;\n    mov.f32 %m1,%mnew1;\n";

    // 4. O += P.V : HALF the PV — V columns hdim ∈ [hbase, hbase+hh). V read from SMEM at bufc+ksz.
    for nt in 0..nto {
        s += &format!(
            "    add.u32 %idx,%grp,{};\n    add.u32 %idx,%idx,%hbase;\n",
            nt * 8
        );
        s += &format!("    mul.lo.u32 %tmp,%tg2,{d};\n    add.u32 %tmp,%tmp,%idx;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,{ksz};\n    add.u32 %sbase,%sbase,%tmp;\n");
        s += &format!("    ld.shared.u16 %hr0,[%sbase];\n    ld.shared.u16 %hr1,[%sbase+{}];\n    shl.b32 %hr1,%hr1,16;\n    or.b32 %b0,%hr0,%hr1;\n", 2 * d);
        s += &format!("    ld.shared.u16 %hr0,[%sbase+{}];\n    ld.shared.u16 %hr1,[%sbase+{}];\n    shl.b32 %hr1,%hr1,16;\n    or.b32 %b1,%hr0,%hr1;\n", 16 * d, 18 * d);
        s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
    }

    // 2-warp barrier before the next iteration's prefetch overwrites bufc.
    s += "    bar.sync 0;\n";
    s += "    mov.u32 %bswap,%bufc;\n    mov.u32 %bufc,%bufp;\n    mov.u32 %bufp,%bswap;\n";
    s += &format!("    mov.u32 %kb,%next;\n    bra KB_{name};\n");

    // store O[query][hbase + nt*8 + ...] = o / l.
    s += &format!("DONE_{name}:\n");
    for nt in 0..nto {
        s += &format!("    div.rn.f32 %o{nt}_0,%o{nt}_0,%l0;\n    div.rn.f32 %o{nt}_1,%o{nt}_1,%l0;\n    div.rn.f32 %o{nt}_2,%o{nt}_2,%l1;\n    div.rn.f32 %o{nt}_3,%o{nt}_3,%l1;\n");
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,%hbase;\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_0;\n    st.global.f32 [%base+4],%o{nt}_1;\n", nt * 8);
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,%hbase;\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_2;\n    st.global.f32 [%base+4],%o{nt}_3;\n", nt * 8);
    }
    s += "    ret;\n}\n";
    s
}

/// Generate the **multi-warp-CTA** `cp.async`-pipelined flash kernel (`flash_d{d}_mp{warps}`,
/// non-causal). Identical per-warp math to [`entry_mma_reg_pipe`] — each warp owns a 16-query-row block
/// and keeps O/m/l in registers — but `warps` warps share **one** CTA and **cooperatively stage a single
/// K/V block into shared memory, which all `warps` warps then read**. The single-warp `flash_d64_mp`
/// re-streams the entire K/V from global for every 16-query-row block; once `cp.async` has hidden the
/// per-block *latency*, the residual cost is that K/V *traffic volume*. Sharing one staged block across
/// `warps` query-row blocks cuts the K/V global/L2 traffic **`warps`×**, and folding `warps` warps into
/// one CTA shares the 8 KB double-buffer (vs 8 KB per single-warp CTA) so occupancy rises too.
///
/// Layout: `qblock = ctaid.x*warps + warpid`, `row = qblock*16`. The CTA's `32*warps` threads stage each
/// contiguous K+V slab (guarded `cp.async` over `ceil(2·d/(32·warps))` unrolled chunk passes); a single
/// `bar.sync` after the prefetch makes it visible to every warp. **No-deadlock:** the cooperative stage,
/// the `cp.async.wait_group`, the `bar.sync`, and the buffer swap run on *every* thread; only the per-warp
/// QKᵀ/softmax/PV and the final store are predicated on `active = row < S`, so a ragged final CTA (when
/// `warps ∤ S/16`) still stages and still hits the barrier. Non-causal only (a causal CTA would need its
/// warps to diverge on the key range, breaking the shared barrier); causal stays on `flash_d64_mpc`.
fn entry_mma_reg_pipe_mw(d: usize, warps: usize) -> String {
    assert!(d % 16 == 0, "mma flash needs D % 16 == 0");
    assert!(warps >= 1, "warps must be >= 1");
    let ktq = d / 16;
    let nto = d / 8;
    let ksz = 16 * d * 2; // bytes of one staged K block (== V block)
    let bufsz = 2 * ksz; // K+V per pipeline buffer
    let nthreads = 32 * warps;
    let nchunk = 16 * d / 8; // 16-byte cp.async chunks per tensor (= 2·d)
    let stage_iters = nchunk.div_ceil(nthreads); // unrolled, predicated chunk passes
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = format!("flash_d{d}_mp{warps}");

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0,%pnext,%active,%pc;\n";
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
    let mut br = String::new();
    for kt in 0..ktq {
        for r in 0..4 {
            br += &format!("%qa{kt}_{r},");
        }
    }
    br += "%a0,%a1,%a2,%a3,%b0,%b1,%h0,%h1,";
    s += &format!(
        "    .reg .b32 {br}%S,%tix,%lane,%warpid,%grp,%tg,%tg2,%qblock,%row,%qr0,%qr1,%kb,%gkey,%idx,%tmp,%hoff,%bufc,%bufp,%next,%sbase,%sk,%chunk,%lkey,%bswap;\n"
    );
    s += "    .reg .b64 %Q,%K,%V,%O,%base,%off;\n";
    s += &format!("    .shared .align 16 .b8 smem_{name}[{}];\n", 2 * bufsz);

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    // multi-head head base offset (ctaid.y), identical to entry_mma_reg.
    s += &format!("    mov.u32 %hoff,%ctaid.y;\n    mul.lo.u32 %hoff,%hoff,%S;\n    mul.lo.u32 %hoff,%hoff,{d};\n");
    s += "    mul.wide.u32 %off,%hoff,2;\n    add.s64 %Q,%Q,%off;\n    add.s64 %K,%K,%off;\n    add.s64 %V,%V,%off;\n";
    s += "    mul.wide.u32 %off,%hoff,4;\n    add.s64 %O,%O,%off;\n";
    // tid → warpid/lane; lane → grp/tg/tg2. qblock = ctaid.x*warps + warpid; row = qblock*16.
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpid,%tix,5;\n    and.b32 %lane,%tix,31;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg2,%tg,1;\n";
    s += &format!("    mov.u32 %qblock,%ctaid.x;\n    mul.lo.u32 %qblock,%qblock,{warps};\n    add.u32 %qblock,%qblock,%warpid;\n    shl.b32 %row,%qblock,4;\n    add.u32 %qr0,%row,%grp;\n    add.u32 %qr1,%qr0,8;\n    setp.lt.u32 %active,%row,%S;\n");

    // Load Q A-fragments once (active warps only — guards the global read).
    s += &format!("    @!%active bra SKIPQ_{name};\n");
    for kt in 0..ktq {
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_0,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_2,[%base];\n");
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_1,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_3,[%base];\n");
    }
    s += &format!("SKIPQ_{name}:\n");
    for nt in 0..nto {
        for r in 0..4 {
            s += &format!("    mov.f32 %o{nt}_{r},0f00000000;\n");
        }
    }
    s += "    mov.f32 %m0,0fFF800000;\n    mov.f32 %m1,0fFF800000;\n    mov.f32 %l0,0f00000000;\n    mov.f32 %l1,0f00000000;\n";

    // CTA-cooperative stage of the 16-key K+V slab at key index `kbreg` into buffer `bufreg` — all
    // `nthreads` threads, `stage_iters` unrolled chunk passes, each predicated `chunk < nchunk`.
    let stage = |kbreg: &str, bufreg: &str| -> String {
        let mut t = String::new();
        for ci in 0..stage_iters {
            t += &format!("    add.u32 %chunk,%tix,{};\n", ci * nthreads);
            t += &format!("    setp.lt.u32 %pc,%chunk,{nchunk};\n");
            t += &format!("    mul.lo.u32 %tmp,{kbreg},{d};\n    shl.b32 %sk,%chunk,3;\n    add.u32 %tmp,%tmp,%sk;\n    mul.wide.u32 %off,%tmp,2;\n");
            t += &format!("    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,{bufreg};\n    shl.b32 %sk,%chunk,4;\n    add.u32 %sbase,%sbase,%sk;\n");
            t += "    add.s64 %base,%K,%off;\n    @%pc cp.async.cg.shared.global [%sbase],[%base],16;\n";
            t += &format!("    add.u32 %sbase,%sbase,{ksz};\n    add.s64 %base,%V,%off;\n    @%pc cp.async.cg.shared.global [%sbase],[%base],16;\n");
        }
        t
    };

    s += &format!("    mov.u32 %bufc,0;\n    mov.u32 %bufp,{bufsz};\n    mov.u32 %kb,0;\n");
    s += &stage("%kb", "%bufc");
    s += "    cp.async.commit_group;\n";

    // for kb in 0..S step 16 (uniform across all warps; non-causal).
    s += &format!("KB_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra DONE_{name};\n");
    s += "    add.u32 %next,%kb,16;\n    setp.ge.u32 %pnext,%next,%S;\n";
    s += &format!("    @%pnext bra NOPF_{name};\n");
    s += &stage("%next", "%bufp");
    s += &format!(
        "    cp.async.commit_group;\n    cp.async.wait_group 1;\n    bra PFDONE_{name};\n"
    );
    s += &format!("NOPF_{name}:\n    cp.async.wait_group 0;\n");
    s += &format!("PFDONE_{name}:\n    bar.sync 0;\n");
    // per-warp compute (active warps only); inactive warps fall through to the swap + next iteration.
    s += &format!("    @!%active bra SKIPC_{name};\n");

    // 1. S = Q.Kt from shared K at bufc (local key = 8nk+grp).
    for nk in 0..2 {
        for r in 0..4 {
            s += &format!("    mov.f32 %s{nk}_{r},0f00000000;\n");
        }
        s += &format!("    add.u32 %lkey,%grp,{};\n", nk * 8);
        for kt in 0..ktq {
            s += &format!("    mul.lo.u32 %tmp,%lkey,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,%tmp;\n", kt * 16);
            s += "    ld.shared.b32 %b0,[%sbase];\n    ld.shared.b32 %b1,[%sbase+16];\n";
            s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}};\n");
        }
    }
    // 2. online softmax (register-resident) — identical to entry_mma_reg_pipe.
    s += "    max.f32 %lmax0,%s0_0,%s0_1;\n    max.f32 %lmax0,%lmax0,%s1_0;\n    max.f32 %lmax0,%lmax0,%s1_1;\n    mul.f32 %lmax0,%lmax0,%scale;\n";
    s += "    max.f32 %lmax1,%s0_2,%s0_3;\n    max.f32 %lmax1,%lmax1,%s1_2;\n    max.f32 %lmax1,%lmax1,%s1_3;\n    mul.f32 %lmax1,%lmax1,%scale;\n";
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax0,{off},0x1f,0xffffffff;\n    max.f32 %lmax0,%lmax0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax1,{off},0x1f,0xffffffff;\n    max.f32 %lmax1,%lmax1,%rt;\n");
    }
    s += &format!("    max.f32 %mnew0,%m0,%lmax0;\n    sub.f32 %corr0,%m0,%mnew0;\n    mul.f32 %corr0,%corr0,{log2e};\n    ex2.approx.f32 %corr0,%corr0;\n");
    s += &format!("    max.f32 %mnew1,%m1,%lmax1;\n    sub.f32 %corr1,%m1,%mnew1;\n    mul.f32 %corr1,%corr1,{log2e};\n    ex2.approx.f32 %corr1,%corr1;\n");
    for nt in 0..nto {
        s += &format!("    mul.f32 %o{nt}_0,%o{nt}_0,%corr0;\n    mul.f32 %o{nt}_1,%o{nt}_1,%corr0;\n    mul.f32 %o{nt}_2,%o{nt}_2,%corr1;\n    mul.f32 %o{nt}_3,%o{nt}_3,%corr1;\n");
    }
    let prob = |dst: &str, sreg: &str, mnew: &str| -> String {
        format!("    mul.f32 %pp,{sreg},%scale;\n    sub.f32 %pp,%pp,{mnew};\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 {dst},%pp;\n")
    };
    let pack = |dst: &str, lo: &str, hi: &str| -> String {
        format!("    cvt.rn.f16.f32 %h0,{lo};\n    and.b32 %h0,%h0,65535;\n    cvt.rn.f16.f32 %h1,{hi};\n    shl.b32 %h1,%h1,16;\n    or.b32 {dst},%h0,%h1;\n")
    };
    s += &prob("%tp0", "%s0_0", "%mnew0");
    s += &prob("%tp1", "%s0_1", "%mnew0");
    s += &prob("%tp2", "%s1_0", "%mnew0");
    s += &prob("%tp3", "%s1_1", "%mnew0");
    s += "    add.f32 %psum0,%tp0,%tp1;\n    add.f32 %psum0,%psum0,%tp2;\n    add.f32 %psum0,%psum0,%tp3;\n";
    s += &pack("%a0", "%tp0", "%tp1");
    s += &pack("%a2", "%tp2", "%tp3");
    s += &prob("%tp0", "%s0_2", "%mnew1");
    s += &prob("%tp1", "%s0_3", "%mnew1");
    s += &prob("%tp2", "%s1_2", "%mnew1");
    s += &prob("%tp3", "%s1_3", "%mnew1");
    s += "    add.f32 %psum1,%tp0,%tp1;\n    add.f32 %psum1,%psum1,%tp2;\n    add.f32 %psum1,%psum1,%tp3;\n";
    s += &pack("%a1", "%tp0", "%tp1");
    s += &pack("%a3", "%tp2", "%tp3");
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum0,{off},0x1f,0xffffffff;\n    add.f32 %psum0,%psum0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum1,{off},0x1f,0xffffffff;\n    add.f32 %psum1,%psum1,%rt;\n");
    }
    s += "    fma.rn.f32 %l0,%l0,%corr0,%psum0;\n    fma.rn.f32 %l1,%l1,%corr1,%psum1;\n";
    s += "    mov.f32 %m0,%mnew0;\n    mov.f32 %m1,%mnew1;\n";
    // 4. O += P.V from shared V at bufc+ksz.
    for nt in 0..nto {
        s += &format!("    add.u32 %idx,%grp,{};\n", nt * 8);
        s += &format!("    mul.lo.u32 %tmp,%tg2,{d};\n    add.u32 %tmp,%tmp,%idx;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,{ksz};\n    add.u32 %sbase,%sbase,%tmp;\n");
        s += &format!("    ld.shared.u16 %h0,[%sbase];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b0,%h0,%h1;\n", 2 * d);
        s += &format!("    ld.shared.u16 %h0,[%sbase+{}];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b1,%h0,%h1;\n", 16 * d, 18 * d);
        s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
    }
    // Second barrier: all warps must finish reading bufc before the NEXT iteration's prefetch overwrites
    // it (double-buffering gives one iteration of slack, but a fast warp could lap a slow one without
    // this). The single-warp kernels omit it — one warp is self-ordered; a multi-warp CTA is not.
    s += &format!("SKIPC_{name}:\n    bar.sync 0;\n");
    // swap buffers (all threads, uniform), advance kb.
    s += "    mov.u32 %bswap,%bufc;\n    mov.u32 %bufc,%bufp;\n    mov.u32 %bufp,%bswap;\n";
    s += &format!("    mov.u32 %kb,%next;\n    bra KB_{name};\n");

    // store O[row][c] = o / l_row (active warps only).
    s += &format!("DONE_{name}:\n    @!%active bra RET_{name};\n");
    for nt in 0..nto {
        s += &format!("    div.rn.f32 %o{nt}_0,%o{nt}_0,%l0;\n    div.rn.f32 %o{nt}_1,%o{nt}_1,%l0;\n    div.rn.f32 %o{nt}_2,%o{nt}_2,%l1;\n    div.rn.f32 %o{nt}_3,%o{nt}_3,%l1;\n");
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_0;\n    st.global.f32 [%base+4],%o{nt}_1;\n", nt * 8);
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_2;\n    st.global.f32 [%base+4],%o{nt}_3;\n", nt * 8);
    }
    s += &format!("RET_{name}:\n    ret;\n}}\n");
    s
}

/// **Wide-key-tile** `cp.async`-pipelined register-resident flash (`flash_d{d}_mpw{nkb}`, non-causal).
/// Identical online-softmax math to [`entry_mma_reg_pipe`] but processes **`BK = 16·nkb` keys per
/// online-softmax step** instead of 16. The narrow kernel pays the softmax *fixed* cost — two warp
/// `shfl` reductions (running max + denominator), the `ex2.approx` max-correction, and the rescale of
/// all `nto·4` O accumulators — **once per 16 keys**, and between the QKᵀ and PV `mma`s the tensor
/// cores idle through that scalar/SFU work (the measured ~9 TFLOP/s plateau vs cuDNN's scaling to 20).
/// Widening to `BK` keys amortises that fixed cost over `nkb×` more `mma`: one softmax over `2·nkb`
/// score n-tiles, one O rescale, one P pack of `nkb` k-tiles, then `nto·nkb` PV `mma`. The online
/// softmax is associative over any tile width, so O equals `flash_d{d}_mp` up to f32 rounding (gated
/// vs `ref_attn`). The staged K+V slab grows `nkb×` (`2·BK·d·2` bytes / double-buffer = 16 KB at
/// nkb=2, d=64); **requires `S % BK == 0`** (each block is full — no ragged-tail mask), which the WMMA
/// layer's `S % 64 == 0` already gives for nkb ≤ 4. Single warp / CTA (one 16-query-row block), so it
/// composes with the multi-warp lever orthogonally.
///
/// **MEASURED NEGATIVE RESULT (kept as a documented A/B — `flash_wide_vs_mp`).** Clock-cancelled
/// same-round ratios show wide **loses** to the narrow `mp`: `mpw2/mp ≈ 0.65–0.91×` at the realistic
/// multi-head regime, *worse* at large S (≈0.65× at H=8/S=4096). The softmax fixed cost was NOT the
/// binding constraint — the wider tile's extra score/P registers and `nkb×` SMEM cut occupancy
/// (fewer concurrent warps / CTAs-per-SM) by more than the amortisation saves, and the penalty
/// compounds over the longer K-loop. The real ceiling is occupancy + the strided SMEM V-load, not the
/// per-step softmax. So the production path stays `mp`/`mp4`; this kernel is retained only to localise
/// the comparison should a future GPU change the occupancy/softmax balance.
///
/// **Both stated mechanisms of that negative are budget-bound, and the dynamic-SMEM window moves the
/// budget** — which is why this generator now takes `stages`, `pv_ldmatrix` and `smem_budget`:
///   * *"`nkb×` SMEM cut occupancy"* was measured against a 48 KiB static ceiling on a 20-SM Ada part.
///     `nkb=4` at D=64 is 32 KiB double-buffered; the same tile keeps 2 CTAs/SM on A100 (164 KiB) and
///     3 on H100 (228 KiB) at depths this card cannot even declare (D6 §3.3).
///   * *"the strided SMEM V-load"* is exactly what `pv_ldmatrix` removes, and it was never wired into
///     the wide kernel — the narrow `_lm` measured +18–23% at D=128, where the V feed dominates. The
///     wide `_lm` path is the narrow one with the k-tile's key base (`16·kk·d`) folded into the
///     address: same instruction, same fragment order, one extra add.
///
/// `stages`/`smem_budget` mean exactly what they mean in [`entry_mma_reg_pipe`] (add+wrap ring at
/// depth ≥ 3, one committed `cp.async` group per body, `wait_group stages-1`, static `.shared` at or
/// below 48 KiB and the module-scope window beyond it, loud panic over budget). At `(stages, lm) ==
/// (2, false)` the emitted text is byte-identical to the shipped `flash_d{d}_mpw{nkb}`.
fn entry_mma_reg_pipe_wide(
    d: usize,
    nkb: usize,
    stages: usize,
    pv_ldmatrix: bool,
    smem_budget: usize,
) -> String {
    assert!(d % 16 == 0, "mma flash needs D % 16 == 0");
    assert!(nkb >= 1, "nkb must be >= 1");
    assert!(stages >= 2, "the wide flash ring needs >= 2 buffers");
    // The cooperative stage is an exact, unguarded `cpl = BK·D/256` 16-byte chunks per lane; a
    // non-integral count would silently leave the tail of every staged slab unwritten.
    assert!(
        (16 * nkb * d) % 256 == 0,
        "flash_d{d}_mpw{nkb}: BK*D must be a multiple of 256 (32 lanes x 16-byte cp.async chunks)"
    );
    let ktq = d / 16; // Q·Kt contraction tiles (over hdim)
    let nto = d / 8; // P·V output n-tiles (over hdim)
    let bk = 16 * nkb; // keys staged + softmaxed per step
    let ntile = 2 * nkb; // 8-key QKᵀ score n-tiles per step
    let ksz = bk * d * 2; // bytes of one staged K block (== one V block)
    let bufsz = 2 * ksz; // K+V slab per pipeline buffer
    let cpl = bk * d / 256; // 16-byte cp.async chunks per lane per tensor (bk*d*2/16/32)
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    // SMEM(stages, nkb) = stages * 2 * (16·nkb · d · 2) = 64 · stages · nkb · d — the family's closed
    // form, shared with the narrow kernel at nkb = 1.
    let smem_total = stages * bufsz;
    let mode = smem_mode_for(smem_total);
    let name = {
        let base = if pv_ldmatrix {
            format!("flash_d{d}_mpw{nkb}_lm")
        } else {
            format!("flash_d{d}_mpw{nkb}")
        };
        if stages == 2 {
            base
        } else {
            format!("{base}_s{stages}")
        }
    };
    assert!(
        smem_total <= smem_budget,
        "{name}: flash SMEM {smem_total} B (stages={stages}, BK={bk}, D={d}) exceeds the budget {smem_budget} B"
    );
    let sym = match mode {
        SmemMode::Static => format!("smem_{name}"),
        SmemMode::Dynamic(_) => DSMEM_SYM.to_string(),
    };

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0,%pnext;\n";
    let mut fr = String::from(
        "%scale,%m0,%m1,%mnew0,%mnew1,%corr0,%corr1,%l0,%l1,%lmax0,%lmax1,%rt,%psum0,%psum1,%pp,%tp0,%tp1,%tp2,%tp3",
    );
    for nt in 0..ntile {
        for r in 0..4 {
            fr += &format!(",%s{nt}_{r}");
        }
    }
    for nt in 0..nto {
        for r in 0..4 {
            fr += &format!(",%o{nt}_{r}");
        }
    }
    s += &format!("    .reg .f32 {fr};\n");
    let mut br = String::new();
    for kt in 0..ktq {
        for r in 0..4 {
            br += &format!("%qa{kt}_{r},");
        }
    }
    for kk in 0..nkb {
        for r in 0..4 {
            br += &format!("%pa{kk}_{r},");
        }
    }
    br += "%b0,%b1,%h0,%h1,";
    s += &format!(
        "    .reg .b32 {br}%S,%lane,%grp,%tg,%tg2,%row,%qr0,%qr1,%kb,%idx,%tmp,%hoff,%bufc,%bufp,%next,%sbase,%sk,%chunk,%lkey,%bswap;\n"
    );
    s += "    .reg .b64 %Q,%K,%V,%O,%base,%off;\n";
    if !mode.is_dynamic() {
        s += &format!("    .shared .align 16 .b8 {sym}[{smem_total}];\n");
    }

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    // multi-head head base offset (ctaid.y), identical to entry_mma_reg_pipe.
    s += &format!("    mov.u32 %hoff,%ctaid.y;\n    mul.lo.u32 %hoff,%hoff,%S;\n    mul.lo.u32 %hoff,%hoff,{d};\n");
    s += "    mul.wide.u32 %off,%hoff,2;\n    add.s64 %Q,%Q,%off;\n    add.s64 %K,%K,%off;\n    add.s64 %V,%V,%off;\n";
    s += "    mul.wide.u32 %off,%hoff,4;\n    add.s64 %O,%O,%off;\n";
    s += "    mov.u32 %lane,%tid.x;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg2,%tg,1;\n";
    s += "    mov.u32 %row,%ctaid.x;\n    shl.b32 %row,%row,4;\n    add.u32 %qr0,%row,%grp;\n    add.u32 %qr1,%qr0,8;\n";

    // Load Q A-fragments once (reused across every key block).
    for kt in 0..ktq {
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_0,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_2,[%base];\n");
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_1,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_3,[%base];\n");
    }
    for nt in 0..nto {
        for r in 0..4 {
            s += &format!("    mov.f32 %o{nt}_{r},0f00000000;\n");
        }
    }
    s += "    mov.f32 %m0,0fFF800000;\n    mov.f32 %m1,0fFF800000;\n    mov.f32 %l0,0f00000000;\n    mov.f32 %l1,0f00000000;\n";

    // cp.async stage of the BK-key K+V slab at global key index `kbreg` into buffer `bufreg`.
    let stage = |kbreg: &str, bufreg: &str| -> String {
        let mut t = String::new();
        for ci in 0..cpl {
            t += &format!("    add.u32 %chunk,%lane,{};\n", ci * 32);
            t += &format!("    mul.lo.u32 %tmp,{kbreg},{d};\n    shl.b32 %sk,%chunk,3;\n    add.u32 %tmp,%tmp,%sk;\n    mul.wide.u32 %off,%tmp,2;\n");
            t += &format!("    mov.u32 %sbase,{sym};\n    add.u32 %sbase,%sbase,{bufreg};\n    shl.b32 %sk,%chunk,4;\n    add.u32 %sbase,%sbase,%sk;\n");
            t += "    add.s64 %base,%K,%off;\n    cp.async.cg.shared.global [%sbase],[%base],16;\n";
            t += &format!("    add.u32 %sbase,%sbase,{ksz};\n    add.s64 %base,%V,%off;\n    cp.async.cg.shared.global [%sbase],[%base],16;\n");
        }
        t
    };

    if stages == 2 {
        s += &format!("    mov.u32 %bufc,0;\n    mov.u32 %bufp,{bufsz};\n    mov.u32 %kb,0;\n");
        s += &stage("%kb", "%bufc");
        s += "    cp.async.commit_group;\n";

        // for kb in 0..S step BK (S % BK == 0 ⇒ every block full).
        s += &format!("KB_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra DONE_{name};\n");
        s += &format!("    add.u32 %next,%kb,{bk};\n    setp.ge.u32 %pnext,%next,%S;\n");
        s += &format!("    @%pnext bra NOPF_{name};\n");
        s += &stage("%next", "%bufp");
        s += &format!(
            "    cp.async.commit_group;\n    cp.async.wait_group 1;\n    bra PFDONE_{name};\n"
        );
        s += &format!("NOPF_{name}:\n    cp.async.wait_group 0;\n");
        s += &format!("PFDONE_{name}:\n    bar.sync 0;\n");
    } else {
        // The `stages`-deep add+wrap ring — same bookkeeping as `entry_mma_reg_pipe`'s, with BK-key
        // blocks instead of 16-key ones: prologue stages blocks 0..stages-2 (one commit each, the stage
        // guarded, the commit not), every body commits exactly one group (empty in the tail), and
        // `wait_group stages-1` retires precisely the copy that filled the buffer this body reads.
        s += "    mov.u32 %kb,0;\n";
        s += &stage("%kb", "0");
        s += "    cp.async.commit_group;\n";
        for i in 1..stages - 1 {
            s += &format!("    add.u32 %kb,%kb,{bk};\n    setp.ge.u32 %p0,%kb,%S;\n");
            s += &format!("    @%p0 bra PS{i}_{name};\n");
            s += &stage("%kb", &format!("{}", i * bufsz));
            s += &format!("PS{i}_{name}:\n    cp.async.commit_group;\n");
        }
        s += &format!(
            "    mov.u32 %kb,0;\n    mov.u32 %bufc,0;\n    mov.u32 %bufp,{};\n",
            (stages - 1) * bufsz
        );
        s += &format!("KB_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra DONE_{name};\n");
        s += &format!(
            "    add.u32 %next,%kb,{};\n    setp.ge.u32 %pnext,%next,%S;\n",
            (stages - 1) * bk
        );
        s += &format!("    @%pnext bra NOPF_{name};\n");
        s += &stage("%next", "%bufp");
        s += &format!("NOPF_{name}:\n    cp.async.commit_group;\n");
        s += &format!("    cp.async.wait_group {};\n    bar.sync 0;\n", stages - 1);
    }

    // 1. S = Q·Kt : ntile key n-tiles, K read from SMEM at bufc (local key = 8·nt+grp).
    for nt in 0..ntile {
        for r in 0..4 {
            s += &format!("    mov.f32 %s{nt}_{r},0f00000000;\n");
        }
        if pv_ldmatrix {
            // K via `ldmatrix.x2` (no trans), exactly as in the narrow `_lm` kernel: K is staged
            // [key][hdim] = col-major B for `mma.row.col`, so no transpose is needed. Per kt the source
            // row is key = 8·nt + (lane&7) — the narrow formula with the wide kernel's n-tile index —
            // and ((lane>>3)&1)·8 picks the k-low/k-high half of the 16-hdim contraction tile.
            for kt in 0..ktq {
                s += &format!(
                    "    and.b32 %lkey,%lane,7;\n    add.u32 %lkey,%lkey,{};\n",
                    nt * 8
                );
                s += &format!("    mul.lo.u32 %tmp,%lkey,{d};\n");
                s += "    shr.u32 %idx,%lane,3;\n    and.b32 %idx,%idx,1;\n    shl.b32 %idx,%idx,3;\n";
                s += &format!("    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%idx;\n    shl.b32 %tmp,%tmp,1;\n", kt * 16);
                s += &format!("    mov.u32 %sbase,{sym};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,%tmp;\n");
                s += "    ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%b0,%b1},[%sbase];\n";
                s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{nt}_0,%s{nt}_1,%s{nt}_2,%s{nt}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%s{nt}_0,%s{nt}_1,%s{nt}_2,%s{nt}_3}};\n");
            }
        } else {
            s += &format!("    add.u32 %lkey,%grp,{};\n", nt * 8);
            for kt in 0..ktq {
                s += &format!("    mul.lo.u32 %tmp,%lkey,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,{sym};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,%tmp;\n", kt * 16);
                s += "    ld.shared.b32 %b0,[%sbase];\n    ld.shared.b32 %b1,[%sbase+16];\n";
                s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{nt}_0,%s{nt}_1,%s{nt}_2,%s{nt}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%s{nt}_0,%s{nt}_1,%s{nt}_2,%s{nt}_3}};\n");
            }
        }
    }

    // 2. online softmax over the full BK-wide score row (register-resident).
    s += "    max.f32 %lmax0,%s0_0,%s0_1;\n";
    for nt in 1..ntile {
        s += &format!("    max.f32 %lmax0,%lmax0,%s{nt}_0;\n    max.f32 %lmax0,%lmax0,%s{nt}_1;\n");
    }
    s += "    mul.f32 %lmax0,%lmax0,%scale;\n";
    s += "    max.f32 %lmax1,%s0_2,%s0_3;\n";
    for nt in 1..ntile {
        s += &format!("    max.f32 %lmax1,%lmax1,%s{nt}_2;\n    max.f32 %lmax1,%lmax1,%s{nt}_3;\n");
    }
    s += "    mul.f32 %lmax1,%lmax1,%scale;\n";
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax0,{off},0x1f,0xffffffff;\n    max.f32 %lmax0,%lmax0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax1,{off},0x1f,0xffffffff;\n    max.f32 %lmax1,%lmax1,%rt;\n");
    }
    s += &format!("    max.f32 %mnew0,%m0,%lmax0;\n    sub.f32 %corr0,%m0,%mnew0;\n    mul.f32 %corr0,%corr0,{log2e};\n    ex2.approx.f32 %corr0,%corr0;\n");
    s += &format!("    max.f32 %mnew1,%m1,%lmax1;\n    sub.f32 %corr1,%m1,%mnew1;\n    mul.f32 %corr1,%corr1,{log2e};\n    ex2.approx.f32 %corr1,%corr1;\n");
    for nt in 0..nto {
        s += &format!("    mul.f32 %o{nt}_0,%o{nt}_0,%corr0;\n    mul.f32 %o{nt}_1,%o{nt}_1,%corr0;\n    mul.f32 %o{nt}_2,%o{nt}_2,%corr1;\n    mul.f32 %o{nt}_3,%o{nt}_3,%corr1;\n");
    }
    let prob = |dst: &str, sreg: &str, mnew: &str| -> String {
        format!("    mul.f32 %pp,{sreg},%scale;\n    sub.f32 %pp,%pp,{mnew};\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 {dst},%pp;\n")
    };
    let pack = |dst: &str, lo: &str, hi: &str| -> String {
        format!("    cvt.rn.f16.f32 %h0,{lo};\n    and.b32 %h0,%h0,65535;\n    cvt.rn.f16.f32 %h1,{hi};\n    shl.b32 %h1,%h1,16;\n    or.b32 {dst},%h0,%h1;\n")
    };
    // qr0 probabilities → P fragments + psum0 (each n-tile nt → k-tile nt/2, slot 0 if even else 2).
    s += "    mov.f32 %psum0,0f00000000;\n";
    for nt in 0..ntile {
        s += &prob("%tp0", &format!("%s{nt}_0"), "%mnew0");
        s += &prob("%tp1", &format!("%s{nt}_1"), "%mnew0");
        s += "    add.f32 %psum0,%psum0,%tp0;\n    add.f32 %psum0,%psum0,%tp1;\n";
        let (kk, slot) = (nt / 2, if nt % 2 == 0 { 0 } else { 2 });
        s += &pack(&format!("%pa{kk}_{slot}"), "%tp0", "%tp1");
    }
    // qr1 probabilities → P fragments + psum1 (slot 1 if even else 3).
    s += "    mov.f32 %psum1,0f00000000;\n";
    for nt in 0..ntile {
        s += &prob("%tp2", &format!("%s{nt}_2"), "%mnew1");
        s += &prob("%tp3", &format!("%s{nt}_3"), "%mnew1");
        s += "    add.f32 %psum1,%psum1,%tp2;\n    add.f32 %psum1,%psum1,%tp3;\n";
        let (kk, slot) = (nt / 2, if nt % 2 == 0 { 1 } else { 3 });
        s += &pack(&format!("%pa{kk}_{slot}"), "%tp2", "%tp3");
    }
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum0,{off},0x1f,0xffffffff;\n    add.f32 %psum0,%psum0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum1,{off},0x1f,0xffffffff;\n    add.f32 %psum1,%psum1,%rt;\n");
    }
    s += "    fma.rn.f32 %l0,%l0,%corr0,%psum0;\n    fma.rn.f32 %l1,%l1,%corr1,%psum1;\n";
    s += "    mov.f32 %m0,%mnew0;\n    mov.f32 %m1,%mnew1;\n";

    // 4. O += P·V : for each output d-tile, accumulate nkb key k-tiles (V from SMEM at bufc+ksz).
    for nt in 0..nto {
        if pv_ldmatrix {
            // `ldmatrix.x2.trans` per (output d-tile, key k-tile) — the narrow `_lm` gather with the
            // k-tile's key base folded in: lanes 0..15 supply the address of key `16·kk + (lane&15)`,
            // 8 contiguous hdim at `nt*8`, and the hardware 8x8 transpose produces the col-major B
            // fragment the PV `mma` wants. This is what removes the wide kernel's documented
            // "strided SMEM V-load" ceiling (the hand path below 8-way bank-conflicts: the lanes of one
            // grp read keys tg2 = 0,2,4,6 at a fixed hdim, i.e. the same bank).
            for kk in 0..nkb {
                s += &format!("    and.b32 %idx,%lane,15;\n    mul.lo.u32 %tmp,%idx,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,{};\n    shl.b32 %tmp,%tmp,1;\n", 16 * kk * d, nt * 8);
                s += &format!("    mov.u32 %sbase,{sym};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,{ksz};\n    add.u32 %sbase,%sbase,%tmp;\n");
                s += "    ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%b0,%b1},[%sbase];\n";
                s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%pa{kk}_0,%pa{kk}_1,%pa{kk}_2,%pa{kk}_3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
            }
            continue;
        }
        s += &format!("    add.u32 %idx,%grp,{};\n", nt * 8);
        for kk in 0..nkb {
            // base = &smemV[key = 16·kk + tg2][hdim = idx] = smem + bufc + ksz + ((16kk+tg2)·d + idx)·2
            s += &format!("    mul.lo.u32 %tmp,%tg2,{d};\n    add.u32 %tmp,%tmp,%idx;\n    add.u32 %tmp,%tmp,{};\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,{sym};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,{ksz};\n    add.u32 %sbase,%sbase,%tmp;\n", 16 * kk * d);
            s += &format!("    ld.shared.u16 %h0,[%sbase];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b0,%h0,%h1;\n", 2 * d);
            s += &format!("    ld.shared.u16 %h0,[%sbase+{}];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b1,%h0,%h1;\n", 16 * d, 18 * d);
            s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%pa{kk}_0,%pa{kk}_1,%pa{kk}_2,%pa{kk}_3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
        }
    }

    if stages == 2 {
        // advance: swap buffers, kb = next.
        s += "    mov.u32 %bswap,%bufc;\n    mov.u32 %bufc,%bufp;\n    mov.u32 %bufp,%bswap;\n";
        s += &format!("    mov.u32 %kb,%next;\n    bra KB_{name};\n");
    } else {
        // advance: both ring cursors step ONE slot and wrap; kb steps one BK block on its own because
        // `%next` is `stages-1` blocks ahead. (`%bswap` is unused at this depth.)
        let ring = stages * bufsz;
        for reg in ["%bufc", "%bufp"] {
            s += &format!("    add.u32 {reg},{reg},{bufsz};\n    setp.ge.u32 %p0,{reg},{ring};\n    @%p0 sub.u32 {reg},{reg},{ring};\n");
        }
        s += &format!("    add.u32 %kb,%kb,{bk};\n    bra KB_{name};\n");
    }

    // store O[row][c] = o / l_row.
    s += &format!("DONE_{name}:\n");
    for nt in 0..nto {
        s += &format!("    div.rn.f32 %o{nt}_0,%o{nt}_0,%l0;\n    div.rn.f32 %o{nt}_1,%o{nt}_1,%l0;\n    div.rn.f32 %o{nt}_2,%o{nt}_2,%l1;\n    div.rn.f32 %o{nt}_3,%o{nt}_3,%l1;\n");
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_0;\n    st.global.f32 [%base+4],%o{nt}_1;\n", nt * 8);
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_2;\n    st.global.f32 [%base+4],%o{nt}_3;\n", nt * 8);
    }
    s += "    ret;\n}\n";
    s
}

/// **Fused-RoPE** `cp.async`-pipelined register-resident flash (`flash_d{d}_mprope`). Identical to
/// [`entry_mma_reg_pipe`] but applies **rotary position embedding to Q and K inside the kernel**, on the
/// fragments, from host-precomputed `cos`/`sin` tables (`[S, d/2]` f32) passed as two extra pointers —
/// the **library-can't-fuse** lever. A fused-attention library (cuDNN / cutlass) computes
/// `softmax(scale·Q·Kᵀ)·V` and **cannot** absorb RoPE, so a real model runs RoPE as a *separate*
/// elementwise kernel over Q and K first (an HBM round-trip + 1–2 launches). That standalone pass is a
/// large fraction of the attention at small/moderate S (measured +20%→+750% on the cuDNN path here),
/// whereas attention is **tensor-core-bound** so the rotation's CUDA-core ALU runs in the *shadow of the
/// `mma`* — nearly free for Wukong. So `one fused kernel` vs `RoPE-kernel + SDPA` is a genuine,
/// honest win in the regime where Wukong's raw attention is already near parity.
///
/// **Interleaved (GPT-J) RoPE convention** — pairs are the *adjacent* dims `(2t, 2t+1)`, which is exactly
/// what each `b32` A/B fragment register already packs (two consecutive head-dim f16), so the rotation is
/// a pure in-register rewrite of the loaded fragment with no reshuffle: `q'[2t]=q[2t]·c−q[2t+1]·s`,
/// `q'[2t+1]=q[2t]·s+q[2t+1]·c`, with `c=cos(p·θ_t)`, `s=sin(p·θ_t)`, `θ_t=base^(−2t/d)`, `p` the row
/// (Q) or global key (K) position. Q is rotated **once** at load; each K fragment is rotated right after
/// its `ld.shared` (re-rotated per query-block CTA, but that redundant ALU hides behind the `mma`). The
/// `cos`/`sin` tables are exact (host f64→f32), so the only error vs a CPU `rope→attention` oracle is the
/// f16 fragment round-trip + tensor-core accumulation — gated to the same f16 tolerance.
fn entry_mma_reg_pipe_rope(d: usize) -> String {
    assert!(d % 16 == 0, "mma flash needs D % 16 == 0");
    let ktq = d / 16; // Q·Kt contraction tiles (over hdim)
    let nto = d / 8; // P·V output n-tiles (over hdim)
    let half = d / 2; // rotary pairs per head
    let ksz = 16 * d * 2; // bytes of one staged K block (== one V block)
    let bufsz = 2 * ksz;
    let cpl = d / 16; // 16-byte cp.async chunks per lane per tensor
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = format!("flash_d{d}_mprope");

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO,\n    .param .u64 pCos,\n    .param .u64 pSin\n)\n{{\n"
    );
    s += "    .reg .pred %p0,%pnext;\n";
    let mut fr = String::from(
        "%scale,%m0,%m1,%mnew0,%mnew1,%corr0,%corr1,%l0,%l1,%lmax0,%lmax1,%rt,%psum0,%psum1,%pp,%tp0,%tp1,%tp2,%tp3,%cs,%sn,%flo,%fhi,%nlo,%nhi,%tmpf",
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
    let mut br = String::new();
    for kt in 0..ktq {
        for r in 0..4 {
            br += &format!("%qa{kt}_{r},");
        }
    }
    br += "%a0,%a1,%a2,%a3,%b0,%b1,%h0,%h1,";
    s += &format!(
        "    .reg .b32 {br}%S,%lane,%grp,%tg,%tg2,%row,%qr0,%qr1,%kb,%kpos,%ridx,%idx,%tmp,%hoff,%bufc,%bufp,%next,%sbase,%sk,%chunk,%lkey,%bswap;\n"
    );
    s += "    .reg .b16 %rlo,%rhi;\n";
    s += "    .reg .b64 %Q,%K,%V,%O,%Cos,%Sin,%base,%cadr,%off;\n";
    s += &format!("    .shared .align 16 .b8 smem_{name}[{}];\n", 2 * bufsz);

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n    ld.param.u64 %Cos,[pCos];\n    ld.param.u64 %Sin,[pSin];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n    cvta.to.global.u64 %Cos,%Cos;\n    cvta.to.global.u64 %Sin,%Sin;\n";
    // multi-head head base offset (ctaid.y) for Q/K/V/O; Cos/Sin are shared across heads (position×t).
    s += &format!("    mov.u32 %hoff,%ctaid.y;\n    mul.lo.u32 %hoff,%hoff,%S;\n    mul.lo.u32 %hoff,%hoff,{d};\n");
    s += "    mul.wide.u32 %off,%hoff,2;\n    add.s64 %Q,%Q,%off;\n    add.s64 %K,%K,%off;\n    add.s64 %V,%V,%off;\n";
    s += "    mul.wide.u32 %off,%hoff,4;\n    add.s64 %O,%O,%off;\n";
    s += "    mov.u32 %lane,%tid.x;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg2,%tg,1;\n";
    s += "    mov.u32 %row,%ctaid.x;\n    shl.b32 %row,%row,4;\n    add.u32 %qr0,%row,%grp;\n    add.u32 %qr1,%qr0,8;\n";

    // Rotate a b32 fragment holding the adjacent f16 pair (2t, 2t+1) by (%cs, %sn) already loaded.
    let rot = |reg: &str| -> String {
        format!(
            "    mov.b32 {{%rlo,%rhi}},{reg};\n    cvt.f32.f16 %flo,%rlo;\n    cvt.f32.f16 %fhi,%rhi;\n    mul.f32 %nlo,%flo,%cs;\n    mul.f32 %tmpf,%fhi,%sn;\n    sub.f32 %nlo,%nlo,%tmpf;\n    mul.f32 %nhi,%flo,%sn;\n    fma.rn.f32 %nhi,%fhi,%cs,%nhi;\n    cvt.rn.f16.f32 %rlo,%nlo;\n    cvt.rn.f16.f32 %rhi,%nhi;\n    mov.b32 {reg},{{%rlo,%rhi}};\n"
        )
    };
    // Load cos/sin for (position %posreg, pair index = `tconst` + %tg) into %cs/%sn.
    let loadcs = |posreg: &str, tconst: usize| -> String {
        format!(
            "    mul.lo.u32 %ridx,{posreg},{half};\n    add.u32 %ridx,%ridx,%tg;\n    add.u32 %ridx,%ridx,{tconst};\n    mul.wide.u32 %off,%ridx,4;\n    add.s64 %cadr,%Cos,%off;\n    ld.global.f32 %cs,[%cadr];\n    add.s64 %cadr,%Sin,%off;\n    ld.global.f32 %sn,[%cadr];\n"
        )
    };

    // Load Q A-fragments once, then RoPE-rotate each (row qr0/qr1, pair t = kt·8 + tg [+4]).
    for kt in 0..ktq {
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_0,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_2,[%base];\n");
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_1,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_3,[%base];\n");
        s += &loadcs("%qr0", kt * 8);
        s += &rot(&format!("%qa{kt}_0"));
        s += &loadcs("%qr0", kt * 8 + 4);
        s += &rot(&format!("%qa{kt}_2"));
        s += &loadcs("%qr1", kt * 8);
        s += &rot(&format!("%qa{kt}_1"));
        s += &loadcs("%qr1", kt * 8 + 4);
        s += &rot(&format!("%qa{kt}_3"));
    }
    for nt in 0..nto {
        for r in 0..4 {
            s += &format!("    mov.f32 %o{nt}_{r},0f00000000;\n");
        }
    }
    s += "    mov.f32 %m0,0fFF800000;\n    mov.f32 %m1,0fFF800000;\n    mov.f32 %l0,0f00000000;\n    mov.f32 %l1,0f00000000;\n";

    let stage = |kbreg: &str, bufreg: &str| -> String {
        let mut t = String::new();
        for ci in 0..cpl {
            t += &format!("    add.u32 %chunk,%lane,{};\n", ci * 32);
            t += &format!("    mul.lo.u32 %tmp,{kbreg},{d};\n    shl.b32 %sk,%chunk,3;\n    add.u32 %tmp,%tmp,%sk;\n    mul.wide.u32 %off,%tmp,2;\n");
            t += &format!("    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,{bufreg};\n    shl.b32 %sk,%chunk,4;\n    add.u32 %sbase,%sbase,%sk;\n");
            t += "    add.s64 %base,%K,%off;\n    cp.async.cg.shared.global [%sbase],[%base],16;\n";
            t += &format!("    add.u32 %sbase,%sbase,{ksz};\n    add.s64 %base,%V,%off;\n    cp.async.cg.shared.global [%sbase],[%base],16;\n");
        }
        t
    };

    s += &format!("    mov.u32 %bufc,0;\n    mov.u32 %bufp,{bufsz};\n    mov.u32 %kb,0;\n");
    s += &stage("%kb", "%bufc");
    s += "    cp.async.commit_group;\n";

    s += &format!("KB_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra DONE_{name};\n");
    s += "    add.u32 %next,%kb,16;\n    setp.ge.u32 %pnext,%next,%S;\n";
    s += &format!("    @%pnext bra NOPF_{name};\n");
    s += &stage("%next", "%bufp");
    s += &format!(
        "    cp.async.commit_group;\n    cp.async.wait_group 1;\n    bra PFDONE_{name};\n"
    );
    s += &format!("NOPF_{name}:\n    cp.async.wait_group 0;\n");
    s += &format!("PFDONE_{name}:\n    bar.sync 0;\n");

    // 1. S = Q·Kt : K read from SMEM at bufc, then RoPE-rotated in-register (global key kb+lkey).
    for nk in 0..2 {
        for r in 0..4 {
            s += &format!("    mov.f32 %s{nk}_{r},0f00000000;\n");
        }
        s += &format!("    add.u32 %lkey,%grp,{};\n", nk * 8);
        s += &format!("    add.u32 %kpos,%kb,%lkey;\n");
        for kt in 0..ktq {
            s += &format!("    mul.lo.u32 %tmp,%lkey,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,%tmp;\n", kt * 16);
            s += "    ld.shared.b32 %b0,[%sbase];\n    ld.shared.b32 %b1,[%sbase+16];\n";
            s += &loadcs("%kpos", kt * 8);
            s += &rot("%b0");
            s += &loadcs("%kpos", kt * 8 + 4);
            s += &rot("%b1");
            s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}};\n");
        }
    }

    // 2. online softmax (register-resident) — identical to entry_mma_reg_pipe.
    s += "    max.f32 %lmax0,%s0_0,%s0_1;\n    max.f32 %lmax0,%lmax0,%s1_0;\n    max.f32 %lmax0,%lmax0,%s1_1;\n    mul.f32 %lmax0,%lmax0,%scale;\n";
    s += "    max.f32 %lmax1,%s0_2,%s0_3;\n    max.f32 %lmax1,%lmax1,%s1_2;\n    max.f32 %lmax1,%lmax1,%s1_3;\n    mul.f32 %lmax1,%lmax1,%scale;\n";
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax0,{off},0x1f,0xffffffff;\n    max.f32 %lmax0,%lmax0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%lmax1,{off},0x1f,0xffffffff;\n    max.f32 %lmax1,%lmax1,%rt;\n");
    }
    s += &format!("    max.f32 %mnew0,%m0,%lmax0;\n    sub.f32 %corr0,%m0,%mnew0;\n    mul.f32 %corr0,%corr0,{log2e};\n    ex2.approx.f32 %corr0,%corr0;\n");
    s += &format!("    max.f32 %mnew1,%m1,%lmax1;\n    sub.f32 %corr1,%m1,%mnew1;\n    mul.f32 %corr1,%corr1,{log2e};\n    ex2.approx.f32 %corr1,%corr1;\n");
    for nt in 0..nto {
        s += &format!("    mul.f32 %o{nt}_0,%o{nt}_0,%corr0;\n    mul.f32 %o{nt}_1,%o{nt}_1,%corr0;\n    mul.f32 %o{nt}_2,%o{nt}_2,%corr1;\n    mul.f32 %o{nt}_3,%o{nt}_3,%corr1;\n");
    }
    let prob = |dst: &str, sreg: &str, mnew: &str| -> String {
        format!("    mul.f32 %pp,{sreg},%scale;\n    sub.f32 %pp,%pp,{mnew};\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 {dst},%pp;\n")
    };
    let pack = |dst: &str, lo: &str, hi: &str| -> String {
        format!("    cvt.rn.f16.f32 %h0,{lo};\n    and.b32 %h0,%h0,65535;\n    cvt.rn.f16.f32 %h1,{hi};\n    shl.b32 %h1,%h1,16;\n    or.b32 {dst},%h0,%h1;\n")
    };
    s += &prob("%tp0", "%s0_0", "%mnew0");
    s += &prob("%tp1", "%s0_1", "%mnew0");
    s += &prob("%tp2", "%s1_0", "%mnew0");
    s += &prob("%tp3", "%s1_1", "%mnew0");
    s += "    add.f32 %psum0,%tp0,%tp1;\n    add.f32 %psum0,%psum0,%tp2;\n    add.f32 %psum0,%psum0,%tp3;\n";
    s += &pack("%a0", "%tp0", "%tp1");
    s += &pack("%a2", "%tp2", "%tp3");
    s += &prob("%tp0", "%s0_2", "%mnew1");
    s += &prob("%tp1", "%s0_3", "%mnew1");
    s += &prob("%tp2", "%s1_2", "%mnew1");
    s += &prob("%tp3", "%s1_3", "%mnew1");
    s += "    add.f32 %psum1,%tp0,%tp1;\n    add.f32 %psum1,%psum1,%tp2;\n    add.f32 %psum1,%psum1,%tp3;\n";
    s += &pack("%a1", "%tp0", "%tp1");
    s += &pack("%a3", "%tp2", "%tp3");
    for off in [1, 2] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum0,{off},0x1f,0xffffffff;\n    add.f32 %psum0,%psum0,%rt;\n");
        s += &format!("    shfl.sync.bfly.b32 %rt,%psum1,{off},0x1f,0xffffffff;\n    add.f32 %psum1,%psum1,%rt;\n");
    }
    s += "    fma.rn.f32 %l0,%l0,%corr0,%psum0;\n    fma.rn.f32 %l1,%l1,%corr1,%psum1;\n";
    s += "    mov.f32 %m0,%mnew0;\n    mov.f32 %m1,%mnew1;\n";

    // 4. O += P·V : V read from SMEM at bufc+ksz (unrotated — RoPE applies to Q/K only).
    for nt in 0..nto {
        s += &format!("    add.u32 %idx,%grp,{};\n", nt * 8);
        s += &format!("    mul.lo.u32 %tmp,%tg2,{d};\n    add.u32 %tmp,%tmp,%idx;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,{ksz};\n    add.u32 %sbase,%sbase,%tmp;\n");
        s += &format!("    ld.shared.u16 %h0,[%sbase];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b0,%h0,%h1;\n", 2 * d);
        s += &format!("    ld.shared.u16 %h0,[%sbase+{}];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b1,%h0,%h1;\n", 16 * d, 18 * d);
        s += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
    }

    s += "    mov.u32 %bswap,%bufc;\n    mov.u32 %bufc,%bufp;\n    mov.u32 %bufp,%bswap;\n";
    s += &format!("    mov.u32 %kb,%next;\n    bra KB_{name};\n");

    s += &format!("DONE_{name}:\n");
    for nt in 0..nto {
        s += &format!("    div.rn.f32 %o{nt}_0,%o{nt}_0,%l0;\n    div.rn.f32 %o{nt}_1,%o{nt}_1,%l0;\n    div.rn.f32 %o{nt}_2,%o{nt}_2,%l1;\n    div.rn.f32 %o{nt}_3,%o{nt}_3,%l1;\n");
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_0;\n    st.global.f32 [%base+4],%o{nt}_1;\n", nt * 8);
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_2;\n    st.global.f32 [%base+4],%o{nt}_3;\n", nt * 8);
    }
    s += "    ret;\n}\n";
    s
}

/// **Warp-specialized ping-pong** flash (`flash_d{d}_ws{c}` / `flash_d{d}_ws{c}_lm`) — the FA2/FA3-style
/// lever for the long-S plateau (measured 0.46×/0.43× of cuDNN at S=2048/4096, D=64, where Wukong sits
/// ~9.5 TF/s while cuDNN scales to ~20). Every prior lever (mp4/mp8 lockstep multi-warp, mpw wide-Bk,
/// msp QK-ahead-in-one-warp, hs head-split, m1 single-buffer-double-occupancy) measured DEAD because they
/// all keep QKᵀ→softmax→PV **serialized inside one warp's instruction stream**: during the softmax's
/// `ex2.approx`+`shfl` SFU stretch the tensor cores idle, and `ptxas` cannot overlap what one in-order
/// warp serializes. This kernel puts the mma and the SFU softmax in **different warps' streams,
/// phase-locked in anti-phase** so one warp's softmax always has the other warp's mma to hide behind.
///
/// **Layout.** 2 warps/CTA (64 threads); warp `w` owns its own DISJOINT 16-query-row tile
/// `row = (ctaid.x·2 + w)·16` (grid `ceil((S/16)/2)` × heads, block 64). Per-warp online-softmax state
/// (m, l, O) stays in registers exactly as [`entry_mma_reg_pipe`] — no cross-warp softmax merge, so the
/// per-row math and ascending key order are IDENTICAL to `flash_d{d}_mp` (gated by construction). Both
/// warps consume the SAME K/V stream from ONE cooperatively-staged `cp.async` double buffer (D=64: 8 KB,
/// D=128: 16 KB — the same SMEM/CTA as `mp`, but now 2 warps share it: occupancy doubles to ~24 warps/SM
/// at D=64 and the K/V global/L2 traffic halves).
///
/// **The phase machine** (named barriers 1 and 2, both count 64; warp A = warpid 0, B = warpid 1):
/// ```text
///   A: QK(i) | bar1.sync | stage(i+1)→bufp | SM(i) | PV(i) | wait_group 0 | bar2.sync   | swap → i+1
///   B:         bar1.sync | stage(i+1)→bufp | QK(i) | SM(i) | wait_group 0 | bar2.arrive | PV(i) | swap → i+1
/// ```
/// Between bar1(i) and bar2(i), A runs SM(i)+PV(i) while B runs QK(i)+SM(i): **A's SFU softmax overlaps
/// B's QKᵀ mma, and A's PV mma overlaps B's SFU softmax** — the anti-phase the lockstep `mp4` never had.
/// Between bar2(i) and bar1(i+1), A's QK(i+1) mma overlaps B's PV(i) mma (both tensor-core, no conflict).
/// B only *arrives* at barrier 2 (it need not wait for A there); A must sync on it before consuming the
/// freshly staged block.
///
/// **Buffer lifetime / no-deadlock.** Block `i` lives in buffer `i mod 2`. Its last reader is B's PV(i)
/// (after bar2(i), before B's bar1(i+1) arrival); the overwriting stage(i+2) is issued only after
/// bar1(i+1) — every read of a buffer is barrier-ordered before the stage that overwrites it. RAW: each
/// thread `cp.async.wait_group 0`s its own stage(i+1) chunks before arriving at barrier 2 of step `i`,
/// and every consumer of block `i+1` (A's QK after bar2(i).sync, B's QK/PV after bar1(i+1).sync — which
/// waits on A's arrival, itself after A's bar2(i).sync) is separated from every producer's wait by a
/// barrier, which publishes the SMEM writes. Both warps execute the SAME uniform trip count (`kend` is
/// per-CTA), and barriers/staging/commit/wait are NEVER predicated — only the QK/SM/PV compute and the
/// Q-load/store are, on `%pc = active ∧ (¬causal ∨ kb ≤ row)` (warp-uniform), so a ragged final CTA
/// (odd S/16 → warp B inactive) and a causal warp A idling through warp B's diagonal block still arrive
/// at every barrier with matching counts.
///
/// **Causal** (`_wsc`): the CTA's uniform loop bound is `kend = min(rowB+16, S)` (warp B's diagonal —
/// the larger of the two warps' needs); each warp skips compute for `kb > row` and applies the same
/// per-element diagonal mask as `flash_d{d}_mpc`. **`pv_ldmatrix`** (`_lm`, the D=128 production feed):
/// K via `ldmatrix.x2`, V via `ldmatrix.x2.trans`, exactly as [`entry_mma_reg_pipe`]'s `_lm` variants.
fn entry_mma_reg_pipe_ws(d: usize, causal: bool, pv_ldmatrix: bool) -> String {
    assert!(
        d % 32 == 0,
        "ws flash needs D % 32 == 0 (64-thread cooperative stage)"
    );
    let ktq = d / 16; // Q.Kt contraction tiles (over hdim)
    let nto = d / 8; // P.V output n-tiles (over hdim)
    let ksz = 16 * d * 2; // bytes of one staged K block (== one V block)
    let bufsz = 2 * ksz; // K+V slab per pipeline buffer
    let stage_iters = (16 * d / 8) / 64; // 16-byte chunks per tensor / 64 threads (= d/32, exact)
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = match (causal, pv_ldmatrix) {
        (false, false) => format!("flash_d{d}_ws"),
        (true, false) => format!("flash_d{d}_wsc"),
        (false, true) => format!("flash_d{d}_ws_lm"),
        (true, true) => format!("flash_d{d}_wsc_lm"),
    };

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0,%pst,%act,%pc;\n";
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
        "    .reg .b32 {br}%S,%tix,%lane,%warpid,%grp,%tg,%tg2,%row,%qr0,%qr1,%kb,%kend,%next,%idx,%tmp,%hoff,%bufc,%bufp,%sbase,%sk,%chunk,%lkey,%bswap;\n"
    );
    s += "    .reg .b64 %Q,%K,%V,%O,%base,%off;\n";
    s += &format!("    .shared .align 16 .b8 smem_{name}[{}];\n", 2 * bufsz);

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    // multi-head head base offset (ctaid.y), identical to entry_mma_reg_pipe.
    s += &format!("    mov.u32 %hoff,%ctaid.y;\n    mul.lo.u32 %hoff,%hoff,%S;\n    mul.lo.u32 %hoff,%hoff,{d};\n");
    s += "    mul.wide.u32 %off,%hoff,2;\n    add.s64 %Q,%Q,%off;\n    add.s64 %K,%K,%off;\n    add.s64 %V,%V,%off;\n";
    s += "    mul.wide.u32 %off,%hoff,4;\n    add.s64 %O,%O,%off;\n";
    // tid → warpid/lane; lane → grp/tg/tg2. row = (ctaid.x*2 + warpid)*16 (per-warp query tile).
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpid,%tix,5;\n    and.b32 %lane,%tix,31;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg2,%tg,1;\n";
    s += "    mov.u32 %row,%ctaid.x;\n    shl.b32 %row,%row,1;\n    add.u32 %row,%row,%warpid;\n    shl.b32 %row,%row,4;\n    add.u32 %qr0,%row,%grp;\n    add.u32 %qr1,%qr0,8;\n    setp.lt.u32 %act,%row,%S;\n";
    // Uniform per-CTA key-loop bound: non-causal = S; causal = min(rowB+16, S) where rowB is warp 1's
    // tile base — the LARGER of the two warps' diagonal needs (warp 0 idles compute past its own).
    if causal {
        s += "    mov.u32 %tmp,%ctaid.x;\n    shl.b32 %tmp,%tmp,5;\n    add.u32 %tmp,%tmp,32;\n    min.u32 %kend,%tmp,%S;\n";
    } else {
        s += "    mov.u32 %kend,%S;\n";
    }

    // Load Q A-fragments once (active warps only — guards the global read).
    s += &format!("    @!%act bra SKIPQ_{name};\n");
    for kt in 0..ktq {
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_0,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_2,[%base];\n");
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_1,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_3,[%base];\n");
    }
    s += &format!("SKIPQ_{name}:\n");
    for nt in 0..nto {
        for r in 0..4 {
            s += &format!("    mov.f32 %o{nt}_{r},0f00000000;\n");
        }
    }
    s += "    mov.f32 %m0,0fFF800000;\n    mov.f32 %m1,0fFF800000;\n    mov.f32 %l0,0f00000000;\n    mov.f32 %l1,0f00000000;\n";

    // Cooperative 64-thread cp.async stage of the 16-key K+V slab at key index `kbreg` into buffer
    // `bufreg`. `guarded` predicates every copy on %pst (block-exists) — used by the in-loop stage; the
    // chunk passes divide exactly (2·d chunks / 64 threads), so no per-chunk range predicate is needed.
    let stage = |kbreg: &str, bufreg: &str, guarded: bool| -> String {
        let g = if guarded { "@%pst " } else { "" };
        let mut t = String::new();
        for ci in 0..stage_iters {
            t += &format!("    add.u32 %chunk,%tix,{};\n", ci * 64);
            t += &format!("    mul.lo.u32 %tmp,{kbreg},{d};\n    shl.b32 %sk,%chunk,3;\n    add.u32 %tmp,%tmp,%sk;\n    mul.wide.u32 %off,%tmp,2;\n");
            t += &format!("    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,{bufreg};\n    shl.b32 %sk,%chunk,4;\n    add.u32 %sbase,%sbase,%sk;\n");
            t += &format!("    add.s64 %base,%K,%off;\n    {g}cp.async.cg.shared.global [%sbase],[%base],16;\n");
            t += &format!("    add.u32 %sbase,%sbase,{ksz};\n    add.s64 %base,%V,%off;\n    {g}cp.async.cg.shared.global [%sbase],[%base],16;\n");
        }
        t
    };

    // The per-step compute sections, emitted ONCE as strings and spliced into BOTH warps' loops so the
    // two instruction streams stay arithmetically identical (only their barrier phase differs).
    // 1. S = Q.Kt : two key n-tiles, K read from SMEM at bufc.
    let mut qk = String::new();
    for nk in 0..2 {
        for r in 0..4 {
            qk += &format!("    mov.f32 %s{nk}_{r},0f00000000;\n");
        }
        if pv_ldmatrix {
            // K via `ldmatrix.x2` (no trans) — same as entry_mma_reg_pipe's `_lm` QK feed.
            for kt in 0..ktq {
                qk += &format!(
                    "    and.b32 %lkey,%lane,7;\n    add.u32 %lkey,%lkey,{};\n",
                    nk * 8
                );
                qk += &format!("    mul.lo.u32 %tmp,%lkey,{d};\n");
                qk += "    shr.u32 %idx,%lane,3;\n    and.b32 %idx,%idx,1;\n    shl.b32 %idx,%idx,3;\n";
                qk += &format!("    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%idx;\n    shl.b32 %tmp,%tmp,1;\n", kt * 16);
                qk += &format!("    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,%tmp;\n");
                qk += "    ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%b0,%b1},[%sbase];\n";
                qk += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}};\n");
            }
        } else {
            qk += &format!("    add.u32 %lkey,%grp,{};\n", nk * 8);
            for kt in 0..ktq {
                qk += &format!("    mul.lo.u32 %tmp,%lkey,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,%tmp;\n", kt * 16);
                qk += "    ld.shared.b32 %b0,[%sbase];\n    ld.shared.b32 %b1,[%sbase+16];\n";
                qk += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}};\n");
            }
        }
    }
    // causal mask (identical to entry_mma_reg_pipe: global key indices vs query indices).
    if causal {
        qk += "    add.u32 %ck0,%kb,%tg2;\n    add.u32 %ck1,%ck0,1;\n    add.u32 %ck8,%ck0,8;\n    add.u32 %ck9,%ck8,1;\n";
        for (sreg, key, qr) in [
            ("%s0_0", "%ck0", "%qr0"),
            ("%s0_1", "%ck1", "%qr0"),
            ("%s0_2", "%ck0", "%qr1"),
            ("%s0_3", "%ck1", "%qr1"),
            ("%s1_0", "%ck8", "%qr0"),
            ("%s1_1", "%ck9", "%qr0"),
            ("%s1_2", "%ck8", "%qr1"),
            ("%s1_3", "%ck9", "%qr1"),
        ] {
            qk += &format!(
                "    setp.gt.u32 %p0,{key},{qr};\n    selp.f32 {sreg},0fFF800000,{sreg},%p0;\n"
            );
        }
    }

    // 2. online softmax (register-resident) — identical to entry_mma_reg_pipe.
    let mut sm = String::new();
    sm += "    max.f32 %lmax0,%s0_0,%s0_1;\n    max.f32 %lmax0,%lmax0,%s1_0;\n    max.f32 %lmax0,%lmax0,%s1_1;\n    mul.f32 %lmax0,%lmax0,%scale;\n";
    sm += "    max.f32 %lmax1,%s0_2,%s0_3;\n    max.f32 %lmax1,%lmax1,%s1_2;\n    max.f32 %lmax1,%lmax1,%s1_3;\n    mul.f32 %lmax1,%lmax1,%scale;\n";
    for off in [1, 2] {
        sm += &format!("    shfl.sync.bfly.b32 %rt,%lmax0,{off},0x1f,0xffffffff;\n    max.f32 %lmax0,%lmax0,%rt;\n");
        sm += &format!("    shfl.sync.bfly.b32 %rt,%lmax1,{off},0x1f,0xffffffff;\n    max.f32 %lmax1,%lmax1,%rt;\n");
    }
    sm += &format!("    max.f32 %mnew0,%m0,%lmax0;\n    sub.f32 %corr0,%m0,%mnew0;\n    mul.f32 %corr0,%corr0,{log2e};\n    ex2.approx.f32 %corr0,%corr0;\n");
    sm += &format!("    max.f32 %mnew1,%m1,%lmax1;\n    sub.f32 %corr1,%m1,%mnew1;\n    mul.f32 %corr1,%corr1,{log2e};\n    ex2.approx.f32 %corr1,%corr1;\n");
    for nt in 0..nto {
        sm += &format!("    mul.f32 %o{nt}_0,%o{nt}_0,%corr0;\n    mul.f32 %o{nt}_1,%o{nt}_1,%corr0;\n    mul.f32 %o{nt}_2,%o{nt}_2,%corr1;\n    mul.f32 %o{nt}_3,%o{nt}_3,%corr1;\n");
    }
    let prob = |dst: &str, sreg: &str, mnew: &str| -> String {
        format!("    mul.f32 %pp,{sreg},%scale;\n    sub.f32 %pp,%pp,{mnew};\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 {dst},%pp;\n")
    };
    let pack = |dst: &str, lo: &str, hi: &str| -> String {
        format!("    cvt.rn.f16.f32 %h0,{lo};\n    and.b32 %h0,%h0,65535;\n    cvt.rn.f16.f32 %h1,{hi};\n    shl.b32 %h1,%h1,16;\n    or.b32 {dst},%h0,%h1;\n")
    };
    sm += &prob("%tp0", "%s0_0", "%mnew0");
    sm += &prob("%tp1", "%s0_1", "%mnew0");
    sm += &prob("%tp2", "%s1_0", "%mnew0");
    sm += &prob("%tp3", "%s1_1", "%mnew0");
    sm += "    add.f32 %psum0,%tp0,%tp1;\n    add.f32 %psum0,%psum0,%tp2;\n    add.f32 %psum0,%psum0,%tp3;\n";
    sm += &pack("%a0", "%tp0", "%tp1");
    sm += &pack("%a2", "%tp2", "%tp3");
    sm += &prob("%tp0", "%s0_2", "%mnew1");
    sm += &prob("%tp1", "%s0_3", "%mnew1");
    sm += &prob("%tp2", "%s1_2", "%mnew1");
    sm += &prob("%tp3", "%s1_3", "%mnew1");
    sm += "    add.f32 %psum1,%tp0,%tp1;\n    add.f32 %psum1,%psum1,%tp2;\n    add.f32 %psum1,%psum1,%tp3;\n";
    sm += &pack("%a1", "%tp0", "%tp1");
    sm += &pack("%a3", "%tp2", "%tp3");
    for off in [1, 2] {
        sm += &format!("    shfl.sync.bfly.b32 %rt,%psum0,{off},0x1f,0xffffffff;\n    add.f32 %psum0,%psum0,%rt;\n");
        sm += &format!("    shfl.sync.bfly.b32 %rt,%psum1,{off},0x1f,0xffffffff;\n    add.f32 %psum1,%psum1,%rt;\n");
    }
    sm += "    fma.rn.f32 %l0,%l0,%corr0,%psum0;\n    fma.rn.f32 %l1,%l1,%corr1,%psum1;\n";
    sm += "    mov.f32 %m0,%mnew0;\n    mov.f32 %m1,%mnew1;\n";

    // 3. O += P.V : V read from SMEM at bufc+ksz — identical to entry_mma_reg_pipe (hand / ldmatrix).
    let mut pv = String::new();
    if pv_ldmatrix {
        for nt in 0..nto {
            pv += &format!("    and.b32 %idx,%lane,15;\n    mul.lo.u32 %tmp,%idx,{d};\n    add.u32 %tmp,%tmp,{};\n    shl.b32 %tmp,%tmp,1;\n", nt * 8);
            pv += &format!("    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,{ksz};\n    add.u32 %sbase,%sbase,%tmp;\n");
            pv += "    ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%b0,%b1},[%sbase];\n";
            pv += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
        }
    } else {
        for nt in 0..nto {
            pv += &format!("    add.u32 %idx,%grp,{};\n", nt * 8);
            pv += &format!("    mul.lo.u32 %tmp,%tg2,{d};\n    add.u32 %tmp,%tmp,%idx;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,{ksz};\n    add.u32 %sbase,%sbase,%tmp;\n");
            pv += &format!("    ld.shared.u16 %h0,[%sbase];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b0,%h0,%h1;\n", 2 * d);
            pv += &format!("    ld.shared.u16 %h0,[%sbase+{}];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b1,%h0,%h1;\n", 16 * d, 18 * d);
            pv += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
        }
    }

    // Per-iteration compute predicate %pc (warp-uniform; NEVER guards barriers/staging): active, and for
    // causal also kb <= row (warp A idles through warp B's diagonal block at the shared kend).
    let setpc = if causal {
        "    setp.le.u32 %pc,%kb,%row;\n    and.pred %pc,%pc,%act;\n".to_string()
    } else {
        String::from("    mov.pred %pc,%act;\n")
    };
    // In-loop stage of block `next` into `bufp` + always-commit (uniform per-thread group counting:
    // exactly one cp.async-group per iteration per thread, possibly empty at the tail).
    let stage_next = format!(
        "    add.u32 %next,%kb,16;\n    setp.lt.u32 %pst,%next,%kend;\n{}    cp.async.commit_group;\n",
        stage("%next", "%bufp", true)
    );
    let swap = "    mov.u32 %bswap,%bufc;\n    mov.u32 %bufc,%bufp;\n    mov.u32 %bufp,%bswap;\n    mov.u32 %kb,%next;\n";

    // PROLOGUE: stage block 0 into buffer 0 (all 64 threads), drain, publish. kend >= 16 always (the CTA
    // exists ⇒ warp 0's tile exists ⇒ at least one key block), so the unguarded stage is in range.
    s += &format!("    mov.u32 %bufc,0;\n    mov.u32 %bufp,{bufsz};\n    mov.u32 %kb,0;\n");
    s += &stage("%kb", "%bufc", false);
    s += "    cp.async.commit_group;\n    cp.async.wait_group 0;\n    bar.sync 0;\n";
    s += &format!("    setp.eq.u32 %p0,%warpid,1;\n    @%p0 bra LOOPB_{name};\n");

    // ===== WARP A (warpid 0): QK(i) | bar1 | stage | SM(i) PV(i) | wait | bar2.sync | swap =====
    s += &format!("LOOPA_{name}:\n    setp.ge.u32 %p0,%kb,%kend;\n    @%p0 bra STORE_{name};\n");
    s += &setpc;
    s += &format!("    @!%pc bra AQ_{name};\n");
    s += &qk;
    s += &format!("AQ_{name}:\n");
    s += &format!("    bar.sync 1,64;\n");
    s += &stage_next;
    s += &format!("    @!%pc bra AS_{name};\n");
    s += &sm;
    s += &pv;
    s += &format!("AS_{name}:\n");
    s += "    cp.async.wait_group 0;\n";
    s += &format!("    bar.sync 2,64;\n");
    s += swap;
    s += &format!("    bra LOOPA_{name};\n");

    // ===== WARP B (warpid 1): bar1 | stage | QK(i) SM(i) | wait | bar2.arrive | PV(i) | swap =====
    s += &format!("LOOPB_{name}:\n    setp.ge.u32 %p0,%kb,%kend;\n    @%p0 bra STORE_{name};\n");
    s += &setpc;
    s += &format!("    bar.sync 1,64;\n");
    s += &stage_next;
    s += &format!("    @!%pc bra BQ_{name};\n");
    s += &qk;
    s += &sm;
    s += &format!("BQ_{name}:\n");
    s += "    cp.async.wait_group 0;\n";
    s += &format!("    bar.arrive 2,64;\n");
    s += &format!("    @!%pc bra BP_{name};\n");
    s += &pv;
    s += &format!("BP_{name}:\n");
    s += swap;
    s += &format!("    bra LOOPB_{name};\n");

    // store O[row][c] = o / l_row (active warps only) — identical to entry_mma_reg_pipe.
    s += &format!("STORE_{name}:\n    @!%act bra RET_{name};\n");
    for nt in 0..nto {
        s += &format!("    div.rn.f32 %o{nt}_0,%o{nt}_0,%l0;\n    div.rn.f32 %o{nt}_1,%o{nt}_1,%l0;\n    div.rn.f32 %o{nt}_2,%o{nt}_2,%l1;\n    div.rn.f32 %o{nt}_3,%o{nt}_3,%l1;\n");
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_0;\n    st.global.f32 [%base+4],%o{nt}_1;\n", nt * 8);
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_2;\n    st.global.f32 [%base+4],%o{nt}_3;\n", nt * 8);
    }
    s += &format!("RET_{name}:\n    ret;\n}}\n");
    s
}

/// **Stage B of the warp-specialization campaign: the 3-stage `cp.async` ring** (`flash_d{d}_ws3` /
/// `flash_d{d}_ws3_lm`, non-causal probe). Identical phase machine, per-warp math, and barrier roles to
/// [`entry_mma_reg_pipe_ws`] — it deepens ONLY the pipeline. Stage A's double buffer forces the tightest
/// possible stage schedule: block `i+1`'s copy is issued after bar1(i) and must complete by bar2(i)
/// (`cp.async.wait_group 0`), so it has just the SM+PV / QK+SM stretch to land and BOTH warps stall on a
/// late copy. With **three** buffers, block `i+2`'s previous tenant (block `i-1`) is fully read before
/// bar1(i) (its last reader is B's PV(i-1), which precedes B's bar1(i) arrival), so body `i` can issue
/// the stage **two blocks ahead** (`stage(i+2)` after bar1(i)) and relax the drain to
/// `cp.async.wait_group 1` — the copy now has a full extra step (issued at bar1(i), consumed after
/// bar2(i+1)), restoring the `mp` kernel's whole-step prefetch distance *on top of* the ws anti-phase.
/// The price is SMEM: 3 buffers = 12 KB/CTA at D=64 (8 CTAs/SM × 2 warps = 16 warps vs Stage A's 24)
/// and 24 KB at D=128 (4 CTAs × 2 = 8 warps vs 12) — the ring trades occupancy for latency slack, and
/// `flash_ws_vs_mp` measures which side of that trade this GPU is on. Prologue stages blocks 0 AND 1
/// (two commit groups) so the steady-state `wait_group 1` invariant holds from body 0; the tail commits
/// empty groups to keep per-thread group counts uniform. Buffer registers rotate 3-way
/// (`bufc←bufp←bufn←bufc`). Non-causal only (the A/B probe regime); causal follows if the ring wins.
fn entry_mma_reg_pipe_ws3(d: usize, pv_ldmatrix: bool) -> String {
    assert!(
        d % 32 == 0,
        "ws3 flash needs D % 32 == 0 (64-thread cooperative stage)"
    );
    let ktq = d / 16; // Q.Kt contraction tiles (over hdim)
    let nto = d / 8; // P.V output n-tiles (over hdim)
    let ksz = 16 * d * 2; // bytes of one staged K block (== one V block)
    let bufsz = 2 * ksz; // K+V slab per ring buffer
    let stage_iters = (16 * d / 8) / 64; // 16-byte chunks per tensor / 64 threads (= d/32, exact)
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = if pv_ldmatrix {
        format!("flash_d{d}_ws3_lm")
    } else {
        format!("flash_d{d}_ws3")
    };

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0,%pst,%act;\n";
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
    let mut br = String::new();
    for kt in 0..ktq {
        for r in 0..4 {
            br += &format!("%qa{kt}_{r},");
        }
    }
    br += "%a0,%a1,%a2,%a3,%b0,%b1,%h0,%h1,";
    s += &format!(
        "    .reg .b32 {br}%S,%tix,%lane,%warpid,%grp,%tg,%tg2,%row,%qr0,%qr1,%kb,%next,%nn,%idx,%tmp,%hoff,%bufc,%bufp,%bufn,%sbase,%sk,%chunk,%lkey,%bswap;\n"
    );
    s += "    .reg .b64 %Q,%K,%V,%O,%base,%off;\n";
    s += &format!("    .shared .align 16 .b8 smem_{name}[{}];\n", 3 * bufsz);

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    s += &format!("    mov.u32 %hoff,%ctaid.y;\n    mul.lo.u32 %hoff,%hoff,%S;\n    mul.lo.u32 %hoff,%hoff,{d};\n");
    s += "    mul.wide.u32 %off,%hoff,2;\n    add.s64 %Q,%Q,%off;\n    add.s64 %K,%K,%off;\n    add.s64 %V,%V,%off;\n";
    s += "    mul.wide.u32 %off,%hoff,4;\n    add.s64 %O,%O,%off;\n";
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpid,%tix,5;\n    and.b32 %lane,%tix,31;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg2,%tg,1;\n";
    s += "    mov.u32 %row,%ctaid.x;\n    shl.b32 %row,%row,1;\n    add.u32 %row,%row,%warpid;\n    shl.b32 %row,%row,4;\n    add.u32 %qr0,%row,%grp;\n    add.u32 %qr1,%qr0,8;\n    setp.lt.u32 %act,%row,%S;\n";

    s += &format!("    @!%act bra SKIPQ_{name};\n");
    for kt in 0..ktq {
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_0,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_2,[%base];\n");
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_1,[%base];\n", kt * 16);
        s += &format!("    add.u32 %tmp,%tmp,8;\n    mul.wide.u32 %off,%tmp,2;\n    add.s64 %base,%Q,%off;\n    ld.global.b32 %qa{kt}_3,[%base];\n");
    }
    s += &format!("SKIPQ_{name}:\n");
    for nt in 0..nto {
        for r in 0..4 {
            s += &format!("    mov.f32 %o{nt}_{r},0f00000000;\n");
        }
    }
    s += "    mov.f32 %m0,0fFF800000;\n    mov.f32 %m1,0fFF800000;\n    mov.f32 %l0,0f00000000;\n    mov.f32 %l1,0f00000000;\n";

    // Cooperative 64-thread cp.async stage (identical to entry_mma_reg_pipe_ws).
    let stage = |kbreg: &str, bufreg: &str, guarded: bool| -> String {
        let g = if guarded { "@%pst " } else { "" };
        let mut t = String::new();
        for ci in 0..stage_iters {
            t += &format!("    add.u32 %chunk,%tix,{};\n", ci * 64);
            t += &format!("    mul.lo.u32 %tmp,{kbreg},{d};\n    shl.b32 %sk,%chunk,3;\n    add.u32 %tmp,%tmp,%sk;\n    mul.wide.u32 %off,%tmp,2;\n");
            t += &format!("    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,{bufreg};\n    shl.b32 %sk,%chunk,4;\n    add.u32 %sbase,%sbase,%sk;\n");
            t += &format!("    add.s64 %base,%K,%off;\n    {g}cp.async.cg.shared.global [%sbase],[%base],16;\n");
            t += &format!("    add.u32 %sbase,%sbase,{ksz};\n    add.s64 %base,%V,%off;\n    {g}cp.async.cg.shared.global [%sbase],[%base],16;\n");
        }
        t
    };

    // QK / SM / PV sections — one string each, spliced into both warps' loops (as in the ws kernel).
    let mut qk = String::new();
    for nk in 0..2 {
        for r in 0..4 {
            qk += &format!("    mov.f32 %s{nk}_{r},0f00000000;\n");
        }
        if pv_ldmatrix {
            for kt in 0..ktq {
                qk += &format!(
                    "    and.b32 %lkey,%lane,7;\n    add.u32 %lkey,%lkey,{};\n",
                    nk * 8
                );
                qk += &format!("    mul.lo.u32 %tmp,%lkey,{d};\n");
                qk += "    shr.u32 %idx,%lane,3;\n    and.b32 %idx,%idx,1;\n    shl.b32 %idx,%idx,3;\n";
                qk += &format!("    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%idx;\n    shl.b32 %tmp,%tmp,1;\n", kt * 16);
                qk += &format!("    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,%tmp;\n");
                qk += "    ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%b0,%b1},[%sbase];\n";
                qk += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}};\n");
            }
        } else {
            qk += &format!("    add.u32 %lkey,%grp,{};\n", nk * 8);
            for kt in 0..ktq {
                qk += &format!("    mul.lo.u32 %tmp,%lkey,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,%tmp;\n", kt * 16);
                qk += "    ld.shared.b32 %b0,[%sbase];\n    ld.shared.b32 %b1,[%sbase+16];\n";
                qk += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}},{{%qa{kt}_0,%qa{kt}_1,%qa{kt}_2,%qa{kt}_3}},{{%b0,%b1}},{{%s{nk}_0,%s{nk}_1,%s{nk}_2,%s{nk}_3}};\n");
            }
        }
    }

    let mut sm = String::new();
    sm += "    max.f32 %lmax0,%s0_0,%s0_1;\n    max.f32 %lmax0,%lmax0,%s1_0;\n    max.f32 %lmax0,%lmax0,%s1_1;\n    mul.f32 %lmax0,%lmax0,%scale;\n";
    sm += "    max.f32 %lmax1,%s0_2,%s0_3;\n    max.f32 %lmax1,%lmax1,%s1_2;\n    max.f32 %lmax1,%lmax1,%s1_3;\n    mul.f32 %lmax1,%lmax1,%scale;\n";
    for off in [1, 2] {
        sm += &format!("    shfl.sync.bfly.b32 %rt,%lmax0,{off},0x1f,0xffffffff;\n    max.f32 %lmax0,%lmax0,%rt;\n");
        sm += &format!("    shfl.sync.bfly.b32 %rt,%lmax1,{off},0x1f,0xffffffff;\n    max.f32 %lmax1,%lmax1,%rt;\n");
    }
    sm += &format!("    max.f32 %mnew0,%m0,%lmax0;\n    sub.f32 %corr0,%m0,%mnew0;\n    mul.f32 %corr0,%corr0,{log2e};\n    ex2.approx.f32 %corr0,%corr0;\n");
    sm += &format!("    max.f32 %mnew1,%m1,%lmax1;\n    sub.f32 %corr1,%m1,%mnew1;\n    mul.f32 %corr1,%corr1,{log2e};\n    ex2.approx.f32 %corr1,%corr1;\n");
    for nt in 0..nto {
        sm += &format!("    mul.f32 %o{nt}_0,%o{nt}_0,%corr0;\n    mul.f32 %o{nt}_1,%o{nt}_1,%corr0;\n    mul.f32 %o{nt}_2,%o{nt}_2,%corr1;\n    mul.f32 %o{nt}_3,%o{nt}_3,%corr1;\n");
    }
    let prob = |dst: &str, sreg: &str, mnew: &str| -> String {
        format!("    mul.f32 %pp,{sreg},%scale;\n    sub.f32 %pp,%pp,{mnew};\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 {dst},%pp;\n")
    };
    let pack = |dst: &str, lo: &str, hi: &str| -> String {
        format!("    cvt.rn.f16.f32 %h0,{lo};\n    and.b32 %h0,%h0,65535;\n    cvt.rn.f16.f32 %h1,{hi};\n    shl.b32 %h1,%h1,16;\n    or.b32 {dst},%h0,%h1;\n")
    };
    sm += &prob("%tp0", "%s0_0", "%mnew0");
    sm += &prob("%tp1", "%s0_1", "%mnew0");
    sm += &prob("%tp2", "%s1_0", "%mnew0");
    sm += &prob("%tp3", "%s1_1", "%mnew0");
    sm += "    add.f32 %psum0,%tp0,%tp1;\n    add.f32 %psum0,%psum0,%tp2;\n    add.f32 %psum0,%psum0,%tp3;\n";
    sm += &pack("%a0", "%tp0", "%tp1");
    sm += &pack("%a2", "%tp2", "%tp3");
    sm += &prob("%tp0", "%s0_2", "%mnew1");
    sm += &prob("%tp1", "%s0_3", "%mnew1");
    sm += &prob("%tp2", "%s1_2", "%mnew1");
    sm += &prob("%tp3", "%s1_3", "%mnew1");
    sm += "    add.f32 %psum1,%tp0,%tp1;\n    add.f32 %psum1,%psum1,%tp2;\n    add.f32 %psum1,%psum1,%tp3;\n";
    sm += &pack("%a1", "%tp0", "%tp1");
    sm += &pack("%a3", "%tp2", "%tp3");
    for off in [1, 2] {
        sm += &format!("    shfl.sync.bfly.b32 %rt,%psum0,{off},0x1f,0xffffffff;\n    add.f32 %psum0,%psum0,%rt;\n");
        sm += &format!("    shfl.sync.bfly.b32 %rt,%psum1,{off},0x1f,0xffffffff;\n    add.f32 %psum1,%psum1,%rt;\n");
    }
    sm += "    fma.rn.f32 %l0,%l0,%corr0,%psum0;\n    fma.rn.f32 %l1,%l1,%corr1,%psum1;\n";
    sm += "    mov.f32 %m0,%mnew0;\n    mov.f32 %m1,%mnew1;\n";

    let mut pv = String::new();
    if pv_ldmatrix {
        for nt in 0..nto {
            pv += &format!("    and.b32 %idx,%lane,15;\n    mul.lo.u32 %tmp,%idx,{d};\n    add.u32 %tmp,%tmp,{};\n    shl.b32 %tmp,%tmp,1;\n", nt * 8);
            pv += &format!("    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,{ksz};\n    add.u32 %sbase,%sbase,%tmp;\n");
            pv += "    ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%b0,%b1},[%sbase];\n";
            pv += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
        }
    } else {
        for nt in 0..nto {
            pv += &format!("    add.u32 %idx,%grp,{};\n", nt * 8);
            pv += &format!("    mul.lo.u32 %tmp,%tg2,{d};\n    add.u32 %tmp,%tmp,%idx;\n    shl.b32 %tmp,%tmp,1;\n    mov.u32 %sbase,smem_{name};\n    add.u32 %sbase,%sbase,%bufc;\n    add.u32 %sbase,%sbase,{ksz};\n    add.u32 %sbase,%sbase,%tmp;\n");
            pv += &format!("    ld.shared.u16 %h0,[%sbase];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b0,%h0,%h1;\n", 2 * d);
            pv += &format!("    ld.shared.u16 %h0,[%sbase+{}];\n    ld.shared.u16 %h1,[%sbase+{}];\n    shl.b32 %h1,%h1,16;\n    or.b32 %b1,%h0,%h1;\n", 16 * d, 18 * d);
            pv += &format!("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1}},{{%o{nt}_0,%o{nt}_1,%o{nt}_2,%o{nt}_3}};\n");
        }
    }

    // In-loop stage TWO blocks ahead into the free ring slot %bufn + always-commit (uniform counting).
    let stage_next2 = format!(
        "    add.u32 %next,%kb,16;\n    add.u32 %nn,%kb,32;\n    setp.lt.u32 %pst,%nn,%S;\n{}    cp.async.commit_group;\n",
        stage("%nn", "%bufn", true)
    );
    // 3-way ring rotation: bufc <- bufp <- bufn <- bufc; advance kb.
    let rot = "    mov.u32 %bswap,%bufc;\n    mov.u32 %bufc,%bufp;\n    mov.u32 %bufp,%bufn;\n    mov.u32 %bufn,%bswap;\n    mov.u32 %kb,%next;\n";

    // PROLOGUE: stage block 0 AND block 1 (two commit groups — the steady-state `wait_group 1` needs
    // two groups outstanding from body 0), drain block 0, publish.
    s += &format!("    mov.u32 %bufc,0;\n    mov.u32 %bufp,{bufsz};\n    mov.u32 %bufn,{};\n    mov.u32 %kb,0;\n", 2 * bufsz);
    s += &stage("%kb", "%bufc", false);
    s += "    cp.async.commit_group;\n";
    // NB: the block index register must survive all chunk passes — the stage closure clobbers %tmp.
    s += "    mov.u32 %next,16;\n    setp.lt.u32 %pst,%next,%S;\n";
    s += &stage("%next", "%bufp", true);
    s += "    cp.async.commit_group;\n    cp.async.wait_group 1;\n    bar.sync 0;\n";
    s += &format!("    setp.eq.u32 %p0,%warpid,1;\n    @%p0 bra LOOPB_{name};\n");

    // ===== WARP A: QK(i) | bar1 | stage(i+2) | SM(i) PV(i) | wait 1 | bar2.sync | rotate =====
    s += &format!("LOOPA_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra STORE_{name};\n");
    s += &format!("    @!%act bra AQ_{name};\n");
    s += &qk;
    s += &format!("AQ_{name}:\n");
    s += "    bar.sync 1,64;\n";
    s += &stage_next2;
    s += &format!("    @!%act bra AS_{name};\n");
    s += &sm;
    s += &pv;
    s += &format!("AS_{name}:\n");
    s += "    cp.async.wait_group 1;\n";
    s += "    bar.sync 2,64;\n";
    s += rot;
    s += &format!("    bra LOOPA_{name};\n");

    // ===== WARP B: bar1 | stage(i+2) | QK(i) SM(i) | wait 1 | bar2.arrive | PV(i) | rotate =====
    s += &format!("LOOPB_{name}:\n    setp.ge.u32 %p0,%kb,%S;\n    @%p0 bra STORE_{name};\n");
    s += "    bar.sync 1,64;\n";
    s += &stage_next2;
    s += &format!("    @!%act bra BQ_{name};\n");
    s += &qk;
    s += &sm;
    s += &format!("BQ_{name}:\n");
    s += "    cp.async.wait_group 1;\n";
    s += "    bar.arrive 2,64;\n";
    s += &format!("    @!%act bra BP_{name};\n");
    s += &pv;
    s += &format!("BP_{name}:\n");
    s += rot;
    s += &format!("    bra LOOPB_{name};\n");

    // store O[row][c] = o / l_row (active warps only).
    s += &format!("STORE_{name}:\n    @!%act bra RET_{name};\n");
    for nt in 0..nto {
        s += &format!("    div.rn.f32 %o{nt}_0,%o{nt}_0,%l0;\n    div.rn.f32 %o{nt}_1,%o{nt}_1,%l0;\n    div.rn.f32 %o{nt}_2,%o{nt}_2,%l1;\n    div.rn.f32 %o{nt}_3,%o{nt}_3,%l1;\n");
        s += &format!("    mul.lo.s32 %tmp,%qr0,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_0;\n    st.global.f32 [%base+4],%o{nt}_1;\n", nt * 8);
        s += &format!("    mul.lo.s32 %tmp,%qr1,{d};\n    add.u32 %tmp,%tmp,{};\n    add.u32 %tmp,%tmp,%tg2;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %base,%O,%off;\n    st.global.f32 [%base],%o{nt}_2;\n    st.global.f32 [%base+4],%o{nt}_3;\n", nt * 8);
    }
    s += &format!("RET_{name}:\n    ret;\n}}\n");
    s
}

/// Flash-attention module — **every** flash kernel in one `&'static str` (one module key, so `gpu.rs`
/// loads it once): untiled `flash_d{D}` + tiled `flash_d{D}_t` per [`SUPPORTED_D`], the `wmma`
/// `flash_d64_w`/`flash_d64_w4`, the register-resident `flash_d64_m{,c}`, the whole `cp.async`-pipelined
/// `flash_d{64,128}_mp*` family (± causal, ± `_lm`, `_m1`, `_mp{4,8}`, `_mpw{2,4}`, `_msp`, `_hs`,
/// `_mprope`) and the warp-specialized `flash_d{64,128}_ws*` / `_ws3*`. See the crate header for what
/// each family is for; `every_dispatchable_flash_entry_is_defined` pins the dispatched names.
pub fn flash_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        // `sm_80` floor from the single source: this module's whole instruction mix
        // (`mma.sync.m16n8k16.f16`, `wmma`, `ldmatrix`, `cp.async`, `shfl.sync`, `ex2.approx`) is
        // Ampere-legal, and PTX is forward-compatible only — an `sm_89` tag would load on ZERO A100s.
        let mut m = String::from(crate::ptx_target::HDR_SM80);
        for &d in &SUPPORTED_D {
            m += &entry_untiled(d, FLASH_WARPS);
            m += &entry_tiled(d, FLASH_TWARPS);
        }
        m += &entry_wmma(64);
        m += &entry_wmma_wide(64, WMMA_FLASH_NKB);
        m += &entry_mma_reg(64, false);
        m += &entry_mma_reg(64, true);
        // Every entry in THIS module is generated at `STATIC_SMEM_CAP`, so every one of them stays on
        // the static `.shared` emission path and its text is byte-identical to before the budget
        // parameter existed (`flash_ptx_shipped_generators_are_byte_identical` pins that with digests).
        // The >48 KiB depth/width rows live in their own single-entry modules — see
        // [`FLASH_STAGE_VARIANTS`] — because this module is loaded by every flash launch and a shared
        // module would put their JIT cost on the production path.
        m += &entry_mma_reg_pipe(64, false, false, 2, STATIC_SMEM_CAP);
        m += &entry_mma_reg_pipe(64, true, false, 2, STATIC_SMEM_CAP);
        // Single-buffered D=64 occupancy probe (`flash_d64_m1`): one 4 KB K+V slab → ~24 CTAs/SM (~50%
        // occupancy) but no `cp.async` compute/load overlap. Diagnostic ONLY (A/B'd by
        // `flash_single_vs_double` to separate occupancy- from SFU/serial-softmax-bound at long S); NOT
        // a dispatch default. Bit-identical math to `flash_d64_mp`.
        m += &entry_mma_reg_pipe(64, false, false, 1, STATIC_SMEM_CAP);
        // `ldmatrix` SMEM-feed variants (`_lm`): both fragment loads become one warp-collective
        // conflict-free `ldmatrix` — V via `.x2.trans` (transpose-gather), K via `.x2` (no trans; K is
        // already col-major B). Replaces the strided, 8-way-bank-conflicting hand-packed loads — the
        // documented feed ceiling (the `mp4` wash proved this kernel is tensor-core-feed-bound, not
        // occupancy-bound). Same math, gated bit-tolerance-equal vs the hand path; A/B'd by `flash_lm_vs_mp`:
        // wins D=128 ~18–23% (the SMEM feed dominates there, nto=16) and ~2–4% at D=64 — ≥ the hand path
        // at every regime, so it is the default for the cuDNN comparison.
        m += &entry_mma_reg_pipe(64, false, true, 2, STATIC_SMEM_CAP);
        m += &entry_mma_reg_pipe(64, true, true, 2, STATIC_SMEM_CAP);
        // D=128 (Llama/GPT modern head dim): the register-resident mma flash generalizes over d
        // (ktq=d/16=8 QKᵀ tiles, nto=d/8=16 PV n-tiles, cpl=d/16=8 cp.async chunks, 16 KB SMEM,
        // ~120 regs/thread — all within Ada limits). Non-causal + causal, gated vs ref_attn at D=128.
        // The non-causal `_lm` (`flash_d128_mp_lm`) is the production D=128 dispatch (`wmma_flash_entry`).
        m += &entry_mma_reg_pipe(128, false, false, 2, STATIC_SMEM_CAP);
        m += &entry_mma_reg_pipe(128, true, false, 2, STATIC_SMEM_CAP);
        m += &entry_mma_reg_pipe(128, false, true, 2, STATIC_SMEM_CAP);
        m += &entry_mma_reg_pipe(128, true, true, 2, STATIC_SMEM_CAP);
        m += &entry_mma_reg_pipe_mw(64, 4);
        m += &entry_mma_reg_pipe_mw(64, 8);
        m += &entry_mma_reg_pipe_wide(64, 2, 2, false, STATIC_SMEM_CAP);
        m += &entry_mma_reg_pipe_wide(64, 4, 2, false, STATIC_SMEM_CAP);
        m += &entry_mma_reg_pipe_rope(64);
        // Software-pipelined (QKᵀ-ahead overlaps softmax SFU; separate K/V pools keep base occupancy):
        // the lever for the long-S softmax-stall ceiling that no SMEM-feed change touches. Non-causal.
        m += &entry_mma_reg_pipe_sp(64);
        // Head-dim warp-split for D=128: 2 warps share the 16 query rows, each owns half the output hdim
        // (32 not 64 O-accumulators) → ~2× occupancy, the documented D=128 register-pressure bound.
        m += &entry_mma_reg_pipe_hs(128);
        // Warp-specialized ping-pong (FA2/FA3-style): 2 warps/CTA on DISJOINT query tiles share one
        // staged K/V stream, phase-locked in anti-phase by named barriers so one warp's mma always
        // covers the other's ex2/shfl softmax stretch — the long-S lever the lockstep mp4/mp8 lacked.
        // D=64 hand-packed feed (matching the flash_d64_mp production feed) + D=128 ldmatrix feed
        // (matching flash_d128_mp_lm), non-causal + causal. Env-gated dispatch: WUKONG_FLASH_WS=1
        // routes D=64 S>=2048 / D=128 S>=1024 here (gpu::ws_flash_route); default dispatch unchanged.
        m += &entry_mma_reg_pipe_ws(64, false, false);
        m += &entry_mma_reg_pipe_ws(64, true, false);
        m += &entry_mma_reg_pipe_ws(128, false, true);
        m += &entry_mma_reg_pipe_ws(128, true, true);
        // Stage B probe: the 3-stage cp.async ring twin of the ws kernels (whole-step prefetch distance
        // restored via a third buffer + wait_group 1, at 1.5x the SMEM ⇒ lower occupancy). Non-causal
        // only; A/B'd three-way against mp and ws by `flash_ws_vs_mp`.
        m += &entry_mma_reg_pipe_ws3(64, false);
        m += &entry_mma_reg_pipe_ws3(128, true);
        m
    })
    .as_str()
}

/// 16-key sub-tiles staged per online-softmax step in the wide WMMA flash (`flash_d64_w{N}`). `WK = 16·N`
/// keys per step ⇒ `S % WK == 0` required; 4 → 64-key tile, divides every layer seq (which is `S%64==0`).
pub const WMMA_FLASH_NKB: usize = 4;

/// Head dims with a generated kernel (the common transformer values).
pub const SUPPORTED_D: [usize; 3] = [32, 64, 128];

/// One row of the **flash depth × staging-width grid** — the `(D, BK, ring depth, causal, feed)`
/// point, its stable module-cache key **and** its PTX entry name in one `&'static str`.
///
/// The name carries the depth (and the staging width, and the feed) because `Gpu::function` keys the
/// module cache on the key alone and **never re-examines the PTX on a hit**: two rows under one key
/// would silently run the first one's kernel with the second one's launch window — a wrong-output bug
/// that still returns `Ok`. One row = one key = one entry = one module. [`flash_stage_ptx`] asserts
/// that the generator's own derived entry name equals this field, so a rename on either side fails
/// loudly at generation instead of as a `CUDA_ERROR_NOT_FOUND` at the one shape that reaches it.
#[derive(Clone, Copy, Debug)]
pub struct FlashStageCfg {
    /// PTX entry symbol *and* module-cache key (a `&'static str`, as `Gpu::function` requires).
    pub name: &'static str,
    /// Head dim. 64 and 128 are the generated tensor-core dims.
    pub d: usize,
    /// Keys staged per online-softmax step / 16. `1` routes the narrow generator, `>= 2` the wide one.
    pub nkb: usize,
    /// `cp.async` ring depth in K+V slabs. `2` is the shipped double buffer; `>= 3` is the ring.
    pub stages: usize,
    /// Causal masking. Only the narrow generator derives it (`nkb == 1`).
    pub causal: bool,
    /// `ldmatrix` SMEM feed for both the QKt K fragment and the PV V fragment.
    pub pv_ldmatrix: bool,
}

impl FlashStageCfg {
    /// Keys staged per step.
    pub const fn bk(&self) -> usize {
        16 * self.nkb
    }
    /// `stages · 2 · BK · D · 2` — the family's closed form (a K slab and a V slab per ring buffer,
    /// f16). Equals `64 · stages · nkb · D`.
    pub const fn smem_bytes(&self) -> usize {
        self.stages * 2 * self.bk() * self.d * 2
    }
    /// Static at or below the 48 KiB PTX ISA cap, one dynamic window beyond it — the single emission
    /// rule, shared with int8/fp8 through [`crate::gpu::smem_mode_for`].
    pub const fn smem_mode(&self) -> SmemMode {
        smem_mode_for(self.smem_bytes())
    }
    /// One warp per CTA for every row here (both generators put one 16-query-row block on one warp).
    pub const fn threads(&self) -> u32 {
        32
    }
    /// **The dispatcher's decline test.** A row that does not fit the running device's
    /// `Gpu::smem_budget()` must be skipped *here*, before generation: the generators panic on an
    /// over-budget row on purpose, because a clamped launch would compute with a truncated ring.
    pub const fn fits(&self, smem_budget: usize) -> bool {
        self.smem_bytes() <= smem_budget
    }
    /// Shortest S at which this depth is fully used: the prologue stages `stages-1` blocks, so below
    /// `(stages-1)·BK` the deepest buffers never fill. Shorter S is *correct* (every prologue stage is
    /// guarded) — the depth is simply wasted SMEM, so a dispatcher should prefer a shallower row.
    pub const fn min_s(&self) -> usize {
        (self.stages - 1) * self.bk()
    }
}

/// **The flash depth × staging-width grid.** Every row exists because of the dynamic-SMEM window: the
/// static rows are the ones the 48 KiB ISA cap still admits (and are the honest controls), the dynamic
/// ones cannot be declared statically on *any* device, however much carveout it has.
///
/// | row | D | BK | stages | SMEM | form | CTAs/SM: 4050 (100 KiB) / A100 (164) / H100 (228) |
/// |---|---|---|---|---|---|---|
/// | `flash_d64_mp_s3`       | 64  | 16 | 3 | 12 KiB | static  | 8 / 13 / 18 (SMEM never binds; depth is free) |
/// | `flash_d64_mp_s4`       | 64  | 16 | 4 | 16 KiB | static  | 6 / 10 / 14 |
/// | `flash_d64_mpc_s4`      | 64  | 16 | 4 | 16 KiB | static  | causal twin of the row above |
/// | `flash_d128_mp_lm_s4`   | 128 | 16 | 4 | 32 KiB | static  | 3 / 5 / 7 |
/// | `flash_d128_mp_lm_s6`   | 128 | 16 | 6 | 48 KiB | static  | 2 / 3 / 4 — the LAST statically declarable depth |
/// | `flash_d128_mp_lm_s8`   | 128 | 16 | 8 | 64 KiB | **dyn** | 1 / 2 / 3 |
/// | `flash_d128_mpc_lm_s8`  | 128 | 16 | 8 | 64 KiB | **dyn** | causal twin |
/// | `flash_d64_mpw4_s3`     | 64  | 64 | 3 | 48 KiB | static  | 2 / 3 / 4 |
/// | `flash_d64_mpw4_s4`     | 64  | 64 | 4 | 64 KiB | **dyn** | 1 / 2 / 3 |
/// | `flash_d64_mpw4_s5`     | 64  | 64 | 5 | 80 KiB | **dyn** | 1 / 2 / 2 |
/// | `flash_d128_mpw2_lm_s3` | 128 | 32 | 3 | 48 KiB | static  | 2 / 3 / 4 |
/// | `flash_d128_mpw2_lm_s4` | 128 | 32 | 4 | 64 KiB | **dyn** | 1 / 2 / 3 |
/// | `flash_d128_mpw4_lm_s3` | 128 | 64 | 3 | 96 KiB | **dyn** | 1 / 1 / 2 — the deepest row; fits Ada's 99 KiB opt-in with 3 KiB to spare |
///
/// The CTAs/SM column is the **SMEM bound only** — `floor((SMEM_sm - 1 KiB) / row)`, D6 §3.3. It is
/// an upper bound, not a prediction: registers bind too, and the widest D=128 rows are heavy
/// (`nto·4` = 64 O accumulators, `2·nkb·4` score registers, `4·nkb` P fragments, `4·ktq` Q fragments —
/// `flash_d128_mpw4_lm_s3` is the extreme and may well spill). Which of the two binds first is a
/// `ptxas` fact, so it is a *measurement*, not something to assert here.
///
/// **This grid is a correctness deliverable, not a perf bet on this card**, exactly as
/// `INT8_STAGE_VARIANTS` is. The 4050 measured the *width* lever a loss at 48 KiB static
/// (`flash_wide_vs_mp`: `mpw2/mp ≈ 0.65–0.91×`) and named occupancy as the mechanism — which is the
/// one thing the datacenter budgets change (D6 §3.3: the f16 workhorse gets s3 free on the 4050, s5 on
/// A100, s7 on H100 at *unchanged* CTAs/SM). What transfers 100% is the PTX: these are the exact
/// modules an A100/H100 will run, and every structural property of them is gated here at $0.
///
/// Deliberately NOT in the grid: causal × wide (the wide generator has no BK-wide diagonal mask —
/// `flash_stage_ptx` rejects the combination rather than emitting an unmasked kernel), and multi-warp
/// × wide (`entry_mma_reg_pipe_mw` does not take `nkb` yet; D6 §4.2/4.3 name that product as the
/// datacenter flash lever, and it is the natural next commit).
pub const FLASH_STAGE_VARIANTS: &[FlashStageCfg] = &[
    FlashStageCfg {
        name: "flash_d64_mp_s3",
        d: 64,
        nkb: 1,
        stages: 3,
        causal: false,
        pv_ldmatrix: false,
    },
    FlashStageCfg {
        name: "flash_d64_mp_s4",
        d: 64,
        nkb: 1,
        stages: 4,
        causal: false,
        pv_ldmatrix: false,
    },
    FlashStageCfg {
        name: "flash_d64_mpc_s4",
        d: 64,
        nkb: 1,
        stages: 4,
        causal: true,
        pv_ldmatrix: false,
    },
    FlashStageCfg {
        name: "flash_d128_mp_lm_s4",
        d: 128,
        nkb: 1,
        stages: 4,
        causal: false,
        pv_ldmatrix: true,
    },
    FlashStageCfg {
        name: "flash_d128_mp_lm_s6",
        d: 128,
        nkb: 1,
        stages: 6,
        causal: false,
        pv_ldmatrix: true,
    },
    FlashStageCfg {
        name: "flash_d128_mp_lm_s8",
        d: 128,
        nkb: 1,
        stages: 8,
        causal: false,
        pv_ldmatrix: true,
    },
    FlashStageCfg {
        name: "flash_d128_mpc_lm_s8",
        d: 128,
        nkb: 1,
        stages: 8,
        causal: true,
        pv_ldmatrix: true,
    },
    FlashStageCfg {
        name: "flash_d64_mpw4_s3",
        d: 64,
        nkb: 4,
        stages: 3,
        causal: false,
        pv_ldmatrix: false,
    },
    FlashStageCfg {
        name: "flash_d64_mpw4_s4",
        d: 64,
        nkb: 4,
        stages: 4,
        causal: false,
        pv_ldmatrix: false,
    },
    FlashStageCfg {
        name: "flash_d64_mpw4_s5",
        d: 64,
        nkb: 4,
        stages: 5,
        causal: false,
        pv_ldmatrix: false,
    },
    FlashStageCfg {
        name: "flash_d128_mpw2_lm_s3",
        d: 128,
        nkb: 2,
        stages: 3,
        causal: false,
        pv_ldmatrix: true,
    },
    FlashStageCfg {
        name: "flash_d128_mpw2_lm_s4",
        d: 128,
        nkb: 2,
        stages: 4,
        causal: false,
        pv_ldmatrix: true,
    },
    FlashStageCfg {
        name: "flash_d128_mpw4_lm_s3",
        d: 128,
        nkb: 4,
        stages: 3,
        causal: false,
        pv_ldmatrix: true,
    },
];

/// Generate one [`FLASH_STAGE_VARIANTS`] row as a **single-entry module** against `smem_budget` bytes
/// (the running device's `Gpu::smem_budget()`, or a target's budget when enumerating off-device).
/// Returns the module and the [`SmemMode`] its launch must honour — `Gpu::function_smem` consumes
/// exactly this pair, so no caller re-derives a byte count the generator already knows.
///
/// One row = one module, rather than more entries in [`flash_ptx`], for two reasons: that module is
/// loaded by *every* flash launch and these rows would put their JIT cost on the production path, and
/// the module cache keys on a `&'static str` — a per-row key is exactly what the depth/width variants
/// need (hard rule 4).
///
/// The emitted text depends on `smem_budget` only through [`crate::gpu::smem_mode_for`] (i.e. only on
/// the row's own byte count), so a bigger card produces byte-identical PTX and the same cubin-cache
/// entry. Panics (loudly, at generation) if the row does not fit: a decline belongs in the
/// dispatcher's [`FlashStageCfg::fits`], never in a silently-clamped launch.
pub fn flash_stage_ptx(v: &FlashStageCfg, smem_budget: usize) -> (String, SmemMode) {
    let entry = if v.nkb == 1 {
        entry_mma_reg_pipe(v.d, v.causal, v.pv_ldmatrix, v.stages, smem_budget)
    } else {
        // The wide generator masks nothing: its score tile spans BK keys with no diagonal predicate, so
        // a "causal" wide kernel would silently attend to future keys. Reject, never emit.
        assert!(
            !v.causal,
            "{}: causal masking is derived only for the 16-key (nkb == 1) tile",
            v.name
        );
        entry_mma_reg_pipe_wide(v.d, v.nkb, v.stages, v.pv_ldmatrix, smem_budget)
    };
    // The grid's key must BE the generator's own derived entry name — two spellings of one fact is how
    // a dispatch seam silently drifts.
    assert!(
        entry.contains(&format!(".visible .entry {}(", v.name)),
        "{}: the generator derived a different entry name for (d={}, nkb={}, stages={}, causal={}, lm={})",
        v.name,
        v.d,
        v.nkb,
        v.stages,
        v.causal,
        v.pv_ldmatrix
    );
    let mode = v.smem_mode();
    // `sm_80` floor: the ring adds no instruction past the Ampere-legal mix, and the `.extern .shared`
    // window is launch-time state, not an ISA feature (D6 §1.1, verified on-device under 7.8/sm_80).
    let mut m = String::from(crate::ptx_target::HDR_SM80);
    if mode.is_dynamic() {
        // MODULE SCOPE, not inside the entry — the identical line in an entry body is
        // CUDA_ERROR_INVALID_PTX (measured).
        m += DSMEM_DECL;
    }
    m += &entry;
    (m, mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This Ada card's probed `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` (99 KiB), spelled as a literal so
    /// the generator gates stay **device-free** — the whole point of "the budget is a parameter, not a
    /// probe" (D6 §5.1) is that the grid is enumerable and text-gateable with no GPU in the room.
    const ADA_OPTIN_BUDGET: usize = 101_376;
    /// A100's and H100's per-block opt-in ceilings, for the same reason.
    const A100_OPTIN_BUDGET: usize = 166_912;
    const H100_OPTIN_BUDGET: usize = 232_448;

    /// FNV-1a 64. A dependency-free content digest for the byte-identity gate below — the point is a
    /// stable fingerprint of a generated string, not cryptography.
    fn fnv1a64(s: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Every `.visible .entry <name>(` defined in `ptx`, in order.
    fn entry_names(ptx: &str) -> Vec<&str> {
        ptx.match_indices(".visible .entry ")
            .map(|(i, m)| {
                let rest = &ptx[i + m.len()..];
                &rest[..rest
                    .find('(')
                    .expect("an entry declaration opens its param list")]
            })
            .collect()
    }

    /// B1: a single non-ASCII byte anywhere in the module is a `ptxas fatal` at `cuModuleLoadData`,
    /// and this file's prose is written with `.`/`x`-style math characters right next to the
    /// `format!`s that build the kernels. Nothing but this test keeps the emitted text ASCII, and a
    /// GPU-less `cargo test` never loads a module, so the failure would surface only on a machine
    /// with a device -- as every flash launch dying at once.
    #[test]
    fn flash_ptx_is_pure_ascii() {
        let mut modules: Vec<(String, String)> =
            vec![("flash".to_string(), flash_ptx().to_string())];
        // The depth/width grid lives in its own single-entry modules, so the family's ASCII gate has
        // to reach them explicitly — they are generated by the same `format!`s next to the same
        // `x`/`.`-heavy prose, and a GPU-less `cargo test` never loads a module.
        for v in FLASH_STAGE_VARIANTS {
            modules.push((v.name.to_string(), flash_stage_ptx(v, ADA_OPTIN_BUDGET).0));
        }
        for (label, ptx) in modules {
            if let Some((i, line)) = ptx.lines().enumerate().find(|(_, l)| !l.is_ascii()) {
                panic!(
                    "{label} PTX must be pure ASCII (ptxas fatal otherwise) -- line {}: {line}",
                    i + 1
                );
            }
        }
    }

    /// **The byte-identity gate for the dynamic-SMEM window.** `smem_budget` / `stages` /
    /// `pv_ldmatrix` were threaded through two shipped generators, and the contract of that work is
    /// that at the shipped arguments (`stages` 1 or 2, budget = [`STATIC_SMEM_CAP`]) **not one byte of
    /// the emitted text moves** — the shipped kernels keep their measured behaviour, their cubin-cache
    /// entries stay warm, and nothing that was A/B'd needs re-measuring.
    ///
    /// The digests below were taken from the tree at `4b98e13`, *before* the parameterization, by
    /// running the same FNV-1a over the same generator calls. They are therefore an **external**
    /// golden, not a restatement of the current code: the whole-module pin catches a change anywhere
    /// in the 32 entries, and the per-generator rows localize it to the exact shape that moved.
    #[test]
    fn flash_ptx_shipped_generators_are_byte_identical() {
        let m = flash_ptx();
        assert_eq!(
            m.matches(".visible .entry ").count(),
            32,
            "the shipped flash module must still define exactly 32 entries"
        );
        assert_eq!(
            (m.len(), fnv1a64(m)),
            (1_092_308, 0xaf69_3d1d_98e3_f9d5),
            "flash_ptx() moved: the shipped module is not byte-identical to the pre-window tree"
        );
        let cap = STATIC_SMEM_CAP;
        for (label, text, want_len, want_digest) in [
            (
                "flash_d64_mp",
                entry_mma_reg_pipe(64, false, false, 2, cap),
                26_268usize,
                0x50ed_f623_60dd_ff26u64,
            ),
            (
                "flash_d64_mpc",
                entry_mma_reg_pipe(64, true, false, 2, cap),
                27_004,
                0x7d2e_f984_ae56_cdb7,
            ),
            (
                "flash_d64_m1",
                entry_mma_reg_pipe(64, false, false, 1, cap),
                24_136,
                0xd77c_4442_9f61_d5be,
            ),
            (
                "flash_d64_mp_lm",
                entry_mma_reg_pipe(64, false, true, 2, cap),
                25_998,
                0xa14e_f504_2639_2b16,
            ),
            (
                "flash_d64_mpc_lm",
                entry_mma_reg_pipe(64, true, true, 2, cap),
                26_734,
                0x84ab_96ec_5a98_4201,
            ),
            (
                "flash_d128_mp",
                entry_mma_reg_pipe(128, false, false, 2, cap),
                47_613,
                0xd09e_1e96_e2a4_3fae,
            ),
            (
                "flash_d128_mpc",
                entry_mma_reg_pipe(128, true, false, 2, cap),
                48_373,
                0x639c_637b_c3b3_99ab,
            ),
            (
                "flash_d128_mp_lm",
                entry_mma_reg_pipe(128, false, true, 2, cap),
                47_095,
                0x8342_6fd4_ea81_2284,
            ),
            (
                "flash_d128_mpc_lm",
                entry_mma_reg_pipe(128, true, true, 2, cap),
                47_855,
                0x22a6_1a5b_476e_9783,
            ),
            (
                "flash_d64_mpw2",
                entry_mma_reg_pipe_wide(64, 2, 2, false, cap),
                41_325,
                0xc485_651d_20be_701d,
            ),
            (
                "flash_d64_mpw4",
                entry_mma_reg_pipe_wide(64, 4, 2, false, cap),
                70_414,
                0xac04_5638_9647_7936,
            ),
        ] {
            assert_eq!(
                (text.len(), fnv1a64(&text)),
                (want_len, want_digest),
                "{label}: the parameterized generator no longer reproduces the shipped text byte for \
                 byte — the <= 48 KiB static path is NOT allowed to move"
            );
            assert!(
                m.contains(&text),
                "{label}: flash_ptx() must embed exactly this generator output"
            );
        }
    }

    /// **The shipped module declares no window and touches no window symbol.** Every entry in
    /// `flash_ptx()` is generated at [`STATIC_SMEM_CAP`], so all 32 are on the static path. If a future
    /// entry crosses 48 KiB it will start emitting `wk_dsmem` references while `flash_ptx()` emits no
    /// module-scope `.extern .shared` — a `CUDA_ERROR_INVALID_PTX` for *every* flash launch, since they
    /// all share one module. Fail here instead, where the fix (declare the window, or move the entry to
    /// its own [`flash_stage_ptx`] module) is obvious.
    #[test]
    fn flash_ptx_declares_the_window_iff_an_entry_uses_it() {
        let m = flash_ptx();
        assert_eq!(
            m.contains(DSMEM_SYM),
            m.contains(".extern .shared"),
            "flash_ptx(): the dynamic-SMEM window must be declared at module scope exactly when an \
             entry addresses it"
        );
        assert!(
            !m.contains(DSMEM_SYM),
            "no shipped flash entry should need the window yet — if one now does, give it its own \
             module (flash_stage_ptx) rather than paying its JIT on every flash launch"
        );
    }

    /// **Header-floor gate, device-free.** Every instruction this module emits (`mma.sync.m16n8k16`,
    /// `wmma`, `ldmatrix`, `cp.async`, `shfl.sync`, `ex2.approx`) is Ampere-legal, so the module is
    /// tagged at the `sm_80` floor from [`crate::ptx_target`]. PTX is forward-compatible only: the
    /// old `sm_89` tag cost nothing on Ada and made the module unloadable on every A100.
    #[test]
    fn flash_ptx_is_tagged_at_the_sm80_floor() {
        let ptx = flash_ptx();
        assert!(
            ptx.starts_with(crate::ptx_target::HDR_SM80),
            "flash PTX must open with ptx_target::HDR_SM80, got: {:?}",
            &ptx[..ptx.len().min(64)]
        );
        assert!(
            !ptx.contains(crate::ptx_target::TARGET_SM89),
            "an Ampere-legal module must not claim the Ada floor"
        );
    }

    /// Structural sanity the driver would otherwise be the first to check: balanced braces and one
    /// uniquely named entry per generated kernel (a duplicate name silently shadows a kernel).
    #[test]
    fn flash_ptx_is_structurally_well_formed() {
        let ptx = flash_ptx();
        assert_eq!(
            ptx.matches('{').count(),
            ptx.matches('}').count(),
            "unbalanced braces"
        );
        let names = entry_names(ptx);
        assert!(!names.is_empty(), "the module must define kernels");
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            names.len(),
            "duplicate .visible .entry name in the flash module"
        );
        for n in &names {
            assert_eq!(
                ptx.matches(&format!(".visible .entry {n}(")).count(),
                1,
                "{n} declared twice"
            );
        }
    }

    /// The dispatch seam must close: every entry name a *host* dispatch decision can produce has to
    /// be defined in `flash_ptx()`. Renaming a generator's output (or a routing literal) on one side
    /// only builds clean and passes every non-GPU test; it surfaces as a `CUDA_ERROR_NOT_FOUND` from
    /// `Gpu::function` at exactly the shape that reaches the renamed arm. The names are taken from
    /// the routing functions themselves, not copied, so either side drifting fails here.
    #[test]
    fn every_dispatchable_flash_entry_is_defined() {
        let ptx = flash_ptx();
        let defined = |name: &str| ptx.contains(&format!(".visible .entry {name}("));

        // The f32 flash plan: both the untiled and the SMEM-tiled kernel, for every supported head dim.
        for &d in &SUPPORTED_D {
            for tiled in [false, true] {
                let (name, _) = crate::gpu::flash_plan_forced(d, 512, tiled);
                assert!(
                    defined(&name),
                    "flash_plan_forced({d}, .., {tiled}) picks undefined entry `{name}`"
                );
            }
        }
        // The tensor-core dispatch (`wmma_flash_applies` gates d to 64/128).
        for d in [64usize, 128] {
            for s in [512usize, 1024, 4096] {
                let name = crate::gpu::wmma_flash_entry(d, s);
                assert!(
                    defined(name),
                    "wmma_flash_entry({d}, {s}) picks undefined entry `{name}`"
                );
            }
        }
        // The warp-specialized override (enabled + S >= 4096); the routing table itself is gated in
        // gpu.rs, here only its *products* must exist.
        for d in [64usize, 128] {
            let name = crate::gpu::ws_flash_route_with(true, d, 4096).unwrap_or_else(|| {
                panic!("ws_flash_route_with(true, {d}, 4096) must route somewhere")
            });
            assert!(
                defined(name),
                "ws_flash_route_with(true, {d}, 4096) picks undefined entry `{name}`"
            );
        }
    }

    /// **The occupancy derivation behind [`FLASH_WARPS`], and the claim that it may stay a constant.**
    /// The historical `W = 2` was justified by "Ada's ~24-blocks/SM cap", which is an Ada fact; A100
    /// and H100 allow 32 blocks against 64 warps. What actually transfers is `ceil(warps/blocks)`, and
    /// this pins that it is 2 on every target the retarget covers — so the constant is *correct*
    /// everywhere, not merely unchanged. The 8.6/8.7 row is the counter-example that proves the
    /// derivation has content (it wants 3), and it is unreachable only because [`FLASH_TILE_MIN`] is 0.
    #[test]
    fn flash_warps_is_the_derived_ada_packing() {
        // (cc, max warps/SM, max blocks/SM, derived W)
        for (cc, warps, blocks, want) in [
            ((7, 0), 64u32, 32u32, 2u32), // Volta
            ((7, 5), 32, 16, 2),          // Turing
            ((8, 0), 64, 32, 2),          // A100
            ((8, 6), 48, 16, 3),          // GA10x — the one target the derivation disagrees on
            ((8, 9), 48, 24, 2),          // Ada
            ((9, 0), 64, 32, 2),          // H100
            ((12, 0), 48, 24, 2),         // Blackwell consumer
            ((13, 7), 64, 32, 2),         // unknown/newer: the conservative datacenter fallback
        ] {
            assert_eq!(occupancy_limits(cc.0, cc.1), (warps, blocks), "cc {cc:?}");
            assert_eq!(
                flash_untiled_warps(cc.0, cc.1),
                want,
                "cc {cc:?}: ceil({warps}/{blocks})"
            );
        }
        assert_eq!(
            FLASH_WARPS,
            flash_untiled_warps(8, 9),
            "FLASH_WARPS must stay the derived Ada packing"
        );
        assert_eq!(
            FLASH_WARPS, 2,
            "the shipped untiled PTX and gpu::flash_plan_forced both bake 2"
        );
        for cc in [(8, 0), (8, 9), (9, 0)] {
            assert_eq!(
                flash_untiled_warps(cc.0, cc.1),
                FLASH_WARPS,
                "cc {cc:?} derives a different packing — flash_plan_forced and the PTX would have to \
                 start reading flash_untiled_warps TOGETHER (see the FLASH_WARPS doc comment)"
            );
        }
    }

    /// [`FLASH_TWARPS`] is a *reuse factor*, not an occupancy quantity, so the occupancy limits only
    /// have to not veto it: `32·W <= 1024` threads/CTA, and the CTA count it implies must stay inside
    /// every target's blocks-per-SM cap.
    #[test]
    fn flash_twarps_is_legal_on_every_target() {
        let threads = 32 * FLASH_TWARPS;
        assert!(
            threads <= 1024,
            "{threads}-thread CTA exceeds the 1024-thread block limit"
        );
        for cc in [(7, 0), (7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (12, 0)] {
            let (warps_sm, blocks_sm) = occupancy_limits(cc.0, cc.1);
            assert!(
                FLASH_TWARPS <= warps_sm,
                "cc {cc:?}: a {FLASH_TWARPS}-warp CTA does not fit {warps_sm} warps/SM"
            );
            assert!(
                warps_sm / FLASH_TWARPS <= blocks_sm,
                "cc {cc:?}: {FLASH_TWARPS}-warp CTAs would need more than {blocks_sm} blocks/SM to \
                 fill the warp pool"
            );
        }
    }

    /// **The grid's SMEM arithmetic, emission form and window discipline (no GPU).** Four things a
    /// wrong dynamic-SMEM kernel gets wrong silently, each checked from the text:
    ///   * the closed form `stages·2·BK·D·2` and the 48 KiB boundary that splits static from dynamic;
    ///   * `.extern .shared` sits at **module scope** — the identical line inside the entry body is
    ///     `CUDA_ERROR_INVALID_PTX` (measured) — so its offset must precede `.visible .entry`;
    ///   * exactly **one** window per module (two module-scope externs alias);
    ///   * a dynamic entry declares **no** static `.shared` (it would count against the same opt-in
    ///     ceiling and shrink the window), and a static entry never names the window.
    #[test]
    fn flash_stage_grid_smem_math_and_modes() {
        let expect: [(usize, bool); 13] = [
            (12_288, false),
            (16_384, false),
            (16_384, false),
            (32_768, false),
            (49_152, false),
            (65_536, true),
            (65_536, true),
            (49_152, false),
            (65_536, true),
            (81_920, true),
            (49_152, false),
            (65_536, true),
            (98_304, true),
        ];
        assert_eq!(FLASH_STAGE_VARIANTS.len(), expect.len());
        let mut names: Vec<&str> = FLASH_STAGE_VARIANTS.iter().map(|v| v.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            FLASH_STAGE_VARIANTS.len(),
            "a duplicate grid name is a duplicate module-cache key"
        );
        for (v, (bytes, dynamic)) in FLASH_STAGE_VARIANTS.iter().zip(expect) {
            assert_eq!(v.smem_bytes(), bytes, "{}: SMEM closed form", v.name);
            assert_eq!(
                v.smem_bytes(),
                64 * v.stages * v.nkb * v.d,
                "{}: the two spellings of the closed form must agree",
                v.name
            );
            assert_eq!(
                v.smem_mode().is_dynamic(),
                dynamic,
                "{}: emission form at {bytes} B",
                v.name
            );
            assert_eq!(
                v.smem_mode().launch_bytes(),
                if dynamic { bytes } else { 0 },
                "{}: a static entry must launch with 0 dynamic bytes, or its tile is reserved twice",
                v.name
            );
            // Every row must be runnable on all three retarget budgets — the point of the grid is that
            // the SAME PTX validates here and runs there.
            for (label, budget) in [
                ("4050", ADA_OPTIN_BUDGET),
                ("A100", A100_OPTIN_BUDGET),
                ("H100", H100_OPTIN_BUDGET),
            ] {
                assert!(v.fits(budget), "{}: does not fit {label}", v.name);
            }
            let (ptx, mode) = flash_stage_ptx(v, ADA_OPTIN_BUDGET);
            assert_eq!(mode, v.smem_mode());
            assert!(
                ptx.starts_with(crate::ptx_target::HDR_SM80),
                "{}: the ring adds no instruction past the Ampere floor",
                v.name
            );
            assert_eq!(
                ptx.matches(".visible .entry ").count(),
                1,
                "{}: one row is one single-entry module",
                v.name
            );
            assert!(
                ptx.contains(&format!(".visible .entry {}(", v.name)),
                "{}: the grid key must be the entry symbol",
                v.name
            );
            assert_eq!(
                ptx.matches('{').count(),
                ptx.matches('}').count(),
                "{}: unbalanced braces",
                v.name
            );
            assert_eq!(
                ptx.matches(".extern .shared").count(),
                usize::from(dynamic),
                "{}",
                v.name
            );
            if dynamic {
                let (decl, entry) = (
                    ptx.find(".extern .shared").expect("window"),
                    ptx.find(".visible .entry").expect("entry"),
                );
                assert!(
                    decl < entry,
                    "{}: the window must be declared at MODULE scope",
                    v.name
                );
                assert!(
                    !ptx.contains(".shared .align 16 .b8 smem_"),
                    "{}: no statics beside the window",
                    v.name
                );
                assert!(
                    ptx.contains(&format!("    mov.u32 %sbase,{DSMEM_SYM};")),
                    "{}: SMEM must be addressed through the window symbol, never a literal base",
                    v.name
                );
            } else {
                assert!(
                    ptx.contains(&format!(".shared .align 16 .b8 smem_{}[{bytes}];", v.name)),
                    "{}: static rows keep the historical `.shared` spelling",
                    v.name
                );
                assert!(
                    !ptx.contains(DSMEM_SYM),
                    "{}: a static kernel must not touch the window",
                    v.name
                );
            }
        }
    }

    /// **The ring's `cp.async` bookkeeping, read out of the text.** These four facts are the whole
    /// correctness argument for depth > 2 and every one of them fails silently on a device (a wrong
    /// `wait_group` reads a half-filled buffer; a missing tail commit shifts every later group index):
    ///   * one committed group per prologue slab **plus** one per body = `stages` textual commits;
    ///   * `wait_group stages-1`, and no other wait depth in the kernel;
    ///   * exactly `stages-2` guarded prologue branches — block 0 needs no guard, and the guard must
    ///     never enclose the commit;
    ///   * add+wrap cursors, never the two-buffer swap (`%bswap`), which only cycles two slots.
    #[test]
    fn flash_stage_grid_ring_discipline() {
        for v in FLASH_STAGE_VARIANTS {
            let (ptx, _) = flash_stage_ptx(v, ADA_OPTIN_BUDGET);
            let n = v.name;
            assert!(v.stages >= 3, "{n}: the grid is the DEPTH grid");
            assert_eq!(
                ptx.matches("cp.async.commit_group;").count(),
                v.stages,
                "{n}: one commit per prologue slab plus one per body"
            );
            let waits: Vec<&str> = ptx
                .match_indices("cp.async.wait_group ")
                .map(|(i, m)| {
                    let rest = &ptx[i + m.len()..];
                    &rest[..rest.find(';').expect("wait_group takes an operand")]
                })
                .collect();
            assert!(!waits.is_empty(), "{n}: the ring must drain");
            for w in &waits {
                assert_eq!(
                    *w,
                    (v.stages - 1).to_string(),
                    "{n}: every drain must be wait_group stages-1"
                );
            }
            assert_eq!(
                ptx.matches(&format!("_{n}:\n    cp.async.commit_group;")).count(),
                v.stages - 2 + 1,
                "{n}: the guarded prologue slabs and the body's NOPF land must each commit OUTSIDE \
                 the guard"
            );
            for i in 1..v.stages - 1 {
                assert!(
                    ptx.contains(&format!("@%p0 bra PS{i}_{n};")),
                    "{n}: prologue slab {i} must be guarded against a short S"
                );
            }
            assert!(
                !ptx.contains("mov.u32 %bswap,%bufc;"),
                "{n}: the two-buffer swap only cycles two of {} slots",
                v.stages
            );
            let ring = v.smem_bytes();
            let slab = ring / v.stages;
            for reg in ["%bufc", "%bufp"] {
                assert!(
                    ptx.contains(&format!("    add.u32 {reg},{reg},{slab};\n    setp.ge.u32 %p0,{reg},{ring};\n    @%p0 sub.u32 {reg},{reg},{ring};\n")),
                    "{n}: {reg} must advance one slot and wrap at the end of the ring"
                );
            }
            assert!(
                ptx.contains(&format!(
                    "    add.u32 %next,%kb,{};\n",
                    (v.stages - 1) * v.bk()
                )),
                "{n}: the prefetch must run stages-1 blocks ahead"
            );
            assert!(
                ptx.contains(&format!("    mov.u32 %bufp,{};\n", (v.stages - 1) * slab)),
                "{n}: the prefetch cursor starts at the slot of block stages-1"
            );
            // The feed: `_lm` rows must have replaced BOTH hand-packed gathers, and the counts follow
            // from the tiling (K: ntile x ktq, V: nto x nkb).
            let (ntile, ktq, nto) = (2 * v.nkb, v.d / 16, v.d / 8);
            if v.pv_ldmatrix {
                assert_eq!(
                    ptx.matches("ldmatrix.sync.aligned.m8n8.x2.shared.b16")
                        .count(),
                    ntile * ktq,
                    "{n}: one K ldmatrix per (score n-tile, hdim k-tile)"
                );
                assert_eq!(
                    ptx.matches("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16")
                        .count(),
                    nto * v.nkb,
                    "{n}: one transposing V ldmatrix per (output d-tile, key k-tile)"
                );
                assert!(
                    !ptx.contains("ld.shared.u16"),
                    "{n}: the strided hand-packed V gather is what _lm removes"
                );
            } else {
                assert!(!ptx.contains("ldmatrix"), "{n}: hand-packed feed");
            }
            // Causal rows must actually mask, and only the narrow generator derives the mask.
            assert_eq!(
                ptx.contains("selp.f32 %s0_0,0fFF800000,%s0_0"),
                v.causal,
                "{n}: causal masking must be present iff the row says causal"
            );
            assert!(
                !v.causal || v.nkb == 1,
                "{n}: the wide tile has no diagonal mask"
            );
        }
    }

    /// **Depth is a pure pipeline change: the QKt / softmax / PV emission is byte-identical.** The
    /// deeper ring is only allowed to move *when* a slab is staged. If a future edit slipped a
    /// register or an address change into the compute region under the `stages` branch, the kernel
    /// would still assemble and still look right — and would differ numerically from the shipped `mp`
    /// at exactly the depths nothing on this laptop can run.
    #[test]
    fn flash_deep_ring_changes_only_the_pipeline() {
        let cap = STATIC_SMEM_CAP;
        for (shallow, deep, ship_name, deep_name, slab) in [
            (
                entry_mma_reg_pipe(64, false, false, 2, cap),
                entry_mma_reg_pipe(64, false, false, 3, cap),
                "flash_d64_mp",
                "flash_d64_mp_s3",
                4096usize,
            ),
            (
                entry_mma_reg_pipe_wide(64, 4, 2, false, cap),
                entry_mma_reg_pipe_wide(64, 4, 3, false, cap),
                "flash_d64_mpw4",
                "flash_d64_mpw4_s3",
                16_384,
            ),
        ] {
            let a = shallow.replace(ship_name, "K");
            let b = deep.replace(deep_name, "K");
            let start = "    mov.f32 %s0_0,0f00000000;\n";
            let a_body = &a[a.find(start).expect("QKt section")
                ..a.find("    mov.u32 %bswap,%bufc;").expect("s2 advance")];
            let b_body = &b[b.find(start).expect("QKt section")
                ..b.find(&format!("    add.u32 %bufc,%bufc,{slab};"))
                    .expect("ring advance")];
            assert_eq!(
                a_body, b_body,
                "{deep_name}: the deep ring must change ONLY the staging/drain, not the math"
            );
        }
    }

    /// **An over-budget row is a loud generation failure, never a clamped launch.** The budget is the
    /// device's `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`; a kernel past it cannot run at all, and the
    /// driver's own rejection (`CUDA_ERROR_INVALID_VALUE` at launch) names neither the kernel nor the
    /// ceiling. The dispatcher's decline point is [`FlashStageCfg::fits`], checked before generation.
    #[test]
    #[should_panic(expected = "exceeds the budget")]
    fn flash_stage_over_budget_panics_at_generation() {
        // D=128, BK=64, 3 stages = 96 KiB; a 64 KiB budget (a Turing-class opt-in) cannot hold it.
        let v = FLASH_STAGE_VARIANTS
            .iter()
            .find(|v| v.name == "flash_d128_mpw4_lm_s3")
            .expect("grid row");
        assert!(!v.fits(64 * 1024), "the decline test must reject it first");
        let _ = flash_stage_ptx(v, 64 * 1024);
    }

    /// **A causal wide row is rejected, not silently emitted unmasked.** The wide generator scores a
    /// BK-key tile with no diagonal predicate, so "causal" there would attend to future keys and still
    /// produce a finite, plausible-looking result — the exact failure mode `8a767e1` established must
    /// be a decline.
    #[test]
    #[should_panic(expected = "causal masking is derived only")]
    fn flash_stage_grid_rejects_causal_wide() {
        let v = FlashStageCfg {
            name: "flash_probe_causal_wide",
            d: 64,
            nkb: 2,
            stages: 3,
            causal: true,
            pv_ldmatrix: false,
        };
        let _ = flash_stage_ptx(&v, ADA_OPTIN_BUDGET);
    }
}
