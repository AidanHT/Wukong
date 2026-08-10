//! `ptx_wgmma` — the **Hopper warpgroup-MMA + TMA GEMM generator** ("Act 2").
//!
//! # Why this family exists, in one number
//!
//! `docs/gpu/derive/D1_h100_gemm.md` section 2.5 derives, from issue rate, SMEM fill bandwidth and
//! L2 fill bandwidth, the *binding ceiling* of every `mma.sync` + `cp.async` configuration this
//! backend can express on an H100 SXM5:
//!
//! | config | binding ceiling | vs cuBLAS @4096 cubed (716 TFLOPS) |
//! |---|---|---|
//! | 128x128 w24 (what `ptx_wmma` ships) | 525 TFLOPS (L2 fill) | 73% |
//! | 128x256 w24 (the widest Act-1 tile) | 642 TFLOPS (mma issue) | 90% |
//! | `wgmma` m64n256k16 | ~879 TFLOPS | 123% |
//!
//! **Act 1 cannot reach parity with cuBLAS on Hopper. That is arithmetic, not pessimism**, and no
//! amount of tuning inside `ptx_wmma` changes it: at 128 wide the L2 fill path binds, and widening
//! to 256 moves the bind to `mma.sync`'s own issue rate, which tops out at 90%. `wgmma` is the only
//! instruction that clears the bar, because one `wgmma.m64n256k16` replaces 128 `mma.sync.m16n8k16`
//! issues with a single warpgroup-wide operation whose operands come straight from shared memory.
//!
//! # What this module is, and is not
//!
//! It is a **PTX text generator plus the pure arithmetic that feeds it**: the shared-memory matrix
//! descriptor, the stage/tile lattice, the register and SMEM budgets, and a [`LaunchPlan`] that says
//! exactly how the resulting entry must be launched. It contains no launcher: no `Gpu`, no buffers,
//! no `cuLaunchKernel`. Wiring it into `gpu.rs` is a separate, mechanical commit, and the seam is
//! [`LaunchPlan`] + [`WgmmaCfg::tensor_map_a`]/[`WgmmaCfg::tensor_map_b`] + [`require_sm90a`].
//!
//! # The four hard ISA facts the design is built on (PTX ISA section 9.7.16, D1 section 1.4)
//!
//! 1. **A warpgroup is four contiguous warps** (128 threads) whose first warp-rank is a multiple of
//!    4. Every `wgmma` is warpgroup-wide and `.aligned`: all 128 threads must execute it.
//! 2. **The shape menu for 16-bit dense inputs is `m64n{8,16,..,256}k16`** -- M is *always* 64, K is
//!    *always* 16, N is any multiple of 8 up to 256. A 128-row CTA tile is therefore two warpgroups
//!    of 64 rows each, never one warpgroup of 128. See [`WgmmaShape`].
//! 3. **B always comes from shared memory.** There is an SS form (A and B both from SMEM) and an RS
//!    form (A from registers, B from SMEM); there is no form with B in registers. This generator
//!    emits the SS form, so both operands are addressed by a 64-bit [`SmemDesc`].
//! 4. **`wgmma` requires `.target sm_90a`**, which is architecture-*specific*: the module is legal on
//!    Hopper and on nothing else, in either direction. That makes this a separate module family with
//!    its own header ([`crate::ptx_target::HDR_SM90A_V80`]) and its own capability gate
//!    ([`require_sm90a`]), never a retag of an existing family.
//!
//! # The fifth fact, learned the hard way on 2026-08-10: CORE MATRICES ARE PACKED
//!
//! The first H100 round (`bench/gpu/h100/2026-08-10-h100-s2a-bringup.log`) proved TMA writes the tile
//! **plain row-major within the box** and then found that *neither* reading of the descriptor's two
//! offset fields computed the right product. The reason is not the axis naming. An SS-form `wgmma`
//! operand is not a monolithic strided tile at all: it is a grid of **core matrices**, and for a
//! 16-bit type one core matrix is `8 rows x 16 bytes` stored as **128 CONTIGUOUS bytes**. The
//! hardware's address for element `(r, c)` of core matrix `(i, j)` is
//!
//! ```text
//! start + i * SBO + j * LBO + r * 16 + c * elem      (SWIZZLE_NONE)
//! ```
//!
//! -- the within-core-matrix row stride is the fixed **16**, not the tile's row pitch. LBO is the
//! distance between K-adjacent core matrices and SBO the distance between MN-adjacent ones. So a
//! row-major tile whose rows are wider than one core matrix (any `BK > 8` for 16-bit) is
//! **unrepresentable by any (LBO, SBO) pair**, and the wrong-by-`r*(row_bytes-16)` reading is exactly
//! why the shipped arm scored 64 of 4096 output lanes exact: only rows with `m % 8 == 0` (and columns
//! with `n % 8 == 0`) have `r == 0`. `the_row_major_reading_is_unrepresentable_and_predicts_the_log`
//! re-derives that 64 device-free, from the log.
//!
//! Two layouts a descriptor *can* describe are therefore on the table, and [`SmemLayout`] is the one
//! authority that spells both:
//!
//! * [`SmemLayout::CanonicalNone`] -- the ISA's own packing, core matrices at 128 contiguous bytes.
//!   TMA cannot write it with one tiled copy, so it is a **probe-only** arm: it names the fix.
//! * [`SmemLayout::Swizzle128`] -- **what production ships.** At `BK = 64` for a 16-bit type the SMEM
//!   row is exactly 128 B, which is the 128-B swizzle atom, and the ISA's canonical 128-B-swizzle
//!   K-major layout is *precisely* plain row-major plus the XOR chunk permutation -- which is
//!   precisely what a `CU_TENSOR_MAP_SWIZZLE_128B` TMA copy writes. Row stride 128 B, k-group stride
//!   16 B, `SBO = 8 * 128 = 1024`, no repack and no extra copies. It is also the configuration
//!   CUTLASS ships for every SM90 16-bit mainloop, which is not a coincidence.
//!
//! **Both are still unconfirmed on silicon**, which is what `wgmma_hopper_bringup` stage D -- the
//! one-visit descriptor sweep over [`desc_sweep_candidates`] -- exists to settle.
//!
//! # The sixth fact, added 2026-08-10 evening: THE CLUSTER IS A CORRECTNESS PROTOCOL
//!
//! [`Multicast::ClusterA`] used to be an [`UNSUPPORTED`] decline. It is now generated, because the
//! Act-2 round measured W1 **minus its cluster** at 58.8-67.5% of cuBLAS against a 95-108%
//! prediction and named the missing cluster as the first suspect in its own provenance line. What
//! the arm costs is four mechanisms, each of whose failure mode is silence or a hang rather than an
//! error, so each is stated here and gated separately:
//!
//! 1. **The A tile is SPLIT, not leader-loaded.** Each CTA of the cluster TMA-loads
//!    `BM / cluster_ctas` rows of A and multicasts them to the whole cluster, so the issue work is
//!    balanced and the assembled SMEM image is byte-for-byte the image one unclustered copy would
//!    have written. That is what leaves the descriptor, the per-consumer `m64` slabs and the
//!    epilogue completely unchanged -- the cluster is a *traffic* change, not a layout change.
//! 2. **The transaction count is PER DESTINATION.** A multicast copy of `n` bytes performs a
//!    `complete-tx` of `n` on the mbarrier of *every* destination CTA; it does not divide `n` among
//!    them. So each CTA still declares `tile_a + tile_b` -- see [`WgmmaCfg::stage_tx_bytes`], which
//!    carries the rule. A count that disagrees with the copies hangs; it does not fail.
//! 3. **The `empty` barrier becomes cluster-scoped.** A producer overwrites an A slice every CTA in
//!    the cluster reads, so it may not recycle a stage until every consumer *in the cluster* is
//!    done. Consumers therefore arrive at each CTA's `empty[s]` through `mapa.shared::cluster`, and
//!    the barrier is initialised with [`WgmmaCfg::empty_arrivals`] rather than `consumer_wgs`.
//! 4. **Two cluster rendezvous bracket the kernel.** One after `mbarrier.init` (plus
//!    `fence.mbarrier_init.release.cluster`), because a peer may not signal a barrier that is not
//!    yet initialised; one before `ret`, because a CTA that has exited has no shared memory for a
//!    peer to write into.
//!
//! None of it can be executed on this machine. What *can* be proven here is the mask arithmetic, the
//! transaction arithmetic, the slice geometry, the grid divisibility and the emitted text; the
//! device claim is `gpu::tests::wgmma_cluster_multicast_is_exact`, which demands `==` on a shape
//! whose two cluster CTAs hold different N halves before anything is timed.
//!
//! # The capability gate is a TYPE, not a convention
//!
//! The fp8 families are gated by a textual law (`every_fp8_module_load_is_capability_gated`): every
//! function that reaches the generator surface and loads a module must call `require_fp8` first. That
//! law works, but it is a scan, and a scan can only see the files in its corpus.
//!
//! This family is gated one level down instead: [`wgmma_module`] **cannot be called without an
//! [`Sm90aLicense`]**, and an `Sm90aLicense` can only be produced by a function that has looked at a
//! probed compute capability and found `9.x`. A future launcher that "forgets" the gate does not
//! produce ungated PTX -- it does not compile. The textual law
//! (`every_sm90a_emitter_demands_the_license`) is kept as a second line of defence over this file.
//!
//! # What is NOT built here, and declines loudly rather than guessing
//!
//! * **A layout TMA cannot write.** [`SmemLayout::CanonicalNone`] needs the 8 rows of every core
//!   matrix packed into 128 contiguous bytes, which one tiled TMA copy of a row-major operand does
//!   not produce (it would take `BK / 8` copies per operand per stage, or an SMEM repack). The
//!   *generator* therefore declines it and the sweep probe stages it by hand -- the arm exists to
//!   name the fix, not to ship.
//! * **The pingpong schedule** (D1's W3 as specified). [`WGMMA_W3C`] is the same 128x128x64 tile at 6
//!   stages under the cooperative schedule, which is legal (CTA-M 128 is a multiple of 128) but is
//!   not the two-consumer alternating schedule CUTLASS calls pingpong.
//! * **W2 (256x128x64) and W4 (64x256x64)** are not table rows. W2 is expressible as-is -- 4
//!   consumer warpgroups, 640 threads -- but only at `consumer_regs <= 120`, because
//!   `128 * 32 + 512 * R <= 65536`; nothing has confirmed a 4-warpgroup split at that width, and
//!   `WgmmaCfg::validate` will refuse the row rather than let it be guessed. W4 needs a
//!   *non-cooperative* schedule outright, because cooperative is illegal below CTA-M 128 (D1
//!   section 3.2). Both are follow-ups.
//!
//! # What no test here can prove
//!
//! There is no Hopper part on this machine, and `sm_90a` cannot be JITed, emulated or `ptxas`-checked
//! anywhere in this environment. **Nothing below proves this PTX runs.** What it does prove is the
//! descriptor arithmetic, the shape menu, the warpgroup/thread/register/SMEM budgets, the stage
//! lattice, the ASCII rule, the `.version`/`.target` floor and the capability gate -- everything that
//! is arithmetic or text. The first H100 hour's job is listed in `WGMMA_DEVICE_VALIDATION`.

use crate::ptx_target::HDR_SM90A_V80;
use crate::tma_host::{ptx_param_address, ptx_param_decl, TensorMapArgs, TmaDataType, TmaSwizzle};

/// The prefix every decline from this module carries, mirroring `lower::UNSUPPORTED`: a shape this
/// generator cannot express is a **skip**, never wrong PTX.
pub const UNSUPPORTED: &str = "ptx_wgmma::UNSUPPORTED";

/// The compute capability `wgmma` needs. Hopper, and only Hopper -- see [`Sm90aLicense`].
pub const WGMMA_MIN_CC: (i32, i32) = (9, 0);

/// Shared memory per CTA on an H100, bytes (`MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` = 227 KiB).
///
/// The PTX ISA allows a *static* `.shared` array up to 228 KB on `sm_90a` (section 5.1.7, quoted in
/// D1 section 1.6), but 227 KiB is what the device actually grants a block, so it is the binding
/// number and the one every budget below is checked against. The generator still emits the
/// **dynamic** window rather than a large static array: that path is already plumbed in this crate
/// (`Gpu::function_dyn` + `dyn_launch_cfg` + `Gpu::smem_budget`, verified live at 64 KiB on the Ada
/// part) and it does not depend on the architecture-specific static extension.
pub const HOPPER_SMEM_PER_CTA: usize = 227 * 1024;

/// Registers per CTA -- 64 Ki 32-bit registers, on every part from Volta to Hopper. The
/// warp-specialised register split is checked against this.
pub const REGS_PER_CTA: u32 = 65536;

/// Threads in one warpgroup: four contiguous warps.
pub const WARPGROUP_THREADS: usize = 128;

/// The module-scope dynamic shared-memory window this family carves its stages out of.
///
/// Deliberately a **local** literal rather than `crate::gpu::DSMEM_DECL`: that constant lives behind
/// the `gpu` feature, and keeping this module un-gated is what lets its ASCII, shape and descriptor
/// gates run in a plain, toolchain-free `cargo test`. The duplication is closed by
/// `the_window_declaration_agrees_with_the_host_side_constant`, which asserts the two are the same
/// string whenever the feature is on -- the same "literal plus a gate" pattern the crate uses for
/// literal PTX headers.
///
/// The alignment is **1024**, not the host constant's 16. A TMA destination must be 128-byte aligned,
/// and the 128-B swizzle pattern must begin on a 1024-byte boundary; declaring the window at 1024
/// makes every stage base (all of which are multiples of 1024, asserted) inherit both.
pub const WGMMA_DSMEM_DECL: &str = ".extern .shared .align 1024 .b8 wk_dsmem[];\n";

/// The symbol [`WGMMA_DSMEM_DECL`] declares.
pub const WGMMA_DSMEM_SYM: &str = "wk_dsmem";

// --- capability -----------------------------------------------------------------------------------

/// **Proof that the device this code is being generated for is a Hopper part.**
///
/// Holding one is the precondition of [`wgmma_module`], so `sm_90a` PTX cannot be produced by a code
/// path that has not established the capability. It is deliberately not `Default`, not
/// `Deserialize`, and has no public field: the only ways to get one are [`require_sm90a`], which
/// reads a probed [`crate::gpu::GpuTarget`], and [`Sm90aLicense::for_probed_cc`], whose contract is
/// that the capability it is handed came from a probe.
///
/// **The check is `major == 9`, not `>= (9, 0)`.** Every other capability floor in this crate is a
/// lower bound because the ISA is forward compatible; `sm_90a` is not. An `sm_90a` module fails to
/// load on `sm_100` exactly as it fails on `sm_89`, so a `>=` gate would hand a Blackwell part a
/// module it cannot run and blame the JIT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sm90aLicense {
    cc: (i32, i32),
}

impl Sm90aLicense {
    /// Grant a license for a **probed** compute capability.
    ///
    /// The caller's obligation is that `cc` came from `Gpu::target().cc()` and not from a literal;
    /// [`require_sm90a`] is the version that discharges it. Tests and the device-free module
    /// enumeration pass `(9, 0)` deliberately -- the license only ever asserts "this capability is a
    /// Hopper one", which for a literal `(9, 0)` is simply true.
    pub fn for_probed_cc(cc: (i32, i32)) -> Result<Self, String> {
        if cc.0 == 9 {
            Ok(Self { cc })
        } else if cc < WGMMA_MIN_CC {
            Err(format!(
                "wgmma requires cc>=9.0, device is {}.{}",
                cc.0, cc.1
            ))
        } else {
            Err(format!(
                "wgmma is architecture-locked to sm_90a: cc {}.{} is past Hopper and cannot load an \
                 sm_90a module",
                cc.0, cc.1
            ))
        }
    }

    /// The capability this license was granted for.
    pub fn cc(&self) -> (i32, i32) {
        self.cc
    }
}

/// **The capability gate.** `Ok` with a license iff the probed device is Hopper; otherwise a
/// [`crate::gpu::GpuError::Unsupported`] naming the capability, decided before any PTX exists.
///
/// Call it as the first statement of any wgmma launch wrapper, exactly as `require_fp8` is called.
/// The below-Hopper arm goes through `Gpu::require_cap` so its wording is the crate's single decline
/// format; the above-Hopper arm is this family's own, because no other family has an upper bound.
#[cfg(feature = "gpu")]
pub fn require_sm90a(
    g: &crate::gpu::Gpu,
    what: &str,
) -> Result<Sm90aLicense, crate::gpu::GpuError> {
    g.require_cap(what, "wgmma (sm_90a)", WGMMA_MIN_CC)?;
    Sm90aLicense::for_probed_cc(g.target().cc()).map_err(|m| {
        crate::gpu::GpuError::Unsupported(format!("{what}: {m} ({})", g.target().name))
    })
}

/// **The one place in this crate that spells the `sm_90a` module header**, and the reason the
/// capability law stays a one-line scan even though this file now emits *two* Hopper module families
/// (the GEMM mainloop and the TMA bring-up probe).
///
/// Taking the license by reference is not decoration: it makes "produce an `sm_90a` header" a
/// privileged operation in the type system, so a third emitter added later inherits the gate by
/// construction. `every_sm90a_emitter_demands_the_license` pins both halves — that
/// [`HDR_SM90A_V80`] is named nowhere else, and that every function reaching this one carries the
/// witness in its own signature.
fn sm90a_header(_license: &Sm90aLicense) -> &'static str {
    HDR_SM90A_V80
}

// --- the wgmma shape menu -------------------------------------------------------------------------

/// The input element type. `.dtype` for f16 inputs may be `.f16` or `.f32`; for bf16 it is `.f32`
/// only -- this generator always accumulates in f32, which is legal for both and is what every
/// caller in this backend wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WgmmaDtype {
    F16,
    Bf16,
}

impl WgmmaDtype {
    /// The PTX operand-type spelling (`.f32.f16.f16` / `.f32.bf16.bf16`).
    pub const fn mma_types(self) -> &'static str {
        match self {
            WgmmaDtype::F16 => "f32.f16.f16",
            WgmmaDtype::Bf16 => "f32.bf16.bf16",
        }
    }
    pub const fn tma(self) -> TmaDataType {
        match self {
            WgmmaDtype::F16 => TmaDataType::F16,
            WgmmaDtype::Bf16 => TmaDataType::Bf16,
        }
    }
    pub const fn size(self) -> usize {
        2
    }

    /// **The largest integer this input type holds exactly** — 2048 for f16 (11 significand bits),
    /// **256** for bf16 (8).
    ///
    /// The two are the same width and are eight bits apart in precision, which is the trap: a
    /// bring-up ramp calibrated on f16 runs on the bf16 twin, rounds, and turns an exact `==` verdict
    /// into an unexplained near-miss. `bringup_operands` is checked against this per row.
    pub const fn exact_integer_limit(self) -> f32 {
        match self {
            WgmmaDtype::F16 => 2048.0,
            WgmmaDtype::Bf16 => 256.0,
        }
    }
}

/// One entry of the `wgmma` shape menu, validated on construction.
///
/// M is always 64 and K is always 16 for 16-bit dense inputs; only N varies, over every multiple of
/// 8 from 8 to 256. Nothing else exists, so [`WgmmaShape::new`] rejects rather than rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgmmaShape {
    n: usize,
}

impl WgmmaShape {
    pub const M: usize = 64;
    pub const K: usize = 16;
    pub const N_MIN: usize = 8;
    pub const N_MAX: usize = 256;
    pub const N_STEP: usize = 8;

    pub fn new(n: usize) -> Result<Self, String> {
        if !(Self::N_MIN..=Self::N_MAX).contains(&n) || !n.is_multiple_of(Self::N_STEP) {
            return Err(format!(
                "{UNSUPPORTED}: wgmma N={n} is not on the ISA menu \
                 (m64n{{8,16,..,256}}k16: every multiple of 8 from 8 to 256)"
            ));
        }
        Ok(Self { n })
    }

    pub const fn n(&self) -> usize {
        self.n
    }

    /// The `.m64nNk16` shape token.
    pub fn token(&self) -> String {
        format!("m{}n{}k{}", Self::M, self.n, Self::K)
    }

    /// **f32 accumulator registers per thread**: the warpgroup's `64 x N` output spread over 128
    /// threads is `N / 2` registers each. For N=256 that is 128 registers of accumulator alone,
    /// which is why the consumer warpgroups need `setmaxnreg.inc` and why 4 warpgroups of 64 rows
    /// would not fit.
    pub const fn accum_regs(&self) -> usize {
        self.n / 2
    }
}

// --- the shared-memory matrix descriptor ----------------------------------------------------------

/// The descriptor's 2-bit swizzle field.
///
/// **LANDMINE: this encoding runs BACKWARDS relative to `CUtensorMapSwizzle`.** The descriptor
/// numbers them `0 none, 1 = 128 B, 2 = 64 B, 3 = 32 B`; the TMA descriptor numbers them
/// `0 none, 1 = 32 B, 2 = 64 B, 3 = 128 B`. The two are describing the *same choice* about the *same
/// bytes*, they must agree, and the naive `as u32` cast between them silently swaps 32-B and 128-B.
/// [`SmemSwizzle::from_tma`] is the only conversion, and
/// `the_descriptor_swizzle_encoding_is_reversed_from_tmas` pins both tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum SmemSwizzle {
    None = 0,
    B128 = 1,
    B64 = 2,
    B32 = 3,
}

impl SmemSwizzle {
    /// The matching descriptor mode for a TMA swizzle. The only sanctioned conversion between the
    /// two reversed encodings.
    pub const fn from_tma(s: TmaSwizzle) -> Self {
        match s {
            TmaSwizzle::None => SmemSwizzle::None,
            TmaSwizzle::B32 => SmemSwizzle::B32,
            TmaSwizzle::B64 => SmemSwizzle::B64,
            TmaSwizzle::B128 => SmemSwizzle::B128,
        }
    }
    /// The TMA swizzle this descriptor mode requires the copy to have written.
    pub const fn to_tma(self) -> TmaSwizzle {
        match self {
            SmemSwizzle::None => TmaSwizzle::None,
            SmemSwizzle::B32 => TmaSwizzle::B32,
            SmemSwizzle::B64 => TmaSwizzle::B64,
            SmemSwizzle::B128 => TmaSwizzle::B128,
        }
    }
    /// Byte alignment the *start* of a swizzled matrix must have. The 128-B mode's repeating pattern
    /// is 8 rows of 128 B and must begin on a **1024-byte** boundary; the narrower modes inherit
    /// their own atom.
    pub const fn required_alignment(self) -> u64 {
        match self {
            SmemSwizzle::None => 16,
            SmemSwizzle::B128 => 1024,
            SmemSwizzle::B64 => 512,
            SmemSwizzle::B32 => 256,
        }
    }
}

/// **The 64-bit shared-memory matrix descriptor** a `wgmma` SS-form operand is addressed by.
///
/// Field layout (PTX ISA section 9.7.16, restated in D1 section 1.4):
///
/// | bits | field |
/// |---|---|
/// | 13:0 | start address |
/// | 29:16 | leading dimension byte offset |
/// | 45:32 | stride dimension byte offset |
/// | 51:49 | base offset (3 bits) |
/// | 63:62 | swizzle mode (2 bits) |
///
/// and **every one of the three address/offset fields is encoded as `(x & 0x3FFFF) >> 4`** -- an
/// 18-bit mask followed by a 4-bit right shift, which is to say "byte value, divided by 16, keeping
/// 14 bits". The `>> 4` is silent truncation: a value that is not a multiple of 16 loses its low bits
/// and the hardware reads a different address than the caller computed, with no error anywhere. That
/// is the whole reason [`SmemDesc::pack`] returns a `Result`.
///
/// This packing is the single most load-bearing piece of arithmetic in the family and the only part
/// of it that can be proven without an H100, so it is tested against hand-computed values field by
/// field, at every swizzle mode, at every bit boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmemDesc {
    /// Byte address of the operand's first core matrix, in the **shared** window.
    pub start_addr: u64,
    /// Leading-dimension byte offset -- the distance between K-adjacent core matrices. See
    /// [`SmemLayout`] and [`desc_fields`], which are the only things that decide it.
    pub lbo: u64,
    /// Stride-dimension byte offset -- the distance between MN-adjacent core matrices.
    pub sbo: u64,
    /// 3-bit base offset; the swizzle pattern's phase. Zero for every unswizzled layout.
    pub base_offset: u8,
    pub swizzle: SmemSwizzle,
}

/// Encode one address/offset field: `(x & 0x3FFFF) >> 4`.
///
/// Exposed because it is the operation the *device* also performs on the runtime stage address (the
/// generated PTX does `shr.u64` then `and.b64 0x3FFF`, which is the same function with the mask and
/// the shift commuted), and the two must not drift.
pub const fn encode_desc_field(x: u64) -> u64 {
    (x & 0x3FFFF) >> 4
}

/// **How a `wgmma` SS operand is laid out in shared memory** -- and therefore what its descriptor's
/// two offset fields have to say. THE single authority: [`desc_fields`] turns one of these into the
/// (LBO, SBO, base offset, swizzle mode, per-K-step base advance) tuple, and both the production
/// generator and every sweep candidate read it from there. A new reading is a new variant here, not
/// a fourth place that computes distances.
///
/// The geometry every variant is stated over is fixed by the ISA: a 16-bit core matrix is
/// `8 rows x 16 bytes`, `row_bytes` is the tile's row pitch in shared memory, `rows_per_desc` is how
/// many rows one descriptor covers (64 for A -- each consumer warpgroup owns its own `m64` slab --
/// and the full CTA width for B).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmemLayout {
    /// **Plain row-major within the box** -- what a `SWIZZLE_NONE` TMA tiled copy writes, and what
    /// this family shipped until 2026-08-10.
    ///
    /// It is **not describable**: the hardware's within-core-matrix row stride is the fixed 16 bytes,
    /// so any `row_bytes > 16` (i.e. any `BK > 8` for a 16-bit type) is off by `r * (row_bytes - 16)`
    /// on 7 of every 8 rows no matter what the two fields hold. Kept because the H100 has already
    /// scored both of its readings -- 64/4096 and 0/4096 -- and reproducing those two numbers is how
    /// the sweep validates itself.
    RowMajorNone {
        /// `true` puts the k-adjacent distance (16) in the leading field, `false` the row-group one.
        k_leading: bool,
    },
    /// **The ISA's canonical no-swizzle layout**: every `8 x 16 B` core matrix stored as 128
    /// contiguous bytes, the core-matrix grid traversed K-fastest (`k_fast`) or MN-fastest.
    ///
    /// Describable, and the leading hypothesis -- but **one tiled TMA copy cannot write it**, so
    /// [`wgmma_module`] declines it and only the sweep probe (which repacks by hand) stages it. If
    /// this is the arm that goes exact, the production fix is a layout change, not a field change.
    CanonicalNone {
        k_fast: bool,
        /// The axis-naming coin flip, kept as a candidate rather than as an argument.
        swapped: bool,
    },
    /// **The 128-B XOR swizzle -- what production ships.**
    ///
    /// At `row_bytes == 128` the canonical 128-B-swizzle K-major layout is plain row-major (row
    /// stride 128 B, k-group stride 16 B, `SBO = 8 * 128`) composed with `Swizzle<3,4,3>`, which is
    /// exactly what a `CU_TENSOR_MAP_SWIZZLE_128B` TMA copy writes. No repack, no extra copies, and
    /// the descriptor's own swizzle field tells the hardware to undo the permutation.
    ///
    /// LBO does not appear in that layout at all (the k-group stride is the fixed 16 B), so
    /// `lbo_bytes` is the value the field is *spelled* with; the sweep carries three spellings and
    /// will say whether the hardware reads it.
    ///
    /// `swapped` is the axis-naming flip **again**, and it is here rather than assumed away because
    /// the two independent readings of the ISA disagree exactly at this point: the CUTLASS
    /// descriptor builder assigns the two fields one way for its no-swizzle ("interleave") layout
    /// and the other way for the swizzled ones, while the published tutorial derivation keeps SBO on
    /// the MN axis in both. One candidate settles it; an argument does not.
    Swizzle128 { lbo_bytes: u64, swapped: bool },
}

/// The descriptor fields one [`SmemLayout`] implies, plus the byte distance the descriptor's start
/// address moves per `wgmma` K step (16 elements). Everything a generator needs and nothing it may
/// re-derive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescFields {
    pub lbo: u64,
    pub sbo: u64,
    pub base_offset: u8,
    pub swizzle: SmemSwizzle,
    /// How far the descriptor's start address advances per `wgmma` K step. 32 B for any layout whose
    /// K direction is contiguous in 16-bit elements; a whole pair of core matrices otherwise.
    pub k_step_bytes: u64,
}

/// **THE authority.** One layout in, one field tuple out -- used by [`SmemDesc::for_layout`], by
/// the generator's own descriptor constants and by every row of [`desc_sweep_candidates`], so the
/// reading the H100 crowns reaches production by changing [`SHIPPED_LAYOUT`] and nothing else.
pub fn desc_fields(layout: SmemLayout, row_bytes: u64, rows_per_desc: u64) -> DescFields {
    /// One 16-bit core matrix: 8 rows of 16 bytes, packed.
    const CORE: u64 = 128;
    // Core matrices along K in one tile row, and along MN in one descriptor's slab.
    let ncm_k = row_bytes / 16;
    let ncm_m = rows_per_desc / 8;
    match layout {
        SmemLayout::RowMajorNone { k_leading } => {
            let (a, b) = (16, 8 * row_bytes);
            let (lbo, sbo) = if k_leading { (a, b) } else { (b, a) };
            DescFields {
                lbo,
                sbo,
                base_offset: 0,
                swizzle: SmemSwizzle::None,
                k_step_bytes: 32,
            }
        }
        SmemLayout::CanonicalNone { k_fast, swapped } => {
            // K-fastest: k-adjacent core matrices are 128 B apart, MN-adjacent ones a whole row of
            // core matrices. MN-fastest is the transpose of that packing.
            let (lbo, sbo) = if k_fast {
                (CORE, CORE * ncm_k)
            } else {
                (CORE * ncm_m, CORE)
            };
            let (lbo, sbo) = if swapped { (sbo, lbo) } else { (lbo, sbo) };
            // One wgmma K step is 16 elements = two core matrices along K.
            let step = if k_fast { 2 * CORE } else { 2 * CORE * ncm_m };
            DescFields {
                lbo,
                sbo,
                base_offset: 0,
                swizzle: SmemSwizzle::None,
                k_step_bytes: step,
            }
        }
        SmemLayout::Swizzle128 { lbo_bytes, swapped } => {
            let (lbo, sbo) = if swapped {
                (8 * row_bytes, lbo_bytes)
            } else {
                (lbo_bytes, 8 * row_bytes)
            };
            DescFields {
                lbo,
                sbo,
                base_offset: 0,
                swizzle: SmemSwizzle::B128,
                k_step_bytes: 32,
            }
        }
    }
}

impl SmemLayout {
    /// A stable ASCII spelling for logs and candidate labels.
    pub fn label(self) -> String {
        match self {
            SmemLayout::RowMajorNone { k_leading: true } => {
                "SmemLayout::RowMajorNone { k_leading: true }".into()
            }
            SmemLayout::RowMajorNone { k_leading: false } => {
                "SmemLayout::RowMajorNone { k_leading: false }".into()
            }
            SmemLayout::CanonicalNone { k_fast, swapped } => {
                format!("SmemLayout::CanonicalNone {{ k_fast: {k_fast}, swapped: {swapped} }}")
            }
            SmemLayout::Swizzle128 { lbo_bytes, swapped } => {
                format!("SmemLayout::Swizzle128 {{ lbo_bytes: {lbo_bytes}, swapped: {swapped} }}")
            }
        }
    }

    /// The TMA swizzle the copy that writes this layout must carry. The descriptor's own swizzle
    /// field is the reversed encoding of the same choice ([`SmemSwizzle::from_tma`]).
    pub const fn tma_swizzle(self) -> TmaSwizzle {
        match self {
            SmemLayout::Swizzle128 { .. } => TmaSwizzle::B128,
            _ => TmaSwizzle::None,
        }
    }

    /// **Can one tiled TMA copy leave the operand in this layout?** `false` means the layout needs a
    /// shared-memory repack (or `BK / 8` copies per operand per stage), which the production
    /// generator declines rather than pretending to express.
    pub const fn tma_writable(self) -> bool {
        !matches!(self, SmemLayout::CanonicalNone { .. })
    }
}

impl SmemDesc {
    /// The descriptor for an operand stored in `layout`, at `start_addr`.
    ///
    /// Both GEMM operands of an NT product take the same constructor: A is `M x K` with K
    /// contiguous, and B is `N x K` with K contiguous, which is `B(K x N)` in column-major -- and
    /// column-major B is what `wgmma` expects at `imm-trans-b = 0`. So the NT layout this backend
    /// already stores needs no transpose flag on either operand.
    pub fn for_layout(
        start_addr: u64,
        layout: SmemLayout,
        row_bytes: u64,
        rows_per_desc: u64,
    ) -> Self {
        let f = desc_fields(layout, row_bytes, rows_per_desc);
        Self {
            start_addr,
            lbo: f.lbo,
            sbo: f.sbo,
            base_offset: f.base_offset,
            swizzle: f.swizzle,
        }
    }

    /// Pack to the 64-bit register `wgmma` takes, or say why it cannot be packed.
    ///
    /// Rejects, in order: any of the three byte quantities that is not a multiple of 16 (the `>> 4`
    /// would truncate it); any of them that does not fit its 14-bit field after encoding; a start
    /// address the swizzle mode's pattern cannot begin at; and a base offset past 3 bits.
    ///
    /// The multiple-of-16 rule is checked before the swizzle alignment because it is the more
    /// fundamental of the two: a 16-byte-misaligned matrix is unrepresentable in *every* mode,
    /// swizzled or not, and saying so is more useful than blaming the swizzle.
    pub fn pack(&self) -> Result<u64, String> {
        for (what, v) in [
            ("start address", self.start_addr),
            ("leading byte offset", self.lbo),
            ("stride byte offset", self.sbo),
        ] {
            if v % 16 != 0 {
                return Err(format!(
                    "{UNSUPPORTED}: descriptor {what} {v} is not a multiple of 16; the field is \
                     encoded (x & 0x3FFFF) >> 4, so the low bits would be silently dropped"
                ));
            }
            if v >= 1 << 18 {
                return Err(format!(
                    "{UNSUPPORTED}: descriptor {what} {v} does not fit the 14-bit field (max \
                     {} bytes)",
                    (1u64 << 18) - 16
                ));
            }
        }
        if !self
            .start_addr
            .is_multiple_of(self.swizzle.required_alignment())
        {
            return Err(format!(
                "{UNSUPPORTED}: {:?} swizzle needs the matrix to start on a {}-byte boundary, got \
                 {:#x}",
                self.swizzle,
                self.swizzle.required_alignment(),
                self.start_addr
            ));
        }
        if self.base_offset > 7 {
            return Err(format!(
                "{UNSUPPORTED}: descriptor base offset {} does not fit 3 bits",
                self.base_offset
            ));
        }
        Ok(encode_desc_field(self.start_addr)
            | (encode_desc_field(self.lbo) << 16)
            | (encode_desc_field(self.sbo) << 32)
            | ((self.base_offset as u64) << 49)
            | ((self.swizzle as u64) << 62))
    }

    /// **Where the hardware fetches element `(r, c)` of core matrix `(i, j)` from**, in bytes, under
    /// the unswizzled canonical model.
    ///
    /// `i` is the MN-direction core-matrix index (rows `8i .. 8i+7`), `j` the K-direction one
    /// (columns `8j .. 8j+7` for a 16-bit type), `r` the row **within** the core matrix and `c` the
    /// element within that row. The `r * 16` is the whole point: the within-core-matrix row stride is
    /// a constant the descriptor cannot change, which is what makes a wide row-major tile
    /// unrepresentable.
    ///
    /// This is a **model**. No device verdict consults it -- those are f64 references against a
    /// launched kernel. It exists so the sweep's claims (which candidate should win, and why the
    /// 2026-08-10 log reads 64/4096) are checkable with no Hopper part in the room.
    pub const fn canonical_offset(&self, i: u64, j: u64, r: u64, c: u64, elem: u64) -> u64 {
        self.start_addr + i * self.sbo + j * self.lbo + r * 16 + c * elem
    }

    /// The descriptor with its start-address field zeroed -- the part the host can fold into an
    /// immediate, since only the start address varies per stage at run time.
    ///
    /// The generated kernel completes it with `((addr >> 4) & 0x3FFF) | const_part`, which is exactly
    /// [`encode_desc_field`] for any shared address (shared memory is at most 256 KiB, so the 18-bit
    /// mask never fires and the 14-bit field never overflows -- asserted in
    /// `the_runtime_address_fold_matches_the_host_encoder`).
    pub fn const_part(&self) -> Result<u64, String> {
        let mut z = *self;
        z.start_addr = 0;
        z.pack()
    }

    /// [`SmemDesc::const_part`] under a **stated field encoding**, for the two sweep candidates that
    /// exist to close the encoding question in both directions.
    ///
    /// The ISA says every address/offset field is `(x & 0x3FFFF) >> 4`, and the 2026-08-10 log is
    /// already consistent with that for the start address. [`FieldEncoding::Raw`] and
    /// [`FieldEncoding::Twice`] are the two ways a reader could get the *offset* fields wrong while
    /// the address stays right; one launch each retires both.
    pub fn const_part_as(&self, enc: FieldEncoding) -> Result<u64, String> {
        let f = |v: u64| -> Result<u64, String> {
            let e = match enc {
                FieldEncoding::Standard => encode_desc_field(v),
                FieldEncoding::Raw => v,
                FieldEncoding::Twice => encode_desc_field(encode_desc_field(v)),
            };
            if e >= 1 << 14 {
                return Err(format!(
                    "{UNSUPPORTED}: offset {v} does not fit the 14-bit field under {enc:?}"
                ));
            }
            Ok(e)
        };
        if self.base_offset > 7 {
            return Err(format!(
                "{UNSUPPORTED}: descriptor base offset {} does not fit 3 bits",
                self.base_offset
            ));
        }
        Ok((f(self.lbo)? << 16)
            | (f(self.sbo)? << 32)
            | ((self.base_offset as u64) << 49)
            | ((self.swizzle as u64) << 62))
    }
}

/// How a sweep candidate writes the two offset fields.
///
/// [`FieldEncoding::Standard`] is the ISA's `(x & 0x3FFFF) >> 4` and is what production uses; the
/// other two exist only as sweep candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldEncoding {
    /// `(x & 0x3FFFF) >> 4` -- the ISA's own encoding.
    Standard,
    /// Raw bytes, no shift: the reading in which `>> 4` applies to the start address alone.
    Raw,
    /// The shift applied twice -- the mirror mistake.
    Twice,
}

// --- configuration --------------------------------------------------------------------------------

/// The warp-specialisation schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    /// One producer warpgroup drives TMA; `consumer_wgs` consumer warpgroups each own a 64-row slice
    /// of the CTA tile and run the whole K loop. Legal only when CTA-M is a multiple of 128 (D1
    /// section 3.2, `is_tile_desc_compatible_with_cooperative`).
    Cooperative,
}

/// Cluster multicast of an operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Multicast {
    /// No cluster, no multicast: every CTA loads both of its own tiles.
    None,
    /// **D1's W1 as specified**: a `2x1x1` cluster along the N axis with `.multicast::cluster` on A.
    ///
    /// The two CTAs of a cluster hold adjacent N tiles of the **same** M tile, so they want the
    /// *same* A rows and different B columns. Each CTA TMA-loads `BM / 2` rows of A and multicasts
    /// them to both, so A crosses L2 once per cluster instead of once per CTA; B stays per-CTA and
    /// unmulticast. See [`WgmmaCfg::cluster_ctas`] for the arithmetic and the generator's
    /// `cluster` section for the barrier protocol.
    ClusterA,
}

impl Multicast {
    /// CTAs per cluster. `1` for [`Multicast::None`] -- i.e. no cluster at all, which is the
    /// launch's `1x1x1` and the absence of every directive, register and barrier below.
    pub const fn ctas(self) -> usize {
        match self {
            Multicast::None => 1,
            Multicast::ClusterA => 2,
        }
    }
}

/// **The portable ceiling on CTAs per cluster.** The CUDA programming guide guarantees a maximum
/// cluster size of 8 on every part that supports clusters; anything larger is "non-portable" and
/// needs `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED)` plus a device
/// query. This family declines past it rather than opting in blind.
pub const MAX_PORTABLE_CLUSTER_CTAS: usize = 8;

/// **The `ctaMask` operand of a `.multicast::cluster` copy**: bit `r` set iff the CTA whose
/// `%cluster_ctarank` is `r` is a destination.
///
/// This family always multicasts to the *whole* cluster (every CTA of the cluster wants the same A
/// rows), so the mask is the low `ctas` bits. It is a function rather than a literal because the
/// mask and the cluster shape are two views of one fact: a `2x1x1` cluster with a `0x1` mask would
/// load A into one CTA and leave the other's half of the ring holding whatever the previous
/// iteration left there -- wrong numbers, no error, on exactly half the accumulator rows.
///
/// The operand is 16 bits wide, which is also the hardware ceiling on cluster size.
pub fn multicast_cta_mask(ctas: usize) -> u16 {
    assert!(
        (1..=16).contains(&ctas),
        "multicast_cta_mask: {ctas} CTAs cannot be addressed by a 16-bit ctaMask"
    );
    (((1u32 << ctas) - 1) & 0xffff) as u16
}

/// One `wgmma` GEMM configuration. The generator, every gate and the launch plan all read this one
/// struct, so a new row is generated, budgeted and gated at once.
#[derive(Clone, Copy, Debug)]
pub struct WgmmaCfg {
    /// PTX entry symbol and bench label.
    pub name: &'static str,
    /// `Gpu::function` module-cache key. **Must be unique per generated variant** -- the cache keys
    /// on this string alone and never re-examines the PTX (`Gpu::function`'s landmine).
    pub key: &'static str,
    /// CTA tile rows. Must equal `64 * consumer_wgs` under [`Schedule::Cooperative`].
    pub bm: usize,
    /// CTA tile columns. Must equal the `wgmma` shape's N.
    pub bn: usize,
    /// Staged K width, in elements. 64 for 16-bit operands: `bk * 2 == 128` bytes is both the 128-B
    /// swizzle atom and TMA's maximum contiguous box extent under it, and it is the K-tile CUTLASS
    /// ships for every SM90 f16 mainloop (D1 section 3.1).
    pub bk: usize,
    /// Pipeline depth (shared-memory stages).
    pub stages: usize,
    /// Consumer warpgroups. Producer warpgroups are always 1.
    pub consumer_wgs: usize,
    /// `setmaxnreg.dec` target for the producer warpgroup.
    pub producer_regs: u32,
    /// `setmaxnreg.inc` target for each consumer warpgroup.
    pub consumer_regs: u32,
    pub dtype: WgmmaDtype,
    /// **The shared-memory layout both operands are staged in**, and therefore the descriptor
    /// fields, the swizzle mode of both tensor maps, and the per-K-step base advance. One field, one
    /// authority ([`desc_fields`]) -- so the reading the H100 crowns lands here and nowhere else.
    pub layout: SmemLayout,
    pub schedule: Schedule,
    pub multicast: Multicast,
}

impl WgmmaCfg {
    pub const fn threads(&self) -> usize {
        (1 + self.consumer_wgs) * WARPGROUP_THREADS
    }
    /// The descriptor's swizzle mode, derived from the layout -- never a second field that could
    /// disagree with it.
    pub const fn swizzle(&self) -> SmemSwizzle {
        SmemSwizzle::from_tma(self.layout.tma_swizzle())
    }
    /// The tile's row pitch in shared memory, bytes. A staged tile is `BK` elements wide.
    pub const fn row_bytes(&self) -> u64 {
        (self.bk * self.dtype.size()) as u64
    }
    /// Bytes of one A tile (one stage).
    pub const fn tile_a_bytes(&self) -> usize {
        self.bm * self.bk * self.dtype.size()
    }
    /// Bytes of one B tile (one stage).
    pub const fn tile_b_bytes(&self) -> usize {
        self.bn * self.bk * self.dtype.size()
    }
    /// CTAs per cluster (1 = no cluster). One accessor, so nothing re-derives it from `multicast`.
    pub const fn cluster_ctas(&self) -> usize {
        self.multicast.ctas()
    }
    /// **Rows of A one CTA's TMA copy fetches** -- the whole CTA tile without a cluster, and
    /// `BM / cluster_ctas` with one, because each CTA fetches a `1/cluster_ctas` slice of the shared
    /// A tile and multicasts it to the rest. This is the A tensor map's `box_rows`, so the
    /// descriptor and the copy cannot disagree about it.
    pub const fn a_box_rows(&self) -> usize {
        self.bm / self.cluster_ctas()
    }
    /// Bytes of the A slice ONE CTA fetches and multicasts (`tile_a_bytes` without a cluster).
    pub const fn a_slice_bytes(&self) -> usize {
        self.tile_a_bytes() / self.cluster_ctas()
    }
    /// Byte offset, inside every cluster CTA's identical A stage, at which cluster rank `r` lands
    /// its multicast slice. Slices are laid down in rank order, so the SMEM image of A is exactly
    /// the image one un-clustered copy would have written -- which is why the descriptor, the
    /// per-consumer `m64` slabs and the epilogue's row arithmetic are all unchanged by clustering.
    pub const fn a_slice_off(&self, rank: usize) -> usize {
        rank * self.a_slice_bytes()
    }
    /// **Bytes one TMA stage delivers INTO ONE CTA -- the `expect_tx` count.**
    ///
    /// # The multicast transaction rule (PTX ISA 9.7.9, mbarrier `complete-tx`)
    ///
    /// A transaction count is **per destination CTA**, never divided among them. A
    /// `.multicast::cluster` copy of `n` bytes signals `complete-tx n` on the mbarrier of *every*
    /// destination CTA -- the ISA multicasts the barrier signal to the same CTA-relative offset as
    /// the data -- so a `cluster_ctas`-way split multicast of A delivers
    /// `cluster_ctas * (tile_a / cluster_ctas) = tile_a` bytes into each CTA, and B delivers its own
    /// `tile_b`. The count is therefore `tile_a + tile_b` **whether or not there is a cluster**: the
    /// same number, reached by different arithmetic, which is exactly the kind of coincidence worth
    /// stating rather than leaving for a reader to re-derive.
    ///
    /// What DOES change with a cluster is how many copies reach that count (`cluster_ctas + 1`
    /// instead of 2) and who issues them. A count that disagrees with what the copies move does not
    /// fail -- the barrier never completes and every consumer warpgroup spins forever
    /// (`gpu::sync_within` exists for exactly that).
    pub const fn stage_tx_bytes(&self) -> usize {
        self.tile_a_bytes() + self.tile_b_bytes()
    }
    /// Copies that must complete before one stage's `full` barrier does, from this CTA's point of
    /// view: one multicast A slice per cluster CTA, plus this CTA's own B tile.
    pub const fn stage_copies_per_cta(&self) -> usize {
        self.cluster_ctas() + 1
    }
    /// **Arrivals one stage's `empty` barrier is initialised with.**
    ///
    /// Without a cluster: one per consumer warpgroup of this CTA. With one: one per consumer
    /// warpgroup **of the whole cluster**, because a CTA's producer overwrites an A slice that every
    /// CTA in the cluster reads, so it may not recycle the stage until all of them are done. Every
    /// consumer therefore arrives at every cluster CTA's `empty[s]` through `mapa`.
    pub const fn empty_arrivals(&self) -> usize {
        self.cluster_ctas() * self.consumer_wgs
    }
    /// Byte offset of the A tile for stage `s`.
    pub const fn a_off(&self, s: usize) -> usize {
        s * self.tile_a_bytes()
    }
    /// Byte offset of the B tile for stage `s`.
    pub const fn b_off(&self, s: usize) -> usize {
        self.stages * self.tile_a_bytes() + s * self.tile_b_bytes()
    }
    /// Byte offset of the "tile is full" mbarrier for stage `s`.
    pub const fn full_off(&self, s: usize) -> usize {
        self.stages * (self.tile_a_bytes() + self.tile_b_bytes()) + s * 8
    }
    /// Byte offset of the "tile is free" mbarrier for stage `s`.
    pub const fn empty_off(&self, s: usize) -> usize {
        self.full_off(self.stages) + s * 8
    }
    /// Total dynamic shared memory the entry needs.
    pub const fn smem_bytes(&self) -> usize {
        self.empty_off(self.stages)
    }
    /// Registers the warp-specialised split consumes once `setmaxnreg` has run.
    pub const fn regs_after_split(&self) -> u32 {
        WARPGROUP_THREADS as u32 * self.producer_regs
            + (self.consumer_wgs * WARPGROUP_THREADS) as u32 * self.consumer_regs
    }
    /// The `wgmma` shape one consumer warpgroup issues.
    pub fn shape(&self) -> Result<WgmmaShape, String> {
        WgmmaShape::new(self.bn)
    }
    /// `wgmma` instructions one consumer issues per staged K tile.
    pub const fn wgmma_per_stage(&self) -> usize {
        self.bk / WgmmaShape::K
    }

    /// The TMA descriptor geometry for the A operand of an `M x K` row-major matrix.
    ///
    /// The box is [`WgmmaCfg::a_box_rows`] tall, not `BM`: under a cluster each CTA fetches only its
    /// own `1 / cluster_ctas` slice of the shared tile and multicasts it, so the descriptor
    /// describes the SLICE. Without a cluster the two are the same number.
    pub fn tensor_map_a(&self, m: usize, k: usize) -> TensorMapArgs {
        TensorMapArgs::tiled_2d_row_major(
            self.dtype.tma(),
            m as u64,
            k as u64,
            k as u64,
            self.a_box_rows() as u32,
            self.bk as u32,
            self.layout.tma_swizzle(),
        )
    }
    /// The TMA descriptor geometry for the B operand of an `N x K` row-major matrix (NT GEMM).
    pub fn tensor_map_b(&self, n: usize, k: usize) -> TensorMapArgs {
        TensorMapArgs::tiled_2d_row_major(
            self.dtype.tma(),
            n as u64,
            k as u64,
            k as u64,
            self.bn as u32,
            self.bk as u32,
            self.layout.tma_swizzle(),
        )
    }

    /// Everything a launcher needs and nothing it can re-derive differently.
    pub fn launch_plan(&self) -> LaunchPlan {
        LaunchPlan {
            entry: self.name,
            module_key: self.key,
            bm: self.bm,
            bn: self.bn,
            bk: self.bk,
            block: (self.threads() as u32, 1, 1),
            dyn_smem_bytes: self.smem_bytes(),
            params: PARAM_ORDER,
            // The cluster is along X, which is the N axis of the grid, because the CTAs that share
            // an A tile are the ones that share an M tile and differ in N.
            cluster: (self.cluster_ctas() as u32, 1, 1),
        }
    }

    /// Every structural precondition, in one place. `Ok` means [`wgmma_module`] will emit.
    fn validate(&self) -> Result<WgmmaShape, String> {
        let shape = self.shape()?;
        match self.schedule {
            Schedule::Cooperative => {
                if !self.bm.is_multiple_of(128) {
                    return Err(format!(
                        "{UNSUPPORTED}: {} is cooperative but CTA-M {} is not a multiple of 128 \
                         (cooperative is illegal below CTA-M 128; a 64-row tile needs the pingpong \
                         or plain warp-specialised schedule)",
                        self.name, self.bm
                    ));
                }
            }
        }
        if self.bm != WgmmaShape::M * self.consumer_wgs {
            return Err(format!(
                "{UNSUPPORTED}: {}: CTA-M {} != 64 x {} consumer warpgroups; each consumer owns \
                 exactly one m64 slice",
                self.name, self.bm, self.consumer_wgs
            ));
        }
        if self.consumer_wgs == 0 {
            return Err(format!("{UNSUPPORTED}: {}: no consumers", self.name));
        }
        if self.threads() > 1024 {
            return Err(format!(
                "{UNSUPPORTED}: {}: {} threads exceeds the 1024/CTA limit",
                self.name,
                self.threads()
            ));
        }
        if !self.bk.is_multiple_of(WgmmaShape::K) || !self.bk.is_power_of_two() {
            return Err(format!(
                "{UNSUPPORTED}: {}: BK {} must be a power of two and a multiple of the wgmma K \
                 step ({})",
                self.name,
                self.bk,
                WgmmaShape::K
            ));
        }
        if self.stages < 2 {
            return Err(format!(
                "{UNSUPPORTED}: {}: {} stages -- a producer/consumer pipeline needs at least 2",
                self.name, self.stages
            ));
        }
        // Every stage base must satisfy the descriptor's alignment, since the descriptor's start
        // address is a stage base plus a multiple of the row size.
        let align = self.swizzle().required_alignment() as usize;
        for (what, bytes) in [
            ("A tile", self.tile_a_bytes()),
            ("B tile", self.tile_b_bytes()),
        ] {
            if !bytes.is_multiple_of(align) {
                return Err(format!(
                    "{UNSUPPORTED}: {}: one {what} is {bytes} B, not a multiple of the {align} B \
                     alignment {:?} requires, so stage bases would drift out of alignment",
                    self.name,
                    self.swizzle()
                ));
            }
        }
        if !self.b_off(0).is_multiple_of(align) {
            return Err(format!(
                "{UNSUPPORTED}: {}: the B region starts at {} B, not a multiple of {align}",
                self.name,
                self.b_off(0)
            ));
        }
        if self.smem_bytes() > HOPPER_SMEM_PER_CTA {
            return Err(format!(
                "{UNSUPPORTED}: {}: {} B of shared memory ({} stages x {} B + {} B of barriers) \
                 exceeds the {HOPPER_SMEM_PER_CTA} B per-CTA ceiling",
                self.name,
                self.smem_bytes(),
                self.stages,
                self.stage_tx_bytes(),
                2 * self.stages * 8
            ));
        }
        for (who, r) in [
            ("producer", self.producer_regs),
            ("consumer", self.consumer_regs),
        ] {
            if !(24..=256).contains(&r) || !r.is_multiple_of(8) {
                return Err(format!(
                    "{UNSUPPORTED}: {}: setmaxnreg {who} target {r} must be a multiple of 8 in \
                     24..=256",
                    self.name
                ));
            }
        }
        if self.regs_after_split() > REGS_PER_CTA {
            return Err(format!(
                "{UNSUPPORTED}: {}: the register split needs {} of {REGS_PER_CTA} registers \
                 (1 producer warpgroup at {} + {} consumer warpgroups at {})",
                self.name,
                self.regs_after_split(),
                self.producer_regs,
                self.consumer_wgs,
                self.consumer_regs
            ));
        }
        if shape.accum_regs() as u32 + 32 > self.consumer_regs {
            return Err(format!(
                "{UNSUPPORTED}: {}: a consumer holds {} accumulator registers, which leaves under \
                 32 for addressing at a {} register budget",
                self.name,
                shape.accum_regs(),
                self.consumer_regs
            ));
        }
        validate_cluster(
            self.name,
            self.cluster_ctas(),
            self.bm,
            self.bk * self.dtype.size(),
            align,
        )?;
        if !self.layout.tma_writable() {
            return Err(format!(
                "{UNSUPPORTED}: {}: {} needs every core matrix packed into 128 contiguous bytes, \
                 which one tiled TMA copy of a row-major operand does not write (it would take \
                 BK/8 = {} copies per operand per stage, or a shared-memory repack). The sweep \
                 probe stages it by hand; this generator declines rather than emitting a mainloop \
                 whose operands are in the wrong order.",
                self.name,
                self.layout.label(),
                self.bk / 8
            ));
        }
        if self.swizzle() == SmemSwizzle::B128 && self.row_bytes() != 128 {
            return Err(format!(
                "{UNSUPPORTED}: {}: the 128-B swizzle atom is 128 bytes and this tile's shared row \
                 is {} B (BK={} x {} B). The swizzled canonical layout only coincides with what TMA \
                 writes when the two are equal.",
                self.name,
                self.row_bytes(),
                self.bk,
                self.dtype.size()
            ));
        }
        // The TMA descriptors this config implies must themselves be legal, at a representative
        // shape. Catches a BK whose contiguous extent overflows the swizzle atom before any PTX is
        // built.
        self.tensor_map_a(4096, 4096).validate()?;
        self.tensor_map_b(4096, 4096).validate()?;
        Ok(shape)
    }
}

/// **The cluster's own preconditions, as a pure function of the numbers.**
///
/// A free function rather than a block inside [`WgmmaCfg::validate`] for one reason: today's
/// [`Multicast`] menu can only produce `ctas` of 1 or 2, so several of these arms are unreachable
/// from any shipped row -- and a validator whose negative arms are untested is a validator that
/// passes everything. Stated over plain numbers, every arm is reachable from a device-free test
/// (`the_cluster_preconditions_reject_every_way_a_slice_can_be_wrong`), and each stays a guard
/// against the day the menu grows.
///
/// `row_bytes` is the shared row pitch of one A slice (`BK * elem`) and `align` the alignment the
/// descriptor's swizzle mode requires. Each rule's failure is silent, not loud:
///
/// * a cluster past the portable ceiling is a launch-time rejection that names no kernel;
/// * a `BM` the cluster does not divide leaves A rows nobody fetched -- stale shared memory, read as
///   operands, on a fraction of the accumulator rows;
/// * a slice boundary off the 8-row core matrix puts the 128-B XOR swizzle out of phase between the
///   halves of one stage, and the descriptor undoes exactly one phase;
/// * a slice past TMA's box limit is a descriptor `cuTensorMapEncodeTiled` refuses, an hour later.
fn validate_cluster(
    name: &str,
    ctas: usize,
    bm: usize,
    row_bytes: usize,
    align: usize,
) -> Result<(), String> {
    if ctas <= 1 {
        return Ok(());
    }
    if ctas > MAX_PORTABLE_CLUSTER_CTAS {
        return Err(format!(
            "{UNSUPPORTED}: {name}: a {ctas}-CTA cluster is past the portable ceiling of \
             {MAX_PORTABLE_CLUSTER_CTAS}; anything wider needs \
             CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED and a device query, which this \
             family does not opt into blind"
        ));
    }
    if !bm.is_multiple_of(ctas) {
        return Err(format!(
            "{UNSUPPORTED}: {name}: CTA-M {bm} does not divide by the {ctas}-CTA cluster, so the \
             multicast A slices would not tile the shared A stage"
        ));
    }
    let box_rows = bm / ctas;
    if !box_rows.is_multiple_of(8) {
        return Err(format!(
            "{UNSUPPORTED}: {name}: the multicast A slice is {box_rows} rows, not a multiple of the \
             8-row core matrix / swizzle period, so the slices of one stage would carry different \
             swizzle phases"
        ));
    }
    if box_rows > crate::tma_host::TMA_MAX_BOX_DIM as usize {
        return Err(format!(
            "{UNSUPPORTED}: {name}: the A slice is {box_rows} rows, past TMA's {} element box limit",
            crate::tma_host::TMA_MAX_BOX_DIM
        ));
    }
    let slice_bytes = box_rows * row_bytes;
    if !slice_bytes.is_multiple_of(align) {
        return Err(format!(
            "{UNSUPPORTED}: {name}: one multicast A slice is {slice_bytes} B, not a multiple of the \
             {align} B alignment the descriptor's swizzle requires"
        ));
    }
    Ok(())
}

/// A kernel parameter's kind, in declaration order -- the contract between the generated `.entry`
/// and the future launcher's argument list. Pushing these in the wrong order or count makes the
/// driver read adjacent host stack as a pointer (crate rule #2), so the order is data, not prose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    /// `.param .u32` -- M, N, K.
    U32,
    /// `.param .u64` -- a device pointer.
    GlobalPtr,
    /// `.param .align 64 .b8 [128]` -- a by-value `CUtensorMap`.
    TensorMap,
}

/// The parameter order every entry this module generates declares:
/// `(M: u32, N: u32, K: u32, C: ptr, tensorMap A, tensorMap B)`.
pub const PARAM_ORDER: &[ParamKind] = &[
    ParamKind::U32,
    ParamKind::U32,
    ParamKind::U32,
    ParamKind::GlobalPtr,
    ParamKind::TensorMap,
    ParamKind::TensorMap,
];

/// Everything a launch of one generated entry needs, as pure data.
///
/// This is the seam between this module and the `gpu.rs` wrapper that will eventually call it: the
/// wrapper reads a `LaunchPlan`, it does not re-derive a grid or a shared-memory size. Two
/// derivations of one geometry is how a truncated grid returns pre-zeroed rows and calls them
/// results (`launch_cfg`'s own history in `autotune`).
///
/// # Preconditions a launcher must assert (crate rule #2)
///
/// * `a.len() == m * k`, `b.len() == n * k`, `c.len() == m * n` -- the address arithmetic assumes it
///   and the tensor maps are built from it.
/// * **`m * n` and the largest `row * N + col` must fit `u32`.** The epilogue computes its element
///   index in 32 bits (`mad.lo.s32`) before widening to a byte offset, exactly as every other GEMM
///   epilogue in this crate does. 8192 x 8192 is 67 M elements and fine; a 65536-square output is
///   not.
/// * `k >= 1`. At `k == 0` no `wgmma` issues, so the accumulators are never written, and the kernel
///   deliberately skips the epilogue rather than store uninitialised registers -- `C` is left
///   untouched, not zeroed.
/// * The device's `Gpu::smem_budget()` is at least [`LaunchPlan::dyn_smem_bytes`]. On anything but a
///   datacenter Hopper part it is not, and [`require_sm90a`] has already declined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchPlan {
    pub entry: &'static str,
    pub module_key: &'static str,
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub block: (u32, u32, u32),
    /// Pass this to `Gpu::function_dyn` **and** to `dyn_launch_cfg`. The kernel declares its shared
    /// memory as one `.extern` window, so a launch that passes 0 gets a window of 0 bytes and every
    /// address in the mainloop is out of range.
    pub dyn_smem_bytes: usize,
    pub params: &'static [ParamKind],
    /// **The cluster shape, in CTAs.** `(1, 1, 1)` means no cluster and no launch attribute; a
    /// wider X means the entry carries `.reqnctapercluster` and MUST be launched through
    /// `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` set to exactly this, because
    /// a compiled cluster requirement the launch does not match is a launch failure.
    pub cluster: (u32, u32, u32),
}

impl LaunchPlan {
    /// CTAs per cluster -- the product of the cluster shape.
    pub fn cluster_ctas(&self) -> u32 {
        self.cluster.0 * self.cluster.1 * self.cluster.2
    }

    /// The CTA grid for an `M x N` output: x indexes N tiles, y indexes M tiles.
    ///
    /// Ragged edges need no special case. TMA fills out-of-range elements with zero
    /// ([`crate::tma_host::TmaOobFill::Zero`]), so a partial tile accumulates nothing, and the
    /// epilogue predicates every store on `row < M && col < N`.
    ///
    /// # The cluster rounds the grid up, and that is the same discipline
    ///
    /// **There is no such thing as a partial cluster.** Every grid dimension must be a multiple of
    /// its cluster dimension, so an odd number of N tiles under a `2x1x1` cluster gets one extra
    /// CTA. That CTA is not a special case either: its `ctan = ctaid.x * BN` is `>= N`, so every
    /// element of its B tile is out of range and TMA zero-fills it, every accumulator it computes is
    /// zero, and every store it attempts fails the epilogue's `col < N` predicate. It participates
    /// in the cluster's barriers -- which it must, since the cluster is fixed -- and writes nothing.
    /// Rounding and predicating is exactly what the ragged **edge** already does; the cluster just
    /// makes the rounding coarser.
    ///
    /// **The pad CTA still issues its A multicast, and that is load-bearing, not waste.** Its A
    /// slice is the same M rows as its peer's -- genuinely in range -- and the peer's half of the
    /// stage comes from it. Skipping the copies of an out-of-range CTA would leave the *real* CTA of
    /// that cluster with half an A tile: correct numbers on the rows it fetched itself and stale
    /// shared memory on the rest. Nothing in the generated producer is predicated on `ctan < N` for
    /// exactly this reason.
    pub fn grid(&self, m: usize, n: usize) -> (u32, u32, u32) {
        let cx = self.cluster.0.max(1) as usize;
        let x = n.div_ceil(self.bn).div_ceil(cx) * cx;
        let cy = self.cluster.1.max(1) as usize;
        let y = m.div_ceil(self.bm).div_ceil(cy) * cy;
        (x as u32, y as u32, 1)
    }
}

// --- the shipped configurations -------------------------------------------------------------------

/// **W1 -- the centerpiece** (D1 section 4.5).
///
/// CTA 128x256x64, `wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16` in the SS form, 4 stages
/// (49 152 B per stage, 192 KiB of mainloop), warp-specialised cooperative: 1 producer warpgroup at
/// 32 registers and 2 consumer warpgroups at 232, 384 threads, one CTA per SM.
///
/// This tile is simultaneously **CUTLASS's default emitted SM90 f16 kernel** (D1 section 3.1) and the
/// configuration of an **independently reproduced 107%-of-cuBLAS** kernel (D1 section 3.3). Two
/// independent sources landing on the same point is the strongest prior available, and D1 predicts
/// 95-108% of cuBLAS at 4096-8192 cubed for the full form.
///
/// **This row is the full form minus cluster multicast** ([`Multicast::None`]), which was the right
/// bring-up order: multicast is a traffic optimisation whose failure mode is a deadlock, and the
/// kernel is correct without it. It is now also the **control arm** of the cluster A/B --
/// [`WGMMA_W1_MC`] is the same tile, the same stage depth and the same descriptor reading with the
/// cluster switched on, so the two differ in one fact.
///
/// # The `layout` field is the 2026-08-10 correction, and is AWAITING CONFIRMATION
///
/// This row shipped `RowMajorNone { k_leading: true }` until the first H100 round scored it 64 of
/// 4096 output lanes exact -- the signature of a descriptor whose within-core-matrix row stride is
/// the fixed 16 bytes while the tile's rows are 128 apart. [`SmemLayout::Swizzle128`] is the reading
/// this file now derives (module docs, fifth fact): TMA writes the 128-B-swizzled tile and the
/// descriptor's own `Swizzle<3,4,3>` undoes it, with no repack. **Nothing has executed it.** The gate
/// that will is `wgmma_hopper_bringup` stage D, whose sweep carries this reading, the canonical
/// no-swizzle alternative and both of the readings the log already scored.
pub const WGMMA_W1: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x256x64_s4",
    key: "wgmma_nt_f16_128x256x64_s4",
    bm: 128,
    bn: 256,
    bk: 64,
    stages: 4,
    consumer_wgs: 2,
    producer_regs: 32,
    consumer_regs: 232,
    dtype: WgmmaDtype::F16,
    layout: SHIPPED_LAYOUT,
    schedule: Schedule::Cooperative,
    multicast: Multicast::None,
};

/// **The descriptor reading every shipped row carries**, in one place so the sweep's "ACTION" line is
/// a single-token edit.
///
/// `lbo_bytes: 16` spells the k-group stride the ISA's canonical 128-B-swizzle K-major layout fixes
/// at one core-matrix row; the field does not appear in that layout's address formula at all, so the
/// sweep carries two other spellings and will report whether the hardware reads it.
pub const SHIPPED_LAYOUT: SmemLayout = SmemLayout::Swizzle128 {
    lbo_bytes: 16,
    swapped: false,
};

/// The bf16 twin of [`WGMMA_W1`]. `wgmma` is precision-generic across the 16-bit types -- only the
/// operand-type token changes -- and bf16 is the dominant *training* precision, so the training
/// GEMMs get the same tile without a second mainloop.
pub const WGMMA_W1_BF16: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_bf16_128x256x64_s4",
    key: "wgmma_nt_bf16_128x256x64_s4",
    dtype: WgmmaDtype::Bf16,
    ..WGMMA_W1
};

/// **W3c -- the moderate-size arm** (D1 section 4.5's W3, cooperative rather than pingpong).
///
/// CTA 128x128x64, `m64n128k16`, 6 stages (32 768 B per stage, 192 KiB). D1 wants this shape for
/// roughly 2048-cubed and for any GEMM whose `M x N` is under 4.3e6, where a 256-wide tile
/// quantizes: a `BM x BN` tile only fills H100's 132 SMs once when `M * N >= 132 * BM * BN`, which
/// for 128x256 is 4.33 M output elements and for 128x128 is 2.16 M (D1 section 2.4).
///
/// D1 specifies W3 with the **pingpong** schedule (two consumer warpgroups alternating mainloop and
/// epilogue). This row is cooperative, which is legal here -- CTA-M is 128 -- but is not that
/// schedule. The tile and stage depth are W3's; the scheduler is a follow-up.
pub const WGMMA_W3C: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x128x64_s6",
    key: "wgmma_nt_f16_128x128x64_s6",
    bn: 128,
    stages: 6,
    consumer_regs: 168,
    ..WGMMA_W1
};

/// **W1 as D1 section 4.5 actually specifies it: the `2x1x1` cluster with `.multicast::cluster` on
/// A.**
///
/// The 2026-08-10 Act-2 round measured [`WGMMA_W1`] at 58.8-67.5% of cuBLAS at 4096-8192 cubed
/// against a D1 prediction of 95-108%, on a clean instrument (floors +/-0.03-2.19%, 0.00% clock
/// drift, a healthy peer). That round's own provenance line names the first suspect: it measured W1
/// **minus its cluster**, so the prediction had never been tested. This row is the missing arm.
///
/// # What the cluster buys, and why it is the dossier's named lever
///
/// On an H100 the binding constraint for this tile is the L2 -> SMEM fill path. Two CTAs holding
/// adjacent N tiles of the same M tile read the *same* A rows, so without a cluster A crosses L2
/// twice. In a `2x1x1` cluster each CTA TMA-loads half the A tile and multicasts it to both, which
/// halves A's L2 traffic and leaves B untouched (each CTA has its own N half). Nothing else about
/// the kernel changes: same tile, same 4 stages, same `SHIPPED_LAYOUT`, same register split, same
/// epilogue. That is deliberate -- it is an A/B, and an A/B whose arms differ in two things measures
/// neither.
///
/// # Its own failure mode, and how the gates cover it
///
/// A wrong `ctaMask`, a wrong slice offset or a missed remote arrival is not a crash: it is correct
/// numbers on the rows this CTA fetched itself and stale ones on the rows its peer was supposed to
/// multicast in -- silently, on half the accumulator rows. So the correctness gate
/// (`gpu::tests::wgmma_cluster_multicast_is_exact`) runs shapes wide enough that the cluster spans
/// two CTAs with *different* B halves and both A halves come from a multicast, and demands `==`
/// against the f64 reference before anything is timed.
pub const WGMMA_W1_MC: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x256x64_s4_mc2",
    key: "wgmma_nt_f16_128x256x64_s4_mc2",
    multicast: Multicast::ClusterA,
    ..WGMMA_W1
};

/// Every shipped configuration. The generator, the gates and the device-free module enumeration all
/// iterate this table.
pub const WGMMA_VARIANTS: &[WgmmaCfg] = &[WGMMA_W1, WGMMA_W1_BF16, WGMMA_W3C, WGMMA_W1_MC];

/// Look up a variant by entry name; panics loudly rather than mis-dispatching.
pub fn wgmma_variant(name: &str) -> &'static WgmmaCfg {
    WGMMA_VARIANTS
        .iter()
        .find(|v| v.name == name)
        .unwrap_or_else(|| panic!("unknown wgmma variant {name:?}"))
}

// --- Act 2 round 2: the CONFIG SWEEP table --------------------------------------------------------

/// One row of [`WGMMA_SWEEP_GRID`]: a configuration, a stable label, and the question it answers.
///
/// A row that **declines** is still a row. `WgmmaCfg::validate` is the authority on what fits, and a
/// depth that does not fit Hopper's carveout is a finding the round should print (with the budget
/// arithmetic) rather than a row quietly missing from the table.
#[derive(Clone, Copy, Debug)]
pub struct SweepRow {
    /// Round-log row name and `bench_instrument` sample-label component. ASCII, no whitespace.
    pub label: &'static str,
    pub cfg: &'static WgmmaCfg,
    /// What this row is in the table to answer, in one line.
    pub why: &'static str,
}

impl SweepRow {
    /// `Ok(())` iff this row can be generated at all -- i.e. it will be measured rather than
    /// declined. The message is `WgmmaCfg::validate`'s own, so the reason a row is absent from the
    /// ranked table is the generator's reason and not a second opinion.
    pub fn generatable(&self) -> Result<(), String> {
        self.cfg.validate().map(|_| ())
    }
    /// The shared-memory budget line the round prints for every row, declined or not:
    /// `stages x per-stage + barriers = total of the Hopper carveout`.
    pub fn smem_line(&self) -> String {
        let c = self.cfg;
        format!(
            "{} stages x {} B + {} B barriers = {} B of {} B ({:.1}%)",
            c.stages,
            c.stage_tx_bytes(),
            2 * c.stages * 8,
            c.smem_bytes(),
            HOPPER_SMEM_PER_CTA,
            100.0 * c.smem_bytes() as f64 / HOPPER_SMEM_PER_CTA as f64
        )
    }
}

/// 128x256x64 at **2** stages, no cluster -- the shallow end of the depth axis.
pub const WGMMA_W1_S2: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x256x64_s2",
    key: "wgmma_nt_f16_128x256x64_s2",
    stages: 2,
    ..WGMMA_W1
};
/// 128x256x64 at 2 stages **with** the cluster.
pub const WGMMA_W1_MC_S2: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x256x64_s2_mc2",
    key: "wgmma_nt_f16_128x256x64_s2_mc2",
    stages: 2,
    ..WGMMA_W1_MC
};
/// 128x256x64 at 3 stages, no cluster.
pub const WGMMA_W1_S3: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x256x64_s3",
    key: "wgmma_nt_f16_128x256x64_s3",
    stages: 3,
    ..WGMMA_W1
};
/// 128x256x64 at 3 stages **with** the cluster.
pub const WGMMA_W1_MC_S3: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x256x64_s3_mc2",
    key: "wgmma_nt_f16_128x256x64_s3_mc2",
    stages: 3,
    ..WGMMA_W1_MC
};
/// 128x256x64 at 5 stages. **Does not fit** -- kept so the round prints why the depth axis stops.
pub const WGMMA_W1_MC_S5: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x256x64_s5_mc2",
    key: "wgmma_nt_f16_128x256x64_s5_mc2",
    stages: 5,
    ..WGMMA_W1_MC
};
/// 128x256x64 at 6 stages. **Does not fit** either, by a wider margin.
pub const WGMMA_W1_MC_S6: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x256x64_s6_mc2",
    key: "wgmma_nt_f16_128x256x64_s6_mc2",
    stages: 6,
    ..WGMMA_W1_MC
};
/// [`WGMMA_W3C`] with the cluster: does the lever transfer to the narrower tile?
pub const WGMMA_W3C_MC: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x128x64_s6_mc2",
    key: "wgmma_nt_f16_128x128x64_s6_mc2",
    multicast: Multicast::ClusterA,
    ..WGMMA_W3C
};

/// **The Act-2 round-2 configuration sweep, as data.**
///
/// Round 1 was a single A/B: one kernel against cuBLAS on seven shapes. It produced a number
/// (58.8-67.5% of cuBLAS where D1 section 4.5 predicted 95-108%) and one named suspect -- the
/// cluster its own provenance line said the row declined. A second single A/B would answer one
/// question per rented visit. This table answers four in one, with cuBLAS as the same-run yardstick
/// for every row, so a row's verdict is against the *same* denominator every other row is scored on:
///
/// * **the cluster**, at a fixed tile and depth (`w1_s4_off` vs `w1_s4_mc2`) -- the headline;
/// * **the pipeline depth**, at BOTH cluster settings, so "deeper is better" and "the cluster is
///   better" cannot be confounded with one another;
/// * **the tile**, via D1's moderate-size W3c arm, which round 1 deliberately excluded because a
///   second kernel in the contender arm would have been an A/B against two things at once -- here it
///   is its own row against the same peer, which is a different and legitimate experiment;
/// * **the ceiling on depth**, by keeping the two rows that do not fit and printing the arithmetic.
///
/// **The `wgmma` wait depth is deliberately NOT an axis.** The generated mainloop issues its `BK/16`
/// `wgmma` as one group and then `wgmma.wait_group.sync.aligned 0` immediately before releasing the
/// stage's `empty` barrier. That 0 is not a tuning knob: the release publishes the buffer to the
/// producer, so it may not precede the last read of it, and any depth above 0 there would be a
/// correctness bug rather than a slower or faster kernel. The real lever the round-1 PTX dump
/// exposes is a *restructuring* -- commit per K step and defer the stage release by one, so the
/// tensor cores are not drained to empty between stages -- which is a follow-up, not a sweep row.
/// It is named in the round's own preamble so the log carries the finding.
pub const WGMMA_SWEEP_GRID: &[SweepRow] = &[
    SweepRow {
        label: "w1_s4_off",
        cfg: &WGMMA_W1,
        why: "THE CONTROL. Round 1's measured row, unchanged: 128x256x64 s4, cluster 1x1x1. Its \
              headline here must land near round 1's 67.5% at sq4096 / 58.8% at sq8192, or the \
              instrument moved and no other row is readable",
    },
    SweepRow {
        label: "w1_s4_mc2",
        cfg: &WGMMA_W1_MC,
        why: "THE HEADLINE. The same tile, depth, layout and register split with a 2x1x1 cluster \
              and .multicast::cluster on A -- D1 4.5's W1 as specified. One fact different from the \
              control, so the difference is the cluster",
    },
    SweepRow {
        label: "w1_s3_off",
        cfg: &WGMMA_W1_S3,
        why: "Depth 3 without the cluster: the control arm of the depth axis",
    },
    SweepRow {
        label: "w1_s3_mc2",
        cfg: &WGMMA_W1_MC_S3,
        why: "Depth 3 with the cluster. Halving A's L2 traffic shortens the fill the ring is hiding, \
              so the depth that was right at full traffic need not be right at half",
    },
    SweepRow {
        label: "w1_s2_off",
        cfg: &WGMMA_W1_S2,
        why: "Depth 2 without the cluster -- the shallow end, and the row that says how much of the \
              deficit is fill latency at all",
    },
    SweepRow {
        label: "w1_s2_mc2",
        cfg: &WGMMA_W1_MC_S2,
        why: "Depth 2 with the cluster. Pairing every measurable depth at both settings is what \
              keeps 'deeper is better' and 'clustered is better' from being one measurement",
    },
    SweepRow {
        label: "w1_s5_mc2",
        cfg: &WGMMA_W1_MC_S5,
        why: "Depth 5 at 128x256. EXPECTED TO DECLINE on shared memory -- and the decline is the \
              point: the cluster does not change the ring's size, so the depth axis stops at 4 for \
              this tile at BOTH cluster settings",
    },
    SweepRow {
        label: "w1_s6_mc2",
        cfg: &WGMMA_W1_MC_S6,
        why: "Depth 6 at 128x256. Expected to decline by a wider margin, which pins the ceiling \
              rather than leaving it at one data point",
    },
    SweepRow {
        label: "w3c_s6_off",
        cfg: &WGMMA_W3C,
        why: "D1's moderate-size arm (128x128x64 s6), which round 1 excluded because a second \
              kernel in ITS contender arm would have been an A/B against two things at once. Here \
              every row is scored against the same cuBLAS, so it is a row",
    },
    SweepRow {
        label: "w3c_s6_mc2",
        cfg: &WGMMA_W3C_MC,
        why: "The narrow tile WITH the cluster: does the A-multicast lever transfer to a tile whose \
              per-CTA A traffic is already amortised over half as many output columns?",
    },
];

/// The sweep rows that generate, in table order -- what the round will actually launch.
pub fn wgmma_sweep_measurable() -> Vec<&'static SweepRow> {
    WGMMA_SWEEP_GRID
        .iter()
        .filter(|r| r.generatable().is_ok())
        .collect()
}

/// **The shapes the config sweep measures, named out of [`WGMMA_BENCH_GRID`] rather than restated.**
///
/// Selecting by label rather than by literal is the point: these are the *same* [`GemmPoint`]
/// objects, with the same [`bench_iters`], that `wgmma_vs_cublas` swept on 2026-08-10, so a row's
/// number here and that round's number are comparable without an argument about denominators.
///
/// Three, not seven. The sweep's cost is `rows x shapes`, and these three carry the whole question:
/// `sq4096` and `sq8192` are the two points D1 section 4.5 predicts W1 at 95-108% on (and the two
/// round 1 measured at 67.5% and 58.8%), and `sq2048` is the tile-quantization point D1 puts W3c at.
/// The full seven-shape grid remains `wgmma_vs_cublas`'s job -- once this round says which
/// configuration to run, that bench runs it everywhere.
pub const WGMMA_SWEEP_SHAPES: &[&str] = &["sq2048", "sq4096", "sq8192"];

/// The headline shape of the ranked table: the first of D1 4.5's two prediction points.
pub const WGMMA_SWEEP_HEADLINE: &str = "sq4096";

/// [`WGMMA_SWEEP_SHAPES`], resolved. Panics loudly on a label no grid row carries -- a sweep that
/// silently measured six shapes because one name was misspelled would be worse than one that failed.
pub fn wgmma_sweep_points() -> Vec<&'static GemmPoint> {
    WGMMA_SWEEP_SHAPES
        .iter()
        .map(|l| {
            WGMMA_BENCH_GRID
                .iter()
                .find(|p| p.label == *l)
                .unwrap_or_else(|| {
                    panic!("WGMMA_SWEEP_SHAPES names {l:?}, which is not a grid row")
                })
        })
        .collect()
}

/// **The command that discharges validation item 9's correctness half.**
///
/// Unlike the two round invocations this one names a *gate*, not a bench: it is not `#[ignore]`d and
/// it runs inside a plain `cargo test -p wukong_codegen_gpu --features gpu` on a Hopper part. The
/// spelled-out form exists so the skip message on every other part sends the operator to the right
/// test rather than to the bring-up sequence, which does not exercise a cluster at all.
pub const WGMMA_CLUSTER_GATE_INVOCATION: &str =
    "WUKONG_GPU_REQUIRED=1 cargo test -p wukong_codegen_gpu --features gpu \
     -- --nocapture --test-threads=1 wgmma_cluster_multicast_is_exact";

/// **The exact command the Act-2 round-2 configuration sweep runs.** Data, for the same reason
/// [`WGMMA_BENCH_INVOCATION`] is: the sweep prints a table and asserts no ratio, so a run without
/// `--nocapture` throws away everything the rented minutes produced.
pub const WGMMA_SWEEP_INVOCATION: &str =
    "WUKONG_GPU_REQUIRED=1 WUKONG_PEER_REQUIRED=1 cargo test -p wukong_codegen_gpu --features gpu \
     --release -- --ignored --nocapture --test-threads=1 wgmma_config_sweep";

/// **Every distinct configuration this family can emit a module for**, shipped rows first, then the
/// sweep-only rows that generate -- deduplicated by module key.
///
/// This is the corpus the ASCII rule, the `.version` law and the **ptxas census** all scan, which is
/// the point: a sweep row whose PTX a rented H100 is the first assembler ever to see is exactly the
/// failure the CPU-priced census exists to prevent. `Gpu::function` keys on the module key alone and
/// never re-examines the text, so the dedup is by key and a duplicate key is a bug the launch-plan
/// gate catches.
pub fn wgmma_all_emittable() -> Vec<&'static WgmmaCfg> {
    let mut v: Vec<&'static WgmmaCfg> = WGMMA_VARIANTS.iter().collect();
    for r in WGMMA_SWEEP_GRID {
        if r.generatable().is_ok() && !v.iter().any(|c| c.key == r.cfg.key) {
            v.push(r.cfg);
        }
    }
    v
}

// --- bring-up: operands whose permutation is visible ----------------------------------------------

/// The largest partial sum the bring-up reference may reach and still be an **exact** f32 integer.
///
/// f32 represents every integer up to 2^24 exactly, so a dot product of exact-f16 integers whose
/// running total never passes this bound is computed identically by the tensor cores, by the f64
/// reference and by any reassociation of either. That turns the `DescOrder` verdict from "within
/// tolerance" into `==`, which is the difference between a settled question and an argument about
/// whether 3e-3 is noise.
pub const BRINGUP_EXACT_LIMIT: f64 = 16_777_216.0;

/// The radix of the bring-up positional code at K = `k`: the largest power of two `w` in `2..=8` for
/// which `k * w^6 <= 2^24`.
///
/// Each operand value is at most `w^3` (three base-`w` digits), so the dot product of `k` terms is
/// bounded by `k * w^6`. Wide K therefore gets a narrower code — the alternative is a reference that
/// is only *approximately* right, which is the one thing a bring-up round cannot afford.
pub fn ramp_radix(k: usize) -> usize {
    [8usize, 4, 2]
        .into_iter()
        .find(|w| (k as f64) * (w.pow(6) as f64) <= BRINGUP_EXACT_LIMIT)
        .unwrap_or(2)
}

/// One bring-up operand value: a **three-digit positional code** in base `w`, offset by one so no
/// value is zero (a zero lane cannot distinguish a wrong address from a right one).
///
/// `1 + d0 + w*d1 + w*w*d2`, so a permutation that moves *any* of the three digits changes the value.
const fn ramp_code(w: usize, d0: usize, d1: usize, d2: usize) -> f32 {
    (1 + d0 % w + w * (d1 % w) + w * w * (d2 % w)) as f32
}

/// **The bring-up operands: `A` (`m x k`) and `B` (`n x k`), with ramps a descriptor misreading
/// cannot hide.**
///
/// `WGMMA_DEVICE_VALIDATION` item 1 asks for "distinguishable ramps in A and B", and the word is
/// load-bearing. The failure being probed permutes which shared-memory *core matrix* an operand
/// element is fetched from, so an operand that is constant along either core-grid axis — a plain
/// `A[m][k] = m`, say — produces the identical product under both readings and the round settles
/// nothing while looking clean. Each value here therefore encodes three positions at three different
/// digit weights:
///
/// | digit | A | B |
/// |---|---|---|
/// | `d0` (weight 1) | position *inside* the core matrix | position inside the core matrix, mixed differently |
/// | `d1` (weight `w`) | core-grid **row** index `m / 8` | core-grid **k** index `k / 8` |
/// | `d2` (weight `w^2`) | core-grid **k** index `k / 8` | core-grid **row** index `n / 8` |
///
/// A and B put the two core-grid indices at *opposite* weights on purpose: a swap that accidentally
/// left A's product unchanged could not also leave B's unchanged.
///
/// Every value is a small integer, hence exactly representable in f16 (integers through 2048 are),
/// hence the tensor cores multiply exactly what the reference multiplies. [`ramp_radix`] keeps the
/// accumulation inside [`BRINGUP_EXACT_LIMIT`], so the comparison is `==` rather than a tolerance —
/// both properties are asserted device-free by `the_bringup_operands_are_exact_in_f16_and_f32`.
pub fn bringup_operands(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let w = ramp_radix(k);
    let mut a = Vec::with_capacity(m * k);
    for row in 0..m {
        for col in 0..k {
            a.push(ramp_code(w, (row % 8) + (col % 8), row / 8, col / 8));
        }
    }
    let mut b = Vec::with_capacity(n * k);
    for row in 0..n {
        for col in 0..k {
            b.push(ramp_code(w, 3 * (row % 8) + (col % 8), col / 8, row / 8));
        }
    }
    (a, b)
}

/// **The operand the hardware would see**, element by element, when an *unswizzled* descriptor with
/// fields `f` is pointed at a tile that is physically stored **plain row-major**.
///
/// The descriptor arithmetic, run backwards on the host. A `wgmma` at K step `j0` builds its
/// descriptor at `slab_base + j0 * f.k_step_bytes`, and the hardware fetches element `(r, c)` of core
/// matrix `(i, j)` from `+ i*SBO + j*LBO + r*16 + c*2` ([`SmemDesc::canonical_offset`]). Reading that
/// byte offset back through a row-major tile gives the element the kernel actually multiplies:
///
/// ```text
/// o = j0*k_step + i*SBO + j*LBO + r*16 + c*2       (what the hardware fetches)
/// truth: (8i+r)*row_bytes + (16*j0 + 8j + c)*2     (where the element really is)
/// ```
///
/// The `r*16` against the truth's `r*row_bytes` is the whole 2026-08-10 defect: it cancels only at
/// `r == 0`, so exactly one row in eight is read correctly and the product is exact only where BOTH
/// operands' rows are the eighth one -- 64 lanes of 4096, which is what the H100 measured.
///
/// `rows_per_desc` is how many rows one descriptor covers: 64 for A, where each consumer warpgroup
/// owns its own `m64` slab, and the full tile height for B. Reads past the operand return **zero**,
/// matching TMA's fill past the last real row.
///
/// **This is a model.** No device verdict consults it; it exists so the sweep's predictions and the
/// logged 64/4096 can be checked with no Hopper part in the room
/// (`the_row_major_reading_is_unrepresentable_and_predicts_the_log`).
pub fn unswizzled_read(
    x: &[f32],
    rows: usize,
    k: usize,
    rows_per_desc: usize,
    f: DescFields,
) -> Vec<f32> {
    assert_eq!(x.len(), rows * k, "operand is rows*k");
    assert!(
        rows_per_desc.is_multiple_of(8) && k.is_multiple_of(WgmmaShape::K),
        "the core-matrix grid needs 8 rows and the wgmma K step needs {} elements",
        WgmmaShape::K
    );
    assert_eq!(
        f.swizzle,
        SmemSwizzle::None,
        "this model reads an UNSWIZZLED descriptor; a swizzled one permutes 16-byte chunks too"
    );
    let row_bytes = 2 * k;
    let mut out = vec![0f32; rows * k];
    for dst_row in 0..rows {
        let base_row = (dst_row / rows_per_desc) * rows_per_desc;
        let (i, r) = ((dst_row - base_row) / 8, (dst_row - base_row) % 8);
        for dst_col in 0..k {
            let (j0, rem) = (dst_col / WgmmaShape::K, dst_col % WgmmaShape::K);
            let (j, c) = (rem / 8, rem % 8);
            let o = j0 * f.k_step_bytes as usize
                + i * f.sbo as usize
                + j * f.lbo as usize
                + r * 16
                + c * 2;
            let (src_row, src_col) = (base_row + o / row_bytes, (o % row_bytes) / 2);
            out[dst_row * k + dst_col] = if src_row < rows && src_col < k {
                x[src_row * k + src_col]
            } else {
                0.0
            };
        }
    }
    out
}

// --- bring-up: the single-stage TMA probe ---------------------------------------------------------

/// Threads the TMA probe launches: one warpgroup, so it exercises the same 128-thread shape the
/// mainloop's producer does, with nothing else in the kernel.
pub const TMA_PROBE_THREADS: u32 = 128;

/// The TMA probe's entry symbol.
///
/// Deliberately **not** spelled `wgmma_...`: the probe's whole value is that it contains no `wgmma`,
/// and two laws read that as a substring — `the_tma_stage_probe_is_structurally_a_single_stage_load`
/// asserts the instruction is absent, and the crate-wide `.version` law licenses a module's ISA floor
/// by finding an above-7.8 instruction in its text. A name carrying `wgmma` would satisfy both by
/// accident, which is the quietest way for a law to stop meaning anything. This module earns its
/// `.version 8.0` from `cp.async.bulk`, honestly.
pub const TMA_PROBE_ENTRY: &str = "wk_tma_stage_probe";

/// The TMA probe's `Gpu::function` module-cache key. **One key is correct here** precisely because
/// the kernel is geometry-generic: the transaction size, the tile coordinates and the readback
/// length are all run-time parameters, so every geometry runs the *same* compiled module rather than
/// needing a key each (`Gpu::function`'s landmine, from the other direction).
pub const TMA_PROBE_KEY: &str = "wk_tma_stage_probe";

/// Everything a launcher needs for [`tma_stage_probe_module`], as pure data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TmaProbePlan {
    pub entry: &'static str,
    pub module_key: &'static str,
    pub block: (u32, u32, u32),
    /// Dynamic shared memory for a tile of `tx_bytes`: the tile itself plus the 8-byte mbarrier that
    /// sits immediately after it.
    pub dyn_smem_bytes: usize,
    /// The `expect_tx` byte count, which must equal `TensorMapArgs::transaction_bytes()` of the map
    /// being probed. If the two disagree the barrier never completes — this is the cheapest kernel in
    /// the family on which to find that out.
    pub tx_bytes: usize,
}

/// The probe's launch plan for a tile of `tx_bytes`. Declines rather than rounds: a transaction that
/// is not a multiple of 16 could not have come from a legal TMA box in the first place.
pub fn tma_probe_plan(tx_bytes: usize) -> Result<TmaProbePlan, String> {
    if tx_bytes == 0 || !tx_bytes.is_multiple_of(16) {
        return Err(format!(
            "{UNSUPPORTED}: TMA probe transaction {tx_bytes} B must be a non-zero multiple of 16"
        ));
    }
    if tx_bytes + 8 > HOPPER_SMEM_PER_CTA {
        return Err(format!(
            "{UNSUPPORTED}: TMA probe tile {tx_bytes} B + 8 B barrier exceeds the \
             {HOPPER_SMEM_PER_CTA} B per-CTA ceiling"
        ));
    }
    Ok(TmaProbePlan {
        entry: TMA_PROBE_ENTRY,
        module_key: TMA_PROBE_KEY,
        block: (TMA_PROBE_THREADS, 1, 1),
        dyn_smem_bytes: tx_bytes + 8,
        tx_bytes,
    })
}

/// **`WGMMA_DEVICE_VALIDATION` item 4, as a kernel: one TMA tile in, the same bytes out.**
///
/// The mainloop asks four things of the hardware at once — a tensor map the driver accepted, a
/// transaction count that matches what two copies move, a barrier that completes, and a `wgmma` that
/// reads the tile back through a descriptor. When the answer is wrong, all four are suspects. This
/// kernel removes three of them: it issues **one** `cp.async.bulk.tensor.2d` against **one** mbarrier,
/// waits, and copies the raw 16-bit shared bytes straight to global. Compared against a host copy of
/// the same tile it isolates the descriptor and the transaction count completely, and it does so on a
/// kernel with no `wgmma`, no warp specialisation, no `setmaxnreg` and no pipeline in it at all.
///
/// It is also the right kernel to meet a **hang** on. The `expect_tx` count is the one parameter
/// whose mismatch does not fail but waits forever, and finding that out here costs one launch of the
/// simplest module in the family rather than a wedged context.
///
/// Geometry-generic on purpose: the transaction size and the tile coordinates are parameters, so one
/// module and one cache key cover the A tile, the B tile and any ragged corner of either.
///
/// Parameter order is `(txBytes: u32, coord0: u32, coord1: u32, out: ptr, tensorMap)`. **`coord0` is
/// the CONTIGUOUS axis** (`k`, for the row-major operands this family loads) and `coord1` is the row
/// — dimension 0 is the fastest-varying axis of the descriptor, per `tma_host`'s one memorable fact.
pub fn tma_stage_probe_module(license: &Sm90aLicense) -> Result<String, String> {
    let name = TMA_PROBE_ENTRY;
    let mut s = String::from(sm90a_header(license));
    s += WGMMA_DSMEM_DECL;
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pTx,\n    .param .u32 pC0,\n\
         \x20   .param .u32 pC1,\n    .param .u64 pOut,\n    {}\n)\n.maxntid {TMA_PROBE_THREADS}, 1, 1\n{{\n",
        ptx_param_decl("tmap")
    );
    s += "    .reg .pred %p0,%p1;\n";
    s += "    .reg .b16 %h;\n";
    s += "    .reg .b32 %tx,%c0,%c1,%lin,%nthr,%i,%elems,%ph;\n";
    s += "    .reg .b64 %rdOut,%rdS,%rdBar,%rdT,%rdA,%rdTm,%rdSt;\n\n";
    s += "    ld.param.u32 %tx,[pTx];\n    ld.param.u32 %c0,[pC0];\n    ld.param.u32 %c1,[pC1];\n";
    s += "    ld.param.u64 %rdOut,[pOut];\n    cvta.to.global.u64 %rdOut,%rdOut;\n";
    s += &ptx_param_address("%rdTm", "tmap");
    s += &format!("    mov.u64 %rdS,{WGMMA_DSMEM_SYM};\n");
    // The barrier sits immediately after the tile, so one parameter fixes the whole layout.
    s += "    cvt.u64.u32 %rdT,%tx;\n    add.s64 %rdBar,%rdS,%rdT;\n";
    s += "    shr.u32 %elems,%tx,1;\n";
    s += "    mov.u32 %lin,%tid.x;\n    mov.u32 %nthr,%ntid.x;\n";
    // Phase parity lives in a REGISTER, set before any branch, even though it is the constant 0 for
    // the whole kernel. Not style: the ptxas census (2026-08-10) assembled `try_wait.parity` with a
    // register operand and `arrive.expect_tx` with an immediate one, so those two operand forms are
    // the ones this family has evidence for. The probe needs the transaction count to be a register
    // (that is what makes one module cover every geometry), which leaves the parity as the only
    // operand it could take on faith -- and an operand form ptxas rejects would fail stage A of the
    // bring-up and cost the whole visit. A register is legal wherever an immediate is; the converse
    // is not guaranteed, so the probe spends one `mov` and takes nothing on faith.
    s += "    mov.u32 %ph,0;\n";
    s += "    setp.eq.u32 %p0,%lin,0;\n";
    s += &format!("    @!%p0 bra PROBE_INITED_{name};\n");
    s += "    mbarrier.init.shared::cta.b64 [%rdBar],1;\n";
    // Every thread executes the `bar.sync`, which is why the init branch rejoins ABOVE it: a
    // `bar.sync` some threads skip is a hang, and this kernel exists to not have one.
    s += &format!("PROBE_INITED_{name}:\n    bar.sync 0;\n");
    s += &format!("    @!%p0 bra PROBE_WAIT_{name};\n");
    s += "    mbarrier.arrive.expect_tx.shared::cta.b64 %rdSt,[%rdBar],%tx;\n";
    s += "    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes \
          [%rdS],[%rdTm,{%c0,%c1}],[%rdBar];\n";
    // Phase parity 0: a freshly initialised barrier is in parity 0, so this completes exactly when
    // the transaction does. Same convention, and the same operand form, as the mainloop's consumer.
    s += &format!("PROBE_WAIT_{name}:\n");
    s += "    mbarrier.try_wait.parity.shared::cta.b64 %p1,[%rdBar],%ph;\n";
    s += &format!("    @!%p1 bra PROBE_WAIT_{name};\n");
    s += "    mov.u32 %i,%lin;\n";
    s += &format!("PROBE_COPY_{name}:\n");
    s += &format!("    setp.ge.u32 %p0,%i,%elems;\n    @%p0 bra PROBE_EXIT_{name};\n");
    s += "    mul.wide.u32 %rdT,%i,2;\n    add.s64 %rdA,%rdS,%rdT;\n";
    s += "    ld.shared.b16 %h,[%rdA];\n";
    s += "    add.s64 %rdA,%rdOut,%rdT;\n    st.global.b16 [%rdA],%h;\n";
    s += "    add.u32 %i,%i,%nthr;\n";
    s += &format!("    bra PROBE_COPY_{name};\n");
    s += &format!("PROBE_EXIT_{name}:\n    ret;\n}}\n");
    Ok(s)
}

// --- bring-up: the ONE-VISIT descriptor sweep -----------------------------------------------------

/// The sweep probe's square tile: `M = N = 64`, one warpgroup, one CTA. The smallest geometry in
/// which a core-matrix permutation is visible in **both** operands (8 x 8 core matrices each).
pub const DESC_SWEEP_M: usize = 64;
/// See [`DESC_SWEEP_M`]. `N = 64` also makes the `wgmma` shape `m64n64k16`, whose 32 accumulators fit
/// a plain 128-thread launch with no `setmaxnreg` anywhere in the kernel.
pub const DESC_SWEEP_N: usize = 64;
/// Staged K width in elements. 64 f16 elements = **128 bytes**, which is simultaneously the 128-B
/// swizzle atom, TMA's maximum contiguous box extent under it, and the `BK` every shipped row uses --
/// so the sweep is asking its question at the geometry production runs.
pub const DESC_SWEEP_BK: usize = 64;
/// Bytes of one staged tile: `64 x 64` f16.
pub const DESC_SWEEP_TILE: usize = DESC_SWEEP_M * DESC_SWEEP_BK * 2;
/// Byte offset of the A tile **as TMA wrote it**.
pub const DESC_SWEEP_A_RAW: usize = 0;
/// Byte offset of the B tile **as TMA wrote it**.
pub const DESC_SWEEP_B_RAW: usize = DESC_SWEEP_TILE;
/// Byte offset of the A tile **after the probe-only repack**.
pub const DESC_SWEEP_A_ALT: usize = 2 * DESC_SWEEP_TILE;
/// Byte offset of the B tile after the repack.
pub const DESC_SWEEP_B_ALT: usize = 3 * DESC_SWEEP_TILE;
/// Byte offset of the single mbarrier.
pub const DESC_SWEEP_BAR: usize = 4 * DESC_SWEEP_TILE;
/// **Total dynamic shared memory the probe requests -- deliberately far more than the four tiles and
/// a barrier need.**
///
/// A sweep launches candidates that are *wrong on purpose*, and a wrong (LBO, SBO) does not read a
/// wrong element of the tile -- it reads a wrong ADDRESS. The widest row in the set,
/// `canon-kfast/raw-fields`, hands the hardware a stride field of 1024 *unshifted*, which the
/// hardware then multiplies by 16: 16 KiB between row groups, so its last core matrix sits ~115 KiB
/// past its matrix base. Inside the CTA's window that reads garbage and scores zero, which is the
/// answer wanted. **Outside it, it is `CUDA_ERROR_MISALIGNED_ADDRESS`/`ILLEGAL_ADDRESS`, which makes
/// the context sticky-errored** (crate landmine 6) and ends the round at whichever candidate
/// happened to be first. So the window is sized to hold every candidate's furthest reach, and
/// `every_sweep_candidate_reads_inside_the_window` proves it device-free, before any rental.
///
/// 144 KiB, against H100's 227 KiB opt-in carveout. One CTA, so occupancy is not a consideration.
pub const DESC_SWEEP_SMEM: usize = 144 * 1024;
/// One warpgroup. `wgmma` is warpgroup-wide and `.aligned`; 128 threads is the minimum that can
/// issue one, and the probe deliberately has no producer/consumer split at all.
pub const DESC_SWEEP_THREADS: u32 = 128;
/// The probe's entry symbol. It **does** contain `wgmma` (that is the point), so unlike
/// [`TMA_PROBE_ENTRY`] the name may say so.
pub const DESC_SWEEP_ENTRY: &str = "wk_wgmma_desc_sweep";
/// The probe's `Gpu::function` module-cache key. **One key is correct**: every candidate's descriptor
/// template, staging mode, matrix offsets and K-step advance are run-time *parameters*, so all of
/// them run the same compiled module. That is the entire design -- one load, then a host loop.
pub const DESC_SWEEP_KEY: &str = "wk_wgmma_desc_sweep";
/// The two K values every candidate is run at, in report order.
///
/// `16` issues exactly **one** `wgmma`, so the descriptor's start address never advances and the
/// per-K-step base arithmetic is out of the picture; `64` issues four and puts it back. A candidate
/// exact at 16 and wrong at 64 has a k-step problem and nothing else -- a diagnosis that would
/// otherwise cost a second rental.
pub const DESC_SWEEP_KS: &[usize] = &[16, 64];

/// The transaction one launch of the probe declares: both tiles, one barrier.
pub const fn desc_sweep_tx_bytes() -> usize {
    2 * DESC_SWEEP_TILE
}

/// **How the probe stages the tile before the descriptor reads it.**
///
/// Modes 1-3 are a probe-only repack: plain `ld.shared`/`st.shared`, one element per thread per
/// iteration, correctness over speed. They exist because [`SmemLayout::CanonicalNone`] is a layout no
/// single tiled TMA copy writes, and a round that could not stage it would be unable to tell "the
/// descriptor fields are wrong" from "the descriptor model is wrong".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SweepStaging {
    /// The tile exactly as TMA left it, at the tensor map's own swizzle. No repack.
    AsWritten = 0,
    /// Repacked so each `8 x 16 B` core matrix is 128 contiguous bytes, core-matrix grid K-fastest.
    CanonicalKFast = 1,
    /// The same packing, core-matrix grid MN-fastest.
    CanonicalMnFast = 2,
    /// A **verbatim** element-by-element copy into the alternate region. Not a layout -- a control:
    /// it proves the repack loop itself moves bytes faithfully, so a canonical arm's failure accuses
    /// the layout and not the copier.
    VerbatimCopy = 3,
}

impl SweepStaging {
    /// Byte offsets of the A and B matrices the descriptors must point at under this mode.
    pub const fn operand_offsets(self) -> (usize, usize) {
        match self {
            SweepStaging::AsWritten => (DESC_SWEEP_A_RAW, DESC_SWEEP_B_RAW),
            _ => (DESC_SWEEP_A_ALT, DESC_SWEEP_B_ALT),
        }
    }
    pub const fn label(self) -> &'static str {
        match self {
            SweepStaging::AsWritten => "as-written",
            SweepStaging::CanonicalKFast => "canon-kfast",
            SweepStaging::CanonicalMnFast => "canon-mnfast",
            SweepStaging::VerbatimCopy => "verbatim",
        }
    }
}

/// **One row of the descriptor sweep**: a staging mode, a tensor-map swizzle, and the four descriptor
/// fields, plus the reason it is in the set.
///
/// Every field here becomes a **run-time kernel argument**, never a second PTX module. One
/// `cuModuleLoadData`, then a host loop that reads in seconds.
#[derive(Clone, Copy, Debug)]
pub struct DescCandidate {
    /// Stable ASCII label, printed in the sweep table and quoted in the round log.
    pub label: &'static str,
    /// **Why this row is in the set** -- one line, printed beside it, so a reader never has to guess
    /// what a candidate was testing.
    pub why: &'static str,
    pub staging: SweepStaging,
    pub encoding: FieldEncoding,
    pub lbo: u64,
    pub sbo: u64,
    pub base_offset: u8,
    pub swizzle: SmemSwizzle,
    /// Byte advance of the descriptor's start address per `wgmma` K step (16 elements).
    pub k_step_bytes: u32,
    /// The production [`SmemLayout`] this candidate **is**, when it is one. `None` for a row that
    /// exists only to be ruled out (a bad encoding, a phase probe, the verbatim control).
    pub layout: Option<SmemLayout>,
}

impl DescCandidate {
    /// The descriptor this candidate packs, with the start address left at zero (the kernel ORs in
    /// the run-time shared address).
    pub fn desc(&self) -> SmemDesc {
        SmemDesc {
            start_addr: 0,
            lbo: self.lbo,
            sbo: self.sbo,
            base_offset: self.base_offset,
            swizzle: self.swizzle,
        }
    }
    /// The 64-bit template the host passes to the kernel.
    pub fn template(&self) -> Result<u64, String> {
        self.desc().const_part_as(self.encoding)
    }
    /// The TMA swizzle the tensor maps for this candidate must be encoded with. It is the
    /// *descriptor's* mode translated through the reversed encoding -- never a second choice.
    pub const fn tma_swizzle(&self) -> TmaSwizzle {
        self.swizzle.to_tma()
    }
    /// Byte offsets of the A and B matrices the descriptors point at.
    pub const fn operand_offsets(&self) -> (usize, usize) {
        self.staging.operand_offsets()
    }
    /// **The two offset distances the HARDWARE will use**, in bytes, decoded back out of the packed
    /// template exactly as the hardware decodes them (`field * 16`).
    ///
    /// Not the same as `self.lbo` / `self.sbo` for the non-standard encodings, which is the point:
    /// [`FieldEncoding::Raw`] writes 1024 into a field the hardware reads as 16384 bytes, and it is
    /// the *decoded* value that decides whether a candidate reads outside the shared window.
    pub fn hw_fields(&self) -> Result<(u64, u64), String> {
        let t = self.template()?;
        Ok((((t >> 16) & 0x3FFF) * 16, ((t >> 32) & 0x3FFF) * 16))
    }
    /// **The furthest byte past its matrix base this candidate can make the hardware read**, over
    /// every `wgmma` K step, core matrix and element of the widest pass ([`DESC_SWEEP_KS`]'s last).
    ///
    /// The bound the shared-memory window must clear. See [`DESC_SWEEP_SMEM`] for why a candidate
    /// that reads past it does not score zero -- it ends the round.
    pub fn max_reach(&self) -> Result<u64, String> {
        let (lbo, sbo) = self.hw_fields()?;
        let steps = (*DESC_SWEEP_KS.last().expect("at least one K pass") / WgmmaShape::K) as u64;
        // (i, j) run over the core-matrix grid ONE descriptor covers: 8 MN groups (64 rows) and 2 K
        // groups (16 elements), plus row 7 and element 7 inside the last core matrix.
        Ok((steps - 1) * self.k_step_bytes as u64 + 7 * sbo + lbo + 7 * 16 + 7 * 2)
    }
    /// **The arm this candidate belongs to** -- the (staging, swizzle) pair, i.e. *which bytes are in
    /// shared memory*, as opposed to which field spelling reads them.
    ///
    /// The sweep's verdict is stated over arms rather than over rows because two rows of one arm can
    /// legitimately both be exact (a field the winning mode ignores has more than one spelling),
    /// while two *arms* being exact would mean two different shared-memory images both read
    /// correctly, which is the "the probe settles nothing" failure.
    pub fn arm(&self) -> String {
        format!("{}/{:?}", self.staging.label(), self.swizzle)
    }
    /// Whether this candidate's reading is one production could ship: a layout one tiled TMA copy
    /// writes, staged as TMA wrote it, at the ISA's own field encoding.
    pub fn production_viable(&self) -> bool {
        self.staging == SweepStaging::AsWritten
            && self.encoding == FieldEncoding::Standard
            && matches!(self.layout, Some(l) if l.tma_writable())
    }
}

/// **The candidate set -- derived, not sprayed.**
///
/// Eighteen rows in five arms, every one of which either (a) reproduces a number the 2026-08-10 H100
/// log already measured, (b) is a reading the PTX ISA text plausibly supports, or (c) isolates one
/// mechanism the others share. Every row carries its own `why`, printed beside it.
///
/// The fields of every row that corresponds to a real [`SmemLayout`] come from [`desc_fields`], the
/// single authority the production generator also reads -- so a winner reaches production by editing
/// [`SHIPPED_LAYOUT`] and nothing else.
pub fn desc_sweep_candidates() -> Vec<DescCandidate> {
    let rb = (DESC_SWEEP_BK * 2) as u64; // 128 B of shared row
    let rows = DESC_SWEEP_M as u64; // both operands: 64 rows per descriptor
    let of = |label: &'static str, why: &'static str, layout: SmemLayout| {
        let f = desc_fields(layout, rb, rows);
        DescCandidate {
            label,
            why,
            staging: match layout {
                SmemLayout::CanonicalNone { k_fast: true, .. } => SweepStaging::CanonicalKFast,
                SmemLayout::CanonicalNone { k_fast: false, .. } => SweepStaging::CanonicalMnFast,
                _ => SweepStaging::AsWritten,
            },
            encoding: FieldEncoding::Standard,
            lbo: f.lbo,
            sbo: f.sbo,
            base_offset: f.base_offset,
            swizzle: f.swizzle,
            k_step_bytes: f.k_step_bytes as u32,
            layout: Some(layout),
        }
    };
    let raw = |label: &'static str,
               why: &'static str,
               staging: SweepStaging,
               encoding: FieldEncoding,
               lbo: u64,
               sbo: u64,
               base_offset: u8,
               swizzle: SmemSwizzle,
               k_step_bytes: u32| DescCandidate {
        label,
        why,
        staging,
        encoding,
        lbo,
        sbo,
        base_offset,
        swizzle,
        k_step_bytes,
        layout: None,
    };
    let kf = SmemLayout::CanonicalNone {
        k_fast: true,
        swapped: false,
    };
    vec![
        // --- arm 1: the tile exactly as TMA wrote it, unswizzled. The two CONTROLS come first. ---
        of(
            "ctl/rowmajor-k-leading",
            "the reading this family shipped; the 2026-08-10 H100 log scored it 64/4096 exact, and \
             the sweep reproducing that number is how the sweep validates ITSELF",
            SmemLayout::RowMajorNone { k_leading: true },
        ),
        of(
            "ctl/rowmajor-mn-leading",
            "the other 2026-08-10 arm, logged at 0/4096 -- the axis-naming coin flip the round \
             already spent",
            SmemLayout::RowMajorNone { k_leading: false },
        ),
        raw(
            "rowmajor/lbo128-sbo1024",
            "the ISA figure's own numbers for a COMPACT 64x16 tile (LBO one core matrix, SBO one \
             row group) applied unchanged to the row-major tile -- the misreading's nearest \
             neighbour, and the only descriptor-only row whose LBO is a core matrix rather than a \
             k-group",
            SweepStaging::AsWritten,
            FieldEncoding::Standard,
            128,
            1024,
            0,
            SmemSwizzle::None,
            32,
        ),
        raw(
            "rowmajor/lbo16-sbo128",
            "SBO = ONE core matrix rather than one 8-row group: the reading in which the stride \
             field counts core matrices",
            SweepStaging::AsWritten,
            FieldEncoding::Standard,
            16,
            128,
            0,
            SmemSwizzle::None,
            32,
        ),
        raw(
            "rowmajor/lbo1024-sbo128",
            "the shape a CUTLASS make_gmma_desc printout carries (LBO 64 u128 = 1024 B, SBO 8 u128 \
             = 128 B) read straight onto the row-major tile",
            SweepStaging::AsWritten,
            FieldEncoding::Standard,
            1024,
            128,
            0,
            SmemSwizzle::None,
            32,
        ),
        // --- arm 2: the verbatim-copy control. ---
        raw(
            "copy/rowmajor-k-leading",
            "control row 1's descriptor over a tile the probe's OWN repack loop copied element by \
             element: if this disagrees with row 1 the repack loop is broken and every canonical \
             arm below is uninterpretable",
            SweepStaging::VerbatimCopy,
            FieldEncoding::Standard,
            16,
            1024,
            0,
            SmemSwizzle::None,
            32,
        ),
        // --- arm 3: the ISA's canonical no-swizzle core-matrix layout. THE HYPOTHESIS. ---
        of(
            "canon-kfast/lbo128-sbo1024",
            "THE LEADING HYPOTHESIS: every 8x16 B core matrix packed into 128 CONTIGUOUS bytes, \
             grid K-fastest; LBO = K-adjacent core matrix (128 B), SBO = MN-adjacent core-matrix \
             row (BK/8 * 128 = 1024 B); the base advances two core matrices (256 B) per K step",
            kf,
        ),
        of(
            "canon-kfast/swapped",
            "the axis-naming coin flip, asked over a layout a descriptor can actually describe",
            SmemLayout::CanonicalNone {
                k_fast: true,
                swapped: true,
            },
        ),
        of(
            "canon-mnfast/lbo1024-sbo128",
            "the MN-fastest canonical packing: core matrices along MN contiguous, so LBO \
             (K-adjacent) = rows/8 * 128 and SBO (MN-adjacent) = 128; the base advances a whole \
             column of core matrices per K step",
            SmemLayout::CanonicalNone {
                k_fast: false,
                swapped: false,
            },
        ),
        of(
            "canon-mnfast/swapped",
            "the coin flip over the MN-fastest packing",
            SmemLayout::CanonicalNone {
                k_fast: false,
                swapped: true,
            },
        ),
        raw(
            "canon-kfast/raw-fields",
            "the hypothesis with LBO/SBO written as RAW BYTES: rules out an encoding in which the \
             `>> 4` applies to the start address alone",
            SweepStaging::CanonicalKFast,
            FieldEncoding::Raw,
            128,
            1024,
            0,
            SmemSwizzle::None,
            256,
        ),
        raw(
            "canon-kfast/double-encoded",
            "the hypothesis with the `>> 4` applied TWICE (the mirror mistake); it degenerates to \
             LBO 0 / SBO 4 and is here only so the encoding question is closed in both directions",
            SweepStaging::CanonicalKFast,
            FieldEncoding::Twice,
            128,
            1024,
            0,
            SmemSwizzle::None,
            256,
        ),
        // --- arm 4: the 128-B swizzle. THE PRODUCTION CANDIDATE. ---
        of(
            "sw128/lbo16",
            "THE PRODUCTION CANDIDATE: at BK=64 f16 the shared row is exactly the 128-B swizzle \
             atom, and the ISA's canonical 128-B-swizzle K-major layout IS plain row-major (row \
             stride 128, k-group stride 16, SBO = 8*128) composed with Swizzle<3,4,3> -- precisely \
             what a SWIZZLE_128B TMA copy writes. No repack. LBO does not appear in that formula, \
             so 16 spells the fixed k-group stride",
            SmemLayout::Swizzle128 {
                lbo_bytes: 16,
                swapped: false,
            },
        ),
        of(
            "sw128/lbo128",
            "the same arm with the second plausible spelling of the LBO the mode ignores (one core \
             matrix)",
            SmemLayout::Swizzle128 {
                lbo_bytes: 128,
                swapped: false,
            },
        ),
        of(
            "sw128/lbo1024",
            "the same arm with LBO == SBO, the third spelling seen in the wild",
            SmemLayout::Swizzle128 {
                lbo_bytes: 1024,
                swapped: false,
            },
        ),
        of(
            "sw128/swapped",
            "the 128-B arm with the two fields exchanged (LBO = the 8-row atom, SBO = the k-group \
             stride). CUTLASS's descriptor builder assigns the pair one way for the unswizzled \
             canonical layout and the other way for the swizzled ones; the published derivation \
             keeps SBO on the MN axis in both. One launch settles which",
            SmemLayout::Swizzle128 {
                lbo_bytes: 16,
                swapped: true,
            },
        ),
        raw(
            "sw128/lbo16-bo1",
            "base_offset 1: if the hardware anchors the swizzle pattern at the descriptor's START \
             ADDRESS rather than at the enclosing 1024-B boundary, the 32-B-per-K-step base advance \
             needs a phase and every bo=0 row of this arm fails at K=64 while passing at K=16",
            SweepStaging::AsWritten,
            FieldEncoding::Standard,
            16,
            1024,
            1,
            SmemSwizzle::B128,
            32,
        ),
        raw(
            "sw128/lbo16-bo2",
            "the same phase probe at the value one whole 32-B K step would need",
            SweepStaging::AsWritten,
            FieldEncoding::Standard,
            16,
            1024,
            2,
            SmemSwizzle::B128,
            32,
        ),
        raw(
            "sw128/sbo128",
            "SBO = one 128-B row rather than one 8-row atom, in case the swizzled mode counts rows \
             where the unswizzled one counts row groups",
            SweepStaging::AsWritten,
            FieldEncoding::Standard,
            16,
            128,
            0,
            SmemSwizzle::B128,
            32,
        ),
    ]
}

/// Everything a launcher needs for [`desc_sweep_probe_module`], as pure data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescSweepPlan {
    pub entry: &'static str,
    pub module_key: &'static str,
    pub block: (u32, u32, u32),
    pub dyn_smem_bytes: usize,
    /// The `expect_tx` byte count the kernel declares: both `64 x 64` f16 tiles.
    pub tx_bytes: usize,
    /// Rows and columns of the square operand the host must upload (both operands are `64 x 64`,
    /// zero-filled past the K under test, so one tensor-map geometry covers every K value).
    pub operand: (usize, usize),
}

/// The sweep probe's launch plan. Pure data, so a launcher never re-derives a geometry.
pub const fn desc_sweep_plan() -> DescSweepPlan {
    DescSweepPlan {
        entry: DESC_SWEEP_ENTRY,
        module_key: DESC_SWEEP_KEY,
        block: (DESC_SWEEP_THREADS, 1, 1),
        dyn_smem_bytes: DESC_SWEEP_SMEM,
        tx_bytes: desc_sweep_tx_bytes(),
        operand: (DESC_SWEEP_M, DESC_SWEEP_BK),
    }
}

/// **`WGMMA_DEVICE_VALIDATION` item 1, as ONE module: the descriptor sweep probe.**
///
/// A `64 x 64 x 64` `A * Bt` in the smallest shape that can issue a `wgmma` at all -- one warpgroup,
/// one CTA, one stage, no warp specialisation, no `setmaxnreg`, no pipeline. Everything a candidate
/// varies is a **run-time parameter**:
///
/// * `pDescA` / `pDescB` -- the 64-bit descriptor templates, host-computed. The kernel ORs in
///   `((addr >> 4) & 0x3FFF)` exactly as the mainloop does.
/// * `pAOff` / `pBOff` -- where in the window each descriptor's matrix starts, so a repacked arm and
///   an as-written arm are the same module.
/// * `pKStep` -- how far the descriptor base advances per `wgmma` K step.
/// * `pStage` -- [`SweepStaging`]: 0 leaves the tile as TMA wrote it, 1 and 2 repack it into the
///   canonical core-matrix layouts, 3 copies it verbatim as a control on the repack loop itself.
/// * `pK` -- 16 issues one `wgmma` (no base advance at all), 64 issues four.
///
/// So the whole sweep is **one `cuModuleLoadData` and a host loop**. No per-candidate PTX, no
/// rebuild, no second rental to try the next value -- which is the entire reason this file grew a
/// second probe rather than a second variant table.
///
/// The staging path is the one stage B of the 2026-08-10 round proved byte-exact: the same
/// `cuTensorMapEncodeTiled` geometry, the same `cp.async.bulk.tensor.2d`, the same mbarrier.
///
/// `fence.proxy.async.shared::cta` after the repack is **not decoration**: `wgmma` reads shared
/// memory through the async proxy, the repack writes it with ordinary `st.shared`, and without the
/// fence the generic-proxy writes are not guaranteed visible. TMA's own writes are already
/// async-proxy and are published by the barrier, so the production mainloop needs no such fence --
/// only this probe does, because only this probe writes SMEM by hand.
pub fn desc_sweep_probe_module(license: &Sm90aLicense) -> Result<String, String> {
    let name = DESC_SWEEP_ENTRY;
    let nacc = DESC_SWEEP_N / 2;
    let accs = (0..nacc)
        .map(|i| format!("%acc{i}"))
        .collect::<Vec<_>>()
        .join(",");
    let elems = DESC_SWEEP_TILE / 2;
    let tx = desc_sweep_tx_bytes();

    let mut s = String::from(sm90a_header(license));
    s += WGMMA_DSMEM_DECL;
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pK,\n    .param .u64 pC,\n\
         \x20   .param .u64 pDescA,\n    .param .u64 pDescB,\n    .param .u32 pAOff,\n\
         \x20   .param .u32 pBOff,\n    .param .u32 pKStep,\n    .param .u32 pStage,\n\
         \x20   {},\n    {}\n)\n.maxntid {DESC_SWEEP_THREADS}, 1, 1\n{{\n",
        ptx_param_decl("tmapA"),
        ptx_param_decl("tmapB")
    );
    s += "    .reg .pred %p0,%p1,%p2,%pfirst,%pst;\n";
    s += "    .reg .b16 %h;\n";
    s += "    .reg .b32 %K,%N,%lin,%lane,%wrp,%kt,%ksteps,%aoff,%boff,%kstep,%stg,%ph,%zero,\
          %tmp,%tmp2,%i,%rw,%cl,%ri,%rr,%ci,%cc,%d1,%d2,%dst,%row0,%row1,%colb;\n";
    s += "    .reg .b64 %rdC,%rdS,%rdT,%rdA,%rdB,%rdBar,%rdTmA,%rdTmB,%rdAddr,\
          %descA,%descB,%tmplA,%tmplB,%rdSt;\n";
    s += &format!("    .reg .f32 %acc<{nacc}>;\n\n");

    // --- parameters ---------------------------------------------------------------------------
    s += "    ld.param.u32 %K,[pK];\n";
    s += "    ld.param.u64 %rdC,[pC];\n    cvta.to.global.u64 %rdC,%rdC;\n";
    s += "    ld.param.u64 %tmplA,[pDescA];\n    ld.param.u64 %tmplB,[pDescB];\n";
    s += "    ld.param.u32 %aoff,[pAOff];\n    ld.param.u32 %boff,[pBOff];\n";
    s += "    ld.param.u32 %kstep,[pKStep];\n    ld.param.u32 %stg,[pStage];\n";
    s += &ptx_param_address("%rdTmA", "tmapA");
    s += &ptx_param_address("%rdTmB", "tmapB");
    s += &format!("    mov.u64 %rdS,{WGMMA_DSMEM_SYM};\n");
    s += "    mov.u32 %lin,%tid.x;\n    mov.u32 %lane,%laneid;\n";
    // Both operand forms the 2026-08-10 ptxas census assembled: an immediate expect_tx and a
    // register parity. Set before any branch, so a thread that skips the init does not read an
    // undefined register.
    s += "    mov.u32 %ph,0;\n    mov.u32 %zero,0;\n";
    s += "    shr.u32 %ksteps,%K,4;\n";

    // --- one barrier, two copies -----------------------------------------------------------------
    s += &format!("    add.s64 %rdBar,%rdS,{DESC_SWEEP_BAR};\n");
    s += "    setp.eq.u32 %p0,%lin,0;\n";
    s += &format!("    @!%p0 bra SWP_INITED_{name};\n");
    s += "    mbarrier.init.shared::cta.b64 [%rdBar],1;\n";
    s += &format!("SWP_INITED_{name}:\n    bar.sync 0;\n");
    s += &format!("    @!%p0 bra SWP_WAIT_{name};\n");
    s += &format!("    mbarrier.arrive.expect_tx.shared::cta.b64 %rdSt,[%rdBar],{tx};\n");
    s += &format!("    add.s64 %rdA,%rdS,{DESC_SWEEP_A_RAW};\n");
    s += "    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes \
          [%rdA],[%rdTmA,{%zero,%zero}],[%rdBar];\n";
    s += &format!("    add.s64 %rdB,%rdS,{DESC_SWEEP_B_RAW};\n");
    s += "    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes \
          [%rdB],[%rdTmB,{%zero,%zero}],[%rdBar];\n";
    s += &format!("SWP_WAIT_{name}:\n");
    s += "    mbarrier.try_wait.parity.shared::cta.b64 %p1,[%rdBar],%ph;\n";
    s += &format!("    @!%p1 bra SWP_WAIT_{name};\n");

    // --- the probe-only repack --------------------------------------------------------------------
    s += "    setp.eq.u32 %pst,%stg,0;\n";
    s += &format!("    @%pst bra SWP_STAGED_{name};\n");
    for (tag, src, dst) in [
        ("A", DESC_SWEEP_A_RAW, DESC_SWEEP_A_ALT),
        ("B", DESC_SWEEP_B_RAW, DESC_SWEEP_B_ALT),
    ] {
        s += "    mov.u32 %i,%lin;\n";
        s += &format!("SWP_RP{tag}_{name}:\n");
        s += &format!("    setp.ge.u32 %p0,%i,{elems};\n    @%p0 bra SWP_RPD{tag}_{name};\n");
        // (rw, cl) = (i / 64, i % 64); the core-matrix grid indices and the position inside one.
        s += "    shr.u32 %rw,%i,6;\n    and.b32 %cl,%i,63;\n";
        s += "    shr.u32 %ri,%rw,3;\n    and.b32 %rr,%rw,7;\n";
        s += "    shr.u32 %ci,%cl,3;\n    and.b32 %cc,%cl,7;\n";
        s += "    shl.b32 %tmp,%rr,3;\n    add.u32 %tmp,%tmp,%cc;\n";
        // K-fastest: (ri * ncm_k + ci) * 64 + within;  ncm_k = 64/8 = 8, so ri*512 + ci*64.
        s += "    shl.b32 %d1,%ri,9;\n    shl.b32 %tmp2,%ci,6;\n    add.u32 %d1,%d1,%tmp2;\n\
              \x20   add.u32 %d1,%d1,%tmp;\n";
        // MN-fastest: (ci * ncm_m + ri) * 64 + within.
        s += "    shl.b32 %d2,%ci,9;\n    shl.b32 %tmp2,%ri,6;\n    add.u32 %d2,%d2,%tmp2;\n\
              \x20   add.u32 %d2,%d2,%tmp;\n";
        // Mode 3 (and any other) copies verbatim, which is what makes it a control on this loop.
        s += "    setp.eq.u32 %p1,%stg,1;\n    selp.b32 %dst,%d1,%i,%p1;\n";
        s += "    setp.eq.u32 %p1,%stg,2;\n    selp.b32 %dst,%d2,%dst,%p1;\n";
        s += &format!(
            "    mul.wide.u32 %rdT,%i,2;\n    add.s64 %rdAddr,%rdS,%rdT;\n\
             \x20   add.s64 %rdAddr,%rdAddr,{src};\n    ld.shared.b16 %h,[%rdAddr];\n"
        );
        s += &format!(
            "    mul.wide.u32 %rdT,%dst,2;\n    add.s64 %rdAddr,%rdS,%rdT;\n\
             \x20   add.s64 %rdAddr,%rdAddr,{dst};\n    st.shared.b16 [%rdAddr],%h;\n"
        );
        s += &format!("    add.u32 %i,%i,{DESC_SWEEP_THREADS};\n    bra SWP_RP{tag}_{name};\n");
        s += &format!("SWP_RPD{tag}_{name}:\n");
    }
    s += &format!("SWP_STAGED_{name}:\n    bar.sync 0;\n");
    // Generic-proxy stores must be published to the async proxy before wgmma reads them.
    s += "    fence.proxy.async.shared::cta;\n";

    // --- the mainloop -------------------------------------------------------------------------
    s += "    cvt.u64.u32 %rdA,%aoff;\n    add.s64 %rdA,%rdS,%rdA;\n";
    s += "    cvt.u64.u32 %rdB,%boff;\n    add.s64 %rdB,%rdS,%rdB;\n";
    s += "    setp.eq.u32 %p2,%ksteps,0;\n";
    s += &format!("    @%p2 bra SWP_EXIT_{name};\n");
    s += "    mov.u32 %kt,0;\n";
    s += "    wgmma.fence.sync.aligned;\n";
    s += &format!("SWP_K_{name}:\n");
    s += &format!("    setp.ge.u32 %p0,%kt,%ksteps;\n    @%p0 bra SWP_KEND_{name};\n");
    s += "    mul.lo.s32 %tmp,%kt,%kstep;\n    cvt.u64.u32 %rdT,%tmp;\n";
    s += "    add.s64 %rdAddr,%rdA,%rdT;\n";
    s += "    shr.u64 %rdT,%rdAddr,4;\n    and.b64 %rdT,%rdT,16383;\n    or.b64 %descA,%rdT,%tmplA;\n";
    s += "    mul.lo.s32 %tmp,%kt,%kstep;\n    cvt.u64.u32 %rdT,%tmp;\n";
    s += "    add.s64 %rdAddr,%rdB,%rdT;\n";
    s += "    shr.u64 %rdT,%rdAddr,4;\n    and.b64 %rdT,%rdT,16383;\n    or.b64 %descB,%rdT,%tmplB;\n";
    // scale-d = 0 on the first step only: the ISA's own way to skip zeroing the accumulators.
    s += "    setp.ne.u32 %pfirst,%kt,0;\n";
    s += &format!(
        "    wgmma.mma_async.sync.aligned.m64n{DESC_SWEEP_N}k16.f32.f16.f16 \
         {{{accs}}}, %descA, %descB, %pfirst, 1, 1, 0, 0;\n"
    );
    s += "    add.u32 %kt,%kt,1;\n";
    s += &format!("    bra SWP_K_{name};\n");
    s += &format!("SWP_KEND_{name}:\n");
    s += "    wgmma.commit_group.sync.aligned;\n    wgmma.wait_group.sync.aligned 0;\n";

    // --- epilogue -------------------------------------------------------------------------------
    // The m64nNk16 D fragment, identical to the mainloop's: warp `w` holds rows `16w + lane/4` and
    // `16w + lane/4 + 8`; register group `j` covers columns `8j + 2*(lane%4)` and one past it.
    s += "    shr.u32 %wrp,%lin,5;\n    shl.b32 %wrp,%wrp,4;\n";
    s += "    shr.u32 %tmp2,%lane,2;\n    add.u32 %row0,%wrp,%tmp2;\n    add.u32 %row1,%row0,8;\n";
    s += "    and.b32 %colb,%lane,3;\n    shl.b32 %colb,%colb,1;\n";
    // N in a REGISTER, not an immediate operand of `mad`: the register form is the one the
    // 2026-08-10 ptxas census already assembled in the mainloop's epilogue, and a probe that takes
    // an operand class on faith can fail stage A and cost the visit.
    s += &format!("    mov.u32 %N,{DESC_SWEEP_N};\n");
    s += "    mad.lo.s32 %tmp,%row0,%N,%colb;\n    mul.wide.u32 %rdT,%tmp,4;\n\
          \x20   add.s64 %rdA,%rdC,%rdT;\n";
    s += "    mad.lo.s32 %tmp,%row1,%N,%colb;\n    mul.wide.u32 %rdT,%tmp,4;\n\
          \x20   add.s64 %rdB,%rdC,%rdT;\n";
    // No store predicate: the probe's output is exactly the 64x64 the tile covers, so every
    // accumulator has a lane. (The mainloop's `row < M && col < N` guard exists for ragged shapes,
    // which stage F asks about and this probe deliberately does not.)
    for j in 0..DESC_SWEEP_N / 8 {
        let byte = j * 32;
        s += &format!("    st.global.f32 [%rdA+{byte}],%acc{};\n", 4 * j);
        s += &format!("    st.global.f32 [%rdA+{}],%acc{};\n", byte + 4, 4 * j + 1);
        s += &format!("    st.global.f32 [%rdB+{byte}],%acc{};\n", 4 * j + 2);
        s += &format!("    st.global.f32 [%rdB+{}],%acc{};\n", byte + 4, 4 * j + 3);
    }
    s += &format!("SWP_EXIT_{name}:\n    ret;\n}}\n");
    Ok(s)
}

// --- bring-up: the operator's invocation ----------------------------------------------------------

/// **The exact command the first Hopper round runs.** Kept as data so the decline gate on a
/// non-Hopper part can print it, and so it cannot drift from the test it names.
///
/// `--test-threads=1` is not optional: the bring-up gate is a *sequence* whose whole value is that a
/// failure names which checklist item failed, and libtest interleaves output from concurrent tests.
/// `--nocapture` is not optional either — libtest swallows a passing test's stderr, and every verdict
/// this round produces is printed, not asserted. **Since 2026-08-10 that is doubly true**: stage D is
/// a sweep whose product is a TABLE ([`desc_sweep_candidates`] x [`DESC_SWEEP_KS`]) plus per-candidate
/// 8x8 core-matrix lane maps. A run without `--nocapture` that happens to pass throws away everything
/// the rented minutes were spent producing, and a run that fails prints only the panic.
pub const WGMMA_BRINGUP_INVOCATION: &str =
    "WUKONG_GPU_REQUIRED=1 cargo test -p wukong_codegen_gpu --features gpu \
     -- --nocapture --test-threads=1 wgmma_hopper_bringup";

/// **What the first H100 hour must confirm, in priority order** -- this family's own list of claims
/// no test in this repo can reach, kept as data so a bring-up script can print it.
///
/// There is no Hopper part here and `sm_90a` cannot be JITed, emulated or `ptxas`-checked anywhere in
/// this environment, so everything below is unproven text until it runs. Work down the list; each
/// item's failure mode is stated because several of them are silent.
pub const WGMMA_DEVICE_VALIDATION: &[&str] = &[
    "1. The descriptor reading, settled in ONE visit by a SWEEP. The 2026-08-10 round already spent \
     the two-arm A/B and both arms lost (64/4096 and 0/4096), because the ambiguity was never the \
     axis naming: an SS operand is a grid of core matrices packed at 128 contiguous bytes, so the \
     within-core-matrix row stride is a fixed 16 and a plain row-major tile is unrepresentable. \
     `wgmma_hopper_bringup` stage D therefore loads ONE module -- `desc_sweep_probe_module`, whose \
     descriptor template, matrix offsets, K-step advance and SMEM staging mode are all run-time \
     PARAMETERS -- and a host loop runs every row of `desc_sweep_candidates` at K=16 and K=64 \
     against an f64 reference over `bringup_operands`, whose three-digit positional ramps make a \
     core-matrix permutation visible in A and in B. Exactly one ARM (one shared-memory image) must \
     come out exact, bit-exactly: the operands are exact in f16 and the dot product stays under \
     2^24, so the verdict is `==`, not a tolerance. Two arms means the probe settles nothing; zero \
     means the round prints the per-core-matrix lane map and names the next diagnosis. The two \
     already-scored readings are in the set as CONTROLS -- the sweep reproducing 64/4096 is how it \
     validates itself.",
    "2. The module loads at all -- cuModuleLoadData on the generated text (`wgmma_hopper_bringup` \
     stage A, which loads every shipped row AND the TMA probe AND the descriptor sweep probe before \
     launching anything). First check of the `.target sm_90a` header, of `.version 8.0`, and of \
     every instruction spelling in the family.",
    "3. setmaxnreg and the register split -- cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS) plus a \
     spill check (ptxas -v via WUKONG_PTXAS, or the JIT log). If ptxas cannot fit a consumer's 128 \
     accumulators plus addressing inside 232 registers, consumer_regs must rise and producer_regs \
     fall; the budget assert in WgmmaCfg::validate will keep the pair honest.",
    "4. The TMA descriptors (`wgmma_hopper_bringup` stage B) -- cuTensorMapEncodeTiled succeeding \
     for the A and B geometries at each benched shape, then `tma_stage_probe_module`: a \
     single-stage load compared against a host copy of the same tile, on a kernel with no wgmma, no \
     warp specialisation and no pipeline in it, so a mismatch accuses the descriptor and nothing \
     else.",
    "5. The pipeline does not hang (`wgmma_hopper_bringup` stage C, and the time-box that wraps \
     every launch in this family). The expect_tx count is WgmmaCfg::stage_tx_bytes; if it \
     disagrees with what the two copies actually move, the barrier never completes and every \
     consumer waits forever. `gpu::sync_within` polls cuStreamQuery to a deadline and ENDS THE \
     PROCESS with a diagnosis rather than billing rented silicon until the container timeout; \
     WUKONG_GPU_LAUNCH_TIMEOUT_MS tunes it.",
    "6. The producer warpgroup returns with copies possibly still in flight (CUTLASS's producer \
     tail exists for barrier lifetime, which a CTA whose consumers are still running does not \
     need). `wgmma_hopper_bringup` stage E sweeps K across the ring depth -- fewer K tiles than \
     stages (the producer exits while consumers still wait), exactly as many, and several wraps -- \
     confirming no hang and no early SMEM reclaim, against the reference each time.",
    "7. Ragged shapes (`wgmma_hopper_bringup` stage F) -- M, N and K each not a multiple of the \
     tile, against the f64 reference, to confirm TMA's zero fill and the predicated epilogue \
     together cover the edges.",
    "8. Only then, performance: same-run adjacent A/B against cuBLAS at 4096 and 8192 cubed, with \
     bench_instrument's twin control passing. The gates are `wgmma_vs_cublas` (WGMMA_W1, f16) and \
     `wgmma_bf16_vs_cublas` (WGMMA_W1_BF16), both `#[ignore]`d benches in gpu.rs, both sweeping \
     `WGMMA_BENCH_GRID` -- D1 section 4.4's confirmation grid plus the skinny-N companion -- and \
     both routing every arm through `bench_instrument`: a pre-registered TwinPlan, `run_rotated` \
     with the cuBLAS peer as BOTH the A and the C arm (the cuBLAS-called-twice control plan \
     section 6.2 asks for, so C/A is the peer's own noise floor and B/A is the claim), `analyze` \
     for the per-shape floor, and `Round::publish` as the only way a number leaves the round. They \
     PRINT a table and assert NOTHING about speed -- a slow result is a finding, not a test \
     failure; the only asserts are the exact-integer correctness spot-check that gates the timing \
     and the structural ones. Run them as WGMMA_BENCH_INVOCATION.",
    "9. THE CLUSTER. The 2026-08-10 Act-2 round measured 58.8-67.5% of cuBLAS where D1 4.5 \
     predicts 95-108%, on a clean instrument, and its own provenance line named the reason it \
     could not be the whole story: the measured kernel was W1 MINUS the 2x1x1 cluster with \
     .multicast::cluster on A. WGMMA_W1_MC is that arm. Four mechanisms are unproven and each \
     fails silently or hangs rather than erroring: the ctaMask and the per-rank A slice (a wrong \
     one is stale shared memory on the M rows this CTA did not fetch itself); the per-destination \
     transaction count (a wrong one hangs); the cluster-scoped empty-barrier arrivals through \
     mapa (a missing one lets a producer overwrite a slice a peer is still reading); and the two \
     barrier.cluster rendezvous, without which a peer signals an uninitialised mbarrier or writes \
     the shared memory of a CTA that has exited. `wgmma_cluster_multicast_is_exact` in gpu.rs is \
     the correctness gate -- four shapes that each span at least one full two-CTA cluster, \
     including one with an ODD N tile count so the rounded grid's pad CTA is exercised, each \
     verified `==` against the f64 reference AND bit-identical to the un-clustered row. \
     `wgmma_config_sweep` is the performance round (WGMMA_SWEEP_INVOCATION): cluster off vs on at \
     a fixed tile and depth, the depth axis at BOTH cluster settings so the two are not \
     confounded, W3c as its own row, and the two over-budget depths kept as printed declines.",
];

// --- Act 2: the performance grid --------------------------------------------------------------------

/// **The exact command the Act-2 performance round runs**, as data, for the same reason
/// [`WGMMA_BRINGUP_INVOCATION`] is data: the benches print their tables rather than asserting them,
/// so a run without `--nocapture` throws away everything the rented minutes produced.
///
/// `--release` is load-bearing here and is not in the bring-up invocation: a debug host loop makes
/// the peer's and our own launch overhead dominate a 25-microsecond kernel, and this crate's own
/// history has a debug-build measurement artifact in it. `--ignored` is how libtest reaches an
/// `#[ignore]`d bench at all. The cloud form is
/// `modal run tools/cloud/modal_app.py::bench --name wgmma_vs_cublas --peers`, which sets
/// `WUKONG_GPU_REQUIRED=1` + `WUKONG_PEER_REQUIRED=1` and adds exactly these libtest flags.
pub const WGMMA_BENCH_INVOCATION: &str =
    "WUKONG_GPU_REQUIRED=1 WUKONG_PEER_REQUIRED=1 cargo test -p wukong_codegen_gpu --features gpu \
     --release -- --ignored --nocapture --test-threads=1 wgmma_vs_cublas";

/// One point of the Act-2 performance grid: an `M x N x K` row-major NT GEMM (`C = A * Bt`), with
/// the reason D1 lists it.
///
/// The grid is **data**, not a literal inside a bench body, for three reasons: a device-free gate can
/// check every point is launchable by every shipped row before an hour is rented
/// (`the_bench_grid_is_launchable_by_every_shipped_row`); the round log can print *why* a row is in
/// the table without a reader opening the dossier; and the two benches (f16 and bf16) provably sweep
/// the same shapes, which a copy-pasted array would not guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmPoint {
    /// Round-log row name and `bench_instrument` sample-label suffix. ASCII, no whitespace.
    pub label: &'static str,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// Why D1 lists this shape, in one line.
    pub why: &'static str,
}

impl GemmPoint {
    /// `2*M*N*K` -- the FLOP count, for turning seconds/launch into a rate. **Diagnostic only**: a
    /// rate is an absolute, and this repo publishes ratios (see `docs/metrics.md`).
    pub fn flop(&self) -> f64 {
        2.0 * self.m as f64 * self.n as f64 * self.k as f64
    }

    /// Device bytes one arm's three buffers occupy: A and B at `elem` bytes, C at 4 (f32 accumulate,
    /// which both our epilogue and the f32-out cuBLAS peer store).
    pub fn device_bytes(&self, elem: usize) -> usize {
        (self.m * self.k + self.n * self.k) * elem + self.m * self.n * 4
    }

    /// `M x N x K`, for a log line.
    pub fn dims(&self) -> String {
        format!("{}x{}x{}", self.m, self.n, self.k)
    }
}

/// **D1 section 4.4's confirmation grid**, plus one derived companion, as the shapes the Act-2 peer
/// rounds sweep.
///
/// D1 asks for `1024 / 2048 / 4096 / 8192` cubed "+ GPT (M=4096, N=4d, K=d) at d in {1024, 4096}",
/// and section 5.3 tabulates where each lands against a 50 MiB L2. The four cubes and the two GPT
/// rows below are those points verbatim.
///
/// **`gpt_d1024_down` is the one row the dossier does not list**, and it is here because section 4.4's
/// A4 hypothesis is explicitly about *which way* a tile should be widened: A3 (128x256) and A4
/// (256x128) have identical CTA intensity and deliberately different warp-tile intensity, so the
/// question "does widening N buy reuse more cheaply than widening M" needs an `N << M` shape as well
/// as the listed `N >> M` one. It is the FFN **down**-projection twin of the listed `d=1024`
/// up-projection: same layer, same `d`, the other GEMM. Labelled as derived rather than quoted.
pub const WGMMA_BENCH_GRID: &[GemmPoint] = &[
    GemmPoint {
        label: "sq1024",
        m: 1024,
        n: 1024,
        k: 1024,
        why: "D1 5.3: 4 MiB working set, 0.08x L2 -- deeply L2-resident, and only 32 CTAs of a \
              128x256 tile, so this row is about wave quantization, not bandwidth",
    },
    GemmPoint {
        label: "sq2048",
        m: 2048,
        n: 2048,
        k: 2048,
        why: "D1 5.3: 16 MiB, 0.32x L2 -- L2-resident. D1 4.5 puts W3 (128x128) here because a \
              256-wide tile quantizes below M*N = 4.3e6; W1 is measured here anyway, as the \
              evidence for that claim",
    },
    GemmPoint {
        label: "sq4096",
        m: 4096,
        n: 4096,
        k: 4096,
        why: "D1 5.3: 64 MiB, 1.28x L2 -- the raster crossover, and the first of the two shapes \
              D1 4.5 predicts W1 at 95-108% of cuBLAS on",
    },
    GemmPoint {
        label: "sq8192",
        m: 8192,
        n: 8192,
        k: 8192,
        why: "D1 5.3: 256 MiB, 5.12x L2 -- raster plus streaming C, and the second 95-108% point",
    },
    GemmPoint {
        label: "gpt_d1024_up",
        m: 4096,
        n: 4096,
        k: 1024,
        why: "SKINNY-K. GPT d=1024 FFN up-projection (M=4096, N=4d, K=d). D1 4.3.2: ws is exactly \
              16.0 MiB, which trips the Act-1 dispatcher's 16 MiB literal into the raster arm \
              although 0.32x L2 is deeply L2-resident",
    },
    GemmPoint {
        label: "gpt_d1024_down",
        m: 4096,
        n: 1024,
        k: 4096,
        why: "SKINNY-N (derived, not in D1's table): the down-projection twin of the row above. \
              D1 4.4's A4 asks whether widening N or widening M buys reuse; W1 is N-major, so the \
              N << M shape is where that costs something",
    },
    GemmPoint {
        label: "gpt_d4096_up",
        m: 4096,
        n: 16384,
        k: 4096,
        why: "GPT d=4096 FFN up-projection. D1 5.3: 160 MiB, 3.20x L2 -- raster, and the widest \
              N the epilogue's u32 element index still holds (M*N = 67.1e6)",
    },
];

/// **The FLOP budget one timed region aims at**: roughly 20 TFLOP, which at a Hopper-class rate is
/// tens of milliseconds -- long enough that the host clock resolves it and short enough that a
/// seven-shape, three-arm, six-round sweep is seconds of rented time rather than minutes.
pub const BENCH_TARGET_FLOP: f64 = 2.0e13;

/// Floor on [`bench_iters`]. A timed region of one launch is a lone timing, which this repo does not
/// treat as a measurement.
pub const BENCH_MIN_ITERS: usize = 16;

/// Ceiling on [`bench_iters`]. At the small end the kernel is launch-bound, and past this the extra
/// launches buy resolution the host clock already has.
pub const BENCH_MAX_ITERS: usize = 1000;

/// Launches per timed region at this shape: [`BENCH_TARGET_FLOP`] worth of work, clamped.
///
/// Pure and device-free so the whole grid's cost is knowable before an hour is rented, and so the
/// two benches provably use the same count at the same shape -- an A/B whose two arms ran a
/// different number of launches would divide by a different denominator.
pub fn bench_iters(p: &GemmPoint) -> usize {
    let want = (BENCH_TARGET_FLOP / p.flop()).round();
    if !want.is_finite() || want <= BENCH_MIN_ITERS as f64 {
        return BENCH_MIN_ITERS;
    }
    (want as usize).min(BENCH_MAX_ITERS)
}

// --- the generator --------------------------------------------------------------------------------

/// Generate the PTX module for one configuration.
///
/// Requires an [`Sm90aLicense`]: `sm_90a` text cannot be produced by a path that has not established
/// the device is Hopper. Returns `Err` with an [`UNSUPPORTED`]-prefixed message for any shape this
/// generator cannot express -- **never** plausible-but-wrong PTX.
pub fn wgmma_module(cfg: &WgmmaCfg, license: &Sm90aLicense) -> Result<String, String> {
    let shape = cfg.validate()?;
    let mut m = String::from(sm90a_header(license));
    m += WGMMA_DSMEM_DECL;
    m += &entry(cfg, shape)?;
    Ok(m)
}

/// The `.visible .entry` for one configuration.
fn entry(cfg: &WgmmaCfg, shape: WgmmaShape) -> Result<String, String> {
    let name = cfg.name;
    let (bm, bn, bk) = (cfg.bm, cfg.bn, cfg.bk);
    let threads = cfg.threads();
    let nacc = shape.accum_regs();
    let row_bytes = (bk * cfg.dtype.size()) as u64;
    let tile_a = cfg.tile_a_bytes();
    let tile_b = cfg.tile_b_bytes();
    let full_base = cfg.full_off(0);
    let empty_base = cfg.empty_off(0);
    let per_consumer_a = 64 * bk * cfg.dtype.size(); // bytes of A one consumer's m64 slice starts at
    let bk_shift = bk.trailing_zeros();
    // --- the cluster, if there is one -------------------------------------------------------------
    // EVERY line below is conditional on `clustered`, deliberately: at `Multicast::None` this
    // generator must emit the byte-identical text the 2026-08-10 Act-2 round measured, so that row
    // remains the control arm of the cluster A/B rather than a second thing that also changed.
    let ctas = cfg.cluster_ctas();
    let clustered = ctas > 1;
    let a_slice = cfg.a_slice_bytes();
    let a_box_rows = cfg.a_box_rows();
    let cmask = multicast_cta_mask(ctas);

    // The two descriptor constants: everything but the start address, which the kernel folds in at
    // run time. Both come from `desc_fields`, the single authority the sweep candidates read too,
    // and they are computed separately because one layout (`CanonicalNone` MN-fastest) makes the
    // fields depend on how many rows a descriptor covers -- 64 for A's per-consumer slab, `bn` for B.
    let a_fields = desc_fields(cfg.layout, row_bytes, WgmmaShape::M as u64);
    let b_fields = desc_fields(cfg.layout, row_bytes, bn as u64);
    let const_a =
        SmemDesc::for_layout(0, cfg.layout, row_bytes, WgmmaShape::M as u64).const_part()?;
    let const_b = SmemDesc::for_layout(0, cfg.layout, row_bytes, bn as u64).const_part()?;
    // How far each descriptor's start address moves per wgmma K step. Also from the authority: a
    // layout whose K direction is not contiguous advances by whole core matrices, not by 32 bytes.
    let (a_step, b_step) = (a_fields.k_step_bytes, b_fields.k_step_bytes);

    let accs = (0..nacc)
        .map(|i| format!("%acc{i}"))
        .collect::<Vec<_>>()
        .join(",");
    let mma = format!(
        "wgmma.mma_async.sync.aligned.{}.{}",
        shape.token(),
        cfg.dtype.mma_types()
    );

    let mut s = String::with_capacity(64 * 1024);
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n\
         \x20   .param .u64 pC,\n    {},\n    {}\n)\n.maxntid {threads}, 1, 1\n",
        ptx_param_decl("tmapA"),
        ptx_param_decl("tmapB")
    );
    if clustered {
        // **The cluster shape is COMPILED IN, not merely requested at launch.** PTX ISA 11.7:
        // "For kernels with .reqnctapercluster directive specified, runtime will use the specified
        // values for configuring the launch if the same are not specified at launch time ... If
        // cluster dimension is explicitly specified at launch time, it should be equal to the values
        // specified in this directive." That is the safety net that matters here: this kernel `mapa`s
        // to rank 1 and multicasts to a two-bit ctaMask, so a `1x1x1` launch of it would be silently
        // WRONG rather than merely unoptimised -- and with this directive such a launch fails
        // instead. `.explicitcluster` closes the other side ("must be launched with cluster dimension
        // explicitly specified, either at launch time or via .reqnctapercluster").
        //
        // The order and the three-operand spelling are nvcc's own for `__cluster_dims__(2,1,1)`:
        // `.maxntid` then `.explicitcluster` then `.reqnctapercluster nx, ny, nz`, all between the
        // parameter list and the body. (`.maxclusterrank` is the one directive that may NOT appear
        // with `.reqnctapercluster`; it does not.) The host passes
        // CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION anyway -- see `gpu::cluster_launch` -- and the
        // CPU-priced `ptxas` census assembles this text before any H100 sees it.
        s += &format!(".explicitcluster\n.reqnctapercluster {ctas}, 1, 1\n");
    }
    s += "{\n";

    // Registers. NB: never name one %tid/%ctaid/%laneid etc -- those are PTX special registers.
    s += "    .reg .pred %p0,%p1,%p2,%q0,%q1,%q2,%q3,%pd0,%pd1,%ptrue,%pfirst;\n";
    s += "    .reg .b32 %M,%N,%K,%lin,%wgi,%lane,%wrp,%kt,%ktiles,%stg,%phf,%phe,%tmp,%tmp2,\
          %row0,%row1,%col,%col1,%colb,%ctam,%ctan,%cwg;\n";
    s += "    .reg .b64 %rdC,%rdS,%rdT,%rdA,%rdB,%rdBar,%rdBarF,%rdTmA,%rdTmB,%rdAddr,%rdOffA,\
          %descA,%descB,%rdSt;\n";
    if clustered {
        // `%crank` is this CTA's rank in its cluster; `%sbar`/`%rbar` are 32-bit SHARED addresses
        // (the width `mapa.shared::cluster` and a `.shared::cluster` mbarrier operand take), and
        // `%cmask` is the 16-bit multicast destination mask.
        s += "    .reg .b32 %crank,%sbar,%rbar;\n";
        s += "    .reg .b16 %cmask;\n";
    }
    s += &format!("    .reg .f32 %acc<{nacc}>;\n\n");

    // --- parameters -------------------------------------------------------------------------------
    s += "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
    s += "    ld.param.u64 %rdC,[pC];\n    cvta.to.global.u64 %rdC,%rdC;\n";
    s += &ptx_param_address("%rdTmA", "tmapA");
    s += &ptx_param_address("%rdTmB", "tmapB");
    s += &format!("    mov.u64 %rdS,{WGMMA_DSMEM_SYM};\n");

    // --- identities -------------------------------------------------------------------------------
    s += "    mov.u32 %lin,%tid.x;\n";
    s += "    shr.u32 %wgi,%lin,7;\n"; // warpgroup index = tid / 128
    s += "    mov.u32 %lane,%laneid;\n";
    if clustered {
        // This CTA's rank in its cluster. Every cluster-relative decision below reads THIS and never
        // `%ctaid.x % ctas`: the two agree for a 1-D cluster along x, and only one of them keeps
        // agreeing when the cluster shape changes.
        s += "    mov.u32 %crank,%cluster_ctarank;\n";
    }
    s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %ctan,%tmp,{bn};\n");
    s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %ctam,%tmp,{bm};\n");
    // ktiles = ceil(K / BK); BK is a power of two (validated), so the divide is a shift.
    s += &format!(
        "    add.u32 %tmp,%K,{};\n    shr.u32 %ktiles,%tmp,{bk_shift};\n",
        bk - 1
    );

    // --- mbarrier initialisation, then one CTA-wide rendezvous ------------------------------------
    // `full[s]` expects a single arrival: the TMA transaction itself, whose byte count the producer
    // declares with `expect_tx`. `empty[s]` expects one arrival per consumer warpgroup.
    s += "    setp.eq.u32 %p0,%lin,0;\n";
    s += &format!("    @!%p0 bra INIT_DONE_{name};\n");
    for st in 0..cfg.stages {
        s += &format!(
            "    add.s64 %rdBar,%rdS,{};\n    mbarrier.init.shared::cta.b64 [%rdBar],1;\n",
            cfg.full_off(st)
        );
        s += &format!(
            "    add.s64 %rdBar,%rdS,{};\n    mbarrier.init.shared::cta.b64 [%rdBar],{};\n",
            cfg.empty_off(st),
            cfg.empty_arrivals()
        );
    }
    if clustered {
        // The mbarriers this CTA just initialised are written by OTHER CTAs (a peer's multicast
        // completes this CTA's `full[s]`; a peer's consumers arrive at this CTA's `empty[s]`), and an
        // `mbarrier.init` is not ordered against a remote access by anything weaker than this. It is
        // the ISA's own fence for exactly this hazard and it is executed by the initialising thread,
        // inside the same guarded region, before the rendezvous below.
        s += "    fence.mbarrier_init.release.cluster;\n";
    }
    s += &format!("INIT_DONE_{name}:\n    bar.sync 0;\n");
    if clustered {
        // `bar.sync` is CTA-scoped, so it says nothing about the peer's barriers being ready. No CTA
        // may touch a peer's shared memory until every CTA in the cluster has passed this point.
        s += "    barrier.cluster.arrive.aligned;\n    barrier.cluster.wait.aligned;\n";
    }

    // --- role split -------------------------------------------------------------------------------
    s += "    setp.eq.u32 %p0,%wgi,0;\n";
    s += &format!("    @!%p0 bra CONSUMER_{name};\n");

    // ================================ producer warpgroup ==========================================
    // One warpgroup, and within it one thread, drives every TMA. The other 127 threads exist only to
    // execute the aligned `setmaxnreg.dec` -- which is the point of the warpgroup: it hands its
    // registers to the consumers and then gets out of the way.
    s += &format!(
        "    setmaxnreg.dec.sync.aligned.u32 {};\n",
        cfg.producer_regs
    );
    s += "    setp.eq.u32 %p0,%lin,0;\n";
    s += &format!("    @!%p0 bra EXIT_{name};\n");
    if clustered {
        // Every CTA of the cluster is a destination of every A multicast, so the mask is the low
        // `ctas` bits and is loop-invariant. Bit `r` names the CTA whose `%cluster_ctarank` is `r`.
        s += &format!("    mov.u16 %cmask,{cmask};\n");
    }
    s += "    mov.u32 %kt,0;\n    mov.u32 %stg,0;\n";
    // The empty-phase parity starts at 1 so the first `stages` acquisitions pass immediately: a
    // freshly initialised barrier is in phase parity 0, and `try_wait.parity 1` completes at once
    // when the current parity is 0. Without this the producer would wait for a release that the
    // consumers cannot yet have made, and the pipeline would never start.
    s += "    mov.u32 %phe,1;\n";
    s += &format!("PLOOP_{name}:\n");
    s += &format!("    setp.ge.u32 %p0,%kt,%ktiles;\n    @%p0 bra EXIT_{name};\n");
    // acquire: wait until stage `stg` is free
    s += "    mul.wide.u32 %rdT,%stg,8;\n    add.s64 %rdBar,%rdS,%rdT;\n";
    s += &format!("    add.s64 %rdBar,%rdBar,{empty_base};\n");
    s += &format!("PWAIT_{name}:\n");
    s += "    mbarrier.try_wait.parity.shared::cta.b64 %p1,[%rdBar],%phe;\n";
    s += &format!("    @!%p1 bra PWAIT_{name};\n");
    // declare the transaction on the full barrier, then issue both tile copies against it
    s += "    mul.wide.u32 %rdT,%stg,8;\n    add.s64 %rdBarF,%rdS,%rdT;\n";
    s += &format!("    add.s64 %rdBarF,%rdBarF,{full_base};\n");
    s += &format!(
        "    mbarrier.arrive.expect_tx.shared::cta.b64 %rdSt,[%rdBarF],{};\n",
        cfg.stage_tx_bytes()
    );
    s += &format!("    mul.wide.u32 %rdT,%stg,{tile_a};\n    add.s64 %rdA,%rdS,%rdT;\n");
    if clustered {
        // This rank's slice of the shared A tile, at the SAME CTA-relative offset in every
        // destination -- which is what makes the assembled SMEM image identical to the one an
        // unclustered copy of the whole tile would have written, and therefore leaves the
        // descriptor, the per-consumer m64 slabs and the epilogue untouched.
        s += &format!("    mul.lo.s32 %tmp,%crank,{a_slice};\n    cvt.u64.u32 %rdOffA,%tmp;\n");
        s += "    add.s64 %rdA,%rdA,%rdOffA;\n";
    }
    s += &format!("    mul.wide.u32 %rdT,%stg,{tile_b};\n    add.s64 %rdB,%rdS,%rdT;\n");
    s += &format!("    add.s64 %rdB,%rdB,{};\n", cfg.b_off(0));
    // The K coordinate, in elements. Tensor coordinates are `{dim0, dim1}` = `{k, row}`, because
    // dimension 0 is the contiguous axis of the descriptor (see `tma_host`).
    s += &format!("    mul.lo.s32 %tmp,%kt,{bk};\n");
    if clustered {
        // ...and the global row this rank's slice starts at. The A tensor map's box is
        // `a_box_rows` tall (WgmmaCfg::tensor_map_a), so the slices tile the CTA's M range exactly.
        s +=
            &format!("    mul.lo.s32 %tmp2,%crank,{a_box_rows};\n    add.u32 %tmp2,%tmp2,%ctam;\n");
        // A, multicast to the whole cluster. `ctaMask` is the last operand; the hardware writes the
        // slice into every destination CTA at the same CTA-relative offset as `%rdA` AND signals the
        // barrier at the same CTA-relative offset as `%rdBarF` in each of them -- which is why every
        // CTA declares the FULL `stage_tx_bytes` and none of them divides it.
        s += "    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes\
              .multicast::cluster [%rdA],[%rdTmA,{%tmp,%tmp2}],[%rdBarF],%cmask;\n";
    } else {
        s +=
            "    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes \
              [%rdA],[%rdTmA,{%tmp,%ctam}],[%rdBarF];\n";
    }
    // B is never multicast: the CTAs of a cluster hold DIFFERENT N halves, so they share no B bytes.
    s += "    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes \
          [%rdB],[%rdTmB,{%tmp,%ctan}],[%rdBarF];\n";
    s += "    add.u32 %kt,%kt,1;\n    add.u32 %stg,%stg,1;\n";
    s += &format!(
        "    setp.lt.u32 %p1,%stg,{};\n    @%p1 bra PLOOP_{name};\n",
        cfg.stages
    );
    s += &format!("    mov.u32 %stg,0;\n    xor.b32 %phe,%phe,1;\n    bra PLOOP_{name};\n");

    // ================================ consumer warpgroups =========================================
    s += &format!("CONSUMER_{name}:\n");
    s += &format!(
        "    setmaxnreg.inc.sync.aligned.u32 {};\n",
        cfg.consumer_regs
    );
    s += "    sub.u32 %cwg,%wgi,1;\n";
    s += &format!("    mul.lo.s32 %tmp,%cwg,{per_consumer_a};\n    cvt.u64.u32 %rdOffA,%tmp;\n");
    s += "    mov.u32 %kt,0;\n    mov.u32 %stg,0;\n    mov.u32 %phf,0;\n";
    s += "    setp.eq.u32 %ptrue,0,0;\n";
    // No accumulator zeroing: the first wgmma of the first K tile takes scale-d = 0, which overwrites
    // D instead of accumulating into it. That is the ISA's own way to skip the initialisation.
    s += &format!("CLOOP_{name}:\n");
    s += &format!("    setp.ge.u32 %p0,%kt,%ktiles;\n    @%p0 bra CEND_{name};\n");
    s += "    mul.wide.u32 %rdT,%stg,8;\n    add.s64 %rdBar,%rdS,%rdT;\n";
    s += &format!("    add.s64 %rdBar,%rdBar,{full_base};\n");
    s += &format!("CWAIT_{name}:\n");
    s += "    mbarrier.try_wait.parity.shared::cta.b64 %p1,[%rdBar],%phf;\n";
    s += &format!("    @!%p1 bra CWAIT_{name};\n");
    // operand bases for this stage
    s += &format!("    mul.wide.u32 %rdT,%stg,{tile_a};\n    add.s64 %rdA,%rdS,%rdT;\n");
    s += "    add.s64 %rdA,%rdA,%rdOffA;\n";
    s += &format!("    mul.wide.u32 %rdT,%stg,{tile_b};\n    add.s64 %rdB,%rdS,%rdT;\n");
    s += &format!("    add.s64 %rdB,%rdB,{};\n", cfg.b_off(0));
    s += "    setp.ne.u32 %pfirst,%kt,0;\n";
    // `wgmma.fence` orders the accumulator registers against the async proxy before the group.
    s += "    wgmma.fence.sync.aligned;\n";
    for j in 0..cfg.wgmma_per_stage() {
        for (reg, base, cst, step) in [
            ("%descA", "%rdA", const_a, a_step),
            ("%descB", "%rdB", const_b, b_step),
        ] {
            let koff = j as u64 * step;
            s += &format!("    add.s64 %rdAddr,{base},{koff};\n");
            s += "    shr.u64 %rdT,%rdAddr,4;\n    and.b64 %rdT,%rdT,16383;\n";
            s += &format!("    or.b64 {reg},%rdT,{cst:#x};\n");
        }
        // scale-d: 0 only for the very first wgmma of the whole K loop, so D is overwritten there and
        // accumulated into everywhere else.
        let scale_d = if j == 0 { "%pfirst" } else { "%ptrue" };
        s += &format!("    {mma} {{{accs}}}, %descA, %descB, {scale_d}, 1, 1, 0, 0;\n");
    }
    s += "    wgmma.commit_group.sync.aligned;\n    wgmma.wait_group.sync.aligned 0;\n";
    // release the stage: one arrival per consumer warpgroup, from its first thread. `wait_group` is
    // warpgroup-aligned, so every thread of this warpgroup is done with the buffer by now.
    s += "    mul.wide.u32 %rdT,%stg,8;\n    add.s64 %rdBar,%rdS,%rdT;\n";
    s += &format!("    add.s64 %rdBar,%rdBar,{empty_base};\n");
    s += "    and.b32 %tmp,%lin,127;\n    setp.eq.u32 %p2,%tmp,0;\n";
    if clustered {
        // This warpgroup releases the stage in EVERY CTA of the cluster, not just its own: the A
        // slice it just finished reading was multicast in by a peer's producer, and that producer
        // may not overwrite it until every consumer in the cluster is done. `empty[s]` is therefore
        // initialised with `cluster_ctas * consumer_wgs` arrivals (WgmmaCfg::empty_arrivals) and gets
        // one from every consumer warpgroup in the cluster.
        //
        // `mapa` takes a 32-bit SHARED address, not the 64-bit generic one the rest of this mainloop
        // computes, and an mbarrier arrival at `.shared::cluster` scope cannot return a state token
        // (hence the `_` sink) -- both are the CUTLASS idiom verbatim.
        s += "    cvta.to.shared.u64 %rdT,%rdBar;\n    cvt.u32.u64 %sbar,%rdT;\n";
        for r in 0..ctas {
            s += &format!("    mov.u32 %tmp2,{r};\n");
            s += "    mapa.shared::cluster.u32 %rbar,%sbar,%tmp2;\n";
            s += "    @%p2 mbarrier.arrive.shared::cluster.b64 _,[%rbar];\n";
        }
    } else {
        s += "    @%p2 mbarrier.arrive.shared::cta.b64 %rdSt,[%rdBar];\n";
    }
    s += "    add.u32 %kt,%kt,1;\n    add.u32 %stg,%stg,1;\n";
    s += &format!(
        "    setp.lt.u32 %p1,%stg,{};\n    @%p1 bra CLOOP_{name};\n",
        cfg.stages
    );
    s += &format!("    mov.u32 %stg,0;\n    xor.b32 %phf,%phf,1;\n    bra CLOOP_{name};\n");

    // --- epilogue ---------------------------------------------------------------------------------
    // Accumulator layout, per the ISA and identical to `mma.sync.m16n8k16`'s C fragment tiled over 4
    // warps and N/8 column blocks: with `grp = lane/4` and `tg = lane%4`, warp `w` of the warpgroup
    // holds rows `w*16 + grp` and `w*16 + grp + 8`; register group `j` covers columns
    // `8j + 2*tg` and `8j + 2*tg + 1`, in the order (row0,c) (row0,c+1) (row1,c) (row1,c+1).
    s += &format!("CEND_{name}:\n");
    // K == 0 means no wgmma ran, so the accumulators were never written -- `scale-d` skips the
    // zero-init precisely by folding it into the first issue, and with zero issues there is no
    // first. Storing here would publish uninitialised registers, so the epilogue is skipped and C is
    // left untouched. (A caller that wants `C = 0` for an empty K must zero it itself; this kernel
    // will not invent a value it never computed.)
    s += &format!("    setp.eq.u32 %p0,%ktiles,0;\n    @%p0 bra EXIT_{name};\n");
    s += "    wgmma.wait_group.sync.aligned 0;\n";
    s += "    and.b32 %tmp,%lin,127;\n    shr.u32 %wrp,%tmp,5;\n    shl.b32 %wrp,%wrp,4;\n";
    s += "    shr.u32 %tmp2,%lane,2;\n    add.u32 %row0,%wrp,%tmp2;\n";
    s += "    mul.lo.s32 %tmp,%cwg,64;\n    add.u32 %row0,%row0,%tmp;\n";
    s += "    add.u32 %row0,%row0,%ctam;\n    add.u32 %row1,%row0,8;\n";
    s += "    and.b32 %colb,%lane,3;\n    shl.b32 %colb,%colb,1;\n    add.u32 %colb,%colb,%ctan;\n";
    s += "    setp.lt.u32 %pd0,%row0,%M;\n    setp.lt.u32 %pd1,%row1,%M;\n";
    s += "    mad.lo.s32 %tmp,%row0,%N,%colb;\n    mul.wide.u32 %rdT,%tmp,4;\n    add.s64 %rdA,%rdC,%rdT;\n";
    s += "    mad.lo.s32 %tmp,%row1,%N,%colb;\n    mul.wide.u32 %rdT,%tmp,4;\n    add.s64 %rdB,%rdC,%rdT;\n";
    for j in 0..bn / 8 {
        let byte = j * 32;
        s += &format!("    add.u32 %col,%colb,{};\n", j * 8);
        s += "    setp.lt.u32 %p0,%col,%N;\n    add.u32 %col1,%col,1;\n    setp.lt.u32 %p1,%col1,%N;\n";
        s += "    and.pred %q0,%pd0,%p0;\n    and.pred %q1,%pd0,%p1;\n";
        s += "    and.pred %q2,%pd1,%p0;\n    and.pred %q3,%pd1,%p1;\n";
        s += &format!("    @%q0 st.global.f32 [%rdA+{byte}],%acc{};\n", 4 * j);
        s += &format!(
            "    @%q1 st.global.f32 [%rdA+{}],%acc{};\n",
            byte + 4,
            4 * j + 1
        );
        s += &format!("    @%q2 st.global.f32 [%rdB+{byte}],%acc{};\n", 4 * j + 2);
        s += &format!(
            "    @%q3 st.global.f32 [%rdB+{}],%acc{};\n",
            byte + 4,
            4 * j + 3
        );
    }
    if clustered {
        // **No CTA may retire while a peer can still touch its shared memory.** A peer's producer
        // multicasts into this CTA's ring and a peer's consumers arrive at this CTA's `empty`
        // barriers; a CTA that exited has no shared memory to write. Every thread of every CTA
        // reaches this label -- the producer's 127 idle threads immediately, its issuing thread when
        // the K loop ends, the consumers after the epilogue (or straight from `ktiles == 0`) -- so
        // the `.aligned` claim holds and the wait cannot be the deadlock.
        s += &format!(
            "EXIT_{name}:\n    barrier.cluster.arrive.aligned;\n    barrier.cluster.wait.aligned;\n\
             \x20   ret;\n}}\n"
        );
    } else {
        s += &format!("EXIT_{name}:\n    ret;\n}}\n");
    }
    Ok(s)
}

/// **Every PTX module this family emits, device-free** -- the corpus its own `.version`/`.target`
/// and ASCII laws scan.
///
/// It is shaped as a `Vec<(String, String)>` on purpose: that is the exact shape of
/// `gpu.rs`'s `device_free_modules()`, which splices this family into the crate-wide `.version` law
/// with one line (`v.extend(crate::ptx_wgmma::wgmma_device_free_modules());`) and one number
/// (`EXPECTED_MODULES`). `the_family_declares_no_floor_it_does_not_need` below applies the identical
/// rule over the identical corpus.
///
/// **The two bring-up probes are in it too**, deliberately: the TMA stage probe and the descriptor
/// sweep probe are text this backend will hand to `cuModuleLoadData` on rented silicon, so the ASCII
/// rule, the `sm_90a` floor and the `.version` law must reach them exactly as they reach a shipped
/// row. A module that only bring-up loads is still a module that can be one stray `->` away from a
/// `ptxas fatal` at the worst possible moment.
pub fn wgmma_device_free_modules() -> Vec<(String, String)> {
    let license = Sm90aLicense::for_probed_cc((9, 0)).expect("(9,0) is Hopper");
    let mut v: Vec<(String, String)> = wgmma_all_emittable()
        .into_iter()
        .map(|c| {
            let ptx = wgmma_module(c, &license)
                .unwrap_or_else(|e| panic!("emittable variant {} must generate: {e}", c.name));
            (format!("wgmma::{}", c.name), ptx)
        })
        .collect();
    v.push((
        format!("wgmma::bringup/{TMA_PROBE_ENTRY}"),
        tma_stage_probe_module(&license).expect("the TMA stage probe must generate"),
    ));
    v.push((
        format!("wgmma::bringup/{DESC_SWEEP_ENTRY}"),
        desc_sweep_probe_module(&license).expect("the descriptor sweep probe must generate"),
    ));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn license() -> Sm90aLicense {
        Sm90aLicense::for_probed_cc((9, 0)).unwrap()
    }

    // --- the capability gate ----------------------------------------------------------------------

    /// **`sm_90a` is a lock, not a floor.** Every other capability in this crate is `cc >= min`
    /// because PTX is forward compatible; an `sm_90a` module loads on Hopper and on nothing else, so
    /// the gate rejects in *both* directions and says which direction it rejected in.
    #[test]
    fn the_license_is_hopper_only_in_both_directions() {
        assert_eq!(license().cc(), (9, 0));
        for cc in [(7, 0), (8, 0), (8, 6), (8, 9)] {
            let e = Sm90aLicense::for_probed_cc(cc).unwrap_err();
            assert!(e.contains("requires cc>=9.0"), "{cc:?}: {e}");
        }
        for cc in [(10, 0), (12, 0)] {
            let e = Sm90aLicense::for_probed_cc(cc).unwrap_err();
            assert!(
                e.contains("architecture-locked"),
                "a post-Hopper part must be told the module cannot load, not that it is too old: \
                 {cc:?}: {e}"
            );
        }
        // Hopper's minor revisions are all cc 9.x.
        assert!(Sm90aLicense::for_probed_cc((9, 1)).is_ok());
    }

    /// **The textual half of the capability law.** The structural half is the type: [`wgmma_module`]
    /// and [`tma_stage_probe_module`] take an `&Sm90aLicense`, so ungated `sm_90a` PTX does not
    /// compile. This scan closes the one remaining hole -- a *new* function in this file that builds
    /// a header string itself instead of going through [`sm90a_header`].
    ///
    /// The law is stated over the funnel rather than over one named generator, because the family now
    /// emits two module shapes and will emit more. Two clauses: [`HDR_SM90A_V80`] is named nowhere
    /// but the `use` and [`sm90a_header`]'s body, and **every** function that calls `sm90a_header`
    /// carries the witness in its own signature.
    #[test]
    fn every_sm90a_emitter_demands_the_license() {
        let src = include_str!("ptx_wgmma.rs");
        let code = src
            .split("\n#[cfg(test)]")
            .next()
            .expect("the test module is cut off at its column-0 attribute");
        // Prose is not code: a doc comment naming `fn ` or the header constant must not satisfy or
        // trip the scan.
        let code: String = code
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        let uses: Vec<&str> = code
            .lines()
            .filter(|l| l.contains("HDR_SM90A_V80"))
            .collect();
        assert_eq!(
            uses.len(),
            2,
            "expected exactly the `use` and the single site inside `sm90a_header`; found {uses:#?}"
        );
        let funnel = code
            .split_once("fn sm90a_header(")
            .expect("the single header site must exist")
            .1;
        assert!(
            funnel.split_once(')').unwrap().0.contains("Sm90aLicense"),
            "the header funnel itself must demand the witness"
        );
        // Every function whose body reaches the funnel takes the license. `fn ` splits cleanly here:
        // comments are gone and no signature in this file contains a brace.
        let mut emitters: Vec<String> = Vec::new();
        for chunk in code.split("fn ").skip(1) {
            let (sig, after) = match chunk.split_once('{') {
                Some(p) => p,
                None => continue,
            };
            let body = after.split("\n}").next().unwrap_or(after);
            if body.contains("sm90a_header(") {
                let name = sig.split('(').next().unwrap_or(sig).trim().to_string();
                assert!(
                    sig.contains("Sm90aLicense"),
                    "`{name}` emits an sm_90a header without a capability witness in its signature: \
                     {sig}"
                );
                emitters.push(name);
            }
        }
        emitters.sort();
        assert_eq!(
            emitters,
            vec![
                "desc_sweep_probe_module".to_string(),
                "tma_stage_probe_module".to_string(),
                "wgmma_module".to_string()
            ],
            "the set of sm_90a emitters changed -- add the new one deliberately"
        );
        // And nothing else in the file may spell the target directly.
        assert!(
            !code.contains("sm_90a\\n"),
            "no hand-rolled sm_90a header -- go through ptx_target::HDR_SM90A_V80"
        );
    }

    // --- the shared-memory descriptor: the device-free proof --------------------------------------

    /// **Every field, at its documented bit position, against hand-computed values.**
    ///
    /// The packing is `(x & 0x3FFFF) >> 4` per address/offset field, placed at bits 13:0, 29:16 and
    /// 45:32, with a 3-bit base offset at 51:49 and the 2-bit swizzle mode at 63:62. Get any of it
    /// wrong and `wgmma` reads a different shared address than the kernel computed, silently.
    #[test]
    fn smem_desc_packs_every_field_at_its_documented_bit_position() {
        // One field at a time, everything else zero.
        let f = |d: SmemDesc| d.pack().unwrap();
        let base = SmemDesc {
            start_addr: 0,
            lbo: 0,
            sbo: 0,
            base_offset: 0,
            swizzle: SmemSwizzle::None,
        };
        assert_eq!(f(base), 0);

        // start address at bits 13:0, encoded /16.
        assert_eq!(
            f(SmemDesc {
                start_addr: 16,
                ..base
            }),
            1
        );
        assert_eq!(
            f(SmemDesc {
                start_addr: 0x3FFF0,
                ..base
            }),
            0x3FFF,
            "the widest encodable address, 262128 B, fills the 14-bit field exactly"
        );

        // leading byte offset at bits 29:16.
        assert_eq!(f(SmemDesc { lbo: 16, ..base }), 1 << 16);
        assert_eq!(f(SmemDesc { lbo: 1024, ..base }), 64 << 16);
        assert_eq!(
            f(SmemDesc {
                lbo: 0x3FFF0,
                ..base
            }),
            0x3FFF << 16
        );

        // stride byte offset at bits 45:32.
        assert_eq!(f(SmemDesc { sbo: 16, ..base }), 1u64 << 32);
        assert_eq!(f(SmemDesc { sbo: 1024, ..base }), 64u64 << 32);
        assert_eq!(
            f(SmemDesc {
                sbo: 0x3FFF0,
                ..base
            }),
            0x3FFFu64 << 32
        );

        // 3-bit base offset at bits 51:49.
        for b in 0u8..8 {
            assert_eq!(
                f(SmemDesc {
                    base_offset: b,
                    ..base
                }),
                (b as u64) << 49
            );
        }

        // 2-bit swizzle mode at bits 63:62 -- all four modes, each at its own start alignment.
        for (sw, bits, addr) in [
            (SmemSwizzle::None, 0u64, 0u64),
            (SmemSwizzle::B128, 1, 1024),
            (SmemSwizzle::B64, 2, 512),
            (SmemSwizzle::B32, 3, 256),
        ] {
            let d = SmemDesc {
                start_addr: addr,
                swizzle: sw,
                ..base
            };
            assert_eq!(d.pack().unwrap(), (bits << 62) | (addr >> 4), "{sw:?}");
        }

        // All fields at once, each at the widest value its own rules allow: nothing overlaps and
        // nothing carries into a neighbour. (The start address is 0x3FF00 rather than 0x3FFF0
        // because the 32-B mode needs a 256-byte boundary and 0x3FFF0 is not one.)
        let all = SmemDesc {
            start_addr: 0x3FF00,
            lbo: 0x3FFF0,
            sbo: 0x3FFF0,
            base_offset: 7,
            swizzle: SmemSwizzle::B32,
        };
        let packed = all.pack().unwrap();
        assert_eq!(
            packed,
            0x3FF0 | (0x3FFF << 16) | (0x3FFFu64 << 32) | (7u64 << 49) | (3u64 << 62)
        );
        // ...and every reserved bit stays zero: 15:14, 31:30, 48:46 and 61:52 are gaps between the
        // fields, so a field one bit too wide would show up here first.
        for b in [14, 15, 30, 31, 46, 47, 48].into_iter().chain(52..62) {
            assert_eq!((packed >> b) & 1, 0, "reserved bit {b} is set");
        }
    }

    #[test]
    fn the_field_encoder_is_mask_then_shift() {
        assert_eq!(encode_desc_field(0), 0);
        assert_eq!(encode_desc_field(16), 1);
        assert_eq!(encode_desc_field(1024), 64);
        // The 18-bit mask wraps before the shift: 0x40000 (262144) encodes as 0, not 0x4000.
        assert_eq!(encode_desc_field(0x40000), 0);
        assert_eq!(encode_desc_field(0x40010), 1);
        // ...which is exactly why `pack` refuses anything at or above 2^18 instead of encoding it.
        assert!(SmemDesc {
            start_addr: 0x40000,
            lbo: 16,
            sbo: 16,
            base_offset: 0,
            swizzle: SmemSwizzle::None,
        }
        .pack()
        .unwrap_err()
        .contains("does not fit the 14-bit field"));
    }

    /// The `>> 4` is silent truncation, so a byte quantity that is not a multiple of 16 must be
    /// rejected rather than rounded. This is the arithmetic that hides a wrong address.
    #[test]
    fn pack_rejects_anything_the_shift_would_truncate() {
        let d = |start, lbo, sbo| SmemDesc {
            start_addr: start,
            lbo,
            sbo,
            base_offset: 0,
            swizzle: SmemSwizzle::None,
        };
        assert!(d(8, 16, 16).pack().unwrap_err().contains("start address"));
        assert!(d(16, 24, 16)
            .pack()
            .unwrap_err()
            .contains("leading byte offset"));
        assert!(d(16, 16, 4)
            .pack()
            .unwrap_err()
            .contains("stride byte offset"));
        assert!(SmemDesc {
            base_offset: 8,
            ..d(16, 16, 16)
        }
        .pack()
        .unwrap_err()
        .contains("base offset"));
    }

    /// **The 128-B swizzle's repeating pattern must begin on a 1024-byte boundary.** A matrix placed
    /// at any other address is not describable in that mode, and the descriptor cannot say so -- the
    /// bits pack fine. So the alignment is a precondition of `pack`, per mode.
    #[test]
    fn the_swizzle_modes_constrain_the_start_address() {
        assert_eq!(SmemSwizzle::None.required_alignment(), 16);
        assert_eq!(SmemSwizzle::B128.required_alignment(), 1024);
        assert_eq!(SmemSwizzle::B64.required_alignment(), 512);
        assert_eq!(SmemSwizzle::B32.required_alignment(), 256);
        let at = |addr, sw| {
            SmemDesc {
                start_addr: addr,
                lbo: 16,
                sbo: 1024,
                base_offset: 0,
                swizzle: sw,
            }
            .pack()
        };
        at(1024, SmemSwizzle::B128).unwrap();
        at(2048, SmemSwizzle::B128).unwrap();
        let e = at(512, SmemSwizzle::B128).unwrap_err();
        assert!(e.contains("1024-byte boundary"), "{e}");
        at(512, SmemSwizzle::B64).unwrap();
        assert!(at(256, SmemSwizzle::B64).is_err());
        at(256, SmemSwizzle::B32).unwrap();
        assert!(at(16, SmemSwizzle::None).is_ok());
    }

    /// **The two swizzle encodings run backwards from each other.** `CUtensorMapSwizzle` counts up
    /// 32/64/128; the wgmma descriptor counts down 128/64/32. A plain `as u32` between them swaps the
    /// narrowest and widest modes, and the result is a TMA copy that writes one pattern and a `wgmma`
    /// that reads another -- correct-looking, silently wrong.
    #[test]
    fn the_descriptor_swizzle_encoding_is_reversed_from_tmas() {
        assert_eq!(SmemSwizzle::None as u64, 0);
        assert_eq!(SmemSwizzle::B128 as u64, 1);
        assert_eq!(SmemSwizzle::B64 as u64, 2);
        assert_eq!(SmemSwizzle::B32 as u64, 3);
        assert_eq!(TmaSwizzle::B32 as u32, 1);
        assert_eq!(TmaSwizzle::B128 as u32, 3);
        // The naive cast is wrong in both directions; `from_tma`/`to_tma` are not.
        assert_ne!(SmemSwizzle::B128 as u64, TmaSwizzle::B128 as u64);
        for t in [
            TmaSwizzle::None,
            TmaSwizzle::B32,
            TmaSwizzle::B64,
            TmaSwizzle::B128,
        ] {
            assert_eq!(SmemSwizzle::from_tma(t).to_tma(), t);
        }
    }

    /// **[`desc_fields`] is the authority, and here is every reading it spells**, at the geometry
    /// every shipped row uses (BK = 64 f16, so a 128-byte shared row, and 64 rows per descriptor).
    #[test]
    fn desc_fields_spells_every_reading_from_one_table() {
        let (rb, rows) = (128u64, 64u64);
        // The two readings the H100 already scored: k-adjacent 16 B, row-group-adjacent 8*128.
        let k = desc_fields(SmemLayout::RowMajorNone { k_leading: true }, rb, rows);
        assert_eq!((k.lbo, k.sbo, k.k_step_bytes), (16, 1024, 32));
        let m = desc_fields(SmemLayout::RowMajorNone { k_leading: false }, rb, rows);
        assert_eq!((m.lbo, m.sbo, m.k_step_bytes), (1024, 16, 32));
        // The canonical no-swizzle packings. A core matrix is 128 B; K-fastest puts k-adjacent ones
        // 128 B apart and a whole row of them (BK/8 = 8) between MN-adjacent ones.
        let kf = desc_fields(
            SmemLayout::CanonicalNone {
                k_fast: true,
                swapped: false,
            },
            rb,
            rows,
        );
        assert_eq!((kf.lbo, kf.sbo, kf.k_step_bytes), (128, 1024, 256));
        let kfs = desc_fields(
            SmemLayout::CanonicalNone {
                k_fast: true,
                swapped: true,
            },
            rb,
            rows,
        );
        assert_eq!((kfs.lbo, kfs.sbo), (1024, 128));
        let mf = desc_fields(
            SmemLayout::CanonicalNone {
                k_fast: false,
                swapped: false,
            },
            rb,
            rows,
        );
        assert_eq!((mf.lbo, mf.sbo, mf.k_step_bytes), (1024, 128, 2048));
        // The shipped 128-B-swizzle reading: SBO = one 8-row atom, and the mode's own bits.
        let sw = desc_fields(SHIPPED_LAYOUT, rb, rows);
        assert_eq!((sw.lbo, sw.sbo, sw.k_step_bytes), (16, 1024, 32));
        assert_eq!(sw.swizzle, SmemSwizzle::B128);
        let swx = desc_fields(
            SmemLayout::Swizzle128 {
                lbo_bytes: 16,
                swapped: true,
            },
            rb,
            rows,
        );
        assert_eq!((swx.lbo, swx.sbo), (1024, 16), "the axis-naming flip");
        assert_eq!(SHIPPED_LAYOUT.tma_swizzle(), TmaSwizzle::B128);
        assert!(SHIPPED_LAYOUT.tma_writable());
        assert!(!SmemLayout::CanonicalNone {
            k_fast: true,
            swapped: false
        }
        .tma_writable());
        // Every reading packs, and the descriptor's swizzle bits are the reversed encoding of the
        // TMA mode -- one choice, two tables, never cast between them.
        for l in [
            SmemLayout::RowMajorNone { k_leading: true },
            SmemLayout::CanonicalNone {
                k_fast: true,
                swapped: false,
            },
            SHIPPED_LAYOUT,
        ] {
            let d = SmemDesc::for_layout(0, l, rb, rows);
            d.const_part()
                .unwrap_or_else(|e| panic!("{}: {e}", l.label()));
            assert_eq!(d.swizzle, SmemSwizzle::from_tma(l.tma_swizzle()));
            assert!(l.label().is_ascii());
        }
        // Hand-computed: the shipped template is lbo 16 -> 1 at bit 16, sbo 1024 -> 64 at bit 32,
        // and the 128-B swizzle's own `1` at bit 62.
        assert_eq!(
            SmemDesc::for_layout(0, SHIPPED_LAYOUT, rb, rows)
                .const_part()
                .unwrap(),
            (1u64 << 16) | (64u64 << 32) | (1u64 << 62)
        );
    }

    /// The kernel folds the runtime stage address into the descriptor with
    /// `((addr >> 4) & 0x3FFF) | const_part`. That must equal what the host encoder would produce for
    /// the same address -- for every address a 227 KiB shared window can hold.
    #[test]
    fn the_runtime_address_fold_matches_the_host_encoder() {
        let layout = SmemLayout::RowMajorNone { k_leading: true };
        let cst = SmemDesc::for_layout(0, layout, 128, 64)
            .const_part()
            .unwrap();
        let mut addr = 0u64;
        while addr < HOPPER_SMEM_PER_CTA as u64 {
            let host = SmemDesc::for_layout(addr, layout, 128, 64).pack().unwrap();
            let device = ((addr >> 4) & 0x3FFF) | cst;
            assert_eq!(host, device, "address {addr:#x}");
            addr += 16;
        }
        // And the 18-bit mask never fires below 256 KiB, which is why the device fold can skip it.
        const _: () = assert!(HOPPER_SMEM_PER_CTA < (1 << 18));
    }

    /// **The three field encodings, and why two of them are candidates rather than code paths.**
    #[test]
    fn the_field_encodings_are_three_distinguishable_templates() {
        let d = SmemDesc {
            start_addr: 0,
            lbo: 128,
            sbo: 1024,
            base_offset: 0,
            swizzle: SmemSwizzle::None,
        };
        assert_eq!(
            d.const_part_as(FieldEncoding::Standard).unwrap(),
            (8u64 << 16) | (64u64 << 32)
        );
        assert_eq!(
            d.const_part_as(FieldEncoding::Raw).unwrap(),
            (128u64 << 16) | (1024u64 << 32)
        );
        // Twice: 128 >> 8 == 0, 1024 >> 8 == 4. Degenerate, and deliberately in the set anyway.
        assert_eq!(d.const_part_as(FieldEncoding::Twice).unwrap(), 4u64 << 32);
        assert_eq!(
            d.const_part_as(FieldEncoding::Standard).unwrap(),
            d.const_part().unwrap()
        );
        // Raw is where a wide offset stops fitting the 14-bit field, and it says so rather than
        // silently truncating into the neighbouring one.
        let wide = SmemDesc { sbo: 1 << 17, ..d };
        assert!(wide
            .const_part_as(FieldEncoding::Raw)
            .unwrap_err()
            .contains("14-bit field"));
        wide.const_part_as(FieldEncoding::Standard).unwrap();
    }

    // --- shape menu, warpgroups, budgets ----------------------------------------------------------

    /// M is 64 and K is 16, always. N is every multiple of 8 from 8 to 256, and nothing else.
    #[test]
    fn the_shape_menu_is_the_isa_menu() {
        assert_eq!(WgmmaShape::M, 64);
        assert_eq!(WgmmaShape::K, 16);
        let mut n = WgmmaShape::N_MIN;
        let mut count = 0;
        while n <= WgmmaShape::N_MAX {
            let s = WgmmaShape::new(n).unwrap_or_else(|e| panic!("N={n}: {e}"));
            assert_eq!(s.token(), format!("m64n{n}k16"));
            assert_eq!(s.accum_regs(), n / 2);
            n += WgmmaShape::N_STEP;
            count += 1;
        }
        assert_eq!(count, 32, "the menu has 32 entries: 8, 16, ..., 256");
        for bad in [0usize, 4, 12, 100, 264, 512] {
            let e = WgmmaShape::new(bad).unwrap_err();
            assert!(e.starts_with(UNSUPPORTED), "N={bad} must decline: {e}");
        }
    }

    /// A warpgroup is 4 warps. The CTA is one producer plus `consumer_wgs` consumers, and every
    /// consumer owns exactly one `m64` slice -- which is what makes CTA-M `64 * consumer_wgs` rather
    /// than a free parameter.
    #[test]
    fn warpgroup_arithmetic_is_the_isa_definition() {
        assert_eq!(WARPGROUP_THREADS, 4 * 32);
        for c in WGMMA_VARIANTS {
            assert_eq!(c.threads() % WARPGROUP_THREADS, 0);
            assert_eq!(c.threads() / WARPGROUP_THREADS, 1 + c.consumer_wgs);
            assert_eq!(c.bm, WgmmaShape::M * c.consumer_wgs);
            assert!(c.threads() <= 1024, "{}: {} threads", c.name, c.threads());
            // The first warp of every warpgroup has a warp-rank that is a multiple of 4, because the
            // split is on `tid / 128`.
            for wg in 0..(1 + c.consumer_wgs) {
                assert_eq!((wg * WARPGROUP_THREADS / 32) % 4, 0);
            }
        }
        assert_eq!(WGMMA_W1.threads(), 384);
    }

    /// The stage/tile lattice, against D1 sections 2.2 and 3.1's own arithmetic.
    #[test]
    fn the_stage_lattice_matches_the_derivation() {
        // D1 section 3.1: 128x256x64 f16 is 16 384 B of A + 32 768 B of B per stage; CUTLASS adds
        // 16 B of pipeline state and gets 49 168, then 4 stages inside 227 KiB.
        assert_eq!(WGMMA_W1.tile_a_bytes(), 16384);
        assert_eq!(WGMMA_W1.tile_b_bytes(), 32768);
        assert_eq!(WGMMA_W1.stage_tx_bytes(), 49152);
        assert_eq!(WGMMA_W1.stages, 4);
        assert_eq!(WGMMA_W1.smem_bytes(), 4 * 49152 + 2 * 4 * 8);
        assert_eq!(WGMMA_W1.smem_bytes(), 196672);
        // 128x128x64 is 32 768 B per stage; 6 stages is 192 KiB and 7 would still fit, so 6 leaves
        // room for a TMA epilogue carveout without changing the row.
        assert_eq!(WGMMA_W3C.stage_tx_bytes(), 32768);
        assert_eq!(WGMMA_W3C.smem_bytes(), 6 * 32768 + 2 * 6 * 8);
        const _: () = assert!(7 * 32768 + 2 * 7 * 8 <= HOPPER_SMEM_PER_CTA);
        // Every shipped row fits, and one more stage of the widest row would not.
        for c in WGMMA_VARIANTS {
            assert!(
                c.smem_bytes() <= HOPPER_SMEM_PER_CTA,
                "{}: {} B > {HOPPER_SMEM_PER_CTA}",
                c.name,
                c.smem_bytes()
            );
        }
        let too_deep = WgmmaCfg {
            stages: 5,
            ..WGMMA_W1
        };
        assert!(too_deep.smem_bytes() > HOPPER_SMEM_PER_CTA);
        assert!(too_deep
            .validate()
            .unwrap_err()
            .contains("exceeds the 232448 B per-CTA ceiling"));
    }

    /// Stage bases must land on the alignment the descriptor's swizzle mode demands, or the operand
    /// at stage 3 is describable and the one at stage 4 is not.
    #[test]
    fn every_stage_base_is_aligned_for_its_descriptor() {
        for c in WGMMA_VARIANTS {
            let align = c.swizzle().required_alignment() as usize;
            for s in 0..c.stages {
                assert_eq!(c.a_off(s) % align, 0, "{} A stage {s}", c.name);
                assert_eq!(c.b_off(s) % align, 0, "{} B stage {s}", c.name);
                // ...and every per-consumer sub-tile within a stage, too.
                for cw in 0..c.consumer_wgs {
                    assert_eq!(
                        (c.a_off(s) + cw * 64 * c.bk * c.dtype.size()) % align,
                        0,
                        "{} A stage {s} consumer {cw}",
                        c.name
                    );
                }
            }
            // mbarriers are 8-byte objects and must be 8-byte aligned.
            for s in 0..c.stages {
                assert_eq!(c.full_off(s) % 8, 0);
                assert_eq!(c.empty_off(s) % 8, 0);
            }
            // The regions do not overlap and the total is the last barrier's end.
            assert_eq!(c.b_off(0), c.stages * c.tile_a_bytes());
            assert_eq!(c.full_off(0), c.b_off(c.stages));
            assert_eq!(c.empty_off(0), c.full_off(c.stages));
            assert_eq!(c.smem_bytes(), c.empty_off(c.stages));
        }
    }

    /// The warp-specialised register split must fit the 64 Ki registers a CTA has, *after*
    /// `setmaxnreg` has run -- and each target must be a legal `setmaxnreg` operand.
    #[test]
    fn the_register_split_fits_the_file() {
        assert_eq!(WGMMA_W1.regs_after_split(), 128 * 32 + 256 * 232);
        assert_eq!(WGMMA_W1.regs_after_split(), 63488);
        for c in WGMMA_VARIANTS {
            assert!(
                c.regs_after_split() <= REGS_PER_CTA,
                "{}: {} registers",
                c.name,
                c.regs_after_split()
            );
            for r in [c.producer_regs, c.consumer_regs] {
                assert!(
                    (24..=256).contains(&r) && r.is_multiple_of(8),
                    "{}: {r}",
                    c.name
                );
            }
            // A consumer's accumulators alone must leave room for addressing.
            assert!(c.shape().unwrap().accum_regs() as u32 + 32 <= c.consumer_regs);
        }
        // And an over-generous split is refused rather than silently truncated by the hardware.
        let greedy = WgmmaCfg {
            consumer_regs: 256,
            producer_regs: 128,
            ..WGMMA_W1
        };
        assert!(greedy
            .validate()
            .unwrap_err()
            .contains("register split needs"));
    }

    // --- the generated text -----------------------------------------------------------------------

    /// **The ASCII gate** (crate rule #1). One non-ASCII byte anywhere in the module is a
    /// `ptxas fatal` inside `cuModuleLoadData`, and this file's own prose is full of arrows and
    /// multiplication signs waiting to be copied into a `format!`.
    #[test]
    fn wgmma_ptx_is_pure_ascii() {
        for (what, ptx) in wgmma_device_free_modules() {
            if let Some((i, line)) = ptx.lines().enumerate().find(|(_, l)| !l.is_ascii()) {
                panic!("{what}: PTX must be pure ASCII -- line {}: {line}", i + 1);
            }
        }
    }

    #[test]
    fn every_module_opens_at_the_architecture_locked_hopper_floor() {
        for (what, ptx) in wgmma_device_free_modules() {
            assert!(
                ptx.starts_with(HDR_SM90A_V80),
                "{what} must open with ptx_target::HDR_SM90A_V80"
            );
            assert!(ptx.contains(".target sm_90a\n"), "{what}");
            assert!(
                !ptx.contains(".target sm_80"),
                "{what}: a wgmma module cannot claim the Ampere floor"
            );
        }
    }

    /// The crate-wide `.version` law (`gpu.rs`'s
    /// `no_module_declares_a_driver_floor_its_instructions_do_not_need`) states the rule over the
    /// *instruction mix*: a module whose text contains none of a listed set of above-ISA-7.8
    /// instructions must declare `.version 7.8`. This family names `wgmma` and `cp.async.bulk`, so it
    /// is licensed -- and this is the identical rule applied over the identical corpus, so the family
    /// is covered today rather than after the wiring commit.
    #[test]
    fn the_family_declares_no_floor_it_does_not_need() {
        const ABOVE_78: &[&str] = &[
            "wgmma",
            "stmatrix",
            "elect.sync",
            "cp.async.bulk",
            "tcgen05",
            "clusterlaunchcontrol",
            "e4m3",
            "e5m2",
        ];
        let mods = wgmma_device_free_modules();
        assert_eq!(
            mods.len(),
            wgmma_all_emittable().len() + 2,
            "every emittable configuration, plus the TMA stage probe and the descriptor sweep probe"
        );
        // 4 shipped rows (W1 f16, W1 bf16, W3c, W1 f16 + 2x1x1 cluster) + the 5 sweep-only rows that
        // fit shared memory (128x256 at s2/s3 in both cluster settings, and W3c clustered) + the two
        // bring-up probes. The two 128x256 rows at s5 and s6 are deliberately NOT here: they decline
        // in `WgmmaCfg::validate` on the carveout, and the sweep prints the arithmetic rather than
        // emitting a module nothing can launch.
        assert_eq!(mods.len(), 11);
        for (what, ptx) in &mods {
            let version = ptx
                .lines()
                .next()
                .and_then(|l| l.strip_prefix(".version "))
                .unwrap_or_else(|| panic!("{what}: no .version directive"));
            let licensed = ABOVE_78.iter().find(|i| ptx.contains(**i));
            assert!(
                licensed.is_some(),
                "{what}: declares .version {version} while emitting nothing above ISA 7.8"
            );
            assert_eq!(
                version, "8.0",
                "{what}: every instruction this family emits was introduced at ISA 8.0, so a higher \
                 .version only raises the driver floor for nothing"
            );
        }
    }

    /// The mainloop's structure, counted rather than eyeballed: one `wgmma` per K-sub-step per
    /// consumer, bracketed by exactly one fence / commit / wait, an mbarrier per stage per direction,
    /// two TMA copies per iteration, and the register split.
    #[test]
    fn the_mainloop_is_structurally_what_the_design_says() {
        for c in wgmma_all_emittable() {
            let ptx = wgmma_module(c, &license()).unwrap();
            let shape = c.shape().unwrap();
            assert_eq!(
                ptx.matches(&format!(".visible .entry {}(", c.name)).count(),
                1
            );
            assert_eq!(ptx.matches('{').count(), ptx.matches('}').count());
            // one wgmma per k-sub-step of the staged tile (the K loop is a real loop, not unrolled)
            assert_eq!(
                ptx.matches("wgmma.mma_async.sync.aligned.").count(),
                c.wgmma_per_stage(),
                "{}: expected BK/16 = {} wgmma in the loop body",
                c.name,
                c.wgmma_per_stage()
            );
            assert!(ptx.contains(&format!(
                "wgmma.mma_async.sync.aligned.{}.{}",
                shape.token(),
                c.dtype.mma_types()
            )));
            // the async-proxy bracket
            assert_eq!(ptx.matches("wgmma.fence.sync.aligned;").count(), 1);
            assert_eq!(ptx.matches("wgmma.commit_group.sync.aligned;").count(), 1);
            assert_eq!(ptx.matches("wgmma.wait_group.sync.aligned 0;").count(), 2);
            // barriers: one full + one empty per stage, initialised once each
            assert_eq!(
                ptx.matches("mbarrier.init.shared::cta.b64").count(),
                2 * c.stages
            );
            assert_eq!(
                ptx.matches("mbarrier.init.shared::cta.b64 [%rdBar],1;")
                    .count(),
                c.stages,
                "{}: the full barrier expects exactly one arrival -- the TMA transaction",
                c.name
            );
            assert_eq!(
                ptx.matches(&format!(
                    "mbarrier.init.shared::cta.b64 [%rdBar],{};",
                    c.empty_arrivals()
                ))
                .count(),
                c.stages,
                "{}: the empty barrier expects one arrival per consumer warpgroup IN THE CLUSTER",
                c.name
            );
            assert_eq!(
                ptx.matches("mbarrier.try_wait.parity.shared::cta.b64")
                    .count(),
                2,
                "{}: one acquire in the producer, one in the consumer",
                c.name
            );
            assert_eq!(
                ptx.matches("mbarrier.arrive.expect_tx.shared::cta.b64")
                    .count(),
                1
            );
            assert!(ptx.contains(&format!(
                "mbarrier.arrive.expect_tx.shared::cta.b64 %rdSt,[%rdBarF],{};",
                c.stage_tx_bytes()
            )));
            // Two TMA copies per iteration, both against the full barrier -- A (multicast under a
            // cluster) and B (never multicast, because cluster peers share no B bytes).
            assert_eq!(
                ptx.matches("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes")
                    .count(),
                2
            );
            if c.cluster_ctas() > 1 {
                // --- the cluster arm ----------------------------------------------------------
                assert_eq!(
                    ptx.matches(
                        ".multicast::cluster [%rdA],[%rdTmA,{%tmp,%tmp2}],[%rdBarF],%cmask;"
                    )
                    .count(),
                    1,
                    "{}: A must be the multicast copy, with the ctaMask as its last operand",
                    c.name
                );
                assert!(
                    ptx.contains("bytes [%rdB],[%rdTmB,{%tmp,%ctan}],[%rdBarF];"),
                    "{}: B must NOT be multicast -- cluster peers hold different N halves",
                    c.name
                );
                assert!(ptx.contains(&format!(
                    "mov.u16 %cmask,{};",
                    multicast_cta_mask(c.cluster_ctas())
                )));
                // nvcc's own order and spelling for `__cluster_dims__`, right after `.maxntid`.
                assert!(ptx.contains(&format!(
                    ".maxntid {}, 1, 1\n.explicitcluster\n.reqnctapercluster {}, 1, 1\n{{\n",
                    c.threads(),
                    c.cluster_ctas()
                )));
                assert!(
                    !ptx.contains(".maxclusterrank"),
                    "{}: .maxclusterrank may not appear with .reqnctapercluster",
                    c.name
                );
                assert!(ptx.contains("mov.u32 %crank,%cluster_ctarank;"));
                // The slice geometry, in the two places it appears.
                assert!(ptx.contains(&format!("mul.lo.s32 %tmp,%crank,{};", c.a_slice_bytes())));
                assert!(ptx.contains(&format!("mul.lo.s32 %tmp2,%crank,{};", c.a_box_rows())));
                // One remote-capable arrival per cluster CTA, per consumer warpgroup.
                assert_eq!(
                    ptx.matches("mbarrier.arrive.shared::cluster.b64 _,[%rbar];")
                        .count(),
                    c.cluster_ctas(),
                    "{}: every CTA of the cluster must be released",
                    c.name
                );
                assert_eq!(
                    ptx.matches("mapa.shared::cluster.u32 %rbar,%sbar,%tmp2;")
                        .count(),
                    c.cluster_ctas()
                );
                assert!(
                    !ptx.contains("mbarrier.arrive.shared::cta.b64 %rdSt,[%rdBar];"),
                    "{}: the CTA-local-only release would let a producer recycle a stage a peer is \
                     still reading",
                    c.name
                );
                // Both rendezvous, and the init fence that makes the first one mean something.
                assert!(ptx.contains("fence.mbarrier_init.release.cluster;"));
                assert_eq!(ptx.matches("barrier.cluster.arrive.aligned;").count(), 2);
                assert_eq!(ptx.matches("barrier.cluster.wait.aligned;").count(), 2);
                assert!(
                    ptx.contains("barrier.cluster.wait.aligned;\n    ret;"),
                    "{}: a CTA must not retire while a peer can still write its shared memory",
                    c.name
                );
            } else {
                assert!(
                    !ptx.contains("multicast")
                        && !ptx.contains("cluster_ctarank")
                        && !ptx.contains("reqnctapercluster")
                        && !ptx.contains("barrier.cluster")
                        && !ptx.contains("mapa"),
                    "{}: an un-clustered row must emit NO cluster machinery -- it is the control arm \
                     of the cluster A/B and its text must stay the text the 2026-08-10 round measured",
                    c.name
                );
                assert!(ptx.contains("@%p2 mbarrier.arrive.shared::cta.b64 %rdSt,[%rdBar];"));
            }
            // the register split
            assert!(ptx.contains(&format!(
                "setmaxnreg.dec.sync.aligned.u32 {};",
                c.producer_regs
            )));
            assert!(ptx.contains(&format!(
                "setmaxnreg.inc.sync.aligned.u32 {};",
                c.consumer_regs
            )));
            assert!(ptx.contains(&format!(".maxntid {}, 1, 1", c.threads())));
            // the dynamic window, its symbol, and no static .shared at all
            assert!(ptx.contains(WGMMA_DSMEM_DECL));
            assert!(
                !ptx.contains("    .shared "),
                "{}: this family uses the extern window, never a static array",
                c.name
            );
            // the two by-value tensor maps and their generic addresses
            assert_eq!(ptx.matches(".param .align 64 .b8 tmap").count(), 2);
            assert_eq!(ptx.matches("cvta.param.u64").count(), 2);
        }
    }

    /// `scale-d` is the "skip the zero-init" trick: the first `wgmma` of the first K tile overwrites
    /// the accumulator instead of adding to it, so no zeroing loop is emitted at all. That is only
    /// correct if exactly one issue site takes the `kt != 0` predicate and every other takes true.
    #[test]
    fn scale_d_replaces_the_accumulator_zeroing() {
        for c in WGMMA_VARIANTS {
            let ptx = wgmma_module(c, &license()).unwrap();
            assert_eq!(
                ptx.matches(", %pfirst, 1, 1, 0, 0;").count(),
                1,
                "{}: exactly one wgmma may overwrite D",
                c.name
            );
            assert_eq!(
                ptx.matches(", %ptrue, 1, 1, 0, 0;").count(),
                c.wgmma_per_stage() - 1,
                "{}: every other wgmma accumulates",
                c.name
            );
            assert!(ptx.contains("setp.ne.u32 %pfirst,%kt,0;"));
            assert!(
                !ptx.contains("mov.f32 %acc"),
                "{}: no accumulator is zeroed by hand",
                c.name
            );
            // NT storage means neither operand is transposed: A is M x K with K contiguous, and
            // B is N x K with K contiguous, which is column-major B -- what wgmma wants at trans=0.
            assert!(!ptx.contains(", 1, 1, 1, 0;") && !ptx.contains(", 1, 1, 0, 1;"));
        }
    }

    /// The epilogue writes every accumulator exactly once, at the address the ISA's D-fragment layout
    /// puts it, with both a row and a column bound on each store.
    #[test]
    fn the_epilogue_stores_every_accumulator_exactly_once_and_bounded() {
        for c in WGMMA_VARIANTS {
            let ptx = wgmma_module(c, &license()).unwrap();
            let nacc = c.shape().unwrap().accum_regs();
            assert_eq!(ptx.matches("st.global.f32").count(), nacc);
            for i in 0..nacc {
                assert_eq!(
                    ptx.matches(&format!("],%acc{i};")).count(),
                    1,
                    "{}: %acc{i} must be stored exactly once",
                    c.name
                );
            }
            // Every store is predicated, and every predicate is a row bound AND a column bound.
            assert_eq!(ptx.matches("@%q0 st.global.f32").count(), nacc / 4);
            assert_eq!(ptx.matches("and.pred %q0,%pd0,%p0;").count(), nacc / 4);
            assert!(ptx.contains("setp.lt.u32 %pd0,%row0,%M;"));
            assert!(ptx.contains("setp.lt.u32 %p0,%col,%N;"));
            // K == 0 issues no wgmma at all, so the accumulators are never written and the epilogue
            // must not publish them.
            assert!(
                ptx.contains("setp.eq.u32 %p0,%ktiles,0;"),
                "{}: an empty K loop must skip the epilogue, not store uninitialised registers",
                c.name
            );
        }
    }

    /// **Every branch target is a defined label, and every label is branched to.** Labels here are
    /// built by string interpolation from the entry name, which makes a one-character drift a
    /// `CUDA_ERROR_INVALID_PTX` on a machine that is not this one -- and PTX labels are module-scoped,
    /// so an un-prefixed label would also collide the day two entries share a module.
    #[test]
    fn every_branch_target_is_a_defined_label() {
        for c in WGMMA_VARIANTS {
            let ptx = wgmma_module(c, &license()).unwrap();
            let defined: Vec<&str> = ptx
                .lines()
                .filter(|l| !l.starts_with(' ') && l.ends_with(':'))
                .map(|l| l.trim_end_matches(':'))
                .collect();
            let mut used: Vec<&str> = Vec::new();
            for l in ptx.lines() {
                if let Some(i) = l.find("bra ") {
                    used.push(l[i + 4..].trim().trim_end_matches(';'));
                }
            }
            assert!(!used.is_empty());
            for u in &used {
                assert!(defined.contains(u), "{}: branch to undefined `{u}`", c.name);
                assert!(
                    u.ends_with(c.name),
                    "{}: label `{u}` is not entry-scoped; PTX labels are module-scoped",
                    c.name
                );
            }
            for d in &defined {
                assert!(
                    used.contains(d),
                    "{}: label `{d}` is defined but never branched to",
                    c.name
                );
            }
        }
    }

    /// The accumulator register file is declared at exactly the size the shape implies -- one short
    /// and the `wgmma` operand list names an undeclared register; one long and the register budget is
    /// wrong.
    #[test]
    fn the_accumulator_vector_matches_the_shape() {
        for c in WGMMA_VARIANTS {
            let ptx = wgmma_module(c, &license()).unwrap();
            let nacc = c.shape().unwrap().accum_regs();
            assert!(ptx.contains(&format!(".reg .f32 %acc<{nacc}>;")));
            assert!(ptx.contains("%acc0,%acc1,"));
            assert!(ptx.contains(&format!("%acc{}}}", nacc - 1)));
        }
        assert_eq!(WGMMA_W1.shape().unwrap().accum_regs(), 128);
        assert_eq!(WGMMA_W3C.shape().unwrap().accum_regs(), 64);
    }

    // --- declines ---------------------------------------------------------------------------------

    /// **Everything this generator cannot express declines with the documented prefix.** A shape that
    /// cannot be expressed is a skip; emitting plausible PTX for it is the one failure this family
    /// must never have.
    #[test]
    fn unsupported_shapes_decline_rather_than_emit() {
        let lic = license();
        let cases: Vec<(&str, WgmmaCfg, &str)> = vec![
            (
                "a clustered row too deep for the carveout",
                WGMMA_W1_MC_S5,
                "exceeds the 232448 B per-CTA ceiling",
            ),
            (
                "a layout no tiled TMA copy writes",
                WgmmaCfg {
                    layout: SmemLayout::CanonicalNone {
                        k_fast: true,
                        swapped: false,
                    },
                    ..WGMMA_W1
                },
                "128 contiguous bytes",
            ),
            (
                "the 128-B swizzle at a row that is not 128 B",
                WgmmaCfg { bk: 32, ..WGMMA_W1 },
                "swizzle atom is 128 bytes",
            ),
            (
                "N off the menu",
                WgmmaCfg {
                    bn: 200 + 4,
                    ..WGMMA_W1
                },
                "is not on the ISA menu",
            ),
            (
                "N past 256",
                WgmmaCfg {
                    bn: 512,
                    ..WGMMA_W1
                },
                "is not on the ISA menu",
            ),
            (
                "CTA-M below the cooperative floor",
                WgmmaCfg {
                    bm: 64,
                    consumer_wgs: 1,
                    ..WGMMA_W1
                },
                "not a multiple of 128",
            ),
            (
                "CTA-M that is not 64 per consumer",
                WgmmaCfg {
                    bm: 256,
                    consumer_wgs: 2,
                    ..WGMMA_W1
                },
                "consumer warpgroups",
            ),
            (
                "a single-buffered pipeline",
                WgmmaCfg {
                    stages: 1,
                    ..WGMMA_W1
                },
                "at least 2",
            ),
            (
                "BK off the wgmma K step",
                WgmmaCfg { bk: 24, ..WGMMA_W1 },
                "power of two",
            ),
            (
                "an illegal setmaxnreg operand",
                WgmmaCfg {
                    consumer_regs: 230,
                    ..WGMMA_W1
                },
                "must be a multiple of 8",
            ),
        ];
        for (what, cfg, needle) in cases {
            let e = wgmma_module(&cfg, &lic).expect_err(&format!("{what} must decline, not emit"));
            assert!(
                e.starts_with(UNSUPPORTED),
                "{what}: decline must carry the {UNSUPPORTED} prefix: {e}"
            );
            assert!(e.contains(needle), "{what}: {e}");
        }
    }

    /// **Every way a multicast slice can be wrong, rejected by name.**
    ///
    /// Stated over numbers rather than over `WgmmaCfg`s because today's [`Multicast`] menu can only
    /// produce a 2-CTA cluster, so most of these arms are unreachable from a shipped row -- and an
    /// unreachable guard that has never been executed is a guard nobody has read. The `align` is the
    /// 128-B swizzle's, which is what every shipped row carries.
    #[test]
    fn the_cluster_preconditions_reject_every_way_a_slice_can_be_wrong() {
        let ok = |ctas, bm, row_bytes| validate_cluster("row", ctas, bm, row_bytes, 128);
        // No cluster: nothing to check, whatever the rest says.
        ok(1, 129, 3).unwrap();
        // The shipped geometry: 2 CTAs, CTA-M 128, a 128 B shared row.
        ok(2, 128, 128).unwrap();
        ok(8, 128, 128).unwrap();

        let e = ok(16, 128, 128).unwrap_err();
        assert!(
            e.starts_with(UNSUPPORTED) && e.contains("portable ceiling"),
            "{e}"
        );
        let e = ok(3, 128, 128).unwrap_err();
        assert!(e.contains("does not divide by the 3-CTA cluster"), "{e}");
        // 128 rows over 8 CTAs is 16 rows each (legal); over 32 it would be 4 -- but 32 trips the
        // portable ceiling first, so the swizzle-phase arm needs a divisible-but-unaligned split.
        let e = ok(2, 4, 128).unwrap_err();
        assert!(e.contains("8-row core matrix"), "{e}");
        // A slice taller than TMA's box limit: 2048 rows over 2 CTAs is 1024 each.
        let e = ok(2, 2048, 128).unwrap_err();
        assert!(e.contains("element box limit"), "{e}");
        // A legal row count whose bytes do not reach the descriptor's alignment.
        let e = ok(2, 16, 8).unwrap_err();
        assert!(e.contains("not a multiple of the 128 B alignment"), "{e}");

        // ...and the two arms that ARE reachable from the table: the deep clustered rows decline on
        // shared memory, with the arithmetic in the message, and every other sweep row generates.
        for r in WGMMA_SWEEP_GRID {
            match r.generatable() {
                Ok(()) => assert!(
                    r.cfg.smem_bytes() <= HOPPER_SMEM_PER_CTA,
                    "{}: generates but does not fit?",
                    r.label
                ),
                Err(e) => {
                    assert!(e.starts_with(UNSUPPORTED), "{}: {e}", r.label);
                    assert!(e.contains("per-CTA ceiling"), "{}: {e}", r.label);
                    assert!(r.cfg.smem_bytes() > HOPPER_SMEM_PER_CTA);
                }
            }
            assert!(r.smem_line().is_ascii());
        }
        // The budget arithmetic itself, at the two ends of the depth axis, spelled out -- so a
        // refactor that changes what a stage costs cannot quietly move where the axis stops.
        // Clustering does NOT change the ring: the A tile is still staged whole in every CTA, it
        // just arrives in `cluster_ctas` multicast pieces instead of one local copy.
        assert_eq!(WGMMA_W1.stage_tx_bytes(), 49152);
        assert_eq!(WGMMA_W1_MC.stage_tx_bytes(), WGMMA_W1.stage_tx_bytes());
        assert_eq!(WGMMA_W1_MC.smem_bytes(), WGMMA_W1.smem_bytes());
        assert_eq!(WGMMA_W1_MC_S2.smem_bytes(), 2 * 49152 + 2 * 2 * 8);
        assert_eq!(WGMMA_W1_MC_S3.smem_bytes(), 3 * 49152 + 2 * 3 * 8);
        assert_eq!(WGMMA_W1_MC_S5.smem_bytes(), 5 * 49152 + 2 * 5 * 8);
        assert!(WGMMA_W1_MC_S3.smem_bytes() <= HOPPER_SMEM_PER_CTA);
        assert!(WGMMA_W1_MC_S5.smem_bytes() > HOPPER_SMEM_PER_CTA);
        assert_eq!(WGMMA_W3C_MC.stage_tx_bytes(), 32768);
        assert_eq!(WGMMA_W3C_MC.smem_bytes(), WGMMA_W3C.smem_bytes());
        // The multicast slice arithmetic of every clustered row, at the same two ends.
        for c in [&WGMMA_W1_MC, &WGMMA_W3C_MC] {
            assert_eq!(c.a_box_rows(), c.bm / 2);
            assert_eq!(c.a_slice_bytes() * 2, c.tile_a_bytes());
            assert_eq!(c.empty_arrivals(), 2 * c.consumer_wgs);
            assert_eq!(c.stage_copies_per_cta(), 3);
        }
    }

    /// A BK whose contiguous extent overflows the TMA box rule is caught by the descriptor
    /// validator, before any PTX exists -- the two halves of the family agree on the same geometry.
    #[test]
    fn the_tma_geometry_is_validated_with_the_kernel() {
        // BK = 256 f16 = 512 B of contiguous box: legal unswizzled (box_dim[0] = 256 is the maximum),
        // and the SMEM budget is what stops it.
        let wide = WgmmaCfg {
            bk: 256,
            stages: 2,
            ..WGMMA_W1
        };
        let e = wide.validate().unwrap_err();
        assert!(e.starts_with(UNSUPPORTED), "{e}");
        // A box dimension past 256 elements is a TMA rule, and it is the TMA validator that says so.
        let huge = WgmmaCfg {
            bk: 512,
            stages: 2,
            ..WGMMA_W1
        };
        let e = huge.validate().unwrap_err();
        assert!(e.contains("box_dim") || e.contains("exceeds"), "{e}");
    }

    // --- the launch seam --------------------------------------------------------------------------

    /// The grid covers the whole output, at exact and at ragged shapes, and the block is the
    /// warpgroup count. This is the arithmetic a launcher must not re-derive.
    #[test]
    fn the_launch_plan_covers_the_output() {
        let p = WGMMA_W1.launch_plan();
        assert_eq!(p.entry, "wgmma_nt_f16_128x256x64_s4");
        assert_eq!(p.block, (384, 1, 1));
        assert_eq!(p.dyn_smem_bytes, WGMMA_W1.smem_bytes());
        assert_eq!(p.params, PARAM_ORDER);
        assert_eq!(p.params.len(), 6);
        for (m, n) in [(4096, 4096), (8192, 8192), (1, 1), (129, 257), (2048, 3072)] {
            let (gx, gy, gz) = p.grid(m, n);
            assert_eq!(gz, 1);
            assert!(
                (gx as usize) * p.bn >= n && (gy as usize) * p.bm >= m,
                "grid {gx}x{gy} does not cover {m}x{n}"
            );
            assert!(
                (gx as usize - 1) * p.bn < n && (gy as usize - 1) * p.bm < m,
                "grid {gx}x{gy} launches a wholly empty tile for {m}x{n}"
            );
        }
        // Every shipped variant's plan is self-consistent with its config.
        for c in WGMMA_VARIANTS {
            let p = c.launch_plan();
            assert_eq!(p.module_key, c.key);
            assert_eq!(p.dyn_smem_bytes, c.smem_bytes());
            assert_eq!(p.block.0 as usize, c.threads());
        }
        // Module-cache keys must be unique: `Gpu::function` keys on the string alone and never
        // re-examines the PTX, so two variants sharing a key silently share one compiled module.
        let mut keys: Vec<&str> = WGMMA_VARIANTS.iter().map(|c| c.key).collect();
        keys.sort_unstable();
        let n = keys.len();
        keys.dedup();
        assert_eq!(
            keys.len(),
            n,
            "duplicate module-cache key among the variants"
        );
    }

    /// **The TMA descriptors the launcher builds and the transaction count the kernel declares are
    /// the same fact viewed twice, and multicast does not divide it.** A mismatch does not fail --
    /// it hangs, which on rented silicon is the whole rest of the hour.
    ///
    /// The clustered rows are where this stops being a restatement. Each CTA fetches
    /// `1 / cluster_ctas` of A and multicasts it, so its A descriptor moves `tile_a / cluster_ctas`
    /// bytes -- but the copy performs a `complete-tx` of that amount on *every* destination CTA's
    /// barrier, so `cluster_ctas` such copies plus one own B tile land the full `tile_a + tile_b` in
    /// each CTA. The arithmetic below is that sentence.
    #[test]
    fn the_declared_transaction_equals_what_the_copies_move_per_destination_cta() {
        for c in wgmma_all_emittable() {
            let a = c.tensor_map_a(4096, 4096);
            let b = c.tensor_map_b(4096, 4096);
            a.validate().unwrap();
            b.validate().unwrap();
            let ctas = c.cluster_ctas();
            assert_eq!(
                ctas * a.transaction_bytes() + b.transaction_bytes(),
                c.stage_tx_bytes(),
                "{}: expect_tx must equal what {ctas} multicast A slices plus one B tile deliver \
                 into ONE CTA",
                c.name
            );
            assert_eq!(a.transaction_bytes(), c.a_slice_bytes());
            assert_eq!(b.transaction_bytes(), c.tile_b_bytes());
            assert_eq!(a.box_dim[0] as usize, c.bk);
            assert_eq!(a.box_dim[1] as usize, c.a_box_rows());
            assert_eq!(b.box_dim[1] as usize, c.bn);
            // The count is the SAME number with and without a cluster -- that is the claim
            // `WgmmaCfg::stage_tx_bytes` makes, and it is worth checking rather than believing.
            assert_eq!(c.stage_tx_bytes(), c.tile_a_bytes() + c.tile_b_bytes());
            // ...and the slices tile the A stage exactly, in rank order, with no gap and no overlap.
            assert_eq!(c.a_slice_off(0), 0);
            assert_eq!(c.a_slice_off(ctas), c.tile_a_bytes());
            assert_eq!(c.stage_copies_per_cta(), ctas + 1);
        }
    }

    /// **The multicast destination mask.** It is a pure function of the cluster size and it is the
    /// only thing standing between "A reaches both CTAs" and "half the accumulator rows read
    /// whatever the previous iteration left in shared memory".
    #[test]
    fn the_multicast_mask_names_every_cta_of_the_cluster() {
        assert_eq!(multicast_cta_mask(1), 0b1);
        assert_eq!(multicast_cta_mask(2), 0b11);
        assert_eq!(multicast_cta_mask(4), 0b1111);
        assert_eq!(multicast_cta_mask(8), 0xff);
        assert_eq!(multicast_cta_mask(16), 0xffff);
        for ctas in 1..=16usize {
            let m = multicast_cta_mask(ctas);
            assert_eq!(m.count_ones() as usize, ctas, "{ctas} CTAs, mask {m:#x}");
            for r in 0..ctas {
                assert_ne!(m & (1 << r), 0, "rank {r} is not a destination");
            }
            for r in ctas..16 {
                assert_eq!(m & (1 << r), 0, "rank {r} is not in the cluster");
            }
        }
        // The shipped cluster row's own mask, spelled out: both CTAs of a 2x1x1 cluster.
        assert_eq!(multicast_cta_mask(WGMMA_W1_MC.cluster_ctas()), 3);
        assert_eq!(multicast_cta_mask(WGMMA_W1.cluster_ctas()), 1);
    }

    /// A cluster is a fixed shape, so the grid must be a multiple of it in every axis. The extra
    /// CTAs an odd tile count produces are the ragged edge one notch coarser -- wholly out of range,
    /// zero-filled by TMA, and predicated out of the epilogue -- and never a partial cluster.
    #[test]
    fn the_grid_is_cluster_divisible_and_still_covers_the_output() {
        for c in wgmma_all_emittable() {
            let p = c.launch_plan();
            let (cx, cy) = (p.cluster.0 as usize, p.cluster.1 as usize);
            assert_eq!(p.cluster_ctas() as usize, c.cluster_ctas());
            assert!(p.cluster_ctas() >= 1 && p.cluster.2 == 1);
            for (m, n) in [
                (1usize, 1usize),
                (128, 256),
                (129, 257),
                (4096, 4096),
                (8192, 8192),
                (4096, 1024),
                // The interesting one: an ODD number of N tiles under a 2-CTA cluster.
                (128, 3 * c.bn),
                (2 * c.bm + 1, 5 * c.bn - 1),
            ] {
                let (gx, gy, gz) = p.grid(m, n);
                assert_eq!(gz, 1);
                assert_eq!(
                    gx as usize % cx,
                    0,
                    "{}: grid x {gx} is not a multiple of the cluster's {cx} -- there is no such \
                     thing as a partial cluster",
                    c.name
                );
                assert_eq!(
                    gy as usize % cy,
                    0,
                    "{}: grid y {gy} vs cluster {cy}",
                    c.name
                );
                assert!(
                    (gx as usize) * p.bn >= n && (gy as usize) * p.bm >= m,
                    "{}: grid {gx}x{gy} does not cover {m}x{n}",
                    c.name
                );
                // Rounding is bounded: never more than one cluster of slack in either axis.
                assert!(
                    (gx as usize) * p.bn < n + cx * p.bn,
                    "{}: grid x {gx} overshoots {n} by more than one cluster",
                    c.name
                );
            }
        }
        // ...and the rounding is exactly the cluster's, not a blanket round-up: the un-clustered row
        // still launches the tight grid the 2026-08-10 round measured.
        assert_eq!(WGMMA_W1.launch_plan().grid(128, 3 * 256), (3, 1, 1));
        assert_eq!(WGMMA_W1_MC.launch_plan().grid(128, 3 * 256), (4, 1, 1));
    }

    /// The unproven-claims list is the honest half of this module and must not quietly empty out or
    /// lose its head item. It is also printed into bring-up logs, so it stays ASCII.
    ///
    /// Every item except 3 and 8 must now name the gate that discharges it. Item 3 (registers and
    /// spills) is already answered by the CPU ptxas census at `sm_90a` -- 168 regs whole-CTA and zero
    /// spills on all three rows, which `setmaxnreg` then splits 32p/232c on the two 128x256 s4 rows
    /// and 32p/168c on the 128x128 s6 one -- and item 8 (performance vs cuBLAS) is explicitly
    /// downstream of correctness. An item that names no gate is an item nobody will run.
    #[test]
    fn the_device_validation_list_is_intact() {
        assert_eq!(WGMMA_DEVICE_VALIDATION.len(), 9);
        // Item 1 is the head item and must name the mechanism that discharges it -- the sweep, and
        // the two controls that make the sweep self-validating.
        for need in [
            "desc_sweep_candidates",
            "desc_sweep_probe_module",
            "CONTROL",
        ] {
            assert!(
                WGMMA_DEVICE_VALIDATION[0].contains(need),
                "item 1 must name {need:?}: {}",
                WGMMA_DEVICE_VALIDATION[0]
            );
        }
        for item in WGMMA_DEVICE_VALIDATION {
            assert!(item.is_ascii(), "{item}");
            assert!(item.len() > 40);
        }
        for i in [0usize, 1, 3, 4, 5, 6] {
            assert!(
                WGMMA_DEVICE_VALIDATION[i].contains("wgmma_hopper_bringup"),
                "item {} names no gate: {}",
                i + 1,
                WGMMA_DEVICE_VALIDATION[i]
            );
        }
        assert!(
            WGMMA_DEVICE_VALIDATION[2].contains("ptxas"),
            "item 3 is answered by the CPU ptxas census and must say so"
        );
        // Item 8 has a gate now. It must name both benches, the instrument they run through and the
        // fact that they do not assert a ratio -- the three things a reader needs before spending an
        // hour on it, and the three that would rot silently if only prose carried them.
        for need in [
            "wgmma_vs_cublas",
            "wgmma_bf16_vs_cublas",
            "bench_instrument",
            "WGMMA_BENCH_GRID",
            "WGMMA_BENCH_INVOCATION",
        ] {
            assert!(
                WGMMA_DEVICE_VALIDATION[7].contains(need),
                "item 8 must name {need:?}: {}",
                WGMMA_DEVICE_VALIDATION[7]
            );
        }
        // Item 9 is the cluster. It must name the four mechanisms that fail silently, the
        // correctness gate that settles them and the performance round that follows -- an item
        // that names no gate is an item nobody will run, and this one's failure mode is a hang.
        for need in [
            "ctaMask",
            "mapa",
            "barrier.cluster",
            "transaction count",
            "wgmma_cluster_multicast_is_exact",
            "wgmma_config_sweep",
            "WGMMA_SWEEP_INVOCATION",
            "WGMMA_W1_MC",
        ] {
            assert!(
                WGMMA_DEVICE_VALIDATION[8].contains(need),
                "item 9 must name {need:?}: {}",
                WGMMA_DEVICE_VALIDATION[8]
            );
        }
    }

    /// The round-2 sweep's own invocation and shape selection, checked device-free -- a misspelled
    /// shape label would otherwise be a rented minute spent on a panic, and a missing `--release`
    /// would mismeasure a 25-microsecond kernel with a debug host loop.
    #[test]
    fn the_sweep_invocation_and_shapes_name_things_that_exist() {
        let gate = WGMMA_CLUSTER_GATE_INVOCATION;
        assert!(gate.is_ascii());
        assert!(gate.contains("wgmma_cluster_multicast_is_exact"));
        assert!(
            !gate.contains("--ignored"),
            "the cluster gate is a GATE, not a bench: it runs in a plain `cargo test`"
        );
        let inv = WGMMA_SWEEP_INVOCATION;
        assert!(inv.is_ascii());
        for need in [
            "WUKONG_GPU_REQUIRED=1",
            "WUKONG_PEER_REQUIRED=1",
            "--features gpu",
            "--release",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
            "wgmma_config_sweep",
        ] {
            assert!(
                inv.contains(need),
                "the sweep invocation must carry {need:?}: {inv}"
            );
        }
        // Every shape is a real grid row, so the sweep and `wgmma_vs_cublas` share denominators.
        let pts = wgmma_sweep_points();
        assert_eq!(pts.len(), WGMMA_SWEEP_SHAPES.len());
        assert!(pts.iter().any(|p| p.label == WGMMA_SWEEP_HEADLINE));
        assert!(
            WGMMA_SWEEP_SHAPES.contains(&"sq4096") && WGMMA_SWEEP_SHAPES.contains(&"sq8192"),
            "the sweep must measure both of D1 4.5's prediction points"
        );
        // Every row's label is a usable log/sample-label token, and unique.
        let mut labels: Vec<&str> = WGMMA_SWEEP_GRID.iter().map(|r| r.label).collect();
        let n = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), n, "duplicate sweep row label");
        for r in WGMMA_SWEEP_GRID {
            assert!(r.label.is_ascii() && !r.label.contains(char::is_whitespace));
            assert!(r.why.is_ascii() && r.why.len() > 40, "{}", r.label);
        }
        // The headline A/B exists, is one fact apart, and both arms are in the table.
        let by = |l: &str| {
            WGMMA_SWEEP_GRID
                .iter()
                .find(|r| r.label == l)
                .unwrap_or_else(|| panic!("the sweep must carry the row {l:?}"))
        };
        let (off, mc) = (by("w1_s4_off").cfg, by("w1_s4_mc2").cfg);
        assert_eq!(off.cluster_ctas(), 1);
        assert_eq!(mc.cluster_ctas(), 2);
        for (what, a, b) in [
            ("bm", off.bm, mc.bm),
            ("bn", off.bn, mc.bn),
            ("bk", off.bk, mc.bk),
            ("stages", off.stages, mc.stages),
            ("consumer_wgs", off.consumer_wgs, mc.consumer_wgs),
            ("threads", off.threads(), mc.threads()),
            ("smem", off.smem_bytes(), mc.smem_bytes()),
        ] {
            assert_eq!(
                a, b,
                "the cluster A/B's arms differ in {what} as well as the cluster"
            );
        }
        assert_eq!(off.layout, mc.layout);
        assert_eq!(off.consumer_regs, mc.consumer_regs);
        assert_eq!(off.producer_regs, mc.producer_regs);
        // ...and the depth axis is paired at every measurable depth, so "deeper" and "clustered"
        // cannot be confounded with one another.
        for depth in [2usize, 3, 4] {
            for want_cluster in [1usize, 2] {
                assert!(
                    WGMMA_SWEEP_GRID.iter().any(|r| {
                        r.cfg.stages == depth
                            && r.cfg.cluster_ctas() == want_cluster
                            && r.cfg.bn == 256
                            && r.generatable().is_ok()
                    }),
                    "the depth axis is missing 128x256 at {depth} stages, cluster {want_cluster}"
                );
            }
        }
    }

    // --- Act 2: the performance grid ---------------------------------------------------------------

    /// The operator's Act-2 invocation is data too, and it needs three flags the bring-up one does
    /// not: `--release` (a debug host loop mismeasures a 25-microsecond kernel), `--ignored` (the
    /// benches are `#[ignore]`d) and `WUKONG_PEER_REQUIRED=1` (a round that cannot reach cuBLAS has
    /// no bar and must fail rather than print `[skip]` and report green).
    #[test]
    fn the_bench_invocation_names_the_bench_and_the_flags_it_needs() {
        let inv = WGMMA_BENCH_INVOCATION;
        assert!(inv.is_ascii());
        for need in [
            "WUKONG_GPU_REQUIRED=1",
            "WUKONG_PEER_REQUIRED=1",
            "--features gpu",
            "--release",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
            "wgmma_vs_cublas",
        ] {
            assert!(
                inv.contains(need),
                "the Act-2 invocation must carry {need:?}: {inv}"
            );
        }
    }

    /// **Every grid point must be launchable by every shipped row, and that is knowable with no
    /// device.**
    ///
    /// The point of checking it here rather than discovering it in the round is cost: a shape whose
    /// tensor map does not encode, whose epilogue index overflows `u32`, or whose ring does not fit
    /// Hopper's 227 KiB carveout is a rented minute spent on a panic. Every predicate below is one
    /// `gemm_nt_wgmma`/`time_gemm_nt_wgmma` asserts at the launch seam, evaluated ahead of time.
    #[test]
    fn the_bench_grid_is_launchable_by_every_shipped_row() {
        assert!(!WGMMA_BENCH_GRID.is_empty());
        let mut seen: Vec<&str> = Vec::new();
        for p in WGMMA_BENCH_GRID {
            assert!(p.label.is_ascii() && !p.label.contains(char::is_whitespace));
            assert!(
                !seen.contains(&p.label),
                "duplicate grid label {:?}",
                p.label
            );
            seen.push(p.label);
            assert!(p.why.is_ascii() && p.why.len() > 40, "{}", p.label);
            assert!(p.m > 0 && p.n > 0 && p.k > 0);
            // The epilogue forms its element index with `mad.lo.s32` before widening.
            assert!(
                (p.m as u64) * (p.n as u64) <= u32::MAX as u64,
                "{}: M*N overflows the u32 element index",
                p.label
            );
            // A tensor map needs a global row stride that is a multiple of 16 B, and K is the
            // contiguous axis of both NT operands -- so `K % 8 != 0` is unencodable, not ragged.
            assert!(
                p.k.is_multiple_of(8),
                "{}: K={} gives a {} B row stride, which cuTensorMapEncodeTiled rejects",
                p.label,
                p.k,
                p.k * 2
            );
            let iters = bench_iters(p);
            assert!(
                (BENCH_MIN_ITERS..=BENCH_MAX_ITERS).contains(&iters),
                "{}: {iters} iters is outside the clamp",
                p.label
            );
            for cfg in WGMMA_VARIANTS {
                cfg.tensor_map_a(p.m, p.k)
                    .validate()
                    .unwrap_or_else(|e| panic!("{} A map at {}: {e}", cfg.name, p.label));
                cfg.tensor_map_b(p.n, p.k)
                    .validate()
                    .unwrap_or_else(|e| panic!("{} B map at {}: {e}", cfg.name, p.label));
                let plan = cfg.launch_plan();
                assert!(
                    plan.dyn_smem_bytes <= HOPPER_SMEM_PER_CTA,
                    "{}: {} B ring exceeds Hopper's {HOPPER_SMEM_PER_CTA} B carveout",
                    cfg.name,
                    plan.dyn_smem_bytes
                );
                let (gx, gy, gz) = plan.grid(p.m, p.n);
                assert!(gx > 0 && gy > 0 && gz == 1, "{}: empty grid", p.label);
            }
        }
    }

    /// D1 section 4.4 names the grid; the constant must actually carry it, and must carry the two
    /// rectangular classes the task of measuring W1 turns on.
    #[test]
    fn the_bench_grid_carries_d1s_points_and_both_rectangular_classes() {
        let has = |m: usize, n: usize, k: usize| {
            WGMMA_BENCH_GRID
                .iter()
                .any(|p| p.m == m && p.n == n && p.k == k)
        };
        for sz in [1024usize, 2048, 4096, 8192] {
            assert!(has(sz, sz, sz), "D1 4.4 asks for {sz} cubed");
        }
        // GPT (M=4096, N=4d, K=d) at d in {1024, 4096} -- D1 4.4, verbatim.
        assert!(has(4096, 4096, 1024), "GPT d=1024 up-projection missing");
        assert!(has(4096, 16384, 4096), "GPT d=4096 up-projection missing");
        // A skinny-K point (K far below M and N) and a skinny-N point (N far below M and K).
        assert!(
            WGMMA_BENCH_GRID
                .iter()
                .any(|p| p.k * 2 <= p.m.min(p.n) && p.m > 1024),
            "the grid has no skinny-K point"
        );
        assert!(
            WGMMA_BENCH_GRID
                .iter()
                .any(|p| p.n * 2 <= p.m.min(p.k) && p.m > 1024),
            "the grid has no skinny-N point"
        );
    }

    /// The iteration rule is the A/B's denominator, so it must be a pure function of the shape --
    /// never of the arm, and never of anything measured. It also has to be monotone: a bigger shape
    /// may not ask for *more* launches, or the sweep's cost stops being predictable.
    #[test]
    fn the_iteration_count_is_a_clamped_monotone_function_of_the_shape() {
        let mut ordered: Vec<&GemmPoint> = WGMMA_BENCH_GRID.iter().collect();
        ordered.sort_by(|a, b| a.flop().partial_cmp(&b.flop()).expect("finite"));
        let mut prev = usize::MAX;
        for p in &ordered {
            let it = bench_iters(p);
            assert!(
                it <= prev,
                "{}: {it} iters at {:.3e} FLOP is more than the smaller shape's {prev}",
                p.label,
                p.flop()
            );
            prev = it;
        }
        // The clamp holds at both ends, including for shapes no grid row has.
        let tiny = GemmPoint {
            label: "tiny",
            m: 8,
            n: 8,
            k: 8,
            why: "",
        };
        let huge = GemmPoint {
            label: "huge",
            m: 65536,
            n: 65536,
            k: 65536,
            why: "",
        };
        assert_eq!(bench_iters(&tiny), BENCH_MAX_ITERS);
        assert_eq!(bench_iters(&huge), BENCH_MIN_ITERS);
    }

    /// The operator's invocation is data, so it cannot drift from the gate it names.
    #[test]
    fn the_bringup_invocation_names_the_gate_and_the_flags_it_needs() {
        let inv = WGMMA_BRINGUP_INVOCATION;
        assert!(inv.is_ascii());
        for need in [
            "WUKONG_GPU_REQUIRED=1",
            "--features gpu",
            "--nocapture",
            "--test-threads=1",
            "wgmma_hopper_bringup",
        ] {
            assert!(
                inv.contains(need),
                "the invocation must carry {need:?}: {inv}"
            );
        }
    }

    // --- bring-up: the DescOrder A/B is a real A/B ------------------------------------------------

    /// f64 reference for `C = A.Bt`, independent of every descriptor, encoder and generator in this
    /// file -- it reads the operand arrays and nothing else.
    fn ref_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for (i, crow) in c.chunks_exact_mut(n).enumerate() {
            for (j, cij) in crow.iter_mut().enumerate() {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += a[i * k + kk] as f64 * b[j * k + kk] as f64;
                }
                *cij = acc as f32;
            }
        }
        c
    }

    /// **THE 2026-08-10 DEFECT, RE-DERIVED WITH NO HOPPER PART IN THE ROOM.**
    ///
    /// The H100 log (`bench/gpu/h100/2026-08-10-h100-s2a-bringup.log`) reports two numbers for the
    /// two readings this family shipped, at M=N=K=64: **64 of 4096 output lanes exact** for
    /// `k_leading: true` and **0 of 4096** for `k_leading: false`. Those two integers are the whole
    /// evidence, and a model that cannot reproduce them has no business choosing the next candidate.
    ///
    /// [`unswizzled_read`] is that model. Three clauses:
    ///
    /// 1. **No (LBO, SBO) pair describes a row-major tile** whose rows are wider than one core
    ///    matrix. The hardware's within-core-matrix row stride is the fixed 16 bytes; the tile's is
    ///    `row_bytes`. Nothing in the descriptor can close that gap, so this is a LAYOUT defect and
    ///    every "try the other field order" is a wasted launch.
    /// 2. The exact-lane count the model predicts for `k_leading: true` is **64**, and for
    ///    `k_leading: false` is **0** -- the log, to the lane.
    /// 3. The two readings still address different elements, so the 2026-08-10 A/B was a real A/B.
    ///    It simply asked a question whose answer was "neither".
    #[test]
    fn the_row_major_reading_is_unrepresentable_and_predicts_the_log() {
        let (m64, n64, k64) = (64usize, 64usize, 64usize);
        let row_bytes = (2 * k64) as u64; // 128 B, the geometry every shipped row uses
        let rows = 64u64;
        let kl = desc_fields(
            SmemLayout::RowMajorNone { k_leading: true },
            row_bytes,
            rows,
        );
        let ml = desc_fields(
            SmemLayout::RowMajorNone { k_leading: false },
            row_bytes,
            rows,
        );
        assert_ne!((kl.lbo, kl.sbo), (ml.lbo, ml.sbo));
        assert_ne!(
            SmemDesc::for_layout(
                0,
                SmemLayout::RowMajorNone { k_leading: true },
                row_bytes,
                rows
            )
            .const_part()
            .unwrap(),
            SmemDesc::for_layout(
                0,
                SmemLayout::RowMajorNone { k_leading: false },
                row_bytes,
                rows
            )
            .const_part()
            .unwrap(),
            "the 2026-08-10 A/B really did launch two different descriptors"
        );

        // 1. Exhaustive over every (LBO, SBO) pair a 14-bit field can hold at 16-byte granularity:
        //    none of them turns the canonical model into the row-major truth, because the r term is
        //    not a function of either field.
        let elem = 2u64;
        let truth = |i: u64, j: u64, r: u64, c: u64| (8 * i + r) * row_bytes + (8 * j + c) * elem;
        let mut representable = false;
        for lbo in (0..=4096u64).step_by(16) {
            for sbo in (0..=4096u64).step_by(16) {
                let d = SmemDesc {
                    start_addr: 0,
                    lbo,
                    sbo,
                    base_offset: 0,
                    swizzle: SmemSwizzle::None,
                };
                // Only the FIRST wgmma K step (j0 = 0), which is the most generous case: if even
                // that cannot be matched, no base advance saves the later ones.
                let ok = (0..8u64).all(|i| {
                    (0..2u64).all(|j| {
                        (0..8u64).all(|r| {
                            (0..8u64)
                                .all(|c| d.canonical_offset(i, j, r, c, elem) == truth(i, j, r, c))
                        })
                    })
                });
                if ok {
                    representable = true;
                }
            }
        }
        assert!(
            !representable,
            "a row-major tile with {row_bytes}-byte rows must be describable by NO descriptor -- if \
             this ever passes, the model is wrong, not the hardware"
        );
        // ...and the reason, isolated: only r == 0 agrees, whatever the two fields hold.
        let d = SmemDesc::for_layout(
            0,
            SmemLayout::RowMajorNone { k_leading: true },
            row_bytes,
            rows,
        );
        for r in 0..8u64 {
            let agrees = d.canonical_offset(0, 0, r, 0, elem) == truth(0, 0, r, 0);
            assert_eq!(agrees, r == 0, "row {r} within the core matrix");
        }

        // 2. The two logged numbers, recomputed.
        let (a, b) = bringup_operands(m64, n64, k64);
        let want = ref_nt(&a, &b, m64, k64, n64);
        for (label, f, expect) in [
            ("k_leading: true", kl, 64usize),
            ("k_leading: false", ml, 0),
        ] {
            let a_seen = unswizzled_read(&a, m64, k64, 64, f);
            let b_seen = unswizzled_read(&b, n64, k64, 64, f);
            let got = ref_nt(&a_seen, &b_seen, m64, k64, n64);
            let exact = got.iter().zip(&want).filter(|(x, y)| x == y).count();
            assert_eq!(
                exact, expect,
                "{label}: the model must reproduce the H100's own exact-lane count \
                 (2026-08-10-h100-s2a-bringup.log stage D)"
            );
        }
        // The 64 is 8 x 8 and not a coincidence: a lane is exact exactly when BOTH operands' rows
        // are the one row in eight the descriptor reads correctly.
        assert_eq!((m64 / 8) * (n64 / 8), 64);

        // 3. The ramps really do move under a wrong reading, so a MATCH would have meant something.
        let a_kl = unswizzled_read(&a, m64, k64, 64, kl);
        let a_ml = unswizzled_read(&a, m64, k64, 64, ml);
        assert_ne!(a_kl, a_ml, "the two readings must fetch different elements");
        assert_ne!(a_kl, a, "and neither of them is the identity");
    }

    /// **The canonical core-matrix layout IS describable** -- the positive half of the law above, and
    /// the reason the sweep has a repack arm at all.
    ///
    /// Staged with every core matrix at 128 contiguous bytes, the ISA's own (LBO, SBO) reproduce the
    /// operand exactly, for both packings and for every `wgmma` K step. If this ever fails, the
    /// hypothesis the sweep is built on is wrong before a single dollar is spent.
    #[test]
    fn the_canonical_core_matrix_layout_is_describable() {
        let (rows, bk, elem) = (64u64, 64u64, 2u64);
        let row_bytes = bk * elem;
        for (k_fast, swapped) in [(true, false), (false, false)] {
            let layout = SmemLayout::CanonicalNone { k_fast, swapped };
            let f = desc_fields(layout, row_bytes, rows);
            let d = SmemDesc::for_layout(0, layout, row_bytes, rows);
            // Where the probe's repack puts element (8i + r, 8j + c): core matrix index, then the
            // 64 elements inside it. Written here from the layout's own definition, not from the
            // descriptor -- if the two agree, the descriptor describes the repack.
            let ncm_k = bk / 8;
            let ncm_m = rows / 8;
            let staged = |i: u64, j: u64, r: u64, c: u64| {
                let cm = if k_fast { i * ncm_k + j } else { j * ncm_m + i };
                (cm * 64 + r * 8 + c) * elem
            };
            for j0 in 0..(bk / 16) {
                for i in 0..ncm_m {
                    for j in 0..2u64 {
                        for r in 0..8u64 {
                            for c in 0..8u64 {
                                let hw = j0 * f.k_step_bytes + d.canonical_offset(i, j, r, c, elem);
                                assert_eq!(
                                    hw,
                                    staged(i, 2 * j0 + j, r, c),
                                    "{} at k-step {j0}, core matrix ({i},{j}), element ({r},{c})",
                                    layout.label()
                                );
                            }
                        }
                    }
                }
            }
            // Four K steps of the advance must land exactly at the end of the tile's K extent, so
            // the base arithmetic covers the staged tile once and does not run off it.
            assert!(f.k_step_bytes * (bk / 16) <= rows * row_bytes);
        }
    }

    /// **The bring-up verdict is `==`, and this is why.** Every operand value is a small integer, so
    /// it survives the host's f32 -> f16 conversion unchanged (f16 holds every integer through 2048),
    /// and [`ramp_radix`] keeps the whole dot product inside [`BRINGUP_EXACT_LIMIT`], where f32 holds
    /// every integer exactly. Tensor-core reassociation therefore cannot move a bit, and a mismatch
    /// on the device is a *fact*, not a tolerance argument.
    #[test]
    fn the_bringup_operands_are_exact_in_f16_and_f32() {
        for (m, n, k) in [
            (64usize, 64usize, 64usize),
            (128, 256, 64),
            (128, 256, 256),
            (128, 256, 1024),
            (129, 257, 176),
            (17, 33, 16),
        ] {
            let w = ramp_radix(k);
            assert!(w.is_power_of_two() && (2..=8).contains(&w), "radix {w}");
            let (a, b) = bringup_operands(m, n, k);
            assert_eq!(a.len(), m * k);
            assert_eq!(b.len(), n * k);
            for (what, v) in [("A", &a), ("B", &b)] {
                for &x in v.iter() {
                    assert_eq!(x.fract(), 0.0, "{what}: {x} is not an integer");
                    assert!(x >= 1.0, "{what}: {x} -- a zero lane distinguishes nothing");
                    assert!(
                        x <= 2048.0,
                        "{what}: {x} is past the largest integer f16 holds exactly"
                    );
                }
            }
            // The worst-case dot product, computed in f64 over the real operands.
            let mut worst = 0f64;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0f64;
                    for kk in 0..k {
                        acc += a[i * k + kk] as f64 * b[j * k + kk] as f64;
                    }
                    worst = worst.max(acc);
                }
            }
            assert!(
                worst <= BRINGUP_EXACT_LIMIT,
                "{m}x{k}x{n}: worst dot product {worst} exceeds the exact-f32 integer limit \
                 {BRINGUP_EXACT_LIMIT} -- the verdict would silently become a tolerance"
            );
            // And the reference really is exact: recomputing it in f32 order changes nothing.
            let r = ref_nt(&a, &b, m, k, n);
            for (i, &c) in r.iter().enumerate() {
                assert_eq!(
                    c.fract(),
                    0.0,
                    "lane {i} of the reference is not an integer"
                );
            }
        }
        // The radix narrows monotonically as K grows, and never below 2.
        let mut last = usize::MAX;
        for k in [16usize, 64, 256, 1024, 4096, 65536, 1 << 20] {
            let w = ramp_radix(k);
            assert!(w <= last, "radix must not widen with K");
            assert!(w >= 2);
            last = w;
        }
    }

    // --- bring-up: the ONE-VISIT descriptor sweep -------------------------------------------------

    /// **The candidate set is derived, complete, and every row says why it is there.**
    ///
    /// The whole value of a sweep is that the next visit needs no third guess, so the set is checked
    /// for the properties that make that true: the two already-scored controls are in it, every arm
    /// that could win is represented, no two rows are the same launch, every label and rationale is
    /// ASCII and non-trivial, and the set is small enough to read.
    #[test]
    fn the_sweep_candidate_set_is_derived_and_complete() {
        let cands = desc_sweep_candidates();
        assert!(
            (8..=40).contains(&cands.len()),
            "a few dozen at most, or the sweep is a spray: {}",
            cands.len()
        );
        // Labels are unique, ASCII, and every row states its reason.
        let mut labels: Vec<&str> = cands.iter().map(|c| c.label).collect();
        let n = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), n, "duplicate candidate label");
        for c in &cands {
            assert!(c.label.is_ascii() && c.why.is_ascii(), "{}", c.label);
            assert!(
                c.why.len() > 40,
                "{}: a candidate with no stated reason is a guess",
                c.label
            );
            c.template()
                .unwrap_or_else(|e| panic!("{} must pack: {e}", c.label));
            // Every candidate's tensor-map swizzle is its descriptor's, translated once.
            assert_eq!(SmemSwizzle::from_tma(c.tma_swizzle()), c.swizzle);
            // The K step must be a multiple of 16, or the descriptor's own `>> 4` truncates the
            // advance and every step after the first reads a different address than intended.
            assert_eq!(c.k_step_bytes % 16, 0, "{}", c.label);
        }
        // No two rows are the same launch: (template, staging, k-step) is the whole input.
        let mut keys: Vec<(u64, u32, u32)> = cands
            .iter()
            .map(|c| (c.template().unwrap(), c.staging as u32, c.k_step_bytes))
            .collect();
        let n = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), n, "two candidates are the same launch");

        // The two CONTROLS: the readings the H100 already scored, so the sweep can validate itself.
        for want in ["ctl/rowmajor-k-leading", "ctl/rowmajor-mn-leading"] {
            let c = cands.iter().find(|c| c.label == want).expect(want);
            assert_eq!(c.staging, SweepStaging::AsWritten);
            assert_eq!(c.swizzle, SmemSwizzle::None);
            assert_eq!(c.encoding, FieldEncoding::Standard);
        }
        assert_eq!(
            cands
                .iter()
                .find(|c| c.label == "ctl/rowmajor-k-leading")
                .unwrap()
                .template()
                .unwrap(),
            (1u64 << 16) | (64u64 << 32),
            "the shipped-until-2026-08-10 template, hand-computed"
        );

        // Every arm that could win is represented, and each staging mode points its descriptors at
        // the region that mode actually writes.
        let arms: std::collections::BTreeSet<String> = cands.iter().map(|c| c.arm()).collect();
        assert!(arms.len() >= 4, "arms: {arms:?}");
        for c in &cands {
            let (a, b) = c.operand_offsets();
            let repacked = c.staging != SweepStaging::AsWritten;
            assert_eq!(repacked, a == DESC_SWEEP_A_ALT, "{}", c.label);
            assert_eq!(repacked, b == DESC_SWEEP_B_ALT, "{}", c.label);
            assert_ne!(a, b);
        }

        // Exactly one candidate carries the reading production ships, and it is production-viable.
        let shipped: Vec<&DescCandidate> = cands
            .iter()
            .filter(|c| c.layout == Some(SHIPPED_LAYOUT))
            .collect();
        assert_eq!(
            shipped.len(),
            1,
            "the sweep must carry the shipped reading exactly once, so the round can say `no edit \
             needed`: {:?}",
            shipped.iter().map(|c| c.label).collect::<Vec<_>>()
        );
        assert!(shipped[0].production_viable());
        // A repack arm is never production-viable, whatever it scores: TMA cannot write it.
        for c in cands
            .iter()
            .filter(|c| c.staging != SweepStaging::AsWritten)
        {
            assert!(!c.production_viable(), "{}", c.label);
        }
        // ...and every production-viable row names a layout `wgmma_module` will actually emit.
        let lic = license();
        for c in cands.iter().filter(|c| c.production_viable()) {
            let cfg = WgmmaCfg {
                layout: c.layout.unwrap(),
                ..WGMMA_W1
            };
            wgmma_module(&cfg, &lic).unwrap_or_else(|e| {
                panic!(
                    "{}: a production-viable candidate must be generatable, or crowning it settles \
                     nothing: {e}",
                    c.label
                )
            });
        }
    }

    /// **The sweep is ONE module and a host loop** -- the property that makes it a single visit.
    ///
    /// Everything a candidate varies must arrive as a run-time parameter, so the generated text is
    /// identical for all of them. If a field ever leaks into the PTX as an immediate, this fails.
    #[test]
    fn the_sweep_probe_is_one_module_for_every_candidate() {
        let ptx = desc_sweep_probe_module(&license()).unwrap();
        assert!(ptx.is_ascii(), "PTX must be pure ASCII");
        assert!(ptx.starts_with(HDR_SM90A_V80));
        assert!(ptx.contains(WGMMA_DSMEM_DECL));
        assert_eq!(
            ptx.matches(&format!(".visible .entry {DESC_SWEEP_ENTRY}("))
                .count(),
            1
        );
        assert_eq!(ptx.matches('{').count(), ptx.matches('}').count());
        // The ten parameters, in the order the launcher pushes them.
        assert_eq!(
            ptx.matches(".param ").count(),
            10,
            "(K, C, descA, descB, aOff, bOff, kStep, stage, tmapA, tmapB)"
        );
        for p in [
            "pK", "pC", "pDescA", "pDescB", "pAOff", "pBOff", "pKStep", "pStage",
        ] {
            assert!(ptx.contains(&format!("[{p}]")), "{p} is never loaded");
        }
        // Not one descriptor immediate anywhere: both templates are OR'd in from a register.
        assert_eq!(ptx.matches("or.b64 %descA,%rdT,%tmplA;").count(), 1);
        assert_eq!(ptx.matches("or.b64 %descB,%rdT,%tmplB;").count(), 1);
        assert!(
            !ptx.contains("or.b64 %descA,%rdT,0x"),
            "a baked descriptor constant would make the sweep one module per candidate"
        );
        // One wgmma, in a run-time K loop, so K=16 and K=64 are the same module.
        assert_eq!(
            ptx.matches("wgmma.mma_async.sync.aligned.").count(),
            1,
            "the K loop is a real loop; unrolling it would bake the k-step advance"
        );
        assert!(ptx.contains(&format!(
            "wgmma.mma_async.sync.aligned.m64n{DESC_SWEEP_N}k16.f32.f16.f16"
        )));
        assert_eq!(ptx.matches("wgmma.fence.sync.aligned;").count(), 1);
        assert_eq!(ptx.matches("wgmma.commit_group.sync.aligned;").count(), 1);
        assert_eq!(ptx.matches("wgmma.wait_group.sync.aligned 0;").count(), 1);
        // The staging path is stage B's, unchanged: one barrier, two tiled copies, one transaction.
        assert_eq!(
            ptx.matches(
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            )
            .count(),
            2
        );
        assert!(ptx.contains(&format!(
            "mbarrier.arrive.expect_tx.shared::cta.b64 %rdSt,[%rdBar],{};",
            desc_sweep_tx_bytes()
        )));
        assert_eq!(
            ptx.matches("mbarrier.init.shared::cta.b64 [%rdBar],1;")
                .count(),
            1
        );
        // The repack writes SMEM with the GENERIC proxy; wgmma reads it through the ASYNC proxy.
        // Without this fence the canonical arms would be a race, not a measurement.
        assert_eq!(ptx.matches("fence.proxy.async.shared::cta;").count(), 1);
        // No warp specialisation anywhere: the probe is one warpgroup and nothing else.
        for banned in ["setmaxnreg", "multicast", "cluster_ctarank"] {
            assert!(
                !ptx.contains(banned),
                "the sweep probe must not contain {banned}"
            );
        }
        assert!(ptx.contains(&format!(".maxntid {DESC_SWEEP_THREADS}, 1, 1")));
        assert_eq!(ptx.matches(".param .align 64 .b8 tmap").count(), 2);
        assert_eq!(ptx.matches("cvta.param.u64").count(), 2);
        // Every accumulator is stored exactly once.
        let nacc = DESC_SWEEP_N / 2;
        assert!(ptx.contains(&format!(".reg .f32 %acc<{nacc}>;")));
        assert_eq!(ptx.matches("st.global.f32").count(), nacc);
        for i in 0..nacc {
            assert_eq!(ptx.matches(&format!("],%acc{i};")).count(), 1, "%acc{i}");
        }
        // Every branch target is defined, branched to, and entry-scoped (PTX labels are
        // module-scoped, so an un-prefixed one collides the day two entries share a module).
        let defined: Vec<&str> = ptx
            .lines()
            .filter(|l| !l.starts_with(' ') && l.ends_with(':'))
            .map(|l| l.trim_end_matches(':'))
            .collect();
        let used: Vec<&str> = ptx
            .lines()
            .filter_map(|l| {
                l.find("bra ")
                    .map(|i| l[i + 4..].trim().trim_end_matches(';'))
            })
            .collect();
        assert!(!used.is_empty());
        for u in &used {
            assert!(defined.contains(u), "branch to undefined `{u}`");
            assert!(
                u.ends_with(DESC_SWEEP_ENTRY),
                "label `{u}` is not entry-scoped"
            );
        }
        for d in &defined {
            assert!(
                used.contains(d),
                "label `{d}` is defined but never branched to"
            );
        }
    }

    /// **No candidate may read outside the shared window** -- the law that keeps a deliberately
    /// wrong descriptor from ending the round instead of scoring zero.
    ///
    /// A wrong (LBO, SBO) reads a wrong ADDRESS, not a wrong element, and an out-of-window shared
    /// access is an illegal-address fault that makes the CUDA context sticky-errored (crate landmine
    /// 6): every later candidate, and every later stage, would then fail identically and the round
    /// would report a cascade with one real cause. So the reach of every row is bounded here, on a
    /// CPU, before a dollar is spent.
    #[test]
    fn every_sweep_candidate_reads_inside_the_window() {
        let mut worst = (0u64, "");
        for c in desc_sweep_candidates() {
            let reach = c.max_reach().unwrap_or_else(|e| panic!("{}: {e}", c.label));
            let (a, b) = c.operand_offsets();
            let far = a.max(b) as u64 + reach;
            assert!(
                far <= DESC_SWEEP_SMEM as u64,
                "{}: reaches {far} B, past the {DESC_SWEEP_SMEM} B window -- that is an illegal \
                 shared address, which wedges the context rather than scoring zero",
                c.label
            );
            if reach > worst.0 {
                worst = (reach, c.label);
            }
        }
        // The widest row really is the raw-encoded one, and it really does reach far: if this ever
        // stops being true the window may have been shrunk for the wrong reason.
        assert_eq!(worst.1, "canon-kfast/raw-fields");
        assert!(
            worst.0 > 64 * 1024,
            "the raw-field candidate is the reason the window is 144 KiB; it reached only {} B",
            worst.0
        );
    }

    /// The sweep's shared-memory map: four tiles and a barrier, every base aligned for the widest
    /// swizzle any candidate asks for, and the whole window inside one CTA's budget.
    #[test]
    fn the_sweep_probe_shared_memory_map_is_aligned_and_bounded() {
        assert_eq!(DESC_SWEEP_TILE, 64 * 64 * 2);
        assert_eq!(
            [
                DESC_SWEEP_A_RAW,
                DESC_SWEEP_B_RAW,
                DESC_SWEEP_A_ALT,
                DESC_SWEEP_B_ALT
            ],
            [0, 8192, 16384, 24576]
        );
        for base in [
            DESC_SWEEP_A_RAW,
            DESC_SWEEP_B_RAW,
            DESC_SWEEP_A_ALT,
            DESC_SWEEP_B_ALT,
        ] {
            assert_eq!(
                base as u64 % SmemSwizzle::B128.required_alignment(),
                0,
                "every operand base must clear the 128-B swizzle's 1024-byte pattern boundary"
            );
        }
        assert_eq!(DESC_SWEEP_BAR % 8, 0);
        const _: () = assert!(DESC_SWEEP_BAR + 8 <= DESC_SWEEP_SMEM);
        const _: () = assert!(DESC_SWEEP_SMEM <= HOPPER_SMEM_PER_CTA);
        let p = desc_sweep_plan();
        assert_eq!(p.entry, DESC_SWEEP_ENTRY);
        assert_eq!(p.module_key, DESC_SWEEP_KEY);
        assert_eq!(p.block, (128, 1, 1));
        assert_eq!(p.dyn_smem_bytes, DESC_SWEEP_SMEM);
        assert_eq!(p.tx_bytes, 2 * DESC_SWEEP_TILE);
        assert_eq!(p.operand, (64, 64));
        // The transaction the kernel declares equals what the two tensor maps move -- a mismatch
        // here does not fail, it HANGS.
        let map = crate::tma_host::TensorMapArgs::tiled_2d_row_major(
            TmaDataType::F16,
            DESC_SWEEP_M as u64,
            DESC_SWEEP_BK as u64,
            DESC_SWEEP_BK as u64,
            DESC_SWEEP_M as u32,
            DESC_SWEEP_BK as u32,
            TmaSwizzle::None,
        );
        map.validate().unwrap();
        assert_eq!(2 * map.transaction_bytes(), p.tx_bytes);
        // ...and the same geometry is legal at the 128-B swizzle, which is the arm that matters.
        let mut sw = map;
        sw.swizzle = TmaSwizzle::B128;
        sw.validate()
            .expect("BK=64 f16 is exactly the 128-B atom, which is why the shipped BK is 64");
        // Both K passes issue a whole number of wgmma steps, and K=16 issues exactly one -- the
        // pass in which the descriptor's start address never moves.
        assert_eq!(DESC_SWEEP_KS, &[16, 64]);
        assert_eq!(DESC_SWEEP_KS[0] / WgmmaShape::K, 1);
        assert_eq!(DESC_SWEEP_KS[1] / WgmmaShape::K, 4);
        for k in DESC_SWEEP_KS {
            assert_eq!(k % WgmmaShape::K, 0);
            assert!(*k <= DESC_SWEEP_BK);
        }
    }

    // --- bring-up: the single-stage TMA probe -----------------------------------------------------

    /// The probe is the *simplest possible* consumer of a tensor map, and its value is entirely in
    /// what it does NOT contain: no `wgmma`, no `setmaxnreg`, no second barrier, no pipeline. If the
    /// probe grew any of that it would stop isolating the descriptor, which is its only job.
    #[test]
    fn the_tma_stage_probe_is_structurally_a_single_stage_load() {
        let ptx = tma_stage_probe_module(&license()).unwrap();
        assert!(ptx.is_ascii(), "PTX must be pure ASCII");
        assert!(ptx.starts_with(HDR_SM90A_V80));
        assert!(ptx.contains(WGMMA_DSMEM_DECL));
        assert_eq!(
            ptx.matches(&format!(".visible .entry {TMA_PROBE_ENTRY}("))
                .count(),
            1
        );
        assert_eq!(ptx.matches('{').count(), ptx.matches('}').count());
        // exactly one copy, one barrier, one transaction declaration -- and the transaction is a
        // REGISTER, which is what lets one module and one cache key cover every geometry.
        assert_eq!(
            ptx.matches(
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            )
            .count(),
            1
        );
        assert_eq!(
            ptx.matches("mbarrier.init.shared::cta.b64 [%rdBar],1;")
                .count(),
            1
        );
        assert_eq!(
            ptx.matches("mbarrier.arrive.expect_tx.shared::cta.b64 %rdSt,[%rdBar],%tx;")
                .count(),
            1,
            "the transaction count must be the run-time parameter, not an immediate"
        );
        // ...and the phase parity is a REGISTER, set before any branch. The probe's operand forms are
        // exactly the two the 2026-08-10 ptxas census already assembled in the mainloop, so stage A
        // of the bring-up cannot fail on an operand class nothing has ever put through an assembler.
        assert_eq!(
            ptx.matches("mbarrier.try_wait.parity.shared::cta.b64 %p1,[%rdBar],%ph;")
                .count(),
            1
        );
        let (before_branch, _) = ptx
            .split_once("    @!%p0 bra ")
            .expect("the probe branches on thread 0");
        assert!(
            before_branch.contains("    mov.u32 %ph,0;\n"),
            "the parity register must be initialised before the first branch, or the threads that \
             take it read an undefined register"
        );
        assert_eq!(ptx.matches("bar.sync 0;").count(), 1);
        for banned in ["wgmma", "setmaxnreg", "multicast"] {
            assert!(!ptx.contains(banned), "the probe must not contain {banned}");
        }
        // the by-value tensor map and its generic address
        assert_eq!(ptx.matches(".param .align 64 .b8 tmap").count(), 1);
        assert_eq!(ptx.matches("cvta.param.u64").count(), 1);
        assert_eq!(ptx.matches(".param ").count(), 5, "(tx, c0, c1, out, tmap)");
        assert!(ptx.contains(&format!(".maxntid {TMA_PROBE_THREADS}, 1, 1")));
        // Every branch target is defined, is branched to, and is entry-scoped (PTX labels are
        // module-scoped, so an un-prefixed one collides the day two entries share a module).
        let defined: Vec<&str> = ptx
            .lines()
            .filter(|l| !l.starts_with(' ') && l.ends_with(':'))
            .map(|l| l.trim_end_matches(':'))
            .collect();
        let used: Vec<&str> = ptx
            .lines()
            .filter_map(|l| {
                l.find("bra ")
                    .map(|i| l[i + 4..].trim().trim_end_matches(';'))
            })
            .collect();
        assert_eq!(defined.len(), 4, "INITED, WAIT, COPY, EXIT");
        assert!(!used.is_empty());
        for u in &used {
            assert!(defined.contains(u), "branch to undefined `{u}`");
            assert!(
                u.ends_with(TMA_PROBE_ENTRY),
                "label `{u}` is not entry-scoped"
            );
        }
        for d in &defined {
            assert!(
                used.contains(d),
                "label `{d}` is defined but never branched to"
            );
        }
    }

    /// The probe's plan is derived from the transaction, and a transaction that could not have come
    /// from a legal TMA box declines rather than being rounded into one.
    #[test]
    fn tma_probe_plan_matches_the_geometry_it_probes() {
        for c in WGMMA_VARIANTS {
            for (what, tx) in [("A", c.tile_a_bytes()), ("B", c.tile_b_bytes())] {
                let p = tma_probe_plan(tx).unwrap_or_else(|e| panic!("{} {what}: {e}", c.name));
                assert_eq!(p.tx_bytes, tx);
                assert_eq!(p.dyn_smem_bytes, tx + 8);
                assert_eq!(p.entry, TMA_PROBE_ENTRY);
                assert_eq!(p.module_key, TMA_PROBE_KEY);
                assert_eq!(p.block, (TMA_PROBE_THREADS, 1, 1));
                assert!(p.dyn_smem_bytes <= HOPPER_SMEM_PER_CTA);
            }
            // The A and B transactions are exactly what the mainloop declares between them.
            assert_eq!(
                tma_probe_plan(c.tile_a_bytes()).unwrap().tx_bytes
                    + tma_probe_plan(c.tile_b_bytes()).unwrap().tx_bytes,
                c.stage_tx_bytes()
            );
        }
        for bad in [0usize, 8, 24, HOPPER_SMEM_PER_CTA] {
            let e = tma_probe_plan(bad).unwrap_err();
            assert!(e.starts_with(UNSUPPORTED), "{bad}: {e}");
        }
    }

    #[test]
    fn variant_lookup_finds_every_shipped_row() {
        for c in WGMMA_VARIANTS {
            assert_eq!(wgmma_variant(c.name).name, c.name);
        }
    }

    #[test]
    #[should_panic(expected = "unknown wgmma variant")]
    fn variant_lookup_panics_rather_than_mis_dispatching() {
        wgmma_variant("wgmma_nt_f16_nope");
    }

    /// Print a variant's PTX, for eyeballing it and for feeding it to a real `ptxas` on a machine
    /// that has one. Not a gate -- `#[ignore]`d like every other inspection helper in this crate.
    ///
    /// `WUKONG_WGMMA_VARIANT` names any shipped or bring-up row, or [`TMA_PROBE_ENTRY`] for the
    /// single-stage TMA probe -- the module the H100 round loads first, and the one worth reading by
    /// eye before it is loaded anywhere.
    ///
    /// ```text
    /// cargo test -p wukong_codegen_gpu --lib dump_wgmma_ptx -- --ignored --nocapture
    /// WUKONG_WGMMA_VARIANT=wk_tma_stage_probe cargo test ... dump_wgmma_ptx -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "inspection helper, not a gate"]
    fn dump_wgmma_ptx() {
        let name = std::env::var("WUKONG_WGMMA_VARIANT").unwrap_or_else(|_| WGMMA_W1.name.into());
        if name == TMA_PROBE_ENTRY {
            println!("{}", tma_stage_probe_module(&license()).unwrap());
            return;
        }
        if name == DESC_SWEEP_ENTRY {
            println!("{}", desc_sweep_probe_module(&license()).unwrap());
            for c in desc_sweep_candidates() {
                println!(
                    "// candidate {:<28} arm {:<14} tmpl {:#018x} aOff {:>6} kStep {:>5} \
                     stage {} reach {:>7}",
                    c.label,
                    c.arm(),
                    c.template().unwrap(),
                    c.operand_offsets().0,
                    c.k_step_bytes,
                    c.staging as u32,
                    c.max_reach().unwrap()
                );
            }
            return;
        }
        let cfg = wgmma_variant(&name);
        println!("{}", wgmma_module(cfg, &license()).unwrap());
    }

    /// The window declaration is duplicated from `gpu::DSMEM_DECL` so this module can stay un-gated;
    /// the duplication is only safe while the two agree on the symbol. The alignment deliberately
    /// differs (1024 here, 16 there) because a TMA destination and a 128-B swizzle pattern both need
    /// more than 16.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_window_declaration_agrees_with_the_host_side_constant() {
        assert_eq!(WGMMA_DSMEM_SYM, crate::gpu::DSMEM_SYM);
        assert!(crate::gpu::DSMEM_DECL.contains(WGMMA_DSMEM_SYM));
        assert!(WGMMA_DSMEM_DECL.contains(".extern .shared .align 1024 .b8 wk_dsmem[];"));
        // Every shipped variant needs the dynamic window: all of them are far past the 48 KiB static
        // ceiling, which is a PTX ISA rule on every non-`a` target.
        for c in WGMMA_VARIANTS {
            assert!(c.smem_bytes() > crate::gpu::STATIC_SMEM_CAP);
            assert!(crate::gpu::smem_mode_for(c.smem_bytes()).is_dynamic());
        }
    }
}
