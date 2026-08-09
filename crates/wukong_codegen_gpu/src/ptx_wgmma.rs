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
//! * **Cluster multicast** (`.multicast::cluster`, D1's W1 as specified). Multicasting the A tile to
//!   both CTAs of a 2x1x1 cluster halves A's L2 traffic, but it makes the *empty* half of the
//!   pipeline cluster-scoped: consumers in CTA 1 must signal a barrier in CTA 0 through `mapa`, and
//!   the CTA mask has to be derived from `%cluster_ctarank`. That is real machinery with a deadlock
//!   as its failure mode, and none of it can be executed here. [`Multicast::ClusterA`] therefore
//!   returns an [`UNSUPPORTED`] decline instead of plausible PTX.
//! * **The 128-B SMEM swizzle.** TMA can write the swizzled layout and the descriptor has a field for
//!   it, but the leading/stride byte offsets a swizzled operand needs are *not* the plain
//!   core-matrix distances, and this module will not guess them. [`SmemSwizzle::B128`] is fully
//!   supported by the *packer* (and tested) and declined by the *generator*.
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
    /// Leading-dimension byte offset -- see [`DescOrder`].
    pub lbo: u64,
    /// Stride-dimension byte offset -- see [`DescOrder`].
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

/// Which geometric distance goes in the descriptor's **leading** field.
///
/// A K-major 16-bit operand in shared memory is a grid of `8 row x 16 byte` core matrices, and the
/// hardware reconstructs core matrix `(i, j)` -- row group `i`, k group `j` -- as
/// `start + i * <one offset> + j * <the other offset>`. The two distances for a tile whose rows are
/// `row_bytes` apart are therefore `16` (k-adjacent) and `8 * row_bytes` (row-group-adjacent), and
/// the only open question is which of them the ISA calls "leading".
///
/// **This is UNCONFIRMED on silicon and it is the first thing an H100 must settle.** The reading
/// this module ships, [`DescOrder::KLeading`], follows the ISA's own no-swizzle figure, in which a
/// densely packed 64x16 A tile has leading offset 128 B (one core matrix, i.e. k-adjacent) and stride
/// offset 256 B (two core matrices, i.e. row-adjacent). If that reading is wrong the two fields are
/// simply swapped, and the failure mode is **silently wrong results, not a JIT error** -- so it is a
/// config field rather than a constant, and flipping it is a one-token A/B on the first Hopper run
/// rather than a code change. See `WGMMA_DEVICE_VALIDATION` item 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescOrder {
    /// Leading = the K direction (16 B between k-adjacent core matrices). The shipped reading.
    KLeading,
    /// Leading = the M/N direction (`8 * row_bytes` between row-group-adjacent core matrices).
    MnLeading,
}

impl SmemDesc {
    /// The descriptor for a **K-major tile of 16-bit elements** whose rows are `row_bytes` apart --
    /// i.e. exactly what a TMA tiled copy of a row-major `rows x bk` operand leaves in shared memory.
    ///
    /// Both GEMM operands of an NT product have this form: A is `M x K` with K contiguous, and B is
    /// `N x K` with K contiguous, which is `B(K x N)` in column-major -- and column-major B is what
    /// `wgmma` expects at `imm-trans-b = 0`. So the NT layout this backend already stores needs no
    /// transpose flag on either operand, and both descriptors are built by this one constructor.
    pub fn k_major(
        start_addr: u64,
        row_bytes: u64,
        order: DescOrder,
        swizzle: SmemSwizzle,
    ) -> Self {
        let k_adjacent = 16;
        let row_group_adjacent = 8 * row_bytes;
        let (lbo, sbo) = match order {
            DescOrder::KLeading => (k_adjacent, row_group_adjacent),
            DescOrder::MnLeading => (row_group_adjacent, k_adjacent),
        };
        Self {
            start_addr,
            lbo,
            sbo,
            base_offset: 0,
            swizzle,
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
    /// D1's W1 as specified: a 2x1x1 cluster with `.multicast::cluster` on A. **Declined** -- see the
    /// module docs.
    ClusterA,
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
    pub swizzle: SmemSwizzle,
    pub desc_order: DescOrder,
    pub schedule: Schedule,
    pub multicast: Multicast,
}

impl WgmmaCfg {
    pub const fn threads(&self) -> usize {
        (1 + self.consumer_wgs) * WARPGROUP_THREADS
    }
    /// Bytes of one A tile (one stage).
    pub const fn tile_a_bytes(&self) -> usize {
        self.bm * self.bk * self.dtype.size()
    }
    /// Bytes of one B tile (one stage).
    pub const fn tile_b_bytes(&self) -> usize {
        self.bn * self.bk * self.dtype.size()
    }
    /// Bytes one TMA stage moves -- the `expect_tx` count. A + B together, since one barrier covers
    /// both copies.
    pub const fn stage_tx_bytes(&self) -> usize {
        self.tile_a_bytes() + self.tile_b_bytes()
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
    pub fn tensor_map_a(&self, m: usize, k: usize) -> TensorMapArgs {
        TensorMapArgs::tiled_2d_row_major(
            self.dtype.tma(),
            m as u64,
            k as u64,
            k as u64,
            self.bm as u32,
            self.bk as u32,
            self.swizzle.to_tma(),
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
            self.swizzle.to_tma(),
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
        let align = self.swizzle.required_alignment() as usize;
        for (what, bytes) in [
            ("A tile", self.tile_a_bytes()),
            ("B tile", self.tile_b_bytes()),
        ] {
            if !bytes.is_multiple_of(align) {
                return Err(format!(
                    "{UNSUPPORTED}: {}: one {what} is {bytes} B, not a multiple of the {align} B \
                     alignment {:?} requires, so stage bases would drift out of alignment",
                    self.name, self.swizzle
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
        if self.multicast != Multicast::None {
            return Err(format!(
                "{UNSUPPORTED}: {}: cluster multicast ({:?}) is not implemented. It needs \
                 cluster-scoped empty-barrier arrivals through `mapa.shared::cluster`, a CTA mask \
                 derived from %cluster_ctarank, and a `.reqnctapercluster` entry directive; none of \
                 it can be executed on this machine and its failure mode is a deadlock. Run the \
                 non-multicast arm first.",
                self.name, self.multicast
            ));
        }
        if self.swizzle != SmemSwizzle::None {
            return Err(format!(
                "{UNSUPPORTED}: {}: {:?} shared-memory swizzle is not implemented. The packer \
                 supports every mode, but a swizzled operand's leading/stride byte offsets are not \
                 the plain core-matrix distances `SmemDesc::k_major` derives, and this generator \
                 will not guess them.",
                self.name, self.swizzle
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
}

impl LaunchPlan {
    /// The CTA grid for an `M x N` output: x indexes N tiles, y indexes M tiles.
    ///
    /// Ragged edges need no special case. TMA fills out-of-range elements with zero
    /// ([`crate::tma_host::TmaOobFill::Zero`]), so a partial tile accumulates nothing, and the
    /// epilogue predicates every store on `row < M && col < N`.
    pub fn grid(&self, m: usize, n: usize) -> (u32, u32, u32) {
        (n.div_ceil(self.bn) as u32, m.div_ceil(self.bm) as u32, 1)
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
/// **This row is the full form minus cluster multicast** ([`Multicast::None`]), which is the right
/// bring-up order anyway: multicast is a traffic optimisation whose failure mode is a deadlock, and
/// the kernel is correct without it.
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
    swizzle: SmemSwizzle::None,
    desc_order: DescOrder::KLeading,
    schedule: Schedule::Cooperative,
    multicast: Multicast::None,
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

/// Every shipped configuration. The generator, the gates and the device-free module enumeration all
/// iterate this table.
pub const WGMMA_VARIANTS: &[WgmmaCfg] = &[WGMMA_W1, WGMMA_W1_BF16, WGMMA_W3C];

/// Look up a variant by entry name; panics loudly rather than mis-dispatching.
pub fn wgmma_variant(name: &str) -> &'static WgmmaCfg {
    WGMMA_VARIANTS
        .iter()
        .find(|v| v.name == name)
        .unwrap_or_else(|| panic!("unknown wgmma variant {name:?}"))
}

/// **What the first H100 hour must confirm, in priority order** -- this family's own list of claims
/// no test in this repo can reach, kept as data so a bring-up script can print it.
///
/// There is no Hopper part here and `sm_90a` cannot be JITed, emulated or `ptxas`-checked anywhere in
/// this environment, so everything below is unproven text until it runs. Work down the list; each
/// item's failure mode is stated because several of them are silent.
pub const WGMMA_DEVICE_VALIDATION: &[&str] = &[
    "1. DescOrder. Run W1 at M=N=K=64 against an f64 reference with distinguishable ramps in A and \
     B. If it is wrong, flip `desc_order` to DescOrder::MnLeading and rerun. Nothing downstream \
     means anything until this is settled, and the wrong choice is silently wrong data, not a JIT \
     error: it is the one place the ISA's naming of the descriptor's two offset fields was read \
     from a figure rather than confirmed.",
    "2. The module loads at all -- cuModuleLoadData on the generated text. First check of the \
     `.target sm_90a` header, of `.version 8.0`, and of every instruction spelling in the family.",
    "3. setmaxnreg and the register split -- cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS) plus a \
     spill check (ptxas -v via WUKONG_PTXAS, or the JIT log). If ptxas cannot fit a consumer's 128 \
     accumulators plus addressing inside 232 registers, consumer_regs must rise and producer_regs \
     fall; the budget assert in WgmmaCfg::validate will keep the pair honest.",
    "4. The TMA descriptors -- cuTensorMapEncodeTiled succeeding for the A and B geometries at each \
     benched shape, then a single-stage load compared against a host copy of the same tile.",
    "5. The pipeline does not hang. The expect_tx count is WgmmaCfg::stage_tx_bytes; if it \
     disagrees with what the two copies actually move, the barrier never completes and every \
     consumer waits forever. Time-box the first launch.",
    "6. The producer warpgroup returns with copies possibly still in flight (CUTLASS's producer \
     tail exists for barrier lifetime, which a CTA whose consumers are still running does not \
     need). Confirm no hang and no early SMEM reclaim.",
    "7. Ragged shapes -- M, N and K each not a multiple of the tile, against the f64 reference, to \
     confirm TMA's zero fill and the predicated epilogue together cover the edges.",
    "8. Only then, performance: same-run adjacent A/B against cuBLAS at 4096 and 8192 cubed, with \
     bench_instrument's twin control passing.",
];

// --- the generator --------------------------------------------------------------------------------

/// Generate the PTX module for one configuration.
///
/// Requires an [`Sm90aLicense`]: `sm_90a` text cannot be produced by a path that has not established
/// the device is Hopper. Returns `Err` with an [`UNSUPPORTED`]-prefixed message for any shape this
/// generator cannot express -- **never** plausible-but-wrong PTX.
pub fn wgmma_module(cfg: &WgmmaCfg, _license: &Sm90aLicense) -> Result<String, String> {
    let shape = cfg.validate()?;
    let mut m = String::from(HDR_SM90A_V80);
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

    // The two descriptor constants: everything but the start address, which the kernel folds in at
    // run time. A and B share a constructor because both operands of an NT product are K-major with
    // the same row pitch.
    let const_a = SmemDesc::k_major(0, row_bytes, cfg.desc_order, cfg.swizzle).const_part()?;
    let const_b = const_a;

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
         \x20   .param .u64 pC,\n    {},\n    {}\n)\n.maxntid {threads}, 1, 1\n{{\n",
        ptx_param_decl("tmapA"),
        ptx_param_decl("tmapB")
    );

    // Registers. NB: never name one %tid/%ctaid/%laneid etc -- those are PTX special registers.
    s += "    .reg .pred %p0,%p1,%p2,%q0,%q1,%q2,%q3,%pd0,%pd1,%ptrue,%pfirst;\n";
    s += "    .reg .b32 %M,%N,%K,%lin,%wgi,%lane,%wrp,%kt,%ktiles,%stg,%phf,%phe,%tmp,%tmp2,\
          %row0,%row1,%col,%col1,%colb,%ctam,%ctan,%cwg;\n";
    s += "    .reg .b64 %rdC,%rdS,%rdT,%rdA,%rdB,%rdBar,%rdBarF,%rdTmA,%rdTmB,%rdAddr,%rdOffA,\
          %descA,%descB,%rdSt;\n";
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
            cfg.consumer_wgs
        );
    }
    s += &format!("INIT_DONE_{name}:\n    bar.sync 0;\n");

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
    s += &format!("    mul.wide.u32 %rdT,%stg,{tile_b};\n    add.s64 %rdB,%rdS,%rdT;\n");
    s += &format!("    add.s64 %rdB,%rdB,{};\n", cfg.b_off(0));
    // The K coordinate, in elements. Tensor coordinates are `{dim0, dim1}` = `{k, row}`, because
    // dimension 0 is the contiguous axis of the descriptor (see `tma_host`).
    s += &format!("    mul.lo.s32 %tmp,%kt,{bk};\n");
    s += "    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes \
          [%rdA],[%rdTmA,{%tmp,%ctam}],[%rdBarF];\n";
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
        let koff = j * WgmmaShape::K * cfg.dtype.size();
        for (reg, base, cst) in [("%descA", "%rdA", const_a), ("%descB", "%rdB", const_b)] {
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
    s += "    @%p2 mbarrier.arrive.shared::cta.b64 %rdSt,[%rdBar];\n";
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
    s += &format!("EXIT_{name}:\n    ret;\n}}\n");
    Ok(s)
}

/// **Every PTX module this family emits, device-free** -- the corpus its own `.version`/`.target`
/// and ASCII laws scan.
///
/// It is shaped as a `Vec<(String, String)>` on purpose: that is the exact shape of
/// `gpu.rs`'s `device_free_modules()`, so the wiring commit splices this family into the crate-wide
/// `.version` law with one line (`v.extend(crate::ptx_wgmma::wgmma_device_free_modules());`) and one
/// number (`EXPECTED_MODULES += 3`). Until then, `the_family_declares_no_floor_it_does_not_need`
/// below applies the identical rule over the identical corpus, so the family is covered rather than
/// invisible.
pub fn wgmma_device_free_modules() -> Vec<(String, String)> {
    let license = Sm90aLicense::for_probed_cc((9, 0)).expect("(9,0) is Hopper");
    WGMMA_VARIANTS
        .iter()
        .map(|c| {
            let ptx = wgmma_module(c, &license)
                .unwrap_or_else(|e| panic!("shipped variant {} must generate: {e}", c.name));
            (format!("wgmma::{}", c.name), ptx)
        })
        .collect()
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
    /// takes an `&Sm90aLicense`, so ungated `sm_90a` PTX does not compile. This scan closes the one
    /// remaining hole -- a *new* function in this file that builds a header string itself instead of
    /// going through the generator -- by requiring that any function naming the `sm_90a` header also
    /// names the license.
    #[test]
    fn every_sm90a_emitter_demands_the_license() {
        let src = include_str!("ptx_wgmma.rs");
        let code = src
            .split("\n#[cfg(test)]")
            .next()
            .expect("the test module is cut off at its column-0 attribute");
        // The header constant is referenced exactly once outside the tests: in `wgmma_module`, whose
        // signature carries the license.
        let uses: Vec<&str> = code
            .lines()
            .filter(|l| l.contains("HDR_SM90A_V80") && !l.trim_start().starts_with("//"))
            .collect();
        assert_eq!(
            uses.len(),
            2,
            "expected exactly the `use` and the single emission site; found {uses:#?}"
        );
        let generator = code
            .split_once("pub fn wgmma_module(")
            .expect("wgmma_module must exist")
            .1;
        let sig = generator.split_once(')').unwrap().0;
        assert!(
            sig.contains("Sm90aLicense"),
            "the only sm_90a emitter must take a capability witness: {sig}"
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

    /// The K-major constructor's two distances, and the one thing about them that is not yet known.
    #[test]
    fn k_major_derives_the_core_matrix_distances() {
        // BK=64 f16 -> 128-byte rows. A core matrix is 8 rows x 16 bytes, so k-adjacent core matrices
        // are 16 B apart and row-group-adjacent ones are 8 * 128 = 1024 B apart.
        let k = SmemDesc::k_major(0, 128, DescOrder::KLeading, SmemSwizzle::None);
        assert_eq!((k.lbo, k.sbo), (16, 1024));
        let m = SmemDesc::k_major(0, 128, DescOrder::MnLeading, SmemSwizzle::None);
        assert_eq!((m.lbo, m.sbo), (1024, 16));
        assert_ne!(
            k.pack().unwrap(),
            m.pack().unwrap(),
            "the two readings are distinguishable, which is why the H100 A/B is one token"
        );
        // The shipped constant, hand-computed: lbo 16 -> 1 at bit 16; sbo 1024 -> 64 at bit 32.
        assert_eq!(k.const_part().unwrap(), (1u64 << 16) | (64u64 << 32));
        assert_eq!(k.const_part().unwrap(), 0x40_0001_0000);
    }

    /// The kernel folds the runtime stage address into the descriptor with
    /// `((addr >> 4) & 0x3FFF) | const_part`. That must equal what the host encoder would produce for
    /// the same address -- for every address a 227 KiB shared window can hold.
    #[test]
    fn the_runtime_address_fold_matches_the_host_encoder() {
        let cst = SmemDesc::k_major(0, 128, DescOrder::KLeading, SmemSwizzle::None)
            .const_part()
            .unwrap();
        let mut addr = 0u64;
        while addr < HOPPER_SMEM_PER_CTA as u64 {
            let host = SmemDesc::k_major(addr, 128, DescOrder::KLeading, SmemSwizzle::None)
                .pack()
                .unwrap();
            let device = ((addr >> 4) & 0x3FFF) | cst;
            assert_eq!(host, device, "address {addr:#x}");
            addr += 16;
        }
        // And the 18-bit mask never fires below 256 KiB, which is why the device fold can skip it.
        const _: () = assert!(HOPPER_SMEM_PER_CTA < (1 << 18));
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
            let align = c.swizzle.required_alignment() as usize;
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
        assert_eq!(mods.len(), 3, "W1 f16, W1 bf16, W3c");
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
        for c in WGMMA_VARIANTS {
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
                    c.consumer_wgs
                ))
                .count(),
                c.stages,
                "{}: the empty barrier expects one arrival per consumer warpgroup",
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
            // two TMA copies per iteration, both against the full barrier
            assert_eq!(
                ptx.matches("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes")
                    .count(),
                2
            );
            assert!(
                !ptx.contains("multicast"),
                "{}: no cluster arm is emitted",
                c.name
            );
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
                "cluster multicast",
                WgmmaCfg {
                    multicast: Multicast::ClusterA,
                    ..WGMMA_W1
                },
                "cluster multicast",
            ),
            (
                "swizzled SMEM",
                WgmmaCfg {
                    swizzle: SmemSwizzle::B128,
                    ..WGMMA_W1
                },
                "swizzle is not implemented",
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

    /// The TMA descriptors the launcher must build, and the transaction count the kernel declares,
    /// are the same fact viewed twice. A mismatch does not fail -- it hangs.
    #[test]
    fn the_declared_transaction_equals_what_the_two_copies_move() {
        for c in WGMMA_VARIANTS {
            let a = c.tensor_map_a(4096, 4096);
            let b = c.tensor_map_b(4096, 4096);
            a.validate().unwrap();
            b.validate().unwrap();
            assert_eq!(
                a.transaction_bytes() + b.transaction_bytes(),
                c.stage_tx_bytes(),
                "{}: expect_tx must equal what the two tensor copies move",
                c.name
            );
            assert_eq!(a.transaction_bytes(), c.tile_a_bytes());
            assert_eq!(b.transaction_bytes(), c.tile_b_bytes());
            assert_eq!(a.box_dim[0] as usize, c.bk);
            assert_eq!(a.box_dim[1] as usize, c.bm);
            assert_eq!(b.box_dim[1] as usize, c.bn);
        }
    }

    /// The unproven-claims list is the honest half of this module and must not quietly empty out or
    /// lose its head item. It is also printed into bring-up logs, so it stays ASCII.
    #[test]
    fn the_device_validation_list_is_intact() {
        assert_eq!(WGMMA_DEVICE_VALIDATION.len(), 8);
        assert!(WGMMA_DEVICE_VALIDATION[0].contains("DescOrder"));
        for item in WGMMA_DEVICE_VALIDATION {
            assert!(item.is_ascii(), "{item}");
            assert!(item.len() > 40);
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
    /// ```text
    /// cargo test -p wukong_codegen_gpu --lib dump_wgmma_ptx -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "inspection helper, not a gate"]
    fn dump_wgmma_ptx() {
        let name = std::env::var("WUKONG_WGMMA_VARIANT").unwrap_or_else(|_| WGMMA_W1.name.into());
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
