//! `ptx_target` — the single source of the `.version`/`.target`/`.address_size` module header
//! every PTX generator opens with, and of the device-arch flag spellings the peer and cubin seams
//! need. Before this module existed, 67 sites across 18 files hardcoded `.target sm_89`.
//!
//! **The emission rule** (GPU_RETARGET_PLAN.md §5, Phase 2 step 2): a PTX module is tagged with the
//! LOWEST target its instruction mix is legal on — `sm_80` for the Ampere-legal majority
//! (`mma.sync m16n8k16/k32`, `ldmatrix`, `cp.async`, `lop3`), `sm_89` for the fp8 families
//! (`e4m3`/`e5m2` mma and the packed `cvt.rn.satfinite.*x2` converters exist nowhere below Ada) —
//! never with the device's own architecture. PTX is forward-compatible only: a module tagged
//! `sm_80` driver-JITs on every later part, while one tagged `sm_89` loads on ZERO A100s. The
//! device's real capability (`Gpu::target()`) matters at three OTHER seams instead: capability
//! gating before dispatch (fp8 requires cc >= 8.9), the NVRTC/ptxas peer arch flags
//! (`compute_arch`/`sm_arch` below), and the cubin-cache key.
//!
//! The 48 KiB static `.shared` ceiling is a PTX ISA rule on every non-`a` target (PTX §5.1.7),
//! NOT a device fact — retagging alone lifts nothing; only the dynamic-SMEM extern window does.
//!
//! Deliberately **un-gated** (pure strings, no `cudarc`): the un-gated `paged_attention`
//! generators route through it, and these gates run in a plain, toolchain-free `cargo test`.

/// `.version 7.8` at the `sm_80` floor — the header for every Ampere-legal family, which since the
/// int8 `.version` float is **every** family but fp8. There is deliberately no `8.4`-at-`sm_80`
/// constant: an ISA-7.0 instruction mix has no business demanding an r550+ driver, and a documented
/// constant for that pair is an invitation to declare it again.
pub const HDR_SM80: &str = ".version 7.8\n.target sm_80\n.address_size 64\n";

/// `.version 8.4` at the `sm_89` floor — the fp8 families, and the only place either 8.4 or sm_89 is
/// earned: the `e4m3`/`e5m2` `mma` and the packed `cvt.rn.satfinite.*x2` converters exist nowhere
/// below Ada, and their PTX-ISA introduction is past 7.8.
pub const HDR_SM89_V84: &str = ".version 8.4\n.target sm_89\n.address_size 64\n";

/// `.version 8.0` at the **`sm_90a`** floor — the Hopper warpgroup family (`wgmma` + TMA +
/// `setmaxnreg` + `mbarrier` transaction barriers). The third and last shipped floor, and the only
/// one that is **architecture-LOCKED** rather than merely arch-floored.
///
/// **Why `sm_90a` and not `sm_90`.** `wgmma.mma_async` is documented "Requires `sm_90a`" (PTX ISA
/// §9.7.16, D1 §1.4). The trailing `a` marks an *architecture-specific* target: unlike `sm_80` or
/// `sm_89`, which are floors that every later part JITs from, an `sm_90a` module is legal on Hopper
/// **and nowhere else**. It will not load on `sm_100`/`sm_120` and it cannot be relaxed to plain
/// `sm_90` — `wgmma` is simply not in that target's instruction set. So this constant forfeits the
/// forward compatibility the emission rule above buys everywhere else, deliberately, because there
/// is no alternative spelling that keeps it.
///
/// The consequence for callers is a **third category** in the emission rule (D1 §6 finding 3):
/// `sm_80`/`sm_89` are floors, `sm_90a` is a lock. A module tagged with it MUST be reached only
/// through a capability gate that has established a cc of 9.x on the *probed* device
/// (`ptx_wgmma::require_sm90a`) — never emitted speculatively, never for a device below cc 9.0, and
/// never assumed to survive onto a later architecture.
///
/// **Why `.version 8.0` and not higher.** The newest instructions the family emits — `wgmma.*`,
/// `cp.async.bulk.tensor.*` and `setmaxnreg` — are "Introduced in PTX ISA version 8.0"; the rest
/// (`mbarrier.arrive.expect_tx`, `mbarrier.try_wait.parity`, `cvta.param`) are older still. Nothing
/// in the mix postdates 8.0, so by the same rule that keeps the Ampere families at 7.8 this declares
/// 8.0 and no more: `.version 8.0` asks for driver r525+, while `.version 8.4` would ask for r550+
/// and buy nothing.
pub const HDR_SM90A_V80: &str = ".version 8.0\n.target sm_90a\n.address_size 64\n";

/// Bare target directives, for containment asserts in tests and for generators that interpolate a
/// header into a larger `format!`. Tests should pin a family's FLOOR via these, never the device.
pub const TARGET_SM80: &str = ".target sm_80";
pub const TARGET_SM89: &str = ".target sm_89";
/// The architecture-LOCKED Hopper target — see [`HDR_SM90A_V80`]. Note it is **not** a prefix-free
/// relative of `.target sm_90`: a containment test for one must not be written as a `contains` of
/// the other (`".target sm_90"` is a substring of `".target sm_90a"`).
pub const TARGET_SM90A: &str = ".target sm_90a";

/// A header from explicit parts, for the odd module whose (version, target) pair is not one of the
/// shipped constants. Prefer the constants — they are the grep point for "what floors exist".
///
/// **Justify the `version` you pass, not just the `target`.** `.version` is a *driver* floor exactly
/// as `.target` is a *device* floor, and `cuModuleLoadData` enforces it: `.version 7.8` needs r520+,
/// `.version 8.4` needs **r550+** — so an 8.4 tag refuses to load on the r535/r545 fleets this
/// retarget targets, for nothing, unless the module actually emits an instruction introduced above
/// ISA 7.8. Pass the LOWEST `.version` its instruction mix is legal on, the same rule `.target` obeys.
pub fn header(version: &str, target: &str) -> String {
    format!(".version {version}\n.target {target}\n.address_size 64\n")
}

/// `compute_XX` — the NVRTC `--gpu-architecture` spelling for a DEVICE. Peers compile for the
/// device the round runs on, not for a family floor; feed this from `Gpu::target()`.
pub fn compute_arch(cc_major: i32, cc_minor: i32) -> String {
    format!("compute_{cc_major}{cc_minor}")
}

/// `sm_XX` — the ptxas `-arch` / cubin-cache-key spelling for a DEVICE.
pub fn sm_arch(cc_major: i32, cc_minor: i32) -> String {
    format!("sm_{cc_major}{cc_minor}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_are_ascii_well_formed_and_agree_with_the_builder() {
        for h in [HDR_SM80, HDR_SM89_V84, HDR_SM90A_V80] {
            assert!(h.is_ascii(), "PTX header must be pure ASCII");
            assert!(h.starts_with(".version "));
            assert!(h.ends_with("\n.address_size 64\n"));
        }
        assert!(HDR_SM80.contains(TARGET_SM80));
        assert!(HDR_SM89_V84.contains(TARGET_SM89));
        assert!(HDR_SM90A_V80.contains(TARGET_SM90A));
        assert_eq!(header("7.8", "sm_80"), HDR_SM80);
        assert_eq!(header("8.4", "sm_89"), HDR_SM89_V84);
        assert_eq!(header("8.0", "sm_90a"), HDR_SM90A_V80);
    }

    /// **The `a` in `sm_90a` is load-bearing and easy to lose.** `.target sm_90` and `.target sm_90a`
    /// differ by one byte, `sm_90` is a *prefix* of `sm_90a`, and the wrong one of the two is not a
    /// compile error anywhere in this repo — it is a `cuModuleLoadData` failure on an H100 six months
    /// from now, naming `wgmma` rather than the target. So the pairing is pinned in both directions:
    /// the Hopper header carries the `a`, and no *other* shipped header may carry a `sm_90*` target
    /// (a family that is merely Hopper-legal belongs at the `sm_80` floor, which JITs there anyway).
    #[test]
    fn the_hopper_floor_is_architecture_locked_and_alone() {
        assert!(
            HDR_SM90A_V80.contains(".target sm_90a\n"),
            "the Hopper floor must be the architecture-specific `sm_90a`, not plain `sm_90`"
        );
        for h in [HDR_SM80, HDR_SM89_V84] {
            assert!(
                !h.contains(".target sm_90"),
                "only the wgmma family may name a Hopper target: {h:?}"
            );
        }
        // `sm_90` is a prefix of `sm_90a`: a `contains(".target sm_90")` test would pass on both, so
        // anything pinning the plain target must pin the newline too. Proven here so the trap is a
        // recorded fact rather than a comment.
        assert!(HDR_SM90A_V80.contains(".target sm_90"));
        assert!(!HDR_SM90A_V80.contains(".target sm_90\n"));
    }

    /// The Hopper family declares the LOWEST `.version` its instruction mix needs, exactly as the
    /// Ampere and Ada floors do. `wgmma`, `cp.async.bulk.tensor`, `mbarrier.arrive.expect_tx`,
    /// `mbarrier.try_wait.parity` and `setmaxnreg` are all "Introduced in PTX ISA version 8.0", so
    /// 8.0 is earned and 8.4 (driver r550+) would be an unearned load failure on an r53x fleet.
    #[test]
    fn the_hopper_floor_asks_for_the_r525_driver_and_no_more() {
        assert!(HDR_SM90A_V80.starts_with(".version 8.0"));
        assert!(!HDR_SM90A_V80.contains(".version 8.4"));
    }

    /// **`.version 8.4` is spelled in exactly one shipped constant, and it is the `sm_89` one.** The
    /// removed `HDR_SM80_V84` was an ISA-7.0 instruction mix (int8/int4: `mma.sync.m16n8k32.u8.s8`,
    /// `ldmatrix`, `cp.async`) tagged with a header that makes `cuModuleLoadData` demand driver r550+ —
    /// a pure load failure on the r535/r545 fleets, bought nothing. It is gone; only fp8, whose
    /// `e4m3`/`e5m2` `mma` genuinely postdates 7.8, keeps an 8.4. A future `8.4`-at-an-Ampere-floor
    /// constant would reintroduce the hole silently, so the pairing is pinned here.
    #[test]
    fn only_the_ada_floor_declares_the_r550_driver_version() {
        for h in [HDR_SM80, HDR_SM89_V84, HDR_SM90A_V80] {
            if h.contains(".version 8.4") {
                assert!(
                    h.contains(TARGET_SM89),
                    "`.version 8.4` demands driver r550+; only the fp8/Ada floor earns it: {h:?}"
                );
            }
        }
        assert!(
            HDR_SM80.starts_with(".version 7.8"),
            "the Ampere floor stays at the r520+ driver"
        );
    }

    #[test]
    fn device_arch_flags_spell_like_the_tools_expect() {
        assert_eq!(compute_arch(8, 9), "compute_89");
        assert_eq!(compute_arch(9, 0), "compute_90");
        assert_eq!(sm_arch(8, 0), "sm_80");
        assert_eq!(sm_arch(12, 0), "sm_120");
    }
}
