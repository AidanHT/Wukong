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

/// `.version 7.8` at the `sm_80` floor — the default header for every Ampere-legal family.
pub const HDR_SM80: &str = ".version 7.8\n.target sm_80\n.address_size 64\n";

/// `.version 8.4` at the `sm_80` floor — for families that keep an 8.4-era `.version` but emit no
/// Ada-only instruction (the int8/int4 families today). When routing a site, keep the file's
/// current `.version`; only the target floor changes.
pub const HDR_SM80_V84: &str = ".version 8.4\n.target sm_80\n.address_size 64\n";

/// `.version 8.4` at the `sm_89` floor — the fp8 families.
pub const HDR_SM89_V84: &str = ".version 8.4\n.target sm_89\n.address_size 64\n";

/// Bare target directives, for containment asserts in tests and for generators that interpolate a
/// header into a larger `format!`. Tests should pin a family's FLOOR via these, never the device.
pub const TARGET_SM80: &str = ".target sm_80";
pub const TARGET_SM89: &str = ".target sm_89";

/// A header from explicit parts, for the odd module whose (version, target) pair is not one of the
/// shipped constants. Prefer the constants — they are the grep point for "what floors exist".
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
        for h in [HDR_SM80, HDR_SM80_V84, HDR_SM89_V84] {
            assert!(h.is_ascii(), "PTX header must be pure ASCII");
            assert!(h.starts_with(".version "));
            assert!(h.ends_with("\n.address_size 64\n"));
        }
        assert!(HDR_SM80.contains(TARGET_SM80));
        assert!(HDR_SM80_V84.contains(TARGET_SM80));
        assert!(HDR_SM89_V84.contains(TARGET_SM89));
        assert_eq!(header("7.8", "sm_80"), HDR_SM80);
        assert_eq!(header("8.4", "sm_80"), HDR_SM80_V84);
        assert_eq!(header("8.4", "sm_89"), HDR_SM89_V84);
    }

    #[test]
    fn device_arch_flags_spell_like_the_tools_expect() {
        assert_eq!(compute_arch(8, 9), "compute_89");
        assert_eq!(compute_arch(9, 0), "compute_90");
        assert_eq!(sm_arch(8, 0), "sm_80");
        assert_eq!(sm_arch(12, 0), "sm_120");
    }
}
