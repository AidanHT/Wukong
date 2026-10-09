//! The one place the runtime asks which instruction-set extensions it may use.
//!
//! Every kernel picks its path at run time: an AVX2/FMA (or F16C, AVX-VNNI, AVX-512) body when the
//! CPU has it, and a bit-exact scalar twin otherwise. The decision used to be ~250 scattered
//! `is_x86_feature_detected!` calls with no override, so on every gate host (all AVX2) the scalar
//! arm of every dispatcher, which is exactly what an ARM or pre-Haswell user runs, was never
//! reached from a compiled program; only the unit tests that call a twin directly touched it.
//!
//! [`MAX_ISA_ENV`] (`WUKONG_MAX_ISA`) caps what the dispatchers see, in the manner of oneDNN's
//! `ONEDNN_MAX_CPU_ISA`: `scalar`, `avx2`, `avx2_vnni` or `avx512` (case-insensitive). Each probe
//! here answers "the CPU has it AND the cap allows it", so a cap can only take a fast path away,
//! never claim an extension the CPU lacks (which would be a `#UD`). Unset, or an unrecognized
//! value, means no cap. The cap is read once per process.
//!
//! `wukong_mir::host_supports_vec_kernels` (the compiler's raw-AVX2 loop kernels) reads the same
//! variable with the same spellings; keep the two parsers in step.

use std::sync::OnceLock;

/// The environment variable that caps the instruction-set level the runtime dispatches to.
pub const MAX_ISA_ENV: &str = "WUKONG_MAX_ISA";

/// An instruction-set tier, ordered: a cap at one level allows every level below it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum IsaLevel {
    /// Baseline only: every kernel takes its scalar twin.
    Scalar,
    /// AVX, AVX2, FMA and F16C (x86-64-v3, Haswell and later).
    Avx2,
    /// The AVX2 tier plus AVX-VNNI (the VEX-encoded int8 dot product, Alder Lake and later).
    Avx2Vnni,
    /// The AVX2-VNNI tier plus AVX-512F.
    Avx512,
    /// No cap: whatever the CPU reports.
    Native,
}

impl IsaLevel {
    /// The level a [`MAX_ISA_ENV`] value names, or `None` for an unrecognized spelling.
    pub fn parse(s: &str) -> Option<IsaLevel> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "scalar" | "none" | "baseline" => IsaLevel::Scalar,
            "avx2" => IsaLevel::Avx2,
            "avx2_vnni" | "avx2-vnni" | "avxvnni" => IsaLevel::Avx2Vnni,
            "avx512" | "avx512f" => IsaLevel::Avx512,
            "native" | "all" => IsaLevel::Native,
            _ => return None,
        })
    }
}

/// The process's cap: [`MAX_ISA_ENV`] parsed once, [`IsaLevel::Native`] when unset or unrecognized.
pub fn cap() -> IsaLevel {
    static CAP: OnceLock<IsaLevel> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var(MAX_ISA_ENV)
            .ok()
            .and_then(|v| IsaLevel::parse(&v))
            .unwrap_or(IsaLevel::Native)
    })
}

/// Does the cap allow code of `level`? (Whether the CPU has it is the probes' other half.)
#[inline]
pub fn allows(level: IsaLevel) -> bool {
    cap() >= level
}

/// One probe per extension the runtime dispatches on: the CPU reports it and the cap allows its
/// tier. Off x86 every probe is `false`, so the scalar twins run.
macro_rules! probe {
    ($(#[$doc:meta])* $name:ident, $feature:tt, $level:expr) => {
        $(#[$doc])*
        #[inline]
        pub fn $name() -> bool {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                allows($level) && std::is_x86_feature_detected!($feature)
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                false
            }
        }
    };
}

probe!(
    /// AVX (256-bit float).
    avx, "avx", IsaLevel::Avx2
);
probe!(
    /// AVX2 (256-bit integer and gathers).
    avx2, "avx2", IsaLevel::Avx2
);
probe!(
    /// FMA3 (fused multiply-add).
    fma, "fma", IsaLevel::Avx2
);
probe!(
    /// F16C (`vcvtph2ps` / `vcvtps2ph`).
    f16c, "f16c", IsaLevel::Avx2
);
probe!(
    /// AVX-VNNI (`vpdpbusd` on 256-bit registers).
    avxvnni, "avxvnni", IsaLevel::Avx2Vnni
);
probe!(
    /// AVX-512 Foundation.
    avx512f, "avx512f", IsaLevel::Avx512
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_parse_case_insensitively_and_order() {
        assert_eq!(IsaLevel::parse("scalar"), Some(IsaLevel::Scalar));
        assert_eq!(IsaLevel::parse(" AVX2 "), Some(IsaLevel::Avx2));
        assert_eq!(IsaLevel::parse("avx2_vnni"), Some(IsaLevel::Avx2Vnni));
        assert_eq!(IsaLevel::parse("AVX512"), Some(IsaLevel::Avx512));
        assert_eq!(IsaLevel::parse("native"), Some(IsaLevel::Native));
        assert_eq!(IsaLevel::parse("sse9"), None);
        assert!(IsaLevel::Scalar < IsaLevel::Avx2);
        assert!(IsaLevel::Avx2 < IsaLevel::Avx2Vnni);
        assert!(IsaLevel::Avx2Vnni < IsaLevel::Avx512);
        assert!(IsaLevel::Avx512 < IsaLevel::Native);
    }

    /// A probe never claims more than the CPU has: under any cap, each one implies the raw
    /// detection, and under the `scalar` cap (the CI lane) every probe is off.
    #[test]
    fn probes_never_exceed_the_cpu_and_honor_the_cap() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            assert!(!avx2() || std::is_x86_feature_detected!("avx2"));
            assert!(!fma() || std::is_x86_feature_detected!("fma"));
            assert!(!f16c() || std::is_x86_feature_detected!("f16c"));
            assert!(!avxvnni() || std::is_x86_feature_detected!("avxvnni"));
            assert!(!avx512f() || std::is_x86_feature_detected!("avx512f"));
        }
        if cap() == IsaLevel::Scalar {
            assert!(!avx() && !avx2() && !fma() && !f16c());
            assert!(!avxvnni() && !avx512f());
        }
        if cap() < IsaLevel::Avx512 {
            assert!(!avx512f());
        }
    }
}
