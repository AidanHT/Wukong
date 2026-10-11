//! The instruction-set level native code is generated for (`--target-cpu`).
//!
//! The JIT runs where it compiles, so it has always used every feature the host CPU reports
//! (`cranelift_native::builder`). An AOT artifact does not: an object built that way on an AVX2 laptop
//! faults with `#UD` on any older x86-64 machine, silently and with nothing said at compile time —
//! and on top of the ISA flags, the general vectorizer's 256-bit recipes are installed into the
//! object as raw AVX2+FMA machine code with no runtime dispatch. [`TargetCpu`] makes the level an
//! explicit choice: `native` keeps the historical behaviour, and the x86-64 micro-architecture levels
//! (the same `x86-64`/`-v2`/`-v3`/`-v4` names gcc, clang and rustc accept for `-march`/`target-cpu`)
//! pin an exact, host-independent feature set. A level below `x86-64-v3` also forbids the raw-AVX2
//! recipes ([`TargetCpu::allows_vec_kernels`]); the driver then builds the MIR without them, and the
//! loop stays on the result-identical 128-bit CLIF strips (the form `WUKONG_P4_NO_256=1` forces).
//!
//! Everything else an AOT binary links is already portable: every `wukong_runtime` kernel selects
//! its AVX2/FMA path at run time (`is_x86_feature_detected!`) and keeps a bit-exact scalar twin.

/// See the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TargetCpu {
    /// Every feature the compiling host's CPU reports (the default, and the only JIT behaviour
    /// before this option existed).
    #[default]
    Native,
    /// The x86-64 baseline (SSE2): runs on every x86-64 CPU.
    X86_64,
    /// x86-64-v2: adds SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT and CMPXCHG16B.
    X86_64V2,
    /// x86-64-v3: adds AVX, AVX2, FMA, BMI1, BMI2 and LZCNT (Haswell and later).
    X86_64V3,
    /// x86-64-v4: adds AVX-512 F, DQ and VL.
    X86_64V4,
}

/// The Cranelift x64 settings of each level, cumulative. Spelled out rather than taken from
/// Cranelift's presets: its `x86-64-v3` preset is registered under a misspelled name and omits
/// `has_avx`, without which `has_avx2`/`has_fma` are never used (`use_avx2 = has_avx && has_avx2`).
const V2: &[&str] = &[
    "has_sse3",
    "has_ssse3",
    "has_sse41",
    "has_sse42",
    "has_popcnt",
    "has_cmpxchg16b",
];
const V3: &[&str] = &[
    "has_sse3",
    "has_ssse3",
    "has_sse41",
    "has_sse42",
    "has_popcnt",
    "has_cmpxchg16b",
    "has_avx",
    "has_avx2",
    "has_fma",
    "has_bmi1",
    "has_bmi2",
    "has_lzcnt",
];
const V4: &[&str] = &[
    "has_sse3",
    "has_ssse3",
    "has_sse41",
    "has_sse42",
    "has_popcnt",
    "has_cmpxchg16b",
    "has_avx",
    "has_avx2",
    "has_fma",
    "has_bmi1",
    "has_bmi2",
    "has_lzcnt",
    "has_avx512f",
    "has_avx512dq",
    "has_avx512vl",
];

impl TargetCpu {
    /// Every accepted spelling, in the order `--help` lists them.
    pub const NAMES: &'static [&'static str] =
        &["native", "x86-64", "x86-64-v2", "x86-64-v3", "x86-64-v4"];

    /// Parse a `--target-cpu` value.
    pub fn parse(s: &str) -> Option<TargetCpu> {
        Some(match s {
            "native" => TargetCpu::Native,
            "x86-64" => TargetCpu::X86_64,
            "x86-64-v2" => TargetCpu::X86_64V2,
            "x86-64-v3" => TargetCpu::X86_64V3,
            "x86-64-v4" => TargetCpu::X86_64V4,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            TargetCpu::Native => "native",
            TargetCpu::X86_64 => "x86-64",
            TargetCpu::X86_64V2 => "x86-64-v2",
            TargetCpu::X86_64V3 => "x86-64-v3",
            TargetCpu::X86_64V4 => "x86-64-v4",
        }
    }

    /// The Cranelift x64 settings this level enables, or `None` for `native` (host detection).
    pub(crate) fn x86_flags(self) -> Option<&'static [&'static str]> {
        match self {
            TargetCpu::Native => None,
            TargetCpu::X86_64 => Some(&[]),
            TargetCpu::X86_64V2 => Some(V2),
            TargetCpu::X86_64V3 => Some(V3),
            TargetCpu::X86_64V4 => Some(V4),
        }
    }

    /// Whether code for this target may contain the general vectorizer's raw AVX2+FMA kernels.
    /// `native` defers to the host gate the vectorizer already applies
    /// (`wukong_mir::host_supports_vec_kernels`).
    pub fn allows_vec_kernels(self) -> bool {
        matches!(
            self,
            TargetCpu::Native | TargetCpu::X86_64V3 | TargetCpu::X86_64V4
        )
    }
}

/// The x64 settings [`TargetCpu::Native`] generates code with under a `WUKONG_MAX_ISA` cap
/// (`wukong_runtime::isa::MAX_ISA_ENV`), given the settings the host reports, or `None` for no cap.
///
/// `None` means host detection exactly as before the cap existed (`cranelift_native::builder`), so
/// an uncapped build emits byte-identical code. A cap keeps only the host flags its level allows:
/// `scalar` the x86-64 baseline (no optional flag), `avx2` (and `avx2_vnni`, which adds nothing
/// Cranelift uses) x86-64-v3, `avx512` x86-64-v4. Like the runtime's probes, a cap can only take an
/// extension away — it never enables one the host lacks, which would be a `#UD`. This is the third
/// reader of the cap, beside the runtime's dispatchers and the raw-AVX2 loop kernels
/// (`wukong_mir::host_supports_vec_kernels`): without it a capped run still JIT-compiled every CLIF
/// loop with VEX/AVX2/FMA encodings, so the "scalar lane" executed AVX instructions.
pub(crate) fn capped_native_flags(
    host: &[&'static str],
    cap: wukong_runtime::isa::IsaLevel,
) -> Option<Vec<&'static str>> {
    use wukong_runtime::isa::IsaLevel;
    let allowed: &[&str] = match cap {
        IsaLevel::Native => return None,
        IsaLevel::Scalar => &[],
        IsaLevel::Avx2 | IsaLevel::Avx2Vnni => V3,
        IsaLevel::Avx512 => V4,
    };
    Some(
        host.iter()
            .copied()
            .filter(|flag| allowed.contains(flag))
            .collect(),
    )
}

/// The capped level flags this host reports (x86-64-v4 is the superset of every level's flags, so
/// this is everything a cap can keep).
fn host_level_flags() -> Vec<&'static str> {
    V4.iter().copied().filter(|flag| host_has(flag)).collect()
}

/// The ISA builder for `target`: host detection for `native` (narrowed by `cap`, see
/// [`capped_native_flags`]), otherwise exactly the level's flags. An explicit level is a request
/// for an exact, host-independent feature set (an AOT target), so the cap does not touch it.
pub(crate) fn isa_builder(
    target: TargetCpu,
    cap: wukong_runtime::isa::IsaLevel,
) -> Result<cranelift_codegen::isa::Builder, String> {
    use cranelift_codegen::settings::Configurable;

    let capped;
    let flags: &[&str] = match target.x86_flags() {
        Some(flags) => flags,
        None => match capped_native_flags(&host_level_flags(), cap) {
            None => return cranelift_native::builder().map_err(|e| e.to_string()),
            // Off x86-64 the cap has nothing to narrow: host detection stands.
            Some(_) if !cfg!(target_arch = "x86_64") => {
                return cranelift_native::builder().map_err(|e| e.to_string());
            }
            Some(flags) => {
                capped = flags;
                &capped
            }
        },
    };
    if !cfg!(target_arch = "x86_64") {
        return Err(format!(
            "unsupported target CPU `{}`: the x86-64 levels require an x86-64 host \
             (cross-compilation is not supported)",
            target.name()
        ));
    }
    let mut builder = cranelift_native::builder_with_options(false).map_err(|e| e.to_string())?;
    for flag in flags {
        builder
            .enable(flag)
            .map_err(|e| format!("unsupported target CPU `{}`: {e}", target.name()))?;
    }
    Ok(builder)
}

/// The features of `target` this host's CPU lacks — the JIT must refuse to run code it would fault
/// on (an AOT object may target any level).
pub(crate) fn missing_on_host(target: TargetCpu) -> Vec<&'static str> {
    target
        .x86_flags()
        .unwrap_or(&[])
        .iter()
        .copied()
        .filter(|flag| !host_has(flag))
        .collect()
}

#[cfg(target_arch = "x86_64")]
fn host_has(flag: &str) -> bool {
    match flag {
        "has_sse3" => std::is_x86_feature_detected!("sse3"),
        "has_ssse3" => std::is_x86_feature_detected!("ssse3"),
        "has_sse41" => std::is_x86_feature_detected!("sse4.1"),
        "has_sse42" => std::is_x86_feature_detected!("sse4.2"),
        "has_popcnt" => std::is_x86_feature_detected!("popcnt"),
        "has_cmpxchg16b" => std::is_x86_feature_detected!("cmpxchg16b"),
        "has_avx" => std::is_x86_feature_detected!("avx"),
        "has_avx2" => std::is_x86_feature_detected!("avx2"),
        "has_fma" => std::is_x86_feature_detected!("fma"),
        "has_bmi1" => std::is_x86_feature_detected!("bmi1"),
        "has_bmi2" => std::is_x86_feature_detected!("bmi2"),
        "has_lzcnt" => std::is_x86_feature_detected!("lzcnt"),
        "has_avx512f" => std::is_x86_feature_detected!("avx512f"),
        "has_avx512dq" => std::is_x86_feature_detected!("avx512dq"),
        "has_avx512vl" => std::is_x86_feature_detected!("avx512vl"),
        _ => false,
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn host_has(_flag: &str) -> bool {
    false
}

/// Refuse a program whose MIR carries raw AVX2+FMA vector kernels when `target` may not run them.
/// The driver builds the MIR without them for such a target, so reaching this is a caller that
/// skipped that step — an AOT object must never embed AVX2 bytes for a baseline target.
pub(crate) fn check_vec_kernels(
    program: &wukong_mir::Program,
    interner: &wukong_span::Interner,
    target: TargetCpu,
) -> Result<(), String> {
    if target.allows_vec_kernels() {
        return Ok(());
    }
    match program.funcs.iter().find(|f| !f.vec_kernels.is_empty()) {
        Some(f) => Err(format!(
            "unsupported target CPU `{}`: function `{}` carries raw AVX2+FMA vector kernels, which \
             need x86-64-v3; build the MIR without them (`LowerOptions::vec_kernels = false`, which \
             `wukongc --target-cpu` does) or target x86-64-v3",
            target.name(),
            interner.resolve(f.name)
        )),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    //! N46: `WUKONG_MAX_ISA` narrows the `native` target's Cranelift flags. The mapping is a pure
    //! function of (host flags, cap), so it is tested with explicit caps and synthetic hosts; the
    //! ISA built from a cap is then read back to show the flags reach Cranelift.
    use super::*;
    use cranelift_codegen::settings::{self, Configurable};
    use wukong_runtime::isa::IsaLevel;

    /// The value of the boolean x64 setting `name` in a finished ISA.
    fn isa_flag(isa: &dyn cranelift_codegen::isa::TargetIsa, name: &str) -> bool {
        isa.isa_flags()
            .iter()
            .find(|v| v.name == name)
            .and_then(|v| v.as_bool())
            .unwrap_or_else(|| panic!("no boolean x64 setting `{name}`"))
    }

    fn finish(
        builder: cranelift_codegen::isa::Builder,
    ) -> std::sync::Arc<dyn cranelift_codegen::isa::TargetIsa> {
        let mut flags = settings::builder();
        flags.set("opt_level", "speed").unwrap();
        builder.finish(settings::Flags::new(flags)).unwrap()
    }

    #[test]
    fn target_cap_scalar_is_the_baseline_on_every_host() {
        // A host that reports every level flag still gets none of them under `scalar`.
        assert_eq!(capped_native_flags(V4, IsaLevel::Scalar), Some(vec![]));
        if !cfg!(target_arch = "x86_64") {
            return;
        }
        let isa = finish(isa_builder(TargetCpu::Native, IsaLevel::Scalar).unwrap());
        for flag in ["has_avx", "has_avx2", "has_fma", "has_sse41", "has_avx512f"] {
            assert!(!isa_flag(&*isa, flag), "scalar cap left `{flag}` on");
        }
    }

    #[test]
    fn target_cap_avx2_never_enables_avx512() {
        let capped = capped_native_flags(V4, IsaLevel::Avx2).unwrap();
        assert_eq!(
            capped,
            V3.to_vec(),
            "avx2 on an AVX-512 host is exactly x86-64-v3"
        );
        assert_eq!(capped_native_flags(V4, IsaLevel::Avx2Vnni).unwrap(), V3);
        assert!(capped.iter().all(|f| !f.starts_with("has_avx512")));
        assert_eq!(capped_native_flags(V4, IsaLevel::Avx512).unwrap(), V4);
        if !cfg!(target_arch = "x86_64") {
            return;
        }
        let isa = finish(isa_builder(TargetCpu::Native, IsaLevel::Avx2).unwrap());
        for flag in ["has_avx512f", "has_avx512dq", "has_avx512vl"] {
            assert!(!isa_flag(&*isa, flag), "avx2 cap left `{flag}` on");
        }
        // Below the cap, the host's own answer stands.
        for flag in ["has_avx", "has_avx2", "has_fma"] {
            assert_eq!(isa_flag(&*isa, flag), host_has(flag), "{flag}");
        }
    }

    /// A cap only ever removes: it never claims an extension the host does not report.
    #[test]
    fn target_cap_never_exceeds_the_host() {
        let host = ["has_sse3", "has_ssse3", "has_popcnt"];
        for cap in [IsaLevel::Avx2, IsaLevel::Avx2Vnni, IsaLevel::Avx512] {
            assert_eq!(capped_native_flags(&host, cap).unwrap(), host, "{cap:?}");
        }
        assert_eq!(capped_native_flags(&[], IsaLevel::Avx512), Some(vec![]));
    }

    /// No cap is host detection itself, untouched — the guarantee that an uncapped build's code
    /// bytes did not change (`determinism.rs` pins the bytes across processes).
    #[test]
    fn target_without_a_cap_is_todays_host_detection() {
        assert_eq!(capped_native_flags(V4, IsaLevel::Native), None);
        let uncapped = finish(isa_builder(TargetCpu::Native, IsaLevel::Native).unwrap());
        let host = finish(cranelift_native::builder().unwrap());
        let render = |isa: &dyn cranelift_codegen::isa::TargetIsa| {
            isa.isa_flags()
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(render(&*uncapped), render(&*host));
        // An explicit level ignores the cap: it is an exact AOT feature set.
        if cfg!(target_arch = "x86_64") {
            let v3 = finish(isa_builder(TargetCpu::X86_64V3, IsaLevel::Scalar).unwrap());
            assert!(isa_flag(&*v3, "has_avx2") && isa_flag(&*v3, "has_fma"));
        }
    }
}
