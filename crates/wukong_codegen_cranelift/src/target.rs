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

/// The ISA builder for `target`: host detection for `native`, otherwise exactly the level's flags.
pub(crate) fn isa_builder(target: TargetCpu) -> Result<cranelift_codegen::isa::Builder, String> {
    use cranelift_codegen::settings::Configurable;

    let Some(flags) = target.x86_flags() else {
        return cranelift_native::builder().map_err(|e| e.to_string());
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
