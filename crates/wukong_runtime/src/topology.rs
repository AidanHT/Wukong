//! Host CPU topology, probed once per process: the cores this process may actually run on and the
//! data-cache geometry, so scheduling thresholds follow the machine instead of one laptop's
//! hardcoded constants.
//!
//! What is probed, and from where:
//!
//! * **Usable logical CPUs** — [`std::thread::available_parallelism`], which on Linux honors the
//!   process affinity mask (`taskset`, `sched_setaffinity`) *and* a cgroup CPU quota (containers,
//!   CI runners). This is the same count rayon's default pool width derives from.
//! * **Usable physical cores** — on Linux, exactly: the CPUs in `/proc/self/status`'s
//!   `Cpus_allowed_list` are mapped to their `(physical_package_id, core_id)` pairs through sysfs and
//!   the distinct pairs counted, then capped by the quota above. Elsewhere (and if any sysfs read
//!   fails) it is estimated from the system-wide counts: the machine's physical cores when the whole
//!   machine is usable, otherwise the usable logical CPUs scaled by the machine's
//!   physical-to-logical ratio. `num_cpus::get_physical()` alone — what the GEMM pool used to read —
//!   counts every core in `/proc/cpuinfo` and ignores both the mask and the quota.
//! * **Cache sizes** (L1D, L2, L3) — on x86-64, CPUID's deterministic cache-parameter leaf (leaf 4
//!   on Intel, extended leaf `0x8000_001D` on AMD); on any other Linux target, sysfs
//!   `/sys/devices/system/cpu/cpu0/cache/index*/`. The L3 figure is the size of one cache instance
//!   (the whole shared L3 on a monolithic die, one CCX's slice on a chiplet part).
//!
//! Every probed value has an environment override, read once at the first probe, so a porter can
//! sweep a threshold without a rebuild and a measurement can pin the machine model it ran under:
//! `WUKONG_CPU_PHYSICAL` (usable physical cores), `WUKONG_CACHE_L1D_KB`, `WUKONG_CACHE_L2_KB` and
//! `WUKONG_CACHE_L3_KB` (KiB). Thread counts are requested with `WUKONG_NUM_THREADS` (or rayon's
//! own `RAYON_NUM_THREADS`), see [`requested_threads`].
//!
//! **The numerical contract.** Nothing probed here may change a computed bit. The values feed only
//! *scheduling* decisions whose alternatives are bit-identical by the crate's standing laws: which
//! pool runs a kernel (every routed kernel chunks independently of the worker count), whether a
//! `_parallel` entry forks or tail-calls its serial sibling (serial == parallel bit-for-bit), and
//! whether a streaming kernel uses `vmovntps` or an ordinary store (identical bits written). The
//! constants that DO fix a result — the GEMM K-block (`select_kc`), the reduction chunk
//! (`reduce.rs`'s `RCHUNK`) — deliberately do not read this module.

use std::sync::OnceLock;

/// The process's view of the host, probed once (see the module docs for sources and overrides).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostTopology {
    /// Logical CPUs this process may use (affinity- and quota-aware). Always `>= 1`.
    pub logical_available: usize,
    /// Physical cores among the usable logical CPUs, capped by `logical_available`. Always in
    /// `1..=logical_available`.
    pub physical_available: usize,
    /// Physical cores in the whole machine, as the OS reports them (`num_cpus::get_physical`).
    pub physical_total: usize,
    /// Per-core L1 data cache, bytes.
    pub l1d_bytes: Option<usize>,
    /// L2 cache visible to one core, bytes.
    pub l2_bytes: Option<usize>,
    /// One last-level (L3) cache instance, bytes.
    pub l3_bytes: Option<usize>,
    /// Where the cache sizes came from: `"cpuid"`, `"sysfs"`, or `"none"` (an override replaces a
    /// field's value without changing this tag).
    pub cache_source: &'static str,
}

impl HostTopology {
    /// Do the usable cores run more than one hardware thread each (SMT / HyperThreading)?
    pub fn smt(&self) -> bool {
        self.physical_available < self.logical_available
    }
}

/// The probed host topology (cached after the first call).
pub fn host_topology() -> &'static HostTopology {
    static HOST: OnceLock<HostTopology> = OnceLock::new();
    HOST.get_or_init(probe)
}

/// The fallback non-temporal-store threshold when no last-level cache size is known: the
/// historical 10 MiB.
pub(crate) const NT_FALLBACK_BYTES: usize = 10 * 1024 * 1024;

/// The non-temporal-store threshold derived from the last-level cache: three quarters of one L3
/// instance (glibc's `__x86_shared_non_temporal_threshold` rule for the same decision in `memcpy`
/// — streaming stores pay off once the written data would evict most of the LLC anyway), clamped
/// to `[4 MiB, 64 MiB]`. Falls back to [`NT_FALLBACK_BYTES`] when the probe found no L3.
/// Throughput-only: an NT store and an ordinary store write identical bits.
pub(crate) fn nt_threshold_from_l3(l3_bytes: Option<usize>) -> usize {
    match l3_bytes {
        Some(b) if b > 0 => (b / 4).saturating_mul(3).clamp(4 << 20, 64 << 20),
        _ => NT_FALLBACK_BYTES,
    }
}

/// Width of the private physical-core kernel pool for a process whose ordinary pool has
/// `pool_width` workers: `Some(physical)` when the usable cores carry SMT siblings to shed
/// (`physical < pool_width`), `None` when the ordinary pool is already one worker per core — a
/// non-SMT host, an affinity mask or quota that leaves one thread per core, or an explicit
/// thread request at or below the core count. In the `None` case the GEMM and every routed kernel
/// share the ordinary (global) pool, so the forward pass is still on ONE pool; nothing is lost by
/// not building a second one of the same width.
pub(crate) fn private_pool_width(physical_available: usize, pool_width: usize) -> Option<usize> {
    if physical_available == 0 || physical_available >= pool_width {
        None
    } else {
        Some(physical_available)
    }
}

/// The thread count the user requested for the runtime's pools: `WUKONG_NUM_THREADS` if set to a
/// positive integer, else `RAYON_NUM_THREADS` likewise, else `None` (the pool sizes itself to the
/// usable logical CPUs). Read once. `wukongc --threads=<n>` sets `WUKONG_NUM_THREADS` for the run.
pub(crate) fn requested_threads() -> Option<usize> {
    static V: OnceLock<Option<usize>> = OnceLock::new();
    *V.get_or_init(|| {
        parse_threads(std::env::var("WUKONG_NUM_THREADS").ok().as_deref())
            .or_else(|| parse_threads(std::env::var("RAYON_NUM_THREADS").ok().as_deref()))
    })
}

/// A thread-count knob: a positive ASCII decimal no larger than 4096; anything else is unset.
fn parse_threads(raw: Option<&str>) -> Option<usize> {
    let s = raw?.trim();
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<usize>().ok().filter(|&t| (1..=4096).contains(&t))
}

/// A positive ASCII-decimal override (`WUKONG_CPU_PHYSICAL`), or `None`.
fn parse_positive(raw: Option<&str>) -> Option<usize> {
    let s = raw?.trim();
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<usize>().ok().filter(|&v| v > 0)
}

/// A positive KiB override (`WUKONG_CACHE_*_KB`) as bytes, or `None` (saturating, never wrapping).
fn parse_kib(raw: Option<&str>) -> Option<usize> {
    parse_positive(raw).map(|kib| kib.saturating_mul(1024))
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn probe() -> HostTopology {
    let logical_available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1);
    let physical_total = num_cpus::get_physical().max(1);
    let logical_total = num_cpus::get().max(logical_available);
    let physical = parse_positive(env("WUKONG_CPU_PHYSICAL").as_deref())
        .or_else(linux_physical_in_affinity)
        .unwrap_or_else(|| estimate_physical(logical_available, logical_total, physical_total));
    let physical_available = physical.clamp(1, logical_available);

    let (mut caches, cache_source) = match probe_caches_cpuid() {
        Some(c) => (c, "cpuid"),
        None => match probe_caches_sysfs() {
            Some(c) => (c, "sysfs"),
            None => (CacheSizes::default(), "none"),
        },
    };
    if let Some(v) = parse_kib(env("WUKONG_CACHE_L1D_KB").as_deref()) {
        caches.l1d = Some(v);
    }
    if let Some(v) = parse_kib(env("WUKONG_CACHE_L2_KB").as_deref()) {
        caches.l2 = Some(v);
    }
    if let Some(v) = parse_kib(env("WUKONG_CACHE_L3_KB").as_deref()) {
        caches.l3 = Some(v);
    }
    HostTopology {
        logical_available,
        physical_available,
        physical_total,
        l1d_bytes: caches.l1d,
        l2_bytes: caches.l2,
        l3_bytes: caches.l3,
        cache_source,
    }
}

/// Usable physical cores when no exact mapping is available: the whole machine's physical count
/// when the whole machine is usable, else the usable logical CPUs scaled by the machine-wide
/// physical-to-logical ratio (rounded up), clamped to `1..=logical_available`.
fn estimate_physical(
    logical_available: usize,
    logical_total: usize,
    physical_total: usize,
) -> usize {
    let logical_total = logical_total.max(logical_available).max(1);
    let est = if logical_available >= logical_total {
        physical_total
    } else {
        logical_available
            .saturating_mul(physical_total)
            .div_ceil(logical_total)
    };
    est.clamp(1, logical_available.max(1))
}

#[cfg_attr(not(any(test, target_os = "linux")), allow(dead_code))]
/// Parse a Linux cpu-list (`"0-3,8,10-11"`) into CPU indices; `None` on any malformed piece. Capped
/// at 1 << 16 CPUs so a corrupt range cannot allocate without bound.
fn parse_cpu_list(s: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (lo, hi) = match part.split_once('-') {
            Some((a, b)) => (
                a.trim().parse::<usize>().ok()?,
                b.trim().parse::<usize>().ok()?,
            ),
            None => {
                let v = part.parse::<usize>().ok()?;
                (v, v)
            }
        };
        if hi < lo || hi >= 1 << 16 {
            return None;
        }
        out.extend(lo..=hi);
    }
    (!out.is_empty()).then_some(out)
}

/// Exact usable-physical-core count on Linux: the distinct `(package, core)` pairs of the CPUs in
/// this process's affinity mask, capped by the quota-aware usable count. `None` off Linux or when
/// any file is unreadable.
#[cfg(target_os = "linux")]
fn linux_physical_in_affinity() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let list = status
        .lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))?;
    let cpus = parse_cpu_list(list)?;
    let read = |cpu: usize, f: &str| -> Option<i64> {
        std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/{f}"))
            .ok()?
            .trim()
            .parse::<i64>()
            .ok()
    };
    let mut cores = std::collections::BTreeSet::new();
    for cpu in cpus {
        cores.insert((read(cpu, "physical_package_id")?, read(cpu, "core_id")?));
    }
    let quota = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    Some(cores.len().min(quota).max(1))
}

#[cfg(not(target_os = "linux"))]
fn linux_physical_in_affinity() -> Option<usize> {
    None
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CacheSizes {
    l1d: Option<usize>,
    l2: Option<usize>,
    l3: Option<usize>,
}

#[cfg_attr(
    not(any(test, target_arch = "x86_64", target_os = "linux")),
    allow(dead_code)
)]
impl CacheSizes {
    /// Record one cache descriptor: `level` 1-3, `kind` 1 = data, 2 = instruction, 3 = unified.
    fn record(&mut self, level: u32, kind: u32, bytes: usize) {
        if kind == 2 || bytes == 0 {
            return; // instruction caches do not hold kernel operands
        }
        let slot = match level {
            1 => &mut self.l1d,
            2 => &mut self.l2,
            3 => &mut self.l3,
            _ => return,
        };
        *slot = Some(slot.map_or(bytes, |b| b.max(bytes)));
    }

    fn any(&self) -> bool {
        self.l1d.is_some() || self.l2.is_some() || self.l3.is_some()
    }
}

#[cfg_attr(not(any(test, target_arch = "x86_64")), allow(dead_code))]
/// Decode one CPUID deterministic-cache-parameters subleaf (leaf 4 / `0x8000_001D`, identical
/// layouts): `Some((level, kind, bytes))`, or `None` for the null descriptor that ends the list.
/// `kind`: EAX[4:0] (1 data, 2 instruction, 3 unified); `level`: EAX[7:5]; size =
/// ways (EBX[31:22]+1) × partitions (EBX[21:12]+1) × line (EBX[11:0]+1) × sets (ECX+1).
fn decode_cache_leaf(eax: u32, ebx: u32, ecx: u32) -> Option<(u32, u32, usize)> {
    let kind = eax & 0x1f;
    if kind == 0 {
        return None;
    }
    let level = (eax >> 5) & 0x7;
    let ways = ((ebx >> 22) & 0x3ff) as usize + 1;
    let parts = ((ebx >> 12) & 0x3ff) as usize + 1;
    let line = (ebx & 0xfff) as usize + 1;
    let sets = ecx as usize + 1;
    let bytes = ways
        .saturating_mul(parts)
        .saturating_mul(line)
        .saturating_mul(sets);
    Some((level, kind, bytes))
}

#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)] // `__cpuid_count` is an `unsafe fn` on older toolchains and safe on newer.
fn cpuid(leaf: u32, sub: u32) -> (u32, u32, u32) {
    // SAFETY: CPUID exists on every x86-64 processor and has no side effects.
    let r = unsafe { std::arch::x86_64::__cpuid_count(leaf, sub) };
    (r.eax, r.ebx, r.ecx)
}

#[cfg(target_arch = "x86_64")]
fn probe_caches_cpuid() -> Option<CacheSizes> {
    let walk = |leaf: u32| {
        let mut c = CacheSizes::default();
        for sub in 0..32 {
            let (a, b, cx) = cpuid(leaf, sub);
            match decode_cache_leaf(a, b, cx) {
                Some((level, kind, bytes)) => c.record(level, kind, bytes),
                None => break,
            }
        }
        c
    };
    let (max_basic, _, _) = cpuid(0, 0);
    if max_basic >= 4 {
        let c = walk(4);
        if c.any() {
            return Some(c);
        }
    }
    let (max_ext, _, _) = cpuid(0x8000_0000, 0);
    if max_ext >= 0x8000_001D {
        let c = walk(0x8000_001D);
        if c.any() {
            return Some(c);
        }
    }
    None
}

#[cfg(not(target_arch = "x86_64"))]
fn probe_caches_cpuid() -> Option<CacheSizes> {
    None
}

#[cfg_attr(not(any(test, target_os = "linux")), allow(dead_code))]
/// A sysfs cache `size` value (`"48K"`, `"2048K"`, `"24M"`, `"1G"`, or plain bytes) in bytes.
fn parse_sysfs_size(s: &str) -> Option<usize> {
    let s = s.trim();
    let (digits, scale) = match s.as_bytes().last()? {
        b'K' | b'k' => (&s[..s.len() - 1], 1usize << 10),
        b'M' | b'm' => (&s[..s.len() - 1], 1 << 20),
        b'G' | b'g' => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    digits
        .trim()
        .parse::<usize>()
        .ok()
        .map(|v| v.saturating_mul(scale))
        .filter(|&v| v > 0)
}

#[cfg(target_os = "linux")]
fn probe_caches_sysfs() -> Option<CacheSizes> {
    let mut c = CacheSizes::default();
    for idx in 0..16 {
        let dir = format!("/sys/devices/system/cpu/cpu0/cache/index{idx}");
        let Ok(level) = std::fs::read_to_string(format!("{dir}/level")) else {
            break;
        };
        let level = level.trim().parse::<u32>().ok()?;
        let kind = match std::fs::read_to_string(format!("{dir}/type")).ok()?.trim() {
            "Data" => 1,
            "Instruction" => 2,
            "Unified" => 3,
            _ => continue,
        };
        let bytes = parse_sysfs_size(&std::fs::read_to_string(format!("{dir}/size")).ok()?)?;
        c.record(level, kind, bytes);
    }
    c.any().then_some(c)
}

#[cfg(not(target_os = "linux"))]
fn probe_caches_sysfs() -> Option<CacheSizes> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_plausible_values() {
        let t = host_topology();
        assert!(t.logical_available >= 1);
        assert!((1..=t.logical_available).contains(&t.physical_available));
        assert!(t.physical_total >= 1);
        // Real caches are at least a few KiB and (one instance) at most a few GiB; an override can
        // only come from the environment, which the suite does not set.
        for b in [t.l1d_bytes, t.l2_bytes, t.l3_bytes].into_iter().flatten() {
            assert!(
                (1024..=(4usize << 30)).contains(&b),
                "implausible cache size {b}"
            );
        }
        if let (Some(l1), Some(l2)) = (t.l1d_bytes, t.l2_bytes) {
            assert!(l1 <= l2, "L1D {l1} larger than L2 {l2}");
        }
        assert!(["cpuid", "sysfs", "none"].contains(&t.cache_source));
        // Probed once: the second read is the same cached value.
        assert_eq!(*host_topology(), *t);
    }

    #[test]
    fn cpuid_cache_leaf_decodes_the_documented_fields() {
        // A 48 KiB, 12-way, 64-byte-line L1 data cache with 64 sets: kind 1, level 1.
        let eax = 1 | (1 << 5);
        let ebx = (11 << 22) | 63;
        assert_eq!(decode_cache_leaf(eax, ebx, 63), Some((1, 1, 48 * 1024)));
        // A 24 MiB, 12-way unified L3 with 32768 sets: kind 3, level 3.
        let eax = 3 | (3 << 5);
        assert_eq!(
            decode_cache_leaf(eax, (11 << 22) | 63, 32767),
            Some((3, 3, 24 << 20))
        );
        // The null descriptor ends the walk.
        assert_eq!(decode_cache_leaf(0, 0xffff_ffff, 0xffff_ffff), None);
        // Instruction caches are not recorded; the larger of two same-level data caches wins.
        let mut c = CacheSizes::default();
        c.record(1, 2, 64 * 1024);
        assert_eq!(c.l1d, None);
        c.record(2, 3, 2 << 20);
        c.record(2, 3, 4 << 20);
        assert_eq!(c.l2, Some(4 << 20));
        c.record(4, 3, 128 << 20);
        assert_eq!(c.l3, None, "an L4 is not an L3");
    }

    #[test]
    fn sysfs_and_cpu_list_parsers() {
        assert_eq!(parse_sysfs_size("48K\n"), Some(48 * 1024));
        assert_eq!(parse_sysfs_size("24M"), Some(24 << 20));
        assert_eq!(parse_sysfs_size("1G"), Some(1 << 30));
        assert_eq!(parse_sysfs_size("4096"), Some(4096));
        assert_eq!(parse_sysfs_size("0K"), None);
        assert_eq!(parse_sysfs_size("K"), None);
        assert_eq!(parse_sysfs_size(""), None);
        assert_eq!(
            parse_cpu_list("0-3,8,10-11\n"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_cpu_list("5"), Some(vec![5]));
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("a-b"), None);
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("0-70000"), None);
    }

    #[test]
    fn physical_estimate_respects_masks_and_ratios() {
        // Whole machine usable: the OS's physical count (a 6P(HT)+8E+2LPE part: 22 logical, 16 cores).
        assert_eq!(estimate_physical(22, 22, 16), 16);
        // No SMT: one core per logical CPU.
        assert_eq!(estimate_physical(16, 16, 16), 16);
        // A mask/quota of 8 of 32 logical on a 16-core 2-way SMT machine: 4 cores.
        assert_eq!(estimate_physical(8, 32, 16), 4);
        // Rounds up and never drops below one core or exceeds the usable logical count.
        assert_eq!(estimate_physical(1, 32, 16), 1);
        assert_eq!(estimate_physical(3, 32, 16), 2);
        assert_eq!(estimate_physical(4, 4, 64), 4);
    }

    /// P13.10: the private physical-core pool exists exactly when the usable cores carry SMT
    /// siblings to shed — and a non-SMT host, an affinity mask or a quota no longer makes the
    /// decision compare a machine-wide core count against a masked pool width.
    #[test]
    fn unified_pool_width_follows_usable_cores() {
        // This laptop: 16 usable cores under a 22-wide pool -> a 16-worker private pool.
        assert_eq!(private_pool_width(16, 22), Some(16));
        // SMT off / one thread per core: the global pool already is the physical-core pool.
        assert_eq!(private_pool_width(16, 16), None);
        // `taskset -c 0-7` on a 2-way SMT host: 4 usable cores under an 8-wide pool. The old
        // predicate compared the machine's 32 cores against 8 and built nothing.
        assert_eq!(private_pool_width(4, 8), Some(4));
        // An explicit thread request at or below the core count keeps one pool.
        assert_eq!(private_pool_width(16, 4), None);
        assert_eq!(private_pool_width(0, 4), None);
    }

    #[test]
    fn nt_threshold_tracks_the_last_level_cache() {
        assert_eq!(nt_threshold_from_l3(None), NT_FALLBACK_BYTES);
        assert_eq!(nt_threshold_from_l3(Some(0)), NT_FALLBACK_BYTES);
        // 24 MiB L3 (this laptop) -> 18 MiB; the old constant's "~10 MiB ~= this machine's L3" was
        // off by 2.4x against the hardware it named.
        assert_eq!(nt_threshold_from_l3(Some(24 << 20)), 18 << 20);
        // Clamped at both ends.
        assert_eq!(nt_threshold_from_l3(Some(2 << 20)), 4 << 20);
        assert_eq!(nt_threshold_from_l3(Some(512 << 20)), 64 << 20);
        assert_eq!(nt_threshold_from_l3(Some(usize::MAX)), 64 << 20);
    }

    #[test]
    fn knob_parsers_reject_junk() {
        assert_eq!(parse_threads(Some("8")), Some(8));
        assert_eq!(parse_threads(Some(" 3 ")), Some(3));
        for bad in [
            None,
            Some(""),
            Some("0"),
            Some("-1"),
            Some("+2"),
            Some("4097"),
            Some("x"),
        ] {
            assert_eq!(parse_threads(bad), None, "{bad:?}");
        }
        assert_eq!(parse_kib(Some("48")), Some(48 * 1024));
        assert_eq!(parse_kib(Some("0")), None);
        assert_eq!(parse_kib(Some(&usize::MAX.to_string())), Some(usize::MAX));
        assert_eq!(parse_positive(Some("12")), Some(12));
        assert_eq!(parse_positive(Some("1.5")), None);
    }
}
