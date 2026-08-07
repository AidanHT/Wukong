//! Persistent **cubin cache** — skip the driver's PTX→SASS JIT on warm starts (milestone M10).
//!
//! Wukong's GPU path emits PTX and hands it to the driver, which JIT-compiles PTX→SASS on every
//! `cuModuleLoadData`. Within one process the [`crate::gpu::Gpu`] module map already avoids re-JIT,
//! but a fresh process (each `wukongc --backend=gpu` invocation, each test-binary run) pays the JIT
//! again. cuBLAS, by contrast, ships precompiled SASS and has *zero* compile cost — so to be "tied
//! with cuBLAS warm" we must cache the compiled cubin and load that instead of re-JITing.
//!
//! The driver exposes its in-process JIT compiler through `cuLink*` (no external `ptxas` needed):
//! [`ptx_to_cubin`] runs it and hands back the SASS image, which we persist keyed by a hash of the
//! PTX, **the device's own `sm_XX`** ([`device_sm_arch`]) and the driver version (a cubin is
//! driver-ABI specific). Warm loads then go through cudarc's `Ptx::from_file` → `cuModuleLoad`,
//! which auto-detects the cubin and loads it with no compilation. Every step degrades gracefully: a
//! missing/stale/incompatible cubin, an unavailable linker, or an unwritable cache dir all fall back
//! to the proven direct-PTX JIT (see `Gpu::load_module_cached`).
//!
//! **Why the device arch is in the key** (it used to be absent, on the stated assumption that "the
//! PTX text embeds `.target sm_89`, so the arch is in the key"). Two facts kill that assumption:
//!
//! 1. After the header retarget (`ptx_target`, GPU_RETARGET_PLAN.md §5 Phase 2 step 2) a module's
//!    PTX carries the **family floor** — `.target sm_80` for the Ampere-legal majority — not the
//!    device. One PTX text is now shared by every part from an A100 to a 5090, so the PTX hash no
//!    longer separates them at all.
//! 2. A cubin is **SASS**, which the driver compiles for the *current device*, not for the PTX's
//!    `.target`. And SASS binary compatibility runs forward across minor revisions only (CUDA
//!    Programming Guide, "Binary Compatibility": a cubin for `X.y` runs on `X.z` iff `z >= y`). So
//!    an `sm_80` cubin JIT'd on an A100 **loads without error on an Ada `sm_89` part** — the
//!    unkeyed cache's failure mode was not a loud `NO_BINARY_FOR_GPU`, it was silently running
//!    Ampere SASS on Ada. The reverse direction (Ada's cubin on an A100) *is* loud, and the
//!    cross-major pairs (`sm_90`, `sm_120`) are loud too — which is exactly why the silent direction
//!    had to be closed by the key rather than left to the loader.
//!
//! An entry written before this change has the old, arch-free file name, so it can never be
//! *loaded* by the new key — it is a clean miss that re-JITs (and the old files are inert litter in
//! a cache dir the docs already call trivially clearable).

use std::ffi::{c_void, CString};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use cudarc::driver::{sys, DriverError};

/// Compile PTX text to a cubin (SASS) image via the driver's JIT linker (`cuLinkCreate` /
/// `cuLinkAddData` / `cuLinkComplete`). This runs the very same in-driver `ptxas` a plain module load
/// would, but returns the compiled bytes so they can be cached. The link state owns the returned
/// buffer until `cuLinkDestroy`, so we copy it out first.
pub fn ptx_to_cubin(ptx: &str) -> Result<Vec<u8>, DriverError> {
    // PTX is ASCII with no interior NUL; the driver wants a NUL-terminated buffer (size incl. NUL).
    let ptx_c = CString::new(ptx).map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE))?;
    let name = CString::new("wukong_kernel").unwrap();
    unsafe {
        let mut state: sys::CUlinkState = std::ptr::null_mut();
        sys::cuLinkCreate_v2(0, std::ptr::null_mut(), std::ptr::null_mut(), &mut state).result()?;
        // From here on, ensure the state is destroyed on every exit path.
        let add = sys::cuLinkAddData_v2(
            state,
            sys::CUjitInputType::CU_JIT_INPUT_PTX,
            ptx_c.as_ptr() as *mut c_void,
            ptx.len() + 1, // include the trailing NUL
            name.as_ptr(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
        .result();
        if let Err(e) = add {
            sys::cuLinkDestroy(state);
            return Err(e);
        }
        let mut cubin_out: *mut c_void = std::ptr::null_mut();
        let mut size_out: usize = 0;
        let complete = sys::cuLinkComplete(state, &mut cubin_out, &mut size_out).result();
        if let Err(e) = complete {
            sys::cuLinkDestroy(state);
            return Err(e);
        }
        // Copy before destroying the link state, which frees `cubin_out`.
        let bytes = std::slice::from_raw_parts(cubin_out as *const u8, size_out).to_vec();
        sys::cuLinkDestroy(state);
        Ok(bytes)
    }
}

/// The installed CUDA driver version (e.g. 12090), or 0 if it can't be queried. Part of the cache key
/// because a cubin's SASS/ABI is tied to the driver that produced it.
pub fn driver_version() -> i32 {
    let mut v: i32 = 0;
    unsafe {
        if sys::cuDriverGetVersion(&mut v).result().is_err() {
            return 0;
        }
    }
    v
}

/// Directory holding cached cubins. Overridable via `WUKONG_CUBIN_CACHE`; defaults to a subdir of the
/// system temp dir so it survives across process runs but is trivially clearable.
pub fn cache_dir() -> PathBuf {
    std::env::var_os("WUKONG_CUBIN_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("wukong_cubin_cache"))
}

/// The arch component used when the device's compute capability cannot be queried. A self-consistent
/// bucket that no real device shares (every probed arch spells `sm_<major><minor>`), so an unprobed
/// run can neither read nor poison a probed run's entries.
pub const UNKNOWN_ARCH: &str = "smunknown";

/// `sm_XX` for the device this process will JIT for — the **device** half of the cache key.
///
/// Probed through the raw driver (like [`driver_version`]) instead of taken from `Gpu::target()`, so
/// [`cache_path`] keeps its two-argument shape and no caller has to thread an arch through. It asks
/// `cuCtxGetDevice` first — the device of the context current on this thread is the device
/// `cuLinkComplete` compiles SASS for — and falls back to ordinal 0, the device `Gpu::new` retains,
/// because `Gpu::load_module_cached` computes the path *before* it binds the context. Both are
/// device-level queries needing only `cuInit`; if neither answers (no driver, no device, a plain
/// unit test) the result is [`UNKNOWN_ARCH`].
///
/// Deliberately **not** memoized: a compute capability cannot change under a running process, but a
/// process that later drives a *second* device must not stamp device 0's arch onto its cubins. Two
/// integer attribute queries are noise next to the PTX→SASS compile this key exists to skip.
pub fn device_sm_arch() -> String {
    unsafe {
        let mut dev: sys::CUdevice = 0;
        if sys::cuCtxGetDevice(&mut dev).result().is_err() {
            dev = 0;
        }
        use sys::CUdevice_attribute as A;
        let (mut major, mut minor) = (0i32, 0i32);
        let ok = sys::cuDeviceGetAttribute(&mut major, A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
            .result()
            .is_ok()
            && sys::cuDeviceGetAttribute(&mut minor, A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)
                .result()
                .is_ok();
        if ok {
            crate::ptx_target::sm_arch(major, minor)
        } else {
            UNKNOWN_ARCH.to_string()
        }
    }
}

/// Cache file path for `ptx` compiled by driver `tag` **for device arch `arch`** — the pure,
/// device-free form of [`cache_path`], so the key's shape is testable on a box with no GPU.
///
/// `arch` is keyed twice on purpose: it is hashed *with* the PTX **and** spelled in the file name.
/// The hash makes the key unforgeable, the literal name makes a cache dir readable (`ls` says which
/// device each cubin is for) and guarantees that an entry written before the arch existed in the key
/// — `drv12090-<len>-<hash>.cubin` — cannot collide with a new one, so it is a miss, never a
/// wrong-arch hit. The 64-bit hash is SipHash via `DefaultHasher` (deterministic across runs);
/// collisions are astronomically unlikely, and a wrong/stale hit merely fails to load and recompiles.
pub fn cache_path_for(ptx: &str, tag: i32, arch: &str) -> PathBuf {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ptx.hash(&mut h);
    arch.hash(&mut h);
    let hash = h.finish();
    cache_dir().join(format!("drv{tag}-{arch}-{}-{hash:016x}.cubin", ptx.len()))
}

/// Cache file path for `ptx` under driver `tag`, on the device this process is JIT-ing for. Thin
/// wrapper over [`cache_path_for`] that fills the arch from [`device_sm_arch`].
pub fn cache_path(ptx: &str, tag: i32) -> PathBuf {
    cache_path_for(ptx, tag, &device_sm_arch())
}

/// Write `bytes` to `path` atomically (write a per-process temp file, then rename), creating the
/// cache dir if needed. Best-effort: any I/O error is returned for the caller to ignore and fall back.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    // Rename is atomic on the same filesystem; if a concurrent writer won the race that's fine.
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The PTX every arch-key gate below hashes: the floored header the retarget now emits, which is
    /// precisely the text that is *identical* on an A100 and on this Ada part.
    const FLOORED_PTX: &str = "\
.version 7.8
.target sm_80
.address_size 64
.visible .entry k() { ret; }
";

    /// **One floored PTX, two devices, two cache entries** — the defect this key closes.
    ///
    /// Before the arch joined the key, `cache_path` hashed the PTX text and the driver version only,
    /// on the assumption that `.target sm_89` inside the PTX carried the arch. The retarget makes
    /// every Ampere-legal module say `.target sm_80` on *every* device, so that assumption now maps
    /// an A100's SASS and a 4050's SASS onto one file name — and an `sm_80` cubin loads happily on
    /// `sm_89` (binary compat is forward across minor revisions), so the wrong-arch hit would be
    /// silent. Both halves of the key are checked: the file names differ, and they still differ once
    /// the literal arch is spliced out, which is only true if the hash covers the arch too.
    #[test]
    fn cache_key_separates_devices_that_share_one_floored_ptx() {
        let drv = 12090;
        let a100 = cache_path_for(FLOORED_PTX, drv, "sm_80");
        let ada = cache_path_for(FLOORED_PTX, drv, "sm_89");
        let hopper = cache_path_for(FLOORED_PTX, drv, "sm_90");
        let blackwell = cache_path_for(FLOORED_PTX, drv, "sm_120");
        let all = [&a100, &ada, &hopper, &blackwell];
        for (i, p) in all.iter().enumerate() {
            for q in all.iter().skip(i + 1) {
                assert_ne!(p, q, "two devices must never share one cubin cache entry");
            }
        }
        // The hash — not just the printed name — must cover the arch: erase the arch text from each
        // name and the remainders must STILL differ.
        let bare = |p: &PathBuf, arch: &str| p.file_name().unwrap().to_string_lossy().replace(arch, "");
        assert_ne!(
            bare(&a100, "sm_80"),
            bare(&ada, "sm_89"),
            "the arch is only spelled in the file name, not hashed — splice it out and the key collides"
        );
        // The arch is legible in the name, so a cache dir listing says which device each cubin is for.
        assert!(ada.file_name().unwrap().to_string_lossy().contains("sm_89"));
    }

    /// The other two key components still bite, and the key is a pure function of its inputs (a
    /// non-deterministic key would silently disable the cache — always a miss, always a re-JIT).
    #[test]
    fn cache_key_is_deterministic_and_still_covers_ptx_and_driver() {
        let ada = |ptx: &str, drv: i32| cache_path_for(ptx, drv, "sm_89");
        assert_eq!(ada(FLOORED_PTX, 12090), ada(FLOORED_PTX, 12090));
        assert_ne!(ada(FLOORED_PTX, 12090), ada(FLOORED_PTX, 12080), "driver version must key");
        let other = FLOORED_PTX.replace("entry k()", "entry j()");
        assert_ne!(ada(FLOORED_PTX, 12090), ada(&other, 12090), "PTX text must key");
    }

    /// **A pre-change entry must be a clean MISS, never a wrong-arch load.** The old name shape is
    /// reconstructed here exactly as the previous `cache_path` built it; the new key cannot produce
    /// it for any arch, so `path.exists()` is false and `load_module_cached` takes the cold branch.
    #[test]
    fn a_pre_change_cache_entry_can_never_be_hit() {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        FLOORED_PTX.hash(&mut h);
        let old = cache_dir().join(format!("drv12090-{}-{:016x}.cubin", FLOORED_PTX.len(), h.finish()));
        for arch in ["sm_80", "sm_89", "sm_90", "sm_120", UNKNOWN_ARCH] {
            assert_ne!(cache_path_for(FLOORED_PTX, 12090, arch), old, "arch {arch} rehabilitated a stale entry");
        }
    }

    /// [`cache_path`] must be exactly [`cache_path_for`] at the probed arch — otherwise the gates
    /// above pin a key the loader does not actually use. Also pins that a device-less environment
    /// answers [`UNKNOWN_ARCH`] rather than guessing an arch (this test has no context of its own; on
    /// a box where another test already built the `Gpu`, the probe legitimately answers `sm_XX`).
    #[test]
    fn cache_path_uses_the_probed_device_arch() {
        let arch = device_sm_arch();
        assert!(
            arch == UNKNOWN_ARCH || (arch.starts_with("sm_") && arch[3..].chars().all(|c| c.is_ascii_digit())),
            "device_sm_arch answered {arch:?}, which is neither a device arch nor the unknown bucket"
        );
        assert_eq!(cache_path(FLOORED_PTX, 12090), cache_path_for(FLOORED_PTX, 12090, &arch));
    }

    /// **An unusable entry at the new key must degrade to the direct PTX JIT** — the promise this
    /// module's docs make, and the one that makes an arch-keyed cache safe to change at all. A cubin
    /// that the driver refuses (here: a file that is neither SASS nor PTX, standing in for the
    /// wrong-arch and driver-mismatch cases that are hard to fabricate on one box) must not fail the
    /// load: `load_module_cached` drops it, re-JITs, and re-persists. Driven through the public
    /// `Gpu::function`, since the caching wrapper itself is private.
    #[test]
    fn a_corrupt_cache_entry_degrades_to_the_ptx_jit() {
        let mut guard = crate::gpu::gpu();
        let Some(g) = guard.as_mut() else {
            let why = crate::gpu::init_error().unwrap_or("no CUDA device reachable");
            assert!(!crate::gpu::gpu_required(), "WUKONG_GPU_REQUIRED is set but the GPU is unusable: {why}");
            eprintln!("[skip] a_corrupt_cache_entry_degrades_to_the_ptx_jit: GPU unavailable: {why}");
            return;
        };
        // A module no other test shares, so this cannot race for a cache path.
        let ptx = format!(
            "{}.visible .entry wukong_cubin_probe()\n{{\n    ret;\n}}\n",
            crate::ptx_target::HDR_SM80
        );
        let path = cache_path(&ptx, driver_version());
        write_atomic(&path, b"neither SASS nor PTX").expect("plant a corrupt cache entry");
        let f = g.function("wukong_cubin_probe", &ptx, "wukong_cubin_probe");
        assert!(f.is_ok(), "a corrupt cache entry broke a load that plain PTX JIT would have served: {f:?}");
        let after = std::fs::read(&path).unwrap_or_default();
        assert!(
            after.len() > 64 && after.starts_with(b"\x7fELF"),
            "the corrupt entry was not replaced by a freshly JIT'd cubin ({} bytes)",
            after.len()
        );
        let _ = std::fs::remove_file(&path);
        eprintln!("[gate] corrupt cubin at {path:?} -> dropped, re-JIT'd, re-persisted ✓");
    }

    /// On a real device, the probe must agree with `Gpu::target()` — the campaign's single source of
    /// device identity. A drift here would key the cache to a *different* arch than the one the
    /// backend gates fp8/SMEM decisions on.
    #[test]
    fn probed_arch_agrees_with_gpu_target() {
        let mut guard = crate::gpu::gpu();
        let Some(g) = guard.as_mut() else {
            let why = crate::gpu::init_error().unwrap_or("no CUDA device reachable");
            assert!(!crate::gpu::gpu_required(), "WUKONG_GPU_REQUIRED is set but the GPU is unusable: {why}");
            eprintln!("[skip] probed_arch_agrees_with_gpu_target: GPU unavailable: {why}");
            return;
        };
        let t = g.target();
        let want = crate::ptx_target::sm_arch(t.cc_major, t.cc_minor);
        assert_eq!(device_sm_arch(), want, "cubin cache key arch != Gpu::target()'s arch");
        eprintln!("[gate] cubin cache keys on the probed device arch {want} ({}) ✓", t.name);
    }
}
