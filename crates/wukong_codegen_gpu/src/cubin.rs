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
//! PTX (the PTX text embeds `.target sm_89`, so the arch is in the key) plus the driver version (a
//! cubin is driver-ABI specific). Warm loads then go through cudarc's `Ptx::from_file` →
//! `cuModuleLoad`, which auto-detects the cubin and loads it with no compilation. Every step degrades
//! gracefully: a missing/stale/incompatible cubin, an unavailable linker, or an unwritable cache dir
//! all fall back to the proven direct-PTX JIT (see `Gpu::load_module_cached`).

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

/// Cache file path for `ptx` under driver `tag`. The name folds in the PTX length and a 64-bit hash
/// (SipHash via `DefaultHasher`, deterministic across runs) — the arch lives inside the PTX text, so
/// it is covered by the hash. Collisions are astronomically unlikely, and a wrong/stale hit merely
/// fails to load and triggers a recompile.
pub fn cache_path(ptx: &str, tag: i32) -> PathBuf {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ptx.hash(&mut h);
    let hash = h.finish();
    cache_dir().join(format!("drv{tag}-{}-{hash:016x}.cubin", ptx.len()))
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
