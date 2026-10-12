//! Memory-mapped loads: the runtime of `map_<T>(path)`, `st_map_<T>(path, name)`,
//! `npy_map_<T>(path)` and `gguf_map_<T>(path, name)`, each of which returns a `[]T` slice holding a
//! file's bytes — the whole file, or one tensor of a safetensors / `.npy` / GGUF container — that
//! `free(s)` releases like any `alloc_<T>` buffer.
//!
//! # Semantics: a copy, made lazily
//!
//! The slice behaves exactly like a heap buffer that was filled from the file when the builtin
//! returned: it can be read and **written**, a write changes the buffer and never the file, and
//! `free` releases it. That is what the interpreter does — it allocates a block and decodes the
//! bytes into it — and it is the contract the native backend keeps with a **copy-on-write private
//! mapping** (`mmap(MAP_PRIVATE)` on Unix, `MapViewOfFile(FILE_MAP_COPY)` on Windows): no byte is
//! read until a page is first touched, untouched pages are the OS page cache's own, and a load of a
//! multi-gigabyte checkpoint returns at once. Where a mapping is impossible — a payload offset that
//! is not a multiple of the element size (a safetensors tensor after an odd-length half-precision
//! one), a big-endian host, an OS that refuses — the native backend falls back to that same copy
//! (a [`crate::wukong_rt_alloc`] block filled by one read), so the builtin never fails for a reason
//! the interpreter does not share.
//!
//! The one thing a mapping adds is the usual one: the file must not be truncated or rewritten while
//! the slice is live (on Unix a page past a truncated end faults; on Windows the rewrite itself
//! fails while a view is open), and pages not yet touched may observe another process's write.
//! Programs that rewrite a file they hold mapped are outside the differential contract.
//!
//! # What may be mapped
//!
//! A map **never converts**: the element type must be the stored one — a safetensors or `.npy`
//! tensor's own dtype, a plain GGUF tensor's own type, or `u8` for a block-quantized GGUF tensor's
//! raw blocks (the `gemv_q8_0` / `gemv_q4_0` operand). A `bf16` checkpoint maps as `[]bf16`, not
//! `[]f32`; `st_read_f32` is the converting load. `map_<T>` maps the whole file as
//! `file_size / sizeof(T)` elements (a trailing partial element is not part of the slice, as with
//! `read_<T>`).
//!
//! Any failure — the file cannot be opened, the container is malformed, there is no such tensor,
//! the dtype is not the stored one, a column-major `.npy` — returns the **empty slice** (null data,
//! length 0), which `free` accepts, as does an empty payload. The matching reader names the cause:
//! `st_read_<T>(path, name, buf)` with an empty `buf` reports the same `-1`/`-3`/`-5`/`-6`, and `-4`
//! when it could load the tensor — including by a `bf16`/`f16` → `f32` widening, which a map never
//! performs.
//!
//! # Release
//!
//! Every live mapping is recorded in a small registry keyed by the slice's data pointer;
//! [`crate::wukong_rt_free`] consults it first (one atomic load while nothing is mapped) and unmaps
//! a registered pointer instead of handing it to the heap. Freeing a pointer that is neither — or
//! the same one twice — is undefined, exactly as for the heap.
//!
//! The OS calls are declared here directly (`mmap`/`munmap` from the C library, the four
//! `kernel32` mapping functions) rather than through the `libc` / `windows-sys` crates: both
//! libraries are linked by `std` already, while `windows-sys` reaches `kernel32` through
//! `windows-targets`' import libraries, whose search path a `--emit=exe` link of the runtime rlib
//! never sees.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::fileio::{
    self, dtype_size, Container, DT_U8, IO_DTYPE_MISMATCH, IO_ERROR, IO_OPEN_FAILED,
    IO_SHAPE_MISMATCH,
};

/// `map_<T>(path)`: the whole file.
pub const MAP_FILE: i64 = 0;
/// `st_map_<T>(path, name)`: one safetensors tensor.
pub const MAP_SAFETENSORS: i64 = 1;
/// `npy_map_<T>(path)`: the array of a `.npy` file.
pub const MAP_NPY: i64 = 2;
/// `gguf_map_<T>(path, name)`: one GGUF tensor (raw blocks for a quantized one).
pub const MAP_GGUF: i64 = 3;

/// The mapping granularity: a view starts at the payload's offset rounded down to 64 KiB, a
/// multiple of every page size this runs on (4, 16 and 64 KiB) and of Windows' allocation
/// granularity.
const GRANULE: u64 = 1 << 16;

/// Where a map's bytes lie: the open file, the payload's byte offset, and its element count.
pub struct MapPlan {
    pub file: File,
    pub off: u64,
    pub count: usize,
}

/// Open `path` and locate the bytes a map of `container` (`MAP_*`; `name` for the named
/// containers) yields as elements of dtype `dt`, with every check the readers make, in their
/// order. Shared by the native entry and the interpreter, so both fail on exactly the same files.
pub fn map_plan(path: &Path, container: i64, name: &str, dt: i64) -> Result<MapPlan, i64> {
    let sz = dtype_size(dt).ok_or(IO_ERROR)? as u64;
    let count = |n: u64| usize::try_from(n).map_err(|_| IO_ERROR);
    match container {
        MAP_FILE => {
            let file = File::open(path).map_err(|_| IO_OPEN_FAILED)?;
            let len = file.metadata().map_err(|_| IO_ERROR)?.len();
            Ok(MapPlan {
                file,
                off: 0,
                count: count(len / sz)?,
            })
        }
        MAP_SAFETENSORS | MAP_NPY => {
            let c = if container == MAP_NPY {
                Container::Npy
            } else {
                Container::Safetensors(name)
            };
            let (file, loc) = fileio::open_container(path, &c)?;
            if loc.dtype != dt {
                return Err(IO_DTYPE_MISMATCH);
            }
            if loc.fortran && loc.shape.len() >= 2 {
                return Err(IO_SHAPE_MISMATCH);
            }
            Ok(MapPlan {
                file,
                off: loc.data_off,
                count: count(fileio::shape_count(&loc.shape)?)?,
            })
        }
        MAP_GGUF => {
            let (file, g, file_len) = crate::gguf::gguf_open(path, false)?;
            let loc = crate::gguf::gguf_locate(&g, name.as_bytes(), file_len)?;
            let n = match loc.ty.dt {
                Some(t) if t == dt => fileio::shape_count(&loc.shape)?,
                None if dt == DT_U8 => loc.nbytes,
                _ => return Err(IO_DTYPE_MISMATCH),
            };
            Ok(MapPlan {
                file,
                off: loc.data_off,
                count: count(n)?,
            })
        }
        _ => Err(IO_ERROR),
    }
}

// --- the registry ----------------------------------------------------------------------------

/// How many mappings are live: [`release`]'s fast path while it is zero.
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// One live mapping: the slice's data pointer, and the view it lies in.
struct Mapping {
    data: usize,
    base: usize,
    len: usize,
}

/// Every live mapping.
static MAPS: Mutex<Vec<Mapping>> = Mutex::new(Vec::new());

fn maps() -> std::sync::MutexGuard<'static, Vec<Mapping>> {
    MAPS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The number of live mappings (a diagnostics hook).
pub fn live_mappings() -> usize {
    LIVE.load(Ordering::Acquire)
}

/// Whether `data` is the data pointer of a live mapping (rather than a heap copy).
pub fn is_mapped(data: *const u8) -> bool {
    LIVE.load(Ordering::Acquire) != 0 && maps().iter().any(|m| m.data == data as usize)
}

/// Unmap `data` if it is the data pointer of a live mapping, returning whether it was — the first
/// thing [`crate::wukong_rt_free`] does.
pub(crate) fn release(data: *mut u8) -> bool {
    if LIVE.load(Ordering::Acquire) == 0 {
        return false;
    }
    let found = {
        let mut maps = maps();
        let at = maps.iter().position(|m| m.data == data as usize);
        at.map(|i| maps.swap_remove(i))
    };
    let Some(m) = found else {
        return false;
    };
    LIVE.fetch_sub(1, Ordering::AcqRel);
    // SAFETY: `m` was registered by `os_map` for a view of exactly `len` bytes at `base` that
    // nothing else unmaps, and it has just been removed, so it is unmapped once.
    unsafe { sys::unmap(m.base as *mut u8, m.len) };
    true
}

/// Map `nbytes` bytes of `file` from `off` copy-on-write, registering the view; `None` when the OS
/// declines.
fn os_map(file: &File, off: u64, nbytes: usize) -> Option<*mut u8> {
    let start = off - off % GRANULE;
    let delta = usize::try_from(off - start).ok()?;
    let len = delta.checked_add(nbytes)?;
    let base = sys::map(file, start, len)?;
    // SAFETY: the view is `len = delta + nbytes` bytes long, so `base + delta` is inside it.
    let data = unsafe { base.add(delta) };
    maps().push(Mapping {
        data: data as usize,
        base: base as usize,
        len,
    });
    LIVE.fetch_add(1, Ordering::AcqRel);
    Some(data)
}

/// The native result of a successful plan: a copy-on-write view of the payload when the payload is
/// aligned for its elements on a little-endian host and the OS maps it, else a heap copy. `(null,
/// 0)` for an empty payload or a failed copy.
fn map_native(plan: MapPlan, dt: i64) -> (*mut u8, usize) {
    let null = (std::ptr::null_mut(), 0);
    let Some(sz) = dtype_size(dt) else {
        return null;
    };
    let Some(nbytes) = plan.count.checked_mul(sz) else {
        return null;
    };
    if plan.count == 0 {
        return null;
    }
    if plan.off.is_multiple_of(sz as u64) && cfg!(target_endian = "little") {
        if let Some(p) = os_map(&plan.file, plan.off, nbytes) {
            return (p, plan.count);
        }
    }
    let p = crate::wukong_rt_alloc(plan.count as i64, sz as i64, 0);
    if p.is_null() {
        return null;
    }
    // SAFETY: `wukong_rt_alloc` returned `count * sz` writable bytes at `p`.
    let out = unsafe { std::slice::from_raw_parts_mut(p, nbytes) };
    let mut file = plan.file;
    if file.seek(SeekFrom::Start(plan.off)).is_err() || file.read_exact(out).is_err() {
        crate::wukong_rt_free(p);
        return null;
    }
    fileio::le_to_host_in_place(out, sz);
    (p, plan.count)
}

/// `wukong_rt_map(path, name, container, dtype, out_len) -> *mut u8` — the data pointer of a `[]T`
/// holding the bytes [`map_plan`] locates (see the module docs), its element count stored to
/// `*out_len`; null and `0` on any failure or an empty payload.
///
/// # Safety
/// `path` and `name` must be null or NUL-terminated (`name` is read only for the named
/// containers); `out_len` must address one writable `i64`.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_map(
    path: *const u8,
    name: *const u8,
    container: i64,
    dtype: i64,
    out_len: *mut i64,
) -> *mut u8 {
    // P14.5: debug-build contract checks (the `# Safety` section above); `path`/`name` are
    // NUL-terminated strings (no extent to check).
    crate::safety::debug_extent(out_len, 1);
    // SAFETY: forwarded from this function's contract.
    let path = unsafe { crate::rt_path(path) };
    let name = if matches!(container, MAP_SAFETENSORS | MAP_GGUF) {
        // SAFETY: forwarded from this function's contract.
        unsafe { fileio::cstr_lossy(name) }
    } else {
        String::new()
    };
    let (data, len) = match path.map(|p| map_plan(&p, container, &name, dtype)) {
        Some(Ok(plan)) => map_native(plan, dtype),
        _ => (std::ptr::null_mut(), 0),
    };
    // SAFETY: the caller guarantees one writable `i64` at `out_len`.
    unsafe { out_len.write(len as i64) };
    data
}

// --- the OS layer -----------------------------------------------------------------------------

#[cfg(all(unix, target_pointer_width = "64"))]
mod sys {
    use std::ffi::{c_int, c_void};
    use std::fs::File;
    use std::os::fd::AsRawFd;

    // The same values on Linux, the BSDs and macOS.
    const PROT_READ: c_int = 1;
    const PROT_WRITE: c_int = 2;
    const MAP_PRIVATE: c_int = 2;

    extern "C" {
        // `off_t` is 64-bit on every 64-bit Unix (and `mmap` is glibc's `mmap64` there).
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            off: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }

    /// A private, copy-on-write, read-write view of `len` bytes of `file` from `off` (a multiple
    /// of the page size).
    pub(super) fn map(file: &File, off: u64, len: usize) -> Option<*mut u8> {
        let off = i64::try_from(off).ok()?;
        // SAFETY: a fresh mapping of an open descriptor; the kernel validates every argument.
        let p = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE,
                file.as_raw_fd(),
                off,
            )
        };
        // `MAP_FAILED` is `(void *)-1`.
        (p as isize != -1 && !p.is_null()).then_some(p as *mut u8)
    }

    /// Release a view [`map`] returned.
    ///
    /// # Safety
    /// `base` and `len` must be exactly a live view's.
    pub(super) unsafe fn unmap(base: *mut u8, len: usize) {
        // SAFETY: forwarded from this function's contract.
        unsafe { munmap(base as *mut c_void, len) };
    }
}

#[cfg(windows)]
mod sys {
    use std::ffi::c_void;
    use std::fs::File;
    use std::os::windows::io::AsRawHandle;

    const PAGE_WRITECOPY: u32 = 0x08;
    const FILE_MAP_COPY: u32 = 0x01;

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateFileMappingW(
            file: *mut c_void,
            attributes: *const c_void,
            protect: u32,
            max_size_high: u32,
            max_size_low: u32,
            name: *const u16,
        ) -> *mut c_void;
        fn MapViewOfFile(
            mapping: *mut c_void,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(base: *const c_void) -> i32;
        fn CloseHandle(handle: *mut c_void) -> i32;
    }

    /// A copy-on-write view of `len` bytes of `file` from `off` (a multiple of the allocation
    /// granularity). The mapping object is closed at once: the view keeps it alive.
    pub(super) fn map(file: &File, off: u64, len: usize) -> Option<*mut u8> {
        // SAFETY: the handle is an open file's; a null mapping or view is checked before use, and
        // the mapping handle is closed exactly once.
        unsafe {
            let h = CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_WRITECOPY,
                0,
                0,
                std::ptr::null(),
            );
            if h.is_null() {
                return None;
            }
            let v = MapViewOfFile(h, FILE_MAP_COPY, (off >> 32) as u32, off as u32, len);
            CloseHandle(h);
            (!v.is_null()).then_some(v as *mut u8)
        }
    }

    /// Release a view [`map`] returned.
    ///
    /// # Safety
    /// `base` must be a live view's base address.
    pub(super) unsafe fn unmap(base: *mut u8, _len: usize) {
        // SAFETY: forwarded from this function's contract.
        unsafe { UnmapViewOfFile(base as *const c_void) };
    }
}

/// No mapping on other targets: every map takes the copy path.
#[cfg(not(any(all(unix, target_pointer_width = "64"), windows)))]
mod sys {
    pub(super) fn map(_file: &std::fs::File, _off: u64, _len: usize) -> Option<*mut u8> {
        None
    }

    /// Never called: [`map`] never succeeds.
    ///
    /// # Safety
    /// Trivially safe; `unsafe` only to match the mapping targets' signature.
    pub(super) unsafe fn unmap(_base: *mut u8, _len: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fileio::{Scratch, DT_BF16, DT_F32, DT_I16, IO_NAME_NOT_FOUND};

    /// Call the native entry; returns the data pointer and the stored length.
    fn map(path: *const u8, name: &str, container: i64, dt: i64) -> (*mut u8, i64) {
        let name = std::ffi::CString::new(name).unwrap();
        let mut len = -99i64;
        // SAFETY: both strings are NUL-terminated and `len` is one writable i64.
        let p = unsafe { wukong_rt_map(path, name.as_ptr() as *const u8, container, dt, &mut len) };
        (p, len)
    }

    /// The `n` elements at `p` as a vector (`p` addresses `n` readable `T`).
    fn lanes<T: Copy>(p: *mut u8, n: i64) -> Vec<T> {
        // SAFETY: the caller passes a live map result of `n` elements of `T`.
        unsafe { std::slice::from_raw_parts(p as *const T, n as usize).to_vec() }
    }

    #[test]
    fn a_whole_file_maps_copy_on_write_and_frees() {
        let vals = [1.5f32, -2.0, 3.25, 0.0];
        let mut bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        bytes.push(0xAB); // a trailing partial element is not part of the slice
        let f = Scratch::new("map_whole", &bytes);
        let (p, n) = map(f.ptr(), "", MAP_FILE, DT_F32);
        assert_eq!(n, 4);
        assert_eq!(lanes::<f32>(p, n), vals);
        // A write changes the buffer, never the file.
        // SAFETY: `p` addresses 4 writable f32 (a private copy-on-write view).
        unsafe { (p as *mut f32).write(7.0) };
        assert_eq!(lanes::<f32>(p, n)[0], 7.0);
        assert_eq!(std::fs::read(&f.0).unwrap(), bytes);
        if cfg!(any(all(unix, target_pointer_width = "64"), windows)) {
            assert!(is_mapped(p), "an aligned payload is mapped");
        }
        crate::wukong_rt_free(p);
        // u16 over 17 bytes: eight whole elements.
        let (p, n) = map(f.ptr(), "", MAP_FILE, crate::fileio::DT_U16);
        assert_eq!(n, 8);
        crate::wukong_rt_free(p);
    }

    #[test]
    fn container_maps_check_like_the_readers_and_never_convert() {
        // "b" BF16 [3] at [0, 6), then "w" F32 [2] at [6, 14): misaligned for f32, so it is copied.
        let mut payload = Vec::new();
        for bits in [0x3F80u16, 0xC000, 0x4040] {
            payload.extend_from_slice(&bits.to_le_bytes());
        }
        for v in [0.5f32, -8.0] {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let header = r#"{"b":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]},"w":{"dtype":"F32","shape":[2],"data_offsets":[6,14]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&payload);
        let f = Scratch::new("map_st", &bytes);
        let (p, n) = map(f.ptr(), "b", MAP_SAFETENSORS, DT_BF16);
        assert_eq!(lanes::<u16>(p, n), [0x3F80, 0xC000, 0x4040]);
        crate::wukong_rt_free(p);
        let (p, n) = map(f.ptr(), "w", MAP_SAFETENSORS, DT_F32);
        assert!(!is_mapped(p), "a misaligned payload is a heap copy");
        assert_eq!(lanes::<f32>(p, n), [0.5, -8.0]);
        // SAFETY: the copy is writable like any heap block.
        unsafe { (p as *mut f32).write(1.0) };
        crate::wukong_rt_free(p);
        // No widening, no missing names, no wrong dtypes: an empty slice, and the same codes the
        // plan reports.
        for (name, dt) in [("b", DT_F32), ("w", DT_I16), ("x", DT_F32)] {
            let (p, n) = map(f.ptr(), name, MAP_SAFETENSORS, dt);
            assert!(p.is_null() && n == 0, "{name}");
        }
        let path = f.0.clone();
        assert_eq!(
            map_plan(&path, MAP_SAFETENSORS, "b", DT_F32).err(),
            Some(IO_DTYPE_MISMATCH)
        );
        assert_eq!(
            map_plan(&path, MAP_SAFETENSORS, "x", DT_F32).err(),
            Some(IO_NAME_NOT_FOUND)
        );
        let missing = std::ffi::CString::new("wukong_map_absent_zz.bin").unwrap();
        let (p, n) = map(missing.as_ptr() as *const u8, "", MAP_FILE, DT_F32);
        assert!(p.is_null() && n == 0);
        // An empty file maps to the empty slice too.
        let empty = Scratch::new("map_empty", &[]);
        let (p, n) = map(empty.ptr(), "", MAP_FILE, DT_F32);
        assert!(p.is_null() && n == 0);
        crate::wukong_rt_free(p);
    }

    #[test]
    fn a_payload_past_a_granule_maps_from_the_right_byte() {
        // 70 000 bytes of padding, then 100 i16: the view starts on a 64 KiB boundary before it.
        let mut bytes = vec![0x5Au8; 70_000];
        for i in 0..100i16 {
            bytes.extend_from_slice(&(i * 300 - 7).to_le_bytes());
        }
        let npy_hdr = "{'descr': '<i2', 'fortran_order': False, 'shape': (100,), }";
        // A `.npy` whose array starts at byte 70 000: a v2 header (a u32 length) padded with spaces.
        let pad = 70_000 - 12 - npy_hdr.len() - 1;
        let mut npy = b"\x93NUMPY\x02\x00".to_vec();
        npy.extend_from_slice(&((npy_hdr.len() + pad + 1) as u32).to_le_bytes());
        npy.extend_from_slice(npy_hdr.as_bytes());
        npy.extend(std::iter::repeat_n(b' ', pad));
        npy.push(b'\n');
        assert_eq!(npy.len(), 70_000);
        npy.extend_from_slice(&bytes[70_000..]);
        let f = Scratch::new("map_npy", &npy);
        let (p, n) = map(f.ptr(), "", MAP_NPY, DT_I16);
        if cfg!(any(all(unix, target_pointer_width = "64"), windows)) {
            assert!(
                is_mapped(p),
                "the view starts at the granule below byte 70 000"
            );
        }
        let want: Vec<i16> = (0..100).map(|i| i * 300 - 7).collect();
        assert_eq!(lanes::<i16>(p, n), want);
        crate::wukong_rt_free(p);
    }
}
