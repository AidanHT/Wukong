//! Typed, offset-addressed file I/O: the dtype-coded raw entry points behind every `read_<T>`,
//! `read_<T>_at`, `write_<T>` and `file_size` builtin, plus the destination check and the payload
//! transfer the safetensors and `.npy` container readers share.
//!
//! # One table of element types
//!
//! Every entry point here takes the element type as a **dtype code** ([`DT_F32`] … [`DT_U64`])
//! instead of being stamped out once per type. `wukong_sema::io_dtype_code` is the compiler's copy of
//! the table (the compiler does not depend on this crate; a test in `wukong_interp` pins the two
//! together), the interpreter calls the pure helpers below directly, and the native backend calls
//! the `extern "C"` entries — so a new element type is one row, not one more symbol per direction.
//! The six frozen per-type entries (`wukong_rt_read_f32` …) in `lib.rs` are thin wrappers over
//! [`wukong_rt_read_dt`] / [`wukong_rt_write_dt`], so there is one implementation of the contract.
//!
//! # The contract, and why the order is part of it
//!
//! A read opens `path` first (`-1` on failure, even for an empty destination — the interpreter is
//! the semantic oracle and has always behaved that way), then reads the file length (`-2` on
//! failure), then plans the transfer with [`plan_raw_read`]: a negative byte offset is `-2`, and
//! otherwise `n = min(len, (file_len - byte_off) / sizeof(T))` elements move, clamped at zero. The
//! elements are decoded little-endian whatever the host's byte order.
//!
//! A container read ([`open_container`]) opens, sizes and parses the header in that order, then
//! [`check_dest`] compares the tensor against the destination; every step reports one of the
//! negative codes below, and the interpreter runs the very same functions, so a malformed or
//! mismatched file yields the same code on every backend.
//!
//! # No staging copy
//!
//! The original reader allocated a full-size `Vec<u8>`, `read_exact`ed into it and decoded element
//! by element, so loading a 500 MB checkpoint peaked at 1 GB and touched every byte twice. On a
//! little-endian host the file's bytes ARE the destination's bytes, so the native path now
//! `read_exact`s straight into the destination; a big-endian host byte-swaps each element in place
//! afterwards. A `bf16`/`f16` → `f32` widening load stages one bounded chunk at a time. The one
//! observable consequence: after a `-2` (a mid-read I/O error, which a well-formed run never sees)
//! the destination's first `n` elements are unspecified on native, where the interpreter — which
//! must decode into typed slots anyway — leaves them untouched. Writes go the same way: straight
//! from the source bytes on a little-endian host.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// `f32` element (4 bytes).
pub const DT_F32: i64 = 0;
/// `f64` element (8 bytes).
pub const DT_F64: i64 = 1;
/// IEEE `f16` element (2 bytes, the raw half bits).
pub const DT_F16: i64 = 2;
/// `bf16` element (2 bytes, the top half of an `f32`).
pub const DT_BF16: i64 = 3;
/// `i8` element.
pub const DT_I8: i64 = 4;
/// `i16` element.
pub const DT_I16: i64 = 5;
/// `i32` element.
pub const DT_I32: i64 = 6;
/// `i64` element.
pub const DT_I64: i64 = 7;
/// `u8` element.
pub const DT_U8: i64 = 8;
/// `u16` element.
pub const DT_U16: i64 = 9;
/// `u32` element.
pub const DT_U32: i64 = 10;
/// `u64` element.
pub const DT_U64: i64 = 11;
/// A container dtype with no Wukong element type (`BOOL`, `F8_E4M3`, a big-endian `.npy`, …).
/// Never a valid destination dtype, so it always fails the dtype check with [`IO_DTYPE_MISMATCH`].
pub const DT_UNSUPPORTED: i64 = -1;

/// The path could not be opened (or created, for a write).
pub const IO_OPEN_FAILED: i64 = -1;
/// An I/O error after the open: metadata, seek, a short or failed read/write, a negative offset.
pub const IO_ERROR: i64 = -2;
/// The container's tensor dtype is neither the destination's nor a `bf16`/`f16` → `f32` widening.
pub const IO_DTYPE_MISMATCH: i64 = -3;
/// The container's element count, rank or dims disagree with the destination's.
pub const IO_SHAPE_MISMATCH: i64 = -4;
/// The safetensors header has no tensor of that name.
pub const IO_NAME_NOT_FOUND: i64 = -5;
/// The container's header is malformed: bad magic, unparseable header, a payload outside the
/// file, or a payload whose byte length disagrees with its dtype and shape.
pub const IO_BAD_HEADER: i64 = -6;

/// Elements per staged chunk of a widening load (128 KiB of half-precision input).
const WIDEN_CHUNK: usize = 1 << 16;

/// Byte width of an element of dtype `dt`, or `None` for a code outside the table.
pub fn dtype_size(dt: i64) -> Option<usize> {
    Some(match dt {
        DT_I8 | DT_U8 => 1,
        DT_F16 | DT_BF16 | DT_I16 | DT_U16 => 2,
        DT_F32 | DT_I32 | DT_U32 => 4,
        DT_F64 | DT_I64 | DT_U64 => 8,
        _ => return None,
    })
}

/// Plan a raw typed read: the absolute byte offset to seek to and the element count to transfer,
/// for a destination of `len` elements of dtype `dt` reading a file of `file_len` bytes from
/// `byte_off`. A negative offset or an unknown dtype is [`IO_ERROR`]; an offset at or past the end
/// of the file plans zero elements. Shared verbatim by the native entry and the interpreter, so the
/// returned count cannot differ between them.
pub fn plan_raw_read(file_len: u64, byte_off: i64, len: i64, dt: i64) -> Result<(u64, usize), i64> {
    let sz = dtype_size(dt).ok_or(IO_ERROR)? as u64;
    let off = u64::try_from(byte_off).map_err(|_| IO_ERROR)?;
    let avail = file_len.saturating_sub(off) / sz;
    let n = if len <= 0 { 0 } else { (len as u64).min(avail) };
    Ok((off, usize::try_from(n).map_err(|_| IO_ERROR)?))
}

/// Convert little-endian elements of `sz` bytes to the host's order, in place. A no-op on a
/// little-endian host (the only kind this repo's CI and dev box run), a per-element byte reversal on
/// a big-endian one. The same function converts host order back to little-endian, since a byte
/// reversal is its own inverse.
pub fn le_to_host_in_place(bytes: &mut [u8], sz: usize) {
    if cfg!(target_endian = "big") && sz > 1 {
        for elem in bytes.chunks_exact_mut(sz) {
            elem.reverse();
        }
    }
}

/// Widen the raw bits of a `bf16` or `f16` element to the `f32` it denotes — exact for both, so a
/// half-precision checkpoint loads into an `f32` model without rounding. Any other `src_dt` is a
/// caller bug and yields `NaN` rather than a plausible number.
pub fn widen_half(bits: u16, src_dt: i64) -> f32 {
    match src_dt {
        DT_BF16 => crate::bf16_bits_to_f32(bits),
        DT_F16 => crate::f16_bits_to_f32(bits),
        _ => f32::NAN,
    }
}

/// A tensor payload located inside a container file by [`crate::safetensors::st_locate`] or
/// [`crate::npy::npy_locate`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorLoc {
    /// The payload's dtype code, or [`DT_UNSUPPORTED`].
    pub dtype: i64,
    /// The declared shape, outermost dimension first.
    pub shape: Vec<u64>,
    /// Absolute byte offset of the payload in the file.
    pub data_off: u64,
    /// Payload length in bytes.
    pub nbytes: u64,
    /// `.npy` `fortran_order: True` — the payload is column-major.
    pub fortran: bool,
}

/// The element count of `shape` (1 for a rank-0 scalar), or [`IO_BAD_HEADER`] on overflow.
pub fn shape_count(shape: &[u64]) -> Result<u64, i64> {
    shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or(IO_BAD_HEADER)
}

/// Check a located payload against the file it sits in: it must lie inside the file, and for a
/// dtype this table knows its byte length must be exactly `count(shape) × sizeof(dtype)`. Shared by
/// both container parsers.
pub fn check_payload(loc: &TensorLoc, file_len: u64) -> Result<(), i64> {
    let end = loc.data_off.checked_add(loc.nbytes).ok_or(IO_BAD_HEADER)?;
    if end > file_len {
        return Err(IO_BAD_HEADER);
    }
    if let Some(sz) = dtype_size(loc.dtype) {
        let want = shape_count(&loc.shape)?
            .checked_mul(sz as u64)
            .ok_or(IO_BAD_HEADER)?;
        if want != loc.nbytes {
            return Err(IO_BAD_HEADER);
        }
    }
    Ok(())
}

/// Validate a located tensor against its destination, returning the element count to transfer and
/// whether each element must be widened from `bf16`/`f16` to `f32`.
///
/// The checks run in a fixed order so both backends report the same code for a file that is wrong
/// in several ways at once: the dtype ([`IO_DTYPE_MISMATCH`] unless equal, or a half → `f32`
/// widening), then a column-major payload of rank ≥ 2 ([`IO_SHAPE_MISMATCH`]: a row-major destination
/// would silently receive the transpose), then — when the destination is a tensor view — the rank
/// and every dimension against `dims`, then the element count against `dest_len`. The count check is
/// an equality, not a `min`: a container read is the load-time half of Wukong's shape safety, so a
/// buffer one element too long is as wrong as one too short.
pub fn check_dest(
    loc: &TensorLoc,
    dest_dt: i64,
    dest_len: i64,
    dims: Option<&[i64]>,
) -> Result<(usize, bool), i64> {
    let widen = if loc.dtype == dest_dt && dtype_size(dest_dt).is_some() {
        false
    } else if dest_dt == DT_F32 && matches!(loc.dtype, DT_BF16 | DT_F16) {
        true
    } else {
        return Err(IO_DTYPE_MISMATCH);
    };
    let count = shape_count(&loc.shape)?;
    if loc.fortran && loc.shape.len() >= 2 {
        return Err(IO_SHAPE_MISMATCH);
    }
    if let Some(dims) = dims {
        let same = dims.len() == loc.shape.len()
            && dims
                .iter()
                .zip(&loc.shape)
                .all(|(&d, &s)| u64::try_from(d).ok() == Some(s));
        if !same {
            return Err(IO_SHAPE_MISMATCH);
        }
    }
    if u64::try_from(dest_len).ok() != Some(count) {
        return Err(IO_SHAPE_MISMATCH);
    }
    let count = usize::try_from(count).map_err(|_| IO_SHAPE_MISMATCH)?;
    Ok((count, widen))
}

/// Which container a typed load parses, with the tensor name a safetensors file is indexed by.
pub enum Container<'a> {
    /// A safetensors file and the name of the tensor to load.
    Safetensors(&'a str),
    /// A NumPy `.npy` file (one array).
    Npy,
}

/// Open `path`, size it and parse the container header, returning the open file and the located
/// tensor. The order — open (`-1`), metadata (`-2`), header — is the contract both backends share;
/// the interpreter calls this very function with the path it reconstructed from its slot memory.
pub fn open_container(path: &Path, container: &Container<'_>) -> Result<(File, TensorLoc), i64> {
    let mut file = File::open(path).map_err(|_| IO_OPEN_FAILED)?;
    let file_len = file.metadata().map_err(|_| IO_ERROR)?.len();
    let loc = match container {
        Container::Safetensors(name) => crate::safetensors::st_locate(&mut file, file_len, name)?,
        Container::Npy => crate::npy::npy_locate(&mut file, file_len)?,
    };
    Ok((file, loc))
}

/// Read a whole stream position-by-position into `buf`, mapping any failure (including a short
/// file) to [`IO_ERROR`].
pub fn read_exact_or_io(file: &mut impl Read, buf: &mut [u8]) -> Result<(), i64> {
    file.read_exact(buf).map_err(|_| IO_ERROR)
}

/// Transfer a checked container payload into native memory: `count` elements of `loc.dtype` at
/// `loc.data_off`, either verbatim into `data` (the destination has the payload's dtype) or widened
/// to `f32` one bounded chunk at a time. Returns the element count, or [`IO_ERROR`].
///
/// # Safety
/// `data` must address `count` writable destination elements — of `loc.dtype` when `widen` is
/// false, of `f32` when it is true.
unsafe fn read_payload_native(
    file: &mut File,
    loc: &TensorLoc,
    data: *mut u8,
    count: usize,
    widen: bool,
) -> i64 {
    if count == 0 {
        return 0;
    }
    if file.seek(SeekFrom::Start(loc.data_off)).is_err() {
        return IO_ERROR;
    }
    if !widen {
        let Some(sz) = dtype_size(loc.dtype) else {
            return IO_ERROR;
        };
        // SAFETY: the caller guarantees `count` writable elements of `sz` bytes; `check_payload`
        // proved `count * sz` equals the payload length, which fits in the file and hence in u64.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, count * sz) };
        if file.read_exact(bytes).is_err() {
            return IO_ERROR;
        }
        le_to_host_in_place(bytes, sz);
        return count as i64;
    }
    // SAFETY: `widen` means the destination holds `count` f32 elements (4 bytes each).
    let out = unsafe { std::slice::from_raw_parts_mut(data, count * 4) };
    let mut stage = vec![0u8; count.min(WIDEN_CHUNK) * 2];
    let mut done = 0usize;
    while done < count {
        let m = (count - done).min(WIDEN_CHUNK);
        let chunk = &mut stage[..m * 2];
        if file.read_exact(chunk).is_err() {
            return IO_ERROR;
        }
        for (j, half) in chunk.chunks_exact(2).enumerate() {
            let x = widen_half(u16::from_le_bytes([half[0], half[1]]), loc.dtype);
            let at = (done + j) * 4;
            out[at..at + 4].copy_from_slice(&x.to_ne_bytes());
        }
        done += m;
    }
    count as i64
}

/// The shared native body of the container entries: open and locate, check against the
/// destination, transfer.
///
/// # Safety
/// As [`read_payload_native`], for `len` elements of `dest_dt` at `data`.
pub(crate) unsafe fn read_container_native(
    path: Option<std::path::PathBuf>,
    container: &Container<'_>,
    data: *mut u8,
    len: i64,
    dest_dt: i64,
    dims: Option<&[i64]>,
) -> i64 {
    let Some(path) = path else {
        return IO_OPEN_FAILED;
    };
    let (mut file, loc) = match open_container(&path, container) {
        Ok(found) => found,
        Err(code) => return code,
    };
    let (count, widen) = match check_dest(&loc, dest_dt, len, dims) {
        Ok(plan) => plan,
        Err(code) => return code,
    };
    // SAFETY: `count == len` (checked), and the caller guarantees `len` destination elements.
    unsafe { read_payload_native(&mut file, &loc, data, count, widen) }
}

/// The destination's declared dims, as the container entries receive them: `rank < 0` means the
/// destination is a flat buffer (a slice or an array) with no shape to check.
///
/// # Safety
/// For `rank > 0`, `dims` must address `rank` readable `i64`s.
pub(crate) unsafe fn dims_arg<'a>(dims: *const i64, rank: i64) -> Option<&'a [i64]> {
    match rank {
        r if r < 0 => None,
        0 => Some(&[]),
        // SAFETY: forwarded from this function's contract.
        r => Some(unsafe { std::slice::from_raw_parts(dims, r as usize) }),
    }
}

/// A NUL-terminated name argument, decoded lossily (an invalid UTF-8 sequence becomes U+FFFD —
/// exactly how the interpreter reconstructs the same bytes from its slots). Null is the empty name.
///
/// # Safety
/// `p` must be null or a NUL-terminated byte string.
pub(crate) unsafe fn cstr_lossy(p: *const u8) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: forwarded from this function's contract.
    let cstr = unsafe { std::ffi::CStr::from_ptr(p as *const std::ffi::c_char) };
    cstr.to_string_lossy().into_owned()
}

/// Write a host-order element buffer as little-endian bytes: straight from `bytes` on a
/// little-endian host, through a bounded byte-swapped chunk otherwise.
fn write_le(file: &mut File, bytes: &[u8], sz: usize) -> std::io::Result<()> {
    if cfg!(target_endian = "little") || sz == 1 {
        return file.write_all(bytes);
    }
    let mut chunk = Vec::with_capacity(bytes.len().min(sz * 4096));
    for block in bytes.chunks(sz * 4096) {
        chunk.clear();
        chunk.extend_from_slice(block);
        le_to_host_in_place(&mut chunk, sz);
        file.write_all(&chunk)?;
    }
    Ok(())
}

/// `wukong_rt_read_dt(path, byte_off, data, len, dtype) -> i64` — read up to `len` little-endian
/// elements of `dtype` from `path`, starting `byte_off` bytes into the file, into `data`. Backs every
/// `read_<T>` (with `byte_off = 0`) and `read_<T>_at` builtin. Returns the element count read, `-1`
/// if `path` cannot be opened (checked first, even for `len <= 0`), or `-2` on an I/O error or a
/// negative `byte_off`. `data[n..]` is left as the caller allocated it.
///
/// # Safety
/// `data` must address at least `len` writable elements of `dtype`, and `path` must be null or a
/// NUL-terminated byte string.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_read_dt(
    path: *const u8,
    byte_off: i64,
    data: *mut u8,
    len: i64,
    dtype: i64,
) -> i64 {
    // SAFETY: forwarded from this function's contract.
    let Some(path) = (unsafe { crate::rt_path(path) }) else {
        return IO_OPEN_FAILED;
    };
    let Ok(mut file) = File::open(&path) else {
        return IO_OPEN_FAILED;
    };
    let file_len = match file.metadata() {
        Ok(m) => m.len(),
        Err(_) => return IO_ERROR,
    };
    let (off, n) = match plan_raw_read(file_len, byte_off, len, dtype) {
        Ok(plan) => plan,
        Err(code) => return code,
    };
    if n == 0 {
        return 0;
    }
    let Some(sz) = dtype_size(dtype) else {
        return IO_ERROR;
    };
    if file.seek(SeekFrom::Start(off)).is_err() {
        return IO_ERROR;
    }
    // SAFETY: `n <= len` elements of `sz` bytes; `n * sz <= file_len`, so it cannot overflow.
    let bytes = unsafe { std::slice::from_raw_parts_mut(data, n * sz) };
    if file.read_exact(bytes).is_err() {
        return IO_ERROR;
    }
    le_to_host_in_place(bytes, sz);
    n as i64
}

/// `wukong_rt_write_dt(path, data, len, dtype) -> i64` — create or truncate `path` and write all
/// `len` elements of `dtype` from `data` as little-endian bytes. Returns `len` (`0` for `len <= 0`,
/// which still truncates), `-1` if the file cannot be created, `-2` on a write error.
///
/// # Safety
/// `data` must address at least `len` readable elements of `dtype`, and `path` must be null or a
/// NUL-terminated byte string.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_write_dt(
    path: *const u8,
    data: *const u8,
    len: i64,
    dtype: i64,
) -> i64 {
    // SAFETY: forwarded from this function's contract.
    let Some(path) = (unsafe { crate::rt_path(path) }) else {
        return IO_OPEN_FAILED;
    };
    let Some(sz) = dtype_size(dtype) else {
        return IO_ERROR;
    };
    let Ok(mut file) = File::create(&path) else {
        return IO_OPEN_FAILED;
    };
    let count = len.max(0) as usize;
    if count == 0 {
        return 0;
    }
    let Some(nbytes) = count.checked_mul(sz) else {
        return IO_ERROR;
    };
    // SAFETY: the caller guarantees `len` readable elements of `sz` bytes at `data`.
    let bytes = unsafe { std::slice::from_raw_parts(data, nbytes) };
    if write_le(&mut file, bytes, sz).is_err() {
        return IO_ERROR;
    }
    count as i64
}

/// `wukong_rt_file_size(path) -> i64` — the byte length of `path`, `-1` if it cannot be opened,
/// `-2` if its metadata cannot be read. Backs the `file_size(path)` builtin, so a program can size
/// an `alloc_<T>` buffer from the file it is about to read.
///
/// # Safety
/// `path` must be null or a NUL-terminated byte string.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_file_size(path: *const u8) -> i64 {
    // SAFETY: forwarded from this function's contract.
    let Some(path) = (unsafe { crate::rt_path(path) }) else {
        return IO_OPEN_FAILED;
    };
    let Ok(file) = File::open(&path) else {
        return IO_OPEN_FAILED;
    };
    match file.metadata() {
        Ok(m) => i64::try_from(m.len()).unwrap_or(IO_ERROR),
        Err(_) => IO_ERROR,
    }
}

/// A uniquely named scratch file, removed on drop — shared by this crate's I/O unit tests.
#[cfg(test)]
pub(crate) struct Scratch(pub(crate) std::path::PathBuf, std::ffi::CString);

#[cfg(test)]
impl Scratch {
    pub(crate) fn new(tag: &str, bytes: &[u8]) -> Scratch {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("wukong_io_{tag}_{}_{seq}.bin", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let c = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        Scratch(path, c)
    }

    pub(crate) fn ptr(&self) -> *const u8 {
        self.1.as_ptr() as *const u8
    }
}

#[cfg(test)]
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_table_covers_every_code_once() {
        let sizes: Vec<_> = (DT_F32..=DT_U64).map(dtype_size).collect();
        assert_eq!(
            sizes,
            [4, 8, 2, 2, 1, 2, 4, 8, 1, 2, 4, 8].map(Some).to_vec()
        );
        assert_eq!(dtype_size(DT_UNSUPPORTED), None);
        assert_eq!(dtype_size(12), None);
    }

    #[test]
    fn plan_clamps_to_the_file_and_rejects_negative_offsets() {
        // 9 bytes of u16 from offset 2: four whole elements, the trailing byte ignored.
        assert_eq!(plan_raw_read(11, 2, 100, DT_U16), Ok((2, 4)));
        assert_eq!(plan_raw_read(11, 2, 3, DT_U16), Ok((2, 3)));
        assert_eq!(plan_raw_read(11, 11, 3, DT_U16), Ok((11, 0)));
        assert_eq!(plan_raw_read(11, 400, 3, DT_U16), Ok((400, 0)));
        assert_eq!(plan_raw_read(11, 0, 0, DT_U16), Ok((0, 0)));
        assert_eq!(plan_raw_read(11, 0, -5, DT_U16), Ok((0, 0)));
        assert_eq!(plan_raw_read(11, -1, 3, DT_U16), Err(IO_ERROR));
        assert_eq!(plan_raw_read(11, 0, 3, 99), Err(IO_ERROR));
    }

    #[test]
    fn read_dt_reads_in_place_from_an_offset() {
        let mut bytes = vec![0xEEu8; 3];
        for v in [1u16, 0x8000, 0xFFFF, 0x1234] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let f = Scratch::new("read_at", &bytes);
        let mut out = [7u16; 5];
        // SAFETY: `out` holds 5 u16; the read fetches 5 elements from offset 3 of the scratch file.
        let n = unsafe { wukong_rt_read_dt(f.ptr(), 3, out.as_mut_ptr() as *mut u8, 5, DT_U16) };
        assert_eq!(n, 4);
        assert_eq!(out, [1, 0x8000, 0xFFFF, 0x1234, 7]);
        // The frozen whole-file entry reads the same bytes from offset 0.
        let mut b = [0u8; 4];
        // SAFETY: `b` holds 4 u8; the whole-file entry reads 4 bytes from offset 0.
        let n = unsafe { wukong_rt_read_dt(f.ptr(), 0, b.as_mut_ptr(), 4, DT_U8) };
        assert_eq!((n, b), (4, [0xEE, 0xEE, 0xEE, 1]));
        // SAFETY: `f.ptr()` is a valid NUL-terminated path for the scratch file.
        assert_eq!(unsafe { wukong_rt_file_size(f.ptr()) }, 11);
    }

    #[test]
    fn read_dt_open_failure_wins_and_leaves_data_alone() {
        let missing = std::ffi::CString::new("wukong_io_definitely_absent_zz.bin").unwrap();
        let mut out = [9i32; 2];
        let p = missing.as_ptr() as *const u8;
        assert_eq!(
            // SAFETY: a missing-file path with n == 0 is handled without a read; `out` is untouched.
            unsafe { wukong_rt_read_dt(p, 0, out.as_mut_ptr() as *mut u8, 0, DT_I32) },
            IO_OPEN_FAILED
        );
        assert_eq!(
            // SAFETY: a negative offset is rejected without a read; `out` is untouched.
            unsafe { wukong_rt_read_dt(p, -4, out.as_mut_ptr() as *mut u8, 2, DT_I32) },
            IO_OPEN_FAILED
        );
        assert_eq!(out, [9, 9]);
        // SAFETY: a missing-file path returns IO_OPEN_FAILED from file_size without a read.
        assert_eq!(unsafe { wukong_rt_file_size(p) }, IO_OPEN_FAILED);
        assert_eq!(
            // SAFETY: a missing-file path is rejected without touching the output buffer.
            unsafe {
                wukong_rt_read_dt(std::ptr::null(), 0, out.as_mut_ptr() as *mut u8, 2, DT_I32)
            },
            IO_OPEN_FAILED
        );
    }

    #[test]
    fn write_dt_round_trips_little_endian() {
        let f = Scratch::new("write", &[]);
        let src: [u64; 3] = [0x0102_0304_0506_0708, u64::MAX, 0];
        // SAFETY: `src` holds 3 u64; the write emits exactly 3 elements.
        let n = unsafe { wukong_rt_write_dt(f.ptr(), src.as_ptr() as *const u8, 3, DT_U64) };
        assert_eq!(n, 3);
        let on_disk = std::fs::read(&f.0).unwrap();
        let want: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(on_disk, want);
        // A zero-length write truncates and returns 0 without touching the (null) data pointer.
        // SAFETY: a zero-length write truncates and returns 0 without touching the null data pointer.
        let n = unsafe { wukong_rt_write_dt(f.ptr(), std::ptr::null(), 0, DT_F16) };
        assert_eq!(n, 0);
        assert_eq!(std::fs::read(&f.0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn half_widening_is_exact() {
        assert_eq!(widen_half(0x3F80, DT_BF16), 1.0);
        assert_eq!(widen_half(0xC040, DT_BF16), -3.0);
        assert_eq!(widen_half(0x3C00, DT_F16), 1.0);
        assert_eq!(widen_half(0x8000, DT_F16).to_bits(), (-0.0f32).to_bits());
        assert_eq!(widen_half(0x0001, DT_F16), 2f32.powi(-24));
        assert!(widen_half(0x3C00, DT_I16).is_nan());
    }

    #[test]
    fn payload_must_fit_the_file_and_match_its_dtype_and_shape() {
        let loc = TensorLoc {
            dtype: DT_F32,
            shape: vec![2, 3],
            data_off: 8,
            nbytes: 24,
            fortran: false,
        };
        assert_eq!(check_payload(&loc, 32), Ok(()));
        assert_eq!(check_payload(&loc, 31), Err(IO_BAD_HEADER));
        let short = TensorLoc {
            nbytes: 20,
            ..loc.clone()
        };
        assert_eq!(check_payload(&short, 64), Err(IO_BAD_HEADER));
        // An unknown dtype has no element size to check; only the extent is validated.
        let odd = TensorLoc {
            dtype: DT_UNSUPPORTED,
            nbytes: 5,
            ..loc
        };
        assert_eq!(check_payload(&odd, 13), Ok(()));
        assert_eq!(shape_count(&[]), Ok(1));
        assert_eq!(shape_count(&[u64::MAX, 2]), Err(IO_BAD_HEADER));
    }

    #[test]
    fn check_dest_orders_dtype_then_layout_then_dims_then_count() {
        let loc = TensorLoc {
            dtype: DT_BF16,
            shape: vec![2, 3],
            data_off: 0,
            nbytes: 12,
            fortran: false,
        };
        assert_eq!(check_dest(&loc, DT_BF16, 6, None), Ok((6, false)));
        assert_eq!(check_dest(&loc, DT_F32, 6, None), Ok((6, true)));
        assert_eq!(check_dest(&loc, DT_F64, 6, None), Err(IO_DTYPE_MISMATCH));
        // A dtype mismatch is reported even when the count is also wrong.
        assert_eq!(check_dest(&loc, DT_I16, 5, None), Err(IO_DTYPE_MISMATCH));
        assert_eq!(check_dest(&loc, DT_BF16, 5, None), Err(IO_SHAPE_MISMATCH));
        assert_eq!(check_dest(&loc, DT_BF16, 7, None), Err(IO_SHAPE_MISMATCH));
        assert_eq!(check_dest(&loc, DT_BF16, 6, Some(&[2, 3])), Ok((6, false)));
        assert_eq!(
            check_dest(&loc, DT_BF16, 6, Some(&[3, 2])),
            Err(IO_SHAPE_MISMATCH)
        );
        assert_eq!(
            check_dest(&loc, DT_BF16, 6, Some(&[6])),
            Err(IO_SHAPE_MISMATCH)
        );
        let fortran = TensorLoc {
            fortran: true,
            ..loc.clone()
        };
        assert_eq!(
            check_dest(&fortran, DT_BF16, 6, None),
            Err(IO_SHAPE_MISMATCH)
        );
        let unsupported = TensorLoc {
            dtype: DT_UNSUPPORTED,
            ..loc
        };
        assert_eq!(
            check_dest(&unsupported, DT_F32, 6, None),
            Err(IO_DTYPE_MISMATCH)
        );
    }
}
