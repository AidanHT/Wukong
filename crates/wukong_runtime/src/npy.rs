//! The NumPy `.npy` reader behind `npy_read_<T>(path, buf)`.
//!
//! An `.npy` file is the magic `\x93NUMPY`, a format version (`1.0`, `2.0` or `3.0`), a header
//! length (`u16` little-endian for version 1, `u32` for 2 and 3), and a header that is a Python
//! dict literal — `{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }` — padded with
//! spaces and a newline. The array's raw bytes follow. The header is parsed here by a small parser
//! for exactly the literal forms `numpy.lib.format` writes (strings, `True`/`False`/`None`,
//! integers, tuples and lists), and the array is then checked against its destination by
//! [`fileio::check_dest`]: dtype, layout, rank and dims (for a shape-typed tensor), element count.
//!
//! Supported `descr`s are the little-endian (`<`), byte-order-free (`|`) and native (`=`, read as
//! little-endian on a little-endian host) spellings of `f2 f4 f8 i1 i2 i4 i8 u1 u2 u4 u8`. A
//! big-endian array, a `bool` array, a structured dtype and anything else parse fine and then fail
//! the dtype check with [`fileio::IO_DTYPE_MISMATCH`]. A `fortran_order: True` array of rank ≥ 2 is
//! [`fileio::IO_SHAPE_MISMATCH`]: reading it row-major would silently transpose it.

use std::io::{Read, Seek, SeekFrom};

use crate::fileio::{
    self, Container, TensorLoc, DT_F16, DT_F32, DT_F64, DT_I16, DT_I32, DT_I64, DT_I8, DT_U16,
    DT_U32, DT_U64, DT_U8, DT_UNSUPPORTED, IO_BAD_HEADER,
};

/// The `.npy` magic string.
const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// The largest header this reader accepts. `numpy.lib.format` writes a few hundred bytes; the cap
/// keeps a corrupt length field from becoming a large allocation.
const MAX_HEADER: u64 = 1 << 20;

/// How deeply header values may nest (a structured `descr` nests; nothing else does).
const MAX_DEPTH: u32 = 32;

/// The dtype code a `descr` string denotes, or [`DT_UNSUPPORTED`].
pub fn npy_dtype(descr: &str) -> i64 {
    let b = descr.as_bytes();
    let (order, kind) = match b.first() {
        Some(&o @ (b'<' | b'>' | b'|' | b'=')) => (o, &descr[1..]),
        _ => return DT_UNSUPPORTED,
    };
    // `=` is native order, so it is little-endian only on a little-endian host. (Clippy's
    // `matches!(order, b'<' | b'|' | b'=')` suggestion would drop that host test.)
    let little = matches!(order, b'<' | b'|') || (order == b'=' && cfg!(target_endian = "little"));
    let dt = match kind {
        "f2" => DT_F16,
        "f4" => DT_F32,
        "f8" => DT_F64,
        "i1" => DT_I8,
        "i2" => DT_I16,
        "i4" => DT_I32,
        "i8" => DT_I64,
        "u1" => DT_U8,
        "u2" => DT_U16,
        "u4" => DT_U32,
        "u8" => DT_U64,
        _ => return DT_UNSUPPORTED,
    };
    // Byte order is irrelevant to a one-byte element; a multi-byte one must be little-endian.
    if little || matches!(dt, DT_I8 | DT_U8) {
        dt
    } else {
        DT_UNSUPPORTED
    }
}

/// A parsed header literal. An integer keeps its value only when it is non-negative and fits `u64`.
#[derive(Debug, PartialEq)]
pub(crate) enum Py {
    Str(String),
    Bool(bool),
    /// `None`: valid in a header, never needed by one.
    NoneLit,
    Int(Option<u64>),
    /// A tuple or a list.
    Seq(Vec<Py>),
    Dict(Vec<(String, Py)>),
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: u32,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while matches!(self.b.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn word(&mut self, w: &[u8]) -> bool {
        if self.b.get(self.i..self.i + w.len()) == Some(w) {
            self.i += w.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Option<Py> {
        self.ws();
        match *self.b.get(self.i)? {
            q @ (b'\'' | b'"') => self.string(q).map(Py::Str),
            b'(' => self.nested(|p| p.seq(b')')),
            b'[' => self.nested(|p| p.seq(b']')),
            b'{' => self.nested(Self::dict),
            b'0'..=b'9' | b'-' => self.int(),
            _ => {
                if self.word(b"True") {
                    Some(Py::Bool(true))
                } else if self.word(b"False") {
                    Some(Py::Bool(false))
                } else if self.word(b"None") {
                    Some(Py::NoneLit)
                } else {
                    None
                }
            }
        }
    }

    fn nested(&mut self, f: impl FnOnce(&mut Self) -> Option<Py>) -> Option<Py> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let v = f(self);
        self.depth -= 1;
        v
    }

    /// A quoted string (single or double quotes). The writer never escapes anything in a header,
    /// so a backslash is rejected rather than half-interpreted.
    fn string(&mut self, quote: u8) -> Option<String> {
        self.i += 1;
        let start = self.i;
        loop {
            match *self.b.get(self.i)? {
                b'\\' => return None,
                c if c == quote => break,
                _ => self.i += 1,
            }
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).ok()?.to_owned();
        self.i += 1;
        Some(s)
    }

    /// A decimal integer, with the `L` suffix a Python 2 writer adds.
    fn int(&mut self) -> Option<Py> {
        let neg = self.b.get(self.i) == Some(&b'-');
        if neg {
            self.i += 1;
        }
        let start = self.i;
        while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        if self.i == start {
            return None;
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).ok()?;
        let v = if neg { None } else { text.parse::<u64>().ok() };
        if self.b.get(self.i) == Some(&b'L') {
            self.i += 1;
        }
        Some(Py::Int(v))
    }

    /// A tuple or list body up to `close`, a trailing comma allowed (`(3,)` is the 1-tuple).
    fn seq(&mut self, close: u8) -> Option<Py> {
        self.i += 1;
        let mut items = Vec::new();
        loop {
            if self.eat(close) {
                return Some(Py::Seq(items));
            }
            items.push(self.value()?);
            if !self.eat(b',') {
                return self.eat(close).then_some(Py::Seq(items));
            }
        }
    }

    fn dict(&mut self) -> Option<Py> {
        self.i += 1;
        let mut members = Vec::new();
        loop {
            if self.eat(b'}') {
                return Some(Py::Dict(members));
            }
            self.ws();
            let q = *self.b.get(self.i)?;
            if q != b'\'' && q != b'"' {
                return None;
            }
            let key = self.string(q)?;
            if !self.eat(b':') {
                return None;
            }
            let v = self.value()?;
            members.push((key, v));
            if !self.eat(b',') {
                return self.eat(b'}').then_some(Py::Dict(members));
            }
        }
    }
}

/// Parse a complete header literal, or `None` if it is malformed.
pub(crate) fn parse_header(bytes: &[u8]) -> Option<Py> {
    let mut p = Parser {
        b: bytes,
        i: 0,
        depth: 0,
    };
    let v = p.value()?;
    p.ws();
    (p.i == bytes.len()).then_some(v)
}

/// Locate the array in the `.npy` file `f` of `file_len` bytes: check the magic and version, parse
/// the header dict, map `descr`, and check the payload against the file.
pub fn npy_locate<R: Read + Seek>(f: &mut R, file_len: u64) -> Result<TensorLoc, i64> {
    if file_len < 10 {
        return Err(IO_BAD_HEADER);
    }
    f.seek(SeekFrom::Start(0)).map_err(|_| fileio::IO_ERROR)?;
    let mut pre = [0u8; 8];
    fileio::read_exact_or_io(f, &mut pre)?;
    if &pre[..6] != MAGIC {
        return Err(IO_BAD_HEADER);
    }
    let (hlen, prefix) = match pre[6] {
        1 => {
            let mut l = [0u8; 2];
            fileio::read_exact_or_io(f, &mut l)?;
            (u16::from_le_bytes(l) as u64, 10u64)
        }
        2 | 3 => {
            if file_len < 12 {
                return Err(IO_BAD_HEADER);
            }
            let mut l = [0u8; 4];
            fileio::read_exact_or_io(f, &mut l)?;
            (u32::from_le_bytes(l) as u64, 12u64)
        }
        _ => return Err(IO_BAD_HEADER),
    };
    if hlen > MAX_HEADER || hlen > file_len - prefix {
        return Err(IO_BAD_HEADER);
    }
    let mut header = vec![0u8; hlen as usize];
    fileio::read_exact_or_io(f, &mut header)?;
    let Some(Py::Dict(members)) = parse_header(&header) else {
        return Err(IO_BAD_HEADER);
    };
    let field = |k: &str| members.iter().find(|(m, _)| m == k).map(|(_, v)| v);
    let dtype = match field("descr") {
        Some(Py::Str(s)) => npy_dtype(s),
        // A structured dtype (a list of fields) is a valid header with no element type here.
        Some(Py::Seq(_)) => DT_UNSUPPORTED,
        _ => return Err(IO_BAD_HEADER),
    };
    let fortran = match field("fortran_order") {
        Some(Py::Bool(b)) => *b,
        _ => return Err(IO_BAD_HEADER),
    };
    let shape = match field("shape") {
        Some(Py::Seq(dims)) => dims
            .iter()
            .map(|d| match d {
                Py::Int(Some(n)) => Some(*n),
                _ => None,
            })
            .collect::<Option<Vec<u64>>>()
            .ok_or(IO_BAD_HEADER)?,
        _ => return Err(IO_BAD_HEADER),
    };
    let data_off = prefix + hlen;
    // The payload is everything after the header; for a known dtype it must hold the whole array
    // (trailing bytes after it are ignored, as `numpy.load` ignores them).
    let nbytes = match fileio::dtype_size(dtype) {
        Some(sz) => fileio::shape_count(&shape)?
            .checked_mul(sz as u64)
            .ok_or(IO_BAD_HEADER)?,
        None => file_len - data_off,
    };
    let loc = TensorLoc {
        dtype,
        shape,
        data_off,
        nbytes,
        fortran,
    };
    fileio::check_payload(&loc, file_len)?;
    Ok(loc)
}

/// `wukong_rt_npy_read(path, data, len, dtype, dims, rank) -> i64` — load the array in the `.npy`
/// file `path` into `data` (`len` elements of `dtype`, or `f32` widened from an `f2` payload).
/// `rank < 0` means the destination is a flat buffer; otherwise `dims[..rank]` is its declared
/// shape and must equal the array's. Returns the element count, or `-1` (open), `-2` (I/O), `-3`
/// (dtype), `-4` (shape, count or Fortran order), `-6` (malformed header).
///
/// # Safety
/// `path` must be null or NUL-terminated; `data` must address `len` writable elements of `dtype`;
/// for `rank > 0`, `dims` must address `rank` readable `i64`s.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_npy_read(
    path: *const u8,
    data: *mut u8,
    len: i64,
    dtype: i64,
    dims: *const i64,
    rank: i64,
) -> i64 {
    // SAFETY: every pointer is forwarded under this function's own contract.
    unsafe {
        fileio::read_container_native(
            crate::rt_path(path),
            &Container::Npy,
            data,
            len,
            dtype,
            fileio::dims_arg(dims, rank),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fileio::{Scratch, IO_DTYPE_MISMATCH, IO_SHAPE_MISMATCH};

    /// A version-1.0 `.npy` file with header dict text `dict` and `payload`, padded the way numpy
    /// pads (the whole preamble a multiple of 64 bytes, ending in a newline).
    fn npy_file(dict: &str, payload: &[u8]) -> Vec<u8> {
        let mut header = dict.as_bytes().to_vec();
        while !(10 + header.len() + 1).is_multiple_of(64) {
            header.push(b' ');
        }
        header.push(b'\n');
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&[1, 0]);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn descr_table() {
        assert_eq!(npy_dtype("<f4"), DT_F32);
        assert_eq!(npy_dtype("<f2"), DT_F16);
        assert_eq!(npy_dtype("|u1"), DT_U8);
        assert_eq!(npy_dtype(">i1"), DT_I8);
        assert_eq!(npy_dtype("<u8"), DT_U64);
        assert_eq!(npy_dtype(">f4"), DT_UNSUPPORTED);
        assert_eq!(npy_dtype("|b1"), DT_UNSUPPORTED);
        assert_eq!(npy_dtype("<c8"), DT_UNSUPPORTED);
        assert_eq!(npy_dtype("f4"), DT_UNSUPPORTED);
        assert_eq!(npy_dtype(""), DT_UNSUPPORTED);
        let native = if cfg!(target_endian = "little") {
            DT_I32
        } else {
            DT_UNSUPPORTED
        };
        assert_eq!(npy_dtype("=i4"), native);
    }

    #[test]
    fn header_parser_reads_what_numpy_writes() {
        let v = parse_header(b"{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }  \n");
        assert_eq!(
            v,
            Some(Py::Dict(vec![
                ("descr".into(), Py::Str("<f4".into())),
                ("fortran_order".into(), Py::Bool(false)),
                (
                    "shape".into(),
                    Py::Seq(vec![Py::Int(Some(3)), Py::Int(Some(4))])
                ),
            ]))
        );
        assert_eq!(
            parse_header(b"{\"shape\": (5L,), \"x\": [None, -2]}"),
            Some(Py::Dict(vec![
                ("shape".into(), Py::Seq(vec![Py::Int(Some(5))])),
                ("x".into(), Py::Seq(vec![Py::NoneLit, Py::Int(None)])),
            ]))
        );
        assert!(parse_header(b"{'shape': ()}").is_some());
        for bad in [
            &b"{'a': 1"[..],
            b"{'a' 1}",
            b"{a: 1}",
            b"{'a': (1 2)}",
            b"{'a': 'x\\'y'}",
            b"{} junk",
            b"{'a': Truth}",
        ] {
            assert_eq!(
                parse_header(bad),
                None,
                "{:?}",
                String::from_utf8_lossy(bad)
            );
        }
    }

    #[test]
    fn locate_checks_magic_version_fields_and_extent() {
        let ok = npy_file(
            "{'descr': '<i2', 'fortran_order': False, 'shape': (2, 3), }",
            &[0u8; 12],
        );
        let mut cur = std::io::Cursor::new(&ok);
        let loc = npy_locate(&mut cur, ok.len() as u64).unwrap();
        assert_eq!(
            (
                loc.dtype,
                loc.shape.clone(),
                loc.data_off,
                loc.nbytes,
                loc.fortran
            ),
            (DT_I16, vec![2, 3], 128, 12, false)
        );
        // One payload byte short.
        let mut cur = std::io::Cursor::new(&ok[..ok.len() - 1]);
        assert_eq!(
            npy_locate(&mut cur, ok.len() as u64 - 1),
            Err(IO_BAD_HEADER)
        );
        let mut bad_magic = ok.clone();
        bad_magic[1] = b'X';
        let mut cur = std::io::Cursor::new(&bad_magic);
        assert_eq!(
            npy_locate(&mut cur, bad_magic.len() as u64),
            Err(IO_BAD_HEADER)
        );
        let mut bad_version = ok.clone();
        bad_version[6] = 9;
        let mut cur = std::io::Cursor::new(&bad_version);
        assert_eq!(
            npy_locate(&mut cur, bad_version.len() as u64),
            Err(IO_BAD_HEADER)
        );
        for dict in [
            "{'descr': '<i2', 'shape': (2, 3), }",
            "{'descr': '<i2', 'fortran_order': 0, 'shape': (2, 3), }",
            "{'descr': '<i2', 'fortran_order': False, 'shape': (2, -3), }",
            "{'descr': 5, 'fortran_order': False, 'shape': (6,), }",
        ] {
            let f = npy_file(dict, &[0u8; 12]);
            let mut cur = std::io::Cursor::new(&f);
            assert_eq!(
                npy_locate(&mut cur, f.len() as u64),
                Err(IO_BAD_HEADER),
                "{dict}"
            );
        }
        // A scalar array (shape `()`) holds one element.
        let s = npy_file(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (), }",
            &[0u8; 8],
        );
        let mut cur = std::io::Cursor::new(&s);
        let loc = npy_locate(&mut cur, s.len() as u64).unwrap();
        assert_eq!((loc.shape.len(), loc.nbytes), (0, 8));
    }

    #[test]
    fn npy_read_loads_checks_and_widens() {
        // f2 [3]: 1.0, -0.5, 65504 (f16::MAX).
        let mut payload = Vec::new();
        for bits in [0x3C00u16, 0xB800, 0x7BFF] {
            payload.extend_from_slice(&bits.to_le_bytes());
        }
        let f = Scratch::new(
            "npy",
            &npy_file(
                "{'descr': '<f2', 'fortran_order': False, 'shape': (3,), }",
                &payload,
            ),
        );
        let dims = [3i64];
        let mut out = [0f32; 3];
        // SAFETY: `out` holds 3 f32 and `dims` describes 3 elements.
        let got = unsafe {
            wukong_rt_npy_read(
                f.ptr(),
                out.as_mut_ptr() as *mut u8,
                3,
                DT_F32,
                dims.as_ptr(),
                1,
            )
        };
        assert_eq!((got, out), (3, [1.0, -0.5, 65504.0]));
        let mut raw = [0u16; 3];
        // SAFETY: `raw` holds 3 u16 for the f16 read path; `dims` describes 3 elements.
        let got = unsafe {
            wukong_rt_npy_read(
                f.ptr(),
                raw.as_mut_ptr() as *mut u8,
                3,
                DT_F16,
                dims.as_ptr(),
                -1,
            )
        };
        assert_eq!((got, raw), (3, [0x3C00, 0xB800, 0x7BFF]));
        let mut wrong = [0f64; 3];
        // SAFETY: `wrong` holds 3 f64 and `dims` describes 3 elements for this read attempt.
        let got = unsafe {
            wukong_rt_npy_read(
                f.ptr(),
                wrong.as_mut_ptr() as *mut u8,
                3,
                DT_F64,
                dims.as_ptr(),
                -1,
            )
        };
        assert_eq!(got, IO_DTYPE_MISMATCH);
        // A Fortran-order matrix would land transposed: refused.
        let fm = Scratch::new(
            "npy_f",
            &npy_file(
                "{'descr': '<i4', 'fortran_order': True, 'shape': (2, 2), }",
                &[0u8; 16],
            ),
        );
        let mut m = [7i32; 4];
        // SAFETY: `m` holds 4 i32 and `dims` describes the read extent.
        let got = unsafe {
            wukong_rt_npy_read(
                fm.ptr(),
                m.as_mut_ptr() as *mut u8,
                4,
                DT_I32,
                dims.as_ptr(),
                -1,
            )
        };
        assert_eq!((got, m), (IO_SHAPE_MISMATCH, [7; 4]));
    }
}
