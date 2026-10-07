//! The safetensors reader behind `st_read_<T>(path, name, buf)`.
//!
//! A safetensors file is an 8-byte little-endian header length `N`, `N` bytes of UTF-8 JSON (padded
//! with trailing spaces), then one byte buffer. The JSON maps each tensor name to
//! `{"dtype": "BF16", "shape": [768, 3072], "data_offsets": [begin, end]}` — offsets relative to the
//! start of the byte buffer — plus an optional string-to-string `"__metadata__"` entry. The header
//! is parsed here with a small, strict, depth-limited JSON parser (no new dependency), and the named
//! tensor's dtype, shape and extent are handed to [`fileio::check_dest`], which compares them with
//! the destination — the element count always, and the rank and every dimension when the
//! destination is a shape-typed tensor. That is the load-time half of Wukong's shape safety: a
//! checkpoint whose `wte.weight` is `[50257, 768]` cannot be loaded into a `Tensor[f32, 768, 50257]`.
//!
//! The parse is deliberately strict where leniency would make two readers disagree: a header that
//! is not valid UTF-8 or not valid JSON, an entry missing a field, an offset pair out of order or
//! outside the file, a byte length that disagrees with dtype × shape, and a name that occurs twice
//! are all [`fileio::IO_BAD_HEADER`]; the other tensors are not validated, so one tensor can be read
//! out of a shard without walking the rest.

use std::io::{Read, Seek, SeekFrom};

use crate::fileio::{
    self, Container, TensorLoc, DT_BF16, DT_F16, DT_F32, DT_F64, DT_I16, DT_I32, DT_I64, DT_I8,
    DT_U16, DT_U32, DT_U64, DT_U8, DT_UNSUPPORTED, IO_BAD_HEADER, IO_NAME_NOT_FOUND,
};

/// The largest header this reader accepts — the limit the reference implementation enforces, and
/// what keeps a corrupt length prefix from becoming a multi-gigabyte allocation.
const MAX_HEADER: u64 = 100_000_000;

/// How deeply JSON values may nest. A safetensors header nests three levels; the limit only exists
/// so a hostile header cannot exhaust the stack.
const MAX_DEPTH: u32 = 64;

/// The dtype code a safetensors dtype string denotes ([`DT_UNSUPPORTED`] for `BOOL`, the fp8
/// formats and anything else Wukong has no element type for).
pub fn st_dtype(s: &str) -> i64 {
    match s {
        "F32" => DT_F32,
        "F64" => DT_F64,
        "F16" => DT_F16,
        "BF16" => DT_BF16,
        "I8" => DT_I8,
        "I16" => DT_I16,
        "I32" => DT_I32,
        "I64" => DT_I64,
        "U8" => DT_U8,
        "U16" => DT_U16,
        "U32" => DT_U32,
        "U64" => DT_U64,
        _ => DT_UNSUPPORTED,
    }
}

/// A parsed JSON value. Numbers keep only what a header needs: the value of a non-negative integer
/// that fits `u64`, and "some other number" for anything else.
#[derive(Debug, PartialEq)]
pub(crate) enum Json {
    /// `true`, `false` or `null`: valid in a header, but never needed by one.
    Literal,
    Num(Option<u64>),
    Str(String),
    Arr(Vec<Json>),
    /// Members in source order (duplicates preserved, so the caller can reject them).
    Obj(Vec<(String, Json)>),
}

/// A strict recursive-descent JSON parser over the header bytes.
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

    fn eat(&mut self, c: u8) -> Option<()> {
        self.ws();
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            Some(())
        } else {
            None
        }
    }

    fn value(&mut self) -> Option<Json> {
        self.ws();
        match *self.b.get(self.i)? {
            b'{' => self.nested(Self::object),
            b'[' => self.nested(Self::array),
            b'"' => self.string().map(Json::Str),
            b't' => self.literal(b"true"),
            b'f' => self.literal(b"false"),
            b'n' => self.literal(b"null"),
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    fn nested(&mut self, f: fn(&mut Self) -> Option<Json>) -> Option<Json> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let v = f(self);
        self.depth -= 1;
        v
    }

    fn literal(&mut self, word: &[u8]) -> Option<Json> {
        if self.b.get(self.i..self.i + word.len())? == word {
            self.i += word.len();
            Some(Json::Literal)
        } else {
            None
        }
    }

    fn object(&mut self) -> Option<Json> {
        self.eat(b'{')?;
        let mut members = Vec::new();
        if self.eat(b'}').is_some() {
            return Some(Json::Obj(members));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.eat(b':')?;
            let v = self.value()?;
            members.push((key, v));
            if self.eat(b',').is_some() {
                continue;
            }
            self.eat(b'}')?;
            return Some(Json::Obj(members));
        }
    }

    fn array(&mut self) -> Option<Json> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        if self.eat(b']').is_some() {
            return Some(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            if self.eat(b',').is_some() {
                continue;
            }
            self.eat(b']')?;
            return Some(Json::Arr(items));
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let digits = self.b.get(self.i..self.i + 4)?;
        let mut v = 0u32;
        for &d in digits {
            v = v * 16 + (d as char).to_digit(16)?;
        }
        self.i += 4;
        Some(v)
    }

    /// A string at the cursor (which must be on its opening quote). Escapes are decoded, a raw
    /// control character is rejected, and a `\u` surrogate must be properly paired.
    fn string(&mut self) -> Option<String> {
        if self.b.get(self.i) != Some(&b'"') {
            return None;
        }
        self.i += 1;
        let mut out = Vec::new();
        loop {
            let c = *self.b.get(self.i)?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).ok(),
                b'\\' => {
                    let e = *self.b.get(self.i)?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                if self.b.get(self.i..self.i + 2)? != b"\\u" {
                                    return None;
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return None;
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            char::from_u32(code)?
                        }
                        _ => return None,
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                0x00..=0x1F => return None,
                _ => out.push(c),
            }
        }
    }

    /// A JSON number. Only a plain non-negative integer that fits `u64` keeps its value.
    fn number(&mut self) -> Option<Json> {
        let start = self.i;
        let neg = self.b.get(self.i) == Some(&b'-');
        if neg {
            self.i += 1;
        }
        let int_start = self.i;
        match self.b.get(self.i)? {
            b'0' => self.i += 1,
            b'1'..=b'9' => {
                while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
                    self.i += 1;
                }
            }
            _ => return None,
        }
        let int_end = self.i;
        let mut plain = !neg;
        if self.b.get(self.i) == Some(&b'.') {
            plain = false;
            self.i += 1;
            self.digits()?;
        }
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            plain = false;
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            self.digits()?;
        }
        debug_assert!(self.i > start);
        if !plain {
            return Some(Json::Num(None));
        }
        let text = std::str::from_utf8(&self.b[int_start..int_end]).ok()?;
        Some(Json::Num(text.parse::<u64>().ok()))
    }

    /// One or more decimal digits.
    fn digits(&mut self) -> Option<()> {
        let s = self.i;
        while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        (self.i > s).then_some(())
    }
}

/// Parse a complete JSON document (surrounding whitespace allowed), or `None` if it is malformed.
pub(crate) fn parse_json(bytes: &[u8]) -> Option<Json> {
    let mut p = Parser {
        b: bytes,
        i: 0,
        depth: 0,
    };
    let v = p.value()?;
    p.ws();
    (p.i == bytes.len()).then_some(v)
}

/// The value of member `key` in an object's member list (the first, if a malformed header repeats
/// it — the tensor entries' own fields are not checked for duplicates).
fn member<'a>(members: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    members.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// A JSON array of plain non-negative integers.
fn u64_list(v: &Json) -> Option<Vec<u64>> {
    match v {
        Json::Arr(items) => items
            .iter()
            .map(|x| match x {
                Json::Num(Some(n)) => Some(*n),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

/// Locate tensor `name` in the safetensors file `f` of `file_len` bytes: parse the header, find the
/// entry, map its dtype, and check its extent against the file and its byte length against
/// dtype × shape.
pub fn st_locate<R: Read + Seek>(f: &mut R, file_len: u64, name: &str) -> Result<TensorLoc, i64> {
    if file_len < 8 {
        return Err(IO_BAD_HEADER);
    }
    f.seek(SeekFrom::Start(0)).map_err(|_| fileio::IO_ERROR)?;
    let mut len_le = [0u8; 8];
    fileio::read_exact_or_io(f, &mut len_le)?;
    let hlen = u64::from_le_bytes(len_le);
    if hlen > MAX_HEADER || hlen > file_len - 8 {
        return Err(IO_BAD_HEADER);
    }
    let mut header = Vec::new();
    header
        .try_reserve_exact(hlen as usize)
        .map_err(|_| IO_BAD_HEADER)?;
    header.resize(hlen as usize, 0u8);
    fileio::read_exact_or_io(f, &mut header)?;
    let Some(Json::Obj(members)) = parse_json(&header) else {
        return Err(IO_BAD_HEADER);
    };
    if name == "__metadata__" {
        return Err(IO_NAME_NOT_FOUND);
    }
    let mut hits = members.iter().filter(|(k, _)| k == name);
    let Some((_, entry)) = hits.next() else {
        return Err(IO_NAME_NOT_FOUND);
    };
    if hits.next().is_some() {
        return Err(IO_BAD_HEADER);
    }
    let Json::Obj(fields) = entry else {
        return Err(IO_BAD_HEADER);
    };
    let dtype = match member(fields, "dtype") {
        Some(Json::Str(s)) => st_dtype(s),
        _ => return Err(IO_BAD_HEADER),
    };
    let shape = member(fields, "shape")
        .and_then(u64_list)
        .ok_or(IO_BAD_HEADER)?;
    let offsets = member(fields, "data_offsets")
        .and_then(u64_list)
        .ok_or(IO_BAD_HEADER)?;
    let (begin, end) = match offsets.as_slice() {
        [b, e] => (*b, *e),
        _ => return Err(IO_BAD_HEADER),
    };
    if begin > end {
        return Err(IO_BAD_HEADER);
    }
    let data_off = (8 + hlen).checked_add(begin).ok_or(IO_BAD_HEADER)?;
    let loc = TensorLoc {
        dtype,
        shape,
        data_off,
        nbytes: end - begin,
        fortran: false,
    };
    fileio::check_payload(&loc, file_len)?;
    Ok(loc)
}

/// `wukong_rt_st_read(path, name, data, len, dtype, dims, rank) -> i64` — load safetensors tensor
/// `name` from `path` into `data` (`len` elements of `dtype`, or `f32` widened from a `BF16`/`F16`
/// payload). `rank < 0` means the destination is a flat buffer; otherwise `dims[..rank]` is its
/// declared shape and must equal the tensor's. Returns the element count, or `-1` (open), `-2`
/// (I/O), `-3` (dtype), `-4` (shape or count), `-5` (no such name), `-6` (malformed header).
///
/// # Safety
/// `path` and `name` must be null or NUL-terminated; `data` must address `len` writable elements of
/// `dtype`; for `rank > 0`, `dims` must address `rank` readable `i64`s.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_st_read(
    path: *const u8,
    name: *const u8,
    data: *mut u8,
    len: i64,
    dtype: i64,
    dims: *const i64,
    rank: i64,
) -> i64 {
    // SAFETY: every pointer is forwarded under this function's own contract.
    unsafe {
        let name = fileio::cstr_lossy(name);
        fileio::read_container_native(
            crate::rt_path(path),
            &Container::Safetensors(&name),
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
    use crate::fileio::{Scratch, IO_DTYPE_MISMATCH, IO_OPEN_FAILED, IO_SHAPE_MISMATCH};

    /// A safetensors file holding `header` (JSON text) and `payload`.
    fn st_file(header: &str, payload: &[u8]) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn json_parser_accepts_the_grammar_and_rejects_the_rest() {
        let v = parse_json(br#" {"a": [1, 0, 18446744073709551615], "b": {"c": "x\"\u00e9\ud83d\ude00"}, "d": [true, false, null, -1, 2.5, 1e3]} "#);
        let Some(Json::Obj(m)) = v else {
            panic!("expected an object, got {v:?}")
        };
        assert_eq!(
            m[0].1,
            Json::Arr(vec![
                Json::Num(Some(1)),
                Json::Num(Some(0)),
                Json::Num(Some(u64::MAX))
            ])
        );
        assert_eq!(
            m[1].1,
            Json::Obj(vec![("c".into(), Json::Str("x\"\u{e9}\u{1F600}".into()))])
        );
        assert_eq!(
            m[2].1,
            Json::Arr(vec![
                Json::Literal,
                Json::Literal,
                Json::Literal,
                Json::Num(None),
                Json::Num(None),
                Json::Num(None)
            ])
        );
        // u64::MAX + 1 keeps "a number" but loses its value.
        assert_eq!(parse_json(b"18446744073709551616"), Some(Json::Num(None)));
        for bad in [
            &b"{"[..],
            b"{\"a\" 1}",
            b"[1,]",
            b"{\"a\":1,}",
            b"01",
            b"\"\x01\"",
            b"\"\\ud800\"",
            b"\"\\q\"",
            b"tru",
            b"{} x",
            b"",
        ] {
            assert_eq!(parse_json(bad), None, "{:?}", String::from_utf8_lossy(bad));
        }
        let deep = "[".repeat(65) + &"]".repeat(65);
        assert_eq!(parse_json(deep.as_bytes()), None);
        let ok = "[".repeat(64) + &"]".repeat(64);
        assert!(parse_json(ok.as_bytes()).is_some());
    }

    #[test]
    fn locate_finds_the_named_tensor_and_validates_its_extent() {
        let header = r#"{"__metadata__":{"format":"pt"},"b":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]},"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}  "#;
        let bytes = st_file(header, &[0u8; 12]);
        let len = bytes.len() as u64;
        let mut cur = std::io::Cursor::new(&bytes);
        let a = st_locate(&mut cur, len, "a").unwrap();
        let base = 8 + header.len() as u64;
        assert_eq!(
            a,
            TensorLoc {
                dtype: DT_F32,
                shape: vec![2],
                data_off: base,
                nbytes: 8,
                fortran: false
            }
        );
        let b = st_locate(&mut cur, len, "b").unwrap();
        assert_eq!((b.dtype, b.data_off, b.nbytes), (DT_BF16, base + 8, 4));
        assert_eq!(st_locate(&mut cur, len, "c"), Err(IO_NAME_NOT_FOUND));
        assert_eq!(
            st_locate(&mut cur, len, "__metadata__"),
            Err(IO_NAME_NOT_FOUND)
        );
        // The payload ends past the file: one byte short.
        let mut cut = std::io::Cursor::new(&bytes[..bytes.len() - 1]);
        assert_eq!(st_locate(&mut cut, len - 1, "b"), Err(IO_BAD_HEADER));
    }

    #[test]
    fn locate_rejects_malformed_headers() {
        let cases = [
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#, // 4 bytes for 2 f32
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#, // begin > end
            r#"{"a":{"dtype":"F32","shape":[1]}}"#,                      // no offsets
            r#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#, // negative dim
            r#"{"a":{"dtype":3,"shape":[1],"data_offsets":[0,4]}}"#,     // non-string dtype
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            r#"["a"]"#,
            r#"{"a":"#,
        ];
        for h in cases {
            let bytes = st_file(h, &[0u8; 16]);
            let mut cur = std::io::Cursor::new(&bytes);
            assert_eq!(
                st_locate(&mut cur, bytes.len() as u64, "a"),
                Err(IO_BAD_HEADER),
                "{h}"
            );
        }
        // A length prefix larger than the file.
        let mut bytes = 1000u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        let mut cur = std::io::Cursor::new(&bytes);
        assert_eq!(
            st_locate(&mut cur, bytes.len() as u64, "a"),
            Err(IO_BAD_HEADER)
        );
        let mut tiny = std::io::Cursor::new(&[0u8; 4]);
        assert_eq!(st_locate(&mut tiny, 4, "a"), Err(IO_BAD_HEADER));
        // An unsupported dtype parses; the destination check rejects it.
        let bytes = st_file(
            r#"{"a":{"dtype":"BOOL","shape":[3],"data_offsets":[0,3]}}"#,
            &[1, 0, 1],
        );
        let mut cur = std::io::Cursor::new(&bytes);
        assert_eq!(
            st_locate(&mut cur, bytes.len() as u64, "a").map(|l| l.dtype),
            Ok(DT_UNSUPPORTED)
        );
    }

    #[test]
    fn st_read_loads_verbatim_widens_halves_and_checks_the_destination() {
        // "w": bf16 [2, 2] = 1.0, -2.0, 0.5, 3.0; "n": i16 [3] = -1, 2, 300.
        let mut payload = Vec::new();
        for bits in [0x3F80u16, 0xC000, 0x3F00, 0x4040] {
            payload.extend_from_slice(&bits.to_le_bytes());
        }
        for v in [-1i16, 2, 300] {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let header = r#"{"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"n":{"dtype":"I16","shape":[3],"data_offsets":[8,14]}}"#;
        let f = Scratch::new("st", &st_file(header, &payload));
        let w = std::ffi::CString::new("w").unwrap();
        let n = std::ffi::CString::new("n").unwrap();
        let wp = w.as_ptr() as *const u8;
        let np = n.as_ptr() as *const u8;
        let dims = [2i64, 2];
        let mut out = [0f32; 4];
        let got = unsafe {
            wukong_rt_st_read(
                f.ptr(),
                wp,
                out.as_mut_ptr() as *mut u8,
                4,
                DT_F32,
                dims.as_ptr(),
                2,
            )
        };
        assert_eq!((got, out), (4, [1.0, -2.0, 0.5, 3.0]));
        let mut raw = [0u16; 4];
        let got = unsafe {
            wukong_rt_st_read(
                f.ptr(),
                wp,
                raw.as_mut_ptr() as *mut u8,
                4,
                DT_BF16,
                dims.as_ptr(),
                -1,
            )
        };
        assert_eq!((got, raw), (4, [0x3F80, 0xC000, 0x3F00, 0x4040]));
        let mut ints = [0i16; 3];
        let got = unsafe {
            wukong_rt_st_read(
                f.ptr(),
                np,
                ints.as_mut_ptr() as *mut u8,
                3,
                DT_I16,
                dims.as_ptr(),
                -1,
            )
        };
        assert_eq!((got, ints), (3, [-1, 2, 300]));
        // Wrong dims, wrong count, wrong dtype: nothing is written.
        let mut keep = [9f32; 4];
        let bad_dims = [4i64];
        let got = unsafe {
            wukong_rt_st_read(
                f.ptr(),
                wp,
                keep.as_mut_ptr() as *mut u8,
                4,
                DT_F32,
                bad_dims.as_ptr(),
                1,
            )
        };
        assert_eq!((got, keep), (IO_SHAPE_MISMATCH, [9.0; 4]));
        let got = unsafe {
            wukong_rt_st_read(
                f.ptr(),
                wp,
                keep.as_mut_ptr() as *mut u8,
                3,
                DT_F32,
                dims.as_ptr(),
                -1,
            )
        };
        assert_eq!(got, IO_SHAPE_MISMATCH);
        let got = unsafe {
            wukong_rt_st_read(
                f.ptr(),
                np,
                keep.as_mut_ptr() as *mut u8,
                3,
                DT_F32,
                dims.as_ptr(),
                -1,
            )
        };
        assert_eq!(got, IO_DTYPE_MISMATCH);
        assert_eq!(keep, [9.0; 4]);
        let missing = std::ffi::CString::new("wukong_st_absent_zz.safetensors").unwrap();
        let got = unsafe {
            wukong_rt_st_read(
                missing.as_ptr() as *const u8,
                wp,
                keep.as_mut_ptr() as *mut u8,
                4,
                DT_F32,
                dims.as_ptr(),
                -1,
            )
        };
        assert_eq!(got, IO_OPEN_FAILED);
    }
}
