//! The GGUF reader behind `gguf_read_<T>(path, name, buf)` and the `gguf_find` / `gguf_rank` /
//! `gguf_dim` / `gguf_type` queries: llama.cpp's checkpoint container, with the ggml
//! block-quantized formats dequantized to `f32` by the reference layouts.
//!
//! # The container (versions 2 and 3; every integer little-endian)
//!
//! ```text
//! "GGUF"  u32 version  u64 tensor_count  u64 kv_count
//! kv_count × { string key, u32 value_type, value }
//! tensor_count × { string name, u32 n_dims, u64 ne[n_dims], u32 ggml_type, u64 offset }
//! padding to `general.alignment` (default 32), then the tensor data
//! ```
//!
//! A string is a `u64` byte length and that many bytes (no terminator). A value is one of thirteen
//! types — `u8 i8 u16 i16 u32 i32 f32 bool string array u64 i64 f64` (ids 0–12) — and an array is
//! `{u32 element_type, u64 count, count values}`, arrays of arrays included. Every value is parsed
//! (and kept, as [`MetaValue`]): the tensor table sits after the metadata, so a reader cannot reach
//! it without walking every value, and `general.alignment` must be a `u32` power of two.
//!
//! A tensor's `ne[0]` is its **innermost** (fastest-varying) dimension; Wukong, like safetensors,
//! NumPy and PyTorch, spells a shape outermost first, so the GGUF dims `[4096, 11008]` are the
//! tensor `Tensor[f32, 11008, 4096]`. Every shape this module returns ([`GgufLoc::shape`], the
//! `gguf_dim` query) is in that outermost-first order. A tensor's payload starts `offset` bytes
//! into the data section (a multiple of the alignment) and is `numel / block · type_size` bytes,
//! from ggml's type table ([`ggml_type`]).
//!
//! The parse is strict where leniency would make two readers disagree, matching llama.cpp's
//! `gguf_init_from_file`: a bad magic, a version other than 2 or 3, an unknown value type, a
//! duplicated key or tensor name, more than four dimensions, a dimension above `i64::MAX`, an
//! alignment that is not a `u32` power of two, and a length that runs past the end of the file are
//! all [`IO_BAD_HEADER`]. The *named* tensor is then checked like a safetensors entry: a row length
//! that is not a whole number of blocks, an offset off the alignment and a payload outside the file
//! are [`IO_BAD_HEADER`]; a type id outside ggml's table is [`IO_UNSUPPORTED_TYPE`].
//!
//! # Reading a tensor
//!
//! [`gguf_plan`] compares the tensor with the destination in [`fileio::check_dest`]'s order —
//! dtype, then rank and dims (for a shape-typed destination), then the exact element count:
//!
//! * `F32 F64 F16 BF16 I8 I16 I32 I64` load **verbatim** into a buffer of that element type, and
//!   `F16`/`BF16` also **widen exactly** into `f32` (the safetensors rule);
//! * a **block-quantized** tensor dequantizes into `f32` ([`Dequant`]: `Q4_0 Q4_1 Q5_0 Q5_1 Q8_0`,
//!   the K-quants `Q2_K Q3_K Q4_K Q5_K Q6_K Q8_K`, and `IQ4_NL`/`IQ4_XS`), or loads its **raw block
//!   bytes** into a `u8` buffer — the operand the `gemv_q8_0` / `gemv_q4_0` decode builtins take,
//!   whose `[N, ceil(K/32)]` block rows are exactly ggml's row-major layout. A tensor-view `u8`
//!   destination is checked against the *storage* shape: the logical shape with its innermost
//!   dimension replaced by its row's byte length;
//! * a block type with no dequantizer here (the grid-based i-quants, `Q8_1`, `TQ1_0`/`TQ2_0`,
//!   `MXFP4`) read into `f32` is [`IO_UNSUPPORTED_TYPE`]; any other pairing is
//!   [`IO_DTYPE_MISMATCH`].
//!
//! # Dequantization
//!
//! Each `dequantize_row_*` of ggml's reference C (`ggml-quants.c`) is transcribed block for block:
//! the same `f16` → `f32` scales, the same integer unpacking, and the same `f32` arithmetic in the
//! same order, each product and sum rounded once (`x0·d + m`, `d1·q − m1`, `(d·sc)·q`) — no fused
//! multiply-add. The interpreter calls these very functions, so a dequantized value cannot differ
//! between backends.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::fileio::{
    self, TensorLoc, DT_BF16, DT_F16, DT_F32, DT_F64, DT_I16, DT_I32, DT_I64, DT_I8, DT_U8,
    IO_BAD_HEADER, IO_DTYPE_MISMATCH, IO_ERROR, IO_NAME_NOT_FOUND, IO_OPEN_FAILED,
    IO_SHAPE_MISMATCH, IO_UNSUPPORTED_TYPE,
};

/// The four magic bytes every GGUF file starts with.
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
/// The data-section alignment when `general.alignment` is absent.
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
/// The most dimensions a tensor may have (ggml's `GGML_MAX_DIMS`).
pub const GGUF_MAX_DIMS: usize = 4;
/// How deeply arrays may nest. The format allows arrays of arrays; the limit only exists so a
/// hostile header cannot exhaust the stack.
const MAX_NESTING: u32 = 64;

/// Value type ids of the metadata key/value section.
pub mod vt {
    /// `u8`.
    pub const U8: u32 = 0;
    /// `i8`.
    pub const I8: u32 = 1;
    /// `u16`.
    pub const U16: u32 = 2;
    /// `i16`.
    pub const I16: u32 = 3;
    /// `u32`.
    pub const U32: u32 = 4;
    /// `i32`.
    pub const I32: u32 = 5;
    /// `f32`.
    pub const F32: u32 = 6;
    /// `bool` (one byte; any non-zero byte is `true`, as llama.cpp reads it).
    pub const BOOL: u32 = 7;
    /// A `u64`-length-prefixed byte string.
    pub const STRING: u32 = 8;
    /// `{u32 element_type, u64 count, values}`.
    pub const ARRAY: u32 = 9;
    /// `u64`.
    pub const U64: u32 = 10;
    /// `i64`.
    pub const I64: u32 = 11;
    /// `f64`.
    pub const F64: u32 = 12;
}

/// One parsed metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    /// The raw bytes (GGUF strings are UTF-8 by convention; nothing here depends on it).
    Str(Vec<u8>),
    /// An array: its element type id (kept, so an empty array round-trips) and its values.
    Arr(u32, Vec<MetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    /// This value's type id ([`vt`]).
    pub fn type_id(&self) -> u32 {
        match self {
            MetaValue::U8(_) => vt::U8,
            MetaValue::I8(_) => vt::I8,
            MetaValue::U16(_) => vt::U16,
            MetaValue::I16(_) => vt::I16,
            MetaValue::U32(_) => vt::U32,
            MetaValue::I32(_) => vt::I32,
            MetaValue::F32(_) => vt::F32,
            MetaValue::Bool(_) => vt::BOOL,
            MetaValue::Str(_) => vt::STRING,
            MetaValue::Arr(..) => vt::ARRAY,
            MetaValue::U64(_) => vt::U64,
            MetaValue::I64(_) => vt::I64,
            MetaValue::F64(_) => vt::F64,
        }
    }
}

/// One entry of the tensor table, as stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufTensorInfo {
    /// The tensor's name (raw bytes).
    pub name: Vec<u8>,
    /// `ne[0..n_dims]`: the **innermost** dimension first, as the file stores it.
    pub ne: Vec<u64>,
    /// The ggml type id.
    pub ggml_type: u32,
    /// Byte offset of the payload from the start of the data section.
    pub offset: u64,
}

/// A parsed GGUF header.
#[derive(Clone, Debug, PartialEq)]
pub struct Gguf {
    /// The format version (2 or 3).
    pub version: u32,
    /// Every metadata key/value pair, in file order.
    pub metadata: Vec<(Vec<u8>, MetaValue)>,
    /// The tensor table, in file order.
    pub tensors: Vec<GgufTensorInfo>,
    /// The data-section alignment (`general.alignment`, else 32).
    pub alignment: u64,
    /// Absolute byte offset of the data section.
    pub data_start: u64,
}

impl Gguf {
    /// The value of metadata key `key`, if present.
    pub fn meta(&self, key: &[u8]) -> Option<&MetaValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// The index of the tensor called `name` in the tensor table, if present.
    pub fn find(&self, name: &[u8]) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }
}

/// A block-quantized format this module can dequantize to `f32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum Dequant {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ4_NL,
    IQ4_XS,
}

/// One row of ggml's type table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GgmlType {
    /// The ggml type id (`GGML_TYPE_*`).
    pub id: u32,
    /// ggml's name for the type.
    pub name: &'static str,
    /// Elements per block (1 for a plain element type).
    pub block: u64,
    /// Bytes per block.
    pub bytes: u64,
    /// The Wukong dtype code of a plain element type (loaded verbatim), else `None`.
    pub dt: Option<i64>,
    /// The dequantizer of a block type, if this module has one.
    pub dequant: Option<Dequant>,
}

impl GgmlType {
    /// Whether this is a block-quantized type (more than one element per block, or one whose
    /// elements are not a Wukong scalar).
    pub fn is_block(&self) -> bool {
        self.dt.is_none()
    }
}

/// The row of ggml's type table for id `id`, or `None` for an id ggml never assigned or retired
/// (4 and 5 were `Q4_2`/`Q4_3`, 31–33 and 36–38 the repacked `Q4_0_4_4` … `IQ4_NL_8_8` layouts,
/// which never appear in a file).
pub fn ggml_type(id: u32) -> Option<GgmlType> {
    let (name, block, bytes, dt, dequant) = match id {
        0 => ("F32", 1, 4, Some(DT_F32), None),
        1 => ("F16", 1, 2, Some(DT_F16), None),
        2 => ("Q4_0", 32, 18, None, Some(Dequant::Q4_0)),
        3 => ("Q4_1", 32, 20, None, Some(Dequant::Q4_1)),
        6 => ("Q5_0", 32, 22, None, Some(Dequant::Q5_0)),
        7 => ("Q5_1", 32, 24, None, Some(Dequant::Q5_1)),
        8 => ("Q8_0", 32, 34, None, Some(Dequant::Q8_0)),
        9 => ("Q8_1", 32, 36, None, None),
        10 => ("Q2_K", 256, 84, None, Some(Dequant::Q2_K)),
        11 => ("Q3_K", 256, 110, None, Some(Dequant::Q3_K)),
        12 => ("Q4_K", 256, 144, None, Some(Dequant::Q4_K)),
        13 => ("Q5_K", 256, 176, None, Some(Dequant::Q5_K)),
        14 => ("Q6_K", 256, 210, None, Some(Dequant::Q6_K)),
        15 => ("Q8_K", 256, 292, None, Some(Dequant::Q8_K)),
        16 => ("IQ2_XXS", 256, 66, None, None),
        17 => ("IQ2_XS", 256, 74, None, None),
        18 => ("IQ3_XXS", 256, 98, None, None),
        19 => ("IQ1_S", 256, 50, None, None),
        20 => ("IQ4_NL", 32, 18, None, Some(Dequant::IQ4_NL)),
        21 => ("IQ3_S", 256, 110, None, None),
        22 => ("IQ2_S", 256, 82, None, None),
        23 => ("IQ4_XS", 256, 136, None, Some(Dequant::IQ4_XS)),
        24 => ("I8", 1, 1, Some(DT_I8), None),
        25 => ("I16", 1, 2, Some(DT_I16), None),
        26 => ("I32", 1, 4, Some(DT_I32), None),
        27 => ("I64", 1, 8, Some(DT_I64), None),
        28 => ("F64", 1, 8, Some(DT_F64), None),
        29 => ("IQ1_M", 256, 56, None, None),
        30 => ("BF16", 1, 2, Some(DT_BF16), None),
        34 => ("TQ1_0", 256, 54, None, None),
        35 => ("TQ2_0", 256, 66, None, None),
        39 => ("MXFP4", 32, 17, None, None),
        _ => return None,
    };
    Some(GgmlType {
        id,
        name,
        block,
        bytes,
        dt,
        dequant,
    })
}

impl Dequant {
    /// Elements per block.
    pub fn block(self) -> usize {
        match self {
            Dequant::Q4_0
            | Dequant::Q4_1
            | Dequant::Q5_0
            | Dequant::Q5_1
            | Dequant::Q8_0
            | Dequant::IQ4_NL => 32,
            _ => 256,
        }
    }

    /// Bytes per block.
    pub fn bytes(self) -> usize {
        match self {
            Dequant::Q4_0 => 18,
            Dequant::Q4_1 => 20,
            Dequant::Q5_0 => 22,
            Dequant::Q5_1 => 24,
            Dequant::Q8_0 => 34,
            Dequant::Q2_K => 84,
            Dequant::Q3_K => 110,
            Dequant::Q4_K => 144,
            Dequant::Q5_K => 176,
            Dequant::Q6_K => 210,
            Dequant::Q8_K => 292,
            Dequant::IQ4_NL => 18,
            Dequant::IQ4_XS => 136,
        }
    }
}

// --- the header parser ------------------------------------------------------------------------

/// A bounded little-endian reader: every read is checked against the file's length first, so a
/// length that runs past the end is [`IO_BAD_HEADER`] (never a giant allocation), and only a real
/// read failure is [`IO_ERROR`].
struct Cursor<R> {
    r: R,
    pos: u64,
    len: u64,
}

impl<R: Read> Cursor<R> {
    fn need(&self, n: u64) -> Result<(), i64> {
        if self.len.saturating_sub(self.pos) < n {
            return Err(IO_BAD_HEADER);
        }
        Ok(())
    }

    fn fill(&mut self, buf: &mut [u8]) -> Result<(), i64> {
        self.need(buf.len() as u64)?;
        self.r.read_exact(buf).map_err(|_| IO_ERROR)?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn arr<const N: usize>(&mut self) -> Result<[u8; N], i64> {
        let mut b = [0u8; N];
        self.fill(&mut b)?;
        Ok(b)
    }

    fn u32(&mut self) -> Result<u32, i64> {
        Ok(u32::from_le_bytes(self.arr()?))
    }

    fn u64(&mut self) -> Result<u64, i64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }

    fn string(&mut self) -> Result<Vec<u8>, i64> {
        let n = self.u64()?;
        self.need(n)?;
        let n = usize::try_from(n).map_err(|_| IO_BAD_HEADER)?;
        let mut s = Vec::new();
        s.try_reserve_exact(n).map_err(|_| IO_BAD_HEADER)?;
        s.resize(n, 0u8);
        self.fill(&mut s)?;
        Ok(s)
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<MetaValue, i64> {
        Ok(match ty {
            vt::U8 => MetaValue::U8(self.arr::<1>()?[0]),
            vt::I8 => MetaValue::I8(self.arr::<1>()?[0] as i8),
            vt::U16 => MetaValue::U16(u16::from_le_bytes(self.arr()?)),
            vt::I16 => MetaValue::I16(i16::from_le_bytes(self.arr()?)),
            vt::U32 => MetaValue::U32(self.u32()?),
            vt::I32 => MetaValue::I32(i32::from_le_bytes(self.arr()?)),
            vt::F32 => MetaValue::F32(f32::from_le_bytes(self.arr()?)),
            vt::BOOL => MetaValue::Bool(self.arr::<1>()?[0] != 0),
            vt::STRING => MetaValue::Str(self.string()?),
            vt::ARRAY => {
                if depth >= MAX_NESTING {
                    return Err(IO_BAD_HEADER);
                }
                let elem = self.u32()?;
                let n = self.u64()?;
                // Every element occupies at least this many bytes, so a count the rest of the file
                // cannot hold is malformed before anything is allocated for it.
                let min = match elem {
                    vt::U8 | vt::I8 | vt::BOOL => 1,
                    vt::U16 | vt::I16 => 2,
                    vt::U32 | vt::I32 | vt::F32 => 4,
                    vt::U64 | vt::I64 | vt::F64 | vt::STRING => 8,
                    vt::ARRAY => 12,
                    _ => return Err(IO_BAD_HEADER),
                };
                self.need(n.checked_mul(min).ok_or(IO_BAD_HEADER)?)?;
                let n = usize::try_from(n).map_err(|_| IO_BAD_HEADER)?;
                let mut items = Vec::new();
                items.try_reserve_exact(n).map_err(|_| IO_BAD_HEADER)?;
                for _ in 0..n {
                    items.push(self.value(elem, depth + 1)?);
                }
                MetaValue::Arr(elem, items)
            }
            vt::U64 => MetaValue::U64(self.u64()?),
            vt::I64 => MetaValue::I64(i64::from_le_bytes(self.arr()?)),
            vt::F64 => MetaValue::F64(f64::from_le_bytes(self.arr()?)),
            _ => return Err(IO_BAD_HEADER),
        })
    }
}

/// Parse the GGUF header of a stream of `file_len` bytes positioned at its first byte: the magic,
/// the version, every metadata value and the whole tensor table (see the module docs for what is
/// malformed). Nothing past the tensor table is read.
pub fn gguf_parse<R: Read>(r: R, file_len: u64) -> Result<Gguf, i64> {
    let mut c = Cursor {
        r,
        pos: 0,
        len: file_len,
    };
    if c.arr::<4>()? != GGUF_MAGIC {
        return Err(IO_BAD_HEADER);
    }
    let version = c.u32()?;
    if !(2..=3).contains(&version) {
        return Err(IO_BAD_HEADER);
    }
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;
    let mut metadata: Vec<(Vec<u8>, MetaValue)> = Vec::new();
    let mut keys = std::collections::HashSet::new();
    for _ in 0..n_kv {
        let key = c.string()?;
        let ty = c.u32()?;
        let v = c.value(ty, 0)?;
        if !keys.insert(key.clone()) {
            return Err(IO_BAD_HEADER);
        }
        metadata.push((key, v));
    }
    let alignment = match metadata
        .iter()
        .find(|(k, _)| k == b"general.alignment")
        .map(|(_, v)| v)
    {
        None => GGUF_DEFAULT_ALIGNMENT,
        Some(MetaValue::U32(a)) if a.is_power_of_two() => u64::from(*a),
        Some(_) => return Err(IO_BAD_HEADER),
    };
    let mut tensors = Vec::new();
    let mut names = std::collections::HashSet::new();
    for _ in 0..n_tensors {
        let name = c.string()?;
        let n_dims = c.u32()? as usize;
        if n_dims > GGUF_MAX_DIMS {
            return Err(IO_BAD_HEADER);
        }
        let mut ne = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            let d = c.u64()?;
            if d > i64::MAX as u64 {
                return Err(IO_BAD_HEADER);
            }
            ne.push(d);
        }
        let ggml_type = c.u32()?;
        let offset = c.u64()?;
        if !names.insert(name.clone()) {
            return Err(IO_BAD_HEADER);
        }
        tensors.push(GgufTensorInfo {
            name,
            ne,
            ggml_type,
            offset,
        });
    }
    let data_start = c
        .pos
        .checked_next_multiple_of(alignment)
        .ok_or(IO_BAD_HEADER)?;
    Ok(Gguf {
        version,
        metadata,
        tensors,
        alignment,
        data_start,
    })
}

/// Open `path`, size it and parse its GGUF header — `-1` if it cannot be opened, `-2` if it cannot
/// be sized, else the parse's own result. The order is the contract both backends share.
pub fn gguf_open(path: &Path) -> Result<(File, Gguf, u64), i64> {
    let mut file = File::open(path).map_err(|_| IO_OPEN_FAILED)?;
    let file_len = file.metadata().map_err(|_| IO_ERROR)?.len();
    let g = gguf_parse(BufReader::new(&mut file), file_len)?;
    Ok((file, g, file_len))
}

// --- locating and planning a read --------------------------------------------------------------

/// A named tensor's payload, checked against the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufLoc {
    /// Its type.
    pub ty: GgmlType,
    /// Its shape, **outermost first** (the reverse of the file's `ne`).
    pub shape: Vec<u64>,
    /// Absolute byte offset of the payload.
    pub data_off: u64,
    /// Payload length in bytes.
    pub nbytes: u64,
}

impl GgufLoc {
    /// The storage shape of a block type: [`Self::shape`] with its innermost dimension replaced by
    /// one row's byte length. A plain type's storage shape is its shape.
    pub fn storage_shape(&self) -> Vec<u64> {
        let mut s = self.shape.clone();
        if let Some(inner) = s.last_mut() {
            *inner = *inner / self.ty.block * self.ty.bytes;
        }
        s
    }
}

/// Locate tensor `name` of the parsed file `g` (`file_len` bytes): `-5` if there is none, `-7` if
/// its type id is not in ggml's table, `-6` if its row is not a whole number of blocks, its offset
/// is off the alignment, or its payload is not inside the file.
pub fn gguf_locate(g: &Gguf, name: &[u8], file_len: u64) -> Result<GgufLoc, i64> {
    let t = &g.tensors[g.find(name).ok_or(IO_NAME_NOT_FOUND)?];
    let ty = ggml_type(t.ggml_type).ok_or(IO_UNSUPPORTED_TYPE)?;
    let numel = fileio::shape_count(&t.ne)?;
    let inner = t.ne.first().copied().unwrap_or(1);
    if !inner.is_multiple_of(ty.block) {
        return Err(IO_BAD_HEADER);
    }
    let nbytes = (numel / ty.block)
        .checked_mul(ty.bytes)
        .ok_or(IO_BAD_HEADER)?;
    if !t.offset.is_multiple_of(g.alignment) {
        return Err(IO_BAD_HEADER);
    }
    let data_off = g.data_start.checked_add(t.offset).ok_or(IO_BAD_HEADER)?;
    let end = data_off.checked_add(nbytes).ok_or(IO_BAD_HEADER)?;
    if end > file_len {
        return Err(IO_BAD_HEADER);
    }
    Ok(GgufLoc {
        ty,
        shape: t.ne.iter().rev().copied().collect(),
        data_off,
        nbytes,
    })
}

/// How a checked GGUF tensor moves into its destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufPlan {
    /// `count` elements of the destination's own dtype, byte for byte.
    Verbatim { count: usize },
    /// `count` `bf16`/`f16` elements widened exactly to `f32`.
    Widen { count: usize },
    /// `count` elements dequantized to `f32`.
    Dequant { count: usize, kind: Dequant },
    /// `count` raw block bytes into a `u8` destination.
    Raw { count: usize },
}

impl GgufPlan {
    /// The element count the read transfers (its result).
    pub fn count(self) -> usize {
        match self {
            GgufPlan::Verbatim { count }
            | GgufPlan::Widen { count }
            | GgufPlan::Dequant { count, .. }
            | GgufPlan::Raw { count } => count,
        }
    }
}

/// Validate a located tensor against a destination of `dest_len` elements of `dest_dt` (with
/// tensor-view `dims`, outermost first), in the module docs' order: the type and dtype (`-7`,
/// `-3`), then the dims and the exact count (`-4`).
pub fn gguf_plan(
    loc: &GgufLoc,
    dest_dt: i64,
    dest_len: i64,
    dims: Option<&[i64]>,
) -> Result<GgufPlan, i64> {
    if let Some(dt) = loc.ty.dt {
        let flat = TensorLoc {
            dtype: dt,
            shape: loc.shape.clone(),
            data_off: loc.data_off,
            nbytes: loc.nbytes,
            fortran: false,
        };
        let (count, widen) = fileio::check_dest(&flat, dest_dt, dest_len, dims)?;
        return Ok(if widen {
            GgufPlan::Widen { count }
        } else {
            GgufPlan::Verbatim { count }
        });
    }
    match dest_dt {
        DT_F32 => {
            let kind = loc.ty.dequant.ok_or(IO_UNSUPPORTED_TYPE)?;
            let count = fileio::check_shape(&loc.shape, dest_len, dims)?;
            Ok(GgufPlan::Dequant { count, kind })
        }
        DT_U8 => {
            let count = fileio::check_shape(&loc.storage_shape(), dest_len, dims)?;
            Ok(GgufPlan::Raw { count })
        }
        _ => Err(IO_DTYPE_MISMATCH),
    }
}

/// Byte width of one destination element under `plan` for a destination of `dest_dt`.
fn dest_size(plan: GgufPlan, dest_dt: i64) -> Option<usize> {
    match plan {
        GgufPlan::Widen { .. } | GgufPlan::Dequant { .. } => Some(4),
        GgufPlan::Raw { .. } => Some(1),
        GgufPlan::Verbatim { .. } => fileio::dtype_size(dest_dt),
    }
}

/// Blocks (or half elements) per staged chunk of a widening or dequantizing transfer.
const STAGE_UNITS: usize = 1 << 12;

/// Move the payload of `loc` into `out` under `plan`, as the destination's **little-endian** bytes
/// (`plan.count()` elements of the destination dtype: exactly `out.len()` bytes). Verbatim and raw
/// transfers read straight into `out`; a widening or dequantizing one stages a bounded chunk at a
/// time. Any read failure is [`IO_ERROR`].
pub fn gguf_transfer<R: Read + Seek>(
    f: &mut R,
    loc: &GgufLoc,
    plan: GgufPlan,
    out: &mut [u8],
) -> Result<(), i64> {
    if plan.count() == 0 {
        return Ok(());
    }
    f.seek(SeekFrom::Start(loc.data_off))
        .map_err(|_| IO_ERROR)?;
    match plan {
        GgufPlan::Verbatim { .. } | GgufPlan::Raw { .. } => f.read_exact(out).map_err(|_| IO_ERROR),
        GgufPlan::Widen { count } => {
            let src_dt = loc.ty.dt.unwrap_or(DT_F16);
            let mut stage = vec![0u8; count.min(STAGE_UNITS) * 2];
            let mut done = 0usize;
            while done < count {
                let m = (count - done).min(STAGE_UNITS);
                let chunk = &mut stage[..m * 2];
                f.read_exact(chunk).map_err(|_| IO_ERROR)?;
                for (j, h) in chunk.chunks_exact(2).enumerate() {
                    let x = fileio::widen_half(u16::from_le_bytes([h[0], h[1]]), src_dt);
                    let at = (done + j) * 4;
                    out[at..at + 4].copy_from_slice(&x.to_le_bytes());
                }
                done += m;
            }
            Ok(())
        }
        GgufPlan::Dequant { count, kind } => {
            let (qk, bb) = (kind.block(), kind.bytes());
            let blocks = count / qk;
            let mut stage = vec![0u8; blocks.min(STAGE_UNITS) * bb];
            let mut y = vec![0f32; qk];
            let mut done = 0usize;
            while done < blocks {
                let m = (blocks - done).min(STAGE_UNITS);
                let chunk = &mut stage[..m * bb];
                f.read_exact(chunk).map_err(|_| IO_ERROR)?;
                for (j, blk) in chunk.chunks_exact(bb).enumerate() {
                    dequant_block(kind, blk, &mut y);
                    let at = (done + j) * qk * 4;
                    for (k, v) in y.iter().enumerate() {
                        out[at + 4 * k..at + 4 * k + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
                done += m;
            }
            Ok(())
        }
    }
}

/// Open, parse, locate and plan a read of tensor `name` of `path` — every check, in the contract's
/// order, before a byte of payload moves. Shared by the native entry and the interpreter.
pub fn gguf_open_tensor(
    path: &Path,
    name: &[u8],
    dest_dt: i64,
    dest_len: i64,
    dims: Option<&[i64]>,
) -> Result<(File, GgufLoc, GgufPlan), i64> {
    let (file, g, file_len) = gguf_open(path)?;
    let loc = gguf_locate(&g, name, file_len)?;
    let plan = gguf_plan(&loc, dest_dt, dest_len, dims)?;
    Ok((file, loc, plan))
}

/// `wukong_rt_gguf_read(path, name, data, len, dtype, dims, rank) -> i64` — load GGUF tensor `name`
/// from `path` into `data` (`len` elements of `dtype`): verbatim, widened, dequantized to `f32`, or
/// as raw block bytes (see the module docs). `rank < 0` means the destination is a flat buffer;
/// otherwise `dims[..rank]` is its declared shape (outermost first) and must equal the tensor's
/// (its storage shape, for raw blocks). Returns the element count, or `-1` (open), `-2` (I/O), `-3`
/// (dtype), `-4` (shape or count), `-5` (no such name), `-6` (malformed header), `-7` (a type with
/// no decoder). On every code but `-2` the destination is untouched.
///
/// # Safety
/// `path` and `name` must be null or NUL-terminated; `data` must address `len` writable elements of
/// `dtype`; for `rank > 0`, `dims` must address `rank` readable `i64`s.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_gguf_read(
    path: *const u8,
    name: *const u8,
    data: *mut u8,
    len: i64,
    dtype: i64,
    dims: *const i64,
    rank: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); `path`/`name` are
    // NUL-terminated strings (no extent to check).
    if let Some(sz) = fileio::dtype_size(dtype) {
        crate::safety::debug_extent_bytes(data, len, sz);
    }
    crate::safety::debug_extent(dims, rank);
    // SAFETY: every pointer is forwarded under this function's own contract.
    let (path, name, dims) = unsafe {
        (
            crate::rt_path(path),
            fileio::cstr_lossy(name),
            fileio::dims_arg(dims, rank),
        )
    };
    let Some(path) = path else {
        return IO_OPEN_FAILED;
    };
    let (mut file, loc, plan) = match gguf_open_tensor(&path, name.as_bytes(), dtype, len, dims) {
        Ok(found) => found,
        Err(code) => return code,
    };
    let count = plan.count();
    let Some(sz) = dest_size(plan, dtype) else {
        return IO_ERROR;
    };
    if count == 0 {
        return 0;
    }
    // SAFETY: `count == len` (checked by the plan) elements of `sz` bytes are writable at `data`.
    let out = unsafe { std::slice::from_raw_parts_mut(data, count * sz) };
    if let Err(code) = gguf_transfer(&mut file, &loc, plan, out) {
        return code;
    }
    fileio::le_to_host_in_place(out, sz);
    count as i64
}

// --- the queries -------------------------------------------------------------------------------

/// `gguf_find`: the tensor's index in the file's tensor table.
pub const GGUF_Q_FIND: i64 = 0;
/// `gguf_rank`: its number of dimensions.
pub const GGUF_Q_RANK: i64 = 1;
/// `gguf_dim`: dimension `k`, outermost first.
pub const GGUF_Q_DIM: i64 = 2;
/// `gguf_type`: its ggml type id.
pub const GGUF_Q_TYPE: i64 = 3;

/// Answer query `what` ([`GGUF_Q_FIND`] …) about tensor `name` of the parsed file `g`: the answer,
/// `-5` if there is no such tensor, `-4` for a `gguf_dim` index outside `0..rank`, `-2` for an
/// unknown query code.
pub fn gguf_query(g: &Gguf, name: &[u8], what: i64, k: i64) -> i64 {
    let Some(i) = g.find(name) else {
        return IO_NAME_NOT_FOUND;
    };
    let t = &g.tensors[i];
    match what {
        GGUF_Q_FIND => i as i64,
        GGUF_Q_RANK => t.ne.len() as i64,
        GGUF_Q_DIM => match usize::try_from(k) {
            // `ne` is innermost first; dimension `k` counts from the outermost.
            Ok(k) if k < t.ne.len() => t.ne[t.ne.len() - 1 - k] as i64,
            _ => IO_SHAPE_MISMATCH,
        },
        GGUF_Q_TYPE => i64::from(t.ggml_type),
        _ => IO_ERROR,
    }
}

/// [`gguf_query`] on the file at `path`: `-1` if it cannot be opened, `-2` if it cannot be sized,
/// `-6` if its header is malformed. Shared by the native entry and the interpreter.
pub fn gguf_query_path(path: &Path, name: &[u8], what: i64, k: i64) -> i64 {
    match gguf_open(path) {
        Ok((_, g, _)) => gguf_query(&g, name, what, k),
        Err(code) => code,
    }
}

/// `wukong_rt_gguf_query(path, name, what, k) -> i64` — the `gguf_find` / `gguf_rank` /
/// `gguf_dim` / `gguf_type` builtins ([`gguf_query_path`]).
///
/// # Safety
/// `path` and `name` must be null or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_gguf_query(
    path: *const u8,
    name: *const u8,
    what: i64,
    k: i64,
) -> i64 {
    // SAFETY-EXEMPT: `path` and `name` are NUL-terminated strings -- they carry no extent to check.
    // SAFETY: forwarded from this function's contract.
    let (path, name) = unsafe { (crate::rt_path(path), fileio::cstr_lossy(name)) };
    match path {
        Some(path) => gguf_query_path(&path, name.as_bytes(), what, k),
        None => IO_OPEN_FAILED,
    }
}

// --- metadata ---------------------------------------------------------------------------------

/// `gguf_meta_type`: the value's type id ([`vt`]; `9` for an array).
pub const GGUF_M_TYPE: i64 = 0;
/// `gguf_meta_len`: an array's element count, a string's byte length, `1` for any other value.
pub const GGUF_M_LEN: i64 = 1;
/// `gguf_meta_i64`: an integer or `bool` value (or array element), else the default.
pub const GGUF_M_I64: i64 = 2;

/// Value `i` of metadata key `key`: element `i` of an array, or the value itself when `i == 0`.
/// `Err(-5)` for a missing key, `Err(-4)` for an index outside the value.
pub fn meta_elem<'a>(g: &'a Gguf, key: &[u8], i: i64) -> Result<&'a MetaValue, i64> {
    let v = g.meta(key).ok_or(IO_NAME_NOT_FOUND)?;
    let at = usize::try_from(i).map_err(|_| IO_SHAPE_MISMATCH)?;
    match v {
        MetaValue::Arr(_, items) => items.get(at).ok_or(IO_SHAPE_MISMATCH),
        _ if at == 0 => Ok(v),
        _ => Err(IO_SHAPE_MISMATCH),
    }
}

/// An integer or `bool` value as an `i64` (a `u64` keeps its bits, as `x as i64` would).
pub fn meta_as_i64(v: &MetaValue) -> Option<i64> {
    Some(match *v {
        MetaValue::U8(x) => i64::from(x),
        MetaValue::I8(x) => i64::from(x),
        MetaValue::U16(x) => i64::from(x),
        MetaValue::I16(x) => i64::from(x),
        MetaValue::U32(x) => i64::from(x),
        MetaValue::I32(x) => i64::from(x),
        MetaValue::U64(x) => x as i64,
        MetaValue::I64(x) => x,
        MetaValue::Bool(b) => i64::from(b),
        _ => return None,
    })
}

/// A numeric value as an `f64`: a float exactly, an integer as `x as f64` (a `u64` by its unsigned
/// value; a `bool` is not a number).
pub fn meta_as_f64(v: &MetaValue) -> Option<f64> {
    Some(match *v {
        MetaValue::F32(x) => f64::from(x),
        MetaValue::F64(x) => x,
        MetaValue::U64(x) => x as f64,
        MetaValue::Bool(_) => return None,
        _ => meta_as_i64(v)? as f64,
    })
}

/// Answer `gguf_meta_type` / `gguf_meta_len` (`what`) about key `key`, or `gguf_meta_i64` of value
/// `i` falling back to `default`. The two queries report `-5` for a missing key; the value read
/// reports nothing but its default.
pub fn gguf_meta(g: &Gguf, key: &[u8], what: i64, i: i64, default: i64) -> i64 {
    match what {
        GGUF_M_TYPE => g
            .meta(key)
            .map_or(IO_NAME_NOT_FOUND, |v| i64::from(v.type_id())),
        GGUF_M_LEN => g.meta(key).map_or(IO_NAME_NOT_FOUND, |v| match v {
            MetaValue::Str(s) => s.len() as i64,
            MetaValue::Arr(_, items) => items.len() as i64,
            _ => 1,
        }),
        GGUF_M_I64 => meta_elem(g, key, i)
            .ok()
            .and_then(meta_as_i64)
            .unwrap_or(default),
        _ => IO_ERROR,
    }
}

/// [`gguf_meta`] on the file at `path`; a file that cannot be opened, sized or parsed answers its
/// code to the two queries and the default to the value read. Shared by both backends.
pub fn gguf_meta_path(path: &Path, key: &[u8], what: i64, i: i64, default: i64) -> i64 {
    match gguf_open(path) {
        Ok((_, g, _)) => gguf_meta(&g, key, what, i, default),
        Err(_) if what == GGUF_M_I64 => default,
        Err(code) => code,
    }
}

/// `gguf_meta_f64`: value `i` of key `key` as an `f64`, or `default` (a missing file, key or
/// index, or a value that is not a number). Shared by both backends.
pub fn gguf_meta_f64_path(path: &Path, key: &[u8], i: i64, default: f64) -> f64 {
    let Ok((_, g, _)) = gguf_open(path) else {
        return default;
    };
    meta_elem(&g, key, i)
        .ok()
        .and_then(meta_as_f64)
        .unwrap_or(default)
}

/// `gguf_meta_str`: the bytes of string value `i` of key `key` (a string, or element `i` of a
/// string array): `Err(-5)` for a missing key, `-3` for a value that is not a string, `-4` for an
/// index outside the value, or the file's own code. Shared by both backends.
pub fn gguf_meta_str_path(path: &Path, key: &[u8], i: i64) -> Result<Vec<u8>, i64> {
    let (_, g, _) = gguf_open(path)?;
    match meta_elem(&g, key, i)? {
        MetaValue::Str(s) => Ok(s.clone()),
        _ => Err(IO_DTYPE_MISMATCH),
    }
}

/// `wukong_rt_gguf_meta(path, key, what, i, default) -> i64` — `gguf_meta_type`,
/// `gguf_meta_len` and `gguf_meta_i64` ([`gguf_meta_path`]).
///
/// # Safety
/// `path` and `key` must be null or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_gguf_meta(
    path: *const u8,
    key: *const u8,
    what: i64,
    i: i64,
    default: i64,
) -> i64 {
    // SAFETY-EXEMPT: `path` and `key` are NUL-terminated strings -- they carry no extent to check.
    // SAFETY: forwarded from this function's contract.
    let (path, key) = unsafe { (crate::rt_path(path), fileio::cstr_lossy(key)) };
    match path {
        Some(path) => gguf_meta_path(&path, key.as_bytes(), what, i, default),
        None if what == GGUF_M_I64 => default,
        None => IO_OPEN_FAILED,
    }
}

/// `wukong_rt_gguf_meta_f64(path, key, i, default) -> f64` — `gguf_meta_f64`
/// ([`gguf_meta_f64_path`]).
///
/// # Safety
/// `path` and `key` must be null or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_gguf_meta_f64(
    path: *const u8,
    key: *const u8,
    i: i64,
    default: f64,
) -> f64 {
    // SAFETY-EXEMPT: `path` and `key` are NUL-terminated strings -- they carry no extent to check.
    // SAFETY: forwarded from this function's contract.
    let (path, key) = unsafe { (crate::rt_path(path), fileio::cstr_lossy(key)) };
    match path {
        Some(path) => gguf_meta_f64_path(&path, key.as_bytes(), i, default),
        None => default,
    }
}

/// Copy `s` into a `len`-byte buffer with `snprintf` semantics — at most `len - 1` bytes, then a
/// NUL; nothing for `len <= 0` — returning how many bytes were copied (the NUL not counted).
pub fn copy_c_string(s: &[u8], len: usize, mut put: impl FnMut(usize, u8)) -> usize {
    if len == 0 {
        return 0;
    }
    let n = s.len().min(len - 1);
    for (k, &b) in s[..n].iter().enumerate() {
        put(k, b);
    }
    put(n, 0);
    n
}

/// `wukong_rt_gguf_meta_str(path, key, i, buf, len) -> i64` — `gguf_meta_str`: copy string value
/// `i` of key `key` into the `len`-byte `buf` as `sprintf` does (truncated to `len - 1` bytes and
/// always NUL-terminated) and return its full byte length, or a negative code
/// ([`gguf_meta_str_path`]); `buf` is untouched on a negative code.
///
/// # Safety
/// `path` and `key` must be null or NUL-terminated; `buf` must address `len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_gguf_meta_str(
    path: *const u8,
    key: *const u8,
    i: i64,
    buf: *mut u8,
    len: i64,
) -> i64 {
    crate::safety::debug_extent(buf, len);
    // SAFETY: forwarded from this function's contract.
    let (path, key) = unsafe { (crate::rt_path(path), fileio::cstr_lossy(key)) };
    let Some(path) = path else {
        return IO_OPEN_FAILED;
    };
    match gguf_meta_str_path(&path, key.as_bytes(), i) {
        Ok(s) => {
            let len = usize::try_from(len).unwrap_or(0);
            // SAFETY: `copy_c_string` writes only at offsets below `len`, which the caller
            // guarantees are writable.
            copy_c_string(&s, len, |k, b| unsafe { buf.add(k).write(b) });
            s.len() as i64
        }
        Err(code) => code,
    }
}

// --- dequantization ----------------------------------------------------------------------------

/// The 16 non-uniform levels of `IQ4_NL` / `IQ4_XS` (ggml's `kvalues_iq4nl`).
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// The little-endian `f16` at byte `at` of `b`, widened exactly.
fn half(b: &[u8], at: usize) -> f32 {
    crate::f16_bits_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

/// The 6-bit scale and min of sub-block `j` (0–7) of a `Q4_K` / `Q5_K` block's 12 packed scale
/// bytes (ggml's `get_scale_min_k4`).
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantize one block of `kind` — `blk` is exactly [`Dequant::bytes`] bytes — into `y`, exactly
/// [`Dequant::block`] values, by ggml's reference `dequantize_row_*` (see the module docs).
pub fn dequant_block(kind: Dequant, blk: &[u8], y: &mut [f32]) {
    debug_assert_eq!(blk.len(), kind.bytes());
    debug_assert_eq!(y.len(), kind.block());
    match kind {
        Dequant::Q4_0 => {
            // { f16 d; u8 qs[16] }: element j in the low nibble of qs[j], j + 16 in the high.
            let d = half(blk, 0);
            let qs = &blk[2..18];
            for j in 0..16 {
                let x0 = i32::from(qs[j] & 0x0F) - 8;
                let x1 = i32::from(qs[j] >> 4) - 8;
                y[j] = x0 as f32 * d;
                y[j + 16] = x1 as f32 * d;
            }
        }
        Dequant::Q4_1 => {
            // { f16 d; f16 m; u8 qs[16] }: x·d + m.
            let d = half(blk, 0);
            let m = half(blk, 2);
            let qs = &blk[4..20];
            for j in 0..16 {
                y[j] = f32::from(qs[j] & 0x0F) * d + m;
                y[j + 16] = f32::from(qs[j] >> 4) * d + m;
            }
        }
        Dequant::Q5_0 | Dequant::Q5_1 => {
            // { f16 d; [f16 m;] u8 qh[4]; u8 qs[16] }: bit j of the little-endian u32 `qh` is the
            // fifth bit of element j (and bit j + 16 that of element j + 16).
            let affine = kind == Dequant::Q5_1;
            let d = half(blk, 0);
            let (m, h) = if affine { (half(blk, 2), 4) } else { (0.0, 2) };
            let qh = u32::from_le_bytes([blk[h], blk[h + 1], blk[h + 2], blk[h + 3]]);
            let qs = &blk[h + 4..h + 20];
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                let x0 = (u32::from(qs[j] & 0x0F) | xh0) as i32;
                let x1 = (u32::from(qs[j] >> 4) | xh1) as i32;
                if affine {
                    y[j] = x0 as f32 * d + m;
                    y[j + 16] = x1 as f32 * d + m;
                } else {
                    y[j] = (x0 - 16) as f32 * d;
                    y[j + 16] = (x1 - 16) as f32 * d;
                }
            }
        }
        Dequant::Q8_0 => {
            // { f16 d; i8 qs[32] }.
            let d = half(blk, 0);
            for j in 0..32 {
                y[j] = f32::from(blk[2 + j] as i8) * d;
            }
        }
        Dequant::Q2_K => {
            // { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin }: each scale byte is a 4-bit scale (low)
            // and a 4-bit min (high) of one 16-element group; a quant byte holds four 2-bit values.
            let scales = &blk[0..16];
            let qs = &blk[16..80];
            let d = half(blk, 80);
            let dmin = half(blk, 82);
            let (mut yi, mut is) = (0usize, 0usize);
            for n in 0..2 {
                let q = &qs[n * 32..n * 32 + 32];
                for shift in [0u32, 2, 4, 6] {
                    for half16 in 0..2 {
                        let sc = scales[is];
                        is += 1;
                        let dl = d * f32::from(sc & 0xF);
                        let ml = dmin * f32::from(sc >> 4);
                        for l in 0..16 {
                            let v = (q[l + 16 * half16] >> shift) & 3;
                            y[yi] = dl * f32::from(v) - ml;
                            yi += 1;
                        }
                    }
                }
            }
        }
        Dequant::Q3_K => {
            // { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d }: a 2-bit low part in `qs`, the
            // third bit in `hmask` (a CLEAR bit subtracts 4), and sixteen 6-bit scales biased by
            // 32.
            let hm = &blk[0..32];
            let qs = &blk[32..96];
            let sr = &blk[96..108];
            let d_all = half(blk, 108);
            let mut scales = [0i32; 16];
            for (k, s) in scales.iter_mut().enumerate() {
                let low = if k < 8 { sr[k] & 0xF } else { sr[k - 8] >> 4 };
                let high = (sr[8 + k % 4] >> (2 * (k / 4))) & 3;
                *s = i32::from(low | (high << 4)) - 32;
            }
            let (mut yi, mut is) = (0usize, 0usize);
            let mut m = 1u8;
            for n in 0..2 {
                let q = &qs[n * 32..n * 32 + 32];
                for shift in [0u32, 2, 4, 6] {
                    for half16 in 0..2 {
                        let dl = d_all * scales[is] as f32;
                        is += 1;
                        for l in 0..16 {
                            let at = l + 16 * half16;
                            let lo = i32::from((q[at] >> shift) & 3);
                            let v = lo - if hm[at] & m != 0 { 0 } else { 4 };
                            y[yi] = dl * v as f32;
                            yi += 1;
                        }
                    }
                    m = m.wrapping_shl(1);
                }
            }
        }
        Dequant::Q4_K | Dequant::Q5_K => {
            // Q4_K { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] };
            // Q5_K { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128] }: eight 32-element
            // sub-blocks, each with a 6-bit scale and min; Q5_K's fifth bits sit in `qh`, bit 2c
            // (low nibbles) and 2c + 1 (high nibbles) of byte l for 64-element chunk c.
            let five = kind == Dequant::Q5_K;
            let d = half(blk, 0);
            let dmin = half(blk, 2);
            let scales = &blk[4..16];
            let (qh, qs) = if five {
                (&blk[16..48], &blk[48..176])
            } else {
                (&blk[0..0], &blk[16..144])
            };
            let mut yi = 0usize;
            for c in 0..4 {
                let q = &qs[c * 32..c * 32 + 32];
                let (sc1, m1) = scale_min_k4(2 * c, scales);
                let (sc2, m2) = scale_min_k4(2 * c + 1, scales);
                let (d1, m1) = (d * f32::from(sc1), dmin * f32::from(m1));
                let (d2, m2) = (d * f32::from(sc2), dmin * f32::from(m2));
                let (u1, u2) = (1u8 << (2 * c), 2u8 << (2 * c));
                for l in 0..32 {
                    let hi = if five && qh[l] & u1 != 0 { 16 } else { 0 };
                    y[yi] = d1 * f32::from((q[l] & 0xF) + hi) - m1;
                    yi += 1;
                }
                for l in 0..32 {
                    let hi = if five && qh[l] & u2 != 0 { 16 } else { 0 };
                    y[yi] = d2 * f32::from((q[l] >> 4) + hi) - m2;
                    yi += 1;
                }
            }
        }
        Dequant::Q6_K => {
            // { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d }: a 6-bit value is 4 low bits from
            // `ql` and 2 high bits from `qh`, biased by 32, with a signed scale per 16 elements.
            let d = half(blk, 208);
            for n in 0..2 {
                let ql = &blk[n * 64..n * 64 + 64];
                let qh = &blk[128 + n * 32..128 + n * 32 + 32];
                let sc = &blk[192 + n * 8..192 + n * 8 + 8];
                let y = &mut y[n * 128..n * 128 + 128];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = i32::from((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                    let q2 = i32::from((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    let s = |k: usize| d * f32::from(sc[k] as i8);
                    y[l] = s(is) * q1 as f32;
                    y[l + 32] = s(is + 2) * q2 as f32;
                    y[l + 64] = s(is + 4) * q3 as f32;
                    y[l + 96] = s(is + 6) * q4 as f32;
                }
            }
        }
        Dequant::Q8_K => {
            // { f32 d; i8 qs[256]; i16 bsums[16] } (the sums only feed ggml's dot products).
            let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
            for j in 0..256 {
                y[j] = d * f32::from(blk[4 + j] as i8);
            }
        }
        Dequant::IQ4_NL => {
            // { f16 d; u8 qs[16] }: each nibble indexes the non-uniform `KVALUES_IQ4NL` levels.
            let d = half(blk, 0);
            let qs = &blk[2..18];
            for j in 0..16 {
                y[j] = d * f32::from(KVALUES_IQ4NL[usize::from(qs[j] & 0xF)]);
                y[j + 16] = d * f32::from(KVALUES_IQ4NL[usize::from(qs[j] >> 4)]);
            }
        }
        Dequant::IQ4_XS => {
            // { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128] }: eight 32-element sub-blocks,
            // each with a 6-bit scale (4 low bits from `scales_l`, 2 high from `scales_h`) biased
            // by 32, over the `IQ4_NL` levels.
            let d = half(blk, 0);
            let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
            let scales_l = &blk[4..8];
            let qs = &blk[8..136];
            for ib in 0..8 {
                let lo = i32::from((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF);
                let hi = i32::from((scales_h >> (2 * ib)) & 3);
                let dl = d * ((lo | (hi << 4)) - 32) as f32;
                let q = &qs[ib * 16..ib * 16 + 16];
                for j in 0..16 {
                    y[ib * 32 + j] = dl * f32::from(KVALUES_IQ4NL[usize::from(q[j] & 0xF)]);
                    y[ib * 32 + j + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(q[j] >> 4)]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Every file is assembled byte by byte here, every quantized block is packed by hand — by a
    //! packer transcribed from ggml's *quantizer* side (`quantize_row_*_ref`), the inverse of the
    //! code under test — and every expected value is the format's definition evaluated on the
    //! chosen scales and quants, picked so that each `f32` product and sum is exact.

    use super::*;
    use crate::fileio::Scratch;

    /// A `u64`-length-prefixed GGUF string.
    fn gstr(s: &[u8]) -> Vec<u8> {
        let mut out = (s.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(s);
        out
    }

    /// Encode a metadata value's payload (no type id), independently of the parser.
    fn enc(v: &MetaValue, out: &mut Vec<u8>) {
        match v {
            MetaValue::U8(x) => out.push(*x),
            MetaValue::I8(x) => out.push(*x as u8),
            MetaValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::Bool(b) => out.push(u8::from(*b)),
            MetaValue::Str(s) => out.extend_from_slice(&gstr(s)),
            MetaValue::Arr(et, items) => {
                out.extend_from_slice(&et.to_le_bytes());
                out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for it in items {
                    enc(it, out);
                }
            }
            MetaValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        }
    }

    /// A GGUF file under construction.
    struct Gb {
        magic: [u8; 4],
        version: u32,
        kv: Vec<u8>,
        n_kv: u64,
        ti: Vec<u8>,
        n_t: u64,
        data: Vec<u8>,
        align: usize,
    }

    impl Gb {
        fn new() -> Gb {
            Gb {
                magic: GGUF_MAGIC,
                version: 3,
                kv: Vec::new(),
                n_kv: 0,
                ti: Vec::new(),
                n_t: 0,
                data: Vec::new(),
                align: 32,
            }
        }

        fn kv_raw(&mut self, key: &[u8], ty: u32, payload: &[u8]) -> &mut Gb {
            self.kv.extend_from_slice(&gstr(key));
            self.kv.extend_from_slice(&ty.to_le_bytes());
            self.kv.extend_from_slice(payload);
            self.n_kv += 1;
            self
        }

        fn kv(&mut self, key: &str, v: &MetaValue) -> &mut Gb {
            let mut b = Vec::new();
            enc(v, &mut b);
            self.kv_raw(key.as_bytes(), v.type_id(), &b)
        }

        /// Declare a tensor at an explicit data-section offset (no payload is added).
        fn tensor_at(&mut self, name: &str, ne: &[u64], ty: u32, off: u64) -> &mut Gb {
            self.ti.extend_from_slice(&gstr(name.as_bytes()));
            self.ti.extend_from_slice(&(ne.len() as u32).to_le_bytes());
            for d in ne {
                self.ti.extend_from_slice(&d.to_le_bytes());
            }
            self.ti.extend_from_slice(&ty.to_le_bytes());
            self.ti.extend_from_slice(&off.to_le_bytes());
            self.n_t += 1;
            self
        }

        /// Append `payload` at the next aligned data offset and declare it.
        fn tensor(&mut self, name: &str, ne: &[u64], ty: u32, payload: &[u8]) -> &mut Gb {
            let off = self.data.len().next_multiple_of(self.align);
            self.data.resize(off, 0);
            self.data.extend_from_slice(payload);
            self.tensor_at(name, ne, ty, off as u64)
        }

        /// The header (magic through the tensor table), unpadded.
        fn header(&self) -> Vec<u8> {
            let mut out = self.magic.to_vec();
            out.extend_from_slice(&self.version.to_le_bytes());
            out.extend_from_slice(&self.n_t.to_le_bytes());
            out.extend_from_slice(&self.n_kv.to_le_bytes());
            out.extend_from_slice(&self.kv);
            out.extend_from_slice(&self.ti);
            out
        }

        fn bytes(&self) -> Vec<u8> {
            let mut out = self.header();
            out.resize(out.len().next_multiple_of(self.align), 0);
            out.extend_from_slice(&self.data);
            out
        }
    }

    fn parse(bytes: &[u8]) -> Result<Gguf, i64> {
        gguf_parse(bytes, bytes.len() as u64)
    }

    fn f16b(x: f32) -> [u8; 2] {
        crate::f32_to_f16_bits(x).to_le_bytes()
    }

    // --- the header ------------------------------------------------------------------------------

    /// One value of every type, arrays of scalars, strings and arrays (and an empty one) included.
    fn every_value() -> Vec<(&'static str, MetaValue)> {
        use MetaValue as V;
        vec![
            ("u8", V::U8(200)),
            ("i8", V::I8(-100)),
            ("u16", V::U16(65_000)),
            ("i16", V::I16(-30_000)),
            ("u32", V::U32(4_000_000_000)),
            ("i32", V::I32(-2_000_000_000)),
            ("f32", V::F32(-1.5e-3)),
            ("bool.t", V::Bool(true)),
            ("bool.f", V::Bool(false)),
            ("str", V::Str("llama \u{e9}\u{1F600}".as_bytes().to_vec())),
            ("str.empty", V::Str(Vec::new())),
            ("u64", V::U64(u64::MAX)),
            ("i64", V::I64(i64::MIN)),
            ("f64", V::F64(std::f64::consts::PI)),
            (
                "arr.i32",
                V::Arr(vt::I32, vec![V::I32(1), V::I32(-2), V::I32(3)]),
            ),
            (
                "arr.str",
                V::Arr(
                    vt::STRING,
                    vec![
                        V::Str(b"<s>".to_vec()),
                        V::Str(Vec::new()),
                        V::Str(b"x".to_vec()),
                    ],
                ),
            ),
            ("arr.empty", V::Arr(vt::F64, Vec::new())),
            (
                "arr.nested",
                V::Arr(
                    vt::ARRAY,
                    vec![
                        V::Arr(vt::U8, vec![V::U8(1), V::U8(2)]),
                        V::Arr(vt::BOOL, vec![V::Bool(false)]),
                    ],
                ),
            ),
            (
                "arr.mixed_widths",
                V::Arr(vt::U16, vec![V::U16(0), V::U16(u16::MAX)]),
            ),
            ("arr.f32", V::Arr(vt::F32, vec![V::F32(0.25), V::F32(-0.0)])),
            ("arr.u64", V::Arr(vt::U64, vec![V::U64(7)])),
            ("arr.i8", V::Arr(vt::I8, vec![V::I8(-1)])),
            ("arr.i16", V::Arr(vt::I16, vec![V::I16(-2)])),
            ("arr.u32", V::Arr(vt::U32, vec![V::U32(3)])),
            ("arr.i64", V::Arr(vt::I64, vec![V::I64(-4)])),
            ("arr.f64", V::Arr(vt::F64, vec![V::F64(0.5)])),
        ]
    }

    #[test]
    fn metadata_round_trips_every_value_type() {
        for version in [2, 3] {
            let mut b = Gb::new();
            b.version = version;
            for (k, v) in every_value() {
                b.kv(k, &v);
            }
            let bytes = b.bytes();
            let g = parse(&bytes).unwrap();
            assert_eq!(g.version, version);
            let want: Vec<(Vec<u8>, MetaValue)> = every_value()
                .into_iter()
                .map(|(k, v)| (k.as_bytes().to_vec(), v))
                .collect();
            assert_eq!(g.metadata, want);
            // Every one of the thirteen type ids occurs at top level.
            let mut ids: Vec<u32> = g.metadata.iter().map(|(_, v)| v.type_id()).collect();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids, (0..=12).collect::<Vec<u32>>());
            assert_eq!(g.alignment, GGUF_DEFAULT_ALIGNMENT);
            assert!(g.tensors.is_empty());
            assert_eq!(g.data_start, (bytes.len() as u64).next_multiple_of(32));
            assert_eq!(
                g.meta(b"arr.str"),
                Some(&MetaValue::Arr(
                    vt::STRING,
                    vec![
                        MetaValue::Str(b"<s>".to_vec()),
                        MetaValue::Str(Vec::new()),
                        MetaValue::Str(b"x".to_vec())
                    ]
                ))
            );
        }
        // A bool byte other than 0/1 reads as true, as llama.cpp reads it.
        let mut b = Gb::new();
        b.kv_raw(b"b", vt::BOOL, &[7]);
        assert_eq!(
            parse(&b.bytes()).unwrap().metadata[0].1,
            MetaValue::Bool(true)
        );
    }

    #[test]
    fn tensor_table_alignment_and_data_start() {
        let mut b = Gb::new();
        b.align = 64;
        b.kv("general.alignment", &MetaValue::U32(64))
            .kv("general.name", &MetaValue::Str(b"tiny".to_vec()))
            .tensor("a", &[3, 2], 0, &[0u8; 24])
            .tensor("b", &[32], 8, &[0u8; 34]);
        let bytes = b.bytes();
        let g = parse(&bytes).unwrap();
        assert_eq!(g.alignment, 64);
        assert_eq!(g.data_start % 64, 0);
        assert_eq!(g.data_start, (b.header().len() as u64).next_multiple_of(64));
        assert_eq!(
            g.tensors,
            vec![
                GgufTensorInfo {
                    name: b"a".to_vec(),
                    ne: vec![3, 2],
                    ggml_type: 0,
                    offset: 0
                },
                GgufTensorInfo {
                    name: b"b".to_vec(),
                    ne: vec![32],
                    ggml_type: 8,
                    offset: 64
                },
            ]
        );
        let a = gguf_locate(&g, b"a", bytes.len() as u64).unwrap();
        assert_eq!(
            a.shape,
            vec![2, 3],
            "outermost first: ne [3, 2] is a [2, 3] tensor"
        );
        assert_eq!((a.data_off, a.nbytes), (g.data_start, 24));
        let q = gguf_locate(&g, b"b", bytes.len() as u64).unwrap();
        assert_eq!(
            (q.ty.name, q.data_off, q.nbytes),
            ("Q8_0", g.data_start + 64, 34)
        );
        assert_eq!(q.storage_shape(), vec![34]);
        assert_eq!(gguf_query(&g, b"b", GGUF_Q_FIND, 0), 1);
        assert_eq!(gguf_query(&g, b"a", GGUF_Q_RANK, 0), 2);
        assert_eq!(gguf_query(&g, b"a", GGUF_Q_DIM, 0), 2);
        assert_eq!(gguf_query(&g, b"a", GGUF_Q_DIM, 1), 3);
        assert_eq!(gguf_query(&g, b"a", GGUF_Q_DIM, 2), IO_SHAPE_MISMATCH);
        assert_eq!(gguf_query(&g, b"a", GGUF_Q_DIM, -1), IO_SHAPE_MISMATCH);
        assert_eq!(gguf_query(&g, b"b", GGUF_Q_TYPE, 0), 8);
        assert_eq!(gguf_query(&g, b"c", GGUF_Q_RANK, 0), IO_NAME_NOT_FOUND);
        assert_eq!(gguf_query(&g, b"a", 9, 0), IO_ERROR);
    }

    #[test]
    fn malformed_headers_are_rejected() {
        let good = || {
            let mut b = Gb::new();
            b.kv("k", &MetaValue::U32(1))
                .tensor("t", &[4], 0, &[0u8; 16]);
            b
        };
        assert!(parse(&good().bytes()).is_ok());
        let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
        let mut b = good();
        b.magic = *b"GGUG";
        cases.push(("bad magic", b.bytes()));
        for v in [0u32, 1, 4, 0x0300_0000] {
            let mut b = good();
            b.version = v;
            cases.push(("version", b.bytes()));
        }
        let mut b = good();
        b.kv_raw(b"x", 13, &[0; 8]);
        cases.push(("unknown value type", b.bytes()));
        let mut b = good();
        b.kv_raw(
            b"x",
            vt::ARRAY,
            &[&99u32.to_le_bytes()[..], &0u64.to_le_bytes()].concat(),
        );
        cases.push(("unknown array element type", b.bytes()));
        let mut b = good();
        b.kv_raw(
            b"x",
            vt::ARRAY,
            &[&vt::U8.to_le_bytes()[..], &(1u64 << 40).to_le_bytes()].concat(),
        );
        cases.push(("array count past the file", b.bytes()));
        let mut b = good();
        b.kv("k", &MetaValue::U8(2));
        cases.push(("duplicate key", b.bytes()));
        let mut b = good();
        b.tensor("t", &[4], 0, &[0u8; 16]);
        cases.push(("duplicate tensor", b.bytes()));
        let mut b = good();
        b.tensor_at("five", &[1, 1, 1, 1, 1], 0, 0);
        cases.push(("five dims", b.bytes()));
        let mut b = good();
        b.tensor_at("huge", &[1u64 << 63], 0, 0);
        cases.push(("dim above i64::MAX", b.bytes()));
        for (what, v) in [
            ("alignment of another type", MetaValue::U64(32)),
            ("zero alignment", MetaValue::U32(0)),
            ("alignment not a power of two", MetaValue::U32(48)),
        ] {
            let mut b = good();
            b.kv("general.alignment", &v);
            cases.push((what, b.bytes()));
        }
        let mut b = Gb::new();
        b.kv_raw(b"s", vt::STRING, &(1u64 << 62).to_le_bytes());
        cases.push(("string length past the file", b.bytes()));
        let mut nested = Vec::new();
        for _ in 0..65 {
            nested.extend_from_slice(&vt::ARRAY.to_le_bytes());
            nested.extend_from_slice(&1u64.to_le_bytes());
        }
        nested.extend_from_slice(&vt::U8.to_le_bytes());
        nested.extend_from_slice(&0u64.to_le_bytes());
        let mut b = Gb::new();
        b.kv_raw(b"deep", vt::ARRAY, &nested);
        cases.push(("arrays nested past the limit", b.bytes()));
        for (what, bytes) in cases {
            assert_eq!(parse(&bytes), Err(IO_BAD_HEADER), "{what}");
        }
        // A file cut anywhere inside its header is malformed, never a short read.
        let full = good().header();
        for cut in 0..full.len() {
            assert_eq!(parse(&full[..cut]), Err(IO_BAD_HEADER), "cut at {cut}");
        }
    }

    #[test]
    fn locate_checks_the_named_tensor_only() {
        let mut b = Gb::new();
        b.tensor("ok", &[32, 2], 8, &[0u8; 68])
            .tensor_at("odd_row", &[48], 8, 128)
            .tensor_at("misaligned", &[4], 0, 8)
            .tensor_at("past_end", &[4], 0, 4096)
            .tensor_at("unknown", &[4], 4, 0)
            .tensor_at("retired", &[4], 33, 0)
            .tensor_at("future", &[4], 1000, 0);
        let bytes = b.bytes();
        let g = parse(&bytes).unwrap();
        let n = bytes.len() as u64;
        let ok = gguf_locate(&g, b"ok", n).unwrap();
        assert_eq!((ok.shape.clone(), ok.nbytes), (vec![2, 32], 68));
        assert_eq!(ok.storage_shape(), vec![2, 34]);
        assert_eq!(gguf_locate(&g, b"odd_row", n), Err(IO_BAD_HEADER));
        assert_eq!(gguf_locate(&g, b"misaligned", n), Err(IO_BAD_HEADER));
        assert_eq!(gguf_locate(&g, b"past_end", n), Err(IO_BAD_HEADER));
        for t in ["unknown", "retired", "future"] {
            assert_eq!(
                gguf_locate(&g, t.as_bytes(), n),
                Err(IO_UNSUPPORTED_TYPE),
                "{t}"
            );
            // The queries still answer: the shape and the raw type id are in the table.
            assert_eq!(gguf_query(&g, t.as_bytes(), GGUF_Q_DIM, 0), 4);
        }
        assert_eq!(gguf_query(&g, b"future", GGUF_Q_TYPE, 0), 1000);
        assert_eq!(gguf_locate(&g, b"absent", n), Err(IO_NAME_NOT_FOUND));
        // The payload must lie inside the file: one byte short.
        let g2 = parse(&bytes[..bytes.len() - 1]).unwrap();
        assert_eq!(gguf_locate(&g2, b"ok", n - 1), Err(IO_BAD_HEADER));
    }

    #[test]
    fn type_table_matches_ggml() {
        // (id, name, block, bytes) for every id ggml assigns and a file can hold.
        let want = [
            (0, "F32", 1, 4),
            (1, "F16", 1, 2),
            (2, "Q4_0", 32, 18),
            (3, "Q4_1", 32, 20),
            (6, "Q5_0", 32, 22),
            (7, "Q5_1", 32, 24),
            (8, "Q8_0", 32, 34),
            (9, "Q8_1", 32, 36),
            (10, "Q2_K", 256, 84),
            (11, "Q3_K", 256, 110),
            (12, "Q4_K", 256, 144),
            (13, "Q5_K", 256, 176),
            (14, "Q6_K", 256, 210),
            (15, "Q8_K", 256, 292),
            (16, "IQ2_XXS", 256, 66),
            (17, "IQ2_XS", 256, 74),
            (18, "IQ3_XXS", 256, 98),
            (19, "IQ1_S", 256, 50),
            (20, "IQ4_NL", 32, 18),
            (21, "IQ3_S", 256, 110),
            (22, "IQ2_S", 256, 82),
            (23, "IQ4_XS", 256, 136),
            (24, "I8", 1, 1),
            (25, "I16", 1, 2),
            (26, "I32", 1, 4),
            (27, "I64", 1, 8),
            (28, "F64", 1, 8),
            (29, "IQ1_M", 256, 56),
            (30, "BF16", 1, 2),
            (34, "TQ1_0", 256, 54),
            (35, "TQ2_0", 256, 66),
            (39, "MXFP4", 32, 17),
        ];
        let mut known = 0;
        for id in 0..64u32 {
            let got = ggml_type(id).map(|t| (t.id, t.name, t.block, t.bytes));
            let exp = want.iter().find(|w| w.0 == id).copied();
            assert_eq!(got, exp, "id {id}");
            if let Some(t) = ggml_type(id) {
                known += 1;
                if let Some(dq) = t.dequant {
                    assert_eq!(
                        (dq.block() as u64, dq.bytes() as u64),
                        (t.block, t.bytes),
                        "{}",
                        t.name
                    );
                }
                assert_eq!(t.dt.is_some(), t.block == 1, "{}", t.name);
            }
        }
        assert_eq!(known, want.len());
    }

    // --- dequantization ---------------------------------------------------------------------

    fn dq(kind: Dequant, blk: &[u8]) -> Vec<f32> {
        let mut y = vec![f32::NAN; kind.block()];
        dequant_block(kind, blk, &mut y);
        y
    }

    /// Compare lane by lane, bit for bit (every expected value here is exact in `f32`).
    fn assert_lanes(got: &[f32], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                (w as f32).to_bits(),
                "{what} lane {i}: {g} vs {w}"
            );
        }
    }

    #[test]
    fn q8_0_and_q4_0_blocks() {
        // Q8_0: d = -0.5, q_j = 5j - 77 (covers both signs; -128 and 127 at the ends).
        let mut blk = f16b(-0.5).to_vec();
        let mut q: Vec<i32> = (0..32).map(|j| 5 * j - 77).collect();
        q[0] = -128;
        q[31] = 127;
        blk.extend(q.iter().map(|&v| v as i8 as u8));
        let want: Vec<f64> = q.iter().map(|&v| f64::from(v) * -0.5).collect();
        assert_lanes(&dq(Dequant::Q8_0, &blk), &want, "Q8_0");

        // Q4_0: d = 2, low nibble of byte j = j % 16, high nibble = 15 - j; value (n - 8)·d.
        let mut blk = f16b(2.0).to_vec();
        for j in 0..16u8 {
            blk.push((j % 16) | ((15 - j) << 4));
        }
        let want: Vec<f64> = (0..32)
            .map(|e| {
                let n = if e < 16 { e } else { 15 - (e - 16) };
                (f64::from(n) - 8.0) * 2.0
            })
            .collect();
        assert_lanes(&dq(Dequant::Q4_0, &blk), &want, "Q4_0");
    }

    /// The existing GGML packers (`quant.rs`, byte-compatible with llama.cpp) and this dequantizer
    /// agree on the layout: every value comes back within the format's rounding step.
    #[test]
    fn q8_0_and_q4_0_dequantize_what_the_runtime_packers_wrote() {
        let x: Vec<f32> = (0..64)
            .map(|i| ((i * 37 % 64) as f32 - 31.0) * 0.173)
            .collect();
        for (kind, bb) in [(Dequant::Q8_0, 34usize), (Dequant::Q4_0, 18)] {
            let mut packed = vec![0u8; 2 * bb];
            match kind {
                Dequant::Q8_0 => crate::quant::quantize_row_q8_0(&x, &mut packed),
                _ => crate::quant::quantize_row_q4_0(&x, &mut packed),
            }
            for (b, blk) in packed.chunks_exact(bb).enumerate() {
                let d = half(blk, 0).abs();
                let y = dq(kind, blk);
                for (j, &v) in y.iter().enumerate() {
                    let xv = x[b * 32 + j];
                    // Half a step, plus the scale's own rounding to f16 (the packers quantize with
                    // the unrounded scale, as llama.cpp's do), plus one step where Q4_0 clamps.
                    let tol = if kind == Dequant::Q8_0 { 0.6 } else { 1.1 };
                    assert!(
                        (v - xv).abs() <= d * tol,
                        "{kind:?} block {b} lane {j}: {v} vs {xv} (d = {d})"
                    );
                }
            }
        }
    }

    #[test]
    fn q4_1_q5_0_q5_1_blocks() {
        // Q4_1: d = 0.25, m = -1.5; nibbles as in the Q4_0 test.
        let mut blk = [f16b(0.25), f16b(-1.5)].concat();
        for j in 0..16u8 {
            blk.push(j | ((15 - j) << 4));
        }
        let want: Vec<f64> = (0..32)
            .map(|e| {
                let n = if e < 16 { e } else { 31 - e };
                f64::from(n) * 0.25 - 1.5
            })
            .collect();
        assert_lanes(&dq(Dequant::Q4_1, &blk), &want, "Q4_1");

        // Q5_0 / Q5_1: 5-bit values v_e = (7e + 3) % 32; the packer of `quantize_row_q5_*_ref`
        // puts the fifth bit of element j at bit j of qh, that of element j + 16 at bit j + 16.
        let v: Vec<u32> = (0..32).map(|e| (7 * e + 3) % 32).collect();
        let mut qh = 0u32;
        let mut qs = Vec::new();
        for j in 0..16 {
            qs.push(((v[j] & 0xF) | ((v[j + 16] & 0xF) << 4)) as u8);
            qh |= ((v[j] & 0x10) >> 4) << j;
            qh |= ((v[j + 16] & 0x10) >> 4) << (j + 16);
        }
        let blk = [&f16b(1.5)[..], &qh.to_le_bytes(), &qs].concat();
        let want: Vec<f64> = v.iter().map(|&x| (f64::from(x) - 16.0) * 1.5).collect();
        assert_lanes(&dq(Dequant::Q5_0, &blk), &want, "Q5_0");
        let blk = [&f16b(0.5)[..], &f16b(-4.0), &qh.to_le_bytes(), &qs].concat();
        let want: Vec<f64> = v.iter().map(|&x| f64::from(x) * 0.5 - 4.0).collect();
        assert_lanes(&dq(Dequant::Q5_1, &blk), &want, "Q5_1");
    }

    #[test]
    fn q2_k_block() {
        // 16 groups of 16: scale nibble s_g = (5g + 1) % 16, min nibble m_g = (3g + 7) % 16; 2-bit
        // quants q_e = (e·7 + e/16) % 4. Packed as `quantize_row_q2_K_ref` packs:
        // scales[g] = s | m << 4, and qs[n·32 + l] = L[n·128 + l] | L[+32] << 2 | L[+64] << 4 |
        // L[+96] << 6.
        let s: Vec<u8> = (0..16).map(|g| ((5 * g + 1) % 16) as u8).collect();
        let m: Vec<u8> = (0..16).map(|g| ((3 * g + 7) % 16) as u8).collect();
        let q: Vec<u8> = (0..256).map(|e| ((e * 7 + e / 16) % 4) as u8).collect();
        let mut blk: Vec<u8> = (0..16).map(|g| s[g] | (m[g] << 4)).collect();
        for n in 0..2 {
            for l in 0..32 {
                let b = n * 128 + l;
                blk.push(q[b] | (q[b + 32] << 2) | (q[b + 64] << 4) | (q[b + 96] << 6));
            }
        }
        blk.extend_from_slice(&f16b(0.5));
        blk.extend_from_slice(&f16b(0.25));
        let want: Vec<f64> = (0..256)
            .map(|e| 0.5 * f64::from(s[e / 16]) * f64::from(q[e]) - 0.25 * f64::from(m[e / 16]))
            .collect();
        assert_lanes(&dq(Dequant::Q2_K, &blk), &want, "Q2_K");
    }

    #[test]
    fn q3_k_block() {
        // Sixteen 6-bit scales L_g = (13g + 5) % 64 (signed scale L_g − 32), 3-bit values
        // L_e = (3e + e/7) % 8 (signed value L_e − 4). Packed as `quantize_row_q3_K_ref` packs:
        // the scale nibbles into bytes 0–7 (g < 8 low, g ≥ 8 high), their top two bits into byte
        // 8 + g%4 at bit 2·(g/4); the value's bit 2 into hmask[e % 32] bit e/32; its low two bits
        // into qs[n·32 + l] at 2·quarter.
        let ls: Vec<u8> = (0..16).map(|g| ((13 * g + 5) % 64) as u8).collect();
        let lv: Vec<u8> = (0..256).map(|e| ((3 * e + e / 7) % 8) as u8).collect();
        let mut scales = [0u8; 12];
        for (g, &l) in ls.iter().enumerate() {
            if g < 8 {
                scales[g] = l & 0xF;
            } else {
                scales[g - 8] |= (l & 0xF) << 4;
            }
            scales[g % 4 + 8] |= (l >> 4) << (2 * (g / 4));
        }
        let mut hmask = [0u8; 32];
        for (e, &l) in lv.iter().enumerate() {
            if l > 3 {
                hmask[e % 32] |= 1 << (e / 32);
            }
        }
        let mut qs = [0u8; 64];
        for n in 0..2 {
            for l in 0..32 {
                let b = n * 128 + l;
                qs[n * 32 + l] = (lv[b] & 3)
                    | ((lv[b + 32] & 3) << 2)
                    | ((lv[b + 64] & 3) << 4)
                    | ((lv[b + 96] & 3) << 6);
            }
        }
        let blk = [&hmask[..], &qs, &scales, &f16b(0.25)].concat();
        let want: Vec<f64> = (0..256)
            .map(|e| 0.25 * (f64::from(ls[e / 16]) - 32.0) * (f64::from(lv[e]) - 4.0))
            .collect();
        assert_lanes(&dq(Dequant::Q3_K, &blk), &want, "Q3_K");
    }

    /// Pack eight 6-bit (scale, min) pairs the way `quantize_row_q4_K_ref` / `_q5_K_ref` do.
    fn pack_k4_scales(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
        let mut s = [0u8; 12];
        for j in 0..8 {
            if j < 4 {
                s[j] = sc[j];
                s[j + 4] = mn[j];
            } else {
                s[j + 4] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4);
                s[j - 4] |= (sc[j] >> 4) << 6;
                s[j] |= (mn[j] >> 4) << 6;
            }
        }
        s
    }

    /// Sub-block scales and mins that exercise all six bits of both, in every slot.
    const K4_SC: [u8; 8] = [63, 1, 34, 17, 50, 33, 7, 62];
    const K4_MN: [u8; 8] = [0, 63, 5, 40, 63, 31, 48, 2];

    #[test]
    fn q4_k_and_q5_k_blocks() {
        let scales = pack_k4_scales(&K4_SC, &K4_MN);
        let (d, dmin) = (0.5f64, 0.125f64);
        // Q4_K: 4-bit values L_e = (5e + e/32) % 16;
        // qs[c·32 + l] = L[c·64 + l] | L[c·64 + l + 32] << 4.
        let l4: Vec<u8> = (0..256).map(|e| ((5 * e + e / 32) % 16) as u8).collect();
        let mut qs = [0u8; 128];
        for c in 0..4 {
            for l in 0..32 {
                qs[c * 32 + l] = l4[c * 64 + l] | (l4[c * 64 + l + 32] << 4);
            }
        }
        let blk = [&f16b(0.5)[..], &f16b(0.125), &scales, &qs].concat();
        let want: Vec<f64> = (0..256)
            .map(|e| {
                d * f64::from(K4_SC[e / 32]) * f64::from(l4[e]) - dmin * f64::from(K4_MN[e / 32])
            })
            .collect();
        assert_lanes(&dq(Dequant::Q4_K, &blk), &want, "Q4_K");

        // Q5_K: 5-bit values L_e = (11e + e/64) % 32; the low nibbles as Q4_K, the fifth bit of
        // L[c·64 + j] at qh[j] bit 2c and of L[c·64 + j + 32] at bit 2c + 1.
        let l5: Vec<u8> = (0..256).map(|e| ((11 * e + e / 64) % 32) as u8).collect();
        let mut qh = [0u8; 32];
        let mut ql = [0u8; 128];
        let (mut m1, mut m2) = (1u8, 2u8);
        for c in 0..4 {
            for j in 0..32 {
                let (mut a, mut b) = (l5[c * 64 + j], l5[c * 64 + j + 32]);
                if a > 15 {
                    a -= 16;
                    qh[j] |= m1;
                }
                if b > 15 {
                    b -= 16;
                    qh[j] |= m2;
                }
                ql[c * 32 + j] = a | (b << 4);
            }
            m1 = m1.wrapping_shl(2);
            m2 = m2.wrapping_shl(2);
        }
        let blk = [&f16b(0.5)[..], &f16b(0.125), &scales, &qh, &ql].concat();
        let want: Vec<f64> = (0..256)
            .map(|e| {
                d * f64::from(K4_SC[e / 32]) * f64::from(l5[e]) - dmin * f64::from(K4_MN[e / 32])
            })
            .collect();
        assert_lanes(&dq(Dequant::Q5_K, &blk), &want, "Q5_K");
    }

    #[test]
    fn q6_k_block() {
        // Signed scales per 16 elements (every sign, the int8 extremes), 6-bit values
        // L_e = (9e + e/5) % 64 (signed L_e − 32), packed as `quantize_row_q6_K_ref` packs.
        let sc: Vec<i8> = (0..16)
            .map(|g| match g {
                0 => -128,
                1 => 127,
                _ => (g as i8 - 8) * 9,
            })
            .collect();
        let lq: Vec<u8> = (0..256).map(|e| ((9 * e + e / 5) % 64) as u8).collect();
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for n in 0..2 {
            let j = n * 128;
            for l in 0..32 {
                let (q1, q2, q3, q4) = (lq[j + l], lq[j + l + 32], lq[j + l + 64], lq[j + l + 96]);
                ql[n * 64 + l] = (q1 & 0xF) | ((q3 & 0xF) << 4);
                ql[n * 64 + l + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);
                qh[n * 32 + l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
            }
        }
        let scb: Vec<u8> = sc.iter().map(|&s| s as u8).collect();
        let blk = [&ql[..], &qh, &scb, &f16b(0.0625)].concat();
        let want: Vec<f64> = (0..256)
            .map(|e| 0.0625 * f64::from(sc[e / 16]) * (f64::from(lq[e]) - 32.0))
            .collect();
        assert_lanes(&dq(Dequant::Q6_K, &blk), &want, "Q6_K");
    }

    #[test]
    fn q8_k_iq4_nl_iq4_xs_blocks() {
        // Q8_K: an f32 scale (not f16), 256 signed quants, 16 i16 block sums the dequant ignores.
        let q: Vec<i8> = (0..256).map(|e| ((e * 13) % 256) as u8 as i8).collect();
        let mut blk = 0.375f32.to_le_bytes().to_vec();
        blk.extend(q.iter().map(|&v| v as u8));
        blk.extend_from_slice(&[0xAB; 32]);
        let want: Vec<f64> = q.iter().map(|&v| 0.375 * f64::from(v)).collect();
        assert_lanes(&dq(Dequant::Q8_K, &blk), &want, "Q8_K");

        // IQ4_NL: d = 0.5; nibble n_e = (3e + 1) % 16 indexes the non-uniform levels.
        let nib: Vec<u8> = (0..32).map(|e| ((3 * e + 1) % 16) as u8).collect();
        let mut blk = f16b(0.5).to_vec();
        for j in 0..16 {
            blk.push(nib[j] | (nib[j + 16] << 4));
        }
        let want: Vec<f64> = nib
            .iter()
            .map(|&n| 0.5 * f64::from(KVALUES_IQ4NL[usize::from(n)]))
            .collect();
        assert_lanes(&dq(Dequant::IQ4_NL, &blk), &want, "IQ4_NL");

        // IQ4_XS: eight 6-bit scales ls_b = (23b + 9) % 64 (scale ls − 32), packed as
        // `quantize_row_iq4_xs_impl` packs them; nibbles n_e = (5e + e/32) % 16.
        let ls: Vec<u16> = (0..8).map(|b| (23 * b + 9) % 64).collect();
        let mut scales_h = 0u16;
        let mut scales_l = [0u8; 4];
        for (b, &l) in ls.iter().enumerate() {
            scales_l[b / 2] |= ((l & 0xF) as u8) << (4 * (b % 2));
            scales_h |= (l >> 4) << (2 * b);
        }
        let nib: Vec<u8> = (0..256).map(|e| ((5 * e + e / 32) % 16) as u8).collect();
        let mut qs = [0u8; 128];
        for b in 0..8 {
            for j in 0..16 {
                qs[b * 16 + j] = nib[b * 32 + j] | (nib[b * 32 + j + 16] << 4);
            }
        }
        let blk = [&f16b(0.25)[..], &scales_h.to_le_bytes(), &scales_l, &qs].concat();
        let want: Vec<f64> = (0..256)
            .map(|e| {
                0.25 * (f64::from(ls[e / 32]) - 32.0)
                    * f64::from(KVALUES_IQ4NL[usize::from(nib[e])])
            })
            .collect();
        assert_lanes(&dq(Dequant::IQ4_XS, &blk), &want, "IQ4_XS");
    }

    #[test]
    fn metadata_accessors_read_every_kind_and_fall_back_to_the_default() {
        use MetaValue as V;
        let mut b = Gb::new();
        b.kv("general.architecture", &V::Str(b"llama".to_vec()))
            .kv("llama.block_count", &V::U32(12))
            .kv("llama.rope.freq_base", &V::F32(10000.0))
            .kv("big", &V::U64(u64::MAX))
            .kv("flag", &V::Bool(true))
            .kv("neg", &V::I8(-5))
            .kv(
                "tokens",
                &V::Arr(
                    vt::STRING,
                    vec![V::Str(b"<s>".to_vec()), V::Str(b"wukong".to_vec())],
                ),
            )
            .kv("scores", &V::Arr(vt::F32, vec![V::F32(0.5), V::F32(-1.25)]))
            .kv("ids", &V::Arr(vt::I32, vec![V::I32(3), V::I32(-4)]));
        let g = parse(&b.bytes()).unwrap();
        let q = |k: &str, what: i64| gguf_meta(&g, k.as_bytes(), what, 0, 0);
        assert_eq!(q("llama.block_count", GGUF_M_TYPE), i64::from(vt::U32));
        assert_eq!(q("tokens", GGUF_M_TYPE), i64::from(vt::ARRAY));
        assert_eq!(q("absent", GGUF_M_TYPE), IO_NAME_NOT_FOUND);
        assert_eq!(q("general.architecture", GGUF_M_LEN), 5);
        assert_eq!(q("tokens", GGUF_M_LEN), 2);
        assert_eq!(q("flag", GGUF_M_LEN), 1);
        assert_eq!(q("absent", GGUF_M_LEN), IO_NAME_NOT_FOUND);
        let int = |k: &str, i: i64| gguf_meta(&g, k.as_bytes(), GGUF_M_I64, i, -77);
        assert_eq!(int("llama.block_count", 0), 12);
        assert_eq!(int("big", 0), -1, "a u64 keeps its bits");
        assert_eq!(int("flag", 0), 1);
        assert_eq!(int("neg", 0), -5);
        assert_eq!(int("ids", 1), -4);
        for (k, i) in [
            ("ids", 2),
            ("ids", -1),
            ("llama.block_count", 1),
            ("llama.rope.freq_base", 0),
            ("tokens", 0),
            ("absent", 0),
        ] {
            assert_eq!(int(k, i), -77, "{k}[{i}]");
        }
        let f = Scratch::new("gguf_meta", &b.bytes());
        let fl = |k: &str, i: i64| gguf_meta_f64_path(&f.0, k.as_bytes(), i, -0.5);
        assert_eq!(fl("llama.rope.freq_base", 0), 10000.0);
        assert_eq!(fl("scores", 1), -1.25);
        assert_eq!(fl("llama.block_count", 0), 12.0, "an integer converts");
        assert_eq!(
            fl("big", 0),
            u64::MAX as f64,
            "a u64 converts by its unsigned value"
        );
        assert_eq!(fl("flag", 0), -0.5, "a bool is not a number");
        assert_eq!(fl("general.architecture", 0), -0.5);
        assert_eq!(
            gguf_meta_str_path(&f.0, b"tokens", 1),
            Ok(b"wukong".to_vec())
        );
        assert_eq!(
            gguf_meta_str_path(&f.0, b"tokens", 2),
            Err(IO_SHAPE_MISMATCH)
        );
        assert_eq!(gguf_meta_str_path(&f.0, b"ids", 0), Err(IO_DTYPE_MISMATCH));
        assert_eq!(
            gguf_meta_str_path(&f.0, b"absent", 0),
            Err(IO_NAME_NOT_FOUND)
        );
        // The native entries, including snprintf truncation and the untouched buffer on a code.
        let key = std::ffi::CString::new("general.architecture").unwrap();
        let kp = key.as_ptr() as *const u8;
        let mut buf = [0xAAu8; 4];
        // SAFETY: `buf` holds 4 writable bytes; both strings are NUL-terminated.
        let n = unsafe { wukong_rt_gguf_meta_str(f.ptr(), kp, 0, buf.as_mut_ptr(), 4) };
        assert_eq!((n, buf), (5, *b"lla\0"));
        let bad = std::ffi::CString::new("ids").unwrap();
        let mut keep = [0xAAu8; 4];
        // SAFETY: as above.
        let n = unsafe {
            wukong_rt_gguf_meta_str(f.ptr(), bad.as_ptr() as *const u8, 0, keep.as_mut_ptr(), 4)
        };
        assert_eq!((n, keep), (IO_DTYPE_MISMATCH, [0xAA; 4]));
        // SAFETY: both strings are NUL-terminated.
        unsafe {
            assert_eq!(wukong_rt_gguf_meta(f.ptr(), kp, GGUF_M_LEN, 0, 0), 5);
            assert_eq!(
                wukong_rt_gguf_meta(std::ptr::null(), kp, GGUF_M_I64, 0, 9),
                9
            );
            assert_eq!(
                wukong_rt_gguf_meta(std::ptr::null(), kp, GGUF_M_TYPE, 0, 9),
                IO_OPEN_FAILED
            );
            assert_eq!(wukong_rt_gguf_meta_f64(std::ptr::null(), kp, 0, 2.5), 2.5);
        }
        assert_eq!(
            copy_c_string(b"abc", 0, |_, _| panic!("nothing is written")),
            0
        );
    }

    // --- the entry point ----------------------------------------------------------------------

    /// A file holding one tensor of every read route.
    fn sample_file() -> (Vec<u8>, Vec<f32>, Vec<u8>) {
        // "w" F32 ne [3, 2] = [[1, -2, 3], [0.5, -0.25, 8]]; "h" F16 ne [4]; "b" BF16 ne [2];
        // "q" Q8_0 ne [32, 2] (two rows of one block); "iq" IQ2_XXS (no dequantizer); "i" I16
        // ne [3].
        let w = [1.0f32, -2.0, 3.0, 0.5, -0.25, 8.0];
        let wb: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let hb: Vec<u8> = [1.0f32, -0.5, 65504.0, 0.0]
            .iter()
            .flat_map(|&v| f16b(v))
            .collect();
        let bb: Vec<u8> = [0x3F80u16, 0xC040]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut qb = Vec::new();
        let mut qv = Vec::new();
        for (r, d) in [(0i32, 0.5f32), (1, -2.0)] {
            qb.extend_from_slice(&f16b(d));
            for j in 0..32i32 {
                let q = (j * 3 + r * 11) % 41 - 20;
                qb.push(q as i8 as u8);
                qv.push(q as f32 * d);
            }
        }
        let ib: Vec<u8> = [-1i16, 300, i16::MIN]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut b = Gb::new();
        b.kv("general.architecture", &MetaValue::Str(b"tiny".to_vec()))
            .tensor("w", &[3, 2], 0, &wb)
            .tensor("h", &[4], 1, &hb)
            .tensor("b", &[2], 30, &bb)
            .tensor("q", &[32, 2], 8, &qb)
            .tensor("iq", &[256], 16, &[0u8; 66])
            .tensor("i", &[3], 25, &ib);
        (b.bytes(), qv, qb)
    }

    /// Call the native entry on the scratch file `f`.
    fn read<T: Copy>(f: &Scratch, name: &str, out: &mut [T], dt: i64, dims: Option<&[i64]>) -> i64 {
        read_path(f.ptr(), name, out, dt, dims)
    }

    /// Call the native entry on the NUL-terminated `path`.
    fn read_path<T: Copy>(
        path: *const u8,
        name: &str,
        out: &mut [T],
        dt: i64,
        dims: Option<&[i64]>,
    ) -> i64 {
        let name = std::ffi::CString::new(name).unwrap();
        let (dp, rank) = match dims {
            Some(d) => (d.as_ptr(), d.len() as i64),
            None => (std::ptr::null(), -1),
        };
        // SAFETY: `out` holds `out.len()` elements of the dtype the caller names, and `dims` (when
        // given) addresses `rank` i64s.
        unsafe {
            wukong_rt_gguf_read(
                path,
                name.as_ptr() as *const u8,
                out.as_mut_ptr() as *mut u8,
                out.len() as i64,
                dt,
                dp,
                rank,
            )
        }
    }

    #[test]
    fn gguf_read_routes_every_type_and_checks_the_destination() {
        let (bytes, qv, qb) = sample_file();
        let f = Scratch::new("gguf", &bytes);
        // F32 verbatim into a [2, 3] tensor; [3, 2] is the transpose and is refused.
        let mut w = [0f32; 6];
        assert_eq!(read(&f, "w", &mut w, DT_F32, Some(&[2, 3])), 6);
        assert_eq!(w, [1.0, -2.0, 3.0, 0.5, -0.25, 8.0]);
        let mut keep = [9f32; 6];
        assert_eq!(
            read(&f, "w", &mut keep, DT_F32, Some(&[3, 2])),
            IO_SHAPE_MISMATCH
        );
        assert_eq!(
            read(&f, "w", &mut keep[..5], DT_F32, None),
            IO_SHAPE_MISMATCH
        );
        assert_eq!(keep, [9.0; 6]);
        // F16 widens into f32 and loads verbatim into f16; BF16 likewise.
        let mut h = [0f32; 4];
        assert_eq!(read(&f, "h", &mut h, DT_F32, None), 4);
        assert_eq!(h, [1.0, -0.5, 65504.0, 0.0]);
        let mut hraw = [0u16; 4];
        assert_eq!(read(&f, "h", &mut hraw, DT_F16, None), 4);
        assert_eq!(hraw, [0x3C00, 0xB800, 0x7BFF, 0]);
        let mut b = [0f32; 2];
        assert_eq!(read(&f, "b", &mut b, DT_F32, Some(&[2])), 2);
        assert_eq!(b, [1.0, -3.0]);
        let mut braw = [0u16; 2];
        assert_eq!(read(&f, "b", &mut braw, DT_BF16, None), 2);
        assert_eq!(braw, [0x3F80, 0xC040]);
        // Q8_0 dequantizes into a [2, 32] f32 tensor, or loads its raw blocks into a [2, 34] u8
        // one.
        let mut q = [0f32; 64];
        assert_eq!(read(&f, "q", &mut q, DT_F32, Some(&[2, 32])), 64);
        assert_eq!(q.to_vec(), qv);
        let mut raw = [0u8; 68];
        assert_eq!(read(&f, "q", &mut raw, DT_U8, Some(&[2, 34])), 68);
        assert_eq!(raw.to_vec(), qb);
        assert_eq!(
            read(&f, "q", &mut raw, DT_U8, Some(&[2, 32])),
            IO_SHAPE_MISMATCH
        );
        assert_eq!(read(&f, "q", &mut raw, DT_U8, None), 68);
        // Integers load verbatim; no other conversion exists.
        let mut i = [0i16; 3];
        assert_eq!(read(&f, "i", &mut i, DT_I16, None), 3);
        assert_eq!(i, [-1, 300, i16::MIN]);
        let mut i32s = [7i32; 3];
        assert_eq!(read(&f, "i", &mut i32s, DT_I32, None), IO_DTYPE_MISMATCH);
        assert_eq!(
            read(&f, "w", &mut [0f64; 6], DT_F64, None),
            IO_DTYPE_MISMATCH
        );
        assert_eq!(
            read(&f, "q", &mut [0i8; 64], DT_I8, None),
            IO_DTYPE_MISMATCH
        );
        assert_eq!(
            read(&f, "w", &mut [0u8; 24], DT_U8, None),
            IO_DTYPE_MISMATCH
        );
        // A block type with no dequantizer is -7 into f32; its raw blocks still load.
        let mut iq = [5f32; 256];
        assert_eq!(read(&f, "iq", &mut iq, DT_F32, None), IO_UNSUPPORTED_TYPE);
        assert_eq!(iq, [5.0; 256]);
        assert_eq!(read(&f, "iq", &mut [1u8; 66], DT_U8, None), 66);
        // Name, open and header errors.
        assert_eq!(read(&f, "absent", &mut w, DT_F32, None), IO_NAME_NOT_FOUND);
        let missing = std::ffi::CString::new("wukong_gguf_absent_zz.gguf").unwrap();
        let mp = missing.as_ptr() as *const u8;
        assert_eq!(read_path(mp, "w", &mut w, DT_F32, None), IO_OPEN_FAILED);
        let bad = Scratch::new("gguf_bad", b"GGUF\x01\x00\x00\x00");
        assert_eq!(read(&bad, "w", &mut w, DT_F32, None), IO_BAD_HEADER);
        let cut = Scratch::new("gguf_cut", &bytes[..bytes.len() - 1]);
        assert_eq!(read(&cut, "i", &mut i, DT_I16, None), IO_BAD_HEADER);
    }

    #[test]
    fn query_entry_reads_the_file() {
        let (bytes, _, _) = sample_file();
        let f = Scratch::new("gguf_q", &bytes);
        let q = |name: &str, what: i64, k: i64| {
            let name = std::ffi::CString::new(name).unwrap();
            // SAFETY: both strings are NUL-terminated.
            unsafe { wukong_rt_gguf_query(f.ptr(), name.as_ptr() as *const u8, what, k) }
        };
        assert_eq!(q("q", GGUF_Q_FIND, 0), 3);
        assert_eq!(q("q", GGUF_Q_RANK, 0), 2);
        assert_eq!((q("q", GGUF_Q_DIM, 0), q("q", GGUF_Q_DIM, 1)), (2, 32));
        assert_eq!(q("iq", GGUF_Q_TYPE, 0), 16);
        assert_eq!(q("nope", GGUF_Q_TYPE, 0), IO_NAME_NOT_FOUND);
        // SAFETY: a null path is the documented open failure.
        let null = unsafe { wukong_rt_gguf_query(std::ptr::null(), std::ptr::null(), 0, 0) };
        assert_eq!(null, IO_OPEN_FAILED);
    }

    #[test]
    fn a_multi_block_dequant_streams_across_chunks() {
        // More blocks than one staging chunk holds, so the chunk loop's offsets are exercised.
        let blocks = STAGE_UNITS + 3;
        let mut payload = Vec::with_capacity(blocks * 18);
        for b in 0..blocks {
            payload.extend_from_slice(&f16b(if b % 2 == 0 { 1.0 } else { -0.5 }));
            for j in 0..16u8 {
                payload.push(((b + usize::from(j)) % 16) as u8 | ((15 - j) << 4));
            }
        }
        let mut g = Gb::new();
        g.tensor("big", &[32 * blocks as u64], 2, &payload);
        let f = Scratch::new("gguf_big", &g.bytes());
        let mut y = vec![0f32; 32 * blocks];
        assert_eq!(read(&f, "big", &mut y, DT_F32, None), (32 * blocks) as i64);
        for b in [0, 1, STAGE_UNITS - 1, STAGE_UNITS, blocks - 1] {
            let d = if b % 2 == 0 { 1.0 } else { -0.5 };
            for j in 0..16 {
                let lo = ((b + j) % 16) as f32 - 8.0;
                let hi = (15 - j) as f32 - 8.0;
                assert_eq!(y[b * 32 + j], lo * d, "block {b} lane {j}");
                assert_eq!(y[b * 32 + j + 16], hi * d, "block {b} lane {}", j + 16);
            }
        }
    }
}
