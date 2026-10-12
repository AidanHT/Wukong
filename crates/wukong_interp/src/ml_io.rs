//! The interpreter twins of the ML I/O and randomness runtime entries: dtype-coded file I/O,
//! `file_size`, the safetensors / `.npy` / GGUF container readers and the GGUF tensor-table
//! queries, the Philox fill, and the top-k / argsort / categorical-sampling kernels.
//!
//! Every arm here mirrors one `wukong_runtime` entry and shares that entry's *pure* code — the
//! transfer plan and the container parsers in `wukong_runtime::fileio` (for GGUF, the parser, the
//! plan and the dequantizing transfer in `wukong_runtime::gguf`), the generator in
//! `wukong_runtime::random`, the selection functions in `wukong_runtime::topk` — so the return code,
//! the element count and every value cannot drift between the two backends; only the destination
//! differs. Native memory holds raw element bytes; this interpreter's memory holds one typed
//! [`Value`] per scalar slot, so a read decodes each element into its slot's canonical value, a
//! write encodes each slot back to bytes, and a kernel marshals slots through a host buffer.
//!
//! **Canonical integer slots.** The interpreter stores every integer sign-extended from its own
//! width — the value `exec` produces when it masks a result to its MIR type — and a `Load` returns
//! the slot verbatim. An *unsigned* element therefore decodes sign-extended too: the `u8` byte
//! `0xC8` is `Value::Int(-56)`, exactly what a `200u8` constant or an `i8`-typed store leaves in a
//! slot. Decoding it zero-extended (`200`) made `b == 200u8` and `b > 150u8` false in the
//! interpreter and true on native, because the comparison sees a different representation of the
//! same byte.

use std::io::{Seek, SeekFrom};
use std::path::Path;

use wukong_runtime::fileio::{
    check_dest, dtype_size, open_container, plan_raw_read, Container, DT_BF16, DT_F16, DT_F32,
    DT_F64, DT_I16, DT_I32, DT_I8, DT_U16, DT_U32, DT_U8, IO_ERROR, IO_OPEN_FAILED,
};
use wukong_runtime::gguf::{
    copy_c_string, gguf_meta_f64_path, gguf_meta_path, gguf_meta_str_path, gguf_open_tensor,
    gguf_query_path, gguf_transfer,
};
use wukong_runtime::mmap::{map_plan, MAP_GGUF, MAP_SAFETENSORS};

use crate::{
    file_read_path, kernel_buffer, kernel_scratch, kernel_values, Interp, Intrinsic, Value,
};

/// Decode one little-endian element of dtype `dt` from `b` (exactly `dtype_size(dt)` bytes) into the
/// canonical slot value: floats as their exact `f64` image (half precision widened exactly), integers
/// sign-extended from their own width whatever their signedness (see the module docs).
pub(crate) fn decode_elem(dt: i64, b: &[u8]) -> Value {
    match dt {
        DT_F32 => Value::Float(f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64),
        DT_F64 => Value::Float(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ])),
        DT_F16 => {
            Value::Float(wukong_runtime::f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])) as f64)
        }
        DT_BF16 => {
            Value::Float(wukong_runtime::bf16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])) as f64)
        }
        DT_I8 | DT_U8 => Value::Int(b[0] as i8 as i128),
        DT_I16 | DT_U16 => Value::Int(i16::from_le_bytes([b[0], b[1]]) as i128),
        DT_I32 | DT_U32 => Value::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i128),
        // DT_I64 | DT_U64 — the only other codes `dtype_size` accepts.
        _ => {
            Value::Int(i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as i128)
        }
    }
}

/// Encode one slot value as a little-endian element of dtype `dt`, appending to `out`. Narrowing is
/// the store native performs: a float slot rounds through its element type (a `bf16`/`f16` slot
/// rounds to its grid exactly as native's narrowing store does), an integer slot truncates to its
/// width.
pub(crate) fn encode_elem(dt: i64, v: Value, out: &mut Vec<u8>) {
    match dt {
        DT_F32 => out.extend_from_slice(&(v.as_float() as f32).to_le_bytes()),
        DT_F64 => out.extend_from_slice(&v.as_float().to_le_bytes()),
        DT_F16 => out
            .extend_from_slice(&wukong_runtime::f32_to_f16_bits(v.as_float() as f32).to_le_bytes()),
        DT_BF16 => out.extend_from_slice(
            &wukong_runtime::f32_to_bf16_bits(v.as_float() as f32).to_le_bytes(),
        ),
        DT_I8 | DT_U8 => out.push(v.as_int() as u8),
        DT_I16 | DT_U16 => out.extend_from_slice(&(v.as_int() as i16).to_le_bytes()),
        DT_I32 | DT_U32 => out.extend_from_slice(&(v.as_int() as i32).to_le_bytes()),
        _ => out.extend_from_slice(&(v.as_int() as i64).to_le_bytes()),
    }
}

/// An `i64` ABI argument as the native entry receives it. MIR passes these as `i64` values, so the
/// slot is in range; clamping keeps a hand-built call from wrapping.
fn arg_i64(args: &[Value], i: usize) -> i64 {
    args.get(i)
        .map(|v| v.as_int().clamp(i64::MIN as i128, i64::MAX as i128) as i64)
        .unwrap_or(0)
}

/// The pointer argument at `i`.
fn ptr_arg(args: &[Value], i: usize, ctx: &str, what: &str) -> Result<usize, String> {
    match args.get(i).copied() {
        Some(Value::Ptr(p)) => Ok(p),
        _ => Err(format!("{ctx}: {what} is not a pointer")),
    }
}

/// The `*u8` string argument at `i`, reconstructed with the same lossy rule as the frozen family
/// (and as the native entries' lossy C-string decode). The null pointer — a `[]u8` operand with no
/// NUL, through `wukong_rt_cstr` — is the empty string, as natively: an empty path cannot be
/// opened (`-1`) and an empty name is just a name.
fn str_arg(memory: &[Value], args: &[Value], i: usize, ctx: &str) -> Result<String, String> {
    match ptr_arg(args, i, ctx, "string")? {
        crate::NULL_SLOT => Ok(String::new()),
        p => file_read_path(memory, p),
    }
}

/// A non-negative element count as a slot count.
fn count(n: i64) -> usize {
    usize::try_from(n.max(0)).unwrap_or(usize::MAX)
}

impl Interp<'_, '_> {
    /// Dispatch an ML I/O or randomness runtime entry, or `None` when `id` is not one of them
    /// (the caller then reports an unknown intrinsic).
    pub(crate) fn ml_intrinsic(
        &mut self,
        id: Intrinsic,
        args: &[Value],
    ) -> Option<Result<Value, String>> {
        use Intrinsic as I;
        Some(match id {
            I::RtReadDt => self.ml_read_dt(args),
            I::RtWriteDt => self.ml_write_dt(args),
            I::RtFileSize => self.ml_file_size(args),
            I::RtStRead => self.ml_container_read(args, true),
            I::RtNpyRead => self.ml_container_read(args, false),
            I::RtGgufRead => self.ml_gguf_read(args),
            I::RtGgufQuery => self.ml_gguf_query(args),
            I::RtGgufMeta => self.ml_gguf_meta(args),
            I::RtGgufMetaF64 => self.ml_gguf_meta_f64(args),
            I::RtGgufMetaStr => self.ml_gguf_meta_str(args),
            I::RtMap => self.ml_map(args),
            I::RtCstr => self.ml_cstr(args),
            I::RandFillF32 => self.ml_rand_fill(args),
            I::TopkF32 => self.ml_topk(args),
            I::ArgsortF32 => self.ml_argsort(args),
            I::SampleCategoricalF32 => self.ml_sample_categorical(args),
            _ => return None,
        })
    }

    /// Validate that `n` slots from `base` — the call's operand `k` — lie inside the arena, and
    /// inside one live allocation, before anything is read or written.
    fn ml_check_extent(&self, k: usize, base: usize, n: usize, ctx: &str) -> Result<(), String> {
        let end = base
            .checked_add(n)
            .ok_or_else(|| format!("{ctx}: data out of bounds"))?;
        if end > self.memory.len() {
            return Err(format!("{ctx}: data out of bounds"));
        }
        self.slots.live.check(ctx, k, base, n)
    }

    /// Read `n` elements of `dt` from `file`'s current position and decode them into
    /// `memory[base..base + n]`. The arena is touched only after the whole read succeeded;
    /// `Ok(None)` is an I/O failure (`-2`).
    fn ml_decode_into(
        &mut self,
        file: &mut std::fs::File,
        (k, base): (usize, usize),
        n: usize,
        dt: i64,
        ctx: &str,
    ) -> Result<Option<()>, String> {
        use std::io::Read;
        let sz = dtype_size(dt).ok_or_else(|| format!("{ctx}: unknown dtype {dt}"))?;
        self.ml_check_extent(k, base, n, ctx)?;
        let byte_len = n
            .checked_mul(sz)
            .ok_or_else(|| format!("{ctx}: byte extent overflow"))?;
        let mut bytes = kernel_scratch(byte_len, ctx)?;
        bytes.resize(byte_len, 0u8);
        if file.read_exact(&mut bytes).is_err() {
            return Ok(None);
        }
        for (slot, elem) in self.memory[base..base + n]
            .iter_mut()
            .zip(bytes.chunks_exact(sz))
        {
            *slot = decode_elem(dt, elem);
        }
        Ok(Some(()))
    }

    /// `wukong_rt_read_dt(path, byte_off, data, len, dtype) -> i64`.
    fn ml_read_dt(&mut self, args: &[Value]) -> Result<Value, String> {
        const CTX: &str = "file-io read";
        let path = str_arg(&self.memory, args, 0, CTX)?;
        let byte_off = arg_i64(args, 1);
        let len = arg_i64(args, 3);
        let dt = arg_i64(args, 4);
        let Ok(mut file) = std::fs::File::open(&path) else {
            return Ok(Value::Int(IO_OPEN_FAILED as i128));
        };
        let file_len = match file.metadata() {
            Ok(m) => m.len(),
            Err(_) => return Ok(Value::Int(IO_ERROR as i128)),
        };
        let (off, n) = match plan_raw_read(file_len, byte_off, len, dt) {
            Ok(plan) => plan,
            Err(code) => return Ok(Value::Int(code as i128)),
        };
        if n == 0 {
            return Ok(Value::Int(0));
        }
        let base = ptr_arg(args, 2, CTX, "data")?;
        self.ml_check_extent(2, base, n, CTX)?;
        if file.seek(SeekFrom::Start(off)).is_err() {
            return Ok(Value::Int(IO_ERROR as i128));
        }
        Ok(
            match self.ml_decode_into(&mut file, (2, base), n, dt, CTX)? {
                Some(()) => Value::Int(n as i128),
                None => Value::Int(IO_ERROR as i128),
            },
        )
    }

    /// `wukong_rt_write_dt(path, data, len, dtype) -> i64`. The source extent is validated before the
    /// file is created, so a malformed call reports a diagnostic without truncating anything.
    fn ml_write_dt(&mut self, args: &[Value]) -> Result<Value, String> {
        use std::io::Write;
        const CTX: &str = "file-io write";
        let path = str_arg(&self.memory, args, 0, CTX)?;
        let len = arg_i64(args, 2);
        let dt = arg_i64(args, 3);
        let Some(sz) = dtype_size(dt) else {
            return Ok(Value::Int(IO_ERROR as i128));
        };
        let n = count(len);
        let mut out = Vec::new();
        if n > 0 {
            let base = ptr_arg(args, 1, CTX, "data")?;
            self.ml_check_extent(1, base, n, CTX)?;
            let byte_len = n
                .checked_mul(sz)
                .ok_or_else(|| format!("{CTX}: byte extent overflow"))?;
            out = kernel_scratch(byte_len, CTX)?;
            for &v in &self.memory[base..base + n] {
                encode_elem(dt, v, &mut out);
            }
        }
        let Ok(mut file) = std::fs::File::create(&path) else {
            return Ok(Value::Int(IO_OPEN_FAILED as i128));
        };
        if file.write_all(&out).is_err() {
            return Ok(Value::Int(IO_ERROR as i128));
        }
        Ok(Value::Int(n as i128))
    }

    /// `wukong_rt_file_size(path) -> i64`.
    fn ml_file_size(&mut self, args: &[Value]) -> Result<Value, String> {
        let path = str_arg(&self.memory, args, 0, "file_size")?;
        let Ok(file) = std::fs::File::open(&path) else {
            return Ok(Value::Int(IO_OPEN_FAILED as i128));
        };
        Ok(Value::Int(match file.metadata() {
            Ok(m) => i64::try_from(m.len()).unwrap_or(IO_ERROR) as i128,
            Err(_) => IO_ERROR as i128,
        }))
    }

    /// `wukong_rt_st_read(path, name, data, len, dtype, dims, rank) -> i64` (`safetensors`) and
    /// `wukong_rt_npy_read(path, data, len, dtype, dims, rank) -> i64`: the runtime's own
    /// [`open_container`] and [`check_dest`], then a decode of the payload — in its *source* dtype,
    /// so a `bf16`/`f16` payload widens into an `f32` destination exactly as native's widening load.
    fn ml_container_read(&mut self, args: &[Value], safetensors: bool) -> Result<Value, String> {
        let ctx = if safetensors { "st_read" } else { "npy_read" };
        let path = str_arg(&self.memory, args, 0, ctx)?;
        // The safetensors entry has one more leading operand (the tensor name).
        let at = usize::from(safetensors);
        let name = if safetensors {
            Some(str_arg(&self.memory, args, 1, ctx)?)
        } else {
            None
        };
        let len = arg_i64(args, at + 2);
        let dt = arg_i64(args, at + 3);
        let dims = self.ml_dims(args, at + 4, ctx)?;
        let container = match &name {
            Some(n) => Container::Safetensors(n),
            None => Container::Npy,
        };
        let (mut file, loc) = match open_container(Path::new(&path), &container) {
            Ok(found) => found,
            Err(code) => return Ok(Value::Int(code as i128)),
        };
        let (n, _widen) = match check_dest(&loc, dt, len, dims.as_deref()) {
            Ok(plan) => plan,
            Err(code) => return Ok(Value::Int(code as i128)),
        };
        if n == 0 {
            return Ok(Value::Int(0));
        }
        let base = ptr_arg(args, at + 1, ctx, "data")?;
        self.ml_check_extent(at + 1, base, n, ctx)?;
        if file.seek(SeekFrom::Start(loc.data_off)).is_err() {
            return Ok(Value::Int(IO_ERROR as i128));
        }
        Ok(
            match self.ml_decode_into(&mut file, (at + 1, base), n, loc.dtype, ctx)? {
                Some(()) => Value::Int(n as i128),
                None => Value::Int(IO_ERROR as i128),
            },
        )
    }

    /// The `(dims, rank)` operands at `at` / `at + 1`: `None` for a flat destination (`rank < 0`),
    /// else the tensor view's dims.
    fn ml_dims(&self, args: &[Value], at: usize, ctx: &str) -> Result<Option<Vec<i64>>, String> {
        let rank = arg_i64(args, at + 1);
        if rank < 0 {
            return Ok(None);
        }
        let base = ptr_arg(args, at, ctx, "dims")?;
        let vals = kernel_values(arena!(self), at, base, 1, count(rank), ctx)?;
        Ok(Some(vals.iter().map(|v| v.as_int() as i64).collect()))
    }

    /// `wukong_rt_gguf_read(path, name, data, len, dtype, dims, rank) -> i64`: the runtime's own
    /// open / locate / plan ([`gguf_open_tensor`]) and transfer ([`gguf_transfer`]) into a host
    /// buffer of the destination's little-endian bytes, then decoded slot by slot — so a widened or
    /// dequantized value is the native read's, bit for bit.
    fn ml_gguf_read(&mut self, args: &[Value]) -> Result<Value, String> {
        const CTX: &str = "gguf_read";
        let path = str_arg(&self.memory, args, 0, CTX)?;
        let name = str_arg(&self.memory, args, 1, CTX)?;
        let len = arg_i64(args, 3);
        let dt = arg_i64(args, 4);
        let dims = self.ml_dims(args, 5, CTX)?;
        let (mut file, loc, plan) =
            match gguf_open_tensor(Path::new(&path), name.as_bytes(), dt, len, dims.as_deref()) {
                Ok(found) => found,
                Err(code) => return Ok(Value::Int(code as i128)),
            };
        let n = plan.count();
        if n == 0 {
            return Ok(Value::Int(0));
        }
        let sz = dtype_size(dt).ok_or_else(|| format!("{CTX}: unknown dtype {dt}"))?;
        let base = ptr_arg(args, 2, CTX, "data")?;
        self.ml_check_extent(2, base, n, CTX)?;
        let byte_len = n
            .checked_mul(sz)
            .ok_or_else(|| format!("{CTX}: byte extent overflow"))?;
        let mut bytes = kernel_scratch(byte_len, CTX)?;
        bytes.resize(byte_len, 0u8);
        if let Err(code) = gguf_transfer(&mut file, &loc, plan, &mut bytes) {
            return Ok(Value::Int(code as i128));
        }
        for (slot, elem) in self.memory[base..base + n]
            .iter_mut()
            .zip(bytes.chunks_exact(sz))
        {
            *slot = decode_elem(dt, elem);
        }
        Ok(Value::Int(n as i128))
    }

    /// `wukong_rt_gguf_query(path, name, what, k) -> i64`.
    fn ml_gguf_query(&mut self, args: &[Value]) -> Result<Value, String> {
        const CTX: &str = "gguf_query";
        let path = str_arg(&self.memory, args, 0, CTX)?;
        let name = str_arg(&self.memory, args, 1, CTX)?;
        let (what, k) = (arg_i64(args, 2), arg_i64(args, 3));
        let got = gguf_query_path(Path::new(&path), name.as_bytes(), what, k);
        Ok(Value::Int(got as i128))
    }

    /// `wukong_rt_cstr(data, len) -> data | null`: the buffer when one of its first `len` slots
    /// holds a zero byte (the low byte, as `file_read_path` reads it), else the null pointer.
    fn ml_cstr(&mut self, args: &[Value]) -> Result<Value, String> {
        const CTX: &str = "string operand";
        let n = count(arg_i64(args, 1));
        if n == 0 {
            return Ok(Value::Ptr(crate::NULL_SLOT));
        }
        let base = ptr_arg(args, 0, CTX, "data")?;
        self.ml_check_extent(0, base, n, CTX)?;
        let terminated = self.memory[base..base + n]
            .iter()
            .any(|v| v.as_int() as u8 == 0);
        Ok(Value::Ptr(if terminated { base } else { crate::NULL_SLOT }))
    }

    /// `wukong_rt_map(path, name, container, dtype, out_len) -> data`: the runtime's own
    /// [`map_plan`], then a block of slots made exactly as `alloc_<T>` makes one (so `free`
    /// releases and reuses it) holding the decoded payload — the copy that native's copy-on-write
    /// mapping is indistinguishable from. Any failure, and an empty payload, is the null pointer
    /// with length 0, as natively.
    fn ml_map(&mut self, args: &[Value]) -> Result<Value, String> {
        use std::io::Read;
        const CTX: &str = "map";
        let path = str_arg(&self.memory, args, 0, CTX)?;
        let container = arg_i64(args, 2);
        let name = if matches!(container, MAP_SAFETENSORS | MAP_GGUF) {
            str_arg(&self.memory, args, 1, CTX)?
        } else {
            String::new()
        };
        let dt = arg_i64(args, 3);
        let len_slot = ptr_arg(args, 4, CTX, "length")?;
        self.ml_check_extent(4, len_slot, 1, CTX)?;
        let mut out = (Value::Ptr(crate::NULL_SLOT), 0usize);
        if let (Ok(plan), Some(sz)) = (
            map_plan(Path::new(&path), container, &name, dt),
            dtype_size(dt),
        ) {
            if plan.count > 0 {
                let byte_len = plan
                    .count
                    .checked_mul(sz)
                    .ok_or_else(|| format!("{CTX}: byte extent overflow"))?;
                let mut bytes = kernel_scratch(byte_len, CTX)?;
                bytes.resize(byte_len, 0u8);
                let mut file = plan.file;
                let read = file.seek(SeekFrom::Start(plan.off)).is_ok()
                    && file.read_exact(&mut bytes).is_ok();
                if read {
                    let is_float = matches!(dt, DT_F32 | DT_F64 | DT_F16 | DT_BF16);
                    let alloc = [
                        Value::Int(plan.count as i128),
                        Value::Int(sz as i128),
                        Value::Int(i128::from(is_float)),
                    ];
                    let block = self.call_intrinsic(Intrinsic::RtAlloc, &alloc)?;
                    if let Value::Ptr(base) = block {
                        for (slot, elem) in self.memory[base..base + plan.count]
                            .iter_mut()
                            .zip(bytes.chunks_exact(sz))
                        {
                            *slot = decode_elem(dt, elem);
                        }
                        out = (block, plan.count);
                    }
                }
            }
        }
        self.memory[len_slot] = Value::Int(out.1 as i128);
        Ok(out.0)
    }

    /// `wukong_rt_gguf_meta(path, key, what, i, default) -> i64`.
    fn ml_gguf_meta(&mut self, args: &[Value]) -> Result<Value, String> {
        const CTX: &str = "gguf_meta";
        let path = str_arg(&self.memory, args, 0, CTX)?;
        let key = str_arg(&self.memory, args, 1, CTX)?;
        let (what, i, default) = (arg_i64(args, 2), arg_i64(args, 3), arg_i64(args, 4));
        let got = gguf_meta_path(Path::new(&path), key.as_bytes(), what, i, default);
        Ok(Value::Int(got as i128))
    }

    /// `wukong_rt_gguf_meta_f64(path, key, i, default) -> f64`.
    fn ml_gguf_meta_f64(&mut self, args: &[Value]) -> Result<Value, String> {
        const CTX: &str = "gguf_meta_f64";
        let path = str_arg(&self.memory, args, 0, CTX)?;
        let key = str_arg(&self.memory, args, 1, CTX)?;
        let i = arg_i64(args, 2);
        let default = args.get(3).map(|v| v.as_float()).unwrap_or(0.0);
        let got = gguf_meta_f64_path(Path::new(&path), key.as_bytes(), i, default);
        Ok(Value::Float(got))
    }

    /// `wukong_rt_gguf_meta_str(path, key, i, buf, len) -> i64`: the runtime's own lookup, then
    /// the same `snprintf` copy into the buffer's slots (each byte the canonical `u8` slot value).
    fn ml_gguf_meta_str(&mut self, args: &[Value]) -> Result<Value, String> {
        const CTX: &str = "gguf_meta_str";
        let path = str_arg(&self.memory, args, 0, CTX)?;
        let key = str_arg(&self.memory, args, 1, CTX)?;
        let i = arg_i64(args, 2);
        let len = count(arg_i64(args, 4));
        match gguf_meta_str_path(Path::new(&path), key.as_bytes(), i) {
            Ok(s) => {
                if len > 0 {
                    let base = ptr_arg(args, 3, CTX, "buffer")?;
                    self.ml_check_extent(3, base, len, CTX)?;
                    let memory = &mut self.memory;
                    copy_c_string(&s, len, |k, b| memory[base + k] = decode_elem(DT_U8, &[b]));
                }
                Ok(Value::Int(s.len() as i128))
            }
            Err(code) => Ok(Value::Int(code as i128)),
        }
    }

    /// `n` `f32` slots of operand `k` from `base`, marshalled for a host kernel.
    fn ml_f32_buffer(
        &self,
        k: usize,
        base: usize,
        n: usize,
        ctx: &str,
    ) -> Result<Vec<f32>, String> {
        let vals = kernel_values(arena!(self), k, base, 1, n, ctx)?;
        kernel_buffer(vals, ctx, |v| v.kernel_float(ctx).map(|f| f as f32))
    }

    /// `wukong_rand_fill_f32(out, n, seed, ctr0)`: the runtime's own [`wukong_runtime::random::rand_f32`]
    /// per slot, the counter wrapping like the native kernel's.
    fn ml_rand_fill(&mut self, args: &[Value]) -> Result<Value, String> {
        const CTX: &str = "rand_fill_f32";
        let n = count(arg_i64(args, 1));
        if n == 0 {
            return Ok(Value::Unit);
        }
        let seed = arg_i64(args, 2) as u64;
        let ctr0 = arg_i64(args, 3) as u64;
        let base = ptr_arg(args, 0, CTX, "data")?;
        self.ml_check_extent(0, base, n, CTX)?;
        for (i, slot) in self.memory[base..base + n].iter_mut().enumerate() {
            let x = wukong_runtime::random::rand_f32(seed, ctr0.wrapping_add(i as u64));
            *slot = Value::Float(x as f64);
        }
        Ok(Value::Unit)
    }

    /// `wukong_topk_f32(x, n, vals, nv, idx, ni) -> i64`: the input is copied in whole before any
    /// output is written, matching the native entry's overlap rule.
    fn ml_topk(&mut self, args: &[Value]) -> Result<Value, String> {
        const CTX: &str = "topk_f32";
        let n = count(arg_i64(args, 1));
        let k = n.min(count(arg_i64(args, 3))).min(count(arg_i64(args, 5)));
        let xs = self.ml_f32_buffer(0, ptr_arg(args, 0, CTX, "x")?, n, CTX)?;
        if k == 0 {
            return Ok(Value::Int(0));
        }
        let vals_base = ptr_arg(args, 2, CTX, "vals")?;
        let idx_base = ptr_arg(args, 4, CTX, "idx")?;
        self.ml_check_extent(2, vals_base, k, CTX)?;
        self.ml_check_extent(4, idx_base, k, CTX)?;
        let mut vals = vec![0f32; k];
        let mut idx = vec![0i64; k];
        let got = wukong_runtime::topk::topk(&xs, &mut vals, &mut idx);
        for (j, v) in vals.into_iter().enumerate() {
            self.memory[vals_base + j] = Value::Float(v as f64);
        }
        for (j, i) in idx.into_iter().enumerate() {
            self.memory[idx_base + j] = Value::Int(i as i128);
        }
        Ok(Value::Int(got as i128))
    }

    /// `wukong_argsort_f32(x, n, idx, ni, descending) -> i64`.
    fn ml_argsort(&mut self, args: &[Value]) -> Result<Value, String> {
        const CTX: &str = "argsort_f32";
        let n = count(arg_i64(args, 1));
        let m = n.min(count(arg_i64(args, 3)));
        let descending = arg_i64(args, 4) != 0;
        let xs = self.ml_f32_buffer(0, ptr_arg(args, 0, CTX, "x")?, n, CTX)?;
        if m == 0 {
            return Ok(Value::Int(0));
        }
        let idx_base = ptr_arg(args, 2, CTX, "idx")?;
        self.ml_check_extent(2, idx_base, m, CTX)?;
        let mut idx = vec![0i64; m];
        let got = wukong_runtime::topk::argsort(&xs, &mut idx, descending);
        for (j, i) in idx.into_iter().enumerate() {
            self.memory[idx_base + j] = Value::Int(i as i128);
        }
        Ok(Value::Int(got as i128))
    }

    /// `wukong_sample_categorical_f32(p, n, u) -> i64`.
    fn ml_sample_categorical(&mut self, args: &[Value]) -> Result<Value, String> {
        const CTX: &str = "sample_categorical";
        let n = count(arg_i64(args, 1));
        let p = self.ml_f32_buffer(0, ptr_arg(args, 0, CTX, "probs")?, n, CTX)?;
        let u = args.get(2).map(|v| v.as_float() as f32).unwrap_or(0.0);
        Ok(Value::Int(
            wukong_runtime::topk::sample_categorical(&p, u) as i128
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::{run_with_output, Interner};
    use wukong_runtime::fileio as rt;
    use wukong_span::SourceId;

    fn stdout_of(src: &str) -> String {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        let main = interner.intern("main");
        let (_, out) = run_with_output(&program, main, &interner).unwrap();
        String::from_utf8(out).unwrap()
    }

    /// The compiler's dtype codes (`wukong_sema::io_dtype_code`, emitted as MIR constants) and the
    /// runtime's (`fileio::DT_*`, decoded by both backends) are two copies of one table: pin every
    /// row, since a skew would read the right bytes as the wrong type on every backend at once.
    #[test]
    fn compiler_and_runtime_dtype_codes_agree() {
        let rows = [
            ("f32", rt::DT_F32),
            ("f64", rt::DT_F64),
            ("f16", rt::DT_F16),
            ("bf16", rt::DT_BF16),
            ("i8", rt::DT_I8),
            ("i16", rt::DT_I16),
            ("i32", rt::DT_I32),
            ("i64", rt::DT_I64),
            ("u8", rt::DT_U8),
            ("u16", rt::DT_U16),
            ("u32", rt::DT_U32),
            ("u64", rt::DT_U64),
        ];
        for (suffix, dt) in rows {
            let s = wukong_sema::io_dtype(suffix).unwrap();
            assert_eq!(wukong_sema::io_dtype_code(s), dt, "{suffix}");
        }
    }

    /// The GGUF query codes (`wukong_sema::GgufQuery::code`, emitted as MIR constants) are the
    /// runtime's `GGUF_Q_*`.
    #[test]
    fn compiler_and_runtime_gguf_query_codes_agree() {
        use wukong_runtime::gguf as g;
        use wukong_sema::GgufQuery as Q;
        for (q, code) in [
            (Q::Find, g::GGUF_Q_FIND),
            (Q::Rank, g::GGUF_Q_RANK),
            (Q::Dim, g::GGUF_Q_DIM),
            (Q::Type, g::GGUF_Q_TYPE),
        ] {
            assert_eq!(q.code(), code, "{q:?}");
        }
    }

    /// The GGUF metadata accessor codes (`wukong_sema::GgufMeta::code`) are the runtime's
    /// `GGUF_M_*`.
    #[test]
    fn compiler_and_runtime_gguf_meta_codes_agree() {
        use wukong_runtime::gguf as g;
        use wukong_sema::GgufMeta as M;
        for (m, code) in [
            (M::Type, g::GGUF_M_TYPE),
            (M::Len, g::GGUF_M_LEN),
            (M::I64, g::GGUF_M_I64),
        ] {
            assert_eq!(m.code(), code, "{m:?}");
        }
    }

    /// The map container codes (`wukong_sema::MapSource::code`) are the runtime's `MAP_*`.
    #[test]
    fn compiler_and_runtime_map_codes_agree() {
        use wukong_runtime::mmap as m;
        use wukong_sema::MapSource as S;
        for (s, code) in [
            (S::File, m::MAP_FILE),
            (S::Safetensors, m::MAP_SAFETENSORS),
            (S::Npy, m::MAP_NPY),
            (S::Gguf, m::MAP_GGUF),
        ] {
            assert_eq!(s.code(), code, "{s:?}");
            assert_eq!(
                s.named(),
                matches!(code, m::MAP_SAFETENSORS | m::MAP_GGUF),
                "{s:?}"
            );
        }
    }

    /// The RNG builtins lower to plain integer MIR; the runtime's `random` module is their
    /// reference (itself pinned to the Random123 known answers). Sweep seeds whose high word is set
    /// (a negative seed) and counters past 2^32, so both 32-bit halves of each operand matter.
    #[test]
    fn rng_lowering_matches_the_runtime_reference() {
        let src = "fn main() -> i32 { \
                   let mut s: i64 = 0; \
                   while s < 4 { \
                     let mut c: i64 = 0; \
                     while c < 6 { \
                       let seed: i64 = s * 1000003 - 7; \
                       let ctr: i64 = c * 77777777777 + s; \
                       print(rand_u32(seed, ctr)); \
                       print((rand_f32(seed, ctr) * 16777216.0) as i64); \
                       print((randn_f32(seed, ctr) * 1000000.0) as i64); \
                       c = c + 1; \
                     } \
                     s = s + 1; \
                   } \
                   return 0; }";
        let out = stdout_of(src);
        let mut lines = out.lines();
        for s in 0..4i64 {
            for c in 0..6i64 {
                let seed = (s * 1_000_003 - 7) as u64;
                let ctr = (c * 77_777_777_777 + s) as u64;
                let w = wukong_runtime::random::philox_block(seed, ctr);
                let got_u32: u64 = lines.next().unwrap().parse().unwrap();
                assert_eq!(got_u32, u64::from(w[0]), "rand_u32 s={s} c={c}");
                let got_f32: i64 = lines.next().unwrap().parse().unwrap();
                assert_eq!(got_f32, i64::from(w[0] >> 8), "rand_f32 s={s} c={c}");
                // Box-Muller in f64 from the same two words; the lowering computes it in f32 on
                // the compiler's log/cos, so compare to a few millionths.
                let u1 = f64::from((w[0] >> 8) + 1) / 16_777_216.0;
                let u2 = f64::from(w[1] >> 8) / 16_777_216.0;
                let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                let got_n: i64 = lines.next().unwrap().parse().unwrap();
                assert!(
                    (got_n as f64 - z * 1e6).abs() <= 25.0,
                    "randn s={s} c={c}: {got_n} vs {}",
                    z * 1e6
                );
            }
        }
        assert_eq!(lines.next(), None);
    }

    /// `rand_fill_f32` (a runtime kernel) and `rand_f32` (inline MIR) are one stream.
    #[test]
    fn rng_fill_matches_the_scalar_builtin() {
        let src = "fn main() -> i32 { \
                   let mut a: []f32 = alloc_f32(9); \
                   rand_fill_f32(a, 1234, -4); \
                   for i in 0..9 { \
                     let r: f32 = rand_f32(1234, i - 4); \
                     if a[i] == r { print(1); } else { print(0); } \
                   } \
                   free(a); \
                   return 0; }";
        assert_eq!(stdout_of(src), "1\n".repeat(9));
    }
}
