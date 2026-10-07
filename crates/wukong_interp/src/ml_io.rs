//! The interpreter twins of the ML I/O runtime entries: dtype-coded file I/O, `file_size`, and the
//! safetensors / `.npy` container readers.
//!
//! Every arm here mirrors one `wukong_runtime` entry and shares that entry's *pure* code — the
//! transfer plan and the container parsers in `wukong_runtime::fileio` — so the return code,
//! the element count and every value cannot drift between the two backends; only the destination
//! differs. Native memory holds raw element bytes; this interpreter's memory holds one typed
//! [`Value`] per scalar slot, so a read decodes each element into its slot's canonical value, a
//! write encodes each slot back to bytes.
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

use crate::{file_read_path, kernel_scratch, kernel_values, Interp, Value};

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
/// (and as the native entries' lossy C-string decode).
fn str_arg(memory: &[Value], args: &[Value], i: usize, ctx: &str) -> Result<String, String> {
    file_read_path(memory, ptr_arg(args, i, ctx, "string")?)
}

/// A non-negative element count as a slot count.
fn count(n: i64) -> usize {
    usize::try_from(n.max(0)).unwrap_or(usize::MAX)
}

impl Interp<'_, '_> {
    /// Dispatch an ML I/O or randomness runtime entry, or `None` when `name` is not one of them
    /// (the caller then reports an unknown intrinsic).
    pub(crate) fn ml_intrinsic(
        &mut self,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, String>> {
        Some(match name {
            "wukong_rt_read_dt" => self.ml_read_dt(args),
            "wukong_rt_write_dt" => self.ml_write_dt(args),
            "wukong_rt_file_size" => self.ml_file_size(args),
            "wukong_rt_st_read" => self.ml_container_read(args, true),
            "wukong_rt_npy_read" => self.ml_container_read(args, false),
            _ => return None,
        })
    }

    /// Validate that `n` slots from `base` lie inside the arena before anything is read or written.
    fn ml_check_extent(&self, base: usize, n: usize, ctx: &str) -> Result<(), String> {
        let end = base
            .checked_add(n)
            .ok_or_else(|| format!("{ctx}: data out of bounds"))?;
        if end > self.memory.len() {
            return Err(format!("{ctx}: data out of bounds"));
        }
        Ok(())
    }

    /// Read `n` elements of `dt` from `file`'s current position and decode them into
    /// `memory[base..base + n]`. The arena is touched only after the whole read succeeded;
    /// `Ok(None)` is an I/O failure (`-2`).
    fn ml_decode_into(
        &mut self,
        file: &mut std::fs::File,
        base: usize,
        n: usize,
        dt: i64,
        ctx: &str,
    ) -> Result<Option<()>, String> {
        use std::io::Read;
        let sz = dtype_size(dt).ok_or_else(|| format!("{ctx}: unknown dtype {dt}"))?;
        self.ml_check_extent(base, n, ctx)?;
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
        self.ml_check_extent(base, n, CTX)?;
        if file.seek(SeekFrom::Start(off)).is_err() {
            return Ok(Value::Int(IO_ERROR as i128));
        }
        Ok(match self.ml_decode_into(&mut file, base, n, dt, CTX)? {
            Some(()) => Value::Int(n as i128),
            None => Value::Int(IO_ERROR as i128),
        })
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
            self.ml_check_extent(base, n, CTX)?;
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
        let rank = arg_i64(args, at + 5);
        let dims: Option<Vec<i64>> = if rank < 0 {
            None
        } else {
            let base = ptr_arg(args, at + 4, ctx, "dims")?;
            let vals = kernel_values(&self.memory, base, 1, count(rank), ctx)?;
            Some(vals.iter().map(|v| v.as_int() as i64).collect())
        };
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
        self.ml_check_extent(base, n, ctx)?;
        if file.seek(SeekFrom::Start(loc.data_off)).is_err() {
            return Ok(Value::Int(IO_ERROR as i128));
        }
        Ok(
            match self.ml_decode_into(&mut file, base, n, loc.dtype, ctx)? {
                Some(()) => Value::Int(n as i128),
                None => Value::Int(IO_ERROR as i128),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use wukong_runtime::fileio as rt;

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
}
