//! The interpreter twins of the text runtime entries (`wukong_runtime::fmt`): the formatting calls
//! `printf`, `sprintf`, the structural `print` of an aggregate and the raw byte writers lower to —
//! and of the process interface (`wukong_runtime::process`: arguments, environment, standard input,
//! `exit`). A string the process interface returns (`arg(i)`, `env(name)`) is copied into fresh
//! byte slots per call, NUL-terminated like a string literal's static data.
//!
//! Every arm renders through the runtime's own pure functions — `format_int`, `format_f32`,
//! `format_seq`, the [`fmt::buffer_write`] plan — so the bytes and the returned position cannot
//! drift from a native run; only the destination differs. A stream write goes to this run's
//! captured stdout (or the process's stderr); a buffer write lands in byte slots, each the
//! canonical `u8` slot value a `read_u8` would decode (`ml_io::decode_elem`), and a string or a
//! buffer is read back out of slots the way `print_str` reads them.

use wukong_runtime::fileio::DT_U8;
use wukong_runtime::fmt::{self, Spec, TARGET_BUFFER, TARGET_STDERR};
use wukong_runtime::process;
use wukong_runtime::utf8;

use crate::{Interp, Intrinsic, Value};

/// The integer operand at `i` as the native entry receives it (an `i64`).
fn int(args: &[Value], i: usize) -> i64 {
    args.get(i).map_or(0, |v| {
        v.as_int().clamp(i64::MIN as i128, i64::MAX as i128) as i64
    })
}

/// The float operand at `i`.
fn float(args: &[Value], i: usize) -> f64 {
    args.get(i).map_or(0.0, |v| v.as_float())
}

impl Interp<'_, '_> {
    /// Dispatch a text runtime entry, or `None` when `id` is not one of them.
    pub(crate) fn text_intrinsic(
        &mut self,
        id: Intrinsic,
        args: &[Value],
    ) -> Option<Result<Value, String>> {
        use Intrinsic as I;
        let mut out = Vec::new();
        Some(match id {
            I::RtFmtInt => {
                fmt::format_int(&mut out, int(args, 4), Spec::unpack(int(args, 5)));
                self.text_emit(args, &out)
            }
            I::RtFmtF64 => {
                fmt::format_f64(&mut out, float(args, 4), Spec::unpack(int(args, 5)));
                self.text_emit(args, &out)
            }
            I::RtFmtF32 => {
                // Exact: an `f32` operand's slot holds the `f64` image of an `f32`.
                fmt::format_f32(&mut out, float(args, 4) as f32, Spec::unpack(int(args, 5)));
                self.text_emit(args, &out)
            }
            I::RtFmtStr => self.text_fmt_str(args),
            I::RtFmtLit => {
                let (bytes, n) = fmt::lit_operand(int(args, 4), int(args, 5));
                self.text_emit(args, &bytes[..n])
            }
            I::RtFmtSlice => self.text_fmt_slice(args),
            I::RtArgCount => Ok(Value::Int(i128::from(process::arg_count()))),
            I::RtArg => {
                let arg = process::arg_cstr(int(args, 0)).to_bytes();
                Ok(self.text_alloc_str(arg))
            }
            I::RtArgI64 => {
                let arg = process::arg_cstr(int(args, 0)).to_bytes();
                let v = process::parse_i64_arg(arg, int(args, 1));
                Ok(Value::Int(i128::from(v)))
            }
            I::RtArgF32 => {
                let arg = process::arg_cstr(int(args, 0)).to_bytes();
                let v = process::parse_f32_arg(arg, float(args, 1) as f32);
                Ok(Value::Float(f64::from(v)))
            }
            I::RtEnv => self.text_env(args),
            I::RtReadStdin => self.text_read_stdin(args),
            // The run stops here; `run_capturing` / `run_streaming` report the code as its result.
            I::RtExit => Err(format!("{}{}", process::EXIT_SENTINEL, int(args, 0))),
            I::RtUtf8Check => self
                .utf8_operand(args)
                .map(|b| Value::Int(i128::from(utf8::pack_fault(utf8::validate(&b))))),
            I::RtUtf8Count => self
                .utf8_operand(args)
                .map(|b| Value::Int(utf8::char_count(&b) as i128)),
            I::RtUtf8Decode => self.utf8_operand(args).map(|b| {
                let packed = match usize::try_from(int(args, 2)) {
                    Ok(i) if i < b.len() => {
                        let (cp, w) = utf8::decode_at(&b, i);
                        utf8::pack_char(cp, w)
                    }
                    _ => utf8::pack_char(utf8::REPLACEMENT, 1),
                };
                Value::Int(i128::from(packed))
            }),
            _ => return None,
        })
    }

    /// Copy `bytes` into fresh slots, NUL-terminated, and return their base: a `*u8` string. Each
    /// byte is held as the canonical `u8` slot value a `read_u8` decodes (sign-extended from 8
    /// bits), the form a string literal's bytes and every stored byte take: a `u8 == u8` compares
    /// slots without widening, so a zero-extended `arg`/`env` byte at or above 0x80 compared unequal
    /// to the same byte of a literal on this backend only.
    fn text_alloc_str(&mut self, bytes: &[u8]) -> Value {
        let base = self.memory.len();
        self.memory.extend(
            bytes
                .iter()
                .map(|&b| crate::ml_io::decode_elem(DT_U8, &[b])),
        );
        self.memory.push(Value::Int(0));
        self.slots.live.insert(base, bytes.len() + 1);
        Value::Ptr(base)
    }

    /// `wukong_rt_env(name) -> *u8`.
    fn text_env(&mut self, args: &[Value]) -> Result<Value, String> {
        let name = match args.first() {
            Some(Value::Ptr(p)) if !self.is_null(*p) => self.text_read_bytes(0, *p, -1)?,
            _ => Vec::new(),
        };
        let value = process::env_cstr(&name).to_bytes();
        Ok(self.text_alloc_str(value))
    }

    /// `wukong_rt_read_stdin(buf, cap) -> i64`: each byte read lands in its slot as the canonical
    /// `u8` value a `read_u8` would decode.
    fn text_read_stdin(&mut self, args: &[Value]) -> Result<Value, String> {
        let cap = usize::try_from(int(args, 1)).unwrap_or(0);
        let base = match args.first() {
            Some(Value::Ptr(p)) if !self.is_null(*p) && cap > 0 => *p,
            _ => return Ok(Value::Int(0)),
        };
        if base
            .checked_add(cap)
            .is_none_or(|end| end > self.memory.len())
        {
            return Err("read_stdin_u8: buffer out of bounds".into());
        }
        self.slots.live.check("read_stdin_u8", 0, base, cap)?;
        let mut bytes = vec![0u8; cap];
        let n = process::read_stdin_into(&mut bytes);
        let got = usize::try_from(n).unwrap_or(0);
        for (k, &b) in bytes[..got].iter().enumerate() {
            self.memory[base + k] = crate::ml_io::decode_elem(DT_U8, &[b]);
        }
        Ok(Value::Int(i128::from(n)))
    }

    /// Deliver formatted `bytes` to the sink named by the leading `(target, buf, cap, pos)`
    /// operands and return the position after them, as `wukong_runtime::fmt::emit` does natively.
    fn text_emit(&mut self, args: &[Value], bytes: &[u8]) -> Result<Value, String> {
        let target = int(args, 0);
        let pos = int(args, 3);
        if target != TARGET_BUFFER {
            if target == TARGET_STDERR {
                // The runtime's own stream writer, which a linked executable and the JIT use too
                // (and which honours a capture: `fmt::begin_stderr_capture`).
                fmt::write_stream(TARGET_STDERR, bytes);
            } else {
                self.emit_stdout(bytes);
            }
            let end = pos.saturating_add(i64::try_from(bytes.len()).unwrap_or(i64::MAX));
            return Ok(Value::Int(i128::from(end)));
        }
        let w = fmt::buffer_write(int(args, 2), pos, bytes.len());
        if let Some(nul) = w.nul {
            let base = match args.get(1) {
                Some(Value::Ptr(p)) => *p,
                _ => return Err("text format: the buffer operand is not a pointer".into()),
            };
            // The NUL is the highest slot written; check the whole extent before touching memory.
            if base
                .checked_add(nul)
                .is_none_or(|last| last >= self.memory.len())
            {
                return Err("text format: buffer out of bounds".into());
            }
            // Through the NUL, inside the buffer's one allocation.
            self.slots.live.check("text format", 1, base, nul + 1)?;
            for (k, &b) in bytes[..w.count].iter().enumerate() {
                self.memory[base + w.start + k] = crate::ml_io::decode_elem(DT_U8, &[b]);
            }
            self.memory[base + nul] = Value::Int(0);
        }
        Ok(Value::Int(i128::from(w.end)))
    }

    /// `n` bytes of the string at slot `base` (the call's operand `k`), or up to its NUL when
    /// `n < 0`. An explicit length must stay inside the string's allocation.
    fn text_read_bytes(&self, k: usize, base: usize, n: i64) -> Result<Vec<u8>, String> {
        let tail = self.memory.get(base..).unwrap_or(&[]);
        let len = match usize::try_from(n) {
            Ok(n) if n > tail.len() => return Err("text format: string out of bounds".into()),
            Ok(n) => {
                self.slots.live.check("text format", k, base, n)?;
                n
            }
            Err(_) => tail
                .iter()
                .position(|v| v.as_int() as u8 == 0)
                .unwrap_or(tail.len()),
        };
        Ok(tail[..len].iter().map(|v| v.as_int() as u8).collect())
    }

    /// The bytes `(data, len)` a `wukong_rt_utf8_*` entry reads (its operands 0 and 1): none for a
    /// null pointer or a non-positive length, as natively.
    fn utf8_operand(&self, args: &[Value]) -> Result<Vec<u8>, String> {
        match args.first() {
            Some(Value::Ptr(p)) if !self.is_null(*p) && int(args, 1) > 0 => {
                self.text_read_bytes(0, *p, int(args, 1))
            }
            _ => Ok(Vec::new()),
        }
    }

    /// `wukong_rt_fmt_str(target, buf, cap, pos, s, n, spec)`.
    fn text_fmt_str(&mut self, args: &[Value]) -> Result<Value, String> {
        let bytes = match args.get(4) {
            Some(Value::Ptr(p)) if !self.is_null(*p) => {
                self.text_read_bytes(4, *p, int(args, 5))?
            }
            _ => Vec::new(),
        };
        let mut out = Vec::new();
        fmt::format_str(&mut out, &bytes, Spec::unpack(int(args, 6)));
        self.text_emit(args, &out)
    }

    /// `wukong_rt_fmt_slice(target, buf, cap, pos, data, len, dtype, spec)`: one slot per element.
    fn text_fmt_slice(&mut self, args: &[Value]) -> Result<Value, String> {
        let spec = Spec::unpack(int(args, 7));
        let dt = int(args, 6);
        let base = match args.get(4) {
            Some(Value::Ptr(p)) if !self.is_null(*p) && fmt::elem_size(dt).is_some() => *p,
            _ => 0,
        };
        let len = if base == 0 {
            0
        } else {
            usize::try_from(int(args, 5)).unwrap_or(0)
        };
        let end = base
            .checked_add(len)
            .ok_or("text format: buffer extent overflow")?;
        if end > self.memory.len() {
            return Err("text format: buffer out of bounds".into());
        }
        self.slots.live.check("text format", 4, base, len)?;
        let slots = &self.memory[base..end];
        let mut out = Vec::new();
        fmt::format_seq(
            &mut out,
            len,
            |i| {
                let v = slots[i];
                fmt::elem_of(dt, v.as_int() as i64, v.as_float(), spec.style)
            },
            spec,
        );
        self.text_emit(args, &out)
    }
}

#[cfg(test)]
mod tests {
    use wukong_runtime::fmt::Spec;
    use wukong_sema::text::{self as st, FmtSpec};

    /// The compiler packs a placeholder's spec (`wukong_sema::text::FmtSpec::pack`) and the
    /// runtime unpacks it (`wukong_runtime::fmt::Spec::unpack`); the two crates do not depend on
    /// each other, so this pins every field of the shared layout, and the shared constants.
    #[test]
    fn compiler_spec_packing_matches_the_runtime() {
        use wukong_runtime::fmt as rt;
        let spec = FmtSpec {
            width: 12,
            precision: Some(3),
            fill: b'#',
            align: st::ALIGN_CENTER,
            plus: true,
            alt: true,
            zero: false,
            style: st::STYLE_UPPER_EXP,
        };
        for (kind, rt_kind) in [
            (st::KIND_SIGNED, rt::KIND_SIGNED),
            (st::KIND_UNSIGNED, rt::KIND_UNSIGNED),
            (st::KIND_BOOL, rt::KIND_BOOL),
            (st::KIND_CHAR, rt::KIND_CHAR),
        ] {
            let got = Spec::unpack(spec.pack(kind));
            assert_eq!(
                got,
                Spec {
                    width: 12,
                    precision: Some(3),
                    fill: b'#',
                    align: rt::ALIGN_CENTER,
                    plus: true,
                    alt: true,
                    zero: false,
                    style: rt::STYLE_UPPER_EXP,
                    kind: rt_kind,
                }
            );
        }
        for (a, b) in [
            (st::STYLE_DISPLAY, rt::STYLE_DISPLAY),
            (st::STYLE_DEBUG, rt::STYLE_DEBUG),
            (st::STYLE_LOWER_HEX, rt::STYLE_LOWER_HEX),
            (st::STYLE_UPPER_HEX, rt::STYLE_UPPER_HEX),
            (st::STYLE_OCTAL, rt::STYLE_OCTAL),
            (st::STYLE_BINARY, rt::STYLE_BINARY),
            (st::STYLE_LOWER_EXP, rt::STYLE_LOWER_EXP),
            (st::STYLE_UPPER_EXP, rt::STYLE_UPPER_EXP),
            (st::ALIGN_NONE, rt::ALIGN_NONE),
            (st::ALIGN_LEFT, rt::ALIGN_LEFT),
            (st::ALIGN_RIGHT, rt::ALIGN_RIGHT),
        ] {
            assert_eq!(a, b);
        }
        assert_eq!(st::PRINT_DT_BOOL, rt::DT_BOOL);
        assert_eq!(st::PRINT_DT_CHAR, rt::DT_CHAR);
        assert_eq!(st::SUMMARIZE_THRESHOLD as usize, rt::SUMMARIZE_THRESHOLD);
        assert_eq!(st::EDGE_ITEMS as usize, rt::EDGE_ITEMS);
        // Every printable scalar maps to a dtype the runtime knows the width of.
        for suffix in [
            "f32", "f64", "f16", "bf16", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
        ] {
            let s = wukong_sema::io_dtype(suffix).expect("an I/O dtype");
            assert!(rt::elem_size(st::print_dtype(s)).is_some(), "{suffix}");
        }
        assert_eq!(rt::elem_size(rt::DT_BOOL), Some(1));
        assert_eq!(rt::elem_size(rt::DT_CHAR), Some(4));
    }
}
