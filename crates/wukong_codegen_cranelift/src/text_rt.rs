//! The runtime entry points of the text builtins, declared, JIT-bound and called from **one
//! table** — the `ml_rt` pattern: one [`Entry`] row carries the symbol, its parameter and return
//! ABI and the address the JIT binds it to, and declaration, JIT binding and call lowering all
//! iterate the same rows, so they cannot disagree.
//!
//! The scalar printers (`wukong_rt_print_f32` / `_bool` / `_char`) are the per-type twins of
//! `wukong_rt_print_{i64,u64,f64,str}`: the JIT binds them to the capturing functions below, a
//! linked executable resolves them from the generated Rust shim or the C fallback runtime
//! (`wukong_driver`), and every one renders through `wukong_runtime::fmt`, the interpreter's
//! renderer.
//!
//! The formatting entries (`wukong_rt_fmt_*`: `printf`, `sprintf`, the structural `print` of an
//! aggregate) are bound to capturing twins too: a buffer write lands in the caller's memory exactly
//! as the runtime's own entry writes it, stdout goes to the run's captured output (so its bytes
//! interleave with every other print in program order), stderr to the process's stderr. A linked
//! executable resolves them from `wukong_runtime`'s rlib, which writes to the real streams.

use std::collections::HashMap;

use cranelift_codegen::ir::{types, AbiParam, InstBuilder, Signature, Type, Value};
use cranelift_codegen::isa::CallConv;
use cranelift_module::{FuncId, Linkage, Module};
use wukong_mir::ValueId;

use crate::FnTranslator;

/// `print`/`println` of an `f32` (or an `f16`/`bf16` in an `f32` register): `(x: f32)`.
pub(crate) const RT_PRINT_F32: &str = "wukong_rt_print_f32";
/// `print`/`println` of a `bool` (mir_build zero-extends it to `i64`): `(x: i64)`.
pub(crate) const RT_PRINT_BOOL: &str = "wukong_rt_print_bool";
/// `print`/`println` of a `char` (mir_build zero-extends it to `i64`): `(x: i64)`.
pub(crate) const RT_PRINT_CHAR: &str = "wukong_rt_print_char";

/// The ABI class of one parameter or return value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Abi {
    /// A pointer (`ptr_ty`). An absent pointer is the integer `0`, MIR's null, which has the
    /// pointer's width on every 64-bit target this backend emits for.
    Ptr,
    /// A 64-bit integer; any MIR integer argument is widened to it (sign-extending).
    I64,
    /// An `f32`; an `f64` argument is demoted to it.
    F32,
    /// An `f64`; a narrower float argument is widened to it (exactly).
    F64,
}

impl Abi {
    fn ty(self, ptr_ty: Type) -> Type {
        match self {
            Abi::Ptr => ptr_ty,
            Abi::I64 => types::I64,
            Abi::F32 => types::F32,
            Abi::F64 => types::F64,
        }
    }
}

/// One runtime entry point.
pub(crate) struct Entry {
    pub(crate) name: &'static str,
    pub(crate) params: &'static [Abi],
    pub(crate) ret: Option<Abi>,
    /// The in-process address the JIT binds `name` to.
    pub(crate) addr: *const u8,
}

// The JIT's capturing implementations. Each renders through `wukong_runtime::fmt` — the renderer
// the interpreter uses — and appends to the run's captured stdout.

extern "C" fn rt_print_f32(x: f32) {
    crate::emit_stdout(wukong_runtime::fmt::f32_line(x).as_bytes());
}

extern "C" fn rt_print_bool(x: i64) {
    crate::emit_stdout(wukong_runtime::fmt::bool_line(x).as_bytes());
}

extern "C" fn rt_print_char(x: i64) {
    crate::emit_stdout(wukong_runtime::fmt::char_line(x).as_bytes());
}

/// Deliver formatted `bytes` from a JIT run: a buffer write lands in the caller's memory
/// (`wukong_runtime::fmt::write_buffer`, the native entries' own), stdout in the run's captured
/// output, stderr on the process's stderr. Returns the position after the bytes.
///
/// # Safety
/// For a buffer target, `buf` must be valid for writes of `cap` bytes when `cap > 0`.
unsafe fn jit_emit(target: i64, buf: *mut u8, cap: i64, pos: i64, bytes: &[u8]) -> i64 {
    use wukong_runtime::fmt;
    if target == fmt::TARGET_BUFFER {
        // SAFETY: forwarded from this function's contract.
        return unsafe { fmt::write_buffer(buf, cap, pos, bytes) };
    }
    if target == fmt::TARGET_STDERR {
        fmt::write_stream(target, bytes);
    } else {
        crate::emit_stdout(bytes);
    }
    pos.saturating_add(i64::try_from(bytes.len()).unwrap_or(i64::MAX))
}

extern "C" fn rt_fmt_int(target: i64, buf: *mut u8, cap: i64, pos: i64, x: i64, spec: i64) -> i64 {
    let mut s = Vec::new();
    wukong_runtime::fmt::format_int(&mut s, x, wukong_runtime::fmt::Spec::unpack(spec));
    // SAFETY: the generated code passes a buffer of `cap` writable bytes, or a stream target.
    unsafe { jit_emit(target, buf, cap, pos, &s) }
}

extern "C" fn rt_fmt_f64(target: i64, buf: *mut u8, cap: i64, pos: i64, x: f64, spec: i64) -> i64 {
    let mut s = Vec::new();
    wukong_runtime::fmt::format_f64(&mut s, x, wukong_runtime::fmt::Spec::unpack(spec));
    // SAFETY: as in `rt_fmt_int`.
    unsafe { jit_emit(target, buf, cap, pos, &s) }
}

extern "C" fn rt_fmt_f32(target: i64, buf: *mut u8, cap: i64, pos: i64, x: f32, spec: i64) -> i64 {
    let mut s = Vec::new();
    wukong_runtime::fmt::format_f32(&mut s, x, wukong_runtime::fmt::Spec::unpack(spec));
    // SAFETY: as in `rt_fmt_int`.
    unsafe { jit_emit(target, buf, cap, pos, &s) }
}

extern "C" fn rt_fmt_str(
    target: i64,
    buf: *mut u8,
    cap: i64,
    pos: i64,
    s: *const u8,
    n: i64,
    spec: i64,
) -> i64 {
    // SAFETY: the generated code passes `n` readable bytes at `s`, or a NUL-terminated string
    // when `n < 0`.
    let bytes = unsafe { wukong_runtime::fmt::str_operand(s, n) };
    let mut out = Vec::new();
    wukong_runtime::fmt::format_str(&mut out, bytes, wukong_runtime::fmt::Spec::unpack(spec));
    // SAFETY: as in `rt_fmt_int`.
    unsafe { jit_emit(target, buf, cap, pos, &out) }
}

extern "C" fn rt_fmt_lit(
    target: i64,
    buf: *mut u8,
    cap: i64,
    pos: i64,
    packed: i64,
    n: i64,
) -> i64 {
    let (bytes, n) = wukong_runtime::fmt::lit_operand(packed, n);
    // SAFETY: as in `rt_fmt_int`.
    unsafe { jit_emit(target, buf, cap, pos, &bytes[..n]) }
}

// One parameter per C-ABI operand, deliberately: the entry mirrors `wukong_rt_fmt_slice`.
#[allow(clippy::too_many_arguments)]
extern "C" fn rt_fmt_slice(
    target: i64,
    buf: *mut u8,
    cap: i64,
    pos: i64,
    data: *const u8,
    len: i64,
    dtype: i64,
    spec: i64,
) -> i64 {
    use wukong_runtime::fmt;
    let spec = fmt::Spec::unpack(spec);
    let len = if data.is_null() || fmt::elem_size(dtype).is_none() {
        0
    } else {
        usize::try_from(len).unwrap_or(0)
    };
    let mut out = Vec::new();
    // SAFETY: the generated code passes `len` readable elements of `dtype` at `data`.
    fmt::format_seq(
        &mut out,
        len,
        |i| unsafe { fmt::read_elem(data, i, dtype, spec.style) },
        spec,
    );
    // SAFETY: as in `rt_fmt_int`.
    unsafe { jit_emit(target, buf, cap, pos, &out) }
}

/// The table. The addresses are taken here, at run time, because a function-pointer-to-raw-pointer
/// cast is not a constant expression.
pub(crate) fn entries() -> [Entry; 9] {
    use Abi::{Ptr, F32, F64, I64};
    [
        Entry {
            name: RT_PRINT_F32,
            params: &[F32],
            ret: None,
            addr: rt_print_f32 as *const u8,
        },
        Entry {
            name: RT_PRINT_BOOL,
            params: &[I64],
            ret: None,
            addr: rt_print_bool as *const u8,
        },
        Entry {
            name: RT_PRINT_CHAR,
            params: &[I64],
            ret: None,
            addr: rt_print_char as *const u8,
        },
        // Every formatting entry leads with the sink `(target, buf, cap, pos)` and returns the
        // position after its output.
        Entry {
            name: "wukong_rt_fmt_int",
            params: &[I64, Ptr, I64, I64, I64, I64],
            ret: Some(I64),
            addr: rt_fmt_int as *const u8,
        },
        Entry {
            name: "wukong_rt_fmt_f64",
            params: &[I64, Ptr, I64, I64, F64, I64],
            ret: Some(I64),
            addr: rt_fmt_f64 as *const u8,
        },
        Entry {
            name: "wukong_rt_fmt_f32",
            params: &[I64, Ptr, I64, I64, F32, I64],
            ret: Some(I64),
            addr: rt_fmt_f32 as *const u8,
        },
        Entry {
            name: "wukong_rt_fmt_str",
            params: &[I64, Ptr, I64, I64, Ptr, I64, I64],
            ret: Some(I64),
            addr: rt_fmt_str as *const u8,
        },
        Entry {
            name: "wukong_rt_fmt_lit",
            params: &[I64, Ptr, I64, I64, I64, I64],
            ret: Some(I64),
            addr: rt_fmt_lit as *const u8,
        },
        Entry {
            name: "wukong_rt_fmt_slice",
            params: &[I64, Ptr, I64, I64, Ptr, I64, I64, I64],
            ret: Some(I64),
            addr: rt_fmt_slice as *const u8,
        },
    ]
}

/// Declare every table entry as an import and record it in `rt_ids` — the module-wide symbol set
/// that both resolves a runtime call and arms `lower_call`'s arity guard.
pub(crate) fn declare<M: Module>(
    module: &mut M,
    ptr_ty: Type,
    call_conv: CallConv,
    rt_ids: &mut HashMap<&'static str, FuncId>,
) -> Result<(), String> {
    for e in entries() {
        let mut sig = Signature::new(call_conv);
        for p in e.params {
            sig.params.push(AbiParam::new(p.ty(ptr_ty)));
        }
        if let Some(r) = e.ret {
            sig.returns.push(AbiParam::new(r.ty(ptr_ty)));
        }
        let id = module
            .declare_function(e.name, Linkage::Import, &sig)
            .map_err(|err| err.to_string())?;
        rt_ids.insert(e.name, id);
    }
    Ok(())
}

/// Bind every table entry to its in-process implementation for the JIT.
pub(crate) fn bind_jit(builder: &mut cranelift_jit::JITBuilder) {
    for e in entries() {
        builder.symbol(e.name, e.addr);
    }
}

impl FnTranslator<'_> {
    /// Lower a call to a table entry: `None` when `name` is not in the table or the arity does not
    /// match (the caller's guard then reports the mismatch), otherwise `Some(result)`.
    pub(crate) fn lower_text_rt_call(
        &mut self,
        name: &str,
        args: &[ValueId],
    ) -> Option<Option<Value>> {
        let e = entries().into_iter().find(|e| e.name == name)?;
        if e.params.len() != args.len() {
            return None;
        }
        let mut vals = Vec::with_capacity(args.len());
        for (&p, &a) in e.params.iter().zip(args) {
            vals.push(match p {
                Abi::Ptr => self.val(a),
                Abi::I64 => self.coerce_to_i64(a),
                Abi::F32 => self.coerce_to_f32(a),
                Abi::F64 => self.coerce_to_f64(a),
            });
        }
        let fref = self.runtime_ref(e.name);
        let call = self.builder.ins().call(fref, &vals);
        Some(if e.ret.is_some() {
            self.builder.inst_results(call).first().copied()
        } else {
            None
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_is_named_once_and_bound() {
        let all = entries();
        for (i, e) in all.iter().enumerate() {
            assert!(e.name.starts_with("wukong_rt_"), "{}", e.name);
            assert!(!e.addr.is_null(), "{}", e.name);
            assert!(
                all[..i].iter().all(|o| o.name != e.name),
                "duplicate entry {}",
                e.name
            );
        }
        // No row may collide with an `ml_rt` row: both tables feed the one `rt_ids` map.
        for m in crate::ml_rt::entries() {
            assert!(all.iter().all(|e| e.name != m.name), "{}", m.name);
        }
    }
}
