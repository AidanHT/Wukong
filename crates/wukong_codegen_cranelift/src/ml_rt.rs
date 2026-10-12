//! The runtime entry points of the ML I/O and randomness builtins — dtype-coded file I/O,
//! `file_size`, the safetensors / `.npy` / GGUF readers and the GGUF tensor-table queries, the
//! Philox fill and the top-k / argsort / categorical-sampling kernels — and of the decode builtins
//! (KV-cache attention, the KV append, the half-weight and Q8_0 / Q4_0 GEMVs and packers),
//! declared, JIT-bound and called from **one table**.
//!
//! Every other runtime family in this crate is spelled out four times: a name constant, a
//! `Signature` built by hand in `populate_module`, an `rt_ids` row, and a `builder.symbol` line in
//! each of the two JIT entry points, plus a hand-written arm in `lower_call`. Forgetting the
//! `builder.symbol` line links fine for `--emit=obj` and fails only at JIT time; forgetting the
//! arity in `lower_call` is the silent-deletion hazard its trailing guard exists for. Here one
//! [`Entry`] row carries the name, the parameter ABI, the return ABI and the address, and the four
//! uses iterate it — a new entry point is one row, and the four uses cannot disagree.

use std::collections::HashMap;

use cranelift_codegen::ir::{types, AbiParam, InstBuilder, Signature, Type, Value};
use cranelift_codegen::isa::CallConv;
use cranelift_module::{FuncId, Linkage, Module};
use wukong_mir::ValueId;

use crate::FnTranslator;

/// The ABI class of one parameter or return value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Abi {
    /// A pointer (`ptr_ty`).
    Ptr,
    /// A 64-bit integer; any MIR integer argument is widened to it (sign-extending).
    I64,
    /// An `f32`; an `f64` argument is demoted to it.
    F32,
}

impl Abi {
    fn ty(self, ptr_ty: Type) -> Type {
        match self {
            Abi::Ptr => ptr_ty,
            Abi::I64 => types::I64,
            Abi::F32 => types::F32,
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

/// The table. The addresses are taken here, at run time, because a function-pointer-to-raw-pointer
/// cast is not a constant expression.
pub(crate) fn entries() -> [Entry; 24] {
    use Abi::{Ptr, F32, I64};
    [
        // (path, byte_off, data, len, dtype) -> i64: `read_<T>` outside the frozen six, and every
        // `read_<T>_at`.
        Entry {
            name: "wukong_rt_read_dt",
            params: &[Ptr, I64, Ptr, I64, I64],
            ret: Some(I64),
            addr: wukong_runtime::wukong_rt_read_dt as *const u8,
        },
        // (path, data, len, dtype) -> i64: every `write_<T>` outside the frozen six.
        Entry {
            name: "wukong_rt_write_dt",
            params: &[Ptr, Ptr, I64, I64],
            ret: Some(I64),
            addr: wukong_runtime::wukong_rt_write_dt as *const u8,
        },
        // (path) -> i64: `file_size(path)`.
        Entry {
            name: "wukong_rt_file_size",
            params: &[Ptr],
            ret: Some(I64),
            addr: wukong_runtime::wukong_rt_file_size as *const u8,
        },
        // (path, name, data, len, dtype, dims, rank) -> i64: `st_read_<T>`.
        Entry {
            name: "wukong_rt_st_read",
            params: &[Ptr, Ptr, Ptr, I64, I64, Ptr, I64],
            ret: Some(I64),
            addr: wukong_runtime::wukong_rt_st_read as *const u8,
        },
        // (path, data, len, dtype, dims, rank) -> i64: `npy_read_<T>`.
        Entry {
            name: "wukong_rt_npy_read",
            params: &[Ptr, Ptr, I64, I64, Ptr, I64],
            ret: Some(I64),
            addr: wukong_runtime::wukong_rt_npy_read as *const u8,
        },
        // (path, name, data, len, dtype, dims, rank) -> i64: `gguf_read_<T>`.
        Entry {
            name: "wukong_rt_gguf_read",
            params: &[Ptr, Ptr, Ptr, I64, I64, Ptr, I64],
            ret: Some(I64),
            addr: wukong_runtime::wukong_rt_gguf_read as *const u8,
        },
        // (path, name, what, k) -> i64: `gguf_find` / `gguf_rank` / `gguf_dim` / `gguf_type`.
        Entry {
            name: "wukong_rt_gguf_query",
            params: &[Ptr, Ptr, I64, I64],
            ret: Some(I64),
            addr: wukong_runtime::wukong_rt_gguf_query as *const u8,
        },
        // (out, n, seed, ctr0): `rand_fill_f32`.
        Entry {
            name: "wukong_rand_fill_f32",
            params: &[Ptr, I64, I64, I64],
            ret: None,
            addr: wukong_runtime::wukong_rand_fill_f32 as *const u8,
        },
        // (x, n, vals, nv, idx, ni) -> i64: `topk_f32`.
        Entry {
            name: "wukong_topk_f32",
            params: &[Ptr, I64, Ptr, I64, Ptr, I64],
            ret: Some(I64),
            addr: wukong_runtime::wukong_topk_f32 as *const u8,
        },
        // (x, n, idx, ni, descending) -> i64: `argsort_f32` / `argsort_desc_f32`.
        Entry {
            name: "wukong_argsort_f32",
            params: &[Ptr, I64, Ptr, I64, I64],
            ret: Some(I64),
            addr: wukong_runtime::wukong_argsort_f32 as *const u8,
        },
        // (p, n, u) -> i64: `sample_categorical`.
        Entry {
            name: "wukong_sample_categorical_f32",
            params: &[Ptr, I64, F32],
            ret: Some(I64),
            addr: wukong_runtime::wukong_sample_categorical_f32 as *const u8,
        },
        // The decode builtins (`wukong_mir_build`'s `decode_builtins.rs`). Every buffer travels
        // with its length and every entry returns a status the lowering asserts is zero.
        // (q, nq, k, nk, v, nv, o, no, sq, skv, d, q_pos0, n_heads, n_kv_heads, q_stride,
        // kv_stride, scale, causal) -> i64: `sdpa_kv`.
        Entry {
            name: "wukong_attention_kv_f32",
            params: ATTENTION_KV,
            ret: Some(I64),
            addr: wukong_runtime::wukong_attention_kv_f32 as *const u8,
        },
        Entry {
            name: "wukong_attention_kv_f32_parallel",
            params: ATTENTION_KV,
            ret: Some(I64),
            addr: wukong_runtime::wukong_attention_kv_f32_parallel as *const u8,
        },
        // (cache, ncache, row, nrow, pos, stride) -> i64: `kv_append`.
        Entry {
            name: "wukong_kv_append_f32",
            params: &[Ptr, I64, Ptr, I64, I64, I64],
            ret: Some(I64),
            addr: wukong_runtime::wukong_kv_append_f32 as *const u8,
        },
        // (w, nw, x, nx, y, ny) -> i64: `gemv_bf16` / `gemv_f16`.
        Entry {
            name: "wukong_sgemv_bf16",
            params: GEMV,
            ret: Some(I64),
            addr: wukong_runtime::wukong_sgemv_bf16 as *const u8,
        },
        Entry {
            name: "wukong_sgemv_bf16_parallel",
            params: GEMV,
            ret: Some(I64),
            addr: wukong_runtime::wukong_sgemv_bf16_parallel as *const u8,
        },
        Entry {
            name: "wukong_sgemv_f16",
            params: GEMV,
            ret: Some(I64),
            addr: wukong_runtime::wukong_sgemv_f16 as *const u8,
        },
        Entry {
            name: "wukong_sgemv_f16_parallel",
            params: GEMV,
            ret: Some(I64),
            addr: wukong_runtime::wukong_sgemv_f16_parallel as *const u8,
        },
        // (x, nx, q, nq, k) -> i64: `quantize_q8_0` / `quantize_q4_0`.
        Entry {
            name: "wukong_quantize_q8_0_f32",
            params: &[Ptr, I64, Ptr, I64, I64],
            ret: Some(I64),
            addr: wukong_runtime::wukong_quantize_q8_0_f32 as *const u8,
        },
        Entry {
            name: "wukong_quantize_q4_0_f32",
            params: &[Ptr, I64, Ptr, I64, I64],
            ret: Some(I64),
            addr: wukong_runtime::wukong_quantize_q4_0_f32 as *const u8,
        },
        // (w, nw, x, nx, y, ny) -> i64: `gemv_q8_0` / `gemv_q4_0`.
        Entry {
            name: "wukong_gemv_q8_0_f32",
            params: GEMV,
            ret: Some(I64),
            addr: wukong_runtime::wukong_gemv_q8_0_f32 as *const u8,
        },
        Entry {
            name: "wukong_gemv_q8_0_f32_parallel",
            params: GEMV,
            ret: Some(I64),
            addr: wukong_runtime::wukong_gemv_q8_0_f32_parallel as *const u8,
        },
        Entry {
            name: "wukong_gemv_q4_0_f32",
            params: GEMV,
            ret: Some(I64),
            addr: wukong_runtime::wukong_gemv_q4_0_f32 as *const u8,
        },
        Entry {
            name: "wukong_gemv_q4_0_f32_parallel",
            params: GEMV,
            ret: Some(I64),
            addr: wukong_runtime::wukong_gemv_q4_0_f32_parallel as *const u8,
        },
    ]
}

/// The `sdpa_kv` entry ABI: four `(buffer, length)` pairs, eight `i64` geometry values, the `f32`
/// scale and the `i64` causal flag.
const ATTENTION_KV: &[Abi] = &[
    Abi::Ptr,
    Abi::I64,
    Abi::Ptr,
    Abi::I64,
    Abi::Ptr,
    Abi::I64,
    Abi::Ptr,
    Abi::I64,
    Abi::I64,
    Abi::I64,
    Abi::I64,
    Abi::I64,
    Abi::I64,
    Abi::I64,
    Abi::I64,
    Abi::I64,
    Abi::F32,
    Abi::I64,
];

/// The checked GEMV ABI: weight, activation and output, each with its length.
const GEMV: &[Abi] = &[Abi::Ptr, Abi::I64, Abi::Ptr, Abi::I64, Abi::Ptr, Abi::I64];

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
    pub(crate) fn lower_ml_rt_call(
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
            assert!(e.name.starts_with("wukong_"), "{}", e.name);
            assert!(!e.addr.is_null(), "{}", e.name);
            assert!(
                all[..i].iter().all(|o| o.name != e.name),
                "duplicate entry {}",
                e.name
            );
        }
    }
}
