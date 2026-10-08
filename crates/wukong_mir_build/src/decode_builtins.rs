//! Lowering of the decode builtins `wukong_sema::decode_builtins` types: `sdpa_kv`, `kv_append`,
//! `gemv_bf16` / `gemv_f16`, `quantize_q8_0` / `quantize_q4_0` and `gemv_q8_0` / `gemv_q4_0`.
//!
//! **Calls.** Each builtin lowers to one `Op::Call` of its `wukong_runtime` entry. A buffer operand —
//! a `[]T` slice, a `[T; N]` array or a contiguous `Tensor[T, …]` — becomes its data pointer and its
//! `i64` element count (the ML builtins' `ml_buffer`), an integer operand is widened to `i64` by its
//! own signedness, and the scale to `f32`; so every call site of one entry has the same MIR
//! signature, which the textual-LLVM backend relies on. Inside a `@parallel` function the
//! multicore twin is called (`sdpa_kv` and the GEMVs; each is bit-identical to its serial entry).
//!
//! **Shapes.** The runtime validates every extent against the operand lengths before it reads or
//! writes anything and returns a status; the lowering asserts that status is `0`, so an
//! out-of-range geometry traps exactly like a failed `assert` (exit 1, `assertion failed`) on every
//! backend — the interpreter runs the same validation and returns the same status. An output that
//! overlaps an input is handled by the entries themselves (value semantics), so no address test is
//! emitted here.

use wukong_ast::Expr;
use wukong_mir::{CmpOp, MirType, Op, ValueId};
use wukong_sema::{decode_builtin, DecodeBuiltin};
use wukong_span::{Interner, Span, Symbol};
use wukong_types::Scalar;

use crate::FnLowerer;

/// The runtime entries this module lowers to, interned once at the top of `lower_program` and
/// carried on `GemmSyms` (`FnLowerer` holds only a shared `&Interner`).
#[derive(Clone, Copy)]
pub(crate) struct DecodeSyms {
    attn_kv: Symbol,
    attn_kv_par: Symbol,
    kv_append: Symbol,
    sgemv_bf16: Symbol,
    sgemv_bf16_par: Symbol,
    sgemv_f16: Symbol,
    sgemv_f16_par: Symbol,
    gemv_q8: Symbol,
    gemv_q8_par: Symbol,
    gemv_q4: Symbol,
    gemv_q4_par: Symbol,
    quant_q8: Symbol,
    quant_q4: Symbol,
}

impl DecodeSyms {
    pub(crate) fn new(interner: &mut Interner) -> DecodeSyms {
        DecodeSyms {
            attn_kv: interner.intern("wukong_attention_kv_f32"),
            attn_kv_par: interner.intern("wukong_attention_kv_f32_parallel"),
            kv_append: interner.intern("wukong_kv_append_f32"),
            sgemv_bf16: interner.intern("wukong_sgemv_bf16"),
            sgemv_bf16_par: interner.intern("wukong_sgemv_bf16_parallel"),
            sgemv_f16: interner.intern("wukong_sgemv_f16"),
            sgemv_f16_par: interner.intern("wukong_sgemv_f16_parallel"),
            gemv_q8: interner.intern("wukong_gemv_q8_0_f32"),
            gemv_q8_par: interner.intern("wukong_gemv_q8_0_f32_parallel"),
            gemv_q4: interner.intern("wukong_gemv_q4_0_f32"),
            gemv_q4_par: interner.intern("wukong_gemv_q4_0_f32_parallel"),
            quant_q8: interner.intern("wukong_quantize_q8_0_f32"),
            quant_q4: interner.intern("wukong_quantize_q4_0_f32"),
        }
    }
}

impl FnLowerer<'_> {
    /// Lower a call to a decode builtin, or return `None` for any other callee (or an operand shape
    /// sema should have rejected), so `lower_call` falls through to its other paths. A user function
    /// of the same name never reaches here: `lower_call` resolves it first.
    pub(crate) fn lower_decode_builtin(
        &mut self,
        name: Symbol,
        args: &[Expr],
        span: Span,
    ) -> Option<ValueId> {
        let b = decode_builtin(self.interner.resolve(name))?;
        let s = self.gemm.decode;
        let par = self.parallel_fn;
        let pick = |serial: Symbol, parallel: Symbol| if par { parallel } else { serial };
        let status = match b {
            DecodeBuiltin::SdpaKv => {
                if args.len() != 14 {
                    return None;
                }
                let mut call = Vec::with_capacity(18);
                for a in &args[..4] {
                    let buf = self.ml_buffer(a)?;
                    call.push(buf.data);
                    call.push(buf.len);
                }
                for a in &args[4..12] {
                    let v = self.ml_i64_arg(a);
                    call.push(v);
                }
                let scale = self.ml_f32_arg(&args[12]);
                call.push(scale);
                // `causal`: an integer or a `bool`, nonzero meaning causal.
                let causal = self.ml_i64_arg(&args[13]);
                call.push(causal);
                self.ml_call(pick(s.attn_kv, s.attn_kv_par), call)
            }
            DecodeBuiltin::KvAppend => {
                if args.len() != 4 {
                    return None;
                }
                let cache = self.ml_buffer(&args[0])?;
                let row = self.ml_buffer(&args[1])?;
                let pos = self.ml_i64_arg(&args[2]);
                let stride = self.ml_i64_arg(&args[3]);
                self.ml_call(
                    s.kv_append,
                    vec![cache.data, cache.len, row.data, row.len, pos, stride],
                )
            }
            DecodeBuiltin::GemvHalf(t) => {
                let func = if t == Scalar::F16 {
                    pick(s.sgemv_f16, s.sgemv_f16_par)
                } else {
                    pick(s.sgemv_bf16, s.sgemv_bf16_par)
                };
                self.lower_decode_gemv(func, args)?
            }
            DecodeBuiltin::GemvQ8 => {
                self.lower_decode_gemv(pick(s.gemv_q8, s.gemv_q8_par), args)?
            }
            DecodeBuiltin::GemvQ4 => {
                self.lower_decode_gemv(pick(s.gemv_q4, s.gemv_q4_par), args)?
            }
            DecodeBuiltin::QuantizeQ8 | DecodeBuiltin::QuantizeQ4 => {
                if args.len() != 3 {
                    return None;
                }
                let x = self.ml_buffer(&args[0])?;
                let q = self.ml_buffer(&args[1])?;
                let k = self.ml_i64_arg(&args[2]);
                let func = if b == DecodeBuiltin::QuantizeQ8 {
                    s.quant_q8
                } else {
                    s.quant_q4
                };
                self.ml_call(func, vec![x.data, x.len, q.data, q.len, k])
            }
        };
        // The entry ran iff it returned 0; a refused shape stops the program like a failed
        // `assert`.
        let zero = self.ml_i64_const(0);
        let ok = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Eq, status, zero));
        if !self.emit_builtin_checks(vec![ok]) {
            self.unsupported(
                span,
                "a decode builtin in a program that defines its own `assert`",
            );
        }
        Some(self.const_zero(MirType::I32))
    }

    /// `func(w, len(w), x, len(x), y, len(y)) -> i64` — the checked GEMV ABI shared by the half and
    /// quantized forms.
    fn lower_decode_gemv(&mut self, func: Symbol, args: &[Expr]) -> Option<ValueId> {
        if args.len() != 3 {
            return None;
        }
        let w = self.ml_buffer(&args[0])?;
        let x = self.ml_buffer(&args[1])?;
        let y = self.ml_buffer(&args[2])?;
        Some(self.ml_call(func, vec![w.data, w.len, x.data, x.len, y.data, y.len]))
    }
}

#[cfg(test)]
mod tests {
    use crate::lower_program;
    use wukong_span::{Interner, SourceId};

    /// Lower `src` (which must type-check) and return its verified MIR text and the diagnostics.
    fn mir(src: &str) -> (String, Vec<wukong_diag::Diagnostic>) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (prog, diags) = lower_program(&module, &sema, &mut interner);
        let errors = wukong_mir::verify::verify_program(&prog);
        assert!(errors.is_empty(), "invalid MIR: {errors:?}");
        (wukong_mir::print::print_program(&prog, &interner), diags)
    }

    #[test]
    fn each_builtin_is_one_checked_runtime_call() {
        let (text, diags) = mir(
            "fn step(q: []f32, mut kc: []f32, mut vc: []f32, k: []f32, v: []f32, mut o: []f32, \
             t: i64) { \
             kv_append(kc, k, t, 16); kv_append(vc, v, t, 16); \
             sdpa_kv(q, kc, vc, o, 1, t + 1, 8, t, 4, 2, 32, 16, 0.25, true); } \
             fn proj(w: []bf16, h: []f16, wq: []u8, x: []f32, mut y: []f32, mut q: []u8) { \
             gemv_bf16(w, x, y); gemv_f16(h, x, y); \
             quantize_q8_0(x, q, 32); gemv_q8_0(wq, x, y); \
             quantize_q4_0(x, q, 32); gemv_q4_0(wq, x, y); }",
        );
        assert!(diags.is_empty(), "{diags:?}");
        for (sym, count) in [
            ("wukong_kv_append_f32", 2),
            ("wukong_attention_kv_f32", 1),
            ("wukong_sgemv_bf16", 1),
            ("wukong_sgemv_f16", 1),
            ("wukong_quantize_q8_0_f32", 1),
            ("wukong_gemv_q8_0_f32", 1),
            ("wukong_quantize_q4_0_f32", 1),
            ("wukong_gemv_q4_0_f32", 1),
        ] {
            assert_eq!(
                text.matches(&format!("call {sym}(")).count(),
                count,
                "{sym}: {text}"
            );
        }
        // One status check per call.
        assert_eq!(text.matches("call assert(").count(), 9, "{text}");
        assert!(!text.contains("_parallel("), "{text}");
    }

    #[test]
    fn a_parallel_function_calls_the_multicore_twins() {
        let (text, diags) = mir(
            "@parallel fn proj(w: []bf16, wq: []u8, x: []f32, mut y: []f32) { \
             gemv_bf16(w, x, y); gemv_q4_0(wq, x, y); }",
        );
        assert!(diags.is_empty(), "{diags:?}");
        assert!(text.contains("call wukong_sgemv_bf16_parallel("), "{text}");
        assert!(
            text.contains("call wukong_gemv_q4_0_f32_parallel("),
            "{text}"
        );
    }
}
