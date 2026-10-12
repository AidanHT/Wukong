//! Lowering of the decode builtins `wukong_sema::decode_builtins` types: `sdpa_kv`,
//! `sdpa_kv_batch`, `kv_append`, `gemv_bf16` / `gemv_f16`, `quantize_q8_0` / `quantize_q4_0` and
//! `gemv_q8_0` / `gemv_q4_0`.
//!
//! **Calls.** Each builtin lowers to one `Op::Call` of its `wukong_runtime` entry. A buffer operand —
//! a `[]T` slice, a `[T; N]` array or a contiguous `Tensor[T, …]` — becomes its data pointer and its
//! `i64` element count (the ML builtins' `ml_buffer`), an integer operand is widened to `i64` by its
//! own signedness, and the scale to `f32`; so every call site of one entry has the same MIR
//! signature, which the textual-LLVM backend relies on. Inside a `@parallel` function the
//! multicore twin is called (`sdpa_kv` and the GEMVs; each is bit-identical to its serial entry).
//!
//! **`sdpa_kv_batch`** is shape-checked by sema, which records its six dimensions `<B, SQ, SKV, H,
//! HKV, D>` like a named kernel builtin's ([`wukong_sema::BuiltinCall`]); `builtins.rs` hands the
//! recorded call to [`FnLowerer::lower_sdpa_kv_batch`], which passes each dimension as an `i64` (a
//! constant, a turbofish variable or generic dimension read through its binding, or a derived
//! extent evaluated from the bindings) after the six buffers. Its runtime checks the operand
//! lengths against those dimensions and every `kv_lens[b]` against `SKV`.
//!
//! **Shapes.** The runtime validates every extent against the operand lengths before it reads or
//! writes anything and returns a status; the lowering asserts that status is `0`, so an
//! out-of-range geometry traps exactly like a failed `assert` (exit 1, `assertion failed`) on every
//! backend — the interpreter runs the same validation and returns the same status. An output that
//! overlaps an input is handled by the entries themselves (value semantics), so no address test is
//! emitted here.
//!
//! **The half-weight GEMV recognizer.** The loop nest a decode step writes by hand,
//! `for j in 0..N { let s = 0.0; for p in 0..K { s = s + x[p] * (w[j*K + p] as f32) }; y[j] = s }`
//! over a `bf16`/`f16` weight, dispatches to the same `wukong_sgemv_{bf16,f16}` entry
//! ([`match_half_gemv`], [`FnLowerer::emit_half_gemv`]), behind the usual run-time alias guard and
//! empty-contraction test. The kernel computes each output with the f32 GEMM's per-element order —
//! for `K <= 384` exactly the nest's own FMA-contracted chain from `0.0`, beyond that K-blocked, the
//! documented reassociation every reduction kernel carries. The quantized GEMVs have no recognizer:
//! their per-call activation quantization is part of their definition, so no plain loop means them.

use wukong_ast::{self as ast, Block, Expr, ExprKind, ForIter, Pattern, StmtKind};
use wukong_mir::{CmpOp, MirType, Op, ValueId};
use wukong_sema::{decode_builtin, BuiltinCall, BuiltinDim, DecodeBuiltin, SemaResult};
use wukong_span::{Interner, Span, Symbol};
use wukong_types::Scalar;

use crate::alias::OperandPair;
use crate::kernel_guard::GuardOperand;
use crate::{
    as_dim, as_index1, as_int_lit, fusable_for, is_f32_expr, is_float_zero, match_row_col,
    range_bounds, scalar_of, single_path, Dim, FnLowerer,
};

/// The runtime entries this module lowers to, interned once at the top of `lower_program` and
/// carried on `GemmSyms` (`FnLowerer` holds only a shared `&Interner`).
#[derive(Clone, Copy)]
pub(crate) struct DecodeSyms {
    attn_kv: Symbol,
    attn_kv_par: Symbol,
    attn_kv_batch: Symbol,
    attn_kv_batch_par: Symbol,
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
            attn_kv_batch: interner.intern("wukong_attention_kv_batch_f32"),
            attn_kv_batch_par: interner.intern("wukong_attention_kv_batch_f32_parallel"),
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
        if b == DecodeBuiltin::SdpaKvBatch {
            // Lowered from sema's shape record (`lower_kernel_builtin`); a call without one never
            // type-checked.
            return None;
        }
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
            DecodeBuiltin::SdpaKvBatch => return None,
        };
        Some(self.assert_decode_status(status, span))
    }

    /// The entry ran iff it returned 0; a refused shape stops the program like a failed `assert`.
    /// Returns the call's unit value.
    fn assert_decode_status(&mut self, status: ValueId, span: Span) -> ValueId {
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
        self.const_zero(MirType::I32)
    }

    /// `sdpa_kv_batch(q, k, v, out, kv_lens, q_pos, scale, causal)` from sema's record `call`:
    /// `wukong_attention_kv_batch_f32[_parallel](q, nq, k, nk, v, nv, out, nout, kv_lens, nl, q_pos,
    /// np, B, SQ, SKV, H, HKV, D, scale, causal) -> i64`, its status asserted zero. Every operand is
    /// lowered once, in source order, before the dimensions are read.
    pub(crate) fn lower_sdpa_kv_batch(
        &mut self,
        e: &Expr,
        args: &[Expr],
        call: &BuiltinCall,
    ) -> ValueId {
        if args.len() != 8 || call.dims.len() != 6 {
            self.unsupported(e.span, "an `sdpa_kv_batch` call sema did not resolve");
            return self.const_zero(MirType::I32);
        }
        let mut bufs = Vec::with_capacity(6);
        for a in &args[..6] {
            let Some(b) = self.ml_buffer(a) else {
                self.unsupported(a.span, "an `sdpa_kv_batch` operand that is not a buffer");
                return self.const_zero(MirType::I32);
            };
            bufs.push(b);
        }
        let scale = self.ml_f32_arg(&args[6]);
        let causal = self.ml_i64_arg(&args[7]);
        let mut dims = Vec::with_capacity(6);
        for d in &call.dims {
            let v = match d {
                BuiltinDim::Const(n) => i64::try_from(*n).ok().map(|n| self.ml_i64_const(n)),
                &BuiltinDim::Named { name, signed } => self.lookup(name).map(|(slot, ty)| {
                    let v = self.builder.build(ty.clone(), Op::Load(slot, ty.clone()));
                    self.coerce_to(v, &ty, &MirType::I64, signed)
                }),
                BuiltinDim::Derived(dim) => self.tensor_dim_value(dim),
                BuiltinDim::SliceLen { .. } => None,
            };
            let Some(v) = v else {
                self.unsupported(
                    e.span,
                    "an `sdpa_kv_batch` dimension with no binding in scope or outside the \
                     kernel's signed 64-bit ABI",
                );
                return self.const_zero(MirType::I32);
            };
            dims.push(v);
        }
        let s = self.gemm.decode;
        let func = if self.parallel_fn {
            s.attn_kv_batch_par
        } else {
            s.attn_kv_batch
        };
        let mut callargs = Vec::with_capacity(20);
        for b in &bufs {
            callargs.push(b.data);
            callargs.push(b.len);
        }
        callargs.extend(dims);
        callargs.push(scale);
        callargs.push(causal);
        let status = self.ml_call(func, callargs);
        self.assert_decode_status(status, e.span)
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

    /// Emit the `wukong_sgemv_{bf16,f16}[_parallel]` call for a recognized half-weight GEMV nest, or
    /// decline (false) when an operand or extent is unbound here or the alias plan refuses, so the
    /// caller lowers the nest itself. The weight, activation and output extents the nest implies
    /// (`N·K`, `K`, `N`) are the call's lengths, so the entry's shape check passes for every
    /// non-negative `N`; the output must be disjoint from both inputs (versioned at run time when the
    /// compiler cannot tell).
    pub(crate) fn emit_half_gemv(&mut self, nest: &HalfGemvNest, parallel: bool) -> bool {
        let (Some(w), Some(x), Some(y)) = (
            self.kernel_base_ptr(nest.w),
            self.kernel_base_ptr(nest.x),
            self.kernel_base_ptr(nest.y),
        ) else {
            return false;
        };
        let (Some(n), Some(k)) = (self.dim_value(nest.n), self.dim_value(nest.k)) else {
            return false;
        };
        let w_elems = self.guard_extent(n, k);
        let half = if nest.f16 {
            MirType::F16
        } else {
            MirType::BF16
        };
        let ops = [
            GuardOperand::new(nest.w, w, w_elems, half),
            GuardOperand::new(nest.x, x, k, MirType::F32),
            GuardOperand::new(nest.y, y, n, MirType::F32),
        ];
        if !self.open_kernel_guard(
            &ops,
            &[OperandPair::disjoint(0, 2), OperandPair::disjoint(1, 2)],
        ) {
            return false;
        }
        let s = self.gemm.decode;
        let func = match (nest.f16, parallel) {
            (false, false) => s.sgemv_bf16,
            (false, true) => s.sgemv_bf16_par,
            (true, false) => s.sgemv_f16,
            (true, true) => s.sgemv_f16_par,
        };
        // A negative `N` (a loop that runs no row) is refused by the entry with nothing written —
        // exactly what the nest does; the status is otherwise always 0, so it is not checked.
        self.ml_call(func, vec![w, w_elems, x, k, y, n]);
        true
    }
}

/// A recognized half-weight GEMV nest `y[j] = Σ_p x[p]·(w[j·K + p] as f32)`: `w` a `bf16`/`f16`
/// buffer `[N·K]` (row-major `[N, K]`), `x` an `f32` buffer `[K]`, `y` an `f32` buffer `[N]`.
pub(crate) struct HalfGemvNest {
    w: Symbol,
    x: Symbol,
    y: Symbol,
    /// Rows: the outer bound.
    n: Dim,
    /// The contraction: the inner bound.
    k: Dim,
    /// `f16` weights (else `bf16`).
    f16: bool,
}

impl HalfGemvNest {
    /// The contraction extent, which the empty-contraction test keys on.
    pub(crate) fn k(&self) -> Dim {
        self.k
    }
}

/// One factor order of the product: `x_e` is `x[p]` (an `f32` element indexed by the inner variable
/// alone) and `w_e` is `w[row·K + p] as f32` over a `bf16`/`f16` buffer. Returns `(w, x, f16)`.
#[allow(clippy::too_many_arguments)]
fn half_gemv_operands(
    x_e: &Expr,
    w_e: &Expr,
    row: Symbol,
    p: Symbol,
    k: Dim,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(Symbol, Symbol, bool)> {
    let (x_sym, x_idx) = as_index1(x_e)?;
    if single_path(x_idx) != Some(p) || !is_f32_expr(x_e, sema) {
        return None;
    }
    let ExprKind::Cast { expr: inner, .. } = &w_e.kind else {
        return None;
    };
    if !is_f32_expr(w_e, sema) {
        return None;
    }
    let f16 = match scalar_of(inner, sema)? {
        Scalar::Bf16 => false,
        Scalar::F16 => true,
        _ => return None,
    };
    let (w_sym, w_idx) = as_index1(inner)?;
    let (stride, col) = match_row_col(w_idx, row, interner)?;
    if stride != k || col != p {
        return None;
    }
    Some((w_sym, x_sym, f16))
}

/// Recognize the half-weight GEMV nest rooted at `for j in 0..N { let s = 0.0; for p in 0..K {
/// s = s + x[p] * (w[j*K + p] as f32) }; y[j] = s }` — either factor order, `s += …` too, an `f32`
/// accumulator and an `f32` output, a plain store (no scale). See [`HalfGemvNest`].
pub(crate) fn match_half_gemv(
    pat: &Pattern,
    iter: &ForIter,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<HalfGemvNest> {
    let ast::PatKind::Ident(row) = &pat.kind else {
        return None;
    };
    let row = *row;
    let (start, end) = range_bounds(iter)?;
    if as_int_lit(start, interner)? != 0 {
        return None;
    }
    let n = as_dim(end, interner)?;
    if body.tail.is_some() || body.stmts.len() != 3 {
        return None;
    }
    let StmtKind::Let {
        pat: sp,
        init: Some(s0),
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    let ast::PatKind::Ident(s_sym) = &sp.kind else {
        return None;
    };
    let s_sym = *s_sym;
    if !is_float_zero(s0, interner) || !is_f32_expr(s0, sema) {
        return None;
    }
    let (ppat, piter, pbody) = fusable_for(&body.stmts[1])?;
    let ast::PatKind::Ident(p_sym) = &ppat.kind else {
        return None;
    };
    let p_sym = *p_sym;
    let (ps, pe) = range_bounds(piter)?;
    if as_int_lit(ps, interner)? != 0 {
        return None;
    }
    let k = as_dim(pe, interner)?;
    if pbody.tail.is_some() || pbody.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &pbody.stmts[0].kind else {
        return None;
    };
    if single_path(target) != Some(s_sym) {
        return None;
    }
    let prod = match op {
        ast::AssignOp::Add => value,
        ast::AssignOp::Assign => {
            let ExprKind::Binary {
                op: ast::BinOp::Add,
                lhs,
                rhs,
            } = &value.kind
            else {
                return None;
            };
            if single_path(lhs) != Some(s_sym) {
                return None;
            }
            rhs
        }
        _ => return None,
    };
    if !is_f32_expr(prod, sema) {
        return None;
    }
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &prod.kind
    else {
        return None;
    };
    let (w, x, f16) = half_gemv_operands(lhs, rhs, row, p_sym, k, sema, interner)
        .or_else(|| half_gemv_operands(rhs, lhs, row, p_sym, k, sema, interner))?;
    let StmtKind::Assign {
        target: yt,
        op: ast::AssignOp::Assign,
        value: yv,
    } = &body.stmts[2].kind
    else {
        return None;
    };
    if single_path(yv) != Some(s_sym) || !is_f32_expr(yt, sema) {
        return None;
    }
    let (y, y_idx) = as_index1(yt)?;
    if single_path(y_idx) != Some(row) || y == w || y == x {
        return None;
    }
    Some(HalfGemvNest { w, x, y, n, k, f16 })
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
            "fn decode_step(q: []f32, mut kc: []f32, mut vc: []f32, k: []f32, v: []f32, mut o: []f32, \
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

    /// `sdpa_kv_batch` lowers from sema's shape record: one checked call per site whose six
    /// dimensions follow the buffers — constants, a generic dimension read through its hidden
    /// parameter — and the multicore twin inside `@parallel`.
    #[test]
    fn sdpa_kv_batch_is_one_checked_call_with_its_recorded_dimensions() {
        let (text, diags) = mir(
            "fn decode_one<B, S>(q: Tensor[f32, B, 1, 4, 8], k: Tensor[f32, B, S, 2, 8],              v: Tensor[f32, B, S, 2, 8], mut o: Tensor[f32, B, 1, 4, 8], l: Tensor[i32, B],              p: Tensor[i32, B]) { sdpa_kv_batch(q, k, v, o, l, p, 0.35355339, true); }              fn flat(q: [f32; 64], k: [f32; 512], v: [f32; 512], mut o: [f32; 64],              l: [i32; 2], p: [i32; 2]) {              sdpa_kv_batch::<2, 1, 16, 4, 2, 8>(q, k, v, o, l, p, 0.5, 0); }              @parallel fn par(q: [f32; 64], k: [f32; 512], v: [f32; 512], mut o: [f32; 64],              l: [i32; 2], p: [i32; 2]) {              sdpa_kv_batch::<2, 1, 16, 4, 2, 8>(q, k, v, o, l, p, 0.5, 0); }",
        );
        assert!(diags.is_empty(), "{diags:?}");
        assert_eq!(
            text.matches("call wukong_attention_kv_batch_f32(").count(),
            2,
            "{text}"
        );
        assert_eq!(
            text.matches("call wukong_attention_kv_batch_f32_parallel(")
                .count(),
            1,
            "{text}"
        );
        // One status check per call, and no other extent check: the runtime validates the shapes.
        assert_eq!(text.matches("call assert(").count(), 3, "{text}");
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

    #[test]
    fn a_half_weight_gemv_nest_dispatches_to_the_half_gemv() {
        let (text, diags) = mir("fn main() -> i32 { let w: [bf16; 12] = [0.5 as bf16; 12]; \
             let h: [f16; 12] = [0.25 as f16; 12]; let x: [f32; 4] = [1.0; 4]; \
             let mut y: [f32; 3] = [0.0; 3]; let mut z: [f32; 3] = [0.0; 3]; \
             for j in 0..3 { let mut s: f32 = 0.0; \
               for p in 0..4 { s = s + x[p] * (w[j * 4 + p] as f32); } y[j] = s; } \
             for j in 0..3 { let mut s: f32 = 0.0; \
               for p in 0..4 { s += (h[j * 4 + p] as f32) * x[p]; } z[j] = s; } \
             return (y[0] + z[2]) as i32; }");
        assert!(diags.is_empty(), "{diags:?}");
        assert_eq!(text.matches("call wukong_sgemv_bf16(").count(), 1, "{text}");
        assert_eq!(text.matches("call wukong_sgemv_f16(").count(), 1, "{text}");
    }

    #[test]
    fn a_half_gemv_with_a_scaled_store_is_left_to_the_loop() {
        let (text, diags) = mir("fn main() -> i32 { let w: [bf16; 12] = [0.5 as bf16; 12]; \
             let x: [f32; 4] = [1.0; 4]; let mut y: [f32; 3] = [0.0; 3]; \
             for j in 0..3 { let mut s: f32 = 0.0; \
               for p in 0..4 { s = s + x[p] * (w[j * 4 + p] as f32); } y[j] = s * 2.0; } \
             return y[0] as i32; }");
        assert!(diags.is_empty(), "{diags:?}");
        assert!(!text.contains("wukong_sgemv_bf16"), "{text}");
    }
}
