//! `mercury_mir_build` — lowers the type-checked AST into MIR.
//!
//! Strategy: every local (and every parameter) gets a stack slot via `alloca`; reads `load` and
//! writes `store`. This keeps lowering simple and non-recursive in its SSA reasoning — a later
//! `mem2reg` pass promotes the slots to SSA registers. Control flow is lowered directly to a CFG
//! of basic blocks with `br`/`cond_br`.
//!
//! The lowerer covers the scalar + pointer + control-flow + direct-call core end to end. Tensor,
//! SIMD-method, and parallel-loop constructs are not yet lowered; encountering one records a
//! diagnostic and substitutes a placeholder so the rest of the function still lowers.

use std::collections::{HashMap, HashSet};

use mercury_ast::{
    self as ast, Block, Expr, ExprKind, FnDecl, ForIter, Module, Pattern, Stmt, StmtKind,
};
use mercury_diag::Diagnostic;
use mercury_mir::{
    BinOp, Builder, CastKind, CmpOp, Function, MirType, Op, Program, RoundMode, ValueId,
};
use mercury_sema::{DefKind, SemaResult};
use mercury_span::{Interner, Span, Symbol};
use mercury_types::Ty;

/// Array-repeat initializers (`[v; n]`) with at most this many elements are unrolled to
/// straight-line stores; larger ones lower to a fill loop to keep the IR compact.
const REPEAT_UNROLL_LIMIT: u32 = 8;

/// Lower a whole module to a MIR [`Program`]. Only functions with bodies are lowered.
pub fn lower_program(
    module: &Module,
    sema: &SemaResult,
    interner: &mut Interner,
) -> (Program, Vec<Diagnostic>) {
    let mut diags = Vec::new();
    let mut program = Program::new();
    // Runtime symbols the matmul recognizer lowers a GEMM nest to (interned once, threaded down).
    let gemm = GemmSyms {
        mm: interner.intern("mercury_sgemm"),
        mm_par: interner.intern("mercury_sgemm_parallel"),
        print_str: interner.intern("print_str"),
        println_str: interner.intern("println_str"),
        print_u: interner.intern("print_u"),
        println_u: interner.intern("println_u"),
        nt: interner.intern("mercury_sgemm_nt"),
        nt_par: interner.intern("mercury_sgemm_nt_parallel"),
        tn: interner.intern("mercury_sgemm_tn"),
        tn_par: interner.intern("mercury_sgemm_tn_parallel"),
        nt_epi: interner.intern("mercury_sgemm_nt_epi"),
        nt_epi_par: interner.intern("mercury_sgemm_nt_epi_parallel"),
        vmath: interner.intern("mercury_vmath_f32"),
        vmath2: interner.intern("mercury_vmath2_f32"),
        vmath_bf16: interner.intern("mercury_vmath_bf16"),
        vmath_f16: interner.intern("mercury_vmath_f16"),
        velem: interner.intern("mercury_velem_f32"),
        vhorner: interner.intern("mercury_vhorner_f32"),
        sred_par: interner.intern("mercury_sreduce_f32_parallel"),
        argreduce: interner.intern("mercury_argreduce_f32"),
        argreduce_par: interner.intern("mercury_argreduce_f32_parallel"),
        norm: interner.intern("mercury_norm_f32"),
        norm_par: interner.intern("mercury_norm_f32_parallel"),
        norm_affine: interner.intern("mercury_norm_affine_f32"),
        norm_affine_par: interner.intern("mercury_norm_affine_f32_parallel"),
        i8nt: interner.intern("mercury_i8gemm_nt"),
        i8nt_par: interner.intern("mercury_i8gemm_nt_parallel"),
        i8deq: interner.intern("mercury_i8gemm_nt_deq"),
        i8deq_par: interner.intern("mercury_i8gemm_nt_deq_parallel"),
        bf16_nt: interner.intern("mercury_sgemm_bf16_nt"),
        bf16_nt_par: interner.intern("mercury_sgemm_bf16_nt_parallel"),
        f16_nt: interner.intern("mercury_sgemm_f16_nt"),
        f16_nt_par: interner.intern("mercury_sgemm_f16_nt_parallel"),
        bf16_nt_epi: interner.intern("mercury_sgemm_bf16_nt_epi"),
        bf16_nt_epi_par: interner.intern("mercury_sgemm_bf16_nt_epi_parallel"),
        f16_nt_epi: interner.intern("mercury_sgemm_f16_nt_epi"),
        f16_nt_epi_par: interner.intern("mercury_sgemm_f16_nt_epi_parallel"),
        bf16_tn: interner.intern("mercury_sgemm_bf16_tn"),
        bf16_tn_par: interner.intern("mercury_sgemm_bf16_tn_parallel"),
        f16_tn: interner.intern("mercury_sgemm_f16_tn"),
        f16_tn_par: interner.intern("mercury_sgemm_f16_tn_parallel"),
        dot_bf16: interner.intern("mercury_dot_bf16"),
        sum_bf16: interner.intern("mercury_sum_bf16"),
        reduce_bf16: interner.intern("mercury_reduce_bf16"),
        dot_f16: interner.intern("mercury_dot_f16"),
        sum_f16: interner.intern("mercury_sum_f16"),
        reduce_f16: interner.intern("mercury_reduce_f16"),
        axpby_bf16: interner.intern("mercury_axpby_bf16"),
        axpby_f16: interner.intern("mercury_axpby_f16"),
        transpose: interner.intern("mercury_transpose_f32"),
        transpose_par: interner.intern("mercury_transpose_f32_parallel"),
        transpose_u16: interner.intern("mercury_transpose_u16"),
        transpose_u16_par: interner.intern("mercury_transpose_u16_parallel"),
        colsum: interner.intern("mercury_colsum_f32"),
        colsum_par: interner.intern("mercury_colsum_f32_parallel"),
        colmax: interner.intern("mercury_colmax_f32"),
        colmax_par: interner.intern("mercury_colmax_f32_parallel"),
        colmin: interner.intern("mercury_colmin_f32"),
        colmin_par: interner.intern("mercury_colmin_f32_parallel"),
        colmaxabs: interner.intern("mercury_colmaxabs_f32"),
        colmaxabs_par: interner.intern("mercury_colmaxabs_f32_parallel"),
        colmean: interner.intern("mercury_colmean_f32"),
        colmean_par: interner.intern("mercury_colmean_f32_parallel"),
        colsumsq: interner.intern("mercury_colsumsq_f32"),
        colsumsq_par: interner.intern("mercury_colsumsq_f32_parallel"),
        coll2: interner.intern("mercury_coll2_f32"),
        coll2_par: interner.intern("mercury_coll2_f32_parallel"),
        colrms: interner.intern("mercury_colrms_f32"),
        colrms_par: interner.intern("mercury_colrms_f32_parallel"),
        softmax_bwd: interner.intern("mercury_softmax_bwd_f32"),
        softmax_bwd_par: interner.intern("mercury_softmax_bwd_f32_parallel"),
        rmsnorm_bwd: interner.intern("mercury_rmsnorm_bwd_f32"),
        rmsnorm_bwd_par: interner.intern("mercury_rmsnorm_bwd_f32_parallel"),
        layernorm_bwd: interner.intern("mercury_layernorm_bwd_f32"),
        layernorm_bwd_par: interner.intern("mercury_layernorm_bwd_f32_parallel"),
        xent: interner.intern("mercury_xent_fwd_f32"),
        xent_par: interner.intern("mercury_xent_fwd_f32_parallel"),
        xent_bwd: interner.intern("mercury_xent_bwd_f32"),
        xent_bwd_par: interner.intern("mercury_xent_bwd_f32_parallel"),
        rope: interner.intern("mercury_rope_f32"),
        rope_par: interner.intern("mercury_rope_f32_parallel"),
        rope_bwd: interner.intern("mercury_rope_bwd_f32"),
        rope_bwd_par: interner.intern("mercury_rope_bwd_f32_parallel"),
        logsumexp: interner.intern("mercury_logsumexp_f32"),
        logsumexp_par: interner.intern("mercury_logsumexp_f32_parallel"),
        kldiv: interner.intern("mercury_kldiv_f32"),
        kldiv_par: interner.intern("mercury_kldiv_f32_parallel"),
        entropy: interner.intern("mercury_entropy_f32"),
        entropy_par: interner.intern("mercury_entropy_f32_parallel"),
        kd_loss: interner.intern("mercury_kd_loss_f32"),
        kd_loss_par: interner.intern("mercury_kd_loss_f32_parallel"),
        rowargmax: interner.intern("mercury_rowargmax_i32"),
        rowargmax_par: interner.intern("mercury_rowargmax_i32_parallel"),
        rowargmin: interner.intern("mercury_rowargmin_i32"),
        rowargmin_par: interner.intern("mercury_rowargmin_i32_parallel"),
        colargmax: interner.intern("mercury_colargmax_i32"),
        colargmax_par: interner.intern("mercury_colargmax_i32_parallel"),
        colargmin: interner.intern("mercury_colargmin_i32"),
        colargmin_par: interner.intern("mercury_colargmin_i32_parallel"),
        cumsum: interner.intern("mercury_cumsum_f32"),
        cumsum_par: interner.intern("mercury_cumsum_f32_parallel"),
        cumprod: interner.intern("mercury_cumprod_f32"),
        cumprod_par: interner.intern("mercury_cumprod_f32_parallel"),
        lrscan: interner.intern("mercury_lrscan_f32"),
        lrscan_par: interner.intern("mercury_lrscan_f32_parallel"),
        cummax: interner.intern("mercury_cummax_f32"),
        cummax_par: interner.intern("mercury_cummax_f32_parallel"),
        cummin: interner.intern("mercury_cummin_f32"),
        cummin_par: interner.intern("mercury_cummin_f32_parallel"),
        embedding: interner.intern("mercury_embedding_f32"),
        embedding_par: interner.intern("mercury_embedding_f32_parallel"),
        scatter_add: interner.intern("mercury_scatter_add_f32"),
        scatter_add_par: interner.intern("mercury_scatter_add_f32_parallel"),
        maxpool2d: interner.intern("mercury_maxpool2d_f32"),
        maxpool2d_par: interner.intern("mercury_maxpool2d_f32_parallel"),
        avgpool2d: interner.intern("mercury_avgpool2d_f32"),
        avgpool2d_par: interner.intern("mercury_avgpool2d_f32_parallel"),
    };
    for item in &module.items {
        if let ast::ItemKind::Fn(f) = &item.kind {
            if let Some(body) = &f.body {
                // Whole-function matmul: lower the entire nest to a single (optionally parallel)
                // `mercury_sgemm` call — the tuned 256-bit AVX2/FMA microkernel.
                if let Some(nest) = matmul_fn(body, sema, interner) {
                    let parallel = has_parallel_attr(item, interner);
                    let func =
                        lower_matmul_fn(f, &nest, parallel, sema, interner, gemm, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // Whole-function int8 quantized matmul (`u8×i8→i32` `C = A·Bᵀ`) → the int8 GEMM
                // microkernel. Checked *before* the `@parallel` outliner below so a `@parallel` int8
                // kernel dispatches to the multicore `mercury_i8gemm_nt_parallel` instead of being
                // outlined to a scalar loop (rows are independent, so it stays deterministic).
                // Integer math, so the kernel equals the scalar nest bit-for-bit (no reassoc).
                if let Some(nest) = i8matmul_fn(body, sema, interner) {
                    let parallel = has_parallel_attr(item, interner);
                    let func =
                        lower_i8matmul_fn(f, &nest, parallel, sema, interner, gemm, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function bf16/f16 matmul: intercept before the elementwise
                // outliner below (which would outline the outer row loop into per-row scalar loops and
                // lose the kernel dispatch). Lower it normally with `parallel = true`; the embedded
                // matmul recognizer in `lower_for` then emits the multicore half GEMM. A non-`@parallel`
                // one reaches the serial kernel via the ordinary `lower_fn` path at the end.
                if has_parallel_attr(item, interner) && lowp_matmul_fn(body, sema, interner).is_some()
                {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function matrix transpose: intercept before the elementwise
                // outliner (which would outline it into per-row scalar loops and lose the blocked
                // kernel). Lower it normally with `parallel = true`; the embedded `match_transpose` in
                // `lower_for` then emits the multicore `mercury_transpose_f32_parallel`. A non-`@parallel`
                // one reaches the serial kernel via the ordinary `lower_fn` path at the end.
                if has_parallel_attr(item, interner) && transpose_fn(body, sema, interner).is_some() {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function 2D pooling nest: intercept before the elementwise outliner
                // (which would split the channel loop into per-chunk scalar loops and lose the AVX2
                // kernel). Lower it normally with `parallel = true`; the embedded `match_pool2d` in
                // `lower_for` then emits the multicore `mercury_{max,avg}pool2d_f32_parallel` (channels
                // across cores, bit-equal to serial — channels independent, no cross-channel combine).
                if has_parallel_attr(item, interner) && pool2d_fn(body, sema, interner).is_some() {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function column reduction: intercept before the outliner (which
                // would split it into per-column-chunk scalar loops and lose the SIMD kernel). Lower it
                // normally with `parallel = true`; the embedded `match_colsum` then emits the multicore
                // `mercury_colsum_f32_parallel` (disjoint column stripes, bit-equal to serial).
                if has_parallel_attr(item, interner) && colsum_fn(body, sema, interner).is_some() {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function per-column argmax/argmin: intercept before the outliner,
                // like the column reduction above. Rows are scanned per disjoint column stripe → the
                // multicore `mercury_colarg*_i32_parallel` is bit-equal to serial.
                if has_parallel_attr(item, interner) && colarg_fn(body, sema, interner).is_some() {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function batched softmax-backward: intercept before the outliner
                // (which would split the rows into scalar loops and lose the fused dot+apply kernel).
                // The embedded `match_softmax_bwd` then emits the multicore `mercury_softmax_bwd_f32_parallel`
                // (rows across cores, bit-equal to serial — rows independent).
                if has_parallel_attr(item, interner) && softmax_bwd_fn(body, sema, interner).is_some() {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function batched RMSNorm-backward: intercept before the outliner
                // (which would split the rows into scalar loops and lose the fused two-reduction+apply
                // kernel). The embedded `match_rmsnorm_bwd` then emits the multicore
                // `mercury_rmsnorm_bwd_f32_parallel` (rows across cores, bit-equal to serial — rows
                // independent, each row reduces over its own `C` columns).
                if has_parallel_attr(item, interner) && rmsnorm_bwd_fn(body, sema, interner).is_some() {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function batched LayerNorm-backward: intercept before the outliner.
                if has_parallel_attr(item, interner)
                    && layernorm_bwd_fn(body, sema, interner).is_some()
                {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function batched cross-entropy loss: intercept before the outliner
                // (which would split the rows into scalar loops and lose the fused max+Σexp+gather
                // kernel). The embedded `match_xent` then emits the multicore `mercury_xent_fwd_f32_parallel`
                // (rows across cores, bit-equal to serial — rows independent).
                if has_parallel_attr(item, interner) && xent_fn(f, body, sema, interner, gemm) {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function cross-entropy backward: intercept before the outliner.
                if has_parallel_attr(item, interner) && xent_bwd_fn(f, body, sema, interner, gemm) {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function RoPE: intercept before the outliner (which would split
                // the rows into scalar loops and lose the inline-sincos kernel). The embedded
                // `match_rope` then emits the multicore `mercury_rope_f32_parallel` (rows independent).
                if has_parallel_attr(item, interner) && rope_fn(body, sema, interner).is_some() {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function batched log-sum-exp: intercept before the outliner.
                if has_parallel_attr(item, interner) && logsumexp_fn(f, body, sema, interner, gemm) {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // `@parallel` whole-function per-row loss reductions (KL / entropy / soft-label xent):
                // intercept before the outliner, like the other per-row loss interceptors.
                if has_parallel_attr(item, interner)
                    && (probe_single_for(f, body, sema, interner, gemm, |p, pat, it, lb| {
                        p.match_kldiv(pat, it, lb).is_some()
                    }) || probe_single_for(f, body, sema, interner, gemm, |p, pat, it, lb| {
                        p.match_entropy(pat, it, lb).is_some()
                    }) || probe_single_for(f, body, sema, interner, gemm, |p, pat, it, lb| {
                        p.match_kd_loss(pat, it, lb).is_some()
                    }))
                {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // `@parallel` whole-function per-row argmax/argmin: intercept before the outliner (which
                // would split the rows into per-row scalar loops and lose the kernel). Rows independent →
                // serial == parallel, so the multicore `mercury_rowarg*_i32_parallel` stays deterministic.
                if has_parallel_attr(item, interner)
                    && probe_single_for(f, body, sema, interner, gemm, |p, pat, it, lb| {
                        p.match_rowarg(pat, it, lb).is_some()
                    })
                {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // `@parallel` whole-function per-row cumsum: intercept before the outliner (rows
                // independent → the multicore `mercury_cumsum_f32_parallel` is bit-equal to serial).
                if has_parallel_attr(item, interner)
                    && probe_single_for(f, body, sema, interner, gemm, |p, pat, it, lb| {
                        p.match_cumsum(pat, it, lb).is_some()
                    })
                {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // `@parallel` whole-function per-row prefix product (cumprod): same interceptor — rows
                // independent → the multicore kernel is bit-equal to serial.
                if has_parallel_attr(item, interner)
                    && probe_single_for(f, body, sema, interner, gemm, |p, pat, it, lb| {
                        p.match_cumprod(pat, it, lb).is_some()
                    })
                {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // `@parallel` whole-function per-row linear-recurrence scan (SSM/Mamba): same
                // interceptor — the independent rows map across cores (bit-equal to serial).
                if has_parallel_attr(item, interner)
                    && probe_single_for(f, body, sema, interner, gemm, |p, pat, it, lb| {
                        p.match_lrscan(pat, it, lb).is_some()
                    })
                {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // `@parallel` whole-function per-row cumulative max/min: same interceptor pattern.
                if has_parallel_attr(item, interner)
                    && probe_single_for(f, body, sema, interner, gemm, |p, pat, it, lb| {
                        p.match_cumminmax(pat, it, lb).is_some()
                    })
                {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // `@parallel` whole-function embedding lookup: intercept before the outliner (which would
                // split the token rows into per-row scalar loops and lose the gather kernel). The embedded
                // `match_embedding` then emits the multicore `mercury_embedding_f32_parallel` (output rows
                // independent → bit-equal to serial).
                if has_parallel_attr(item, interner)
                    && probe_single_for(f, body, sema, interner, gemm, |p, pat, it, lb| {
                        p.match_embedding(pat, it, lb).is_some()
                    })
                {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // `@parallel` whole-function scatter-add (embedding-gradient backward): intercept before
                // the outliner; `match_scatter` then emits `mercury_scatter_add_f32_parallel`, which splits
                // the V output rows across cores (disjoint writes → race-free, deterministic == serial).
                if has_parallel_attr(item, interner)
                    && probe_single_for(f, body, sema, interner, gemm, |p, pat, it, lb| {
                        p.match_scatter(pat, it, lb).is_some()
                    })
                {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` whole-function residual projection (`x = x + act(x·Wᵀ + bias)`):
                // intercept before the outliner (which would split it into per-row scalar loops and
                // lose the fused-epilogue kernel). Lower it normally with `parallel = true`; the
                // embedded `match_matmul_residual` in `lower_for` then emits the multicore nt_epi.
                if has_parallel_attr(item, interner) && matmul_residual_fn(body, sema, interner) {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // A `@parallel` *batched norm* (`fn f(x){ for r in 0..R { <norm row r over x[r*C+i]> } }`)
                // dispatches to the multicore `mercury_norm_f32_parallel`: rows are independent (so it
                // is deterministic and bit-equal to the serial kernel the interpreter calls) and each
                // row is one fused pass. Checked *before* the generic `@parallel` outliner below — which
                // would instead split the rows into chunks of per-row *vectorized* loops and lose the
                // single-pass fusion — mirroring the sgemm/int8 whole-function interceptions above.
                if has_parallel_attr(item, interner)
                    && is_batched_norm_fn(f, body, sema, interner, gemm)
                {
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // `@parallel` on a function whose whole body is `for i in 0..n { … }` over array
                // (pointer) parameters is lowered to a multi-threaded runtime dispatch: the loop
                // body becomes a separate ranged function, and the original becomes a thin wrapper
                // that hands chunks to `mercury_parallel_for`.
                if has_parallel_attr(item, interner) {
                    if let Some((idx, hi, loop_body)) = parallel_spec(f, body, sema, interner) {
                        let base = interner.resolve(f.name.sym).to_string();
                        let par_sym = interner.intern(&format!("{base}$par"));
                        let pfor_sym = interner.intern("mercury_parallel_for");
                        let (outlined, wrapper) = lower_parallel(
                            f, par_sym, pfor_sym, idx, hi, loop_body, sema, interner, gemm,
                            &mut diags,
                        );
                        program.funcs.push(outlined);
                        program.funcs.push(wrapper);
                        continue;
                    }
                    // A `@parallel` function that is not a single elementwise loop — e.g. a reduction
                    // (`let mut s = 0; for k { s += x[k]*y[k] }; …`). Lower it normally, but with any
                    // recognized reduction loop dispatched to the multicore reduction kernel.
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                let func = lower_fn(f, body, sema, interner, gemm, false, &mut diags);
                program.funcs.push(func);
            }
        }
    }
    (program, diags)
}

/// Does this item carry a `@parallel` attribute?
fn has_parallel_attr(item: &ast::Item, interner: &Interner) -> bool {
    item.attrs
        .iter()
        .any(|a| interner.resolve(a.name.sym) == "parallel")
}

/// Recognize a parallelizable function: its entire body is a single `for idx in 0..hi { … }` over
/// pointer (array) parameters. Returns the index name, the upper-bound expression, and the loop
/// body. Anything else falls back to ordinary sequential lowering.
fn parallel_spec<'a>(
    f: &'a FnDecl,
    body: &'a Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(Symbol, &'a Expr, &'a Block)> {
    // Captures are exactly the parameters, so they must all be arrays (passed by pointer).
    let param_tys: Vec<Ty> = match sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
        Some(DefKind::Fn(sig)) => sig.params.clone(),
        _ => return None,
    };
    if param_tys.is_empty()
        || !param_tys
            .iter()
            .all(|t| matches!(mir_ty(t), MirType::Array(..)))
    {
        return None;
    }
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    let ForIter::Range {
        start,
        end: Some(end),
        inclusive: false,
        step: None,
    } = iter
    else {
        return None;
    };
    // Require the range to start at literal 0 (the runtime iterates [0, hi)).
    let ExprKind::Int(s) = &start.kind else {
        return None;
    };
    if parse_int(interner.resolve(*s)) != 0 {
        return None;
    }
    let ast::PatKind::Ident(name) = &pat.kind else {
        return None;
    };
    Some((*name, end, lb))
}

/// Is this function body a single batched-norm loop (`for r in 0..R { <RMSNorm over x[r*C + i]> }`)?
/// Probed with a throwaway lowerer — the recognizer (`match_batched_norm`) is pure: it reads
/// sema/interner and emits no MIR, so a never-built `Builder` is harmless. The `@parallel` driver runs
/// this *before* the generic loop outliner so a `@parallel` batched norm is routed through normal
/// lowering (where `emit_norm` dispatches to the multicore `mercury_norm_f32_parallel`, one fused pass
/// per row across cores) rather than outlined into chunks of per-row *vectorized* loops that lose the
/// fusion — mirroring the int8/sgemm whole-function interceptions.
fn is_batched_norm_fn(
    f: &FnDecl,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
) -> bool {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return false;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return false;
    };
    let mut diags = Vec::new();
    let probe = FnLowerer {
        builder: Builder::new(f.name.sym, MirType::I64),
        sema,
        interner,
        diags: &mut diags,
        scopes: vec![HashMap::new()],
        terminated: false,
        loops: Vec::new(),
        gemm,
        parallel_fn: false,
        vec_loads: HashMap::new(),
        sret: None,
    };
    probe.match_batched_norm(pat, iter, lb).is_some()
}

#[allow(clippy::too_many_arguments)]
fn lower_fn(
    f: &FnDecl,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
    parallel_fn: bool,
    diags: &mut Vec<Diagnostic>,
) -> Function {
    // Recover the resolved signature for parameter/return types.
    let (param_tys, ret_ty) = match sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
        Some(DefKind::Fn(sig)) => (sig.params.clone(), sig.ret.clone()),
        _ => (f.params.iter().map(|_| Ty::Unknown).collect(), Ty::Unit),
    };
    // An aggregate (struct/tuple) return uses an **sret ABI**: the function returns `Void` and takes
    // a hidden leading pointer parameter that the caller fills with a destination buffer; the body
    // deep-copies the returned value into it. No aggregate ever rides in a register, so both backends
    // execute only pointer passing + copies they already support.
    let ret_is_agg = ty_is_aggregate(&ret_ty, sema);
    let ret_mir = if ret_is_agg { MirType::Void } else { mir_ty(&ret_ty) };

    let mut fl = FnLowerer {
        builder: Builder::new(f.name.sym, ret_mir.clone()),
        sema,
        interner,
        diags,
        scopes: vec![HashMap::new()],
        terminated: false,
        loops: Vec::new(),
        gemm,
        parallel_fn,
        vec_loads: HashMap::new(),
        sret: None,
    };

    // The sret pointer is parameter 0 — declared before the real params so the call site can prepend
    // the destination buffer to the argument list.
    if ret_is_agg {
        let sret_ptr = fl.builder.add_param(MirType::Ptr);
        fl.sret = Some((sret_ptr, ret_ty.clone()));
    }

    // Declare all parameters first (so their value ids are contiguous), then materialize each.
    // Arrays AND aggregates (tuples/structs) are passed by base pointer (ABI type `Ptr`); scalars by
    // value. `mir_ty_of` (registry-aware) resolves a named-struct param to its byte-buffer `Array`
    // type — the free `mir_ty` falls back to `I32`, which mistyped a struct param as a scalar (the
    // root of the by-value-struct miscompile and the `mem2reg` panic that promoted that bogus slot).
    let param_abis: Vec<MirType> = param_tys.iter().map(|pty| fl.param_abi(pty)).collect();
    let param_vals: Vec<ValueId> = param_abis
        .iter()
        .map(|abi| fl.builder.add_param(abi.clone()))
        .collect();
    for ((p, pty), val) in f.params.iter().zip(&param_tys).zip(param_vals) {
        let mty = fl.mir_ty_of(pty);
        if matches!(mty, MirType::Array(..)) {
            // The parameter value *is* the aggregate's base pointer; bind it directly so field/index
            // access geps off it (no copy into a local slot).
            fl.bind(p.name.sym, val, mty);
        } else {
            let slot = fl.builder.alloca(mty.clone());
            fl.builder.build_void(Op::Store {
                ptr: slot,
                value: val,
            });
            fl.bind(p.name.sym, slot, mty);
        }
    }

    let tail = fl.lower_block(body);
    if !fl.terminated {
        if let Some((sret_ptr, rty)) = fl.sret.clone() {
            // A fell-through aggregate body: its tail expression (a struct/tuple value) is the
            // return value — deep-copy it into the sret buffer, then return void.
            if let Some(v) = tail {
                fl.emit_copy(sret_ptr, v, &rty);
            }
            fl.builder.ret(None);
        } else {
            match (&ret_mir, tail) {
                (MirType::Void, _) => fl.builder.ret(None),
                // Coerce the implicit tail value to the return type, exactly like an explicit
                // `return` — `fn g() -> i64 { 5 }` must not return the `i32` literal `5`.
                (_, Some(v)) => {
                    let cv = body
                        .tail
                        .as_ref()
                        .map(|te| fl.coerce_return_value(v, te))
                        .unwrap_or(v);
                    fl.builder.ret(Some(cv));
                }
                (_, None) => fl.builder.set_term(mercury_mir::Terminator::Unreachable),
            }
        }
    }
    fl.builder.finish()
}

/// True if `ty` is an aggregate (struct/tuple/array) the ABI passes/returns by pointer — the same
/// classification [`FnLowerer::mir_ty_of`] makes (a `Ty::Named` is aggregate iff it resolves to a
/// declared struct). Free-standing so `lower_fn` can pick the sret ABI before the builder exists.
fn ty_is_aggregate(ty: &Ty, sema: &SemaResult) -> bool {
    match ty {
        Ty::Tuple(_) | Ty::Array { .. } => true,
        Ty::Named(sym) => matches!(
            sema.defs.lookup(*sym).map(|d| &d.kind),
            Some(DefKind::Struct(_))
        ),
        _ => false,
    }
}

/// The MIR type a control-flow merge param (an `if`/`match` *value*) carries for a given result type.
/// An aggregate flows through the CFG as its base **pointer** (the by-pointer convention the sret call
/// path also uses), so its merge param is `Ptr`, not the byte-buffer `Array` type. Typing it `Array`
/// matched the arm's pointer arg only under the interpreter's loose typing (-O0); `mem2reg`'s verifier
/// rejected it (`branch arg (ptr) does not match param ([N x i8])`), so an aggregate-valued `if`/`match`
/// compiled at -O0 but panicked at -O2. Scalars are unchanged (`merge_repr_ty(scalar) == scalar`).
fn merge_repr_ty(result_ty: &MirType) -> MirType {
    match result_ty {
        MirType::Array(..) => MirType::Ptr,
        other => other.clone(),
    }
}

/// Lower a `@parallel for idx in 0..hi { body }` function into two MIR functions:
///   * `par_sym(start: i64, end: i64, env: *ptr)` — the loop body over `[start, end)`, reading the
///     array base pointers back from `env`;
///   * the original `f.name(params…)` — a wrapper that packs the parameter pointers into a stack
///     `env`, then calls `mercury_parallel_for(hi, &par, env)`.
#[allow(clippy::too_many_arguments)]
fn lower_parallel(
    f: &FnDecl,
    par_sym: Symbol,
    pfor_sym: Symbol,
    idx: Symbol,
    hi: &Expr,
    loop_body: &Block,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
    diags: &mut Vec<Diagnostic>,
) -> (Function, Function) {
    let (param_tys, ret_ty) = match sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
        Some(DefKind::Fn(sig)) => (sig.params.clone(), sig.ret.clone()),
        _ => (
            f.params.iter().map(|_| Ty::Unknown).collect::<Vec<_>>(),
            Ty::Unit,
        ),
    };
    let ret_mir = mir_ty(&ret_ty);
    let k = f.params.len();

    // ---- outlined body: par_sym(start, end, env) ----
    let outlined = {
        let mut fl = FnLowerer {
            builder: Builder::new(par_sym, MirType::Void),
            sema,
            interner,
            diags: &mut *diags,
            scopes: vec![HashMap::new()],
            terminated: false,
            loops: Vec::new(),
            gemm,
            parallel_fn: false,
            vec_loads: HashMap::new(),
            sret: None,
        };
        let start = fl.builder.add_param(MirType::I64);
        let end = fl.builder.add_param(MirType::I64);
        let env = fl.builder.add_param(MirType::Ptr);
        // Recover each array base pointer from env[k] and bind it to the parameter name.
        for (idx_k, (p, pty)) in f.params.iter().zip(&param_tys).enumerate() {
            let mty = mir_ty(pty);
            let kidx = fl
                .builder
                .build(MirType::I64, Op::ConstInt(idx_k as i128, MirType::I64));
            let slot = fl.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: env,
                    index: kidx,
                    elem: MirType::Ptr,
                },
            );
            let base = fl.builder.build(MirType::Ptr, Op::Load(slot, MirType::Ptr));
            fl.bind(p.name.sym, base, mty);
        }
        // The index variable's source type (the range's element type) drives the loop so the
        // body's index arithmetic matches sema; the i64 runtime bounds are coerced into it.
        let ity = fl.expr_mir(hi);
        let ity = if ity.is_int() { ity } else { MirType::I64 };
        fl.lower_ranged_loop(idx, start, end, ity, loop_body);
        if !fl.terminated {
            fl.builder.ret(None);
        }
        fl.builder.finish()
    };

    // ---- wrapper: f.name(params…) ----
    let wrapper = {
        let mut fl = FnLowerer {
            builder: Builder::new(f.name.sym, ret_mir.clone()),
            sema,
            interner,
            diags: &mut *diags,
            scopes: vec![HashMap::new()],
            terminated: false,
            loops: Vec::new(),
            gemm,
            parallel_fn: false,
            vec_loads: HashMap::new(),
            sret: None,
        };
        let param_vals: Vec<ValueId> = param_tys
            .iter()
            .map(|pty| fl.builder.add_param(param_abi_ty(pty)))
            .collect();
        let env = fl
            .builder
            .alloca(MirType::Array(Box::new(MirType::Ptr), k as u32));
        for (idx_k, pv) in param_vals.iter().enumerate() {
            let kidx = fl
                .builder
                .build(MirType::I64, Op::ConstInt(idx_k as i128, MirType::I64));
            let slot = fl.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: env,
                    index: kidx,
                    elem: MirType::Ptr,
                },
            );
            fl.builder.build_void(Op::Store {
                ptr: slot,
                value: *pv,
            });
        }
        let n_raw = fl.lower_expr(hi);
        let n_ty = fl.expr_mir(hi);
        let n = fl.coerce_to(n_raw, &n_ty, &MirType::I64, true);
        let bodyaddr = fl.builder.build(MirType::Ptr, Op::FuncAddr(par_sym));
        fl.builder.build_void(Op::Call {
            func: pfor_sym,
            args: vec![n, bodyaddr, env],
        });
        if !fl.terminated {
            match ret_mir {
                MirType::Void => fl.builder.ret(None),
                _ => {
                    let z = fl.const_zero(ret_mir.clone());
                    fl.builder.ret(Some(z));
                }
            }
        }
        fl.builder.finish()
    };

    (outlined, wrapper)
}

/// The runtime entry points the recognizers dispatch to: the four GEMM kernels (`C = A·B` and the
/// `nn.Linear` form `C = A·Bᵀ`, each serial or `@parallel`), the fused-epilogue Linear, and the
/// vectorized elementwise-math kernel.
#[derive(Clone, Copy)]
struct GemmSyms {
    mm: Symbol,
    mm_par: Symbol,
    /// Runtime print of a null-terminated string buffer (a `*u8` argument to `print`/`println`),
    /// as opposed to the numeric `print`/`println`. Renders the bytes, not the pointer value.
    print_str: Symbol,
    println_str: Symbol,
    /// Runtime print of an **unsigned** integer (a `u8`/.../`u64`/`usize` argument): renders the
    /// 64-bit value as `u64`, so a high-bit-set value prints its magnitude, not the signed
    /// two's-complement reinterpretation the default signed `print` would show.
    print_u: Symbol,
    println_u: Symbol,
    nt: Symbol,
    nt_par: Symbol,
    /// The transposed-A weight-gradient kernel (`mercury_sgemm_tn[_parallel]`): `C = Aᵀ·B`, where A is
    /// stored `[k, m]` (the `dW = dYᵀ·X` training backward GEMM — the contraction/batch axis is the
    /// outer index of both operands). gcc/rustc compile the naive nest with column-strided A reads that
    /// defeat vectorization; the kernel transposes A once and reuses the NN microkernel, so it is the
    /// same differential contract the interpreter marshals. Only the `ijk` dot-product form emits it.
    tn: Symbol,
    tn_par: Symbol,
    /// The fused-epilogue `nn.Linear` kernel (`mercury_sgemm_nt_epi`): `C = act(A·Bᵀ + bias)`. A
    /// matmul immediately followed by a bias-add / ReLU loop over its output lowers to this.
    nt_epi: Symbol,
    /// The multicore fused-epilogue `nn.Linear` (`mercury_sgemm_nt_epi_parallel`): the same
    /// `C = act(A·Bᵀ + bias)` fusion in a `@parallel` function, run across cores. Bit-identical to the
    /// serial `nt_epi` the interpreter calls (each C tile owned by one task), so the gate stays exact.
    nt_epi_par: Symbol,
    /// The 256-bit AVX2 elementwise-math kernel (`mercury_vmath_f32(x, out, n, op)`): an
    /// `out[i] = f(x[i])` transcendental loop lowers to this (the width Cranelift can't emit).
    vmath: Symbol,
    /// The **two-input** twin (`mercury_vmath2_f32(x, y, out, n, op)`): an `out[i] = f(x[i], y[i])`
    /// loop for `pow`/`atan2`/`hypot` lowers to this — the 256-bit kernel, vs the inlined 128-bit poly.
    vmath2: Symbol,
    /// The **bf16-input** twin (`mercury_vmath_bf16(x, out, n, op)`): an `out[i] = f((x[i] as f32))`
    /// loop over a `[bf16]` array (f32 output) lowers to this — same 256-bit kernel, half the input
    /// bytes (a lossless widen), so the cheap memory-bound ops gain ~1.3× over the f32 path.
    vmath_bf16: Symbol,
    /// The **f16-input** twin (`mercury_vmath_f16`): same as `vmath_bf16` but widens with F16C.
    vmath_f16: Symbol,
    /// The streaming affine+activation kernel (`mercury_velem_f32(x, y, out, n, a, b, c, op)`): a
    /// recognized `out[i] = act(a·x[i] (+ b·y[i]) + c)` map loop (saxpy / scale / residual-add /
    /// bias / ReLU / ReLU6) lowers to this — 256-bit AVX2 + non-temporal stores for a large output.
    velem: Symbol,
    /// The streaming Horner-polynomial kernel (`mercury_vhorner_f32(x, out, n, coeffs, ncoeff)`): a
    /// recognized `r = c0; r = r*x + c1; …; out[i] = r` per-element polynomial lowers to this.
    vhorner: Symbol,
    /// The multicore deterministic f32 reduction kernel (`mercury_sreduce_f32_parallel(x, y, n, op)
    /// -> f32`): a reduction loop in a `@parallel` function lowers to this. It is bit-equal to the
    /// serial `mercury_sreduce_f32` the interpreter calls, so native and interp stay bit-exact.
    sred_par: Symbol,
    /// The deterministic argmax/argmin reduction (`mercury_argreduce_f32(x, n, op) -> i64`, lowest
    /// index on ties). Serial form (bit-identical to the parallel one), which the interpreter also
    /// calls — so a recognized argmax loop stays exact across backends.
    argreduce: Symbol,
    /// The multicore variant (`mercury_argreduce_f32_parallel`): a recognized global argmax/argmin in
    /// a `@parallel` function lowers to this. It is **bit-identical** to the serial `argreduce` (fixed
    /// `RCHUNK` decomposition, ascending partial fold — index independent of thread count), which the
    /// interpreter calls, so native @parallel and interp stay exact.
    argreduce_par: Symbol,
    /// The fused single-pass row-wise normalization kernel (`mercury_norm_f32(x, out, rows, cols,
    /// eps_bits, op)`): an idiomatic multi-pass softmax / LayerNorm / RMSNorm written in plain loops
    /// lowers to this one call. The interpreter marshals through the identical kernel.
    norm: Symbol,
    /// The multicore variant of `mercury_norm_f32` (`mercury_norm_f32_parallel`): a *batched* norm
    /// (`rows > 1`) in a `@parallel` function maps its independent rows across cores here. Each row is
    /// normalized by the same per-row routine with no cross-row combine, so it is bit-equal to the
    /// serial kernel the interpreter calls — the differential gate holds regardless of thread count.
    norm_par: Symbol,
    /// The affine fused norm kernel (`mercury_norm_affine_f32(x, out, gamma, beta, rows, cols,
    /// eps_bits, op)`): a LayerNorm/RMSNorm whose normalize step also applies a per-column scale
    /// `gamma` (and, for LayerNorm, a shift `beta`) lowers here instead — the real transformer form.
    norm_affine: Symbol,
    /// The multicore variant of `mercury_norm_affine_f32` (`mercury_norm_affine_f32_parallel`): a
    /// *batched* affine norm (`rows > 1`) in a `@parallel` function maps its independent rows across
    /// cores here, bit-equal to the serial affine kernel the interpreter marshals (no cross-row combine).
    norm_affine_par: Symbol,
    /// The int8 quantized `nn.Linear` kernel (`mercury_i8gemm_nt[_parallel](a, b, c, m, k, n)`): a
    /// `u8×i8→i32` `C = A·Bᵀ` nest lowers to this. Integer arithmetic, so the fused kernel equals the
    /// naive loop bit-for-bit (no reassociation exception).
    i8nt: Symbol,
    i8nt_par: Symbol,
    /// The fused int8 GEMM + dequant kernel (`mercury_i8gemm_nt_deq[_parallel](a, b, out, m, k, n,
    /// scale_a, scale_b, bias, act)`): an int8 `C = A·Bᵀ` nest immediately followed by a per-channel
    /// dequant `out[i*N+j] = act((c[i*N+j] as f32) * scale_a * scale_b[j] [+ bias[j]])` folds to this
    /// one call — the i32 accumulator never touches memory (the kernel dequants each tile in registers
    /// straight to the f32 output). cuBLAS/oneDNN emit the i32 GEMM and the dequant as two passes; this
    /// is the fusion they structurally can't express. Both backends marshal the identical kernel.
    i8deq: Symbol,
    i8deq_par: Symbol,
    /// The bf16 / f16 mixed-precision GEMM kernels (`mercury_sgemm_{bf16,f16}_nt[_parallel](a, b, c,
    /// m, k, n, beta)`): a `C = A·Bᵀ` nest whose `A`/`B` are `[bf16]`/`[f16]` arrays widened with
    /// `as f32` and accumulated in an f32 `s` lowers here — the standard mixed-precision transformer
    /// matmul. The kernel widens the half inputs (lossless) and runs the identical tuned f32 GEMM, so
    /// the result is bit-for-bit the f32 GEMM on the widened values (the differential contract). The
    /// naive half nest otherwise falls to a scalar widening loop the autovectorizer can't reach.
    bf16_nt: Symbol,
    bf16_nt_par: Symbol,
    f16_nt: Symbol,
    f16_nt_par: Symbol,
    /// The fused-epilogue bf16/f16 mixed-precision GEMM kernels (`mercury_sgemm_{bf16,f16}_nt_epi[_
    /// parallel](a, b, c, m, k, n, beta, bias, act)`): a bf16/f16 `C = A·Bᵀ` nest immediately followed
    /// by its bias-add / activation loop folds here — the mixed-precision transformer FFN projection,
    /// fused. The lossless widen prepass feeds the f32 `gemm_dispatch`, which applies the identical
    /// `nt_epi` epilogue (bias + identity/ReLU/GELU/SiLU) in its C writeback, so the result is the f32
    /// fused FFN on the widened operands — bit-for-bit across backends. The `f32` twin is `nt_epi`.
    bf16_nt_epi: Symbol,
    bf16_nt_epi_par: Symbol,
    f16_nt_epi: Symbol,
    f16_nt_epi_par: Symbol,
    /// The bf16/f16 mixed-precision **weight-gradient** GEMM kernels (`mercury_sgemm_{bf16,f16}_tn[_
    /// parallel](a, b, c, m, k, n, beta)`): a bf16/f16 `C = Aᵀ·B` nest (A stored `[k,m]`, B `[k,n]` —
    /// the `dW = dYᵀ·X` training backward) folds here. The lossless widen prepass feeds the f32
    /// `mercury_sgemm_tn` (transpose A once + the tuned `C = A·B` kernel), so the result is bit-for-bit
    /// the f32 TN GEMM on the widened operands. Same 7-arg ABI as the `nt` kernels; the `f32` twin is `tn`.
    bf16_tn: Symbol,
    bf16_tn_par: Symbol,
    f16_tn: Symbol,
    f16_tn_par: Symbol,
    /// The bf16 mixed-precision reduction kernels (`mercury_dot_bf16(x, y, n) -> f32` and
    /// `mercury_sum_bf16(x, n) -> f32`): a reduction loop `s += (x[k] as f32) [* (y[k] as f32)]` over
    /// `[bf16; _]` arrays with an f32 accumulator lowers to one of these — bf16 storage, f32
    /// accumulate (the standard ML mixed-precision contract). bf16 is bit-exact across interp/native
    /// and both call the identical kernel, so the differential gate stays exact (a reassociation
    /// exception, like the f32 reduction kernel).
    dot_bf16: Symbol,
    sum_bf16: Symbol,
    /// `mercury_reduce_bf16(x, n, op) -> f32` — the bf16 **max-family** reduction (`RED_MAX`/`RED_MIN`/
    /// `RED_MAXABS`): a `m = fmax(m, (x[k] as f32))` / `fmin` / `fmax(m, abs(...))` loop over `[bf16]`
    /// lowers here. The widen is lossless and max/min round nothing, so it is the exact reduction; the
    /// outer combine is `Cmp+Select` (as in the f32 path), and interp calls the identical kernel.
    reduce_bf16: Symbol,
    /// The IEEE-f16 twins of the reductions above (`mercury_dot_f16` / `mercury_sum_f16` /
    /// `mercury_reduce_f16`): same dispatch, but the kernel widens f16→f32 with F16C `vcvtph2ps`
    /// instead of the bf16 `<<16`. A `[f16]` reduction loop routes here.
    dot_f16: Symbol,
    sum_f16: Symbol,
    reduce_f16: Symbol,
    /// `mercury_axpby_bf16(x, y, out, n, a, b)` — bf16→f32 streaming axpby (`out = a*x + b*y`, bf16
    /// inputs, f32 output, f32 math). The mixed-precision elementwise twin of the f32 streaming kernel.
    axpby_bf16: Symbol,
    /// `mercury_axpby_f16` — the F16C twin of `axpby_bf16` (f16 inputs widened with `vcvtph2ps`).
    axpby_f16: Symbol,
    /// `mercury_transpose_f32[_parallel](src, dst, rows, cols)` — the cache-blocked matrix transpose
    /// (`dst[j,i] = src[i,j]`, `[rows,cols]` → `[cols,rows]`). A recognized transpose nest dispatches
    /// here; it is pure data movement (a permutation), so bit-identical to the scalar nest on both
    /// backends (no reassociation — the differential gate is trivial). The `_parallel` form spreads
    /// the independent row blocks across cores.
    transpose: Symbol,
    transpose_par: Symbol,
    /// The 16-bit (`bf16`/`f16`, stored as `u16`) transpose (`mercury_transpose_u16[_parallel]`): a
    /// transpose nest over a half-precision array dispatches here — the same cache-blocked kernel, half
    /// the bytes. A transpose moves the raw bits, so one `u16` kernel serves both bf16 and f16.
    transpose_u16: Symbol,
    transpose_u16_par: Symbol,
    /// The SIMD column reduction (`mercury_colsum_f32[_parallel](x, out, rows, cols)`): a `out[j] =
    /// Σ_i x[i*cols+j]` nest (the bias gradient / batch sum, a reduce along axis 0) dispatches here.
    /// The strided naive form gcc/rustc leave scalar; this streams `x` row-major + 8-wide. Bit-exact
    /// (i-ascending per column, same order), so both backends marshal the identical kernel.
    colsum: Symbol,
    colsum_par: Symbol,
    /// The SIMD column **max**/**min** (`mercury_col{max,min}_f32[_parallel](x, out, rows, cols)`):
    /// `out[j] = max_i x[i*cols+j]` / `min` (per-channel statistics for quantization, axis-0 max/min
    /// pooling), the max/min siblings of `colsum`. Same strided gap (gcc/rustc stay scalar — verified)
    /// and the same i-ascending fold (`_mm256_max_ps`/`_mm256_min_ps`), so both backends marshal the
    /// identical kernel and it is bit-exact.
    colmax: Symbol,
    colmax_par: Symbol,
    colmin: Symbol,
    colmin_par: Symbol,
    /// The SIMD column **abs-max** (`mercury_colmaxabs_f32[_parallel]`): `out[j] = max_i |x[i*cols+j]|`,
    /// the per-channel symmetric int8-quantization scale (`amax_j`). Same strided gap and i-ascending
    /// fold as colmax, with each element abs'd first (sign-mask `andnot`).
    colmaxabs: Symbol,
    colmaxabs_par: Symbol,
    /// The SIMD per-channel **statistics** family (`mercury_col{mean,sumsq,l2,rms}_f32[_parallel]`,
    /// same `(x, out, rows, cols)` ABI as colsum): `out[j] = mean_i x[i,j]` (BatchNorm mean),
    /// `Σ_i x[i,j]²` (energy), `sqrt(Σ x²)` (column L2), `sqrt(mean x²)` (per-channel RMS). Same strided
    /// fold as colsum + a per-column finalize; both backends marshal the identical kernel.
    colmean: Symbol,
    colmean_par: Symbol,
    colsumsq: Symbol,
    colsumsq_par: Symbol,
    coll2: Symbol,
    coll2_par: Symbol,
    colrms: Symbol,
    colrms_par: Symbol,
    /// The fused softmax-backward (`mercury_softmax_bwd_f32[_parallel](y, dy, dx, rows, cols)`): a
    /// `dx = y·(dy − Σ y·dy)` batched nest dispatches here. The per-row dot gcc/rustc keep scalar; the
    /// kernel reuses the bit-exact `sreduce` dot + an 8-wide apply. Rows independent → serial == parallel.
    softmax_bwd: Symbol,
    softmax_bwd_par: Symbol,
    /// The fused RMSNorm backward (`mercury_rmsnorm_bwd_f32[_parallel](x, dy, gamma, dx, rows, cols,
    /// eps_bits)`): the `dx = r·(g − x·r²·(Σ g·x)/C)` input-gradient nest dispatches here. The two
    /// per-row reductions gcc/rustc keep scalar; rows independent → serial == parallel.
    rmsnorm_bwd: Symbol,
    rmsnorm_bwd_par: Symbol,
    /// The fused LayerNorm backward (`mercury_layernorm_bwd_f32[_parallel](x, dy, gamma, dx, rows,
    /// cols, eps_bits)`): the input-gradient nest dispatches here. Same 7-arg ABI as rmsnorm_bwd.
    layernorm_bwd: Symbol,
    layernorm_bwd_par: Symbol,
    /// The fused softmax cross-entropy forward loss (`mercury_xent_fwd_f32[_parallel](x, target, loss,
    /// rows, cols)`): the `loss[r] = lse(x[r]) − x[r, target[r]]` nest dispatches here. C/Rust keep the
    /// `expf`/`logf` reduction scalar; rows independent → serial == parallel.
    xent: Symbol,
    xent_par: Symbol,
    /// Softmax cross-entropy backward (`mercury_xent_bwd_f32[_parallel](x, target, dx, rows, cols)`):
    /// the `dx = softmax(x) − onehot(target)` gradient nest dispatches here.
    xent_bwd: Symbol,
    xent_bwd_par: Symbol,
    /// RoPE (`mercury_rope_f32[_parallel](x, inv_freq, out, rows, half)`): the inline-sin/cos rotary
    /// embedding nest dispatches here. C/Rust keep sinf/cosf scalar; rows independent → serial==parallel.
    rope: Symbol,
    rope_par: Symbol,
    /// RoPE backward (`mercury_rope_bwd_f32[_parallel]`): the transpose/inverse rotation (the gradient).
    rope_bwd: Symbol,
    rope_bwd_par: Symbol,
    /// Batched log-sum-exp (`mercury_logsumexp_f32[_parallel](x, out, rows, cols)`): the
    /// `out[r] = m + log(Σexp(x[r,·]−m))` log-partition nest dispatches here. Same vmath shape as the
    /// transcendental kernels. C/Rust keep the expf reduction scalar; rows independent → serial==parallel.
    logsumexp: Symbol,
    logsumexp_par: Symbol,
    /// Per-row loss reductions (`mercury_kldiv_f32` / `mercury_entropy_f32` / `mercury_kd_loss_f32`,
    /// each `[_parallel]`): KL divergence, Shannon entropy, soft-label cross-entropy. All log/exp-bound.
    kldiv: Symbol,
    kldiv_par: Symbol,
    entropy: Symbol,
    entropy_par: Symbol,
    kd_loss: Symbol,
    kd_loss_par: Symbol,
    /// Per-row arg-reductions returning an **i32** index (`mercury_rowarg{max,min}_i32[_parallel](x,
    /// out, rows, cols)`): `out[r] = {argmax,argmin}_j x[r,j]` — the classification-head / greedy-decode
    /// top-1 (lowest index wins on a value tie). The index bookkeeping keeps gcc/rustc scalar; AVX2
    /// tracks 8 (value,index) lanes via blend. `(ptr,ptr,i64,i64)` = the `sig_vmath` shape, but `out` is
    /// an i32 buffer (the interp writes `Value::Int`). Rows independent → serial == parallel.
    rowargmax: Symbol,
    rowargmax_par: Symbol,
    rowargmin: Symbol,
    rowargmin_par: Symbol,
    /// Per-**column** arg-reductions returning an i32 row index (`mercury_colarg{max,min}_i32[_parallel]
    /// (x, out, rows, cols)`): `out[j] = {argmax,argmin}_i x[i,j]`. The strided axis-0 sibling of rowarg;
    /// same i32-output `sig_vmath` ABI. The column-outer access gcc/rustc leave scalar.
    colargmax: Symbol,
    colargmax_par: Symbol,
    colargmin: Symbol,
    colargmin_par: Symbol,
    /// Per-row inclusive prefix sum (`mercury_cumsum_f32[_parallel](x, out, rows, cols)`): `out[r,i] =
    /// Σ_{k<=i} x[r,k]`. gcc/rustc keep the loop-carried `out[i]=out[i-1]+x[i]` scalar; the SIMD
    /// Hillis-Steele scan + carry vectorizes it. The in-lane tree reassociates → the kernel is the oracle.
    cumsum: Symbol,
    cumsum_par: Symbol,
    /// Inclusive per-row prefix product `out[r,i] = Π_{k<=i} x[r,k]` (cumulative product / scan),
    /// `mercury_cumprod_f32[_parallel](x, out, rows, cols)`. Loop-carried like cumsum → gcc/rustc keep it
    /// scalar; the kernel's lever is 4-row-interleaved ILP. Bit-exact (a product is not fused — no
    /// reassociation, unlike the prefix sum).
    cumprod: Symbol,
    cumprod_par: Symbol,
    /// Linear-recurrence / selective scan `out[r,t] = a[r,t]·h_{t-1} + b[r,t]` (SSM/Mamba/S4/EMA), each
    /// row an independent first-order recurrence (`mercury_lrscan_f32[_parallel](a, b, out, rows, cols)`).
    /// Loop-carried within a row like cumsum → gcc/rustc keep the carry scalar; the lever is the
    /// `_parallel` map of the independent rows across cores (memory-bound single-core tie otherwise).
    lrscan: Symbol,
    lrscan_par: Symbol,
    /// Per-row cumulative max / min (`mercury_cum{max,min}_f32[_parallel]`): `out[r,i] = max/min_{k<=i}
    /// x[r,k]`. Same SIMD-scan lever as cumsum, but max/min select a value (no reassociation) → bit-exact.
    cummax: Symbol,
    cummax_par: Symbol,
    cummin: Symbol,
    cummin_par: Symbol,
    /// Embedding lookup (`mercury_embedding_f32[_parallel](out, weight, ids, t, h, v)`): `out[t,:] =
    /// weight[ids[t],:]` — the first layer of every LLM (token-id row gather). `ids` is an `i32` index
    /// array; pure data movement (a row copy), so the kernel is bit-identical to the scalar gather on
    /// both backends — no reassociation, the differential gate is trivial. Rows independent → `_parallel`
    /// maps the `T` output rows across cores, bit-equal to serial.
    embedding: Symbol,
    embedding_par: Symbol,
    /// Scatter-add / embedding-gradient backward `grad_w[ids[t], :] += grad_out[t, :]`
    /// (`mercury_scatter_add_f32[_parallel](grad_w, grad_out, ids, T, H, V)`). The dual of the embedding
    /// gather; the `_parallel` one splits the V output rows across cores (collision-free, deterministic).
    scatter_add: Symbol,
    scatter_add_par: Symbol,
    /// 2D max/avg pooling (`mercury_{max,avg}pool2d_f32[_parallel](x, out, channels, h, w, kh, kw,
    /// sh, sw)`): `out[c,oy,ox] = ⊕ over the kh×kw window of x[c, oy*sh+dy, ox*sw+dx]` over a
    /// `[channels, h, w]` row-major input, no padding (`⊕` = max / sum÷(kh·kw)). The idiomatic 5-deep
    /// CNN downsampling nest dispatches here; the AVX2 kernel folds 8 output columns at once (the
    /// strided window gcc/rustc leave scalar). Channels independent → `_parallel` is bit-equal to
    /// serial; max idempotent and the avg sum order is fixed, so both backends marshal the identical
    /// kernel (no reassociation exception).
    maxpool2d: Symbol,
    maxpool2d_par: Symbol,
    avgpool2d: Symbol,
    avgpool2d_par: Symbol,
}

// Elementwise-math op codes — must match `mercury_runtime::vmath`'s `VM_*` (mir_build does not depend
// on the runtime crate; same arrangement as the EPI_ACT_* codes mirroring the runtime's).
const VMATH_EXP: u32 = 0;
const VMATH_LOG: u32 = 1;
const VMATH_TANH: u32 = 2;
const VMATH_SIGMOID: u32 = 3;
const VMATH_SILU: u32 = 5;
const VMATH_GELU: u32 = 6;
const VMATH_ELU: u32 = 7;
const VMATH_LEAKYRELU: u32 = 8;
const VMATH_SOFTPLUS: u32 = 9;
const VMATH_MISH: u32 = 10;
const VMATH_SELU: u32 = 11;
const VMATH_TANHSHRINK: u32 = 12;
const VMATH_HARDSIGMOID: u32 = 13;
const VMATH_HARDSWISH: u32 = 14;
const VMATH_SIN: u32 = 15;
const VMATH_COS: u32 = 16;
const VMATH_ERF: u32 = 17;
const VMATH_EXP2: u32 = 18;
const VMATH_LOG2: u32 = 19;
const VMATH_SINH: u32 = 20;
const VMATH_COSH: u32 = 21;
const VMATH_ASINH: u32 = 22;
const VMATH_ACOSH: u32 = 23;
const VMATH_ATANH: u32 = 24;
const VMATH_ATAN: u32 = 25;
const VMATH_EXPM1: u32 = 26;
const VMATH_LOG1P: u32 = 27;
const VMATH_EXP10: u32 = 28;
const VMATH_LOG10: u32 = 29;
const VMATH_SOFTSIGN: u32 = 30;
const VMATH_LOGSIGMOID: u32 = 31;
const VMATH_TAN: u32 = 32;
const VMATH_ASIN: u32 = 33;
const VMATH_ACOS: u32 = 34;
const VMATH_CBRT: u32 = 35;
// Two-input kernel op codes (`mercury_vmath2_f32`, separate namespace — must match `vmath`'s `VM2_*`).
const VMATH2_POW: u32 = 0;
const VMATH2_ATAN2: u32 = 1;
const VMATH2_HYPOT: u32 = 2;
// Activation backward `dx = dy·act'(x)` (two inputs `(x, dy)`) — must match `vmath`'s `VM2_*`.
const VMATH2_SILU_BWD: u32 = 3;
const VMATH2_GELU_BWD: u32 = 4;
const VMATH2_SIGMOID_BWD: u32 = 5;
const VMATH2_TANH_BWD: u32 = 6;
const VMATH2_ELU_BWD: u32 = 7;
const VMATH2_SOFTPLUS_BWD: u32 = 8;
// Gated-FFN activation `out = act(a)·b` (SwiGLU/GeGLU) — must match `mercury_runtime::vmath`'s
// `VM2_{SILU,GELU}_GATE`. Inputs positional `(a, b)`.
const VMATH2_SILU_GATE: u32 = 9;
const VMATH2_GELU_GATE: u32 = 10;

// Streaming affine+activation op codes — must match `mercury_runtime::velem`'s `VE_*`. The low byte
// is the activation; `VE_USE_Y` (bit 8) flags that the kernel reads `y`.
const VE_ID: i64 = 0; // out = a·x (+ b·y) + c
const VE_RELU: i64 = 1; // out = max(.., 0)
const VE_RELU6: i64 = 2; // out = min(max(.., 0), 6)
const VE_USE_Y: i64 = 256;
const VE_HADAMARD: i64 = 512; // out = act(x·y) — Hadamard product (kernel reads y)
const VE_DIV: i64 = 1024; // out = act(x / y) — elementwise quotient

/// One additive term of a recognized streaming affine body. `Scaled(arr, s)` is `arr[j]` (`s = None`,
/// coefficient 1) or `s·arr[j]` / `arr[j]·s` for a loop-invariant f32 scalar `s`; `Const(s)` is a
/// loop-invariant f32 scalar added in (the bias). Borrows the coefficient exprs from the body.
enum VTerm<'b> {
    Scaled(Symbol, Option<&'b Expr>),
    Const(&'b Expr),
}

/// A recognized `out[j] = act(a·x[j] (+ b·y[j]) + c)` streaming map: resolved array base pointers plus
/// the (loop-invariant) coefficient exprs, lowered to ValueIds at emit time so a runtime scale such as
/// saxpy's `a` works. `op` is the activation byte; `VE_USE_Y` is set iff `y` is present.
struct VElemPlan<'b> {
    // The operand *symbols* (not pre-resolved values): a pure matcher can't emit the load that pulls
    // a tensor/pointer param's base out of its slot, so the base pointer is resolved at emit time via
    // `kernel_base_ptr` (a no-op for an array operand, a `Load` for a `Tensor`/pointer one). Storing
    // the slot value here instead would GEP off the slot address for a `Tensor[..]` param — the
    // 1-D-tensor-kernel segfault/divergence.
    out: Symbol,
    x: Symbol,
    y: Option<Symbol>,
    a: Option<&'b Expr>,
    b: Option<&'b Expr>,
    c: Option<&'b Expr>,
    op: i64,
}

// Reduction op codes — must match `mercury_runtime::reduce`'s `RED_*`. `x[k]*x[k]` recognizes as
// `RED_DOT` with both bases equal (≡ sum-of-squares), so the recognizer needs only these. The
// additive ops fold by `+`; `RED_MAX`/`RED_MIN` fold by `fmax`/`fmin` (per-tensor max/absmax for
// softmax stability and dynamic int8 quantization), and the outer combine is a `Cmp+Select`.
const RED_DOT: i64 = 0; // sum(x[k] * y[k])
const RED_SSD: i64 = 1; // sum((x[k] - y[k])^2)
const RED_SUM: i64 = 2; // sum(x[k])
const RED_MAX: i64 = 4; // max(x[k])  — fold by fmax
const RED_MIN: i64 = 5; // min(x[k])  — fold by fmin
const RED_MAXABS: i64 = 6; // max(|x[k]|) — fmax(m, abs(x[k])), symmetric int8 quant absmax
const RED_ARGMAX: i64 = 7; // argmax_i x[i] — greedy decode / top-1 (lowest index on ties)
const RED_ARGMIN: i64 = 8; // argmin_i x[i]

// Fused-normalization op codes — must match `mercury_runtime::norm`'s `NORM_*`.
const NORM_SOFTMAX: i64 = 0; // out = softmax(x) over the row
const NORM_LAYERNORM: i64 = 1; // out = (x - mean) / sqrt(var + eps)
const NORM_RMSNORM: i64 = 2; // out = x / sqrt(mean(x^2) + eps)
const NORM_LOGSOFTMAX: i64 = 3; // out = (x - m) - log(sum(exp(x - m))) — stable log-softmax
const NORM_L2NORM: i64 = 4; // out = x / sqrt(sum(x^2) + eps) — L2 / unit normalize (no mean divisor)

struct FnLowerer<'a> {
    builder: Builder,
    sema: &'a SemaResult,
    interner: &'a Interner,
    diags: &'a mut Vec<Diagnostic>,
    scopes: Vec<HashMap<Symbol, (ValueId, MirType)>>,
    terminated: bool,
    /// (optional label, continue target, break target) for the enclosing loops, innermost last. A
    /// labeled `break`/`continue` `'l` searches this stack for the matching label; an unlabeled one
    /// targets the innermost (the top).
    loops: Vec<(Option<Symbol>, mercury_mir::BlockId, mercury_mir::BlockId)>,
    /// Pre-interned runtime symbols the matmul recognizer lowers a GEMM nest to.
    gemm: GemmSyms,
    /// True while lowering the body of a `@parallel` function: a recognized reduction loop dispatches
    /// to the multicore `mercury_sreduce_f32_parallel` instead of the sequential vectorizer.
    parallel_fn: bool,
    /// Within one vectorized loop-body copy, the vector already loaded for an index expression
    /// (keyed by its canonical text), so `x[i]` read twice (e.g. relu's `if x[i]>0 {x[i]}`) loads
    /// once. Cleared between unroll copies (addresses differ) and after any store (avoid staleness).
    vec_loads: HashMap<String, ValueId>,
    /// Set when the function returns an aggregate (struct/tuple) by value: the hidden leading
    /// **sret** pointer parameter the caller passes a destination buffer in, paired with the
    /// aggregate return type. A `return <aggregate>` deep-copies into this pointer and returns void
    /// (the function's MIR return type is `Void`), so both backends only ever pass/copy pointers —
    /// no aggregate ever rides in a register. `None` for a scalar/void return.
    sret: Option<(ValueId, Ty)>,
}

impl FnLowerer<'_> {
    fn unsupported(&mut self, span: Span, what: &str) {
        // A construct codegen cannot lower yields incomplete/invalid MIR, so this is a hard error:
        // the compiler refuses to emit a broken program rather than silently producing one. The
        // front end (parsing, type and shape checking) still accepts these constructs — only
        // lowering to runnable code is unsupported so far.
        self.diags.push(
            Diagnostic::error(format!("`{what}` is not yet supported by codegen"))
                .with_code("C0001")
                .primary(span, "this construct cannot be lowered yet"),
        );
    }

    // ---- scopes ----

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn bind(&mut self, name: Symbol, slot: ValueId, ty: MirType) {
        self.scopes.last_mut().unwrap().insert(name, (slot, ty));
    }

    fn lookup(&self, name: Symbol) -> Option<(ValueId, MirType)> {
        for s in self.scopes.iter().rev() {
            if let Some(v) = s.get(&name) {
                return Some(v.clone());
            }
        }
        None
    }

    // ---- type helpers ----

    fn expr_ty(&self, e: &Expr) -> Ty {
        self.sema.types.get(&e.id).cloned().unwrap_or(Ty::Unknown)
    }

    /// Is `e`'s type a string (`*u8`)? Routes a `print`/`println` argument to the byte-rendering
    /// `print_str` path instead of printing the raw pointer value. A string literal and a
    /// `let s = "…"` binding both type as `*u8` in sema, so both are caught.
    fn is_string_arg(&self, e: &Expr) -> bool {
        matches!(self.expr_ty(e), Ty::Ptr { pointee, .. } if matches!(*pointee, Ty::Scalar(mercury_types::Scalar::U8)))
    }

    /// Is `e`'s type an unsigned integer scalar (`u8`/`u16`/`u32`/`u64`/`usize`)? Routes a
    /// `print`/`println` argument to the unsigned-rendering `print_u` path so a high-bit-set value
    /// prints its magnitude rather than the signed reinterpretation. `bool` is excluded (`is_int`
    /// excludes it); signed integers and floats take the default path.
    fn is_unsigned_int_arg(&self, e: &Expr) -> bool {
        matches!(self.expr_ty(e), Ty::Scalar(sc) if sc.is_int() && !sc.is_signed())
    }

    fn expr_mir(&self, e: &Expr) -> MirType {
        self.mir_ty_of(&self.expr_ty(e))
    }

    /// `mir_ty`, but resolves a named struct (`Ty::Named`) to its byte-buffer storage type using the
    /// struct's field layout from sema (the free `mir_ty` has no def access and would fall back to
    /// `I32`). A struct value, like a tuple, is a flat padded byte buffer addressed by field offset.
    /// Recurses through arrays and tuples so a *nested* struct (a struct field, or an element of an
    /// array of structs) also resolves — `mir_ty` stops at the first `Named` and mis-sizes the rest.
    fn mir_ty_of(&self, ty: &Ty) -> MirType {
        match ty {
            Ty::Named(sym) => match self.struct_size(*sym) {
                Some(size) => MirType::Array(Box::new(MirType::I8), size as u32),
                None => mir_ty(ty),
            },
            Ty::Array { elem, len } => {
                MirType::Array(Box::new(self.mir_ty_of(elem)), *len as u32)
            }
            Ty::Tuple(_) => {
                MirType::Array(Box::new(MirType::I8), self.ty_size(ty).unwrap_or(0) as u32)
            }
            _ => mir_ty(ty),
        }
    }

    /// The MIR slot type for a `let x: T;` annotation with **no** initializer — the registry-aware
    /// companion to the free `mir_ty_of_ast` (which mistypes a named struct / a tuple as `i32`,
    /// under-allocating the slot so a later field write GEPs off a scalar and ICEs). A named struct
    /// resolves to its byte buffer (`mir_ty_of`), a tuple to a padded byte buffer sized from its
    /// elements, an array recurses (so an array-of-struct element is sized correctly), and every
    /// pointer/scalar form matches `mir_ty_of_ast`.
    fn mir_ty_of_ann(&self, t: &ast::TypeExpr) -> MirType {
        use ast::TypeKind::*;
        match &t.kind {
            Path(p) => {
                let sym = p.segments.last().unwrap().sym;
                let name = self.interner.resolve(sym);
                if let Some(s) = mercury_types::Scalar::from_name(name) {
                    MirType::from_scalar(s)
                } else if matches!(
                    self.sema.defs.lookup(sym).map(|d| &d.kind),
                    Some(DefKind::Struct(_))
                ) {
                    self.mir_ty_of(&Ty::Named(sym))
                } else {
                    MirType::I32
                }
            }
            Array { elem, len } => match const_usize_expr(len, self.interner) {
                Some(n) => MirType::Array(Box::new(self.mir_ty_of_ann(elem)), n),
                None => MirType::Ptr,
            },
            Tuple(fields) => {
                // A padded byte buffer sized from the element MIR types — the same layout the
                // with-initializer path derives via `mir_ty_of(Ty::Tuple(..))`/`ty_size`.
                let elems: Vec<MirType> = fields.iter().map(|f| self.mir_ty_of_ann(f)).collect();
                let mut size = 0u64;
                let mut align = 1u64;
                for e in &elems {
                    let a = mir_byte_align(e);
                    size = round_up(size, a);
                    size += mir_byte_size(e);
                    align = align.max(a);
                }
                MirType::Array(Box::new(MirType::I8), round_up(size, align) as u32)
            }
            Pointer { .. } | Ref { .. } | Slice(_) | Tensor { .. } => MirType::Ptr,
            Vector { elem, lanes } => MirType::Vec(Box::new(self.mir_ty_of_ann(elem)), *lanes),
            Unit => MirType::Void,
            _ => MirType::I32,
        }
    }

    /// Zero-initialize the freshly-alloca'd slot of a no-initializer `let`. A scalar gets one typed
    /// zero store; a scalar array fills (unrolled when small, a fill loop otherwise); an aggregate
    /// (struct/tuple byte buffer, or an array of aggregates) recurses so every leaf is zeroed. The
    /// effect mirrors the interpreter's zero-initialized memory, so an uninitialized read agrees
    /// bit-for-bit across backends and opt levels. (A `Ptr`/`Vec` slot — a rare no-init form — is
    /// left alone: there is no valid typed-zero MIR constant for those, and neither was a reported
    /// divergence; this is strictly an improvement over the prior garbage-read behavior.)
    fn zero_init(&mut self, slot: ValueId, ty: &MirType) {
        if ty.is_int() || ty.is_float() {
            let z = self.const_zero(ty.clone());
            self.builder.build_void(Op::Store {
                ptr: slot,
                value: z,
            });
        } else if let MirType::Array(elem, n) = ty {
            if elem.is_int() || elem.is_float() {
                let z = self.const_zero((**elem).clone());
                if *n <= REPEAT_UNROLL_LIMIT {
                    for i in 0..*n as i128 {
                        self.store_element(slot, elem, i, z);
                    }
                } else {
                    self.lower_fill_loop(slot, elem, *n, z);
                }
            } else {
                for i in 0..*n as i128 {
                    let ep = self.gep_elem(slot, elem, i);
                    self.zero_init(ep, elem);
                }
            }
        }
    }

    /// The ABI type of a parameter of semantic type `ty`: an aggregate (array/tuple/struct, whose
    /// `mir_ty_of` is an `Array` byte buffer) is passed by base **pointer**; a scalar by value. The
    /// registry-aware companion to the free `param_abi_ty` (which mistypes a named struct as `I32`).
    fn param_abi(&self, ty: &Ty) -> MirType {
        match self.mir_ty_of(ty) {
            MirType::Array(..) => MirType::Ptr,
            t => t,
        }
    }

    /// Size in bytes of `ty`, resolving named structs through the sema registry — the registry-aware
    /// companion to `Ty::size_of` (which returns `None` for `Ty::Named`, since the leaf type crate
    /// has no def access). Recurses through arrays/tuples so nested structs lay out correctly.
    fn ty_size(&self, ty: &Ty) -> Option<u64> {
        match ty {
            Ty::Named(sym) => self.struct_size(*sym),
            Ty::Array { elem, len } => Some(self.ty_size(elem)? * len),
            Ty::Tuple(fields) => self.aggregate_layout(fields).map(|(_, size, _)| size),
            _ => ty.size_of(),
        }
    }

    /// Alignment of `ty`, resolving named structs through the sema registry (see [`ty_size`]).
    fn ty_align(&self, ty: &Ty) -> Option<u64> {
        match ty {
            Ty::Named(sym) => self.struct_align(*sym),
            Ty::Array { elem, .. } => self.ty_align(elem),
            Ty::Tuple(fields) => fields
                .iter()
                .try_fold(1u64, |a, f| Some(a.max(self.ty_align(f)?))),
            _ => ty.align_of(),
        }
    }

    /// Padded field offsets + total size + alignment for a sequence of field types — the single
    /// layout authority shared by structs and tuples (the same `round_up` accumulation as
    /// `Ty::size_of`, but registry-aware so a named-struct field is sized recursively). `None` if any
    /// field is genuinely unsized (a slice/tensor/unresolved name).
    fn aggregate_layout(&self, fields: &[Ty]) -> Option<(Vec<u64>, u64, u64)> {
        let mut offsets = Vec::with_capacity(fields.len());
        let mut size = 0u64;
        let mut align = 1u64;
        for f in fields {
            let fa = self.ty_align(f)?;
            let fs = self.ty_size(f)?;
            size = round_up(size, fa);
            offsets.push(size);
            size += fs;
            align = align.max(fa);
        }
        Some((offsets, round_up(size, align), align))
    }

    /// The field layout of a declared struct: `(field name, byte offset, field MIR type)` in
    /// declaration order. Uses the registry-aware `aggregate_layout` (so a field that is itself a
    /// struct lays out correctly) and `mir_ty_of` for each field type (so a nested-struct field gets
    /// its byte-buffer type, not the `I32` fallback). `None` if `name` is not a struct or is unsized.
    fn struct_layout(&self, name: Symbol) -> Option<Vec<(Symbol, u64, MirType)>> {
        let DefKind::Struct(fields) = &self.sema.defs.lookup(name)?.kind else {
            return None;
        };
        let tys: Vec<Ty> = fields.iter().map(|(_, t)| t.clone()).collect();
        let (offsets, _, _) = self.aggregate_layout(&tys)?;
        Some(
            fields
                .iter()
                .zip(offsets)
                .map(|((fname, fty), off)| (*fname, off, self.mir_ty_of(fty)))
                .collect(),
        )
    }

    /// `(byte offset, field type)` for each field of struct `name`, in declaration order — the
    /// semantic-type companion to `struct_layout` (which gives MIR types). Drives nested aggregate
    /// initialization and copies, which need the `Ty` to recurse.
    fn struct_field_tys(&self, name: Symbol) -> Option<Vec<(u64, Ty)>> {
        let DefKind::Struct(fields) = &self.sema.defs.lookup(name)?.kind else {
            return None;
        };
        let tys: Vec<Ty> = fields.iter().map(|(_, t)| t.clone()).collect();
        let (offsets, _, _) = self.aggregate_layout(&tys)?;
        Some(tys.into_iter().zip(offsets).map(|(t, o)| (o, t)).collect())
    }

    /// Total padded byte size of a declared struct (its alloca size), registry-aware.
    fn struct_size(&self, name: Symbol) -> Option<u64> {
        let DefKind::Struct(fields) = &self.sema.defs.lookup(name)?.kind else {
            return None;
        };
        let tys: Vec<Ty> = fields.iter().map(|(_, t)| t.clone()).collect();
        self.aggregate_layout(&tys).map(|(_, size, _)| size)
    }

    /// Alignment of a declared struct (the max field alignment), registry-aware.
    fn struct_align(&self, name: Symbol) -> Option<u64> {
        let DefKind::Struct(fields) = &self.sema.defs.lookup(name)?.kind else {
            return None;
        };
        fields
            .iter()
            .try_fold(1u64, |a, (_, f)| Some(a.max(self.ty_align(f)?)))
    }

    /// Address + MIR type of struct field `fname` of the struct expression `base`. The base lowers to
    /// a pointer to the struct buffer in every case: a struct *local* (its bound value *is* the buffer
    /// pointer, like a tuple/array), and **a pointer/reference to a struct** — `p.f` on a `&Pt` /
    /// `*mut Pt` param auto-derefs (the bound `Ptr` slot loads the pointer, then this GEPs the field).
    /// So a struct passed by `&`/`*` works the same as a local. Drives `s.f` reads and `s.f = …` writes.
    /// Bind a tuple destructuring pattern `(a, b, …)` against an aggregate `base` of tuple type
    /// `tty`: each sub-pattern is bound to its field's place (byte offset within `base`). A scalar
    /// field binds a pointer that reads via `Load` (like a `let` slot); an aggregate field binds its
    /// pointer directly (the by-pointer convention); a nested tuple pattern recurses; a wildcard
    /// binds nothing. Used by `let (a, b) = …`.
    fn bind_tuple_pattern(&mut self, base: ValueId, tty: &Ty, subs: &[Pattern]) {
        let Ty::Tuple(fields) = tty else {
            return;
        };
        let Some((offsets, _, _)) = self.aggregate_layout(fields) else {
            return;
        };
        for (i, sub) in subs.iter().enumerate() {
            let (Some(off), Some(fty)) = (offsets.get(i), fields.get(i)) else {
                continue;
            };
            let fmir = self.mir_ty_of(fty);
            let fptr = self.field_ptr(base, *off);
            match &sub.kind {
                ast::PatKind::Ident(name) => self.bind(*name, fptr, fmir),
                ast::PatKind::Tuple(inner) => self.bind_tuple_pattern(fptr, fty, inner),
                _ => {}
            }
        }
    }

    /// If `base.name` is a C-style enum-variant access `E::B` (a `Field` whose base is a single
    /// segment path naming a declared enum, and `name` is one of its variants), return the variant's
    /// integer discriminant. Lowered to that constant (the enum value's runtime representation).
    fn enum_variant_value(&self, base: &Expr, name: Symbol) -> Option<i64> {
        let ExprKind::Path(p) = &base.kind else {
            return None;
        };
        if !p.is_single() {
            return None;
        }
        let DefKind::Enum(variants) = &self.sema.defs.lookup(p.first().sym)?.kind else {
            return None;
        };
        variants.iter().find(|(v, _)| *v == name).map(|(_, d)| *d)
    }

    fn struct_field_place(&mut self, base: &Expr, fname: Symbol) -> (ValueId, MirType) {
        let struct_sym = match self.expr_ty(base) {
            Ty::Named(sym) => Some(sym),
            Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } => match *pointee {
                Ty::Named(sym) => Some(sym),
                _ => None,
            },
            _ => None,
        };
        if let Some(sym) = struct_sym {
            if let Some(layout) = self.struct_layout(sym) {
                if let Some((_, off, fmty)) = layout.iter().find(|(n, _, _)| *n == fname) {
                    let off = *off;
                    let fmty = fmty.clone();
                    let base_ptr = self.lower_expr(base);
                    let p = self.field_ptr(base_ptr, off);
                    return (p, fmty);
                }
            }
        }
        self.unsupported(base.span, "field access on a non-struct value");
        let ty = MirType::I32;
        (self.builder.alloca(ty.clone()), ty)
    }

    /// Lower a struct literal `Name { f: v, … }` into the byte buffer at `base`: each field value is
    /// initialized at its declared byte offset (literal field order may differ from declaration order
    /// — each value goes to its named field's offset). A field that is itself a struct/tuple/array
    /// recurses (or byte-copies) via `init_field`, so nested aggregates work.
    fn lower_struct_init(&mut self, base: ValueId, sym: Symbol, fields: &[ast::FieldInit], span: Span) {
        let Some(field_tys) = self.struct_field_tys(sym) else {
            self.unsupported(span, "struct with an unsized field");
            return;
        };
        let by_name: HashMap<Symbol, (u64, Ty)> = self
            .sema
            .defs
            .lookup(sym)
            .and_then(|d| match &d.kind {
                DefKind::Struct(decl) => Some(
                    decl.iter()
                        .map(|(n, _)| *n)
                        .zip(field_tys.iter().cloned())
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();
        for fi in fields {
            let Some((off, fty)) = by_name.get(&fi.name.sym).cloned() else {
                self.unsupported(fi.name.span, "unknown struct field");
                continue;
            };
            let p = self.field_ptr(base, off);
            self.init_field(p, &fty, &fi.value);
        }
    }

    /// Initialize the location `dst` (a pointer into an aggregate buffer) of semantic type `fty` from
    /// initializer `value`. A nested struct/tuple/array literal recurses *directly* into `dst` (no
    /// temporary buffer + copy); a non-literal aggregate value is deep-copied from its buffer; a
    /// scalar is coerced to the field type and stored. The one initializer primitive shared by struct,
    /// tuple, and (aggregate-element) array lowering.
    fn init_field(&mut self, dst: ValueId, fty: &Ty, value: &Expr) {
        match (&value.kind, fty) {
            (ExprKind::StructLit { fields, .. }, Ty::Named(sym)) => {
                self.lower_struct_init(dst, *sym, fields, value.span);
            }
            (ExprKind::TupleLit(items), Ty::Tuple(_)) => {
                self.lower_tuple_init(dst, fty, items);
            }
            (ExprKind::ArrayLit(_) | ExprKind::ArrayRepeat { .. }, Ty::Array { elem, len }) => {
                let emir = self.mir_ty_of(elem);
                self.lower_array_init(dst, &emir, *len as u32, value);
            }
            _ => {
                let fmty = self.mir_ty_of(fty);
                if matches!(fmty, MirType::Array(..)) {
                    // An aggregate value from a non-literal expression (a variable, a call result, a
                    // field): the expression yields a base pointer; deep-copy its leaves into `dst`.
                    let src = self.lower_expr(value);
                    self.emit_copy(dst, src, fty);
                } else {
                    let v0 = self.lower_expr(value);
                    let vty = self.expr_mir(value);
                    let v = self.coerce_to(v0, &vty, &fmty, self.signed(value));
                    self.builder.build_void(Op::Store { ptr: dst, value: v });
                }
            }
        }
    }

    /// Deep-copy a value of type `ty` from buffer `src` to buffer `dst` (both base pointers), using
    /// the *same* GEP discipline as field/element access so it is correct under both the interpreter's
    /// slot-indexed memory and native's byte-indexed memory: struct/tuple fields recurse through the
    /// byte-offset `field_ptr`, array elements through an element-typed GEP, and scalar leaves are a
    /// single load+store. A flat byte `memcpy` would be wrong for the interpreter (a non-leading
    /// scalar field lives at its byte-offset slot, which an 8-byte chunked copy would skip).
    fn emit_copy(&mut self, dst: ValueId, src: ValueId, ty: &Ty) {
        match ty {
            Ty::Named(sym) => {
                if let Some(layout) = self.struct_field_tys(*sym) {
                    for (off, fty) in layout {
                        let s = self.field_ptr(src, off);
                        let d = self.field_ptr(dst, off);
                        self.emit_copy(d, s, &fty);
                    }
                }
            }
            Ty::Tuple(fields) => {
                if let Some((offsets, _, _)) = self.aggregate_layout(fields) {
                    let pairs: Vec<(u64, Ty)> =
                        offsets.into_iter().zip(fields.iter().cloned()).collect();
                    for (off, fty) in pairs {
                        let s = self.field_ptr(src, off);
                        let d = self.field_ptr(dst, off);
                        self.emit_copy(d, s, &fty);
                    }
                }
            }
            Ty::Array { elem, len } => {
                let emir = self.mir_ty_of(elem);
                for i in 0..*len as i128 {
                    let s = self.gep_elem(src, &emir, i);
                    let d = self.gep_elem(dst, &emir, i);
                    self.emit_copy(d, s, elem);
                }
            }
            _ => {
                let mir = self.mir_ty_of(ty);
                let v = self.builder.build(mir.clone(), Op::Load(src, mir.clone()));
                self.builder.build_void(Op::Store { ptr: dst, value: v });
            }
        }
    }

    /// Read a field/element at `ptr` of MIR type `fmty`: a scalar/pointer field is loaded; an
    /// aggregate field (`MirType::Array`, i.e. a nested struct/tuple/inline array) yields `ptr`
    /// itself — the by-pointer convention arrays follow, so a further field/index access GEPs off it
    /// rather than trying to load (and copy) the whole buffer through a register.
    fn load_or_addr(&mut self, ptr: ValueId, fmty: MirType) -> ValueId {
        if matches!(fmty, MirType::Array(..)) {
            ptr
        } else {
            self.builder.build(fmty.clone(), Op::Load(ptr, fmty))
        }
    }

    /// Pointer to element `index` of `base` for an element of MIR type `elem` (the element-typed GEP
    /// the array machinery uses: native scales by `size_of(elem)`, the interpreter indexes slots).
    fn gep_elem(&mut self, base: ValueId, elem: &MirType, index: i128) -> ValueId {
        let idx = self
            .builder
            .build(MirType::I64, Op::ConstInt(index, MirType::I64));
        self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: base,
                index: idx,
                elem: elem.clone(),
            },
        )
    }

    fn signed(&self, e: &Expr) -> bool {
        matches!(self.expr_ty(e), Ty::Scalar(s) if s.is_signed())
    }

    fn const_zero(&mut self, ty: MirType) -> ValueId {
        if ty.is_float() {
            self.builder.build(ty.clone(), Op::ConstFloat(0.0, ty))
        } else if ty.is_int() {
            self.builder.build(ty.clone(), Op::ConstInt(0, ty))
        } else {
            self.builder
                .build(MirType::I32, Op::ConstInt(0, MirType::I32))
        }
    }

    // ---- blocks & statements ----

    fn lower_block(&mut self, b: &Block) -> Option<ValueId> {
        self.push_scope();
        let mut i = 0;
        while i < b.stmts.len() {
            if self.terminated {
                break;
            }
            // Epilogue fusion: a `nn.Linear` matmul immediately followed by its bias-add / ReLU loop
            // folds into one GEMM call with the epilogue applied in the C writeback (no separate pass
            // over C). Checked before the elementwise fusion below — they match disjoint shapes.
            if let Some(n) = self.try_fuse_matmul_epilogue(&b.stmts[i..]) {
                i += n;
                continue;
            }
            // The bf16/f16 twin: a half-precision `nn.Linear` matmul immediately followed by its
            // bias-add / activation loop folds to one `mercury_sgemm_{bf16,f16}_nt_epi` call — the
            // mixed-precision transformer FFN. Disjoint from the f32 epilogue above (half inputs, an
            // `as f32` widen on each factor) and the int8 dequant below (a float matmul, not `u8×i8→i32`).
            if let Some(n) = self.try_fuse_lowp_matmul_epilogue(&b.stmts[i..]) {
                i += n;
                continue;
            }
            // int8 GEMM + per-channel dequant epilogue → one fused `mercury_i8gemm_nt_deq` call (the
            // i32 accumulator is dequanted in registers, never materialized). Structurally disjoint
            // from the f32 epilogue above (an int8 `u8×i8→i32` nest, not a float matmul).
            if let Some(n) = self.try_fuse_i8matmul_dequant_epilogue(b, i) {
                i += n;
                continue;
            }
            // Fused normalization: an idiomatic multi-pass softmax / LayerNorm / RMSNorm over a flat
            // f32 array folds into one `mercury_norm_f32` call (each row loaded once + 256-bit AVX2).
            // Checked before the elementwise-fusion run so it sees the raw loop sequence, not a
            // pre-fused one. The three windows are structurally disjoint (max+exp vs mean+var+shift vs
            // sum-of-squares+scale), so probe order is immaterial.
            // log-softmax (6 stmts) is probed before softmax (7 stmts): they share the leading max
            // pass but diverge at stmt[2] — softmax's is the `exp` rewrite loop, log-softmax's is
            // `let s = 0` — so the two matchers are disjoint (each declines the other's window).
            if let Some((n, arr, n_expr)) = self.match_logsoftmax(b, i, None) {
                if self.emit_norm(arr, None, &n_expr, 0, NORM_LOGSOFTMAX, None, None) {
                    i += n;
                    continue;
                }
            }
            if let Some((n, arr, n_expr)) = self.match_softmax(b, i, None) {
                if self.emit_norm(arr, None, &n_expr, 0, NORM_SOFTMAX, None, None) {
                    i += n;
                    continue;
                }
            }
            if let Some((n, arr, n_expr, eps, gamma, beta)) = self.match_layernorm(b, i, None) {
                if self.emit_norm(arr, None, &n_expr, eps, NORM_LAYERNORM, gamma, beta) {
                    i += n;
                    continue;
                }
            }
            if let Some((n, arr, n_expr, eps, gamma, beta)) = self.match_rmsnorm(b, i, None) {
                if self.emit_norm(arr, None, &n_expr, eps, NORM_RMSNORM, gamma, beta) {
                    i += n;
                    continue;
                }
            }
            // L2-normalize (4 stmts, same shape as RMSNorm but `1/sqrt(Σx² + eps)` — no mean divisor).
            // Probed after RMSNorm; the two are disjoint on the reciprocal (`/N` present xor absent),
            // so neither steals the other's window.
            if let Some((n, arr, n_expr, eps, _g, _b)) = self.match_l2norm(b, i, None) {
                if self.emit_norm(arr, None, &n_expr, eps, NORM_L2NORM, None, None) {
                    i += n;
                    continue;
                }
            }
            // Operator fusion: a run of adjacent same-range elementwise `for` loops whose *fused*
            // body the vectorizer accepts is lowered as one loop (CSE/DSE then forward any
            // intermediate array through registers, cutting its memory traffic).
            match self.try_fuse_run(&b.stmts[i..]) {
                Some(n) => i += n,
                None => {
                    self.lower_stmt(&b.stmts[i]);
                    i += 1;
                }
            }
        }
        let tail = match &b.tail {
            Some(e) if !self.terminated => Some(self.lower_expr(e)),
            _ => None,
        };
        self.pop_scope();
        tail
    }

    /// If `stmts` begins with two or more adjacent `for` loops over the *identical* range whose
    /// concatenated body the vectorizer accepts, lower them as a single fused loop and return how
    /// many statements were consumed. The vectorizer's "no written array touched at a second index"
    /// rule, applied to the fused body, is exactly the condition that makes fusion dependence-safe,
    /// so a successful check both authorizes and SIMD-accelerates the fusion. Returns `None`
    /// otherwise (and the caller lowers the first statement normally).
    fn try_fuse_run(&mut self, stmts: &[Stmt]) -> Option<usize> {
        let (pat0, iter0, _) = fusable_for(&stmts[0])?;
        let var = match &pat0.kind {
            ast::PatKind::Ident(s) => *s,
            _ => return None,
        };
        let (start0, end0) = range_bounds(iter0)?;
        // Maximal run of for-loops over the identical (var, start, end).
        let mut run = 1;
        while run < stmts.len() {
            match fusable_for(&stmts[run]) {
                Some((p, it, _)) => {
                    let same = matches!(&p.kind, ast::PatKind::Ident(s) if *s == var)
                        && range_bounds(it).is_some_and(|(s, e)| {
                            exprs_struct_eq(s, start0) && exprs_struct_eq(e, end0)
                        });
                    if same {
                        run += 1;
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }
        if run < 2 {
            return None;
        }
        // Fuse the longest dependence-safe (vectorizable) prefix of length >= 2.
        for m in (2..=run).rev() {
            let fused = fuse_for_bodies(&stmts[..m]);
            if self.vectorizable(&fused, var).is_some() {
                self.lower_for(None, pat0, iter0, &fused);
                return Some(m);
            }
        }
        None
    }

    // ---- fused normalization recognition (block-level look-ahead, like the matmul epilogue) ----

    /// Match `for v in 0..N { body }` (exclusive, literal-0 start, no step), returning the loop
    /// variable, the upper-bound expression, and the body. Pure (no MIR).
    fn as_range0_for<'b>(&self, stmt: &'b Stmt) -> Option<(Symbol, &'b Expr, &'b Block)> {
        let StmtKind::For {
            pat, iter, body, ..
        } = &stmt.kind
        else {
            return None;
        };
        let ast::PatKind::Ident(v) = &pat.kind else {
            return None;
        };
        let ForIter::Range {
            start,
            end: Some(end),
            inclusive: false,
            step: None,
        } = iter
        else {
            return None;
        };
        if const_usize_expr(start, self.interner) != Some(0) {
            return None;
        }
        Some((*v, end, body))
    }

    /// `let [mut] name [: ty] = init` → `(name, init)`. Pure.
    fn let_init(stmt: &Stmt) -> Option<(Symbol, &Expr)> {
        let StmtKind::Let {
            pat,
            init: Some(init),
            ..
        } = &stmt.kind
        else {
            return None;
        };
        let ast::PatKind::Ident(name) = &pat.kind else {
            return None;
        };
        Some((*name, init))
    }

    /// Body `m = fmax(m, x[v])` (running max into scalar `m`, indexing `x` by `v`) → return `x`. Pure.
    fn match_max_reduce_body(
        &self,
        body: &Block,
        v: Symbol,
        m: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<Symbol> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if single_path(target) != Some(m) {
            return None;
        }
        let ExprKind::Call { callee, args, .. } = &value.kind else {
            return None;
        };
        if args.len() != 2
            || !matches!(
                self.vectorizable_intrinsic(callee),
                Some(MathIntrinsic::Fmax)
            )
        {
            return None;
        }
        // One arg must be `m`; the other must be `x[v]` (row-offset-indexed when batched).
        let xside = if single_path(&args[0]) == Some(m) {
            1
        } else if single_path(&args[1]) == Some(m) {
            0
        } else {
            return None;
        };
        self.index_off(&args[xside], v, batch)
    }

    /// Body `x[v] = exp(x[v] - m)` over f32, in place. Pure.
    fn match_exp_sub_body(
        &self,
        body: &Block,
        v: Symbol,
        x: Symbol,
        m: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<()> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if self.index_off(target, v, batch) != Some(x) || self.expr_mir(value) != MirType::F32 {
            return None;
        }
        let ExprKind::Call { callee, args, .. } = &value.kind else {
            return None;
        };
        if args.len() != 1
            || !matches!(
                self.vectorizable_intrinsic(callee),
                Some(MathIntrinsic::Exp)
            )
        {
            return None;
        }
        let ExprKind::Binary {
            op: ast::BinOp::Sub,
            lhs,
            rhs,
        } = &args[0].kind
        else {
            return None;
        };
        if self.index_off(lhs, v, batch) == Some(x) && single_path(rhs) == Some(m) {
            Some(())
        } else {
            None
        }
    }

    /// Body `s += x[v]` / `s = s + x[v]` (sum into scalar `s`), verifying the summed array is `x`. Pure.
    fn match_sum_body(
        &self,
        body: &Block,
        v: Symbol,
        x: Symbol,
        s: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<()> {
        if self.sum_body_array(body, v, s, batch)? == x {
            Some(())
        } else {
            None
        }
    }

    /// Body `s += x[v]` / `s = s + x[v]` (sum into scalar `s`) → the summed array `x` (whichever it
    /// is). The array-discovering form of [`Self::match_sum_body`] (LayerNorm's leading sum loop is
    /// what first names the row array). Pure.
    fn sum_body_array(
        &self,
        body: &Block,
        v: Symbol,
        s: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<Symbol> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign { target, op, value } = &stmt.kind else {
            return None;
        };
        if single_path(target) != Some(s) {
            return None;
        }
        let addend: &Expr = match op {
            ast::AssignOp::Add => value,
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(lhs) == Some(s) => rhs,
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(rhs) == Some(s) => lhs,
                _ => return None,
            },
            _ => return None,
        };
        self.index_off(addend, v, batch)
    }

    /// Body `s += x[v]*x[v]` (sum of squares into scalar `s`) → the squared array `x` (RMSNorm's lead
    /// loop, and the mean-square reduction). Pure.
    fn match_sumsq_body(
        &self,
        body: &Block,
        v: Symbol,
        s: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<Symbol> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign { target, op, value } = &stmt.kind else {
            return None;
        };
        if single_path(target) != Some(s) {
            return None;
        }
        let addend: &Expr = match op {
            ast::AssignOp::Add => value,
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(lhs) == Some(s) => rhs,
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(rhs) == Some(s) => lhs,
                _ => return None,
            },
            _ => return None,
        };
        // addend must be `x[v] * x[v]` — same array, same index, both factors.
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &addend.kind
        else {
            return None;
        };
        let xl = self.index_off(lhs, v, batch)?;
        let xr = self.index_off(rhs, v, batch)?;
        if xl == xr {
            Some(xl)
        } else {
            None
        }
    }

    /// `arr[v]` where `arr != x` (the data) → `arr`: a per-column affine parameter array (gamma/beta)
    /// indexed by the normalize loop variable. Pure.
    fn affine_index(&self, e: &Expr, v: Symbol, x: Symbol) -> Option<Symbol> {
        match self.index_by_loopvar(e, v) {
            Some(a) if a != x => Some(a),
            _ => None,
        }
    }

    /// Peel an optional affine wrapper `core * gamma[v] (+ beta[v])` off a normalize-loop RHS, in the
    /// canonical `(... ) * gamma[i] + beta[i]` spelling (gamma on either side of its `*`, beta on
    /// either side of its `+`). Returns the remaining `core` expression plus the gamma/beta arrays
    /// (each `None` when absent). `gamma`/`beta` must be indexed by the loop var `v` and be arrays
    /// other than the data `x`. Pure — leaves `core` == `value` when there is no affine wrapper.
    fn peel_affine<'e>(
        &self,
        value: &'e Expr,
        v: Symbol,
        x: Symbol,
    ) -> (&'e Expr, Option<Symbol>, Option<Symbol>) {
        // outermost `+ beta[v]`
        let (after_beta, beta) = match &value.kind {
            ExprKind::Binary {
                op: ast::BinOp::Add,
                lhs,
                rhs,
            } => {
                if let Some(b) = self.affine_index(rhs, v, x) {
                    (lhs.as_ref(), Some(b))
                } else if let Some(b) = self.affine_index(lhs, v, x) {
                    (rhs.as_ref(), Some(b))
                } else {
                    (value, None)
                }
            }
            _ => (value, None),
        };
        // then `* gamma[v]`
        let (core, gamma) = match &after_beta.kind {
            ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } => {
                if let Some(g) = self.affine_index(rhs, v, x) {
                    (lhs.as_ref(), Some(g))
                } else if let Some(g) = self.affine_index(lhs, v, x) {
                    (rhs.as_ref(), Some(g))
                } else {
                    (after_beta, None)
                }
            }
            _ => (after_beta, None),
        };
        (core, gamma, beta)
    }

    /// Body `x[v] = x[v] * inv [* gamma[v] [+ beta[v]]]` (scale by an invariant scalar, either operand
    /// order, with an optional affine wrapper for RMSNorm). Returns the captured `(gamma, beta)`
    /// arrays (both `None` for the plain form). Pure.
    fn match_scale_body(
        &self,
        body: &Block,
        v: Symbol,
        x: Symbol,
        inv: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<(Option<Symbol>, Option<Symbol>)> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        // The data `x` is row-offset-indexed when batched; gamma/beta stay column-indexed (per-column,
        // shared across rows), so `peel_affine` is unbatched.
        if self.index_off(target, v, batch) != Some(x) {
            return None;
        }
        let (core, gamma, beta) = self.peel_affine(value, v, x);
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &core.kind
        else {
            return None;
        };
        let ok = (self.index_off(lhs, v, batch) == Some(x) && single_path(rhs) == Some(inv))
            || (self.index_off(rhs, v, batch) == Some(x) && single_path(lhs) == Some(inv));
        if ok {
            Some((gamma, beta))
        } else {
            None
        }
    }

    /// Is `init` a sound running-max seed for softmax over `x` — `x[0]`, or a large-negative literal
    /// (`<= -1e30`)? Either is `<= max(x)`, so the user's `fmax(seed, …)` equals the true max the
    /// kernel computes; anything else might change behavior, so we decline. Pure.
    fn is_max_seed(&self, init: &Expr, x: Symbol, batch: Option<(Symbol, &Expr)>) -> bool {
        if let ExprKind::Index { base, indices } = &init.kind {
            if single_path(base) != Some(x) || indices.len() != 1 {
                return false;
            }
            // Single-row: the seed is `x[0]`. Batched row `r`: the seed is `x[r*C]` — the row's first
            // element (the bare `row*cols` base offset, no `+ i`), which `is_mul_of` matches.
            return match batch {
                None => {
                    matches!(&indices[0].kind, ExprKind::Int(t) if parse_int(self.interner.resolve(*t)) == 0)
                }
                Some((row, cols)) => self.is_mul_of(&indices[0], row, cols),
            };
        }
        let v = match &init.kind {
            ExprKind::Float(t) => parse_float(self.interner.resolve(*t)),
            ExprKind::Unary {
                op: ast::UnOp::Neg,
                expr,
            } => match &expr.kind {
                ExprKind::Float(t) => -parse_float(self.interner.resolve(*t)),
                _ => return false,
            },
            _ => return false,
        };
        v <= -1e30
    }

    /// `let name = 1.0 / s` → `name`. Pure.
    fn match_recip(&self, stmt: &Stmt, s: Symbol) -> Option<Symbol> {
        let (name, init) = Self::let_init(stmt)?;
        let ExprKind::Binary {
            op: ast::BinOp::Div,
            lhs,
            rhs,
        } = &init.kind
        else {
            return None;
        };
        let one = matches!(&lhs.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 1.0);
        if one && single_path(rhs) == Some(s) {
            Some(name)
        } else {
            None
        }
    }

    /// Recognize the canonical in-place flat softmax window (7 statements) at `b.stmts[at..]`:
    ///
    /// ```text
    /// let mut m = x[0];                       // seed: x[0] or a large-negative literal
    /// for i in 0..N { m = fmax(m, x[i]); }    // row max
    /// for i in 0..N { x[i] = exp(x[i] - m); } // exp(x - max), in place
    /// let mut s = 0.0;
    /// for i in 0..N { s += x[i]; }            // sum
    /// let inv = 1.0 / s;
    /// for i in 0..N { x[i] = x[i] * inv; }    // normalize
    /// ```
    ///
    /// Every loop must range over the identical `0..N` and index the *same* array `x` exactly by its
    /// loop variable; the scalars must chain (max → exp, sum → reciprocal, reciprocal → scale) and the
    /// internal scalars must not be read after the window (the kernel hides them). Pure: returns
    /// `(consumed, array, N)` with `N` cloned so the caller can emit freely. `None` on any deviation
    /// (the generic vectorizer + vmath path then lowers the loops correctly).
    fn match_softmax(
        &self,
        b: &Block,
        at: usize,
        batch: Option<Symbol>,
    ) -> Option<(usize, Symbol, Expr)> {
        let stmts = &b.stmts[at..];
        if stmts.len() < 7 {
            return None;
        }
        let (m, seed) = Self::let_init(&stmts[0])?;
        let (v1, n_expr, body1) = self.as_range0_for(&stmts[1])?;
        // Batched: the data is row-offset-indexed `row*cols + i` (and the max-seed is `x[row*cols]`);
        // single-row: just `x[i]` (seed `x[0]`). The max / sum scalars and `1/s` are per-row, unchanged.
        let data_batch = batch.map(|row| (row, n_expr));
        let x = self.match_max_reduce_body(body1, v1, m, data_batch)?;
        if !self.is_max_seed(seed, x, data_batch) {
            return None;
        }
        let (v2, n2, body2) = self.as_range0_for(&stmts[2])?;
        if !exprs_struct_eq(n2, n_expr) {
            return None;
        }
        self.match_exp_sub_body(body2, v2, x, m, data_batch)?;
        let (s, s_init) = Self::let_init(&stmts[3])?;
        if !matches!(&s_init.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 0.0)
        {
            return None;
        }
        let (v4, n4, body4) = self.as_range0_for(&stmts[4])?;
        if !exprs_struct_eq(n4, n_expr) {
            return None;
        }
        self.match_sum_body(body4, v4, x, s, data_batch)?;
        let inv = self.match_recip(&stmts[5], s)?;
        let (v6, n6, body6) = self.as_range0_for(&stmts[6])?;
        if !exprs_struct_eq(n6, n_expr) {
            return None;
        }
        // softmax's normalize is a plain `x[i] *= inv`; reject any affine wrapper (softmax has no
        // gamma/beta) so it falls back to the generic vectorizer rather than silently dropping it.
        if self.match_scale_body(body6, v6, x, inv, data_batch)? != (None, None) {
            return None;
        }
        // The three internal scalars must not be read after the window — the kernel hides them.
        let rest = &b.stmts[at + 7..];
        let tail = b.tail.as_deref();
        for sc in [m, s, inv] {
            if block_mentions(rest, tail, sc) {
                return None;
            }
        }
        Some((7, x, n_expr.clone()))
    }

    /// Recognize a stable **log-softmax** window (6 statements), the classification / LM-training loss
    /// epilogue: `let m = x[0]; for i { m = fmax(m, x[i]) }; let s = 0; for i { s += exp(x[i]-m) };
    /// let ls = log(s); for i { x[i] = (x[i]-m) - ls }`. The first two passes are softmax's max +
    /// sum-of-exp; the divergence is the scalar `log(s)` and the final subtract (no normalize-by-inv).
    /// Folds to one `mercury_norm_f32(x, x, 1, N, 0, NORM_LOGSOFTMAX)`. Pure. `None` on any deviation
    /// (the generic vectorizer + vmath path then lowers the loops). Batch-aware like `match_softmax`.
    fn match_logsoftmax(
        &self,
        b: &Block,
        at: usize,
        batch: Option<Symbol>,
    ) -> Option<(usize, Symbol, Expr)> {
        let stmts = &b.stmts[at..];
        if stmts.len() < 6 {
            return None;
        }
        let (m, seed) = Self::let_init(&stmts[0])?;
        let (v1, n_expr, body1) = self.as_range0_for(&stmts[1])?;
        let data_batch = batch.map(|row| (row, n_expr));
        let x = self.match_max_reduce_body(body1, v1, m, data_batch)?;
        if !self.is_max_seed(seed, x, data_batch) {
            return None;
        }
        let (s, s_init) = Self::let_init(&stmts[2])?;
        if !matches!(&s_init.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 0.0)
        {
            return None;
        }
        let (v3, n3, body3) = self.as_range0_for(&stmts[3])?;
        if !exprs_struct_eq(n3, n_expr) {
            return None;
        }
        self.match_sumexp_sub_body(body3, v3, x, m, s, data_batch)?;
        let ls = self.match_log_of(&stmts[4], s)?;
        let (v5, n5, body5) = self.as_range0_for(&stmts[5])?;
        if !exprs_struct_eq(n5, n_expr) {
            return None;
        }
        self.match_logsoftmax_norm_body(body5, v5, x, m, ls, data_batch)?;
        // The three internal scalars must not be read after the window — the kernel hides them.
        let rest = &b.stmts[at + 6..];
        let tail = b.tail.as_deref();
        for sc in [m, s, ls] {
            if block_mentions(rest, tail, sc) {
                return None;
            }
        }
        Some((6, x, n_expr.clone()))
    }

    /// Body `s += exp(x[v] - m)` / `s = s + exp(x[v]-m)` (sum of exp-of-centered into scalar `s`,
    /// reading `x[v]`, no in-place write — log-softmax overwrites `x` only in its final pass). Pure.
    fn match_sumexp_sub_body(
        &self,
        body: &Block,
        v: Symbol,
        x: Symbol,
        m: Symbol,
        s: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<()> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign { target, op, value } = &stmt.kind else {
            return None;
        };
        if single_path(target) != Some(s) {
            return None;
        }
        let addend: &Expr = match op {
            ast::AssignOp::Add => value,
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(lhs) == Some(s) => rhs,
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(rhs) == Some(s) => lhs,
                _ => return None,
            },
            _ => return None,
        };
        if self.expr_mir(addend) != MirType::F32 {
            return None;
        }
        let ExprKind::Call { callee, args, .. } = &addend.kind else {
            return None;
        };
        if args.len() != 1
            || !matches!(self.vectorizable_intrinsic(callee), Some(MathIntrinsic::Exp))
        {
            return None;
        }
        let ExprKind::Binary {
            op: ast::BinOp::Sub,
            lhs,
            rhs,
        } = &args[0].kind
        else {
            return None;
        };
        if self.index_off(lhs, v, batch) == Some(x) && single_path(rhs) == Some(m) {
            Some(())
        } else {
            None
        }
    }

    /// `let name = log(s)` → name (the log-sum-exp binding; mirror of `match_recip`). Pure.
    fn match_log_of(&self, stmt: &Stmt, s: Symbol) -> Option<Symbol> {
        let (name, init) = Self::let_init(stmt)?;
        let ExprKind::Call { callee, args, .. } = &init.kind else {
            return None;
        };
        if args.len() == 1
            && matches!(self.vectorizable_intrinsic(callee), Some(MathIntrinsic::Log))
            && single_path(&args[0]) == Some(s)
        {
            Some(name)
        } else {
            None
        }
    }

    /// Recognize the **batched softmax cross-entropy forward loss** nest and dispatch it to
    /// `mercury_xent_fwd_f32`. The canonical per-row form (logits `x[R, C]`, integer labels
    /// `target[R]`, scalar loss `loss[R]`):
    ///
    /// ```text
    /// for r in 0..R {
    ///     let mut m = x[r*C + 0];
    ///     for i in 0..C { m = fmax(m, x[r*C + i]); }       // row max (stability)
    ///     let mut s = 0.0;
    ///     for i in 0..C { s = s + exp(x[r*C + i] - m); }   // Σ exp(x − m)
    ///     loss[r] = m + log(s) - x[r*C + target[r]];       // lse − x[target] = −log softmax[target]
    /// }
    /// ```
    ///
    /// The first four statements are softmax's stabilizing max + Σexp passes (the *shared* matchers
    /// `match_max_reduce_body`/`is_max_seed`/`match_sumexp_sub_body`); the fifth is the new scalar store
    /// with the **data-dependent gather** `x[r*C + target[r]]` (`target[r]` an `i32` index). C/Rust keep
    /// the `expf`/`logf` reduction scalar, so the fused 256-bit kernel wins. The reductions reassociate
    /// (the documented exception — both backends run *this* kernel), so the differential gate holds.
    fn match_xent(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<XentNest> {
        let ast::PatKind::Ident(r) = &pat.kind else {
            return None;
        };
        let r = *r;
        let (rs, re) = range_bounds(iter)?;
        if as_int_lit(rs, self.interner)? != 0 {
            return None;
        }
        let rows = as_dim(re, self.interner)?;
        if body.tail.is_some() || body.stmts.len() != 5 {
            return None;
        }
        // [0] let m = x[r*C+0];  [1] for i { m = fmax(m, x[r*C+i]) }
        let (m, seed) = Self::let_init(&body.stmts[0])?;
        let (v1, n_expr, body1) = self.as_range0_for(&body.stmts[1])?;
        let data_batch = Some((r, n_expr));
        let x = self.match_max_reduce_body(body1, v1, m, data_batch)?;
        if !self.is_max_seed(seed, x, data_batch) {
            return None;
        }
        // [2] let s = 0.0;  [3] for i { s = s + exp(x[r*C+i] - m) }
        let (s, s_init) = Self::let_init(&body.stmts[2])?;
        if !matches!(&s_init.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 0.0)
        {
            return None;
        }
        let (v3, n3, body3) = self.as_range0_for(&body.stmts[3])?;
        if !exprs_struct_eq(n3, n_expr) {
            return None;
        }
        self.match_sumexp_sub_body(body3, v3, x, m, s, data_batch)?;
        // [4] loss[r] = m + log(s) - x[r*C + target[r]]
        let StmtKind::Assign {
            target: losst,
            op: ast::AssignOp::Assign,
            value,
        } = &body.stmts[4].kind
        else {
            return None;
        };
        let loss = index_by_var(losst, r)?; // loss[r], indexed by the outer var
        let ExprKind::Binary {
            op: ast::BinOp::Sub,
            lhs: lse,
            rhs: gather,
        } = &value.kind
        else {
            return None;
        };
        self.match_m_plus_log(lse, m, s)?; // lse = m + log(s)
        let target = self.match_xent_gather(gather, x, r, n_expr)?; // x[r*C + target[r]]
        let cols = as_dim(n_expr, self.interner)?;
        if loss == x || loss == target || target == x {
            return None;
        }
        Some(XentNest {
            x,
            target,
            loss,
            rows,
            cols,
        })
    }

    /// Match `m + log(s)` (the log-sum-exp, either operand order): one side is the path `m`, the other a
    /// single-arg `log(s)` call.
    fn match_m_plus_log(&self, e: &Expr, m: Symbol, s: Symbol) -> Option<()> {
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs,
            rhs,
        } = &e.kind
        else {
            return None;
        };
        let is_log_s = |x: &Expr| match &x.kind {
            ExprKind::Call { callee, args, .. } => {
                args.len() == 1
                    && matches!(self.vectorizable_intrinsic(callee), Some(MathIntrinsic::Log))
                    && single_path(&args[0]) == Some(s)
            }
            _ => false,
        };
        if (single_path(lhs) == Some(m) && is_log_s(rhs))
            || (single_path(rhs) == Some(m) && is_log_s(lhs))
        {
            Some(())
        } else {
            None
        }
    }

    /// Match the gather `x[r*C + target[r]]` → the `target` array symbol: a single-index read of `x`
    /// whose index is `r*C + target[r]` (either addend order; `target[r]` may carry an `as` cast to the
    /// index type). `is_mul_of` pins the `r*C` term to the outer row var and the column count.
    fn match_xent_gather(&self, e: &Expr, x: Symbol, r: Symbol, cols: &Expr) -> Option<Symbol> {
        let ExprKind::Index { base, indices } = &e.kind else {
            return None;
        };
        if indices.len() != 1 || single_path(base) != Some(x) {
            return None;
        }
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs,
            rhs,
        } = &indices[0].kind
        else {
            return None;
        };
        fn peel(e: &Expr) -> &Expr {
            match &e.kind {
                ExprKind::Cast { expr, .. } => expr,
                _ => e,
            }
        }
        if self.is_mul_of(lhs, r, cols) {
            return index_by_var(peel(rhs), r);
        }
        if self.is_mul_of(rhs, r, cols) {
            return index_by_var(peel(lhs), r);
        }
        None
    }

    /// Recognize the embedding-lookup nest (the first layer of every LLM — token ids select rows of the
    /// embedding table) and dispatch it to `mercury_embedding_f32`:
    ///
    /// ```text
    /// for t in 0..T { for d in 0..H { out[t*H + d] = weight[ids[t]*H + d]; } }
    /// ```
    ///
    /// `out` (`[T, H]`) is the gather of `weight` (`[V, H]`) rows by the length-`T` `i32` index array
    /// `ids`. The store index is `t*H + d` (`is_mul_of`/`index_off` pin the stride `H` to the inner loop
    /// bound) and the load is `ids[t]*H + d` — the **data-dependent gather** `weight[ids[t], :]`, the row
    /// chosen at runtime by `ids[t]` (`match_embed_load`). `out`/`weight` are f32 arrays, `ids` an `i32`
    /// array; `out` must differ from `weight` and `ids`. Pure data movement (a row copy), so the kernel is
    /// **bit-identical** to this scalar nest on both backends — no float reassociation, the differential
    /// gate is trivial (like the transpose). The naive source has no bounds check, so the emitter passes a
    /// large `V` sentinel and the kernel's out-of-range→zero clamp never fires for in-range ids. Pure
    /// (`&self`).
    fn match_embedding(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<EmbeddingNest> {
        // for t in 0..T { <single inner for> }
        let ast::PatKind::Ident(t) = &pat.kind else {
            return None;
        };
        let t = *t;
        let (ts, te) = range_bounds(iter)?;
        if as_int_lit(ts, self.interner)? != 0 {
            return None;
        }
        let t_rows = as_dim(te, self.interner)?;
        if body.tail.is_some() || body.stmts.len() != 1 {
            return None;
        }
        // for d in 0..H { <single assignment> }
        let (dpat, diter, dbody) = fusable_for(&body.stmts[0])?;
        let ast::PatKind::Ident(d) = &dpat.kind else {
            return None;
        };
        let d = *d;
        let (ds, de) = range_bounds(diter)?;
        if as_int_lit(ds, self.interner)? != 0 {
            return None;
        }
        let h = as_dim(de, self.interner)?;
        if dbody.tail.is_some() || dbody.stmts.len() != 1 {
            return None;
        }
        // out[t*H + d] = weight[ids[t]*H + d];
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &dbody.stmts[0].kind
        else {
            return None;
        };
        // Store: `out[t*H + d]` (row-major, stride H = the inner bound `de`).
        let out = self.index_off(target, d, Some((t, de)))?;
        // Load: `weight[ids[t]*H + d]` — the data-dependent row gather with the same stride.
        let (weight, ids) = self.match_embed_load(value, t, d, de)?;
        // f32 data, i32 indices; `out` distinct from the inputs (a gather never writes its sources).
        if scalar_of(target, self.sema) != Some(mercury_types::Scalar::F32)
            || scalar_of(value, self.sema) != Some(mercury_types::Scalar::F32)
        {
            return None;
        }
        if out == weight || out == ids {
            return None;
        }
        Some(EmbeddingNest {
            out,
            weight,
            ids,
            t_rows,
            h,
        })
    }

    /// Match the embedding load `weight[ids[t]*H + d]` → `(weight, ids)`: a single-index read of an array
    /// `weight` whose flat index is `ids[t]*H + d` (either addend order), where `ids[t]` is itself a
    /// single-index read of an `i32` array `ids` at exactly the outer var `t`. The `*H` stride is pinned
    /// to the inner loop bound `cols` (matching the store's stride) so it never misfires on a non-gather.
    /// Pure.
    fn match_embed_load(
        &self,
        e: &Expr,
        t: Symbol,
        d: Symbol,
        cols: &Expr,
    ) -> Option<(Symbol, Symbol)> {
        let ExprKind::Index { base, indices } = &e.kind else {
            return None;
        };
        if indices.len() != 1 {
            return None;
        }
        let weight = single_path(base)?;
        // index = `ids[t]*H + d` / `d + ids[t]*H`.
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs,
            rhs,
        } = &indices[0].kind
        else {
            return None;
        };
        // `ids[t] * H` (either factor order), with `H` structurally equal to the inner bound, and the
        // other addend exactly the inner var `d`.
        let row_mul = |me: &Self, e: &Expr| -> Option<Symbol> {
            let ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } = &e.kind
            else {
                return None;
            };
            if exprs_struct_eq(rhs, cols) {
                me.match_id_gather(lhs, t)
            } else if exprs_struct_eq(lhs, cols) {
                me.match_id_gather(rhs, t)
            } else {
                None
            }
        };
        let ids = if single_path(rhs) == Some(d) {
            row_mul(self, lhs)?
        } else if single_path(lhs) == Some(d) {
            row_mul(self, rhs)?
        } else {
            return None;
        };
        Some(ids)
            .map(|i| (weight, i))
            .filter(|(w, i)| w != i)
    }

    /// Match `ids[t]` — a single-index read of an `i32` array `ids` at exactly the outer var `t` (an
    /// optional `as` cast to the index type is peeled) → `ids`. The token-id row selector of the
    /// embedding gather. Pure.
    fn match_id_gather(&self, e: &Expr, t: Symbol) -> Option<Symbol> {
        let inner = match &e.kind {
            ExprKind::Cast { expr, .. } => expr.as_ref(),
            _ => e,
        };
        let ids = index_by_var(inner, t)?;
        // `ids` must be an i32 array (the token-id buffer the kernel reads as i32).
        if scalar_of(inner, self.sema) != Some(mercury_types::Scalar::I32) {
            return None;
        }
        Some(ids)
    }

    /// Emit one `mercury_embedding_f32[_parallel](out, weight, ids, T, H, V)` call for a recognized
    /// embedding-lookup nest. Bails (false) if an operand/dim is unbound. `parallel` selects the multicore
    /// kernel (rows independent → bit-identical to serial). `V` is a large `i64` sentinel: the naive
    /// source omits the bounds check, so passing a huge table height keeps the kernel's out-of-range→zero
    /// clamp from ever firing on the in-range ids a well-typed program produces (both backends call the
    /// identical kernel, so the differential gate holds regardless of the sentinel value).
    fn emit_embedding(&mut self, nest: &EmbeddingNest, parallel: bool) -> bool {
        let (Some((out, _)), Some((weight, _)), Some((ids, _))) = (
            self.lookup(nest.out),
            self.lookup(nest.weight),
            self.lookup(nest.ids),
        ) else {
            return false;
        };
        let (Some(t_rows), Some(h)) = (self.dim_value(nest.t_rows), self.dim_value(nest.h)) else {
            return false;
        };
        // `V` sentinel — larger than any realistic vocabulary, so `0 <= id < V` always holds for valid
        // ids (the only case a well-typed gather produces) and the clamp is a no-op, matching the naive
        // unchecked read. `i64::MAX` would risk an `id as usize * h` overflow in the kernel's bounds math
        // for pathological inputs; a still-astronomical 2^48 leaves ample headroom.
        let v = self
            .builder
            .build(MirType::I64, Op::ConstInt(1i128 << 48, MirType::I64));
        let func = if parallel {
            self.gemm.embedding_par
        } else {
            self.gemm.embedding
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![out, weight, ids, t_rows, h, v],
        });
        true
    }

    /// Recognize the **scatter-add / embedding-gradient backward** — the dual of the embedding gather:
    /// ```text
    /// for t in 0..T { for d in 0..H { grad_w[ids[t]*H + d] += grad_out[t*H + d]; } }
    /// ```
    /// `grad_w[ids[t], :] += grad_out[t, :]` → `mercury_scatter_add_f32[_parallel]`. The data-dependent
    /// row index `ids[t]` is on the **write** side (where the gather has it on the read side), and the
    /// op is `+=` (colliding tokens — a word occurring twice — sum into one weight row). The kernel folds
    /// each row's collisions in ascending token order, the *same* order this nest runs, so it is
    /// **bit-identical** to the scalar loop — no reassociation. The `@parallel` kernel splits the output
    /// rows (not the tokens) across cores so writes never collide: lock-free, race-free, and
    /// deterministic == serial (a structural win — a C author would need non-deterministic atomic adds).
    /// `grad_w` must be the zeroed accumulator the program is filling (`+=` into its current contents).
    /// Pure (`&self`).
    fn match_scatter(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<ScatterNest> {
        // for t in 0..T { <single inner for> }
        let ast::PatKind::Ident(t) = &pat.kind else {
            return None;
        };
        let t = *t;
        let (ts, te) = range_bounds(iter)?;
        if as_int_lit(ts, self.interner)? != 0 {
            return None;
        }
        let t_rows = as_dim(te, self.interner)?;
        if body.tail.is_some() || body.stmts.len() != 1 {
            return None;
        }
        // for d in 0..H { <single assignment> }
        let (dpat, diter, dbody) = fusable_for(&body.stmts[0])?;
        let ast::PatKind::Ident(d) = &dpat.kind else {
            return None;
        };
        let d = *d;
        let (ds, de) = range_bounds(diter)?;
        if as_int_lit(ds, self.interner)? != 0 {
            return None;
        }
        let h = as_dim(de, self.interner)?;
        if dbody.tail.is_some() || dbody.stmts.len() != 1 {
            return None;
        }
        // grad_w[ids[t]*H + d] += grad_out[t*H + d];  (`+=` — the embedding gradient sums collisions)
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Add,
            value,
        } = &dbody.stmts[0].kind
        else {
            return None;
        };
        // Store: the data-dependent indirect row `grad_w[ids[t]*H + d]` — the SAME shape as the embedding
        // gather LOAD (`match_embed_load`), but here on the WRITE side (scatter is the gather's dual).
        let (grad_w, ids) = self.match_embed_load(target, t, d, de)?;
        // Read: the plain row-major upstream gradient `grad_out[t*H + d]`.
        let grad_out = self.index_off(value, d, Some((t, de)))?;
        if scalar_of(target, self.sema) != Some(mercury_types::Scalar::F32)
            || scalar_of(value, self.sema) != Some(mercury_types::Scalar::F32)
        {
            return None;
        }
        // The accumulator, the upstream gradient, and the id buffer are three distinct arrays.
        if grad_w == grad_out || grad_w == ids || grad_out == ids {
            return None;
        }
        // V (grad_w's table height) must be REAL for the parallel kernel's output-row partition (a
        // sentinel would pile all work in chunk 0). Recover the total length `V*H` from grad_w's sema
        // array type — present even for an array PARAMETER (MIR lowers it to a bare pointer, but sema
        // keeps the declared `[f32; V*H]`); decline if grad_w is not a statically-sized array.
        let ExprKind::Index { base: gw_base, .. } = &target.kind else {
            return None;
        };
        let total = match self.expr_ty(gw_base) {
            Ty::Array { len, .. } => len,
            _ => return None,
        };
        Some(ScatterNest {
            grad_w,
            grad_out,
            ids,
            t_rows,
            h,
            total,
        })
    }

    /// Emit one `mercury_scatter_add_f32[_parallel](grad_w, grad_out, ids, T, H, V)` call for a recognized
    /// scatter-add. `V = total / H` is the **real** table height (not a sentinel like the embedding's),
    /// derived from grad_w's array length, because the parallel kernel partitions the `V` output rows
    /// across cores. `parallel` selects the multicore kernel (output-row split → bit-identical to serial).
    fn emit_scatter(&mut self, nest: &ScatterNest, parallel: bool) -> bool {
        let (Some((grad_w, _)), Some((grad_out, _)), Some((ids, _))) = (
            self.lookup(nest.grad_w),
            self.lookup(nest.grad_out),
            self.lookup(nest.ids),
        ) else {
            return false;
        };
        let (Some(t_rows), Some(h)) = (self.dim_value(nest.t_rows), self.dim_value(nest.h)) else {
            return false;
        };
        // V = grad_w length / H (integer div; `total = V*H` exactly for a well-formed `[f32; V*H]`).
        let total = self
            .builder
            .build(MirType::I64, Op::ConstInt(nest.total as i128, MirType::I64));
        let v = self.builder.build(MirType::I64, Op::Bin(BinOp::UDiv, total, h));
        let func = if parallel {
            self.gemm.scatter_add_par
        } else {
            self.gemm.scatter_add
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![grad_w, grad_out, ids, t_rows, h, v],
        });
        true
    }

    /// Match `exp(x[r*C+v] - m)` (the recomputed centered exponential) — `true` if `e` is a single-arg
    /// `exp` call whose argument is `x[r*C+v] - m`. Mirrors [`match_exp_sub_body`]'s inner shape.
    fn is_exp_centered(
        &self,
        e: &Expr,
        x: Symbol,
        m: Symbol,
        v: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> bool {
        let ExprKind::Call { callee, args, .. } = &e.kind else {
            return false;
        };
        if args.len() != 1
            || !matches!(self.vectorizable_intrinsic(callee), Some(MathIntrinsic::Exp))
        {
            return false;
        }
        matches!(&args[0].kind, ExprKind::Binary { op: ast::BinOp::Sub, lhs, rhs }
            if self.index_off(lhs, v, batch) == Some(x) && single_path(rhs) == Some(m))
    }

    /// Body `dx[r*C+v] = exp(x[r*C+v] - m) * invZ` (the recompute-softmax write into a *separate* array
    /// `dx`) — returns `dx`. The Mul's factors are the centered exp and the `invZ` reciprocal (either
    /// order). Pure.
    fn match_softmax_recompute_body(
        &self,
        body: &Block,
        v: Symbol,
        x: Symbol,
        m: Symbol,
        invz: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<Symbol> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        let dx = self.index_off(target, v, batch)?;
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &value.kind
        else {
            return None;
        };
        let ok = (self.is_exp_centered(lhs, x, m, v, batch) && single_path(rhs) == Some(invz))
            || (self.is_exp_centered(rhs, x, m, v, batch) && single_path(lhs) == Some(invz));
        if ok {
            Some(dx)
        } else {
            None
        }
    }

    /// Recognize the **batched softmax cross-entropy backward** (input gradient) and dispatch it to
    /// `mercury_xent_bwd_f32`. The canonical per-row form (logits `x[R,C]`, i32 labels `target[R]`,
    /// gradient `dx[R,C]`):
    ///
    /// ```text
    /// for r in 0..R {
    ///     let mut m = x[r*C];
    ///     for i in 0..C { m = fmax(m, x[r*C+i]); }              // row max
    ///     let mut Z = 0.0;
    ///     for i in 0..C { Z = Z + exp(x[r*C+i] - m); }          // Σ exp(x−m)
    ///     let invZ = 1.0 / Z;
    ///     for i in 0..C { dx[r*C+i] = exp(x[r*C+i] - m) * invZ; } // softmax(x)[i]
    ///     dx[r*C + target[r]] = dx[r*C + target[r]] - 1.0;        // − onehot(target) scatter
    /// }
    /// ```
    ///
    /// The gradient `softmax(x) − onehot(target)` of every classifier/LM training step. The softmax
    /// (max + Σexp + recompute) reuses the shared matchers; the trailing **data-dependent scatter** is
    /// new. C/Rust keep the expf reduction + per-element exp scalar, so the fused 256-bit kernel wins.
    /// Reductions reassociate (the documented exception — both backends run the kernel).
    fn match_xent_bwd(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<XentBwdNest> {
        let ast::PatKind::Ident(r) = &pat.kind else {
            return None;
        };
        let r = *r;
        let (rs, re) = range_bounds(iter)?;
        if as_int_lit(rs, self.interner)? != 0 {
            return None;
        }
        let rows = as_dim(re, self.interner)?;
        if body.tail.is_some() || body.stmts.len() != 7 {
            return None;
        }
        let (m, seed) = Self::let_init(&body.stmts[0])?;
        let (v1, n_expr, body1) = self.as_range0_for(&body.stmts[1])?;
        let data_batch = Some((r, n_expr));
        let x = self.match_max_reduce_body(body1, v1, m, data_batch)?;
        if !self.is_max_seed(seed, x, data_batch) {
            return None;
        }
        let (z, z0) = Self::let_init(&body.stmts[2])?;
        if !matches!(&z0.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 0.0) {
            return None;
        }
        let (v3, n3, body3) = self.as_range0_for(&body.stmts[3])?;
        if !exprs_struct_eq(n3, n_expr) {
            return None;
        }
        self.match_sumexp_sub_body(body3, v3, x, m, z, data_batch)?;
        let invz = self.match_recip(&body.stmts[4], z)?;
        let (v5, n5, body5) = self.as_range0_for(&body.stmts[5])?;
        if !exprs_struct_eq(n5, n_expr) {
            return None;
        }
        let dx = self.match_softmax_recompute_body(body5, v5, x, m, invz, data_batch)?;
        // [6] dx[r*C + target[r]] = dx[r*C + target[r]] - 1.0
        let StmtKind::Assign {
            target: scat_t,
            op: ast::AssignOp::Assign,
            value,
        } = &body.stmts[6].kind
        else {
            return None;
        };
        let target = self.match_xent_gather(scat_t, dx, r, n_expr)?;
        let ExprKind::Binary {
            op: ast::BinOp::Sub,
            lhs,
            rhs,
        } = &value.kind
        else {
            return None;
        };
        if !matches!(&rhs.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 1.0) {
            return None;
        }
        if self.match_xent_gather(lhs, dx, r, n_expr)? != target {
            return None;
        }
        let cols = as_dim(n_expr, self.interner)?;
        if dx == x || dx == target || target == x {
            return None;
        }
        Some(XentBwdNest {
            x,
            target,
            dx,
            rows,
            cols,
        })
    }

    /// Recognize the **batched log-sum-exp** (the stable log-partition `out[r] = m + log(Σexp(x[r,·]−m))`,
    /// the softmax denominator in log space — CRF/structured prediction, mixture models, log-prob
    /// normalizers) and dispatch it to `mercury_logsumexp_f32`. It is `match_xent` minus the gather: the
    /// same softmax max + Σexp prefix (shared matchers), then a per-row scalar store `out[r] = m + log(s)`.
    /// C/Rust keep the expf reduction scalar; the fused 256-bit kernel wins. Reductions reassociate (the
    /// documented exception — both backends run the kernel), so the gate holds.
    fn match_logsumexp(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<LogsumexpNest> {
        let ast::PatKind::Ident(r) = &pat.kind else {
            return None;
        };
        let r = *r;
        let (rs, re) = range_bounds(iter)?;
        if as_int_lit(rs, self.interner)? != 0 {
            return None;
        }
        let rows = as_dim(re, self.interner)?;
        if body.tail.is_some() || body.stmts.len() != 5 {
            return None;
        }
        let (m, seed) = Self::let_init(&body.stmts[0])?;
        let (v1, n_expr, body1) = self.as_range0_for(&body.stmts[1])?;
        let data_batch = Some((r, n_expr));
        let x = self.match_max_reduce_body(body1, v1, m, data_batch)?;
        if !self.is_max_seed(seed, x, data_batch) {
            return None;
        }
        let (s, s_init) = Self::let_init(&body.stmts[2])?;
        if !matches!(&s_init.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 0.0)
        {
            return None;
        }
        let (v3, n3, body3) = self.as_range0_for(&body.stmts[3])?;
        if !exprs_struct_eq(n3, n_expr) {
            return None;
        }
        self.match_sumexp_sub_body(body3, v3, x, m, s, data_batch)?;
        // [4] out[r] = m + log(s)
        let StmtKind::Assign {
            target: outt,
            op: ast::AssignOp::Assign,
            value,
        } = &body.stmts[4].kind
        else {
            return None;
        };
        let out = index_by_var(outt, r)?;
        self.match_m_plus_log(value, m, s)?;
        let cols = as_dim(n_expr, self.interner)?;
        if out == x {
            return None;
        }
        Some(LogsumexpNest {
            x,
            out,
            rows,
            cols,
        })
    }

    /// Recognize a batched per-row argmax/argmin returning an **index** (the classification-head / greedy-
    /// decode top-1):
    /// ```text
    /// for r in 0..R {
    ///     let mut bv: f32 = x[r*C];               // seed value = the row's element 0
    ///     let mut bi = 0;                          // seed index = 0
    ///     for j in <0|1>..C { if x[r*C+j] CMP bv { bv = x[r*C+j]; bi = j; } }
    ///     out[r] = bi;                             // out is an i32 array
    /// }
    /// ```
    /// `CMP` is `>` (argmax) or `<` (argmin) — a **strict** compare, so the lowest index wins on a value
    /// tie, exactly the `mercury_rowarg{max,min}_i32` semantics. The seed pins index 0 (`bv = x[r*C]`,
    /// `bi = 0`); the inner loop may start at 0 or 1 (index 0 is the seed either way — a strict compare
    /// never displaces it) and covers the rest of the row, so the source scans the whole row `[0,C)`,
    /// equal to the kernel. `out` must be an **i32** array (the kernel writes 4-byte indices; a wider
    /// slot would stride-mismatch in native), and must differ from `x`. Pure (`&self`).
    fn match_rowarg(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<RowArgNest> {
        let ast::PatKind::Ident(r) = &pat.kind else {
            return None;
        };
        let r = *r;
        let (rs, re) = range_bounds(iter)?;
        if as_int_lit(rs, self.interner)? != 0 {
            return None;
        }
        let rows = as_dim(re, self.interner)?;
        if body.tail.is_some() || body.stmts.len() != 4 {
            return None;
        }
        // [0] let bv: f32 = x[r*C]   (the running-best value, seeded to the row's element 0)
        let (bv, bv_init) = Self::let_init(&body.stmts[0])?;
        // [1] let bi = 0             (the running-best index, seeded to 0)
        let (bi, bi_init) = Self::let_init(&body.stmts[1])?;
        if !is_int_zero(bi_init, self.interner) {
            return None;
        }
        // [2] for j in <0|1>..C { if x[r*C+j] CMP bv { bv = x[r*C+j]; bi = j } }
        let StmtKind::For {
            pat: jp,
            iter: jit,
            body: jb,
            ..
        } = &body.stmts[2].kind
        else {
            return None;
        };
        let ast::PatKind::Ident(j) = &jp.kind else {
            return None;
        };
        let j = *j;
        let ForIter::Range {
            start: js,
            end: Some(je),
            inclusive: false,
            step: None,
        } = jit
        else {
            return None;
        };
        let je: &Expr = je;
        let jstart = as_int_lit(js, self.interner)?;
        if jstart != 0 && jstart != 1 {
            return None;
        }
        let batch = Some((r, je));
        let (x, is_max) = self.match_rowarg_inner(jb, j, bv, bi, batch)?;
        // The seed value must be exactly `x[r*C]` (the row's element 0): `Index { base == x, idx == r*C }`.
        let ExprKind::Index { base, indices } = &bv_init.kind else {
            return None;
        };
        if indices.len() != 1
            || !self.is_mul_of(&indices[0], r, je)
            || single_path(base) != Some(x)
        {
            return None;
        }
        // [3] out[r] = bi  — `out` an i32 array distinct from `x`.
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &body.stmts[3].kind
        else {
            return None;
        };
        let out = index_by_var(target, r)?;
        if single_path(value) != Some(bi) || out == x {
            return None;
        }
        if scalar_of(target, self.sema) != Some(mercury_types::Scalar::I32) {
            return None;
        }
        let cols = as_dim(je, self.interner)?;
        Some(RowArgNest {
            x,
            out,
            rows,
            cols,
            is_max,
        })
    }

    /// The argmax/argmin inner body `if x[r*C+j] CMP bv { bv = x[r*C+j]; bi = j; }` (strict `>` → argmax /
    /// `<` → argmin; `bi = j` may be `j as <int>`), returning `(x_array, is_max)`. The lone `if` may be
    /// the body's single statement or its tail. Mirrors [`match_argreduce_kernel`] with a row-major batch
    /// offset (`index_off` with `batch`). Pure.
    fn match_rowarg_inner(
        &self,
        body: &Block,
        j: Symbol,
        bv: Symbol,
        bi: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<(Symbol, bool)> {
        let if_expr = match (body.stmts.as_slice(), &body.tail) {
            ([only], None) => match &only.kind {
                StmtKind::Expr(e) => e,
                _ => return None,
            },
            ([], Some(e)) => e.as_ref(),
            _ => return None,
        };
        let ExprKind::If {
            cond,
            then_branch,
            else_branch: None,
        } = &if_expr.kind
        else {
            return None;
        };
        let ExprKind::Binary { op, lhs, rhs } = &cond.kind else {
            return None;
        };
        // Canonical `x[r*C+j] CMP bv`: row-major data read on the left, running-best scalar on the right.
        let is_max = match op {
            ast::BinOp::Gt => true,
            ast::BinOp::Lt => false,
            _ => return None,
        };
        let x = self.index_off(lhs, j, batch)?;
        if single_path(rhs) != Some(bv) {
            return None;
        }
        // then-branch: exactly `bv = x[r*C+j]; bi = j;` (order-flexible), no tail value.
        if then_branch.tail.is_some() || then_branch.stmts.len() != 2 {
            return None;
        }
        let (mut saw_val, mut saw_idx) = (false, false);
        for s in &then_branch.stmts {
            let StmtKind::Assign {
                target,
                op: ast::AssignOp::Assign,
                value,
            } = &s.kind
            else {
                return None;
            };
            let t = single_path(target)?;
            if t == bv {
                // bv = x[r*C+j]
                if self.index_off(value, j, batch) != Some(x) {
                    return None;
                }
                saw_val = true;
            } else if t == bi {
                // bi = j  (or `bi = j as <int>`)
                let is_j = single_path(value) == Some(j)
                    || matches!(&value.kind, ExprKind::Cast { expr, .. } if single_path(expr) == Some(j));
                if !is_j {
                    return None;
                }
                saw_idx = true;
            } else {
                return None;
            }
        }
        if saw_val && saw_idx {
            Some((x, is_max))
        } else {
            None
        }
    }

    /// Recognize a batched per-row **inclusive prefix sum** (cumsum / scan):
    /// ```text
    /// for r in 0..R {
    ///     let mut acc: f32 = 0.0;
    ///     for i in 0..C {
    ///         acc = acc + x[r*C + i];     // or  acc += x[r*C + i]
    ///         out[r*C + i] = acc;
    ///     }
    /// }
    /// ```
    /// `out[r,i] = Σ_{k<=i} x[r,k]` → `mercury_cumsum_f32[_parallel]`. The loop-carried `acc` recurrence
    /// is exactly what gcc/rustc keep **scalar** (they cannot auto-vectorize a prefix sum); the SIMD
    /// Hillis-Steele scan + per-row carry vectorizes it. `out` is f32, distinct from `x`. The in-lane tree
    /// scan reassociates the float sum (the documented reduction exception — both backends run the
    /// identical kernel, so interp == native holds). Pure (`&self`).
    fn match_cumsum(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<CumsumNest> {
        let ast::PatKind::Ident(r) = &pat.kind else {
            return None;
        };
        let r = *r;
        let (rs, re) = range_bounds(iter)?;
        if as_int_lit(rs, self.interner)? != 0 {
            return None;
        }
        let rows = as_dim(re, self.interner)?;
        if body.tail.is_some() || body.stmts.len() != 2 {
            return None;
        }
        // [0] let acc: f32 = 0.0;
        let (acc, acc0) = Self::let_init(&body.stmts[0])?;
        if !is_float_zero(acc0, self.interner) {
            return None;
        }
        // [1] for i in 0..C { acc = acc + x[r*C+i]; out[r*C+i] = acc; }
        let (i, ce, ibody) = self.as_range0_for(&body.stmts[1])?;
        let cols = as_dim(ce, self.interner)?;
        let batch = Some((r, ce));
        if ibody.tail.is_some() || ibody.stmts.len() != 2 {
            return None;
        }
        // inner [0] acc = acc + x[r*C+i]  (or  acc += x[r*C+i])
        let x = self.match_acc_add(&ibody.stmts[0], acc, i, batch)?;
        // inner [1] out[r*C+i] = acc
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &ibody.stmts[1].kind
        else {
            return None;
        };
        let out = self.index_off(target, i, batch)?;
        if single_path(value) != Some(acc) || out == x {
            return None;
        }
        if scalar_of(target, self.sema) != Some(mercury_types::Scalar::F32) {
            return None;
        }
        Some(CumsumNest {
            x,
            out,
            rows,
            cols,
        })
    }

    /// Recognize a batched per-row **inclusive prefix product** (cumulative product / scan):
    /// ```text
    /// for r in 0..R {
    ///     let mut p: f32 = 1.0;            // multiplicative identity, resets every row
    ///     for i in 0..C { p = p * x[r*C+i]; out[r*C+i] = p; }
    /// }
    /// ```
    /// `out[r,i] = Π_{k<=i} x[r,k]` → `mercury_cumprod_f32[_parallel]`. Mirrors [`Self::match_cumsum`]
    /// exactly but with the multiplicative seed `1.0` and a `*` accumulate. The loop-carried product is
    /// what gcc/rustc keep scalar; the kernel's 4-row-interleaved ILP wins. **Bit-exact** — a bare product
    /// is not fused, so it folds strictly left-to-right == the scalar nest (no reassociation, unlike the
    /// prefix sum). `out` is f32, distinct from `x`. Pure (`&self`).
    fn match_cumprod(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<CumsumNest> {
        let ast::PatKind::Ident(r) = &pat.kind else {
            return None;
        };
        let r = *r;
        let (rs, re) = range_bounds(iter)?;
        if as_int_lit(rs, self.interner)? != 0 {
            return None;
        }
        let rows = as_dim(re, self.interner)?;
        if body.tail.is_some() || body.stmts.len() != 2 {
            return None;
        }
        // [0] let p: f32 = 1.0;  (multiplicative identity)
        let (p, p_init) = Self::let_init(&body.stmts[0])?;
        if !matches!(&p_init.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 1.0) {
            return None;
        }
        // [1] for i in 0..C { p = p * x[r*C+i]; out[r*C+i] = p; }
        let (i, ce, ibody) = self.as_range0_for(&body.stmts[1])?;
        let cols = as_dim(ce, self.interner)?;
        let batch = Some((r, ce));
        if ibody.tail.is_some() || ibody.stmts.len() != 2 {
            return None;
        }
        let x = self.match_prod_body(&ibody.stmts[0], p, i, batch)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &ibody.stmts[1].kind
        else {
            return None;
        };
        let out = self.index_off(target, i, batch)?;
        if single_path(value) != Some(p) || out == x {
            return None;
        }
        if scalar_of(target, self.sema) != Some(mercury_types::Scalar::F32) {
            return None;
        }
        Some(CumsumNest {
            x,
            out,
            rows,
            cols,
        })
    }

    /// The prefix-product accumulate `p = p * x[r*C+i]` (or the compound `p *= x[r*C+i]`), target already
    /// `p`. Returns the data array `x` (the read indexed `r*C + i`). The multiplicative twin of
    /// [`Self::match_acc_add`]. Pure.
    fn match_prod_body(
        &self,
        stmt: &Stmt,
        acc: Symbol,
        i: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<Symbol> {
        let StmtKind::Assign { target, op, value } = &stmt.kind else {
            return None;
        };
        if single_path(target) != Some(acc) {
            return None;
        }
        match op {
            // `p *= x[r*C+i]`
            ast::AssignOp::Mul => self.index_off(value, i, batch),
            // `p = p * x[r*C+i]` (either operand order)
            ast::AssignOp::Assign => {
                let ExprKind::Binary {
                    op: ast::BinOp::Mul,
                    lhs,
                    rhs,
                } = &value.kind
                else {
                    return None;
                };
                if single_path(lhs) == Some(acc) {
                    self.index_off(rhs, i, batch)
                } else if single_path(rhs) == Some(acc) {
                    self.index_off(lhs, i, batch)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// The prefix-sum accumulate `acc = acc + x[r*C+i]` (or the compound `acc += x[r*C+i]`), target
    /// already `acc`. Returns the data array `x` (the read indexed `r*C + i`). Pure.
    fn match_acc_add(
        &self,
        stmt: &Stmt,
        acc: Symbol,
        i: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<Symbol> {
        let StmtKind::Assign { target, op, value } = &stmt.kind else {
            return None;
        };
        if single_path(target) != Some(acc) {
            return None;
        }
        match op {
            // `acc += x[r*C+i]`
            ast::AssignOp::Add => self.index_off(value, i, batch),
            // `acc = acc + x[r*C+i]` (either operand order)
            ast::AssignOp::Assign => {
                let ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } = &value.kind
                else {
                    return None;
                };
                if single_path(lhs) == Some(acc) {
                    self.index_off(rhs, i, batch)
                } else if single_path(rhs) == Some(acc) {
                    self.index_off(lhs, i, batch)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Recognize a batched per-row **first-order linear recurrence / selective scan** (SSM/Mamba/EMA):
    /// ```text
    /// for r in 0..R {
    ///     let mut h: f32 = 0.0;              // zero initial state, resets every row
    ///     for t in 0..C {
    ///         h = a[r*C + t] * h + b[r*C + t];  // h_t = gate·h_{t-1} + input
    ///         out[r*C + t] = h;
    ///     }
    /// }
    /// ```
    /// `out[r,t] = a[r,t]·h_{t-1} + b[r,t]` → `mercury_lrscan_f32[_parallel]`. The carry `h` makes the
    /// inner loop a true loop-carried dependency that gcc/rustc keep **scalar** (like cumsum); rows are
    /// independent so the `_parallel` map across rows is the lever. **No reassociation** is possible (the
    /// recurrence is inherently sequential within a row), so the kernel is bit-identical to the scalar
    /// nest — the interpreter marshals the *same* kernel. `out` is f32, distinct from `a`/`b`. Pure.
    fn match_lrscan(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<LrscanNest> {
        let ast::PatKind::Ident(r) = &pat.kind else {
            return None;
        };
        let r = *r;
        let (rs, re) = range_bounds(iter)?;
        if as_int_lit(rs, self.interner)? != 0 {
            return None;
        }
        let rows = as_dim(re, self.interner)?;
        if body.tail.is_some() || body.stmts.len() != 2 {
            return None;
        }
        // [0] let h: f32 = 0.0;  (zero initial hidden state)
        let (h, h0) = Self::let_init(&body.stmts[0])?;
        if !is_float_zero(h0, self.interner) {
            return None;
        }
        // [1] for t in 0..C { h = a[r*C+t]*h + b[r*C+t]; out[r*C+t] = h; }
        let (t, ce, ibody) = self.as_range0_for(&body.stmts[1])?;
        let cols = as_dim(ce, self.interner)?;
        let batch = Some((r, ce));
        if ibody.tail.is_some() || ibody.stmts.len() != 2 {
            return None;
        }
        // inner [0] the recurrence step h = a[r*C+t]*h + b[r*C+t]  →  (a, b)
        let (a, b) = self.match_lrscan_step(&ibody.stmts[0], h, t, batch)?;
        // inner [1] out[r*C+t] = h
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &ibody.stmts[1].kind
        else {
            return None;
        };
        let out = self.index_off(target, t, batch)?;
        if single_path(value) != Some(h) || out == a || out == b {
            return None;
        }
        if scalar_of(target, self.sema) != Some(mercury_types::Scalar::F32) {
            return None;
        }
        Some(LrscanNest {
            a,
            b,
            out,
            rows,
            cols,
        })
    }

    /// The recurrence step `h = a[r*C+t]·h + b[r*C+t]` (target already the carried scalar `h`): returns
    /// `(a, b)` — `a` the per-step gate (multiplied by the carry), `b` the per-step input (added). The
    /// top-level op is an `Add` whose one operand is the gated carry `a[..]·h` and whose other is the
    /// plain data read `b[..]` (either addend order). Pure.
    fn match_lrscan_step(
        &self,
        stmt: &Stmt,
        h: Symbol,
        t: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<(Symbol, Symbol)> {
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if single_path(target) != Some(h) {
            return None;
        }
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs,
            rhs,
        } = &value.kind
        else {
            return None;
        };
        if let Some(a) = self.match_gated_carry(lhs, h, t, batch) {
            let b = self.index_off(rhs, t, batch)?;
            Some((a, b))
        } else if let Some(a) = self.match_gated_carry(rhs, h, t, batch) {
            let b = self.index_off(lhs, t, batch)?;
            Some((a, b))
        } else {
            None
        }
    }

    /// `a[r*C+t] · h` (either factor order, `h` the carried scalar) → the gate array `a` (the factor
    /// indexed `r*C + t`). Pure.
    fn match_gated_carry(
        &self,
        e: &Expr,
        h: Symbol,
        t: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<Symbol> {
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &e.kind
        else {
            return None;
        };
        if single_path(rhs) == Some(h) {
            self.index_off(lhs, t, batch)
        } else if single_path(lhs) == Some(h) {
            self.index_off(rhs, t, batch)
        } else {
            None
        }
    }

    /// Recognize a batched per-row **cumulative max / min** (running max/min scan):
    /// ```text
    /// for r in 0..R {
    ///     let mut m: f32 = x[r*C];      // seed = the row's first element (or a <= -1e30 / >= 1e30 sentinel)
    ///     for i in 0..C {
    ///         m = fmax(m, x[r*C + i]);   // or fmin  — `cummin`
    ///         out[r*C + i] = m;
    ///     }
    /// }
    /// ```
    /// `out[r,i] = max/min_{k<=i} x[r,k]` → `mercury_cummax_f32[_parallel]` / `mercury_cummin_f32[_…]`.
    /// gcc/rustc keep the loop-carried `out[i]=fmax(out[i-1],x[i])` scalar; the SIMD in-lane max/min scan
    /// vectorizes it. **No reassociation** — max/min select an input value, so the kernel is *bit-exact*
    /// vs the scalar scan (unlike cumsum). `out` is f32, distinct from `x`. Pure (`&self`).
    fn match_cumminmax(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<CumMinMaxNest> {
        let ast::PatKind::Ident(r) = &pat.kind else {
            return None;
        };
        let r = *r;
        let (rs, re) = range_bounds(iter)?;
        if as_int_lit(rs, self.interner)? != 0 {
            return None;
        }
        let rows = as_dim(re, self.interner)?;
        if body.tail.is_some() || body.stmts.len() != 2 {
            return None;
        }
        // [0] let m = <seed>;
        let (m, seed) = Self::let_init(&body.stmts[0])?;
        // [1] for i in 0..C { m = fmax/fmin(m, x[r*C+i]); out[r*C+i] = m; }
        let (i, ce, ibody) = self.as_range0_for(&body.stmts[1])?;
        let cols = as_dim(ce, self.interner)?;
        let batch = Some((r, ce));
        let (x, out, is_max) = self.match_cumminmax_inner(ibody, i, m, batch)?;
        // The seed must be `<= max(x)` (max) / `>= min(x)` (min) so `fXX(seed, x[0])` == `x[0]` (the
        // kernel's out[0]): the row's first element `x[r*C]`, or an extreme sentinel.
        if !self.is_cum_seed(seed, x, batch, is_max) || out == x {
            return None;
        }
        Some(CumMinMaxNest {
            x,
            out,
            rows,
            cols,
            is_max,
        })
    }

    /// The cumulative max/min inner body `m = fmax(m, x[r*C+i]); out[r*C+i] = m;` (or `fmin`), returning
    /// `(x, out, is_max)`. Pure.
    fn match_cumminmax_inner(
        &self,
        body: &Block,
        i: Symbol,
        m: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<(Symbol, Symbol, bool)> {
        if body.tail.is_some() || body.stmts.len() != 2 {
            return None;
        }
        // [0] m = fmax/fmin(m, x[r*C+i])
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &body.stmts[0].kind
        else {
            return None;
        };
        if single_path(target) != Some(m) {
            return None;
        }
        let ExprKind::Call { callee, args, .. } = &value.kind else {
            return None;
        };
        if args.len() != 2 {
            return None;
        }
        let is_max = match self.vectorizable_intrinsic(callee) {
            Some(MathIntrinsic::Fmax) => true,
            Some(MathIntrinsic::Fmin) => false,
            _ => return None,
        };
        let xside = if single_path(&args[0]) == Some(m) {
            1
        } else if single_path(&args[1]) == Some(m) {
            0
        } else {
            return None;
        };
        let x = self.index_off(&args[xside], i, batch)?;
        // [1] out[r*C+i] = m
        let StmtKind::Assign {
            target: ot,
            op: ast::AssignOp::Assign,
            value: ov,
        } = &body.stmts[1].kind
        else {
            return None;
        };
        let out = self.index_off(ot, i, batch)?;
        if single_path(ov) != Some(m) || scalar_of(ot, self.sema) != Some(mercury_types::Scalar::F32) {
            return None;
        }
        Some((x, out, is_max))
    }

    /// The cumulative-max/min seed is valid iff `fXX(seed, x[r*C])` equals `x[r*C]` for any data: the
    /// row's first element `x[r*C]`, or an extreme sentinel (`<= -1e30` for max, `>= 1e30` for min). The
    /// max/`x[r*C]` cases match [`is_max_seed`]; this generalizes it to min. Pure.
    fn is_cum_seed(
        &self,
        init: &Expr,
        x: Symbol,
        batch: Option<(Symbol, &Expr)>,
        is_max: bool,
    ) -> bool {
        if let ExprKind::Index { base, indices } = &init.kind {
            if single_path(base) != Some(x) || indices.len() != 1 {
                return false;
            }
            return match batch {
                None => {
                    matches!(&indices[0].kind, ExprKind::Int(t) if parse_int(self.interner.resolve(*t)) == 0)
                }
                Some((row, cols)) => self.is_mul_of(&indices[0], row, cols),
            };
        }
        let v = match &init.kind {
            ExprKind::Float(t) => parse_float(self.interner.resolve(*t)),
            ExprKind::Unary {
                op: ast::UnOp::Neg,
                expr,
            } => match &expr.kind {
                ExprKind::Float(t) => -parse_float(self.interner.resolve(*t)),
                _ => return false,
            },
            _ => return false,
        };
        if is_max {
            v <= -1e30
        } else {
            v >= 1e30
        }
    }

    /// Match `log(arr[r*C+v])` → the indexed array `arr` (a single-arg `log` call over a row-major read).
    fn match_log_index(&self, e: &Expr, v: Symbol, batch: Option<(Symbol, &Expr)>) -> Option<Symbol> {
        let ExprKind::Call { callee, args, .. } = &e.kind else {
            return None;
        };
        if args.len() != 1 || !matches!(self.vectorizable_intrinsic(callee), Some(MathIntrinsic::Log))
        {
            return None;
        }
        self.index_off(&args[0], v, batch)
    }

    /// The shared head of the per-row reduction losses: `for r in 0..R { let s = 0.0; for i in 0..C {
    /// s = s + <term(i)> } <store> }` — returns `(r, rows, s, iv, n_expr, term, store_value)` where
    /// `term` is the per-element addend and `store_value` is `<store>`'s RHS. Verifies the 3-statement
    /// shape, the `0.0` seed, the inner `0..C` reduction, and that the store targets `out[r]`.
    #[allow(clippy::type_complexity)]
    fn match_row_reduce_head<'b>(
        &self,
        pat: &Pattern,
        iter: &ForIter,
        body: &'b Block,
    ) -> Option<(Symbol, Dim, Symbol, Symbol, &'b Expr, Symbol, &'b Expr)> {
        let ast::PatKind::Ident(r) = &pat.kind else {
            return None;
        };
        let r = *r;
        let (rs, re) = range_bounds(iter)?;
        if as_int_lit(rs, self.interner)? != 0 {
            return None;
        }
        let rows = as_dim(re, self.interner)?;
        if body.tail.is_some() || body.stmts.len() != 3 {
            return None;
        }
        let (s, s0) = Self::let_init(&body.stmts[0])?;
        if !matches!(&s0.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 0.0) {
            return None;
        }
        let (iv, n_expr, b1) = self.as_range0_for(&body.stmts[1])?;
        let term = match_add_accum(b1, s)?;
        // [2] out[r] = s   OR   out[r] = -s  (caller checks the sign and binds out)
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &body.stmts[2].kind
        else {
            return None;
        };
        let out = index_by_var(target, r)?;
        // Caller validates `value` (s or -s) and the term; we hand back the loop var/dims/term.
        let _ = value;
        Some((r, rows, s, iv, n_expr, out, term))
    }

    /// Recognize the **batched KL divergence** `out[r] = Σ_i p[r,i]·(log(p[r,i]) − log(q[r,i]))` (the
    /// knowledge-distillation / VAE loss) and dispatch it to `mercury_kldiv_f32`. C/Rust keep the logf
    /// reduction scalar. The reduction reassociates (the documented exception).
    fn match_kldiv(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<KldivNest> {
        let (r, rows, s, iv, n_expr, out, term) = self.match_row_reduce_head(pat, iter, body)?;
        // store must be `out[r] = s`
        let StmtKind::Assign { value, .. } = &body.stmts[2].kind else {
            return None;
        };
        if single_path(value) != Some(s) {
            return None;
        }
        let batch = Some((r, n_expr));
        // term = p[r*C+iv] * (log(p[r*C+iv]) - log(q[r*C+iv]))
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &term.kind
        else {
            return None;
        };
        // Identify the `p[r*C+iv]` factor and the `(log p - log q)` factor.
        let (p, diff) = if let Some(p) = self.index_off(lhs, iv, batch) {
            (p, rhs.as_ref())
        } else if let Some(p) = self.index_off(rhs, iv, batch) {
            (p, lhs.as_ref())
        } else {
            return None;
        };
        let ExprKind::Binary {
            op: ast::BinOp::Sub,
            lhs: lp,
            rhs: lq,
        } = &diff.kind
        else {
            return None;
        };
        if self.match_log_index(lp, iv, batch) != Some(p) {
            return None;
        }
        let q = self.match_log_index(lq, iv, batch)?;
        let cols = as_dim(n_expr, self.interner)?;
        if out == p || out == q {
            return None;
        }
        Some(KldivNest {
            p,
            q,
            out,
            rows,
            cols,
        })
    }

    /// Recognize the **batched Shannon entropy** `out[r] = − Σ_i p[r,i]·log(p[r,i])` (RL policy entropy)
    /// → `mercury_entropy_f32`. The store is `out[r] = -s`. C/Rust keep the logf reduction scalar.
    fn match_entropy(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<EntropyNest> {
        let (r, rows, s, iv, n_expr, out, term) = self.match_row_reduce_head(pat, iter, body)?;
        // store must be `out[r] = -s` (unary negation of the accumulator)
        let StmtKind::Assign { value, .. } = &body.stmts[2].kind else {
            return None;
        };
        let ExprKind::Unary {
            op: ast::UnOp::Neg,
            expr,
        } = &value.kind
        else {
            return None;
        };
        if single_path(expr) != Some(s) {
            return None;
        }
        let batch = Some((r, n_expr));
        // term = p[r*C+iv] * log(p[r*C+iv])
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &term.kind
        else {
            return None;
        };
        let p = if let Some(p) = self.index_off(lhs, iv, batch) {
            if self.match_log_index(rhs, iv, batch) != Some(p) {
                return None;
            }
            p
        } else if let Some(p) = self.index_off(rhs, iv, batch) {
            if self.match_log_index(lhs, iv, batch) != Some(p) {
                return None;
            }
            p
        } else {
            return None;
        };
        let cols = as_dim(n_expr, self.interner)?;
        if out == p {
            return None;
        }
        Some(EntropyNest {
            p,
            out,
            rows,
            cols,
        })
    }

    /// Recognize the **batched soft-label cross-entropy** (distillation loss) `out[r] = Σ_i q[r,i]·(lse
    /// − x[r,i])`, `lse = m + log(Σexp(x[r,·]−m))` → `mercury_kd_loss_f32`. The hard-label `xent`'s
    /// generalization to a full target distribution `q`. Reuses xent's max+Σexp+lse prefix. C/Rust keep
    /// the expf/logf reductions scalar.
    fn match_kd_loss(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<KdLossNest> {
        let ast::PatKind::Ident(r) = &pat.kind else {
            return None;
        };
        let r = *r;
        let (rs, re) = range_bounds(iter)?;
        if as_int_lit(rs, self.interner)? != 0 {
            return None;
        }
        let rows = as_dim(re, self.interner)?;
        if body.tail.is_some() || body.stmts.len() != 8 {
            return None;
        }
        // [0..4] the xent prefix: max + Σexp + lse = m + log(z)
        let (m, seed) = Self::let_init(&body.stmts[0])?;
        let (v1, n_expr, body1) = self.as_range0_for(&body.stmts[1])?;
        let data_batch = Some((r, n_expr));
        let x = self.match_max_reduce_body(body1, v1, m, data_batch)?;
        if !self.is_max_seed(seed, x, data_batch) {
            return None;
        }
        let (z, z0) = Self::let_init(&body.stmts[2])?;
        if !matches!(&z0.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 0.0) {
            return None;
        }
        let (v3, n3, body3) = self.as_range0_for(&body.stmts[3])?;
        if !exprs_struct_eq(n3, n_expr) {
            return None;
        }
        self.match_sumexp_sub_body(body3, v3, x, m, z, data_batch)?;
        let (lse, lse0) = Self::let_init(&body.stmts[4])?;
        self.match_m_plus_log(lse0, m, z)?;
        // [5] let s = 0.0;  [6] for i { s = s + q[r*C+i]*(lse - x[r*C+i]) };  [7] out[r] = s
        let (s, s0) = Self::let_init(&body.stmts[5])?;
        if !matches!(&s0.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 0.0) {
            return None;
        }
        let (v6, n6, body6) = self.as_range0_for(&body.stmts[6])?;
        if !exprs_struct_eq(n6, n_expr) {
            return None;
        }
        let term = match_add_accum(body6, s)?;
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &term.kind
        else {
            return None;
        };
        // one factor is q[r*C+v6], the other is (lse - x[r*C+v6])
        let is_lse_sub = |e: &Expr| {
            matches!(&e.kind, ExprKind::Binary { op: ast::BinOp::Sub, lhs, rhs }
                if single_path(lhs) == Some(lse) && self.index_off(rhs, v6, data_batch) == Some(x))
        };
        let q = if let Some(q) = self.index_off(lhs, v6, data_batch) {
            if !is_lse_sub(rhs) {
                return None;
            }
            q
        } else if let Some(q) = self.index_off(rhs, v6, data_batch) {
            if !is_lse_sub(lhs) {
                return None;
            }
            q
        } else {
            return None;
        };
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &body.stmts[7].kind
        else {
            return None;
        };
        let out = index_by_var(target, r)?;
        if single_path(value) != Some(s) {
            return None;
        }
        let cols = as_dim(n_expr, self.interner)?;
        if out == x || out == q {
            return None;
        }
        Some(KdLossNest {
            x,
            q,
            out,
            rows,
            cols,
        })
    }

    /// Body `x[v] = (x[v] - m) - ls` (center, then subtract the log-sum-exp, in place). The outermost
    /// `Sub` is by `ls`; the inner is the centered `x[v] - m` (`is_centered`). Pure.
    fn match_logsoftmax_norm_body(
        &self,
        body: &Block,
        v: Symbol,
        x: Symbol,
        m: Symbol,
        ls: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<()> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if self.index_off(target, v, batch) != Some(x) {
            return None;
        }
        let ExprKind::Binary {
            op: ast::BinOp::Sub,
            lhs,
            rhs,
        } = &value.kind
        else {
            return None;
        };
        if single_path(rhs) == Some(ls) && self.is_centered(lhs, v, x, m, batch) {
            Some(())
        } else {
            None
        }
    }

    /// Emit one in-place recognized norm: `mercury_norm_f32(x, x, 1, N, eps_bits, op)` for a plain
    /// (gamma=1, beta=0) norm, or `mercury_norm_affine_f32(x, x, gamma, beta, 1, N, eps_bits, op)` when
    /// the normalize step carried a per-column scale `gamma` (and optional shift `beta`) — the real
    /// transformer form. Bails (false) if the data array or a captured affine array is somehow unbound,
    /// so the caller lowers the loops normally.
    fn emit_norm(
        &mut self,
        arr: Symbol,
        rows: Option<&Expr>,
        n: &Expr,
        eps_bits: i64,
        op: i64,
        gamma: Option<Symbol>,
        beta: Option<Symbol>,
    ) -> bool {
        let Some((xv, _)) = self.lookup(arr) else {
            return false;
        };
        // A batched norm (`rows > 1`) inside a `@parallel` function maps its independent rows across
        // cores via the multicore kernel; rows are normalized independently (no cross-row combine), so
        // it stays bit-equal to the serial kernel the interpreter marshals. A single-row norm has no
        // row parallelism, so it stays serial even in a `@parallel` function (one row = no speedup).
        let batched_parallel = self.parallel_fn && rows.is_some();
        let n_ty = self.expr_mir(n);
        let nval = self.lower_expr(n);
        let nval = self.coerce_to(nval, &n_ty, &MirType::I64, true);
        // `rows` is 1 for a single-row norm, or the batched outer-loop trip count `R` (a `[R, N]`
        // matrix). The kernel normalizes each of the `rows` rows independently — and the `_parallel`
        // variant maps that across cores, which only does real work when `rows > 1`.
        let rows = match rows {
            None => self
                .builder
                .build(MirType::I64, Op::ConstInt(1, MirType::I64)),
            Some(r) => {
                let rty = self.expr_mir(r);
                let rv = self.lower_expr(r);
                self.coerce_to(rv, &rty, &MirType::I64, true)
            }
        };
        let epsv = self
            .builder
            .build(MirType::I64, Op::ConstInt(eps_bits as i128, MirType::I64));
        let opv = self
            .builder
            .build(MirType::I64, Op::ConstInt(op as i128, MirType::I64));
        if gamma.is_none() && beta.is_none() {
            let func = if batched_parallel {
                self.gemm.norm_par
            } else {
                self.gemm.norm
            };
            self.builder.build_void(Op::Call {
                func,
                args: vec![xv, xv, rows, nval, epsv, opv],
            });
            return true;
        }
        // Affine: resolve gamma/beta to their array base pointers. An absent param is a null pointer,
        // built as an integer `0` (a `Ptr`-typed `ConstInt` is invalid MIR; the verifier requires
        // integer-typed int consts). On the native side ptr_ty == i64, so the i64 zero is passed as the
        // null pointer the kernel checks; the interpreter sees a `Value::Int(0)` (distinct from any
        // real array's `Value::Ptr` by variant) and marshals it as absent → scale-1 / shift-0.
        let ptr_or_null = |me: &mut Self, sym: Option<Symbol>| -> Option<ValueId> {
            match sym {
                Some(s) => me.lookup(s).map(|(v, _)| v),
                None => Some(
                    me.builder
                        .build(MirType::I64, Op::ConstInt(0, MirType::I64)),
                ),
            }
        };
        let (Some(gptr), Some(bptr)) = (ptr_or_null(self, gamma), ptr_or_null(self, beta)) else {
            return false;
        };
        let func = if batched_parallel {
            self.gemm.norm_affine_par
        } else {
            self.gemm.norm_affine
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![xv, xv, gptr, bptr, rows, nval, epsv, opv],
        });
        true
    }

    // ---- LayerNorm / RMSNorm recognition (the per-token transformer normalizations) ----

    /// Is `e` a float literal (or its negation)? → its `f32` bit pattern (the kernel's `eps` ABI). Pure.
    fn float_lit_bits(&self, e: &Expr) -> Option<i64> {
        let v: f64 = match &e.kind {
            ExprKind::Float(t) => parse_float(self.interner.resolve(*t)),
            ExprKind::Unary {
                op: ast::UnOp::Neg,
                expr,
            } => match &expr.kind {
                ExprKind::Float(t) => -parse_float(self.interner.resolve(*t)),
                _ => return None,
            },
            _ => return None,
        };
        Some((v as f32).to_bits() as i64)
    }

    /// Does `d` denote the row length `n` as an `f32` divisor — a float literal equal to a literal
    /// count, an `(n as f32)` cast of the exact bound, or the bound expression itself? (The mean /
    /// mean-square divides the row sum by the element count; this pins that divisor to the loop trip
    /// count, so we only fuse a genuine per-row mean.) Pure.
    fn count_as_f32(&self, d: &Expr, n: &Expr) -> bool {
        if let (ExprKind::Float(f), ExprKind::Int(k)) = (&d.kind, &n.kind) {
            return parse_float(self.interner.resolve(*f))
                == parse_int(self.interner.resolve(*k)) as f64;
        }
        if let ExprKind::Cast { expr, .. } = &d.kind {
            return self.expr_mir(d) == MirType::F32 && exprs_struct_eq(expr, n);
        }
        exprs_struct_eq(d, n)
    }

    /// Is `f` the float literal `1/count` (the multiply-by-reciprocal mean, e.g. `* 0.0625` for 16)?
    /// This is exactly what the kernel computes (`* invn`), so it is the *most* faithful spelling. Pure.
    fn recip_of_count(&self, f: &Expr, n: &Expr) -> bool {
        if let (ExprKind::Float(t), ExprKind::Int(k)) = (&f.kind, &n.kind) {
            let cnt = parse_int(self.interner.resolve(*k)) as f64;
            return cnt != 0.0 && parse_float(self.interner.resolve(*t)) == 1.0 / cnt;
        }
        false
    }

    /// Does `e` scale the row sum `s` by `1/count` — `s / count` or `s * (1/count)` (either factor
    /// order)? Both the mean and the mean-square divide a row sum by the element count; this accepts
    /// the divide and the (kernel-faithful) multiply-by-reciprocal spellings. Pure.
    fn scaled_by_inv_count(&self, e: &Expr, s: Symbol, n: &Expr) -> bool {
        match &e.kind {
            ExprKind::Binary {
                op: ast::BinOp::Div,
                lhs,
                rhs,
            } => single_path(lhs) == Some(s) && self.count_as_f32(rhs, n),
            ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } => {
                (single_path(lhs) == Some(s) && self.recip_of_count(rhs, n))
                    || (single_path(rhs) == Some(s) && self.recip_of_count(lhs, n))
            }
            _ => false,
        }
    }

    /// `let name = s / count` or `s * (1/count)` (row sum → mean), the scale pinned to the trip count
    /// `n`. Pure.
    fn match_mean(&self, stmt: &Stmt, s: Symbol, n: &Expr) -> Option<Symbol> {
        let (name, init) = Self::let_init(stmt)?;
        if self.scaled_by_inv_count(init, s, n) {
            Some(name)
        } else {
            None
        }
    }

    /// Is `e` the centered value `x[v] - mean`? Pure.
    fn is_centered(
        &self,
        e: &Expr,
        v: Symbol,
        x: Symbol,
        mean: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> bool {
        matches!(
            &e.kind,
            ExprKind::Binary { op: ast::BinOp::Sub, lhs, rhs }
                if self.index_off(lhs, v, batch) == Some(x) && single_path(rhs) == Some(mean)
        )
    }

    /// Body `acc += (x[v]-mean)*(x[v]-mean)` (sum of squared deviations). Pure.
    fn match_var_body(
        &self,
        body: &Block,
        v: Symbol,
        x: Symbol,
        mean: Symbol,
        acc: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<()> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign { target, op, value } = &stmt.kind else {
            return None;
        };
        if single_path(target) != Some(acc) {
            return None;
        }
        let addend: &Expr = match op {
            ast::AssignOp::Add => value,
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(lhs) == Some(acc) => rhs,
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(rhs) == Some(acc) => lhs,
                _ => return None,
            },
            _ => return None,
        };
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &addend.kind
        else {
            return None;
        };
        if self.is_centered(lhs, v, x, mean, batch) && self.is_centered(rhs, v, x, mean, batch) {
            Some(())
        } else {
            None
        }
    }

    /// Body `x[v] = (x[v]-mean) * inv` (center then scale, in place; either operand order). Pure.
    fn match_shift_scale_body(
        &self,
        body: &Block,
        v: Symbol,
        x: Symbol,
        mean: Symbol,
        inv: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<(Option<Symbol>, Option<Symbol>)> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if self.index_off(target, v, batch) != Some(x) {
            return None;
        }
        let (core, gamma, beta) = self.peel_affine(value, v, x);
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &core.kind
        else {
            return None;
        };
        let ok = (self.is_centered(lhs, v, x, mean, batch) && single_path(rhs) == Some(inv))
            || (self.is_centered(rhs, v, x, mean, batch) && single_path(lhs) == Some(inv));
        if ok {
            Some((gamma, beta))
        } else {
            None
        }
    }

    /// The argument `X` of `1.0 / sqrt(X)` or `rsqrt(X)`. Pure.
    fn as_rsqrt_arg<'b>(&self, e: &'b Expr) -> Option<&'b Expr> {
        match &e.kind {
            ExprKind::Binary {
                op: ast::BinOp::Div,
                lhs,
                rhs,
            } if matches!(&lhs.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 1.0) =>
            {
                let ExprKind::Call { callee, args, .. } = &rhs.kind else {
                    return None;
                };
                if args.len() == 1
                    && matches!(
                        self.vectorizable_intrinsic(callee),
                        Some(MathIntrinsic::Sqrt)
                    )
                {
                    Some(&args[0])
                } else {
                    None
                }
            }
            ExprKind::Call { callee, args, .. }
                if args.len() == 1
                    && matches!(
                        self.vectorizable_intrinsic(callee),
                        Some(MathIntrinsic::Rsqrt)
                    ) =>
            {
                Some(&args[0])
            }
            _ => None,
        }
    }

    /// `let inv = 1.0 / sqrt(sum/count + eps)` (or `rsqrt(...)`, `eps` either side) → `(inv, eps_bits)`,
    /// the reciprocal-standard-deviation binding shared by LayerNorm (variance sum) and RMSNorm
    /// (mean-square sum). Pure.
    fn match_inv_rstd(&self, stmt: &Stmt, sum: Symbol, n: &Expr) -> Option<(Symbol, i64)> {
        let (name, init) = Self::let_init(stmt)?;
        let arg = self.as_rsqrt_arg(init)?;
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs,
            rhs,
        } = &arg.kind
        else {
            return None;
        };
        // One operand is the eps literal; the other is `sum / count`.
        let (ms, eps_bits) = if let Some(b) = self.float_lit_bits(rhs) {
            (lhs.as_ref(), b)
        } else if let Some(b) = self.float_lit_bits(lhs) {
            (rhs.as_ref(), b)
        } else {
            return None;
        };
        if self.scaled_by_inv_count(ms, sum, n) {
            Some((name, eps_bits))
        } else {
            None
        }
    }

    /// Recognize the canonical in-place flat LayerNorm window (7 statements) at `b.stmts[at..]`:
    ///
    /// ```text
    /// let mut s = 0.0;
    /// for i in 0..N { s += x[i]; }                          // row sum
    /// let mean = s / (N as f32);                            // mean
    /// let mut v = 0.0;
    /// for i in 0..N { v += (x[i]-mean)*(x[i]-mean); }       // variance sum
    /// let inv = 1.0 / sqrt(v / (N as f32) + eps);           // 1/sqrt(var+eps)
    /// for i in 0..N { x[i] = (x[i]-mean) * inv * g[i] + b[i]; }  // normalize (affine g/b optional)
    /// ```
    ///
    /// Same discipline as [`Self::match_softmax`]: every loop ranges the identical `0..N` over the same
    /// array `x`; the divisors are pinned to the trip count; the scalars chain; and the internal
    /// scalars must not be read after the window (the kernel hides them). The normalize step may carry
    /// an optional per-column affine `* gamma[i] (+ beta[i])` (the real transformer form) — those
    /// arrays are captured and returned. Returns `(consumed, array, N, eps_bits, gamma, beta)`; `None`
    /// on any deviation (the generic vectorizer then lowers the loops).
    fn match_layernorm(
        &self,
        b: &Block,
        at: usize,
        batch: Option<Symbol>,
    ) -> Option<(usize, Symbol, Expr, i64, Option<Symbol>, Option<Symbol>)> {
        let stmts = &b.stmts[at..];
        if stmts.len() < 7 {
            return None;
        }
        let (s, s_init) = Self::let_init(&stmts[0])?;
        if !self.is_zero_lit(s_init) {
            return None;
        }
        let (v1, n_expr, body1) = self.as_range0_for(&stmts[1])?;
        // Batched: the data is row-offset-indexed `row*cols + i` (cols == this inner bound); single-row:
        // just `i`. The mean/variance scalars and the `/cols` divisor are per-row, unchanged either way;
        // the affine gamma/beta (peeled in the scale body) stay column-indexed, shared across rows.
        let data_batch = batch.map(|row| (row, n_expr));
        let x = self.sum_body_array(body1, v1, s, data_batch)?;
        let mean = self.match_mean(&stmts[2], s, n_expr)?;
        let (vv, vv_init) = Self::let_init(&stmts[3])?;
        if !self.is_zero_lit(vv_init) {
            return None;
        }
        let (v4, n4, body4) = self.as_range0_for(&stmts[4])?;
        if !exprs_struct_eq(n4, n_expr) {
            return None;
        }
        self.match_var_body(body4, v4, x, mean, vv, data_batch)?;
        let (inv, eps_bits) = self.match_inv_rstd(&stmts[5], vv, n_expr)?;
        let (v6, n6, body6) = self.as_range0_for(&stmts[6])?;
        if !exprs_struct_eq(n6, n_expr) {
            return None;
        }
        let (gamma, beta) = self.match_shift_scale_body(body6, v6, x, mean, inv, data_batch)?;
        let rest = &b.stmts[at + 7..];
        let tail = b.tail.as_deref();
        for sc in [s, mean, vv, inv] {
            if block_mentions(rest, tail, sc) {
                return None;
            }
        }
        Some((7, x, n_expr.clone(), eps_bits, gamma, beta))
    }

    /// Recognize the canonical in-place flat RMSNorm window (4 statements) at `b.stmts[at..]`:
    ///
    /// ```text
    /// let mut s = 0.0;
    /// for i in 0..N { s += x[i]*x[i]; }              // mean-square sum
    /// let inv = 1.0 / sqrt(s / (N as f32) + eps);    // 1/sqrt(ms+eps)
    /// for i in 0..N { x[i] = x[i] * inv * g[i]; }     // normalize (affine scale g[i] optional)
    /// ```
    ///
    /// The normalize step may carry an optional per-column scale `* gamma[i]` (the real transformer
    /// form; RMSNorm has no shift, but a `+ beta[i]` is also accepted). Returns `(consumed, array, N,
    /// eps_bits, gamma, beta)`; `None` on any deviation.
    fn match_rmsnorm(
        &self,
        b: &Block,
        at: usize,
        batch: Option<Symbol>,
    ) -> Option<(usize, Symbol, Expr, i64, Option<Symbol>, Option<Symbol>)> {
        let stmts = &b.stmts[at..];
        if stmts.len() < 4 {
            return None;
        }
        let (s, s_init) = Self::let_init(&stmts[0])?;
        if !self.is_zero_lit(s_init) {
            return None;
        }
        let (v1, n_expr, body1) = self.as_range0_for(&stmts[1])?;
        // Batched: the data is indexed `row*cols + i` (cols == this inner bound); single-row: just `i`.
        let data_batch = batch.map(|row| (row, n_expr));
        let x = self.match_sumsq_body(body1, v1, s, data_batch)?;
        let (inv, eps_bits) = self.match_inv_rstd(&stmts[2], s, n_expr)?;
        let (v3, n3, body3) = self.as_range0_for(&stmts[3])?;
        if !exprs_struct_eq(n3, n_expr) {
            return None;
        }
        let (gamma, beta) = self.match_scale_body(body3, v3, x, inv, data_batch)?;
        let rest = &b.stmts[at + 4..];
        let tail = b.tail.as_deref();
        for sc in [s, inv] {
            if block_mentions(rest, tail, sc) {
                return None;
            }
        }
        Some((4, x, n_expr.clone(), eps_bits, gamma, beta))
    }

    /// `let inv = 1.0 / sqrt(sum + eps)` (or `rsqrt(...)`, `eps` either side, or the bare
    /// `1.0/sqrt(sum)` with implicit `eps = 0`) → `(inv, eps_bits)`. The L2-normalization
    /// reciprocal-norm binding: like [`Self::match_inv_rstd`] but the sqrt argument is the **raw**
    /// sum-of-squares — no `/N` mean divisor (the feature that distinguishes RMSNorm). The two are
    /// structurally disjoint (a `/N` node is present xor absent), so a window matches at most one. Pure.
    fn match_inv_l2norm(&self, stmt: &Stmt, sum: Symbol) -> Option<(Symbol, i64)> {
        let (name, init) = Self::let_init(stmt)?;
        let arg = self.as_rsqrt_arg(init)?;
        // Bare `sqrt(sum)` — no eps term: eps defaults to 0.0.
        if single_path(arg) == Some(sum) {
            return Some((name, 0.0f32.to_bits() as i64));
        }
        // `sqrt(sum + eps)` — eps is one operand; the other must be the bare sum-of-squares symbol.
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs,
            rhs,
        } = &arg.kind
        else {
            return None;
        };
        let (base, eps_bits) = if let Some(b) = self.float_lit_bits(rhs) {
            (lhs.as_ref(), b)
        } else if let Some(b) = self.float_lit_bits(lhs) {
            (rhs.as_ref(), b)
        } else {
            return None;
        };
        if single_path(base) == Some(sum) {
            Some((name, eps_bits))
        } else {
            None
        }
    }

    /// Recognize the in-place flat **L2-normalization** window (`out = x / ‖x‖₂`, the cosine-similarity
    /// / normalized-embedding / retrieval-key projection) at `b.stmts[at..]` — structurally RMSNorm
    /// minus the mean divisor:
    ///
    /// ```text
    /// let mut s = 0.0;
    /// for i in 0..N { s += x[i]*x[i]; }      // sum of squares = ‖x‖₂²
    /// let inv = 1.0 / sqrt(s + eps);         // 1/‖x‖₂   (no `/N` — that is RMSNorm)
    /// for i in 0..N { x[i] = x[i] * inv; }   // unit-normalize
    /// ```
    ///
    /// Returns `(consumed, x, n_expr, eps_bits, None, None)`. Reuses every RMSNorm helper
    /// (`match_sumsq_body`, `match_scale_body`); only the reciprocal binding differs
    /// (`match_inv_l2norm` vs `match_inv_rstd`) and the op code is [`NORM_L2NORM`]. An affine wrapper
    /// is **declined** (L2-normalize has no learned scale/shift — there is no affine L2 kernel), so a
    /// trailing `* gamma[i]` falls to the generic vectorizer rather than being silently dropped. Pure.
    fn match_l2norm(
        &self,
        b: &Block,
        at: usize,
        batch: Option<Symbol>,
    ) -> Option<(usize, Symbol, Expr, i64, Option<Symbol>, Option<Symbol>)> {
        let stmts = &b.stmts[at..];
        if stmts.len() < 4 {
            return None;
        }
        let (s, s_init) = Self::let_init(&stmts[0])?;
        if !self.is_zero_lit(s_init) {
            return None;
        }
        let (v1, n_expr, body1) = self.as_range0_for(&stmts[1])?;
        let data_batch = batch.map(|row| (row, n_expr));
        let x = self.match_sumsq_body(body1, v1, s, data_batch)?;
        let (inv, eps_bits) = self.match_inv_l2norm(&stmts[2], s)?;
        let (v3, n3, body3) = self.as_range0_for(&stmts[3])?;
        if !exprs_struct_eq(n3, n_expr) {
            return None;
        }
        let (gamma, beta) = self.match_scale_body(body3, v3, x, inv, data_batch)?;
        // L2-normalize is non-affine; a `* gamma[i]` (or `+ beta[i]`) window is not an L2 norm.
        if gamma.is_some() || beta.is_some() {
            return None;
        }
        let rest = &b.stmts[at + 4..];
        let tail = b.tail.as_deref();
        for sc in [s, inv] {
            if block_mentions(rest, tail, sc) {
                return None;
            }
        }
        Some((4, x, n_expr.clone(), eps_bits, None, None))
    }

    /// Is `e` the float literal `0.0`? Pure.
    fn is_zero_lit(&self, e: &Expr) -> bool {
        matches!(&e.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 0.0)
    }

    fn lower_stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Let { pat, ty, init, .. } => {
                let mty = match (ty, init) {
                    // An aggregate (struct/tuple) initializer — a literal OR a value (a call that
                    // returns a struct/tuple by value, or an aggregate variable) — sizes its slot
                    // from the init's registry-aware MIR type; the free annotation lowering
                    // (`mir_ty_of_ast`) falls back to `I32` for a named struct / tuple type. Array
                    // literals/repeats are excluded so they keep their existing annotation-preferred
                    // path (`mir_ty_of_ast` carries the explicit element type/length).
                    (_, Some(e))
                        if matches!(self.expr_mir(e), MirType::Array(..))
                            && !matches!(
                                &e.kind,
                                ExprKind::ArrayLit(_) | ExprKind::ArrayRepeat { .. }
                            ) =>
                    {
                        self.expr_mir(e)
                    }
                    // An array literal/repeat *with* an annotation prefers the annotation for its
                    // explicit element type + length (so an unsuffixed-literal element like
                    // `[1, 2]: [u8; 2]` takes the annotated scalar, not the i32 default). But
                    // `mir_ty_of_ast` is registry-blind: a struct/tuple element resolves to the
                    // `I32` fallback, under-allocating the slot and mis-striding `a[i]`, so
                    // `a[i].field` reads past the buffer / segfaults in native. When the init's
                    // registry-aware element is an aggregate byte buffer, splice it into the
                    // annotated array type (keeping the annotated length).
                    (Some(t), Some(e))
                        if matches!(
                            &e.kind,
                            ExprKind::ArrayLit(_) | ExprKind::ArrayRepeat { .. }
                        ) =>
                    {
                        match (mir_ty_of_ast(t, self.interner), self.expr_mir(e)) {
                            (MirType::Array(ae, n), MirType::Array(re, _))
                                if !matches!(*ae, MirType::Array(..))
                                    && matches!(*re, MirType::Array(..)) =>
                            {
                                MirType::Array(re, n)
                            }
                            (ann, _) => ann,
                        }
                    }
                    // A `let x: T;` with no initializer. Resolve `T` registry-aware so a no-init
                    // struct/tuple local allocates its real byte buffer — the registry-blind
                    // `mir_ty_of_ast` falls back to `i32`, under-allocating the slot so a later
                    // `p.f = …` GEPs off an `i32` and emits MIR the verifier/Cranelift reject (an ICE).
                    (Some(t), None) => self.mir_ty_of_ann(t),
                    (Some(t), _) => mir_ty_of_ast(t, self.interner),
                    (None, Some(e)) => self.expr_mir(e),
                    (None, None) => MirType::I32,
                };
                let slot = self.builder.alloca(mty.clone());
                if let Some(e) = init {
                    // A tuple/struct initializer fills the byte buffer field-by-field (the slot *is*
                    // the buffer, like an array). Detected by the literal shape so non-aggregate
                    // inits are unaffected.
                    if let ExprKind::TupleLit(items) = &e.kind {
                        let tty = self.expr_ty(e);
                        self.lower_tuple_init(slot, &tty, items);
                    } else if let ExprKind::StructLit { fields, .. } = &e.kind {
                        if let Ty::Named(sym) = self.expr_ty(e) {
                            self.lower_struct_init(slot, sym, fields, e.span);
                        }
                    } else if matches!(&e.kind, ExprKind::ArrayLit(_) | ExprKind::ArrayRepeat { .. })
                    {
                        if let MirType::Array(elem, n) = &mty {
                            self.lower_array_init(slot, elem, *n, e);
                        }
                    } else if matches!(&mty, MirType::Array(..)) {
                        // A non-literal aggregate value (a call returning a struct/tuple/array by
                        // value, or another aggregate variable): the expression yields a base
                        // pointer; deep-copy its leaves into the slot so the local owns its storage
                        // (value semantics) and both backends agree.
                        let src = self.lower_expr(e);
                        let ty = self.expr_ty(e);
                        self.emit_copy(slot, src, &ty);
                    } else {
                        let v = self.lower_expr(e);
                        self.builder.build_void(Op::Store {
                            ptr: slot,
                            value: v,
                        });
                    }
                } else {
                    // No initializer: zero-initialize the slot so a read-before-write yields a
                    // deterministic zero on every backend — matching the interpreter oracle's
                    // zero-initialized memory and mem2reg's read-before-write zero substitution.
                    // Native -O0 otherwise reads stack garbage, a three-way divergence (interp 0 /
                    // native-O0 garbage / native-O2 0 via mem2reg) that violates both hard
                    // invariants at once. Scalars, arrays, and struct/tuple byte buffers are covered.
                    self.zero_init(slot, &mty);
                }
                match &pat.kind {
                    ast::PatKind::Ident(name) => self.bind(*name, slot, mty),
                    // Destructuring `let (a, b) = …`: bind each sub-pattern to its tuple field's
                    // place within the slot (a scalar field reads via a `Load`, an aggregate field
                    // binds its pointer). Recurses for a nested tuple pattern.
                    ast::PatKind::Tuple(subs) => {
                        let tty = init
                            .as_ref()
                            .map(|e| self.expr_ty(e))
                            .unwrap_or(Ty::Unknown);
                        self.bind_tuple_pattern(slot, &tty, subs);
                    }
                    _ => {}
                }
            }
            StmtKind::Assign { target, op, value } => {
                // Whole-aggregate assignment — `*p = Pt{..}`, `s.f = other_struct`, `t.0 = (..)`,
                // `a[i] = some_struct`. The destination is a flat byte buffer (a `MIR Array`), and a
                // plain `Op::Store` of the RHS would store the RHS buffer's *base pointer* into the
                // destination's first slot rather than its contents (a silent miscompile on both
                // backends). Route it through `init_field`, which recurses a struct/tuple/array
                // *literal* directly into the destination pointer and deep-copies a non-literal
                // aggregate value leaf-by-leaf via `emit_copy` — the same GEP discipline that keeps
                // the interpreter's slot-indexed and native's byte-indexed memory in agreement.
                // (Compound assignment on an aggregate is not a valid program, so only `=`.) The
                // aggregate check is a pure type query (`expr_mir` emits no MIR), so the common
                // scalar path below keeps its original RHS-then-place evaluation order untouched.
                if matches!(op, ast::AssignOp::Assign)
                    && matches!(self.expr_mir(target), MirType::Array(..))
                {
                    let (ptr, _) = self.lower_place(target);
                    let dst_ty = self.expr_ty(target);
                    self.init_field(ptr, &dst_ty, value);
                    return;
                }
                let rhs0 = self.lower_expr(value);
                let rhs_ty = self.expr_mir(value);
                let (ptr, elem) = self.lower_place(target);
                // Coerce the value to the place's type before storing, so a narrowing store (e.g.
                // an `f32` value into a `[bf16; N]` slot) carries the element type — the backends
                // then store the right width and round to bf16. A no-op when the types match.
                let rhs = self.coerce_to(rhs0, &rhs_ty, &elem, self.signed(value));
                let store_val = match op {
                    ast::AssignOp::Assign => rhs,
                    _ => {
                        let cur = self
                            .builder
                            .build(elem.clone(), Op::Load(ptr, elem.clone()));
                        let bin = compound_binop(*op, elem.is_float(), self.signed(target));
                        self.builder.build(elem.clone(), Op::Bin(bin, cur, rhs))
                    }
                };
                self.builder.build_void(Op::Store {
                    ptr,
                    value: store_val,
                });
            }
            StmtKind::Expr(e) => {
                self.lower_expr_stmt(e);
            }
            StmtKind::Return(opt) => {
                if let Some((sret_ptr, rty)) = self.sret.clone() {
                    // Aggregate return (sret ABI): deep-copy the value into the caller-provided
                    // buffer — a struct/tuple *literal* recurses directly into it, a non-literal
                    // aggregate value is copied leaf-by-leaf — then return void.
                    if let Some(e) = opt {
                        self.init_field(sret_ptr, &rty, e);
                    }
                    self.builder.ret(None);
                } else {
                    let v = opt.as_ref().map(|e| {
                        let val = self.lower_expr(e);
                        self.coerce_return_value(val, e)
                    });
                    self.builder.ret(v);
                }
                self.terminated = true;
            }
            StmtKind::While {
                cond, body, label, ..
            } => self.lower_while(label.map(|l| l.sym), cond, body),
            StmtKind::For {
                pat,
                iter,
                body,
                label,
                ..
            } => self.lower_for(label.map(|l| l.sym), pat, iter, body),
            StmtKind::Loop { body, label, .. } => self.lower_loop(label.map(|l| l.sym), body),
            StmtKind::Break(label) => {
                if let Some((_, _, brk)) = self.find_loop(label.as_ref().map(|l| l.sym)) {
                    self.builder.br(brk, vec![]);
                }
                self.terminated = true;
            }
            StmtKind::Continue(label) => {
                if let Some((_, cont, _)) = self.find_loop(label.as_ref().map(|l| l.sym)) {
                    self.builder.br(cont, vec![]);
                }
                self.terminated = true;
            }
            StmtKind::Defer(e) => {
                // Defer semantics (run at scope exit) are not modeled yet; lower for effects.
                self.unsupported(e.span, "defer");
                self.lower_expr_stmt(e);
            }
        }
    }

    /// Lower an expression used in statement position (control-flow expressions handled here).
    fn lower_expr_stmt(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.lower_if(cond, then_branch, else_branch.as_deref());
            }
            ExprKind::Block(b) => {
                self.lower_block(b);
            }
            _ => {
                self.lower_expr(e);
            }
        }
    }

    /// Resolve a `break`/`continue` target. A labeled `'l` finds the nearest enclosing loop with
    /// that label; an unlabeled one is the innermost loop (the top of the stack). `None` if there is
    /// no such loop (sema rejects out-of-loop and unknown-label cases with `E0303` first, so this is
    /// only reachable in a malformed module — the branch is simply omitted).
    fn find_loop(
        &self,
        label: Option<Symbol>,
    ) -> Option<(Option<Symbol>, mercury_mir::BlockId, mercury_mir::BlockId)> {
        match label {
            Some(l) => self
                .loops
                .iter()
                .rev()
                .find(|(lbl, ..)| *lbl == Some(l))
                .copied(),
            None => self.loops.last().copied(),
        }
    }

    /// Coerce a `return`/tail value (lowered from `e`) to the function's declared return type, so the
    /// emitted `Ret`/branch arg is well-typed. Without this, `fn f() -> i64 { return 0; }` returns an
    /// `i32` literal from an `i64` function — MIR the native verifier and `mem2reg` reject while the
    /// interpreter silently runs it (and truncates a too-wide value, a silent wrong answer). Mirrors
    /// the coercion a `let`-annotation / argument / array-element already applies. A `Void` return
    /// type (a unit fn) is left alone — there is no scalar to coerce.
    fn coerce_return_value(&mut self, val: ValueId, e: &Expr) -> ValueId {
        let to = self.builder.ret_type().clone();
        if matches!(to, MirType::Void) {
            return val;
        }
        let from = self.expr_mir(e);
        self.coerce_to(val, &from, &to, self.signed(e))
    }

    fn lower_while(&mut self, label: Option<Symbol>, cond: &Expr, body: &Block) {
        let header = self.builder.new_block();
        let body_bb = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);

        self.builder.switch_to(header);
        self.terminated = false;
        let c = self.lower_bool_cond(cond);
        self.builder.cond_br(c, body_bb, vec![], exit, vec![]);

        self.builder.switch_to(body_bb);
        self.terminated = false;
        self.loops.push((label, header, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            self.builder.br(header, vec![]);
        }

        self.builder.switch_to(exit);
        self.terminated = false;
    }

    fn lower_loop(&mut self, label: Option<Symbol>, body: &Block) {
        let header = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);
        self.builder.switch_to(header);
        self.terminated = false;
        self.loops.push((label, header, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            self.builder.br(header, vec![]);
        }
        self.builder.switch_to(exit);
        self.terminated = false;
    }

    /// Emit a call to the GEMM microkernel for a recognized nest. Returns `false` (and emits
    /// nothing) if any operand array is not a pointer in scope, so the caller lowers it normally.
    /// The base pointer of a kernel operand `sym`. An array operand binds *directly* to its base
    /// pointer (a `MirType::Array` slot value), so it is used as-is; a tensor (or pointer) operand has
    /// MIR type `Ptr` and binds to a *slot* holding the pointer, so it must be loaded first. A no-op
    /// for every array operand (so existing matmuls are byte-identical) — it only adds the load that
    /// makes a shape-typed `Tensor[..]` operand reach the kernel as its actual base pointer.
    fn kernel_base_ptr(&mut self, sym: Symbol) -> Option<ValueId> {
        let (val, ty) = self.lookup(sym)?;
        Some(if matches!(ty, MirType::Ptr) {
            self.builder.build(MirType::Ptr, Op::Load(val, MirType::Ptr))
        } else {
            val
        })
    }

    fn emit_sgemm(&mut self, nest: &MatmulNest<'_>, parallel: bool) -> bool {
        let (Some(a), Some(b), Some(c)) = (
            self.kernel_base_ptr(nest.a),
            self.kernel_base_ptr(nest.b),
            self.kernel_base_ptr(nest.c),
        ) else {
            return false;
        };
        // `C = Aᵀ·B` (TN) has a kernel only for the plain 2-D form: a batched transposed-A nest would
        // need a per-batch transpose the kernel doesn't do, and `Aᵀ·Bᵀ` has no kernel at all. Decline
        // those to the scalar nest (bail before emitting any GEP). The common `dW = Aᵀ·B` is 2-D.
        if nest.transposed_a
            && (nest.transposed
                || !nest.a_off.is_empty()
                || !nest.b_off.is_empty()
                || !nest.c_off.is_empty())
        {
            return false;
        }
        // Apply any per-operand base offset (the batch/head index of a batched matmul) as a pointer
        // GEP; a plain 2-D matmul has empty offsets and passes the array base straight through. The
        // inner matmul is identical under a constant base shift, so both backends stay bit-exact.
        let a = self.offset_base(a, &nest.a_off);
        let b = self.offset_base(b, &nest.b_off);
        let c = self.offset_base(c, &nest.c_off);
        // Materialize each dimension as an i64 value — a constant for a literal dim, or a load of
        // the runtime dimension variable (a function param/local). Bails to the scalar nest if a
        // variable dim is somehow out of scope at the call site.
        let (Some(m), Some(k), Some(n)) = (
            self.dim_value(nest.m),
            self.dim_value(nest.k),
            self.dim_value(nest.n),
        ) else {
            return false;
        };
        let beta = self
            .builder
            .build(MirType::I64, Op::ConstInt(nest.beta as i128, MirType::I64));
        let func = match (parallel, nest.transposed_a, nest.transposed) {
            (false, false, false) => self.gemm.mm,
            (true, false, false) => self.gemm.mm_par,
            (false, false, true) => self.gemm.nt,
            (true, false, true) => self.gemm.nt_par,
            (false, true, false) => self.gemm.tn,
            (true, true, false) => self.gemm.tn_par,
            // `Aᵀ·Bᵀ` (both transposed) has no kernel; the guard above already declined it.
            (_, true, true) => return false,
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![a, b, c, m, k, n, beta],
        });
        true
    }

    /// Emit one `mercury_i8gemm_nt[_parallel](a, b, c, m, k, n)` call for a recognized int8 quantized
    /// `nn.Linear` (`C = A·Bᵀ`, `u8`×`i8`→`i32`). Bails (false) if an operand/dim is unbound at the
    /// call site, so the caller lowers the scalar nest. `parallel` selects the multicore kernel (rows
    /// are independent, so it is bit-identical to the serial one the interpreter runs).
    fn emit_i8gemm(&mut self, nest: &I8MatmulNest, parallel: bool) -> bool {
        let (Some((a, _)), Some((b, _)), Some((c, _))) = (
            self.lookup(nest.a),
            self.lookup(nest.b),
            self.lookup(nest.c),
        ) else {
            return false;
        };
        let (Some(m), Some(k), Some(n)) = (
            self.dim_value(nest.m),
            self.dim_value(nest.k),
            self.dim_value(nest.n),
        ) else {
            return false;
        };
        let func = if parallel {
            self.gemm.i8nt_par
        } else {
            self.gemm.i8nt
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![a, b, c, m, k, n],
        });
        true
    }

    /// Emit one `mercury_sgemm_{bf16,f16}_nt[_parallel](a, b, c, m, k, n, beta)` call for a recognized
    /// bf16/f16 mixed-precision `C = A·Bᵀ` (half inputs, f32 accumulate). `beta = 0` (the dot-product
    /// form overwrites C). Bails (false) if an operand/dim is unbound at the call site, so the caller
    /// lowers the scalar widening nest. `parallel` selects the multicore kernel (rows independent → it
    /// is bit-identical to the serial one the interpreter marshals). The widen is lossless, so the
    /// kernel equals the naive nest under the documented matmul reassociation.
    fn emit_lowp_gemm(&mut self, nest: &LowpMatmulNest, parallel: bool) -> bool {
        let (Some((a, _)), Some((b, _)), Some((c, _))) = (
            self.lookup(nest.a),
            self.lookup(nest.b),
            self.lookup(nest.c),
        ) else {
            return false;
        };
        let (Some(m), Some(k), Some(n)) = (
            self.dim_value(nest.m),
            self.dim_value(nest.k),
            self.dim_value(nest.n),
        ) else {
            return false;
        };
        let beta = self
            .builder
            .build(MirType::I64, Op::ConstInt(0, MirType::I64));
        // NT (`C = A·Bᵀ`, nn.Linear forward) vs TN (`C = Aᵀ·B`, the `dW = dYᵀ·X` weight gradient) —
        // the recognizer guarantees exactly one operand transposed, so `transposed_a` selects the kernel.
        let func = match (parallel, nest.f16, nest.transposed_a) {
            (false, false, false) => self.gemm.bf16_nt,
            (true, false, false) => self.gemm.bf16_nt_par,
            (false, true, false) => self.gemm.f16_nt,
            (true, true, false) => self.gemm.f16_nt_par,
            (false, false, true) => self.gemm.bf16_tn,
            (true, false, true) => self.gemm.bf16_tn_par,
            (false, true, true) => self.gemm.f16_tn,
            (true, true, true) => self.gemm.f16_tn_par,
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![a, b, c, m, k, n, beta],
        });
        true
    }

    /// Emit one `mercury_{max,avg}pool2d_f32[_parallel](x, out, channels, h, w, kh, kw, sh, sw)` call
    /// for a recognized 2D pooling nest. Bails (false) if an operand/dim is unbound (the caller then
    /// lowers the scalar nest). `parallel` selects the multicore kernel (channels across cores →
    /// bit-identical to the serial one, which the interpreter marshals; channels are independent, no
    /// cross-channel combine). Max is idempotent and the avg sum order is fixed, so the kernel equals
    /// the scalar nest bit-for-bit (no reassociation exception).
    fn emit_pool2d(&mut self, nest: &Pool2dNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((out, _))) = (self.lookup(nest.x), self.lookup(nest.out)) else {
            return false;
        };
        let (Some(channels), Some(h), Some(w)) = (
            self.dim_value(nest.channels),
            self.dim_value(nest.h),
            self.dim_value(nest.w),
        ) else {
            return false;
        };
        let (Some(kh), Some(kw), Some(sh), Some(sw)) = (
            self.dim_value(nest.kh),
            self.dim_value(nest.kw),
            self.dim_value(nest.sh),
            self.dim_value(nest.sw),
        ) else {
            return false;
        };
        let func = match (nest.op, parallel) {
            (POOL_MAX, false) => self.gemm.maxpool2d,
            (POOL_MAX, true) => self.gemm.maxpool2d_par,
            (_, false) => self.gemm.avgpool2d,
            (_, true) => self.gemm.avgpool2d_par,
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, out, channels, h, w, kh, kw, sh, sw],
        });
        true
    }

    /// Emit one `mercury_transpose_f32[_parallel](src, dst, rows, cols)` call for a recognized matrix
    /// transpose. Bails (false) if an operand/dim is unbound at the call site (the caller then lowers
    /// the scalar nest). `parallel` selects the multicore kernel (the row blocks write disjoint `dst`
    /// columns → bit-identical to the serial one, which is a plain permutation the interpreter marshals).
    fn emit_transpose(&mut self, nest: &TransposeNest, parallel: bool) -> bool {
        let (Some((src, _)), Some((dst, _))) = (self.lookup(nest.src), self.lookup(nest.dst)) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = match (parallel, nest.elem_u16) {
            (false, false) => self.gemm.transpose,
            (true, false) => self.gemm.transpose_par,
            (false, true) => self.gemm.transpose_u16,
            (true, true) => self.gemm.transpose_u16_par,
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![src, dst, rows, cols],
        });
        true
    }

    /// Emit one `mercury_colsum_f32[_parallel](x, out, rows, cols)` call for a recognized column
    /// reduction. Bails (false) if an operand/dim is unbound (the caller then lowers the scalar nest).
    /// `parallel` selects the multicore kernel (disjoint column stripes → bit-identical to the serial
    /// one, which sums each column in the same i-order the interpreter marshals).
    fn emit_colsum(&mut self, nest: &ColSumNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((out, _))) = (self.lookup(nest.x), self.lookup(nest.out)) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = match (nest.op, parallel) {
            (COL_MAX, false) => self.gemm.colmax,
            (COL_MAX, true) => self.gemm.colmax_par,
            (COL_MIN, false) => self.gemm.colmin,
            (COL_MIN, true) => self.gemm.colmin_par,
            (COL_MAXABS, false) => self.gemm.colmaxabs,
            (COL_MAXABS, true) => self.gemm.colmaxabs_par,
            (COL_MEAN, false) => self.gemm.colmean,
            (COL_MEAN, true) => self.gemm.colmean_par,
            (COL_SUMSQ, false) => self.gemm.colsumsq,
            (COL_SUMSQ, true) => self.gemm.colsumsq_par,
            (COL_L2, false) => self.gemm.coll2,
            (COL_L2, true) => self.gemm.coll2_par,
            (COL_RMS, false) => self.gemm.colrms,
            (COL_RMS, true) => self.gemm.colrms_par,
            (_, false) => self.gemm.colsum,
            (_, true) => self.gemm.colsum_par,
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, out, rows, cols],
        });
        true
    }

    /// Emit one `mercury_softmax_bwd_f32[_parallel](y, dy, dx, rows, cols)` call for a recognized batched
    /// softmax-backward. Bails (false) if an operand/dim is unbound (the caller lowers the scalar nest).
    /// `parallel` selects the multicore kernel (rows across cores → bit-identical to the serial one the
    /// interpreter marshals; rows are independent, no cross-row combine).
    fn emit_softmax_bwd(&mut self, nest: &SoftmaxBwdNest, parallel: bool) -> bool {
        let (Some((y, _)), Some((dy, _)), Some((dx, _))) =
            (self.lookup(nest.y), self.lookup(nest.dy), self.lookup(nest.dx))
        else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = if parallel {
            self.gemm.softmax_bwd_par
        } else {
            self.gemm.softmax_bwd
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![y, dy, dx, rows, cols],
        });
        true
    }

    /// Emit one `mercury_rmsnorm_bwd_f32[_parallel](x, dy, gamma, dx, rows, cols, eps_bits)` call for a
    /// recognized batched RMSNorm backward. Bails (false) if an operand/dim is unbound. `parallel`
    /// selects the multicore kernel (rows across cores → bit-identical to the serial one; rows
    /// independent, no cross-row combine).
    fn emit_rmsnorm_bwd(&mut self, nest: &RmsNormBwdNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((dy, _)), Some((gamma, _)), Some((dx, _))) = (
            self.lookup(nest.x),
            self.lookup(nest.dy),
            self.lookup(nest.gamma),
            self.lookup(nest.dx),
        ) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let eps = self.builder.build(
            MirType::I64,
            Op::ConstInt(nest.eps_bits as i128, MirType::I64),
        );
        let func = if parallel {
            self.gemm.rmsnorm_bwd_par
        } else {
            self.gemm.rmsnorm_bwd
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, dy, gamma, dx, rows, cols, eps],
        });
        true
    }

    /// Emit one `mercury_layernorm_bwd_f32[_parallel](x, dy, gamma, dx, rows, cols, eps_bits)` call for
    /// a recognized batched LayerNorm backward. Same shape as `emit_rmsnorm_bwd`.
    fn emit_layernorm_bwd(&mut self, nest: &LayerNormBwdNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((dy, _)), Some((gamma, _)), Some((dx, _))) = (
            self.lookup(nest.x),
            self.lookup(nest.dy),
            self.lookup(nest.gamma),
            self.lookup(nest.dx),
        ) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let eps = self.builder.build(
            MirType::I64,
            Op::ConstInt(nest.eps_bits as i128, MirType::I64),
        );
        let func = if parallel {
            self.gemm.layernorm_bwd_par
        } else {
            self.gemm.layernorm_bwd
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, dy, gamma, dx, rows, cols, eps],
        });
        true
    }

    /// Emit one `mercury_xent_fwd_f32[_parallel](x, target, loss, rows, cols)` call for a recognized
    /// batched cross-entropy loss. Bails (false) if an operand/dim is unbound. `parallel` selects the
    /// multicore kernel (rows independent → bit-identical to serial).
    fn emit_xent(&mut self, nest: &XentNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((target, _)), Some((loss, _))) = (
            self.lookup(nest.x),
            self.lookup(nest.target),
            self.lookup(nest.loss),
        ) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = if parallel {
            self.gemm.xent_par
        } else {
            self.gemm.xent
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, target, loss, rows, cols],
        });
        true
    }

    /// Emit one `mercury_xent_bwd_f32[_parallel](x, target, dx, rows, cols)` call for a recognized
    /// cross-entropy backward nest. Bails (false) if an operand/dim is unbound.
    fn emit_xent_bwd(&mut self, nest: &XentBwdNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((target, _)), Some((dx, _))) = (
            self.lookup(nest.x),
            self.lookup(nest.target),
            self.lookup(nest.dx),
        ) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = if parallel {
            self.gemm.xent_bwd_par
        } else {
            self.gemm.xent_bwd
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, target, dx, rows, cols],
        });
        true
    }

    /// Emit one `mercury_rope_f32[_parallel](x, inv_freq, out, rows, half)` call for a recognized RoPE
    /// nest. Bails (false) if an operand/dim is unbound. `parallel` selects the multicore kernel (rows
    /// independent → bit-identical to serial).
    fn emit_rope(&mut self, nest: &RopeNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((inv_freq, _)), Some((out, _))) = (
            self.lookup(nest.x),
            self.lookup(nest.inv_freq),
            self.lookup(nest.out),
        ) else {
            return false;
        };
        let (Some(rows), Some(half)) = (self.dim_value(nest.rows), self.dim_value(nest.half)) else {
            return false;
        };
        let func = match (nest.backward, parallel) {
            (false, false) => self.gemm.rope,
            (false, true) => self.gemm.rope_par,
            (true, false) => self.gemm.rope_bwd,
            (true, true) => self.gemm.rope_bwd_par,
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, inv_freq, out, rows, half],
        });
        true
    }

    /// Emit one `mercury_logsumexp_f32[_parallel](x, out, rows, cols)` call for a recognized log-sum-exp
    /// nest. Bails (false) if an operand/dim is unbound. `parallel` selects the multicore kernel.
    fn emit_logsumexp(&mut self, nest: &LogsumexpNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((out, _))) = (self.lookup(nest.x), self.lookup(nest.out)) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = if parallel {
            self.gemm.logsumexp_par
        } else {
            self.gemm.logsumexp
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, out, rows, cols],
        });
        true
    }

    /// Emit `mercury_colarg{max,min}_i32[_parallel](x, out, rows, cols)` for a recognized per-column arg
    /// nest. Same `(ptr,ptr,i64,i64)` i32-output ABI as `emit_rowarg`; `out` is the per-column row-index.
    fn emit_colarg(&mut self, nest: &ColArgNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((out, _))) = (self.lookup(nest.x), self.lookup(nest.out)) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = match (nest.is_max, parallel) {
            (true, false) => self.gemm.colargmax,
            (true, true) => self.gemm.colargmax_par,
            (false, false) => self.gemm.colargmin,
            (false, true) => self.gemm.colargmin_par,
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, out, rows, cols],
        });
        true
    }

    /// Emit `mercury_rowarg{max,min}_i32[_parallel](x, out, rows, cols)` for a recognized per-row arg nest.
    /// Same `(ptr,ptr,i64,i64)` shape as `mercury_logsumexp_f32`, but `out` is an i32 index buffer.
    fn emit_rowarg(&mut self, nest: &RowArgNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((out, _))) = (self.lookup(nest.x), self.lookup(nest.out)) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = match (nest.is_max, parallel) {
            (true, false) => self.gemm.rowargmax,
            (true, true) => self.gemm.rowargmax_par,
            (false, false) => self.gemm.rowargmin,
            (false, true) => self.gemm.rowargmin_par,
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, out, rows, cols],
        });
        true
    }

    /// Emit `mercury_cumsum_f32[_parallel](x, out, rows, cols)` for a recognized per-row prefix sum.
    /// Same `(ptr,ptr,i64,i64)` `sig_vmath` shape; the `_parallel` one maps rows across cores.
    fn emit_cumsum(&mut self, nest: &CumsumNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((out, _))) = (self.lookup(nest.x), self.lookup(nest.out)) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = if parallel {
            self.gemm.cumsum_par
        } else {
            self.gemm.cumsum
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, out, rows, cols],
        });
        true
    }

    /// Emit `mercury_cumprod_f32[_parallel](x, out, rows, cols)` for a recognized prefix product. Same
    /// 2-ptr + 2-i64 `sig_vmath` ABI as cumsum; the `_parallel` one maps independent rows across cores.
    fn emit_cumprod(&mut self, nest: &CumsumNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((out, _))) = (self.lookup(nest.x), self.lookup(nest.out)) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = if parallel {
            self.gemm.cumprod_par
        } else {
            self.gemm.cumprod
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, out, rows, cols],
        });
        true
    }

    /// Emit `mercury_lrscan_f32[_parallel](a, b, out, rows, cols)` for a recognized linear-recurrence scan.
    /// 3 pointers + 2 i64 (the `softmax_bwd` ABI); the `_parallel` one maps the independent rows across
    /// cores (bit-identical to serial — no cross-row combine).
    fn emit_lrscan(&mut self, nest: &LrscanNest, parallel: bool) -> bool {
        let (Some((a, _)), Some((b, _)), Some((out, _))) =
            (self.lookup(nest.a), self.lookup(nest.b), self.lookup(nest.out))
        else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = if parallel {
            self.gemm.lrscan_par
        } else {
            self.gemm.lrscan
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![a, b, out, rows, cols],
        });
        true
    }

    /// Emit `mercury_cum{max,min}_f32[_parallel](x, out, rows, cols)` for a recognized cumulative max/min.
    fn emit_cumminmax(&mut self, nest: &CumMinMaxNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((out, _))) = (self.lookup(nest.x), self.lookup(nest.out)) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = match (nest.is_max, parallel) {
            (true, false) => self.gemm.cummax,
            (true, true) => self.gemm.cummax_par,
            (false, false) => self.gemm.cummin,
            (false, true) => self.gemm.cummin_par,
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, out, rows, cols],
        });
        true
    }

    /// Emit `mercury_kldiv_f32[_parallel](p, q, out, rows, cols)` for a recognized KL-divergence nest.
    fn emit_kldiv(&mut self, nest: &KldivNest, parallel: bool) -> bool {
        let (Some((p, _)), Some((q, _)), Some((out, _))) =
            (self.lookup(nest.p), self.lookup(nest.q), self.lookup(nest.out))
        else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = if parallel {
            self.gemm.kldiv_par
        } else {
            self.gemm.kldiv
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![p, q, out, rows, cols],
        });
        true
    }

    /// Emit `mercury_entropy_f32[_parallel](p, out, rows, cols)` for a recognized row-entropy nest.
    fn emit_entropy(&mut self, nest: &EntropyNest, parallel: bool) -> bool {
        let (Some((p, _)), Some((out, _))) = (self.lookup(nest.p), self.lookup(nest.out)) else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = if parallel {
            self.gemm.entropy_par
        } else {
            self.gemm.entropy
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![p, out, rows, cols],
        });
        true
    }

    /// Emit `mercury_kd_loss_f32[_parallel](x, q, out, rows, cols)` for a recognized soft-label xent nest.
    fn emit_kd_loss(&mut self, nest: &KdLossNest, parallel: bool) -> bool {
        let (Some((x, _)), Some((q, _)), Some((out, _))) =
            (self.lookup(nest.x), self.lookup(nest.q), self.lookup(nest.out))
        else {
            return false;
        };
        let (Some(rows), Some(cols)) = (self.dim_value(nest.rows), self.dim_value(nest.cols)) else {
            return false;
        };
        let func = if parallel {
            self.gemm.kd_loss_par
        } else {
            self.gemm.kd_loss
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![x, q, out, rows, cols],
        });
        true
    }

    /// Emit the fused `mercury_sgemm_{bf16,f16}_nt_epi[_parallel](a, b, c, m, k, n, beta, bias, act)`
    /// call for a recognized bf16/f16 `nn.Linear`+epilogue (`C = act(A·Bᵀ + bias)`, half inputs / f32
    /// output). Bails (false) if any operand/dim is unbound at the call site, so the caller lowers the
    /// matmul and the epilogue loop separately. `beta = 0` (the dot-product form overwrites C). The
    /// kernel widens A/B losslessly and runs the f32 `nt_epi` epilogue, so fused == the f32 fused FFN on
    /// the widened operands; in a `@parallel` function the multicore kernel runs (each C tile owned by
    /// one task → bit-identical to the serial kernel the interpreter marshals).
    fn emit_lowp_gemm_epi(&mut self, nest: &LowpMatmulNest, bias: Option<Symbol>, act: u32) -> bool {
        let (Some((a, _)), Some((b, _)), Some((c, _))) = (
            self.lookup(nest.a),
            self.lookup(nest.b),
            self.lookup(nest.c),
        ) else {
            return false;
        };
        let (Some(m), Some(k), Some(n)) = (
            self.dim_value(nest.m),
            self.dim_value(nest.k),
            self.dim_value(nest.n),
        ) else {
            return false;
        };
        // An absent bias is a null pointer, built as an integer `0` (a `Ptr`-typed `ConstInt` is invalid
        // MIR): the kernel checks `bias.is_null()`, and the interpreter distinguishes the `Value::Int(0)`
        // from a real array's `Value::Ptr` by variant — same convention as the f32 `nt_epi` epilogue.
        let bias_ptr = match bias {
            Some(s) => match self.lookup(s) {
                Some((v, _)) => v,
                None => return false,
            },
            None => self
                .builder
                .build(MirType::I64, Op::ConstInt(0, MirType::I64)),
        };
        let beta = self
            .builder
            .build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let act_v = self
            .builder
            .build(MirType::I64, Op::ConstInt(act as i128, MirType::I64));
        let func = match (self.parallel_fn, nest.f16) {
            (false, false) => self.gemm.bf16_nt_epi,
            (true, false) => self.gemm.bf16_nt_epi_par,
            (false, true) => self.gemm.f16_nt_epi,
            (true, true) => self.gemm.f16_nt_epi_par,
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![a, b, c, m, k, n, beta, bias_ptr, act_v],
        });
        true
    }

    /// Fuse a bf16/f16 `nn.Linear` matmul immediately followed by its bias-add / activation epilogue
    /// into one `mercury_sgemm_{bf16,f16}_nt_epi` call (`C = act(A·Bᵀ + bias)`, half inputs / f32
    /// output) — the mixed-precision transformer FFN projection. The half twin of
    /// [`Self::try_fuse_matmul_epilogue`]: `match_matmul_lowp` recognizes the half matmul, then the
    /// *identical* strict epilogue match (`match_bias_act_epilogue`, reused via the `(m, n, c)` shape)
    /// and the same `EPI_ACT_*` codes. Returns the statements consumed (always 2), else `None`. The
    /// kernel folds bias + activation into the widened-f32 GEMM writeback, so the fused result equals
    /// the unfused half `matmul → [bias →] activation`, bit-for-bit across backends. **NT only:** the
    /// fused-epilogue kernel exists only for `C = A·Bᵀ`, so a transposed-A (TN) nest declines here and
    /// lowers as a plain TN GEMM + a separate epilogue loop (no half TN epilogue kernel).
    fn try_fuse_lowp_matmul_epilogue(&mut self, stmts: &[Stmt]) -> Option<usize> {
        if stmts.len() < 2 {
            return None;
        }
        let StmtKind::For {
            pat, iter, body, ..
        } = &stmts[0].kind
        else {
            return None;
        };
        let nest = match_matmul_lowp(pat, iter, body, self.sema, self.interner)?;
        // The fused epilogue kernel is NT-only (`mercury_sgemm_{bf16,f16}_nt_epi`); a TN weight-gradient
        // nest has no epilogue kernel, so decline (it lowers as a plain TN GEMM + a separate epilogue).
        if nest.transposed_a {
            return None;
        }
        let (bias, act) =
            match_bias_act_epilogue(&stmts[1], nest.m, nest.n, nest.c, self.interner)?;
        if self.emit_lowp_gemm_epi(&nest, bias, act) {
            Some(2)
        } else {
            None
        }
    }

    /// Fuse a `nn.Linear` matmul immediately followed by its bias-add / activation epilogue into one
    /// `mercury_sgemm_nt_epi` call (`C = act(A·Bᵀ + bias)`), folding the epilogue into the GEMM's C
    /// writeback so C is written once instead of paying a separate read-modify-write pass. Fires only
    /// for the plain 2-D `C = A·Bᵀ` form (no batch offsets) immediately followed by the recognized
    /// epilogue loop over the same C; returns the number of statements consumed (always 2), else
    /// `None`. The match is strict (dims/strides/output/column all verified) so it never misfires.
    fn try_fuse_matmul_epilogue(&mut self, stmts: &[Stmt]) -> Option<usize> {
        if stmts.len() < 2 {
            return None;
        }
        let StmtKind::For {
            pat, iter, body, ..
        } = &stmts[0].kind
        else {
            return None;
        };
        let nest = recognize_matmul(pat, iter, body, self.sema, self.interner)?;
        // Only the plain 2-D nn.Linear form `C = A·Bᵀ` — no batch/head offsets (the epilogue kernel
        // is serial nt-only).
        if !nest.transposed
            || !nest.a_off.is_empty()
            || !nest.b_off.is_empty()
            || !nest.c_off.is_empty()
        {
            return None;
        }
        let (bias, act) =
            match_bias_act_epilogue(&stmts[1], nest.m, nest.n, nest.c, self.interner)?;
        if self.emit_sgemm_epi(&nest, bias, act) {
            Some(2)
        } else {
            None
        }
    }

    /// Emit the fused `mercury_sgemm_nt_epi(a, b, c, m, k, n, beta, bias, act)` call for a recognized
    /// Linear+epilogue. Bails (false) if any operand/dim is somehow unbound at the call site, so the
    /// caller falls back to lowering the matmul and the epilogue loop separately.
    fn emit_sgemm_epi(&mut self, nest: &MatmulNest<'_>, bias: Option<Symbol>, act: u32) -> bool {
        let (Some((a, _)), Some((b, _)), Some((c, _)), Some(m), Some(k), Some(n)) = (
            self.lookup(nest.a),
            self.lookup(nest.b),
            self.lookup(nest.c),
            self.dim_value(nest.m),
            self.dim_value(nest.k),
            self.dim_value(nest.n),
        ) else {
            return false;
        };
        // An absent bias is a null pointer, built as an integer `0` (a `Ptr`-typed `ConstInt` is
        // invalid MIR): the kernel checks `bias.is_null()`, and the interpreter distinguishes the
        // `Value::Int(0)` from a real array's `Value::Ptr` by variant — same convention as the affine
        // norm null params.
        let bias_ptr = match bias {
            Some(s) => match self.lookup(s) {
                Some((v, _)) => v,
                None => return false,
            },
            None => self
                .builder
                .build(MirType::I64, Op::ConstInt(0, MirType::I64)),
        };
        let beta = self
            .builder
            .build(MirType::I64, Op::ConstInt(nest.beta as i128, MirType::I64));
        let act_v = self
            .builder
            .build(MirType::I64, Op::ConstInt(act as i128, MirType::I64));
        // In a `@parallel` function, dispatch the fused FFN across cores (bit-identical to the serial
        // kernel, which the interpreter calls as the oracle). Before, the epilogue forced serial — so a
        // `@parallel` fused FFN could use the fusion or the cores, but not both.
        let func = if self.parallel_fn {
            self.gemm.nt_epi_par
        } else {
            self.gemm.nt_epi
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![a, b, c, m, k, n, beta, bias_ptr, act_v],
        });
        true
    }

    /// Fuse a recognized int8 GEMM immediately followed by its per-channel dequant epilogue into one
    /// `mercury_i8gemm_nt_deq` call (`out = act((A·Bᵀ as f32)·scale_a·scale_b [+ bias])`). The i32
    /// accumulator never reaches memory — the kernel dequants each output tile in registers straight
    /// to the f32 output, the fusion cuBLAS/oneDNN can't express (they emit the i32 GEMM and the
    /// dequant as two passes over a full i32 buffer). Fires only when the i32 accumulator `c` is a
    /// `let`-local of this block that is **dead** after the dequant loop, so dropping its separate
    /// materialization is sound. Returns the number of statements consumed (always 2), else `None`.
    fn try_fuse_i8matmul_dequant_epilogue(&mut self, b: &Block, i: usize) -> Option<usize> {
        if i + 1 >= b.stmts.len() {
            return None;
        }
        // stmts[i]: the int8 `C = A·Bᵀ` nest, writing the i32 accumulator `c`.
        let StmtKind::For {
            pat, iter, body, ..
        } = &b.stmts[i].kind
        else {
            return None;
        };
        let nest = match_matmul_i8_nt(pat, iter, body, self.sema, self.interner)?;
        // stmts[i+1]: the dequant loop over the same `c`, writing the f32 output.
        let (out, scale_a, scale_b, bias, act) =
            match_i8_dequant_epilogue(&b.stmts[i + 1], &nest, self.sema, self.interner)?;
        // Soundness: the fused kernel never writes `c` (it dequants in registers), so `c`'s separate
        // materialization may be dropped only if `c` is provably dead afterward. Require `c` to be a
        // `let`-local declared earlier in *this* block (lexical scoping then forbids it escaping to an
        // outer scope) and unmentioned in every statement after the dequant loop (and the block tail).
        if !block_declares_local(&b.stmts[..i], nest.c)
            || block_mentions(&b.stmts[i + 2..], b.tail.as_deref(), nest.c)
        {
            return None;
        }
        if self.emit_i8gemm_deq(&nest, out, scale_a, scale_b, bias, act) {
            Some(2)
        } else {
            None
        }
    }

    /// Emit the fused `mercury_i8gemm_nt_deq[_parallel](a, b, out, m, k, n, scale_a, scale_b, bias,
    /// act)` call for a recognized int8 GEMM + dequant. `scale_a` is the per-tensor activation scale (a
    /// scalar f32, loaded from its slot; `None` ⇒ the scale was folded into `scale_b`, so pass `1.0`).
    /// A null bias is the integer `0` (a `Ptr`-typed const is invalid MIR — same convention as the
    /// affine norm / fused-epilogue null params; the kernel checks `bias.is_null()` and the interpreter
    /// distinguishes `Value::Int(0)` from a real array's `Value::Ptr` by variant). Bails (false) if an
    /// operand/dim is unbound at the call site, so the caller lowers the GEMM and the dequant loop
    /// separately (still correct, just unfused). `@parallel` selects the multicore kernel (rows
    /// independent → bit-identical to the serial kernel the interpreter marshals).
    fn emit_i8gemm_deq(
        &mut self,
        nest: &I8MatmulNest,
        out: Symbol,
        scale_a: Option<Symbol>,
        scale_b: Symbol,
        bias: Option<Symbol>,
        act: u32,
    ) -> bool {
        let (Some((a, _)), Some((b, _)), Some((out_v, _)), Some((sb, _))) = (
            self.lookup(nest.a),
            self.lookup(nest.b),
            self.lookup(out),
            self.lookup(scale_b),
        ) else {
            return false;
        };
        let (Some(m), Some(k), Some(n)) = (
            self.dim_value(nest.m),
            self.dim_value(nest.k),
            self.dim_value(nest.n),
        ) else {
            return false;
        };
        // scale_a: load the scalar (the kernel takes it by value as f32), or the literal 1.0 when the
        // activation scale was folded into the per-channel scale_b.
        let scale_a_v = match scale_a {
            Some(s) => match self.lookup(s) {
                Some((slot, ty)) => self.builder.build(ty.clone(), Op::Load(slot, ty)),
                None => return false,
            },
            None => self
                .builder
                .build(MirType::F32, Op::ConstFloat(1.0, MirType::F32)),
        };
        let bias_ptr = match bias {
            Some(s) => match self.lookup(s) {
                Some((v, _)) => v,
                None => return false,
            },
            None => self
                .builder
                .build(MirType::I64, Op::ConstInt(0, MirType::I64)),
        };
        let act_v = self
            .builder
            .build(MirType::I64, Op::ConstInt(act as i128, MirType::I64));
        let func = if self.parallel_fn {
            self.gemm.i8deq_par
        } else {
            self.gemm.i8deq
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![a, b, out_v, m, k, n, scale_a_v, sb, bias_ptr, act_v],
        });
        true
    }

    /// GEP `base` by the sum of the `offset` terms (element indices) — the batch/head base shift of
    /// a batched matmul. Returns `base` unchanged when there is no offset (a plain 2-D matmul). Each
    /// term is lowered in the current scope (the enclosing batch-loop variable and the dimensions are
    /// all live) and coerced to `i64` before summing.
    fn offset_base(&mut self, base: ValueId, offset: &[&Expr]) -> ValueId {
        if offset.is_empty() {
            return base;
        }
        let mut acc: Option<ValueId> = None;
        for &t in offset {
            let ty = self.expr_mir(t);
            let v = self.lower_expr(t);
            let v = self.coerce_to(v, &ty, &MirType::I64, true);
            acc = Some(match acc {
                None => v,
                Some(prev) => self
                    .builder
                    .build(MirType::I64, Op::Bin(BinOp::Add, prev, v)),
            });
        }
        let idx = acc.unwrap();
        self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: base,
                index: idx,
                elem: MirType::F32,
            },
        )
    }

    /// Materialize a matmul dimension as an `i64` MIR value: a constant for a literal, or a load
    /// (coerced to i64) of the runtime dimension variable.
    fn dim_value(&mut self, dim: Dim) -> Option<ValueId> {
        match dim {
            Dim::Lit(v) => Some(
                self.builder
                    .build(MirType::I64, Op::ConstInt(v as i128, MirType::I64)),
            ),
            Dim::Var(sym) => {
                let (slot, ty) = self.lookup(sym)?;
                let v = self.builder.build(ty.clone(), Op::Load(slot, ty.clone()));
                Some(self.coerce_to(v, &ty, &MirType::I64, true))
            }
        }
    }

    /// Recognize a kernel-dispatchable reduction body over the loop variable `k`: an additive fold
    /// `s = s + f(x[k], y[k])` (or `s += …`) — dot `x[k]*y[k]`, ssd `(x[k]-y[k])²`, sum `x[k]` — or a
    /// running max/min `m = fmax(m, x[k])` / `m = fmin(m, x[k])`. Returns the accumulator symbol, the
    /// `RED_*` op code, and the two array bases (`y == x` for the unary sum/max/min). Strict pure-AST
    /// match — single statement, index exactly `k`.
    fn match_reduction_kernel(
        &self,
        body: &Block,
        k: Symbol,
    ) -> Option<(Symbol, i64, Symbol, Symbol)> {
        if body.tail.is_some() || body.stmts.len() != 1 {
            return None;
        }
        let StmtKind::Assign { target, op, value } = &body.stmts[0].kind else {
            return None;
        };
        let s = single_path(target)?;
        if s == k {
            return None;
        }
        // `base[k]` with the index exactly the loop variable → the base array symbol.
        let idx_base = |e: &Expr| -> Option<Symbol> {
            let ExprKind::Index { base, indices } = &e.kind else {
                return None;
            };
            if indices.len() != 1 || single_path(&indices[0]) != Some(k) {
                return None;
            }
            single_path(base)
        };
        // Running max/min: `m = fmax(m, x[k])` / `m = fmin(m, x[k])` (either operand order). The other
        // operand must be `x[k]`; the kernel folds by `fmax`/`fmin`, the outer combine by `Cmp+Select`.
        if let ast::AssignOp::Assign = op {
            if let ExprKind::Call { callee, args, .. } = &value.kind {
                if args.len() == 2 {
                    let red = match self.vectorizable_intrinsic(callee) {
                        Some(MathIntrinsic::Fmax) => Some(RED_MAX),
                        Some(MathIntrinsic::Fmin) => Some(RED_MIN),
                        _ => None,
                    };
                    if let Some(red) = red {
                        let other = if single_path(&args[0]) == Some(s) {
                            &args[1]
                        } else if single_path(&args[1]) == Some(s) {
                            &args[0]
                        } else {
                            return None;
                        };
                        // `other` is `x[k]` (RED_MAX/MIN), or — for fmax — `abs(x[k])` (running
                        // absmax, the symmetric int8-quant scale). The kernel applies the abs.
                        if let Some(a) = idx_base(other) {
                            return Some((s, red, a, a));
                        }
                        if red == RED_MAX {
                            if let ExprKind::Call {
                                callee: ac,
                                args: aargs,
                                ..
                            } = &other.kind
                            {
                                if aargs.len() == 1
                                    && matches!(
                                        self.vectorizable_intrinsic(ac),
                                        Some(MathIntrinsic::Abs)
                                    )
                                {
                                    let a = idx_base(&aargs[0])?;
                                    return Some((s, RED_MAXABS, a, a));
                                }
                            }
                        }
                        return None;
                    }
                }
            }
        }
        // `s += addend`, or `s = s + addend` / `s = addend + s`.
        let addend: &Expr = match op {
            ast::AssignOp::Add => value,
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(lhs) == Some(s) => rhs,
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(rhs) == Some(s) => lhs,
                _ => return None,
            },
            _ => return None,
        };
        if expr_uses_sym(addend, s) {
            return None;
        }
        match &addend.kind {
            // dot `a[k]*b[k]` (a==b ≡ sum-of-squares), or ssd `(a[k]-b[k])*(a[k]-b[k])`.
            ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } => {
                if let (
                    ExprKind::Binary {
                        op: ast::BinOp::Sub,
                        lhs: l1,
                        rhs: r1,
                    },
                    ExprKind::Binary {
                        op: ast::BinOp::Sub,
                        lhs: l2,
                        rhs: r2,
                    },
                ) = (&lhs.kind, &rhs.kind)
                {
                    let (a1, b1, a2, b2) =
                        (idx_base(l1)?, idx_base(r1)?, idx_base(l2)?, idx_base(r2)?);
                    return if a1 == a2 && b1 == b2 {
                        Some((s, RED_SSD, a1, b1))
                    } else {
                        None
                    };
                }
                Some((s, RED_DOT, idx_base(lhs)?, idx_base(rhs)?))
            }
            // sum `a[k]`.
            ExprKind::Index { .. } => {
                let a = idx_base(addend)?;
                Some((s, RED_SUM, a, a))
            }
            _ => None,
        }
    }

    /// Recognize an argmax/argmin loop body `if x[k] CMP bv { bv = x[k]; bi = k; }` (strict `>` →
    /// argmax / `<` → argmin; `bi = k` may be `k as <int>`). Returns `(bv, bi, op, x)`. The inner `if`
    /// may be the body's single statement *or* its tail (a Unit-valued `if` with no else). Pure.
    fn match_argreduce_kernel(
        &self,
        body: &Block,
        k: Symbol,
    ) -> Option<(Symbol, Symbol, i64, Symbol)> {
        // The lone `if` may sit in `stmts` (followed by `;`) or be the block's tail expression.
        let if_expr = match (body.stmts.as_slice(), &body.tail) {
            ([only], None) => match &only.kind {
                StmtKind::Expr(e) => e,
                _ => return None,
            },
            ([], Some(e)) => e.as_ref(),
            _ => return None,
        };
        let ExprKind::If {
            cond,
            then_branch,
            else_branch: None,
        } = &if_expr.kind
        else {
            return None;
        };
        let ExprKind::Binary { op, lhs, rhs } = &cond.kind else {
            return None;
        };
        // Canonical `x[k] CMP bv`: idx on the left, running-best scalar on the right.
        let red_op = match op {
            ast::BinOp::Gt => RED_ARGMAX,
            ast::BinOp::Lt => RED_ARGMIN,
            _ => return None,
        };
        let x = self.index_by_loopvar(lhs, k)?;
        let bv = single_path(rhs)?;
        // then-branch: exactly `bv = x[k]; bi = k;` (order-flexible), no tail value.
        if then_branch.tail.is_some() || then_branch.stmts.len() != 2 {
            return None;
        }
        let mut bi: Option<Symbol> = None;
        let mut saw_val = false;
        for s in &then_branch.stmts {
            let StmtKind::Assign {
                target,
                op: ast::AssignOp::Assign,
                value,
            } = &s.kind
            else {
                return None;
            };
            let t = single_path(target)?;
            if t == bv {
                // bv = x[k]
                if self.index_by_loopvar(value, k) != Some(x) {
                    return None;
                }
                saw_val = true;
            } else {
                // bi = k  (or `bi = k as <int>`)
                let is_k = single_path(value) == Some(k)
                    || matches!(&value.kind, ExprKind::Cast { expr, .. } if single_path(expr) == Some(k));
                if !is_k {
                    return None;
                }
                bi = Some(t);
            }
        }
        match (saw_val, bi) {
            (true, Some(bi)) => Some((bv, bi, red_op, x)),
            _ => None,
        }
    }

    /// Lower a recognized argmax/argmin loop `for k in 0..n { if x[k] CMP bv { bv = x[k]; bi = k } }`
    /// to one `mercury_argreduce_f32(x, n, op)` call plus a branchless reconcile of the kernel's
    /// (value, index) against the loop's running `(bv, bi)`: `(bv,bi) = arg_fold((bv,bi),(x[ki],ki))`.
    /// Because `arg_fold` is associative (lowest-index tie-break, a total order) and the loop covers
    /// `x[0..n]` (start 0), the loop result equals this reconcile for **any** seed — so the preceding
    /// `let bv = …; let bi = …;` need not be inspected. Both backends marshal the identical kernel, so
    /// the differential oracle stays exact. Returns false (fall back to the scalar loop) on any mismatch.
    fn try_emit_argreduce(&mut self, pat: &Pattern, start: &Expr, end: &Expr, body: &Block) -> bool {
        // Only `0..n`: the loop must cover the whole array from index 0 so the kernel's reduction over
        // x[0..n], reconciled with the seed, equals the loop independent of the seed value.
        if const_usize_expr(start, self.interner) != Some(0) {
            return false;
        }
        let Pattern {
            kind: ast::PatKind::Ident(k),
            ..
        } = pat
        else {
            return false;
        };
        let Some((bv, bi, op, xb)) = self.match_argreduce_kernel(body, *k) else {
            return false;
        };
        // `bv` an in-scope f32 scalar slot; `bi` an in-scope integer slot; `x` an array base in scope.
        let Some((bv_slot, MirType::F32)) = self.lookup(bv) else {
            return false;
        };
        let Some((bi_slot, bi_ty)) = self.lookup(bi) else {
            return false;
        };
        if !matches!(
            bi_ty,
            MirType::I64 | MirType::I32 | MirType::I16 | MirType::I8
        ) {
            return false;
        }
        let Some((xv, _)) = self.lookup(xb) else {
            return false;
        };
        let n_ty = self.expr_mir(end);
        let n = self.lower_expr(end);
        let n = self.coerce_to(n, &n_ty, &MirType::I64, true);
        let opv = self
            .builder
            .build(MirType::I64, Op::ConstInt(op as i128, MirType::I64));
        // ki = lowest-index argmax/argmin over x[0..n]; kv = x[ki]. In a `@parallel` function the
        // multicore kernel is selected — it folds the same fixed chunk decomposition in ascending
        // order, so the returned index is bit-identical to the serial one the interpreter calls.
        let argreduce_sym = if self.parallel_fn {
            self.gemm.argreduce_par
        } else {
            self.gemm.argreduce
        };
        let ki = self.builder.build(
            MirType::I64,
            Op::Call {
                func: argreduce_sym,
                args: vec![xv, n, opv],
            },
        );
        let kptr = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: xv,
                index: ki,
                elem: MirType::F32,
            },
        );
        let kv = self.builder.build(MirType::F32, Op::Load(kptr, MirType::F32));
        // Reconcile with the running (bv, bi): better = (kv CMP bv) || (kv == bv && ki < bi).
        let bv_cur = self
            .builder
            .build(MirType::F32, Op::Load(bv_slot, MirType::F32));
        let bi_cur = self
            .builder
            .build(bi_ty.clone(), Op::Load(bi_slot, bi_ty.clone()));
        let pred = if op == RED_ARGMAX {
            CmpOp::Fogt
        } else {
            CmpOp::Folt
        };
        let strictly = self
            .builder
            .build(MirType::I1, Op::Cmp(pred, kv, bv_cur));
        let eq = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Foeq, kv, bv_cur));
        let ki_bi = self.coerce_to(ki, &MirType::I64, &bi_ty, true);
        let idx_lt = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, ki_bi, bi_cur));
        let tie = self.builder.build(MirType::I1, Op::Bin(BinOp::And, eq, idx_lt));
        let better = self
            .builder
            .build(MirType::I1, Op::Bin(BinOp::Or, strictly, tie));
        let new_bv = self
            .builder
            .build(MirType::F32, Op::Select(better, kv, bv_cur));
        let new_bi = self
            .builder
            .build(bi_ty.clone(), Op::Select(better, ki_bi, bi_cur));
        self.builder.build_void(Op::Store {
            ptr: bv_slot,
            value: new_bv,
        });
        self.builder.build_void(Op::Store {
            ptr: bi_slot,
            value: new_bi,
        });
        true
    }

    /// Lower a recognized `@parallel` reduction `for k in 0..n { s += f(x[k], y[k]) }` to one
    /// `s = s + mercury_sreduce_f32_parallel(x, y, n, op)`. The kernel returns the same value the loop
    /// would (a reassociation of the same terms), and the interpreter calls the identical *serial*
    /// kernel — which is bit-equal to the parallel one — so native and interp agree. Returns false
    /// (fall back to the scalar/vector loop) unless the range is `0..n`, the accumulator is an
    /// in-scope f32 scalar, and both arrays are in scope.
    fn try_emit_parallel_reduction(&mut self, pat: &Pattern, iter: &ForIter, body: &Block) -> bool {
        let ForIter::Range {
            start,
            end: Some(end),
            inclusive: false,
            step: None,
        } = iter
        else {
            return false;
        };
        // Only `0..n`; a non-zero start would need a pointer/length shift the call does not do.
        if const_usize_expr(start, self.interner) != Some(0) {
            return false;
        }
        let Pattern {
            kind: ast::PatKind::Ident(k),
            ..
        } = pat
        else {
            return false;
        };
        let Some((s, op, xb, yb)) = self.match_reduction_kernel(body, *k) else {
            return false;
        };
        // Accumulator must be an in-scope f32 scalar; both arrays must be in scope (base pointers).
        let Some((s_slot, MirType::F32)) = self.lookup(s) else {
            return false;
        };
        // Resolve each operand's base pointer — `kernel_base_ptr` loads it out of a `Tensor[..]`/
        // pointer param's slot (a no-op for an array operand). A plain `self.lookup(..).0` here passed
        // a 1-D `Tensor` param's *slot address* to the kernel: the interpreter trapped while native
        // read past the slot — the 1-D-tensor reduction divergence.
        let (Some(xv), Some(yv)) = (self.kernel_base_ptr(xb), self.kernel_base_ptr(yb)) else {
            return false;
        };
        let n_ty = self.expr_mir(end);
        let n = self.lower_expr(end);
        let n = self.coerce_to(n, &n_ty, &MirType::I64, true);
        let opv = self
            .builder
            .build(MirType::I64, Op::ConstInt(op as i128, MirType::I64));
        let result = self.builder.build(
            MirType::F32,
            Op::Call {
                func: self.gemm.sred_par,
                args: vec![xv, yv, n, opv],
            },
        );
        // Combine the kernel result into the accumulator. Additive: `s = s + result` (matches the
        // loop's `s_final = s_init + Σ`, reassociated inside the kernel). Max/min: `s = fmax(s,
        // result)` as `Cmp(Fogt/Folt)+Select` — the identical fold the kernel and the sequential
        // `fmax` vectorizer use, so interp (serial kernel) and native (parallel kernel) agree.
        let cur = self
            .builder
            .build(MirType::F32, Op::Load(s_slot, MirType::F32));
        let new_s = match op {
            RED_MAX | RED_MIN | RED_MAXABS => {
                // max/maxabs fold by fmax (Fogt); min by fmin (Folt).
                let pred = if op == RED_MIN {
                    CmpOp::Folt
                } else {
                    CmpOp::Fogt
                };
                let mask = self
                    .builder
                    .build(mask_ty(&MirType::F32), Op::Cmp(pred, cur, result));
                self.builder
                    .build(MirType::F32, Op::Select(mask, cur, result))
            }
            _ => self
                .builder
                .build(MirType::F32, Op::Bin(BinOp::FAdd, cur, result)),
        };
        self.builder.build_void(Op::Store {
            ptr: s_slot,
            value: new_s,
        });
        true
    }

    /// Recognize a **bf16 mixed-precision** reduction body over `k`: `s = s + (x[k] as f32)` (sum) or
    /// `s = s + (x[k] as f32) * (y[k] as f32)` (dot), where `x` (and `y`) are `[bf16; _]` arrays read
    /// through an `as f32` widening cast. Returns the accumulator symbol, `is_dot`, and the array
    /// bases (`y == x` for sum). Strict pure-AST match — single statement, index exactly `k`, both
    /// loads bf16, cast target f32. The f32 accumulate is the standard ML mixed-precision contract.
    fn match_lowp_reduction(
        &self,
        body: &Block,
        k: Symbol,
    ) -> Option<(Symbol, i64, Symbol, Symbol, bool)> {
        if body.tail.is_some() || body.stmts.len() != 1 {
            return None;
        }
        let StmtKind::Assign { target, op, value } = &body.stmts[0].kind else {
            return None;
        };
        let s = single_path(target)?;
        if s == k {
            return None;
        }
        // `(base[k] as f32)` where `base` is a `[bf16]` *or* `[f16]` array indexed exactly by `k`: peel
        // the cast (must target f32), require the inner scalar be bf16/f16, return the base symbol and
        // which precision it is (`true` = f16). The reduction dispatches to the matching half kernel.
        let lowp_load = |e: &Expr| -> Option<(Symbol, bool)> {
            let ExprKind::Cast { expr: inner, .. } = &e.kind else {
                return None;
            };
            if scalar_of(e, self.sema) != Some(mercury_types::Scalar::F32) {
                return None;
            }
            let ExprKind::Index { base, indices } = &inner.kind else {
                return None;
            };
            if indices.len() != 1 || single_path(&indices[0]) != Some(k) {
                return None;
            }
            let is_f16 = match scalar_of(inner, self.sema) {
                Some(mercury_types::Scalar::Bf16) => false,
                Some(mercury_types::Scalar::F16) => true,
                _ => return None,
            };
            Some((single_path(base)?, is_f16))
        };
        // Running max/min over widened bf16/f16: `m = fmax(m, (x[k] as f32))` / `fmin` (either operand
        // order), or — for fmax — `m = fmax(m, abs((x[k] as f32)))` (running **absmax**, the symmetric
        // int8-quant scale a low-precision weight tensor needs). Dispatches to `mercury_reduce_{bf16,
        // f16}` (op-coded), outer-combined by `Cmp+Select` exactly like the f32 reduction. Mirrors
        // `match_reduction_kernel` but through the widening load.
        if let ast::AssignOp::Assign = op {
            if let ExprKind::Call { callee, args, .. } = &value.kind {
                if args.len() == 2 {
                    let red = match self.vectorizable_intrinsic(callee) {
                        Some(MathIntrinsic::Fmax) => Some(RED_MAX),
                        Some(MathIntrinsic::Fmin) => Some(RED_MIN),
                        _ => None,
                    };
                    if let Some(red) = red {
                        let other = if single_path(&args[0]) == Some(s) {
                            &args[1]
                        } else if single_path(&args[1]) == Some(s) {
                            &args[0]
                        } else {
                            return None;
                        };
                        if let Some((a, f16)) = lowp_load(other) {
                            return Some((s, red, a, a, f16));
                        }
                        if red == RED_MAX {
                            if let ExprKind::Call {
                                callee: ac,
                                args: aargs,
                                ..
                            } = &other.kind
                            {
                                if aargs.len() == 1
                                    && matches!(
                                        self.vectorizable_intrinsic(ac),
                                        Some(MathIntrinsic::Abs)
                                    )
                                {
                                    let (a, f16) = lowp_load(&aargs[0])?;
                                    return Some((s, RED_MAXABS, a, a, f16));
                                }
                            }
                        }
                        return None;
                    }
                }
            }
        }
        // `s += addend`, or `s = s + addend` / `s = addend + s`.
        let addend: &Expr = match op {
            ast::AssignOp::Add => value,
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(lhs) == Some(s) => rhs,
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(rhs) == Some(s) => lhs,
                _ => return None,
            },
            _ => return None,
        };
        if expr_uses_sym(addend, s) {
            return None;
        }
        match &addend.kind {
            // dot: `(x[k] as f32) * (y[k] as f32)` — both operands must be the same precision.
            ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } => {
                let (xb, xf) = lowp_load(lhs)?;
                let (yb, yf) = lowp_load(rhs)?;
                if xf != yf {
                    return None;
                }
                Some((s, RED_DOT, xb, yb, xf))
            }
            // sum: `(x[k] as f32)`
            ExprKind::Cast { .. } => {
                let (xb, xf) = lowp_load(addend)?;
                Some((s, RED_SUM, xb, xb, xf))
            }
            _ => None,
        }
    }

    /// Pure recognizer for a **bf16→f32 axpby** body `out[k] = A + B` where each of `A`, `B` is either
    /// `(arr[k] as f32)` (coefficient 1) or `coef * (arr[k] as f32)` with `coef` loop-invariant — i.e.
    /// `out[k] = a*(x[k] as f32) + b*(y[k] as f32)`, the mixed-precision saxpy/axpby (bf16 inputs, f32
    /// output). Requires **two** additive terms (a 1-term scale `a*x[k]` would force `b=0, y=x` and a
    /// `0*inf=NaN` the source never has — and the interp==native gate, both calling the same kernel,
    /// would not catch it). Returns `(out, x, y, a_coef?, b_coef?)`, where a `None` coef means literal 1.
    #[allow(clippy::type_complexity)]
    fn match_lowp_axpby<'b>(
        &self,
        body: &'b Block,
        k: Symbol,
    ) -> Option<(
        Symbol,
        Symbol,
        Symbol,
        Option<&'b Expr>,
        Option<&'b Expr>,
        bool,
    )> {
        if body.tail.is_some() || body.stmts.len() != 1 {
            return None;
        }
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &body.stmts[0].kind
        else {
            return None;
        };
        // Target is `out[k]` with `out` an f32 array indexed exactly by k.
        let ExprKind::Index { base, indices } = &target.kind else {
            return None;
        };
        if indices.len() != 1 || single_path(&indices[0]) != Some(k) {
            return None;
        }
        if scalar_of(target, self.sema) != Some(mercury_types::Scalar::F32) {
            return None;
        }
        let out = single_path(base)?;
        // `(arr[k] as f32)` with arr `[bf16]`/`[f16]`, indexed exactly by k → (arr symbol, is_f16).
        let lowp_load = |e: &Expr| -> Option<(Symbol, bool)> {
            let ExprKind::Cast { expr: inner, .. } = &e.kind else {
                return None;
            };
            if scalar_of(e, self.sema) != Some(mercury_types::Scalar::F32) {
                return None;
            }
            let ExprKind::Index { base, indices } = &inner.kind else {
                return None;
            };
            if indices.len() != 1 || single_path(&indices[0]) != Some(k) {
                return None;
            }
            let is_f16 = match scalar_of(inner, self.sema) {
                Some(mercury_types::Scalar::Bf16) => false,
                Some(mercury_types::Scalar::F16) => true,
                _ => return None,
            };
            Some((single_path(base)?, is_f16))
        };
        // One additive term → (coef?, arr, is_f16). `coef * load` (either factor order) or a bare load.
        let term = |e: &'b Expr| -> Option<(Option<&'b Expr>, Symbol, bool)> {
            if let Some((arr, f16)) = lowp_load(e) {
                return Some((None, arr, f16));
            }
            let ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } = &e.kind
            else {
                return None;
            };
            if let Some((arr, f16)) = lowp_load(rhs) {
                if !expr_uses_sym(lhs, k) {
                    return Some((Some(lhs.as_ref()), arr, f16));
                }
            }
            if let Some((arr, f16)) = lowp_load(lhs) {
                if !expr_uses_sym(rhs, k) {
                    return Some((Some(rhs.as_ref()), arr, f16));
                }
            }
            None
        };
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs,
            rhs,
        } = &value.kind
        else {
            return None;
        };
        let (a, x, xf) = term(lhs)?;
        let (b, y, yf) = term(rhs)?;
        // Both inputs must be the same precision (one kernel widens one width).
        if xf != yf {
            return None;
        }
        Some((out, x, y, a, b, xf))
    }

    /// Lower a recognized bf16→f32 axpby `for k in 0..n { out[k] = a*(x[k] as f32) + b*(y[k] as f32) }`
    /// to one `mercury_axpby_bf16(x, y, out, n, a, b)` call (bf16 in, f32 out, f32 math). Half-width
    /// inputs ⇒ ~1.5× the streamed bytes saved vs the f32 kernel. Both backends marshal the identical
    /// kernel, so the differential gate stays exact. Falls back unless the range is `0..n` and out / x /
    /// y are in scope.
    fn try_emit_bf16_axpby(&mut self, pat: &Pattern, iter: &ForIter, body: &Block) -> bool {
        let ForIter::Range {
            start,
            end: Some(end),
            inclusive: false,
            step: None,
        } = iter
        else {
            return false;
        };
        if const_usize_expr(start, self.interner) != Some(0) {
            return false;
        }
        let Pattern {
            kind: ast::PatKind::Ident(k),
            ..
        } = pat
        else {
            return false;
        };
        let Some((out, x, y, a_expr, b_expr, is_f16)) = self.match_lowp_axpby(body, *k) else {
            return false;
        };
        let (Some((outv, _)), Some((xv, _)), Some((yv, _))) =
            (self.lookup(out), self.lookup(x), self.lookup(y))
        else {
            return false;
        };
        // Coefficients: lower the invariant expr (coerced to f32), or a literal 1.0 when implicit.
        let mut coef = |e: Option<&Expr>| -> ValueId {
            match e {
                Some(e) => {
                    let ty = self.expr_mir(e);
                    let v = self.lower_expr(e);
                    self.coerce_to(v, &ty, &MirType::F32, true)
                }
                None => self
                    .builder
                    .build(MirType::F32, Op::ConstFloat(1.0, MirType::F32)),
            }
        };
        let av = coef(a_expr);
        let bv = coef(b_expr);
        let n_ty = self.expr_mir(end);
        let n = self.lower_expr(end);
        let n = self.coerce_to(n, &n_ty, &MirType::I64, true);
        self.builder.build_void(Op::Call {
            func: if is_f16 {
                self.gemm.axpby_f16
            } else {
                self.gemm.axpby_bf16
            },
            args: vec![xv, yv, outv, n, av, bv],
        });
        true
    }

    /// Lower a recognized bf16 reduction `for k in 0..n { s += (x[k] as f32) [* (y[k] as f32)] }` to
    /// `s = s + mercury_dot_bf16(x, y, n)` (or `mercury_sum_bf16(x, n)`). The kernel widens bf16→f32
    /// and accumulates in f32; the interpreter marshals the identical kernel (reconstructing the bf16
    /// bits from its bf16-rounded storage), so native and interp agree bit-for-bit despite the
    /// kernel's reassociated 8-lane accumulation. Falls back (returns false) unless the range is
    /// `0..n`, the accumulator is an in-scope f32 scalar, and the arrays are in scope.
    fn try_emit_lowp_reduction(&mut self, pat: &Pattern, iter: &ForIter, body: &Block) -> bool {
        let ForIter::Range {
            start,
            end: Some(end),
            inclusive: false,
            step: None,
        } = iter
        else {
            return false;
        };
        if const_usize_expr(start, self.interner) != Some(0) {
            return false;
        }
        let Pattern {
            kind: ast::PatKind::Ident(k),
            ..
        } = pat
        else {
            return false;
        };
        let Some((s, red_op, xb, yb, is_f16)) = self.match_lowp_reduction(body, *k) else {
            return false;
        };
        // Accumulator must be an in-scope f32 scalar; both arrays must be in scope (base pointers).
        let Some((s_slot, MirType::F32)) = self.lookup(s) else {
            return false;
        };
        // Resolve each operand's base pointer — `kernel_base_ptr` loads it out of a `Tensor[..]`/
        // pointer param's slot (a no-op for an array operand). A plain `self.lookup(..).0` here passed
        // a 1-D `Tensor` param's *slot address* to the kernel: the interpreter trapped while native
        // read past the slot — the 1-D-tensor reduction divergence.
        let (Some(xv), Some(yv)) = (self.kernel_base_ptr(xb), self.kernel_base_ptr(yb)) else {
            return false;
        };
        let n_ty = self.expr_mir(end);
        let n = self.lower_expr(end);
        let n = self.coerce_to(n, &n_ty, &MirType::I64, true);
        // Pick the kernel by op and precision: additive dot/sum have dedicated symbols; the max-family
        // routes through the op-coded reduce kernel. f16 uses the F16C twins, bf16 the `<<16` ones.
        let (func, args) = match red_op {
            RED_DOT => (
                if is_f16 {
                    self.gemm.dot_f16
                } else {
                    self.gemm.dot_bf16
                },
                vec![xv, yv, n],
            ),
            RED_SUM => (
                if is_f16 {
                    self.gemm.sum_f16
                } else {
                    self.gemm.sum_bf16
                },
                vec![xv, n],
            ),
            _ => {
                let opv = self
                    .builder
                    .build(MirType::I64, Op::ConstInt(red_op as i128, MirType::I64));
                (
                    if is_f16 {
                        self.gemm.reduce_f16
                    } else {
                        self.gemm.reduce_bf16
                    },
                    vec![xv, n, opv],
                )
            }
        };
        let result = self.builder.build(MirType::F32, Op::Call { func, args });
        let cur = self
            .builder
            .build(MirType::F32, Op::Load(s_slot, MirType::F32));
        // Combine the kernel result into the accumulator — additive: `s = s + result` (matches the
        // loop's reassociated `s_final = s_init + Σ`); max/min: `s = fmax/fmin(s, result)` as
        // `Cmp(Fogt/Folt)+Select`, the identical fold the kernel and the source loop use, so interp
        // (same kernel) and native agree.
        let new_s = match red_op {
            RED_MAX | RED_MIN | RED_MAXABS => {
                let pred = if red_op == RED_MIN {
                    CmpOp::Folt
                } else {
                    CmpOp::Fogt
                };
                let mask = self
                    .builder
                    .build(mask_ty(&MirType::F32), Op::Cmp(pred, cur, result));
                self.builder
                    .build(MirType::F32, Op::Select(mask, cur, result))
            }
            _ => self
                .builder
                .build(MirType::F32, Op::Bin(BinOp::FAdd, cur, result)),
        };
        self.builder.build_void(Op::Store {
            ptr: s_slot,
            value: new_s,
        });
        true
    }

    /// Pure structural recognizer for a **batched** row normalization `for r in 0..R { <per-row norm
    /// over x[r*C + i]> }` over a flat `[R, C]` matrix. The body must be a complete norm window whose
    /// data accesses are row-offset-indexed by `r*C` (verified by `match_rmsnorm`'s offset machinery),
    /// consuming every body statement. Returns the data array, the per-row width `C`, eps, and any
    /// affine params — everything `emit_norm` needs besides `rows = R` (the caller holds the loop
    /// bound). `&self` (reads sema/interner, emits no MIR), so it doubles as the whole-function probe
    /// the `@parallel` driver runs before the generic outliner. RMSNorm only for now (the highest-value
    /// modern norm); the same offset path generalizes to LayerNorm/softmax.
    fn match_batched_norm(
        &self,
        pat: &Pattern,
        iter: &ForIter,
        body: &Block,
    ) -> Option<(Symbol, Expr, i64, i64, Option<Symbol>, Option<Symbol>)> {
        let ForIter::Range {
            start,
            end: Some(_),
            inclusive: false,
            step: None,
        } = iter
        else {
            return None;
        };
        if const_usize_expr(start, self.interner) != Some(0) {
            return None;
        }
        let Pattern {
            kind: ast::PatKind::Ident(r),
            ..
        } = pat
        else {
            return None;
        };
        // The body must be exactly one norm window (offset-indexed by `r*C`), nothing else. softmax
        // (7 stmts, fmax/exp), LayerNorm (7 stmts, mean/variance), and RMSNorm (4 stmts) are pairwise
        // structurally disjoint, so probe order is immaterial.
        if body.tail.is_some() {
            return None;
        }
        if let Some((consumed, x, cols)) = self.match_softmax(body, 0, Some(*r)) {
            if consumed == body.stmts.len() {
                return Some((x, cols, 0, NORM_SOFTMAX, None, None));
            }
        }
        if let Some((consumed, x, cols, eps, gamma, beta)) = self.match_layernorm(body, 0, Some(*r))
        {
            if consumed == body.stmts.len() {
                return Some((x, cols, eps, NORM_LAYERNORM, gamma, beta));
            }
        }
        if let Some((consumed, x, cols, eps, gamma, beta)) = self.match_rmsnorm(body, 0, Some(*r)) {
            if consumed == body.stmts.len() {
                return Some((x, cols, eps, NORM_RMSNORM, gamma, beta));
            }
        }
        if let Some((consumed, x, cols, eps, _g, _b)) = self.match_l2norm(body, 0, Some(*r)) {
            if consumed == body.stmts.len() {
                return Some((x, cols, eps, NORM_L2NORM, None, None));
            }
        }
        None
    }

    /// Dispatch a recognized batched norm to one fused norm kernel with `rows = R`. Runs pre-opt, so
    /// `-O0`==`-O3`; both backends marshal the identical kernel, so the differential gate stays
    /// bit-exact. In a `@parallel` function `emit_norm` selects the multicore `_parallel` variant.
    fn try_emit_batched_norm(&mut self, pat: &Pattern, iter: &ForIter, body: &Block) -> bool {
        let ForIter::Range { end: Some(end), .. } = iter else {
            return false;
        };
        if let Some((x, cols, eps, op, gamma, beta)) = self.match_batched_norm(pat, iter, body) {
            return self.emit_norm(x, Some(end), &cols, eps, op, gamma, beta);
        }
        false
    }

    fn lower_for(&mut self, label: Option<Symbol>, pat: &Pattern, iter: &ForIter, body: &Block) {
        // A matmul nest lowers to the tuned microkernel (single-threaded on this statement path; the
        // whole-function `@parallel` form is handled earlier in `lower_program`).
        if let Some(nest) = recognize_matmul(pat, iter, body, self.sema, self.interner) {
            if self.emit_sgemm(&nest, false) {
                return;
            }
        }
        // A fused residual projection `c[i*N+j] = act(c[i*N+j] + dot [+ bias[j]])` — the transformer
        // skip connection `x = x + act(x·Wᵀ + bias)`, whose accumulate store blocks the bare matmul
        // recognizer above (so without this the whole nest falls to a scalar loop). It dispatches to
        // `mercury_sgemm_nt_epi` with beta=1 (the kernel accumulates the matmul into the residual
        // already in C), reusing the fused-epilogue kernel with no backend change. Tried after the
        // bare matmul (the two store forms — `c = s` vs `c = act(c + s + …)` — are disjoint).
        if let Some((nest, bias, act)) =
            match_matmul_residual(pat, iter, body, self.sema, self.interner)
        {
            if self.emit_sgemm_epi(&nest, bias, act) {
                return;
            }
        }
        // int8 quantized `nn.Linear` (`u8×i8→i32` `C = A·Bᵀ`) → the int8 GEMM microkernel. Integer
        // arithmetic, so the kernel equals the scalar nest bit-for-bit; in a `@parallel` function the
        // multicore kernel is used (rows independent → deterministic, so the gate stays exact).
        if let Some(nest) = match_matmul_i8_nt(pat, iter, body, self.sema, self.interner) {
            if self.emit_i8gemm(&nest, self.parallel_fn) {
                return;
            }
        }
        // bf16/f16 mixed-precision `C = A·Bᵀ` (NT, nn.Linear forward) or `C = Aᵀ·B` (TN, the
        // `dW = dYᵀ·X` weight gradient) — half inputs widened to f32, f32 accumulate → the half GEMM
        // microkernel. The widen is lossless, so the kernel equals the scalar nest under the matmul
        // reassociation; in a `@parallel` function the multicore kernel runs (rows independent).
        if let Some(nest) = match_matmul_lowp(pat, iter, body, self.sema, self.interner) {
            if self.emit_lowp_gemm(&nest, self.parallel_fn) {
                return;
            }
        }
        // Matrix transpose `for i { for j { dst[j*R+i] = src[i*C+j] } }` → the cache-blocked
        // `mercury_transpose_f32` (the `_parallel` one in a `@parallel` function). Pure data movement,
        // bit-identical to the scalar nest; the block tiling is the win `-O3` won't do for a transpose.
        if let Some(nest) = match_transpose(pat, iter, body, self.sema, self.interner) {
            if self.emit_transpose(&nest, self.parallel_fn) {
                return;
            }
        }
        // 2D max/avg pooling `for c { for oy { for ox { seed; for dy { for dx { fold window } }; store } } }`
        // → the AVX2 `mercury_{max,avg}pool2d_f32` (the `_parallel` one in a `@parallel` function). The
        // strided window gcc/rustc leave scalar; the kernel folds 8 output columns at once. Max is
        // idempotent and the avg sum order is fixed → bit-identical to the scalar nest.
        if let Some(nest) = match_pool2d(pat, iter, body, self.sema, self.interner) {
            if self.emit_pool2d(&nest, self.parallel_fn) {
                return;
            }
        }
        // Column reduction `for j { let s=0; for i { s += x[i*N+j] }; out[j] = s }` (the bias gradient
        // / batch sum, a reduce along axis 0) → the SIMD `mercury_colsum_f32` (the `_parallel` one in a
        // `@parallel` function). The strided naive form gcc leaves scalar; the kernel streams row-major
        // + 8-wide. Bit-exact (i-ascending per column, same order as the scalar nest).
        if let Some(nest) = match_colsum(pat, iter, body, self.sema, self.interner) {
            if self.emit_colsum(&nest, self.parallel_fn) {
                return;
            }
        }
        // Per-column argmax/argmin `for j { let bv=x[j]; let bi=0; for i in 1..R { if x[i*C+j] >|< bv {
        // bv=x[i*C+j]; bi=i } }; out[j]=bi }` (axis-0 top-1) → the i32-index `mercury_colarg{max,min}_i32`
        // (the `_parallel` one in a `@parallel` fn). The strided column-outer (value,index) scan gcc/rustc
        // keep scalar; the kernel streams row-major tracking 8 column lanes via blend.
        if let Some(nest) = match_colarg(pat, iter, body, self.sema, self.interner) {
            if self.emit_colarg(&nest, self.parallel_fn) {
                return;
            }
        }
        // Batched softmax-backward `for r { let s=0; for j { s += y·dy }; for i { dx = y·(dy−s) } }`
        // → the fused `mercury_softmax_bwd_f32` (the `_parallel` one in a `@parallel` function). The
        // per-row dot gcc keeps scalar; the kernel reuses the bit-exact sreduce dot + an 8-wide apply.
        if let Some(nest) = match_softmax_bwd(pat, iter, body, self.sema, self.interner) {
            if self.emit_softmax_bwd(&nest, self.parallel_fn) {
                return;
            }
        }
        // A batched RMSNorm backward `for r { <dx = r·(g − x·r²·Σg·x/C) over x/dy/gamma[r*C+i]> }` →
        // `mercury_rmsnorm_bwd_f32` (the `_parallel` one in a `@parallel` function). The two per-row
        // reductions gcc keeps scalar; the kernel folds them 8-wide then applies the gradient.
        if let Some(nest) = match_rmsnorm_bwd(pat, iter, body, self.sema, self.interner) {
            if self.emit_rmsnorm_bwd(&nest, self.parallel_fn) {
                return;
            }
        }
        // Batched LayerNorm backward `for r { mean; var; rstd; Σg; Σg·xhat; apply }` →
        // `mercury_layernorm_bwd_f32` (the `_parallel` one in a `@parallel` function). Four per-row
        // reductions gcc keeps scalar; the kernel folds them 8-wide.
        if let Some(nest) = match_layernorm_bwd(pat, iter, body, self.sema, self.interner) {
            if self.emit_layernorm_bwd(&nest, self.parallel_fn) {
                return;
            }
        }
        // A batched softmax cross-entropy loss `for r { max; Σexp; loss[r] = lse − x[r,target[r]] }`
        // → `mercury_xent_fwd_f32` (the `_parallel` one in a `@parallel` function). The expf/logf
        // reduction gcc keeps scalar; the kernel folds it 8-wide and gathers the target logit.
        if let Some(nest) = self.match_xent(pat, iter, body) {
            if self.emit_xent(&nest, self.parallel_fn) {
                return;
            }
        }
        // Batched cross-entropy backward `for r { softmax(x) into dx; dx[target[r]] -= 1 }`
        // → `mercury_xent_bwd_f32` (the `_parallel` one in a `@parallel` function).
        if let Some(nest) = self.match_xent_bwd(pat, iter, body) {
            if self.emit_xent_bwd(&nest, self.parallel_fn) {
                return;
            }
        }
        // A RoPE nest `for r { for j { rotate x[r,j]/x[r,j+H] by inline cos/sin(r·inv_freq[j]) } }`
        // → `mercury_rope_f32` (the `_parallel` one in a `@parallel` function). The inline sinf/cosf
        // gcc keeps scalar; the kernel computes them 8-wide. Bit-identical (a rotation, no reassoc).
        if let Some(nest) = match_rope(pat, iter, body, false, self.sema, self.interner)
            .or_else(|| match_rope(pat, iter, body, true, self.sema, self.interner))
        {
            if self.emit_rope(&nest, self.parallel_fn) {
                return;
            }
        }
        // A batched log-sum-exp `for r { max; Σexp; out[r] = m + log(s) }` → `mercury_logsumexp_f32`
        // (the `_parallel` one in a `@parallel` function). The expf reduction gcc keeps scalar.
        if let Some(nest) = self.match_logsumexp(pat, iter, body) {
            if self.emit_logsumexp(&nest, self.parallel_fn) {
                return;
            }
        }
        // Per-row loss reductions (KL divergence / Shannon entropy / soft-label cross-entropy) → the
        // fused log/exp-bound kernels (the `_parallel` ones in a `@parallel` function). C keeps the
        // logf/expf reductions scalar.
        if let Some(nest) = self.match_kldiv(pat, iter, body) {
            if self.emit_kldiv(&nest, self.parallel_fn) {
                return;
            }
        }
        if let Some(nest) = self.match_entropy(pat, iter, body) {
            if self.emit_entropy(&nest, self.parallel_fn) {
                return;
            }
        }
        if let Some(nest) = self.match_kd_loss(pat, iter, body) {
            if self.emit_kd_loss(&nest, self.parallel_fn) {
                return;
            }
        }
        // Batched per-row argmax/argmin (classification top-1 / greedy decode) → the index-returning
        // `mercury_rowarg{max,min}_i32` (`_parallel` in a `@parallel` fn). The (value,index) bookkeeping
        // keeps gcc/rustc scalar; the AVX2 kernel tracks 8 lanes of (value,index) via blend.
        if let Some(nest) = self.match_rowarg(pat, iter, body) {
            if self.emit_rowarg(&nest, self.parallel_fn) {
                return;
            }
        }
        // Batched per-row inclusive prefix sum (cumsum) → `mercury_cumsum_f32` (the `_parallel` one in a
        // `@parallel` fn). gcc/rustc keep the loop-carried scan scalar; the SIMD Hillis-Steele + carry
        // vectorizes it. The in-lane tree reassociates (the documented exception — both backends run it).
        if let Some(nest) = self.match_cumsum(pat, iter, body) {
            if self.emit_cumsum(&nest, self.parallel_fn) {
                return;
            }
        }
        // Batched per-row inclusive prefix product (cumprod) → `mercury_cumprod_f32` (the `_parallel` one
        // in a `@parallel` fn). Loop-carried like cumsum; the kernel's lever is 4-row-interleaved ILP.
        // Bit-exact (a bare product is not fused → strict left-to-right, no reassociation).
        if let Some(nest) = self.match_cumprod(pat, iter, body) {
            if self.emit_cumprod(&nest, self.parallel_fn) {
                return;
            }
        }
        // Batched per-row first-order linear recurrence (SSM/Mamba/EMA selective scan) →
        // `mercury_lrscan_f32` (the `_parallel` one in a `@parallel` fn). The carry defeats gcc/rustc
        // auto-vectorization (scalar, like cumsum); rows independent → multicore over rows. Bit-exact
        // (the recurrence is inherently sequential within a row — no reassociation).
        if let Some(nest) = self.match_lrscan(pat, iter, body) {
            if self.emit_lrscan(&nest, self.parallel_fn) {
                return;
            }
        }
        // Batched per-row cumulative max / min (running extreme scan) → mercury_cum{max,min}_f32. Same
        // loop-carried scan gcc/rustc keep scalar; bit-exact (max/min select a value, no reassociation).
        if let Some(nest) = self.match_cumminmax(pat, iter, body) {
            if self.emit_cumminmax(&nest, self.parallel_fn) {
                return;
            }
        }
        // Embedding lookup `for t { for d { out[t*H+d] = weight[ids[t]*H+d] } }` (the first layer of every
        // LLM — token ids gather rows of the embedding table) → `mercury_embedding_f32` (the `_parallel`
        // one in a `@parallel` fn). The data-dependent row gather is pure data movement, bit-identical to
        // the scalar nest (no reassociation, like the transpose); rows independent → parallel == serial.
        if let Some(nest) = self.match_embedding(pat, iter, body) {
            if self.emit_embedding(&nest, self.parallel_fn) {
                return;
            }
        }
        // Scatter-add / embedding-gradient backward `for t { for d { grad_w[ids[t]*H+d] += grad_out[t*H+d] } }`
        // → `mercury_scatter_add_f32` (the `_parallel` one in a `@parallel` fn). The dual of the embedding
        // gather; bit-identical to the scalar nest (collisions sum in token order, no reassociation). It is
        // probed AFTER embedding — disjoint store ops (`+=` indirect-write vs `=` indirect-read), so neither
        // steals the other's nest.
        if let Some(nest) = self.match_scatter(pat, iter, body) {
            if self.emit_scatter(&nest, self.parallel_fn) {
                return;
            }
        }
        // Inside a `@parallel` function, a recognized reduction loop (`s += x[k]*y[k]`, etc.) lowers
        // to one multicore `mercury_sreduce_f32_parallel` call instead of the sequential vectorizer.
        if self.parallel_fn && self.try_emit_parallel_reduction(pat, iter, body) {
            return;
        }
        // A bf16 mixed-precision reduction `for k in 0..n { s += (x[k] as f32) [* (y[k] as f32)] }`
        // over `[bf16; _]` arrays with an f32 accumulator dispatches to the bf16 dot/sum kernel (f32
        // accumulate). bf16 storage is bit-exact across interp/native and both call the identical
        // kernel, so the differential gate stays exact (a reassociation exception, like the f32 path).
        if self.try_emit_lowp_reduction(pat, iter, body) {
            return;
        }
        // A bf16 mixed-precision elementwise `for k in 0..n { out[k] = a*(x[k] as f32) + b*(y[k] as
        // f32) }` (bf16 inputs, f32 output) dispatches to the streaming axpby kernel — half-width
        // inputs, so ~1.5× the bytes saved on this memory-bound shape. Same exact-gate rationale.
        if self.try_emit_bf16_axpby(pat, iter, body) {
            return;
        }
        // A `for r in 0..R { <per-row norm over x[r*C + i]> }` batched normalization dispatches to the
        // fused single-pass norm kernel with `rows = R` (in a `@parallel` fn, the multicore variant
        // that maps rows across cores). The real transformer shape: norm over `[batch*seq, hidden]`.
        if self.try_emit_batched_norm(pat, iter, body) {
            return;
        }
        // A `for r in 0..R { for j in 0..C { out[r*C + j] = f(x[r*C + j]) } }` batched activation nest
        // (the `[tokens, hidden]` FFN/attention shape) dispatches to one flat 256-bit `mercury_vmath_f32`
        // over the whole `[0, R*C)` buffer — restoring the full kernel width the offset-indexed inner
        // loop would otherwise lose to the 128-bit generic vectorizer.
        if self.try_emit_batched_vmath(pat, iter, body) {
            return;
        }
        let (start, end, inclusive, step) = match iter {
            ForIter::Range {
                start,
                end: Some(end),
                inclusive,
                step,
            } => (start, end, *inclusive, step),
            _ => {
                // `for <pat> in arr` over a fixed-size array desugars to the indexed range loop
                // `for i in 0..N { let <pat> = arr[i]; <body> }` — a pure front-end rewrite reusing
                // the existing Gep/Load/CFG, so the interpreter and native backend agree bit-for-bit
                // with no backend change. Anything not a statically-sized array (a tensor, slice,
                // dynamic length, or a non-`Ident`/`_` pattern) declines and stays unsupported.
                if let ForIter::Expr(e) = iter {
                    if self.lower_for_array(label, pat, e, body) {
                        return;
                    }
                }
                self.unsupported(body.span, "for over a non-range iterator");
                return;
            }
        };

        // Drive the loop by the wider of the start/end bound types: a literal-`0` start lowers to
        // `i32`, so `for i in 0..n` with `n: i64` would make an `i32` counter and then compare it to
        // the `i64` end — verifier-invalid MIR (`cmp.i32 i32, i64`) that crashed the native backend.
        // (Only the *scalar* loop hit this: an array loop vectorizes this away, so the bug surfaced
        // only on the tensor-param path, which the vectorizer declines.) The body's index arithmetic
        // widens with the counter. The `@parallel` range path already drives by the end's type.
        let sty = self.expr_mir(start);
        let ety = self.expr_mir(end);
        let (ity, signed) = if ety.is_int() && sty.is_int() && mir_byte_size(&ety) > mir_byte_size(&sty)
        {
            (ety.clone(), self.signed(end))
        } else {
            (sty.clone(), self.signed(start))
        };

        // argmax/argmin: `for k in 0..n { if x[k] CMP bv { bv = x[k]; bi = k } }` → one deterministic
        // `mercury_argreduce_f32` call + a branchless reconcile. Tried before the vectorizer (which
        // would otherwise if-convert the branch into a scalar lane loop). The greedy-decode hot path.
        if !inclusive && step.is_none() && self.try_emit_argreduce(pat, start, end, body) {
            return;
        }

        // Straight-line elementwise loops lower to SIMD (vector main loop + scalar remainder); this
        // is purely an optimization, so on any doubt it returns false and we lower scalar below.
        if !inclusive && step.is_none() && self.try_vectorize_for(pat, start, end, body) {
            return;
        }

        // i = start
        let slot = self.builder.alloca(ity.clone());
        let s0 = self.lower_expr(start);
        let s0 = self.coerce_to(s0, &sty, &ity, signed);
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: s0,
        });

        self.push_scope();
        if let Pattern {
            kind: ast::PatKind::Ident(name),
            ..
        } = pat
        {
            self.bind(*name, slot, ity.clone());
        }

        let header = self.builder.new_block();
        let body_bb = self.builder.new_block();
        let latch = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);

        // header: i < end (or <=)
        self.builder.switch_to(header);
        self.terminated = false;
        let i_val = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let end_val = self.lower_expr(end);
        let end_val = self.coerce_to(end_val, &ety, &ity, signed);
        let pred = match (inclusive, signed) {
            (false, true) => CmpOp::Slt,
            (true, true) => CmpOp::Sle,
            (false, false) => CmpOp::Ult,
            (true, false) => CmpOp::Ule,
        };
        let c = self
            .builder
            .build(MirType::I1, Op::Cmp(pred, i_val, end_val));
        self.builder.cond_br(c, body_bb, vec![], exit, vec![]);

        // body. `continue` must target the *latch* (which performs `i += step`), not the header, or
        // the step is skipped and `for i in 0..n { …; continue; }` loops forever (the step lives at
        // the body's tail, unlike a `while`, whose header re-evaluates the user's own condition).
        self.builder.switch_to(body_bb);
        self.terminated = false;
        self.loops.push((label, latch, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            self.builder.br(latch, vec![]);
        }

        // latch: i += step; back to the header. Reached by the body's fall-through and every
        // `continue`, so the loop variable always advances.
        self.builder.switch_to(latch);
        self.terminated = false;
        let cur = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let step_val = match step {
            Some(st) => self.lower_expr(st),
            None => self
                .builder
                .build(ity.clone(), Op::ConstInt(1, ity.clone())),
        };
        let next = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, cur, step_val));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: next,
        });
        self.builder.br(header, vec![]);

        self.pop_scope();
        self.builder.switch_to(exit);
        self.terminated = false;
    }

    /// Desugar `for <pat> in <array>` — iterate the elements of a fixed-size array — into the same
    /// CFG the indexed range loop `for i in 0..N { let <pat> = <array>[i]; <body> }` produces.
    /// Returns `true` if it handled the loop. It fires only when the iterand's sema type is a
    /// statically sized `[T; N]` and the pattern is an identifier (`for x in a`) or `_` wildcard
    /// (`for _ in a`); anything else (a tensor, slice, dynamic length, or a destructuring pattern)
    /// returns `false` so the caller emits the existing `unsupported` diagnostic. Pure desugaring:
    /// the array base pointer + per-iteration `Gep`/`Load` are exactly what `for i in 0..N { let x
    /// = a[i]; … }` already lowers to, so the interpreter and native backend agree bit-for-bit with
    /// **no backend change** (and `-O0`==`-O3`, since recognition runs pre-opt like the rest).
    fn lower_for_array(
        &mut self,
        label: Option<Symbol>,
        pat: &Pattern,
        e: &Expr,
        body: &Block,
    ) -> bool {
        // Only a bare identifier (`for x in a`) or wildcard (`for _ in a`) is supported; a
        // destructuring / literal pattern declines to the `unsupported` fallback.
        let bind_name = match &pat.kind {
            ast::PatKind::Ident(name) => Some(*name),
            ast::PatKind::Wildcard => None,
            _ => return false,
        };
        // The iterand must be a fixed-size array; its length N and element type come straight from
        // sema. A tensor / slice / dynamic-length iterand has no `Ty::Array` here, so it declines.
        let Ty::Array { elem, len } = self.expr_ty(e) else {
            return false;
        };
        let elem_mir = self.mir_ty_of(&elem);
        let n = len as i128;

        // The array's base pointer, evaluated once before the loop. An array local/param's bound
        // `ValueId` *is* its base pointer; any other array-typed expression (a struct field, an
        // element of an array-of-arrays, an array literal) likewise lowers to its base address (the
        // by-pointer convention `lower_expr` upholds for every aggregate), so a base pointer is
        // always recoverable for a `Ty::Array` iterand.
        let base_ptr = self.lower_expr(e);

        // i = 0 — the hidden induction variable, I64 like the array-index GEPs.
        let ity = MirType::I64;
        let slot = self.builder.alloca(ity.clone());
        let zero = self
            .builder
            .build(ity.clone(), Op::ConstInt(0, ity.clone()));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: zero,
        });

        self.push_scope();

        let header = self.builder.new_block();
        let body_bb = self.builder.new_block();
        let latch = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);

        // header: i < N
        self.builder.switch_to(header);
        self.terminated = false;
        let i_val = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let nval = self
            .builder
            .build(ity.clone(), Op::ConstInt(n, ity.clone()));
        let c = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, i_val, nval));
        self.builder.cond_br(c, body_bb, vec![], exit, vec![]);

        // body: bind `<pat>` to `array[i]`, then lower the user body. `continue` targets the
        // *latch* (which performs `i += 1`), matching the range-`for` so `continue` advances.
        self.builder.switch_to(body_bb);
        self.terminated = false;
        if let Some(name) = bind_name {
            let i_cur = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
            let elem_ptr = self.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: base_ptr,
                    index: i_cur,
                    elem: elem_mir.clone(),
                },
            );
            if matches!(elem_mir, MirType::Array(..)) {
                // An aggregate element (struct / tuple / array) binds **by pointer** — `x.field`
                // / `x[k]` / `x.0` GEP off it — exactly as the `a[i]` aggregate read arm does.
                self.bind(name, elem_ptr, elem_mir.clone());
            } else {
                // A scalar element loads into a fresh slot (hoisted to the entry block), so the
                // loop variable is an ordinary mutable local — the faithful `let x = a[i]`.
                let v = self
                    .builder
                    .build(elem_mir.clone(), Op::Load(elem_ptr, elem_mir.clone()));
                let xslot = self.builder.alloca(elem_mir.clone());
                self.builder.build_void(Op::Store {
                    ptr: xslot,
                    value: v,
                });
                self.bind(name, xslot, elem_mir.clone());
            }
        }
        self.loops.push((label, latch, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            self.builder.br(latch, vec![]);
        }

        // latch: i += 1; back to the header (reached by fall-through and every `continue`).
        self.builder.switch_to(latch);
        self.terminated = false;
        let cur = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let one = self
            .builder
            .build(ity.clone(), Op::ConstInt(1, ity.clone()));
        let next = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, cur, one));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: next,
        });
        self.builder.br(header, vec![]);

        self.pop_scope();
        self.builder.switch_to(exit);
        self.terminated = false;
        true
    }

    // ============================ SIMD loop vectorizer ============================
    //
    // Recognizes `for j in lo..hi { <straight-line elementwise body> }` and lowers it to a vector
    // main loop processing `W` lanes per iteration plus a scalar remainder for the tail. It fires
    // only when every array access is unit-stride in `j` (vector load/store) or invariant in `j`
    // (splat), no written array is touched at a second index (no loop-carried dependence), the body
    // is branch/call-free, and all lanes share one 32/64-bit scalar type. Lane `k` of the vector
    // loop then computes exactly what scalar iteration `base+k` would, so the result is identical.
    // Distinct array parameters are assumed not to alias (the usual kernel ABI). On any failure it
    // returns `false` and the caller falls back to the scalar lowering.

    /// If `e` is `arr[j]` — a single-segment array path indexed by exactly the loop variable `j`
    /// (unit stride, zero offset) — return the array's symbol. The shape the vmath kernel needs.
    fn index_by_loopvar(&self, e: &Expr, j: Symbol) -> Option<Symbol> {
        self.index_off(e, j, None)
    }

    /// Like [`index_by_loopvar`], but for a **batched** row normalization the data index is
    /// `row*cols + j` (row-major), where `batch = Some((row, cols))` carries the outer row variable
    /// and the per-row width. With `batch = None` it is exactly `base[j]` (the single-row case). The
    /// `row*cols` term may be written either factor order. Returns the base array symbol. Pure.
    fn index_off(&self, e: &Expr, j: Symbol, batch: Option<(Symbol, &Expr)>) -> Option<Symbol> {
        let ExprKind::Index { base, indices } = &e.kind else {
            return None;
        };
        if indices.len() != 1 {
            return None;
        }
        let idx = &indices[0];
        match batch {
            None => {
                if single_path(idx) != Some(j) {
                    return None;
                }
            }
            Some((row, cols)) => {
                // `row*cols + j` / `j + row*cols` (the additive split of a row-major flat index).
                let ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } = &idx.kind
                else {
                    return None;
                };
                let is_row_off = |me: &Self, e: &Expr| me.is_mul_of(e, row, cols);
                let ok = (is_row_off(self, lhs) && single_path(rhs) == Some(j))
                    || (is_row_off(self, rhs) && single_path(lhs) == Some(j));
                if !ok {
                    return None;
                }
            }
        }
        single_path(base)
    }

    /// Is `e` the product `row * cols` (either factor order) — the row base offset of a flat
    /// `[rows, cols]` index? `row` is matched by symbol, `cols` structurally (it is the loop bound). Pure.
    fn is_mul_of(&self, e: &Expr, row: Symbol, cols: &Expr) -> bool {
        matches!(&e.kind,
            ExprKind::Binary { op: ast::BinOp::Mul, lhs, rhs }
                if (single_path(lhs) == Some(row) && exprs_struct_eq(rhs, cols))
                    || (single_path(rhs) == Some(row) && exprs_struct_eq(lhs, cols)))
    }

    /// Match one statement `out[j] = f(x[j])` for a supported unary intrinsic `f` (exp/log/tanh/
    /// sigmoid/silu/gelu) over `f32` arrays, returning `(out_array, x_array, op_code)`. Pure.
    fn match_vmath_stmt(
        &self,
        stmt: &Stmt,
        j: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<(Symbol, Symbol, u32, MirType)> {
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        let out_sym = self.index_off(target, j, batch)?;
        // Gated SiLU / swish written as a product — `out[j] = x[j] * sigmoid(x[j])` — the textbook
        // definition a programmer writes before reaching for the `silu()` intrinsic (and the value==gate
        // case of a SwiGLU gate). Dispatch to the existing 256-bit `VMATH_SILU` kernel, which *is*
        // `x·sigmoid(x)` (`vmath::silu1`), so it is bit-identical to the inlined `emit_silu` the generic
        // 128-bit vectorizer would otherwise lower it to. `tests/run/activations.mer` writes this form.
        if let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &value.kind
        {
            if self.expr_mir(value) == MirType::F32 {
                if let Some(x_sym) = self
                    .match_gated_silu(lhs, rhs, j, batch)
                    .or_else(|| self.match_gated_silu(rhs, lhs, j, batch))
                {
                    return Some((out_sym, x_sym, VMATH_SILU, MirType::F32));
                }
            }
        }
        let ExprKind::Call { callee, args, .. } = &value.kind else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }
        let opcode = match self.vectorizable_intrinsic(callee) {
            Some(MathIntrinsic::Exp) => VMATH_EXP,
            Some(MathIntrinsic::Log) => VMATH_LOG,
            Some(MathIntrinsic::Tanh) => VMATH_TANH,
            Some(MathIntrinsic::Sigmoid) => VMATH_SIGMOID,
            Some(MathIntrinsic::Silu) => VMATH_SILU,
            Some(MathIntrinsic::Gelu) => VMATH_GELU,
            Some(MathIntrinsic::Elu) => VMATH_ELU,
            Some(MathIntrinsic::LeakyRelu) => VMATH_LEAKYRELU,
            Some(MathIntrinsic::Softplus) => VMATH_SOFTPLUS,
            Some(MathIntrinsic::Mish) => VMATH_MISH,
            Some(MathIntrinsic::Selu) => VMATH_SELU,
            Some(MathIntrinsic::Tanhshrink) => VMATH_TANHSHRINK,
            Some(MathIntrinsic::HardSigmoid) => VMATH_HARDSIGMOID,
            Some(MathIntrinsic::HardSwish) => VMATH_HARDSWISH,
            // sin/cos (RoPE) and erf (exact GELU): C/Rust call scalar libm sinf/cosf/erff and cannot
            // vectorize a loop with the call, so the 256-bit kernel is a clean compute-bound win. The
            // kernel mirrors the inlined `emit_trig_f32`/`emit_erf_f32`, so a dispatched loop and a
            // composed expression agree.
            Some(MathIntrinsic::Sin) => VMATH_SIN,
            Some(MathIntrinsic::Cos) => VMATH_COS,
            Some(MathIntrinsic::Erf) => VMATH_ERF,
            // Comprehensive vectorized elementwise math: base-2 exp/log (FlashAttention-2 base-2
            // softmax, quantization bit-width, entropy in bits) and hyperbolic sinh/cosh — all compose
            // the shared ≈1-ULP exp/log, all scalar in C/Rust libm (no vectorized call), so all win.
            Some(MathIntrinsic::Exp2) => VMATH_EXP2,
            Some(MathIntrinsic::Log2) => VMATH_LOG2,
            Some(MathIntrinsic::Sinh) => VMATH_SINH,
            Some(MathIntrinsic::Cosh) => VMATH_COSH,
            Some(MathIntrinsic::Asinh) => VMATH_ASINH,
            Some(MathIntrinsic::Acosh) => VMATH_ACOSH,
            Some(MathIntrinsic::Atanh) => VMATH_ATANH,
            Some(MathIntrinsic::Atan) => VMATH_ATAN,
            Some(MathIntrinsic::Expm1) => VMATH_EXPM1,
            Some(MathIntrinsic::Log1p) => VMATH_LOG1P,
            Some(MathIntrinsic::Exp10) => VMATH_EXP10,
            Some(MathIntrinsic::Log10) => VMATH_LOG10,
            // softsign (bounded poly activation) and logsigmoid (stable log-sigmoid for
            // BCE-with-logits / contrastive losses): C/Rust compute these as scalar libm
            // (logsigmoid has no libm entry at all — it's two scalar calls), so the 256-bit
            // dispatch wins; both compose existing kernels, so dispatched == composed.
            Some(MathIntrinsic::Softsign) => VMATH_SOFTSIGN,
            Some(MathIntrinsic::LogSigmoid) => VMATH_LOGSIGMOID,
            // tan/asin/acos complete the trig family (sin/cos/atan): geometry, 3D vision, graphics
            // ML (rotations, NeRF/SLAM angles). C/Rust call scalar libm tanf/asinf/acosf — a loop with
            // the call won't vectorize — and these compose the shared sin/cos/atan, so dispatched ==
            // composed and the 256-bit kernel wins.
            Some(MathIntrinsic::Tan) => VMATH_TAN,
            Some(MathIntrinsic::Asin) => VMATH_ASIN,
            Some(MathIntrinsic::Acos) => VMATH_ACOS,
            // cbrt completes the root family (sqrt/rsqrt/cbrt): LAB color, variance-stabilizing
            // transforms. C's `cbrtf` is scalar libm; composes the shared exp/log so dispatched ==
            // composed.
            Some(MathIntrinsic::Cbrt) => VMATH_CBRT,
            _ => return None,
        };
        // The kernel computes (and writes) f32, so the activation's result must be f32.
        if self.expr_mir(value) != MirType::F32 {
            return None;
        }
        let arg = &args[0];
        // bf16/f16-input form `f((x[j] as f32))`: x a `[bf16]`/`[f16]` array, widened losslessly, out
        // f32. Dispatch to `mercury_vmath_{bf16,f16}` (half the input bytes). Require the out array f32
        // (the kernel stores f32 — never let an f32 store land in a 2-byte slot).
        if let ExprKind::Cast { expr: inner, .. } = &arg.kind {
            let in_elem = match scalar_of(inner, self.sema) {
                Some(mercury_types::Scalar::Bf16) => MirType::BF16,
                Some(mercury_types::Scalar::F16) => MirType::F16,
                _ => return None,
            };
            if self.expr_mir(target) == MirType::F32
                && scalar_of(arg, self.sema) == Some(mercury_types::Scalar::F32)
            {
                let x_sym = self.index_off(inner, j, batch)?;
                return Some((out_sym, x_sym, opcode, in_elem));
            }
            return None;
        }
        // f32 form `f(x[j])`: bail on f64 / unknown.
        if self.expr_mir(arg) != MirType::F32 {
            return None;
        }
        let x_sym = self.index_off(arg, j, batch)?;
        Some((out_sym, x_sym, opcode, MirType::F32))
    }

    /// One ordering of the gated-SiLU product `out[j] = x[j] * sigmoid(x[j])`: `val` must be the
    /// unit-stride f32 read `x[j]` and `gate` the call `sigmoid(x[j])` over the *same* array. Returns
    /// that array's symbol, or `None`. Pure — used by [`match_vmath_stmt`] to fold the product form
    /// into the `VMATH_SILU` dispatch (`silu(x) == x·sigmoid(x)`, bit-identical to the inlined form).
    fn match_gated_silu(
        &self,
        val: &Expr,
        gate: &Expr,
        j: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<Symbol> {
        let xs = self.index_off(val, j, batch)?;
        if self.expr_mir(val) != MirType::F32 {
            return None;
        }
        let ExprKind::Call { callee, args, .. } = &gate.kind else {
            return None;
        };
        if args.len() != 1 || !matches!(self.vectorizable_intrinsic(callee), Some(MathIntrinsic::Sigmoid))
        {
            return None;
        }
        let xs2 = self.index_off(&args[0], j, batch)?;
        (xs == xs2).then_some(xs)
    }

    /// Match a transcendental-activation loop body: every statement must be an independent
    /// `out[j] = f(x[j])` over f32 (see [`match_vmath_stmt`]), with every array in scope. Returns the
    /// resolved `(out_base, x_base, op)` per statement, or `None` if any statement fails. Pure — emits
    /// no MIR — so it is safe to call before deciding whether to lower the loop bounds.
    fn match_vmath_body(
        &self,
        j: Symbol,
        body: &Block,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<Vec<(Symbol, Symbol, u32, MirType)>> {
        if body.tail.is_some() || body.stmts.is_empty() {
            return None;
        }
        let mut calls = Vec::with_capacity(body.stmts.len());
        for stmt in &body.stmts {
            let (out_sym, x_sym, opcode, in_elem) = self.match_vmath_stmt(stmt, j, batch)?;
            // Keep the operand *symbols* (validated in scope); the base pointer is resolved at emit
            // time via `kernel_base_ptr`, which loads it out of a `Tensor[..]`/pointer param's slot.
            self.lookup(out_sym)?;
            self.lookup(x_sym)?;
            calls.push((out_sym, x_sym, opcode, in_elem));
        }
        Some(calls)
    }

    /// Emit one `mercury_vmath_f32(x+s, out+s, e-s, op)` call per resolved statement over the i64
    /// range `[s, e)`. Each is a full-range pass; in source order they preserve a fused multi-statement
    /// body's per-element semantics (an array is fully written before a later pass reads it).
    fn emit_vmath_calls(
        &mut self,
        s: ValueId,
        e: ValueId,
        calls: Vec<(Symbol, Symbol, u32, MirType)>,
    ) {
        let n = self.builder.build(MirType::I64, Op::Bin(BinOp::Sub, e, s));
        for (out_sym, x_sym, opcode, in_elem) in calls {
            // Resolve each operand's base pointer (a `Load` out of a `Tensor[..]`/pointer param's
            // slot, a no-op for an array). The matcher validated every symbol.
            let out_base = self
                .kernel_base_ptr(out_sym)
                .expect("vmath `out` operand validated in matcher");
            let x_base = self
                .kernel_base_ptr(x_sym)
                .expect("vmath `x` operand validated in matcher");
            // The input strides by its element width (f32/bf16/f16) through the matching kernel; the
            // output is always f32.
            let func = match in_elem {
                MirType::BF16 => self.gemm.vmath_bf16,
                MirType::F16 => self.gemm.vmath_f16,
                _ => self.gemm.vmath,
            };
            let xp = self.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: x_base,
                    index: s,
                    elem: in_elem,
                },
            );
            let outp = self.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: out_base,
                    index: s,
                    elem: MirType::F32,
                },
            );
            let opv = self
                .builder
                .build(MirType::I64, Op::ConstInt(opcode as i128, MirType::I64));
            self.builder.build_void(Op::Call {
                func,
                args: vec![xp, outp, n, opv],
            });
        }
    }

    /// Recognize an elementwise transcendental loop `for j in lo..hi { ... }` whose body is one or
    /// more independent `out[j] = f(x[j])` activations (exp/log/tanh/sigmoid/silu/gelu over f32), and
    /// lower it to one `mercury_vmath_f32` call per statement — the 256-bit AVX2 kernel (Cranelift's
    /// vectorizer is capped at 128-bit). `out` and `x` may be the same array (in-place). The
    /// interpreter marshals through the identical kernel, so the differential oracle stays exact.
    /// Returns false (fall back to the generic vectorizer) unless every statement matches.
    fn try_vmath_for(&mut self, j: Symbol, start: &Expr, end: &Expr, body: &Block) -> bool {
        let Some(calls) = self.match_vmath_body(j, body, None) else {
            return false;
        };
        // Bounds GEP each base by the start index, so a loop from `lo` begins at element `lo`.
        let sty = self.expr_mir(start);
        let s = self.lower_expr(start);
        let s = self.coerce_to(s, &sty, &MirType::I64, true);
        let ety = self.expr_mir(end);
        let e = self.lower_expr(end);
        let e = self.coerce_to(e, &ety, &MirType::I64, true);
        self.emit_vmath_calls(s, e, calls);
        true
    }

    /// Recognize a **batched** transcendental-activation nest `for r in 0..R { for j in 0..C { out[r*C
    /// + j] = f(x[r*C + j]) } }` — the real `[tokens, hidden]` FFN/attention activation shape — and
    /// lower it to a single flat 256-bit `mercury_vmath_f32` call over the whole `[0, R*C)` buffer. A
    /// per-element activation over a contiguous `[R, C]` matrix *is* one flat activation of `R*C`
    /// elements, so the row structure is irrelevant to the kernel — one call covers it, at true 256-bit
    /// width (the generic vectorizer that would otherwise lower the offset-indexed inner loop is capped
    /// at 128-bit SSE, so the batched shape ran at half the width of the flat `for i in 0..N` form). The
    /// kernel mirrors the inlined poly, so dispatched == the generic-vectorized form bit-for-bit, and
    /// the interpreter marshals the same kernel — the differential gate stays exact. Returns false (fall
    /// through) unless the nest matches exactly; a partial match leaves the generic vectorizer to lower
    /// the loops correctly. Both `r` and `j` must start at 0 so the touched indices are exactly the
    /// contiguous `[0, R*C)` (a non-zero inner start would leave per-row gaps the flat call can't model).
    fn try_emit_batched_vmath(&mut self, pat: &Pattern, iter: &ForIter, body: &Block) -> bool {
        let ForIter::Range {
            start: r_start,
            end: Some(r_end),
            inclusive: false,
            step: None,
        } = iter
        else {
            return false;
        };
        if const_usize_expr(r_start, self.interner) != Some(0) {
            return false;
        }
        let Pattern {
            kind: ast::PatKind::Ident(r),
            ..
        } = pat
        else {
            return false;
        };
        // The outer body must be exactly one inner `for j in 0..C { <vmath stmts over x[r*C + j]> }`.
        if body.tail.is_some() || body.stmts.len() != 1 {
            return false;
        }
        let StmtKind::For {
            pat: jpat,
            iter: jiter,
            body: inner,
            ..
        } = &body.stmts[0].kind
        else {
            return false;
        };
        let ForIter::Range {
            start: j_start,
            end: Some(cols),
            inclusive: false,
            step: None,
        } = jiter
        else {
            return false;
        };
        if const_usize_expr(j_start, self.interner) != Some(0) {
            return false;
        }
        let Pattern {
            kind: ast::PatKind::Ident(jvar),
            ..
        } = jpat
        else {
            return false;
        };
        let Some(calls) = self.match_vmath_body(*jvar, inner, Some((*r, cols))) else {
            return false;
        };
        // Flat range `[0, R*C)`: each base GEPs from element 0, length `R*C`.
        let rty = self.expr_mir(r_end);
        let rv = self.lower_expr(r_end);
        let rv = self.coerce_to(rv, &rty, &MirType::I64, true);
        let cty = self.expr_mir(cols);
        let cv = self.lower_expr(cols);
        let cv = self.coerce_to(cv, &cty, &MirType::I64, true);
        let total = self.builder.build(MirType::I64, Op::Bin(BinOp::Mul, rv, cv));
        let zero = self
            .builder
            .build(MirType::I64, Op::ConstInt(0, MirType::I64));
        self.emit_vmath_calls(zero, total, calls);
        true
    }

    /// Match one statement `out[j] = f(x[j], y[j])` for a two-input transcendental `f`
    /// (`pow`/`atan2`/`hypot`) over `f32` arrays, returning `(out, x, y, op)` — the two operands are
    /// positional (the kernel interprets them per op). Pure. The 256-bit twin of the inlined poly.
    fn match_vmath2_stmt(&self, stmt: &Stmt, j: Symbol) -> Option<(Symbol, Symbol, Symbol, u32)> {
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        let out_sym = self.index_by_loopvar(target, j)?;
        // Gated-FFN activation `out[j] = act(a[j]) * b[j]` (SwiGLU/GeGLU): a Mul of a *one-arg*
        // activation call (silu/gelu) on `a[j]` and a second unit-stride array read `b[j]` (either
        // factor order). Folds to `mercury_vmath2_f32(a, b, out, n, VMATH2_*_GATE)` — the kernel
        // computes act(a)·b, so the activated operand must be passed *first*. Tried before the 2-arg
        // call shape below. The activation folds an exp C/Rust keep scalar, so the 256-bit gate wins.
        if let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &value.kind
        {
            for (act_e, lin_e) in [(lhs.as_ref(), rhs.as_ref()), (rhs.as_ref(), lhs.as_ref())] {
                if let ExprKind::Call { callee, args, .. } = &act_e.kind {
                    if args.len() == 1 {
                        let gop = match self.vectorizable_intrinsic(callee) {
                            Some(MathIntrinsic::Silu) => Some(VMATH2_SILU_GATE),
                            Some(MathIntrinsic::Gelu) => Some(VMATH2_GELU_GATE),
                            _ => None,
                        };
                        if let Some(gop) = gop {
                            if self.expr_mir(value) == MirType::F32
                                && self.expr_mir(&args[0]) == MirType::F32
                                && self.expr_mir(lin_e) == MirType::F32
                            {
                                if let (Some(a), Some(b)) = (
                                    self.index_by_loopvar(&args[0], j),
                                    self.index_by_loopvar(lin_e, j),
                                ) {
                                    return Some((out_sym, a, b, gop));
                                }
                            }
                        }
                    }
                }
            }
        }
        let ExprKind::Call { callee, args, .. } = &value.kind else {
            return None;
        };
        if args.len() != 2 {
            return None;
        }
        let opcode = match self.vectorizable_intrinsic(callee) {
            Some(MathIntrinsic::Pow) => VMATH2_POW,
            Some(MathIntrinsic::Atan2) => VMATH2_ATAN2,
            Some(MathIntrinsic::Hypot) => VMATH2_HYPOT,
            // Activation backward `dx[j] = act_backward(x[j], dy[j])` = `dy·act'(x)`: the training
            // gradient through SiLU/GELU. The derivative folds a sigmoid/tanh (an `expf`) C/Rust keep
            // scalar, so the 256-bit fused kernel wins like the forward activation dispatch.
            Some(MathIntrinsic::SiluBackward) => VMATH2_SILU_BWD,
            Some(MathIntrinsic::GeluBackward) => VMATH2_GELU_BWD,
            Some(MathIntrinsic::SigmoidBackward) => VMATH2_SIGMOID_BWD,
            Some(MathIntrinsic::TanhBackward) => VMATH2_TANH_BWD,
            Some(MathIntrinsic::EluBackward) => VMATH2_ELU_BWD,
            Some(MathIntrinsic::SoftplusBackward) => VMATH2_SOFTPLUS_BWD,
            _ => return None,
        };
        // The kernel computes (and writes) f32; both operands must be unit-stride f32 array reads.
        if self.expr_mir(value) != MirType::F32
            || self.expr_mir(&args[0]) != MirType::F32
            || self.expr_mir(&args[1]) != MirType::F32
        {
            return None;
        }
        let x_sym = self.index_by_loopvar(&args[0], j)?;
        let y_sym = self.index_by_loopvar(&args[1], j)?;
        Some((out_sym, x_sym, y_sym, opcode))
    }

    /// Match a two-input transcendental loop body: every statement is an independent
    /// `out[j] = f(x[j], y[j])` (see [`match_vmath2_stmt`]). Returns the resolved bases per statement,
    /// or `None` if any fails. Pure — emits no MIR.
    fn match_vmath2_body(
        &self,
        j: Symbol,
        body: &Block,
    ) -> Option<Vec<(Symbol, Symbol, Symbol, u32)>> {
        if body.tail.is_some() || body.stmts.is_empty() {
            return None;
        }
        let mut calls = Vec::with_capacity(body.stmts.len());
        for stmt in &body.stmts {
            let (out_sym, x_sym, y_sym, opcode) = self.match_vmath2_stmt(stmt, j)?;
            // Keep the operand symbols; the base pointer is resolved (loaded from a tensor slot) at
            // emit time via `kernel_base_ptr`.
            self.lookup(out_sym)?;
            self.lookup(x_sym)?;
            self.lookup(y_sym)?;
            calls.push((out_sym, x_sym, y_sym, opcode));
        }
        Some(calls)
    }

    /// Emit one `mercury_vmath2_f32(x+s, y+s, out+s, e-s, op)` call per resolved statement over `[s, e)`.
    fn emit_vmath2_calls(
        &mut self,
        s: ValueId,
        e: ValueId,
        calls: Vec<(Symbol, Symbol, Symbol, u32)>,
    ) {
        let n = self.builder.build(MirType::I64, Op::Bin(BinOp::Sub, e, s));
        for (out_sym, x_sym, y_sym, opcode) in calls {
            let out_base = self
                .kernel_base_ptr(out_sym)
                .expect("vmath2 `out` operand validated in matcher");
            let x_base = self
                .kernel_base_ptr(x_sym)
                .expect("vmath2 `x` operand validated in matcher");
            let y_base = self
                .kernel_base_ptr(y_sym)
                .expect("vmath2 `y` operand validated in matcher");
            let gep = |this: &mut Self, base: ValueId, elem: MirType| {
                this.builder.build(
                    MirType::Ptr,
                    Op::Gep {
                        ptr: base,
                        index: s,
                        elem,
                    },
                )
            };
            let xp = gep(self, x_base, MirType::F32);
            let yp = gep(self, y_base, MirType::F32);
            let outp = gep(self, out_base, MirType::F32);
            let opv = self
                .builder
                .build(MirType::I64, Op::ConstInt(opcode as i128, MirType::I64));
            let func = self.gemm.vmath2;
            self.builder.build_void(Op::Call {
                func,
                args: vec![xp, yp, outp, n, opv],
            });
        }
    }

    /// Recognize a two-input transcendental loop `for j in lo..hi { out[j] = f(x[j], y[j]) }`
    /// (`pow`/`atan2`/`hypot`) and lower it to one `mercury_vmath2_f32` call per statement — the 256-bit
    /// AVX2 kernel, vs the generic vectorizer's inlined 128-bit poly. The interpreter marshals through
    /// the identical kernel, so the differential oracle stays exact. Returns false to fall back.
    fn try_vmath2_for(&mut self, j: Symbol, start: &Expr, end: &Expr, body: &Block) -> bool {
        let Some(calls) = self.match_vmath2_body(j, body) else {
            return false;
        };
        let sty = self.expr_mir(start);
        let s = self.lower_expr(start);
        let s = self.coerce_to(s, &sty, &MirType::I64, true);
        let ety = self.expr_mir(end);
        let e = self.lower_expr(end);
        let e = self.coerce_to(e, &ety, &MirType::I64, true);
        self.emit_vmath2_calls(s, e, calls);
        true
    }

    /// A loop-invariant f32 coefficient: an expr provably free of the loop var `j` and typed f32 (a
    /// literal like `2.0`, or an outer scalar like saxpy's `a`). Lowered to a ValueId at emit time.
    /// Uses the **conservative** [`expr_mentions`] (recurses through calls/casts/fields and assumes a
    /// use for anything it cannot model), so a per-element factor like `sigmoid(x[j])` is correctly
    /// rejected rather than mistaken for an invariant scale. Pure.
    fn velem_coeff<'b>(&self, e: &'b Expr, j: Symbol) -> Option<&'b Expr> {
        if expr_mentions(e, j) || self.expr_mir(e) != MirType::F32 {
            return None;
        }
        Some(e)
    }

    /// Classify one additive term of a streaming affine body over loop var `j`: a (possibly scaled)
    /// unit-stride f32 array read `arr[j]`, or a loop-invariant f32 scalar (bias). Pure.
    fn velem_term<'b>(&self, e: &'b Expr, j: Symbol) -> Option<VTerm<'b>> {
        if let Some(arr) = self.index_by_loopvar(e, j) {
            if self.expr_mir(e) != MirType::F32 {
                return None;
            }
            return Some(VTerm::Scaled(arr, None));
        }
        if let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &e.kind
        {
            if let Some(arr) = self.index_by_loopvar(lhs, j) {
                if self.expr_mir(lhs) == MirType::F32 {
                    if let Some(s) = self.velem_coeff(rhs, j) {
                        return Some(VTerm::Scaled(arr, Some(s)));
                    }
                }
            }
            if let Some(arr) = self.index_by_loopvar(rhs, j) {
                if self.expr_mir(rhs) == MirType::F32 {
                    if let Some(s) = self.velem_coeff(lhs, j) {
                        return Some(VTerm::Scaled(arr, Some(s)));
                    }
                }
            }
            return None;
        }
        self.velem_coeff(e, j).map(VTerm::Const)
    }

    /// Recognize a streaming affine+activation map body — one statement `out[j] = act(a·x[j] (+
    /// b·y[j]) + c)` whose RHS is a sum of (scaled) f32 array reads plus an optional invariant bias —
    /// covering saxpy / scale / residual-add / bias / copy. Up to two distinct array reads (`x`, then
    /// `y`) and one bias. Pure (emits no MIR): returns the resolved bases + coefficient exprs, or
    /// `None` to fall through to ReLU recognition / the generic vectorizer. The activation `act` is
    /// supplied by the caller (`VE_ID` for the bare arithmetic forms; ReLU/ReLU6 peel their wrapper
    /// first and pass the inner affine `value` here with `act = VE_RELU`/`VE_RELU6`).
    fn match_velem_affine<'b>(
        &self,
        j: Symbol,
        value: &'b Expr,
        target: &Expr,
        act: i64,
    ) -> Option<VElemPlan<'b>> {
        let out_sym = self.index_by_loopvar(target, j)?;
        if self.expr_mir(value) != MirType::F32 {
            return None;
        }
        let mut terms = Vec::new();
        flatten_add_terms(value, &mut terms);
        let mut arrays: Vec<(Symbol, Option<&Expr>)> = Vec::new();
        let mut consts: Vec<&Expr> = Vec::new();
        for t in &terms {
            match self.velem_term(t, j)? {
                VTerm::Scaled(arr, s) => arrays.push((arr, s)),
                VTerm::Const(s) => consts.push(s),
            }
        }
        // Must read at least one array (else it is not a map over x); at most two; at most one bias.
        if arrays.is_empty() || arrays.len() > 2 || consts.len() > 1 {
            return None;
        }
        let (x_sym, a) = arrays[0];
        self.lookup(x_sym)?; // validate the array/tensor is a bound local/param
        let (y, b, op_y) = if arrays.len() == 2 {
            let (y_sym, b) = arrays[1];
            self.lookup(y_sym)?;
            (Some(y_sym), b, VE_USE_Y)
        } else {
            (None, None, 0)
        };
        self.lookup(out_sym)?;
        Some(VElemPlan {
            out: out_sym,
            x: x_sym,
            y,
            a,
            b,
            c: consts.first().copied(),
            op: act | op_y,
        })
    }

    /// Recognize a streaming elementwise **binary** map — one statement `out[j] = x[j] op y[j]` for
    /// `op ∈ {*, /}`: the **Hadamard product** (gating, attention masks, residual scaling, RoPE) and
    /// the elementwise **quotient** (normalize-by-per-element-scale). Both operands are unit-stride f32
    /// reads of the loop var, so the affine matcher (which requires one factor to be a loop-invariant
    /// coefficient) declines them — they would otherwise drop to Cranelift's 128-bit vectorizer.
    /// Dispatches to `mercury_velem_f32` with `VE_HADAMARD`/`VE_DIV` (256-bit + non-temporal stores).
    /// `act` is supplied by the caller (`VE_ID`, or a peeled ReLU/ReLU6). `x` may equal `y`
    /// (`x[j]*x[j]` = square). Pure. The interpreter marshals the identical kernel, so it stays exact.
    fn match_velem_binary<'b>(
        &self,
        j: Symbol,
        value: &'b Expr,
        target: &Expr,
        act: i64,
    ) -> Option<VElemPlan<'b>> {
        let out_sym = self.index_by_loopvar(target, j)?;
        if self.expr_mir(value) != MirType::F32 {
            return None;
        }
        let ExprKind::Binary { op, lhs, rhs } = &value.kind else {
            return None;
        };
        let bin = match op {
            ast::BinOp::Mul => VE_HADAMARD,
            ast::BinOp::Div => VE_DIV,
            _ => return None,
        };
        // Both sides must be unit-stride f32 reads `x[j]`/`y[j]` (a coefficient·array form is the
        // affine matcher's job and is tried first, so only a both-mention-`j` product reaches here).
        let x_sym = self.index_by_loopvar(lhs, j)?;
        let y_sym = self.index_by_loopvar(rhs, j)?;
        if self.expr_mir(lhs) != MirType::F32 || self.expr_mir(rhs) != MirType::F32 {
            return None;
        }
        self.lookup(out_sym)?;
        self.lookup(x_sym)?;
        self.lookup(y_sym)?;
        Some(VElemPlan {
            out: out_sym,
            x: x_sym,
            y: Some(y_sym),
            a: None,
            b: None,
            c: None,
            op: act | bin | VE_USE_Y,
        })
    }

    /// Peel a ReLU / ReLU6 activation wrapper off `value`, returning the inner (affine) expr and the
    /// `VE_RELU`/`VE_RELU6` code. ReLU is the idiomatic `if INNER > 0.0 { INNER } else { 0.0 }`
    /// (`max(INNER, 0)`); ReLU6 is `if INNER < 6.0 { <ReLU of INNER> } else { 6.0 }` (`clamp(INNER,
    /// 0, 6)`), where the outer guard compares the *same* `INNER`. Pure.
    fn peel_velem_act<'b>(&self, value: &'b Expr) -> Option<(&'b Expr, i64)> {
        let ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } = &value.kind
        else {
            return None;
        };
        let ExprKind::Binary { op, lhs, rhs } = &cond.kind else {
            return None;
        };
        let else_e = branch_value(else_branch.as_deref()?)?;
        let then_e = block_value(then_branch)?;
        let is_lit = |e: &Expr, v: f32| matches!(&e.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) as f32 == v);
        match op {
            // ReLU: `if INNER > 0 { INNER } else { 0 }` — the then-branch returns the compared INNER.
            ast::BinOp::Gt
                if is_lit(rhs, 0.0) && is_lit(else_e, 0.0) && exprs_struct_eq(then_e, lhs) =>
            {
                Some((lhs, VE_RELU))
            }
            // ReLU6: `if INNER < 6 { ReLU(INNER) } else { 6 }` — the then-branch is a ReLU whose inner
            // is the same INNER the guard compares.
            ast::BinOp::Lt if is_lit(rhs, 6.0) && is_lit(else_e, 6.0) => {
                let (inner, inner_act) = self.peel_velem_act(then_e)?;
                if inner_act == VE_RELU && exprs_struct_eq(inner, lhs) {
                    Some((inner, VE_RELU6))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Recognize the whole body of a streaming elementwise map: a single `out[j] = …` assignment whose
    /// RHS is either a bare affine form (saxpy / scale / add / bias / copy) or a ReLU/ReLU6-wrapped
    /// one. Pure (emits no MIR).
    fn match_velem_body<'b>(&self, j: Symbol, body: &'b Block) -> Option<VElemPlan<'b>> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if let Some(plan) = self.match_velem_affine(j, value, target, VE_ID) {
            return Some(plan);
        }
        if let Some(plan) = self.match_velem_binary(j, value, target, VE_ID) {
            return Some(plan);
        }
        let (inner, act) = self.peel_velem_act(value)?;
        if let Some(plan) = self.match_velem_affine(j, inner, target, act) {
            return Some(plan);
        }
        self.match_velem_binary(j, inner, target, act)
    }

    /// Lower a coefficient expr (or a default constant when absent) to an f32 ValueId.
    fn lower_coeff(&mut self, e: Option<&Expr>, default: f64) -> ValueId {
        match e {
            Some(e) => {
                let ty = self.expr_mir(e);
                let v = self.lower_expr(e);
                self.coerce_to(v, &ty, &MirType::F32, true)
            }
            None => self
                .builder
                .build(MirType::F32, Op::ConstFloat(default, MirType::F32)),
        }
    }

    /// Emit one `mercury_velem_f32(x+s, y+s, out+s, e-s, a, b, c, op)` call for a recognized streaming
    /// map over the i64 range `[s, e)`. When `y` is absent the kernel never reads it (`VE_USE_Y`
    /// unset), so the `x` pointer is reused for the unused `y` argument (a valid, never-dereferenced
    /// pointer — no need for a Ptr-typed null const, which is invalid MIR).
    fn emit_velem_call(&mut self, s: ValueId, e: ValueId, plan: &VElemPlan) {
        let n = self.builder.build(MirType::I64, Op::Bin(BinOp::Sub, e, s));
        // Resolve each operand's base pointer (loading it out of the slot for a `Tensor[..]`/pointer
        // param; a no-op for an array operand). The matcher validated every symbol, so this resolves.
        let x_base = self
            .kernel_base_ptr(plan.x)
            .expect("velem `x` operand validated in matcher");
        let y_base = plan
            .y
            .map(|y| self.kernel_base_ptr(y).expect("velem `y` operand validated in matcher"));
        let out_base = self
            .kernel_base_ptr(plan.out)
            .expect("velem `out` operand validated in matcher");
        let gep = |me: &mut Self, base: ValueId| {
            me.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: base,
                    index: s,
                    elem: MirType::F32,
                },
            )
        };
        let xp = gep(self, x_base);
        let yp = match y_base {
            Some(y) => gep(self, y),
            None => xp,
        };
        let outp = gep(self, out_base);
        // Default `b` is 1.0 when `y` is read, else 0.0 (unused); `a` defaults to 1.0, `c` to 0.0.
        let a = self.lower_coeff(plan.a, 1.0);
        let b = self.lower_coeff(plan.b, if plan.y.is_some() { 1.0 } else { 0.0 });
        let c = self.lower_coeff(plan.c, 0.0);
        let opv = self
            .builder
            .build(MirType::I64, Op::ConstInt(plan.op as i128, MirType::I64));
        self.builder.build_void(Op::Call {
            func: self.gemm.velem,
            args: vec![xp, yp, outp, n, a, b, c, opv],
        });
    }

    /// Recognize a streaming elementwise map `for j in lo..hi { out[j] = act(a·x[j] (+ b·y[j]) + c) }`
    /// (saxpy / scale / residual-add / bias) and lower it to one `mercury_velem_f32` call — 256-bit
    /// AVX2 + non-temporal stores, the width Cranelift can't reach and a store path gcc won't emit.
    /// `out` may alias `x`/`y` (in-place). The interpreter marshals the identical kernel, so the
    /// differential oracle stays exact. Returns false (fall through) unless the body matches.
    fn try_velem_for(&mut self, j: Symbol, start: &Expr, end: &Expr, body: &Block) -> bool {
        let Some(plan) = self.match_velem_body(j, body) else {
            return false;
        };
        let sty = self.expr_mir(start);
        let s = self.lower_expr(start);
        let s = self.coerce_to(s, &sty, &MirType::I64, true);
        let ety = self.expr_mir(end);
        let e = self.lower_expr(end);
        let e = self.coerce_to(e, &ety, &MirType::I64, true);
        self.emit_velem_call(s, e, &plan);
        true
    }

    /// Match `r = r·v + Ck` (the running Horner step) for accumulator `r` and per-element value `v`,
    /// either factor order and either `Add` operand order. `Ck` must be a loop-invariant f32 (free of
    /// the loop var `j`). Returns the coefficient expr. Pure.
    fn match_horner_step<'b>(
        &self,
        stmt: &'b Stmt,
        r: Symbol,
        v: Symbol,
        j: Symbol,
    ) -> Option<&'b Expr> {
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if single_path(target) != Some(r) {
            return None;
        }
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs,
            rhs,
        } = &value.kind
        else {
            return None;
        };
        let is_rv = |e: &Expr| {
            matches!(&e.kind, ExprKind::Binary { op: ast::BinOp::Mul, lhs, rhs }
                if (single_path(lhs) == Some(r) && single_path(rhs) == Some(v))
                    || (single_path(lhs) == Some(v) && single_path(rhs) == Some(r)))
        };
        let ck = if is_rv(lhs) {
            rhs
        } else if is_rv(rhs) {
            lhs
        } else {
            return None;
        };
        if expr_mentions(ck, j) || self.expr_mir(ck) != MirType::F32 {
            return None;
        }
        Some(ck)
    }

    /// Recognize a per-element Horner polynomial body and return `(out_base, x_base, coeffs)` (highest
    /// degree first). The canonical shape (the one gcc/rustc also vectorize, but only at 128-bit):
    ///
    /// ```text
    /// let v: f32 = x[j];          // the element
    /// let mut r: f32 = C0;        // seed = leading coefficient
    /// r = r * v + C1;             // one or more Horner steps
    /// …
    /// out[j] = r;                 // store
    /// ```
    ///
    /// `v`/`r` are body-local scalars; the coefficients are loop-invariant f32 (the benchmark's are
    /// literals). Pure (emits no MIR). `None` on any deviation (the generic vectorizer then lowers it).
    fn match_vhorner_body<'b>(
        &self,
        j: Symbol,
        body: &'b Block,
    ) -> Option<(ValueId, ValueId, Vec<&'b Expr>)> {
        if body.tail.is_some() || body.stmts.len() < 4 {
            return None;
        }
        let stmts = &body.stmts;
        let n = stmts.len();
        // `let v = x[j]`
        let (v, v_init) = Self::let_init(&stmts[0])?;
        let x_sym = self.index_by_loopvar(v_init, j)?;
        if self.expr_mir(v_init) != MirType::F32 {
            return None;
        }
        // `let mut r = C0` (leading coefficient, loop-invariant f32)
        let (r, c0) = Self::let_init(&stmts[1])?;
        if expr_mentions(c0, j) || self.expr_mir(c0) != MirType::F32 {
            return None;
        }
        let mut coeffs = vec![c0];
        for stmt in &stmts[2..n - 1] {
            coeffs.push(self.match_horner_step(stmt, r, v, j)?);
        }
        // `out[j] = r`
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmts[n - 1].kind
        else {
            return None;
        };
        let out_sym = self.index_by_loopvar(target, j)?;
        if single_path(value) != Some(r) {
            return None;
        }
        Some((self.lookup(out_sym)?.0, self.lookup(x_sym)?.0, coeffs))
    }

    /// Emit one `mercury_vhorner_f32(x+s, out+s, e-s, coeffs, ncoeff)` call: materialize the
    /// coefficient array on the stack (an entry-block alloca + a store per coefficient, lowered from
    /// their loop-invariant exprs), then GEP `x`/`out` by `s` and call. The interpreter marshals the
    /// identical kernel, so the differential oracle stays exact.
    fn emit_vhorner(
        &mut self,
        s: ValueId,
        e: ValueId,
        out_base: ValueId,
        x_base: ValueId,
        coeffs: &[&Expr],
    ) {
        let n = self.builder.build(MirType::I64, Op::Bin(BinOp::Sub, e, s));
        let arr = self
            .builder
            .alloca(MirType::Array(Box::new(MirType::F32), coeffs.len() as u32));
        for (k, ce) in coeffs.iter().enumerate() {
            let idx = self
                .builder
                .build(MirType::I64, Op::ConstInt(k as i128, MirType::I64));
            let p = self.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: arr,
                    index: idx,
                    elem: MirType::F32,
                },
            );
            let cv = self.lower_coeff(Some(ce), 0.0);
            self.builder.build_void(Op::Store { ptr: p, value: cv });
        }
        let gep = |me: &mut Self, base: ValueId| {
            me.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: base,
                    index: s,
                    elem: MirType::F32,
                },
            )
        };
        let xp = gep(self, x_base);
        let outp = gep(self, out_base);
        let ncoeff = self.builder.build(
            MirType::I64,
            Op::ConstInt(coeffs.len() as i128, MirType::I64),
        );
        self.builder.build_void(Op::Call {
            func: self.gemm.vhorner,
            args: vec![xp, outp, n, arr, ncoeff],
        });
    }

    /// Recognize a per-element Horner polynomial loop `for j in lo..hi { let v=x[j]; let mut r=c0; r =
    /// r*v+c1; …; out[j]=r }` and lower it to one `mercury_vhorner_f32` call — 256-bit AVX2 + (for a
    /// large output) non-temporal stores, both beyond Cranelift's 128-bit vectorizer. Returns false
    /// (fall through) unless the body matches.
    fn try_vhorner_for(&mut self, j: Symbol, start: &Expr, end: &Expr, body: &Block) -> bool {
        let Some((out_base, x_base, coeffs)) = self.match_vhorner_body(j, body) else {
            return false;
        };
        let sty = self.expr_mir(start);
        let s = self.lower_expr(start);
        let s = self.coerce_to(s, &sty, &MirType::I64, true);
        let ety = self.expr_mir(end);
        let e = self.lower_expr(end);
        let e = self.coerce_to(e, &ety, &MirType::I64, true);
        self.emit_vhorner(s, e, out_base, x_base, &coeffs);
        true
    }

    /// Attempt SIMD lowering of `for j in start..end { body }`. Returns true on success.
    fn try_vectorize_for(&mut self, pat: &Pattern, start: &Expr, end: &Expr, body: &Block) -> bool {
        let j = match &pat.kind {
            ast::PatKind::Ident(name) => *name,
            _ => return false,
        };
        let ity = self.expr_mir(start);
        if !ity.is_int() {
            return false;
        }
        // A pure elementwise transcendental `out[j] = f(x[j])` (exp/log/tanh/sigmoid) lowers to the
        // 256-bit AVX2 runtime kernel — the width Cranelift's 128-bit vectorizer can't reach. Tried
        // before the generic (Cranelift-emitted) vectorizer, which would otherwise inline the poly.
        if self.try_vmath_for(j, start, end, body) {
            return true;
        }
        // A two-input transcendental map `out[j] = pow/atan2/hypot(x[j], y[j])` dispatches to the
        // 256-bit `mercury_vmath2_f32` kernel (vs the generic vectorizer's inlined 128-bit poly).
        if self.try_vmath2_for(j, start, end, body) {
            return true;
        }
        // A streaming affine map `out[j] = a·x[j] (+ b·y[j]) + c` (saxpy/scale/add/bias) dispatches to
        // the 256-bit AVX2 + non-temporal-store kernel — both wider than and store-cheaper than the
        // generic 128-bit vectorizer. Tried before it (which would otherwise emit cacheable stores).
        if self.try_velem_for(j, start, end, body) {
            return true;
        }
        // A per-element Horner polynomial `let v=x[j]; let mut r=c0; r=r*v+c1; …; out[j]=r` dispatches
        // to the 256-bit AVX2 + non-temporal-store Horner kernel (the generic vectorizer would inline
        // it at 128-bit with cacheable stores).
        if self.try_vhorner_for(j, start, end, body) {
            return true;
        }
        // Pure analyses first (emit no MIR). A reduction (`s += elementwise`) has a body shape
        // disjoint from the elementwise *store* loops, so try it first. Reductions are vectorized
        // only on this (sequential) path — not the `@parallel` per-thread path, where folding into
        // a shared accumulator across threads would race.
        let reduction = self.reduction_of(body, j);
        if reduction.is_none() && self.vectorizable(body, j).is_none() {
            return false;
        }
        // Lower the bounds (coerced to the index type) and hand off to the matching emitter.
        let start_ty = self.expr_mir(start);
        let s0 = self.lower_expr(start);
        let s0 = self.coerce_to(s0, &start_ty, &ity, true);
        let end_ty = self.expr_mir(end);
        let e0 = self.lower_expr(end);
        let e0 = self.coerce_to(e0, &end_ty, &ity, true);
        if let Some((s_sym, addend, lane, w, redop)) = reduction {
            self.emit_reduction(j, s0, e0, &ity, s_sym, addend, &lane, w, redop);
            true
        } else {
            self.try_vectorize_ranged(j, s0, e0, &ity, body)
        }
    }

    /// Recognise a vectorizable float reduction `for j in .. { s = s + <expr(j)> }` (or `s += ..`),
    /// where `s` is an outer float scalar that the body otherwise does not touch and `<expr>` is an
    /// elementwise value over `j` that does not read `s`. Returns `(s, addend, lane, width)`.
    /// Vectorizing this reassociates the float sum (lane accumulators + a horizontal reduce) — the
    /// standard reduction optimization. Pure analysis; emits no MIR.
    fn reduction_of<'b>(
        &self,
        body: &'b Block,
        j: Symbol,
    ) -> Option<(Symbol, &'b Expr, MirType, u32, RedOp)> {
        if body.tail.is_some() || body.stmts.len() != 1 {
            return None;
        }
        let StmtKind::Assign { target, op, value } = &body.stmts[0].kind else {
            return None;
        };
        let s = single_path(target)?;
        if s == j {
            return None;
        }
        let (_, sty) = self.lookup(s)?;
        if !(sty.is_float() || sty.is_int()) {
            return None;
        }
        // The addend and fold kind: `s += addend` / `s = s + addend` / `s = addend + s` (sum), or
        // `m = fmax(m, addend)` / `m = fmin(m, addend)` (running max/min, either operand order).
        let is_s = |e: &Expr| single_path(e) == Some(s);
        let (addend, redop): (&Expr, RedOp) = match op {
            ast::AssignOp::Add => (value, RedOp::Add),
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if is_s(lhs) => (rhs, RedOp::Add),
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if is_s(rhs) => (lhs, RedOp::Add),
                ExprKind::Call { callee, args, .. } if args.len() == 2 => {
                    let red = match self.vectorizable_intrinsic(callee) {
                        Some(MathIntrinsic::Fmax) => RedOp::Fmax,
                        Some(MathIntrinsic::Fmin) => RedOp::Fmin,
                        _ => return None,
                    };
                    if is_s(&args[0]) {
                        (&args[1], red)
                    } else if is_s(&args[1]) {
                        (&args[0], red)
                    } else {
                        return None;
                    }
                }
                _ => return None,
            },
            _ => return None,
        };
        if expr_uses_sym(addend, s) {
            return None;
        }
        let mut lane: Option<MirType> = None;
        let mut acc: Vec<(Symbol, &Expr, bool, bool)> = Vec::new();
        if !self.vec_check_value(addend, j, &HashSet::new(), &mut lane, &mut acc) {
            return None;
        }
        let lane = lane?;
        // Require the lane type to match the accumulator exactly, so the horizontal reduce
        // (`s = s + lane`) is well-typed — bails on a mixed-width reduction (e.g. `f64 += f32`).
        if lane != sty {
            return None;
        }
        // fmax/fmin fold via a float compare + select; they need a float lane.
        if matches!(redop, RedOp::Fmax | RedOp::Fmin) && !lane.is_float() {
            return None;
        }
        let w = vector_width(&lane)?;
        Some((s, addend, lane, w, redop))
    }

    /// Attempt SIMD lowering of a loop whose bounds are already lowered to `ity` values (the form
    /// the `@parallel` outliner produces). Returns true on success.
    fn try_vectorize_ranged(
        &mut self,
        j: Symbol,
        start_val: ValueId,
        end_val: ValueId,
        ity: &MirType,
        body: &Block,
    ) -> bool {
        // A transcendental-activation chunk dispatches to the 256-bit AVX2 kernel here too, so an
        // `@parallel` activation runs multicore × 256-bit (each thread's chunk is one kernel call).
        // Elementwise, so the interpreter's whole-range pass and the native per-chunk passes agree.
        if let Some(calls) = self.match_vmath_body(j, body, None) {
            let s = self.coerce_to(start_val, ity, &MirType::I64, true);
            let e = self.coerce_to(end_val, ity, &MirType::I64, true);
            self.emit_vmath_calls(s, e, calls);
            return true;
        }
        // A streaming affine map per `@parallel` chunk → the same 256-bit + non-temporal-store kernel,
        // so an `@parallel` saxpy/scale/add runs multicore × 256-bit with RFO-free stores. Elementwise,
        // so each thread's per-chunk pass agrees with the interpreter's whole-range pass.
        if let Some(plan) = self.match_velem_body(j, body) {
            let s = self.coerce_to(start_val, ity, &MirType::I64, true);
            let e = self.coerce_to(end_val, ity, &MirType::I64, true);
            self.emit_velem_call(s, e, &plan);
            return true;
        }
        // A bf16/f16→f32 axpby per `@parallel` chunk → `mercury_axpby_{bf16,f16}` over `[s, e)`, so a
        // mixed-precision residual-add / saxpy runs multicore (the bandwidth payoff is largest here,
        // ≫ L3). Each chunk GEPs the half-width inputs and f32 output by `s`; elementwise, so the
        // per-chunk passes agree with the interpreter's whole-range marshal of the same kernel.
        if let Some((out, x, y, a_expr, b_expr, is_f16)) = self.match_lowp_axpby(body, j) {
            if let (Some((outv, _)), Some((xv, _)), Some((yv, _))) =
                (self.lookup(out), self.lookup(x), self.lookup(y))
            {
                let s = self.coerce_to(start_val, ity, &MirType::I64, true);
                let e = self.coerce_to(end_val, ity, &MirType::I64, true);
                let av = self.lower_coeff(a_expr, 1.0);
                let bv = self.lower_coeff(b_expr, 1.0);
                let n = self.builder.build(MirType::I64, Op::Bin(BinOp::Sub, e, s));
                let in_elem = if is_f16 { MirType::F16 } else { MirType::BF16 };
                let xp = self.builder.build(
                    MirType::Ptr,
                    Op::Gep {
                        ptr: xv,
                        index: s,
                        elem: in_elem.clone(),
                    },
                );
                let yp = self.builder.build(
                    MirType::Ptr,
                    Op::Gep {
                        ptr: yv,
                        index: s,
                        elem: in_elem,
                    },
                );
                let outp = self.builder.build(
                    MirType::Ptr,
                    Op::Gep {
                        ptr: outv,
                        index: s,
                        elem: MirType::F32,
                    },
                );
                self.builder.build_void(Op::Call {
                    func: if is_f16 {
                        self.gemm.axpby_f16
                    } else {
                        self.gemm.axpby_bf16
                    },
                    args: vec![xp, yp, outp, n, av, bv],
                });
                return true;
            }
        }
        // A Horner polynomial per `@parallel` chunk → the same 256-bit AVX2 + NT-store Horner kernel.
        if let Some((out_base, x_base, coeffs)) = self.match_vhorner_body(j, body) {
            let s = self.coerce_to(start_val, ity, &MirType::I64, true);
            let e = self.coerce_to(end_val, ity, &MirType::I64, true);
            self.emit_vhorner(s, e, out_base, x_base, &coeffs);
            return true;
        }
        let Some((lane, w)) = self.vectorizable(body, j) else {
            return false;
        };
        self.emit_vectorized_for(j, start_val, end_val, ity, &lane, w, body);
        true
    }

    /// Validate that `body` is vectorizable over loop variable `j`; return the shared lane type and
    /// vector width, or `None` to bail. Pure analysis (emits no MIR).
    fn vectorizable(&self, body: &Block, j: Symbol) -> Option<(MirType, u32)> {
        if body.tail.is_some() || body.stmts.is_empty() {
            return None;
        }
        // inner `let` names become vector temps; they may not appear inside index expressions.
        let mut locals: HashSet<Symbol> = HashSet::new();
        // every array access: (base, index expr, unit-stride?, is_write).
        let mut acc: Vec<(Symbol, &Expr, bool, bool)> = Vec::new();
        let mut lane: Option<MirType> = None;

        for s in &body.stmts {
            match &s.kind {
                StmtKind::Let {
                    pat:
                        Pattern {
                            kind: ast::PatKind::Ident(name),
                            ..
                        },
                    init: Some(e),
                    ..
                } => {
                    if !self.vec_check_value(e, j, &locals, &mut lane, &mut acc) {
                        return None;
                    }
                    locals.insert(*name);
                }
                StmtKind::Assign { target, op, value } => {
                    if !self.vec_check_value(value, j, &locals, &mut lane, &mut acc) {
                        return None;
                    }
                    let _ = op; // compound vs plain handled identically for validation
                    match &target.kind {
                        // scalar reassignment of an inner vector temp (e.g. Horner `r = r*v + c`)
                        ExprKind::Path(p) if p.is_single() && locals.contains(&p.first().sym) => {}
                        // store to an array element at a unit-stride index
                        ExprKind::Index { base, indices } if indices.len() == 1 => {
                            let bsym = single_path(base)?;
                            if uses_any(&indices[0], &locals) {
                                return None; // index must not depend on inner temps
                            }
                            if affine_stride(&indices[0], j) != Some(1) {
                                return None; // stores must be unit-stride
                            }
                            let elem = self.array_elem(base)?;
                            set_or_check(&mut lane, &elem)?;
                            acc.push((bsym, &indices[0], true, true));
                        }
                        _ => return None,
                    }
                }
                _ => return None, // no nested control flow / calls / returns in a vector body
            }
        }

        // No array that is written may be accessed at a second (different) index: that would be a
        // loop-carried dependence the lane-parallel form would break.
        for &(base, idx, _unit, is_w) in &acc {
            if is_w || acc.iter().any(|a| a.0 == base && a.3) {
                // `base` is written somewhere; every access to it must be the identical index.
                if acc
                    .iter()
                    .any(|a| a.0 == base && !exprs_struct_eq(a.1, idx))
                {
                    return None;
                }
            }
        }

        let lane = lane?;
        let w = vector_width(&lane)?;
        Some((lane, w))
    }

    /// Validate that `e` is a vectorizable value-expression (produces lane-typed vectors), recording
    /// any array reads into `acc` and pinning the shared lane type. Returns false to bail.
    /// The math intrinsic a callee names — if it is one and is *not* shadowed by a user function
    /// (mirroring `lower_call`'s precedence). Lets the vectorizer decide which calls it can lower
    /// per lane.
    fn vectorizable_intrinsic(&self, callee: &Expr) -> Option<MathIntrinsic> {
        let ExprKind::Path(p) = &callee.kind else {
            return None;
        };
        if !p.is_single() {
            return None;
        }
        let name = p.first().sym;
        if matches!(
            self.sema.defs.lookup(name).map(|d| &d.kind),
            Some(DefKind::Fn(_))
        ) {
            return None;
        }
        math_intrinsic(self.interner.resolve(name))
    }

    fn vec_check_value<'b>(
        &self,
        e: &'b Expr,
        j: Symbol,
        locals: &HashSet<Symbol>,
        lane: &mut Option<MirType>,
        acc: &mut Vec<(Symbol, &'b Expr, bool, bool)>,
    ) -> bool {
        match &e.kind {
            // an inner vector temp: lane-typed by construction.
            ExprKind::Path(p) if p.is_single() && locals.contains(&p.first().sym) => true,
            // an invariant scalar (param/outer local): must be the lane type, and must not be the
            // loop variable used as a value (that would need an index vector, which we don't form).
            ExprKind::Path(p) if p.is_single() => {
                let t = self.expr_mir(e);
                p.first().sym != j && is_numeric(&t) && set_or_check(lane, &t).is_some()
            }
            // a numeric literal: lane-typed (the splat rounds once, like the scalar path).
            ExprKind::Int(_) | ExprKind::Float(_) => {
                let t = self.expr_mir(e);
                is_numeric(&t) && set_or_check(lane, &t).is_some()
            }
            ExprKind::Index { base, indices } if indices.len() == 1 => {
                if single_path(base).is_none() || uses_any(&indices[0], locals) {
                    return false;
                }
                match affine_stride(&indices[0], j) {
                    Some(0) | Some(1) => {}
                    _ => return false,
                }
                let Some(elem) = self.array_elem(base) else {
                    return false;
                };
                if set_or_check(lane, &elem).is_none() {
                    return false;
                }
                let bsym = single_path(base).unwrap();
                let unit = affine_stride(&indices[0], j) == Some(1);
                acc.push((bsym, &indices[0], unit, false));
                true
            }
            ExprKind::Binary { op, lhs, rhs } => {
                use ast::BinOp::*;
                if !matches!(op, Add | Sub | Mul | Div) {
                    return false;
                }
                self.vec_check_value(lhs, j, locals, lane, acc)
                    && self.vec_check_value(rhs, j, locals, lane, acc)
            }
            ExprKind::Unary {
                op: ast::UnOp::Neg,
                expr,
            } => self.vec_check_value(expr, j, locals, lane, acc),
            // `if cond { a } else { b }` if-converts to a vector compare + blend (e.g. ReLU).
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let (Some(else_e), Some(tv)) = (else_branch.as_deref(), block_value(then_branch))
                else {
                    return false;
                };
                let (Some(ev), ExprKind::Binary { op, lhs, rhs }) =
                    (branch_value(else_e), &cond.kind)
                else {
                    return false;
                };
                use ast::BinOp::*;
                if !matches!(op, Lt | Le | Gt | Ge | Eq | Ne) {
                    return false;
                }
                self.vec_check_value(lhs, j, locals, lane, acc)
                    && self.vec_check_value(rhs, j, locals, lane, acc)
                    && self.vec_check_value(tv, j, locals, lane, acc)
                    && self.vec_check_value(ev, j, locals, lane, acc)
            }
            // Element-wise math intrinsics over vectorizable args. sqrt/rsqrt are one lane op each;
            // fmax/fmin are a lane compare + blend; exp is the f32 polynomial expanded per lane.
            ExprKind::Call { callee, args, .. } => match self.vectorizable_intrinsic(callee) {
                Some(
                    MathIntrinsic::Sqrt
                    | MathIntrinsic::Rsqrt
                    | MathIntrinsic::Abs
                    | MathIntrinsic::Round
                    | MathIntrinsic::Floor
                    | MathIntrinsic::Ceil
                    | MathIntrinsic::Trunc,
                ) => args.len() == 1 && self.vec_check_value(&args[0], j, locals, lane, acc),
                Some(MathIntrinsic::Fmax | MathIntrinsic::Fmin) => {
                    args.len() == 2
                        && self.vec_check_value(&args[0], j, locals, lane, acc)
                        && self.vec_check_value(&args[1], j, locals, lane, acc)
                }
                Some(
                    MathIntrinsic::Exp
                    | MathIntrinsic::Log
                    | MathIntrinsic::Exp2
                    | MathIntrinsic::Log2
                    | MathIntrinsic::Sinh
                    | MathIntrinsic::Cosh
                    | MathIntrinsic::Asinh
                    | MathIntrinsic::Acosh
                    | MathIntrinsic::Atanh
                    | MathIntrinsic::Atan
                    | MathIntrinsic::Expm1
                    | MathIntrinsic::Log1p
                    | MathIntrinsic::Exp10
                    | MathIntrinsic::Log10
                    | MathIntrinsic::Erf
                    | MathIntrinsic::Sin
                    | MathIntrinsic::Cos
                    | MathIntrinsic::Tanh
                    | MathIntrinsic::Sigmoid
                    | MathIntrinsic::Silu
                    | MathIntrinsic::Gelu
                    | MathIntrinsic::Elu
                    | MathIntrinsic::LeakyRelu
                    | MathIntrinsic::Softplus
                    | MathIntrinsic::Mish
                    | MathIntrinsic::Selu
                    | MathIntrinsic::Tanhshrink
                    | MathIntrinsic::HardSigmoid
                    | MathIntrinsic::HardSwish
                    | MathIntrinsic::Softsign
                    | MathIntrinsic::LogSigmoid
                    | MathIntrinsic::Tan
                    | MathIntrinsic::Asin
                    | MathIntrinsic::Acos
                    | MathIntrinsic::Cbrt,
                ) => {
                    // These build on the exp/log polynomials (or, for leaky-relu, the f32 select),
                    // which vectorize only for an f32 lane (their IEEE-754 surgery is f32-specific).
                    args.len() == 1
                        && self.vec_check_value(&args[0], j, locals, lane, acc)
                        && *lane == Some(MirType::F32)
                }
                Some(
                    MathIntrinsic::Pow
                    | MathIntrinsic::Atan2
                    | MathIntrinsic::Hypot
                    | MathIntrinsic::SiluBackward
                    | MathIntrinsic::GeluBackward
                    | MathIntrinsic::SigmoidBackward
                    | MathIntrinsic::TanhBackward
                    | MathIntrinsic::EluBackward
                    | MathIntrinsic::SoftplusBackward,
                ) => {
                    // Two-arg transcendentals (pow = exp(y·log(x)); atan2; hypot; the activation
                    // backwards `dy·act'(x)`); f32 lane only, same reason as exp/log — the IEEE surgery
                    // in the composed polys (sigmoid/tanh) is f32-specific.
                    args.len() == 2
                        && self.vec_check_value(&args[0], j, locals, lane, acc)
                        && self.vec_check_value(&args[1], j, locals, lane, acc)
                        && *lane == Some(MirType::F32)
                }
                None => false,
            },
            _ => false,
        }
    }

    /// The element MIR type of an array-valued base expression, via sema.
    fn array_elem(&self, base: &Expr) -> Option<MirType> {
        match self.expr_ty(base) {
            Ty::Array { elem, .. } => Some(mir_ty(&elem)),
            _ => None,
        }
    }

    /// Emit the vectorized loop as three strips that share one index slot `j`: an unrolled vector
    /// loop (`VEC_UNROLL` independent vector groups per iteration), then a single-vector loop, then
    /// a scalar remainder. The bounds are already-lowered `ity` values.
    #[allow(clippy::too_many_arguments)]
    fn emit_vectorized_for(
        &mut self,
        j: Symbol,
        s0: ValueId,
        end_v: ValueId,
        ity: &MirType,
        lane: &MirType,
        w: u32,
        body: &Block,
    ) {
        let vty = MirType::Vec(Box::new(lane.clone()), w);
        let slot = self.builder.alloca(ity.clone());
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: s0,
        });
        // A scratch slot holding the per-unroll-copy index (`jbase + u*W`), so unit-stride accesses
        // pick up the lane offset with no per-access arithmetic.
        let jtmp = self.builder.alloca(ity.clone());

        self.push_scope();
        self.bind(j, slot, ity.clone());

        self.emit_vector_strip(j, slot, jtmp, end_v, ity, lane, &vty, w, VEC_UNROLL, body);
        self.emit_vector_strip(j, slot, jtmp, end_v, ity, lane, &vty, w, 1, body);
        self.emit_scalar_tail(j, slot, end_v, ity, body);

        self.pop_scope();
    }

    /// One strip-mined vector loop: while a full `unroll * W` block fits, emit `unroll` independent
    /// vector groups (fresh `vlocals` each, so their dependency chains overlap on the FP units),
    /// then advance `j` by `unroll * W`.
    #[allow(clippy::too_many_arguments)]
    fn emit_vector_strip(
        &mut self,
        j: Symbol,
        slot: ValueId,
        jtmp: ValueId,
        end_v: ValueId,
        ity: &MirType,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        unroll: u32,
        body: &Block,
    ) {
        let span = (unroll * w) as i128;
        let off = self
            .builder
            .build(ity.clone(), Op::ConstInt(span - 1, ity.clone()));
        let vlimit = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Sub, end_v, off));

        let hdr = self.builder.new_block();
        let bb = self.builder.new_block();
        let done = self.builder.new_block();
        self.builder.br(hdr, vec![]);

        // header: while j < end - (unroll*W - 1)
        self.builder.switch_to(hdr);
        self.terminated = false;
        self.bind(j, slot, ity.clone());
        let jv = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let c = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, jv, vlimit));
        self.builder.cond_br(c, bb, vec![], done, vec![]);

        // body: `unroll` vector groups at offsets 0, W, 2W, …; then j += unroll*W
        self.builder.switch_to(bb);
        self.terminated = false;
        let jbase = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        for u in 0..unroll {
            let ju = if u == 0 {
                jbase
            } else {
                let o = self
                    .builder
                    .build(ity.clone(), Op::ConstInt((u * w) as i128, ity.clone()));
                self.builder
                    .build(ity.clone(), Op::Bin(BinOp::Add, jbase, o))
            };
            self.builder.build_void(Op::Store {
                ptr: jtmp,
                value: ju,
            });
            self.bind(j, jtmp, ity.clone());
            let mut vlocals: HashMap<Symbol, ValueId> = HashMap::new();
            // Each unroll copy reads different addresses (jbase + u*W), so the load cache is per-copy.
            self.vec_loads.clear();
            for s in &body.stmts {
                self.vec_lower_stmt(s, j, lane, vty, w, &mut vlocals);
            }
        }
        self.bind(j, slot, ity.clone());
        let jc = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let stepc = self
            .builder
            .build(ity.clone(), Op::ConstInt(span, ity.clone()));
        let jn = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, jc, stepc));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: jn,
        });
        self.builder.br(hdr, vec![]);

        self.builder.switch_to(done);
        self.terminated = false;
    }

    /// The scalar remainder: the original body lowered normally, stepping `j` by 1 to `end`.
    fn emit_scalar_tail(
        &mut self,
        j: Symbol,
        slot: ValueId,
        end_v: ValueId,
        ity: &MirType,
        body: &Block,
    ) {
        self.bind(j, slot, ity.clone());
        let hdr = self.builder.new_block();
        let bb = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(hdr, vec![]);

        self.builder.switch_to(hdr);
        self.terminated = false;
        let jr = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let rc = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, jr, end_v));
        self.builder.cond_br(rc, bb, vec![], exit, vec![]);

        self.builder.switch_to(bb);
        self.terminated = false;
        // A vectorized loop body has no break/continue (vectorizability rejects them), so it never
        // needs a label.
        self.loops.push((None, hdr, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            let jc = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
            let one = self
                .builder
                .build(ity.clone(), Op::ConstInt(1, ity.clone()));
            let jn = self
                .builder
                .build(ity.clone(), Op::Bin(BinOp::Add, jc, one));
            self.builder.build_void(Op::Store {
                ptr: slot,
                value: jn,
            });
            self.builder.br(hdr, vec![]);
        }
        self.builder.switch_to(exit);
        self.terminated = false;
    }

    /// Emit a vectorized float reduction over `[s0, end)`: `VEC_UNROLL` independent vector-lane
    /// accumulators summed in an unrolled main loop, a single-vector cleanup loop, then a horizontal
    /// reduce of the lanes into `s`, then a scalar remainder. This reassociates the sum (vs strict
    /// left-to-right), which is sound for a reduction and is what makes it fast — `w` lanes × unroll
    /// independent FMA chains instead of one serial dependency.
    #[allow(clippy::too_many_arguments)]
    fn emit_reduction(
        &mut self,
        j: Symbol,
        s0: ValueId,
        end_v: ValueId,
        ity: &MirType,
        s_sym: Symbol,
        addend: &Expr,
        lane: &MirType,
        w: u32,
        redop: RedOp,
    ) {
        let vty = MirType::Vec(Box::new(lane.clone()), w);
        let slot = self.builder.alloca(ity.clone());
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: s0,
        });
        let jtmp = self.builder.alloca(ity.clone());

        // `VEC_UNROLL` accumulators, each `w` contiguous lanes (an `Array` slot so the vector
        // store/load address `w` real elements), initialised to the fold's identity before the
        // loop: 0 for a sum, -inf for `fmax`, +inf for `fmin` (so empty tail lanes never win).
        let vid = match redop {
            RedOp::Add => {
                let zero = self.const_zero(lane.clone());
                self.builder.build(vty.clone(), Op::Splat(zero))
            }
            RedOp::Fmax => self.splat_const_f(f64::NEG_INFINITY, &vty),
            RedOp::Fmin => self.splat_const_f(f64::INFINITY, &vty),
        };
        let accs: Vec<ValueId> = (0..VEC_UNROLL)
            .map(|_| {
                let a = self
                    .builder
                    .alloca(MirType::Array(Box::new(lane.clone()), w));
                self.builder.build_void(Op::Store { ptr: a, value: vid });
                a
            })
            .collect();

        self.push_scope();
        self.bind(j, slot, ity.clone());

        self.emit_reduction_strip(
            j, slot, jtmp, end_v, ity, lane, &vty, w, VEC_UNROLL, &accs, addend, redop,
        );
        self.emit_reduction_strip(
            j,
            slot,
            jtmp,
            end_v,
            ity,
            lane,
            &vty,
            w,
            1,
            &accs[..1],
            addend,
            redop,
        );

        // Combine the accumulators, then horizontally reduce the lanes into `s` with the same fold
        // (sum / max / min) the loop body used.
        let mut total = self
            .builder
            .build(vty.clone(), Op::Load(accs[0], vty.clone()));
        for &a in &accs[1..] {
            let v = self.builder.build(vty.clone(), Op::Load(a, vty.clone()));
            total = self.reduce_combine_vec(redop, lane, &vty, w, total, v);
        }
        let scratch = self
            .builder
            .alloca(MirType::Array(Box::new(lane.clone()), w));
        self.builder.build_void(Op::Store {
            ptr: scratch,
            value: total,
        });
        let (s_slot, s_ty) = self.lookup(s_sym).expect("reduction var in scope");
        for k in 0..w {
            let idxk = self
                .builder
                .build(MirType::I64, Op::ConstInt(k as i128, MirType::I64));
            let addr = self.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: scratch,
                    index: idxk,
                    elem: lane.clone(),
                },
            );
            let lane_v = self
                .builder
                .build(lane.clone(), Op::Load(addr, lane.clone()));
            let sv = self
                .builder
                .build(s_ty.clone(), Op::Load(s_slot, s_ty.clone()));
            let sum = self.reduce_combine_scalar(redop, &s_ty, sv, lane_v);
            self.builder.build_void(Op::Store {
                ptr: s_slot,
                value: sum,
            });
        }

        self.emit_reduction_tail(j, slot, end_v, ity, s_sym, addend, redop);
        self.pop_scope();
    }

    /// One strip of the reduction main loop: while a full `unroll*W` block fits, fold `unroll`
    /// independent vector groups into `accs[0..unroll]` (separate accumulators so their FMA chains
    /// overlap), then advance `j` by `unroll*W`.
    #[allow(clippy::too_many_arguments)]
    fn emit_reduction_strip(
        &mut self,
        j: Symbol,
        slot: ValueId,
        jtmp: ValueId,
        end_v: ValueId,
        ity: &MirType,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        unroll: u32,
        accs: &[ValueId],
        addend: &Expr,
        redop: RedOp,
    ) {
        let span = (unroll * w) as i128;
        let off = self
            .builder
            .build(ity.clone(), Op::ConstInt(span - 1, ity.clone()));
        let vlimit = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Sub, end_v, off));

        let hdr = self.builder.new_block();
        let bb = self.builder.new_block();
        let done = self.builder.new_block();
        self.builder.br(hdr, vec![]);

        self.builder.switch_to(hdr);
        self.terminated = false;
        self.bind(j, slot, ity.clone());
        let jv = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let c = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, jv, vlimit));
        self.builder.cond_br(c, bb, vec![], done, vec![]);

        self.builder.switch_to(bb);
        self.terminated = false;
        let jbase = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        for u in 0..unroll {
            let ju = if u == 0 {
                jbase
            } else {
                let o = self
                    .builder
                    .build(ity.clone(), Op::ConstInt((u * w) as i128, ity.clone()));
                self.builder
                    .build(ity.clone(), Op::Bin(BinOp::Add, jbase, o))
            };
            self.builder.build_void(Op::Store {
                ptr: jtmp,
                value: ju,
            });
            self.bind(j, jtmp, ity.clone());
            let acc_slot = accs[u as usize];
            let cur = self
                .builder
                .build(vty.clone(), Op::Load(acc_slot, vty.clone()));
            let mut vlocals: HashMap<Symbol, ValueId> = HashMap::new();
            // Per-copy load cache (so `(x[i]-y[i])*(x[i]-y[i])` loads x[i],y[i] once each).
            self.vec_loads.clear();
            let nv = match redop {
                RedOp::Add => self.vec_accumulate(cur, addend, j, lane, vty, w, &mut vlocals),
                RedOp::Fmax | RedOp::Fmin => {
                    let vx = self.vec_lower_value(addend, j, lane, vty, w, &mut vlocals);
                    self.reduce_combine_vec(redop, lane, vty, w, cur, vx)
                }
            };
            self.builder.build_void(Op::Store {
                ptr: acc_slot,
                value: nv,
            });
        }
        self.bind(j, slot, ity.clone());
        let jc = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let stepc = self
            .builder
            .build(ity.clone(), Op::ConstInt(span, ity.clone()));
        let jn = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, jc, stepc));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: jn,
        });
        self.builder.br(hdr, vec![]);

        self.builder.switch_to(done);
        self.terminated = false;
    }

    /// The reduction's scalar remainder: `while j < end { s = fold(s, addend); j += 1; }`.
    #[allow(clippy::too_many_arguments)]
    fn emit_reduction_tail(
        &mut self,
        j: Symbol,
        slot: ValueId,
        end_v: ValueId,
        ity: &MirType,
        s_sym: Symbol,
        addend: &Expr,
        redop: RedOp,
    ) {
        self.bind(j, slot, ity.clone());
        let hdr = self.builder.new_block();
        let bb = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(hdr, vec![]);

        self.builder.switch_to(hdr);
        self.terminated = false;
        let jr = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let rc = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, jr, end_v));
        self.builder.cond_br(rc, bb, vec![], exit, vec![]);

        self.builder.switch_to(bb);
        self.terminated = false;
        self.bind(j, slot, ity.clone());
        let (s_slot, s_ty) = self.lookup(s_sym).expect("reduction var in scope");
        let sv = self
            .builder
            .build(s_ty.clone(), Op::Load(s_slot, s_ty.clone()));
        let sum = match redop {
            RedOp::Add => self.scalar_accumulate(sv, addend, &s_ty),
            RedOp::Fmax | RedOp::Fmin => {
                let from = self.expr_mir(addend);
                let a = self.lower_expr(addend);
                let a = self.coerce_to(a, &from, &s_ty, true);
                self.reduce_combine_scalar(redop, &s_ty, sv, a)
            }
        };
        self.builder.build_void(Op::Store {
            ptr: s_slot,
            value: sum,
        });
        let jc = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let one = self
            .builder
            .build(ity.clone(), Op::ConstInt(1, ity.clone()));
        let jn = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, jc, one));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: jn,
        });
        self.builder.br(hdr, vec![]);

        self.builder.switch_to(exit);
        self.terminated = false;
    }

    /// Combine two vectors under the reduction's fold: lane-wise sum (`FAdd`/`Add`), or `fmax`/`fmin`
    /// as the same compare + select the scalar intrinsic lowers to (so vector and scalar agree).
    fn reduce_combine_vec(
        &mut self,
        redop: RedOp,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        a: ValueId,
        b: ValueId,
    ) -> ValueId {
        match redop {
            RedOp::Add => {
                let op = if lane.is_float() {
                    BinOp::FAdd
                } else {
                    BinOp::Add
                };
                self.builder.build(vty.clone(), Op::Bin(op, a, b))
            }
            RedOp::Fmax | RedOp::Fmin => {
                let pred = if redop == RedOp::Fmax {
                    CmpOp::Fogt
                } else {
                    CmpOp::Folt
                };
                let mty = MirType::Vec(Box::new(mask_lane_type(lane)), w);
                let mask = self.builder.build(mty, Op::Cmp(pred, a, b));
                self.builder.build(vty.clone(), Op::Select(mask, a, b))
            }
        }
    }

    /// Scalar counterpart of `reduce_combine_vec`, used for the horizontal lane reduce and the
    /// scalar remainder.
    fn reduce_combine_scalar(
        &mut self,
        redop: RedOp,
        ty: &MirType,
        a: ValueId,
        b: ValueId,
    ) -> ValueId {
        match redop {
            RedOp::Add => {
                let op = if ty.is_float() {
                    BinOp::FAdd
                } else {
                    BinOp::Add
                };
                self.builder.build(ty.clone(), Op::Bin(op, a, b))
            }
            RedOp::Fmax | RedOp::Fmin => {
                let pred = if redop == RedOp::Fmax {
                    CmpOp::Fogt
                } else {
                    CmpOp::Folt
                };
                let mask = self.builder.build(mask_ty(ty), Op::Cmp(pred, a, b));
                self.builder.build(ty.clone(), Op::Select(mask, a, b))
            }
        }
    }

    /// Fold `addend` into vector accumulator `acc`. For a float reduction, `acc + a*b` fuses into one
    /// `Fma`; integer reductions use a plain vector `Add` (no integer FMA, and reassociation is
    /// exact so it needs none).
    #[allow(clippy::too_many_arguments)]
    fn vec_accumulate(
        &mut self,
        acc: ValueId,
        addend: &Expr,
        j: Symbol,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        vlocals: &mut HashMap<Symbol, ValueId>,
    ) -> ValueId {
        if lane.is_float() {
            if let ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } = &addend.kind
            {
                let a = self.vec_lower_value(lhs, j, lane, vty, w, vlocals);
                let b = self.vec_lower_value(rhs, j, lane, vty, w, vlocals);
                return self.builder.build(vty.clone(), Op::Fma(a, b, acc));
            }
        }
        let vx = self.vec_lower_value(addend, j, lane, vty, w, vlocals);
        let add = if lane.is_float() {
            BinOp::FAdd
        } else {
            BinOp::Add
        };
        self.builder.build(vty.clone(), Op::Bin(add, acc, vx))
    }

    /// Scalar `acc + addend`, fusing `acc + a*b` into one `Fma` for floats; plain `Add` for ints.
    fn scalar_accumulate(&mut self, acc: ValueId, addend: &Expr, ty: &MirType) -> ValueId {
        if ty.is_float() {
            if let ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } = &addend.kind
            {
                let a = self.lower_fma_operand(lhs, ty);
                let b = self.lower_fma_operand(rhs, ty);
                return self.builder.build(ty.clone(), Op::Fma(a, b, acc));
            }
        }
        let v = self.lower_expr(addend);
        let vty = self.expr_mir(addend);
        let v = self.coerce_to(v, &vty, ty, true);
        let add = if ty.is_float() {
            BinOp::FAdd
        } else {
            BinOp::Add
        };
        self.builder.build(ty.clone(), Op::Bin(add, acc, v))
    }

    /// Lower one statement of a vector-loop body. Mirrors the validated shapes in `vectorizable`.
    fn vec_lower_stmt(
        &mut self,
        s: &Stmt,
        j: Symbol,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        vlocals: &mut HashMap<Symbol, ValueId>,
    ) {
        match &s.kind {
            StmtKind::Let {
                pat:
                    Pattern {
                        kind: ast::PatKind::Ident(name),
                        ..
                    },
                init: Some(e),
                ..
            } => {
                let v = self.vec_lower_value(e, j, lane, vty, w, vlocals);
                vlocals.insert(*name, v);
            }
            StmtKind::Assign { target, op, value } => {
                let rhs = self.vec_lower_value(value, j, lane, vty, w, vlocals);
                match &target.kind {
                    ExprKind::Path(p) if p.is_single() && vlocals.contains_key(&p.first().sym) => {
                        let name = p.first().sym;
                        let stored = if matches!(op, ast::AssignOp::Assign) {
                            rhs
                        } else {
                            let cur = vlocals[&name];
                            let bin = compound_binop(*op, lane.is_float(), lane_signed(lane));
                            self.builder.build(vty.clone(), Op::Bin(bin, cur, rhs))
                        };
                        vlocals.insert(name, stored);
                    }
                    ExprKind::Index { base, indices } => {
                        let addr = self.vec_elem_addr(base, &indices[0], lane);
                        let stored = if matches!(op, ast::AssignOp::Assign) {
                            rhs
                        } else {
                            let cur = self.builder.build(vty.clone(), Op::Load(addr, vty.clone()));
                            let bin = compound_binop(*op, lane.is_float(), lane_signed(lane));
                            self.builder.build(vty.clone(), Op::Bin(bin, cur, rhs))
                        };
                        self.builder.build_void(Op::Store {
                            ptr: addr,
                            value: stored,
                        });
                        // A store may invalidate any cached load (conservatively, all of them).
                        self.vec_loads.clear();
                        // Intermediate-forwarding (the fused-elementwise-chain win): the value just
                        // stored *is* the current content of `target[index]`. For a unit-stride store,
                        // re-seed the load cache with it so a later read of the same element in this
                        // fused body uses the register value instead of reloading from memory. A fused
                        // `t[i] = f(x[i]); out[i] = g(t[i])` then costs read-x + write-t + write-out and
                        // drops the read-t reload (4 streams -> 3; longer chains drop one reload each).
                        // Only the just-written key survives the clear above, so any *aliasing* store
                        // still invalidates it (the next store clears the cache again before re-seeding).
                        if affine_stride(&indices[0], j) == Some(1) {
                            if let Some(k) = load_key(base, &indices[0], self.interner) {
                                self.vec_loads.insert(k, stored);
                            }
                        }
                    }
                    _ => unreachable!("vec_lower_stmt on unvalidated target"),
                }
            }
            _ => unreachable!("vec_lower_stmt on unvalidated statement"),
        }
    }

    /// Lower a value-expression to a `vty` vector. Array reads become vector loads (unit-stride) or
    /// scalar-load-then-splat (invariant); scalars splat; arithmetic is lane-wise.
    fn vec_lower_value(
        &mut self,
        e: &Expr,
        j: Symbol,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        vlocals: &mut HashMap<Symbol, ValueId>,
    ) -> ValueId {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() && vlocals.contains_key(&p.first().sym) => {
                vlocals[&p.first().sym]
            }
            ExprKind::Index { base, indices } if indices.len() == 1 => {
                // Reuse a vector already loaded for this exact index in the current body copy: a
                // body like relu's `if x[i] > 0 { x[i] } else { 0 }` reads x[i] twice; without this
                // it loads twice (50% extra memory traffic), since loads aren't CSE'd (alias-unsafe).
                let key = load_key(base, &indices[0], self.interner);
                if let Some(k) = &key {
                    if let Some(&cached) = self.vec_loads.get(k) {
                        return cached;
                    }
                }
                let addr = self.vec_elem_addr(base, &indices[0], lane);
                let v = if affine_stride(&indices[0], j) == Some(1) {
                    self.builder.build(vty.clone(), Op::Load(addr, vty.clone()))
                } else {
                    // invariant in j: load one scalar and broadcast.
                    let scalar = self
                        .builder
                        .build(lane.clone(), Op::Load(addr, lane.clone()));
                    self.builder.build(vty.clone(), Op::Splat(scalar))
                };
                if let Some(k) = key {
                    self.vec_loads.insert(k, v);
                }
                v
            }
            ExprKind::Binary { op, lhs, rhs } => {
                // Contract a float `x + y*z` into one lane-wise fused multiply-add — this is the
                // matmul/saxpy inner-loop win (one `vfmadd` per vector instead of mul+add).
                if *op == ast::BinOp::Add && lane.is_float() {
                    if let Some(v) = self.vec_try_fma(lhs, rhs, j, lane, vty, w, vlocals) {
                        return v;
                    }
                }
                let l = self.vec_lower_value(lhs, j, lane, vty, w, vlocals);
                let r = self.vec_lower_value(rhs, j, lane, vty, w, vlocals);
                let bin = arith_binop(*op, lane.is_float(), lane_signed(lane));
                self.builder.build(vty.clone(), Op::Bin(bin, l, r))
            }
            ExprKind::Unary {
                op: ast::UnOp::Neg,
                expr,
            } => {
                let v = self.vec_lower_value(expr, j, lane, vty, w, vlocals);
                self.builder.build(vty.clone(), Op::Neg(v))
            }
            // if-conversion: `if a CMP b { t } else { e }` -> vector compare mask + lane-wise blend.
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let ExprKind::Binary { op, lhs, rhs } = &cond.kind else {
                    unreachable!("vec_lower_value on unvalidated if-condition");
                };
                let lv = self.vec_lower_value(lhs, j, lane, vty, w, vlocals);
                let rv = self.vec_lower_value(rhs, j, lane, vty, w, vlocals);
                let mask_ty = MirType::Vec(Box::new(mask_lane_type(lane)), w);
                let pred = cmp_pred(*op, lane.is_float(), lane_signed(lane));
                let mask = self.builder.build(mask_ty, Op::Cmp(pred, lv, rv));
                let tv = block_value(then_branch).unwrap();
                let ev = branch_value(else_branch.as_deref().unwrap()).unwrap();
                let tvec = self.vec_lower_value(tv, j, lane, vty, w, vlocals);
                let evec = self.vec_lower_value(ev, j, lane, vty, w, vlocals);
                self.builder
                    .build(vty.clone(), Op::Select(mask, tvec, evec))
            }
            // Element-wise math intrinsics, lowered per lane (see `vec_check_value`).
            ExprKind::Call { callee, args, .. } => match self.vectorizable_intrinsic(callee) {
                Some(MathIntrinsic::Sqrt) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.builder.build(vty.clone(), Op::Sqrt(x))
                }
                Some(MathIntrinsic::Rsqrt) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let s = self.builder.build(vty.clone(), Op::Sqrt(x));
                    let one = self.splat_const_f(1.0, vty);
                    self.builder
                        .build(vty.clone(), Op::Bin(BinOp::FDiv, one, s))
                }
                Some(MathIntrinsic::Abs) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    if lane.is_int() {
                        // Integer abs is `select(x < 0, -x, x)` (signed, matching the scalar
                        // `emit_int_abs`). The float `emit_abs` (FSub/Cmp(Fogt)/Select) would emit a
                        // float op on an int vector — MIR the verifier and Cranelift reject, which the
                        // interpreter ran lossily at -O0; the same guard the scalar path already has.
                        let zero = self.splat_const_i(0, vty);
                        let neg = self
                            .builder
                            .build(vty.clone(), Op::Bin(BinOp::Sub, zero, x));
                        let mty = MirType::Vec(Box::new(mask_lane_type(lane)), w);
                        let isneg = self.builder.build(mty, Op::Cmp(CmpOp::Slt, x, zero));
                        self.builder.build(vty.clone(), Op::Select(isneg, neg, x))
                    } else {
                        self.emit_abs(x, vty)
                    }
                }
                Some(
                    op @ (MathIntrinsic::Round
                    | MathIntrinsic::Floor
                    | MathIntrinsic::Ceil
                    | MathIntrinsic::Trunc),
                ) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    if lane.is_int() {
                        // Rounding an integer is the identity (matching the scalar guard); `Op::Round`
                        // is a float op the verifier rejects on an int vector.
                        x
                    } else {
                        self.builder
                            .build(vty.clone(), Op::Round(round_mode(op), x))
                    }
                }
                Some(op @ (MathIntrinsic::Fmax | MathIntrinsic::Fmin)) => {
                    let a = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let b = self.vec_lower_value(&args[1], j, lane, vty, w, vlocals);
                    let pred = if matches!(op, MathIntrinsic::Fmax) {
                        CmpOp::Fogt
                    } else {
                        CmpOp::Folt
                    };
                    let mty = MirType::Vec(Box::new(mask_lane_type(lane)), w);
                    let mask = self.builder.build(mty, Op::Cmp(pred, a, b));
                    self.builder.build(vty.clone(), Op::Select(mask, a, b))
                }
                Some(MathIntrinsic::Exp) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_exp_f32(x, vty)
                }
                Some(MathIntrinsic::Log) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_log_f32(x, vty)
                }
                Some(MathIntrinsic::Exp2) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let ln2 = self.splat_const_f(std::f64::consts::LN_2, vty);
                    let xl = self
                        .builder
                        .build(vty.clone(), Op::Bin(BinOp::FMul, x, ln2));
                    self.emit_exp_f32(xl, vty)
                }
                Some(MathIntrinsic::Log2) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let lx = self.emit_log_f32(x, vty);
                    let log2e = self.splat_const_f(std::f64::consts::LOG2_E, vty);
                    self.builder
                        .build(vty.clone(), Op::Bin(BinOp::FMul, lx, log2e))
                }
                Some(MathIntrinsic::Exp10) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let ln10 = self.splat_const_f(std::f64::consts::LN_10, vty);
                    let xl = self
                        .builder
                        .build(vty.clone(), Op::Bin(BinOp::FMul, x, ln10));
                    self.emit_exp_f32(xl, vty)
                }
                Some(MathIntrinsic::Log10) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let lx = self.emit_log_f32(x, vty);
                    let log10e = self.splat_const_f(std::f64::consts::LOG10_E, vty);
                    self.builder
                        .build(vty.clone(), Op::Bin(BinOp::FMul, lx, log10e))
                }
                Some(op @ (MathIntrinsic::Sinh | MathIntrinsic::Cosh)) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_sinh_cosh(x, vty, matches!(op, MathIntrinsic::Cosh))
                }
                Some(MathIntrinsic::Asinh) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_asinh(x, vty)
                }
                Some(MathIntrinsic::Acosh) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_acosh(x, vty)
                }
                Some(MathIntrinsic::Atanh) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_atanh(x, vty)
                }
                Some(MathIntrinsic::Atan) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_atan(x, vty)
                }
                Some(MathIntrinsic::Expm1) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_expm1(x, vty)
                }
                Some(MathIntrinsic::Log1p) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_log1p(x, vty)
                }
                Some(MathIntrinsic::Pow) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let y = self.vec_lower_value(&args[1], j, lane, vty, w, vlocals);
                    let lx = self.emit_log_f32(x, vty);
                    let ylx = self.builder.build(vty.clone(), Op::Bin(BinOp::FMul, y, lx));
                    self.emit_exp_f32(ylx, vty)
                }
                Some(MathIntrinsic::Atan2) => {
                    let y = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let x = self.vec_lower_value(&args[1], j, lane, vty, w, vlocals);
                    self.emit_atan2(y, x, vty)
                }
                Some(MathIntrinsic::Hypot) => {
                    let a = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let b = self.vec_lower_value(&args[1], j, lane, vty, w, vlocals);
                    self.emit_hypot(a, b, vty)
                }
                Some(MathIntrinsic::SiluBackward) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let dy = self.vec_lower_value(&args[1], j, lane, vty, w, vlocals);
                    self.emit_silu_backward(x, dy, vty)
                }
                Some(MathIntrinsic::GeluBackward) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let dy = self.vec_lower_value(&args[1], j, lane, vty, w, vlocals);
                    self.emit_gelu_backward(x, dy, vty)
                }
                Some(MathIntrinsic::SigmoidBackward) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let dy = self.vec_lower_value(&args[1], j, lane, vty, w, vlocals);
                    self.emit_sigmoid_backward(x, dy, vty)
                }
                Some(MathIntrinsic::TanhBackward) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let dy = self.vec_lower_value(&args[1], j, lane, vty, w, vlocals);
                    self.emit_tanh_backward(x, dy, vty)
                }
                Some(MathIntrinsic::EluBackward) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let dy = self.vec_lower_value(&args[1], j, lane, vty, w, vlocals);
                    self.emit_elu_backward(x, dy, vty)
                }
                Some(MathIntrinsic::SoftplusBackward) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let dy = self.vec_lower_value(&args[1], j, lane, vty, w, vlocals);
                    self.emit_softplus_backward(x, dy, vty)
                }
                Some(MathIntrinsic::Erf) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_erf_f32(x, vty)
                }
                Some(op @ (MathIntrinsic::Sin | MathIntrinsic::Cos)) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_trig_f32(x, vty, matches!(op, MathIntrinsic::Cos))
                }
                Some(MathIntrinsic::Tanh) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_tanh(x, vty)
                }
                Some(MathIntrinsic::Sigmoid) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_sigmoid(x, vty)
                }
                Some(MathIntrinsic::Silu) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_silu(x, vty)
                }
                Some(MathIntrinsic::Gelu) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_gelu(x, vty)
                }
                Some(MathIntrinsic::Elu) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_elu(x, vty)
                }
                Some(MathIntrinsic::LeakyRelu) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_leaky_relu(x, vty)
                }
                Some(MathIntrinsic::Softplus) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_softplus(x, vty)
                }
                Some(MathIntrinsic::Mish) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_mish(x, vty)
                }
                Some(MathIntrinsic::Selu) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_selu(x, vty)
                }
                Some(MathIntrinsic::Tanhshrink) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_tanhshrink(x, vty)
                }
                Some(MathIntrinsic::HardSigmoid) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_hardsigmoid(x, vty)
                }
                Some(MathIntrinsic::HardSwish) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_hardswish(x, vty)
                }
                Some(MathIntrinsic::Softsign) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_softsign(x, vty)
                }
                Some(MathIntrinsic::LogSigmoid) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_logsigmoid(x, vty)
                }
                Some(MathIntrinsic::Tan) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_tan(x, vty)
                }
                Some(MathIntrinsic::Asin) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_asin(x, vty)
                }
                Some(MathIntrinsic::Acos) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_acos(x, vty)
                }
                Some(MathIntrinsic::Cbrt) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_cbrt(x, vty)
                }
                None => unreachable!("vectorizer accepted a call it cannot lower"),
            },
            // an invariant scalar or literal: lower as a scalar (coerced to the lane type) and splat.
            _ => {
                let from = self.expr_mir(e);
                let scalar = self.lower_expr(e);
                let scalar = self.coerce_to(scalar, &from, lane, true);
                self.builder.build(vty.clone(), Op::Splat(scalar))
            }
        }
    }

    /// In a vectorized loop body, contract a float `x + y*z` (or `y*z + x`) into one lane-wise
    /// `Fma`. Returns the fused vector value, or `None` if neither side is a multiply (the caller
    /// then emits a plain vector add). The whole expression tree was already accepted by the
    /// vectorizer's dependence check, so lowering its leaves here is sound.
    #[allow(clippy::too_many_arguments)]
    fn vec_try_fma(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        j: Symbol,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        vlocals: &mut HashMap<Symbol, ValueId>,
    ) -> Option<ValueId> {
        fn as_fmul(e: &Expr) -> Option<(&Expr, &Expr)> {
            match &e.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Mul,
                    lhs,
                    rhs,
                } => Some((lhs.as_ref(), rhs.as_ref())),
                _ => None,
            }
        }
        if let Some((y, z)) = as_fmul(lhs) {
            let yv = self.vec_lower_value(y, j, lane, vty, w, vlocals);
            let zv = self.vec_lower_value(z, j, lane, vty, w, vlocals);
            let xv = self.vec_lower_value(rhs, j, lane, vty, w, vlocals);
            return Some(self.builder.build(vty.clone(), Op::Fma(yv, zv, xv)));
        }
        if let Some((y, z)) = as_fmul(rhs) {
            let xv = self.vec_lower_value(lhs, j, lane, vty, w, vlocals);
            let yv = self.vec_lower_value(y, j, lane, vty, w, vlocals);
            let zv = self.vec_lower_value(z, j, lane, vty, w, vlocals);
            return Some(self.builder.build(vty.clone(), Op::Fma(yv, zv, xv)));
        }
        None
    }

    /// The element address `&base[index]` (a `gep` by the scalar index, reusing the loop index slot
    /// bound for `j`), used as the base of a vector load/store.
    fn vec_elem_addr(&mut self, base: &Expr, index: &Expr, lane: &MirType) -> ValueId {
        let base_ptr = self.lower_expr(base);
        let idx = self.lower_expr(index);
        self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: base_ptr,
                index: idx,
                elem: lane.clone(),
            },
        )
    }

    /// Lower `for idx in start..end { body }` where `start`/`end` are already-lowered `i64` values
    /// (used by parallel outlining). The loop runs in `ity` — the index variable's source type, so
    /// the body's index arithmetic stays type-consistent — with the `i64` bounds coerced into it.
    fn lower_ranged_loop(
        &mut self,
        idx: Symbol,
        start: ValueId,
        end: ValueId,
        ity: MirType,
        body: &Block,
    ) {
        // The runtime hands us `[start, end)` as i64; narrow to the index type the body expects.
        let start = self.coerce_to(start, &MirType::I64, &ity, true);
        let end = self.coerce_to(end, &MirType::I64, &ity, true);

        // SIMD-vectorize the per-thread chunk too, so `@parallel` kernels run vectorized on every
        // core (parallelism × SIMD), not scalar-per-core.
        if self.try_vectorize_ranged(idx, start, end, &ity, body) {
            return;
        }

        let slot = self.builder.alloca(ity.clone());
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: start,
        });
        self.push_scope();
        self.bind(idx, slot, ity.clone());

        let header = self.builder.new_block();
        let body_bb = self.builder.new_block();
        let latch = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);

        self.builder.switch_to(header);
        self.terminated = false;
        let i_val = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let c = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, i_val, end));
        self.builder.cond_br(c, body_bb, vec![], exit, vec![]);

        // `continue` targets the latch (the increment), not the header — see `lower_for`.
        self.builder.switch_to(body_bb);
        self.terminated = false;
        // The `@parallel` per-thread ranged loop carries no user label (a labeled break across the
        // parallel boundary is not modeled); an unlabeled break/continue still targets it.
        self.loops.push((None, latch, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            self.builder.br(latch, vec![]);
        }

        self.builder.switch_to(latch);
        self.terminated = false;
        let cur = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let one = self
            .builder
            .build(ity.clone(), Op::ConstInt(1, ity.clone()));
        let next = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, cur, one));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: next,
        });
        self.builder.br(header, vec![]);

        self.pop_scope();
        self.builder.switch_to(exit);
        self.terminated = false;
    }

    /// Lower a condition used in a **boolean context** — the condition of `if`/`while`, the operands
    /// of short-circuit `&&`/`||`, and the argument of `assert` — normalizing it to an `i1`.
    ///
    /// The language deliberately permits a C-like non-bool scalar condition (`if 5 {}`, `if x {}`):
    /// a nonzero value is true. An `i1`/integer condition is already consistent across both backends
    /// (nonzero int = true) and passes through unchanged. A **float** condition, however, must be
    /// normalized to `cond != 0.0` (an `Op::Cmp(Fone)` against a float zero of the same type,
    /// yielding an `i1`): otherwise the native (Cranelift) backend's `brif`/`fcvt` truncates
    /// `0.5 -> 0` (so `assert(0.5)` wrongly traps) or rejects the float controlling type outright
    /// (`brif.f32 ... has an invalid controlling type`), while the interpreter applied a *different*
    /// truthiness (`f != 0.0` in its `assert` arm vs `as_int() != 0` for `if`/`while`, which itself
    /// truncates `0.5 -> 0`) — a three-way divergence on a program the front-end accepts. Emitting
    /// the compare here makes the condition an `i1` *before* it reaches any backend, so they all
    /// agree. `NaN != 0.0` is true (intentional C-like truthiness — the interpreter's Rust `!=` and
    /// Cranelift's `FloatCC::NotEqual` both treat NaN as nonzero/true). The integer/`i1` path is
    /// untouched.
    fn lower_bool_cond(&mut self, cond: &Expr) -> ValueId {
        let v = self.lower_expr(cond);
        let ty = self.expr_mir(cond);
        if ty == MirType::I1 {
            // Already a boolean (a comparison result — the common case). Pass through unchanged.
            v
        } else if ty.is_float() {
            let zero = self
                .builder
                .build(ty.clone(), Op::ConstFloat(0.0, ty.clone()));
            self.builder
                .build(MirType::I1, Op::Cmp(CmpOp::Fone, v, zero))
        } else if ty.is_int() {
            // A non-bool integer condition (`if 5`, `if x` for `x: i32`): C-like truthiness, nonzero
            // is true. Normalize to `cond != 0` -> i1 so `cond_br` never receives a wider integer.
            // The verifier accepts that at -O0 but mem2reg rejects it at -O2 ("cond_br condition has
            // type i32 but expected i1") — a latent compiler crash at -O2 on a program the front-end
            // accepts. The bool/i1 path above is untouched, so the comparison-condition corpus is
            // unaffected.
            let zero = self.builder.build(ty.clone(), Op::ConstInt(0, ty.clone()));
            self.builder.build(MirType::I1, Op::Cmp(CmpOp::Ne, v, zero))
        } else {
            v
        }
    }

    fn lower_if(&mut self, cond: &Expr, then_branch: &Block, else_branch: Option<&Expr>) {
        let c = self.lower_bool_cond(cond);
        let then_bb = self.builder.new_block();
        let merge = self.builder.new_block();
        let else_bb = if else_branch.is_some() {
            self.builder.new_block()
        } else {
            merge
        };
        self.builder.cond_br(c, then_bb, vec![], else_bb, vec![]);

        self.builder.switch_to(then_bb);
        self.terminated = false;
        self.lower_block(then_branch);
        if !self.terminated {
            self.builder.br(merge, vec![]);
        }

        if let Some(e) = else_branch {
            self.builder.switch_to(else_bb);
            self.terminated = false;
            self.lower_expr_stmt(e);
            if !self.terminated {
                self.builder.br(merge, vec![]);
            }
        }

        self.builder.switch_to(merge);
        self.terminated = false;
    }

    /// Lower an `if`/`else` used as a *value* (e.g. a block tail or `let x = if …`). When the
    /// expression has a real (non-unit) type and both arms exist, the merge block takes a
    /// parameter that each arm passes its value to; otherwise this behaves like the statement form
    /// and yields a dummy zero (the value is unused).
    fn lower_if_value(
        &mut self,
        cond: &Expr,
        then_branch: &Block,
        else_branch: Option<&Expr>,
        e: &Expr,
    ) -> ValueId {
        let result_ty = self.expr_mir(e);
        let produces_value = else_branch.is_some() && result_ty != MirType::Void;
        // An aggregate result flows as its base pointer, so the merge param is `Ptr` (see
        // `merge_repr_ty`); scalars are unchanged.
        let merge_ty = merge_repr_ty(&result_ty);

        let c = self.lower_bool_cond(cond);
        let then_bb = self.builder.new_block();
        let merge = self.builder.new_block();
        let else_bb = if else_branch.is_some() {
            self.builder.new_block()
        } else {
            merge
        };
        let merge_param = if produces_value {
            Some(self.builder.block_param(merge, merge_ty.clone()))
        } else {
            None
        };
        self.builder.cond_br(c, then_bb, vec![], else_bb, vec![]);

        // then arm
        self.builder.switch_to(then_bb);
        self.terminated = false;
        let tv = self.lower_block(then_branch);
        if !self.terminated {
            let args = match (merge_param.is_some(), tv) {
                (true, Some(v)) => {
                    // Coerce the arm's value to the merged result type (the join of the two arms) so
                    // both arms pass the merge param the *same* MIR type. Without this, arms of
                    // different width/kind (`if c { 1 } else { 2.5 }`) pass a mismatched value: the
                    // native verifier rejects the merge while the interpreter runs loosely — a
                    // backend divergence on a program the front-end accepted.
                    let (from, signed) = match &then_branch.tail {
                        Some(t) => (self.expr_mir(t), self.signed(t)),
                        None => (result_ty.clone(), true),
                    };
                    vec![self.coerce_to(v, &from, &merge_ty, signed)]
                }
                (true, None) => vec![self.const_zero(merge_ty.clone())],
                (false, _) => vec![],
            };
            self.builder.br(merge, args);
        }

        // else arm
        if let Some(els) = else_branch {
            self.builder.switch_to(else_bb);
            self.terminated = false;
            let ev = self.lower_expr(els);
            if !self.terminated {
                let args = if merge_param.is_some() {
                    let from = self.expr_mir(els);
                    vec![self.coerce_to(ev, &from, &merge_ty, self.signed(els))]
                } else {
                    vec![]
                };
                self.builder.br(merge, args);
            }
        }

        self.builder.switch_to(merge);
        self.terminated = false;
        match merge_param {
            Some(p) => p,
            None => self.const_zero(if result_ty == MirType::Void {
                MirType::I32
            } else {
                merge_ty
            }),
        }
    }

    // ---- places (lvalues) ----

    /// Initialize an array alloca (`base`) of `n` elements of type `elem` from an array-literal or
    /// array-repeat initializer, storing each element through a `gep`.
    fn lower_array_init(&mut self, base: ValueId, elem: &MirType, n: u32, init: &Expr) {
        // An aggregate element (an array of structs/tuples — `elem` is a byte-buffer `Array`) must be
        // deep-copied into each element slot; a plain `store_element` would store the element's base
        // *pointer* as a scalar (a silent miscompile). A scalar element keeps the store path.
        let aggregate = matches!(elem, MirType::Array(..));
        match &init.kind {
            ExprKind::ArrayLit(elems) => {
                for (i, el) in elems.iter().enumerate() {
                    if aggregate {
                        let ep = self.gep_elem(base, elem, i as i128);
                        let ety = self.expr_ty(el);
                        self.init_field(ep, &ety, el);
                    } else {
                        let v0 = self.lower_expr(el);
                        let vty = self.expr_mir(el);
                        // Coerce to the element type so e.g. a `[bf16; N]` literal stores bf16-rounded
                        // 16-bit values, not raw f32. A no-op when the element already matches.
                        let v = self.coerce_to(v0, &vty, elem, self.signed(el));
                        self.store_element(base, elem, i as i128, v);
                    }
                }
            }
            ExprKind::ArrayRepeat { value, .. } => {
                if aggregate {
                    // `[agg; n]`: deep-copy the aggregate into every element slot. Re-emitting the
                    // literal/copy per element is correct (each element owns its storage); the scalar
                    // fill-loop path below does not apply.
                    let ety = self.expr_ty(value);
                    for i in 0..n as i128 {
                        let ep = self.gep_elem(base, elem, i);
                        self.init_field(ep, &ety, value);
                    }
                    return;
                }
                // `[value; n]` evaluates `value` once and fills every slot with it. Small arrays
                // unroll to straight-line stores; large ones lower to a fill loop so that, e.g.,
                // `[0; 1_000_000]` does not generate a million instructions.
                let v0 = self.lower_expr(value);
                let vty = self.expr_mir(value);
                let v = self.coerce_to(v0, &vty, elem, self.signed(value));
                if n <= REPEAT_UNROLL_LIMIT {
                    for i in 0..n as i128 {
                        self.store_element(base, elem, i, v);
                    }
                } else {
                    self.lower_fill_loop(base, elem, n, v);
                }
            }
            _ => {
                self.unsupported(init.span, "array initializer (expected `[..]` or `[v; n]`)");
            }
        }
    }

    /// Emit `for i in 0..n { base[i] = v }` as a CFG loop. Used for large array-repeat initializers
    /// so the IR stays compact; mem2reg later promotes the loop counter to a register.
    fn lower_fill_loop(&mut self, base: ValueId, elem: &MirType, n: u32, v: ValueId) {
        let i64t = MirType::I64;
        let slot = self.builder.alloca(i64t.clone());
        let zero = self
            .builder
            .build(i64t.clone(), Op::ConstInt(0, i64t.clone()));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: zero,
        });

        let header = self.builder.new_block();
        let body = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);

        self.builder.switch_to(header);
        let i_val = self
            .builder
            .build(i64t.clone(), Op::Load(slot, i64t.clone()));
        let nconst = self
            .builder
            .build(i64t.clone(), Op::ConstInt(n as i128, i64t.clone()));
        let cond = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, i_val, nconst));
        self.builder.cond_br(cond, body, vec![], exit, vec![]);

        self.builder.switch_to(body);
        let i_cur = self
            .builder
            .build(i64t.clone(), Op::Load(slot, i64t.clone()));
        let p = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: base,
                index: i_cur,
                elem: elem.clone(),
            },
        );
        self.builder.build_void(Op::Store { ptr: p, value: v });
        let one = self
            .builder
            .build(i64t.clone(), Op::ConstInt(1, i64t.clone()));
        let next = self
            .builder
            .build(i64t.clone(), Op::Bin(BinOp::Add, i_cur, one));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: next,
        });
        self.builder.br(header, vec![]);

        self.builder.switch_to(exit);
        self.terminated = false;
    }

    /// Store `value` into `base[index]` for an array element of type `elem`.
    fn store_element(&mut self, base: ValueId, elem: &MirType, index: i128, value: ValueId) {
        let idx = self
            .builder
            .build(MirType::I64, Op::ConstInt(index, MirType::I64));
        let p = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: base,
                index: idx,
                elem: elem.clone(),
            },
        );
        self.builder.build_void(Op::Store { ptr: p, value });
    }

    /// Pointer to byte offset `off` within an aggregate buffer `base`. Using `elem = I8` makes the
    /// GEP index raw bytes (Cranelift scales by `size_of(I8) = 1`; the interpreter, which indexes
    /// slots, uses the byte offset directly — distinct field offsets never alias, so both backends
    /// observe the same field values). This is the one primitive tuple/struct field access needs.
    fn field_ptr(&mut self, base: ValueId, off: u64) -> ValueId {
        let idx = self
            .builder
            .build(MirType::I64, Op::ConstInt(off as i128, MirType::I64));
        self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: base,
                index: idx,
                elem: MirType::I8,
            },
        )
    }

    /// Lower a tuple literal `(a, b, …)` into the byte buffer at `base`: each field is initialized at
    /// its padded byte offset (`aggregate_layout` is the registry-aware layout authority). A field
    /// that is itself a struct/tuple/array recurses via `init_field`; a scalar field is coerced to
    /// its type and stored (so a narrowing field, e.g. a `bf16`, rounds on store).
    fn lower_tuple_init(&mut self, base: ValueId, tuple_ty: &Ty, items: &[Expr]) {
        let Ty::Tuple(field_tys) = tuple_ty else {
            return;
        };
        let Some((offsets, _, _)) = self.aggregate_layout(field_tys) else {
            if let Some(first) = items.first() {
                self.unsupported(first.span, "tuple with an unsized field");
            }
            return;
        };
        let plan: Vec<(u64, Ty)> = offsets.into_iter().zip(field_tys.iter().cloned()).collect();
        for (item, (off, fty)) in items.iter().zip(plan) {
            let p = self.field_ptr(base, off);
            self.init_field(p, &fty, item);
        }
    }

    /// Address + MIR type of tuple field `index` of the tuple expression `base`. The tuple local's
    /// value *is* its buffer pointer (an `Array`-typed slot returns the slot directly), so this just
    /// GEPs to the field's padded byte offset. Drives both reads (`t.0`) and writes (`t.0 = …`).
    fn tuple_field_place(&mut self, base: &Expr, index: usize) -> (ValueId, MirType) {
        if let Ty::Tuple(fields) = self.expr_ty(base) {
            if let Some((offsets, _, _)) = self.aggregate_layout(&fields) {
                if let (Some(&off), Some(fty)) = (offsets.get(index), fields.get(index)) {
                    let fmty = self.mir_ty_of(fty);
                    let base_ptr = self.lower_expr(base);
                    let p = self.field_ptr(base_ptr, off);
                    return (p, fmty);
                }
            }
        }
        self.unsupported(base.span, "tuple field access");
        let ty = MirType::I32;
        (self.builder.alloca(ty.clone()), ty)
    }

    /// Row-major element strides for a tensor `base`, if computable. `stride_k` = product of the
    /// dims *after* position `k`; computable when every dim after the first is a compile-time
    /// `Const` (the leading dim may be `Var`/`Dynamic` — it never contributes to a stride). Returns
    /// the per-axis strides and the element's MIR type, or `None` (caller falls back to scalar/error).
    /// This is what makes the shape-typed surface — `a[i, j]` on `Tensor[f32, M, N]` — executable.
    fn tensor_strides(&self, base: &Expr) -> Option<(Vec<usize>, MirType)> {
        let Ty::Tensor {
            elem,
            shape,
            layout,
        } = self.expr_ty(base)
        else {
            return None;
        };
        // Only row-major (contiguous) tensors flatten to `i0*s0 + … + i_{n-1}` with these strides.
        if !matches!(layout, mercury_types::Layout::Contiguous) {
            return None;
        }
        let dims = &shape.0;
        let rank = dims.len();
        if rank == 0 {
            return None;
        }
        let mut strides = vec![1usize; rank];
        for k in (0..rank - 1).rev() {
            let next = match dims[k + 1] {
                mercury_types::Dim::Const(d) => d as usize,
                _ => return None, // a symbolic/dynamic interior dim — stride unknown at compile time
            };
            strides[k] = strides[k + 1] * next;
        }
        Some((strides, mir_ty(&Ty::Scalar(elem))))
    }

    /// Lower an index expression and widen it to `I64`, so a multi-dimensional flat-offset
    /// computation is single-typed regardless of the index's source width (loop vars are usually
    /// `i32`). Non-negative loop counters, so sign/zero-extension agree; we pick by signedness.
    fn lower_index_i64(&mut self, e: &Expr) -> ValueId {
        let v = self.lower_expr(e);
        let ty = self.expr_mir(e);
        if ty == MirType::I64 {
            return v;
        }
        let kind = if self.signed(e) {
            CastKind::SExt
        } else {
            CastKind::ZExt
        };
        self.builder
            .build(MirType::I64, Op::Cast(kind, v, MirType::I64))
    }

    /// Lower a multi-dimensional tensor index `base[i0, i1, …]` to a single `Gep` at the row-major
    /// flat element offset `Σ iₖ·strideₖ`. Returns `None` if `base` is not a flattenable tensor.
    fn lower_multi_index(&mut self, base: &Expr, indices: &[Expr]) -> Option<(ValueId, MirType)> {
        let (strides, elem) = self.tensor_strides(base)?;
        if strides.len() != indices.len() {
            return None; // rank mismatch (sema already reported E0501); fall back to unsupported
        }
        let base_ptr = self.lower_expr(base);
        let mut flat: Option<ValueId> = None;
        for (ix, &st) in indices.iter().zip(strides.iter()) {
            let iv = self.lower_index_i64(ix);
            let term = if st == 1 {
                iv
            } else {
                let s = self
                    .builder
                    .build(MirType::I64, Op::ConstInt(st as i128, MirType::I64));
                self.builder.build(MirType::I64, Op::Bin(BinOp::Mul, iv, s))
            };
            flat = Some(match flat {
                None => term,
                Some(acc) => self
                    .builder
                    .build(MirType::I64, Op::Bin(BinOp::Add, acc, term)),
            });
        }
        let flat = flat.expect("indices non-empty");
        let p = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: base_ptr,
                index: flat,
                elem: elem.clone(),
            },
        );
        Some((p, elem))
    }

    fn lower_place(&mut self, e: &Expr) -> (ValueId, MirType) {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() => {
                if let Some((slot, ty)) = self.lookup(p.first().sym) {
                    return (slot, ty);
                }
                self.unsupported(p.span, "assignment to this name");
                let ty = self.expr_mir(e);
                (self.builder.alloca(ty.clone()), ty)
            }
            ExprKind::Unary {
                op: ast::UnOp::Deref,
                expr,
            } => {
                let ptr = self.lower_expr(expr);
                (ptr, self.expr_mir(e))
            }
            ExprKind::Index { base, indices } if indices.len() == 1 => {
                let base_ptr = self.lower_expr(base);
                let idx = self.lower_expr(&indices[0]);
                // Prefer the element type from the base's array type; fall back to the indexed
                // expression's own type (slices/tensors/pointers). Resolve through the
                // registry-aware `mir_ty_of` so a struct/tuple element becomes its byte-buffer
                // `Array` type (the GEP strides by the real element size, and the read path below
                // treats it as an aggregate address) rather than the registry-blind `I32` fallback.
                let elem = match self.expr_ty(base) {
                    Ty::Array { elem, .. } => self.mir_ty_of(&elem),
                    _ => self.expr_mir(e),
                };
                let p = self.builder.build(
                    MirType::Ptr,
                    Op::Gep {
                        ptr: base_ptr,
                        index: idx,
                        elem: elem.clone(),
                    },
                );
                (p, elem)
            }
            // `t.0 = …` — assign to a tuple field at its byte offset.
            ExprKind::TupleField { base, index } => self.tuple_field_place(base, *index as usize),
            // `s.field = …` — assign to a struct field at its declared byte offset.
            ExprKind::Field { base, name } => self.struct_field_place(base, name.sym),
            // Multi-dimensional tensor indexing `t[i, j, …]` — the shape-typed surface. Flatten to a
            // row-major offset using the tensor's static strides.
            ExprKind::Index { base, indices } if indices.len() >= 2 => {
                if let Some(pe) = self.lower_multi_index(base, indices) {
                    return pe;
                }
                self.unsupported(
                    e.span,
                    "multi-dimensional indexing on a non-contiguous or symbolically-strided tensor",
                );
                let ty = self.expr_mir(e);
                (self.builder.alloca(ty.clone()), ty)
            }
            _ => {
                self.unsupported(e.span, "assignment target");
                let ty = self.expr_mir(e);
                (self.builder.alloca(ty.clone()), ty)
            }
        }
    }

    // ---- expressions ----

    fn lower_expr(&mut self, e: &Expr) -> ValueId {
        match &e.kind {
            ExprKind::Int(s) => {
                let ty = self.expr_mir(e);
                let v = parse_int(self.interner.resolve(*s));
                let ty = if ty.is_int() { ty } else { MirType::I32 };
                self.builder.build(ty.clone(), Op::ConstInt(v, ty))
            }
            ExprKind::Float(s) => {
                let ty = self.expr_mir(e);
                let v = parse_float(self.interner.resolve(*s));
                let ty = if ty.is_float() { ty } else { MirType::F32 };
                self.builder.build(ty.clone(), Op::ConstFloat(v, ty))
            }
            ExprKind::Bool(b) => self
                .builder
                .build(MirType::I1, Op::ConstInt(*b as i128, MirType::I1)),
            // A char literal is its Unicode scalar value — sema types it `u32`, so it lowers like an
            // integer constant of that value (escapes/`\x`/`\u{…}` decoded by `decode_char_literal`).
            ExprKind::Char(s) => {
                let v = decode_char_literal(self.interner.resolve(*s));
                let ty = self.expr_mir(e);
                let ty = if ty.is_int() { ty } else { MirType::I32 };
                self.builder.build(ty.clone(), Op::ConstInt(v as i128, ty))
            }
            // A string literal materializes its UTF-8 bytes (plus a trailing NUL) into a fresh stack
            // byte buffer and yields the base pointer — sema types it `*u8`, so it follows the same
            // by-pointer convention as an array. `print`/`println` of a `*u8` reads it back
            // byte-by-byte until the NUL (see the intrinsic-call lowering). Both backends GEP/Store
            // one element per byte, so the interpreter's slot-indexed memory and native's byte memory
            // agree (`store_element` strides by element, which is 1 byte for `I8`).
            ExprKind::Str(s) => {
                let bytes = decode_string_literal(self.interner.resolve(*s));
                let elem = MirType::I8;
                let n = bytes.len() as u32 + 1; // + NUL terminator
                let base = self
                    .builder
                    .alloca(MirType::Array(Box::new(elem.clone()), n));
                for (i, b) in bytes.iter().enumerate() {
                    let v = self
                        .builder
                        .build(MirType::I8, Op::ConstInt(*b as i128, MirType::I8));
                    self.store_element(base, &elem, i as i128, v);
                }
                let nul = self.builder.build(MirType::I8, Op::ConstInt(0, MirType::I8));
                self.store_element(base, &elem, bytes.len() as i128, nul);
                base
            }
            ExprKind::Path(p) if p.is_single() => {
                if let Some((slot, ty)) = self.lookup(p.first().sym) {
                    // An array variable *is* its storage: its value is the base pointer, so reads
                    // don't load — indexing geps off this pointer.
                    if matches!(ty, MirType::Array(..)) {
                        slot
                    } else {
                        self.builder.build(ty.clone(), Op::Load(slot, ty))
                    }
                } else if let Some(init) = self.sema.consts.get(&p.first().sym).cloned() {
                    // A top-level `const`: inline its initializer at the use site (the def map records
                    // only the const's type, not its value). The initializer was type-checked by
                    // sema, so its nodes carry types and lower correctly; a const referencing another
                    // const recurses through this same arm.
                    self.lower_expr(&init)
                } else {
                    self.unsupported(p.span, "value reference");
                    let t = self.expr_mir(e);
                    self.const_zero(t)
                }
            }
            ExprKind::Unary { op, expr } => self.lower_unary(*op, expr, e),
            ExprKind::Binary { op, lhs, rhs } => self.lower_binary(*op, lhs, rhs, e),
            ExprKind::Call { callee, args, .. } => self.lower_call(callee, args, e),
            ExprKind::Index { base, indices } if !indices.is_empty() => {
                let (ptr, elem) = self.lower_place(e);
                let _ = (base, indices);
                // A scalar element loads; an aggregate element (an array of structs/tuples) yields
                // its address — the by-pointer convention, so a further `.field`/`[i]`/`.0` GEPs
                // off it. Mirrors the `Field`/`TupleField` read arms. An unconditional `Op::Load`
                // here mis-loaded an aggregate element as a scalar (native verifier reject for a
                // struct element, segfault for a wider tuple element).
                self.load_or_addr(ptr, elem)
            }
            // `t.0` — read tuple field 0 by GEP to its byte offset. A scalar field loads; an
            // aggregate field (a nested struct/tuple/array) yields its address, the by-pointer
            // convention arrays follow, so a further `.field`/`[i]` GEPs off it.
            ExprKind::TupleField { base, index } => {
                let (ptr, fmty) = self.tuple_field_place(base, *index as usize);
                self.load_or_addr(ptr, fmty)
            }
            // `s.field` — read a struct field by GEP to its declared byte offset (scalar loads,
            // aggregate yields its address — see `TupleField`).
            ExprKind::Field { base, name } => {
                // `E::B` parses as a field access on the enum-name path `E`; lower it to the
                // variant's integer discriminant (a C-style enum value is its discriminant).
                if let Some(disc) = self.enum_variant_value(base, name.sym) {
                    self.builder
                        .build(MirType::I32, Op::ConstInt(disc as i128, MirType::I32))
                } else {
                    let (ptr, fmty) = self.struct_field_place(base, name.sym);
                    self.load_or_addr(ptr, fmty)
                }
            }
            // A struct literal in value position materializes a fresh byte buffer, yielding its base
            // pointer (the same by-pointer convention as arrays/tuples).
            ExprKind::StructLit { fields, .. } => {
                let ty = self.expr_ty(e);
                let size = self.mir_ty_of(&ty);
                let buf = self.builder.alloca(size);
                if let Ty::Named(sym) = ty {
                    self.lower_struct_init(buf, sym, fields, e.span);
                }
                buf
            }
            // A tuple literal in value position (a call argument, a nested field) materializes a
            // fresh byte buffer and yields its base pointer — the same by-pointer convention an
            // array value follows.
            ExprKind::TupleLit(items) => {
                let tty = self.expr_ty(e);
                let size = self.ty_size(&tty).unwrap_or(0) as u32;
                let buf = self
                    .builder
                    .alloca(MirType::Array(Box::new(MirType::I8), size));
                self.lower_tuple_init(buf, &tty, items);
                buf
            }
            ExprKind::Cast { expr, .. } => self.lower_cast(expr, e),
            ExprKind::Block(b) => match self.lower_block(b) {
                Some(v) => v,
                None => {
                    let t = self.expr_mir(e);
                    self.const_zero(t)
                }
            },
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => self.lower_if_value(cond, then_branch, else_branch.as_deref(), e),
            ExprKind::Match { scrutinee, arms } => self.lower_match(scrutinee, arms, e),
            _ => {
                self.unsupported(e.span, "expression");
                let t = self.expr_mir(e);
                self.const_zero(t)
            }
        }
    }

    /// Lower a `match` expression (or statement) to an if-else chain over the arms. The scrutinee is
    /// evaluated once; each arm in turn tests the scrutinee against its pattern (a literal compares
    /// for equality; a wildcard/identifier always matches) and, if present, its guard, branching to
    /// the arm body or the next test. An `Ident` pattern binds the scrutinee value in the arm scope.
    /// When the `match` is used as a value, a merge-block parameter collects each arm body's result.
    /// An unconditional catch-all arm (a bare `_`/identifier with no guard) ends the chain; if none is
    /// present the structurally-emitted fallthrough is `Unreachable` (sema's E0405 rejects any
    /// value-producing non-exhaustive match, so it is dynamically dead).
    fn lower_match(&mut self, scrutinee: &Expr, arms: &[ast::MatchArm], e: &Expr) -> ValueId {
        let result_ty = self.expr_mir(e);
        let produces_value = result_ty != MirType::Void;
        // An aggregate result flows as its base pointer, so the merge param is `Ptr` (see
        // `merge_repr_ty`); scalars are unchanged.
        let merge_ty = merge_repr_ty(&result_ty);
        let scrut_mir = self.expr_mir(scrutinee);
        let scrut_ty = self.expr_ty(scrutinee);
        let scrut = self.lower_expr(scrutinee);

        let merge = self.builder.new_block();
        let merge_param = if produces_value {
            Some(self.builder.block_param(merge, merge_ty.clone()))
        } else {
            None
        };

        // Branch to `merge` with the arm body's value (or no arg for a unit match).
        let mut handled_default = false;
        for arm in arms {
            let unconditional = matches!(
                &arm.pat.kind,
                ast::PatKind::Wildcard | ast::PatKind::Ident(_) | ast::PatKind::Unit
            ) && arm.guard.is_none();

            if unconditional {
                // Always matches: lower the body directly, then the remaining arms are unreachable.
                self.push_scope();
                self.bind_match_ident(&arm.pat, scrut, &scrut_mir, &scrut_ty);
                self.emit_match_arm_body(&arm.body, merge, merge_param, &merge_ty);
                self.pop_scope();
                handled_default = true;
                break;
            }

            let body_bb = self.builder.new_block();
            let next_bb = self.builder.new_block();
            // Bind first so an `Ident` pattern's guard can reference the binding; the binding is
            // scoped to this arm (popped after the body).
            self.push_scope();
            self.bind_match_ident(&arm.pat, scrut, &scrut_mir, &scrut_ty);
            let cond =
                self.match_arm_cond(&arm.pat, scrut, &scrut_mir, &scrut_ty, arm.guard.as_ref());
            self.builder
                .cond_br(cond, body_bb, vec![], next_bb, vec![]);

            self.builder.switch_to(body_bb);
            self.terminated = false;
            self.emit_match_arm_body(&arm.body, merge, merge_param, &merge_ty);
            self.pop_scope();

            self.builder.switch_to(next_bb);
            self.terminated = false;
        }

        // No arm matched. With an unconditional catch-all this block is never emitted; without one it
        // is reached only when no arm's pattern fires. Sema's exhaustiveness check (E0405) rejects any
        // *value-producing* non-exhaustive match before lowering, so for a value match this point is
        // dynamically dead (an exhaustive enum/bool match with no `_` still emits this block
        // structurally — every arm is a conditional test — but one arm always matches at runtime).
        // Emit `Unreachable` rather than a zero default: a scalar zero was a silent wrong answer and
        // an aggregate zero was invalid MIR (`const.i32 0` into an aggregate merge param → an ICE the
        // verifier/`mem2reg` rejected). A unit/statement match has no value to merge and just falls
        // through. (A genuinely value-producing match always has a value-producing arm that branches
        // to `merge`, so the merge param never lacks a provider.)
        if !handled_default && !self.terminated {
            match merge_param {
                Some(_) => self
                    .builder
                    .set_term(mercury_mir::Terminator::Unreachable),
                None => self.builder.br(merge, vec![]),
            }
        }

        self.builder.switch_to(merge);
        self.terminated = false;
        match merge_param {
            Some(p) => p,
            None => self.const_zero(if result_ty == MirType::Void {
                MirType::I32
            } else {
                merge_ty
            }),
        }
    }

    /// Bind a match pattern's identifiers to the scrutinee for the arm's scope: a scalar `Ident` is
    /// stored into a fresh slot (so a `Path` read loads it, like a param); an aggregate `Ident` binds
    /// its base pointer directly. A `Tuple` pattern recurses into each field's place (so
    /// `(x, y) => x + y` binds `x`/`y` to the tuple's fields). Literal / wildcard / unit bind nothing.
    fn bind_match_ident(
        &mut self,
        pat: &Pattern,
        scrut: ValueId,
        scrut_mir: &MirType,
        scrut_ty: &Ty,
    ) {
        match &pat.kind {
            ast::PatKind::Ident(name) => {
                if matches!(scrut_mir, MirType::Array(..)) {
                    self.bind(*name, scrut, scrut_mir.clone());
                } else {
                    let slot = self.builder.alloca(scrut_mir.clone());
                    self.builder.build_void(Op::Store {
                        ptr: slot,
                        value: scrut,
                    });
                    self.bind(*name, slot, scrut_mir.clone());
                }
            }
            ast::PatKind::Tuple(subs) => self.bind_tuple_match(subs, scrut, scrut_ty),
            _ => {}
        }
    }

    /// Bind the identifiers of a tuple pattern to their field places within the tuple buffer at base
    /// pointer `base` (sema type `ty`). A scalar field `Ident` is copied into a fresh slot (so reads
    /// load it and a mutated binding doesn't write back into the scrutinee); an aggregate field
    /// `Ident` binds the field address (the array/by-pointer convention); a nested tuple pattern
    /// recurses; a literal/wildcard sub-pattern binds nothing.
    fn bind_tuple_match(&mut self, subs: &[Pattern], base: ValueId, ty: &Ty) {
        let Ty::Tuple(ftys) = ty else { return };
        let Some((offsets, _, _)) = self.aggregate_layout(ftys) else {
            return;
        };
        for (i, sub) in subs.iter().enumerate() {
            let (Some(&off), Some(fty)) = (offsets.get(i), ftys.get(i)) else {
                continue;
            };
            let fmty = self.mir_ty_of(fty);
            let fptr = self.field_ptr(base, off);
            match &sub.kind {
                ast::PatKind::Ident(name) => {
                    if matches!(fmty, MirType::Array(..)) {
                        self.bind(*name, fptr, fmty);
                    } else {
                        let val = self.builder.build(fmty.clone(), Op::Load(fptr, fmty.clone()));
                        let slot = self.builder.alloca(fmty.clone());
                        self.builder.build_void(Op::Store { ptr: slot, value: val });
                        self.bind(*name, slot, fmty);
                    }
                }
                ast::PatKind::Tuple(inner) => self.bind_tuple_match(inner, fptr, fty),
                _ => {}
            }
        }
    }

    /// The i1 condition under which a match arm fires: the pattern test (a literal compares equal; a
    /// wildcard/identifier is always true) conjoined with the optional guard. The guard is lowered in
    /// the current (test) block, after any `Ident` binding, so it may reference the binding.
    fn match_arm_cond(
        &mut self,
        pat: &Pattern,
        scrut: ValueId,
        scrut_mir: &MirType,
        scrut_ty: &Ty,
        guard: Option<&Expr>,
    ) -> ValueId {
        let pat_cond = self.pattern_cond(pat, scrut, scrut_mir, scrut_ty, pat.span);
        // The guard is normalized to `i1` like every other boolean condition (`if`/`while`/`&&`):
        // a non-bool guard (`match k { x if x => .. }`) would otherwise feed its raw `i32` into the
        // arm's `And` / `cond_br`, MIR the verifier and Cranelift reject (the `lower_bool_cond` fix
        // covered if/while/short-circuit/assert but not the match-guard path).
        match (pat_cond, guard) {
            (Some(pc), Some(g)) => {
                let gv = self.lower_bool_cond(g);
                self.builder.build(MirType::I1, Op::Bin(BinOp::And, pc, gv))
            }
            (Some(pc), None) => pc,
            (None, Some(g)) => self.lower_bool_cond(g),
            (None, None) => self.builder.build(MirType::I1, Op::ConstInt(1, MirType::I1)),
        }
    }

    /// The i1 condition under which `pat` matches the scrutinee. For a scalar pattern `scrut` is the
    /// loaded scrutinee value; for a tuple pattern it is the tuple buffer's base pointer. `None` means
    /// the pattern is unconditional (a wildcard / identifier binding). Literal int/bool/enum-variant/
    /// range patterns compare the value; a tuple ANDs its fields; an or-pattern ORs its alternatives.
    fn pattern_cond(
        &mut self,
        pat: &Pattern,
        scrut: ValueId,
        scrut_mir: &MirType,
        scrut_ty: &Ty,
        span: Span,
    ) -> Option<ValueId> {
        match &pat.kind {
            ast::PatKind::Wildcard | ast::PatKind::Ident(_) | ast::PatKind::Unit => None,
            ast::PatKind::Int { sym, neg } => {
                let mut v = parse_int(self.interner.resolve(*sym));
                if *neg {
                    v = -v;
                }
                let c = self
                    .builder
                    .build(scrut_mir.clone(), Op::ConstInt(v, scrut_mir.clone()));
                Some(self.builder.build(MirType::I1, Op::Cmp(CmpOp::Eq, scrut, c)))
            }
            // A char-literal pattern compares the scrutinee (a `char` is its integer code point) to
            // the literal's decoded code point — the same equality test as an integer-literal pattern.
            ast::PatKind::Char(sym) => {
                let v = decode_char_literal(self.interner.resolve(*sym)) as i128;
                let c = self
                    .builder
                    .build(scrut_mir.clone(), Op::ConstInt(v, scrut_mir.clone()));
                Some(self.builder.build(MirType::I1, Op::Cmp(CmpOp::Eq, scrut, c)))
            }
            ast::PatKind::Bool(b) => {
                let c = self
                    .builder
                    .build(MirType::I1, Op::ConstInt(*b as i128, MirType::I1));
                Some(self.builder.build(MirType::I1, Op::Cmp(CmpOp::Eq, scrut, c)))
            }
            // `Enum::Variant` — compare the scrutinee (an enum value is its discriminant) to the
            // variant's discriminant. An unresolved path is rejected (a hard error, never a no-op).
            ast::PatKind::Path(path) => {
                let Some(disc) = self.enum_path_value(path) else {
                    self.unsupported(span, "match pattern");
                    return Some(self.builder.build(MirType::I1, Op::ConstInt(0, MirType::I1)));
                };
                let c = self.builder.build(
                    scrut_mir.clone(),
                    Op::ConstInt(disc as i128, scrut_mir.clone()),
                );
                Some(self.builder.build(MirType::I1, Op::Cmp(CmpOp::Eq, scrut, c)))
            }
            // A range pattern `lo..hi` / `lo..=hi`: `lo <= scrut` AND `scrut < hi` (or `<= hi`),
            // with the comparison signedness taken from the scrutinee's type.
            ast::PatKind::Range {
                lo,
                hi,
                inclusive,
            } => {
                let signed = !matches!(scrut_ty, Ty::Scalar(s) if !s.is_signed());
                let lo_v = self.pattern_int_value(lo);
                let hi_v = self.pattern_int_value(hi);
                let lo_c = self
                    .builder
                    .build(scrut_mir.clone(), Op::ConstInt(lo_v, scrut_mir.clone()));
                let hi_c = self
                    .builder
                    .build(scrut_mir.clone(), Op::ConstInt(hi_v, scrut_mir.clone()));
                let ge = self.builder.build(
                    MirType::I1,
                    Op::Cmp(if signed { CmpOp::Sge } else { CmpOp::Uge }, scrut, lo_c),
                );
                let hi_op = match (*inclusive, signed) {
                    (true, true) => CmpOp::Sle,
                    (false, true) => CmpOp::Slt,
                    (true, false) => CmpOp::Ule,
                    (false, false) => CmpOp::Ult,
                };
                let lt = self.builder.build(MirType::I1, Op::Cmp(hi_op, scrut, hi_c));
                Some(self.builder.build(MirType::I1, Op::Bin(BinOp::And, ge, lt)))
            }
            // `scrut` is the tuple buffer's base pointer; AND each field's sub-pattern test.
            ast::PatKind::Tuple(subs) => self.tuple_pattern_cond(subs, scrut, scrut_ty, span),
            // An or-pattern matches if any alternative does: OR each alternative's condition. An
            // unconditional alternative (a wildcard/ident) makes the whole or-pattern unconditional.
            ast::PatKind::Or(alts) => {
                let mut acc: Option<ValueId> = None;
                for alt in alts {
                    match self.pattern_cond(alt, scrut, scrut_mir, scrut_ty, alt.span) {
                        None => return None,
                        Some(c) => {
                            acc = Some(match acc {
                                Some(a) => {
                                    self.builder.build(MirType::I1, Op::Bin(BinOp::Or, a, c))
                                }
                                None => c,
                            });
                        }
                    }
                }
                acc
            }
        }
    }

    /// The integer value of an int- or char-literal range bound (`lo`/`hi`). A char decodes to its
    /// code point, so `'a'..='z'` ranges work. A non-literal bound yields 0.
    fn pattern_int_value(&self, pat: &Pattern) -> i128 {
        match &pat.kind {
            ast::PatKind::Int { sym, neg } => {
                let v = parse_int(self.interner.resolve(*sym));
                if *neg {
                    -v
                } else {
                    v
                }
            }
            ast::PatKind::Char(sym) => decode_char_literal(self.interner.resolve(*sym)) as i128,
            _ => 0,
        }
    }

    /// Resolve an enum-variant path pattern (`Enum::Variant`) to its integer discriminant.
    fn enum_path_value(&self, path: &ast::Path) -> Option<i64> {
        if path.segments.len() != 2 {
            return None;
        }
        let DefKind::Enum(variants) = &self.sema.defs.lookup(path.segments[0].sym)?.kind else {
            return None;
        };
        let var = path.segments[1].sym;
        variants.iter().find(|(v, _)| *v == var).map(|(_, d)| *d)
    }

    /// Whether a pattern must be tested against the scrutinee's *address* (a tuple field that is
    /// itself an aggregate) rather than its loaded value.
    fn pattern_needs_ptr(pat: &Pattern) -> bool {
        match &pat.kind {
            ast::PatKind::Tuple(_) => true,
            ast::PatKind::Or(alts) => alts.iter().any(Self::pattern_needs_ptr),
            _ => false,
        }
    }

    /// The i1 condition under which a tuple pattern matches the tuple at base pointer `base` (sema
    /// type `ty`): the AND of each field sub-pattern's condition. `None` (every field unconditional)
    /// means the whole tuple matches unconditionally. If the tuple can't be laid out (a non-tuple type
    /// or an unsizeable field) the arm is rejected with `unsupported` and a `false` condition — never
    /// a silent over-match, which would violate the differential-correctness invariant.
    fn tuple_pattern_cond(
        &mut self,
        subs: &[Pattern],
        base: ValueId,
        ty: &Ty,
        span: Span,
    ) -> Option<ValueId> {
        let layout = match ty {
            Ty::Tuple(ftys) => self
                .aggregate_layout(ftys)
                .map(|(offs, _, _)| (ftys.clone(), offs)),
            _ => None,
        };
        let Some((ftys, offsets)) = layout else {
            self.unsupported(span, "tuple pattern");
            return Some(self.builder.build(MirType::I1, Op::ConstInt(0, MirType::I1)));
        };
        let mut acc: Option<ValueId> = None;
        for (i, sub) in subs.iter().enumerate() {
            let (Some(&off), Some(fty)) = (offsets.get(i), ftys.get(i)) else {
                continue;
            };
            let fmty = self.mir_ty_of(fty);
            let fptr = self.field_ptr(base, off);
            // A nested aggregate sub-pattern tests against the field address; a scalar sub-pattern
            // tests against the loaded field value.
            let field_scrut = if Self::pattern_needs_ptr(sub) {
                fptr
            } else {
                self.builder.build(fmty.clone(), Op::Load(fptr, fmty.clone()))
            };
            if let Some(c) = self.pattern_cond(sub, field_scrut, &fmty, fty, span) {
                acc = Some(match acc {
                    Some(a) => self.builder.build(MirType::I1, Op::Bin(BinOp::And, a, c)),
                    None => c,
                });
            }
        }
        acc
    }

    /// Lower a match arm body and, unless it diverged, branch to `merge` passing the body value when
    /// the match produces one.
    fn emit_match_arm_body(
        &mut self,
        body: &Expr,
        merge: mercury_mir::BlockId,
        merge_param: Option<ValueId>,
        result_ty: &MirType,
    ) {
        let bv = self.lower_expr(body);
        if !self.terminated {
            let args = if merge_param.is_some() {
                // Coerce the arm value to the match's merged result type, so every arm passes the
                // merge param the *same* MIR type. Without this, arms of different width/kind
                // (`match n { 0 => 10, _ => 2.5 }`) pass a mismatched value the native verifier
                // rejects while the interpreter runs loosely — a backend divergence. A unit-typed
                // body with a value-producing match yields a zero so the edge arity still matches.
                let from = self.expr_mir(body);
                if matches!(from, MirType::Void) {
                    vec![self.const_zero(result_ty.clone())]
                } else {
                    vec![self.coerce_to(bv, &from, result_ty, self.signed(body))]
                }
            } else {
                vec![]
            };
            self.builder.br(merge, args);
        }
    }

    fn lower_unary(&mut self, op: ast::UnOp, operand: &Expr, e: &Expr) -> ValueId {
        match op {
            ast::UnOp::Neg => {
                let v = self.lower_expr(operand);
                let ty = self.expr_mir(e);
                self.builder.build(ty, Op::Neg(v))
            }
            ast::UnOp::Not => {
                let v = self.lower_expr(operand);
                let ty = self.expr_mir(e);
                self.builder.build(ty, Op::Not(v))
            }
            ast::UnOp::Deref => {
                let ptr = self.lower_expr(operand);
                let ty = self.expr_mir(e);
                self.builder.build(ty.clone(), Op::Load(ptr, ty))
            }
            ast::UnOp::Ref | ast::UnOp::RefMut => {
                let (ptr, _) = self.lower_place(operand);
                ptr
            }
        }
    }

    fn lower_binary(&mut self, op: ast::BinOp, lhs: &Expr, rhs: &Expr, e: &Expr) -> ValueId {
        use ast::BinOp::*;
        match op {
            Eq | Ne | Lt | Le | Gt | Ge => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                // Compare in a common type: the front-end's loose literal typing can leave the two
                // sides at different widths, but a `Cmp`'s operands must agree.
                let lty = self.expr_mir(lhs);
                let rty = self.expr_mir(rhs);
                let common = numeric_join(&lty, &rty);
                let l = self.coerce_to(l, &lty, &common, self.signed(lhs));
                let r = self.coerce_to(r, &rty, &common, self.signed(rhs));
                let pred = cmp_pred(op, common.is_float(), self.signed(lhs));
                self.builder.build(MirType::I1, Op::Cmp(pred, l, r))
            }
            And => self.lower_short_circuit(lhs, rhs, true),
            Or => self.lower_short_circuit(lhs, rhs, false),
            _ => {
                let ty = self.expr_mir(e);
                // Contract a float `x + y*z` into one fused multiply-add before falling back to a
                // plain `Bin`.
                if let Some(v) = self.try_contract_fma(op, lhs, rhs, &ty) {
                    return v;
                }
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                // Coerce both operands to the result type so the `Bin` is well-typed (e.g. an
                // `f32` literal added to an `f64` is promoted), matching the verifier's contract.
                let lty = self.expr_mir(lhs);
                let rty = self.expr_mir(rhs);
                let l = self.coerce_to(l, &lty, &ty, self.signed(lhs));
                let r = self.coerce_to(r, &rty, &ty, self.signed(rhs));
                let bin = arith_binop(op, ty.is_float(), self.signed(lhs));
                self.builder.build(ty, Op::Bin(bin, l, r))
            }
        }
    }

    /// Short-circuit `&&` / `||`: the RHS is evaluated only when the LHS doesn't already decide the
    /// result. `a && b` ≡ `if a { b } else { false }`; `a || b` ≡ `if a { true } else { b }`. Lowered
    /// to a branch + a merge block param — NOT a bitwise `and`/`or` of both operands — so a
    /// side-effecting or unsafe RHS (`p_in_bounds && load(p)`) does not run when the LHS already
    /// settles it. Both backends execute the identical CFG, so the differential gate holds.
    fn lower_short_circuit(&mut self, lhs: &Expr, rhs: &Expr, is_and: bool) -> ValueId {
        // Both operands are conditions: normalize a float operand to `!= 0.0` so the LHS reaches
        // `cond_br` as an `i1` (not a raw float native's `brif` rejects) and the RHS matches the
        // `i1` merge param. The integer/`i1` path is unchanged.
        let l = self.lower_bool_cond(lhs);
        let rhs_bb = self.builder.new_block();
        let merge = self.builder.new_block();
        let res = self.builder.block_param(merge, MirType::I1);
        // The short-circuit value passed to `merge` when the LHS decides it: `false` for `&&` (LHS
        // false), `true` for `||` (LHS true). Built in the current (predecessor) block.
        let short = self.builder.build(
            MirType::I1,
            Op::ConstInt(if is_and { 0 } else { 1 }, MirType::I1),
        );
        if is_and {
            // LHS true → evaluate RHS; LHS false → merge(false).
            self.builder.cond_br(l, rhs_bb, vec![], merge, vec![short]);
        } else {
            // LHS true → merge(true); LHS false → evaluate RHS.
            self.builder.cond_br(l, merge, vec![short], rhs_bb, vec![]);
        }
        self.builder.switch_to(rhs_bb);
        self.terminated = false;
        let r = self.lower_bool_cond(rhs);
        if !self.terminated {
            self.builder.br(merge, vec![r]);
        }
        self.builder.switch_to(merge);
        self.terminated = false;
        res
    }

    /// Contract a float `x + y*z` (or `y*z + x`) into one fused multiply-add. FMA rounds once
    /// instead of twice — faster (a single `vfmadd`) and more accurate — and the interpreter
    /// mirrors it with `mul_add`, so the native and interpreter backends stay bit-identical.
    /// Returns `None` when the shape or types don't permit it; the caller then emits a plain add.
    fn try_contract_fma(
        &mut self,
        op: ast::BinOp,
        lhs: &Expr,
        rhs: &Expr,
        ty: &MirType,
    ) -> Option<ValueId> {
        if op != ast::BinOp::Add || !ty.is_float() {
            return None;
        }
        // A local `fn` (not a closure) so the borrow of the returned sub-exprs ties to the
        // argument's lifetime rather than a single inferred one.
        fn as_fmul(e: &Expr) -> Option<(&Expr, &Expr)> {
            match &e.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Mul,
                    lhs,
                    rhs,
                } => Some((lhs.as_ref(), rhs.as_ref())),
                _ => None,
            }
        }
        // Lower operands in source order so any side effects keep their original sequencing.
        if let Some((y, z)) = as_fmul(lhs) {
            let yv = self.lower_fma_operand(y, ty);
            let zv = self.lower_fma_operand(z, ty);
            let xv = self.lower_fma_operand(rhs, ty);
            return Some(self.builder.build(ty.clone(), Op::Fma(yv, zv, xv)));
        }
        if let Some((y, z)) = as_fmul(rhs) {
            let xv = self.lower_fma_operand(lhs, ty);
            let yv = self.lower_fma_operand(y, ty);
            let zv = self.lower_fma_operand(z, ty);
            return Some(self.builder.build(ty.clone(), Op::Fma(yv, zv, xv)));
        }
        None
    }

    /// Lower an FMA operand and coerce it to the (float) result type.
    fn lower_fma_operand(&mut self, e: &Expr, ty: &MirType) -> ValueId {
        let v = self.lower_expr(e);
        let ety = self.expr_mir(e);
        self.coerce_to(v, &ety, ty, self.signed(e))
    }

    /// Insert a numeric cast so `v` (currently `from`) has type `to`. Non-numeric operands and
    /// equal types pass through unchanged.
    fn coerce_to(&mut self, v: ValueId, from: &MirType, to: &MirType, signed: bool) -> ValueId {
        if from == to || !is_numeric(from) || !is_numeric(to) {
            return v;
        }
        let kind = cast_kind(from, to, signed);
        self.builder
            .build(to.clone(), Op::Cast(kind, v, to.clone()))
    }

    fn lower_call(&mut self, callee: &Expr, args: &[Expr], e: &Expr) -> ValueId {
        if let ExprKind::Path(p) = &callee.kind {
            if p.is_single() {
                let name = p.first().sym;
                if matches!(
                    self.sema.defs.lookup(name).map(|d| &d.kind),
                    Some(DefKind::Fn(_))
                ) {
                    let argvals: Vec<ValueId> = args.iter().map(|a| self.lower_expr(a)).collect();
                    let ret = self.expr_mir(e);
                    if matches!(ret, MirType::Array(..)) {
                        // The callee returns an aggregate by value (sret ABI): allocate the
                        // destination buffer here, pass it as the hidden leading argument, and yield
                        // it as the call's value (the by-pointer aggregate convention — a further
                        // `.field`/`[i]` GEPs off it).
                        let dst = self.builder.alloca(ret);
                        let mut call_args = Vec::with_capacity(argvals.len() + 1);
                        call_args.push(dst);
                        call_args.extend(argvals);
                        self.builder.build_void(Op::Call {
                            func: name,
                            args: call_args,
                        });
                        return dst;
                    }
                    if ret == MirType::Void {
                        self.builder.build_void(Op::Call {
                            func: name,
                            args: argvals,
                        });
                        return self.const_zero(MirType::I32);
                    }
                    return self.builder.build(
                        ret,
                        Op::Call {
                            func: name,
                            args: argvals,
                        },
                    );
                }
                // Math builtins (sqrt/rsqrt/exp/fmax/fmin) lower to primitive ops. User functions
                // shadow them (handled just above), so this catches only the genuine builtins.
                if let Some(v) = self.lower_math_intrinsic(name, args, e) {
                    return v;
                }
                // Built-in intrinsics (print, ...) lower to a void call the interpreter handles.
                if is_intrinsic(self.interner.resolve(name)) {
                    // A `*u8` (string) argument to `print`/`println` renders its bytes rather than the
                    // pointer value: route it to the dedicated `print_str`/`println_str` symbol, which
                    // both backends read as a null-terminated buffer. (Printing a raw pointer as a
                    // number is already non-differential — the interpreter prints a slot index, native
                    // a real address — so no well-formed program loses behavior here.)
                    let nm = self.interner.resolve(name);
                    if (nm == "print" || nm == "println")
                        && args.len() == 1
                        && self.is_string_arg(&args[0])
                    {
                        let s = self.lower_expr(&args[0]);
                        let func = if nm == "println" {
                            self.gemm.println_str
                        } else {
                            self.gemm.print_str
                        };
                        self.builder.build_void(Op::Call {
                            func,
                            args: vec![s],
                        });
                        return self.const_zero(MirType::I32);
                    }
                    // An *unsigned* integer argument must format as unsigned: the value is stored
                    // sign-extended, so a high-bit-set `u32`/`u64` (a quantization scale, a hash, a
                    // size) would print as its negative two's-complement reinterpretation under the
                    // default signed `print`. Zero-extend to 64 bits (clearing the high bits of a
                    // narrow value; a no-op for `u64`/`usize`) and route to `print_u`/`println_u`,
                    // which render the bits as `u64`. Both backends marshal the same symbol.
                    if (nm == "print" || nm == "println")
                        && args.len() == 1
                        && self.is_unsigned_int_arg(&args[0])
                    {
                        let v0 = self.lower_expr(&args[0]);
                        let from = self.expr_mir(&args[0]);
                        let v = self.coerce_to(v0, &from, &MirType::I64, false);
                        let func = if nm == "println" {
                            self.gemm.println_u
                        } else {
                            self.gemm.print_u
                        };
                        self.builder.build_void(Op::Call {
                            func,
                            args: vec![v],
                        });
                        return self.const_zero(MirType::I32);
                    }
                    // `assert(cond)` takes a boolean condition: normalize a float argument to
                    // `cond != 0.0` (an `i1`) so a fractional `assert(0.5)` is *true* on both
                    // backends, rather than native truncating `0.5 -> 0` and trapping while the
                    // interpreter's `f != 0.0` passes. The integer/`i1` path is unchanged.
                    if nm == "assert" && args.len() == 1 {
                        let c = self.lower_bool_cond(&args[0]);
                        self.builder.build_void(Op::Call {
                            func: name,
                            args: vec![c],
                        });
                        return self.const_zero(MirType::I32);
                    }
                    let argvals: Vec<ValueId> = args.iter().map(|a| self.lower_expr(a)).collect();
                    self.builder.build_void(Op::Call {
                        func: name,
                        args: argvals,
                    });
                    return self.const_zero(MirType::I32);
                }
            }
        }
        // Unmodeled builtin/method call.
        for a in args {
            self.lower_expr(a);
        }
        self.unsupported(e.span, "call");
        let t = self.expr_mir(e);
        self.const_zero(t)
    }

    /// Lower a math builtin (`sqrt`/`rsqrt`/`exp`/`fmax`/`fmin`) to primitive MIR ops, or `None`
    /// for any other name. `sqrt` is one `Op::Sqrt`; `rsqrt` is its reciprocal; `fmax`/`fmin` are a
    /// compare + select (NaN- and signed-zero-correct, identical in both backends); `exp` is a
    /// polynomial (see `emit_exp`). Built only from already-bit-exact primitives, so the
    /// interpreter and native backend agree with no separate hand-written implementation each.
    fn lower_math_intrinsic(&mut self, name: Symbol, args: &[Expr], e: &Expr) -> Option<ValueId> {
        let op = math_intrinsic(self.interner.resolve(name))?;
        let rty = self.expr_mir(e);
        // `abs`/`round`/`floor`/`ceil`/`trunc` are type-preserving on integers (sema types them as
        // the argument's integer type): integer abs is `select(x < 0, -x, x)`; rounding an integer is
        // the identity. Without this they would emit float ops on an int operand — MIR the verifier
        // rejects on the native backend while the interpreter silently ran it (lossily, via f32).
        if rty.is_int() {
            match op {
                MathIntrinsic::Abs => {
                    let v = self.lower_expr(args.first()?);
                    return Some(self.emit_int_abs(v, &rty));
                }
                MathIntrinsic::Round
                | MathIntrinsic::Floor
                | MathIntrinsic::Ceil
                | MathIntrinsic::Trunc => return Some(self.lower_expr(args.first()?)),
                _ => {}
            }
        }
        match op {
            MathIntrinsic::Sqrt => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.builder.build(rty.clone(), Op::Sqrt(x)))
            }
            MathIntrinsic::Rsqrt => {
                let x = self.lower_math_arg(args.first()?, &rty);
                let s = self.builder.build(rty.clone(), Op::Sqrt(x));
                let one = self.splat_const_f(1.0, &rty);
                Some(
                    self.builder
                        .build(rty.clone(), Op::Bin(BinOp::FDiv, one, s)),
                )
            }
            MathIntrinsic::Abs => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_abs(x, &rty))
            }
            MathIntrinsic::Round
            | MathIntrinsic::Floor
            | MathIntrinsic::Ceil
            | MathIntrinsic::Trunc => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(
                    self.builder
                        .build(rty.clone(), Op::Round(round_mode(op), x)),
                )
            }
            MathIntrinsic::Exp => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_exp(x, &rty))
            }
            MathIntrinsic::Log => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_log(x, &rty))
            }
            // exp2(x)=exp(x·ln2), log2(x)=log(x)·log2(e), sinh/cosh=(eˣ∓e⁻ˣ)/2 — composed from the
            // shared exp/log so they vectorize and stay bit-exact across backends; the dispatched
            // 256-bit kernel mirrors this op-for-op.
            MathIntrinsic::Exp2 => {
                let x = self.lower_math_arg(args.first()?, &rty);
                let ln2 = self.splat_const_f(std::f64::consts::LN_2, &rty);
                let xl = self
                    .builder
                    .build(rty.clone(), Op::Bin(BinOp::FMul, x, ln2));
                Some(self.emit_exp(xl, &rty))
            }
            MathIntrinsic::Log2 => {
                let x = self.lower_math_arg(args.first()?, &rty);
                let lx = self.emit_log(x, &rty);
                let log2e = self.splat_const_f(std::f64::consts::LOG2_E, &rty);
                Some(
                    self.builder
                        .build(rty.clone(), Op::Bin(BinOp::FMul, lx, log2e)),
                )
            }
            MathIntrinsic::Exp10 => {
                let x = self.lower_math_arg(args.first()?, &rty);
                let ln10 = self.splat_const_f(std::f64::consts::LN_10, &rty);
                let xl = self
                    .builder
                    .build(rty.clone(), Op::Bin(BinOp::FMul, x, ln10));
                Some(self.emit_exp(xl, &rty))
            }
            MathIntrinsic::Log10 => {
                let x = self.lower_math_arg(args.first()?, &rty);
                let lx = self.emit_log(x, &rty);
                let log10e = self.splat_const_f(std::f64::consts::LOG10_E, &rty);
                Some(
                    self.builder
                        .build(rty.clone(), Op::Bin(BinOp::FMul, lx, log10e)),
                )
            }
            MathIntrinsic::Sinh | MathIntrinsic::Cosh => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_sinh_cosh(x, &rty, matches!(op, MathIntrinsic::Cosh)))
            }
            // asinh = sign(x)·log(|x|+√(x²+1)),  acosh = log(x+√(x²−1)),
            // atanh = ½·log((1+x)/(1−x)) — composed from the shared `log` (and `√`), so they vectorize
            // and the dispatched 256-bit kernel mirrors this op-for-op.
            MathIntrinsic::Asinh => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_asinh(x, &rty))
            }
            MathIntrinsic::Acosh => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_acosh(x, &rty))
            }
            MathIntrinsic::Atanh => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_atanh(x, &rty))
            }
            MathIntrinsic::Atan => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_atan(x, &rty))
            }
            MathIntrinsic::Expm1 => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_expm1(x, &rty))
            }
            MathIntrinsic::Log1p => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_log1p(x, &rty))
            }
            MathIntrinsic::Pow => {
                // pow(x, y) = exp(y * log(x)), reusing the two polynomials (so it vectorizes and is
                // bit-exact across backends for free). Defined for x > 0, like the rest of the suite.
                if args.len() != 2 {
                    return None;
                }
                let x = self.lower_math_arg(&args[0], &rty);
                let y = self.lower_math_arg(&args[1], &rty);
                let lx = self.emit_log(x, &rty);
                let ylx = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, y, lx));
                Some(self.emit_exp(ylx, &rty))
            }
            MathIntrinsic::Atan2 => {
                if args.len() != 2 {
                    return None;
                }
                let y = self.lower_math_arg(&args[0], &rty);
                let x = self.lower_math_arg(&args[1], &rty);
                Some(self.emit_atan2(y, x, &rty))
            }
            MathIntrinsic::Hypot => {
                if args.len() != 2 {
                    return None;
                }
                let a = self.lower_math_arg(&args[0], &rty);
                let b = self.lower_math_arg(&args[1], &rty);
                Some(self.emit_hypot(a, b, &rty))
            }
            MathIntrinsic::Erf => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_erf(x, &rty))
            }
            MathIntrinsic::Sin => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_trig(x, &rty, false))
            }
            MathIntrinsic::Cos => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_trig(x, &rty, true))
            }
            MathIntrinsic::Tanh => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_tanh(x, &rty))
            }
            MathIntrinsic::Sigmoid => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_sigmoid(x, &rty))
            }
            MathIntrinsic::Silu => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_silu(x, &rty))
            }
            MathIntrinsic::Gelu => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_gelu(x, &rty))
            }
            // Activation backward `act_backward(x, dy) = dy · act'(x)` — two args. The non-dispatched
            // path (a `while` loop / standalone call); the elementwise `for` form goes 256-bit via
            // `match_vmath2_stmt`. Inlined form is bit-identical to the kernel (mirrors `*_bwd8`).
            MathIntrinsic::SiluBackward => {
                if args.len() != 2 {
                    return None;
                }
                let x = self.lower_expr(&args[0]);
                let dy = self.lower_expr(&args[1]);
                Some(self.emit_silu_backward(x, dy, &rty))
            }
            MathIntrinsic::GeluBackward => {
                if args.len() != 2 {
                    return None;
                }
                let x = self.lower_expr(&args[0]);
                let dy = self.lower_expr(&args[1]);
                Some(self.emit_gelu_backward(x, dy, &rty))
            }
            MathIntrinsic::SigmoidBackward => {
                if args.len() != 2 {
                    return None;
                }
                let x = self.lower_expr(&args[0]);
                let dy = self.lower_expr(&args[1]);
                Some(self.emit_sigmoid_backward(x, dy, &rty))
            }
            MathIntrinsic::TanhBackward => {
                if args.len() != 2 {
                    return None;
                }
                let x = self.lower_expr(&args[0]);
                let dy = self.lower_expr(&args[1]);
                Some(self.emit_tanh_backward(x, dy, &rty))
            }
            MathIntrinsic::EluBackward => {
                if args.len() != 2 {
                    return None;
                }
                let x = self.lower_expr(&args[0]);
                let dy = self.lower_expr(&args[1]);
                Some(self.emit_elu_backward(x, dy, &rty))
            }
            MathIntrinsic::SoftplusBackward => {
                if args.len() != 2 {
                    return None;
                }
                let x = self.lower_expr(&args[0]);
                let dy = self.lower_expr(&args[1]);
                Some(self.emit_softplus_backward(x, dy, &rty))
            }
            MathIntrinsic::Elu => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_elu(x, &rty))
            }
            MathIntrinsic::LeakyRelu => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_leaky_relu(x, &rty))
            }
            MathIntrinsic::Softplus => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_softplus(x, &rty))
            }
            MathIntrinsic::Mish => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_mish(x, &rty))
            }
            MathIntrinsic::Selu => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_selu(x, &rty))
            }
            MathIntrinsic::Tanhshrink => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_tanhshrink(x, &rty))
            }
            MathIntrinsic::HardSigmoid => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_hardsigmoid(x, &rty))
            }
            MathIntrinsic::HardSwish => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_hardswish(x, &rty))
            }
            MathIntrinsic::Softsign => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_softsign(x, &rty))
            }
            MathIntrinsic::LogSigmoid => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_logsigmoid(x, &rty))
            }
            MathIntrinsic::Tan => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_tan(x, &rty))
            }
            MathIntrinsic::Asin => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_asin(x, &rty))
            }
            MathIntrinsic::Acos => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_acos(x, &rty))
            }
            MathIntrinsic::Cbrt => {
                let x = self.lower_math_arg(args.first()?, &rty);
                Some(self.emit_cbrt(x, &rty))
            }
            MathIntrinsic::Fmax | MathIntrinsic::Fmin => {
                if args.len() != 2 {
                    return None;
                }
                // Coerce each operand to the (float) result type, exactly like sqrt/exp/pow above.
                // Using a bare `lower_expr` here left an integer operand at its int type, then the
                // float `Cmp(Fogt/Folt)` below ran on `i32` — MIR the verifier and Cranelift reject
                // (the native backend errored out) while the interpreter computed an integer max and
                // returned silently: a backend divergence on `fmax(int, int)`. `lower_math_arg`
                // inserts the int→float coercion so both backends run the identical float compare.
                let a = self.lower_math_arg(&args[0], &rty);
                let b = self.lower_math_arg(&args[1], &rty);
                let pred = if matches!(op, MathIntrinsic::Fmax) {
                    CmpOp::Fogt
                } else {
                    CmpOp::Folt
                };
                let c = self.builder.build(mask_ty(&rty), Op::Cmp(pred, a, b));
                Some(self.builder.build(rty.clone(), Op::Select(c, a, b)))
            }
        }
    }

    /// `abs(x) = max(x, −x)` as a compare + select — the same form `emit_softplus` uses for `|x|`, so
    /// the two agree. Lane-type-agnostic (f32 or f64) and built only from primitives (`FSub`/`Cmp`/
    /// `Select`), so it vectorizes and is bit-identical across backends. Differs from the reduction
    /// kernel's bit-clear abs only at ±0 (a `−0` here yields `+0`, harmless and never compared against
    /// the kernel — a `@parallel` absmax dispatches to the kernel, a scalar/sequential one uses this).
    fn emit_abs(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let zero = self.splat_const_f(0.0, rty);
        let negx = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, zero, x));
        let gtm = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Fogt, x, negx));
        self.builder.build(rty.clone(), Op::Select(gtm, x, negx))
    }

    /// Integer absolute value: `select(x < 0, 0 - x, x)`. Wraps for `INT_MIN` (like C/Rust's
    /// `wrapping_abs`) and is plain integer sub/cmp/select, so both backends agree bit-for-bit.
    fn emit_int_abs(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let zero = self
            .builder
            .build(rty.clone(), Op::ConstInt(0, rty.clone()));
        let neg = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::Sub, zero, x));
        let isneg = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, x, zero));
        self.builder.build(rty.clone(), Op::Select(isneg, neg, x))
    }

    /// Lower a math-intrinsic argument, coercing an integer operand to the float result type `rty`
    /// (so `sqrt(16)` promotes the `16` to `16.0` rather than feeding a float op an int operand,
    /// which the verifier rejects). A same-typed float operand is returned unchanged.
    fn lower_math_arg(&mut self, arg: &Expr, rty: &MirType) -> ValueId {
        let v = self.lower_expr(arg);
        let from = self.expr_mir(arg);
        let signed = self.signed(arg);
        self.coerce_to(v, &from, rty, signed)
    }

    /// `exp(x)` as a fast, deterministic polynomial (≈1 ULP of the true `exp`). Always computed in
    /// `f32`; for an `f64` result the argument is demoted and the result promoted (exact in both
    /// backends). Works on a scalar or a SIMD-vector `f32`.
    fn emit_exp(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let want_f64 = matches!(rty.lane_type(), MirType::F64);
        let f32ty = float_ty_like(rty, MirType::F32);
        let xf = if want_f64 {
            self.builder
                .build(f32ty.clone(), Op::Cast(CastKind::FpTrunc, x, f32ty.clone()))
        } else {
            x
        };
        let r = self.emit_exp_f32(xf, &f32ty);
        if want_f64 {
            self.builder
                .build(rty.clone(), Op::Cast(CastKind::FpExt, r, rty.clone()))
        } else {
            r
        }
    }

    /// `sigmoid(x) = 1 / (1 + exp(-x))`, built on the exp polynomial. Works on a scalar or a SIMD
    /// vector; bit-identical across backends because every step is (see `emit_exp`).
    fn emit_sigmoid(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let neg1 = self.splat_const_f(-1.0, rty);
        let negx = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, x, neg1));
        let e = self.emit_exp(negx, rty);
        let one = self.splat_const_f(1.0, rty);
        let denom = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, one, e));
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FDiv, one, denom))
    }

    /// `tanh(x) = 1 - 2 / (exp(2x) + 1)`, built on the exp polynomial (same identity the GELU tanh
    /// approximation uses). Works on a scalar or a SIMD vector; bit-identical across backends.
    fn emit_tanh(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let two = self.splat_const_f(2.0, rty);
        let twox = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, x, two));
        let e = self.emit_exp(twox, rty);
        let one = self.splat_const_f(1.0, rty);
        let denom = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, e, one));
        let frac = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FDiv, two, denom));
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, one, frac))
    }

    /// `silu(x) = x · sigmoid(x)` (swish). The scalar/composed path; an `out[i] = silu(x[i])` loop
    /// dispatches to the fused AVX2 kernel instead. Bit-identical across backends (see `emit_exp`).
    fn emit_silu(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let s = self.emit_sigmoid(x, rty);
        self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x, s))
    }

    /// `gelu(x)` (tanh approximation): `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`. Mirrors the
    /// fused AVX2 `gelu8` op-for-op (so the scalar form and the dispatched loop form agree). The
    /// BERT/GPT-2/ViT activation; bit-identical across backends.
    fn emit_gelu(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let x2 = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x, x));
        let x3 = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x2, x));
        let c1 = self.splat_const_f(0.044715, rty);
        let t = self.builder.build(rty.clone(), Op::Fma(c1, x3, x)); // 0.044715·x³ + x
        let c0 = self.splat_const_f(0.7978845608, rty);
        let inner = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, c0, t));
        let th = self.emit_tanh(inner, rty);
        let one = self.splat_const_f(1.0, rty);
        let onep = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, one, th));
        let half = self.splat_const_f(0.5, rty);
        let hx = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, half, x));
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, hx, onep))
    }

    /// `silu_backward(x, dy) = dy · silu'(x)`, `silu'(x) = fma(x·s, 1−s, s)` with `s = sigmoid(x)`.
    /// Mirrors the runtime `silu_bwd8`/`silu_bwd_2` op-for-op (single-rounded FMA), so the inlined
    /// fallback (a `while` loop / standalone call) equals the 256-bit `mercury_vmath2_f32` dispatch.
    /// The training gradient through a SiLU/swish gate; bit-identical across backends.
    fn emit_silu_backward(&mut self, x: ValueId, dy: ValueId, rty: &MirType) -> ValueId {
        let s = self.emit_sigmoid(x, rty);
        let one = self.splat_const_f(1.0, rty);
        let oms = self.builder.build(rty.clone(), Op::Bin(BinOp::FSub, one, s));
        let xs = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x, s));
        let g = self.builder.build(rty.clone(), Op::Fma(xs, oms, s)); // x·s·(1−s) + s
        self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, dy, g))
    }

    /// `gelu_backward(x, dy) = dy · gelu'(x)` (tanh approximation, paired with [`emit_gelu`]). With
    /// `I = c0·(x + c1·x³)`, `u = tanh(I)`: `gelu'(x) = ½(1+u) + ½·x·(1−u²)·c0·(1 + 3·c1·x²)`. Mirrors
    /// the runtime `gelu_bwd8`/`gelu_bwd_2` op-for-op (the inner `I`/`u` identical to `emit_gelu`, the
    /// `3·c1·x²` written as `fma(c1, 3x², 1)` so no new constant), so the inlined fallback equals the
    /// 256-bit dispatch. The BERT/GPT-2/ViT training gradient; bit-identical across backends.
    fn emit_gelu_backward(&mut self, x: ValueId, dy: ValueId, rty: &MirType) -> ValueId {
        let c0 = self.splat_const_f(0.7978845608, rty);
        let c1 = self.splat_const_f(0.044715, rty);
        let one = self.splat_const_f(1.0, rty);
        let half = self.splat_const_f(0.5, rty);
        let x2 = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x, x));
        let x3 = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x2, x));
        let t = self.builder.build(rty.clone(), Op::Fma(c1, x3, x)); // c1·x³ + x
        let inner = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, c0, t));
        let u = self.emit_tanh(inner, rty);
        let onep = self.builder.build(rty.clone(), Op::Bin(BinOp::FAdd, one, u));
        let half_onep = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, half, onep)); // ½(1+u)
        let u2 = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, u, u));
        let sech2 = self.builder.build(rty.clone(), Op::Bin(BinOp::FSub, one, u2)); // 1 − u²
        let three = self.splat_const_f(3.0, rty);
        let three_x2 = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, three, x2));
        let di = self.builder.build(rty.clone(), Op::Fma(c1, three_x2, one)); // c1·3x² + 1
        let dinner = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, c0, di)); // I'(x)
        let hx = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, half, x));
        let a = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, hx, sech2));
        let term2 = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, a, dinner)); // ½x(1−u²)I'
        let gp = self.builder.build(rty.clone(), Op::Bin(BinOp::FAdd, half_onep, term2));
        self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, dy, gp))
    }

    /// `sigmoid_backward(x, dy) = dy · σ(x)·(1 − σ(x))`. Mirrors `sigmoid_bwd8`/`sigmoid_bwd_2`, so the
    /// inlined fallback equals the 256-bit dispatch. The logistic-gate training gradient.
    fn emit_sigmoid_backward(&mut self, x: ValueId, dy: ValueId, rty: &MirType) -> ValueId {
        let s = self.emit_sigmoid(x, rty);
        let one = self.splat_const_f(1.0, rty);
        let oms = self.builder.build(rty.clone(), Op::Bin(BinOp::FSub, one, s));
        let sp = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, s, oms)); // s·(1−s)
        self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, dy, sp))
    }

    /// `tanh_backward(x, dy) = dy · (1 − tanh²(x))`. Mirrors `tanh_bwd8`/`tanh_bwd_2`, so the inlined
    /// fallback equals the 256-bit dispatch. The tanh-gate (RNN/LSTM cell) training gradient.
    fn emit_tanh_backward(&mut self, x: ValueId, dy: ValueId, rty: &MirType) -> ValueId {
        let t = self.emit_tanh(x, rty);
        let one = self.splat_const_f(1.0, rty);
        let t2 = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, t, t));
        let sech2 = self.builder.build(rty.clone(), Op::Bin(BinOp::FSub, one, t2)); // 1 − t²
        self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, dy, sech2))
    }

    /// `elu_backward(x, dy) = dy · (x>0 ? 1 : eˣ)` (α=1). Mirrors `elu_bwd8`/`elu_bwd_2` (the same
    /// `Cmp(Fogt)`+`Select` the forward `emit_elu` uses, so dispatched == composed). The ELU training grad.
    fn emit_elu_backward(&mut self, x: ValueId, dy: ValueId, rty: &MirType) -> ValueId {
        let mty = mask_ty(rty);
        let e = self.emit_exp(x, rty);
        let zero = self.splat_const_f(0.0, rty);
        let one = self.splat_const_f(1.0, rty);
        let pos = self.builder.build(mty, Op::Cmp(CmpOp::Fogt, x, zero)); // x > 0
        let g = self.builder.build(rty.clone(), Op::Select(pos, one, e)); // x>0 ? 1 : eˣ
        self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, dy, g))
    }

    /// `softplus_backward(x, dy) = dy · σ(x)` (since `softplus'(x) = σ(x)`). Mirrors
    /// `softplus_bwd8`/`softplus_bwd_2`. The softplus (VAE/flow/Mish) training gradient.
    fn emit_softplus_backward(&mut self, x: ValueId, dy: ValueId, rty: &MirType) -> ValueId {
        let s = self.emit_sigmoid(x, rty);
        self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, dy, s))
    }

    /// `elu(x) = x>0 ? x : e^x − 1` (α=1), the exponential linear unit. Mirrors the fused AVX2 `elu8`
    /// (`blendv(exp−1, x, x>0)`) — `x` passes through for the positive lane, `exp(x)−1` for the
    /// negative — so the scalar form and the dispatched loop agree; bit-identical across backends.
    fn emit_elu(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let e = self.emit_exp(x, rty);
        let one = self.splat_const_f(1.0, rty);
        let em1 = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, e, one));
        let zero = self.splat_const_f(0.0, rty);
        let pos = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Fogt, x, zero));
        self.builder.build(rty.clone(), Op::Select(pos, x, em1))
    }

    /// `leaky_relu(x) = x>0 ? x : 0.01·x`, the leaky rectifier. Mirrors the fused AVX2 `leakyrelu8`
    /// (`blendv(0.01·x, x, x>0)`). No transcendental — just a scaled negative slope.
    fn emit_leaky_relu(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let alpha = self.splat_const_f(0.01, rty);
        let scaled = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, x, alpha));
        let zero = self.splat_const_f(0.0, rty);
        let pos = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Fogt, x, zero));
        self.builder.build(rty.clone(), Op::Select(pos, x, scaled))
    }

    /// `softplus(x) = ln(1 + e^x)`, the smooth ReLU, in the stable form `max(x,0) + ln(1 + e^{−|x|})`
    /// (the `exp` never overflows). `|x|` is `max(x, −x)` here (vs the kernel's bit-clear abs — the two
    /// differ only at ±0, which is washed out by the following `exp`, and a loop dispatches to the
    /// kernel regardless). Reuses `emit_exp`/`emit_log`, so it is bit-identical across backends.
    fn emit_softplus(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let zero = self.splat_const_f(0.0, rty);
        let negx = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, zero, x));
        let gtm = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Fogt, x, negx));
        let absx = self.builder.build(rty.clone(), Op::Select(gtm, x, negx));
        let nabs = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, zero, absx));
        let e = self.emit_exp(nabs, rty);
        let one = self.splat_const_f(1.0, rty);
        let onepe = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, one, e));
        let l = self.emit_log(onepe, rty);
        let posm = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Fogt, x, zero));
        let relux = self.builder.build(rty.clone(), Op::Select(posm, x, zero));
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, relux, l))
    }

    /// `mish(x) = x · tanh(softplus(x))`, the self-gated smooth activation. Mirrors `mish8` op-for-op.
    fn emit_mish(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let sp = self.emit_softplus(x, rty);
        let th = self.emit_tanh(sp, rty);
        self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x, th))
    }

    /// `softsign(x) = x / (1 + |x|)`, the bounded polynomial activation. `|x|` is `max(x, −x)` here
    /// (vs the kernel's bit-clear abs — differing only at ±0, which the `x/(1+|x|)` then washes out:
    /// the result is ±0 either way). No transcendental, so it vectorizes and is bit-identical across
    /// backends; a loop dispatches to `softsign8` regardless.
    fn emit_softsign(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let ax = self.emit_abs(x, rty);
        let one = self.splat_const_f(1.0, rty);
        let den = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, one, ax));
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FDiv, x, den))
    }

    /// `logsigmoid(x) = ln(σ(x)) = −softplus(−x)`, the stable log-sigmoid. Reuses `emit_softplus`
    /// (itself the stable `max(t,0)+ln(1+e^{−|t|})`), so it is bit-identical to `logsigmoid8` and to
    /// the scalar twin; the two negations are exact.
    fn emit_logsigmoid(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let zero = self.splat_const_f(0.0, rty);
        let nx = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, zero, x));
        let sp = self.emit_softplus(nx, rty);
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, zero, sp))
    }

    /// `selu(x) = λ·(x>0 ? x : α·(eˣ−1))`, the scaled ELU of self-normalizing networks. Mirrors `selu8`
    /// op-for-op, including the multiply order `λ·(α·em1)` (float mul isn't associative).
    fn emit_selu(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let lambda = self.splat_const_f(1.050_700_98, rty);
        let posval = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, lambda, x));
        let e = self.emit_exp(x, rty);
        let one = self.splat_const_f(1.0, rty);
        let em1 = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, e, one));
        let alpha = self.splat_const_f(1.673_263_2, rty);
        let aem1 = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, alpha, em1));
        let negval = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, lambda, aem1));
        let zero = self.splat_const_f(0.0, rty);
        let pos = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Fogt, x, zero));
        self.builder
            .build(rty.clone(), Op::Select(pos, posval, negval))
    }

    /// `tanhshrink(x) = x − tanh(x)`. Mirrors `tanhshrink8`.
    fn emit_tanhshrink(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let th = self.emit_tanh(x, rty);
        self.builder.build(rty.clone(), Op::Bin(BinOp::FSub, x, th))
    }

    /// `hardsigmoid(x) = clamp(x+3, 0, 6)·(1/6)`. The clamp is `Cmp(Fogt)`/`Cmp(Folt)` + `Select`,
    /// which matches the AVX2 `hardsigmoid8` (`max_ps`/`min_ps`) bit-for-bit (ordered compares are
    /// false for NaN, so both return the bound).
    fn emit_hardsigmoid(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let three = self.splat_const_f(3.0, rty);
        let y = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, x, three));
        let zero = self.splat_const_f(0.0, rty);
        let gt0 = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Fogt, y, zero));
        let lo = self.builder.build(rty.clone(), Op::Select(gt0, y, zero));
        let six = self.splat_const_f(6.0, rty);
        let lt6 = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Folt, lo, six));
        let hi = self.builder.build(rty.clone(), Op::Select(lt6, lo, six));
        let inv6 = self.splat_const_f(1.0 / 6.0, rty);
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, hi, inv6))
    }

    /// `hardswish(x) = x · hardsigmoid(x)`. Mirrors `hardswish8`.
    fn emit_hardswish(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let hs = self.emit_hardsigmoid(x, rty);
        self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x, hs))
    }

    /// `erf(x)` (the Gauss error function — `exact` GELU is `0.5·x·(1 + erf(x/√2))`). Always computed
    /// in `f32`; an `f64` result is demoted/promoted like `exp`/`log`. Works on a scalar or a SIMD
    /// vector; bit-identical across backends because every step is a primitive op (see `emit_erf_f32`).
    fn emit_erf(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let want_f64 = matches!(rty.lane_type(), MirType::F64);
        let f32ty = float_ty_like(rty, MirType::F32);
        let xf = if want_f64 {
            self.builder
                .build(f32ty.clone(), Op::Cast(CastKind::FpTrunc, x, f32ty.clone()))
        } else {
            x
        };
        let r = self.emit_erf_f32(xf, &f32ty);
        if want_f64 {
            self.builder
                .build(rty.clone(), Op::Cast(CastKind::FpExt, r, rty.clone()))
        } else {
            r
        }
    }

    /// The erf approximation in `f32` (Abramowitz–Stegun 7.1.26, ~1.5e-7 error). `|x|` and the
    /// odd-function sign are handled with compare/select (no IEEE bit surgery), and `e^(-x²)` reuses
    /// the exp polynomial — so it vectorizes (`fty` may be a `Vec` of `f32`) and both backends agree.
    fn emit_erf_f32(&mut self, x: ValueId, fty: &MirType) -> ValueId {
        let mty = mask_ty(fty);
        let one = self.splat_const_f(1.0, fty);
        let neg1 = self.splat_const_f(-1.0, fty);
        // ax = |x| = max(x, -x), via compare + select.
        let negx = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, x, neg1));
        let gt = self
            .builder
            .build(mty.clone(), Op::Cmp(CmpOp::Fogt, x, negx));
        let ax = self.builder.build(fty.clone(), Op::Select(gt, x, negx));
        // t = 1 / (1 + P*ax)
        let p = self.splat_const_f(ERF_P, fty);
        let denom = self.builder.build(fty.clone(), Op::Fma(p, ax, one));
        let t = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FDiv, one, denom));
        // poly(t) = t·(a₁ + t·(a₂ + t·(a₃ + t·(a₄ + t·a₅)))) by Horner with fmas.
        let mut h = self.splat_const_f(ERF_A[4], fty);
        for &c in ERF_A[..4].iter().rev() {
            let cc = self.splat_const_f(c, fty);
            h = self.builder.build(fty.clone(), Op::Fma(h, t, cc));
        }
        let poly = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, h, t));
        // e = exp(-ax²)
        let ax2 = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, ax, ax));
        let neg_ax2 = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, ax2, neg1));
        let e = self.emit_exp_f32(neg_ax2, fty);
        // erf(|x|) = 1 - poly·e
        let pe = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, poly, e));
        let mag = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, one, pe));
        // erf is odd: result = (x >= 0) ? mag : -mag.
        let neg_mag = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, mag, neg1));
        let zero = self.splat_const_f(0.0, fty);
        let ge = self.builder.build(mty, Op::Cmp(CmpOp::Foge, x, zero));
        self.builder
            .build(fty.clone(), Op::Select(ge, mag, neg_mag))
    }

    /// `sinh(x)` (`is_cosh == false`) or `cosh(x)` (`true`) = `(eˣ ∓ e⁻ˣ)·0.5`, composed from the
    /// shared `exp` so it vectorizes and stays bit-identical across backends; the dispatched
    /// `sinh8`/`cosh8` kernel mirrors this op-for-op (`-x` via `·-1` to match). Overflows like libm.
    fn emit_sinh_cosh(&mut self, x: ValueId, rty: &MirType, is_cosh: bool) -> ValueId {
        let neg1 = self.splat_const_f(-1.0, rty);
        let nx = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, x, neg1));
        let ex = self.emit_exp(x, rty);
        let enx = self.emit_exp(nx, rty);
        let combined = if is_cosh {
            self.builder
                .build(rty.clone(), Op::Bin(BinOp::FAdd, ex, enx))
        } else {
            self.builder
                .build(rty.clone(), Op::Bin(BinOp::FSub, ex, enx))
        };
        let half = self.splat_const_f(0.5, rty);
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, combined, half))
    }

    /// `asinh(x) = sign(x)·log(|x| + √(x²+1))` — the sign-stable inverse hyperbolic sine (evaluating
    /// `log` on `|x|+√(…) ≥ 1` dodges the `x+√(x²+1)` cancellation for large negative `x`). Reuses
    /// `emit_abs`/`emit_log`; works on a scalar or a SIMD vector, bit-identical across backends.
    fn emit_asinh(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let ax = self.emit_abs(x, rty);
        let x2 = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, ax, ax));
        let one = self.splat_const_f(1.0, rty);
        let inner = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, x2, one));
        let s = self.builder.build(rty.clone(), Op::Sqrt(inner));
        let sum = self.builder.build(rty.clone(), Op::Bin(BinOp::FAdd, ax, s));
        let t = self.emit_log(sum, rty);
        // copysign(t, x) via select (t ≥ 0): x < 0 ? −t : t.
        let neg1 = self.splat_const_f(-1.0, rty);
        let negt = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, t, neg1));
        let zero = self.splat_const_f(0.0, rty);
        let isneg = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Folt, x, zero));
        self.builder.build(rty.clone(), Op::Select(isneg, negt, t))
    }

    /// `acosh(x) = log(x + √(x²−1))` for `x ≥ 1` (`√` of a negative → `NaN` below, matching `libm`).
    /// Reuses `emit_log`; works on a scalar or a SIMD vector, bit-identical across backends.
    fn emit_acosh(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let x2 = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x, x));
        let one = self.splat_const_f(1.0, rty);
        let inner = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, x2, one));
        let s = self.builder.build(rty.clone(), Op::Sqrt(inner));
        let sum = self.builder.build(rty.clone(), Op::Bin(BinOp::FAdd, x, s));
        self.emit_log(sum, rty)
    }

    /// `atanh(x) = ½·log((1+x)/(1−x))` for `|x| < 1` (the Fisher z-transform). Reuses `emit_log`;
    /// works on a scalar or a SIMD vector, bit-identical across backends.
    fn emit_atanh(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let one = self.splat_const_f(1.0, rty);
        let num = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, one, x));
        let den = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, one, x));
        let r = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FDiv, num, den));
        let l = self.emit_log(r, rty);
        let half = self.splat_const_f(0.5, rty);
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, l, half))
    }

    /// `expm1(x) = eˣ − 1` via Kahan's stable correction `(u−1)·x/log(u)`, `u = eˣ`, guarding `u==1 → x`
    /// (the `0·∞` otherwise). Reuses `emit_exp`/`emit_log`; mirrors `expm1_1`/`expm1_8` op-for-op, so for
    /// finite `x` the inlined form equals the dispatched kernel. Scalar or SIMD; bit-identical backends.
    fn emit_expm1(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let u = self.emit_exp(x, rty);
        let one = self.splat_const_f(1.0, rty);
        let um1 = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, u, one));
        let lu = self.emit_log(u, rty);
        let ratio = self.builder.build(rty.clone(), Op::Bin(BinOp::FDiv, x, lu));
        let val = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, um1, ratio));
        let is1 = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Foeq, u, one));
        self.builder.build(rty.clone(), Op::Select(is1, x, val))
    }

    /// `log1p(x) = ln(1+x)` via Kahan's stable correction `log(u)·x/(u−1)`, `u = 1+x`, guarding `u==1 → x`.
    /// Reuses `emit_log`; mirrors `log1p_1`/`log1p_8` op-for-op. Scalar or SIMD; bit-identical backends.
    fn emit_log1p(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let one = self.splat_const_f(1.0, rty);
        let u = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, one, x));
        let d = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, u, one));
        let lu = self.emit_log(u, rty);
        let ratio = self.builder.build(rty.clone(), Op::Bin(BinOp::FDiv, x, d));
        let val = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, lu, ratio));
        let is1 = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Foeq, u, one));
        self.builder.build(rty.clone(), Op::Select(is1, x, val))
    }

    /// `sin(x)` (`is_cos == false`) or `cos(x)` (`true`) as a fast, deterministic polynomial. Always
    /// computed in `f32`; an `f64` result is demoted/promoted like `exp`. Works on a scalar or a SIMD
    /// vector; bit-identical across backends. Enables rotary position embeddings (RoPE).
    fn emit_trig(&mut self, x: ValueId, rty: &MirType, is_cos: bool) -> ValueId {
        let want_f64 = matches!(rty.lane_type(), MirType::F64);
        let f32ty = float_ty_like(rty, MirType::F32);
        let xf = if want_f64 {
            self.builder
                .build(f32ty.clone(), Op::Cast(CastKind::FpTrunc, x, f32ty.clone()))
        } else {
            x
        };
        let r = self.emit_trig_f32(xf, &f32ty, is_cos);
        if want_f64 {
            self.builder
                .build(rty.clone(), Op::Cast(CastKind::FpExt, r, rty.clone()))
        } else {
            r
        }
    }

    /// The sin/cos polynomial in `f32` (`fty` is `f32` or a `Vec` of `f32`). Reduces `x` to
    /// `r ∈ [-π/4, π/4]` by `q = round(x·2/π)` quadrants (round-to-nearest via the add-magic trick),
    /// evaluates the Cephes `sinf`/`cosf` minimax polynomials on `r`, and selects ±sin/±cos by
    /// `q mod 4`. The quadrant is reduced to an exact small float so every blend uses a float-compare
    /// mask (the proven `select` path), and every step is a primitive op, so both backends agree.
    fn emit_trig_f32(&mut self, x: ValueId, fty: &MirType, is_cos: bool) -> ValueId {
        let ity = match fty {
            MirType::Vec(_, n) => MirType::Vec(Box::new(MirType::I32), *n),
            _ => MirType::I32,
        };
        let mty = mask_ty(fty);
        // q = round(x · 2/π) via add-magic / sub-magic (round-to-nearest-even, pure f32).
        let two_pi = self.splat_const_f(TWO_OVER_PI, fty);
        let magic = self.splat_const_f(EXP_MAGIC, fty);
        let tt = self.builder.build(fty.clone(), Op::Fma(x, two_pi, magic));
        let qf = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, tt, magic));
        // r = ((x - qf·PIO2_1) - qf·PIO2_2) - qf·PIO2_3   (3-part π/2 split keeps the reduction exact).
        let n1 = self.splat_const_f(-PIO2_1, fty);
        let n2 = self.splat_const_f(-PIO2_2, fty);
        let n3 = self.splat_const_f(-PIO2_3, fty);
        let r = self.builder.build(fty.clone(), Op::Fma(qf, n1, x));
        let r = self.builder.build(fty.clone(), Op::Fma(qf, n2, r));
        let r = self.builder.build(fty.clone(), Op::Fma(qf, n3, r));
        let z = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, r, r));
        // sin_p(r) = r·(1 + z·poly) with poly = ((s₀z + s₁)z + s₂).
        let mut s = self.splat_const_f(SIN_P[0], fty);
        for &c in &SIN_P[1..] {
            let cc = self.splat_const_f(c, fty);
            s = self.builder.build(fty.clone(), Op::Fma(s, z, cc));
        }
        let sz = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, s, z));
        let sin_p = self.builder.build(fty.clone(), Op::Fma(sz, r, r));
        // cos_p(r) = (1 - 0.5z) + z²·poly with poly = ((c₀z + c₁)z + c₂).
        let mut cc0 = self.splat_const_f(COS_P[0], fty);
        for &c in &COS_P[1..] {
            let k = self.splat_const_f(c, fty);
            cc0 = self.builder.build(fty.clone(), Op::Fma(cc0, z, k));
        }
        let z2 = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, z, z));
        let cz2 = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, cc0, z2));
        let neg_half = self.splat_const_f(-0.5, fty);
        let one = self.splat_const_f(1.0, fty);
        let hz = self.builder.build(fty.clone(), Op::Fma(neg_half, z, one));
        let cos_p = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FAdd, hz, cz2));
        // ±variants for the quadrant blend.
        let neg1 = self.splat_const_f(-1.0, fty);
        let neg_sin = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, sin_p, neg1));
        let neg_cos = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, cos_p, neg1));
        // quad = (int)qf & 3, then back to an exact float (0/1/2/3) for float-mask selects. For
        // negative q the two's-complement `& 3` is still the correct mod-4 quadrant.
        let qi = self
            .builder
            .build(ity.clone(), Op::Cast(CastKind::FpToSi, qf, ity.clone()));
        let three = self.splat_const_i(3, &ity);
        let quad_i = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::And, qi, three));
        let quad = self
            .builder
            .build(fty.clone(), Op::Cast(CastKind::SiToFp, quad_i, fty.clone()));
        // sin: [sin, cos, -sin, -cos];  cos: [cos, -sin, -cos, sin], indexed by quad.
        let (a0, a1, a2, a3) = if is_cos {
            (cos_p, neg_sin, neg_cos, sin_p)
        } else {
            (sin_p, cos_p, neg_sin, neg_cos)
        };
        let k0 = self.splat_const_f(0.0, fty);
        let k1 = self.splat_const_f(1.0, fty);
        let k2 = self.splat_const_f(2.0, fty);
        let eq0 = self
            .builder
            .build(mty.clone(), Op::Cmp(CmpOp::Foeq, quad, k0));
        let eq1 = self
            .builder
            .build(mty.clone(), Op::Cmp(CmpOp::Foeq, quad, k1));
        let eq2 = self.builder.build(mty, Op::Cmp(CmpOp::Foeq, quad, k2));
        let sel23 = self.builder.build(fty.clone(), Op::Select(eq2, a2, a3));
        let sel123 = self.builder.build(fty.clone(), Op::Select(eq1, a1, sel23));
        self.builder.build(fty.clone(), Op::Select(eq0, a0, sel123))
    }

    /// `atan(x)`, computed in `f32` (demote/promote an `f64` result like `exp`). Works on a scalar or
    /// a SIMD vector; bit-identical across backends. Enables angle/geometry ops and `atan2`-style schemes.
    fn emit_atan(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let want_f64 = matches!(rty.lane_type(), MirType::F64);
        let f32ty = float_ty_like(rty, MirType::F32);
        let xf = if want_f64 {
            self.builder
                .build(f32ty.clone(), Op::Cast(CastKind::FpTrunc, x, f32ty.clone()))
        } else {
            x
        };
        let r = self.emit_atan_f32(xf, &f32ty);
        if want_f64 {
            self.builder
                .build(rty.clone(), Op::Cast(CastKind::FpExt, r, rty.clone()))
        } else {
            r
        }
    }

    /// The Cephes `atan` poly in `f32` (`fty` is `f32` or a `Vec` of `f32`) — mirrors `atan1`/`atan8`
    /// op-for-op: `|x|`, the two breakpoint masks, both reduced candidates blended in (`big` overrides
    /// `mid`), a degree-3 odd FMA poly, the `π/4`/`π/2` offset, and a sign restore by select. For finite
    /// `x` this equals the dispatched 256-bit kernel bit-for-bit (abs/copysign agree with the kernel's
    /// bit-mask forms on finite inputs), so dispatched and composed `atan` agree.
    fn emit_atan_f32(&mut self, x: ValueId, fty: &MirType) -> ValueId {
        let mty = mask_ty(fty);
        let ax = self.emit_abs(x, fty);
        let tan3 = self.splat_const_f(ATAN_TAN_3PI8, fty);
        let tan1 = self.splat_const_f(ATAN_TAN_PI8, fty);
        let big = self
            .builder
            .build(mty.clone(), Op::Cmp(CmpOp::Fogt, ax, tan3));
        let mid = self
            .builder
            .build(mty.clone(), Op::Cmp(CmpOp::Fogt, ax, tan1));
        let one = self.splat_const_f(1.0, fty);
        let neg1 = self.splat_const_f(-1.0, fty);
        // mid candidate (ax−1)/(ax+1); big candidate −1/ax.
        let axm1 = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, ax, one));
        let axp1 = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FAdd, ax, one));
        let xr_mid = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FDiv, axm1, axp1));
        let xr_big = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FDiv, neg1, ax));
        let xr = self.builder.build(fty.clone(), Op::Select(mid, xr_mid, ax));
        let xr = self.builder.build(fty.clone(), Op::Select(big, xr_big, xr));
        // offset 0 → π/4 (mid) → π/2 (big).
        let zero = self.splat_const_f(0.0, fty);
        let pio4 = self.splat_const_f(ATAN_PIO4, fty);
        let pio2 = self.splat_const_f(ATAN_PIO2, fty);
        let y = self.builder.build(fty.clone(), Op::Select(mid, pio4, zero));
        let y = self.builder.build(fty.clone(), Op::Select(big, pio2, y));
        // degree-3 odd minimax via FMA Horner: ((((P0·z+P1)·z+P2)·z+P3)·z·xr) + xr.
        let z = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, xr, xr));
        let mut p = self.splat_const_f(ATAN_P[0], fty);
        for &c in &ATAN_P[1..] {
            let cc = self.splat_const_f(c, fty);
            p = self.builder.build(fty.clone(), Op::Fma(p, z, cc));
        }
        let pz = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, p, z));
        let pzx = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, pz, xr));
        let res = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FAdd, pzx, xr));
        let yf = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FAdd, y, res));
        // copysign(yf, x) via select (yf ≥ 0): x < 0 ? −yf : yf.
        let negyf = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, yf, neg1));
        let isneg = self.builder.build(mty, Op::Cmp(CmpOp::Folt, x, zero));
        self.builder
            .build(fty.clone(), Op::Select(isneg, negyf, yf))
    }

    /// `tan(x) = sin(x)/cos(x)` — reuses `emit_trig` (which mirrors `sincos1`/`sin8`), so the composed
    /// and dispatched forms agree. Works on a scalar or a SIMD vector.
    fn emit_tan(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let s = self.emit_trig(x, rty, false);
        let c = self.emit_trig(x, rty, true);
        self.builder.build(rty.clone(), Op::Bin(BinOp::FDiv, s, c))
    }

    /// `asin(x) = atan(x/√(1−x²))` over `[−1, 1]` — mirrors `asin1`/`asin8` op-for-op (`x²`, `1−x²`,
    /// `Op::Sqrt`, divide, then `emit_atan`), so dispatched and composed agree bit-for-bit on `f32`.
    fn emit_asin(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let x2 = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x, x));
        let one = self.splat_const_f(1.0, rty);
        let om = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, one, x2));
        let sq = self.builder.build(rty.clone(), Op::Sqrt(om));
        let d = self.builder.build(rty.clone(), Op::Bin(BinOp::FDiv, x, sq));
        self.emit_atan(d, rty)
    }

    /// `acos(x) = π/2 − asin(x)` over `[−1, 1]` — mirrors `acos1`/`acos8`. The π/2 constant is the same
    /// `FRAC_PI_2 as f32` the kernel uses, so it matches bit-for-bit.
    fn emit_acos(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let asin = self.emit_asin(x, rty);
        let pio2 = self.splat_const_f(ATAN_PIO2, rty);
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, pio2, asin))
    }

    /// `cbrt(x) = copysign(e^{ln|x|/3}, x)` with the `|x|==0 → 0` guard — mirrors `cbrt1`/`cbrt8`,
    /// reusing `emit_exp`/`emit_log`. The ±0 sign follows the `emit_asinh` precedent (select-based
    /// copysign). Works scalar or vector; bit-identical to the dispatched kernel.
    fn emit_cbrt(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let ax = self.emit_abs(x, rty);
        let lx = self.emit_log(ax, rty);
        let third = self.splat_const_f(1.0 / 3.0, rty);
        let scaled = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, lx, third));
        let mag = self.emit_exp(scaled, rty);
        let zero = self.splat_const_f(0.0, rty);
        let iszero = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Foeq, ax, zero));
        let mag = self
            .builder
            .build(rty.clone(), Op::Select(iszero, zero, mag));
        // copysign(mag, x) via select (mag ≥ 0): x < 0 ? −mag : mag.
        let neg1 = self.splat_const_f(-1.0, rty);
        let negmag = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, mag, neg1));
        let isneg = self
            .builder
            .build(mask_ty(rty), Op::Cmp(CmpOp::Folt, x, zero));
        self.builder
            .build(rty.clone(), Op::Select(isneg, negmag, mag))
    }

    /// `atan2(y, x)` — the full-circle angle of `(x, y)` (geometry, robotics, complex argument, RoPE-style
    /// angle recovery). `atan(y/x)` then a quadrant fix: when `x < 0`, add `+π` (for `y ≥ 0`) or `−π`
    /// (for `y < 0`). `x = 0` falls out (`y/x = ±∞`, `atan(±∞) = ±π/2`, no fix since `x` is not `< 0`).
    /// Built only from `emit_atan` + primitives, so it vectorizes and is bit-identical across backends;
    /// matches libm except at the `(0,0)` origin (→ `NaN` here vs libm's `0`). Works scalar or vector.
    fn emit_atan2(&mut self, y: ValueId, x: ValueId, rty: &MirType) -> ValueId {
        let mty = mask_ty(rty);
        let q = self.builder.build(rty.clone(), Op::Bin(BinOp::FDiv, y, x));
        let a = self.emit_atan(q, rty);
        let zero = self.splat_const_f(0.0, rty);
        let pi = self.splat_const_f(std::f64::consts::PI, rty);
        let neg_pi = self.splat_const_f(-std::f64::consts::PI, rty);
        // copysign(π, y) via select: y < 0 → −π, else +π.
        let yneg = self
            .builder
            .build(mty.clone(), Op::Cmp(CmpOp::Folt, y, zero));
        let pi_signed = self
            .builder
            .build(rty.clone(), Op::Select(yneg, neg_pi, pi));
        let xneg = self.builder.build(mty, Op::Cmp(CmpOp::Folt, x, zero));
        let adj = self
            .builder
            .build(rty.clone(), Op::Select(xneg, pi_signed, zero));
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, a, adj))
    }

    /// `hypot(a, b) = √(a² + b²)`, the overflow-safe 2-norm/magnitude (gradient norms, complex modulus,
    /// 2-D distance). Scaled by `m = max(|a|, |b|)` so `(a/m)² + (b/m)² ≤ 2` never overflows; the `m = 0`
    /// case (both zero) is guarded to `0` (else `0/0 = NaN`). All primitive ops, so it vectorizes and is
    /// bit-identical across backends. Works scalar or vector.
    fn emit_hypot(&mut self, a: ValueId, b: ValueId, rty: &MirType) -> ValueId {
        let mty = mask_ty(rty);
        let aa = self.emit_abs(a, rty);
        let bb = self.emit_abs(b, rty);
        let agtb = self
            .builder
            .build(mty.clone(), Op::Cmp(CmpOp::Fogt, aa, bb));
        let m = self.builder.build(rty.clone(), Op::Select(agtb, aa, bb));
        let ra = self.builder.build(rty.clone(), Op::Bin(BinOp::FDiv, a, m));
        let rb = self.builder.build(rty.clone(), Op::Bin(BinOp::FDiv, b, m));
        let ra2 = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, ra, ra));
        let sum = self.builder.build(rty.clone(), Op::Fma(rb, rb, ra2)); // rb² + ra²
        let root = self.builder.build(rty.clone(), Op::Sqrt(sum));
        let scaled = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, m, root));
        let zero = self.splat_const_f(0.0, rty);
        let mzero = self.builder.build(mty, Op::Cmp(CmpOp::Foeq, m, zero));
        self.builder
            .build(rty.clone(), Op::Select(mzero, zero, scaled))
    }

    /// The exp polynomial in `f32` (`fty` is `f32` or a `Vec` of `f32`). Range-reduces `x` to
    /// `r = x - n*ln2`, evaluates a degree-5 minimax poly for `e^r`, then scales by `2^n` (assembled
    /// from the IEEE-754 exponent bits). Every step is a primitive op the two backends already agree
    /// on bit-for-bit, so `exp` does too.
    fn emit_exp_f32(&mut self, x: ValueId, fty: &MirType) -> ValueId {
        let lanes = match fty {
            MirType::Vec(_, n) => Some(*n),
            _ => None,
        };
        let ity = match lanes {
            Some(n) => MirType::Vec(Box::new(MirType::I32), n),
            None => MirType::I32,
        };
        let mty = mask_ty(fty);

        // Clamp so 2^n stays representable (exp under/overflows to 0 / +inf outside this range).
        let hi = self.splat_const_f(EXP_HI, fty);
        let gt = self.builder.build(mty.clone(), Op::Cmp(CmpOp::Fogt, x, hi));
        let x = self.builder.build(fty.clone(), Op::Select(gt, hi, x));
        let lo = self.splat_const_f(EXP_LO, fty);
        let lt = self.builder.build(mty.clone(), Op::Cmp(CmpOp::Folt, x, lo));
        let x = self.builder.build(fty.clone(), Op::Select(lt, lo, x));

        // n = round(x * log2(e)) via the add-magic / sub-magic trick: round-to-nearest-even in
        // pure f32, so both backends agree and no rounding-mode instruction is needed.
        let log2e = self.splat_const_f(LOG2EF, fty);
        let magic = self.splat_const_f(EXP_MAGIC, fty);
        let t = self.builder.build(fty.clone(), Op::Fma(x, log2e, magic));
        let n = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, t, magic));

        // r = x - n*ln2, with ln2 split into hi/lo parts for extra precision (two fmas).
        let neg_c1 = self.splat_const_f(-EXP_C1, fty);
        let neg_c2 = self.splat_const_f(-EXP_C2, fty);
        let r = self.builder.build(fty.clone(), Op::Fma(n, neg_c1, x));
        let r = self.builder.build(fty.clone(), Op::Fma(n, neg_c2, r));

        // Degree-5 minimax polynomial for e^r on the reduced range, by Horner with fmas.
        let mut p = self.splat_const_f(EXP_P[0], fty);
        for &c in &EXP_P[1..] {
            let cc = self.splat_const_f(c, fty);
            p = self.builder.build(fty.clone(), Op::Fma(p, r, cc));
        }
        // e^r = p*r^2 + r + 1
        let r2 = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, r, r));
        let p = self.builder.build(fty.clone(), Op::Fma(p, r2, r));
        let one = self.splat_const_f(1.0, fty);
        let p = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FAdd, p, one));

        // 2^n by assembling the IEEE-754 exponent field: bitcast((n + 127) << 23).
        let ni = self
            .builder
            .build(ity.clone(), Op::Cast(CastKind::FpToSi, n, ity.clone()));
        let bias = self.splat_const_i(127, &ity);
        let biased = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, ni, bias));
        // `<< 23` written as `* 2^23`: keeps both Bin operands the same (vector) type. Cranelift's
        // vector `ishl` requires a *scalar* shift amount, but `imul` takes two vectors; since
        // `n + 127 <= 254` the product never overflows i32, so it equals the shift bit-for-bit.
        let pow = self.splat_const_i(8_388_608, &ity);
        let shifted = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Mul, biased, pow));
        let pow2 = self.builder.build(
            fty.clone(),
            Op::Cast(CastKind::Bitcast, shifted, fty.clone()),
        );

        self.builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, p, pow2))
    }

    /// `log(x)` (natural log) as a fast, deterministic polynomial (≈1 ULP of the true `log`). Always
    /// computed in `f32`; for an `f64` result the argument is demoted and the result promoted (the
    /// same f32-precision model `exp` uses). Works on a scalar or a SIMD-vector `f32`.
    fn emit_log(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let want_f64 = matches!(rty.lane_type(), MirType::F64);
        let f32ty = float_ty_like(rty, MirType::F32);
        let xf = if want_f64 {
            self.builder
                .build(f32ty.clone(), Op::Cast(CastKind::FpTrunc, x, f32ty.clone()))
        } else {
            x
        };
        let r = self.emit_log_f32(xf, &f32ty);
        if want_f64 {
            self.builder
                .build(rty.clone(), Op::Cast(CastKind::FpExt, r, rty.clone()))
        } else {
            r
        }
    }

    /// The natural-log polynomial in `f32` (`fty` is `f32` or a `Vec` of `f32`). Decomposes
    /// `x = m·2^e` with `m ∈ [0.5,1)` by IEEE-754 bit surgery — **no shift**: the exponent field is
    /// masked off (so the integer is `exp_field·2^23`, exact in `f32` for the ≤8-bit field), widened
    /// and scaled by `2^-23` to recover the count; the mantissa is OR-ed with biased exponent 126.
    /// Then a Cephes degree-8 minimax poly gives `log(m)`, and `e·ln2` is added back (same `ln2`
    /// split as `exp`). Every step is a primitive op both backends agree on bit-for-bit, so `log`
    /// does too. Assumes `x > 0` (like the rest of the kernels, no domain guard).
    fn emit_log_f32(&mut self, x: ValueId, fty: &MirType) -> ValueId {
        let lanes = match fty {
            MirType::Vec(_, n) => Some(*n),
            _ => None,
        };
        let ity = match lanes {
            Some(n) => MirType::Vec(Box::new(MirType::I32), n),
            None => MirType::I32,
        };
        let mty = mask_ty(fty);

        let bits = self
            .builder
            .build(ity.clone(), Op::Cast(CastKind::Bitcast, x, ity.clone()));

        // e = (float)(exponent_field) - 126, without a shift: keep only the exponent bits (value is
        // `exp_field·2^23`, exact in f32), widen to f32, scale by 2^-23.
        let expmask = self.splat_const_i(0x7F80_0000, &ity);
        let epart = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::And, bits, expmask));
        let epart_f = self
            .builder
            .build(fty.clone(), Op::Cast(CastKind::SiToFp, epart, fty.clone()));
        let inv = self.splat_const_f(INV_2P23, fty);
        let efield = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, epart_f, inv));
        let bias = self.splat_const_f(126.0, fty);
        let mut e = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, efield, bias));

        // m = bitcast((bits & 0x007fffff) | 0x3f000000) — mantissa with biased exponent 126 → [0.5,1).
        let mmask = self.splat_const_i(0x007F_FFFF, &ity);
        let mant = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::And, bits, mmask));
        let half_exp = self.splat_const_i(0x3F00_0000, &ity);
        let mbits = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Or, mant, half_exp));
        let mut m = self
            .builder
            .build(fty.clone(), Op::Cast(CastKind::Bitcast, mbits, fty.clone()));

        // if m < √0.5: e -= 1; m = 2m - 1; else m -= 1  (branchless via select).
        let sqrthf = self.splat_const_f(LOG_SQRTHF, fty);
        let lt = self
            .builder
            .build(mty.clone(), Op::Cmp(CmpOp::Folt, m, sqrthf));
        let one = self.splat_const_f(1.0, fty);
        let m2 = self.builder.build(fty.clone(), Op::Bin(BinOp::FAdd, m, m));
        let m_lt = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, m2, one)); // 2m - 1
        let m_ge = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, m, one)); // m - 1
        m = self.builder.build(fty.clone(), Op::Select(lt, m_lt, m_ge));
        let e_dec = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, e, one)); // e - 1
        e = self.builder.build(fty.clone(), Op::Select(lt, e_dec, e));

        // Degree-8 minimax poly for log(m) on the reduced range, Horner via fmas, then × m × z.
        let z = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, m, m));
        let mut p = self.splat_const_f(LOG_P[0], fty);
        for &c in &LOG_P[1..] {
            let cc = self.splat_const_f(c, fty);
            p = self.builder.build(fty.clone(), Op::Fma(p, m, cc));
        }
        let pm = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, p, m));
        let mut y = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, pm, z));

        // y += e·C2 (ln2 low);  y -= 0.5·z
        let c2 = self.splat_const_f(EXP_C2, fty);
        y = self.builder.build(fty.clone(), Op::Fma(e, c2, y));
        let neg_half = self.splat_const_f(-0.5, fty);
        y = self.builder.build(fty.clone(), Op::Fma(z, neg_half, y));

        // r = m + y + e·C1 (ln2 high)
        let r = self.builder.build(fty.clone(), Op::Bin(BinOp::FAdd, m, y));
        let c1 = self.splat_const_f(EXP_C1, fty);
        self.builder.build(fty.clone(), Op::Fma(e, c1, r))
    }

    /// Build a float constant of type `fty` — a scalar `ConstFloat`, or one splatted to a vector.
    fn splat_const_f(&mut self, val: f64, fty: &MirType) -> ValueId {
        match fty {
            MirType::Vec(lane, _) => {
                let s = self
                    .builder
                    .build((**lane).clone(), Op::ConstFloat(val, (**lane).clone()));
                self.builder.build(fty.clone(), Op::Splat(s))
            }
            _ => self
                .builder
                .build(fty.clone(), Op::ConstFloat(val, fty.clone())),
        }
    }

    /// Build an int constant of type `ity` — a scalar `ConstInt`, or one splatted to a vector.
    fn splat_const_i(&mut self, val: i128, ity: &MirType) -> ValueId {
        match ity {
            MirType::Vec(lane, _) => {
                let s = self
                    .builder
                    .build((**lane).clone(), Op::ConstInt(val, (**lane).clone()));
                self.builder.build(ity.clone(), Op::Splat(s))
            }
            _ => self
                .builder
                .build(ity.clone(), Op::ConstInt(val, ity.clone())),
        }
    }

    fn lower_cast(&mut self, operand: &Expr, e: &Expr) -> ValueId {
        let v = self.lower_expr(operand);
        let from = self.expr_mir(operand);
        let to = self.expr_mir(e);
        if from == to {
            return v;
        }
        // For a float→int cast the signed/unsigned choice comes from the TARGET integer (`x as i32`
        // is signed → fptosi); for every other direction (int→float, int widening) it comes from the
        // source operand. Using the operand's signedness for float→int picks fptoui, where the native
        // backend saturates a negative float to 0 while the interpreter keeps the signed value — a
        // native≠interpreter divergence (e.g. `(-0.5 * 10.0) as i32` gave 0 on native, −5 on interp).
        let signed = if from.is_float() && to.is_int() {
            self.signed(e)
        } else {
            self.signed(operand)
        };
        let kind = cast_kind(&from, &to, signed);
        self.builder.build(to.clone(), Op::Cast(kind, v, to))
    }
}

// ---- free helpers ----

/// The MIR type a parameter is passed as at the call boundary. Arrays decay to a base pointer.
fn param_abi_ty(ty: &Ty) -> MirType {
    match mir_ty(ty) {
        MirType::Array(..) => MirType::Ptr,
        t => t,
    }
}

/// Round `x` up to the next multiple of `align` (a power of two) — field-offset padding, mirroring
/// the private `round_up` in `mercury_types` (the registry-aware aggregate layout reimplements the
/// same accumulation here because it must resolve named-struct field sizes the leaf crate can't).
fn round_up(x: u64, align: u64) -> u64 {
    if align <= 1 {
        x
    } else {
        (x + align - 1) & !(align - 1)
    }
}

/// Byte size of a MIR type (scalars by width, an array/vector as count × element). Used to size a
/// no-init tuple's byte buffer from its element MIR types (`mir_ty_of_ann`). Matches `Ty::size_of`'s
/// scalar widths so the buffer is identical to the with-initializer path's `ty_size`.
fn mir_byte_size(t: &MirType) -> u64 {
    match t {
        MirType::I1 | MirType::I8 => 1,
        MirType::I16 | MirType::F16 | MirType::BF16 => 2,
        MirType::I32 | MirType::F32 => 4,
        MirType::I64 | MirType::F64 | MirType::Ptr => 8,
        MirType::Vec(e, n) | MirType::Array(e, n) => mir_byte_size(e) * (*n as u64),
        MirType::Void => 0,
    }
}

/// Alignment of a MIR type: a scalar aligns to its size, an array/vector to its element. Companion
/// to [`mir_byte_size`] for padding a no-init tuple's byte buffer.
fn mir_byte_align(t: &MirType) -> u64 {
    match t {
        MirType::Vec(e, _) | MirType::Array(e, _) => mir_byte_align(e),
        other => mir_byte_size(other).max(1),
    }
}

fn mir_ty(ty: &Ty) -> MirType {
    match ty {
        Ty::Scalar(s) => MirType::from_scalar(*s),
        Ty::Array { elem, len } => MirType::Array(Box::new(mir_ty(elem)), *len as u32),
        Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Tensor { .. } | Ty::Slice(_) => MirType::Ptr,
        Ty::Vector { elem, lanes } => MirType::Vec(Box::new(MirType::from_scalar(*elem)), *lanes),
        Ty::Unit => MirType::Void,
        // A tuple (and any other aggregate) is a flat byte buffer; its local *value* is the base
        // pointer (like an array), and field access GEPs to a padded byte offset. `tuple_offsets`
        // is the layout authority. Falls back to a 0-byte buffer for an unsized field (never read).
        Ty::Tuple(_) => MirType::Array(
            Box::new(MirType::I8),
            ty.size_of().unwrap_or(0) as u32,
        ),
        _ => MirType::I32,
    }
}

fn mir_ty_of_ast(t: &ast::TypeExpr, interner: &Interner) -> MirType {
    use ast::TypeKind::*;
    match &t.kind {
        Path(p) => {
            let name = interner.resolve(p.segments.last().unwrap().sym);
            match mercury_types::Scalar::from_name(name) {
                Some(s) => MirType::from_scalar(s),
                None => MirType::I32,
            }
        }
        Array { elem, len } => match const_usize_expr(len, interner) {
            // A literal-length array lowers to an array type; otherwise fall back to an opaque ptr.
            Some(n) => MirType::Array(Box::new(mir_ty_of_ast(elem, interner)), n),
            None => MirType::Ptr,
        },
        Pointer { .. } | Ref { .. } | Slice(_) | Tensor { .. } => MirType::Ptr,
        Vector { elem, lanes } => {
            let e = mir_ty_of_ast(elem, interner);
            MirType::Vec(Box::new(e), *lanes)
        }
        Unit => MirType::Void,
        _ => MirType::I32,
    }
}

/// Evaluate a compile-time array length that is a plain integer literal.
fn const_usize_expr(e: &Expr, interner: &Interner) -> Option<u32> {
    match &e.kind {
        ExprKind::Int(s) => Some(parse_int(interner.resolve(*s)) as u32),
        _ => None,
    }
}

fn arith_binop(op: ast::BinOp, float: bool, signed: bool) -> BinOp {
    use ast::BinOp as A;
    match op {
        A::Add => {
            if float {
                BinOp::FAdd
            } else {
                BinOp::Add
            }
        }
        A::Sub => {
            if float {
                BinOp::FSub
            } else {
                BinOp::Sub
            }
        }
        A::Mul => {
            if float {
                BinOp::FMul
            } else {
                BinOp::Mul
            }
        }
        A::Div => {
            if float {
                BinOp::FDiv
            } else if signed {
                BinOp::SDiv
            } else {
                BinOp::UDiv
            }
        }
        A::Rem => {
            if float {
                BinOp::FRem
            } else if signed {
                BinOp::SRem
            } else {
                BinOp::URem
            }
        }
        A::BitAnd => BinOp::And,
        A::BitOr => BinOp::Or,
        A::BitXor => BinOp::Xor,
        A::Shl => BinOp::Shl,
        A::Shr => {
            if signed {
                BinOp::AShr
            } else {
                BinOp::LShr
            }
        }
        _ => BinOp::Add,
    }
}

fn compound_binop(op: ast::AssignOp, float: bool, signed: bool) -> BinOp {
    use ast::AssignOp as A;
    let bin = match op {
        A::Add => ast::BinOp::Add,
        A::Sub => ast::BinOp::Sub,
        A::Mul => ast::BinOp::Mul,
        A::Div => ast::BinOp::Div,
        A::Rem => ast::BinOp::Rem,
        A::BitAnd => ast::BinOp::BitAnd,
        A::BitOr => ast::BinOp::BitOr,
        A::BitXor => ast::BinOp::BitXor,
        A::Shl => ast::BinOp::Shl,
        A::Shr => ast::BinOp::Shr,
        A::Assign => ast::BinOp::Add,
    };
    arith_binop(bin, float, signed)
}

fn cmp_pred(op: ast::BinOp, float: bool, signed: bool) -> CmpOp {
    use ast::BinOp as A;
    match op {
        A::Eq => {
            if float {
                CmpOp::Foeq
            } else {
                CmpOp::Eq
            }
        }
        A::Ne => {
            if float {
                CmpOp::Fone
            } else {
                CmpOp::Ne
            }
        }
        A::Lt => float_or(float, CmpOp::Folt, signed, CmpOp::Slt, CmpOp::Ult),
        A::Le => float_or(float, CmpOp::Fole, signed, CmpOp::Sle, CmpOp::Ule),
        A::Gt => float_or(float, CmpOp::Fogt, signed, CmpOp::Sgt, CmpOp::Ugt),
        A::Ge => float_or(float, CmpOp::Foge, signed, CmpOp::Sge, CmpOp::Uge),
        _ => CmpOp::Eq,
    }
}

fn float_or(float: bool, f: CmpOp, signed: bool, s: CmpOp, u: CmpOp) -> CmpOp {
    if float {
        f
    } else if signed {
        s
    } else {
        u
    }
}

fn is_numeric(t: &MirType) -> bool {
    t.is_int() || t.is_float()
}

// ---- vectorizer analysis helpers (pure AST/type predicates) ----

/// The symbol of a single-segment path expression, if `e` is one.
fn single_path(e: &Expr) -> Option<Symbol> {
    match &e.kind {
        ExprKind::Path(p) if p.is_single() => Some(p.first().sym),
        _ => None,
    }
}

/// The single statement of a one-statement, tail-less block (the shape every per-pass norm loop body
/// has); `None` otherwise.
fn single_stmt(body: &Block) -> Option<&Stmt> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    Some(&body.stmts[0])
}

/// Does `e` reference any symbol in `set`? (Used to keep inner vector temps out of index exprs.)
fn uses_any(e: &Expr, set: &HashSet<Symbol>) -> bool {
    match &e.kind {
        ExprKind::Path(p) => p.is_single() && set.contains(&p.first().sym),
        ExprKind::Binary { lhs, rhs, .. } => uses_any(lhs, set) || uses_any(rhs, set),
        ExprKind::Unary { expr, .. } => uses_any(expr, set),
        ExprKind::Index { base, indices } => {
            uses_any(base, set) || indices.iter().any(|i| uses_any(i, set))
        }
        ExprKind::Cast { expr, .. } => uses_any(expr, set),
        _ => false,
    }
}

/// Does `e` mention the loop variable `sym`?
fn expr_uses_sym(e: &Expr, sym: Symbol) -> bool {
    match &e.kind {
        ExprKind::Path(p) => p.is_single() && p.first().sym == sym,
        ExprKind::Binary { lhs, rhs, .. } => expr_uses_sym(lhs, sym) || expr_uses_sym(rhs, sym),
        ExprKind::Unary { expr, .. } => expr_uses_sym(expr, sym),
        ExprKind::Index { base, indices } => {
            expr_uses_sym(base, sym) || indices.iter().any(|i| expr_uses_sym(i, sym))
        }
        ExprKind::Cast { expr, .. } => expr_uses_sym(expr, sym),
        _ => false,
    }
}

/// Whole-expression "does this reference `sym`?" — unlike [`expr_uses_sym`] (which only walks the
/// index/arith subset), this recurses through calls, fields, and casts, and is **conservative**:
/// expression kinds it does not model return `true`. Used to prove a fused region's internal scalars
/// do not leak to later code (a `true` just means "can't prove it's safe", so we decline to fuse).
fn expr_mentions(e: &Expr, sym: Symbol) -> bool {
    match &e.kind {
        ExprKind::Path(p) => p.is_single() && p.first().sym == sym,
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_) => false,
        ExprKind::Unary { expr, .. } => expr_mentions(expr, sym),
        ExprKind::Binary { lhs, rhs, .. } => expr_mentions(lhs, sym) || expr_mentions(rhs, sym),
        ExprKind::Call { callee, args, .. } => {
            expr_mentions(callee, sym) || args.iter().any(|a| expr_mentions(a, sym))
        }
        ExprKind::Index { base, indices } => {
            expr_mentions(base, sym) || indices.iter().any(|i| expr_mentions(i, sym))
        }
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => {
            expr_mentions(base, sym)
        }
        ExprKind::Cast { expr, .. } => expr_mentions(expr, sym),
        // struct/array literals, if/match/block exprs, method calls, ranges, …: assume a use.
        _ => true,
    }
}

/// Conservative "does any statement (or the tail) reference `sym`?" — companion to [`expr_mentions`].
fn block_mentions(stmts: &[Stmt], tail: Option<&Expr>, sym: Symbol) -> bool {
    stmts.iter().any(|s| stmt_mentions(s, sym)) || tail.is_some_and(|e| expr_mentions(e, sym))
}

fn stmt_mentions(s: &Stmt, sym: Symbol) -> bool {
    match &s.kind {
        StmtKind::Let { init, .. } => init.as_ref().is_some_and(|e| expr_mentions(e, sym)),
        StmtKind::Assign { target, value, .. } => {
            expr_mentions(target, sym) || expr_mentions(value, sym)
        }
        StmtKind::Expr(e) | StmtKind::Defer(e) => expr_mentions(e, sym),
        StmtKind::Return(o) => o.as_ref().is_some_and(|e| expr_mentions(e, sym)),
        StmtKind::Break(_) | StmtKind::Continue(_) => false,
        StmtKind::While { cond, body, .. } => {
            expr_mentions(cond, sym) || block_mentions(&body.stmts, body.tail.as_deref(), sym)
        }
        StmtKind::For { iter, body, .. } => {
            for_iter_mentions(iter, sym) || block_mentions(&body.stmts, body.tail.as_deref(), sym)
        }
        StmtKind::Loop { body, .. } => block_mentions(&body.stmts, body.tail.as_deref(), sym),
    }
}

fn for_iter_mentions(it: &ForIter, sym: Symbol) -> bool {
    match it {
        ForIter::Range {
            start, end, step, ..
        } => {
            expr_mentions(start, sym)
                || end.as_ref().is_some_and(|e| expr_mentions(e, sym))
                || step.as_ref().is_some_and(|e| expr_mentions(e, sym))
        }
        ForIter::Expr(e) => expr_mentions(e, sym),
    }
}

/// The coefficient of `j` in an affine index expression: `Some(0)` invariant, `Some(1)` unit-stride,
/// other constants for non-unit strides, `None` if not provably affine in `j`. Only `+`/`-` combine
/// `j`; a `*` involving `j` is conservatively rejected (we don't constant-evaluate factors).
fn affine_stride(e: &Expr, j: Symbol) -> Option<i64> {
    if !expr_uses_sym(e, j) {
        return Some(0);
    }
    match &e.kind {
        ExprKind::Path(p) if p.is_single() && p.first().sym == j => Some(1),
        ExprKind::Binary { op, lhs, rhs } => match op {
            ast::BinOp::Add => affine_stride(lhs, j)?.checked_add(affine_stride(rhs, j)?),
            ast::BinOp::Sub => affine_stride(lhs, j)?.checked_sub(affine_stride(rhs, j)?),
            _ => None,
        },
        _ => None,
    }
}

/// Structural equality over the elementwise-expression subset (ints, floats, single paths, unary,
/// `+`/`-`/`*`, and unit-stride `Index`). Used to confirm a written array is touched at exactly one
/// index (no loop-carried dependence) and that a ReLU's then-branch returns the value its guard
/// compares (`if v > 0 { v } else { 0 }`). The `Index` arm is what lets `x[i] == x[i]` hold.
fn exprs_struct_eq(a: &Expr, b: &Expr) -> bool {
    match (&a.kind, &b.kind) {
        (ExprKind::Int(x), ExprKind::Int(y)) => x == y,
        (ExprKind::Float(x), ExprKind::Float(y)) => x == y,
        (ExprKind::Path(p), ExprKind::Path(q)) => {
            p.is_single() && q.is_single() && p.first().sym == q.first().sym
        }
        (ExprKind::Unary { op: o1, expr: e1 }, ExprKind::Unary { op: o2, expr: e2 }) => {
            o1 == o2 && exprs_struct_eq(e1, e2)
        }
        (
            ExprKind::Binary {
                op: o1,
                lhs: l1,
                rhs: r1,
            },
            ExprKind::Binary {
                op: o2,
                lhs: l2,
                rhs: r2,
            },
        ) => o1 == o2 && exprs_struct_eq(l1, l2) && exprs_struct_eq(r1, r2),
        (
            ExprKind::Index {
                base: b1,
                indices: i1,
            },
            ExprKind::Index {
                base: b2,
                indices: i2,
            },
        ) => {
            i1.len() == i2.len()
                && exprs_struct_eq(b1, b2)
                && i1.iter().zip(i2).all(|(x, y)| exprs_struct_eq(x, y))
        }
        _ => false,
    }
}

/// Pin a shared lane type: set it if unset, else require equality. `None` means a type clash (bail).
fn set_or_check(slot: &mut Option<MirType>, t: &MirType) -> Option<()> {
    match slot {
        Some(existing) if existing == t => Some(()),
        Some(_) => None,
        None => {
            *slot = Some(t.clone());
            Some(())
        }
    }
}

// --- matmul recognition ---------------------------------------------------------------------
//
// A naive matmul loop nest is the single most important ML kernel and the one a general C/Rust
// compiler optimizes least (it vectorizes the inner loop but never register-blocks, cache-tiles, or
// packs). Mercury recognizes the canonical f32 `ikj` nest and lowers the *whole nest* to a call into
// the tuned `mercury_sgemm` microkernel (true 256-bit AVX2/FMA) — exactly how XLA/TVM/oneDNN lower a
// matmul op. The interpreter runs the identical kernel via marshalling, so the two stay bit-exact.

/// A recognized GEMM nest computing `C[m,n] = A[m,k]·B[k,n]` (row-major, contiguous, f32), or its
/// `nn.Linear` transpose `C[m,n] = A[m,k]·B[n,k]ᵀ` when `transposed`.
struct MatmulNest<'a> {
    a: Symbol,
    b: Symbol,
    c: Symbol,
    m: Dim,
    k: Dim,
    n: Dim,
    /// 0 = overwrite C (a zero-init loop was present), 1 = accumulate into C.
    beta: i64,
    /// `true` for `C = A·Bᵀ` (B indexed `[j,k]` instead of `[k,j]`).
    transposed: bool,
    /// `true` for `C = Aᵀ·B` (A indexed `[k,i]` instead of `[i,k]`) — the `dW = dYᵀ·X` weight-gradient
    /// GEMM. Only the `ijk` dot-product form recognizes it, and `emit_sgemm` requires it be mutually
    /// exclusive with `transposed` and offset-free (the both-transposed `Aᵀ·Bᵀ` and a batched
    /// transposed-A have no kernel, so they fall back to the scalar nest).
    transposed_a: bool,
    /// Per-operand base offsets: the additive index terms left over after the 2-D `row*stride + col`
    /// is peeled off — e.g. the batch/head term `h*S*D` of a **batched** matmul (multi-head
    /// attention is one matmul per head: `scores[h] = Q[h]·K[h]ᵀ`). Each term is verified invariant
    /// in the matmul's `(i,j,k)`, so `emit_sgemm` simply GEPs the base pointer by their sum before
    /// the kernel call — the inner matmul is identical regardless of the offset. Empty for a plain
    /// 2-D matmul; only the `ijk` dot-product form populates these.
    a_off: Vec<&'a Expr>,
    b_off: Vec<&'a Expr>,
    c_off: Vec<&'a Expr>,
}

/// Canonical text of an affine index/base expression (paths, ints, `+`/`-`/`*`, casts), used to key
/// the vectorizer's per-body load cache. `None` for anything outside that subset (not cached).
fn index_canon(e: &Expr, interner: &Interner) -> Option<String> {
    match &e.kind {
        ExprKind::Path(p) if p.is_single() => Some(interner.resolve(p.first().sym).to_string()),
        ExprKind::Int(s) => Some(interner.resolve(*s).to_string()),
        ExprKind::Binary { op, lhs, rhs } => Some(format!(
            "({} {} {})",
            index_canon(lhs, interner)?,
            op.glyph(),
            index_canon(rhs, interner)?
        )),
        ExprKind::Cast { expr, .. } => index_canon(expr, interner),
        _ => None,
    }
}

/// A cache key identifying the memory a `base[index]` read touches (`base` is a single path).
fn load_key(base: &Expr, index: &Expr, interner: &Interner) -> Option<String> {
    Some(format!(
        "{}[{}]",
        index_canon(base, interner)?,
        index_canon(index, interner)?
    ))
}

/// A non-negative integer literal.
fn as_int_lit(e: &Expr, interner: &Interner) -> Option<i64> {
    match &e.kind {
        ExprKind::Int(s) => i64::try_from(parse_int(interner.resolve(*s))).ok(),
        _ => None,
    }
}

/// A matmul dimension or stride: a compile-time literal, or a runtime variable (function param or
/// local). Two `Var`s compare equal iff they name the same binding, so the recognizer's stride/bound
/// consistency checks (`sa == k`, `sc == n`, …) hold symbolically — which is what lets a matmul with
/// runtime dimensions dispatch to the tuned GEMM kernel instead of falling back to a scalar nest.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dim {
    Lit(i64),
    Var(Symbol),
}

/// A non-negative dimension/stride expression: an integer literal or a single variable path.
fn as_dim(e: &Expr, interner: &Interner) -> Option<Dim> {
    if let Some(v) = as_int_lit(e, interner) {
        return Some(Dim::Lit(v));
    }
    single_path(e).map(Dim::Var)
}

/// `e == row * stride` (either factor order) for the *known* row variable; returns the stride. The
/// known row resolves the otherwise-ambiguous `i * N` form (both factors are paths under symbolic
/// dims) — the factor that is `row` is the index, the other is the stride.
fn mul_with_row(e: &Expr, row: Symbol, interner: &Interner) -> Option<Dim> {
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &e.kind
    else {
        return None;
    };
    if single_path(lhs) == Some(row) {
        return as_dim(rhs, interner);
    }
    if single_path(rhs) == Some(row) {
        return as_dim(lhs, interner);
    }
    None
}

/// A flattened 2-D index `row * stride + col` (either addend order) for a known `row`. Returns
/// `(stride, col)`.
fn match_row_col(idx: &Expr, row: Symbol, interner: &Interner) -> Option<(Dim, Symbol)> {
    let ExprKind::Binary {
        op: ast::BinOp::Add,
        lhs,
        rhs,
    } = &idx.kind
    else {
        return None;
    };
    if let (Some(stride), Some(col)) = (mul_with_row(lhs, row, interner), single_path(rhs)) {
        return Some((stride, col));
    }
    if let (Some(col), Some(stride)) = (single_path(lhs), mul_with_row(rhs, row, interner)) {
        return Some((stride, col));
    }
    None
}

/// Flatten the additive terms of `e`, recursing only through `+`. `i*K + k + h*S*D` yields the three
/// terms `[i*K, k, h*S*D]` (left-association is irrelevant). Used to peel a batch/base offset off a
/// flattened tensor index.
fn flatten_add_terms<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    if let ExprKind::Binary {
        op: ast::BinOp::Add,
        lhs,
        rhs,
    } = &e.kind
    {
        flatten_add_terms(lhs, out);
        flatten_add_terms(rhs, out);
    } else {
        out.push(e);
    }
}

/// Like [`match_row_col`] but tolerant of a leading **base offset**: parses
/// `idx == row*stride + col + offset_terms…` for a known `row` and a known expected `col` symbol
/// (the matmul column that the caller already knows must appear). Returns `(stride, offset_terms)`,
/// where `offset_terms` are the remaining addends (empty for a plain 2-D index). The caller must
/// verify the offset is invariant in the matmul's bound variables. Passing the expected `col` is what
/// disambiguates the bare column term from an offset that is itself a bare path.
fn match_row_col_off<'a>(
    idx: &'a Expr,
    row: Symbol,
    col: Symbol,
    interner: &Interner,
) -> Option<(Dim, Vec<&'a Expr>)> {
    let mut terms = Vec::new();
    flatten_add_terms(idx, &mut terms);
    // The unique `row * stride` term.
    let row_pos = terms
        .iter()
        .position(|t| mul_with_row(t, row, interner).is_some())?;
    let stride = mul_with_row(terms[row_pos], row, interner)?;
    terms.remove(row_pos);
    // The bare column term `col` (must be present exactly as the expected symbol).
    let col_pos = terms.iter().position(|t| single_path(t) == Some(col))?;
    terms.remove(col_pos);
    // Whatever is left is the base offset (a batch/head index for a batched matmul).
    Some((stride, terms))
}

/// `base[index]` with a single-segment `base` path and exactly one index. Returns `(base, index)`.
fn as_index1(e: &Expr) -> Option<(Symbol, &Expr)> {
    match &e.kind {
        ExprKind::Index { base, indices } if indices.len() == 1 => {
            Some((single_path(base)?, &indices[0]))
        }
        _ => None,
    }
}

/// Is `e` the float literal `0.0` (any spelling)?
fn is_float_zero(e: &Expr, interner: &Interner) -> bool {
    matches!(&e.kind, ExprKind::Float(s) if parse_float(interner.resolve(*s)) == 0.0)
}

/// Is `e`'s sema type `f32`? (matmul lowering is f32-only — that's what `mercury_sgemm` computes.)
fn is_f32_expr(e: &Expr, sema: &SemaResult) -> bool {
    matches!(sema.types.get(&e.id), Some(t) if mir_ty(t) == MirType::F32)
}

/// `for col in 0..n { c[row*stride + col] = 0.0; }` (or the 2-index `c[row, col] = 0.0`) — the per-row
/// zero-init of a beta-0 matmul. Returns `(c, stride, n)` with the outer row variable `row`.
fn match_zero_init(
    s: &Stmt,
    row: Symbol,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(Symbol, Dim, Dim)> {
    let (pat, iter, body) = fusable_for(s)?;
    let col = match &pat.kind {
        ast::PatKind::Ident(c) => *c,
        _ => return None,
    };
    let (start, end) = range_bounds(iter)?;
    if as_int_lit(start, interner)? != 0 {
        return None;
    }
    let n = as_dim(end, interner)?;
    if body.stmts.len() != 1 || body.tail.is_some() {
        return None;
    }
    let StmtKind::Assign {
        target,
        op: ast::AssignOp::Assign,
        value,
    } = &body.stmts[0].kind
    else {
        return None;
    };
    if !is_float_zero(value, interner) {
        return None;
    }
    let (cbase, stride, cc) = match_operand_row_then_col(target, row, sema, interner)?;
    if cc != col {
        return None;
    }
    Some((cbase, stride, n))
}

/// Classify a `Mul` product as `A·B`, returning `(a, sa, b, sb, transposed)`. `aik` carries an
/// optional pre-bound `let aik = A[row*sa+k]` (the `ikj` form); otherwise A is found inline. B is
/// `B[k*N+j]` (normal) or `B[j*K+k]` (transposed — the nn.Linear `A·Bᵀ`). Either factor order.
fn match_product_ab(
    prod: &Expr,
    row: Symbol,
    kvar: Symbol,
    jvar: Symbol,
    aik: Option<(Symbol, Symbol, Dim)>,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(Symbol, Dim, Symbol, Dim, bool)> {
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: f1,
        rhs: f2,
    } = &prod.kind
    else {
        return None;
    };
    let is_a = |f: &Expr| -> Option<(Symbol, Dim)> {
        if let Some((aik_sym, asym, sa)) = aik {
            if single_path(f) == Some(aik_sym) {
                return Some((asym, sa));
            }
        }
        let (abase, asa, ak) = match_operand_row_then_col(f, row, sema, interner)?;
        (ak == kvar).then_some((abase, asa))
    };
    let is_b = |f: &Expr| -> Option<(Symbol, Dim, bool)> {
        // normal `B[k*N+j]` / `B[k,j]`: row is k, col is j; transposed `B[j*K+k]` / `B[j,k]`: row j, col k.
        if let Some((bbase, sb, bc)) = match_operand_row_then_col(f, kvar, sema, interner) {
            if bc == jvar {
                return Some((bbase, sb, false));
            }
        }
        if let Some((bbase, sb, bc)) = match_operand_row_then_col(f, jvar, sema, interner) {
            if bc == kvar {
                return Some((bbase, sb, true));
            }
        }
        None
    };
    let pair = |fa: &Expr, fb: &Expr| match (is_a(fa), is_b(fb)) {
        (Some((a, sa)), Some((b, sb, t))) => Some((a, sa, b, sb, t)),
        _ => None,
    };
    pair(f1, f2).or_else(|| pair(f2, f1))
}

/// The row-major inner-dimension stride of a **rank-2 contiguous tensor** operand, as a recognizer
/// `Dim` — for `A: Tensor[f32, M, N]` accessed `A[i, j]`, the axis-0 stride is the inner dim `N`
/// (a `Const` → `Dim::Lit`, a bound symbolic `Var` → `Dim::Var`). This lets the shape-typed 2-index
/// spelling `a[i, k]` supply the *same* stride the flat `a[i*K + k]` form derives from its index
/// arithmetic — so the idiomatic tensor matmul dispatches to the GEMM kernel. `None` for a
/// non-tensor, a non-contiguous layout, a non-rank-2 tensor, or a `Dynamic` (`?`) inner dim.
fn tensor_inner_stride(base: &Expr, sema: &SemaResult) -> Option<Dim> {
    let Some(Ty::Tensor { shape, layout, .. }) = sema.types.get(&base.id) else {
        return None;
    };
    if !matches!(layout, mercury_types::Layout::Contiguous) || shape.0.len() != 2 {
        return None;
    }
    match &shape.0[1] {
        mercury_types::Dim::Const(v) => Some(Dim::Lit(*v as i64)),
        mercury_types::Dim::Var(s) => Some(Dim::Var(*s)),
        mercury_types::Dim::Dynamic => None,
    }
}

/// Decompose a matmul operand access into `(base, row_stride, offset_terms)` for a known `row` index
/// and an expected `col` index. Accepts BOTH spellings:
///   * the flat form `base[row*stride + col (+ offset…)]` — stride read from the index arithmetic
///     (delegates to [`match_row_col_off`], so the flat path is byte-identical to before), and
///   * the shape-typed 2-index form `base[row, col]` — stride = the tensor's inner dim, no offset.
/// The 2-index branch is what makes `c[i,j] += a[i,k]*b[k,j]` dispatch to the tuned GEMM kernel
/// instead of running as a scalar nest. Both indices must be exactly the expected `row`/`col` vars
/// (a strided or offset 2-index access is not a plain matmul operand). `None` if neither shape matches.
fn match_operand_row_col_off<'a>(
    f: &'a Expr,
    row: Symbol,
    col: Symbol,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(Symbol, Dim, Vec<&'a Expr>)> {
    match &f.kind {
        ExprKind::Index { base, indices } if indices.len() == 1 => {
            let abase = single_path(base)?;
            let (stride, off) = match_row_col_off(&indices[0], row, col, interner)?;
            Some((abase, stride, off))
        }
        ExprKind::Index { base, indices } if indices.len() == 2 => {
            if single_path(&indices[0])? != row || single_path(&indices[1])? != col {
                return None;
            }
            let abase = single_path(base)?;
            let stride = tensor_inner_stride(base, sema)?;
            Some((abase, stride, Vec::new()))
        }
        _ => None,
    }
}

/// Like [`match_operand_row_col_off`] but for the offset-free `ikj` accumulate matmul, whose helpers
/// *discover* the column rather than knowing it in advance: `base[row*stride + col]` (flat) or
/// `base[row, col]` (2-index tensor, stride = inner dim) for a known `row`. Returns `(base, stride,
/// col)`. The 2-index branch is what dispatches `c[i,j] += a[i,k]*b[k,j]` written in tensor notation.
fn match_operand_row_then_col(
    f: &Expr,
    row: Symbol,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(Symbol, Dim, Symbol)> {
    match &f.kind {
        ExprKind::Index { base, indices } if indices.len() == 1 => {
            let abase = single_path(base)?;
            let (stride, col) = match_row_col(&indices[0], row, interner)?;
            Some((abase, stride, col))
        }
        ExprKind::Index { base, indices } if indices.len() == 2 => {
            if single_path(&indices[0])? != row {
                return None;
            }
            let col = single_path(&indices[1])?;
            let abase = single_path(base)?;
            let stride = tensor_inner_stride(base, sema)?;
            Some((abase, stride, col))
        }
        _ => None,
    }
}

/// An A factor of the inline `ijk` product: `A[row*sa + k (+ off)]` / `A[row, k]` (normal) or
/// `A[k*sa + row (+ off)]` / `A[k, row]` (transposed — the `dW = Aᵀ·B` weight-gradient spelling, the
/// contraction `k` being the outer index of A's storage). Returns `(base, sa, offset, transposed_a)`.
/// Both the flat and shape-typed 2-index spellings dispatch (via [`match_operand_row_col_off`]).
fn match_a_factor<'a>(
    f: &'a Expr,
    row: Symbol,
    kvar: Symbol,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(Symbol, Dim, Vec<&'a Expr>, bool)> {
    if let Some((abase, sa, off)) = match_operand_row_col_off(f, row, kvar, sema, interner) {
        return Some((abase, sa, off, false));
    }
    if let Some((abase, sa, off)) = match_operand_row_col_off(f, kvar, row, sema, interner) {
        return Some((abase, sa, off, true));
    }
    None
}

/// A B factor of the inline `ijk` product: `B[k*sb + j (+ off)]` / `B[k, j]` (normal) or
/// `B[j*sb + k (+ off)]` / `B[j, k]` (transposed — the `A·Bᵀ` `nn.Linear` spelling). Returns
/// `(base, sb, offset, transposed)`. Both flat and 2-index spellings dispatch.
fn match_b_factor<'a>(
    f: &'a Expr,
    kvar: Symbol,
    jvar: Symbol,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(Symbol, Dim, Vec<&'a Expr>, bool)> {
    if let Some((bbase, sb, off)) = match_operand_row_col_off(f, kvar, jvar, sema, interner) {
        return Some((bbase, sb, off, false));
    }
    if let Some((bbase, sb, off)) = match_operand_row_col_off(f, jvar, kvar, sema, interner) {
        return Some((bbase, sb, off, true));
    }
    None
}

/// Like [`match_product_ab`] but for the inline `ijk` form (A read directly, never via an `aik`
/// binding) and tolerant of a per-operand **base offset** (the batch/head index of a batched matmul).
/// Returns `(a, sa, a_off, b, sb, b_off, transposed, transposed_a)`.
#[allow(clippy::type_complexity)]
fn match_product_ab_off<'a>(
    prod: &'a Expr,
    row: Symbol,
    kvar: Symbol,
    jvar: Symbol,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(Symbol, Dim, Vec<&'a Expr>, Symbol, Dim, Vec<&'a Expr>, bool, bool)> {
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: f1,
        rhs: f2,
    } = &prod.kind
    else {
        return None;
    };
    // Either factor order: `A*B` or `B*A`.
    for (fa, fb) in [(f1, f2), (f2, f1)] {
        if let (Some((a, sa, aoff, ta)), Some((b, sb, boff, t))) = (
            match_a_factor(fa, row, kvar, sema, interner),
            match_b_factor(fb, kvar, jvar, sema, interner),
        ) {
            return Some((a, sa, aoff, b, sb, boff, t, ta));
        }
    }
    None
}

/// Every term of `off` is invariant in all of `vars` (the matmul's bound `i`/`j`/`k`). A base offset
/// that mentioned a loop variable would not be a constant per-call pointer shift, so it is rejected.
fn offset_invariant(off: &[&Expr], vars: &[Symbol]) -> bool {
    off.iter()
        .all(|t| vars.iter().all(|&v| !expr_uses_sym(t, v)))
}

/// Try both recognized matmul spellings: the `ikj` accumulate form and the `ijk` dot-product form.
fn recognize_matmul<'a>(
    pat: &Pattern,
    iter: &ForIter,
    body: &'a Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<MatmulNest<'a>> {
    match_matmul(pat, iter, body, sema, interner)
        .or_else(|| match_matmul_ijk(pat, iter, body, sema, interner))
}

// Fused-epilogue activation codes — must match `mercury_runtime`'s gemm kernel
// (ACT_IDENTITY/RELU/GELU/SILU).
const EPI_ACT_IDENTITY: u32 = 0;
const EPI_ACT_RELU: u32 = 1;
const EPI_ACT_GELU: u32 = 2;
const EPI_ACT_SILU: u32 = 3;

/// If `e` is `base_sym[idx]` (single index off the path `base_sym`), return `idx`.
fn index_of(e: &Expr, base_sym: Symbol) -> Option<&Expr> {
    if let ExprKind::Index { base, indices } = &e.kind {
        if indices.len() == 1 && single_path(base) == Some(base_sym) {
            return Some(&indices[0]);
        }
    }
    None
}

/// Is `e` the matmul output element `C[i*N+j]` (row `ivar`, col `jvar`, stride `n`)?
fn is_c_elem(
    e: &Expr,
    c_sym: Symbol,
    ivar: Symbol,
    jvar: Symbol,
    n: Dim,
    interner: &Interner,
) -> bool {
    index_of(e, c_sym)
        .and_then(|idx| match_row_col(idx, ivar, interner))
        .is_some_and(|(stride, col)| stride == n && col == jvar)
}

/// Match `C[i*N+j] + bias[j]` (either addend order) over the matmul output; return the bias array.
fn match_c_plus_bias(
    e: &Expr,
    c_sym: Symbol,
    ivar: Symbol,
    jvar: Symbol,
    n: Dim,
    interner: &Interner,
) -> Option<Symbol> {
    let ExprKind::Binary {
        op: ast::BinOp::Add,
        lhs,
        rhs,
    } = &e.kind
    else {
        return None;
    };
    // The non-C addend must be `bias[j]` — a single-index access by the column variable.
    let bias_of = |x: &Expr| -> Option<Symbol> {
        if let ExprKind::Index { base, indices } = &x.kind {
            if indices.len() == 1 && single_path(&indices[0]) == Some(jvar) {
                return single_path(base);
            }
        }
        None
    };
    if is_c_elem(lhs, c_sym, ivar, jvar, n, interner) {
        return bias_of(rhs);
    }
    if is_c_elem(rhs, c_sym, ivar, jvar, n, interner) {
        return bias_of(lhs);
    }
    None
}

/// Match the epilogue RHS over the matmul output `C[i*N+j]`: bare `C+bias` (identity), `fmax(_, 0)`
/// (ReLU), or a `gelu(_)` / `silu(_)` activation call (the transformer FFN `act(x·Wᵀ [+ bias])`
/// shape). Bias is **optional for the activation forms** — the bias-free `silu(x·Wᵀ)` is the
/// LLaMA/Mistral SwiGLU FFN — but required for identity (a bare `C = C` copy is a no-op, nothing to
/// fuse). Returns `(optional_bias_array, act_code)`.
fn match_epi_value(
    e: &Expr,
    c_sym: Symbol,
    ivar: Symbol,
    jvar: Symbol,
    n: Dim,
    interner: &Interner,
) -> Option<(Option<Symbol>, u32)> {
    // `C[i*N+j] + bias[j]` → `Some(bias)`; the bare output element `C[i*N+j]` → `None`; anything else
    // is not an epilogue over this matmul's output.
    let c_with_opt_bias = |x: &Expr| -> Option<Option<Symbol>> {
        if let Some(bias) = match_c_plus_bias(x, c_sym, ivar, jvar, n, interner) {
            Some(Some(bias))
        } else if is_c_elem(x, c_sym, ivar, jvar, n, interner) {
            Some(None)
        } else {
            None
        }
    };
    if let ExprKind::Call { callee, args, .. } = &e.kind {
        // ReLU written as `fmax(inner, 0.0)`.
        if args.len() == 2
            && single_path(callee).is_some_and(|s| interner.resolve(s) == "fmax")
            && is_float_zero(&args[1], interner)
        {
            return Some((c_with_opt_bias(&args[0])?, EPI_ACT_RELU));
        }
        // GELU / SiLU activation wrapping the (optional) bias-add (`gelu(C[i*N+j] + bias[j])` or the
        // bias-free `silu(C[i*N+j])`). Both are first-class intrinsics, so a single-arg call by that
        // name is unambiguous; the runtime epilogue applies the identical scalar form
        // (`mercury_runtime::vmath::{gelu1,silu1}`), so the fused result equals the unfused
        // `matmul → [bias →] activation` the recognizer replaces.
        if args.len() == 1 {
            let act = match single_path(callee).map(|s| interner.resolve(s)) {
                Some("gelu") => Some(EPI_ACT_GELU),
                Some("silu") => Some(EPI_ACT_SILU),
                _ => None,
            };
            if let Some(act) = act {
                return Some((c_with_opt_bias(&args[0])?, act));
            }
        }
    }
    // Identity: just the bias add (bias required — see above).
    let bias = match_c_plus_bias(e, c_sym, ivar, jvar, n, interner)?;
    Some((Some(bias), EPI_ACT_IDENTITY))
}

/// Match the bias/activation epilogue loop following a recognized `nn.Linear` matmul:
/// `for i in 0..M { for j in 0..N { C[i*N+j] = act(C[i*N+j] [+ bias[j]]) } }`. `M`/`N`/the stride/the
/// output array/the column index must all match the matmul's `(m, n, c)`, so it never misfires.
/// Returns `(optional_bias_array, act_code)` (bias optional for the activation forms — see
/// [`match_epi_value`]), else `None` (the loop is then lowered normally as a separate pass). Takes the
/// matmul shape as `(m, n, c)` rather than a `MatmulNest` so the bf16/f16 `LowpMatmulNest` reuses it.
fn match_bias_act_epilogue(
    stmt: &Stmt,
    m: Dim,
    n: Dim,
    c: Symbol,
    interner: &Interner,
) -> Option<(Option<Symbol>, u32)> {
    // for i in 0..M { <single nested loop> }
    let (ipat, iiter, ibody) = fusable_for(stmt)?;
    let ivar = match &ipat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (istart, iend) = range_bounds(iiter)?;
    if as_int_lit(istart, interner)? != 0 || as_dim(iend, interner)? != m {
        return None;
    }
    if ibody.tail.is_some() || ibody.stmts.len() != 1 {
        return None;
    }
    // for j in 0..N { <single assignment> }
    let (jpat, jiter, jbody) = fusable_for(&ibody.stmts[0])?;
    let jvar = match &jpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (jstart, jend) = range_bounds(jiter)?;
    if as_int_lit(jstart, interner)? != 0 || as_dim(jend, interner)? != n {
        return None;
    }
    if jbody.tail.is_some() || jbody.stmts.len() != 1 {
        return None;
    }
    // C[i*N+j] = <epilogue>   (plain assignment to the matmul's output element)
    let StmtKind::Assign {
        target,
        op: ast::AssignOp::Assign,
        value,
    } = &jbody.stmts[0].kind
    else {
        return None;
    };
    if !is_c_elem(target, c, ivar, jvar, n, interner) {
        return None;
    }
    match_epi_value(value, c, ivar, jvar, n, interner)
}

/// Flatten the multiplicative factors of `e`, recursing only through `*`. `(c as f32) * sa * sb[j]`
/// yields the three factors `[(c as f32), sa, sb[j]]` (left-association is irrelevant — the kernel is
/// the oracle, so any recognized association maps to the same fused call). The int8 dequant analog of
/// [`flatten_add_terms`].
fn flatten_mul_terms<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    if let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &e.kind
    {
        flatten_mul_terms(lhs, out);
        flatten_mul_terms(rhs, out);
    } else {
        out.push(e);
    }
}

/// If `e` is `base[var]` (single index exactly the path `var`), return `base`. The per-column access
/// shape of the weight scale `scale_b[j]` / `bias[j]` in the int8 dequant.
fn index_by_var(e: &Expr, var: Symbol) -> Option<Symbol> {
    if let ExprKind::Index { base, indices } = &e.kind {
        if indices.len() == 1 && single_path(&indices[0]) == Some(var) {
            return single_path(base);
        }
    }
    None
}

/// Peel an optional activation wrapper off the int8 dequant value: `fmax(inner, 0.0)` → ReLU,
/// `gelu(inner)` / `silu(inner)` → that activation, else the expression itself (identity). Mirrors
/// [`match_epi_value`]'s activation detection; the runtime `dequant_row` applies the identical scalar
/// form (`vmath::{gelu1,silu1}`), so fused == unfused.
fn peel_dequant_act<'a>(e: &'a Expr, interner: &Interner) -> (&'a Expr, u32) {
    if let ExprKind::Call { callee, args, .. } = &e.kind {
        if args.len() == 2
            && single_path(callee).is_some_and(|s| interner.resolve(s) == "fmax")
            && is_float_zero(&args[1], interner)
        {
            return (&args[0], EPI_ACT_RELU);
        }
        if args.len() == 1 {
            match single_path(callee).map(|s| interner.resolve(s)) {
                Some("gelu") => return (&args[0], EPI_ACT_GELU),
                Some("silu") => return (&args[0], EPI_ACT_SILU),
                _ => {}
            }
        }
    }
    (e, EPI_ACT_IDENTITY)
}

/// Peel an optional `+ bias[j]` (either addend order) off the int8 dequant value; return the
/// remaining product and the bias array. `None` when there is no per-column add (bias-free decode).
fn peel_bias_add<'a>(e: &'a Expr, jvar: Symbol) -> Option<(&'a Expr, Symbol)> {
    let ExprKind::Binary {
        op: ast::BinOp::Add,
        lhs,
        rhs,
    } = &e.kind
    else {
        return None;
    };
    if let Some(b) = index_by_var(rhs, jvar) {
        return Some((lhs, b));
    }
    if let Some(b) = index_by_var(lhs, jvar) {
        return Some((rhs, b));
    }
    None
}

/// Does any statement bind `sym` via a `let` at this block level? A `let`-local cannot escape the
/// block lexically, so — combined with "unmentioned after the window" (`block_mentions`) — it proves
/// `sym` is dead, which is what makes dropping the int8 accumulator's materialization sound.
fn block_declares_local(stmts: &[Stmt], sym: Symbol) -> bool {
    stmts.iter().any(|s| {
        matches!(&s.kind, StmtKind::Let { pat, .. }
            if matches!(&pat.kind, ast::PatKind::Ident(b) if *b == sym))
    })
}

/// Match the per-channel dequant epilogue that follows an int8 GEMM, decoding the i32 accumulator to
/// f32:
///
/// ```text
/// for i in 0..M { for j in 0..N {
///   out[i*N + j] = act((c[i*N + j] as f32) * scale_a * scale_b[j] [+ bias[j]]);
/// } }
/// ```
///
/// the standard quantized `nn.Linear` decode — a per-tensor activation scale `scale_a` (scalar), a
/// per-channel weight scale `scale_b[j]`, an optional `bias[j]`, and an optional activation. `M`/`N`/
/// the output stride / the i32 source array (`nest.c`) / the column index must all match `nest`, so it
/// never misfires. The three multiplicative factors are matched in **any** association (the fused
/// kernel is the differential oracle). `scale_a` may be **absent** (the scale folded into `scale_b` —
/// only a `cast * scale_b[j]` product), in which case the emitter passes `1.0`. Returns the f32 output
/// array, the optional scalar `scale_a` symbol, the per-channel `scale_b` array, the optional `bias`
/// array, and the activation code.
fn match_i8_dequant_epilogue(
    stmt: &Stmt,
    nest: &I8MatmulNest,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(Symbol, Option<Symbol>, Symbol, Option<Symbol>, u32)> {
    // for i in 0..M { <single nested loop> }
    let (ipat, iiter, ibody) = fusable_for(stmt)?;
    let ivar = match &ipat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (istart, iend) = range_bounds(iiter)?;
    if as_int_lit(istart, interner)? != 0 || as_dim(iend, interner)? != nest.m {
        return None;
    }
    if ibody.tail.is_some() || ibody.stmts.len() != 1 {
        return None;
    }
    // for j in 0..N { <single assignment> }
    let (jpat, jiter, jbody) = fusable_for(&ibody.stmts[0])?;
    let jvar = match &jpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (jstart, jend) = range_bounds(jiter)?;
    if as_int_lit(jstart, interner)? != 0 || as_dim(jend, interner)? != nest.n {
        return None;
    }
    if jbody.tail.is_some() || jbody.stmts.len() != 1 {
        return None;
    }
    // out[i*N + j] = <dequant value>   (a different buffer than the i32 accumulator `c`)
    let StmtKind::Assign {
        target,
        op: ast::AssignOp::Assign,
        value,
    } = &jbody.stmts[0].kind
    else {
        return None;
    };
    let (out_sym, oidx) = as_index1(target)?;
    let (ostride, ocol) = match_row_col(oidx, ivar, interner)?;
    if ostride != nest.n || ocol != jvar || out_sym == nest.c {
        return None;
    }
    if scalar_of(target, sema) != Some(mercury_types::Scalar::F32) {
        return None;
    }
    // Peel the optional activation, then the optional `+ bias[j]`.
    let (core, act) = peel_dequant_act(value, interner);
    let (prod, bias) = match peel_bias_add(core, jvar) {
        Some((p, b)) => (p, Some(b)),
        None => (core, None),
    };
    // The product must be `(c[i*N+j] as f32) * scale_a? * scale_b[j]` in any association: exactly one
    // cast of the i32 output element, exactly one per-column `scale_b[j]`, and at most one scalar
    // `scale_a`. Any other factor declines the fusion (the loop then lowers normally).
    let mut factors = Vec::new();
    flatten_mul_terms(prod, &mut factors);
    if factors.len() < 2 || factors.len() > 3 {
        return None;
    }
    let mut cast_seen = false;
    let mut scale_b: Option<Symbol> = None;
    let mut scale_a: Option<Symbol> = None;
    for f in factors {
        // `(c[i*N+j] as i32) as f32` — the dequant cast of the GEMM output element.
        if let ExprKind::Cast { expr, .. } = &f.kind {
            if is_c_elem(expr, nest.c, ivar, jvar, nest.n, interner)
                && scalar_of(f, sema) == Some(mercury_types::Scalar::F32)
            {
                if cast_seen {
                    return None;
                }
                cast_seen = true;
                continue;
            }
        }
        // `scale_b[j]` — a per-column array (the weight scale). Not the data or the output buffer.
        if let Some(b) = index_by_var(f, jvar) {
            if b == nest.c || b == out_sym || scale_b.is_some() {
                return None;
            }
            scale_b = Some(b);
            continue;
        }
        // `scale_a` — a scalar f32 (the per-tensor activation scale; the kernel takes it by value).
        if let Some(s) = single_path(f) {
            if scale_a.is_some() || scalar_of(f, sema) != Some(mercury_types::Scalar::F32) {
                return None;
            }
            scale_a = Some(s);
            continue;
        }
        return None;
    }
    if !cast_seen {
        return None;
    }
    let scale_b = scale_b?;
    Some((out_sym, scale_a, scale_b, bias, act))
}

/// A recognized int8 quantized `nn.Linear` nest: `C[m,n] (i32) = A[m,k] (u8) · B[n,k] (i8)ᵀ`.
struct I8MatmulNest {
    a: Symbol,
    b: Symbol,
    c: Symbol,
    m: Dim,
    k: Dim,
    n: Dim,
}

/// Peel an `as`-cast wrapper (`x as T` → `x`); the expression itself otherwise. int8 GEMM source
/// casts each `u8`/`i8` element to `i32` before multiplying (the product can't fit `i8`).
fn peel_cast(e: &Expr) -> &Expr {
    match &e.kind {
        ExprKind::Cast { expr, .. } => expr,
        _ => e,
    }
}

/// The scalar type sema assigned to `e`, if any.
fn scalar_of(e: &Expr, sema: &SemaResult) -> Option<mercury_types::Scalar> {
    match sema.types.get(&e.id) {
        Some(Ty::Scalar(s)) => Some(*s),
        _ => None,
    }
}

/// A literal integer `0` (the int8 accumulator seed `let mut s: i32 = 0`).
fn is_int_zero(e: &Expr, interner: &Interner) -> bool {
    matches!(&e.kind, ExprKind::Int(t) if parse_int(interner.resolve(*t)) == 0)
}

/// Recognize the int8 quantized `nn.Linear` nest `C = A·Bᵀ` — the dot-product `ijk` form with `u8`
/// activations, `i8` weights, and an `i32` accumulator:
///
/// ```text
/// for i in 0..M { for j in 0..N {
///   let mut s: i32 = 0;
///   for k in 0..K { s = s + (a[i*K + k] as i32) * (b[j*K + k] as i32); }
///   c[i*N + j] = s;
/// } }
/// ```
///
/// Returns the nest iff A is `u8`, B is `i8`, C is `i32`, B is transposed (`b[j*K+k]` — the weight
/// layout `mercury_i8gemm_nt` expects), the strides are consistent (`sa = sb = K`, `sc = N`), and
/// there are no batch offsets (plain 2-D). The signedness is enforced because the kernel zero-extends
/// A and sign-extends B; matching the wrong signedness would miscompile, so anything else falls back
/// to the scalar nest. Integer arithmetic means the kernel equals the scalar nest bit-for-bit.
fn match_matmul_i8_nt(
    pat: &Pattern,
    iter: &ForIter,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<I8MatmulNest> {
    let row = match &pat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (start, end) = range_bounds(iter)?;
    if as_int_lit(start, interner)? != 0 {
        return None;
    }
    let m = as_dim(end, interner)?;
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let (jpat, jiter, jbody) = fusable_for(&body.stmts[0])?;
    let jvar = match &jpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (js, je) = range_bounds(jiter)?;
    if as_int_lit(js, interner)? != 0 {
        return None;
    }
    let n = as_dim(je, interner)?;
    if jbody.tail.is_some() || jbody.stmts.len() != 3 {
        return None;
    }
    // [0] let mut s: i32 = 0;
    let StmtKind::Let {
        pat: sp,
        init: Some(s0),
        ..
    } = &jbody.stmts[0].kind
    else {
        return None;
    };
    let s_sym = match &sp.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    if !is_int_zero(s0, interner) {
        return None;
    }
    // [1] for k in 0..K { s = s + (a[..] as i32) * (b[..] as i32); }
    let (kpat, kiter, kbody) = fusable_for(&jbody.stmts[1])?;
    let kvar = match &kpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (ks, ke) = range_bounds(kiter)?;
    if as_int_lit(ks, interner)? != 0 {
        return None;
    }
    let kdim = as_dim(ke, interner)?;
    if kbody.tail.is_some() || kbody.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &kbody.stmts[0].kind else {
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
    // The product must be i32 (the casts widen to i32; the accumulation matches the kernel's i32).
    if !matches!(sema.types.get(&prod.id), Some(t) if mir_ty(t) == MirType::I32) {
        return None;
    }
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: f1,
        rhs: f2,
    } = &prod.kind
    else {
        return None;
    };
    // Each factor is `(arr[idx] as i32)`. Peel the cast and identify A (row i) and B (row j, the
    // transposed weight layout), enforcing A:u8 × B:i8 and no batch offset, in either factor order.
    let (a_sym, sa, b_sym, sb) = {
        let mut found = None;
        for (fa, fb) in [(f1, f2), (f2, f1)] {
            let (ai, bi) = (peel_cast(fa), peel_cast(fb));
            let Some((a_sym, sa, a_off, a_trans)) = match_a_factor(ai, row, kvar, sema, interner)
            else {
                continue;
            };
            let Some((b_sym, sb, b_off, transposed)) =
                match_b_factor(bi, kvar, jvar, sema, interner)
            else {
                continue;
            };
            // int8 has only the `C = A·Bᵀ` (NT) kernel — a transposed A (`A[k*M+i]`) has no int8
            // variant, so decline it to the scalar nest.
            if !transposed || a_trans || !a_off.is_empty() || !b_off.is_empty() {
                continue;
            }
            if scalar_of(ai, sema) != Some(mercury_types::Scalar::U8)
                || scalar_of(bi, sema) != Some(mercury_types::Scalar::I8)
            {
                continue;
            }
            found = Some((a_sym, sa, b_sym, sb));
            break;
        }
        found?
    };
    // [2] c[i*N + j] = s;  (C must be i32, plain 2-D, strides consistent.)
    let StmtKind::Assign {
        target: ct,
        op: ast::AssignOp::Assign,
        value: cv,
    } = &jbody.stmts[2].kind
    else {
        return None;
    };
    if single_path(cv) != Some(s_sym) {
        return None;
    }
    let (cbase, cidx) = as_index1(ct)?;
    let (sc, c_off) = match_row_col_off(cidx, row, jvar, interner)?;
    if !c_off.is_empty() || sa != kdim || sb != kdim || sc != n {
        return None;
    }
    if scalar_of(ct, sema) != Some(mercury_types::Scalar::I32) {
        return None;
    }
    // An input aliasing the output is a hazard (the kernel writes C in a different order). A == B is
    // fine (both read-only).
    if a_sym == cbase || b_sym == cbase {
        return None;
    }
    Some(I8MatmulNest {
        a: a_sym,
        b: b_sym,
        c: cbase,
        m,
        k: kdim,
        n,
    })
}

/// A recognized bf16/f16 mixed-precision `nn.Linear` nest: `C[m,n] (f32) = A[m,k] · B[n,k]ᵀ`, where A
/// and B are `[bf16]` or `[f16]` (the same precision), each element widened `as f32`, accumulated in an
/// f32 `s`. Same shape as [`I8MatmulNest`]; `f16` selects the storage format (bf16 widens via `<<16`,
/// f16 via F16C).
struct LowpMatmulNest {
    a: Symbol,
    b: Symbol,
    c: Symbol,
    m: Dim,
    k: Dim,
    n: Dim,
    f16: bool,
    /// `false` → `C = A·Bᵀ` (NT, the `nn.Linear` weight layout — A `[m,k]`, B `[n,k]`); `true` →
    /// `C = Aᵀ·B` (TN, the `dW = dYᵀ·X` weight-gradient — A stored `[k,m]`, B `[k,n]`). Exactly one
    /// operand is transposed (the recognizer rejects plain `A·B` and `Aᵀ·Bᵀ`, which have no half kernel).
    transposed_a: bool,
}

/// Recognize the bf16/f16 mixed-precision matmul nest — the f32 `ijk` dot-product matmul but over
/// `[bf16]`/`[f16]` operands widened to f32, with an f32 accumulator (the standard mixed-precision
/// transformer matmul). Two shapes, distinguished by which operand is transposed:
///
/// ```text
/// // NT (transposed_a = false): C = A·Bᵀ, the nn.Linear forward (A [m,k], B [n,k])
/// for i in 0..M { for j in 0..N {
///   let mut s: f32 = 0.0;
///   for k in 0..K { s = s + (a[i*K + k] as f32) * (b[j*K + k] as f32); }
///   c[i*N + j] = s;
/// } }
/// // TN (transposed_a = true): C = Aᵀ·B, the dW = dYᵀ·X weight gradient (A [k,m], B [k,n])
/// for i in 0..M { for j in 0..N {
///   let mut s: f32 = 0.0;
///   for p in 0..K { s = s + (a[p*M + i] as f32) * (b[p*N + j] as f32); }
///   c[i*N + j] = s;
/// } }
/// ```
///
/// Returns the nest iff A and B are the **same** low precision (both bf16 or both f16), the casts
/// target f32, the product / accumulator / `c` are f32, **exactly one** operand is transposed (NT: B
/// is `b[j*K+k]`; TN: A is `a[k*M+i]`), the strides are consistent (NT: `sa = sb = K`; TN: `sa = M`,
/// `sb = N`; both `sc = N`), and there are no batch offsets. Both shapes feed a lossless widen prepass
/// then the *identical* proven f32 kernel (NT → `gemm_dispatch`, TN → `mercury_sgemm_tn`), so the
/// result is bit-for-bit the f32 GEMM on the widened values across backends. The naive nest otherwise
/// falls to a scalar widening loop the autovectorizer can't reach (and TN additionally has the
/// column-strided A reads that defeat C/Rust). The int8 twin is [`match_matmul_i8_nt`]; this is its
/// float-accumulator sibling. Plain `A·B` and `Aᵀ·Bᵀ` are rejected (no half kernel — the scalar nest).
fn match_matmul_lowp(
    pat: &Pattern,
    iter: &ForIter,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<LowpMatmulNest> {
    let row = match &pat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (start, end) = range_bounds(iter)?;
    if as_int_lit(start, interner)? != 0 {
        return None;
    }
    let m = as_dim(end, interner)?;
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let (jpat, jiter, jbody) = fusable_for(&body.stmts[0])?;
    let jvar = match &jpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (js, je) = range_bounds(jiter)?;
    if as_int_lit(js, interner)? != 0 {
        return None;
    }
    let n = as_dim(je, interner)?;
    if jbody.tail.is_some() || jbody.stmts.len() != 3 {
        return None;
    }
    // [0] let mut s: f32 = 0.0;
    let StmtKind::Let {
        pat: sp,
        init: Some(s0),
        ..
    } = &jbody.stmts[0].kind
    else {
        return None;
    };
    let s_sym = match &sp.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    if !is_float_zero(s0, interner) {
        return None;
    }
    // [1] for k in 0..K { s = s + (a[..] as f32) * (b[..] as f32); }
    let (kpat, kiter, kbody) = fusable_for(&jbody.stmts[1])?;
    let kvar = match &kpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (ks, ke) = range_bounds(kiter)?;
    if as_int_lit(ks, interner)? != 0 {
        return None;
    }
    let kdim = as_dim(ke, interner)?;
    if kbody.tail.is_some() || kbody.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &kbody.stmts[0].kind else {
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
    // The product (hence the accumulator) must be f32 — the mixed-precision contract.
    if !matches!(sema.types.get(&prod.id), Some(t) if mir_ty(t) == MirType::F32) {
        return None;
    }
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: f1,
        rhs: f2,
    } = &prod.kind
    else {
        return None;
    };
    // Each factor is `(arr[idx] as f32)` over a `[bf16]`/`[f16]` array. Peel the cast (must target
    // f32), classify A and B (each may be normal or transposed), require A and B the SAME precision,
    // accept **exactly one** transposed (NT `A·Bᵀ` or TN `Aᵀ·B`) and reject batch offsets, either order.
    let (a_sym, sa, b_sym, sb, f16, transposed_a) = {
        let mut found = None;
        for (fa, fb) in [(f1, f2), (f2, f1)] {
            if scalar_of(fa, sema) != Some(mercury_types::Scalar::F32)
                || scalar_of(fb, sema) != Some(mercury_types::Scalar::F32)
            {
                continue;
            }
            let (ai, bi) = (peel_cast(fa), peel_cast(fb));
            let Some((a_sym, sa, a_off, a_trans)) = match_a_factor(ai, row, kvar, sema, interner)
            else {
                continue;
            };
            let Some((b_sym, sb, b_off, transposed)) =
                match_b_factor(bi, kvar, jvar, sema, interner)
            else {
                continue;
            };
            // Exactly one operand transposed: NT (A normal, B transposed) or TN (A transposed, B
            // normal). Plain `A·B` and `Aᵀ·Bᵀ` have no half kernel — decline to the scalar nest.
            if a_trans == transposed || !a_off.is_empty() || !b_off.is_empty() {
                continue;
            }
            let af = match scalar_of(ai, sema) {
                Some(mercury_types::Scalar::Bf16) => false,
                Some(mercury_types::Scalar::F16) => true,
                _ => continue,
            };
            let bf = match scalar_of(bi, sema) {
                Some(mercury_types::Scalar::Bf16) => false,
                Some(mercury_types::Scalar::F16) => true,
                _ => continue,
            };
            if af != bf {
                continue;
            }
            found = Some((a_sym, sa, b_sym, sb, af, a_trans));
            break;
        }
        found?
    };
    // [2] c[i*N + j] = s;  (C must be f32, plain 2-D, strides consistent.)
    let StmtKind::Assign {
        target: ct,
        op: ast::AssignOp::Assign,
        value: cv,
    } = &jbody.stmts[2].kind
    else {
        return None;
    };
    if single_path(cv) != Some(s_sym) {
        return None;
    }
    let (cbase, cidx) = as_index1(ct)?;
    let (sc, c_off) = match_row_col_off(cidx, row, jvar, interner)?;
    // Normal A's contraction stride is K (`a[i*K+k]`), transposed A's is the output-row count M
    // (`a[k*M+i]`); normal B's stride is N (`b[k*N+j]`), transposed B's is K (`b[j*K+k]`). Since
    // exactly one is transposed, `transposed_a` picks both: TN → (sa=M, sb=N), NT → (sa=K, sb=K).
    let sa_ok = if transposed_a { sa == m } else { sa == kdim };
    let sb_ok = if transposed_a { sb == n } else { sb == kdim };
    if !c_off.is_empty() || !sa_ok || !sb_ok || sc != n {
        return None;
    }
    if scalar_of(ct, sema) != Some(mercury_types::Scalar::F32) {
        return None;
    }
    // An input aliasing the output is a hazard (the kernel writes C in a different order). A == B is
    // fine (both read-only).
    if a_sym == cbase || b_sym == cbase {
        return None;
    }
    Some(LowpMatmulNest {
        a: a_sym,
        b: b_sym,
        c: cbase,
        m,
        k: kdim,
        n,
        f16,
        transposed_a,
    })
}

/// Recognize the textbook `ijk` dot-product matmul:
/// `for i { for j { let s = 0.0; for k { s = s + A[i,k]*B[..]; } c[i*N+j] = s; } }`. This is the
/// natural way to write `C = A·Bᵀ` (both A and B read contiguously). Always `beta = 0` (s overwrites
/// c). See [`MatmulNest`].
fn match_matmul_ijk<'a>(
    pat: &Pattern,
    iter: &ForIter,
    body: &'a Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<MatmulNest<'a>> {
    let row = match &pat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (start, end) = range_bounds(iter)?;
    if as_int_lit(start, interner)? != 0 {
        return None;
    }
    let m = as_dim(end, interner)?;
    // Outer body is a single `for j` loop.
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let (jpat, jiter, jbody) = fusable_for(&body.stmts[0])?;
    let jvar = match &jpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (js, je) = range_bounds(jiter)?;
    if as_int_lit(js, interner)? != 0 {
        return None;
    }
    let n = as_dim(je, interner)?;
    // j body: [ let s = 0.0; for k {...}; c[i*N+j] = s ].
    if jbody.tail.is_some() || jbody.stmts.len() != 3 {
        return None;
    }
    let StmtKind::Let {
        pat: sp,
        init: Some(s0),
        ..
    } = &jbody.stmts[0].kind
    else {
        return None;
    };
    let s_sym = match &sp.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    if !is_float_zero(s0, interner) {
        return None;
    }
    // The K loop: `for k in 0..K { s = s + A*B; }`.
    let (kpat, kiter, kbody) = fusable_for(&jbody.stmts[1])?;
    let kvar = match &kpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (ks, ke) = range_bounds(kiter)?;
    if as_int_lit(ks, interner)? != 0 {
        return None;
    }
    let kdim = as_dim(ke, interner)?;
    if kbody.tail.is_some() || kbody.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &kbody.stmts[0].kind else {
        return None;
    };
    if single_path(target) != Some(s_sym) {
        return None;
    }
    let prod = match op {
        ast::AssignOp::Add => value,
        ast::AssignOp::Assign => {
            // s = s + A*B
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
    // The inline `ijk` form tolerates a per-operand base offset (a batch/head index): A, B and C may
    // each be indexed `… + h*S*D`, the hallmark of a batched matmul (multi-head attention is one
    // matmul per head). The offsets are peeled off here and applied as pointer GEPs in `emit_sgemm`.
    let (a_sym, sa, a_off, b_sym, sb, b_off, transposed, transposed_a) =
        match_product_ab_off(prod, row, kvar, jvar, sema, interner)?;
    // Final store: c[i*N + j (+ off)] = s.
    let StmtKind::Assign {
        target: ct,
        op: ast::AssignOp::Assign,
        value: cv,
    } = &jbody.stmts[2].kind
    else {
        return None;
    };
    if single_path(cv) != Some(s_sym) {
        return None;
    }
    // The output store `c[i*N + j (+ off)] = s` or the shape-typed `c[i, j] = s`.
    let (cbase, sc, c_off) = match_operand_row_col_off(ct, row, jvar, sema, interner)?;
    // Normal A's contraction stride is K (`A[i*K+k]`); transposed A's is the output-row count M
    // (`A[k*M+i]`). Normal B's is N; transposed B's is K.
    let sa_ok = if transposed_a { sa == m } else { sa == kdim };
    let sb_ok = if transposed { sb == kdim } else { sb == n };
    if !sa_ok || !sb_ok || sc != n {
        return None;
    }
    // Every base offset must be invariant in the matmul's own `(i,j,k)` — otherwise it is not a
    // constant per-call pointer shift and the nest is not a batched matmul. (A plain 2-D matmul has
    // empty offsets and trivially passes.)
    let bound = [row, jvar, kvar];
    if !offset_invariant(&a_off, &bound)
        || !offset_invariant(&b_off, &bound)
        || !offset_invariant(&c_off, &bound)
    {
        return None;
    }
    // A and B may be the *same* array (a Gram matrix `A·Aᵀ`, or self-attention `Q·Kᵀ` with a shared
    // operand): both sides are read-only, which the kernel packs into separate scratch panels, so it
    // is safe. Only an input aliasing the output C is a hazard (the blocked kernel writes C in a
    // different order than the scalar nest reads it).
    if a_sym == cbase || b_sym == cbase {
        return None;
    }
    Some(MatmulNest {
        a: a_sym,
        b: b_sym,
        c: cbase,
        m,
        k: kdim,
        n,
        beta: 0,
        transposed,
        transposed_a,
        a_off,
        b_off,
        c_off,
    })
}

/// Match the **residual** store value of a fused residual projection:
/// `act(c[i*N+j] + s [+ bias[j]])` — the matmul output `c` read back and added to the fresh dot `s`,
/// with an optional per-column bias and an optional activation wrapping the whole sum (the transformer
/// skip connection). Flattens the additive terms in any association (so `c + s + bias`, `s + c`, … all
/// match) and requires **exactly** one residual `c[i*N+j]` and one dot `s`, plus at most one `bias[j]`;
/// any other term rejects. Returns `(optional_bias, act_code)`. The activation peeling and codes mirror
/// the plain epilogue (`peel_dequant_act`), so the fused `nt_epi` (beta = 1) result is bit-identical to
/// the unfused `c = act(c + matmul + bias)`.
fn match_residual_store_value(
    value: &Expr,
    c_sym: Symbol,
    ivar: Symbol,
    jvar: Symbol,
    n: Dim,
    s_sym: Symbol,
    interner: &Interner,
) -> Option<(Option<Symbol>, u32)> {
    let (inner, act) = peel_dequant_act(value, interner);
    let mut terms = Vec::new();
    flatten_add_terms(inner, &mut terms);
    let (mut saw_c, mut saw_s) = (false, false);
    let mut bias: Option<Symbol> = None;
    for t in terms {
        if is_c_elem(t, c_sym, ivar, jvar, n, interner) {
            if saw_c {
                return None; // the residual must appear exactly once
            }
            saw_c = true;
        } else if single_path(t) == Some(s_sym) {
            if saw_s {
                return None;
            }
            saw_s = true;
        } else if let Some(b) = index_by_var(t, jvar) {
            if bias.is_some() {
                return None; // at most one per-column (bias) term
            }
            bias = Some(b);
        } else {
            return None; // an unrecognized additive term — not a residual projection
        }
    }
    if saw_c && saw_s {
        Some((bias, act))
    } else {
        None
    }
}

/// Recognize the **fused residual projection** — the textbook `ijk` `nn.Linear` (`A·Bᵀ`) nest whose
/// store *accumulates* into its own output (the transformer skip connection `x = x + act(x·Wᵀ + bias)`):
///
/// ```text
/// for i in 0..M { for j in 0..N {
///   let mut s: f32 = 0.0;
///   for k in 0..K { s = s + a[i*K+k] * b[j*K+k]; }
///   c[i*N+j] = act(c[i*N+j] + s [+ bias[j]]);   // residual c + dot, optional bias, optional act
/// } }
/// ```
///
/// Identical to [`match_matmul_ijk`] except the store reads `c` back (`match_residual_store_value`), so
/// it maps to `mercury_sgemm_nt_epi` with **beta = 1**: the kernel computes `act(beta·c_old + A·Bᵀ +
/// bias)` = `act(c_residual + A·Bᵀ + bias)`, reusing the exact fused-epilogue kernel — *no new symbol,
/// no backend change* (the interpreter already marshals `nt_epi` reading the old `c`, and beta flows
/// through). Without this the accumulate store (not `c = s`) blocks the matmul recognizer and the whole
/// nest falls to a scalar loop. **NT only, offset-free** (the epilogue kernel is the plain 2-D `A·Bᵀ`);
/// returns the nest (`beta = 1`) plus the optional bias and the activation code.
fn match_matmul_residual<'a>(
    pat: &Pattern,
    iter: &ForIter,
    body: &'a Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(MatmulNest<'a>, Option<Symbol>, u32)> {
    let row = match &pat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (start, end) = range_bounds(iter)?;
    if as_int_lit(start, interner)? != 0 {
        return None;
    }
    let m = as_dim(end, interner)?;
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let (jpat, jiter, jbody) = fusable_for(&body.stmts[0])?;
    let jvar = match &jpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (js, je) = range_bounds(jiter)?;
    if as_int_lit(js, interner)? != 0 {
        return None;
    }
    let n = as_dim(je, interner)?;
    // j body: [ let s = 0.0; for k {...}; c[i*N+j] = act(c[i*N+j] + s [+ bias[j]]) ].
    if jbody.tail.is_some() || jbody.stmts.len() != 3 {
        return None;
    }
    let StmtKind::Let {
        pat: sp,
        init: Some(s0),
        ..
    } = &jbody.stmts[0].kind
    else {
        return None;
    };
    let s_sym = match &sp.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    if !is_float_zero(s0, interner) {
        return None;
    }
    let (kpat, kiter, kbody) = fusable_for(&jbody.stmts[1])?;
    let kvar = match &kpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (ks, ke) = range_bounds(kiter)?;
    if as_int_lit(ks, interner)? != 0 {
        return None;
    }
    let kdim = as_dim(ke, interner)?;
    if kbody.tail.is_some() || kbody.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &kbody.stmts[0].kind else {
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
    let (a_sym, sa, a_off, b_sym, sb, b_off, transposed, transposed_a) =
        match_product_ab_off(prod, row, kvar, jvar, sema, interner)?;
    // The fused-epilogue kernel is the plain 2-D `A·Bᵀ`: require transposed B, normal A, no batch
    // offsets (a TN / batched residual has no epilogue kernel — fall back to the scalar nest).
    if !transposed || transposed_a || !a_off.is_empty() || !b_off.is_empty() {
        return None;
    }
    // The residual store: c[i*N+j] = act(c[i*N+j] + s [+ bias[j]]).
    let StmtKind::Assign {
        target: ct,
        op: ast::AssignOp::Assign,
        value: cv,
    } = &jbody.stmts[2].kind
    else {
        return None;
    };
    let (cbase, cidx) = as_index1(ct)?;
    let (sc, c_off) = match_row_col_off(cidx, row, jvar, interner)?;
    if sa != kdim || sb != kdim || sc != n || !c_off.is_empty() {
        return None;
    }
    let (bias, act) = match_residual_store_value(cv, cbase, row, jvar, n, s_sym, interner)?;
    // An input aliasing the output is a hazard (the blocked kernel writes C in a different order).
    if a_sym == cbase || b_sym == cbase {
        return None;
    }
    let nest = MatmulNest {
        a: a_sym,
        b: b_sym,
        c: cbase,
        m,
        k: kdim,
        n,
        beta: 1, // accumulate the matmul into the residual already in C
        transposed,
        transposed_a,
        a_off,
        b_off,
        c_off,
    };
    Some((nest, bias, act))
}

/// Recognize the canonical f32 matmul nest rooted at `for row in 0..M { … }`. See [`MatmulNest`].
fn match_matmul<'a>(
    pat: &Pattern,
    iter: &ForIter,
    body: &'a Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<MatmulNest<'a>> {
    let row = match &pat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (start, end) = range_bounds(iter)?;
    if as_int_lit(start, interner)? != 0 {
        return None;
    }
    let m = as_dim(end, interner)?;
    if body.tail.is_some() {
        return None;
    }

    // Body is either [k-loop] (accumulate) or [zero-init, k-loop] (overwrite).
    let (beta, czero, k_stmt) = match body.stmts.as_slice() {
        [k] => (1i64, None, k),
        [z, k] => (0i64, Some(z), k),
        _ => return None,
    };

    // The K loop: `for k in 0..K { [let aik = a[row*sa + k];] for j in 0..N { … } }`.
    let (kpat, kiter, kbody) = fusable_for(k_stmt)?;
    let kvar = match &kpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (kstart, kend) = range_bounds(kiter)?;
    if as_int_lit(kstart, interner)? != 0 {
        return None;
    }
    let kdim = as_dim(kend, interner)?;

    // Optional `let aik = a[row*sa + k];` binding, then the inner J loop.
    let (aik, a_sym, sa, j_stmt) = match kbody.stmts.as_slice() {
        [j] if kbody.tail.is_none() => (None, None, None, j),
        [let_s, j] if kbody.tail.is_none() => {
            let StmtKind::Let {
                pat: lp,
                init: Some(init),
                ..
            } = &let_s.kind
            else {
                return None;
            };
            let aik_sym = match &lp.kind {
                ast::PatKind::Ident(s) => *s,
                _ => return None,
            };
            let (abase, asa, ak) = match_operand_row_then_col(init, row, sema, interner)?;
            if ak != kvar {
                return None;
            }
            (Some(aik_sym), Some(abase), Some(asa), j)
        }
        _ => return None,
    };

    // Inner J loop: `for j in 0..N { c[row*sc + j] (=|+=) … a … * … b[k*sb + j] … }`.
    let (jpat, jiter, jbody) = fusable_for(j_stmt)?;
    let jvar = match &jpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (jstart, jend) = range_bounds(jiter)?;
    if as_int_lit(jstart, interner)? != 0 {
        return None;
    }
    let n = as_dim(jend, interner)?;
    if jbody.stmts.len() != 1 || jbody.tail.is_some() {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &jbody.stmts[0].kind else {
        return None;
    };
    let (cbase, sc, cj) = match_operand_row_then_col(target, row, sema, interner)?;
    if cj != jvar {
        return None;
    }

    // The product `a_term * b_term`, possibly read-add'd into c.
    let prod = match op {
        ast::AssignOp::Add => value,
        ast::AssignOp::Assign => {
            // value must be `c[row*sc + j] + (a*b)`.
            let ExprKind::Binary {
                op: ast::BinOp::Add,
                lhs,
                rhs,
            } = &value.kind
            else {
                return None;
            };
            let (clhs, clsc, clj) = match_operand_row_then_col(lhs, row, sema, interner)?;
            if clhs != cbase || clsc != sc || clj != jvar {
                return None;
            }
            rhs
        }
        _ => return None,
    };
    if !is_f32_expr(prod, sema) {
        return None;
    }
    let aik_info = match (aik, a_sym, sa) {
        (Some(s), Some(b), Some(st)) => Some((s, b, st)),
        _ => None,
    };
    let (a_sym, sa, b_sym, sb, transposed) =
        match_product_ab(prod, row, kvar, jvar, aik_info, sema, interner)?;

    // Strides must describe contiguous row-major A[m,k] and C[m,n], and B[k,n] (normal) or B[n,k]
    // (transposed) — i.e. B's contraction stride is N normally, K when transposed.
    let sb_ok = if transposed { sb == kdim } else { sb == n };
    if sa != kdim || !sb_ok || sc != n {
        return None;
    }
    if beta == 0 {
        let (cz, scz, nz) = match_zero_init(czero?, row, sema, interner)?;
        if cz != cbase || scz != n || nz != n {
            return None;
        }
    }
    // A and B may be the *same* array (a Gram matrix `A·Aᵀ`, or self-attention `Q·Kᵀ` with a shared
    // operand): both sides are read-only, which the kernel packs into separate scratch panels, so it
    // is safe. Only an input aliasing the output C is a hazard (the blocked kernel writes C in a
    // different order than the scalar nest reads it).
    if a_sym == cbase || b_sym == cbase {
        return None;
    }
    Some(MatmulNest {
        a: a_sym,
        b: b_sym,
        c: cbase,
        m,
        k: kdim,
        n,
        beta,
        transposed,
        // The `ikj` accumulate form binds `let aik = A[i*K+k]` with the 2-term `match_row_col`, which
        // matches only the normal A layout — a transposed `A[k*M+i]` falls through to the scalar nest.
        transposed_a: false,
        // The `ikj` accumulate form parses its A/C indices with the 2-term `match_row_col`, so a
        // batched (offset) index falls back to the scalar nest; offsets are always empty here.
        a_off: Vec::new(),
        b_off: Vec::new(),
        c_off: Vec::new(),
    })
}

/// Recognize a function whose entire body is a matmul nest (`{ for i in 0..M { … } }`).
fn matmul_fn<'a>(
    body: &'a Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<MatmulNest<'a>> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    recognize_matmul(pat, iter, lb, sema, interner)
}

/// Lower a recognized matmul function to a thin wrapper that binds its array params to base
/// pointers and tail-calls `mercury_sgemm`/`mercury_sgemm_parallel`.
#[allow(clippy::too_many_arguments)]
fn lower_matmul_fn(
    f: &FnDecl,
    nest: &MatmulNest<'_>,
    parallel: bool,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
    diags: &mut Vec<Diagnostic>,
) -> Function {
    let (param_tys, ret_ty) = match sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
        Some(DefKind::Fn(sig)) => (sig.params.clone(), sig.ret.clone()),
        _ => (f.params.iter().map(|_| Ty::Unknown).collect(), Ty::Unit),
    };
    let ret_mir = mir_ty(&ret_ty);
    let mut fl = FnLowerer {
        builder: Builder::new(f.name.sym, ret_mir.clone()),
        sema,
        interner,
        diags,
        scopes: vec![HashMap::new()],
        terminated: false,
        loops: Vec::new(),
        gemm,
        parallel_fn: false,
        vec_loads: HashMap::new(),
        sret: None,
    };
    let param_vals: Vec<ValueId> = param_tys
        .iter()
        .map(|pty| fl.builder.add_param(param_abi_ty(pty)))
        .collect();
    for ((p, pty), val) in f.params.iter().zip(&param_tys).zip(param_vals) {
        let mty = mir_ty(pty);
        if matches!(mty, MirType::Array(..)) {
            fl.bind(p.name.sym, val, mty);
        } else {
            let slot = fl.builder.alloca(mty.clone());
            fl.builder.build_void(Op::Store {
                ptr: slot,
                value: val,
            });
            fl.bind(p.name.sym, slot, mty);
        }
    }
    fl.emit_sgemm(nest, parallel);
    if !fl.terminated {
        match ret_mir {
            MirType::Void => fl.builder.ret(None),
            _ => {
                let z = fl.const_zero(ret_mir.clone());
                fl.builder.ret(Some(z));
            }
        }
    }
    fl.builder.finish()
}

/// Recognize a function whose entire body is an int8 matmul nest (`{ for i in 0..M { … } }`) — the
/// int8 twin of `matmul_fn`, used for whole-function quantized `nn.Linear` kernels.
fn i8matmul_fn(body: &Block, sema: &SemaResult, interner: &Interner) -> Option<I8MatmulNest> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    match_matmul_i8_nt(pat, iter, lb, sema, interner)
}

/// Is the whole function body a single bf16/f16 mixed-precision matmul nest (`C = A·Bᵀ` NT or
/// `C = Aᵀ·B` TN)? Used to intercept a `@parallel` half-precision matmul *before* the elementwise
/// outliner (which would outline the outer row loop into per-row scalar loops and lose the kernel
/// dispatch). Detection only — the function is then lowered normally (`lower_fn`, `parallel = true`)
/// and the embedded recognizer in `lower_for` emits the multicore half GEMM. A non-`@parallel`
/// whole-function matmul reaches the serial kernel the same way via the ordinary `lower_fn` path.
fn lowp_matmul_fn(body: &Block, sema: &SemaResult, interner: &Interner) -> Option<LowpMatmulNest> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    match_matmul_lowp(pat, iter, lb, sema, interner)
}

/// A recognized matrix transpose `dst = srcᵀ` (`src` is `[rows, cols]`, `dst` is `[cols, rows]`).
struct TransposeNest {
    src: Symbol,
    dst: Symbol,
    rows: Dim,
    cols: Dim,
    /// `true` for a 16-bit (`bf16`/`f16`) transpose (→ `mercury_transpose_u16`), `false` for f32.
    elem_u16: bool,
}

/// Recognize the matrix-transpose nest and dispatch it to the cache-blocked `mercury_transpose_f32`:
///
/// ```text
/// for i in 0..R { for j in 0..C { dst[j*R + i] = src[i*C + j]; } }
/// ```
///
/// `dst` (`[C, R]`) is written as the transpose of `src` (`[R, C]`). Both must be f32 arrays and
/// **distinct** (an in-place transpose aliases — a different computation the kernel does not do). The
/// store index is `j*R + i` (column-major in `src`'s frame), the load `i*C + j` (row-major); the
/// strides pin `R`/`C` to the loop bounds, so it never misfires. Pure data movement (a permutation),
/// so the kernel is bit-identical to this nest on both backends — no reassociation, the differential
/// gate is trivial. Naive C/Rust write `dst` with stride `R` (a cache miss per element for large `R`);
/// the blocked kernel keeps a tile L1-resident, which `-O3` does not do for a transpose.
fn match_transpose(
    pat: &Pattern,
    iter: &ForIter,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<TransposeNest> {
    // for i in 0..R { <single inner for> }
    let row = match &pat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (rs, re) = range_bounds(iter)?;
    if as_int_lit(rs, interner)? != 0 {
        return None;
    }
    let rows = as_dim(re, interner)?;
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    // for j in 0..C { <single assignment> }
    let (jpat, jiter, jbody) = fusable_for(&body.stmts[0])?;
    let col = match &jpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (cs, ce) = range_bounds(jiter)?;
    if as_int_lit(cs, interner)? != 0 {
        return None;
    }
    let cols = as_dim(ce, interner)?;
    if jbody.tail.is_some() || jbody.stmts.len() != 1 {
        return None;
    }
    // dst[j*R + i] = src[i*C + j];
    let StmtKind::Assign {
        target,
        op: ast::AssignOp::Assign,
        value,
    } = &jbody.stmts[0].kind
    else {
        return None;
    };
    let (dbase, didx) = as_index1(target)?;
    let (sbase, sidx) = as_index1(value)?;
    // dst index `j*R + i` (col outer, stride R) and src index `i*C + j` (row outer, stride C), both
    // offset-free; the strides must equal the opposite loop bound.
    let (sd, d_off) = match_row_col_off(didx, col, row, interner)?;
    let (ss, s_off) = match_row_col_off(sidx, row, col, interner)?;
    if !d_off.is_empty() || !s_off.is_empty() || sd != rows || ss != cols {
        return None;
    }
    // Both operands the SAME scalar type (a transpose is a copy), and distinct (an input aliasing the
    // output is an in-place transpose hazard). f32 → the f32 kernel; bf16/f16 (16-bit storage, no cast
    // in the copy) → the precision-agnostic u16 kernel.
    let st = scalar_of(target, sema);
    if st != scalar_of(value, sema) || sbase == dbase {
        return None;
    }
    let elem_u16 = match st {
        Some(mercury_types::Scalar::F32) => false,
        Some(mercury_types::Scalar::Bf16) | Some(mercury_types::Scalar::F16) => true,
        _ => return None,
    };
    Some(TransposeNest {
        src: sbase,
        dst: dbase,
        rows,
        cols,
        elem_u16,
    })
}

/// Is the whole function body a single transpose nest? Used to intercept a `@parallel` transpose
/// *before* the elementwise outliner (which would outline it into per-row *scalar* loops and lose the
/// blocked kernel). Detection only — the function is then lowered normally (`lower_fn`, `parallel =
/// true`) and the embedded recognizer in `lower_for` emits the multicore transpose. Mirrors the
/// sgemm/norm whole-function interceptions.
fn transpose_fn(body: &Block, sema: &SemaResult, interner: &Interner) -> Option<TransposeNest> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    match_transpose(pat, iter, lb, sema, interner)
}

// ============================ 2D pooling (max / avg) ============================
//
// Pool op tags: which fold the recognized pooling nest dispatches to (each maps to a distinct runtime
// symbol pair). Internal to `mir_build`.
const POOL_MAX: i64 = 0;
const POOL_AVG: i64 = 1;

/// A recognized 2D max/avg pooling nest over a `[channels, h, w]` row-major input (no padding), `kh×kw`
/// window, stride `sh×sw`. `out` is `[channels, oh, ow]`. `op` is `POOL_MAX` / `POOL_AVG`.
struct Pool2dNest {
    x: Symbol,
    out: Symbol,
    channels: Dim,
    h: Dim,
    w: Dim,
    kh: Dim,
    kw: Dim,
    sh: Dim,
    sw: Dim,
    op: i64,
}

/// `e == a * b` (left-associated `(a*b)`) where `single_path(a) == var`; returns `b` as a `Dim` (the
/// stride factor). The shape `oy*2`, `ox*2`, `c*4`, etc. Used to peel an index term.
fn mul_var_dim(e: &Expr, var: Symbol, interner: &Interner) -> Option<Dim> {
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &e.kind
    else {
        return None;
    };
    if single_path(lhs) == Some(var) {
        return as_dim(rhs, interner);
    }
    if single_path(rhs) == Some(var) {
        return as_dim(lhs, interner);
    }
    None
}

/// Recognize the idiomatic 2D pooling nest and dispatch it to `mercury_{max,avg}pool2d_f32`:
///
/// ```text
/// for c in 0..C { for oy in 0..OH { for ox in 0..OW {
///   // MAX: seed the window's first cell, fold the rest by fmax.
///   var m = x[c*H*W + (oy*SH)*W + (ox*SW)];
///   for dy in 0..KH { for dx in 0..KW {
///     let v = x[c*H*W + (oy*SH+dy)*W + (ox*SW+dx)];
///     m = fmax(m, v);
///   } }
///   out[c*OH*OW + oy*OW + ox] = m;
///   // AVG: seed 0, fold by +, divide by KH*KW.
/// } } }
/// ```
///
/// The match is strict — every stride is pinned to the dims, the window data index must be exactly
/// `c*(H*W) + (oy*SH + dy)*W + (ox*SW + dx)`, the output index `c*(OH*OW) + oy*OW + ox` with
/// `OH=(H-KH)/SH+1`, `OW=(W-KW)/SW+1` (so it never misfires), and `x`/`out` are distinct f32 arrays.
/// Max is idempotent/associative and the avg sum order is fixed (the kernel folds (dy,dx) ascending
/// then one divide), so the kernel is bit-identical to this nest — the differential gate is trivial.
/// The strided window gcc/rustc leave scalar; the AVX2 kernel folds 8 output columns at once.
fn match_pool2d(
    pat: &Pattern,
    iter: &ForIter,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<Pool2dNest> {
    // for c in 0..C { <single inner for> }
    let cvar = match &pat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (cs, ce) = range_bounds(iter)?;
    if as_int_lit(cs, interner)? != 0 {
        return None;
    }
    let channels = as_dim(ce, interner)?;
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    // for oy in 0..OH { <single inner for> }
    let (oypat, oyiter, oybody) = fusable_for(&body.stmts[0])?;
    let oyvar = match &oypat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (oys, oye) = range_bounds(oyiter)?;
    if as_int_lit(oys, interner)? != 0 {
        return None;
    }
    let oh = as_dim(oye, interner)?;
    if oybody.tail.is_some() || oybody.stmts.len() != 1 {
        return None;
    }
    // for ox in 0..OW { <3 stmts: seed; window-fold for-nest; store> }
    let (oxpat, oxiter, oxbody) = fusable_for(&oybody.stmts[0])?;
    let oxvar = match &oxpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (oxs, oxe) = range_bounds(oxiter)?;
    if as_int_lit(oxs, interner)? != 0 {
        return None;
    }
    let ow = as_dim(oxe, interner)?;
    if oxbody.tail.is_some() || oxbody.stmts.len() != 3 {
        return None;
    }
    // [0] var acc = <seed>;  (an accumulator local; the seed/op classified below.)
    let StmtKind::Let {
        pat: ap,
        init: Some(seed),
        ..
    } = &oxbody.stmts[0].kind
    else {
        return None;
    };
    let acc = match &ap.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    // [1] for dy in 0..KH { for dx in 0..KW { let v = x[..]; acc = fold(acc, v); } }
    let (dypat, dyiter, dybody) = fusable_for(&oxbody.stmts[1])?;
    let dyvar = match &dypat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (dys, dye) = range_bounds(dyiter)?;
    if as_int_lit(dys, interner)? != 0 {
        return None;
    }
    let kh = as_dim(dye, interner)?;
    if dybody.tail.is_some() || dybody.stmts.len() != 1 {
        return None;
    }
    let (dxpat, dxiter, dxbody) = fusable_for(&dybody.stmts[0])?;
    let dxvar = match &dxpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (dxs, dxe) = range_bounds(dxiter)?;
    if as_int_lit(dxs, interner)? != 0 {
        return None;
    }
    let kw = as_dim(dxe, interner)?;
    // The window body is two statements: `let v = x[..];` then `acc = fold(acc, v);`.
    if dxbody.tail.is_some() || dxbody.stmts.len() != 2 {
        return None;
    }
    let StmtKind::Let {
        pat: vp,
        init: Some(vinit),
        ..
    } = &dxbody.stmts[0].kind
    else {
        return None;
    };
    let vvar = match &vp.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    // The window data load `x[c*(H*W) + (oy*SH + dy)*W + (ox*SW + dx)]` — pin every stride to a dim.
    let (xbase, w, sh, sw) = match_pool_window_index(vinit, cvar, oyvar, oxvar, dyvar, dxvar, interner)?;
    if scalar_of(vinit, sema) != Some(mercury_types::Scalar::F32) {
        return None;
    }
    // The channel stride is `H*W`; we know `W = w` (the row stride), so `H = channel_stride / W`. But to
    // keep dims symbolic we instead re-derive `H` from the channel base term and require it consistent.
    let h = match_pool_channel_h(vinit, cvar, w, interner)?;
    // [1.2] acc = fold(acc, v): `fmax(acc, v)` (MAX) or `acc = acc + v` / `acc += v` (AVG sum).
    let op = classify_pool_fold(&dxbody.stmts[1], acc, vvar, sema, interner)?;
    // The seed must match the fold: MAX seeds the window's first cell `x[c*(H*W) + (oy*SH)*W + ox*SW]`
    // (dy=dx=0), AVG seeds the float literal `0.0`.
    match op {
        POOL_MAX => {
            let (sbase, sw_w, ssh, ssw) =
                match_pool_seed_index(seed, cvar, oyvar, oxvar, interner)?;
            if sbase != xbase || sw_w != w || ssh != sh || ssw != sw {
                return None;
            }
            // The seed's channel stride must also be H*W with the same H.
            if match_pool_channel_h(seed, cvar, w, interner)? != h {
                return None;
            }
        }
        _ => {
            if !is_float_zero(seed, interner) {
                return None;
            }
        }
    }
    // [2] out[c*(OH*OW) + oy*OW + ox] = <finalize>(acc): `acc` (MAX) or `acc / (KH*KW)` (AVG).
    let StmtKind::Assign {
        target: ot,
        op: ast::AssignOp::Assign,
        value: ov,
    } = &oxbody.stmts[2].kind
    else {
        return None;
    };
    // The store value: MAX writes `acc`; AVG writes `acc / count` where count == KH*KW.
    match op {
        POOL_MAX => {
            if single_path(ov) != Some(acc) {
                return None;
            }
        }
        _ => {
            if !pool_avg_divide_matches(ov, acc, kh, kw, interner) {
                return None;
            }
        }
    }
    // The output index `c*(OH*OW) + oy*OW + ox` — channel-major over the output plane, row stride OW.
    // `match_row_col_off` pins the `oy*OW` term + bare `ox`, leaving the channel base as the offset.
    let (obase, oidx) = as_index1(ot)?;
    let (out_ow, oy_off) = match_row_col_off(oidx, oyvar, oxvar, interner)?;
    if out_ow != ow || oy_off.len() != 1 {
        return None;
    }
    // The leftover offset term is the channel base `c*(OH*OW)` (= `(c*OH)*OW`): verify OH == oh.
    if match_pool_channel_h(oy_off[0], cvar, ow, interner) != Some(oh) {
        return None;
    }
    if scalar_of(ot, sema) != Some(mercury_types::Scalar::F32) || obase == xbase {
        return None;
    }
    Some(Pool2dNest {
        x: xbase,
        out: obase,
        channels,
        h,
        w,
        kh,
        kw,
        sh,
        sw,
        op,
    })
}

/// Match the pooling window data index `x[c*(H*W) + (oy*SH + dy)*W + (ox*SW + dx)]`, returning
/// `(x, W, SH, SW)` (the row stride and the two strides). The channel term `c*(H*W)` is left for
/// [`match_pool_channel_h`] (it derives `H`). Flattens the additive terms and classifies each by which
/// loop var it carries, so it never misfires on a non-pooling index.
fn match_pool_window_index(
    e: &Expr,
    cvar: Symbol,
    oyvar: Symbol,
    oxvar: Symbol,
    dyvar: Symbol,
    dxvar: Symbol,
    interner: &Interner,
) -> Option<(Symbol, Dim, Dim, Dim)> {
    let (xbase, idx) = as_index1(e)?;
    let mut terms = Vec::new();
    flatten_add_terms(idx, &mut terms);
    // The bare `dx` term (offset within the window's x).
    let dx_pos = terms.iter().position(|t| single_path(t) == Some(dxvar))?;
    terms.remove(dx_pos);
    // The `ox * SW` term (the window's left input column).
    let ox_pos = terms
        .iter()
        .position(|t| mul_var_dim(t, oxvar, interner).is_some())?;
    let sw = mul_var_dim(terms[ox_pos], oxvar, interner)?;
    terms.remove(ox_pos);
    // The `(oy*SH + dy) * W` term (the window's top input row, scaled by the row width W). It is a
    // product of an `Add(oy*SH, dy)` and `W`; identify the row factor by it containing `oy`.
    let row_pos = terms.iter().position(|t| {
        matches!(&t.kind, ExprKind::Binary { op: ast::BinOp::Mul, .. })
            && pool_row_factor(t, oyvar, dyvar, interner).is_some()
    })?;
    let (sh, w) = pool_row_factor(terms[row_pos], oyvar, dyvar, interner)?;
    terms.remove(row_pos);
    // The remaining term is the channel base `c*(H*W)` — verify it carries `c` (H derived elsewhere).
    if terms.len() != 1 || !pool_term_has_var(terms[0], cvar) {
        return None;
    }
    Some((xbase, w, sh, sw))
}

/// For a `(oy*SH + dy) * W` term (either `*` operand order, either `+` operand order), return
/// `(SH, W)`. The row factor is the `Add` side that contains both `oy` and `dy`; `W` is the other.
fn pool_row_factor(e: &Expr, oyvar: Symbol, dyvar: Symbol, interner: &Interner) -> Option<(Dim, Dim)> {
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &e.kind
    else {
        return None;
    };
    for (inner, wfac) in [(lhs, rhs), (rhs, lhs)] {
        // `inner == oy*SH + dy` (either addend order).
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs: a,
            rhs: b,
        } = &inner.kind
        else {
            continue;
        };
        let sh = if single_path(b) == Some(dyvar) {
            mul_var_dim(a, oyvar, interner)
        } else if single_path(a) == Some(dyvar) {
            mul_var_dim(b, oyvar, interner)
        } else {
            None
        };
        if let (Some(sh), Some(w)) = (sh, as_dim(wfac, interner)) {
            return Some((sh, w));
        }
    }
    None
}

/// Match the MAX-pool seed index `x[c*(H*W) + (oy*SH)*W + (ox*SW)]` (the window's first cell, dy=dx=0):
/// returns `(x, W, SH, SW)`. Same flatten-and-classify approach as the window index, but with no `dy`/
/// `dx` terms (the row term is `(oy*SH)*W` = `((oy*SH))*W`).
fn match_pool_seed_index(
    e: &Expr,
    cvar: Symbol,
    oyvar: Symbol,
    oxvar: Symbol,
    interner: &Interner,
) -> Option<(Symbol, Dim, Dim, Dim)> {
    let (xbase, idx) = as_index1(e)?;
    let mut terms = Vec::new();
    flatten_add_terms(idx, &mut terms);
    // `ox * SW` (the window's left column; dx=0).
    let ox_pos = terms
        .iter()
        .position(|t| mul_var_dim(t, oxvar, interner).is_some())?;
    let sw = mul_var_dim(terms[ox_pos], oxvar, interner)?;
    terms.remove(ox_pos);
    // `(oy*SH) * W` — a product whose one factor is `oy*SH` (contains oy) and the other is W.
    let row_pos = terms.iter().position(|t| {
        matches!(&t.kind, ExprKind::Binary { op: ast::BinOp::Mul, .. })
            && pool_seed_row_factor(t, oyvar, interner).is_some()
    })?;
    let (sh, w) = pool_seed_row_factor(terms[row_pos], oyvar, interner)?;
    terms.remove(row_pos);
    // The channel base `c*(H*W)`.
    if terms.len() != 1 || !pool_term_has_var(terms[0], cvar) {
        return None;
    }
    Some((xbase, w, sh, sw))
}

/// For a `(oy*SH) * W` term, return `(SH, W)`: the factor containing `oy` (itself `oy*SH`) yields SH,
/// the other is W. Handles both `*` operand orders.
fn pool_seed_row_factor(e: &Expr, oyvar: Symbol, interner: &Interner) -> Option<(Dim, Dim)> {
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &e.kind
    else {
        return None;
    };
    for (inner, wfac) in [(lhs, rhs), (rhs, lhs)] {
        if let (Some(sh), Some(w)) = (mul_var_dim(inner, oyvar, interner), as_dim(wfac, interner)) {
            return Some((sh, w));
        }
    }
    None
}

/// Given a flat index expression that contains a channel base term `c*(stride*last)` (= `(c*stride)*last`
/// — i.e. `H*W` for an input index or `OH*OW` for an output index), where `last` is the already-known
/// trailing stride (`W` or `OW`), derive and return the leading dim (`H` or `OH`). Flattens the additive
/// terms, finds the unique term carrying `c`, and matches `(c * lead) * last`.
fn match_pool_channel_h(e: &Expr, cvar: Symbol, last: Dim, interner: &Interner) -> Option<Dim> {
    let idx = match &e.kind {
        ExprKind::Index { indices, .. } if indices.len() == 1 => &indices[0],
        _ => e,
    };
    let mut terms = Vec::new();
    flatten_add_terms(idx, &mut terms);
    let cterm = terms.into_iter().find(|t| pool_term_has_var(t, cvar))?;
    // `cterm == (c * lead) * last` (left-associated). The outer `* last` peels first.
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &cterm.kind
    else {
        return None;
    };
    // Outer factor `last`; inner is `c * lead`.
    for (inner, lastfac) in [(lhs, rhs), (rhs, lhs)] {
        if as_dim(lastfac, interner) == Some(last) {
            if let Some(lead) = mul_var_dim(inner, cvar, interner) {
                return Some(lead);
            }
        }
    }
    None
}

/// Does `e`'s expression subtree mention the variable `v` as a bare path anywhere? A conservative scan
/// (used only to confirm the channel base term is the one carrying `c`).
fn pool_term_has_var(e: &Expr, v: Symbol) -> bool {
    if single_path(e) == Some(v) {
        return true;
    }
    match &e.kind {
        ExprKind::Binary { lhs, rhs, .. } => pool_term_has_var(lhs, v) || pool_term_has_var(rhs, v),
        ExprKind::Unary { expr, .. } => pool_term_has_var(expr, v),
        _ => false,
    }
}

/// Classify the pooling fold statement `acc = fold(acc, v)`: `acc = fmax(acc, v)` (either operand order)
/// → POOL_MAX; `acc = acc + v` / `acc += v` → POOL_AVG (the running window sum). `acc`/`v` are f32.
fn classify_pool_fold(
    stmt: &Stmt,
    acc: Symbol,
    vvar: Symbol,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<i64> {
    let StmtKind::Assign { target, op, value } = &stmt.kind else {
        return None;
    };
    if single_path(target) != Some(acc) {
        return None;
    }
    // `acc = fmax(acc, v)` — the MAX fold.
    if let ast::AssignOp::Assign = op {
        if let ExprKind::Call { callee, args, .. } = &value.kind {
            if args.len() == 2
                && matches!(intrinsic_callee(callee, sema, interner), Some(MathIntrinsic::Fmax))
            {
                let (a0, a1) = (single_path(&args[0]), single_path(&args[1]));
                if (a0 == Some(acc) && a1 == Some(vvar)) || (a1 == Some(acc) && a0 == Some(vvar)) {
                    return Some(POOL_MAX);
                }
            }
            return None;
        }
    }
    // `acc = acc + v` or `acc += v` — the AVG sum (accumulator on the left, matching the kernel fold).
    let addend = match op {
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
            if single_path(lhs) != Some(acc) {
                return None;
            }
            rhs
        }
        _ => return None,
    };
    if single_path(addend) == Some(vvar) {
        Some(POOL_AVG)
    } else {
        None
    }
}

/// Does the avg-pool store value `acc / (KH*KW)` divide the accumulator by the window count? Accepts
/// `acc / d` where `d` equals the product `KH*KW` (matched against the loop bounds, either factor
/// order; literal product or `(KH as f32)*(KW as f32)` etc.) — pins the divisor to the true window
/// size so it never misfires.
fn pool_avg_divide_matches(ov: &Expr, acc: Symbol, kh: Dim, kw: Dim, interner: &Interner) -> bool {
    let ExprKind::Binary {
        op: ast::BinOp::Div,
        lhs,
        rhs,
    } = &ov.kind
    else {
        return false;
    };
    if single_path(lhs) != Some(acc) {
        return false;
    }
    pool_count_matches(rhs, kh, kw, interner)
}

/// Is `d` the window count `KH*KW`? Accepts the float literal equal to `KH*KW` (when both are literal
/// dims), or `a * b` with `{a,b}` == `{KH, KW}` as dims (each a literal or a `(K as f32)` cast).
fn pool_count_matches(d: &Expr, kh: Dim, kw: Dim, interner: &Interner) -> bool {
    // A bare float/int literal equal to KH*KW (only when both dims are literals).
    if let (Dim::Lit(a), Dim::Lit(b)) = (kh, kw) {
        let prod = (a * b) as f64;
        if let ExprKind::Float(f) = &d.kind {
            return parse_float(interner.resolve(*f)) == prod;
        }
        if let ExprKind::Int(i) = &d.kind {
            return parse_int(interner.resolve(*i)) as f64 == prod;
        }
    }
    // A product `KH * KW` (each factor a dim, possibly `as f32`-cast).
    if let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &d.kind
    {
        let ld = pool_factor_dim(lhs, interner);
        let rd = pool_factor_dim(rhs, interner);
        if let (Some(ld), Some(rd)) = (ld, rd) {
            return (ld == kh && rd == kw) || (ld == kw && rd == kh);
        }
    }
    false
}

/// A window-count factor as a `Dim`: a bare literal/path, or a `(expr as <ty>)` cast around one.
fn pool_factor_dim(e: &Expr, interner: &Interner) -> Option<Dim> {
    if let ExprKind::Cast { expr, .. } = &e.kind {
        return as_dim(expr, interner);
    }
    as_dim(e, interner)
}

/// Is the whole function body a single pooling nest? Intercepts a `@parallel` pooling function *before*
/// the elementwise outliner (which would split the channel loop into per-chunk scalar loops and lose
/// the AVX2 kernel). Detection only — the function is then lowered normally (`lower_fn`, `parallel =
/// true`) and the embedded `match_pool2d` in `lower_for` emits the multicore kernel. Mirrors the
/// sgemm/transpose/colsum whole-function interceptions.
fn pool2d_fn(body: &Block, sema: &SemaResult, interner: &Interner) -> Option<Pool2dNest> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    match_pool2d(pat, iter, lb, sema, interner)
}

/// A recognized column reduction `out[j] = Σ_i x[i, j]` (`x` is `[rows, cols]`, `out` is `[cols]`).
// Column-reduction op tags (which fold the recognized nest dispatches to). Internal to `mir_build` —
// each maps to a distinct runtime symbol pair (sum / max / min), not a kernel op argument.
const COL_SUM: i64 = 0;
const COL_MAX: i64 = 1;
const COL_MIN: i64 = 2;
const COL_MAXABS: i64 = 3;
// The per-channel statistics family (same strided fold + a per-column finalize): MEAN = SUM/rows,
// SUMSQ = Σx², L2 = sqrt(SUMSQ), RMS = sqrt(SUMSQ/rows). MEAN derives from a SUM fold + a `/M` store;
// SUMSQ/L2/RMS derive from a square fold (`s += x[..]*x[..]`) + a (sqrt[/M]) store.
const COL_MEAN: i64 = 4;
const COL_SUMSQ: i64 = 5;
const COL_L2: i64 = 6;
const COL_RMS: i64 = 7;

struct ColSumNest {
    x: Symbol,
    out: Symbol,
    rows: Dim,
    cols: Dim,
    op: i64,
}

/// Recognize a column-reduction nest and dispatch it to the SIMD `mercury_col{sum,max,min}_f32`:
///
/// ```text
/// for j in 0..N { let mut s: f32 = 0.0;  for i in 0..M { s = s + x[i*N + j]; }       out[j] = s; }  // SUM
/// for j in 0..N { let mut s: f32 = x[j]; for i in 1..M { s = fmax(s, x[i*N + j]); }  out[j] = s; }  // MAX
/// for j in 0..N { let mut s: f32 = x[j]; for i in 1..M { s = fmin(s, x[i*N + j]); }  out[j] = s; }  // MIN
/// ```
///
/// — reduce each column of `x` (`[M, N]`) over the outer/batch axis into `out` (`[N]`): the **sum** is
/// the bias gradient `db = Σ_batch dY` / batch sum; **max**/**min** are per-channel statistics (the
/// quantization range, axis-0 max/min pooling). The data index `i*N + j` strides by `N` over the inner
/// loop, which gcc/rustc leave scalar (verified) for *all three* folds; the kernel streams `x` row-major
/// + 8-wide. The fold order is i-ascending per column — exactly the scalar nest's — so the kernel is its
/// own bit-exact oracle (no reassociation; both backends marshal the identical kernel). `x` and `out`
/// must be distinct f32 arrays. The strides pin `N`/`M` to the loop bounds, so it never misfires. The
/// max/min seed is the first row `x[0,j] = x[j]` (so the inner loop folds `1..M`, idempotent from 0).
fn match_colsum(
    pat: &Pattern,
    iter: &ForIter,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<ColSumNest> {
    // for j in 0..N { <3 stmts> }
    let jvar = match &pat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (js, je) = range_bounds(iter)?;
    if as_int_lit(js, interner)? != 0 {
        return None;
    }
    let cols = as_dim(je, interner)?;
    if body.tail.is_some() || body.stmts.len() != 3 {
        return None;
    }
    // [0] let mut s: f32 = <seed>;  (seed value is checked against the fold op below)
    let StmtKind::Let {
        pat: sp,
        init: Some(s0),
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    let s_sym = match &sp.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    // [1] for i in <i0>..M { s = s ⊕ x[i*N + j]; }
    let (ipat, iiter, ibody) = fusable_for(&body.stmts[1])?;
    let ivar = match &ipat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (is_, ie) = range_bounds(iiter)?;
    let istart = as_int_lit(is_, interner)?;
    let rows = as_dim(ie, interner)?;
    if ibody.tail.is_some() || ibody.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &ibody.stmts[0].kind else {
        return None;
    };
    if single_path(target) != Some(s_sym) {
        return None;
    }
    // Classify the fold + extract the reduced data expr: `s + d` / `s += d` (SUM), or `fmax(s, d)` /
    // `fmin(s, d)` (MAX / MIN, either operand order).
    let (colop, data) = classify_colreduce_body(op, value, s_sym, sema, interner)?;
    // The data is `x[i*N + j]` (index `i*cols + j`, stride `cols = N`, offset-free), f32.
    let (xbase, xidx) = as_index1(data)?;
    let (stride, off) = match_row_col_off(xidx, ivar, jvar, interner)?;
    if !off.is_empty() || stride != cols {
        return None;
    }
    if scalar_of(data, sema) != Some(mercury_types::Scalar::F32) {
        return None;
    }
    // Seed + inner-start must match the fold: SUM seeds `0.0` and folds `0..M`; MAX/MIN/MAXABS seed the
    // first row (`x[j]`, or `abs(x[j])` for MAXABS) and fold `1..M` (or `0..M` — the redundant first fold
    // is idempotent).
    match colop {
        COL_SUM | COL_SUMSQ => {
            // The additive folds (Σx and Σx²) seed `0.0` and fold `0..M`.
            if !is_float_zero(s0, interner) || istart != 0 {
                return None;
            }
        }
        _ => {
            if istart != 0 && istart != 1 {
                return None;
            }
            // The seed is the first row's element; for MAXABS it is wrapped in `abs(...)`.
            let seed_inner = if colop == COL_MAXABS {
                match &s0.kind {
                    ExprKind::Call { callee, args, .. }
                        if args.len() == 1
                            && matches!(
                                intrinsic_callee(callee, sema, interner),
                                Some(MathIntrinsic::Abs)
                            ) =>
                    {
                        &args[0]
                    }
                    _ => return None,
                }
            } else {
                s0
            };
            if index_by_var(seed_inner, jvar) != Some(xbase) {
                return None; // seed must be x[0, j] = x[j] (abs'd for MAXABS)
            }
        }
    }
    // [2] out[j] = <finalize>(s) — `s` (SUM/SUMSQ/MAX/MIN/MAXABS), `s/M` (MEAN), `sqrt(s)` (L2), or
    // `sqrt(s/M)` (RMS). The finalize divisor is pinned to the row count `ie` (= M).
    let StmtKind::Assign {
        target: ot,
        op: ast::AssignOp::Assign,
        value: ov,
    } = &body.stmts[2].kind
    else {
        return None;
    };
    let final_op = colreduce_final_op(ov, s_sym, colop, ie, sema, interner)?;
    let obase = index_by_var(ot, jvar)?; // out[j]
    if scalar_of(ot, sema) != Some(mercury_types::Scalar::F32) || xbase == obase {
        return None;
    }
    Some(ColSumNest {
        x: xbase,
        out: obase,
        rows,
        cols,
        op: final_op,
    })
}

/// Classify a column-reduction fold body `s = s ⊕ x[..]` (target already checked `== s`): returns the
/// op tag and the reduced data expr. `s + d` / `s += d` → SUM; `fmax(s, d)` / `fmin(s, d)` (either
/// operand order) → MAX / MIN. The `data` operand is returned for the caller to pin to `x[i*N+j]`.
fn classify_colreduce_body<'a>(
    op: &ast::AssignOp,
    value: &'a Expr,
    s_sym: Symbol,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(i64, &'a Expr)> {
    // `fmax(s, d)` / `fmin(s, d)` — an `=` assign of a 2-arg intrinsic call.
    if let ast::AssignOp::Assign = op {
        if let ExprKind::Call { callee, args, .. } = &value.kind {
            if args.len() == 2 {
                let colop = match intrinsic_callee(callee, sema, interner) {
                    Some(MathIntrinsic::Fmax) => Some(COL_MAX),
                    Some(MathIntrinsic::Fmin) => Some(COL_MIN),
                    _ => None,
                };
                if let Some(colop) = colop {
                    let data = if single_path(&args[0]) == Some(s_sym) {
                        &args[1]
                    } else if single_path(&args[1]) == Some(s_sym) {
                        &args[0]
                    } else {
                        return None;
                    };
                    // `fmax(s, abs(x[..]))` is the running **absmax** (the symmetric-quant scale): peel
                    // the abs and tag COL_MAXABS (the kernel abs's each element before the max fold).
                    if colop == COL_MAX {
                        if let ExprKind::Call {
                            callee: ac,
                            args: aargs,
                            ..
                        } = &data.kind
                        {
                            if aargs.len() == 1
                                && matches!(
                                    intrinsic_callee(ac, sema, interner),
                                    Some(MathIntrinsic::Abs)
                                )
                            {
                                return Some((COL_MAXABS, &aargs[0]));
                            }
                        }
                    }
                    return Some((colop, data));
                }
            }
        }
    }
    // `s += d`, or `s = s + d` (the additive accumulator must be on the left, matching the kernel fold).
    let addend = match op {
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
    // `s += d*d` (a square of structurally-equal factors) is the **sum-of-squares** fold (the energy /
    // L2 / RMS family); the inner factor is returned as the `data` so the stride/offset check pins
    // `x[i*N+j]`. Otherwise it is a plain `Σ` (the data is the addend).
    if let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &addend.kind
    {
        if exprs_struct_eq(lhs, rhs) {
            return Some((COL_SUMSQ, lhs));
        }
    }
    Some((COL_SUM, addend))
}

/// Is `e` a `sqrt(arg)` intrinsic call? Returns `arg`. The free twin of the column-finalize sqrt check
/// (no `FnLowerer` in hand), gated like `intrinsic_callee` so a user `fn sqrt` shadows it.
fn col_sqrt_arg<'a>(e: &'a Expr, sema: &SemaResult, interner: &Interner) -> Option<&'a Expr> {
    let ExprKind::Call { callee, args, .. } = &e.kind else {
        return None;
    };
    if args.len() == 1 && matches!(intrinsic_callee(callee, sema, interner), Some(MathIntrinsic::Sqrt)) {
        Some(&args[0])
    } else {
        None
    }
}

/// Does the divisor `d` equal the trip count `count` (the row count `M`) as an f32 — `M` itself, a
/// `(M as f32)` cast, or the literal `M.0`? Pins a `/M` column-mean/RMS divisor to the fold's row
/// count so it never misfires. The free twin of `FnLowerer::count_as_f32`.
fn col_divisor_matches(d: &Expr, count: &Expr, interner: &Interner) -> bool {
    if let (ExprKind::Float(f), ExprKind::Int(k)) = (&d.kind, &count.kind) {
        return parse_float(interner.resolve(*f)) == parse_int(interner.resolve(*k)) as f64;
    }
    if let ExprKind::Cast { expr, .. } = &d.kind {
        return exprs_struct_eq(expr, count);
    }
    exprs_struct_eq(d, count)
}

/// Classify the column-reduction **store** `out[j] = <finalize>(s)` into the final op code, given the
/// fold's base op (`COL_SUM`/`COL_SUMSQ`/max/min/maxabs) and the row count `count` (`= M`):
/// - `out[j] = s`              → the base op unchanged (SUM/SUMSQ/MAX/MIN/MAXABS).
/// - `out[j] = s / M`          → COL_MEAN   (only from a SUM fold).
/// - `out[j] = sqrt(s)`        → COL_L2     (only from a SUMSQ fold).
/// - `out[j] = sqrt(s / M)`    → COL_RMS    (only from a SUMSQ fold).
/// Returns `None` for any other store, and rejects a finalize on a non-additive base (max/min/maxabs).
fn colreduce_final_op(
    ov: &Expr,
    s_sym: Symbol,
    base_op: i64,
    count: &Expr,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<i64> {
    // out[j] = s — the un-finalized store.
    if single_path(ov) == Some(s_sym) {
        return Some(base_op);
    }
    // out[j] = sqrt(...) — L2 / RMS (SUMSQ only).
    if let Some(arg) = col_sqrt_arg(ov, sema, interner) {
        if base_op != COL_SUMSQ {
            return None;
        }
        if single_path(arg) == Some(s_sym) {
            return Some(COL_L2); // sqrt(s)
        }
        // sqrt(s / M)
        if let ExprKind::Binary {
            op: ast::BinOp::Div,
            lhs,
            rhs,
        } = &arg.kind
        {
            if single_path(lhs) == Some(s_sym) && col_divisor_matches(rhs, count, interner) {
                return Some(COL_RMS);
            }
        }
        return None;
    }
    // out[j] = s / M — MEAN (SUM only).
    if let ExprKind::Binary {
        op: ast::BinOp::Div,
        lhs,
        rhs,
    } = &ov.kind
    {
        if base_op == COL_SUM
            && single_path(lhs) == Some(s_sym)
            && col_divisor_matches(rhs, count, interner)
        {
            return Some(COL_MEAN);
        }
    }
    None
}

/// Resolve a call's callee to a vectorizable math intrinsic — the free-function twin of
/// `FnLowerer::vectorizable_intrinsic` (a user `fn` of the same name shadows the intrinsic). Used by the
/// free column-reduction matcher, which has no `FnLowerer` in hand.
fn intrinsic_callee(
    callee: &Expr,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<MathIntrinsic> {
    let ExprKind::Path(p) = &callee.kind else {
        return None;
    };
    if !p.is_single() {
        return None;
    }
    let name = p.first().sym;
    if matches!(sema.defs.lookup(name).map(|d| &d.kind), Some(DefKind::Fn(_))) {
        return None;
    }
    math_intrinsic(interner.resolve(name))
}

/// Is the whole function body a single column-reduction nest? Used to intercept a `@parallel` column
/// reduction *before* the elementwise outliner (which would split it into per-column-chunk scalar loops
/// and lose the SIMD kernel), mirroring the sgemm/norm/transpose interceptions.
fn colsum_fn(body: &Block, sema: &SemaResult, interner: &Interner) -> Option<ColSumNest> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    match_colsum(pat, iter, lb, sema, interner)
}

/// A recognized per-column arg-reduction nest (see [`match_colarg`]). `out` is an i32 row-index buffer;
/// `is_max` selects argmax (`true`) vs argmin (`false`).
struct ColArgNest {
    x: Symbol,
    out: Symbol,
    rows: Dim,
    cols: Dim,
    is_max: bool,
}

/// Recognize a per-**column** argmax/argmin returning the ROW index — the strided axis-0 sibling of the
/// per-row [`FnLowerer::match_rowarg`]:
/// ```text
/// for j in 0..C {
///     let mut bv: f32 = x[j];                   // seed = row 0, column j  (x[0*C + j] = x[j])
///     let mut bi = 0;                           // seed row index
///     for i in <0|1>..R { if x[i*C+j] CMP bv { bv = x[i*C+j]; bi = i; } }
///     out[j] = bi;                              // out: [i32; C]
/// }
/// ```
/// `out[j] = {argmax,argmin}_i x[i,j]`, the lowest ROW index winning on a value tie (a strict `>`/`<`
/// compare). The strided column-outer access `x[i*C+j]` (stride `C` = the inner row bound, pinned by
/// `match_row_col_off`, no batch offset) is exactly what gcc/rustc leave fully **scalar** — the column-
/// reduction lever. `out` must be an i32 array distinct from `x` (the kernel writes 4-byte row indices).
/// Free fn, like [`match_colsum`].
fn match_colarg(
    pat: &Pattern,
    iter: &ForIter,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<ColArgNest> {
    let ast::PatKind::Ident(jvar) = &pat.kind else {
        return None;
    };
    let jvar = *jvar;
    let (js, je) = range_bounds(iter)?;
    if as_int_lit(js, interner)? != 0 {
        return None;
    }
    let cols = as_dim(je, interner)?;
    if body.tail.is_some() || body.stmts.len() != 4 {
        return None;
    }
    // [0] let bv: f32 = x[j]   (the running-best value, seeded to row 0 of column j)
    let StmtKind::Let {
        pat: bvp,
        init: Some(bv_init),
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    let ast::PatKind::Ident(bv) = &bvp.kind else {
        return None;
    };
    let bv = *bv;
    // [1] let bi = 0            (the running-best row index)
    let StmtKind::Let {
        pat: bip,
        init: Some(bi_init),
        ..
    } = &body.stmts[1].kind
    else {
        return None;
    };
    let ast::PatKind::Ident(bi) = &bip.kind else {
        return None;
    };
    let bi = *bi;
    if !is_int_zero(bi_init, interner) {
        return None;
    }
    // [2] for i in <0|1>..R { if x[i*C+j] CMP bv { bv = x[i*C+j]; bi = i } }
    let (ipat, iiter, ibody) = fusable_for(&body.stmts[2])?;
    let ast::PatKind::Ident(ivar) = &ipat.kind else {
        return None;
    };
    let ivar = *ivar;
    let (is_, ie) = range_bounds(iiter)?;
    let istart = as_int_lit(is_, interner)?;
    if istart != 0 && istart != 1 {
        return None;
    }
    let rows = as_dim(ie, interner)?;
    let (xbase, is_max) = match_colarg_inner(ibody, ivar, jvar, bv, bi, cols, sema, interner)?;
    // The seed value must be exactly `x[j]` = `x[0*C + j]` (row 0 of column j) over the same base array.
    if index_by_var(bv_init, jvar) != Some(xbase) {
        return None;
    }
    // [3] out[j] = bi  — `out` an i32 array distinct from `x`.
    let StmtKind::Assign {
        target,
        op: ast::AssignOp::Assign,
        value,
    } = &body.stmts[3].kind
    else {
        return None;
    };
    let obase = index_by_var(target, jvar)?;
    if single_path(value) != Some(bi) || obase == xbase {
        return None;
    }
    if scalar_of(target, sema) != Some(mercury_types::Scalar::I32) {
        return None;
    }
    Some(ColArgNest {
        x: xbase,
        out: obase,
        rows,
        cols,
        is_max,
    })
}

/// The column argmax/argmin inner body `if x[i*C+j] CMP bv { bv = x[i*C+j]; bi = i; }` (strict `>` →
/// argmax / `<` → argmin; `bi = i` may be `i as <int>`), returning `(x_base, is_max)`. The lone `if` may
/// be the body's single statement or its tail. The data reads are the strided `x[i*C + j]` (stride `C`),
/// pinned by `as_index1` + `match_row_col_off`. Free twin of [`FnLowerer::match_rowarg_inner`].
fn match_colarg_inner(
    body: &Block,
    ivar: Symbol,
    jvar: Symbol,
    bv: Symbol,
    bi: Symbol,
    cols: Dim,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(Symbol, bool)> {
    let if_expr = match (body.stmts.as_slice(), &body.tail) {
        ([only], None) => match &only.kind {
            StmtKind::Expr(e) => e,
            _ => return None,
        },
        ([], Some(e)) => e.as_ref(),
        _ => return None,
    };
    let ExprKind::If {
        cond,
        then_branch,
        else_branch: None,
    } = &if_expr.kind
    else {
        return None;
    };
    let ExprKind::Binary { op, lhs, rhs } = &cond.kind else {
        return None;
    };
    let is_max = match op {
        ast::BinOp::Gt => true,
        ast::BinOp::Lt => false,
        _ => return None,
    };
    // `x[i*C + j]`: stride `C` over the row var `i`, bare column term `j`, no batch offset.
    let col_data = |e: &Expr| -> Option<Symbol> {
        let (xb, xi) = as_index1(e)?;
        let (stride, off) = match_row_col_off(xi, ivar, jvar, interner)?;
        if !off.is_empty() || stride != cols {
            return None;
        }
        Some(xb)
    };
    let xbase = col_data(lhs)?;
    if scalar_of(lhs, sema) != Some(mercury_types::Scalar::F32) {
        return None;
    }
    if single_path(rhs) != Some(bv) {
        return None;
    }
    if then_branch.tail.is_some() || then_branch.stmts.len() != 2 {
        return None;
    }
    let (mut saw_val, mut saw_idx) = (false, false);
    for s in &then_branch.stmts {
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &s.kind
        else {
            return None;
        };
        let t = single_path(target)?;
        if t == bv {
            if col_data(value) != Some(xbase) {
                return None;
            }
            saw_val = true;
        } else if t == bi {
            let is_i = single_path(value) == Some(ivar)
                || matches!(&value.kind, ExprKind::Cast { expr, .. } if single_path(expr) == Some(ivar));
            if !is_i {
                return None;
            }
            saw_idx = true;
        } else {
            return None;
        }
    }
    if saw_val && saw_idx {
        Some((xbase, is_max))
    } else {
        None
    }
}

/// Whole-function per-column arg-reduction — the `@parallel` interceptor probe (free twin of
/// [`colsum_fn`]).
fn colarg_fn(body: &Block, sema: &SemaResult, interner: &Interner) -> Option<ColArgNest> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    match_colarg(pat, iter, lb, sema, interner)
}

struct SoftmaxBwdNest {
    y: Symbol,
    dy: Symbol,
    dx: Symbol,
    rows: Dim,
    cols: Dim,
}

/// Recognize the **batched softmax-backward** nest and dispatch it to `mercury_softmax_bwd_f32`:
///
/// ```text
/// for r in 0..R {
///     let mut s: f32 = 0.0;
///     for j in 0..C { s = s + y[r*C + j] * dy[r*C + j]; }   // the per-row dot  Σ y·dy
///     for i in 0..C { dx[r*C + i] = y[r*C + i] * (dy[r*C + i] - s); }
/// }
/// ```
///
/// — the Jacobian-vector product of the row softmax (`dx = y·(dy − Σ y·dy)`), the gradient through every
/// attention block / classification head. The per-row dot is a reduction gcc/rustc keep **scalar**
/// (verified: no `vaddps` accumulator at `-O3`); the kernel delegates it to the proven bit-exact
/// `mercury_sreduce_f32(RED_DOT)` then applies `y·(dy − s)` 8-wide. `r*C + idx` indices are pinned by
/// `match_row_col_off` (stride `C` = the inner bound). `y`/`dy`/`dx` are f32; `dx` distinct from `y`/`dy`
/// (the apply reads them). The dot reassociates (the documented reduction exception — both backends run
/// this same kernel), so the differential gate holds. `s` is loop-local, so it cannot leak.
fn match_softmax_bwd(
    pat: &Pattern,
    iter: &ForIter,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<SoftmaxBwdNest> {
    // for r in 0..R { <3 stmts> }
    let ast::PatKind::Ident(rvar) = &pat.kind else {
        return None;
    };
    let rvar = *rvar;
    let (rs, re) = range_bounds(iter)?;
    if as_int_lit(rs, interner)? != 0 {
        return None;
    }
    let rows = as_dim(re, interner)?;
    if body.tail.is_some() || body.stmts.len() != 3 {
        return None;
    }
    // [0] let mut s: f32 = 0.0;
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
    if !is_float_zero(s0, interner) {
        return None;
    }
    // [1] for j in 0..C { s = s + a[r*C+j] * b[r*C+j]; }  (the dot; a/b roles fixed by the apply below)
    let (jpat, jiter, jbody) = fusable_for(&body.stmts[1])?;
    let ast::PatKind::Ident(jvar) = &jpat.kind else {
        return None;
    };
    let jvar = *jvar;
    let (js, je) = range_bounds(jiter)?;
    if as_int_lit(js, interner)? != 0 {
        return None;
    }
    let cols = as_dim(je, interner)?;
    if jbody.tail.is_some() || jbody.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &jbody.stmts[0].kind else {
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
    // prod = a[r*C+j] * b[r*C+j] — two row-major-indexed reads (order is symmetric; roles set below).
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: pa,
        rhs: pb,
    } = &prod.kind
    else {
        return None;
    };
    let dot_a = index_rowmaj(pa, rvar, jvar, &cols, interner)?;
    let dot_b = index_rowmaj(pb, rvar, jvar, &cols, interner)?;
    if scalar_of(pa, sema) != Some(mercury_types::Scalar::F32) {
        return None;
    }
    // [2] for i in 0..C { dx[r*C+i] = y[r*C+i] * (dy[r*C+i] - s); }
    let (ipat, iiter, ibody) = fusable_for(&body.stmts[2])?;
    let ast::PatKind::Ident(ivar) = &ipat.kind else {
        return None;
    };
    let ivar = *ivar;
    let (is_, ie) = range_bounds(iiter)?;
    if as_int_lit(is_, interner)? != 0 || as_dim(ie, interner)? != cols {
        return None;
    }
    if ibody.tail.is_some() || ibody.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign {
        target: dxt,
        op: ast::AssignOp::Assign,
        value: av,
    } = &ibody.stmts[0].kind
    else {
        return None;
    };
    let dx = index_rowmaj(dxt, rvar, ivar, &cols, interner)?;
    // av = y[r*C+i] * (dy[r*C+i] - s)  (Mul, either operand order: one Index, one Sub).
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: ml,
        rhs: mr,
    } = &av.kind
    else {
        return None;
    };
    // Identify the y factor (an index read) and the (dy - s) factor (a subtraction).
    let (y_expr, sub_expr) = if matches!(&ml.kind, ExprKind::Index { .. }) {
        (ml.as_ref(), mr.as_ref())
    } else {
        (mr.as_ref(), ml.as_ref())
    };
    let y = index_rowmaj(y_expr, rvar, ivar, &cols, interner)?;
    let ExprKind::Binary {
        op: ast::BinOp::Sub,
        lhs: dyl,
        rhs: sref,
    } = &sub_expr.kind
    else {
        return None;
    };
    if single_path(sref) != Some(s_sym) {
        return None;
    }
    let dy = index_rowmaj(dyl, rvar, ivar, &cols, interner)?;
    // The dot must be over the same two arrays {y, dy} (in either product order).
    if !((dot_a == y && dot_b == dy) || (dot_a == dy && dot_b == y)) {
        return None;
    }
    // dx must be distinct from the inputs (the apply reads y and dy while writing dx).
    if dx == y || dx == dy {
        return None;
    }
    Some(SoftmaxBwdNest {
        y,
        dy,
        dx,
        rows,
        cols,
    })
}

/// Helper: an `arr[row*C + col]` row-major read/write — return the base array symbol if `e` indexes
/// `arr` by exactly `row*cols + col` (stride `cols`, no extra offset), else `None`.
fn index_rowmaj(
    e: &Expr,
    row: Symbol,
    col: Symbol,
    cols: &Dim,
    interner: &Interner,
) -> Option<Symbol> {
    let (base, idx) = as_index1(e)?;
    let (stride, off) = match_row_col_off(idx, row, col, interner)?;
    if !off.is_empty() || &stride != cols {
        return None;
    }
    Some(base)
}

/// Is the whole function body a single batched softmax-backward nest? Intercepts a `@parallel` softmax
/// backward *before* the elementwise outliner (which would split the rows into scalar loops and lose the
/// fused dot+apply kernel), mirroring the colsum/sgemm/norm interceptions.
fn softmax_bwd_fn(body: &Block, sema: &SemaResult, interner: &Interner) -> Option<SoftmaxBwdNest> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    match_softmax_bwd(pat, iter, lb, sema, interner)
}

struct RmsNormBwdNest {
    x: Symbol,
    dy: Symbol,
    gamma: Symbol,
    dx: Symbol,
    rows: Dim,
    cols: Dim,
    eps_bits: i64,
}

/// A recognized batched softmax cross-entropy forward-loss nest (see [`FnLowerer::match_xent`]).
struct XentNest {
    x: Symbol,      // logits [rows, cols]
    target: Symbol, // i32 labels [rows]
    loss: Symbol,   // f32 output [rows]
    rows: Dim,
    cols: Dim,
}

/// A recognized batched cross-entropy backward nest (see [`FnLowerer::match_xent_bwd`]).
struct XentBwdNest {
    x: Symbol,      // logits [rows, cols]
    target: Symbol, // i32 labels [rows]
    dx: Symbol,     // f32 gradient [rows, cols]
    rows: Dim,
    cols: Dim,
}

/// Is the whole function body a single batched cross-entropy **backward** nest? `@parallel` interceptor
/// probe, like [`xent_fn`].
fn xent_bwd_fn(
    f: &FnDecl,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
) -> bool {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return false;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return false;
    };
    let mut diags = Vec::new();
    let probe = FnLowerer {
        builder: Builder::new(f.name.sym, MirType::I64),
        sema,
        interner,
        diags: &mut diags,
        scopes: vec![HashMap::new()],
        terminated: false,
        loops: Vec::new(),
        gemm,
        parallel_fn: false,
        vec_loads: HashMap::new(),
        sret: None,
    };
    probe.match_xent_bwd(pat, iter, lb).is_some()
}

/// Is the whole function body a single batched cross-entropy nest? Intercepts a `@parallel` xent
/// *before* the elementwise outliner, mirroring [`is_batched_norm_fn`] (a throwaway `FnLowerer` probe).
fn xent_fn(
    f: &FnDecl,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
) -> bool {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return false;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return false;
    };
    let mut diags = Vec::new();
    let probe = FnLowerer {
        builder: Builder::new(f.name.sym, MirType::I64),
        sema,
        interner,
        diags: &mut diags,
        scopes: vec![HashMap::new()],
        terminated: false,
        loops: Vec::new(),
        gemm,
        parallel_fn: false,
        vec_loads: HashMap::new(),
        sret: None,
    };
    probe.match_xent(pat, iter, lb).is_some()
}

/// A recognized batched log-sum-exp nest (see [`FnLowerer::match_logsumexp`]).
struct LogsumexpNest {
    x: Symbol,
    out: Symbol,
    rows: Dim,
    cols: Dim,
}

/// Is the whole function body a single batched log-sum-exp nest? Intercepts a `@parallel` logsumexp
/// before the elementwise outliner (a throwaway `FnLowerer` probe, like [`xent_fn`]).
fn logsumexp_fn(
    f: &FnDecl,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
) -> bool {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return false;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return false;
    };
    let mut diags = Vec::new();
    let probe = FnLowerer {
        builder: Builder::new(f.name.sym, MirType::I64),
        sema,
        interner,
        diags: &mut diags,
        scopes: vec![HashMap::new()],
        terminated: false,
        loops: Vec::new(),
        gemm,
        parallel_fn: false,
        vec_loads: HashMap::new(),
        sret: None,
    };
    probe.match_logsumexp(pat, iter, lb).is_some()
}

/// A recognized batched KL-divergence nest (see [`FnLowerer::match_kldiv`]).
struct KldivNest {
    p: Symbol,
    q: Symbol,
    out: Symbol,
    rows: Dim,
    cols: Dim,
}

/// A recognized batched row-entropy nest (see [`FnLowerer::match_entropy`]).
struct EntropyNest {
    p: Symbol,
    out: Symbol,
    rows: Dim,
    cols: Dim,
}

/// A recognized batched soft-label cross-entropy nest (see [`FnLowerer::match_kd_loss`]).
struct KdLossNest {
    x: Symbol,
    q: Symbol,
    out: Symbol,
    rows: Dim,
    cols: Dim,
}

/// A recognized batched per-row arg-reduction nest (see [`FnLowerer::match_rowarg`]). `out` is an i32
/// index buffer; `is_max` selects argmax (`true`) vs argmin (`false`).
struct RowArgNest {
    x: Symbol,
    out: Symbol,
    rows: Dim,
    cols: Dim,
    is_max: bool,
}

/// A recognized batched per-row prefix-sum (cumsum) nest (see [`FnLowerer::match_cumsum`]).
struct CumsumNest {
    x: Symbol,
    out: Symbol,
    rows: Dim,
    cols: Dim,
}

/// A recognized batched per-row first-order linear-recurrence scan (see [`FnLowerer::match_lrscan`]):
/// `out[r,t] = a[r,t]·h_{t-1} + b[r,t]`, `h_{-1} = 0` per row. `a` is the gate, `b` the input.
struct LrscanNest {
    a: Symbol,
    b: Symbol,
    out: Symbol,
    rows: Dim,
    cols: Dim,
}

/// A recognized batched per-row cumulative max/min nest (see [`FnLowerer::match_cumminmax`]). `is_max`
/// selects cummax (`true`) vs cummin (`false`).
struct CumMinMaxNest {
    x: Symbol,
    out: Symbol,
    rows: Dim,
    cols: Dim,
    is_max: bool,
}

/// A recognized embedding-lookup nest (see [`FnLowerer::match_embedding`]): `out[t,:] = weight[ids[t],:]`.
/// `t_rows` is the token count (outer loop bound), `h` the hidden width (inner bound = the row stride).
/// There is no compile-time table-height (`V`) in the naive source — the emitter passes a large sentinel
/// so the kernel's out-of-range→zero clamp never fires for the in-range ids a well-typed program uses.
struct EmbeddingNest {
    out: Symbol,
    weight: Symbol,
    ids: Symbol,
    t_rows: Dim,
    h: Dim,
}

/// A recognized scatter-add / embedding-gradient-backward nest (see [`FnLowerer::match_scatter`]):
/// `grad_w[ids[t], :] += grad_out[t, :]`. `t_rows` is the token count, `h` the hidden width (row stride),
/// and `total` is grad_w's full array length `V*H` (from its sema type) — the emitter divides it by `H`
/// to recover the real table height `V` the parallel kernel partitions across cores.
struct ScatterNest {
    grad_w: Symbol,
    grad_out: Symbol,
    ids: Symbol,
    t_rows: Dim,
    h: Dim,
    total: u64,
}

/// Build a throwaway `FnLowerer` probe over a single-`for`-statement body and run `check` on the inner
/// loop — the shared `@parallel`-interceptor scaffold (the per-recognizer twin of [`is_batched_norm_fn`]).
fn probe_single_for(
    f: &FnDecl,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
    check: impl FnOnce(&FnLowerer, &Pattern, &ForIter, &Block) -> bool,
) -> bool {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return false;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return false;
    };
    let mut diags = Vec::new();
    let probe = FnLowerer {
        builder: Builder::new(f.name.sym, MirType::I64),
        sema,
        interner,
        diags: &mut diags,
        scopes: vec![HashMap::new()],
        terminated: false,
        loops: Vec::new(),
        gemm,
        parallel_fn: false,
        vec_loads: HashMap::new(),
        sret: None,
    };
    check(&probe, pat, iter, lb)
}

/// The `f32` bit pattern of a non-negative float literal (the `eps` ABI slot), or `None`. Free twin of
/// `FnLowerer::float_lit_bits` (positive case only — `eps` is positive).
fn float_lit_bits_free(e: &Expr, interner: &Interner) -> Option<i64> {
    let ExprKind::Float(t) = &e.kind else {
        return None;
    };
    Some((parse_float(interner.resolve(*t)) as f32).to_bits() as i64)
}

/// Match a single-statement reduction body `acc = acc + <addend>` (or `acc += <addend>`), returning the
/// addend expr. Free, shared by the norm-backward matchers.
fn match_add_accum<'a>(body: &'a Block, acc: Symbol) -> Option<&'a Expr> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &body.stmts[0].kind else {
        return None;
    };
    if single_path(target) != Some(acc) {
        return None;
    }
    match op {
        ast::AssignOp::Add => Some(value),
        ast::AssignOp::Assign => {
            let ExprKind::Binary {
                op: ast::BinOp::Add,
                lhs,
                rhs,
            } = &value.kind
            else {
                return None;
            };
            if single_path(lhs) != Some(acc) {
                return None;
            }
            Some(rhs)
        }
        _ => None,
    }
}

/// Recognize the **batched RMSNorm backward** (input-gradient) nest and dispatch it to
/// `mercury_rmsnorm_bwd_f32`. The canonical per-row form (over `x`/`dy`/`gamma` `[R, C]`, the learned
/// per-column scale `gamma` required):
///
/// ```text
/// for r in 0..R {
///     let mut ms: f32 = 0.0;
///     for i in 0..C { ms = ms + x[r*C + i] * x[r*C + i]; }      // mean-square sum
///     let rinv: f32 = 1.0 / sqrt(ms / C + eps);                 // the rms_inv scale
///     let mut sg: f32 = 0.0;
///     for i in 0..C { sg = sg + dy[r*C + i] * gamma[i] * x[r*C + i]; }  // the grad dot
///     let coef: f32 = rinv * rinv * sg / C;                     // loop-invariant
///     for i in 0..C { dx[r*C + i] = rinv * (dy[r*C + i] * gamma[i] - x[r*C + i] * coef); }
/// }
/// ```
///
/// — the gradient `dx = r·(g − x·r²·(Σ g·x)/C)`, `g = dy·γ`, that flows through every RMSNorm in a
/// transformer's backward pass (Llama/Mistral/Qwen). The two per-row reductions (`Σx²`, `Σ g·x`)
/// gcc/rustc keep **scalar** (no `vaddps` accumulator at `-O3`); the kernel folds them 8-wide then
/// applies the gradient. The intermediates `ms`/`rinv`/`sg`/`coef` are loop-local to the `for r` body,
/// so they cannot leak. The reductions reassociate (the documented exception — both backends run this
/// same kernel), so the differential gate holds.
fn match_rmsnorm_bwd(
    pat: &Pattern,
    iter: &ForIter,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<RmsNormBwdNest> {
    // for r in 0..R { <7 stmts> }
    let ast::PatKind::Ident(rvar) = &pat.kind else {
        return None;
    };
    let rvar = *rvar;
    let (rs, re) = range_bounds(iter)?;
    if as_int_lit(rs, interner)? != 0 {
        return None;
    }
    let rows = as_dim(re, interner)?;
    if body.tail.is_some() || body.stmts.len() != 7 {
        return None;
    }
    // [0] let mut ms = 0.0;
    let (ms, ms0) = stmt_let_init(&body.stmts[0])?;
    if !is_float_zero(ms0, interner) {
        return None;
    }
    // [1] for i in 0..C { ms = ms + x[r*C+i] * x[r*C+i]; }
    let (iv1, ce1, b1) = stmt_range0_for(&body.stmts[1], interner)?;
    let cols = as_dim(ce1, interner)?;
    let sq = match_add_accum(b1, ms)?;
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: sqa,
        rhs: sqb,
    } = &sq.kind
    else {
        return None;
    };
    let x = index_rowmaj(sqa, rvar, iv1, &cols, interner)?;
    if index_rowmaj(sqb, rvar, iv1, &cols, interner)? != x {
        return None;
    }
    if scalar_of(sqa, sema) != Some(mercury_types::Scalar::F32) {
        return None;
    }
    // [2] let rinv = 1.0 / sqrt(ms / C + eps);
    let (rinv, rinv0) = stmt_let_init(&body.stmts[2])?;
    let eps_bits = match_rsqrt_meansq(rinv0, ms, ce1, sema, interner)?;
    // [3] let mut sg = 0.0;
    let (sg, sg0) = stmt_let_init(&body.stmts[3])?;
    if !is_float_zero(sg0, interner) {
        return None;
    }
    // [4] for i in 0..C { sg = sg + dy[r*C+i] * gamma[i] * x[r*C+i]; }  ( ((dy*gamma)*x) )
    let (iv4, ce4, b4) = stmt_range0_for(&body.stmts[4], interner)?;
    if !exprs_struct_eq(ce4, ce1) {
        return None;
    }
    let dot = match_add_accum(b4, sg)?;
    let (dy, gamma) = match_dygx(dot, x, rvar, iv4, &cols, interner)?;
    // [5] let coef = rinv * rinv * sg / C;  ( ((rinv*rinv)*sg) / C )
    let (coef, coef0) = stmt_let_init(&body.stmts[5])?;
    let ExprKind::Binary {
        op: ast::BinOp::Div,
        lhs: cnum,
        rhs: cden,
    } = &coef0.kind
    else {
        return None;
    };
    if !col_divisor_matches(cden, ce1, interner) {
        return None;
    }
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: rr,
        rhs: sgf,
    } = &cnum.kind
    else {
        return None;
    };
    if single_path(sgf) != Some(sg) {
        return None;
    }
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: r1,
        rhs: r2,
    } = &rr.kind
    else {
        return None;
    };
    if single_path(r1) != Some(rinv) || single_path(r2) != Some(rinv) {
        return None;
    }
    // [6] for i in 0..C { dx[r*C+i] = rinv * (dy[r*C+i]*gamma[i] - x[r*C+i]*coef); }
    let (iv6, ce6, b6) = stmt_range0_for(&body.stmts[6], interner)?;
    if !exprs_struct_eq(ce6, ce1) {
        return None;
    }
    if b6.tail.is_some() || b6.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign {
        target: dxt,
        op: ast::AssignOp::Assign,
        value: av,
    } = &b6.stmts[0].kind
    else {
        return None;
    };
    let dx = index_rowmaj(dxt, rvar, iv6, &cols, interner)?;
    // av = rinv * (g - x*coef): a Mul of `rinv` and a Sub (either operand order).
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: al,
        rhs: ar,
    } = &av.kind
    else {
        return None;
    };
    let sub = if single_path(al) == Some(rinv) {
        ar.as_ref()
    } else if single_path(ar) == Some(rinv) {
        al.as_ref()
    } else {
        return None;
    };
    let ExprKind::Binary {
        op: ast::BinOp::Sub,
        lhs: gterm,
        rhs: xterm,
    } = &sub.kind
    else {
        return None;
    };
    // gterm = dy[r*C+i]*gamma[i] (same dy, gamma as the dot); xterm = x[r*C+i]*coef.
    let (dy2, gamma2) = match_dygamma(gterm, rvar, iv6, &cols, interner)?;
    if dy2 != dy || gamma2 != gamma {
        return None;
    }
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: xl,
        rhs: xr,
    } = &xterm.kind
    else {
        return None;
    };
    let (xf, cf) = if single_path(xr) == Some(coef) {
        (xl.as_ref(), xr.as_ref())
    } else if single_path(xl) == Some(coef) {
        (xr.as_ref(), xl.as_ref())
    } else {
        return None;
    };
    let _ = cf;
    if index_rowmaj(xf, rvar, iv6, &cols, interner)? != x {
        return None;
    }
    if dx == x || dx == dy || dx == gamma {
        return None;
    }
    Some(RmsNormBwdNest {
        x,
        dy,
        gamma,
        dx,
        rows,
        cols,
        eps_bits,
    })
}

/// Match `let name = init` returning `(name, init)`. Free twin of `FnLowerer::let_init`.
fn stmt_let_init(stmt: &Stmt) -> Option<(Symbol, &Expr)> {
    let StmtKind::Let {
        pat,
        init: Some(init),
        ..
    } = &stmt.kind
    else {
        return None;
    };
    let ast::PatKind::Ident(s) = &pat.kind else {
        return None;
    };
    Some((*s, init))
}

/// Match `for v in 0..N { body }` returning `(v, N, body)`. Free twin of `FnLowerer::as_range0_for`.
fn stmt_range0_for<'a>(stmt: &'a Stmt, interner: &Interner) -> Option<(Symbol, &'a Expr, &'a Block)> {
    let StmtKind::For {
        pat, iter, body, ..
    } = &stmt.kind
    else {
        return None;
    };
    let ast::PatKind::Ident(v) = &pat.kind else {
        return None;
    };
    let (s, e) = range_bounds(iter)?;
    if as_int_lit(s, interner)? != 0 {
        return None;
    }
    Some((*v, e, body))
}

/// The argument of a reciprocal-square-root, written either as `rsqrt(arg)` or `1.0 / sqrt(arg)` (the
/// two spellings the forward norm's `as_rsqrt_arg` accepts), or `None`. Free twin of that method.
fn recip_sqrt_arg<'a>(e: &'a Expr, sema: &SemaResult, interner: &Interner) -> Option<&'a Expr> {
    // rsqrt(arg)
    if let ExprKind::Call { callee, args, .. } = &e.kind {
        if args.len() == 1
            && matches!(
                intrinsic_callee(callee, sema, interner),
                Some(MathIntrinsic::Rsqrt)
            )
        {
            return Some(&args[0]);
        }
    }
    // 1.0 / sqrt(arg)
    let ExprKind::Binary {
        op: ast::BinOp::Div,
        lhs,
        rhs,
    } = &e.kind
    else {
        return None;
    };
    if !matches!(&lhs.kind, ExprKind::Float(t) if parse_float(interner.resolve(*t)) == 1.0) {
        return None;
    }
    col_sqrt_arg(rhs, sema, interner)
}

/// Match `rsqrt(ms / C + eps)` (or `1.0 / sqrt(ms / C + eps)`) — the rms_inv binding → the `eps` bits;
/// pins the divisor to `C` (`ce`) and the dividend to `ms`.
fn match_rsqrt_meansq(
    e: &Expr,
    ms: Symbol,
    ce: &Expr,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<i64> {
    let arg = recip_sqrt_arg(e, sema, interner)?;
    let ExprKind::Binary {
        op: ast::BinOp::Add,
        lhs: msdiv,
        rhs: eps,
    } = &arg.kind
    else {
        return None;
    };
    let ExprKind::Binary {
        op: ast::BinOp::Div,
        lhs: msl,
        rhs: msr,
    } = &msdiv.kind
    else {
        return None;
    };
    if single_path(msl) != Some(ms) || !col_divisor_matches(msr, ce, interner) {
        return None;
    }
    float_lit_bits_free(eps, interner)
}

/// Match `dy[r*C+i] * gamma[i] * x[r*C+i]` (left-assoc `((dy*gamma)*x)`) → `(dy, gamma)`, pinning the
/// `x` factor to `x` and `gamma` to a column-indexed (by `i`) array.
fn match_dygx(
    e: &Expr,
    x: Symbol,
    rvar: Symbol,
    iv: Symbol,
    cols: &Dim,
    interner: &Interner,
) -> Option<(Symbol, Symbol)> {
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &e.kind
    else {
        return None;
    };
    // One factor is x[r*C+i]; the other is the (dy*gamma) product.
    let (dyg, xf) = if index_rowmaj(rhs, rvar, iv, cols, interner) == Some(x) {
        (lhs.as_ref(), rhs.as_ref())
    } else if index_rowmaj(lhs, rvar, iv, cols, interner) == Some(x) {
        (rhs.as_ref(), lhs.as_ref())
    } else {
        return None;
    };
    let _ = xf;
    match_dygamma(dyg, rvar, iv, cols, interner)
}

/// Match `dy[r*C+i] * gamma[i]` → `(dy, gamma)`: one factor a row-major `[r*C+i]` read (`dy`), the other
/// a column `[i]` read (`gamma`).
fn match_dygamma(
    e: &Expr,
    rvar: Symbol,
    iv: Symbol,
    cols: &Dim,
    interner: &Interner,
) -> Option<(Symbol, Symbol)> {
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &e.kind
    else {
        return None;
    };
    if let (Some(dy), Some(g)) = (
        index_rowmaj(lhs, rvar, iv, cols, interner),
        index_by_var(rhs, iv),
    ) {
        return Some((dy, g));
    }
    if let (Some(dy), Some(g)) = (
        index_rowmaj(rhs, rvar, iv, cols, interner),
        index_by_var(lhs, iv),
    ) {
        return Some((dy, g));
    }
    None
}

/// Is the whole function body a single batched RMSNorm-backward nest? Intercepts a `@parallel` RMSNorm
/// backward *before* the elementwise outliner, mirroring the softmax-backward interception.
fn rmsnorm_bwd_fn(body: &Block, sema: &SemaResult, interner: &Interner) -> Option<RmsNormBwdNest> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    match_rmsnorm_bwd(pat, iter, lb, sema, interner)
}

/// A recognized batched LayerNorm-backward (input-gradient) nest (see [`match_layernorm_bwd`]). Same
/// 7-arg ABI as [`RmsNormBwdNest`] — distinct struct for clarity.
struct LayerNormBwdNest {
    x: Symbol,
    dy: Symbol,
    gamma: Symbol,
    dx: Symbol,
    rows: Dim,
    cols: Dim,
    eps_bits: i64,
}

/// Is `e` the centered read `x[r*C+iv] - mean`?
fn is_centered_sub(
    e: &Expr,
    x: Symbol,
    mean: Symbol,
    rvar: Symbol,
    iv: Symbol,
    cols: &Dim,
    interner: &Interner,
) -> bool {
    matches!(&e.kind, ExprKind::Binary { op: ast::BinOp::Sub, lhs, rhs }
        if index_rowmaj(lhs, rvar, iv, cols, interner) == Some(x) && single_path(rhs) == Some(mean))
}

/// Is `e` the normalized read `xhat = (x[r*C+iv] - mean) * rstd` (either factor order)?
fn is_xhat(
    e: &Expr,
    x: Symbol,
    mean: Symbol,
    rstd: Symbol,
    rvar: Symbol,
    iv: Symbol,
    cols: &Dim,
    interner: &Interner,
) -> bool {
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &e.kind
    else {
        return false;
    };
    (is_centered_sub(lhs, x, mean, rvar, iv, cols, interner) && single_path(rhs) == Some(rstd))
        || (is_centered_sub(rhs, x, mean, rvar, iv, cols, interner) && single_path(lhs) == Some(rstd))
}

/// Recognize the **batched LayerNorm backward** (input gradient) and dispatch it to
/// `mercury_layernorm_bwd_f32`. The canonical per-row form (`xhat = (x−mean)·rstd`, `g = dy·γ`):
///
/// ```text
/// for r in 0..R {
///     let mut sm = 0.0; for i { sm = sm + x[r*C+i]; }          let mean = sm / C;
///     let mut vv = 0.0; for i { vv = vv + (x[r*C+i]-mean)*(x[r*C+i]-mean); }
///     let rstd = 1.0 / sqrt(vv / C + eps);
///     let mut s1 = 0.0; for i { s1 = s1 + dy[r*C+i]*gamma[i]; }
///     let mut s2 = 0.0; for i { s2 = s2 + dy[r*C+i]*gamma[i]*((x[r*C+i]-mean)*rstd); }
///     let m1 = s1 / C; let m2 = s2 / C;
///     for i { dx[r*C+i] = rstd * (dy[r*C+i]*gamma[i] - m1 - ((x[r*C+i]-mean)*rstd)*m2); }
/// }
/// ```
///
/// — the gradient `dx = rstd·(g − mean(g) − xhat·mean(g·xhat))` that flows through every LayerNorm
/// (GPT-2/BERT/ViT). Four per-row reductions (Σx, Σ(x−mean)², Σg, Σg·xhat) gcc/rustc keep scalar; the
/// kernel folds them 8-wide. The reductions reassociate (the documented exception — both backends run
/// this kernel), so the differential gate holds.
fn match_layernorm_bwd(
    pat: &Pattern,
    iter: &ForIter,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<LayerNormBwdNest> {
    let ast::PatKind::Ident(rvar) = &pat.kind else {
        return None;
    };
    let rvar = *rvar;
    let (rs, re) = range_bounds(iter)?;
    if as_int_lit(rs, interner)? != 0 {
        return None;
    }
    let rows = as_dim(re, interner)?;
    if body.tail.is_some() || body.stmts.len() != 13 {
        return None;
    }
    // [0] let sm = 0.0;  [1] for i { sm = sm + x[r*C+i] }
    let (sm, sm0) = stmt_let_init(&body.stmts[0])?;
    if !is_float_zero(sm0, interner) {
        return None;
    }
    let (iv1, ce, b1) = stmt_range0_for(&body.stmts[1], interner)?;
    let cols = as_dim(ce, interner)?;
    let x = index_rowmaj(match_add_accum(b1, sm)?, rvar, iv1, &cols, interner)?;
    // [2] let mean = sm / C
    let (mean, mean0) = stmt_let_init(&body.stmts[2])?;
    let ExprKind::Binary {
        op: ast::BinOp::Div,
        lhs: ml,
        rhs: mr,
    } = &mean0.kind
    else {
        return None;
    };
    if single_path(ml) != Some(sm) || !col_divisor_matches(mr, ce, interner) {
        return None;
    }
    // [3] let vv = 0.0;  [4] for i { vv = vv + (x[r*C+i]-mean)*(x[r*C+i]-mean) }
    let (vv, vv0) = stmt_let_init(&body.stmts[3])?;
    if !is_float_zero(vv0, interner) {
        return None;
    }
    let (iv4, ce4, b4) = stmt_range0_for(&body.stmts[4], interner)?;
    if !exprs_struct_eq(ce4, ce) {
        return None;
    }
    let sq = match_add_accum(b4, vv)?;
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: sql,
        rhs: sqr,
    } = &sq.kind
    else {
        return None;
    };
    if !is_centered_sub(sql, x, mean, rvar, iv4, &cols, interner)
        || !is_centered_sub(sqr, x, mean, rvar, iv4, &cols, interner)
    {
        return None;
    }
    // [5] let rstd = 1.0/sqrt(vv/C + eps)
    let (rstd, rstd0) = stmt_let_init(&body.stmts[5])?;
    let eps_bits = match_rsqrt_meansq(rstd0, vv, ce, sema, interner)?;
    // [6] let s1 = 0.0;  [7] for i { s1 = s1 + dy[r*C+i]*gamma[i] }
    let (s1, s10) = stmt_let_init(&body.stmts[6])?;
    if !is_float_zero(s10, interner) {
        return None;
    }
    let (iv7, ce7, b7) = stmt_range0_for(&body.stmts[7], interner)?;
    if !exprs_struct_eq(ce7, ce) {
        return None;
    }
    let (dy, gamma) = match_dygamma(match_add_accum(b7, s1)?, rvar, iv7, &cols, interner)?;
    // [8] let s2 = 0.0;  [9] for i { s2 = s2 + dy[r*C+i]*gamma[i] * ((x[r*C+i]-mean)*rstd) }
    let (s2, s20) = stmt_let_init(&body.stmts[8])?;
    if !is_float_zero(s20, interner) {
        return None;
    }
    let (iv9, ce9, b9) = stmt_range0_for(&body.stmts[9], interner)?;
    if !exprs_struct_eq(ce9, ce) {
        return None;
    }
    let s2add = match_add_accum(b9, s2)?;
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: s2l,
        rhs: s2r,
    } = &s2add.kind
    else {
        return None;
    };
    // one factor is dy*gamma (== (dy,gamma)), the other is xhat
    let s2_ok = (match_dygamma(s2l, rvar, iv9, &cols, interner) == Some((dy, gamma))
        && is_xhat(s2r, x, mean, rstd, rvar, iv9, &cols, interner))
        || (match_dygamma(s2r, rvar, iv9, &cols, interner) == Some((dy, gamma))
            && is_xhat(s2l, x, mean, rstd, rvar, iv9, &cols, interner));
    if !s2_ok {
        return None;
    }
    // [10] let m1 = s1 / C;  [11] let m2 = s2 / C
    let (m1, m10) = stmt_let_init(&body.stmts[10])?;
    if !matches!(&m10.kind, ExprKind::Binary { op: ast::BinOp::Div, lhs, rhs }
        if single_path(lhs) == Some(s1) && col_divisor_matches(rhs, ce, interner))
    {
        return None;
    }
    let (m2, m20) = stmt_let_init(&body.stmts[11])?;
    if !matches!(&m20.kind, ExprKind::Binary { op: ast::BinOp::Div, lhs, rhs }
        if single_path(lhs) == Some(s2) && col_divisor_matches(rhs, ce, interner))
    {
        return None;
    }
    // [12] for i { dx[r*C+i] = rstd * (dy[r*C+i]*gamma[i] - m1 - ((x[r*C+i]-mean)*rstd)*m2) }
    let (iv12, ce12, b12) = stmt_range0_for(&body.stmts[12], interner)?;
    if !exprs_struct_eq(ce12, ce) || b12.tail.is_some() || b12.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign {
        target: dxt,
        op: ast::AssignOp::Assign,
        value: av,
    } = &b12.stmts[0].kind
    else {
        return None;
    };
    let dx = index_rowmaj(dxt, rvar, iv12, &cols, interner)?;
    // av = rstd * inner  (Mul, either order)
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: al,
        rhs: ar,
    } = &av.kind
    else {
        return None;
    };
    let inner = if single_path(al) == Some(rstd) {
        ar.as_ref()
    } else if single_path(ar) == Some(rstd) {
        al.as_ref()
    } else {
        return None;
    };
    // inner = (g - m1) - xhat*m2   (left-assoc Sub of Sub)
    let ExprKind::Binary {
        op: ast::BinOp::Sub,
        lhs: gm1,
        rhs: xm2,
    } = &inner.kind
    else {
        return None;
    };
    let ExprKind::Binary {
        op: ast::BinOp::Sub,
        lhs: g_e,
        rhs: m1_e,
    } = &gm1.kind
    else {
        return None;
    };
    if match_dygamma(g_e, rvar, iv12, &cols, interner) != Some((dy, gamma))
        || single_path(m1_e) != Some(m1)
    {
        return None;
    }
    // xm2 = xhat * m2  (either order)
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: xl,
        rhs: xr,
    } = &xm2.kind
    else {
        return None;
    };
    let xm2_ok = (is_xhat(xl, x, mean, rstd, rvar, iv12, &cols, interner)
        && single_path(xr) == Some(m2))
        || (is_xhat(xr, x, mean, rstd, rvar, iv12, &cols, interner)
            && single_path(xl) == Some(m2));
    if !xm2_ok {
        return None;
    }
    if dx == x || dx == dy || dx == gamma {
        return None;
    }
    Some(LayerNormBwdNest {
        x,
        dy,
        gamma,
        dx,
        rows,
        cols,
        eps_bits,
    })
}

/// Whole-function `@parallel` LayerNorm-backward interceptor (before the outliner).
fn layernorm_bwd_fn(
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<LayerNormBwdNest> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    match_layernorm_bwd(pat, iter, lb, sema, interner)
}

/// A recognized RoPE (rotary position embedding) nest (see [`match_rope`]). `backward` selects the
/// transpose/inverse rotation (the gradient) and thus the `mercury_rope_bwd_f32` kernel.
struct RopeNest {
    x: Symbol,
    inv_freq: Symbol,
    out: Symbol,
    rows: Dim,
    half: Dim,
    backward: bool,
}

/// `base[row*stride + col (+ extra…)]` — return `(base, stride, extra_offset_terms)`. The generic twin
/// of `index_rowmaj` that *keeps* any leftover offset beyond `row*stride + col` (for RoPE's `+ half`).
fn index_strided<'a>(
    e: &'a Expr,
    row: Symbol,
    col: Symbol,
    interner: &Interner,
) -> Option<(Symbol, Dim, Vec<&'a Expr>)> {
    let (base, idx) = as_index1(e)?;
    let (stride, off) = match_row_col_off(idx, row, col, interner)?;
    Some((base, stride, off))
}

/// The store `out[row*stride + col (+ extra)] = value` — return `(out, value, stride, extra)`.
fn assign_strided<'a>(
    stmt: &'a Stmt,
    row: Symbol,
    col: Symbol,
    interner: &Interner,
) -> Option<(Symbol, &'a Expr, Dim, Vec<&'a Expr>)> {
    let StmtKind::Assign {
        target,
        op: ast::AssignOp::Assign,
        value,
    } = &stmt.kind
    else {
        return None;
    };
    let (out, stride, off) = index_strided(target, row, col, interner)?;
    Some((out, value, stride, off))
}

/// Is `e` the product `p · q` (either factor order, by single-segment path)?
fn is_prod_of(e: &Expr, p: Symbol, q: Symbol) -> bool {
    matches!(&e.kind,
        ExprKind::Binary { op: ast::BinOp::Mul, lhs, rhs }
            if (single_path(lhs) == Some(p) && single_path(rhs) == Some(q))
                || (single_path(lhs) == Some(q) && single_path(rhs) == Some(p)))
}

/// Is `e` a single-arg call to the intrinsic `intr` over the path `arg`? (`cos(theta)` / `sin(theta)`.)
fn is_unary_intrinsic_of(
    e: &Expr,
    intr: MathIntrinsic,
    arg: Symbol,
    sema: &SemaResult,
    interner: &Interner,
) -> bool {
    matches!(&e.kind, ExprKind::Call { callee, args, .. }
        if args.len() == 1
            && intrinsic_callee(callee, sema, interner) == Some(intr)
            && single_path(&args[0]) == Some(arg))
}

/// The single leftover term is the half offset — `exprs_struct_eq` to the inner loop bound `he`.
fn offset_is_half(off: &[&Expr], he: &Expr) -> bool {
    off.len() == 1 && exprs_struct_eq(off[0], he)
}

/// The stride `D` is twice the half `H` — both must be integer literals with `D == 2·H` (so the
/// source's `[rows, 2·half]` layout matches what the kernel assumes; symbolic dims decline).
fn stride_is_twice_half(stride: &Dim, half: &Dim) -> bool {
    matches!((stride, half), (Dim::Lit(d), Dim::Lit(h)) if *d == 2 * *h)
}

/// Recognize the **RoPE (rotary position embedding)** nest and dispatch it to `mercury_rope_f32`. The
/// canonical half-split (Llama/GPT-NeoX) form over `x[R, D]`, `D = 2·H`, `inv_freq[H]` (row `r`'s
/// absolute position is the row index `r`):
///
/// ```text
/// for r in 0..R {
///     for j in 0..H {
///         let theta = (r as f32) * inv_freq[j];
///         let c = cos(theta);
///         let s = sin(theta);
///         let a = x[r*D + j];
///         let b = x[r*D + j + H];
///         out[r*D + j]     = a*c - b*s;
///         out[r*D + j + H] = b*c + a*s;
///     }
/// }
/// ```
///
/// The angles' `cos`/`sin` are computed **inline** per element, so C/Rust keep `sinf`/`cosf` scalar; the
/// 256-bit kernel computes them 8-wide (the shared `sin8`/`cos8`), the win. `out` may alias `x` (each
/// `(r,j)` touches only columns `j` and `j+H`, written after both are read). Pure data rotation, so the
/// kernel is bit-identical to the scalar nest (no reassociation) — the differential gate is trivial.
fn match_rope(
    pat: &Pattern,
    iter: &ForIter,
    body: &Block,
    backward: bool,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<RopeNest> {
    let ast::PatKind::Ident(rvar) = &pat.kind else {
        return None;
    };
    let rvar = *rvar;
    let (rs, re) = range_bounds(iter)?;
    if as_int_lit(rs, interner)? != 0 {
        return None;
    }
    let rows = as_dim(re, interner)?;
    // The outer body is exactly one inner `for j in 0..H` loop.
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let (jvar, he, inner) = stmt_range0_for(&body.stmts[0], interner)?;
    let half = as_dim(he, interner)?;
    if inner.tail.is_some() || inner.stmts.len() != 7 {
        return None;
    }
    // [0] let theta = (r as f32) * inv_freq[j]
    let (theta, t0) = stmt_let_init(&inner.stmts[0])?;
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: tl,
        rhs: tr,
    } = &t0.kind
    else {
        return None;
    };
    let is_r_cast = |e: &Expr| matches!(&e.kind, ExprKind::Cast { expr, .. } if single_path(expr) == Some(rvar));
    let inv_freq = if is_r_cast(tl) {
        index_by_var(tr, jvar)?
    } else if is_r_cast(tr) {
        index_by_var(tl, jvar)?
    } else {
        return None;
    };
    // [1] let c = cos(theta);  [2] let s = sin(theta)
    let (c, c0) = stmt_let_init(&inner.stmts[1])?;
    if !is_unary_intrinsic_of(c0, MathIntrinsic::Cos, theta, sema, interner) {
        return None;
    }
    let (s, s0) = stmt_let_init(&inner.stmts[2])?;
    if !is_unary_intrinsic_of(s0, MathIntrinsic::Sin, theta, sema, interner) {
        return None;
    }
    // [3] let a = x[r*D + j];  [4] let b = x[r*D + j + H]
    let (a, a0) = stmt_let_init(&inner.stmts[3])?;
    let (x, stride, off_a) = index_strided(a0, rvar, jvar, interner)?;
    if !off_a.is_empty() {
        return None;
    }
    let (bb, b0) = stmt_let_init(&inner.stmts[4])?;
    let (xb, stride_b, off_b) = index_strided(b0, rvar, jvar, interner)?;
    if xb != x || stride_b != stride || !offset_is_half(&off_b, he) {
        return None;
    }
    if !stride_is_twice_half(&stride, &half) {
        return None;
    }
    // [5] forward: out[r*D+j] = a*c - b*s   |   backward: dx[r*D+j] = a*c + b*s
    let (out, v5, stride5, off5) = assign_strided(&inner.stmts[5], rvar, jvar, interner)?;
    if stride5 != stride || !off5.is_empty() {
        return None;
    }
    let ExprKind::Binary {
        op: op5,
        lhs: s5l,
        rhs: s5r,
    } = &v5.kind
    else {
        return None;
    };
    let ok5 = match (backward, op5) {
        // forward `a*c - b*s` — a Sub with the order fixed (a·c minuend, b·s subtrahend).
        (false, ast::BinOp::Sub) => is_prod_of(s5l, a, c) && is_prod_of(s5r, bb, s),
        // backward `a*c + b*s` — an Add, either operand order.
        (true, ast::BinOp::Add) => {
            (is_prod_of(s5l, a, c) && is_prod_of(s5r, bb, s))
                || (is_prod_of(s5l, bb, s) && is_prod_of(s5r, a, c))
        }
        _ => false,
    };
    if !ok5 {
        return None;
    }
    // [6] forward: out[r*D+j+H] = b*c + a*s   |   backward: dx[r*D+j+H] = b*c - a*s
    let (out6, v6, stride6, off6) = assign_strided(&inner.stmts[6], rvar, jvar, interner)?;
    if out6 != out || stride6 != stride || !offset_is_half(&off6, he) {
        return None;
    }
    let ExprKind::Binary {
        op: op6,
        lhs: s6l,
        rhs: s6r,
    } = &v6.kind
    else {
        return None;
    };
    let ok6 = match (backward, op6) {
        // forward `b*c + a*s` — an Add, either operand order.
        (false, ast::BinOp::Add) => {
            (is_prod_of(s6l, bb, c) && is_prod_of(s6r, a, s))
                || (is_prod_of(s6l, a, s) && is_prod_of(s6r, bb, c))
        }
        // backward `b*c - a*s` — a Sub with the order fixed (b·c minuend, a·s subtrahend).
        (true, ast::BinOp::Sub) => is_prod_of(s6l, bb, c) && is_prod_of(s6r, a, s),
        _ => false,
    };
    if !ok6 {
        return None;
    }
    // inv_freq must be a distinct array (not x/out); x and out may alias (in-place RoPE is sound).
    if inv_freq == x || inv_freq == out {
        return None;
    }
    Some(RopeNest {
        x,
        inv_freq,
        out,
        rows,
        half,
        backward,
    })
}

/// Is the whole function body a single RoPE (forward or backward) nest? Intercepts a `@parallel` RoPE
/// before the outliner.
fn rope_fn(body: &Block, sema: &SemaResult, interner: &Interner) -> Option<RopeNest> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    match_rope(pat, iter, lb, false, sema, interner)
        .or_else(|| match_rope(pat, iter, lb, true, sema, interner))
}

/// Is the whole function body a single fused residual projection (`x = x + act(x·Wᵀ + bias)`)? Used to
/// intercept a `@parallel` residual *before* the elementwise outliner (which would split it into
/// per-row scalar loops and lose the fused-epilogue kernel), mirroring the sgemm/norm interceptions.
fn matmul_residual_fn(body: &Block, sema: &SemaResult, interner: &Interner) -> bool {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return false;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return false;
    };
    match_matmul_residual(pat, iter, lb, sema, interner).is_some()
}

/// Lower a recognized int8 matmul function to a thin wrapper that binds its array params to base
/// pointers and tail-calls `mercury_i8gemm_nt`/`mercury_i8gemm_nt_parallel` — the int8 twin of
/// `lower_matmul_fn`.
#[allow(clippy::too_many_arguments)]
fn lower_i8matmul_fn(
    f: &FnDecl,
    nest: &I8MatmulNest,
    parallel: bool,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
    diags: &mut Vec<Diagnostic>,
) -> Function {
    let (param_tys, ret_ty) = match sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
        Some(DefKind::Fn(sig)) => (sig.params.clone(), sig.ret.clone()),
        _ => (f.params.iter().map(|_| Ty::Unknown).collect(), Ty::Unit),
    };
    let ret_mir = mir_ty(&ret_ty);
    let mut fl = FnLowerer {
        builder: Builder::new(f.name.sym, ret_mir.clone()),
        sema,
        interner,
        diags,
        scopes: vec![HashMap::new()],
        terminated: false,
        loops: Vec::new(),
        gemm,
        parallel_fn: false,
        vec_loads: HashMap::new(),
        sret: None,
    };
    let param_vals: Vec<ValueId> = param_tys
        .iter()
        .map(|pty| fl.builder.add_param(param_abi_ty(pty)))
        .collect();
    for ((p, pty), val) in f.params.iter().zip(&param_tys).zip(param_vals) {
        let mty = mir_ty(pty);
        if matches!(mty, MirType::Array(..)) {
            fl.bind(p.name.sym, val, mty);
        } else {
            let slot = fl.builder.alloca(mty.clone());
            fl.builder.build_void(Op::Store {
                ptr: slot,
                value: val,
            });
            fl.bind(p.name.sym, slot, mty);
        }
    }
    fl.emit_i8gemm(nest, parallel);
    if !fl.terminated {
        match ret_mir {
            MirType::Void => fl.builder.ret(None),
            _ => {
                let z = fl.const_zero(ret_mir.clone());
                fl.builder.ret(Some(z));
            }
        }
    }
    fl.builder.finish()
}

/// SIMD width for a lane type: lanes per `VEC_REG_BYTES`-wide register (f32x8, f64x4 at 256-bit).
/// Returns `None` for lane types we don't vectorize.
fn vector_width(lane: &MirType) -> Option<u32> {
    let bytes = match lane {
        MirType::F32 | MirType::I32 => 4,
        MirType::F64 | MirType::I64 => 8,
        _ => return None,
    };
    Some(VEC_REG_BYTES / bytes)
}

/// SIMD register width in bytes. Cranelift's vector ISA is 128-bit (16 bytes); wider types such as
/// `f32x8` are not legalized ("Unexpected SSA-value type"), so we pack one 128-bit register and
/// recover AVX-class throughput via unrolling (see `VEC_UNROLL`) rather than a wider lane type.
const VEC_REG_BYTES: u32 = 16;

/// Vector groups processed per iteration of the unrolled main loop. Independent 128-bit chains
/// issue across the core's multiple FP units (≈ AVX throughput from SSE ops) and hide FP latency in
/// reduction-style bodies (Horner, matmul accumulate). 4×f32x4 = 16 f32/iteration.
const VEC_UNROLL: u32 = 4;

/// Default signedness for a lane type (only affects integer div/rem op selection).
fn lane_signed(lane: &MirType) -> bool {
    lane.is_int()
}

/// The pure tail value of a braces-only block (`{ e }`), or `None` if it has statements.
fn block_value(b: &Block) -> Option<&Expr> {
    if b.stmts.is_empty() {
        b.tail.as_deref()
    } else {
        None
    }
}

/// The value of an `if` branch: a braces-only block yields its tail; a non-block expression (e.g. a
/// chained `else if`) yields itself.
fn branch_value(e: &Expr) -> Option<&Expr> {
    match &e.kind {
        ExprKind::Block(b) => block_value(b),
        _ => Some(e),
    }
}

/// The integer lane type of a vector compare mask: same bit width as the value lane (so the mask
/// reinterprets cleanly for a bitwise blend on the native side).
fn mask_lane_type(lane: &MirType) -> MirType {
    match lane {
        MirType::F64 | MirType::I64 => MirType::I64,
        _ => MirType::I32,
    }
}

/// A `for v in a..b { body }` over a half-open, unit-step range with an identifier binding — the
/// shape eligible for fusion. Returns `(pattern, iter, body)`.
fn fusable_for(s: &Stmt) -> Option<(&Pattern, &ForIter, &Block)> {
    if let StmtKind::For {
        pat, iter, body, ..
    } = &s.kind
    {
        if matches!(&pat.kind, ast::PatKind::Ident(_))
            && matches!(
                iter,
                ForIter::Range {
                    end: Some(_),
                    inclusive: false,
                    step: None,
                    ..
                }
            )
        {
            return Some((pat, iter, body));
        }
    }
    None
}

/// The `(start, end)` expressions of a half-open range iterator.
fn range_bounds(iter: &ForIter) -> Option<(&Expr, &Expr)> {
    match iter {
        ForIter::Range {
            start,
            end: Some(end),
            ..
        } => Some((start, end)),
        _ => None,
    }
}

/// Concatenate the bodies of a run of `for` statements into one block (statements cloned; their
/// `NodeId`s are preserved so sema type lookups still resolve). The wrapper block reuses the first
/// body's id/span, which are not used for typing.
fn fuse_for_bodies(stmts: &[Stmt]) -> Block {
    let mut fused = Vec::new();
    let mut id = None;
    let mut span = None;
    for s in stmts {
        if let StmtKind::For { body, .. } = &s.kind {
            if id.is_none() {
                id = Some(body.id);
                span = Some(body.span);
            }
            fused.extend(body.stmts.iter().cloned());
        }
    }
    Block {
        id: id.unwrap(),
        stmts: fused,
        tail: None,
        span: span.unwrap(),
    }
}

/// The wider of two numeric MIR types (float beats int), used to pick a common comparison type.
/// Non-numeric or equal types yield the left type.
fn numeric_join(a: &MirType, b: &MirType) -> MirType {
    if a == b || !is_numeric(a) || !is_numeric(b) {
        return a.clone();
    }
    let bits = |t: &MirType| match t {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 | MirType::F16 | MirType::BF16 => 16,
        MirType::I32 | MirType::F32 => 32,
        _ => 64,
    };
    match (a.is_float(), b.is_float()) {
        (true, false) => a.clone(),
        (false, true) => b.clone(),
        _ => {
            if bits(a) >= bits(b) {
                a.clone()
            } else {
                b.clone()
            }
        }
    }
}

fn cast_kind(from: &MirType, to: &MirType, signed: bool) -> CastKind {
    let isz = |t: &MirType| match t {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => 0,
    };
    let fsz = |t: &MirType| match t {
        MirType::F16 | MirType::BF16 => 16,
        MirType::F32 => 32,
        MirType::F64 => 64,
        _ => 0,
    };
    match (from.is_int(), to.is_int(), from.is_float(), to.is_float()) {
        (true, true, _, _) => {
            if isz(to) > isz(from) {
                if signed {
                    CastKind::SExt
                } else {
                    CastKind::ZExt
                }
            } else {
                CastKind::Trunc
            }
        }
        (true, _, _, true) => {
            if signed {
                CastKind::SiToFp
            } else {
                CastKind::UiToFp
            }
        }
        (_, true, true, _) => {
            if signed {
                CastKind::FpToSi
            } else {
                CastKind::FpToUi
            }
        }
        (_, _, true, true) => {
            if fsz(to) > fsz(from) {
                CastKind::FpExt
            } else {
                CastKind::FpTrunc
            }
        }
        _ if matches!(from, MirType::Ptr) && to.is_int() => CastKind::PtrToInt,
        _ if from.is_int() && matches!(to, MirType::Ptr) => CastKind::IntToPtr,
        _ => CastKind::Bitcast,
    }
}

/// How a recognised float/int reduction folds its lanes: a sum, or a running `fmax`/`fmin`.
/// `Add` covers `s += x` / `s = s + x` (float reassociated, int exact); `Fmax`/`Fmin` cover
/// `m = fmax(m, x)` / `m = fmin(m, x)` (float only — exactly associative for finite, non-NaN
/// inputs, so the vectorized fold is bit-identical to the scalar one).
#[derive(Clone, Copy, PartialEq, Eq)]
enum RedOp {
    Add,
    Fmax,
    Fmin,
}

/// The math builtins lowered directly to primitive MIR ops (not runtime calls).
#[derive(Clone, Copy, PartialEq, Eq)]
enum MathIntrinsic {
    Sqrt,
    Rsqrt,
    Abs,
    Round,
    Floor,
    Ceil,
    Trunc,
    Exp,
    Log,
    Exp2,
    Log2,
    Exp10,
    Log10,
    Softsign,
    LogSigmoid,
    Tan,
    Asin,
    Acos,
    Atan2,
    Hypot,
    Cbrt,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
    Atan,
    Expm1,
    Log1p,
    Pow,
    Erf,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Silu,
    Gelu,
    /// Activation backward (training gradient `dy · act'(x)`), two args `(x, dy)`. The 256-bit
    /// `mercury_vmath2_f32` dispatch; the derivative folds a transcendental (sigmoid/tanh) C/Rust
    /// keep scalar.
    SiluBackward,
    GeluBackward,
    SigmoidBackward,
    TanhBackward,
    EluBackward,
    SoftplusBackward,
    Elu,
    LeakyRelu,
    Softplus,
    Mish,
    Selu,
    Tanhshrink,
    HardSigmoid,
    HardSwish,
    Fmax,
    Fmin,
}

/// The `RoundMode` for a rounding intrinsic (`Round`→Nearest, else the matching mode). Panics on a
/// non-rounding intrinsic — callers gate on the four rounding variants.
fn round_mode(op: MathIntrinsic) -> RoundMode {
    match op {
        MathIntrinsic::Round => RoundMode::Nearest,
        MathIntrinsic::Floor => RoundMode::Floor,
        MathIntrinsic::Ceil => RoundMode::Ceil,
        MathIntrinsic::Trunc => RoundMode::Trunc,
        _ => unreachable!("round_mode on non-rounding intrinsic"),
    }
}

fn math_intrinsic(name: &str) -> Option<MathIntrinsic> {
    Some(match name {
        "sqrt" => MathIntrinsic::Sqrt,
        "rsqrt" => MathIntrinsic::Rsqrt,
        "abs" => MathIntrinsic::Abs,
        "round" => MathIntrinsic::Round,
        "floor" => MathIntrinsic::Floor,
        "ceil" => MathIntrinsic::Ceil,
        "trunc" => MathIntrinsic::Trunc,
        "exp" => MathIntrinsic::Exp,
        "log" => MathIntrinsic::Log,
        "exp2" => MathIntrinsic::Exp2,
        "log2" => MathIntrinsic::Log2,
        "exp10" => MathIntrinsic::Exp10,
        "log10" => MathIntrinsic::Log10,
        "softsign" => MathIntrinsic::Softsign,
        "logsigmoid" => MathIntrinsic::LogSigmoid,
        "tan" => MathIntrinsic::Tan,
        "asin" => MathIntrinsic::Asin,
        "acos" => MathIntrinsic::Acos,
        "atan2" => MathIntrinsic::Atan2,
        "hypot" => MathIntrinsic::Hypot,
        "cbrt" => MathIntrinsic::Cbrt,
        "sinh" => MathIntrinsic::Sinh,
        "cosh" => MathIntrinsic::Cosh,
        "asinh" => MathIntrinsic::Asinh,
        "acosh" => MathIntrinsic::Acosh,
        "atanh" => MathIntrinsic::Atanh,
        "atan" => MathIntrinsic::Atan,
        "expm1" => MathIntrinsic::Expm1,
        "log1p" => MathIntrinsic::Log1p,
        "pow" => MathIntrinsic::Pow,
        "erf" => MathIntrinsic::Erf,
        "sin" => MathIntrinsic::Sin,
        "cos" => MathIntrinsic::Cos,
        "tanh" => MathIntrinsic::Tanh,
        "sigmoid" => MathIntrinsic::Sigmoid,
        "silu" => MathIntrinsic::Silu,
        "gelu" => MathIntrinsic::Gelu,
        "silu_backward" => MathIntrinsic::SiluBackward,
        "gelu_backward" => MathIntrinsic::GeluBackward,
        "sigmoid_backward" => MathIntrinsic::SigmoidBackward,
        "tanh_backward" => MathIntrinsic::TanhBackward,
        "elu_backward" => MathIntrinsic::EluBackward,
        "softplus_backward" => MathIntrinsic::SoftplusBackward,
        "elu" => MathIntrinsic::Elu,
        "leaky_relu" => MathIntrinsic::LeakyRelu,
        "softplus" => MathIntrinsic::Softplus,
        "mish" => MathIntrinsic::Mish,
        "selu" => MathIntrinsic::Selu,
        "tanhshrink" => MathIntrinsic::Tanhshrink,
        "hardsigmoid" => MathIntrinsic::HardSigmoid,
        "hardswish" => MathIntrinsic::HardSwish,
        "fmax" => MathIntrinsic::Fmax,
        "fmin" => MathIntrinsic::Fmin,
        _ => return None,
    })
}

/// The same shape as `ty` (scalar or `Vec`) but with float lane `lane`.
fn float_ty_like(ty: &MirType, lane: MirType) -> MirType {
    match ty {
        MirType::Vec(_, n) => MirType::Vec(Box::new(lane), *n),
        _ => lane,
    }
}

/// The boolean-mask type a compare on `ty` yields: `i1` for a scalar, or a per-lane integer vector
/// (matching the native backend's vector compare result) for a `Vec`.
fn mask_ty(ty: &MirType) -> MirType {
    match ty {
        MirType::Vec(lane, n) => MirType::Vec(Box::new(mask_lane_type(lane)), *n),
        _ => MirType::I1,
    }
}

// `exp` polynomial constants (Cephes single-precision `expf`), evaluated in f32 by both backends.
const LOG2EF: f64 = std::f64::consts::LOG2_E;
const EXP_MAGIC: f64 = 12582912.0; // 1.5 * 2^23 — round-to-nearest via add then sub
const EXP_C1: f64 = 0.693359375; // ln2, high part
const EXP_C2: f64 = -2.1219444e-4; // ln2, low correction
const EXP_HI: f64 = 88.3762626647949;
const EXP_LO: f64 = -88.3762626647949;
const EXP_P: [f64; 6] = [
    1.98756915e-4,
    1.3981999507e-3,
    8.3334519073e-3,
    4.1665795894e-2,
    1.6666665459e-1,
    5.0000001201e-1,
];

/// Natural-log polynomial constants (Cephes `logf`). `LOG_SQRTHF` is the `√0.5` split point that
/// keeps the reduced mantissa centered; `LOG_P` is the degree-8 minimax poly on it. `e·ln2` is
/// added back with the *same* split `EXP_C1`/`EXP_C2` that `exp` uses (ln2 = C1 + C2).
const LOG_SQRTHF: f64 = std::f64::consts::FRAC_1_SQRT_2; // 1/√2 = √0.5
const INV_2P23: f64 = 1.0 / 8_388_608.0; // 2^-23 (exact): scales the masked exponent field to a count
const LOG_P: [f64; 9] = [
    7.0376836292e-2,
    -1.1514610310e-1,
    1.1676998740e-1,
    -1.2420140846e-1,
    1.4249322787e-1,
    -1.6668057665e-1,
    2.0000714765e-1,
    -2.4999993993e-1,
    3.3333331174e-1,
];

// `erf` constants (Abramowitz–Stegun 7.1.26): erf(|x|) = 1 - (a₁t + a₂t² + … + a₅t⁵)·e^(-x²),
// t = 1/(1 + P·|x|). Max error ~1.5e-7 — f32-grade — and bit-identical across backends since it is
// built from primitive ops plus `exp`. Enables exact (erf-based) GELU, the original BERT/GPT-2 form.
const ERF_P: f64 = 0.327_591_1;
const ERF_A: [f64; 5] = [
    0.254_829_592,
    -0.284_496_736,
    1.421_413_741,
    -1.453_152_027,
    1.061_405_429,
];

// `sin`/`cos` constants (Cephes single-precision `sinf`/`cosf`). Reduce `x` to `r ∈ [-π/4, π/4]` by
// `q = round(x·2/π)` quadrants, then `r = x - q·(π/2)` with π/2 split into three parts (`PIO2_*`, the
// Cephes π/4 `DP` constants doubled) so the cancellation stays accurate. `q mod 4` picks ±sin/±cos of
// the reduced angle. All primitive ops + the proven round-to-nearest magic, so both backends agree.
const TWO_OVER_PI: f64 = std::f64::consts::FRAC_2_PI; // 2/π — quadrant count = round(x·2/π)
const PIO2_1: f64 = 1.5703125; // π/2 high (2 × Cephes DP1 = 2 × 0.78515625)
const PIO2_2: f64 = 4.837_512_969_970_703e-4; // π/2 mid  (2 × DP2)
const PIO2_3: f64 = 7.549_789_954_891_88e-8; // π/2 low  (2 × DP3)
const SIN_P: [f64; 3] = [-1.9515295891e-4, 8.3321608736e-3, -1.6666654611e-1];
const COS_P: [f64; 3] = [
    2.443_315_711_809_948e-5,
    -1.388_731_625_493_765e-3,
    4.166_664_568_298_827e-2,
];
// atan (Cephes): 3-region reduction breakpoints + the π/4·π/2 offsets + the degree-3 odd minimax poly
// (mirror `mercury_runtime::vmath`'s ATAN_* so the inlined form equals the dispatched kernel).
const ATAN_TAN_3PI8: f64 = 2.414213562373095; // tan(3π/8) = 1 + √2
const ATAN_TAN_PI8: f64 = 0.4142135623730950; // tan(π/8) = √2 − 1
const ATAN_PIO2: f64 = std::f64::consts::FRAC_PI_2;
const ATAN_PIO4: f64 = std::f64::consts::FRAC_PI_4;
const ATAN_P: [f64; 4] = [
    0.080_537_444_953_8,
    -0.138_776_856_032,
    0.199_777_106_478,
    -0.333_329_491_539,
];

/// Names that lower to runtime/interpreter intrinsics rather than user functions.
pub fn is_intrinsic(name: &str) -> bool {
    matches!(name, "print" | "println" | "assert")
}

/// Parse an integer literal's source text to its value. Handles the radix prefixes `0x`/`0o`/`0b`
/// (case-insensitive), digit separators `_`, an explicit integer type suffix (`10i64`, `250u8`,
/// `5usize`), and an optional leading sign. The previous version kept only the leading run of
/// decimal digits, so every non-decimal literal (`0xFF`, `0b1010`, `0o17`) silently parsed to `0`.
fn parse_int(text: &str) -> i128 {
    let mut s = text.trim();
    let neg = s.starts_with('-');
    if neg || s.starts_with('+') {
        s = &s[1..];
    }
    // Strip an integer type suffix (longest-first so `usize`/`isize` win over a shorter prefix). A
    // suffix uses letters `i`/`u`/`s`/`z`/`n` that are never hex digits, so this can't truncate a
    // hex literal's digits.
    for suf in [
        "usize", "isize", "u128", "i128", "u64", "i64", "u32", "i32", "u16", "i16", "u8", "i8",
    ] {
        if let Some(stripped) = s.strip_suffix(suf) {
            s = stripped;
            break;
        }
    }
    let body = s.replace('_', "");
    let val = if let Some(h) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        i128::from_str_radix(h, 16)
    } else if let Some(o) = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")) {
        i128::from_str_radix(o, 8)
    } else if let Some(b) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
        i128::from_str_radix(b, 2)
    } else {
        body.parse::<i128>()
    }
    .unwrap_or(0);
    if neg {
        -val
    } else {
        val
    }
}

fn parse_float(text: &str) -> f64 {
    // Strip a trailing type suffix (bf16/f16/f32/f64) before parsing.
    let mut core = text;
    for suf in ["bf16", "f16", "f32", "f64"] {
        if let Some(stripped) = core.strip_suffix(suf) {
            core = stripped;
            break;
        }
    }
    core.parse().unwrap_or(0.0)
}

/// Decode a char literal's raw source text (including the surrounding single quotes) into its
/// Unicode scalar value. Handles the one-character escapes (`\n` `\r` `\t` `\\` `\'` `\"` `\0`),
/// `\xHH`, and `\u{…}`. Returns 0 for an empty/malformed literal (the lexer already reported any
/// lexical error). This is the value a `'c'` literal lowers to (sema types it `u32`).
fn decode_char_literal(text: &str) -> u32 {
    let inner = text
        .strip_prefix('\'')
        .and_then(|t| t.strip_suffix('\''))
        .unwrap_or(text);
    let mut chars = inner.chars();
    match chars.next() {
        Some('\\') => decode_escape(&mut chars),
        Some(c) => c as u32,
        None => 0,
    }
}

/// Decode a string literal's raw source text (including the surrounding `"`) into its UTF-8 bytes,
/// resolving the same escapes as a char literal (`\n` `\t` `\\` `\"` `\0`, `\xHH`, `\u{…}`); each
/// decoded code point is re-encoded as UTF-8. The caller appends the NUL terminator. Used by the
/// `ExprKind::Str` lowering to materialize the byte buffer.
fn decode_string_literal(text: &str) -> Vec<u8> {
    let inner = text
        .strip_prefix('"')
        .and_then(|t| t.strip_suffix('"'))
        .unwrap_or(text);
    let mut out = Vec::new();
    let mut buf = [0u8; 4];
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        let cp = if c == '\\' {
            decode_escape(&mut chars)
        } else {
            c as u32
        };
        match char::from_u32(cp) {
            Some(ch) => out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes()),
            None => out.push(cp as u8),
        }
    }
    out
}

/// Decode the body of a backslash escape (the `\` already consumed) to a code point.
fn decode_escape(chars: &mut std::str::Chars) -> u32 {
    match chars.next() {
        Some('n') => '\n' as u32,
        Some('r') => '\r' as u32,
        Some('t') => '\t' as u32,
        Some('\\') => '\\' as u32,
        Some('\'') => '\'' as u32,
        Some('"') => '"' as u32,
        Some('0') => 0,
        // `\xHH` — up to two hex digits (the lexer consumed at most two).
        Some('x') => chars.by_ref().take(2).fold(0u32, |v, c| {
            c.to_digit(16).map_or(v, |d| v * 16 + d)
        }),
        // `\u{HHHH}` — the hex digits between the braces. Saturating, so an over-long escape
        // (`\u{100000000}`, ≥ 9 hex digits) clamps to an invalid code point instead of overflowing
        // the accumulator and panicking the compiler; a valid code point is ≤ 6 hex digits anyway.
        Some('u') => chars
            .by_ref()
            .skip_while(|&c| c != '{')
            .skip(1)
            .take_while(|&c| c != '}')
            .fold(0u32, |v, c| {
                c.to_digit(16)
                    .map_or(v, |d| v.saturating_mul(16).saturating_add(d))
            }),
        Some(c) => c as u32,
        None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mercury_mir::verify::verify_function;
    use mercury_span::SourceId;

    fn lower(src: &str) -> (Program, Vec<Diagnostic>, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = mercury_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (prog, diags) = lower_program(&module, &sema, &mut interner);
        (prog, diags, interner)
    }

    #[test]
    fn lowers_and_verifies_loop_sum() {
        let src = "fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
                   while i < 10 { s += i; i += 1; } return s; }";
        let (prog, diags, _) = lower(src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        for f in &prog.funcs {
            let errs = verify_function(f);
            assert!(errs.is_empty(), "verify: {errs:?}");
        }
    }

    #[test]
    fn lowers_and_verifies_array_ops() {
        // Array literal init, indexed store, and indexed load must lower to a verifiable function
        // with an array alloca.
        let src = "fn main() -> i32 { let mut xs: [i32; 3] = [1, 2, 3]; \
                   xs[1] = 9; return xs[1]; }";
        let (prog, diags, _) = lower(src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        let main = &prog.funcs[0];
        assert!(
            main.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(&i.op, Op::Alloca(MirType::Array(_, 3)))),
            "expected an array alloca"
        );
        assert!(verify_function(main).is_empty(), "verify failed");
    }

    #[test]
    fn multi_dim_tensor_index_lowers_to_row_major_gep() {
        // The shape-typed surface: `t[i, j]` on `Tensor[f32, M, N]` must lower (no `unsupported`
        // C0001) to a flat row-major offset `i*N + j`, and the function must verify.
        let src = "fn k(a: Tensor[f32, 3, 4], out: Tensor[f32, 3, 4]) { \
                   for i in 0..3 { for j in 0..4 { out[i, j] = a[i, j] * 2.0; } } }";
        let (prog, diags, mut interner) = lower(src);
        assert!(
            diags.iter().all(|d| !d.is_error()),
            "multi-dim index should lower cleanly: {diags:?}"
        );
        let k_sym = interner.intern("k");
        let k = prog.funcs.iter().find(|f| f.name == k_sym).expect("fn k");
        assert!(
            verify_function(k).is_empty(),
            "verify: {:?}",
            verify_function(k)
        );
        // The inner stride is the trailing dim N=4: expect a `* 4` in the offset arithmetic.
        assert!(
            k.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(&i.op, Op::ConstInt(4, MirType::I64))),
            "expected a stride-4 (trailing dim) constant in the flat-index arithmetic"
        );
    }

    #[test]
    fn array_parameter_passes_by_pointer() {
        // An array parameter has ABI type Ptr (passed by base pointer), not an array alloca.
        let src = "fn dot(x: [i32; 2], y: [i32; 2]) -> i32 { return x[0]*y[0] + x[1]*y[1]; } \
                   fn main() -> i32 { let a: [i32;2] = [1,2]; let b: [i32;2] = [3,4]; \
                   return dot(a, b); }";
        let (prog, diags, interner) = lower(src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        let dot = prog
            .funcs
            .iter()
            .find(|f| interner.resolve(f.name) == "dot")
            .unwrap();
        // Both params are pointers; no array alloca inside `dot`.
        for &p in &dot.params {
            assert_eq!(dot.value_type(p), &MirType::Ptr);
        }
        assert!(
            !dot.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(&i.op, Op::Alloca(MirType::Array(..)))),
            "array params must not alloca array storage in the callee"
        );
        assert!(verify_function(dot).is_empty());
    }

    #[test]
    fn lowers_and_verifies_recursive_fib() {
        let src = "fn fib(n: i32) -> i32 { if n < 2 { return n; } \
                   return fib(n - 1) + fib(n - 2); } \
                   fn main() -> i32 { return fib(10); }";
        let (prog, _diags, _) = lower(src);
        assert_eq!(prog.funcs.len(), 2);
        for f in &prog.funcs {
            assert!(
                verify_function(f).is_empty(),
                "verify failed for a function"
            );
        }
    }
}
