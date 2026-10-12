//! The interpreter's runtime-intrinsic table: every callee name the interpreter implements itself,
//! listed ONCE, each with a dense [`Intrinsic`] id.
//!
//! A call used to reach its intrinsic through a string `match` over ~170 literals in
//! `Interp::intrinsic`, then — for anything that match did not claim — a second, third and fourth
//! literal scan in `text_io.rs`, `ml_io.rs` and `decode.rs`, on every call: a recognized kernel
//! inside a loop paid the whole compare chain each iteration, and a decode builtin paid all four.
//! Now `CalleeTable` (in `lib.rs`) resolves a callee symbol to `Callee::Intrinsic(id)` the first
//! time a run calls it — the same resolve-once-per-run the table already did for Wukong functions
//! and `extern "C"` declarations — and every dispatcher matches on the id.
//!
//! The table is the single authority for names: `Intrinsic::name` and `Intrinsic::from_name`
//! are generated from it, and the unit tests below pin (a) that every name the string dispatch
//! accepted at main@1934a82e still resolves, (b) that every id reaches an implementing arm, and
//! (c) that a name outside the table still fails with the old error text. Adding an intrinsic is
//! one line here plus its arm in the module that implements it.

/// Generates [`Intrinsic`], its name table and the reverse lookup from one list.
macro_rules! intrinsic_table {
    ($($variant:ident = $name:literal,)*) => {
        /// A runtime intrinsic the interpreter implements, named by the callee symbol the compiler
        /// emits for it ([`Intrinsic::name`]).
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub(crate) enum Intrinsic {
            $($variant,)*
        }

        impl Intrinsic {
            /// Every intrinsic, in table order.
            #[cfg(test)]
            pub(crate) const ALL: &'static [Intrinsic] = &[$(Intrinsic::$variant,)*];

            /// The callee name the compiler emits for this intrinsic.
            pub(crate) const fn name(self) -> &'static str {
                match self {
                    $(Intrinsic::$variant => $name,)*
                }
            }

            /// The intrinsic `name` denotes, or `None` when the interpreter implements no such
            /// intrinsic. Called once per callee symbol per run, never per call.
            pub(crate) fn from_name(name: &str) -> Option<Intrinsic> {
                match name {
                    $($name => Some(Intrinsic::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

intrinsic_table! {
    // Printing and assertions (`Interp::call_intrinsic`).
    Print = "print",
    Println = "println",
    PrintU = "print_u",
    PrintlnU = "println_u",
    PrintBool = "print_bool",
    PrintlnBool = "println_bool",
    PrintChar = "print_char",
    PrintlnChar = "println_char",
    PrintStr = "print_str",
    PrintlnStr = "println_str",
    Assert = "assert",
    RtBoundsFail = "wukong_rt_bounds_fail",

    // Heap and scratch allocation, the clock (`Interp::call_intrinsic`).
    RtAlloc = "wukong_rt_alloc",
    RtFree = "wukong_rt_free",
    RtScratchAlloc = "wukong_rt_scratch_alloc",
    RtScratchMark = "wukong_rt_scratch_mark",
    RtScratchRelease = "wukong_rt_scratch_release",
    NowNs = "wukong_now_ns",

    // Raw file I/O (`Interp::call_intrinsic`).
    RtReadF32 = "wukong_rt_read_f32",
    RtReadI32 = "wukong_rt_read_i32",
    RtReadI64 = "wukong_rt_read_i64",
    RtReadU8 = "wukong_rt_read_u8",
    RtReadF64 = "wukong_rt_read_f64",
    RtReadI8 = "wukong_rt_read_i8",
    RtWriteF32 = "wukong_rt_write_f32",
    RtWriteI32 = "wukong_rt_write_i32",
    RtWriteI64 = "wukong_rt_write_i64",
    RtWriteU8 = "wukong_rt_write_u8",
    RtWriteF64 = "wukong_rt_write_f64",
    RtWriteI8 = "wukong_rt_write_i8",

    // Recognized kernels and the scalar math ABI (`Interp::call_intrinsic`).
    Sgemm = "wukong_sgemm",
    SgemmParallel = "wukong_sgemm_parallel",
    SgemmNt = "wukong_sgemm_nt",
    SgemmNtParallel = "wukong_sgemm_nt_parallel",
    SgemmTn = "wukong_sgemm_tn",
    SgemmTnParallel = "wukong_sgemm_tn_parallel",
    Sgemv = "wukong_sgemv",
    SgemvParallel = "wukong_sgemv_parallel",
    SgemvAlpha = "wukong_sgemv_alpha",
    SgemvAlphaParallel = "wukong_sgemv_alpha_parallel",
    SgevmF32 = "wukong_sgevm_f32",
    SgevmF32Parallel = "wukong_sgevm_f32_parallel",
    I8gemmNt = "wukong_i8gemm_nt",
    I8gemmNtParallel = "wukong_i8gemm_nt_parallel",
    I8gemmNtDeq = "wukong_i8gemm_nt_deq",
    I8gemmNtDeqParallel = "wukong_i8gemm_nt_deq_parallel",
    DequantF32 = "wukong_dequant_f32",
    DequantF32Parallel = "wukong_dequant_f32_parallel",
    DequantPerchanF32 = "wukong_dequant_perchan_f32",
    DequantPerchanF32Parallel = "wukong_dequant_perchan_f32_parallel",
    SgemmNtEpi = "wukong_sgemm_nt_epi",
    SgemmNtEpiParallel = "wukong_sgemm_nt_epi_parallel",
    SgemmNtAlpha = "wukong_sgemm_nt_alpha",
    SgemmNtAlphaParallel = "wukong_sgemm_nt_alpha_parallel",
    VmathF32 = "wukong_vmath_f32",
    VmathF32Parallel = "wukong_vmath_f32_parallel",
    TransposeF32 = "wukong_transpose_f32",
    TransposeF32Parallel = "wukong_transpose_f32_parallel",
    Maxpool2dF32 = "wukong_maxpool2d_f32",
    Maxpool2dF32Parallel = "wukong_maxpool2d_f32_parallel",
    Avgpool2dF32 = "wukong_avgpool2d_f32",
    Avgpool2dF32Parallel = "wukong_avgpool2d_f32_parallel",
    TransposeU16 = "wukong_transpose_u16",
    TransposeU16Parallel = "wukong_transpose_u16_parallel",
    ColsumF32 = "wukong_colsum_f32",
    ColsumF32Parallel = "wukong_colsum_f32_parallel",
    ColmaxF32 = "wukong_colmax_f32",
    ColmaxF32Parallel = "wukong_colmax_f32_parallel",
    ColminF32 = "wukong_colmin_f32",
    ColminF32Parallel = "wukong_colmin_f32_parallel",
    ColmaxabsF32 = "wukong_colmaxabs_f32",
    ColmaxabsF32Parallel = "wukong_colmaxabs_f32_parallel",
    ColmeanF32 = "wukong_colmean_f32",
    ColmeanF32Parallel = "wukong_colmean_f32_parallel",
    ColsumsqF32 = "wukong_colsumsq_f32",
    ColsumsqF32Parallel = "wukong_colsumsq_f32_parallel",
    Coll2F32 = "wukong_coll2_f32",
    Coll2F32Parallel = "wukong_coll2_f32_parallel",
    ColrmsF32 = "wukong_colrms_f32",
    ColrmsF32Parallel = "wukong_colrms_f32_parallel",
    RtTrigF32 = "wukong_rt_trig_f32",
    RtMathF64 = "wukong_rt_math_f64",
    RtMath2F64 = "wukong_rt_math2_f64",
    Vmath2F32 = "wukong_vmath2_f32",
    SoftmaxBwdF32 = "wukong_softmax_bwd_f32",
    SoftmaxBwdF32Parallel = "wukong_softmax_bwd_f32_parallel",
    LrscanF32 = "wukong_lrscan_f32",
    LrscanF32Parallel = "wukong_lrscan_f32_parallel",
    RmsnormBwdF32 = "wukong_rmsnorm_bwd_f32",
    RmsnormBwdF32Parallel = "wukong_rmsnorm_bwd_f32_parallel",
    LayernormBwdF32 = "wukong_layernorm_bwd_f32",
    LayernormBwdF32Parallel = "wukong_layernorm_bwd_f32_parallel",
    XentFwdF32 = "wukong_xent_fwd_f32",
    XentFwdF32Parallel = "wukong_xent_fwd_f32_parallel",
    XentBwdF32 = "wukong_xent_bwd_f32",
    XentBwdF32Parallel = "wukong_xent_bwd_f32_parallel",
    RopeF32 = "wukong_rope_f32",
    RopeF32Parallel = "wukong_rope_f32_parallel",
    RopeBwdF32 = "wukong_rope_bwd_f32",
    RopeBwdF32Parallel = "wukong_rope_bwd_f32_parallel",
    LogsumexpF32 = "wukong_logsumexp_f32",
    LogsumexpF32Parallel = "wukong_logsumexp_f32_parallel",
    EntropyF32 = "wukong_entropy_f32",
    EntropyF32Parallel = "wukong_entropy_f32_parallel",
    RowargmaxI32 = "wukong_rowargmax_i32",
    RowargmaxI32Parallel = "wukong_rowargmax_i32_parallel",
    RowargminI32 = "wukong_rowargmin_i32",
    RowargminI32Parallel = "wukong_rowargmin_i32_parallel",
    ColargmaxI32 = "wukong_colargmax_i32",
    ColargmaxI32Parallel = "wukong_colargmax_i32_parallel",
    ColargminI32 = "wukong_colargmin_i32",
    ColargminI32Parallel = "wukong_colargmin_i32_parallel",
    CumsumF32 = "wukong_cumsum_f32",
    CumsumF32Parallel = "wukong_cumsum_f32_parallel",
    CummaxF32 = "wukong_cummax_f32",
    CummaxF32Parallel = "wukong_cummax_f32_parallel",
    CumminF32 = "wukong_cummin_f32",
    CumminF32Parallel = "wukong_cummin_f32_parallel",
    CumprodF32 = "wukong_cumprod_f32",
    CumprodF32Parallel = "wukong_cumprod_f32_parallel",
    KldivF32 = "wukong_kldiv_f32",
    KldivF32Parallel = "wukong_kldiv_f32_parallel",
    KdLossF32 = "wukong_kd_loss_f32",
    KdLossF32Parallel = "wukong_kd_loss_f32_parallel",
    VmathBf16 = "wukong_vmath_bf16",
    VmathF16 = "wukong_vmath_f16",
    VmathBf16Out = "wukong_vmath_bf16_out",
    VmathF16Out = "wukong_vmath_f16_out",
    VelemF32 = "wukong_velem_f32",
    VelemF32Parallel = "wukong_velem_f32_parallel",
    BiasBcastF32 = "wukong_bias_bcast_f32",
    BiasBcastF32Parallel = "wukong_bias_bcast_f32_parallel",
    VhornerF32 = "wukong_vhorner_f32",
    SreduceF32 = "wukong_sreduce_f32",
    SreduceF32Parallel = "wukong_sreduce_f32_parallel",
    ArgreduceF32 = "wukong_argreduce_f32",
    ArgreduceF32Parallel = "wukong_argreduce_f32_parallel",
    DotBf16 = "wukong_dot_bf16",
    SumBf16 = "wukong_sum_bf16",
    DotBf16Parallel = "wukong_dot_bf16_parallel",
    SumBf16Parallel = "wukong_sum_bf16_parallel",
    ReduceBf16 = "wukong_reduce_bf16",
    ReduceBf16Parallel = "wukong_reduce_bf16_parallel",
    DotF16 = "wukong_dot_f16",
    SumF16 = "wukong_sum_f16",
    DotF16Parallel = "wukong_dot_f16_parallel",
    SumF16Parallel = "wukong_sum_f16_parallel",
    ReduceF16 = "wukong_reduce_f16",
    ReduceF16Parallel = "wukong_reduce_f16_parallel",
    AxpbyBf16 = "wukong_axpby_bf16",
    AxpbyF16 = "wukong_axpby_f16",
    AxpbyBf16Out = "wukong_axpby_bf16_out",
    AxpbyF16Out = "wukong_axpby_f16_out",
    SgemmBf16Nt = "wukong_sgemm_bf16_nt",
    SgemmBf16NtParallel = "wukong_sgemm_bf16_nt_parallel",
    SgemmF16Nt = "wukong_sgemm_f16_nt",
    SgemmF16NtParallel = "wukong_sgemm_f16_nt_parallel",
    SgemmBf16Tn = "wukong_sgemm_bf16_tn",
    SgemmBf16TnParallel = "wukong_sgemm_bf16_tn_parallel",
    SgemmF16Tn = "wukong_sgemm_f16_tn",
    SgemmF16TnParallel = "wukong_sgemm_f16_tn_parallel",
    SgemmBf16NtEpi = "wukong_sgemm_bf16_nt_epi",
    SgemmBf16NtEpiParallel = "wukong_sgemm_bf16_nt_epi_parallel",
    SgemmF16NtEpi = "wukong_sgemm_f16_nt_epi",
    SgemmF16NtEpiParallel = "wukong_sgemm_f16_nt_epi_parallel",
    NormF32 = "wukong_norm_f32",
    NormF32Parallel = "wukong_norm_f32_parallel",
    NormAffineF32 = "wukong_norm_affine_f32",
    NormAffineF32Parallel = "wukong_norm_affine_f32_parallel",
    EmbeddingF32 = "wukong_embedding_f32",
    EmbeddingF32Parallel = "wukong_embedding_f32_parallel",
    ScatterAddF32 = "wukong_scatter_add_f32",
    ScatterAddF32Parallel = "wukong_scatter_add_f32_parallel",
    AttentionF32 = "wukong_attention_f32",

    // The frame loop runs a `wukong_parallel_for` body as a call (`Interp::dispatch_call`).
    ParallelFor = "wukong_parallel_for",

    // Text formatting and the process interface (`text_io.rs`).
    RtFmtInt = "wukong_rt_fmt_int",
    RtFmtF64 = "wukong_rt_fmt_f64",
    RtFmtF32 = "wukong_rt_fmt_f32",
    RtFmtStr = "wukong_rt_fmt_str",
    RtFmtLit = "wukong_rt_fmt_lit",
    RtFmtSlice = "wukong_rt_fmt_slice",
    RtArgCount = "wukong_rt_arg_count",
    RtArg = "wukong_rt_arg",
    RtArgI64 = "wukong_rt_arg_i64",
    RtArgF32 = "wukong_rt_arg_f32",
    RtEnv = "wukong_rt_env",
    RtReadStdin = "wukong_rt_read_stdin",
    RtExit = "wukong_rt_exit",

    // ML I/O, sampling and sorting builtins (`ml_io.rs`).
    RtReadDt = "wukong_rt_read_dt",
    RtWriteDt = "wukong_rt_write_dt",
    RtFileSize = "wukong_rt_file_size",
    RtStRead = "wukong_rt_st_read",
    RtNpyRead = "wukong_rt_npy_read",
    RtGgufRead = "wukong_rt_gguf_read",
    RtGgufQuery = "wukong_rt_gguf_query",
    RtMap = "wukong_rt_map",
    RandFillF32 = "wukong_rand_fill_f32",
    TopkF32 = "wukong_topk_f32",
    ArgsortF32 = "wukong_argsort_f32",
    SampleCategoricalF32 = "wukong_sample_categorical_f32",

    // Decode builtins: KV-cache attention, half and quantized GEMV (`decode.rs`).
    AttentionKvF32 = "wukong_attention_kv_f32",
    AttentionKvF32Parallel = "wukong_attention_kv_f32_parallel",
    KvAppendF32 = "wukong_kv_append_f32",
    SgemvBf16 = "wukong_sgemv_bf16",
    SgemvBf16Parallel = "wukong_sgemv_bf16_parallel",
    SgemvF16 = "wukong_sgemv_f16",
    SgemvF16Parallel = "wukong_sgemv_f16_parallel",
    QuantizeQ80F32 = "wukong_quantize_q8_0_f32",
    QuantizeQ40F32 = "wukong_quantize_q4_0_f32",
    GemvQ80F32 = "wukong_gemv_q8_0_f32",
    GemvQ80F32Parallel = "wukong_gemv_q8_0_f32_parallel",
    GemvQ40F32 = "wukong_gemv_q4_0_f32",
    GemvQ40F32Parallel = "wukong_gemv_q4_0_f32_parallel",
}

#[cfg(test)]
mod tests {
    use super::Intrinsic;
    use crate::{HashMap, Interner, Interp, Program, ScratchSlots, SlotReuse, Value};

    /// Every name the string dispatch accepted at main@1934a82e — the `match name` arms of
    /// `Interp::intrinsic`, `text_intrinsic`, `ml_intrinsic` and `decode_intrinsic`, extracted from
    /// that source. Frozen on purpose: a name the table drops fails here, while a name the table
    /// adds needs no edit.
    const STRING_DISPATCH_NAMES: &[&str] = &[
        "print",
        "println",
        "print_u",
        "println_u",
        "print_bool",
        "println_bool",
        "print_char",
        "println_char",
        "print_str",
        "println_str",
        "assert",
        "wukong_rt_bounds_fail",
        "wukong_rt_alloc",
        "wukong_rt_free",
        "wukong_rt_scratch_alloc",
        "wukong_rt_scratch_mark",
        "wukong_rt_scratch_release",
        "wukong_now_ns",
        "wukong_rt_read_f32",
        "wukong_rt_read_i32",
        "wukong_rt_read_i64",
        "wukong_rt_read_u8",
        "wukong_rt_read_f64",
        "wukong_rt_read_i8",
        "wukong_rt_write_f32",
        "wukong_rt_write_i32",
        "wukong_rt_write_i64",
        "wukong_rt_write_u8",
        "wukong_rt_write_f64",
        "wukong_rt_write_i8",
        "wukong_sgemm",
        "wukong_sgemm_parallel",
        "wukong_sgemm_nt",
        "wukong_sgemm_nt_parallel",
        "wukong_sgemm_tn",
        "wukong_sgemm_tn_parallel",
        "wukong_sgemv",
        "wukong_sgemv_parallel",
        "wukong_sgemv_alpha",
        "wukong_sgemv_alpha_parallel",
        "wukong_sgevm_f32",
        "wukong_sgevm_f32_parallel",
        "wukong_i8gemm_nt",
        "wukong_i8gemm_nt_parallel",
        "wukong_i8gemm_nt_deq",
        "wukong_i8gemm_nt_deq_parallel",
        "wukong_dequant_f32",
        "wukong_dequant_f32_parallel",
        "wukong_dequant_perchan_f32",
        "wukong_dequant_perchan_f32_parallel",
        "wukong_sgemm_nt_epi",
        "wukong_sgemm_nt_epi_parallel",
        "wukong_sgemm_nt_alpha",
        "wukong_sgemm_nt_alpha_parallel",
        "wukong_vmath_f32",
        "wukong_vmath_f32_parallel",
        "wukong_transpose_f32",
        "wukong_transpose_f32_parallel",
        "wukong_maxpool2d_f32",
        "wukong_maxpool2d_f32_parallel",
        "wukong_avgpool2d_f32",
        "wukong_avgpool2d_f32_parallel",
        "wukong_transpose_u16",
        "wukong_transpose_u16_parallel",
        "wukong_colsum_f32",
        "wukong_colsum_f32_parallel",
        "wukong_colmax_f32",
        "wukong_colmax_f32_parallel",
        "wukong_colmin_f32",
        "wukong_colmin_f32_parallel",
        "wukong_colmaxabs_f32",
        "wukong_colmaxabs_f32_parallel",
        "wukong_colmean_f32",
        "wukong_colmean_f32_parallel",
        "wukong_colsumsq_f32",
        "wukong_colsumsq_f32_parallel",
        "wukong_coll2_f32",
        "wukong_coll2_f32_parallel",
        "wukong_colrms_f32",
        "wukong_colrms_f32_parallel",
        "wukong_rt_trig_f32",
        "wukong_rt_math_f64",
        "wukong_rt_math2_f64",
        "wukong_vmath2_f32",
        "wukong_softmax_bwd_f32",
        "wukong_softmax_bwd_f32_parallel",
        "wukong_lrscan_f32",
        "wukong_lrscan_f32_parallel",
        "wukong_rmsnorm_bwd_f32",
        "wukong_rmsnorm_bwd_f32_parallel",
        "wukong_layernorm_bwd_f32",
        "wukong_layernorm_bwd_f32_parallel",
        "wukong_xent_fwd_f32",
        "wukong_xent_fwd_f32_parallel",
        "wukong_xent_bwd_f32",
        "wukong_xent_bwd_f32_parallel",
        "wukong_rope_f32",
        "wukong_rope_f32_parallel",
        "wukong_rope_bwd_f32",
        "wukong_rope_bwd_f32_parallel",
        "wukong_logsumexp_f32",
        "wukong_logsumexp_f32_parallel",
        "wukong_entropy_f32",
        "wukong_entropy_f32_parallel",
        "wukong_rowargmax_i32",
        "wukong_rowargmax_i32_parallel",
        "wukong_rowargmin_i32",
        "wukong_rowargmin_i32_parallel",
        "wukong_colargmax_i32",
        "wukong_colargmax_i32_parallel",
        "wukong_colargmin_i32",
        "wukong_colargmin_i32_parallel",
        "wukong_cumsum_f32",
        "wukong_cumsum_f32_parallel",
        "wukong_cummax_f32",
        "wukong_cummax_f32_parallel",
        "wukong_cummin_f32",
        "wukong_cummin_f32_parallel",
        "wukong_cumprod_f32",
        "wukong_cumprod_f32_parallel",
        "wukong_kldiv_f32",
        "wukong_kldiv_f32_parallel",
        "wukong_kd_loss_f32",
        "wukong_kd_loss_f32_parallel",
        "wukong_vmath_bf16",
        "wukong_vmath_f16",
        "wukong_vmath_bf16_out",
        "wukong_vmath_f16_out",
        "wukong_velem_f32",
        "wukong_velem_f32_parallel",
        "wukong_bias_bcast_f32",
        "wukong_bias_bcast_f32_parallel",
        "wukong_vhorner_f32",
        "wukong_sreduce_f32",
        "wukong_sreduce_f32_parallel",
        "wukong_argreduce_f32",
        "wukong_argreduce_f32_parallel",
        "wukong_dot_bf16",
        "wukong_sum_bf16",
        "wukong_dot_bf16_parallel",
        "wukong_sum_bf16_parallel",
        "wukong_reduce_bf16",
        "wukong_reduce_bf16_parallel",
        "wukong_dot_f16",
        "wukong_sum_f16",
        "wukong_dot_f16_parallel",
        "wukong_sum_f16_parallel",
        "wukong_reduce_f16",
        "wukong_reduce_f16_parallel",
        "wukong_axpby_bf16",
        "wukong_axpby_f16",
        "wukong_axpby_bf16_out",
        "wukong_axpby_f16_out",
        "wukong_sgemm_bf16_nt",
        "wukong_sgemm_bf16_nt_parallel",
        "wukong_sgemm_f16_nt",
        "wukong_sgemm_f16_nt_parallel",
        "wukong_sgemm_bf16_tn",
        "wukong_sgemm_bf16_tn_parallel",
        "wukong_sgemm_f16_tn",
        "wukong_sgemm_f16_tn_parallel",
        "wukong_sgemm_bf16_nt_epi",
        "wukong_sgemm_bf16_nt_epi_parallel",
        "wukong_sgemm_f16_nt_epi",
        "wukong_sgemm_f16_nt_epi_parallel",
        "wukong_norm_f32",
        "wukong_norm_f32_parallel",
        "wukong_norm_affine_f32",
        "wukong_norm_affine_f32_parallel",
        "wukong_embedding_f32",
        "wukong_embedding_f32_parallel",
        "wukong_scatter_add_f32",
        "wukong_scatter_add_f32_parallel",
        "wukong_attention_f32",
        "wukong_rt_fmt_int",
        "wukong_rt_fmt_f64",
        "wukong_rt_fmt_f32",
        "wukong_rt_fmt_str",
        "wukong_rt_fmt_lit",
        "wukong_rt_fmt_slice",
        "wukong_rt_arg_count",
        "wukong_rt_arg",
        "wukong_rt_arg_i64",
        "wukong_rt_arg_f32",
        "wukong_rt_env",
        "wukong_rt_read_stdin",
        "wukong_rt_exit",
        "wukong_rt_read_dt",
        "wukong_rt_write_dt",
        "wukong_rt_file_size",
        "wukong_rt_st_read",
        "wukong_rt_npy_read",
        "wukong_rand_fill_f32",
        "wukong_topk_f32",
        "wukong_argsort_f32",
        "wukong_sample_categorical_f32",
        "wukong_attention_kv_f32",
        "wukong_attention_kv_f32_parallel",
        "wukong_kv_append_f32",
        "wukong_sgemv_bf16",
        "wukong_sgemv_bf16_parallel",
        "wukong_sgemv_f16",
        "wukong_sgemv_f16_parallel",
        "wukong_quantize_q8_0_f32",
        "wukong_quantize_q4_0_f32",
        "wukong_gemv_q8_0_f32",
        "wukong_gemv_q8_0_f32_parallel",
        "wukong_gemv_q4_0_f32",
        "wukong_gemv_q4_0_f32_parallel",
    ];

    fn interpreter<'a>(program: &'a Program, interner: &'a Interner) -> Interp<'a, 'a> {
        Interp {
            program,
            interner,
            memory: Vec::new(),
            stdout: Vec::new(),
            scratch_arena: ScratchSlots::default(),
            slots: SlotReuse::default(),
            frames: Vec::new(),
            call_args: Vec::new(),
            scratch: Vec::new(),
            vecs: Vec::new(),
            vector_peak: 0,
            data_addrs: HashMap::new(),
            function_indices: None,
            static_indices: None,
            accel: None,
            depth: 0,
        }
    }

    #[test]
    fn every_name_the_string_dispatch_accepted_still_resolves() {
        assert_eq!(STRING_DISPATCH_NAMES.len(), 209);
        for &name in STRING_DISPATCH_NAMES {
            let id = Intrinsic::from_name(name)
                .unwrap_or_else(|| panic!("`{name}` no longer resolves to an intrinsic"));
            assert_eq!(id.name(), name);
        }
        // The one intrinsic the frame loop dispatches before the string match ever ran.
        assert_eq!(
            Intrinsic::from_name("wukong_parallel_for"),
            Some(Intrinsic::ParallelFor)
        );
    }

    #[test]
    fn the_table_round_trips_and_names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for &id in Intrinsic::ALL {
            assert!(seen.insert(id.name()), "duplicate name `{}`", id.name());
            assert_eq!(Intrinsic::from_name(id.name()), Some(id), "{id:?}");
        }
        assert_eq!(seen.len(), Intrinsic::ALL.len());
        for name in [
            "",
            "printx",
            "wukong_sgemm_",
            "WUKONG_SGEMM",
            "wukong_parallel",
        ] {
            assert_eq!(Intrinsic::from_name(name), None, "{name:?}");
        }
    }

    /// Every id reaches an implementing arm: called with a full row of undefined operands it may
    /// fail (no pointer, no path) or no-op (a zero extent), but never as an unknown callee — the
    /// error a table entry with no arm behind it would produce. `wukong_parallel_for` is the
    /// exception by design: the frame loop runs its body, and a direct call is unknown as before.
    #[test]
    fn every_intrinsic_reaches_an_implementing_arm() {
        let program = Program::new();
        let interner = Interner::new();
        let mut interp = interpreter(&program, &interner);
        let args = [Value::Unit; 24];
        for &id in Intrinsic::ALL {
            let result = interp.call_intrinsic(id, &args);
            let unknown = matches!(&result, Err(e) if e.contains("unknown function or intrinsic"));
            assert_eq!(unknown, id == Intrinsic::ParallelFor, "{id:?}: {result:?}");
        }
    }

    #[test]
    fn unknown_names_fail_with_the_string_dispatch_error_text() {
        let program = Program::new();
        let interner = Interner::new();
        let mut interp = interpreter(&program, &interner);
        for name in ["no_such_intrinsic", "wukong_parallel_for", ""] {
            assert_eq!(
                interp.intrinsic(name, &[]),
                Err(format!("call to unknown function or intrinsic `{name}`"))
            );
        }
    }
}
