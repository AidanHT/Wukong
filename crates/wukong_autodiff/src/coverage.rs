//! The recognizer ↔ autodiff coupling table: every runtime symbol `wukong_mir_build` can emit that
//! the tape does NOT differentiate, each with the reason a user sees when it blocks a gradient.
//!
//! A recognizer arm that lands without an autodiff mirror used to be invisible until a user hit the
//! generic "no VJP rule" refusal (the historical `norm_f32_parallel` hole). Now every `"wukong_*"`
//! symbol in `wukong_mir_build`'s source must be exactly one of: a forward kernel the tape
//! differentiates (`tape::FORWARD_KERNELS`), a routine the scalar core differentiates
//! (`SCALAR_CORE`), or an entry below — and `tests.rs`
//! (`every_emitted_kernel_symbol_is_differentiable_or_declined_by_name`) fails the build otherwise,
//! in both directions (a new symbol with no disposition, a stale entry for a symbol that is gone).
//!
//! Adding a recognizer: give its kernel a VJP rule (an entry in `tape::FORWARD_KERNELS` plus the
//! rule), or list it here with an honest reason.

/// Runtime routines the scalar core (`lib.rs::diff_inst`) differentiates as value-returning calls:
/// the f32 trig routine and the double-precision library (`wukong_rt_math_f64` /
/// `wukong_rt_math2_f64`, closed-form derivatives per op code). (Read only by the coupling test:
/// the scalar core matches each routine by name.)
#[cfg(test)]
pub(crate) const SCALAR_CORE: &[&str] = &[
    "wukong_rt_trig_f32",
    "wukong_rt_math_f64",
    "wukong_rt_math2_f64",
];

const INDEX_VALUED: &str = "returns an index (argmax / argmin): piecewise constant, so its \
                            derivative is zero almost everywhere and undefined at a tie";
const HALF_PRECISION: &str = "half-precision (bf16 / f16) storage: the buffer tape differentiates \
                              f32 buffers; differentiate the f32 model and cast at the edges";
const QUANTIZED: &str = "integer-quantized operands carry no gradient (quantization-aware \
                         training is not supported)";
const SECOND_ORDER: &str = "a training-backward kernel: differentiating it is second order, \
                            which the buffer tape does not support (see `grad_dot` / `hvp` for \
                            the scalar subset)";
const SCAN: &str = "a scan (prefix recurrence) has no VJP rule yet";
const COLUMN_STAT: &str = "this column statistic has no VJP rule yet (colsum, colmean and \
                           colsumsq are differentiable)";
const NO_GRADIENT: &str = "carries no gradient";
const FILE_READ: &str = "a file read overwrites its buffer with data; it carries no gradient";

/// Every non-differentiable symbol `wukong_mir_build` can emit, with the reason a decline reports.
pub(crate) const NOT_DIFFERENTIABLE: &[(&str, &str)] = &[
    ("wukong_argreduce_f32", INDEX_VALUED),
    ("wukong_argreduce_f32_parallel", INDEX_VALUED),
    ("wukong_rowargmax_i32", INDEX_VALUED),
    ("wukong_rowargmax_i32_parallel", INDEX_VALUED),
    ("wukong_rowargmin_i32", INDEX_VALUED),
    ("wukong_rowargmin_i32_parallel", INDEX_VALUED),
    ("wukong_colargmax_i32", INDEX_VALUED),
    ("wukong_colargmax_i32_parallel", INDEX_VALUED),
    ("wukong_colargmin_i32", INDEX_VALUED),
    ("wukong_colargmin_i32_parallel", INDEX_VALUED),
    (
        "wukong_attention_f32",
        "the fused scaled-dot-product attention has no backward kernel yet; spell attention as \
         matmul + softmax + matmul (each differentiable) to train it",
    ),
    ("wukong_avgpool2d_f32", "2-D pooling has no VJP rule yet"),
    (
        "wukong_avgpool2d_f32_parallel",
        "2-D pooling has no VJP rule yet",
    ),
    ("wukong_maxpool2d_f32", "2-D pooling has no VJP rule yet"),
    (
        "wukong_maxpool2d_f32_parallel",
        "2-D pooling has no VJP rule yet",
    ),
    ("wukong_axpby_bf16", HALF_PRECISION),
    ("wukong_axpby_bf16_out", HALF_PRECISION),
    ("wukong_axpby_f16", HALF_PRECISION),
    ("wukong_axpby_f16_out", HALF_PRECISION),
    ("wukong_dot_bf16", HALF_PRECISION),
    ("wukong_dot_bf16_parallel", HALF_PRECISION),
    ("wukong_dot_f16", HALF_PRECISION),
    ("wukong_dot_f16_parallel", HALF_PRECISION),
    ("wukong_sum_bf16", HALF_PRECISION),
    ("wukong_sum_bf16_parallel", HALF_PRECISION),
    ("wukong_sum_f16", HALF_PRECISION),
    ("wukong_sum_f16_parallel", HALF_PRECISION),
    ("wukong_reduce_bf16", HALF_PRECISION),
    ("wukong_reduce_bf16_parallel", HALF_PRECISION),
    ("wukong_reduce_f16", HALF_PRECISION),
    ("wukong_reduce_f16_parallel", HALF_PRECISION),
    ("wukong_sgemm_bf16_nt", HALF_PRECISION),
    ("wukong_sgemm_bf16_nt_parallel", HALF_PRECISION),
    ("wukong_sgemm_bf16_nt_epi", HALF_PRECISION),
    ("wukong_sgemm_bf16_nt_epi_parallel", HALF_PRECISION),
    ("wukong_sgemm_bf16_tn", HALF_PRECISION),
    ("wukong_sgemm_bf16_tn_parallel", HALF_PRECISION),
    ("wukong_sgemm_f16_nt", HALF_PRECISION),
    ("wukong_sgemm_f16_nt_parallel", HALF_PRECISION),
    ("wukong_sgemm_f16_nt_epi", HALF_PRECISION),
    ("wukong_sgemm_f16_nt_epi_parallel", HALF_PRECISION),
    ("wukong_sgemm_f16_tn", HALF_PRECISION),
    ("wukong_sgemm_f16_tn_parallel", HALF_PRECISION),
    ("wukong_transpose_u16", HALF_PRECISION),
    ("wukong_transpose_u16_parallel", HALF_PRECISION),
    ("wukong_vmath_bf16", HALF_PRECISION),
    ("wukong_vmath_bf16_out", HALF_PRECISION),
    ("wukong_vmath_f16", HALF_PRECISION),
    ("wukong_vmath_f16_out", HALF_PRECISION),
    ("wukong_coll2_f32", COLUMN_STAT),
    ("wukong_coll2_f32_parallel", COLUMN_STAT),
    ("wukong_colrms_f32", COLUMN_STAT),
    ("wukong_colrms_f32_parallel", COLUMN_STAT),
    ("wukong_colmax_f32", COLUMN_STAT),
    ("wukong_colmax_f32_parallel", COLUMN_STAT),
    ("wukong_colmin_f32", COLUMN_STAT),
    ("wukong_colmin_f32_parallel", COLUMN_STAT),
    ("wukong_colmaxabs_f32", COLUMN_STAT),
    ("wukong_colmaxabs_f32_parallel", COLUMN_STAT),
    ("wukong_cummax_f32", SCAN),
    ("wukong_cummax_f32_parallel", SCAN),
    ("wukong_cummin_f32", SCAN),
    ("wukong_cummin_f32_parallel", SCAN),
    ("wukong_cumprod_f32", SCAN),
    ("wukong_cumprod_f32_parallel", SCAN),
    ("wukong_cumsum_f32", SCAN),
    ("wukong_cumsum_f32_parallel", SCAN),
    ("wukong_lrscan_f32", SCAN),
    ("wukong_lrscan_f32_parallel", SCAN),
    ("wukong_dequant_f32", QUANTIZED),
    ("wukong_dequant_f32_parallel", QUANTIZED),
    ("wukong_dequant_perchan_f32", QUANTIZED),
    ("wukong_dequant_perchan_f32_parallel", QUANTIZED),
    ("wukong_i8gemm_nt", QUANTIZED),
    ("wukong_i8gemm_nt_parallel", QUANTIZED),
    ("wukong_i8gemm_nt_deq", QUANTIZED),
    ("wukong_i8gemm_nt_deq_parallel", QUANTIZED),
    (
        "wukong_kd_loss_f32",
        "the fused distillation loss has no VJP rule yet (`wukong_kldiv_f32` is differentiable)",
    ),
    (
        "wukong_kd_loss_f32_parallel",
        "the fused distillation loss has no VJP rule yet (`wukong_kldiv_f32` is differentiable)",
    ),
    ("wukong_layernorm_bwd_f32", SECOND_ORDER),
    ("wukong_layernorm_bwd_f32_parallel", SECOND_ORDER),
    ("wukong_rmsnorm_bwd_f32", SECOND_ORDER),
    ("wukong_rmsnorm_bwd_f32_parallel", SECOND_ORDER),
    ("wukong_softmax_bwd_f32", SECOND_ORDER),
    ("wukong_softmax_bwd_f32_parallel", SECOND_ORDER),
    ("wukong_xent_bwd_f32", SECOND_ORDER),
    ("wukong_xent_bwd_f32_parallel", SECOND_ORDER),
    ("wukong_rope_bwd_f32", SECOND_ORDER),
    ("wukong_rope_bwd_f32_parallel", SECOND_ORDER),
    ("wukong_scatter_add_f32", SECOND_ORDER),
    ("wukong_scatter_add_f32_parallel", SECOND_ORDER),
    (
        "wukong_vhorner_f32",
        "the streaming Horner polynomial has no VJP rule yet",
    ),
    (
        "wukong_parallel_for",
        "a mid-function @parallel region: the outlined body is opaque to the tape; keep loops \
         whose gradient is wanted in the serial spelling (whole-function @parallel is fine)",
    ),
    ("wukong_now_ns", NO_GRADIENT),
    ("wukong_rt_alloc", NO_GRADIENT),
    ("wukong_rt_free", NO_GRADIENT),
    (
        "wukong_rt_read_f32",
        "a file read overwrites its buffer with data; it carries no gradient",
    ),
    (
        "wukong_rt_read_f64",
        "a file read overwrites its buffer with data; it carries no gradient",
    ),
    (
        "wukong_rt_read_i32",
        "a file read overwrites its buffer with data; it carries no gradient",
    ),
    (
        "wukong_rt_read_i64",
        "a file read overwrites its buffer with data; it carries no gradient",
    ),
    (
        "wukong_rt_read_i8",
        "a file read overwrites its buffer with data; it carries no gradient",
    ),
    (
        "wukong_rt_read_u8",
        "a file read overwrites its buffer with data; it carries no gradient",
    ),
    ("wukong_rt_write_f32", NO_GRADIENT),
    ("wukong_rt_write_f64", NO_GRADIENT),
    ("wukong_rt_write_i32", NO_GRADIENT),
    ("wukong_rt_write_i64", NO_GRADIENT),
    ("wukong_rt_write_i8", NO_GRADIENT),
    ("wukong_rt_write_u8", NO_GRADIENT),
    // The ML I/O and randomness entries (see `tape::ml_call_gradient` for the per-entry decision).
    ("wukong_rt_read_dt", FILE_READ),
    ("wukong_rt_st_read", FILE_READ),
    ("wukong_rt_npy_read", FILE_READ),
    ("wukong_rt_write_dt", NO_GRADIENT),
    ("wukong_rt_file_size", NO_GRADIENT),
    (
        "wukong_rand_fill_f32",
        "counter-based random numbers depend on no input; the fill carries no gradient",
    ),
    (
        "wukong_argsort_f32",
        "an argsort writes indices: piecewise constant, so it carries no gradient",
    ),
    (
        "wukong_sample_categorical_f32",
        "a categorical draw returns an index: a discrete sample carries no gradient",
    ),
    (
        "wukong_topk_f32",
        "top-k has no VJP rule yet (the gradient would scatter back to the selected indices); \
         autodiff refuses it rather than drop the gradient",
    ),
    // Scratch-arena bookkeeping and the slice bounds-failure edge.
    (
        "wukong_rt_scratch_alloc",
        "scratch-arena storage: the tape does not model scratch lifetimes; allocate buffers on \
         the gradient path with `alloc_<T>`",
    ),
    ("wukong_rt_scratch_mark", NO_GRADIENT),
    (
        "wukong_rt_scratch_release",
        "scratch-arena bookkeeping: the tape does not model scratch lifetimes; allocate buffers \
         on the gradient path with `alloc_<T>`",
    ),
    (
        "wukong_rt_bounds_fail",
        "the failure edge of a slice bounds check: it never returns, so it carries no gradient",
    ),
];

/// The reason `name` is not differentiable, when the table knows it.
pub(crate) fn not_differentiable_reason(name: &str) -> Option<&'static str> {
    NOT_DIFFERENTIABLE
        .iter()
        .find(|(n, _)| *n == name)
        .map(|&(_, why)| why)
}

/// Void calls that read values but write no memory a gradient flows through — printing, asserting,
/// releasing a heap buffer, writing a file. The reverse pass replays them in the forward and skips
/// them, instead of refusing the whole loss over a debug `print`.
pub(crate) fn is_effect_only(name: &str) -> bool {
    matches!(
        name,
        "print"
            | "println"
            | "print_str"
            | "println_str"
            | "print_u"
            | "println_u"
            | "print_bool"
            | "println_bool"
            | "print_char"
            | "println_char"
            | "assert"
    ) || name == "wukong_rt_free"
        || name.starts_with("wukong_rt_write_")
}
