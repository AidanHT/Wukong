//! Tensor-tape VJPs: differentiating a straight-line sequence of recognized **kernel calls**.
//!
//! Wukong lowers tensor ops to runtime-kernel calls — the GEMM family (`wukong_sgemm*`, GEMV/GEVM),
//! elementwise activations (`wukong_vmath_f32`, `wukong_vmath2_f32`), streaming affine / Hadamard /
//! quotient maps (`wukong_velem_f32`), reductions (`wukong_sreduce_f32`), the row-wise norms
//! (`wukong_norm_f32`, `wukong_norm_affine_f32`), the broadcast bias, transposes, column reductions,
//! the classification losses and the transformer pieces (RoPE, embedding gather, the fused
//! scaled-dot-product attention) — each also in its `@parallel` spelling where the runtime has one.
//! The interpreter runs the *same* kernels, so a forward MIR function can be a tape of these calls
//! plus a final scalar loss. Where the scalar core ([`crate::Vjp`]) tracks an adjoint per SSA value,
//! this module tracks an adjoint per **buffer** (memory), the PyTorch-style picture tensor ops need:
//! each forward buffer has a gradient buffer, set by a consumer's VJP and read by the producer's.
//!
//! The backward emits **more tuned kernel calls**: the GEMM adjoints are the runtime's own
//! transposed GEMM spellings (`wukong_sgemm_nt` / `wukong_sgemm_tn`), the norm backwards are the
//! runtime's fused `wukong_{softmax,layernorm,rmsnorm}_bwd_f32`, the smooth activations ride the
//! fused `wukong_vmath2_f32` backward codes, and the `@parallel` forward spelling selects the
//! `@parallel` backward spelling. Small synthesized counted loops remain only where no kernel
//! exists (an algebraic activation derivative, a sign mask, a per-row scale).
//!
//! # Gradient accumulation (fan-out)
//!
//! A forward buffer consumed by several ops (a residual connection, a weight shared by two matmuls,
//! a loss reading a buffer twice) receives several gradient contributions. The first contribution
//! is written straight into the gradient buffer; every later one is computed into a heap scratch
//! and added (`Vjp::begin` / `Vjp::end`). The GEMM adjoints accumulate natively through `beta = 1`.
//! Scratch comes from `wukong_rt_alloc` (zeroed, so no garbage ever reaches a kernel) and is
//! released with `wukong_rt_free` before the gradient returns.
//!
//! Buffers are differentiated under the same single-block contract as the scalar core; the only new
//! control flow is the counted loops this module synthesizes. (So the *emitted* gradient function is
//! multi-block: it is not itself re-differentiable, which `grad` says by name; `crate::grad_dot`
//! is the second-order entry point for the scalar subset.)

use crate::Vjp;
use std::collections::HashMap;
use wukong_mir::{BinOp, BlockId, CastKind, CmpOp, MirType, Op, ValueId};
use wukong_span::{Interner, Symbol};

// --- runtime kernel op codes (mirrored from wukong_runtime; the interp dispatches the same) ------
// A renumber in the runtime silently changes which rule fires here, so every code is a copy of the
// runtime's `pub const` of the same name.
pub(crate) const VM_EXP: i64 = 0;
pub(crate) const VM_LOG: i64 = 1;
pub(crate) const VM_TANH: i64 = 2;
pub(crate) const VM_SIGMOID: i64 = 3;
pub(crate) const VM_RELU: i64 = 4;
pub(crate) const VM_SILU: i64 = 5;
pub(crate) const VM_GELU: i64 = 6;
pub(crate) const VM_ELU: i64 = 7;
pub(crate) const VM_LEAKYRELU: i64 = 8;
pub(crate) const VM_SOFTPLUS: i64 = 9;
pub(crate) const VM_MISH: i64 = 10;
pub(crate) const VM_SELU: i64 = 11;
pub(crate) const VM_TANHSHRINK: i64 = 12;
pub(crate) const VM_HARDSIGMOID: i64 = 13;
pub(crate) const VM_HARDSWISH: i64 = 14;
pub(crate) const VM_SIN: i64 = 15;
pub(crate) const VM_COS: i64 = 16;
pub(crate) const VM_ERF: i64 = 17;
pub(crate) const VM_EXP2: i64 = 18;
pub(crate) const VM_LOG2: i64 = 19;
pub(crate) const VM_SINH: i64 = 20;
pub(crate) const VM_COSH: i64 = 21;
pub(crate) const VM_ASINH: i64 = 22;
pub(crate) const VM_ACOSH: i64 = 23;
pub(crate) const VM_ATANH: i64 = 24;
pub(crate) const VM_ATAN: i64 = 25;
pub(crate) const VM_EXPM1: i64 = 26;
pub(crate) const VM_LOG1P: i64 = 27;
pub(crate) const VM_EXP10: i64 = 28;
pub(crate) const VM_LOG10: i64 = 29;
pub(crate) const VM_SOFTSIGN: i64 = 30;
pub(crate) const VM_LOGSIGMOID: i64 = 31;
pub(crate) const VM_TAN: i64 = 32;
pub(crate) const VM_ASIN: i64 = 33;
pub(crate) const VM_ACOS: i64 = 34;
pub(crate) const VM_CBRT: i64 = 35;
// The natural spellings of the activations (`wukong_runtime::vspell`, P12.10): the same functions
// written another way, each with its own forward kernel. `tape_nn::spelling_base` routes each to
// the backward rule of the activation it spells.
pub(crate) const VM_GELU_TANH: i64 = 36;
pub(crate) const VM_GELU_TANH_HF: i64 = 37;
pub(crate) const VM_GELU_TANH_NOFMA: i64 = 38;
pub(crate) const VM_GELU_TANH_FAST: i64 = 39;
pub(crate) const VM_GELU_TANH_CDF: i64 = 40;
pub(crate) const VM_QUICK_GELU: i64 = 41;
pub(crate) const VM_GELU_ERF_DIV: i64 = 42;
pub(crate) const VM_GELU_ERF_MUL: i64 = 43;
pub(crate) const VM_GELU_ERF_CDF: i64 = 44;
pub(crate) const VM_SILU_DIV: i64 = 45;
pub(crate) const VM_SILU_SX: i64 = 46;
pub(crate) const VM_SIGMOID_RATIO: i64 = 47;
pub(crate) const VM_SOFTPLUS_NAIVE: i64 = 48;
pub(crate) const VM_SOFTPLUS_LOG1P: i64 = 49;
pub(crate) const VM_MISH_NAIVE: i64 = 50;
pub(crate) const VM_MISH_SOFTPLUS: i64 = 51;
pub(crate) const VM_MISH_LOG1P: i64 = 52;
pub(crate) const VM_HARDSIGMOID_MINMAX: i64 = 53;
pub(crate) const VM_HARDSIGMOID_CLAMP: i64 = 54;
pub(crate) const VM_HARDSWISH_MINMAX: i64 = 55;
pub(crate) const VM_HARDSWISH_CLAMP: i64 = 56;
pub(crate) const VM_ELU_EXPM1: i64 = 57;
pub(crate) const VM_LEAKY_LT: i64 = 58;
/// QuickGELU's slope (`wukong_runtime::vspell`'s `QUICK`, the `f32` of `1.702`), mirrored.
pub(crate) const QUICK_GELU_SLOPE: f64 = 1.702;
/// The leaky-ReLU negative slope baked into `wukong_runtime::vmath` (`LEAKY_ALPHA`). Mirrored, so a
/// change there must be mirrored here or the backward stops matching the forward.
pub(crate) const LEAKY_ALPHA: f64 = 0.01;
/// SELU's `lambda`/`alpha` (`wukong_runtime::vmath::SELU_LAMBDA`/`SELU_ALPHA`), mirrored.
pub(crate) const SELU_LAMBDA: f64 = 1.050_701;
pub(crate) const SELU_ALPHA: f64 = 1.673_263_2;
/// `1/6`, the hard-sigmoid / hard-swish scale (`wukong_runtime::vmath::INV6`), mirrored.
pub(crate) const INV6: f64 = 1.0 / 6.0;
// `wukong_vmath2_f32(x, y, out, n, op)` op codes — a namespace separate from `VM_*`. The forward
// two-input transcendentals and gates, and the fused activation *backward* codes
// (`dx = dy · act'(x)` in one 256-bit pass, inputs `(x, dy)`).
pub(crate) const VM2_POW: i64 = 0;
pub(crate) const VM2_ATAN2: i64 = 1;
pub(crate) const VM2_HYPOT: i64 = 2;
pub(crate) const VM2_SILU_BWD: i64 = 3;
pub(crate) const VM2_GELU_BWD: i64 = 4;
pub(crate) const VM2_SIGMOID_BWD: i64 = 5;
pub(crate) const VM2_TANH_BWD: i64 = 6;
pub(crate) const VM2_ELU_BWD: i64 = 7;
pub(crate) const VM2_SOFTPLUS_BWD: i64 = 8;
pub(crate) const VM2_SILU_GATE: i64 = 9;
pub(crate) const VM2_GELU_GATE: i64 = 10;
pub(crate) const VM2_SIGMOID_GATE: i64 = 11;
pub(crate) const RED_DOT: i64 = 0;
pub(crate) const RED_SSD: i64 = 1;
pub(crate) const RED_SUM: i64 = 2;
pub(crate) const RED_SUMSQ: i64 = 3;
pub(crate) const RED_MAX: i64 = 4;
pub(crate) const RED_MIN: i64 = 5;
pub(crate) const RED_MAXABS: i64 = 6;
pub(crate) const RED_SUMABS: i64 = 9;
pub(crate) const RED_ABSDIFF: i64 = 10;
pub(crate) const NORM_SOFTMAX: i64 = 0;
pub(crate) const NORM_LAYERNORM: i64 = 1;
pub(crate) const NORM_RMSNORM: i64 = 2;
pub(crate) const NORM_LOGSOFTMAX: i64 = 3;
pub(crate) const NORM_L2NORM: i64 = 4;
/// `wukong_bias_bcast_f32`'s "no fused activation" op code (`wukong_runtime::bias::BIAS_ACT_NONE`).
/// Any other value selects a fused `VM_*` activation.
pub(crate) const BIAS_ACT_NONE: i64 = -1;
/// velem activation: identity (`out = a*x + b*y + c`).
pub(crate) const VE_ID: i64 = 0;
/// velem activation: ReLU (`out = max(.., 0)`).
pub(crate) const VE_RELU: i64 = 1;
/// velem activation: ReLU6 (`out = min(max(.., 0), 6)`).
pub(crate) const VE_RELU6: i64 = 2;
/// OR'd into a velem op when `y` is read (`b` may be non-zero).
pub(crate) const VE_USE_Y: i64 = 256;
/// velem *compute mode*: the elementwise Hadamard product (`out = act(x · y)`). It lives ABOVE the
/// low activation byte, so a `op & 0xff` test does not see it — and it is **non-affine**.
pub(crate) const VE_HADAMARD: i64 = 512;
/// velem *compute mode*: the elementwise quotient (`out = act(x / y)`) — likewise non-affine.
pub(crate) const VE_DIV: i64 = 1024;
/// `wukong_sgemm_nt_epi`'s fused epilogue activations (`wukong_runtime::gemm::ACT_*`).
pub(crate) const EPI_IDENTITY: i64 = 0;
pub(crate) const EPI_RELU: i64 = 1;
pub(crate) const EPI_GELU: i64 = 2;
pub(crate) const EPI_SILU: i64 = 3;

pub(crate) const F32: MirType = MirType::F32;
pub(crate) const I64T: MirType = MirType::I64;
pub(crate) const PTR: MirType = MirType::Ptr;

/// The largest recognized ABI (`wukong_sgemm_nt_epi`'s nine operands).
const MAX_ARITY: usize = 9;

/// A recognized forward kernel family. Each has one VJP rule; the serial and `@parallel` spellings
/// of a family share it (the parallel kernels are bit-identical to the serial ones by contract).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kernel {
    Sreduce,
    Sgemm,
    SgemmNt,
    SgemmTn,
    SgemmNtAlpha,
    SgemmNtEpi,
    Sgemv,
    SgemvAlpha,
    Sgevm,
    Vmath,
    Vmath2,
    Velem,
    Norm,
    NormAffine,
    Bias,
    Transpose,
    ColSum,
    ColMean,
    ColSumSq,
    Xent,
    Logsumexp,
    Entropy,
    Kldiv,
    Rope,
    Embedding,
    Attention,
}

/// Every forward kernel symbol the tape differentiates: `(symbol, family, @parallel spelling)`.
///
/// This table is the single source of truth for `is_kernel`, the arity check and the dispatch, so a
/// symbol cannot be recognized without also having an arity and a rule. The coupling test in
/// `tests.rs` (`every_emitted_kernel_symbol_is_differentiable_or_declined_by_name`) scans every
/// `"wukong_*"` symbol `wukong_mir_build` can emit and fails unless it is listed here, in
/// `crate::coverage::NOT_DIFFERENTIABLE` with a reason, or handled by the scalar core.
pub(crate) const FORWARD_KERNELS: &[(&str, Kernel, bool)] = &[
    ("wukong_sreduce_f32", Kernel::Sreduce, false),
    ("wukong_sreduce_f32_parallel", Kernel::Sreduce, true),
    ("wukong_sgemm", Kernel::Sgemm, false),
    ("wukong_sgemm_parallel", Kernel::Sgemm, true),
    ("wukong_sgemm_nt", Kernel::SgemmNt, false),
    ("wukong_sgemm_nt_parallel", Kernel::SgemmNt, true),
    ("wukong_sgemm_tn", Kernel::SgemmTn, false),
    ("wukong_sgemm_tn_parallel", Kernel::SgemmTn, true),
    ("wukong_sgemm_nt_alpha", Kernel::SgemmNtAlpha, false),
    ("wukong_sgemm_nt_alpha_parallel", Kernel::SgemmNtAlpha, true),
    ("wukong_sgemm_nt_epi", Kernel::SgemmNtEpi, false),
    ("wukong_sgemm_nt_epi_parallel", Kernel::SgemmNtEpi, true),
    ("wukong_sgemv", Kernel::Sgemv, false),
    ("wukong_sgemv_parallel", Kernel::Sgemv, true),
    ("wukong_sgemv_alpha", Kernel::SgemvAlpha, false),
    ("wukong_sgemv_alpha_parallel", Kernel::SgemvAlpha, true),
    ("wukong_sgevm_f32", Kernel::Sgevm, false),
    ("wukong_sgevm_f32_parallel", Kernel::Sgevm, true),
    ("wukong_vmath_f32", Kernel::Vmath, false),
    ("wukong_vmath_f32_parallel", Kernel::Vmath, true),
    ("wukong_vmath2_f32", Kernel::Vmath2, false),
    ("wukong_velem_f32", Kernel::Velem, false),
    ("wukong_velem_f32_parallel", Kernel::Velem, true),
    ("wukong_norm_f32", Kernel::Norm, false),
    ("wukong_norm_f32_parallel", Kernel::Norm, true),
    ("wukong_norm_affine_f32", Kernel::NormAffine, false),
    ("wukong_norm_affine_f32_parallel", Kernel::NormAffine, true),
    ("wukong_bias_bcast_f32", Kernel::Bias, false),
    ("wukong_bias_bcast_f32_parallel", Kernel::Bias, true),
    ("wukong_transpose_f32", Kernel::Transpose, false),
    ("wukong_transpose_f32_parallel", Kernel::Transpose, true),
    ("wukong_colsum_f32", Kernel::ColSum, false),
    ("wukong_colsum_f32_parallel", Kernel::ColSum, true),
    ("wukong_colmean_f32", Kernel::ColMean, false),
    ("wukong_colmean_f32_parallel", Kernel::ColMean, true),
    ("wukong_colsumsq_f32", Kernel::ColSumSq, false),
    ("wukong_colsumsq_f32_parallel", Kernel::ColSumSq, true),
    ("wukong_xent_fwd_f32", Kernel::Xent, false),
    ("wukong_xent_fwd_f32_parallel", Kernel::Xent, true),
    ("wukong_logsumexp_f32", Kernel::Logsumexp, false),
    ("wukong_logsumexp_f32_parallel", Kernel::Logsumexp, true),
    ("wukong_entropy_f32", Kernel::Entropy, false),
    ("wukong_entropy_f32_parallel", Kernel::Entropy, true),
    ("wukong_kldiv_f32", Kernel::Kldiv, false),
    ("wukong_kldiv_f32_parallel", Kernel::Kldiv, true),
    ("wukong_rope_f32", Kernel::Rope, false),
    ("wukong_rope_f32_parallel", Kernel::Rope, true),
    ("wukong_embedding_f32", Kernel::Embedding, false),
    ("wukong_embedding_f32_parallel", Kernel::Embedding, true),
    // No `_parallel` twin: multi-head work is spread by the caller, not inside the kernel.
    ("wukong_attention_f32", Kernel::Attention, false),
];

impl Kernel {
    /// The C ABI's operand count — validated before any rule indexes `args[..]` raw, because
    /// `is_kernel` matches by NAME and a `.wk` program may declare a kernel name in an `extern "C"`
    /// block with any signature (an arity mismatch is a diagnostic, never an index panic).
    pub(crate) fn arity(self) -> usize {
        match self {
            Kernel::Sreduce => 4,                                   // (x, y, n, op) -> f32
            Kernel::Sgemm | Kernel::SgemmNt | Kernel::SgemmTn => 7, // (a, b, c, m, k, n, beta)
            Kernel::SgemmNtAlpha => 8, // (a, b, c, m, k, n, beta, alpha)
            Kernel::SgemmNtEpi => 9,   // (a, b, c, m, k, n, beta, bias, act)
            Kernel::Sgemv => 5,        // (a, x, y, m, n)
            Kernel::SgemvAlpha => 6,   // (a, x, y, m, n, alpha)
            Kernel::Sgevm => 6,        // (w, a, out, rows, cols, alpha)
            Kernel::Vmath => 4,        // (x, out, n, op)
            Kernel::Vmath2 => 5,       // (x, y, out, n, op)
            Kernel::Velem => 8,        // (x, y, out, n, a, b, c, op)
            Kernel::Norm => 6,         // (x, out, rows, cols, eps_bits, op)
            Kernel::NormAffine => 8,   // (x, out, gamma, beta, rows, cols, eps_bits, op)
            Kernel::Bias => 6,         // (x, b, out, rows, cols, op)
            Kernel::Transpose => 4,    // (src, dst, rows, cols)
            Kernel::ColSum | Kernel::ColMean | Kernel::ColSumSq => 4, // (x, out, rows, cols)
            Kernel::Xent => 5,         // (x, target, loss, rows, cols)
            Kernel::Logsumexp | Kernel::Entropy => 4, // (x, out, rows, cols)
            Kernel::Kldiv => 5,        // (p, q, out, rows, cols)
            Kernel::Rope => 5,         // (x, inv_freq, out, rows, half)
            Kernel::Embedding => 6,    // (out, weight, ids, t, h, v)
            Kernel::Attention => 8,    // (q, k, v, out, s, d, scale, causal)
        }
    }

    /// Which operand is the buffer the kernel WRITES — `None` for a reduction, whose result is a
    /// scalar.
    pub(crate) fn out_index(self) -> Option<usize> {
        match self {
            Kernel::Sreduce => None,
            Kernel::Embedding => Some(0),
            Kernel::Attention => Some(3),
            Kernel::Vmath
            | Kernel::Norm
            | Kernel::NormAffine
            | Kernel::Transpose
            | Kernel::ColSum
            | Kernel::ColMean
            | Kernel::ColSumSq
            | Kernel::Logsumexp
            | Kernel::Entropy => Some(1),
            _ => Some(2),
        }
    }

    /// Whether the family carries a GEMM `beta` operand (index 6): `beta != 0` reads the output.
    fn has_beta(self) -> bool {
        matches!(
            self,
            Kernel::Sgemm
                | Kernel::SgemmNt
                | Kernel::SgemmTn
                | Kernel::SgemmNtAlpha
                | Kernel::SgemmNtEpi
        )
    }
}

/// The recognized forward family of the runtime symbol `name`, by name (for callers holding only a
/// shared interner; [`Syms::kernel`] is the interned lookup the reverse pass uses).
pub(crate) fn kernel_by_name(name: &str) -> Option<Kernel> {
    FORWARD_KERNELS
        .iter()
        .find(|&&(n, _, _)| n == name)
        .map(|&(_, kind, _)| kind)
}

/// Pick the serial or the `@parallel` spelling of an emitted backward kernel, following the forward.
pub(crate) fn pick(serial: Symbol, parallel: Symbol, par: bool) -> Symbol {
    if par {
        parallel
    } else {
        serial
    }
}

/// Interned symbols for the kernels we recognize (forward) and emit (backward). Interning dedups,
/// so these compare equal to the symbols a forward tape was built with.
#[derive(Clone)]
pub(crate) struct Syms {
    /// Forward recognition: symbol -> (family, `@parallel` spelling). Built from [`FORWARD_KERNELS`];
    /// only ever looked up, never iterated, so its hash order cannot reach the emitted MIR.
    forward: HashMap<Symbol, (Kernel, bool)>,
    pub sgemm: Symbol,
    pub sgemm_par: Symbol,
    pub sgemm_nt: Symbol,
    pub sgemm_nt_par: Symbol,
    /// `C = Aᵀ·B` — the weight-gradient GEMM. It transposes `A` into its own scratch and runs the
    /// identical `A·B` microkernel, so it replaced the flat MIR transpose loop (an `SDiv` and an
    /// `SRem` per element plus an `m·n` scratch) the NT rule used to synthesize.
    pub sgemm_tn: Symbol,
    pub sgemm_tn_par: Symbol,
    /// The fused-epilogue GEMM, emitted with the IDENTITY epilogue to recompute a pre-activation.
    pub sgemm_nt_epi: Symbol,
    /// `C = α·A·Bᵀ`, emitted to recompute the attention weights `scale·Q·Kᵀ`.
    pub sgemm_nt_alpha: Symbol,
    pub sgemv_alpha: Symbol,
    pub sgevm: Symbol,
    pub vmath: Symbol,
    pub vmath2: Symbol,
    pub velem: Symbol,
    pub sreduce: Symbol,
    pub norm: Symbol,
    pub colsum: Symbol,
    /// `wukong_rt_math_f64(x, op)` — the value-returning f64 transcendental library an `f64`
    /// `exp`/`log`/`sin`/… lowers to. **Not** a kernel (it writes no buffer): it is differentiated
    /// on the scalar path in `lib.rs` (`diff_inst`), where its derivative is a closed form in `x`
    /// and the forward result, emitted as further calls of this symbol. Recorded here so the VJP
    /// can emit those calls without a mutable interner.
    pub math_f64: Symbol,
    /// `wukong_rt_math2_f64(a, b, op)` — the two-argument twin (`pow`/`atan2`/`hypot`); same rule.
    pub math2_f64: Symbol,
    pub bias: Symbol,
    pub softmax_bwd: Symbol,
    pub softmax_bwd_par: Symbol,
    pub layernorm_bwd: Symbol,
    pub layernorm_bwd_par: Symbol,
    pub rmsnorm_bwd: Symbol,
    pub rmsnorm_bwd_par: Symbol,
    pub transpose: Symbol,
    pub transpose_par: Symbol,
    pub xent_bwd: Symbol,
    pub xent_bwd_par: Symbol,
    pub rope_bwd: Symbol,
    pub rope_bwd_par: Symbol,
    pub scatter_add: Symbol,
    pub scatter_add_par: Symbol,
    /// Zeroed heap scratch (`wukong_rt_alloc(count, elem_size, elem_is_float) -> ptr`) and its
    /// release. Gradient scratch lives on the heap, not in `alloca`s: a native backward must not
    /// put megabytes of activations' gradients on the stack, and a zeroed buffer means no rule can
    /// ever read uninitialized memory (a `0·NaN` garbage fill is NaN).
    pub rt_alloc: Symbol,
    pub rt_free: Symbol,
    /// `wukong_parallel_for(n, body, env)`: a `@parallel` region in the forward, and the call that
    /// runs its backward (`region.rs`).
    pub pfor: Symbol,
}

impl Syms {
    pub fn new(it: &mut Interner) -> Syms {
        let forward = FORWARD_KERNELS
            .iter()
            .map(|&(name, kind, par)| (it.intern(name), (kind, par)))
            .collect();
        Syms {
            forward,
            sgemm: it.intern("wukong_sgemm"),
            sgemm_par: it.intern("wukong_sgemm_parallel"),
            sgemm_nt: it.intern("wukong_sgemm_nt"),
            sgemm_nt_par: it.intern("wukong_sgemm_nt_parallel"),
            sgemm_tn: it.intern("wukong_sgemm_tn"),
            sgemm_tn_par: it.intern("wukong_sgemm_tn_parallel"),
            sgemm_nt_epi: it.intern("wukong_sgemm_nt_epi"),
            sgemm_nt_alpha: it.intern("wukong_sgemm_nt_alpha"),
            sgemv_alpha: it.intern("wukong_sgemv_alpha"),
            sgevm: it.intern("wukong_sgevm_f32"),
            vmath: it.intern("wukong_vmath_f32"),
            vmath2: it.intern("wukong_vmath2_f32"),
            velem: it.intern("wukong_velem_f32"),
            sreduce: it.intern("wukong_sreduce_f32"),
            norm: it.intern("wukong_norm_f32"),
            colsum: it.intern("wukong_colsum_f32"),
            math_f64: it.intern("wukong_rt_math_f64"),
            math2_f64: it.intern("wukong_rt_math2_f64"),
            bias: it.intern("wukong_bias_bcast_f32"),
            softmax_bwd: it.intern("wukong_softmax_bwd_f32"),
            softmax_bwd_par: it.intern("wukong_softmax_bwd_f32_parallel"),
            layernorm_bwd: it.intern("wukong_layernorm_bwd_f32"),
            layernorm_bwd_par: it.intern("wukong_layernorm_bwd_f32_parallel"),
            rmsnorm_bwd: it.intern("wukong_rmsnorm_bwd_f32"),
            rmsnorm_bwd_par: it.intern("wukong_rmsnorm_bwd_f32_parallel"),
            transpose: it.intern("wukong_transpose_f32"),
            transpose_par: it.intern("wukong_transpose_f32_parallel"),
            xent_bwd: it.intern("wukong_xent_bwd_f32"),
            xent_bwd_par: it.intern("wukong_xent_bwd_f32_parallel"),
            rope_bwd: it.intern("wukong_rope_bwd_f32"),
            rope_bwd_par: it.intern("wukong_rope_bwd_f32_parallel"),
            scatter_add: it.intern("wukong_scatter_add_f32"),
            scatter_add_par: it.intern("wukong_scatter_add_f32_parallel"),
            rt_alloc: it.intern("wukong_rt_alloc"),
            rt_free: it.intern("wukong_rt_free"),
            pfor: it.intern("wukong_parallel_for"),
        }
    }

    /// The recognized family (and spelling) of a forward call, if any.
    pub(crate) fn kernel(&self, func: Symbol) -> Option<(Kernel, bool)> {
        self.forward.get(&func).copied()
    }
}

/// A synthesized counted loop, threaded between [`Vjp::open_loop`] and [`Vjp::close_loop`]. After
/// `open_loop` the builder's current block IS the body, so the caller emits straight into it and
/// `close_loop` appends the back-edge there.
pub(crate) struct Loop {
    header: BlockId,
    exit: BlockId,
    /// The `i64` loop index, live in the body block.
    pub(crate) idx: ValueId,
}

/// Where one gradient contribution to a buffer is written: straight into its gradient buffer (the
/// first contribution), or into a scratch that [`Vjp::end`] then adds into it (every later one).
pub(crate) struct Contrib {
    /// The buffer the rule writes its contribution into (overwriting).
    pub(crate) write: ValueId,
    /// `(gradient buffer, element count)` to accumulate `write` into, for a non-first contribution.
    acc: Option<(ValueId, ValueId)>,
}

impl<'a> Vjp<'a> {
    /// Is `func` a tensor kernel this module differentiates?
    pub(crate) fn is_kernel(&self, func: Symbol) -> bool {
        self.syms.kernel(func).is_some()
    }

    /// Record every buffer a recognized kernel WRITES in the forward pass (canonicalized, so the
    /// `gep(buf, 0)` and bare-alloca spellings key the same entry), with the writing call — and which
    /// buffers more than one kernel writes, and which a non-constant float `store` writes.
    ///
    /// `grad_target` needs the writer to size a gradient for a non-differentiated PARAMETER a kernel
    /// writes (a scalar read of a kernel output must reach the kernel's inputs, and a parameter
    /// carries no element count of its own). `multi_written` and `store_written` drive the
    /// write-after-write rule in [`Vjp::diff_kernel_call`].
    pub(crate) fn scan_kernel_outputs(&mut self) {
        let insts = &self.entry_block().insts;
        let mut writes: HashMap<ValueId, usize> = HashMap::new();
        let mut outs: Vec<(ValueId, Kernel, &'a [ValueId])> = Vec::new();
        for inst in insts {
            if let Op::Store { ptr, value } = &inst.op {
                if self.fwd.value_type(*value).is_float() && !self.is_const(*value) {
                    let base = self.store_base(*ptr);
                    self.store_written.insert(base);
                }
                continue;
            }
            let Op::Call { func, args } = &inst.op else {
                continue;
            };
            let Some((kind, _)) = self.syms.kernel(*func) else {
                continue;
            };
            if kind.arity() != args.len() {
                continue; // a mis-declared ABI: `diff_kernel_call` reports it with the arity error
            }
            let Some(oi) = kind.out_index() else {
                continue;
            };
            let out = self.canon(args[oi]);
            *writes.entry(out).or_insert(0) += 1;
            outs.push((out, kind, args.as_slice()));
        }
        // Forward order, so the LAST writer of a buffer is the one recorded.
        for (out, kind, args) in outs {
            self.kernel_out.insert(out, (kind, args));
        }
        // A buffer a `@parallel` region writes has another writer like a stored-into one (the
        // write-after-write rule), and a written parameter needs a gradient when it is read later.
        let regions = self.regions;
        for r in regions {
            for (k, v) in r.slots.iter().enumerate() {
                let (Some(v), Some(u)) = (v, r.body.slots.get(k)) else {
                    continue;
                };
                if !u.write {
                    continue;
                }
                let base = self.canon(*v);
                self.store_written.insert(base);
                if self.fwd.params.contains(&base) {
                    self.region_out.insert(base);
                }
            }
        }
        self.multi_written
            .extend(writes.into_iter().filter(|&(_, c)| c > 1).map(|(b, _)| b));
    }

    /// Canonicalize a kernel buffer argument to the base buffer it addresses: peel a whole-buffer
    /// `gep(buf, 0)` (the form MIR lowering emits for *some* kernel operands — e.g. `vmath`'s output —
    /// while others, like `sreduce`'s input, pass the bare `alloca`). Without this, the two forms key
    /// *different* buffer-adjoint entries and the gradient chain silently breaks between the ops. A
    /// non-zero-offset gep is a genuine sub-slice and is left alone. Dim/op-code operands are constants
    /// (not geps), so canonicalizing the whole argument list is a no-op for them.
    pub(crate) fn canon(&self, mut v: ValueId) -> ValueId {
        while let Some(Op::Gep { ptr, index, .. }) = self.def_op.get(&v) {
            if !self.is_const_zero(*index) {
                break;
            }
            v = *ptr;
        }
        v
    }

    /// Whether `v` is the integer constant 0, tracing through the sign-/zero-extend casts lowering
    /// inserts on a loop index (`gep buf, sext(const.i32 0)` is still the buffer base).
    fn is_const_zero(&self, mut v: ValueId) -> bool {
        loop {
            match self.def_op.get(&v) {
                Some(Op::ConstInt(0, _)) => return true,
                Some(Op::Cast(_, inner, _)) => v = *inner,
                _ => return false,
            }
        }
    }

    /// Differentiate one recognized kernel call. `args`/`result` are the forward (old) value ids.
    pub(crate) fn diff_kernel_call(
        &mut self,
        func: Symbol,
        raw_args: &[ValueId],
        result: Option<ValueId>,
    ) -> Result<(), String> {
        let (kind, par) = self
            .syms
            .kernel(func)
            .ok_or_else(|| "autodiff: kernel call has no VJP rule".to_string())?;
        let want = kind.arity();
        if raw_args.len() != want {
            return Err(format!(
                "autodiff: call to `{}` has {} argument(s), expected {} — not the recognized \
                 kernel ABI",
                self.it.resolve(func),
                raw_args.len(),
                want
            ));
        }
        // Normalize every buffer operand to its base alloca/param so the buffer-adjoint bookkeeping
        // is keyed consistently regardless of which whole-buffer pointer form lowering chose. (The
        // scalar `result` of a reduction is never a gep, so it needs no canonicalization.) Kept on
        // the stack rather than allocated once per tape call.
        let mut normalized = [ValueId(0); MAX_ARITY];
        for (arg, &raw) in normalized.iter_mut().zip(raw_args) {
            *arg = self.canon(raw);
        }
        let args = &normalized[..want];
        self.reject_live_sub_slice_output(func, kind, args)?;
        match kind {
            Kernel::Sreduce => self.diff_sreduce(args, result)?,
            Kernel::Sgemm => self.diff_sgemm(args, par)?,
            Kernel::SgemmNt => self.diff_sgemm_nt(args, par)?,
            Kernel::SgemmTn => self.diff_sgemm_tn(args, par)?,
            Kernel::SgemmNtAlpha => self.diff_sgemm_nt_alpha(args, par)?,
            Kernel::SgemmNtEpi => self.diff_sgemm_nt_epi(args, par)?,
            Kernel::Sgemv => self.diff_sgemv(args, None, par)?,
            Kernel::SgemvAlpha => self.diff_sgemv(args, Some(args[5]), par)?,
            Kernel::Sgevm => self.diff_sgevm(args, par)?,
            Kernel::Vmath => self.diff_vmath(args)?,
            Kernel::Vmath2 => self.diff_vmath2(args)?,
            Kernel::Velem => self.diff_velem(args)?,
            Kernel::Norm => self.diff_norm(args, par)?,
            Kernel::NormAffine => self.diff_norm_affine(args, par)?,
            Kernel::Bias => self.diff_bias(args)?,
            Kernel::Transpose => self.diff_transpose(args, par)?,
            Kernel::ColSum | Kernel::ColMean | Kernel::ColSumSq => {
                self.diff_colreduce(kind, args)?
            }
            Kernel::Xent => self.diff_xent(args, par)?,
            Kernel::Logsumexp => self.diff_logsumexp(args)?,
            Kernel::Entropy => self.diff_entropy(args)?,
            Kernel::Kldiv => self.diff_kldiv(args)?,
            Kernel::Rope => self.diff_rope(args, par)?,
            Kernel::Embedding => self.diff_embedding(args, par)?,
            Kernel::Attention => self.diff_attention(args)?,
        }
        self.kill_overwritten(kind, args);
        Ok(())
    }

    /// A kernel whose output operand is a SUB-SLICE (`gep(buf, off)`, `off != 0`) of a buffer whose
    /// gradient is live cannot be differentiated: the tape keys one gradient per whole buffer, so
    /// every rule looks up `buf_adj[gep]`, finds nothing (a sub-slice never gets its own gradient)
    /// and returns as if the output were unused — while a load of `buf[off + i]` (or a whole-buffer
    /// consumer of `buf`) has put the real adjoint under `buf`. The kernel's inputs would come back
    /// with a silently ZERO gradient at exit 0. Refuse, naming the kernel.
    fn reject_live_sub_slice_output(
        &self,
        func: Symbol,
        kind: Kernel,
        args: &[ValueId],
    ) -> Result<(), String> {
        let Some(oi) = kind.out_index() else {
            return Ok(());
        };
        let out = args[oi];
        let base = self.store_base(out);
        // A slice of a region's captured buffer is keyed on its own (`Vjp::region_setup`).
        if base != out && self.buf_adj.contains_key(&base) && !self.buf_adj.contains_key(&out) {
            return Err(format!(
                "autodiff: `{}` writes a sub-slice (v{}) of buffer v{}, whose gradient is live; \
                 the buffer tape keeps one gradient per whole buffer, so the kernel's output \
                 adjoint cannot be routed to it — differentiate a whole-buffer spelling",
                self.it.resolve(func),
                out.0,
                base.0
            ));
        }
        Ok(())
    }

    /// The write-after-write rule. A kernel that OVERWRITES its output buffer without reading it
    /// destroys the buffer's previous contents, so the adjoint of those contents is zero over the
    /// region it wrote. When a second, earlier kernel also wrote that buffer, the earlier writer's
    /// rule must see that zero — not the later writer's output adjoint, which used to flow straight
    /// into it (a silently wrong gradient for `h = f(x); h = g(y)`). The same holds for an earlier
    /// non-constant scalar store (`h[0] = x[0]; h = g(y)`): without the zero, `diff_store` would route
    /// the kernel's output adjoint to `x[0]`, which the kernel overwrote. Only buffers with another
    /// writer need it; a single writer has no earlier producer to mislead.
    fn kill_overwritten(&mut self, kind: Kernel, args: &[ValueId]) {
        let Some(oi) = kind.out_index() else {
            return;
        };
        let out = args[oi];
        // In a region's iteration, a write into a captured buffer overwrote what the forward held
        // there before the region: that slice's adjoint stops here.
        let has_other_writer = self.multi_written.contains(&out)
            || self.store_written.contains(&out)
            || self.region_written(out);
        if !has_other_writer || self.kernel_reads_output(kind, args) {
            return;
        }
        let Some(g) = self.buf_adj.get(&out).copied() else {
            return;
        };
        if let Some(n) = self.kernel_out_extent(kind, args) {
            let z = self.cf32(0.0);
            self.fill(g, n, z);
        }
    }

    /// Whether a kernel call READS its own output buffer: it is also one of the input operands
    /// (an in-place kernel), or a GEMM accumulates into it (`beta != 0`, or a beta that is not a
    /// compile-time constant, conservatively).
    fn kernel_reads_output(&self, kind: Kernel, args: &[ValueId]) -> bool {
        let Some(oi) = kind.out_index() else {
            return false;
        };
        let out = args[oi];
        // A velem whose op reads no `y` (no USE_Y and neither non-affine compute mode) may name the
        // output as the placeholder `y`; that operand is never read, so it is not an in-place read
        // (treating it as one would skip the kill and hand an earlier writer this kernel's adjoint).
        let y_unread = kind == Kernel::Velem
            && matches!(
                self.const_i64(args[7]),
                Ok(op) if op & (VE_USE_Y | VE_HADAMARD | VE_DIV) == 0
            );
        if args
            .iter()
            .enumerate()
            .any(|(i, &a)| i != oi && !(y_unread && i == 1) && a == out)
        {
            return true;
        }
        kind.has_beta() && !matches!(self.const_i64(args[6]), Ok(0))
    }

    /// The number of `f32` elements a kernel call writes, in the new value space (`None` for a
    /// reduction). Used to size the gradient of a non-differentiated parameter a kernel writes, and
    /// the region the write-after-write rule zeroes.
    pub(crate) fn kernel_out_extent(&mut self, kind: Kernel, args: &[ValueId]) -> Option<ValueId> {
        let n = match kind {
            Kernel::Sreduce => return None,
            Kernel::Sgemm
            | Kernel::SgemmNt
            | Kernel::SgemmTn
            | Kernel::SgemmNtAlpha
            | Kernel::SgemmNtEpi => self.dims_product(&[args[3], args[5]]),
            Kernel::Sgemv | Kernel::SgemvAlpha => self.dim(args[3]),
            Kernel::Sgevm => self.dim(args[4]),
            Kernel::Vmath => self.dim(args[2]),
            Kernel::Vmath2 | Kernel::Velem => self.dim(args[3]),
            Kernel::Norm | Kernel::Transpose => self.dims_product(&[args[2], args[3]]),
            Kernel::NormAffine => self.dims_product(&[args[4], args[5]]),
            Kernel::Bias => self.dims_product(&[args[3], args[4]]),
            Kernel::ColSum | Kernel::ColMean | Kernel::ColSumSq => self.dim(args[3]),
            Kernel::Xent | Kernel::Kldiv => self.dim(args[3]),
            Kernel::Logsumexp | Kernel::Entropy => self.dim(args[2]),
            Kernel::Rope => {
                let rh = self.dims_product(&[args[3], args[4]]);
                let two = self.cint(2);
                self.mul64(rh, two)
            }
            Kernel::Embedding => self.dims_product(&[args[3], args[4]]),
            Kernel::Attention => self.dims_product(&[args[4], args[5]]),
        };
        Some(n)
    }

    // --- reductions: scalar adjoint -> buffer adjoint ------------------------------------------

    /// `loss = sreduce(x, y, n, op)` produces a scalar. Its adjoint `g = adj[loss]` flows back into a
    /// *buffer* gradient. SUM: `dx = g` (broadcast). DOT (`Σ x·y`): `dx = g·y`, `dy = g·x` — or, when
    /// both operands are the SAME buffer (the sum-of-squares loss), the single `dx = 2g·x`; SUMSQ is
    /// that same rule. SSD (`Σ (x−y)²`): `dx = 2g(x−y)`, `dy = −2g(x−y)` — the MSE-loss backbone.
    /// SUMABS / ABSDIFF: the `sgn` subgradient. MAX / MIN / MAXABS: `g` split evenly over the
    /// elements that attain the extremum (the JAX / PyTorch tie convention).
    fn diff_sreduce(&mut self, args: &[ValueId], result: Option<ValueId>) -> Result<(), String> {
        let loss = result.ok_or("autodiff: sreduce call must produce the scalar loss")?;
        let g = match self.adj.get(&loss).copied() {
            Some(g) => g,
            None => return Ok(()), // loss unused downstream -> zero adjoint
        };
        let op = self.const_i64(args[3])?;
        let n = self.dim(args[2]);
        let (x, y) = (args[0], args[1]);
        let xn = self.remap_v(x);
        let yn = self.remap_v(y);
        match op {
            RED_SUM => {
                if let Some(c) = self.begin(x, n)? {
                    self.fill(c.write, n, g);
                    self.end(c);
                }
            }
            // The self-dot `Σ x²` (and SUMSQ, whose kernel ignores `y`): one op with a repeated
            // operand, `dx = 2g·x`. (`x`/`y` are canon-normalized, so pointer identity is the test.)
            RED_SUMSQ => self.self_square_back(x, xn, g, n)?,
            RED_DOT if x == y => self.self_square_back(x, xn, g, n)?,
            RED_DOT => {
                if let Some(c) = self.begin(x, n)? {
                    self.velem_scale(c.write, yn, g, n);
                    self.end(c);
                }
                if let Some(c) = self.begin(y, n)? {
                    self.velem_scale(c.write, xn, g, n);
                    self.end(c);
                }
            }
            RED_SSD => {
                let two = self.cf32(2.0);
                let g2 = self.fmul(g, two, &F32); // 2g
                let ng2 = self.neg(g2, &F32); // -2g
                if let Some(c) = self.begin(x, n)? {
                    // dx = 2g*x + (-2g)*y
                    self.velem_affine(c.write, xn, yn, g2, ng2, n);
                    self.end(c);
                }
                if let Some(c) = self.begin(y, n)? {
                    // dy = -2g*x + 2g*y
                    self.velem_affine(c.write, xn, yn, ng2, g2, n);
                    self.end(c);
                }
            }
            RED_SUMABS => {
                // loss = g·Σ|x[i]|  ->  dx = g·sgn(x), with `sgn(0) = 0` — the subgradient
                // convention every framework uses (PyTorch's `abs_backward` is `grad * self.sgn()`).
                if let Some(c) = self.begin(x, n)? {
                    self.sign_fill(c.write, xn, None, g, n);
                    self.end(c);
                }
            }
            RED_ABSDIFF => {
                // loss = g·Σ|x[i] - y[i]|  (MAE / SAD)  ->  dx = g·sgn(x-y), dy = -g·sgn(x-y).
                if x == y {
                    // |x-x| is identically 0: the loss does not depend on x at all.
                    return Ok(());
                }
                let ng = self.neg(g, &F32);
                if let Some(c) = self.begin(x, n)? {
                    self.sign_fill(c.write, xn, Some(yn), g, n);
                    self.end(c);
                }
                if let Some(c) = self.begin(y, n)? {
                    self.sign_fill(c.write, xn, Some(yn), ng, n);
                    self.end(c);
                }
            }
            RED_MAX | RED_MIN | RED_MAXABS => {
                if let Some(c) = self.begin(x, n)? {
                    let m = self.remap_v(loss);
                    self.extremum_back(op, c.write, xn, m, g, n);
                    self.end(c);
                }
            }
            other => {
                return Err(format!(
                    "autodiff: no VJP for sreduce op {other} \
                     (supported: SUM, DOT, SSD, SUMSQ, MAX, MIN, MAXABS, SUMABS, ABSDIFF)"
                ))
            }
        }
        Ok(())
    }

    /// `dx += 2g·x` — the gradient of `Σ x²`.
    fn self_square_back(
        &mut self,
        x: ValueId,
        xn: ValueId,
        g: ValueId,
        n: ValueId,
    ) -> Result<(), String> {
        if let Some(c) = self.begin(x, n)? {
            let two = self.cf32(2.0);
            let g2 = self.fmul(g, two, &F32);
            self.velem_scale(c.write, xn, g2, n);
            self.end(c);
        }
        Ok(())
    }

    /// The max / min / max-abs reduction backward: `dst[i] = g·s(x[i])·[key(x[i]) == m] / count`,
    /// where `m` is the forward result, `key` is `x` (or `|x|` for MAXABS), `s` is 1 (or `sgn(x)` for
    /// MAXABS) and `count` is the number of elements attaining `m` — the even split over ties that
    /// `torch.amax` / `jnp.max` use. One mask loop, one `RED_SUMABS` count, one velem scale.
    fn extremum_back(
        &mut self,
        op: i64,
        dst: ValueId,
        x: ValueId,
        m: ValueId,
        g: ValueId,
        n: ValueId,
    ) {
        let zero = self.cf32(0.0);
        let one = self.cf32(1.0);
        let none = self.cf32(-1.0);
        let lp = self.open_loop(n);
        let i = lp.idx;
        let xi = self.load_at(x, i);
        let v = if op == RED_MAXABS {
            let nx = self.neg(xi, &F32);
            let isneg = self.b.build(MirType::I1, Op::Cmp(CmpOp::Folt, xi, zero));
            let ax = self.b.build(F32, Op::Select(isneg, nx, xi));
            let hit = self.b.build(MirType::I1, Op::Cmp(CmpOp::Foeq, ax, m));
            let ispos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, zero));
            let lo = self.b.build(F32, Op::Select(isneg, none, zero));
            let sg = self.b.build(F32, Op::Select(ispos, one, lo));
            self.b.build(F32, Op::Select(hit, sg, zero))
        } else {
            let hit = self.b.build(MirType::I1, Op::Cmp(CmpOp::Foeq, xi, m));
            self.b.build(F32, Op::Select(hit, one, zero))
        };
        self.store_at(dst, i, v);
        self.close_loop(lp);
        // The number of attaining elements: Σ|mask| (MAXABS masks carry the sign).
        let count = self.row_reduce_at(dst, dst, n, RED_SUMABS);
        let scale = self.fdiv(g, count, &F32);
        self.velem_scale(dst, dst, scale, n);
    }

    /// Emit `dst[i] = coef · sgn(a[i] - b[i])` (or `sgn(a[i])` when `b` is `None`) over `n`
    /// elements, with `sgn(0) = 0` — the subgradient of `|.|`. One synthesized counted loop; there
    /// is no `velem`/`sreduce` compute mode for a sign mask, so this is the cheapest correct form.
    fn sign_fill(
        &mut self,
        dst: ValueId,
        a: ValueId,
        b: Option<ValueId>,
        coef: ValueId,
        n: ValueId,
    ) {
        let zero = self.cf32(0.0);
        let ncoef = self.neg(coef, &F32);
        let lp = self.open_loop(n);
        let i = lp.idx;
        let av = self.load_at(a, i);
        let d = match b {
            Some(bb) => {
                let bv = self.load_at(bb, i);
                self.b.build(F32, Op::Bin(BinOp::FSub, av, bv))
            }
            None => av,
        };
        let pos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, d, zero));
        let neg = self.b.build(MirType::I1, Op::Cmp(CmpOp::Folt, d, zero));
        let lo = self.b.build(F32, Op::Select(neg, ncoef, zero));
        let v = self.b.build(F32, Op::Select(pos, coef, lo));
        self.store_at(dst, i, v);
        self.close_loop(lp);
    }

    // --- elementwise activation (vmath) --------------------------------------------------------

    /// `vmath(x, out, n, op)` computes `out = f(x)` elementwise. Backward `dx = dout (.) f'(x)` —
    /// see [`Vjp::act_backward`] for the per-op routing (fused `vmath2` backward kernels, kernel
    /// compositions, or a synthesized loop).
    ///
    /// **An in-place activation is refused.** `wukong_vmath_f32` explicitly permits `out == x`, so
    /// `x[i] = log(x[i])` is a legal forward. The buffer tape cannot follow it: one forward buffer gets
    /// one gradient buffer, and every derivative that reads the forward INPUT would read `f(x)`.
    fn diff_vmath(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (x, out) = (args[0], args[1]);
        let dout = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        let op = self.const_i64(args[3])?;
        if out == x {
            return Err(format!(
                "autodiff: vmath VJP cannot differentiate an in-place activation (op {op}: \
                 the output buffer v{} aliases the input); one buffer cannot hold both the \
                 input and the output adjoint, and every derivative that reads x would read \
                 f(x) instead",
                out.0
            ));
        }
        let n = self.dim(args[2]);
        let xn = self.remap_v(x);
        let yn = self.remap_v(out);
        if let Some(c) = self.begin(x, n)? {
            self.act_backward(op, dout, xn, yn, c.write, n)?;
            self.end(c);
        }
        Ok(())
    }

    // --- streaming affine / residual (velem) ---------------------------------------------------

    /// `velem(x, y, out, n, a, b, c, op)` computes `out = act(mode(x, y))`, where the mode is the
    /// affine `a*x + b*y + c`, the Hadamard product `x·y` or the quotient `x/y`, and `act` is the
    /// identity, ReLU or ReLU6. A fused activation is differentiated through its OUTPUT (`relu(z) > 0`
    /// exactly when `z > 0`, and `0 < relu6(z) < 6` exactly when `0 < z < 6`), so no pre-activation
    /// is recomputed; the mode rule then runs on `dz`.
    ///
    /// The dispatch is a WHITELIST: the activation byte and the compute mode are split and each must
    /// be one this rule implements, so an unknown future bit fails closed instead of quietly taking a
    /// rule written for a different kernel (the affine VJP applied to a Hadamard product once
    /// shipped as a silently wrong gradient).
    fn diff_velem(&mut self, args: &[ValueId]) -> Result<(), String> {
        let op = self.const_i64(args[7])?;
        let act = op & 0xFF;
        let mode = op & !(0xFF | VE_USE_Y);
        if !matches!(act, VE_ID | VE_RELU | VE_RELU6) {
            return Err(format!(
                "autodiff: velem VJP has no rule for the fused activation in op {op} (low byte \
                 {act}); only identity ({VE_ID}), ReLU ({VE_RELU}) and ReLU6 ({VE_RELU6}) are \
                 differentiated"
            ));
        }
        if !matches!(mode, 0 | VE_HADAMARD | VE_DIV) {
            return Err(format!(
                "autodiff: velem VJP has no rule for compute mode {mode} (op {op})"
            ));
        }
        let (x, y, out) = (args[0], args[1], args[2]);
        let dout = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        let n = self.dim(args[3]);
        let uses_y = mode != 0 || op & VE_USE_Y != 0;
        let inplace = out == x || (uses_y && out == y);
        let dz = if act == VE_ID {
            dout
        } else {
            if inplace {
                return Err(format!(
                    "autodiff: velem VJP cannot differentiate an in-place map with a fused \
                     activation (op {op}: output buffer v{} aliases an input)",
                    out.0
                ));
            }
            let s = self.scratch(n);
            let outn = self.remap_v(out);
            self.relu_mask_from_output(dout, outn, s, n, act == VE_RELU6);
            s
        };
        match mode {
            0 => self.diff_velem_affine(args, op, dz, n),
            VE_HADAMARD => self.diff_velem_hadamard(args, dz, n),
            _ => self.diff_velem_div(args, dz, n),
        }
    }

    /// The affine form `out = a*x + b*y + c`: linear, so `dx = a*dz`, `dy = b*dz`.
    ///
    /// **In place** (`x[i] = x[i] + r[i]`, the residual spelling): the output buffer IS an input, so
    /// its adjoint becomes — in place — the adjoint of the input it overwrote. The contribution to
    /// the OTHER operand is taken first, while the buffer still holds the output adjoint.
    fn diff_velem_affine(
        &mut self,
        args: &[ValueId],
        op: i64,
        dz: ValueId,
        n: ValueId,
    ) -> Result<(), String> {
        let (x, y, out) = (args[0], args[1], args[2]);
        let (a, b) = (self.remap_v(args[4]), self.remap_v(args[5]));
        let uses_y = op & VE_USE_Y != 0;
        let x_in = out == x;
        let y_in = uses_y && out == y;
        if x_in && y_in {
            // out = (a + b)·out_old + c
            let s = self.fadd(a, b, &F32);
            self.velem_scale(dz, dz, s, n);
            return Ok(());
        }
        if x_in {
            if uses_y {
                if let Some(c) = self.begin(y, n)? {
                    self.velem_scale(c.write, dz, b, n);
                    self.end(c);
                }
            }
            self.velem_scale(dz, dz, a, n);
            return Ok(());
        }
        if y_in {
            if let Some(c) = self.begin(x, n)? {
                self.velem_scale(c.write, dz, a, n);
                self.end(c);
            }
            self.velem_scale(dz, dz, b, n);
            return Ok(());
        }
        if let Some(c) = self.begin(x, n)? {
            self.velem_scale(c.write, dz, a, n);
            self.end(c);
        }
        if uses_y {
            if let Some(c) = self.begin(y, n)? {
                self.velem_scale(c.write, dz, b, n);
                self.end(c);
            }
        }
        Ok(())
    }

    /// The Hadamard product `out = x (.) y` — the gate in every SwiGLU/GeGLU MLP, every attention
    /// mask and every dropout mask. Its VJP is the product rule, `dx = dz (.) y` and
    /// `dy = dz (.) x`: **the forward kernel is its own backward**, one `VE_HADAMARD` call each.
    /// `out = x (.) x` is one op with a repeated operand: `dx = 2·dz(.)x`.
    fn diff_velem_hadamard(
        &mut self,
        args: &[ValueId],
        dz: ValueId,
        n: ValueId,
    ) -> Result<(), String> {
        let (x, y, out) = (args[0], args[1], args[2]);
        self.reject_inplace_velem(x, y, out, "Hadamard-product")?;
        let (xn, yn) = (self.remap_v(x), self.remap_v(y));
        if x == y {
            if let Some(c) = self.begin(x, n)? {
                self.velem_hadamard(c.write, dz, xn, n);
                let two = self.cf32(2.0);
                self.velem_scale(c.write, c.write, two, n);
                self.end(c);
            }
            return Ok(());
        }
        if let Some(c) = self.begin(x, n)? {
            self.velem_hadamard(c.write, dz, yn, n);
            self.end(c);
        }
        if let Some(c) = self.begin(y, n)? {
            self.velem_hadamard(c.write, dz, xn, n);
            self.end(c);
        }
        Ok(())
    }

    /// The elementwise quotient `out = x / y`: `dx = dz / y` and `dy = -(dz (.) out) / y`
    /// (writing `x/y²` as `out/y` keeps the second gradient to one squaring-free pass and reuses the
    /// forward output the tape already has). With a fused ReLU/ReLU6 `out` differs from `x/y` only
    /// where the mask already zeroed `dz`, so `dz (.) out` is still `dz (.) (x/y)`.
    fn diff_velem_div(&mut self, args: &[ValueId], dz: ValueId, n: ValueId) -> Result<(), String> {
        let (x, y, out) = (args[0], args[1], args[2]);
        self.reject_inplace_velem(x, y, out, "elementwise-division")?;
        if x == y {
            return Err(
                "autodiff: velem VJP cannot differentiate `x / x` (the elementwise-division \
                 compute mode with both operands the same buffer)"
                    .to_string(),
            );
        }
        let (yn, outn) = (self.remap_v(y), self.remap_v(out));
        if let Some(c) = self.begin(x, n)? {
            self.velem_div(c.write, dz, yn, n);
            self.end(c);
        }
        if let Some(c) = self.begin(y, n)? {
            self.velem_hadamard(c.write, dz, outn, n); // dz . (x/y)
            self.velem_div(c.write, c.write, yn, n); //   dz·x/y^2
            let neg1 = self.cf32(-1.0);
            self.velem_scale(c.write, c.write, neg1, n);
            self.end(c);
        }
        Ok(())
    }

    /// A `velem` whose output aliases an input destroys the forward value the Hadamard/quotient
    /// VJPs read back (`x *= y` lowers to exactly that). Refuse rather than differentiate against a
    /// buffer whose contents the forward has already overwritten.
    fn reject_inplace_velem(
        &self,
        x: ValueId,
        y: ValueId,
        out: ValueId,
        what: &str,
    ) -> Result<(), String> {
        if out == x || out == y {
            return Err(format!(
                "autodiff: velem VJP cannot differentiate an in-place {what} compute mode \
                 (output buffer v{} aliases an input); the backward needs the un-overwritten input",
                out.0
            ));
        }
        Ok(())
    }

    /// `dst[i] = (0 < out[i] (< 6)) ? dout[i] : 0` — the ReLU / ReLU6 backward read off the
    /// activation OUTPUT (which is exactly as informative as the pre-activation for these two).
    pub(crate) fn relu_mask_from_output(
        &mut self,
        dout: ValueId,
        out: ValueId,
        dst: ValueId,
        n: ValueId,
        six: bool,
    ) {
        let zero = self.cf32(0.0);
        let six_c = self.cf32(6.0);
        let lp = self.open_loop(n);
        let i = lp.idx;
        let o = self.load_at(out, i);
        let d = self.load_at(dout, i);
        let pos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, o, zero));
        let pass = if six {
            let below = self.b.build(MirType::I1, Op::Cmp(CmpOp::Folt, o, six_c));
            self.b.build(F32, Op::Select(below, d, zero))
        } else {
            d
        };
        let v = self.b.build(F32, Op::Select(pos, pass, zero));
        self.store_at(dst, i, v);
        self.close_loop(lp);
    }

    // --- row-wise normalization ----------------------------------------------------------------

    /// `norm(x, out, rows, cols, eps_bits, op)` row-normalizes `x`. The backwards are the runtime's
    /// own fused kernels — `wukong_softmax_bwd_f32` (`dx = y(.)(dy − Σy·dy)`, the same per-row dot +
    /// elementwise tail the tape used to synthesize, bit for bit), `wukong_layernorm_bwd_f32` and
    /// `wukong_rmsnorm_bwd_f32` (statistics recomputed from `x` with the forward's stable centered
    /// variance) — in the `@parallel` spelling when the forward used it. They need no scratch and no
    /// compile-time `cols`, so a symbolic-generic row width differentiates. Log-softmax
    /// (`dx = dy − exp(y)·Σdy`) and L2 (`dx = (1/r)(dy − y·<dy, y>)`) have no runtime backward and
    /// stay small synthesized row loops.
    ///
    /// **In place** (`x = softmax(x)`, the spelling the recognizer matches): softmax and log-softmax
    /// read only `y` and `dy`, so the output adjoint is transformed into the input adjoint in place
    /// (the runtime kernel permits `dx == dy`). LayerNorm / RMSNorm / L2-norm recompute the row
    /// statistics from `x`, which an in-place forward has destroyed, so they refuse.
    fn diff_norm(&mut self, args: &[ValueId], par: bool) -> Result<(), String> {
        let (x, out) = (args[0], args[1]);
        let op = self.const_i64(args[5])?;
        let dy = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        let inplace = out == x;
        if inplace && !matches!(op, NORM_SOFTMAX | NORM_LOGSOFTMAX) {
            return Err(format!(
                "autodiff: norm VJP cannot differentiate an in-place norm (op {op}: the \
                 output buffer v{} aliases the input); one buffer cannot hold both the input \
                 and the output adjoint, and LayerNorm/RMSNorm/L2-norm recompute the row \
                 statistics from x, which the kernel has already overwritten",
                out.0
            ));
        }
        let rows = self.dim(args[2]);
        let cols = self.dim(args[3]);
        let total = self.mul64(rows, cols);
        let xn = self.remap_v(x);
        let y = self.remap_v(out); // the forward normalized output
        match op {
            NORM_SOFTMAX => {
                let f = pick(self.syms.softmax_bwd, self.syms.softmax_bwd_par, par);
                if inplace {
                    self.emit(f, vec![y, dy, dy, rows, cols]);
                } else if let Some(c) = self.begin(x, total)? {
                    self.emit(f, vec![y, dy, c.write, rows, cols]);
                    self.end(c);
                }
            }
            NORM_LOGSOFTMAX => {
                // `y = log softmax(x)`, so `softmax(x) = exp(y)`: dx = dy - exp(y)·Σ_j dy_j per row
                // — PyTorch's `_log_softmax_backward_data` form. One `wukong_vmath_f32` EXP pass
                // recovers the probabilities into a heap scratch (the same `exp1` polynomial the
                // forward used).
                let p = self.scratch(total);
                self.vmath_call(y, p, total, VM_EXP);
                if inplace {
                    self.logsoftmax_rows(dy, p, dy, rows, cols);
                } else if let Some(c) = self.begin(x, total)? {
                    self.logsoftmax_rows(dy, p, c.write, rows, cols);
                    self.end(c);
                }
            }
            NORM_LAYERNORM | NORM_RMSNORM => {
                let eps_bits = self.dim(args[4]);
                // No gamma: a `Ptr`-typed null, the spelling the kernel and the interpreter's
                // marshalling both read as "scale 1".
                let null = self.b.build(PTR, Op::ConstInt(0, PTR));
                let f = if op == NORM_LAYERNORM {
                    pick(self.syms.layernorm_bwd, self.syms.layernorm_bwd_par, par)
                } else {
                    pick(self.syms.rmsnorm_bwd, self.syms.rmsnorm_bwd_par, par)
                };
                if let Some(c) = self.begin(x, total)? {
                    self.emit(f, vec![xn, dy, null, c.write, rows, cols, eps_bits]);
                    self.end(c);
                }
            }
            NORM_L2NORM => {
                // eps rides in as f32 bits in an i64; the synthesized backward needs its value.
                let eps_bits = self.const_i64(args[4])? as u32;
                let eps = self.cf32(f32::from_bits(eps_bits) as f64);
                if let Some(c) = self.begin(x, total)? {
                    // L2 (unit-norm) is RMSNorm's backward with the `1/n` divisor set to 1:
                    // `r = sqrt(Σx² + eps)`, dx = (1/r)(dy - y·<dy, y>) — the
                    // `torch.nn.functional.normalize` gradient.
                    let nf = self.cf32(1.0);
                    self.scale_norm_back(dy, c.write, xn, y, rows, cols, eps, nf);
                    self.end(c);
                }
            }
            other => {
                return Err(format!(
                    "autodiff: no VJP for norm op {other} \
                     (supported: SOFTMAX, LAYERNORM, RMSNORM, LOGSOFTMAX, L2NORM)"
                ))
            }
        }
        Ok(())
    }

    /// `norm_affine(x, out, gamma, beta, rows, cols, eps_bits, op)` computes
    /// `out[i,j] = xhat[i,j]·gamma[j] + beta[j]` (LayerNorm or RMSNorm `xhat`). `gamma`/`beta` are
    /// `cols` long and shared across rows; either may be a null pointer (a `Ptr`-typed
    /// `ConstInt(0)`), meaning scale 1 / no shift. The three gradients:
    /// - `dbeta[j]  = Σ_i dy[i,j]`            — one `wukong_colsum_f32`;
    /// - `dgamma[j] = Σ_i dy[i,j]·xhat[i,j]`  — `xhat` recomputed by the forward `wukong_norm_f32`,
    ///   one Hadamard, one column sum;
    /// - `dx` — the runtime's `wukong_{layernorm,rmsnorm}_bwd_f32`, which take `gamma` directly.
    ///
    /// The only scratch is the `rows·cols` `xhat`, on the heap and only when `dgamma` is wanted: the
    /// three stack `alloca`s (~4.7 MB for a 512x768 block) the synthesized version needed were a
    /// stack overflow waiting for the native `--train` path.
    fn diff_norm_affine(&mut self, args: &[ValueId], par: bool) -> Result<(), String> {
        let (x, out, gamma, beta) = (args[0], args[1], args[2], args[3]);
        let op = self.const_i64(args[7])?;
        if op != NORM_LAYERNORM && op != NORM_RMSNORM {
            return Err(format!(
                "autodiff: no VJP for norm_affine op {op} (the affine kernel implements only \
                 LAYERNORM ({NORM_LAYERNORM}) and RMSNORM ({NORM_RMSNORM}))"
            ));
        }
        if out == x {
            return Err(
                "autodiff: norm_affine VJP cannot differentiate an in-place norm (output aliases \
                 the input); the backward recomputes the row statistics from x"
                    .to_string(),
            );
        }
        let dy = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        let rows = self.dim(args[4]);
        let cols = self.dim(args[5]);
        let total = self.mul64(rows, cols);
        let eps_bits = self.dim(args[6]);
        let xn = self.remap_v(x);
        let gn = self.remap_v(gamma); // a replayed null stays null
        if !self.is_null_ptr(beta) {
            if let Some(c) = self.begin(beta, cols)? {
                self.colsum(dy, c.write, rows, cols);
                self.end(c);
            }
        }
        if !self.is_null_ptr(gamma) {
            if let Some(c) = self.begin(gamma, cols)? {
                let xh = self.scratch(total);
                let opv = self.cint(op);
                let f = self.syms.norm;
                self.emit(f, vec![xn, xh, rows, cols, eps_bits, opv]);
                self.velem_hadamard(xh, dy, xh, total);
                self.colsum(xh, c.write, rows, cols);
                self.end(c);
            }
        }
        if let Some(c) = self.begin(x, total)? {
            let f = if op == NORM_LAYERNORM {
                pick(self.syms.layernorm_bwd, self.syms.layernorm_bwd_par, par)
            } else {
                pick(self.syms.rmsnorm_bwd, self.syms.rmsnorm_bwd_par, par)
            };
            self.emit(f, vec![xn, dy, gn, c.write, rows, cols, eps_bits]);
            self.end(c);
        }
        Ok(())
    }

    /// Whether `v` is a `Ptr`-typed null (`ConstInt(0)`) — how lowering spells an absent operand.
    pub(crate) fn is_null_ptr(&self, v: ValueId) -> bool {
        matches!(self.def_op.get(&v), Some(Op::ConstInt(0, _)))
    }

    /// Log-softmax rows: `dst[r, j] = dy[r, j] - p[r, j]·Σ_k dy[r, k]` (`p = exp(y)`). `dst` may be
    /// `dy` (the in-place forward): each row's sum is taken before any of the row is written.
    fn logsoftmax_rows(
        &mut self,
        dy: ValueId,
        p: ValueId,
        dst: ValueId,
        rows: ValueId,
        cols: ValueId,
    ) {
        let outer = self.open_loop(rows);
        let r = outer.idx;
        let rc = self.mul64(r, cols);
        let s = self.row_reduce(dy, dy, rc, cols, RED_SUM); // Σ_j dy_j
        let inner = self.open_loop(cols);
        let j = inner.idx;
        let idx = self.b.build(I64T, Op::Bin(BinOp::Add, rc, j));
        let dyv = self.load_at(dy, idx);
        let pv = self.load_at(p, idx);
        let ps = self.fmul(pv, s, &F32);
        let dxv = self.b.build(F32, Op::Bin(BinOp::FSub, dyv, ps));
        self.store_at(dst, idx, dxv);
        self.close_loop(inner);
        self.close_loop(outer);
    }

    /// L2-norm backward (`nf = 1`; the same emitter serves any root-of-scaled-sum-of-squares norm):
    /// with `r = sqrt(Σx²/nf + eps)`, `y = x/r`, `dx = (1/r) (dy - y * (<dy, y>/nf))` per row. The
    /// statistic is a sum of squares, with no mean to cancel against, so this form is stable.
    #[allow(clippy::too_many_arguments)]
    fn scale_norm_back(
        &mut self,
        dy: ValueId,
        dx: ValueId,
        x: ValueId,
        y: ValueId,
        rows: ValueId,
        cols: ValueId,
        eps: ValueId,
        nf: ValueId,
    ) {
        let one = self.cf32(1.0);
        let outer = self.open_loop(rows);
        let r = outer.idx;
        let rc = self.mul64(r, cols);
        let sumsq_x = self.row_reduce(x, x, rc, cols, RED_SUMSQ);
        let ms = self.fdiv(sumsq_x, nf, &F32);
        let mseps = self.fadd(ms, eps, &F32);
        let rr = self.b.build(F32, Op::Sqrt(mseps));
        let inv = self.fdiv(one, rr, &F32);
        let dot_dyy = self.row_reduce(dy, y, rc, cols, RED_DOT);
        let mean_dyy = self.fdiv(dot_dyy, nf, &F32);
        let inner = self.open_loop(cols);
        let j = inner.idx;
        let idx = self.b.build(I64T, Op::Bin(BinOp::Add, rc, j));
        let dyv = self.load_at(dy, idx);
        let yv = self.load_at(y, idx);
        let ymdyy = self.fmul(yv, mean_dyy, &F32);
        let t = self.b.build(F32, Op::Bin(BinOp::FSub, dyv, ymdyy));
        let dxv = self.fmul(inv, t, &F32);
        self.store_at(dx, idx, dxv);
        self.close_loop(inner);
        self.close_loop(outer);
    }

    /// A per-row reduction over the `cols`-wide slice at offset `rc`: `sreduce(&a[rc], &b[rc], cols,
    /// op)`. For SUM/SUMSQ `b` is ignored (pass `a`); for DOT it is the second operand.
    pub(crate) fn row_reduce(
        &mut self,
        a: ValueId,
        b: ValueId,
        rc: ValueId,
        cols: ValueId,
        op: i64,
    ) -> ValueId {
        let ap = self.gep(a, rc);
        let bp = self.gep(b, rc);
        self.row_reduce_at(ap, bp, cols, op)
    }

    /// [`Vjp::row_reduce`] with the two operand pointers already formed.
    pub(crate) fn row_reduce_at(
        &mut self,
        ap: ValueId,
        bp: ValueId,
        cols: ValueId,
        op: i64,
    ) -> ValueId {
        let opv = self.cint(op);
        let f = self.syms.sreduce;
        self.b.build(
            F32,
            Op::Call {
                func: f,
                args: vec![ap, bp, cols, opv],
            },
        )
    }

    // --- buffer-gradient bookkeeping -----------------------------------------------------------

    /// The gradient buffer for forward buffer `buf`, created on first use:
    /// - an existing one;
    /// - the appended, caller-zeroed gradient parameter, for a `wrt` input;
    /// - for a parameter NOT in `wrt`: nothing (a read-only input such as the target labels) — unless
    ///   a recognized kernel WRITES it, in which case its adjoint is what carries the chain to that
    ///   kernel's own inputs, so it gets a zeroed heap scratch sized from the writer's extent;
    /// - for an intermediate (`alloca`) buffer: a zeroed heap scratch of its element count.
    pub(crate) fn grad_target(&mut self, buf: ValueId) -> Result<Option<ValueId>, String> {
        if let Some(&g) = self.buf_adj.get(&buf) {
            return Ok(Some(g));
        }
        // Region mode: a captured buffer's gradient is the pointer the chunk driver handed in.
        if let Some(target) = self.region_grad_target(buf) {
            return target;
        }
        if let Some(&gp) = self.grad_buf.get(&buf) {
            self.buf_adj.insert(buf, gp);
            return Ok(Some(gp));
        }
        if self.fwd.params.contains(&buf) {
            if !self.kernel_out.contains_key(&buf) && self.region_out.contains(&buf) {
                // A parameter only a region writes: size its gradient from the declared extent,
                // in its own element type (a region may compute in `f64`).
                let sized = [F32, MirType::F64]
                    .into_iter()
                    .find_map(|e| Some((self.region_buffer_size(buf, &e)?, e)));
                let (n, elem) = sized.ok_or_else(|| {
                    format!(
                        "autodiff: buffer v{} is written by a @parallel region and read later, \
                         but it is not in the differentiated (`wrt`) set and its extent is not a \
                         fixed float length — add v{} to the wrt list",
                        buf.0, buf.0
                    )
                })?;
                let count = self.cint(n as i64);
                let width = self.cint(if elem == MirType::F64 { 8 } else { 4 });
                let is_float = self.cint(1);
                let f = self.syms.rt_alloc;
                let g = self.b.build(
                    PTR,
                    Op::Call {
                        func: f,
                        args: vec![count, width, is_float],
                    },
                );
                self.scratch.push(g);
                self.buf_adj.insert(buf, g);
                return Ok(Some(g));
            }
            let Some((kind, args)) = self.kernel_out.get(&buf).copied() else {
                return Ok(None); // a non-differentiated input (e.g. the target labels)
            };
            if self.multi_written.contains(&buf) {
                return Err(format!(
                    "autodiff: buffer v{} is written by more than one recognized kernel and is \
                     not in the differentiated (`wrt`) set, so its gradient has no single extent \
                     — add v{} to the wrt list",
                    buf.0, buf.0
                ));
            }
            let n = self
                .kernel_out_extent(kind, args)
                .ok_or("autodiff: internal: a recorded kernel output has no extent")?;
            let g = self.scratch(n);
            self.buf_adj.insert(buf, g);
            return Ok(Some(g));
        }
        let count = *self
            .buf_count
            .get(&buf)
            .ok_or_else(|| format!("autodiff: unknown size for intermediate buffer v{}", buf.0))?;
        let n = self.cint(i64::from(count));
        let gb = self.scratch(n);
        self.buf_adj.insert(buf, gb);
        Ok(Some(gb))
    }

    /// Open one gradient contribution to `buf` (`n` elements). `None` when `buf` is not
    /// differentiated. The first contribution writes straight into the gradient buffer; every later
    /// one (fan-out: a residual connection, a weight two ops share, a buffer the loss also reads)
    /// writes a scratch that [`Vjp::end`] adds in — so no rule ever clobbers an earlier
    /// contribution, and none needs to know whether it is first.
    pub(crate) fn begin(&mut self, buf: ValueId, n: ValueId) -> Result<Option<Contrib>, String> {
        let Some(g) = self.grad_target(buf)? else {
            return Ok(None);
        };
        if self.contributed.insert(buf) {
            Ok(Some(Contrib {
                write: g,
                acc: None,
            }))
        } else {
            let s = self.scratch(n);
            Ok(Some(Contrib {
                write: s,
                acc: Some((g, n)),
            }))
        }
    }

    /// Close a contribution opened by [`Vjp::begin`]: add the scratch into the gradient buffer
    /// (`g = 1·g + 1·s`, one rounding per element) when it was not the first.
    pub(crate) fn end(&mut self, c: Contrib) {
        if let Some((g, n)) = c.acc {
            let one = self.cf32(1.0);
            self.velem_affine(g, g, c.write, one, one, n);
        }
    }

    /// `beta` for a matmul gradient write: 0 (overwrite) on the first contribution to `buf`, 1
    /// (accumulate) afterwards — the GEMM's own accumulation, so no scratch is needed.
    pub(crate) fn beta_for(&mut self, buf: ValueId) -> ValueId {
        let first = !self.contributed.contains(&buf);
        self.cint(if first { 0 } else { 1 })
    }

    /// A zeroed heap scratch of `n` f32 (`wukong_rt_alloc`), released before the gradient returns.
    /// Never call this inside a synthesized loop body: the release at the exit must be dominated by
    /// the allocation.
    pub(crate) fn scratch(&mut self, n: ValueId) -> ValueId {
        let four = self.cint(4);
        let is_float = self.cint(1);
        let f = self.syms.rt_alloc;
        let p = self.b.build(
            PTR,
            Op::Call {
                func: f,
                args: vec![n, four, is_float],
            },
        );
        self.scratch.push(p);
        p
    }

    /// Release every heap scratch (`wukong_rt_free`), in allocation order. Called once, at the exit.
    pub(crate) fn free_scratch(&mut self) {
        let f = self.syms.rt_free;
        for p in std::mem::take(&mut self.scratch) {
            self.b.build_void(Op::Call {
                func: f,
                args: vec![p],
            });
        }
    }

    // --- emit helpers: kernel calls ------------------------------------------------------------

    /// Emit a void call.
    pub(crate) fn emit(&mut self, f: Symbol, args: Vec<ValueId>) {
        self.b.build_void(Op::Call { func: f, args });
    }

    /// Fill `dst[0..n]` with the scalar `v`. A synthesized store loop rather than a `velem` with
    /// `a = 0`: that form computes `0·dst + v` and so turns any non-finite garbage it reads into NaN.
    pub(crate) fn fill(&mut self, dst: ValueId, n: ValueId, v: ValueId) {
        let lp = self.open_loop(n);
        let i = lp.idx;
        self.store_at(dst, i, v);
        self.close_loop(lp);
    }

    /// `dst = coef * src` over `n` elements: `velem(src, src, dst, n, coef, 0, 0, ID)`. `dst` may be
    /// `src` (the kernel permits in-place).
    pub(crate) fn velem_scale(&mut self, dst: ValueId, src: ValueId, coef: ValueId, n: ValueId) {
        let z = self.cf32(0.0);
        let op = self.cint(VE_ID);
        let f = self.syms.velem;
        self.emit(f, vec![src, src, dst, n, coef, z, z, op]);
    }

    /// `dst = a*x + b*y` over `n` elements: `velem(x, y, dst, n, a, b, 0, ID|USE_Y)`.
    pub(crate) fn velem_affine(
        &mut self,
        dst: ValueId,
        x: ValueId,
        y: ValueId,
        a: ValueId,
        b: ValueId,
        n: ValueId,
    ) {
        let z = self.cf32(0.0);
        let op = self.cint(VE_ID | VE_USE_Y);
        let f = self.syms.velem;
        self.emit(f, vec![x, y, dst, n, a, b, z, op]);
    }

    /// `dst = a*x + c` over `n` elements (`y` unread): `velem(x, x, dst, n, a, 0, c, ID)`.
    pub(crate) fn velem_axpc(&mut self, dst: ValueId, x: ValueId, a: f64, c: f64, n: ValueId) {
        let av = self.cf32(a);
        let z = self.cf32(0.0);
        let cv = self.cf32(c);
        let op = self.cint(VE_ID);
        let f = self.syms.velem;
        self.emit(f, vec![x, x, dst, n, av, z, cv, op]);
    }

    /// `dst = u (.) v` over `n` elements: `velem(u, v, dst, n, 1, 1, 0, VE_HADAMARD)`. The kernel
    /// ignores `a`/`b`/`c` in this compute mode, so they are only placeholders.
    pub(crate) fn velem_hadamard(&mut self, dst: ValueId, u: ValueId, v: ValueId, n: ValueId) {
        let one = self.cf32(1.0);
        let z = self.cf32(0.0);
        let op = self.cint(VE_HADAMARD);
        let f = self.syms.velem;
        self.emit(f, vec![u, v, dst, n, one, one, z, op]);
    }

    /// `dst = u / v` over `n` elements: `velem(u, v, dst, n, 1, 1, 0, VE_DIV)`.
    pub(crate) fn velem_div(&mut self, dst: ValueId, u: ValueId, v: ValueId, n: ValueId) {
        let one = self.cf32(1.0);
        let z = self.cf32(0.0);
        let op = self.cint(VE_DIV);
        let f = self.syms.velem;
        self.emit(f, vec![u, v, dst, n, one, one, z, op]);
    }

    /// `wukong_vmath_f32(x, out, n, op)`: the elementwise activation kernel (`out` may be `x`).
    pub(crate) fn vmath_call(&mut self, x: ValueId, out: ValueId, n: ValueId, op: i64) {
        let opv = self.cint(op);
        let f = self.syms.vmath;
        self.emit(f, vec![x, out, n, opv]);
    }

    /// `wukong_vmath2_f32(x, y, out, n, op)`: the two-input kernel (a fused activation backward is
    /// `(x, dy) -> dy·act'(x)`).
    pub(crate) fn vmath2_call(
        &mut self,
        x: ValueId,
        y: ValueId,
        out: ValueId,
        n: ValueId,
        op: i64,
    ) {
        let opv = self.cint(op);
        let f = self.syms.vmath2;
        self.emit(f, vec![x, y, out, n, opv]);
    }

    /// `wukong_colsum_f32(x, out, rows, cols)`: `out[j] = Σ_i x[i*cols + j]` (overwrites `out`).
    pub(crate) fn colsum(&mut self, x: ValueId, out: ValueId, rows: ValueId, cols: ValueId) {
        let f = self.syms.colsum;
        self.emit(f, vec![x, out, rows, cols]);
    }

    /// `dst[r, :] *= dvec[r]` for every row of a `rows x cols` buffer — the per-row scale a vector
    /// adjoint (one value per row: a per-row loss, a log-sum-exp) puts on its row-wise backward.
    pub(crate) fn row_scale(&mut self, dst: ValueId, dvec: ValueId, rows: ValueId, cols: ValueId) {
        let lp = self.open_loop(rows);
        let r = lp.idx;
        let s = self.load_at(dvec, r);
        let rc = self.mul64(r, cols);
        let p = self.gep(dst, rc);
        self.velem_scale(p, p, s, cols);
        self.close_loop(lp);
    }

    // --- emit helpers: synthesized loops -------------------------------------------------------

    /// Open a counted loop `for idx in 0..n`. Emits the back-edge condition and switches the builder
    /// to the (empty) body block, with `idx` available; pair with [`Vjp::close_loop`].
    pub(crate) fn open_loop(&mut self, n: ValueId) -> Loop {
        let zero = self.cint(0);
        let header = self.b.new_block();
        let body = self.b.new_block();
        let exit = self.b.new_block();
        let i_h = self.b.block_param(header, I64T);
        let i_b = self.b.block_param(body, I64T);
        // current block -> br header(0)
        self.b.br(header, vec![zero]);
        // header: cond = i < n; cond_br body(i) / exit
        self.b.switch_to(header);
        let cond = self.b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i_h, n));
        self.b.cond_br(cond, body, vec![i_h], exit, vec![]);
        // body (caller fills it):
        self.b.switch_to(body);
        Loop {
            header,
            exit,
            idx: i_b,
        }
    }

    /// Close the loop opened by [`Vjp::open_loop`]: append `idx += 1; br header(idx+1)` and switch
    /// the builder to the exit block. Assumes the body is straight-line (current block is the body).
    pub(crate) fn close_loop(&mut self, lp: Loop) {
        let one = self.cint(1);
        let i1 = self.b.build(I64T, Op::Bin(BinOp::Add, lp.idx, one));
        self.b.br(lp.header, vec![i1]);
        self.b.switch_to(lp.exit);
    }

    // --- small helpers -------------------------------------------------------------------------

    /// `&base[idx]` as an `f32` element pointer.
    pub(crate) fn gep(&mut self, base: ValueId, idx: ValueId) -> ValueId {
        self.b.build(
            PTR,
            Op::Gep {
                ptr: base,
                index: idx,
                elem: F32,
            },
        )
    }

    /// Load `base[idx]` as an `f32` (`gep` then `load`).
    pub(crate) fn load_at(&mut self, base: ValueId, idx: ValueId) -> ValueId {
        let p = self.gep(base, idx);
        self.b.build(F32, Op::Load(p, F32))
    }

    /// Store `val` to `base[idx]` (`f32`).
    pub(crate) fn store_at(&mut self, base: ValueId, idx: ValueId, val: ValueId) {
        let p = self.gep(base, idx);
        self.b.build_void(Op::Store { ptr: p, value: val });
    }

    /// The new-function value for a forward (old) value.
    pub(crate) fn remap_v(&self, v: ValueId) -> ValueId {
        self.fwd_to_new[&v]
    }

    /// A kernel dimension operand in the new value space, widened to `i64` if lowering passed a
    /// narrower integer — every emitted kernel ABI and every synthesized loop bound is `i64`.
    pub(crate) fn dim(&mut self, v: ValueId) -> ValueId {
        let nv = self.remap_v(v);
        let ty = self.b.value_type(nv).clone();
        if ty != I64T && ty.is_int() {
            self.b.build(I64T, Op::Cast(CastKind::SExt, nv, I64T))
        } else {
            nv
        }
    }

    /// The product of several kernel dimensions, as an `i64` in the new value space.
    pub(crate) fn dims_product(&mut self, dims: &[ValueId]) -> ValueId {
        let mut acc = self.dim(dims[0]);
        for &d in &dims[1..] {
            let dv = self.dim(d);
            acc = self.mul64(acc, dv);
        }
        acc
    }

    /// `a * b` in `i64`.
    pub(crate) fn mul64(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.b.build(I64T, Op::Bin(BinOp::Mul, a, b))
    }

    /// Read a compile-time `i64` constant operand (an op code). Errors if not a constant.
    pub(crate) fn const_i64(&self, v: ValueId) -> Result<i64, String> {
        match self.def_op.get(&v) {
            Some(Op::ConstInt(c, _)) => Ok(*c as i64),
            _ => Err(format!(
                "autodiff: kernel operand v{} must be a compile-time constant",
                v.0
            )),
        }
    }

    pub(crate) fn cint(&mut self, x: i64) -> ValueId {
        self.b.build(I64T, Op::ConstInt(x as i128, I64T))
    }

    pub(crate) fn cf32(&mut self, x: f64) -> ValueId {
        self.b.build(F32, Op::ConstFloat(x, F32))
    }
}

/// What the reverse walk does at a call to one of the ML I/O and randomness runtime entries, none of
/// which is a tensor kernel with a VJP. The decision is explicit per entry so a new one is a choice,
/// not a fall-through:
/// * the typed reads (`wukong_rt_read_*`, `wukong_rt_st_read`, `wukong_rt_npy_read`,
///   `wukong_rt_gguf_read`) and the GGUF table queries (`wukong_rt_gguf_query`), `argsort` and
///   `sample_categorical` return an `i64` and are left to the scalar path: what they produce — file
///   data, integer indices — is constant with respect to every input, so it carries no gradient;
/// * [`MlCallGradient::Constant`]: `wukong_rand_fill_f32` writes counter-based random numbers that
///   depend on no input — skipped exactly like a constant store (`[0.0; N]`), never refused;
/// * [`MlCallGradient::Refuse`]: `wukong_topk_f32` writes `vals`, a differentiable function of `x`
///   (`torch.topk` scatters the gradient back to the selected indices), and it has no VJP rule yet —
///   so differentiating through it is an error rather than a silently dropped gradient.
pub(crate) enum MlCallGradient {
    Constant,
    Refuse,
}

/// The [`MlCallGradient`] of the runtime entry `name`, or `None` for any other callee.
pub(crate) fn ml_call_gradient(name: &str) -> Option<MlCallGradient> {
    match name {
        "wukong_rand_fill_f32" => Some(MlCallGradient::Constant),
        "wukong_topk_f32" => Some(MlCallGradient::Refuse),
        // The decode builtins write their output buffer and return only a status, so the
        // value-returning path would skip them and drop the gradient through that buffer: refuse.
        "wukong_attention_kv_f32"
        | "wukong_attention_kv_f32_parallel"
        | "wukong_attention_kv_batch_f32"
        | "wukong_attention_kv_batch_f32_parallel"
        | "wukong_kv_append_f32"
        | "wukong_sgemv_bf16"
        | "wukong_sgemv_bf16_parallel"
        | "wukong_sgemv_f16"
        | "wukong_sgemv_f16_parallel"
        | "wukong_quantize_q8_0_f32"
        | "wukong_quantize_q4_0_f32"
        | "wukong_gemv_q8_0_f32"
        | "wukong_gemv_q8_0_f32_parallel"
        | "wukong_gemv_q4_0_f32"
        | "wukong_gemv_q4_0_f32_parallel"
        | "wukong_dequant_i4_f32"
        | "wukong_dequant_u4_f32" => Some(MlCallGradient::Refuse),
        _ => None,
    }
}
