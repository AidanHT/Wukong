//! Tensor-tape VJPs: differentiating a straight-line sequence of recognized **kernel calls**.
//!
//! Wukong lowers tensor ops to runtime-kernel calls (`wukong_sgemm_nt` = `nn.Linear`,
//! `wukong_sgemm` = plain matmul, `wukong_vmath_f32` = elementwise activation,
//! `wukong_sreduce_f32` = reduction, `wukong_velem_f32` = streaming affine / Hadamard / quotient,
//! `wukong_bias_bcast_f32` = broadcast bias add, `wukong_norm_f32` = row-wise softmax / LayerNorm /
//! RMSNorm / log-softmax / L2-norm, `wukong_norm_affine_f32` = the same with a learned gamma/beta;
//! each also in its `@parallel` spelling) — the interpreter runs the *same* kernels, so a
//! forward MIR function can be a tape of these calls plus a final scalar loss. Where the scalar core
//! ([`crate::Vjp`]) tracks an adjoint per SSA value, this module tracks an adjoint per **buffer**
//! (memory), the PyTorch-style picture tensor ops need: each forward buffer has a gradient buffer,
//! set by a consumer's VJP and read by the producer's. The backward emits *more kernel calls* (so the
//! matmul gradients ride the same tuned GEMM) plus small synthesized loops for the transpose and the
//! elementwise (Hadamard) combines that no single existing kernel covers.
//!
//! Buffers are differentiated under the same single-block contract as the scalar core; the only new
//! control flow is the counted loops this module synthesizes — one for an algebraic activation
//! backward or a sign mask, a nested rows x cols pair for the norm backwards. (So the *emitted*
//! gradient function is multi-block: it is not itself re-differentiable, which `grad` now says by
//! name.) The plain-matmul adjoints synthesize no loop at all: they are the runtime's own
//! transposed GEMM spellings (`wukong_sgemm_nt` / `wukong_sgemm_tn`).

use crate::Vjp;
use wukong_mir::{BinOp, BlockId, CastKind, CmpOp, MirType, Op, ValueId};
use wukong_span::{Interner, Symbol};

// --- runtime kernel op codes (mirrored from wukong_runtime; the interp dispatches the same) ------
const VM_EXP: i64 = 0;
const VM_LOG: i64 = 1;
const VM_TANH: i64 = 2;
const VM_SIGMOID: i64 = 3;
const VM_RELU: i64 = 4;
const VM_SILU: i64 = 5;
const VM_GELU: i64 = 6;
const VM_ELU: i64 = 7;
const VM_LEAKYRELU: i64 = 8;
const VM_SOFTPLUS: i64 = 9;
const VM_SELU: i64 = 11;
const VM_TANHSHRINK: i64 = 12;
const VM_HARDSIGMOID: i64 = 13;
const VM_HARDSWISH: i64 = 14;
const VM_EXP2: i64 = 18;
const VM_LOG2: i64 = 19;
const VM_SINH: i64 = 20;
const VM_ASINH: i64 = 22;
const VM_ACOSH: i64 = 23;
const VM_ATANH: i64 = 24;
const VM_ATAN: i64 = 25;
const VM_EXPM1: i64 = 26;
const VM_LOG1P: i64 = 27;
const VM_EXP10: i64 = 28;
const VM_LOG10: i64 = 29;
const VM_SOFTSIGN: i64 = 30;
const VM_TAN: i64 = 32;
const VM_ASIN: i64 = 33;
const VM_ACOS: i64 = 34;
const VM_CBRT: i64 = 35;
/// The leaky-ReLU negative slope baked into `wukong_runtime::vmath` (`LEAKY_ALPHA`). Mirrored, so a
/// change there must be mirrored here or the backward stops matching the forward.
const LEAKY_ALPHA: f64 = 0.01;
/// SELU's `lambda`/`alpha` (`wukong_runtime::vmath::SELU_LAMBDA`/`SELU_ALPHA`), mirrored.
const SELU_LAMBDA: f64 = 1.050_701;
const SELU_ALPHA: f64 = 1.673_263_2;
/// `1/6`, the hard-sigmoid / hard-swish scale (`wukong_runtime::vmath::INV6`), mirrored.
const INV6: f64 = 1.0 / 6.0;
// `wukong_vmath2_f32(x, dy, dx, n, op)` *backward* op codes: the kernel fuses the upstream multiply
// into a 256-bit activation derivative (`dx = dy · act'(x)`), so a whole activation-backward pass is
// one call — and is bit-identical with the forward (it reuses the same sigmoid/tanh polynomials). The
// smooth activations whose derivative is a *transcendental* (so no closed form in `x`/`y` alone) ride
// this instead of the synthesized loop the algebraic ones (relu/sigmoid/tanh/exp) use.
const VM2_SILU_BWD: i64 = 3;
const VM2_GELU_BWD: i64 = 4;
const VM2_ELU_BWD: i64 = 7;
const VM2_SOFTPLUS_BWD: i64 = 8;
const RED_DOT: i64 = 0;
const RED_SSD: i64 = 1;
const RED_SUM: i64 = 2;
const RED_SUMSQ: i64 = 3;
const RED_SUMABS: i64 = 9;
const RED_ABSDIFF: i64 = 10;
const NORM_SOFTMAX: i64 = 0;
const NORM_LAYERNORM: i64 = 1;
const NORM_RMSNORM: i64 = 2;
const NORM_LOGSOFTMAX: i64 = 3;
const NORM_L2NORM: i64 = 4;
/// `wukong_bias_bcast_f32`'s "no fused activation" op code (`wukong_runtime::bias::BIAS_ACT_NONE`).
/// Any other value selects a fused `VM_*` activation, which this rule does not model.
const BIAS_ACT_NONE: i64 = -1;
/// velem activation: identity (`out = a*x + b*y + c`).
const VE_ID: i64 = 0;
/// OR'd into a velem op when `y` is read (`b` may be non-zero).
const VE_USE_Y: i64 = 256;
/// velem *compute mode*: the elementwise Hadamard product (`out = act(x · y)`). Mirrored from
/// `wukong_runtime::velem`. It lives ABOVE the low activation byte, so a `op & 0xff` test does not
/// see it — and it is **non-affine**, so the affine VJP below is not a rule for it.
const VE_HADAMARD: i64 = 512;
/// velem *compute mode*: the elementwise quotient (`out = act(x / y)`) — likewise non-affine.
const VE_DIV: i64 = 1024;

const F32: MirType = MirType::F32;
const I64T: MirType = MirType::I64;
const PTR: MirType = MirType::Ptr;

/// Interned symbols for the kernels we recognize (forward) and emit (backward). Interning dedups,
/// so these compare equal to the symbols a forward tape was built with.
pub(crate) struct Syms {
    /// The plain `C = A·B` GEMM. The tape both **emits** it (a matmul input adjoint) and now
    /// **recognizes** it on the forward side — mir_build lowers an ordinary `matmul` (no transpose)
    /// to this symbol, so without the forward arm a plain matmul took the generic
    /// buffer-writing-call refusal while the `nn.Linear` spelling differentiated.
    pub sgemm: Symbol,
    /// The `@parallel` plain GEMM — same `(a, b, c, m, k, n, beta)` ABI and identical (fixed
    /// blocking) result, so it differentiates through the very same rule.
    pub sgemm_parallel: Symbol,
    /// `C = Aᵀ·B` (A stored `[k, m]`): the **weight-gradient** GEMM. Emit-only — it is exactly the
    /// "transpose then multiply" the tape used to synthesize as a flat MIR loop with an `SDiv` and
    /// an `SRem` per element plus an `m·n` scratch alloca. The kernel does the same transpose into
    /// its own scratch and then runs the identical `A·B` microkernel, so the switch is bit-exact.
    pub sgemm_tn: Symbol,
    pub sgemm_nt: Symbol,
    /// The `@parallel` NN GEMM — same `(a, b, c, m, k, n, beta)` ABI and bit-identical (fixed
    /// chunking) result as the serial `sgemm_nt`, so it differentiates through the very same rule.
    /// A matmul embedded in a `@parallel fn` lowers to *this* symbol (mir_build's `lower_for` passes
    /// `parallel_fn`), while single-statement / non-parallel tapes use the serial one — accept both.
    pub sgemm_nt_parallel: Symbol,
    pub vmath: Symbol,
    /// The `@parallel` elementwise activation — same `(x, out, n, op)` ABI and bit-identical
    /// (elementwise, no cross-chunk combine) result as the serial `vmath`, so it differentiates
    /// through the very same rule. An `out[i]=act(x[i])` loop inside a `@parallel fn` lowers to *this*
    /// symbol (mir_build selects it when `parallel_fn`), while single-statement / non-parallel tapes use
    /// the serial one — accept both. Without this the backward pass of every `@parallel`-fn activation
    /// loop fails with an unrecognized-buffer-writing-call error.
    pub vmath_parallel: Symbol,
    pub sreduce: Symbol,
    /// The `@parallel` reduction — same `(x, y, n, op)` ABI and identical (deterministic, fixed
    /// chunking) result as the serial `sreduce`, so it differentiates through the very same rule.
    /// A `.wk` reduction lowered from source produces *this* symbol (the recognizer emits the
    /// parallel kernel), while the hand-built tape tests use the serial one — the tape accepts both.
    pub sreduce_parallel: Symbol,
    pub velem: Symbol,
    /// The `@parallel` streaming map — same `(x, y, out, n, a, b, c, op)` ABI and bit-identical
    /// (elementwise, no cross-chunk combine) result as the serial `velem`, so it differentiates
    /// through the very same rule. A residual-add / saxpy inside a `@parallel fn` lowers to *this*
    /// symbol (mir_build's `emit_velem_call` selects it when `parallel_fn`), while single-statement /
    /// non-parallel tapes use the serial one — the tape accepts both. Without this the backward pass of
    /// every `@parallel`-fn velem loop would fail with an unrecognized-buffer-writing-call error.
    pub velem_parallel: Symbol,
    /// The two-input transcendental kernel; the tape emits its `*_BWD` op codes for the smooth
    /// activation backwards (`dx = dy · act'(x)` in one fused pass).
    pub vmath2: Symbol,
    pub norm: Symbol,
    /// The `@parallel` row-wise norm — same `(x, out, rows, cols, eps_bits, op)` ABI and
    /// bit-identical (each row is independent, no cross-row combine) result as the serial `norm`, so
    /// it differentiates through the very same rule. `emit_norm` selects *this* symbol for every
    /// batched LayerNorm/RMSNorm/softmax inside a `@parallel fn` — the shipped transformer shape —
    /// while single-row / non-parallel tapes use the serial one; accept both. Without it the same
    /// model is differentiable serially and not in parallel.
    pub norm_parallel: Symbol,
    /// `wukong_norm_affine_f32(x, out, gamma, beta, rows, cols, eps_bits, op)` — LayerNorm/RMSNorm
    /// with a learned per-column scale (and optional shift). This is the **shipped transformer
    /// norm**: `emit_norm` picks it the moment a `gamma` appears, so without a rule here the toy
    /// spelling differentiated and the real one did not.
    pub norm_affine: Symbol,
    /// The `@parallel` affine norm — rows are independent, bit-identical to the serial kernel, same
    /// rule (the `_parallel` mirror contract).
    pub norm_affine_parallel: Symbol,
    /// `wukong_bias_bcast_f32(x, b, out, rows, cols, op)` — the broadcast bias add. `y = x·Wᵀ + b`
    /// is the single most common construct in ML; spelled with the bias the recognizer fires this
    /// kernel, so the bias was load-bearing on whether the model could be trained at all.
    pub bias: Symbol,
    /// The `@parallel` broadcast bias add (rows independent, bit-identical, same rule).
    pub bias_parallel: Symbol,
    /// `wukong_colsum_f32(x, out, rows, cols)` — `out[j] = Σ_i x[i·cols + j]`. Emit-only: it is the
    /// bias gradient (`db = colsum(dout)`) and the learned-`gamma`/`beta` gradients.
    pub colsum: Symbol,
}

impl Syms {
    pub fn new(it: &mut Interner) -> Syms {
        Syms {
            sgemm: it.intern("wukong_sgemm"),
            sgemm_parallel: it.intern("wukong_sgemm_parallel"),
            sgemm_tn: it.intern("wukong_sgemm_tn"),
            sgemm_nt: it.intern("wukong_sgemm_nt"),
            sgemm_nt_parallel: it.intern("wukong_sgemm_nt_parallel"),
            vmath: it.intern("wukong_vmath_f32"),
            vmath_parallel: it.intern("wukong_vmath_f32_parallel"),
            sreduce: it.intern("wukong_sreduce_f32"),
            sreduce_parallel: it.intern("wukong_sreduce_f32_parallel"),
            velem: it.intern("wukong_velem_f32"),
            velem_parallel: it.intern("wukong_velem_f32_parallel"),
            vmath2: it.intern("wukong_vmath2_f32"),
            norm: it.intern("wukong_norm_f32"),
            norm_parallel: it.intern("wukong_norm_f32_parallel"),
            norm_affine: it.intern("wukong_norm_affine_f32"),
            norm_affine_parallel: it.intern("wukong_norm_affine_f32_parallel"),
            bias: it.intern("wukong_bias_bcast_f32"),
            bias_parallel: it.intern("wukong_bias_bcast_f32_parallel"),
            colsum: it.intern("wukong_colsum_f32"),
        }
    }
}

/// Map a forward `vmath` activation op code to the `vmath2` *backward* op code that computes its
/// `dx = dy·act'(x)` in one fused pass — or `None` for the algebraic activations (relu/sigmoid/tanh/
/// exp) that the synthesized loop handles without a transcendental.
fn vm2_bwd_code(fwd_op: i64) -> Option<i64> {
    Some(match fwd_op {
        VM_SILU => VM2_SILU_BWD,
        VM_GELU => VM2_GELU_BWD,
        VM_ELU => VM2_ELU_BWD,
        VM_SOFTPLUS => VM2_SOFTPLUS_BWD,
        _ => return None,
    })
}

/// A synthesized counted loop, threaded between [`Vjp::open_loop`] and [`Vjp::close_loop`]. After
/// `open_loop` the builder's current block IS the body, so the caller emits straight into it and
/// `close_loop` appends the back-edge there.
struct Loop {
    header: BlockId,
    exit: BlockId,
    /// The `i64` loop index, live in the body block.
    idx: ValueId,
}

impl<'a> Vjp<'a> {
    /// Is `func` a tensor kernel this module differentiates?
    pub(crate) fn is_kernel(&self, func: Symbol) -> bool {
        func == self.syms.sgemm
            || func == self.syms.sgemm_parallel
            || func == self.syms.sgemm_nt
            || func == self.syms.sgemm_nt_parallel
            || func == self.syms.vmath
            || func == self.syms.vmath_parallel
            || func == self.syms.sreduce
            || func == self.syms.sreduce_parallel
            || func == self.syms.velem
            || func == self.syms.velem_parallel
            || func == self.syms.norm
            || func == self.syms.norm_parallel
            || func == self.syms.norm_affine
            || func == self.syms.norm_affine_parallel
            || func == self.syms.bias
            || func == self.syms.bias_parallel
    }

    /// Which operand of a recognized kernel's ABI is the buffer it **writes** — `None` for a
    /// reduction, whose result is a scalar. Mirrors [`Vjp::kernel_arity`] and
    /// [`Vjp::diff_kernel_call`] one-for-one; only [`Vjp::scan_kernel_outputs`] reads it.
    fn kernel_out_index(&self, func: Symbol) -> Option<usize> {
        if func == self.syms.sreduce || func == self.syms.sreduce_parallel {
            None
        } else if func == self.syms.vmath
            || func == self.syms.vmath_parallel
            || func == self.syms.norm
            || func == self.syms.norm_parallel
            || func == self.syms.norm_affine
            || func == self.syms.norm_affine_parallel
        {
            Some(1) // vmath(x, OUT, ..) / norm(x, OUT, ..) / norm_affine(x, OUT, ..)
        } else {
            Some(2) // sgemm[_nt](a, b, C, ..) / velem(x, y, OUT, ..) / bias(x, b, OUT, ..)
        }
    }

    /// Record every buffer a recognized kernel WRITES in the forward pass (canonicalized, so the
    /// `gep(buf, 0)` and bare-alloca spellings key the same entry).
    ///
    /// `diff_load` needs this to tell two cases apart that look identical to it: a scalar read of a
    /// **read-only input** whose gradient may legitimately be dropped (the target labels), and a
    /// scalar read of a **kernel output**, where dropping the adjoint breaks the chain to that
    /// kernel's own inputs and hands back an all-zero gradient with no diagnostic.
    pub(crate) fn scan_kernel_outputs(&mut self) {
        let insts = &self.entry_block().insts;
        let mut written = Vec::new();
        for inst in insts {
            let Op::Call { func, args } = &inst.op else {
                continue;
            };
            if !self.is_kernel(*func) || self.kernel_arity(*func) != Some(args.len()) {
                continue; // a mis-declared ABI: `diff_kernel_call` reports it with the arity error
            }
            if let Some(i) = self.kernel_out_index(*func) {
                written.push(self.canon(args[i]));
            }
        }
        self.kernel_written.extend(written);
    }

    /// Canonicalize a kernel buffer argument to the base buffer it addresses: peel a whole-buffer
    /// `gep(buf, 0)` (the form MIR lowering emits for *some* kernel operands — e.g. `vmath`'s output —
    /// while others, like `sreduce`'s input, pass the bare `alloca`). Without this, the two forms key
    /// *different* buffer-adjoint entries and the gradient chain silently breaks between the ops. A
    /// non-zero-offset gep is a genuine sub-slice and is left alone. Dim/op-code operands are constants
    /// (not geps), so canonicalizing the whole argument list is a no-op for them.
    fn canon(&self, mut v: ValueId) -> ValueId {
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

    /// The argument count of each recognized kernel's ABI, matching the operand lists the VJP rules
    /// below index. `is_kernel` recognizes a call by SYMBOL NAME alone, and a `.wk` program may
    /// declare that name in an `extern "C"` block with any signature it likes — so the arity must be
    /// checked before those rules index `args[..]` raw, or a mismatched declaration panics the
    /// compiler with an index-out-of-bounds ICE instead of producing a diagnostic.
    /// The arms mirror [`Vjp::diff_kernel_call`]'s dispatch chain one-for-one, so a new kernel arm
    /// there is a visible hole here.
    fn kernel_arity(&self, func: Symbol) -> Option<usize> {
        if func == self.syms.sreduce || func == self.syms.sreduce_parallel {
            Some(4) // (x, y, n, op)
        } else if func == self.syms.sgemm_nt
            || func == self.syms.sgemm_nt_parallel
            || func == self.syms.sgemm
            || func == self.syms.sgemm_parallel
        {
            Some(7) // (a, b, c, m, k, n, beta)
        } else if func == self.syms.vmath || func == self.syms.vmath_parallel {
            Some(4) // (x, out, n, op)
        } else if func == self.syms.velem || func == self.syms.velem_parallel {
            Some(8) // (x, y, out, n, a, b, c, op)
        } else if func == self.syms.norm || func == self.syms.norm_parallel {
            Some(6) // (x, out, rows, cols, eps_bits, op)
        } else if func == self.syms.norm_affine || func == self.syms.norm_affine_parallel {
            Some(8) // (x, out, gamma, beta, rows, cols, eps_bits, op)
        } else if func == self.syms.bias || func == self.syms.bias_parallel {
            Some(6) // (x, b, out, rows, cols, op)
        } else {
            None
        }
    }

    /// Differentiate one recognized kernel call. `args`/`result` are the forward (old) value ids.
    pub(crate) fn diff_kernel_call(
        &mut self,
        func: Symbol,
        raw_args: &[ValueId],
        result: Option<ValueId>,
    ) -> Result<(), String> {
        let want = self
            .kernel_arity(func)
            .ok_or_else(|| "autodiff: kernel call has no VJP rule".to_string())?;
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
        // scalar `result` of a reduction is never a gep, so it needs no canonicalization.)
        // The largest recognized ABI is velem's eight operands (see kernel_arity). Keep this
        // short-lived normalization on the stack instead of allocating once per tape call.
        let mut normalized = [ValueId(0); 8];
        let args = &mut normalized[..want];
        for (arg, &raw) in args.iter_mut().zip(raw_args) {
            *arg = self.canon(raw);
        }
        if func == self.syms.sreduce || func == self.syms.sreduce_parallel {
            self.diff_sreduce(args, result)
        } else if func == self.syms.sgemm_nt || func == self.syms.sgemm_nt_parallel {
            self.diff_sgemm_nt(args)
        } else if func == self.syms.sgemm || func == self.syms.sgemm_parallel {
            self.diff_sgemm(args)
        } else if func == self.syms.vmath || func == self.syms.vmath_parallel {
            self.diff_vmath(args)
        } else if func == self.syms.velem || func == self.syms.velem_parallel {
            self.diff_velem(args)
        } else if func == self.syms.norm || func == self.syms.norm_parallel {
            self.diff_norm(args)
        } else if func == self.syms.norm_affine || func == self.syms.norm_affine_parallel {
            self.diff_norm_affine(args)
        } else if func == self.syms.bias || func == self.syms.bias_parallel {
            self.diff_bias(args)
        } else {
            Err("autodiff: kernel call has no VJP rule".to_string())
        }
    }

    // --- reductions: scalar adjoint -> buffer adjoint ------------------------------------------

    /// `loss = sreduce(x, y, n, op)` produces the scalar loss. Its adjoint `g = adj[loss]` flows back
    /// into a *buffer* gradient. SUM: `dx = g` (broadcast). DOT (`sum x[i]y[i]`): `dx = g*y`,
    /// `dy = g*x` — or, when both operands are the SAME buffer (the sum-of-squares loss), the single
    /// `dx = 2g*x`. SSD (`sum (x-y)^2`): `dx = 2g(x-y)`, `dy = -2g(x-y)` — the MSE-loss backbone.
    /// Every case is streaming-affine, so each gradient is a single velem call.
    fn diff_sreduce(&mut self, args: &[ValueId], result: Option<ValueId>) -> Result<(), String> {
        let loss = result.ok_or("autodiff: sreduce call must produce the scalar loss")?;
        let g = match self.adj.get(&loss).copied() {
            Some(g) => g,
            None => return Ok(()), // loss unused downstream -> zero adjoint
        };
        let op = self.const_i64(args[3])?;
        let n = self.remap_v(args[2]);
        match op {
            RED_SUM => {
                let x = args[0];
                if let Some(gx) = self.grad_target(x)? {
                    self.single(x)?;
                    self.fill_buf(gx, n, g);
                }
            }
            RED_DOT => {
                // loss = sum(x[i] y[i]).  dx = g*y, dy = g*x.
                let (x, y) = (args[0], args[1]);
                let xn = self.remap_v(x);
                let yn = self.remap_v(y);
                // The **self**-dot `loss = sum(x[i]^2)` — the sum-of-squares loss / L2 regularizer,
                // which lowers to `sreduce(x, x, n, RED_DOT)` with both operands the SAME buffer —
                // is one op with a repeated operand, not two separate contributions: `dx = 2g*x`, a
                // single velem scale. (`x`/`y` are canon-normalized above, so pointer identity is
                // the right test.) Without this case the two-buffer path below calls `single(x)`
                // twice and the second call always errors.
                if x == y {
                    if let Some(gx) = self.grad_target(x)? {
                        self.single(x)?;
                        let two = self.cf32(2.0);
                        let g2 = self.fmul(g, two, &F32); // 2g
                        self.velem_scale(gx, xn, g2, n);
                    }
                    return Ok(());
                }
                if let Some(gx) = self.grad_target(x)? {
                    self.single(x)?;
                    self.velem_scale(gx, yn, g, n);
                }
                if let Some(gy) = self.grad_target(y)? {
                    self.single(y)?;
                    self.velem_scale(gy, xn, g, n);
                }
            }
            RED_SSD => {
                let (x, y) = (args[0], args[1]);
                let xn = self.remap_v(x);
                let yn = self.remap_v(y);
                let two = self.cf32(2.0);
                let g2 = self.fmul(g, two, &F32); // 2g
                let ng2 = self.neg(g2, &F32); // -2g
                if let Some(gx) = self.grad_target(x)? {
                    self.single(x)?;
                    // dx = 2g*x + (-2g)*y
                    self.velem_affine(gx, xn, yn, g2, ng2, n);
                }
                if let Some(gy) = self.grad_target(y)? {
                    self.single(y)?;
                    // dy = -2g*x + 2g*y
                    self.velem_affine(gy, xn, yn, ng2, g2, n);
                }
            }
            RED_SUMABS => {
                // loss = g·Σ|x[i]|  (the L1 norm / regularizer)  ->  dx = g·sgn(x).
                // `sgn(0) = 0` is the subgradient convention every framework uses (PyTorch's
                // `abs_backward` is `grad * self.sgn()`), and it is the only choice that keeps the
                // rule odd — picking +g or -g at the kink would make d/dx of Σ|x| disagree with the
                // central difference on a row that straddles zero.
                let x = args[0];
                if let Some(gx) = self.grad_target(x)? {
                    self.single(x)?;
                    let xn = self.remap_v(x);
                    self.sign_fill(gx, xn, None, g, n);
                }
            }
            RED_ABSDIFF => {
                // loss = g·Σ|x[i] - y[i]|  (mean-absolute-error / SAD)  ->  dx = g·sgn(x-y),
                // dy = -g·sgn(x-y).
                let (x, y) = (args[0], args[1]);
                let xn = self.remap_v(x);
                let yn = self.remap_v(y);
                if x == y {
                    // |x-x| is identically 0: the loss does not depend on x at all, and the two
                    // contributions below would cancel. Nothing to accumulate.
                    return Ok(());
                }
                let ng = self.neg(g, &F32);
                if let Some(gx) = self.grad_target(x)? {
                    self.single(x)?;
                    self.sign_fill(gx, xn, Some(yn), g, n);
                }
                if let Some(gy) = self.grad_target(y)? {
                    self.single(y)?;
                    self.sign_fill(gy, xn, Some(yn), ng, n);
                }
            }
            other => {
                return Err(format!(
                    "autodiff: no VJP for sreduce op {other} \
                     (supported: SUM, DOT, SSD, SUMABS, ABSDIFF)"
                ))
            }
        }
        Ok(())
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

    // --- nn.Linear (C = A . B^T) ---------------------------------------------------------------

    /// `sgemm_nt(a, b, c, m, k, n, beta)` computes `C(m x n) = A(m x k) . B(n x k)^T`. With
    /// `dC = buf_adj[c]`: `dA(m x k) = dC . B` (a plain `A.B` GEMM, no transpose) and
    /// `dB(n x k) = dC^T . A` (transpose `dC` then `A.B`). Both ride `wukong_sgemm`, accumulating
    /// into the target via `beta = 1` when it already holds a contribution.
    ///
    /// NOTE the `dB` transpose is deliberately still the synthesized flat loop, even though
    /// [`Vjp::diff_sgemm`] below shows `wukong_sgemm_tn` does the identical job in one call with no
    /// scratch and no `buf_count[c]` requirement: `wukong_driver`'s
    /// `transformer_ffn_block_from_source` and `silu_and_gelu_backward_from_source` assert on the
    /// exact `wukong_sgemm(` spelling this emits, and that file is outside this workstream's fence.
    /// Switching it is a one-line change plus two assertion updates, and belongs in a commit that
    /// can touch both.
    fn diff_sgemm_nt(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (a, b, c) = (args[0], args[1], args[2]);
        let dc = match self.buf_adj.get(&c).copied() {
            Some(dc) => dc,
            None => return Ok(()), // output gradient is zero
        };
        self.reject_inplace_matmul(a, b, c)?;
        let m = self.remap_v(args[3]);
        let k = self.remap_v(args[4]);
        let n = self.remap_v(args[5]);
        let (an, bn) = (self.remap_v(a), self.remap_v(b));

        // dA(m x k) = dC(m x n) . B(n x k)   ->   sgemm(dC, B, dA, m'=m, k'=n, n'=k)
        if let Some(da) = self.grad_target(a)? {
            let beta = self.beta_for(a);
            self.sgemm(dc, bn, da, m, n, k, beta);
            self.contributed.insert(a);
        }
        // dB(n x k) = dC^T(n x m) . A(m x k)   ->   transpose dC, then sgemm(dCt, A, dB, n, m, k)
        if let Some(db) = self.grad_target(b)? {
            let count = *self
                .buf_count
                .get(&c)
                .ok_or("autodiff: unknown size for matmul output buffer")?;
            let dct = self.alloc_buf(count);
            self.transpose(dc, dct, m, n);
            let beta = self.beta_for(b);
            self.sgemm(dct, an, db, n, m, k, beta);
            self.contributed.insert(b);
        }
        Ok(())
    }

    // --- plain matmul (C = A . B) --------------------------------------------------------------

    /// `sgemm(a, b, c, m, k, n, beta)` computes `C(m x n) = A(m x k) . B(k x n)`. With
    /// `dC = buf_adj[c]`:
    /// - `dA(m x k) = dC(m x n) . B(k x n)^T`  ->  `sgemm_nt(dC, B, dA, m'=m, k'=n, n'=k)`
    ///   (B stored `[k, n]` is exactly the `n' x k'` operand `sgemm_nt` transposes);
    /// - `dB(k x n) = A(m x k)^T . dC(m x n)`  ->  `sgemm_tn(A, dC, dB, m'=k, k'=m, n'=n)`.
    ///
    /// Both gradients are one tuned-kernel call with no scratch: the transposes the maths asks for
    /// are exactly the two transposed GEMM spellings the runtime already ships.
    fn diff_sgemm(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (a, b, c) = (args[0], args[1], args[2]);
        let dc = match self.buf_adj.get(&c).copied() {
            Some(dc) => dc,
            None => return Ok(()), // output gradient is zero
        };
        self.reject_inplace_matmul(a, b, c)?;
        let m = self.remap_v(args[3]);
        let k = self.remap_v(args[4]);
        let n = self.remap_v(args[5]);
        let (an, bn) = (self.remap_v(a), self.remap_v(b));

        if let Some(da) = self.grad_target(a)? {
            let beta = self.beta_for(a);
            self.sgemm_nt_call(dc, bn, da, m, n, k, beta);
            self.contributed.insert(a);
        }
        if let Some(db) = self.grad_target(b)? {
            let beta = self.beta_for(b);
            self.sgemm_tn(an, dc, db, k, m, n, beta);
            self.contributed.insert(b);
        }
        Ok(())
    }

    // --- broadcast bias (y = x + b, b shared across rows) --------------------------------------

    /// `bias_bcast(x, b, out, rows, cols, op)` computes `out[i,j] = act(x[i,j] + b[j])`. For the
    /// pure add (`op == BIAS_ACT_NONE`) the VJP is the textbook inner-product bias gradient (the
    /// same `diff_bias = reduce over the batch axis` oneDNN's inner-product backward-weights
    /// computes): `dx = dout` and `db[j] = Σ_i dout[i,j]`, one `wukong_colsum_f32` call.
    ///
    /// A *fused* activation is refused: the recognizer folds `relu(x + b)` into one call, and
    /// differentiating that needs the masked derivative this rule does not carry. Refusing keeps a
    /// performance fusion from silently producing the wrong gradient.
    fn diff_bias(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (x, bvec, out) = (args[0], args[1], args[2]);
        let op = self.const_i64(args[5])?;
        if op != BIAS_ACT_NONE {
            return Err(format!(
                "autodiff: no VJP for `wukong_bias_bcast_f32` with the fused activation op {op}; \
                 only the pure broadcast add (op {BIAS_ACT_NONE}) has a rule"
            ));
        }
        let dout = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        let rows = self.remap_v(args[3]);
        let cols = self.remap_v(args[4]);
        let total = self.b.build(I64T, Op::Bin(BinOp::Mul, rows, cols));
        // dx = dout (the add is the identity in x)
        if let Some(dx) = self.grad_target(x)? {
            self.single(x)?;
            let one = self.cf32(1.0);
            self.velem_scale(dx, dout, one, total);
        }
        // db[j] = Σ_i dout[i, j]
        if let Some(db) = self.grad_target(bvec)? {
            self.single(bvec)?;
            self.colsum(dout, db, rows, cols);
        }
        Ok(())
    }

    // --- elementwise activation (vmath) --------------------------------------------------------

    /// `vmath(x, out, n, op)` computes `out = f(x)` elementwise. Backward `dx = dout (.) f'(x)`,
    /// emitted as one synthesized loop. relu uses the input `x` (`x>0`); the smooth ops reuse the
    /// forward *output* `out` (sigmoid' = y(1-y), tanh' = 1-y^2, exp' = y) so no recompute is needed.
    ///
    /// **An in-place activation is refused.** `wukong_vmath_f32` explicitly permits `out == x`
    /// ("`x` and `out` may alias (the recognizer allows in-place activations)",
    /// `wukong_runtime/src/vmath.rs`), so `x[i] = log(x[i])` is a legal forward. The buffer tape
    /// cannot follow it: one forward buffer gets one gradient buffer, so `x` and `out` would have
    /// to hold the input's adjoint and the output's at once. On top of that, every arm below that
    /// loads the forward INPUT (log, leakyrelu, hardsigmoid, softsign, the inverse trigs, and the
    /// fused `wukong_vmath2_f32` backward, which takes `x` as its first operand) would read `f(x)`.
    ///
    /// This refusal is a REWORDING, not a new gate, and the honest note matters: the aliased case
    /// was already unreachable, because `buf_adj[out]` can only have been set by a consumer rule,
    /// every one of which marks the buffer in `contributed` (or `diff_load` does), so `single(x)`
    /// below already failed — with "receives multiple gradient contributions; accumulation is not
    /// yet supported", which points at the wrong thing. Two reasons to state it here instead.
    /// `single`'s message advertises itself as temporary: the day accumulation lands, the
    /// coincidental guard evaporates and this becomes the silent wrong gradient it currently only
    /// looks like. And a user who wrote an in-place activation should be told that, not sent
    /// looking for a fan-out. (The out-of-place spelling of exactly this shape still
    /// differentiates — see `kernel_output_read_by_scalar_load`.)
    fn diff_vmath(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (x, out) = (args[0], args[1]);
        let dout = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        let op = self.const_i64(args[3])?;
        let n = self.remap_v(args[2]);
        let dx = match self.grad_target(x)? {
            Some(dx) => dx,
            None => return Ok(()), // input not differentiated
        };
        // Before `single`, so the aliasing is what gets reported.
        if out == x {
            return Err(format!(
                "autodiff: vmath VJP cannot differentiate an in-place activation (op {op}: \
                 the output buffer v{} aliases the input); one buffer cannot hold both the \
                 input and the output adjoint, and every derivative that reads x would read \
                 f(x) instead",
                out.0
            ));
        }
        self.single(x)?;
        let xn = self.remap_v(x);
        // The smooth activations whose derivative is a transcendental (silu folds a sigmoid, gelu a
        // tanh, softplus a sigmoid, elu an exp) ride the fused two-input backward kernel
        // `wukong_vmath2_f32(x, dy, dx, n, *_BWD)` — one pass, `dx = dy·act'(x)`, bit-identical with
        // the forward. The algebraic ones (relu/sigmoid/tanh/exp) stay on the synthesized loop below.
        if let Some(bwd) = vm2_bwd_code(op) {
            self.vmath2_bwd(xn, dout, dx, n, bwd);
            return Ok(());
        }
        let yn = self.remap_v(out); // forward output buffer (for the smooth derivatives)
        self.activation_backward(op, dout, xn, yn, dx, n)?;
        Ok(())
    }

    /// Emit `wukong_vmath2_f32(x, dy, dx, n, op)` — the fused activation backward `dx = dy·act'(x)`.
    fn vmath2_bwd(&mut self, x: ValueId, dy: ValueId, dx: ValueId, n: ValueId, op: i64) {
        let opv = self.cint(op);
        let f = self.syms.vmath2;
        self.b.build_void(Op::Call {
            func: f,
            args: vec![x, dy, dx, n, opv],
        });
    }

    // --- streaming affine / residual (velem) ---------------------------------------------------

    /// `velem(x, y, out, n, a, b, c, op)` computes `out = act(a*x + b*y + c)`. With the identity
    /// activation it is linear, so the VJP is exact: `dx = a*dout`, `dy = b*dout` (each a velem
    /// scale). This covers residual add (`a=b=1` -> `dx=dy=dout`) and scaling. Non-identity
    /// activations (relu/relu6) are rejected for now (they need a masked loop like vmath), and so is
    /// every *compute mode* above the activation byte ([`VE_HADAMARD`], [`VE_DIV`]) — those are not
    /// the affine form this rule differentiates. The gate is a WHITELIST of the two op spellings the
    /// affine rule is valid for, so a future high-bit mode fails closed rather than silently taking
    /// the linear VJP.
    fn diff_velem(&mut self, args: &[ValueId]) -> Result<(), String> {
        let op = self.const_i64(args[7])?;
        // The low byte selects a fused activation; the bits ABOVE `VE_USE_Y` select a *compute
        // mode*. Split them and dispatch on the mode, so an unknown future high bit still fails
        // closed instead of quietly taking a rule written for a different kernel.
        let act = op & 0xFF;
        let mode = op & !(0xFF | VE_USE_Y);
        if act != VE_ID {
            return Err(format!(
                "autodiff: velem VJP has no rule for the fused activation in op {op} (low byte \
                 {act}); only the identity activation ({VE_ID}) is differentiated"
            ));
        }
        match mode {
            0 => self.diff_velem_affine(args, op),
            VE_HADAMARD => self.diff_velem_hadamard(args),
            VE_DIV => self.diff_velem_div(args),
            other => Err(format!(
                "autodiff: velem VJP has no rule for compute mode {other} (op {op})"
            )),
        }
    }

    /// The affine form `out = a*x + b*y + c`: linear, so `dx = a*dout`, `dy = b*dout`.
    fn diff_velem_affine(&mut self, args: &[ValueId], op: i64) -> Result<(), String> {
        let (x, y, out) = (args[0], args[1], args[2]);
        let dout = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        let n = self.remap_v(args[3]);
        let (a, b) = (self.remap_v(args[4]), self.remap_v(args[5]));
        let uses_y = op & VE_USE_Y != 0;
        // dx = a * dout
        if let Some(dx) = self.grad_target(x)? {
            self.single(x)?;
            self.velem_scale(dx, dout, a, n);
        }
        // dy = b * dout (only if y was actually read)
        if uses_y {
            if let Some(dy) = self.grad_target(y)? {
                self.single(y)?;
                self.velem_scale(dy, dout, b, n);
            }
        }
        Ok(())
    }

    /// The Hadamard product `out = x (.) y` — the gate in every SwiGLU/GeGLU MLP, every attention
    /// mask and every dropout mask. Its VJP is the product rule, `dx = dout (.) y` and
    /// `dy = dout (.) x`: **the forward kernel is its own backward**, one `VE_HADAMARD` call each,
    /// no scratch. (`a`/`b`/`c` are ignored by the kernel in this mode — see
    /// `wukong_runtime::velem::elem1` — so they carry no coefficient into the gradient.)
    ///
    /// `out = x (.) x` is one op with a repeated operand, not two contributions: `dx = 2·dout(.)x`.
    fn diff_velem_hadamard(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (x, y, out) = (args[0], args[1], args[2]);
        self.reject_inplace_velem(x, y, out, "Hadamard-product")?;
        let dout = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        let n = self.remap_v(args[3]);
        let (xn, yn) = (self.remap_v(x), self.remap_v(y));
        if x == y {
            if let Some(dx) = self.grad_target(x)? {
                self.single(x)?;
                self.velem_hadamard(dx, dout, xn, n);
                let two = self.cf32(2.0);
                self.velem_scale(dx, dx, two, n);
            }
            return Ok(());
        }
        if let Some(dx) = self.grad_target(x)? {
            self.single(x)?;
            self.velem_hadamard(dx, dout, yn, n);
        }
        if let Some(dy) = self.grad_target(y)? {
            self.single(y)?;
            self.velem_hadamard(dy, dout, xn, n);
        }
        Ok(())
    }

    /// The elementwise quotient `out = x / y`: `dx = dout / y` and `dy = -(dout (.) out) / y`
    /// (writing `x/y²` as `out/y` keeps the second gradient to one squaring-free pass and reuses the
    /// forward output the tape already has). `dy` needs one `n`-wide scratch, so it requires a
    /// compile-time `n`; `dx` alone does not.
    fn diff_velem_div(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (x, y, out) = (args[0], args[1], args[2]);
        self.reject_inplace_velem(x, y, out, "elementwise-division")?;
        if x == y {
            return Err(
                "autodiff: velem VJP cannot differentiate `x / x` (the elementwise-division \
                 compute mode with both operands the same buffer)"
                    .to_string(),
            );
        }
        let dout = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        let n = self.remap_v(args[3]);
        let (yn, outn) = (self.remap_v(y), self.remap_v(out));
        if let Some(dx) = self.grad_target(x)? {
            self.single(x)?;
            self.velem_div(dx, dout, yn, n);
        }
        if let Some(dy) = self.grad_target(y)? {
            self.single(y)?;
            let count = self.const_count(args[3], "velem")?;
            let t = self.alloc_buf(count);
            self.velem_hadamard(t, dout, outn, n); // t = dout . (x/y)
            self.velem_div(dy, t, yn, n); //          dy = t / y = dout·x/y^2
            let neg1 = self.cf32(-1.0);
            self.velem_scale(dy, dy, neg1, n);
        }
        Ok(())
    }

    /// A `velem` whose output aliases an input destroys the forward value the Hadamard/quotient
    /// VJPs read back (`x *= y` lowers to exactly that). There is no rule for it — refuse rather
    /// than differentiate against a buffer whose contents the forward has already overwritten.
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

    /// Both matmul VJPs read the forward operands back (`dA` reads `B`, `dB` reads `A`), so a GEMM
    /// whose output buffer aliases either input hands them a buffer the forward has already
    /// overwritten.
    ///
    /// Unlike `vmath` and `norm`, this one is a LIVE silent wrong gradient, not a reworded
    /// refusal: the matmul rules are the only ones that do not go through `single` — they route
    /// through `beta_for`, which merely *observes* `contributed` to choose `beta` — so nothing on
    /// the path fails. With `c == a` the emitted `dA` is `sgemm(dC, B, dA, .., beta = 1)` where
    /// `dA` and `dC` are the same buffer: a GEMM accumulating into the operand it is reading. Exit
    /// 0, garbage numbers. Refuse.
    fn reject_inplace_matmul(&self, a: ValueId, b: ValueId, c: ValueId) -> Result<(), String> {
        if c == a || c == b {
            return Err(format!(
                "autodiff: matmul VJP cannot differentiate an in-place GEMM (output buffer \
                 v{} aliases an input); both adjoints read the un-overwritten forward operands",
                c.0
            ));
        }
        Ok(())
    }

    // --- row-wise normalization (softmax) ------------------------------------------------------

    /// `norm(x, out, rows, cols, eps_bits, op)` row-normalizes `x` (softmax / LayerNorm / RMSNorm).
    /// All three backward passes are per-row reductions (riding `sreduce`) plus an elementwise
    /// combine, emitted as nested counted loops (rows x cols). LayerNorm/RMSNorm/L2-norm recompute
    /// the row statistics from `x` (the forward kernel emits only `y`), so `x` must not be
    /// overwritten in place by the forward norm — a requirement this doc comment used to state and
    /// nothing enforced. It is now a refusal, worded like [`Vjp::diff_vmath`]'s and for the same
    /// two reasons: the buffer tape has one gradient buffer per forward buffer, and `single`'s
    /// pre-existing "multiple gradient contributions" message is both misleading here and
    /// advertised as temporary.
    fn diff_norm(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (x, out) = (args[0], args[1]);
        let op = self.const_i64(args[5])?;
        let dy = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        let dx = match self.grad_target(x)? {
            Some(dx) => dx,
            None => return Ok(()),
        };
        if out == x {
            return Err(format!(
                "autodiff: norm VJP cannot differentiate an in-place norm (op {op}: the \
                 output buffer v{} aliases the input); one buffer cannot hold both the input \
                 and the output adjoint, and LayerNorm/RMSNorm/L2-norm recompute the row \
                 statistics from x, which the kernel has already overwritten",
                out.0
            ));
        }
        self.single(x)?;
        let xn = self.remap_v(x);
        let y = self.remap_v(out); // the forward normalized output
        let rows = self.remap_v(args[2]);
        let cols = self.remap_v(args[3]);
        match op {
            NORM_SOFTMAX => self.softmax_back(dy, dx, y, rows, cols),
            NORM_LOGSOFTMAX => {
                // `y = log softmax(x)`, so `softmax(x) = exp(y)`: dx = dy - exp(y)·Σ_j dy_j per
                // row — the same form PyTorch's `_log_softmax_backward_data` uses
                // (`grad - exp(output) * grad.sum(dim, keepdim=True)`). One `wukong_vmath_f32`
                // EXP pass recovers the probabilities into a rows x cols scratch.
                let count = self.const_count_2d(args[2], args[3])?;
                self.logsoftmax_back(dy, dx, y, rows, cols, count);
                Ok(())
            }
            NORM_LAYERNORM | NORM_RMSNORM | NORM_L2NORM => {
                // eps rides in as f32 bits in an i64; it is a compile-time constant here.
                let eps_bits = self.const_i64(args[4])? as u32;
                let eps = self.cf32(f32::from_bits(eps_bits) as f64);
                match op {
                    NORM_LAYERNORM => {
                        // CAPABILITY NARROWING, deliberate and loud: the stable centered variance
                        // needs a `cols`-wide scratch holding `mu`, and a MIR alloca takes a
                        // compile-time element count. So a LayerNorm whose `cols` is a symbolic
                        // generic dim (`f<M,N>`, where `cols` arrives as a hidden i64 PARAM) now
                        // declines where the old cancelling `E[x^2] - mu^2` form — which needed no
                        // scratch — differentiated. That form is the defect this branch fixed; a
                        // named refusal is the correct trade against a silently NaN gradient.
                        // Lifting it needs a loop-carried accumulator `open_loop` does not have.
                        let cols_count = self.const_count(args[3], "norm").map_err(|_| {
                            "autodiff: the LayerNorm backward needs a compile-time-constant \
                             `cols` — it allocates a cols-wide scratch holding the row mean, \
                             so the variance is the stable centered form the forward kernel \
                             evaluated; a symbolic generic row width is not supported"
                                .to_string()
                        })?;
                        self.layernorm_back(dy, dx, xn, y, rows, cols, eps, cols_count);
                    }
                    NORM_RMSNORM => {
                        // RMSNorm divides the sum of squares by the row width...
                        let nf = self.b.build(F32, Op::Cast(CastKind::SiToFp, cols, F32));
                        self.scale_norm_back(dy, dx, xn, y, rows, cols, eps, nf);
                    }
                    // ...and L2 (unit-norm) does not: `r = sqrt(Σx² + eps)` instead of
                    // `sqrt(Σx²/n + eps)`, so the identical backward runs with a divisor of 1.
                    // dx = (1/r)(dy - y·<dy, y>), the `torch.nn.functional.normalize` gradient.
                    _ => {
                        let nf = self.cf32(1.0);
                        self.scale_norm_back(dy, dx, xn, y, rows, cols, eps, nf);
                    }
                }
                Ok(())
            }
            other => Err(format!(
                "autodiff: no VJP for norm op {other} \
                 (supported: SOFTMAX, LAYERNORM, RMSNORM, LOGSOFTMAX, L2NORM)"
            )),
        }
    }

    // --- row-wise normalization with a learned scale/shift (norm_affine) -----------------------

    /// `norm_affine(x, out, gamma, beta, rows, cols, eps_bits, op)` computes
    /// `out[i,j] = xhat[i,j]·gamma[j] + beta[j]`, where `xhat` is the LayerNorm or RMSNorm of row
    /// `i`. `gamma`/`beta` are `cols` long and shared across rows (the transformer layout); either
    /// may be a null pointer (a `Ptr`-typed `ConstInt(0)`), meaning scale 1 / no shift.
    ///
    /// The backward is the standard three gradients:
    /// - `dbeta[j] = Σ_i dy[i,j]`                      — one column reduction;
    /// - `dgamma[j] = Σ_i dy[i,j]·xhat[i,j]`           — one column reduction of a scratch;
    /// - `dx = normback(dxhat)` with `dxhat = dy·gamma` — the *unscaled* norm backward, run on the
    ///   recomputed `xhat` (NOT on `out`, which carries gamma/beta and would need a division by
    ///   gamma to invert).
    ///
    /// One synthesized `rows x cols` pass materializes `xhat`, `dy(.)xhat` and `dxhat`; the norm
    /// backward then reuses [`Vjp::layernorm_back`] / [`Vjp::scale_norm_back`] unchanged, so the
    /// affine variant inherits their stable centered variance automatically.
    ///
    /// KNOWN LIMIT, recorded rather than assumed away: those three scratches are `rows*cols`
    /// allocas in ONE frame, plus a `cols` one — a 512x768 residual block is ~4.7 MB of stack.
    /// The interpreter's slot-indexed memory absorbs that and `--train` is interpreter-only today,
    /// so it costs nothing now; the moment the backward gets a native path it is a stack overflow,
    /// and the fix (calling `wukong_layernorm_bwd_f32` / `wukong_rmsnorm_bwd_f32`, which take
    /// gamma directly and need no scratch at all) is the remaining half of P4.10. Whoever lands
    /// the native `--train` path owns this.
    fn diff_norm_affine(&mut self, args: &[ValueId]) -> Result<(), String> {
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
        let rows_count = self.const_count(args[4], "norm_affine")?;
        let cols_count = self.const_count(args[5], "norm_affine")?;
        let total_count = rows_count
            .checked_mul(cols_count)
            .ok_or("autodiff: norm_affine rows*cols overflows")?;
        let eps_bits = self.const_i64(args[6])? as u32;
        let eps = self.cf32(f32::from_bits(eps_bits) as f64);
        let rows = self.remap_v(args[4]);
        let cols = self.remap_v(args[5]);
        let xn = self.remap_v(x);
        // A null gamma/beta lowers to a `Ptr`-typed `ConstInt(0)` (the same spelling the
        // interpreter's marshalling keys on), so the absent-parameter case is a compile-time fact.
        let gamma_null = self.is_null_ptr(gamma);
        let beta_null = self.is_null_ptr(beta);

        let xhat = self.alloc_buf(total_count);
        let dxhat = self.alloc_buf(total_count);
        let dyxh = self.alloc_buf(total_count);
        let mubuf = self.alloc_buf(cols_count);
        let nf = self.b.build(F32, Op::Cast(CastKind::SiToFp, cols, F32));
        let one = self.cf32(1.0);
        let gn = self.remap_v(gamma);

        // Pass 1: recompute xhat, and materialize dy(.)xhat (for dgamma) and dy(.)gamma (= dxhat).
        let outer = self.open_loop(rows);
        let r = outer.idx;
        let rc = self.b.build(I64T, Op::Bin(BinOp::Mul, r, cols));
        let (mu, invv) = if op == NORM_LAYERNORM {
            let m = self.row_mean(xn, rc, cols, nf);
            let var = self.row_centered_var(xn, mubuf, rc, cols, nf, m);
            let veps = self.fadd(var, eps, &F32);
            let sigma = self.b.build(F32, Op::Sqrt(veps));
            (Some(m), self.fdiv(one, sigma, &F32))
        } else {
            let sumsq = self.row_reduce(xn, xn, rc, cols, RED_SUMSQ);
            let ms = self.fdiv(sumsq, nf, &F32);
            let mseps = self.fadd(ms, eps, &F32);
            let rr = self.b.build(F32, Op::Sqrt(mseps));
            (None, self.fdiv(one, rr, &F32))
        };
        let inner = self.open_loop(cols);
        let j = inner.idx;
        let idx = self.b.build(I64T, Op::Bin(BinOp::Add, rc, j));
        let xv = self.load_at(xn, idx);
        let xc = match mu {
            Some(m) => self.b.build(F32, Op::Bin(BinOp::FSub, xv, m)),
            None => xv,
        };
        let xh = self.fmul(xc, invv, &F32);
        self.store_at(xhat, idx, xh);
        let dyv = self.load_at(dy, idx);
        let gxh = self.fmul(dyv, xh, &F32);
        self.store_at(dyxh, idx, gxh);
        let gv = if gamma_null { one } else { self.load_at(gn, j) };
        let dxh = self.fmul(dyv, gv, &F32);
        self.store_at(dxhat, idx, dxh);
        self.close_loop(inner);
        self.close_loop(outer);

        // Pass 2: the three gradients.
        if !gamma_null {
            if let Some(dg) = self.grad_target(gamma)? {
                self.single(gamma)?;
                self.colsum(dyxh, dg, rows, cols);
            }
        }
        if !beta_null {
            if let Some(db) = self.grad_target(beta)? {
                self.single(beta)?;
                self.colsum(dy, db, rows, cols);
            }
        }
        if let Some(dx) = self.grad_target(x)? {
            self.single(x)?;
            if op == NORM_LAYERNORM {
                self.layernorm_back(dxhat, dx, xn, xhat, rows, cols, eps, cols_count);
            } else {
                self.scale_norm_back(dxhat, dx, xn, xhat, rows, cols, eps, nf);
            }
        }
        Ok(())
    }

    /// Whether `v` is a `Ptr`-typed null (`ConstInt(0)`) — how lowering spells an absent
    /// `gamma`/`beta` operand.
    fn is_null_ptr(&self, v: ValueId) -> bool {
        matches!(self.def_op.get(&v), Some(Op::ConstInt(0, _)))
    }

    /// softmax: `dx = y (.) (dy - sum_j dy_j y_j)` per row.
    fn softmax_back(
        &mut self,
        dy: ValueId,
        dx: ValueId,
        y: ValueId,
        rows: ValueId,
        cols: ValueId,
    ) -> Result<(), String> {
        let outer = self.open_loop(rows);
        let r = outer.idx;
        let rc = self.b.build(I64T, Op::Bin(BinOp::Mul, r, cols));
        let s = self.row_reduce(dy, y, rc, cols, RED_DOT); // sum_j dy_j y_j
        let inner = self.open_loop(cols);
        let j = inner.idx;
        let idx = self.b.build(I64T, Op::Bin(BinOp::Add, rc, j));
        let yv = self.load_at(y, idx);
        let dyv = self.load_at(dy, idx);
        let diff = self.b.build(F32, Op::Bin(BinOp::FSub, dyv, s));
        let dxv = self.b.build(F32, Op::Bin(BinOp::FMul, yv, diff));
        self.store_at(dx, idx, dxv);
        self.close_loop(inner);
        self.close_loop(outer);
        Ok(())
    }

    /// log-softmax: `y = (x - m) - log Σ exp(x - m)`, so `exp(y)` IS the softmax. Backward per row:
    /// `dx = dy - exp(y)·Σ_j dy_j`. One `wukong_vmath_f32` EXP pass over the whole `rows x cols`
    /// output recovers the probabilities into `p` (there is no scalar `exp` op in MIR, and the
    /// kernel is the same `exp1` polynomial the forward used), then the row loop combines.
    #[allow(clippy::too_many_arguments)]
    fn logsoftmax_back(
        &mut self,
        dy: ValueId,
        dx: ValueId,
        y: ValueId,
        rows: ValueId,
        cols: ValueId,
        total_count: u32,
    ) {
        let p = self.alloc_buf(total_count);
        let total = self.b.build(I64T, Op::Bin(BinOp::Mul, rows, cols));
        self.vmath_call(y, p, total, VM_EXP);
        let outer = self.open_loop(rows);
        let r = outer.idx;
        let rc = self.b.build(I64T, Op::Bin(BinOp::Mul, r, cols));
        let s = self.row_reduce(dy, dy, rc, cols, RED_SUM); // Σ_j dy_j
        let inner = self.open_loop(cols);
        let j = inner.idx;
        let idx = self.b.build(I64T, Op::Bin(BinOp::Add, rc, j));
        let dyv = self.load_at(dy, idx);
        let pv = self.load_at(p, idx);
        let ps = self.fmul(pv, s, &F32);
        let dxv = self.b.build(F32, Op::Bin(BinOp::FSub, dyv, ps));
        self.store_at(dx, idx, dxv);
        self.close_loop(inner);
        self.close_loop(outer);
    }

    /// LayerNorm: with `mu = mean(x)`, `sigma = sqrt(var + eps)`, `y = (x - mu)/sigma`,
    /// `dx = (1/sigma) (dy - mean(dy) - y * mean(dy (.) y))` per row.
    ///
    /// **The variance is the centered two-pass form, matching the forward kernel.** It used to be
    /// recovered as `E[x^2] - mu^2` from a SUM and a SUMSQ, which is the textbook catastrophic
    /// cancellation: in f32 its relative error grows like `eps·(mu/sigma)^2`, and once that error
    /// exceeds `var + eps` the value goes negative and `Sqrt` yields NaN gradients with no
    /// diagnostic. Residual-stream activations routinely have a large `mu/sigma`. The forward
    /// (`wukong_runtime::norm::layernorm_row_scalar` -> `dev2_8_scalar(x, n, mean)`) and the
    /// runtime's own backward kernel (`layernorm_bwd.rs`, "matching the forward variance
    /// discipline") both compute the mean first and then `Σ(x - mu)^2`; so does this now, via a
    /// `cols`-wide scratch filled with `mu` and one `RED_SSD` row reduction. Cost: one extra
    /// `cols`-wide velem write per row, in exchange for the gradient being the gradient of the
    /// function that was actually evaluated.
    #[allow(clippy::too_many_arguments)]
    fn layernorm_back(
        &mut self,
        dy: ValueId,
        dx: ValueId,
        x: ValueId,
        y: ValueId,
        rows: ValueId,
        cols: ValueId,
        eps: ValueId,
        cols_count: u32,
    ) {
        let nf = self.b.build(F32, Op::Cast(CastKind::SiToFp, cols, F32));
        let one = self.cf32(1.0);
        let mubuf = self.alloc_buf(cols_count);
        let outer = self.open_loop(rows);
        let r = outer.idx;
        let rc = self.b.build(I64T, Op::Bin(BinOp::Mul, r, cols));
        let mu = self.row_mean(x, rc, cols, nf);
        let var = self.row_centered_var(x, mubuf, rc, cols, nf, mu);
        let veps = self.fadd(var, eps, &F32);
        let sigma = self.b.build(F32, Op::Sqrt(veps));
        let inv = self.fdiv(one, sigma, &F32);
        let sum_dy = self.row_reduce(dy, dy, rc, cols, RED_SUM);
        let dot_dyy = self.row_reduce(dy, y, rc, cols, RED_DOT);
        let mean_dy = self.fdiv(sum_dy, nf, &F32);
        let mean_dyy = self.fdiv(dot_dyy, nf, &F32);
        let inner = self.open_loop(cols);
        let j = inner.idx;
        let idx = self.b.build(I64T, Op::Bin(BinOp::Add, rc, j));
        let dyv = self.load_at(dy, idx);
        let yv = self.load_at(y, idx);
        let ymdyy = self.fmul(yv, mean_dyy, &F32);
        let t1 = self.b.build(F32, Op::Bin(BinOp::FSub, dyv, mean_dy));
        let t2 = self.b.build(F32, Op::Bin(BinOp::FSub, t1, ymdyy));
        let dxv = self.fmul(inv, t2, &F32);
        self.store_at(dx, idx, dxv);
        self.close_loop(inner);
        self.close_loop(outer);
    }

    /// The mean of the `cols`-wide row at offset `rc`: `Σ x / nf`.
    fn row_mean(&mut self, x: ValueId, rc: ValueId, cols: ValueId, nf: ValueId) -> ValueId {
        let sum_x = self.row_reduce(x, x, rc, cols, RED_SUM);
        self.fdiv(sum_x, nf, &F32)
    }

    /// `Σ_j (x[rc + j] - mu)^2 / nf`, the **centered** row variance — the two-pass form (Chan,
    /// Golub & LeVeque 1983) the forward kernel uses. `mubuf` is a `cols`-wide scratch: fill it with
    /// `mu` (`velem` with `a = b = 0`, `c = mu`, reading the row so no uninitialized memory is
    /// touched), then one `RED_SSD` reduction is exactly `Σ (x - mu)^2`.
    #[allow(clippy::too_many_arguments)]
    fn row_centered_var(
        &mut self,
        x: ValueId,
        mubuf: ValueId,
        rc: ValueId,
        cols: ValueId,
        nf: ValueId,
        mu: ValueId,
    ) -> ValueId {
        let xp = self.gep(x, rc);
        let z = self.cf32(0.0);
        let op = self.cint(VE_ID);
        let f = self.syms.velem;
        self.b.build_void(Op::Call {
            func: f,
            args: vec![xp, xp, mubuf, cols, z, z, mu, op],
        });
        let ssd = self.row_reduce_at(xp, mubuf, cols, RED_SSD);
        self.fdiv(ssd, nf, &F32)
    }

    /// RMSNorm (`nf = cols`) and L2-norm (`nf = 1`): with `r = sqrt(Σx^2/nf + eps)`, `y = x/r`,
    /// `dx = (1/r) (dy - y * (<dy, y>/nf))` per row. The two norms differ only in that divisor —
    /// RMSNorm normalizes by the root-*mean*-square, L2 by the root-*sum*-square — so one emitter
    /// covers both. Both statistics are sums of squares, with no mean to cancel against, so this
    /// form is already numerically stable.
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
        let rc = self.b.build(I64T, Op::Bin(BinOp::Mul, r, cols));
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
    fn row_reduce(
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

    /// [`Vjp::row_reduce`] with the two operand pointers already formed — needed when one operand is
    /// a whole `cols`-wide scratch (offset 0) rather than a row of a `rows x cols` matrix.
    fn row_reduce_at(&mut self, ap: ValueId, bp: ValueId, cols: ValueId, op: i64) -> ValueId {
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

    // --- activation derivatives (the loop body for diff_vmath) ---------------------------------

    fn activation_backward(
        &mut self,
        op: i64,
        dout: ValueId,
        x: ValueId,
        y: ValueId,
        dx: ValueId,
        n: ValueId,
    ) -> Result<(), String> {
        let lp = self.open_loop(n);
        let i = lp.idx;
        let doi = self.load_at(dout, i);
        // f'(x[i]) as a value `fp`.
        let fp = match op {
            VM_RELU => {
                // (x[i] > 0) ? 1 : 0
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let one = self.cf32(1.0);
                let pos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, zero));
                self.b.build(F32, Op::Select(pos, one, zero))
            }
            VM_SIGMOID => {
                // y(1 - y)
                let yi = self.load_at(y, i);
                let one = self.cf32(1.0);
                let omy = self.b.build(F32, Op::Bin(BinOp::FSub, one, yi));
                self.b.build(F32, Op::Bin(BinOp::FMul, yi, omy))
            }
            VM_TANH => {
                // 1 - y^2
                let yi = self.load_at(y, i);
                let one = self.cf32(1.0);
                let y2 = self.b.build(F32, Op::Bin(BinOp::FMul, yi, yi));
                self.b.build(F32, Op::Bin(BinOp::FSub, one, y2))
            }
            VM_EXP => self.load_at(y, i), // exp' = y
            // --- the remaining ALGEBRAIC derivatives: each is a closed form in the input x or the
            // forward output y, so it needs no new runtime kernel and no transcendental. (mish,
            // erf, sin/cos, cosh and logsigmoid are deliberately absent: their derivatives need a
            // transcendental the loop cannot evaluate, or a sign the output alone does not fix, so
            // they keep the loud refusal below rather than a guessed rule.)
            VM_LOG => {
                // log' = 1/x
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                self.fdiv(one, xi, &F32)
            }
            VM_LEAKYRELU => {
                // leakyrelu' = x > 0 ? 1 : alpha   (the kernel's `x > 0.0` test, mirrored)
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let one = self.cf32(1.0);
                let al = self.cf32(LEAKY_ALPHA);
                let pos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, zero));
                self.b.build(F32, Op::Select(pos, one, al))
            }
            VM_SELU => {
                // selu(x) = lambda·x for x > 0, else lambda·alpha·(e^x - 1). For x <= 0 the
                // derivative lambda·alpha·e^x is exactly `y + lambda·alpha`, so no exp is needed.
                let xi = self.load_at(x, i);
                let yi = self.load_at(y, i);
                let zero = self.cf32(0.0);
                let lam = self.cf32(SELU_LAMBDA);
                let la = self.cf32(SELU_LAMBDA * SELU_ALPHA);
                let negb = self.fadd(yi, la, &F32);
                let pos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, zero));
                self.b.build(F32, Op::Select(pos, lam, negb))
            }
            VM_TANHSHRINK => {
                // y = x - tanh(x), so tanh(x) = x - y and tanhshrink' = 1 - (1-tanh^2) = tanh^2.
                let xi = self.load_at(x, i);
                let yi = self.load_at(y, i);
                let t = self.b.build(F32, Op::Bin(BinOp::FSub, xi, yi));
                self.fmul(t, t, &F32)
            }
            VM_SOFTSIGN => {
                // softsign' = 1/(1 + |x|)^2
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let one = self.cf32(1.0);
                let nx = self.neg(xi, &F32);
                let pos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, zero));
                let ax = self.b.build(F32, Op::Select(pos, xi, nx));
                let d = self.fadd(one, ax, &F32);
                let d2 = self.fmul(d, d, &F32);
                self.fdiv(one, d2, &F32)
            }
            VM_HARDSIGMOID => {
                // clamp(x+3, 0, 6)/6  ->  1/6 on (-3, 3), 0 outside.
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let inv6 = self.cf32(INV6);
                let three = self.cf32(3.0);
                let nthree = self.cf32(-3.0);
                let below = self.b.build(MirType::I1, Op::Cmp(CmpOp::Folt, xi, three));
                let inb = self.b.build(F32, Op::Select(below, inv6, zero));
                let above = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, nthree));
                self.b.build(F32, Op::Select(above, inb, zero))
            }
            VM_HARDSWISH => {
                // x·hardsigmoid(x)  ->  0 below -3, 1 above 3, (2x + 3)/6 in between.
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let one = self.cf32(1.0);
                let two = self.cf32(2.0);
                let three = self.cf32(3.0);
                let nthree = self.cf32(-3.0);
                let inv6 = self.cf32(INV6);
                let t2x = self.fmul(two, xi, &F32);
                let t = self.fadd(t2x, three, &F32);
                let ramp = self.fmul(t, inv6, &F32);
                let below = self.b.build(MirType::I1, Op::Cmp(CmpOp::Folt, xi, three));
                let mid = self.b.build(F32, Op::Select(below, ramp, one));
                let above = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, nthree));
                self.b.build(F32, Op::Select(above, mid, zero))
            }
            VM_EXP2 => {
                // (2^x)' = 2^x·ln2 = y·ln2
                let yi = self.load_at(y, i);
                let c = self.cf32(std::f64::consts::LN_2);
                self.fmul(yi, c, &F32)
            }
            VM_EXP10 => {
                let yi = self.load_at(y, i);
                let c = self.cf32(std::f64::consts::LN_10);
                self.fmul(yi, c, &F32)
            }
            VM_LOG2 => {
                // (log2 x)' = 1/(x·ln2)
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let c = self.cf32(std::f64::consts::LN_2);
                let d = self.fmul(xi, c, &F32);
                self.fdiv(one, d, &F32)
            }
            VM_LOG10 => {
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let c = self.cf32(std::f64::consts::LN_10);
                let d = self.fmul(xi, c, &F32);
                self.fdiv(one, d, &F32)
            }
            VM_EXPM1 => {
                // (e^x - 1)' = e^x = y + 1
                let yi = self.load_at(y, i);
                let one = self.cf32(1.0);
                self.fadd(yi, one, &F32)
            }
            VM_LOG1P => {
                // (ln(1+x))' = 1/(1+x)
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let d = self.fadd(one, xi, &F32);
                self.fdiv(one, d, &F32)
            }
            VM_ATAN => {
                // atan' = 1/(1 + x^2)
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let x2 = self.fmul(xi, xi, &F32);
                let d = self.fadd(one, x2, &F32);
                self.fdiv(one, d, &F32)
            }
            VM_ATANH => {
                // atanh' = 1/(1 - x^2)
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let x2 = self.fmul(xi, xi, &F32);
                let d = self.b.build(F32, Op::Bin(BinOp::FSub, one, x2));
                self.fdiv(one, d, &F32)
            }
            VM_TAN => {
                // tan' = 1 + tan^2 = 1 + y^2
                let yi = self.load_at(y, i);
                let one = self.cf32(1.0);
                let y2 = self.fmul(yi, yi, &F32);
                self.fadd(one, y2, &F32)
            }
            VM_ASINH => {
                // asinh' = 1/sqrt(1 + x^2)
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let x2 = self.fmul(xi, xi, &F32);
                let d = self.fadd(one, x2, &F32);
                let s = self.b.build(F32, Op::Sqrt(d));
                self.fdiv(one, s, &F32)
            }
            VM_ACOSH => {
                // acosh' = 1/sqrt(x^2 - 1)   (domain x > 1)
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let x2 = self.fmul(xi, xi, &F32);
                let d = self.b.build(F32, Op::Bin(BinOp::FSub, x2, one));
                let s = self.b.build(F32, Op::Sqrt(d));
                self.fdiv(one, s, &F32)
            }
            VM_ASIN | VM_ACOS => {
                // asin' = 1/sqrt(1 - x^2); acos' is its negation.
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let x2 = self.fmul(xi, xi, &F32);
                let d = self.b.build(F32, Op::Bin(BinOp::FSub, one, x2));
                let s = self.b.build(F32, Op::Sqrt(d));
                let r = self.fdiv(one, s, &F32);
                if op == VM_ACOS {
                    self.neg(r, &F32)
                } else {
                    r
                }
            }
            VM_SINH => {
                // sinh' = cosh = sqrt(1 + sinh^2) = sqrt(1 + y^2)  (cosh is always positive)
                let yi = self.load_at(y, i);
                let one = self.cf32(1.0);
                let y2 = self.fmul(yi, yi, &F32);
                let d = self.fadd(one, y2, &F32);
                self.b.build(F32, Op::Sqrt(d))
            }
            VM_CBRT => {
                // (x^(1/3))' = 1/(3·x^(2/3)) = 1/(3 y^2)
                let yi = self.load_at(y, i);
                let one = self.cf32(1.0);
                let three = self.cf32(3.0);
                let y2 = self.fmul(yi, yi, &F32);
                let d = self.fmul(three, y2, &F32);
                self.fdiv(one, d, &F32)
            }
            other => {
                return Err(format!(
                    "autodiff: no VJP for vmath op {other} (its derivative needs a transcendental \
                     no closed form in x or y provides)"
                ))
            }
        };
        // dx[i] = dout[i] * f'(x[i])
        let v = self.b.build(F32, Op::Bin(BinOp::FMul, doi, fp));
        self.store_at(dx, i, v);
        self.close_loop(lp);
        Ok(())
    }

    // --- buffer-gradient bookkeeping -----------------------------------------------------------

    /// The gradient buffer for forward buffer `buf`: an existing one, the appended grad param (for a
    /// `wrt` input), a freshly-zeroed alloca (for an intermediate), or `None` for a non-`wrt` input
    /// (whose gradient we do not compute). Intermediate buffers must have a known element count.
    pub(crate) fn grad_target(&mut self, buf: ValueId) -> Result<Option<ValueId>, String> {
        if let Some(&g) = self.buf_adj.get(&buf) {
            return Ok(Some(g));
        }
        if let Some(&gp) = self.grad_buf.get(&buf) {
            self.buf_adj.insert(buf, gp);
            return Ok(Some(gp));
        }
        if self.fwd.params.contains(&buf) {
            return Ok(None); // a non-differentiated input (e.g. the target labels)
        }
        let count = *self
            .buf_count
            .get(&buf)
            .ok_or_else(|| format!("autodiff: unknown size for intermediate buffer v{}", buf.0))?;
        let gb = self.alloc_buf(count);
        self.buf_adj.insert(buf, gb);
        Ok(Some(gb))
    }

    /// Assert `buf` receives only one gradient contribution (overwrite-correct). Multi-contribution
    /// (a buffer consumed by several ops, e.g. a residual fan-out) needs accumulation, supported for
    /// matmul via `beta` but not yet for the overwrite-style ops — a loud error, never a wrong grad.
    fn single(&mut self, buf: ValueId) -> Result<(), String> {
        if !self.contributed.insert(buf) {
            return Err(format!(
                "autodiff: buffer v{} receives multiple gradient contributions; accumulation is \
                 not yet supported for this op",
                buf.0
            ));
        }
        Ok(())
    }

    /// `beta` for a matmul gradient write: 0 (overwrite) on the first contribution to `buf`, 1
    /// (accumulate) afterwards — so a buffer feeding two matmuls sums its gradients correctly.
    fn beta_for(&mut self, buf: ValueId) -> ValueId {
        let first = !self.contributed.contains(&buf);
        self.cint(if first { 0 } else { 1 })
    }

    fn alloc_buf(&mut self, count: u32) -> ValueId {
        self.b.alloca(MirType::Array(Box::new(F32), count))
    }

    // --- emit helpers: kernel calls ------------------------------------------------------------

    /// `wukong_sgemm(a, b, c, m, k, n, beta)`: `C(m x n) = A(m x k) . B(k x n) (+ beta C)`.
    // One Rust parameter per C-ABI operand, deliberately: the signature IS the documentation of
    // the runtime call it emits. Bundling into a struct would obscure the mirror.
    #[allow(clippy::too_many_arguments)]
    fn sgemm(
        &mut self,
        a: ValueId,
        b: ValueId,
        c: ValueId,
        m: ValueId,
        k: ValueId,
        n: ValueId,
        beta: ValueId,
    ) {
        let f = self.syms.sgemm;
        self.b.build_void(Op::Call {
            func: f,
            args: vec![a, b, c, m, k, n, beta],
        });
    }

    /// Fill `dst[0..n]` with the scalar value `v`: `velem(dst, dst, dst, n, 0, 0, v, ID)`.
    fn fill_buf(&mut self, dst: ValueId, n: ValueId, v: ValueId) {
        let z = self.cf32(0.0);
        let op = self.cint(VE_ID);
        let f = self.syms.velem;
        self.b.build_void(Op::Call {
            func: f,
            args: vec![dst, dst, dst, n, z, z, v, op],
        });
    }

    /// `dst = coef * src` over `n` elements: `velem(src, src, dst, n, coef, 0, 0, ID)`.
    fn velem_scale(&mut self, dst: ValueId, src: ValueId, coef: ValueId, n: ValueId) {
        let z = self.cf32(0.0);
        let op = self.cint(VE_ID);
        let f = self.syms.velem;
        self.b.build_void(Op::Call {
            func: f,
            args: vec![src, src, dst, n, coef, z, z, op],
        });
    }

    /// `wukong_sgemm_nt(a, b, c, m, k, n, beta)`: `C(m x n) = A(m x k) . B(n x k)^T (+ beta C)`.
    #[allow(clippy::too_many_arguments)]
    fn sgemm_nt_call(
        &mut self,
        a: ValueId,
        b: ValueId,
        c: ValueId,
        m: ValueId,
        k: ValueId,
        n: ValueId,
        beta: ValueId,
    ) {
        let f = self.syms.sgemm_nt;
        self.b.build_void(Op::Call {
            func: f,
            args: vec![a, b, c, m, k, n, beta],
        });
    }

    /// `wukong_sgemm_tn(a, b, c, m, k, n, beta)`: `C(m x n) = A(k x m)^T . B(k x n) (+ beta C)` —
    /// the weight-gradient GEMM. It transposes `a` into its own scratch and runs the identical
    /// `A.B` microkernel `sgemm` uses, so it is bit-for-bit the transpose-then-multiply pair it
    /// replaces.
    #[allow(clippy::too_many_arguments)]
    fn sgemm_tn(
        &mut self,
        a: ValueId,
        b: ValueId,
        c: ValueId,
        m: ValueId,
        k: ValueId,
        n: ValueId,
        beta: ValueId,
    ) {
        let f = self.syms.sgemm_tn;
        self.b.build_void(Op::Call {
            func: f,
            args: vec![a, b, c, m, k, n, beta],
        });
    }

    /// `wukong_colsum_f32(x, out, rows, cols)`: `out[j] = Σ_i x[i*cols + j]` (overwrites `out`).
    fn colsum(&mut self, x: ValueId, out: ValueId, rows: ValueId, cols: ValueId) {
        let f = self.syms.colsum;
        self.b.build_void(Op::Call {
            func: f,
            args: vec![x, out, rows, cols],
        });
    }

    /// `wukong_vmath_f32(x, out, n, op)`: the elementwise activation kernel.
    fn vmath_call(&mut self, x: ValueId, out: ValueId, n: ValueId, op: i64) {
        let opv = self.cint(op);
        let f = self.syms.vmath;
        self.b.build_void(Op::Call {
            func: f,
            args: vec![x, out, n, opv],
        });
    }

    /// `dst = u (.) v` over `n` elements: `velem(u, v, dst, n, 1, 1, 0, VE_HADAMARD)`. The kernel
    /// ignores `a`/`b`/`c` in this compute mode, so they are only placeholders.
    fn velem_hadamard(&mut self, dst: ValueId, u: ValueId, v: ValueId, n: ValueId) {
        let one = self.cf32(1.0);
        let z = self.cf32(0.0);
        let op = self.cint(VE_HADAMARD);
        let f = self.syms.velem;
        self.b.build_void(Op::Call {
            func: f,
            args: vec![u, v, dst, n, one, one, z, op],
        });
    }

    /// `dst = u / v` over `n` elements: `velem(u, v, dst, n, 1, 1, 0, VE_DIV)`.
    fn velem_div(&mut self, dst: ValueId, u: ValueId, v: ValueId, n: ValueId) {
        let one = self.cf32(1.0);
        let z = self.cf32(0.0);
        let op = self.cint(VE_DIV);
        let f = self.syms.velem;
        self.b.build_void(Op::Call {
            func: f,
            args: vec![u, v, dst, n, one, one, z, op],
        });
    }

    /// A kernel dimension operand as a compile-time element count, for sizing a scratch buffer.
    /// `what` names the kernel in the decline, since a non-constant dimension is the user's
    /// problem to spell differently.
    fn const_count(&self, v: ValueId, what: &str) -> Result<u32, String> {
        let c = self.const_i64(v)?;
        u32::try_from(c).map_err(|_| {
            format!("autodiff: {what} dimension {c} is not a usable scratch-buffer size")
        })
    }

    /// `rows * cols` as a compile-time element count.
    fn const_count_2d(&self, rows: ValueId, cols: ValueId) -> Result<u32, String> {
        let r = self.const_count(rows, "norm")?;
        let c = self.const_count(cols, "norm")?;
        r.checked_mul(c)
            .ok_or_else(|| "autodiff: norm rows*cols overflows a scratch-buffer size".to_string())
    }

    /// `dst = a*x + b*y` over `n` elements: `velem(x, y, dst, n, a, b, 0, ID|USE_Y)`.
    fn velem_affine(
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
        self.b.build_void(Op::Call {
            func: f,
            args: vec![x, y, dst, n, a, b, z, op],
        });
    }

    // --- emit helpers: synthesized loops -------------------------------------------------------

    /// Transpose `src (m x n)` into `dst (n x m)` with a single flat loop: for flat index `f`,
    /// `row = f / n`, `col = f % n`, `dst[col*m + row] = src[f]`.
    fn transpose(&mut self, src: ValueId, dst: ValueId, m: ValueId, n: ValueId) {
        let total = self.b.build(I64T, Op::Bin(BinOp::Mul, m, n));
        let lp = self.open_loop(total);
        let f = lp.idx;
        let row = self.b.build(I64T, Op::Bin(BinOp::SDiv, f, n));
        let col = self.b.build(I64T, Op::Bin(BinOp::SRem, f, n));
        let cm = self.b.build(I64T, Op::Bin(BinOp::Mul, col, m));
        let didx = self.b.build(I64T, Op::Bin(BinOp::Add, cm, row));
        let v = self.load_at(src, f);
        self.store_at(dst, didx, v);
        self.close_loop(lp);
    }

    /// Open a counted loop `for idx in 0..n`. Emits the back-edge condition and switches the builder
    /// to the (empty) body block, with `idx` available; pair with [`close_loop`].
    fn open_loop(&mut self, n: ValueId) -> Loop {
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

    /// Close the loop opened by [`open_loop`]: append `idx += 1; br header(idx+1)` and switch the
    /// builder to the exit block. Assumes the body is straight-line (current block is the body).
    fn close_loop(&mut self, lp: Loop) {
        let one = self.cint(1);
        let i1 = self.b.build(I64T, Op::Bin(BinOp::Add, lp.idx, one));
        self.b.br(lp.header, vec![i1]);
        self.b.switch_to(lp.exit);
    }

    // --- small helpers -------------------------------------------------------------------------

    /// `&base[idx]` as an `f32` element pointer.
    fn gep(&mut self, base: ValueId, idx: ValueId) -> ValueId {
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
    fn load_at(&mut self, base: ValueId, idx: ValueId) -> ValueId {
        let p = self.b.build(
            PTR,
            Op::Gep {
                ptr: base,
                index: idx,
                elem: F32,
            },
        );
        self.b.build(F32, Op::Load(p, F32))
    }

    /// Store `val` to `base[idx]` (`f32`).
    fn store_at(&mut self, base: ValueId, idx: ValueId, val: ValueId) {
        let p = self.b.build(
            PTR,
            Op::Gep {
                ptr: base,
                index: idx,
                elem: F32,
            },
        );
        self.b.build_void(Op::Store { ptr: p, value: val });
    }

    /// The new-function value for a forward (old) value.
    fn remap_v(&self, v: ValueId) -> ValueId {
        self.fwd_to_new[&v]
    }

    /// Read a compile-time `i64` constant operand (a dim or op code). Errors if not a constant.
    fn const_i64(&self, v: ValueId) -> Result<i64, String> {
        match self.def_op.get(&v) {
            Some(Op::ConstInt(c, _)) => Ok(*c as i64),
            _ => Err(format!(
                "autodiff: kernel operand v{} must be a compile-time constant",
                v.0
            )),
        }
    }

    fn cint(&mut self, x: i64) -> ValueId {
        self.b.build(I64T, Op::ConstInt(x as i128, I64T))
    }

    fn cf32(&mut self, x: f64) -> ValueId {
        self.b.build(F32, Op::ConstFloat(x, F32))
    }
}

/// What the reverse walk does at a call to one of the ML I/O and randomness runtime entries, none of
/// which is a tensor kernel with a VJP. The decision is explicit per entry so a new one is a choice,
/// not a fall-through:
/// * the typed reads (`wukong_rt_read_*`, `wukong_rt_st_read`, `wukong_rt_npy_read`), `argsort` and
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
        _ => None,
    }
}
