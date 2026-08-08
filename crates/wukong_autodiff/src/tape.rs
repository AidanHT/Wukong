//! Tensor-tape VJPs: differentiating a straight-line sequence of recognized **kernel calls**.
//!
//! Wukong lowers tensor ops to runtime-kernel calls (`wukong_sgemm_nt` = `nn.Linear`,
//! `wukong_vmath_f32` = elementwise activation, `wukong_sreduce_f32` = reduction,
//! `wukong_velem_f32` = streaming affine/residual, `wukong_norm_f32` = row-wise softmax/LayerNorm/
//! RMSNorm; each also in its `@parallel` spelling) — the interpreter runs the *same* kernels, so a
//! forward MIR function can be a tape of these calls plus a final scalar loss. Where the scalar core
//! ([`crate::Vjp`]) tracks an adjoint per SSA value, this module tracks an adjoint per **buffer**
//! (memory), the PyTorch-style picture tensor ops need: each forward buffer has a gradient buffer,
//! set by a consumer's VJP and read by the producer's. The backward emits *more kernel calls* (so the
//! matmul gradients ride the same tuned GEMM) plus small synthesized loops for the transpose and the
//! elementwise (Hadamard) combines that no single existing kernel covers.
//!
//! Buffers are differentiated under the same single-block contract as the scalar core; the only new
//! control flow is the counted loops this module synthesizes — one for the transpose and for an
//! algebraic activation backward, a nested rows x cols pair for the norm backwards. (So the *emitted*
//! gradient function is multi-block: it is not itself re-differentiable.)

use crate::Vjp;
use wukong_mir::{BinOp, BlockId, CastKind, CmpOp, MirType, Op, ValueId};
use wukong_span::{Interner, Symbol};

// --- runtime kernel op codes (mirrored from wukong_runtime; the interp dispatches the same) ------
const VM_EXP: i64 = 0;
const VM_TANH: i64 = 2;
const VM_SIGMOID: i64 = 3;
const VM_RELU: i64 = 4;
const VM_SILU: i64 = 5;
const VM_GELU: i64 = 6;
const VM_ELU: i64 = 7;
const VM_SOFTPLUS: i64 = 9;
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
const NORM_SOFTMAX: i64 = 0;
const NORM_LAYERNORM: i64 = 1;
const NORM_RMSNORM: i64 = 2;
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
    pub sgemm: Symbol,
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
}

impl Syms {
    pub fn new(it: &mut Interner) -> Syms {
        Syms {
            sgemm: it.intern("wukong_sgemm"),
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
        func == self.syms.sgemm_nt
            || func == self.syms.sgemm_nt_parallel
            || func == self.syms.vmath
            || func == self.syms.vmath_parallel
            || func == self.syms.sreduce
            || func == self.syms.sreduce_parallel
            || func == self.syms.velem
            || func == self.syms.velem_parallel
            || func == self.syms.norm
            || func == self.syms.norm_parallel
    }

    /// Canonicalize a kernel buffer argument to the base buffer it addresses: peel a whole-buffer
    /// `gep(buf, 0)` (the form MIR lowering emits for *some* kernel operands — e.g. `vmath`'s output —
    /// while others, like `sreduce`'s input, pass the bare `alloca`). Without this, the two forms key
    /// *different* buffer-adjoint entries and the gradient chain silently breaks between the ops. A
    /// non-zero-offset gep is a genuine sub-slice and is left alone. Dim/op-code operands are constants
    /// (not geps), so canonicalizing the whole argument list is a no-op for them.
    fn canon(&self, v: ValueId) -> ValueId {
        if let Some(Op::Gep { ptr, index, .. }) = self.def_op.get(&v) {
            if self.is_const_zero(*index) {
                return self.canon(*ptr);
            }
        }
        v
    }

    /// Whether `v` is the integer constant 0, tracing through the sign-/zero-extend casts lowering
    /// inserts on a loop index (`gep buf, sext(const.i32 0)` is still the buffer base).
    fn is_const_zero(&self, v: ValueId) -> bool {
        match self.def_op.get(&v) {
            Some(Op::ConstInt(0, _)) => true,
            Some(Op::Cast(_, inner, _)) => self.is_const_zero(*inner),
            _ => false,
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
        } else if func == self.syms.sgemm_nt || func == self.syms.sgemm_nt_parallel {
            Some(7) // (a, b, c, m, k, n, beta)
        } else if func == self.syms.vmath || func == self.syms.vmath_parallel {
            Some(4) // (x, out, n, op)
        } else if func == self.syms.velem || func == self.syms.velem_parallel {
            Some(8) // (x, y, out, n, a, b, c, op)
        } else if func == self.syms.norm || func == self.syms.norm_parallel {
            Some(6) // (x, out, rows, cols, eps_bits, op)
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
        let args: Vec<ValueId> = raw_args.iter().map(|&a| self.canon(a)).collect();
        let args = &args[..];
        if func == self.syms.sreduce || func == self.syms.sreduce_parallel {
            self.diff_sreduce(args, result)
        } else if func == self.syms.sgemm_nt || func == self.syms.sgemm_nt_parallel {
            self.diff_sgemm_nt(args)
        } else if func == self.syms.vmath || func == self.syms.vmath_parallel {
            self.diff_vmath(args)
        } else if func == self.syms.velem || func == self.syms.velem_parallel {
            self.diff_velem(args)
        } else if func == self.syms.norm || func == self.syms.norm_parallel {
            self.diff_norm(args)
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
            other => {
                return Err(format!(
                    "autodiff: no VJP for sreduce op {other} (supported: SUM, DOT, SSD)"
                ))
            }
        }
        Ok(())
    }

    // --- nn.Linear (C = A . B^T) ---------------------------------------------------------------

    /// `sgemm_nt(a, b, c, m, k, n, beta)` computes `C(m x n) = A(m x k) . B(n x k)^T`. With
    /// `dC = buf_adj[c]`: `dA(m x k) = dC . B` (a plain `A.B` GEMM, no transpose) and
    /// `dB(n x k) = dC^T . A` (transpose `dC` then `A.B`). Both ride `wukong_sgemm`, accumulating
    /// into the target via `beta = 1` when it already holds a contribution.
    fn diff_sgemm_nt(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (a, b, c) = (args[0], args[1], args[2]);
        let dc = match self.buf_adj.get(&c).copied() {
            Some(dc) => dc,
            None => return Ok(()), // output gradient is zero
        };
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

    // --- elementwise activation (vmath) --------------------------------------------------------

    /// `vmath(x, out, n, op)` computes `out = f(x)` elementwise. Backward `dx = dout (.) f'(x)`,
    /// emitted as one synthesized loop. relu uses the input `x` (`x>0`); the smooth ops reuse the
    /// forward *output* `out` (sigmoid' = y(1-y), tanh' = 1-y^2, exp' = y) so no recompute is needed.
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
        let (x, y, out) = (args[0], args[1], args[2]);
        let op = self.const_i64(args[7])?;
        if op != VE_ID && op != (VE_ID | VE_USE_Y) {
            let what = if op & VE_HADAMARD != 0 {
                "the Hadamard-product compute mode (out = x . y)"
            } else if op & VE_DIV != 0 {
                "the elementwise-division compute mode (out = x / y)"
            } else {
                "a non-identity activation (relu/relu6)"
            };
            return Err(format!(
                "autodiff: velem VJP only supports the identity affine form (op {VE_ID} or \
                 {}); op {op} selects {what}",
                VE_ID | VE_USE_Y
            ));
        }
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

    // --- row-wise normalization (softmax) ------------------------------------------------------

    /// `norm(x, out, rows, cols, eps_bits, op)` row-normalizes `x` (softmax / LayerNorm / RMSNorm).
    /// All three backward passes are per-row reductions (riding `sreduce`) plus an elementwise
    /// combine, emitted as nested counted loops (rows x cols). LayerNorm/RMSNorm recompute the row
    /// statistics from `x` (the forward kernel emits only `y`), so `x` must not be overwritten
    /// in place by the forward norm.
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
        self.single(x)?;
        let xn = self.remap_v(x);
        let y = self.remap_v(out); // the forward normalized output
        let rows = self.remap_v(args[2]);
        let cols = self.remap_v(args[3]);
        match op {
            NORM_SOFTMAX => self.softmax_back(dy, dx, y, rows, cols),
            NORM_LAYERNORM | NORM_RMSNORM => {
                // eps rides in as f32 bits in an i64; it is a compile-time constant here.
                let eps_bits = self.const_i64(args[4])? as u32;
                let eps = self.cf32(f32::from_bits(eps_bits) as f64);
                if op == NORM_LAYERNORM {
                    self.layernorm_back(dy, dx, xn, y, rows, cols, eps);
                } else {
                    self.rmsnorm_back(dy, dx, xn, y, rows, cols, eps);
                }
                Ok(())
            }
            other => Err(format!("autodiff: no VJP for norm op {other}")),
        }
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

    /// LayerNorm: with `mu = mean(x)`, `sigma = sqrt(var + eps)`, `y = (x - mu)/sigma`,
    /// `dx = (1/sigma) (dy - mean(dy) - y * mean(dy (.) y))` per row. var is recovered from
    /// `E[x^2] - mu^2` (a SUM and a SUMSQ of the row).
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
    ) {
        let nf = self.b.build(F32, Op::Cast(CastKind::SiToFp, cols, F32));
        let one = self.cf32(1.0);
        let outer = self.open_loop(rows);
        let r = outer.idx;
        let rc = self.b.build(I64T, Op::Bin(BinOp::Mul, r, cols));
        let sum_x = self.row_reduce(x, x, rc, cols, RED_SUM);
        let sumsq_x = self.row_reduce(x, x, rc, cols, RED_SUMSQ);
        let mu = self.fdiv(sum_x, nf, &F32);
        let ex2 = self.fdiv(sumsq_x, nf, &F32);
        let mu2 = self.fmul(mu, mu, &F32);
        let var = self.b.build(F32, Op::Bin(BinOp::FSub, ex2, mu2));
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

    /// RMSNorm: with `r = sqrt(mean(x^2) + eps)`, `y = x/r`,
    /// `dx = (1/r) (dy - y * mean(dy (.) y))` per row.
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_back(
        &mut self,
        dy: ValueId,
        dx: ValueId,
        x: ValueId,
        y: ValueId,
        rows: ValueId,
        cols: ValueId,
        eps: ValueId,
    ) {
        let nf = self.b.build(F32, Op::Cast(CastKind::SiToFp, cols, F32));
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
            other => {
                return Err(format!(
                    "autodiff: no VJP for vmath op {other} (relu/sigmoid/tanh/exp so far)"
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
    fn grad_target(&mut self, buf: ValueId) -> Result<Option<ValueId>, String> {
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
