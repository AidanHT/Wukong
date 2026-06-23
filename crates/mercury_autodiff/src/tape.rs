//! Tensor-tape VJPs: differentiating a straight-line sequence of recognized **kernel calls**.
//!
//! Mercury lowers tensor ops to runtime-kernel calls (`mercury_sgemm_nt` = `nn.Linear`,
//! `mercury_vmath_f32` = elementwise activation, `mercury_sreduce_f32` = reduction,
//! `mercury_velem_f32` = streaming affine/residual) — the interpreter runs the *same* kernels, so a
//! forward MIR function can be a tape of these calls plus a final scalar loss. Where the scalar core
//! ([`crate::Vjp`]) tracks an adjoint per SSA value, this module tracks an adjoint per **buffer**
//! (memory), the PyTorch-style picture tensor ops need: each forward buffer has a gradient buffer,
//! set by a consumer's VJP and read by the producer's. The backward emits *more kernel calls* (so the
//! matmul gradients ride the same tuned GEMM) plus small synthesized loops for the transpose and the
//! elementwise (Hadamard) combines that no single existing kernel covers.
//!
//! Buffers are differentiated under the same single-block contract as the scalar core; the only new
//! control flow is the counted loops this module synthesizes for transpose / activation-backward.

use crate::Vjp;
use mercury_mir::{BinOp, BlockId, CmpOp, MirType, Op, ValueId};
use mercury_span::{Interner, Symbol};

// --- runtime kernel op codes (mirrored from mercury_runtime; the interp dispatches the same) ------
const VM_EXP: i64 = 0;
const VM_TANH: i64 = 2;
const VM_SIGMOID: i64 = 3;
const VM_RELU: i64 = 4;
const RED_DOT: i64 = 0;
const RED_SSD: i64 = 1;
const RED_SUM: i64 = 2;
const NORM_SOFTMAX: i64 = 0;
/// velem activation: identity (`out = a*x + b*y + c`).
const VE_ID: i64 = 0;
/// OR'd into a velem op when `y` is read (`b` may be non-zero).
const VE_USE_Y: i64 = 256;

const F32: MirType = MirType::F32;
const I64T: MirType = MirType::I64;
const PTR: MirType = MirType::Ptr;

/// Interned symbols for the kernels we recognize (forward) and emit (backward). Interning dedups,
/// so these compare equal to the symbols a forward tape was built with.
pub(crate) struct Syms {
    pub sgemm: Symbol,
    pub sgemm_nt: Symbol,
    pub vmath: Symbol,
    pub sreduce: Symbol,
    pub velem: Symbol,
    pub norm: Symbol,
}

impl Syms {
    pub fn new(it: &mut Interner) -> Syms {
        Syms {
            sgemm: it.intern("mercury_sgemm"),
            sgemm_nt: it.intern("mercury_sgemm_nt"),
            vmath: it.intern("mercury_vmath_f32"),
            sreduce: it.intern("mercury_sreduce_f32"),
            velem: it.intern("mercury_velem_f32"),
            norm: it.intern("mercury_norm_f32"),
        }
    }
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
            || func == self.syms.vmath
            || func == self.syms.sreduce
            || func == self.syms.velem
            || func == self.syms.norm
    }

    /// Differentiate one recognized kernel call. `args`/`result` are the forward (old) value ids.
    pub(crate) fn diff_kernel_call(
        &mut self,
        func: Symbol,
        args: &[ValueId],
        result: Option<ValueId>,
    ) -> Result<(), String> {
        if func == self.syms.sreduce {
            self.diff_sreduce(args, result)
        } else if func == self.syms.sgemm_nt {
            self.diff_sgemm_nt(args)
        } else if func == self.syms.vmath {
            self.diff_vmath(args)
        } else if func == self.syms.velem {
            self.diff_velem(args)
        } else if func == self.syms.norm {
            self.diff_norm(args)
        } else {
            Err("autodiff: kernel call has no VJP rule".to_string())
        }
    }

    // --- reductions: scalar adjoint -> buffer adjoint ------------------------------------------

    /// `loss = sreduce(x, y, n, op)` produces the scalar loss. Its adjoint `g = adj[loss]` flows back
    /// into a *buffer* gradient. SUM: `dx = g` (broadcast). SSD (`sum (x-y)^2`): `dx = 2g(x-y)`,
    /// `dy = -2g(x-y)` — the MSE-loss backbone — each one streaming-affine, so a single velem call.
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
                    "autodiff: no VJP for sreduce op {other} (only SUM and SSD so far)"
                ))
            }
        }
        Ok(())
    }

    // --- nn.Linear (C = A . B^T) ---------------------------------------------------------------

    /// `sgemm_nt(a, b, c, m, k, n, beta)` computes `C(m x n) = A(m x k) . B(n x k)^T`. With
    /// `dC = buf_adj[c]`: `dA(m x k) = dC . B` (a plain `A.B` GEMM, no transpose) and
    /// `dB(n x k) = dC^T . A` (transpose `dC` then `A.B`). Both ride `mercury_sgemm`, accumulating
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
        let yn = self.remap_v(out); // forward output buffer (for the smooth derivatives)
        self.activation_backward(op, dout, xn, yn, dx, n)?;
        Ok(())
    }

    // --- streaming affine / residual (velem) ---------------------------------------------------

    /// `velem(x, y, out, n, a, b, c, op)` computes `out = act(a*x + b*y + c)`. With the identity
    /// activation it is linear, so the VJP is exact: `dx = a*dout`, `dy = b*dout` (each a velem
    /// scale). This covers residual add (`a=b=1` -> `dx=dy=dout`) and scaling. Non-identity
    /// activations (relu/relu6) are rejected for now (they need a masked loop like vmath).
    fn diff_velem(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (x, y, out) = (args[0], args[1], args[2]);
        let op = self.const_i64(args[7])?;
        if op & 0xff != VE_ID {
            return Err(format!(
                "autodiff: velem VJP only supports the identity activation (op {op})"
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

    /// `norm(x, out, rows, cols, eps_bits, op)` row-normalizes `x`. For **softmax** (`out = y`), the
    /// VJP per row is `dx = y (.) (dy - sum_j dy_j y_j)`: a per-row dot (`sreduce`, riding the tuned
    /// reduction) then an elementwise combine — emitted as nested counted loops (rows x cols).
    fn diff_norm(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (x, out) = (args[0], args[1]);
        let op = self.const_i64(args[5])?;
        if op != NORM_SOFTMAX {
            return Err(format!(
                "autodiff: only the softmax norm VJP is implemented (op {op}); LayerNorm/RMSNorm \
                 backward are not yet supported"
            ));
        }
        let dy = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        let dx = match self.grad_target(x)? {
            Some(dx) => dx,
            None => return Ok(()),
        };
        self.single(x)?;
        let y = self.remap_v(out); // the forward softmax output
        let rows = self.remap_v(args[2]);
        let cols = self.remap_v(args[3]);
        let dotop = self.cint(RED_DOT);
        let sreduce = self.syms.sreduce;

        // for r in 0..rows:
        let outer = self.open_loop(rows);
        let r = outer.idx;
        let rc = self.b.build(I64T, Op::Bin(BinOp::Mul, r, cols)); // row base offset
        let dy_row = self.gep(dy, rc);
        let y_row = self.gep(y, rc);
        // s = sum_j dy[r,j] * y[r,j]   (per-row dot)
        let s = self.b.build(
            F32,
            Op::Call {
                func: sreduce,
                args: vec![dy_row, y_row, cols, dotop],
            },
        );
        // for j in 0..cols: dx[r,j] = y[r,j] * (dy[r,j] - s)
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
        let count = *self.buf_count.get(&buf).ok_or_else(|| {
            format!("autodiff: unknown size for intermediate buffer v{}", buf.0)
        })?;
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

    /// `mercury_sgemm(a, b, c, m, k, n, beta)`: `C(m x n) = A(m x k) . B(k x n) (+ beta C)`.
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
