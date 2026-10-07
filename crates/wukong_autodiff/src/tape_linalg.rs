//! Linear-algebra VJPs for the buffer tape: the whole GEMM family the recognizers emit (plain
//! `A·B`, `nn.Linear` `A·Bᵀ`, the weight-gradient `Aᵀ·B`, the α-scaled attention-score GEMM and the
//! fused `act(A·Bᵀ + bias)` epilogue), GEMV / GEVM, the broadcast bias, transposes and the column
//! reductions.
//!
//! Every adjoint is itself one tuned kernel call — the transposes the maths asks for are exactly the
//! transposed GEMM spellings the runtime ships, so no rule synthesizes a transpose loop — and the
//! `@parallel` forward spelling selects the `@parallel` backward spelling (bit-identical by the
//! runtime's fixed-blocking contract).

use crate::tape::{
    pick, BIAS_ACT_NONE, EPI_GELU, EPI_IDENTITY, EPI_RELU, EPI_SILU, F32, VM2_GELU_BWD,
    VM2_SILU_BWD, VM_RELU,
};
use crate::Vjp;
use wukong_mir::{CastKind, Op, ValueId};

impl<'a> Vjp<'a> {
    /// The adjoints of `C(m x n) = A(m x k) · B(n x k)ᵀ` for a given `dC`:
    /// - `dA(m x k) = dC · B`   ->  `sgemm(dC, B, dA, m'=m, k'=n, n'=k)`;
    /// - `dB(n x k) = dCᵀ · A`  ->  `sgemm_tn(dC, A, dB, m'=n, k'=m, n'=k)` (`dC` stored `[m, n]` is
    ///   the `[k', m']` operand the TN kernel transposes).
    ///
    /// Both accumulate through the GEMM's own `beta = 1` when the target already holds a
    /// contribution (a weight shared by two matmuls, an input feeding two layers).
    #[allow(clippy::too_many_arguments)]
    fn nt_adjoints(
        &mut self,
        a: ValueId,
        b: ValueId,
        dc: ValueId,
        m: ValueId,
        k: ValueId,
        n: ValueId,
        par: bool,
    ) -> Result<(), String> {
        let (an, bn) = (self.remap_v(a), self.remap_v(b));
        if let Some(da) = self.grad_target(a)? {
            let beta = self.beta_for(a);
            let f = pick(self.syms.sgemm, self.syms.sgemm_par, par);
            self.emit(f, vec![dc, bn, da, m, n, k, beta]);
            self.contributed.insert(a);
        }
        if let Some(db) = self.grad_target(b)? {
            let beta = self.beta_for(b);
            let f = pick(self.syms.sgemm_tn, self.syms.sgemm_tn_par, par);
            self.emit(f, vec![dc, an, db, n, m, k, beta]);
            self.contributed.insert(b);
        }
        Ok(())
    }

    /// `sgemm_nt(a, b, c, m, k, n, beta)`: `C = A · Bᵀ` — the `nn.Linear` form.
    pub(crate) fn diff_sgemm_nt(&mut self, args: &[ValueId], par: bool) -> Result<(), String> {
        let (a, b, c) = (args[0], args[1], args[2]);
        let dc = match self.buf_adj.get(&c).copied() {
            Some(dc) => dc,
            None => return Ok(()), // output gradient is zero
        };
        self.reject_inplace_matmul(a, b, c)?;
        let m = self.dim(args[3]);
        let k = self.dim(args[4]);
        let n = self.dim(args[5]);
        self.nt_adjoints(a, b, dc, m, k, n, par)
    }

    /// `sgemm(a, b, c, m, k, n, beta)` computes `C(m x n) = A(m x k) · B(k x n)`. With `dC`:
    /// - `dA(m x k) = dC · Bᵀ`  ->  `sgemm_nt(dC, B, dA, m'=m, k'=n, n'=k)` (`B` stored `[k, n]` is
    ///   exactly the `n' x k'` operand `sgemm_nt` transposes);
    /// - `dB(k x n) = Aᵀ · dC`  ->  `sgemm_tn(A, dC, dB, m'=k, k'=m, n'=n)`.
    pub(crate) fn diff_sgemm(&mut self, args: &[ValueId], par: bool) -> Result<(), String> {
        let (a, b, c) = (args[0], args[1], args[2]);
        let dc = match self.buf_adj.get(&c).copied() {
            Some(dc) => dc,
            None => return Ok(()),
        };
        self.reject_inplace_matmul(a, b, c)?;
        let m = self.dim(args[3]);
        let k = self.dim(args[4]);
        let n = self.dim(args[5]);
        let (an, bn) = (self.remap_v(a), self.remap_v(b));
        if let Some(da) = self.grad_target(a)? {
            let beta = self.beta_for(a);
            let f = pick(self.syms.sgemm_nt, self.syms.sgemm_nt_par, par);
            self.emit(f, vec![dc, bn, da, m, n, k, beta]);
            self.contributed.insert(a);
        }
        if let Some(db) = self.grad_target(b)? {
            let beta = self.beta_for(b);
            let f = pick(self.syms.sgemm_tn, self.syms.sgemm_tn_par, par);
            self.emit(f, vec![an, dc, db, k, m, n, beta]);
            self.contributed.insert(b);
        }
        Ok(())
    }

    /// `sgemm_tn(a, b, c, m, k, n, beta)` computes `C(m x n) = Aᵀ · B` with `A` stored `[k, m]` and
    /// `B` stored `[k, n]` — the weight-gradient GEMM, reached from source as a matmul whose
    /// contraction index is the OUTER index of both operands. With `dC`:
    /// - `dA([k, m]) = B · dCᵀ`  ->  `sgemm_nt(B, dC, dA, m'=k, k'=n, n'=m)`;
    /// - `dB([k, n]) = A · dC`   ->  `sgemm(A, dC, dB, m'=k, k'=m, n'=n)`.
    pub(crate) fn diff_sgemm_tn(&mut self, args: &[ValueId], par: bool) -> Result<(), String> {
        let (a, b, c) = (args[0], args[1], args[2]);
        let dc = match self.buf_adj.get(&c).copied() {
            Some(dc) => dc,
            None => return Ok(()),
        };
        self.reject_inplace_matmul(a, b, c)?;
        let m = self.dim(args[3]);
        let k = self.dim(args[4]);
        let n = self.dim(args[5]);
        let (an, bn) = (self.remap_v(a), self.remap_v(b));
        if let Some(da) = self.grad_target(a)? {
            let beta = self.beta_for(a);
            let f = pick(self.syms.sgemm_nt, self.syms.sgemm_nt_par, par);
            self.emit(f, vec![bn, dc, da, k, n, m, beta]);
            self.contributed.insert(a);
        }
        if let Some(db) = self.grad_target(b)? {
            let beta = self.beta_for(b);
            let f = pick(self.syms.sgemm, self.syms.sgemm_par, par);
            self.emit(f, vec![an, dc, db, k, m, n, beta]);
            self.contributed.insert(b);
        }
        Ok(())
    }

    /// `sgemm_nt_alpha(a, b, c, m, k, n, beta, alpha)`: `C = α·(A · Bᵀ)` — the attention score
    /// `Q·Kᵀ/√d`. The adjoint of the scale is a scale: `dC' = α·dC` into a heap scratch, then the
    /// `nn.Linear` adjoints. The kernel defines `beta != 0` as `α·(C_old + A·Bᵀ)` (not BLAS's
    /// `C_old + α·A·Bᵀ`) and the recognizer never emits it, so only `beta == 0` is differentiated.
    pub(crate) fn diff_sgemm_nt_alpha(
        &mut self,
        args: &[ValueId],
        par: bool,
    ) -> Result<(), String> {
        let (a, b, c) = (args[0], args[1], args[2]);
        let dc = match self.buf_adj.get(&c).copied() {
            Some(dc) => dc,
            None => return Ok(()),
        };
        self.reject_inplace_matmul(a, b, c)?;
        self.require_beta_zero(args[6], "wukong_sgemm_nt_alpha")?;
        let m = self.dim(args[3]);
        let k = self.dim(args[4]);
        let n = self.dim(args[5]);
        let alpha = self.remap_v(args[7]);
        let total = self.mul64(m, n);
        let dcs = self.scratch(total);
        self.velem_scale(dcs, dc, alpha, total);
        self.nt_adjoints(a, b, dcs, m, k, n, par)
    }

    /// `sgemm_nt_epi(a, b, c, m, k, n, beta, bias, act)`: `C = act(A · Bᵀ + bias)` — the fused FFN
    /// epilogue the recognizer folds a `matmul → bias-add → activation` chain into. Before this rule
    /// a performance fusion firing was exactly what removed differentiability.
    ///
    /// `dZ = dC (.) act'(Z)`: identity passes `dC` through; ReLU reads the mask off the OUTPUT
    /// (`relu(z) > 0` exactly when `z > 0`, so no recompute); GELU / SiLU recompute the
    /// pre-activation with the SAME kernel and the identity epilogue (bit-identical to the `Z` the
    /// fused writeback activated) and ride the fused `wukong_vmath2_f32` backward — the epilogue
    /// reuses `vmath`'s `gelu1`/`silu1`, the very functions those backward codes differentiate.
    /// Then the `nn.Linear` adjoints of `dZ`, and `dbias[j] = Σ_i dZ[i, j]`.
    pub(crate) fn diff_sgemm_nt_epi(&mut self, args: &[ValueId], par: bool) -> Result<(), String> {
        let (a, b, c, bias) = (args[0], args[1], args[2], args[7]);
        let dc = match self.buf_adj.get(&c).copied() {
            Some(dc) => dc,
            None => return Ok(()),
        };
        self.reject_inplace_matmul(a, b, c)?;
        if c == bias {
            return Err(format!(
                "autodiff: fused-epilogue GEMM output v{} aliases its bias; the backward reads \
                 the un-overwritten bias",
                c.0
            ));
        }
        self.require_beta_zero(args[6], "wukong_sgemm_nt_epi")?;
        let act = self.const_i64(args[8])?;
        let m = self.dim(args[3]);
        let k = self.dim(args[4]);
        let n = self.dim(args[5]);
        let total = self.mul64(m, n);
        let dz = match act {
            EPI_IDENTITY => dc,
            EPI_RELU => {
                let s = self.scratch(total);
                let cn = self.remap_v(c);
                self.relu_mask_from_output(dc, cn, s, total, false);
                s
            }
            EPI_GELU | EPI_SILU => {
                let (an, bn, biasn) = (self.remap_v(a), self.remap_v(b), self.remap_v(bias));
                let z = self.scratch(total);
                let zero = self.cint(0);
                let ident = self.cint(EPI_IDENTITY);
                let f = self.syms.sgemm_nt_epi;
                self.emit(f, vec![an, bn, z, m, k, n, zero, biasn, ident]);
                let dzb = self.scratch(total);
                let bwd = if act == EPI_GELU {
                    VM2_GELU_BWD
                } else {
                    VM2_SILU_BWD
                };
                self.vmath2_call(z, dc, dzb, total, bwd);
                dzb
            }
            other => {
                return Err(format!(
                    "autodiff: no VJP for the fused GEMM epilogue activation {other} \
                     (supported: identity, ReLU, GELU, SiLU)"
                ))
            }
        };
        self.nt_adjoints(a, b, dz, m, k, n, par)?;
        if !self.is_null_ptr(bias) {
            if let Some(cb) = self.begin(bias, n)? {
                self.colsum(dz, cb.write, m, n);
                self.end(cb);
            }
        }
        Ok(())
    }

    /// The GEMMs whose `beta != 0` the rules do not model (the epilogue kernels fold `beta` under
    /// the scale/activation) must have a constant zero `beta`.
    fn require_beta_zero(&self, beta: ValueId, what: &str) -> Result<(), String> {
        match self.const_i64(beta) {
            Ok(0) => Ok(()),
            _ => Err(format!(
                "autodiff: no VJP for `{what}` with a nonzero or non-constant beta (its epilogue \
                 applies to the accumulated value, which the rule does not model)"
            )),
        }
    }

    /// `sgemv(a, x, y, m, n)` / `sgemv_alpha(a, x, y, m, n, alpha)`: `y = α·A·x` (`A` `[m, n]`).
    /// - `dx[n] = α·Aᵀ·dy`  ->  `wukong_sgevm_f32(dy, A, dx, m, n, α)` (the vector·matrix kernel is
    ///   exactly the transposed GEMV);
    /// - `dA = α·dy ⊗ x`    ->  a `k = 1` GEMM `sgemm(α·dy, x, dA, m, 1, n)`, accumulating via `beta`.
    pub(crate) fn diff_sgemv(
        &mut self,
        args: &[ValueId],
        alpha: Option<ValueId>,
        par: bool,
    ) -> Result<(), String> {
        let (a, x, y) = (args[0], args[1], args[2]);
        let dy = match self.buf_adj.get(&y).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        if y == a || y == x {
            return Err(format!(
                "autodiff: GEMV VJP cannot differentiate an in-place GEMV (output buffer v{} \
                 aliases an input)",
                y.0
            ));
        }
        let m = self.dim(args[3]);
        let n = self.dim(args[4]);
        let alpha_n = match alpha {
            Some(v) => self.remap_v(v),
            None => self.cf32(1.0),
        };
        let (an, xn) = (self.remap_v(a), self.remap_v(x));
        if let Some(c) = self.begin(x, n)? {
            let f = self.syms.sgevm;
            self.emit(f, vec![dy, an, c.write, m, n, alpha_n]);
            self.end(c);
        }
        if let Some(da) = self.grad_target(a)? {
            let s = if alpha.is_some() {
                let s = self.scratch(m);
                self.velem_scale(s, dy, alpha_n, m);
                s
            } else {
                dy
            };
            let one = self.cint(1);
            let beta = self.beta_for(a);
            let f = pick(self.syms.sgemm, self.syms.sgemm_par, par);
            self.emit(f, vec![s, xn, da, m, one, n, beta]);
            self.contributed.insert(a);
        }
        Ok(())
    }

    /// `sgevm(w, a, out, rows, cols, alpha)`: `out[j] = α·Σ_i w[i]·A[i, j]` — the decode-time
    /// attention `probs · V`.
    /// - `dw[rows] = α·A·dout`  ->  `wukong_sgemv_alpha(A, dout, dw, rows, cols, α)`;
    /// - `dA = α·w ⊗ dout`      ->  `sgemm(α·w, dout, dA, rows, 1, cols)`.
    pub(crate) fn diff_sgevm(&mut self, args: &[ValueId], par: bool) -> Result<(), String> {
        let (w, a, out) = (args[0], args[1], args[2]);
        let dout = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        if out == w || out == a {
            return Err(format!(
                "autodiff: GEVM VJP cannot differentiate an in-place vector-matrix product \
                 (output buffer v{} aliases an input)",
                out.0
            ));
        }
        let rows = self.dim(args[3]);
        let cols = self.dim(args[4]);
        let alpha = self.remap_v(args[5]);
        let (wn, an) = (self.remap_v(w), self.remap_v(a));
        if let Some(c) = self.begin(w, rows)? {
            let f = self.syms.sgemv_alpha;
            self.emit(f, vec![an, dout, c.write, rows, cols, alpha]);
            self.end(c);
        }
        if let Some(da) = self.grad_target(a)? {
            let s = self.scratch(rows);
            self.velem_scale(s, wn, alpha, rows);
            let one = self.cint(1);
            let beta = self.beta_for(a);
            let f = pick(self.syms.sgemm, self.syms.sgemm_par, par);
            self.emit(f, vec![s, dout, da, rows, one, cols, beta]);
            self.contributed.insert(a);
        }
        Ok(())
    }

    /// Both matmul VJPs read the forward operands back (`dA` reads `B`, `dB` reads `A`), so a GEMM
    /// whose output buffer aliases either input hands them a buffer the forward has already
    /// overwritten. The matmul rules route through `beta_for`, which merely *observes*
    /// `contributed`, so nothing else on the path would fail: with `c == a` the emitted `dA` would be
    /// a GEMM accumulating into the operand it is reading. Refuse.
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

    // --- broadcast bias (y = act(x + b), b shared across rows) ---------------------------------

    /// `bias_bcast(x, b, out, rows, cols, op)` computes `out[i,j] = act(x[i,j] + b[j])`. The VJP is
    /// the textbook inner-product bias gradient (oneDNN's backward-weights `diff_bias`):
    /// `dz = dout (.) act'(z)`, `dx = dz`, `db[j] = Σ_i dz[i,j]` (one `wukong_colsum_f32`).
    ///
    /// A fused activation recomputes the pre-activation `z = x + b` with the same kernel and the
    /// pure-add op (bit-identical to the `z` the fused pass activated), then takes the full
    /// activation backward. **In place** (`h[i,j] = act(h[i,j] + b[j])`) the output adjoint IS the
    /// input adjoint: the pure add passes it through unchanged and a fused ReLU masks it in place
    /// from the output; any other fused activation has lost its pre-activation and refuses.
    pub(crate) fn diff_bias(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (x, bvec, out) = (args[0], args[1], args[2]);
        let op = self.const_i64(args[5])?;
        let dout = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        if bvec == out {
            return Err(format!(
                "autodiff: broadcast-bias output v{} aliases the bias vector",
                out.0
            ));
        }
        let rows = self.dim(args[3]);
        let cols = self.dim(args[4]);
        let total = self.mul64(rows, cols);
        if out == x {
            if op == VM_RELU {
                let outn = self.remap_v(out);
                self.relu_mask_from_output(dout, outn, dout, total, false);
            } else if op != BIAS_ACT_NONE {
                return Err(format!(
                    "autodiff: no VJP for an in-place `wukong_bias_bcast_f32` with the fused \
                     activation op {op}: the pre-activation was overwritten, and only ReLU \
                     ({VM_RELU}) is recoverable from the output"
                ));
            }
            if let Some(c) = self.begin(bvec, cols)? {
                self.colsum(dout, c.write, rows, cols);
                self.end(c);
            }
            return Ok(());
        }
        let dz = if op == BIAS_ACT_NONE {
            dout
        } else {
            let (xn, bn, outn) = (self.remap_v(x), self.remap_v(bvec), self.remap_v(out));
            let z = self.scratch(total);
            let none = self.cint(BIAS_ACT_NONE);
            let f = self.syms.bias;
            self.emit(f, vec![xn, bn, z, rows, cols, none]);
            let dzb = self.scratch(total);
            self.act_backward(op, dout, z, outn, dzb, total)?;
            dzb
        };
        if let Some(c) = self.begin(x, total)? {
            let one = self.cf32(1.0);
            self.velem_scale(c.write, dz, one, total);
            self.end(c);
        }
        if let Some(c) = self.begin(bvec, cols)? {
            self.colsum(dz, c.write, rows, cols);
            self.end(c);
        }
        Ok(())
    }

    // --- transpose and column reductions -------------------------------------------------------

    /// `transpose(src, dst, rows, cols)`: `dst = srcᵀ`. Its VJP is the transpose back,
    /// `dsrc = transpose(ddst)` — one cache-blocked kernel call.
    pub(crate) fn diff_transpose(&mut self, args: &[ValueId], par: bool) -> Result<(), String> {
        let (src, dst) = (args[0], args[1]);
        let dd = match self.buf_adj.get(&dst).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        if src == dst {
            return Err(format!(
                "autodiff: transpose VJP cannot differentiate an in-place transpose (v{})",
                dst.0
            ));
        }
        let rows = self.dim(args[2]);
        let cols = self.dim(args[3]);
        let total = self.mul64(rows, cols);
        if let Some(c) = self.begin(src, total)? {
            let f = pick(self.syms.transpose, self.syms.transpose_par, par);
            // ddst is [cols, rows]; its transpose is the [rows, cols] dsrc.
            self.emit(f, vec![dd, c.write, cols, rows]);
            self.end(c);
        }
        Ok(())
    }

    /// The column reductions `out[j] = Σ_i x[i,j]` (colsum), `Σ_i x[i,j] / rows` (colmean) and
    /// `Σ_i x[i,j]²` (colsumsq — the bias / LayerNorm-statistics reductions). Each backward is a
    /// row-broadcast of `dout` — a zero fill plus an in-place `wukong_bias_bcast_f32` — scaled by
    /// `1/rows`, or multiplied by `2x`.
    pub(crate) fn diff_colreduce(
        &mut self,
        kind: crate::tape::Kernel,
        args: &[ValueId],
    ) -> Result<(), String> {
        let (x, out) = (args[0], args[1]);
        let dout = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        if x == out {
            return Err(format!(
                "autodiff: column-reduction output v{} aliases its input",
                out.0
            ));
        }
        let rows = self.dim(args[2]);
        let cols = self.dim(args[3]);
        let total = self.mul64(rows, cols);
        if let Some(c) = self.begin(x, total)? {
            let zero = self.cf32(0.0);
            self.fill(c.write, total, zero);
            let none = self.cint(BIAS_ACT_NONE);
            let f = self.syms.bias;
            self.emit(f, vec![c.write, dout, c.write, rows, cols, none]);
            match kind {
                crate::tape::Kernel::ColMean => {
                    let rf = self.b.build(F32, Op::Cast(CastKind::SiToFp, rows, F32));
                    let one = self.cf32(1.0);
                    let inv = self.fdiv(one, rf, &F32);
                    self.velem_scale(c.write, c.write, inv, total);
                }
                crate::tape::Kernel::ColSumSq => {
                    let xn = self.remap_v(x);
                    self.velem_hadamard(c.write, c.write, xn, total);
                    let two = self.cf32(2.0);
                    self.velem_scale(c.write, c.write, two, total);
                }
                _ => {}
            }
            self.end(c);
        }
        Ok(())
    }
}
