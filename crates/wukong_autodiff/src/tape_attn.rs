//! The fused scaled-dot-product attention VJP for the buffer tape.
//!
//! `wukong_attention_f32(q, k, v, o, s, d, scale, causal)` computes one head of
//! `O = softmax(scale·Q·Kᵀ [+ causal mask])·V` with every operand row-major `[s, d]`, through an
//! online (flash) softmax that never materializes the `s×s` weights. Its backward is the
//! FlashAttention recompute: the forward keeps nothing, so the backward rebuilds the weights `P` from
//! `Q` and `K` with the kernels the materialized forward would have used, then runs the four
//! adjoint GEMMs — every one a tuned runtime kernel the tape already emits:
//!
//! | step | kernel |
//! |---|---|
//! | `P = scale·Q·Kᵀ` | `wukong_sgemm_nt_alpha` |
//! | causal mask (`P[i, j] = -inf`, `j > i`) | one synthesized triangle loop |
//! | `P = softmax_rows(P)` | `wukong_norm_f32` (`NORM_SOFTMAX`, in place) |
//! | `dV = Pᵀ·dO` | `wukong_sgemm_tn` |
//! | `dP = dO·Vᵀ` | `wukong_sgemm_nt` |
//! | `dS = P ⊙ (dP − Σ_j P ⊙ dP)` | `wukong_softmax_bwd_f32` (in place) |
//! | `dS ← scale·dS` | `wukong_velem_f32` |
//! | `dQ = dS·K` | `wukong_sgemm` |
//! | `dK = dSᵀ·Q` | `wukong_sgemm_tn` |
//!
//! Masked weights come out of the softmax as exactly `+0.0` (the runtime `exp` flushes `-inf` to
//! zero), so `dS` is zero above the diagonal and no masked key contributes to `dQ` or `dK`, and none
//! of `dV`'s rows receives a weight a query never used. The recomputed weights differ from the
//! forward's online ones only by the reassociation of the softmax sums — the same tolerance-level
//! relation the kernel's own tests hold it to against the materialized reference.

use crate::tape::{F32, I64T, NORM_SOFTMAX};
use crate::Vjp;
use wukong_mir::{BinOp, CmpOp, MirType, Op, ValueId};

/// How the backward reproduces the forward's causal mask.
enum Mask {
    /// The flag is a compile-time constant: no mask, or the full upper triangle.
    Static(bool),
    /// The flag is a run-time `i64`: the triangle loop selects `-inf` where the flag is nonzero.
    Dynamic(ValueId),
}

impl<'a> Vjp<'a> {
    /// `attention(q, k, v, o, s, d, scale, causal)` — see the module docs for the recipe. `dQ` and
    /// `dK` share the `dS` pass, which is skipped when neither is wanted (a `V`-only gradient needs
    /// only the recomputed weights). Each adjoint GEMM accumulates through `beta_for`, so a buffer
    /// that is two of `q`/`k`/`v` at once (self-attention over one buffer) sums its contributions.
    pub(crate) fn diff_attention(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (q, k, v, o) = (args[0], args[1], args[2], args[3]);
        let d_o = match self.buf_adj.get(&o).copied() {
            Some(g) => g,
            None => return Ok(()),
        };
        if o == q || o == k || o == v {
            return Err(format!(
                "autodiff: attention output v{} aliases an input; the backward recomputes the \
                 attention weights from the un-overwritten Q and K and reads V",
                o.0
            ));
        }
        if self.fwd.value_type(args[6]) != &F32 {
            return Err(format!(
                "autodiff: attention scale operand v{} is {}, not f32 — not the recognized \
                 kernel ABI",
                args[6].0,
                self.fwd.value_type(args[6]).display()
            ));
        }
        let mask = match self.const_i64(args[7]) {
            Ok(flag) => Mask::Static(flag != 0),
            Err(_) => {
                if !self.fwd.value_type(args[7]).is_int() {
                    return Err(format!(
                        "autodiff: attention causal operand v{} is {}, not an integer",
                        args[7].0,
                        self.fwd.value_type(args[7]).display()
                    ));
                }
                Mask::Dynamic(self.dim(args[7]))
            }
        };
        let s = self.dim(args[4]);
        let d = self.dim(args[5]);
        let scale = self.remap_v(args[6]);
        let (qn, kn, vn) = (self.remap_v(q), self.remap_v(k), self.remap_v(v));
        let ss = self.mul64(s, s);

        // The attention weights, recomputed: P = softmax_rows(scale·Q·Kᵀ [masked]).
        let p = self.scratch(ss);
        let zero = self.cint(0);
        let f = self.syms.sgemm_nt_alpha;
        self.emit(f, vec![qn, kn, p, s, d, s, zero, scale]);
        match mask {
            Mask::Static(false) => {}
            Mask::Static(true) => self.mask_future_keys(p, s, None),
            Mask::Dynamic(flag) => {
                let z = self.cint(0);
                let on = self.b.build(MirType::I1, Op::Cmp(CmpOp::Ne, flag, z));
                self.mask_future_keys(p, s, Some(on));
            }
        }
        let eps = self.cint(0);
        let softmax = self.cint(NORM_SOFTMAX);
        let f = self.syms.norm;
        self.emit(f, vec![p, p, s, s, eps, softmax]);

        // dV = Pᵀ·dO: `sgemm_tn(P, dO, dV, m=s, k=s, n=d)` (P stored [s, s] is the [k, m] operand).
        if let Some(gv) = self.grad_target(v)? {
            let beta = self.beta_for(v);
            let f = self.syms.sgemm_tn;
            self.emit(f, vec![p, d_o, gv, s, s, d, beta]);
            self.contributed.insert(v);
        }

        let gq = self.grad_target(q)?;
        let gk = self.grad_target(k)?;
        if gq.is_none() && gk.is_none() {
            return Ok(());
        }
        // dP = dO·Vᵀ: `sgemm_nt(dO, V, dP, m=s, k=d, n=s)`.
        let dp = self.scratch(ss);
        let zero = self.cint(0);
        let f = self.syms.sgemm_nt;
        self.emit(f, vec![d_o, vn, dp, s, d, s, zero]);
        // dS = P ⊙ (dP − rowsum(P ⊙ dP)), in place (the kernel permits `dx == dy`), then the scale
        // the scores carried: d(Q·Kᵀ) = scale·dS.
        let f = self.syms.softmax_bwd;
        self.emit(f, vec![p, dp, dp, s, s]);
        self.velem_scale(dp, dp, scale, ss);
        // dQ = dS·K: `sgemm(dS, K, dQ, m=s, k=s, n=d)`.
        if let Some(gq) = gq {
            let beta = self.beta_for(q);
            let f = self.syms.sgemm;
            self.emit(f, vec![dp, kn, gq, s, s, d, beta]);
            self.contributed.insert(q);
        }
        // dK = dSᵀ·Q: `sgemm_tn(dS, Q, dK, m=s, k=s, n=d)`.
        if let Some(gk) = gk {
            let beta = self.beta_for(k);
            let f = self.syms.sgemm_tn;
            self.emit(f, vec![dp, qn, gk, s, s, d, beta]);
            self.contributed.insert(k);
        }
        Ok(())
    }

    /// Overwrite the strict upper triangle of the row-major `s×s` score matrix `p` (key `j > i` for
    /// query `i`) with `-inf` — the forward's causal mask, which the softmax turns into an exact zero
    /// weight. With `on = Some(flag)` each masked slot becomes `select(flag, -inf, p[i, j])`, the
    /// run-time spelling of a causal flag that is not a compile-time constant.
    fn mask_future_keys(&mut self, p: ValueId, s: ValueId, on: Option<ValueId>) {
        let neg_inf = self.cf32(f64::NEG_INFINITY);
        let one = self.cint(1);
        let outer = self.open_loop(s);
        let i = outer.idx;
        let first = self.b.build(I64T, Op::Bin(BinOp::Add, i, one)); // the first masked key, i + 1
        let count = self.b.build(I64T, Op::Bin(BinOp::Sub, s, first));
        let row = self.mul64(i, s);
        let base = self.b.build(I64T, Op::Bin(BinOp::Add, row, first));
        let inner = self.open_loop(count);
        let idx = self.b.build(I64T, Op::Bin(BinOp::Add, base, inner.idx));
        let value = match on {
            None => neg_inf,
            Some(flag) => {
                let old = self.load_at(p, idx);
                self.b.build(F32, Op::Select(flag, neg_inf, old))
            }
        };
        self.store_at(p, idx, value);
        self.close_loop(inner);
        self.close_loop(outer);
    }
}
