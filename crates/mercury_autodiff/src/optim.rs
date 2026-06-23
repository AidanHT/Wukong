//! Optimizer steps as Mercury MIR kernels.
//!
//! The backward pass produces gradients; an optimizer turns them into a parameter update. SGD is a
//! single streaming-affine op (`w -= lr*g` is one `mercury_velem_f32` call), so it needs no kernel
//! of its own. **AdamW** does — moments, bias correction, an `rsqrt`, and decoupled weight decay,
//! fused into one pass over the parameters — so [`build_adamw_step`] emits it as a counted-loop MIR
//! function. Keeping it in MIR means the *same* update runs on the interpreter (gated here against a
//! reference) and, once lowered, on the GPU as one fused kernel — the fusion a tensor library splits
//! into several launches.

use mercury_mir::{BinOp, Builder, CmpOp, Function, MirType, Op, ValueId};
use mercury_span::{Interner, Symbol};

const F32: MirType = MirType::F32;
const I64T: MirType = MirType::I64;
const PTR: MirType = MirType::Ptr;

/// The element index into the hyperparameter buffer an AdamW step reads. The caller fills these
/// per step (the bias corrections `bc1 = 1 - beta1^t`, `bc2 = 1 - beta2^t` advance with `t`).
pub mod hp {
    pub const LR: usize = 0;
    pub const BETA1: usize = 1;
    pub const BETA2: usize = 2;
    pub const EPS: usize = 3;
    pub const WD: usize = 4;
    /// `1 - beta1^t`
    pub const BC1: usize = 5;
    /// `1 - beta2^t`
    pub const BC2: usize = 6;
    pub const LEN: usize = 7;
}

/// Build a fused **AdamW** update kernel `adamw_step(w, g, m, v, hp)` for a parameter tensor of `n`
/// elements. Buffers: `w` (weights, updated in place), `g` (gradient), `m`/`v` (first/second moment
/// state, updated in place), `hp` (the [`hp`] hyperparameter vector). Per element:
///
/// ```text
/// m = beta1*m + (1-beta1)*g
/// v = beta2*v + (1-beta2)*g^2
/// w -= lr * ( (m/bc1) / (sqrt(v/bc2) + eps) + wd*w )     // decoupled weight decay
/// ```
///
/// All parameters are pointers (the kernel-entry ABI), so it runs directly under
/// `mercury_interp::run_kernel_f32`.
pub fn build_adamw_step(it: &mut Interner, n: usize) -> Function {
    // The element count is baked in, so the name carries it — two AdamW kernels for different-sized
    // parameter tensors must be distinct functions (a shared name would alias in a `Program`).
    let name = it.intern(&format!("adamw_step_{n}"));
    let mut b = Builder::new(name, MirType::Void);
    let w = b.add_param(PTR);
    let g = b.add_param(PTR);
    let m = b.add_param(PTR);
    let v = b.add_param(PTR);
    let hpb = b.add_param(PTR);

    // Load the (loop-invariant) hyperparameters once.
    let lr = load_const(&mut b, hpb, hp::LR);
    let beta1 = load_const(&mut b, hpb, hp::BETA1);
    let beta2 = load_const(&mut b, hpb, hp::BETA2);
    let eps = load_const(&mut b, hpb, hp::EPS);
    let wd = load_const(&mut b, hpb, hp::WD);
    let bc1 = load_const(&mut b, hpb, hp::BC1);
    let bc2 = load_const(&mut b, hpb, hp::BC2);
    let one = cf(&mut b, 1.0);
    let om1 = b.build(F32, Op::Bin(BinOp::FSub, one, beta1)); // 1 - beta1
    let om2 = b.build(F32, Op::Bin(BinOp::FSub, one, beta2)); // 1 - beta2
    let nv = ci(&mut b, n as i64);

    counted_loop(&mut b, nv, |b, i| {
        let gi = load_at(b, g, i);
        let mi_old = load_at(b, m, i);
        let vi_old = load_at(b, v, i);
        let wi_old = load_at(b, w, i);

        // m = beta1*m + (1-beta1)*g
        let t1 = b.build(F32, Op::Bin(BinOp::FMul, om1, gi));
        let mi = b.build(F32, Op::Fma(beta1, mi_old, t1));
        // v = beta2*v + (1-beta2)*g^2
        let g2 = b.build(F32, Op::Bin(BinOp::FMul, gi, gi));
        let t2 = b.build(F32, Op::Bin(BinOp::FMul, om2, g2));
        let vi = b.build(F32, Op::Fma(beta2, vi_old, t2));
        store_at(b, m, i, mi);
        store_at(b, v, i, vi);

        // w -= lr * ( (m/bc1) / (sqrt(v/bc2) + eps) + wd*w )
        let mhat = b.build(F32, Op::Bin(BinOp::FDiv, mi, bc1));
        let vhat = b.build(F32, Op::Bin(BinOp::FDiv, vi, bc2));
        let sq = b.build(F32, Op::Sqrt(vhat));
        let denom = b.build(F32, Op::Bin(BinOp::FAdd, sq, eps));
        let step = b.build(F32, Op::Bin(BinOp::FDiv, mhat, denom));
        let wdterm = b.build(F32, Op::Bin(BinOp::FMul, wd, wi_old));
        let upd = b.build(F32, Op::Bin(BinOp::FAdd, step, wdterm));
        let lupd = b.build(F32, Op::Bin(BinOp::FMul, lr, upd));
        let wi = b.build(F32, Op::Bin(BinOp::FSub, wi_old, lupd));
        store_at(b, w, i, wi);
    });
    b.ret(None);
    b.finish()
}

/// Emit `for idx in 0..n { body(builder, idx) }` as a counted loop (header / body / exit blocks),
/// leaving the builder positioned at the exit. The body must be straight-line.
fn counted_loop<F: FnOnce(&mut Builder, ValueId)>(b: &mut Builder, n: ValueId, body: F) {
    let zero = ci(b, 0);
    let header = b.new_block();
    let bodyb = b.new_block();
    let exit = b.new_block();
    let i_h = b.block_param(header, I64T);
    let i_b = b.block_param(bodyb, I64T);
    b.br(header, vec![zero]);
    b.switch_to(header);
    let cond = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i_h, n));
    b.cond_br(cond, bodyb, vec![i_h], exit, vec![]);
    b.switch_to(bodyb);
    body(b, i_b);
    let one = ci(b, 1);
    let i1 = b.build(I64T, Op::Bin(BinOp::Add, i_b, one));
    b.br(header, vec![i1]);
    b.switch_to(exit);
}

fn load_at(b: &mut Builder, base: ValueId, idx: ValueId) -> ValueId {
    let p = b.build(
        PTR,
        Op::Gep {
            ptr: base,
            index: idx,
            elem: F32,
        },
    );
    b.build(F32, Op::Load(p, F32))
}

fn load_const(b: &mut Builder, base: ValueId, idx: usize) -> ValueId {
    let iv = ci(b, idx as i64);
    load_at(b, base, iv)
}

fn store_at(b: &mut Builder, base: ValueId, idx: ValueId, val: ValueId) {
    let p = b.build(
        PTR,
        Op::Gep {
            ptr: base,
            index: idx,
            elem: F32,
        },
    );
    b.build_void(Op::Store { ptr: p, value: val });
}

fn ci(b: &mut Builder, x: i64) -> ValueId {
    b.build(I64T, Op::ConstInt(x as i128, I64T))
}

fn cf(b: &mut Builder, x: f64) -> ValueId {
    b.build(F32, Op::ConstFloat(x, F32))
}

/// The interned name [`build_adamw_step`] gives its kernel for parameter count `n` (it encodes the
/// size, so distinct sizes are distinct functions) — for looking it up in a `Program`.
pub fn adamw_step_name(it: &mut Interner, n: usize) -> Symbol {
    it.intern(&format!("adamw_step_{n}"))
}
