//! Finite-difference gradient gate.
//!
//! Each test builds a forward MIR function (in `f64`, so the central-difference reference stays far
//! below tolerance), runs it through [`grad`], and asserts the analytic gradient the transform emits
//! matches the central finite difference over the whole gradient buffer — and, where a closed form
//! exists, matches that exactly. Both passes run on the `wukong_interp` oracle.

use crate::grad;
use wukong_interp::run_kernel_f64;
use wukong_mir::{BinOp, Builder, Function, MirType, Op, Program, ValueId};
use wukong_span::{Interner, Symbol};

const F64: MirType = MirType::F64;
const PTR: MirType = MirType::Ptr;

/// A forward kernel plus the metadata the gate needs to drive it.
struct Fwd {
    func: Function,
    /// Element count of each parameter buffer, in parameter order.
    lens: Vec<usize>,
    /// Parameter index whose element 0 receives the scalar loss.
    loss_out: usize,
}

/// Build `Program { fwd, fwd_grad }` and return (program, grad_fn_name).
fn build(fwd: &Function, wrt: &[usize], interner: &mut Interner) -> (Program, Symbol) {
    let g = grad(fwd, wrt, interner).expect("grad transform failed");
    let gname = g.name;
    let prog = Program {
        funcs: vec![fwd.clone(), g],
        statics: Vec::new(),
        level: wukong_mir::MirLevel::Low,
    };
    (prog, gname)
}

/// Run the forward kernel with the given input buffers and return the scalar loss (`out[0]`).
fn loss_at(
    prog: &Program,
    fwd_name: Symbol,
    bufs: &mut [Vec<f64>],
    loss_out: usize,
    interner: &Interner,
) -> f64 {
    let mut views: Vec<&mut [f64]> = bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
    run_kernel_f64(prog, fwd_name, &mut views, interner).expect("forward run failed");
    bufs[loss_out][0]
}

/// Run the gradient kernel and return the gradient buffer for each `wrt` input (in `wrt` order).
fn analytic_grad(
    prog: &Program,
    gname: Symbol,
    fwd: &Fwd,
    inputs: &[Vec<f64>],
    wrt: &[usize],
    interner: &Interner,
) -> Vec<Vec<f64>> {
    // Forward params, then one zeroed gradient buffer per wrt input.
    let mut bufs: Vec<Vec<f64>> = inputs.to_vec();
    for &wi in wrt {
        bufs.push(vec![0.0; fwd.lens[wi]]);
    }
    let mut views: Vec<&mut [f64]> = bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
    run_kernel_f64(prog, gname, &mut views, interner).expect("gradient run failed");
    let nparams = fwd.func.params.len();
    (0..wrt.len()).map(|i| bufs[nparams + i].clone()).collect()
}

/// Central finite-difference gradient of the loss w.r.t. each `wrt` input element.
fn fd_grad(
    prog: &Program,
    fwd: &Fwd,
    inputs: &[Vec<f64>],
    wrt: &[usize],
    eps: f64,
    interner: &Interner,
) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for &wi in wrt {
        let mut g = vec![0.0; fwd.lens[wi]];
        for j in 0..fwd.lens[wi] {
            let mut bufs = inputs.to_vec();
            let orig = bufs[wi][j];
            bufs[wi][j] = orig + eps;
            let lp = loss_at(prog, fwd.func.name, &mut bufs, fwd.loss_out, interner);
            bufs[wi][j] = orig - eps;
            let lm = loss_at(prog, fwd.func.name, &mut bufs, fwd.loss_out, interner);
            g[j] = (lp - lm) / (2.0 * eps);
        }
        out.push(g);
    }
    out
}

/// Gate: analytic gradient must match the finite difference over every element.
fn gate(fwd: &Fwd, wrt: &[usize], inputs: &[Vec<f64>], interner: &mut Interner) -> Vec<Vec<f64>> {
    let (prog, gname) = build(&fwd.func, wrt, interner);
    let analytic = analytic_grad(&prog, gname, fwd, inputs, wrt, interner);
    let fd = fd_grad(&prog, fwd, inputs, wrt, 1e-6, interner);
    for (gi, (a, f)) in analytic.iter().zip(fd.iter()).enumerate() {
        for (j, (&av, &fv)) in a.iter().zip(f.iter()).enumerate() {
            let tol = 1e-6 + 1e-5 * fv.abs();
            assert!(
                (av - fv).abs() <= tol,
                "wrt#{gi}[{j}]: analytic {av} vs finite-diff {fv} (|d|={:.3e} > tol {:.3e})",
                (av - fv).abs(),
                tol
            );
        }
    }
    analytic
}

/// Assert a gradient buffer equals a closed-form expectation to tight f64 tolerance.
fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (j, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 + 1e-9 * w.abs(),
            "{what}[{j}]: got {g}, want {w}"
        );
    }
}

fn sym(interner: &mut Interner, s: &str) -> Symbol {
    interner.intern(s)
}

/// Helper: load element `idx` (a constant) of buffer parameter `p`.
fn load_elem(b: &mut Builder, p: ValueId, idx: i64) -> ValueId {
    if idx == 0 {
        b.build(F64, Op::Load(p, F64))
    } else {
        let i = b.build(MirType::I64, Op::ConstInt(idx as i128, MirType::I64));
        let gp = b.build(PTR, Op::Gep { ptr: p, index: i, elem: F64 });
        b.build(F64, Op::Load(gp, F64))
    }
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[test]
fn square_scalar() {
    // f(x) = x * x  ->  df/dx = 2x
    let mut it = Interner::default();
    let mut b = Builder::new(sym(&mut it, "square"), MirType::Void);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    let x0 = load_elem(&mut b, x, 0);
    let sq = b.build(F64, Op::Bin(BinOp::FMul, x0, x0));
    b.build_void(Op::Store { ptr: out, value: sq });
    b.ret(Some(sq));
    let fwd = Fwd {
        func: b.finish(),
        lens: vec![1, 1],
        loss_out: 1,
    };

    let inputs = vec![vec![3.0], vec![0.0]];
    let g = gate(&fwd, &[0], &inputs, &mut it);
    assert_close(&g[0], &[6.0], "d(x^2)/dx at x=3");
}

#[test]
fn two_op_chain() {
    // f(a, b) = a*b + a   ->  df/da = b + 1, df/db = a
    let mut it = Interner::default();
    let mut b = Builder::new(sym(&mut it, "chain"), MirType::Void);
    let pa = b.add_param(PTR);
    let pb = b.add_param(PTR);
    let out = b.add_param(PTR);
    let a = load_elem(&mut b, pa, 0);
    let bb = load_elem(&mut b, pb, 0);
    let ab = b.build(F64, Op::Bin(BinOp::FMul, a, bb));
    let l = b.build(F64, Op::Bin(BinOp::FAdd, ab, a));
    b.build_void(Op::Store { ptr: out, value: l });
    b.ret(Some(l));
    let fwd = Fwd {
        func: b.finish(),
        lens: vec![1, 1, 1],
        loss_out: 2,
    };

    let inputs = vec![vec![2.0], vec![5.0], vec![0.0]];
    let g = gate(&fwd, &[0, 1], &inputs, &mut it);
    assert_close(&g[0], &[6.0], "df/da = b+1 at b=5"); // 5 + 1
    assert_close(&g[1], &[2.0], "df/db = a at a=2");
}

#[test]
fn div_and_sub() {
    // f(a, b) = a/b - b  ->  df/da = 1/b, df/db = -a/b^2 - 1
    let mut it = Interner::default();
    let mut b = Builder::new(sym(&mut it, "divsub"), MirType::Void);
    let pa = b.add_param(PTR);
    let pb = b.add_param(PTR);
    let out = b.add_param(PTR);
    let a = load_elem(&mut b, pa, 0);
    let bb = load_elem(&mut b, pb, 0);
    let q = b.build(F64, Op::Bin(BinOp::FDiv, a, bb));
    let l = b.build(F64, Op::Bin(BinOp::FSub, q, bb));
    b.build_void(Op::Store { ptr: out, value: l });
    b.ret(Some(l));
    let fwd = Fwd {
        func: b.finish(),
        lens: vec![1, 1, 1],
        loss_out: 2,
    };

    let (av, bv) = (7.0, 2.0);
    let inputs = vec![vec![av], vec![bv], vec![0.0]];
    let g = gate(&fwd, &[0, 1], &inputs, &mut it);
    assert_close(&g[0], &[1.0 / bv], "df/da = 1/b");
    assert_close(&g[1], &[-av / (bv * bv) - 1.0], "df/db = -a/b^2 - 1");
}

#[test]
fn sqrt_rule() {
    // f(x) = sqrt(x)  ->  df/dx = 0.5 / sqrt(x)
    let mut it = Interner::default();
    let mut b = Builder::new(sym(&mut it, "sq"), MirType::Void);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    let x0 = load_elem(&mut b, x, 0);
    let r = b.build(F64, Op::Sqrt(x0));
    b.build_void(Op::Store { ptr: out, value: r });
    b.ret(Some(r));
    let fwd = Fwd {
        func: b.finish(),
        lens: vec![1, 1],
        loss_out: 1,
    };

    let xv = 9.0;
    let inputs = vec![vec![xv], vec![0.0]];
    let g = gate(&fwd, &[0], &inputs, &mut it);
    assert_close(&g[0], &[0.5 / xv.sqrt()], "d(sqrt x)/dx at x=9");
}

#[test]
fn fma_rule() {
    // f(a, b, c) = fma(a, b, c) = a*b + c  ->  (b, a, 1)
    let mut it = Interner::default();
    let mut b = Builder::new(sym(&mut it, "fma"), MirType::Void);
    let pa = b.add_param(PTR);
    let pb = b.add_param(PTR);
    let pc = b.add_param(PTR);
    let out = b.add_param(PTR);
    let a = load_elem(&mut b, pa, 0);
    let bb = load_elem(&mut b, pb, 0);
    let c = load_elem(&mut b, pc, 0);
    let l = b.build(F64, Op::Fma(a, bb, c));
    b.build_void(Op::Store { ptr: out, value: l });
    b.ret(Some(l));
    let fwd = Fwd {
        func: b.finish(),
        lens: vec![1, 1, 1, 1],
        loss_out: 3,
    };

    let inputs = vec![vec![3.0], vec![4.0], vec![5.0], vec![0.0]];
    let g = gate(&fwd, &[0, 1, 2], &inputs, &mut it);
    assert_close(&g[0], &[4.0], "df/da = b");
    assert_close(&g[1], &[3.0], "df/db = a");
    assert_close(&g[2], &[1.0], "df/dc = 1");
}

#[test]
fn sum_of_squares_vector() {
    // f(x) = sum_i x[i]^2  over a length-N buffer  ->  df/dx[i] = 2 x[i].
    // Exercises repeated geps and adjoint accumulation across many loads.
    const N: usize = 6;
    let mut it = Interner::default();
    let mut b = Builder::new(sym(&mut it, "ssq"), MirType::Void);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    // acc = 0; for each i: acc = fma(x[i], x[i], acc)  (unrolled, straight-line)
    let mut acc = b.build(F64, Op::ConstFloat(0.0, F64));
    for i in 0..N as i64 {
        let xi = load_elem(&mut b, x, i);
        acc = b.build(F64, Op::Fma(xi, xi, acc));
    }
    b.build_void(Op::Store { ptr: out, value: acc });
    b.ret(Some(acc));
    let fwd = Fwd {
        func: b.finish(),
        lens: vec![N, 1],
        loss_out: 1,
    };

    let xs: Vec<f64> = (0..N).map(|i| (i as f64) - 2.5).collect();
    let inputs = vec![xs.clone(), vec![0.0]];
    let g = gate(&fwd, &[0], &inputs, &mut it);
    let want: Vec<f64> = xs.iter().map(|&v| 2.0 * v).collect();
    assert_close(&g[0], &want, "d(sum x^2)/dx");
}

#[test]
fn relu_via_select() {
    // f(x) = sum_i relu(x[i]) with relu(v) = select(v > 0, v, 0)  ->  df/dx[i] = (x[i] > 0) ? 1 : 0
    const N: usize = 5;
    let mut it = Interner::default();
    let mut b = Builder::new(sym(&mut it, "relusum"), MirType::Void);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    let zero = b.build(F64, Op::ConstFloat(0.0, F64));
    let mut acc = b.build(F64, Op::ConstFloat(0.0, F64));
    for i in 0..N as i64 {
        let xi = load_elem(&mut b, x, i);
        let pos = b.build(MirType::I1, Op::Cmp(wukong_mir::CmpOp::Fogt, xi, zero));
        let r = b.build(F64, Op::Select(pos, xi, zero));
        acc = b.build(F64, Op::Bin(BinOp::FAdd, acc, r));
    }
    b.build_void(Op::Store { ptr: out, value: acc });
    b.ret(Some(acc));
    let fwd = Fwd {
        func: b.finish(),
        lens: vec![N, 1],
        loss_out: 1,
    };

    // Avoid the kink at exactly 0 (FD is undefined there).
    let xs = vec![-2.0, 1.5, -0.5, 3.0, 0.25];
    let inputs = vec![xs.clone(), vec![0.0]];
    let g = gate(&fwd, &[0], &inputs, &mut it);
    let want: Vec<f64> = xs.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect();
    assert_close(&g[0], &want, "d(sum relu)/dx");
}

// ---------------------------------------------------------------------------------------------
// End-to-end vertical slice: a small MLP, fully unrolled in scalar f64 MIR, differentiated w.r.t.
// both weight matrices, gradients FD-gated, then trained by SGD with the loss asserted to fall.
// matmul appears as FMA chains, relu as select; this proves the engine composes correctly into a
// trainable network using only the scalar core — no tensor kernels yet.
// ---------------------------------------------------------------------------------------------

const IN: usize = 3;
const HID: usize = 4;
const OUT: usize = 2;

/// Forward: `h = relu(W1 . x)`, `y = W2 . h`, `loss = sum_o (y[o] - t[o])^2` (MSE).
/// Params: [W1 (HID*IN), x (IN), W2 (OUT*HID), t (OUT), out (1)]; differentiable inputs W1, W2.
fn build_mlp(it: &mut Interner) -> Fwd {
    let mut b = Builder::new(sym(it, "mlp"), MirType::Void);
    let w1 = b.add_param(PTR);
    let x = b.add_param(PTR);
    let w2 = b.add_param(PTR);
    let t = b.add_param(PTR);
    let out = b.add_param(PTR);
    let zero = b.build(F64, Op::ConstFloat(0.0, F64));

    let xs: Vec<ValueId> = (0..IN as i64).map(|k| load_elem(&mut b, x, k)).collect();

    // Hidden layer: h[j] = relu(sum_k W1[j*IN + k] * x[k]).
    let mut h = Vec::with_capacity(HID);
    for j in 0..HID {
        let mut acc = b.build(F64, Op::ConstFloat(0.0, F64));
        for k in 0..IN {
            let w = load_elem(&mut b, w1, (j * IN + k) as i64);
            acc = b.build(F64, Op::Fma(w, xs[k], acc));
        }
        let pos = b.build(MirType::I1, Op::Cmp(wukong_mir::CmpOp::Fogt, acc, zero));
        h.push(b.build(F64, Op::Select(pos, acc, zero)));
    }

    // Output + MSE loss: loss = sum_o (sum_j W2[o*HID + j] * h[j] - t[o])^2.
    let mut loss = b.build(F64, Op::ConstFloat(0.0, F64));
    for o in 0..OUT {
        let mut acc = b.build(F64, Op::ConstFloat(0.0, F64));
        for j in 0..HID {
            let w = load_elem(&mut b, w2, (o * HID + j) as i64);
            acc = b.build(F64, Op::Fma(w, h[j], acc));
        }
        let to = load_elem(&mut b, t, o as i64);
        let diff = b.build(F64, Op::Bin(BinOp::FSub, acc, to));
        loss = b.build(F64, Op::Fma(diff, diff, loss)); // loss += diff^2
    }
    b.build_void(Op::Store { ptr: out, value: loss });
    b.ret(Some(loss));
    Fwd {
        func: b.finish(),
        lens: vec![HID * IN, IN, OUT * HID, OUT, 1],
        loss_out: 4,
    }
}

/// Tiny deterministic LCG → values in roughly [-0.5, 0.5], for reproducible weight init.
fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f64 / (1u64 << 24) as f64) - 0.5
}

#[test]
fn mlp_gradient_matches_finite_difference() {
    let mut it = Interner::default();
    let fwd = build_mlp(&mut it);
    let mut seed = 0x1234_5678u64;
    let w1: Vec<f64> = (0..HID * IN).map(|_| lcg(&mut seed)).collect();
    let w2: Vec<f64> = (0..OUT * HID).map(|_| lcg(&mut seed)).collect();
    let x: Vec<f64> = vec![0.7, -0.3, 0.9];
    let t: Vec<f64> = vec![0.4, -0.6];
    let inputs = vec![w1, x, w2, t, vec![0.0]];
    // Gradients w.r.t. both weight matrices, full buffer, gated against the central difference.
    gate(&fwd, &[0, 2], &inputs, &mut it);
}

#[test]
fn mlp_sgd_decreases_loss() {
    let mut it = Interner::default();
    let fwd = build_mlp(&mut it);
    let (prog, gname) = build(&fwd.func, &[0, 2], &mut it);

    let mut seed = 0xC0FFEEu64;
    let mut w1: Vec<f64> = (0..HID * IN).map(|_| lcg(&mut seed)).collect();
    let mut w2: Vec<f64> = (0..OUT * HID).map(|_| lcg(&mut seed)).collect();
    let x: Vec<f64> = vec![0.5, 0.8, -0.4];
    let t: Vec<f64> = vec![0.7, -0.2];

    let loss_now = |w1: &[f64], w2: &[f64], it: &Interner| -> f64 {
        let mut bufs = vec![
            w1.to_vec(),
            x.clone(),
            w2.to_vec(),
            t.clone(),
            vec![0.0],
        ];
        loss_at(&prog, fwd.func.name, &mut bufs, 4, it)
    };

    let lr = 0.03;
    let l0 = loss_now(&w1, &w2, &it);
    let mut prev = l0;
    let mut last = l0;
    for step in 0..400 {
        let inputs = vec![w1.clone(), x.clone(), w2.clone(), t.clone(), vec![0.0]];
        let g = analytic_grad(&prog, gname, &fwd, &inputs, &[0, 2], &it);
        for (wi, gi) in w1.iter_mut().zip(&g[0]) {
            *wi -= lr * gi;
        }
        for (wi, gi) in w2.iter_mut().zip(&g[1]) {
            *wi -= lr * gi;
        }
        last = loss_now(&w1, &w2, &it);
        // Full-batch GD with a small step size descends monotonically here (allow ~rounding slack).
        assert!(
            last <= prev + 1e-9,
            "loss rose at step {step}: {prev} -> {last}"
        );
        prev = last;
    }
    // And it should make substantial progress, not just inch down.
    assert!(
        last < 0.1 * l0,
        "SGD did not converge: loss {l0} -> {last} after 400 steps"
    );
}

// ---------------------------------------------------------------------------------------------
// Tensor-tape tests: forward passes built from real runtime-kernel calls (sgemm_nt = nn.Linear,
// vmath = activation, sreduce = reduction, velem = residual). These run in f32 (the kernels are
// f32), so the gate pairs a looser f32 finite difference with a tight f64 closed-form cross-check
// computed directly from the inputs — the closed form is the real correctness gate.
// ---------------------------------------------------------------------------------------------

use wukong_interp::run_kernel_f32;

// Runtime op codes (mirrored from wukong_runtime).
const VM_TANH: i64 = 2;
const VM_SIGMOID: i64 = 3;
const VM_RELU: i64 = 4;
const VM_SILU: i64 = 5;
const VM_GELU: i64 = 6;
const RED_SUM: i64 = 2;
const RED_SSD: i64 = 1;
const VE_ID: i64 = 0;
const VE_USE_Y: i64 = 256;
// velem *compute modes* — they live above the low activation byte, so `op & 0xff` does not see them.
const VE_HADAMARD: i64 = 512;
const VE_DIV: i64 = 1024;

fn ci(b: &mut Builder, x: i64) -> ValueId {
    b.build(MirType::I64, Op::ConstInt(x as i128, MirType::I64))
}
fn cf(b: &mut Builder, x: f64) -> ValueId {
    b.build(MirType::F32, Op::ConstFloat(x, MirType::F32))
}
fn arr(n: usize) -> MirType {
    MirType::Array(Box::new(MirType::F32), n as u32)
}

/// Run an f32 tape and return the scalar loss (`out[0]`).
fn loss_at_f32(
    prog: &Program,
    name: Symbol,
    bufs: &mut [Vec<f32>],
    loss_out: usize,
    it: &Interner,
) -> f64 {
    let mut views: Vec<&mut [f32]> = bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
    run_kernel_f32(prog, name, &mut views, it).expect("forward f32 run failed");
    bufs[loss_out][0] as f64
}

/// Run the gradient tape, returning the gradient buffer (as f64) for each `wrt` input.
fn analytic_grad_f32(
    prog: &Program,
    gname: Symbol,
    fwd: &Fwd,
    inputs: &[Vec<f32>],
    wrt: &[usize],
    it: &Interner,
) -> Vec<Vec<f64>> {
    let mut bufs: Vec<Vec<f32>> = inputs.to_vec();
    for &wi in wrt {
        bufs.push(vec![0.0; fwd.lens[wi]]);
    }
    let mut views: Vec<&mut [f32]> = bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
    run_kernel_f32(prog, gname, &mut views, it).expect("gradient f32 run failed");
    let np = fwd.func.params.len();
    (0..wrt.len())
        .map(|i| bufs[np + i].iter().map(|&v| v as f64).collect())
        .collect()
}

fn fd_grad_f32(
    prog: &Program,
    fwd: &Fwd,
    inputs: &[Vec<f32>],
    wrt: &[usize],
    eps: f32,
    it: &Interner,
) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for &wi in wrt {
        let mut g = vec![0.0; fwd.lens[wi]];
        for j in 0..fwd.lens[wi] {
            let mut bufs = inputs.to_vec();
            let orig = bufs[wi][j];
            bufs[wi][j] = orig + eps;
            let lp = loss_at_f32(prog, fwd.func.name, &mut bufs, fwd.loss_out, it);
            bufs[wi][j] = orig - eps;
            let lm = loss_at_f32(prog, fwd.func.name, &mut bufs, fwd.loss_out, it);
            g[j] = (lp - lm) / (2.0 * eps as f64);
        }
        out.push(g);
    }
    out
}

/// Tensor-tape gate: analytic gradient vs (loose) f32 finite difference and vs (tight) f64 closed
/// form. Returns the analytic gradients.
fn tape_gate(
    fwd: &Fwd,
    wrt: &[usize],
    inputs: &[Vec<f32>],
    closed_form: &[Vec<f64>],
    it: &mut Interner,
) -> Vec<Vec<f64>> {
    let (prog, gname) = build(&fwd.func, wrt, it);
    let analytic = analytic_grad_f32(&prog, gname, fwd, inputs, wrt, it);
    let fd = fd_grad_f32(&prog, fwd, inputs, wrt, 5e-3, it);
    for (gi, (a, f)) in analytic.iter().zip(fd.iter()).enumerate() {
        for (j, (&av, &fv)) in a.iter().zip(f.iter()).enumerate() {
            let tol = 5e-3 + 3e-2 * fv.abs();
            assert!(
                (av - fv).abs() <= tol,
                "wrt#{gi}[{j}]: analytic {av} vs finite-diff {fv} (|d|={:.3e} > {:.3e})",
                (av - fv).abs(),
                tol
            );
        }
    }
    for (gi, (a, c)) in analytic.iter().zip(closed_form.iter()).enumerate() {
        for (j, (&av, &cv)) in a.iter().zip(c.iter()).enumerate() {
            let tol = 3e-3 + 3e-3 * cv.abs();
            assert!(
                (av - cv).abs() <= tol,
                "wrt#{gi}[{j}]: analytic {av} vs closed-form {cv} (|d|={:.3e} > {:.3e})",
                (av - cv).abs(),
                tol
            );
        }
    }
    analytic
}

// f64 reference matmul P[m,n] = sum_k X[m,k] * W[n,k]  (the nn.Linear C = X . W^T form).
fn matmul_nt_f64(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut p = vec![0.0; m * n];
    for mm in 0..m {
        for nn in 0..n {
            let mut s = 0.0;
            for kk in 0..k {
                s += x[mm * k + kk] as f64 * w[nn * k + kk] as f64;
            }
            p[mm * n + nn] = s;
        }
    }
    p
}

/// Given dP (the gradient of the M×N pre-reduction tensor), the matmul backward gradients:
/// dX[m,k] = sum_n dP[m,n] W[n,k], dW[n,k] = sum_m dP[m,n] X[m,k].
fn matmul_nt_backward(
    dp: &[f64],
    x: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut dx = vec![0.0; m * k];
    let mut dw = vec![0.0; n * k];
    for mm in 0..m {
        for nn in 0..n {
            let g = dp[mm * n + nn];
            for kk in 0..k {
                dx[mm * k + kk] += g * w[nn * k + kk] as f64;
                dw[nn * k + kk] += g * x[mm * k + kk] as f64;
            }
        }
    }
    (dx, dw)
}

fn rand_vec(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n).map(|_| lcg(seed) as f32).collect()
}

#[test]
fn linear_sum_vjp() {
    // loss = sum(X . W^T)  ->  dP = 1 everywhere.
    let (m, k, n) = (3, 4, 2);
    let mut it = Interner::default();
    let mut b = Builder::new(sym(&mut it, "lin_sum"), MirType::Void);
    let x = b.add_param(PTR);
    let w = b.add_param(PTR);
    let out = b.add_param(PTR);
    let p = b.alloca(arr(m * n));
    let (mv, kv, nv, beta) = (
        ci(&mut b, m as i64),
        ci(&mut b, k as i64),
        ci(&mut b, n as i64),
        ci(&mut b, 0),
    );
    let sgemm_nt = sym(&mut it, "wukong_sgemm_nt");
    b.build_void(Op::Call {
        func: sgemm_nt,
        args: vec![x, w, p, mv, kv, nv, beta],
    });
    let (mn, sumop) = (ci(&mut b, (m * n) as i64), ci(&mut b, RED_SUM));
    let sreduce = sym(&mut it, "wukong_sreduce_f32");
    let loss = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![p, p, mn, sumop],
        },
    );
    b.build_void(Op::Store { ptr: out, value: loss });
    b.ret(Some(loss));
    let fwd = Fwd {
        func: b.finish(),
        lens: vec![m * k, n * k, 1],
        loss_out: 2,
    };

    let mut seed = 0xABCDu64;
    let xb = rand_vec(&mut seed, m * k);
    let wb = rand_vec(&mut seed, n * k);
    let dp = vec![1.0; m * n];
    let (dx, dw) = matmul_nt_backward(&dp, &xb, &wb, m, k, n);
    let inputs = vec![xb, wb, vec![0.0]];
    tape_gate(&fwd, &[0, 1], &inputs, &[dx, dw], &mut it);
}

/// Build `loss = reduce(act(X . W^T))`: a sgemm_nt, an optional activation (vmath `act` op), then a
/// reduction — sum, or SSD against a target buffer `T` (= MSE loss). Params: X, W, [T if mse], out.
fn build_linear(it: &mut Interner, m: usize, k: usize, n: usize, act: Option<i64>, mse: bool) -> Fwd {
    let sgemm_nt = it.intern("wukong_sgemm_nt");
    let vmath = it.intern("wukong_vmath_f32");
    let sreduce = it.intern("wukong_sreduce_f32");
    let mut b = Builder::new(it.intern("lin"), MirType::Void);
    let x = b.add_param(PTR);
    let w = b.add_param(PTR);
    let t = if mse { Some(b.add_param(PTR)) } else { None };
    let out = b.add_param(PTR);

    let p = b.alloca(arr(m * n));
    let (mv, kv, nv, beta) = (
        ci(&mut b, m as i64),
        ci(&mut b, k as i64),
        ci(&mut b, n as i64),
        ci(&mut b, 0),
    );
    b.build_void(Op::Call {
        func: sgemm_nt,
        args: vec![x, w, p, mv, kv, nv, beta],
    });
    let mn = ci(&mut b, (m * n) as i64);

    let activated = if let Some(op) = act {
        let h = b.alloca(arr(m * n));
        let opv = ci(&mut b, op);
        b.build_void(Op::Call {
            func: vmath,
            args: vec![p, h, mn, opv],
        });
        h
    } else {
        p
    };

    let loss = if mse {
        let ssdop = ci(&mut b, RED_SSD);
        b.build(
            MirType::F32,
            Op::Call {
                func: sreduce,
                args: vec![activated, t.unwrap(), mn, ssdop],
            },
        )
    } else {
        let sumop = ci(&mut b, RED_SUM);
        b.build(
            MirType::F32,
            Op::Call {
                func: sreduce,
                args: vec![activated, activated, mn, sumop],
            },
        )
    };
    b.build_void(Op::Store { ptr: out, value: loss });
    b.ret(Some(loss));

    let (lens, loss_out) = if mse {
        (vec![m * k, n * k, m * n, 1], 3)
    } else {
        (vec![m * k, n * k, 1], 2)
    };
    Fwd {
        func: b.finish(),
        lens,
        loss_out,
    }
}

#[test]
fn linear_relu_sum_vjp() {
    // loss = sum(relu(X . W^T))  ->  dP[i] = (P[i] > 0) ? 1 : 0.
    let (m, k, n) = (3, 4, 2);
    let mut it = Interner::default();
    let fwd = build_linear(&mut it, m, k, n, Some(VM_RELU), false);
    // Keep every pre-activation P comfortably away from the relu kink at 0, so the +-eps finite
    // difference never flips a mask (which would make the central difference invalid). Mixed signs
    // still exercise both the active and the zeroed gradient paths.
    let mut seed = 0x5151u64;
    let (xb, wb, p) = loop {
        let xb = rand_vec(&mut seed, m * k);
        let wb = rand_vec(&mut seed, n * k);
        let p = matmul_nt_f64(&xb, &wb, m, k, n);
        if p.iter().all(|&v| v.abs() > 0.2) && p.iter().any(|&v| v > 0.0) && p.iter().any(|&v| v < 0.0)
        {
            break (xb, wb, p);
        }
    };
    let dp: Vec<f64> = p.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect();
    let (dx, dw) = matmul_nt_backward(&dp, &xb, &wb, m, k, n);
    let inputs = vec![xb, wb, vec![0.0]];
    tape_gate(&fwd, &[0, 1], &inputs, &[dx, dw], &mut it);
}

#[test]
fn linear_mse_vjp() {
    // loss = sum((X . W^T - T)^2)  ->  dP[i] = 2 (P[i] - T[i]).  (the regression-training loss)
    let (m, k, n) = (3, 4, 2);
    let mut it = Interner::default();
    let fwd = build_linear(&mut it, m, k, n, None, true);
    let mut seed = 0x9090u64;
    let xb = rand_vec(&mut seed, m * k);
    let wb = rand_vec(&mut seed, n * k);
    let tb = rand_vec(&mut seed, m * n);
    let p = matmul_nt_f64(&xb, &wb, m, k, n);
    let dp: Vec<f64> = p
        .iter()
        .zip(&tb)
        .map(|(&pv, &tv)| 2.0 * (pv - tv as f64))
        .collect();
    let (dx, dw) = matmul_nt_backward(&dp, &xb, &wb, m, k, n);
    let inputs = vec![xb, wb, tb, vec![0.0]];
    tape_gate(&fwd, &[0, 1], &inputs, &[dx, dw], &mut it);
}

#[test]
fn residual_two_linears_vjp() {
    // loss = sum(X . W1^T + X . W2^T): a velem residual add, and X feeds BOTH matmuls — so its
    // gradient must accumulate (matmul beta=1 on the second contribution).
    let (m, k, n) = (3, 4, 2);
    let mut it = Interner::default();
    let sgemm_nt = sym(&mut it, "wukong_sgemm_nt");
    let velem = sym(&mut it, "wukong_velem_f32");
    let sreduce = sym(&mut it, "wukong_sreduce_f32");
    let mut b = Builder::new(sym(&mut it, "resid"), MirType::Void);
    let x = b.add_param(PTR);
    let w1 = b.add_param(PTR);
    let w2 = b.add_param(PTR);
    let out = b.add_param(PTR);
    let p1 = b.alloca(arr(m * n));
    let p2 = b.alloca(arr(m * n));
    let z = b.alloca(arr(m * n));
    let (mv, kv, nv, beta) = (
        ci(&mut b, m as i64),
        ci(&mut b, k as i64),
        ci(&mut b, n as i64),
        ci(&mut b, 0),
    );
    b.build_void(Op::Call {
        func: sgemm_nt,
        args: vec![x, w1, p1, mv, kv, nv, beta],
    });
    b.build_void(Op::Call {
        func: sgemm_nt,
        args: vec![x, w2, p2, mv, kv, nv, beta],
    });
    // z = 1*p1 + 1*p2  (velem identity, reads y)
    let mn = ci(&mut b, (m * n) as i64);
    let (one_a, one_b, zero_c) = (cf(&mut b, 1.0), cf(&mut b, 1.0), cf(&mut b, 0.0));
    let veop = ci(&mut b, VE_ID | VE_USE_Y);
    b.build_void(Op::Call {
        func: velem,
        args: vec![p1, p2, z, mn, one_a, one_b, zero_c, veop],
    });
    let sumop = ci(&mut b, RED_SUM);
    let loss = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![z, z, mn, sumop],
        },
    );
    b.build_void(Op::Store { ptr: out, value: loss });
    b.ret(Some(loss));
    let fwd = Fwd {
        func: b.finish(),
        lens: vec![m * k, n * k, n * k, 1],
        loss_out: 3,
    };

    let mut seed = 0x7777u64;
    let xb = rand_vec(&mut seed, m * k);
    let w1b = rand_vec(&mut seed, n * k);
    let w2b = rand_vec(&mut seed, n * k);
    let dp = vec![1.0; m * n];
    let (dx1, dw1) = matmul_nt_backward(&dp, &xb, &w1b, m, k, n);
    let (dx2, dw2) = matmul_nt_backward(&dp, &xb, &w2b, m, k, n);
    let dx: Vec<f64> = dx1.iter().zip(&dx2).map(|(a, b)| a + b).collect();
    let inputs = vec![xb, w1b, w2b, vec![0.0]];
    tape_gate(&fwd, &[0, 1, 2], &inputs, &[dx, dw1, dw2], &mut it);
}

/// FD-only tape gate, for smooth activations whose kernel uses an f32 polynomial approximation (so a
/// tight f64 closed form is not available — the analytic VJP reuses the kernel's own f32 output).
/// The activation is smooth (no kink), so the finite difference is reliable.
fn tape_gate_fd(fwd: &Fwd, wrt: &[usize], inputs: &[Vec<f32>], it: &mut Interner) -> Vec<Vec<f64>> {
    let (prog, gname) = build(&fwd.func, wrt, it);
    let analytic = analytic_grad_f32(&prog, gname, fwd, inputs, wrt, it);
    let fd = fd_grad_f32(&prog, fwd, inputs, wrt, 5e-3, it);
    // Guard against a silently-zero gradient passing the FD check vacuously.
    assert!(
        analytic.iter().flatten().any(|&v| v.abs() > 1e-3),
        "gradient is trivially zero — likely a missing VJP route"
    );
    for (gi, (a, f)) in analytic.iter().zip(fd.iter()).enumerate() {
        for (j, (&av, &fv)) in a.iter().zip(f.iter()).enumerate() {
            let tol = 1e-2 + 4e-2 * fv.abs();
            assert!(
                (av - fv).abs() <= tol,
                "wrt#{gi}[{j}]: analytic {av} vs finite-diff {fv} (|d|={:.3e} > {:.3e})",
                (av - fv).abs(),
                tol
            );
        }
    }
    analytic
}

#[test]
fn linear_sigmoid_sum_vjp() {
    // loss = sum(sigmoid(X . W^T)); backward reuses the forward output: sigmoid' = y(1-y).
    let (m, k, n) = (3, 4, 2);
    let mut it = Interner::default();
    let fwd = build_linear(&mut it, m, k, n, Some(VM_SIGMOID), false);
    let mut seed = 0x3333u64;
    let xb = rand_vec(&mut seed, m * k);
    let wb = rand_vec(&mut seed, n * k);
    let inputs = vec![xb, wb, vec![0.0]];
    tape_gate_fd(&fwd, &[0, 1], &inputs, &mut it);
}

#[test]
fn linear_tanh_sum_vjp() {
    // loss = sum(tanh(X . W^T)); backward reuses the forward output: tanh' = 1 - y^2.
    let (m, k, n) = (3, 4, 2);
    let mut it = Interner::default();
    let fwd = build_linear(&mut it, m, k, n, Some(VM_TANH), false);
    let mut seed = 0x4444u64;
    let xb = rand_vec(&mut seed, m * k);
    let wb = rand_vec(&mut seed, n * k);
    let inputs = vec![xb, wb, vec![0.0]];
    tape_gate_fd(&fwd, &[0, 1], &inputs, &mut it);
}

#[test]
fn linear_silu_sum_vjp() {
    // loss = sum(silu(X . W^T)); the silu backward rides the fused wukong_vmath2_f32 (VM2_SILU_BWD),
    // dx = dy·silu'(x) in one pass, bit-identical with the forward silu (shared sigmoid polynomial).
    let (m, k, n) = (3, 4, 2);
    let mut it = Interner::default();
    let fwd = build_linear(&mut it, m, k, n, Some(VM_SILU), false);
    let mut seed = 0x5170u64;
    let xb = rand_vec(&mut seed, m * k);
    let wb = rand_vec(&mut seed, n * k);
    let inputs = vec![xb, wb, vec![0.0]];
    tape_gate_fd(&fwd, &[0, 1], &inputs, &mut it);
}

#[test]
fn linear_gelu_sum_vjp() {
    // loss = sum(gelu(X . W^T)); the gelu backward rides wukong_vmath2_f32 (VM2_GELU_BWD), matching
    // the forward gelu's tanh-approximation derivative exactly.
    let (m, k, n) = (3, 4, 2);
    let mut it = Interner::default();
    let fwd = build_linear(&mut it, m, k, n, Some(VM_GELU), false);
    let mut seed = 0x6E10u64;
    let xb = rand_vec(&mut seed, m * k);
    let wb = rand_vec(&mut seed, n * k);
    let inputs = vec![xb, wb, vec![0.0]];
    tape_gate_fd(&fwd, &[0, 1], &inputs, &mut it);
}

// ---------------------------------------------------------------------------------------------
// Softmax VJP: loss = sum_ij C[i,j] * softmax(X)[i,j] (a coefficient-weighted softmax). Backward
// per row is dx = y (.) (dy - sum_j dy_j y_j) with dy = C — emitted as nested loops with a per-row
// dot (sreduce). Gated by the f64 closed form.
// ---------------------------------------------------------------------------------------------

const NORM_SOFTMAX: i64 = 0;
const NORM_LAYERNORM: i64 = 1;
const NORM_RMSNORM: i64 = 2;
const RED_DOT: i64 = 0;

/// Build `y = norm(x); loss = sum_i C[i]*y[i]` — a coefficient-weighted norm (so dy = C is
/// non-trivial). Params: X, C, out; intermediate: y (alloca).
fn build_norm_dot(it: &mut Interner, rows: usize, cols: usize, op: i64, eps_bits: i64) -> Fwd {
    build_norm_dot_sym(it, rows, cols, op, eps_bits, "wukong_norm_f32")
}

/// As [`build_norm_dot`], but with the norm kernel symbol chosen by the caller — so the same tape
/// can be built against the serial `wukong_norm_f32` and its `@parallel` twin.
fn build_norm_dot_sym(
    it: &mut Interner,
    rows: usize,
    cols: usize,
    op: i64,
    eps_bits: i64,
    norm_sym: &str,
) -> Fwd {
    let norm = it.intern(norm_sym);
    let sreduce = it.intern("wukong_sreduce_f32");
    let mut b = Builder::new(it.intern("normdot"), MirType::Void);
    let x = b.add_param(PTR);
    let c = b.add_param(PTR);
    let out = b.add_param(PTR);
    let y = b.alloca(arr(rows * cols));
    let (rv, cv, epsv, smop) = (
        ci(&mut b, rows as i64),
        ci(&mut b, cols as i64),
        ci(&mut b, eps_bits),
        ci(&mut b, op),
    );
    // y = norm(x) per row
    b.build_void(Op::Call {
        func: norm,
        args: vec![x, y, rv, cv, epsv, smop],
    });
    // loss = sum_i C[i] * y[i]   (dot)
    let (rc, dotop) = (ci(&mut b, (rows * cols) as i64), ci(&mut b, RED_DOT));
    let loss = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![c, y, rc, dotop],
        },
    );
    b.build_void(Op::Store { ptr: out, value: loss });
    b.ret(Some(loss));
    Fwd {
        func: b.finish(),
        lens: vec![rows * cols, rows * cols, 1],
        loss_out: 2,
    }
}

fn softmax_f64(x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let mut y = vec![0.0; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = row.iter().map(|&v| (v as f64 - mx).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for (j, e) in exps.iter().enumerate() {
            y[r * cols + j] = e / sum;
        }
    }
    y
}

#[test]
fn softmax_dot_vjp() {
    let (rows, cols) = (3, 4);
    let mut it = Interner::default();
    let fwd = build_norm_dot(&mut it, rows, cols, NORM_SOFTMAX, 0);
    let mut seed = 0x50F7u64;
    let xb = rand_vec(&mut seed, rows * cols);
    let cb = rand_vec(&mut seed, rows * cols);
    // dy = C; dx[r,j] = y[r,j]*(C[r,j] - sum_k C[r,k] y[r,k]).
    let y = softmax_f64(&xb, rows, cols);
    let mut dx = vec![0.0; rows * cols];
    for r in 0..rows {
        let s: f64 = (0..cols)
            .map(|j| cb[r * cols + j] as f64 * y[r * cols + j])
            .sum();
        for j in 0..cols {
            let idx = r * cols + j;
            dx[idx] = y[idx] * (cb[idx] as f64 - s);
        }
    }
    let inputs = vec![xb, cb, vec![0.0]];
    tape_gate(&fwd, &[0], &inputs, &[dx], &mut it);
}

#[test]
fn layernorm_dot_vjp() {
    let (rows, cols) = (3, 5);
    let eps = 1e-5f32;
    let mut it = Interner::default();
    let fwd = build_norm_dot(&mut it, rows, cols, NORM_LAYERNORM, eps.to_bits() as i64);
    let mut seed = 0x1A4Eu64;
    let xb = rand_vec(&mut seed, rows * cols);
    let cb = rand_vec(&mut seed, rows * cols);
    // y = (x - mu)/sigma per row; dy = C; dx = (1/sigma)(dy - mean(dy) - y*mean(dy*y)).
    let mut dx = vec![0.0; rows * cols];
    for r in 0..rows {
        let row = &xb[r * cols..(r + 1) * cols];
        let nf = cols as f64;
        let mu = row.iter().map(|&v| v as f64).sum::<f64>() / nf;
        let var = row.iter().map(|&v| (v as f64 - mu).powi(2)).sum::<f64>() / nf;
        let sigma = (var + eps as f64).sqrt();
        let y: Vec<f64> = row.iter().map(|&v| (v as f64 - mu) / sigma).collect();
        let mean_dy = (0..cols).map(|j| cb[r * cols + j] as f64).sum::<f64>() / nf;
        let mean_dyy = (0..cols).map(|j| cb[r * cols + j] as f64 * y[j]).sum::<f64>() / nf;
        for j in 0..cols {
            dx[r * cols + j] =
                (1.0 / sigma) * (cb[r * cols + j] as f64 - mean_dy - y[j] * mean_dyy);
        }
    }
    let inputs = vec![xb, cb, vec![0.0]];
    tape_gate(&fwd, &[0], &inputs, &[dx], &mut it);
}

#[test]
fn rmsnorm_dot_vjp() {
    let (rows, cols) = (3, 5);
    let eps = 1e-5f32;
    let mut it = Interner::default();
    let fwd = build_norm_dot(&mut it, rows, cols, NORM_RMSNORM, eps.to_bits() as i64);
    let mut seed = 0x71A3u64;
    let xb = rand_vec(&mut seed, rows * cols);
    let cb = rand_vec(&mut seed, rows * cols);
    // y = x/r, r = sqrt(mean(x^2)+eps); dy = C; dx = (1/r)(dy - y*mean(dy*y)).
    let mut dx = vec![0.0; rows * cols];
    for r in 0..rows {
        let row = &xb[r * cols..(r + 1) * cols];
        let nf = cols as f64;
        let ms = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / nf;
        let rr = (ms + eps as f64).sqrt();
        let y: Vec<f64> = row.iter().map(|&v| v as f64 / rr).collect();
        let mean_dyy = (0..cols).map(|j| cb[r * cols + j] as f64 * y[j]).sum::<f64>() / nf;
        for j in 0..cols {
            dx[r * cols + j] = (1.0 / rr) * (cb[r * cols + j] as f64 - y[j] * mean_dyy);
        }
    }
    let inputs = vec![xb, cb, vec![0.0]];
    tape_gate(&fwd, &[0], &inputs, &[dx], &mut it);
}

/// The section-5 coupling contract: every `_parallel` recognizer arm in mir_build must have a
/// counterpart in `Syms`/`is_kernel`/`diff_kernel_call`. `wukong_norm_f32_parallel` — which
/// `emit_norm` selects for every batched norm inside a `@parallel fn`, i.e. the shipped transformer
/// shape — had none, so a `@parallel` RMSNorm loss declined with
/// "unrecognized buffer-writing call has no VJP rule" while the byte-identical serial spelling
/// differentiated fine (observed with the shipped release compiler). Each row is independent, so the
/// parallel kernel is bit-identical to the serial one and takes the very same rule; this gates that
/// it now produces the same closed-form gradient as `rmsnorm_dot_vjp`.
#[test]
fn parallel_norm_dot_vjp() {
    let (rows, cols) = (3, 5);
    let eps = 1e-5f32;
    let mut it = Interner::default();
    let fwd = build_norm_dot_sym(
        &mut it,
        rows,
        cols,
        NORM_RMSNORM,
        eps.to_bits() as i64,
        "wukong_norm_f32_parallel",
    );
    let mut seed = 0x71A3u64;
    let xb = rand_vec(&mut seed, rows * cols);
    let cb = rand_vec(&mut seed, rows * cols);
    let mut dx = vec![0.0; rows * cols];
    for r in 0..rows {
        let row = &xb[r * cols..(r + 1) * cols];
        let nf = cols as f64;
        let ms = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / nf;
        let rr = (ms + eps as f64).sqrt();
        let y: Vec<f64> = row.iter().map(|&v| v as f64 / rr).collect();
        let mean_dyy = (0..cols).map(|j| cb[r * cols + j] as f64 * y[j]).sum::<f64>() / nf;
        for j in 0..cols {
            dx[r * cols + j] = (1.0 / rr) * (cb[r * cols + j] as f64 - y[j] * mean_dyy);
        }
    }
    let inputs = vec![xb, cb, vec![0.0]];
    tape_gate(&fwd, &[0], &inputs, &[dx], &mut it);
}

/// A full **pre-norm transformer block** tape: `h = rmsnorm(x); p = h·Wᵀ; a = silu(p); loss = Σa`.
/// Its backward composes all three recognized-kernel backward families in one reverse pass — the sum
/// seed, the silu backward (`wukong_vmath2_f32`), the matmul adjoints (transpose + two GEMMs), and
/// the RMSNorm backward (per-row reductions + an elementwise combine) — the exact composition a
/// transformer layer differentiates through. Params: X, W, out; intermediates h, p, a (allocas).
fn build_prenorm_block(it: &mut Interner, rows: usize, cols: usize, n: usize, eps_bits: i64) -> Fwd {
    let norm = it.intern("wukong_norm_f32");
    let sgemm_nt = it.intern("wukong_sgemm_nt");
    let vmath = it.intern("wukong_vmath_f32");
    let sreduce = it.intern("wukong_sreduce_f32");
    let mut b = Builder::new(it.intern("prenorm_block"), MirType::Void);
    let x = b.add_param(PTR);
    let w = b.add_param(PTR);
    let out = b.add_param(PTR);
    let h = b.alloca(arr(rows * cols));
    let p = b.alloca(arr(rows * n));
    let a = b.alloca(arr(rows * n));

    // h = rmsnorm(x) per row (NOT in place — the backward recomputes the row statistics from x).
    let (rv, cv, epsv, rmsop) = (
        ci(&mut b, rows as i64),
        ci(&mut b, cols as i64),
        ci(&mut b, eps_bits),
        ci(&mut b, NORM_RMSNORM),
    );
    b.build_void(Op::Call {
        func: norm,
        args: vec![x, h, rv, cv, epsv, rmsop],
    });
    // p = h · Wᵀ   (M=rows, K=cols, N=n)
    let (mv, kv, nv, beta) = (
        ci(&mut b, rows as i64),
        ci(&mut b, cols as i64),
        ci(&mut b, n as i64),
        ci(&mut b, 0),
    );
    b.build_void(Op::Call {
        func: sgemm_nt,
        args: vec![h, w, p, mv, kv, nv, beta],
    });
    // a = silu(p)
    let (rn, siluop) = (ci(&mut b, (rows * n) as i64), ci(&mut b, VM_SILU));
    b.build_void(Op::Call {
        func: vmath,
        args: vec![p, a, rn, siluop],
    });
    // loss = Σ a
    let sumop = ci(&mut b, RED_SUM);
    let loss = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![a, a, rn, sumop],
        },
    );
    b.build_void(Op::Store { ptr: out, value: loss });
    b.ret(Some(loss));
    Fwd {
        func: b.finish(),
        lens: vec![rows * cols, n * cols, 1],
        loss_out: 2,
    }
}

#[test]
fn prenorm_block_vjp() {
    // The composite gradient of rmsnorm → linear → silu → sum, w.r.t. both the input x and the
    // weight W — finite-difference-gated (all ops smooth, so the central difference is valid).
    let (rows, cols, n) = (3, 4, 2);
    let eps = 1e-5f32;
    let mut it = Interner::default();
    let fwd = build_prenorm_block(&mut it, rows, cols, n, eps.to_bits() as i64);
    let mut seed = 0xB10Cu64;
    let xb = rand_vec(&mut seed, rows * cols);
    let wb = rand_vec(&mut seed, n * cols);
    let inputs = vec![xb, wb, vec![0.0]];
    tape_gate_fd(&fwd, &[0, 1], &inputs, &mut it);
}

// ---------------------------------------------------------------------------------------------
// A real two-layer kernel MLP: P1 = X.W1^T; H = relu(P1); P2 = H.W2^T; loss = sum((P2 - T)^2).
// Chains two sgemm_nt matmuls through a relu and an MSE loss — the full backward path the engine
// emits (two transposes, a masked relu loop, two NN GEMMs, the SSD affine). Gated by the f64
// closed form, then trained with SGD to confirm the loss decreases.
// ---------------------------------------------------------------------------------------------

const B: usize = 2;
const MIN: usize = 3; // input features
const MHID: usize = 4;
const MOUT: usize = 2;

fn build_mlp2(it: &mut Interner) -> Fwd {
    let sgemm_nt = it.intern("wukong_sgemm_nt");
    let vmath = it.intern("wukong_vmath_f32");
    let sreduce = it.intern("wukong_sreduce_f32");
    let mut b = Builder::new(it.intern("mlp2"), MirType::Void);
    let x = b.add_param(PTR); // B x MIN
    let w1 = b.add_param(PTR); // MHID x MIN
    let w2 = b.add_param(PTR); // MOUT x MHID
    let t = b.add_param(PTR); // B x MOUT
    let out = b.add_param(PTR);

    let p1 = b.alloca(arr(B * MHID));
    let h = b.alloca(arr(B * MHID));
    let p2 = b.alloca(arr(B * MOUT));
    let zero = ci(&mut b, 0);

    // P1 = X . W1^T   (m=B, k=MIN, n=MHID)
    let (bv, inv, hidv) = (
        ci(&mut b, B as i64),
        ci(&mut b, MIN as i64),
        ci(&mut b, MHID as i64),
    );
    b.build_void(Op::Call {
        func: sgemm_nt,
        args: vec![x, w1, p1, bv, inv, hidv, zero],
    });
    // H = relu(P1)
    let bhid = ci(&mut b, (B * MHID) as i64);
    let reluop = ci(&mut b, VM_RELU);
    b.build_void(Op::Call {
        func: vmath,
        args: vec![p1, h, bhid, reluop],
    });
    // P2 = H . W2^T   (m=B, k=MHID, n=MOUT)
    let outv = ci(&mut b, MOUT as i64);
    b.build_void(Op::Call {
        func: sgemm_nt,
        args: vec![h, w2, p2, bv, hidv, outv, zero],
    });
    // loss = sum((P2 - T)^2)
    let bout = ci(&mut b, (B * MOUT) as i64);
    let ssdop = ci(&mut b, RED_SSD);
    let loss = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![p2, t, bout, ssdop],
        },
    );
    b.build_void(Op::Store { ptr: out, value: loss });
    b.ret(Some(loss));
    Fwd {
        func: b.finish(),
        lens: vec![B * MIN, MHID * MIN, MOUT * MHID, B * MOUT, 1],
        loss_out: 4,
    }
}

/// f64 reference forward + backward for the two-layer MLP. Returns (loss, dW1, dW2).
fn mlp2_reference(x: &[f32], w1: &[f32], w2: &[f32], t: &[f32]) -> (f64, Vec<f64>, Vec<f64>) {
    let p1 = matmul_nt_f64(x, w1, B, MIN, MHID); // B x MHID
    let h: Vec<f32> = p1.iter().map(|&v| v.max(0.0) as f32).collect();
    let p2 = matmul_nt_f64(&h, w2, B, MHID, MOUT); // B x MOUT
    let loss: f64 = p2
        .iter()
        .zip(t)
        .map(|(&pv, &tv)| (pv - tv as f64).powi(2))
        .sum();
    let dp2: Vec<f64> = p2
        .iter()
        .zip(t)
        .map(|(&pv, &tv)| 2.0 * (pv - tv as f64))
        .collect();
    let (dh, dw2) = matmul_nt_backward(&dp2, &h, w2, B, MHID, MOUT);
    let dp1: Vec<f64> = dh
        .iter()
        .zip(&p1)
        .map(|(&g, &pv)| if pv > 0.0 { g } else { 0.0 })
        .collect();
    let (_dx, dw1) = matmul_nt_backward(&dp1, x, w1, B, MIN, MHID);
    (loss, dw1, dw2)
}

#[test]
fn mlp2_kernel_gradient() {
    let mut it = Interner::default();
    let fwd = build_mlp2(&mut it);
    // Data with the hidden pre-activations away from the relu kink (valid finite difference).
    let mut seed = 0xBEEFu64;
    let (xb, w1b, w2b, tb) = loop {
        let xb = rand_vec(&mut seed, B * MIN);
        let w1b = rand_vec(&mut seed, MHID * MIN);
        let w2b = rand_vec(&mut seed, MOUT * MHID);
        let tb = rand_vec(&mut seed, B * MOUT);
        let p1 = matmul_nt_f64(&xb, &w1b, B, MIN, MHID);
        if p1.iter().all(|&v| v.abs() > 0.2)
            && p1.iter().any(|&v| v > 0.0)
            && p1.iter().any(|&v| v < 0.0)
        {
            break (xb, w1b, w2b, tb);
        }
    };
    let (_loss, dw1, dw2) = mlp2_reference(&xb, &w1b, &w2b, &tb);
    let inputs = vec![xb, w1b, w2b, tb, vec![0.0]];
    tape_gate(&fwd, &[1, 2], &inputs, &[dw1, dw2], &mut it);
}

#[test]
fn mlp2_kernel_sgd_decreases_loss() {
    let mut it = Interner::default();
    let fwd = build_mlp2(&mut it);
    let (prog, gname) = build(&fwd.func, &[1, 2], &mut it);

    // A reachable target: the output of a "teacher" net on this input. The student (different init)
    // can in principle fit it, so SGD should drive the loss down substantially.
    let mut seed = 0xD00Du64;
    let xb = rand_vec(&mut seed, B * MIN);
    let teach_w1 = rand_vec(&mut seed, MHID * MIN);
    let teach_w2 = rand_vec(&mut seed, MOUT * MHID);
    let tb: Vec<f32> = {
        let p1 = matmul_nt_f64(&xb, &teach_w1, B, MIN, MHID);
        let h: Vec<f32> = p1.iter().map(|&v| v.max(0.0) as f32).collect();
        matmul_nt_f64(&h, &teach_w2, B, MHID, MOUT)
            .iter()
            .map(|&v| v as f32)
            .collect()
    };
    let mut w1 = rand_vec(&mut seed, MHID * MIN);
    let mut w2 = rand_vec(&mut seed, MOUT * MHID);

    let loss_now = |w1: &[f32], w2: &[f32], it: &Interner| -> f64 {
        let mut bufs = vec![xb.clone(), w1.to_vec(), w2.to_vec(), tb.clone(), vec![0.0]];
        loss_at_f32(&prog, fwd.func.name, &mut bufs, 4, it)
    };

    let lr = 0.05f32;
    let l0 = loss_now(&w1, &w2, &it);
    let mut prev = l0;
    let mut last = l0;
    for step in 0..1000 {
        let inputs = vec![xb.clone(), w1.clone(), w2.clone(), tb.clone(), vec![0.0]];
        let g = analytic_grad_f32(&prog, gname, &fwd, &inputs, &[1, 2], &it);
        for (wi, gi) in w1.iter_mut().zip(&g[0]) {
            *wi -= lr * *gi as f32;
        }
        for (wi, gi) in w2.iter_mut().zip(&g[1]) {
            *wi -= lr * *gi as f32;
        }
        last = loss_now(&w1, &w2, &it);
        // Full-batch GD with a small step descends monotonically (allow f32 rounding slack).
        assert!(
            last <= prev + 1e-5,
            "loss rose at step {step}: {prev} -> {last}"
        );
        prev = last;
    }
    // The autodiff gradients drive real learning toward the teacher's target.
    assert!(
        last < 0.1 * l0,
        "kernel MLP SGD did not converge: loss {l0} -> {last}"
    );
}

// ---------------------------------------------------------------------------------------------
// AdamW optimizer kernel (optim::build_adamw_step): gate the fused MIR update against an f64
// reference, then train the two-layer MLP with it (loss decreases).
// ---------------------------------------------------------------------------------------------

use crate::optim::{build_adamw_step, hp};

/// f64 reference AdamW update (in place on `w`, `m`, `v`).
#[allow(clippy::too_many_arguments)]
fn adamw_ref_f64(
    w: &mut [f64],
    g: &[f64],
    m: &mut [f64],
    v: &mut [f64],
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
    bc1: f64,
    bc2: f64,
) {
    for i in 0..w.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
        let mhat = m[i] / bc1;
        let vhat = v[i] / bc2;
        w[i] -= lr * (mhat / (vhat.sqrt() + eps) + wd * w[i]);
    }
}

#[test]
fn adamw_step_matches_reference() {
    let n = 16;
    let mut it = Interner::default();
    let f = build_adamw_step(&mut it, n);
    let name = f.name;
    let prog = Program {
        funcs: vec![f],
        statics: Vec::new(),
        level: wukong_mir::MirLevel::Low,
    };

    let mut seed = 0x4D11u64;
    let mut w = rand_vec(&mut seed, n);
    let g = rand_vec(&mut seed, n);
    let mut m = vec![0.0f32; n];
    let mut v = vec![0.0f32; n];

    let (lr, beta1, beta2, eps, wd) = (0.01f64, 0.9f64, 0.999f64, 1e-8f64, 0.01f64);
    let mut wr: Vec<f64> = w.iter().map(|&x| x as f64).collect();
    let gr: Vec<f64> = g.iter().map(|&x| x as f64).collect();
    let mut mr = vec![0.0f64; n];
    let mut vr = vec![0.0f64; n];

    // Several steps with a fixed gradient exercise the moment accumulation + bias correction.
    for t in 1..=5i32 {
        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);
        let mut hpbuf = vec![0.0f32; hp::LEN];
        hpbuf[hp::LR] = lr as f32;
        hpbuf[hp::BETA1] = beta1 as f32;
        hpbuf[hp::BETA2] = beta2 as f32;
        hpbuf[hp::EPS] = eps as f32;
        hpbuf[hp::WD] = wd as f32;
        hpbuf[hp::BC1] = bc1 as f32;
        hpbuf[hp::BC2] = bc2 as f32;

        let mut bufs = vec![w.clone(), g.clone(), m.clone(), v.clone(), hpbuf];
        let mut views: Vec<&mut [f32]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
        run_kernel_f32(&prog, name, &mut views, &it).expect("adamw run failed");
        w = bufs[0].clone();
        m = bufs[2].clone();
        v = bufs[3].clone();

        adamw_ref_f64(&mut wr, &gr, &mut mr, &mut vr, lr, beta1, beta2, eps, wd, bc1, bc2);

        for (j, (&wk, &wref)) in w.iter().zip(&wr).enumerate() {
            assert!(
                (wk as f64 - wref).abs() <= 1e-4 + 1e-4 * wref.abs(),
                "step {t} w[{j}]: kernel {wk} vs ref {wref}"
            );
        }
    }
}

#[test]
fn mlp2_adamw_decreases_loss() {
    let mut it = Interner::default();
    let fwd = build_mlp2(&mut it);
    let (gprog, gname) = build(&fwd.func, &[1, 2], &mut it);
    let n1 = MHID * MIN;
    let n2 = MOUT * MHID;
    // One AdamW kernel per parameter-tensor size (here both layers differ).
    let adamw1 = build_adamw_step(&mut it, n1);
    let a1name = adamw1.name;
    let adamw2 = build_adamw_step(&mut it, n2);
    let a2name = adamw2.name;
    let aprog = Program {
        funcs: vec![adamw1, adamw2],
        statics: Vec::new(),
        level: wukong_mir::MirLevel::Low,
    };

    let mut seed = 0xADA3u64;
    let xb = rand_vec(&mut seed, B * MIN);
    // A teacher with larger weights makes a non-trivial target; the student starts near zero, so the
    // initial loss is substantial (real optimization work, not a lucky near-optimal init).
    let teach_w1: Vec<f32> = rand_vec(&mut seed, n1).iter().map(|v| v * 2.0).collect();
    let teach_w2: Vec<f32> = rand_vec(&mut seed, n2).iter().map(|v| v * 2.0).collect();
    let tb: Vec<f32> = {
        let p1 = matmul_nt_f64(&xb, &teach_w1, B, MIN, MHID);
        let h: Vec<f32> = p1.iter().map(|&v| v.max(0.0) as f32).collect();
        matmul_nt_f64(&h, &teach_w2, B, MHID, MOUT)
            .iter()
            .map(|&v| v as f32)
            .collect()
    };
    let mut w1: Vec<f32> = rand_vec(&mut seed, n1).iter().map(|v| v * 0.1).collect();
    let mut w2: Vec<f32> = rand_vec(&mut seed, n2).iter().map(|v| v * 0.1).collect();
    let mut m1 = vec![0.0f32; n1];
    let mut v1 = vec![0.0f32; n1];
    let mut m2 = vec![0.0f32; n2];
    let mut v2 = vec![0.0f32; n2];

    let loss_now = |w1: &[f32], w2: &[f32], it: &Interner| -> f64 {
        let mut bufs = vec![xb.clone(), w1.to_vec(), w2.to_vec(), tb.clone(), vec![0.0]];
        loss_at_f32(&gprog, fwd.func.name, &mut bufs, 4, it)
    };
    let run_adamw =
        |name: Symbol, w: &mut Vec<f32>, g: &[f32], m: &mut Vec<f32>, v: &mut Vec<f32>, hpbuf: &[f32], it: &Interner| {
            let mut bufs = vec![w.clone(), g.to_vec(), m.clone(), v.clone(), hpbuf.to_vec()];
            let mut views: Vec<&mut [f32]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
            run_kernel_f32(&aprog, name, &mut views, it).expect("adamw run");
            *w = bufs[0].clone();
            *m = bufs[2].clone();
            *v = bufs[3].clone();
        };

    let (lr, beta1, beta2, eps, wd) = (0.01f64, 0.9f64, 0.999f64, 1e-8f64, 0.0f64);
    let l0 = loss_now(&w1, &w2, &it);
    assert!(l0 > 0.1, "test setup: initial loss should be substantial, got {l0}");
    let mut last = l0;
    for t in 1..=300i32 {
        let inputs = vec![xb.clone(), w1.clone(), w2.clone(), tb.clone(), vec![0.0]];
        let g = analytic_grad_f32(&gprog, gname, &fwd, &inputs, &[1, 2], &it);
        let g1: Vec<f32> = g[0].iter().map(|&x| x as f32).collect();
        let g2: Vec<f32> = g[1].iter().map(|&x| x as f32).collect();
        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);
        let mut hpbuf = vec![0.0f32; hp::LEN];
        hpbuf[hp::LR] = lr as f32;
        hpbuf[hp::BETA1] = beta1 as f32;
        hpbuf[hp::BETA2] = beta2 as f32;
        hpbuf[hp::EPS] = eps as f32;
        hpbuf[hp::WD] = wd as f32;
        hpbuf[hp::BC1] = bc1 as f32;
        hpbuf[hp::BC2] = bc2 as f32;
        run_adamw(a1name, &mut w1, &g1, &mut m1, &mut v1, &hpbuf, &it);
        run_adamw(a2name, &mut w2, &g2, &mut m2, &mut v2, &hpbuf, &it);
        last = loss_now(&w1, &w2, &it);
    }
    // AdamW's normalized step oscillates near the optimum (it won't reach ~0 with a fixed lr), but a
    // large, stable reduction shows the fused kernel optimizes correctly with the autodiff gradients.
    assert!(
        last.is_finite() && last < 0.4 * l0,
        "AdamW did not reduce the loss enough: {l0} -> {last}"
    );
}

/// The tape.rs coupling contract for mid-function `@parallel` loop regions (mir_build's
/// `try_emit_parallel_region`): a forward function containing a region — an `Op::FuncAddr` of the
/// outlined body plus a void `wukong_parallel_for` call — must DECLINE loudly. The runtime
/// dispatch writes buffers no VJP rule models, so `grad` must surface the existing
/// unrecognized-buffer-writing-call error instead of silently emitting a zero gradient. A function
/// whose backward is wanted must keep its loops in the serial spelling (exactly the behavior the
/// whole-function `@parallel` wrapper already has).
#[test]
fn parallel_region_declines_loudly() {
    let mut it = Interner::default();
    let pfor = sym(&mut it, "wukong_parallel_for");
    let region_body = sym(&mut it, "wukong$par$0");
    let mut b = Builder::new(sym(&mut it, "region_fwd"), MirType::Void);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    // The wrapper shape mir_build emits: env pointer-table pack + parallel_for call.
    let env = b.alloca(MirType::Array(Box::new(PTR), 2));
    let i0 = ci(&mut b, 0);
    let s0 = b.build(
        PTR,
        Op::Gep {
            ptr: env,
            index: i0,
            elem: PTR,
        },
    );
    b.build_void(Op::Store { ptr: s0, value: x });
    let i1 = ci(&mut b, 1);
    let s1 = b.build(
        PTR,
        Op::Gep {
            ptr: env,
            index: i1,
            elem: PTR,
        },
    );
    b.build_void(Op::Store { ptr: s1, value: out });
    let n = ci(&mut b, 4);
    let addr = b.build(PTR, Op::FuncAddr(region_body));
    b.build_void(Op::Call {
        func: pfor,
        args: vec![n, addr, env],
    });
    // The scalar loss a real forward would return (read back from the region output buffer) —
    // `grad` requires a `ret <loss>` tail before it walks the tape in reverse.
    let loss = b.build(F64, Op::Load(out, F64));
    b.ret(Some(loss));
    let fwd = b.finish();

    let err = grad(&fwd, &[0], &mut it).expect_err("a parallel region must not tape");
    assert!(
        err.contains("no VJP rule"),
        "the decline must be the loud unrecognized-call error, got: {err}"
    );
}

/// The scalar path (`diff_load`) ACCUMULATES into the gradient buffer (read-add-write) while the
/// kernel path (`fill_buf` / `velem_scale` / `velem_affine`) OVERWRITES it. A loss that both reduces
/// a `wrt` buffer with a recognized kernel AND reads one of its elements as a scalar therefore lost
/// the scalar contribution: observed with the shipped release compiler on
///
/// ```wukong
/// @parallel fn loss(x:[f32;8], mut out:[f32;1]) -> f32 {
///   let mut l: f32 = 0.0; for i in 0..8 { l = l + x[i]; }
///   let e: f32 = x[0]; let r: f32 = l + e; out[0] = r; return r; }
/// ```
///
/// which exited 0 emitting `grad[0] += 1` and then a broadcast `velem` fill of 1.0 over all 8
/// elements that clobbered it — gradient [1,1,...] where the truth is [2,1,1,...]. `diff_load` now
/// records the contribution, so `single()` sees the mix and declines instead of answering wrongly.
#[test]
fn scalar_load_plus_kernel_reduction_declines_loudly() {
    let n = 8;
    let mut it = Interner::default();
    let sreduce = sym(&mut it, "wukong_sreduce_f32");
    let mut b = Builder::new(sym(&mut it, "mixfwd"), MirType::Void);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    // l = Σ x   (kernel path: an overwriting broadcast fill in the backward)
    let (nv, sumop) = (ci(&mut b, n as i64), ci(&mut b, RED_SUM));
    let l = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![x, x, nv, sumop],
        },
    );
    // e = x[0]  (scalar path: an accumulating read-add-write in the backward)
    let e = b.build(MirType::F32, Op::Load(x, MirType::F32));
    let r = b.build(MirType::F32, Op::Bin(BinOp::FAdd, l, e));
    b.build_void(Op::Store { ptr: out, value: r });
    b.ret(Some(r));
    let fwd = b.finish();

    let err = grad(&fwd, &[0], &mut it)
        .expect_err("a mixed scalar+kernel contribution must not be silently clobbered");
    assert!(
        err.contains("multiple gradient contributions"),
        "the decline must be the accumulation error, got: {err}"
    );
}

/// `is_kernel` recognizes a call by SYMBOL NAME alone, but every VJP rule indexes the forward
/// argument vector raw (`args[0]`..`args[7]`). A `.wk` program may declare a runtime kernel name in
/// an `extern "C"` block with any signature, so a one-argument `wukong_norm_f32` used to reach
/// `diff_norm` and panic the compiler:
/// `index out of bounds: the len is 1 but the index is 1` at tape.rs (exit 127, observed with the
/// shipped release compiler on an `extern "C" { fn wukong_norm_f32(p: *mut f32); }` program).
/// The ABI arity is now checked up front, so the same input gets a diagnostic.
#[test]
fn kernel_call_with_wrong_arity_declines_loudly() {
    for (name, nargs, want) in [
        ("wukong_norm_f32", 1usize, 6usize),
        ("wukong_velem_f32", 3, 8),
        ("wukong_sgemm_nt", 2, 7),
        ("wukong_vmath_f32", 9, 4),
    ] {
        let mut it = Interner::default();
        let kern = sym(&mut it, name);
        let mut b = Builder::new(sym(&mut it, "extfwd"), MirType::Void);
        let x = b.add_param(PTR);
        let out = b.add_param(PTR);
        b.build_void(Op::Call {
            func: kern,
            args: vec![out; nargs],
        });
        let x0 = b.build(MirType::F32, Op::Load(x, MirType::F32));
        let sq = b.build(MirType::F32, Op::Bin(BinOp::FMul, x0, x0));
        b.build_void(Op::Store { ptr: out, value: sq });
        b.ret(Some(sq));
        let fwd = b.finish();

        let err = grad(&fwd, &[0], &mut it)
            .expect_err("a non-ABI arity must be a diagnostic, not an index panic");
        assert!(
            err.contains(name) && err.contains(&format!("expected {want}")),
            "{name}/{nargs} must name the callee and the expected arity, got: {err}"
        );
    }
}

/// A scalar element write into a buffer that a downstream recognized kernel then REDUCES is on the
/// gradient path, but the reverse walk has no VJP rule for read-after-write through memory. It used
/// to be skipped as "the loss sink", which made the whole reverse pass contribute nothing: observed
/// with the shipped release compiler on
///
/// ```wukong
/// @parallel fn loss(x:[f32;8], mut out:[f32;1]) -> f32 {
///   let mut h: [f32; 8] = [0.0; 8];
///   h[0] = x[0] * x[0];
///   let mut l: f32 = 0.0; for i in 0..8 { l = l + h[i]; }
///   out[0] = l; return l; }
/// ```
///
/// `--emit=grad` exited 0 with no diagnostic and never stored to the appended gradient parameter at
/// all — an all-zero gradient for every x, where the truth is `[2*x[0], 0, ...]`. It must refuse.
#[test]
fn store_into_live_gradient_buffer_declines_loudly() {
    let n = 8;
    let mut it = Interner::default();
    let sreduce = sym(&mut it, "wukong_sreduce_f32");
    let mut b = Builder::new(sym(&mut it, "storefwd"), MirType::Void);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    let h = b.alloca(arr(n));
    // h[0] = x[0] * x[0]   (a scalar write into the buffer the reduction below reads)
    let x0 = b.build(MirType::F32, Op::Load(x, MirType::F32));
    let sq = b.build(MirType::F32, Op::Bin(BinOp::FMul, x0, x0));
    b.build_void(Op::Store { ptr: h, value: sq });
    // loss = Σ h
    let (nv, sumop) = (ci(&mut b, n as i64), ci(&mut b, RED_SUM));
    let loss = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![h, h, nv, sumop],
        },
    );
    b.build_void(Op::Store { ptr: out, value: loss });
    b.ret(Some(loss));
    let fwd = b.finish();

    let err = grad(&fwd, &[0], &mut it)
        .expect_err("a store feeding a reduced buffer must not silently zero the gradient");
    assert!(
        err.contains("store into buffer") && err.contains("no VJP rule"),
        "the decline must be the loud store error, got: {err}"
    );
}

/// The autovectorizer's `Op::VecKernelCall` is void and writes through buffer pointers, exactly like
/// an unrecognized `Op::Call` — but the loud guard only matched `Op::Call`, so it fell through to
/// the `result.is_none() -> Ok(())` skip and contributed no adjoint at all. (Not reachable from
/// `.wk` source at HEAD: `emit_veckernel_for` always leaves a scalar tail loop, so such a function
/// has >1 block and `grad` rejects it earlier. This pins the crate's own contract so a future
/// CFG-simplification that folds an empty tail cannot turn it into a silent zero gradient.)
#[test]
fn veckernel_declines_loudly() {
    let mut it = Interner::default();
    let mut b = Builder::new(sym(&mut it, "veckernel_fwd"), MirType::Void);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    let ptrs = b.alloca(MirType::Array(Box::new(PTR), 2));
    let i0 = ci(&mut b, 0);
    let s0 = b.build(PTR, Op::Gep { ptr: ptrs, index: i0, elem: PTR });
    b.build_void(Op::Store { ptr: s0, value: x });
    let i1 = ci(&mut b, 1);
    let s1 = b.build(PTR, Op::Gep { ptr: ptrs, index: i1, elem: PTR });
    b.build_void(Op::Store { ptr: s1, value: out });
    let scalars = b.alloca(MirType::Array(Box::new(MirType::F32), 1));
    let n = ci(&mut b, 8);
    b.build_void(Op::VecKernelCall {
        kernel: 0,
        ptrs,
        scalars,
        n,
    });
    let loss = b.build(F64, Op::Load(out, F64));
    b.ret(Some(loss));
    let fwd = b.finish();

    let err = grad(&fwd, &[0], &mut it).expect_err("a veckernel must not tape");
    assert!(
        err.contains("no VJP rule") && err.contains("vector kernel"),
        "the decline must name the vector kernel, got: {err}"
    );
}

/// Build `t = velem(x, y, n, op); loss = Σ t` — the tape mir_build's `match_velem_binary` emits for
/// `t[i] = x[i] * y[i]` (op = `VE_HADAMARD|VE_USE_Y`) or `x[i] / y[i]` (op = `VE_DIV|VE_USE_Y`).
/// Params: X, Y, out; intermediate: t (alloca).
fn build_velem_sum(it: &mut Interner, n: usize, op: i64) -> Fwd {
    let velem = it.intern("wukong_velem_f32");
    let sreduce = it.intern("wukong_sreduce_f32");
    let mut b = Builder::new(it.intern("velem_sum"), MirType::Void);
    let x = b.add_param(PTR);
    let y = b.add_param(PTR);
    let out = b.add_param(PTR);
    let t = b.alloca(arr(n));
    let nv = ci(&mut b, n as i64);
    let (a_c, b_c, c_c) = (cf(&mut b, 1.0), cf(&mut b, 1.0), cf(&mut b, 0.0));
    let opv = ci(&mut b, op);
    b.build_void(Op::Call {
        func: velem,
        args: vec![x, y, t, nv, a_c, b_c, c_c, opv],
    });
    let sumop = ci(&mut b, RED_SUM);
    let loss = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![t, t, nv, sumop],
        },
    );
    b.build_void(Op::Store { ptr: out, value: loss });
    b.ret(Some(loss));
    Fwd {
        func: b.finish(),
        lens: vec![n, n, 1],
        loss_out: 2,
    }
}

/// A velem *compute mode* (`VE_HADAMARD` = 512, `VE_DIV` = 1024) lives ABOVE the activation byte, so
/// the old `op & 0xff != VE_ID` gate let it through and applied the AFFINE VJP `dx = a·dout` to a
/// nonlinear kernel: for `loss = Σ x[i]·y[i]` the emitted gradient was 1.0 everywhere instead of
/// `y[i]`, with no diagnostic. The gate is now a whitelist of the two affine spellings, so both
/// modes decline loudly. (The real Hadamard/quotient VJP can be added later; the refusal is what
/// stops the wrong answer.)
#[test]
fn velem_compute_modes_decline_loudly() {
    for (op, needle) in [
        (VE_HADAMARD | VE_USE_Y, "Hadamard"),
        (VE_DIV | VE_USE_Y, "division"),
    ] {
        let mut it = Interner::default();
        let fwd = build_velem_sum(&mut it, 8, op);
        let err = grad(&fwd.func, &[0, 1], &mut it)
            .expect_err("a non-affine velem compute mode must not take the affine VJP");
        assert!(
            err.contains("velem VJP") && err.contains(needle),
            "op {op} must decline naming the compute mode, got: {err}"
        );
    }
}

/// The two spellings the affine rule IS valid for must keep differentiating: the whitelist gate must
/// not have narrowed the accepted set. `loss = Σ (a·x + b·y)` -> `dx = a`, `dy = b` per element.
#[test]
fn velem_affine_still_differentiates() {
    let n = 8;
    let mut it = Interner::default();
    let fwd = build_velem_sum(&mut it, n, VE_ID | VE_USE_Y);
    let mut seed = 0xBE1Eu64;
    let xb = rand_vec(&mut seed, n);
    let yb = rand_vec(&mut seed, n);
    let inputs = vec![xb, yb, vec![0.0]];
    tape_gate(&fwd, &[0, 1], &inputs, &[vec![1.0; n], vec![1.0; n]], &mut it);
}
