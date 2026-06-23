//! Finite-difference gradient gate.
//!
//! Each test builds a forward MIR function (in `f64`, so the central-difference reference stays far
//! below tolerance), runs it through [`grad`], and asserts the analytic gradient the transform emits
//! matches the central finite difference over the whole gradient buffer — and, where a closed form
//! exists, matches that exactly. Both passes run on the `mercury_interp` oracle.

use crate::grad;
use mercury_interp::run_kernel_f64;
use mercury_mir::{BinOp, Builder, Function, MirType, Op, Program, ValueId};
use mercury_span::{Interner, Symbol};

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
        level: mercury_mir::MirLevel::Low,
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
        let pos = b.build(MirType::I1, Op::Cmp(mercury_mir::CmpOp::Fogt, xi, zero));
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
        let pos = b.build(MirType::I1, Op::Cmp(mercury_mir::CmpOp::Fogt, acc, zero));
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
