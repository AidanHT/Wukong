//! P4.6 (differentiating across user function calls) and P4.15 (second order): `inline_calls`
//! splices straight-line helpers into the loss, and `grad_dot` / `hvp` build the directional
//! derivative and the Hessian-vector product of a scalar-path loss.

use super::*;
use crate::{grad, grad_dot, hvp, inline_calls};
use wukong_mir::{BinOp, Builder, Function, MirType, Op, Program};
use wukong_span::Interner;

fn program_of(funcs: Vec<Function>) -> Program {
    Program {
        funcs,
        statics: Vec::new(),
        externs: Vec::new(),
        level: wukong_mir::MirLevel::Low,
    }
}

fn has_call(f: &Function) -> bool {
    f.blocks
        .iter()
        .flat_map(|bb| &bb.insts)
        .any(|inst| matches!(inst.op, Op::Call { .. }))
}

/// `loss(x) = triple(sqsum(x))` with `sqsum(p) = p[0]² + p[1]²` and `triple(s) = 3s`, so
/// `dL/dx = 6x`. Without flattening the call is a refusal that names the callee; with it the helpers
/// are spliced (no call remains) and the gradient is exact.
#[test]
fn helper_calls_are_inlined_for_differentiation() {
    let mut it = Interner::default();
    // sqsum(p: ptr) -> f64
    let mut b = Builder::new(sym(&mut it, "sqsum"), F64);
    let p = b.add_param(PTR);
    let a0 = load_elem(&mut b, p, 0);
    let a1 = load_elem(&mut b, p, 1);
    let s0 = b.build(F64, Op::Bin(BinOp::FMul, a0, a0));
    let s1 = b.build(F64, Op::Bin(BinOp::FMul, a1, a1));
    let s = b.build(F64, Op::Bin(BinOp::FAdd, s0, s1));
    b.ret(Some(s));
    let sqsum = b.finish();
    // triple(s: f64) -> f64
    let mut b = Builder::new(sym(&mut it, "triple"), F64);
    let sv = b.add_param(F64);
    let three = b.build(F64, Op::ConstFloat(3.0, F64));
    let t = b.build(F64, Op::Bin(BinOp::FMul, three, sv));
    b.ret(Some(t));
    let triple = b.finish();
    // loss(x, out) -> f64
    let (sqsum_sym, triple_sym) = (sqsum.name, triple.name);
    let mut b = Builder::new(sym(&mut it, "helper_loss"), F64);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    let s = b.build(
        F64,
        Op::Call {
            func: sqsum_sym,
            args: vec![x],
        },
    );
    let l = b.build(
        F64,
        Op::Call {
            func: triple_sym,
            args: vec![s],
        },
    );
    b.build_void(Op::Store { ptr: out, value: l });
    b.ret(Some(l));
    let loss = b.finish();

    let err = grad(&loss, &[0], &mut it).expect_err("an un-flattened helper call must refuse");
    assert!(
        err.contains("triple"),
        "the decline must name the callee, got: {err}"
    );

    let prog = program_of(vec![loss.clone(), sqsum, triple]);
    let flat = inline_calls(&prog, &loss);
    assert!(
        !has_call(&flat),
        "every straight-line helper must be spliced"
    );
    let fwd = Fwd {
        func: flat,
        lens: vec![2, 1],
        loss_out: 1,
    };
    let xs = vec![0.7, -1.3];
    let want: Vec<f64> = xs.iter().map(|&v| 6.0 * v).collect();
    let g = gate(&fwd, &[0], &[xs, vec![0.0]], &mut it);
    assert_close(&g[0], &want, "d(3·(x0² + x1²))/dx");
}

/// A helper that indexes a SUB-SLICE: `second(p) = p[1]` called as `second(&x[2])`. Spliced, the
/// load is a two-level gep chain (`gep(gep(x, 2), 1)`), whose indices sum to the element the forward
/// read: `L = x[3]²`, `dL/dx = [0, 0, 0, 2·x[3]]`. (A one-level-only route used to refuse it.)
#[test]
fn sub_slice_helper_routes_through_a_nested_gep() {
    let mut it = Interner::default();
    let mut b = Builder::new(sym(&mut it, "second"), F64);
    let p = b.add_param(PTR);
    let v = load_elem(&mut b, p, 1);
    b.ret(Some(v));
    let second = b.finish();
    let second_sym = second.name;
    let mut b = Builder::new(sym(&mut it, "slice_loss"), F64);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    let two = b.build(MirType::I64, Op::ConstInt(2, MirType::I64));
    let q = b.build(
        PTR,
        Op::Gep {
            ptr: x,
            index: two,
            elem: F64,
        },
    );
    let v = b.build(
        F64,
        Op::Call {
            func: second_sym,
            args: vec![q],
        },
    );
    let l = b.build(F64, Op::Bin(BinOp::FMul, v, v));
    b.build_void(Op::Store { ptr: out, value: l });
    b.ret(Some(l));
    let loss = b.finish();
    let prog = program_of(vec![loss.clone(), second]);
    let fwd = Fwd {
        func: inline_calls(&prog, &loss),
        lens: vec![4, 1],
        loss_out: 1,
    };
    let xs = vec![0.1, -0.2, 0.3, 1.25];
    let g = gate(&fwd, &[0], &[xs, vec![0.0]], &mut it);
    assert_close(&g[0], &[0.0, 0.0, 0.0, 2.5], "d(x[3]²)/dx");
}

/// A recursive helper cannot be flattened (splicing it would not terminate): one level is spliced
/// and the remaining self-call stays a call, which the reverse pass refuses BY NAME.
#[test]
fn recursive_helper_is_named_in_the_decline() {
    let mut it = Interner::default();
    let rec_sym = sym(&mut it, "rec_helper");
    let mut b = Builder::new(rec_sym, F64);
    let p = b.add_param(PTR);
    let v = b.build(
        F64,
        Op::Call {
            func: rec_sym,
            args: vec![p],
        },
    );
    b.ret(Some(v));
    let rec = b.finish();
    let mut b = Builder::new(sym(&mut it, "rec_loss"), F64);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    let v = b.build(
        F64,
        Op::Call {
            func: rec_sym,
            args: vec![x],
        },
    );
    b.build_void(Op::Store { ptr: out, value: v });
    b.ret(Some(v));
    let loss = b.finish();
    let prog = program_of(vec![loss.clone(), rec]);
    let flat = inline_calls(&prog, &loss);
    let err = grad(&flat, &[0], &mut it).expect_err("a recursive helper must refuse");
    assert!(
        err.contains("rec_helper"),
        "the decline must name the callee, got: {err}"
    );
}

/// `f(x) = x0²·x1 + x1³` (f64, scalar path). `∇f = [2·x0·x1, x0² + 3·x1²]`,
/// `H = [[2·x1, 2·x0], [2·x0, 6·x1]]`.
fn cubic_fwd(it: &mut Interner) -> Fwd {
    let mut b = Builder::new(sym(it, "cubic"), F64);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    let a = load_elem(&mut b, x, 0);
    let c = load_elem(&mut b, x, 1);
    let a2 = b.build(F64, Op::Bin(BinOp::FMul, a, a));
    let t1 = b.build(F64, Op::Bin(BinOp::FMul, a2, c));
    let c2 = b.build(F64, Op::Bin(BinOp::FMul, c, c));
    let c3 = b.build(F64, Op::Bin(BinOp::FMul, c2, c));
    let f = b.build(F64, Op::Bin(BinOp::FAdd, t1, c3));
    b.build_void(Op::Store { ptr: out, value: f });
    b.ret(Some(f));
    Fwd {
        func: b.finish(),
        lens: vec![2, 1],
        loss_out: 1,
    }
}

/// P4.15: the Hessian-vector product `H·v` — `grad` of the directional derivative `<v, ∇f>`, which
/// (unlike the void, multi-block `*_grad`) is a single-block scalar function. Gated against the
/// closed-form Hessian and against the central difference of the first-order gradient along `v`
/// (exact for a cubic: its gradient is quadratic).
#[test]
fn hessian_vector_product_matches_closed_form_and_fd() {
    let mut it = Interner::default();
    let fwd = cubic_fwd(&mut it);
    let (gdot, hv) = hvp(&fwd.func, &[0], &mut it).expect("hvp of a scalar-path loss");
    assert_eq!(
        gdot.blocks.len(),
        1,
        "the directional derivative is straight-line"
    );
    let (x0, x1) = (0.7f64, -1.3f64);
    let (v0, v1) = (0.4f64, 1.1f64);
    let want = [2.0 * x1 * v0 + 2.0 * x0 * v1, 2.0 * x0 * v0 + 6.0 * x1 * v1];

    let hv_name = hv.name;
    let prog = program_of(vec![fwd.func.clone(), gdot, hv]);
    // hv(x, out, v, hv_out)
    let mut bufs = vec![vec![x0, x1], vec![0.0], vec![v0, v1], vec![0.0, 0.0]];
    {
        let mut views: Vec<&mut [f64]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
        run_kernel_f64(&prog, hv_name, &mut views, &it).expect("hvp run failed");
    }
    assert_close(&bufs[3], &want, "H·v");

    // Central difference of the first-order gradient along v.
    let g1 = grad(&fwd.func, &[0], &mut it).expect("first-order grad");
    let g1_name = g1.name;
    let prog1 = program_of(vec![fwd.func.clone(), g1]);
    let grad_at = |x: [f64; 2], it: &Interner| -> Vec<f64> {
        let mut bufs = vec![x.to_vec(), vec![0.0], vec![0.0, 0.0]];
        let mut views: Vec<&mut [f64]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
        run_kernel_f64(&prog1, g1_name, &mut views, it).expect("grad run failed");
        bufs[2].clone()
    };
    let eps = 1e-3;
    let gp = grad_at([x0 + eps * v0, x1 + eps * v1], &it);
    let gm = grad_at([x0 - eps * v0, x1 - eps * v1], &it);
    for (k, ((&p, &m), &h)) in gp.iter().zip(&gm).zip(&bufs[3]).enumerate() {
        let fd = (p - m) / (2.0 * eps);
        assert!(
            (fd - h).abs() <= 1e-6,
            "H·v[{k}]: analytic {h} vs finite-diff {fd}"
        );
    }
}

/// The second-order subset is stated exactly: a recognized tensor kernel on the gradient path
/// declines by name rather than producing a directional derivative the tape cannot re-differentiate.
#[test]
fn grad_dot_declines_tensor_kernels_by_name() {
    let mut it = Interner::default();
    let fwd = build_linear(&mut it, 3, 4, 2, None, false);
    let err = grad_dot(&fwd.func, &[0, 1], &mut it).expect_err("kernels are first-order only");
    assert!(
        err.contains("second-order") && err.contains("wukong_sreduce_f32"),
        "the decline must name the boundary and the kernel, got: {err}"
    );
}
