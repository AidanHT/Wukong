//! N87: reverse mode through `@parallel` loop regions (`region.rs`), on hand-built MIR in the exact
//! shape `wukong_mir_build` outlines and `-O1` leaves — a `wukong_parallel_for(n, &P, env)` call and
//! the canonical counted loop `P(start, end, env)` — gated in `f64` against the central finite
//! difference and a closed form. The source-level twins (every `-O` level, the implicit grain, the
//! native backend) are `wukong_driver::region_grad_tests` and `wukongc`'s `autodiff_cli.rs`.

use super::*;
use crate::{grad_dot, grad_program};
use wukong_mir::{BlockId, CmpOp, ParamAttrs, ParamExtent};

const I64: MirType = MirType::I64;

fn cint(b: &mut Builder, v: i64) -> ValueId {
    b.build(I64, Op::ConstInt(v as i128, I64))
}

fn gep(b: &mut Builder, base: ValueId, idx: ValueId, elem: MirType) -> ValueId {
    b.build(
        PTR,
        Op::Gep {
            ptr: base,
            index: idx,
            elem,
        },
    )
}

fn extent(elems: u64) -> ParamAttrs {
    ParamAttrs {
        extent: Some(ParamExtent {
            elems,
            elem_bytes: 8,
        }),
        ..ParamAttrs::default()
    }
}

/// The outlined body `name(start, end, env)`: `env` holds `ncap` pointers, loaded in the entry; the
/// loop runs `hh` over `[start, end)` and `iteration(b, hh, caps)` emits one straight-line iteration.
/// `extra_block` adds a block the iteration branches through (a non-canonical body).
fn outlined(
    it: &mut Interner,
    name: &str,
    ncap: usize,
    extra_block: bool,
    iteration: impl Fn(&mut Builder, ValueId, &[ValueId]),
) -> Function {
    let mut b = Builder::new(sym(it, name), MirType::Void);
    let start = b.add_param(I64);
    let end = b.add_param(I64);
    let env = b.add_param(PTR);
    let mut caps = Vec::new();
    for k in 0..ncap {
        let slot = if k == 0 {
            env
        } else {
            let kv = cint(&mut b, k as i64);
            gep(&mut b, env, kv, PTR)
        };
        caps.push(b.build(PTR, Op::Load(slot, PTR)));
    }
    let header: BlockId = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();
    let hh = b.block_param(header, I64);
    b.br(header, vec![start]);
    b.switch_to(header);
    let more = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, hh, end));
    b.cond_br(more, body, vec![], exit, vec![]);
    b.switch_to(body);
    iteration(&mut b, hh, &caps);
    if extra_block {
        let mid = b.new_block();
        b.br(mid, vec![]);
        b.switch_to(mid);
    }
    let one = cint(&mut b, 1);
    let next = b.build(I64, Op::Bin(BinOp::Add, hh, one));
    b.br(header, vec![next]);
    b.switch_to(exit);
    b.ret(None);
    b.finish()
}

/// In the loss being built: `env = alloca [k x ptr]` filled with `caps`, then
/// `wukong_parallel_for(n, &body, env)`.
fn call_region(b: &mut Builder, it: &mut Interner, body: &str, n: i64, caps: &[ValueId]) {
    let env = b.alloca(MirType::Array(Box::new(PTR), caps.len() as u32));
    for (k, &c) in caps.iter().enumerate() {
        let slot = if k == 0 {
            env
        } else {
            let kv = cint(b, k as i64);
            gep(b, env, kv, PTR)
        };
        b.build_void(Op::Store {
            ptr: slot,
            value: c,
        });
    }
    let nv = cint(b, n);
    let addr = b.build(PTR, Op::FuncAddr(sym(it, body)));
    let pfor = sym(it, "wukong_parallel_for");
    b.build_void(Op::Call {
        func: pfor,
        args: vec![nv, addr, env],
    });
}

/// `s[i] = x[i]·w[0] + w[1]·w[1]` for `i < n` in a region (x read in its own lane, `w` read by every
/// iteration — privatized), then `L = Σ s[i]²` in the loss. `(loss, outlined body)`.
fn shared_weight_region(it: &mut Interner, n: usize) -> (Function, Function) {
    let body = outlined(it, "wukong$par$0", 3, false, |b, hh, caps| {
        let (x, w, s) = (caps[0], caps[1], caps[2]);
        let xp = gep(b, x, hh, F64);
        let xi = b.build(F64, Op::Load(xp, F64));
        let w0 = b.build(F64, Op::Load(w, F64));
        let one = cint(b, 1);
        let w1p = gep(b, w, one, F64);
        let w1 = b.build(F64, Op::Load(w1p, F64));
        let a = b.build(F64, Op::Bin(BinOp::FMul, xi, w0));
        let c = b.build(F64, Op::Bin(BinOp::FMul, w1, w1));
        let si = b.build(F64, Op::Bin(BinOp::FAdd, a, c));
        let sp = gep(b, s, hh, F64);
        b.build_void(Op::Store { ptr: sp, value: si });
    });
    let mut b = Builder::new(sym(it, "region_loss"), F64);
    let x = b.add_param(PTR);
    let w = b.add_param(PTR);
    let s = b.add_param(PTR);
    let out = b.add_param(PTR);
    b.set_param_attrs(x, extent(n as u64));
    b.set_param_attrs(w, extent(2));
    b.set_param_attrs(s, extent(n as u64));
    call_region(&mut b, it, "wukong$par$0", n as i64, &[x, w, s]);
    let mut acc = b.build(F64, Op::ConstFloat(0.0, F64));
    for i in 0..n as i64 {
        let si = load_elem(&mut b, s, i);
        acc = b.build(F64, Op::Fma(si, si, acc));
    }
    b.build_void(Op::Store {
        ptr: out,
        value: acc,
    });
    b.ret(Some(acc));
    (b.finish(), body)
}

/// `{loss, P, loss_grad, region backward…}` through `grad_program`.
fn region_program(loss: &Function, body: Function, wrt: &[usize], it: &mut Interner) -> Program {
    let src = Program {
        funcs: vec![loss.clone(), body.clone()],
        statics: Vec::new(),
        externs: Vec::new(),
        level: wukong_mir::MirLevel::Low,
    };
    let g = grad_program(&src, loss, wrt, it).expect("grad_program failed");
    assert_eq!(
        g.support.len(),
        2,
        "one iteration backward and its chunk driver"
    );
    let mut funcs = vec![loss.clone(), body, g.func];
    funcs.extend(g.support);
    let prog = Program {
        funcs,
        statics: Vec::new(),
        externs: Vec::new(),
        level: wukong_mir::MirLevel::Low,
    };
    let problems = wukong_mir::verify::verify_program(&prog);
    assert!(
        problems.is_empty(),
        "the gradient program verifies: {problems:?}"
    );
    prog
}

/// The shared adjoint of `w` over 37 iterations: more than the 16 private copies, in chunks of two
/// or three iterations (five of three, then eleven of two) — every iteration must land in exactly
/// one chunk.
/// `dL/dx[i] = 2·s[i]·w[0]`, `dL/dw[0] = Σ 2·s[i]·x[i]`, `dL/dw[1] = Σ 4·s[i]·w[1]`, and
/// `dL/ds = 0` (the region overwrites all of `s`).
#[test]
fn shared_weight_region_matches_f64_finite_differences() {
    let n = 37;
    let mut it = Interner::default();
    let (loss, body) = shared_weight_region(&mut it, n);
    let wrt = [0, 1, 2];
    let prog = region_program(&loss, body, &wrt, &mut it);
    let gname = it.intern("region_loss_grad");
    let fwd = Fwd {
        func: loss,
        lens: vec![n, 2, n, 1],
        loss_out: 3,
    };
    let x: Vec<f64> = (0..n)
        .map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.125)
        .collect();
    let w = vec![0.75, -0.5];
    let inputs = vec![x.clone(), w.clone(), vec![0.0; n], vec![0.0]];
    let got = analytic_grad(&prog, gname, &fwd, &inputs, &wrt, &it);
    let fd = fd_grad(&prog, &fwd, &inputs, &wrt, 1e-6, &it);
    for (gi, (a, f)) in got.iter().zip(&fd).enumerate() {
        for (j, (&av, &fv)) in a.iter().zip(f).enumerate() {
            assert!(
                (av - fv).abs() <= 1e-6 + 1e-6 * fv.abs(),
                "wrt#{gi}[{j}]: analytic {av} vs finite-diff {fv}"
            );
        }
    }
    let s: Vec<f64> = x.iter().map(|&xi| xi * w[0] + w[1] * w[1]).collect();
    let dx: Vec<f64> = s.iter().map(|&si| 2.0 * si * w[0]).collect();
    let dw0: f64 = s.iter().zip(&x).map(|(&si, &xi)| 2.0 * si * xi).sum();
    let dw1: f64 = s.iter().map(|&si| 4.0 * si * w[1]).sum();
    assert_close(&got[0], &dx, "dL/dx");
    assert_close(&got[1], &[dw0, dw1], "dL/dw");
    assert_close(&got[2], &vec![0.0; n], "dL/ds");
}

/// A region the region rule cannot take apart says so by name: an iteration with control flow, a
/// region reading and writing one buffer, a region in a second-order loss, and a region whose body
/// the program does not hold.
#[test]
fn regions_outside_the_supported_shape_decline_by_name() {
    let n = 4;
    // An iteration that branches through a second block.
    let mut it = Interner::default();
    let (loss, _) = shared_weight_region(&mut it, n);
    let bent = outlined(&mut it, "wukong$par$0", 3, true, |b, hh, caps| {
        let xp = gep(b, caps[0], hh, F64);
        let xi = b.build(F64, Op::Load(xp, F64));
        let sp = gep(b, caps[2], hh, F64);
        b.build_void(Op::Store { ptr: sp, value: xi });
    });
    let src = Program {
        funcs: vec![loss.clone(), bent],
        statics: Vec::new(),
        externs: Vec::new(),
        level: wukong_mir::MirLevel::Low,
    };
    let err = grad_program(&src, &loss, &[0], &mut it).expect_err("a branching iteration");
    assert!(
        err.contains("wukong$par$0") && err.contains("not one straight-line block"),
        "{err}"
    );

    // In place: the iteration reads `s[hh]` and writes it back.
    let inplace = outlined(&mut it, "wukong$par$0", 3, false, |b, hh, caps| {
        let sp = gep(b, caps[2], hh, F64);
        let si = b.build(F64, Op::Load(sp, F64));
        let d = b.build(F64, Op::Bin(BinOp::FMul, si, si));
        b.build_void(Op::Store { ptr: sp, value: d });
    });
    let src = Program {
        funcs: vec![loss.clone(), inplace],
        statics: Vec::new(),
        externs: Vec::new(),
        level: wukong_mir::MirLevel::Low,
    };
    let err = grad_program(&src, &loss, &[0], &mut it).expect_err("an in-place region");
    assert!(err.contains("both reads and writes"), "{err}");

    // Second order, and `grad` with no program to find the body in.
    let err = grad_dot(&loss, &[0], &mut it).expect_err("second order through a region");
    assert!(
        err.contains("second-order") && err.contains("@parallel region"),
        "{err}"
    );
    let err = grad(&loss, &[0], &mut it).expect_err("no outlined body");
    assert!(
        err.contains("no VJP rule") && err.contains("grad_program"),
        "{err}"
    );
}
