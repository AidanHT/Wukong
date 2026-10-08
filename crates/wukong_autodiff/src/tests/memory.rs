//! Gates for the buffer tape's memory model: gradient accumulation across fan-out, the in-place
//! spellings the recognizers actually emit (`x = softmax(x)`, `h += r`, `h = relu(h + b)`), the
//! write-after-write rule, scalar reads of local arrays a kernel wrote, and the effect-only calls
//! (`print`) a loss may contain.

use super::kernels::{call, kernel_dot};
use super::*;
use crate::grad;
use wukong_mir::{BinOp, Builder, MirType, Op};
use wukong_span::Interner;

/// Fan-out: `x` feeds an activation AND a reduction. `L = Σ C·exp(x) + Σ D·x`, so
/// `dL/dx = C·exp(x) + D` — the second contribution must be ADDED (it used to refuse with
/// "multiple gradient contributions; accumulation is not yet supported").
#[test]
fn fan_out_gradients_accumulate() {
    let n = 5usize;
    let mut it = Interner::default();
    let vmath = sym(&mut it, "wukong_vmath_f32");
    let sreduce = sym(&mut it, "wukong_sreduce_f32");
    let mut b = Builder::new(sym(&mut it, "fanout"), MirType::Void);
    let x = b.add_param(PTR);
    let c = b.add_param(PTR);
    let d = b.add_param(PTR);
    let out = b.add_param(PTR);
    let h = b.alloca(arr(n));
    let (nv, expop) = (ci(&mut b, n as i64), ci(&mut b, VM_EXP));
    b.build_void(Op::Call {
        func: vmath,
        args: vec![x, h, nv, expop],
    });
    let dotop = ci(&mut b, RED_DOT);
    let l1 = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![c, h, nv, dotop],
        },
    );
    let l2 = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![d, x, nv, dotop],
        },
    );
    let l = b.build(MirType::F32, Op::Bin(BinOp::FAdd, l1, l2));
    b.build_void(Op::Store { ptr: out, value: l });
    b.ret(Some(l));
    let fwd = Fwd {
        func: b.finish(),
        lens: vec![n, n, n, 1],
        loss_out: 3,
    };
    let mut seed = 0xFA40u64;
    let xb = rand_vec(&mut seed, n);
    let cb = coef_vec(n);
    let db = rand_vec(&mut seed, n);
    let dx: Vec<f64> = (0..n)
        .map(|i| cb[i] as f64 * (xb[i] as f64).exp() + db[i] as f64)
        .collect();
    let inputs = vec![xb, cb, db, vec![0.0]];
    tape_gate(&fwd, &[0], &inputs, &[dx], &mut it);
}

/// Build `norm(x, x, rows, cols, 0, op)` IN PLACE then `loss = Σ C·x` — the spelling the softmax
/// recognizer matches (`x[i] = exp(x[i] - m) / s` overwrites its input). Params: X, C, out.
fn build_inplace_norm(it: &mut Interner, rows: usize, cols: usize, op: i64) -> Fwd {
    let norm = it.intern("wukong_norm_f32");
    let sreduce = it.intern("wukong_sreduce_f32");
    let mut b = Builder::new(it.intern("inorm"), MirType::Void);
    let x = b.add_param(PTR);
    let c = b.add_param(PTR);
    let out = b.add_param(PTR);
    let (rv, cv, epsv, opv) = (
        ci(&mut b, rows as i64),
        ci(&mut b, cols as i64),
        ci(&mut b, 0),
        ci(&mut b, op),
    );
    b.build_void(Op::Call {
        func: norm,
        args: vec![x, x, rv, cv, epsv, opv],
    });
    let (rc, dotop) = (ci(&mut b, (rows * cols) as i64), ci(&mut b, RED_DOT));
    let loss = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![c, x, rc, dotop],
        },
    );
    b.build_void(Op::Store {
        ptr: out,
        value: loss,
    });
    b.ret(Some(loss));
    Fwd {
        func: b.finish(),
        lens: vec![rows * cols, rows * cols, 1],
        loss_out: 2,
    }
}

/// In-place softmax and log-softmax read only the output and its adjoint, so the output adjoint is
/// transformed into the input adjoint IN PLACE (the runtime's softmax backward permits `dx == dy`).
/// The gradient is w.r.t. the INITIAL contents of `x`: softmax `y·(C - Σ C·y)`, log-softmax
/// `C - softmax·Σ C`.
#[test]
fn inplace_softmax_and_logsoftmax_vjp() {
    let (rows, cols) = (2usize, 4usize);
    for op in [NORM_SOFTMAX, NORM_LOGSOFTMAX] {
        let mut it = Interner::default();
        let fwd = build_inplace_norm(&mut it, rows, cols, op);
        let mut seed = 0x1A50u64 + op as u64;
        let xb = rand_vec(&mut seed, rows * cols);
        let cb = rand_vec(&mut seed, rows * cols);
        let p = softmax_f64(&xb, rows, cols);
        let mut dx = vec![0.0; rows * cols];
        for r in 0..rows {
            let base = r * cols;
            let s_cy: f64 = (0..cols).map(|j| cb[base + j] as f64 * p[base + j]).sum();
            let s_c: f64 = (0..cols).map(|j| cb[base + j] as f64).sum();
            for j in 0..cols {
                let i = base + j;
                dx[i] = if op == NORM_SOFTMAX {
                    p[i] * (cb[i] as f64 - s_cy)
                } else {
                    cb[i] as f64 - p[i] * s_c
                };
            }
        }
        let inputs = vec![xb, cb, vec![0.0]];
        tape_gate(&fwd, &[0], &inputs, &[dx], &mut it);
    }
}

/// `P = X·Wᵀ; P += R` in place (the residual spelling, `velem(P, R, P, .., ID|USE_Y)`), then
/// `loss = Σ G·P`. The output adjoint becomes, in place, the adjoint of the overwritten `P`:
/// `dR = G`, `dP = G`, so `dX`/`dW` are the `nn.Linear` adjoints of `G`.
#[test]
fn inplace_residual_add_vjp() {
    let (m, k, n) = (3usize, 4usize, 2usize);
    let mut it = Interner::default();
    let fwd = kernel_dot(&mut it, &[m * k, n * k, m * n], m * n, |b, it, ins, o| {
        let args = vec![
            ins[0],
            ins[1],
            o,
            ci(b, m as i64),
            ci(b, k as i64),
            ci(b, n as i64),
            ci(b, 0),
        ];
        call(b, it, "wukong_sgemm_nt", args);
        let args = vec![
            o,
            ins[2],
            o,
            ci(b, (m * n) as i64),
            cf(b, 1.0),
            cf(b, 1.0),
            cf(b, 0.0),
            ci(b, VE_ID | VE_USE_Y),
        ];
        call(b, it, "wukong_velem_f32", args);
    });
    let mut seed = 0x2E51u64;
    let xb = rand_vec(&mut seed, m * k);
    let wb = rand_vec(&mut seed, n * k);
    let rb = rand_vec(&mut seed, m * n);
    let gb = rand_vec(&mut seed, m * n);
    let dp: Vec<f64> = gb.iter().map(|&g| g as f64).collect();
    let (dx, dw) = matmul_nt_backward(&dp, &xb, &wb, m, k, n);
    let inputs = vec![xb, wb, rb, gb, vec![0.0]];
    tape_gate(&fwd, &[0, 1, 2], &inputs, &[dx, dw, dp], &mut it);
}

/// `P = X·Wᵀ; P = act(P + b)` with the bias add IN PLACE: the pure add passes the adjoint through
/// and `db = colsum(G)`; a fused ReLU masks it in place from the output (`dz = G·[out > 0]`).
#[test]
fn inplace_bias_vjp() {
    let (m, k, n) = (3usize, 4usize, 2usize);
    for op in [BIAS_ACT_NONE, VM_RELU] {
        let mut it = Interner::default();
        let fwd = kernel_dot(&mut it, &[m * k, n * k, n], m * n, |b, it, ins, o| {
            let args = vec![
                ins[0],
                ins[1],
                o,
                ci(b, m as i64),
                ci(b, k as i64),
                ci(b, n as i64),
                ci(b, 0),
            ];
            call(b, it, "wukong_sgemm_nt", args);
            let args = vec![o, ins[2], o, ci(b, m as i64), ci(b, n as i64), ci(b, op)];
            call(b, it, "wukong_bias_bcast_f32", args);
        });
        let mut seed = 0xB1A7u64 ^ op as u64;
        let (xb, wb, bb) = loop {
            let xb = rand_vec(&mut seed, m * k);
            let wb = rand_vec(&mut seed, n * k);
            let bb = rand_vec(&mut seed, n);
            let p = matmul_nt_f64(&xb, &wb, m, k, n);
            let z: Vec<f64> = (0..m * n).map(|i| p[i] + bb[i % n] as f64).collect();
            if op == BIAS_ACT_NONE
                || (z.iter().all(|&v| v.abs() > 0.15)
                    && z.iter().any(|&v| v > 0.0)
                    && z.iter().any(|&v| v < 0.0))
            {
                break (xb, wb, bb);
            }
        };
        let gb = rand_vec(&mut seed, m * n);
        let p = matmul_nt_f64(&xb, &wb, m, k, n);
        let dz: Vec<f64> = (0..m * n)
            .map(|i| {
                if op == VM_RELU && p[i] + bb[i % n] as f64 <= 0.0 {
                    0.0
                } else {
                    gb[i] as f64
                }
            })
            .collect();
        let (dx, dw) = matmul_nt_backward(&dz, &xb, &wb, m, k, n);
        let db: Vec<f64> = (0..n)
            .map(|j| (0..m).map(|i| dz[i * n + j]).sum())
            .collect();
        let inputs = vec![xb, wb, bb, gb, vec![0.0]];
        tape_gate(&fwd, &[0, 1, 2], &inputs, &[dx, dw, db], &mut it);
    }
}

/// Write-after-write: `h = exp(x1); h = tanh(x2); loss = Σ C·h`. The second kernel overwrites every
/// element the first wrote, so `dL/dx1` is EXACTLY zero and `dL/dx2 = C·(1 - tanh²)`. Without the
/// rule the first writer received the second writer's output adjoint — `C·exp(x1)`, a plausible,
/// wrong, nonzero gradient at exit 0.
#[test]
fn kernel_overwrite_kills_the_earlier_writer() {
    let n = 4usize;
    let mut it = Interner::default();
    let fwd = kernel_dot(&mut it, &[n, n], n, |b, it, ins, o| {
        let args = vec![ins[0], o, ci(b, n as i64), ci(b, VM_EXP)];
        call(b, it, "wukong_vmath_f32", args);
        let args = vec![ins[1], o, ci(b, n as i64), ci(b, VM_TANH)];
        call(b, it, "wukong_vmath_f32", args);
    });
    let mut seed = 0x0DD5u64;
    let x1 = rand_vec(&mut seed, n);
    let x2 = rand_vec(&mut seed, n);
    let cb = coef_vec(n);
    let dx2: Vec<f64> = (0..n)
        .map(|i| {
            let t = (x2[i] as f64).tanh();
            cb[i] as f64 * (1.0 - t * t)
        })
        .collect();
    let inputs = vec![x1, x2, cb, vec![0.0]];
    let g = tape_gate(&fwd, &[0, 1], &inputs, &[vec![0.0; n], dx2], &mut it);
    assert!(
        g[0].iter().all(|&v| v == 0.0),
        "the overwritten writer's input gradient must be exactly zero, got {:?}",
        g[0]
    );
}

/// The store half of the write-after-write rule: `h[0] = x[0]` (a non-constant scalar store), then
/// `h = exp(y)` overwrites every element, then `loss = Σ C·h`. `x[0]` cannot influence the loss, so
/// `dL/dx` is EXACTLY zero and `dL/dy = C·exp(y)`. Without the rule the kernel's output adjoint was
/// still live when the reverse walk reached the store, and `diff_store` handed `C[0]` to `x[0]` — a
/// plausible, wrong, nonzero gradient at exit 0.
#[test]
fn kernel_overwrite_kills_an_earlier_scalar_store() {
    let n = 4usize;
    let mut it = Interner::default();
    let vmath = sym(&mut it, "wukong_vmath_f32");
    let sreduce = sym(&mut it, "wukong_sreduce_f32");
    let mut b = Builder::new(sym(&mut it, "storekill"), MirType::Void);
    let x = b.add_param(PTR);
    let y = b.add_param(PTR);
    let c = b.add_param(PTR);
    let out = b.add_param(PTR);
    let h = b.alloca(arr(n));
    let x0 = b.build(MirType::F32, Op::Load(x, MirType::F32));
    b.build_void(Op::Store { ptr: h, value: x0 });
    let (nv, expop) = (ci(&mut b, n as i64), ci(&mut b, VM_EXP));
    b.build_void(Op::Call {
        func: vmath,
        args: vec![y, h, nv, expop],
    });
    let dotop = ci(&mut b, RED_DOT);
    let l = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![c, h, nv, dotop],
        },
    );
    b.build_void(Op::Store { ptr: out, value: l });
    b.ret(Some(l));
    let fwd = Fwd {
        func: b.finish(),
        lens: vec![n, n, n, 1],
        loss_out: 3,
    };
    let mut seed = 0x5707u64;
    let xb = rand_vec(&mut seed, n);
    let yb = rand_vec(&mut seed, n);
    let cb = coef_vec(n);
    let dy: Vec<f64> = (0..n)
        .map(|i| cb[i] as f64 * (yb[i] as f64).exp())
        .collect();
    let inputs = vec![xb, yb, cb, vec![0.0]];
    let g = tape_gate(&fwd, &[0, 1], &inputs, &[vec![0.0; n], dy], &mut it);
    assert!(
        g[0].iter().all(|&v| v == 0.0),
        "the overwritten store's value gradient must be exactly zero, got {:?}",
        g[0]
    );
}

/// A scalar read of a LOCAL array a kernel wrote: `h = exp(x)` into an `alloca`, then
/// `loss = 2·h[1]`. The load routes into the local array's gradient (a zeroed scratch sized from its
/// `alloca`), which the vmath rule then chains to `x`: `dx = [0, 2·exp(x[1]), 0, 0]`. It used to
/// refuse ("not a parameter or a one-level gep of a parameter").
#[test]
fn scalar_load_of_a_local_kernel_output() {
    let n = 4usize;
    let mut it = Interner::default();
    let vmath = sym(&mut it, "wukong_vmath_f32");
    let mut b = Builder::new(sym(&mut it, "localread"), MirType::Void);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    let h = b.alloca(arr(n));
    let (nv, expop) = (ci(&mut b, n as i64), ci(&mut b, VM_EXP));
    b.build_void(Op::Call {
        func: vmath,
        args: vec![x, h, nv, expop],
    });
    let i1 = ci(&mut b, 1);
    let hp = b.build(
        PTR,
        Op::Gep {
            ptr: h,
            index: i1,
            elem: MirType::F32,
        },
    );
    let h1 = b.build(MirType::F32, Op::Load(hp, MirType::F32));
    let two = cf(&mut b, 2.0);
    let l = b.build(MirType::F32, Op::Bin(BinOp::FMul, two, h1));
    b.build_void(Op::Store { ptr: out, value: l });
    b.ret(Some(l));
    let fwd = Fwd {
        func: b.finish(),
        lens: vec![n, 1],
        loss_out: 1,
    };
    let mut seed = 0x10CAu64;
    let xb = rand_vec(&mut seed, n);
    let mut dx = vec![0.0; n];
    dx[1] = 2.0 * (xb[1] as f64).exp();
    let inputs = vec![xb, vec![0.0]];
    tape_gate(&fwd, &[0], &inputs, &[dx], &mut it);
}

/// A debug `print` inside a loss reads a value and writes no differentiated memory: it is replayed
/// in the forward and skipped by the reverse pass, instead of refusing the whole gradient as an
/// "unrecognized buffer-writing call".
#[test]
fn effect_only_calls_do_not_block_the_gradient() {
    let mut it = Interner::default();
    let print = sym(&mut it, "print");
    let mut b = Builder::new(sym(&mut it, "noisy"), MirType::Void);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    let x0 = load_elem(&mut b, x, 0);
    b.build_void(Op::Call {
        func: print,
        args: vec![x0],
    });
    let sq = b.build(F64, Op::Bin(BinOp::FMul, x0, x0));
    b.build_void(Op::Store {
        ptr: out,
        value: sq,
    });
    b.ret(Some(sq));
    let fwd = b.finish();
    let g = grad(&fwd, &[0], &mut it).expect("a print in the loss must not block the gradient");
    assert!(
        g.blocks
            .iter()
            .flat_map(|bb| &bb.insts)
            .any(|inst| matches!(&inst.op, Op::Call { func, .. } if *func == print)),
        "the forward replay keeps the print"
    );
    // ...and an unknown void call still refuses, naming the callee.
    let mut b = Builder::new(sym(&mut it, "opaque"), MirType::Void);
    let x = b.add_param(PTR);
    let out = b.add_param(PTR);
    let ext = sym(&mut it, "user_extern_writer");
    b.build_void(Op::Call {
        func: ext,
        args: vec![x],
    });
    let x0 = load_elem(&mut b, x, 0);
    b.build_void(Op::Store {
        ptr: out,
        value: x0,
    });
    b.ret(Some(x0));
    let fwd: wukong_mir::Function = b.finish();
    let err = grad(&fwd, &[0], &mut it).expect_err("an opaque writer must refuse");
    assert!(
        err.contains("user_extern_writer"),
        "the decline must name the callee, got: {err}"
    );
}

/// A kernel writing a SUB-SLICE of a buffer whose gradient is live: `h[2..4] = exp(x)` through
/// `gep(h, 2)`, then `loss = Σ C·h` over the whole of `h`. The tape keys one gradient per whole
/// buffer, so the vmath rule found no adjoint under the sub-slice pointer and emitted nothing —
/// `dL/dx` silently zero at exit 0, where the truth is `C[2..4]·exp(x)`. It must refuse by name.
#[test]
fn kernel_writing_a_live_sub_slice_declines_loudly() {
    let mut it = Interner::default();
    let vmath = sym(&mut it, "wukong_vmath_f32");
    let sreduce = sym(&mut it, "wukong_sreduce_f32");
    let mut b = Builder::new(sym(&mut it, "subslice"), MirType::Void);
    let x = b.add_param(PTR);
    let c = b.add_param(PTR);
    let out = b.add_param(PTR);
    let h = b.alloca(arr(4));
    let two = ci(&mut b, 2);
    let tail = b.build(
        PTR,
        Op::Gep {
            ptr: h,
            index: two,
            elem: MirType::F32,
        },
    );
    let expop = ci(&mut b, VM_EXP);
    b.build_void(Op::Call {
        func: vmath,
        args: vec![x, tail, two, expop],
    });
    let (four, dotop) = (ci(&mut b, 4), ci(&mut b, RED_DOT));
    let l = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![c, h, four, dotop],
        },
    );
    b.build_void(Op::Store { ptr: out, value: l });
    b.ret(Some(l));
    let fwd = b.finish();
    let err = grad(&fwd, &[0], &mut it).expect_err("a live sub-slice output must not be dropped");
    assert!(
        err.contains("wukong_vmath_f32") && err.contains("sub-slice"),
        "the decline must name the kernel and the cause, got: {err}"
    );
}
