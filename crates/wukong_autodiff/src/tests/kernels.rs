//! VJP gates for the kernel families the tape gained in the GEMM / transformer / activation
//! coverage pass: the TN, α-scaled and fused-epilogue GEMMs, GEMV / GEVM, transpose, the column
//! reductions, the row-wise losses, RoPE, the embedding gather, the two-input kernel family, the
//! extremum reductions, the fused velem activations and every transcendental-derivative
//! activation. Each is gated against an f64 closed form (and the f32 finite difference) through
//! `tape_gate`, or by the finite difference alone where the kernel's own polynomial is the only
//! reference (`tape_gate_fd`). Every loss is coefficient-weighted (`Σ G·out`), so a rule that drops
//! the upstream adjoint fails at every element.

use super::*;
use crate::grad;
use std::f64::consts::{FRAC_2_SQRT_PI, PI};
use wukong_mir::{Builder, Function, MirType, Op, ValueId};
use wukong_span::Interner;

const VM_SIN: i64 = 15;
const VM_COS: i64 = 16;
const VM_ERF: i64 = 17;
const VM_COSH: i64 = 21;
const VM_LOGSIGMOID: i64 = 31;
const RED_MAX: i64 = 4;
const RED_MIN: i64 = 5;
const RED_MAXABS: i64 = 6;
const VE_RELU6: i64 = 2;
const VM2_POW: i64 = 0;
const VM2_ATAN2: i64 = 1;
const VM2_HYPOT: i64 = 2;
const VM2_SILU_GATE: i64 = 9;
const VM2_GELU_GATE: i64 = 10;
const VM2_SIGMOID_GATE: i64 = 11;
const EPI_IDENTITY: i64 = 0;
const EPI_RELU: i64 = 1;
const EPI_GELU: i64 = 2;
const EPI_SILU: i64 = 3;

/// Build `o = kernel(inputs...); loss = Σ_i G[i]·o[i]`. `emit` receives the builder, the interner,
/// the input parameters and the `out_len` output alloca, and emits the kernel call(s). Params:
/// inputs..., G, out.
pub(super) fn kernel_dot(
    it: &mut Interner,
    input_lens: &[usize],
    out_len: usize,
    emit: impl FnOnce(&mut Builder, &mut Interner, &[ValueId], ValueId),
) -> Fwd {
    let sreduce = it.intern("wukong_sreduce_f32");
    let mut b = Builder::new(it.intern("kdot"), MirType::Void);
    let ins: Vec<ValueId> = input_lens.iter().map(|_| b.add_param(PTR)).collect();
    let g = b.add_param(PTR);
    let out = b.add_param(PTR);
    let o = b.alloca(arr(out_len));
    emit(&mut b, it, &ins, o);
    let (nv, dotop) = (ci(&mut b, out_len as i64), ci(&mut b, RED_DOT));
    let loss = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![g, o, nv, dotop],
        },
    );
    b.build_void(Op::Store {
        ptr: out,
        value: loss,
    });
    b.ret(Some(loss));
    let mut lens = input_lens.to_vec();
    lens.push(out_len);
    lens.push(1);
    let loss_out = lens.len() - 1;
    Fwd {
        func: b.finish(),
        lens,
        loss_out,
    }
}

/// Emit `name(args...)` as a void call.
pub(super) fn call(b: &mut Builder, it: &mut Interner, name: &str, args: Vec<ValueId>) {
    let func = it.intern(name);
    b.build_void(Op::Call { func, args });
}

/// Does `grad(fwd, wrt)` emit a call to `name`? (Recognized backward kernels are gate-blind: the
/// finite difference cannot tell WHICH kernel computed a correct gradient, so the routing is pinned
/// structurally.)
pub(super) fn emits(fwd: &Function, wrt: &[usize], it: &mut Interner, name: &str) -> bool {
    let g = grad(fwd, wrt, it).expect("grad transform failed");
    let s = it.intern(name);
    g.blocks
        .iter()
        .flat_map(|bb| &bb.insts)
        .any(|inst| matches!(&inst.op, Op::Call { func, .. } if *func == s))
}

/// Uniform values in `[lo, hi)`.
fn rand_in(seed: &mut u64, n: usize, lo: f64, hi: f64) -> Vec<f32> {
    (0..n)
        .map(|_| (lo + (hi - lo) * (lcg(seed) + 0.5)) as f32)
        .collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The f64 derivative of each activation whose backward rides a fused kernel or a kernel
/// composition, from the activation's own definition.
fn smooth_derivative_f64(op: i64, x: f64) -> f64 {
    match op {
        VM_SIGMOID => sigmoid(x) * (1.0 - sigmoid(x)),
        VM_TANH => 1.0 - x.tanh() * x.tanh(),
        VM_SILU => sigmoid(x) * (1.0 + x * (1.0 - sigmoid(x))),
        VM_GELU => {
            // The tanh approximation the runtime's gelu1 evaluates.
            let c0 = (2.0 / PI).sqrt();
            let c1 = 0.044715;
            let u = (c0 * (x + c1 * x * x * x)).tanh();
            0.5 * (1.0 + u) + 0.5 * x * (1.0 - u * u) * c0 * (1.0 + 3.0 * c1 * x * x)
        }
        VM_ELU => {
            if x > 0.0 {
                1.0
            } else {
                x.exp()
            }
        }
        VM_SOFTPLUS => sigmoid(x),
        VM_MISH => {
            let t = x.exp().ln_1p().tanh();
            t + x * sigmoid(x) * (1.0 - t * t)
        }
        VM_ERF => FRAC_2_SQRT_PI * (-x * x).exp(),
        VM_SIN => x.cos(),
        VM_COS => -x.sin(),
        VM_COSH => x.sinh(),
        VM_LOGSIGMOID => sigmoid(-x),
        other => panic!("no f64 reference derivative for vmath op {other}"),
    }
}

/// P4.8: the activations whose derivative is a transcendental — six fused `wukong_vmath2_f32`
/// backward codes (sigmoid and tanh newly routed there) and six kernel compositions (mish, erf, sin,
/// cos, cosh, logsigmoid, which used to decline) — against the f64 closed form. With the algebraic
/// set in `vmath_algebraic_activation_vjps`, all 36 runtime activation codes are differentiable.
#[test]
fn smooth_activation_vjps() {
    let xs: &[f32] = &[-1.7, -0.6, 0.4, 1.3];
    for op in [
        VM_SIGMOID,
        VM_TANH,
        VM_SILU,
        VM_GELU,
        VM_ELU,
        VM_SOFTPLUS,
        VM_MISH,
        VM_ERF,
        VM_SIN,
        VM_COS,
        VM_COSH,
        VM_LOGSIGMOID,
    ] {
        let mut it = Interner::default();
        let n = xs.len();
        let fwd = build_vmath_dot(&mut it, n, op);
        let cb = coef_vec(n);
        let want: Vec<f64> = (0..n)
            .map(|i| cb[i] as f64 * smooth_derivative_f64(op, xs[i] as f64))
            .collect();
        let inputs = vec![xs.to_vec(), cb, vec![0.0]];
        tape_gate(&fwd, &[0], &inputs, &[want], &mut it);
    }
}

/// sigmoid and tanh used to run a synthesized per-element loop on the stored output while the
/// runtime shipped fused 256-bit backward kernels for exactly them (`VM2_SIGMOID_BWD`,
/// `VM2_TANH_BWD`). The routing is gate-blind, so pin it.
#[test]
fn sigmoid_and_tanh_ride_the_fused_backward_kernel() {
    for op in [VM_SIGMOID, VM_TANH] {
        let mut it = Interner::default();
        let fwd = build_linear(&mut it, 3, 4, 2, Some(op), false);
        assert!(
            emits(&fwd.func, &[0, 1], &mut it, "wukong_vmath2_f32"),
            "op {op}: backward does not ride wukong_vmath2_f32"
        );
    }
}

/// P4.2: the weight-gradient GEMM in the FORWARD — `C(m x n) = Aᵀ·B`, `A` stored `[k, m]`, `B`
/// stored `[k, n]` (a matmul whose contraction index is the outer index of both operands).
/// `dA[p,i] = Σ_j G[i,j]·B[p,j]`, `dB[p,j] = Σ_i A[p,i]·G[i,j]`.
#[test]
fn matmul_tn_vjp() {
    let (m, k, n) = (3usize, 4usize, 2usize);
    let mut it = Interner::default();
    let fwd = kernel_dot(&mut it, &[k * m, k * n], m * n, |b, it, ins, o| {
        let args = vec![
            ins[0],
            ins[1],
            o,
            ci(b, m as i64),
            ci(b, k as i64),
            ci(b, n as i64),
            ci(b, 0),
        ];
        call(b, it, "wukong_sgemm_tn", args);
    });
    let mut seed = 0x7E11u64;
    let ab = rand_vec(&mut seed, k * m);
    let bb = rand_vec(&mut seed, k * n);
    let gb = rand_vec(&mut seed, m * n);
    let mut da = vec![0.0; k * m];
    let mut db = vec![0.0; k * n];
    for i in 0..m {
        for j in 0..n {
            let g = gb[i * n + j] as f64;
            for p in 0..k {
                da[p * m + i] += g * bb[p * n + j] as f64;
                db[p * n + j] += g * ab[p * m + i] as f64;
            }
        }
    }
    let inputs = vec![ab, bb, gb, vec![0.0]];
    tape_gate(&fwd, &[0, 1], &inputs, &[da, db], &mut it);
}

/// P4.2: the α-scaled `nn.Linear` — the attention score `Q·Kᵀ/√d`. `dC' = α·G`, then the
/// `nn.Linear` adjoints.
#[test]
fn scaled_gemm_nt_alpha_vjp() {
    let (m, k, n) = (3usize, 4usize, 2usize);
    let alpha = 0.375f32;
    let mut it = Interner::default();
    let fwd = kernel_dot(&mut it, &[m * k, n * k], m * n, |b, it, ins, o| {
        let args = vec![
            ins[0],
            ins[1],
            o,
            ci(b, m as i64),
            ci(b, k as i64),
            ci(b, n as i64),
            ci(b, 0),
            cf(b, alpha as f64),
        ];
        call(b, it, "wukong_sgemm_nt_alpha", args);
    });
    let mut seed = 0xA1F4u64;
    let xb = rand_vec(&mut seed, m * k);
    let wb = rand_vec(&mut seed, n * k);
    let gb = rand_vec(&mut seed, m * n);
    let dp: Vec<f64> = gb.iter().map(|&g| alpha as f64 * g as f64).collect();
    let (dx, dw) = matmul_nt_backward(&dp, &xb, &wb, m, k, n);
    let inputs = vec![xb, wb, gb, vec![0.0]];
    tape_gate(&fwd, &[0, 1], &inputs, &[dx, dw], &mut it);
}

/// Build the fused-epilogue tape `o = act(X·Wᵀ + bias)` (bias optional: a `Ptr` null when absent).
/// Params: X, W, [bias], G, out.
fn epilogue_fwd(it: &mut Interner, m: usize, k: usize, n: usize, act: i64, bias: bool) -> Fwd {
    let ins: Vec<usize> = if bias {
        vec![m * k, n * k, n]
    } else {
        vec![m * k, n * k]
    };
    kernel_dot(it, &ins, m * n, |b, it, ins, o| {
        let bias_v = if bias {
            ins[2]
        } else {
            b.build(PTR, Op::ConstInt(0, PTR))
        };
        let args = vec![
            ins[0],
            ins[1],
            o,
            ci(b, m as i64),
            ci(b, k as i64),
            ci(b, n as i64),
            ci(b, 0),
            bias_v,
            ci(b, act),
        ];
        call(b, it, "wukong_sgemm_nt_epi", args);
    })
}

/// P4.2 / P12.2: the fused FFN epilogue `act(X·Wᵀ + bias)` — the kernel a `matmul → bias → act`
/// chain fuses into. Before this rule a performance fusion firing is exactly what removed
/// differentiability. Identity (with and without bias) and ReLU against the closed form; data keeps
/// every ReLU pre-activation clear of the kink.
#[test]
fn fused_epilogue_vjp() {
    let (m, k, n) = (3usize, 4usize, 2usize);
    for (act, bias) in [
        (EPI_IDENTITY, true),
        (EPI_IDENTITY, false),
        (EPI_RELU, true),
    ] {
        let mut it = Interner::default();
        let fwd = epilogue_fwd(&mut it, m, k, n, act, bias);
        let mut seed = 0xE910u64 ^ ((act as u64) << 8) ^ (bias as u64);
        let (xb, wb, bb) = loop {
            let xb = rand_vec(&mut seed, m * k);
            let wb = rand_vec(&mut seed, n * k);
            let bb = rand_vec(&mut seed, n);
            let p = matmul_nt_f64(&xb, &wb, m, k, n);
            let clear = (0..m * n).all(|i| {
                let z = p[i] + if bias { bb[i % n] as f64 } else { 0.0 };
                z.abs() > 0.15
            });
            if act != EPI_RELU || clear {
                break (xb, wb, bb);
            }
        };
        let gb = rand_vec(&mut seed, m * n);
        let p = matmul_nt_f64(&xb, &wb, m, k, n);
        let dz: Vec<f64> = (0..m * n)
            .map(|i| {
                let z = p[i] + if bias { bb[i % n] as f64 } else { 0.0 };
                if act == EPI_RELU && z <= 0.0 {
                    0.0
                } else {
                    gb[i] as f64
                }
            })
            .collect();
        let (dx, dw) = matmul_nt_backward(&dz, &xb, &wb, m, k, n);
        if bias {
            let db: Vec<f64> = (0..n)
                .map(|j| (0..m).map(|i| dz[i * n + j]).sum())
                .collect();
            let inputs = vec![xb, wb, bb, gb, vec![0.0]];
            tape_gate(&fwd, &[0, 1, 2], &inputs, &[dx, dw, db], &mut it);
        } else {
            let inputs = vec![xb, wb, gb, vec![0.0]];
            tape_gate(&fwd, &[0, 1], &inputs, &[dx, dw], &mut it);
        }
    }
}

/// The GELU / SiLU epilogues recompute the pre-activation with the identity epilogue and ride the
/// fused `wukong_vmath2_f32` backward; finite-difference gated (the epilogue's own polynomial is the
/// reference), with the routing pinned structurally.
#[test]
fn fused_epilogue_smooth_vjp() {
    let (m, k, n) = (3usize, 4usize, 2usize);
    for act in [EPI_GELU, EPI_SILU] {
        let mut it = Interner::default();
        let fwd = epilogue_fwd(&mut it, m, k, n, act, true);
        assert!(emits(&fwd.func, &[0, 1, 2], &mut it, "wukong_vmath2_f32"));
        let mut seed = 0x5E10u64 + act as u64;
        let xb = rand_vec(&mut seed, m * k);
        let wb = rand_vec(&mut seed, n * k);
        let bb = rand_vec(&mut seed, n);
        let gb = coef_vec(m * n);
        let inputs = vec![xb, wb, bb, gb, vec![0.0]];
        tape_gate_fd(&fwd, &[0, 1, 2], &inputs, &mut it);
    }
}

/// P4.2: GEMV `y = α·A·x` (plain and α-scaled). `dx[j] = α·Σ_i G[i]·A[i,j]` (the GEVM kernel),
/// `dA[i,j] = α·G[i]·x[j]` (a `k = 1` GEMM).
#[test]
fn gemv_vjp() {
    let (m, n) = (5usize, 3usize);
    for alpha in [None, Some(0.625f32)] {
        let mut it = Interner::default();
        let fwd = kernel_dot(&mut it, &[m * n, n], m, |b, it, ins, o| {
            let mut args = vec![ins[0], ins[1], o, ci(b, m as i64), ci(b, n as i64)];
            match alpha {
                Some(a) => {
                    args.push(cf(b, a as f64));
                    call(b, it, "wukong_sgemv_alpha", args);
                }
                None => call(b, it, "wukong_sgemv", args),
            }
        });
        let al = alpha.unwrap_or(1.0) as f64;
        let mut seed = 0x6E3Au64;
        let ab = rand_vec(&mut seed, m * n);
        let xb = rand_vec(&mut seed, n);
        let gb = rand_vec(&mut seed, m);
        let mut da = vec![0.0; m * n];
        let mut dx = vec![0.0; n];
        for i in 0..m {
            for j in 0..n {
                da[i * n + j] = al * gb[i] as f64 * xb[j] as f64;
                dx[j] += al * gb[i] as f64 * ab[i * n + j] as f64;
            }
        }
        let inputs = vec![ab, xb, gb, vec![0.0]];
        tape_gate(&fwd, &[0, 1], &inputs, &[da, dx], &mut it);
    }
}

/// GEVM `out[j] = α·Σ_i w[i]·A[i,j]` — decode-time attention `probs · V`.
/// `dw[i] = α·Σ_j A[i,j]·G[j]` (an α-scaled GEMV), `dA[i,j] = α·w[i]·G[j]`.
#[test]
fn gevm_vjp() {
    let (rows, cols) = (4usize, 3usize);
    let alpha = 0.5f32;
    let mut it = Interner::default();
    let fwd = kernel_dot(&mut it, &[rows, rows * cols], cols, |b, it, ins, o| {
        let args = vec![
            ins[0],
            ins[1],
            o,
            ci(b, rows as i64),
            ci(b, cols as i64),
            cf(b, alpha as f64),
        ];
        call(b, it, "wukong_sgevm_f32", args);
    });
    let mut seed = 0x6E7Bu64;
    let wb = rand_vec(&mut seed, rows);
    let ab = rand_vec(&mut seed, rows * cols);
    let gb = rand_vec(&mut seed, cols);
    let al = alpha as f64;
    let mut dw = vec![0.0; rows];
    let mut da = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            dw[i] += al * ab[i * cols + j] as f64 * gb[j] as f64;
            da[i * cols + j] = al * wb[i] as f64 * gb[j] as f64;
        }
    }
    let inputs = vec![wb, ab, gb, vec![0.0]];
    tape_gate(&fwd, &[0, 1], &inputs, &[dw, da], &mut it);
}

/// `D = Sᵀ` (S `[rows, cols]`): `dS[r, c] = G[c, r]` — the transpose back.
#[test]
fn transpose_vjp() {
    let (rows, cols) = (3usize, 5usize);
    let mut it = Interner::default();
    let fwd = kernel_dot(&mut it, &[rows * cols], rows * cols, |b, it, ins, o| {
        let args = vec![ins[0], o, ci(b, rows as i64), ci(b, cols as i64)];
        call(b, it, "wukong_transpose_f32", args);
    });
    let mut seed = 0x7A05u64;
    let sb = rand_vec(&mut seed, rows * cols);
    let gb = rand_vec(&mut seed, rows * cols);
    let mut ds = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            ds[r * cols + c] = gb[c * rows + r] as f64;
        }
    }
    let inputs = vec![sb, gb, vec![0.0]];
    tape_gate(&fwd, &[0], &inputs, &[ds], &mut it);
}

/// The column reductions: colsum `dX[i,j] = G[j]`, colmean `G[j]/rows`, colsumsq `2·X[i,j]·G[j]`.
#[test]
fn column_reduction_vjps() {
    let (rows, cols) = (4usize, 3usize);
    for name in [
        "wukong_colsum_f32",
        "wukong_colmean_f32",
        "wukong_colsumsq_f32",
    ] {
        let mut it = Interner::default();
        let fwd = kernel_dot(&mut it, &[rows * cols], cols, |b, it, ins, o| {
            let args = vec![ins[0], o, ci(b, rows as i64), ci(b, cols as i64)];
            call(b, it, name, args);
        });
        let mut seed = 0xC010u64 + name.len() as u64;
        let xb = rand_vec(&mut seed, rows * cols);
        let gb = rand_vec(&mut seed, cols);
        let dx: Vec<f64> = (0..rows * cols)
            .map(|idx| {
                let g = gb[idx % cols] as f64;
                match name {
                    "wukong_colsum_f32" => g,
                    "wukong_colmean_f32" => g / rows as f64,
                    _ => 2.0 * xb[idx] as f64 * g,
                }
            })
            .collect();
        let inputs = vec![xb, gb, vec![0.0]];
        tape_gate(&fwd, &[0], &inputs, &[dx], &mut it);
    }
}

fn softmax_row_f64(row: &[f32]) -> Vec<f64> {
    let mx = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
    let e: Vec<f64> = row.iter().map(|&v| (v as f64 - mx).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// P12.2: the softmax cross-entropy of every classifier / LM head. `loss[r] = lse_r - x[r, t_r]`, so
/// `dX[r,i] = G[r]·(softmax(x_r)[i] - [i == t_r])` — the fused `wukong_xent_bwd_f32` plus a row scale.
/// The labels travel as f32 values here (the interpreter reads them with `as_int`).
#[test]
fn cross_entropy_vjp() {
    let (rows, cols) = (3usize, 4usize);
    let mut it = Interner::default();
    let fwd = kernel_dot(&mut it, &[rows * cols, rows], rows, |b, it, ins, o| {
        let args = vec![ins[0], ins[1], o, ci(b, rows as i64), ci(b, cols as i64)];
        call(b, it, "wukong_xent_fwd_f32", args);
    });
    let mut seed = 0x0E17u64;
    let xb = rand_vec(&mut seed, rows * cols);
    let tb: Vec<f32> = vec![2.0, 0.0, 3.0];
    let gb = coef_vec(rows);
    let mut dx = vec![0.0; rows * cols];
    for r in 0..rows {
        let p = softmax_row_f64(&xb[r * cols..(r + 1) * cols]);
        for i in 0..cols {
            let onehot = if i == tb[r] as usize { 1.0 } else { 0.0 };
            dx[r * cols + i] = gb[r] as f64 * (p[i] - onehot);
        }
    }
    let inputs = vec![xb, tb, gb, vec![0.0]];
    tape_gate(&fwd, &[0], &inputs, &[dx], &mut it);
}

/// log-sum-exp `dX = G[r]·softmax(x_r)`; entropy `dP = -G[r]·(ln p + 1)`; KL divergence
/// `dP = G[r]·(ln p - ln q + 1)`, `dQ = -G[r]·p/q`.
#[test]
fn row_loss_vjps() {
    let (rows, cols) = (2usize, 5usize);
    let mut seed = 0x1055u64;
    // logsumexp
    {
        let mut it = Interner::default();
        let fwd = kernel_dot(&mut it, &[rows * cols], rows, |b, it, ins, o| {
            let args = vec![ins[0], o, ci(b, rows as i64), ci(b, cols as i64)];
            call(b, it, "wukong_logsumexp_f32", args);
        });
        let xb = rand_vec(&mut seed, rows * cols);
        let gb = coef_vec(rows);
        let mut dx = vec![0.0; rows * cols];
        for r in 0..rows {
            let p = softmax_row_f64(&xb[r * cols..(r + 1) * cols]);
            for i in 0..cols {
                dx[r * cols + i] = gb[r] as f64 * p[i];
            }
        }
        let inputs = vec![xb, gb, vec![0.0]];
        tape_gate(&fwd, &[0], &inputs, &[dx], &mut it);
    }
    // entropy (strictly positive probabilities, well inside the log's domain)
    {
        let mut it = Interner::default();
        let fwd = kernel_dot(&mut it, &[rows * cols], rows, |b, it, ins, o| {
            let args = vec![ins[0], o, ci(b, rows as i64), ci(b, cols as i64)];
            call(b, it, "wukong_entropy_f32", args);
        });
        let pb = rand_in(&mut seed, rows * cols, 0.2, 0.9);
        let gb = coef_vec(rows);
        let dp: Vec<f64> = (0..rows * cols)
            .map(|i| -(gb[i / cols] as f64) * ((pb[i] as f64).ln() + 1.0))
            .collect();
        let inputs = vec![pb, gb, vec![0.0]];
        tape_gate(&fwd, &[0], &inputs, &[dp], &mut it);
    }
    // KL divergence
    {
        let mut it = Interner::default();
        let fwd = kernel_dot(
            &mut it,
            &[rows * cols, rows * cols],
            rows,
            |b, it, ins, o| {
                let args = vec![ins[0], ins[1], o, ci(b, rows as i64), ci(b, cols as i64)];
                call(b, it, "wukong_kldiv_f32", args);
            },
        );
        let pb = rand_in(&mut seed, rows * cols, 0.2, 0.9);
        let qb = rand_in(&mut seed, rows * cols, 0.2, 0.9);
        let gb = coef_vec(rows);
        let dp: Vec<f64> = (0..rows * cols)
            .map(|i| gb[i / cols] as f64 * ((pb[i] as f64).ln() - (qb[i] as f64).ln() + 1.0))
            .collect();
        let dq: Vec<f64> = (0..rows * cols)
            .map(|i| -(gb[i / cols] as f64) * pb[i] as f64 / qb[i] as f64)
            .collect();
        let inputs = vec![pb, qb, gb, vec![0.0]];
        tape_gate(&fwd, &[0, 1], &inputs, &[dp, dq], &mut it);
    }
}

/// P12.2: rotary position embedding. Each pair `(a, b) = (x[j], x[j+half])` of row `r` is rotated by
/// `θ = r·f[j]`, so `dx = R(θ)ᵀ·G` (`wukong_rope_bwd_f32`) and — for a trainable frequency table —
/// `df[j] = Σ_r r·(G_b·o_a - G_a·o_b)` over the forward output `o`.
#[test]
fn rope_vjp() {
    let (rows, half) = (3usize, 2usize);
    let dim = 2 * half;
    let mut it = Interner::default();
    let fwd = kernel_dot(&mut it, &[rows * dim, half], rows * dim, |b, it, ins, o| {
        let args = vec![ins[0], ins[1], o, ci(b, rows as i64), ci(b, half as i64)];
        call(b, it, "wukong_rope_f32", args);
    });
    let mut seed = 0x80EEu64;
    let xb = rand_vec(&mut seed, rows * dim);
    let fb: Vec<f32> = vec![0.7, 0.3];
    let gb = rand_vec(&mut seed, rows * dim);
    let mut dx = vec![0.0; rows * dim];
    let mut df = vec![0.0; half];
    for r in 0..rows {
        for j in 0..half {
            let th = r as f64 * fb[j] as f64;
            let (s, c) = th.sin_cos();
            let (ia, ib) = (r * dim + j, r * dim + j + half);
            let (a, bv) = (xb[ia] as f64, xb[ib] as f64);
            let (ga, gbv) = (gb[ia] as f64, gb[ib] as f64);
            dx[ia] = ga * c + gbv * s;
            dx[ib] = -ga * s + gbv * c;
            let (oa, ob) = (a * c - bv * s, bv * c + a * s);
            df[j] += r as f64 * (gbv * oa - ga * ob);
        }
    }
    let inputs = vec![xb, fb, gb, vec![0.0]];
    tape_gate(&fwd, &[0, 1], &inputs, &[dx, df], &mut it);
}

/// P12.2: the embedding gather — the first layer of every language model. `dW[id, :] += G[t, :]`
/// for every token `t` (the scatter-add); token ids 1 and 3 collide on purpose, so the backward
/// must SUM them. The ids travel as f32 values (the interpreter reads them with `as_int`).
#[test]
fn embedding_vjp() {
    let (t, h, v) = (4usize, 3usize, 5usize);
    let mut it = Interner::default();
    let fwd = kernel_dot(&mut it, &[v * h, t], t * h, |b, it, ins, o| {
        let args = vec![
            o,
            ins[0],
            ins[1],
            ci(b, t as i64),
            ci(b, h as i64),
            ci(b, v as i64),
        ];
        call(b, it, "wukong_embedding_f32", args);
    });
    let mut seed = 0xE3BDu64;
    let wb = rand_vec(&mut seed, v * h);
    let ids: Vec<f32> = vec![4.0, 2.0, 0.0, 2.0];
    let gb = rand_vec(&mut seed, t * h);
    let mut dw = vec![0.0; v * h];
    for (tt, &id) in ids.iter().enumerate() {
        for j in 0..h {
            dw[id as usize * h + j] += gb[tt * h + j] as f64;
        }
    }
    let inputs = vec![wb, ids, gb, vec![0.0]];
    tape_gate(&fwd, &[0], &inputs, &[dw], &mut it);
}

/// The two-input kernel family: `pow`, `atan2`, `hypot` and the GLU gates.
#[test]
fn vmath2_family_vjps() {
    let n = 4usize;
    let mut seed = 0x2F00u64;
    for op in [
        VM2_POW,
        VM2_ATAN2,
        VM2_HYPOT,
        VM2_SILU_GATE,
        VM2_SIGMOID_GATE,
        VM2_GELU_GATE,
    ] {
        let mut it = Interner::default();
        let fwd = kernel_dot(&mut it, &[n, n], n, |b, it, ins, o| {
            let args = vec![ins[0], ins[1], o, ci(b, n as i64), ci(b, op)];
            call(b, it, "wukong_vmath2_f32", args);
        });
        // pow needs a positive base; atan2/hypot stay clear of the origin.
        let (xb, yb) = if op == VM2_POW {
            (
                rand_in(&mut seed, n, 0.6, 1.8),
                rand_in(&mut seed, n, -1.2, 1.4),
            )
        } else {
            (
                rand_in(&mut seed, n, 0.4, 1.5),
                rand_in(&mut seed, n, -1.3, -0.3),
            )
        };
        let cb = coef_vec(n);
        let (mut dx, mut dy) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for ((&xv, &yv), &cv) in xb.iter().zip(&yb).zip(&cb) {
            let (x, y, c) = (xv as f64, yv as f64, cv as f64);
            let (gx, gy) = match op {
                VM2_POW => (y * x.powf(y - 1.0), x.powf(y) * x.ln()),
                VM2_ATAN2 => (y / (x * x + y * y), -x / (x * x + y * y)),
                VM2_HYPOT => (x / x.hypot(y), y / x.hypot(y)),
                VM2_SILU_GATE => (
                    y * sigmoid(x) * (1.0 + x * (1.0 - sigmoid(x))),
                    x * sigmoid(x),
                ),
                VM2_SIGMOID_GATE => (y * sigmoid(x) * (1.0 - sigmoid(x)), sigmoid(x)),
                _ => (0.0, 0.0), // gelu gate: finite difference only, below
            };
            dx.push(c * gx);
            dy.push(c * gy);
        }
        let inputs = vec![xb, yb, cb, vec![0.0]];
        if op == VM2_GELU_GATE {
            tape_gate_fd(&fwd, &[0, 1], &inputs, &mut it);
        } else {
            tape_gate(&fwd, &[0, 1], &inputs, &[dx, dy], &mut it);
        }
    }
}

/// MAX / MIN / MAXABS reductions: `g` split evenly over the elements attaining the extremum — the
/// `torch.amax` / `jnp.max` convention. Each data set has a TIE, and at a tie the central difference
/// is exactly that even split (one side moves the extremum, the other does not), so the finite
/// difference and the closed form agree.
#[test]
fn extremum_reductions_split_ties() {
    const K: f64 = 2.5;
    for (op, xs, want) in [
        (
            RED_MAX,
            vec![0.3f32, -0.9, 0.7, 0.7, -0.2],
            vec![0.0, 0.0, K / 2.0, K / 2.0, 0.0],
        ),
        (
            RED_MIN,
            vec![0.3f32, -0.9, 0.7, -0.9, -0.2],
            vec![0.0, K / 2.0, 0.0, K / 2.0, 0.0],
        ),
        (
            RED_MAXABS,
            vec![0.3f32, -0.9, 0.9, 0.2, -0.4],
            vec![0.0, -K / 2.0, K / 2.0, 0.0, 0.0],
        ),
    ] {
        let mut it = Interner::default();
        let fwd = build_reduce2(&mut it, xs.len(), op, true, K);
        let inputs = vec![xs.clone(), xs.clone(), vec![0.0]];
        tape_gate(&fwd, &[0], &inputs, &[want], &mut it);
    }
}

/// The fused velem activations: `relu(x + y)`, `relu(x · y)` and `relu6(x + y)`, differentiated
/// through the OUTPUT mask (`0 < out (< 6)`). Data sits clear of every kink and covers the masked,
/// passing and (for ReLU6) saturated regions.
#[test]
fn velem_fused_activation_vjps() {
    let cb = coef_vec(6);
    // relu(x + y)
    {
        let xs = vec![0.9f32, -0.8, 0.4, -1.5, 2.0, 0.1];
        let ys = vec![0.5f32, 0.3, -0.9, 0.2, 1.0, 0.6];
        let mut it = Interner::default();
        let fwd = build_velem_dot(&mut it, 6, VE_RELU | VE_USE_Y);
        let d: Vec<f64> = (0..6)
            .map(|i| {
                if xs[i] + ys[i] > 0.0 {
                    cb[i] as f64
                } else {
                    0.0
                }
            })
            .collect();
        let inputs = vec![xs, ys, cb.clone(), vec![0.0]];
        tape_gate(&fwd, &[0, 1], &inputs, &[d.clone(), d], &mut it);
    }
    // relu(x · y)
    {
        let xs = vec![0.9f32, -0.8, 0.4, -1.5, 2.0, 0.7];
        let ys = vec![0.5f32, 0.3, -0.9, -0.2, 1.0, 0.6];
        let mut it = Interner::default();
        let fwd = build_velem_dot(&mut it, 6, VE_RELU | VE_HADAMARD);
        let pos = |i: usize| xs[i] * ys[i] > 0.0;
        let dx: Vec<f64> = (0..6)
            .map(|i| {
                if pos(i) {
                    cb[i] as f64 * ys[i] as f64
                } else {
                    0.0
                }
            })
            .collect();
        let dy: Vec<f64> = (0..6)
            .map(|i| {
                if pos(i) {
                    cb[i] as f64 * xs[i] as f64
                } else {
                    0.0
                }
            })
            .collect();
        let inputs = vec![xs.clone(), ys.clone(), cb.clone(), vec![0.0]];
        tape_gate(&fwd, &[0, 1], &inputs, &[dx, dy], &mut it);
    }
    // relu6(x + y): passing on (0, 6), zero below 0 and above 6
    {
        let xs = vec![3.0f32, -0.8, 4.5, 6.5, 1.0, 2.25];
        let ys = vec![1.5f32, 0.3, 2.5, 0.75, 0.5, -0.5];
        let mut it = Interner::default();
        let fwd = build_velem_dot(&mut it, 6, VE_RELU6 | VE_USE_Y);
        let d: Vec<f64> = (0..6)
            .map(|i| {
                let z = xs[i] + ys[i];
                if z > 0.0 && z < 6.0 {
                    cb[i] as f64
                } else {
                    0.0
                }
            })
            .collect();
        let inputs = vec![xs, ys, cb.clone(), vec![0.0]];
        tape_gate(&fwd, &[0, 1], &inputs, &[d.clone(), d], &mut it);
    }
}

/// P4.10: a `@parallel` forward selects the `@parallel` backward kernels — the GEMM adjoints and the
/// fused norm backward — so a native backward uses every core the forward did. (Bit-identical to the
/// serial spelling by the runtime's fixed-blocking contract; gate-blind, so pinned structurally.)
#[test]
fn parallel_forward_selects_parallel_backward_kernels() {
    let mut it = Interner::default();
    let fwd = kernel_dot(&mut it, &[3 * 4, 2 * 4], 3 * 2, |b, it, ins, o| {
        let args = vec![ins[0], ins[1], o, ci(b, 3), ci(b, 4), ci(b, 2), ci(b, 0)];
        call(b, it, "wukong_sgemm_nt_parallel", args);
    });
    for name in ["wukong_sgemm_parallel", "wukong_sgemm_tn_parallel"] {
        assert!(emits(&fwd.func, &[0, 1], &mut it, name), "missing {name}");
    }
    let mut it = Interner::default();
    let fwd = build_norm_dot_sym(&mut it, 3, 4, NORM_SOFTMAX, 0, "wukong_norm_f32_parallel");
    assert!(emits(
        &fwd.func,
        &[0],
        &mut it,
        "wukong_softmax_bwd_f32_parallel"
    ));
}

/// P4.10: the norm and GEMM backwards are the runtime's tuned kernels, not synthesized MIR loops:
/// softmax / LayerNorm / RMSNorm ride `wukong_{softmax,layernorm,rmsnorm}_bwd_f32`, and the
/// `nn.Linear` weight gradient rides `wukong_sgemm_tn` (no transpose loop, no `m·n` scratch).
#[test]
fn backward_rides_the_tuned_runtime_kernels() {
    for (op, name) in [
        (NORM_SOFTMAX, "wukong_softmax_bwd_f32"),
        (NORM_LAYERNORM, "wukong_layernorm_bwd_f32"),
        (NORM_RMSNORM, "wukong_rmsnorm_bwd_f32"),
    ] {
        let mut it = Interner::default();
        let fwd = build_norm_dot(&mut it, 3, 4, op, 1e-5f32.to_bits() as i64);
        assert!(
            emits(&fwd.func, &[0], &mut it, name),
            "op {op}: missing {name}"
        );
    }
    let mut it = Interner::default();
    let fwd = build_linear(&mut it, 3, 4, 2, None, false);
    assert!(emits(&fwd.func, &[1], &mut it, "wukong_sgemm_tn"));
}

/// Every heap scratch the backward allocates is released before it returns, from a block every
/// allocation dominates (a scratch allocated inside a synthesized loop body would not dominate the
/// release — the verifier half of this is the driver's post-autodiff verification).
#[test]
fn heap_scratch_is_released() {
    let mut it = Interner::default();
    // log-softmax (probability scratch) + a fan-out add (accumulation scratch).
    let fwd = build_norm_dot(&mut it, 3, 4, NORM_LOGSOFTMAX, 0);
    let g = grad(&fwd.func, &[0], &mut it).expect("grad");
    let (alloc, free) = (it.intern("wukong_rt_alloc"), it.intern("wukong_rt_free"));
    let count = |s: wukong_span::Symbol| {
        g.blocks
            .iter()
            .flat_map(|bb| &bb.insts)
            .filter(|inst| matches!(&inst.op, Op::Call { func, .. } if *func == s))
            .count()
    };
    assert!(
        count(alloc) > 0,
        "the log-softmax backward allocates a scratch"
    );
    assert_eq!(count(alloc), count(free), "every scratch must be freed");
    // ...and the release sits in the final block, after every loop.
    let last = g
        .blocks
        .iter()
        .find(|bb| matches!(bb.term, wukong_mir::Terminator::Ret(_)));
    let frees_in_last = last
        .expect("a returning block")
        .insts
        .iter()
        .filter(|inst| matches!(&inst.op, Op::Call { func, .. } if *func == free))
        .count();
    assert_eq!(frees_in_last, count(free));
}
