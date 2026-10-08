//! N83: the fused scaled-dot-product attention (`wukong_attention_f32`, the `sdpa` kernel) VJP —
//! the FlashAttention recompute (`tape_attn.rs`). Every gate is an f64 closed form derived from the
//! attention definition (materialized scores, a max-subtracted softmax, the textbook adjoints), plus
//! the f32 finite difference, over a coefficient-weighted loss `L = Σ G·O` so a dropped upstream
//! adjoint fails at every element.

use super::kernels::{call, emits, kernel_dot};
use super::*;
use crate::grad;
use wukong_mir::{Builder, CastKind, MirType, Op};
use wukong_span::Interner;

/// `1/√d` for `d = 4`, exact in f32.
const SCALE: f64 = 0.5;

/// The three adjoints `(dQ, dK, dV)` of `L = Σ_{i,t} G[i,t]·O[i,t]` with
/// `O = softmax(scale·Q·Kᵀ [causal])·V`, all `[s, d]` row-major, in f64 from the f32 inputs:
/// `P_ij = softmax_j(scale·q_i·k_j)` over the keys query `i` sees, `dP_ij = G_i·v_j`,
/// `dS_ij = P_ij·(dP_ij − Σ_l P_il·dP_il)`, `dQ_i = scale·Σ_j dS_ij·k_j`, `dK_j = scale·Σ_i dS_ij·q_i`,
/// `dV_j = Σ_i P_ij·G_i`.
#[allow(clippy::too_many_arguments)]
fn attention_adjoints(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    s: usize,
    d: usize,
    scale: f64,
    causal: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let dot = |a: &[f32], ai: usize, b: &[f32], bi: usize| -> f64 {
        (0..d)
            .map(|t| a[ai * d + t] as f64 * b[bi * d + t] as f64)
            .sum::<f64>()
    };
    let mut dq = vec![0.0; s * d];
    let mut dk = vec![0.0; s * d];
    let mut dv = vec![0.0; s * d];
    for i in 0..s {
        let keys = if causal { i + 1 } else { s };
        let scores: Vec<f64> = (0..keys).map(|j| scale * dot(q, i, k, j)).collect();
        let mx = scores.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x));
        let e: Vec<f64> = scores.iter().map(|&x| (x - mx).exp()).collect();
        let z: f64 = e.iter().sum();
        let p: Vec<f64> = e.iter().map(|&x| x / z).collect();
        let dp: Vec<f64> = (0..keys).map(|j| dot(g, i, v, j)).collect();
        let pdp: f64 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
        for j in 0..keys {
            let ds = p[j] * (dp[j] - pdp) * scale;
            for t in 0..d {
                dv[j * d + t] += p[j] * g[i * d + t] as f64;
                dq[i * d + t] += ds * k[j * d + t] as f64;
                dk[j * d + t] += ds * q[i * d + t] as f64;
            }
        }
    }
    (dq, dk, dv)
}

/// `o = attention(q, k, v, s, d, SCALE, causal)` over three `[s, d]` inputs; params q, k, v, G, out.
fn attention_fwd(it: &mut Interner, s: usize, d: usize, causal: i64) -> Fwd {
    let n = s * d;
    kernel_dot(it, &[n, n, n], n, |b, it, ins, o| {
        let args = vec![
            ins[0],
            ins[1],
            ins[2],
            o,
            ci(b, s as i64),
            ci(b, d as i64),
            cf(b, SCALE),
            ci(b, causal),
        ];
        call(b, it, "wukong_attention_f32", args);
    })
}

/// The non-causal and the causal head, at two shapes, against the closed form and the finite
/// difference — every element of `dQ`, `dK` and `dV`.
#[test]
fn attention_vjp_matches_closed_form() {
    for (s, d) in [(3usize, 4usize), (4, 4)] {
        for causal in [false, true] {
            let mut it = Interner::default();
            let fwd = attention_fwd(&mut it, s, d, causal as i64);
            let mut seed = 0xA77E_u64 ^ ((s as u64) << 8) ^ causal as u64;
            let q = rand_vec(&mut seed, s * d);
            let k = rand_vec(&mut seed, s * d);
            let v = rand_vec(&mut seed, s * d);
            let g = rand_vec(&mut seed, s * d);
            let (dq, dk, dv) = attention_adjoints(&q, &k, &v, &g, s, d, SCALE, causal);
            let inputs = vec![q, k, v, g, vec![0.0]];
            let got = tape_gate(&fwd, &[0, 1, 2], &inputs, &[dq, dk, dv], &mut it);
            assert!(
                got.iter().flatten().any(|x| x.abs() > 1e-3),
                "{s}x{d} causal={causal}: the gradient is trivially zero"
            );
            if causal {
                // Query 0 sees only key 0: its weight is exactly 1, so dS row 0 — and dQ row 0 —
                // is zero, and no masked key leaks a contribution into it.
                let row0 = &got[0][..d];
                assert!(row0.iter().all(|x| x.abs() < 1e-6), "dQ row 0 = {row0:?}");
            }
        }
    }
}

/// Self-attention over ONE buffer (`q = k = v = x`): the three adjoint GEMMs land in the same
/// gradient buffer, so the second and third must accumulate (`beta = 1`) — `dX = dQ + dK + dV`.
#[test]
fn attention_over_one_buffer_accumulates_all_three_adjoints() {
    let (s, d) = (3usize, 4usize);
    let n = s * d;
    for causal in [false, true] {
        let mut it = Interner::default();
        let fwd = kernel_dot(&mut it, &[n], n, |b, it, ins, o| {
            let args = vec![
                ins[0],
                ins[0],
                ins[0],
                o,
                ci(b, s as i64),
                ci(b, d as i64),
                cf(b, SCALE),
                ci(b, causal as i64),
            ];
            call(b, it, "wukong_attention_f32", args);
        });
        let mut seed = 0x5E1F_u64 ^ causal as u64;
        let x = rand_vec(&mut seed, n);
        let g = rand_vec(&mut seed, n);
        let (dq, dk, dv) = attention_adjoints(&x, &x, &x, &g, s, d, SCALE, causal);
        let dx: Vec<f64> = (0..n).map(|i| dq[i] + dk[i] + dv[i]).collect();
        let inputs = vec![x, g, vec![0.0]];
        tape_gate(&fwd, &[0], &inputs, &[dx], &mut it);
    }
}

/// A causal flag that is NOT a compile-time constant (here read from a buffer) is reproduced in the
/// backward by a run-time select over the triangle: flag 1 must match the causal closed form, flag 0
/// the full one.
#[test]
fn attention_runtime_causal_flag() {
    let (s, d) = (4usize, 4usize);
    let n = s * d;
    for flag in [1.0f32, 0.0] {
        let mut it = Interner::default();
        let fwd = kernel_dot(&mut it, &[n, n, n, 1], n, |b, it, ins, o| {
            let raw = b.build(MirType::F32, Op::Load(ins[3], MirType::F32));
            let causal = b.build(MirType::I64, Op::Cast(CastKind::FpToSi, raw, MirType::I64));
            let args = vec![
                ins[0],
                ins[1],
                ins[2],
                o,
                ci(b, s as i64),
                ci(b, d as i64),
                cf(b, SCALE),
                causal,
            ];
            call(b, it, "wukong_attention_f32", args);
        });
        let mut seed = 0xF1A6_u64;
        let q = rand_vec(&mut seed, n);
        let k = rand_vec(&mut seed, n);
        let v = rand_vec(&mut seed, n);
        let g = rand_vec(&mut seed, n);
        let (dq, dk, dv) = attention_adjoints(&q, &k, &v, &g, s, d, SCALE, flag != 0.0);
        let inputs = vec![q, k, v, vec![flag], g, vec![0.0]];
        tape_gate(&fwd, &[0, 1, 2], &inputs, &[dq, dk, dv], &mut it);
    }
}

/// The backward is the FlashAttention recompute on the tuned kernels — the α-scaled score GEMM, the
/// fused softmax backward and the transposed weight-gradient GEMM — and a `V`-only gradient skips
/// the `dS` pass entirely. (Recognized kernels are gate-blind, so the routing is pinned
/// structurally.)
#[test]
fn attention_backward_rides_the_tuned_kernels() {
    let mut it = Interner::default();
    let fwd = attention_fwd(&mut it, 3, 4, 1);
    for name in [
        "wukong_sgemm_nt_alpha",
        "wukong_norm_f32",
        "wukong_softmax_bwd_f32",
        "wukong_sgemm_tn",
        "wukong_sgemm",
    ] {
        assert!(
            emits(&fwd.func, &[0, 1, 2], &mut it, name),
            "missing {name}"
        );
    }
    assert!(!emits(&fwd.func, &[2], &mut it, "wukong_softmax_bwd_f32"));
    assert!(emits(&fwd.func, &[2], &mut it, "wukong_sgemm_tn"));
}

/// An attention whose output overwrites one of its inputs has destroyed the operand the backward
/// recomputes from; refuse by name rather than differentiate against the overwritten buffer.
#[test]
fn attention_with_output_aliasing_an_input_declines_loudly() {
    let (s, d) = (2usize, 2usize);
    let mut it = Interner::default();
    let attn = sym(&mut it, "wukong_attention_f32");
    let sreduce = sym(&mut it, "wukong_sreduce_f32");
    let mut b = Builder::new(sym(&mut it, "attn_inplace"), MirType::F32);
    let q = b.add_param(PTR);
    let k = b.add_param(PTR);
    let v = b.add_param(PTR);
    let out = b.add_param(PTR);
    let args = vec![
        q,
        k,
        v,
        q,
        ci(&mut b, s as i64),
        ci(&mut b, d as i64),
        cf(&mut b, SCALE),
        ci(&mut b, 0),
    ];
    b.build_void(Op::Call { func: attn, args });
    let (nv, sum) = (ci(&mut b, (s * d) as i64), ci(&mut b, RED_SUM));
    let loss = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![q, q, nv, sum],
        },
    );
    b.build_void(Op::Store {
        ptr: out,
        value: loss,
    });
    b.ret(Some(loss));
    let err = grad(&b.finish(), &[1], &mut it).expect_err("an in-place attention must decline");
    assert!(err.contains("attention output"), "{err}");
}
