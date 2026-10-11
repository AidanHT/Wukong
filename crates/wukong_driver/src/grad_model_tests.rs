//! Autodiff from `.wk` source, end to end (P4.6 / P4.7): lex → parse → sema → mir_build → optimize
//! (at every `-O` level, preserving the loss the way the CLI does) → `build_grad` → run, gated
//! against closed forms derived by hand from the source's scalar semantics (stated next to each
//! expectation) and against the finite difference. Each model is a shape the in-crate MIR tests
//! cannot reach, because the recognizers and the optimizer — not a hand-built tape — decide what the
//! backward sees: helper functions at `-O0`/`-O1` (no inliner), a fused `matmul → bias → ReLU`
//! epilogue, a reduction fanned out with a scalar read, and the softmax cross-entropy head.

use super::*;
use wukong_mir::{MirLevel, Program};
use wukong_span::{SourceMap, Symbol};

/// Lex → parse → sema → mir_build, then optimize at `level` (floored at 1, the autodiff contract)
/// preserving `root` exactly as `compile` does for `--emit=grad` / `--train`.
fn compile_at(src: &str, level: u8, root: &str) -> (Program, Interner) {
    let mut sm = SourceMap::new();
    let id = sm.add("grad_model.wk".to_string(), src.to_string());
    let (tokens, ld) = wukong_lexer::tokenize(sm.source(id), id);
    assert!(!ld.iter().any(|d| d.is_error()), "lex errors");
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
    assert!(!pd.iter().any(|d| d.is_error()), "parse errors: {pd:?}");
    let (sema, sd) = wukong_sema::check(&module, &interner);
    assert!(!sd.iter().any(|d| d.is_error()), "sema errors: {sd:?}");
    let (mut program, md) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(!md.iter().any(|d| d.is_error()), "mir_build errors: {md:?}");
    let root = interner.intern(root);
    wukong_opt::optimize_preserving(&mut program, level.max(1), &[root]);
    (program, interner)
}

fn grad_options(of: &str, wrt: &[usize]) -> Options {
    Options {
        grad: GradOptions {
            of: Some(of.to_string()),
            wrt: wrt.to_vec(),
            ..GradOptions::default()
        },
        ..Options::default()
    }
}

/// `{forward, backward}` through the real `build_grad`.
fn grad_program(
    program: &Program,
    interner: &mut Interner,
    of: &str,
    wrt: &[usize],
) -> (Program, Symbol, Symbol) {
    let (fwd, g) = build_grad(program, interner, &grad_options(of, wrt))
        .unwrap_or_else(|e| panic!("build_grad({of}) failed: {e}"));
    let (f, gn) = (fwd.name, g.name);
    let prog = Program {
        funcs: vec![fwd, g],
        level: MirLevel::Low,
        statics: program.statics.clone(),
        externs: program.externs.clone(),
    };
    (prog, f, gn)
}

fn run(prog: &Program, name: Symbol, bufs: &mut [Vec<f32>], it: &Interner) {
    let mut views: Vec<&mut [f32]> = bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
    wukong_interp::run_kernel_f32(prog, name, &mut views, it).expect("kernel run failed");
}

/// The gradient buffer of each `wrt` parameter (one zeroed buffer appended per entry).
fn analytic(
    prog: &Program,
    gname: Symbol,
    inputs: &[Vec<f32>],
    wrt: &[usize],
    it: &Interner,
) -> Vec<Vec<f32>> {
    let mut bufs = inputs.to_vec();
    for &wi in wrt {
        bufs.push(vec![0.0; inputs[wi].len()]);
    }
    run(prog, gname, &mut bufs, it);
    bufs.split_off(inputs.len())
}

/// Assert the analytic gradient matches the central finite difference of the forward's
/// `out[loss_out][0]` over every `wrt` element.
#[allow(clippy::too_many_arguments)]
fn assert_matches_fd(
    prog: &Program,
    fname: Symbol,
    inputs: &[Vec<f32>],
    wrt: &[usize],
    grads: &[Vec<f32>],
    loss_out: usize,
    eps: f32,
    it: &Interner,
) {
    let loss = |bufs: &mut Vec<Vec<f32>>| -> f64 {
        run(prog, fname, bufs, it);
        bufs[loss_out][0] as f64
    };
    for (gi, &wi) in wrt.iter().enumerate() {
        for j in 0..inputs[wi].len() {
            let mut plus = inputs.to_vec();
            plus[wi][j] += eps;
            let mut minus = inputs.to_vec();
            minus[wi][j] -= eps;
            let fd = (loss(&mut plus) - loss(&mut minus)) / (2.0 * eps as f64);
            let a = grads[gi][j] as f64;
            assert!(
                (a - fd).abs() <= 1e-2 + 5e-2 * fd.abs(),
                "wrt#{gi}[{j}]: analytic {a} vs finite-diff {fd}"
            );
        }
    }
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (j, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5 + 1e-5 * w.abs(),
            "{what}[{j}]: got {g}, want {w}"
        );
    }
}

/// P4.6: a loss written with a helper function differentiates at EVERY `-O` level — at `-O0`/`-O1`
/// no inliner runs, and the call used to be the refusal "no VJP rule for `call`". `inline_calls`
/// splices the straight-line helper before the reverse pass.
///
/// `L = sq(x0) + 3·sq(x1)` with `sq(a) = a·a`, so `dL/dx = [2·x0, 6·x1]`; at `x = [0.75, -1.25]`
/// every product and sum is exact in f32, so the gradient is `[1.5, -7.5]` bit for bit.
#[test]
fn helper_function_loss_differentiates_at_every_level() {
    let src = "fn sq(a: f32) -> f32 { return a * a; }\n\
               fn loss(x: [f32; 2], mut out: [f32; 1]) -> f32 {\n\
               let l: f32 = sq(x[0]) + 3.0 * sq(x[1]); out[0] = l; return l; }";
    for level in 0..=3u8 {
        let (program, mut it) = compile_at(src, level, "loss");
        let (prog, _f, g) = grad_program(&program, &mut it, "loss", &[0]);
        let grads = analytic(&prog, g, &[vec![0.75, -1.25], vec![0.0]], &[0], &it);
        assert_eq!(grads[0], vec![1.5, -7.5], "-O{level}");
    }
}

/// A helper taking ARRAY parameters (`mse2(p, t)`): spliced, its loads address the caller's
/// buffers. `L = (x0 - t0)² + (x1 - t1)²`, `dL/dx = 2(x - t)`; at `x = [0.5, -0.25]`,
/// `t = [1.5, 0.25]` that is exactly `[-2.0, -1.0]`.
#[test]
fn array_helper_loss_differentiates() {
    let src = "fn mse2(p: [f32; 2], t: [f32; 2]) -> f32 {\n\
               let d0: f32 = p[0] - t[0]; let d1: f32 = p[1] - t[1];\n\
               return d0 * d0 + d1 * d1; }\n\
               fn loss(x: [f32; 2], t: [f32; 2], mut out: [f32; 1]) -> f32 {\n\
               let l: f32 = mse2(x, t); out[0] = l; return l; }";
    let (program, mut it) = compile_at(src, 1, "loss");
    let (prog, _f, g) = grad_program(&program, &mut it, "loss", &[0]);
    let inputs = [vec![0.5, -0.25], vec![1.5, 0.25], vec![0.0]];
    let grads = analytic(&prog, g, &inputs, &[0], &it);
    assert_eq!(grads[0], vec![-2.0, -1.0]);
}

/// `--train` over a helper-function loss: the training program runs the FLATTENED forward, so it
/// never calls a helper it does not contain. `L = sq(w)`, SGD at `lr = 1/4` halves `w` exactly each
/// step: the trajectory is `[w², (w/2)², (w/4)²]` bit for bit.
#[test]
fn train_runs_a_helper_function_loss() {
    let src = "fn sq(a: f32) -> f32 { return a * a; }\n\
               fn loss(w: [f32; 1], mut out: [f32; 1]) -> f32 {\n\
               let l: f32 = sq(w[0]); out[0] = l; return l; }";
    let (program, mut it) = compile_at(src, 1, "loss");
    let mut opts = grad_options("loss", &[0]);
    opts.grad.train = true;
    opts.grad.train_steps = 2;
    opts.grad.train_lr = 0.25;
    opts.grad.train_opt = TrainOpt::Sgd;
    let traj = train_loop(&program, &mut it, &opts, &[1, 1]).expect("train_loop");
    let mut seed = opts.grad.train_seed;
    let w = train_rand(&mut seed, 1)[0];
    let (h, q) = (w * 0.5, w * 0.25);
    assert_eq!(traj, vec![w * w, h * h, q * q]);
}

/// Fan-out through a recognized reduction and a scalar read of the same buffer — the source that
/// once shipped a silently clobbered gradient and then refused: `L = Σ x + x[0]`, so
/// `dL/dx = [2, 1, 1, 1, 1, 1, 1, 1]` exactly (the two contributions now accumulate).
#[test]
fn reduction_plus_scalar_read_from_source() {
    let src = "@parallel fn loss(x: [f32; 8], mut out: [f32; 1]) -> f32 {\n\
               let mut l: f32 = 0.0; for i in 0..8 { l = l + x[i]; }\n\
               let e: f32 = x[0]; let r: f32 = l + e; out[0] = r; return r; }";
    let (program, mut it) = compile_at(src, 1, "loss");
    let (prog, _f, g) = grad_program(&program, &mut it, "loss", &[0]);
    let x: Vec<f32> = (0..8).map(|i| 0.125 * i as f32 - 0.4).collect();
    let grads = analytic(&prog, g, &[x, vec![0.0]], &[0], &it);
    assert_eq!(grads[0], vec![2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

/// P4.2 from source: `h = relu(x·Wᵀ + b)`, `L = Σ (h - t)²` — the `nn.Linear` + bias + ReLU layer,
/// which the recognizers fuse (`wukong_sgemm_nt_epi`) and which, spelled WITH the bias, used to be
/// undifferentiable. Closed form, derived by hand from the source (x = [[1,2],[3,4]],
/// W = [[0.5,-0.25],[-0.5,0.75]], b = [-0.25,-1.25], t = [0.1,0.2,0.3,0.4]):
/// z = x·Wᵀ + b = [[-0.25,-0.25],[0.25,0.25]] (every |z| = 0.25, clear of the kink for the FD step),
/// h = [0,0,0.25,0.25], dh = 2(h - t) = [-0.2,-0.4,-0.1,-0.3], dz = dh·[z>0] = [0,0,-0.1,-0.3],
/// db = colsum(dz) = [-0.1,-0.3], dW[j,:] = Σ_i dz[i,j]·x[i,:] = [[-0.3,-0.4],[-0.9,-1.2]],
/// dx[i,:] = Σ_j dz[i,j]·W[j,:] = [[0,0],[0.1,-0.2]].
#[test]
fn linear_bias_relu_mse_from_source() {
    let src = "@parallel fn loss(x: [f32; 4], w: [f32; 4], bias: [f32; 2], t: [f32; 4], \
               mut out: [f32; 1]) -> f32 {\n\
               let mut h: [f32; 4] = [0.0; 4];\n\
               for i in 0..2 { for j in 0..2 { let mut s: f32 = 0.0;\n\
                 for k in 0..2 { s = s + x[i * 2 + k] * w[j * 2 + k]; } h[i * 2 + j] = s; } }\n\
               for i in 0..2 { for j in 0..2 { h[i * 2 + j] = fmax(h[i * 2 + j] + bias[j], 0.0); } }\n\
               let mut l: f32 = 0.0;\n\
               for i in 0..4 { l = l + (h[i] - t[i]) * (h[i] - t[i]); }\n\
               out[0] = l; return l; }";
    let (program, mut it) = compile_at(src, 1, "loss");
    let (prog, f, g) = grad_program(&program, &mut it, "loss", &[0, 1, 2]);
    let inputs = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.5, -0.25, -0.5, 0.75],
        vec![-0.25, -1.25],
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.0],
    ];
    let grads = analytic(&prog, g, &inputs, &[0, 1, 2], &it);
    assert_close(&grads[0], &[0.0, 0.0, 0.1, -0.2], "dx");
    assert_close(&grads[1], &[-0.3, -0.4, -0.9, -1.2], "dW");
    assert_close(&grads[2], &[-0.1, -0.3], "db");
    assert_matches_fd(&prog, f, &inputs, &[0, 1, 2], &grads, 4, 1e-2, &it);
}

/// P12.2 from source: the softmax cross-entropy head (the recognized `wukong_xent_fwd_f32`), its
/// per-row losses read back as scalars. `L = Σ_r lse(x_r) - x[r, t_r]`, so
/// `dL/dx[r,i] = softmax(x_r)[i] - [i == t_r]`. Row 0 is `[0,0,0,0]` (softmax exactly 1/4, target
/// 2: `[0.25, 0.25, -0.75, 0.25]`); row 1 is finite-difference gated.
#[test]
fn cross_entropy_head_from_source() {
    let src = "@parallel fn loss(x: [f32; 8], target: [i32; 2], mut out: [f32; 1]) -> f32 {\n\
               let mut lv: [f32; 2] = [0.0; 2];\n\
               for r in 0..2 {\n\
                 let mut m: f32 = x[r * 4];\n\
                 for i in 0..4 { m = fmax(m, x[r * 4 + i]); }\n\
                 let mut s: f32 = 0.0;\n\
                 for i in 0..4 { s = s + exp(x[r * 4 + i] - m); }\n\
                 lv[r] = m + log(s) - x[r * 4 + target[r]];\n\
               }\n\
               let l: f32 = lv[0] + lv[1];\n\
               out[0] = l; return l; }";
    let (program, mut it) = compile_at(src, 1, "loss");
    let (prog, f, g) = grad_program(&program, &mut it, "loss", &[0]);
    // The labels travel in an f32 buffer; the interpreter reads them with `as_int`.
    let inputs = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.3, -0.2, 0.9, 0.1],
        vec![2.0, 0.0],
        vec![0.0],
    ];
    let grads = analytic(&prog, g, &inputs, &[0], &it);
    assert_close(&grads[0][..4], &[0.25, 0.25, -0.75, 0.25], "row 0");
    assert_matches_fd(&prog, f, &inputs, &[0], &grads, 2, 1e-2, &it);
}

/// P4.3 (the ahead-of-time half): `--emit=obj --grad-of=loss` exports the backward next to the
/// forward, so a host training loop links `loss_grad` with no compiler at run time. The appended
/// program verifies, the object's symbol table names `loss_grad`, and a name clash is refused.
#[test]
fn object_emission_exports_the_gradient() {
    let src = "fn loss(x: [f32; 2], mut out: [f32; 1]) -> f32 {\n\
               let l: f32 = x[0] * x[1]; out[0] = l; return l; }";
    let (mut program, mut it) = compile_at(src, 1, "loss");
    let mut opts = grad_options("loss", &[0]);
    opts.emit = EmitStage::Obj;
    assert!(exports_gradient(&opts));
    append_gradient(&mut program, &mut it, &opts).expect("append_gradient");
    let gsym = it.intern("loss_grad");
    assert!(
        program.function(gsym).is_some(),
        "loss_grad was not appended"
    );
    assert!(!verify_or_ice(&program), "the exported program must verify");
    let obj = wukong_codegen_cranelift::emit_object(&program, &it).expect("object emission");
    assert!(
        obj.windows(b"loss_grad".len()).any(|w| w == b"loss_grad"),
        "the object's symbol table must name loss_grad"
    );
    let err = append_gradient(&mut program, &mut it, &opts).unwrap_err();
    assert!(err.contains("loss_grad"), "{err}");
    // Without `--grad-of`, `--emit=obj` is unchanged.
    let plain = Options {
        emit: EmitStage::Obj,
        ..Options::default()
    };
    assert!(!exports_gradient(&plain));
}

/// The `sdpa` loss of N83, written as source: three `@noalias` `[s, d]` inputs, the attention
/// output and the loss sink. `L = Σ_i o_i²` with `o = sdpa(q, k, v)` — every output lane carries a
/// nonzero upstream adjoint `2·o_i`, so a dropped or misrouted adjoint fails somewhere. `CAUSAL` is
/// replaced by the flag under test.
const SDPA_LOSS: &str = "@parallel\n\
    fn loss(@noalias q: [f32; 32], @noalias k: [f32; 32], @noalias v: [f32; 32], \
    mut o: [f32; 32], mut out: [f32; 1]) -> f32 {\n\
    sdpa(q, k, v, o, 4, 8, 0.35355339, CAUSAL);\n\
    let mut l: f32 = 0.0;\n\
    for i in 0..32 { l = l + o[i] * o[i]; }\n\
    out[0] = l;\n\
    return l;\n\
    }";

/// `Σ_i o_i²` with `o = softmax(scale·Q·Kᵀ [causal])·V`, every operand `[s, d]` row-major, in f64:
/// the scores are materialized, the softmax is max-subtracted, and query `i` sees keys `0..=i`
/// under the causal mask — the attention definition, independent of the kernel's online recurrence.
fn sdpa_loss_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    s: usize,
    d: usize,
    scale: f64,
    causal: bool,
) -> f64 {
    let mut loss = 0.0;
    for i in 0..s {
        let keys = if causal { i + 1 } else { s };
        let scores: Vec<f64> = (0..keys)
            .map(|j| scale * (0..d).map(|t| q[i * d + t] * k[j * d + t]).sum::<f64>())
            .collect();
        let mx = scores.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x));
        let e: Vec<f64> = scores.iter().map(|&x| (x - mx).exp()).collect();
        let z: f64 = e.iter().sum();
        for t in 0..d {
            let o: f64 = (0..keys).map(|j| e[j] / z * v[j * d + t]).sum();
            loss += o * o;
        }
    }
    loss
}

/// N83: a loss calling the `sdpa` builtin differentiates from source. The lowering's static form
/// (constant `s`/`d`, fixed-array operands, every input proved apart from `o`) is one bare
/// `wukong_attention_f32` in a straight-line function, which is what `build_grad` accepts; the
/// guarded form used to be refused as "`loss` has 16 blocks". `dQ`, `dK` and `dV` are checked at
/// every element against the f64 CENTRAL FINITE DIFFERENCE of [`sdpa_loss_f64`] — the source's
/// loss re-evaluated in f64 at the same f32 inputs, so the difference is free of the f32 forward's
/// rounding (step `h = 1e-4`: truncation `O(h²)`, cancellation `~1e-16/h`, both far below the band).
///
/// The band is DECLARED, not fitted: `3e-3 + 3e-3·|ref|`, the closed-form band
/// `wukong_autodiff`'s `tape_gate` holds every buffer-tape VJP to against an f64 reference — the
/// band the in-crate attention VJP (`tests/attention.rs`) already meets on hand-built MIR. The
/// analytic side runs the f32 recompute (`sgemm_nt_alpha`, softmax, the four adjoint GEMMs), which
/// is what that band budgets for.
#[test]
fn sdpa_loss_from_source_matches_f64_finite_differences() {
    let (s, d) = (4usize, 8usize);
    let n = s * d;
    let scale_f32: f32 = 0.353_553_39;
    let scale = scale_f32 as f64;
    // Exact dyadic values in [-11/16, 11/16], different for each operand.
    let fill = |seed: usize| -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 37 + seed * 11) % 23) as f32 / 16.0 - 11.0 / 16.0)
            .collect()
    };
    for causal in [0u8, 1] {
        let src = SDPA_LOSS.replace("CAUSAL", &causal.to_string());
        for level in [1u8, 2] {
            let (program, mut it) = compile_at(&src, level, "loss");
            let (prog, _f, g) = grad_program(&program, &mut it, "loss", &[0, 1, 2]);
            let inputs = vec![fill(1), fill(2), fill(3), vec![0.0; n], vec![0.0]];
            let grads = analytic(&prog, g, &inputs, &[0, 1, 2], &it);
            assert!(
                grads.iter().flatten().any(|x| x.abs() > 1e-2),
                "causal={causal} -O{level}: the gradient is trivially zero"
            );
            let wide: Vec<Vec<f64>> = inputs[..3]
                .iter()
                .map(|b| b.iter().map(|&x| x as f64).collect())
                .collect();
            let h = 1e-4;
            for (gi, grad) in grads.iter().enumerate() {
                for j in 0..n {
                    let at = |delta: f64| {
                        let mut ops = wide.clone();
                        ops[gi][j] += delta;
                        sdpa_loss_f64(&ops[0], &ops[1], &ops[2], s, d, scale, causal == 1)
                    };
                    let fd = (at(h) - at(-h)) / (2.0 * h);
                    let a = grad[j] as f64;
                    assert!(
                        (a - fd).abs() <= 3e-3 + 3e-3 * fd.abs(),
                        "causal={causal} -O{level} d{}[{j}]: analytic {a} vs f64 central \
                         difference {fd}",
                        ["q", "k", "v"][gi]
                    );
                }
            }
        }
    }
}
