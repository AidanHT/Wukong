//! N87: autodiff through `@parallel` loop regions, end to end from `.wk` source. Each loss runs a
//! loop the region outliner turns into `wukong_parallel_for(n, &body, env)`; its backward is the
//! region rule of `wukong_autodiff::grad_program` (`region.rs`). Every gradient is gated against a
//! closed form derived by hand from the source's scalar semantics (stated next to each test) and
//! against the central finite difference, at every `-O` level, with and without the implicit grain
//! the CLI applies — and the outlined call is asserted, so a loss that stopped being a region would
//! fail here rather than pass as an ordinary serial loss.

use super::*;
use wukong_mir::{MirLevel, Program};
use wukong_span::{SourceMap, Symbol};

/// Lex → parse → sema → mir_build (the gradient-compile lowering, `implicit_grain` as given), then
/// optimize at `level` (floored at 1) without SLP and put the `-O1` region bodies in place, exactly
/// as `compile` does for `--emit=grad` / `--train`.
fn compile_at(src: &str, level: u8, implicit_grain: bool) -> (Program, Interner) {
    let mut sm = SourceMap::new();
    let id = sm.add("region_grad.wk".to_string(), src.to_string());
    let (tokens, ld) = wukong_lexer::tokenize(sm.source(id), id);
    assert!(!ld.iter().any(|d| d.is_error()), "lex errors");
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
    assert!(!pd.iter().any(|d| d.is_error()), "parse errors: {pd:?}");
    let (sema, sd) = wukong_sema::check(&module, &interner);
    assert!(!sd.iter().any(|d| d.is_error()), "sema errors: {sd:?}");
    let lower = wukong_mir_build::LowerOptions {
        implicit_grain,
        ..wukong_mir_build::LowerOptions::default()
    };
    let (mut program, md) =
        wukong_mir_build::lower_program_with(&module, &sema, &mut interner, lower);
    assert!(!md.iter().any(|d| d.is_error()), "mir_build errors: {md:?}");
    let root = interner.intern("loss");
    let level = level.max(1);
    let snapshot = if level >= 2 {
        RegionSnapshot::take(&program, &interner)
    } else {
        RegionSnapshot::default()
    };
    wukong_opt::optimize_preserving_with(
        &mut program,
        level,
        &[root],
        wukong_opt::OptimizeOptions::AUTODIFF_INPUT,
    );
    snapshot
        .restore_at_o1(&mut program, true)
        .expect("region bodies optimize at -O1");
    (program, interner)
}

fn grad_options(wrt: &[usize]) -> Options {
    Options {
        grad: GradOptions {
            of: Some("loss".to_string()),
            wrt: wrt.to_vec(),
            ..GradOptions::default()
        },
        ..Options::default()
    }
}

/// `{forward, backward, region backwards, outlined bodies}` through the real `build_grad_full`.
fn differentiate(
    program: &Program,
    interner: &mut Interner,
    wrt: &[usize],
) -> (Program, Symbol, Symbol) {
    let built = build_grad_full(program, interner, &grad_options(wrt))
        .unwrap_or_else(|e| panic!("build_grad_full failed: {e}"));
    let (f, g) = (built.fwd.name, built.grad.name);
    let calls_region = |func: &wukong_mir::Function| {
        func.blocks.iter().flat_map(|b| &b.insts).any(|i| {
            matches!(&i.op, wukong_mir::Op::Call { func, .. }
                if interner.resolve(*func) == "wukong_parallel_for")
        })
    };
    assert!(
        calls_region(&built.fwd),
        "the forward must run a @parallel region"
    );
    assert!(
        calls_region(&built.grad),
        "the backward must run the region's backward through wukong_parallel_for"
    );
    let prog = Program {
        funcs: built.into_funcs(),
        level: MirLevel::Low,
        statics: program.statics.clone(),
        externs: program.externs.clone(),
    };
    (prog, f, g)
}

fn run_f32(prog: &Program, name: Symbol, bufs: &mut [Vec<f32>], it: &Interner) {
    let mut views: Vec<&mut [f32]> = bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
    wukong_interp::run_kernel_f32(prog, name, &mut views, it).expect("kernel run failed");
}

fn run_f64(prog: &Program, name: Symbol, bufs: &mut [Vec<f64>], it: &Interner) {
    let mut views: Vec<&mut [f64]> = bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
    wukong_interp::run_kernel_f64(prog, name, &mut views, it).expect("kernel run failed");
}

/// The gradient of each `wrt` parameter, through the interpreter.
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
    run_f32(prog, gname, &mut bufs, it);
    bufs.split_off(inputs.len())
}

/// The f32 central difference of `out[loss_out][0]` over every `wrt` element, against `grads`.
#[allow(clippy::too_many_arguments)]
fn assert_matches_fd(
    prog: &Program,
    fname: Symbol,
    inputs: &[Vec<f32>],
    wrt: &[usize],
    grads: &[Vec<f32>],
    loss_out: usize,
    it: &Interner,
    what: &str,
) {
    let eps = 1e-2f32;
    let loss = |bufs: &mut Vec<Vec<f32>>| -> f64 {
        run_f32(prog, fname, bufs, it);
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
                (a - fd).abs() <= 2e-2 + 2e-2 * fd.abs(),
                "{what}: wrt#{gi}[{j}]: analytic {a} vs finite-diff {fd}"
            );
        }
    }
}

/// Analytic `got` against the closed form `want` (computed in f64 from the source's semantics).
fn assert_close(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (j, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g as f64 - w).abs() <= 1e-4 + 1e-5 * w.abs(),
            "{what}[{j}]: got {g}, want {w}"
        );
    }
}

/// Small deterministic inputs in [-1, 1).
fn values(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let k = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(40503));
            ((k >> 8) % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

/// Gate `src` (whose loss's buffer parameters have the lengths `lens`, the last one the loss
/// output) at -O0..-O3, with and without the implicit grain: the gradient of each `wrt` parameter
/// must equal `want` and the finite difference, with the same bits in every configuration.
fn gate(
    src: &str,
    lens: &[usize],
    wrt: &[usize],
    want: impl Fn(&[Vec<f32>]) -> Vec<Vec<f64>>,
    what: &str,
) {
    let inputs: Vec<Vec<f32>> = lens
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            if i + 1 == lens.len() {
                vec![0.0; n]
            } else {
                values(n, i as u32 + 1)
            }
        })
        .collect();
    let expected = want(&inputs);
    let mut first: Option<Vec<Vec<u32>>> = None;
    for level in 0..=3u8 {
        for grain in [false, true] {
            let (program, mut it) = compile_at(src, level, grain);
            let (prog, f, g) = differentiate(&program, &mut it, wrt);
            let grads = analytic(&prog, g, &inputs, wrt, &it);
            let tag = format!("{what} -O{level} grain={grain}");
            for (gi, (got, w)) in grads.iter().zip(&expected).enumerate() {
                assert_close(got, w, &format!("{tag} wrt#{gi}"));
            }
            assert_matches_fd(&prog, f, &inputs, wrt, &grads, lens.len() - 1, &it, &tag);
            // The same bits at every level and grain: optimization is invariant and the chunks of
            // the backward depend on the trip count alone.
            let bits: Vec<Vec<u32>> = grads
                .iter()
                .map(|v| v.iter().map(|x| x.to_bits()).collect())
                .collect();
            match &first {
                None => first = Some(bits),
                Some(b) => assert_eq!(b, &bits, "{tag}: gradient bits differ across levels"),
            }
        }
    }
}

/// An elementwise region: `h[i] = x[2i]·x[2i+1]`, `L = Σ h[i]²`. Each iteration reads its own pair
/// (stride 2, band 0), so no adjoint is privatized: `dL/dx[2i] = 2·h[i]·x[2i+1]` and
/// `dL/dx[2i+1] = 2·h[i]·x[2i]`.
#[test]
fn elementwise_region_differentiates() {
    const SRC: &str = "@parallel
fn loss(x: [f32; 16], mut out: [f32; 1]) -> f32 {
    let mut h: [f32; 8] = [0.0; 8];
    for i in 0..8 {
        h[i] = x[2 * i] * x[2 * i + 1];
    }
    let mut l: f32 = 0.0;
    for i in 0..8 {
        l = l + h[i] * h[i];
    }
    out[0] = l;
    return l;
}
";
    gate(
        SRC,
        &[16, 1],
        &[0],
        |inp| {
            let x: Vec<f64> = inp[0].iter().map(|&v| v as f64).collect();
            let mut dx = vec![0.0; 16];
            for i in 0..8 {
                let h = x[2 * i] * x[2 * i + 1];
                dx[2 * i] = 2.0 * h * x[2 * i + 1];
                dx[2 * i + 1] = 2.0 * h * x[2 * i];
            }
            vec![dx]
        },
        "elementwise",
    );
}

/// A row reduction against a SHARED weight vector: `s[r] = Σ_j x[4r+j]·w[j]`, `L = Σ s[r]²`. Every
/// iteration reads all of `w`, so its adjoint is privatized per chunk and added back in chunk order:
/// `dL/dw[j] = Σ_r 2·s[r]·x[4r+j]`, `dL/dx[4r+j] = 2·s[r]·w[j]`.
#[test]
fn shared_weight_row_reduction_region_differentiates() {
    const SRC: &str = "@parallel
fn loss(x: [f32; 16], w: [f32; 4], mut out: [f32; 1]) -> f32 {
    let mut s: [f32; 4] = [0.0; 4];
    for r in 0..4 {
        s[r] = x[r * 4] * w[0] + x[r * 4 + 1] * w[1] + x[r * 4 + 2] * w[2] + x[r * 4 + 3] * w[3];
    }
    let mut l: f32 = 0.0;
    for r in 0..4 {
        l = l + s[r] * s[r];
    }
    out[0] = l;
    return l;
}
";
    gate(
        SRC,
        &[16, 4, 1],
        &[0, 1],
        |inp| {
            let x: Vec<f64> = inp[0].iter().map(|&v| v as f64).collect();
            let w: Vec<f64> = inp[1].iter().map(|&v| v as f64).collect();
            let (mut dx, mut dw) = (vec![0.0; 16], vec![0.0; 4]);
            for r in 0..4 {
                let s: f64 = (0..4).map(|j| x[4 * r + j] * w[j]).sum();
                for j in 0..4 {
                    dx[4 * r + j] = 2.0 * s * w[j];
                    dw[j] += 2.0 * s * x[4 * r + j];
                }
            }
            vec![dx, dw]
        },
        "shared weight",
    );
}

/// A kernel call inside the region: `t = dot(x, w)` (a whole-buffer `wukong_sreduce_f32` in every
/// iteration), `h[i] = x[i]·t`, `L = Σ h[i]² = t²·S` with `S = Σ x[i]²`. Both captures are read by a
/// kernel, so both adjoints are privatized: `dL/dx[j] = 2·t·S·w[j] + 2·t²·x[j]` and
/// `dL/dw[j] = 2·t·S·x[j]`.
#[test]
fn kernel_call_inside_region_differentiates() {
    const SRC: &str = "@parallel
fn loss(x: [f32; 8], w: [f32; 8], mut out: [f32; 1]) -> f32 {
    let mut h: [f32; 8] = [0.0; 8];
    for i in 0..8 {
        let t: f32 = dot(x, w);
        h[i] = x[i] * t;
    }
    let mut l: f32 = 0.0;
    for i in 0..8 {
        l = l + h[i] * h[i];
    }
    out[0] = l;
    return l;
}
";
    gate(
        SRC,
        &[8, 8, 1],
        &[0, 1],
        |inp| {
            let x: Vec<f64> = inp[0].iter().map(|&v| v as f64).collect();
            let w: Vec<f64> = inp[1].iter().map(|&v| v as f64).collect();
            let t: f64 = x.iter().zip(&w).map(|(a, b)| a * b).sum();
            let s: f64 = x.iter().map(|a| a * a).sum();
            let dx = (0..8)
                .map(|j| 2.0 * t * s * w[j] + 2.0 * t * t * x[j])
                .collect();
            let dw = (0..8).map(|j| 2.0 * t * s * x[j]).collect();
            vec![dx, dw]
        },
        "kernel call",
    );
}

/// A captured SCALAR: `t = w[0]·w[1]` lives in a slot the region reads in every iteration (a
/// one-element shared adjoint), `h[i] = x[2i]·t + x[2i+1]`, `L = Σ h[i]²`:
/// `dL/dx[2i] = 2·h[i]·t`, `dL/dx[2i+1] = 2·h[i]`, and with `dt = Σ 2·h[i]·x[2i]`,
/// `dL/dw[0] = dt·w[1]`, `dL/dw[1] = dt·w[0]`.
#[test]
fn captured_scalar_region_differentiates() {
    const SRC: &str = "@parallel
fn loss(x: [f32; 16], w: [f32; 4], mut out: [f32; 1]) -> f32 {
    let t: f32 = w[0] * w[1];
    let mut h: [f32; 8] = [0.0; 8];
    for i in 0..8 {
        h[i] = x[2 * i] * t + x[2 * i + 1];
    }
    let mut l: f32 = 0.0;
    for i in 0..8 {
        l = l + h[i] * h[i];
    }
    out[0] = l;
    return l;
}
";
    gate(
        SRC,
        &[16, 4, 1],
        &[0, 1],
        |inp| {
            let x: Vec<f64> = inp[0].iter().map(|&v| v as f64).collect();
            let w: Vec<f64> = inp[1].iter().map(|&v| v as f64).collect();
            let t = w[0] * w[1];
            let (mut dx, mut dt) = (vec![0.0; 16], 0.0);
            for i in 0..8 {
                let h = x[2 * i] * t + x[2 * i + 1];
                dx[2 * i] = 2.0 * h * t;
                dx[2 * i + 1] = 2.0 * h;
                dt += 2.0 * h * x[2 * i];
            }
            vec![dx, vec![dt * w[1], dt * w[0], 0.0, 0.0]]
        },
        "captured scalar",
    );
}

/// A region inside a HELPER: a whole-function `@parallel` row kernel (outlined as `rows$par`) that
/// `inline_calls` splices into the loss, so the loss runs its region. Same math as the shared-weight
/// test.
#[test]
fn region_in_spliced_helper_differentiates() {
    const SRC: &str = "@parallel
fn rows(x: [f32; 16], w: [f32; 4], @noalias mut s: [f32; 4]) {
    for r in 0..4 {
        s[r] = x[r * 4] * w[0] + x[r * 4 + 1] * w[1] + x[r * 4 + 2] * w[2] + x[r * 4 + 3] * w[3];
    }
}

@parallel
fn loss(x: [f32; 16], w: [f32; 4], mut out: [f32; 1]) -> f32 {
    let mut s: [f32; 4] = [0.0; 4];
    rows(x, w, s);
    let mut l: f32 = 0.0;
    for r in 0..4 {
        l = l + s[r] * s[r];
    }
    out[0] = l;
    return l;
}
";
    gate(
        SRC,
        &[16, 4, 1],
        &[0, 1],
        |inp| {
            let x: Vec<f64> = inp[0].iter().map(|&v| v as f64).collect();
            let w: Vec<f64> = inp[1].iter().map(|&v| v as f64).collect();
            let (mut dx, mut dw) = (vec![0.0; 16], vec![0.0; 4]);
            for r in 0..4 {
                let s: f64 = (0..4).map(|j| x[4 * r + j] * w[j]).sum();
                for j in 0..4 {
                    dx[4 * r + j] = 2.0 * s * w[j];
                    dw[j] += 2.0 * s * x[4 * r + j];
                }
            }
            vec![dx, dw]
        },
        "spliced helper",
    );
}

/// A buffer the region only stores into may change afterwards: `h[i] = x[2i]·x[2i+1]`, then
/// `h = 2·h` in place (a `wukong_velem_f32` after the region), `L = Σ h[i]`:
/// `dL/dx[2i] = 2·x[2i+1]`, `dL/dx[2i+1] = 2·x[2i]`.
#[test]
fn region_output_rewritten_afterwards_differentiates() {
    const SRC: &str = "@parallel
fn loss(x: [f32; 16], mut out: [f32; 1]) -> f32 {
    let mut h: [f32; 8] = [0.0; 8];
    for i in 0..8 {
        h[i] = x[2 * i] * x[2 * i + 1];
    }
    for i in 0..8 {
        h[i] = h[i] * 2.0;
    }
    let mut l: f32 = 0.0;
    for i in 0..8 {
        l = l + h[i];
    }
    out[0] = l;
    return l;
}
";
    gate(
        SRC,
        &[16, 1],
        &[0],
        |inp| {
            let x: Vec<f64> = inp[0].iter().map(|&v| v as f64).collect();
            let mut dx = vec![0.0; 16];
            for i in 0..8 {
                dx[2 * i] = 2.0 * x[2 * i + 1];
                dx[2 * i + 1] = 2.0 * x[2 * i];
            }
            vec![dx]
        },
        "rewritten output",
    );
}

/// The shared-weight row reduction in `f64`, gated against the `f64` central difference (and the
/// closed form) at a tight tolerance — the region's own arithmetic, with no `f32` rounding in the
/// way. Its output `s` is a buffer parameter (the tape keeps `f64` gradients only for parameters),
/// differentiated too: `dL/ds = 0` because the region overwrites all of it.
#[test]
fn f64_region_matches_f64_finite_differences() {
    const SRC: &str = "@parallel
fn loss(x: [f64; 16], w: [f64; 4], @noalias mut s: [f64; 4], mut out: [f64; 1]) -> f64 {
    for r in 0..4 {
        s[r] = x[r * 4] * w[0] + x[r * 4 + 1] * w[1] + x[r * 4 + 2] * w[2] + x[r * 4 + 3] * w[3];
    }
    let l: f64 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3];
    out[0] = l;
    return l;
}
";
    let lens = [16usize, 4, 4, 1];
    let wrt = [0usize, 1, 2];
    let inputs: Vec<Vec<f64>> = lens
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            if i >= 2 {
                vec![0.0; n]
            } else {
                values(n, i as u32 + 7).iter().map(|&v| v as f64).collect()
            }
        })
        .collect();
    let (x, w) = (&inputs[0], &inputs[1]);
    let (mut dx, mut dw) = (vec![0.0; 16], vec![0.0; 4]);
    for r in 0..4 {
        let s: f64 = (0..4).map(|j| x[4 * r + j] * w[j]).sum();
        for j in 0..4 {
            dx[4 * r + j] = 2.0 * s * w[j];
            dw[j] += 2.0 * s * x[4 * r + j];
        }
    }
    let want = [dx, dw, vec![0.0; 4]];
    for level in 0..=3u8 {
        for grain in [false, true] {
            let (program, mut it) = compile_at(SRC, level, grain);
            let (prog, f, g) = differentiate(&program, &mut it, &wrt);
            let mut bufs = inputs.clone();
            for &wi in &wrt {
                bufs.push(vec![0.0; lens[wi]]);
            }
            run_f64(&prog, g, &mut bufs, &it);
            let grads = bufs.split_off(lens.len());
            let eps = 1e-6;
            for (gi, &wi) in wrt.iter().enumerate() {
                for j in 0..lens[wi] {
                    let mut plus = inputs.clone();
                    plus[wi][j] += eps;
                    run_f64(&prog, f, &mut plus, &it);
                    let mut minus = inputs.clone();
                    minus[wi][j] -= eps;
                    run_f64(&prog, f, &mut minus, &it);
                    let fd = (plus[3][0] - minus[3][0]) / (2.0 * eps);
                    let a = grads[gi][j];
                    let tag = format!("f64 -O{level} grain={grain} wrt#{gi}[{j}]");
                    assert!(
                        (a - fd).abs() <= 1e-7 + 1e-6 * fd.abs(),
                        "{tag}: {a} vs fd {fd}"
                    );
                    let c = want[gi][j];
                    assert!(
                        (a - c).abs() <= 1e-12 + 1e-12 * c.abs(),
                        "{tag}: {a} vs {c}"
                    );
                }
            }
        }
    }
}

/// A region reading an integer mask through a slice (`@unchecked`, so the iteration is straight-line)
/// still differentiates when what changes after it is provably elsewhere: the store through `zs`
/// lands in `z`, which the slice `ms` cannot address. `h[i] = x[i]·m[i]`, `L = Σ h[i]²`:
/// `dL/dx[i] = 2·x[i]·m[i]²` with `m = [1, 0, 1, 1, 0, 1, 0, 1]`.
#[test]
fn slice_mask_region_with_an_unrelated_later_write_differentiates() {
    const SRC: &str = "@parallel @unchecked
fn loss(x: [f32; 8], mut out: [f32; 1]) -> f32 {
    let mk: [i32; 8] = [1, 0, 1, 1, 0, 1, 0, 1];
    let ms: []i32 = mk;
    let mut h: [f32; 8] = [0.0; 8];
    for i in 0..8 {
        h[i] = x[i] * (ms[i] as f32);
    }
    let mut z: [f32; 2] = [0.0; 2];
    let mut zs: []f32 = z;
    let l: f32 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2] + h[3] * h[3] + h[4] * h[4] + h[5] * h[5] + h[6] * h[6] + h[7] * h[7];
    zs[0] = l;
    out[0] = l;
    return l;
}
";
    gate(
        SRC,
        &[8, 1],
        &[0],
        |inp| {
            let m = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
            let dx = (0..8)
                .map(|i| 2.0 * inp[0][i] as f64 * m[i] * m[i])
                .collect();
            vec![dx]
        },
        "slice mask",
    );
}

/// Every refusal names the region and the reason, instead of a silently wrong gradient.
#[test]
fn unsupported_regions_decline_by_name() {
    let cases: [(&str, &str); 8] = [
        // In place: the iteration reads what it writes, which the forward has already overwritten.
        (
            "@parallel
fn loss(x: [f32; 16], mut out: [f32; 1]) -> f32 {
    let mut h: [f32; 8] = [1.0; 8];
    for i in 0..8 {
        h[i] = h[i] * x[2 * i];
    }
    let mut l: f32 = 0.0;
    for i in 0..8 {
        l = l + h[i];
    }
    out[0] = l;
    return l;
}
",
            "both reads and writes",
        ),
        // An inner loop puts control flow in the iteration.
        (
            "@parallel
fn loss(x: [f32; 32], w: [f32; 8], mut out: [f32; 1]) -> f32 {
    let mut s: [f32; 4] = [0.0; 4];
    for r in 0..4 {
        let mut acc: f32 = 0.0;
        for j in 0..8 {
            acc = acc + x[r * 8 + j];
        }
        s[r] = acc * w[r];
    }
    let mut l: f32 = 0.0;
    for i in 0..4 {
        l = l + s[i];
    }
    out[0] = l;
    return l;
}
",
            "not one straight-line block",
        ),
        // A buffer the region reads changes after it: the recompute would read the new values.
        (
            "@parallel
fn loss(x: [f32; 16], mut out: [f32; 1]) -> f32 {
    let mut a: [f32; 16] = x;
    let mut h: [f32; 8] = [0.0; 8];
    for i in 0..8 {
        h[i] = a[2 * i] * a[2 * i + 1];
    }
    a[3] = 0.0;
    let mut l: f32 = 0.0;
    for i in 0..8 {
        l = l + h[i];
    }
    let r: f32 = l + a[3];
    out[0] = r;
    return r;
}
",
            "written again after it",
        ),
        // A runtime call after the region redraws a buffer it reads (dropout masks drawn into one
        // buffer): `wukong_rand_fill_f32` is no recognized kernel, but it writes its operand.
        (
            "@parallel
fn loss(x: [f32; 16], w: [f32; 4], mut out: [f32; 1]) -> f32 {
    let mut m: [f32; 4] = [0.0; 4];
    rand_fill_f32(m, 1, 0);
    let mut s: [f32; 4] = [0.0; 4];
    for r in 0..4 {
        s[r] = x[r * 4] * w[0] * m[0] + x[r * 4 + 1] * w[1] * m[1] + x[r * 4 + 2] * w[2] * m[2] + x[r * 4 + 3] * w[3] * m[3];
    }
    rand_fill_f32(m, 2, 0);
    let mut l: f32 = 0.0;
    for r in 0..4 {
        l = l + s[r] * s[r];
    }
    out[0] = l;
    return l;
}
",
            "written again after it",
        ),
        // A value-returning file read after the region overwrites a buffer it reads.
        (
            "@parallel
fn loss(x: [f32; 16], w: [f32; 4], mut out: [f32; 1]) -> f32 {
    let mut m: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut s: [f32; 4] = [0.0; 4];
    for r in 0..4 {
        s[r] = x[r * 4] * w[0] * m[0] + x[r * 4 + 1] * w[1] * m[1] + x[r * 4 + 2] * w[2] * m[2] + x[r * 4 + 3] * w[3] * m[3];
    }
    let n: i64 = read_f32(\"autodiff_region_absent_zz.bin\", m);
    let mut l: f32 = 0.0;
    for r in 0..4 {
        l = l + s[r] * s[r];
    }
    out[0] = l + (n as f32) * 0.0;
    return l;
}
",
            "written again after it",
        ),
        // A store through a slice view of a buffer the region reads, after it.
        (
            "@parallel @unchecked
fn loss(x: [f32; 16], w: [f32; 4], mut out: [f32; 1]) -> f32 {
    let mut m: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut ms: []f32 = m;
    let mut s: [f32; 4] = [0.0; 4];
    for r in 0..4 {
        s[r] = x[r * 4] * w[0] * m[0] + x[r * 4 + 1] * w[1] * m[1] + x[r * 4 + 2] * w[2] * m[2] + x[r * 4 + 3] * w[3] * m[3];
    }
    ms[1] = 7.0;
    let mut l: f32 = 0.0;
    for r in 0..4 {
        l = l + s[r] * s[r];
    }
    out[0] = l;
    return l;
}
",
            "written again after it",
        ),
        // The region reads a mask through a slice, and the array behind the slice changes after it.
        (
            "@parallel @unchecked
fn loss(x: [f32; 8], mut out: [f32; 1]) -> f32 {
    let mut mk: [i32; 8] = [1, 0, 1, 1, 0, 1, 0, 1];
    let ms: []i32 = mk;
    let mut h: [f32; 8] = [0.0; 8];
    for i in 0..8 {
        h[i] = x[i] * (ms[i] as f32);
    }
    mk[2] = 0;
    let l: f32 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2] + h[3] * h[3] + h[4] * h[4] + h[5] * h[5] + h[6] * h[6] + h[7] * h[7];
    out[0] = l + (mk[2] as f32);
    return l;
}
",
            "written again after it",
        ),
        // Under `@unchecked` a slice store is straight-line, but a slice's elements have no
        // gradient the loss can key: refused by name, not dropped as a dead output.
        (
            "@parallel @unchecked
fn loss(w: [f32; 4], mut out: [f32; 1]) -> f32 {
    let t: f32 = w[0] * w[1];
    let mut h: [f32; 8] = [0.0; 8];
    let mut hs: []f32 = h;
    for i in 0..8 {
        hs[i] = t * (i as f32);
    }
    let l: f32 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2] + h[3] * h[3] + h[4] * h[4] + h[5] * h[5] + h[6] * h[6] + h[7] * h[7];
    out[0] = l;
    return l;
}
",
            "writes through captured slice",
        ),
    ];
    for (src, why) in cases {
        for level in [1u8, 2] {
            let (program, mut it) = compile_at(src, level, true);
            let err = build_grad_full(&program, &mut it, &grad_options(&[0]))
                .err()
                .unwrap_or_else(|| panic!("-O{level}: must decline ({why})"));
            assert!(
                err.contains("@parallel region") && err.contains(why),
                "-O{level}: the decline must name the region and say `{why}`, got: {err}"
            );
        }
    }
}

/// `--train` through a region on both backends: the same weights, bit for bit, after every step
/// (the privatized shared adjoint is summed in one chunk order on the interpreter and on the native
/// pool), at every `-O` level.
#[test]
fn training_through_a_region_is_bit_identical_on_both_backends() {
    const SRC: &str = "@parallel
fn loss(w: [f32; 4], x: [f32; 16], mut out: [f32; 1]) -> f32 {
    let mut s: [f32; 4] = [0.0; 4];
    for r in 0..4 {
        s[r] = x[r * 4] * w[0] + x[r * 4 + 1] * w[1] + x[r * 4 + 2] * w[2] + x[r * 4 + 3] * w[3];
    }
    let mut l: f32 = 0.0;
    for r in 0..4 {
        l = l + s[r] * s[r];
    }
    out[0] = l;
    return l;
}
";
    let mut reference: Option<(Vec<u32>, Vec<Vec<u32>>)> = None;
    for level in 0..=3u8 {
        let (program, mut it) = compile_at(SRC, level, true);
        for backend in [BackendKind::Interp, BackendKind::Native] {
            let mut opts = Options {
                opt_level: level,
                backend,
                ..Options::default()
            };
            opts.grad = GradOptions {
                of: Some("loss".to_string()),
                train: true,
                wrt: vec![0],
                train_steps: 5,
                train_lr: 0.05,
                train_seed: 11,
                ..GradOptions::default()
            };
            let (trajectory, buffers) =
                train_loop_with_buffers(&program, &mut it, &opts, &[4, 16, 1])
                    .unwrap_or_else(|e| panic!("-O{level} {backend:?}: {e}"));
            assert!(
                trajectory.last() < trajectory.first(),
                "-O{level} {backend:?}: the loss must fall: {trajectory:?}"
            );
            let bits = (
                trajectory.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                buffers
                    .iter()
                    .map(|b| b.iter().map(|x| x.to_bits()).collect())
                    .collect::<Vec<_>>(),
            );
            match &reference {
                None => reference = Some(bits),
                Some(r) => assert_eq!(r, &bits, "-O{level} {backend:?} differs"),
            }
        }
    }
}

/// More iterations than private copies: 37 rows against the shared `w`, so the region backward runs
/// 16 chunks, five of three rows then eleven of two — the closed form holds only if every
/// row lands in exactly one chunk. Same math as the four-row test.
#[test]
fn shared_weight_over_more_rows_than_private_copies() {
    const SRC: &str = "@parallel
fn loss(x: [f32; 148], w: [f32; 4], mut out: [f32; 1]) -> f32 {
    let mut s: [f32; 37] = [0.0; 37];
    for r in 0..37 {
        s[r] = x[r * 4] * w[0] + x[r * 4 + 1] * w[1] + x[r * 4 + 2] * w[2] + x[r * 4 + 3] * w[3];
    }
    let mut l: f32 = 0.0;
    for r in 0..37 {
        l = l + s[r] * s[r];
    }
    out[0] = l;
    return l;
}
";
    gate(
        SRC,
        &[148, 4, 1],
        &[0, 1],
        |inp| {
            let x: Vec<f64> = inp[0].iter().map(|&v| v as f64).collect();
            let w: Vec<f64> = inp[1].iter().map(|&v| v as f64).collect();
            let (mut dx, mut dw) = (vec![0.0; 148], vec![0.0; 4]);
            for r in 0..37 {
                let s: f64 = (0..4).map(|j| x[4 * r + j] * w[j]).sum();
                for j in 0..4 {
                    dx[4 * r + j] = 2.0 * s * w[j];
                    dw[j] += 2.0 * s * x[4 * r + j];
                }
            }
            vec![dx, dw]
        },
        "37 rows",
    );
}

/// The recognizer ↔ tape coupling inside a region: the outlined body dispatches the SERIAL kernel
/// (the region supplies the threading), so the iteration backward differentiates that spelling and
/// emits serial backward kernels — no `_parallel` call nested inside a parallel region — while the
/// loss's own `@parallel` kernels keep their `_parallel` backward.
#[test]
fn an_iteration_backward_rides_the_serial_kernels() {
    const SRC: &str = "@parallel
fn loss(x: [f32; 8], w: [f32; 8], mut out: [f32; 1]) -> f32 {
    let mut h: [f32; 8] = [0.0; 8];
    for i in 0..8 {
        let t: f32 = dot(x, w);
        h[i] = x[i] * t;
    }
    let mut l: f32 = 0.0;
    for i in 0..8 {
        l = l + h[i] * h[i];
    }
    out[0] = l;
    return l;
}
";
    for level in 0..=3u8 {
        let (program, mut it) = compile_at(SRC, level, true);
        let built = build_grad_full(&program, &mut it, &grad_options(&[0, 1]))
            .unwrap_or_else(|e| panic!("-O{level}: {e}"));
        let calls = |f: &wukong_mir::Function| -> Vec<String> {
            f.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter_map(|i| match &i.op {
                    wukong_mir::Op::Call { func, .. } => Some(it.resolve(*func).to_string()),
                    _ => None,
                })
                .collect()
        };
        let iteration = built
            .support
            .iter()
            .find(|f| it.resolve(f.name) == "loss_grad$par0$it")
            .unwrap_or_else(|| panic!("-O{level}: no iteration backward"));
        let inner = calls(iteration);
        assert!(
            inner.iter().any(|c| c == "wukong_sreduce_f32"),
            "-O{level}: the iteration must replay the serial dot: {inner:?}"
        );
        assert!(
            inner.iter().any(|c| c == "wukong_velem_f32"),
            "-O{level}: the dot's backward is a serial velem scale: {inner:?}"
        );
        assert!(
            !inner.iter().any(|c| c.ends_with("_parallel")),
            "-O{level}: no parallel kernel inside the region backward: {inner:?}"
        );
        assert!(
            calls(&built.grad)
                .iter()
                .any(|c| c == "wukong_sreduce_f32_parallel"),
            "-O{level}: the loss's own reduction keeps its @parallel spelling"
        );
    }
}
