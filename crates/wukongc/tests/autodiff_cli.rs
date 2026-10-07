//! The autodiff surface end to end through the real `wukongc` binary (P4.3 / P4.6 / P4.7):
//!
//! * `--emit=grad` over source models whose backward must ride the tuned runtime kernels (the GEMM
//!   adjoints, the bias column sum, the fused cross-entropy backward) at every `-O` level — the
//!   routing is gate-blind, so it is pinned on the printed MIR;
//! * a loss written with nested helper functions differentiates at every `-O` level, `-O0`/`-O1`
//!   included, where no inliner runs;
//! * `--train` of a kernel-backed model on the interpreter AND the native JIT, required to agree byte
//!   for byte at every level — the only differential the synthesized backward has — and to descend;
//! * `--emit=obj --grad-of` exports the gradient as a linkable symbol.
//!
//! Gradient VALUES are gated in-crate (finite difference + closed forms in `wukong_autodiff` and
//! `wukong_driver::grad_model_tests`) and by the hand-derived `tests/run/autodiff_train_*.wk`
//! trajectories; this file pins the CLI contract.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A per-test scratch directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir =
            std::env::temp_dir().join(format!("wukong-autodiff-cli-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    fn source(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, text).expect("write source");
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn wukongc(args: &[&str], source: &Path) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    // Autodiff differentiates the recognized kernels; the instrument would make every model refuse.
    cmd.env_remove(wukong_driver::NO_RECOGNIZE_ENV);
    cmd.args(args).arg(source).output().expect("spawn wukongc")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn assert_ok(out: &Output, what: &str) {
    assert!(
        out.status.success(),
        "{what}: exit {:?}\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `h = relu(x·Wᵀ + bias)`, `L = Σ (h - t)²` — the `nn.Linear` + bias + ReLU layer, which the
/// recognizers fuse into one GEMM epilogue and which, spelled WITH the bias, used to be
/// undifferentiable.
const LINEAR_BIAS_RELU: &str = "\
@parallel
fn loss(x: [f32; 4], w: [f32; 4], bias: [f32; 2], t: [f32; 4], mut out: [f32; 1]) -> f32 {
    let mut h: [f32; 4] = [0.0; 4];
    for i in 0..2 { for j in 0..2 { let mut s: f32 = 0.0;
        for k in 0..2 { s = s + x[i * 2 + k] * w[j * 2 + k]; } h[i * 2 + j] = s; } }
    for i in 0..2 { for j in 0..2 { h[i * 2 + j] = fmax(h[i * 2 + j] + bias[j], 0.0); } }
    let mut l: f32 = 0.0;
    for i in 0..4 { l = l + (h[i] - t[i]) * (h[i] - t[i]); }
    out[0] = l;
    return l;
}
fn main() -> i32 { return 0; }
";

/// The same layer without the activation: least squares in `(W, bias)`, so gradient descent with a
/// small step descends strictly.
const LINEAR_BIAS: &str = "\
@parallel
fn loss(x: [f32; 4], w: [f32; 4], bias: [f32; 2], t: [f32; 4], mut out: [f32; 1]) -> f32 {
    let mut h: [f32; 4] = [0.0; 4];
    for i in 0..2 { for j in 0..2 { let mut s: f32 = 0.0;
        for k in 0..2 { s = s + x[i * 2 + k] * w[j * 2 + k]; } h[i * 2 + j] = s; } }
    for i in 0..2 { for j in 0..2 { h[i * 2 + j] = h[i * 2 + j] + bias[j]; } }
    let mut l: f32 = 0.0;
    for i in 0..4 { l = l + (h[i] - t[i]) * (h[i] - t[i]); }
    out[0] = l;
    return l;
}
fn main() -> i32 { return 0; }
";

/// The softmax cross-entropy head (`wukong_xent_fwd_f32`), its per-row losses read back as scalars.
const CROSS_ENTROPY: &str = "\
@parallel
fn loss(x: [f32; 8], target: [i32; 2], mut out: [f32; 1]) -> f32 {
    let mut lv: [f32; 2] = [0.0; 2];
    for r in 0..2 {
        let mut m: f32 = x[r * 4];
        for i in 0..4 { m = fmax(m, x[r * 4 + i]); }
        let mut s: f32 = 0.0;
        for i in 0..4 { s = s + exp(x[r * 4 + i] - m); }
        lv[r] = m + log(s) - x[r * 4 + target[r]];
    }
    let l: f32 = lv[0] + lv[1];
    out[0] = l;
    return l;
}
fn main() -> i32 { return 0; }
";

/// A loss written as nested helper functions.
const HELPERS: &str = "\
fn sq(a: f32) -> f32 { return a * a; }
fn norm2(a: f32, b: f32) -> f32 { return sq(a) + sq(b); }
fn loss(w: [f32; 2], mut out: [f32; 1]) -> f32 {
    let l: f32 = norm2(w[0], w[1]);
    out[0] = l;
    return l;
}
fn main() -> i32 { return 0; }
";

/// The printed backward function (everything from `fn loss_grad(` on).
fn backward(mir: &str) -> &str {
    let at = mir
        .find("fn loss_grad(")
        .unwrap_or_else(|| panic!("no generated backward in:\n{mir}"));
    &mir[at..]
}

/// P4.2 / P4.10 / P12.2: the backward of each model rides the tuned runtime kernels, at every level.
#[test]
fn grad_emission_rides_the_tuned_backward_kernels() {
    let scratch = Scratch::new("emit");
    let cases: [(&str, &str, &[&str]); 2] = [
        (
            LINEAR_BIAS_RELU,
            "--grad-wrt=0,1,2",
            // dX on the NN GEMM, dW on the weight-gradient TN GEMM, dbias on the column sum.
            &[
                "call wukong_sgemm_parallel(",
                "call wukong_sgemm_tn_parallel(",
                "call wukong_colsum_f32(",
            ],
        ),
        (
            CROSS_ENTROPY,
            "--grad-wrt=0",
            &["call wukong_xent_bwd_f32_parallel("],
        ),
    ];
    for (i, &(src, wrt, needles)) in cases.iter().enumerate() {
        let path = scratch.source(&format!("model{i}.wk"), src);
        for level in ["-O0", "-O1", "-O2", "-O3"] {
            let out = wukongc(&["--emit=grad", level, wrt], &path);
            assert_ok(&out, &format!("model{i} {level}"));
            let mir = stdout(&out);
            let bwd = backward(&mir);
            for &needle in needles {
                assert!(
                    bwd.contains(needle),
                    "model{i} {level}: the backward does not ride `{needle}`:\n{bwd}"
                );
            }
        }
    }
}

/// P4.6: a loss calling nested helper functions differentiates at every `-O` level. At `-O0`/`-O1`
/// no inliner runs, and the call used to be the refusal "no VJP rule for `call`"; the helpers are
/// now spliced before the reverse pass, so neither printed function calls them.
#[test]
fn helper_function_loss_differentiates_at_every_level() {
    let scratch = Scratch::new("helpers");
    let path = scratch.source("helpers.wk", HELPERS);
    for level in ["-O0", "-O1", "-O2", "-O3"] {
        let out = wukongc(&["--emit=grad", level, "--grad-wrt=0"], &path);
        assert_ok(&out, level);
        let mir = stdout(&out);
        assert!(
            mir.contains("fn loss_grad("),
            "{level}: no backward:\n{mir}"
        );
        for helper in ["call sq(", "call norm2("] {
            assert!(
                !mir.contains(helper),
                "{level}: the differentiated forward still makes `{helper}`:\n{mir}"
            );
        }
    }
}

/// The losses `--train` printed, in step order.
fn trajectory(text: &str) -> Vec<f32> {
    text.lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let value = rest
                .split("loss ")
                .nth(1)
                .unwrap_or_else(|| panic!("malformed step line: {rest}"));
            value
                .trim()
                .parse::<f32>()
                .unwrap_or_else(|e| panic!("bad loss {value:?}: {e}"))
        })
        .collect()
}

/// P4.3: training a kernel-backed model (GEMM + broadcast bias + MSE reduction) runs on the native
/// JIT as well as the interpreter, and the two agree byte for byte at every level — the whole
/// synthesized backward (heap scratch, accumulation, the kernels' adjoints) has a differential peer.
/// Least squares in `(W, bias)` with a small step, so the loss must strictly descend.
#[test]
fn training_a_kernel_model_agrees_between_interpreter_and_native() {
    let scratch = Scratch::new("train");
    let path = scratch.source("linear_bias.wk", LINEAR_BIAS);
    let args = [
        "--train",
        "--grad-wrt=1,2",
        "--train-steps=4",
        "--train-lr=0.05",
        "--train-seed=11",
    ];
    let mut baseline: Option<String> = None;
    for level in ["-O0", "-O1", "-O2", "-O3"] {
        for backend in ["--backend=interp", "--backend=native"] {
            let mut full: Vec<&str> = args.to_vec();
            full.push(level);
            full.push(backend);
            let out = wukongc(&full, &path);
            assert_ok(&out, &format!("{level} {backend}"));
            let text = stdout(&out);
            let losses = trajectory(&text);
            assert_eq!(losses.len(), 5, "{level} {backend}: {text}");
            assert!(
                losses.iter().all(|l| l.is_finite()),
                "{level} {backend}: non-finite loss: {text}"
            );
            assert!(
                losses.windows(2).all(|p| p[1] < p[0]),
                "{level} {backend}: least squares must descend strictly: {text}"
            );
            match &baseline {
                Some(b) => assert_eq!(&text, b, "{level} {backend} differs from -O0 interp"),
                None => baseline = Some(text),
            }
        }
    }
}

/// P4.3 (ahead of time): `--emit=obj --grad-of=loss` exports `loss_grad` next to `loss`, for a host
/// training loop to link; without `--grad-of` the object is unchanged.
#[test]
fn object_emission_exports_the_gradient() {
    let scratch = Scratch::new("obj");
    let path = scratch.source("linear_bias_relu.wk", LINEAR_BIAS_RELU);
    let with = scratch.0.join("with_grad.o");
    let out = wukongc(
        &[
            "--emit=obj",
            "--grad-of=loss",
            "-o",
            with.to_str().expect("utf-8 scratch path"),
        ],
        &path,
    );
    assert_ok(&out, "--emit=obj --grad-of=loss");
    let bytes = std::fs::read(&with).expect("the object was written");
    let names = |needle: &[u8]| bytes.windows(needle.len()).any(|w| w == needle);
    assert!(names(b"loss_grad"), "the object must export loss_grad");

    let without = scratch.0.join("plain.o");
    let out = wukongc(
        &[
            "--emit=obj",
            "-o",
            without.to_str().expect("utf-8 scratch path"),
        ],
        &path,
    );
    assert_ok(&out, "--emit=obj");
    let plain = std::fs::read(&without).expect("the object was written");
    assert!(
        !plain.windows(b"loss_grad".len()).any(|w| w == b"loss_grad"),
        "without --grad-of the object must not grow a gradient"
    );
}
