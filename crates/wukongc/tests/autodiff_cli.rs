//! The autodiff surface end to end through the real `wukongc` binary (P4.3 / P4.6 / P4.7):
//!
//! * `--emit=grad` over source models whose backward must ride the tuned runtime kernels (the GEMM
//!   adjoints, the bias column sum, the fused cross-entropy backward) at every `-O` level — the
//!   routing is gate-blind, so it is pinned on the printed MIR;
//! * a loss written with nested helper functions differentiates at every `-O` level, `-O0`/`-O1`
//!   included, where no inliner runs;
//! * `--train` of a kernel-backed model on the interpreter AND the native JIT, required to agree byte
//!   for byte at every level — the only differential the synthesized backward has — and to descend;
//! * `--emit=obj --grad-of` exports the gradient as a linkable symbol;
//! * a loss whose four straight-line stores into a kernel-read buffer are an SLP bundle in an
//!   ordinary `-O2` compile still differentiates and trains at every level (gradient compiles
//!   optimize the forward without SLP);
//! * a loss running a `@parallel` loop region differentiates into a parallel backward, and training
//!   through it is byte-identical on the interpreter and on the native pool at every thread count
//!   (N87).
//!
//! The training loop ahead of time (`--train --emit=exe`, byte-identical to `--train` on both
//! backends, this file's GEMM + bias model among its cases) is `train_exe.rs`.
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
/// Each requirement lists its acceptable spellings: the backward follows the forward's serial or
/// `@parallel` spelling, and which one the forward dispatches is the recognizers' business.
#[test]
fn grad_emission_rides_the_tuned_backward_kernels() {
    let scratch = Scratch::new("emit");
    let cases: [(&str, &str, &[&[&str]]); 2] = [
        (
            LINEAR_BIAS_RELU,
            "--grad-wrt=0,1,2",
            // dX on the NN GEMM, dW on the weight-gradient TN GEMM, dbias on the column sum.
            &[
                &["call wukong_sgemm(", "call wukong_sgemm_parallel("],
                &["call wukong_sgemm_tn(", "call wukong_sgemm_tn_parallel("],
                &["call wukong_colsum_f32("],
            ],
        ),
        (
            CROSS_ENTROPY,
            "--grad-wrt=0",
            &[&[
                "call wukong_xent_bwd_f32(",
                "call wukong_xent_bwd_f32_parallel(",
            ]],
        ),
    ];
    for (i, &(src, wrt, requirements)) in cases.iter().enumerate() {
        let path = scratch.source(&format!("model{i}.wk"), src);
        for level in ["-O0", "-O1", "-O2", "-O3"] {
            let out = wukongc(&["--emit=grad", level, wrt], &path);
            assert_ok(&out, &format!("model{i} {level}"));
            let mir = stdout(&out);
            let bwd = backward(&mir);
            for &spellings in requirements {
                assert!(
                    spellings.iter().any(|s| bwd.contains(*s)),
                    "model{i} {level}: the backward rides none of {spellings:?}:\n{bwd}"
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

/// Four straight-line stores into a local buffer the `sreduce` kernel then reads: `h[k] = x[k]²`
/// for `k < 4`, `L = Σ h = Σ_{k<4} x[k]²`. An ordinary `-O2` compile packs the four stores into one
/// `<4 x f32>` store (SLP); `wukong_autodiff` has no VJP rule for a vector store into a buffer whose
/// gradient is live, so a gradient compile must not see that bundle.
const LANE_STORES_LOCAL: &str = "@parallel
fn loss(x: [f32; 8], mut out: [f32; 1]) -> f32 {
    let mut h: [f32; 8] = [0.0; 8];
    h[0] = x[0] * x[0];
    h[1] = x[1] * x[1];
    h[2] = x[2] * x[2];
    h[3] = x[3] * x[3];
    let mut l: f32 = 0.0;
    for i in 0..8 { l = l + h[i]; }
    out[0] = l;
    return l;
}
fn main() -> i32 { return 0; }
";

/// The same bundle into a `@noalias` parameter, differentiated under the default `--grad-wrt` (every
/// buffer, `h` included): it packs with no dead-store elimination involved.
const LANE_STORES_PARAM: &str = "@parallel
fn loss(x: [f32; 8], @noalias mut h: [f32; 8], mut out: [f32; 1]) -> f32 {
    h[0] = x[0] * x[0];
    h[1] = x[1] * x[1];
    h[2] = x[2] * x[2];
    h[3] = x[3] * x[3];
    let mut l: f32 = 0.0;
    for i in 0..8 { l = l + h[i]; }
    out[0] = l;
    return l;
}
fn main() -> i32 { return 0; }
";

/// A forward whose straight-line stores SLP packs in an ordinary `-O2` compile differentiates at
/// every level, on every gradient surface: `--emit=grad`, `--train` (interpreter and native JIT)
/// and `--emit=obj --grad-of`. At `-O2`/`-O3` these used to exit 1 with "store into buffer vN has
/// no VJP rule, but that buffer's gradient is live", because the optimized forward autodiff takes
/// held a `<4 x f32>` store.
#[test]
fn slp_shaped_lane_stores_differentiate_at_every_level() {
    let scratch = Scratch::new("slp");
    let local = scratch.source("lane_stores_local.wk", LANE_STORES_LOCAL);
    let param = scratch.source("lane_stores_param.wk", LANE_STORES_PARAM);
    // Not vacuous: an ordinary -O2 compile of each forward does pack the bundle.
    for path in [&local, &param] {
        let out = wukongc(&["--emit=mir", "-O2"], path);
        assert_ok(&out, "--emit=mir -O2");
        assert!(
            stdout(&out).contains("load <4 x f32>"),
            "{}: SLP no longer packs this shape, so this test proves nothing:
{}",
            path.display(),
            stdout(&out)
        );
    }
    for (path, wrt) in [(&local, Some("--grad-wrt=0")), (&param, None)] {
        for level in ["-O0", "-O1", "-O2", "-O3"] {
            let mut args = vec!["--emit=grad", level];
            args.extend(wrt);
            let out = wukongc(&args, path);
            assert_ok(&out, &format!("{} {level}", path.display()));
            backward(&stdout(&out));
        }
    }
    // Training descends, identically on both backends and at every level. SGD at lr = 0.25 on
    // L = Σ_{k<4} x[k]² sets x ← x - 0.25·2x = x/2, exact in binary floating point, so each step's
    // loss is exactly a quarter of the last (up to the 6-decimal print).
    let mut baseline: Option<String> = None;
    for level in ["-O0", "-O1", "-O2", "-O3"] {
        for backend in ["--backend=interp", "--backend=native"] {
            let args = [
                "--train",
                "--grad-wrt=0",
                "--train-steps=3",
                "--train-lr=0.25",
                "--train-seed=7",
                level,
                backend,
            ];
            let out = wukongc(&args, &local);
            assert_ok(&out, &format!("--train {level} {backend}"));
            let text = stdout(&out);
            let losses = trajectory(&text);
            assert_eq!(losses.len(), 4, "{level} {backend}: {text}");
            assert!(
                losses[0] > 1e-3,
                "{level} {backend}: degenerate start: {text}"
            );
            for pair in losses.windows(2) {
                assert!(
                    (pair[1] - pair[0] / 4.0).abs() <= 1e-6,
                    "{level} {backend}: a step must quarter the loss: {text}"
                );
            }
            match &baseline {
                Some(b) => assert_eq!(&text, b, "{level} {backend} differs from -O0 interp"),
                None => baseline = Some(text),
            }
        }
    }
    let obj = scratch.0.join("lane_stores.o");
    let out = wukongc(
        &[
            "--emit=obj",
            "-O2",
            "--grad-of=loss",
            "-o",
            obj.to_str().expect("utf-8 scratch path"),
        ],
        &local,
    );
    assert_ok(&out, "--emit=obj -O2 --grad-of=loss");
    let bytes = std::fs::read(&obj).expect("the object was written");
    assert!(
        bytes.windows(b"loss_grad".len()).any(|w| w == b"loss_grad"),
        "the object must export loss_grad"
    );
}

/// N87: a row reduction against a SHARED weight vector inside a `@parallel` loop region, over 40
/// rows — more than the 16 private adjoint copies the region backward keeps, so its chunks run
/// several rows each. `s[r] = Σ_j x[4r+j]·w[j]`, `L = Σ s[r]²`; `w` comes first so `--train`
/// trains it and draws `x` as data.
const REGION_ROWS: &str = "@parallel
fn loss(w: [f32; 4], x: [f32; 160], mut out: [f32; 1]) -> f32 {
    let mut s: [f32; 40] = [0.0; 40];
    for r in 0..40 {
        s[r] = x[r * 4] * w[0] + x[r * 4 + 1] * w[1] + x[r * 4 + 2] * w[2] + x[r * 4 + 3] * w[3];
    }
    let mut l: f32 = 0.0;
    for r in 0..40 {
        l = l + s[r] * s[r];
    }
    out[0] = l;
    return l;
}
fn main() -> i32 { return 0; }
";

/// N87: a loss running a `@parallel` region differentiates at every level, and its backward is a
/// region too — the gradient hands `{loss}_grad$par0` to `wukong_parallel_for`, which runs one
/// iteration's backward (`{loss}_grad$par0$it`) per index.
#[test]
fn a_parallel_region_differentiates_into_a_parallel_backward() {
    let scratch = Scratch::new("region-grad");
    let path = scratch.source("region_rows.wk", REGION_ROWS);
    for level in ["-O0", "-O1", "-O2", "-O3"] {
        let out = wukongc(&["--emit=grad", "--grad-wrt=0,1", level], &path);
        assert_ok(&out, &format!("--emit=grad {level}"));
        let mir = stdout(&out);
        let grad = backward(&mir);
        let body = &grad[..grad.find("\n}\n").expect("the backward ends")];
        assert!(
            body.matches("call wukong_parallel_for(").count() == 2,
            "{level}: the backward must replay the forward region and run the region backward:\n{body}"
        );
        assert!(
            body.contains("func_addr loss_grad$par0"),
            "{level}: the backward must hand the chunk driver to wukong_parallel_for:\n{body}"
        );
        for f in [
            "fn loss_grad$par0(",
            "fn loss_grad$par0$it(",
            "fn wukong$par$0(",
        ] {
            assert!(mir.contains(f), "{level}: `{f}` missing from:\n{mir}");
        }
    }
}

/// N87: the region backward accumulates the shared weight's adjoint in private per-chunk copies and
/// adds them in chunk order, with chunks that depend on the trip count alone — so `--train` gives
/// the same trajectory AND the same final weights, byte for byte, on the interpreter (one sequential
/// pass) and on the native pool at every thread count, at every level.
#[test]
fn training_through_a_region_is_bit_identical_across_backends_and_thread_counts() {
    let scratch = Scratch::new("region-train");
    let path = scratch.source("region_rows.wk", REGION_ROWS);
    let mut baseline: Option<(String, Vec<u8>)> = None;
    let mut runs = 0;
    for level in ["-O0", "-O2"] {
        for (backend, threads) in [
            ("--backend=interp", "--threads=1"),
            ("--backend=native", "--threads=1"),
            ("--backend=native", "--threads=2"),
            ("--backend=native", "--threads=7"),
            ("--backend=native", "--threads=16"),
        ] {
            let save = scratch.0.join(format!("w-{runs}.bin"));
            runs += 1;
            let save_arg = format!("--train-save=0={}", save.display());
            let args = [
                "--train",
                "--grad-wrt=0",
                "--train-steps=4",
                "--train-lr=0.01",
                "--train-seed=5",
                save_arg.as_str(),
                level,
                backend,
                threads,
            ];
            let out = wukongc(&args, &path);
            let what = format!("{level} {backend} {threads}");
            assert_ok(&out, &what);
            let text = stdout(&out);
            let losses = trajectory(&text);
            assert_eq!(losses.len(), 5, "{what}: {text}");
            assert!(
                losses.windows(2).all(|p| p[1] < p[0]),
                "{what}: least squares must descend: {text}"
            );
            let weights = std::fs::read(&save).expect("--train-save wrote the weights");
            assert_eq!(weights.len(), 16, "{what}: four f32 weights");
            match &baseline {
                Some((t, w)) => {
                    assert_eq!(&text, t, "{what}: trajectory differs");
                    assert_eq!(&weights, w, "{what}: trained weights differ");
                }
                None => baseline = Some((text, weights)),
            }
        }
    }
}
