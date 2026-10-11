//! N83 through the real `wukongc` binary: a loss that calls the `sdpa` builtin is differentiable.
//!
//! `sdpa` used to lower to run-time extent checks plus a scratch-copy overlap guard, so a loss
//! calling it was multi-block and `wukong_autodiff` refused it ("`loss` has 16 blocks") although
//! `tape_attn.rs` carries the fused attention VJP. With fixed-array operands and every input proved
//! apart from the output (here by `@noalias`), the lowering now emits one bare
//! `wukong_attention_f32` (`wukong_mir_build::builtins`) — here with constant `s`/`d`, so not even
//! an extent `assert` precedes it — and this file pins what that buys at the CLI:
//!
//! * `--emit=grad` succeeds at every `-O` level, and the backward replays the forward attention
//!   and then rides the FlashAttention recompute — every kernel of `tape_attn.rs`'s table. The
//!   routing is gate-blind (both backends call the same symbols), so it is pinned on the printed
//!   MIR;
//! * `--train` runs on the interpreter AND the native JIT, byte-identical at `-O0` and `-O2` — the
//!   synthesized backward's only differential peer — and the loss descends.
//!
//! Gradient VALUES are gated in-crate against the f64 central finite difference of the same
//! source loss (`wukong_driver::grad_model_tests::sdpa_loss_from_source_matches_f64_finite_differences`,
//! and `..._with_runtime_dims_...` for `s`/`d` read at run time) and the f64 closed form on
//! hand-built MIR (`wukong_autodiff`'s `tests/attention.rs`).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A per-test scratch directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!(
            "wukong-autodiff-attention-{tag}-{}",
            std::process::id()
        ));
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

/// One head of attention, `s = 4`, `d = 8`, `scale = 1/√8`, and `L = Σ o²` (a `@parallel`
/// function, so the reduction dispatches the reduction kernel the tape differentiates). `CAUSAL`
/// is replaced by the flag under test.
const SDPA_LOSS: &str = "\
@parallel
fn loss(@noalias q: [f32; 32], @noalias k: [f32; 32], @noalias v: [f32; 32], mut o: [f32; 32],
        mut out: [f32; 1]) -> f32 {
    sdpa(q, k, v, o, 4, 8, 0.35355339, CAUSAL);
    let mut l: f32 = 0.0;
    for i in 0..32 { l = l + o[i] * o[i]; }
    out[0] = l;
    return l;
}
fn main() -> i32 { return 0; }
";

/// The two spellings of the source, one per causal flag.
fn sources(scratch: &Scratch) -> Vec<(u8, PathBuf)> {
    [0u8, 1]
        .iter()
        .map(|&c| {
            let text = SDPA_LOSS.replace("CAUSAL", &c.to_string());
            (c, scratch.source(&format!("sdpa_loss_causal{c}.wk"), &text))
        })
        .collect()
}

/// The printed backward function (everything from `fn loss_grad(` on).
fn backward(mir: &str) -> &str {
    let at = mir
        .find("fn loss_grad(")
        .unwrap_or_else(|| panic!("no generated backward in:\n{mir}"));
    &mir[at..]
}

/// `--emit=grad` differentiates the `sdpa` loss at every level: the backward replays the forward
/// attention and then rides every kernel of the recompute table in `tape_attn.rs` (`P = scale·Q·Kᵀ`
/// on the α-scaled GEMM, the in-place softmax, `dV = Pᵀ·dO` and `dK = dSᵀ·Q` on the TN GEMM,
/// `dP = dO·Vᵀ` on the NT GEMM, the fused softmax backward, the `scale·dS` velem pass and
/// `dQ = dS·K` on the NN GEMM). Each requirement lists its serial and `@parallel` spellings.
#[test]
fn grad_emission_of_an_sdpa_loss_rides_the_attention_recompute() {
    let scratch = Scratch::new("emit");
    let requirements: &[&[&str]] = &[
        &["call wukong_attention_f32("],
        &[
            "call wukong_sgemm_nt_alpha(",
            "call wukong_sgemm_nt_alpha_parallel(",
        ],
        &["call wukong_norm_f32(", "call wukong_norm_f32_parallel("],
        &["call wukong_sgemm_tn(", "call wukong_sgemm_tn_parallel("],
        &["call wukong_sgemm_nt(", "call wukong_sgemm_nt_parallel("],
        &[
            "call wukong_softmax_bwd_f32(",
            "call wukong_softmax_bwd_f32_parallel(",
        ],
        &["call wukong_velem_f32(", "call wukong_velem_f32_parallel("],
        &["call wukong_sgemm(", "call wukong_sgemm_parallel("],
    ];
    for (causal, path) in sources(&scratch) {
        for level in ["-O0", "-O1", "-O2", "-O3"] {
            let out = wukongc(&["--emit=grad", level, "--grad-wrt=0,1,2"], &path);
            assert_ok(&out, &format!("causal={causal} {level}"));
            let mir = stdout(&out);
            let bwd = backward(&mir);
            for &spellings in requirements {
                assert!(
                    spellings.iter().any(|s| bwd.contains(*s)),
                    "causal={causal} {level}: the backward rides none of {spellings:?}:\n{bwd}"
                );
            }
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

/// `--train --train-steps=3` of the `sdpa` loss in `(q, k, v)`: the interpreter and the native JIT
/// print byte-identical trajectories at `-O0` and `-O2`, and SGD at the default small step descends
/// strictly (`L = Σ o²` is smooth in every input, and `-∇L` is a descent direction).
#[test]
fn training_an_sdpa_loss_agrees_between_interpreter_and_native() {
    let scratch = Scratch::new("train");
    for (causal, path) in sources(&scratch) {
        let mut baseline: Option<String> = None;
        for level in ["-O0", "-O2"] {
            for backend in ["--backend=interp", "--backend=native"] {
                let args = [
                    "--train",
                    "--grad-wrt=0,1,2",
                    "--train-steps=3",
                    "--train-seed=7",
                    level,
                    backend,
                ];
                let out = wukongc(&args, &path);
                assert_ok(&out, &format!("causal={causal} {level} {backend}"));
                let text = stdout(&out);
                let losses = trajectory(&text);
                assert_eq!(losses.len(), 4, "causal={causal} {level} {backend}: {text}");
                assert!(
                    losses.iter().all(|l| l.is_finite()),
                    "causal={causal} {level} {backend}: non-finite loss: {text}"
                );
                assert!(
                    losses.windows(2).all(|p| p[1] < p[0]),
                    "causal={causal} {level} {backend}: the loss must descend: {text}"
                );
                match &baseline {
                    Some(b) => assert_eq!(
                        &text, b,
                        "causal={causal} {level} {backend} differs from -O0 interp"
                    ),
                    None => baseline = Some(text),
                }
            }
        }
    }
}
