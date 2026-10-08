//! The two-layer character-level language model of `tests/run/train_tiny_lm.wk`, trained by the real
//! binary (N83 / N82): an embedding gather, a tanh hidden layer, an output projection and the softmax
//! cross-entropy, differentiated through the tape's embedding, GEMM, tanh and cross-entropy rules.
//!
//! The corpus harness only checks that the fixture's `--train` run exits 0. This test checks what it
//! computes: the loss trajectory must match an f64 reference of the same model — the same seeded
//! weights (`--train-seed=11`, uniform in [-0.4, 0.4)), the exact gradient, three SGD updates at
//! lr = 1/2 — and decrease strictly, byte-identically on the interpreter and the native JIT at -O0
//! and -O2. The reference, computed in Python/numpy in f64 from the model's definition, is
//! 12.406603, 11.984528, 10.952965, 8.654537; the runtime's f32 exp/tanh/GEMM kernels stay within
//! ~1e-5 of it, so a 1e-3 band fails any wrong VJP (a dropped term moves the second loss by > 0.1).
use std::process::Command;

const REFERENCE: [f32; 4] = [12.406603, 11.984528, 10.952965, 8.654537];

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/run/train_tiny_lm.wk")
}

fn trajectory(stdout: &str) -> Vec<f32> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let value = rest.split("loss ").nth(1).expect("malformed step line");
            value.trim().parse::<f32>().expect("loss is a float")
        })
        .collect()
}

#[test]
fn tiny_lm_trains_and_matches_the_f64_reference_on_both_backends() {
    let mut baseline: Option<Vec<u8>> = None;
    for level in ["-O0", "-O2"] {
        for backend in ["--backend=interp", "--backend=native"] {
            let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
                .args([
                    "--train",
                    "--grad-wrt=0,1,2",
                    "--train-steps=3",
                    "--train-lr=0.5",
                    "--train-seed=11",
                    level,
                    backend,
                ])
                .arg(fixture())
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(out.status.success(), "{level} {backend}: {stderr}");
            let text = String::from_utf8_lossy(&out.stdout);
            let losses = trajectory(&text);
            assert_eq!(losses.len(), 4, "{level} {backend}: {text}");
            for (step, (&got, &want)) in losses.iter().zip(&REFERENCE).enumerate() {
                assert!(
                    (got - want).abs() < 1e-3,
                    "{level} {backend}: step {step} loss {got}, reference {want}\n{text}"
                );
            }
            assert!(
                losses.windows(2).all(|w| w[1] < w[0]),
                "{level} {backend}: the loss must decrease strictly\n{text}"
            );
            match &baseline {
                Some(b) => assert_eq!(&out.stdout, b, "{level} {backend} differs from -O0 interp"),
                None => baseline = Some(out.stdout),
            }
        }
    }
}
