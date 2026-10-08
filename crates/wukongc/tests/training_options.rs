//! Exercise real CLI parsing, target retention, and AdamW decay through source compilation.
use std::process::Command;

#[test]
fn training_weight_decay_and_autodiff_targets() {
    let scratch =
        std::env::temp_dir().join(format!("wukong-training-options-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let source = scratch.join("loss.wk");
    let compiler = env!("CARGO_BIN_EXE_wukongc");
    for name in ["loss", "objective"] {
        std::fs::write(
            &source,
            format!(
                "fn {name}(w: [f32; 1], mut out: [f32; 1]) -> f32 {{ \
             let y: f32 = w[0]; out[0] = y; return y; }} \
             fn main() -> i32 {{ let w: [f32; 1] = [2.0]; \
             let mut out: [f32; 1] = [0.0]; {name}(w, out); return 0; }}"
            ),
        )
        .unwrap();
        for level in 0..=3 {
            let mut common = vec![format!("-O{level}"), "--verify-each".into()];
            if name != "loss" {
                common.push(format!("--grad-of={name}"));
            }
            let grad = Command::new(compiler)
                .args(&common)
                .arg("--emit=grad")
                .arg(&source)
                .output()
                .unwrap();
            assert!(
                grad.status.success(),
                "{}",
                String::from_utf8_lossy(&grad.stderr)
            );
            assert!(!grad.stdout.is_empty());
            for decay in [None, Some(0.0f32), Some(0.5)] {
                let mut cmd = Command::new(compiler);
                cmd.args(&common).args([
                    "--train",
                    "--train-opt=adamw",
                    "--train-steps=2",
                    "--train-lr=0.25",
                    "--train-seed=7",
                ]);
                if let Some(wd) = decay {
                    cmd.arg(format!("--train-weight-decay={wd}"));
                }
                let result = cmd.arg(&source).output().unwrap();
                assert!(
                    result.status.success(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
                let output = String::from_utf8(result.stdout).unwrap();
                let header = if decay.unwrap_or(0.0) == 0.0 {
                    "train: 2 step(s), lr=0.25, optimizer=adamw"
                } else {
                    "train: 2 step(s), lr=0.25, optimizer=adamw, weight_decay=0.5"
                };
                assert_eq!(output.lines().next(), Some(header));
                let actual: Vec<_> = output.lines().filter(|s| s.starts_with("step ")).collect();
                // Linear loss L(w)=w has gradient exactly one. Compute the two AdamW updates
                // independently in Rust, including the documented f32 rounding and moment FMA.
                let seed = 7u64
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut w = (((seed >> 40) as f32) * (1.0 / 16_777_216.0) - 0.5) * 0.8;
                let mut expected = vec![format!("step {:>4}: loss {w:.6}", 0)];
                let (mut m, mut v) = (0.0f32, 0.0f32);
                for t in 1..=2 {
                    m = 0.9f32.mul_add(m, 1.0 - 0.9f32);
                    v = 0.999f32.mul_add(v, 1.0 - 0.999f32);
                    let mhat = m / (1.0 - 0.9f64.powi(t)) as f32;
                    let vhat = v / (1.0 - 0.999f64.powi(t)) as f32;
                    w -= 0.25 * (mhat / (vhat.sqrt() + 1e-8f32) + decay.unwrap_or(0.0) * w);
                    expected.push(format!("step {t:>4}: loss {w:.6}"));
                }
                assert_eq!(actual, expected, "{name}, O{level}, decay={decay:?}");
            }
        }
    }
    for value in ["-1", "NaN", "inf", "-inf", "bogus"] {
        let result = Command::new(compiler)
            .args(["--train", "--train-opt=adamw"])
            .arg(format!("--train-weight-decay={value}"))
            .arg(&source)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("--train-weight-decay"));
    }
    let result = Command::new(compiler)
        .args([
            "--train",
            "--grad-of=objective",
            "--train-opt=sgd",
            "--train-weight-decay=0.5",
        ])
        .arg(&source)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("requires --train-opt=adamw"));
    std::fs::remove_file(&source).unwrap();
    std::fs::remove_dir(&scratch).unwrap();
}
