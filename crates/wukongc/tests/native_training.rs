//! Real CLI coverage for backend selection and the training buffer/artifact contracts.
use std::process::Command;

fn compiler() -> Command {
    Command::new(env!("CARGO_BIN_EXE_wukongc"))
}

#[test]
fn native_training_cli_matches_interpreter_at_every_level() {
    let scratch =
        std::env::temp_dir().join(format!("wukong-native-training-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let source = scratch.join("loss.wk");
    std::fs::write(
        &source,
        "fn loss(w: [f32; 1], data: [f32; 1], mut out: [f32; 1]) -> f32 { \
         let d: f32 = w[0] - data[0]; let y: f32 = d*d; out[0] = y; return y; } \
         fn main() -> i32 { return 0; }",
    )
    .unwrap();
    for level in 0..=3 {
        for verified in [false, true] {
            let mut expected = None;
            for backend in ["interp", "native"] {
                let mut cmd = compiler();
                cmd.args([
                    "--train",
                    "--run",
                    "--grad-wrt=0",
                    "--train-steps=3",
                    "--train-opt=adamw",
                    "--train-weight-decay=0.125",
                    "--train-seed=7",
                ])
                .arg(format!("--backend={backend}"))
                .arg(format!("-O{level}"));
                if verified {
                    cmd.arg("--verify-each");
                }
                let result = cmd.arg(&source).output().unwrap();
                assert!(
                    result.status.success(),
                    "{backend} O{level}: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
                assert!(
                    result.stderr.is_empty(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
                assert!(String::from_utf8_lossy(&result.stdout).contains("step    3: loss"));
                if let Some(expected) = &expected {
                    assert_eq!(&result.stdout, expected);
                } else {
                    expected = Some(result.stdout);
                }
            }
        }
    }
    std::fs::remove_dir_all(scratch).unwrap();
}

#[test]
fn training_cli_rejects_unsupported_buffers_backends_and_artifacts() {
    let scratch = std::env::temp_dir().join(format!(
        "wukong-native-training-contracts-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&scratch).unwrap();
    let source = scratch.join("loss.wk");
    for ty in ["f32", "[f64; 1]", "[i64; 1]", "[f16; 1]"] {
        std::fs::write(&source, format!("fn loss(w: {ty}, mut out: [f32; 1]) -> f32 {{ out[0] = 0.0; return 0.0; }} fn main() -> i32 {{ return 0; }}")).unwrap();
        for backend in ["interp", "native"] {
            let result = compiler()
                .args(["--train", "--train-steps=0"])
                .arg(format!("--backend={backend}"))
                .arg(&source)
                .output()
                .unwrap();
            assert!(!result.status.success(), "accepted {ty}");
            let stderr = String::from_utf8_lossy(&result.stderr);
            assert!(
                stderr.contains("parameter 0")
                    && stderr.contains("concrete f32 array or dense tensor"),
                "{ty}: {stderr}"
            );
            assert!(result.stdout.is_empty());
        }
    }
    // Nested f32 arrays keep their dense ABI and are accepted by both execution paths.
    std::fs::write(&source, "fn loss(w: [[f32; 2]; 2], mut out: [f32; 1]) -> f32 { let y: f32 = w[0][0]; out[0] = y; return y; } fn main() -> i32 { return 0; }").unwrap();
    for backend in ["interp", "native"] {
        let result = compiler()
            .args(["--train", "--train-steps=0"])
            .arg(format!("--backend={backend}"))
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    for backend in ["gpu", "gpu-native"] {
        // Negative CPU-only dispatch check: no GPU feature or device is used.
        let result = compiler()
            .args(["--train", "--train-steps=0"])
            .arg(format!("--backend={backend}"))
            .arg(&source)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr)
            .contains("supports only --backend=interp or --backend=native"));
    }
    for stage in ["tokens", "mir-high", "grad", "obj", "exe"] {
        let result = compiler()
            .args(["--train", "--backend=native"])
            .arg(format!("--emit={stage}"))
            .arg(&source)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&result.stderr).contains("training executes immediately"));
        assert!(result.stdout.is_empty());
    }
    let artifact = scratch.join("training.exe");
    let result = compiler()
        .args(["--train", "--backend=native", "-o"])
        .arg(&artifact)
        .arg(&source)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("training executes immediately"));
    assert!(!artifact.exists());
    std::fs::remove_dir_all(scratch).unwrap();
}
