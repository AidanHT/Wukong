//! N82 (stage 1) through the real CLI: `--train` reads initial weights (`--train-init`), streams
//! batches of a fixed input (`--train-data`), writes the trained weights (`--train-save`), and
//! applies the global-norm clip and the warmup / cosine schedule — byte-identically on the
//! interpreter and the native JIT. Every expectation is derived by hand from the loss's scalar
//! semantics (stated next to each one), never recorded from the compiler.
use std::path::{Path, PathBuf};
use std::process::Command;

/// `L = (w0 - x0)² + (w1 - x1)²`: `w` trained, `x` the streamed batch, `out` the loss sink.
const LOSS: &str = "\
fn loss(w: [f32; 2], x: [f32; 2], mut out: [f32; 1]) -> f32 {
    let d0: f32 = w[0] - x[0];
    let d1: f32 = w[1] - x[1];
    let l: f32 = d0 * d0 + d1 * d1;
    out[0] = l;
    return l;
}
fn main() -> i32 { return 0; }
";

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir =
            std::env::temp_dir().join(format!("wukong-training-data-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_f32(path: &Path, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn train(source: &Path, backend: &str, extra: &[String]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .args([
            "--train",
            "--grad-wrt=0",
            "--train-steps=3",
            "--train-lr=0.25",
        ])
        .arg(format!("--backend={backend}"))
        .args(extra)
        .arg(source)
        .output()
        .unwrap()
}

/// Initial `w = [1, -1]` from a file; `x` streams two batches, `[0.5, 0.5]` and `[-0.5, 0.25]`, so
/// updates 0, 1, 2 train on batches 0, 1, 0 and the final report uses batch 1. SGD at lr = 1/4 on
/// `dL/dw = 2(w - x)`, every value dyadic so f32 is exact:
/// * update 0: d = [0.5, -1.5], L = 2.5, w -> [0.75, -0.25];
/// * update 1: d = [1.25, -0.5], L = 1.8125, w -> [0.125, 0];
/// * update 2: d = [-0.375, -0.5], L = 0.390625, w -> [0.3125, 0.25];
/// * final, batch 1: d = [0.8125, 0], L = 0.66015625; 2.5 / 0.66015625 = 3.787 in f32.
#[test]
fn train_reads_init_streams_batches_and_saves_weights() {
    let s = Scratch::new("stream");
    let source = s.file("loss.wk");
    std::fs::write(&source, LOSS).unwrap();
    let init = s.file("w0.bin");
    write_f32(&init, &[1.0, -1.0]);
    let data = s.file("x.bin");
    write_f32(&data, &[0.5, 0.5, -0.5, 0.25]);
    let expected = "train: 3 step(s), lr=0.25, optimizer=sgd\n\
                    step    0: loss 2.500000\n\
                    step    1: loss 1.812500\n\
                    step    2: loss 0.390625\n\
                    step    3: loss 0.660156\n\
                    loss 2.500000 -> 0.660156  (3.79x reduction)\n";
    for backend in ["interp", "native"] {
        let saved = s.file(&format!("w_{backend}.bin"));
        let extra = [
            format!("--train-init=0={}", init.display()),
            format!("--train-data=1={}", data.display()),
            format!("--train-save=0={}", saved.display()),
        ];
        let out = train(&source, backend, &extra);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{backend}: {stderr}");
        assert_eq!(String::from_utf8_lossy(&out.stdout), expected, "{backend}");
        assert!(stderr.contains("wrote "), "{backend}: {stderr}");
        assert_eq!(read_f32(&saved), vec![0.3125, 0.25], "{backend}");
    }
}

/// The same run with `--train-clip=1 --train-warmup=1 --train-schedule=cosine`. Update 0 is the
/// warmup (lr = 1/4·1/1); updates 1 and 2 follow the cosine over the remaining two (lr = 1/4, then
/// 1/4·(1 + cos(π/2))/2 = 1/8). Every gradient's norm exceeds 1, so each is scaled by
/// `1 / (norm + 1e-6)` — computed by hand in f32 with the norm in f64, as `clip_grad_norm_` does —
/// which leaves `w = [0.6935036, -0.4950207]` (bits 0x3f318974, 0xbefd735a).
#[test]
fn train_clips_the_global_norm_and_follows_the_schedule() {
    let s = Scratch::new("clip");
    let source = s.file("loss.wk");
    std::fs::write(&source, LOSS).unwrap();
    let init = s.file("w0.bin");
    write_f32(&init, &[1.0, -1.0]);
    let data = s.file("x.bin");
    write_f32(&data, &[0.5, 0.5, -0.5, 0.25]);
    let mut outputs = Vec::new();
    for backend in ["interp", "native"] {
        let saved = s.file(&format!("w_{backend}.bin"));
        let extra = [
            format!("--train-init=0={}", init.display()),
            format!("--train-data=1={}", data.display()),
            format!("--train-save=0={}", saved.display()),
            "--train-clip=1".to_string(),
            "--train-warmup=1".to_string(),
            "--train-schedule=cosine".to_string(),
        ];
        let out = train(&source, backend, &extra);
        assert!(
            out.status.success(),
            "{backend}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        assert_eq!(
            stdout.lines().next(),
            Some("train: 3 step(s), lr=0.25, optimizer=sgd, clip=1, warmup=1, schedule=cosine"),
            "{backend}"
        );
        let bits: Vec<u32> = read_f32(&saved).iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, vec![0x3f31_8974, 0xbefd_735a], "{backend}");
        outputs.push(stdout);
    }
    assert_eq!(
        outputs[0], outputs[1],
        "interp and native trajectories differ"
    );
}

/// A data file that is not a whole number of batches is refused before training starts, naming the
/// file; so is a streamed buffer that is also a trainable weight.
#[test]
fn train_refuses_a_ragged_data_file_and_a_streamed_weight() {
    let s = Scratch::new("bad");
    let source = s.file("loss.wk");
    std::fs::write(&source, LOSS).unwrap();
    let data = s.file("x.bin");
    write_f32(&data, &[0.5, 0.5, -0.5]);
    let out = train(
        &source,
        "interp",
        &[format!("--train-data=1={}", data.display())],
    );
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("whole number of batches"), "{stderr}");
    assert!(out.stdout.is_empty());
    let out = train(
        &source,
        "interp",
        &[format!("--train-data=0={}", data.display())],
    );
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("trainable weight"), "{stderr}");
}
