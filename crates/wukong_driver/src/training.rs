//! The `--train` loop's data plumbing and update policy (N82, stage 1): initial buffers and streamed
//! batches read from files, trained buffers written back, the learning-rate schedule (linear
//! warmup, then constant or cosine decay) and the global gradient-norm clip.
//!
//! Every file is the layout the language's own `read_f32`/`write_f32` builtins use — a headerless
//! run of little-endian IEEE-754 binary32 values — so a `.wk` program, a `--train-save`
//! checkpoint and `numpy.ndarray.astype('<f4').tofile` interoperate with no converter.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// The learning-rate schedule `--train` applies after the warmup (`--train-schedule`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrainSchedule {
    /// The base rate on every step after the warmup (the default).
    Constant,
    /// Cosine decay from the base rate toward zero over the steps after the warmup —
    /// `torch.optim.lr_scheduler.CosineAnnealingLR` with `T_max` = those steps and `eta_min = 0`.
    Cosine,
}

/// The learning rate of update `step` (0-based) of a `steps`-update run.
///
/// The first `warmup` updates ramp linearly to the base rate (`base·(step + 1)/warmup`, so the
/// last warmup update uses the full rate); after that the schedule applies, the cosine one over the
/// remaining `steps - warmup` updates. With no warmup and the constant schedule this is `base`
/// itself, bit for bit — the default run is unchanged.
pub(crate) fn lr_at(
    base: f32,
    step: usize,
    steps: usize,
    warmup: usize,
    schedule: TrainSchedule,
) -> f32 {
    if step < warmup {
        return (f64::from(base) * (step + 1) as f64 / warmup as f64) as f32;
    }
    match schedule {
        TrainSchedule::Constant => base,
        TrainSchedule::Cosine => {
            let span = steps.saturating_sub(warmup).max(1);
            let progress = (step - warmup) as f64 / span as f64;
            (f64::from(base) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())) as f32
        }
    }
}

/// `torch.nn.utils.clip_grad_norm_`: the global L2 norm over every trainable gradient (accumulated
/// in f64, so the norm itself carries no f32 summation-order error); when `max_norm / (norm + 1e-6)`
/// is below one, every gradient is scaled by it. Returns the norm before clipping.
pub(crate) fn clip_global_norm(grads: &mut [Vec<f32>], max_norm: f32) -> f64 {
    let norm = grads
        .iter()
        .flatten()
        .map(|&g| f64::from(g) * f64::from(g))
        .sum::<f64>()
        .sqrt();
    let coef = f64::from(max_norm) / (norm + 1e-6);
    if coef < 1.0 {
        let c = coef as f32;
        for g in grads.iter_mut().flatten() {
            *g *= c;
        }
    }
    norm
}

/// Decode little-endian f32 values from `bytes` (a multiple of 4 long) into `dst`.
fn decode_f32(bytes: &[u8], dst: &mut [f32]) {
    for (d, chunk) in dst.iter_mut().zip(bytes.chunks_exact(4)) {
        *d = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
}

/// `--train-init=<i>=<path>`: buffer `index` (`len` elements) read from `path`, which must hold
/// exactly `len` f32 — a checkpoint of another shape is an error, never a silent prefix.
pub(crate) fn read_init(path: &Path, index: usize, len: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("--train-init: cannot read {}: {e}", path.display()))?;
    let want = len as u64 * 4;
    if bytes.len() as u64 != want {
        return Err(format!(
            "--train-init: {} holds {} bytes, but buffer {index} needs exactly {want} ({len} \
             little-endian f32)",
            path.display(),
            bytes.len()
        ));
    }
    let mut out = vec![0.0f32; len];
    decode_f32(&bytes, &mut out);
    Ok(out)
}

/// `--train-data=<i>=<path>`: a file of whole batches for buffer `index`, streamed one batch per
/// update — update `step` reads batch `step mod batches`, through one reused `len·4`-byte buffer, so
/// a dataset larger than memory trains in O(batch) space.
pub(crate) struct DataStream {
    pub(crate) index: usize,
    path: PathBuf,
    file: File,
    batches: u64,
    bytes: Vec<u8>,
}

impl DataStream {
    pub(crate) fn open(path: &Path, index: usize, len: usize) -> Result<DataStream, String> {
        let file = File::open(path)
            .map_err(|e| format!("--train-data: cannot open {}: {e}", path.display()))?;
        let size = file
            .metadata()
            .map_err(|e| format!("--train-data: cannot stat {}: {e}", path.display()))?
            .len();
        let batch = len as u64 * 4;
        if batch == 0 || size == 0 || !size.is_multiple_of(batch) {
            return Err(format!(
                "--train-data: {} holds {size} bytes, not a positive whole number of batches of \
                 buffer {index} ({len} little-endian f32 = {batch} bytes each)",
                path.display()
            ));
        }
        Ok(DataStream {
            index,
            path: path.to_path_buf(),
            file,
            batches: size / batch,
            bytes: vec![0; len * 4],
        })
    }

    /// Overwrite `dst` with the batch update `step` trains on (`step mod batches`).
    pub(crate) fn load(&mut self, step: usize, dst: &mut [f32]) -> Result<(), String> {
        let batch = step as u64 % self.batches;
        let at = batch * self.bytes.len() as u64;
        self.file
            .seek(SeekFrom::Start(at))
            .and_then(|_| self.file.read_exact(&mut self.bytes))
            .map_err(|e| {
                format!(
                    "--train-data: cannot read batch {batch} of {}: {e}",
                    self.path.display()
                )
            })?;
        decode_f32(&self.bytes, dst);
        Ok(())
    }
}

/// `--train-save=<i>=<path>`: write `data` to `path` as little-endian f32.
pub(crate) fn save_f32(path: &Path, data: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)
        .map_err(|e| format!("--train-save: cannot write {}: {e}", path.display()))
}

/// The structural checks on the file options, before any buffer is allocated or file touched:
/// every index names a loss parameter, none is the scalar loss output (except a save, which may
/// record the final loss), a streamed buffer is not a trainable weight, and no buffer is both
/// initialized and streamed or named twice by one option.
pub(crate) fn validate(
    init: &[(usize, PathBuf)],
    data: &[(usize, PathBuf)],
    save: &[(usize, PathBuf)],
    nparams: usize,
    loss_out: usize,
    wrt: &[usize],
) -> Result<(), String> {
    for (flag, list) in [
        ("--train-init", init),
        ("--train-data", data),
        ("--train-save", save),
    ] {
        let mut seen = std::collections::HashSet::new();
        for &(i, _) in list {
            if i >= nparams {
                return Err(format!(
                    "{flag} index {i} is out of range (the loss function has {nparams} \
                     parameter(s))"
                ));
            }
            if flag != "--train-save" && i == loss_out {
                return Err(format!(
                    "{flag} index {i} is the scalar loss output (the last parameter)"
                ));
            }
            if !seen.insert(i) {
                return Err(format!("{flag} names buffer {i} more than once"));
            }
        }
    }
    for &(i, _) in data {
        if wrt.contains(&i) {
            return Err(format!(
                "--train-data buffer {i} is a trainable weight (in --grad-wrt); stream data into \
                 the fixed inputs"
            ));
        }
        if init.iter().any(|&(j, _)| j == i) {
            return Err(format!(
                "buffer {i} is given both --train-init and --train-data"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_schedule_is_the_base_rate_bit_for_bit() {
        for step in [0, 1, 7, 99] {
            let lr = lr_at(0.013, step, 100, 0, TrainSchedule::Constant);
            assert_eq!(lr.to_bits(), 0.013f32.to_bits());
        }
    }

    /// Warmup `w = 4`: updates 0..3 use base·1/4 .. base·4/4, then the schedule. Cosine over the
    /// remaining 8 updates: 1, (1 + cos(π/8))/2, ..., never zero on an update.
    #[test]
    fn warmup_then_cosine_follows_the_torch_schedule() {
        let base = 0.5f32;
        let got: Vec<f32> = (0..12)
            .map(|s| lr_at(base, s, 12, 4, TrainSchedule::Cosine))
            .collect();
        assert_eq!(&got[..4], &[0.125f32, 0.25, 0.375, 0.5]);
        for (k, &lr) in got[4..].iter().enumerate() {
            let want = 0.5 * 0.5 * (1.0 + (std::f64::consts::PI * k as f64 / 8.0).cos());
            assert!(
                (f64::from(lr) - want).abs() < 1e-7,
                "update {}: {lr}",
                k + 4
            );
            assert!(lr > 0.0);
        }
        // A constant schedule after the same warmup holds the base rate.
        assert_eq!(lr_at(base, 9, 12, 4, TrainSchedule::Constant), base);
        // Cosine with nothing after the warmup still answers (no division by zero).
        assert_eq!(lr_at(base, 4, 4, 4, TrainSchedule::Cosine), base);
    }

    /// `clip_grad_norm_`: [3, 4] has norm 5; clipped to 1 it becomes [0.6, 0.8] (up to the 1e-6
    /// guard); a norm already under the bound is untouched, bit for bit.
    #[test]
    fn clip_scales_only_above_the_bound() {
        let mut g = vec![vec![3.0f32], vec![4.0f32]];
        let norm = clip_global_norm(&mut g, 1.0);
        assert_eq!(norm, 5.0);
        assert!(
            (g[0][0] - 0.6).abs() < 1e-6 && (g[1][0] - 0.8).abs() < 1e-6,
            "{g:?}"
        );
        let mut small = vec![vec![0.3f32, -0.4]];
        clip_global_norm(&mut small, 1.0);
        assert_eq!(small, vec![vec![0.3f32, -0.4]]);
    }

    #[test]
    fn files_round_trip_and_batches_stream_in_order() {
        let dir = std::env::temp_dir().join(format!("wukong-train-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.bin");
        save_f32(&path, &[1.5, -2.0, 0.25]).unwrap();
        assert_eq!(read_init(&path, 0, 3).unwrap(), vec![1.5, -2.0, 0.25]);
        let err = read_init(&path, 2, 4).unwrap_err();
        assert!(err.contains("needs exactly 16"), "{err}");

        // Three batches of two: steps 0..4 read batches 0, 1, 2, 0, 1.
        let data = dir.join("d.bin");
        save_f32(&data, &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0]).unwrap();
        let mut stream = DataStream::open(&data, 1, 2).unwrap();
        assert_eq!(stream.batches, 3);
        let mut buf = [0.0f32; 2];
        for (step, first) in [(0usize, 0.0f32), (1, 10.0), (2, 20.0), (3, 0.0), (4, 10.0)] {
            stream.load(step, &mut buf).unwrap();
            assert_eq!(buf, [first, first + 1.0], "step {step}");
        }
        // Windows refuses to delete a file that is still open.
        drop(stream);
        // Not a whole number of batches.
        let err = DataStream::open(&data, 1, 4).err().unwrap();
        assert!(err.contains("whole number of batches"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn validation_names_the_offending_option() {
        let p = || PathBuf::from("x.bin");
        let ok = validate(&[(0, p())], &[(1, p())], &[(0, p()), (3, p())], 4, 3, &[0]);
        assert!(ok.is_ok());
        for (init, data, save, want) in [
            (
                vec![(4, p())],
                vec![],
                vec![],
                "--train-init index 4 is out of range",
            ),
            (vec![(3, p())], vec![], vec![], "scalar loss output"),
            (vec![], vec![(0, p())], vec![], "is a trainable weight"),
            (
                vec![(1, p())],
                vec![(1, p())],
                vec![],
                "both --train-init and --train-data",
            ),
            (vec![], vec![(1, p()), (1, p())], vec![], "more than once"),
            (vec![], vec![], vec![(9, p())], "--train-save index 9"),
        ] {
            let err = validate(&init, &data, &save, 4, 3, &[0]).unwrap_err();
            assert!(err.contains(want), "{err}");
        }
    }
}
