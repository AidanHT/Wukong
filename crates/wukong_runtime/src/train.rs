//! The `--train` loop: ONE implementation for every way a training run executes.
//!
//! `wukongc --train` runs the loop in process — the loss forward, its generated backward and the
//! fused AdamW kernels on the interpreter or the Cranelift JIT — and `wukongc --train --emit=exe`
//! bakes the same loop into an executable, whose synthesized `main` hands
//! [`wukong_rt_train_main`] the run's [`TrainSpec`] (encoded, as a `.rodata` blob) and the address
//! of the compiled kernel dispatcher. Both paths call [`run_training`] here — the deterministic
//! init, the per-step snapshot/restore of the buffers the forward writes, the global gradient-norm
//! clip, the learning-rate schedule, SGD and the AdamW hyperparameter buffer — and print the
//! trajectory with [`format_report`]. The artifact's stdout therefore equals the in-process run's
//! byte for byte **by construction**: there is no second, mirrored loop to drift.
//!
//! # The kernel ABI
//!
//! Every kernel the loop calls takes a table of f32 buffers and is chosen by a selector
//! ([`KernelSel`]): the forward ([`SEL_FORWARD`]) takes the loss function's buffers; the backward
//! ([`SEL_GRADIENT`]) takes those plus one gradient buffer per trainable weight; the AdamW kernel for
//! the `k`-th distinct trainable-tensor size, ascending ([`TrainSpec::adamw_sizes`]), is selector
//! `SEL_ADAMW + k` and takes `(w, g, m, v, hp)`. [`TrainSpec::kernel_buffers`] lists each selector's
//! buffer lengths, and the driver builds its dispatcher from that list, so the two sides cannot
//! disagree on a selector.

use std::io::Write;

/// The element index into the AdamW hyperparameter buffer. The compiler's
/// `wukong_autodiff::optim::hp` (which builds the kernel that reads it) is the other copy; the
/// compiler does not depend on this crate's internals for it, so `wukong_driver` pins the two tables
/// equal at compile time and in a test.
pub mod hp {
    pub const LR: usize = 0;
    pub const BETA1: usize = 1;
    pub const BETA2: usize = 2;
    pub const EPS: usize = 3;
    pub const WD: usize = 4;
    /// `1 - beta1^t`
    pub const BC1: usize = 5;
    /// `1 - beta2^t`
    pub const BC2: usize = 6;
    pub const LEN: usize = 7;
}

/// A dispatcher selector (see the module docs).
pub type KernelSel = usize;
/// The loss forward: the loss function's own buffers.
pub const SEL_FORWARD: KernelSel = 0;
/// The backward (which replays the forward): the loss buffers, then one gradient per weight.
pub const SEL_GRADIENT: KernelSel = 1;
/// The first fused AdamW kernel; the one for the `k`-th entry of [`TrainSpec::adamw_sizes`] is
/// `SEL_ADAMW + k`.
pub const SEL_ADAMW: KernelSel = 2;

/// The optimizer the loop applies to the trainable weights.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Optimizer {
    /// Plain SGD, `w -= lr * g`, applied here (no kernel).
    Sgd,
    /// Fused decoupled-weight-decay AdamW, one kernel call per trainable weight.
    AdamW,
}

/// The learning-rate schedule after the warmup.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Schedule {
    /// The base rate on every update after the warmup.
    Constant,
    /// Cosine decay from the base rate toward zero over the updates after the warmup —
    /// `torch.optim.lr_scheduler.CosineAnnealingLR` with `T_max` = those updates and `eta_min = 0`.
    Cosine,
}

/// Everything a training run is, apart from the kernels themselves.
///
/// Convention (the `--train` one): `lens` are the element counts of the loss function's buffer
/// parameters, the **last** of which is the scalar loss output `[f32; 1]`; `wrt` names the trainable
/// weights; every other buffer is fixed data. `restore` lists the buffers (never the loss output)
/// the forward may write: each is snapshotted before every gradient run and put back after it, so
/// every update sees its data, and the optimizer its weights, exactly as they were before the
/// forward ran.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainSpec {
    pub lens: Vec<usize>,
    pub wrt: Vec<usize>,
    pub restore: Vec<usize>,
    pub steps: usize,
    pub lr: f32,
    pub optimizer: Optimizer,
    pub seed: u64,
    pub weight_decay: f32,
    pub clip: Option<f32>,
    pub warmup: usize,
    pub schedule: Schedule,
}

/// The `.rodata` spec blob's magic, `WKTRAIN` plus a NUL.
pub const SPEC_MAGIC: [u8; 8] = *b"WKTRAIN\0";
/// The spec blob layout version. Bump it with any layout change: a blob of another version is
/// refused rather than misread.
pub const SPEC_VERSION: u32 = 1;

/// Vec allocations and native pointer offsets must fit `isize`, not merely `usize`.
pub fn valid_extent(count: usize) -> bool {
    count
        .checked_mul(std::mem::size_of::<f32>())
        .is_some_and(|bytes| bytes <= isize::MAX as usize)
}

impl TrainSpec {
    /// The index of the scalar loss output (the last buffer).
    pub fn loss_out(&self) -> usize {
        self.lens.len().saturating_sub(1)
    }

    /// The distinct trainable-tensor sizes, ascending — one fused AdamW kernel each (none for SGD).
    pub fn adamw_sizes(&self) -> Vec<usize> {
        if self.optimizer != Optimizer::AdamW {
            return Vec::new();
        }
        let mut sizes: Vec<usize> = self.wrt.iter().map(|&wi| self.lens[wi]).collect();
        sizes.sort_unstable();
        sizes.dedup();
        sizes
    }

    /// The buffer lengths each selector's kernel takes, indexed by selector.
    pub fn kernel_buffers(&self) -> Vec<Vec<usize>> {
        let mut grad = self.lens.clone();
        grad.extend(self.wrt.iter().map(|&wi| self.lens[wi]));
        let mut out = vec![self.lens.clone(), grad];
        out.extend(
            self.adamw_sizes()
                .into_iter()
                .map(|n| vec![n, n, n, n, hp::LEN]),
        );
        out
    }

    /// The structural checks every run makes before allocating a buffer. The driver reports the
    /// user-facing ones earlier with the same wording; this is the backstop for a decoded blob.
    pub fn validate(&self) -> Result<(), String> {
        let nparams = self.lens.len();
        if nparams == 0 {
            return Err("the loss function has no buffer parameters".into());
        }
        if self.lens.iter().any(|&n| !valid_extent(n)) {
            return Err("training buffer byte extent is not representable".into());
        }
        let loss_out = nparams - 1;
        if self.lens[loss_out] != 1 {
            return Err(
                "the last parameter must be the scalar loss output `[f32; 1]` (the `--train` \
                 convention)"
                    .into(),
            );
        }
        if self.wrt.is_empty() {
            return Err("there is no trainable weight".into());
        }
        for (what, list) in [("--grad-wrt", &self.wrt), ("restored", &self.restore)] {
            let mut seen = vec![false; nparams];
            for &i in list.iter() {
                if i >= nparams {
                    return Err(format!(
                        "{what} index {i} is out of range (the loss function has {nparams} \
                         parameter(s))"
                    ));
                }
                if i == loss_out {
                    return Err(format!(
                        "{what} index {i} is the loss-output parameter (the last one)"
                    ));
                }
                if std::mem::replace(&mut seen[i], true) {
                    return Err(format!("{what} index {i} is repeated"));
                }
            }
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err("--train-weight-decay must be finite and non-negative".into());
        }
        if self.weight_decay != 0.0 && self.optimizer != Optimizer::AdamW {
            return Err("nonzero --train-weight-decay requires --train-opt=adamw".into());
        }
        if self.clip.is_some_and(|c| !c.is_finite() || c <= 0.0) {
            return Err("--train-clip must be finite and positive".into());
        }
        Ok(())
    }

    /// The versioned little-endian blob a training executable carries in `.rodata`: the magic, the
    /// version, then every field in declaration order (`usize` as `u64`, `f32` as its bits, an
    /// enum or an `Option` flag as one byte).
    pub fn encode(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(64 + 8 * (self.lens.len() + self.wrt.len() + self.restore.len()));
        out.extend_from_slice(&SPEC_MAGIC);
        out.extend_from_slice(&SPEC_VERSION.to_le_bytes());
        for list in [&self.lens, &self.wrt, &self.restore] {
            out.extend_from_slice(&(list.len() as u64).to_le_bytes());
            for &v in list.iter() {
                out.extend_from_slice(&(v as u64).to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.steps as u64).to_le_bytes());
        out.extend_from_slice(&self.lr.to_bits().to_le_bytes());
        out.push(match self.optimizer {
            Optimizer::Sgd => 0,
            Optimizer::AdamW => 1,
        });
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&self.weight_decay.to_bits().to_le_bytes());
        match self.clip {
            None => out.extend_from_slice(&[0, 0, 0, 0, 0]),
            Some(c) => {
                out.push(1);
                out.extend_from_slice(&c.to_bits().to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.warmup as u64).to_le_bytes());
        out.push(match self.schedule {
            Schedule::Constant => 0,
            Schedule::Cosine => 1,
        });
        out
    }

    /// Decode (and [`validate`](Self::validate)) a blob [`encode`](Self::encode) produced. A wrong
    /// magic or version, a truncated or over-long blob, or an out-of-range enum is an error — never
    /// a partially-read spec.
    pub fn decode(bytes: &[u8]) -> Result<TrainSpec, String> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(8)? != SPEC_MAGIC {
            return Err("not a Wukong training spec (bad magic)".into());
        }
        let version = u32::from_le_bytes(r.array()?);
        if version != SPEC_VERSION {
            return Err(format!(
                "training spec version {version}, but this runtime reads version {SPEC_VERSION}"
            ));
        }
        let mut lists: [Vec<usize>; 3] = Default::default();
        for list in &mut lists {
            let n = r.usize()?;
            // Each entry is 8 bytes, so a count the blob cannot hold is refused before reserving.
            if n > r.remaining() / 8 {
                return Err("training spec list is longer than the blob".into());
            }
            *list = (0..n).map(|_| r.usize()).collect::<Result<_, _>>()?;
        }
        let [lens, wrt, restore] = lists;
        let steps = r.usize()?;
        let lr = f32::from_bits(u32::from_le_bytes(r.array()?));
        let optimizer = match r.byte()? {
            0 => Optimizer::Sgd,
            1 => Optimizer::AdamW,
            other => {
                return Err(format!(
                    "unknown optimizer code {other} in the training spec"
                ))
            }
        };
        let seed = u64::from_le_bytes(r.array()?);
        let weight_decay = f32::from_bits(u32::from_le_bytes(r.array()?));
        let has_clip = r.byte()?;
        let clip_bits = u32::from_le_bytes(r.array()?);
        let clip = match has_clip {
            0 => None,
            1 => Some(f32::from_bits(clip_bits)),
            other => return Err(format!("bad clip flag {other} in the training spec")),
        };
        let warmup = r.usize()?;
        let schedule = match r.byte()? {
            0 => Schedule::Constant,
            1 => Schedule::Cosine,
            other => {
                return Err(format!(
                    "unknown schedule code {other} in the training spec"
                ))
            }
        };
        if r.remaining() != 0 {
            return Err(format!(
                "{} trailing byte(s) after the training spec",
                r.remaining()
            ));
        }
        let spec = TrainSpec {
            lens,
            wrt,
            restore,
            steps,
            lr,
            optimizer,
            seed,
            weight_decay,
            clip,
            warmup,
            schedule,
        };
        spec.validate()?;
        Ok(spec)
    }
}

/// A bounds-checked little-endian cursor over the spec blob.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.at
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.remaining() {
            return Err("training spec is truncated".into());
        }
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn byte(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn usize(&mut self) -> Result<usize, String> {
        usize::try_from(u64::from_le_bytes(self.array()?))
            .map_err(|_| "training spec value does not fit this host's usize".to_string())
    }
}

/// A tiny deterministic LCG producing `f32` uniform in `[-0.4, 0.4)` — seeds the training buffers
/// so a run is reproducible (no hidden entropy) and a test can assert a specific loss trajectory.
///
/// The draw is the standard u64 -> f32 conversion: the TOP 24 bits of the state, scaled by `2^-24`
/// (exact in f32), give `u` uniform in `[0, 1)`; `(u - 0.5)·0.8` is then symmetric about zero. (It
/// used to keep 31 bits, `seed >> 33`, and divide by `u32::MAX`, so `u` never reached 0.5 and every
/// value lay in `[-0.4, 0)`: an all-negative init that starts a ReLU net mostly dead.)
pub fn init_uniform(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((*seed >> 40) as f32) * (1.0 / 16_777_216.0);
            (u - 0.5) * 0.8
        })
        .collect()
}

/// The learning rate of update `step` (0-based) of a `steps`-update run.
///
/// The first `warmup` updates ramp linearly to the base rate (`base·(step + 1)/warmup`, so the
/// last warmup update uses the full rate); after that the schedule applies, the cosine one over the
/// remaining `steps - warmup` updates. With no warmup and the constant schedule this is `base`
/// itself, bit for bit — the default run is unchanged.
pub fn lr_at(base: f32, step: usize, steps: usize, warmup: usize, schedule: Schedule) -> f32 {
    if step < warmup {
        return (f64::from(base) * (step + 1) as f64 / warmup as f64) as f32;
    }
    match schedule {
        Schedule::Constant => base,
        Schedule::Cosine => {
            let span = steps.saturating_sub(warmup).max(1);
            let progress = (step - warmup) as f64 / span as f64;
            (f64::from(base) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())) as f32
        }
    }
}

/// `torch.nn.utils.clip_grad_norm_`: the global L2 norm over every trainable gradient (accumulated
/// in f64, so the norm itself carries no f32 summation-order error); when `max_norm / (norm + 1e-6)`
/// is below one, every gradient is scaled by it. Returns the norm before clipping.
pub fn clip_global_norm(grads: &mut [Vec<f32>], max_norm: f32) -> f64 {
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

/// What a training run produced: `trajectory[i]` is the loss after `i` updates (so it holds
/// `steps + 1` entries, the last from one final forward), and `buffers` the loss function's buffers
/// after the last update — the trained weights, with every restored buffer as it was.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainRun {
    pub trajectory: Vec<f32>,
    pub buffers: Vec<Vec<f32>>,
}

/// A kernel call: `(selector, buffers)`, the buffers exactly [`TrainSpec::kernel_buffers`]'s
/// lengths for that selector.
pub type Dispatch<'a> = dyn FnMut(KernelSel, &mut [Vec<f32>]) -> Result<(), String> + 'a;

/// Run the training loop of `spec`, calling every kernel through `dispatch`.
pub fn run_training(spec: &TrainSpec, dispatch: &mut Dispatch<'_>) -> Result<TrainRun, String> {
    run_training_with(spec, Vec::new(), &mut |_, _| Ok(()), dispatch)
}

/// [`run_training`] with the in-process run's two file hooks: `inits` replace buffers after the
/// seeded init (`--train-init`), and `load(step, buffers)` is called before update `step` — and
/// with `step == steps` before the final forward — to stream a batch into the fixed inputs
/// (`--train-data`). `buffers` there are the loss function's.
pub fn run_training_with(
    spec: &TrainSpec,
    inits: Vec<(usize, Vec<f32>)>,
    load: &mut Dispatch<'_>,
    dispatch: &mut Dispatch<'_>,
) -> Result<TrainRun, String> {
    spec.validate()?;
    let lens = &spec.lens;
    let nparams = lens.len();
    let loss_out = nparams - 1;

    // Deterministic buffer init (every buffer draws in order, so an init file replacing one buffer
    // leaves the others' values unchanged); the loss output is zeroed, moment state starts at zero.
    let mut seed = spec.seed;
    let mut bufs: Vec<Vec<f32>> = lens.iter().map(|&n| init_uniform(&mut seed, n)).collect();
    bufs[loss_out] = vec![0.0; 1];
    for (i, values) in inits {
        if i >= nparams || i == loss_out || values.len() != lens[i] {
            return Err(format!("initial values for buffer {i} do not fit it"));
        }
        bufs[i] = values;
    }
    // The gradient kernel's buffers — the forward's, then one gradient per trainable weight — live
    // for the whole run and are updated in place: no buffer is copied per update. A buffer the
    // forward may write (other than the loss output) is snapshotted before each gradient run and put
    // back after it — the semantics of a fresh per-update copy, paid only for the buffers the
    // forward writes (none, for a loss whose buffers only feed kernels).
    bufs.extend(spec.wrt.iter().map(|&wi| vec![0.0; lens[wi]]));
    let restore = &spec.restore;
    let mut snapshot: Vec<Vec<f32>> = restore.iter().map(|&i| vec![0.0; lens[i]]).collect();
    let mut mstate: Vec<Vec<f32>> = spec.wrt.iter().map(|&wi| vec![0.0; lens[wi]]).collect();
    let mut vstate: Vec<Vec<f32>> = spec.wrt.iter().map(|&wi| vec![0.0; lens[wi]]).collect();
    let mut hpbuf = vec![0.0f32; hp::LEN];
    let adamw_sizes = spec.adamw_sizes();

    let steps = spec.steps;
    let (beta1, beta2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    // Grown amortized rather than pre-reserved: `steps` is an unvalidated CLI integer, and
    // reserving from it let `--train-steps=999999999999` abort inside the allocator before a
    // single update ran. `usize::MAX` additionally wrapped `+ 1` to a zero reservation.
    let mut traj = Vec::new();

    for step in 0..steps {
        load(step, &mut bufs[..nparams])?;
        for (saved, &i) in snapshot.iter_mut().zip(restore) {
            saved.copy_from_slice(&bufs[i]);
        }
        bufs[loss_out][0] = 0.0;
        for g in &mut bufs[nparams..] {
            g.fill(0.0);
        }
        // The gradient kernel replays the forward, so one call gives the current loss + the
        // gradients.
        dispatch(SEL_GRADIENT, &mut bufs)?;
        traj.push(bufs[loss_out][0]);
        for (saved, &i) in snapshot.iter().zip(restore) {
            bufs[i].copy_from_slice(saved);
        }

        let (params, grads) = bufs.split_at_mut(nparams);
        if let Some(max_norm) = spec.clip {
            clip_global_norm(grads, max_norm);
        }
        let lr = lr_at(spec.lr, step, steps, spec.warmup, spec.schedule);
        match spec.optimizer {
            Optimizer::Sgd => {
                for (g, &wi) in grads.iter().zip(&spec.wrt) {
                    for (wv, &gv) in params[wi].iter_mut().zip(g) {
                        *wv -= lr * gv;
                    }
                }
            }
            Optimizer::AdamW => {
                let t = (step + 1) as i32;
                hpbuf.fill(0.0);
                hpbuf[hp::LR] = lr;
                hpbuf[hp::BETA1] = beta1 as f32;
                hpbuf[hp::BETA2] = beta2 as f32;
                hpbuf[hp::EPS] = eps as f32;
                hpbuf[hp::WD] = spec.weight_decay;
                hpbuf[hp::BC1] = (1.0 - beta1.powi(t)) as f32;
                hpbuf[hp::BC2] = (1.0 - beta2.powi(t)) as f32;
                for (i, &wi) in spec.wrt.iter().enumerate() {
                    let k = adamw_sizes
                        .binary_search(&lens[wi])
                        .map_err(|_| "internal: no AdamW kernel for a weight size".to_string())?;
                    // The kernel's five buffers are moved in and back out — no copies.
                    let mut ab = [
                        std::mem::take(&mut params[wi]),
                        std::mem::take(&mut grads[i]),
                        std::mem::take(&mut mstate[i]),
                        std::mem::take(&mut vstate[i]),
                        std::mem::take(&mut hpbuf),
                    ];
                    let ran = dispatch(SEL_ADAMW + k, &mut ab[..]);
                    let [w, g, m, v, h] = ab;
                    params[wi] = w;
                    grads[i] = g;
                    mstate[i] = m;
                    vstate[i] = v;
                    hpbuf = h;
                    ran?;
                }
            }
        }
    }
    // One final forward to report the loss after the last update (on the batch the next update
    // would see).
    load(steps, &mut bufs[..nparams])?;
    bufs[loss_out][0] = 0.0;
    bufs.truncate(nparams);
    for (saved, &i) in snapshot.iter_mut().zip(restore) {
        saved.copy_from_slice(&bufs[i]);
    }
    dispatch(SEL_FORWARD, &mut bufs)?;
    traj.push(bufs[loss_out][0]);
    // The returned buffers are the trained values, not whatever the reporting forward left in a
    // buffer it writes (the same restore every update does).
    for (saved, &i) in snapshot.iter().zip(restore) {
        bufs[i].copy_from_slice(saved);
    }
    Ok(TrainRun {
        trajectory: traj,
        buffers: bufs,
    })
}

/// The text a training run prints: a header naming the run (the update policy only where it
/// differs from the default, so a default run prints the header it always has), one line per
/// trajectory entry at six decimals, and the first -> last summary.
pub fn format_report(spec: &TrainSpec, traj: &[f32]) -> String {
    let mut out = String::new();
    let opt = match spec.optimizer {
        Optimizer::Sgd => "sgd",
        Optimizer::AdamW => "adamw",
    };
    let decay = if spec.weight_decay == 0.0 {
        String::new()
    } else {
        format!(", weight_decay={}", spec.weight_decay)
    };
    let mut policy = String::new();
    if let Some(clip) = spec.clip {
        policy.push_str(&format!(", clip={clip}"));
    }
    if spec.warmup > 0 {
        policy.push_str(&format!(", warmup={}", spec.warmup));
    }
    if spec.schedule == Schedule::Cosine {
        policy.push_str(", schedule=cosine");
    }
    out.push_str(&format!(
        "train: {} step(s), lr={}, optimizer={opt}{decay}{policy}\n",
        traj.len().saturating_sub(1),
        spec.lr
    ));
    for (i, l) in traj.iter().enumerate() {
        out.push_str(&format!("step {i:>4}: loss {l:.6}\n"));
    }
    if let (Some(&first), Some(&last)) = (traj.first(), traj.last()) {
        out.push_str(&format!(
            "loss {first:.6} -> {last:.6}  ({:.2}x reduction)\n",
            if last != 0.0 {
                first / last
            } else {
                f32::INFINITY
            }
        ));
    }
    out
}

/// The compiled kernel dispatcher a training executable passes in: `(buffer table, selector)`.
pub type TrainDispatchFn = extern "C" fn(*const *mut f32, u64);

/// The entry a `wukongc --train --emit=exe|obj` artifact's `main` calls: decode the baked spec,
/// run [`run_training`] with every kernel call going through `dispatch`, and print
/// [`format_report`] — the very text `wukongc --train` prints. Returns the process exit code: `0`,
/// or `1` after an `error: --train: …` line on stderr.
///
/// # Safety
/// `spec` must address `spec_len` readable bytes. `dispatch` must be the compiler-generated
/// dispatcher for exactly this spec: for each selector it reads only the table entries
/// [`TrainSpec::kernel_buffers`] lists, each a dense f32 buffer of that length, and it retains no
/// pointer after it returns.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_train_main(
    spec: *const u8,
    spec_len: i64,
    dispatch: TrainDispatchFn,
) -> i32 {
    let bytes: Option<&[u8]> = match usize::try_from(spec_len) {
        Ok(0) => Some(&[]),
        // SAFETY: the caller passes the address and length of the baked `.rodata` blob.
        Ok(n) if !spec.is_null() => Some(unsafe { std::slice::from_raw_parts(spec, n) }),
        _ => None,
    };
    train_main(
        bytes,
        dispatch,
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    )
}

/// [`wukong_rt_train_main`] with its blob already in hand and its two streams as parameters.
fn train_main(
    bytes: Option<&[u8]>,
    dispatch: TrainDispatchFn,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    let fail = |err: &mut dyn Write, message: &str| {
        let _ = writeln!(err, "error: --train: {message}");
        1
    };
    let Some(bytes) = bytes else {
        return fail(err, "invalid training spec: no spec blob");
    };
    let spec = match TrainSpec::decode(bytes) {
        Ok(spec) => spec,
        Err(e) => return fail(err, &format!("invalid training spec: {e}")),
    };
    let mut call = |selector: KernelSel, buffers: &mut [Vec<f32>]| -> Result<(), String> {
        // Rebuilt per call: AdamW moves buffers in and out, so their allocations change.
        let table: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
        dispatch(table.as_ptr(), selector as u64);
        Ok(())
    };
    match run_training(&spec, &mut call) {
        Ok(run) => {
            // Flushed here: an executable's `main` is not Rust's, so nothing flushes stdout at
            // exit on this crate's behalf.
            let _ = out.write_all(format_report(&spec, &run.trajectory).as_bytes());
            let _ = out.flush();
            0
        }
        Err(e) => fail(err, &e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> TrainSpec {
        TrainSpec {
            lens: vec![3, 2, 1],
            wrt: vec![0],
            restore: vec![1],
            steps: 4,
            lr: 0.125,
            optimizer: Optimizer::AdamW,
            seed: 7,
            weight_decay: 0.25,
            clip: Some(1.5),
            warmup: 2,
            schedule: Schedule::Cosine,
        }
    }

    #[test]
    fn train_spec_round_trips_and_refuses_damaged_blobs() {
        let s = spec();
        let blob = s.encode();
        assert_eq!(TrainSpec::decode(&blob).unwrap(), s);
        let plain = TrainSpec {
            optimizer: Optimizer::Sgd,
            weight_decay: 0.0,
            clip: None,
            schedule: Schedule::Constant,
            ..s.clone()
        };
        assert_eq!(TrainSpec::decode(&plain.encode()).unwrap(), plain);
        // Every truncation is refused, never read as a shorter spec.
        for n in 0..blob.len() {
            assert!(
                TrainSpec::decode(&blob[..n]).is_err(),
                "prefix of {n} bytes"
            );
        }
        let mut long = blob.clone();
        long.push(0);
        assert!(TrainSpec::decode(&long).unwrap_err().contains("trailing"));
        let mut magic = blob.clone();
        magic[0] ^= 1;
        assert!(TrainSpec::decode(&magic).unwrap_err().contains("magic"));
        let mut version = blob.clone();
        version[8] = 2;
        assert!(TrainSpec::decode(&version)
            .unwrap_err()
            .contains("version 2"));
        // A list count the blob cannot hold is refused before anything is reserved.
        let mut huge = blob.clone();
        huge[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(TrainSpec::decode(&huge).is_err());
        // The decoded spec is validated: the loss output cannot be a weight.
        let bad = TrainSpec {
            wrt: vec![2],
            ..s.clone()
        };
        assert!(TrainSpec::decode(&bad.encode())
            .unwrap_err()
            .contains("loss-output"));
    }

    #[test]
    fn train_kernel_selectors_follow_the_adamw_sizes() {
        let s = TrainSpec {
            lens: vec![4, 2, 4, 3, 1],
            wrt: vec![2, 0, 1],
            restore: vec![],
            ..spec()
        };
        assert_eq!(s.adamw_sizes(), vec![2, 4]);
        assert_eq!(
            s.kernel_buffers(),
            vec![
                vec![4, 2, 4, 3, 1],
                vec![4, 2, 4, 3, 1, 4, 4, 2],
                vec![2, 2, 2, 2, hp::LEN],
                vec![4, 4, 4, 4, hp::LEN],
            ]
        );
        let sgd = TrainSpec {
            optimizer: Optimizer::Sgd,
            weight_decay: 0.0,
            ..s
        };
        assert!(sgd.adamw_sizes().is_empty());
        assert_eq!(sgd.kernel_buffers().len(), 2);
    }

    #[test]
    fn train_default_schedule_is_the_base_rate_bit_for_bit() {
        for step in [0, 1, 7, 99] {
            let lr = lr_at(0.013, step, 100, 0, Schedule::Constant);
            assert_eq!(lr.to_bits(), 0.013f32.to_bits());
        }
    }

    /// Warmup `w = 4`: updates 0..3 use base·1/4 .. base·4/4, then the schedule. Cosine over the
    /// remaining 8 updates: 1, (1 + cos(π/8))/2, ..., never zero on an update.
    #[test]
    fn train_warmup_then_cosine_follows_the_torch_schedule() {
        let base = 0.5f32;
        let got: Vec<f32> = (0..12)
            .map(|s| lr_at(base, s, 12, 4, Schedule::Cosine))
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
        assert_eq!(lr_at(base, 9, 12, 4, Schedule::Constant), base);
        // Cosine with nothing after the warmup still answers (no division by zero).
        assert_eq!(lr_at(base, 4, 4, 4, Schedule::Cosine), base);
    }

    /// `clip_grad_norm_`: [3, 4] has norm 5; clipped to 1 it becomes [0.6, 0.8] (up to the 1e-6
    /// guard); a norm already under the bound is untouched, bit for bit.
    #[test]
    fn train_clip_scales_only_above_the_bound() {
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

    /// N80: the initializer covers its documented range symmetrically. It used to keep 31 bits of
    /// the LCG state and divide by `u32::MAX`, so every value lay in `[-0.4, 0)`. (The seed is the
    /// `--train-seed` default.)
    #[test]
    fn train_init_is_uniform_and_symmetric_about_zero() {
        let mut seed = 0x5EED_1234;
        let v = init_uniform(&mut seed, 4096);
        assert!(
            v.iter().all(|x| (-0.4..0.4).contains(x)),
            "out of [-0.4, 0.4)"
        );
        let negative = v.iter().filter(|&&x| x < 0.0).count();
        assert!(
            (1848..=2248).contains(&negative),
            "{negative} of 4096 negative"
        );
        let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!(v.iter().any(|&x| x > 0.35) && v.iter().any(|&x| x < -0.35));
    }

    /// The loop itself, against an independent closed form: `L(w) = (w - x)^2` with one weight,
    /// one fixed input the forward also WRITES (it must be restored after every gradient run), and
    /// SGD at lr 1/8. The dispatcher here is a plain Rust function, so this pins the loop's order of
    /// operations (draws, zeroing, restore, update, final forward) independently of any backend.
    #[test]
    fn train_loop_matches_a_hand_written_reference() {
        let s = TrainSpec {
            lens: vec![1, 1, 1],
            wrt: vec![0],
            restore: vec![1],
            steps: 3,
            lr: 0.125,
            optimizer: Optimizer::Sgd,
            seed: 7,
            weight_decay: 0.0,
            clip: None,
            warmup: 0,
            schedule: Schedule::Constant,
        };
        let mut calls = Vec::new();
        let mut dispatch = |sel: KernelSel, b: &mut [Vec<f32>]| -> Result<(), String> {
            calls.push(sel);
            let d = b[0][0] - b[1][0];
            b[2][0] = d * d;
            if sel == SEL_GRADIENT {
                b[3][0] += d + d;
            }
            // The forward scribbles on its fixed input; the loop must put it back.
            b[1][0] = 99.0;
            Ok(())
        };
        let run = run_training(&s, &mut dispatch).unwrap();
        assert_eq!(
            calls,
            vec![SEL_GRADIENT, SEL_GRADIENT, SEL_GRADIENT, SEL_FORWARD]
        );
        let mut seed = 7;
        let mut w = init_uniform(&mut seed, 1)[0];
        let x = init_uniform(&mut seed, 1)[0];
        let mut want = Vec::new();
        for _ in 0..3 {
            let d = w - x;
            want.push(d * d);
            w -= 0.125 * (d + d);
        }
        want.push((w - x) * (w - x));
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&run.trajectory), bits(&want));
        assert_eq!(run.buffers[0][0].to_bits(), w.to_bits());
        assert_eq!(
            run.buffers[1][0].to_bits(),
            x.to_bits(),
            "fixed input restored"
        );
        let report = format_report(&s, &run.trajectory);
        assert!(report.starts_with("train: 3 step(s), lr=0.125, optimizer=sgd\n"));
        assert_eq!(report.lines().count(), 6);
    }

    /// The report names only the non-default policy, so a default run prints the header it always
    /// has, and the summary guards a zero final loss.
    #[test]
    fn train_report_names_only_non_default_policy() {
        let s = spec();
        let text = format_report(&s, &[2.0, 0.0]);
        assert_eq!(
            text,
            "train: 1 step(s), lr=0.125, optimizer=adamw, weight_decay=0.25, clip=1.5, \
             warmup=2, schedule=cosine\nstep    0: loss 2.000000\nstep    1: loss 0.000000\n\
             loss 2.000000 -> 0.000000  (infx reduction)\n"
        );
    }

    extern "C" fn square_dispatch(table: *const *mut f32, selector: u64) {
        // SAFETY: the test's spec is `[1, 1]` (weight, loss); every selector's table holds at
        // least those two single-element buffers, and the gradient one a third.
        unsafe {
            let w = **table;
            **table.add(1) = w * w;
            if selector == SEL_GRADIENT as u64 {
                **table.add(2) += w + w;
            }
        }
    }

    /// The C entry decodes the blob, drives the dispatcher and prints exactly [`format_report`]'s
    /// text; a damaged or missing blob is exit 1 with an `error: --train:` line and no stdout.
    #[test]
    fn train_main_entry_runs_and_refuses_a_bad_blob() {
        let s = TrainSpec {
            lens: vec![1, 1],
            wrt: vec![0],
            restore: vec![],
            steps: 2,
            lr: 0.25,
            optimizer: Optimizer::Sgd,
            seed: 3,
            weight_decay: 0.0,
            clip: None,
            warmup: 0,
            schedule: Schedule::Constant,
        };
        let blob = s.encode();
        let (mut out, mut err) = (Vec::new(), Vec::new());
        assert_eq!(
            train_main(Some(&blob), square_dispatch, &mut out, &mut err),
            0
        );
        assert!(err.is_empty());
        // The same loop driven in Rust, against which the C entry's text must match byte for byte.
        let mut rust = |sel: KernelSel, b: &mut [Vec<f32>]| -> Result<(), String> {
            let mut table: Vec<*mut f32> = b.iter_mut().map(|v| v.as_mut_ptr()).collect();
            square_dispatch(table.as_mut_ptr() as *const *mut f32, sel as u64);
            Ok(())
        };
        let run = run_training(&s, &mut rust).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            format_report(&s, &run.trajectory)
        );
        for bytes in [Some(&blob[..3]), None] {
            let (mut out, mut err) = (Vec::new(), Vec::new());
            assert_eq!(train_main(bytes, square_dispatch, &mut out, &mut err), 1);
            assert!(out.is_empty());
            let err = String::from_utf8(err).unwrap();
            assert!(
                err.starts_with("error: --train: invalid training spec"),
                "{err}"
            );
        }
        // SAFETY: a null blob is refused before it is read; the failure goes to stderr.
        let code = unsafe { wukong_rt_train_main(std::ptr::null(), 8, square_dispatch) };
        assert_eq!(code, 1);
    }
}
