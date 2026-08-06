//! The GENERAL-CODE benchmark: Wukong written the way an ML engineer actually writes it.
//!
//! Everything else in this crate benchmarks the *recognizer dialect* — the handful of syntactic loop
//! shapes `wukong_mir_build` pattern-matches and replaces with a hand-written AVX2 kernel from
//! `wukong_runtime`. That is a real and load-bearing part of the compiler, but it means the existing
//! suite is structurally incapable of measuring what a program that misses those patterns costs.
//! It always writes the pattern.
//!
//! Recognized kernels are GATE-BLIND: a recognizer fires pre-optimization and in BOTH backends, so
//! `native_matches_interpreter` and `optimization_is_observationally_invariant` agree on the same
//! answer whether one fired or not. No correctness gate in the repo can see the difference. This
//! module therefore *prints the dispatch census* for every program it times — the equivalent of
//! `wukongc --emit=mir -O2 prog.wk | grep -oE 'wukong_[a-z0-9_]+'` — and writes the exact source it
//! compiled into the xbench temp directory so the census can be reproduced against the real binary.
//!
//! # Program 1: STRUCTURE TAX
//!
//! One pre-norm transformer block (RMSNorm -> RoPE -> causal MHA -> RMSNorm -> SwiGLU MLP) written
//! five ways, all computing the SAME f32 arithmetic in the SAME order:
//!
//! | | spelling | what it isolates |
//! |---|---|---|
//! | (a) | monolithic, recognizer dialect | the baseline: everything dispatches |
//! | (b) | monolithic, natural spelling (hoisted row bases, activation in the store) | pure spelling |
//! | (c) | factored into `rmsnorm`/`linear`/`rope`/`attend`/`swiglu` over `[]f32` | function factoring + runtime dims |
//! | (d) | (a) verbatim with the weights in a `struct Layer` | struct-held parameters |
//! | (e) | (a) verbatim in shape-typed `Tensor[f32, S, D]` with `a[i, j]` | the type system, as advertised |
//!
//! gcc/g++/rustc's spread across five such spellings is ~1.0 — they all lower to the same loops.
//! Wukong's spread IS the benchmark.
//!
//! # Program 2: FUSED CUSTOM LOSS
//!
//! Focal loss (gamma = 2) with label smoothing and per-class weights, forward AND a hand-written
//! backward, over logits `[R, C]`. Every loss Wukong ships a kernel for is a pre-baked pattern; a
//! loss invented after the recognizer table was written cannot be in it, and a new objective is the
//! most common thing a researcher writes.
//!
//! # Program 3: MAMBA / S6 SELECTIVE SCAN
//!
//! `h[d,n] = exp(dt·A)·h + dt·B·x ; y = sum_n C·h + D·x`, with a `[D, N]` matrix state and both the
//! decay and the input gate computed inside the loop from selective parameters. `wukong_lrscan_f32`
//! covers only a SCALAR-state recurrence with precomputed gates. The `t` loop is genuinely
//! sequential, so whole-loop pattern replacement has nothing to replace.
//!
//! # Honesty rules this module obeys
//!
//! * **The peer is written once, competently.** One C source ([`wk/general/peer_block.c`]), compiled
//!   unchanged by gcc and g++; one Rust source. Correct loop order for the memory layout (the NT
//!   weight layout streams both operands contiguously; V is transposed once per head so the P·V
//!   product streams too), `__restrict__` on all 27 genuinely-distinct buffers, `-O3 -march=native`.
//!   This suite has a cautionary tale of its own: its `c_colsum` peer is written column-outer over a
//!   row-major array and measures 57-70x slower than the competent row-outer form producing
//!   bit-identical output.
//! * **A `C(fast)` column.** Variant (a)'s dispatched kernels reassociate their reductions; an
//!   IEEE-serial C peer cannot vectorize an f32 dot product at all. `-ffast-math` is the like-for-like
//!   comparison and is reported next to the honest-flags one, never instead of it.
//! * **An f64 scalar reference, checked on EVERY lane.** Never a reduction — a partially-correct
//!   buffer passes a sum. The interpreter cannot be the oracle at this size (its `Value` enum costs
//!   ~24 bytes per f32 element), so the oracle is an f64 recomputation in this file.
//! * **Round-interleaved timings, ratios formed WITHIN a round.** Absolute GFLOP/s swings ~3x with
//!   this laptop's power and thermal state, so only same-run adjacent ratios are reportable — and
//!   "adjacent" has to be enforced, not hoped for. Every column of a section is built first and
//!   timed once per ROUND in a rotating, direction-alternating order ([`crate::round_order`]); the
//!   ratio is formed inside each round and reported as the MEDIAN across rounds with the range the
//!   rounds spanned. A ratio whose rounds disagree by more than [`crate::spread_limit`] prints as
//!   INCONCLUSIVE instead of as a number, and a power-state change anywhere in the section replaces
//!   every ratio in it with NON-REPORTABLE.
//! * **A `C(twin)` CONTROL.** The identical C source through the identical compiler at the identical
//!   flags, in a second DLL — a column with no language content in it at all, whose ratio against
//!   `C` therefore has expected value 1.00. What it reads away from 1.00 is this run's own floor,
//!   and when the control itself fails the spread gate the section says so and claims nothing.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use wukong_span::{Interner, SourceId};

// ---------------------------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------------------------

/// Block dimensions. Defaults are a small-but-realistic decoder block: S=256 tokens, D=256 model
/// dim, 4 heads of 64, SwiGLU inner dim 512 — ~403 MFLOP per forward, ~5.5 MB of live buffers, so
/// the GEMMs run out of L3 rather than L1 and the measurement is not a cache toy. Overridable with
/// `XBENCH_GEN_S` / `XBENCH_GEN_D` / `XBENCH_GEN_H` / `XBENCH_GEN_F` for a sweep.
#[derive(Clone, Copy)]
pub(crate) struct Dims {
    pub s: usize,
    pub d: usize,
    pub h: usize,
    pub f: usize,
}

impl Dims {
    fn from_env() -> Dims {
        let get = |k: &str, dflt: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(dflt)
        };
        Dims {
            s: get("XBENCH_GEN_S", 256),
            d: get("XBENCH_GEN_D", 256),
            h: get("XBENCH_GEN_H", 4),
            f: get("XBENCH_GEN_F", 512),
        }
    }
    fn hd(&self) -> usize {
        self.d / self.h
    }
    fn hd2(&self) -> usize {
        self.hd() / 2
    }
    /// 1/sqrt(head_dim). Exact in f32 whenever `hd` is a power of four, which every default is;
    /// the value is substituted into all five Wukong sources AND both peers as one decimal literal,
    /// so no variant can differ from another by its rounding.
    fn scale(&self) -> f64 {
        1.0 / (self.hd() as f64).sqrt()
    }
    /// Multiply-accumulates in one forward: 4·S·D² (QKV + output projection) + 2·S²·D (the two
    /// attention GEMMs, summed over heads since H·HD = D) + 3·S·D·F (gate, up, down).
    fn flops(&self) -> f64 {
        let (s, d, f) = (self.s as f64, self.d as f64, self.f as f64);
        2.0 * (4.0 * s * d * d + 2.0 * s * s * d + 3.0 * s * d * f)
    }
    fn ok(&self) -> Result<(), String> {
        if self.d % self.h != 0 {
            return Err(format!("D={} is not divisible by H={}", self.d, self.h));
        }
        if self.hd() % 2 != 0 {
            return Err(format!("head dim {} must be even for RoPE", self.hd()));
        }
        Ok(())
    }
}

/// Substitute the `$TOKEN$` placeholders. The `$`-delimited form makes the replacements
/// order-independent: `$S$` cannot match inside `$SD$`, nor `$HD$` inside `$HD2$`.
fn subst(src: &str, m: Dims) -> String {
    let hd = m.hd();
    let pairs: [(&str, String); 15] = [
        ("$SD$", (m.s * m.d).to_string()),
        ("$SS$", (m.s * m.s).to_string()),
        ("$SHD$", (m.s * hd).to_string()),
        ("$SH2$", (m.s * m.hd2()).to_string()),
        ("$SF$", (m.s * m.f).to_string()),
        ("$DD$", (m.d * m.d).to_string()),
        ("$FD$", (m.f * m.d).to_string()),
        ("$DF$", (m.d * m.f).to_string()),
        ("$HD2$", m.hd2().to_string()),
        ("$HD$", hd.to_string()),
        ("$SCALE$", m.scale().to_string()),
        ("$S$", m.s.to_string()),
        ("$D$", m.d.to_string()),
        ("$H$", m.h.to_string()),
        ("$F$", m.f.to_string()),
    ];
    let mut out = src.to_string();
    for (tok, val) in &pairs {
        out = out.replace(tok, val);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// The block ABI: 13 const + 14 mut buffers
// ---------------------------------------------------------------------------------------------

/// `kbench(x, g1, g2, wq, wk, wv, wo, w1, w3, w2, b1, rc, rs, nrm, q, k, v, sc, qh, kh, vt, ah,
/// ctx, h, f1, f3, out)`. Every Wukong variant and every peer lowers to exactly this, so the five
/// spellings and the three peer languages are called through one signature.
type BlockFn = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
);

const N_IN: usize = 13;
const N_SCRATCH: usize = 14;
/// Index of `out` within the scratch group (the buffer every checker reads).
const OUT_IX: usize = 13;

/// The 27 buffers, in ABI order. `inp` is read-only for the whole run and is built once; `scr` is
/// rewritten by every call and is what the correctness check reads back.
struct Bufs {
    inp: Vec<Vec<f32>>,
    scr: Vec<Vec<f32>>,
}

impl Bufs {
    /// Deterministic, well-conditioned data: bounded weights so no projection overflows f32 and the
    /// softmax stays in range, but with enough variation that a kernel that dropped a lane or read a
    /// stale row could not accidentally agree with the reference.
    fn new(m: Dims) -> Bufs {
        let (s, d, f, hd, hd2) = (m.s, m.d, m.f, m.hd(), m.hd2());
        // A cheap, reproducible PRNG (SplitMix64) — same values every run, no rand dependency.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // uniform in [-1, 1), then rounded to f32 so the reference and the kernels start from
            // the identical bits.
            ((z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        };
        let mut fill = |n: usize, sd: f32| -> Vec<f32> { (0..n).map(|_| next() * sd).collect() };

        // Weight scale ~ 1/sqrt(fan_in), the standard init, so activations stay O(1) through the
        // block and the f64 reference is not comparing two different kinds of overflow.
        let wq_sd = 1.0 / (d as f32).sqrt();
        let w1_sd = 1.0 / (d as f32).sqrt();
        let w2_sd = 1.0 / (f as f32).sqrt();

        let x = fill(s * d, 1.0);
        let g1 = (0..d).map(|i| 1.0 + 0.1 * ((i % 7) as f32 - 3.0)).collect();
        let g2 = (0..d).map(|i| 1.0 + 0.1 * ((i % 5) as f32 - 2.0)).collect();
        let wq = fill(d * d, wq_sd);
        let wk = fill(d * d, wq_sd);
        let wv = fill(d * d, wq_sd);
        let wo = fill(d * d, wq_sd);
        let w1 = fill(f * d, w1_sd);
        let w3 = fill(f * d, w1_sd);
        let w2 = fill(d * f, w2_sd);
        let b1 = fill(f, 0.1);
        // RoPE tables: the usual 10000^(-2t/hd) inverse frequencies.
        let mut rc = vec![0.0f32; s * hd2];
        let mut rs = vec![0.0f32; s * hd2];
        for r in 0..s {
            for t in 0..hd2 {
                let inv_freq = 1.0f64 / 10000f64.powf(2.0 * t as f64 / hd as f64);
                let ang = r as f64 * inv_freq;
                rc[r * hd2 + t] = ang.cos() as f32;
                rs[r * hd2 + t] = ang.sin() as f32;
            }
        }
        let inp = vec![x, g1, g2, wq, wk, wv, wo, w1, w3, w2, b1, rc, rs];
        assert_eq!(inp.len(), N_IN);

        let scr = vec![
            vec![0.0f32; s * d],  // nrm
            vec![0.0f32; s * d],  // q
            vec![0.0f32; s * d],  // k
            vec![0.0f32; s * d],  // v
            vec![0.0f32; s * s],  // sc
            vec![0.0f32; s * hd], // qh
            vec![0.0f32; s * hd], // kh
            vec![0.0f32; hd * s], // vt
            vec![0.0f32; s * hd], // ah
            vec![0.0f32; s * d],  // ctx
            vec![0.0f32; s * d],  // h
            vec![0.0f32; s * f],  // f1
            vec![0.0f32; s * f],  // f3
            vec![0.0f32; s * d],  // out
        ];
        assert_eq!(scr.len(), N_SCRATCH);
        Bufs { inp, scr }
    }
}

/// Call a block through the 27-pointer ABI.
///
/// ONE-LIVE-POINTER LAW (the same law [`crate::bench_wukong`] documents). Every pointer handed to
/// the callee is derived HERE, from the owning `Vec`s, inside the same borrow that is live across
/// the call. A caller that hoisted the `as_mut_ptr()`s and then also passed `&mut` to the same
/// buffers would be telling the optimizer they are `noalias` while an opaque callee writes through
/// them, and the read-back could be folded to the zeroing that preceded it — every correctness
/// check in this module would then compare all-zeros against all-zeros and pass vacuously.
///
/// # Safety
/// `f` must be a lowered `kbench` whose signature was checked to be 27 pointers returning void, and
/// `bufs` must have been built by [`Bufs::new`] with the same [`Dims`] the callee was compiled for.
unsafe fn call_block(f: BlockFn, bufs: &mut Bufs) {
    let i: Vec<*const f32> = bufs.inp.iter().map(|v| v.as_ptr()).collect();
    let o: Vec<*mut f32> = bufs.scr.iter_mut().map(|v| v.as_mut_ptr()).collect();
    f(
        i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11], i[12], o[0], o[1],
        o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11], o[12], o[13],
    );
}

// ---------------------------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------------------------

/// ONE round-local sample of one column: a single warm-up call, then the minimum of [`SAMPLES`]
/// calls. The minimum, not the mean: on a machine with an OS, background work can only ever make a
/// sample slower.
///
/// Deliberately short, and deliberately not the old two-warm-ups-plus-best-of-7. Every sample taken
/// inside one visit is drawn from the same instant of the machine's clock and thermal state, so
/// piling samples into a visit tightens the estimate of *this instant's* speed while saying nothing
/// about the drift between visits — which is the entire problem on this laptop. The dispersion this
/// module reports therefore comes from repeating the ROUND (see [`Rounds`]), and the per-visit
/// budget is spent on rounds instead. Total calls per column is `rounds * (1 + SAMPLES)`, which at
/// the default 6 rounds is 24, against the 9 the single-visit protocol used.
const SAMPLES: usize = 3;

fn time_round(mut run: impl FnMut()) -> f64 {
    // One warm-up: the column last ran a whole round ago and every other column has evicted its
    // working set since, so without this the first timed call would charge one column's cache
    // refill to it and not to the others.
    run();
    let mut best = f64::INFINITY;
    for _ in 0..SAMPLES {
        let t = Instant::now();
        run();
        let e = t.elapsed().as_secs_f64() * 1e9;
        if e < best {
            best = e;
        }
    }
    best
}

// ---------------------------------------------------------------------------------------------
// The round-interleaved measurement core, shared by all three programs
// ---------------------------------------------------------------------------------------------

/// Every column's per-round timing, plus what the machine was doing during each round.
///
/// A section builds ALL its columns first (compiling and loading is not timed and must not sit
/// between two timed columns), then hands them here. Column `i` is timed once per round; the visit
/// order rotates and alternates direction so no column keeps a slot — see [`crate::round_order`].
struct Rounds {
    /// How many rounds were run. Not derived from a vector length: `power` deliberately carries one
    /// MORE entry than there are rounds.
    rounds: usize,
    /// `ns[column][round]` — the round-local minimum for that column in that round.
    ns: Vec<Vec<f64>>,
    /// The power STATE (percentage stripped) sampled at the top of every round AND once more after
    /// the last one — `rounds + 1` entries. The closing sample is what makes the detector cover the
    /// whole section: without it a state change during the final round would go unseen, and the
    /// final round is where a long section is most likely to run the battery over a threshold.
    power: Vec<String>,
    /// The short FMA clock probe at the top of every round, GFLOP/s. A RELATIVE trace across rounds,
    /// never a roofline — see [`crate::clock_probe_gflops`].
    clock: Vec<f64>,
}

impl Rounds {
    /// Time `n` columns over `rounds` rounds. `visit(i, r)` takes exactly one round-local sample of
    /// column `i` in round `r` and returns its ns; the caller is responsible for snapshotting that
    /// column's output buffer on round 0, because only it knows which buffer that is.
    fn run(n: usize, rounds: usize, mut visit: impl FnMut(usize, usize) -> f64) -> Rounds {
        let mut ns = vec![vec![f64::NAN; rounds]; n];
        let mut power = Vec::with_capacity(rounds + 1);
        let mut clock = Vec::with_capacity(rounds);
        for r in 0..rounds {
            power.push(crate::model::power_state_key());
            clock.push(crate::clock_probe_gflops());
            for &i in &crate::round_order(n, r) {
                ns[i][r] = visit(i, r);
            }
        }
        power.push(crate::model::power_state_key());
        Rounds {
            rounds,
            ns,
            power,
            clock,
        }
    }

    /// Best observed time for a column across all rounds — the point estimate for the ms column.
    fn best(&self, i: usize) -> f64 {
        self.ns[i].iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Did the machine stay in one power state for the whole section? When it did not, every ratio
    /// in the section is replaced by `NON-REPORTABLE` rather than merely flagged: the three states
    /// (battery / AC+charging / AC+full) are three different machines, and a ratio whose two columns
    /// were measured on different machines is not a ratio.
    fn reportable(&self) -> bool {
        self.power.windows(2).all(|w| w[0] == w[1])
    }

    /// The per-round ratio `column(num) / column(den)` as a median plus the range the rounds spanned.
    fn ratio(&self, num: usize, den: usize) -> Option<crate::RatioStat> {
        crate::RatioStat::over_rounds(&self.ns[num], &self.ns[den])
    }

    /// How far the short FMA clock probe moved across the rounds, as a factor ≥ 1. This is the
    /// drift the ratios had to survive, measured rather than assumed.
    fn clock_spread(&self) -> f64 {
        let live: Vec<f64> = self.clock.iter().copied().filter(|c| *c > 0.0).collect();
        if live.len() < 2 {
            return 1.0;
        }
        let lo = live.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = live.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        hi / lo
    }
}

/// Whatever must stay alive for a built column to remain callable. Held for the whole section: with
/// round interleaving every column has to be loaded before ANY of them is timed, so nothing is
/// dropped until the last round is done.
// Both payloads are write-only by design: the fn pointer taken at build time is what gets called,
// and these exist solely so the JIT module stays mapped and the DLL stays loaded until the last
// round. Dropping either early would leave a live `Col::f` pointing at unmapped memory.
#[allow(dead_code)]
enum Keep {
    Jit(wukong_codegen_cranelift::JitModuleHandle),
    Dll(libloading::Library),
}

/// The `vs C` cell for column `i`: the word `reference` in C's own row (a row reading "1.00x faster
/// than itself" is noise), a round-formed [`crate::RatioStat`] everywhere else.
fn vs_c_cell(i: usize, c_ix: Option<usize>, rr: &Rounds, reportable: bool) -> String {
    if Some(i) == c_ix {
        return "reference".to_string();
    }
    crate::fmt_ratio_stat(c_ix.and_then(|c| rr.ratio(c, i)), reportable)
}

/// The warning that has to sit under every round-interleaved table. The `ms` column is a best-of-
/// rounds minimum and the ratio column is a median of per-round ratios, so they are NOT consistent
/// with each other by construction — and they must not be, because two columns' minima can fall in
/// different rounds and their quotient is then exactly the cross-round artifact this whole protocol
/// exists to remove. Say so where the numbers are, not in a doc comment nobody reads.
const DIVIDE_WARNING: &str =
    "  Do NOT divide two ms entries: their minima can come from DIFFERENT rounds, and that\n\
     \x20 quotient is precisely the cross-round artifact this protocol removes. The ratio column\n\
     \x20 is formed inside single rounds and is the only comparison here; the ms column is\n\
     \x20 ROUND-LOCAL context.";

/// The control ratio `C / C(twin)`, the run's own measured floor. `None` when either column failed
/// to build, in which case the run genuinely has no floor and [`print_instrument`] says so.
fn twin_ratio(labels: &[String], rr: &Rounds) -> Option<crate::RatioStat> {
    let c = ix_of(labels, "C")?;
    let t = ix_of(labels, "C(twin)")?;
    rr.ratio(c, t)
}

/// The block every round-interleaved section prints under its table: the protocol, what the
/// machine's clock did across the rounds, and the run's own measured floor.
///
/// `twin` is the `C(twin)` control's ratio against `C` — identical source, identical compiler,
/// identical flags — so its expected value is 1.00 with no language content, and what it reads away
/// from 1.00 is the instrument rather than a language. See [`PEER_SPECS`] for what that does and does
/// not bound.
fn print_instrument(rr: &Rounds, twin: Option<crate::RatioStat>, per_round_calls: usize) {
    let rounds = rr.rounds;
    println!(
        "  PROTOCOL: {rounds} rounds; every column timed once per round in a rotating,\n\
         \x20   direction-alternating order (1 warm-up + best of {SAMPLES} calls per visit,\n\
         \x20   {per_round_calls} calls per column in total). Each ratio is formed WITHIN a round;\n\
         \x20   the cell shows the median across rounds and [min-max] of the per-round ratios.\n\
         \x20   A ratio whose rounds disagree by more than {:.2}x prints INCONCLUSIVE, not a number\n\
         \x20   — its [min-max] is still shown, so a range lying entirely on one side of 1.00 still\n\
         \x20   tells you the DIRECTION; what the run failed to establish is the SIZE.",
        crate::spread_limit()
    );
    if rr.clock_spread() > 1.0 {
        println!(
            "  CLOCK TRACE across the rounds (short FMA probe, RELATIVE only — not the roofline):\n\
             \x20   {} GF/s — spread {:.2}x. This is the drift the ratios above had to survive.\n\
             \x20   It is EVIDENCE, not a correction: normalizing a per-round ratio by it cancels\n\
             \x20   exactly (both columns ran under that same clock), and normalizing the ms column\n\
             \x20   by it was measured to make the ms column WORSE. See `clock_probe_gflops`.",
            rr.clock
                .iter()
                .map(|c| format!("{c:.0}"))
                .collect::<Vec<_>>()
                .join(" -> "),
            rr.clock_spread(),
        );
    }
    match twin {
        Some(t) => println!(
            "  FLOOR (measured, this run): C(twin) is a CONTROL — the IDENTICAL C source through the\n\
             \x20   IDENTICAL compiler at the IDENTICAL flags, in a second DLL — so its ratio against C\n\
             \x20   has expected value 1.00 and no language content. It read {:.3}x (rounds\n\
             \x20   {:.3}-{:.3}, spread {:.3}x). Any \"vs C++\"/\"vs Rust\" figure in this table that is\n\
             \x20   closer to 1.00 than that is not distinguishable from the instrument.",
            t.med,
            t.lo,
            t.hi,
            t.spread(),
        ),
        None => println!("  FLOOR: C(twin) control unavailable — this run has no measured floor."),
    }
    // The control is the run's self-check, so it gets the same test as everything else. When a
    // column that CANNOT differ from `C` nevertheless fails the spread gate, the failure is the
    // instrument's, and no cross-language reading from this section should be carried anywhere —
    // including into a summary that only quotes the rows which happened to pass.
    if let Some(t) = twin {
        if !t.conclusive() {
            println!(
                "  !!! THE CONTROL ITSELF FAILED ({:.3}x spread against a true value of 1.00).\n\
                 \x20   This section did not measure anything cross-language: a column that cannot\n\
                 \x20   differ from C moved more than the limit, so any row here that did pass the\n\
                 \x20   limit passed it by luck. Re-run on AC+full, idle, and compare."
                ,
                t.spread()
            );
        }
    }
    if !rr.reportable() {
        println!(
            "  !!! POWER STATE CHANGED DURING THIS SECTION ({}). Every ratio above reads\n\
             \x20   NON-REPORTABLE: the three power states are three different machines, and a\n\
             \x20   ratio whose columns were measured on different machines is not a ratio.",
            rr.power.join(" -> ")
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Wukong compilation
// ---------------------------------------------------------------------------------------------

/// The real pipeline — parse -> sema -> mir_build -> optimize(-O3) -> Cranelift JIT — plus the
/// optimized MIR text the dispatch census is read from. `-O3` is byte-identical to `-O2` here
/// (`wukong_opt` only tests `>= 1` and `>= 2`), so the census equals `--emit=mir -O2`'s.
struct WukBlock {
    handle: wukong_codegen_cranelift::JitModuleHandle,
    ptr: *const u8,
    compile: Duration,
    census: Vec<(String, usize)>,
}

fn compile_block(src: &str, label: &str, nparams: usize) -> Option<WukBlock> {
    let t = Instant::now();
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        eprintln!("general/{label}: wukong parse error: {pd:?}");
        return None;
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        eprintln!("general/{label}: wukong sema error: {sd:?}");
        return None;
    }
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        eprintln!("general/{label}: wukong lower error: {ld:?}");
        return None;
    }
    wukong_opt::optimize(&mut program, 3);
    let mir = wukong_mir::print::print_program(&program, &interner);
    let census = kernel_census(&mir);
    let sym = interner.intern("kbench");
    if !block_abi_ok(&program, sym, label, nparams) {
        return None;
    }
    let handle = match wukong_codegen_cranelift::jit_module(&program, &mut interner) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("general/{label}: wukong codegen error: {e}");
            return None;
        }
    };
    let ptr = handle.func_ptr(sym)?;
    Some(WukBlock {
        handle,
        ptr,
        compile: t.elapsed(),
        census,
    })
}

/// Assert the lowered `kbench` really is 27 pointers -> void before anything transmutes it. A
/// mismatch here is an arity bug in a `.wk` source, and without this check it would be a silent
/// register-garbage read rather than a diagnosable failure.
fn block_abi_ok(
    program: &wukong_mir::Program,
    sym: wukong_span::Symbol,
    label: &str,
    n: usize,
) -> bool {
    let Some(f) = program.function(sym) else {
        eprintln!("general/{label}: the lowered module has no `kbench`");
        return false;
    };
    let all_ptr = f
        .params
        .iter()
        .all(|&p| matches!(f.value_type(p), wukong_mir::MirType::Ptr));
    if f.params.len() != n || !all_ptr || !matches!(f.ret, wukong_mir::MirType::Void) {
        eprintln!(
            "general/{label}: kbench lowered to {} param(s) -> {}, expected {n} pointers -> void",
            f.params.len(),
            f.ret.display(),
        );
        return false;
    }
    true
}

/// Count `wukong_*` runtime-kernel calls in optimized MIR — the dispatch census. Textually the same
/// scan as `wukongc --emit=mir -O2 prog.wk | grep -oE 'wukong_[a-z0-9_]+'`.
fn kernel_census(mir: &str) -> Vec<(String, usize)> {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for (i, _) in mir.match_indices("wukong_") {
        let rest = &mir[i..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        *counts.entry(&rest[..end]).or_insert(0) += 1;
    }
    counts.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn census_line(census: &[(String, usize)]) -> String {
    if census.is_empty() {
        return "NOTHING — every loop stays generic Cranelift code".to_string();
    }
    let total: usize = census.iter().map(|(_, n)| n).sum();
    let body = census
        .iter()
        .map(|(k, n)| {
            if *n == 1 {
                k.clone()
            } else {
                format!("{k}x{n}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{total} call site(s): {body}")
}

/// Build one Wukong column of the structure-tax table. Build ONLY — nothing is timed here, because
/// round interleaving requires every column to exist before any of them is measured, and a JIT
/// compile sitting between two timed columns is exactly the kind of gap the drift hides in.
fn build_wukong_col(src: &str, label: &str, note: &str) -> Option<Col<BlockFn>> {
    let blk = compile_block(src, label, N_IN + N_SCRATCH)?;
    // SAFETY: `block_abi_ok` just proved the lowered `kbench` is 27 pointers -> void, which is
    // exactly `BlockFn`; the handle is moved into `Keep` and outlives every call.
    let f: BlockFn = unsafe { std::mem::transmute(blk.ptr) };
    Some(Col {
        label: label.to_string(),
        note: note.to_string(),
        census: census_line(&blk.census),
        compile: blk.compile,
        f,
        _keep: Keep::Jit(blk.handle),
    })
}

// ---------------------------------------------------------------------------------------------
// Peer compilation
// ---------------------------------------------------------------------------------------------

/// Compile a peer source to a shared library and load it. Shared by every program in this module;
/// the caller does the symbol lookup, because the programs have different ABIs.
fn load_peer(
    ext: &str,
    src: &str,
    dir: &Path,
    name: &str,
    compiler: &str,
    args: &[&str],
) -> Option<(libloading::Library, Duration)> {
    let safe: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    let src_path = dir.join(format!("{safe}.{ext}"));
    let dll: PathBuf = dir.join(format!("{safe}_{ext}.dll"));
    if std::fs::write(&src_path, src).is_err() {
        eprintln!("general: could not write {}", src_path.display());
        return None;
    }
    let t = Instant::now();
    let status = Command::new(compiler)
        .args(args)
        .arg("-o")
        .arg(&dll)
        .arg(&src_path)
        .status();
    let compile = t.elapsed();
    match status {
        Ok(s) if s.success() => {}
        Ok(_) => {
            eprintln!("general: {compiler} failed to compile {name}.{ext}");
            return None;
        }
        Err(_) => {
            eprintln!("general: could not run `{compiler}` (skipping {name})");
            return None;
        }
    }
    match unsafe { libloading::Library::new(&dll) } {
        Ok(l) => Some((l, compile)),
        Err(e) => {
            eprintln!("general: load {}: {e}", dll.display());
            None
        }
    }
}

/// Compile a peer source, load it and pull out its `kbench` as `F`. Build ONLY — see
/// [`build_wukong_col`] for why nothing is timed at build time any more. One helper for all three
/// programs' ABIs, so no program can drift into a different loading path than another.
///
/// # Safety
/// `F` must be exactly the `extern "C"` signature the peer's `kbench` was written with. Every caller
/// passes the same `BlockFn`/`LossFn`/`ScanFn` the Wukong side is arity-checked against by
/// [`block_abi_ok`].
unsafe fn peer_col<F: Copy>(
    ext: &str,
    src: &str,
    dir: &Path,
    name: &str,
    compiler: &str,
    args: &[&str],
) -> Option<(F, libloading::Library, Duration)> {
    let (lib, compile) = load_peer(ext, src, dir, name, compiler, args)?;
    let f: F = match lib.get::<F>(b"kbench\0") {
        Ok(s) => *s,
        Err(e) => {
            eprintln!("general: symbol kbench in {name}.{ext}: {e}");
            return None;
        }
    };
    Some((f, lib, compile))
}

/// The five peer columns every program in this module carries, in build order.
///
/// `C(twin)` is the IDENTICAL C source through the IDENTICAL compiler at the IDENTICAL flags,
/// written to a second DLL. It is a CONTROL, not a peer: it changes no other column's flags and
/// takes nothing away from anyone. What it measures is everything that separates two columns
/// *except* the language — timing noise, scheduling, thermal drift between two slots of the same
/// round, and code/data placement luck (the two DLLs are separate images and can land on different
/// alignments). Its ratio against `C` therefore has an expected value of 1.00 and no language
/// content whatsoever, so whatever it reads away from 1.00 bounds from below what any *other*
/// column's departure from 1.00 has to beat before it means anything.
///
/// Note the honest limit of the control: it bounds the confounds it shares with the real columns.
/// It cannot bound a confound unique to one language's column (a peer that happens to alias
/// differently, say). It is a floor on believability, never a certificate.
const RUSTC_ARGS: [&str; 3] = ["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"];
const HONEST_FLAGS: [&str; 4] = ["-O3", "-march=native", "-ffp-contract=fast", "-shared"];
const FASTM_FLAGS: [&str; 5] = [
    "-O3",
    "-march=native",
    "-ffp-contract=fast",
    "-ffast-math",
    "-shared",
];

/// `(label, extension, compiler-selector, flags, dll-name-suffix)` for the peer columns. The
/// selector is resolved by the caller because `cc`/`cxx` arrive as parameters.
enum PeerCc {
    Cc,
    Cxx,
    Rustc,
}

struct PeerSpec {
    label: &'static str,
    ext: &'static str,
    cc: PeerCc,
    args: &'static [&'static str],
    /// Distinguishes the two columns built from the identical C source at identical flags.
    suffix: &'static str,
}

const PEER_SPECS: [PeerSpec; 5] = [
    PeerSpec { label: "C", ext: "c", cc: PeerCc::Cc, args: &HONEST_FLAGS, suffix: "" },
    PeerSpec { label: "C(twin)", ext: "c", cc: PeerCc::Cc, args: &HONEST_FLAGS, suffix: "_twin" },
    PeerSpec { label: "C(fast)", ext: "c", cc: PeerCc::Cc, args: &FASTM_FLAGS, suffix: "_fast" },
    PeerSpec { label: "C++", ext: "cpp", cc: PeerCc::Cxx, args: &HONEST_FLAGS, suffix: "" },
    PeerSpec { label: "Rust", ext: "rs", cc: PeerCc::Rustc, args: &RUSTC_ARGS, suffix: "" },
];

impl PeerSpec {
    fn compiler<'a>(&self, cc: &'a str, cxx: &'a str) -> &'a str {
        match self.cc {
            PeerCc::Cc => cc,
            PeerCc::Cxx => cxx,
            PeerCc::Rustc => "rustc",
        }
    }
    /// The C source for the C/C++/C(fast)/C(twin) columns, the Rust source for the Rust column.
    fn source<'a>(&self, c_src: &'a str, rs_src: &'a str) -> &'a str {
        match self.cc {
            PeerCc::Rustc => rs_src,
            _ => c_src,
        }
    }
    fn note(&self) -> &'static str {
        if self.suffix == "_twin" {
            "CONTROL: identical source+compiler+flags as C — the run's own floor"
        } else {
            "peer, written once, competently"
        }
    }
}

/// Build every peer column for one program, in [`PEER_SPECS`] order. `stem` is the DLL-name stem
/// (`general_block` / `general_loss` / `general_scan`); the spec's suffix is what keeps `C` and
/// `C(twin)` in two separate images. Build ONLY — see [`build_wukong_col`].
///
/// One helper for all three programs, so no program can drift into a different peer set, a
/// different flag set or a different loading path than another.
///
/// # Safety
/// `F` must be exactly the `extern "C"` signature the program's peer `kbench` is declared with.
unsafe fn build_peer_cols<F: Copy>(
    stem: &str,
    c_src: &str,
    rs_src: &str,
    dir: &Path,
    cc: &str,
    cxx: &str,
) -> Vec<Col<F>> {
    let mut out = Vec::new();
    for spec in &PEER_SPECS {
        let name = format!("{stem}{}", spec.suffix);
        // SAFETY: forwarded from this function's own contract.
        let built = unsafe {
            peer_col::<F>(
                spec.ext,
                spec.source(c_src, rs_src),
                dir,
                &name,
                spec.compiler(cc, cxx),
                spec.args,
            )
        };
        match built {
            Some((f, lib, compile)) => out.push(Col {
                label: spec.label.to_string(),
                note: spec.note().to_string(),
                census: "-".to_string(),
                compile,
                f,
                _keep: Keep::Dll(lib),
            }),
            None => println!("  ! {} peer unavailable — column dropped", spec.label),
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// The f64 oracle
// ---------------------------------------------------------------------------------------------

/// The whole block recomputed in f64, from the same input bits, by a straight scalar transcription
/// of the specification. This is the ONLY oracle: the interpreter cannot hold these buffers (its
/// `Value` enum costs ~24 bytes per f32 element), and every f32 implementation here — Wukong's
/// dispatched AVX2 kernels included — is being asked whether it still computes the block.
///
/// Returns the full `out` buffer. The check downstream is per-lane; a reduction (a sum, a norm)
/// would let a buffer that is wrong in half its lanes pass on cancellation.
fn reference_f64(m: Dims, inp: &[Vec<f32>]) -> Vec<f64> {
    let (s, d, f, hd, hd2) = (m.s, m.d, m.f, m.hd(), m.hd2());
    let g = |ix: usize| -> Vec<f64> { inp[ix].iter().map(|&v| v as f64).collect() };
    let (x, g1, g2) = (g(0), g(1), g(2));
    let (wq, wk, wv, wo) = (g(3), g(4), g(5), g(6));
    let (w1, w3, w2, b1) = (g(7), g(8), g(9), g(10));
    let (rc, rs) = (g(11), g(12));
    let scale = m.scale();

    let mut nrm = vec![0.0f64; s * d];
    let mut q = vec![0.0f64; s * d];
    let mut k = vec![0.0f64; s * d];
    let mut v = vec![0.0f64; s * d];
    let mut sc = vec![0.0f64; s * s];
    let mut qh = vec![0.0f64; s * hd];
    let mut kh = vec![0.0f64; s * hd];
    let mut vt = vec![0.0f64; hd * s];
    let mut ah = vec![0.0f64; s * hd];
    let mut ctx = vec![0.0f64; s * d];
    let mut h = vec![0.0f64; s * d];
    let mut f1 = vec![0.0f64; s * f];
    let mut out = vec![0.0f64; s * d];

    let norm = |src: &[f64], gam: &[f64], dst: &mut [f64]| {
        for r in 0..s {
            let rb = r * d;
            let mut ss = 0.0f64;
            for i in 0..d {
                ss += src[rb + i] * src[rb + i];
            }
            let inv = 1.0 / (ss / d as f64 + 1.0e-5).sqrt();
            for i in 0..d {
                dst[rb + i] = src[rb + i] * inv * gam[i];
            }
        }
    };
    let linear = |a: &[f64], w: &[f64], c: &mut [f64], mm: usize, n: usize, kk: usize| {
        for i in 0..mm {
            let (ib, ob) = (i * kk, i * n);
            for j in 0..n {
                let jb = j * kk;
                let mut acc = 0.0f64;
                for p in 0..kk {
                    acc += a[ib + p] * w[jb + p];
                }
                c[ob + j] = acc;
            }
        }
    };

    norm(&x, &g1, &mut nrm);
    linear(&nrm, &wq, &mut q, s, d, d);
    linear(&nrm, &wk, &mut k, s, d, d);
    linear(&nrm, &wv, &mut v, s, d, d);

    for r in 0..s {
        let (tb, rb) = (r * hd2, r * d);
        for e in 0..m.h {
            let hb = rb + e * hd;
            for t in 0..hd2 {
                let (cc, sn) = (rc[tb + t], rs[tb + t]);
                let (q0, q1) = (q[hb + t], q[hb + t + hd2]);
                q[hb + t] = q0 * cc - q1 * sn;
                q[hb + t + hd2] = q0 * sn + q1 * cc;
                let (k0, k1) = (k[hb + t], k[hb + t + hd2]);
                k[hb + t] = k0 * cc - k1 * sn;
                k[hb + t + hd2] = k0 * sn + k1 * cc;
            }
        }
    }

    for e in 0..m.h {
        let ho = e * hd;
        for i in 0..s {
            let (src, dst) = (i * d + ho, i * hd);
            for p in 0..hd {
                qh[dst + p] = q[src + p];
                kh[dst + p] = k[src + p];
            }
        }
        for j in 0..hd {
            let dst = j * s;
            for p in 0..s {
                vt[dst + p] = v[p * d + ho + j];
            }
        }
        for i in 0..s {
            let (ib, so) = (i * hd, i * s);
            for j in 0..s {
                let jb = j * hd;
                let mut acc = 0.0f64;
                for p in 0..hd {
                    acc += qh[ib + p] * kh[jb + p];
                }
                sc[so + j] = scale * acc;
            }
        }
        for i in 0..s {
            let so = i * s;
            for j in 0..s {
                if j > i {
                    sc[so + j] = -1.0e30;
                }
            }
        }
        for r in 0..s {
            let so = r * s;
            let mut mx = sc[so];
            for i in 0..s {
                if sc[so + i] > mx {
                    mx = sc[so + i];
                }
            }
            for i in 0..s {
                sc[so + i] = (sc[so + i] - mx).exp();
            }
            let mut sm = 0.0f64;
            for i in 0..s {
                sm += sc[so + i];
            }
            let inv = 1.0 / sm;
            for i in 0..s {
                sc[so + i] *= inv;
            }
        }
        for i in 0..s {
            let (so, ib) = (i * s, i * hd);
            for j in 0..hd {
                let jb = j * s;
                let mut acc = 0.0f64;
                for p in 0..s {
                    acc += sc[so + p] * vt[jb + p];
                }
                ah[ib + j] = acc;
            }
        }
        for i in 0..s {
            let (dst, ib) = (i * d + ho, i * hd);
            for j in 0..hd {
                ctx[dst + j] = ah[ib + j];
            }
        }
    }

    linear(&ctx, &wo, &mut h, s, d, d);
    for i in 0..s * d {
        h[i] += x[i];
    }
    norm(&h, &g2, &mut nrm);

    let mut f3 = vec![0.0f64; s * f];
    linear(&nrm, &w1, &mut f1, s, f, d);
    linear(&nrm, &w3, &mut f3, s, f, d);
    for i in 0..s {
        for j in 0..f {
            let z = f1[i * f + j] + b1[j];
            f1[i * f + j] = z * (1.0 / (1.0 + (-z).exp())) * f3[i * f + j];
        }
    }
    linear(&f1, &w2, &mut out, s, d, f);
    for i in 0..s * d {
        out[i] += h[i];
    }
    out
}

/// Worst per-lane relative deviation of `got` from the f64 reference, with the lane it happened at.
/// Relative to `max(|ref|, 1e-3)` so a lane that is legitimately near zero cannot manufacture a huge
/// ratio out of an absolute error that is nothing.
fn max_rel_vs_ref(got: &[f32], want: &[f64]) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
        let e = (a as f64 - b).abs() / b.abs().max(1.0e-3);
        if e > worst {
            worst = e;
            at = i;
        }
    }
    (worst, at)
}

/// Worst relative deviation between two f32 buffers, plus whether they are bit-identical.
fn max_rel_pair(a: &[f32], b: &[f32]) -> (f64, bool) {
    let mut worst = 0.0f64;
    let mut same = a.len() == b.len();
    for (&p, &q) in a.iter().zip(b.iter()) {
        if p.to_bits() != q.to_bits() {
            same = false;
        }
        let e = (p as f64 - q as f64).abs() / (q as f64).abs().max(1.0e-3);
        if e > worst {
            worst = e;
        }
    }
    (worst, same)
}

// ---------------------------------------------------------------------------------------------
// Program 1: the structure tax
// ---------------------------------------------------------------------------------------------

/// The five spellings, in the order they are reported.
const VARIANTS: [(&str, &str, &str); 5] = [
    (
        "a-recognizer",
        include_str!("../wk/general/block_a_recognizer.wk"),
        "monolithic, recognizer dialect",
    ),
    (
        "b-natural",
        include_str!("../wk/general/block_b_natural.wk"),
        "monolithic, natural spelling",
    ),
    (
        "c-helpers",
        include_str!("../wk/general/block_c_helpers.wk"),
        "factored into []f32 helpers",
    ),
    (
        "d-struct",
        include_str!("../wk/general/block_d_struct.wk"),
        "(a) with weights in a struct",
    ),
    (
        "e-tensor",
        include_str!("../wk/general/block_e_tensor.wk"),
        "(a) in shape-typed Tensor[..]",
    ),
];

const PEER_C: &str = include_str!("../wk/general/peer_block.c");
const PEER_RS: &str = include_str!("../wk/general/peer_block.rs");

/// One built, callable column of a round-interleaved table — a Wukong spelling or a peer, held in
/// ONE type so the round schedule cannot tell them apart and cannot favour either. `F` is the
/// program's `extern "C"` ABI: [`BlockFn`], [`LossFn`] or [`ScanFn`].
struct Col<F> {
    label: String,
    note: String,
    census: String,
    compile: Duration,
    f: F,
    /// The JIT module or DLL that keeps `f` callable, held until the last round is done.
    _keep: Keep,
}

/// The column labels, in build order — what [`Rounds`] indices mean.
fn labels_of<F>(cols: &[Col<F>]) -> Vec<String> {
    cols.iter().map(|c| c.label.clone()).collect()
}

/// Position of a column by label.
fn ix_of(labels: &[String], name: &str) -> Option<usize> {
    labels.iter().position(|l| l == name)
}

/// Entry point: every program in the general-code suite, in priority order.
pub(crate) fn bench_general(cc: &str, cxx: &str, dir: &Path) {
    bench_ablation();
    println!();
    bench_structure_tax(cc, cxx, dir);
    println!();
    bench_focal_loss(cc, cxx, dir);
    println!();
    bench_scan(cc, cxx, dir);
}

fn bench_structure_tax(cc: &str, cxx: &str, dir: &Path) {
    let m = Dims::from_env();
    if let Err(e) = m.ok() {
        println!("general: bad dimensions ({e}) — skipping");
        return;
    }
    let hd = m.hd();

    println!("================================================================================");
    println!("GENERAL-CODE BENCHMARK — Wukong outside the recognizer dialect");
    println!("================================================================================");
    println!(
        "Program 1: STRUCTURE TAX. One pre-norm transformer block (RMSNorm -> RoPE -> causal MHA\n\
         -> RMSNorm -> SwiGLU MLP) written five ways, all computing the same f32 arithmetic in the\n\
         same order. S={} D={} H={} (head dim {hd}) F={} — {:.1} MFLOP/forward, {:.1} MB live.",
        m.s,
        m.d,
        m.h,
        m.f,
        m.flops() / 1e6,
        total_bytes(m) as f64 / (1024.0 * 1024.0),
    );
    println!("  {}", crate::model::power_status_line());
    println!();

    let mut bufs = Bufs::new(m);

    // ---- the oracle, first: everything below is checked against it ----
    let t = Instant::now();
    let reference = reference_f64(m, &bufs.inp);
    println!(
        "  f64 scalar reference computed in {:.2} s ({} lanes, every one checked below)",
        t.elapsed().as_secs_f64(),
        reference.len()
    );

    // ---- PHASE 1: BUILD every column. Nothing is timed here. ----
    // Until 2026-08-06 this section timed all five Wukong spellings to completion and only then
    // built and timed the peers, one language after another. Two consequences, both systematic and
    // neither visible in the output: any drift across the section was charged to whichever column
    // ran last (always Rust), and the Wukong columns always got the earliest, coolest slots — a bias
    // in Wukong's own favour, in Wukong's own benchmark. Building everything first and putting all
    // ten columns into ONE rotating round schedule removes both.
    let mut cols: Vec<Col<BlockFn>> = Vec::new();
    for (label, src, note) in VARIANTS {
        let text = subst(src, m);
        // Dump the exact compiled source so `wukongc --emit=mir -O2 <file>` reproduces the census.
        let _ = std::fs::write(dir.join(format!("general_block_{label}.wk")), &text);
        match build_wukong_col(&text, label, note) {
            Some(c) => cols.push(c),
            None => println!("  ! {label}: did not compile — row dropped"),
        }
    }
    let n_wuk = cols.len();

    let c_src = subst(PEER_C, m);
    let rs_src = subst(PEER_RS, m);
    // SAFETY: every peer source in this program declares `kbench` with the 27-pointer block ABI,
    // which is exactly `BlockFn`; the Wukong side is checked against the same arity by
    // `block_abi_ok`.
    cols.extend(unsafe {
        build_peer_cols::<BlockFn>("general_block", &c_src, &rs_src, dir, cc, cxx)
    });
    if cols.is_empty() {
        println!("  ! no column built — section skipped");
        return;
    }
    let labels = labels_of(&cols);

    // ---- PHASE 2: ROUNDS. Every column once per round, rotating and alternating direction. ----
    let rounds = crate::gen_rounds();
    let mut outs: Vec<Vec<f32>> = vec![Vec::new(); cols.len()];
    let rr = Rounds::run(cols.len(), rounds, |i, r| {
        let f = cols[i].f;
        bufs.scr[OUT_IX].iter_mut().for_each(|v| *v = 0.0);
        // SAFETY: `f` came from `build_wukong_col` (arity-checked) or `peer_col` over a source that
        // declares the 27-pointer ABI; `bufs` was built by `Bufs::new(m)`, the same `m` every column
        // was compiled for. The one-live-pointer law is obeyed inside `call_block`.
        let ns = time_round(|| unsafe { call_block(f, &mut bufs) });
        if r == 0 {
            // The peers and the JIT are deterministic, so one snapshot is the whole story; taking it
            // on round 0 also keeps the later rounds pure timing.
            outs[i] = bufs.scr[OUT_IX].clone();
        }
        ns
    });

    // ---- PHASE 3: report ----
    let c_ix = ix_of(&labels, "C");
    let reportable = rr.reportable();

    println!();
    println!(
        "  {:<14} {:>10} {:>9}  {:<28}  {}",
        "variant", "ms/fwd", "GF/s*", "vs C (median [min-max])", "spelling"
    );
    let rule = "-".repeat(110);
    println!("  {rule}");
    for (i, col) in cols.iter().enumerate() {
        if i == n_wuk {
            println!("  {rule}");
        }
        let best = rr.best(i);
        println!(
            "  {:<14} {:>10.2} {:>9.1}  {:<28}  {}",
            col.label,
            best / 1e6,
            m.flops() / best,
            vs_c_cell(i, c_ix, &rr, reportable),
            col.note,
        );
    }
    println!(
        "  * ms/fwd is the best of all {rounds} rounds; GF/s is derived from it. This laptop's clock\n\
         \x20   swings ~3x with power and thermal state, so no absolute figure here is reportable."
    );
    println!("{DIVIDE_WARNING}");
    println!();
    print_instrument(&rr, twin_ratio(&labels, &rr), rounds * (1 + SAMPLES));
    println!();
    print_round_trace(&labels, &rr);

    // ---- dispatch census ----
    println!();
    println!("  DISPATCH CENSUS (= wukongc --emit=mir -O2 <src> | grep -oE 'wukong_[a-z0-9_]+')");
    for col in cols.iter().take(n_wuk) {
        println!("    {:<14} {}", col.label, col.census);
    }
    println!(
        "    sources written to {} — rerun the grep against the real binary to confirm.",
        dir.display()
    );

    // ---- correctness ----
    println!();
    println!("  CORRECTNESS vs the f64 scalar reference (worst of all {} lanes)", reference.len());
    for (i, col) in cols.iter().enumerate() {
        let (dev, at) = max_rel_vs_ref(&outs[i], &reference);
        println!("    {:<14} max rel dev {dev:.3e} at lane {at}", col.label);
    }
    if let Some(base_ix) = ix_of(&labels, "b-natural") {
        println!();
        println!("  AGREEMENT between spellings (vs b-natural, the fully-scalar one)");
        for (i, col) in cols.iter().enumerate().take(n_wuk) {
            if i == base_ix {
                continue;
            }
            let (dev, same) = max_rel_pair(&outs[i], &outs[base_ix]);
            println!(
                "    {:<14} {}",
                col.label,
                if same {
                    "bit-identical".to_string()
                } else {
                    format!("max rel {dev:.3e} (kernel reassociation, not a different program)")
                }
            );
        }
        // Which spellings produce byte-identical buffers: the sharper statement, because it says
        // exactly which pairs the compiler treated as the same program. Two variants land in one
        // class iff every one of their lanes agrees to the bit.
        let mut classes: Vec<Vec<&str>> = Vec::new();
        for i in 0..n_wuk {
            match classes.iter_mut().find(|c| {
                let rep = cols.iter().position(|x| x.label == c[0]).unwrap();
                outs[rep].iter().zip(outs[i].iter()).all(|(a, b)| a.to_bits() == b.to_bits())
            }) {
                Some(c) => c.push(&cols[i].label),
                None => classes.push(vec![&cols[i].label]),
            }
        }
        println!(
            "    bit-identical classes: {}",
            classes
                .iter()
                .map(|c| format!("{{{}}}", c.join(", ")))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }

    // ---- the headline ----
    //
    // Per-round max/min over the five spellings re-selects which columns are the extremes every
    // round, and an extreme-value statistic is noisier than a fixed pair — so the alternative was
    // measured before settling here: pinning the pair by its best-of-rounds ranking and forming
    // THAT ratio per round gave round-to-round spreads of 2.39x / 2.61x / 2.71x / 2.99x against this
    // estimator's 2.23x / 2.26x / 2.77x / 1.46x on the same four runs. Fixing the pair is no better
    // and mostly worse, so the instability is the machine, not the estimator. Do not "fix" it.
    //
    // The structure tax is a WUKONG-INTERNAL ratio, so it is computed the same round-local way as
    // every other ratio here: within each round, the slowest spelling over the fastest. It is
    // subject to the same INCONCLUSIVE rule and the same power invalidation — being internal buys it
    // no exemption, because the five spellings have very different compute/memory mixes (b-natural
    // is scalar where a-recognizer is a dispatched AVX2 kernel) and a clock change is therefore not
    // a common factor that cancels.
    //
    // That is not a hypothetical, and it qualifies the project's "Wukong-internal ratios ARE usable"
    // rule. Four runs of the old harness published a structure tax of 6.15x / 8.33x / 7.64x / 7.79x;
    // the same four programs under this protocol found the per-round tax moving 3.5x-11.0x WITHIN a
    // single run. The tax is comfortably real (every round of every run exceeded 3.4x) but the
    // two-decimal figure was not, and it is an internal ratio. "Internal" buys immunity from a
    // uniform clock scaling, not from a clock change that hits a scalar loop and a dispatched AVX2
    // kernel differently.
    println!();
    if n_wuk > 0 {
        let per_round_max: Vec<f64> = (0..rounds)
            .map(|r| (0..n_wuk).map(|i| rr.ns[i][r]).fold(f64::NEG_INFINITY, f64::max))
            .collect();
        let per_round_min: Vec<f64> = (0..rounds)
            .map(|r| (0..n_wuk).map(|i| rr.ns[i][r]).fold(f64::INFINITY, f64::min))
            .collect();
        let tax = crate::RatioStat::over_rounds(&per_round_max, &per_round_min);
        let fastest = (0..n_wuk).min_by(|&a, &b| rr.best(a).total_cmp(&rr.best(b))).unwrap();
        let slowest = (0..n_wuk).max_by(|&a, &b| rr.best(a).total_cmp(&rr.best(b))).unwrap();
        match tax {
            Some(t) if reportable && t.conclusive() => println!(
                "  ==> STRUCTURE TAX: {:.2}x spread across five spellings of ONE block\n      \
                 (per-round spread {:.2}-{:.2} over {rounds} rounds; fastest {} at {:.2} ms,\n      \
                 slowest {} at {:.2} ms). gcc compiles ONE source; its spread here is 1.0 by\n      \
                 construction.",
                t.med,
                t.lo,
                t.hi,
                cols[fastest].label,
                rr.best(fastest) / 1e6,
                cols[slowest].label,
                rr.best(slowest) / 1e6,
            ),
            Some(t) => println!(
                "  ==> STRUCTURE TAX: {} — per-round spread {:.2}-{:.2} over {rounds} rounds\n      \
                 (fastest {}, slowest {}).",
                if reportable { "INCONCLUSIVE" } else { "NON-REPORTABLE (power state changed)" },
                t.lo,
                t.hi,
                cols[fastest].label,
                cols[slowest].label,
            ),
            None => println!("  ==> STRUCTURE TAX: not measurable (a round produced no timing)."),
        }
    }
    println!();
    println!(
        "  compile time: {}",
        cols.iter()
            .take(n_wuk)
            .map(|c| format!("{} {:.0}ms", c.label, c.compile.as_secs_f64() * 1e3))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  {}", crate::model::power_status_line());
}

/// The raw per-round timings for every column, so the whole table above can be re-derived — and
/// disagreed with — from the printed log rather than taken on trust. Cheap: `columns x rounds`
/// numbers.
fn print_round_trace(labels: &[String], rr: &Rounds) {
    println!("  PER-ROUND ns (raw; every number above is derived from this table)");
    print!("    {:<14}", "round");
    for r in 0..rr.rounds {
        print!(" {:>10}", r + 1);
    }
    println!();
    for (i, label) in labels.iter().enumerate() {
        print!("    {label:<14}");
        for r in 0..rr.rounds {
            print!(" {:>10.3}", rr.ns[i][r] / 1e6);
        }
        println!();
    }
    println!(
        "    (ms per call; power state sampled {} times, before each round and after the last: {})",
        rr.power.len(),
        rr.power.join(", ")
    );
}

// `fmt_ratio` (a bare "N.NNx faster" with no dispersion) and `peer_short` are gone: every ratio in
// this module is now a `crate::RatioStat` rendered by `crate::fmt_ratio_stat`, which cannot print a
// point estimate without the range its rounds spanned, and every column carries its own short label.

fn total_bytes(m: Dims) -> usize {
    let hd = m.hd();
    let n = m.s * m.d * 8       // x, nrm, q, k, v, ctx, h, out
        + m.s * m.s             // sc
        + 4 * m.s * hd          // qh, kh, vt, ah
        + 4 * m.d * m.d         // wq, wk, wv, wo
        + 3 * m.d * m.f         // w1, w3, w2
        + 2 * m.s * m.f         // f1, f3
        + 2 * m.d + m.f         // g1, g2, b1
        + 2 * m.s * m.hd2(); // rc, rs
    n * 4
}

// ---------------------------------------------------------------------------------------------
// Program 2: a fused CUSTOM LOSS — focal + label smoothing + class weights, forward and backward
// ---------------------------------------------------------------------------------------------

/// Loss dimensions. `R` rows of `C` logits; `eps` is the label-smoothing mass.
///
/// `eps` defaults to 0.125 rather than the conventional 0.1 for a measurement reason: with a
/// power-of-two `C`, both smoothed targets (`eps/C` and `1-eps+eps/C`) are then EXACTLY representable
/// in f32 and are substituted into the Wukong, C and Rust sources as one short decimal literal each.
/// A literal that each front end rounded slightly differently would show up in the deviation column
/// as if one of the compilers had miscomputed the loss.
#[derive(Clone, Copy)]
struct LossDims {
    r: usize,
    c: usize,
    eps: f32,
}

impl LossDims {
    fn from_env() -> LossDims {
        let get = |k: &str, dflt: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(dflt)
        };
        LossDims {
            r: get("XBENCH_GEN_R", 8192),
            c: get("XBENCH_GEN_NC", 1024),
            eps: 0.125,
        }
    }
    /// Smoothed target mass on a non-target class.
    fn qo(&self) -> f32 {
        self.eps / self.c as f32
    }
    /// Smoothed target mass on the target class.
    fn qt(&self) -> f32 {
        1.0 - self.eps + self.qo()
    }
}

fn subst_loss(src: &str, m: LossDims) -> String {
    // Rust's `{}` for floats is shortest-round-trip and never switches to scientific notation, so
    // the literal it emits parses back to the identical f32 in Wukong, C and Rust alike.
    let pairs: [(&str, String); 5] = [
        ("$RC$", (m.r * m.c).to_string()),
        ("$QT$", m.qt().to_string()),
        ("$QO$", m.qo().to_string()),
        ("$R$", m.r.to_string()),
        ("$C$", m.c.to_string()),
    ];
    let mut out = src.to_string();
    for (tok, val) in &pairs {
        out = out.replace(tok, val);
    }
    out
}

/// `kbench(z, alpha, tgt, p, loss, dz)`.
type LossFn = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const i32,
    *mut f32,
    *mut f32,
    *mut f32,
);

const LOSS_PARAMS: usize = 6;

struct LossBufs {
    z: Vec<f32>,
    alpha: Vec<f32>,
    tgt: Vec<i32>,
    p: Vec<f32>,
    loss: Vec<f32>,
    dz: Vec<f32>,
}

impl LossBufs {
    fn new(m: LossDims) -> LossBufs {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            z
        };
        let unit = |u: u64| (u >> 11) as f64 / (1u64 << 53) as f64;
        // Logits in [-3, 3): wide enough that the softmax has real dynamic range (so the stable-max
        // subtraction and the 1/p term in the gradient are both exercised) and narrow enough that
        // no p underflows to a denormal, which would make the f64 reference and the f32 kernels
        // disagree about something that is not the compiler's doing.
        let z: Vec<f32> = (0..m.r * m.c).map(|_| (unit(next()) * 6.0 - 3.0) as f32).collect();
        // Class weights around 1, the usual imbalance correction.
        let alpha: Vec<f32> = (0..m.c).map(|_| (0.5 + unit(next())) as f32).collect();
        let tgt: Vec<i32> = (0..m.r).map(|_| (unit(next()) * m.c as f64) as i32 % m.c as i32).collect();
        LossBufs {
            z,
            alpha,
            tgt,
            p: vec![0.0; m.r * m.c],
            loss: vec![0.0; m.r],
            dz: vec![0.0; m.r * m.c],
        }
    }
}

/// See [`call_block`] for the one-live-pointer law this obeys.
///
/// # Safety
/// `f` must be a lowered `kbench` checked to be 6 pointers returning void, and `b` must have been
/// built by [`LossBufs::new`] with the same [`LossDims`].
unsafe fn call_loss(f: LossFn, b: &mut LossBufs) {
    f(
        b.z.as_ptr(),
        b.alpha.as_ptr(),
        b.tgt.as_ptr(),
        b.p.as_mut_ptr(),
        b.loss.as_mut_ptr(),
        b.dz.as_mut_ptr(),
    );
}

/// The loss and its gradient recomputed in f64 by a straight transcription of the specification.
/// Returns `(loss, dz)`, both checked lane by lane — a mean over the loss vector would hide a row
/// whose gradient is entirely wrong.
fn reference_loss_f64(m: LossDims, b: &LossBufs) -> (Vec<f64>, Vec<f64>) {
    let (r, c) = (m.r, m.c);
    let (qt, qo) = (m.qt() as f64, m.qo() as f64);
    let mut loss = vec![0.0f64; r];
    let mut dz = vec![0.0f64; r * c];
    let mut p = vec![0.0f64; c];
    for row in 0..r {
        let rb = row * c;
        let mut mx = b.z[rb] as f64;
        for j in 0..c {
            let v = b.z[rb + j] as f64;
            if v > mx {
                mx = v;
            }
        }
        let mut sm = 0.0f64;
        for j in 0..c {
            let e = (b.z[rb + j] as f64 - mx).exp();
            p[j] = e;
            sm += e;
        }
        let inv = 1.0 / sm;
        for j in 0..c {
            p[j] *= inv;
        }
        let t = b.tgt[row];
        let mut lr = 0.0f64;
        let mut sg = 0.0f64;
        for j in 0..c {
            let q = if j as i32 == t { qt } else { qo };
            let pc = p[j];
            let om = 1.0 - pc;
            let lg = pc.max(1.0e-30).ln();
            let aq = b.alpha[j] as f64 * q;
            lr -= aq * om * om * lg;
            let gc = aq * (2.0 * om * lg - om * om / pc);
            dz[rb + j] = gc;
            sg += gc * pc;
        }
        loss[row] = lr;
        for j in 0..c {
            dz[rb + j] = p[j] * (dz[rb + j] - sg);
        }
    }
    (loss, dz)
}

const LOSS_WK: &str = include_str!("../wk/general/loss_focal.wk");
const LOSS_C: &str = include_str!("../wk/general/peer_loss.c");
const LOSS_RS: &str = include_str!("../wk/general/peer_loss.rs");

fn bench_focal_loss(cc: &str, cxx: &str, dir: &Path) {
    let m = LossDims::from_env();
    let bytes = (2 * m.r * m.c + m.r * m.c) * 4;
    println!("================================================================================");
    println!(
        "Program 2: FUSED CUSTOM LOSS — focal (gamma=2) + label smoothing (eps={}) + per-class\n\
         weights, forward AND a hand-written backward, over logits [{}, {}] ({:.1} MB touched).\n\
         Every loss Wukong has a kernel for is a pre-baked pattern; a loss invented after the\n\
         recognizer table was written cannot be in it.",
        m.eps,
        m.r,
        m.c,
        bytes as f64 / (1024.0 * 1024.0),
    );
    println!("  {}", crate::model::power_status_line());

    let mut b = LossBufs::new(m);
    let t = Instant::now();
    let (ref_loss, ref_dz) = reference_loss_f64(m, &b);
    println!(
        "  f64 scalar reference computed in {:.2} s ({} gradient lanes + {} loss lanes, all checked)",
        t.elapsed().as_secs_f64(),
        ref_dz.len(),
        ref_loss.len()
    );

    // ---- PHASE 1: BUILD every column. Nothing is timed here. ----
    // Until 2026-08-06 this section timed Wukong to completion, then built and timed the peers.
    // Building sat between timed columns and section-wide drift was charged to whoever ran last;
    // the two-pass alternation added at 17ab8d0 removed the *ordering* bias but still could not
    // report how far the two passes had disagreed. Everything is built first now and all six
    // columns go into ONE rotating round schedule, so every ratio is formed inside a single round.
    let mut cols: Vec<Col<LossFn>> = Vec::new();
    let wk_src = subst_loss(LOSS_WK, m);
    let path = dir.join("general_loss_focal.wk");
    let _ = std::fs::write(&path, &wk_src);
    match compile_block(&wk_src, "loss", LOSS_PARAMS) {
        Some(blk) => {
            // SAFETY: `block_abi_ok` proved the lowered `kbench` is 6 pointers -> void = `LossFn`;
            // the handle is moved into `Keep` and outlives every call.
            let f: LossFn = unsafe { std::mem::transmute(blk.ptr) };
            cols.push(Col {
                label: "Wukong".to_string(),
                note: "the .wk source, through the real pipeline".to_string(),
                census: census_line(&blk.census),
                compile: blk.compile,
                f,
                _keep: Keep::Jit(blk.handle),
            });
        }
        None => println!("  ! Wukong loss did not compile — row dropped"),
    }
    let n_wuk = cols.len();

    let c_src = subst_loss(LOSS_C, m);
    let rs_src = subst_loss(LOSS_RS, m);
    // SAFETY: every loss peer declares `kbench` with the 6-pointer loss ABI, which is exactly
    // `LossFn`; the Wukong side is arity-checked against `LOSS_PARAMS` by `block_abi_ok`.
    cols.extend(unsafe { build_peer_cols::<LossFn>("general_loss", &c_src, &rs_src, dir, cc, cxx) });
    if cols.is_empty() {
        println!("  ! no column built — section skipped");
        return;
    }
    let labels = labels_of(&cols);

    // ---- PHASE 2: ROUNDS ----
    let rounds = crate::gen_rounds();
    // (dz deviation, its lane, loss deviation) — recorded on round 0; every column here is
    // deterministic, so the later rounds cannot change them.
    let mut devs = vec![(0.0f64, 0usize, 0.0f64); cols.len()];
    let rr = Rounds::run(cols.len(), rounds, |i, r| {
        let f = cols[i].f;
        b.dz.iter_mut().for_each(|v| *v = 0.0);
        // SAFETY: `f` is the 6-pointer `kbench` of column `i`; `b` was built by `LossBufs::new(m)`,
        // the same `m` every column was compiled for.
        let ns = time_round(|| unsafe { call_loss(f, &mut b) });
        if r == 0 {
            let (dev, at) = max_rel_vs_ref(&b.dz, &ref_dz);
            let (ldev, _) = max_rel_vs_ref(&b.loss, &ref_loss);
            devs[i] = (dev, at, ldev);
        }
        ns
    });

    // ---- PHASE 3: report ----
    let c_ix = ix_of(&labels, "C");
    let reportable = rr.reportable();

    println!();
    println!(
        "  {:<10} {:>10} {:>12}  {:<28}  {}",
        "impl", "ms/call", "Melem/s", "vs C (median [min-max])", "what"
    );
    let rule = "-".repeat(100);
    println!("  {rule}");
    for (i, col) in cols.iter().enumerate() {
        if i == n_wuk {
            println!("  {rule}");
        }
        let best = rr.best(i);
        println!(
            "  {:<10} {:>10.2} {:>12.1}  {:<28}  {}",
            col.label,
            best / 1e6,
            (m.r * m.c) as f64 / best * 1e3,
            vs_c_cell(i, c_ix, &rr, reportable),
            col.note,
        );
    }
    println!("  ms/call is the best of all {rounds} rounds and is ROUND-LOCAL context.");
    println!("{DIVIDE_WARNING}");
    println!();
    print_instrument(&rr, twin_ratio(&labels, &rr), rounds * (1 + SAMPLES));
    println!();
    print_round_trace(&labels, &rr);
    println!();
    println!("  DISPATCH CENSUS");
    for col in cols.iter().take(n_wuk) {
        println!("    {:<10} {}", col.label, col.census);
    }
    println!("    source written to {}", path.display());
    println!();
    println!("  CORRECTNESS vs the f64 reference (worst of ALL gradient lanes, and of all losses)");
    for (i, col) in cols.iter().enumerate() {
        let (dev, at, ldev) = devs[i];
        println!(
            "    {:<10} dz max rel {dev:.3e} at lane {at}; loss max rel {ldev:.3e}",
            col.label
        );
    }
    println!("  {}", crate::model::power_status_line());
}

// ---------------------------------------------------------------------------------------------
// Program 0: the recognizer-fragility ablation (census only — nothing is timed here)
// ---------------------------------------------------------------------------------------------

/// Minimal PAIRS of programs that compute bit-identical results and differ by one edit each, so the
/// structure tax can be attributed to a specific edit rather than to "the natural spelling" as a
/// blob. Every pair is small enough to read side by side, which is the point: the diff is the
/// finding.
///
/// Nothing here is timed. The only output is the dispatch census, so this section is immune to the
/// machine's power and thermal state and can be trusted from any run.
const ABLATIONS: [(&str, &str, &str); 10] = [
    (
        "gemm: flat index",
        "baseline",
        r#"module ab
fn kbench(a: [f32; 65536], w: [f32; 65536], mut c: [f32; 65536]) {
    for i in 0..256 { for j in 0..256 {
        let mut s: f32 = 0.0;
        for p in 0..256 { s = s + a[i*256+p] * w[j*256+p]; }
        c[i*256+j] = s;
    } }
}
"#,
    ),
    (
        "gemm: row base hoisted",
        "`let ib = i*256;` then a[ib+p]",
        r#"module ab
fn kbench(a: [f32; 65536], w: [f32; 65536], mut c: [f32; 65536]) {
    for i in 0..256 {
        let ib: i32 = i * 256;
        for j in 0..256 {
            let jb: i32 = j * 256;
            let mut s: f32 = 0.0;
            for p in 0..256 { s = s + a[ib+p] * w[jb+p]; }
            c[ib+j] = s;
        }
    }
}
"#,
    ),
    (
        "gemm: ONE base hoisted",
        "only a's row base hoisted",
        r#"module ab
fn kbench(a: [f32; 65536], w: [f32; 65536], mut c: [f32; 65536]) {
    for i in 0..256 {
        let ib: i32 = i * 256;
        for j in 0..256 {
            let mut s: f32 = 0.0;
            for p in 0..256 { s = s + a[ib+p] * w[j*256+p]; }
            c[i*256+j] = s;
        }
    }
}
"#,
    ),
    (
        "gemm: i64 named dim",
        "`let n: i64 = 256;` then a[i*n+p]",
        r#"module ab
fn kbench(a: [f32; 65536], w: [f32; 65536], mut c: [f32; 65536]) {
    let n: i64 = 256i64;
    for i in 0i64..256i64 { for j in 0i64..256i64 {
        let mut s: f32 = 0.0;
        for p in 0i64..256i64 { s = s + a[i*n+p] * w[j*n+p]; }
        c[i*n+j] = s;
    } }
}
"#,
    ),
    (
        "epilogue: own loop",
        "baseline",
        r#"module ab
fn kbench(a: [f32; 65536], w: [f32; 65536], b: [f32; 256], mut c: [f32; 65536]) {
    for i in 0..256 { for j in 0..256 {
        let mut s: f32 = 0.0;
        for p in 0..256 { s = s + a[i*256+p] * w[j*256+p]; }
        c[i*256+j] = b[j] + s;
    } }
    for i in 0..256 { for j in 0..256 { c[i*256+j] = silu(c[i*256+j]); } }
}
"#,
    ),
    (
        "epilogue: in the store",
        "c[..] = silu(b[j] + s)",
        r#"module ab
fn kbench(a: [f32; 65536], w: [f32; 65536], b: [f32; 256], mut c: [f32; 65536]) {
    for i in 0..256 { for j in 0..256 {
        let mut s: f32 = 0.0;
        for p in 0..256 { s = s + a[i*256+p] * w[j*256+p]; }
        c[i*256+j] = silu(b[j] + s);
    } }
}
"#,
    ),
    (
        "residual: own loop",
        "baseline",
        r#"module ab
fn kbench(a: [f32; 65536], w: [f32; 65536], x: [f32; 65536], mut c: [f32; 65536]) {
    for i in 0..256 { for j in 0..256 {
        let mut s: f32 = 0.0;
        for p in 0..256 { s = s + a[i*256+p] * w[j*256+p]; }
        c[i*256+j] = s;
    } }
    for i in 0..65536 { c[i] = x[i] + c[i]; }
}
"#,
    ),
    (
        "residual: in the store",
        "c[..] = x[..] + s",
        r#"module ab
fn kbench(a: [f32; 65536], w: [f32; 65536], x: [f32; 65536], mut c: [f32; 65536]) {
    for i in 0..256 { for j in 0..256 {
        let mut s: f32 = 0.0;
        for p in 0..256 { s = s + a[i*256+p] * w[j*256+p]; }
        c[i*256+j] = x[i*256+j] + s;
    } }
}
"#,
    ),
    (
        "rmsnorm: flat index",
        "baseline (out-of-place is fine too)",
        r#"module ab
fn kbench(x: [f32; 65536], g: [f32; 256], mut o: [f32; 65536]) {
    for r in 0..256 {
        let mut s: f32 = 0.0;
        for i in 0..256 { s = s + x[r*256+i] * x[r*256+i]; }
        let inv: f32 = rsqrt(s / 256.0 + 0.00001);
        for i in 0..256 { o[r*256+i] = x[r*256+i] * inv * g[i]; }
    }
}
"#,
    ),
    (
        "rmsnorm: base hoisted",
        "`let rb = r*256;` then x[rb+i]",
        r#"module ab
fn kbench(x: [f32; 65536], g: [f32; 256], mut o: [f32; 65536]) {
    for r in 0..256 {
        let rb: i32 = r * 256;
        let mut s: f32 = 0.0;
        for i in 0..256 { s = s + x[rb+i] * x[rb+i]; }
        let inv: f32 = rsqrt(s / 256.0 + 0.00001);
        for i in 0..256 { o[rb+i] = x[rb+i] * inv * g[i]; }
    }
}
"#,
    ),
];

/// Compile each ablation probe and print only its dispatch census. Not a measurement — it is a
/// statement about the compiler that holds on any machine in any power state.
fn bench_ablation() {
    println!("================================================================================");
    println!(
        "Program 0: RECOGNIZER FRAGILITY — pairs of programs that compute bit-identical results\n\
         and differ by one edit. Census only; nothing is timed, so this section is independent of\n\
         the machine's state."
    );
    println!();
    println!("  {:<26} {:<36} {}", "probe", "edit", "dispatches");
    println!("  {}", "-".repeat(100));
    for (name, edit, src) in ABLATIONS {
        println!("  {name:<26} {edit:<36} {}", probe_census(src, name));
    }
}

/// Parse -> sema -> mir_build -> optimize(-O3) and return the dispatch census, with no ABI check
/// and no JIT: the ablation probes have different arities and are never called.
fn probe_census(src: &str, label: &str) -> String {
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        return format!("PARSE ERROR in {label}");
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return format!("SEMA ERROR in {label}: {sd:?}");
    }
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        return format!("LOWER ERROR in {label}");
    }
    wukong_opt::optimize(&mut program, 3);
    let mir = wukong_mir::print::print_program(&program, &interner);
    census_line(&kernel_census(&mir))
}

// ---------------------------------------------------------------------------------------------
// Program 3: Mamba / S6 selective scan with a vector state
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct ScanDims {
    t: usize,
    d: usize,
    n: usize,
}

impl ScanDims {
    fn from_env() -> ScanDims {
        let get = |k: &str, dflt: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&x| x > 0)
                .unwrap_or(dflt)
        };
        ScanDims {
            t: get("XBENCH_GEN_T", 2048),
            d: get("XBENCH_GEN_SD", 256),
            n: get("XBENCH_GEN_N", 16),
        }
    }
}

fn subst_scan(src: &str, m: ScanDims) -> String {
    let pairs: [(&str, String); 6] = [
        ("$TD$", (m.t * m.d).to_string()),
        ("$DN$", (m.d * m.n).to_string()),
        ("$TN$", (m.t * m.n).to_string()),
        ("$T$", m.t.to_string()),
        ("$D$", m.d.to_string()),
        ("$N$", m.n.to_string()),
    ];
    let mut out = src.to_string();
    for (tok, val) in &pairs {
        out = out.replace(tok, val);
    }
    out
}

/// `kbench(x, dt, a, bmat, cmat, dskip, h, y)`.
type ScanFn = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
);

const SCAN_PARAMS: usize = 8;

struct ScanBufs {
    x: Vec<f32>,
    dt: Vec<f32>,
    a: Vec<f32>,
    bmat: Vec<f32>,
    cmat: Vec<f32>,
    dskip: Vec<f32>,
    h: Vec<f32>,
    y: Vec<f32>,
}

impl ScanBufs {
    /// Parameters in the ranges a trained S6 layer actually occupies: `A` negative so the decay
    /// `exp(dt·A)` lands in (0, 1), `dt` the small positive output of a softplus. Chosen so the
    /// carried state converges instead of exploding — an overflowing recurrence would make every
    /// implementation "agree" on infinities.
    fn new(m: ScanDims) -> ScanBufs {
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        };
        let x = (0..m.t * m.d).map(|_| (next() * 2.0 - 1.0) as f32).collect();
        let dt = (0..m.t * m.d).map(|_| (0.01 + 0.09 * next()) as f32).collect();
        let a = (0..m.d * m.n)
            .map(|i| -(0.5 + (i % m.n) as f64 + 0.25 * next()) as f32)
            .collect();
        let bmat = (0..m.t * m.n).map(|_| (next() * 2.0 - 1.0) as f32).collect();
        let cmat = (0..m.t * m.n).map(|_| (next() * 2.0 - 1.0) as f32).collect();
        let dskip = (0..m.d).map(|_| (0.5 + next()) as f32).collect();
        ScanBufs {
            x,
            dt,
            a,
            bmat,
            cmat,
            dskip,
            h: vec![0.0; m.d * m.n],
            y: vec![0.0; m.t * m.d],
        }
    }
}

/// # Safety
/// `f` must be a lowered `kbench` checked to be 8 pointers returning void; `b` must match `m`.
unsafe fn call_scan(f: ScanFn, b: &mut ScanBufs) {
    f(
        b.x.as_ptr(),
        b.dt.as_ptr(),
        b.a.as_ptr(),
        b.bmat.as_ptr(),
        b.cmat.as_ptr(),
        b.dskip.as_ptr(),
        b.h.as_mut_ptr(),
        b.y.as_mut_ptr(),
    );
}

/// The scan recomputed in f64 — a sequential recurrence, so an error at any `t` propagates to every
/// later step. Every one of the T·D output lanes is checked.
fn reference_scan_f64(m: ScanDims, b: &ScanBufs) -> Vec<f64> {
    let (t_n, d_n, n_n) = (m.t, m.d, m.n);
    let mut h = vec![0.0f64; d_n * n_n];
    let mut y = vec![0.0f64; t_n * d_n];
    for t in 0..t_n {
        let (tb, nb) = (t * d_n, t * n_n);
        for d in 0..d_n {
            let db = d * n_n;
            let dtv = b.dt[tb + d] as f64;
            let xv = b.x[tb + d] as f64;
            let gate = dtv * xv;
            let mut acc = 0.0f64;
            for n in 0..n_n {
                let decay = (dtv * b.a[db + n] as f64).exp();
                let hn = decay * h[db + n] + gate * b.bmat[nb + n] as f64;
                h[db + n] = hn;
                acc += b.cmat[nb + n] as f64 * hn;
            }
            y[tb + d] = acc + b.dskip[d] as f64 * xv;
        }
    }
    y
}

const SCAN_WK: &str = include_str!("../wk/general/scan_s6.wk");
const SCAN_C: &str = include_str!("../wk/general/peer_scan.c");
const SCAN_RS: &str = include_str!("../wk/general/peer_scan.rs");

fn bench_scan(cc: &str, cxx: &str, dir: &Path) {
    let m = ScanDims::from_env();
    println!("================================================================================");
    println!(
        "Program 3: MAMBA / S6 SELECTIVE SCAN, vector state. T={} D={} N={} — the state is a\n\
         [D, N] matrix and both the decay and the input gate are computed inside the loop from\n\
         selective parameters. `wukong_lrscan_f32` covers only a SCALAR-state recurrence with\n\
         precomputed gates, so this shape has no kernel and could not have one without a new\n\
         recognizer per state-space variant.",
        m.t, m.d, m.n
    );
    println!("  {}", crate::model::power_status_line());

    let mut b = ScanBufs::new(m);
    let t0 = Instant::now();
    let reference = reference_scan_f64(m, &b);
    println!(
        "  f64 scalar reference computed in {:.2} s ({} lanes, every one checked)",
        t0.elapsed().as_secs_f64(),
        reference.len()
    );

    // ---- PHASE 1: BUILD every column. Nothing is timed here. ----
    let mut cols: Vec<Col<ScanFn>> = Vec::new();
    let wk_src = subst_scan(SCAN_WK, m);
    let path = dir.join("general_scan_s6.wk");
    let _ = std::fs::write(&path, &wk_src);
    match compile_block(&wk_src, "scan", SCAN_PARAMS) {
        Some(blk) => {
            // SAFETY: `block_abi_ok` proved the lowered `kbench` is 8 pointers -> void = `ScanFn`;
            // the handle is moved into `Keep` and outlives every call.
            let f: ScanFn = unsafe { std::mem::transmute(blk.ptr) };
            cols.push(Col {
                label: "Wukong".to_string(),
                note: "the .wk source, through the real pipeline".to_string(),
                census: census_line(&blk.census),
                compile: blk.compile,
                f,
                _keep: Keep::Jit(blk.handle),
            });
        }
        None => println!("  ! Wukong scan did not compile — row dropped"),
    }
    let n_wuk = cols.len();

    let c_src = subst_scan(SCAN_C, m);
    let rs_src = subst_scan(SCAN_RS, m);
    // SAFETY: every scan peer declares `kbench` with the 8-pointer scan ABI, which is exactly
    // `ScanFn`; the Wukong side is arity-checked against `SCAN_PARAMS` by `block_abi_ok`.
    cols.extend(unsafe { build_peer_cols::<ScanFn>("general_scan", &c_src, &rs_src, dir, cc, cxx) });
    if cols.is_empty() {
        println!("  ! no column built — section skipped");
        return;
    }
    let labels = labels_of(&cols);

    // ---- PHASE 2: ROUNDS ----
    let rounds = crate::gen_rounds();
    let mut devs = vec![(0.0f64, 0usize); cols.len()];
    let rr = Rounds::run(cols.len(), rounds, |i, r| {
        let f = cols[i].f;
        b.y.iter_mut().for_each(|v| *v = 0.0);
        // SAFETY: `f` is the 8-pointer `kbench` of column `i`; `b` was built by `ScanBufs::new(m)`,
        // the same `m` every column was compiled for.
        let ns = time_round(|| unsafe { call_scan(f, &mut b) });
        if r == 0 {
            devs[i] = max_rel_vs_ref(&b.y, &reference);
        }
        ns
    });

    // ---- PHASE 3: report ----
    let c_ix = ix_of(&labels, "C");
    let reportable = rr.reportable();

    println!();
    println!(
        "  {:<10} {:>10} {:>12}  {:<28}  {}",
        "impl", "ms/call", "Mstate/s", "vs C (median [min-max])", "what"
    );
    let rule = "-".repeat(100);
    println!("  {rule}");
    for (i, col) in cols.iter().enumerate() {
        if i == n_wuk {
            println!("  {rule}");
        }
        let best = rr.best(i);
        println!(
            "  {:<10} {:>10.2} {:>12.1}  {:<28}  {}",
            col.label,
            best / 1e6,
            (m.t * m.d * m.n) as f64 / best * 1e3,
            vs_c_cell(i, c_ix, &rr, reportable),
            col.note,
        );
    }
    println!("  ms/call is the best of all {rounds} rounds and is ROUND-LOCAL context.");
    println!("{DIVIDE_WARNING}");
    println!();
    print_instrument(&rr, twin_ratio(&labels, &rr), rounds * (1 + SAMPLES));
    println!();
    print_round_trace(&labels, &rr);
    println!();
    println!("  DISPATCH CENSUS");
    for col in cols.iter().take(n_wuk) {
        println!("    {:<10} {}", col.label, col.census);
    }
    println!("    source written to {}", path.display());
    println!();
    println!("  CORRECTNESS vs the f64 reference (worst of all {} lanes)", reference.len());
    for (i, col) in cols.iter().enumerate() {
        let (dev, at) = devs[i];
        println!("    {:<10} max rel {dev:.3e} at lane {at}", col.label);
    }
    println!("  {}", crate::model::power_status_line());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `$TOKEN$` in every corpus source must be one this module knows how to substitute —
    /// an unsubstituted token reaches the parser as a syntax error at benchmark time, which
    /// `cargo test` would never see because xbench compiles its `.wk` peers at RUNTIME.
    #[test]
    fn every_placeholder_is_substituted() {
        let m = Dims {
            s: 64,
            d: 32,
            h: 2,
            f: 64,
        };
        for (label, src, _) in VARIANTS {
            let out = subst(src, m);
            assert!(
                !out.contains('$'),
                "{label}: unsubstituted placeholder near {:?}",
                &out[out.find('$').unwrap()..][..40.min(out.len() - out.find('$').unwrap())]
            );
        }
        for (name, src) in [("peer.c", PEER_C), ("peer.rs", PEER_RS)] {
            let out = subst(src, m);
            assert!(!out.contains('$'), "{name}: unsubstituted placeholder");
        }
        let lm = LossDims {
            r: 4,
            c: 8,
            eps: 0.125,
        };
        for (name, src) in [
            ("loss.wk", LOSS_WK),
            ("loss.c", LOSS_C),
            ("loss.rs", LOSS_RS),
        ] {
            let out = subst_loss(src, lm);
            assert!(!out.contains('$'), "{name}: unsubstituted placeholder");
        }
        let sm = ScanDims { t: 4, d: 8, n: 4 };
        for (name, src) in [
            ("scan.wk", SCAN_WK),
            ("scan.c", SCAN_C),
            ("scan.rs", SCAN_RS),
        ] {
            let out = subst_scan(src, sm);
            assert!(!out.contains('$'), "{name}: unsubstituted placeholder");
        }
    }

    /// Every ablation probe must reach optimized MIR — a probe that failed to compile would print
    /// "no dispatches" and read exactly like a recognizer decline, which is the opposite claim.
    #[test]
    fn every_ablation_probe_compiles() {
        for (name, _, src) in ABLATIONS {
            let text = probe_census(src, name);
            assert!(
                !text.contains("ERROR"),
                "{name}: probe did not compile — {text}"
            );
        }
    }

    /// The ablation is only evidence if the baseline of each pair really does dispatch. If a
    /// baseline ever stopped dispatching, every row would read "NOTHING" and the section would
    /// silently stop saying anything at all.
    #[test]
    fn ablation_baselines_still_dispatch() {
        for name in [
            "gemm: flat index",
            "epilogue: own loop",
            "residual: own loop",
            "rmsnorm: flat index",
        ] {
            let (_, _, src) = ABLATIONS.iter().find(|(n, _, _)| *n == name).unwrap();
            let text = probe_census(src, name);
            assert!(
                text.contains("wukong_"),
                "{name}: baseline no longer dispatches ({text}) — the ablation proves nothing"
            );
        }
    }

    /// The scan reference must be a converging recurrence with a live state: an exploding or dead
    /// state would make the comparison vacuous in opposite ways.
    #[test]
    fn scan_reference_is_stable_and_live() {
        let m = ScanDims { t: 32, d: 8, n: 4 };
        let b = ScanBufs::new(m);
        let y = reference_scan_f64(m, &b);
        assert_eq!(y.len(), m.t * m.d);
        assert!(y.iter().all(|v| v.is_finite()), "the recurrence diverged");
        assert!(y.iter().any(|v| v.abs() > 1.0e-6), "the state never became live");
        // Every decay must be a genuine contraction, or the state is not a state.
        for (i, &a) in b.a.iter().enumerate() {
            assert!(a < 0.0, "A[{i}] = {a} is not negative");
        }
    }

    /// The smoothed targets must sum to exactly 1 over the row, and each must round-trip through
    /// the decimal literal that is substituted into three different front ends. A literal that one
    /// of them rounded differently would surface as a phantom correctness deviation.
    #[test]
    fn smoothed_targets_are_exact_and_round_trip() {
        let m = LossDims {
            r: 4,
            c: 1024,
            eps: 0.125,
        };
        let total = m.qt() + m.qo() * (m.c - 1) as f32;
        assert!((total - 1.0).abs() < 1.0e-6, "targets sum to {total}, not 1");
        for v in [m.qt(), m.qo()] {
            let text = v.to_string();
            assert!(!text.contains('e'), "{text} uses scientific notation");
            assert_eq!(text.parse::<f32>().unwrap().to_bits(), v.to_bits());
        }
    }

    /// The loss reference must produce a finite, non-trivial gradient — a silently-zero `dz` would
    /// let any implementation "agree" with it.
    #[test]
    fn loss_reference_is_nontrivial() {
        let m = LossDims {
            r: 4,
            c: 16,
            eps: 0.125,
        };
        let b = LossBufs::new(m);
        let (loss, dz) = reference_loss_f64(m, &b);
        assert_eq!(dz.len(), m.r * m.c);
        assert!(loss.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(dz.iter().all(|v| v.is_finite()));
        assert!(dz.iter().any(|v| v.abs() > 1.0e-9));
        // Softmax gradients of a scalar loss sum to ~0 per row: a real check that the Jacobian
        // contraction is the one the specification asks for, not just "not zero".
        for r in 0..m.r {
            let s: f64 = dz[r * m.c..(r + 1) * m.c].iter().sum();
            assert!(s.abs() < 1.0e-9, "row {r} gradient sums to {s}, not 0");
        }
    }

    /// The five spellings must declare the identical 27-pointer ABI, or the harness would be
    /// calling different functions and calling the difference a structure tax.
    #[test]
    fn all_variants_declare_the_same_arity() {
        for (label, src, _) in VARIANTS {
            let sig_start = src.find("fn kbench(").expect("kbench");
            let sig_end = sig_start + src[sig_start..].find(") {").expect("signature end");
            let sig = &src[sig_start..sig_end];
            let n = sig.matches(':').count();
            assert_eq!(n, N_IN + N_SCRATCH, "{label} declares {n} kbench parameters");
        }
    }

    /// `Dims::flops` and `total_bytes` must agree with the buffer set actually allocated: a wrong
    /// FLOP count turns the GF/s column into fiction.
    #[test]
    fn buffer_sizes_match_the_reported_footprint() {
        let m = Dims {
            s: 8,
            d: 8,
            h: 2,
            f: 16,
        };
        let b = Bufs::new(m);
        let live: usize =
            b.inp.iter().map(|v| v.len()).sum::<usize>() + b.scr.iter().map(|v| v.len()).sum::<usize>();
        assert_eq!(live * 4, total_bytes(m));
    }

    /// The reference must actually be a transformer block, not zeros: a silently-zero oracle would
    /// make every implementation "agree" with it at any relative tolerance around zero.
    #[test]
    fn reference_is_nontrivial() {
        let m = Dims {
            s: 8,
            d: 8,
            h: 2,
            f: 16,
        };
        let b = Bufs::new(m);
        let r = reference_f64(m, &b.inp);
        assert_eq!(r.len(), m.s * m.d);
        assert!(
            r.iter().any(|v| v.abs() > 1.0e-6),
            "reference is all ~zero — the oracle would be vacuous"
        );
        assert!(r.iter().all(|v| v.is_finite()), "reference has non-finite lanes");
    }
}
