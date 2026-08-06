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
//! Variant (b) is the one that has moved. It began at 7 dispatched call sites against (a)'s 16 and
//! was ~6x slower than the other four; it now dispatches 15 — (a)'s multiset minus the one
//! `wukong_vmath_f32`, which (b) does not need because its activation rides the GEMM epilogue — and
//! sits inside the pack. **Do not "fix" (b) by rewriting it into the dialect.** It is the only
//! spelling here written with no regard for the recognizers, and rewriting it would delete the
//! measurement instead of the tax.
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
//! * **Round-local timings.** Absolute GFLOP/s and %-of-roofline swing ~3x with this laptop's power
//!   and thermal state; only the same-run adjacent ratios are reportable. The power state is printed
//!   before AND after.

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

/// A single measured run: best-observed time, the compile time that produced it, and the output
/// buffer as it stood after the last call.
struct Measure {
    compile: Duration,
    ns_per_call: f64,
    out: Vec<f32>,
}

/// Symmetric warm-up, then best-of-N minima. Identical protocol for Wukong and every peer — the
/// only comparison this laptop supports is a same-run adjacent A/B of two best-observed times.
///
/// A block forward at the default size is tens of milliseconds, so this is the "already above the
/// sampling target" path: two warm-ups, then the minimum of `N_SAMPLES` single calls. The minimum,
/// not the mean: on a machine with an OS, background work can only ever make a sample slower.
fn time_ns(mut run: impl FnMut()) -> f64 {
    const N_SAMPLES: usize = 7;
    run();
    run();
    let mut best = f64::INFINITY;
    for _ in 0..N_SAMPLES {
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

fn bench_wukong_block(src: &str, label: &str, bufs: &mut Bufs) -> Option<(Measure, Vec<(String, usize)>)> {
    let blk = compile_block(src, label, N_IN + N_SCRATCH)?;
    // SAFETY: `block_abi_ok` just proved the lowered `kbench` is 27 pointers -> void, which is
    // exactly `BlockFn`; `blk.handle` is kept alive until after the last call.
    let f: BlockFn = unsafe { std::mem::transmute(blk.ptr) };
    bufs.scr[OUT_IX].iter_mut().for_each(|v| *v = 0.0);
    let ns = time_ns(|| unsafe { call_block(f, bufs) });
    let out = bufs.scr[OUT_IX].clone();
    let census = blk.census.clone();
    drop(blk.handle);
    Some((
        Measure {
            compile: blk.compile,
            ns_per_call: ns,
            out,
        },
        census,
    ))
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

/// Compile a peer source and time it through the identical protocol as the Wukong variants.
/// Obeys the same one-live-pointer law: the output pointer is derived inside [`call_block`].
fn bench_peer(
    ext: &str,
    src: &str,
    dir: &Path,
    name: &str,
    compiler: &str,
    args: &[&str],
    bufs: &mut Bufs,
) -> Option<Measure> {
    let (lib, compile) = load_peer(ext, src, dir, name, compiler, args)?;
    unsafe {
        let sym: libloading::Symbol<BlockFn> = match lib.get(b"kbench\0") {
            Ok(s) => s,
            Err(e) => {
                eprintln!("general: symbol kbench in {name}.{ext}: {e}");
                return None;
            }
        };
        let f: BlockFn = *sym;
        bufs.scr[OUT_IX].iter_mut().for_each(|v| *v = 0.0);
        let ns = time_ns(|| call_block(f, bufs));
        let out = bufs.scr[OUT_IX].clone();
        Some(Measure {
            compile,
            ns_per_call: ns,
            out,
        })
    }
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

/// A finished row of the headline table.
struct Row {
    label: &'static str,
    note: &'static str,
    ns: f64,
    census: String,
    dev: f64,
    dev_at: usize,
    compile: Duration,
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
    let power_before = crate::model::power_status_line();
    println!("  {power_before}");
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

    // ---- the five Wukong spellings ----
    let mut rows: Vec<Row> = Vec::new();
    let mut outs: Vec<(&str, Vec<f32>)> = Vec::new();
    for (label, src, note) in VARIANTS {
        let text = subst(src, m);
        // Dump the exact compiled source so `wukongc --emit=mir -O2 <file>` reproduces the census.
        let path = dir.join(format!("general_block_{label}.wk"));
        let _ = std::fs::write(&path, &text);
        let Some((mm, census)) = bench_wukong_block(&text, label, &mut bufs) else {
            println!("  ! {label}: did not compile — row dropped");
            continue;
        };
        let (dev, at) = max_rel_vs_ref(&mm.out, &reference);
        rows.push(Row {
            label,
            note,
            ns: mm.ns_per_call,
            census: census_line(&census),
            dev,
            dev_at: at,
            compile: mm.compile,
        });
        outs.push((label, mm.out));
    }

    // ---- the peers ----
    let c_src = subst(PEER_C, m);
    let rs_src = subst(PEER_RS, m);
    let honest = &["-O3", "-march=native", "-ffp-contract=fast", "-shared"];
    let fastm = &[
        "-O3",
        "-march=native",
        "-ffp-contract=fast",
        "-ffast-math",
        "-shared",
    ];
    let peers: Vec<(&'static str, Option<Measure>)> = vec![
        (
            "C (gcc -O3, IEEE order)",
            bench_peer("c", &c_src, dir, "general_block", cc, honest, &mut bufs),
        ),
        (
            "C(fast) (gcc -ffast-math)",
            bench_peer(
                "c",
                &c_src,
                dir,
                "general_block_fast",
                cc,
                fastm,
                &mut bufs,
            ),
        ),
        (
            "C++ (g++ -O3, IEEE order)",
            bench_peer("cpp", &c_src, dir, "general_block", cxx, honest, &mut bufs),
        ),
        (
            "Rust (rustc -O, IEEE order)",
            bench_peer(
                "rs",
                &rs_src,
                dir,
                "general_block",
                "rustc",
                &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut bufs,
            ),
        ),
    ];

    // ---- report ----
    println!();
    println!("  {:<14} {:>10} {:>9} {:>11}  {}", "variant", "ms/fwd", "GF/s*", "vs C", "spelling");
    println!("  {}", "-".repeat(92));
    let c_ns = peers
        .iter()
        .find(|(l, _)| l.starts_with("C ("))
        .and_then(|(_, mm)| mm.as_ref())
        .map(|mm| mm.ns_per_call);
    for r in &rows {
        println!(
            "  {:<14} {:>10.2} {:>9.1} {:>11}  {}",
            r.label,
            r.ns / 1e6,
            m.flops() / r.ns,
            match c_ns {
                Some(c) => fmt_ratio(c / r.ns),
                None => "-".to_string(),
            },
            r.note,
        );
    }
    println!("  {}", "-".repeat(92));
    for (label, mm) in &peers {
        match mm {
            Some(mm) => println!(
                "  {:<14} {:>10.2} {:>9.1} {:>11}  peer, written once, competently",
                peer_short(label),
                mm.ns_per_call / 1e6,
                m.flops() / mm.ns_per_call,
                match c_ns {
                    Some(c) => fmt_ratio(c / mm.ns_per_call),
                    None => "-".to_string(),
                },
            ),
            None => println!("  {:<14} {:>10}  (unavailable)", peer_short(label), "-"),
        }
    }
    println!(
        "  * GF/s is ROUND-LOCAL: this laptop's clock swings ~3x with power and thermal state.\n\
           Only the same-run ratios in this table are reportable; the absolute column is context."
    );

    // ---- dispatch census ----
    println!();
    println!("  DISPATCH CENSUS (= wukongc --emit=mir -O2 <src> | grep -oE 'wukong_[a-z0-9_]+')");
    for r in &rows {
        println!("    {:<14} {}", r.label, r.census);
    }
    println!(
        "    sources written to {} — rerun the grep against the real binary to confirm.",
        dir.display()
    );

    // ---- correctness ----
    println!();
    println!("  CORRECTNESS vs the f64 scalar reference (worst of all {} lanes)", reference.len());
    for r in &rows {
        println!(
            "    {:<14} max rel dev {:.3e} at lane {}",
            r.label, r.dev, r.dev_at
        );
    }
    for (label, mm) in &peers {
        if let Some(mm) = mm {
            let (dev, at) = max_rel_vs_ref(&mm.out, &reference);
            println!(
                "    {:<14} max rel dev {:.3e} at lane {}",
                peer_short(label),
                dev,
                at
            );
        }
    }
    if let Some((_, base)) = outs.iter().find(|(l, _)| *l == "b-natural") {
        println!();
        // b-natural is the *reference* spelling here only in the sense that it is the one written
        // without any regard for the recognizers. It is NOT "the fully-scalar one" any more — since
        // the store-fused epilogue / cross-buffer residual / dual-store arms landed it dispatches 15
        // kernels of its own. The independent oracle is the f64 recomputation above, checked on every
        // lane; this block only reports whether the five spellings agree with each other.
        println!("  AGREEMENT between spellings (vs b-natural, the one written with no regard for");
        println!("  the recognizers; the independent oracle is the f64 reference above)");
        for (label, o) in &outs {
            if *label == "b-natural" {
                continue;
            }
            let (dev, same) = max_rel_pair(o, base);
            println!(
                "    {:<14} {}",
                label,
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
        for (label, o) in &outs {
            match classes.iter_mut().find(|c| {
                let rep = outs.iter().find(|(l, _)| l == &c[0]).unwrap();
                rep.1.iter().zip(o.iter()).all(|(a, b)| a.to_bits() == b.to_bits())
            }) {
                Some(c) => c.push(label),
                None => classes.push(vec![label]),
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
    println!();
    if !rows.is_empty() {
        let fastest = rows.iter().min_by(|a, b| a.ns.total_cmp(&b.ns)).unwrap();
        let slowest = rows.iter().max_by(|a, b| a.ns.total_cmp(&b.ns)).unwrap();
        let (fast, slow) = (fastest.ns, slowest.ns);
        println!(
            "  ==> STRUCTURE TAX: {:.2}x spread across five spellings of ONE block\n      \
             (fastest {} at {:.2} ms, slowest {} at {:.2} ms).",
            slow / fast,
            fastest.label,
            fast / 1e6,
            slowest.label,
            slow / 1e6,
        );
        if let Some(c) = c_ns {
            println!(
                "      For scale, gcc compiles ONE source; its spread across these spellings is 1.0\n      \
                 by construction. C is at {:.2} ms.",
                c / 1e6
            );
        }
    }
    println!();
    println!(
        "  compile time: {}",
        rows.iter()
            .map(|r| format!("{} {:.0}ms", r.label, r.compile.as_secs_f64() * 1e3))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let power_after = crate::model::power_status_line();
    println!("  {power_after}");
    if power_after != power_before {
        println!(
            "  ! POWER STATE CHANGED DURING THE RUN — these numbers are NOT comparable across rows."
        );
    }
}

fn fmt_ratio(r: f64) -> String {
    if r >= 1.0 {
        format!("{r:.2}x faster")
    } else {
        format!("{:.2}x slower", 1.0 / r)
    }
}

fn peer_short(label: &str) -> &str {
    label.split(' ').next().unwrap_or(label)
}

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
    let power_before = crate::model::power_status_line();
    println!("  {power_before}");

    let mut b = LossBufs::new(m);
    let t = Instant::now();
    let (ref_loss, ref_dz) = reference_loss_f64(m, &b);
    println!(
        "  f64 scalar reference computed in {:.2} s ({} gradient lanes + {} loss lanes, all checked)",
        t.elapsed().as_secs_f64(),
        ref_dz.len(),
        ref_loss.len()
    );

    // ---- Wukong ----
    let wk_src = subst_loss(LOSS_WK, m);
    let path = dir.join("general_loss_focal.wk");
    let _ = std::fs::write(&path, &wk_src);
    let mut rows: Vec<(String, f64, String, f64, usize, f64)> = Vec::new();
    if let Some(blk) = compile_block(&wk_src, "loss", LOSS_PARAMS) {
        // SAFETY: `block_abi_ok` proved the lowered `kbench` is 6 pointers -> void = `LossFn`;
        // the handle outlives the last call.
        let f: LossFn = unsafe { std::mem::transmute(blk.ptr) };
        b.dz.iter_mut().for_each(|v| *v = 0.0);
        let ns = time_ns(|| unsafe { call_loss(f, &mut b) });
        let (dev, at) = max_rel_vs_ref(&b.dz, &ref_dz);
        let (ldev, _) = max_rel_vs_ref(&b.loss, &ref_loss);
        rows.push((
            "Wukong".to_string(),
            ns,
            census_line(&blk.census),
            dev,
            at,
            ldev,
        ));
        drop(blk.handle);
    } else {
        println!("  ! Wukong loss did not compile — row dropped");
    }

    // ---- peers ----
    let c_src = subst_loss(LOSS_C, m);
    let rs_src = subst_loss(LOSS_RS, m);
    let honest = &["-O3", "-march=native", "-ffp-contract=fast", "-shared"];
    let fastm = &[
        "-O3",
        "-march=native",
        "-ffp-contract=fast",
        "-ffast-math",
        "-shared",
    ];
    let peers: [(&str, &str, &str, &str, &[&str]); 4] = [
        ("C", "c", &c_src, cc, honest),
        ("C(fast)", "c", &c_src, cc, fastm),
        ("C++", "cpp", &c_src, cxx, honest),
        (
            "Rust",
            "rs",
            &rs_src,
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
        ),
    ];
    // Load EVERY peer before timing ANY of them, then time them in alternating order — forward
    // pass, then reverse pass — keeping each peer's own minimum. `time_ns` already does a
    // symmetric warm-up plus best-of-7, but running all of one peer's samples before any of the
    // next's charges slow thermal/clock drift across the section to whichever peer is timed last.
    // Here that was always Rust, and C++ always sat third — a fixed-order bias in a section whose
    // C++ and Rust columns are published. Alternating gives every peer both an early and a late
    // slot, so drift cancels instead of accumulating against one language. Same reasoning as the
    // A-B-B-A pairing `bench_c_cpp*` uses in main.rs; this is its N-peer form.
    let mut loaded: Vec<(String, libloading::Library, LossFn)> = Vec::new();
    for (label, ext, src, compiler, args) in peers {
        let name = format!("general_loss_{}", label.replace(['(', ')'], "_"));
        let Some((lib, _)) = load_peer(ext, src, dir, &name, compiler, args) else {
            println!("  ! {label} loss peer unavailable — row dropped");
            continue;
        };
        let f: LossFn = unsafe {
            match lib.get::<LossFn>(b"kbench\0") {
                Ok(s) => *s,
                Err(e) => {
                    println!("  ! {label}: symbol kbench: {e}");
                    continue;
                }
            }
        };
        loaded.push((label.to_string(), lib, f));
    }
    let mut best = vec![f64::INFINITY; loaded.len()];
    // (dz deviation, its lane, loss deviation) — recorded once; the peers are deterministic, so
    // the extra timing passes cannot change them.
    let mut dev_of = vec![(0.0f64, 0usize, 0.0f64); loaded.len()];
    for pass in 0..2usize {
        for step in 0..loaded.len() {
            let i = if pass == 0 {
                step
            } else {
                loaded.len() - 1 - step
            };
            let f = loaded[i].2;
            b.dz.iter_mut().for_each(|v| *v = 0.0);
            let ns = time_ns(|| unsafe { call_loss(f, &mut b) });
            if ns < best[i] {
                best[i] = ns;
            }
            if pass == 0 {
                let (dev, at) = max_rel_vs_ref(&b.dz, &ref_dz);
                let (ldev, _) = max_rel_vs_ref(&b.loss, &ref_loss);
                dev_of[i] = (dev, at, ldev);
            }
        }
    }
    for (i, (label, _lib, _f)) in loaded.iter().enumerate() {
        let (dev, at, ldev) = dev_of[i];
        rows.push((label.clone(), best[i], "-".to_string(), dev, at, ldev));
    }

    println!();
    println!("  {:<10} {:>10} {:>12} {:>13}", "impl", "ms/call", "Melem/s", "vs C");
    println!("  {}", "-".repeat(52));
    let c_ns = rows.iter().find(|r| r.0 == "C").map(|r| r.1);
    for (label, ns, _, _, _, _) in &rows {
        println!(
            "  {:<10} {:>10.2} {:>12.1} {:>13}",
            label,
            ns / 1e6,
            (m.r * m.c) as f64 / ns * 1e3,
            match c_ns {
                Some(c) => fmt_ratio(c / ns),
                None => "-".to_string(),
            }
        );
    }
    println!();
    println!("  DISPATCH CENSUS");
    for (label, _, census, _, _, _) in &rows {
        if census != "-" {
            println!("    {label:<10} {census}");
        }
    }
    println!("    source written to {}", path.display());
    println!();
    println!("  CORRECTNESS vs the f64 reference (worst of ALL gradient lanes, and of all losses)");
    for (label, _, _, dev, at, ldev) in &rows {
        println!("    {label:<10} dz max rel {dev:.3e} at lane {at}; loss max rel {ldev:.3e}");
    }
    let power_after = crate::model::power_status_line();
    println!("  {power_after}");
    if power_after != power_before {
        println!("  ! POWER STATE CHANGED DURING THE RUN — rows are not comparable.");
    }
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
    let power_before = crate::model::power_status_line();
    println!("  {power_before}");

    let mut b = ScanBufs::new(m);
    let t0 = Instant::now();
    let reference = reference_scan_f64(m, &b);
    println!(
        "  f64 scalar reference computed in {:.2} s ({} lanes, every one checked)",
        t0.elapsed().as_secs_f64(),
        reference.len()
    );

    let mut rows: Vec<(String, f64, String, f64, usize)> = Vec::new();
    let wk_src = subst_scan(SCAN_WK, m);
    let path = dir.join("general_scan_s6.wk");
    let _ = std::fs::write(&path, &wk_src);
    if let Some(blk) = compile_block(&wk_src, "scan", SCAN_PARAMS) {
        // SAFETY: `block_abi_ok` proved the lowered `kbench` is 8 pointers -> void = `ScanFn`.
        let f: ScanFn = unsafe { std::mem::transmute(blk.ptr) };
        b.y.iter_mut().for_each(|v| *v = 0.0);
        let ns = time_ns(|| unsafe { call_scan(f, &mut b) });
        let (dev, at) = max_rel_vs_ref(&b.y, &reference);
        rows.push(("Wukong".into(), ns, census_line(&blk.census), dev, at));
        drop(blk.handle);
    } else {
        println!("  ! Wukong scan did not compile — row dropped");
    }

    let c_src = subst_scan(SCAN_C, m);
    let rs_src = subst_scan(SCAN_RS, m);
    let honest = &["-O3", "-march=native", "-ffp-contract=fast", "-shared"];
    let fastm = &[
        "-O3",
        "-march=native",
        "-ffp-contract=fast",
        "-ffast-math",
        "-shared",
    ];
    let peers: [(&str, &str, &str, &str, &[&str]); 4] = [
        ("C", "c", &c_src, cc, honest),
        ("C(fast)", "c", &c_src, cc, fastm),
        ("C++", "cpp", &c_src, cxx, honest),
        (
            "Rust",
            "rs",
            &rs_src,
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
        ),
    ];
    // Load every peer before timing any of them, then alternate the timing order across two
    // passes and keep each peer's minimum — see the identical treatment in the loss section above
    // for why a fixed order silently charges section-wide drift to whichever peer goes last.
    let mut loaded: Vec<(String, libloading::Library, ScanFn)> = Vec::new();
    for (label, ext, src, compiler, args) in peers {
        let name = format!("general_scan_{}", label.replace(['(', ')'], "_"));
        let Some((lib, _)) = load_peer(ext, src, dir, &name, compiler, args) else {
            println!("  ! {label} scan peer unavailable — row dropped");
            continue;
        };
        let f: ScanFn = unsafe {
            match lib.get::<ScanFn>(b"kbench\0") {
                Ok(s) => *s,
                Err(e) => {
                    println!("  ! {label}: symbol kbench: {e}");
                    continue;
                }
            }
        };
        loaded.push((label.to_string(), lib, f));
    }
    let mut best = vec![f64::INFINITY; loaded.len()];
    let mut dev_of = vec![(0.0f64, 0usize); loaded.len()];
    for pass in 0..2usize {
        for step in 0..loaded.len() {
            let i = if pass == 0 {
                step
            } else {
                loaded.len() - 1 - step
            };
            let f = loaded[i].2;
            b.y.iter_mut().for_each(|v| *v = 0.0);
            let ns = time_ns(|| unsafe { call_scan(f, &mut b) });
            if ns < best[i] {
                best[i] = ns;
            }
            if pass == 0 {
                dev_of[i] = max_rel_vs_ref(&b.y, &reference);
            }
        }
    }
    for (i, (label, _lib, _f)) in loaded.iter().enumerate() {
        let (dev, at) = dev_of[i];
        rows.push((label.clone(), best[i], "-".into(), dev, at));
    }

    println!();
    println!("  {:<10} {:>10} {:>12} {:>13}", "impl", "ms/call", "Mstate/s", "vs C");
    println!("  {}", "-".repeat(52));
    let c_ns = rows.iter().find(|r| r.0 == "C").map(|r| r.1);
    for (label, ns, _, _, _) in &rows {
        println!(
            "  {:<10} {:>10.2} {:>12.1} {:>13}",
            label,
            ns / 1e6,
            (m.t * m.d * m.n) as f64 / ns * 1e3,
            match c_ns {
                Some(c) => fmt_ratio(c / ns),
                None => "-".to_string(),
            }
        );
    }
    println!();
    println!("  DISPATCH CENSUS");
    for (label, _, census, _, _) in &rows {
        if census != "-" {
            println!("    {label:<10} {census}");
        }
    }
    println!("    source written to {}", path.display());
    println!();
    println!("  CORRECTNESS vs the f64 reference (worst of all {} lanes)", reference.len());
    for (label, _, _, dev, at) in &rows {
        println!("    {label:<10} max rel {dev:.3e} at lane {at}");
    }
    let power_after = crate::model::power_status_line();
    println!("  {power_after}");
    if power_after != power_before {
        println!("  ! POWER STATE CHANGED DURING THE RUN — rows are not comparable.");
    }
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
