//! **The ruler.** `WUKONG_MASTER_PLAN` Task 0.4 — the honest gpu-native baseline of a `@generic`
//! kernel — with the measurements Phases 1 and 3 owe measured against it.
//!
//! # Why this module exists
//!
//! Task 0.4 was skipped when Phase 0 was executed, so Phases 1 and 3 shipped *speed* features with
//! zero measurement, and Phase 4's own acceptance criterion ("improves vs Phase 3") had no baseline
//! to improve on. Worse, one of those unmeasured features was emitting PTX that faulted the device
//! two different ways (`624816f`), and a single measurement run would have caught it. Nothing after
//! this should start without a number.
//!
//! # What is timed, and why each quantity is named separately
//!
//! The quantity names are **deliberately the same** as
//! [`crate::gpu_e2e_bench`](../gpu_e2e_bench/index.html)'s, so a language number and a campaign
//! number can sit in the same sentence:
//!
//! | quantity | what is inside it | how |
//! |---|---|---|
//! | `compile_ms` | lex, parse, sema, `mir_build`, `wukong_opt` at the requested `-O` | host only, no device, minimum of [`COMPILE_REPS`] |
//! | `kernel_ms` | the megakernel launch alone plus `synchronize`, module resolved and both device buffers already resident — **the published quantity** | `megakernel::time_mega` |
//! | `launch_ms` | `kernel_ms` plus the per-run driver traffic around it (zeroed context up, records back) | the same call |
//! | `cpu_ms` | the same program through `wukong_codegen_cranelift::jit_run`, whole call | minimum of the same rep count |
//!
//! There is no `e2e_ms` here and that omission is deliberate: `--backend=gpu-native` has no offload
//! seam to charge, so `e2e_ms` would be `launch_ms` plus a PTX re-emit, and publishing two names for
//! one quantity is how a table starts lying.
//!
//! # Every number is a difference
//!
//! Three properties of gpu-native, each found by probe and each documented in `bench/lang/`'s own
//! headers, force the fixtures into one shared skeleton:
//!
//!  1. `alloc_f32` never reaches the device (gpu-native declines `wukong_rt_alloc`), so the buffers
//!     are `main` locals, which the megakernel homes in the shared `.global` frame.
//!  2. A program with no `wukong_parallel_for` and no recognized kernel is **not megakernel
//!     eligible**, falls through to the single-thread lowering, and at this size that launch is
//!     impossible -- CUDA caps local memory at 512 KiB per thread, and the frame here is 32 MiB.
//!     (It surfaced as a bare `DriverError(CUDA_ERROR_INVALID_VALUE)` when this was written; it is
//!     an `UNSUPPORTED:` decline now.) So every fixture carries an *anchor* `@parallel` loop whose
//!     only job is to make the program eligible.
//!  3. `[0.0; N]` lowers to a 4x-unrolled scalar store loop the outliner does not claim, so each
//!     declaration is N serial device stores.
//!
//! `base.wk` is the skeleton and nothing else, so `kernel_ms(x) - kernel_ms(base)` is the cost of
//! `x`'s own loop. **A raw `kernel_ms` from this directory is not a result.**
//!
//! # The instrument
//!
//! Every published ratio goes through `wukong_codegen_gpu::bench_instrument`'s twin protocol
//! unmodified: arms `A` and `C` run *identical* work so `C/A` is that shape's own noise floor, arm
//! `B` is the contender, and a cell that does not clear the floor publishes nothing. On this box
//! that is not a formality — an unpinned harness once published a binary against itself as "1.31x
//! faster", and a cold device-keyed cubin cache once read a 27x false regression.

use std::path::{Path, PathBuf};
use std::time::Instant;

use wukong_codegen_gpu::bench_instrument::{analyze, run_rotated, Arm, TwinPlan};
use wukong_codegen_gpu::megakernel;
use wukong_mir::Program;
use wukong_span::{Interner, SourceMap};

/// How many times the host-only compile is repeated for the `compile_ms` reading. The minimum is
/// reported, which is `docs/compile-floor.md`'s regime — and that document's own section 2 says a
/// debug binary makes the backend look ~7x slower and the measurement worthless, so a `compile_ms`
/// read off anything but a `--release` build of this harness is not comparable to anything.
const COMPILE_REPS: usize = 9;

/// Launches per timed sample. The device work here is tens of microseconds, so a handful of reps
/// inside one sample keeps the `Instant` resolution out of the reading without letting the clock
/// drift across the arm boundary.
const DEVICE_REPS: usize = 7;

/// The fixture directory, relative to this crate.
fn lang_bench_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/lang")
}

/// One fixture, compiled once and kept — the compile is not per-arm work.
struct Fixture {
    name: String,
    program: Program,
    interner: Interner,
    entry: wukong_span::Symbol,
    /// Minimum of [`COMPILE_REPS`] host-only compiles, milliseconds.
    compile_ms: f64,
    /// `wukong_parallel_for` call sites in the *pre-optimization* MIR — the GATE-BLIND witness. A
    /// recognizer or an outliner leaves no trace any differential or invariance gate can see, so the
    /// count is carried alongside every timing and printed with it.
    regions: usize,
    /// Recognized `wukong_*` kernel calls other than `wukong_parallel_for`. Must be **zero** for
    /// every fixture here: one of these means the number is a measurement of the dispatcher.
    kernels: Vec<String>,
}

/// Lex -> parse -> sema -> `mir_build` -> `wukong_opt`, exactly as [`crate::compile`] does it for
/// `--backend=gpu-native`, including [`crate::lower_options_for`]'s two switches. Returns the
/// optimized program and, separately, the pre-optimization MIR text for the witness.
fn compile_fixture(path: &Path, opt: u8) -> Result<Fixture, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    let opts = crate::Options {
        input: path.to_path_buf(),
        backend: crate::BackendKind::GpuLower,
        opt_level: opt,
        ..crate::Options::default()
    };
    let lower_opts = crate::lower_options_for(&opts);

    let build = |interner: &mut Interner| -> Result<(Program, Program), String> {
        let mut sm = SourceMap::new();
        let id = sm.add(name.clone(), src.clone());
        let (tokens, ld) = wukong_lexer::tokenize(sm.source(id), id);
        if ld.iter().any(|d| d.is_error()) {
            return Err(format!("{name}: lex errors"));
        }
        let (module, pd) = wukong_parser::parse_module_tokens(&tokens, sm.source(id), interner);
        if pd.iter().any(|d| d.is_error()) {
            return Err(format!("{name}: parse errors"));
        }
        let (sema, sd) = wukong_sema::check(&module, interner);
        if sd.iter().any(|d| d.is_error()) {
            return Err(format!("{name}: sema errors: {sd:?}"));
        }
        let (program, md) =
            wukong_mir_build::lower_program_with(&module, &sema, interner, lower_opts);
        if md.iter().any(|d| d.is_error()) {
            return Err(format!("{name}: mir_build errors: {md:?}"));
        }
        let high = program.clone();
        let mut low = program;
        wukong_opt::optimize(&mut low, opt);
        Ok((high, low))
    };

    let mut interner = Interner::new();
    let (high, program) = build(&mut interner)?;

    // The reading: minimum over repeats, each from a fresh interner so no stage is warm.
    let mut compile_ms = f64::INFINITY;
    for _ in 0..COMPILE_REPS {
        let mut it = Interner::new();
        let t = Instant::now();
        let _ = build(&mut it)?;
        compile_ms = compile_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }

    let (regions, kernels) = witness(&high, &interner);
    let entry = interner.intern("main");
    if program.function(entry).is_none() {
        return Err(format!("{name}: no `main`"));
    }

    Ok(Fixture {
        name,
        program,
        interner,
        entry,
        compile_ms,
        regions,
        kernels,
    })
}

/// The GATE-BLIND witness, read off the **pre-optimization** MIR: how many `@parallel` regions were
/// outlined, and which recognized kernels (if any) claimed a loop.
fn witness(high: &Program, interner: &Interner) -> (usize, Vec<String>) {
    let mut regions = 0usize;
    let mut kernels: Vec<String> = Vec::new();
    for f in &high.funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                if let wukong_mir::Op::Call { func, .. } = &inst.op {
                    let n = interner.resolve(*func);
                    if n == "wukong_parallel_for" {
                        regions += 1;
                    } else if n.starts_with("wukong_") && !kernels.iter().any(|k| k == n) {
                        kernels.push(n.to_string());
                    }
                }
            }
        }
    }
    kernels.sort();
    (regions, kernels)
}

/// The CPU peer for the same program and the same MIR: `wukong_codegen_cranelift::jit_run`, whole
/// call, minimum over `reps`. Returns the milliseconds and the program's stdout, so the caller can
/// gate the timing on the two backends having agreed.
fn time_cpu(fx: &Fixture, reps: usize) -> Result<(f64, Vec<u8>), String> {
    let mut best = f64::INFINITY;
    let mut out = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        let (code, stdout) = wukong_codegen_cranelift::jit_run(&fx.program, fx.entry, &fx.interner)
            .map_err(|e| format!("{}: cranelift jit: {e}", fx.name))?;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        if code != 0 {
            return Err(format!("{}: cpu exit {code}", fx.name));
        }
        out = stdout;
    }
    Ok((best, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The round.** `#[ignore]`d because it needs the local device and takes real time; run it as
    /// its own step, from a `--release` build, with the box on AC and idle:
    ///
    /// ```text
    /// cargo test -p wukong_driver --features gpu --release --lib lang_ruler_round -- --ignored --nocapture
    /// ```
    ///
    /// It publishes nothing on its own — it prints a table. The numbers that reach `CHANGELOG.md`
    /// are read off that table by a human who also read the floor.
    #[test]
    #[ignore]
    fn lang_ruler_round() {
        let dir = lang_bench_dir();
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "wk"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no fixtures in {}", dir.display());

        // ---- compile every fixture once, and print the witness before any timing ----
        let fixtures: Vec<Fixture> = files
            .iter()
            .map(|p| compile_fixture(p, 2).unwrap_or_else(|e| panic!("{e}")))
            .collect();

        println!("\n=== bench/lang witness (--emit=mir-high, -O2) ===");
        println!(
            "{:<14} {:>8}  {:>7}  recognized kernels",
            "fixture", "regions", "cmp ms"
        );
        for fx in &fixtures {
            println!(
                "{:<14} {:>8}  {:>7.2}  {}",
                fx.name,
                fx.regions,
                fx.compile_ms,
                if fx.kernels.is_empty() {
                    "none (good)".to_string()
                } else {
                    fx.kernels.join(", ")
                }
            );
            assert!(
                fx.kernels.is_empty(),
                "{}: a recognizer claimed a loop, so this fixture measures the dispatcher and not \
                 the language. `@generic` is missing, or a new recognizer reaches this shape.",
                fx.name
            );
        }

        // ---- the CPU peer, and the correctness gate on every timing below ----
        println!("\n=== CPU (cranelift jit_run, whole call, min of {DEVICE_REPS}) ===");
        let mut cpu: Vec<(String, f64, Vec<u8>)> = Vec::new();
        for fx in &fixtures {
            let (ms, out) = time_cpu(fx, DEVICE_REPS).unwrap_or_else(|e| panic!("{e}"));
            println!("{:<14} {:>9.3} ms", fx.name, ms);
            cpu.push((fx.name.clone(), ms, out));
        }

        // ---- gpu-native, one full resident wave ----
        println!("\n=== gpu-native (megakernel, full resident wave, min of {DEVICE_REPS}) ===");
        println!(
            "{:<14} {:>10} {:>10} {:>10} {:>7} {:>6} {:>9}",
            "fixture", "kernel ms", "k max ms", "launch ms", "grid", "block", "frame MiB"
        );
        let mut wave: Vec<(String, f64)> = Vec::new();
        for (fx, (_, _, cpu_out)) in fixtures.iter().zip(&cpu) {
            let t = megakernel::time_mega(&fx.program, fx.entry, &fx.interner, DEVICE_REPS, None)
                .unwrap_or_else(|e| panic!("{}: {e}", fx.name));
            let Some(t) = t else {
                panic!(
                    "{}: not megakernel eligible, so it has no gpu-native time at all. Reporting a \
                     single-thread number in its place would compare two different kernels.",
                    fx.name
                );
            };
            assert_eq!(t.exit_code, 0, "{}: gpu exit {}", fx.name, t.exit_code);
            assert_eq!(
                String::from_utf8_lossy(&t.stdout),
                String::from_utf8_lossy(cpu_out),
                "{}: gpu-native and CPU disagree — a timing of a wrong kernel is not a measurement",
                fx.name
            );
            println!(
                "{:<14} {:>10.4} {:>10.4} {:>10.4} {:>7} {:>6} {:>9.1}",
                fx.name,
                t.kernel_ms,
                t.kernel_ms_max,
                t.launch_ms,
                t.grid,
                t.block,
                t.frame_bytes as f64 / (1024.0 * 1024.0),
            );
            wave.push((fx.name.clone(), t.kernel_ms));
        }

        // ---- what each fixture's own loop cost: the difference against `base` ----
        let base = wave
            .iter()
            .find(|(n, _)| n == "base")
            .map(|(_, ms)| *ms)
            .expect("bench/lang must contain base.wk — every number here is a difference");
        println!("\n=== the loop alone (kernel_ms - base), and its bandwidth ===");
        println!("{:<14} {:>12} {:>10}", "fixture", "loop ms", "GB/s");
        const N: f64 = (1u64 << 22) as f64;
        for (name, ms) in &wave {
            if name == "base" {
                continue;
            }
            let loop_ms = ms - base;
            // Bytes are read off the fixture's source, stated in its header, not counted here.
            let bytes = match name.as_str() {
                "write_grid" => 4.0 * N, // N writes
                "saxpy_lane" => 8.0 * N, // N reads + N writes
                _ => f64::NAN,
            };
            let gbs = bytes / (loop_ms * 1e-3) / 1e9;
            println!("{name:<14} {loop_ms:>12.4} {gbs:>10.1}");
        }

        // ---- Phase 1's own A/B: one program, 1 CTA vs a full resident wave ----
        // Same source, same MIR, same PTX, same module — only `plan_grid`'s CTA count differs. `A`
        // and `C` are the SAME arm (one CTA, twice), so `C/A` is this shape's own noise floor and a
        // `B` that does not clear it publishes nothing.
        println!("\n=== Phase 1: grid-stride scaling on base.wk (twin protocol) ===");
        let fx = fixtures.iter().find(|f| f.name == "base").expect("base.wk");
        let plan = TwinPlan::default();
        let samples = run_rotated("lang_grid_stride", &plan, |arm| {
            let grid = match arm {
                Arm::Baseline | Arm::Control => Some(1),
                Arm::Contender => None,
            };
            let t = megakernel::time_mega(&fx.program, fx.entry, &fx.interner, DEVICE_REPS, grid)
                .expect("time_mega")
                .expect("base.wk is eligible");
            vec![("kernel_ms".to_string(), t.kernel_ms)]
        });
        let verdict = analyze(&samples, plan.bar);
        println!("{}", verdict.table());
    }

    /// The witness assertion above is the whole reason this harness is not a dispatcher benchmark,
    /// so pin the fixtures themselves: **every** `bench/lang/*.wk` must carry `@generic`, and must
    /// carry a `@parallel` anchor, or it cannot reach the device at this size. Device-free, so it
    /// runs in the ordinary `cargo test -p wukong_driver --features gpu --lib` step.
    #[test]
    fn every_lang_fixture_is_generic_and_anchored() {
        let dir = lang_bench_dir();
        let mut n = 0usize;
        for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
            let p = e.expect("dir entry").path();
            if p.extension().is_none_or(|x| x != "wk") {
                continue;
            }
            let src = std::fs::read_to_string(&p).expect("read fixture");
            let code: String = src
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                code.contains("@generic"),
                "{}: without `@generic` the recognizers claim these loops and the round measures \
                 the dispatcher",
                p.display()
            );
            assert!(
                code.contains("@parallel"),
                "{}: without a `@parallel` anchor the program is not megakernel eligible, falls to \
                 the single-thread lowering, and its launch is rejected outright at this size",
                p.display()
            );
            n += 1;
        }
        assert!(
            n >= 3,
            "expected the base + contender + finding trio, saw {n}"
        );
    }

    /// `base.wk` is load-bearing: the round subtracts it from every other fixture, so its absence
    /// would turn every published difference into a raw `kernel_ms` that is mostly `[0.0; N]`.
    #[test]
    fn the_control_fixture_exists() {
        let base = lang_bench_dir().join("base.wk");
        assert!(
            base.exists(),
            "bench/lang/base.wk is the control every number in the round is a difference against"
        );
    }
}
