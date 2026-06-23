//! Whole-program **cooperative megakernel** (Phase 8 / M13) — the compiler-only end-to-end lever a
//! kernel library cannot pull: compile an eligible Mercury program into **one** persistent
//! `.visible .entry` kernel run by a *block* of threads, with recognized ops executed cooperatively
//! across the block (no per-op kernel launches, activations resident in one shared `.global` frame).
//!
//! ## How it works (the single-block increment)
//! [`crate::fusion::analyze`] proves a program is *megakernel-safe* (data-independent control flow, so
//! the SPMD threads never diverge and deadlock a `bar.sync`; single entry function; recognized ops
//! only). [`crate::lower::emit_mega_ptx`] then lowers the entry SPMD: the alloca frame moves to one
//! shared `.global` buffer the whole block sub-allocates, every store / side effect is `tid==0`-only,
//! and each recognized op is `bar.sync`-bracketed and run *cooperatively* (a block-wide tree for
//! reductions) or `tid==0`-serial. [`try_run`] launches it as a single block of [`MEGA_BLOCK`]
//! threads and decodes the same print/exit context buffer the single-thread path uses — so the output
//! is byte-identical, while the cooperative ops use the whole block instead of one lane.
//!
//! This is **additive and opt-in**: [`try_run`] returns `Ok(None)` for any program it can't accelerate
//! (the caller falls back to the correct single-thread `--backend=gpu-native` path), and the existing
//! offload `--backend=gpu` path and plain `cargo test` are untouched. The megakernel is the spine the
//! op-graph fusion ([`crate::fusion`]) and the resident-chain comparison ([M13]) build on; cooperative
//! bodies for the remaining recognized ops (GEMM, vmath, norm) and multi-block grid execution are the
//! follow-on perf increments.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use cudarc::driver::{LaunchConfig, PushKernelArg};

use mercury_mir::Program;
use mercury_span::{Interner, Symbol};

use crate::lower::{self, MEGA_KERNEL_NAME};

/// The block size the megakernel launches: a single CTA of this many threads on one SM. A power of
/// two (the cooperative reduction tree requires it); 256 is a good occupancy point and matches the
/// `mrt_red_smem[1024]` scratch ceiling. Multi-block grid execution is a later increment.
pub const MEGA_BLOCK: u32 = 256;

/// Try to run `program`'s `entry` as the cooperative megakernel. Returns:
///  - `Ok(Some((exit, stdout)))` — it ran on the megakernel (eligible + launched);
///  - `Ok(None)` — not megakernel-eligible, an op declined (`UNSUPPORTED:`), or no device: the caller
///    must fall back to the single-thread path (which stays the correctness reference);
///  - `Err(_)` — a genuine JIT / launch / readback failure of an *eligible* program.
///
/// The single-thread path remains the oracle: this never changes a program's result, only how fast an
/// eligible one computes it (proven by `mega_corpus_matches_oracle`).
pub fn try_run(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> Result<Option<(i64, Vec<u8>)>, String> {
    let plan = crate::fusion::analyze(program, entry, interner);
    if !plan.eligible {
        return Ok(None);
    }
    let Some(entry_fn) = program.function(entry) else {
        return Ok(None);
    };
    let frame_bytes = lower::mega_frame_bytes(entry_fn);

    let ptx = match lower::emit_mega_ptx(program, entry, interner) {
        Ok(p) => p,
        // A recognized op declined (e.g. an unsupported vmath op code) -> fall back, don't fail.
        Err(e) if e.starts_with(lower::UNSUPPORTED) => return Ok(None),
        Err(e) => return Err(e),
    };

    let mut guard = crate::gpu::gpu();
    let Some(g) = guard.as_mut() else {
        return Ok(None); // no device: let the single-thread path surface the helpful error
    };
    let r = launch_mega(g, &ptx, frame_bytes)?;
    Ok(Some(r))
}

/// JIT + launch one mega PTX module as a single block of [`MEGA_BLOCK`] threads, returning the
/// decoded `(exit_code, stdout)`. The frame buffer is a fresh zeroed `.global` slab the block shares.
fn launch_mega(
    g: &mut crate::gpu::Gpu,
    ptx: &str,
    frame_bytes: u64,
) -> Result<(i64, Vec<u8>), String> {
    let mut h = DefaultHasher::new();
    ptx.hash(&mut h);
    let key: &'static str = Box::leak(format!("mega_{:016x}", h.finish()).into_boxed_str());

    if std::env::var_os("MERCURY_GPU_DUMP_PTX").is_some() {
        eprintln!("--- gpu-mega PTX ---\n{ptx}\n--- end PTX ---");
    }

    let f = g.function(key, ptx, MEGA_KERNEL_NAME).map_err(|e| {
        let p = std::env::temp_dir().join(format!("{key}.ptx"));
        let _ = std::fs::write(&p, ptx);
        format!("gpu-mega JIT/load failed: {e:?}\n  (PTX written to {})", p.display())
    })?;

    let host = lower::new_ctx_host();
    let mut ctx_d = g
        .stream
        .memcpy_stod(&host)
        .map_err(|e| format!("gpu-mega ctx alloc failed: {e:?}"))?;
    // One shared frame for the whole block (>=8 bytes so a frame-less program still allocs cleanly).
    let mut frame_d = g
        .stream
        .alloc_zeros::<u8>(frame_bytes.max(8) as usize)
        .map_err(|e| format!("gpu-mega frame alloc failed: {e:?}"))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (MEGA_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = g.stream.launch_builder(&f);
    b.arg(&mut ctx_d);
    b.arg(&mut frame_d);
    unsafe {
        b.launch(cfg)
            .map_err(|e| format!("gpu-mega launch failed: {e:?}"))?;
    }

    let out = g
        .stream
        .memcpy_dtov(&ctx_d)
        .map_err(|e| format!("gpu-mega readback failed: {e:?}"))?;
    lower::decode_ctx(&out)
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use mercury_span::SourceMap;
    use std::path::PathBuf;

    fn build(src: &str, opt: u8) -> Option<(Program, Interner)> {
        let mut sm = SourceMap::new();
        let id = sm.add("mega_gate.mer".to_string(), src.to_string());
        let (tokens, ld) = mercury_lexer::tokenize(sm.source(id), id);
        if ld.iter().any(|d| d.is_error()) {
            return None;
        }
        let mut interner = Interner::new();
        let (module, pd) = mercury_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        if pd.iter().any(|d| d.is_error()) {
            return None;
        }
        let (sema, sd) = mercury_sema::check(&module, &interner);
        if sd.iter().any(|d| d.is_error()) {
            return None;
        }
        let (mut program, md) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        if md.iter().any(|d| d.is_error()) {
            return None;
        }
        mercury_opt::optimize(&mut program, opt);
        Some((program, interner))
    }

    fn line_matches(g: &str, c: &str) -> bool {
        if g == c {
            return true;
        }
        match (g.trim().parse::<f64>(), c.trim().parse::<f64>()) {
            (Ok(a), Ok(b)) => {
                let diff = (a - b).abs();
                diff <= 1e-3 || diff <= 1e-2 * b.abs().max(a.abs())
            }
            _ => false,
        }
    }

    fn outputs_match(gpu: &[u8], cpu: &[u8]) -> bool {
        let gs = String::from_utf8_lossy(gpu);
        let cs = String::from_utf8_lossy(cpu);
        let gl: Vec<&str> = gs.lines().collect();
        let cl: Vec<&str> = cs.lines().collect();
        gl.len() == cl.len() && gl.iter().zip(&cl).all(|(a, b)| line_matches(a, b))
    }

    /// Every **megakernel-eligible** `tests/run` program, run through the cooperative megakernel,
    /// matches the interpreter oracle (tolerance for floats, exact otherwise) at both -O0 and -O3.
    /// Ineligible programs are skipped here (the single-thread gate covers them); this proves the
    /// cooperative path is correct on the subset it accelerates — the prerequisite to wiring it into
    /// `jit_run` as the default for eligible programs.
    #[test]
    fn mega_corpus_matches_oracle() {
        if crate::gpu::gpu().is_none() {
            eprintln!("skip mega_corpus_matches_oracle: no CUDA device");
            return;
        }
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/run");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("read tests/run")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "mer").unwrap_or(false))
            .collect();
        files.sort();

        let mut eligible = 0usize;
        let mut ran = 0usize;
        let mut mismatches: Vec<String> = Vec::new();

        for path in &files {
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            let src = std::fs::read_to_string(path).unwrap();
            for opt in [0u8, 3u8] {
                let Some((program, mut interner)) = build(&src, opt) else {
                    continue;
                };
                let entry = interner.intern("main");
                if program.function(entry).is_none() {
                    continue;
                }
                if !crate::fusion::analyze(&program, entry, &interner).eligible {
                    continue;
                }
                eligible += 1;
                let cpu = mercury_interp::run_with_output(&program, entry, &interner);
                let gpu = try_run(&program, entry, &interner);
                match (&cpu, &gpu) {
                    (Err(_), Ok(None)) | (Ok(_), Ok(None)) => {} // declined at launch: not a miscompile
                    (Err(_), Err(_)) => {}                        // both error (e.g. assert) -> agree
                    (Ok((ce, co)), Ok(Some((ge, go)))) => {
                        ran += 1;
                        if ce != ge || !outputs_match(go, co) {
                            mismatches.push(format!(
                                "{name}@O{opt}: cpu=({ce},{:?}) mega=({ge},{:?})",
                                String::from_utf8_lossy(co),
                                String::from_utf8_lossy(go)
                            ));
                        }
                    }
                    (Ok(_), Err(e)) => mismatches.push(format!("{name}@O{opt}: mega errored: {e}")),
                    (Err(ce), Ok(Some((ge, _)))) => {
                        mismatches.push(format!("{name}@O{opt}: cpu err `{ce}` but mega ok ({ge})"))
                    }
                }
            }
        }

        eprintln!(
            "\n=== megakernel: {ran} ran / {eligible} eligible program-configs match the interp oracle ===",
        );
        assert!(
            mismatches.is_empty(),
            "megakernel disagrees with the interpreter oracle:\n{}",
            mismatches.join("\n")
        );
        assert!(ran > 0, "no eligible program ran on the megakernel — pipeline broken");
    }

    /// Same-run latency A/B: the cooperative megakernel vs the single-thread lowering on a
    /// reduction-heavy program (a `[N]` dot reduced `REPEAT` times). Both produce byte-identical
    /// output (checksum cross-check) before any timing counts; we then report the clock-invariant
    /// ratio (single / mega) best-of-N back-to-back in one process — never an absolute ms (≈7× clock
    /// swing on this part, per the honesty law). The cooperative tree turns each reduction's O(N)
    /// serial fold into O(N/block + log block), so the win grows with N·REPEAT.
    #[test]
    #[ignore = "perf bench; run with --ignored --nocapture"]
    fn mega_vs_single_reduce() {
        use std::time::Instant;
        if crate::gpu::gpu().is_none() {
            eprintln!("skip mega_vs_single_reduce: no CUDA device");
            return;
        }
        // N small enough for the single-thread .local frame, REPEAT large enough that compute
        // dominates fixed launch/alloc overhead. dot(ones,ones)=N, acc=N*REPEAT.
        const N: usize = 4096;
        const REPEAT: usize = 2000;
        let src = format!(
            r#"module bench
@parallel
fn dotp(x: [f32; {N}], y: [f32; {N}], o: [f32; 1]) {{
    let mut s: f32 = 0.0;
    for k in 0..{N} {{ s = s + x[k] * y[k]; }}
    o[0] = s;
}}
fn main() -> i32 {{
    let mut x: [f32; {N}] = [1.0; {N}];
    let mut y: [f32; {N}] = [1.0; {N}];
    let mut o: [f32; 1] = [0.0; 1];
    let mut acc: f32 = 0.0;
    let mut r: i32 = 0;
    while r < {REPEAT} {{ dotp(x, y, o); acc = acc + o[0]; r = r + 1; }}
    print(acc as i32);
    return 0;
}}
"#
        );
        // -O2 so the @parallel `dotp` inlines into `main` (-> main calls the recognized reduce
        // directly, making it megakernel-eligible). The opaque `mercury_sreduce_*` call has memory
        // side effects, so LICM keeps it in the loop -> the reduce work happens REPEAT times (the
        // timing below confirms it scales with REPEAT).
        let (program, mut interner) = build(&src, 2).expect("frontend ok");
        let entry = interner.intern("main");
        assert!(
            crate::fusion::analyze(&program, entry, &interner).eligible,
            "bench program must be megakernel-eligible"
        );

        // Correctness cross-check FIRST: mega and single-thread must agree (and with the oracle).
        let mega0 = try_run(&program, entry, &interner).expect("mega run").expect("mega eligible");
        let single0 = lower::jit_run_single(&program, entry, &interner).expect("single run");
        let oracle = mercury_interp::run_with_output(&program, entry, &interner).expect("interp");
        assert_eq!(mega0.0, single0.0, "exit codes differ");
        assert!(outputs_match(&mega0.1, &single0.1), "mega vs single output differs");
        assert!(outputs_match(&mega0.1, &oracle.1), "mega vs oracle output differs");
        eprintln!("checksum (mega==single==oracle): {}", String::from_utf8_lossy(&mega0.1).trim());

        // Warm the JIT/module caches for both paths, then best-of-N (smallest = least contention).
        let _ = try_run(&program, entry, &interner);
        let _ = lower::jit_run_single(&program, entry, &interner);
        let iters = 10;
        let best = |f: &dyn Fn()| {
            let mut b = f64::INFINITY;
            for _ in 0..iters {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let mega_t = best(&|| {
            let _ = try_run(&program, entry, &interner).unwrap().unwrap();
        });
        let single_t = best(&|| {
            let _ = lower::jit_run_single(&program, entry, &interner).unwrap();
        });
        eprintln!(
            "\n=== mega_vs_single_reduce  N={N} REPEAT={REPEAT} ({} reductions, {} fma) ===\n\
             single-thread: {:.3} ms   megakernel(256t): {:.3} ms   speedup: {:.2}x (same-run ratio)",
            REPEAT,
            (N * REPEAT) as f64,
            single_t * 1e3,
            mega_t * 1e3,
            single_t / mega_t,
        );
    }

    /// Same-run A/B for the **chunked-cooperative elementwise** path: a SiLU activation over `[N]`
    /// applied `REPEAT` times (a feedback store keeps the optimizer from hoisting it). Output is
    /// cross-checked mega==single==oracle (tolerance) before timing; the ratio (single/mega) is the
    /// clock-invariant win of running the elementwise kernel across the block vs one thread.
    #[test]
    #[ignore = "perf bench; run with --ignored --nocapture"]
    fn mega_vs_single_vmath() {
        use std::time::Instant;
        if crate::gpu::gpu().is_none() {
            eprintln!("skip mega_vs_single_vmath: no CUDA device");
            return;
        }
        const N: usize = 8192;
        const REPEAT: usize = 600;
        // o[i] = silu(x[i]) is recognized as mercury_vmath_f32; the feedback x[0]=o[0] makes the
        // REPEAT loop non-hoistable so the elementwise work runs REPEAT times.
        let src = format!(
            r#"module bench
fn main() -> i32 {{
    let mut x: [f32; {N}] = [0.5; {N}];
    let mut o: [f32; {N}] = [0.0; {N}];
    let mut r: i32 = 0;
    while r < {REPEAT} {{
        for i in 0..{N} {{ o[i] = silu(x[i]); }}
        x[0] = o[0];
        r = r + 1;
    }}
    print((o[1] * 1000.0) as i32);
    return 0;
}}
"#
        );
        let (program, mut interner) = build(&src, 2).expect("frontend ok");
        let entry = interner.intern("main");
        if !crate::fusion::analyze(&program, entry, &interner).eligible {
            eprintln!("skip mega_vs_single_vmath: program not eligible (recognizer/opt shape)");
            return;
        }
        let mega0 = try_run(&program, entry, &interner).expect("mega").expect("eligible");
        let single0 = lower::jit_run_single(&program, entry, &interner).expect("single");
        let oracle = mercury_interp::run_with_output(&program, entry, &interner).expect("interp");
        assert!(outputs_match(&mega0.1, &single0.1), "mega vs single differ");
        assert!(outputs_match(&mega0.1, &oracle.1), "mega vs oracle differ");
        eprintln!("checksum (mega==single==oracle): {}", String::from_utf8_lossy(&mega0.1).trim());

        let _ = try_run(&program, entry, &interner);
        let _ = lower::jit_run_single(&program, entry, &interner);
        let best = |f: &dyn Fn()| {
            let mut b = f64::INFINITY;
            for _ in 0..10 {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let mega_t = best(&|| {
            let _ = try_run(&program, entry, &interner).unwrap().unwrap();
        });
        let single_t = best(&|| {
            let _ = lower::jit_run_single(&program, entry, &interner).unwrap();
        });
        eprintln!(
            "\n=== mega_vs_single_vmath  N={N} REPEAT={REPEAT} ({} silu) ===\n\
             single-thread: {:.3} ms   megakernel(256t): {:.3} ms   speedup: {:.2}x (same-run ratio)",
            (N * REPEAT) as f64,
            single_t * 1e3,
            mega_t * 1e3,
            single_t / mega_t,
        );
    }

    /// Same-run A/B for the **row-chunked-cooperative GEMM** path: `C[M,N] = A[M,K]·Bᵀ` reduced
    /// `REPEAT` times. The megakernel partitions the M output rows across the block (each thread runs
    /// the serial `mrt_sgemm_nt` over its rows); the single-thread path does all M rows in one lane.
    /// Output cross-checked mega==single==oracle before the ratio is reported.
    #[test]
    #[ignore = "perf bench; run with --ignored --nocapture"]
    fn mega_vs_single_gemm() {
        use std::time::Instant;
        if crate::gpu::gpu().is_none() {
            eprintln!("skip mega_vs_single_gemm: no CUDA device");
            return;
        }
        // M rows chunked across the 256-thread block; K/N kept modest so the single-thread .local
        // frame fits. Constant inputs -> out[i,j]=K, acc=REPEAT*K exactly (no precision drift).
        const M: usize = 256;
        const K: usize = 64;
        const N: usize = 64;
        const REPEAT: usize = 16;
        let src = format!(
            r#"module bench
fn linear(x: [f32; {mk}], w: [f32; {nk}], out: [f32; {mn}]) {{
    for i in 0..{M} {{
        for j in 0..{N} {{
            let mut s: f32 = 0.0;
            for p in 0..{K} {{ s = s + x[i * {K} + p] * w[j * {K} + p]; }}
            out[i * {N} + j] = s;
        }}
    }}
}}
fn main() -> i32 {{
    let x: [f32; {mk}] = [1.0; {mk}];
    let w: [f32; {nk}] = [1.0; {nk}];
    let mut out: [f32; {mn}] = [0.0; {mn}];
    let mut acc: f32 = 0.0;
    let mut r: i32 = 0;
    while r < {REPEAT} {{ linear(x, w, out); acc = acc + out[0]; r = r + 1; }}
    print(acc as i32);
    return 0;
}}
"#,
            mk = M * K,
            nk = N * K,
            mn = M * N,
        );
        let (program, mut interner) = build(&src, 2).expect("frontend ok");
        let entry = interner.intern("main");
        if !crate::fusion::analyze(&program, entry, &interner).eligible {
            eprintln!("skip mega_vs_single_gemm: not eligible (recognizer/opt shape)");
            return;
        }
        let mega0 = try_run(&program, entry, &interner).expect("mega").expect("eligible");
        let single0 = lower::jit_run_single(&program, entry, &interner).expect("single");
        let oracle = mercury_interp::run_with_output(&program, entry, &interner).expect("interp");
        assert!(outputs_match(&mega0.1, &single0.1), "mega vs single differ");
        assert!(outputs_match(&mega0.1, &oracle.1), "mega vs oracle differ");
        eprintln!("checksum (mega==single==oracle): {}", String::from_utf8_lossy(&mega0.1).trim());

        let _ = try_run(&program, entry, &interner);
        let _ = lower::jit_run_single(&program, entry, &interner);
        let best = |f: &dyn Fn()| {
            let mut b = f64::INFINITY;
            for _ in 0..8 {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let mega_t = best(&|| {
            let _ = try_run(&program, entry, &interner).unwrap().unwrap();
        });
        let single_t = best(&|| {
            let _ = lower::jit_run_single(&program, entry, &interner).unwrap();
        });
        eprintln!(
            "\n=== mega_vs_single_gemm  M={M} K={K} N={N} REPEAT={REPEAT} ({} fma) ===\n\
             single-thread: {:.3} ms   megakernel(256t): {:.3} ms   speedup: {:.2}x (same-run ratio)",
            (M * K * N * REPEAT) as f64,
            single_t * 1e3,
            mega_t * 1e3,
            single_t / mega_t,
        );
    }

    /// **M13 — the megakernel's structural win: one launch vs the per-op library chain.** A workload
    /// of `REPEAT` reductions is run two ways, same-run, same launcher (`try_run`), so the only
    /// difference is launch structure:
    ///  - **megakernel**: the whole `REPEAT`-loop program in ONE launch, data resident in the shared
    ///    frame across every reduction (zero per-op launches / re-marshaling);
    ///  - **per-op chain (the offload / library model)**: a single-reduction program launched `REPEAT`
    ///    times — each op its own launch that re-materializes its inputs, exactly what the
    ///    `--backend=gpu` `GpuAccel` path does (per-call H2D + launch + D2H, no residency).
    ///
    /// The ratio (chain / mega) is the launch-overhead + residency win a kernel library *cannot* get
    /// without fusing the whole graph (Mirage/FlashFormer-class). Reported as a clock-invariant ratio
    /// only; correctness is the loop program's `acc = REPEAT*N` vs the chain's per-launch `N` summed.
    #[test]
    #[ignore = "perf bench (M13); run with --ignored --nocapture"]
    fn mega_vs_chain_reduce() {
        use std::time::Instant;
        if crate::gpu::gpu().is_none() {
            eprintln!("skip mega_vs_chain_reduce: no CUDA device");
            return;
        }
        const N: usize = 4096;
        const REPEAT: usize = 400;
        // Loop program: the whole chain in one megakernel launch. acc = REPEAT*N (dot of ones).
        let loop_src = format!(
            r#"module bench
@parallel
fn dotp(x: [f32; {N}], y: [f32; {N}], o: [f32; 1]) {{
    let mut s: f32 = 0.0;
    for k in 0..{N} {{ s = s + x[k] * y[k]; }}
    o[0] = s;
}}
fn main() -> i32 {{
    let mut x: [f32; {N}] = [1.0; {N}];
    let mut y: [f32; {N}] = [1.0; {N}];
    let mut o: [f32; 1] = [0.0; 1];
    let mut acc: f32 = 0.0;
    let mut r: i32 = 0;
    while r < {REPEAT} {{ dotp(x, y, o); acc = acc + o[0]; r = r + 1; }}
    print(acc as i32);
    return 0;
}}
"#
        );
        // Single-op program: one reduction per launch (the per-op chain element). prints N.
        let one_src = format!(
            r#"module bench
@parallel
fn dotp(x: [f32; {N}], y: [f32; {N}], o: [f32; 1]) {{
    let mut s: f32 = 0.0;
    for k in 0..{N} {{ s = s + x[k] * y[k]; }}
    o[0] = s;
}}
fn main() -> i32 {{
    let mut x: [f32; {N}] = [1.0; {N}];
    let mut y: [f32; {N}] = [1.0; {N}];
    let mut o: [f32; 1] = [0.0; 1];
    dotp(x, y, o);
    print(o[0] as i32);
    return 0;
}}
"#
        );
        let (loop_p, mut li) = build(&loop_src, 2).expect("frontend");
        let loop_e = li.intern("main");
        let (one_p, mut oi) = build(&one_src, 2).expect("frontend");
        let one_e = oi.intern("main");
        assert!(crate::fusion::analyze(&loop_p, loop_e, &li).eligible);
        assert!(crate::fusion::analyze(&one_p, one_e, &oi).eligible);

        // Correctness: megakernel computes the whole chain (acc=REPEAT*N); each chain element computes
        // N; the two agree when the chain elements are summed -> the megakernel fused them losslessly.
        let mega0 = try_run(&loop_p, loop_e, &li).expect("mega").expect("eligible");
        let one0 = try_run(&one_p, one_e, &oi).expect("one").expect("eligible");
        let mega_val: i64 = String::from_utf8_lossy(&mega0.1).trim().parse().unwrap();
        let one_val: i64 = String::from_utf8_lossy(&one0.1).trim().parse().unwrap();
        assert_eq!(mega_val, one_val * REPEAT as i64, "mega chain != sum of per-op results");
        eprintln!("checksum: megakernel acc={mega_val} == {REPEAT} * per-op {one_val}");

        // Warm both, then best-of-N.
        let _ = try_run(&loop_p, loop_e, &li);
        let _ = try_run(&one_p, one_e, &oi);
        let best = |f: &dyn Fn()| {
            let mut b = f64::INFINITY;
            for _ in 0..6 {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let mega_t = best(&|| {
            let _ = try_run(&loop_p, loop_e, &li).unwrap().unwrap();
        });
        // The per-op chain: REPEAT separate launches, each re-marshaling its inputs (offload model).
        let chain_t = best(&|| {
            for _ in 0..REPEAT {
                let _ = try_run(&one_p, one_e, &oi).unwrap().unwrap();
            }
        });
        eprintln!(
            "\n=== M13 mega_vs_chain_reduce  N={N} REPEAT={REPEAT} ===\n\
             per-op chain ({REPEAT} launches): {:.3} ms   megakernel (1 launch): {:.3} ms   \
             win: {:.2}x (same-run ratio)",
            chain_t * 1e3,
            mega_t * 1e3,
            chain_t / mega_t,
        );
    }
}
