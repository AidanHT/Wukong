//! `mercury_driver` — orchestrates the compilation pipeline.
//!
//! The driver owns the [`SourceMap`] and the [`DiagnosticSink`], runs each stage in order, and
//! honors the requested `--emit` target. Stages are wired in as they come online; today the
//! lexer is connected and `--emit=tokens` works end to end.

use std::path::PathBuf;

use mercury_diag::{Diagnostic, DiagnosticSink, Renderer};
use mercury_span::{Interner, SourceMap};

pub use mercury_diag::{all_explanations, explain, Explanation};

/// The GPU offloading bridge (`--backend=gpu`), compiled only with the `gpu` feature.
#[cfg(feature = "gpu")]
mod gpu_accel;

/// Which intermediate (or final) artifact the user asked to produce.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EmitStage {
    Tokens,
    Ast,
    MirHigh,
    Mir,
    LlvmIr,
    Obj,
    Exe,
}

impl EmitStage {
    pub fn parse(s: &str) -> Option<EmitStage> {
        Some(match s {
            "tokens" => EmitStage::Tokens,
            "ast" => EmitStage::Ast,
            "mir-high" => EmitStage::MirHigh,
            "mir" | "mir-low" => EmitStage::Mir,
            "llvm-ir" => EmitStage::LlvmIr,
            "obj" => EmitStage::Obj,
            "exe" => EmitStage::Exe,
            _ => return None,
        })
    }
}

/// How diagnostics are presented to the user.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ErrorFormat {
    /// rustc-style annotated terminal output.
    Human,
    /// One JSON object per diagnostic, on its own line (JSON Lines).
    Json,
}

/// Which backend executes (`--run`) or emits native code.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendKind {
    /// The tree-walking MIR interpreter (zero dependencies, the reference oracle).
    Interp,
    /// Cranelift native codegen — JIT for `--run`, object/exe for `--emit`.
    Native,
    /// GPU offload (`--features gpu`): the interpreter tree-walks the program but runs recognized
    /// GEMM / norm / activation / reduction calls on the local CUDA device. Tolerance-gated against
    /// the interpreter oracle (the CPU↔GPU boundary cannot be bit-exact).
    Gpu,
    /// General GPU lowering (`--features gpu`, `--backend=gpu-native`): lowers the *whole* program's
    /// MIR to PTX and runs it on the device (Phase 4) — not the recognizer offload. Additive to and
    /// independent of [`BackendKind::Gpu`]; tolerance-gated against the interpreter oracle.
    GpuLower,
}

/// Compiler invocation options, normally produced by the CLI.
#[derive(Clone, Debug)]
pub struct Options {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub emit: EmitStage,
    pub run: bool,
    pub opt_level: u8,
    pub color: bool,
    pub error_format: ErrorFormat,
    /// Which backend to run/emit with. `--run` defaults to the interpreter (the reference oracle);
    /// native object/exe emission always uses Cranelift.
    pub backend: BackendKind,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            input: PathBuf::new(),
            output: None,
            emit: EmitStage::Exe,
            run: false,
            opt_level: 0,
            color: true,
            error_format: ErrorFormat::Human,
            backend: BackendKind::Interp,
        }
    }
}

/// Process exit codes used by the driver and CLI.
pub mod exit {
    pub const OK: i32 = 0;
    pub const COMPILE_ERROR: i32 = 1;
    pub const UNIMPLEMENTED: i32 = 2;
    pub const IO_ERROR: i32 = 3;
}

/// Run the pipeline for `opts`, printing artifacts to stdout and diagnostics to stderr.
/// Returns a process exit code.
pub fn compile(opts: &Options) -> i32 {
    let src = match std::fs::read_to_string(&opts.input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", opts.input.display(), e);
            return exit::IO_ERROR;
        }
    };

    let mut sm = SourceMap::new();
    let id = sm.add(opts.input.display().to_string(), src);
    let renderer = Renderer::new(opts.color);
    let mut sink = DiagnosticSink::new();

    // --- Lexing ---
    let (tokens, lex_diags) = mercury_lexer::tokenize(sm.source(id), id);
    for d in lex_diags {
        sink.emit(d);
    }

    if opts.emit == EmitStage::Tokens {
        render_all(opts.error_format, &renderer, &sink, &sm);
        print!("{}", mercury_lexer::dump(&tokens, sm.source(id)));
        return if sink.has_errors() {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    // --- Parsing ---
    let mut interner = Interner::new();
    let (module, parse_diags) =
        mercury_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
    for d in parse_diags {
        sink.emit(d);
    }
    render_all(opts.error_format, &renderer, &sink, &sm);

    if opts.emit == EmitStage::Ast {
        print!("{}", mercury_ast::print::print_module(&module, &interner));
        return if sink.has_errors() {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    if sink.has_errors() {
        return exit::COMPILE_ERROR;
    }

    // --- Semantic analysis (name resolution, type checking, shape checking) ---
    let (sema, sema_diags) = mercury_sema::check(&module, &interner);
    for d in &sema_diags {
        emit_diag(d, opts.error_format, &renderer, &sm);
    }
    if sema_diags.iter().any(|d| d.is_error()) {
        return exit::COMPILE_ERROR;
    }

    // --- MIR construction ---
    let (mut program, lower_diags) =
        mercury_mir_build::lower_program(&module, &sema, &mut interner);
    for d in &lower_diags {
        emit_diag(d, opts.error_format, &renderer, &sm);
    }

    // High MIR is the pre-optimization form. It is still dumped for debugging even when lowering
    // failed, since seeing the partial MIR is useful.
    if opts.emit == EmitStage::MirHigh {
        emit_mir(&program, &interner);
        return if lower_diags.iter().any(|d| d.is_error()) {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    // A construct codegen could not lower leaves the MIR invalid, so do not optimize, run, or hand
    // it to a backend — that would crash or miscompile. Stop with a clear error instead.
    if lower_diags.iter().any(|d| d.is_error()) {
        return exit::COMPILE_ERROR;
    }

    // --- Optimization ---
    mercury_opt::optimize(&mut program, opts.opt_level);

    // --- Run via the selected backend (interpreter by default, Cranelift JIT with --backend=native) ---
    if opts.run {
        let main = interner.intern("main");
        let result = match opts.backend {
            BackendKind::Interp => mercury_interp::run_with_output(&program, main, &interner),
            BackendKind::Native => mercury_codegen_cranelift::jit_run(&program, main, &interner),
            BackendKind::Gpu => run_on_gpu(&program, main, &interner),
            BackendKind::GpuLower => run_on_gpu_lower(&program, main, &interner),
        };
        return match result {
            Ok((exit_code, stdout)) => {
                use std::io::Write;
                let _ = std::io::stdout().write_all(&stdout);
                exit_code as i32
            }
            Err(e) => {
                eprintln!("error: {e}");
                exit::COMPILE_ERROR
            }
        };
    }

    if opts.emit == EmitStage::Mir {
        emit_mir(&program, &interner);
        return exit::OK;
    }

    // --- LLVM backend ---
    if opts.emit == EmitStage::LlvmIr {
        print!(
            "{}",
            mercury_codegen_llvm::emit_llvm_ir(&program, &interner)
        );
        return exit::OK;
    }
    if matches!(opts.emit, EmitStage::Obj | EmitStage::Exe) {
        return emit_native(&program, &interner, opts);
    }

    exit::OK
}

/// Execute `entry` on the GPU backend: the interpreter tree-walks the program while recognized kernel
/// calls run on the device (see [`gpu_accel`]). Built only with `--features gpu`.
#[cfg(feature = "gpu")]
fn run_on_gpu(
    program: &mercury_mir::Program,
    entry: mercury_span::Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    let mut guard = mercury_codegen_gpu::gpu();
    match guard.as_mut() {
        Some(g) => {
            let mut accel = gpu_accel::GpuAccel::new(g);
            mercury_interp::run_with_output_accel(program, entry, interner, &mut accel)
        }
        None => Err(
            "`--backend=gpu` requires a CUDA device, but none was reachable (the driver \
                     dlopens `nvcuda.dll`; check the NVIDIA driver is installed)"
                .into(),
        ),
    }
}

/// Without the `gpu` feature, `--backend=gpu` is a clear build-time-capability error rather than a
/// silent fallback (which would dishonestly run on the CPU when the user asked for the GPU).
#[cfg(not(feature = "gpu"))]
fn run_on_gpu(
    _program: &mercury_mir::Program,
    _entry: mercury_span::Symbol,
    _interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    Err("this `mercuryc` was built without GPU support; rebuild with `--features gpu` (or, from the \
         workspace, `-p mercuryc --features gpu`) to use `--backend=gpu`"
        .into())
}

/// Execute `entry` on the **general MIR→PTX** GPU backend (`--backend=gpu-native`, Phase 4): the
/// whole program is lowered to PTX and run on the device — distinct from the recognizer offload
/// ([`run_on_gpu`]), which it leaves untouched. Built only with `--features gpu`.
#[cfg(feature = "gpu")]
fn run_on_gpu_lower(
    program: &mercury_mir::Program,
    entry: mercury_span::Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    mercury_codegen_gpu::lower::jit_run(program, entry, interner)
}

/// Without the `gpu` feature, `--backend=gpu-native` is a clear build-capability error rather than a
/// silent CPU fallback (mirrors [`run_on_gpu`]).
#[cfg(not(feature = "gpu"))]
fn run_on_gpu_lower(
    _program: &mercury_mir::Program,
    _entry: mercury_span::Symbol,
    _interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    Err("this `mercuryc` was built without GPU support; rebuild with `--features gpu` (or, from the \
         workspace, `-p mercuryc --features gpu`) to use `--backend=gpu-native`"
        .into())
}

/// The C runtime linked into native executables: it backs the `print`/`assert` intrinsics that the
/// Cranelift object leaves as undefined imports. The JIT path binds the same symbols to Rust
/// functions instead, so the two stay in lockstep.
const MERCURY_RT_C: &str = "#include <stdio.h>\n\
#include <stdlib.h>\n\
#include <math.h>\n\
void mercury_rt_print_i64(long long x) { printf(\"%lld\\n\", x); }\n\
void mercury_rt_print_u64(unsigned long long x) { printf(\"%llu\\n\", x); }\n\
void mercury_rt_print_f64(double x) { printf(\"%g\\n\", x); }\n\
void mercury_rt_assert(long long c) { if (!c) { fprintf(stderr, \"assertion failed\\n\"); exit(101); } }\n\
double mercury_rt_fmod_f64(double a, double b) { return fmod(a, b); }\n\
float mercury_rt_fmod_f32(float a, float b) { return fmodf(a, b); }\n";

/// Emit a native object via Cranelift (no LLVM) and, for `--emit=exe`, link it with a small C
/// runtime using the system C compiler. `CC` overrides the compiler (default `cc`).
fn emit_native(program: &mercury_mir::Program, interner: &Interner, opts: &Options) -> i32 {
    use std::process::Command;

    let obj = match mercury_codegen_cranelift::emit_object(program, interner) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cranelift codegen failed: {e}");
            return exit::COMPILE_ERROR;
        }
    };

    let stem = opts
        .input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".to_string());

    let is_obj = opts.emit == EmitStage::Obj;
    let obj_path = if is_obj {
        opts.output
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("{stem}.o")))
    } else {
        PathBuf::from(format!("{stem}.o"))
    };
    if let Err(e) = std::fs::write(&obj_path, &obj) {
        eprintln!("error: could not write `{}`: {e}", obj_path.display());
        return exit::IO_ERROR;
    }
    if is_obj {
        eprintln!("wrote {}", obj_path.display());
        return exit::OK;
    }

    // exe: emit the C runtime next to the object and link them.
    let rt_path = PathBuf::from(format!("{stem}_rt.c"));
    if let Err(e) = std::fs::write(&rt_path, MERCURY_RT_C) {
        eprintln!("error: could not write `{}`: {e}", rt_path.display());
        return exit::IO_ERROR;
    }
    let out = opts
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("{stem}.exe")));
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let status = Command::new(&cc)
        .arg(&obj_path)
        .arg(&rt_path)
        .arg("-o")
        .arg(&out)
        .arg("-O2")
        .status();
    match status {
        Ok(s) if s.success() => {
            eprintln!("wrote {}", out.display());
            exit::OK
        }
        Ok(_) => {
            eprintln!(
                "error: `{cc}` failed to link the native object `{}`",
                obj_path.display()
            );
            exit::COMPILE_ERROR
        }
        Err(_) => {
            eprintln!(
                "error: could not run `{cc}` to link.\n\
                 note: the native object was written to `{}`.\n\
                 help: install a C compiler (gcc/clang/cc) or set CC, or use `--run` to execute.",
                obj_path.display()
            );
            exit::UNIMPLEMENTED
        }
    }
}

fn emit_mir(program: &mercury_mir::Program, interner: &Interner) {
    for f in &program.funcs {
        for ice in mercury_mir::verify::verify_function(f) {
            eprintln!("internal compiler error (MIR verify): {ice}");
        }
    }
    print!("{}", mercury_mir::print::print_program(program, interner));
}

/// Emit a single diagnostic to stderr in the requested format.
fn emit_diag(d: &Diagnostic, fmt: ErrorFormat, renderer: &Renderer, sm: &SourceMap) {
    match fmt {
        ErrorFormat::Human => eprintln!("{}", renderer.render(d, sm)),
        ErrorFormat::Json => eprintln!("{}", mercury_diag::to_json(d, sm)),
    }
}

fn render_all(fmt: ErrorFormat, renderer: &Renderer, sink: &DiagnosticSink, sm: &SourceMap) {
    for d in sink.diagnostics() {
        emit_diag(d, fmt, renderer, sm);
    }
}

/// End-to-end GPU-backend gate: the `--backend=gpu` path (offloading interpreter + [`gpu_accel`])
/// must match the pure-interpreter oracle within the CPU↔GPU tolerance over the **same** lowered MIR.
/// Only built with `--features gpu`; skips (does not fail) when no CUDA device is present.
#[cfg(all(test, feature = "gpu"))]
mod gpu_e2e_tests {
    use super::*;
    use mercury_codegen_gpu::diff::{assert_close, Rng};
    use mercury_span::SourceMap;

    /// Lex → parse → sema → mir_build → opt(2), asserting each stage is clean. The recognizers run in
    /// mir_build, so the resulting MIR already carries the `mercury_sgemm_nt` call the GPU offloads.
    fn build(src: &str) -> (mercury_mir::Program, Interner) {
        let mut sm = SourceMap::new();
        let id = sm.add("gpu_e2e.mer".to_string(), src.to_string());
        let (tokens, ld) = mercury_lexer::tokenize(sm.source(id), id);
        assert!(!ld.iter().any(|d| d.is_error()), "lex errors");
        let mut interner = Interner::new();
        let (module, pd) =
            mercury_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        assert!(!pd.iter().any(|d| d.is_error()), "parse errors");
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()), "sema errors");
        let (mut program, md) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(!md.iter().any(|d| d.is_error()), "mir_build errors");
        mercury_opt::optimize(&mut program, 2);
        (program, interner)
    }

    /// The `ijk` `C = A·Bᵀ` (nn.Linear) nest that the recognizer lowers to `mercury_sgemm_nt`.
    fn linear_src(m: usize, k: usize, n: usize) -> String {
        format!(
            "module m\nfn lin(a:[f32;{mk}],b:[f32;{nk}],c:[f32;{mn}]) {{ \
             for i in 0..{m} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for kk in 0..{k} {{ s = s + a[i*{k}+kk] * b[j*{k}+kk]; }} c[i*{n}+j] = s; }} }} }}",
            mk = m * k,
            nk = n * k,
            mn = m * n
        )
    }

    /// The `C = A·Bᵀ` nest followed by a separate elementwise activation loop over `C` — which the
    /// recognizer folds into one `mercury_sgemm_nt_epi` (bias-free, the SwiGLU/FFN `act(x·Wᵀ)` shape).
    /// `actname` is `relu`/`silu`/`gelu`.
    fn linear_act_src(actname: &str, m: usize, k: usize, n: usize) -> String {
        let cij = format!("c[i*{n}+j]");
        let act = match actname {
            "relu" => format!("fmax({cij}, 0.0)"),
            "silu" => format!("silu({cij})"),
            "gelu" => format!("gelu({cij})"),
            other => panic!("unknown activation {other}"),
        };
        format!(
            "module m\nfn lin(a:[f32;{mk}],b:[f32;{nk}],c:[f32;{mn}]) {{ \
             for i in 0..{m} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for kk in 0..{k} {{ s = s + a[i*{k}+kk] * b[j*{k}+kk]; }} c[i*{n}+j] = s; }} }} \
             for i in 0..{m} {{ for j in 0..{n} {{ c[i*{n}+j] = {act}; }} }} }}",
            mk = m * k,
            nk = n * k,
            mn = m * n
        )
    }

    /// The `C = A·Bᵀ` nest followed by a **bias-add (+ optional activation)** epilogue loop — which the
    /// recognizer folds into one `mercury_sgemm_nt_epi` with a *non-null* bias (the canonical
    /// `nn.Linear`/FFN `act(x·Wᵀ + bias)`). `actname` is `none` (affine Linear), `relu`, `silu`, `gelu`.
    /// The `bias` param sits between `b` and `c`, so the buffer list is `[a, b, bias, c]`.
    fn linear_bias_act_src(actname: &str, m: usize, k: usize, n: usize) -> String {
        let inner = format!("c[i*{n}+j] + bias[j]");
        let act = match actname {
            "none" => inner.clone(),
            "relu" => format!("fmax({inner}, 0.0)"),
            "silu" => format!("silu({inner})"),
            "gelu" => format!("gelu({inner})"),
            other => panic!("unknown activation {other}"),
        };
        format!(
            "module m\nfn lin(a:[f32;{mk}],b:[f32;{nk}],bias:[f32;{n}],c:[f32;{mn}]) {{ \
             for i in 0..{m} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for kk in 0..{k} {{ s = s + a[i*{k}+kk] * b[j*{k}+kk]; }} c[i*{n}+j] = s; }} }} \
             for i in 0..{m} {{ for j in 0..{n} {{ c[i*{n}+j] = {act}; }} }} }}",
            mk = m * k,
            nk = n * k,
            mn = m * n
        )
    }

    #[test]
    fn gpu_backend_linear_matches_interp_within_tol() {
        let mut guard = mercury_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                eprintln!("skip gpu_backend_linear_matches_interp_within_tol: no CUDA device");
                return;
            }
        };
        let mut rng = Rng::new(0x00C0FFEE);
        for &(m, k, n) in &[(8usize, 16usize, 8usize), (16, 32, 16), (32, 48, 24)] {
            let (program, mut interner) = build(&linear_src(m, k, n));
            let entry = interner.intern("lin");
            let a = rng.vec(m * k, -1.0, 1.0);
            let b = rng.vec(n * k, -1.0, 1.0);

            // CPU oracle (no accelerator).
            let (mut ac, mut bc, mut cc) = (a.clone(), b.clone(), vec![0f32; m * n]);
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ac, &mut bc, &mut cc];
                mercury_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
            }

            // GPU offload over the identical MIR + inputs.
            let (mut ag, mut bg, mut cg) = (a.clone(), b.clone(), vec![0f32; m * n]);
            let mut accel = gpu_accel::GpuAccel::new(&mut *g);
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ag, &mut bg, &mut cg];
                mercury_interp::run_kernel_f32_accel(
                    &program, entry, &mut bufs, &interner, &mut accel,
                )
                .unwrap();
            }
            assert!(
                accel.calls >= 1,
                "{m}x{k}x{n}: GPU offload never fired — the nest did not lower to sgemm_nt, so this \
                 would silently test CPU-vs-CPU"
            );

            // tol = c·√K·ε (the sgemm_matches_naive shape): both sides are f32, K-reduction order differs.
            let rel_tol = (16.0 * (k as f64).sqrt() * f32::EPSILON as f64).max(1e-5);
            let s = assert_close(&format!("linear {m}x{k}x{n}"), &cg, &cc, 1e-4, rel_tol);
            eprintln!(
                "gpu --backend linear {m}x{k}x{n}: {} GPU call(s), max_abs={:.2e} max_rel={:.2e}",
                accel.calls, s.max_abs, s.max_rel
            );
        }
    }

    /// **`act(matmul(x,w))` from Mercury source runs the fused tensor-core kernel** (Phase-2 fusion
    /// engine). The recognizer folds the matmul + activation loop into `mercury_sgemm_nt_epi`; the GPU
    /// `Accelerator` routes that to the single fused WMMA kernel (`gemm_nt_f16_sm_db_{relu,silu,gelu}`,
    /// the one that beats the cuBLAS GEMM+activation chain). Aligned shapes (M,N %64, K %16) so the
    /// tensor-core tiling accepts them. The offload must fire (`calls >= 1`, else it would silently
    /// test CPU-vs-CPU); the fp16 path is tolerance-gated against the f32 CPU oracle, not bit-exact.
    #[test]
    fn gpu_backend_fused_epilogue_matches_interp() {
        let mut guard = mercury_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                eprintln!("skip gpu_backend_fused_epilogue_matches_interp: no CUDA device");
                return;
            }
        };
        let mut rng = Rng::new(0x0FADE5);
        for actname in ["relu", "silu", "gelu"] {
            for &(m, k, n) in &[(64usize, 16usize, 64usize), (64, 32, 128)] {
                let (program, mut interner) = build(&linear_act_src(actname, m, k, n));
                let entry = interner.intern("lin");
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);

                // CPU oracle (no accelerator) — the f32 fused epilogue.
                let (mut ac, mut bc, mut cc) = (a.clone(), b.clone(), vec![0f32; m * n]);
                {
                    let mut bufs: [&mut [f32]; 3] = [&mut ac, &mut bc, &mut cc];
                    mercury_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
                }

                // GPU offload over the identical MIR + inputs.
                let (mut ag, mut bg, mut cg) = (a.clone(), b.clone(), vec![0f32; m * n]);
                let mut accel = gpu_accel::GpuAccel::new(&mut *g);
                {
                    let mut bufs: [&mut [f32]; 3] = [&mut ag, &mut bg, &mut cg];
                    mercury_interp::run_kernel_f32_accel(
                        &program, entry, &mut bufs, &interner, &mut accel,
                    )
                    .unwrap();
                }
                assert!(
                    accel.calls >= 1,
                    "{actname} {m}x{k}x{n}: fused epilogue never offloaded — the act(matmul) nest did \
                     not fold to sgemm_nt_epi or the GPU declined, so this would silently test CPU-vs-CPU"
                );

                // fp16 tensor-core path → fp16 tolerance (looser than the f32 GEMM bound).
                let s = assert_close(&format!("fused {actname} {m}x{k}x{n}"), &cg, &cc, 5e-2, 2e-2);
                eprintln!(
                    "gpu --backend fused {actname} {m}x{k}x{n}: {} GPU call(s), max_abs={:.2e} max_rel={:.2e}",
                    accel.calls, s.max_abs, s.max_rel
                );
            }
        }
    }

    /// **`act(matmul(x,w) + bias)` from Mercury source runs the fused tensor-core *bias* kernel** — the
    /// canonical `nn.Linear`/FFN epilogue (the affine `none` case is a plain biased Linear). The
    /// recognizer folds matmul + bias-add (+ activation) into `mercury_sgemm_nt_epi` with a non-null
    /// bias; the GPU `Accelerator` routes it to `gemm_nt_f16_sm_db_bias{,_relu,_silu,_gelu}` (which
    /// store each tile through SMEM to add the per-column bias the opaque WMMA fragment layout otherwise
    /// blocks). Aligned shapes; the offload must fire (`calls >= 1`, else it would silently test
    /// CPU-vs-CPU); fp16 path → tolerance-gated against the f32 CPU oracle, not bit-exact.
    #[test]
    fn gpu_backend_fused_bias_epilogue_matches_interp() {
        let mut guard = mercury_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                eprintln!("skip gpu_backend_fused_bias_epilogue_matches_interp: no CUDA device");
                return;
            }
        };
        let mut rng = Rng::new(0x0B1A5E);
        for actname in ["none", "relu", "silu", "gelu"] {
            for &(m, k, n) in &[(64usize, 16usize, 64usize), (64, 32, 128)] {
                let (program, mut interner) = build(&linear_bias_act_src(actname, m, k, n));
                let entry = interner.intern("lin");
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let bias = rng.vec(n, -0.5, 0.5);

                // CPU oracle (no accelerator) — the f32 fused bias epilogue. Buffers: [a, b, bias, c].
                let (mut ac, mut bc, mut biasc, mut cc) =
                    (a.clone(), b.clone(), bias.clone(), vec![0f32; m * n]);
                {
                    let mut bufs: [&mut [f32]; 4] = [&mut ac, &mut bc, &mut biasc, &mut cc];
                    mercury_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
                }

                // GPU offload over the identical MIR + inputs.
                let (mut ag, mut bg, mut biasg, mut cg) =
                    (a.clone(), b.clone(), bias.clone(), vec![0f32; m * n]);
                let mut accel = gpu_accel::GpuAccel::new(&mut *g);
                {
                    let mut bufs: [&mut [f32]; 4] = [&mut ag, &mut bg, &mut biasg, &mut cg];
                    mercury_interp::run_kernel_f32_accel(
                        &program, entry, &mut bufs, &interner, &mut accel,
                    )
                    .unwrap();
                }
                assert!(
                    accel.calls >= 1,
                    "{actname} {m}x{k}x{n}: fused bias epilogue never offloaded — the act(matmul+bias) \
                     nest did not fold to sgemm_nt_epi or the GPU declined, so this would silently test \
                     CPU-vs-CPU"
                );

                let s =
                    assert_close(&format!("fused bias {actname} {m}x{k}x{n}"), &cg, &cc, 5e-2, 2e-2);
                eprintln!(
                    "gpu --backend fused bias {actname} {m}x{k}x{n}: {} GPU call(s), max_abs={:.2e} max_rel={:.2e}",
                    accel.calls, s.max_abs, s.max_rel
                );
            }
        }
    }

    /// Run `entry` over `init` (full buffer list, inputs + zeroed outputs) twice — once on the
    /// interpreter oracle, once on the GPU offload — and return both buffer sets plus how many
    /// kernels actually ran on the device. The two runs see identical inputs and the same MIR.
    fn run_both(
        g: &mut mercury_codegen_gpu::Gpu,
        program: &mercury_mir::Program,
        entry: mercury_span::Symbol,
        interner: &Interner,
        init: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, u32) {
        let mut cpu: Vec<Vec<f32>> = init.to_vec();
        {
            let mut refs: Vec<&mut [f32]> = cpu.iter_mut().map(|v| v.as_mut_slice()).collect();
            mercury_interp::run_kernel_f32(program, entry, &mut refs, interner).unwrap();
        }
        let mut gpu: Vec<Vec<f32>> = init.to_vec();
        let mut accel = gpu_accel::GpuAccel::new(g);
        {
            let mut refs: Vec<&mut [f32]> = gpu.iter_mut().map(|v| v.as_mut_slice()).collect();
            mercury_interp::run_kernel_f32_accel(program, entry, &mut refs, interner, &mut accel)
                .unwrap();
        }
        (cpu, gpu, accel.calls)
    }

    /// The other three GPU kernel families behind `--backend=gpu`: activation (vmath), reduction
    /// (sreduce), and a fused row norm (softmax). Each must fire on the device and match the interp
    /// oracle within tolerance (transcendental SFU approximations are looser than the GEMM bound).
    #[test]
    fn gpu_backend_activation_reduction_norm_match_interp() {
        let mut guard = mercury_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                eprintln!(
                    "skip gpu_backend_activation_reduction_norm_match_interp: no CUDA device"
                );
                return;
            }
        };
        let mut rng = Rng::new(0x5EED_1234);

        // 1) Activation: out[i] = silu(x[i]) -> mercury_vmath_f32 (SFU sigmoid approx on GPU).
        {
            let n = 64usize;
            let src = format!(
                "module m\nfn act(x:[f32;{n}], out:[f32;{n}]) {{ \
                 for i in 0..{n} {{ out[i] = silu(x[i]); }} }}"
            );
            let (program, mut interner) = build(&src);
            let entry = interner.intern("act");
            let init = vec![rng.vec(n, -4.0, 4.0), vec![0.0; n]];
            let (cpu, gpu, calls) = run_both(g, &program, entry, &interner, &init);
            assert!(calls >= 1, "activation offload never fired");
            let s = assert_close("silu activation", &gpu[1], &cpu[1], 2e-3, 5e-3);
            eprintln!(
                "gpu --backend silu[{n}]: {calls} call(s), max_abs={:.2e} max_rel={:.2e}",
                s.max_abs, s.max_rel
            );
        }

        // 2) Reduction: o[0] = Σ x·y -> mercury_sreduce_f32_parallel (dot, written to a [1] buffer).
        {
            let n = 1024usize;
            let src = format!(
                "@parallel fn dotp(x:[f32;{n}], y:[f32;{n}], o:[f32;1]) {{ \
                 let mut s: f32 = 0.0; for k in 0..{n} {{ s = s + x[k] * y[k]; }} o[0] = s; }}"
            );
            let (program, mut interner) = build(&src);
            let entry = interner.intern("dotp");
            let init = vec![rng.vec(n, -1.0, 1.0), rng.vec(n, -1.0, 1.0), vec![0.0; 1]];
            let (cpu, gpu, calls) = run_both(g, &program, entry, &interner, &init);
            assert!(calls >= 1, "reduction offload never fired");
            let rel = (16.0 * (n as f64).sqrt() * f32::EPSILON as f64).max(1e-5);
            let s = assert_close("dot reduction", &gpu[2], &cpu[2], 1e-3, rel);
            eprintln!(
                "gpu --backend dot[{n}]: {calls} call(s), max_abs={:.2e} max_rel={:.2e}",
                s.max_abs, s.max_rel
            );
        }

        // 3) Fused norm: batched softmax over [R,C] -> mercury_norm_f32 (in place; GPU uses ex2.approx).
        {
            let (r, c) = (4usize, 16usize);
            let n = r * c;
            let src = format!(
                "module m\nfn sm(x:[f32;{n}]) {{ for row in 0..{r} {{ \
                 let mut m: f32 = x[row*{c}]; for i in 0..{c} {{ m = fmax(m, x[row*{c}+i]); }} \
                 for i in 0..{c} {{ x[row*{c}+i] = exp(x[row*{c}+i] - m); }} \
                 let mut s: f32 = 0.0; for i in 0..{c} {{ s = s + x[row*{c}+i]; }} \
                 let inv: f32 = 1.0 / s; for i in 0..{c} {{ x[row*{c}+i] = x[row*{c}+i] * inv; }} }} }}"
            );
            let (program, mut interner) = build(&src);
            let entry = interner.intern("sm");
            let init = vec![rng.vec(n, -3.0, 3.0)];
            let (cpu, gpu, calls) = run_both(g, &program, entry, &interner, &init);
            assert!(
                calls >= 1,
                "norm offload never fired (softmax did not lower to mercury_norm_f32)"
            );
            let s = assert_close("batched softmax", &gpu[0], &cpu[0], 2e-3, 5e-3);
            eprintln!(
                "gpu --backend softmax[{r}x{c}]: {calls} call(s), max_abs={:.2e} max_rel={:.2e}",
                s.max_abs, s.max_rel
            );
        }
    }
}
