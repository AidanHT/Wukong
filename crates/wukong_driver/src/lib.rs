//! `wukong_driver` — orchestrates the compilation pipeline.
//!
//! The driver owns the [`SourceMap`], the [`Interner`] and the [`DiagnosticSink`], runs each stage in
//! order, and honors the requested `--emit` target. [`compile`] wires the whole pipeline: lex
//! (`--emit=tokens`) → parse (`--emit=ast`) → import loading (the `loader` module) → sema → MIR
//! construction (`--emit=mir-high`) → optimization → the autodiff surface (`--train`,
//! `--emit=grad`) → `--run` on the selected backend → `--emit=mir` → the verify gate → the
//! LLVM-IR / object / exe emitters.
//!
//! Flag *combinations* are not validated here: that is `wukongc`'s `parse_args`, the layer that knows
//! whether a flag was given explicitly. This crate owns the option *types* and the stage dispatch.

use std::path::{Path, PathBuf};

use wukong_diag::{Diagnostic, DiagnosticSink, Renderer};
use wukong_span::{Interner, SourceMap};

pub use wukong_diag::{all_explanations, explain, Explanation};

/// The GPU offloading bridge (`--backend=gpu`), compiled only with the `gpu` feature.
#[cfg(feature = "gpu")]
mod gpu_accel;

/// The multi-file front end: resolves `import`s to files, loads each once (depth-first,
/// cycle-safe), and splices their items into the root module — one merged flat namespace.
mod loader;

/// Which intermediate (or final) artifact the user asked to produce.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EmitStage {
    Tokens,
    Ast,
    MirHigh,
    Mir,
    /// The **backward** MIR of a designated loss function (reverse-mode autodiff): the forward
    /// function followed by its `{name}_grad` twin (see `--grad-of` / `--grad-wrt`). Additive to
    /// the other stages — it runs after optimization (forcing at least `-O1`, which the autodiff
    /// transform requires for its single-block SSA input) and emits both functions' MIR.
    Grad,
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
            "grad" => EmitStage::Grad,
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
    /// Autodiff / training options (see [`GradOptions`]). Only consulted by `--emit=grad` and
    /// `--train`; inert for every other stage.
    pub grad: GradOptions,
}

/// Options for the autodiff CLI surface: which loss function to differentiate, which of its buffer
/// parameters to differentiate with respect to, and (for `--train`) the optimizer loop settings.
#[derive(Clone, Debug)]
pub struct GradOptions {
    /// The function to differentiate (`--grad-of=<name>`). Defaults to `loss` when unset.
    pub of: Option<String>,
    /// Parameter indices to differentiate w.r.t. (`--grad-wrt=0,1`). Empty means *every* pointer
    /// (buffer) parameter — the gradient of an unread/output buffer is simply zero, so this is safe —
    /// except under `--train`, where the default excludes the **last** parameter (the scalar loss
    /// output) because `train_loop` rejects it as a trainable weight (see `grad_wrt`).
    pub wrt: Vec<usize>,
    /// Run a fwd→bwd→optimizer training loop instead of just emitting the backward (`--train`).
    pub train: bool,
    /// Number of training steps (`--train-steps=N`).
    pub train_steps: usize,
    /// Learning rate for the training loop (`--train-lr=<f>`).
    pub train_lr: f32,
    /// Which optimizer the training loop uses (`--train-opt=sgd|adamw`).
    pub train_opt: TrainOpt,
    /// Seed for the deterministic buffer initialization the training loop uses (`--train-seed=<u64>`).
    pub train_seed: u64,
}

impl Default for GradOptions {
    fn default() -> GradOptions {
        GradOptions {
            of: None,
            wrt: Vec::new(),
            train: false,
            train_steps: 100,
            train_lr: 0.01,
            train_opt: TrainOpt::Sgd,
            train_seed: 0x5EED_1234,
        }
    }
}

/// The optimizer the `--train` loop applies to the differentiated parameters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrainOpt {
    /// Plain SGD (`w -= lr·g`) — needs no kernel of its own.
    Sgd,
    /// Fused decoupled-weight-decay AdamW (the `wukong_autodiff::optim` kernel).
    AdamW,
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
            grad: GradOptions::default(),
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
    let (tokens, lex_diags) = wukong_lexer::tokenize(sm.source(id), id);
    for d in lex_diags {
        sink.emit(d);
    }

    if opts.emit == EmitStage::Tokens {
        render_all(opts.error_format, &renderer, &sink, &sm);
        print_artifact(&wukong_lexer::dump(&tokens, sm.source(id)));
        return if sink.has_errors() {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    // --- Parsing ---
    let mut interner = Interner::new();
    let (mut module, parse_diags, mut next_node) =
        wukong_parser::parse_module_tokens_from(&tokens, sm.source(id), &mut interner, 0);
    for d in parse_diags {
        sink.emit(d);
    }

    // `--emit=ast` (like `--emit=tokens` above) deliberately operates on the ROOT file only —
    // these stages inspect the lex/parse of the exact file you point the compiler at, so imports
    // are not loaded or spliced here. The merged multi-file program is what every later stage
    // (`--emit=mir-high` onward, `--run`) sees.
    if opts.emit == EmitStage::Ast {
        render_all(opts.error_format, &renderer, &sink, &sm);
        print_artifact(&wukong_ast::print::print_module(&module, &interner));
        return if sink.has_errors() {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    if sink.has_errors() {
        render_all(opts.error_format, &renderer, &sink, &sm);
        return exit::COMPILE_ERROR;
    }

    // --- Import loading (the multi-file front end; see `loader`) ---
    // Each `import`ed file is resolved relative to the root source file's directory, loaded once
    // (cycles/diamonds dedup by canonical path), and its items spliced into `module`: the program
    // is checked and lowered as ONE merged flat namespace. Lex/parse diagnostics from imported
    // files — and E0305 for an unresolvable import — land in the same sink and render with their
    // own file's path/line through the shared SourceMap.
    loader::load_imports(
        &mut module,
        &opts.input,
        &mut sm,
        &mut interner,
        &mut sink,
        &mut next_node,
    );
    render_all(opts.error_format, &renderer, &sink, &sm);
    if sink.has_errors() {
        return exit::COMPILE_ERROR;
    }

    // --- Semantic analysis (name resolution, type checking, shape checking) ---
    let (sema, sema_diags) = wukong_sema::check(&module, &interner);
    for d in &sema_diags {
        emit_diag(d, opts.error_format, &renderer, &sm);
    }
    if sema_diags.iter().any(|d| d.is_error()) {
        return exit::COMPILE_ERROR;
    }

    // For `--train`, recover the loss function's buffer element counts from the semantic types now,
    // while the `[f32; N]` / `Tensor[..]` annotations are still available (they are lost once the
    // params lower to `Ptr`).
    let train_lens = if opts.grad.train {
        match loss_param_lens(&sema, &mut interner, opts) {
            Ok(l) => Some(l),
            Err(e) => {
                eprintln!("error: --train: {e}");
                return exit::COMPILE_ERROR;
            }
        }
    } else {
        None
    };

    // --- MIR construction ---
    let (mut program, lower_diags) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    for d in &lower_diags {
        emit_diag(d, opts.error_format, &renderer, &sm);
    }

    // High MIR is the pre-optimization form. It is still dumped for debugging even when lowering
    // failed, since seeing the partial MIR is useful.
    if opts.emit == EmitStage::MirHigh {
        // Dump the (possibly partial) MIR, then fail the process if lowering errored or the verifier
        // reported an internal-compiler-error — a printed ICE must never report success.
        let verify_failed = emit_mir(&program, &interner);
        return if verify_failed || lower_diags.iter().any(|d| d.is_error()) {
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
    // The autodiff transform (`--emit=grad` / `--train`) consumes single-block SSA, so it forces at
    // least `-O1` (mem2reg + simplify-cfg) regardless of the requested level. Every other path honors
    // the requested level exactly.
    let opt_level = if opts.emit == EmitStage::Grad || opts.grad.train {
        opts.opt_level.max(1)
    } else {
        opts.opt_level
    };
    wukong_opt::optimize(&mut program, opt_level);

    // --- Autodiff CLI surface (additive; runs after optimization, before any backend) ---
    if opts.grad.train {
        let lens = train_lens.expect("train_lens computed when --train is set");
        return run_train(&program, &mut interner, opts, &lens);
    }
    if opts.emit == EmitStage::Grad {
        return emit_grad(&program, &mut interner, opts);
    }

    // --- Run via the selected backend (interpreter by default, Cranelift JIT with --backend=native) ---
    if opts.run {
        // Gate the backend on a clean MIR verify. The optimizer verifies the forms its passes
        // produce, but at -O0 it runs no passes, so an invalid-MIR lowering bug would otherwise reach
        // the backend unchecked — and the native -O0 JIT codegens it into a SIGSEGV rather than the
        // clean ICE that `--emit=mir` and the optimizer already report. Verifying here makes a
        // lowering bug a diagnosable internal-compiler-error instead of a crash or miscompile.
        if verify_or_ice(&program) {
            return exit::COMPILE_ERROR;
        }
        let main = interner.intern("main");
        let result = match opts.backend {
            BackendKind::Interp => wukong_interp::run_with_output(&program, main, &interner),
            BackendKind::Native => wukong_codegen_cranelift::jit_run(&program, main, &interner),
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
        // A verifier failure means we just printed an internal-compiler-error; never report success.
        return if emit_mir(&program, &interner) {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    // The AOT exits get the same gate as `--run` above. They used to have none: the verify loop was
    // lexically inside the `if opts.run` block, and `wukong_opt`'s per-pass `verify-each` is
    // `#[cfg(debug_assertions)]`, so a release `wukongc` handed unverified MIR to `emit_llvm_ir` and
    // to Cranelift — the one path where an invalid lowering became a raw backend assertion or a
    // silently miscompiled artifact instead of a diagnosable ICE. (`--emit=mir`/`mir-high` are
    // deliberately *not* gated here: they verify inside `emit_mir` after dumping, so a broken program
    // still prints the MIR that shows why.)
    if matches!(
        opts.emit,
        EmitStage::LlvmIr | EmitStage::Obj | EmitStage::Exe
    ) && verify_or_ice(&program)
    {
        return exit::COMPILE_ERROR;
    }

    // --- LLVM backend ---
    if opts.emit == EmitStage::LlvmIr {
        print_artifact(&wukong_codegen_llvm::emit_llvm_ir(&program, &interner));
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
    program: &wukong_mir::Program,
    entry: wukong_span::Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    let mut guard = wukong_codegen_gpu::gpu();
    match guard.as_mut() {
        Some(g) => {
            let mut accel = gpu_accel::GpuAccel::new(g);
            wukong_interp::run_with_output_accel(program, entry, interner, &mut accel)
        }
        // The driver library's file name is per-OS (`nvcuda.dll` on Windows, `libcuda.so.1` on
        // Linux); it comes from `wukong_codegen_gpu`'s one table so this message and the
        // `--backend=gpu-native` twin in `lower.rs` cannot drift apart.
        None => Err(format!(
            "`--backend=gpu` requires a CUDA device, but none was reachable (the driver \
             dlopens `{}`; check the NVIDIA driver is installed)",
            wukong_codegen_gpu::baselines::cuda_driver_lib_name()
        )),
    }
}

/// Without the `gpu` feature, `--backend=gpu` is a clear build-time-capability error rather than a
/// silent fallback (which would dishonestly run on the CPU when the user asked for the GPU).
#[cfg(not(feature = "gpu"))]
fn run_on_gpu(
    _program: &wukong_mir::Program,
    _entry: wukong_span::Symbol,
    _interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    Err("this `wukongc` was built without GPU support; rebuild with `--features gpu` (or, from the \
         workspace, `-p wukongc --features gpu`) to use `--backend=gpu`"
        .into())
}

/// Execute `entry` on the **general MIR→PTX** GPU backend (`--backend=gpu-native`, Phase 4): the
/// whole program is lowered to PTX and run on the device — distinct from the recognizer offload
/// ([`run_on_gpu`]), which it leaves untouched. Built only with `--features gpu`.
#[cfg(feature = "gpu")]
fn run_on_gpu_lower(
    program: &wukong_mir::Program,
    entry: wukong_span::Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    wukong_codegen_gpu::lower::jit_run(program, entry, interner)
}

/// Without the `gpu` feature, `--backend=gpu-native` is a clear build-capability error rather than a
/// silent CPU fallback (mirrors [`run_on_gpu`]).
#[cfg(not(feature = "gpu"))]
fn run_on_gpu_lower(
    _program: &wukong_mir::Program,
    _entry: wukong_span::Symbol,
    _interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    Err("this `wukongc` was built without GPU support; rebuild with `--features gpu` (or, from the \
         workspace, `-p wukongc --features gpu`) to use `--backend=gpu-native`"
        .into())
}

/// The C runtime linked into native executables by the **`cc` fallback** path: it backs the
/// `print`/`assert`/`fmod` and heap intrinsics that the Cranelift object leaves as undefined imports.
/// The JIT path binds those same symbols to Rust functions instead. It is deliberately *not* a full
/// mirror — there is no `wukong_rt_print_str` here and `printf("%g")` is under-precise for floats —
/// which is why this path is for scalar, no-string programs only; [`WUKONG_RT_SHIM`] on the preferred
/// [`rustc_link`] path is the one whose output is byte-identical to the interpreter oracle.
const WUKONG_RT_C: &str = "#include <stdio.h>\n\
#include <stdlib.h>\n\
#include <math.h>\n\
#ifdef _WIN32\n\
#include <io.h>\n\
#include <fcntl.h>\n\
/* Keep stdout in binary mode so the MSVCRT does not translate the runtime's `\\n` into CRLF. The\n\
   interpreter oracle (and the JIT) emit LF; without this the *linked exe* printed CRLF on Windows,\n\
   so its stdout differed from every other backend. Runs before main via the constructor attribute. */\n\
__attribute__((constructor)) static void wukong_rt_init(void) { _setmode(_fileno(stdout), _O_BINARY); }\n\
#endif\n\
void wukong_rt_print_i64(long long x) { printf(\"%lld\\n\", x); }\n\
void wukong_rt_print_u64(unsigned long long x) { printf(\"%llu\\n\", x); }\n\
void wukong_rt_print_f64(double x) { printf(\"%g\\n\", x); }\n\
void wukong_rt_assert(long long c) { if (!c) { fprintf(stderr, \"assertion failed\\n\"); exit(101); } }\n\
double wukong_rt_fmod_f64(double a, double b) { return fmod(a, b); }\n\
float wukong_rt_fmod_f32(float a, float b) { return fmodf(a, b); }\n\
/* Heap builtins (alloc_<T>/free). calloc zero-initializes — the determinism contract — and\n\
   handles the count*size overflow; a negative count clamps to an empty (null) allocation. */\n\
void* wukong_rt_alloc(long long count, long long elem_size, long long elem_is_float) {\n\
    (void)elem_is_float;\n\
    if (count < 0 || elem_size <= 0) return 0;\n\
    return calloc((size_t)count, (size_t)elem_size);\n\
}\n\
void wukong_rt_free(void* p) { free(p); }\n";

/// A per-invocation scratch directory for the generated `--emit=exe` link inputs (the Rust link
/// shim and the C fallback runtime). They used to be written into the process's *current* directory
/// as `{stem}_shim.rs` / `{stem}_rt.c`, where the shim overwrote — and then unconditionally deleted
/// — any user file of that name, and two concurrent links on the same stem raced on it. Keyed by
/// pid so concurrent invocations cannot collide; removed when the link finishes either way.
struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    /// `None` when the directory cannot be created — the caller then falls back to the old
    /// working-directory paths rather than failing the link outright.
    fn new(stem: &str) -> Option<ScratchDir> {
        let path = std::env::temp_dir().join(format!("wukongc-{}-{stem}", std::process::id()));
        std::fs::create_dir_all(&path).ok()?;
        Some(ScratchDir { path })
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Where a generated link input goes: inside the scratch directory when one could be created, else
/// the working directory (the historical behaviour).
fn scratch_path(scratch: Option<&ScratchDir>, name: String) -> PathBuf {
    match scratch {
        Some(s) => s.path.join(name),
        None => PathBuf::from(name),
    }
}

/// Emit a native object via Cranelift (no LLVM) and, for `--emit=exe`, link it into an executable.
///
/// The link is attempted in one order: [`rustc_link`] first (rustc as the *link driver*, which pulls
/// in the `wukong_runtime` rlib plus the generated [`WUKONG_RT_SHIM`]), and only on
/// [`LinkOutcome::Unavailable`] (no rustc on PATH, or no rlib next to `current_exe()`) the `$CC` /
/// `cc` fallback with the generated C runtime shim [`WUKONG_RT_C`]. A [`LinkOutcome::Failed`] — rustc
/// ran and the link itself failed — is a real error and exits [`exit::COMPILE_ERROR`] without falling
/// back. If the `cc` fallback cannot even be spawned, the object file has already been written, so the
/// driver says where it is and exits [`exit::UNIMPLEMENTED`] (2).
fn emit_native(program: &wukong_mir::Program, interner: &Interner, opts: &Options) -> i32 {
    use std::process::Command;

    let obj = match wukong_codegen_cranelift::emit_object(program, interner) {
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

    // The default executable name is `<stem>` plus the host's executable suffix: `.exe` on Windows,
    // **nothing** on unix, where `foo.exe` would be a lie about the file's format and would not be
    // what any adjacent tooling (or the user) looks for. `std::env::consts::EXE_SUFFIX` is the
    // platform's own answer, so the Windows spelling is unchanged, byte for byte.
    let out = opts
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("{stem}{}", std::env::consts::EXE_SUFFIX)));

    // The generated link inputs are compiler intermediates, not user artifacts — keep them out of
    // the directory the user invoked us in (see [`ScratchDir`]).
    let scratch = ScratchDir::new(&stem);

    // Prefer the **rustc-driven link**: `rustc` invokes the object's *native* platform linker and
    // links `wukong_runtime` (the AVX2 kernels) as a real dependency, so a recognized-kernel program
    // resolves its `wukong_*` symbols AND the read-only string `.rodata` relocations link. The MinGW
    // `cc` path below can neither (it crashes on the MSVC object's data relocations and can't consume
    // the Rust runtime staticlib). Output is bit-identical to `--run`. Falls back to `cc` only when
    // rustc or the runtime rlib is unavailable (a scalar, no-data, no-kernel program still links).
    match rustc_link(scratch.as_ref(), &stem, &obj_path, &out) {
        LinkOutcome::Linked => {
            eprintln!("wrote {}", out.display());
            return exit::OK;
        }
        LinkOutcome::Failed(msg) => {
            eprintln!("error: {msg}");
            return exit::COMPILE_ERROR;
        }
        LinkOutcome::Unavailable => { /* fall through to the C-runtime `cc` link */ }
    }

    // exe (fallback): emit the C runtime into the scratch dir and link it with the system C compiler.
    let rt_path = scratch_path(scratch.as_ref(), format!("{stem}_rt.c"));
    if let Err(e) = std::fs::write(&rt_path, WUKONG_RT_C) {
        eprintln!("error: could not write `{}`: {e}", rt_path.display());
        return exit::IO_ERROR;
    }
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let status = Command::new(&cc)
        .args(cc_link_args(&obj_path, &rt_path, &out))
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

/// The `cc` fallback's command line, factored out of [`emit_native`] so it can be pinned by a test.
///
/// **`-lm` is load-bearing and must stay last.** [`WUKONG_RT_C`] *unconditionally* defines
/// `wukong_rt_fmod_f64`/`_f32` over C's `fmod`/`fmodf`, and on glibc those live in `libm`, which
/// `cc` does **not** link by default — so without this flag the fallback could not link *any*
/// program on Linux (`undefined reference to 'fmod'`), not merely one that uses `%` on floats. It is
/// a no-op where libm is already folded into the C library (macOS/libSystem, MinGW's stub
/// `libm.a`), so the Windows and macOS link lines are unaffected. Position matters: GNU `ld`
/// resolves left to right, so a library must follow the objects that reference it.
fn cc_link_args(obj_path: &Path, rt_path: &Path, out: &Path) -> Vec<std::ffi::OsString> {
    vec![
        obj_path.into(),
        rt_path.into(),
        "-o".into(),
        out.into(),
        "-O2".into(),
        "-lm".into(),
    ]
}

/// The outcome of the [`rustc_link`] attempt (the preferred `--emit=exe` link path).
enum LinkOutcome {
    /// The executable was produced.
    Linked,
    /// rustc or the `wukong_runtime` rlib is not available — the caller should try the `cc` fallback.
    Unavailable,
    /// rustc ran but the link failed (a genuine error — do not silently fall back).
    Failed(String),
}

/// The Rust link shim generated for the rustc-driven `--emit=exe` link. It provides the `wukong_rt_*`
/// runtime the Cranelift object calls (formatting through Rust's stdout, so output is byte-identical
/// to the interpreter/JIT oracle — raw LF on Windows, and floats via `Display` like the interp's
/// `format!("{f}")`, unlike the C runtime's under-precise `printf("%g")`), and `extern crate
/// wukong_runtime` pulls the AVX2 microkernels into the link so a recognized-kernel program resolves
/// its `wukong_*` symbols. `#![no_main]` leaves the Cranelift object's own `main` as the CRT entry.
const WUKONG_RT_SHIM: &str = r##"#![no_main]
//! Auto-generated by `wukongc --emit=exe`. Do not edit; it is rewritten on every link.
extern crate wukong_runtime;
use std::io::Write;
#[no_mangle]
pub extern "C" fn wukong_rt_print_i64(x: i64) {
    let o = std::io::stdout();
    let _ = write!(o.lock(), "{x}\n");
}
#[no_mangle]
pub extern "C" fn wukong_rt_print_u64(x: u64) {
    let o = std::io::stdout();
    let _ = write!(o.lock(), "{x}\n");
}
#[no_mangle]
pub extern "C" fn wukong_rt_print_f64(x: f64) {
    let o = std::io::stdout();
    let _ = write!(o.lock(), "{x}\n");
}
#[no_mangle]
pub extern "C" fn wukong_rt_print_str(p: *const u8) {
    if p.is_null() {
        return;
    }
    // SAFETY: a Wukong `*u8` string buffer, always NUL-terminated by the string lowering.
    unsafe {
        let mut n = 0isize;
        while *p.offset(n) != 0 {
            n += 1;
        }
        let s = std::slice::from_raw_parts(p, n as usize);
        let o = std::io::stdout();
        let mut o = o.lock();
        let _ = o.write_all(s);
        let _ = o.write_all(b"\n");
    }
}
#[no_mangle]
pub extern "C" fn wukong_rt_assert(c: i64) {
    if c == 0 {
        eprintln!("assertion failed");
        std::process::exit(101);
    }
}
#[no_mangle]
pub extern "C" fn wukong_rt_fmod_f64(a: f64, b: f64) -> f64 {
    a % b
}
#[no_mangle]
pub extern "C" fn wukong_rt_fmod_f32(a: f32, b: f32) -> f32 {
    a % b
}
"##;

/// Locate the `wukong_runtime` rlib that [`rustc_link`] will link, given the directory holding the
/// compiler binary (`current_exe().parent()`, i.e. cargo's `target/<profile>/`). `None` means the
/// preferred `--emit=exe` link path is unavailable here and the `cc` fallback is all there is.
///
/// **Two locations, and the second is the one that is always there.** `target/<profile>/` holds
/// `libwukong_runtime.rlib` only when cargo *uplifted* it out of `deps/`, and cargo only uplifts a
/// library when that library is a **root unit of a `build`** — i.e. after a plain `cargo build`.
/// Under `cargo test --workspace` the package's root unit is its *test* binary and the plain rlib is
/// merely a dependency unit, so it stays in `target/<profile>/deps/` under its hash-suffixed name
/// and nothing is uplifted; `cargo run -p wukongc` does not uplift it either, because `-p` makes
/// `wukong_runtime` a dependency rather than a selected member. A tree that has the uplifted copy
/// has it as a *leftover* of some earlier `cargo build`, which is why this looked fine locally and
/// broke on every clean checkout: CI runs `cargo test --workspace`, never `cargo build`, so the
/// preferred link path silently degraded to the `cc` fallback — whose output is knowingly *not*
/// byte-identical to the oracle (no `wukong_rt_print_str`, no `wukong_*` kernels, under-precise
/// `printf("%g")` floats). Searching `deps/` too makes the preferred path available whenever cargo
/// built the runtime at all, however the compiler was built.
///
/// A `deps/` directory can hold several hash-suffixed rlibs (a metadata hash changes when the
/// profile or a dependency does, and cargo does not garbage-collect the old ones), so the newest by
/// mtime wins — the one cargo just linked everything else against. Ties break on the path so the
/// choice is deterministic.
pub fn runtime_rlib_in(dir: &Path) -> Option<PathBuf> {
    // The uplifted name: what `cargo build` (and an installed target layout) leaves next to the
    // binary. Preferred, because it is unambiguous.
    let uplifted = dir.join("libwukong_runtime.rlib");
    if uplifted.is_file() {
        return Some(uplifted);
    }
    // `deps/libwukong_runtime-<metadata hash>.rlib` — where cargo *always* writes it. The trailing
    // `-` in the prefix is load-bearing: it keeps a hypothetical `libwukong_runtime_foo` crate out.
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(dir.join("deps")).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rlib") {
            continue;
        }
        let matches = path
            .file_stem()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.starts_with("libwukong_runtime-"));
        if !matches {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let better = match &best {
            None => true,
            Some((t, p)) => (mtime, &path) > (*t, p),
        };
        if better {
            best = Some((mtime, path));
        }
    }
    best.map(|(_, p)| p)
}

/// Link the Cranelift object into an executable with **`rustc` as the link driver**. rustc invokes
/// the object's native platform linker (on the MSVC host, the installed `link.exe`, discovered
/// automatically) and links `wukong_runtime` as a real dependency, so a recognized-kernel program's
/// `wukong_*` symbols resolve and the string `.rodata` relocations link — neither of which the MinGW
/// `cc` path can do here. The runtime rlib is located by [`runtime_rlib_in`] from the directory
/// holding this compiler binary, and `deps/` next to it goes on `-L dependency=` for the runtime's
/// own dependencies. Returns [`LinkOutcome::Unavailable`] when rustc or the rlib is absent so the
/// caller can try the `cc` fallback.
fn rustc_link(
    scratch: Option<&ScratchDir>,
    stem: &str,
    obj_path: &Path,
    out: &Path,
) -> LinkOutcome {
    use std::process::Command;

    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(_) => return LinkOutcome::Unavailable,
    };
    let dir = match exe.parent() {
        Some(d) => d.to_path_buf(),
        None => return LinkOutcome::Unavailable,
    };
    let deps = dir.join("deps");
    let rlib = match runtime_rlib_in(&dir) {
        Some(p) => p,
        // Not a cargo target layout (e.g. an installed binary without the rlib) — use the cc path.
        None => return LinkOutcome::Unavailable,
    };

    let shim_path = scratch_path(scratch, format!("{stem}_shim.rs"));
    if let Err(e) = std::fs::write(&shim_path, WUKONG_RT_SHIM) {
        return LinkOutcome::Failed(format!("could not write `{}`: {e}", shim_path.display()));
    }

    let status = Command::new("rustc")
        .arg(&shim_path)
        .arg("--edition")
        .arg("2021")
        // The rlib we link against is built by this workspace, whose `[profile.release]` sets
        // `panic = "abort"`. rustc defaults the shim to `unwind`, and a strategy mismatch is a hard
        // metadata error ("the crate `wukong_runtime` requires panic strategy `abort` which is
        // incompatible with this crate's strategy of `unwind`") that rejected every release-profile
        // link before it ever reached the linker. An abort-strategy crate may link unwind-built
        // dependencies, so this is equally correct against a debug-profile rlib.
        .arg("-C")
        .arg("panic=abort")
        .arg("--extern")
        .arg(format!("wukong_runtime={}", rlib.display()))
        .arg("-L")
        .arg(format!("dependency={}", deps.display()))
        .arg("-C")
        .arg(format!("link-arg={}", obj_path.display()))
        .arg("-o")
        .arg(out)
        .status();
    let _ = std::fs::remove_file(&shim_path);

    match status {
        Ok(s) if s.success() => LinkOutcome::Linked,
        Ok(_) => LinkOutcome::Failed(format!(
            "`rustc` failed to link the native object `{}`",
            obj_path.display()
        )),
        // rustc not on PATH: fall back to the C-runtime `cc` link (scalar, no-data programs only).
        Err(_) => LinkOutcome::Unavailable,
    }
}

/// `--emit=grad`: differentiate the designated loss function and print the forward + backward MIR.
///
/// The loss function must already be single-block SSA (the caller forces `-O1`); its buffer
/// parameters lower to `Ptr` and the scalar loss is `ret`-ed. Any recognized kernel calls in it
/// (`wukong_sgemm_nt`, `wukong_vmath_f32`, `wukong_sreduce_f32`, `wukong_norm_f32`,
/// `wukong_velem_f32` — each recognized in its serial *and* its `_parallel` spelling) get their
/// **tuned-kernel** backward — so the emitted gradient rides the same
/// kernels as the forward. A differentiable value with no VJP rule is a hard error, never a
/// silently-zero gradient (see `wukong_autodiff`).
fn emit_grad(program: &wukong_mir::Program, interner: &mut Interner, opts: &Options) -> i32 {
    let (fwd, gradfn) = match build_grad(program, interner, opts) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("error: --emit=grad: {e}");
            return exit::COMPILE_ERROR;
        }
    };
    let out = wukong_mir::Program {
        funcs: vec![fwd, gradfn],
        level: program.level,
        statics: program.statics.clone(),
    };
    // A verifier failure means we just printed an internal-compiler-error; never report success —
    // the same mapping the `--emit=mir` / `--emit=mir-high` callers already apply.
    if emit_mir(&out, interner) {
        exit::COMPILE_ERROR
    } else {
        exit::OK
    }
}

/// Shared core of `--emit=grad` and `--train`: resolve the target loss function and its `wrt` list
/// from `opts`, run the autodiff transform, and return `(forward_clone, gradient_function)`.
fn build_grad(
    program: &wukong_mir::Program,
    interner: &mut Interner,
    opts: &Options,
) -> Result<(wukong_mir::Function, wukong_mir::Function), String> {
    let name = opts.grad.of.clone().unwrap_or_else(|| "loss".to_string());
    let sym = interner.intern(&name);
    let func = program.function(sym).ok_or_else(|| {
        format!("no function `{name}` in the module (choose the target with --grad-of=<fn>)")
    })?;
    let wrt = grad_wrt(func, &opts.grad.wrt, opts.grad.train)?;
    let gradfn = wukong_autodiff::grad(func, &wrt, interner)?;
    Ok((func.clone(), gradfn))
}

/// The parameter indices to differentiate w.r.t.: the explicit `--grad-wrt` list (validated to be
/// distinct, in-range pointer parameters), or — when empty — *every* pointer (buffer) parameter of
/// the loss. Differentiating an unread or output buffer is harmless (its gradient is simply zero),
/// so the "all buffers" default never produces a wrong answer, only occasionally an unused zero
/// buffer.
///
/// `training` selects the `--train` flavour of the default: the trainable weights, i.e. every buffer
/// parameter *except* the last one, which by the `--train` convention is the scalar loss output.
/// Without that exclusion the documented default expanded to a list containing the loss-output index
/// and `train_loop` then rejected it, so `--train` could never run without an explicit `--grad-wrt`.
fn grad_wrt(
    func: &wukong_mir::Function,
    requested: &[usize],
    training: bool,
) -> Result<Vec<usize>, String> {
    let is_ptr = |i: usize| *func.value_type(func.params[i]) == wukong_mir::MirType::Ptr;
    if requested.is_empty() {
        let loss_out = func.params.len().wrapping_sub(1);
        let all: Vec<usize> = (0..func.params.len())
            .filter(|&i| is_ptr(i) && !(training && i == loss_out))
            .collect();
        if all.is_empty() {
            return Err(if training {
                "the loss function has no trainable buffer parameters (--train differentiates every \
                 buffer parameter except the last, which is the scalar loss output)"
                    .into()
            } else {
                "the loss function has no buffer (pointer) parameters to differentiate".to_string()
            });
        }
        return Ok(all);
    }
    let mut seen = std::collections::HashSet::new();
    for &wi in requested {
        if wi >= func.params.len() {
            return Err(format!(
                "--grad-wrt index {wi} is out of range (the function has {} parameter(s))",
                func.params.len()
            ));
        }
        if !is_ptr(wi) {
            return Err(format!(
                "--grad-wrt index {wi} is not a buffer (pointer) parameter"
            ));
        }
        // A repeated index appends a second gradient buffer to the backward's parameter list, but
        // `Vjp`'s param -> gradient-buffer map keeps only the last insert, so the earlier buffer is
        // never written and its caller reads back all zeros.
        if !seen.insert(wi) {
            return Err(format!(
                "--grad-wrt index {wi} is repeated; each parameter may be differentiated at most once"
            ));
        }
    }
    Ok(requested.to_vec())
}

/// The number of `f32` leaf elements in a buffer parameter's semantic type — the count the
/// interpreter's `run_kernel_f32` ABI expects (one slot per scalar leaf). `Array` multiplies by its
/// concrete length, `Tensor` by the product of its (fully-const) dims. `None` for a param whose size
/// is not statically known (a symbolic tensor dim, a slice) — `--train` needs a concrete size.
fn ty_elem_count(ty: &wukong_types::Ty) -> Option<u64> {
    use wukong_types::{Dim, Ty};
    match ty {
        Ty::Scalar(_) => Some(1),
        Ty::Array { elem, len } => Some(ty_elem_count(elem)? * len),
        Ty::Tensor { shape, .. } => shape.0.iter().try_fold(1u64, |acc, d| match d {
            Dim::Const(n) => Some(acc * n),
            _ => None,
        }),
        _ => None,
    }
}

/// Recover the element count of each parameter of the `--train` / `--grad-of` loss function from the
/// semantic types (the `[f32; N]` / `Tensor[..]` annotations lost when the params lower to `Ptr`).
fn loss_param_lens(
    sema: &wukong_sema::SemaResult,
    interner: &mut Interner,
    opts: &Options,
) -> Result<Vec<usize>, String> {
    let name = opts.grad.of.clone().unwrap_or_else(|| "loss".to_string());
    let sym = interner.intern(&name);
    let def = sema
        .defs
        .lookup(sym)
        .ok_or_else(|| format!("no function `{name}` in the module"))?;
    let sig = match &def.kind {
        wukong_sema::DefKind::Fn(sig) => sig,
        _ => return Err(format!("`{name}` is not a function")),
    };
    sig.params
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            ty_elem_count(ty).map(|n| n as usize).ok_or_else(|| {
                format!(
                    "parameter {i} of `{name}` has no statically-known element count \
                     (a symbolic tensor dim or slice); --train needs concrete buffer sizes"
                )
            })
        })
        .collect()
}

/// A tiny deterministic LCG producing `f32` in ~`[-0.4, 0.4]` — seeds the `--train` buffers so a run
/// is reproducible (honest: no hidden entropy) and a test can assert a specific loss trajectory.
fn train_rand(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((*seed >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.8
        })
        .collect()
}

/// `--train`: run a fwd→bwd→optimizer loop on the loss function and print the loss trajectory.
///
/// Convention: `--grad-wrt` names the **trainable** buffer parameters (the weights); every other
/// buffer is fixed input data; the **last** parameter is the scalar loss output `[f32; 1]`. All
/// buffers are seeded deterministically (`--train-seed`); each step runs the gradient kernel (which
/// replays the forward, so it yields both the loss and the gradients), then applies SGD or the fused
/// AdamW kernel to the weights. The printed trajectory is the honest end-to-end signal — on a
/// convex objective it strictly decreases.
fn run_train(
    program: &wukong_mir::Program,
    interner: &mut Interner,
    opts: &Options,
    lens: &[usize],
) -> i32 {
    match train_loop(program, interner, opts, lens) {
        Ok(traj) => {
            use std::io::Write;
            let mut out = String::new();
            let opt = match opts.grad.train_opt {
                TrainOpt::Sgd => "sgd",
                TrainOpt::AdamW => "adamw",
            };
            out.push_str(&format!(
                "train: {} step(s), lr={}, optimizer={opt}\n",
                traj.len().saturating_sub(1),
                opts.grad.train_lr
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
            let _ = std::io::stdout().write_all(out.as_bytes());
            exit::OK
        }
        Err(e) => {
            eprintln!("error: --train: {e}");
            exit::COMPILE_ERROR
        }
    }
}

/// The testable core of `--train`: build the {forward, backward, optimizer} program, initialize the
/// buffers, run the loop, and return the loss trajectory (`traj[i]` = loss after `i` updates, with a
/// final entry after the last update — so `traj.len() == steps + 1`).
fn train_loop(
    program: &wukong_mir::Program,
    interner: &mut Interner,
    opts: &Options,
    lens: &[usize],
) -> Result<Vec<f32>, String> {
    use wukong_autodiff::optim::{self, hp};
    use wukong_mir::Program;

    let name = opts.grad.of.clone().unwrap_or_else(|| "loss".to_string());
    let fsym = interner.intern(&name);
    let (nparams, wrt) = {
        let loss_fn = program
            .function(fsym)
            .ok_or_else(|| format!("no function `{name}` in the module"))?;
        (
            loss_fn.params.len(),
            grad_wrt(loss_fn, &opts.grad.wrt, true)?,
        )
    };
    if lens.len() != nparams {
        return Err(format!(
            "internal: {} param sizes for a {nparams}-param loss function",
            lens.len()
        ));
    }
    let loss_out = nparams - 1;
    if lens[loss_out] != 1 {
        return Err(
            "the last parameter must be the scalar loss output `[f32; 1]` (the `--train` convention)"
                .into(),
        );
    }
    if wrt.contains(&loss_out) {
        return Err("--grad-wrt must not include the loss-output parameter (the last one)".into());
    }

    // Build {forward, backward} plus one fused AdamW kernel per distinct trainable-tensor size.
    let (fwd, gradfn) = build_grad(program, interner, opts)?;
    let (fwd_name, gname) = (fwd.name, gradfn.name);
    let mut funcs = vec![fwd, gradfn];
    let mut adamw: std::collections::HashMap<usize, wukong_span::Symbol> =
        std::collections::HashMap::new();
    if opts.grad.train_opt == TrainOpt::AdamW {
        let mut sizes: Vec<usize> = wrt.iter().map(|&wi| lens[wi]).collect();
        sizes.sort_unstable();
        sizes.dedup();
        for n in sizes {
            let f = optim::build_adamw_step(interner, n);
            adamw.insert(n, f.name);
            funcs.push(f);
        }
    }
    let train_prog = Program {
        funcs,
        level: program.level,
        statics: program.statics.clone(),
    };

    // Deterministic buffer init; the loss output is zeroed, moment state starts at zero.
    let mut seed = opts.grad.train_seed;
    let mut bufs: Vec<Vec<f32>> = lens.iter().map(|&n| train_rand(&mut seed, n)).collect();
    bufs[loss_out] = vec![0.0; 1];
    let mut mstate: Vec<Vec<f32>> = wrt.iter().map(|&wi| vec![0.0; lens[wi]]).collect();
    let mut vstate: Vec<Vec<f32>> = wrt.iter().map(|&wi| vec![0.0; lens[wi]]).collect();

    let lr = opts.grad.train_lr;
    let (beta1, beta2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    // Grown amortized rather than pre-reserved: `--train-steps` is an unvalidated CLI integer, and
    // reserving from it let `--train-steps=999999999999` abort inside the allocator ("memory
    // allocation of 4000000000000 bytes failed", exit 127) before a single step ran — raw runtime
    // text with no diagnostic. `usize::MAX` additionally wrapped `+ 1` to a zero reservation.
    let mut traj = Vec::new();

    let run = |prog: &Program, entry, bufs: &mut [Vec<f32>], it: &Interner| -> Result<(), String> {
        let mut views: Vec<&mut [f32]> = bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
        wukong_interp::run_kernel_f32(prog, entry, &mut views, it)
    };

    for step in 0..opts.grad.train_steps {
        // The gradient kernel replays the forward, so one call gives the current loss + the gradients.
        let mut gbufs: Vec<Vec<f32>> = bufs.clone();
        for &wi in &wrt {
            gbufs.push(vec![0.0; lens[wi]]);
        }
        run(&train_prog, gname, &mut gbufs, interner)?;
        traj.push(gbufs[loss_out][0]);

        match opts.grad.train_opt {
            TrainOpt::Sgd => {
                for (i, &wi) in wrt.iter().enumerate() {
                    let g = &gbufs[nparams + i];
                    for (wv, &gv) in bufs[wi].iter_mut().zip(g) {
                        *wv -= lr * gv;
                    }
                }
            }
            TrainOpt::AdamW => {
                let t = (step + 1) as i32;
                let mut hpbuf = vec![0.0f32; hp::LEN];
                hpbuf[hp::LR] = lr;
                hpbuf[hp::BETA1] = beta1 as f32;
                hpbuf[hp::BETA2] = beta2 as f32;
                hpbuf[hp::EPS] = eps as f32;
                hpbuf[hp::WD] = 0.0;
                hpbuf[hp::BC1] = (1.0 - beta1.powi(t)) as f32;
                hpbuf[hp::BC2] = (1.0 - beta2.powi(t)) as f32;
                for (i, &wi) in wrt.iter().enumerate() {
                    let aname = adamw[&lens[wi]];
                    let mut ab = vec![
                        bufs[wi].clone(),
                        gbufs[nparams + i].clone(),
                        mstate[i].clone(),
                        vstate[i].clone(),
                        hpbuf.clone(),
                    ];
                    run(&train_prog, aname, &mut ab, interner)?;
                    bufs[wi] = ab[0].clone();
                    mstate[i] = ab[2].clone();
                    vstate[i] = ab[3].clone();
                }
            }
        }
    }
    // One final forward to report the loss after the last update.
    run(&train_prog, fwd_name, &mut bufs, interner)?;
    traj.push(bufs[loss_out][0]);
    Ok(traj)
}

/// Print the MIR for `program`, running the verifier first. Any verifier failure is emitted as an
/// `internal compiler error (MIR verify)` line to stderr. Returns `true` if the verifier reported
/// at least one failure, so callers can map an internal-compiler-error to a nonzero process exit
/// status instead of reporting success on invalid MIR.
fn emit_mir(program: &wukong_mir::Program, interner: &Interner) -> bool {
    let verify_failed = verify_or_ice(program);
    print_artifact(&wukong_mir::print::print_program(program, interner));
    verify_failed
}

/// Run the MIR verifier over every function, reporting each violation as an
/// `internal compiler error (MIR verify)` line. Returns `true` if anything failed. A failure is a
/// bug in a compiler pass, not in the user's program, so it is reported and the process stops —
/// never handed on to a backend.
fn verify_or_ice(program: &wukong_mir::Program) -> bool {
    let mut verify_failed = false;
    for f in &program.funcs {
        for ice in wukong_mir::verify::verify_function(f) {
            eprint_line(&format!("internal compiler error (MIR verify): {ice}"));
            verify_failed = true;
        }
    }
    verify_failed
}

/// Write a compiler artifact to stdout.
///
/// Deliberately a *fallible* write rather than `print!`: `print!` panics when the consumer closes
/// the pipe, and with the workspace's `panic = "abort"` release profile that panic aborts the whole
/// process. `wukongc --emit=mir big.wk | head -1` therefore printed Rust's internal
/// "failed printing to stdout: The pipe has been ended. (os error 109)" and exited 127 instead of
/// reporting the compile result. A closed consumer is not a compiler error — drop the rest of the
/// artifact and let the normal exit code stand.
fn print_artifact(s: &str) {
    use std::io::Write;
    let _ = std::io::stdout().write_all(s.as_bytes());
}

/// One line of compiler output on stderr, fallible for the same reason as [`print_artifact`]: a
/// many-diagnostic file piped into `head` aborted the process partway through the diagnostic stream.
fn eprint_line(s: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "{s}");
}

/// Emit a single diagnostic to stderr in the requested format.
fn emit_diag(d: &Diagnostic, fmt: ErrorFormat, renderer: &Renderer, sm: &SourceMap) {
    match fmt {
        ErrorFormat::Human => eprint_line(&renderer.render(d, sm)),
        ErrorFormat::Json => eprint_line(&wukong_diag::to_json(d, sm)),
    }
}

fn render_all(fmt: ErrorFormat, renderer: &Renderer, sink: &DiagnosticSink, sm: &SourceMap) {
    for d in sink.diagnostics() {
        emit_diag(d, fmt, renderer, sm);
    }
}

/// The `--emit=grad` CLI surface, end to end from `.wk` source: compile → optimize → differentiate
/// the loss function, then **finite-difference-gate** the emitted backward. This is the real proof
/// that autodiff-from-source is correct — the recognized backward kernels are gate-blind (both
/// backends call the same symbol), so the f64/closed-form finite-difference check, not interp==native,
/// is what proves the gradient. Toolchain-free: it runs entirely on the interpreter oracle.
#[cfg(test)]
mod grad_cli_tests {
    use super::*;
    use wukong_mir::{MirLevel, Program};
    use wukong_span::{SourceMap, Symbol};

    /// Lex → parse → sema → mir_build → opt(1). Panics with the offending stage's diagnostics.
    fn compile_o1(src: &str) -> (Program, Interner) {
        let mut sm = SourceMap::new();
        let id = sm.add("grad_test.wk".to_string(), src.to_string());
        let (tokens, ld) = wukong_lexer::tokenize(sm.source(id), id);
        assert!(!ld.iter().any(|d| d.is_error()), "lex errors");
        let mut interner = Interner::new();
        let (module, pd) =
            wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        assert!(!pd.iter().any(|d| d.is_error()), "parse errors");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()), "sema errors: {sd:?}");
        let (mut program, md) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(!md.iter().any(|d| d.is_error()), "mir_build errors: {md:?}");
        wukong_opt::optimize(&mut program, 1);
        (program, interner)
    }

    /// Drive the real `build_grad` path (`--grad-of`/`--grad-wrt`) and return the `{fwd, fwd_grad}`
    /// program, the forward and gradient function names, the **resolved** `wrt` list (so `[]`
    /// expands to every buffer parameter, matching the appended gradient buffers), and the interner.
    fn grad_prog(
        src: &str,
        of: &str,
        wrt: &[usize],
    ) -> (Program, Symbol, Symbol, Vec<usize>, Interner) {
        let (program, mut interner) = compile_o1(src);
        let opts = Options {
            grad: GradOptions {
                of: Some(of.to_string()),
                wrt: wrt.to_vec(),
                ..GradOptions::default()
            },
            ..Options::default()
        };
        let fsym = interner.intern(of);
        let resolved = grad_wrt(
            program.function(fsym).expect("loss fn"),
            wrt,
            opts.grad.train,
        )
        .expect("wrt");
        let (fwd, g) = build_grad(&program, &mut interner, &opts).expect("build_grad failed");
        let (fname, gname) = (fwd.name, g.name);
        let prog = Program {
            funcs: vec![fwd, g],
            level: MirLevel::Low,
            statics: program.statics.clone(),
        };
        (prog, fname, gname, resolved, interner)
    }

    fn run_f32(prog: &Program, name: Symbol, bufs: &mut [Vec<f32>], it: &Interner) {
        let mut views: Vec<&mut [f32]> = bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
        wukong_interp::run_kernel_f32(prog, name, &mut views, it).expect("kernel run failed");
    }

    /// Run the gradient kernel over `inputs` (one zeroed gradient buffer appended per `wrt`) and
    /// return the gradient buffer for each `wrt` (as f64).
    fn analytic(
        prog: &Program,
        gname: Symbol,
        inputs: &[Vec<f32>],
        lens: &[usize],
        wrt: &[usize],
        it: &Interner,
    ) -> Vec<Vec<f64>> {
        let mut bufs = inputs.to_vec();
        for &wi in wrt {
            bufs.push(vec![0.0; lens[wi]]);
        }
        run_f32(prog, gname, &mut bufs, it);
        let np = inputs.len();
        (0..wrt.len())
            .map(|i| bufs[np + i].iter().map(|&v| v as f64).collect())
            .collect()
    }

    /// Central finite-difference gradient of `out[loss_out][0]` w.r.t. each `wrt` element.
    #[allow(clippy::too_many_arguments)]
    fn finite_diff(
        prog: &Program,
        fwd_name: Symbol,
        inputs: &[Vec<f32>],
        lens: &[usize],
        wrt: &[usize],
        loss_out: usize,
        eps: f32,
        it: &Interner,
    ) -> Vec<Vec<f64>> {
        let loss = |bufs: &mut [Vec<f32>]| -> f64 {
            run_f32(prog, fwd_name, bufs, it);
            bufs[loss_out][0] as f64
        };
        let mut out = Vec::new();
        for &wi in wrt {
            let mut g = vec![0.0; lens[wi]];
            for j in 0..lens[wi] {
                let mut bufs = inputs.to_vec();
                let orig = bufs[wi][j];
                bufs[wi][j] = orig + eps;
                let lp = loss(&mut bufs);
                bufs[wi][j] = orig - eps;
                let lm = loss(&mut bufs);
                g[j] = (lp - lm) / (2.0 * eps as f64);
            }
            out.push(g);
        }
        out
    }

    /// Full gate: analytic gradient (from the emitted backward) vs the finite difference, over every
    /// element. Returns the analytic gradients so a caller can additionally assert a closed form.
    #[allow(clippy::too_many_arguments)]
    fn gate(
        src: &str,
        of: &str,
        wrt: &[usize],
        inputs: &[Vec<f32>],
        lens: &[usize],
        loss_out: usize,
        eps: f32,
        rel: f64,
        abs: f64,
    ) -> Vec<Vec<f64>> {
        let (prog, fwd_name, gname, wrt, it) = grad_prog(src, of, wrt);
        let analytic = analytic(&prog, gname, inputs, lens, &wrt, &it);
        let fd = finite_diff(&prog, fwd_name, inputs, lens, &wrt, loss_out, eps, &it);
        for (gi, (a, f)) in analytic.iter().zip(fd.iter()).enumerate() {
            for (j, (&av, &fv)) in a.iter().zip(f.iter()).enumerate() {
                let tol = abs + rel * fv.abs();
                assert!(
                    (av - fv).abs() <= tol,
                    "wrt#{gi}[{j}]: analytic {av} vs finite-diff {fv} (|d|={:.3e} > {:.3e})",
                    (av - fv).abs(),
                    tol
                );
            }
        }
        analytic
    }

    fn assert_close(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (j, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 3e-3 + 3e-3 * w.abs(),
                "{what}[{j}]: got {g}, want {w}"
            );
        }
    }

    #[test]
    fn cubic_scalar_grad() {
        // loss = x^3  ->  dL/dx = 3x^2. Differentiated straight from `.wk` source via --emit=grad.
        let src = "module m\nfn loss(x:[f32;1], mut out:[f32;1]) -> f32 {\n\
                   let a: f32 = x[0]; let l: f32 = a*a*a; out[0] = l; return l; }";
        let x = 1.3f32;
        let inputs = vec![vec![x], vec![0.0]];
        let g = gate(src, "loss", &[0], &inputs, &[1, 1], 1, 1e-2, 2e-2, 1e-3);
        assert_close(&g[0], &[3.0 * (x * x) as f64], "d(x^3)/dx");
    }

    #[test]
    fn raw_pointer_parameter_grad() {
        // A loss whose buffers are raw `*mut f32` parameters rather than `[f32; N]` arrays.
        //
        // Until `mem2reg` promoted pointer slots this could not be differentiated at all: the
        // parameter's base pointer reached autodiff as `load ptr <slot>`, and `Vjp::canon` refused
        // it — `--emit=grad` reported "cannot route gradient for load pointer v31 (not a parameter
        // or a one-level gep of a parameter)". With the slot promoted, the base pointer *is* the
        // parameter value, so every access is a one-level gep off a parameter and the gradient
        // routes. `--emit=grad` forces `-O1`, so the promotion is always in effect here.
        //
        // loss = sum_i (w_i * x_i)^2  ->  dL/dw_i = 2 w_i x_i^2, dL/dx_i = 2 w_i^2 x_i.
        let src = "module m\nfn loss(w: *mut f32, x: *mut f32, mut out:[f32;1]) -> f32 {\n\
                   let a: f32 = w[0]*x[0]; let b: f32 = w[1]*x[1]; let c: f32 = w[2]*x[2];\n\
                   let l: f32 = a*a + b*b + c*c; out[0] = l; return l; }";
        let w = [0.5f32, -1.25, 0.75];
        let x = [1.5f32, 0.25, -2.0];
        let inputs = vec![w.to_vec(), x.to_vec(), vec![0.0]];
        let g = gate(
            src,
            "loss",
            &[0, 1],
            &inputs,
            &[3, 3, 1],
            2,
            1e-2,
            3e-2,
            2e-3,
        );
        let dw: Vec<f64> = (0..3).map(|i| (2.0 * w[i] * x[i] * x[i]) as f64).collect();
        let dx: Vec<f64> = (0..3).map(|i| (2.0 * w[i] * w[i] * x[i]) as f64).collect();
        assert_close(&g[0], &dw, "dL/dw = 2 w x^2");
        assert_close(&g[1], &dx, "dL/dx = 2 w^2 x");
    }

    #[test]
    fn bilinear_two_input_grad() {
        // loss = a * b^2  ->  dL/da = b^2, dL/db = 2ab.
        let src = "module m\nfn loss(a:[f32;1], b:[f32;1], mut out:[f32;1]) -> f32 {\n\
                   let av: f32 = a[0]; let bv: f32 = b[0]; let l: f32 = av*bv*bv;\n\
                   out[0] = l; return l; }";
        let (a, b) = (0.7f32, 1.1f32);
        let inputs = vec![vec![a], vec![b], vec![0.0]];
        let g = gate(
            src,
            "loss",
            &[0, 1],
            &inputs,
            &[1, 1, 1],
            2,
            1e-2,
            3e-2,
            1e-3,
        );
        assert_close(&g[0], &[(b * b) as f64], "dL/da = b^2");
        assert_close(&g[1], &[(2.0 * a * b) as f64], "dL/db = 2ab");
    }

    #[test]
    fn default_wrt_is_all_buffers() {
        // With no --grad-wrt, every buffer parameter is differentiated; the output buffer (only
        // stored, never read) gets a correct zero gradient.
        let src = "module m\nfn loss(x:[f32;1], mut out:[f32;1]) -> f32 {\n\
                   let a: f32 = x[0]; let l: f32 = a*a; out[0] = l; return l; }";
        let x = 2.0f32;
        let inputs = vec![vec![x], vec![0.0]];
        // wrt defaults to [0, 1] (both buffers).
        let g = gate(src, "loss", &[], &inputs, &[1, 1], 1, 1e-2, 2e-2, 1e-3);
        assert_close(&g[0], &[2.0 * x as f64], "dL/dx = 2x");
        assert_close(&g[1], &[0.0], "dL/d(out) = 0");
    }

    // --- Recognized-kernel tape: a real linear + MSE model, differentiated from source. -----------

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((*seed >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.8 // ~[-0.4, 0.4]
    }
    fn rand_vec(seed: &mut u64, n: usize) -> Vec<f32> {
        (0..n).map(|_| lcg(seed)).collect()
    }

    /// f64 reference for `p = X·Wᵀ` (m×k · n×k → m×n), the MSE loss `Σ(p−t)²`, and its gradients
    /// `dX[m,k] = Σ_n 2(p−t)[m,n] W[n,k]`, `dW[n,k] = Σ_m 2(p−t)[m,n] X[m,k]`.
    fn linmse_ref(
        x: &[f32],
        w: &[f32],
        t: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut d = vec![0.0f64; m * n];
        for mm in 0..m {
            for nn in 0..n {
                let mut s = 0.0;
                for kk in 0..k {
                    s += x[mm * k + kk] as f64 * w[nn * k + kk] as f64;
                }
                d[mm * n + nn] = s - t[mm * n + nn] as f64;
            }
        }
        let mut dx = vec![0.0f64; m * k];
        let mut dw = vec![0.0f64; n * k];
        for mm in 0..m {
            for nn in 0..n {
                let g = 2.0 * d[mm * n + nn];
                for kk in 0..k {
                    dx[mm * k + kk] += g * w[nn * k + kk] as f64;
                    dw[nn * k + kk] += g * x[mm * k + kk] as f64;
                }
            }
        }
        (dx, dw)
    }

    /// `--emit=grad` of a **real** linear-regression model written in `.wk`: `p = x·wᵀ` (recognized
    /// as `wukong_sgemm_nt`) then the MSE reduction `Σ(p−t)²` (recognized as the `@parallel`
    /// `wukong_sreduce_f32_parallel`). The emitted backward rides the tuned kernels — `wukong_velem`
    /// for the `2(p−t)` seed, a transpose loop, and `wukong_sgemm` for `dw = dpᵀ·x` / `dx = dp·w` —
    /// and is finite-difference-gated against the forward loss and the f64 closed form.
    #[test]
    fn linear_mse_from_source() {
        let (m, k, n) = (3usize, 4usize, 2usize);
        let src = "@parallel fn loss(x:[f32;12], w:[f32;8], t:[f32;6], mut out:[f32;1]) -> f32 {\n\
                   let mut p: [f32; 6] = [0.0; 6];\n\
                   for i in 0..3 { for j in 0..2 { let mut s: f32 = 0.0;\n\
                     for kk in 0..4 { s = s + x[i*4+kk] * w[j*4+kk]; }\n\
                     p[i*2+j] = s; } }\n\
                   let mut loss: f32 = 0.0;\n\
                   for i in 0..6 { loss = loss + (p[i]-t[i])*(p[i]-t[i]); }\n\
                   out[0] = loss; return loss; }";
        let mut seed = 0xA11CEu64;
        let x = rand_vec(&mut seed, m * k);
        let w = rand_vec(&mut seed, n * k);
        let t = rand_vec(&mut seed, m * n);
        let (dx_ref, dw_ref) = linmse_ref(&x, &w, &t, m, k, n);
        let inputs = vec![x, w, t, vec![0.0]];
        let lens = [m * k, n * k, m * n, 1];

        // Weights (param 1): the trainable gradient — must ride the GEMM and match the closed form.
        let gw = gate(src, "loss", &[1], &inputs, &lens, 3, 5e-3, 5e-2, 5e-3);
        assert_close(&gw[0], &dw_ref, "dW = dpᵀ·x");

        // Input (param 0): dx = dp·w — the other GEMM adjoint.
        let gx = gate(src, "loss", &[0], &inputs, &lens, 3, 5e-3, 5e-2, 5e-3);
        assert_close(&gx[0], &dx_ref, "dX = dp·w");
    }

    // --- Activation backward: a `matmul → act → sum` model, act backward riding wukong_vmath2_f32. ---

    /// The printed MIR of `{loss, loss_grad}` for the given source, so a test can assert the emitted
    /// backward rides a particular runtime kernel (the recognized backward is otherwise gate-blind).
    fn grad_mir(src: &str, of: &str, wrt: &[usize]) -> String {
        let (prog, _fwd, _g, _wrt, it) = grad_prog(src, of, wrt);
        wukong_mir::print::print_program(&prog, &it)
    }

    /// `p = x·wᵀ; h = act(p); loss = Σ h` — the smallest model that exercises a nonlinear activation
    /// backward composed with the matmul backward.
    fn act_sum_src(act: &str) -> String {
        format!(
            "@parallel fn loss(x:[f32;12], w:[f32;8], mut out:[f32;1]) -> f32 {{\n\
             let mut p: [f32; 6] = [0.0; 6];\n\
             for i in 0..3 {{ for j in 0..2 {{ let mut s: f32 = 0.0;\n\
               for kk in 0..4 {{ s = s + x[i*4+kk] * w[j*4+kk]; }}\n\
               p[i*2+j] = s; }} }}\n\
             let mut h: [f32; 6] = [0.0; 6];\n\
             for i in 0..6 {{ h[i] = {act}(p[i]); }}\n\
             let mut loss: f32 = 0.0;\n\
             for i in 0..6 {{ loss = loss + h[i]; }}\n\
             out[0] = loss; return loss; }}"
        )
    }

    /// silu and gelu — the modern-transformer activations — differentiate from source: the backward
    /// rides the fused `wukong_vmath2_f32` (`dx = dy·act'(x)`, bit-identical with the forward) for the
    /// activation and `wukong_sgemm` for the matmul adjoint. Finite-difference-gated against the
    /// forward loss (the runtime activation's own derivative is the oracle, so no closed form to
    /// re-derive — the fused kernel and the forward share the same sigmoid/tanh polynomials).
    #[test]
    fn silu_and_gelu_backward_from_source() {
        let (m, k, n) = (3usize, 4usize, 2usize);
        let lens = [m * k, n * k, 1];
        for act in ["silu", "gelu"] {
            let src = act_sum_src(act);
            // The backward must ride the fused activation kernel + the tuned GEMM.
            let mir = grad_mir(&src, "loss", &[1]);
            assert!(
                mir.contains("wukong_vmath2_f32"),
                "{act}: backward does not ride wukong_vmath2_f32\n{mir}"
            );
            assert!(
                mir.contains("wukong_sgemm("),
                "{act}: matmul backward does not ride wukong_sgemm\n{mir}"
            );
            // Data kept modest so the activations stay in a well-conditioned range for the difference.
            let mut seed = 0xE1F5u64 ^ (act.as_bytes()[0] as u64);
            let x = rand_vec(&mut seed, m * k);
            let w = rand_vec(&mut seed, n * k);
            let inputs = vec![x, w, vec![0.0]];
            // wrt w and wrt x — both adjoints flow through the activation backward.
            gate(&src, "loss", &[1], &inputs, &lens, 2, 1e-2, 4e-2, 1e-2);
            gate(&src, "loss", &[0], &inputs, &lens, 2, 1e-2, 4e-2, 1e-2);
        }
    }

    // --- --train: fwd → bwd → optimizer, the end-to-end loss-decrease signal. -----------------------

    /// A least-squares linear regression `loss = Σ(x·wᵀ − t)²` — convex in the weights `w`, so
    /// full-batch gradient descent with a small step descends monotonically.
    const REGRESSION_SRC: &str =
        "@parallel fn loss(x:[f32;12], w:[f32;8], t:[f32;6], mut out:[f32;1]) -> f32 {\n\
         let mut p: [f32; 6] = [0.0; 6];\n\
         for i in 0..3 { for j in 0..2 { let mut s: f32 = 0.0;\n\
           for kk in 0..4 { s = s + x[i*4+kk] * w[j*4+kk]; }\n\
           p[i*2+j] = s; } }\n\
         let mut loss: f32 = 0.0;\n\
         for i in 0..6 { loss = loss + (p[i]-t[i])*(p[i]-t[i]); }\n\
         out[0] = loss; return loss; }";

    #[allow(clippy::too_many_arguments)]
    fn train_traj(
        src: &str,
        wrt: &[usize],
        lens: &[usize],
        steps: usize,
        lr: f32,
        opt: TrainOpt,
        seed: u64,
    ) -> Vec<f32> {
        let (program, mut interner) = compile_o1(src);
        let opts = Options {
            grad: GradOptions {
                of: Some("loss".to_string()),
                wrt: wrt.to_vec(),
                train: true,
                train_steps: steps,
                train_lr: lr,
                train_opt: opt,
                train_seed: seed,
            },
            ..Options::default()
        };
        train_loop(&program, &mut interner, &opts, lens).expect("train_loop")
    }

    /// The loss function's buffer sizes are recovered from the semantic types (`[f32; N]`), not the
    /// `Ptr`-erased MIR — so `--train` can allocate the right buffers with no user-supplied shapes.
    #[test]
    fn param_sizes_recovered_from_types() {
        let mut sm = SourceMap::new();
        let id = sm.add("sz.wk".to_string(), REGRESSION_SRC.to_string());
        let (tokens, _) = wukong_lexer::tokenize(sm.source(id), id);
        let mut interner = Interner::new();
        let (module, _) = wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()));
        let opts = Options {
            grad: GradOptions {
                of: Some("loss".to_string()),
                ..GradOptions::default()
            },
            ..Options::default()
        };
        let lens = loss_param_lens(&sema, &mut interner, &opts).expect("param sizes");
        assert_eq!(lens, vec![12, 8, 6, 1], "x[12], w[8], t[6], out[1]");
    }

    #[test]
    fn train_sgd_loss_strictly_decreases() {
        // Full-batch GD on a convex least-squares objective: the loss never rises (f32 rounding slack).
        let traj = train_traj(
            REGRESSION_SRC,
            &[1],
            &[12, 8, 6, 1],
            120,
            0.05,
            TrainOpt::Sgd,
            0xA5A5,
        );
        for w in traj.windows(2) {
            assert!(w[1] <= w[0] + 1e-6, "SGD loss rose: {} -> {}", w[0], w[1]);
        }
        let (l0, lf) = (traj[0], *traj.last().unwrap());
        assert!(
            l0 > 0.05,
            "test setup: initial loss should be substantial, got {l0}"
        );
        assert!(
            lf < 0.3 * l0,
            "SGD did not reduce the loss enough: {l0} -> {lf}"
        );
    }

    /// The documented `--grad-wrt` default ("all buffer parameters of the loss function ... for
    /// `--train`, these are the trainable weights") must actually run. It used to expand to a list
    /// that necessarily contained the loss-output parameter, which `train_loop` then rejected, so
    /// `wukongc --train --grad-of=loss` always died with "--grad-wrt must not include the
    /// loss-output parameter (the last one)" unless an explicit list was supplied.
    #[test]
    fn train_default_wrt_excludes_the_loss_output() {
        let traj = train_traj(
            REGRESSION_SRC,
            &[],
            &[12, 8, 6, 1],
            40,
            0.02,
            TrainOpt::Sgd,
            0xA5A5,
        );
        assert_eq!(
            traj.len(),
            41,
            "traj[i] = loss after i updates, plus a final"
        );
        let (l0, lf) = (traj[0], *traj.last().unwrap());
        assert!(
            lf < l0,
            "the default --grad-wrt did not train: {l0} -> {lf}"
        );
    }

    /// A repeated `--grad-wrt` index appends a second gradient buffer to the backward's parameter
    /// list while `Vjp`'s param -> buffer map keeps only the last insert, so the first appended
    /// buffer is never written and the caller reads back zeros with exit 0. Reject it up front.
    #[test]
    fn duplicate_grad_wrt_index_is_rejected() {
        let (program, mut interner) = compile_o1(REGRESSION_SRC);
        let opts = Options {
            grad: GradOptions {
                of: Some("loss".to_string()),
                wrt: vec![1, 1],
                ..GradOptions::default()
            },
            ..Options::default()
        };
        let err = build_grad(&program, &mut interner, &opts)
            .expect_err("a repeated --grad-wrt index must be rejected");
        assert!(err.contains("repeated"), "unexpected error: {err}");
    }

    #[test]
    fn train_adamw_reduces_loss() {
        // The fused AdamW kernel drives a much larger reduction than plain SGD in the same budget.
        let traj = train_traj(
            REGRESSION_SRC,
            &[1],
            &[12, 8, 6, 1],
            150,
            0.05,
            TrainOpt::AdamW,
            0xA5A5,
        );
        let (l0, lf) = (traj[0], *traj.last().unwrap());
        assert!(
            lf.is_finite() && lf < 0.05 * l0,
            "AdamW did not converge: {l0} -> {lf}"
        );
    }

    // --- Transformer FFN block: two matmuls + activation + reduction, differentiated from source. ---

    /// A transformer **feed-forward block** `y = silu(x·W1ᵀ)·W2ᵀ; loss = Σy`, the composition at the
    /// heart of every transformer layer. Its whole backward rides tuned kernels: `wukong_vmath2_f32`
    /// for the activation and *three* `wukong_sgemm` for the matmul adjoints (dW2, dA, dW1) — the
    /// headline capability, a real block differentiated straight from `.wk` and finite-difference-gated.
    #[test]
    fn transformer_ffn_block_from_source() {
        // x[M=2,K=4], W1[H=3,K=4] -> p[2,3]; a=silu(p); W2[N=2,H=3] -> y[2,2]; loss=Σy.
        let src =
            "@parallel fn loss(x:[f32;8], w1:[f32;12], w2:[f32;6], mut out:[f32;1]) -> f32 {\n\
             let mut p: [f32; 6] = [0.0; 6];\n\
             for i in 0..2 { for j in 0..3 { let mut s: f32 = 0.0;\n\
               for kk in 0..4 { s = s + x[i*4+kk] * w1[j*4+kk]; } p[i*3+j] = s; } }\n\
             let mut a: [f32; 6] = [0.0; 6];\n\
             for i in 0..6 { a[i] = silu(p[i]); }\n\
             let mut y: [f32; 4] = [0.0; 4];\n\
             for i in 0..2 { for j in 0..2 { let mut s: f32 = 0.0;\n\
               for kk in 0..3 { s = s + a[i*3+kk] * w2[j*3+kk]; } y[i*2+j] = s; } }\n\
             let mut loss: f32 = 0.0;\n\
             for i in 0..4 { loss = loss + y[i]; }\n\
             out[0] = loss; return loss; }";
        // The composite backward must ride the fused activation kernel + the tuned GEMM (three of them).
        let mir = grad_mir(src, "loss", &[1, 2]);
        assert!(
            mir.contains("wukong_vmath2_f32"),
            "silu backward missing\n{mir}"
        );
        assert_eq!(
            mir.matches("wukong_sgemm(").count(),
            3,
            "expected 3 GEMM adjoints (dW2, dA, dW1)\n{mir}"
        );

        let mut seed = 0x77A0u64;
        let x = rand_vec(&mut seed, 8);
        let w1 = rand_vec(&mut seed, 12);
        let w2 = rand_vec(&mut seed, 6);
        let inputs = vec![x, w1, w2, vec![0.0]];
        let lens = [8, 12, 6, 1];
        // Finite-difference-gate the composite gradient w.r.t. both weight matrices and the input.
        gate(src, "loss", &[1], &inputs, &lens, 3, 1e-2, 5e-2, 1e-2); // dW1
        gate(src, "loss", &[2], &inputs, &lens, 3, 1e-2, 5e-2, 1e-2); // dW2
        gate(src, "loss", &[0], &inputs, &lens, 3, 1e-2, 5e-2, 1e-2); // dX
    }
}

/// The MIR-verifier gate that stands in front of every backend entry.
#[cfg(test)]
mod verify_gate_tests {
    use super::*;
    use wukong_mir::{BasicBlock, BlockId, Function, MirType, Program, Terminator, ValueId};

    /// A function that returns a value no block parameter and no instruction ever defines — the
    /// "use of undefined value" class the verifier exists to catch.
    fn invalid_program(interner: &mut Interner) -> Program {
        let f = Function {
            name: interner.intern("bad"),
            params: Vec::new(),
            ret: MirType::I32,
            blocks: vec![BasicBlock {
                id: BlockId(0),
                params: Vec::new(),
                insts: Vec::new(),
                term: Terminator::Ret(Some(ValueId(0))),
            }],
            value_types: vec![MirType::I32],
            entry: BlockId(0),
            vec_kernels: Vec::new(),
        };
        Program {
            funcs: vec![f],
            statics: Vec::new(),
            level: wukong_mir::MirLevel::Low,
        }
    }

    /// `verify_or_ice` is what `--run`, `--emit=llvm-ir`, `--emit=obj` and `--emit=exe` now all call
    /// before entering a backend; before this it was inlined in the `--run` arm only, so the three
    /// AOT exits reached their backend unverified.
    #[test]
    fn verify_or_ice_reports_an_undefined_value_use() {
        let mut interner = Interner::new();
        assert!(
            verify_or_ice(&invalid_program(&mut interner)),
            "a `ret` of an undefined value must be reported"
        );
    }

    /// And it must stay quiet on well-formed MIR, or every AOT compile would start failing.
    #[test]
    fn verify_or_ice_accepts_a_real_compiled_program() {
        let src = "module m\nfn main() -> i32 { let x: f32 = 0.5; print(x); return 0; }";
        let mut sm = SourceMap::new();
        let id = sm.add("v.wk".to_string(), src.to_string());
        let (tokens, _) = wukong_lexer::tokenize(sm.source(id), id);
        let mut interner = Interner::new();
        let (module, _) = wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()), "sema errors");
        for level in [0u8, 1, 2] {
            let (mut program, md) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
            assert!(!md.iter().any(|d| d.is_error()), "mir_build errors");
            wukong_opt::optimize(&mut program, level);
            assert!(
                !verify_or_ice(&program),
                "clean program failed verify at -O{level}"
            );
        }
    }
}

/// The `--emit=exe` link path's generated inputs (the Rust shim, the C fallback runtime).
#[cfg(test)]
mod native_link_tests {
    use super::*;

    /// They must never be generated in the directory the user invoked the compiler from:
    /// `{stem}_shim.rs` was written there and then unconditionally deleted, so an `--emit=exe` run
    /// in a source directory holding a file of that name destroyed it.
    #[test]
    fn exe_link_inputs_are_generated_outside_the_working_directory() {
        let cwd = std::env::current_dir().expect("cwd");
        let scratch = ScratchDir::new("scratch_probe").expect("scratch dir");
        let shim = scratch_path(Some(&scratch), "scratch_probe_shim.rs".to_string());
        let rt = scratch_path(Some(&scratch), "scratch_probe_rt.c".to_string());
        for p in [&shim, &rt] {
            assert!(
                !p.starts_with(&cwd),
                "`{}` lands in the working directory",
                p.display()
            );
        }
        std::fs::write(&shim, "// probe").expect("write shim");
        let dir = scratch.path.clone();
        drop(scratch);
        assert!(
            !dir.exists(),
            "the scratch directory must be removed when the link finishes"
        );
    }

    /// The `cc` fallback must link `libm`. `WUKONG_RT_C` always defines `wukong_rt_fmod_*` over
    /// `fmod`/`fmodf`, which glibc keeps in `libm` and `cc` does not link by default — so the whole
    /// fallback path failed to link *every* program on Linux with `undefined reference to 'fmod'`.
    /// A library must also follow the objects that reference it, or GNU `ld` discards it unused.
    #[test]
    fn the_cc_fallback_links_libm_after_its_objects() {
        let args = cc_link_args(
            Path::new("prog.o"),
            Path::new("prog_rt.c"),
            Path::new("prog"),
        );
        let lm = args
            .iter()
            .position(|a| a == "-lm")
            .expect("the `cc` fallback link line must pass `-lm`");
        let rt = args
            .iter()
            .position(|a| a == "prog_rt.c")
            .expect("the generated C runtime must be on the link line");
        assert!(
            lm > rt,
            "`-lm` must follow `prog_rt.c` (GNU ld resolves left to right): {args:?}"
        );
    }

    /// The runtime rlib must be found in `deps/` under its hash-suffixed name, not only under the
    /// uplifted `libwukong_runtime.rlib`. Cargo uplifts a library only when it is a root unit of a
    /// **`build`**, so after `cargo test --workspace` (what CI runs) or `cargo run -p wukongc` the
    /// uplifted copy does not exist and only `deps/` has it — and `--emit=exe` silently degraded to
    /// the `cc` fallback, which cannot resolve `wukong_*` at all.
    #[test]
    fn the_runtime_rlib_is_found_in_the_deps_directory() {
        let dir = std::env::temp_dir().join(format!("wukong_rlib_probe-{}", std::process::id()));
        let deps = dir.join("deps");
        // A pid can be recycled, and a previous *failed* run leaves its files behind — start clean
        // so the "nothing matches" assertions below cannot be poisoned by an earlier process.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&deps).expect("create probe layout");

        // Nothing at all: the `cc` fallback is genuinely all there is.
        assert_eq!(runtime_rlib_in(&dir), None, "empty layout must not match");

        // Decoys: another crate's rlib, a same-prefixed crate name, and a non-rlib artifact.
        for decoy in [
            "libwukong_interp-1111111111111111.rlib",
            "libwukong_runtime_extra-2222222222222222.rlib",
            "libwukong_runtime-3333333333333333.rmeta",
        ] {
            std::fs::write(deps.join(decoy), b"x").expect("write decoy");
        }
        assert_eq!(runtime_rlib_in(&dir), None, "a decoy must not match");

        // The cargo-test / cargo-run layout: hash-suffixed, in `deps/` only.
        let hashed = deps.join("libwukong_runtime-4444444444444444.rlib");
        std::fs::write(&hashed, b"x").expect("write hashed rlib");
        assert_eq!(
            runtime_rlib_in(&dir),
            Some(hashed),
            "the hash-suffixed rlib in `deps/` must be found"
        );

        // The `cargo build` layout wins when both exist: it is the unambiguous one.
        let uplifted = dir.join("libwukong_runtime.rlib");
        std::fs::write(&uplifted, b"x").expect("write uplifted rlib");
        assert_eq!(
            runtime_rlib_in(&dir),
            Some(uplifted),
            "the uplifted rlib must take precedence"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// End-to-end GPU-backend gate: the `--backend=gpu` path (offloading interpreter + [`gpu_accel`])
/// must match the pure-interpreter oracle within the CPU↔GPU tolerance over the **same** lowered MIR.
/// Only built with `--features gpu`; skips when no CUDA device is present — loudly, and as a
/// *failure* under `WUKONG_GPU_REQUIRED=1` (see `skip_no_device`).
#[cfg(all(test, feature = "gpu"))]
mod gpu_e2e_tests {
    use super::*;
    use wukong_codegen_gpu::diff::{assert_close, Rng};
    use wukong_span::SourceMap;

    /// Every gate below early-`return`s when no CUDA device is reachable. libtest captures stderr on
    /// a *pass*, so that return is invisible: the crate reports `test result: ok` having offloaded
    /// nothing. `WUKONG_GPU_REQUIRED=1` — what a rented-GPU run sets — must turn that into a failure,
    /// exactly as it already does for every device gate inside `wukong_codegen_gpu`
    /// (`wukong_codegen_gpu::diff::skip_or_fail`, which this delegates to so one env var still
    /// governs every skip in the workspace).
    fn skip_no_device(name: &str) {
        wukong_codegen_gpu::diff::skip_or_fail(
            name,
            wukong_codegen_gpu::gpu::init_error().unwrap_or("no CUDA device reachable"),
        );
    }

    /// Lex → parse → sema → mir_build → opt(2), asserting each stage is clean. The recognizers run in
    /// mir_build, so the resulting MIR already carries the `wukong_sgemm_nt` call the GPU offloads.
    fn build(src: &str) -> (wukong_mir::Program, Interner) {
        let mut sm = SourceMap::new();
        let id = sm.add("gpu_e2e.wk".to_string(), src.to_string());
        let (tokens, ld) = wukong_lexer::tokenize(sm.source(id), id);
        assert!(!ld.iter().any(|d| d.is_error()), "lex errors");
        let mut interner = Interner::new();
        let (module, pd) =
            wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        assert!(!pd.iter().any(|d| d.is_error()), "parse errors");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()), "sema errors");
        let (mut program, md) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(!md.iter().any(|d| d.is_error()), "mir_build errors");
        wukong_opt::optimize(&mut program, 2);
        (program, interner)
    }

    /// The `ijk` `C = A·Bᵀ` (nn.Linear) nest that the recognizer lowers to `wukong_sgemm_nt`.
    fn linear_src(m: usize, k: usize, n: usize) -> String {
        format!(
            "module m\nfn lin(a:[f32;{mk}],b:[f32;{nk}],mut c:[f32;{mn}]) {{ \
             for i in 0..{m} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for kk in 0..{k} {{ s = s + a[i*{k}+kk] * b[j*{k}+kk]; }} c[i*{n}+j] = s; }} }} }}",
            mk = m * k,
            nk = n * k,
            mn = m * n
        )
    }

    /// The `C = A·Bᵀ` nest followed by a separate elementwise activation loop over `C` — which the
    /// recognizer folds into one `wukong_sgemm_nt_epi` (bias-free, the SwiGLU/FFN `act(x·Wᵀ)` shape).
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
            "module m\nfn lin(a:[f32;{mk}],b:[f32;{nk}],mut c:[f32;{mn}]) {{ \
             for i in 0..{m} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for kk in 0..{k} {{ s = s + a[i*{k}+kk] * b[j*{k}+kk]; }} c[i*{n}+j] = s; }} }} \
             for i in 0..{m} {{ for j in 0..{n} {{ c[i*{n}+j] = {act}; }} }} }}",
            mk = m * k,
            nk = n * k,
            mn = m * n
        )
    }

    /// The `C = A·Bᵀ` nest followed by a **bias-add (+ optional activation)** epilogue loop — which the
    /// recognizer folds into one `wukong_sgemm_nt_epi` with a *non-null* bias (the canonical
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
            "module m\nfn lin(a:[f32;{mk}],b:[f32;{nk}],bias:[f32;{n}],mut c:[f32;{mn}]) {{ \
             for i in 0..{m} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for kk in 0..{k} {{ s = s + a[i*{k}+kk] * b[j*{k}+kk]; }} c[i*{n}+j] = s; }} }} \
             for i in 0..{m} {{ for j in 0..{n} {{ c[i*{n}+j] = {act}; }} }} }}",
            mk = m * k,
            nk = n * k,
            mn = m * n
        )
    }

    /// **The tolerance band a plain-GEMM offload is held to, chosen by ROUTE and not by device.**
    ///
    /// [`gpu_accel::GemmRoute::Existing`] is `gpu::gemm_nt`, an f32 kernel: both sides are f32 and
    /// only the K-reduction order differs, so the band is `c·√K·ε` — the `sgemm_matches_naive` shape.
    /// [`gpu_accel::GemmRoute::Wgmma`] converts both operands to f16 on the host before accumulating
    /// in f32, so it carries an input rounding the CPU oracle does not and needs the fp16 band the
    /// fused-epilogue gates below already use.
    ///
    /// It is a function of the route because that is the fact that changed: sizing it by device would
    /// silently widen the band on a Hopper part that had *declined* the wgmma route (an unencodable
    /// K, or `WUKONG_GPU_NO_WGMMA=1`) and let a real f32 regression through.
    fn gemm_tolerance(route: gpu_accel::GemmRoute, k: usize) -> (f64, f64) {
        match route {
            gpu_accel::GemmRoute::Existing => (
                1e-4,
                (16.0 * (k as f64).sqrt() * f32::EPSILON as f64).max(1e-5),
            ),
            gpu_accel::GemmRoute::Wgmma => (5e-2, 2e-2),
        }
    }

    #[test]
    fn gpu_backend_linear_matches_interp_within_tol() {
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device("gpu_backend_linear_matches_interp_within_tol");
                return;
            }
        };
        let cc = g.target().cc();
        let mut rng = Rng::new(0x00C0FFEE);
        for &(m, k, n) in &[(8usize, 16usize, 8usize), (16, 32, 16), (32, 48, 24)] {
            let (program, mut interner) = build(&linear_src(m, k, n));
            let entry = interner.intern("lin");
            let a = rng.vec(m * k, -1.0, 1.0);
            let b = rng.vec(n * k, -1.0, 1.0);

            // CPU oracle (no accelerator).
            let (mut ac, mut bc, mut cc_buf) = (a.clone(), b.clone(), vec![0f32; m * n]);
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ac, &mut bc, &mut cc_buf];
                wukong_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
            }

            // GPU offload over the identical MIR + inputs.
            let (mut ag, mut bg, mut cg) = (a.clone(), b.clone(), vec![0f32; m * n]);
            let mut accel = gpu_accel::GpuAccel::new(&mut *g);
            let route = accel.gemm_route();
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ag, &mut bg, &mut cg];
                wukong_interp::run_kernel_f32_accel(
                    &program, entry, &mut bufs, &interner, &mut accel,
                )
                .unwrap();
            }
            assert!(
                accel.calls >= 1,
                "{m}x{k}x{n}: GPU offload never fired — the nest did not lower to sgemm_nt, so this \
                 would silently test CPU-vs-CPU"
            );
            // The routing law, on whatever device is actually present: `wgmma` is architecture-locked
            // to `sm_90a`, so anything that is not Hopper must be on the pre-Hopper launcher. On this
            // repo's RTX 4050 (cc 8.9) that is the assertion that keeps wave 4 from changing the
            // arithmetic of a path it was never meant to touch.
            assert!(
                cc.0 == 9 || route == gpu_accel::GemmRoute::Existing,
                "cc {cc:?} is not Hopper but took {route:?} — an sm_90a module cannot load here"
            );

            let (abs_tol, rel_tol) = gemm_tolerance(route, k);
            let s = assert_close(
                &format!("linear {m}x{k}x{n}"),
                &cg,
                &cc_buf,
                abs_tol,
                rel_tol,
            );
            eprintln!(
                "gpu --backend linear {m}x{k}x{n} [{route:?}]: {} GPU call(s), max_abs={:.2e} max_rel={:.2e}",
                accel.calls, s.max_abs, s.max_rel
            );
        }
    }

    /// **The wgmma route's end-to-end gate: a `.wk` matmul reaching `gemm_nt_wgmma`.**
    ///
    /// `#[ignore]`d because it can only pass on a Hopper part, and this repo develops on an RTX 4050.
    /// The reachability law says `::bench --name X` runs ONLY `#[ignore]`d tests and `::test --filter
    /// Y` runs only plain ones, so the H100 invocation is
    ///
    /// ```text
    ///   WK_GPU=H100 modal run --detach tools/cloud/modal_app.py::bench \
    ///       --name gpu_backend_linear_routes_through_wgmma_on_hopper --package wukong_driver
    /// ```
    ///
    /// (after `::build --release`, which `::bench` requires). It asserts three separate things, and
    /// the first two are the ones a green-but-vacuous run would skip: that the route on this device
    /// really is [`gpu_accel::GemmRoute::Wgmma`], that the offload fired at all, and only then that
    /// the numbers match the CPU oracle.
    ///
    /// Three shapes, one per property, and they live in [`gpu_accel::HOPPER_GATE_SHAPES`] because
    /// `gpu_accel::tests::the_hopper_gate_shapes_reach_both_regime_arms` checks *on this laptop* what
    /// they are chosen for. `256x1024x512` is a whole 128×256 tile grid. `130x1032x258` is ragged in
    /// M, N **and** K at once, which is what proves the driver seam did not quietly need an alignment
    /// gate (the `m%64/n%64/k%16` rule in `sgemm_nt_epi` is a *wmma* constraint — wgmma predicates its
    /// own edge). `4096x256x2048` has `M*N = 8_388_608` output elements, just past
    /// `ptx_wgmma::W1_CLUSTER_MIN_OUTPUT_ELEMS`, so it is the only one that makes the config seam
    /// return the **clustered** row and the launch carry a `1x2x1` cluster attribute — the arm a
    /// small-shape-only gate would leave entirely unexercised from the driver side, and a *launch*
    /// failure rather than a wrong number if it were ever mismatched.
    ///
    /// Every `N` here is EVEN on purpose, and it is not a raggedness choice: both shipped rows carry
    /// the `st.global.v2.f32` epilogue, whose 8-byte pair is aligned only when `N` is, so an odd `N`
    /// is a *sticky* `CUDA_ERROR_MISALIGNED_ADDRESS` rather than a wrong number. The driver declines
    /// it (`wgmma_declines`) and a device-free unit test pins that; putting an odd `N` in this gate
    /// would only prove the decline works by never reaching the kernel.
    #[test]
    #[ignore = "needs a Hopper (sm_90a) device; run it on H100 via ::bench --package wukong_driver"]
    fn gpu_backend_linear_routes_through_wgmma_on_hopper() {
        const NAME: &str = "gpu_backend_linear_routes_through_wgmma_on_hopper";
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device(NAME);
                return;
            }
        };
        let cc = g.target().cc();
        if cc.0 != 9 {
            // A CAPABILITY skip, not a device skip, so it must NOT escalate under
            // `WUKONG_GPU_REQUIRED=1`: `sm_90a` is an architecture lock, and declining on a part that
            // is not Hopper is the correct outcome (the rule `gpu.rs`'s `with_hopper` states).
            eprintln!(
                "[skip] {NAME}: wgmma is architecture-locked to sm_90a and this device is cc {}.{} \
                 ({}) — the route is correctly `Existing` here",
                cc.0,
                cc.1,
                g.device_name()
            );
            return;
        }
        let mut rng = Rng::new(0x090A_C0DE);
        for &(m, k, n) in &gpu_accel::HOPPER_GATE_SHAPES {
            let (program, mut interner) = build(&linear_src(m, k, n));
            let entry = interner.intern("lin");
            let a = rng.vec(m * k, -1.0, 1.0);
            let b = rng.vec(n * k, -1.0, 1.0);

            let (mut ac, mut bc, mut cpu) = (a.clone(), b.clone(), vec![0f32; m * n]);
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ac, &mut bc, &mut cpu];
                wukong_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
            }

            let (mut ag, mut bg, mut got) = (a.clone(), b.clone(), vec![0f32; m * n]);
            let mut accel = gpu_accel::GpuAccel::new(&mut *g);
            let route = accel.gemm_route();
            assert_eq!(
                route,
                gpu_accel::GemmRoute::Wgmma,
                "this device is Hopper (cc {cc:?}) but the plain-GEMM offload routed to {route:?} — \
                 either the routing rule regressed or WUKONG_GPU_NO_WGMMA is set, and in both cases \
                 the rest of this gate would measure the kernel it exists to bypass"
            );
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ag, &mut bg, &mut got];
                wukong_interp::run_kernel_f32_accel(
                    &program, entry, &mut bufs, &interner, &mut accel,
                )
                .unwrap();
            }
            assert!(
                accel.calls >= 1,
                "{m}x{k}x{n}: nothing offloaded, so this would silently test CPU-vs-CPU"
            );

            let (abs_tol, rel_tol) = gemm_tolerance(route, k);
            let s = assert_close(
                &format!("wgmma linear {m}x{k}x{n}"),
                &got,
                &cpu,
                abs_tol,
                rel_tol,
            );
            eprintln!(
                "gpu --backend wgmma linear {m}x{k}x{n} (M%128={} N%256={} K%64={}): \
                 {} GPU call(s), max_abs={:.2e} max_rel={:.2e}",
                m % 128,
                n % 256,
                k % 64,
                accel.calls,
                s.max_abs,
                s.max_rel
            );
        }
    }

    /// **`act(matmul(x,w))` from Wukong source runs the fused tensor-core kernel** (Phase-2 fusion
    /// engine). The recognizer folds the matmul + activation loop into `wukong_sgemm_nt_epi`; the GPU
    /// `Accelerator` routes that to the single fused WMMA kernel (`gemm_nt_f16_sm_db_{relu,silu,gelu}`,
    /// the one that beats the cuBLAS GEMM+activation chain). Aligned shapes (M,N %64, K %16) so the
    /// tensor-core tiling accepts them. The offload must fire (`calls >= 1`, else it would silently
    /// test CPU-vs-CPU); the fp16 path is tolerance-gated against the f32 CPU oracle, not bit-exact.
    #[test]
    fn gpu_backend_fused_epilogue_matches_interp() {
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device("gpu_backend_fused_epilogue_matches_interp");
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
                    wukong_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
                }

                // GPU offload over the identical MIR + inputs.
                let (mut ag, mut bg, mut cg) = (a.clone(), b.clone(), vec![0f32; m * n]);
                let mut accel = gpu_accel::GpuAccel::new(&mut *g);
                {
                    let mut bufs: [&mut [f32]; 3] = [&mut ag, &mut bg, &mut cg];
                    wukong_interp::run_kernel_f32_accel(
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
                let s = assert_close(
                    &format!("fused {actname} {m}x{k}x{n}"),
                    &cg,
                    &cc,
                    5e-2,
                    2e-2,
                );
                eprintln!(
                    "gpu --backend fused {actname} {m}x{k}x{n}: {} GPU call(s), max_abs={:.2e} max_rel={:.2e}",
                    accel.calls, s.max_abs, s.max_rel
                );
            }
        }
    }

    /// **`act(matmul(x,w) + bias)` from Wukong source runs the fused tensor-core *bias* kernel** — the
    /// canonical `nn.Linear`/FFN epilogue (the affine `none` case is a plain biased Linear). The
    /// recognizer folds matmul + bias-add (+ activation) into `wukong_sgemm_nt_epi` with a non-null
    /// bias; the GPU `Accelerator` routes it to `gemm_nt_f16_sm_db_bias{,_relu,_silu,_gelu}` (which
    /// store each tile through SMEM to add the per-column bias the opaque WMMA fragment layout otherwise
    /// blocks). Aligned shapes; the offload must fire (`calls >= 1`, else it would silently test
    /// CPU-vs-CPU); fp16 path → tolerance-gated against the f32 CPU oracle, not bit-exact.
    #[test]
    fn gpu_backend_fused_bias_epilogue_matches_interp() {
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device("gpu_backend_fused_bias_epilogue_matches_interp");
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
                    wukong_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
                }

                // GPU offload over the identical MIR + inputs.
                let (mut ag, mut bg, mut biasg, mut cg) =
                    (a.clone(), b.clone(), bias.clone(), vec![0f32; m * n]);
                let mut accel = gpu_accel::GpuAccel::new(&mut *g);
                {
                    let mut bufs: [&mut [f32]; 4] = [&mut ag, &mut bg, &mut biasg, &mut cg];
                    wukong_interp::run_kernel_f32_accel(
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

                let s = assert_close(
                    &format!("fused bias {actname} {m}x{k}x{n}"),
                    &cg,
                    &cc,
                    5e-2,
                    2e-2,
                );
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
        g: &mut wukong_codegen_gpu::Gpu,
        program: &wukong_mir::Program,
        entry: wukong_span::Symbol,
        interner: &Interner,
        init: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, u32) {
        let mut cpu: Vec<Vec<f32>> = init.to_vec();
        {
            let mut refs: Vec<&mut [f32]> = cpu.iter_mut().map(|v| v.as_mut_slice()).collect();
            wukong_interp::run_kernel_f32(program, entry, &mut refs, interner).unwrap();
        }
        let mut gpu: Vec<Vec<f32>> = init.to_vec();
        let mut accel = gpu_accel::GpuAccel::new(g);
        {
            let mut refs: Vec<&mut [f32]> = gpu.iter_mut().map(|v| v.as_mut_slice()).collect();
            wukong_interp::run_kernel_f32_accel(program, entry, &mut refs, interner, &mut accel)
                .unwrap();
        }
        (cpu, gpu, accel.calls)
    }

    /// The other three GPU kernel families behind `--backend=gpu`: activation (vmath), reduction
    /// (sreduce), and a fused row norm (softmax). Each must fire on the device and match the interp
    /// oracle within tolerance (transcendental SFU approximations are looser than the GEMM bound).
    #[test]
    fn gpu_backend_activation_reduction_norm_match_interp() {
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device("gpu_backend_activation_reduction_norm_match_interp");
                return;
            }
        };
        let mut rng = Rng::new(0x5EED_1234);

        // 1) Activation: out[i] = silu(x[i]) -> wukong_vmath_f32 (SFU sigmoid approx on GPU).
        {
            let n = 64usize;
            let src = format!(
                "module m\nfn act(x:[f32;{n}], mut out:[f32;{n}]) {{ \
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

        // 2) Reduction: o[0] = Σ x·y -> wukong_sreduce_f32_parallel (dot, written to a [1] buffer).
        {
            let n = 1024usize;
            let src = format!(
                "@parallel fn dotp(x:[f32;{n}], y:[f32;{n}], mut o:[f32;1]) {{ \
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

        // 3) Fused norm: batched softmax over [R,C] -> wukong_norm_f32 (in place; GPU uses ex2.approx).
        {
            let (r, c) = (4usize, 16usize);
            let n = r * c;
            let src = format!(
                "module m\nfn sm(mut x:[f32;{n}]) {{ for row in 0..{r} {{ \
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
                "norm offload never fired (softmax did not lower to wukong_norm_f32)"
            );
            let s = assert_close("batched softmax", &gpu[0], &cpu[0], 2e-3, 5e-3);
            eprintln!(
                "gpu --backend softmax[{r}x{c}]: {calls} call(s), max_abs={:.2e} max_rel={:.2e}",
                s.max_abs, s.max_rel
            );
        }
    }

    /// The two `NORM_*` op codes the recognizer emits but the GPU `norm` kernel has no PTX entry for
    /// (LOGSOFTMAX=3, L2NORM=4). Before the `norm_supported` gate these forwarded to `gpu::norm` and
    /// hit its `_ => panic!("norm op {op} not implemented on GPU yet")`, aborting the process on
    /// ordinary user input. They must now decline to the CPU kernel: zero device calls, and — since
    /// nothing ran on the device — output bit-identical to the interpreter oracle.
    #[test]
    fn gpu_backend_declines_unimplemented_norm_ops_to_cpu() {
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device("gpu_backend_declines_unimplemented_norm_ops_to_cpu");
                return;
            }
        };
        let mut rng = Rng::new(0x1EAF_0011);
        let n = 64usize;

        // NORM_LOGSOFTMAX (op 3): the stable max / log-sum-exp / subtract row window, in place.
        let logsoftmax = format!(
            "module m\nfn lsm(mut x:[f32;{n}]) {{ let mut m: f32 = x[0]; \
             for i in 0..{n} {{ m = fmax(m, x[i]); }} let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + exp(x[i] - m); }} let ls: f32 = log(s); \
             for i in 0..{n} {{ x[i] = (x[i] - m) - ls; }} }}"
        );
        // NORM_L2NORM (op 4): sum of squares / rsqrt / scale, x -> out.
        let l2norm = format!(
            "module m\nfn l2n(x:[f32;{n}], mut out:[f32;{n}]) {{ let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + x[i]*x[i]; }} let inv: f32 = rsqrt(s + 0.00001); \
             for i in 0..{n} {{ out[i] = x[i]*inv; }} }}"
        );

        for (label, src, entry_name, init) in [
            (
                "log_softmax",
                logsoftmax,
                "lsm",
                vec![rng.vec(n, -3.0, 3.0)],
            ),
            (
                "l2norm",
                l2norm,
                "l2n",
                vec![rng.vec(n, -3.0, 3.0), vec![0.0; n]],
            ),
        ] {
            let (program, mut interner) = build(&src);
            let entry = interner.intern(entry_name);
            // The window must actually lower to `wukong_norm_f32`, else this tests nothing.
            let mir = wukong_mir::print::print_program(&program, &interner);
            assert!(
                mir.contains("wukong_norm_f32"),
                "{label}: did not lower to a fused norm — the gate would never be consulted"
            );
            let (cpu, gpu, calls) = run_both(g, &program, entry, &interner, &init);
            assert_eq!(
                calls, 0,
                "{label}: an unimplemented norm op must not reach the device"
            );
            let last = cpu.len() - 1;
            assert_eq!(
                gpu[last], cpu[last],
                "{label}: the CPU fallback must be bit-identical to the oracle"
            );
        }
    }
}
