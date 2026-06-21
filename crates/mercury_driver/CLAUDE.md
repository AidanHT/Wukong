# mercury_driver

Orchestrates the full compile pipeline: owns the `SourceMap`/`Interner`/`DiagnosticSink`, runs each stage in order, and honors `--emit`. Sits between the CLI (`mercuryc`) and every front-end/back-end crate.

## Layout
- `src/lib.rs` — the whole crate: `Options`, `EmitStage`, `ErrorFormat`, `BackendKind`, the `exit` codes module, and `compile()` plus the private `emit_native`/`emit_mir`/`emit_diag`/`render_all`/`run_on_gpu` helpers.
- `src/gpu_accel.rs` — **(behind `--features gpu`)** `GpuAccel`, the `mercury_interp::Accelerator` impl that forwards recognized kernels to `mercury_codegen_gpu` launch wrappers — incl. `sgemm_nt_epi`, which routes a source-level `act(matmul [+ bias])` to the single fused WMMA kernel: bias-free → `gemm_nt_f16_sm_db_{relu,silu,gelu}`, with a per-column bias → `gemm_nt_f16_sm_db_bias{,_relu,_silu,_gelu}` (the canonical `nn.Linear`/FFN `act(x·Wᵀ+bias)`; ACT_IDENTITY+bias = affine Linear). relu/gelu/silu, aligned shapes (M,N %64, K %16); a bias-free identity GEMM and everything else decline to CPU. It counts device calls (`calls`) so the e2e tests can assert the offload actually fired. A GPU error becomes `Some(Err(..))` (never `None`/silent CPU fallback when the user asked for the GPU).

## Backends & the `gpu` feature
`BackendKind` is `Interp` (default, the oracle), `Native` (Cranelift JIT), or `Gpu`. `--backend=gpu` requires building with the driver's `gpu` feature (off by default → toolchain-free core; it pulls `mercury_codegen_gpu/gpu` + `cudarc`). `run_on_gpu` acquires the process-wide `Gpu`, wraps it in `GpuAccel`, and runs the program via `mercury_interp::run_with_output_accel` (the offloading interpreter — recognized GEMM/activation/reduction/norm calls run on the device, everything else on the CPU). Without the feature, `--backend=gpu` is a clear build-capability error, not a silent fallback. Gated by the `gpu_e2e_tests` module (only with `--features gpu`; skips with no device).

## Key types & entry points
- `compile(opts: &Options) -> i32` (`src/lib.rs`) — single entry point. Reads the input file, runs lexer -> parser -> sema -> mir_build -> opt -> backend, returns a process exit code. Each `--emit` stage short-circuits with its own dump.
- `Options` (`src/lib.rs`) — invocation config (`input`, `output`, `emit`, `run`, `opt_level`, `color`, `error_format`). `Default` emits `Exe` at `opt_level 0`, color on, human errors, `input` empty.
- `EmitStage` (`src/lib.rs`) — `Tokens`/`Ast`/`MirHigh`/`Mir`/`LlvmIr`/`Obj`/`Exe`. `EmitStage::parse` maps CLI strings; note `"mir" | "mir-low"` both map to `Mir`.
- `ErrorFormat` (`src/lib.rs`) — `Human` (rustc-style via `Renderer`) or `Json` (JSON Lines via `mercury_diag::to_json`).
- `exit` module — `OK=0`, `COMPILE_ERROR=1`, `UNIMPLEMENTED=2`, `IO_ERROR=3`.
- Re-exports `explain`, `all_explanations`, `Explanation` from `mercury_diag` for the CLI's `--explain`.

## Connects to
Upstream: `mercury_span` (`SourceMap`/`Interner`), `mercury_diag` (`Diagnostic`/`DiagnosticSink`/`Renderer`/`to_json`), `mercury_lexer::{tokenize,dump}`, `mercury_parser::parse_module_tokens`, `mercury_ast::print::print_module`, `mercury_sema::check`, `mercury_mir_build::lower_program`, `mercury_mir::{print,verify}`, `mercury_opt::optimize`, `mercury_backend` (`Backend` trait + `Artifact`), `mercury_interp::Interpreter`, `mercury_codegen_llvm::emit_llvm_ir`. Downstream: the `mercuryc` binary calls `compile()`.

## Gotchas
- Diagnostic plumbing is split: lexer+parser diags go into a `DiagnosticSink`, flushed via `render_all`; sema/mir_build diags are emitted one-by-one via `emit_diag` (never enter the sink). `sink.has_errors()` only reflects lex/parse — later stages re-check with `diags.iter().any(|d| d.is_error())`.
- The `Interner` is created at parse time (after lexing); the lexer runs on raw source text only.
- After `mir_build`, an `is_error` in `lower_diags` means the MIR is invalid — the driver must NOT optimize/run/back-end it (would crash/miscompile), so it stops with `COMPILE_ERROR`. Exception: `--emit=mir-high` still dumps the partial MIR for debugging before bailing.
- `--run` short-circuits BEFORE the `--emit=mir`/`llvm-ir`/native checks. It runs the optimized program via the `mercury_backend::Backend` trait on `mercury_interp::Interpreter`, writes the program's captured stdout, and returns its `exit_code`. Zero LLVM required (the interpreter is the differential oracle).
- `emit_mir` runs `mercury_mir::verify::verify_function` on every func and prints any ICE to stderr, but still prints the program — verify failures do NOT change the `--emit=mir` exit code (always `OK`).
- Native (`Obj`/`Exe`) shells out to `clang` (the only common driver that eats textual IR), writing a `<stem>.ll` sidecar first; output defaults to `<stem>.o`/`<stem>.exe` unless `--output` is set. Missing `clang` returns `UNIMPLEMENTED` (2) by design (local env has no LLVM); `clang` running but failing returns `COMPILE_ERROR` (1).
- Artifacts print to stdout; diagnostics and the `wrote <file>` note go to stderr.
- The top-of-file doc comment ("today the lexer is connected...") is stale; `compile()` wires the full pipeline.
