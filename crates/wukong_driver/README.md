# wukong_driver

Pipeline orchestration for the Wukong compiler. `wukongc` compiles user `.wk` programs; this crate
sequences the language pipeline and honors the requested `--emit` target or `--run` backend. Benches
measure the compiler — they are not the product.

**Place in pipeline:** owns the `SourceMap`, `Interner`, and `DiagnosticSink`; runs each stage in
order via [`compile`](src/lib.rs). Flag *combinations* are validated in `wukongc`'s `parse_args`; this
crate owns the option *types* and stage dispatch.

## Pipeline (`compile`, `src/lib.rs`)

```
input.wk
  → lex        wukong_lexer::tokenize                    → tokens        (--emit=tokens exits)
  → parse      wukong_parser::parse_module_tokens_from   → AST           (--emit=ast exits)
  → imports    loader::load_imports                      merge a.b.c → <root-dir>/a/b/c.wk
  → sema       wukong_sema::check                        name res, types, SHAPE check
  → build      wukong_mir_build::lower_program           → MIR + kernel recognizers (--emit=mir-high exits)
  → optimize   wukong_opt::optimize(&mut program, lvl)   fixpoint SSA passes
  → autodiff   wukong_autodiff::grad                     --emit=grad / --train (force opt_level.max(1))
  → run        wukong_interp | wukong_codegen_cranelift::jit_run
               | run_on_gpu (offload) | run_on_gpu_lower (MIR→PTX)        (--emit=mir exits here)
  → verify     verify_or_ice                             before EVERY backend entry, AOT included
  → emit       wukong_codegen_llvm::emit_llvm_ir | wukong_codegen_cranelift::emit_object
               + emit_native (the --emit=exe link)
```

`--emit=tokens` / `--emit=ast` see the **root file only** — they exit before `load_imports`.

## Key files

| file | role |
|---|---|
| `src/lib.rs` | `Options`, `EmitStage`, `BackendKind`, `GradOptions`, `compile()` |
| `src/loader.rs` | Multi-file front end: resolve `import`s, cycle-safe, one merged flat namespace |
| `src/gpu_accel.rs` | `--backend=gpu` recognizer offload bridge (`--features gpu`) |
| `src/gpu_e2e_bench.rs` | Test-only GPU measurement round (`#[cfg(all(test, feature = "gpu"))]`) |

## Execution backends (`--run`)

| `--backend=` | what it is |
|---|---|
| `interp` (default) | `wukong_interp` tree-walks MIR — **the differential oracle** |
| `native` | Cranelift JIT (`jit_run`); same crate's `emit_object` behind `--emit=obj\|exe` |
| `gpu` | Interpreter tree-walks; recognized kernel calls offloaded via `gpu_accel::GpuAccel` |
| `gpu-native` | Whole-program MIR→PTX via `wukong_codegen_gpu::lower::jit_run` |

Both GPU variants need `--features gpu`. Without it they return a clear build-capability error —
never a silent CPU fallback. CPU↔GPU agreement is tolerance-gated; interp-vs-native is bit-exact.

## `verify_or_ice`

Called before **every** backend entry, including AOT (`--emit=llvm-ir`, `obj`, `exe`). A lowering bug
must surface as a diagnosable ICE, not a backend crash or miscompile. `--emit=mir` / `mir-high`
verify inside `emit_mir` after dumping (so broken MIR is still visible).

## Invariants / landmines

- **`Options::default()`:** `emit=Exe`, `run=false`, `opt_level=0`, `backend=Interp`.
- **Autodiff forces `-O1` minimum** when `--emit=grad` or `--train` (single-block SSA input).
- **Exit codes:** `0` OK, `1` COMPILE_ERROR, `2` UNIMPLEMENTED, `3` IO_ERROR.
- Artifacts print to **stdout**; diagnostics and `wrote <file>` notes go to **stderr**.

## Testing

```sh
cargo test -p wukong_driver --lib
cargo test -p wukong_driver --features gpu --lib   # GPU e2e laws (device-free subset runs without CUDA)
```

End-to-end gates live in `wukongc/tests/` (see that crate's README).
