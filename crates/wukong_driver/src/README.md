# `wukong_driver` modules

| file | exports / role |
|---|---|
| `lib.rs` | `Options`, `EmitStage` (incl. `MirText`), `BackendKind`, `GradOptions`, `TrainOpt`, `compile()`, `verify_or_ice`, `emit_native`, `explain` re-exports |
| `loader.rs` | `load_imports` — resolve `import a.b.c` → `<root-dir>/a/b/c.wk`, depth-first, cycle-safe |
| `passes.rs` | `--passes` / `--print-after`: `FUNCTION_PASSES`, `PROGRAM_STEPS`, `parse_spec`, `standard_spec`, `run_steps` — a replica of `wukong_opt::optimize_preserving` from its public API, law-tested byte-identical to `optimize` at `-O0`..`-O3`; `compile_mir_text`, the `.mir` input path (parse, verify, optimize, print) |
| `gpu_accel.rs` | `GpuAccel` — `--backend=gpu` recognizer offload (`#[cfg(feature = "gpu")]`) |
| `gpu_e2e_bench.rs` | Test-only end-to-end GPU measurement round (`#[cfg(all(test, feature = "gpu"))]`) |
