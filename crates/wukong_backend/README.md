# wukong_backend

The seam between MIR and everything that consumes it: one trait, one enum, ~50 lines in a single
file. Wukong is the language; user `.wk` programs compile to MIR elsewhere — this crate declares the
**shape** of "a thing that executes or emits a `wukong_mir::Program`" so no consumer grows its own
bespoke entry signature.

There is no logic here and no `Backend` implementation. Implementations live in the codegen and
interp crates.

Dependencies: `wukong_mir` (`Program`) and `wukong_span` (`Symbol`, `Interner`). Nothing else, ever —
every backend crate depends on this one.

## The two items

```rust
pub trait Backend {
    fn name(&self) -> &'static str;
    fn compile(&self, program: &Program, entry: Symbol, interner: &Interner)
        -> Result<Artifact, String>;
}

pub enum Artifact {
    Executed { exit_code: i64, stdout: Vec<u8> },
    Emitted  { llvm_ir: Option<String>, object: Option<PathBuf> },
}
```

- `Artifact::Executed` — the program ran: exit code plus captured stdout. Three implementors
  produce this shape, which is why the differential gate can treat them interchangeably.
- `Artifact::Emitted` — native artifacts: textual IR and/or an object/executable path.
- Errors are plain `String`, not `wukong_diag` diagnostics.

## Who implements it

| Type | Crate | Returns |
| --- | --- | --- |
| `Interpreter` | `wukong_interp` | `Executed` (via `run_with_output`) |
| `CraneliftBackend` | `wukong_codegen_cranelift` | `Executed` (via `jit_run`) |
| `GpuLowerBackend` | `wukong_codegen_gpu` (feature `gpu`) | `Executed` (via its own `jit_run`) |
| `LlvmBackend` | `wukong_codegen_llvm` | `Emitted` (via `emit_llvm_ir`) |

Each `compile` is a thin adapter over the crate's real free function. Put the work in a plain `pub fn`
and let the trait impl be three lines.

## LANDMINE — declared seam, not live dispatch

**Nothing in the workspace calls `Backend::compile`.** `wukong_driver::compile` matches on its own
`BackendKind` (`Interp | Native | Gpu | GpuLower`) and calls concrete functions directly —
`wukong_interp::run_with_output`, `wukong_codegen_cranelift::jit_run`, `run_on_gpu`,
`run_on_gpu_lower` — with `--emit=llvm-ir`/`obj`/`exe` handled in their own arms.

- `BackendKind::Gpu` (`--backend=gpu`, the offloading interpreter) has **no** `Backend` impl.
- Adding an `Artifact` variant or trait method does not reach the CLI. A new target needs a
  `BackendKind` variant, its dispatch arm, and the `--backend=` parse in `wukongc`.
- Adding a `Backend` impl costs nothing at the CLI and buys uniformity for tests — but document
  `Artifact`, not "the driver dispatches this".

## Invariants

- `compile` receives fully-lowered MIR. The guarantee comes from `wukong_driver::verify_or_ice`,
  which runs `wukong_mir::verify::verify_function` over every function before every backend entry.
- The two `Executed` producers (interp and native) must agree **bit-for-bit** on stdout and exit
  code for any program in `tests/run`. GPU-native is held to a CPU↔GPU *tolerance* gate instead.
- `compile` takes `&self`; backends are unit structs with no state.

## Testing

No tests in this crate (two type definitions). Covered transitively:

```sh
cargo check -p wukong_backend
cargo test -p wukongc --test run
cargo check --features gpu --all-targets
```

See [`src/README.md`](src/README.md).
