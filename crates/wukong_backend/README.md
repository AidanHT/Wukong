# wukong_backend

The seam between MIR and everything that consumes it: one trait, one enum, ~50 lines in a single
file. Wukong is the language; user `.wk` programs compile to MIR elsewhere — this crate declares the
**shape** of "a thing that executes or emits a `wukong_mir::Program`" so no consumer grows its own
bespoke entry signature.

There is no logic here and no production `Backend` implementation. Implementations live in the
codegen and interp crates.

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

## LANDMINE — declared seam, not live dispatch (and that is intentional)

**Nothing in the workspace calls `Backend::compile`.** `wukong_driver::compile` matches on its own
`BackendKind` (`Interp | Native | Gpu | GpuLower`) and calls concrete functions directly —
`wukong_interp::run_with_output`, `wukong_codegen_cranelift::jit_run`, `run_on_gpu`,
`run_on_gpu_lower` — with `--emit=llvm-ir`/`obj`/`exe` handled in their own arms.

This is a decision, not an oversight waiting to be wired up. The driver needs per-backend arguments
the trait does not carry (the `gpu` offload accelerator, the `--emit=exe` link step, the object path,
`verify_or_ice` before each entry), and routing it through `&dyn Backend` would add an indirection
without removing any of that. The trait's job is narrower: to fix the *shape* of "a thing that runs or
emits a `Program`" so the four adapters cannot drift into four signatures, and so a test can hold the
two `Executed` producers to one comparison.

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

The crate's own contract tests are `src/lib.rs` `mod tests` (P7.25), five of them:

| test | the contract it pins |
| --- | --- |
| `compile_receives_the_program_entry_and_interner_it_was_given` | `compile` forwards `program`, `entry` and `interner` unchanged |
| `executing_backend_carries_exit_code_and_stdout` | an executing backend returns `Executed` with both fields intact |
| `emitting_backend_yields_emitted_not_executed` | an emitting backend returns `Emitted`, never `Executed` |
| `backend_is_object_safe_and_name_dispatches` | the trait is object-safe and `name` dispatches through `&dyn Backend` |
| `a_declined_compile_is_a_string_error` | a declined compile is a plain `String` error (the `UNSUPPORTED` path) |

They run against a recording test double, **not** against the four production adapters, and they
cannot: `wukong_interp`, `wukong_codegen_cranelift`, `wukong_codegen_llvm` and `wukong_codegen_gpu`
all depend on this crate, so naming any of their types here would be a dependency cycle. The real
adapters are exercised where they live (`LlvmBackend.compile` by
`crates/wukong_codegen_llvm/src/closeout_tests.rs`), and the two `Executed` producers' agreement is the
corpus differential `native_matches_interpreter`, which calls the crates' free functions directly:

```sh
cargo test -p wukong_backend                 # the five contract tests
cargo test -p wukongc --test run             # interp vs native, bit for bit, over tests/run
cargo check --features gpu --all-targets     # GpuLowerBackend still type-checks
```

See [`src/README.md`](src/README.md).
