# mercury_backend

The backend seam: a `Backend` trait + `Artifact` enum decoupling the driver from how fully-lowered (Low) MIR is consumed (interpreted or emitted as native code).

## Layout
- `src/lib.rs` — the entire crate: `Backend` trait and `Artifact` enum (~36 lines).

## Key types & entry points
- `Backend` (`src/lib.rs`) — trait with `name(&self) -> &'static str` and `compile(&self, program: &Program, entry: Symbol, interner: &Interner) -> Result<Artifact, String>`. The interpreter, the Cranelift native backend, and the LLVM backend all implement it, keeping the driver backend-agnostic.
- `Artifact` (`src/lib.rs`) — what `compile` returns. `Executed { exit_code: i64, stdout: Vec<u8> }` (the interpreter or the Cranelift JIT ran the program) or `Emitted { llvm_ir: Option<String>, object: Option<PathBuf> }` (LLVM emitted textual IR and/or an object/exe path).

## Connects to
Upstream (depends on): `mercury_mir` (`Program`), `mercury_span` (`Symbol`, `Interner`). Downstream (consumers): the interpreter, Cranelift native, and LLVM backends implement `Backend`, keeping them interchangeable behind one seam. In practice the `mercuryc` driver's `compile()` calls each backend's concrete entry point directly (`run_with_output` / `jit_run` / `emit_llvm_ir` / `emit_native`) rather than dispatching a `dyn Backend`.

## Gotchas
- Abstraction only — no `Backend` implementations live here; they are in other crates.
- `compile` assumes MIR already lowered to Low `MirLevel`; nothing here enforces it (optimizer/verifier's job upstream).
- Errors are plain `String`, not a structured diagnostic — keep messages human-readable.
- `entry: Symbol` is meaningful only for backends needing an entry point ("where relevant" per the doc); a pure object-emitter may ignore it.
