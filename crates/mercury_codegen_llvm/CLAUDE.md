# mercury_codegen_llvm

LLVM backend: lowers a MIR `Program` to **textual** LLVM IR. Sits at the end of the pipeline, parallel to the interpreter backend, behind the shared `Backend` trait. Gating is in the driver (`--features llvm`); the crate itself has no `[features]`.

## Layout
- `src/lib.rs` — entire crate: `Backend` impl, MIR-to-IR emitter, type/op name mappers, `#[cfg(test)]` inline tests.

## Key types & entry points
- `LlvmBackend` (`src/lib.rs`) — unit struct implementing `mercury_backend::Backend`. `name()` returns `"llvm"`. `compile` returns `Artifact::Emitted { llvm_ir: Some(..), object: None }` — IR text only, never an object file.
- `emit_llvm_ir(program, interner) -> String` — top-level lowering entry; iterates `program.funcs`, calling `emit_function` per function.
- `emit_function` (free fn) — per-function driver: pre-scans the function to build the `consts` map, writes the `define ...` signature, then runs the `Emitter` over each block.
- `Emitter<'a>` — per-function context holding `f`, `interner`, `consts`, `edges`. Methods `emit_block`/`emit_phis`/`emit_inst`/`emit_term` do the work; `operand`/`ty` are helpers.
- `edge_args(f) -> EdgeArgs` — builds `HashMap<block_u32, Vec<(pred_u32, Vec<ValueId>)>>`: per target block, the args each predecessor passes. Source for phi incoming values.
- `llvm_ty`, `bin_name`, `cmp_instr`, `cast_name`, `fmt_float` — pure free-fn mappers from MIR enums to IR strings. `cmp_instr` returns `(instr, predicate)`, e.g. `("icmp","slt")` / `("fcmp","oeq")`.

## Connects to
Upstream: `mercury_mir` (`Program`/`Function`/`BasicBlock`/`Inst`/`Op`/`Terminator`/`MirType`/`BinOp`/`CmpOp`/`CastKind`/`ValueId`), `mercury_backend` (`Backend`/`Artifact`), `mercury_span` (`Interner`/`Symbol`). Tests also use `mercury_parser`, `mercury_sema`, `mercury_mir_build`, `mercury_opt`. Downstream: the driver (`mercuryc`) when built with `--features llvm`; emitted text is meant for external `clang`/`llc` (not invoked here).

## Gotchas
- **Block-param SSA -> phi bridge.** Non-entry block params become `phi` nodes built from `edge_args`; entry-block params are the function signature args (no phi — guarded by `b.id != f.entry`). At `-O0` the front-end emits no block params, so no phis; phis only appear after `mem2reg`. Phis must lead the block — `emit_phis` runs before instructions.
- **CondBr to the same block on both arms = ONE LLVM predecessor.** `edge_args` deliberately skips the `else` edge when `else_blk == then_blk` to avoid a duplicate/conflicting phi entry.
- **Constants are inlined, not instructions.** `Op::ConstInt`/`Op::ConstFloat` are collected into `consts` (in `emit_function`) and emitted as literal operands; `emit_block`/`emit_inst` skip them. `operand()` returns the literal for const values, else `%v{id}`.
- **Cmp operands are typed by the LEFT operand**, not the result. `emit_inst` uses `self.ty(*l)` for the operand type (the result is `i1`). Keep both operands the same MIR type.
- **Label vs reference syntax.** Block labels are defined as `bb{N}:` (no `%`); all references (`br`, phi `%bb{N}` incoming) use the `%bb{N}` form. Keep them in sync.
- **Void calls.** `Op::Call` with no `result` (e.g. the `print` intrinsic) emits `call void @...`; do not type a missing result (regression-tested — was a panic on indexing `value_type` with a dummy id).
- **No native emission, no libLLVM.** Deliberately emits text rather than driving inkwell/llvm-sys (fragile on the Windows+MinGW host). `object` is always `None`; producing a binary is an out-of-process step.
- **`Neg` is type-dependent:** `fneg` for floats, `sub <ty> 0, x` for ints. `Not` is `xor <ty> x, -1`.
- **`fmt_float`** prints whole, finite floats as `N.0` (`{:.1}`) and others via `{:?}` to keep IR-valid float literals.
