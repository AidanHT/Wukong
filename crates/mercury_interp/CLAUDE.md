# mercury_interp

Zero-dependency tree-walking MIR interpreter: the always-available backend and the oracle differential tests compare LLVM output against.

## Layout
- `src/lib.rs` — entire crate. `Value`, the `Interpreter` `Backend` impl, the `Interp` evaluator, free helpers (`reg`, `pass_args`, `ptr`, `default_value`, `int_bits`, `mask`, `apply_bin`, `apply_cmp`, `apply_cast`), and the `#[cfg(test)]` module.

## Key types & entry points
- `run` / `run_with_output` (`src/lib.rs`) — public entry points. Take `&Program`, an entry `Symbol`, and `&Interner`. `run` returns the entry's result as an `i64` exit code; `run_with_output` also returns captured stdout bytes.
- `Interpreter` — the `mercury_backend::Backend` impl; `compile` just calls `run_with_output` and wraps it in `Artifact::Executed { exit_code, stdout }`.
- `Value` — runtime value: `Int(i128)` (width-agnostic, masked per result type), `Float(f64)`, `Ptr(usize)` (index into flat `memory`), `Unit`.
- `Interp<'a>` — per-run state: `program`, `interner`, flat `memory: Vec<Value>`, `stdout`, a `frames` pool of recycled register files, and a `scratch` buffer for block-arg passing.
- `exec` (`src/lib.rs`) — the CFG loop: runs each block's insts, masks int results to their declared width, then dispatches on `Terminator`. `run_function` sets up/recycles a register file; `eval` evaluates one `Op`.

## Connects to
Upstream (runtime deps): `mercury_mir` (`Program`/`Function`/`BasicBlock`/`Op`/`Terminator`/`MirType`/enums), `mercury_span` (`Symbol`/`Interner`), `mercury_backend` (`Backend`/`Artifact`). This crate only executes MIR; lowering lives in `mercury_mir_build` (a dev-dependency, used only by the test module). Downstream: the `mercuryc` driver (default backend) and `mercury_bench`; the differential oracle for the LLVM backend.

## Gotchas
- Memory is a flat `Vec<Value>` with no free/reuse: `Alloca` only ever grows it (an `Array(elem, count)` reserves `count` contiguous slots). Pointers are raw indices; `Gep` is `base + index`. No teardown between calls — fine for short test/bench runs, not a real allocator.
- Two distinct "zero" defaults: register files are sized to `func.value_types.len()` and init to `Value::Unit` (the "undefined"); fresh `memory` slots use typed zeros via `default_value` (`Int(0)`/`Float(0.0)`/`Ptr(0)`). No def-before-use checking; correctness relies on well-formed SSA.
- Int results are normalized via `mask` in `exec` after each inst (`i1` forced to low bit, so `!true == 0`; wider widths sign-extend). `eval` itself only masks `ConstInt`/`Cast` — `Bin`/`Cmp` results are masked by `exec`, not `eval`.
- Div/rem by zero returns `0` (no trap) for SDiv/UDiv/SRem/URem; keep in sync with the LLVM backend or differential tests diverge.
- Only two intrinsics, dispatched by resolved name when `program.function(func)` is `None`: `print`/`println` (append `"{value}\n"` to stdout) and `assert` (returns `Err("assertion failed")` on falsy). Any other unknown callee errors.
- `Terminator::Unreachable` returns an error rather than UB. Infinite-loop guard: a hard step limit of `100_000_000` in `exec` errors instead of hanging.
- `Value::as_int`/`as_float` coerce loosely (`Ptr` -> int, `Unit` -> 0, int <-> float); relies on MIR being type-correct rather than enforcing it.
