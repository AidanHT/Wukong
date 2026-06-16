# mercury_mir_build

Lowers the type-checked Mercury AST into MIR. Sits between `mercury_sema` and the optimizer/backend.

## Layout
- `src/lib.rs` — the entire crate: `lower_program` entry, the `FnLowerer` walker, the free type/op-mapping helpers, and inline `#[cfg(test)]` tests (no `tests/` dir).

## Key types & entry points
- `lower_program(module, sema, interner) -> (Program, Vec<Diagnostic>)` — top-level entry; lowers only `Fn` items that have a body, returns a `mercury_mir::Program` plus diagnostics.
- `lower_fn` — per-function setup: recovers the resolved signature from `sema.defs` (`DefKind::Fn`; falls back to `Ty::Unknown` params / `Ty::Unit` ret if absent), declares all params via `Builder::add_param` first, materializes each, lowers the body, and emits a fallback terminator if it fell through (`ret(None)` for Void, `ret(Some(tail))` if there is a tail value, else `Terminator::Unreachable`).
- `FnLowerer` — the stateful lowerer. Holds the `mercury_mir::Builder`, `sema`/`interner` refs, `diags`, a scope stack (`Vec<HashMap<Symbol, (ValueId, MirType)>>`), a `terminated` flag, and a `loops` stack of `(continue_target, break_target)` `BlockId`s.
- Strategy is **alloca-per-local** (clang-style): every `let`/param gets a stack slot; reads `Load`, writes `Store`. SSA is recovered later by the `mem2reg` opt pass — this crate never reasons about SSA/phi.
- Free helpers: `mir_ty` (`Ty` -> `MirType`), `mir_ty_of_ast` (used when a `let` has an explicit type annotation), `const_usize_expr` (literal array length), `param_abi_ty`, `arith_binop`/`compound_binop`/`cmp_pred`/`cast_kind` (AST op -> MIR op, float/signed-aware), `is_intrinsic` (`pub`), `parse_int`/`parse_float`.

## Connects to
Upstream: `mercury_ast` (input), `mercury_sema` (`SemaResult`: `sema.types` by `Expr.id`, `sema.defs`), `mercury_types` (`Ty`), `mercury_span` (`Interner`, `Symbol`, `Span`), `mercury_diag`. Builds on `mercury_mir` (`Builder`, `Op`, `MirType`, `BinOp`/`CmpOp`/`CastKind`, `Program`). Downstream: the optimizer (esp. `mem2reg`) and backends consume the `Program`. Tests use `mercury_parser` (dev-dep).

## Gotchas
- **Arrays are not alloca'd as values.** An array local's bound `ValueId` *is* its base pointer: a `Path` read of an array returns the slot directly (no `Load`); indexing `Gep`s off it. Array params get ABI type `Ptr` (`param_abi_ty`) and are bound directly with no slot copy. See `array_parameter_passes_by_pointer` test.
- Unsupported constructs (tensor ops, SIMD/method calls, `for` over non-range, `defer`, unknown calls/places) are a **hard error** via `unsupported` (code `C0001`, message "... not yet supported by codegen") but lowering continues with a placeholder (`const_zero` / fresh alloca) so the rest of the function still lowers. The front end accepts these; only codegen rejects them.
- `terminated` must be reset to `false` after every `switch_to` to a fresh block, or subsequent statements get skipped. Every CFG helper (`lower_while`/`lower_loop`/`lower_for`/`lower_if`/`lower_fill_loop`) does this manually — preserve the pattern when adding new CFG lowering.
- `break`/`continue` read `self.loops.last()` and set `terminated = true`; they only branch when inside a loop (silently no-op otherwise).
- Only single-segment paths (`Path::is_single`) and single-index `Index` (`indices.len() == 1`) are handled in `lower_place`/`lower_expr`; multi-index indexing falls through to `unsupported`.
- Op selection is type-driven from sema: float vs int and signedness come from `expr_ty`/`signed` (looked up in `sema.types` by `Expr.id`); a missing type defaults to `Ty::Unknown` / `MirType::I32`.
- `let` element type for arrays prefers the annotation (`mir_ty_of_ast`) over the init expr; `mir_ty_of_ast` falls back to `Ptr` for non-literal array lengths and `I32` for unrecognized scalar names.
- Array-repeat `[v; n]` evaluates `v` once; unrolls to straight-line stores when `n <= REPEAT_UNROLL_LIMIT` (8), else emits a `lower_fill_loop` CFG loop (counter typed `I64`).
- Bool literals lower to `I1` `ConstInt`. Void/unit calls (user fns and intrinsics) emit a void `Call` then return a dummy `const_zero(I32)` so the expression still yields a `ValueId`.
- Int/float literals coerce their MIR type to `I32`/`F32` if the sema type is not int/float.
