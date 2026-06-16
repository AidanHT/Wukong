# mercury_mir

Mercury's typed, block-structured SSA IR: data types, an imperative `Builder`, a pretty-printer, and a verifier. Sits between `mir_build` (produces it) and `opt`/backends (consume it).

## Layout
- `src/lib.rs` — core data types: `ValueId`, `BlockId`, `MirType`, `BasicBlock`, `Function`, `Program`, `MirLevel`.
- `src/inst.rs` — instruction vocabulary: `BinOp`, `CmpOp`, `CastKind`, `Op`, `Inst`, `Terminator`.
- `src/builder.rs` — `Builder` for constructing one `Function` imperatively.
- `src/print.rs` — `print_program`/`print_function`, deterministic text (`--emit=mir`, snapshots).
- `src/verify.rs` — `verify_program`/`verify_function` well-formedness checks.

## Key types & entry points
- `Function` (`src/lib.rs`) — value arena `value_types: Vec<MirType>` indexed by `ValueId`, `blocks: Vec<BasicBlock>`, `entry`. Helpers `value_type(v)`, `block(b)`.
- `Program` (`src/lib.rs`) — `funcs` plus a `MirLevel` (`High`/`Low`); `Program::new()`/`default()` start at `Low`. `function(name)` looks up by `Symbol`. Backend only ever sees `Low`.
- `Op` / `Terminator` (`src/inst.rs`) — the instruction set. SSA merges use block params + branch args, NOT phi nodes. `Store` (never) and `Call` (may be void or value-producing) are the only ops allowed to lack a result; all others must produce one.
- `Builder` (`src/builder.rs`) — `new` auto-creates `bb0` as entry. `add_param` adds an entry-block param; `block_param(blk, ty)` adds a param to any block (a merge point). `build`/`build_void`/`push` append to the current block (set via `switch_to`); `alloca` always inserts into the entry block. `finish()` derives `Function::params` from the entry block's params.
- `verify_function` (`src/verify.rs`) — returns `Vec<String>` (empty = ok). Failures are internal compiler errors (pass bugs), not user diagnostics. Checks: uses are defined, per-op operand/result types, `Cmp` operands equal + predicate float-ness matches, branch targets exist, edge arg arity + types match target block params, and terminator/return-type agreement.

## Connects to
Upstream (depends on): `mercury_span` (`Symbol`, `Interner`), `mercury_types` (`Scalar`, via `MirType::from_scalar`). Downstream: `mercury_mir_build` constructs it; `mercury_opt` rewrites it; interpreter and LLVM backends consume `Low` MIR; `print`/`verify` invoked across the pipeline (verify runs under `--verify-each`).

## Gotchas
- `MirType` integers are signless (LLVM-style); signedness lives on the op (`SDiv`/`UDiv`, `Slt`/`Ult`, `SExt`/`ZExt`). `I1` counts as int (`is_int`).
- `MirType::Array(elem, n)` is only the *operand* type of an array `alloca`; the alloca's result is always `Ptr` to the first element. `Vec(elem, n)` is the SIMD vector type.
- `MirLevel` is a plain `Program` field — the verifier checks structural/type well-formedness but does NOT assert High-vs-Low op restrictions or even look at the level.
- `ValueId`/`BlockId` are dense `u32` indices into `value_types`/`blocks`; the verifier checks `block.id.0 == index`. Don't reorder blocks/values without keeping ids consistent.
- New blocks get `Terminator::Unreachable` until `set_term` (`ret`/`br`/`cond_br`); forgetting a real terminator leaves a dangling block.
- The verifier does NOT detect duplicate result `ValueId`s: `defined` is a `HashSet`, so a re-assigned id is silently absorbed. It assumes each `ValueId` is assigned once.
- Bin-op type rule exempts only `And`/`Or`/`Xor` from the int-result requirement (so they're allowed on `I1`); shifts (`Shl`/`LShr`/`AShr`) and the rest still require an int result.
