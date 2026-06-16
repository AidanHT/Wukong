# mercury_opt

The optimizer: a function-level pass manager, CFG/dominator analyses, and the MIR transforms that run between `mir_build` and the backend.

## Layout
- `src/lib.rs` — `Pass` trait, `PassManager`, `optimize()` entry, and the shared use-visiting helpers (`map_op_uses`/`each_op_use`/`map_term_uses`/`each_term_use`/`has_side_effects`). Source->run integration tests live here.
- `src/cfg.rs` — CFG analyses: `successors`, `predecessors`, `reverse_postorder`, `reachable`, `prune_unreachable`.
- `src/dom.rs` — dominator analysis (Cooper–Harvey–Kennedy): `idoms`, `dominance_frontiers`, `dom_children`.
- `src/mem2reg.rs` — `Mem2Reg`: promote scalar `alloca`/`load`/`store` to block-parameter SSA.
- `src/simplify.rs` — `Simplify`: constant folding + algebraic identities.
- `src/simplify_cfg.rs` — `SimplifyCfg`: constant-branch folding, straight-line block merging, unreachable pruning.
- `src/phi.rs` — `SimplifyPhis`: drop dead / trivial block parameters.
- `src/cse.rs` — `Cse`: dominator-tree value numbering + intra-block load forwarding.
- `src/dse.rs` — `Dse`: intra-block dead-store elimination.
- `src/dce.rs` — `Dce`: remove unused pure instructions and unused allocas.
- `src/licm.rs` — `Licm`: hoist loop-invariant, non-trapping ops into an existing preheader.
- `src/inline.rs` — `inline_program`: whole-program inlining of small leaf functions.

## Key types & entry points
- `optimize(program, opt_level)` (`src/lib.rs`) — top-level entry. At `-O2`+ runs `inline_program` (whole-program) first, then the function-level pipeline. `-O0` does nothing.
- `Pass` trait (`src/lib.rs`) — `name(&self) -> &'static str` + `run_function(&self, &mut Function) -> bool` (returns "changed"). Passes are zero-sized unit structs; every transform except inlining implements it.
- `PassManager::standard(opt_level)` / `run` (`src/lib.rs`) — builds and runs the pipeline. Per-function fixpoint loop, ~100 iterations max (`iterations > 100`). `-O1`: mem2reg, simplify, simplify-cfg, simplify-phis, dce. `-O2` adds cse, dse, licm.
- `inline_program(program)` (`src/inline.rs`) — the only program-level transform; a free function, not a `Pass`.

## Connects to
Upstream: operates on `mercury_mir` `Function`/`Program` (`Op`, `Terminator`, `BasicBlock`, `BlockId`, `ValueId`, `MirType`); `mercury_span::Symbol` for function names. Consumes the output of `mercury_mir_build`. Downstream: the optimized `Program` goes to a backend (interpreter or LLVM). Dev-deps (`mercury_parser`/`sema`/`mir_build`/`interp`) are only for the source->run integration tests in `lib.rs`.

## Gotchas
- Pipeline order is load-bearing: mem2reg runs first because every value-based pass is far more effective on SSA than on memory traffic. Don't reorder casually.
- The use-visiting helpers in `lib.rs` are the single source of truth for which operands each `Op`/`Terminator` reads. Adding an `Op` variant means updating all four (`map_op_uses`, `each_op_use`, `map_term_uses`, `each_term_use`) and `has_side_effects` — otherwise passes silently miss uses and corrupt SSA. `has_side_effects` is the `Store`/`Call` allowlist DCE keys on.
- `prune_unreachable` must run before any dominance use; `mem2reg`, `cse`, and `licm` call it first. Dominance (`dom.rs`) is only defined on blocks reachable from entry — `idoms` assumes the input is fully reachable.
- Under `debug_assertions` (tests/CI), `PassManager::run_function` calls `mercury_mir::verify::verify_function` after every pass and panics naming the culprit. Compiled out of release.
- mem2reg only promotes scalar int/float slots (`is_promotable_ty`); pointer/array/vector allocas stay in memory (avoids synthesizing typed "undef"). A slot is promotable only if its pointer is used *solely* as the address of `load`/`store` — any `gep`/call/stored-pointer/terminator use disqualifies it. Read-before-write slots get a zero constant in the entry block (one cached const *per type*, matching interpreter zero-init memory).
- mem2reg renames in three phases: a non-mutating dominator-tree walk (`Rename::visit`) collects edits (`replace`/`delete`/`append_*` maps), then phase 3 applies them; deletes are keyed by *original* `(block, inst index)`.
- CSE's pure-op equality is a `format!` string key (`pure_key`) with operands resolved through prior rewrites. New cacheable `Op` variants must be added there or CSE ignores them.
- CSE load forwarding and DSE are both intra-block only and conservatively clear all slot tracking on a store/load through an unknown (non-alloca) pointer or any `Call`. Cross-block memory is left to the LLVM backend.
- `simplify` keeps integer self-comparisons constant (`fold_cmp_self`) but never folds float self-comparison (`NaN != NaN`); `fold_int` returns `None` (skips folding) on division/remainder by zero rather than trapping; `mask` treats `i1` as unsigned low bit.
- LICM never synthesizes a preheader — it only hoists into a loop that already has one (single out-of-loop predecessor branching unconditionally to and dominating the header). `safe_to_hoist` excludes loads/stores/calls/allocas and integer div/rem (trapping).
- Inlining is leaf-only (callee calls no *user* function; intrinsic calls are fine) and size-capped (`SIZE_LIMIT = 40`, per-caller `GROWTH_LIMIT = 5000`). Leaf-only is what makes recursion/runaway inlining impossible — no cycle detection exists (a redundant self-call guard remains anyway). Inlined-and-now-uncalled callees are dropped; `main` survives because it is never a call target.
