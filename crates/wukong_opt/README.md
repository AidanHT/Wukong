# wukong_opt

MIR optimizer for Wukong (`.wk`) programs. Wukong is the language; this crate transforms MIR
between `wukong_mir_build` and every execution or emission backend. It must be **observationally
transparent** — the same program at `-O0` and `-O3` produces identical stdout and exit code on every
backend.

Depends only on `wukong_mir` and `wukong_span`. Dev-dependencies (`wukong_parser`, `wukong_sema`,
`wukong_mir_build`, `wukong_interp`) exist solely for source→run integration tests in `lib.rs`.

## Entry points

- `optimize(&mut Program, opt_level: u8)` — the production path. `-O0` is a no-op.
- `optimize_timed(&mut Program, u8) -> Timings` — identical mutations with per-pass timing; for
  `wukong_bench` only. Must stay byte-identical to `optimize`.
- `PassManager::standard(opt_level)` — builds the per-function pass list.

## Optimization levels

The only level tests are `>= 1` and `>= 2`, so **`-O3` is byte-identical to `-O2`**. Do not add
`-O3`-only passes without revisiting the differential gates.

| Level | What runs |
| --- | --- |
| `-O0` | nothing |
| `-O1` | per-function fixpoint pipeline (below) |
| `-O2` | `inline_program` first, then the pipeline, then `unroll_program` and straight-line `slp_program`, each with a reduced cleanup sweep over the functions it changed |
| `-O3` | same as `-O2` |

`wukong_driver` forces `opt_level.max(1)` for `--emit=grad` and `--train` because
autodiff consumes SSA MIR.

## Per-function pass order (`PassManager::standard`)

The list runs to a fixpoint (stopping after 101 sweeps at the latest) with per-pass clean-tracking. Order is
load-bearing.

**`-O1`:** mem2reg → simplify → simplify-cfg → simplify-phis → dce

**`-O2` appends:** cse → dse → loop-canon → licm → vectorize

- **mem2reg** — scalar `alloca`/`load`/`store` → block-parameter SSA (Cytron), including pointer
  slots whose initialization is proven. First because later passes work directly on SSA values.
- **simplify** — constant folding and algebraic identities.
- **simplify-cfg** — constant-branch folding, straight-line merge, prune unreachable blocks. Like
  every pass that changes CFG edges or blocks, it must invalidate the analysis cache.
- **simplify-phis** — drop dead or trivial block parameters mem2reg introduced.
- **dce** — retain the dependency closure of effectful instructions and terminators, removing
  complete dead instruction chains in one linear worklist traversal.
- **cse** — dominator-tree value numbering plus load forwarding within blocks and across
  single-predecessor edges; joins and loop headers discard incoming memory facts. Keys recognize
  commuted integer arithmetic and distinguish vector splat widths; float operand order is retained.
- **dse** — eliminate overwritten stores within a block and in-bounds stores to unread,
  unescaped private stack slots across the function. Empty initialization loops are not deleted.
- **loop-canon** — one shape per loop: preheader, single latch, one exit-test polarity. LICM hoists
  only into an existing preheader.
- **licm** — hoist loop-invariant non-trapping ops into that preheader.
- **vectorize** — widen a canonical loop body to 128-bit SIMD (if-converting conditionals to lane
  masks), keeping the original loop as scalar epilogue. Constant integer left shifts use wrapping
  vector multiplication at `i16`/`i32` widths; other shifts retain scalar lowering.

**Whole-program stages at `-O2`+ (not `Pass` traits):**

- `inline_program` — runs *before* the per-function pipeline, bottom-up over call-graph strongly
  connected components. Recursive callees are excluded; per-site budgets account for sole callers,
  loops and constant arguments, with absolute growth, relative growth and splice-count limits.
- `unroll_program` — runs *after* the pipeline; changed functions get a reduced cleanup pass
  (`simplify` → `cse` → `dce`), folding the synthesized constant-bound guard as well as the body.
- `slp_program` — runs *after* unrolling (so a complete unroll's copies are visible with their
  counters folded) and before switch formation: `W` stores to consecutive elements of one object
  whose values share one lane-wise operation tree become one 128-bit vector store over vector
  loads, splats and lane builds. It moves the chunk's stores and loads down to its last store, so
  every access they pass must be proved disjoint (`AliasInfo::may_alias_sized`), a call must find
  them call-proof, an `atomic.add` must find them call-proof and disjoint from its location, and a
  barrier declines. Packed functions get the same reduced cleanup.
  `WUKONG_NO_SLP` turns it off; `WUKONG_SLP_TRACE=1` prints each decision. A program autodiff will
  differentiate is optimized with `OptimizeOptions::AUTODIFF_INPUT` (`optimize_preserving_with` /
  `optimize_verified_preserving_with`), which skips it: autodiff has no VJP rule for a vector store.

`cse`, `dse`, and `licm` consult `alias` — provenance analysis answering whether a store through `q`
can be seen by a load through `p`. The Cranelift backend uses it too.

## Analysis cache (`CfgAnalyses`)

`preds`, `rpo`, `idom`, `df`, `children` depend only on the block set and terminator successor
edges. One cache per function is threaded through every pass and fixpoint iteration. **If a pass
changes the block set or any successor edge, it must call `cache.invalidate()`** — stale dominators
silently produce wrong CSE/LICM/mem2reg results the verifier will not catch. `simplify-cfg`, the
one pass that restructures the CFG, invalidates *after each* fold, merge and prune so that it can
itself read predecessors and reachability from the cache in the steady state; `simplify-phis`
reads the predecessor slice in place (it borrows the cache, not the function). Both are measured by
`wukong-bench compile-work`, which counts allocations per pass and per call.

Under `debug_assertions`, `wukong_mir::verify::verify_function` runs after every pass that executed
and panics naming the culprit through the legacy API. `optimize_verified` and
`PassManager::run_verified` instead return a structured `VerificationFailure`, in debug and release
alike. The CLI exposes this as `--verify-each`; pass failures stop compilation with a human or JSON
internal-error diagnostic. Input/output and inlining/unrolling boundaries also check whole-program
callee signatures, including at O0. Per-pass checks cover individual functions, not external runtime
signatures or semantic correctness. A panic inside a transform itself is not intercepted.

## FxHash (`src/fxhash.rs`)

The optimizer is HashMap-bound. Keys are small integers drawn from the program (`ValueId`,
`BlockId`, `Symbol`) — SipHash's collision resistance buys nothing. `fxhash.rs` is a dependency-free
rotate-xor-multiply hasher with a **fixed seed** (`0x517cc1b727220a95`), so iteration order is
reproducible run-to-run.

**Do not casually unify this with `wukong_span::fxhash`.** Never let `FxHash*` iteration order reach
emitted IR; where order matters, sort explicitly (LICM sorts loops by header id).

## Testing

```sh
cargo test -p wukong_opt --lib
```

Source integration tests follow parse → sema → mir_build → optimize → interp, asserting clean
`verify_function` and comparing levels 0/1/2/3. Individual modules also contain structural tests.

The September 2026 CPU optimization work did not execute tests or benchmarks. Its changes,
limitations and deferred validation commands are recorded in
[`docs/cpu-optimizer-review-2026-09.md`](../../docs/cpu-optimizer-review-2026-09.md).

The real gate for any change here:

```sh
cargo test -p wukongc --test run
```

`optimization_is_observationally_invariant`, `native_matches_interpreter`, and
`native_optimization_is_observationally_invariant` over all `tests/run` fixtures.

## Layout

See [`src/README.md`](src/README.md) for the module map. Development tools in
[`examples/`](examples/README.md) dump or census loop shapes.
