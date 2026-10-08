# `wukong_opt` source layout

| Module | Role |
| --- | --- |
| `lib.rs` | `Pass` trait, `PassManager`, `optimize`/`optimize_timed`, shared use-visiting helpers, integration tests |
| `cfg.rs` | `successors`, `predecessors`, `reverse_postorder`, `reachable`, `prune_unreachable` |
| `dom.rs` | Cooper–Harvey–Kennedy dominators: `idoms_from`, `dominance_frontiers_from`, `dom_children` |
| `cache.rs` | `CfgAnalyses` — lazily computed, structurally-invalidated per-function analysis cache |
| `sroa.rs` | `Sroa` — split a constant-indexed aggregate slot into scalar leaf slots for mem2reg (`-O2`) |
| `mem2reg.rs` | `Mem2Reg` — scalar alloca/load/store → block-parameter SSA |
| `simplify.rs` | `Simplify` — constant folding + algebraic identities |
| `simplify_cfg.rs` | `SimplifyCfg` — constant-branch folding, straight-line merge, prune |
| `phi.rs` | `SimplifyPhis` — drop dead/trivial block parameters and edge args |
| `dce.rs` | `Dce` — dependency worklist removes complete dead instruction chains |
| `cse.rs` | `Cse` — typed dominator-tree value numbering, integer commutation, load forwarding across single-predecessor edges |
| `dse.rs` | `Dse` — dead-store elimination: must-alias overwrites in a block and on every path, end-of-lifetime stores, loop-bounded stores into unread slots |
| `loop_canon.rs` | `LoopCanon` — canonical loop shape (preheader, latch, exit polarity) |
| `loop_delete.rs` | `LoopDelete` — remove a provably finite loop with no observable effect |
| `licm.rs` | `Licm` — hoist loop-invariant non-trapping ops into an existing preheader; loads by dereferenceability, header position, or guaranteed execution under a first-iteration guard |
| `bce.rs` | `Bce` — slice bounds-check elimination: versions an innermost counted loop on a preheader guard so a check-free copy runs when every check is proved |
| `promote.rs` | `Promote` — keep a loop-invariant location the loop reads and writes in a register (preheader load, exit stores) |
| `vectorize.rs` | `Vectorize` — 128-bit SIMD widening with scalar epilogue; constant `i16`/`i32` left shifts use wrapping multiply |
| `inline.rs` | `inline_program` — bottom-up call-graph SCC inlining with per-site growth budgets |
| `unroll.rs` | `unroll_program` — complete (small known trip count) or partial loop unrolling at `-O2`+ |
| `alias.rs` | `AliasInfo`, `Prov` — provenance for load/store disambiguation, plus the `@noalias` parameter facts (`promised_storage`, `is_invariant_slice_header`, `promised_disjoint_bases`) |
| `effects.rs` | `CallEffects`, `CallGraph` — interprocedural call-effect summaries (side-effect free, reads memory, always returns, speculatable) built callees-first for `cse`/`dce`/`licm` |
| `loop_info.rs` | Natural-loop analysis (induction vars, trip counts, dependences) |
| `fxhash.rs` | `FxHashMap`/`FxHashSet` — crate-internal fixed-seed hasher |
| `branch_fold.rs` | jump threading of constant edge arguments and folding a branch into a predecessor with the same destination (`and`/`or`), both run inside `SimplifyCfg` |
| `tre.rs` | `Tre` — tail-recursion elimination (plain and integer-accumulator), plus `pre_inline` so a fully tail-recursive function becomes inlinable |
| `fuse.rs` | `LoopFuse` — fusion of adjacent innermost counted loops over the same range, with an affine/alias legality proof |
| `unswitch.rs` | `LoopUnswitch` — induction-variable range splitting and non-trivial unswitching of innermost loops |
| `switch_form.rs` | `FormSwitches` / `form_switches` — late `switch` formation from equality chains; constant/degenerate switch folding |

## Cross-cutting helpers in `lib.rs`

The four use-visiting helpers (`map_op_uses`, `each_op_use`, `map_term_uses`, `each_term_use`) and
`has_side_effects` are the single source of truth for which operands each `Op`/`Terminator` reads. A
new `Op` variant must update all of them plus `has_side_effects` (DCE and mem2reg key on it). A new
cacheable op also needs a `Key` variant in `cse.rs` and a `safe_to_hoist` classification in
`licm.rs`.
