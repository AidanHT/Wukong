# `wukong_opt` source layout

| Module | Role |
| --- | --- |
| `lib.rs` | `Pass` trait, `PassManager`, `optimize`/`optimize_timed`, shared use-visiting helpers, integration tests |
| `cfg.rs` | `successors`, `predecessors`, `reverse_postorder`, `reachable`, `prune_unreachable` |
| `dom.rs` | Cooper–Harvey–Kennedy dominators: `idoms_from`, `dominance_frontiers_from`, `dom_children` |
| `cache.rs` | `CfgAnalyses` — lazily computed, structurally-invalidated per-function analysis cache |
| `mem2reg.rs` | `Mem2Reg` — scalar alloca/load/store → block-parameter SSA |
| `simplify.rs` | `Simplify` — constant folding + algebraic identities |
| `simplify_cfg.rs` | `SimplifyCfg` — constant-branch folding, straight-line merge, prune |
| `phi.rs` | `SimplifyPhis` — drop dead/trivial block parameters and edge args |
| `dce.rs` | `Dce` — dependency worklist removes complete dead instruction chains |
| `cse.rs` | `Cse` — typed dominator-tree value numbering, integer commutation, load forwarding across single-predecessor edges |
| `dse.rs` | `Dse` — intra-block dead-store elimination |
| `loop_canon.rs` | `LoopCanon` — canonical loop shape (preheader, latch, exit polarity) |
| `licm.rs` | `Licm` — hoist loop-invariant non-trapping ops into an existing preheader |
| `vectorize.rs` | `Vectorize` — 128-bit SIMD widening with scalar epilogue; constant `i16`/`i32` left shifts use wrapping multiply |
| `inline.rs` | `inline_program` — bottom-up call-graph SCC inlining with per-site growth budgets |
| `unroll.rs` | `unroll_program` — partial loop unrolling at `-O2`+ |
| `alias.rs` | `AliasInfo`, `Prov` — provenance for load/store disambiguation, plus the `@noalias` parameter facts (`promised_storage`, `is_invariant_slice_header`, `promised_disjoint_bases`) |
| `loop_info.rs` | Natural-loop analysis (induction vars, trip counts, dependences) |
| `fxhash.rs` | `FxHashMap`/`FxHashSet` — crate-internal fixed-seed hasher |

## Cross-cutting helpers in `lib.rs`

The four use-visiting helpers (`map_op_uses`, `each_op_use`, `map_term_uses`, `each_term_use`) and
`has_side_effects` are the single source of truth for which operands each `Op`/`Terminator` reads. A
new `Op` variant must update all of them plus `has_side_effects` (DCE and mem2reg key on it). A new
cacheable op also needs a `Key` variant in `cse.rs` and a `safe_to_hoist` classification in
`licm.rs`.
