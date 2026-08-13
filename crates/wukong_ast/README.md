# wukong_ast

The Wukong abstract syntax tree: a **data-only** crate. `wukong_parser` builds it, `wukong_sema`
reads it and annotates through `NodeId`-keyed side tables, `wukong_mir_build` lowers it. Its only
dependency is `wukong_span` (`Span`, `Symbol`, `Interner`) — no types, no diagnostics, no analysis.

**Place in pipeline:** parser output; sema and mir_build consume it. `--emit=ast` pretty-prints via
`print::print_module`.

## Key files

| file | role |
|---|---|
| `src/lib.rs:3-5` | **sema never rewrites the tree** — side tables keyed by `NodeId` |
| `src/lib.rs:23-25` | `NodeId`, `Ident`, `Path`, `Module` |
| `src/expr.rs:158` | `MAX_CONST_ARRAY_LEN` |
| `src/expr.rs:196` | **`BinOp::fold_const_len`** — shared const-array-length folder |
| `src/expr.rs:267-326` | 5 tests encoding fold invariants |
| `src/print.rs` | deterministic pretty-printer for `--emit=ast` |

## The side-table contract

- **Sema must not rewrite the tree.** Results are stored keyed by `NodeId`; the AST the parser
  produced is the AST mir_build lowers.
- `NodeId`s are unique **per parse run only** — multi-file programs thread the watermark through
  `parse_module_tokens_from` (see `wukong_parser`).
- `NodeId::DUMMY` is `u32::MAX` for synthesized nodes.

## LANDMINE: `BinOp::fold_const_len`

Const array lengths (`[i32; 2+2]`) are evaluated **twice**:

| side | function | width |
|---|---|---|
| sema | `eval_usize_depth` → `u64` (`wukong_sema/src/lib.rs:1025`) | folds in `u64` |
| mir_build | `const_usize_depth` → `Option<u32>` (`wukong_mir_build/src/lib.rs:19216`) | narrows to MIR slot width `u32` |

Both route arithmetic through `BinOp::fold_const_len(self, l: u64, r: u64) -> u64`
(`expr.rs:196`) so they cannot disagree. **The operand bound is load-bearing, not just the result
bound** — operands above `MAX_CONST_ARRAY_LEN` (`u32::MAX as u64`) must fold to `0` (invalid-length
sentinel) because mir_build truncates `ExprKind::Int` leaves to `u32` before widening back.

Rules: never change one evaluator without the other; `0` is the single "unrepresentable" sentinel;
division/shift-by-zero and comparisons fold to `0`, not panic.

## Other invariants

- Literals are stored as raw source `Symbol`s (including quotes/prefixes); decoding is downstream.
- `Attr::args()` returns arguments in source order — not a keyed lookup. Match on
  `AttrArg::KeyValue` to read `@parallel(grain = 1, restrict)`.
- `Path::first()` panics on empty path — parser guarantees ≥1 segment.
- `StructLit.rest` exists in the AST but the parser never fills it (functional update rejected).

## Testing

```sh
cargo test -p wukong_ast --lib
```

8 tests (5 for `fold_const_len`, 3 for the printer). Mirror covered by `cargo test -p wukongc --test run`
and `tests/fail` fixtures for E0501.
