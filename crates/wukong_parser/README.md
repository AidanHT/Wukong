# wukong_parser

Hand-written **recursive descent** for items, statements, types and patterns, plus a **Pratt**
(precedence-climbing) parser for expressions. Turns a `wukong_lexer` token slice into `wukong_ast`
nodes. Allocates `NodeId`s, interns identifiers and literal text, and recovers from every error —
it never panics and never gives up on the rest of the file.

**Place in pipeline:** after `tokenize`, before `wukong_sema`. `wukong_driver` parses the root file
with `parse_module_tokens_with_lints` (which also returns the parser's lints) and the import loader
calls `parse_module_tokens_from`, so imported files get disjoint `NodeId` ranges.

## Key files

| file | role |
|---|---|
| `src/items.rs:14-20` | `parse_module_tokens` — driver's entry point |
| `src/items.rs:30-41` | `parse_module_tokens_from` — multi-file `NodeId` watermark |
| `src/lib.rs:177` | `MAX_DEPTH = 1024` — E0209 nesting limit |
| `src/lib.rs:182-189` | `too_deep` — reports E0209 at most once |
| `src/lib.rs` | Pratt expressions, statements, patterns, attributes |
| `src/items.rs` | module header, `fn`/`struct`/`enum`/`const`/`import`/`extern` |

## Entry points

- `parse_module_tokens(tokens, src, &mut Interner) -> (Module, Vec<Diagnostic>)`
- `parse_module_tokens_from(…, first_node_id) -> (Module, Vec<Diagnostic>, u32)` — multi-file
- `parse_module_tokens_with_lints(…, first_node_id) -> ParseOutput` — the same, plus `lints`: the
  parser's warnings `W0030` (missing `;`) and `W0031` (a line-leading `(`, `[` or touching `-` that
  continues the expression above). Every other entry point drops them, so a caller asserting a
  clean parse is never failed by a warning.
- `parse_module(src, SourceId, &mut Interner)` — lex + parse (tests)
- `parse_expr_str` / `parse_type_str` — standalone parse for tests

## Diagnostics owned (E02xx)

| code | raised by |
|---|---|
| `E0200` | `expect()` — specific token required |
| `E0201` | `ident()` — name required; also `..base` rejection |
| `E0202` | `parse_primary` fallthrough; tuple field index overflow |
| `E0203` | `parse_type` fallthrough |
| `E0204` | `parse_layout` — unknown layout name |
| `E0205` | turbofish `::<…>` not followed by `(` |
| `E0206` | `parse_pattern_primary` |
| `E0207` | `parse_attr_arg` / `parse_attr_val` |
| `E0208` | `parse_item` — expected item at module scope |
| `E0209` | `too_deep` — nesting limit |
| `W0030` | `missing_semi` (lint channel) — statement without `;`; quiet after an error in the statement |
| `W0031` | `ambiguous_continuation` (lint channel) — only where the innermost open delimiter is `{` |

Codes are stable — never renumber, never recycle. Fixtures: `tests/fail/expected_expr.wk` (E0202),
`expected_item.wk` (E0208), `deep_nesting.wk` + `deep_blocks.wk` (E0209); others in
`LEXER_PARSER_PROBES` in `crates/wukongc/tests/fail.rs`.

## Invariants / landmines

- **Never aborts** — placeholders on error (`TupleLit([])`, `TypeKind::Unit`, `PatKind::Wildcard`,
  interned `«error»`); `recover_item` skips to the next item keyword.
- **`expect()` consumes nothing on mismatch** — callers must not assume the token was eaten.
- **Depth budget (E0209):** every recursive production and every postfix/binop/cast **fold** charges
  `depth`. A new postfix or fold form must charge the budget or deep input stack-overflows later
  consumers (sema, MIR lowering).
- **Forward progress:** once depth trips, over-deep sub-parses return without consuming — statement/
  arm/item loops must bump when nothing was consumed or the parser hangs.
- **Struct functional update `..base` is rejected** (`lib.rs:1028`) — `rest` has no lowering; filling
  it would leave fields uninitialized with no diagnostic.
- **`parse_u64_literal` must mirror** `wukong_sema::parse_u64_text` / `wukong_mir_build::parse_int`.

## Testing

```sh
cargo test -p wukong_parser --lib
```

22 tests (precedence, cast level, turbofish, radix extents, four depth/E0209 suites, recovery,
one-diagnostic regressions). Eyeball: `cargo run -q -p wukongc -- --emit=ast <file.wk>`.
