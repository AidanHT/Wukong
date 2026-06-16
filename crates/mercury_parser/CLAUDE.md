# mercury_parser

Hand-written recursive-descent + Pratt parser: turns a `mercury_lexer` token slice into a `mercury_ast` `Module`/`Expr`/`TypeExpr`. Sits between the lexer and sema.

## Layout
- `src/lib.rs` — the `Parser` struct, cursor helpers, and parsers for types, expressions (Pratt), statements, blocks, patterns, and attributes; the `binop_bp`/`token_to_binop`/`cur_assign_op`/`split_vector_ident` free fns; `parse_expr_str`/`parse_type_str` test entries.
- `src/items.rs` — `impl Parser` for item-level grammar: module header, `fn`/`struct`/`enum`/`const`/`import`/`extern`, generics, params. Public entries `parse_module`/`parse_module_tokens`.

## Key types & entry points
- `parse_module_tokens` (`src/items.rs`) — driver entry: takes already-lexed tokens, returns `(Module, Vec<Diagnostic>)` of parser-only diags (lexer tokenizes/reports separately).
- `parse_module` (`src/items.rs`) — lexes + parses in one call (tests/standalone); merges lexer + parser diags.
- `parse_expr_str` / `parse_type_str` (`src/lib.rs`) — standalone expr/type parse for tests.
- `Parser<'a>` (`src/lib.rs`) — `pub(crate)` state: `tokens`/`pos` cursor, `src`, `&mut Interner`, `next_node: u32` for `NodeId` allocation, `diags`.
- `parse_expr_bp` + `binop_bp` / `token_to_binop` (`src/lib.rs`) — Pratt precedence climbing; bp table 1 (`||`) .. 9 (`* / %`), all left-assoc (recurses at `bp + 1`).

## Connects to
Upstream: `mercury_lexer` (`Token`/`TokenKind`), `mercury_span` (`Interner`, `SourceId`, `Span`, `Symbol`), `mercury_diag` (`Diagnostic`). Downstream: produces `mercury_ast` nodes consumed by sema/mir_build.

## Gotchas
- Error recovery, never panics: bad input emits a diag (stable codes E0200–E0208) and synthesizes a placeholder node (e.g. `TupleLit([])` for a missing expr, `«error»` interned ident, `TypeKind::Unit` for a missing type). `module()` guarantees forward progress by bumping if `parse_item` stalls; `recover_item` skips to the next item-starting keyword/`@`/`pub`/Eof.
- `NodeId`s are sequential per-`Parser` from 0 — unique within one parse run only, not globally.
- Several "keywords" are recognized by source text, not token kind: `vec[...]`, `Tensor[...]`, `sizeof[...]`, `alignof[...]`, layout names (`.contiguous`/`.col_major`/`.strided`/`.tiled(...)`), and the `_` wildcard pattern. In `Tensor[...]` the layout (a `.name`) must come last; the dim loop breaks on the first `.`.
- SIMD vector types parse two ways: bare `vec[elem, N]` and the fused-identifier form `f32x8` via `split_vector_ident` (only for names in `is_scalar_name`). Note `vec[elem, N]` (comma) is a vector but `[elem; N]` (semi, leading `[`) is an array type.
- A parenthesized expr `(e)` reuses the inner node's `id`/`kind` but widens the span; it is NOT a distinct AST node. `(a,)` / `()` are tuples (`TupleLit`).
- Turbofish only triggers on `::<` immediately before a call; a bare `::name` is parsed as a `Field` access. `as` is postfix `Cast`. `::<...>` not followed by `(` emits E0205 and yields an empty arg list.
- Block tail vs statement: a trailing expr with no `;` before `}` becomes the block `tail`; otherwise it's a `StmtKind::Expr`. An assignment op (`=`, `+=`, …, via `cur_assign_op`) after an expr produces `StmtKind::Assign`. A local `const X: T = v;` is lowered to an immutable `StmtKind::Let`.
- `fn` bodies parse two forms: a `{...}` block, or `= expr;` (synthesizes a `Block` whose `tail` is the expr). A declaration with neither (just `;`) has `body: None`.
- `ident_like` accepts keyword tokens by their text (used for `@`-attribute names that collide with keywords); plain `ident` does not.
- Partially accepted-but-dropped syntax: `where` clauses (skipped to `{`/`;`/`=`/Eof), struct/enum field attrs, and enum `: repr` (parsed, not stored). Don't assume these survive into the AST.
- `expect` consumes nothing on mismatch (only emits the diag); callers must not assume the expected token was eaten. `bump` never advances past `Eof`.
