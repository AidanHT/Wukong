# wukong_lexer

`&str` → a flat `Vec<Token>` plus diagnostics. First stage of the pipeline; its output is consumed
by `wukong_parser`. Depends only on `wukong_span` (`SourceId`, `Span`) and `wukong_diag`
(`Diagnostic`).

**Place in pipeline:** `wukong_driver` calls `tokenize` on each source file before parsing.
`--emit=tokens` dumps the stream via `dump`.

## Key files

| file | role |
|---|---|
| `src/lib.rs:3-5` | **never aborts** — doc comment states the contract |
| `src/lib.rs:20-27` | `tokenize` entry point |
| `src/lib.rs:29-40` | `tokenize_with_comments` — the same scan plus the comment side channel |
| `src/lib.rs:42-81` | `CommentKind`, `Comment` (`text`, `is_doc`) |
| `src/lib.rs:83-91` | `dump` — backs `--emit=tokens` |
| `src/lib.rs:93+` | private `Lexer` byte-cursor scanner; `skip_trivia` records comments only when its `comments` field is `Some` |
| `src/token.rs` | `Token`, `TokenKind`, keyword/operator tables |

## Entry points

- `tokenize(src, source: SourceId) -> (Vec<Token>, Vec<Diagnostic>)` — stream **always** ends
  with `TokenKind::Eof`.
- `tokenize_with_comments(src, source) -> (Vec<Token>, Vec<Comment>, Vec<Diagnostic>)` — the same
  tokens and diagnostics as `tokenize` (a unit test pins it, BOM, CRLF and E0103 included), plus
  every comment `skip_trivia` skipped, in source order. Used by `wukong_fmt`. Plain `tokenize`
  passes `None` for the side channel, so its tokens, cost and `--emit=tokens` output are unchanged.
- `Comment { kind: CommentKind, span }` — `CommentKind::Line` spans `//` up to, not including, the
  line feed (a CRLF's carriage return is inside the span); `CommentKind::Block` spans `/*` through
  the matching `*/`, counting nesting (an unterminated one runs to end of file).
  `Comment::text(src)` is the comment's text minus one trailing carriage return;
  `Comment::is_doc(src)` is true for `///` (not `////`) and `//!`.
- `trivia_comments(src, source, lo, hi) -> Vec<Comment>` — the comments in one inter-token gap,
  by the same trivia scan; how the parser finds doc comments without a second token stream.
- `dump(tokens, src) -> String` — one `Kind "text"` line per token.

Literal values are **not stored** in tokens — later stages re-slice the source text via `span`.

## Diagnostics owned (E01xx)

| code | condition |
|---|---|
| `E0101` | unexpected character — emits `TokenKind::Error`, keeps scanning |
| `E0102` | unterminated string literal |
| `E0103` | unterminated block comment (nesting depth > 0 at EOF) |
| `E0104` | char-literal failures: empty `''`, multi-codepoint, unterminated |

Codes are stable — never renumber, never recycle. Catalogue entries in
`wukong_diag/src/catalog.rs`. E0101 is covered by `tests/fail/unexpected_char.wk`; E0102/E0103/E0104
by `LEXER_PARSER_PROBES` in `crates/wukongc/tests/fail.rs` (cannot appear in well-formed `.wk`
fixtures).

## Invariants / landmines

- The lexer **never aborts**: unrecognized input becomes `TokenKind::Error` plus a diagnostic;
  scanning continues.
- **Char-literal resync** (`lib.rs:513-525`): multi-codepoint `'ab'` must jump past its own closing
  quote or the `;` is swallowed and a second diagnostic fires.
- Scanning is byte-level for operators/identifiers/numbers; non-ASCII goes through `utf8_len`.
- `Span` offsets are byte offsets — spans not on char boundaries panic downstream.
- Four keyword tables (`keyword`, `is_keyword`, `glyph`, `name`) are independent — dropping a
  variant from `is_keyword()` is silent. Pinned by `keyword_tables_agree` (`lib.rs:940`).

## Testing

```sh
cargo test -p wukong_lexer --lib
```

Unit tests: keyword round-trip, numbers, operators, strings/chars, nesting comments, each E01xx
code, char resync, the `dump` snapshot, and the comment side channel (`tokenize_with_comments`
agrees with `tokenize`; comment spans, kinds and doc flags). End-to-end code reachability:
`cargo test -p wukongc --test fail`.
