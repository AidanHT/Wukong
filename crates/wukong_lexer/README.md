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
| `src/lib.rs:15-27` | `tokenize` entry point |
| `src/lib.rs:31-38` | `dump` — backs `--emit=tokens` |
| `src/lib.rs:40+` | private `Lexer` byte-cursor scanner |
| `src/token.rs` | `Token`, `TokenKind`, keyword/operator tables |

## Entry points

- `tokenize(src, source: SourceId) -> (Vec<Token>, Vec<Diagnostic>)` — stream **always** ends
  with `TokenKind::Eof`.
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
- **Char-literal resync** (`lib.rs:582-598`): multi-codepoint `'ab'` must jump past its own closing
  quote or the `;` is swallowed and a second diagnostic fires.
- Scanning is byte-level for operators/identifiers/numbers; non-ASCII goes through `utf8_len`.
- `Span` offsets are byte offsets — spans not on char boundaries panic downstream.
- Four keyword tables (`keyword`, `is_keyword`, `glyph`, `name`) are independent — dropping a
  variant from `is_keyword()` is silent. Pinned by `keyword_tables_agree` (`lib.rs:601`).

## Testing

```sh
cargo test -p wukong_lexer --lib
```

14 unit tests (keyword round-trip, numbers, operators, strings/chars, nesting comments, each E01xx
code, char resync, `dump` snapshot). End-to-end code reachability:
`cargo test -p wukongc --test fail`.
