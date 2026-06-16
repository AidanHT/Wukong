# mercury_lexer

`&str` -> flat `Vec<Token>` (+ diagnostics). First stage of the pipeline; feeds `mercury_parser`.

## Layout
- `src/lib.rs` — the `Lexer` scanner, `tokenize`/`dump` entry points, trivia/number/string/op scanning, `utf8_len`, inline unit tests.
- `src/token.rs` — `Token`, `TokenKind` enum, and its `keyword`/`is_keyword`/`describe`/`glyph`/`name` methods.

## Key types & entry points
- `tokenize(src, source) -> (Vec<Token>, Vec<Diagnostic>)` (`src/lib.rs`) — the only real entry point. Token stream always ends in a `TokenKind::Eof` token.
- `dump(tokens, src) -> String` (`src/lib.rs`) — one `Kind "text"` line per token (slices the span out of `src`); backs `--emit=tokens` and snapshot tests.
- `Token` (`src/token.rs`) — just `{ kind: TokenKind, span: Span }`; `Copy`. Literal *values* are NOT stored — read them from source via `span`.
- `TokenKind` (`src/token.rs`) — `Copy` enum of all kinds. `keyword(s)` maps idents to keyword kinds; `glyph()` gives canonical fixed text for keywords/punctuation (None for value-carrying/sentinel kinds); `describe()` for "expected X" messages (falls back to `glyph()`); `name()` for dumps.
- `Lexer` (`src/lib.rs`) — private byte-cursor (`bytes`, `pos`); `next_token` dispatches on the first byte.

## Connects to
Upstream: `mercury_span` (`SourceId`, `Span`), `mercury_diag` (`Diagnostic`). Downstream: `mercury_parser` consumes the `Vec<Token>`.

## Gotchas
- Never aborts: an unknown byte becomes a `TokenKind::Error` token + diagnostic, then scanning continues. Callers must handle `Error` tokens.
- Diagnostic codes are E01xx: `E0101` unexpected char, `E0102` unterminated string, `E0103` unterminated block comment, `E0104` unterminated char.
- Strings are single-line: a newline (or EOF) inside a `"..."` ends the string with an `E0102`. The closing-quote consumes; the literal kind is always `Str` even on error.
- Number lexing has subtle lookahead: a `.` is a fractional point only if a digit follows, so `1..5` lexes as `Int DotDot Int` and `1.method` keeps the int separate. An `f` type suffix forces `Float`; an `e`/`E` exponent forces `Float`. Radix ints (`0x`/`0b`/`0o`) are always `Int`. `_` is allowed within digit runs.
- Identifiers are greedy over `[A-Za-z0-9_]`, so `f32x8` is one `Ident`, not `f32` + `x8`. `true`/`false` are keyword kinds (`True`/`False`), not literals.
- Block comments nest (`/* /* */ */`); both line and block comments are trivia, skipped silently (no tokens emitted).
- Byte-level scanning: non-ASCII is handled only via `utf8_len` when consuming a single char (string/char bodies, unknown-byte recovery). Operator/ident/number scanning assumes ASCII.
- Adding a `TokenKind` variant means updating `name()` (and usually `glyph()`/`describe()`, plus `keyword`/`is_keyword` for keywords) — they are exhaustive hand-written matches.
- `@` is just `TokenKind::At`; attribute syntax (`@tile(64)`) is not parsed here — it lexes as `At Ident LParen Int RParen` and is assembled later.
