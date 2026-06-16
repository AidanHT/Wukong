# mercury_diag

Compiler diagnostics: the `Diagnostic` data model, a stable error-code catalog, and JSON + rustc-style terminal renderers. Every front-end and MIR stage produces these.

## Layout
- `src/lib.rs` — core types: `Severity`, `NoteKind`, `Label`, `Diagnostic` (fluent builder), `DiagnosticSink`; re-exports the other modules.
- `src/catalog.rs` — `CATALOG`: static table of `Explanation`s for `--explain`; `explain()` / `all()`.
- `src/render.rs` — `Renderer`: multi-line `error[E0501]: ...` terminal output with carets and gutter.
- `src/json.rs` — `to_json`: hand-rolled single-line JSON (JSON Lines) for `--error-format=json`.

## Key types & entry points
- `Diagnostic` (`src/lib.rs`) — built fluently: `Diagnostic::error(msg).with_code("E0501").primary(span, "...").help("...")`. `primary`/`secondary` push `Label`s; `note`/`help` push `(NoteKind, String)`. `code` is `Option<&'static str>` so codes must be string literals. `is_error()` is true for `Error` and `Bug`. `primary_span()` returns the first primary label's span, else the first label's.
- `DiagnosticSink` (`src/lib.rs`) — accumulator; `emit` bumps an error counter for error/bug severities. `take()` drains and resets (counter too); `diagnostics()` borrows without draining.
- `Renderer` (`src/render.rs`) — `new(color)` then `render(&d, &sm)`; needs a `mercury_span::SourceMap` to resolve spans to file/line/col and source text. Returns a string with NO trailing newline.
- `to_json` (`src/json.rs`) — needs a `SourceMap`; emits one object, no trailing newline.
- `Explanation` / `explain` / `all` (`src/catalog.rs`) — `explain` is case-insensitive (upper-cases input). `all` is re-exported from `src/lib.rs` as `all_explanations`.

## Connects to
Upstream: depends only on `mercury_span` (`Span`, `SourceMap`, `SourceId`). Producers (lexer, parser, sema, MIR verifier) construct `Diagnostic`s. Downstream: the `mercury_driver` crate owns the `SourceMap` + `DiagnosticSink` and drives `Renderer`, `to_json`, and `explain`/`all_explanations` (for `--explain`).

## Gotchas
- Error-code ranges live only as a doc comment + the `CATALOG` table in `src/catalog.rs`; codes are NOT enums. Adding a code means adding an `entry!` here AND passing the matching literal to `.with_code(...)` at the emit site — nothing enforces they stay in sync. A test asserts each entry's body is > 20 chars and the code starts with `E`/`C`.
- Both renderers SKIP dummy spans (`span.is_dummy()`); a diagnostic with only dummy labels renders header + notes with no source frame (see `renders_without_labels`).
- `src/json.rs` is hand-written (zero deps) — `esc` must stay correct; control chars < 0x20 become `\u00xx`. No serde.
- `render.rs` caret length: same-line spans use `hi.col - lo.col` (min 1); multi-line spans underline only to end of the first line. Columns are 1-based (`lo.col - 1` for indent).
- Renderer tests assert exact string output (`renders_header_location_and_caret`); changing spacing, gutter width, or the `|`/`-->`/`=` framing will break them.
- ANSI codes are split: `Severity::ansi()` (header colors, `pub(crate)`) lives in `src/lib.rs`; frame/caret colors (`BLUE`, `span_color`) live in `src/render.rs`.
