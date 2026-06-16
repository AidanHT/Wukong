# mercury_span

Foundational, dependency-free types shared by every later compiler stage: source positions (`Span`/`SourceId`), the `SourceMap` (owns text, maps byte offsets to line/col), and the string `Interner` (`Symbol`s).

## Layout
- `src/lib.rs` — `SourceId`, `Span`, and re-exports of everything else.
- `src/source_map.rs` — `Location`, `SourceFile`, `SourceMap`.
- `src/intern.rs` — `Symbol` and `Interner`.

## Key types & entry points
- `Span` (`src/lib.rs`) — fields `source: SourceId`, `lo`/`hi: u32` (all `pub`); half-open byte range `[lo, hi)`; 12 bytes, `Copy`. Helpers: `new`, `len`/`is_empty`, `to` (smallest covering span), `shrink_to_lo`/`shrink_to_hi`, `dummy`/`is_dummy`.
- `SourceId(pub u32)` (`src/lib.rs`) — index into the private `SourceMap.files` vec; just the insertion order.
- `SourceMap` (`src/source_map.rs`) — `add(name, src) -> SourceId` loads a file; `location(id, offset)` / `span_location(span)` give 1-based line/col; `span_text`, `line_text`, `source`, `name`, `file_count`.
- `Interner` (`src/intern.rs`) — `intern(&str) -> Symbol` (dedups via a `HashMap<Box<str>, Symbol>`), `resolve(Symbol) -> &str`, `len`/`is_empty`.
- `Symbol(pub u32)` (`src/intern.rs`) — `Copy`/`Eq`/`Hash` handle to an interned string.

## Connects to
Upstream: none (zero deps, leaf crate). Downstream: every front-end and later crate — lexer/parser attach `Span`s, names become `Symbol`s, diagnostics resolve via `SourceMap`.

## Gotchas
- `Span::dummy()` uses `SourceId(u32::MAX)`; check `is_dummy()` before indexing a `SourceMap` with `span.source` — it is not a valid file id.
- `Location` line/col are 1-based; `col` counts Unicode scalar values (chars) from line start, NOT bytes, so multi-byte UTF-8 advances col by one.
- `SourceFile::line_index` uses `binary_search` on `line_starts` (always begins with `0`); offset `0` maps to line index 0.
- `line_starts` is built by scanning for `\n` bytes only; `line_text` strips trailing `\n`/`\r`, but a bare `\r` does not start a new line.
- `Symbol`s are only valid within the `Interner` that created them; `resolve` indexes the `strings` vec and panics on a foreign/out-of-range symbol. Same for `SourceId` vs its `SourceMap` (`file()` panics out of range).
- `Span::to` `debug_assert`s both spans share a source; cross-source merges only fail loudly in debug builds.
