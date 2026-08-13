# wukong_span

Foundational leaf crate: source positions (`Span`/`SourceId`), the `SourceMap` (owns loaded text,
maps byte offsets to 1-based line/column), the string `Interner` (`Symbol`s), and a dependency-free
FxHash. Every other compiler crate sits above this one.

**Place in pipeline:** created once per compilation by `wukong_driver`; every stage that needs
source locations or interned names depends on this crate. The lexer attaches `Span`s to tokens; the
parser interns identifiers; `wukong_diag` renders diagnostics against a `SourceMap`.

## Key files

| file | role |
|---|---|
| `src/lib.rs:19-27` | `SourceId`, `Span` definitions and re-exports |
| `src/source_map.rs:87-96` | `SourceMap::location` — 1-based line/col, char-counted columns |
| `src/intern.rs` | `Symbol` and `Interner` |
| `src/fxhash.rs` | `FxHasher` plus `FxHashMap`/`FxHashSet` aliases |
| `Cargo.toml:10` | **`[dependencies]` is empty — keep it that way** |

## Entry points

- `Span::new(source, lo, hi)`, `Span::dummy()`, `to(other)`, `shrink_to_lo()` / `shrink_to_hi()`
- `SourceMap::new()`, `add(name, src) -> SourceId`, `span_text`, `location`, `span_location`,
  `line_text`
- `Interner::new()`, `intern(&str) -> Symbol`, `resolve(Symbol) -> &str`

## Invariants / landmines

- **`Span` is a half-open BYTE range `[lo, hi)`**, not chars or columns. `hi >= lo` is assumed,
  never checked. Slices must land on UTF-8 boundaries or `span_text` panics.
- **`Span::dummy()` uses `SourceId(u32::MAX)`** — always `is_dummy()`-guard before indexing a
  `SourceMap` with `span.source`. Both `wukong_diag` renderers skip dummy labels for this reason.
- **`Location.col` counts Unicode scalar values**, not bytes (`source_map.rs:91`). Do not use
  `col - 1` as a byte offset.
- **`SourceId` is the insertion index** into one `SourceMap`; ids are only meaningful relative to
  the map that minted them. The driver and import loader share a single map per compilation.
- **`Symbol`s are valid only inside the `Interner` that minted them** — never persist or compare
  across interners.
- **FxHash determinism:** unlike `std`'s randomly-seeded `HashMap`, FxHash has a fixed seed.
  Never derive observable output order (MIR text, diagnostic order) from iterating an
  `FxHashMap`/`FxHashSet` — sort, or use an ordered container. `wukong_opt` keeps its own private
  `fxhash.rs` copy on purpose; do not unify casually.
- This crate owns **no diagnostic codes** and must not depend on `wukong_diag`.

## Testing

```sh
cargo test -p wukong_span --lib
```

10 unit tests (span algebra, multi-line/multibyte locations, `line_text`, interner dedup). No
integration tests in this crate; downstream gates cover it indirectly.
