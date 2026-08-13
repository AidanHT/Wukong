# wukong_diag

The diagnostic model (`Diagnostic`, `Label`, `Severity`, `DiagnosticSink`), the **stable error-code
catalogue** behind `wukongc --explain`, and two renderers: rustc-style terminal output and
single-line JSON. Every front-end and MIR stage produces `Diagnostic`s; only `wukong_driver`
renders them.

**Place in pipeline:** lexer, parser, sema, mir_build, and the driver push diagnostics into a
shared `DiagnosticSink`. After compilation, the driver picks `Renderer` or `to_json` from
`--error-format`. Depends on `wukong_span` alone (no serde — JSON is hand-rolled).

## Key files

| file | role |
|---|---|
| `src/lib.rs:1-20` | crate overview; `Severity`, `Diagnostic` builder, `DiagnosticSink` |
| `src/catalog.rs:7-18` | code ranges; **`E0001` retired**, never reuse |
| `src/catalog.rs:48-267` | `CATALOG` table (30 entries at HEAD) |
| `src/catalog.rs:178-190` | `E0306` — unknown / misplaced `@attribute`, the closed known set |
| `src/catalog.rs:190-225` | E04xx entries — note **no `E0404`** (intentional gap) |
| `src/render.rs` | terminal renderer; tab expansion / caret alignment |
| `src/json.rs` | `to_json` — one JSON object per line |
| `tests/catalog_coverage.rs` | U6 two-direction catalogue gate |

## The code catalogue

| range | stage |
|---|---|
| `E00xx` | generic / internal *(none currently)* |
| `E01xx` | lexer |
| `E02xx` | parser |
| `E03xx` | name resolution |
| `E04xx` | type checking |
| `E05xx` | shape checking |
| `C0xxx` | codegen / lowering limits |

**Codes are permanently stable. Never renumber and never recycle a retired number.**

- **`E0001` is RETIRED** — was "malformed type syntax"; no stage ever emitted it (the parser
  reports malformed types as `E0203`/`E0204`). Must never be reused.
- **`E0404` has no entry** — leave the gap alone rather than filling it with an unrelated rule.

## Adding a diagnostic (three mandatory steps)

1. **Catalogue entry** in `src/catalog.rs`: `entry!("E0206", "<title>", "<body>")` in numeric
   order. Body must exceed 20 chars.
2. **Emit site** in the producing crate:
   `Diagnostic::error("…").with_code("E0206").primary(span, "…")` — the code must be a **bare
   string literal**, not built with `format!` or a `const`, or the U6 scan cannot see it.
3. **Fixture or probe:** `tests/fail/<name>.wk` with `// EXPECT-CODE: E0206`, or an entry in
   `LEXER_PARSER_PROBES` in `crates/wukongc/tests/fail.rs` for sources that cannot be a
   well-formed `.wk` file.

## Invariants / landmines

- Producers must **not print** — push into `DiagnosticSink`; let the driver render.
- Both renderers **skip dummy spans** (`span.is_dummy()`).
- Renderer output has **no trailing newline**; the driver adds one.
- `TAB_WIDTH = 4` in `render.rs` — if you touch tab expansion, touch all three helpers together.
- JSON key order and `"code":"Ennnn"` formatting are part of the contract (`wukongc/tests/fail.rs`
  greps stderr for exact substrings).
- U6 (`tests/catalog_coverage.rs`): `every_emitted_code_is_catalogued` and
  `every_catalogued_code_has_an_emitter`. `RESERVED` is empty — keep it that way unless deliberate.

## Testing

```sh
cargo test -p wukong_diag --lib    # unit tests + doctests
cargo test -p wukong_diag          # includes U6 integration tests in tests/
```

Codes end-to-end: `cargo test -p wukongc --test fail`.
