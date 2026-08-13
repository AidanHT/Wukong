# `wukong_diag` modules

| file | exports |
|---|---|
| `lib.rs` | `Severity`, `NoteKind`, `Label`, `Diagnostic`, `DiagnosticSink`; re-exports below |
| `catalog.rs` | `Explanation`, `explain`, `all` — the stable code catalogue |
| `render.rs` | `Renderer` → rustc-style terminal output |
| `json.rs` | `to_json` — one JSON object per diagnostic |
