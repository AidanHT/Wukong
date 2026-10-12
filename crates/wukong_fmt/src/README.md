# `wukong_fmt` modules

| file | exports |
|---|---|
| `lib.rs` | `format_source`, `FmtOptions`, `FmtError`; the pipeline (lex with comments, parse, print, self-check, line endings) |
| `printer.rs` | `Src` (source text, tokens, comments, interner, with token lookups by position), `Printer` (items, statements, expressions, types, patterns), `print_module` |
| `comments.rs` | `Printer::leading` / `trailing` / `close_at` / `close_bare` / `flush_rest`: comment placement and blank lines |
| `check.rs` | `self_check` (the four laws) and `mask_spans` |
| `tests.rs` | golden tests (`#[cfg(test)]`) |
