# `wukong_parser` modules

| file | exports |
|---|---|
| `lib.rs` | `Parser` struct; types, expressions (Pratt), statements, blocks, patterns, attributes; `parse_expr_str`, `parse_type_str` |
| `items.rs` | `parse_module`, `parse_module_tokens`, `parse_module_tokens_from`; item-level parsing (`fn`/`struct`/`enum`/`const`/`import`/`extern`) |
