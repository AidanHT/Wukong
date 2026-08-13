# `wukongc` source

| file | role |
|---|---|
| `main.rs` | `USAGE` string (authoritative CLI doc), `parse_args`, `run_compile` (256 MiB stack thread), `print_explanation` |

All pipeline logic lives in `wukong_driver`; this crate only parses flags and calls `compile(&opts)`.
