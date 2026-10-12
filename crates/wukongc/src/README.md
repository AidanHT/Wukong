# `wukongc` source

| file | role |
|---|---|
| `main.rs` | `USAGE` string (authoritative CLI doc), `parse_args`, `run_compile` (256 MiB stack thread), `print_explanation`; `main` hands `wukongc fmt ...` (`args[1] == "fmt"`) to `fmt_cmd::run` before any flag is parsed |
| `fmt_cmd.rs` | the `wukongc fmt` subcommand: `FMT_USAGE` (its own help text), argument parsing, the sorted recursive `*.wk` walk (linked directories not followed), `--check` / `--stdout` / stdin modes, rendering `FmtError::Syntax` with `wukong_diag::Renderer`, and `write_atomically` (resolves a symlink to its target, copies the file's permissions, refuses a read-only file) |

All compile-pipeline logic lives in `wukong_driver`: for a compile, this crate only parses flags and
calls `compile(&opts)`. The one exception is `fmt`, which never compiles: `fmt_cmd.rs` calls
`wukong_fmt::format_source` directly.
