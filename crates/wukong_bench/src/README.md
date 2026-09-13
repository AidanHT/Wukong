# `wukong_bench` modules

| file | role |
|---|---|
| `main.rs` | CLI dispatch, corpus walk, `-O0`/`-O3` MIR-count and interpreter timing report |
| `compile_time.rs` | `compile-time` mode — per-stage optimizer timing |
| `compile_vs.rs` | `compile-vs` mode — cross-compiler compile-time comparison |
| `compile_work.rs` | `compile-work` mode — the counting instrument (allocations + pass invocations per stage/pass); also home of the binary's counting `#[global_allocator]` |
| `profile.rs` | `compile-profile` and `spawn-overhead` modes |

All modes share the same corpus-selection logic and the embedded `-O0`/`-O3` correctness gate.
