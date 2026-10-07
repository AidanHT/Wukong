# general-code benchmark fixtures

Sources for the `general` section of `wukong_xbench` (`src/general.rs`). Wukong programs written the
way an ML engineer actually writes them — not forced into the recognizer dialect.

## Wukong programs

| file | variant | what it isolates |
|---|---|---|
| `block_a_recognizer.wk` | (a) monolithic, recognizer dialect | baseline — everything dispatches |
| `block_b_natural.wk` | (b) monolithic, natural spelling | pure spelling tax (do not rewrite into dialect) |
| `block_c_helpers.wk` | (c) factored helpers over `[]f32` | function factoring + runtime dims |
| `block_d_struct.wk` | (d) weights in a `struct Layer` | struct-held parameters |
| `block_e_tensor.wk` | (e) shape-typed `Tensor[f32, S, D]` | the type system |
| `loss_focal.wk` | focal loss forward + hand-written backward | custom objective not in recognizer table |
| `scan_s6.wk` | Mamba/S6 selective scan | matrix-state recurrence beyond `wukong_lrscan_f32` |

## Peer libraries

| file | language | paired with |
|---|---|---|
| `peer_block.c` / `peer_block.rs` | C / Rust | block variants (a)–(e) |
| `peer_loss.c` / `peer_loss.rs` | C / Rust | `loss_focal.wk` |
| `peer_scan.c` / `peer_scan.rs` | C / Rust | `scan_s6.wk` |

gcc/g++/rustc spread across the five block spellings is ~1.0 (same loops). Wukong's spread **is** the
benchmark. The harness prints a dispatch census so recognizer firing is visible — recognized kernels
are GATE-BLIND to `cargo test`.

## What the census actually says (measured 2026-08-23)

"Spellings that may miss the recognizers" is how this corpus was designed and is **not** what it
does. Every one of the five block spellings dispatches — 13 to 16 call sites each, `b-natural`
included, the one written with no regard for the recognizers at all. Row (a) is not "the one that
dispatches"; it is the one that dispatches *most*. `loss_focal.wk` and `scan_s6.wk` are the two that
genuinely reach nothing.

That is why `src/general.rs` builds each Wukong program **twice** — once as written, once with
recognition off (the `(gen)` columns). Without the second lowering this section has no row a §3B
language ratio can be read from, and its geomean is a measurement of the dispatcher wearing the word
"general". Both halves are kept: what the recognizer reaches is a real result, it is just a different
one. Reproduce either census with

```sh
                       wukongc --emit=mir -O2 <dumped .wk>   # the recognizer-on column
WUKONG_NO_RECOGNIZE=1  wukongc --emit=mir -O2 <dumped .wk>   # the (gen) column
```

on the source the harness dumps next to the peer libraries — same file, the switch is the only
difference.
