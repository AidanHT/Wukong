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
