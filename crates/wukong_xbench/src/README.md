# `wukong_xbench` modules

| file | role |
|---|---|
| `main.rs` | CLI, peer compilation (C/C++/Rust shared libs), elementwise/GEMM/reduction tables, timing loop |
| `general.rs` | GENERAL-CODE benchmark — structure tax, fused custom loss, Mamba/S6 scan; prints dispatch census |
| `model.rs` | 12-layer transformer forward vs PyTorch |

Peer ABIs are plain C (`extern "C" fn(*const f32, *const f32, *mut f32)` and variants for i8/bf16).
Wukong peers are compiled from generated `.wk` in a temp directory at runtime.
