# examples/

Dev-only performance probes for iterating on the GEMM microkernel. **Not** part of the compiler
test gate and not shipped benchmarks.

| example | purpose | run |
|---|---|---|
| `gemm_scaling.rs` | Adjacent serial-vs-`wukong_sgemm_nt_parallel` timing at transformer GEMM shapes; prints GFLOP/s and speedup | `cargo run -p wukong_runtime --release --example gemm_scaling` |
| `gemm_var.rs` | ABBA-discipline run-variance probe: Wukong parallel vs oneMKL `cblas_sgemm` per round | `cargo run -p wukong_runtime --release --example gemm_var [-- <rounds>]` |

Both require `--release` — debug builds make the hand-written kernels look far slower than JIT loops.
`gemm_var` optionally loads MKL via `WUKONG_MKL_DLL`; `GEMM_VAR_HUGE=1` adds a 2048³ shape.
