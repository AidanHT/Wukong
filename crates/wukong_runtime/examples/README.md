# examples/

Dev-only performance probes for iterating on the hand-written kernels. **Not** part of the compiler
test gate and not shipped benchmarks.

| example | purpose | run |
|---|---|---|
| `gemm_scaling.rs` | Adjacent serial-vs-`wukong_sgemm_nt_parallel` timing at transformer GEMM shapes; prints GFLOP/s and speedup | `cargo run -p wukong_runtime --release --example gemm_scaling` |
| `gemm_var.rs` | ABBA-discipline run-variance probe: Wukong parallel vs oneMKL `cblas_sgemm` per round | `cargo run -p wukong_runtime --release --example gemm_var [-- <rounds>]` |
| `normbwd_var.rs` | LayerNorm-backward pass decomposition (mean / var / s1+s2 / apply, against a one-pass streaming floor) plus an ABBA A/B of the shipped kernel against a verbatim copy of its pre-multi-accumulator form; also probes `softmax_bwd`'s delegated dot | `cargo run -p wukong_runtime --release --example normbwd_var [-- <rounds>]` |

All three require `--release` — debug builds make the hand-written kernels look far slower than JIT
loops. `gemm_var` optionally loads MKL via `WUKONG_MKL_DLL`; `GEMM_VAR_HUGE=1` adds a 2048³ shape.

`normbwd_var` writes every timed arm into **one** shared output buffer, deliberately: giving each arm
its own 4-8 MB destination pushes the working set past this host's 24 MB L3 and turns every arm
DRAM-bound, which halves the apparent difference between them. Its ratios are same-process only —
never quote its absolute nanoseconds.
