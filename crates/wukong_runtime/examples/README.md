# examples/

Dev-only performance probes for iterating on the hand-written kernels. **Not** part of the compiler
test gate and not shipped benchmarks.

| example | purpose | run |
|---|---|---|
| `gemm_scaling.rs` | Adjacent serial-vs-`wukong_sgemm_nt_parallel` timing at transformer GEMM shapes; prints GFLOP/s and speedup | `cargo run -p wukong_runtime --release --example gemm_scaling` |
| `gemm_var.rs` | ABBA-discipline run-variance probe: Wukong parallel vs oneMKL `cblas_sgemm` per round | `cargo run -p wukong_runtime --release --example gemm_var [-- <rounds>]` |
| `gemm_skinny.rs` | Small-M (skinny) NT GEMM probe: an `M`-sweep at fixed `(K, N)` that separates pack amortization from a peer gap, plus the six xbench `matmul_skinny` shapes timed adjacent to oneMKL single-thread in one process (timing thread pinned) | `cargo run -p wukong_runtime --release --example gemm_skinny` |
| `norm_var.rs` | Single-row fused-norm cost decomposition: splits the benchmarked program into `wukong_velem_f32` (the recognized `out[c] = x[c]` copy), `wukong_norm_f32`, and their sum, beside a bare 256-bit copy floor and the **same-process** gcc `-ffast-math` peer | `cargo run -p wukong_runtime --release --example norm_var [-- <n,n,..>]` |
| `normbwd_var.rs` | LayerNorm-backward pass decomposition (mean / var / s1+s2 / apply, against a one-pass streaming floor) plus an ABBA A/B of the shipped kernel against a verbatim copy of its pre-multi-accumulator form; also probes `softmax_bwd`'s delegated dot | `cargo run -p wukong_runtime --release --example normbwd_var [-- <rounds>]` |
| `normfwd_split.rs` | Splits the benchmarked batched-norm program into its two dispatched calls (`wukong_velem_f32` copy, then `wukong_norm_f32`) and prints each one's share, so the reachable ceiling for a norm-kernel change is a number | `cargo run -p wukong_runtime --release --example normfwd_split [-- <rounds>]` |

All six require `--release` — debug builds make the hand-written kernels look far slower than JIT
loops. `gemm_var` optionally loads MKL via `WUKONG_MKL_DLL`; `GEMM_VAR_HUGE=1` adds a 2048³ shape.
`norm_var` builds its peer with `$CC` (default `gcc`) and needs it on `PATH`.

Why `norm_var` exists: the 2026-09 CPU evaluation recorded the `1x768` / `1x4096` fused-norm rows as
the suite's worst peer losses, and a single wall time per row cannot say whether the nanoseconds are
in the reduction, the recognized copy that precedes it, or per-call overhead. Sweeping `n` and
reading the intercept separates the fixed cost from the per-element cost. Both arms are timed in one
process, so the ratio is a same-run ratio rather than a cross-process comparison.

`normbwd_var` writes every timed arm into **one** shared output buffer, deliberately: giving each arm
its own 4-8 MB destination pushes the working set past this host's 24 MB L3 and turns every arm
DRAM-bound, which halves the apparent difference between them. Its ratios are same-process only —
never quote its absolute nanoseconds.
