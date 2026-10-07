# `src/` module index

Flat module tree — one `.rs` file per kernel family, all wired from `lib.rs`.

## `lib.rs`

Crate root: re-exports every `wukong_*` kernel symbol, `bf16`/`f16` conversion helpers, `Arena`,
`parallel_for` (sequential reference), `wukong_parallel_for`, rayon pool setup (`ensure_global_pool`,
`run_on_wuk_pool`, `gemm_pool` routing), `wukong_rt_alloc`/`free`, file-I/O macros, and the
`#[cfg(test)]` suite for arena/alloc/I/O/parallel_for.

## Linear algebra

| file | kernels |
|---|---|
| `gemm.rs` | `wukong_sgemm*` — f32/bf16/f16 NT/TN/EPI variants, serial + `_parallel`, private GEMM pool |
| `gemv.rs` | `wukong_sgemv*` — matrix-vector |
| `gevm.rs` | `wukong_sgevm_f32*` — vector-matrix |
| `i8gemm.rs` | `wukong_i8gemm_nt*` — int8 matmul + dequant |
| `transpose.rs` | `wukong_transpose_f32*` / `u16*` |

## Elementwise & math

| file | kernels |
|---|---|
| `vmath.rs` | `wukong_vmath_f32*` / `bf16` / `f16` — transcendental & activation ops (`VM_*` op ids) |
| `velem.rs` | `wukong_velem_f32*`, `wukong_vhorner_f32` — binary elementwise (`VE_*` ids) |
| `lowp.rs` | `wukong_axpby_*`, `wukong_dot_*`, `wukong_sum_*`, `wukong_reduce_*`, `wukong_vmath_*_out` |
| `dequant.rs` | `wukong_dequant_f32*`, `wukong_dequant_perchan_f32*` (`DQ_*` ids) |
| `bias.rs` | `wukong_bias_bcast_f32*` |

## Reductions & column ops

| file | kernels |
|---|---|
| `reduce.rs` | `wukong_sreduce_f32*`, `wukong_argreduce_f32*` (`RED_*` ids) |
| `colreduce.rs` | `wukong_col{sum,mean,rms,max,min,maxabs,sumsq,l2}_f32*`, `wukong_coll2_f32*` |
| `rowarg.rs` | `wukong_rowarg{max,min}_i32*` |
| `colarg.rs` | `wukong_colarg{max,min}_i32*` |

## Normalization & softmax

| file | kernels |
|---|---|
| `norm.rs` | `wukong_norm_f32*`, `wukong_norm_affine_f32*` (`NORM_*` ids) |
| `logsoftmax.rs` | `wukong_logsoftmax_f32*`, `wukong_logsumexp_f32*` |
| `softmax_bwd.rs` | `wukong_softmax_bwd_f32*` |
| `rmsnorm_bwd.rs` | `wukong_rmsnorm_bwd_f32*` |
| `layernorm_bwd.rs` | `wukong_layernorm_bwd_f32*` |

## Attention & embeddings

| file | kernels |
|---|---|
| `attention.rs` | `wukong_attention_f32` |
| `embedding.rs` | `wukong_embedding_f32*` |
| `rope.rs` | `wukong_rope_f32*` |
| `rope_bwd.rs` | `wukong_rope_bwd_f32*` |

## Loss & training helpers

| file | kernels |
|---|---|
| `xent.rs` | `wukong_xent_fwd_f32*` |
| `xent_bwd.rs` | `wukong_xent_bwd_f32*` |
| `kldiv.rs` | `wukong_kldiv_f32*` |
| `entropy.rs` | `wukong_entropy_f32*` |
| `kd_loss.rs` | `wukong_kd_loss_f32*` |

## Scan, pool, scatter

| file | kernels |
|---|---|
| `cumsum.rs` | `wukong_cumsum_f32*` |
| `cumminmax.rs` | `wukong_cum{min,max}_f32*` |
| `cumprod.rs` | `wukong_cumprod_f32*` |
| `lrscan.rs` | `wukong_lrscan_f32*` |
| `pool2d.rs` | `wukong_{avg,max}pool2d_f32*` |
| `scatter.rs` | `wukong_scatter_add_f32*` |

## Scheduling policy (no kernels)

| file | contents |
|---|---|
| `par.rs` | Shared `_parallel` fork gate (`fork_rows`/`fork_units`, `WUKONG_PAR_MIN_WORK`), the atomic claim queue (`claim_chunks`), `for_each_row` (`WUKONG_KERNEL_DYN=1` A/B) |
| `topology.rs` | Host probe: usable logical/physical cores, L1D/L2/L3 (CPUID / sysfs), `WUKONG_CPU_PHYSICAL` / `WUKONG_CACHE_*_KB` / `WUKONG_NUM_THREADS` |
| `nt_policy.rs` | Non-temporal-store threshold (3/4 of the probed L3, `WUKONG_NT_MIN_BYTES`) |

Each kernel file follows the same shape: `extern "C"` AVX2 entry, scalar twin, `#[cfg(test)]`
`simd_equals_scalar_twin` table.
