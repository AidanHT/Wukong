# `wukong_interp` source layout

The whole crate is `lib.rs` (~4.8k lines):

| Section | Role |
| --- | --- |
| `Accelerator` trait | GPU offload hook seam (`sgemm_nt`, `vmath`, `norm`, `sreduce`, `sgemm_nt_epi`) |
| `run` / `run_with_output` / `run_with_output_accel` | Public entry points |
| `run_kernel_f32` / `run_kernel_f64` / `run_kernel_i8` | Typed kernel-entry ABI |
| `Interpreter` | `Backend` impl (thin adapter over `run_with_output`) |
| `Interp` | Per-run state: `run_function`, `exec_frame`, `eval`, `intrinsic` |
| Semantic helpers | `mask`, `mask_lane`, `slot_count`, `push_defaults`, `int_bits`, `uval`, `int_to_float`, `apply_bin`, `apply_cmp`, `apply_cast`, `apply_fma`, `apply_sqrt`, `apply_round` |
| `mod tests` | 19 unit/integration tests |

## `intrinsic`

Every recognized-kernel symbol needs an arm here. Kernel arms call `wukong_runtime` serial symbols
and also match `_parallel` spellings. The `dim!` macro guards non-positive extents. Optional pointer
params (`bias`, `gamma`/`beta`) arrive as `Value::Int(0)` and are distinguished by variant.

## `run_function` / `exec_frame` / `eval`

Tree-walk over MIR instructions, **stackless**: `run_function` keeps an explicit frame stack, and
`exec_frame` runs the top frame until it calls a Wukong function (or a `wukong_parallel_for` body,
run as one sequential call) or returns, so a Wukong call never recurses on the host stack and every
entry point runs on its caller's thread. `canonical_result` masks integer results and rounds
floats; vector ops use lane-wise arms. Step limit (100M blocks per frame) and the call-depth ceiling
(`MAX_CALL_DEPTH` = 300_000 in every build profile, overridable with `WUKONG_INTERP_MAX_DEPTH`) are
checked here.

`wukong_codegen_gpu` mirrors `apply_cast` and `int_bits` from this file; `fusion.rs` mirrors the
`Accelerator` recognizer list. Editing cast or width semantics here without mirroring there
desynchronizes the GPU backend.
