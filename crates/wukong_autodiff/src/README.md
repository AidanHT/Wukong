# `wukong_autodiff` source layout

| Module | Role |
| --- | --- |
| `lib.rs` | `grad` / `grad_dot` / `hvp` / `inline_calls` entries, the `Vjp` worker (`replay` → `seed` → `reverse` → `finish`), scalar SSA VJP rules (`diff_inst`), the memory rules (`diff_load`, `diff_store`, `resolve_access`) |
| `tape.rs` | Buffer tape core: the `Kernel` families and the `FORWARD_KERNELS` table (recognition, arity, dispatch from one source of truth), `Syms`, gradient accumulation (`begin`/`end`), heap scratch, the write-after-write rule, reductions, `velem`, `vmath`, the norms |
| `tape_linalg.rs` | GEMM family (NN, NT, TN, α-scaled, fused epilogue), GEMV / GEVM, broadcast bias, transpose, column reductions |
| `tape_nn.rs` | All 36 activation backwards (`act_backward`), the `vmath2` family, cross-entropy / log-sum-exp / entropy / KL, RoPE, embedding |
| `tape_attn.rs` | The fused scaled-dot-product attention (`wukong_attention_f32`): the FlashAttention recompute of the weights, then the four adjoint GEMMs and the fused softmax backward |
| `coverage.rs` | `NOT_DIFFERENTIABLE`: every emitted runtime symbol the tape declines, with the reason a user sees; effect-only calls (`print`, `assert`, ...) |
| `inline.rs` | Call flattening: splice straight-line helper functions into the loss before differentiation |
| `region.rs` | `@parallel` regions (`grad_program`): take the outlined body apart, the iteration backward in region mode, the chunk driver, privatized shared adjoints added back in chunk order |
| `writes.rs` | `written_params`: which buffer parameters a function may write (a kernel's output operand, stores, unknown calls, escapes) — what lets `--train` run in place |
| `optim.rs` | `build_adamw_step`, `adamw_step_name`, `hp` index module for AdamW hyperparameters |
| `tests.rs` + `tests/` | Finite-difference + closed-form gate: per-rule tests, `kernels.rs` (new families), `attention.rs` (the fused attention), `writes.rs` (`written_params`), `memory.rs` (accumulation, in-place, stores), `calls.rs` (helpers, Hessian-vector products), `mirror.rs` (recognizer ↔ autodiff coupling), `regions.rs` (`@parallel` regions, f64 finite differences) |

## Adding a kernel rule

1. Add the symbol(s) — serial and `_parallel` — to `tape::FORWARD_KERNELS` under a `Kernel` family
   (a new family needs its `arity` and `out_index` arms).
2. Add the dispatch arm in `diff_kernel_call` and the rule. Write each contribution through
   `begin`/`end` (or a GEMM's `beta_for`), and allocate scratch only outside synthesized loops.
3. Mirror any new runtime op code constant at the top of `tape.rs`.
4. Gate it in `tests/kernels.rs` against an f64 closed form (`tape_gate`) or the finite difference
   (`tape_gate_fd`).

A recognizer arm that cannot be differentiated goes into `coverage::NOT_DIFFERENTIABLE` with an
honest reason instead — `tests/mirror.rs` fails the build until one of the two happens.

## `canon` vs `store_base` vs `resolve_access`

`canon` (`tape.rs`) peels only a **zero**-index `gep` before keying `buf_adj` — lowering passes some
kernel operands as `gep(buf, 0)` and others as bare `alloca`. `store_base` (`lib.rs`) peels a whole
`gep` chain regardless of index, because `h[3] = v` still writes into `h`. `resolve_access` (`lib.rs`)
walks a gep chain whose element type is the accessed type and sums the indices, so a gradient lands
on exactly the element the forward touched. Do not merge them.
