# `wukong_autodiff` source layout

| Module | Role |
| --- | --- |
| `lib.rs` | `grad` entry, the `Vjp` worker (`replay` → `seed` → `reverse` → `finish`), scalar SSA VJP rules (`diff_inst`), `resolve_buffer_ptr`, `store_base` |
| `tape.rs` | Buffer tape: `Syms`, `is_kernel`, `kernel_arity`, `diff_kernel_call`, per-kernel VJP rules, synthesized-loop emitter (`open_loop`/`close_loop`). Private module reached through `impl Vjp` methods. |
| `optim.rs` | `build_adamw_step`, `adamw_step_name`, `hp` index module for AdamW hyperparameters |
| `tests.rs` | Finite-difference + closed-form gate and per-rule tests (37 tests) |

## `tape.rs` dispatch chain

Recognition is by interned symbol name; serial and `@parallel` spelling both:

| Kernel (+ `_parallel`) | Arity | Rule |
| --- | --- | --- |
| `wukong_sreduce_f32` | 4 `(x,y,n,op)` | SUM / DOT / SSD → velem fill / scale / affine |
| `wukong_sgemm_nt` | 7 `(a,b,c,m,k,n,beta)` | `dA = dC·B`; `dB = dCᵀ·A` via synthesized transpose + `wukong_sgemm` |
| `wukong_vmath_f32` | 4 `(x,out,n,op)` | relu/sigmoid/tanh/exp → synthesized loop; silu/gelu/elu/softplus → fused `wukong_vmath2_f32` BWD |
| `wukong_velem_f32` | 8 `(x,y,out,n,a,b,c,op)` | identity-affine only: `dx = a·dout`, `dy = b·dout` |
| `wukong_norm_f32` | 6 `(x,out,rows,cols,eps_bits,op)` | softmax / LayerNorm / RMSNorm backward per row |

## `canon` vs `store_base`

`canon` (`tape.rs`) peels only a **zero**-index `gep` before keying `buf_adj` — lowering passes
some kernel operands as `gep(buf, 0)` and others as bare `alloca`. Without canonicalization the two
spellings key different buffer adjoints.

`store_base` (`lib.rs`) peels a whole `gep` chain regardless of index, because `h[3] = v` still
writes into `h`. Do not merge these helpers.

## Adding a kernel rule

1. Intern the symbol (both spellings) in `Syms`.
2. Add to `is_kernel`.
3. Add the `kernel_arity` arm.
4. Add the `diff_kernel_call` dispatch arm + rule.
5. Mirror any new runtime op code constant.
6. Add a `tape_gate` test.
