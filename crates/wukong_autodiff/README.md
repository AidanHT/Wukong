# wukong_autodiff

Reverse-mode automatic differentiation as a **MIR→MIR transform**. Wukong is the language; this
crate turns a forward MIR function computing a scalar loss into a backward function that accumulates
gradients w.r.t. designated input buffers (the vector-Jacobian product). Depends only on
`wukong_mir` and `wukong_span`; `wukong_interp` is a dev-dependency (the finite-difference gate's
oracle).

## Entry point

```rust
grad(func: &Function, wrt: &[usize], interner: &mut Interner) -> Result<Function, String>
```

Contract on `func`:

- Exactly one basic block, SSA form, terminated by `ret <float scalar>`.
- Every `wrt` index must be an in-range `MirType::Ptr` parameter.

The result `{name}_grad` returns `Void` and has the forward params **plus one appended `Ptr` per
`wrt` entry, in `wrt` order**. Gradients accumulate — the caller must pass zeroed buffers.

`wukong_driver` forces `opt_level.max(1)` for `--emit=grad` and `--train` regardless of the
requested `-O`, because autodiff consumes SSA MIR.

## Two adjoint models

**Scalar (`lib.rs`)** — one adjoint SSA value per forward value. Rules for `FAdd`/`FSub`/`FMul`/`FDiv`,
`Neg`, `Fma`, `Sqrt`, `Select`, and float `Load` (read-add-write into a gradient buffer). Constants
terminate an adjoint. Anything else with a live adjoint is an error naming the op.

**Buffer tape (`tape.rs`)** — one gradient buffer per forward buffer, keyed by the forward buffer
`ValueId`. Kernel calls (`wukong_sgemm_nt`, `wukong_vmath_f32`, `wukong_velem_f32`,
`wukong_sreduce_f32`, `wukong_norm_f32`) replay through `diff_kernel_call`, emitting backward work
via the same tuned runtime kernels as the forward pass.

The two meet at reductions: `diff_sreduce` turns a scalar adjoint of an `sreduce` result into a
buffer gradient.

## Optimizer kernels (`optim.rs`)

`build_adamw_step` / `adamw_step_name` — fused AdamW step as a MIR kernel. SGD (`w -= lr*g`) needs
no kernel here; the driver's `--train` SGD arm does it in Rust with one `wukong_velem_f32` call.

## LANDMINE — mirror `_parallel` recognizer arms

A new `_parallel` recognizer arm in `wukong_mir_build` **must** be mirrored in `tape.rs`:

1. `Syms` — intern both serial and `_parallel` spellings.
2. `is_kernel` — name dispatch.
3. `kernel_arity` — arity validation before indexing `args[..]`.
4. `diff_kernel_call` — the actual VJP rule.

Currently mirrored parallel twins: `sgemm_nt_parallel`, `vmath_f32_parallel`,
`sreduce_f32_parallel`, `velem_f32_parallel`, `norm_f32_parallel`. An unmirrored one makes the same
model differentiable in its serial spelling and *not* in its `@parallel` spelling.

`kernel_arity` must stay one-for-one with `diff_kernel_call`. `is_kernel` matches on name alone; a
`.wk` program may declare a kernel name with any signature, so arity is validated before the rules
index `args[..]` raw.

Other guards: refuse buffer-writing `Op::Call` with no rule; refuse `Op::VecKernelCall`; refuse
`Op::Store` into a buffer whose gradient is already live; velem ops outside the affine whitelist;
mid-function `@parallel` regions (`wukong_parallel_for`); norms that overwrite `x` in place;
`wukong_norm_affine_f32` (no rule). Runtime op codes (`VM_*`, `RED_*`, `NORM_*`, `VE_*`) at the top
of `tape.rs` mirror `wukong_runtime` — a renumber there silently changes which rule fires.

Errors are plain `Result<_, String>` prefixed `autodiff: …`; no `E`/`C` diagnostic codes.

## Testing

```sh
cargo test -p wukong_autodiff --lib
```

37 tests. The FD gate is the contract: `gate` runs scalar rules in f64 against central difference
and closed form; `tape_gate` runs the f32 kernel tape against loose f32 FD and tight f64 closed
form. Every new rule needs one of these, plus a `*_declines_loudly` test if it adds a guard.

End-to-end coverage lives in `wukong_driver`'s `grad_cli_tests`. Recognized kernels are gate-blind
(interp and native call the same symbol) — only the FD/closed-form check proves a gradient.

## Layout

See [`src/README.md`](src/README.md).
