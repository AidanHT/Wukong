# wukong_autodiff

Reverse-mode automatic differentiation as a **MIR→MIR transform**. Wukong is the language; this
crate turns a forward MIR function computing a scalar loss into a backward function that accumulates
gradients w.r.t. designated input buffers (the vector-Jacobian product). Depends only on
`wukong_mir` and `wukong_span`; `wukong_interp` is a dev-dependency (the finite-difference gate's
oracle).

## Entry points

```rust
grad(func: &Function, wrt: &[usize], interner: &mut Interner) -> Result<Function, String>
grad_program(program, func, wrt, interner) -> Result<Gradient, String>   // + @parallel regions
inline_calls(program: &Program, func: &Function) -> Function          // splice straight-line helpers
grad_dot(func, wrt, interner) -> Result<Function, String>             // <v, ∇L>, single-block
hvp(func, wrt, interner) -> Result<(Function, Function), String>      // (grad_dot, Hessian·v)
written_params(func: &Function, interner: &Interner) -> Vec<bool>     // which buffers func may write
```

`written_params` is the tape's kernel knowledge turned outward: a recognized kernel writes only its
output operand, so a weight that only feeds kernels is provably unchanged by a call; stores,
unrecognized calls and pointer escapes are conservatively writes. The `--train` loop uses it to run
the gradient on the live buffers and snapshot only the buffers the forward writes.

Contract on `func`:

- Exactly one basic block, SSA form, terminated by `ret <float scalar>`.
- Every `wrt` index must be an in-range `MirType::Ptr` parameter.

The result `{name}_grad` returns `Void` and has the forward params **plus one appended `Ptr` per
`wrt` entry, in `wrt` order**. The caller must pass zeroed buffers.

`wukong_driver` forces `opt_level.max(1)` for `--emit=grad`, `--train` and `--emit=obj --grad-of`
regardless of the requested `-O`, because autodiff consumes SSA MIR, and runs `inline_calls` on the
loss first, so a loss written with helper functions differentiates at every level.

## Two adjoint models

**Scalar (`lib.rs`)** — one adjoint SSA value per forward value. Rules for `FAdd`/`FSub`/`FMul`/`FDiv`,
`Neg`, `Fma`, `Sqrt`, `Select`, precision casts, scalar trig, float `Load` (read-add-write into a
gradient buffer, through a whole `gep` chain) and float `Store` (the memory adjoint rule). Constants
and integer leaves terminate an adjoint. Anything else with a live adjoint is an error naming the op
or the callee.

**Buffer tape (`tape.rs`, `tape_linalg.rs`, `tape_nn.rs`)** — one gradient buffer per forward
buffer. Every recognized kernel family is listed once in `tape::FORWARD_KERNELS` (recognition,
arity and dispatch from one table; 50 symbols, serial and `@parallel`). The backward emits the
runtime's own tuned kernels — transposed GEMM spellings, `wukong_{softmax,layernorm,rmsnorm}_bwd_f32`,
the fused `wukong_vmath2_f32` backward codes, `wukong_xent_bwd_f32`, `wukong_rope_bwd_f32`,
`wukong_scatter_add_f32` — in the `@parallel` spelling when the forward used it. Gradients
accumulate across fan-out (`Vjp::begin`/`end`, a GEMM's `beta = 1`); scratch is zeroed heap
(`wukong_rt_alloc`), released before the gradient returns.

## `@parallel` regions (`region.rs`)

`grad_program` is `grad` for a loss that runs `wukong_parallel_for(n, &P, env)` — a loop region of a
`@parallel` function, or a whole-function `@parallel` helper spliced in. `program` supplies `P`,
which must be the canonical `-O1` counted loop with ONE straight-line iteration block (the driver
re-optimizes outlined bodies at `-O1` for a gradient compile). The result's `support` holds, per
region, `{name}_grad$par<i>$it(hh, env, gc)` — one iteration recomputed and differentiated by the
single-block rules in region mode (captured buffers are bases whose gradients arrive in `gc`; every
contribution accumulates; the replay never rewrites a captured buffer) — and the chunk driver
`{name}_grad$par<i>(cs, ce, benv)` the gradient hands to `wukong_parallel_for`. A buffer several
iterations read gets `K = min(N, 16)` private zeroed adjoint copies, added back in chunk order; the
chunks depend on `N` only, so the gradient's bits do not depend on the thread count or the backend.
Declines (by name): control flow in the iteration, a store through a captured slice, a region reading
what it writes (under one name or two: a slice and its array), a captured buffer the loss rewrites
after the region under any name (a slice view, a later region, a kernel, a runtime or user call),
a pointer the alias model (`check_aliasing`) cannot place, a written slot that is a view into
another buffer, a shared buffer of unknown extent, second order.

## LANDMINE — the recognizer ↔ autodiff coupling

Every `"wukong_*"` symbol `wukong_mir_build` can emit must be in `tape::FORWARD_KERNELS` (with a VJP
rule), in `coverage::SCALAR_CORE`, in `coverage::REGION_CALLS` (`wukong_parallel_for`, the region
rule), or in `coverage::NOT_DIFFERENTIABLE` with the reason a decline reports. `tests/mirror.rs` scans `wukong_mir_build`'s sources and fails the build otherwise — a new
recognizer arm (serial or `_parallel`) cannot land without an autodiff disposition. Runtime op codes
(`VM_*`, `VM2_*`, `RED_*`, `NORM_*`, `VE_*`, `EPI_*`) at the top of `tape.rs` mirror
`wukong_runtime` — a renumber there silently changes which rule fires.

Errors are plain `Result<_, String>` prefixed `autodiff: …`; no `E`/`C` diagnostic codes.

## Optimizer kernels (`optim.rs`)

`build_adamw_step` / `adamw_step_name` — fused AdamW step as a MIR kernel. SGD (`w -= lr*g`) needs
no kernel here; the driver's `--train` SGD arm does it in Rust.

## Testing

```sh
cargo test -p wukong_autodiff --lib
```

The FD gate is the contract: `gate` runs scalar rules in f64 against central difference and closed
form; `tape_gate` runs the f32 kernel tape against loose f32 FD and tight f64 closed form. Every new
rule needs one of these (`tests/kernels.rs` for kernel families), plus a `*_declines_loudly` test if
it adds a guard; routing to a tuned backward kernel is gate-blind and pinned structurally (`emits`).

End-to-end coverage: `wukong_driver`'s `grad_cli_tests` / `grad_model_tests` (source models at
O0–O3), `crates/wukongc/tests/autodiff_cli.rs` (the CLI, interp-vs-native training) and
`tests/run/autodiff_train_*.wk` (hand-derived trajectories).

## Layout

See [`src/README.md`](src/README.md).
