# mercury_autodiff

Reverse-mode automatic differentiation as a **MIR→MIR transform** — the backward path that turns
Mercury from an inference compiler into a training one. Given a forward MIR function computing a
scalar loss, [`grad`] emits a new MIR function that runs the same forward pass and also accumulates
the gradient of that loss w.r.t. designated input buffers (the vector-Jacobian product).

## Layout
- `src/lib.rs` — the engine. `grad(func, wrt, interner) -> Function` entry; the `Vjp` worker
  (replay → seed → reverse) and the **scalar SSA** VJP rules (the adjoint-per-value model).
- `src/tape.rs` — the **tensor-tape** VJP rules (the adjoint-per-buffer model): differentiate a
  straight-line sequence of recognized runtime-kernel calls, emitting more kernel calls (so matmul
  gradients ride the tuned GEMM) plus synthesized counted loops for the transpose / Hadamard /
  per-row-reduction combines no single kernel covers. Also the `Syms` (interned kernel names) and the
  loop emitter (`open_loop`/`close_loop`, which nest).
- `src/optim.rs` — optimizer steps as MIR kernels. `build_adamw_step(it, n)` emits a fused AdamW
  update (moments + bias correction + rsqrt + decoupled weight decay in one counted-loop pass). SGD
  needs no kernel (`w -= lr*g` is one `mercury_velem_f32` call).
- `src/tests.rs` — the finite-difference + closed-form gradient gate and every op's test.

## The two adjoint models
- **Scalar (`lib.rs`)** — an adjoint SSA value per forward value, accumulated as the reverse walk
  emits ops. For straight-line scalar compute (unrolled nets, the loss glue). Rules: FAdd/FSub/FMul/
  FDiv, Neg, Fma, Sqrt, Select (relu-as-select), and Load-from-buffer routing (one-level gep / bare
  param, read-add-write so repeated loads accumulate). Constants terminate adjoints.
- **Buffer/tape (`tape.rs`)** — a gradient buffer per forward tensor buffer (PyTorch-style), set by a
  consumer's VJP and read by the producer's. The two meet at a reduction (a scalar-returning
  `sreduce` call seeds a buffer gradient via broadcast). Covered kernels: `sgemm_nt` (nn.Linear:
  dA = dC·B, dB = dCᵀ·A — one transpose loop + two NN GEMMs, `beta=1` to accumulate when a buffer
  feeds two matmuls), `vmath` activation (relu/sigmoid/tanh/exp, `dx = dout⊙f'(x)`), `sreduce`
  (SUM/SSD/DOT), `velem` identity (residual/scale), and `mercury_norm_f32` (softmax / LayerNorm /
  RMSNorm backward, per-row reductions + an elementwise combine in nested loops).

## Key entry points
- `grad(func, wrt, interner)` — the transform. `func`: a single-block, SSA-form forward function
  ending in `ret <loss>`, all params `Ptr` (buffers). `wrt`: parameter indices to differentiate. The
  result `{name}_grad` has the same params plus one appended `Ptr` per `wrt` (caller passes a zeroed
  gradient buffer for each, since gradients accumulate).
- `optim::build_adamw_step(it, n)` — the fused AdamW kernel `adamw_step_n(w, g, m, v, hp)`; `hp` is
  the `optim::hp::{LR,BETA1,BETA2,EPS,WD,BC1,BC2}` vector (the bias corrections `bc = 1 - beta^t`
  advance with the step).

## Correctness gate
Every VJP rule is **finite-difference-gated**. The scalar rules run forward + backward in **f64**
(via `mercury_interp::run_kernel_f64`, added for this) so ε-noise stays far below tolerance, and
match a closed form to <1e-9. The tensor rules run in **f32** (the kernels are f32) against both a
finite difference and a **tight f64 closed form** computed from the inputs — the closed form is the
real gate; relu/norm data is chosen away from kinks so the difference stays valid. End-to-end:
unrolled and two-layer kernel MLPs train (SGD and the fused AdamW) with the loss asserted to fall
monotonically. A differentiable value whose op has no rule, an unrecognized buffer-writing call, and
a second overwrite-style contribution to a buffer are all **hard errors** — never a silent wrong
gradient. (For reductions a wrong gradient is a miscompile under the project's two laws.)

## Connects to
Upstream: `mercury_mir` (the IR it consumes and builds), `mercury_span` (`Symbol`/`Interner`).
Dev-only: `mercury_interp` (the gate runs forward + backward on the oracle). The backward functions
emit calls to the same `mercury_*` runtime kernels the forward uses, and those are exactly the ops
the GPU `Accelerator` already offloads — so a backward pass run through the GPU offload path moves its
matmul / activation / reduction / norm work to the device with no new kernels.

## Gotchas
- Input must be **single-block SSA** (run `-O1` to mem2reg + simplify-cfg first): the engine assumes
  the only memory traffic is buffer (`Ptr`-param) reads/writes; scalar locals must already be SSA.
- The loss must be **both** `ret`-ed (seeds adjoint 1) **and** stored to an output buffer (so the FD
  gate can read it via `run_kernel_f32`/`_f64`). Stores are skipped in the reverse pass (output sink).
- Two AdamW kernels for different-sized tensors must be distinct functions — the kernel bakes `n` in,
  so the name encodes it (`adamw_step_{n}`); a shared name would alias in a `Program` and overrun the
  smaller buffer (a real bug that was caught here).
- Tensor buffers and kernels are **f32**; the synthesized loops and emitted calls are f32. The scalar
  core is type-generic (gated in f64).
- LayerNorm/RMSNorm backward recompute the row statistics from `x`, so the forward norm must **not**
  be in-place (it would overwrite `x` with `y`).
- Op codes (`VM_*`, `RED_*`, `NORM_*`, `VE_*`) are mirrored from `mercury_runtime`; keep them in sync.

## Status / frontier
Done & gated: the full scalar core; the tensor tape (matmul, relu/sigmoid/tanh activations, sum/SSD/
dot reductions, residual, softmax/LayerNorm/RMSNorm); the fused AdamW kernel; and end-to-end MLP
training with SGD and AdamW (loss decreases). **Frontier (M8):** a GPU-resident training step that
beats PyTorch eager. The pieces in place: the backward emits GPU-offloadable kernels, and AdamW is a
single fused kernel. Remaining: lower the synthesized loops (transpose / activation-backward / the
optimizer) to GPU kernels so the whole step stays device-resident, add the attention (FA2 dQ/dK/dV)
backward against the flash forward, and measure tokens/s same-run vs PyTorch eager. Until measured,
no throughput is claimed.
