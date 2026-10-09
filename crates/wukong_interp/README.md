# wukong_interp

A from-scratch tree-walking MIR interpreter. Wukong is the language; this crate **executes** MIR
produced from user `.wk` programs. It is the default `--backend=interp` path and **the differential
oracle** — whatever this crate computes *is* the language's runtime specification. A divergence
against native is never "the oracle is loose"; it is a bug in one of the two.

Runtime deps: `wukong_mir`, `wukong_span`, `wukong_backend`, `wukong_runtime`. Recognized kernel
arms call the *identical* runtime microkernel the native backend links, marshalling slot memory into
real buffers — wider lanes or reassociated accumulation cannot make the oracle disagree.

## Entry points

- `run(program, entry, interner) -> Result<i64, String>`
- `run_with_output(...) -> Result<(i64, Vec<u8>), String>` — exit code plus captured stdout.
- `run_with_output_accel(..., accel: &mut dyn Accelerator)` — `--backend=gpu` path: same tree-walk,
  recognized kernel calls offloaded.
- `run_kernel_f32` / `run_kernel_f32_accel` / `run_kernel_f64` / `run_kernel_i8` — typed kernel-entry
  ABI for the full-buffer fuzzer and `wukong_autodiff`'s finite-difference gate.

Calls are **stackless**: `run_function` keeps an explicit frame stack, so a Wukong call never
recurses on the host stack and every entry point runs on its caller's thread (no worker thread).

`Interpreter` implements `wukong_backend::Backend` (returns `Artifact::Executed`), but the driver
calls `run_with_output` directly — see `wukong_backend`.

## Slot-indexed memory

`memory: Vec<Value>` is flat — **one slot per scalar leaf**, not bytes. `Alloca` pushes typed zeros;
`Value::Ptr(usize)` is a slot index; `Gep` computes `base + index * slot_count(elem)`. Memory is
reused (`SlotReuse`): a returning frame cuts it back to its entry length, never below the end of
the last non-alloca growth (heap/scratch block, static string, kernel-entry buffer), and
`wukong_rt_free` puts a live `alloc_*` block on a size-bucketed free list that the next fitting
allocation takes back re-zeroed.

**Address 0 is the reserved null (`NULL_SLOT`).** Every public entry point starts its memory with
one never-allocated sentinel slot (`NULL_SENTINEL`), so no `alloca`, heap block, static or
kernel-entry buffer ever lands at slot 0 and `Value::Ptr(0)` means what `inttoptr 0` means natively.
A load or store through it is a `null pointer dereference` diagnostic. That is what makes the
`PtrToInt(p) != 0` pointer-condition lowering (reverted once at 0900352, when slot 0 was the first
live allocation) and mem2reg's null for a read-before-write pointer slot sound. **Keep it:** a new
entry point that builds an `Interp` must start from `null_reserved_memory()`, or the first
allocation of its run reads as null again. (The unit tests that build an `Interp` by hand with an
empty `memory` keep their old addressing; the null checks fire only when slot 0 holds the
sentinel.)

## `Accelerator` trait — `--backend=gpu` hook seam

Five methods, each `-> Option<Result<…>>` where `None` means fall back to the CPU kernel. Default
impls all return `None`, so with no accelerator the oracle path is byte-for-byte unchanged:

- `sgemm_nt` — `C = A·Bᵀ` with `beta == 0`
- `vmath` — elementwise activations (`op` is runtime code)
- `norm` — fused row norm (softmax, LayerNorm, RMSNorm, …)
- `sreduce` — deterministic scalar reduction (sum / dot / max / …)
- `sgemm_nt_epi` — fused-epilogue Linear `C = act(A·Bᵀ + bias)`

The GPU implementation lives in `wukong_driver/src/gpu_accel.rs`. Adding a method obligates that
file. CPU↔GPU agreement is a **tolerance** gate (`c·√K·ε`), not bit-exactness — that is not this
crate's job to claim.

## Cross-crate coupling

- Every recognized-kernel symbol `wukong_mir_build` emits needs an arm in `intrinsic`. Unknown callees
  fail immediately on the default backend.
- Kernel arms must call the **serial** runtime kernel and match the `_parallel` name they also
  match. Mirror serial kernel arms in `intrinsic` when adding a recognizer.
- Marshalling order: read **all** inputs before writing any output (in-place `x == out` correctness).
- Null optional pointers arrive as `Value::Int(0)` — distinguished by **variant**, not value.

## Semantics this crate pins

Changing these re-specifies the language:

- Integer results masked to declared MIR width; sub-`f64` floats rounded to `f32`. `Op::Load` is
  exempt from the integer mask (byte-wise aggregate copies).
- Vector lanes truncate to lane width via `mask_lane`.
- Non-positive kernel extent is a no-op (`dim!` macro), not an abort.
- Resource limits report diagnostics: heap exhaustion, 100M-step guard (per frame), call-depth
  ceiling (`MAX_CALL_DEPTH` = 300_000 in every build profile, `WUKONG_INTERP_MAX_DEPTH` overrides).
- Scalar arithmetic matched to Cranelift: wrapping int ops; div/rem by zero → `0`; shift counts
  masked; `RoundMode::Nearest` is `round_ties_even`; bf16/f16 live as rounded `f32`.
- `wukong_embedding_f32` gathers read live memory (no snapshot when `out` aliases `weight`).

## Testing

```sh
cargo test -p wukong_interp --lib
```

19 tests including `integer_vector_lanes_wrap_at_lane_width`,
`non_positive_kernel_extent_is_a_no_op`, `deep_recursion_does_not_overflow_oracle`.

The differential gate:

```sh
cargo test -p wukongc --test run
```

`native_matches_interpreter` requires identical stdout **and** exit code at every `-O` level.
Interp vs native is bit-exact. GPU is tolerance — not proven here.

Errors are plain `String`s with stable substrings (`call-depth limit exceeded`, `step limit exceeded`,
`heap exhausted`). This crate owns no `E0xxx`/`C0xxx` codes.

## Layout

See [`src/README.md`](src/README.md).
