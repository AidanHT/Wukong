# wukong_mir_build

Lowers the type-checked Wukong `.wk` AST into MIR. Sits between `wukong_sema` (shape checking is
already done) and `wukong_opt` / the backends. GPU is a backend — this crate emits MIR and calls
`wukong_runtime` kernels; it does not compile PTX.

Also owns **every kernel recognizer**: loop nests rewritten into a `wukong_runtime` symbol call, plus
`@parallel` outlining (whole-function and mid-function regions), type-generic monomorphization, and
the 128-bit / 256-bit vectorizer paths.

Strategy: **alloca-per-local** (`Load`/`Store`); SSA recovered later by `mem2reg` in `wukong_opt`.

## Pipeline fit

```
wukong_sema::check → wukong_mir_build::lower_program → wukong_opt::optimize → backends
```

Public API: `lower_program` (and `is_intrinsic`). Called from `wukong_driver`, benches, and every
backend's test harness.

## Entry points

| symbol | file:line | role |
|---|---|---|
| `lower_program` | `src/lib.rs:880` | sole public lowering entry |
| `is_intrinsic` | `src/lib.rs:27923` | `print` / `println` / `assert` |
| `has_parallel_attr` | `src/lib.rs:1790` | `@parallel` attribute probe |
| `unsupported` → `C0001` | `src/lib.rs:3119` | hard codegen-limit error |
| `ty_size` / `ty_align` | `src/lib.rs:3481` | registry-aware layout; delegates to `wukong_types` |
| `aggregate_layout` | `src/lib.rs:3528` | tuple/struct byte layout |
| `bind_slice` | `src/lib.rs:3245` | marks fat-pointer slots for `kernel_base_ptr` |
| `kernel_base_ptr` | `src/lib.rs:8443` | **every** kernel buffer operand routes here |

`lower_program` runs `tindex::linearize_module` then `canon::canonicalize_module` before recognizers
see the AST.

## Recognizer cascade

Two levels, order matters:

1. **Whole-function** (`lower_program`): `matmul_fn` / `i8matmul_fn`, then `has_parallel_attr &&
   <probe>` interceptors (GEMV, transpose, norms, cumsum, embedding, …) — each runs before the
   generic `@parallel` outliner would chunk the nest.
2. **Statement-level** (`lower_for`, ~line 10012): matmul family → reductions → bias-bcast →
   `try_vectorize_for` → mid-function `@parallel` region → scalar CFG loop.

Recognizers are **GATE-BLIND**: interp and native call the same `wukong_runtime` symbol at every `-O`
level, so `native_matches_interpreter` and opt-invariance cannot see whether a recognizer fired or
whether it is wrong.

Proof a recognizer fired:

```sh
cargo run -q -p wukongc -- --emit=mir -O2 <file.wk>
```

grep for `wukong_`. Fixtures must **print every lane** (not a reduction); expected values come from
scalar loop semantics, not compiler output.

## `@parallel` and `@tile`

- **`@parallel`** — whole-function outliner (`parallel_spec`, `lower_parallel`) and mid-function
  regions (`try_emit_parallel_region`). Legality proofs are conservative; `escapes_parallel_chunk_block`
  declines `return`/`break`/`continue` that escape the parallelized loop.
- **`@tile`** — parsed by `wukong_parser` into `item.attrs` (alongside `@simd`, `@align`, …). This
  crate reads **only** `@parallel` via `has_parallel_attr`; `@tile` is stored on the AST and
  discarded for lowering today. Tensor layout `.tiled(...)` in type syntax is separate (parsed as
  `Layout::Tiled` in `wukong_types`).

## Diagnostic codes owned

| code | file:line | meaning |
|---|---|---|
| `C0001` | `src/lib.rs:3126` | `unsupported` — not yet supported by codegen (hard error) |
| `E0401` | `src/lib.rs:16619` | array initializer length ≠ destination extent |
| `E0401` | `src/lib.rs:17669` | `match` integer literal outside scrutinee range |

(`E0401` in sema is the much larger type-mismatch bucket — these two sites are mir_build-specific
lowering checks.)

## `eval_usize_depth` mirror

`const_usize_expr` / `const_usize_depth` (`src/lib.rs:20462`) must mirror sema's `eval_usize_depth`.
Both fold through `BinOp::fold_const_len`; operand bound clamp to `u32::MAX` is load-bearing (see
`wukong_sema` README).

## Landmines

1. **`kernel_base_ptr`** — every kernel **buffer** operand must use `kernel_base_ptr` /
   `kernel_base_ptrs` / `kernel_operand_ptr`. Slice slots are 16-byte fat pointers; `bind_slice` (not
   plain `bind`) records slice-ness from the **sema** type. `bind` on a slice hands the kernel the
   fat pointer → prints zeros on both backends. Whole-function lowerers (`lower_matmul_fn`, etc.) need
   their own `bind_slice` — no shared path.
2. **Recognizers GATE-BLIND** — see above. Runtime kernel tests and lane-printing fixtures are the
   only real proof.
3. **Pointer truthiness** — `if p` / `while p` lower to `ptrtoint p != 0`, sound because the
   interpreter reserves slot 0 as null (`wukong_interp::NULL_SLOT`); an aggregate, slice, tensor,
   vector or function condition is `E0401` in sema. Slice accesses `s[i]` carry a bounds check
   (`src/bounds.rs`, a dead-end failure block calling `wukong_rt_bounds_fail`) unless the function
   is `@unchecked`; `wukong_opt::bce` keys on exactly that shape.
4. **`mono_type_name` must stay injective** — non-injective mangling merges instances and corrupts ABI.
5. **New `_parallel` recognizer** — must mirror `wukong_autodiff/src/tape.rs` (`Syms`, `is_kernel`,
   `diff_kernel_call`) or `@parallel` backward breaks.
6. **PTX is not here** — but every new kernel symbol must match `wukong_runtime` exactly.

Submodules: `src/canon.rs` (AST canonicalization, plus the recognizer-only value-binding view),
`src/tindex.rs` (tensor index linearization), and the recognizer spelling-coverage modules —
`src/elementwise.rs` (value-binding view probes, the hand-written tanh GELU), `src/ucount.rs`
(`usize` / symbolic-dim counters behind a run-time `end >= 0` test), `src/colacc.rs` (row-outer
column reductions), `src/softmax_oop.rs` (out-of-place softmax / log-softmax), `src/detile.rs`
(hand-tiled GEMM chains, including `min`-clamped and counted tiles behind run-time cover tests, and
the `C`-seeded `C += A·B` form), `src/dispatch.rs` (per-loop `@generic`, `--remarks=dispatch`) —
and `src/minmax.rs` (the integer/float `min`/`max` builtins).

## Testing

```sh
cargo test -p wukong_mir_build --lib
```

Inline unit tests at bottom of `lib.rs` (`@parallel` legality, kernel tables, lowering/verify). Real
coverage: `cargo test -p wukongc --test run` and `wukong_codegen_cranelift` fuzz/differential tests
that call `lower_program` directly.
