# wukong_runtime

CPU microkernel library for the Wukong compiler. **Wukong is the language** (`.wk` programs with
compile-time shape safety); this crate is an implementation detail — the `extern "C"` kernels that
`wukong_mir_build` recognizers dispatch to, and that both the interpreter and the Cranelift native
backend link against.

## Role in the pipeline

```
.wk source
  → wukong_mir_build recognizers emit kernel calls
  → wukong_interp (oracle)  ─┐
  → wukong_codegen_cranelift  ─┴─ both call the same wukong_runtime symbols
```

The interpreter calls these functions directly. Cranelift binds them as JIT symbols for `--run`, or
as undefined imports that `--emit=obj|exe` resolves at link time. Whatever the interpreter computes
via these kernels **is** the language spec for recognized ops.

## What lives here

- **~150 `extern "C"` AVX2/FMA microkernels** — each family has a scalar twin used as the reference
  in unit tests. On x86-64 hosts without AVX2+FMA3 the SIMD path is skipped and the scalar twin runs.
- **Scalar twins** — every AVX2 kernel must agree **bit-exactly** with its scalar sibling, including
  on NaN, ±0, and ties. `_mm256_max_ps` returns `b` on a tie or NaN, so the twin is
  `(a > b) ? a : b`, not `f32::max`.
- **Thread pool** — rayon-based scheduling for `_parallel` kernel variants and
  `wukong_parallel_for` (the native lowering of `@parallel for`). A private physical-core GEMM pool
  and a 16 MiB-stack global/region pool back the parallel paths.
- **`Arena`** — bump allocator (reference surface; kernels use thread-local `Vec` scratch instead).
- **Heap / I/O / timing shims** — `wukong_rt_alloc`/`free`, headerless little-endian file I/O, and
  `wukong_now_ns`, shared with the native backend's `rt_*` print/assert shims.
- **Low-precision helpers** — shared `bf16`/`f16` round-trip definitions (`half` crate for f16) so
  interpreter and native agree bit-for-bit.

## Dependencies

Leaf crate — no workspace dependencies. External crates only:

| crate | purpose |
|---|---|
| `rayon` | parallel `_parallel` kernels and `wukong_parallel_for` |
| `half` | IEEE f16 scalar conversion (matches F16C `vcvtph2ps` path) |
| `num_cpus` | physical core count for the private GEMM thread pool |

## Performance note

Debug builds make the hand-written kernels look far slower than the JIT-compiled loops they replace.
Kernel A/B timing requires a `--release` build; discard the first warm-up run.

## Layout

| path | contents |
|---|---|
| [`src/`](src/) | kernel modules + `lib.rs` (re-exports, pool, arena, alloc/I/O) |
| [`examples/`](examples/) | dev-only GEMM scaling/variance probes (not shipped benches) |

## Tests

`cargo test -p wukong_runtime --lib` — scalar-twin parity, arena/parallel_for reference behaviour,
bf16/f16 round-trip, heap alloc, file I/O contract. Per-kernel AVX2-vs-scalar tables live in each
kernel module's `#[cfg(test)]` block (the `simd_equals_scalar_twin` pattern).
