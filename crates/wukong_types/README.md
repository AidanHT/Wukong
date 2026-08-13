# wukong_types

The semantic type vocabulary shared by sema, `mir_build`, and every backend. Wukong is a language for
`.wk` tensor kernels — this crate defines what a type *means* (scalar, shape, layout) and is the
**layout authority**: the one place that decides how many bytes a type occupies and how it is aligned.
Shape checking lives in `wukong_sema`; GPU is a backend, not what this crate is about.

Pure leaf crate: only dependency is `wukong_span` (`Symbol`, `Interner`).

## Pipeline fit

```
parse → wukong_sema (types + shapes) → wukong_mir_build (lowering uses sizes/aligns) → backends
```

Sema lowers AST type syntax into these `Ty`/`Shape`/`Dim` values and unifies them. `mir_build` resolves
`Ty::Named` through the sema def registry, then **delegates** to `Ty::size_of` / `Ty::align_of` for
everything else (`ty_size` / `ty_align` / `aggregate_layout` in `wukong_mir_build`). Two independent
layout computations is exactly the desync that corrupts stack slots.

## Key files

| region | file | what |
|---|---|---|
| entire crate | `src/lib.rs` | `Scalar`, `Dim`, `Shape`, `Layout`, `Ty`, layout helpers, unit tests |
| layout authority | `src/lib.rs:226` | `Ty::size_of` |
| layout authority | `src/lib.rs:255` | `Ty::align_of` |
| tuple layout | `src/lib.rs:282` | `Ty::tuple_offsets` |
| vector align pin | `src/lib.rs:411` | `align_of_satisfies_the_mir_build_round_up_mirror` test |

## The vocabulary

- **`Scalar`** — 16 primitives (`I8`…`Char`). `from_name`/`name` round-trip source spelling. `Char` is
  4 bytes, unsigned, lowers to 32-bit int in MIR. `is_int()` excludes `Bool`.
- **`Dim`** — one tensor extent: `Const(u64)`, `Var(Symbol)`, `Dynamic` (`?`). Unification lives in
  sema, never here.
- **`Shape`** — ordered `Vec<Dim>`. `symbolic_dims()` returns `Var` dims in axis order (repeats
  preserved) — source of truth for hidden-runtime-dim ABI.
- **`Layout`** — `Contiguous`, `ColMajor`, `Strided`, `Tiled(Vec<u64>)`. Part of the type; sema
  reports layout mismatch as `E0502`.
- **`Ty`** — `Scalar`, `Unit`, `Ptr`/`Ref`, `Slice`, `Array`, `Tuple`, `Vector`, `Tensor`, `Named`,
  `Fn`, `Unknown`, `Error`. `Unknown` unifies leniently; `Error` is poison.

## Layout rules (call these, do not recompute)

- `size_of` / `align_of` answer for scalars, `Unit` (0/1), `Ptr`/`Ref` (8/8), `Vector`, `Slice`,
  `Array`, `Tuple`. `None` for `Tensor`, `Named`, `Fn`, `Unknown`, `Error`, or unsized aggregates.
- **Slice is a sized 16-byte aggregate** `(data: *T, len: usize)` — do not "fix" to `None`.
- **64-bit target assumed**: `Usize`/`Isize` are 8 bytes; every `Ptr`/`Ref` is hard-coded 8/8.

## Landmines

- **Vector alignment is an alignment, not its size** (`align_of` for `Vector` uses
  `(elem.size() * lanes).next_power_of_two()`). `f32x3` parses today; returning `12` would hand out a
  non-power-of-two "alignment" and make `mir_build`'s bitmask `round_up` disagree silently.
- **`Unknown` vs `Error`** — distinct on purpose; do not tighten unification without false positives.
- **`Named` carries only a symbol** — `size_of`/`align_of` return `None`; registry resolution is
  `mir_build`'s job.

## Testing

```sh
cargo test -p wukong_types --lib
```

Nine inline unit tests (layout pins and scalar round-trips). Aggregate/tuple fixtures in `tests/run`
exercise layout end-to-end via the differential gate.
