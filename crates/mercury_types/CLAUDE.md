# mercury_types

The semantic type vocabulary shared by sema and MIR: scalars, SIMD vectors, shape-typed tensors, and memory layouts. Pure leaf crate — depends only on `mercury_span` for interned symbols.

## Layout
- `src/lib.rs` — everything: `Scalar`, `Dim`, `Shape`, `Layout`, `Ty`, their inherent impls, the `round_up`/`dim_str` helpers, and unit tests.

## Key types & entry points
- `Ty` (`src/lib.rs`) — the central enum. Variants: `Scalar`, `Unit`, `Ptr`/`Ref` (both `{mutable, pointee: Box<Ty>}`), `Slice(Box<Ty>)`, `Array{elem,len}`, `Tuple(Vec<Ty>)`, `Vector{elem: Scalar, lanes: u32}` (SIMD), `Tensor{elem: Scalar, shape: Shape, layout: Layout}`, `Named(Symbol)` (struct/enum by interned name), `Fn{params,ret}`, `Unknown`, `Error`. Constructor `Ty::scalar(s)`.
- `Scalar` — the 15 primitive numeric/bool kinds. `from_name`/`name` round-trip the source spelling; `size`/`align`/`is_float`/`is_int`/`is_signed` are the layout/category predicates.
- `Dim` — one tensor extent: `Const(u64)`, `Var(Symbol)` (bound symbolic), or `Dynamic` (runtime `?`). This three-way split is what enables compile-time shape checking.
- `Shape(pub Vec<Dim>)` — ordered dims; `rank()` is the length.
- `Layout` — tensor physical layout: `Contiguous`, `ColMajor`, `Strided`, `Tiled(Vec<u64>)`.
- `Ty::size_of()` / `Ty::align_of()` — return `Option<u64>`; `None` for unsized/deferred types (`Slice`, `Tensor`, `Named`, `Fn`, `Unknown`, `Error`). Tuple layout does real field padding via `round_up`.
- `Ty::display(&Interner)` — diagnostic rendering; needs the interner to resolve `Named`/`Dim::Var` symbols.

## Connects to
Upstream: `mercury_span` (`Symbol`, `Interner`). Downstream: sema (name res + type/shape check) and `mercury_mir` consume `Ty`/`Shape`; diagnostics use `display`.

## Gotchas
- Layout assumes a **64-bit target**: `Scalar::size` gives `Usize`/`Isize` 8 bytes, and `Ty::size_of`/`align_of` hard-code 8 for all `Ptr`/`Ref`. No target abstraction.
- `Vector` alignment equals its full size (`elem.size() * lanes`), not the element align — natural SIMD alignment.
- `is_int()` excludes `Bool`; `is_numeric()` (on `Ty`) covers scalar numbers and numeric SIMD vectors only — not tensors.
- `Unknown` and `Error` are distinct on purpose: `Unknown` is lenient inference for unmodeled builtins (does **not** error); `Error` is the poison value that suppresses cascading diagnostics. Check with `is_unknown()`/`is_error()`.
- `Named` carries only the interned symbol — no field/variant info lives here; `size_of`/`align_of` return `None` for it.
- `Dim::Var` equality is by `Symbol`; symbolic shape unification lives in sema, not here — this crate only stores dims.
