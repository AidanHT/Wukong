# `src/` module index

| file | contents |
|---|---|
| `lib.rs` | `Sema` struct, `check` entry, `collect` / body passes, type lowering, literals, exhaustiveness, `eval_usize_depth`, `array_len`, builtin tables, inline tests |
| `shape.rs` | `impl Sema` continuation: `type_call`, `check_fn_call`, `unify` / `unify_dim`, `type_index`, `infer_type_generics`, `apply_subst`, `dims_equal` |
| `attrs.rs` | The closed v1 `@attribute` set (`KNOWN_ATTRS`), placement (`FN_ITEM_ONLY`), and `E0306` |
| `builtins.rs` | Named kernel builtins (`matmul`, `softmax`, `layernorm`, `gelu(x, out)`, `sum`, `sdpa`, …): shape signatures, dimension resolution, `E0409`; records `SemaResult::builtin_calls` |
| `methods.rs` | `impl` method tables, trait conformance (`E0410`), bounds and trait-bound calls, function-typed parameters (`E0411`), unresolved methods (`E0315`); records `SemaResult::methods` |

`shape.rs` is not a separate crate module tree — it extends `Sema` from `lib.rs`.

`attrs.rs` is standalone: it walks the module itself (items, parameters, statements, every nested
block) rather than riding the `Sema` passes, because an attribute names no value and has no type. It
runs FIRST in `check`, so `@bogus` is reported even in a module whose bodies do not resolve. Adding an
attribute to the language starts here — a name absent from `KNOWN_ATTRS` is `E0306`, which is the point
(an unimplemented attribute cannot be used).
