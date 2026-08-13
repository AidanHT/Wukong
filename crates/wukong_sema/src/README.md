# `src/` module index

| file | contents |
|---|---|
| `lib.rs` | `Sema` struct, `check` entry, `collect` / body passes, type lowering, literals, exhaustiveness, `eval_usize_depth`, `array_len`, builtin tables, inline tests |
| `shape.rs` | `impl Sema` continuation: `type_call`, `check_fn_call`, `unify` / `unify_dim`, `type_index`, `infer_type_generics`, `apply_subst`, `dims_equal` |

`shape.rs` is not a separate crate module tree — it extends `Sema` from `lib.rs`.
