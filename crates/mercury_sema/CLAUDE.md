# mercury_sema

Semantic analysis: name resolution, type checking, and SHAPE checking. Runs after the parser (typed AST) and before `mir_build`. Annotates the AST through side tables keyed by `NodeId`; it never rewrites the tree.

## Layout
- `src/lib.rs` — collection pass, type lowering, body/stmt/expr type checking, the `Sema` struct, public API, and inline tests.
- `src/shape.rs` — `impl Sema` continuation: `type_call`/`type_index` plus call-site dimension unification (the headline shape feature).

## Key types & entry points
- `check(module, interner) -> (SemaResult, Vec<Diagnostic>)` (`src/lib.rs`) — sole entry point. Runs `collect` (register top-level defs) then `check_bodies` (type fn bodies).
- `SemaResult` (`src/lib.rs`) — `{ types: HashMap<NodeId, Ty>, defs: DefMap }`. Per-expression types feed `mir_build`.
- `Sema<'a>` (`src/lib.rs`) — checker state: `defs`, `types`, lexical `scopes: Vec<HashMap<Symbol, Ty>>`, current `generics: HashSet<Symbol>`, `ret_ty`, `diags`, `interner`. Methods split across both files (`shape.rs` is also `impl Sema`).
- `DefMap` / `Def` / `DefKind` / `FnSig` (`src/lib.rs`) — resolved top-level defs (`Fn`/`Const`/`Struct`/`Enum`), looked up by `Symbol` via `DefMap::lookup`.
- `lower_type` (`src/lib.rs`) — AST `TypeExpr` -> `mercury_types::Ty`. Maps `Tensor`/`Vector`/dims; an unknown path name becomes `Ty::Named` (generic or forward ref).
- `type_call` / `check_fn_call` / `unify` / `unify_dim` / `apply_subst` (`src/shape.rs`) — the shape engine: binds dim vars from turbofish or argument inference, reports conflicts. `apply_subst` is a free fn; `type_call`/`type_index` are `pub(crate)`, the rest are private.

## Connects to
Upstream: `mercury_ast` (input tree), `mercury_types` (`Ty`/`Dim`/`Shape`/`Scalar`/`Layout`), `mercury_span` (`Interner`/`Symbol`/`Span`), `mercury_diag` (`Diagnostic`). Dev-only: `mercury_parser` (tests). Downstream: `mir_build` consumes `SemaResult.types`.

## Diagnostic codes
- `E0300` duplicate name; `E0301` unresolved name; `E0302` non-scalar tensor/vector element.
- `E0401` scalar/vector/let type mismatch; `E0501` tensor rank / index-count mismatch; `E0502` dimension-value or tensor-element-type conflict; `E0503` wrong argument or generic-arg count.

## Gotchas
- Deliberately LENIENT: unmodeled constructs (methods, fields, struct literals, multi-segment paths, builtins like `f32x8::load`) yield `Ty::Unknown`, which unifies with anything. `join`/`compatible`/`unify` all treat `Unknown`/`Error` as compatible — errors fire only when certain. Don't "tighten" these without expecting false positives.
- Type lowering is intentionally partial: `TypeKind::Int` lowers to `Ty::Error`; `eval_usize`/`parse_dim_text` read only a leading-digit prefix and fall back to `0` on anything non-trivial.
- Numeric literals: unsuffixed int/float literals adapt to a `let` annotation (`let_compatible`), but a bare `ExprKind::Int` otherwise defaults to `Scalar::I32` and float to `Scalar::F32` (`int_lit_scalar`/`float_lit_scalar`).
- `resolve_value`: a generic name used as a value resolves to `Ty::Scalar(Scalar::Usize)`; structs/enums and scalar/`Tensor`/vector type names used as namespaces resolve to `Ty::Unknown`.
- `self.generics` is set/cleared around each item; it is shared mutable state, so `lower_type` behaves differently inside vs. outside an item's generic scope.
- Tests live inline in `src/lib.rs` (`#[cfg(test)]`), parsing real source via `mercury_parser`; there is no `tests/` dir.
