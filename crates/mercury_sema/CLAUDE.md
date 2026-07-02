# mercury_sema

Semantic analysis: name resolution, type checking, and SHAPE checking. Runs after the parser (typed AST) and before `mir_build`. Annotates the AST through side tables keyed by `NodeId`; it never rewrites the tree.

## Layout
- `src/lib.rs` — collection pass, type lowering, body/stmt/expr type checking, the `Sema` struct, public API, and inline tests.
- `src/shape.rs` — `impl Sema` continuation: `type_call`/`type_index` plus call-site dimension unification (the headline shape feature).

## Key types & entry points
- `check(module, interner) -> (SemaResult, Vec<Diagnostic>)` (`src/lib.rs`) — sole entry point. Runs `collect` (register top-level defs) then `check_bodies` (type fn bodies).
- `SemaResult` (`src/lib.rs`) — `{ types: HashMap<NodeId, Ty>, defs: DefMap, consts }`. Per-expression types feed `mir_build`; `consts` records each top-level `const`'s checked initializer expression so `mir_build` can inline a `const` used as a value.
- `Sema<'a>` (`src/lib.rs`) — checker state: `defs`, `types`, lexical `scopes: Vec<HashMap<Symbol, Ty>>`, current `generics: HashSet<Symbol>`, `ret_ty`, `diags`, `interner`. Methods split across both files (`shape.rs` is also `impl Sema`).
- `DefMap` / `Def` / `DefKind` / `FnSig` (`src/lib.rs`) — resolved top-level defs (`Fn`/`Const`/`Struct`/`Enum`), looked up by `Symbol` via `DefMap::lookup`. `DefKind::Enum` carries each variant's computed integer discriminant (explicit `= <int>` via `eval_const_int`, else auto-incrementing from the previous), so a C-style `E::Variant` types as the enum's nominal `Ty::Named` and resolves to a value (checked before the struct-field path).
- `lower_type` (`src/lib.rs`) — AST `TypeExpr` -> `mercury_types::Ty`. Maps `Tensor`/`Vector`/dims; an unknown path name becomes `Ty::Named` (generic or forward ref).
- `type_call` / `check_fn_call` / `unify` / `unify_dim` / `apply_subst` (`src/shape.rs`) — the shape engine: binds dim vars from turbofish or argument inference, reports conflicts. `apply_subst` is a free fn; `type_call`/`type_index` are `pub(crate)`, the rest are private.

## Connects to
Upstream: `mercury_ast` (input tree), `mercury_types` (`Ty`/`Dim`/`Shape`/`Scalar`/`Layout`), `mercury_span` (`Interner`/`Symbol`/`Span`), `mercury_diag` (`Diagnostic`). Dev-only: `mercury_parser` (tests). Downstream: `mir_build` consumes `SemaResult.types`.

## Diagnostic codes
- `E0300` duplicate name; `E0301` unresolved name; `E0302` non-scalar tensor/vector element; `E0303` `break`/`continue` outside a loop (tracked by a `loop_depth` counter; fixes a backend divergence on the out-of-loop trap); `E0304` assigning/mutating an immutable binding — a `let` without `mut` (direct rebind only) or a parameter without `mut` (rebind, or an aggregate projection `p.f`/`p[i]` that, by the by-reference ABI, would mutate the caller). Tracked by `immutable_locals` / `immutable_params` (both 1:1 with `scopes`); a pointer-deref target (`*p`) is exempt.
- `E0401` scalar/vector/let type mismatch; `E0501` tensor rank / index-count mismatch; `E0502` dimension-value or tensor-element-type conflict; `E0503` wrong argument or generic-arg count; `E0504` **unknown tensor dimension** — a dim name that is not a declared generic (nor an integer, `?`, or `const`), e.g. the typo `Tensor[f32, KK]` for `K`; emitted by `lower_dim` with a `nearest_generic` did-you-mean hint, gated on the `checking_bodies` flag so it fires once per site (only in the body pass, after every generic/`const` is known).

## Gotchas
- Deliberately LENIENT: unmodeled constructs (methods, fields, struct literals, multi-segment paths, builtins like `f32x8::load`) yield `Ty::Unknown`, which unifies with anything. `join`/`compatible`/`unify` all treat `Unknown`/`Error` as compatible — errors fire only when certain. Don't "tighten" these without expecting false positives.
- Type lowering is intentionally partial: `TypeKind::Int` lowers to `Ty::Error`; `eval_usize`/`parse_dim_text` read only a leading-digit prefix and fall back to `0` on anything non-trivial.
- Numeric literals: unsuffixed int/float literals adapt to a `let` annotation (`let_compatible`), but a bare `ExprKind::Int` otherwise defaults to `Scalar::I32` and float to `Scalar::F32` (`int_lit_scalar`/`float_lit_scalar`).
- `resolve_value`: a generic name used as a value resolves to `Ty::Scalar(Scalar::Usize)`; structs/enums and scalar/`Tensor`/vector type names used as namespaces resolve to `Ty::Unknown`.
- `self.generics` is a `HashSet` (membership only) set/cleared around each item; it is shared mutable state, so `lower_type` behaves differently inside vs. outside an item's generic scope. **Never collect an ordered list out of it** — `FnSig.generics` (the Vec a turbofish `f::<2, 3>` binds by position) must come from the AST slice via `generic_param_sym`, not `self.generics.iter()`, or the per-process hash order makes generic dim-binding nondeterministic across runs.
- Body shape checks (`check_return_shape`, `check_binop_shapes`) call `unify(..., rigid=true)`: a function's own generic dims are matched by identity (`dims_equal`), never bound — so it can't lie about its return/operand shape. Call-site unification (`check_fn_call`) uses `rigid=false` to *infer* a callee's dims from the arguments. Don't flip these.
- Tests live inline in `src/lib.rs` (`#[cfg(test)]`), parsing real source via `mercury_parser`; there is no `tests/` dir.
