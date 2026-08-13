# wukong_sema

Semantic analysis for Wukong `.wk` programs: **name resolution**, type inference/checking, and
**shape checking** — tensor dimensions live in the type system, so a shape mismatch at a call site or
an index is a compile error, not a runtime crash. This is real compile-time shape safety, not a kernel
compiler; recognized kernels are lowered later in `wukong_mir_build` and executed by CPU or GPU
backends.

Runs after the parser, before `wukong_mir_build`. Annotates the AST through side tables keyed by
`NodeId` and **never rewrites the tree** (except re-typing an adapted literal's side-table entry).

The checker is deliberately **lenient** about constructs it does not model (builtin methods,
`f32x8::load`, `row_ptr`): those yield `Ty::Unknown`, which unifies with anything. Errors are
reserved for what can be proved wrong.

## Pipeline fit

```
parse → wukong_sema::check → wukong_mir_build::lower_program → opt → backends
```

`wukong_driver::compile` refuses `mir_build` if any returned diagnostic `is_error()`. `SemaResult`
(`types`, `defs`, `consts`) is the contract `mir_build` reads.

## Entry point

`check(module, &Interner) -> (SemaResult, Vec<Diagnostic>)` at `src/lib.rs:316`.

Load-bearing stage order: `collect` → `check_recursive_types` → `check_recursive_consts` → set
`checking_bodies = true` → `recheck_item_signatures` → `check_bodies`.

## Key files

| region | file:line | what |
|---|---|---|
| entry | `src/lib.rs:316` | `check` |
| const length fold | `src/lib.rs:1030` | `eval_usize_depth` (u64) |
| array length gate | `src/lib.rs:956` | `array_len` |
| reserved kernel prefix | `src/lib.rs:759` | `wukong_` → `E0300` |
| scope stacks | `src/lib.rs:1524` | `push_scope` / `pop_scope` |
| shape continuation | `src/shape.rs:147` | `type_call` |
| call unification | `src/shape.rs:451` | `check_fn_call` |
| unify + declared names | `src/shape.rs:557` | `unify_decl` |
| dim binding | `src/shape.rs:878` | `unify_dim` |
| shared const fold | `wukong_ast::expr.rs:196` | `BinOp::fold_const_len` |

## Diagnostic codes owned

Sema emits every `E03xx` / `E04xx` / `E05xx` except `E0305` (unresolved import — driver's loader).

| band | codes | examples |
|---|---|---|
| `E03xx` resolution | `E0300`–`E0304` | duplicate def, name not found, immutability |
| `E04xx` types | `E0401`–`E0405` | type mismatch, recursive types/consts, non-exhaustive `match` |
| `E05xx` shapes | `E0501`–`E0504` | rank/index mismatch, dim/layout mismatch, arity, unknown dim |

Prose lives in `wukong_diag/src/catalog.rs` — codes are stable; never renumber.

The `E05xx` primary messages **name what disagrees**: a dimension mismatch names the dim and both
values (`` dimension mismatch on `K` (axis 0): expected 512, found 513 ``), taking the name from the
callee's declared signature — which is why `unify_decl` carries the pre-substitution parameter type
next to the substituted one. Four headlines are asserted verbatim by
`crates/wukongc/tests/fail.rs::shape_diagnostics_name_the_dim_in_json`, so a reword there is a gate
failure, not a silent regression.

## `eval_usize_depth` vs `mir_build::const_usize_depth`

Compile-time integer evaluation is **mirrored** in `mir_build` and must not drift:

- sema: `eval_usize_depth` folds in **u64** (`src/lib.rs:1030`)
- mir_build: `const_usize_depth` narrows to **u32** for slot sizes (`wukong_mir_build/src/lib.rs:20462`)
- both fold binary ops through shared `wukong_ast::BinOp::fold_const_len`

**LANDMINE — operand bound is load-bearing**, not just the result bound: `fold_const_len` returns the
`0` sentinel when either operand or the result exceeds `MAX_CONST_ARRAY_LEN` (`u32::MAX`). Without
that, `[i32; 2-5]` folded to `2^64-3` in sema while `mir_build` allocated the low 32 bits. `array_len`
separately rejects length above `u32::MAX` (`E0401`).

## Unification modes (`unify`, `rigid` flag)

- `rigid == false` — **call sites only**: callee dim vars are inference variables.
- `rigid == true` — **body checks**: dims match by identity (`dims_equal`); stops a generic function
  lying about its return shape.
- `Dim::Dynamic` (`?`) is the lenient escape hatch in both modes.
- `unify_dim` refuses `N := Dynamic` and `Const` vs `Var` on the call path.

## Landmines

- **`check_fn` / `check_const` reset `scopes` directly** — any new per-scope stack must be reset in
  `push_scope`, `check_fn`, and `check_const` or checks silently never fire.
- **`FnSig.generics` is ORDERED from the AST** — building from a `HashSet` made turbofish pairing
  nondeterministic.
- **Literal defaults** (`int_lit_scalar` defaults `i32`, widening to `i64`/`u64`) — never re-default
  to fixed `i32`; both backends agree on the wrong value and gates stay green.
- **Shared builtin tables** (`heap_alloc_elem`, `file_io_elem`, `is_now_ns`) — `mir_build` reads the
  same tables; user fn shadows builtin.

## Testing

```sh
cargo test -p wukong_sema --lib
```

Inline `#[cfg(test)]` tests parse real source via `wukong_parser` (dev-dependency). New rejections
want a `tests/fail/*.wk` fixture with `// EXPECT-CODE:`; new acceptances want `tests/run/*.wk`.
