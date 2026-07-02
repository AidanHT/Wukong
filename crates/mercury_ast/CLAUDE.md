# mercury_ast

The Mercury abstract syntax tree: a data-only crate the parser produces and sema annotates via side tables (the tree is never rewritten).

## Layout
- `src/lib.rs` — crate root; re-exports all modules. Defines `NodeId`, `Ident`, `Path`, `Module`.
- `src/expr.rs` — `Expr`/`ExprKind`; operator enums `UnOp`, `BinOp`, `AssignOp` (each has `.glyph()`); `FieldInit`, `MatchArm`.
- `src/stmt.rs` — `Block`, `Stmt`/`StmtKind`, `ForIter`, `Pattern`/`PatKind`.
- `src/ty.rs` — type syntax: `TypeExpr`/`TypeKind`, tensor `Dim`/`DimKind`, `Layout`.
- `src/item.rs` — top-level `Item`/`ItemKind` (Fn/Struct/Enum/Const/Import/Extern), `GenericParam`, `Param`, `Attr`/`AttrArg`/`AttrVal`.
- `src/print.rs` — deterministic indented pretty-printer (largest file; holds the crate's only unit test).

## Key types & entry points
- `NodeId` (`src/lib.rs`) — `pub u32` newtype keying every node and all semantic side tables. `NodeId::DUMMY` (`u32::MAX`) marks synthesized nodes. `Debug` prints `n{0}`.
- `Module` (`src/lib.rs`) — root of a parsed file: optional `module a.b;` header (`name: Option<Path>`) + `Vec<Item>`.
- `ExprKind`, `StmtKind`, `TypeKind`, `ItemKind` — the four central recursive enums. Each wrapper struct (`Expr`, `Stmt`, `TypeExpr`, `Item`, `Block`, `Pattern`, `Param`) carries `id: NodeId`, a `kind`, and `span`.
- `print_module` / `print_expr` / `print_type` (`src/print.rs`) — public render entry points for `--emit=ast` and parser/snapshot tests; all three take `&Interner` to resolve `Symbol`s.

## Connects to
Upstream (depends on): `mercury_span` only (`Span`, `Symbol`, `Interner`). Downstream (consumers): `mercury_parser` builds it; `mercury_sema` reads it and annotates via `NodeId`-keyed tables; `mercury_mir_build` lowers the typed AST to MIR. `TypeKind` is resolved to `mercury_types::Type` in sema.

## Gotchas
- Side-table model: sema must NOT mutate the tree — it stores results keyed by `NodeId`. Allocate a fresh `NodeId` for any synthesized node (or use `DUMMY`); never reuse one.
- Literals are stored as the raw source `Symbol` (incl. suffix and quotes) and parsed later in sema, not here. Pre-parsed exceptions: `ExprKind::Bool(bool)`, `TupleField.index: u32`, `TypeKind::Vector.lanes: u32`, `DimKind::Int(u64)`, `Layout::Tiled(Vec<u64>)`. Note `TypeKind::Int` (a const-generic/dim integer arg) stays a raw `Symbol`, unlike `DimKind::Int`.
- A "dimension variable" and a type variable share the same surface form: `TypeKind::Path` and `GenericParamKind::Type(Ident)` cover both; the role is decided later in sema.
- The pretty-printer is intentionally lossy — for human/test inspection, not round-tripping. Enum variants print name only (payload dropped), extern fn signatures shrink to `fn name`, `Layout::Tiled` prints `.tiled(..)`, and `expr_inline` (array lengths / const generics) falls back to `<expr>` for anything beyond literals/paths/simple un/binary ops.
- `Attr::args()` just returns the `args` slice — a redundant getter; read `attr.args` directly.
- `Path::first()` indexes `segments[0]` and panics on an empty path (the parser guarantees >=1 segment); `Path::is_single()` checks for exactly one segment.
