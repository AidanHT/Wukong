# `wukong_ast` modules

| file | exports |
|---|---|
| `lib.rs` | `NodeId`, `Ident`, `Path`, `Module`; flat re-exports below |
| `expr.rs` | `Expr`/`ExprKind`, `UnOp`/`BinOp`/`AssignOp`, `BinOp::fold_const_len` |
| `stmt.rs` | `Block`, `Stmt`/`StmtKind`, `ForIter`, `Pattern`/`PatKind` |
| `ty.rs` | `TypeExpr`/`TypeKind`, tensor `Dim`/`DimKind`, `Layout` |
| `item.rs` | `Item`/`ItemKind`, `FnDecl`, `Param`, generics, `Attr`/`AttrArg` |
| `print.rs` | `print_module`, `print_expr`, `print_type` (`pub mod print`) |
