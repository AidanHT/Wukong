# wukong_fmt

The Wukong source formatter behind `wukongc fmt`: an AST pretty-printer that keeps comments and
**refuses rather than change a program's meaning**. Depends on `wukong_span`, `wukong_diag`,
`wukong_lexer` (for `tokenize_with_comments`), `wukong_ast` and `wukong_parser`; no crates.io
dependencies. Used by `wukongc` (`crates/wukongc/src/fmt_cmd.rs`) and by `wukong_driver`, whose
`--emit=doc` stage is `doc::document`.

**Place in pipeline:** none — it is a tool beside the compiler. It lexes and parses exactly as the
driver does, prints, and then re-lexes and re-parses its own output to prove the result means the
same program.

## Entry points

- `format_source(src, SourceId, &FmtOptions) -> Result<String, FmtError>` — the formatted text
  (identical to `src` when the file is already formatted).
- `FmtOptions { max_width }` — default 100.
- `doc::document(src, SourceId, file_name, &FmtOptions) -> Result<String, FmtError>` — the file's
  documentation as Markdown (`--emit=doc`): title, `//!` docs, then every item in source order with
  its signature (this crate's printer, with no comments and no blank lines) and `///` docs; fields,
  variants, methods and `extern` declarations listed under their item.
- `FmtError::Syntax(Vec<Diagnostic>)` — the lexer's or parser's errors; render them as a compile
  would. `FmtError::Unsupported(String)` — the formatter cannot prove its output means the same
  program; the message names the failing self-check law and some context. In both cases the
  caller must leave the file untouched.

## Key files

| file | role |
|---|---|
| `src/lib.rs` | `format_source`, `FmtOptions`, `FmtError`; line endings (CRLF if the input's first line break is CRLF) and a leading BOM |
| `src/printer.rs` | the printer: exhaustive over every AST enum (no `_ =>` arm), source-preserved spellings, desugar reconstruction, minimal parentheses, wrapping |
| `src/comments.rs` | comment placement (one cursor, own-line elements, trailing comments) and blank lines |
| `src/doc.rs` | `--emit=doc`: the Markdown page; signatures through `Printer::fn_signature` / `Printer::item` on a comment-free `Src`, list entries through the flat printer |
| `src/doc_tests.rs` | hand-written pages, and doc comments through the formatter (they stay on what they document; law (b) sees one that moved) |
| `src/check.rs` | the four self-check laws: (a) clean re-parse, no lint; (b) identical AST, NodeIds included, spans masked; (c) same comments; (d) same identifier, literal, keyword and `@` tokens (every `TokenKind` classified exhaustively) |
| `src/tests.rs` | golden input/output pairs for every rule, each checked for idempotence; refusals; the self-check's own teeth |
| `tests/corpus.rs` | every `.wk` in the corpus formats with zero refusals and is a fixed point; comments inserted at token boundaries never cause a refusal |

## Invariants / landmines

- **Adding syntax means extending this crate in the same commit.** A new AST variant fails the
  build here; a new fixture the printer cannot reprint fails `tests/corpus.rs`.
- **Never relax the self-check to let a file through.** Fix the printer, or leave the file refused.
  The NodeId comparison is what catches a reprint that re-parses into a differently shaped tree
  (a dropped type parenthesis, a `{ e }` body for `= e`, a re-associated operator).
- **A desugared construct is printed from its shape.** `impl` methods, receivers, `Self`, `if let`
  (with chains), `while let`, `let … else`, a local `const`, `defer x op= v`, `= e` bodies, `where`
  bounds and range indexes (`$slice`/`$slice_from`) are recognized as the parser builds them; if
  the parser's desugaring changes, the printer must change with it (the self-check refuses until
  it does).
- **The parser keeps parsing after a block-like expression** (unlike Rust), so a block-like
  statement gets a `;` when the next statement starts with `( [ - * & ! ~ . < > = + / % ^ |` or
  `as`, or when it is the last statement of a block with no tail.

## Testing

```sh
cargo test -p wukong_fmt                       # goldens + refusals + the corpus suites
cargo test -p wukongc --test fmt               # the CLI contract and the MIR-identity gate
```

See `docs/internals.md`, *Formatter*, for the full rule set.
