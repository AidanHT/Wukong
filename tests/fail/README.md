# `tests/fail/` — compile-fail fixtures

**112** user-written `.wk` programs that must be **rejected at compile time** with a pinned diagnostic
code. Directory name is `tests/fail` — **not** `tests/compile-fail`.

## Format

```wk
// EXPECT-CODE: E0501
module tests.fail.my_fixture
```

- Exactly **one** `// EXPECT-CODE:` per file; the first match wins.
- A fixture without it panics the whole suite.
- Harness runs `--error-format=json --emit=mir` and requires the first diagnostic to be an error with the pinned code. A later matching code cannot hide an unrelated first error, a code-less diagnostic or leading non-diagnostic stderr.

Codes are permanently stable — never renumber or recycle. `E0001` is retired.

## Fixtures whose MESSAGE is pinned too

A handful of shape fixtures are also listed in `NAMED_DIM_MESSAGES`
(`crates/wukongc/tests/fail.rs`), which asserts the exact headline in the JSON stream: `wrong_k.wk`
and `wrong_m.wk` (the dimension is named — `` dimension mismatch on `K` (axis 0) ``), `rank.wk` (rank
against rank plus both shapes) and `tensor_index_oob.wk` (the axis of the offending index). Editing
the prose of one of those messages means editing that table in the same commit; the code alone
staying green does not mean the diagnostic is still useful.

## Adding a fixture

Drop `foo.wk` here with `// EXPECT-CODE: Ennnn`. If the code is new, add it to
`crates/wukong_diag/src/catalog.rs` in the same change. New catalogue entries without a fixture or
probe fail `every_catalogued_code_is_exercised`.

Imported support for multi-file failures lives in [`lib/`](lib/) — not scanned as fixtures.
