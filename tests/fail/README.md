# `tests/fail/` — compile-fail fixtures

**110** user-written `.wk` programs that must be **rejected at compile time** with a pinned diagnostic
code. Directory name is `tests/fail` — **not** `tests/compile-fail`.

## Format

```wk
// EXPECT-CODE: E0501
module tests.fail.my_fixture
```

- Exactly **one** `// EXPECT-CODE:` per file; the first match wins.
- A fixture without it panics the whole suite.
- Harness runs `--error-format=json --emit=mir` and checks stderr for `"code":"<code>"`.

Codes are permanently stable — never renumber or recycle. `E0001` is retired.

## Adding a fixture

Drop `foo.wk` here with `// EXPECT-CODE: Ennnn`. If the code is new, add it to
`crates/wukong_diag/src/catalog.rs` in the same change. New catalogue entries without a fixture or
probe fail `every_catalogued_code_is_exercised`.

Imported support for multi-file failures lives in [`lib/`](lib/) — not scanned as fixtures.
