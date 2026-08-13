# `tests/run/` — passing-program fixtures

**363** user-written `.wk` programs that must compile and execute. This directory is the primary
correctness corpus for the language — not `bench/` (measurement harnesses) or `tools/` (scripts and
cloud launchers).

Every harness collects fixtures with a **non-recursive** `read_dir` filtered on `extension == "wk"`.
Anything dropped here becomes a gated fixture immediately. Imported support modules live in
[`lib/`](lib/) — that subdirectory is never scanned.

## Directives

Put at the top of each file:

```wk
// RUN: --run
// EXPECT-EXIT: 0
// EXPECT-OUT: <line>
```

- `// RUN:` replaces the whole arg list (default `--run`). Only `run_suite` honours it; differential
  gates build their own args.
- `// EXPECT-EXIT:` is required for a meaningful gate result.
- `// EXPECT-OUT:` lines must match stdout exactly, in order. Omit all of them to skip stdout check.

## Authoring rules

- Print **every lane**, not a reduction.
- Expected values come from **scalar semantics**, never from pasting compiler output.
- Recognizer/kernel fixtures are **GATE-BLIND** — verify with `--emit=mir` and grep for `wukong_`.
- File-I/O writers need unique `io_test_*.bin` names (harness uses per-run scratch CWDs).

## `lang_*` — the thesis fixtures

`lang_generic_saxpy.wk` and `lang_generic_naive_gemm.wk` each carry the **same body twice**, once under
`@generic` and once without, and print every lane of both. The pair exists because this corpus cannot
otherwise see a recognizer: both spellings compute the same bytes, so the proof that `@generic`
suppressed the kernel — and that it suppressed it for **one function**, not the module — is a MIR count
asserted in `crates/wukongc/tests/recognize.rs`:

```sh
cargo run -q -p wukongc -- --emit=mir -O0 tests/run/lang_generic_naive_gemm.wk   # exactly one sgemm
```

A new `lang_*` fixture should keep that shape: a written-out loop, an un-attributed twin where one is
meaningful, every lane printed, and a MIR witness in `recognize.rs` rather than a behavioral claim.

## Run one fixture

```sh
cargo run -q -p wukongc -- --run -O2 tests/run/foo.wk
```
