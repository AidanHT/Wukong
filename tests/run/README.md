# `tests/run/` — passing-program fixtures

**361** user-written `.wk` programs that must compile and execute. This directory is the primary
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

## Run one fixture

```sh
cargo run -q -p wukongc -- --run -O2 tests/run/foo.wk
```
