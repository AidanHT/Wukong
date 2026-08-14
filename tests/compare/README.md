# Wukong comparison corpus

This is the Phase 0 ruler for comparing idiomatic Wukong loops with competent C
and CUDA peers. It is evidence infrastructure, not a performance result.

`tests/compare/` is deliberately separate from `tests/run/` and `tests/fail/`.
Those harnesses use non-recursive `read_dir` scans of their own directories and
do not discover anything here. A later gate must opt into this corpus
explicitly.

## Recognition-off runs

The Wukong fixtures mark their kernel functions `@generic`. Once the recognition
controls land, either spelling below must leave the source loop generic:

```powershell
$env:WUKONG_NO_RECOGNIZE = "1"
cargo run -q -p wukongc -- --run --backend=native -O2 tests/compare/saxpy/saxpy.wk
```

```sh
WUKONG_NO_RECOGNIZE=1 cargo run -q -p wukongc -- \
  --run --backend=native -O2 tests/compare/saxpy/saxpy.wk
```

Until that implementation merges, inspect `--emit=mir -O2` and report any
`wukong_*` call honestly. The unknown-op fixture must have no
`wukong_sgemm`, `wukong_vmath`, `wukong_norm`, `wukong_axpby`, or
`wukong_velem` call even on today's branch.

## Peer rules

- Compare the same algorithm and data layout.
- Use the natural loop order for each peer; do not pessimize C to flatter
  Wukong.
- Give C pointer arguments `__restrict__` where alias freedom is part of the
  Wukong spelling.
- Give CUDA a competent coalesced launch, not a serial or one-block strawman.
- Do not report a win from this directory without a fresh, named-device,
  same-run measurement under the repository measurement rules.

## Line-count protocol

Every implemented fixture records the physical line counts of its Wukong, C,
and CUDA files in the header comments of all three peers. Counts include
comments and blank lines. Recount and update all three headers whenever any
peer's surface changes; the committed files, not prose estimates, are the
source of truth.

## Fixtures

- `saxpy/`: known op, recognition intentionally disabled.
- `naive_gemm/`: known op, recognition intentionally disabled; `wrong_k.c`
  demonstrates that C accepts a mismatched inner dimension.
- `unknown_op/`: a small three-point stencil no current named recognizer
  matches.
- `tiled_transpose/`: Phase 2 placeholder; no implementation is claimed.

The shape-mismatched Wukong counterpart belongs in the compile-fail corpus, not
here. See `tests/fail/wrong_k.wk` when the sibling Phase 0 shape-diagnostic work
lands.

`run_cpu.ps1` builds each C peer with `gcc -O3 -march=native` when available and
runs the matching Wukong source on the native backend. It prints program output
only; it does not fabricate or publish timings.
