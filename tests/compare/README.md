# Wukong comparison corpus

This is the Phase 0 ruler for comparing idiomatic Wukong loops with competent C
and CUDA peers. It is evidence infrastructure, not a performance result.

`tests/compare/` is deliberately separate from `tests/run/` and `tests/fail/`.
Those harnesses use non-recursive `read_dir` scans of their own directories and
do not discover anything here. A later gate must opt into this corpus
explicitly.

## Recognition-off runs

The Wukong fixtures mark their kernel functions `@generic`. **The recognition
controls have since landed** (Phase 0 tasks 0.2 and 0.3), so `@generic` on the
function is the per-fixture opt-out and either spelling below is the
whole-program instrument:

```powershell
$env:WUKONG_NO_RECOGNIZE = "1"
cargo run -q -p wukongc -- --run --backend=native -O2 tests/compare/saxpy/saxpy.wk
```

```sh
WUKONG_NO_RECOGNIZE=1 cargo run -q -p wukongc -- \
  --run --backend=native -O2 tests/compare/saxpy/saxpy.wk
```

Recognizers are gate-blind, so `--emit=mir -O2` is the only proof that exists
and every fixture header records its census:

```sh
cargo run -q -p wukongc -- --emit=mir -O2 <fixture>.wk \
  | grep -o 'wukong_[a-z0-9_]*' | sort | uniq -c
```

Measured at the time of writing, `saxpy`, `naive_gemm` and `unknown_op` emit
**no** `wukong_*` call at all, and the two `tiled_transpose` programs emit only
`wukong_parallel_for` -- the `@parallel` outliner, not a named kernel.

## Timing — the master plan §7.1 "CPU within the §3.3 band" condition

`run_cpu.ps1` prints program output only; it does not and will not fabricate
timings. The timing surface for this corpus is

```sh
cargo run -p wukong_xbench --release -- compare
```

which times a **timing-scale twin** of each CPU fixture's loop (same algorithm,
same spelling, same `@generic`; the toy extent replaced by 2²² elements /
256³) against the committed C peer's kernel exported as `kbench`, on the
pinned, round-interleaved, `C(twin)`-controlled protocol, with every column
checked per-lane against an f64 scalar reference and the dispatch census
printed as the recognition-off witness. The twins live in
`crates/wukong_xbench/src/compare.rs` and cite these fixtures; keeping the two
in step is part of reviewing any edit to either. Measured floors live in
`bench/sota/FLOOR.md` and are enforced by `wukong_xbench -- sota-check` — which
is built but **not yet armed**: that file does not exist yet, so the mode refuses
(exit 2) rather than passing. See `crates/wukong_xbench/README.md`.

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
- `tiled_transpose/`: the Phase 2 surface end to end -- `@tile` + `shared` +
  `sync_block()` -- against the canonical coalesced CUDA `__shared__` transpose
  and a cache-blocked C peer. `naive_scatter.wk` is the §7.1 baseline,
  `wrong_shape.c` the safety mutant, and `shared_in_loop_hang.wk` committed
  evidence of a gpu-native defect the fixture found. Its README carries the
  census, the line counts and the timing that is still owed.

The shape-mismatched Wukong counterpart belongs in the compile-fail corpus, not
here: `tests/fail/wrong_k.wk` for `naive_gemm`, and
`tests/fail/transpose_wrong_shape.wk` for `tiled_transpose`.

`run_cpu.ps1` builds each C peer with `gcc -O3 -march=native` when available and
runs the matching Wukong source on the native backend. It prints program output
only; it does not fabricate or publish timings.
