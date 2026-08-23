# Tiled transpose — the `shared` + `sync_block()` comparison fixture

The master plan §7.1 row that carries the "easier than a CUDA `__shared__` kernel" claim. It is
evidence infrastructure, not a performance result: **no timing is taken here and none is claimed.**

| File | What it is |
|---|---|
| `tiled_transpose.wk` | the fixture: `@tile` + `shared [f32; T*T]` + `sync_block()`, `@generic`, 8x8 |
| `tiled_transpose.c` | the C peer — cache-blocked, `__restrict__`, natural loop order. Built and run |
| `tiled_transpose.cu` | the CUDA peer — the canonical coalesced 32x32 `__shared__` tile with `+1` padding. **Committed, not built** (no `nvcc` on this box), the same standing as `saxpy.cu` |
| `naive_scatter.wk` | the baseline §7.1 names ("faster than Wukong's own naive scatter"). Same output, same recognition-off spelling; only the schedule differs |
| `wrong_shape.c` | the safety mutant (T4): C compiles and runs it clean |
| `shared_in_loop_hang.wk` | committed evidence of a defect this fixture found. **Do not run on gpu-native** |

## What the fixture shows

All three peers transpose the same row-major 8x8 `f32` matrix through the same algorithm: stage a
4x4 tile, then write it back transposed, so that both the read of `a` and the write of `b` run along
rows and the stride-N turn happens inside the tile.

- **The Wukong source is one program for three backends.** The same file runs on the interpreter, on
  the Cranelift native backend and on `--backend=gpu-native`, bit-identically. There is no host-side
  device code: no `cudaMalloc`, no `cudaMemcpy`, no grid/block arithmetic, no `cudaFree`. Nine of the
  CUDA peer's lines exist only to move data to a device and back.
- **`shared` and `sync_block()` have a sequential CPU meaning** (T6), which is what makes that
  possible: `shared` is an ordinary local and `sync_block()` a no-op/fence on the CPU backends, and
  PTX `.shared` + `bar.sync 0` on gpu-native.
- **The shape is in the type**, so the transpose mutant in `wrong_shape.c` — which C compiles, runs
  and silently gets wrong — is a compile error here (see "Safety" below).

**It does not show that the Wukong source is shorter.** The counted lines are

| | physical (headers) | code only |
|---|---|---|
| `tiled_transpose.wk` | 63 | 41 |
| `tiled_transpose.c` | 41 | 25 |
| `tiled_transpose.cu` | 58 | 40 |

so Wukong ties the CUDA peer on code lines and loses to C. The kernels alone are 25 Wukong lines
against 20 CUDA lines: CUDA gets its tile origin free from `blockIdx`/`threadIdx` and the launch
geometry, while the `.wk` writes the tile-origin loops itself and pays a second barrier for reusing
one tile across them. What Wukong removes is the 9 lines of `cudaMalloc`/`cudaMemcpy`/`dim3`/`<<<>>>`
/`cudaFree` on the other side of that trade, and the recompile-for-another-backend. The effort
claim this fixture supports is "one source, no device plumbing, three backends, shapes checked", not
"far fewer lines" (T5).

Physical counts include comments and blanks and are recorded in all three headers; recount with
`wc -l` and update all three together whenever any peer's surface changes.

## Why two barriers

`tiled_transpose.wk` declares its tile once at function scope, which is where CUDA declares
`__shared__`. That makes it a buffer **reused** across tile-loop iterations, so the loop body is
fenced at both ends: the first `sync_block()` publishes the loads before the transposed reads, and
the second stops the next iteration's loads from overwriting entries the current one is still
reading. The CUDA peer needs one barrier because its tile loop comes from the launch grid — one
block, one tile, no reuse. Declaring the tile *inside* the loop would need only one barrier and is
what `tests/run/lang_tiled_transpose.wk` does; on this fixture's shape that spelling hangs the
device, which is what `shared_in_loop_hang.wk` records.

## MIR witness (T2) — the census

Recognizers are gate-blind, so this grep is the only proof that exists:

```sh
cargo run -q -p wukongc -- --emit=mir -O2 tests/compare/tiled_transpose/tiled_transpose.wk \
  | grep -o 'wukong_[a-z0-9_]*' | sort | uniq -c
```

| Program | census at `-O2` | kernel dispatches |
|---|---|---|
| `tiled_transpose.wk` | 2 x `wukong_parallel_for` (`wukong$par$0/1`, the `@parallel` region outliner) | **none** |
| `naive_scatter.wk` | 1 x `wukong_parallel_for` | **none** |

No `wukong_transpose_f32`, no `wukong_velem_*`, no `wukong_sgemm`, no `wukong_norm`.

**`@generic` is load-bearing on `naive_scatter.wk` and inert on `tiled_transpose.wk`**, and both
halves matter. Delete `@generic` from the naive scatter and the census becomes
`1 wukong_transpose_f32_parallel`: the whole loop is replaced by a tuned runtime kernel, and the
§7.1 pass condition would then be comparing the tiled loop against the *dispatcher* rather than
against the language. Delete it from `tiled_transpose.wk` and the census does not change — no
recognizer matches the tiled spelling — so the attribute there is an honesty marker, not a change.

## Correctness

Every backend, both `-O` levels, 64 printed lanes, one identical stdout (`md5 4e0f86a3…`):

| | `-O0` | `-O2` |
|---|---|---|
| `--run` (interp, the oracle) | pass | pass |
| `--run --backend=native` | pass | pass |
| `--run --backend=gpu-native` (RTX 4050, `sm_89`) | pass | pass |

Expected values come from the scalar semantics of the loop, not from compiler output: with
`a[k] = k + 1`, `b[r*8 + c] = a[c*8 + r] = c*8 + r + 1`. The C peer prints the same 64 lines
(`gcc -O3 -march=native`; MSVCRT emits CRLF, the bytes are otherwise identical).

## Safety (T4)

`tests/fail/transpose_wrong_shape.wk` allocates the output of a transpose with the *input's* shape —
`Tensor[f32, 8, 4]` where `Tensor[f32, 4, 8]` is required. Wukong rejects it with two `E0502`s that
name the disagreeing dimension.

`wrong_shape.c` is the same mistake in the idiomatic C interface, and C has nothing to check: the
shape lives in the index arithmetic, not in a type. It compiles clean at
`gcc -O3 -march=native -Wall -Wextra`, runs to completion, and every index is in bounds — both
shapes hold exactly 32 floats — so no sanitizer fires either. Only the row structure of the answer is
wrong. That is the sharper half of the claim: the failure mode is a silently wrong result, not a
crash.

A flat `[f32; 32]` spelling could not catch this. Both shapes have the same length; only the
shape-typed spelling carries enough information.

## What is owed

- **The §7.1 timing condition — "faster than Wukong's own naive scatter" — has not been measured.**
  This lane authored the fixture on battery, and the repository measurement rules require AC+full,
  an idle box, a release build, a pinned timing thread, best-of-N minima and a `C(twin)` control.
  `naive_scatter.wk` is committed so the A/B is a run away, but no number exists and none is claimed.
- **The timing needs a bigger extent than the committed one.** At 8x8 the whole problem is in L1 and
  tiling cannot show anything. The fixture is written over `const N` / `const T` so the extent is a
  one-line change, but T3 ("print every lane") then needs a dense prefix plus corners rather than all
  64 lanes.
- **The C peer has never been timed against either `.wk`** (§3B), so this fixture has no SOTA row.
  It joins `saxpy`, `naive_gemm` and `unknown_op` in the W1b compare-timing harness.
