# wukong_xbench

Cross-language benchmark: Wukong (Cranelift JIT) vs C (gcc) vs Rust (rustc) vs optional peers
(oneMKL, `matrixmultiply`, PyTorch in the `model` section). Measures generated-kernel performance;
it is not the product.

**Critical landmine:** this harness compiles its `.wk` peer programs at **runtime**. A language,
sema, or recognizer change can break every generated kernel without breaking `cargo test`. Smoke
after any language change:

```sh
cargo run -p wukong_xbench --release -- saxpy
```

(Do not rely on `cargo test` alone for xbench health.)

## What it does

For each kernel, builds the same computation several ways and times them through one identical Rust
loop over the same buffers:

- **Wukong:** `.wk` source generated and JIT-compiled in process
- **C/C++:** compiled to a shared library at runtime (`CC`/`CXX` override, default gcc/g++)
- **Rust:** `rustc -C opt-level=3 -C target-cpu=native`

Reports compile time and ns/call per column. A missing toolchain prints `n/a` and continues.

Every battery row additionally reports a **`wk(gen)` recognition-off line** (master plan §7.2): the
same source lowered with every kernel recognizer skipped (`LowerOptions::recognize = false`, the
`WUKONG_NO_RECOGNIZE` / `@generic` lowering), its ratio vs C and vs the recognizer-on column, and
the MIR dispatch census for **both** lowerings — recognized kernels are gate-blind, so the census is
the only proof of what a number measured. The whole run pins its timing thread (`XBENCH_PIN=off`
disables; see the CPU PINNING block in `src/main.rs`).

The two lowerings are **built first and then timed A B B A**, and each row prints the `row floor`
its own samples measured. Both matter for the same reason. Timed in source order, the
recognition-off column was always second with a full parse + sema + lower + `optimize(3)` + JIT
between the two timed regions, so every within-pair drift landed on `wk(gen)`, in every family, in
one direction — the identical systematic bias `abba_min` exists to remove for C vs C++, sitting on
the column the language's §3B ratio is read from rather than on context. And because the interleave
times each build twice, the spread of one build's two samples is a twin control for that column, in
that run, for free: `row floor 1.02x` is what an effect on that row has to beat, and a ratio inside
it is printed `[INSIDE FLOOR]` rather than quoted. Before this the recognition-off headline had no
control of any kind.

Optional first CLI argument is a substring filter over kernel/section names (e.g. `saxpy`).

## Sections

| section | module | focus |
|---|---|---|
| elementwise / GEMM / … | `src/main.rs` | recognizer-dialect kernels |
| general code | `src/general.rs` | ML-engineer spellings, each lowered twice: as written, and `(gen)` = recognition off |
| model | `src/model.rs` | 12-layer forward vs PyTorch |

The `general` section prints a **dispatch census** (`grep wukong_` equivalent) because recognized
kernels are GATE-BLIND to `cargo test`.

This section was built on the premise that these spellings *miss* the recognizers, and measurement
says otherwise: all five block spellings dispatch 13-16 call sites, `b-natural` included. So each
Wukong program is built twice — as written, and again with recognition off (the `(gen)` columns) —
and only the `(gen)` rows are floorable under §3B, because only they measure the language rather than
the dispatcher. The recognizer-on rows are kept as context and `sota-check` marks them `off-bar`.

## Usage

```sh
cargo run -p wukong_xbench --release --             # all kernels
cargo run -p wukong_xbench --release -- saxpy       # smoke filter
cargo run -p wukong_xbench --release -- general     # general-code section only
cargo run -p wukong_xbench --release -- compare     # tests/compare/ timed vs its committed C peers
cargo run -p wukong_xbench --release -- sota-check  # gate: sweep vs bench/sota/FLOOR.md (exit code!)
```

`compare` times timing-scale twins of the `tests/compare/` fixtures (recognition-off, `@generic`,
census-witnessed) against the committed C peers' kernels on the round-interleaved pinned protocol.
`sota-check` is part 11 of the language close-out verification gate: it re-runs the in-scope sweep
(general + compare) and exits non-zero when a ratio falls below its recorded floor in
`bench/sota/FLOOR.md`, when a program leaves the sweep, when a bar peer fails to compile, or when
the run is not evaluable (battery / AC+charging / power-state change / failed oracle). It requires
AC+full and an idle box; `WUKONG_SOTA_FLOOR` overrides the floor-file path.

> **`sota-check` is built but NOT YET ARMED.** `bench/sota/FLOOR.md` does not exist, so the mode
> exits 2 with `cannot read …/bench/sota/FLOOR.md` and has never held a floor. Read every sentence
> above as a description of the mechanism, not of a gate that is currently guarding anything. Arming
> it needs a measurement round the machine cannot currently give: the floor values are transcribed
> from a sweep run at **AC+full on an idle box**, and this laptop's other two power states (battery,
> AC+charging) are different machines. Until that round happens and the owner has read the file,
> part 11 of the ten-part gate is a step that refuses rather than a step that passes.

**Timing requires `--release`.** Peers need gcc/g++/rustc on PATH; Wukong needs no external LLVM.

## Testing

The timed rows are the instrument and have no `cargo test` equivalent: Wukong output is checked
against the interpreter oracle (bit-exact) inside each row as it runs. What `cargo test -p
wukong_xbench` does cover is the *pure* half — the f64 comparison oracles, the round/spread
statistics, `parse_floor`, and `evaluate`'s refusals — which is where a measurement instrument
fails silently rather than loudly. Two properties worth keeping there: the oracles return infinity
for a non-finite deviation (`NaN > worst` is false, so a naive loop scores an all-NaN buffer as
bit-exact), and `evaluate` refuses a run whose `C(twin)` control did not settle (`floor_factor`
*grows* with a degraded control, so a wider tolerance would otherwise make the ratchet more
forgiving exactly when the instrument is worst).
