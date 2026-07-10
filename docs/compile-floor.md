# The compile-time floor — code → object at its principled minimum

> **STATUS: DRAFT.** This document defines the *method* and the *floor model*. The measured tables
> are placeholders to be filled by an authoritative central run — no absolute compile-time numbers
> are claimed here (the recording machine was on battery when this was written, so its clock is not
> reportable; only stage **shares/ratios**, which are clock-invariant, are noted, and only where
> explicitly labeled *provisional*). Fill the bracketed cells from a release-binary run of the two
> instrumentation modes below and promote to non-draft.

## 1. The question

Two adjacent claims need evidence:

1. **Wukong's code→object compile time is at a principled floor** — the compiler does close to the
   irreducible work and little else.
2. **The in-process-vs-spawned-toolchain tradeoff is fully characterized** — we know exactly what a
   user pays to invoke `wukongc.exe` as a process versus calling the compile API in-process, and what
   the `--emit=exe` link adds on top.

The optimizer half of (1) is already well-instrumented: `wukong-bench compile-time` attributes time
pass-by-pass, and prior work (FxHash keying, CSE packed-keys, clean-tracking) drove the optimizer to
a documented local optimum *under the byte-identical-output constraint*. What was missing is the
**whole-pipeline** picture (the optimizer framed against every other stage) and the **spawn tax**.
Two new `wukong-bench` modes supply both.

## 2. Instrumentation (how to reproduce the tables)

Build the release binaries first (a debug `wukongc`/bench makes the backend look ~7× slower and the
measurement is worthless — see project memory):

```text
cargo build --release -p wukongc -p wukong_bench
```

**`compile-profile`** — full-pipeline per-stage breakdown, in-process, best-of-N minimum per stage
per file, over the corpus:

```text
cargo run -p wukong_bench --release -- compile-profile tests/run examples bench/kernels
```

It re-runs each stage in isolation on a fixed upstream artifact (a stage that mutates the interner —
parse, mir_build — rebuilds its input each rep *outside* the timed region), so each stage's minimum
reflects that stage alone. It prints (a) stage totals + shares + a throughput column, (b) the work
totals (source bytes, tokens, AST nodes, MIR ops, object bytes), and (c) the heaviest ~10 files.

**`spawn-overhead`** — the same source compiled in-process (API, to object bytes in memory) vs by
spawning `wukongc.exe`, cold-start and warm steady-state, with the `--emit=exe` link broken out:

```text
cargo run -p wukong_bench --release -- spawn-overhead tests/run examples bench/kernels
```

**`compile-vs`** (pre-existing) — the reference comparison, same-run wukongc vs gcc/g++/rustc
compiling equivalent kernels to an object:

```text
cargo run -p wukong_bench --release -- compile-vs
```

Every table these modes print carries a **regime label** (`cold-start / first call` or
`warm steady-state / best-of-N min`) per the campaign's regime discipline. The floor is a
warm-steady-state property; cold-start is a separate axis owned by the spawn characterization.

### Stage-boundary implementation choice

Boundaries are timed **bench-side**, by calling the stage crates directly
(`wukong_lexer::tokenize` → `wukong_parser::parse_module_tokens_from` → `wukong_sema::check` →
`wukong_mir_build::lower_program` → `wukong_opt::optimize` → `wukong_codegen_cranelift::emit_object`),
exactly as `compile-time` already calls `wukong_opt` directly. The driver
(`wukong_driver::compile`) is left untouched: it is a thin sequencer, and threading `Instant` hooks
through it would instrument the product path for a measurement concern. The bench reproduces the same
stage order the driver uses, minus the driver's I/O and diagnostics front-matter — which is exactly
the in-process "work only" quantity the floor is about (and the spawn tax below is what re-adds the
driver's front-matter).

**One boundary is not split:** the Cranelift backend is measured as a single `codegen+obj` stage
because `wukong_codegen_cranelift` exposes only the combined `emit_object` (ISA build → `populate_module`
isel/regalloc/machine-code → object-container `emit`). Splitting instruction-selection from
object-byte serialization would need a timing hook inside that crate. The object-serialization share
is instead bounded analytically in §4 from the emitted byte count, which `compile-profile` reports
per file.

## 3. The irreducible work (what genuinely must happen)

Compiling one source file to an object *cannot* cost less than doing this work once:

| Stage | Irreducible work | Size counter |
|---|---|---|
| Lex | Read every source byte once, classify into tokens | source bytes |
| Parse | Consume every token once, build the AST | tokens → AST nodes |
| Sema | Visit every AST node once: resolve names, infer/check types & shapes | AST nodes |
| MIR build | Lower every checked node to MIR instructions | AST nodes → MIR ops |
| Optimize | Rewrite MIR to the `-O2` fixpoint (at minimum, read every MIR op once per pass that runs) | MIR ops |
| Codegen | Select an instruction for every lowered MIR op; allocate registers | MIR ops → machine bytes |
| Object emit | Serialize the machine code + relocations + symbols into the platform object container | object bytes |

The floor is the sum of each stage doing its counter's worth of work at the best throughput that
stage's data structures physically allow. A stage is **at floor** when its measured throughput sits
at the ceiling its access pattern permits; it is **above floor** (has avoidable cost) when it does
work not implied by its counter — redundant passes, re-hashing, re-allocation, re-derivation.

Two structural facts make Wukong's floor unusually low and worth defending:

- **No LLVM.** The backend is Cranelift, a single-pass-ish isel with fast register allocation, not a
  multi-pass optimizing IR. Codegen is O(MIR ops) with a small constant, not the super-linear
  behavior of an LLVM `-O2` pipeline. This is the structural reason the `compile-vs` ratios exist.
- **One merged flat module, block-parameter SSA from the start.** MIR is already Cranelift's model
  (block params = φ-nodes, explicit `alloca`/`load`/`store`), so `mir_build → codegen` is close to a
  transliteration, not a re-analysis.

## 4. The floor model (per-stage ceilings)

For each stage, the ceiling is set by its access pattern. Fill the measured throughput from
`compile-profile`'s throughput column and mark each stage at/near/above floor.

| Stage | Ceiling set by | Expected regime | Measured throughput | At floor? |
|---|---|---|---|---|
| Lex | Linear byte scan, branch-predicted; ~memcpy-class | 100s of MB/s | `[fill]` | `[fill]` |
| Parse | Token scan + AST-node bump-allocation; interner hits | 10s of Mtok/s | `[fill]` | `[fill]` |
| Sema | Node visit + hash-map side tables (types/consts/defs) keyed by `NodeId` | hash-bound | `[fill]` | `[fill]` |
| MIR build | Node → op emission + interner extension | alloc-bound | `[fill]` | `[fill]` |
| Optimize | `-O2` fixpoint; HashMap-bound (see `compile-time`) | pass-count × op scan | `[fill]` | `[fill]` |
| Codegen+obj | Cranelift isel/regalloc/emit + object serialize | O(MIR ops), small const | `[fill]` | `[fill]` |

**Object-emit sub-share (analytic).** Object serialization writes `object bytes` into a COFF/ELF
container: a linear memcpy-class write of the machine-code, relocation, and symbol tables. At a
conservative ~1 GB/s container-write rate, a typical kernel object of `[obj bytes]` costs on the
order of `[obj bytes] / 1e9` s — i.e. object-emit is a small fraction of the `codegen+obj` stage;
the stage's time is dominated by isel + register allocation. (Promote this to a measured split only
if a hook is added to `wukong_codegen_cranelift`.)

### Provisional shares (clock-invariant; smoke-test only — NOT authoritative)

> These are shape observations from a battery-state smoke run over `examples` + `bench/kernels` (13
> files), kept only because *ratios* survive clock swing. The absolute times are NOT reportable and
> are omitted. Replace with a release central run over the full corpus.

- **Stage-share ordering, whole-to-object (largest → smallest):**
  `codegen+obj (~74%) > optimize (~19%) > mir_build (~5%) > sema (~1.1%) ≈ parse (~1.0%) > lex (~0.2%)`.
  The headline shape: **once the backend is included, the Cranelift `codegen+obj` stage — not the
  optimizer — is the dominant cost.** This is the whole-pipeline correction to the optimizer-only
  view; `compile-time`'s "optimizer is ~80–85%" is a statement about *front→O2 only* (excludes the
  backend), and it cross-checks: optimize's share of the front→O2 subtotal here is ≈ 19 / (19 + 5 +
  1.1 + 1.0 + 0.2) ≈ **72%**, in the same ballpark.
- **A fixed object-container floor.** Every file — even a 400-byte `fib_rec.wk` — emits ~6.5 KB of
  object. The `codegen+obj` per-file time barely falls below a floor (~0.4 ms in the smoke run) for
  the smallest inputs, consistent with a fixed COFF-container + symbol/reloc-table cost that
  dominates small objects and is the first thing to characterize when the backend split lands.

## 5. The reference comparison (why gcc/rustc -O2)

`compile-vs` compiles an *equivalent* tensor kernel (GEMM / SAXPY / DOT) to an object with each of
`wukongc --emit=obj -O2`, `gcc -O2 -c`, `g++ -O2 -c`, `rustc -O --emit=obj`. This is the right
yardstick for a floor claim because:

- **It is what a user waits for.** Each command is compile-only (no link), producing an object from
  the same computation, so the number reflects the compilers' own work: front-end + optimizer +
  native codegen + process startup. Startup + backend weight is exactly what a lean Cranelift path
  avoids, so including it is fair — it is user-visible latency.
- **Level 2 across the board** is symmetric: gcc/g++ default the comparison to `-O2`, and rustc `-O`
  is opt-level 2, so no compiler is charged a heavier optimizer than another.
- **Bare translation units.** The C/C++ kernels have no `#include` and no `main`, matching the bare
  `#[no_mangle]` Rust kernel, so none is charged a header-parse the others avoid (this was an actual
  fairness fix — see the benchmark-fairness audit in project memory).

The historical standing is ~9–12× faster than gcc/g++/rustc on these kernels (recorded in
`compile-time-session-x1`). **The floor claim is: after subtracting process startup, the remaining
in-process wukongc time (from `compile-profile`) is within a small constant of the §3/§4 floor** —
i.e. the ratio is not a startup artifact but real backend-weight difference. `spawn-overhead`
isolates exactly that startup so the two halves compose.

Measured (fill from `compile-vs`, same-run ratios only):

| kernel | wukongc | gcc/mc | g++/mc | rustc/mc |
|---|---|---|---|---|
| gemm4x4 | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| saxpy | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| dot | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

## 6. The spawn tradeoff (in-process vs process)

`spawn-overhead` reports, per file, cold-start and warm steady-state:

- **(a) in-process obj** — `lex..emit_object` in memory (the §3 work only).
- **(b) spawn obj** — `wukongc.exe --emit=obj` end-to-end child wall.
- **spawn-tax = (b) − (a)** — process creation + runtime init + file I/O + the driver's
  source-map/diagnostics/import front-matter that the in-process path skips.
- **(c) spawn exe** and **link = exe − obj** — the `--emit=exe` path. The driver links via a
  **rustc-driven link** (rustc invokes the platform linker and pulls in `wukong_runtime`'s AVX2
  microkernels + resolves the `.rodata` string relocations; a scalar no-kernel program can fall back
  to a `cc` link). This link is a *whole second process* (rustc), so it dominates the exe path.

| regime | in-proc obj | spawn obj | spawn-tax | spawn exe | link(exe−obj) |
|---|---|---|---|---|---|
| cold-start / first call | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| warm steady-state | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

**Structural findings (mechanism, not absolutes; ratios from the battery smoke run):**

- The spawn tax is **fixed overhead, essentially size-independent** — across the smoke files it held
  a near-constant band while the in-process compile itself varied several-fold with source size. It
  is process/OS/I/O cost, so it is proportionally huge for small kernels and amortizes only for large
  inputs. This is why the in-process API is the right integration surface for anything that compiles
  many small units (a JIT, a test harness, a build server), and the CLI is the right surface for
  one-shot builds.
- **The link is ~95% of the `--emit=exe` path** in the smoke run (link ≈ exe − obj, with obj a small
  remainder). The exe path's latency is almost entirely the rustc subprocess, not Wukong's own work —
  see item 6.
- **Cold vs warm** matters on Windows: the first spawn pays image load + page-in of `wukongc.exe`
  and its dependencies; the warm best-of-N runs from the OS file/image cache. The gap between the two
  cold-start columns and the warm table is the page-cache cost — a one-time tax per fresh process
  image, not per compile.
- **The link dominates `--emit=exe`.** Because linking spawns rustc, the exe path's cost is mostly a
  second toolchain process, not Wukong's codegen. A user who needs many executables should prefer
  `--emit=obj` + a single batched link, or the in-process path + one link.

## 7. Avoidable costs, ranked (candidates for elimination)

From the model and what the instrumentation exposes, ranked by expected payoff. This is the
elimination worklist the floor exercise exists to produce; confirm each against the measured tables
before acting.

> **Provisional priority correction (smoke run).** Once the backend is counted, `codegen+obj`
> (~74%) — not the optimizer (~19%) — is the largest *wall-time* stage. So while the optimizer is
> the biggest *avoidable front-end* cost (items 1–2), the backend/object-emit items (4, promoted
> below) may hold more total wall time. Confirm against the central run, then re-rank if the shares
> hold.

1. **Optimizer pass redundancy (biggest front-end lever, already the focus).** The optimizer is the
   largest *front-end* in-process stage and is HashMap-bound (per `compile-time`). Any pass that
   re-scans MIR without changing it, or re-hashes keys a prior pass already computed, is above floor.
   `compile-time`'s per-pass table names the hottest pass; that is the standing lever.
2. **Interner re-hashing across stages.** Parse and mir_build both intern; if a symbol is re-hashed
   at each stage rather than carried as a resolved `Symbol`, that is avoidable. Sema's `NodeId`-keyed
   hash side tables (`types`, `consts`, `defs`) are a candidate hot spot — a dense `Vec<_>` indexed by
   `NodeId` would beat a `HashMap` if `NodeId`s are contiguous.
3. **Redundant MIR re-derivation between mir_build and codegen.** If codegen re-derives layout/size
   facts mir_build already computed (e.g. `size_of` recomputed per `gep`), caching them is a win.
4. **Backend fixed-object floor (promoted — likely large for small units).** The smoke run shows
   `codegen+obj` dominating wall time and a per-file floor that barely moves for tiny inputs, matching
   a fixed COFF-container + symbol/relocation-table cost (~6.5 KB object even for a 400-byte source).
   For a workload of many small compiles this fixed cost, not the isel of the actual ops, is the
   backend floor — the first thing a `codegen`-vs-`object-emit` split (a hook in
   `wukong_codegen_cranelift`) should quantify. Analytically the *byte-write* is memcpy-class (§4), so
   if the fixed cost is real it is container/symbol bookkeeping, not raw serialization bandwidth.
5. **Spawn tax for many-unit workloads.** Not a per-compile cost but an integration choice: exposing
   a stable in-process compile-to-object API (the bench already calls it) lets multi-unit callers
   avoid N process spawns. This is the highest-leverage *product* change if the workload is many small
   compiles.
6. **`--emit=exe` link = a second process.** If fast one-shot exe latency ever matters, an in-process
   or incremental linker would remove the rustc spawn; today it is correctly traded for
   link correctness (runtime kernels + `.rodata` relocations).

## 8. What "at the floor" will mean when the tables are filled

The floor claim is **substantiated** when, in a warm-steady-state release run:

- each front-end stage's throughput (§4) sits at the ceiling its access pattern allows (linear scans
  near memory bandwidth; hash-bound stages near the hash's throughput), AND
- `codegen+obj` is O(MIR ops) with object-emit a small analytic fraction, AND
- the `compile-vs` advantage survives subtracting the `spawn-overhead` process cost — i.e. it is
  real backend weight (no LLVM), not merely faster startup.

Anything on the §7 list that measures non-trivial is the gap between the current compiler and that
floor, and the ranked worklist for closing it.
