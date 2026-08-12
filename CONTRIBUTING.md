# Contributing to Wukong

Thanks for your interest! Wukong is built incrementally, one small, tested change at a time.

## Ground rules

- **Everything builds and tests without LLVM.** The front-end, optimizer, and the MIR interpreter
  must keep working with plain `cargo test`; native codegen is Cranelift (pure Rust, no toolchain).
  The textual LLVM IR backend (`--emit=llvm-ir`) is also pure Rust and always built — `clang`/`llc`
  are only needed to compile the *emitted* IR, never by the build or tests. No backend may become a
  hard toolchain dependency of the default workspace; the GPU backend is the only opt-in one
  (`--features gpu`).
- **Each change is small and self-contained**, with tests, and leaves the tree green.
- **No new warnings.** CI builds with `-D warnings` and runs `cargo fmt --check` and
  `cargo clippy --all-targets`.

## Before you push

```sh
cargo fmt --all
RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets
cargo test --workspace
cargo check --features gpu --all-targets
```

CI sets `RUSTFLAGS: -D warnings` for the whole job, so a lint is an error there; without it your
local clippy is strictly weaker than the gate.

That last line is not optional. All GPU code is behind `--features gpu`, so `cargo test --workspace`
compiles only the two deliberately un-gated modules of `wukong_codegen_gpu` (`paged_kv`,
`paged_attention`), none of `wukong_driver::gpu_accel`, and neither of the two `Op` matches in
`wukong_codegen_gpu/src/lower.rs` — so it is your responsibility before pushing. (CI's `gpu-check`
job does build and lint the feature, but `cargo check`/`cargo clippy` compile a test without running
it.) If you touched the GPU backend, also run **both** GPU suites:

```sh
cargo test -p wukong_codegen_gpu --features gpu   # PTX generators, launch wrappers, MIR->PTX
cargo test -p wukong_driver --features gpu --lib  # the driver-side dispatch rules
```

Both skip rather than fail with no CUDA device, and `WUKONG_GPU_REQUIRED=1` turns those skips into
failures on a machine that has one. The second line is not implied by the first: `-p
wukong_codegen_gpu` does not build `wukong_driver`'s tests at all, and `wukong_driver` reports 20
tests without the feature against 38 with it. The 18 in that delta are the driver's half of GPU
dispatch — which route a recognized `C = A·Bᵀ` takes per compute capability, which shapes and which
operand *values* the Hopper `wgmma` seam declines before launching, and the counter algebra that
tells a wgmma launch from a fallback. They are pure functions of a capability and a shape, so they
run anywhere; CI's `gpu-check` job runs exactly this command on a device-free runner.

CI runs two more gates; run them too if you touched lowering, the optimizer, or a backend:

```sh
cargo run -p wukongc -- --run examples/fib.wk        # plus dot.wk, saxpy_array.wk
cargo run -p wukong_bench --release -- tests/run examples bench/kernels
```

The second is the optimizer-effectiveness report *and* a correctness gate: for every program that
lowers cleanly it verifies the post-optimization MIR, requires `-O0` and `-O3` to agree on exit code
**and** stdout, and requires the Cranelift backend at `-O3` to agree with the interpreter at `-O3`.
It exits non-zero on any divergence, and it is the only opt-invariance gate over `bench/kernels/` —
`tests/run` and `examples/` are additionally covered by `crates/wukongc/tests/run.rs` and
`crates/wukongc/tests/examples.rs`.

Run tests unpiped, as their own step: `cargo test … | grep X && git commit` takes *grep's* exit
status and commits on a failing suite.

## Where things live

See [`docs/internals.md`](docs/internals.md) for the crate layering and pipeline. In short:

- Foundations, under everything: `wukong_span` (`Span`/`SourceMap`, the string `Interner`/`Symbol`)
  and `wukong_diag` (`Diagnostic`/`DiagnosticSink`, the stable error-code catalogue behind
  `--explain`, and the terminal + JSON renderers).
- Front-end: `wukong_lexer`, `wukong_parser`, `wukong_ast`.
- Types & checking: `wukong_types`, `wukong_sema` (including shape checking).
- Middle-end: `wukong_mir`, `wukong_mir_build`, `wukong_opt`.
- Back-ends: `wukong_interp` (default oracle), `wukong_codegen_cranelift` (Cranelift: the JIT behind
  `--run --backend=native` **and** the host-object emitter behind `--emit=obj` / `--emit=exe` — the
  only native-object path in the tree), `wukong_codegen_llvm` (textual LLVM IR),
  `wukong_codegen_gpu` (PTX GPU backend, `--features gpu`). Runtime microkernels: `wukong_runtime`.
  `wukong_backend` is the declared seam — the `Backend` trait and the `Artifact` enum — that a
  "thing which executes or emits a `wukong_mir::Program`" implements. It is a *declared* seam, not
  the live dispatch path: `wukong_driver::compile` matches on its own `BackendKind` and calls the
  concrete functions, so a new target also needs a `BackendKind` variant, its dispatch arm, and the
  `--backend=` string in `wukongc`.
- Training: `wukong_autodiff` (reverse-mode autodiff, a MIR→MIR transform).
- Driver/CLI: `wukong_driver`, `wukongc`. Harness: `wukong_bench`, `wukong_xbench`.

Non-dev dependencies flow **downward** through those groups and there are no cycles; inside a group
the order is by pipeline stage, not by dependency (`wukong_parser` depends on `wukong_ast`), and
dev-dependencies are exempt (`wukong_opt`'s tests use `wukong_interp`). If you want an upward edge,
the abstraction belongs lower (or the caller belongs higher). `wukong_span` and `wukong_mir` sit
under everything, so a dependency added to either is added to the whole workspace — `wukong_span` has
an empty `[dependencies]` and `wukong_opt` deliberately keeps its own private `fxhash.rs` rather than
reusing `wukong_span::fxhash` (it already depends on `wukong_span`, just not on that type); do not
"unify" them casually.

Adding a crate: create `crates/wukong_<name>/` with a `Cargo.toml` inheriting the workspace package
fields (`version.workspace = true`, …); it becomes a member automatically
(`members = ["crates/*"]`). Register it in the root `[workspace.dependencies]` by path unless it is
a leaf binary nothing depends on (`wukong_bench`, `wukong_xbench` and `wukongc` are intentionally
absent), then add only downward edges.

## Adding or editing a benchmark peer

`crates/wukong_xbench` generates a C, a C++ and a Rust peer for every kernel and compiles them at
runtime. **A weakened peer breaks nothing observable** — the suite still runs, the cross-language
check still passes (a slow kernel is not a wrong kernel), and the only symptom is that Wukong's
published multiple goes up. That failure mode is not hypothetical: a 2026-08-04 audit found four
systematic peer defects that had inflated published figures by up to 63×. So:

1. **`__restrict__` on every pointer parameter** whose buffer is genuinely distinct from the others
   at the call site. If one may legitimately alias, leave it off *and* say why at the call site.
2. **Natural loop order.** The innermost loop must walk memory with unit stride wherever the
   algorithm allows. Column-outer traversals of a row-major matrix are the classic strawman.
3. **The same algorithmic opportunity Wukong's kernel has.** If Wukong dispatches a cache-blocked
   kernel, the peer is blocked. If Wukong's kernel reassociates a float reduction, a `C(fast)`
   [`-ffast-math`] column must exist *and be printed*, not merely computed.
4. Ask the question the whole exercise turns on: **is this how a competent C/C++/Rust programmer
   would write it?** If the answer needs a caveat, the caveat belongs in `BENCHMARKS.md`.
5. **Cross-check the peer's output**, not just its time — otherwise a peer can get "faster" by not
   doing the work.

Four tests in `crates/wukong_xbench/src/main.rs` enforce (1) and (2) mechanically
(`every_generated_c_kernel_declares_restrict`, `column_family_peers_stay_row_outer`,
`matmul_tn_peer_stays_kij`, `transpose_peer_stays_cache_blocked`); the rest is review.

**Measuring a peer change: use the harness's compilation model.** A standalone probe that puts the
kernel and its caller in one translation unit over `static` arrays lets gcc's interprocedural alias
analysis prove non-overlap by itself, so `restrict` measures as a no-op. The harness compiles each
peer to a **shared library**. Put the kernel in its own TU with heap buffers, or the probe answers a
different question — measured: the direct-convolution kernel reads 0.42 vs 0.47 ms (`restrict`
"hurts") in one TU, and 1.99 vs 0.40 ms (`restrict` is worth 5×) in its own.

## Adding a diagnostic

Pick the next free code in the range for the stage that emits it: `E00xx` generic/internal, `E01xx`
lexer, `E02xx` parser, `E03xx` name resolution, `E04xx` type checking, `E05xx` shape checking,
`C0xxx` codegen/lowering limitations (see `crates/wukong_diag/src/catalog.rs`, lines 7-18). **Codes
are permanently stable: never renumber one and never recycle a retired number.** `E0001` is retired
(it was "malformed type syntax"; no stage ever emitted it — the parser reports malformed types as
`E0203`/`E0204`). `E0404` is simply an unused gap; leave it alone rather than filling it with an
unrelated rule.

Three things must land together, because two gates enforce the pairing:

1. **Catalogue entry** — `entry!("E0210", "<one-line title>", "<body>")` in numeric order in
   `CATALOG` (`crates/wukong_diag/src/catalog.rs`). A unit test requires a leading `E`/`C`, a
   non-empty title and a body longer than 20 chars, so a placeholder fails the build.
   `wukongc --explain E0210` prints it.
2. **Emit site** — `Diagnostic::error("…").with_code("E0210").primary(span, "…")` in the producing
   crate. The code must be a bare `"E0210"` **string literal on a non-comment line**: the coverage
   gate scans textually for `"Ennnn"` literals, so a code assembled at run time (e.g. with
   `format!`) is invisible to it — a `const` that holds the literal is still found (prose and doc
   comments may cite codes freely — those lines are skipped).
3. **A fixture or probe** — `tests/fail/<name>.wk` whose first line is `// EXPECT-CODE: E0210`,
   followed by a `module tests.fail.<name>` header, a comment saying *why* the construct is
   rejected, and the minimal program that trips it. `crates/wukongc/tests/fail.rs` compiles every
   fixture with `--error-format=json --emit=mir` and asserts the process fails, that stderr contains
   `"code":"E0210"`, and that `explain` knows the code. Lexer/parser codes that cannot live in a
   well-formed file (unterminated string/comment) go in the inline `LEXER_PARSER_PROBES` table in
   that same file instead.

`crates/wukong_diag/tests/catalog_coverage.rs` enforces the catalogue in **both** directions — a
code some stage emits with no entry (`every_emitted_code_is_catalogued`) and an entry no stage emits
(`every_catalogued_code_has_an_emitter`) both fail — and `every_catalogued_code_is_exercised` in
`crates/wukongc/tests/fail.rs` requires every catalogued code to be reached by a fixture or a probe.
Verify the fixture against the real binary; plausible spellings often reach a *different* code, and
write the body against what the check actually rejects, not what it was designed to reject.

## Adding an optimizer pass

Implement `wukong_opt::Pass` and register it in `PassManager::standard` — the level tests there are
exactly `>= 1` and `>= 2`, so **`-O3` adds nothing and must stay byte-identical to `-O2`**; put the
pass in one of those two arms rather than adding a third. `run_function` loops the whole list to a
fixpoint, so return `true` from your `run_function` only when you actually changed the function. The
`CfgAnalyses` cache is shared across every pass and every iteration: **if your pass changes the
block set or any successor edge it must call `cache.invalidate()`**
(`crates/wukong_opt/src/cache.rs`) — a stale dominator tree silently corrupts CSE/LICM/mem2reg and
the verifier will not catch it. Under `debug_assertions` the pass manager verifies the function
after every pass that ran and panics naming the culprit, so run the tests in a debug build.

Add a test asserting both an effect (e.g. instruction-count reduction) and that results are
unchanged. The opt-level differential test (`crates/wukongc/tests/run.rs`) will also exercise it.

Two transforms deliberately sit **outside** the fixpoint, in `optimize` itself: `inline_program`
before it (it needs whole-program information) and `unroll_program` after it (it needs the canonical
two-block loop the fixpoint produces, and re-running it on its own output would unroll the same loop
every sweep). If you write a pass like that, give it a way to recognize what it already did, and
measure its compile-time cost — `cargo run --release -p wukong_bench -- compile-time` reports each
stage's in-process share, and a whole-program pass that rebuilds a dominator tree per function will
show up there immediately.

**Never trade exactness for speed in a pass.** Reassociating float arithmetic — the classic
multiple-accumulator reduction unroll — computes a different number, which breaks both the
interp-vs-native bit-exactness gate and `-O0` ≡ `-O{1,2,3}`. `wukong_opt::unroll` unrolls without
reassociating for exactly this reason, and takes the smaller win.

## Adding a language feature

Prefer to land it end to end: parse → type/shape-check → lower → run via the interpreter, with a
`tests/run/<name>.wk` program that demonstrates it. Fixture rules:

- Directives, in leading `//` comments: `// RUN: <extra cli args>` (default `--run`),
  `// EXPECT-EXIT: <n>`, and one `// EXPECT-OUT: <line>` per expected stdout line, in order.
- A `tests/run` program is checked four ways: it runs under the interpreter, its behaviour is
  invariant across `-O0`/`-O1`/`-O2`/`-O3`, the **Cranelift native backend must agree with the
  interpreter** on stdout and exit code, and native optimization must be invariant too. The
  interpreter is the oracle; if the two disagree, the interpreter is right until proven otherwise.
- It must also lower end to end: `crates/wukongc/tests/emit.rs` requires every `tests/run` program
  to reach `--emit=mir-high`, `--emit=mir` and `--emit=llvm-ir` at both `-O0` and `-O2`.
- **A program whose point is that it is rejected goes in `tests/fail`**, with
  `// EXPECT-CODE: <code>` on line 1 — not in `tests/run`.

`examples/` is a gated corpus, not a scratch directory. `crates/wukongc/tests/examples.rs` runs
every example through the real binary and requires the interpreter and the Cranelift backend to
agree on stdout and exit code at -O0 and -O2, with interpreter optimization observationally invariant
across -O0/-O1/-O2/-O3. Each example carries a declared class and anything unlisted defaults to
`Runs` (exit 0, non-empty stdout) — **a
new example must execute, or be declared** in `class_of`. The existing exceptions are `matmul.wk` /
`softmax.wk` / `vadd.wk` (`NotLowered`: exit 1 with `error[C0001]`), `gpt2_config.wk` (`NoEntry`: a
constants-only module), and the three GPT-2 programs that early-return when the weight blob is
absent (`DataGuarded`). Prefer `tests/run/` for a feature fixture and `examples/` only for a
showcase program.

Update `docs/language-guide.md` and its maturity legend.

## Adding a MIR `Op`

Prefer *not* to: `abs`, `exp`, `silu` and friends are built from existing primitives in
`wukong_mir_build` and touch none of this. If you must, the compiler finds the sites for you
(non-exhaustive match) — build the workspace and work the list:

1. `crates/wukong_mir/src/inst.rs` — the `Op` variant (plus any parameter enum, e.g. `RoundMode`),
   re-exported from `src/lib.rs` if public.
2. `crates/wukong_mir/src/verify.rs` — a `check_op` arm.
3. `crates/wukong_mir/src/print.rs` — a `fmt_op` arm.
4. `crates/wukong_interp/src/lib.rs` — an `eval` arm (per-lane for a vector type). This is the
   oracle: get it right first.
5. `crates/wukong_codegen_cranelift/src/lib.rs` — a `lower_inst` arm.
6. `crates/wukong_codegen_llvm/src/lib.rs` — an `emit_inst` arm (text only, not the oracle).
7. `crates/wukong_opt/src/lib.rs` — `map_op_uses` **and** `each_op_use` (and
   `map_term_uses`/`each_term_use` if you touched a terminator), plus `has_side_effects` if the op
   writes memory or can trap. These helpers are the single source of truth for which operands an
   `Op` reads; miss one and the passes silently corrupt SSA.
8. `crates/wukong_opt/src/cse.rs` — a `pure_key` arm if the op is cacheable (include *every*
   parameter in the key), and `safe_to_hoist` in `crates/wukong_opt/src/licm.rs` (default `false` if
   it can trap or write memory). If the op **produces or consumes a pointer**, also
   `crates/wukong_opt/src/alias.rs`: a new pointer-producing op must get a `Prov` (default
   `Unknown`), and a new *writer* must be classified in `AliasInfo::may_clobber` — an op that writes
   memory and is not listed as a writer is an unsound no-alias answer, i.e. a miscompile. That match
   is **exhaustive with no `_` arm on purpose**, so the compiler stops you rather than letting the
   omission pass silently; keep it that way. The escape scan needs no such care in the unsafe
   direction — its fallback arm marks *every* operand as escaping, so a new op is conservative by
   default there; add an arm only to buy back precision for a pure *addressing* use.
9. `crates/wukong_codegen_gpu/src/lower.rs` — `lower_inst` **and** `lower_vec_inst`, plus
   `op_operands` in `src/fusion.rs` (these live behind `--features gpu`, so only
   `cargo check --features gpu --all-targets` sees them); and `crates/wukong_autodiff/src/lib.rs` —
   `diff_inst`, `remap_op`, `op_name`.

Front-end plumbing (`wukong_sema`'s intrinsic return type, `wukong_mir_build`'s intrinsic map and
both vectorizer arms) closes the loop. Gate with `cargo test -p wukongc --test run`.

## Adding a runtime kernel (and its recognizer)

A "recognized kernel" is a loop shape that `wukong_mir_build` pattern-matches and replaces with a
call to a hand-written `extern "C"` kernel in `wukong_runtime`. Six layers, all required:

1. **Kernel** — `crates/wukong_runtime/src/<family>.rs`: the AVX2 body, its **scalar twin**, the
   `#[no_mangle] pub unsafe extern "C"` serial entry, the `_parallel` twin, and `#[cfg(test)] mod
   tests` asserting AVX2 == scalar (including non-finite data), serial == parallel bit-for-bit, and
   degenerate shapes (a non-positive extent must be a no-op). Extents are `i64`.
2. **Re-export** — `mod <family>;` + `pub use <family>::{…};` in `crates/wukong_runtime/src/lib.rs`,
   op-code constants included.
3. **Cranelift seam** — `crates/wukong_codegen_cranelift/src/lib.rs`: the `const RT_*` name, a
   `RtFuncs` field plus its `declare_function(..., Linkage::Import, ...)`, the
   `rt_refs.insert(...)`, a name-match arm in the call lowering, and
   `builder.symbol(RT_*, wukong_runtime::…)` in **both** JIT builders (`jit_compile` and
   `jit_module`) — miss one and only that JIT entry point dies when the module is finalized. The
   object path needs no entry: `emit_object_ex` leaves every `wukong_*` symbol as an undefined
   import, and `wukong_driver::emit_native`'s rustc-driven link resolves it out of the
   `wukong_runtime` rlib. A `wukong_*` name that reaches codegen with an arity no declared ABI
   matches is reported as an ICE rather than silently lowering to nothing.
4. **Recognizer** — `crates/wukong_mir_build/src/lib.rs`: intern the name in the `GemmSyms` pool,
   add the `match_<family>` loop-shape matcher and the emit site. **Write the decline path with
   it**: any shape your kernel cannot represent must fall through to the scalar nest, which is
   always correct. Most recognizers are f32 (argmax/argmin produce i32 labels), but the family also
   covers bf16, f16 and int8 — `wukong_sgemm_bf16_nt`, `wukong_dot_f16`, `wukong_i8gemm_nt_deq` and
   friends. All of them only fire on half-open unit-step `for` ranges.
5. **Interpreter twin** — `crates/wukong_interp/src/lib.rs`: a name arm that copies slot memory into
   `Vec`s, calls **the same `wukong_runtime::` function** (always the *serial* one, even for a
   `_parallel` symbol), and writes the results back. Without it the interpreter aborts on the call.
6. **Fixture** — `tests/run/<name>.wk`, which `cargo test -p wukongc --test run` runs on both
   backends at every `-O` level.

If the kernel is differentiable, mirror it in `crates/wukong_autodiff/src/tape.rs` (`Syms`,
`is_kernel`, `diff_kernel_call`) — **including the `_parallel` spelling**: a `_parallel` recognizer
arm added without its tape counterpart breaks every `@parallel` backward, so run the full suite.

LANDMINE: kernel dispatch is **gate-blind**. The interpreter calls the same function, so a kernel
bug is identical on both backends and the differential gate stays green — the in-crate
scalar-twin/naive tests are the only real oracle. To confirm a program actually dispatches, grep the
output of `wukongc --emit=mir -O2 prog.wk` for `wukong_`.
