# Mercury / mercuryc

Mercury (`.mer`) is a low-level systems language for ML/DL tensor kernels with **compile-time shape
safety** (tensor shapes live in the type system). `mercuryc` is its compiler — a Rust workspace whose
front-end, optimizer, and a from-scratch MIR interpreter build and test with plain `cargo test` (no
LLVM). Native codegen is opt-in behind `--features llvm`.

## Build / test / run

```sh
cargo build                                  # the compiler (interpreter backend, no LLVM)
cargo test                                   # unit + golden + e2e tests (no toolchain needed)
cargo run -p mercuryc -- --help
cargo run -p mercuryc -- --run examples/fib.mer            # compile + interpret
cargo run -p mercuryc -- --emit=mir -O2 examples/dot.mer   # dump an intermediate stage
cargo run -p mercury_bench --release -- tests/run examples bench/kernels   # optimizer report
cargo build --features llvm                  # native codegen path (requires LLVM 19; see below)
```

CLI flags (`crates/mercuryc/src/main.rs` → `mercury_driver::Options`):
- `--run` — compile and execute via the interpreter (the default execution path here).
- `--emit=<stage>` — emit one artifact and stop: `tokens`, `ast`, `mir-high`, `mir` (alias `mir-low`),
  `llvm-ir`, `obj`, `exe` (default `exe`). Artifacts go to stdout; diagnostics to stderr.
- `-O0|-O1|-O2|-O3` — optimization level (default `-O0`).
- `-o <path>`, `--color=auto|always|never`, `--error-format=human|json`, `--explain <CODE>`,
  `-h/--help`, `-V/--version`.
- `--emit=obj|exe` shells out to `clang` on the textual IR; with no LLVM installed it returns
  exit code 2 (`UNIMPLEMENTED`) after writing the `.ll` sidecar. Use `--run` or `--emit=llvm-ir` instead.

## Pipeline

```
source.mer
  → lexer       mercury_lexer::tokenize              -> tokens
  → parser      mercury_parser::parse_module_tokens  -> AST
  → sema        mercury_sema::check                  name res, types, SHAPE check
  → mir_build   mercury_mir_build::lower_program     -> MIR (High, alloca-per-local)
  → opt         mercury_opt::optimize                fixpoint SSA passes -> MIR (Low)
  → backend     mercury_interp (--run) | mercury_codegen_llvm (--emit=llvm-ir|obj|exe)
```

`mercury_driver::compile` orchestrates this and honors `--emit=<stage>` to stop early.

## Crate map (dependency order, low-level first)

- `mercury_span` — source spans, source map, string interning primitives.
- `mercury_diag` — diagnostic model, error-code catalog, terminal/JSON renderers.
- `mercury_lexer` — source text to flat token stream plus diagnostics.
- `mercury_ast` — data-only AST: parser builds, sema annotates via NodeId side tables.
- `mercury_parser` — recursive-descent + Pratt parser: lexer tokens to AST.
- `mercury_types` — type vocabulary: scalars, SIMD vectors, shape-typed tensors, layouts.
- `mercury_sema` — name resolution, type checking, and tensor shape checking.
- `mercury_mir` — typed block-structured SSA IR: types, builder, printer, verifier.
- `mercury_mir_build` — lowers type-checked Mercury AST into MIR (alloca-per-local).
- `mercury_opt` — pass manager, CFG/dominator analyses, and SSA MIR transforms.
- `mercury_backend` — `Backend` trait + `Artifact` enum: the MIR-to-execution/emission seam.
- `mercury_interp` — zero-dep tree-walking MIR interpreter; default backend and differential oracle.
- `mercury_codegen_llvm` — LLVM backend: lowers MIR to textual LLVM IR.
- `mercury_runtime` — minimal runtime: bump arena allocator and sequential `parallel_for`.
- `mercury_driver` — orchestrates the compile pipeline; owns SourceMap/Interner and honors `--emit`.
- `mercuryc` — CLI binary parsing args and delegating compilation to `mercury_driver`.
- `mercury_bench` — benchmark harness: optimizer effectiveness, interpreter timing, equivalence gate.

Dependencies flow strictly downward (no cycles); every crate is prefixed `mercury_` (binary is `mercuryc`).

## Mandate: performance first

**The goal is to beat C, C++, and Rust on the metrics that matter for an ML/DL compiler** — generated
kernel throughput (matmul/conv/attention GFLOP/s, elementwise/reduction GB/s), compile time, and
shape safety. Performance wins override every *stylistic* convention below. You are free to: adopt
wider SIMD (AVX2/AVX-512) by any means including raw machine-code microkernels, add a new backend or
new dependencies, break "interpreter-first" primacy, add domain-aware passes (tiling, packing,
register-blocking, op-graph fusion), and rewrite any documentation that is out of date. If something
blocks a higher number, remove it.

**The one hard invariant — correctness.** Every backend must agree with the interpreter oracle
bit-for-bit on the differential gate, and `-O0` must match `-O{1,2,3}` on stdout/exit. A fast
compiler that miscompiles is worthless; this gate is precisely what makes aggressive optimization
*safe*, so it stays. (For reassociated float reductions, the documented exception is that the
reassociated form is the oracle — all backends run the same reassociated IR and must still agree.)

## Architecture conventions (change freely if they block performance)

- **One progressively-lowered SSA MIR** (not separate HIR/MIR/LIR): born *High* (structured
  tensor/loop ops), rewritten down to *Low* (scalar SSA). A `MirLevel` invariant is verifier-enforced.
- **Block-parameter SSA, not phi nodes.** Blocks take typed params; branches pass args. Maps to LLVM phis.
- **Shapes as const generics.** Static dims unify by equality, symbolic by binding, `?` defers to runtime.
- **Stable diagnostic codes.** `E01xx` lexer, `E02xx` parser, `E03xx` name-res, `E04xx` types,
  `E05xx` shapes, `C0xxx` codegen. `mercuryc --explain <CODE>` prints the extended explanation.
- **Optimizer fixpoint passes** (`-O1`: mem2reg, simplify, simplify-cfg, simplify-phis, dce;
  `-O2` adds whole-program inlining, cse, dse, licm). `mem2reg` is the keystone (alloca → SSA).
- The interpreter is retained as the **correctness oracle**, not as a performance ceiling. The native
  (Cranelift) path is the fast path and the default for benchmarking.

## Environment gotchas

- Development is on **Windows 11 with PowerShell** as the primary shell (a Bash tool is also available —
  use POSIX syntax there). Use absolute paths in agent threads.
- **LLVM is NOT installed here** (a physical fact, not a rule): `clang`/`llc` do not exist, so the
  textual-LLVM `--features llvm` path cannot link/run. The native path is **Cranelift** (pure Rust,
  builds and JITs here with zero external toolchain) plus any raw-codegen microkernels we add. Plain
  `cargo test` needs no toolchain. `gcc`/`g++`/`rustc` (MSYS2) *are* present — that's what `mercury_xbench`
  compiles the C/Rust baselines with.
- Wider SIMD: Cranelift historically rejected `f32x8` types in CLIF. That ceiling is a target to break,
  not a law — verify the current Cranelift's capability empirically, and where it can't reach, emit
  AVX2/AVX-512 microkernels directly (the differential gate keeps any such path honest).
- Test fixtures live at the **repo root** (`tests/run/*.mer`, `tests/fail/*.mer`, `examples/`,
  `bench/kernels/`), not under any crate; e2e tests spawn the real binary via `CARGO_BIN_EXE_mercuryc`.

## Where to look

- `docs/internals.md` — architecture, crate layering, MIR, optimizer, testing strategy.
- `docs/language-guide.md` — the language surface, with a maturity legend.
- `docs/roadmap.md` — what runs end-to-end, what's checked-only, what's planned, sharp edges.
- `docs/llvm-setup.md` — optional native-codegen toolchain (LLVM 19).
- `examples/*.mer` — runnable kernels (`dot`, `saxpy_array`, `matmul`, `relu`, `softmax`, …).
- `tests/run/` (e2e `// EXPECT-*` directives) and `tests/fail/` (compile-fail `// EXPECT-CODE:`).
- Each crate has its own `CLAUDE.md` with layout, key types, connections, and gotchas.
