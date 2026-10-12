# crates/ — the `wukong_*` compiler workspace

Wukong (`.wk`) is a language for user-written ML/DL tensor kernels with **compile-time shape
safety** — tensor dimensions live in the type system, so a shape mismatch at a call site or an
index is a compile error, not a runtime crash. These crates are the **compiler front end and
pipeline**, not a GPU library. Execution backends (interpreter, native JIT, optional GPU) sit
downstream of the stages listed here.

Every compiler stage lives here as its own crate. **`[dependencies]` flow strictly downward**
(verified against every `Cargo.toml`; there are no cycles), and `wukong_backend` is the declared
seam every execution/emission target plugs into.

## Crates, in dependency order (low-level first)

| crate | one line |
|---|---|
| `wukong_span` | Leaf: `Span`/`SourceId`, `SourceMap`, string `Interner`/`Symbol`, and a zero-dep FxHash. **Empty `[dependencies]` — keep it that way.** |
| `wukong_runtime` | Leaf: ~150 `extern "C"` AVX2/FMA microkernels + scalar twins, the thread pool, `Arena`, heap/file-I/O/clock intrinsics. |
| `wukong_diag` | `Diagnostic`/`Label`/`Severity`/`DiagnosticSink`, the stable error-code catalogue (29 entries) behind `--explain`, and the terminal + JSON renderers. |
| `wukong_lexer` | `&str` → flat `Vec<Token>` + diagnostics. Never aborts; owns E01xx. |
| `wukong_ast` | Data-only AST + the deterministic pretty-printer. Sema annotates through `NodeId` side tables and never rewrites it. Owns `BinOp::fold_const_len`. |
| `wukong_parser` | Recursive descent (items/stmts/types/patterns) + Pratt expressions → AST. Allocates `NodeId`s, recovers from everything; owns E02xx. |
| `wukong_fmt` | The source formatter behind `wukongc fmt`: AST + the lexer's comment side channel → canonical text, self-checked to re-parse to the identical AST with the same comments and tokens (refuses otherwise). Also `--emit=doc`'s Markdown page (`doc.rs`), whose signatures its printer writes. Used by `wukongc` and `wukong_driver`. |
| `wukong_types` | The semantic type vocabulary (`Scalar`, `Dim`, `Shape`, `Layout`, `Ty`) and the **layout authority** (`size_of`/`align_of`/`tuple_offsets`). |
| `wukong_sema` | Name resolution, type inference, generics, match exhaustiveness, and tensor **shape checking**. Owns E03xx/E04xx/E05xx (except E0305, and two `E0401` sites in `wukong_mir_build`). |
| `wukong_mir` | The typed block-parameter SSA IR: types, `Builder`, printer, verifier, and `VecKernel` (the AVX2 recipe). No passes. |
| `wukong_mir_build` | AST → MIR (alloca-per-local), monomorphization, `@parallel` outlining, and **every kernel recognizer**. Owns C0001 plus two `E0401` sites. |
| `wukong_opt` | Pass manager + CFG/dominator analyses + the MIR→MIR transforms (mem2reg, simplify, simplify-cfg, phis, dce, cse, dse, licm, inlining). |
| `wukong_autodiff` | Reverse-mode AD as a MIR→MIR transform: scalar VJP rules + the buffer tape over recognized kernels + a fused AdamW kernel. |
| `wukong_backend` | The seam: the `Backend` trait and the `Artifact` enum in one ~50-line file. No logic, no impls. |
| `wukong_interp` | Tree-walking MIR interpreter — the default backend **and the differential oracle**. Slot-indexed memory; calls the same `wukong_runtime` kernels. |
| `wukong_codegen_llvm` | MIR → **textual** LLVM IR. Inspection path only; no inkwell, no LLVM installed, never assembles or runs. |
| `wukong_codegen_cranelift` | The real native backend: MIR → CLIF → JIT (`jit_run`) or host object (`emit_object`), plus the raw-AVX2 256-bit encoder and both fuzzers. |
| `wukong_codegen_gpu` | GPU backend behind `--features gpu`: PTX generators + `cudarc` driver-JIT, the recognizer offload, general MIR→PTX (`lower.rs`), the megakernel, serving/training stacks. |
| `wukong_driver` | Sequences the pipeline; owns `SourceMap`/`Interner`/`DiagnosticSink`, the `Options` types, the import loader (E0305), the `--emit=exe` link, and `gpu_accel`. |
| `wukongc` | The CLI binary. Parses argv, validates flag *combinations*, delegates to `wukong_driver::compile`; the `fmt` subcommand delegates to `wukong_fmt`. Its `tests/` hold every CLI-level gate. |
| `wukong_bench` | Leaf binary `wukong-bench`: optimizer effectiveness, interp/native timing, per-pass and per-stage compile time, compiler-vs-compiler. |
| `wukong_xbench` | Leaf binary `wukong-xbench`: cross-language benchmark (Wukong vs C/C++/Rust/MKL/matrixmultiply/PyTorch) through one harness. |

`wukong_span` and `wukong_runtime` depend on no other workspace crate. `wukong_bench` and
`wukong_xbench` call the stage crates directly rather than going through `wukong_driver`, so the
driver stays a thin sequencer with no instrumentation woven through it.

## The layering rule

- A crate's `[dependencies]` may point only **downward** in the table above. If you find yourself
  wanting an upward edge, the abstraction belongs lower (or the caller belongs higher). Two *dev*-
  dependency edges do point upward — `wukong_opt` and `wukong_autodiff` both test against
  `wukong_interp` as their oracle; that is allowed, since dev-deps never enter the library graph.
- **`wukong_backend` is the seam.** A "thing that executes or emits a `wukong_mir::Program`" declares
  itself by implementing `Backend` and returning an `Artifact`. The trait is a *declared* seam, not
  the live dispatch path — `wukong_driver::compile` matches on its own `BackendKind` and calls the
  concrete free functions.
- `wukong_span` sits under every crate except `wukong_runtime`, and `wukong_mir` under everything from
  `wukong_mir_build` upward: a dependency added to either propagates to nearly the whole build.
- Only the **GPU backend** is opt-in (`--features gpu`, forwarded `wukongc` → `wukong_driver` →
  `wukong_codegen_gpu`). Cranelift native codegen and the textual-LLVM emitter are always built and
  need no external toolchain.

## Front-end pipeline (this README's scope)

```
input.wk
 → lex   wukong_lexer::tokenize          (--emit=tokens)
 → parse wukong_parser::parse_module_*   (--emit=ast; --emit=doc via wukong_fmt::doc)
 → …     wukong_sema, wukong_mir_build, wukong_opt, backends …
```

Shared infrastructure: `wukong_span` (positions, interning) and `wukong_diag` (diagnostics +
catalogue). See each crate's `README.md` for entry points, invariants, and how to test just that
crate (`cargo test -p <crate> --lib`).

## Workspace conventions (root `Cargo.toml`)

- **Virtual workspace**, `resolver = "2"`, `members = ["crates/*"]` — a new directory under `crates/`
  is a member automatically. No root package.
- `[workspace.dependencies]` registers **every internal crate by path**. Three crates are
  intentionally absent from that list because nothing depends on them: `wukong_bench`,
  `wukong_xbench` (leaf binaries), and `wukongc` (the binary itself).
- `[profile.release]`: `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`, **`panic = "abort"`**.
