# Mercury Compiler Internals

This document describes how `mercuryc` is built: the crate layering, the pipeline, the MIR, and the
key design bets. It is aimed at contributors.

## Design bets

1. **Interpreter-first, LLVM-quarantined.** The entire front-end, optimizer, and a from-scratch MIR
   interpreter build and test with plain `cargo test` on any machine — no LLVM required. LLVM
   (textual IR + `clang`) is only needed for native object/exe emission. The interpreter is both the
   always-available execution path and the differential-testing oracle.
2. **One progressively-lowered SSA MIR.** Instead of separate HIR/MIR/LIR, there is a single
   block-structured SSA IR that is born *High* (structured tensor/loop ops) and rewritten *down to
   Low* (scalar SSA, explicit loops, SIMD ops). A `MirLevel` invariant is enforced by a verifier.
3. **Block-parameter SSA, not phi nodes.** Basic blocks take typed parameters; branches pass
   arguments. This is simpler to construct and verify than phi placement and maps cleanly onto LLVM
   phis at codegen time.
4. **Shapes are checked like const generics.** Tensor dimensions are generic parameters; static dims
   unify by equality, symbolic dims by binding, `?` defers to runtime.

## Crate layering

The workspace is a virtual cargo workspace; every crate is prefixed `mercury_`. Dependencies flow
strictly downward (no cycles):

```
mercury_span      Span, SourceMap (line index), string Interner / Symbol
mercury_diag      Diagnostic, Severity, error-code catalog, JSON + terminal renderers
mercury_lexer     &str -> tokens (+ @attributes, error recovery)
mercury_ast       AST nodes, NodeId, pretty-printer
mercury_parser    recursive-descent + Pratt expressions -> AST
mercury_types     Ty, Scalar, Shape/Dim/Layout — the shared semantic type vocabulary
mercury_sema      name resolution + type checking + SHAPE checking
mercury_mir       MIR data, builder, pretty-printer, verifier, MirLevel
mercury_mir_build typed AST -> MIR (alloca-per-local lowering)
mercury_opt       pass manager + transforms (simplify, simplify-cfg, dce, cse)
mercury_backend   `Backend` trait + `Artifact`
mercury_interp    zero-dependency MIR interpreter backend (+ oracle)
mercury_codegen_llvm  textual LLVM IR backend
mercury_runtime   C-ABI arena allocator + deterministic parallel_for
mercury_driver    Session + compile() pipeline + --emit handling
mercuryc          thin CLI binary
mercury_bench     optimizer-effectiveness + interpreter-timing harness
```

`mercury_types` is shared by sema and MIR; `mercury_mir` is independent of the front-end; both
backends sit behind the `Backend` trait in `mercury_backend`.

## Pipeline

```
source
  → lexer        (mercury_lexer::tokenize)
  → parser       (mercury_parser::parse_module_tokens)  -> AST
  → sema         (mercury_sema::check)                  name res, types, SHAPE check
  → mir_build    (mercury_mir_build::lower_program)     -> MIR (High)
  → opt          (mercury_opt::optimize)                fixpoint passes
  → backend      interpreter (--run) | LLVM (--emit=llvm-ir|obj|exe)
```

`mercury_driver::compile` orchestrates this and honors `--emit=<stage>` to stop early and print the
chosen artifact. Diagnostics from every stage flow through a single `emit_diag` honoring
`--error-format`.

## Diagnostics

Every diagnostic has a stable code (`E01xx` lexer, `E02xx` parser, `E03xx` name res, `E04xx` types,
`E05xx` shapes, `C0xxx` codegen). The catalog in `mercury_diag::catalog` gives each code a title and
an extended explanation surfaced by `mercuryc --explain <CODE>`. The terminal renderer is
rustc-style (carets, gutters, `--> file:line:col`); `--error-format=json` emits one JSON object per
diagnostic for tooling.

## MIR

A `Function` owns a value arena (`ValueId -> MirType`), a list of `BasicBlock`s, and an entry block.
Each block has typed parameters, a straight-line list of `Inst { result: Option<ValueId>, op: Op }`,
and exactly one `Terminator` (`Ret`, `Br`, `CondBr`, `Unreachable`). `Op` spans constants, binary/
comparison/cast ops, `Select`, memory (`Alloca`/`Load`/`Store`/`Gep`), and `Call`.

The front-end lowers in **clang style**: one `alloca` per local, with `load`/`store` on every use.
This keeps lowering simple and correct; promoting allocas to registers (mem2reg) is deferred to the
optimizer and to LLVM.

The **verifier** (`mercury_mir::verify`) checks that every used value is defined, types are
consistent, and CFG edges are valid. It runs in `--emit=mir` and can be enabled after every pass.

## Optimizer

`mercury_opt::PassManager` runs a list of function-level `Pass`es to a per-function fixpoint.

| Pass          | Level | What it does |
|---------------|-------|--------------|
| `Simplify`    | -O1   | constant folding + algebraic identities (`x+0`, `x*1`, `x*0`, `x^x`, `x&x`, `x\|x`, `x%1`) and integer self-comparison folding |
| `SimplifyCfg` | -O1   | constant-branch folding + unreachable-block pruning (with renumbering) |
| `Dce`         | -O1   | remove pure instructions whose results are unused, and dead allocas |
| `Cse`         | -O2   | local value numbering with alloca-aware load forwarding |
| `Dse`         | -O2   | dead-store elimination (overwritten stores to a slot with no intervening read) |

Because the front-end emits alloca-based IR, `Cse` forwards loads (tracking the current value of
each alloca slot, invalidated by unknown-pointer stores or calls) so that redundant pure ops built
on reloaded values actually collapse; `Dse` is its dual, removing stores that are overwritten before
being read. Float predicates are never folded on self-comparison (NaN != NaN).

The opt pipeline is guarded by a **differential test**: every end-to-end program is run at -O0 and at
-O1/-O2/-O3 and must produce identical stdout and exit code. The `mercury_bench` crate reports the
instruction-count reduction the optimizer achieves per program.

## Interpreter

`mercury_interp` is a zero-dependency CFG walker over MIR. `Value` is `Int(i128) | Float(f64) |
Ptr(usize) | Unit`; a step limit guards against runaway loops. `run_with_output` returns
`(exit_code, stdout)`; intrinsics like `print` format into the captured stdout buffer. The
interpreter executes both High and (eventually) Low MIR identically, with deterministic floating
point, so it is a sound oracle for differential testing against the LLVM backend.

## Testing strategy

- **Unit tests** per crate (lexer, parser, sema, MIR verifier, opt passes, interpreter).
- **End-to-end** (`tests/run/*.mer`): the real `mercuryc` binary compiles and runs each program;
  stdout/exit are checked against `// EXPECT-*` directives embedded in the file.
- **Differential**: opt-level invariance today; interpreter-vs-LLVM once native codegen is wired.

All of the above runs with `cargo test` and no external toolchain.
