# wukong_codegen_cranelift

Native code generation for Wukong via Cranelift — **no LLVM toolchain required**. This is the fast
CPU execution path: `--backend=native` and `--emit=obj|exe`.

## Role in the pipeline

```
MIR (post-opt, verified)
  → jit_run(program, entry, interner)     # --run --backend=native
  → emit_object(program, interner)        # --emit=obj
  → wukong_driver::emit_native            # --emit=exe (link step, not in this crate)
```

`jit_run` JIT-compiles and executes in-process, capturing stdout and exit code for bit-exact
comparison with the interpreter oracle. `emit_object` writes a native object file with undefined
imports for `wukong_runtime` kernels and `rt_*` shims resolved at link time.

## Two codegen paths

| path | what | when |
|---|---|---|
| **Cranelift CLIF** | scalar + 128-bit vector SSA | default lowering for all MIR ops |
| **Raw AVX2 (`avx2.rs`)** | 256-bit machine code via `iced-x86` | `Op::VecKernelCall` recipes from the general vectorizer |

Cranelift's x64 vector ISA caps at 128-bit (`f32x4`). The `avx2` module closes the gap for 256-bit
`f32x8` bodies — same approach as `wukong_runtime`'s hand-written microkernels.

### AVX2 gate

The 256-bit path requires **Windows x86-64 + AVX2 + FMA3**. `avx2::host_supports_kernels()` delegates
to `wukong_mir::host_supports_vec_kernels()` (single source of truth); `wukong_mir_build` consults
the same predicate before attaching a `VecKernel` recipe. Hosts outside the gate stay on 128-bit
CLIF strips (result-identical).

## Runtime binding

Generated code calls:

- **`rt_*` shims** defined in `lib.rs` — `print`/`println`/`assert`, `fmod`, heap alloc/free
- **`wukong_runtime` kernels** — the same `extern "C"` symbols the interpreter uses

## Dependencies

| crate | role |
|---|---|
| `wukong_mir`, `wukong_span`, `wukong_backend` | MIR lowering seam |
| `wukong_runtime` | kernel symbols for JIT binding / object imports |
| `wukong_opt` | `alias` provenance → Cranelift `AliasRegion` on loads/stores |
| `cranelift-*` 0.124 | codegen, frontend, JIT, object emission |
| `iced-x86` | raw AVX2 assembler (`code_asm` feature) |
| `rayon` | optional parallel per-function codegen in `emit_object` (`WUKONG_PAR_CODEGEN=0` kills) |

## Layout

| path | contents |
|---|---|
| [`src/`](src/) | `lib.rs` (lowering + JIT + object), `backend.rs`, `avx2.rs`, tests |

## Tests

`cargo test -p wukong_codegen_cranelift --lib` — JIT correctness, interpreter-vs-native differential,
optimization invariance, kernel fuzzing (`fuzz.rs`, `fuzz_grammar.rs`).

Failed assertions stop the current JIT function immediately and propagate through direct callers
before they can observe a return value or execute subsequent side effects. Only functions that
transitively call `assert` require status polling. This uses ordinary generated returns rather than
unwinding through an `extern "C"` frame. `run()` retains its `Err("assertion failed")` contract;
the timing-only `call()` returns exit code 1 and resets failure state for each invocation.

Assertion-capable functions cannot currently be taken as callback addresses: native compilation
rejects them with a cancellation-ABI diagnostic. This includes parallel callbacks; a shared flag
alone cannot stop effects already running on another worker. AOT Rust/C shims terminate at failure
with diagnostic `assertion failed` and exit 1. Previously emitted AOT output remains visible;
the existing `--run` error API discards captured output.
