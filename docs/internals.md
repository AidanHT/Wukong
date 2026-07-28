# Wukong Compiler Internals

This document describes how `wukongc` is built: the crate layering, the pipeline, the MIR, and the
key design bets. It is aimed at contributors.

## Design bets

1. **Interpreter-first, native code without LLVM.** The entire front-end, optimizer, and a
   from-scratch MIR interpreter build and test with plain `cargo test` on any machine — no LLVM
   required. Native code is produced by a **Cranelift** backend (`wukong_codegen_cranelift`): JIT
   for `--run --backend=native` and object/exe for `--emit`, again with no LLVM toolchain. (A textual
   LLVM IR backend also exists for `--emit=llvm-ir`.) The interpreter is the always-available
   execution path and the differential-testing oracle; the native backend is validated bit-for-bit
   against it (see Testing). See `BENCHMARKS.md` for the cross-language standing.
2. **One progressively-lowered SSA MIR.** Instead of separate HIR/MIR/LIR, there is a single
   block-structured SSA IR that is born *High* (structured tensor/loop ops) and rewritten *down to
   Low* (scalar SSA, explicit loops, SIMD ops). A `MirLevel` invariant is enforced by a verifier.
3. **Block-parameter SSA, not phi nodes.** Basic blocks take typed parameters; branches pass
   arguments. This is simpler to construct and verify than phi placement and maps cleanly onto LLVM
   phis at codegen time.
4. **Shapes are checked like const generics.** Tensor dimensions are generic parameters; static dims
   unify by equality, symbolic dims by binding, `?` defers to runtime.

## Crate layering

The workspace is a virtual cargo workspace; every crate is prefixed `wukong_`. Dependencies flow
strictly downward (no cycles):

```
wukong_span      Span, SourceMap (line index), string Interner / Symbol
wukong_diag      Diagnostic, Severity, error-code catalog, JSON + terminal renderers
wukong_lexer     &str -> tokens (+ @attributes, error recovery)
wukong_ast       AST nodes, NodeId, pretty-printer
wukong_parser    recursive-descent + Pratt expressions -> AST
wukong_types     Ty, Scalar, Shape/Dim/Layout — the shared semantic type vocabulary
wukong_sema      name resolution + type checking + SHAPE checking
wukong_mir       MIR data, builder, pretty-printer, verifier, MirLevel
wukong_mir_build typed AST -> MIR (alloca-per-local lowering; SIMD loop auto-vectorization)
wukong_opt       pass manager + analyses (cfg, dominators) + transforms (inlining,
                  mem2reg, simplify, simplify-cfg, simplify-phis, dce, cse, dse, licm)
wukong_autodiff  reverse-mode autodiff as a MIR->MIR transform (scalar + tensor-tape VJP
                  rules, fused AdamW; finite-difference-gated) — the training backward path (driven by --emit=grad / --train)
wukong_backend   `Backend` trait + `Artifact`
wukong_interp    zero-dependency MIR interpreter backend (+ oracle; lane-wise vector exec)
wukong_codegen_cranelift  native backend via Cranelift — JIT (--run) + object/exe, no LLVM
wukong_codegen_llvm  textual LLVM IR backend
wukong_codegen_gpu   GPU backend (--features gpu): PTX emit + cudarc driver-JIT — recognizer
                  offload (--backend=gpu) + general MIR→PTX (--backend=gpu-native), no CUDA toolkit
wukong_runtime   C-ABI arena + rayon parallel_for + the AVX2/FMA microkernels (GEMM, vmath,
                  reductions, norms, int8 — the symbols the recognizers dispatch to)
wukong_driver    Session + compile() pipeline + --emit / --backend handling
wukongc          thin CLI binary
wukong_bench     optimizer-effectiveness + interp-vs-native timing & equivalence gate
wukong_xbench    cross-language benchmark (Wukong vs C, C++, and Rust) — see BENCHMARKS.md
```

`wukong_types` is shared by sema and MIR; `wukong_mir` is independent of the front-end; all
backends sit behind the `Backend` trait in `wukong_backend`.

## Pipeline

```
source
  → lexer        (wukong_lexer::tokenize)
  → parser       (wukong_parser::parse_module_tokens)  -> AST
  → sema         (wukong_sema::check)                  name res, types, SHAPE check
  → mir_build    (wukong_mir_build::lower_program)     -> MIR (High)
  → opt          (wukong_opt::optimize)                fixpoint passes
  → backend      interpreter (--run) | Cranelift native (--backend=native / --emit=obj|exe)
                 | GPU (--features gpu: --backend=gpu offload, --backend=gpu-native MIR→PTX)
                 | textual LLVM IR (--emit=llvm-ir)
```

`wukong_driver::compile` orchestrates this and honors `--emit=<stage>` to stop early and print the
chosen artifact. Diagnostics from every stage flow through a single `emit_diag` honoring
`--error-format`.

## Diagnostics

Every diagnostic has a stable code (`E01xx` lexer, `E02xx` parser, `E03xx` name res, `E04xx` types,
`E05xx` shapes, `C0xxx` codegen). The catalog in `wukong_diag::catalog` gives each code a title and
an extended explanation surfaced by `wukongc --explain <CODE>`. The terminal renderer is
rustc-style (carets, gutters, `--> file:line:col`); `--error-format=json` emits one JSON object per
diagnostic for tooling.

## MIR

A `Function` owns a value arena (`ValueId -> MirType`), a list of `BasicBlock`s, and an entry block.
Each block has typed parameters, a straight-line list of `Inst { result: Option<ValueId>, op: Op }`,
and exactly one `Terminator` (`Ret`, `Br`, `CondBr`, `Unreachable`). `Op` spans constants, binary/
comparison/cast ops, `Select`, memory (`Alloca`/`Load`/`Store`/`Gep`), `Call`, the float/SIMD
primitives (`Splat`/`Fma`/`Sqrt`/`Round`), and address-of ops (`FuncAddr`, and `GlobalAddr`). A
string literal lowers to a read-only `.rodata` blob in `Program::statics` (a `StaticData { name,
bytes }`) addressed by `Op::GlobalAddr`, so a returned or threaded `*u8` stays valid after its
defining frame is gone. `Op::VecKernelCall` — a call into a synthesized 256-bit AVX2 kernel — is
covered under the vectorizer below.

The front-end lowers in **clang style**: one `alloca` per local, with `load`/`store` on every use.
This keeps lowering simple and correct; the optimizer's `mem2reg` pass then promotes those slots to
SSA registers (see below), which is what makes the value-based passes effective.

There is **no dedicated aggregate MIR type**. A tuple or struct is a flat, padded byte buffer whose
local value *is* its base pointer (the convention arrays already follow); `t.0` / `s.f` is a typed
`load`/`store` at the field's byte offset, and `mem2reg` leaves the slot in memory. Nested aggregates
(a struct/tuple field that is itself a struct, or an array of structs) lay out recursively — the
registry-aware layout helpers resolve a named-struct field that the leaf type crate marks unsized — and
an aggregate field initialized from a *non-literal* value is a leaf-precise deep copy, not a flat
`memcpy` (which would skip a padded non-leading scalar slot under the interpreter's slot-indexed
memory). Pointers/references reuse the same `Alloca`/`Load`/`Store`/`Gep` ops: `&mut x` takes a slot's
address, `*p` loads/stores through it, and `mem2reg` refuses to promote a slot whose address escapes,
so `-O0` ≡ `-O3`. A **constant-shape tensor** lowers like an array — the parameter is a base pointer
and a multi-dimensional index `a[i, j]` flattens to a row-major `Gep` — so the shape-typed surface
*executes*, not just shape-checks. A matmul written in that tensor notation (`c[i,j] = Σ a[i,k]·b[k,j]`,
both the dot-product and accumulate spellings) dispatches to the tuned `wukong_sgemm` microkernel
just like the flat `a[i*K+k]` form, because a 2-index access supplies its row stride from the
operand's inner tensor dimension. A *symbolic*-generic nest `matmul<M, N, K>` reaches the same
kernel too: its symbolic dims are threaded in as hidden runtime `i64` parameters, `emit_sgemm`
materializes each `Dim::Var` from its bound value, and the result is bit-identical to the
constant-shape matmul on both backends at any runtime size (`tests/run/generic_shape_matmul.wk`). The aggregate path is differentially gated
bit-for-bit against the interpreter by the `differential_tuple`/`differential_struct`/
`differential_nested_struct` Cranelift tests. Returning an aggregate *by value* from a function (and a
by-value aggregate parameter) lowers through a **MIR-level sret ABI** — the callee takes a hidden
leading destination pointer and returns void, `return Struct{..}` deep-copies into it, and the call
site allocates the buffer, passes it as the hidden first argument, and uses it as the call's value, so
no aggregate ever rides in a register and the two backends still agree
(`differential_struct_across_fns`/`differential_struct_return`).

The **verifier** (`wukong_mir::verify`) checks that every used value is defined, types are
consistent, and CFG edges are valid. It runs in `--emit=mir` and can be enabled after every pass.

## Optimizer

`wukong_opt::optimize` first runs a whole-program **inliner** (`-O2`), then `PassManager` runs a
list of function-level `Pass`es to a per-function fixpoint. Two shared analyses back them: `cfg`
(successors/predecessors, reverse postorder, reachability, unreachable-block pruning) and `dom`
(Cooper–Harvey–Kennedy immediate dominators, dominance frontiers, and the dominator tree).

| Pass            | Level | What it does |
|-----------------|-------|--------------|
| `inline_program`| -O2   | inline small, non-recursive **leaf** functions (whole-program), then drop callees left uncalled; runs before the function pipeline so the spliced code optimizes in context |
| `Mem2Reg`       | -O1   | promote scalar int/float `alloca` slots to block-parameter SSA via dominance-frontier phi placement and a dominator-tree rename |
| `Simplify`    | -O1   | constant folding + algebraic identities (`x+0`, `x*1`, `x*0`, `x^x`, `x&x`, `x\|x`, `x%1`) and integer self-comparison folding |
| `SimplifyCfg` | -O1   | constant-branch folding, straight-line block merging, and unreachable-block pruning (with renumbering) |
| `SimplifyPhis`| -O1   | drop dead and trivial block parameters that mem2reg introduced |
| `Dce`         | -O1   | remove pure instructions whose results are unused, and dead allocas |
| `Cse`         | -O2   | local value numbering with alloca-aware load forwarding |
| `Dse`         | -O2   | dead-store elimination (overwritten stores to a slot with no intervening read) |
| `Licm`        | -O2   | hoist loop-invariant, side-effect-free, non-trapping ops to the loop preheader |

`-O3` currently runs the same pass pipeline as `-O2`: there are no `-O3`-exclusive passes yet
(`PassManager::standard` adds passes at `-O1` and `-O2` only).

`Mem2Reg` is the keystone: the front-end's memory traffic hides constants, common subexpressions,
and induction variables, so promoting slots to SSA is what lets the rest of the pipeline fire. It
leaves arrays, address-taken, and pointer slots in memory; a read before any write becomes a zero
constant, matching the interpreter's zero-initialized memory. `Cse`/`Dse` still handle the residual
memory (load forwarding and its dual) for the slots that stay in memory. `Licm` uses the dominator
analysis to find natural loops and only hoists into loops that already have a preheader, so it is
always legal; loads, stores, calls, and integer division are never moved. Float predicates are never
folded on self-comparison (NaN != NaN).

The opt pipeline is guarded by a **differential test**: every end-to-end program is run at -O0 and at
-O1/-O2/-O3 and must produce identical stdout and exit code. The `wukong_bench` crate reports, per
program, the IR-op reduction and the -O0-vs--O3 interpreter speedup, and exits non-zero if any
program that lowers cleanly disagrees across optimization levels — a soundness gate over every
benchmark kernel. Across the run suite and kernels, -O3 removes **~42% of IR ops (~48–54% on the
heavy transformer/GEMM kernels)** and runs ~1.5–2.5x faster than -O0 under the interpreter.

## Interpreter

`wukong_interp` is a zero-dependency CFG walker over MIR. `Value` is `Int(i128) | Float(f64) |
Ptr(usize) | VecRef(u32) | Unit`; a step limit guards against runaway loops, and the walk runs on a
scoped 512 MiB-stack worker thread so deep recursion does not overflow the host stack (an uncatchable
overflow would otherwise abort the oracle). `run_with_output` returns `(exit_code, stdout)`;
intrinsics like `print` format into the captured stdout buffer. `f32` ops are computed in `f32`
(single rounding), and an `int → f32` cast rounds straight to `f32` (not via `f64`), so the
interpreter matches native's single conversion bit-for-bit. SIMD
vectors are executed lane-wise via a side arena (`VecRef` indexes it, keeping `Value` `Copy`). The
interpreter is the sound oracle for differential testing.

## Native backend (Cranelift) and the vectorizer

`wukong_codegen_cranelift` lowers Low MIR to Cranelift IR — an almost 1:1 map (block-parameter SSA,
signless ints, explicit `alloca`/`load`/`store`/`gep`). It JIT-compiles in-process for
`--backend=native` and emits a host object for `--emit=obj|exe` — for `exe` it prefers a rustc-driven link that pulls
in the `wukong_runtime` kernels (so a recognized-kernel program and string `.rodata` both resolve),
falling back to a `cc`/`$CC` C-runtime link. The
only semantic bridges to stay identical to the interpreter: divide-by-zero yields 0, float→int casts
saturate, and `i1` results are masked to their low bit.

SIMD **auto-vectorization** happens in `wukong_mir_build` while lowering a `for` loop: a
straight-line elementwise body over unit-stride array accesses (plus loop-invariant splats, and
`if`/`else` value-expressions via if-conversion to a vector compare + blend) lowers to vector
`load`/`store`/`bin`/`cmp`/`select` and `Op::Splat`. The loop is strip-mined into an unrolled main
loop (`VEC_UNROLL` independent 128-bit groups per iteration — recovering AVX-class throughput from
SSE via dual-issue), a single-vector loop, and a scalar remainder, all sharing one index slot. It
bails to scalar on any loop-carried dependence, non-unit stride, call, or mixed lane type, so lane
`k` always computes exactly what scalar iteration `base+k` would. `@parallel` per-thread chunks go
through the same vectorizer, so they run SIMD × cores.

Cranelift's CLIF vector ISA still caps at 128-bit — a 256-bit `f32x8` SSA value is rejected at
`define_function` (the `cranelift_still_rejects_f32x8` / `p4_probe_vec256_ops` tripwires keep that
documented). So for an eligible f32 elementwise body the vectorizer captures the loop as a flat,
backend-agnostic `VecKernel` recipe and, when register pressure fits, emits one `Op::VecKernelCall`:
the Cranelift backend assembles that recipe to **true 256-bit AVX2 machine code** via `iced-x86`
(`wukong_codegen_cranelift::avx2`, VEX-encoded), while the interpreter marshals the *same* recipe
lane-wise — so the two stay bit-identical (elementwise lanes carry no reassociation, so the gate
holds element-for-element). `WUKONG_P4_NO_256` forces the 128-bit path — a same-run A/B knob and a
kill-switch. The float-reduction path has an analogous 256-bit kernel, gated to large trips
(`VEC256_REDUCTION_MIN_TRIP` = 2048 elements) because the out-of-line call loses to the inlined
128-bit reduction on small arrays.

Two refinements target the dominant ML arithmetic. A float `x + y*z` **contracts to a fused
multiply-add** (`Op::Fma`, one rounding, a hardware `vfmadd`) in both the scalar and vector lowering
paths; the interpreter mirrors it with `mul_add`, so the two backends stay bit-identical. And a
**float reduction** `for k in .. { s = s + x[k]*y[k] }` (or `s += ..`) is recognised and lowered to
`VEC_UNROLL` independent vector-lane accumulators (each an FMA chain), a single-vector cleanup loop,
a horizontal reduce of the lanes into `s`, and a scalar remainder. This reassociates the sum (the
standard reduction optimization) — sound because both backends execute the same reassociated MIR, so
the differential oracle still holds bit-for-bit. Reductions vectorize only on the sequential path,
never the `@parallel` one (folding into a shared accumulator across threads would race).

## Matmul recognition → tuned GEMM microkernel

The width that matters most for ML is the matmul inner product, and it is exactly where a 128-bit
general vectorizer leaves performance on the table. So matmul gets a dedicated path: `wukong_mir_build`
**recognizes a matmul loop nest** at the AST level — the `ikj` accumulate form (`c[i*N+j] +=
a[i*K+k]*b[k*N+j]`, with or without a per-row zero-init or a `let aik` binding) and the textbook `ijk`
dot-product form (`for j { let s=0; for k s+=a[i,k]*b[..]; c[i*N+j]=s }`), including the `C = A·Bᵀ`
(`nn.Linear`) spelling where B is indexed `[j*K+k]`. The recognizer (`recognize_matmul`) verifies the
strides describe contiguous row-major operands, then lowers the *whole nest* to a single call:
`wukong_sgemm` / `wukong_sgemm_nt` (serial) or their `_parallel` variants, chosen by the `@parallel`
attribute and the transpose flag.

The kernel itself (`wukong_runtime::gemm`) is a classic BLIS-style GEMM: a **6×16 register tile**
(12 live `__m256` accumulators, 12 FMAs per K-step), `MC/KC/NC` **cache blocking**, and **packed**
A/B panels streamed with unit stride — true **256-bit AVX2 + FMA** (the width Cranelift's IR cannot
express), runtime-detected with a scalar fallback. The parallel variant packs A and B once per K
block — the packing itself **fanned across cores** (each `MR`-row / `NR`-col panel is independent),
since with the C compute spread over ~14 cores a serial pack would be the Amdahl bottleneck — then
runs the C tile grid across cores. It also **falls back to the serial kernel below a work threshold**
(`m·n·k < 2²³`): on this P+E hybrid, cross-core wake/sync costs more than it saves only for the very
smallest matrices (the 2026-07-11 dynamic-claiming default lowered this gate from 2²⁶, so 256³ now
runs parallel at ~94–110% of MKL-all rather than deliberately serial). This is the same shape XLA/TVM/oneDNN lower a matmul op to, and
it is why the win over gcc/rustc's naive nest *grows* with size (their version falls out of cache; the
packed kernel does not).

Crucially this stays inside the differential oracle: the interpreter, on a `wukong_sgemm*` call,
**marshals its abstract `Value` memory into real f32 buffers and calls the identical kernel**, then
marshals the result back — so native and interpreter agree bit-for-bit despite the reassociated
accumulation (the parallel kernel is bit-identical to the serial one by construction).

## GPU backend

`wukong_codegen_gpu` (behind `--features gpu`) is a third execution path: being a compiler, it
**emits PTX text** and **driver-JIT-loads it via `cudarc`** (`cuModuleLoadData`), so no `nvcc`/CUDA
toolkit is needed — only the NVIDIA driver. It offers two modes. `--backend=gpu` runs the program on
an **offloading interpreter** (the CPU tree-walks every op as the oracle does, but recognized
GEMM/activation/reduction/norm calls execute on the device). `--backend=gpu-native` (the `GpuLower`
backend) instead lowers the **whole** MIR to PTX, so arbitrary non-recognized kernels run GPU-side
too; an eligible program can be fused into a single-block cooperative *megakernel* (one launch, no
host round-trips). The CPU↔GPU boundary is **tolerance-gated** (`c·√K·ε`) rather than bit-exact — the
GPU analogue of the CPU differential oracle — and the path stays optimization-invariant (`-O0` ≡
`-O3`). It builds but is inert without the feature, so plain `cargo test` is unaffected.

## Testing strategy

- **Unit tests** per crate (lexer, parser, sema, MIR verifier, opt passes, interpreter, vectorizer).
- **End-to-end** (`tests/run/*.wk`): the real `wukongc` binary compiles and runs each program;
  stdout/exit are checked against `// EXPECT-*` directives embedded in the file.
- **Differential**: `-O0`-vs-`-O{1,2,3}` invariance, **interpreter-vs-native (Cranelift)** on stdout
  and exit code (in `wukong_bench`'s equivalence gate and the cranelift unit tests), and vectorized
  kernels checked against independent scalar references across remainder-exercising sizes.

All of the above runs with `cargo test` and no external toolchain (Cranelift is a pure-Rust crate).
