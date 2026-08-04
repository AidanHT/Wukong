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
   against it (see Testing). See `BENCHMARKS.md` for the cross-language standing. One platform
   caveat: the 256-bit raw-AVX2 kernel path (`wukong_codegen_cranelift::avx2`) encodes the Win64
   argument registers literally and has no in-kernel dispatch, so `assemble_kernel` refuses on any
   host that is not x86-64 Windows with runtime AVX2+FMA3. `WUKONG_P4_NO_256=1` forces the portable
   128-bit path. Everything else — front end, optimizer, interpreter, and the ordinary Cranelift
   lowering — is host-agnostic.
2. **One SSA MIR, built low.** Instead of separate HIR/MIR/LIR there is a single block-structured
   SSA IR, and `wukong_mir_build` produces scalar SSA (explicit loops, `alloca`/`load`/`store`, SIMD
   ops) *directly* — there is one lowering stage, not two. The `MirLevel` enum and `Program::level`
   field exist but are **inert**: nothing constructs `High`, `Program::new` starts at `Low`, no pass
   ever branches on `level` (the only reads copy it verbatim into a newly built `Program` —
   `wukong_driver` lib.rs:781, 1032), and the verifier does not check it, so do not write a pass that
   branches on it.
   (`--emit=mir-high` means *pre-optimization* MIR and `--emit=mir` — alias `mir-low` — means
   *post-optimization* MIR; both print the same IR at different points, and neither has anything to
   do with `MirLevel::High`.) What *is* enforced is well-formedness — single definition per value,
   definitions dominating their uses, per-op types, entry-block and terminator rules — by the
   verifier described in the MIR section.
3. **Block-parameter SSA, not phi nodes.** Basic blocks take typed parameters; branches pass
   arguments. This is simpler to construct and verify than phi placement and maps cleanly onto LLVM
   phis at codegen time.
4. **Shapes are checked like const generics.** Tensor dimensions are generic parameters. At a **call
   site** the callee's dims are inference variables bound from the argument shapes; inside a
   function's **own body** (the return-shape and binop checks) the same dims are universally
   quantified and match by *identity*, which is what stops a generic function declaring a return
   shape its body does not actually produce. Static dims unify by equality; `?` (`Dim::Dynamic`) is
   the lenient escape hatch in both modes and defers to runtime. A tensor's `layout` is part of the
   type too, so a `.col_major` value cannot be routed through a `Contiguous`-typed callee (E0502).

## Crate layering

The workspace is a virtual cargo workspace (`members = ["crates/*"]`, no root package); every crate
is prefixed `wukong_` except the CLI binary `wukongc`. Dependencies flow strictly downward (no
cycles):

```
wukong_span      Span, SourceMap (line index), string Interner / Symbol
wukong_runtime   the AVX2/FMA microkernels the recognizers dispatch to (GEMM/GEMV, vmath,
                  reductions, norms, int8, bf16/f16) + the rayon-backed `wukong_parallel_for` that
                  `@parallel` lowers to — a leaf: no workspace deps, and both the interpreter and
                  the native backend call into it. The bump `Arena` and the sequential
                  `parallel_for` are reference surface with no caller anywhere in the workspace
                  (the kernels use thread-local `Vec` scratch)
wukong_diag      Diagnostic, Severity, error-code catalog, JSON + terminal renderers
wukong_lexer     &str -> tokens (+ @attributes, error recovery)
wukong_ast       AST nodes, NodeId, pretty-printer; owns `BinOp::fold_const_len` +
                  `MAX_CONST_ARRAY_LEN` (= u32::MAX) — the const-array-length folder SHARED by
                  sema and mir_build so the two cannot disagree on a length
wukong_parser    recursive-descent + Pratt expressions -> AST
wukong_types     Ty, Scalar, Shape/Dim/Layout — the shared semantic type vocabulary
wukong_sema      name resolution + type checking + SHAPE checking
wukong_mir       MIR data, builder, pretty-printer, verifier, `VecKernel` (the 256-bit AVX2
                  recipe). `MirLevel` lives here too but is inert — see Design bet 2.
wukong_mir_build typed AST -> MIR (alloca-per-local lowering; SIMD loop auto-vectorization)
wukong_opt       pass manager + analyses (cfg, dominators) + transforms (inlining,
                  mem2reg, simplify, simplify-cfg, simplify-phis, dce, cse, dse, licm) plus
                  the `alias` provenance analysis every memory transform queries
wukong_autodiff  reverse-mode autodiff as a MIR->MIR transform (scalar + tensor-tape VJP
                  rules, fused AdamW; finite-difference-gated) — the training backward path (driven by --emit=grad / --train)
wukong_backend   `Backend` trait + `Artifact`
wukong_interp    from-scratch MIR interpreter backend — the reference oracle; lane-wise vector
                  exec. Needs no toolchain, but does link `wukong_runtime` so a recognized-kernel
                  call executes the identical microkernel the native backend does
wukong_codegen_cranelift  native backend via Cranelift — JIT (--run) + object/exe, no LLVM
wukong_codegen_llvm  textual LLVM IR backend
wukong_codegen_gpu   GPU backend (--features gpu): PTX emit + cudarc driver-JIT — recognizer
                  offload (--backend=gpu) + general MIR→PTX (--backend=gpu-native), no CUDA toolkit
wukong_driver    Options/EmitStage/BackendKind + compile() pipeline + the multi-file import
                  loader (owns E0305) + --emit / --backend handling + the --emit=exe link
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
  → parser       (wukong_parser::parse_module_tokens_from)  -> AST (NodeId watermark threaded)
  → loader       (wukong_driver::loader::load_imports)  splice `import`ed files -> one flat module
  → sema         (wukong_sema::check)                  name res, types, SHAPE check
  → mir_build    (wukong_mir_build::lower_program)     -> MIR (scalar SSA; always MirLevel::Low)
  → opt          (wukong_opt::optimize)                fixpoint passes
  → autodiff     (wukong_autodiff::grad)               --emit=grad / --train only; forces >= -O1
  → verify       (wukong_mir::verify, via wukong_driver::verify_or_ice)
                 gate on --run and on every AOT exit
  → backend      interpreter (--run) | Cranelift native (--backend=native / --emit=obj|exe)
                 | GPU (--features gpu: --backend=gpu offload, --backend=gpu-native MIR→PTX)
                 | textual LLVM IR (--emit=llvm-ir)
```

`wukong_driver::compile` orchestrates this and honors `--emit=<stage>` to stop early. Every stage
prints its artifact on stdout except `--emit=obj` and `--emit=exe`, which write a file (`-o <path>`).
Two combinations are usage errors rather than silently discarded work: `--run` with an explicit
`--emit=<stage>` (the driver honours exactly one), and `-o` with `--run` or with any stage other than
`obj`/`exe`. Diagnostics from every stage flow through a single `emit_diag` honoring
`--error-format`.

`--emit=tokens` and `--emit=ast` return *before* import loading, so they show the root file only;
every later stage — and `--run` — sees the merged program. A verifier failure is an internal compiler
error (a bug in a pass), reported as prose and mapped to a non-zero exit, never a user diagnostic;
`--emit=mir`/`--emit=mir-high` verify inside `emit_mir` and print the dump regardless of the result —
the ICE lines go to stderr first and a failure only changes the exit code — so a broken program still
prints the MIR that shows why. `wukong_opt`'s per-pass verify-each is `#[cfg(debug_assertions)]`, so
in a release compiler this gate is the only verification that runs.

## Diagnostics

Every diagnostic has a stable code (`E01xx` lexer, `E02xx` parser, `E03xx` name res, `E04xx` types,
`E05xx` shapes, `C0xxx` codegen); `E00xx` is reserved for generic/internal-surface errors and
currently holds none. Codes are permanently stable: never renumber one, and never recycle a retired
number — `E0001` ("malformed type syntax") is RETIRED and unallocated, so `--explain E0001` resolves
to nothing and malformed types report `E0203`/`E0204`. `E0404` is likewise an unfilled gap; leave it
alone.

The catalog in `wukong_diag::catalog` gives each code a title and an extended explanation surfaced by
`wukongc --explain <CODE>`. It holds 29 entries at HEAD and is gated in both directions by
`wukong_diag/tests/catalog_coverage.rs`: every code a crate emits must have an entry, and every entry
must have an emitter (its `RESERVED` exception list is empty, which is the healthy state). A third
gate, `every_catalogued_code_is_exercised` in `crates/wukongc/tests/fail.rs`, requires every
catalogued code to be reached by a `tests/fail/` fixture or by an inline `LEXER_PARSER_PROBES` probe.
So adding a diagnostic is always three edits: the catalogue entry, an emit site spelling the code as a
bare `"Ennnn"` literal (the scanner skips comment lines, so prose may cite codes freely, but a
`format!`-built code is invisible to it), and a fixture or probe.

The terminal renderer is rustc-style (carets, gutters, `--> file:line:col`). Two rendering rules are
contractual: a caret run is as wide as the span's *rendered* width, never the column difference, and
both the echoed source line and the caret padding expand tabs at a fixed `TAB_WIDTH = 4` — so carets
land under the construct in tab-indented source. Columns past the end of a line count one each, and a
multi-line span underlines only its first line. Reported line/col are unaffected, so the JSON output
does not change; if you touch `TAB_WIDTH`, `rendered_width` or `expand_tabs`, touch all three.

`--error-format=json` emits one JSON object per line (JSON Lines) for tooling: `severity`, an optional
`code`, `message`, a `spans` array of `{file, line, col, primary, label}` (dummy spans omitted; `col`
is the span's start — there is no end position), and a `notes` array of `{kind, message}`. The writer
is hand-rolled (no serde): `"`, `\`, `\n`, `\r`, `\t` get short escapes and every other character
below `0x20` becomes `\u00xx`, so the line always stays parseable. `crates/wukongc/tests/fail.rs`
greps for the literal substring `"code":"Ennnn"`, so key order and the absence of whitespace around
`:` are part of the contract.

## MIR

A `Function` owns a value arena (`ValueId -> MirType`), a list of `BasicBlock`s, and an entry block.
Each block has typed parameters, a straight-line list of `Inst { result: Option<ValueId>, op: Op }`,
and exactly one `Terminator` (`Ret`, `Br`, `CondBr`, `Unreachable`). `Op` has 20 variants: constants
(`ConstInt`/`ConstFloat`), `Bin`/`Cmp`/`Cast`, the unary `Neg`/`Not`, `Select`, memory
(`Alloca`/`Load`/`Store`/`Gep`), `Call`, the float/SIMD primitives (`Splat`/`Fma`/`Sqrt`/`Round`), the
address-of ops (`FuncAddr`, `GlobalAddr`), and `VecKernelCall`. Every op produces exactly one result
except `Store` (never), `Call` (void callees such as `print`) and an elementwise `VecKernelCall`. A
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

The **verifier** (`wukong_mir::verify`) enforces true SSA plus per-op typing: every value is defined
exactly once and inside the value arena; every use is *dominated* by its definition (a real dominator
tree — DFS postorder, a Cooper–Harvey–Kennedy idom fixpoint, then an explicit-stack preorder walk);
the entry block exists, its parameters equal `Function::params` exactly, and it has no predecessors;
`block.id` equals the block's index; operand/result types line up per op (float-ness classified by
*lane* type, so a `<4 x f32>` `fadd` checks against `f32`); a `VecKernelCall`'s `kernel` index is in
range for the owning function's `vec_kernels` and its result matches the recipe kind (a reduction
yields an `f32`, an elementwise one yields nothing); and terminators are type-correct, including
branch-argument arity/types and the function return type. Blocks unreachable from the entry keep their
type checks with the dominance rule switched off. Given a whole `Program`, `verify_program`
additionally cross-checks every call against the callee's declared signature — external symbols
(`print`, the `wukong_*` kernels) are exempt.

It runs before every backend entry — `--run` and `--emit=llvm-ir|obj|exe` all go through
`wukong_driver::verify_or_ice` — and inside `--emit=mir`/`--emit=mir-high`, where the dump is printed
regardless of the verify result, so a broken program still prints the MIR that shows why (a printed
ICE always exits non-zero). In debug builds `wukong_opt` additionally asserts well-formedness after
every pass that ran; that check is `#[cfg(debug_assertions)]`, not a CLI flag, so it is compiled out
of release. Note the pipeline calls `verify_function`, so `verify_program`'s call-vs-callee
cross-check is exercised only by `wukong_mir`'s own tests.

`wukong_mir::print` renders MIR deterministically — the text behind `--emit=mir` and
`--emit=mir-high`, and stable enough for snapshot tests. It is also the grep surface for checking what
lowering did: a SIMD type prints LLVM-style as `<4 x f32>` (so `--emit=mir -O2 | grep '<4 x f32>'`
tells you the 128-bit vectorizer fired), a synthesized 256-bit kernel prints as
`veckernel #0(ptrs=…, scalars=…, n=…)`, and a dispatched runtime kernel shows up as a plain
`call wukong_…`. Types are printed on `const.<ty>`, `load <ty> p`, `alloca <ty>`, `<cast> v to <ty>`
and every function/block parameter list; plain arithmetic prints untyped (`v5 = fadd v1, v2`), so read
the operand's defining line for its type.

## Optimizer

`wukong_opt::optimize` first runs a whole-program **inliner** (`-O2`), then `PassManager` runs a
list of function-level `Pass`es to a per-function fixpoint (capped at 100 sweeps). The fixpoint uses
per-pass **clean-tracking**: a pass is a deterministic function of the MIR, so one that reported "no
change" cannot fire again until some *other* pass mutates the function; clean passes are skipped and
any mutation re-dirties all of them. That changes only how often a pass runs, never the sequence of
mutations, so the resulting MIR is identical to the naive fixpoint — which is exactly why a new pass
must stay deterministic and must report `changed` honestly.

Two shared analyses back them: `cfg` (successors/predecessors, reverse postorder, reachability,
unreachable-block pruning) and `dom` (Cooper–Harvey–Kennedy immediate dominators, dominance frontiers,
and the dominator tree). Both are reached through `CfgAnalyses` (`src/cache.rs`), a single per-function
cache the pass manager threads through every pass and every fixpoint iteration, each analysis computed
lazily on first use. It depends only on the block set and terminator successor edges, so **a pass that
changes either must call `cache.invalidate()`** — a stale dominator tree silently produces wrong
mem2reg/CSE/LICM results the verifier does not catch. Passes that only touch instructions, block
parameters or edge arguments leave the cache alone, and the prune-first passes (`mem2reg`, `cse`,
`licm`) use the free `all_reachable` test to skip their reachability DFS in the steady state.

| Pass            | Level | What it does |
|-----------------|-------|--------------|
| `inline_program`| -O2   | inline small, non-recursive **leaf** functions (whole-program), then drop callees left uncalled; runs before the function pipeline so the spliced code optimizes in context |
| `Mem2Reg`       | -O1   | promote scalar int/float `alloca` slots to block-parameter SSA via dominance-frontier phi placement and a dominator-tree rename |
| `Simplify`    | -O1   | constant folding + algebraic identities (`x+0`, `x*1`, `x*0`, `x^x`, `x&x`, `x\|x`, `x%1`) and integer self-comparison folding |
| `SimplifyCfg` | -O1   | constant-branch folding, straight-line block merging, and unreachable-block pruning (with renumbering) |
| `SimplifyPhis`| -O1   | drop dead and trivial block parameters that mem2reg introduced |
| `Dce`         | -O1   | remove pure instructions whose results are unused, and dead allocas |
| `Cse`         | -O2   | dominator-tree value numbering (a pure computation is reused by every block it dominates, via a scoped table) plus **intra-block** load forwarding keyed on `(address value, accessed type)`, with `alias` deciding which entries a store or call invalidates |
| `Dse`         | -O2   | dead-store elimination (a store fully overwritten by a later one with no intervening read that `alias` says could observe it) |
| `Licm`        | -O2   | hoist loop-invariant, side-effect-free, non-trapping ops to the loop preheader — including a **load**, when `alias` proves nothing in the body writes it *and* the access is dereferenceable |

`-O3` currently runs the same pass pipeline as `-O2`: there are no `-O3`-exclusive passes yet
(`PassManager::standard` adds passes at `-O1` and `-O2` only).

One splice detail generalizes to every MIR pass: `Op::VecKernelCall.kernel` is a **function-local**
index, not a `ValueId`, so value renaming neither sees nor fixes it, and `inline_call_site` must carry
the callee's recipes into the caller and rebase every copied index (see the vectorizer section).

`Mem2Reg` is the keystone: the front-end's memory traffic hides constants, common subexpressions,
and induction variables, so promoting slots to SSA is what lets the rest of the pipeline fire. It
leaves arrays, address-taken, and pointer slots in memory; a read before any write becomes a zero
constant, matching the interpreter's zero-initialized memory. `Cse`/`Dse` still handle the residual
memory (load forwarding and its dual) for the slots that stay in memory. `Licm` uses the dominator
analysis to find natural loops and only hoists into loops that already have a preheader, so it is
always legal; stores, calls, vector-kernel calls, allocas, and integer division/remainder (which can
trap on a zero divisor) are never moved. Float predicates are never folded on self-comparison
(NaN != NaN).

### Alias analysis (`wukong_opt::alias`)

`mir_build` erases every pointer-ish source type — `Ty::Ptr`, `Ty::Ref`, `Ty::Tensor` and `Ty::Slice`
all become a bare `MirType::Ptr` — so no later pass can tell one buffer from another by *type*. What
survives is **provenance**, and `AliasInfo::analyze` recovers it: every pointer value is classified as
the `alloca` it came from, the *parameter* it came from, the `.rodata` blob it came from, or
`Unknown`; a `gep` inherits its base and tracks the constant byte offset when it has one. `Cse`,
`Dse`, `Licm` and the Cranelift backend all query it.

`may_alias(a, b)` answers `false` only for facts the language actually guarantees: two distinct
`alloca`s; an `alloca` of this function versus a *parameter* of it (the slot is created after entry,
so the caller cannot have handed us its address — not even under recursion, where the incoming
pointer belongs to another frame's slot); an `alloca` versus a `.rodata` blob, and two distinct
blobs; a **non-escaping** `alloca` versus anything untracked, including whatever a `call` may write;
and disjoint constant byte intervals off one base (`may_alias_sized`). "Non-escaping" is checked
literally — the address may appear only as the pointer operand of a `gep`/`load`/`store`, and any
other use publishes it.

**Two distinct pointer parameters may alias.** Wukong makes no promise otherwise: nothing rejects
`f(a, a)`, and there is no `restrict`/`&mut` annotation to carry a promise, so the analysis returns
"may alias" and `tests/run/alias_slice_params.wk` pins the behaviour that depends on it. This is the
single biggest fact the optimizer is missing, and closing it is a **language** decision, not an
analysis one — see the roadmap.

`is_dereferenceable(ptr, bytes)` is a separate question that only a *speculating* transform asks: may
this access run on a path the program might not have taken? It is true only for a known in-bounds
offset into an `alloca` of this function, which is live for the whole function and default-initialized
by both backends. That is what lets `Licm` hoist a loop-invariant load out of a possibly-zero-trip
loop for a local `[]T`/struct/tuple, and what stops it doing the same through a parameter.

Two more folds are refused for the same `-O0` ≡ `-O{1,2,3}` reason. A bf16/f16 constant is never
folded — arithmetic or comparison — because the folder rounds to the f32 grid while the backends round
to the narrow grid at the store, so two distinct f32 literals sharing one bf16 grid point would fold to
"not equal" and compare equal at runtime. And the self-operand folds (`x - x`, `x ^ x`, `x & 0`,
`x <cmp> x`) decline when the result type is a vector: they would materialize a scalar constant into an
`<N x iW>` result, which the verifier rejects.

The opt pipeline is guarded by a **differential test**: every end-to-end program is run at -O0 and at
-O1/-O2/-O3 and must produce identical stdout and exit code. The `wukong_bench` crate reports, per
program, the IR-op reduction and the -O0-vs--O3 interpreter speedup, and exits non-zero if any
program that lowers cleanly disagrees across optimization levels — a soundness gate over every
benchmark kernel. Across the run suite and kernels, -O3 removes **~42% of IR ops (~48–54% on the
heavy transformer/GEMM kernels)** and runs ~1.5–2.5x faster than -O0 under the interpreter.

## Interpreter

`wukong_interp` is a from-scratch CFG walker over MIR that adds no external crate of its own — it
depends only on `wukong_span`, `wukong_mir`, `wukong_backend` and `wukong_runtime`, the last so a
recognized-kernel call executes the *identical* microkernel the native backend links. `Value` is
`Int(i128) | Float(f64) | Ptr(usize) | VecRef(u32) | Unit`.

Memory is a single flat `Vec<Value>` with **one typed slot per scalar leaf** (not bytes): an `alloca`
— or a heap `alloc_<T>(n)` — pushes a typed zero per leaf slot and yields the first slot's index as its
`Ptr`, so a read before any write observes exactly the zero native's zeroed bytes decode to. `Gep`
strides by `slot_count(elem)`, which recurses through arrays of aggregates so element `i` lands where
native's `size_of(elem)` scaling puts it. The arena is run-scoped and never reclaimed, so a `.rodata`
address or a heap pointer stays valid after its defining frame is gone and `free` is a no-op. Vectors
never enter memory: they live in the `vecs` side arena (`VecRef` indexes it, keeping `Value` `Copy`),
and SIMD is executed lane-wise there.

A 100M-step guard bounds runaway loops *within one frame*, and because that counter is local to a
single `exec` invocation, a separate call-depth counter bounds recursion: `MAX_CALL_DEPTH` is 300_000
in release (the measured ceiling) and 40_000 in debug, and exceeding it returns the diagnostic
"interpreter call-depth limit exceeded (likely unbounded recursion)". Every public entry point runs the
walk on a scoped 512 MiB-stack worker thread, since a host stack overflow is uncatchable and would
abort the oracle; heap exhaustion likewise goes through `try_reserve` and returns an error instead of
reaching Rust's abort-on-allocation-failure handler. The rule is that the interpreter reports resource
exhaustion as a diagnostic (stderr, exit 1), never as a process abort.

`run_with_output` returns `(exit_code, stdout)`; intrinsics like `print` format into the captured
stdout buffer. `f32` ops are computed in `f32` (single rounding), and an `int → f32` cast rounds
straight to `f32` (not via `f64`), so the interpreter matches native's single conversion bit-for-bit.
Integer results are likewise normalized to their declared MIR width on every write (`i1` stays an
unsigned 0/1), so two's-complement wrapping matches the machine; a `Load` is exempt because memory
holds one typed value per slot and the compiler's byte-wise aggregate copies move wider payload slots
through `i8`-typed load/store pairs. Integer SIMD lanes get the same treatment lane-by-lane, so a
vector `add`/`neg`/`not` wraps at the declared lane width exactly as the native vector instruction
does. The interpreter is the sound oracle for differential testing.

## Native backend (Cranelift) and the vectorizer

`wukong_codegen_cranelift` lowers MIR to Cranelift IR — an almost 1:1 map (block-parameter SSA,
signless ints, explicit `alloca`/`load`/`store`/`gep`). It JIT-compiles in-process for
`--backend=native` and emits a host object for `--emit=obj|exe` — for `exe` it prefers a rustc-driven link that pulls
in the `wukong_runtime` kernels (so a recognized-kernel program and string `.rodata` both resolve),
falling back to a `cc`/`$CC` C-runtime link. The rustc path is chosen only when
`libwukong_runtime.rlib` sits next to the running compiler (a cargo target layout); otherwise the
driver reports that path unavailable and uses `cc`. The rustc invocation passes `-C panic=abort`,
because the workspace's release profile sets `panic = "abort"` and rustc would otherwise reject the
shim's default `unwind` strategy against that rlib. The generated link inputs (the Rust shim, or the C
runtime for the fallback) are written into a pid-keyed scratch directory that is deleted when the link
finishes — never into the process's working directory, unless that directory cannot be created.

The only semantic bridges to stay identical to the interpreter: divide-by-zero yields 0, float→int
casts saturate, and `i1` results are masked to their low bit. Two further contracts belong to the same
seam. First, every generated load/store carries `notrap` **only** — deliberately not Cranelift's
`trusted()` (`notrap | aligned`). The backend promises the address is in bounds by construction (bounds
checks happen upstream) but promises **nothing about alignment**: the 128-bit CLIF vectorizer emits
`load.f32x4`/`store.f32x4` at addresses it knows are not 16-byte aligned (a loop starting at `i = 1`
over an `[f32; N]` local loads at `alloca_base + 4`). With `aligned` set, an x86-64 host without AVX
folds the misaligned operand into a legacy SSE instruction and #GPs on the first vectorized iteration,
so the flag must never be re-added; dropping it is strictly weaker and can at worst cost a folded load.
Second, a float-returning entry point is invoked through the ABI its own width implies: `f16`/`bf16`/
`f32` all return a single-precision XMM0 (`cl_type` computes the narrow floats in f32 registers), so
they must not share the `f64` arm — reading those bits as an `f64` turns a live float into a denormal
and the exit code diverges from the oracle.

SIMD **auto-vectorization** happens in `wukong_mir_build` while lowering a `for` loop: a
straight-line elementwise body over unit-stride array accesses (plus loop-invariant splats, and
`if`/`else` value-expressions via if-conversion to a vector compare + blend) lowers to vector
`load`/`store`/`bin`/`cmp`/`select` and `Op::Splat`. The loop is strip-mined into an unrolled main
loop (`VEC_UNROLL` independent 128-bit groups per iteration — recovering AVX-class throughput from
SSE via dual-issue), a single-vector loop, and a scalar remainder, all sharing one index slot. It
bails to scalar on any loop-carried dependence, non-unit stride, call to a **user** function, or mixed
lane type, so lane `k` always computes exactly what scalar iteration `base+k` would. Calls to
*unshadowed math intrinsics* are the exception and do vectorize: `sqrt`/`rsqrt`/`abs` and the rounding
family (`round`/`floor`/`ceil`/`trunc`) as one lane op, `fmax`/`fmin` as a lane compare + blend, and the
transcendentals and activations (`exp`, `log`, `tanh`, `sigmoid`, `silu`, `gelu`, …) as their inline f32
polynomial expanded per lane — only for an f32 lane, and only while the name does not resolve to a user
`fn` (a user `fn gelu` wins and the loop stays scalar). `@parallel` per-thread chunks go through the
same vectorizer, so they run SIMD × cores.

Cranelift's CLIF vector ISA still caps at 128-bit — a 256-bit `f32x8` SSA value is rejected at
`define_function` (the `cranelift_still_rejects_f32x8` / `p4_probe_vec256_ops` tripwires keep that
documented). So for an eligible f32 elementwise body the vectorizer captures the loop as a flat,
backend-agnostic `VecKernel` recipe and, when register pressure fits, emits one `Op::VecKernelCall`:
the Cranelift backend assembles that recipe to **true 256-bit AVX2 machine code** via `iced-x86`
(`wukong_codegen_cranelift::avx2`, VEX-encoded), while the interpreter marshals the *same* recipe
lane-wise — so the two stay bit-identical (elementwise lanes carry no reassociation, so the gate
holds element-for-element). `Op::VecKernelCall.kernel` is an index into the *owning* function's
`vec_kernels` recipe table, not a `ValueId`, so a MIR pass that moves instructions between functions
must copy the callee's recipes into the caller and shift every copied index (`wukong_opt::inline` does
exactly that; without it a spliced call names one of the caller's own recipes). The verifier
range-checks the index and requires the call's result to match the recipe kind: a reduction recipe
yields an `f32`, an elementwise one yields nothing.

That assembler path is host-gated: `avx2::host_supports_kernels()` requires `target_arch = "x86_64"`
**and** `target_os = "windows"` **and** runtime AVX2 + FMA3, because the emitted bytes read their three
arguments out of the Win64 registers rcx/rdx/r8 literally and there is no dispatch inside a kernel.
`assemble_kernel` refuses on that gate before it encodes anything, so on any other host a program whose
loop produced a recipe fails to compile with an `avx2:` error rather than emitting bytes that would
`#UD` or dereference garbage. The kernels are assembled while the module is populated — the same code
path for the JIT and for `--emit=obj` — so an emitted object is host-ISA- and host-ABI-specific.
`WUKONG_P4_NO_256` forces the portable 128-bit path — a same-run A/B knob and a kill-switch. The
float-reduction path has an analogous 256-bit kernel, gated to large trips
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
(`nn.Linear`) spelling where B is indexed `[j*K+k]`. One precondition is shared by the entire
recognizer cascade, not just matmul: the loop header must be a **half-open, unit-step** range. Every
recognizer reads its trip count as `end - start`, so the shared `range_bounds` helper declines
`for i in 0..=n` and `for i in 0..n step k`, and such a nest lowers as the ordinary (correct,
unaccelerated) scalar loops.

The recognizer (`recognize_matmul`) verifies the strides describe contiguous row-major operands, then
lowers the *whole nest* to a single call: `wukong_sgemm` (`C = A·B`), `wukong_sgemm_nt` (`C = A·Bᵀ`, the
`nn.Linear` spelling) or `wukong_sgemm_tn` (`C = Aᵀ·B`, the weight-gradient shape) — each with a
`_parallel` twin picked by the `@parallel` attribute. A peeled scalar multiplier routes to
`wukong_sgemm_nt_alpha`, a fused `act(x·Wᵀ + bias)` epilogue to `wukong_sgemm_nt_epi`, and bf16/f16/int8
operands to the corresponding `wukong_sgemm_{bf16,f16}_nt*` / `wukong_i8gemm_nt` members of the same
family. `C = Aᵀ·Bᵀ` has no kernel, so it declines.

Each buffer operand is resolved through the shared `kernel_base_ptr`: a fixed `[T; N]` array's slot *is*
its storage and is passed as-is, while a `Tensor[..]`/pointer operand and a `[]T` slice both keep their
base in the slot and are loaded first (a slice's data pointer is the first word of its fat pointer).
Slice-ness is keyed off the **sema** type, recorded by `bind_slice` at every binding site — never off
the MIR slot shape, which cannot distinguish `[]T` from a tuple, a 16-byte struct or a user's
`[u8; 16]`. That is what makes `[]T` slices legal operands across the whole recognized-kernel family
(the only way to hold a runtime-sized weight blob).

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

Recognition is always *optional*. When the emitter cannot represent a shape the matcher accepted — an
unsupported transpose combination, an unbound operand, a peeled α or a bias the NT-only epilogue kernel
has no room for — `emit_sgemm` returns `false` and the nest lowers as the ordinary scalar loops. The
whole-function form behaves identically: `lower_matmul_fn` returns `Option<Function>` and falls through
to the normal `lower_fn` path on a decline. So a declined matmul is unaccelerated, never silently
dropped or miscompiled.

Crucially this stays inside the differential oracle: the interpreter, on a `wukong_sgemm*` call,
**marshals its abstract `Value` memory into real f32 buffers and calls the identical kernel**, then
marshals the result back — so native and interpreter agree bit-for-bit despite the reassociated
accumulation (the parallel kernel is bit-identical to the serial one by construction).

## GPU backend

`wukong_codegen_gpu` (behind `--features gpu`) is a third execution path: being a compiler, it
**emits PTX text** and **driver-JIT-loads it via `cudarc`**, so no `nvcc`/CUDA toolkit is needed — only
the NVIDIA driver. Module loading goes through a persistent **cubin cache**: `Gpu::load_module_cached`
loads a previously compiled SASS image for this PTX hash + driver version when one exists, otherwise
drives the driver's own in-process `cuLink*` JIT once, persists the cubin and loads that, and degrades
to a plain PTX `cuModuleLoadData` on any failure (no linker, unwritable cache directory, a stale or
driver-incompatible cubin). Every route is inside the driver, so the no-toolkit property holds either
way, and both GPU backends share it.

It offers two modes. `--backend=gpu` runs the program on an **offloading interpreter** (the CPU
tree-walks every op as the oracle does, but the recognized calls the accelerator seam covers execute
on the device). That seam is `wukong_interp::Accelerator`, implemented by
`wukong_driver::gpu_accel::GpuAccel`, and it has exactly five hooks — `sgemm_nt`, `sgemm_nt_epi`,
`vmath`, `norm`, `sreduce`. Each declines what its PTX table does not cover: the GEMM offload fires
only for the `C = A·Bᵀ` overwrite form (so a plain `C = A·B` or `C = Aᵀ·B` kernel call stays on the
CPU), and `vmath_supported` / `reduce_supported` / `norm_supported` filter op codes (log-softmax and
L2-norm have no PTX entry). A decline runs the identical `wukong_runtime` kernel on the CPU; a device
*failure* is surfaced as an error, never silently downgraded — that distinction is the point of the
seam.

`--backend=gpu-native` (the `GpuLower` backend) instead lowers the MIR itself to PTX, so non-recognized
code runs GPU-side too, with the recognized `wukong_*` symbols (serial and `_parallel` spellings alike)
mapped by `rt_helper` onto `mrt_*` PTX helper kernels. Coverage is honest rather than total: a construct
the lowering does not handle returns an error whose message begins with `UNSUPPORTED:`
(`lower::UNSUPPORTED`), which the coverage sweep counts as not-yet-covered and distinguishes from a
genuine JIT/launch failure — and, unlike the offload path, this backend never falls back to the CPU, so
the run reports that error. An eligible program is fused into a single-block cooperative *megakernel*
(one launch, no host round-trips): `lower::jit_run` tries it first on every program and falls through to
the single-thread lowering — which stays the universal correctness reference — whenever
`megakernel::try_run` returns `Ok(None)`. Eligibility is proved by `fusion::analyze`: control flow must
be data-independent so the SPMD threads cannot diverge and deadlock a `bar.sync`. The kernel launches
one block of `MEGA_BLOCK` (256) threads, with the alloca frame in one shared `.global` buffer and side
effects `tid == 0`-guarded. `WUKONG_GPU_NO_MEGA=1` forces the single-thread path for A/B timing and
debugging.

The CPU↔GPU boundary is **tolerance-gated** (`c·√K·ε`) rather than bit-exact — the GPU analogue of the
CPU differential oracle — and the path stays optimization-invariant (`-O0` ≡ `-O3`). Every module except
`paged_kv` and `paged_attention` is behind the feature, so a plain `cargo test` compiles an almost-empty
crate and the toolchain-free core is unaffected. Those two are deliberately un-gated: their host-only
contents (the KV block allocator and geometry, the paged-decode PTX generators, the int8 quantizer, the
f64 reference) need no device, so their shape and PTX-ASCII gates run in the default build; only the
launchers and device caches inside them are feature-gated. The flip side is that `cargo test` does not
type-check the GPU backend at all — `cargo check --features gpu --all-targets` is the check that does,
and it is a required half of the repo gate.

Beyond the two CLI backends the crate holds device infrastructure and whole stacks that are **library
surface**, exercised by its own `--features gpu` tests and benches rather than reachable from `wukongc`:
`pool.rs` (a bump-arena device allocator, and the prerequisite for graph capture — a captured region may
contain no synchronizing allocation), `graph.rs` (CUDA-graph capture/replay), `cubin.rs` (the cubin
cache above), `autotune.rs` (per-(op, shape, dtype) selection over the int8/W4A16 GEMM variants with a
validated on-disk config cache), `paged_kv.rs` + `paged_attention.rs` + `serving.rs` (paged KV-cache,
warp-cooperative paged decode attention, batched decode and scheduler), `train_resident.rs` (a
GPU-resident training step), and the `ptx_*` kernel-generator families. The compiler itself reaches the
device only through `gpu_accel`'s five hooks and `lower.rs`.

## Testing strategy

- **Unit tests** per crate (lexer, parser, sema, MIR verifier, opt passes, interpreter, vectorizer).
- **End-to-end** (`tests/run/*.wk`, 333 fixtures): the real `wukongc` binary compiles and runs each
  program; stdout/exit are checked against the `// EXPECT-EXIT:` / `// EXPECT-OUT:` directives embedded
  in the file, and a `// RUN:` directive replaces the default `--run` argument list (a fixture that must
  pin the *native* side carries `// RUN: --run --backend=native`). Placement rule: everything here must
  lower end to end — a separate gate requires every run-suite program to succeed at `--emit=mir-high`,
  `--emit=mir` and `--emit=llvm-ir`, at -O0 and -O2 — so a program whose purpose is to be **rejected**
  belongs in `tests/fail` instead.
- **Compile-fail** (`tests/fail/*.wk`, 110 fixtures, driven by `crates/wukongc/tests/fail.rs`): each
  program must be *rejected*, and with the exact stable code its `// EXPECT-CODE:` directive names; the
  harness drives the real binary with `--error-format=json --emit=mir`. Eleven lexer/parser codes that
  no committed `.wk` file can hold cleanly (unterminated string, unterminated block comment, …) are
  probed from inline sources in the same file. Three coverage invariants are gated: every code any crate
  emits must have a catalogue entry, every catalogued entry must have an emitter
  (`crates/wukong_diag/tests/catalog_coverage.rs`), and every catalogued code must be reached by a
  fixture or a probe — the last with a floor on catalogue size so it cannot pass vacuously.
- **Differential**: `-O0`-vs-`-O{1,2,3}` invariance, and **interpreter-vs-native (Cranelift)** on stdout
  and exit code — the primary gate is `native_matches_interpreter` in `crates/wukongc/tests/run.rs`,
  which runs every `tests/run` fixture through the CLI on both backends (at -O0 and -O2) and collects
  all mismatches so one failure reports the full set; `native_optimization_is_observationally_invariant`
  adds -O0-vs--O{1,2,3} on the native path, and `wukong_bench`'s equivalence gate plus the cranelift
  unit tests cover the same invariant from the library side. Vectorized kernels are additionally checked
  against independent scalar references across remainder-exercising sizes. Two randomized fuzzers run in
  the default `cargo test` too: a **full-buffer** differential (`wukong_codegen_cranelift::fuzz`) that
  drives the typed kernel-entry ABI on both backends over identical random buffers and requires the
  *entire* output buffer to be bit-identical (f32 and int8), and a **grammar** fuzzer (`fuzz_grammar`)
  that generates random well-typed programs through the real pipeline and asserts interpreter == native
  and -O0 == -O2 == -O3 on (exit, stdout). Both use a seeded PRNG, so a failure reproduces exactly from
  the seed it prints; `WUKONG_FUZZ_PROGRAMS=N` deepens the grammar run and an unusable value falls back
  to the default rather than silently disabling the gate.
- **Determinism** (`crates/wukongc/tests/determinism.rs`): the compiler must be a pure function of its
  input. Three *fresh processes* per compile (each with its own `HashMap` hash seed) must agree
  byte-for-byte on `--emit=mir -O2`, on `--emit=obj -O2` (and the `WUKONG_PAR_CODEGEN=0` serial object
  path must equal the parallel one), and on the `--error-format=json` diagnostic stream over the
  compile-fail corpus. Stated limit: the seed mechanism only bites std `HashMap`/`HashSet`; the
  optimizer's FxHash maps have a fixed seed, so these gates say nothing about insertion-order dependence
  there.
- **AOT executables** (`crates/wukongc/tests/exe.rs`): curated fixtures are linked with `--emit=exe` and
  their stdout/exit compared against `--run`, extending the differential gate to the linked exe (string
  `.rodata` and the linked `wukong_runtime` kernels). The gate probes once which link path exists on the
  box and may skip only when neither does; a failure of the compiler's own link is reported with its
  captured stderr.
- **Examples** (`crates/wukongc/tests/examples.rs`): every `examples/*.wk` program is an
  interpreter-vs-native differential fixture *and* carries a declared execution class (`Runs`,
  `NotLowered`, `NoEntry`, `DataGuarded`), because a program that fails to compile trivially satisfies
  agreement on both backends. `Runs` is the default, so a new example must actually execute or be
  declared otherwise.
- **Emit stages** (`crates/wukongc/tests/emit.rs`): `--emit=tokens|ast` over the examples and the run
  suite, and `--emit=mir-high|mir|llvm-ir` over the run suite at -O0 and -O2 — which is also what keeps
  the MIR verifier and the pretty-printers honest.

All of the above runs with `cargo test` and needs no C or LLVM toolchain (Cranelift is a pure-Rust
crate). Two suites are conditional and say so out loud: the `--emit=exe` gate needs either `rustc` with
`libwukong_runtime.rlib` beside the compiler binary or a working `cc`/`$CC`, and prints why it skipped
otherwise; and the GPU suites need `--features gpu` plus a reachable device — plain `cargo test` does not
even type-check that backend, so `cargo check --features gpu --all-targets` is a required second half of
the gate. `WUKONG_GPU_REQUIRED=1` and `WUKONG_PEER_REQUIRED=1` turn a device- or peer-absent skip into a
failure on machines that are supposed to have them.
