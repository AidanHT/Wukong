# Wukong Compiler Internals

This document describes how `wukongc` is built: the crate layering, the pipeline, the MIR, and the
key design bets. It is aimed at contributors.

The compiler exists so programmers write `.wk` kernels and pass them through a real front end,
semantic analyzer (`wukong_sema` — including compile-time shape checking), and MIR pipeline. Execution
backends (interpreter, Cranelift native codegen, optional GPU recognizer offload and MIR→PTX) are
interchangeable ways to run the same program; they are not the product on their own. Mission and lowering thesis live in [WUKONG_MASTER_PLAN.md](WUKONG_MASTER_PLAN.md) and [LANGUAGE_DIRECTION.md](LANGUAGE_DIRECTION.md).

## Design bets

1. **Interpreter-first, native code without LLVM.** The entire front-end, optimizer, and a
   from-scratch MIR interpreter build and test with plain `cargo test` on every platform in README *Platform support* — no LLVM
   required. Native code is produced by a **Cranelift** backend (`wukong_codegen_cranelift`): JIT
   for `--run --backend=native` and object/exe for `--emit`, again with no LLVM toolchain. (A textual
   LLVM IR backend also exists for `--emit=llvm-ir`.) The interpreter is the always-available
   execution path and the differential-testing oracle; the native backend is validated bit-for-bit
   against it (see Testing). See `BENCHMARKS.md` for the cross-language standing. One platform
   caveat: the 256-bit raw-AVX2 kernel path (`wukong_codegen_cranelift::avx2`) encodes the Win64
   argument registers literally and has no in-kernel dispatch, so it is available only on an x86-64
   Windows host with runtime AVX2+FMA3. That test is one function —
   `wukong_mir::host_supports_vec_kernels` — and **the vectorizer consults it too**, so off-gate the
   recipe is simply never built and the loop keeps the 128-bit CLIF form, exactly as
   `WUKONG_P4_NO_256=1` forces. That changes speed, never a result: a vectorized reduction computes
   the same canonical tree in both forms (see the vectorizer section). Everything else — front end,
   optimizer, interpreter, and the ordinary Cranelift lowering — is host-agnostic.
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
   Because a call-site parameter is *substituted* before it is unified, the diagnostic path carries
   the pre-substitution declared type alongside it (`unify_decl`) purely so the message can name the
   dimension the signature wrote — ``dimension mismatch on `K` (axis 0): expected 512, found 513``.

## Crate layering

The workspace is a virtual cargo workspace (`members = ["crates/*"]`, no root package); every crate
is prefixed `wukong_` except the CLI binary `wukongc`. Dependencies flow strictly downward (no
cycles):

```
wukong_span      Span, SourceMap (line index), string Interner / Symbol
wukong_runtime   the AVX2/FMA microkernels the recognizers dispatch to (GEMM/GEMV, vmath,
                  reductions, norms, int8, bf16/f16) + the rayon-backed `wukong_parallel_for` that
                  `@parallel` lowers to + `math64`, the pure-Rust double-precision transcendental
                  library (fdlibm/musl algorithms, ≤1 ULP core, Payne–Hanek trig reduction) that
                  every `f64` `exp`/`log`/`sin`/`pow`/… lowers to as one `wukong_rt_math_f64` /
                  `wukong_rt_math2_f64` call — a leaf: no workspace deps, and both the interpreter and
                  the native backend call into it. It also backs the language's allocators:
                  `wukong_rt_alloc` (64-byte-aligned zeroed heap) and `scratch` (the per-thread
                  LIFO arena behind `scratch_<T>`/`scratch_mark`/`scratch_release`; the
                  interpreter mirrors its mark count and region reuse). The bump `Arena` and the
                  sequential `parallel_for` are reference surface with no caller anywhere in the
                  workspace (the kernels use thread-local `Vec` scratch)
wukong_diag      Diagnostic, Severity, error-code catalog, JSON + terminal renderers
wukong_lexer     &str -> tokens (+ @attributes, error recovery); `tokenize_with_comments` adds the
                  comment side channel the formatter needs (plain `tokenize` is unchanged)
wukong_ast       AST nodes, NodeId, pretty-printer; owns `BinOp::fold_const_len` +
                  `MAX_CONST_ARRAY_LEN` (= u32::MAX) — the const-array-length folder SHARED by
                  sema and mir_build so the two cannot disagree on a length
wukong_parser    recursive-descent + Pratt expressions -> AST
wukong_fmt       the source formatter behind `wukongc fmt`: AST + comments -> canonical text,
                  self-checked to re-parse to the identical AST (see Formatter). Used only by wukongc
wukong_types     Ty, Scalar, Shape/Dim/Layout — the shared semantic type vocabulary
wukong_sema      name resolution + type checking + SHAPE checking
wukong_mir       MIR data, builder, pretty-printer, verifier, `VecKernel` (the 256-bit AVX2
                  recipe). `MirLevel` lives here too but is inert — see Design bet 2.
wukong_mir_build typed AST -> MIR (alloca-per-local lowering; SIMD loop auto-vectorization)
wukong_opt       pass manager + analyses (cfg, dominators) + transforms (inlining, sroa,
                  mem2reg, simplify, jump threading, simplify-cfg, simplify-phis, dce, tre,
                  cse, dse, loop-delete, fuse, licm, bounds-check elimination, unswitch, promote,
                  loop unrolling, switch formation) plus the `alias` provenance analysis
                  every memory transform queries
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
  → loader       (wukong_driver::loader::load_imports)  load `import`ed files, resolve each file's
                 names in its own module scope (pub / as / {x, y}), rename imported items to unique
                 `<module>.<name>` symbols, splice -> one module
  → sema         (wukong_sema::check)                  name res, types, SHAPE check
  → mir_build    (wukong_mir_build::lower_program)     -> MIR (scalar SSA; always MirLevel::Low)
  → opt          (wukong_opt::optimize)                fixpoint passes
  → autodiff     (wukong_autodiff::grad)               --emit=grad / --train / --emit=obj --grad-of;
                                                        forces >= -O1
  → verify       (wukong_mir::verify, via wukong_driver::verify_or_ice)
                 gate on --run and on every AOT exit
  → backend      interpreter (--run) | Cranelift native (--backend=native / --emit=obj|exe)
                 | GPU (--features gpu: --backend=gpu offload, --backend=gpu-native MIR→PTX)
                 | textual LLVM IR (--emit=llvm-ir)
```

Three front-end surfaces lower to plain calls before any recognizer sees the tree. **Closures**:
`wukong_sema::closures` checks each `|x| body` literal where it is written — only as an argument of
a function-typed parameter, `E0411` elsewhere — typing its parameters from that parameter's
function type and recording, per literal, its parameter, return and capture types
(`SemaResult::closures`). A capture is every enclosing local the body reads: `note_local_use` sees
each local read with the index of the scope that binds it, and a binding below a closure's base
scope is a capture of that closure (and of every closure it is nested in); only a scalar, C-style
enum, pointer or reference can be captured (`E0418`), and writing one is `E0304`. The **first**
rewrite of `wukong_mir_build::lower_program` (`closures::lift_module`, before the method
desugaring) lifts each literal, innermost first, to a registered top-level function
`<fn>$closure$<k>` whose leading parameters are the captures, and replaces the literal by its
**value**: the tuple `(L, c0, c1, ..)` of the function and the captured locals (closure conversion,
typed as the closure's function type), or the bare name `L` when it captures nothing. Every pass
below sees the function-typed-parameter machinery it already has (`closures::closure_code` is the
function a value names), and sees each captured read where its copy is taken. That second half is
load-bearing: `canon` inlines and deletes an index-only `let` and sinks an accumulator seed to its
first mention, the fused-window recognizers prove the scalars a kernel hides are not read after the
window (`block_mentions`), and `defer` refuses to rebind a name a pending call reads — all from what
the tree mentions. With a bare `L` in the tree a capture was invisible to all three, the `let` it
read was deleted, moved or hidden, and the copy resolved to an outer binding of the same name
(`tests/run/closure_hidden_reads.wk`). The lifted item inherits the enclosing function's
`@strict_fp`, `@generic` and `@unchecked`, which lowering keys by item: the closure body is that
function's code (`tests/run/closure_strict_fp.wk`). The **environment ABI**: an instance whose function-typed
parameter `f` is bound to a closure with captures declares one hidden MIR parameter per capture at
`f`'s position and binds them as `$env$f$<j>`; `lower_call` passes, at a function-typed argument's
position, the capture values — the value's captures, lowered at the literal's own site, or
forwarded from the caller's `$env$g$<j>` — and a call through `f` calls the lifted function with
`$env$f$..` first. No function pointer, no new MIR construct.
**Methods**: the
parser flattens each `impl` method into a top-level function `Type.method` (receiver desugared to a
`self: Type` parameter) beside an `ItemKind::Impl` marker; `wukong_sema::methods` builds the method
tables, checks trait conformance and bounds, and records every resolved call; `wukong_mir_build`'s
`methods::desugar_module` then rewrites `x.m(a)` to `Type.m(x, a)` ahead of `tindex`/`canon`/alias
analysis. A call through a trait bound resolves per monomorphized instance, and a function-typed
parameter is a pseudo-generic (`$fn$<param>` bound to `Ty::Named(<fn>)`), specialized away with no
runtime value; the functions such calls reach are excluded from the alias-free fact.
**Generic instances**: sema types a generic body once with each type parameter opaque, then
`wukong_sema::instantiate` re-checks every concrete `(function, substitution)` a call requires
(transitively, at most 512 per module) with `lower_type` reading the parameters through the
substitution; it restores every published table afterwards, so it only adds diagnostics, each
labelled "required by this instantiation of `f` with T = …". A dtype-generic tensor is
`Ty::Tensor { elem_var: Some(T), .. }` (its `elem` is then a placeholder): sema binds `T` from a
tensor or array argument's scalar leaf (`dtype.rs`), and mir_build's monomorphizer substitutes the
element into every tensor type (`subst_ty`), so `axpy$f32` lowers like the hand-written `f32`
function; a recognizer that reads sema's raw generic-body types declines on an instance instead. A
fixed-element builtin (kernel or typed I/O) accepts a `Tensor[T, ..]` operand in the generic pass
and the instance re-check reports a wrong binding (`Sema::rechecking_instance`, `E0415`/`E0416`).
`where` dimension predicates and dimension `static_assert`s are collected and checked against each
call's bound dimensions by `wukong_sema::constraints`; `static_assert` lowers to nothing.
**`str` is erased at the sema boundary.** `Ty::Str` exists only inside `wukong_sema`, which is
the single authority that a `str` is never written (no conversion to or from `[]u8`/`*u8`; an
element store, `atomic_add(s[i], v)` or `&mut s[i]` is `E0304`, through the one
`check_projection_write` hook every place-writing statement shares; a `str` comes only from a literal or from slicing a
`str`). `check` publishes every type through `Ty::erase_str` (`strs::erase_published`: the type
table, function signatures, `const` types, struct fields, enum payloads), so `mir_build`, the
optimizer and every backend see the `[]u8` fat pointer they already run; a published `Ty::Str` is
refused by `mir_ty_of` (a debug panic, `C0001` in release), never sized. Because the erasure
precedes monomorphization, a type parameter bound to `str` is `E0420`. The one fact a slice cannot
carry crosses as a side table: `SemaResult::str_print_args`, the `NodeId`s of every print or format
argument of type `str`, which `mir_build`'s `text.rs` prints as text (`wukong_rt_fmt_str` over the
fat pointer's data and length) rather than as a list of bytes; `defer`'s copies carry the
membership (`DeferOverlay`). A literal sema adapted to `str` lowers to a fresh fat pointer over its
`.rodata` bytes; `==` on two `str`s is a block-parameter byte loop (`mir_build::strs`). The range
index `s[a..b]` is parsed as the call `$slice(s, a, b)` (`$slice_from(s, a)` with no upper bound; a
missing lower bound is a zero-width `0` literal at the dots) — a name no identifier can spell —
typed for a `str` or any `[]T` base, and lowered to `{ data + a, b - a }` behind
`bounds::emit_range_bounds_check` (`hi <= len`, then `lo <= hi`, unsigned; `ule`, which
`wukong_opt::bce` deliberately does not key on). A range view is a place in its base, never a
call's fresh result: `@noalias` roots it at its base's storage, so `k(s[0..2], s[1..3])` is
`E0316`, and `mir_build`'s whole-program kernel alias analysis (`alias.rs::origin_of`) projects it
like `s[i]`, so `let va = s[..]; let vc = s[..];` are two names for one buffer and a recognized
kernel over both is guarded (`tests/run/alias_range_views.wk`). It recognizes the callee without an
interner through `SemaResult::range_slice_callees` / `is_range_slice_call`.
**Storage classification is total.** `wukong_mir_build` sizes every value's slot through
`FnLowerer::mir_ty_of` (a `Ty`) and `mir_ty_of_ann` (an annotation), each with an arm per form and
no `_ => I32` fallback: a C-style enum is the one `Ty::Named` whose representation is `i32`; the
registry-blind free `mir_ty` answers `Option` (`None` for any `Named`, and for `Fn`/`Unknown`/`Error`).
A type with no representation is recorded once per function (`FnLowerer::unrepresentable`) and
refused at `finish` as `C0001` naming it. Only a caller that builds MIR from the type records
it: a slot, an op, a coercion target. A caller that only classifies a value asks
`expr_mir_peek`/`mir_ty_peek`, which record nothing. Examples are a `let` asking whether its
initializer is an aggregate, `return` asking what it coerces from, and an assignment asking whether
its target is a buffer. So an untyped initializer under an annotation lowers. The refusal is
reachable from a value sema leaves `Unknown` whose storage or op is sized from that type, such as a
`let x;` with neither annotation nor initializer, or arithmetic on an unannotated binding of a
leniently typed call. It is also reachable from a type parameter no call binds. `expr_ty` recovers
`x.len()`/`x.dim(k)` on a type parameter as the `i64` extent of the array, slice or tensor the
instance binds it to (sema types it `Unknown` in the template, which instances lower from). The enclosing function's own generics
decide what counts as a type parameter (`FnLowerer::type_params`), ahead of the item registry, so a
parameter that shadows a struct, enum, `const` or `fn` is refused rather than sized as that item.
It is also reachable from a generic struct's own type parameter, and from a function value (a
function name inside an array or tuple literal, or alone as a statement; sema rejects one only when
it is bound to a local, field or element type, `E0411`). Only the forms the driver never lets past
sema panic in a debug build instead: `Ty::Error`, and an integer, a function type or `Name<..>`
written in a `let` annotation. `repr_tests.rs` drives every arm.
**Kernel builtins**: `wukong_sema::builtins` shape-checks `matmul`/`softmax`/…/`sdpa` and records each
call's resolved dimensions (`SemaResult::builtin_calls`); `wukong_mir_build::builtins` lowers it to one
runtime kernel behind runtime extent checks (trapping through `assert`) and an output-overlap guard
(scratch + copy-out), independent of `@generic`/recognition. `sdpa` alone has a **static form**: a
check whose outcome is a compile-time fact is not emitted, and an input whose `StorageRoot` (a
fixed-array local's own `alloca`, or a fixed-array/static-tensor parameter bound to its storage —
`noalias.rs`'s `param_is_its_storage`; roots compare by VALUE, since a `match` binding gives one
parameter a second name) is proved apart from `out`'s — different allocas, an alloca and a
parameter, or two different parameters `alias::plan_kernel_alias` answers `Dispatch` for or a
`@noalias` promise separates — gets no overlap test. With all three inputs proved apart the call
is one bare `wukong_attention_f32` in a straight-line block that `wukong_autodiff` differentiates;
the checks a runtime `s`/`d` or extent still needs are `assert` calls in that block, which the tape
treats as effects. A dtype-generic `Tensor[T, ..]`
operand of a kernel builtin or of the ML I/O family (`read_*`/`write_*`, `st_read_*`, `topk_f32`, …)
is accepted in the generic body and re-checked per instance (`E0415` kernel / `E0416` I/O when `T`
is not the builtin's element). That re-check stops at 512 instances while the monomorphizer lowers
them all, so each family's lowering refuses (`C0001`) an operand whose substituted element is still
wrong — `builtin_operands_are_f32` (kernel) and `MlBuiltin::buffer_elem` (I/O). The decode builtins
have no such backstop and do not defer: a `Tensor[T, ..]` operand of one is `E0437` in the generic
body. The **decode builtins** (`sdpa_kv`,
`kv_append`, `gemv_bf16`/`gemv_f16`, `quantize_q{8,4}_0`, `gemv_q{8,4}_0`) are typed in
`wukong_sema::decode_builtins` and lowered in `wukong_mir_build::decode_builtins` to one checked
runtime call each (`wukong_runtime::{attention_kv, gemv_half, quant}`): their extents are values, so
the entry validates them against the operand lengths and returns a status the lowering asserts is
zero; the interpreter runs the same validation and calls the same serial entry.

**Match lowering and the exhaustiveness verdict.** `lower_match` lowers a `match` to an if-else chain
of arm tests (`pattern_cond`); an arm's guard is lowered in a block only the pattern's success edge
reaches, so it runs only after the pattern matched. Sema's pattern-usefulness analysis
(`wukong_sema::usefulness`) decides E0405 and W0020, and it also publishes a positive verdict,
`SemaResult::exhaustive_matches`: the `Match` ids it proved exhaustive over a concrete scrutinee type
whose all-zero value it models (`zero_value_modeled` — mir_build zero-fills `let e: E;`, so an enum
with no discriminant 0 would otherwise hold a value no arm models). It is recorded last-write-wins
across literal-inference re-passes and never from an instance re-check. For a proven match
`lower_match` takes the last unguarded arm untested — it binds the pattern, emits no compare and no
`unreachable` fall-through — so the semantics are the same at every `-O` level and on every backend,
with no MIR op or optimizer fold involved. That makes the analysis and `pattern_cond` a **mirror
pair**: they must agree on which values each pattern matches, or a value they disagree on silently
takes the last arm. `wukong_opt`'s `exhaustive_int_match_agrees_with_rust_on_every_value` pins the
integer side on every `u8`/`i8` value; the two `char`-literal decoders are not pinned. An unproven
match keeps every test and its trapping fall-through. The parser desugars `if let P = e && c` /
`while let P = e && c` into that `match` with `c` as the first arm's guard, and `lower_for_array`
binds a destructuring `for` head at each element's place through the same `bind_field_pattern` a
match arm uses (sema requires the head irrefutable by form, so it is never tested).

`wukong_driver::compile` orchestrates this and honors `--emit=<stage>` to stop early. Every stage
prints its artifact on stdout except `--emit=obj` and `--emit=exe`, which write a file (`-o <path>`).
Two combinations are usage errors rather than silently discarded work: `--run` with an explicit
`--emit=<stage>` (the driver honours exactly one), and `-o` with `--run` or with any stage other than
`obj`/`exe`. Diagnostics from every stage flow through a single `emit_diag` honoring
`--error-format` — including native-backend failures, which `wukong_codegen_cranelift::classify_error`
maps to `C0002` (a layout above 4 GiB), `C0001` (a by-design decline) or an internal-compiler-error
diagnostic. A failure that belongs to one function carries a primary span (`NativeFailure::site`):
one raised while lowering an instruction points at the source construct of the MIR value being
lowered (read from the MIR source-location table — see "MIR"), else at the function's declaration,
which is also where a parameterized entry point, a declined compact boolean-vector legalization and
a function Cranelift's verifier rejects are reported. A module-level failure — an unresolved
`extern "C"` symbol, a target CPU the host cannot run, an object-writer error — names no construct
and has none. The driver keeps the source map alive past lowering only where it can use it
(`--run --backend=native`, and `--emit=obj|exe` without `--run`); the interpreter and GPU runs and
`--train` release it before optimization, and a `--train` artifact's failures stay span-less.
`--target-cpu` fixes the Cranelift ISA level for `--emit=obj|exe` and the native JIT and,
below x86-64-v3, builds the MIR without the raw-AVX2 recipes (`LowerOptions::vec_kernels`). The default
`native` level is the host's flags narrowed by `WUKONG_MAX_ISA` (`wukong_runtime::isa`, read once per
process): `scalar` gives the x86-64 baseline, `avx2`/`avx2_vnni` at most x86-64-v3, `avx512` at most
x86-64-v4, and no cap is host detection itself, so uncapped code bytes are unchanged
(`target::capped_native_flags`, applied in `make_isa_full`, which every JIT and object entry reaches).
A cap only removes flags; an explicit `--target-cpu` level is exact and ignores it. So in a
`WUKONG_MAX_ISA=scalar` run no code Wukong generates or dispatches to uses AVX: not the runtime's
dispatch, not the raw-AVX2 loop kernels, not Cranelift's own code. The C library is outside the cap
and selects its own paths from the real CPU (glibc's IFUNC `memcpy`, and the `fmaf`/`floorf`/
`nearbyintf` libcalls Cranelift emits for `fma` and rounding below FMA/SSE4.1), so the cap emulates
a smaller machine for Wukong's code, not for the whole process.

`--emit=check` returns after sema and the lints (lex, parse, imports, sema, lints; no MIR lowering),
printing any lint warnings and exiting 0 when no error (or denied lint) was reported.
`--emit=tokens` and `--emit=ast` return *before* import loading, so they show the root file only;
every later stage — and `--run` — sees the merged program. A verifier failure is an internal compiler
error (a bug in a pass), reported as prose and mapped to a non-zero exit, never a user diagnostic;
`--emit=mir`/`--emit=mir-high` verify inside `emit_mir` and print the dump regardless of the result —
the ICE lines go to stderr first and a failure only changes the exit code — so a broken program still
prints the MIR that shows why. `wukong_opt`'s per-pass verify-each is `#[cfg(debug_assertions)]`, so
in a release compiler this gate is the only verification that runs.

**Compile-time shape of a compile.** For `--run` and `--emit=exe` (one entry point, `main`) the driver
drops every function `main` cannot reach through calls or `FuncAddr` after lowering
(`wukong_driver::prune`); `--emit=obj` exports everything and the MIR dumps, autodiff stages and
`--backend=gpu-native` keep the whole program. The native backend follows the source level:
`-O0` builds the Cranelift ISA at `opt_level=none`, `-O1`+ at `speed` (`BackendOpt`), and both the JIT
and the object path compile functions in parallel (`WUKONG_PAR_CODEGEN=0` serializes). Lowering
keeps aggregate copies straight-line only when a gradient was requested
(`LowerOptions::unroll_for_autodiff`), capped at 1 KiB per operation. `--time-passes` (stderr table,
or one JSON line under `--error-format=json`) and `--time-trace=<f>` (Chrome trace JSON) report what a
compile spent per stage, with the optimizer's per-pass breakdown (`optimize_timed`, byte-identical
MIR), Cranelift's codegen/object-write split and the heaviest MIR functions (`wukong_driver::report`).
The timer is inert unless asked for and reports from its `Drop`, so every exit path prints.
`--depfile=<f>` writes a Make rule over the `SourceMap`'s files — exactly the root and each import
the loader read — with gcc `-MP`-style empty rules for the imports.

**Several artifacts, one compile (`--emit=a,b,..`).** `EmitStage::parse_list` sorts the list into
pipeline order — `EmitStage`'s declaration order, which its derived `Ord` makes load-bearing — and
splits it into `Options::emit`, the stage the compile runs to (the last), and `Options::emit_also`,
the stages emitted on the way. `compile` is unchanged for a single stage: every stage stop asks
`opts.emits(stage)` and returns only when the stage is `emit` (or failed), and everything that
decides *what* is compiled — the autodiff lowering, pruning to `main`, `lint_options_for`'s library
roots, `native_reachable`, the object kind, `exports_gradient` — still keys on `emit`, so the
list's last stage decides what is compiled. That is why `check` and `grad` combine only with
`tokens`/`ast` (no later stage runs after `check`; `--emit=grad` returns from `compile` where
`emit_grad` prints the backward MIR, before the MIR and IR stages print) and why `obj` and `exe`
exclude each other (library object vs pruned, mangled executable): any other list is one compile
whose every artifact, without `exe` and without `--grad-of` on an `obj` list, is byte-identical to
that stage alone (`crates/wukongc/tests/emit.rs` checks all six text stages on four programs at
`-O0` and `-O2`). With `exe` the compile is the executable's, so the MIR and IR it also emits are
the pruned program it links. With `obj` and `--grad-of` it is the gradient export's
(`exports_gradient`): `unroll_for_autodiff` lowering (so `mir-high` differs too), at least `-O1`
with `OptimizeOptions::AUTODIFF_INPUT`, and `append_gradient`'s `<fn>_grad` in the `mir`,
`mir-text` and `llvm-ir` — the program the object is built from, which a lone `--emit=mir`
(it ignores `--grad-of`) does not show (`emit.rs` pins the object against `--emit=obj --grad-of`'s
and the gradient in the MIR and IR). Artifacts leave through one function, `put_artifact`:
stdout for a single stage (byte-for-byte the old path), stdout under a `==> <stage> <==` header for
a list, or `Options::artifact_path` — `-o`'s path with the stage's `EmitStage::extension` — for a
list with `-o`; the object and executable reach `emit_native` through a clone of the options whose
`output` is that path. `report::depfile_targets` names every file written. The combination rules
that involve other flags (`-o`, `--run`, `--train`, `--depfile`, `--passes`, a `.mir` input) stay in
`wukongc`'s `parse_args`.

**The REPL (`wukong_driver::repl`).** `wukongc --repl` keeps the inputs it accepted as source text
and, for every new input, compiles a whole program from them through the same front end, lowering,
optimizer and backends as `--run`. Each input is its own `SourceMap` file, `<repl:N>`, so diagnostics
need no remapping: an item input parses with `parse_module_tokens_with_lints`, a statement input
with `wukong_parser::parse_stmts_tokens_with_lints` (a block body to end of input, the tail
expression kept), and the wrapper `main` is parsed from source too — `fn main() -> i32 {
__wukong_repl_replay; printf(MARK); eprintf(MARK); __wukong_repl_input; [printf("{:?}\n",
__wukong_repl_value);] printf(END); return 0; }` — and its placeholders are replaced by the parsed
statements of every earlier statement input (the replay), the new input's statements, and its tail.
The run captures stdout (`run_capturing` on either backend) and stderr
(`wukong_runtime::fmt::begin_stderr_capture`, which `write_stream` and the interpreter's stderr
path honour); only what follows `MARK` is shown, and `END` present is how the session knows the
input ran to its end (an `exit(0)` is otherwise indistinguishable from completion). An item input is
compiled with the replay and the replay is run, so a redefinition that breaks an earlier input is
refused; items are grouped (an `impl` marker with its flattened methods) and keyed by name, and a
later group with the same key drops the earlier one. A tail's value is printed when sema accepts it
as a `printf("{:?}")` argument (a second check of the printing program decides, so the REPL never
re-derives which types print); otherwise its type is shown. `wukong_runtime::process::detach_stdin`
gives every program the session runs an empty standard input. Where an input ends is decided on
the lexer's tokens (`repl::needs_more`): a stack of the brackets still open, from which an
unterminated string (`E0102`; a string ends at its line) drops every bracket opened on its line
before it, an open block comment (`E0103`), and an input that is so far only attributes
(`repl::after_attributes`, the skipper `classify` also uses) all read on. The rules a user sees
are in the language guide, "The REPL".

## Diagnostics

Every diagnostic has a stable code (`E01xx` lexer, `E02xx` parser, `E03xx` name res, `E04xx` types,
`E05xx` shapes, `C0xxx` codegen); `E00xx` is reserved for generic/internal-surface errors and
currently holds none. Codes are permanently stable: never renumber one, and never recycle a retired
number — `E0001` ("malformed type syntax") is RETIRED and unallocated, so `--explain E0001` resolves
to nothing and malformed types report `E0203`/`E0204`. `E0404` is likewise an unfilled gap; leave it
alone.

The catalog in `wukong_diag::catalog` gives each code a title and an extended explanation surfaced by
`wukongc --explain <CODE>`. It holds 30 entries at HEAD and is gated in both directions by
`wukong_diag/tests/catalog_coverage.rs`: every code a crate emits must have an entry, and every entry
must have an emitter (its `RESERVED` exception list is empty, which is the healthy state). A third
gate, `every_catalogued_code_is_exercised` in `crates/wukongc/tests/fail.rs`, requires every
catalogued code to be reached by a `tests/fail/` fixture or by an inline `LEXER_PARSER_PROBES` probe.
So adding a diagnostic is always three edits: the catalogue entry, an emit site spelling the code as a
bare `"Ennnn"` literal (the scanner skips comment lines, so prose may cite codes freely, but a
`format!`-built code is invisible to it), and a fixture or probe.

The lexer's one table-driven diagnostic is `E0105`: `wukong_lexer`'s `confusables` module lists the
Unicode lookalikes of ASCII punctuation, operators and whitespace (after rustc's
`unicode_chars.rs`), each with its name, the ASCII text it imitates and a recovery — produce that
token, skip it as whitespace, or (for a quote, which cannot open a literal) an error token — so the
parse after a pasted `−` is the parse of `-`. A table test pins that every token replacement is
spelled exactly like the token it produces.

The parser's recovery state is two small stacks on `Parser`, both maintained by `bump`: `delims`,
the open `(`/`[`/`{` not yet closed (a closer pops its innermost matching opener and anything
unmatched above it), and `expected`, every token an `eat` tried and failed to find since the last
consumed token. A failed `expect` reports `expected one of <expected ∪ wanted>`, anchors at
`prev_span().shrink_to_hi()` when the token found is on a later line, labels the innermost matching
opener of a missing closer (then drops it, so the next failure labels the next opener out), and steps
over the lexer's `Error` tokens first — `Parser::error` likewise drops a complaint about an `Error`
token, which the lexer has already reported.

Warnings (`W0xxx`) come from `wukong_sema::lints`, a pass separate from `wukong_sema::check` so
that `check`'s result — and every test pinning its exact diagnostics — is unchanged. The driver
calls `lints::run` only after a clean `check`, then `lints::apply_levels`, which resolves each
finding's level (the lint's default, then the command line's `-A/-W/-D`, then the innermost
`@allow/@warn/@deny` whose item, parameter or statement span contains the finding) and drops,
keeps or promotes it. A denied lint stops the build before lowering; warnings are held back and
rendered after lowering's diagnostics, so a failed build's first diagnostic is always an error.
The pass does its own scope-faithful name resolution over the AST and reads only `SemaResult`'s
node types and def map. Each lint errs toward silence — `unreachable_code` reuses sema's
definite-return rule for divergence (a `match` diverges only with a catch-all arm) so the user can
always delete what it reports. `dead_code` roots come from the driver (`lint_options_for`: `main`,
the autodiff root, everything under `--emit=obj|llvm-ir`). Lint codes are catalogued like errors;
`catalog_coverage.rs` scans `"Wnnnn"` literals too, and a `tests/warn` fixture (the whole stream,
in order, is compared by `warn.rs`) exercises each one. `warn.rs`'s `corpus_is_warning_free` holds
every `tests/run` and `examples` program (and every module they import) to zero warnings.

One lint is not found by `lints::run`: `W0008 unused_imports` needs to know which `import` a name
came through, which only the import loader's module-scoped resolution knows — the merged program
sema sees has one flat table of renamed symbols. `loader/resolve.rs` records, while it rewrites each
file, the import every resolved name, qualifier and `{x, y}` selection went through (`ImportUse`),
plus every method name called and every trait named as a bound anywhere in the program. An `impl`
is found without a name, and sema's method table and trait-bound check are program-wide, so an
import is also used when a module it loads (transitively, not counting its own importer, nor the
always-loaded root) holds an `impl` defining a called method or implementing a bound trait
(`impls_through_imports`) — whichever file the call or the bound is in. That keeps every edge of a
shortest path from the root to a needed `impl`, so applying every W0008 fix at once is safe.
`load_imports` returns a W0008 for every import nothing went through — in any file — with a
machine-applicable deletion, and the driver adds them to the lint stage's findings unless the
artifact is a library (`--emit=obj|llvm-ir`). Imported files' `import` items stay in the merged
module (as the root's always did; nothing downstream reads them), which is what lets
`apply_levels` honour an `@allow(unused_imports)` written on one.

The parser contributes two lints of its own, `W0030` (a statement's missing `;`) and `W0031` (a
line-leading `(`, `[` or touching `-` that continues the expression above), into a separate
`Parser::lints` list. Only `parse_module_tokens_with_lints` returns it (as `ParseOutput::lints`);
every other entry point drops it, so the many callers that assert a parse produced no diagnostic
are untouched by a warning. The driver parses the root with it and the import loader parses every
imported file with it too, appending that file's lints (each span keeps its file's `SourceId`) to
the root's, so all of them reach the same level resolution. Every simple statement arm of
`parse_block` — `defer` included — ends through `end_stmt`, and `W0030` carries a
machine-applicable `;` insertion. `W0030` stays quiet for a statement that pushed an error or contains a
lexer `Error` token (its `;` is not what is missing); `W0031` fires only when the innermost open
delimiter is a `{`, i.e. where a statement could have begun. The driver keeps the root file's
parser lints until the lint stage and resolves them with the sema lints in one `apply_levels`
call; on any earlier failure it renders them after the errors (`render_lints`).

The terminal renderer is rustc-style (carets, gutters, `--> file:line:col`). Two rendering rules are
contractual: a caret run is as wide as the span's *rendered* width, never the column difference, and
both the echoed source line and the caret padding expand tabs at a fixed `TAB_WIDTH = 4` — so carets
land under the construct in tab-indented source. East-Asian wide characters count two columns, and
columns past the end of a line count one each. If you touch `TAB_WIDTH`, `rendered_width` or
`expand_tabs`, touch all three.

A multi-line span is drawn whole, with rustc-style connectors. A label ends at its *last* character,
the one before `hi`, so a span whose `hi` is column 1 of the next line ends on the line before. A
label is multi-line when that character is on a later line than its first. At most one label per
diagnostic gets connectors: the primary label if it is multi-line, else the first multi-line label
in position order. Such a diagnostic gives every echoed line and underline a 2-column margin after
`| `. The connector label's first line is marked `/` when only whitespace precedes the span.
Otherwise a start-marker line of `_` runs to a `^` (`-` for a secondary label) under its first
character. Its later lines carry `|`, and an end-marker line `|___^ message` points at its last
character:

```text
 9 |   fn pick(n: i32) -> i32 {
   |  ________________________^
10 | |     if n > 0 {
11 | |         return 1;
12 | |     }
13 | | }
   | |_^

 9 | / struct List {
10 | |     head: i32,
11 | |     tail: List,
12 | | }
   | |_^
```

A label longer than 6 lines prints its first two and last two lines around one `...` gutter line,
and the gutter is then at least 3 wide. The gutter also widens to fit the label's last line. Other
labels keep their own frames at their sorted positions and are not interleaved into the connector
frame (rustc interleaves them). Any other multi-line label is cut at the end of its first line,
underlined to one column past it. A diagnostic with no multi-line label renders exactly as it did
before connectors existed. Only the human form, and JSON's `rendered` copy of it, changes: the JSON
span positions never depend on the renderer.

The printed `-->`/`:::` column is the 1-based *char* column, the same number as JSON `col`. That is
the editor/LSP jump contract, and it is kept by decision (N6): display width (tabs, wide characters)
only places carets, underscores and marks. The tests `tab_indented_line_aligns_with_its_caret`,
`caret_aligns_under_a_wide_character` and `header_column_is_the_char_column_json_reports` in
`render.rs` pin it. A display-width header column would be a separate change to both `render.rs` and
`json.rs`.

`--error-format=json` emits one JSON object per line (JSON Lines) for tooling: `severity`, an optional
`code`, `message`, a `spans` array of `{file, line, col, primary, label, line_end, col_end,
byte_start, byte_end}` (dummy spans omitted; `line`/`col` are the span's start, `line_end`/`col_end`
its exclusive end), and a `notes` array of `{kind, message}` — that prefix is schema v0 and never
moves. Schema v1 appends `suggestions` (`{message, replacement, applicability, span}`, `span` null
when dummy), `children` (rustc-style `{severity, message, spans}` mirroring the notes), `rendered`
(the color-off human form) and a final `"version":1`. Driver-level errors with no source span (IO,
`--train`, autodiff, run and link failures) go through the same writer as code-less diagnostics. The writer
is hand-rolled (no serde): `"`, `\`, `\n`, `\r`, `\t` get short escapes and every other character
below `0x20` becomes `\u00xx`, so the line always stays parseable. `crates/wukongc/tests/fail.rs`
greps for the literal substring `"code":"Ennnn"`, so key order and the absence of whitespace around
`:` are part of the contract.

## MIR

A `Function` owns a value arena (`ValueId -> MirType`), a list of `BasicBlock`s, and an entry block.
Each block has typed parameters, a straight-line list of `Inst { result: Option<ValueId>, op: Op }`,
and exactly one `Terminator` (`Ret`, `Br`, `CondBr`, `Switch`, `Unreachable`). `Switch` is a multi-way
branch on an `i8`..`i64` scrutinee whose case values are stored canonically (sign-extended from the
type's width) and pairwise distinct; only `wukong_opt`'s late switch formation creates one, and the
verifier, interpreter, Cranelift (`cranelift_frontend::Switch`), textual LLVM (`switch`) and every
terminator visitor handle it (the `gpu-native` lowering declines it). `Op` has 29 variants: constants
(`ConstInt`/`ConstFloat`), `Bin`/`Cmp`/`Cast`, the unary `Neg`/`Not`, `Select`, memory
(`Alloca`/`Load`/`Store`/`Gep`), `AtomicAdd` (`atomic.add vV, vP`: an indivisible, sequentially
consistent integer read-modify-write from `atomic_add(place, v)`, wrapping at the value's width,
with no result), `Call`, `Barrier` (cooperative thread-block fence from
`sync_block()`), the float/SIMD primitives
(`Splat`/`ExtractLane`/`Iota`/`Fma`/`Sqrt`/`Round`), the lane-moving SIMD ops
(`Widen`/`Narrow`/`Shuffle`/`InsertLane`), `IntUnary` (`popcnt`/`clz`/`ctz`/`bswap` on a scalar
`i8`..`i64`, semantics in `IntUnOp::eval`; no `i8` `bswap`, never widened by the vectorizer,
declined by `gpu-native`), the address-of ops (`FuncAddr`, `GlobalAddr`), and `VecKernelCall`.
`Widen` converts one half of a vector's lanes to twice their width (`SExt`/`ZExt`
/`FpExt`) and `Narrow` joins two vectors into one of half the lane width (truncating, saturating
signed or unsigned, or `f64 -> f32`) — the two ops that let one loop compute at several lane widths;
`Shuffle` is a two-source constant lane permutation and `InsertLane` writes one lane. `Bin` has one
operand-shape exception: a vector **shifted by a scalar** amount of its lane type, masked to the
lane width exactly as a scalar shift is (the shape every vector ISA's shift takes). All four are pure
lane maps, so the interpreter and the native backend agree on them bit for bit; `gpu-native`
scalarizes them. Of the SIMD three, `Splat` broadcasts one scalar to every lane, `ExtractLane`
reads one lane back out (the only way from vector to scalar form, and what keeps a widened float
reduction's accumulate serial), and `Iota` is the lane-index ramp `<0, 1, .., n-1>` — the one
non-uniform vector *source*, since every other op is lane-wise and so any expression over splatted
scalars is the same in every lane. The loop vectorizer needs it to widen a body that reads its
induction variable as a value rather than as an address: lane `k` is scalar iteration `i + k`, so
`if c == target` widens to `cmp.eq(splat(i) + iota, splat(target))`, a per-lane mask. Every op produces exactly one result
except `Store` and `AtomicAdd` (never), `Call` (void callees such as `print`) and an elementwise
`VecKernelCall`. `AtomicAdd`'s value is a scalar `i8`..`i64` (never `i1`, a float or a vector — the
verifier checks it), and its old value is deliberately unobservable: integer addition mod 2^n
commutes, so the final memory is the same under every interleaving, which is what keeps `@parallel`
code that combines into one location bit-identical between the serial interpreter and native. The
interpreter adds and masks; Cranelift emits `atomic_rmw add` with plain flags (no alias region —
the escape scan's fallback publishes the atomic's pointer, so its location is never private or
`@noalias`-promised, and every other access to it is untagged too); the LLVM text emits
`atomicrmw add ... seq_cst`; `gpu-native` declines it. Every optimizer pass treats it as an opaque
memory effect, exactly where it treats `VecKernelCall` as one (see `CONTRIBUTING.md`, "Adding a MIR
`Op`"; `wukong_opt/src/atomic_tests.rs` pins the hazards). The source builtin
`atomic_add(place, v)` is typed in `wukong_sema` (`type_atomic_add`: a place other than a bare
name, an integer place type, a value of that type, `=`'s projection-write rule via
`check_projection_write`) and lowered in `wukong_mir_build/src/atomics.rs` with the compound
assignment's evaluation order (value, then the place's GEP and slice bounds check). The `@parallel`
region proof (`match_parallel_region`, shared by the whole-function outliner and mid-function
regions) admits it on one element of a captured fixed-size array: such an array needs no
disjointness proof, because the adds commute, but no other access in the region may touch it or any
captured parameter that can share its storage (`noalias::could_share_storage` without a
`promised_disjoint` promise). Every other place declines the region. A
string literal lowers to a read-only `.rodata` blob in `Program::statics` (a `StaticData { name,
bytes }`) addressed by `Op::GlobalAddr`, so a returned or threaded `*u8` stays valid after its
defining frame is gone. `Op::VecKernelCall` — a call into a synthesized 256-bit AVX2 kernel — is
covered under the vectorizer below.

The front-end lowers in **clang style**: one `alloca` per local, with `load`/`store` on every use.
`Op::Alloca` carries an `align` (bytes; `0` = natural, which every compiler-made slot uses and
which prints nothing, so dumps are unchanged; otherwise a power of two up to
`MAX_ALLOCA_ALIGN = 4096`, checked by the verifier, from `@align(N)` on a `let` of an owned
aggregate). It only ever raises the alignment: Cranelift reserves `bytes + want - 16` and rounds
the base up at run time when the wanted alignment exceeds the stack pointer's 16 (the same scheme
32-byte slots always used), the LLVM text emits `align max(natural, N)`, the interpreter ignores it
and `gpu-native` declines anything above 8. `Op::Alloca` also carries an `AddrSpace` (`Local` or
`Shared`); `shared [T; N]` lowers to
`AddrSpace::Shared`. `sync_block()` lowers to `Op::Barrier`. On CPU backends shared is an ordinary
stack slot and `Barrier` is a no-op (interpreter) or compiler fence (Cranelift). On gpu-native,
`lower.rs` maps shared allocas to PTX `.shared __user_smem[]` and `Barrier` to `bar.sync 0` — no
changes to `gpu.rs` or the `ptx_*.rs` families. At `-O2`, `wukong_opt::coalesce` classifies loop
memory accesses by IV element stride (`loop_info`). The matching emission — `ld.global.v4.f32` /
`st.global.v4.f32` in place of four scalar generic ops — is **implemented but declined**: a
`.v4.f32` access needs a pointer that is both provably in the global window and provably 16-byte
aligned, and only the first is decidable here. PTX has three address windows and which one holds the
frame depends on the emission mode (the megakernel entry's frame is the global `p_frame` buffer; the
single-thread kernel and mega callees use a `.local __frame`), so provenance is tracked positively
in `FnEmit::global_ptrs`. Alignment is not: frame slots pack at 8 bytes, `CoalesceHint::UnitStride`
states a stride rather than an alignment, and `wukong_opt::vectorize` starts its vector loop at the
loop's own `start` without peeling. gpu-native therefore lowers these as generic scalar lanes, which
are correct in every window at every alignment. `@tile(B)` strip-mining in `wukong_mir_build::tile`
expands a half-open unit-step range `for` into tiled inner loops plus a remainder loop before the
recognizer cascade runs. `lower_for` settles the **loop form first** (`tile::tileable_range`) and only
then consumes a width: popping the function-item `@tile(BM, BN, …)` queue before the form check both
discarded the width in silence and shifted the queue, so the next loop was strip-mined at the wrong
tile. A width with nowhere to go — and any argument that is not a positive integer literal — is
`E0309`, and nothing is consumed.
This keeps lowering simple and correct; the optimizer's `mem2reg` pass then promotes those slots to
SSA registers (see below), which is what makes the value-based passes effective.

There is **no dedicated aggregate MIR type**. A tuple or struct is a flat, padded byte buffer whose
local value *is* its base pointer (the convention arrays already follow); `t.0` / `s.f` is a typed
`load`/`store` at the field's byte offset, and `mem2reg` leaves the aggregate slot in memory (its
address is taken by every `gep`) — unless, at `-O2`, `sroa` has split it first: when every access
reaches it through constant-index `gep`s, each accessed leaf becomes its own scalar slot that
`mem2reg` then promotes. Nested aggregates
(a struct/tuple field that is itself a struct, or an array of structs) lay out recursively — the
registry-aware layout helpers resolve a named-struct field that the leaf type crate marks unsized — and
an aggregate field initialized from a *non-literal* value is a leaf-precise deep copy, not a flat
`memcpy` (which would skip a padded non-leading scalar slot under the interpreter's slot-indexed
memory). Pointers/references reuse the same `Alloca`/`Load`/`Store`/`Gep` ops: `&mut x` takes a slot's
address, `*p` loads/stores through it, and `mem2reg` refuses to promote a slot whose address escapes,
so `-O0` ≡ `-O3`. The *pointer itself* is promoted when the entry block initializes it, which is why
`p = &a` / `p = &b` merging at a loop header becomes a `ptr` block parameter while `a` and `b` — now
escaping as branch arguments — stay in memory (`tests/run/ptr_slot_promotion.wk`).
A **constant-shape tensor** lowers like an array — literally: a `tindex` pre-pass
rewrites every multi-dimensional index `a[i, j]` into the flat row-major index expression
`a[i*N + j]` *before* lowering, and the parameter binds as the buffer it is (`[T; d0·…·dn]`, not an
opaque `Ptr`), so the base pointer is not reloaded per element. That matters far more than it sounds:
every kernel recognizer and the whole autovectorizer match a **single-index** `ExprKind::Index`, so
until the normalization existed a 2-index access was invisible to all of them and the shape-typed
spelling of an elementwise loop measured 5.69× slower than its hand-flattened twin. The two spellings
now compile to byte-identical MIR (`crates/wukongc/tests/tensor_parity.rs`). The rewrite is confined
to a contiguous tensor with all-`Const` extents, ≤ `i32::MAX` elements and one ≥32-bit index type —
the bounds that make the flat offset value-identical to the `i64` one it replaces — so a *symbolic*
`Tensor[f32, M, N]` keeps the runtime-dim path below unchanged.
A matmul written in that tensor notation (`c[i,j] = Σ a[i,k]·b[k,j]`,
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
yields its lane type, an elementwise one yields nothing); and terminators are type-correct, including
branch-argument arity/types and the function return type. Blocks unreachable from the entry keep their
type checks with the dominance rule switched off. Given a whole `Program`, `verify_program`
additionally cross-checks every call against the callee's declared signature, including the
`extern "C"` imports in `Program::externs` — compiler-owned external symbols (`print`, the
`wukong_*` kernels) are exempt.

It runs before every backend entry — `--run` and `--emit=llvm-ir|obj|exe` all go through
`wukong_driver::verify_or_ice` — and inside `--emit=mir`/`--emit=mir-high`, where the dump is printed
regardless of the verify result, so a broken program still prints the MIR that shows why (a printed
ICE always exits non-zero). In debug builds `wukong_opt` additionally asserts well-formedness after
every pass that ran; that check is `#[cfg(debug_assertions)]`, not a CLI flag, so it is compiled out
of release. Note the pipeline calls `verify_function`, so `verify_program`'s call-vs-callee
cross-check is exercised only by `wukong_mir`'s own tests.

**Source locations.** Every `Function` carries a side table, `Function::locs`
(`wukong_mir::SourceLocs`): the span of the declaration it was lowered from, and one span per value,
index-parallel to `value_types`. Its contract mirrors `ParamAttrs`: the table may be shorter than the
arena, a dummy span means unknown, and dropping entries is always sound — only a *wrong* entry is a
bug, since a diagnostic would point at code that did not produce the value. No pass renumbers a
`ValueId` (the arena only grows), so an entry can be lost but never transferred to another value; the
verifier rejects a table longer than the arena. `mir_build` fills it through the `Builder`: every new
value takes the builder's current location, and the lowering entry points (`lower_expr`,
`lower_place`, `lower_stmt`, `lower_block`) are wrappers that make their node's span current while
the `*_kind` body runs and restore the outer one after, so a value is attributed to the innermost
construct that produced it, a synthesized node (a dummy span) inherits its surroundings, and a
statement window a recognizer replaces with one kernel call is attributed to its first statement.
Each source function starts at its name (`Builder::set_decl`; a generic instance at its template's,
a lifted closure at its literal) and an outlined `@parallel` region at its loop, so straight out of
lowering every value of a source function is located (`crates/wukong_opt/tests/source_locations.rs`
pins 100% over `tests/run`). Of the optimizer's passes only inlining (the callee's values keep the
callee's spans) and loop unrolling (each copy keeps its original's) carry locations onto the values
they create; every other new value stays unknown, and the same census pins the share that survives
`-O2` from below. The Cranelift backend gives each instruction it emits a `SourceLoc` that is a
`ValueId` (`FnTranslator::translate`): an instruction with a result carries that value when it is
located; a store or void call, which has no table entry of its own, carries the value it reads when
that value was lowered just before it; a terminator carries its condition, scrutinee or return
value. Everything else — an unlocated value's code, the head of a block, a call that reads nothing
just lowered — carries the default (unknown) `SourceLoc`: a location is dropped, never borrowed from
a neighbour. A machine-code range therefore maps back to a span through `Function::value_loc` — the
input a DWARF line table needs (not emitted yet) — and a backend failure is reported at the span of
the value being lowered, else at the function's declaration. Locations never change code: the
object bytes with and without the table are pinned equal. The interpreter and the printer ignore
the table, so `--emit=mir` is unchanged by it.

`wukong_mir::print` renders MIR deterministically — the text behind `--emit=mir` and
`--emit=mir-high`, and stable enough for snapshot tests. It is also the grep surface for checking what
lowering did: a SIMD type prints LLVM-style as `<4 x f32>` (so `--emit=mir -O2 | grep '<4 x f32>'`
tells you the 128-bit vectorizer fired), a synthesized 256-bit kernel prints as
`veckernel #0(ptrs=…, scalars=…, n=…)`, and a dispatched runtime kernel shows up as a plain
`call wukong_…`. Types are printed on `const.<ty>`, `load <ty> p`, `alloca <ty>`, `<cast> v to <ty>`
and every function/block parameter list; plain arithmetic prints untyped (`v5 = fadd v1, v2`), so read
the operand's defining line for its type.

### Typed textual MIR (`--emit=mir-text`, `.mir` input)

`wukong_mir::text` is the form of MIR that reads back. `print_program_typed` prints
`print_function`'s lines with what a reader needs added:

- a first line `; wukong mir text v1`, and `;` comment lines anywhere;
- `declare extern "C" fn NAME(i32 signed, ptr) -> i32 signed`, with the C signedness explicit;
- `static NAME = hex <bytes>` for each `.rodata` blob;
- parameter promises on the function header, `v0 ptr noalias slice extent=64x4`;
- `  entry bbK` when the entry block is not `bb0`;
- `  values N` when the value arena outlives the last defined value;
- a type on every result, `v7: <4 x f32> = fadd v5, v6`;
- every float constant as its `f64` bits, `const.f32 0x8000000000000000` (`Display` drops a NaN's
  payload).

`parse_program` reads it back: ids are kept verbatim (the arena is sized to the largest id plus one,
or `values N`), blocks must come in id order, and names are split right-anchored, so a generic
instance such as `f$[f32; 4]` survives. The printer post-processes `print_function` in lockstep and
asserts its line count, so a layout change in `print.rs` fails loudly instead of producing text the
parser misreads. A synthesized 256-bit vector-kernel recipe has no textual form: a `veckernel` line
is a parse error, and `--target-cpu=x86-64-v2` (below x86-64-v3, no recipe is built) gives a
dumpable program.

`wukongc --emit=mir-text f.wk` prints the optimized program in this form. Given an input ending in
`.mir`, `wukongc` skips the front end: it parses the file (a syntax error is
`<file>:<line>: invalid MIR: <msg>`, exit 1), verifies it (an invalid program is the verifier's ICE,
exit 1), runs the `-O` level's pass list or `--passes` (with `--print-after` and `--verify-each`) and
prints `--emit=mir` or `--emit=mir-text`, the only two stages a `.mir` input takes. The laws,
checked per program over the `tests/run` corpus through the binary
(`crates/wukongc/tests/mir_text.rs`, with `-O2 --target-cpu=x86-64-v2`). Laws 1-2 run on every
program and laws 3-5, which compile it three more times, on every second one by default;
`WUKONG_MIR_TEXT_ALL=1` runs all five everywhere, and all five held on all 548 programs when the
harness landed:

1. `--emit=mir-text` of a dump is the dump, byte for byte;
2. `--emit=mir` of a dump is the original compile's `--emit=mir`;
3. `-O0 --passes=<the -O2 list>` is `-O2` (the replica law of "Running a chosen pass list");
4. `-O2` of the *unoptimized* dump is `-O2` of the program;
5. the `-O2` list cut in two through a dump (its function passes, then
   `unroll,slp,form-switches` on the `.mir`) is `-O2`. About a third of those halfway dumps need
   their `values N` line for this to hold.

Laws 4 and 5 are what make a dump a reproducer: the rest of the pipeline does to the parsed program
exactly what it did to the original. `wukong_mir`'s own tests round-trip a program with every `Op`
variant (built through an exhaustive match, so a new variant does not compile there until it is
covered), every operator kind and terminator, NaN payloads, `-0.0`, `i64::MIN`, statics, signed
externs and parameter promises.

## Optimizer

`wukong_opt::optimize` first runs a whole-program **inliner** (`-O2`), then `PassManager` runs a
list of function-level `Pass`es to a per-function fixpoint (capped at 100 sweeps), then a
whole-program **loop unroller** (`-O2`) with a reduced cleanup sweep over just the functions it
changed, then straight-line **SLP** (`-O2`) with the same sweep over the functions it packed. Functions are optimized **callees first** (call-graph SCCs in Tarjan order), and each SCC is
summarized as soon as it is done (`src/effects.rs`), so a caller's passes know which of its calls
have no side effect. The fixpoint uses
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

The pass manager's cache is *managed*: it also holds the function's `AliasInfo`, shared by `cse`,
`dse` and `licm` while nothing mutates and dropped as soon as any pass reports a change or the CFG is
invalidated, and the call-effect table for the callees optimized so far. A cache built by hand (tests,
`inline`'s specialization) is unmanaged and recomputes alias facts on every request.

| Pass            | Level | What it does |
|-----------------|-------|--------------|
| `inline_program`| -O2   | first rewrite every fully tail-recursive function into a loop (`tre::pre_inline`, on a mem2reg'd copy), then inline non-recursive callees **bottom-up over the call graph** (reverse topological order from Tarjan SCCs, so a helper chain collapses in one sweep), scored by a cost model rather than one blanket size cap, then drop callees left uncalled; runs before the function pipeline so the spliced code optimizes in context |
| `Sroa`          | -O2   | scalar replacement of aggregates, ahead of `Mem2Reg`: split a local `Array` slot whose every access is a load/store through a constant-index `gep` chain into one scalar slot per accessed leaf. Each access carries its interpreter slot *and* native byte offset; one cell must be one byte range of one type, distinct cells disjoint, so the split preserves both memory models; a leaf whose fresh default differs from the byte it replaces needs a must-be-written proof |
| `Mem2Reg`       | -O1   | promote scalar int/float **and pointer** `alloca` slots to block-parameter SSA via dominance-frontier phi placement and a dominator-tree rename (a pointer slot's read-before-write path gets the null `inttoptr 0` — see below) |
| `Simplify`    | -O1   | constant folding + algebraic identities (`x+0`, `x*1`, `x*0`, `x^x`, `x&x`, `x\|x`, `x%1`) and integer self-comparison folding |
| `JumpThread`  | -O2   | just before `SimplifyCfg`: an edge into a block whose branch that edge already decides -- a constant argument for the block's parameters (folded width-correctly), or a compare implied by a known compare of the **same two values** (the predecessor's own arm, or a dominating branch arm that is the only way into the dominator chain below it, `and`/`or` operands included; signed and unsigned orders relate only through equality, float compares never) -- goes to a private copy of the block that branches straight to the decided successor. At most 8 copied instructions (no calls, allocas or barriers), 16 threads per call, a 4096-instruction function cap; never copies or targets a loop header. A value of the copied block read past it gets on-demand SSA repair (new block parameters where the copy and the original meet) (`src/jump_thread.rs`) |
| `SimplifyCfg` | -O1   | constant-branch folding (including a constant or degenerate `switch`), jump threading of an `i1` constant edge argument through an empty test block (never past a loop header), straight-line block merging, folding a branch into a predecessor that reaches the same destination (one `cond_br` on `and`/`or`, speculating at most four pure non-trapping non-memory instructions — `src/branch_fold.rs`), and unreachable-block pruning (with renumbering) |
| `SimplifyPhis`| -O1   | drop dead and trivial block parameters that mem2reg introduced |
| `Dce`         | -O1   | remove pure instructions whose results are unused, dead allocas, and unused calls to callees summarized as side-effect free and always returning |
| `Tre`         | -O1   | tail-recursion elimination: a self-call whose result is returned becomes a branch to a loop header carrying the parameters; integer `add`/`mul`/`and`/`or`/`xor` accumulator recursion (`n * f(n - 1)`) gets an accumulator parameter (never a float op); a function with a frame slot left is skipped (`src/tre.rs`) |
| `Cse`         | -O2   | global value numbering: dominator-tree value numbering (a pure computation is reused by every block it dominates, via a scoped table); load forwarding keyed on `(address value, accessed type)` along single-predecessor edges and **through joins** (the intersection of the predecessors' exit states — a location every edge knows with different values becomes a lazily created block parameter); block-parameter numbering (trivial and duplicate parameters); a phi for an expression every arm of a join computed; and calls to side-effect-free callees numbered (read nothing) or forwarded like loads (read memory). `alias` decides which entries a store or opaque call invalidates. Loop headers start with an empty memory table |
| `Dse`         | -O2   | dead-store elimination: a store overwritten at the same must-alias location (same base and constant offset, not just the same SSA value) before any read in its block, or on **every path** out of it for a stable location (a constant in-bounds offset of an entry-block slot, or of a parameter), or retired by the end of the function for a slot no callee can name; plus every store into a never-read private slot that is provably in bounds — a constant offset, or a loop induction variable over a constant range inside the slot in both memory models |
| `LoopCanon`   | -O2   | canonicalize loop entry/latch structure before loop-invariant code motion |
| `LoopDelete`  | -O2   | bypass a loop with a preheader, one exit edge and no side effect whose values nobody outside uses, when it provably terminates (a constant trip count whose induction variable never wraps, or a unit-step strict bound) — what removes an initialization loop `Dse` emptied |
| `LoopFuse`    | -O2   | fuse two adjacent innermost counted loops over the same range: the second's pure preheader moves ahead, its body joins the first's -- appended to the first's block when both are two-block loops, otherwise spliced as regions (a body of at most 8 blocks of jumps and two-way branches, entered by one parameterless block; the first latch absorbs the second body's entry, the second latch closes the fused loop). Every access of every block takes part in the legality rule below; a conditional access only removes executions. Every cross-loop access pair with a write on one object (the same base, or two bases the alias analysis places a whole number of elements apart) needs the same constant stride `c` and a **non-negative distance** `(ka - kb) / c` -- a read `d` iterations behind the writer (`t[j - d]` after `t[i]`) fuses, a read ahead does not -- or no common element; pairs on different objects must be proved disjoint (distinct slots, `@noalias` parameters, a `@noalias` slice's reloaded data pointer, whose header no store may write). The first loop's live-outs may reach the code after the second, read from the first header or through the second preheader's parameters. Never fuses a widenable loop with an unwidenable one, and fuses two widenable loops of which either branches only when `vectorize::would_widen` still holds on a fused copy (two if-converted diamonds would sink stores to two addresses); two unwidenable loops stay apart when either branches on a condition `LoopUnswitch` may decide after LICM -- an invariant, or the counter (or `ext(iv ± k)`) against an invariant (`unswitch::may_decide_a_branch`, a conservative over-approximation of its planners) -- since its split or unswitched copies may widen alone, but not with the other loop's scalar work in each (`src/fuse.rs`) |
| `Licm`        | -O2   | hoist loop-invariant, side-effect-free, non-trapping ops to the loop preheader — including a **load** nothing in the body may write, when it is dereferenceable, sits in the header (which runs on every entry), or runs on every iteration of a single-exit loop, in which case it reads through `select(first_test, addr, dummy)` so a zero-trip entry reads a private dummy slot, and a call to a callee summarized as speculatable |
| `Bce`         | -O2   | slice bounds-check elimination: version an innermost counted loop on a preheader guard (`start < end`, then per check `lo <=u hi && hi <u len` at the two ends of the index range, overflow-safe) so a check-free copy runs when every check is proved and the original, checked loop runs otherwise; a caller-owned slice header is read early only behind the guards of the checks that precede it (`bce.rs`) |
| `LoopUnswitch`| -O2   | after LICM: split a counted unit-step loop whose body branches on `iv` against an invariant at the clamped iteration where the guard changes (two back-to-back loops, each with the guard decided; side exits such as a slice bounds check's trap or a `break` are allowed, a loop value the trap block reads becoming an edge argument). The guard may test a **derived** quantity `ext(iv ± k) ⋈ n` -- the stencil `i + 1 < n`, the tile edge `jo + j < n` with `jo` the enclosing counter (split at `n - jo`), a counter sign/zero-extended to its bound's type, or a counter compared with the other signedness (`(i as u64) < m`) -- once a range analysis (constants, extensions, wrap-free arithmetic, enclosing counters bounded by their header tests) proves `iv ± k` cannot wrap on any iteration the body runs, values non-negative where the two signednesses must agree; an unproved guard declines, and the split point is computed without an intermediate that can overflow. Or unswitch a branch on an invariant `i1` into a preheader branch between the loop and a copy; innermost loops without frame slots, barriers, atomics or vector-kernel calls, size-capped. A loop holding calls is rewritten only when the decided condition is exactly some body block's branch condition and one of its arms -- not the latch, entered only from that block -- dominates every call block: one copy loses that arm, and every call, to simplify-cfg, so no call is duplicated (`calls_confined_to_one_arm`; `src/unswitch.rs`) |
| `Promote`     | -O2   | scalar promotion: a loop-invariant location the loop both reads and writes, with no other possibly aliasing access, is loaded once in the preheader, accessed through a fresh slot (which `Mem2Reg` turns into a loop-carried value next sweep) and stored back on every exit -- in the exit block when the loop alone reaches it, otherwise in a new block on the loop's edges into it (bounds-check elimination's checked and check-free copies share one exit, so a slice reduction needs this) -- with the same speculation rules as `Licm` and a store-safety rule (private slot, or a store that runs whenever the loop does) |
| `Vectorize`   | -O2   | widen eligible canonical MIR loops after LICM, then feed the result through the normal fixpoint cleanup |
| `unroll_program`| -O2 | unroll a counted two-block loop: **completely** when its exact trip count is at most 32 and the copies total at most 128 instructions; otherwise 4x (body up to 16 instructions) or 2x (up to 32), with a wrap-safe guard and a remainder that is a loop for an unknown trip count, nothing when the factor divides a known one, and straight-line copies for a known leftover; runs **after** the function pipeline, and the functions it changed get a reduced simplify + simplify-cfg + simplify-phis + cse + dce cleanup sweep |
| `unroll_jam` | -O2 | inside `unroll_program`, ahead of the unroller: a canonical two-deep nest whose inner loop carries a recurrence or reduction runs **four outer iterations side by side** — four row prologues, one shared inner counter, four inner bodies per iteration, four epilogues in row order — with the untouched nest as the remainder; nothing is reassociated, so each row's values are bit-identical. Legal only when every stored object is split into per-row slabs (row-major `x[r*P + t]` with the window checked, the pitch possibly the inner loop's runtime bound, or time-major `x[t*D + d]` with every row below `D`), objects that may be one buffer are range-checked once in the preheader, and nothing calls, allocates or can trap (`src/unroll_jam.rs`; `WUKONG_NO_UNROLL_JAM=1` turns it off) |
| `slp_program` | -O2 | once, after unrolling and its cleanup: pack `W = 16 / bytes(T)` stores in one block to consecutive elements of one object (`f32`/`f64`/`i32`/`i64`), whose values are one lane-wise operation tree, into one 128-bit vector store; the packed functions get the unrolling cleanup sweep (`src/slp.rs`; see **Straight-line SLP** below) |
| `FormSwitches` | -O2 | once, after unrolling: turn a chain of four or more `cmp.eq`/`cmp.ne` tests of one integer scrutinee against constants into one `switch` terminator, hoisting the pure link compares into the head. A match sema proved exhaustive reaches it with no trap at all (`lower_match` took its last arm untested, so that arm's body is the default); an unproven match's trapping (`unreachable`) fall-through stays the default, never folded, because that block can be reached and folding it here would make `-O0` and `-O2` disagree (`src/switch_form.rs`) |

`-O3` currently runs the same pass pipeline as `-O2`: there are no `-O3`-exclusive passes yet
(`PassManager::standard` adds passes at `-O1` and `-O2` only).

### Running a chosen pass list: `--passes` and `--print-after`

`wukong_driver::passes` (`crates/wukong_driver/src/passes.rs`) is a replica of
`optimize_preserving` built only from `wukong_opt`'s public API, so a pipeline can be named on the
command line. `--passes=<list>` takes comma-separated names:

- **Function passes**, by their `Pass::name()`: `sroa`, `mem2reg`, `simplify`, `jump-thread`,
  `simplify-cfg`, `simplify-phis`, `dce`, `tre`, `cse`, `dse`, `loop-canon`, `loop-delete`, `fuse`,
  `licm`, `bce`, `unswitch`, `promote`, `vectorize`. Consecutive ones form one group, run to a
  per-function fixpoint exactly as `PassManager::run` runs the standard list (callees first, call
  effects summarized per SCC, clean-tracking). A name may repeat.
- **Program steps**, run once where named, each closing the group before it: `inline`
  (`inline_program`), `unroll` (`unroll_program` plus `PassManager::unroll_cleanup` over the functions
  it changed), `slp` (`slp_program` plus the same cleanup; `WUKONG_NO_SLP` still applies) and
  `form-switches`. Under `--verify-each` every pass of those cleanup sweeps is verified as
  `optimize_verified` verifies it (`PassManager::run_subset_verified`), so a failure there names
  the cleanup pass and the function, like a failure in a group.

Each `-O` level is one such list (`passes::standard_spec`):

| Level | List |
|---|---|
| `-O0` | (empty) |
| `-O1` | `mem2reg,simplify,simplify-cfg,simplify-phis,dce,tre` |
| `-O2`, `-O3` | `inline,sroa,mem2reg,simplify,jump-thread,simplify-cfg,simplify-phis,dce,tre,cse,dse,loop-canon,loop-delete,fuse,licm,bce,unswitch,promote,vectorize,unroll,slp,form-switches` |

**The law:** `run_steps(parse_spec(standard_spec(l)))` leaves MIR byte-identical to
`wukong_opt::optimize(p, l)` at every level, with and without `--verify-each`
(`passes::tests::standard_spec_replica_matches_optimize_at_every_level`, over programs that reach
inlining, the loop passes, complete unrolling, SLP, switch formation and tail calls). It makes the
list syntax an executable specification of the standard pipeline, and it is what catches drift: a
pass added to `PassManager::standard`, or a new stage in `optimize_preserving`, fails it until
`standard_spec` and `passes::Step` follow; `crates/wukongc/tests/mir_text.rs` checks the same law
at the CLI for every corpus program. `-O` still sets the Cranelift level when `--passes` names
the list. On this path `--time-passes` reports the optimize stage as one interval, without the
per-pass split `optimize_timed` gives.

`--print-after=<name>` runs the level's list (or the `--passes` list) through the same replica and
writes to stderr, leaving stdout byte-identical: after each run of the named function pass on a
function, `; *** MIR after <pass> (fn <name>) ***` and the function, or the one line
`; *** MIR after <pass> (fn <name>): unchanged ***` when that run changed nothing; after a named
program step, `; *** MIR after <step> ***` and the whole program. `all` matches every pass and step.
A snapshot is written the moment its pass returns (the wrapper renders through a symbol-for-symbol
copy of the interner), so a pass that panics leaves every earlier snapshot on stderr. The cleanup
sweeps inside `unroll` and `slp` are part of the step and print with it. Both flags are refused
(exit 2) with `--emit=tokens|ast|check|mir-high`, where no optimizer runs, and on a gradient compile
(`--emit=grad`, `--train`, `--emit=obj --grad-of`): the replica has no autodiff roots
(`inline_program_preserving` is crate-private), and autodiff differentiates the standard pipeline's
forward without SLP.

**The MIR loop vectorizer** (`src/vectorize.rs`) widens a canonical counted loop (a step of `+1`
or `-1`; one straight-line body, one if-converted diamond, or a small nest of conditionals) into
128-bit vectors, keeping the original loop as the scalar epilogue. Its invariant is that lane `k` of a group executes exactly what scalar
iteration `i + k` would have, so interp and native stay bit-identical and `-O0` matches `-O2`. The
group width is `W = 16 / narrowest lane bytes` over everything the body widens (LLVM
LoopVectorize's rule); a wider value is up to four parts, and lane-count-changing casts are
`Widen`/`Narrow` chains with exact lane semantics (a saturating `f32 -> i8` goes through `i32`; an
unsigned-to-float through a zero-extend). A **complete interleave group** — accesses `x[k*i + r]` for
every residue `r` of a stride `k <= 4`, all loads or all stores, one element type — becomes `k`
contiguous vector accesses and a tree of two-source `Shuffle`s; `loop_info::never_the_same_element`
(a GCD test) proves two stores `x[2i]`/`x[2i+1]` never meet, and the runtime alias guard scales each
range by its stride. A step with **no vector form** is replicated (LLVM VPlan's REPLICATE recipe):
it runs once per lane on that lane's own scalars, and its result is packed with `InsertLane` only
where a widened step reads it. That covers integer division (MIR's scalar forms never trap), a
per-lane shift amount, a cast with no route, and a **gather** — `loop_info` gives a load whose index
is data the object it reads (`MemAccess::gather_base`) and accepts it only when no store in the loop
can reach that object (two slots, `@noalias`, or a non-escaping local). Two guards keep replication
honest: the induction variable's scalar *address* face may never meet a lane-varying operand
(`x[i + q[i]]` is declined, not replicated with the group's first `i`), and replication is taken only
beside at least as much widened arithmetic. A **scatter** — a store `o[q[i]] = v` through a
per-lane address — is replicated the same way (`Plan::Scatter`): `W` scalar stores, emitted in lane
order, which is iteration order in both directions, so of two lanes (or groups) that write one
element the later iteration lands last, exactly as in the scalar loop. It is legal only when **no
other access in the loop** — load or store, affine, invariant, gathered or scattered — can touch the
scattered object (`loop_info::dependence` records the object in `MemAccess::gather_base` and tests
every other access's object against it with the gather's apart test), because then nothing in the
loop can observe the order. A read-modify-write histogram (`h[q[i]] += x`, which the `scatter_add`
recognizer covers), two scatters into one object, a scatter beside any other access of its own
object, an object the analysis cannot prove apart, a scatter inside an if-converted arm and a
checked scatter (its bounds check is not on the counter) stay scalar
(`tests/run/vec5_scatter.wk`). Gapped groups, strides above 4 and masked group accesses stay scalar
too.

**Countdowns** (`VecPlan::dir`): `while i > 0 { i = i - 1; .. }` and `while i >= 0 { ..; i = i - 1
}` widen with the group walking down. Lane `k` is still the group's `k`-th iteration (counter
`i - k`), so every fold, merge and mask is unchanged; a contiguous access loads or stores the `W`
elements ending at lane 0's element as one vector per part, reversed by a single-source `Shuffle`
(bit-exact, nothing computed), the lane ramp is `splat(i) - iota`, and the trip is `start - end`
(`+ 1` for `>=`, taken only below the type's maximum). The runtime alias guard turns the countdown's
range around before building each window. Search loops, arm bounds checks, argmax/FindLastIV (whose
folds read counter order as iteration order) and interleave groups stay with ascending loops, as
does a negative stride on an ascending counter (`x[n - 1 - i]`). `WUKONG_NO_REVERSE_VEC=1` restores
the decline.

**Trip counts.** A runtime trip is `end - start` in the counter's own type, which wraps: an `i32`
loop from `2147483600` to `-2147483600` runs no iteration but that difference is `96`. So the guard
also tests `start < end` under the header's own signedness (`start > end` for a countdown) before
reading the difference as a count — the vector loop runs no group whenever the scalar loop runs
none, and every fold form then hands back its seed bit for bit (`tests/run/vec4_zero_trip_reductions.wk`).

**What the guard may read.** The guard runs on every entry, a zero trip included, and re-reads the
fat pointer of each slice the loop indexes (for an alias check's range, and to hoist the data
pointer the body reloads). A fat pointer that cannot fault is read there freely: a `[]T`
parameter's or a stack slot's (`never_faults`). Any other one — a slice field reached through a raw
`p: *mut Layer` — is read by every iteration of the loop (an arm may not speculate it), so the
scalar loop reads it once it runs at all, and the guard reads it only when a group will run:
directly under a constant trip, through `select(n_vec != 0, addr, slot)` under a runtime one, a
private stack slot standing in when no group runs (`GuardReads`). A search loop may leave before
it reaches such a read, so its guard re-reads only a fat pointer that cannot fault, and a hoisted
data pointer it may not reach stays where the group reaches it. Before this, `k(x, 0 as *mut
Layer, 0)` over `(*p).w[i] = x[i]` returned at `-O0` and stopped with "null pointer dereference"
at `-O2` (`tests/run/vec4_speculation_raw_pointer.wk`).

**Reductions** come from `loop_info::classify_carried`. Besides a binary or `fma` combine, two
families are recognized. A **min/max** spelled as `select(cmp(pred, a, b), a, b)` (the lowering of
`max`/`min`/`fmax`/`fmin`) or as a branch that only chooses between the accumulator and the value it
was compared against (`if x > m { m = x }`) records its compare and the accumulator's positions
(`SelectCombine`). A **guarded accumulate** (`if c { s = s + x }`, either arm, `Reduction::merge`)
is a join parameter fed by the update from one arm and the untouched accumulator from the other.
Both carry exact read counts so no partial is observed. In the vector loop, integer
`add`/`mul`/bitwise reductions and integer min/max up to 32-bit lanes keep lane-parallel partials
(seeded with the identity: `INT_MIN`/`INT_MAX`/`0`/all-ones for min/max), a guarded one blended per
part with `select(mask, acc op x, acc)`; every float reduction and every 64-bit min/max folds `W`
lanes into the scalar accumulator **in index order**, re-emitting the loop's own combine (its
compare and select for a min/max), and a guarded one keeps `c_k ? acc op x_k : acc` per lane. No
float is reassociated and no identity element is ever folded into a float, so a running `-0.0`, a
NaN payload and a NaN-then-number sequence come out exactly as the scalar loop computes them. A
branch that also records the counter (`if x[i] > m { m = x[i]; at = i; }`, an argmax/argmin) is one
reduction *as a pair* (`Reduction::arg`, an `ArgIndex`): with a strict compare, the update on the
`then` arm and the counter as the index, each lane keeps its own `(value, index)` partial (the
index selected from the lane ramp under the value compare's mask), and the vector exit folds the
lanes by the loop's own compare first and iteration order second (`==` ties, so `±0` tie, broken by
`(at - start)` unsigned) — the earliest member of the best class, NaN elements never updating, a NaN
start updating nothing, exactly as the scalar loop (`emit_arg_combine`). A non-strict `>=`/`<=`
moves the index on a tie, so the scalar answer is the LAST member of the best class: its index lanes
start at `start - 1` (an updated lane can now tie an un-updated one, and only the sentinel tells them
apart), and the fold takes the strict form's better value or, on a tie, the larger
`at - (start - 1)`, falling back to the loop's own index when no lane moved. An `else`-arm update, a
non-counter index and 64-bit values stay scalar. **FindLastIV** (`if c { last = i }`,
`RedKind::FindLast`) keeps per-lane last counters seeded at `start - 1`; the exit takes the unsigned
max of `at - (start - 1)`, where `0` means nothing was recorded (`emit_find_last_combine`). The runtime alias guard is budgeted at
eight checks per loop (LLVM's default threshold); `WUKONG_VEC_TRACE=1` prints each widening with a
per-group cost estimate (`cost=vector/scalar`, the scalar side counting `W` iterations of the
longest path). It is **enforced at its one untuned threshold**: a loop whose vector group costs more
than its `W` scalar iterations is declined (`cost_declines`; `WUKONG_VEC_COST=0` turns that off for a
same-run A/B). Over `tests/run` that declined only two loops the nested if-converter had just started
taking; a lower threshold needs a same-run release measurement first.

**Search loops** (`EarlyExit`): a loop whose body block leaves on one arm (`if a[i] > t { break }`)
before falling into a straight-line latch widens as one body split at the test. The condition goes
lane-wide and is folded over its lanes (parts combined, then a halving tree of lane shuffles, lane 0
against zero); if any lane would leave, the group branches to the vector exit with its first
iteration and starting values, and the original loop re-runs at most `W` iterations and leaves at
exactly the iteration it always did. The rest of the group — its stores and accumulates — runs only
when every lane continues. Nothing before the exit may store, gather or load an interleave window,
and every contiguous load before it must provably read memory that exists for the whole trip: a
stack array `[T; N]` with constant endpoints inside it, a `[T; N]` or static-shape tensor
**parameter** under the same constant-range test, or a `[]T` slice searched up to its own length. A
fixed-extent buffer parameter carries its extent into MIR as `ParamAttrs::extent` (scalar-leaf count
and leaf width — the unit both the slot-indexed interpreter and the byte-addressed native backend
agree on). It is the declared type's contract, and no view reinterprets a buffer: another constant
length is `E0501`, but a `?` (run-time) extent binds a fixed one unchecked, so a shorter buffer bound
that way is outside the defined contract (the language guide's rule for every run-time size) — and
there `-O2` may read up to `W - 1` elements past where `-O0` stopped. A runtime bound, or any other
pointer, keeps the search scalar.

**Predicated arms**: if-conversion runs both arms for every lane, so an access only one arm makes
cannot simply be widened. A store only one arm makes (`PredStore`) and a load no unconditional
access covers (`PredLoad`) instead run per lane through `select(lane took the arm, own element,
scratch)`, where the scratch is one private stack slot per element type in the entry block. No lane
reads or writes an element the scalar loop leaves alone — unlike a read-modify-write blend, which
could race with another thread writing the skipped elements or read past a shorter buffer. Stores
in both arms must still pair into one masked store. Only affine element accesses are predicated; an
arm that alone subscripts a `[]T` re-reads its data pointer from the fat pointer there, and that one
load is speculated and hoisted into the guard only when the fat pointer is an object live for the
whole call: a `[]T` parameter's (`ParamAttrs::slice`, now recorded on every slice parameter as a
fact of its type, `@noalias` or not) or a stack slot of the function (`speculable_ptr_load`,
`never_faults`). A pointer reached any other way may be null where no lane takes the arm —
`(*p).w[i]` through a raw `p: *mut Layer` reads `p` itself, and the scalar loop survives
`0 as *mut Layer` by never taking the arm — so such an arm declines. The same proof gates an arm's
bounds check that re-reads its length (`LenSrc::Load`).

**Nested if-conversion** (`linearize_tree`, tried when the single-diamond flattener declines): the
body is walked as a structured DAG — each conditional's two sides meet at its immediate
post-dominator, entered by exactly those two edges — so every block runs under a conjunction of
branch conditions (its path predicate) and every join's parameters are one `select` on the branch
that split it. Stores after the first conditional are sunk to the region's end and must all name one
address; what each lane's path leaves there is a `StoreTree` (the last store on the path, or
nothing), so an `if / else if / else` chain storing on every path is one vector store of nested
selects and a chain with no final `else` stores per lane to the element or the scratch slot. Arm
loads are speculated when an access on every path covers them and predicated by their path
predicate otherwise; no load may follow, on a path, a store it could read. A **bounds check inside
an arm** — `cond_br (iv + k <u len), cont, dead end` — is read as a jump: its compare and length load
leave the vector body, and the vector loop runs only when every index of its own iterations passes
(`emit_len_checks`: `0 <= start + k`, `start + n_vec + k <= len`, no wrap), so a one-armed store over
a checked `[]T` parameter widens and a buffer shorter than the trip runs the scalar loop, which
checks and aborts where it always did. A check on the body's unconditional spine — the slow copy
`bce` keeps — and any outer loop are still declined before the value analyses run. Nested guarded
reductions, stores to two addresses in one nested region, and a bounds check whose index is not the
counter plus a constant stay scalar.

**Unrolling never reassociates.** Splitting a float reduction across several accumulators would
break the 4-cycle `addss` dependency chain, and it computes a different number — which breaks both
the interp-vs-native bit-exactness gate and `-O0` ≡ `-O{1,2,3}`. So the unrolled body is the original
body, four times, in the original order, computing the original values; what it recovers is loop
overhead and the instruction-level parallelism between the independent parts of successive
iterations. The pass recognizes only the two-block counted loop the front end emits for
`for i in a..b` (header = compare + `cond_br`, body = straight line + back edge, single unconditional
preheader, positive constant step, loop-invariant bound, no call/alloca/vector-kernel in the body).
The main loop's guard is `iv < bound - 3`, and because that subtraction wraps when `bound` sits
within 3 of the bottom of its type, the preheader computes it as
`select (bound-3 < bound), bound-3, INT_MIN` — all loop-invariant, so the steady-state cost is zero,
and a wrapped limit sends every iteration to the remainder loop instead of running the body past the
end of the range. `tests/run/loop_unroll.wk` pins that guard, the `trip mod 4` remainder, and the
zero-trip case. `WUKONG_NO_UNROLL=1` disables the pass.

A **known trip count** is computed in the induction variable's own width and predicate, and only
when the final increment does not wrap (an `i8` loop stepping by 2 to `127` never ends, so its
closed-form count is not its behavior and it is treated as unknown). With it the unroller replaces a
short loop by its iterations — the header keeps its instructions as iteration 0's test, every body
copy feeds the next header copy, a final header copy produces the exit values — and gives a partially
unrolled loop no remainder loop at all: none when the factor divides the count, straight-line copies
for the leftover otherwise.

**Straight-line SLP** (`src/slp.rs`, `slp_program`) is the compiler's first vectorizer of code
that is not a loop: `W` stores in one block to consecutive elements of one object become one
`Vec(T, W)` store of one vector computation (store-seeded, as LLVM's SLP vectorizer is). It runs
after the unroller, so a complete unroll's copies — `for k in 0..8 step 2`, which the loop
vectorizer declines for its step — are visible with their counters folded, and it catches
hand-written `vec4`/quaternion/complex-pair code and small fixed-array kernels. Seeds are a block's
stores of `f32`/`f64`/`i32`/`i64`, grouped by `(base, i64 symbolic index, element type)` with a
constant offset each (`gep b, c`, `gep b, x + c`, `x - c`, `x`; an `i32` `x + c` is refused because
it can wrap), runs of consecutive offsets cut into chunks of `W = 16 / bytes(T)`. The tree is built
from the stored values in lane order: a **VECTOR** node is `W` distinct single-use results of the
block with one lane-wise shape (a `Bin` the loop vectorizer widens at `T` — no shift, no division —
an `fma`/`sqrt`/`round` of floats, a `not` of integers, a `neg`), children by operand position and
commutative operands **never swapped** (a float swap would change which NaN survives); a **LOAD**
node is `W` loads of consecutive elements (extra readers keep their scalar load); a **SPLAT** is
one value or one constant; anything else is a **BUILD** (a splat and `W - 1` lane inserts). The
untuned cost model packs a tree with a VECTOR or LOAD node whose vector side (the store, one per
VECTOR/LOAD/SPLAT node, `W` per BUILD) is strictly below its scalar side (the `W` stores, `W` per
VECTOR node, each single-use LOAD lane). Everything is emitted just before the chunk's last store
and the `W` scalar stores are deleted, so stores and LOAD loads move *down*; the soundness argument
is one crossing rule, checked with `AliasInfo::may_alias_sized` on exact widths: every unmoved store
a moved access passes, and every unmoved load a moved store passes, must be disjoint from it; a call
in between declines unless the moved access is call-proof; an `atomic.add` in between declines
unless the moved access is call-proof *and* disjoint from its location; a barrier declines; and a
chunk store that preceded a LOAD lane must be disjoint from it, because the one vector load now
runs before the one vector store. Blocks go in index order, seeds by their first store, the
function is rescanned after every pack (64 packs at most; functions above 6000 instructions are
skipped).
`WUKONG_NO_SLP` (presence) turns it off — read inside `slp_program`, so `optimize`, the verified and
timed paths and the `wukong_bench compile-work` replica all honour it — and `WUKONG_SLP_TRACE=1`
prints `slp: PACK bb<N> ty=<T> w=<W> nodes=<n> cost=<V>/<S>` or `slp: declined bb<N>: <reason>`
per decision. A gradient compile (`--emit=grad`, `--train`, `--emit=obj --grad-of`) optimizes the
forward it hands to autodiff without it: the driver passes `OptimizeOptions::AUTODIFF_INPUT` to
`optimize_preserving_with` / `optimize_verified_preserving_with`, because `wukong_autodiff`
differentiates one scalar at a time and has no VJP rule for a vector load or store — a packed store
into a buffer whose gradient is live is a hard error there. The generated backward, which nothing
differentiates, is packed as usual. It adds no MIR op and changed no backend; over `tests/run` and `examples` it packs
`tensor_dim_arith.wk` and `type_alias.wk` and costs 0.2% of optimizer time (debug `--time-passes`,
summed in process). Its vector loads and stores sit **outside loops, frame slots included**, which
`gpu-native` takes through its generic scalar-lane path — only the on-device suite (gate part 10)
proves that (`tests/run/vec5_slp_bundles.wk`).

**Call effects** (`src/effects.rs`). A user function is summarized only when calling it has no side
effect: every store targets its own frame, every call is to a summarized callee or its own SCC (the
greatest fixpoint, so recursion is fine), no barrier or shared slot or element-wise vector kernel,
and no pointer return, pointer compare or `ptrtoint` unless the pointer was handed in or names static
data. The summary records whether it reads caller-visible memory, whether it always returns (no
loop, recursion or `unreachable`), and whether it is speculatable (always returns, reads nothing, no
integer division, and no load or store outside a known in-bounds offset of its own frame — exactly
what LICM would speculate itself). External `print`/`wukong_*` symbols are never summarized. The
post-unroll cleanup (`PassManager::run_subset`) re-summarizes the whole program from its unrolled
bodies first, so a callee whose short loop was unrolled away now counts as always returning.

One splice detail generalizes to every MIR pass: `Op::VecKernelCall.kernel` is a **function-local**
index, not a `ValueId`, so value renaming neither sees nor fixes it, and `inline_call_site` must carry
the callee's recipes into the caller and rebase every copied index (see the vectorizer section).

`Mem2Reg` is the keystone: the front-end's memory traffic hides constants, common subexpressions,
and induction variables, so promoting slots to SSA is what lets the rest of the pipeline fire. It
promotes integer, float and pointer slots and leaves arrays, vectors, address-taken slots, and any
slot accessed at a width other than its own type in memory. For an integer or float slot a read
before any write becomes a zero constant, matching the interpreter's zero-initialized memory.

A **pointer** slot's zero is the null pointer, `inttoptr 0`. That used to be unsound — interpreter
slot 0 was the first live allocation of a run, so a materialized null would have aliased real memory
(the reverted `PtrToInt(p) != 0` attempt, `0900352`) — and mem2reg promoted only the
definitely-initialized subset of pointer slots. The interpreter now reserves slot 0 as a
never-allocated null at every entry point (`wukong_interp::NULL_SLOT`; a load or store through it is
the diagnostic `null pointer dereference`), so its zero-initialized pointer memory and native's
`inttoptr 0` are the same null and every pointer slot is promoted like an integer slot: a
read-before-write path gets the null, materialized once in the entry block as `inttoptr (i64 0)`.
The same reservation is what lets `mir_build` lower a pointer condition (`if p`, `while p`, `p && q`)
as the null test `ptrtoint p != 0`; it used to reach `cond_br` unchanged and fail verification with
no span. That matters for the case pointer promotion was written for — `mir_build` gives every
pointer-typed parameter (`*T`, `&T`, every `Tensor[…]`) an `alloca ptr` plus a `store <param>` and
re-loads the base pointer at every element access (`tests/run/ptr_slot_promotion.wk`) — and now also
for a pointer local whose `alloca` and initializing store are separated by a branch, which used to
keep its slot and reload it on every use (`tests/run/pointer_null_condition.wk`).

A `[]T` **slice** parameter is *not* covered: a slice is a 16-byte `{ data, len }` fat pointer passed
by address, so its base comes from `load ptr (gep <param>, 0)` — caller-owned memory, not a local
slot. Hoisting that needs loop-invariant load motion: `licm` hoists such a load when it runs on every
iteration (behind a first-iteration guard) **and** no store in the loop may alias it — and every write
through a slice goes through a pointer loaded the same way, of unknown provenance, so a slice loop
that writes anything still reloads (the vectorizer hoists those bases under its own runtime guard).
Removing the rest needs a `noalias` fact.

`Cse`/`Dse` still handle the residual memory (load forwarding and its dual) for the slots that stay
in memory. `Licm` uses the dominator
analysis to find natural loops and only hoists into loops that already have a preheader, so it is
always legal; stores, vector-kernel calls, allocas, integer division/remainder (which can trap on a
zero divisor) and calls other than to a speculatable side-effect-free callee are never moved. Float predicates are never folded on self-comparison
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

**Two distinct pointer parameters may alias — unless the source promised otherwise.** For an
ordinary parameter nothing rejects `f(a, a)`, so the analysis returns "may alias" and
`tests/run/alias_slice_params.wk` pins the behaviour that depends on it. The language decision that
closes the gap is the `@noalias` parameter attribute (C's `restrict`): `wukong_sema::noalias` rejects
the call sites and re-pointings that provably break it (`E0316`; a name a `for` head or a `match` arm
binds to an aggregate part is rooted at the place `wukong_mir_build` binds it into,
`bind_place_pattern`), `wukong_mir_build` records it as
`wukong_mir::ParamAttrs` on the MIR parameter (printed `noalias`), and `AliasInfo` then answers
"disjoint" for a promised parameter against every other parameter, gives a promised slice's elements
their own provenance (`Prov::ParamData`, demoted if the fat pointer is ever written or published), and
— when nothing derived from the parameter escapes — keeps its storage apart from untracked pointers
and callees (`promised_storage`, `is_call_proof`). The loop dependence test and the vectorizer's
runtime alias guard ask `promised_disjoint_bases`; Cranelift maps promised storage to its own alias
regions (`Heap` written, `Table` read-only) and marks an invariant promised fat-pointer load
`readonly + can_move`; the LLVM emitter writes `ptr noalias`. Every one of these facts is gated on a
promise: a function without one builds no per-parameter table and analyzes exactly as before.

`is_dereferenceable(ptr, bytes)` is a separate question that only a *speculating* transform asks: may
this access run on a path the program might not have taken? It is true for a known in-bounds
offset into an `alloca` of this function, which is live for the whole function and default-initialized
by both backends, and for an offset inside the 16-byte fat pointer of a `@noalias` slice parameter,
which the caller keeps alive for the whole call. That is what lets `Licm` hoist a loop-invariant load
out of a possibly-zero-trip loop for a local `[]T`/struct/tuple or a promised slice's `data` field
unconditionally. Through any other parameter it needs more: the load must run on every iteration, and
it is then read through `select(first_test, addr, dummy)` so the preheader never touches the
parameter's memory on an entry that would not have.

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
`Int(i128) | Float(f64) | Ptr(usize) | VecRef(usize) | Unit`.

Memory is a single flat `Vec<Value>`. Scalar MIR arrays use one typed slot per element. Source
structs and tuples lower to byte buffers, which use **one slot per layout byte**, including padding;
a field value occupies the slot at its starting byte offset. For example, an eight-byte aggregate
buffer consumes eight `Value` slots, while `[i64; 1]` consumes one. These slots are not packed host
bytes. `alloca` initializes the MIR storage with typed zeros and returns its first slot index.
`Gep` strides by `slot_count(elem)`, recursively counting array elements, including aggregate byte
buffers. The scalar memory arena is reused (`SlotReuse`): a returning frame cuts it back to its
entry length, but never below the end of the last growth that was not an alloca (a heap or
scratch block, a materialized string, a kernel-entry buffer), which is detected rather than
enumerated; `free` moves a live `alloc_*` block to a size-bucketed free list, and the next
allocation that fits it (at most twice the request) takes it back re-zeroed with its own typed
zero. A pointer that is not a live block is ignored.

A recognized-kernel operand is bounded by its **allocation**, not by the arena. Because every
allocation sits back to back in the one arena, an extent that ran off the end of one array into
the next used to pass a range check against the arena and be read (or written) as the same buffer —
a plausible answer where native reads past a real allocation. `SlotReuse` keeps a live-allocation
table (`marshal::LiveBlocks`, sorted disjoint ranges): each `alloca` is one block however its
aggregate nests, as is each `alloc_*` heap block (at the size requested, not the reused capacity),
each scratch region, each materialized static or runtime string, and each kernel-entry buffer; a
frame release, `free` and `scratch_release` drop theirs. Every marshalling helper (`kernel_values`,
`read_f32`, `output_f32`, `write_back`, `check_output`, `elementwise_live`, …), the vector-kernel
arm, and the file and text transfer arms require `[base, base + len)` to lie inside one live block
before any scratch is reserved, and report `<kernel>: operand <k> extends past its allocation`
(`k` the argument index) otherwise. An operand read in pieces at program-computed offsets, with no
trustworthy whole extent (the embedding gather's weight rows — a runtime-sized table is passed a
sentinel height), is bounded per piece by the allocation holding the operand's own pointer
(`LiveBlocks::check_within`), not by whichever allocation holds the piece; the KV append's cache is
checked whole (`[cache, cache + ncache)`) before its row is copied. An empty range is no access and
is never checked — in particular a zero-trip vector-kernel call, whose stream bases
`marshal_vec_kernel` still offsets to the range start (possibly past the arena's end), checks no
stream, as the native kernel touches none. Every public entry point runs tracked; a unit test that
builds an `Interp` by hand and assigns `memory` directly keeps an untracked table, bounded by the
arena.

SIMD vectors live in the `vecs` side arena, and vector stores scatter scalar lanes into memory.
Each call owns a suffix of that arena. Block boundaries retain register-referenced vectors and
remap their handles; returns preserve caller-owned handles or promote a newly created vector.
Errors reclaim the frame's vectors. Nested vector handles are rejected at runtime, preventing
their escape through scalar storage. This bounds SIMD retention by active frames and MIR size;
it does not establish SIMD throughput.

Calls are **stackless**: `run_function` keeps an explicit frame stack (function, register file,
block, instruction index, SIMD-arena watermark, step count), and `exec_frame` runs the top frame
until it reaches an `Op::Call` of a Wukong function or a `wukong_parallel_for` body (run as one
sequential call, whose result is `Unit`) or returns. A call pushes a frame and the loop continues; a
return pops it and resumes the caller at its pending call. No Wukong call consumes host stack, so
every public entry point runs on its caller's thread (the former per-run 512 MiB worker thread, its
spawn+join, and the `AccelPtr` hand-off are gone), and an error unwinds every frame the run entered.
A 100M-step guard bounds runaway loops *within one frame*; a call-depth counter bounds recursion:
`MAX_CALL_DEPTH` is 300_000 in every build profile (overridable per run with
`WUKONG_INTERP_MAX_DEPTH`), and exceeding it returns the diagnostic "interpreter call-depth limit
exceeded (likely unbounded recursion)". A callee symbol is resolved once per run (`CalleeTable`) to
a function, an `extern "C"`, or a dense `Intrinsic` id from the single name table in
`intrinsics.rs`, and every intrinsic call dispatches on that id; a name in none of them fails with
"call to unknown function or intrinsic". Register files are pooled per depth and not refilled on
entry — verified MIR writes a register before it reads it — and a returning frame resets only its
vector-typed registers, so no stale SIMD handle reaches the next frame's arena compaction. SIMD lane/arena allocation and several other growth paths use `try_reserve` to
report allocation errors. Some remaining paths still allocate infallibly; a diagnostic for every
possible resource-exhaustion case is not established.

`run_with_output` returns `(exit_code, stdout)`; intrinsics like `print` format into the captured
stdout buffer. `wukongc --run` uses `run_streaming` instead (and the JIT `JitProgram::run_streaming`),
which writes the same bytes through to the process's stdout as they are produced: per line on a
terminal, in 64 KiB blocks into a pipe. Every printed value is rendered by `wukong_runtime::fmt` (the
interpreter's `text_io.rs`, the JIT's `text_rt.rs` and a linked executable all call it), so a scalar
prints at its own type (`f32` shortest round-trip, `bool` as `true`/`false`, `char` as its glyph) and
`printf` / structural aggregate printing cannot differ between backends; `exit(code)` is an error
carrying `wukong_runtime::process::EXIT_SENTINEL` in the interpreter and a recorded failure that
returns through every caller on the JIT, reported as the run's exit code by both.

`f32` ops are computed in `f32` (single rounding), and an `int → f32` cast rounds straight to `f32`
(not via `f64`), so the interpreter matches native's single conversion bit-for-bit.
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
falling back to a `cc`/`$CC` C-runtime link. The rustc path is chosen when the `wukong_runtime` rlib
is reachable from the running compiler's directory — either the uplifted `libwukong_runtime.rlib`
next to it (what a plain `cargo build` leaves) or a hash-suffixed `deps/libwukong_runtime-*.rlib`
(where cargo *always* writes it, and the only copy a `cargo test` / `cargo run -p wukongc` tree has,
because cargo uplifts a library only when it is a root unit of a `build`). Both are resolved by the
one function `wukong_driver::runtime_rlib_in`, which `tests/exe.rs` also calls so its probe cannot
drift; only when it finds nothing does the driver report that path unavailable and use `cc`. The
`cc` fallback link line ends in `-lm`, because the generated C runtime always defines
`wukong_rt_fmod_*` over `fmod`/`fmodf` and glibc keeps those in libm. The rustc invocation passes
`-C panic=abort`,
because the workspace's release profile sets `panic = "abort"` and rustc would otherwise reject the
shim's default `unwind` strategy against that rlib. The generated link inputs (the Rust shim, or the C
runtime for the fallback) are written into a pid-keyed scratch directory that is deleted when the link
finishes — never into the process's working directory, unless that directory cannot be created.

**The rlib's shape decides the rest of the link line, and the release shape is not the debug one.**
The workspace `[profile.release]` sets `lto = "thin"`, so cargo compiles every *dependency* of an LTO
unit with `-C linker-plugin-lto`: the `wukong_runtime` rlib still holds a `*.rcgu.o` member, but that
member is an **LLVM bitcode module, not a COFF/ELF object**, because the final link is expected to be
the one that turns it into machine code. MSVC `link.exe` cannot read such a member — it answers
`LNK1136: invalid or corrupt file` and then reports every `wukong_*` kernel as an unresolved external
— so the AOT path had never linked a single release-profile executable, while debug linked all of
them (cargo compiles a dev-profile dependency with `-C embed-bitcode=no`, whose members are ordinary
objects). `wukong_driver::rlib_holds_bitcode_members` walks the archive's member headers and asks
whether any member *starts* with the LLVM bitcode magic — the right question, because on an MSVC
target the `lib.rmeta` member is a COFF wrapper in **both** shapes, and cargo's `ObjectAndBitcode`
shape keeps its bitcode in a `.llvmbc` section *inside* a real object. When the answer is yes,
`rustc_link` adds `-C lto=thin -C opt-level=3`, mirroring the profile the rlib was built under and
letting rustc do the LTO that the bitcode was emitted for. `opt-level` is load-bearing, not cosmetic:
the LTO backend re-codegens imported modules at the *linking* crate's level, and at rustc's default
`-C opt-level=0` the AVX2 kernels come out as `vfmadd213ps %ymm2,%ymm1,%ymm0` fed by stack reloads
instead of the release form `vfmadd231ps (%rdx,%r11,1),%ymm2,%ymm0`. The path rustc is given for `-o`
is absolutized for the same reason the LTO flag exists: with `-C lto=thin`, rustc 1.97.1 names its
per-module LTO objects from the output *stem* and hands the linker a path only valid relative to its
own temp directory, so a relative `-o hello.exe` fails with `LNK1181: cannot open input file
'hello.<crate>…rcgu.o.rcgu.o'`. Nothing here touches the debug path: a native rlib links exactly as
before, with neither flag. The cost is confined to release AOT, where the link is a whole LTO of the
runtime and std — tens of seconds per executable rather than under a second.

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
`fn` (a user `fn gelu` wins and the loop stays scalar). The integer-or-float `min`/`max` builtins
(`wukong_mir_build/src/minmax.rs`, a compare in the result's signedness + `select`) are deliberately
not in that set yet, so a body calling them stays scalar. `@parallel` per-thread chunks go through the
same vectorizer, so they run SIMD × cores.

The vectorizer streams over every buffer form a kernel is written over: a `[T; N]` array, a
contiguous `Tensor[T, …]` of constant **or symbolic** extents, and a `[]T` slice. Each stream's data
pointer resolves through `kernel_base_ptr`, the rule every recognizer already uses (an `Array`-bound
name *is* its storage; a `Ptr`-bound tensor and a slice's fat pointer hold it one load away). The two
runtime-sized forms bring two facts only the run time knows, and `vector_guarded_end` tests both once,
ahead of the loop, as loop versioning: every slice access must stay inside the slice's length over the
whole range, and every written stream must be disjoint from every other stream of the body whenever
either one is runtime-addressed. If either test fails the vector part covers nothing and the scalar
remainder — the loop exactly as written — does all the work. The overlap test is the run-time alias
check gcc and LLVM version a loop on; it is needed on top of the whole-program `provably_disjoint`
fact because that fact assumes every caller is visible, which an `--emit=obj` export, a harness
entering through a raw pointer, or one of the call-borrowed views `alias.rs` lists as holes all
violate. A pair of two `[T; N]` arrays keeps the static fact alone. Two lane-type rules keep the
vector lanes equal to the scalar tail: integer lanes compare **signed** in both vector forms (AVX2 has
only `vpcmpgt*`), so an ordered compare of unsigned operands keeps the loop scalar; and a compound
assign vectorizes only where its one lane-wise op exists (`/` on a float lane, the bitwise ops on an
integer lane, never `%` or a shift).

Cranelift's CLIF vector ISA still caps at 128-bit — a 256-bit `f32x8` SSA value is rejected at
`define_function` (the `cranelift_still_rejects_f32x8` / `p4_probe_vec256_ops` tripwires keep that
documented). So for an eligible body the vectorizer captures the loop as a flat, backend-agnostic
`VecKernel` recipe and, when register pressure fits, emits one `Op::VecKernelCall`: the Cranelift
backend assembles that recipe to **true 256-bit AVX2 machine code** via `iced-x86`
(`wukong_codegen_cranelift::avx2`, VEX-encoded), while the interpreter marshals the *same* recipe
lane-wise — so the two stay bit-identical (elementwise lanes carry no reassociation, so the gate
holds element-for-element). A recipe has one lane type (`VecLane`): **f32** (8 lanes, the `*ps`
family), **f64** (4 lanes, `*pd`), and the wrapping integer lanes **i32** (8 lanes,
`vpaddd`/`vpsubd`/`vpmulld`) and **i64** (4 lanes, `vpaddq`/`vpsubq` — AVX2 has no 64-bit lane
multiply, so `VecKernel::pressure` refuses one and that body keeps the 128-bit strips). The reference
evaluators (`eval_lane`, `eval_reduction`) are generic over the matching Rust type, so each lane type
has exactly the arithmetic its instruction family performs. A kernel keeps up to eleven stream base
pointers in GPRs (the volatile r9/r10/r11/rdx plus the callee-saved rsi/rdi/rbx/r12–r15 it pushes),
so the five-plus-stream bodies of fused epilogues stay 256-bit. An elementwise kernel whose
output-only streams are all 32-byte aligned and whose streamed traffic reaches 10 MiB (the runtime's
`nt_policy` default) stores them with `vmovnt*` and `sfence`s before returning — the read-for-ownership
a cacheable store pays on a stream larger than the caches is the largest lever on a bandwidth-bound
loop; the choice is made inside the kernel at run time and only picks which store instruction writes
the identical values (`WUKONG_P4_NO_NT=1` assembles only the cacheable form). There is deliberately no
software prefetch: the hardware prefetchers already track unit-stride streams, and `gcc -O3` emits none
either. A stack slot of 32 bytes or more has a 32-byte-aligned base (`align_shift` caps at 32 there,
16 below), so the base of a local array never splits a cache line under a 256-bit access. A request
alone would not do it: Cranelift aligns a slot's *offset* within a frame whose base is only 16-byte
aligned and never realigns the stack, so such a slot reserves 16 bytes of slack and its address is
rounded up at run time (`(base + 16) & !31`); `stack_slot_alignment_*` probe both stack parities and
fail without the round-up. Interior addresses stay unaligned, so `mem_flags` still never sets
`aligned`. A literal trip shorter than one
register group keeps the inlined strips (the kernel could only be called with `n = 0`); a runtime
trip always takes the kernel. `Op::VecKernelCall.kernel` is an index into the *owning* function's
`vec_kernels` recipe table, not a `ValueId`, so a MIR pass that moves instructions between functions
must copy the callee's recipes into the caller and shift every copied index (`wukong_opt::inline` does
exactly that; without it a spliced call names one of the caller's own recipes). The verifier
range-checks the index and requires the call's result to match the recipe kind: a reduction recipe
yields its lane type, an elementwise one yields nothing.

That assembler path is host-gated: `wukong_mir::host_supports_vec_kernels()` requires
`target_arch = "x86_64"` **and** `target_os = "windows"` **and** runtime AVX2 + FMA3, because the
emitted bytes read their three arguments out of the Win64 registers rcx/rdx/r8 literally and there is
no dispatch inside a kernel. The predicate lives in `wukong_mir` — below both users — because two
crates must answer it identically: `avx2::host_supports_kernels()` is a thin wrapper over it and makes
`assemble_kernel` refuse before it encodes anything, and **`wukong_mir_build` calls it as part of the
same condition as `WUKONG_P4_NO_256`**, so off-gate no recipe is built and the loop keeps the 128-bit
CLIF strips. Only the second half of that was true originally, and the consequence was that on every
non-Windows host `mir_build` attached a recipe `populate_module` could not assemble and the *entire*
compile failed with `avx2 assemble vec kernel …` — a hard error where a documented, result-identical
fallback already existed. `assemble_kernel`'s `Err` is now a backstop: reaching it means the gating
disagreed with the emitter, which is a bug rather than a platform outcome. The kernels are assembled
while the module is populated — the same code path for the JIT and for `--emit=obj` — so an emitted
object is host-ISA- and host-ABI-specific.
`WUKONG_P4_NO_256` forces the portable 128-bit path — a same-run A/B knob and a kill-switch; it
changes instruction selection and never a result, reductions included (below).

Two refinements target the dominant ML arithmetic. A float `x + y*z` **contracts to a fused
multiply-add** (`Op::Fma`, one rounding, a hardware `vfmadd`) in both the scalar and vector lowering
paths; the interpreter mirrors it with `mul_add`, so the two backends stay bit-identical. And a
**reduction** `for k in .. { s = s + x[k]*y[k] }` (or `s += ..`, a running `fmax`/`fmin`, or any `+`
chain that holds `s` once — `s = s + x + y` is the reduction `s += x + y`) over an f32, f64, i32 or
i64 accumulator is recognised and lowered to independent lane accumulators (an FMA chain when the
addend is a product), a cleanup loop, a horizontal fold, and a scalar remainder.

### The canonical reduction tree

A vectorized float reduction regroups its addends, and the grouping decides the low bits of the
answer. That grouping is fixed **once, as part of the language**, and every lowering of the loop
computes it: the 256-bit kernel, the portable 128-bit strips, the interpreter, every `-O` level, every
host, a literal trip or a runtime one, any addend. With `L` lanes per 256-bit register (8 for f32/i32,
4 for f64/i64) and `U = 4` (`VEC_UNROLL`):

1. `U` accumulators of `L` lanes, initialised to the fold identity (`+0.0`, `-∞` for max, `+∞` for min).
2. Main loop: element `i + u·L + l` of each `U·L` block folds into accumulator `u`, lane `l`.
3. Cleanup: every remaining full `L`-element group folds into accumulator 0.
4. Accumulators `1..U` fold into accumulator 0 lane-wise, in order `1, 2, 3`.
5. Lanes `0..L` of accumulator 0 fold left to right into one value `r`.
6. If at least one full group ran, the scalar seed `s` becomes `fold(s, r)`; otherwise the seed is
   untouched (a `-0.0` or NaN seed is never disturbed by the identity).
7. The `< L` remaining elements fold into `s` one at a time, in index order.

`fold` is `+` — a single-rounded `fma(x, y, acc)` when the addend is a product `x*y` — or the x86
operand-order `max`/`min` (`acc > x ? acc : x`, which matches `vmaxps` on ties and NaN). Integer sums
wrap and are exact under any grouping; they use the same tree anyway. The reference implementation is
`VecKernel::eval_reduction` with its module docs in `wukong_mir::vec_kernel`; the portable form
represents each logical accumulator as a pair of 128-bit vectors (`emit_reduction_portable`,
`finish_portable_reduction`), and `tests/run/reduction_canonical_tree.wk` pins the tree with
hand-evaluated values at sizes that exercise every step.

What is deliberately **not** an input to the tree: the host ISA or OS (`host_supports_vec_kernels`),
`WUKONG_P4_NO_256`, the lowering option that keeps recipes away from the PTX backend, whether the
trip is a literal, and the addend's register pressure. The last used to pick the 256-bit kernel's
accumulator count, so the same `.wk` source summed differently on a Linux CI box, under the kill
switch, or after an innocuous edit to the addend. A body too large for `U` accumulators beside one bank
of its own registers now simply does not take the kernel (`VecKernel::reduction_fits`); one that fits
beside the accumulators but not `U` times over shares register banks between the unrolled copies
(`reduction_banks`) — that changes the schedule, never the grouping.

Which form runs is a performance choice only: the raw-AVX2 kernel when the recipe fits and the trip
reaches `VEC256_REDUCTION_MIN_TRIP` (2048 elements, the measured crossover of the out-of-line call
against the inlined form for the highest-pressure body), the portable form otherwise. A literal trip
decides it statically; a runtime trip — the `fn k(a: []f32, n: i64)` shape every real kernel has —
carries both forms behind one run-time compare, since both compute the same tree. Two cases fall
outside the tree, both by construction: when `vector_guarded_end` sends a loop scalar (an
out-of-bounds slice access or an overlap, i.e. a program whose scalar semantics the vectorized loop
must not change) the whole loop is the strict left fold the source spells; and a loop the front end
does not vectorize (a `while`, or a body it declines) reaches `wukong_opt::vectorize`, which never
reassociates a float accumulate. Recognized kernels (`wukong_sreduce_*` and friends) define their own
documented reduction order, chosen by the program's spelling and never by the host. Reductions
vectorize only on the sequential path, never the `@parallel` one (folding into a shared accumulator
across threads would race). The tree is a build contract, not a promise of strict source-order
summation or stability across compiler releases.

The contraction rule is exact and syntactic (`FnLowerer::try_contract_fma`, mirrored by `vec_try_fma`
and the 256-bit recipe's `Add` arm): only a binary `+` with a `*` operand contracts; `x - y*z`, a
product bound by `let`, and a compound `s += y*z` outside a vectorized reduction do not; and when both
operands are products the **left** one is fused (`a*b + c*d` → `fma(a, b, round(c*d))`). The language
guide's "Floating-point semantics" section is the user-facing statement and
`tests/run/fp_contract_rules.wk` pins each clause; `fp_contract_tests.rs` pins the MIR shape.

**Strict floating point** (`@strict_fp`, `--fp=strict` → `LowerOptions::strict_fp`) is decided
entirely here. `strict_fp_gemm` stamps `GemmSyms::strict_fp` and clears `GemmSyms::recognize` for the
function (and its monomorphized instances and outlined `@parallel` regions, which inherit the struct).
The four contraction sites return early on the flag, and `try_vectorize_for` drops a float reduction
before `emit_reduction`, so the loop lowers as its sequential scalar fold; integer reductions and
elementwise loops still vectorize. No other crate needs to know: `wukong_opt` never reassociates (its
vectorizer folds reduction lanes in index order, its unroller keeps one accumulator) and never
creates an `Op::Fma`; Cranelift lowers `fadd`/`fmul` without fusion; the LLVM emitter writes no
fast-math flags. That is also why the mode survives inlining in both directions — a strict body's
separately rounded ops and a fast body's `Fma`s are already fixed in MIR when the inliner moves them.
The driver refuses `--fp=strict` with `--emit=grad`/`--train`/`--emit=obj --grad-of` for the reason it refuses
`WUKONG_NO_RECOGNIZE`: the VJP rules are keyed on recognized kernels.

## Textual LLVM IR (`wukong_codegen_llvm`)

`--emit=llvm-ir` prints a module that means what the interpreter means: unordered float `!=`, the
guarded division/shift contracts, `f16`/`bf16` in `float` registers rounded at the same boundaries,
booleans as bytes in memory. CI executes only regression modules (`verify_semantics.py`,
`verify_math_specials.py`, under `llvmlite`), so what the corpus's modules promise is checked
structurally:

- **Target.** It names the host `target datalayout` and `target triple` (`host_target`): the LLVM
  15–17 x86-64 layout strings, which LLVM 18+ upgrade on read, so one text serves CI's LLVM 15
  `llvmlite` and a current `clang`.
- **Link ABI.** Every `wukong_*` runtime symbol is declared from its `#[no_mangle] extern "C"`
  definition — `src/runtime_abi.rs`, generated from `wukong_runtime`'s sources and checked equal to
  them by `runtime_abi_table_is_the_runtime_source` (`WUKONG_UPDATE_RUNTIME_ABI=1` regenerates it) —
  and each call converts its arguments to the declared slots as the Cranelift call does (integers
  sign-extended to `i64`, a literal zero to `ptr null`, a narrower result truncated after). Inferring a
  declaration from call sites made a pointer slot `i64` in any module whose calls all passed a null
  bias. `link_abi_law_holds_on_the_run_corpus` lowers every run fixture at `-O0` and `-O2` and requires
  each `wukong_*` declaration to equal its definition (print/assert against the `--emit=exe` link
  shim) and every symbol to have one signature across its `define`, `declare` and calls. With opaque
  pointers LLVM's verifier checks neither: the general vectorizer's f64/i32/i64 recipes were defined in
  `float` and called as `double`/`i64`, and the corpus verified clean.
- **Alignment.** A vector load or store carries `align <lane bytes>` — LLVM otherwise assumes the whole
  vector's alignment, and the vectorizer accesses `<4 x float>` at every element offset, so executing
  a module faulted in `movaps` — and a stack slot carries the native backend's alignment, which is
  what makes an `[N x i8]` aggregate's naturally aligned field loads true.
- **bf16 narrowing** is `wukong_runtime::f32_to_bf16_bits` spelled in integer IR (`narrow_half`), as
  the native backend spells it in CLIF: `fptrunc … to bfloat` is a `__truncsfbf2` libcall no link
  this project performs provides. `semantic_tests::bf16_narrowing_modules` exports a scalar and a
  vector module that `verify_semantics.py` executes against the transcribed reference on 4000+ bit
  patterns (ties, exponent carries, infinities, signalling and quiet NaN payloads).
- **Declines.** A construct with no faithful spelling is refused with `C0001` and a tagged message
  (`LLVM emission [<tag>]: …`); `DECLINED` lists every tag with its reason and
  `every_decline_is_catalogued` triggers each one. Nothing in the run corpus or the examples declines;
  `crates/wukongc/tests/emit.rs` holds that count at a ratchet of zero, and
  `tools/verify_llvm_corpus.py` reports declines separately from verifier failures.

## Spelling canonicalization before recognition

Every recognizer below matches a *syntactic* shape on the raw AST, inside `lower_program` — before
`wukong_opt` has run mem2reg, CSE or any canonicalization of its own. That made recognition brittle
in a way that has nothing to do with the arithmetic. Hoisting a row base into a local
(`let ib = i*256;` then `a[ib + p]`) is plain common-subexpression elimination — it changes no
floating-point result and every optimizer performs it anyway — but it moves the index arithmetic
*out* of the index expression, so `match_row_col` no longer saw `i*N + p` and the whole GEMM fell
back to a scalar nest. A dimension written as a bare module `const` was worse: every stride/bound
comparison succeeded *symbolically* (`Dim::Var(N)` on both sides) and the nest then declined at
`dim_value`, which resolves a `Dim::Var` by looking the name up in the function's **locals**.

`wukong_mir_build::canon` closes that gap with a pure source-to-source AST rewrite run once at the
top of `lower_program`, before anything else looks at the tree:

- **Index-local forward substitution.** A `let` bound to a pure integer expression — integer
  literals, integer variables and `+ - *` — whose every use is in index position is inlined at
  those uses, and the now-dead `let` is deleted. The deletion matters as much as the substitution: a
  leftover statement in a loop body is by itself enough to make a whole-nest matcher (which requires
  the body to be exactly the inner `for`) decline.
- **Integer const folding.** A module-level `const` of integer scalar type whose initializer is a
  plain integer literal is inlined at its uses. This is exactly what `FnLowerer`'s `Path` arm already
  does at lowering time — the def map records a const's type, not its value — so the emitted MIR is
  unchanged and only the recognizers, which run earlier, see a difference.
- **Accumulator-seed sinking.** Several recognizers match a *window* of consecutive statements and
  index it by position — softmax is seven statements, log-softmax six, the cross-entropy forward
  four plus a store. Declaring the sum accumulator one statement early splits the window, and the
  whole row-op then dispatches nothing. So a `let mut x = <literal>;` is moved **down** to sit
  immediately before the first statement that mentions `x`, provided that statement *assigns* it.
  The initializer must be a bare literal (it reads no memory, calls nothing, cannot trap), every
  statement it passes must mention the name nowhere at all — not read, written, `&`-taken or
  re-declared — and it never leaves its block, so it still runs exactly as often.
  The `mut`-and-assigned condition confines the rewrite to an accumulator seed and is load-bearing:
  a move always lands *between* two statements, and the block-level fusions match adjacent pairs, so
  sinking an immutable constant between an int8 GEMM and its dequant epilogue cost that pair its
  fused `wukong_i8gemm_nt_deq`. An immutable binding is a value definition and stays put.

All three rewrites are value-identical at *every node*, which is what lets the moved subtree keep its
`NodeId`s: `sema.types` answers for it exactly as before, and `SemaResult::defs` is keyed by name
rather than by `NodeId`, so a duplicated id is read-only aliasing and never a collision. No new sema
entries are minted.

Legality is proved over the **region** a binding dominates — the statements after its `let` in the
same block, plus that block's tail. Over that region neither the binding nor any free variable of
its initializer may be re-declared, assigned or `&`-taken (which is what rejects `let b = t*4;` in a
loop that later does `t = t + 1`), and every use site's recorded type must **equal** the
initializer's (which is what rejects a narrowing annotation like `let ib: i32 = i*256;` with
`i: i64` silently becoming full-width arithmetic). `/` and `%` are excluded from the pure set
because they can trap and a trap must not be re-sited under a branch that may not be taken; calls,
indexing and field access are excluded because their value depends on memory a later statement may
have written. (`defer` never reaches this pass: `defer.rs` copies every deferred expression onto
its block's exits as the first normalization of `lower_program_with`.) A local or parameter that
shadows a const keeps that const unfolded in that function. The pass answers "nothing to do" from a
read-only feasibility scan, so a program it cannot help never pays for the AST copy.

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

The store itself does not have to be the bare `c[i*N+j] = s`. Three shapes an engineer writes
naturally — with the follow-up work folded *into* the store instead of trailing as its own loop —
are peeled apart by the recognizer and lower to exactly what the factored spelling lowers to:

| store | lowers to |
|---|---|
| `c[i*N+j] = act(bias[j] + s)`, or the bias-free `act(s)` | one `wukong_sgemm_nt_epi` carrying that act code |
| `c[i*N+j] = x[i*N+j] + s`, `x` a *different* array | `wukong_sgemm_nt` (β = 0) + a flat `wukong_velem_f32` residual sweep |
| `c[i*N+j] = s;` **and** `d[i*N+j] = d[i*N+j] * s` in the same body | `wukong_sgemm_nt` + `wukong_velem_f32` (Hadamard) |

The last two are two-kernel lowerings, and they are bit-identical to the split source because the
matmul store writes exactly `s` and an f32 store/load round-trip is exact. All three are NT-only,
offset-free and mutually exclusive with a peeled α (`wukong_sgemm_nt_epi` pins its `alpha` to 1.0),
and the two-kernel forms additionally require the second buffer to be distinct from `a`, `b` and `c`
— the split writes all of C before any of D, so an overlap would be a different program. The
*in-place* residual `c[i*N+j] = act(c[i*N+j] + s + bias[j])` stays the strictly better single-call
form: one `wukong_sgemm_nt_epi` with β = 1.

Each buffer operand is resolved through the shared `kernel_base_ptr`: a fixed `[T; N]` array's slot *is*
its storage and is passed as-is, while a `Tensor[..]`/pointer operand and a `[]T` slice both keep their
base in the slot and are loaded first (a slice's data pointer is the first word of its fat pointer).
Slice-ness is keyed off the **sema** type, recorded by `bind_slice` at every binding site — never off
the MIR slot shape, which cannot distinguish `[]T` from a tuple, a 16-byte struct or a user's
`[u8; 16]`. That is what makes `[]T` slices legal operands across the whole recognized-kernel family
(the only way to hold a runtime-sized weight blob).

An operand may also live in a **struct field** — `l.wq[j*D + p]`, how a real model groups its
weights. Recognizers name their operands by a bare `Symbol`, so `lower_program` interns a synthetic
`"<local>.<field>"` name (a text no identifier can equal) for every `local.field[…]` in the module,
and `FnLowerer` records each struct local/parameter's *buffer* fields — their addresses inside the
struct's byte buffer — as it binds them. `kernel_base_ptr` consults that table when the plain scope
lookup misses, then applies the identical rule above, so a slice field yields its data pointer and a
fixed-array field its storage address. The table is deliberately **not** in the scope chain: a
synthetic name is invisible to every non-kernel path, which therefore declines instead of mistaking
a field's address for its data. A record is revalidated against the current binding of its base
name, so an inner `let l: *mut Layer` that shadows the struct cannot resolve to the outer one's
field. Nested (`l.inner.w`), tuple-field (`t.0`) and through-a-pointer projections are not recorded
and lower scalar.

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

Four refinements (2026-10, the P9 runtime findings) remove the remaining non-compute work, none of
which changes a bit:

- **Single-sweep B packing.** When a macro-block's packed B panel would be swept by exactly one row
  block — the serial kernel at `m ≤ MC` and every 2D block-parallel block — `macro_kernel_packb`
  packs each `KC×NR` micropanel into a 24 KB L1-resident buffer immediately before the row sweep
  that consumes it, and threads a `PfWalk` through that sweep's FMA loop (one `PREFETCHT1` per four
  K steps) so the *next* micropanel's source is in L2 when it is packed. The whole-panel pack stays
  for `m > MC`, where every extra `ic` pass re-reads it. `WUKONG_GEMM_FUSED_PACK=0` restores the
  whole-panel pack everywhere (an A/B instrument; the default is unmeasured at landing).
- **Edge tiles.** A tile with fewer than 6 valid rows runs a row-trimmed K loop (only the live
  rows' FMAs), and every partial tile — and every fused-epilogue partial tile — writes back with the
  full-tile path's own 256-bit sequence under a `vmaskmovps` column mask instead of a scalar
  element loop.
- **`Aᵀ·B` without a copy.** The kernel takes an `at` flag: A stored `[k, m]` is packed straight
  from its storage (each `[kc][MR]` panel row is six *consecutive* stored floats), so the TN
  weight-gradient entries no longer allocate, zero and transpose an `m·k` scratch matrix.
- **Half inputs.** The bf16/f16 entries widen into a reused per-thread scratch (no per-call
  allocation) with 8-lane bodies (`vpslld` for bf16, F16C for f16), and the TN form widens A in its
  stored layout and dispatches with `at`.

The int8 kernel (`wukong_runtime::i8gemm`) needs no repacking — both operands are contiguous over
the contraction and `vpdpbusd` consumes 32-byte runs directly — but it does need blocking: column
blocks of 1 MiB of B (`i8_col_block`) are walked outermost so every A row streams past an
L2-resident block, and the `_parallel` entries split C into a (row block × column block) task grid
(`i8_par_grid`) rather than one task per row, which also lets a single-token decode GEMM use the
whole pool.

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

### What a recognized matmul computes — the GEMM numeric contract

`interp == native` holds for a recognized matmul because both backends call one function, which is
a property of the *implementation*; this section is the property of the *result* that the tests pin
independently of that.

- **Order.** For each `(i, j)` independently, K is split by `select_kc(k)` into
  `nb = ceil(k / kc)` equal-ish blocks of at most `KC = 384` steps (a multiple of 4). Each block
  accumulates ascending `p` from `+0.0` with **one rounding per step** (`vfmadd`; `f32::mul_add` in
  the scalar twin) and is folded into C with one more rounding (`beta_eff = 1.0` after the first
  block); a non-zero caller `beta` is a flag that makes the first block accumulate too (never a
  scale). The fused epilogue `act(alpha·x + bias[j])` is applied once, after the final block, and is
  bit-identical to the unfused sequence.
- **Invariant under** `MC`, `NC` (now host-derived, see below), the 6×16 register tile, packing (whole-panel or single-sweep,
  row-major or transposed A), edge-tile handling, the AVX-512 twin, every parallel schedule and
  thread count, every `WUKONG_GEMM_*` knob, and the AVX2-versus-portable choice: a host without
  AVX2+FMA — and every non-x86 target — runs `sgemm_scalar`, which follows the same order with a
  correctly-rounded `mul_add`. Pinned by `sgemm_scalar_matches_avx2_bit_for_bit` (every layout,
  `beta`, epilogue and 1–3 K-blocks), the serial==parallel tests of every schedule,
  `fused_b_pack_is_bit_identical_to_the_classic_pack` and
  `tn_direct_pack_matches_nn_on_the_transpose_in_every_schedule`.
- **A function of `KC`.** Retuning the K-block cap changes the low bits of every recognized matmul,
  so `select_kc_partition_is_pinned` pins the partition by value. Like the reduction partition
  above, the order is a build contract: stable across hosts, thread counts and backends, **not**
  promised across releases.
- **Error bound.** Every reference test asserts
  `|got − exact| ≤ γ_r · (Σ_p |a_ip|·|b_pj| + |β·c_ij|)` against an f64 reference, with
  `r = kc + nb − 1 (+1 when β ≠ 0)` and `γ_r = r·u/(1 − r·u)`, `u = 2⁻²⁴` — netlib's Level-3 BLAS
  test scaling (`SMMCH`) with this order's own rounding count (Higham, Lemma 3.1). The test shape
  lists are derived from `MR/NR/MC/KC/NC` and `assert_blocking_coverage` fails if a retune leaves a
  blocking regime (multiple K-blocks, a second MC or NC block, the L2-capped column split, full and
  edge tiles) unreached.
- **Half inputs** widen exactly and then follow the f32 contract; **int8** is exact modulo 2³²
  (wrapping `i32` addition is associative, so blocking and threading cannot change a bit).
- **Unrecognized matmuls** (declined, `@generic`, `WUKONG_NO_RECOGNIZE`) compute the loop as written
  under the FMA-contraction and reduction-partition rules above. The two coincide when K is below one
  vector group (pinned with fused-vs-unfused-distinguishing data by
  `tests/run/lang_generic_gemm_fma_order.wk`); beyond that they may differ in the last bits.
  `tests/run/gemm_blocked_every_lane.wk` uses integer data (exact in any order) to check all three
  f32 kernel forms lane by lane across three K-blocks and full and edge tiles end to end.

**Cache blocking follows the host (P13.4).** The five-loop's A-panel height `mc` and B-panel width
`nc` are resolved once per process (`gemm::gemm_blocking`): `WUKONG_GEMM_MC` / `WUKONG_GEMM_NC` when
set to a positive multiple of `MR = 6` / `NR = 16` (at most 65536; anything else is ignored), else the
measured `MC = 144` / `NC = 4080` scaled by the probed L2 / L3 against the 2 MiB / 24 MiB part they
were measured on -- the panel keeps its measured share of its cache level, rounded down to the
`MR`/`NR` grid and clamped to within 4x of the measurement, so the derivation is the identity on the
reference part. It is licensed only by `mc_nc_blocking_never_changes_a_bit` (the serial kernel under
blockings from one micro-panel to 4x the reference, every pack schedule, both transposes, both
`beta` rules, a bias+ReLU epilogue) and `legacy_parallel_nc_never_changes_a_bit` (the fork-join and
persistent-region schedules). `KC` and the reduction `RCHUNK` fix a summation order and stay
constants (P13.9).

### Turning recognition off: `WUKONG_NO_RECOGNIZE` (an instrument, not a feature)

The interpreter/native bit-for-bit agreement is also why recognizers are **gate-blind**. A recognizer
fires identically on the interpreter and on native at every `-O` level — both call the same
`wukong_runtime` symbol — so `native_matches_interpreter` and the opt-invariance suites pass whether
the kernel fired or not, and whether it was right or not. Nothing observable distinguishes "the
compiler lowered this loop well" from "a pattern matcher swapped in a hand-written kernel."

Setting `WUKONG_NO_RECOGNIZE` (to any value) builds the program with the recognizers off:
`wukong_driver::compile` turns it into `wukong_mir_build::LowerOptions { recognize: false }`, which
skips all **four** places a recognizer is probed from, so every loop nest lowers as written:

| Cascade head | What it claims |
|---|---|
| `lower_program`, per item | Whole-function interceptions (`matmul_fn`, the `@parallel` kernel probes) |
| `FnLowerer::lower_for` | The ordered per-loop `match_*` cascade (~40 arms), plus `try_emit_argreduce` |
| `FnLowerer::lower_block` | Statement **windows**: GEMM + epilogue loop, the multi-pass softmax / LayerNorm / RMSNorm / L2-norm idiom (including the seed-then-normalize-in-place spelling, whose leading `for c { out[c] = x[c] }` is folded into the norm's source operand rather than emitted as its own pass), a flat 1-D scan's `let acc` + `for` |
| `try_vectorize_for` / `try_vectorize_ranged` | The elementwise probes that open the vectorizer (vmath, velem, vhorner, dequant, the low-precision axpby twins) |

What stays on is everything that compiles the loop the user actually wrote: the autovectorizer,
`try_fuse_run` (fusing adjacent same-range loops emits vector MIR, not a kernel call), `@parallel`
outlining (`wukong_parallel_for` is the user asking for threads, not a guess at intent), `tindex` and
`canon` normalization, and the whole optimizer. So do the **explicit builtins** — `alloc_*`/`free`,
`read_*`/`write_*`, `now_ns`, `sdpa` all still lower to their `wukong_*` runtime entry points, because
the user named them; declining those would be a compile error, not a generic lowering.

The witness, and the shape of any claim about the language:

```sh
cargo run -q -p wukongc -- --emit=mir -O2 tests/run/tensor_matmul.wk       # call wukong_sgemm(...)
$env:WUKONG_NO_RECOGNIZE='1'
cargo run -q -p wukongc -- --emit=mir -O2 tests/run/tensor_matmul.wk       # no kernel call at all
```

Two rules about it:

- **It is an instrument, not a user feature**, which is why it is an environment variable in the
  style of `WUKONG_GPU_NO_MEGA` and not a CLI flag (nothing in `USAGE` mentions it). The supported
  surface for "lower the loop I wrote" is the `@generic` attribute, on a function or on a single
  `for`/`while` loop (`wukong_mir_build::dispatch`).
- **The per-site view is `--remarks=dispatch`** (`wukong_mir_build::lower_program_reporting_dispatch`):
  a `note` at each statement `lower_block` lowered into a recognized `wukong_*` call (the innermost
  one; an enclosing statement that only contains it is skipped), and at a function's name when a
  whole-function interception replaced it. It observes the builder, never steers it, so the MIR is
  byte-identical with it on or off.
- **Autodiff refuses to run under it**, by name: `--train` and `--emit=grad` exit 1 as soon as the
  variable is set, because `wukong_autodiff`'s VJP rules are keyed on the recognized kernel calls. A
  named refusal, never a silently zero gradient.

`crates/wukongc/tests/recognize.rs` gates all of this: one witness fixture per cascade head (its
kernel is dispatched by default and gone with the variable set — the first half matters, or the
second is vacuous), no `call wukong_` at all survives for `tensor_matmul.wk`, the program still prints
the same bytes on both backends at `-O0` and `-O2`, and autodiff is refused. Corpus-wide,
`every_recognizer_dependent_program_agrees_with_its_generic_lowering` takes every `tests/run`
program whose MIR loses a `wukong_*` call with the variable set (a count floored by
`the_recognizer_dependent_corpus_holds_its_floor`) and runs it with recognition on and off at `-O2`
on the interpreter, `-O2` on native and `-O0` on the interpreter: same exit code, and stdout
byte-identical unless the fixture is listed in `REASSOCIATION_TOLERANCES` with a derived relative
bound. A listed fixture's recognition-off stdout must then also satisfy the fixture's own
`EXPECT-OUT*` lines (`judge_on_off`), so the flat on/off bound — sized for its loosest line — never
loosens a tighter lane's contract on the one column no other gate executes. (The 361 fixtures
present when the switch landed were also swept on/off by hand; that sweep is history, this test is
the gate.)

### Aliased buffers: the kernel only where the operands are disjoint

Nothing in the language rejects `mm(x, y, x)`, and a recognized kernel equals the loop it replaces
only when its operands do not overlap in a way the kernel cannot see: `wukong_sgemm` writes `C` in
tiled order while it still reads `A`, an 8-lane row scan loads lanes before the source loop's
stores would have changed them. An aliased call therefore means **what the source loop means**, and
the compiler keeps the kernel only where the two agree. This is the blind spot that is worse than
"gate-blind": on an aliased input the two backends need not run the same program at all (the
interpreter used to answer from copies of the operands, native from the overlapping memory), and the
interpreter's answer was the plausible one, so a differential failure would have blamed native.

**Lowering.** Every multi-buffer recognizer asks one classifier,
`wukong_mir_build::alias::plan_kernel_alias`, about the operand pairs its kernel's contract
constrains. Two different symbols in a function the whole-program alias facts vouch for are
`Disjoint` (dispatch); the same symbol is `MustAlias` (dispatch only under an exact-in-place
contract, otherwise lower the loop); anything else is `MayAlias`, and `kernel_guard.rs` versions the
dispatch at run time — `ptrtoint(base)` and `ptrtoint(gep(base, n))` range tests over every
constrained operand pick the kernel when they are disjoint (or exactly in place where permitted) and
the ORIGINAL source, recognition and `@parallel` off, otherwise. That is LLVM's `LoopVersioning`.
The extent end is a `Gep`, so the interpreter (slot stride) and Cranelift (byte stride) reach the
same verdict for in-bounds extents, which matters because a kernel and its loop are not always
bit-identical; "exactly in place" is only ever admitted between operands of one element type, the
one comparison two address units could disagree on.

The versioning contexts cover every cascade: the per-loop recognizers (`emit_versioned_loop`, and
`emit_nonempty_reduction` for the reductions, whose empty-extent arm lowers the same source), the
256-bit elementwise kernel probes inside `try_vectorize_for` (vmath, vmath2, velem, vhorner, the int
dequant — their overlapping arm is the loop with recognition off, which the generic vectorizer then
lowers only with its own disjointness proof, or scalar), and the statement windows
(`emit_versioned_stmts`: the fused norms, the copy-then-norm fold, the flat 1-D scans — their
overlapping arm is the window's own statements). The whole-function GEMM and int8 interceptions
(`lower_matmul_fn`, `lower_i8matmul_fn`) go through `emit_nonempty_reduction_loop` too, so they version the
same way with the function's one loop on the overlapping arm. A recognizer reached with no context — a
probe, an outlined `@parallel` chunk — declines on `MayAlias`, as every recognizer did before
versioning. A coefficient an elementwise kernel HOISTS (`s[0]`, `*p` in `out[j] = s[0] * x[j]`) is a
scalar the range test cannot see, so it is checked at compile time instead (`hoisted_reads_disjoint`):
a read that may come from a buffer the kernel writes keeps the loop. An elementwise dispatch that becomes several full-range passes
(one per statement of a fused body) is guarded over every operand of every pass, because loop
fission is itself an alias-sensitive transformation.

**Empty ranges.** A recognized reduction keeps what the source loop writes for a zero or negative
extent: the storing reductions dispatch only on a positive reduction extent and lower the source
otherwise (`emit_nonempty_reduction`), and the value-returning ones (`wukong_sreduce_f32_parallel`,
the bf16/f16 dot/sum/reduce) select the SEED on an empty range (`keep_seed_when_empty`) — the kernel's
identity combined into the seed is not the seed on every input (`-0.0 + 0.0` is `+0.0`;
`fmax(NaN, -inf)` as the `Cmp(Fogt)+Select` fold is `-inf`). A `Select`, not a branch, so a
straight-line loss stays straight-line for autodiff's tape.

**Oracle.** The interpreter still marshals operands through copies, and `marshal::check_output`
makes that unobservable rather than assumed: an arm reached with operands its kernel's contract
forbids returns an error instead of a copied answer. Exact in-place operation, where permitted, is
equal to out of place for a row-local or elementwise kernel. Three groups run against LIVE memory in
source order instead, because their one meaning under any overlap is the source loop's: the
embedding gather, the scatter-add, and — for a SHIFTED overlap — the elementwise maps (vmath, vmath2,
velem, vhorner: `marshal::elementwise_live`, one element at a time). A call that reaches a kernel
with operands the compiler's guard would have kept away (a direct call, an alias-analysis hole) is
therefore either refused or answered with the source loop's value, never with a copy-in/copy-out
value no backend computes. Operand scratch is recycled per thread (`marshal::f32_scratch` /
`recycle_f32`), so a kernel inside a loop no longer allocates its operands on every iteration, and
every extent is range-checked before any scratch is reserved.

**The aliasing ABI**, one class per exported kernel (each entry's `# Safety` carries an
`**Aliasing:**` line that says which):

| Class | Families |
|---|---|
| All operands disjoint | GEMM (f32, fused epilogue, alpha, residual, bf16/f16, int8, int8+dequant), GEMV, GEVM, transpose, pool2d, column reductions, row/column arg-reductions, log-sum-exp, entropy, KL divergence, soft-label and hard-label cross-entropy forward, LayerNorm backward, attention, embedding, scatter-add, the int dequants, the half-input/f32-output maps (`vmath_{bf16,f16}`, `axpby_{bf16,f16}`) |
| Exactly in place, otherwise disjoint | cumsum, cummax/cummin, cumprod, lrscan (`out` over an input), RoPE and its backward (`out` over `x`; the frequency table disjoint), softmax backward and RMSNorm backward (`dx` over `dy`; `y`/`x`/`gamma` disjoint), cross-entropy backward (`dx` over `x`), velem, vmath, vmath2, vhorner, log-softmax, the fused norms (`out` over `x`; `gamma`/`beta` disjoint), broadcast bias (`out` over `x`; the bias row disjoint), the half-output maps |
| None constrained (read-only) | `sreduce`, `argreduce`, the bf16/f16 dot/sum/reduce |

What remains open is stated in `docs/roadmap.md`: `alias.rs`'s documented holes can still hand a
kernel overlapping storage where the interpreter refuses (cross-element) or answers with the source
loop (elementwise); an outlined `@parallel` chunk has no versioning context and declines rather than
versions; and the `@parallel` region outliner, the generic vectorizer's loop fusion (`try_fuse_run`)
and the half-output runtime kernels' own shifted-overlap handling are outside this classifier.

## CPU training execution

`--train` validates concrete dense f32 buffer types and checked byte extents before allocating
training storage. **The loop is one function, `wukong_runtime::train::run_training`**: the seeded
init, the snapshot/restore, the clip, the schedule, SGD and the AdamW hyperparameter buffer, driven
by a `TrainSpec` (buffer sizes, weights, restored buffers, steps, rate, optimizer, seed, decay,
clip, warmup, schedule), with every kernel reached through a selector — 0 the forward, 1 the
backward, `2 + k` the AdamW kernel for the `k`-th distinct weight size (`TrainSpec::kernel_buffers`
lists each selector's buffers, and the driver builds its kernel list from it). `format_report`
renders the printed text. The driver (`build_training` in `wukong_driver/src/lib.rs`) builds the
forward, backward and AdamW kernels and the spec; in process it encodes the spec, decodes those
bytes and runs the loop through `training_native::TrainingExecutor`. The interpreter remains the
default; `--backend=native` appends one verified `(Ptr, I64) -> Void` dispatcher
(`training_native::add_dispatcher`) and JIT-compiles it once. Its owning `JitBufferKernel` retains
executable memory across steps. Each call rebuilds the pointer table from exclusively borrowed
buffers, since optimizer updates can replace their allocations. What a kernel prints is forwarded
to stdout as each call returns — also when the call fails at run time: `run_kernel_f32_with_output`
and `JitBufferKernel::run` return the captured bytes beside the error, so the bytes the executable
has already written before a failed `assert` are printed before `--train` reports it.

**`--train --emit=exe|obj`** (`emit_train_artifact`) builds the same kernels and the same
dispatcher into the artifact, plus the spec as a `.rodata` static (`TrainSpec::encode`: a
`WKTRAIN\0` magic, a `u32` layout version, then every field little-endian) and a synthesized MIR
`main` — `global_addr` of the blob, its length, `func_addr` of the dispatcher, `call
wukong_rt_train_main` (an `extern "C"` import) — verified by `verify_or_ice` and code-generated at
the training program's `-O1` floor through the ordinary `emit_native`. The runtime entry decodes the
blob (a damaged one is `error: --train: invalid training spec`, exit 1), runs `run_training` with a
dispatch that calls the compiled dispatcher, and prints `format_report`; so the executable's stdout
is the in-process run's by construction. Only the rustc link provides the entry: on the C fallback
the driver refuses by name (exit 2). The AdamW hyperparameter layout exists twice
(`wukong_autodiff::optim::hp` builds the kernel, `wukong_runtime::train::hp` fills the buffer); a
`const` assertion in the driver makes a drift a build error.

The loop keeps one buffer vector for the whole run — the forward's buffers followed by one gradient
per trainable weight — and updates it in place: the AdamW kernel's five operands are moved in and
back out, and SGD writes the weights directly. Per-step copies are needed only to keep a step from
seeing the previous step's forward writes, so the driver asks `wukong_autodiff::written_params` which
parameters the (flattened) forward may write — a store through it, a recognized kernel's *output*
operand, any call the tape does not recognize, any escape — and snapshots and restores just those
around each gradient run (the loss output is reset instead). The training-file plumbing lives in
`wukong_driver/src/training.rs`: `--train-init` reads a whole buffer, `--train-data` seeks to batch
`step mod batches` and decodes it into the live buffer through one reused byte buffer, and
`--train-save` writes after the run, all raw little-endian f32; `run_training_with` takes the
initial buffers and a per-update `load` hook for them. The learning-rate schedule (`lr_at`: linear
warmup, then constant or `CosineAnnealingLR`) and the global-norm clip (`clip_global_norm`, the norm
accumulated in f64) live with the loop in `wukong_runtime::train`.

Native calls share the JIT run lock, output capture, assertion state and scoped large-stack worker.
The public buffer-kernel invocation is unsafe: callers must establish storage type, extent,
alignment, aliasing and lifetime; validating MIR and the dispatcher signature alone cannot do so.
The trainer supplies these obligations for its f32 buffers. General helper-call differentiation
and measured training performance remain separate gaps.

## CPU streaming-store policy

The bias, dequantization, low-precision, vector-elementwise and vector-math kernels share
`crates/wukong_runtime/src/nt_policy.rs`. Their existing saturated byte-traffic formulas
select non-temporal stores once the traffic reaches a threshold derived from the host: three
quarters of one probed L3 instance, clamped to `[4 MiB, 64 MiB]` (glibc's `memcpy` rule for the
same decision), with the historical **10 MiB** as the fallback when no L3 size can be probed. The
cache size comes from `crates/wukong_runtime/src/topology.rs` (CPUID leaf 4 / `0x8000_001D`, Linux
sysfs as the fallback; `WUKONG_CACHE_L3_KB` overrides it). The derivation is a rule, not a
measured optimum for the current host.

Set `WUKONG_NT_MIN_BYTES` before starting the process to override the threshold in bytes.
It accepts a positive ASCII decimal integer that fits `usize`, including leading zeroes.
Absent, non-Unicode, empty, signed, whitespace-containing, zero and overflowing values
are ignored and the derived default applies. The first use caches the value for the process
lifetime. Either way the choice is throughput-only: an NT store writes the same bits.

### Parallel dispatch policy

Every `_parallel` runtime entry asks `crates/wukong_runtime/src/par.rs` whether to fork: at least
two independent units (rows, column stripes, channels), weighted work (rows x cols x a per-element
cost: 1 stream, 4 row-normalization, 16 transcendental) of at least `WUKONG_PAR_MIN_WORK` (default
2^18, not yet measured per family), and a pool with a second worker. Below that it tail-calls its
serial twin, which is bit-identical by the crate's serial == parallel law. Under `cfg(test)` the
threshold is zero, so the crate's own serial == parallel tests always fork; the compiler-level
differential gates run the production default, so set `WUKONG_PAR_MIN_WORK=1` to push their small
shapes through the multicore paths. `par::claim_chunks` is the shared atomic claim queue behind
`wukong_parallel_for`, the int8 GEMM (whose claimed units are the cells of its `i8_par_grid`
row block x column block grid, and whose fork gate counts (row x 4-column) cells rather than rows,
so a single-token decode GEMM still forks) and per-channel dequant. `@parallel(grain = G)` is applied in
`wukong_mir_build` (`par_grain.rs`): the call passes `ceil(n / G)` chunk indices and the outlined
body maps its chunk span back to iterations, so the runtime ABI is unchanged.

**How a `_parallel` entry forks (P9.15).** No kernel forks with a bare `into_par_iter()`. Row,
column-stripe, channel and fixed-chunk maps go through `par::for_each_unit` (on the caller's pool --
the global pool for a top-level call, the enclosing pool inside a region or a GEMM pack phase) or
`par::for_each_row` (on the unified kernel pool); the fixed-`RCHUNK` reductions build their partials
with `par::map_units`, which stores partial `c` in slot `c` whatever ran it, and fold them in index
order. `WUKONG_KERNEL_DYN=1` (read once, default OFF) swaps all three from rayon's adaptive split to
the atomic claim queue on the same pool, so the A/B compares schedules alone; OFF runs exactly the
split the entries spelled before. The GEMM's compute grids keep their own `WUKONG_GEMM_DYN`.
`WUKONG_KERNEL_DYN=1 cargo test -p wukong_runtime --lib` runs every serial == parallel test under
the claim queue.

**The implicit grain (P12.21).** A `@parallel` loop with no `grain` argument gets
`G = max(1, ceil(MIN_CHUNK_OPS / body_ops))`, `MIN_CHUNK_OPS = 2^14`, where `body_ops` is the static
MIR instruction count of its outlined body built at grain 1 (`par_grain::implicit_grain`). A body
with an inner loop (a second back edge) or any call keeps `G = 1`: its static size is not its
per-iteration cost, and a head-parallel region of GEMM calls must not collapse into one chunk. So an
8-iteration straight-line region becomes a one-chunk call that `wukong_parallel_for` runs without a
fork, and a long one is claimed in chunks of `G`. It is `LowerOptions::implicit_grain`, off for
library callers and set by `wukong_driver` for every backend except `--backend=gpu-native`, whose
PTX lowering maps one index to one thread. An explicit `@parallel(grain = G)`, `G = 1` included,
always wins. Scheduling only, decided before the optimizer: `-O0` == `-O2` and the compile stays a
pure function of its input (`tests/run/parallel_implicit_grain.wk` prints every lane of both
outliner forms; its `--emit=mir -O2` shows each `wukong_parallel_for` trip operand as `1`).

### Debug contract checks on the C entries (P14.5)

Every exported pointer-taking `extern "C" fn wukong_*` in `wukong_runtime` opens with
`crate::safety::debug_*` calls that check, in debug builds only, what its `# Safety` section
documents: each nonempty extent's pointer is non-null and its span fits `isize`
(`debug_extent`, `debug_extent_2d`, `debug_extent_bytes` for dtype-coded `*u8` buffers,
`debug_extent_opt` for an optional operand such as a norm's `gamma` or a GEMM bias), and each
aliasing rule holds -- `debug_disjoint` for "all operands disjoint", `debug_inplace_or_disjoint` for
"exactly in place, otherwise disjoint". A failure panics at the entry's line (the helpers are
`#[track_caller]`), which aborts the process because a panic cannot unwind out of an `extern "C"`
function: in a debug build that is the alternative to undefined behaviour. The checks compile out of
release builds. An entry whose pointer has no extent -- a NUL-terminated path or site string,
`wukong_parallel_for`'s opaque capture table, the handle `wukong_rt_free` takes -- carries
`// SAFETY-EXEMPT: <reason>` instead, and the eight half-precision entries whose kernels detect and
define overlap themselves (`wukong_axpby_{bf16,f16}`, `wukong_vmath_{bf16,f16}` and their narrowing
`_out` forms, each pinned by an overlap test) check extents only. `safety::entry_law` scans every
module's source (`include_str!`) and fails on an entry with neither a check nor an exemption, on a
`src/*.rs` file missing from its list, and on an entry whose doc block states an aliasing class
(its `**Aliasing` paragraph) without calling that class's helper -- the eight half-precision
entries are its exact, named allowlist. Every check is a no-op for a non-positive extent, and an
entry that takes a shape wraps its checks in `if cfg!(debug_assertions) && <every dimension> > 0`
(GEMM's `k` included), its own no-op condition negated: a derived extent is not enough, since two
negative dimensions multiply to a positive span and a one-dimension extent stays positive while
the other dimension is empty. A call the entry treats as a no-op therefore never asserts, whatever
pointers it carries (`safety::tests::no_op_calls_never_assert`). The debug corpus
(`cargo test -p wukongc --test run`) runs green with every check armed.

Tests that require streaming execution derive their sizes from the actual policy, retain
existing numerical cases/tail residues, and count executed non-temporal stores. Their
allocation guard rejects thresholds above 64 MiB with an explicit failure; production
accepts every positive `usize` value. Debug builds check 32-byte alignment immediately
before each 256-bit streaming store. This does not establish bounds safety or a speedup.

## GPU backend

> **This section describes structure, not performance — and structure is now device-parameterized.**
> No figure here is a benchmark; for those see `BENCHMARKS.md`, every GPU number in which is an
> **RTX 4050 Laptop** (`sm_89`, 20 SMs, Windows/WDDM) number and must not be generalized. What
> matters architecturally is that the backend no longer *assumes* that card: PTX module headers come
> from one authority (`wukong_codegen_gpu::ptx_target`) at the **lowest legal floor per instruction
> family** — `sm_80` for the Ampere-legal majority, `sm_89` only where `e4m3`/`e5m2` genuinely
> exist, and `sm_90a` for the wgmma/TMA family, which is architecture-*locked* rather than floored
> and is therefore gated by an `Sm90aLicense` type a launcher cannot skip — while **device identity
> is probed once** into `Gpu::target()` (compute capability, SM count, opt-in shared-memory budget,
> L2, VRAM, driver) instead of being hardcoded. The two directions are deliberately opposite and
> both correct: **peers compile for the device, modules tag the floor.** Caches are keyed by device
> accordingly (cubin SASS by arch, autotune by arch + SM count), because a cross-arch cubin load can
> succeed silently under forward-minor compatibility. The retarget's plan and status are
> `GPU_RETARGET_PLAN.md`.

`wukong_codegen_gpu` (behind `--features gpu`) is a third execution path: being a compiler, it
**emits PTX text** and **driver-JIT-loads it via `cudarc`**, so no `nvcc`/CUDA toolkit is needed — only
the NVIDIA driver. (That is a statement about what *Wukong* requires, not about the wider world: on a
box that does have the toolkit the available peers get stronger, and this backend's job is to meet
the stronger bar. Read "toolkit-free" as **toolkit-optional**.)

Module loading goes through a persistent **cubin cache**: `Gpu::load_module_cached`
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

The `sgemm_nt` hook carries a **second dispatch axis, the device architecture**. `gpu_accel`'s
`gemm_route_for(cc, off)` is a pure function of the probed `GpuTarget`: on a Hopper part
(`cc_major == 9`) a recognized `C = A·Bᵀ` goes to the warpgroup-MMA + TMA family
(`gpu::gemm_nt_wgmma` over a config from the one seam function `wgmma_cfg_for`, which delegates to
`ptx_wgmma`'s shipped rule); on every other capability it takes the pre-Hopper launcher,
unchanged. The rule is `== 9` and not `>= (9,0)` for the same reason the module header is: `sm_90a`
is architecture-*locked*, so a Blackwell part must take the other path rather than be handed a module
it cannot load.

**The shipped rule is `ptx_wgmma::wgmma_dispatch_dtype`, a pure function of `(M, N, K)` and the
probed SM count**, and it chooses four things: the CTA tile (the widest of 128×256 / 128×128 / 128×64
whose wave fill clears 0.90 against the probed SM count), the grouped raster (only when a multi-wave
grid's linear wave footprint blows L2), persistence (any multi-wave grid), and the 1×2×1
B-multicast cluster (only when what the multicast would save off the *final* configuration's binding
roof clears its break-even **and** the grid is at least `CLUSTER_MIN_WAVES = 2` waves deep, so the
cluster's cross-SM coupling has more than one device pass to amortize over — at the shipped value of
2 that second term coincides exactly with the persistence lever, and it becomes independent only at 3
or more). **Only the SM count is probed**: `L2_BYTES`, `BW_L2` and `BW_HBM` are constants pinned to
the H100 SXM5 this campaign rents, and that asymmetry is recorded beside `L2_BYTES` rather than
papered over. Until 2026-08-12 the seam instead delegated to a
static two-way split on `M × N` at 8e6 output elements, which could only ever return a 128×256 row;
the H100 visit of that day measured the four schedule levers as sweep arms and found the split
shipping the losing configuration on four of seven benched shapes, which is what promoted the
dispatcher into the product path. `M × N` is retired rather than kept as a fallback because it is not
a physical quantity for any of the four levers — two shapes with identical `M × N` take opposite
cluster verdicts in both directions.

The rule can also **decline**: it names a module or says it cannot, never substitutes a nearby one,
and here a decline is a routing fact that takes the pre-Hopper path with the reason logged. **That
narrows the Hopper fast path, and the narrowing is the promotion's one honest cost.** The retired rule
could not decline — it returned a 128×256 row for every shape — while the tile lever makes
`(tile, schedule)` combinations reachable that no module carries: a schedule lever at a tile whose
table holds only the one-tile-per-CTA row, plus the un-clustered rastered widest tile. Six such
signatures exist, they are contiguous bands of ordinary tall-skinny shapes rather than pathologies,
and shapes inside them now run the pre-Hopper f32 launcher — correct, and much slower.
`ptx_wgmma::tests::the_rules_reachable_configurations_are_enumerated_and_the_decline_set_is_pinned`
sweeps the shape space, buckets every verdict by its `(tile, cluster, raster, persist)` signature, and
pins **both** sets exactly, so the band cannot grow silently and closing it is an argued edit rather
than a drift. It is not closed here because the missing modules would be **unmeasured PTX shipped on
the strength of arithmetic**, which is the opposite of what a promotion is for;
`docs/gpu/derive/WAVE3_PROMOTION_HANDOFF.md` carries the list a later lane needs.

Every other reason the family cannot take a call *that is decidable before the
launch* — a shape whose tensor map is unencodable (`K % 8 != 0` makes the NT row stride not a
multiple of 16 bytes), an output past the epilogue's `u32` element index, a ring larger than the
device's opt-in shared memory, a CTA grid past CUDA's `gridDim.y` ceiling of 65535 (see below), or an
`UNSUPPORTED` from the generator — is likewise a
**decline to the existing path**, never an error, and the launcher's own `assert!`s are pre-decided in
the driver so a user's program can never abort on one. Two outcomes are deliberately *not* declines, and the module
names both rather than implying a total guarantee: a driver rejection **at** the launch that is not
`Unsupported` (a `cuModuleLoadData` refusing an `sm_90a` module on a driver that predates that
virtual architecture, say) stays a `Some(Err(..))` hard error, because a GPU failure under
`--backend=gpu` is never silently answered on the CPU; and a launch that never retires ends the
process with `exit(70)` from the launch wait in `wukong_codegen_gpu::gpu`, since a wedged context
blocks every later driver call anyway. Ragged M/N/K are *not* declines: TMA zero-fills and the
epilogue predicates.

**The `gridDim.y` ceiling is now unreachable through the shipped rule, and is retained as defence in
depth rather than as a live routing path.** `LaunchPlan::grid` puts the M tiles on `y` for a
one-tile-per-CTA schedule, and that schedule is selected iff the grid is one wave, i.e. iff
`tiles <= sm_count` — so `gridDim.y = ceil(M/bm) <= sm_count`, about 132. A persistent schedule's grid
is `(cluster slots, cluster.y, 1)`, bounded by the device. No shape the rule can serve, at either
dtype, can therefore present `gridDim.y > 65535`. The decline stays because `LaunchPlan::grid` is
`pub`, because the tile menu and the schedule set are both things later waves widen, and because what
it prevents is a `CUDA_ERROR_INVALID_VALUE` on a call that had a working fallback; its unit probe
accordingly feeds the launch *plan* an `M` the rule would never route to it.

An **odd `N`** is the one decline that is both live and not the launcher's own: every arm the shipped rule can select carries the
`st.global.v2.f32` epilogue, whose 8-byte pair is aligned only when `N` is even, and a misaligned
store is a *sticky* `CUDA_ERROR_MISALIGNED_ADDRESS` that fails every later call in the process
rather than returning a wrong number. `gemm_nt_wgmma` asserts it only in its timing sibling, so the
driver decides it before the call. One decline reads the **data** rather than the shape, and it cuts
f16's range at *both* ends, because the seam converts both f32 operands with `half::f16::from_f32`
and f16 spans only 6.104e-5 to 65504 against f32's 1.18e-38 to 3.4e38. Past the top, an operand
enters the GEMM as an infinity and comes back `inf`/`NaN`; under the bottom, a **row** whose *whole*
magnitude range is beneath f16's smallest normal enters as subnormals or exact zeros and returns that
entire row (A) or column (B) of `C` as flushed lanes (at ~1e-8 every element is under f16's
round-to-zero point `2^-25`, and at ~1e-7 the one or two surviving bits carry 20–100% relative error)
— where the f32 launcher returns the finite number the CPU oracle returns. Both are different
*answers*, not wider error bars, and no tolerance band covers either; worse, `diff::assert_close`
passes a lane on `abs <= abs_tol || rel <= rel_tol`, so under the wgmma band's `abs_tol = 5e-2` a
zeroed small-scale output passes at 100% relative error. So both decline, at the cost of one
read-only pass over each operand (the same order as the host-side conversion the launcher performs
immediately afterwards). The two rules are deliberately **not symmetric**: overflow is per element,
because one `inf` contaminates every output lane its row or column touches; underflow is per **row**
on that row's maximum (`0 < max|A[i][·]| < f16::MIN_POSITIVE` for any `i`), because
`C[i][j] = Σ_k A[i][k]·B[j][k]` makes a row the unit that feeds an output lane — a flushed element
costs only its own contribution to sums its own row's larger elements also feed, while a wholly
flushed row costs the whole lane. The granularity is load-bearing: a single peak folded over the
whole operand cannot see a quiet row inside a loud one (A with one row of 1.0s and one of 1e-8s has
peak 1.0 and returns half of `C` as zeros), while a per-*element* rule would fire by chance on an
ordinary `U(-1,1)` buffer and decline nearly every real GEMM — a row of one cannot, at
`(6.1e-5)^K ≈ 1e-34` for `K = 8`. An exactly-zero row converts exactly and is not declined. The
residual, stated rather than hidden: inside a row whose peak *is* normal, far smaller lanes keep only
f16 subnormal precision, so a result dominated by those lanes is the shape this backend's GEMM band
does not promise to cover. The route
changes arithmetic class (the wgmma family converts both
operands to f16 on the host and accumulates in f32), which is inside the CPU↔GPU tolerance contract
but is not invisible — the driver's device gate sizes its band from the route that **actually ran**.
The distinction matters because a decline is silent by design: `gemm_route_for(cc, off)` never sees a
shape, so on a Hopper part it answers `Wgmma` for every call including the declined ones, and
`GpuAccel::calls` is bumped identically by both arms. So the seam keeps a witness — per-route
counters `gemm_wgmma_calls` / `gemm_existing_calls`, read by `gemm_route_taken()` — the band is
chosen from *that*, and the Hopper gate asserts every offload took the route it claims to measure.
Without it, a run in which `gemm_nt_wgmma` declined at launch (an unencodable tensor map, a
`wgmma_module` `UNSUPPORTED`) measures `gpu::gemm_nt` and reports it as wgmma, with the f32 result
comfortably inside an f16 band. `WUKONG_GPU_NO_WGMMA=1` forces the pre-Hopper path in the same
binary, for a one-build A/B; `WUKONG_GPU_ROUTE_LOG=1` prints the route and the decline reason for
every dispatch.

Everything on that path *up to* the launch — the route, the config the seam selects, every decline,
and the `LaunchPlan` the config yields — is a pure function of the shape and the probed capability,
so only the launch itself needs Hopper. The end-to-end gate
(`gpu_backend_linear_routes_through_wgmma_on_hopper`) is a plain `#[test]` behind a capability skip
rather than an `#[ignore]`, because it is the only proof anywhere that the family actually *executed*
and so has to run under an entrypoint that can fail: `#[ignore]`d tests are reachable only from the
Modal harness's `::bench`, which appends `--ignored`, and `::bench` used to discard libtest's exit
status entirely — a failed witness assertion completed as a green `modal run`. The invocation is
pinned in `gpu_accel::HOPPER_GATE_INVOCATION` (a `::test --filter` line, under that entrypoint's
`sys.exit`), and a device-free law checks that it selects a test which exists, is not `#[ignore]`d,
and still compares the route counters — plus that both Modal entrypoints can still exit non-zero,
which means the guarding condition **names every `rc` they collect from a cargo run**, not merely
that the string `sys.exit` occurs somewhere in the function. The third way that round could go green
having exercised nothing is the gate **skipping itself**: it capability-skips off Hopper, so on any
non-Hopper container it printed `[skip]` and libtest reported `ok`, with nothing comparing the probed
device against the requested SKU (and `WK_GPU` defaults to `L40S`). `::test` therefore exports
`WUKONG_GPU_REQUIRE_CC` — the capability the round *declared it rented*, from its own SKU table — and
the gate escalates its capability skip to a failure when that declares a part the wgmma route takes
while the device is not one. It cannot be `WUKONG_GPU_REQUIRED`: that is set on every device run,
including the routine `WK_GPU=L4` suite, where skipping a Hopper-only gate is correct. The law also
forbids the invocation acquiring `--no-driver`, which would select zero driver tests and exit 0.
Its shape list
lives in `gpu_accel::HOPPER_GATE_SHAPES` where a device-free law asserts what those shapes are chosen
for: none declines under the H100 opt-in SMEM budget, each grid is a multiple of its cluster on every
axis, and the four of them between them reach **every launch mechanism the shipped rule can select** —
the narrow 128×64 tile with its own `m64n64k16` mainloop, the one-tile-per-CTA schedule, the persistent
tile loop, and the clustered launch (with one shape carrying the clustered *and* persistent
combination, which is the one that can hang). The coverage is asserted against the rule itself rather
than against any threshold, and that is not tidiness — it is the repair of a claim that had already
rotted. The list previously justified its clustered shape by `M * N` crossing a size threshold, and
when the 2026-08-12 promotion replaced that threshold with the per-shape dispatcher the same shape
started taking the un-clustered persistent arm, leaving the `.reqnctapercluster` launch untested from
the driver side while the doc comment still claimed otherwise. Each mechanism matters for a different
reason: the clustered row compiles `.reqnctapercluster` into the entry and a launch that does not
match a compiled cluster requirement *fails* rather than returning a wrong number; the persistent row
resets `%kt` per tile, and a missed reset makes tile 2 accumulate into tile 1 as a plausible exact
integer.

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

**Both of this backend's declines are legible, and they are legible in different shapes because they
are different things.** A megakernel decline is not an error — the single-thread path is the
correctness reference every megakernel run is compared against — so it is a note, printed only when
asked for: `WUKONG_GPU_EXPLAIN=1` (or the older `WUKONG_GPU_DUMP_PTX`) makes `megakernel::decline`
print one line naming `fusion::MegaPlan`'s reason and the fallback. Making it unconditional would
change the behaviour of the *successful* path on the majority of gpu-native runs, since only a
minority of corpus configs are eligible. An **unlaunchable frame** is the opposite: the single-thread
lowering homes the entry's allocas in a per-thread `.local __frame`, and CUDA caps local memory at
512 KiB per thread — a documented ISA constant (CUDA C Programming Guide, *Technical Specifications
per Compute Capability*), not a `CU_DEVICE_ATTRIBUTE_*`, so unlike SM count, L2 or the opt-in SMEM
budget it cannot be probed into `GpuTarget`. `ptxas` compiles a larger frame and the module loads;
only the launch fails, and it fails with a bare `CUDA_ERROR_INVALID_VALUE`. `jit_run_single`
therefore checks the **worst call path** against `lower::MAX_LOCAL_FRAME_BYTES` and returns
`UNSUPPORTED:` before touching the driver. The path is the quantity, not the entry's own frame: CUDA
charges the budget along the path, so two functions each under budget can be unlaunchable together,
and that shape is reachable at the CLI's default `-O0` (at `-O2` the inliner folds them into one
frame and an entry-only test would fire). `worst_call_path_frame` takes the maximum over paths, not
the sum over reachable functions — two sibling callees are never live at once — and declines to
answer for a recursive graph, where depth is a run-time property; those programs keep the entry-only
test. The check sits at the **run** seam and not in `emit_ptx` on purpose: the lowering is correct, it
is the launch that is impossible, and moving it into the emitter would make the device-free coverage
sweeps stop measuring what this backend can lower.

The **megakernel is not exempt**, and the earlier text here said it was. The mega *entry*'s allocas
are homed in the shared `.global` frame buffer, so its size is unbounded by this cap — but
`translate_mega_callee` gives every callee an ordinary per-thread `.local __frame`, charged exactly
like the single-thread path. `megakernel::try_run` checks the worst callee path and declines to the
single-thread lowering, which reports the budget with the full path. Without it a `@parallel` program
with a large per-iteration local died on a bare `gpu-mega launch failed` — with less diagnosis than
the single-thread path gives, and reached by following the single-thread decline's own advice.

`annotate_local_frames` covers what no static walk can. It names an over-budget function only if the
launch can reach it (a never-called one used to explain unrelated faults with its size), and when
nothing is statically over budget it still explains a `CUDA_ERROR_INVALID_VALUE`: `ptxas` charges
register **spills** to the same per-thread cap, measured here at 1,280–1,480 bytes for an entry with
256 live accumulators, against a 1,024-byte reserve. That stays an annotation and never a decline —
the program may be a few hundred bytes from running — and it never reclassifies, so a genuine fault
stays a fault for the corpus gate.

Two things the backend used to leave on the floor now reach the device. **Vectorized MIR** is lowered
by scalarizing each `<N x T>` value into one register per lane — `Splat` aliases a scalar across the
lanes, `Iota` materializes lane *k* as the constant *k*, `ExtractLane` copies one lane register back
out as a scalar, and a vector block param (a loop-carried SIMD accumulator) gets its lane registers
pre-assigned so the edge that writes it can copy lane by lane. The other vectorizer output,
`Op::VecKernelCall`, stays a decline: it is a raw-AVX2 Win64 microkernel recipe with no PTX
analogue, and it is also why the corpus sweeps normalize their MIR (see the coverage note below).
**`wukong_parallel_for`** — the `@parallel` outliner's runtime call — lowers in mega mode as a
grid-stride loop: iteration *i* runs on the thread with flat id `i mod (blockDim·gridDim)` as the
one-iteration chunk `F(i, i+1, ctx)`, with a uniform trip count and an empty range on the tail so the
loop stays convergent and the call stays `call.uni`. Per-iteration chunks rather than one contiguous
range per thread is a coalescing choice: adjacent lanes then touch adjacent elements. Any partition
is legal because `wukong_mir_build` only emits the op after proving the iterations independent, which
is the same fact that lets the CPU runtime pick chunk boundaries from the core count; what that proof
does *not* cover is side effects, so `fusion::analyze` declines an outlined body containing
`print`/`println` (one record per thread instead of one per iteration) while allowing `assert`, whose
flag store is idempotent. On the **single-thread** path `wukong_parallel_for` keeps its sequential
`F(0, n, ctx)` chunk — that kernel is launched with one thread, so there is nothing to hand the
iterations to, and it stays the reference every mega run is compared against. Both lowerings need
`-O1` or better in practice, not because of the ops: at `-O0` the entry's loop counters still live in
allocas, so a loop condition branches on a `Load` and the data-independent-control-flow gate declines
the program before either op is considered. And the grid-stride form is reached only when the
`@parallel` region is in the **entry** function: eligibility rejects an entry that calls a user
function, so the common `@parallel fn saxpy(..)` called from `main` still runs the sequential chunk.
Lifting that means giving callees the mega thread discipline (their stores would need the `tid == 0`
guard the entry's have), which is a design change rather than another arm.

A **cooperative multi-CTA launch path** exists beside that single-block form, because "the whole
program in one kernel" on a one-CTA grid is a statement about one SM rather than about the machine —
which matters little on a 20-SM laptop and makes the result meaningless on a 132-SM part. The launch
layer therefore has three pieces the block-scoped path does not use: `grid_barrier_ptx`, a
sense-reversing grid-wide arrival counter in `.global` with device-scope fences (a `bar.sync` is a
CTA rendezvous and says nothing about the other CTAs), whose state is a launch *parameter* rather
than a module-scope global so a kernel that faults mid-barrier cannot hang the next program;
`plan_grid`, which derives the grid from `cuOccupancyMaxActiveBlocksPerMultiprocessor` × the probed
`GpuTarget::sm_count` — never a literal — and returns `Ok(None)` when the request would not be
simultaneously resident, because a cooperative grid that is not co-resident does not run slowly, it
hangs; and `launch_mega`, which goes through `cuLaunchCooperativeKernel`, the only launch API that
guarantees co-residency. A kernel opts in through its own **ABI**, read back out of the emitted PTX
by `mega_abi`: a block-scoped entry declares `(p_ctx, p_frame)` and is pinned to one CTA, while a
grid-parallel entry declares `(p_ctx, p_frame, p_gbar)` *and* defines `mrt_grid_barrier`. Half of
either shape is a hard error rather than a decline, because the two halves fail in opposite and
equally silent ways — a barrier no launch made resident deadlocks, and a multi-CTA launch of
`tid == 0`-guarded stores duplicates every `print` and races the shared frame. `lower::emit_mega_ptx`
still emits the block-scoped form, so every corpus program is launched exactly as before.
**What that costs is now measured** (`bench/lang/`, 2026-08-22, RTX 4050): the best language-path loop
reaches 2.7-3.0 GB/s, **1.4-1.6 % of the part's ~192 GB/s**, on one CTA of the 80 this device can hold
resident -- and gpu-native is 6.6-13.3x *slower* than the Cranelift CPU backend on the same programs.
Emitting a grid-parallel entry is therefore the largest single lever on this backend, ahead of any
load-width or pipelining work.

The CPU↔GPU boundary is **tolerance-gated** (`c·√K·ε`) rather than bit-exact — the GPU analogue of the
CPU differential oracle — and the path stays optimization-invariant (`-O0` ≡ `-O3`). Nearly every
module is behind the feature, so a plain `cargo test` compiles an almost-empty crate and the
toolchain-free core is unaffected. A short list is deliberately **un-gated**, on one consistent rule:
*whatever is pure host-side data or arithmetic gates in the default build, and only the launch layer
inside it is `#[cfg(feature = "gpu")]`.* Today that is `ptx_target` (the header-floor authority),
`paged_kv` and `paged_attention` (the KV block allocator and geometry, the paged-decode PTX
generators, the int8 quantizer, the f64 reference), `bench_instrument` (the statistics, spec table,
`nvidia-smi` parser, provenance formatting and publish gate), `tma_host` (the `cuTensorMap` argument
bundle and its preconditions — only the one `cuTensorMapEncodeTiled` call is gated), and `ptx_wgmma`
(whose `sm_90a` capability check is an `Sm90aLicense` *type*, so an ungated emitter fails to compile
rather than merely failing a textual scan). A new `src/*.rs` must also be added to `CRATE_SOURCES` in
`gpu.rs`, the corpus every textual law scans — a file outside it is invisible to the ASCII, header-
floor and `.version` laws. The flip side of the feature gate is that `cargo test` does not type-check
the GPU backend at all — `cargo check --features gpu --all-targets` is the check that does, and it is
a required half of the repo gate.

Beyond the two CLI backends the crate holds device infrastructure and whole stacks that are **library
surface**, exercised by its own `--features gpu` tests and benches rather than reachable from `wukongc`:
`pool.rs` (a bump-arena device allocator, and the prerequisite for graph capture — a captured region may
contain no synchronizing allocation), `graph.rs` (CUDA-graph capture/replay), `cubin.rs` (the cubin
cache above), `autotune.rs` (per-(op, shape, dtype) selection over the int8/W4A16 GEMM variants with a
validated on-disk config cache), `paged_kv.rs` + `paged_attention.rs` + `serving.rs` (paged KV-cache,
warp-cooperative paged decode attention, batched decode and scheduler), `train_resident.rs` (a
GPU-resident training step), and the `ptx_*` kernel-generator families. The compiler itself reaches the
device only through `gpu_accel`'s five hooks and `lower.rs`.

## Formatter (`wukong_fmt`, `wukongc fmt`)

`wukongc fmt [--check] [--stdout] <path|->...` reprints `.wk` sources in one canonical style. The
formatter is an AST pretty-printer that keeps comments, and its one guarantee is that **it never
changes what a program means**: when it cannot prove that, it refuses (exit 2) and leaves the file
untouched. It never emits a guess.

**The comment side channel.** The token stream carries no trivia, so `wukong_lexer` grew
`tokenize_with_comments(src, source) -> (tokens, comments, diags)`. It is `tokenize` plus a
`Vec<Comment>` (`kind: Line | Block`, `span`): a line comment's span runs from `//` up to (not
including) the `\n`, a block comment's from `/*` through its matching `*/`, counting nesting.
`Comment::text` drops a trailing `\r`; `Comment::is_doc` is `///` (not `////`) or `//!`. It is an
`Option<Vec<Comment>>` on the private `Lexer`, so plain `tokenize` (and `--emit=tokens`) records
nothing and pays nothing; a unit test pins that the tokens and diagnostics of the two entry points
are identical, BOM, CRLF and E0103 included.

**Pipeline** (`wukong_fmt::format_source`): lex with comments (any lexer error is `Syntax`) →
`parse_module_tokens_with_lints` (any parser *error* is `Syntax`; the input's `W0030`/`W0031` lints
are allowed) → print → self-check → if the input's first line break is `\r\n`, write `\r\n`
throughout; keep a leading BOM.

**The printer** (`src/printer.rs`) matches exhaustively, with no `_ =>` arm, over every `ItemKind`,
`StmtKind`, `ExprKind`, `PatKind`, `TypeKind`, `DimKind`, `Layout`, `AttrArg`, `AttrVal`,
`GenericParamKind`, `VariantData`, `VariantPat` and `ForIter` variant, so a new AST variant fails
this crate's build. Where the AST does not record a spelling, the printer reads it back from the
token stream (binary search on `span.lo`): `::` vs `.` in a field or path, `~` vs `!` (both
`UnOp::Not`), `f32x8` vs `vec[f32, 8]`, `Name::<..>` vs `Name<..>` (the only spelling after `as`),
an extern ABI string only where one was written, the parentheses around a non-tuple type (the
parser allocates a `NodeId` for them), field-pattern shorthand, a struct-variant pattern's `..`, a
dimension's or tile extent's integer text, and a tuple index (`t.0.0` is one float token the parser
splits; `t.1 .2` prints as `(t.1).2`). Expressions get minimal parentheses from the parser's
binding powers (every operator left-associative; a cast binds tighter than any binary operator and
looser than any prefix one), `& &x` never `&&x`, `(a,)` for a 1-tuple, and a struct literal is
parenthesized wherever the parser has struct literals off (an `if`/`while` condition, a `match`
scrutinee or guard, a `for` iterator, a `where` predicate, an `if let`/`while let` head) unless a
delimiter re-enables them (call arguments, `(..)`, array literals, struct-literal fields — **not**
index brackets, and not blocks).

The parser desugars several constructs; the printer recognizes each desugared shape and prints the
surface form again:

| written | the AST the parser builds | how the printer recognizes it |
|---|---|---|
| `impl [Trait for] T { fn m … }` | an `ItemKind::Impl` marker, then one top-level `Fn` per method named `T.m` | the `Impl` item's `ImplMethod.fn_name`s, consumed in order |
| `self` / `mut self` / `&self` / `&mut self` | a parameter `self: T` whose type span is the name's | `ty.span == name.span`; the tokens in the parameter span |
| `Self` inside an `impl` | the type's own symbol | the identifier's source text is `Self` |
| `if let P = e [&& g] { A } [else …]` | `match e { P [if g] => …, _ => … }`, second arm `ArmOrigin::LetFallthrough` | the arm origin; no `else` when both arm bodies share a span |
| `[l:] while let P = e [&& g] { A }` | `[l:] loop { match e { P => { A }, _ => { break; } } }` | a `Loop` whose first token is `while` |
| `let [mut] P = e else { B };` | `let names = match e { P => names, _ => { B } }` (or the bare `match` when `P` binds nothing) | the second arm's `ArmOrigin::LetElse` |
| `const X: T = v;` in a block | an immutable `let` | the token before the pattern is `const` |
| `defer t op= v;` | `defer { t op= v; }` | a `Defer` of a block whose first token is not `{` |
| `fn f() -> T = e;` | a body block whose tail is `e` | the block's span equals its tail's |
| `where T: A + B, N % 4 == 0` | `T`'s bounds (after `T`'s span) + `where_preds` | a bound past its parameter's span; merged with the predicates by position; a `+` before a bound continues the entry |
| `s[a..b]`, `s[a..]`, `s[..b]`, `s[..]` | `$slice(s, a, b)` / `$slice_from(s, a)`; an elided `a` is a zero-width `0` | the `$`-named callee (never printed) |

**Style.** 4-space indent, one final newline, no trailing whitespace on code. `module a.b` without
`;`; `import x;` and `const …;` with one. A node's attributes go on one line above it; parameter
attributes stay inline. Structs and enums put one field or variant per line with a trailing comma.
Blocks are always multi-line except an empty, comment-free `{}`. `} else if c {` / `} else {`. A
match arm is `pat[ if g] => body,`; a block-bodied arm drops the comma unless the next arm's pattern
starts with `(`, `[` or `-` (which would continue the block as a call, an index or a subtraction).
A statement ends in `;` unless it is a block-like expression (`{…}`, `if`, `match`, `loop`), `while`,
`for` or `while let`; a block-like expression statement still gets one when it is the block's last
statement with no tail (else it would re-parse as the tail) or when the next statement starts with a
character that would continue it (`( [ - * & ! ~ . < > = + / % ^ |` or `as`) — Wukong's parser,
unlike Rust's, keeps parsing postfix and binary operators after a block-like expression.

**Wrapping** decisions use a pure `flat()` rendering, which fails on any block-like sub-expression
and never consumes a comment. A signature puts one parameter per line (trailing comma) when its flat
header exceeds `max_width` (100) or a comment lies inside the parameter list; the outermost call or
struct literal of a statement does the same for its arguments/fields; an array literal of simple
elements (literals, paths, negated literals) fills lines greedily and any other array goes one
element per line. Nothing else wraps; a call holding a block-bodied closure (or any block-like
argument) keeps its arguments inline and the block opens over the following lines.

**Comments** are walked with one cursor in source order (`src/comments.rs`). Before an own-line
element (item, attribute line, statement, tail, field, variant, arm, `impl` method, vertical
parameter/argument/element, closing brace) every earlier comment goes on its own line; after the
element's line come the comments that start before the next source line break (and the next
element), plus any earlier one nothing has placed — so a comment inside an expression moves to the
end of its line. A line carries at most one trailing line comment. A blank line is kept before an
own-line element exactly when the source had two or more line breaks before it, never after `{` or
before `}`.

**The self-check** (`src/check.rs`) — `format_source` returns `Unsupported` unless all four hold:

1. **(a)** re-lexing and re-parsing the output with the *same* `Interner` gives no error and no
   parser lint (so `W0030`/`W0031` hazards never survive a format);
2. **(b)** the re-parsed module's `Debug` equals the original's with every span masked — NodeIds
   and symbols compared exactly, so a renumbered node or a different name is caught;
3. **(c)** the comment texts are identical, in order;
4. **(d)** the sorted multiset of `Ident`/`Int`/`Float`/`Str`/`Char`/`Label` token texts, every
   keyword and every `@` is identical. This is the law for source text the AST drops and no
   `NodeId` records: a struct field's or trait method's attributes (a bare `@extern` is an `@` and
   a keyword, with no identifier in it), an enum's discriminant repr (`enum E: u8`), the `pub` that
   `parse_item` eats and discards before `import` and `impl`, and keywords skipped inside an
   `extern` block. Punctuation is not counted, because the printer adds and removes `;`, `,` and
   parentheses on purpose. `check.rs` classifies every `TokenKind` in an exhaustive match, so a new
   token kind fails the build until someone decides whether (d) counts it.

The message names the first failing law with about 80 characters of context (for (d), the token and
its count on each side). These laws are never relaxed to let a file through; the printer is fixed
instead, or the file stays refused.

Two kinds of dropped text are refused by the printer before the self-check runs, with a message
naming the construct. A `pub` before `import` or `impl` is one. The other is any token in an
`extern { .. }` body that no declaration covers: `parse_extern` skips such tokens without a
diagnostic, so the printer walks the body as that loop does (an optional `pub` before `fn`, then
the declaration through its body or signature and the optional `;`). This catches punctuation
junk, which law (d) cannot see. To satisfy law (d), the printer keeps two spellings the AST cannot
distinguish. A `pub` written on a declaration inside a `pub extern { .. }` stays, even though the
block's `pub` already exports it. An empty `where` (`fn f() where { .. }`) is kept rather than
deleted.

**Adding syntax means extending `wukong_fmt` in the same commit.** The exhaustive matches fail the
build on a new AST variant, and `crates/wukong_fmt/tests/corpus.rs` — every `.wk` under `tests/run`
(with `lib/`), `examples`, `tests/imports`, `tests/warn` and `tests/fail` must format (`Syntax` is
allowed in `tests/fail` only), with zero `Unsupported` and every output a fixed point — fails on a
new fixture the printer cannot reprint. A second corpus test inserts comments at token boundaries
and formats at 100 and 40 columns; comments cannot change an AST, so nothing may be refused.
`crates/wukongc/tests/fmt.rs` holds the CLI contract and the **MIR-identity gate**: `tests/run`
(with `lib/`) and `examples/` are formatted in a scratch copy, and every root program's
`--emit=mir-high -O0` (and, for the examples, `--emit=mir -O2`) must be byte-identical before and
after, exit code included.

## Testing strategy

The corpus is the set of top-level `.wk` files in each harness directory. Count the current
checkout from the repository root; imported support files in subdirectories are excluded:

```sh
python -c "from pathlib import Path; dirs=('tests/run','tests/fail','examples','bench/kernels'); pattern='*.wk'; print('\n'.join(f'{d}: {sum(p.is_file() for p in Path(d).glob(pattern))}' for d in dirs))"
```

A file count is an inventory, not a count of passing tests or supported backend cases. Dated
measurement reports retain the corpus sizes used for those runs.

- **Unit tests** per crate (lexer, parser, sema, MIR verifier, opt passes, interpreter, vectorizer).
- **End-to-end** (`tests/run/*.wk`): the real `wukongc` binary compiles and runs each
  program; stdout/exit are checked against the `// EXPECT-EXIT:` / `// EXPECT-OUT:` directives embedded
  in the file, and a `// RUN:` directive replaces the default `--run` argument list (a fixture that must
  pin the *native* side carries `// RUN: --run --backend=native`). Placement rule: everything here must
  lower end to end — a separate gate requires every run-suite program to succeed at `--emit=mir-high`,
  `--emit=mir` and `--emit=llvm-ir`, at -O0 and -O2 — so a program whose purpose is to be **rejected**
  belongs in `tests/fail` instead.
- **Compile-fail** (`tests/fail/*.wk`, driven by `crates/wukongc/tests/fail.rs`): each
  program must be *rejected*, and with the exact stable code its `// EXPECT-CODE:` directive names; the
  harness drives the real binary with `--error-format=json --emit=mir`. Beyond the first error, in
  the style of rustc's UI tests, `// EXPECT-COUNT: <n>` pins the exact number of errors and a
  trailing `//~ <CODE>` (`//~^`/`//~v` for the line above/below) pins an error's primary line;
  a fixture with any `//~` must annotate every error in its own file. Every fixture is annotated,
  and `WUKONG_BLESS=1 cargo test -p wukongc --test fail` rewrites the counts and annotations (never
  `EXPECT-CODE`) from the compiler's output, then fails on purpose so the diff is reviewed. Eleven lexer/parser codes that
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
  *entire* output buffer to be bit-identical (f32 and int8) — plus a **recognition-off arm** that lowers
  each kernel again with `LowerOptions { recognize: false, .. }`, runs it on the interpreter, and holds
  the recognized native output to that second opinion (bit-exact for the elementwise kernels, within a
  derived `2*gamma_n*S` bound for the reductions and GEMMs, within twice the documented f64 bound for
  the norms — a bound on magnitude only: a non-finite lane and a zero lane are held to their exact
  bits, sign of zero included, in every class; the lane classes where a kernel is known to disagree with its source loop are listed with
  reproducers in `KNOWN_OFF_DIVERGENCES` and fail the run once they stop reproducing) — and a **grammar** fuzzer (`fuzz_grammar`)
  that generates random well-typed programs through the real pipeline and asserts interpreter == native
  and -O0 == -O2 == -O3 on (exit, stdout). Both use a seeded PRNG, so a failure reproduces exactly from
  the seed it prints; `WUKONG_FUZZ_PROGRAMS=N` deepens the grammar run and an unusable value falls back
  to the default rather than silently disabling the gate.
  A complementary numeric grammar adds exact width-aware integer and binary floating oracles,
  runtime divisor/shift boundaries, f64/f16/bf16 samples and MIR non-vacuity checks.
  `WUKONG_FUZZ_NUMERIC_PROGRAMS=N` selects its seed count (default four; zero or invalid values
  restore the default). Its finite normal/signed-zero samples are not exhaustive float coverage.
- **Determinism** (`crates/wukongc/tests/determinism.rs`): the compiler must be a pure function of its
  input. Three *fresh processes* per compile (each with its own `HashMap` hash seed) must agree
  byte-for-byte on `--emit=mir -O2`, on `--emit=obj -O2` (and the `WUKONG_PAR_CODEGEN=0` serial object
  path must equal the parallel one), and on the `--error-format=json` diagnostic stream over the
  compile-fail corpus. Stated limit: the seed mechanism only bites std `HashMap`/`HashSet`; the
  optimizer's FxHash maps have a fixed seed, so these gates say nothing about insertion-order dependence
  there.
- **AOT executables** (`crates/wukongc/tests/exe.rs`): curated fixtures (including
  `print_float_display.wk`, which pins how every float prints) are linked with `--emit=exe` and
  their stdout/exit compared against `--run`, extending the differential gate to the linked exe (string
  `.rodata` and the linked `wukong_runtime` kernels). `exe_matches_run_on_a_corpus_slice` does the same
  for every 8th corpus fixture in parallel (`WUKONG_EXE_CORPUS=all` for the whole corpus), skipping only
  documented declines (the native JIT cannot compile it either; on the C fallback, it needs a
  `wukong_*` kernel) and failing if they exceed 10% on the rustc path, and
  `cc_fallback_discovered_from_the_cli_matches_run` reaches the C fallback through a copied `wukongc`
  with no runtime rlib. The gate probes once which link path exists on the box and may skip only when
  neither does; a failure of the compiler's own link is reported with its captured stderr. **Run it under `--release` as well as the default profile** — the two profiles hand
  the linker archives of different shapes (see the bitcode/LTO paragraph in *Native backend* above), so
  a green debug run says nothing about release; `cargo test -p wukongc --release --test exe` is the only
  gate that has ever exercised the release AOT link, and it is minutes of wall time because each fixture
  is a full LTO of the runtime.
- **Examples** (`crates/wukongc/tests/examples.rs`): every `examples/*.wk` program is an
  interpreter-vs-native differential fixture *and* carries a declared execution class (`Runs`,
  `NotLowered`, `NoEntry`, `DataGuarded`), because a program that fails to compile trivially satisfies
  agreement on both backends. `Runs` is the default, so a new example must actually execute or be
  declared otherwise — and a `Runs` example must carry its expected output in the file, as
  `EXPECT-EXIT` / `EXPECT-OUT*` header directives in the `tests/run` grammar (one shared parser,
  `common/mod.rs`), with values derived from the program's own scalar semantics.
- **Forced-parallel lane** (`crates/wukongc/tests/parallel_lane.rs`): every `_parallel` kernel in
  `wukong_runtime` falls back to its serial sibling below a work gate sized for real models, and no
  corpus fixture is that large, so no other gate ever runs a `_parallel` split. This lane reruns
  every program whose census line lists a `_parallel` symbol with the gate overrides at their
  smallest accepted values (`WUKONG_PAR_MIN_WORK=1`, `WUKONG_GEMM_PAR_MIN_MACS=1`,
  `WUKONG_VELEM_PAR_MIN=1`, `WUKONG_PACK_PAR_MIN_KB=0`) and requires native-forced == native-default
  == interpreter, byte for byte. The two gates no knob opens and the fixed-`RCHUNK` reductions are
  reached by fixtures sized above them instead, one each: `TRANSPOSE_PAR_MIN` (65536) by
  `cov4_par_transpose_wide.wk` (65792 elements), `VMATH_PAR_MIN` (16384) by `vmath_parallel.wk`
  (20000 — the only corpus program whose `wukong_vmath_f32_parallel` call crosses it), and `RCHUNK`
  (8192) by `cov4_par_lowp_dot.wk` and `parallel_argmax.wk` (20000). `FIXED_GATE_WITNESSES` in the
  same file pins that list: a witness that leaves the corpus, stops dispatching the gated kernel, or
  no longer spells its size fails `every_fixed_gate_has_a_witness_sized_above_it`.
- **Emittable-surface census** (`recognize.rs`): every `wukong_*` symbol interned anywhere in
  `wukong_mir_build/src` must be emitted by some `tests/run` program at `-O2`, or be listed in
  `NEVER_EMITTED_ALLOWLIST` with a reason. The `tests/run/cov4_par_*.wk` fixtures are the witnesses
  for the `_parallel` twins: a multi-statement `@parallel` function is lowered whole with
  `parallel_fn` set (the outliner takes only a single-loop body), so each recognizer arm in it emits
  its `_parallel` kernel; every lane is printed against hand-derived values.
- **Emit stages** (`crates/wukongc/tests/emit.rs`): `--emit=tokens|ast` over the examples and the run
  suite, and `--emit=mir-high|mir|llvm-ir` over the run suite at -O0 and -O2 — which is also what keeps
  the MIR verifier and the pretty-printers honest.

- **GPU corpus coverage** (`--features gpu`, device required): `lower.rs` and `megakernel.rs` each
  sweep the whole `tests/run` corpus against the interpreter oracle at two `-O` levels and assert a
  **ratcheted floor**, so a construct that silently starts declining goes red instead of quietly
  shrinking coverage. Both sweeps normalize their input through the documented, result-identical
  128-bit CLIF fallback (`lower::hostvec::build`, i.e. `WUKONG_P4_NO_256=1`) before lowering. That is
  not a detail: the general 256-bit AVX2 recipe is Win64-only, `lower.rs` declines the op it
  produces, and so before the normalization the *same commit* read a higher coverage number on Linux
  than on Windows — the metric conflated "what gpu-native can lower" with "did the host CPU
  vectorizer fire", and one hardcoded floor could only ever ratchet one of the two readings. A
  device-free gate proves the normalization is both effective and complete by re-sweeping
  host-native and attributing every gained program to a vectorizer recipe.

Everything above **except the GPU corpus gate** runs with plain `cargo test` and needs no C or LLVM
toolchain (Cranelift is a pure-Rust crate).

One profile caveat, learned the hard way: `wukong_runtime`'s first invariant is that each AVX2 kernel
agrees **bit-for-bit** with its scalar twin, and that is a property a legal optimizer choice can
break — a `colreduce` divergence reproduced byte-identically on Windows and Linux in release while
passing in debug, because it needed LLVM to have auto-vectorized the *scalar twin* before it could
exist at all. Debug-only checking of that crate was therefore checking the wrong profile, since every
performance number the project publishes comes from a release binary; CI now also runs
`cargo test -p wukong_runtime --release --lib`.

Two suites are conditional and say so out loud: the `--emit=exe` gate needs either `rustc` with
a `wukong_runtime` rlib reachable from the compiler binary (see `runtime_rlib_in` above) or a working
`cc`/`$CC`, and prints why it skipped otherwise — and it fails outright if it linked *nothing*, so it
cannot go dark; and the GPU suites need `--features gpu` plus a reachable device — plain `cargo test` does not
even type-check that backend, so `cargo check --features gpu --all-targets` is a required second half of
the gate. `WUKONG_GPU_REQUIRED=1` and `WUKONG_PEER_REQUIRED=1` turn a device- or peer-absent skip into a
failure on machines that are supposed to have them.

Not all of the GPU surface needs the device, though, and the part that does not is gated separately
because the feature flag alone was hiding it. `wukong_driver`'s `gpu` feature carries the driver-side
dispatch rules — which route a recognized `C = A·Bᵀ` takes per compute capability, which shapes and
which operand *values* the Hopper `wgmma` seam declines before launching, and the counters that
distinguish a `wgmma` launch from a fallback (`crates/wukong_driver/src/gpu_accel.rs`).
Those are pure functions of a capability and a shape, so `cargo test -p wukong_driver --features gpu
--lib` runs them anywhere; the six end-to-end gates in the same crate `[skip]` without a device. That
command is its own required gate step and a CI step in the `gpu-check` job, because nothing else
reaches it: the workspace `cargo test` never passes `--features gpu` (the crate reports 20 tests
instead of 47), the device suite is scoped `-p wukong_codegen_gpu`, and `check`/`clippy` compile a
test without running it.
