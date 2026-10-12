# Wukong Roadmap & Known Limitations

Wukong is a **programming language**: users write `.wk` source, `wukongc` checks types and tensor
shapes, and backends execute the result. This page is an honest snapshot of that language and
compiler — what parses, what shape-checks, what runs on interpreter/native/GPU — not a catalog of
hand-tuned GPU microbenchmarks. Silent recognizers and hand-written PTX are current implementation;
the product is the `.wk` surface and the programs that go through it.

Wukong is built openly and incrementally. The sections below record what works, what is
checked-but-not-executed, and what is planned — so expectations match reality.

> **Scope of every performance figure quoted on this page.** CPU numbers are one Windows 11 laptop's
> (Intel Core Ultra 7 155H, AVX2/FMA + AVX-VNNI, **no AVX-512**), and GPU numbers are one **NVIDIA
> RTX 4050 Laptop GPU**'s (`sm_89`, 20 SMs, 6 GB, ~192 GB/s, Windows/WDDM) against the peers that box
> can host. Several families additionally have an **open measurement debt** — an end-to-end model
> ratio that is an upper bound pending re-measurement, ~40 rows whose Rust column needs re-measuring
> after a `noalias` fix, a C++ column that only ever existed in three sections, a general-code suite
> whose tables are superseded with no replacement, and a GPU framework comparison that is against
> **eager** PyTorch rather than `torch.compile`. `BENCHMARKS.md` opens with a per-family standing
> index that says, for any number, whether it currently stands; read it before quoting one from here.

## Works end to end (interpreter `--run`, **and native code** `--backend=native`)

The MIR interpreter and **Cranelift native backend** execute the supported language subset.
Corpus differential tests compare outputs across optimization levels. Unsupported constructs
remain documented limitations; the C-link fallback prints exactly what `--run` prints (it lacks
only the `wukong_*` kernels, a link error), and `exe.rs` compares a slice of the corpus's linked
executables with `--run`. A runtime failure (a failed `assert`, a slice index out of bounds) stops
the program at the same statement on every path — interpreter, JIT, linked executable, and inside
a `@parallel` region (the earliest failing iteration is reported) — keeping the output printed
before it, with exit status 1 (`tests/run/assert_fail_trailing_output.wk`).
Cranelift provides JIT execution with `--run --backend=native` and object/exe output with
`--emit=obj|exe`, with **no LLVM toolchain required**. See `BENCHMARKS.md` for cross-language numbers.
`--target-cpu=<native|x86-64|x86-64-v2|x86-64-v3|x86-64-v4>` picks the instruction-set level of that
native code (default `native`, the host's features); below x86-64-v3 the raw-AVX2 loop kernels are
not used, so an object built on an AVX2 machine runs on a baseline x86-64 CPU. Cross-compilation to
another architecture or OS is not supported. A native-backend failure is a catalogued diagnostic
(`C0002` for a single layout above 4 GiB, `C0001` for a by-design decline) in both `--error-format`s,
and a failure inside one function points at the source construct being lowered (the oversized
`let`), else at that function's declaration; a module-level failure (an unresolved `extern "C"`
symbol, an unsupported target CPU) has no span. Every MIR value is located straight out of lowering,
inlining and unrolling carry locations onto the values they copy, and values other passes create
stay unknown (about 75% of the corpus's instructions are still located at `-O2`); Cranelift's
per-instruction source locations follow the table and drop, never borrow, an unknown one. A DWARF
line table built from them is not emitted yet.

- **Multi-file modules with real namespaces**: `import a.b` loads `a/b.wk` (resolved against the
  *root* source file's directory); files load once each — an import cycle or diamond dedups by
  canonical path, never errors — and an unresolvable import is `E0305`; resolution is case-exact
  on every host (a case-only mismatch with the file on disk is `E0330`, not a host-dependent
  success). Every file is its own
  module: items are private unless `pub` (`E0312` otherwise), imports are per file and not
  transitive, `import a.b;` brings `b`'s `pub` items in unqualified plus the `b::` qualifier,
  `import a.b as c;` binds only `c::`, and `import a.b.{x, y};` selects. Two modules may define the
  same private name; two plain imports exporting one name make an unqualified *use* ambiguous
  (`E0313`). Qualified names work in calls, values, struct literals, enum variants/patterns and types
  (`tests/run/import_namespaces.wk`, `tests/run/import_multi.wk`). Resolution is a loader pass that
  renames each imported item to the unique symbol `<module>.<name>`, so sema, MIR and every backend
  are unchanged. (Field-level `pub` is accepted, not enforced; `--emit=tokens|ast` stay
  root-file-only by design.)
- **`defer`** runs at every exit of its block — fallthrough, `return`, `break`/`continue` — LIFO,
  innermost block first, with returned/broken/tail values computed before the defers
  (`tests/run/defer_scope_exit.wk`). It is a first AST normalization in `mir_build`, so every backend,
  recognizer and autodiff sees plain statements; control flow out of a deferred expression or
  rebinding a name a pending defer uses is `E0314`.
- **C FFI**: `extern "C"` declarations become `Program::externs` imports. Native binds the host C
  symbol (JIT) or leaves it to the linker (`--emit=obj|exe`); the interpreter calls the *same* symbol
  for scalar signatures (≤ 4 int/`bool`/`f32`/`f64` args) and refuses a pointer signature or an
  unresolvable symbol before the run (`C0001`); the LLVM emitter declares it with
  `signext`/`zeroext`. Types without a C mapping are `C0001` at the declaration
  (`tests/run/extern_c_libm.wk`).
- **Named kernel builtins with shape signatures** (`matmul`/`matmul_nt`/`matmul_tn`, `linear`,
  `transpose`, `softmax`/`log_softmax`, `layernorm`/`rmsnorm`, the two-argument activations
  `exp`/`log`/`tanh`/`sigmoid`/`silu`/`gelu`/`relu(x, out)`, `sum`/`amax`/`amin`/`dot`, and `sdpa`):
  each lowers to exactly one runtime kernel on both backends regardless of recognition, after sema
  unifies its dimensions like a generic call (`E0409` when nothing fixes one; `E0501`/`E0502` on a
  disagreement). Runtime extents are checked (a violation traps like `assert`), an output that
  overlaps an input computes through scratch (value semantics), and an empty contraction is the zero
  matrix (`tests/run/builtin_*.wk`, `tests/run/sdpa_blocks.wk`). `sdpa` gained its shape signature
  here; it used to be checked only at lowering, as `C0001`. See the guide's "Kernel builtins".
- **Methods, traits and function-typed parameters — all static dispatch.** `impl Type { .. }`
  methods (`x.m(a)`, `Type::m(a)`, `self`/`&self`/`&mut self`/`mut self`, `Self`, auto-deref through a
  reference) lower to the plain function `Type.method`; `trait` + `impl Trait for Type` are checked
  for conformance (`E0410`), and a bounded type parameter (`<T: Shape>` / `where T: Shape`) calls
  trait methods through monomorphization; a `fn(A) -> R` parameter is specialized per bound
  function (`apply$square`), so no function pointer exists at run time (`E0411` outside a parameter
  or a closure `let`'s annotation).
  Calls such a specialization makes are invisible to the alias analysis, so every function they can
  reach forgoes disjointness-requiring kernels (`tests/run/{methods_impl,traits_static_dispatch,fn_params}.wk`).
- **Closures** (2026-10-12, `gap5/closures`). `|x| x * k`, `|| 0`, `|x: f32| -> f32 { .. }` passed to
  a function-typed parameter: captures are copied by value when the argument is evaluated, the
  literal is lifted to a function whose leading parameters are its captures, and the callee is
  specialized per closure (`apply$main$closure$0`), so a call through the parameter stays a direct
  call with no function pointer (interp and native, every `-O`; `tests/run/closures.wk`). `E0418`
  for an uncopyable capture, `E0419` for a parameter type a callee generic leaves open, `E0411`
  outside a call argument (a `let` initializer too since `gap6/closures-escaping`, below), and
  `E0304` for writing a capture of an immutable binding — a write to a mutable one is a
  by-reference capture since `gap6/closures-escaping`, where `gap5` rejected every capture write.
  See the guide's "Closures".
- **Closures beyond the call argument** (2026-10-13, `gap6/closures-escaping`). A closure bound by
  `let` (`let f = |x: f32| x * k;`) is called (`f(2.0)`) and passed (`apply(f, 3.0)`) with its
  captures copied once, at the `let`, and every use resolved statically (`E0421` for any other use,
  `E0422` for a `mut`, pattern or reassigned binding); a closure argument of a method or associated
  function (`v.map(|a| a * k)`, `V::make(..)`) specializes the method like a free function
  (`tests/run/{closure_let,closure_method}.wk`). A closure captures an array, tuple, struct or
  data-carrying enum by value (a copy), and a variable it writes by reference — by assignment,
  `&mut`, a `mut` parameter or a builtin's output operand — and the enclosing function sees the
  write (`tests/run/{closure_captures,closure_capture_out_params}.wk`). An unannotated closure parameter is
  typed from the call's other arguments when the expected type names a callee generic, and a
  function-typed parameter over a dimension generic (`fn(Tensor[f32, N]) -> f32`) binds it from the
  function passed (`tests/run/closure_infer.wk`). Interp and native agree at every `-O`.
- **Diagnostics and tooling surface.** `--emit=check` runs the front end (lex/parse/imports/sema,
  then the lints) and stops before lowering — a fast editor loop with no `C0001` lowering limits.
  `--error-format=json`
  is a stable schema (v1): spans carry byte and line/column end positions, and each diagnostic
  carries `suggestions`, `children`, a `rendered` copy and `"version":1`, with the v0
  `severity`/`code`/`message` prefix unchanged. Diagnostics may carry a `help:` suggestion (span +
  replacement), a label in another file gets a `::: file:line:col` header, `--color=auto` honours
  `NO_COLOR`/`CLICOLOR_FORCE`/`TERM=dumb` (and enables Windows VT), and `--explain` with no code
  lists every catalogued code. (A full LSP is out of scope; see MASTER_PLAN §3.6.)
- **Name resolution is total.** Every called name, field, builtin-type method and type name resolves
  or is a resolution error with a did-you-mean suggestion: `E0320` (unknown function), `E0321` (no
  such field), `E0322` (no such method on a builtin type), `E0323` (calling a non-function — the
  missing-`;` trap), `E0324` (unknown type, struct literal of an undeclared name, a trait used as a
  type). These used to type as a lenient `Unknown` and fail in lowering as the codegen limit `C0001`,
  and an undeclared type name was lowered as an `i32` slot. One builtin callee registry
  (`wukong_sema::is_builtin_callee`) is read by `mir_build` too, so a single-name call that reaches
  lowering unresolved is an internal compiler error, not a codegen limit
  (`tests/fail/{unknown_fn_call,unknown_field,builtin_method_unknown,call_non_function,unknown_type_name,struct_literal_unknown}.wk`).
- **Call-site shape errors carry provenance.** An `E0502` at a call is reported at the conflicting
  argument, with labels naming the argument axis or turbofish that fixed the dimension and the
  callee parameter that declares it; `E0503` names the callee and labels its definition and the
  extra or missing arguments (pinned in JSON by `crates/wukongc/tests/fail.rs`).
- **Every call dimension is known at type-check time.** A dimension a callee's parameter shapes use
  that no turbofish, argument shape, solvable element count or `?` argument extent fixes is `E0520`
  at the call (with the turbofish to write), instead of the lowering-time `C0001` it used to be
  (`tests/fail/dim_uninferable.wk`). A `const` named in a turbofish is its value, as in a type — it
  used to be an opaque variable that lowering replaced with a `?` argument's run-time extent
  (`tests/run/turbofish_const_dim.wk`). Still open: a turbofish naming a run-time *variable*
  (`f::<n>(..)`) binds the dimension to that variable by name, so an escaping result type re-reads
  `n` — mutating or shadowing `n` after the call changes that tensor's extent.
- **`E0401` is a plain type mismatch again.** The catch-all was split by rule, each with its own
  `--explain` page and a compile-tested erroneous/fixed example: `E0430` struct-literal fields, `E0431`
  malformed or out-of-range literals (and enum discriminants), `E0432` invalid escapes, `E0433`
  operators applied to a type they are not defined for (including indexing and `*`), `E0434` a missing
  return value, `E0435` misused enum payloads, `E0436` non-boolean conditions and `E0437` builtin
  arguments of the wrong kind. `E0401` keeps expected/found mismatches, invalid casts, a value `break`
  out of a statement loop and layout/length limits.
- Functions (including recursion and mutual recursion), and direct calls. The interpreter oracle is
  **stackless**: a Wukong call pushes a frame on an explicit heap stack instead of recursing on the
  host stack, so recursion depth does not depend on the host, the build profile or a worker-thread
  reservation, and every run executes on its caller's thread. Its ceiling is an explicit,
  documented cap: past **300,000** nested Wukong calls in every build (override with the
  `WUKONG_INTERP_MAX_DEPTH` environment variable; it replaced the 40,000-debug / 300,000-release
  constants that were bisected against a 512 MiB host stack) the interpreter reports
  `error: interpreter call-depth limit exceeded (likely unbounded recursion)` and exits 1. That is a
  **diagnostic**, never an uncatchable process abort. Recursion past the cap is a **resource limit
  outside the bit-for-bit differential contract** — the oracle *refuses* where the native backend
  (on its 512 MiB `--run` worker stack) may still complete, the same "outside the defined contract" status
  as an out-of-bounds access. At `-O1`
  and above tail-recursion elimination turns a self-call in tail position into a loop, so a
  tail-recursive function no longer consumes call depth there (an unbounded tail recursion then
  loops until the interpreter's step guard stops it instead of hitting the depth cap).
- `let`/`let mut`/`const`, shadowing, block-as-expression values, **`let` tuple destructuring**
  (`let (a, b) = …`, nested patterns, `_`; `tests/run/let_destructure.wk`), and a **top-level
  `const` used as a value** (its initializer inlined at every use site — arithmetic, array index,
  loop bound, **array length in a type** (`let a: [i32; N]`; `tests/run/const_array_length.wk`),
  const-referencing-const; `tests/run/top_level_const.wk`).
- Integers (`i8..i64`, `u8..u64`, `usize`/`isize`), `bool`, `char` (a 32-bit Unicode scalar,
  interconvertible with the integers via `as`; `tests/run/char_type.wk`), and floats. `f32` is computed at **`f32`
  precision** (interpreter and native agree exactly); **both `bf16` and `f16` are real 2-byte storage**
  rounded to that grid (round-to-nearest-even) on store and on the cast, with `f32` compute. bf16 rounds
  with cheap inline bit-math (it's the top 16 bits of an `f32`); f16's IEEE-half layout has no such
  shortcut and Cranelift x64 has no f16 convert lowering, so f16 cast/load/store call shared `half`-crate
  shims (`wukong_f32_to_f16_bits`/`wukong_f16_bits_to_f32`) the interpreter uses too — bit-exact by
  construction. `f64` is full: arithmetic in double precision, and every transcendental on an
  `f64` operand (the whole `exp`/`log`/trig/inverse-trig/hyperbolic/`erf`/`cbrt`/`pow`/`hypot` set
  and the activations composed from them) is one call into the runtime's pure-Rust double-precision
  library `wukong_runtime::math64` — ≤1 ULP for the core, Payne–Hanek trig reduction for every
  finite argument, C23 special values, identical bits on the interpreter, the JIT and the AOT link
  (it used to be the f32 polynomial behind a demote/promote pair; `tests/run/transcendental_f64.wk`
  prints twelve digits of each). (The GPU adds tensor-core fp16/bf16/fp8 GEMM on top.)
- **Mixed-precision (bf16 *and* f16) CPU suite → SIMD dispatch**: low-precision `[bf16]`/`[f16]` arrays
  read through an `as f32` widening cast (lossless: `<<16` for bf16, F16C `vcvtph2ps` for f16) with
  **f32 accumulate/compute** — the standard ML contract — are recognized and lowered to half-precision
  runtime kernels. Symmetric across both precisions:
  - **reductions** `dot`/`sum` (`wukong_{dot,sum}_{bf16,f16}[_parallel]`) — ~3.0–3.5× vs C for dot,
    ~6–8× for sum, growing as the working set spills L3 (`tests/run/reduce_{bf16,f16}.wk`);
  - **max/min/absmax** (`wukong_reduce_{bf16,f16}[_parallel](x,n,op)`) — the per-tensor absmax is the
    symmetric-quantization scale; exact (max/min round nothing) (`reduce_bf16_minmax.wk`). Inside a
    `@parallel` function each of dot/sum/max/min/absmax dispatches to its multicore `_parallel` twin —
    a deterministic fixed-`RCHUNK` chunk fold (the per-chunk serial kernel + an ascending partial
    combine), bit-identical regardless of thread count and the bf16/f16 analog of the f32
    `wukong_sreduce_f32_parallel` (`tests/run/parallel_reduce_lowp.wk`);
  - **streaming axpby** `out = a·x + b·y` (`wukong_axpby_{bf16,f16}`), half-in/f32-out, ~1.3× ≫ L3
    (requires two additive terms; a 1-term scale would force a `0*inf` the source lacks);
  - **activations** — the full **36**-op transcendental set over a half-precision input
    (`wukong_vmath_{bf16,f16}`, `out[i] = f((x[i] as f32))`).
  The recognizers are precision-generic (`match_lowp_reduction`/`match_lowp_axpby`/`match_vmath_stmt`),
  and the interpreter marshals through the identical kernel, so native == interp bit-for-bit. C/Rust
  can vectorize neither a `libm` call nor the half→f32 widen, so the gap is structural. A half-precision
  **output** now ships too: a narrowing store (`f32_to_{bf16,f16}_bits`, the reverse of the widen)
  feeds all-half twins — the streaming **axpby** `wukong_axpby_{bf16,f16}_out` (bf16/f16 in *and* out,
  6 bytes/elem vs an all-f32 axpby's 12, so ~2× on the memory-write-bound stream) and the half-output
  **activations** `wukong_vmath_{bf16,f16}_out` (half in *and* out — storage-halving, but the
  transcendental itself is call-bound, so a footprint win not a throughput one). The narrowing round
  goes through the same shim the interpreter uses, so native == interp bit-for-bit and `-O0` == `-O3`
  (`tests/run/{axpby_half_out,vmath_half_out}.wk`).
- All arithmetic/comparison/bitwise/boolean operators (`&&`/`||` **short-circuit**), compound
  assignment, casts — including a float → narrow-int cast (`1e30 as i8`) that **saturates** identically
  on both backends (`tests/run/float_cast_narrow.wk`).
- `if`/`else` (statement and value position), `while`, `for … in a..b` / `a..=b` (half-open or
  inclusive) `[step s]` (the inclusive and stepped forms are correct but neither vectorized nor
  kernel-dispatched — see the SIMD bullet), `loop { … }` with
  `break`/`continue`, including **labeled loops** `'outer: for … { … break 'outer; continue 'outer; }`
  — a labeled `break`/`continue` targets the named enclosing loop, not just the innermost (the lexer
  tells a label `'outer` from a char literal `'a'` the way Rust does; `tests/run/labeled_loop.wk`).
  `continue` in a range `for` runs the loop step (`tests/run/for_continue.wk`); a `break`/`continue`
  outside any loop, or one naming an undeclared label, is rejected with `E0303`
  (`tests/fail/{break_outside_loop,break_unknown_label}.wk`). **Loop-as-expression / break-with-value**
  (`let x = loop { break 5; };`) works: `loop` is a value-producing expression whose type is inferred by
  unifying every `break <value>` (composing with `if`/`match` value merges), so a value `loop` is a call
  argument, array element, tail/return value, or aggregate field; a labeled `break 'outer v` carries a
  value out of an outer loop. The value merges on a typed block param on the loop's exit block — the same
  machinery `if`/`match` merges use, so interp == native, `-O0` == `-O3`. A `break <value>` targeting a
  statement-position loop (value discarded, like `while`/`for`) is rejected `E0401`, and breaks with
  mismatched shapes `E0502` (`tests/run/loop_break_{value,labeled,compose}.wk`,
  `tests/fail/break_value_{in_stmt_loop,shape_mismatch}.wk`).
- **`match`** in value and statement position: integer/bool literal, identifier-binding, and wildcard
  `_` patterns, **or-patterns** `1 | 2 | 3`, half-open `0..10` / inclusive `0..=10` **range** patterns,
  **enum-variant** patterns `Color::Red` (matched by discriminant), and **tuple** patterns `(0, _)`
  (per-field tests + bindings, nesting and composition like `(0 | 1, y)`), **struct** patterns
  `Point { x: 0, y, .. }`, **`name @ sub`** bindings and **array** patterns `[first, .., last]` on a
  fixed-size array, each with an optional `if` guard that runs only after its pattern matched.
  `if let`, `while let` and `let … else` (whose `else` must diverge) are rewritten by the parser into
  a `match`; `if let P = e && c1 && c2` / `while let P = e && c` put the `&&` chain on the pattern's
  arm as its guard (Rust's let chains, one `let` per chain; a `||` after the condition is `E0200`),
  and a plain `let` pattern must be irrefutable (`E0442`). The scrutinee is evaluated once and the
  whole `match` lowers to an if-else chain; a match sema proves exhaustive (over a type whose
  all-zero value the analysis models) takes its last unguarded arm untested — no compare and no
  trap, at every `-O` level and on every backend — so an exhaustive `bool` match is one branch. At
  `-O2` four or more equality tests on one integer scrutinee become one MIR `switch` (a Cranelift
  jump table or search tree; the `gpu-native` backend declines it)
  (`tests/run/{match_expr,match_patterns,match_tuple,match_dense_enum_switch,match_exhaustive_fallthrough,if_let_chain,patterns_struct_binding_slice}.wk`).
  **Destructuring `for` heads** over an array or a slice — `for (i, x) in pairs`,
  `for Point { x, y } in pts` — bind each element as `let <pat> = a[i];` would; the pattern must be
  irrefutable by its form (`E0442`; `tests/run/for_destructure.wk`). Not yet: an array pattern on a
  `[]T` slice (its length is a run-time value), a `rest @ ..` sub-slice binding, a binding inside an
  or-pattern (`C0001` at the pattern), and more than one `let` in an `if let` chain.
- **C-style enums** `enum Code { Ok = 10, Err }` — explicit or auto-incrementing discriminants; a
  variant *is* its integer discriminant, usable in `let`, `==`, `as i32`, and as a `match` pattern
  (`tests/run/enum_cstyle.wk`).
- **Data-carrying (tagged-union) enums** `enum Expr { Num(i32), Add(i32, i32), Nil }` — tuple *and*
  struct payloads: construction `E::V(a, b)` / `E::V { x, y }`, and payload `match` with field
  bindings, literal sub-patterns, `if` guards, nesting in an array of enums, and by-value passing to a
  function. A value is a 4-byte i32 discriminant + a padded payload union, addressed by base pointer
  exactly like a struct — no backend change; interp == native bit-for-bit and -O0 == -O3
  (`tests/run/enum_payload_{tuple,struct}.wk`). A payload-carrying variant is *not* castable to an
  integer, and an enum that holds itself by value (`Cons(i32, List)`) is rejected as infinitely sized
  (E0402); a non-exhaustive `match` over the variants is E0405, judged through the payload
  sub-patterns (`Add(0, y)` does not cover all of `Add`) with a witness pattern in the message.
- **Slices `[]T`** — a fat-pointer view `{ data: *T @ 0, len: i64 @ 8 }` over existing array storage:
  length `s.len()`, indexed read/write `s[i]`, iteration `for x in s`, and passing to a function —
  either an already-materialized slice or a fixed-size array *unsized* to a slice parameter at the
  call site (`let s: []T = arr` / `f(arr)`). A write through the slice aliases the backing array. The
  16-byte fat pointer is a by-pointer aggregate the two backends address identically, so interp ==
  native bit-for-bit and -O0 == -O3 (`tests/run/slice_basics.wk`). The element types must match on
  the unsizing coercion, and a scalar passed where a slice is expected is E0401; slice indexing is
  checked against the runtime length in the fat pointer (see *Slice accesses are bounds-checked*).
- **Heap allocation (v1)** — `alloc_<T>(n) -> []T` and `free(s)`: the first *runtime-sized* buffers
  (every other buffer is a fixed-size stack array). The surface is the typed per-scalar family
  `alloc_f32`/`f64`/`i32`/`i64`/`i8`/`u8`/`f16`/`bf16` (chosen over a generic `alloc<T>` — sema types
  builtins nominally in one place, and the diff stays small; a user fn of the same name shadows the
  builtin). Semantics: any integer count (a negative count clamps to an **empty** slice);
  contents **zero-initialized on both backends** (calloc'd bytes on native, typed zero `Value`s in
  the interpreter — the determinism contract); the result is an ordinary slice (`s[i]`, `s.len()`,
  `for x in s`, fn-boundary passing and mutation through a `mut` slice param). Lowered to opaque
  `wukong_rt_alloc`/`wukong_rt_free` runtime calls plus fat-pointer construction — **no new MIR
  op**, and the optimizer's conservative call handling (no CSE/DSE/LICM across calls) keeps stores
  to alloc'd memory and allocation identity sound at every `-O` level. Misuse is a compile error
  (non-integer count / freeing a non-slice E0437, arity E0503 — `tests/fail/heap_*.wk`);
  double-free, freeing a non-alloc slice, or use-after-free is **undefined on native** and
  ignored or harmless (never a crash) in the interpreter, which reuses freed blocks through a
  size-bucketed free list and releases a frame's allocas when it returns. interp == native bit-for-bit and
  -O0 == -O3 (`tests/run/heap_{alloc,alloc_fn,zero_init,len}.wk`, `differential_heap_alloc`, and
  `heap_alloc.wk` in the linked-exe AOT gate — the rustc link resolves the two runtime symbols
  from the `wukong_runtime` rlib). Loops over alloc'd slices reach the **recognized-kernel** path:
  every kernel buffer operand resolves through `kernel_base_ptr`, which loads a `[]T`'s data pointer
  out of its 16-byte fat pointer, so `for i in 0..n { y[i] = exp(x[i]); }` over two `alloc_f32` slices
  emits `wukong_vmath_f32`, and velem / GEMM / reduction nests over slices dispatch too. What still
  declines on a slice base is the *general* auto-vectorizer — the same loop written over slices emits
  no vector ops where the fixed-array form does (a future perf lever, correctness unaffected).
  Native heap data is **64-byte aligned** (the runtime's `HEAP_ALIGN`, mirrored by the `cc`
  fallback shim), not the system allocator's 16.
- **Scratch allocation** — `scratch_<T>(n) -> []T`, `scratch_mark() -> i64`,
  `scratch_release(m)`: a per-thread bump arena for temporaries (`wukong_runtime::scratch`), the
  temporary-storage design of Odin/Jai/Zig. Zeroed, 64-byte-aligned slices; the mark is the count of
  live scratch allocations (identical on every backend); a release pops everything allocated after
  its mark, LIFO, and the bytes are reused (and re-zeroed) by the next allocations, so a balanced
  per-call `mark; scratch_…; release` stops allocating after its first call. Chunks never move and
  grow geometrically. The interpreter mirrors the count and reuses a released region at the same
  depth, so its memory stays bounded too. Lowered to opaque `wukong_rt_scratch_{alloc,mark,release}`
  calls like the heap builtins; misuse is E0437/E0503; use after release is undefined on native,
  like use after `free` (`tests/run/alloc_scratch_arena.wk`, also in the AOT gate). Not yet: an
  uninitialized variant and an arena handle a program can pass around. `@align(N)` on a `let`
  (2026-10-12, `gap5/mir-ops`) aligns a local array, tuple or struct's stack slot to a power of two
  up to 4096 (`Op::Alloca`'s `align`; Cranelift rounds the base up at run time, so it holds at
  every call depth), while every heap and scratch allocation is already 64-byte aligned
  (`tests/run/align_local.wk`).
- **Atomic integer add** — `atomic_add(place, v)` (2026-10-12, `gap5/mir-ops`): `place = place +
  v`, wrapping, as one sequentially consistent read-modify-write (`Op::AtomicAdd`; Cranelift
  `atomic_rmw add`, LLVM `atomicrmw add seq_cst`). `place` is an element, field or pointee of an
  integer type (a bare name is E0437, a float/`bool`/`char` place E0446, a value of another integer
  type E0447). Integer addition commutes, so a `@parallel` histogram or counter prints the
  interpreter's serial answer bit for bit on every backend; the parallel-region proof outlines a
  loop whose only shared writes are atomic adds to a captured array nothing else in the loop touches
  (`tests/run/atomic_add_histogram.wk`: three whole-function bodies and one mid-function region,
  every lane printed).
- **Typed file I/O** — `read_<T>(path, buf) -> i64`, `write_<T>(path, buf) -> i64`,
  `read_<T>_at(path, byte_off, buf) -> i64` and `file_size(path) -> i64` over **every fixed-width
  numeric dtype** (`f32 f64 f16 bf16 i8 i16 i32 i64 u8 u16 u32 u64`), where `path` is a `*u8` and
  `buf` a `[]T` slice, a `[T; N]` array or a contiguous `Tensor[T, …]` with no `?` dimension (writing a
  non-`mut` aggregate parameter is E0304, like `p[i] = …`). The on-disk format is frozen and
  **headerless: raw contiguous little-endian elements**, no magic and no length prefix, so a blob
  written by one backend is byte-identical to the other's. Contract: the open is attempted **first**,
  so a missing or unopenable path is `-1` even for `len <= 0`; a read then transfers
  `n = min(len, (file_size - byte_off) / sizeof)` elements (a trailing partial element is ignored,
  `buf[n..]` left as allocated; a negative offset is `-2`) and returns `n`, or `-2` on a mid-read I/O
  error; a write creates/truncates and returns `len` (`-1` create failed, `-2` write error). Native
  reads land **straight in the destination** (the original reader staged a second full-size copy).
  The six original dtypes keep their `wukong_rt_{read,write}_<T>` symbols; every other dtype and every
  offset read lowers to the dtype-coded `wukong_rt_{read,write}_dt` — **no new MIR op** — and the
  interpreter runs the same planning code (`wukong_runtime::fileio`), so interp == native
  (`tests/run/io_roundtrip_{f32,f64,i32,i64,i8,u8}.wk`, `io_missing_file.wk`, `io_partial_read.wk`,
  `io_typed_dtypes.wk`). A `u8` read from a file now decodes to the same canonical slot value as a
  stored `u8` in the interpreter (it was zero-extended there, so `b == 200` printed 0 under the
  interpreter and 1 natively).
- **Checkpoint readers** — `st_read_<T>(path, name, buf)` (safetensors) and `npy_read_<T>(path, buf)`
  (NumPy `.npy` v1–v3) parse the header in the runtime (strict JSON / Python-literal parsers, no new
  dependency) and check the stored tensor against the destination **at load time**: dtype (a
  `BF16`/`F16` payload widens exactly into `f32`), a `Tensor` destination's rank and every dimension,
  the exact element count, Fortran order refused. Codes `-1` open, `-2` I/O, `-3` dtype, `-4` shape,
  `-5` name, `-6` malformed header; the destination is untouched on every one but `-2`. Runtime unit
  tests cover the parsers and every code; `tests/run/io_safetensors.wk` and `io_npy.wk` assemble files
  byte by byte and print every lane.
- **Counter-based RNG** — `rand_u32` / `rand_f32` / `randn_f32(seed, ctr)` are Philox4x32-10 (key
  = the seed's two words, counter = `(ctr lo, ctr hi, 0, 0)`; `rand_u32(0, 0)` is Random123's known
  answer `0x6627e8d5`), lowered to plain integer MIR (`zext`/`mul`/`lshr`/`trunc`/`xor`), so they are
  bit-identical across backends by construction and accepted in `@parallel` regions; `randn_f32` is
  Box–Muller on the compiler's `log`/`cos`. `rand_fill_f32(buf, seed, ctr0)` is the same stream as one
  runtime kernel. The runtime reference (`wukong_runtime::random`) is checked against all three
  Random123 `philox4x32_10` known-answer vectors, and an interpreter test checks the MIR lowering
  against it over seeds and counters that exercise both 32-bit halves (`tests/run/rand_philox.wk`).
- **Sampling kernels** — `topk_f32(x, vals, idx)`, stable `argsort_f32` / `argsort_desc_f32(x, idx)`
  and `sample_categorical(probs, u)` (inverse CDF over positive, finite weights), each defined by one
  total order (NaN largest, ties to the lower index) so the answer is unique; the interpreter calls the
  same safe runtime functions (`tests/run/topk_sampling.wk`).
- **Radix & char literals**: hex `0xFF` / octal `0o17` / binary `0b1010` integer literals with `_`
  digit separators and type suffixes (`tests/run/radix_literals.wk`), and char literals `'A'` (the
  one-character / `\xHH` / `\u{…}` escapes) typed `char` — a 32-bit Unicode scalar value,
  interconvertible with the integers via `as` (`tests/run/char_literals.wk`, `tests/run/char_type.wk`).
- **String literals**: `"hello"` lives once in a read-only static-data section (`.rodata`), typed
  `*u8`; the escapes `\n` `\r` `\t` `\\` `\"` `\'` `\0` `\xHH` `\u{…}` decode. Each *unique* literal is
  emitted once (deduped by content) and referenced by address via `Op::GlobalAddr`, so a **returned or
  threaded `*u8` no longer dangles** — the pointer stays valid after the callee's frame is gone, and
  the interpreter and native backend agree (`tests/run/string_return.wk`). `print`/`println` of a
  `*u8` (a literal or a `let`-bound string) renders the bytes, not the pointer value
  (`tests/run/string_literal.wk`).
- **`str`**: an immutable UTF-8 byte string, laid out as a `[]u8` fat pointer `(data, byte
  length)`. A string literal adapts to `str` wherever a numeric literal adapts to its annotation
  (and as an `==` operand or a range-index base); `s.len()`, bounds-checked `s[i] -> u8`,
  `for b in s`, `==`/`!=` (length, then bytes), and printing the text (`{:?}` quotes it) all run.
  A `str` can be a parameter, a return value, a `const`, a struct field or an array element
  (`tests/run/str_basics.wk`). Its bytes are never written (`s[i] = b`, `atomic_add(s[i], v)` and `&mut s[i]`
  are `E0304`)
  and it converts to and from nothing (`E0401`). Sema erases `str` to `[]u8` when it publishes its
  types, so lowering, the optimizer and every backend run the existing slice machinery; the new
  lowerings are the byte-compare loop for `==` and the sub-slice.
- **Range indexing** `s[a..b]` / `s[a..]` / `s[..b]` / `s[..]` on a `str` or any `[]T`: a view of
  the half-open range, no copy, checked `a <= b <= len` at run time with the slice index failure
  (`tests/run/slice_range_index.wk`, `tests/run/str_slice_bounds.wk`).
- **Pointers & references**: `&x`/`&mut x` take an address, `*p` loads/stores through it, and a
  pointer parameter threads through calls — address-taken locals correctly stay in memory under the
  optimizer (`tests/run/pointer.wk`). `as` casts bind looser than `*`/unary, tighter than binary
  (`*p as T` is `(*p) as T`).
- **Fixed-size arrays** `[T; N]`: literal/repeat init, indexed load/store, array parameters passed
  by base pointer (out-params). Real kernels run: dot, SAXPY, GEMM, matmul, ReLU, clamp, transpose.
- **Tuples & structs**: `(a, b)` / `Name { f: v, … }` literals, field access `t.0` / `s.f` (read and
  assign), heterogeneous fields with correct padded layout, **and nested aggregates** (struct-in-struct
  to any depth, arrays of structs, an aggregate field deep-copied from a variable). Lowered as a flat
  byte buffer with byte-offset field GEPs (the local's value is its base pointer, like an array; nested
  fields recurse), so the interpreter and native backend agree bit-for-bit with no backend-specific
  aggregate handling (`tests/run/{tuple,struct,struct_nested}.wk`), and nested tuple-field access
  `t.0.1` / `t.0.0.0` plus whole-aggregate assignment `s = other;` both run
  (`tests/run/{nested_tuple_field,struct_assign}.wk`). A tuple/struct also crosses function
  boundaries — passed **in by reference** (base pointer, zero-copy; mutating it needs `mut`, else
  E0304) and **returned by value** via a hidden-pointer (sret) ABI modeled in mir_build, so no
  aggregate ever rides in a register and both backends agree
  (`tests/run/{struct_fn,struct_return}.wk`).
- **Constant-shape tensors** `Tensor[f32, R, C]`: multi-dimensional indexing `a[i, j]` lowers to a
  row-major GEP (the shape-typed surface), so elementwise tensor kernels and tensor matmuls execute
  on both backends (`tests/run/tensor_*.wk`) — and a matmul written in tensor notation dispatches to
  the tuned GEMM kernel (see below). **The shape-typed spelling costs nothing**: a statically-shaped
  contiguous tensor is normalized to its flat row-major index (`a[i, j]` → `a[i*C + j]`) before
  lowering and binds as the buffer it is, so it reaches every kernel recognizer and the
  autovectorizer exactly as `[f32; R*C]` does — the two spellings compile to **byte-identical MIR**
  (`crates/wukongc/tests/tensor_parity.rs` pins this for 2-D elementwise, matmul and a non-square
  rank-3 nest). `tests/run/transformer_block_tensor.wk` is a whole pre-norm transformer block in
  tensor notation with the same GEMM and fused-norm dispatch as its hand-flattened twin. The one
  structural difference that remains: a rank-2 tensor cannot be swept by a single flat index (that
  is the shape check working), so a whole-buffer elementwise pass over one is written as a nest and
  vectorizes per row rather than as a single stream.
- **Symbolic-generic tensor shapes** `fn f<M, N>(a: Tensor[f32, M, N])` **execute** — the capstone of
  the shape-safety story: a shape-generic tensor function *proves* its shapes at compile time (the
  dims are rigid generics in the body, so no shape-lie; see the shape-checking limitation note below)
  **and runs at any per-call size**. The per-call dims are threaded in as hidden leading `i64`
  parameters (the standard "dependent dims become value args" lowering): the row-major index stride of
  `a[i, j]` becomes `i*N + j` with `N` a runtime value, `0..M` reads the dim as a value, and the caller
  supplies each dim from a turbofish (`f::<2, 3>(…)`) or infers it from the argument shapes (a const
  dim, a caller-side symbolic dim, or a decaying array's length). Because the symbolic address
  arithmetic is identical to the constant-shape form when the runtime dims equal the constants, a
  symbolic `f<M, N>` is **byte-identical** to the same kernel written with literal dims — interp ==
  native, `-O0` == `-O{1,2,3}` (`tests/run/generic_shape.wk`). A **symbolic-dim matmul still dispatches
  to the tuned GEMM kernel** (the recognizer already compares strides by dimension identity; once the
  dims are runtime values `emit_sgemm` materializes them from the hidden params), so `matmul<M, N, K>`
  runs on the AVX2/FMA microkernel at any size (`tests/run/generic_shape_matmul.wk`). Ranks 1–3,
  elementwise kernels, and a generic caller forwarding its own symbolic tensors all run — and so does
  a `@parallel` symbolic-shape function (a matmul reaches the multicore `wukong_sgemm_parallel`; an
  elementwise nest runs correctly through the ordinary path, exactly as a constant-shape `@parallel`
  tensor function does).
- **Dtype-generic tensors and per-instance checking.** `Tensor[T, M, N]` with `T` a type parameter
  is monomorphized per element (`axpy$f32`, `axpy$bf16`), `alloc::<T>(n)` allocates a `[]T` per
  instance, and every concrete instantiation of a generic function is type-checked again at its
  types, an error carrying "required by this instantiation of `f` with T = …"
  (`tests/run/dtype_generic_tensor.wk`, `tests/fail/generic_instantiation_bad_op.wk`). A named
  kernel builtin (`matmul`, `sdpa`, …) or a typed file-I/O builtin (`write_f32`, …) takes a
  `Tensor[T, ..]` operand, checked per instantiation: `T` bound to the builtin's element dispatches
  it (`mm$f32` reaches `wukong_sgemm`, `tests/run/dtype_generic_builtin.wk`), any other binding is
  `E0415` / `E0416`. Sharp edges: a `[]T` / `[T; N]` builtin operand is not deferred the same way
  (still the plain mismatch); the decode builtins (`sdpa_kv`, `gemv_bf16`, …) take no
  `Tensor[T, ..]` operand (`E0437`); an instance past the re-check's 512-instance budget is not
  re-checked, so a wrong element there is a `C0001` codegen refusal rather than `E0415`/`E0416`;
  and an in-body recognizer that reads sema's generic-body types declines
  on an instance rather than firing (the instance lowers as ordinary loops). Generic structs
  (`Buf<T>`) are not monomorphized.
- **Type aliases.** `type Real = f32;` and generic `type Mat<M, N> = Tensor[Real, M, N];` (type,
  dimension and `const` parameters, dimension-arithmetic arguments like `Row<N * 2>`) work in every
  type position and across imports (`pub type`). The driver expands each use right after import
  loading (`wukong_driver::loader::alias`) and removes the alias items, so sema, MIR lowering and
  autodiff see only expanded types; errors are `E0326` (cycle), `E0327` (argument count or kind),
  `E0328` (unknown type in the body) — `tests/run/type_alias.wk`, `tests/run/type_alias_import.wk`.
  Sharp edges: an alias names a type only, not a struct literal, pattern or `impl` target, and a
  generic alias as a cast target must be written `x as Mat::<2, 2>`.
- **Dimension constraints.** `where K % 16 == 0, M <= 4096` predicates and `static_assert(cond[,
  "msg"])` are checked at every call whose dimensions are compile-time constants (`E0521`/`E0522`),
  and `const` generic parameters must be `usize` (`E0523`) — `tests/run/where_constraints.wk`. Not
  yet: a constraint is not propagated into a generic caller (a call with still-symbolic dimensions
  is unchecked), and literal dimensions do not yet specialize an instance (`f$K512`), so trip
  counts stay run-time values.
- **Shape arithmetic, views, reshape and extents** run on both backends. A dimension may be computed
  from others (`Tensor[f32, S, 3*D]`, `H*HD`, `(L - K)/S + 1`) and is held in a canonical polynomial
  normal form, so `2*N` and `N + N` are one type, and a call solves a dimension that appears only
  inside one (`2*N = 8`, `N = 4`) — `tests/run/tensor_dim_arith.wk`. Indexing with fewer indices than
  the rank is a zero-copy **view** of the remaining axes (`m[i]` is a row; `m[i][j]` is `m[i, j]`; `for
  row in m` walks rows), lowered as the address of the view's first element, so no recognizer can
  mistake it for an element read — `tests/run/tensor_views.wk`. `x as Tensor[..]` reshapes a
  contiguous buffer when the element count is provably equal — `tests/run/tensor_reshape_heads.wk`.
  `t.dim(k)` / `t.len()` read extents, including a `?` axis of a tensor parameter, whose run-time
  extent the caller passes as a hidden argument (and which binds a callee's generic dimension when
  the tensor is forwarded) — `tests/run/tensor_extent.wk`. Not yet: a strided
  view (a column, a transpose, a `lo..hi` sub-range) — a tensor value carries no stride vector, so
  views are prefixes of the leading axes of a contiguous buffer; a reshape or a view of a `?`
  dimension's *count*; a `?` extent of a tensor returned by a call (a clean `C0001`); and kernel
  recognition over a *derived* inner extent — a `Tensor[f32, S, H*HD]` nest lowers through the
  general strided path, because the GEMM recognizer's stride matcher only knows literal and
  single-variable extents.
- **Matmul → GEMM dispatch**: the compiler recognizes a matmul loop nest (the `ikj` accumulate and
  `ijk` dot-product forms, including the `nn.Linear` `C = A·Bᵀ` spelling) and lowers the whole nest
  to a tuned register-blocked (6×16), cache-tiled, packed **AVX2/FMA** microkernel in the runtime —
  the way XLA/TVM/oneDNN lower a matmul op. Serial and `@parallel`. Beats gcc/rustc's naive nest
  ~3–3.6× single-thread (~110–120 GFLOP/s ≈ 90% of one P-core's roofline) and up to ~18× parallel on `C = A·B` (~19–26× single-core / up to ~104× parallel on `nn.Linear`), the lead
  growing with size. Dimensions may be compile-time literals **or runtime values** (function
  params/locals): the recognizer checks strides symbolically, so a general matmul function dispatches
  to the kernel, not just fixed-size benchmark kernels. Dimensions may also be **module `const`s used
  directly** (`const D: i64 = 768;` … `a[i*D + k]`), and a row base **hoisted into a local**
  (`let ib = i*K;` … `a[ib + k]`) still dispatches: a pre-lowering canonicalization
  (`wukong_mir_build::canon`, `docs/internals.md`) folds an integer const and forward-substitutes a
  pure integer index local before the recognizers run, so recognition no longer turns on how the
  index happens to be spelled. Nor does it turn on whether the follow-up work is written **into the
  store** or as its own loop: `c[i*N+j] = act(bias[j] + s)` (and the bias-free `act(s)`) fuses to one
  `wukong_sgemm_nt_epi`; a residual read from *another* array, `c[i*N+j] = x[i*N+j] + s`, and a second
  store in the same body, `c[i*N+j] = s; d[i*N+j] = d[i*N+j] * s` (the SwiGLU up-projection folded
  into its gate), each lower to the same `wukong_sgemm_nt` + `wukong_velem_f32` pair the two-loop
  spelling dispatches — bit-identical to it, since the matmul store writes exactly `s`
  (`tests/run/linear_{silu_store,bias_relu_store,residual_src,dual_store}.wk`). An operand may also be a **struct field** — `l.wq[j*D + p]`, the way a
  real model groups its weights — and not just a bare local or parameter: a field's base pointer is
  resolved through the same `kernel_base_ptr` path as any other operand, so the projection GEMMs, the
  fused `bias[j] + s` epilogue and the affine norms' `gamma[i]`/`beta[i]` all dispatch from a struct
  exactly as they do from flat parameters (`tests/run/struct_field_kernel_base.wk`). The two factors
  may even be the **same array**
  (a Gram matrix `A·Aᵀ`, or self-attention `Q·Kᵀ` sharing a buffer) — both sides are read-only. The
  interpreter calls the identical kernel (marshalling its memory), so the two stay bit-exact. The
  **transposed-A weight-gradient** form `C = Aᵀ·B` (`dW = dYᵀ·X`, A stored `[k,m]` with the
  contraction axis outermost) is recognized too and dispatched to `wukong_sgemm_tn`, which transposes
  A once then reuses the same NN microkernel — so the training backward pass leaves the scalar nest
  (`tests/run/matmul_tn.wk`). The **bf16/f16 mixed-precision** `nn.Linear` (`[bf16]`/`[f16]` inputs
  widened `as f32`, f32 accumulate) likewise dispatches to `wukong_sgemm_{bf16,f16}_nt` — a lossless
  widen prepass then the same tuned kernel — ~25× the idiomatic bf16 C (`tests/run/linear_{bf16,f16}.wk`),
  and its **fused FFN epilogue** (`act(A·Bᵀ + bias)`) folds to `wukong_sgemm_{bf16,f16}_nt_epi` — bias +
  activation in the GEMM writeback for free (`tests/run/linear_{bf16,f16}_ffn.wk`; see epilogue fusion below).
  The mixed-precision **weight-gradient** `C = Aᵀ·B` (`dW = dYᵀ·X`, the training backward) is recognized
  too → `wukong_sgemm_{bf16,f16}_tn` (widen prepass → the f32 `wukong_sgemm_tn`), closing the half
  training-backward gap where the naive nest loses to both the un-vectorizable widen and the column-strided
  A reads (`tests/run/matmul_{bf16,f16}_tn.wk`).
- **Batched matmul → per-head GEMM dispatch**: a matmul nest wrapped in a batch loop, with each index
  carrying a per-batch base offset (`x[h*S*D + i*K + k]` — the shape of **multi-head attention**, one
  matmul per head), also dispatches. The recognizer peels the offset off each flattened index (it
  must be invariant in the matmul's own `i,j,k`) and the kernel call GEPs each base pointer by it, so
  every head runs the tuned microkernel instead of a scalar nest. Both the `Q·Kᵀ` and `P·V` matmuls of
  an MHA forward dispatch (see `tests/run/{batched_matmul,multi_head_attention}.wk`).
- **Matrix transpose → cache-blocked kernel**: the nest `for i { for j { dst[j*R+i] = src[i*C+j] } }`
  dispatches to a `B=32` cache-blocked `wukong_transpose_f32[_parallel]`. The naive transpose writes
  `dst` with stride `R` (a cache miss per element for large `R`) and gcc/rustc do not loop-tile it at
  `-O3`. Against a peer that is ALSO 32×32 blocked (which is what a competent C programmer writes,
  and what the xbench peer does since 2026-08-04) the blocked kernel was published as a **single-core
  tie** (1.00–1.03×), with the win in the `@parallel` form, ~4.8–7.1× vs 1-thread C and ~1.0–1.1× vs
  an all-core OpenMP peer running the same blocked nest. **⚠ Both figures are unverified as of
  2026-08-05:** they were measured only at 1024²/2048², power-of-two strides at which the peer's
  column-major write stream aliases in L1 — a property of the size, not of anyone's codegen — and
  they no longer reproduce on the current tree. `wukong_xbench` now sweeps 1000², 1031² and 1100×950
  alongside them and reports the two regimes separately; the numbers here await a fresh AC+full-power
  round. See BENCHMARKS.md "Matrix transpose". This is a memory-bound layout op (attention score /
  weight-layout transposes). A permutation, so bit-exact (`tests/run/transpose_f32.wk`).
  **bf16/f16** transposes dispatch to the same blocked kernel at 16-bit width (`wukong_transpose_u16`, one
  kernel for both — a transpose moves the raw bits) for the half-precision KV/attention layouts (`transpose_bf16.wk`).
- **Column reduction → SIMD colsum kernel**: the column-outer nest `for j { for i { s += x[i*N+j] }; out[j]=s }`
  (the bias gradient `db = Σ_batch dY`, batch sum, reduce-along-axis-0) dispatches to
  `wukong_colsum_f32[_parallel]`, which streams `x` row-major and accumulates eight columns at a time into
  a cache-resident `out[]`. Each column sums in `i`-ascending order, so it is bit-exact
  (`tests/run/colsum.wk`). **This family is a measured LOSS, not a win** (corrected 2026-08-04): the
  ~29–47× single-core figure previously published here was measured against a C peer written
  column-outer, the worst loop order for a row-major axis-0 reduction. Against the natural row-outer
  nest `for i { for j { out[j] += x[i*N+j] } }`, which gcc auto-vectorizes, the kernel is
  **a tie at best and 1.8× slower at worst** (15 of 16 measured rows are losses; the >L3 shape is a
  consistent ~1.65× loss across two rounds), and `@parallel` only ties single-threaded C. The
  recognizer fires correctly — the gap is in the kernel, and closing it is open work. The **max**/**min**/
  **abs-max** down the same axis (`out[j] = max/min_i x[i,j]`, `max_i |x[i,j]|` — per-channel quant stats,
  axis-0 max/min pooling, and the symmetric int8-quant scale `amax_j`) dispatch to
  `wukong_col{max,min,maxabs}_f32[_parallel]` (first-row seed + `_mm256_max_ps`/`_mm256_min_ps` fold, abs
  via sign-mask `andnot`); the same strided gap (gcc/rustc stay scalar — `fmax`/`fmin` are non-associative)
  gives ~34–50× single-core / ~37–107× `@parallel` (`tests/run/colmax.wk`, `colmin.wk`, `colmaxabs.wk`).
- **Softmax backward → fused dot+apply kernel**: the batched nest `for r { let s=0; for j { s += y·dy };
  for i { dx = y·(dy−s) } }` (the gradient through a row softmax — attention + classifier training) folds to
  `wukong_softmax_bwd_f32[_parallel]`, which delegates the per-row dot to the bit-exact `sreduce` (8 lane
  accumulators) then applies `y·(dy−s)` 8-wide. gcc/rustc keep the dot's *accumulation* scalar (a serial
  `vaddss` chain), so Wukong wins ~1.0–2.0× single-core (the apply is already vectorized in both) and
  ~5–6× `@parallel` (rows across cores). The dot reassociates (the reduction exception), so the differential
  gate is bit-exact while the cross-language check is a tolerance (`tests/run/softmax_bwd.wk`).
- **Activation backward → 256-bit transcendental gradient**: a `for i { dx[i] = act_backward(x[i],
  dy[i]) }` loop (`act` ∈ {`silu`,`gelu`,`sigmoid`,`tanh`,`elu`,`softplus`} — the gradient through every
  FFN/attention/gate nonlinearity in training) folds to one `wukong_vmath2_f32` call with a new two-input
  op code, fusing the upstream `dy·` multiply into `dy·act'(x)`. The derivative is itself a transcendental
  (`silu'`/`sigmoid'`/`softplus'` fold a sigmoid, `gelu'`/`tanh'` a tanh, `elu'` an exp on the negative
  branch — an `expf` C/Rust keep scalar), so the 256-bit kernel wins **~3–12.5× single-core, ~5.5–25×
  `@parallel`** vs scalar C (`tanh_backward` the largest, `elu_backward` the most modest — only x≤0 folds
  `exp`). Pure elementwise (no reduction) → the kernel is bit-identical lane-for-lane, so the differential
  gate is trivial (no reassociation exception); scalar twin / AVX2 / inlined-MIR fallback share one op
  sequence (`tests/run/{silu,gelu,gate,elu_softplus}_backward.wk`). relu/leaky backward are not added (a
  select on x>0 — gcc vectorizes those, so they'd only tie).
- **Transformer building blocks compose**: a transformer FFN (`gelu(x·W1ᵀ)·W2ᵀ`), scaled
  dot-product attention (`softmax(Q·Kᵀ)·V`), **multi-head** attention (the batched per-head form) and
  its **causal** (decoder/autoregressive) variant, 2D convolution (im2col + matmul), and a full
  pre-norm (Llama-style) transformer block all lower with their matmuls dispatched to the GEMM kernel
  and their softmax/GELU/RMSNorm vectorized — and run bit-identically on both backends (see
  `tests/run/{ffn_block,attention,multi_head_attention,causal_attention,conv_im2col,transformer_block,rmsnorm,log_softmax}.wk`).
- **Autoregressive decode primitives (CPU)**: `sdpa_kv` attends `sq` query rows against a KV cache of
  `skv` rows at a position offset, with grouped-query / multi-query heads and explicit row strides
  (`wukong_attention_kv_f32`; decoding one token at a time is bit-identical to a causal prefill), and
  `kv_append` grows the cache. `gemv_bf16` / `gemv_f16` multiply by half-precision weights without
  ever materializing an f32 copy; the loop nest `y[j] = Σ x[p]·(w[j*K + p] as f32)` over a bf16/f16
  weight is recognized into them, and a recognized bf16/f16 `nn.Linear` with one to four rows now
  takes the same streaming path (bit-identical to the old widen-then-GEMM result). `quantize_q8_0` /
  `quantize_q4_0` pack weights into llama.cpp's `Q8_0` / `Q4_0` blocks and `gemv_q8_0` / `gemv_q4_0`
  run the weight-only quantized GEMV (per-call `Q8_0` activation quantization, exact integer block
  dots). Every kernel validates its shape against its operands at run time (a bad geometry traps like
  a failed `assert`) and has a bit-exact AVX2/portable twin pair (`tests/run/decode_*.wk`). Not
  measured yet: no throughput claim is made for any of them.
- **SIMD auto-vectorization**: straight-line elementwise loops (incl. branchy ones via
  if-conversion) lower to 128-bit vector ops, 4×-unrolled, with a scalar remainder — automatically,
  on the native backend (only a **half-open, unit-step** range vectorizes: an inclusive `a..=b` or a
  `step s` loop lowers as the plain scalar loop — correct, just unaccelerated — and the
  recognized-kernel cascade declines those two forms for the same reason). Where the body fits the
  raw-AVX2 recipe (f32, f64, i32 or i64 lanes; up to eleven streams; register pressure within the 16
  YMM registers) a loop is instead emitted as **true 256-bit AVX2** by a raw machine-code path
  (`crates/wukong_codegen_cranelift/src/avx2.rs`, VEX-encoded via `iced-x86`) that sidesteps
  Cranelift's 128-bit CLIF cap (`WUKONG_P4_NO_256` disables it). The loop may stream over `[T; N]`
  arrays, `[]T` slices and constant- or symbolic-shape tensors alike; a slice or `Ptr`-bound tensor
  gets a one-time run-time bounds and overlap check that sends the loop scalar if it fails. An
  **elementwise** loop takes the kernel at any trip that holds one register group, literal or
  runtime (large output-only streams switch to non-temporal stores inside the kernel). A
  **reduction** takes it at a trip ≥ 2048 (`VEC256_REDUCTION_MIN_TRIP`) — decided statically for a
  literal trip and by one run-time compare for a runtime `n` — and computes the language's
  **canonical reduction tree** either way, so the form never changes the sum (`docs/internals.md`,
  "The canonical reduction tree"). `s = s + x + y` is the reduction `s += x + y`. The raw-AVX2
  emitter is **host-gated**:
  `wukong_mir::host_supports_vec_kernels()` requires **x86-64 Windows with AVX2 + FMA** (the emitted
  body hardcodes the Win64 rcx/rdx/r8 argument registers, and there is no in-kernel fallback).
  `mir_build` tests that predicate alongside `WUKONG_P4_NO_256`, so off-gate no recipe is built and
  the loop keeps the 128-bit CLIF form — the same result-identical fallback the kill-switch forces.
  `assemble_kernel` still refuses on the same predicate, but that is now a backstop: it fires only if
  the two ever disagree.
  saxpy/poly/relu/relu6 vectorize.
- **General loop vectorization** (`wukong_opt::vectorize`, `-O2`): a counted loop the front end did
  not vectorize — a `while`, an inlined body, a declined `for` — widens at MIR level in 128-bit
  vectors. The group width follows the body's **narrowest** lane and wider values are split into
  parts, so mixed-width bodies widen: an `i8`/`u8` load computed in `f32`, a saturating `f32 -> i8`
  quantize, an `i8` dot into an `i32` accumulator, an `f32` stream summed in `f64`. Complete
  interleave groups `x[k*i + r]` (`k` up to 4 — complex multiply, RGB pixels, in-place RoPE) load and
  store through lane shuffles; a shift by a uniform amount widens at every integer width; an `i8`
  multiply goes through `i16`. A step with no vector form — a gather, an integer divide or
  remainder, a per-lane shift amount, `f64 -> i64` — is **replicated** lane by lane while the rest
  widens (`tests/run/vectorized_{mixed_widths,interleave,replicate}.wk` print every lane).
  **Min/max and conditional reductions** widen too: `m = max(m, x[i])` / `min` / `fmax` / `fmin`,
  a branch-spelled `if x[i] > m { m = x[i] }`, and a guarded accumulate `if c { s = s + x }` (either
  arm, a constant addend such as a count included). Integer ones up to 32-bit lanes keep
  lane-parallel partials; float ones (and 64-bit min/max) fold lane by lane in index order with the
  loop's own compare/select, so NaN and `±0` behave exactly as in the scalar loop
  (`tests/run/vectorized_{minmax,guarded}_reductions.wk` print every prefix answer). **Search
  loops** with an early `break` widen too, when every load before the exit provably stays inside its
  object (a local `[T; N]` table, a `[T; N]` or static-shape tensor parameter under a constant bound,
  or a slice searched to its own length): any lane leaving hands the group back to the scalar loop,
  which stops at exactly the same iteration (`tests/run/vectorized_search_loops.wk`,
  `tests/run/vec4_search_param_extent.wk`). **One-armed memory** in an `if` is predicated per lane
  instead of declining the loop: a store only one arm makes, and a load only one arm makes, touch
  each lane's own element only where that lane took the arm and a private scratch slot otherwise
  (`tests/run/vectorized_predicated_arms.wk`). **Argmax / argmin** (`if x[i] > m { m = x[i]; at = i;
  }`, strict compare, ties keep the lowest index, NaN never updates) and **FindLastIV** (`if c { last
  = i }`) widen with per-lane partials and an exit fold by iteration order, bit-exact with the scalar
  loop (`tests/run/vec_argmax_reductions.wk`, `tests/run/vec_find_last.wk`); a `>=` argmax keeps
  the LAST tie (`tests/run/vec4_argmax_last_ties.wk`). **`if / else if` chains** and small nests of
  conditionals are flattened with path predicates — a chain storing on every path is one vector store
  of nested selects, one with a path that stores nothing is predicated per lane — and a **bounds
  check inside an arm** (a one-armed store over a checked `[]T` parameter) is discharged once, by a
  guard that every index of the vector loop passes it (`tests/run/vec4_else_if_predication.wk`).
  **Countdowns** (`while i > 0 { i = i - 1; .. }`, `while i >= 0 { ..; i = i - 1 }`) widen with
  each part loaded and stored as one vector reversed by a shuffle (`tests/run/vec4_reverse_loops.wk`).
  A runtime trip whose `end - start` wraps runs no vector group, and every fold returns its seed
  bit for bit below one group (`tests/run/vec4_zero_trip_reductions.wk`). A **scatter**
  (`o[q[i]] = v`: a permutation or unpermute, an `index_put`, an embedding-row write, a sparse-to-dense
  fill) widens everything around it and stores lane by lane in iteration order, so the last writer
  of a repeated index is the scalar loop's, when nothing else in the loop can touch the scattered
  object — a `@noalias` destination or a local table (`tests/run/vec5_scatter.wk`). Still scalar: an
  `else`-arm argmax update and 64-bit argmax values, stores to different places in one nested
  region, nested guarded reductions, and in a countdown a search, an argmax or an interleave group;
  and among scatters a read-modify-write histogram (the `scatter_add` recognizer's), two scatters
  into one object, a scatter beside any other access of its own object, a **predicated** scatter
  (one inside an `if` arm), a **checked** scatter (its bounds check is on data, not the counter), and
  an unpromised destination the analysis cannot prove apart. There is no hardware gather or scatter:
  each lane is a scalar access (AVX2's `vgatherdps` is not emitted, and AVX2 has no scatter).
  The per-lane accesses are scalar under Cranelift (no masked vector memory op is emitted yet). A
  loop needing more than eight runtime alias checks stays scalar, and so does one whose estimated
  vector group costs more than its `W` scalar iterations (`WUKONG_VEC_COST=0` lifts that). A two-deep nest whose inner loop carries a recurrence or
  a strict-order reduction (an EMA or IIR per row, a hash, a column sum) is **unrolled and jammed**
  instead: four rows run side by side as four independent chains, each bit-identical to the scalar
  loop, when the rows provably touch disjoint memory (`tests/run/unroll_jam_nests.wk`). Its speed-up
  is not yet measured; deeper nests, triangular inner bounds and a nest whose inner loop the
  bounds-check versioning split stay as written.
- **Straight-line SLP** (`wukong_opt::slp`, `-O2`, after unrolling): `W` stores in one block to
  consecutive elements of one object whose values are one lane-wise operation tree pack into one
  128-bit vector store — `f32x4`, `i32x4`, `f64x2`, `i64x2` — over vector loads, splats and lane
  builds: the register tile a complete unroll leaves, hand-written `vec4`/quaternion/complex-pair
  code, small fixed-array kernels (`tests/run/vec5_slp_bundles.wk`). Bit-exact by construction
  (lane-wise ops only, no reassociation, no operand swap). Still scalar: **alternate opcodes**
  (`+ - + -` in one bundle, LLVM's alt-shuffle), **`i8`/`i16` lanes** and `bf16`/`f16`, **`select`,
  compare and cast nodes** (a mask is not a value across backends, and a cast changes the lane
  count), a bundle whose lanes are not in one block or whose stores are not contiguous, a chunk
  shorter than one full vector, a store moved past a possibly-aliased access, a call that can name
  the stored or loaded storage, or a barrier. Reductions and horizontal ops are not formed, and the
  cost model is untuned: no speed claim without a same-run release A/B against `WUKONG_NO_SLP`. The
  vector accesses it emits on frame slots reach `gpu-native` too, which only the device suite checks.
  A gradient compile does not run it on the forward it differentiates: autodiff has no lane-wise
  VJP rules for vector loads, stores and arithmetic, so the whole program of an `--emit=grad`,
  `--train` or `--emit=obj --grad-of` compile stays unpacked (its generated backward is packed).
- **FMA contraction**: a float `x + y*z` becomes one fused multiply-add (`Op::Fma`, a hardware
  `vfmadd`), on both the scalar and vector paths; the interpreter mirrors it with `mul_add`, so the
  two backends stay bit-identical.
- **Reduction vectorization**: a float reduction `s = s + x[k]*y[k]` / `s += ..` lowers to
  vector-lane accumulators (independent FMA chains) + a horizontal reduce + scalar remainder, turning
  the latency-bound serial sum into a throughput-bound one. `dot` runs ~2.7× faster than serial C.
  `fmax`/`fmin` reductions (`m = fmax(m, x[i])`, softmax's row-max) vectorize the same way.
- **`@parallel` reduction → multicore reduction kernel**: a reduction loop in a `@parallel` function
  (dot `x[k]*y[k]`, ssd `(x[k]-y[k])²`, or the unary sum `x[k]`) is **dispatched to a deterministic
  multicore reduction kernel** (`wukong_sreduce_f32_parallel`), spreading the stream across cores to
  aggregate memory bandwidth — `dot@parallel` ~7.9×, `ssd@parallel` ~8.6× faster than single-threaded
  C. The parallel sum is bit-identical to the serial one (fixed-size chunks independent of core count
  + ascending partial combine), so the differential oracle holds.
- **Transcendental intrinsics**: `sqrt`/`rsqrt`/`cbrt` (the root family; `cbrt` for LAB color /
  variance-stabilizing transforms), `exp`/`log`/`exp2`/`log2`/`exp10`/`log10` (≈1-ULP `f32`
  minimax polynomials; base-10 for decibel/log-scale features), `expm1`/`log1p` (Kahan-stable `eˣ−1` / `ln(1+x)`, ≈1-ULP near 0),
  `pow` (= `exp(y·log(x))`), `atan2`/`hypot` (the two-arg geometry pair — full-circle angle, overflow-safe
  2-norm; a `for j { out[j]=f(x[j],y[j]) }` loop dispatches to the 256-bit two-input `wukong_vmath2_f32`),
  `erf` (Abramowitz–Stegun, for **exact** GELU
  `0.5·x·(1+erf(x/√2))`), `sin`/`cos` (Cephes minimax + quadrant reduction, for **RoPE** rotary
  position embeddings), `tan`/`atan`/`asin`/`acos` (Cephes; the inverse trig for angle/geometry/3D-vision
  ops), the activation
  family `tanh`/`sigmoid`/`silu`/`gelu`/`elu`/`leaky_relu`/`softplus`/`softsign`/`logsigmoid`/`mish`/`selu`/`tanhshrink`/
  `hardsigmoid`/`hardswish` (`softsign` a bounded poly activation; `logsigmoid` the stable
  BCE-with-logits primitive), the full hyperbolic family `sinh`/`cosh`/`asinh`/`acosh`/`atanh`
  (the inverse trio for hyperbolic/Poincaré embeddings + the Fisher z-transform), and `fmax`/`fmin` —
  all built from primitive ops both backends agree on bit-for-bit. A pure `out[i] = f(x[i])` loop for
  **any of the 35** single-arg transcendentals (incl. the stable `expm1`/`log1p`) is **dispatched
  to a tuned 256-bit AVX2/FMA kernel** (`wukong_vmath_f32`) — the width Cranelift's general (128-bit)
  vectorizer can't reach; a *composed* use auto-vectorizes the inlined poly at 128-bit. So softmax, layernorm,
  GELU (tanh and exact erf), SiLU/swish, ELU, softplus, mish, tanh, RoPE, and **log-softmax /
  cross-entropy** run on SIMD instead of scalar `libm` — **~2–13× faster** than gcc/rustc's scalar
  `libm` call (which can't vectorize a loop containing it; ~28× across cores under `@parallel`). See `tests/run/{transcendental,softmax,
  layernorm,gelu,elu,leaky_relu,softplus,mish,activations,log,erf,trig,ihyp,atan,log_softmax,ffn_block}.wk`.
- **Math intrinsics on integer operands**: `abs`/`round`/`floor`/`ceil`/`trunc` are type-preserving on
  an integer (integer `abs` = `select(x<0, −x, x)`; rounding an integer is the identity), and `sqrt` /
  the transcendentals promote an integer operand to `f32` — so they no longer emit the float-op-on-int
  MIR that the native backend rejected and the interpreter ran lossily (`tests/run/int_math.wk`).
- **`min(a, b)` / `max(a, b)`** (2026-10-08, `gap/recognizer-coverage`): integer (and float)
  minimum/maximum, a compare in the result type's signedness plus a select, so a `usize` compares
  unsigned; on floats they are `fmin`/`fmax`; the result is the operands' common type, as for `+`.
  Before this a call was `C0001` — which made the ragged-tile spelling `for i in i0..min(i0 + T, M)`
  (the one `examples/matmul.wk` uses) impossible to compile at all (`tests/run/min_max_builtin.wk`).
- **Scalar builtins** (2026-10-08, `gap2/scalar-intrinsics-and-types`): `clamp`; an explicit
  single-rounding `fma`; `popcount`/`clz`/`ctz`/`bswap` (one machine instruction each — a new MIR op,
  `Op::IntUnary`) and `rotl`/`rotr`; `sat_add`/`sat_sub`/`sat_cast::<T>`, `mulhi`, `widening_mul` and
  `overflowing_*`; float bit access (`bits`, `*_from_bits`) and classification (`isnan`, `isinf`,
  `isfinite`, `signbit`, `copysign`, `ldexp`, `frexp`); the special functions `erfc`, `erfinv`,
  `lgamma` (double-precision runtime, ≤ 3 ULP; `lgamma` differentiates through a runtime `digamma`)
  and `logaddexp`; and OCP FP8 conversion on `u8` storage (`f8e4m3_to_f32`, `f32_to_f8e4m3`, and the
  E5M2 pair). All but the bit counts and the special functions are plain compare/select/shift MIR, so
  they agree on every backend by construction. `fmax`/`fmin` on an integer variable is now `E0445`
  (`max`/`min` keep the type); `E0446`/`E0447` reject a wrong operand kind or a mixed-type saturating
  operation (`tests/run/scalar_*.wk`, `tests/run/fp8_convert.wk`).
- **Convolution via im2col + GEMM**: a conv written as an im2col gather followed by a matmul has its
  matmul recognized and dispatched to the tuned GEMM microkernel (the XLA/cuDNN lowering), so Wukong
  runs a 3×3 conv **~1.55× faster** than a hand-written direct convolution in C — and 1.12× *slower*
  than the same nest at `-ffast-math`. See `tests/run/conv_im2col.wk`. *(Corrected 2026-08-04: the
  previous ~6–7× was measured against a peer whose buffers carried no `restrict`, which forced gcc to
  reload the accumulation operands per output pixel and cost it 4.3×.)*
- **Operator fusion**: adjacent same-range elementwise loops (e.g. a linear map then ReLU) fuse into
  one loop when the combined body is dependence-safe; CSE then forwards the intermediate through
  registers rather than memory.
- **GEMM epilogue fusion**: a recognized `nn.Linear` matmul (`C = A·Bᵀ`) immediately followed by a
  bias-add / activation loop over `C` (`C[i,j] = act(C[i,j] [+ bias[j]])`) fuses into one
  `wukong_sgemm_nt_epi` call that folds the bias + activation into the microkernel's C-tile
  writeback — so `C` is written once instead of paying a separate read-modify-write pass over it. The
  saving is a fraction of the C-pass traffic, so it grows as K shrinks: ~1.0× at 512³ (compute-bound,
  no harm), ~1.34× at K=64/N=2048, ~1.65× at K=32/N=4096 — exactly the small-K/large-N projections
  (attention-output, down-projection). Serial **and `@parallel`** — the multicore
  `wukong_sgemm_nt_epi_parallel` is bit-identical to the serial kernel (a fixed-chunk parallel
  reduction), dispatched from a `@parallel` fused-epilogue nest (`tests/run/linear_bias_relu_parallel.wk`).
  The activation set is identity (bias-only), **ReLU,
  GELU, and SiLU** — the transformer FFNs — with **bias optional**, so the bias-free `silu(x·Wᵀ)`
  **SwiGLU** projection (LLaMA/Mistral) fuses too. Both backends call the identical kernel, so it
  stays bit-exact. See `tests/run/{linear_bias_relu,linear_bias_gelu,linear_silu}.wk`. The
  **bf16/f16 mixed-precision** FFN fuses the same way — a half matmul + its bias/activation loop folds
  to `wukong_sgemm_{bf16,f16}_nt_epi` (the widen prepass feeds the identical f32 epilogue), so the
  mixed-precision transformer FFN gets bias + activation for free (`tests/run/linear_{bf16,f16}_ffn.wk`).
  The **residual projection** `x = x + act(x·Wᵀ + bias)` (the transformer skip connection) also folds
  to `wukong_sgemm_nt_epi`, with **beta = 1** so the kernel accumulates `act(x_residual + A·Bᵀ + bias)`
  in its writeback. The accumulate store would otherwise block the matmul recognizer and drop the whole
  nest to a scalar loop, so this recovers the full GEMM dispatch + the fused residual/bias/activation
  with no backend change (`match_matmul_residual`, `tests/run/linear_residual{,_relu}.wk`).
- **int8 quantized `nn.Linear` → VNNI GEMM (+ fused dequant)**: a `u8`×`i8`→`i32` matmul nest in the
  `C = A·Bᵀ` spelling dispatches to `wukong_i8gemm_nt[_parallel]`, and when it is immediately followed
  by a per-channel dequant loop the two fold into one `wukong_i8gemm_nt_deq[_parallel]` call — the
  scale applied in the GEMM's own writeback. The fused form drops the i32 accumulator, so it fires
  only when the accumulator is a local of the block that nothing names after the window and whose
  address was never taken before it (a `let p = &mut c;` or a `let`-bound closure writing `c` keeps
  the unfused GEMM, `tests/run/closure_i8_accumulator.wk`). Whole-function and statement forms both
  fire, and the integer accumulate is exact mod 2^32, so interp == native bit-for-bit and `-O0` ==
  `-O3` (`tests/run/{i8_linear,i8_linear_dequant,i8_linear_parallel,dequant_perchan}.wk`). The
  per-tensor/per-channel `absmax` that produces the symmetric quantization scale is the
  column-reduction family above.
- **`@tile` strip-mining** on a half-open unit-step `for i in start..end`: `@tile(B)` on the function
  (or one loop) expands into full tiles plus a remainder loop, before the recognizer cascade.
  `@tile(BM, BN, …)` queues widths for nested `for` loops in source order (`tests/run/lang_tile_saxpy.wk`).
  A `@tile` the compiler cannot apply is **`E0309`**, not a silent no-op: an unsupported loop form
  (`0..=n`, a stepped or open range, `for x in xs`, a `_` binder), a width that is not a positive
  integer literal (a bare `@tile`, `@tile(0)`, `@tile(rows)`) — including in a *second* `@tile` on the
  same statement, which used to be read by nobody — a width the queue still holds when the body ends,
  and a lowering that carries no queue at all (a whole-function recognizer interception, or the
  whole-function `@parallel` outliner: `@generic` is what keeps the loops tileable). A refused width
  is not consumed, so the function-item queue cannot shift onto the wrong loop, and the queue is
  rewound between a tiled loop's two body copies so one source loop takes one width
  (`tests/fail/tile_bad_form.wk`, `tile_zero_width.wk`, `tile_non_integer_width.wk`,
  `tile_never_applied.wk`, `tile_on_parallel_fn.wk`, `tile_on_recognized_kernel.wk`,
  `tile_second_on_stmt.wk`).   `@tile` is accepted only where something reads it — a `fn` item or the `for` statement it schedules
  — and is **`E0306`** on a `struct`, a parameter, a `let` or a `while`, the same placement code
  `@generic` carries off a `fn` item or a loop (`tests/fail/tile_on_while.wk`).
- **`shared [T; N]`** and **`sync_block()`**: shared address space on `alloca`; cooperative barrier
  inside `@parallel` only (`E0308` otherwise). Static shared size above 48 KiB is `E0307`. CPU:
  shared = stack `alloca`, barrier = no-op/fence. gpu-native: PTX `.shared` + `bar.sync` in
  `lower.rs` only (`tests/run/lang_shared_transpose.wk`, `lang_tiled_transpose.wk`). gpu-native smoke
  on the RTX 4050: all three of `lang_tile_saxpy.wk`, `lang_shared_transpose.wk` and
  `lang_tiled_transpose.wk` run and match the interpreter. Getting there took two separate fixes —
  `cvta.shared` addresses must use generic `ld`/`st` rather than `ld.shared`/`st.shared`, and the
  Phase 3 vector qualifier had to be declined (it was emitting a `.global` 16-byte access on a
  `.local`, 8-aligned frame slot; see the Phase 3 note below).
- **`@parallel`** functions execute across CPU cores (rayon runtime); the per-core chunk is itself
  vectorized. The interpreter runs the same range sequentially, so results stay differential-equal.
  `@parallel` is a *request*, and a shape it cannot prove safe **falls back to serial lowering with no
  diagnostic** — the answer stays right, the parallelism silently does not happen. The whole-function
  form applies only when every parameter is an array (captures are the parameters, passed by pointer),
  the body is exactly one `for` statement over an unlabelled half-open unit-step range starting at
  literal 0 (`0..n`; no label, no `..=`, no `step`), and no `return` or loop-level `break`/`continue`
  escapes the parallelized loop — a `break`/`continue` inside an *inner* loop is fine. Each of those
  used to change what the loop means (a dropped label lowered `break 'l` to `unreachable`: an
  interpreter error, a native SIGILL), so declining is the fix. The mid-function region outliner adds
  one more decline: a modelled body-local that is reassigned inside an inner loop, whose snapshot is
  not flow-sensitive. Verify with `wukongc --emit=mir -O2 f.wk` and look for a `_parallel` symbol or a
  `wukong$par$` region.
  `@parallel(grain = G)` is consumed (each worker claims runs of at least `G` iterations; any other
  `@parallel` argument is `E0306`), a loop with no `grain` gets an implicit one on the CPU backends
  (`ceil(2^14 / body MIR size)` for a straight-line body, 1 for a body with an inner loop or a call;
  `MIN_CHUNK_OPS` is a starting point, **not measured**), any number of regions per program is outlined (the old 32-region
  pool is grown on demand), and the thread count is set with `--threads=<n>` / `WUKONG_NUM_THREADS`.
  The runtime's `_parallel` kernels fork only when rows x columns x a per-element cost clears one
  shared work threshold (`WUKONG_PAR_MIN_WORK`); the threshold is anchored to the existing
  velem/vmath crossovers, **not yet measured per family**. Every such fork honours the
  `WUKONG_KERNEL_DYN=1` claim-queue A/B knob (default off: rayon's split, unchanged); whether the
  claim queue wins on uniform rows is **unmeasured**. GEMM `mc`/`nc` follow the probed L2/L3
  (`WUKONG_GEMM_MC`/`_NC` override), proven bit-neutral; the per-host values are **not re-measured**
  off the reference part. Debug builds check every runtime C entry's documented extents and
  aliasing at its top (`safety::entry_law`). There is still no user-written
  cross-thread combine (no atomics, no `@reduce`): a loop-carried accumulator declines to serial.
- **`@generic`** on a function turns the kernel recognizers **off for that function**: none of the
  four probe sites (whole-function interception, the per-loop `match_*` cascade, `lower_block`'s
  statement windows, the vectorizer's elementwise probes) fire inside it, so its loops compile as
  written — while auto-vectorization, loop fusion, FMA contraction, `@parallel` outlining and the
  optimizer all still apply, as do the builtins you call by name. It is per function or per loop (on a
  `for`/`while` it covers that loop and everything nested in it, `tests/run/generic_loop_optout.wk`),
  not per module
  (`tests/run/lang_generic_saxpy.wk`, `lang_generic_naive_gemm.wk` each hold the same body twice and
  the MIR then holds exactly one kernel call), and it is `E0306` anywhere but a `fn` item or a loop.
  `wukongc --remarks=dispatch` prints, at each statement a recognizer replaced, the kernel symbol it
  became, so dispatch is no longer visible only through `--emit=mir | grep wukong_`. The
  whole-program equivalent, `WUKONG_NO_RECOGNIZE`, is a **measurement instrument** rather than a user
  feature (see `docs/internals.md`): recognizers are otherwise gate-blind, so this is the only way to
  compile and time the honest generic lowering of the same program. Autodiff refuses to run under the
  instrument, by name.
- **`@strict_fp`** on a function (and **`--fp=strict`** for the whole program) evaluates its float
  arithmetic in source order with one IEEE rounding per operation: no FMA contraction (`x + y*z`
  stays a rounded multiply and a rounded add, in scalar code, vector bodies, 256-bit recipes and
  scalar remainders), float reductions keep their sequential fold, and the kernel recognizers are off
  (it implies `@generic`). Elementwise loops and integer reductions still vectorize. Per function,
  survives inlining in both directions, identical at every `-O` level and on both backends
  (`tests/run/strict_fp.wk`, `strict_fp_cli.wk`; the default contraction rule, including "the LEFT
  product of `a*b + c*d` is fused", is pinned by `tests/run/fp_contract_rules.wk`). `--fp=strict`
  is refused with `--emit=grad`/`--train`/`--emit=obj --grad-of`. The language guide's "Floating-point semantics" section
  is the user-facing contract.
- Unknown or misplaced `@attributes` are **`E0306`**, not silence: the name is checked against the
  closed v1 set (`@parallel`, `@generic`, `@tile`, `@extern`, `@strict_fp`, `@inline`, `@simd`,
  `@align`, `@export`), so a typo like `@parellel` no longer compiles to a program with none of the requested
  behavior. The four discarded attributes stay discarded wherever they parse.
- Intrinsics `print`/`println`/`assert`. Every scalar prints at its own type, identically on every
  backend and in a linked executable: an `f32` (and an `f16`/`bf16`) prints its shortest round-trip
  digits at `f32` precision, an `f64` at `f64` precision, a `bool` as `true`/`false`, a `char` as its
  UTF-8 glyph (`tests/run/print_scalar_types.wk`). An aggregate prints structurally, like Rust's
  `{:?}` (`tests/run/print_aggregate.wk`); a value with no printed form (a raw pointer, a
  data-carrying enum) is `E0448` instead of an address that differs between backends.
- **Formatted output**: `printf` / `eprintf` / `sprintf` with a compile-time-checked format string
  (Rust's `format!` spec grammar; `E0449` malformed, `E0450` argument mismatch), `print_raw`,
  `eprint` / `eprintln`, and the raw byte writers `write_stdout` / `write_stderr` / `write_text`
  (`tests/run/printf_format.wk`). A `str` argument prints its text. 🟡 The `cc` fallback link has no
  formatting runtime, and a `sprintf` buffer is still a `[u8; N]` (nothing builds a `str` at run
  time).
- **Process interface**: `arg_count` / `arg` / `arg_i64` / `arg_f32` (program arguments after
  `wukongc --run prog.wk -- …`, or a linked executable's own `argv`), `env`, `read_stdin_u8` and
  `exit(code)` (which returns through every caller on the JIT, like a failed `assert`).
  `wukongc -` reads the program from standard input and `wukongc -e <source>` takes it inline
  (`tests/run/process_exit.wk`, `crates/wukongc/tests/process_io.rs`). 🟡 No interactive REPL yet,
  and no line-at-a-time stdin read.
- **`--run` streams stdout** on the interpreter and the native JIT — per line on a terminal, in
  64 KiB blocks into a pipe — instead of holding everything until the program ends, so a training
  loop shows its progress and output printed before a runtime failure is already out. The GPU
  backends still collect their output and print it after the device run.
- **Source-text diagnostics.** A Unicode lookalike of ASCII punctuation or whitespace (`−` U+2212
  for `-`, `×`, `≤`, fullwidth brackets, curly quotes, no-break/zero-width spaces; ~50 entries) is
  **`E0105`**, naming the code point and the ASCII character it imitates, and the lexer reads it
  *as* that character, so a paste from a paper costs one error, not a lexer error plus a parser
  cascade. Any other non-ASCII character stays `E0101`, now naming the code point (`α` (U+03B1))
  instead of `\u{3b1}`, with a spelled-out suggestion for Greek letters. Identifiers stay ASCII:
  Unicode (XID) identifiers need a symbol-mangling decision because PTX must be ASCII.
- **Parser recovery.** A failed `E0200` names the whole set of tokens the parser tried ("expected one
  of `,` or `)`"), is anchored just after the previous token when the parser had already moved on to
  the next line (where the missing `)` or `;` goes), and labels the unclosed opener of a missing
  `)`/`]`/`}` from a delimiter stack kept in step with every consumed token. A token the lexer
  already rejected never draws a second parser error. Not yet: rustc's indentation-based "this
  delimiter might not be properly closed" guess, and resynchronizing at the next statement after a
  bad one (the parser still recovers token by token).
- **Warnings (lints).** The compiler's first warnings (`wukong_sema::lints`, run after a clean
  sema): `W0001 unused_variables`, `W0002 unused_mut`, `W0003 unreachable_code` (divergence judged
  exactly as the definite-return check judges it), `W0004 unused_result` (`out[i] == x;` written
  for `=`), `W0005 dead_code`, `W0006 unwritten_output` (a `mut` aggregate parameter never stored
  to), `W0007 unknown_lints`, `W0008 unused_imports` (found by the import loader's module-scoped
  resolution, in every file of the program) and `W0020 unreachable_patterns` (a `match` arm every
  value of which an earlier unguarded arm already matches). Levels come from rustc-style `-A/-W/-D <lint>` (`-D warnings`) and
  `@allow/@warn/@deny(<lint>)` on a function, parameter or statement, innermost first. Warnings
  print after errors, so `tests/fail`'s first-diagnostic contract holds; `tests/warn/` pins each
  lint's exact output (`crates/wukongc/tests/warn.rs`), and `corpus_is_warning_free` holds every
  `tests/run` and `examples/` program to zero warnings. Mechanical fixes travel as structured
  suggestions with exact byte ranges (`W0030`'s `;`, `W0008`'s deletion, `E0304`'s `mut ` for an
  assignment to an immutable binding or parameter, the resolution codes' did-you-mean and
  `Type::f` / `()` edits). Not yet: a label on every caret — sema's shared `Sema::error` helper
  still renders an empty one at about 260 sites — and a fix for `E0304`'s other forms (a builtin
  or `mut self` method writing through a non-`mut` parameter, a store through a tensor view).
- **Statement terminators.** `;` stays optional where it always was, but leaving it out is now
  **`W0030 missing_semicolon`** (Rust's exemptions: after a block-like statement, and before the
  closing `}` for anything but a `let`), and a `(`/`[` — or a `-` touching its operand — at the
  start of a line, which silently continues the expression above (`acc = a[i]` then `(b + c)` is
  a call), is **`W0031 ambiguous_continuation`** (only where a statement could begin; `- y` with a
  space stays silent). The parser hands both out on a separate channel
  (`parse_module_tokens_with_lints`), so no caller asserting a clean parse changes. `E0210` is
  reserved for making a missing `;` an error. Not yet: imported files' parser lints (the loader
  still uses the lint-free entry point), and `defer` statements' terminator.
- The optimizer (`-O0..-O3`), backed by CFG and dominator analyses: whole-program **inlining**
  bottom-up over the call graph (non-recursive callees before their callers, scored by a cost model),
  **mem2reg** (alloca → SSA), constant folding, algebraic simplification, CFG cleanup
  with block merging, **jump threading** of constant short-circuit edges (and, at `-O2`, of any
  edge that decides a block's branch -- a constant argument, or a compare implied by a dominating
  compare of the same values -- through a bounded copy of the block) and **branch folding** into
  one `and`/`or` branch, **tail-recursion elimination** (plain and integer-accumulator, `-O1`+; a
  fully tail-recursive function becomes inlinable at `-O2`), a late **`switch` formation** from
  equality chains, dead/trivial block-parameter elimination, DCE, dominator-tree CSE with load
  forwarding, DSE, **loop-invariant code motion**, **MIR loop fusion** of adjacent same-range loops
  after inlining (`-O2`; also at a non-negative read-behind distance, with first-loop live-outs, and
  over proved-disjoint parameter and `@noalias` slice bases, and over bodies that still branch -- an
  `if`/`else` of stores, up to 8 blocks -- when that costs no loop the vector body the vectorizer
  would give it, directly or after a later unswitch or split),
  **loop unswitching** and
  **induction-variable range splitting** of guarded loop bodies (`-O2`, also past a bounds-check
  trap or `break`, on guards derived from the counter -- `i + 1 < n`, the tile edge
  `jo + j < n`, `(i as u64) < m` -- once they are proved not to wrap, and on loops whose calls sit
  in one guarded arm, `if dbg { print(i) }`, whose call-free copy then widens), and **partial loop unrolling** (4×, with a
  wrap-safe guard and a remainder loop; it never reassociates, so a float reduction keeps its exact
  serial accumulate — measured 28% on an integer reduction, 10% on an elementwise store loop, 4% on a
  float reduction). Guarded by an `-O0`-vs-`-O{1,2,3}` differential
  test plus two layers of MIR verification: a per-pass verify-each that names the offending pass,
  `#[cfg(debug_assertions)]` so it runs in tests and CI but is compiled out of a release build; and a
  whole-program verify in the driver before **every** backend entry — `--run`, `--emit=llvm-ir`,
  `--emit=obj`, `--emit=exe` — which is what guards a release compiler (`--emit=mir`/`--emit=mir-high`
  verify *after* dumping on purpose, so a broken program still prints the MIR that explains why).
  Across the run suite and kernels it removes ~42% of IR ops
  (~48–54% on the heavy transformer/GEMM kernels) and runs ~1.5–2.5x faster than `-O0`.
- Compile-time self-reporting and build integration: `--time-passes` (per-stage table on stderr,
  with per-optimizer-pass and Cranelift codegen/object-write splits and the heaviest MIR functions;
  one JSON line under `--error-format=json`), `--time-trace=<f>` (Chrome trace-event JSON) and
  `--depfile=<f>` (Make rule naming the root and every imported file). `--run`/`--emit=exe` compile
  only what `main` reaches; `-O0` native code is built at Cranelift `opt_level=none`, and the JIT
  compiles functions in parallel. Not yet: several artifacts from one `--emit=a,b` invocation (each
  stage is still its own compile), per-function timing in the trace, or a parallel optimizer.
- Optimizer debugging from the command line: `--passes=<list>` runs any list of the optimizer's
  function passes and program steps, and `--print-after=<pass|all>` prints the MIR after each run of
  a pass. Each `-O` level is such a list, held byte-identical to `optimize` by a driver test. Sharp
  edges: neither works on a gradient compile (the replica has no autodiff roots), `--time-passes`
  does not split the optimize stage per pass on that path, and the cleanup sweeps inside `unroll`
  and `slp` print only as part of their step.
- Typed textual MIR: `--emit=mir-text` prints MIR that `wukongc <file>.mir` reads back, verifies,
  optimizes (`-O`, `--passes`) and prints, round-trip-checked over the whole run corpus. Sharp
  edges: a vector-kernel recipe has no textual form (dump with `--target-cpu=x86-64-v2`), and a
  `.mir` input only prints MIR — it cannot be run, linked or emitted as LLVM IR or an object yet.

## GPU backend (NVIDIA, behind `--features gpu`) — RTX 4050 (`sm_89`) and H100 (`sm_90a`)

> **Device scope (2026-08-06, extended 2026-08-09; H100 subsection added 2026-08-11):** every figure
> in this section **except those in the Hopper subsection below** was measured on an
> **NVIDIA RTX 4050 Laptop GPU** (Ada, `sm_89`, **20 SMs**, 6 GB, **~192 GB/s**, power-capped
> ~30–50 W) under **Windows/WDDM**, with only the peers that box can host: cuBLAS / IMMA / cuBLASLt
> and cuDNN through the redistributable DLLs, NVRTC-compiled CUDA-C, PyTorch in **eager** mode
> (Triton does not install on Windows), and **no CUDA toolkit** — hence no CUTLASS, FlashAttention,
> Marlin or vLLM peer. These are properties of that instrument; **do not extrapolate them to
> datacenter parts.** The datacenter retarget, including re-measurement against the peers a Linux
> cloud box can build, is `GPU_RETARGET_PLAN.md`.
>
> **Three families here need more than a scope note** — `BENCHMARKS.md`'s standing index has the full
> text:
>
> 1. **"Beats PyTorch at every S" is against EAGER PyTorch and is not re-earned against the real
>    bar.** `torch.compile` (Inductor+Triton) is the framework bar and is native on Linux; the
>    tooling to build and verify it landed 2026-08-09 under `tools/cloud/peers/`.
> 2. **The CUDA-graph launch-overhead multiples are Windows/WDDM numbers** and should be expected to
>    shrink on the Linux driver before the GPU changes at all.
> 3. **The long-context flash loss is diagnosed as structural "on 20 SMs"** — a premise that
>    dissolves at 108/132 SMs, with the direction there unknown; and the *short*-context occupancy
>    wins rest on 20 SMs being easy to fill and may **invert** on a larger part.
>
> **Retarget status (Phase 0 + Phase 2 landed on `main`).** The backend is no longer fused to this
> card: PTX headers come from `wukong_codegen_gpu::ptx_target` at per-family floors (`sm_80` for the
> Ampere-legal majority, `sm_89` only where fp8 genuinely needs it, `sm_90a` for the wgmma family),
> device identity is probed once into `Gpu::target()` instead of assumed, fp8 is capability-gated
> before any module load, the cubin and autotune caches are device-keyed, and dynamic shared memory
> is plumbed end to end. What that changes for *this* section is only scope, not numbers: the 4050
> behaviour is pinned byte-identical, and every figure below is still a 4050 figure.

### Hopper (`sm_90a`) — what runs, and what is measured (2026-08-11)

A second GEMM family targets **H100** specifically: `wgmma` + TMA, warp-specialized (one producer
warpgroup plus two consumers), 4-stage SMEM ring, optional 1×2×1 B-multicast cluster, fused
`st.global.v2.f32` epilogue — `crates/wukong_codegen_gpu/src/ptx_wgmma.rs`, emitted at an `sm_90a`
module target. It is **capability-gated**: on any non-Hopper device those rows skip rather than
falling back to a different kernel, which is why the 4050 gates above stay green without it.

Measured standing on an **NVIDIA H100 80GB HBM3** (132 SMs, Linux container, cuBLAS with f32 output
as the peer), full tables and a log citation per number in `BENCHMARKS.md` → *GPU backend (NVIDIA
H100 80GB HBM3, `sm_90a`)*. **All of it is iteration-grade**: the rounds ran in a container whose
user is refused `nvidia-smi -lgc`, and `GPU_RETARGET_PLAN.md` §6.3 classes container rounds as
*iteration* data and only locked-clock root-VM rounds as *publication* data — every round records the
refusal itself (`[clock] lock: refused (…); running unlocked`), and five of them also stamp
themselves `ITERATION data, not publication data` in a banner (`BENCHMARKS.md` names which, and why
the K-sweep and the four non-GEMM rounds carry only the `[clock]` line). The label is the docs',
applied uniformly because the premise is. The rounds substitute a before/after drift gate for the lock;
read every percentage below as provisional pending a locked-clock VM re-run.

- **95.3% / 88.6% / 92.3% of cuBLAS f16 (f32 out)** at 2048³ / 4096³ / 8192³, **97.3%** on the GPT
  FFN down-projection, **73–80%** on the wide-N up-projections; bf16 within ~2 points everywhere the
  shapes overlap. **1024³ f16 is refused** — the *peer's* own run-to-run floor was ±15.47% against a
  ±5% bar, so no number was minted for it.
- Copy bandwidth **2922 GB/s = 87.2% of the 3352 GB/s spec peak**; the ≥90% milestone is not met.
  *(Single-shot round, outside the instrument — no twin, no floor, no publish gate.)*
- **Fused int8 GEMM+dequant beats the cuBLAS GEMM+dequant chain 1.08× at 1024³ / 1.15× at 2048³**
  (the first outright peer win on this part) and **loses at 4096³, 0.79×**. *(Also single-shot and
  outside the instrument, though same-run and oracle-gated; a repeat under it is owed.)*
- **Sharp edge:** the 4050's int8 tile/occupancy tuning **loses ground** on Hopper — the shipped
  dispatcher holds a conservative **67.7% / 71.4% / 47.9%** of cuBLAS int8 GEMM at 1024³/2048³/4096³
  there against 86–88% / 96–105% / ~70% on the 4050. **RETRACTED (2026-08-13):** the sharper "22–52%
  of cuBLAS IMMA against 96–105%" contrast published here compared two different kernels — the H100
  figure came from a hand-placed `_smdb` entry that is retired on the 4050 (~44–53% of IMMA there) and
  is dispatched on neither part, so it was never a reading of the shipped Hopper int8 kernel, which
  has still not been timed against IMMA. `BENCHMARKS.md` carries the retraction. Treat every other
  Ada-tuned threshold in the section below as unproven on Hopper until it is measured there — and
  check, before quoting any of them, that the bench behind it times the *dispatched* entry.
- At the time of those rounds the kernel had **no `wukongc` flag or recognizer route** — every
  2026-08-11 number was measured through `wukong_codegen_gpu`'s own device benches, not from a `.wk`
  program. The route has **since landed** (see the offload section below): on a Hopper part a
  recognized `C = A·Bᵀ` now reaches `gemm_nt_wgmma`, with a decline-to-existing-path fallback. No
  H100 number has yet been measured *through* that route.

*Measured and promoted, not yet re-measured as a rule:* the campaign's scheduling wave (CTA raster on
the cluster index, persistent clusters with a continuous ring, a per-shape tile dispatcher, wait-depth
drain arms) was measured as sweep **arms** on 2026-08-12, and three of the four are now the **shipped**
dispatch rule (`ptx_wgmma::wgmma_dispatch_dtype`, both 16-bit dtypes). As arms they moved the suite
mean from 81.9% to 94.1% of cuBLAS f16, with 8192³ at 102.0% and the wide-N up-projection at 101.6% —
the campaign's first shapes at or above the peer. **Two levers were deliberately NOT promoted, on the
measurement:** the wait-depth drain arms moved nothing beyond ~1 point anywhere, and the `GROUP_M = 32`
raster landed strictly worse than 16.

Three scope notes, because the promotion changes what the numbers above describe:

- **The percentages in this section predate the promoted rule** and were measured under the retired
  `M × N ≥ 8e6` split, which the 2026-08-12 round showed shipping the losing arm at four of the seven
  benched shapes. They stand as what that configuration did; they are not the shipped configuration's
  numbers any more.
- **The rule itself is unmeasured.** Its arms were measured individually; nothing has yet run the rule
  end-to-end. `wgmma_vs_cublas` and `wgmma_bf16_vs_cublas` now run the rule rather than a fixed row, so
  the next H100 visit re-measures it.
- **Sharp edge:** the promotion *narrows* the Hopper fast path. The retired rule always returned a
  128×256 row; the new one can decline, and six reachable `(tile, schedule)` signatures have no module,
  so contiguous bands of tall-skinny shapes now take the pre-Hopper f32 launcher — correct, and much
  slower. The set is swept, enumerated and pinned by a device-free law so it cannot grow silently;
  `docs/gpu/derive/WAVE3_PROMOTION_HANDOFF.md` lists the modules that would close it.

A GPU backend, `wukong_codegen_gpu`: being a compiler, it **emits PTX text** and **driver-JIT-loads
it via `cudarc`** (`cuModuleLoadData` — the driver's built-in PTX→SASS JIT, so **no `nvcc`/`ptxas`/CUDA
toolkit** is needed to build or run, only the driver). Every transformer op category is a device
kernel, each gated against a CPU reference by a **tolerance** differential (`c·√K·ε`, deterministic
grids) — the CPU↔GPU analogue of the bit-exact CPU gate. Measured honestly on a power-capped 6 GB
mobile 4050 (see `BENCHMARKS.md`):

- **Tensor-core GEMM** (fp16/bf16/fp8 inputs, f32 accumulate): WMMA `m16n16k16` for fp16/bf16
  (**~5–6× the f32 path** same-run; absolute TFLOP/s is clock-bound — ~7× GPU-clock swing — so only
  the ratio is quoted); **fp8 (E4M3)** via hand-laid `mma.sync.m16n8k32` (no WMMA fp8 on
  `sm_89`), validated against an E4M3-rounded reference (`fp8_gemm_matches_reference_within_tol`), not
  bit-exact. Its fragment-reuse multi-tile kernel (`fp8_gemm_mt_ptx`, 2×4 block of
  16×8 tiles per warp) is now the **fastest** tensor-core path — ~2.1–2.4× the naive single-tile fp8 and
  ~1.3–2.3× fp16/bf16 in the same run (dispatch order inside `gemm_nt_fp8`: the cp.async-pipelined
  kernel `fp8_gemm_pipe` when its 128/128/64 tile divides the shape, then the fragment-reuse `_mt`
  kernel, then the single-tile kernel as the fallback for everything else).
- **Fused flash-attention** (online softmax, never materializes the `S×S` scores — the kernel that
  *loses* on CPU): warp-per-query-row + `cp.async` double-buffering — **beats the genuinely-fused
  cuDNN + cutlass fMHA in the causal-S≤512 and fused-RoPE regimes** (and is 3.6–5.0× the unfused
  cuBLAS chain, 205–738× naive CUDA-C), trailing cuDNN only at long context (S≥2048).
- **Fused row norms** (softmax/LayerNorm/RMSNorm, one warp per row), **activations** (SFU), **reductions**
  (deterministic; max bit-exact), **conv2d**, and elementwise.
- **Quantized GEMM**: int8 **W8A8** (`u8`×`i8`→`i32` via `mma.sync.m16n8k32`, `ldmatrix` +
  XOR-swizzle staging, fused per-channel dequant, split-K) and **W4A16** group-wise int4 weight-only
  decode (`lop3` unpack → the same fp16 tensor cores). Because the int8 accumulate is integer, this
  the `i32`-output int8 kernels are gated **bit-for-bit** (`assert_eq!`) against a wrapping-`i32` CPU
  reference rather than by tolerance — the one place the tolerance contract above is replaced by exact
  equality. The f32-output variants (the fused dequant epilogue, W4A16) stay tolerance-gated.
  `QuantWeight` discriminates symmetric vs asymmetric purely on its `zeros: Option<..>` field. Both are
  Rust launch wrappers in
  `wukong_codegen_gpu`: no `wukongc` flag reaches *these* kernels — `--backend=gpu` offloads only the
  five `Accelerator` families, none of them quantized. (A `.wk` int8 nest does reach the GPU by a
  different route: `--backend=gpu-native` lowers `wukong_i8gemm_nt` to its own scalar `mrt_i8gemm_nt`
  device kernel, not to the tensor-core wrappers here.)
- **A whole pre-norm transformer layer runs end-to-end GPU-resident** — RMSNorm → QKV → flash-attn →
  output proj → residual → RMSNorm → FFN(SiLU) → residual, all on device buffers with no host round-trip
  between ops, matching a CPU f64 reference to max_rel 2.7e-4 and **deterministic** run-to-run.
- **Device-resident training and serving exist as Rust APIs in `wukong_codegen_gpu`, not as language
  features.** `train_resident::MlpTrainer` runs forward + backward + a fused AdamW step entirely on
  device buffers (`ptx_optim.rs`, `ptx_autodiff_bwd.rs`); `serving.rs` / `paged_kv.rs` /
  `paged_attention.rs` implement a paged-KV cache, batched autoregressive decode with Orca selective
  batching, a graph-capturable step, and int8 KV. **Neither is reachable from `.wk` source or from any
  `wukongc` flag** — `--backend=gpu` offloads only the five `Accelerator` families and
  `--backend=gpu-native` lowers MIR. They are library surface with their own gates.

**End-to-end `--backend=gpu`.** Built with `--features gpu` (a *cargo build* flag, not a `wukongc`
runtime flag), `wukongc --backend=gpu --run foo.wk` executes the
program through an **offloading interpreter**: the whole program is tree-walked on the CPU (identical
control flow, buffer layout, and every non-kernel op to the oracle), but recognized GEMM / activation
/ reduction / fused-norm calls run on the device via an `Accelerator` seam (`wukong_interp`) the
driver implements with `wukong_codegen_gpu` (`GpuAccel`). With no accelerator — every other caller,
the differential oracle — the path is byte-for-byte unchanged, so the toolchain-free core is
untouched. Coverage of the offload menu is partial and the fallback is silent-but-correct: a
recognized call the GPU wrappers do not cover **declines to the CPU kernel** rather than failing.
Today that means the 6 activation op codes `vmath_supported` lists (relu, exp, sigmoid, tanh, silu,
gelu); the sum/dot/max reductions; softmax/LayerNorm/RMSNorm but **not** log-softmax or L2-norm
(`norm_supported` covers op codes 0..=2, so `tests/run/log_softmax_fused.wk` and `l2norm.wk` run their
CPU kernels); and the fused epilogue only at `beta == 0` with `M`, `N` multiples of 64 and `K` a
multiple of 16. A genuine *device* error is a different thing and is always surfaced as an error —
`Some(Err(..))` — never turned into a CPU fallback. Gated by `gpu_backend_*` tests (driver,
`--features gpu`): each family runs on the interp oracle and the GPU over identical buffers and matches
within tolerance (GEMM tolerance-gated too, not bit-exact — the device reduces K in a different order;
silu ~5e-7, dot ~7e-7, softmax ~3e-8 abs), asserting the offload actually fired.

The plain-GEMM hook carries **one extra dispatch axis, the device architecture**: on a Hopper part
(`cc_major == 9`) a recognized `C = A·Bᵀ` goes to the warpgroup-MMA + TMA family (`gemm_nt_wgmma`),
the kernel the Act-2 campaign measures and which until then had no caller outside its own test module;
on every other capability it takes the pre-Hopper launcher, unchanged. It is `== 9` and not `>= (9,0)`
because `sm_90a` is architecture-*locked*. Every reason the family cannot take a call **that is
decidable before the launch** — the architecture, an unencodable tensor map (`K % 8 != 0`), an odd `N`
under the `v2` store, an output past the epilogue's `u32` index, a CTA grid past CUDA's `gridDim.y`
ceiling, a ring larger than the device's opt-in shared memory, or an operand
outside f16's *range* at either end (the seam converts f32 operands to f16 on the host: past 65504 an
element comes back `inf` rather than merely rounded, and a **row** whose whole magnitude range is
under f16's smallest normal 6.104e-5 comes back as a whole row — or, in `B`, column — of zeros, since
`C[i][j] = Σ_k A[i][k]·B[j][k]` and at ~1e-8 every element of that row is under f16's round-to-zero
point. Overflow declines per element, underflow per **row** on that row's maximum — the granularity
an output lane is actually decided at, and the one at which a quiet row inside a loud operand is
visible at all — so an ordinary `U(-1,1)` buffer never trips it) — is a **decline to
that same existing path**, never an error, so results are unchanged wherever it declines. Two
outcomes are not declines and are named as such rather than implied away: a driver rejection *at* the
launch that is not `UNSUPPORTED` stays a hard error (a GPU failure is never silently answered on the
CPU), and a launch that never retires exits the process with code 70 from the launch wait. Where it does
route, the tolerance band widens rather than the contract: the wgmma family converts both operands to
f16 on the host, so the driver's device gate sizes its band from the route that **actually ran** —
per-route offload counters, not the device's capability, because a Hopper part is eligible at every
shape while declining a real subset of them, and a fallback's f32 result must keep the f32 band
(`WUKONG_GPU_NO_WGMMA=1` forces the pre-Hopper path in the same binary, for a one-build A/B;
`WUKONG_GPU_ROUTE_LOG=1` prints the route and decline reason per dispatch). The
**fused** epilogue hook is deliberately not routed there yet — no wgmma module computes
`act(A·Bᵀ + bias)`, and composing the plain GEMM with a second pointwise launch would be exactly the
unfused chain that fusion exists to delete.

Run the kernel suite with `cargo test -p wukong_codegen_gpu --features gpu` (skips cleanly with no
GPU).

**General MIR→PTX — `--backend=gpu-native`.** Beyond the recognizer-offload path above, a fourth
backend (`GpuLower`) lowers the *whole* program's MIR to PTX, so arbitrary non-recognized kernels run
GPU-side too; an eligible program is fused into a single-block cooperative **megakernel** (one launch,
no host round-trips). It is tolerance-gated against the interpreter oracle and optimization-invariant
(`-O0` ≡ `-O3`), the same contract as the offload path. Coverage is **partial** (UNSUPPORTED ops
skip). The previously documented general-lowering gaps are all **fixed**: the `velem` Hadamard/Div
binary modes and the `norm` log-softmax/L2 ops are implemented in the static PTX device kernels
(`lower.rs`); the `mrt_sreduce`/`mrt_sreduce_coop` device reductions implement the full
`wukong_sreduce_f32` op set including sumabs(9)/absdiff(10), with the coop tree combining them
additively (`parallel_abssum`); float→narrow-int casts **saturate** like Rust `as`/Cranelift instead
of truncating mod 2^w (`float_cast_narrow`: `300.0 as u8 == 255`, `-300.0 as i8 == -128`); and the
megakernel stores pointer values homed in the shared frame **unconditionally** rather than
`tid==0`-guarded — a frame pointer slot is uniform across the SPMD threads, and the old guard left
threads ≠ 0 loading a zero-initialized slot and dereferencing null in non-recognized scalar loops
(the `tensor_1d_kernels@O3` `CUDA_ERROR_ILLEGAL_ADDRESS`).

Two further gaps closed on 2026-08-13, and both are *generic lowering* rather than new kernels — no
`ptx_*.rs` family and no recognizer arm is involved. **Portable SIMD MIR lowers** (lane-scalarized:
`Splat`, `Iota`, `ExtractLane`, lane-wise arithmetic and loads/stores, and vector block params so a
loop can carry a SIMD accumulator), which is what a vectorized function needed to reach the device at
all; `Op::VecKernelCall`, the raw-AVX2 Win64 recipe, is still declined by design. And **`@parallel`
runs grid-stride** in the megakernel — iteration `i` on the thread with flat id
`i mod (blockDim·gridDim)`, per-iteration chunks so adjacent lanes touch adjacent elements — instead
of being declined there and run as one sequential chunk. An outlined body that *prints* still declines
(it is called by every thread, so a record would be appended per thread, not per iteration); the
single-thread path still runs the sequential chunk, since that kernel is launched with one thread.
Both need `-O1`+ in practice, because at `-O0` a loop condition reads a counter out of an alloca and
the data-independent-control-flow gate declines the program first.

**It has now been timed, and it loses.** `bench/lang/` + `wukong_driver::lang_bench` (2026-08-22, RTX
4050, two adjacent rounds) put a generic `.wk` elementwise kernel at **6.6-13.3x slower than the
Cranelift CPU backend** on the same program, and the fastest language-path loop at **1.4-1.6 % of the
part's ~192 GB/s**. Two things account for it, and neither is the kernel menu:

- **`@parallel` does not reach the canonical elementwise loop at all.** `FnLowerer::lower_for` tries
  `try_vectorize_for` *before* `try_emit_parallel_region`, so `dst[i] = src[i] * 2.0 + 1.0` is claimed
  by the CPU SIMD vectorizer and the region outliner is never asked. Right on a CPU; on gpu-native the
  vectorized form runs on **one** device thread, four lanes at a time -- ~22-25x per iteration against
  the same loop outlined. `tests/run/lang_parallel_saxpy.wk` is affected: it outlines its *setup* loop
  and runs its saxpy serially.
- **Every program runs on one CTA** (the block-scoped megakernel entry, see `docs/internals.md`), so
  even the loop that *is* outlined gets 256 threads out of the 20480 this part can hold resident.

So the sentence above is confirmed in the direction that matters: this path loses to the native CPU
backend today, and it loses to the recognized `wukong_velem` GPU kernel by a wider margin still.

**Its two declines are now legible, which they were not.** A decline is coverage this backend honestly
lacks; both of these used to read as either silence or a crash.

- **Why the megakernel declined: `WUKONG_GPU_EXPLAIN=1`.** `fusion::analyze` always built a
  human-readable reason, and nothing printed it except `WUKONG_GPU_DUMP_PTX`, which buries it under the
  whole PTX module. With the knob set, a fallback prints one line naming the reason and the fallback:

  ```text
  $ WUKONG_GPU_EXPLAIN=1 wukongc --run -O2 --backend=gpu-native prog.wk
  gpu-mega: declined (ineligible: no recognized cooperative op (single-thread path suffices)) --
  falling back to the single-thread gpu-native lowering (one device thread; correct, and the
  reference the megakernel is checked against)
  ```

  It is env-gated rather than always-on because the fallback is **correct** and is the common case —
  the single-thread path is the oracle every megakernel run is checked against — so an unconditional
  note would write to stderr on the majority of gpu-native runs.
- **A per-thread `.local` frame past the CUDA cap declines before the launch.** When a program is not
  megakernel-eligible it runs the single-thread lowering, whose frame is a per-thread `.local __frame`,
  and CUDA caps local memory at **512 KiB per thread**. Everything before the launch succeeds at any
  size — `ptxas` compiles the frame, the module loads — so an over-cap frame used to surface as
  `DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")`, which reads as a crash. It is now
  `UNSUPPORTED:` naming the entry, the size and the budget. The quantity is the worst **call path**,
  not the entry's own frame — CUDA charges the budget along the path, so two functions each under it
  can be unlaunchable together, which is reachable at the CLI's default `-O0` — and the maximum over
  paths is taken, not the sum over reachable functions. A recursive graph is not measured (depth is a
  run-time property) and keeps the entry-only test.
- **The megakernel is not exempt from that cap, only its entry is.** The mega entry's allocas live in
  the shared `.global` frame buffer, but every mega *callee* gets an ordinary per-thread `.local`
  frame, so a `@parallel` program with a large per-iteration local is declined to the single-thread
  path rather than dying on a bare `gpu-mega launch failed`.
- **A frame under budget can still be refused, and the note says so.** `ptxas` charges register spills
  to the same per-thread cap and no static walk sees them — measured at 1,280–1,480 bytes for an entry
  with 256 live accumulators, against a 1,024-byte reserve — so a `CUDA_ERROR_INVALID_VALUE` on this
  path is annotated with the frame arithmetic even when nothing is statically over budget. An
  annotation, never a decline: the program may be a few hundred bytes from running.
  The knobs on this backend are therefore `WUKONG_GPU_EXPLAIN`, `WUKONG_GPU_NO_MEGA`,
  `WUKONG_MEGA_GRID` and `WUKONG_GPU_DUMP_PTX` — environment only, none of them a CLI flag.
  `WUKONG_GPU_EXPLAIN` reads its value, so `=0` is off.

Corpus standing is printed by the gates
themselves, over every fixture in `tests/run` (373 today — `ls tests/run/*.wk | wc -l`;
this line has gone stale before): `lower::tests::run_corpus_matches_interp_oracle`
sweeps each program at `-O0` and `-O3`, requires zero mismatches and zero device faults and non-zero
coverage, and reports the rest as honest `UNSUPPORTED:` skips; `megakernel::tests::mega_corpus_matches_oracle`
does the same over the megakernel-eligible subset, counting (program, opt-level) configs and treating a
launch-time `Ok(None)` decline as neither coverage nor a miscompile. Re-run them for the current
numbers — they are a function of the corpus, not a fixed figure. Both now **ratchet** a floor, so a
construct that silently starts declining goes red rather than quietly shrinking coverage, and both
are **host-vectorizer-invariant** (2026-08-09): they normalize through the documented,
result-identical 128-bit CLIF fallback before lowering, because the general 256-bit AVX2 recipe is
Win64-only and `lower.rs` declines the op it produces — so before the fix the *same commit* read a
higher coverage number on Linux than on Windows, and the metric conflated "what gpu-native can lower"
with "did the host CPU vectorizer fire".
Both gates now also **isolate device faults**: a genuine `ILLEGAL_ADDRESS` poisons the CUDA state
**process-fatally** — measured on this driver (RTX 4050, Windows/WDDM), `cuDevicePrimaryCtxReset`
returns Ok but re-retaining the primary context still returns error 700, and cudarc exposes no
non-primary `cuCtxCreate`, so in-process recovery is impossible. *(Scope: that is a measurement of
**one driver on WDDM**, not a CUDA law. The record-and-skip policy it justifies is correct either
way — a lost device must never be reported as passing — but the underlying "recovery is impossible"
finding is on the Phase-1 list to re-measure on the Linux driver, where it may relax;
`GPU_RETARGET_PLAN.md` §2.6.)* `crate::gpu::reset_gpu` therefore
degrades to marking the device **lost**; the gates record the root fault on a loud ledger and
report every later program as NOT RUN (never as passed, never as spuriously failed) — one faulting
program can no longer cascade into ~100 false failures across both gates.

## Automatic differentiation (`wukong_autodiff`)

Reverse-mode autodiff transforms MIR for a single-block forward function returning a floating-point
scalar loss. `--emit=grad` prints its forward and backward MIR; `--train` runs the forward, backward
and optimizer on the selected **interpreter or native Cranelift JIT** backend. The native path
compiles one owning dispatcher and reuses it for forward, backward and generated AdamW calls.
Training requires concrete f32 arrays or dense tensors with checked byte extents. The loop itself
is `wukong_runtime::train::run_training`, which `--train --emit=exe|obj` also bakes into an artifact
(below). Generated-backward performance comparisons remain open. `--grad-of` selects
the loss and `--grad-wrt` its differentiated buffers. Autodiff requires at least `-O1` for SSA
construction; `-O2`/`-O3` retain the selected loss while still inlining calls and optimizing its body.
Generated gradients receive the existing optimizer with an O1 minimum, matching forward
preparation for autodiff. The backward entry is preserved; the original forward and its callees
are unchanged by this second optimization stage. The driver verifies the whole program before
and after backward optimization and verifies the complete training program before execution.
`--verify-each` additionally checks each optimization pass; gradient MIR emission also verifies its output.

**Kernel coverage.** Every runtime symbol `wukong_mir_build` can emit has an explicit autodiff
disposition, enforced by `every_emitted_kernel_symbol_is_differentiable_or_declined_by_name`: 50
forward symbols (26 kernel families, serial and `@parallel`) have VJP rules, and the rest are
listed with the reason a decline reports (index-valued arg-reductions, half-precision storage,
quantized operands, scans, pooling, the training-backward kernels themselves, I/O). The
differentiable families: the whole f32 GEMM family — plain `A·B`, `nn.Linear` `A·Bᵀ`, the
weight-gradient `Aᵀ·B`, the α-scaled attention-score GEMM and the fused
`act(A·Bᵀ + bias)` epilogue (identity / ReLU / GELU / SiLU) — GEMV and GEVM (plain and α-scaled), the
broadcast bias (pure add or any fused activation, in place or not), transpose, column sum / mean /
sum-of-squares, `sreduce_f32` (SUM, DOT, SSD, SUMSQ, MAX, MIN, MAXABS — ties split evenly — SUMABS,
ABSDIFF), `velem_f32` (affine, product and quotient, with identity / ReLU / ReLU6), all 36
`vmath_f32` activation codes, the two-input `vmath2_f32` family (pow, atan2, hypot and the SwiGLU /
GeGLU / GLU gates), `norm_f32` (softmax, log-softmax, LayerNorm, RMSNorm, L2-norm), `norm_affine_f32`,
softmax cross-entropy, log-sum-exp, entropy, KL divergence, rotary position embedding (including a
trainable frequency table), the embedding gather and the fused scaled-dot-product attention
(`wukong_attention_f32`, causal or not — its backward is the FlashAttention recompute: the weights
are rebuilt with the α-scaled score GEMM and the row softmax, then `dV = Pᵀ·dO`, `dS =
softmax_bwd(P, dO·Vᵀ)·scale`, `dQ = dS·K`, `dK = dSᵀ·Q`, all on the tuned GEMM and
`wukong_softmax_bwd_f32` kernels). A source `sdpa(...)` call differentiates from source when every
input is proved apart from the output — each of `q`, `k`, `v` and `out` a fixed-array local or a
fixed-array / static-tensor parameter, and each input pair with `out` two different locals, a local
and a parameter, or two different parameters the call-site alias fact or a `@noalias` promise
separates. The lowering then emits no overlap test and the call is one bare `wukong_attention_f32`
in a straight-line block; `--emit=grad` / `--train` run through this VJP
(`tests/run/sdpa_loss_train.wk`, `crates/wukongc/tests/autodiff_attention.rs`, gated against the
f64 central finite difference in `wukong_driver`'s `grad_model_tests`). `s` and `d` may be runtime
values: their range checks and the operands' extent checks are then straight-line `assert` calls,
which do not block differentiation (`sdpa_loss_with_runtime_dims_matches_f64_finite_differences`);
constant `s`/`d` with static extents only removes those `assert`s. Any other `sdpa` — a slice
operand, a view, a may-alias pair, one storage under two names (`match m { x => sdpa(x, x, x, m,
..) }`) — and every other named kernel builtin that writes an output buffer other than its own
input still lowers behind a run-time output-overlap dispatch (a branch), so a loss that makes such
a call is not single-block and is refused by the straight-line contract; spell the layer with the
recognized loops to train it from source.

**The backward rides the tuned kernels.** The GEMM adjoints are the runtime's own transposed GEMM
spellings (no synthesized transpose); softmax / LayerNorm / RMSNorm backward are the runtime's
fused `wukong_{softmax,layernorm,rmsnorm}_bwd_f32` (so a symbolic-generic row width differentiates);
sigmoid, tanh, SiLU, GELU, ELU and softplus ride the fused `wukong_vmath2_f32` backward codes;
cross-entropy, RoPE and the embedding ride `wukong_xent_bwd_f32`, `wukong_rope_bwd_f32` and
`wukong_scatter_add_f32`; and a `@parallel` forward selects the `@parallel` backward spelling.
Gradient scratch is zeroed heap (`wukong_rt_alloc`, released before the gradient returns), so a
native backward puts no activation-sized buffers on the stack.

**Memory model.** Gradients accumulate across fan-out (a residual connection, a shared weight, a
buffer the loss also reads as a scalar). The in-place spellings the recognizers emit differentiate:
`x = softmax(x)` / log-softmax, `h += r`, `h = act(h + b)` with a pure add or ReLU. A float store
into a buffer whose gradient is live is differentiated as a memory write, scalar reads of local
arrays route to their gradients, and a kernel that overwrites a buffer another kernel (or a
non-constant scalar store) wrote zeroes the earlier writer's adjoint instead of passing it the later
one. In-place LayerNorm / RMSNorm /
L2-norm, in-place activations and in-place products/quotients refuse by name.

Float precision casts propagate adjoints, and integer-to-float casts terminate integer leaves;
float-to-integer-to-float paths decline. **User-function calls differentiate at every `-O` level**:
straight-line helpers are spliced into the loss before the reverse pass
(`wukong_autodiff::inline_calls`); a multi-block or recursive helper on the gradient path refuses,
naming the callee. Mid-function parallel regions and synthesized `VecKernelCall` recipes remain
unsupported, and every unsupported differentiable operation reports a diagnostic. **Second order**
is supported for scalar-path losses: `wukong_autodiff::grad_dot` emits the single-block directional
derivative `<v, ∇L>` and `hvp` its gradient, the Hessian-vector product; second order through
recognized tensor kernels, and differentiating the emitted void `*_grad` itself, decline by name.
There is no CLI flag for second order yet.

**Ahead of time.** `--emit=obj --grad-of=<fn>` exports `<fn>_grad` next to the forward in the
object (C ABI: the loss's parameters, then one caller-zeroed gradient buffer per `--grad-wrt`
entry), so a host training loop links the backward with no compiler at run time. **The training
loop itself is AOT too:** `--train --emit=exe` links the forward, the backward, the fused AdamW
kernels and a kernel dispatcher into an executable whose synthesized `main` calls the runtime's
`wukong_rt_train_main` with a versioned `.rodata` spec blob (buffer sizes, weights, restored
buffers, steps, rate, optimizer, seed, decay, clip, warmup, schedule) and the dispatcher's address.
`wukongc --train` runs the very same `run_training` in process over the very same spec bytes, so
the executable's stdout is byte-identical to it on the interpreter and the native JIT by
construction; `crates/wukongc/tests/train_exe.rs` links five models (one of them printing from the loss) at
`-O0`/`-O2` under SGD and AdamW and compares (20 of 20 on the rustc path). `--emit=obj` writes the object instead. Sharp
edges: the executable reads and writes no files (`--train-init`/`--train-data`/`--train-save` with an
artifact are refused, exit 2), and it needs the rustc + `wukong_runtime` rlib link (the C fallback
has no training loop; refused by name, exit 2) — an `--emit=obj` object too, which also imports
`wukong_rt_run_main` and the print/assert entries of wukongc's generated link shim. What a loss prints now reaches stdout in process as
well (it used to be captured and dropped), so the two agree there too. `--train-opt=sgd|adamw`
selects the optimizer;
`--train-weight-decay` accepts finite nonnegative AdamW decay (default zero) and rejects nonzero
decay with SGD. **`--train` trains on data:** `--train-init=<i>=<path>` loads a buffer,
`--train-data=<i>=<path>` streams a fixed input one batch per update (batch `s mod batches`, read from
disk into the live buffer), `--train-save=<i>=<path>` writes the trained buffer — all raw
little-endian f32, the `read_f32`/`write_f32` layout — and `--train-clip` (global-norm clipping),
`--train-warmup` and `--train-schedule=cosine` set the update policy. The loop no longer copies
every buffer per step: the gradient runs in place, and only the buffers the forward writes
(`wukong_autodiff::written_params`, which knows each kernel's output operand) are snapshotted around
it. The `--train` initializer draws uniformly from `[-0.4, 0.4)` (it was single-signed, `[-0.4, 0)`,
before). Training remains a CLI loop — an in-language `grad_of` is a declared non-goal. Source-driven gradient tests (helper functions at O0–O3, the fused linear-bias-ReLU
layer, the cross-entropy head, a fanned-out reduction), the real-CLI `training_options` and
`autodiff_cli` tests (the latter trains a GEMM + bias model on the interpreter and the native JIT and
requires byte-identical trajectories at O0–O3) and the hand-derived `tests/run/autodiff_train_*.wk`
trajectories cover the CLI — among them `autodiff_train_mlp.wk`, a two-layer MLP (GEMM + bias +
ReLU, GEMM + bias, mean squared error) whose three SGD steps were derived in Python under several
f32 evaluation orders, every printed value at least 131 ULPs from a rounding boundary. Generated-autodiff performance benchmarks remain absent.

A loss whose buffers are **raw `*T` / `*mut T` parameters** now differentiates. It previously could
not: the front end gives a pointer parameter an `alloca ptr` + `store`, so its base reached autodiff
as `load ptr <slot>` and `Vjp::canon` refused to route through a load whose result is a pointer
(*"cannot route gradient for load pointer … (not a parameter or a one-level gep of a parameter)"*).
`mem2reg` now promotes that slot, so the base pointer *is* the parameter value and every access is a
one-level gep off a parameter (`raw_pointer_parameter_grad`, finite-difference-gated). `--train` still
declines on such a loss — a raw pointer carries no extent, so the trainer cannot size its buffers.

## Checked but not yet executed

- **Explicit SIMD vector types** `f32x8` etc. in *source*: parse and type-check; user-written vector
  *values* are not yet executed, and the native ISA path (Cranelift) caps vector SSA at 128-bit
  (`f32x4`), so a wider explicit `f32x8` cannot lower even once execution lands — it must split into
  128-bit halves. (Loop auto-vectorization above is separate and *does* run — 128-bit + unrolling by
  default, and now **true 256-bit AVX2** via a raw machine-code emitter that sidesteps this CLIF cap —
  for f32, f64, i32 and i64 lanes; an elementwise body takes it at any trip of one register group or
  more, a *reduction* at a trip ≥ 2048 (statically or by a run-time trip test); the recognized
  kernels get 256-bit AVX2 via the runtime microkernels.)
- **Attributes:** sema validates the closed set `@parallel`, `@generic`, `@tile`, `@extern`,
  `@noalias`, `@unchecked`, `@strict_fp`, `@inline`, `@simd`, `@align`, and `@export` (plus the
  lint-level `@allow`/`@warn`/`@deny`, legal anywhere, whose unknown lint names are `W0007`); unknown
  names and restricted-placement violations report `E0306`. `@noalias` (parameter-only) is consumed
  end to end — see the alias bullet below. Name acceptance does not promise execution: general
  `@simd`/`@export` behavior remains incomplete (`@align(N)` is consumed on a `let` only), and the attribute spelling of C interop
  (`@extern("C")`) is still discarded — calling C goes through an `extern "C" { … }` block, which
  does execute. `@parallel` executes and consumes `grain = G` (every other `@parallel` argument is
  `E0306`); `@generic` opts out of recognizers. `@tile` performs loop
  strip-mining and reports `E0309` when it cannot apply. Malformed attribute syntax reports `E0207`.
- **Non-contiguous tensor layouts** `.col_major` / `.strided` / `.tiled(N, M)`: they parse (an unknown
  layout name is `E0204`), they are part of the tensor type, and they are enforced across a call
  boundary — passing a `.col_major` tensor to a `Contiguous`-typed parameter is `E0502`. **`.col_major`
  indexing executes** (`a[i, j]` is byte-identical to `a[i + j*M]` on constant-shape tensors).
  `.strided` / `.tiled(…)` indexing remains `C0001`. Only the default row-major
  `.contiguous` layout was the sole executing non-symbolic layout before Phase 3.
- **Coalesced vector global `ld`/`st` on gpu-native: analyzed, not emitted.**
  `wukong_opt::coalesce` classifies each innermost-loop access by IV element stride and marks
  unit-stride `f32` as a `v4` candidate (Phase 3.1, unit-tested). The matching
  `ld.global.v4.f32` / `st.global.v4.f32` emission is **declined**: a 16-byte vector access needs a
  pointer that is both provably in the global address window *and* provably 16-byte aligned, and
  only the first is decidable in `lower.rs`. Frame slots pack at 8 bytes, `CoalesceHint::UnitStride`
  states a stride rather than an alignment, and the vectorizer starts its vector loop at the loop's
  own `start` without alignment peeling. The first version of this path shipped without either
  check and faulted the RTX 4050 two ways (`CUDA_ERROR_INVALID_ADDRESS_SPACE` on `abs.wk`,
  `CUDA_ERROR_MISALIGNED_ADDRESS` on `linear_f32.wk`) — sticky driver faults that zeroed the whole
  gpu-native corpus sweep. gpu-native emits generic scalar lanes instead: correct in every window at
  every alignment, one vector width slower. Re-enabling it is a Phase 3 redesign, and it is gated on
  a bandwidth A/B that has never been taken for this path.

## Planned

- **Language direction (kernel language, not a recognizer catalog).** The GPU SOTA
  campaign is wrapping special-case PTX separately. The plan for making Wukong
  itself a strong ML/DL kernel language — generic gpu-native lowering, consuming
  `@tile` / `@generic` / `shared`, explicit vendor kernels rather than silent
  rewrites — lives in [`docs/LANGUAGE_DIRECTION.md`](LANGUAGE_DIRECTION.md). Phase 2 (`@tile`,
  `shared`, `sync_block`) is integrated; see roadmap
  "Works end to end" for what runs today. The sequenced
  language mission (including AI-editor / vibe-coding) lives in
  [`docs/WUKONG_MASTER_PLAN.md`](WUKONG_MASTER_PLAN.md).
- Fusing arbitrary elementwise chains *under* `@parallel`. (The **parallel fused-epilogue GEMM** that
  was the remaining item here has since shipped — a `@parallel` `act(A·Bᵀ [+ bias])` nest dispatches
  to the multicore `wukong_sgemm_nt_epi_parallel`; see "Works end to end" above.)
- Execution of explicit `f32x8`-typed *values* (they parse and type-check today — see "Checked but
  not yet executed" above). The *general loop* vectorizer already reaches true 256-bit AVX2 for large
  trips via a raw machine-code emitter (see "Works end to end" and "Known limitations"); it is
  *source-level* `f32x8` values that still do not execute.
- A minimal stdlib of reusable functions/collections — the module system it needs (private items,
  `pub`, per-file namespaces, `as` / `{x, y}` imports) now exists. (The language surface once grouped here —
  structs, enums including data-carrying tagged unions, slices, and multi-dimensional indexing
  `a[i, j]` — now runs end to end; see "Works end to end" above.)
- AMDGPU/ROCm device codegen (the NVIDIA PTX path already ships behind `--features gpu`, and
  reverse-mode autodiff already ships as the `wukong_autodiff` crate — both above).

## Known limitations / sharp edges

- **`str` is the first slice of a string type, not all of one.** An unannotated `let s = "x"` is
  still the NUL-terminated `*u8` (a literal becomes a `str` only where a `str` is expected, and an
  `if`/`match` whose arms are all literals is typed from its arms, as with numbers: an annotated
  `let` does not reach into the arms, so the first arm must be a `str` for the later literal arms
  to adapt). There are no conversions between `str`, `[]u8` and `*u8`, no concatenation or
  formatting into a `str`, no character (UTF-8) decoding, no ordering (`<` is `E0433`) and no `str`
  pattern in a `match`. A `str` cannot bind a type parameter (`E0420`: it is erased to `[]u8`
  before monomorphization), and a `str` inside a printed aggregate is `E0448`. A failing range
  START is reported in the index wording with the range end as the length (`s[4..2]` says "the len
  is 2 but the index is 4").

- **`wukongc fmt` refuses what it cannot reprint, and keeps a few comments only approximately.**
  The formatter never changes a program's meaning (every output must re-parse to the identical AST,
  keep every comment and every identifier, literal and keyword; see `docs/internals.md`,
  Formatter), so its limits are layout and coverage, never correctness. A comment *inside* an
  expression moves to the end of that expression's line, and one inside a signature's generics or
  return type moves into the body. Only signatures, the outermost call or struct literal of a statement, and array
  literals wrap; a long binary-operator chain, a nested call or an `if` condition stays on one line
  however wide. A file that writes text the AST does not keep is refused (exit 2, untouched): a
  struct field's or trait method's attributes (`@align(16) x: f32`, `@extern x: f32`), an enum's
  discriminant repr (`enum E: u8 { .. }`), a `pub` before `import` or `impl` (the parser accepts it
  and discards it), and any token inside an `extern { .. }` block that is not part of a declaration
  (the parser skips it silently). The corpus and `examples/` are not reformatted yet, so
  `wukongc fmt --check` on the repository lists files.

- **A `?` argument binds a fixed-extent parameter unchecked.** `fn find(x: [f32; 37])` called with a
  `Tensor[f32, ?]` (or an array that decayed to one) compiles with no diagnostic and no run-time
  length test, though the buffer may be shorter; only another constant length is `E0501`. A shorter
  buffer is outside the defined contract, like every wrong run-time size, and the optimizer may read
  up to the declared extent (`ParamAttrs::extent`: the vectorizer runs a search loop's test a group
  ahead of the scalar loop over such a parameter). So a 10-element buffer searched by a `[f32; 37]`
  function stops at its hit at `-O0` but reads past its end at `-O2` ("vector load out of bounds" in
  the interpreter). A run-time length assertion where a `?` extent meets a fixed one would close the
  gap. The extent is already in hand there, since a `?` axis travels as a hidden argument.

- **Fast-mode float results depend on how a reduction is compiled.** Under the default `--fp=fast`
  an `f32` reduction is evaluated on one of three trees — the source fold (a loop nothing
  vectorizes), the sequential vectorizer's lane partition, or the `@parallel` kernel's fixed
  8,192-element chunks — and adding or removing `@parallel`, or a reduction declining its kernel,
  moves it between them, changing low-order bits with no diagnostic. Each tree is deterministic
  (independent of threads, `-O` level, backend and host). Use `@strict_fp` where bits must not
  depend on the spelling. Also by design: a compound `s += y*z` outside a vectorized reduction is not
  contracted while `s = s + y*z` is; a strict function that wants a single-rounding multiply-add
  writes the explicit `fma(a, b, c)`, which is one rounding in every mode.

- **Stack-slot coloring follows uses, not scopes** (2026-10-12, `gap6/stack-coloring`). The native
  backend lets allocas whose lifetimes never overlap share one frame slot (`tests/run`: 5.8% fewer
  static slot bytes at `-O0`, 4.8% at `-O2`). A slot is live from a use to a later use — around the
  whole loop when it is used inside one — and dead before a counted loop that overwrites every byte of
  it (the `[v; N]` re-initialization of a `let` inside a loop body), so per-iteration scratch arrays
  share at `-O0`. What it does not separate: scratch declared and zero-initialized at the top of a
  function and overwritten later by a 2-D loop nest, a runtime kernel or, at `-O2`, an unrolled or
  fused loop — those slots are live from the top (`examples/llama_block.wk`'s ten scratch arrays
  share nothing); and at `-O1`+ a slot handed to a runtime kernel never shares with one accessed only
  in place, since the two sit in different Cranelift alias regions. An address passed to a user
  function that may keep it, to `extern "C"` code or into a `@parallel` region keeps its slot for the
  whole function. The interpreter (the oracle) keeps one slot per alloca.

- **Generics sharp edges** (2026-10-11, `gap4/sema-soundness`). By design, a `const` generic
  parameter is a compile-time *dimension*, so it is `usize` and nothing else (`E0523` for
  `const N: i32`); and a symbolic dimension of a tensor signature (`fn f<M, N>(a: Tensor[f32, M,
  N])`) is not specialized per call but passed as a hidden `i64` argument, so `M`/`N` and the trip
  counts they bound are run-time values inside the body (literal dimensions do not yet produce an
  `f$M4` instance). A bounded generic body is checked against its bounds even when nothing calls it
  (`E0315` for a method its bounds lack, `E0410` for forwarding `T` to a bound it does not carry), but
  an operation on an *unbounded* opaque `T` (`x + x`) is checked only per instantiation, like a C++
  template — an uncalled body is not rejected for it. A type parameter that appears in no parameter or
  return type is bound by no call, not even by a turbofish; a value of it is refused at lowering
  (`C0001`) where it used to get an `i32` slot. The same holds when the parameter shadows a struct,
  enum, `const` or `fn` of its name. `x.len()` / `x.dim(k)` on an unbounded `x: T` is typed only per
  instance. The call itself lowers as the `i64` extent everywhere. A binding initialized from it
  stays untyped, though, and arithmetic on it is `C0001` until the binding is annotated
  (`let k: i64 = x.len();`).

- **Atomics are integer add only** (2026-10-12). `atomic_add` returns nothing (no fetch-add),
  has one memory order (sequentially consistent), and has no float form: a float sum's bits depend
  on the order of the adds, so a deterministic float combine needs privatized partials (the
  reduction kernels' fixed chunks) instead. `--backend=gpu-native` does not lower it (the program
  is an `UNSUPPORTED` skip, never wrong PTX). A `@parallel` loop that also *reads* an array it
  atomically updates, or reads an array parameter of the same element type the caller could pass
  as that buffer (no `@noalias` promise between them), stays serial; so does an atomic add to a
  field, a pointee or a multi-index tensor element inside the loop. Reading a location other
  iterations are still adding to is a race, as for a plain store.

- **Closures are resolved at compile time** (2026-10-13). A closure is a call argument or a `let`
  initializer; there are no run-time function values, so a closure (or a closure-bound local)
  cannot be returned, stored in a field, element or tuple, or used as an operand, and a function
  type is not a field, element or return type. A closure argument of a call through a trait bound
  is not supported (a trait method cannot declare a function-typed parameter). A slice, `str`,
  tensor or vector cannot be captured (`E0418`; capture a pointer), nor the enclosing function's
  function-typed parameter, nor a `for` counter whose bound is of a generic type (its storage width
  is per instance; copy it to a local first). Inference of an unannotated parameter from the call's
  other arguments never reaches the enclosing statement: `fold3(|a, b| a + b, 1, 2, 3)` binds
  `T = i32` from the literals, whatever the result is assigned to.

- **Loop fusion over branching bodies is bounded** (2026-10-12). A body that still branches fuses
  only up to 8 blocks per loop, with one latch and no exit but the header's test: a `break`, an
  early `return` or a slice bounds check's trap in either loop keeps the pair apart. A two-block loop
  fused into a branching one gives up the 4x partial unroll, which takes two-block loops only. Two
  widenable loops of which either branches fuse only when the fused body still widens, so two
  `if`/`else` store loops (each if-converted to one store address of its own) stay apart. Two
  unwidenable loops stay apart when either branches on a condition unswitching may decide later
  (an invariant flag, or the counter against an invariant, `if i < m`): split alone, its copies
  may widen, while fused, each would carry the other loop's scalar work. The check is
  conservative, so such a pair stays apart even when no copy would widen (arms that divide, say),
  and it gives up the fused store/reload saving there.

- **A call keeps its loop whole unless one guarded arm holds it** (2026-10-12). Unswitching and
  IV-range splitting clone a loop holding calls only when one arm, which the decided condition kills
  in one copy, dominates every call. A call every iteration makes, calls on both arms of the
  branch, or a call under an `and`/`or` of which the condition is only one operand keep the loop in
  one piece, and the call's "may touch any memory" verdict keeps it scalar. The copy that keeps the
  calls stays scalar too.

- **Scalar-type gaps** (2026-10-08). There is no 128-bit integer (`u128`/`i128` literal suffixes
  are `E0431`; Cranelift's x64 backend has no `i128` divide, so the type needs a runtime first), and
  FP8 is storage-only — E4M3FN/E5M2 codes in `u8` with conversion builtins, no `f8` scalar type that
  loads, stores and promotes by itself; packed `int4` and constructible SIMD values are not started.
  The bit-count builtins (`popcount`, `clz`, `ctz`, `bswap`) keep a loop scalar (the vectorizer does
  not widen `Op::IntUnary`), `gpu-native` declines them, and `erfc`/`erfinv`/`lgamma` are one
  scalar runtime call per element at every precision.

- **Decode-builtin gaps.** Only the plain `y[j] = s` spelling of a half-weight GEMV loop is
  recognized (a scaled store `y[j] = s * c` stays a loop), and a quantized GEMV has no loop spelling
  at all — its per-call activation quantization is part of its definition, so call `gemv_q4_0` by
  name; the K-quant family
  (`Q4_K`, `Q6_K`) and a quantized KV cache are not implemented; `sdpa_kv` has no padding mask or
  per-sequence batch dimension, takes the output's stride from the query's, and has no backward
  kernel (autodiff refuses it). A whole greedy decode loop runs as a gated toy model
  (`tests/run/decode_tiny_lm.wk`: bf16 projections, a growing KV cache, GQA attention, a Q8_0 LM head,
  six tokens checked against an f64 reference); the multi-token GPT-2 example over real weights is
  still to be written. None of the decode kernels is measured yet.
- **ML I/O and randomness gaps.** No memory-mapped load yet (`map_<T>`: every checkpoint load copies
  once, straight into the destination) and no GGUF reader (the `Q8_0` / `Q4_0` block layouts exist as
  the decode builtins' packers and GEMVs, and a raw block blob loads with `read_u8`); `.npy` reads
  little-endian arrays only.
  A fill loop `for i { out[i] = rand_f32(seed, base + i) }` is not recognized into `rand_fill_f32`,
  and the loop vectorizer declines it (mixed 32/64-bit widths), so the scalar builtins run one
  element at a time; `randn_f32` has no fill kernel.
  `topk_f32` has no VJP (autodiff refuses it rather than drop the gradient).
  None of these is measured yet.

- **Recognizer spellings still outside the fold** (2026-10-06, `gap/recognizer-coverage`). The
  hoisted-value, GELU-expansion, row-outer column-reduction, out-of-place softmax, hand-tiled GEMM and
  `usize`-counter spellings now dispatch (`docs/language-guide.md`, Attributes), each only where the
  kernel computes what the source computes. Declined on purpose, and therefore still the 128-bit
  generic loop: HuggingFace's literal `0.044715 * v * v * v` GELU and the erf GELU (no kernel computes
  them bit-for-bit — a new `vmath` opcode would); `out[j] = out[j] + x*x` (the source contracts it to
  one FMA, the `colsumsq` kernel squares and adds — write `+=`); among tilings (2026-10-08: ragged
  `min(i0 + T, M)` tails, counted `let i0 = it * T` tiles, runtime extents and tiled contractions now
  dispatch), a counted tile whose point loop is a full `i0..i0 + T` (it names no extent), a tile width
  that is not a literal or a `const` (a function-local `const` counts only when declared in the same
  block as the tiled loop), a contraction tiled around a ZERO-seeded scalar (it overwrites `C` with
  the last tile's partial sum — not a GEMM), and `examples/matmul.wk` itself, whose inner kernel is
  written in explicit `f32x8` intrinsics that lowering does not handle yet; a value binding read by a
  second store, and a float product bound to a
  name (inlining it beside a `+` would contract it into an FMA the `let` rounded away); and a
  plain float fold with no kernel over a `usize` counter (the generic vectorizer would reassociate a
  sum that folds in source order today; argmax/argmin and the low-precision reduction kernels do
  dispatch over a `usize` counter, behind the same run-time test as the maps). The value view and
  the row-outer/out-of-place windows are flat
  (one row); the batched forms keep their existing matchers. Per-site decline remarks ("this nest
  matched GEMM but its operands are not provably disjoint") are not emitted yet: `--remarks=dispatch`
  reports what fired, not why something did not.

- **A view local read before it is assigned is undefined, and the language has no rule that says so.**
  *(Table measured 2026-08-23 on `lang/w2e-recognizer-gate`; sema still has no definite-assignment
  rule as of 2026-10-06, but the table has not been re-measured.)*
  A `Tensor[T, …]` local is a view: its slot holds a base pointer, not storage. Declaring one and
  assigning it later is legal and works (`tests/run/tensor_local_assign.wk`), and declaring one
  *without* ever assigning it is load-bearing in the shape-check fixtures, which never reach a
  backend. But there is no definite-assignment analysis, so **reading one first is accepted and then
  does three different things**:

  ```wukong
  fn main() -> i32 { let y: Tensor[f32, 4]; y[0] = 1.0; print(y[0] as i32); return 0; }
  ```

  | backend | `-O0` | `-O2` |
  |---|---|---|
  | interp | exit 1, `error: expected a pointer` (no source location) | exit 0, prints `1` |
  | native | exit 139 (SIGSEGV, no diagnostic) | exit 139 |

  The MIR shows the cause directly: `v0 = alloca ptr` then `v2 = load ptr v0` — a load of a slot
  nothing ever stored. The interpreter's own `-O0`/`-O2` disagreement is an opt-invariance violation
  the gate cannot see, because until now no corpus program used a `Tensor` local at all.
  The interpreter half of this is fixed: slot 0 is now a reserved null (`wukong_interp::NULL_SLOT`),
  so the unstored handle is a genuine null at every level and the interpreter reports
  `null pointer dereference (store)` at both `-O0` and `-O2` instead of disagreeing with itself.
  Native still faults on the null store. Giving the local backing storage is not the fix: whole-value
  assignment to a view rebinds the handle, and a buffer-typed slot would have to copy. The remaining
  fix is a **definite-assignment rule** for pointer-backed locals, which is a language feature and
  belongs with the Phase 7 work, not a patch at the lowering.

- `bf16` **and** `f16` are both real 2-byte storage (round-to-nearest-even), f32 compute, with a full
  symmetric mixed-precision op suite (reductions, max-family, axpby, activations — see above) **and a
  mixed-precision GEMM**: a bf16/f16 `C = A·Bᵀ` `nn.Linear` nest dispatches to `wukong_sgemm_{bf16,
  f16}_nt[_parallel]` — a lossless widen prepass (`<<16` / F16C, ~1/n of the GEMM) feeding the tuned
  AVX2 f32 microkernel (`tests/run/linear_{bf16,f16}.wk`). On this AVX2+F16C box (no AVX-512-BF16) the
  GEMM itself runs in f32, so it is a *footprint* feature on the FLOPs — but it still beats the
  idiomatic bf16 C **~25× single-core** (that C can vectorize neither the inline `bf16→f32` widen nor
  the serial reduction), ~10× vs a hand-optimized widen-then-tile bf16 C. A half-precision *output* on
  the streaming ops **now ships** — the narrowing store feeds all-half `wukong_axpby_{bf16,f16}_out` /
  `wukong_vmath_{bf16,f16}_out` twins (~2× on the memory-write-bound axpby, its write traffic halved;
  a footprint win on the call-bound activations); see "Works end to end" above.
- The *general* vectorizer emits **128-bit CLIF by default** — Cranelift's vector ISA still rejects a
  256-bit `f32x8` SSA value (verified empirically on Cranelift 0.124, pinned as the
  `cranelift_still_rejects_f32x8`/`p4_probe_vec256_ops` tripwire tests). An f32/f64/i32/i64
  elementwise loop is instead dispatched to a **raw-AVX2 256-bit machine-code emitter** at any trip
  of one register group or more; a **reduction** takes it at a **trip ≥ 2048 elements**, tested
  statically for a literal trip and at run time for a runtime one
  (`crates/wukong_codegen_cranelift/src/avx2.rs`, VEX-encoded via `iced-x86`;
  `WUKONG_P4_NO_256` disables it) — the same way the GEMM/vmath runtime microkernels reach 256-bit,
  and precisely *why* that raw emitter exists (Cranelift can't legalize the wider lane). That path is
  **platform-gated**: `wukong_mir::host_supports_vec_kernels()` requires x86-64 **Windows** with
  runtime AVX2 + FMA3, because the Win64 argument registers (rcx/rdx/r8) are literal in the emitted
  bytes and there is no in-kernel dispatch. Both ends consult that one predicate — the vectorizer
  declines to emit a kernel call off-gate (so the loop stays 128-bit, and everything still compiles and
  runs), and `assemble_kernel` refuses as a backstop rather than emit non-executable code.
  `WUKONG_P4_NO_256=1` forces the 128-bit CLIF path and is **result-identical** for the supported
  recipes. Elementwise FMA contraction is pinned by `p4_kill_switch_is_result_identical`; reductions
  compute the canonical tree — four accumulators of one 256-bit register's lanes, a fixed combine
  order, a fixed seed fold — at both widths, for every lane type and every addend: register pressure
  changes only how the kernel schedules its copies, never the accumulator count. That is pinned by
  `p4_kill_switch_is_result_identical` (reductions at literal trips ≥ 2048 and runtime trips on both
  sides of the run-time trip test), by
  `reduction_partition_is_independent_of_width_and_threshold` with cancellation and exceptional-value
  oracles, and by the hand-evaluated `tests/run/reduction_canonical_tree.wk`. The portable form
  represents each logical accumulator as two 128-bit vectors; its register/MIR cost against the old
  four-vector form remains unmeasured, and so does the reduction crossover for low-pressure bodies
  (2048 was measured on the highest-pressure body).
  The AST loop vectorizer requires a disjointness proof for differently named buffers paired
  with a write. Potentially aliased parameters retain scalar execution; named parameters alone
  do not imply non-aliasing. `parallel_dependence.wk` checks the shifted in-place recurrence
  that previously returned the same wrong answer on both backends after unsafe SIMD lowering.
  Whole-function parallel outlining uses the shared region dependence proof as well.
  The MIR alias analysis is conservative for parameters unless the source promises otherwise:
  `@noalias` on a parameter (C's `restrict`) makes it disjoint from every other parameter, gives a
  promised slice's elements their own provenance, and — when nothing derived from it escapes — keeps
  it apart from untracked pointers and callees. The loop dependence test, the vectorizer's runtime
  alias guard, Cranelift's alias regions (Heap/Table) and pure fat-pointer loads, the LLVM IR
  (`ptr noalias`), the GEMM-class recognizers (array/static-tensor parameters only) and the
  `@parallel` region outliner all consume it. Sema rejects provably aliased call sites and re-pointing
  a promised view parameter (`E0316`); an unprovable violation is undefined, as in C. Without the
  attribute nothing changes: distinct local allocations and call-site proofs remain the only
  disjointness facts.
- **What the general MIR loop vectorizer still leaves scalar** (`wukong_opt::vectorize`): a loop
  stepping by anything but `+1` or `-1`, and a negative stride on an ascending counter
  (`x[n - 1 - i]`); a strided access outside a *complete* interleave group (a gap, or a stride above
  4), or any interleave group in a countdown; a scatter (a store through a data-dependent index)
  that something else in the loop may touch — a read-modify-write histogram (`h[q[i]] += x`), two
  scatters into one object, a scatter beside another access of its object, a scatter in an `if`
  arm, a checked scatter, or an unpromised destination (when nothing else can touch the object, a
  `@noalias` destination or a local table, the loop widens around the store, which runs once per
  lane in iteration order: 2026-10-12, `gap5/vectorizer-scatter-slp`); and a gather whose
  index is bounds-checked (a checked slice keeps a data-dependent check no guard can cover — the
  gather widens from an array, or under `@unchecked`) or whose object a store in the loop may reach
  (two unpromised slices may be one buffer). Replication is a heuristic, not a measurement: a body
  whose only work is the replicated step (`o[i] = lut[q[i]]`, `o[i] = x[i] as i64` from `f64`)
  stays scalar, and every replicated gather or scatter lane is still a scalar load or store: no AVX2
  `vgatherdps` is ever emitted. None of these forms has been timed.
- **An aliased call to a recognized kernel means what the source loop means.** Every multi-buffer
  recognizer emit site — GEMM in all its forms, GEMV/GEVM, transpose, pooling, the column/row/arg
  reductions, the losses, the norm backwards, RoPE, the scans, embedding/scatter, the fused norms,
  broadcast bias, the int dequants, the half-precision maps and the 256-bit elementwise kernels —
  asks one classifier (`wukong_mir_build::alias::plan_kernel_alias`) and, for an operand pair the
  compiler cannot separate, versions the dispatch on a run-time overlap test whose other arm is the
  ORIGINAL source loop or window (`kernel_guard.rs`, LLVM's `LoopVersioning` shape). A library
  routine `fn mm(a, b, mut c)` called both as `mm(x, y, x)` and `mm(x, y, z)` keeps its kernel for
  the disjoint call and runs the nest for the aliased one (`tests/run/alias_versioned_kernels.wk`
  prints every lane of both). Exact in-place operation is admitted only where the runtime entry
  documents it, and each exported kernel's `# Safety` states its aliasing class. A recognized
  reduction over an empty range keeps the seed exactly. The interpreter refuses a cross-element call
  that reaches it with overlapping operands, and answers a shifted elementwise overlap with the
  source loop's value, instead of computing either from copies.
  **Open:** `alias.rs`'s documented limits (its `# Known limits`) can still hand a kernel overlapping
  storage; an outlined `@parallel` chunk has no versioning context, so an alias-uncertain kernel there
  declines instead of versioning; the generic vectorizer's loop fusion and the `@parallel` region
  outliner keep their own disjointness proofs outside this classifier.
- Array *length* in a type may be an integer literal or a top-level `const` (resolved through
  const-to-const chains; `tests/run/const_array_length.wk`). It may also be **arithmetic over those** —
  `[i32; 2 + 2]`, or a `const N: i32 = 2 + 2` used as a length — because sema's `eval_usize` and
  mir_build's `const_usize_expr` fold through the one shared `wukong_ast::BinOp::fold_const_len`, so
  the allocated slot and the compile-time bounds check cannot disagree (`a[9]` on such a length-4 array
  is still `E0501`). Radix prefixes, `_` separators and integer suffixes all decode (`[i32; 0x10]` is
  16). The ceiling is `MAX_CONST_ARRAY_LEN = u32::MAX`: a longer literal length is `E0401`, and the
  shared folder declines whenever an operand or result exceeds it or arithmetic would wrap.
  Unfoldable lengths report `E0406`; unknown names report `E0301`. Invalid lengths use an error
  type, separate from a genuine zero-length array. Calls, returns, assignments and nested aggregate
  agreement check zero extents too, while preserving compatible inferred empty literals.
  A **symbolic** array length (a generic `N`, or `2*N`) is the shape of a **parameter**: `fn f<N>(a:
  [f32; N])` takes the shape-generic buffer `Tensor[f32, N]`, bound per call from the argument's
  length (`tests/run/generic_array_param.wk`). A local or a return type of generic length (which would
  need run-time-sized storage) and a non-scalar element still report `E0406`. Matmul *dimensions* may be
  runtime values — a runtime-dimension matmul still dispatches to the GEMM kernel.
- Ordered comparison (`< <= > >=`) is not defined on `bool`, so a **chained comparison** `a < b < c`
  (which parses left-associatively as `(a < b) < c`) is a compile error (`E0433`) rather than a silent
  wrong result — write `a < b && b < c` (`tests/fail/chained_comparison.wk`). Equality `==`/`!=`
  between a `bool` and a non-bool scalar is likewise rejected, so the `==`/`!=` chain `5 == 3 == 0` is
  caught too (`tests/fail/chained_equality.wk`).
- **Mixed-type arithmetic** promotes only losslessly and order-independently (same-signedness
  widening, unsigned to a strictly wider signed type, `f16`/`bf16` to `f32` to `f64`, an integer to a
  float that holds it exactly, a never-negative `for` counter or `const` to an unsigned type); every
  other mix is `E0440`, and `/`/`%` divide with the common type's signedness, so `u32 / i64` is a
  signed divide (`tests/run/mixed_width_div_signedness.wk`). Not yet: `min`/`max`/`fmax`-style
  intrinsics still pick their operands' common type by width (`wukong_sema::shape`), and `if`/`match`
  arms of different scalar types still merge to the left arm's type on a width tie.
- **Compound assignment** follows the plain assignment's rule (`E0441`: `f32 += f64`, `i32 += i64`
  no longer narrow the operand silently), and **literal inference** types an unannotated
  literal-initialized `let` from its first constraining use (`let mut s = 0.0; s += x_f64[i]`
  accumulates in `f64`); its first constraining use — an assignment, a `return`, a call argument
  (a method-call or associated-function argument too), a struct-literal field, an operator —
  decides it, as in Rust (`tests/run/literal_infer_struct_field.wk`,
  `tests/run/literal_infer_method_arg.wk`). Not yet: a turbofish does not reach a literal argument
  (`g::<f64>(2.5)` is `E0401`), and a `let x;` with neither an annotation nor an initializer is not
  inferred from its first assignment — lowering refuses it (`C0001`) rather than give it an `i32`
  slot.
- Tensor shape checking is **sound for both concrete and generic shapes**. A function's own generic
  dimension variables are **rigid** inside its body: when two shapes that both carry the function's
  generics are checked — its declared-vs-returned shape, an elementwise operator's operands, an
  assignment, or two `if`/`match` value arms — the dims must match by *identity* (`N` matches only
  `N`); they are never bound to each other or to a constant. So a generic function can no longer **lie
  about its output shape**: `fn f<M, N>(a: Tensor[f32, M, N]) -> Tensor[f32, N, 5]` is rejected
  (`E0502`, `tests/fail/generic_return_shape_lie.wk`). Previously the body-check bound `N := M` and
  silently accepted the constant `5` against the generic `N`, and a turbofish (`f::<2, 2>`) then made
  the lie concrete and over-strided — turning a type-valid index into an out-of-bounds read the
  interpreter trapped on but native did not. Call-site unification is a *different* context and still
  **infers** a callee's dims from the argument shapes (there the callee's generics are inference
  variables, not rigid — `matmul::<…>(a, b, c)` binds `M, N, K` from the arguments as before). A
  related hole is also closed: a rank-1 tensor parameter binds its symbolic dim from a decaying
  array's length, so `f<N>(a: Tensor[f32, N], b: Tensor[f32, N])` rejects arrays of different lengths
  (`tests/fail/generic_tensor_arg_length_mismatch.wk`). The former undeclared-dim **lenience** is now
  a diagnostic: an **undeclared** dim name in a tensor type (not a declared generic, integer, `?`, or
  `const`) is rejected with **E0504** and a did-you-mean hint, so a typo like `Tensor[f32, KK]` for `K`
  no longer silently introduces a fresh implicit dim and drops the shared constraint
  (`tests/fail/generic_shape_unknown_dim.wk`). Three more parts of the callee contract are checked. A
  tensor's **layout** is part of unification: passing a `.col_major` tensor to a `Contiguous`-typed
  parameter is `E0502` (`tensor layout mismatch`), and a fixed-size array's row-major decay is checked
  the same way — previously the value was silently reinterpreted row-major one call away from the
  `C0001` the same access gets in place. A **slice `[]T`, tuple, or declared struct/enum value is not a
  tensor base**: against a `Tensor` parameter it is `E0501`, where it used to fall through the lenient
  arm and hand the callee the address of the slice's 16-byte fat-pointer header (both backends agreed
  on the garbage, so the differential gate could not see it). And the `E0504` unknown-dimension check
  now applies to **struct field types, enum payload types and `extern` block signatures** as well as
  function signatures, so `struct S { t: Tensor[f32, ZZ] }` reports rather than inventing a dim.
- A `for i in 0..n` loop **re-reads its upper bound `n` live each iteration** (it lowers to a C-style
  `while (i < n)`), not Rust-style range capture: mutating `n` inside the body changes the remaining
  iteration count. Defensible for a low-level kernel language, but worth knowing. A descending range
  with a negative step runs zero iterations (the condition stays `i < hi`).
- **Slice accesses are bounds-checked at runtime** (default on; `@unchecked` on a function opts it
  out). `s[i]` is compared against the length in the slice's fat pointer, and an out-of-range access
  stops the program with `error: index out of bounds: the len is N but the index is I at
  file:line:col`, exit 1, identically on the interpreter, the JIT and a linked executable
  (`tests/run/slice_bounds_*.wk`). `-O2` removes the checks from innermost counted loops by
  versioning them on a preheader guard (`wukong_opt::bce`); a loop with a call, a `break`, a
  non-affine index or a header written inside it keeps them. Still unchecked: a **runtime** index into a fixed-size array
  or a constant-shape tensor (the static length is known, so this is the same machinery with a
  constant bound — not yet wired), and the extents a recognized kernel call reads. A
  **compile-time-constant** index past the end is still caught at compile time (`E0501`) — for both a
  fixed-size array (`a[5]` on a `[T; 4]`) and a static tensor dimension (`a[5, 0]` on a
  `Tensor[f32, 2, 2]`). On an unchecked out-of-bounds access the backends differ: the interpreter (the oracle) *traps* on an index it can detect as out of range (a
  debugging aid — its memory is a bounded slot vector), while the native backend reads/writes past the
  buffer, classic UB like C (and an optimization level can even change the garbage observed). An
  out-of-bounds program is therefore outside the defined contract — the differential gate's bit-for-bit
  `interp == native` and `-O0 == -O3` invariants hold only for well-defined programs.
- `mem2reg` promotes scalar integer, float **and pointer** slots. A pointer slot read before it is
  written on some path gets the null pointer (`inttoptr 0`) there, exactly what the interpreter's
  zero-initialized slot holds now that it reserves address 0 as a never-allocated null
  (`wukong_interp::NULL_SLOT`); before that reservation only definitely-initialized pointer slots
  could be promoted, and a pointer local initialized behind a branch kept its slot. At `-O2`, `sroa`
  first splits an array, struct or tuple local whose every access is at a constant position into
  scalar leaf slots, which mem2reg then promotes. Arrays indexed by a computed index, vectors,
  address-taken locals of any type, and any slot accessed at a width other than its own stay in
  memory (the interpreter and `cse`/`dse` handle those directly); a loop-invariant cell of one that a
  loop both reads and writes is still carried in a register by `promote` when nothing else in the
  loop may alias it.
- A **`[]T` slice parameter still re-loads its data pointer at every element access** unless it is
  `@noalias` or the loop only reads. A slice is a 16-byte `{ data, len }` fat pointer passed *by
  address*, so the base comes from `load ptr (gep <param>, 0)` — a load out of caller-owned memory,
  not out of a local slot — and `mem2reg` has nothing to promote. For a `@noalias` slice whose fat
  pointer the function never writes or publishes, `licm` hoists that load (the header is a live
  16-byte object for the whole call, and no store in the loop can reach it) and Cranelift marks it
  `readonly` + `can_move`. For an unpromised slice `licm` hoists the load only when it runs on every
  iteration (behind a first-iteration guard) and no store in the loop may alias it, so a read-only
  slice loop reads its base once; a loop that writes through any other loaded pointer still reloads,
  because another parameter may alias its header.
- **String literals live in a read-only static-data section** (`.rodata`), referenced by address via
  `Op::GlobalAddr` and deduped by content (one blob per unique literal). So **returning or threading a
  `*u8`** that points at a literal created inside a callee is valid — the pointer outlives the frame,
  and the interpreter and native backend print it identically (`tests/run/string_return.wk`). (This
  closed a real interp-vs-native divergence: a literal used to live in the caller's frame, so a
  returned `*u8` dangled on native — freed stack, an empty line — while the interpreter's persistent
  memory masked it.)
- **`--emit=exe` links the `wukong_runtime` kernels**, so a program that dispatches a recognized
  kernel (`for i in 0..N { y[i] = exp(x[i]); }`, a `matmul`, a reduction) compiles to a standalone
  executable whose output matches `--run` (`crates/wukongc/tests/exe.rs`). The Cranelift object is
  linked by a **rustc-driven link** (rustc invokes the object's native platform linker and links
  `wukong_runtime` as a dependency; a generated shim supplies the `wukong_rt_*` runtime), which also
  links the string `.rodata` relocations — neither of which the MinGW `cc` path on this host can do.
  It falls back to the C-runtime `cc` link (scalar, no-data, no-kernel programs) when rustc or the
  runtime rlib is unavailable, and the exe gate skips cleanly when no toolchain can link. The generated
  link inputs (`{stem}_shim.rs` for the rustc path, `{stem}_rt.c` for the `cc` fallback) are written
  into a **pid-keyed temporary scratch directory that is removed when the link finishes**, not into the
  working directory — they used to overwrite and then delete any user file of the same name, and two
  concurrent links on one stem raced on them; the working directory is used only if the temp directory
  cannot be created. The intermediate `{stem}.o` is still written beside the invocation. The rustc link
  passes `-C panic=abort` because the workspace release profile sets `panic = "abort"`, and a strategy
  mismatch is a hard metadata error that rejects the link outright. **The executable's entry
  matches `--run`**: the object's C `main` is synthesized, calls the program's `main` (now the
  internal `_WK4main`) on a 512 MiB worker thread through `wukong_rt_run_main` (in both the Rust
  shim and the C runtime), converts its result as the JIT does and keeps the low 8 bits, so a
  `-> f32`, a void `main` and `return 300` exit as under `--run`, and 30000-deep recursion no longer
  depends on the OS main-thread stack. In an `--emit=exe` object every user function is an internal,
  mangled `_WK<len><name>` symbol, so a `.wk` `write`/`fmod`/`memcpy` cannot collide with or
  interpose on the C library; an `--emit=obj` object keeps its other functions exported under their
  source names (its C interface). **Platforms** (README *Platform support*): Tier 1 is x86-64
  Windows/Linux with AVX2 + FMA, fully gated; Tier 2 is aarch64 macOS/Linux (built and tested in CI,
  non-blocking, scalar runtime) and aarch64 Windows (`cargo check` only). `WUKONG_MAX_ISA`
  (`wukong_runtime::isa`) caps the runtime's ISA dispatch, and a Tier 1 lane runs the corpus with
  `WUKONG_MAX_ISA=scalar`. The cap also narrows the Cranelift ISA flags the JIT and a `native`
  object compile with (`scalar` is the x86-64 baseline), so with `--target-cpu=native` (the
  default) `--emit=obj` is capped as well, while an explicit `--target-cpu` level ignores the cap.
  On the scalar lane no code Wukong generates or dispatches to uses AVX; the C library still picks
  its own paths (including the `fmaf`/`floorf` libcalls Cranelift makes below FMA/SSE4.1). The
  differential oracle remains `--run`.
- **A `mut` aggregate parameter aliases the caller's value — now opt-in.** Aggregate arguments
  (`struct`/array/tuple/tensor) are passed by pointer, the zero-copy tensor-kernel convention, so a
  callee that mutates one writes back into the caller's storage (copying every large buffer by value
  would be the performance footgun). This is no longer a *silent* surprise: mutating a parameter — a
  rebind (`p = …`) or an aggregate projection (`p.f = …`, `p[i] = …`) — now **requires `mut` on the
  parameter**, else it is a compile error (E0304). A `mut` aggregate parameter is thus the idiomatic
  in-place output buffer (`fn relu(mut out: […], x: […])`) and its caller-visible mutation is
  explicit; a non-`mut` aggregate parameter is read-only. Both backends agree bit-for-bit. Take a
  `let q = p;` for an independent copy inside the callee (a plain `let` *does* copy); a pointer
  parameter still writes through its pointee (`*p = …`) without `mut`.
